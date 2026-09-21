#!/usr/bin/env python3
"""e2_quality.py — E2: FP8 W8A8 的端到端质量（含 clip 消融与激活粒度消融）

`e2_gptq_solve.py` 给的是**代理损失**层面的结论；本脚本给**像素层面**的结论。

为什么需要后者：代理损失 `min (W-Q) H (W-Q)^T` 是逐层的、线性化的，
它不知道 30 层 × 4 步的误差如何累积。前人在 Kolors SR 上就出现过
「代理损失改善但端到端不改善」（clip 那条），所以两个层面都要报。

臂（arm）：
  bf16                 参照
  rtn_tensor           per-tensor 权重 RTN + per-tensor 激活 —— 部署形态的下界
  rtn_token            per-tensor 权重 RTN + per-token   激活 —— 看激活粒度值多少
  gptq_cd / gptq_clip_cd / gptq   solve 出来的权重，激活粒度与 --afmt 一致

指标（对 bf16 arm）：latent 相对 L2、像素 PSNR、LPIPS、高频能量比。
不做多 seed 噪声地板分析（SPEC 已排除）。

用法：
    $PY benchmarks/e2_quality.py --device 0 --prompts 3 --out results/E2/quality_pixel.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch

REPO = os.environ.get("GENMODEL_ACCEL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SF_ROOT = os.environ.get("SF_ROOT", "")   # 上游 Wan 推理工作目录（含 wan/、utils/、prompts/）
assert SF_ROOT, "请先 export SF_ROOT=<上游工作目录>"
DISTILL_CKPT = os.environ.get("DISTILL_CKPT", "")   # lightx2v 4 步蒸馏权重 (rev ef72050)
assert DISTILL_CKPT, "请先 export DISTILL_CKPT=<distill_native.pt 的路径>"
PROMPT_FILE = os.path.join(SF_ROOT, "prompts/MovieGenVideoBench.txt")
LATENT_SHAPE = (21, 16, 60, 104)
DSL = [1000, 750, 500, 250]
SHIFT = 5.0

# 校准用了 MovieGenVideoBench 的前 N 条，评测必须用**没见过的**，从尾部取
EVAL_FROM_TAIL = True

sys.path.insert(0, REPO)
sys.path.insert(0, SF_ROOT)


def _ladder():
    """单例加载 `benchmarks/e5_ladder.py`。

    ⚠️ 必须走 `sys.modules` 缓存：用 importlib 加载两次会得到**两个独立的模块对象**，
    于是 `SPARSE_STATE` 有两份，run4 写的 step 计数patched forward 看不到，
    dense warmup 就会静默失效（**症状是数字看起来正常**）。
    """
    import importlib.util as iu
    if "_e5_ladder_singleton" in sys.modules:
        return sys.modules["_e5_ladder_singleton"]
    sp = iu.spec_from_file_location("_e5_ladder_singleton",
                                    os.path.join(REPO, "benchmarks/e5_ladder.py"))
    m = iu.module_from_spec(sp)
    sys.modules["_e5_ladder_singleton"] = m
    sp.loader.exec_module(m)
    return m


def free():
    gc.collect()
    torch.cuda.empty_cache()


def sched_4(dsl=DSL, shift=SHIFT, NT=1000):
    s = torch.linspace(1.0, 0.0, NT + 1)[:-1]
    sig = shift * s / (1 + (shift - 1) * s)
    idx = [NT - x for x in dsl]
    return sig[idx].tolist(), (sig * NT)[idx].tolist()


def run4(w, cond, noise, sigmas, timesteps, dev):
    _st = _ladder().SPARSE_STATE
    lat = noise.to(device=dev, dtype=torch.float32)
    F = lat.shape[1]
    for i in range(len(sigmas)):
        _st["step"] = i
        ts = torch.tensor(timesteps[i], device=dev).float().view(1, 1).expand(1, F)
        v, _ = w(lat.to(torch.bfloat16), cond, ts)
        v = v.float()
        x0 = lat - sigmas[i] * v
        lat = x0 + sigmas[i + 1] * v if i < len(sigmas) - 1 else x0
    return lat


def hf_ratio(a, b):
    """高频能量比：a 相对 b 在空间高频带的能量倍数。>1 = 多出结构化噪声。

    简化版 `<SF_ROOT>/scripts/nvfp4/freq_diag.py` 的 HFhi：
    取亮度，做 2D FFT，统计半径 > 0.5 Nyquist 的能量占比之比。
    """
    def hf(x):  # x [T,H,W] float
        F = torch.fft.rfft2(x)
        P = (F.real ** 2 + F.imag ** 2)
        H, W = P.shape[-2], P.shape[-1]
        fy = torch.fft.fftfreq(x.shape[-2], device=x.device).abs().view(-1, 1)
        fx = torch.fft.rfftfreq(x.shape[-1], device=x.device).abs().view(1, -1)
        r = (fy ** 2 + fx ** 2).sqrt()
        m = r > 0.25          # 0.25 cycles/px = 半个 Nyquist
        return float(P[..., m].sum() / P.sum().clamp_min(1e-30))
    return hf(a) / max(hf(b), 1e-30)


def luma(v):
    # v [1,3,T,H,W] in [-1,1] -> [T,H,W] in [0,1]
    x = (v[0].permute(1, 0, 2, 3).float() + 1) / 2
    return 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--afmt", default="fp8_e4m3:tensor")
    ap.add_argument("--wfmt", default="fp8_e4m3:-1:fp32")
    ap.add_argument("--wdir", default="results/E2/solved")
    ap.add_argument("--arms", default="bf16,rtn_tensor,rtn_token,gptq_cd,gptq_clip_cd")
    ap.add_argument("--out", default="results/E2/quality_pixel.json")
    ap.add_argument("--save-video", action="store_true")
    # ⚠️ 2026-09-20 事故：视频目录原先硬编码为 results/E2/videos，跑 NVFP4 臂时
    #    arm 名同为 `gptq_cd`，**静默覆盖了 FP8 的同名视频**。测量没受影响
    #    （VBench 与抽帧用的是 results/E9/vbench_work 的副本），但那是运气。
    #    → 目录改成可指定，且默认值按 wfmt 分开。
    ap.add_argument("--vdir", default=None,
                    help="视频输出目录；默认按 wfmt 分目录，避免不同格式的同名 arm 互相覆盖")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(args.device)
    dev = f"cuda:{args.device}"
    p = torch.cuda.get_device_properties(args.device)
    out_path = os.path.join(REPO, args.out)
    wdir = os.path.join(REPO, args.wdir)
    if args.vdir:
        vdir = os.path.join(REPO, args.vdir)
    else:
        # 默认按 wfmt 分目录：fp8_e4m3:-1:fp32 -> results/E2/videos（历史路径），其余单独放
        vdir = os.path.join(REPO, "results/E2/videos"
                            if args.wfmt == "fp8_e4m3:-1:fp32"
                            else f"results/videos_{args.wfmt.replace(':', '_')}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(vdir, exist_ok=True)
    os.chdir(SF_ROOT)

    from quant.gptq import formats
    from quant.fp8 import wan_fp8
    from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
    from wan.modules.vae import _video_vae

    wfmt = formats.parse_wfmt(args.wfmt)
    sigmas, timesteps = sched_4()

    # ---- eval prompt（与校准集不相交：从文件尾部取） --------------------
    allp = [l.strip() for l in open(PROMPT_FILE) if l.strip()]
    prompts = (allp[-args.prompts:] if EVAL_FROM_TAIL else allp[:args.prompts])
    embf = os.path.join(REPO, "results/E2/eval_embeds.pt")
    if not os.path.exists(embf):
        enc = WanTextEncoder().eval().requires_grad_(False).to(torch.bfloat16).to(dev)
        embs = [enc(text_prompts=[pr])["prompt_embeds"].to(torch.bfloat16).cpu()
                for pr in prompts]
        torch.save({"prompts": prompts, "embeds": embs}, embf)
        del enc
        free()
    ev = torch.load(embf, weights_only=False)
    print(f"[eval] {len(ev['prompts'])} held-out prompts (from tail of "
          f"{os.path.basename(PROMPT_FILE)}, {len(allp)} total)", flush=True)

    # ---- 逐 arm 生成 latent ---------------------------------------------
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    lats = {}
    arm_meta = {}
    for arm in arms:
        w = WanDiffusionWrapper(is_causal=False)
        sd = torch.load(DISTILL_CKPT, map_location="cpu", weights_only=False)
        miss, unexp = w.model.load_state_dict(sd, strict=False)
        assert not miss and not unexp
        del sd
        w = w.to(dev).to(torch.bfloat16).eval()
        w.model.requires_grad_(False)

        meta = {"arm": arm}
        if arm in ("bf16_sparse", "fp8_sparse", "bf16_sparse_warm1"):
            # SVG mask + sample_mse 分类 + FlexAttention（见 sparse/wan_svg.py 的血统说明）
            _m = _ladder()
            from sparse import wan_svg as _S
            dm = _S.build_dense_masks(device=dev, max_row=10000)
            if arm == "fp8_sparse":
                from fusion import wan_fused
                meta["fp8_linears"] = len(wan_fused.convert_fused(
                    w.model, share_quant=True, fuse_bias=True, fuse_epilogue=True))
            # SVG 的 first_times_fp=0.2 在 4 步下 floor(0.2*4)=0，故默认零预热；
            # `_warm1` 臂给 1 步 dense（4 步下的最小非零预热 = 25% 的步数预算）
            nfp = 1 if arm.endswith("_warm1") else 0
            meta["n_fp_steps"] = nfp
            meta["sparse_layers"] = _m.patch_self_attn_sparse(
                w.model, dmasks=dm, n_fp_steps=nfp)
            _m.SPARSE_STATE["step"] = 0
            meta["note"] = ("SVG mask + sample_mse online classification + FlexAttention; "
                            "reference arm is `bf16` (dense eager), same as the quant arms")
        elif arm != "bf16":
            afmt_spec = args.afmt
            if arm == "rtn_token":
                afmt_spec = "fp8_e4m3:token"
            afmt = formats.parse_afmt(afmt_spec)
            names = wan_fp8.convert_fakequant(w.model, afmt=afmt)
            meta["afmt"] = afmt_spec
            meta["n_linears"] = len(names)
            if arm.startswith("rtn"):
                for n in names:
                    m = w.model.get_submodule(n)
                    m.weight.data.copy_(
                        wfmt.rtn(m.weight.data.float()).to(m.weight.dtype))
                meta["weights"] = f"RTN({wfmt.name})"
            else:
                fp = os.path.join(wdir, f"w8_{arm}.pt")
                if not os.path.exists(fp):
                    print(f"[{arm}] SKIP: {fp} not found", flush=True)
                    del w
                    free()
                    continue
                pack = torch.load(fp, map_location="cpu", weights_only=False)
                n = wan_fp8.load_dequant_weights(w.model, pack["weights"])
                rec = [v["recovery"] for v in pack["infos"].values()]
                meta["weights"] = f"solved {arm}, {n} layers"
                meta["recovery_mean"] = sum(rec) / len(rec)
                del pack
        free()

        outs = []
        t0 = time.time()
        for i, emb in enumerate(ev["embeds"]):
            cond = {"prompt_embeds": emb.to(dev).to(torch.bfloat16)}
            g = torch.Generator(device=dev).manual_seed(7000 + i)
            noise = torch.randn(1, *LATENT_SHAPE, device=dev,
                                dtype=torch.float32, generator=g)
            outs.append(run4(w, cond, noise, sigmas, timesteps, dev).cpu())
            del cond, noise
        lats[arm] = outs
        arm_meta[arm] = meta
        print(f"[{arm}] {len(outs)} latents in {time.time()-t0:.0f}s  {meta}", flush=True)
        del w
        free()

    # ---- VAE decode + 指标 ----------------------------------------------
    ck = os.path.join(SF_ROOT, "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
           3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
    dt = torch.bfloat16      # 解码用 BF16（E1 已判定可用），省一半时间
    vae = _video_vae(pretrained_path=ck, z_dim=16).eval().requires_grad_(False)
    vae = vae.to(device=dev, dtype=dt)
    scale = [torch.tensor(MEAN, device=dev, dtype=dt),
             1.0 / torch.tensor(STD, device=dev, dtype=dt)]

    pix = {}
    for arm, ls in lats.items():
        vs = []
        for lat in ls:
            z = lat.permute(0, 2, 1, 3, 4).contiguous().to(device=dev, dtype=dt)
            vs.append(vae.decode(z, scale).float().clamp_(-1, 1).cpu())
            del z
            free()
        pix[arm] = vs
        print(f"[vae] {arm} decoded {len(vs)}", flush=True)
    del vae
    free()

    try:
        import lpips
        lp = lpips.LPIPS(net="alex").to(dev).eval()
    except Exception as e:
        lp = None
        print(f"[lpips] unavailable: {e}", flush=True)

    ref_lat, ref_pix = lats.get("bf16"), pix.get("bf16")
    metrics = {}
    for arm in lats:
        if arm == "bf16":
            continue
        per = []
        for i in range(len(pix[arm])):
            a, b = pix[arm][i], ref_pix[i]
            mse = float(((a - b) ** 2).mean())
            psnr = float(10 * torch.log10(torch.tensor(4.0 / mse))) if mse > 0 else None
            dl = (lats[arm][i] - ref_lat[i])
            rel = float(dl.norm() / ref_lat[i].norm())
            r = {"latent_rel_l2": rel, "pixel_psnr_db": psnr,
                 "hf_energy_ratio": hf_ratio(luma(a).to(dev), luma(b).to(dev))}
            if lp is not None:
                # 每 10 帧抽一帧，省显存
                fa = a[0].permute(1, 0, 2, 3)[::10].to(dev)
                fb = b[0].permute(1, 0, 2, 3)[::10].to(dev)
                r["lpips"] = float(lp(fa, fb).mean())
                del fa, fb
                free()
            per.append(r)
        agg = {k: sum(x[k] for x in per) / len(per) for k in per[0]}
        metrics[arm] = {"mean": agg, "per_prompt": per, "meta": arm_meta[arm]}
        print(f"[metric] {arm:16s} " + "  ".join(f"{k}={v:.4f}" for k, v in agg.items()),
              flush=True)

    if args.save_video:
        from wan.utils.utils import cache_video
        for arm, vs in pix.items():
            for i, v in enumerate(vs):
                cache_video(v, save_file=os.path.join(vdir, f"{arm}_p{i}.mp4"),
                            fps=16, normalize=True, value_range=(-1, 1))

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": p.name, "torch": torch.__version__,
        "wfmt": args.wfmt, "afmt_default": args.afmt,
        "eval_prompts": ev["prompts"],
        "eval_disjoint_from_calib": "eval prompts taken from the TAIL of "
                                    "MovieGenVideoBench.txt; calibration used the HEAD",
        "vae_dtype_for_eval": "bfloat16",
        "metrics_vs_bf16": metrics,
        "note": "No multi-seed noise-floor analysis (excluded by SPEC). "
                "hf_energy_ratio > 1 means the arm injects excess high-frequency energy "
                "relative to bf16.",
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
