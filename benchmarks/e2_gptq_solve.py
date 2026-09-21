#!/usr/bin/env python3
"""e2_gptq_solve.py — E2: FP8 W8A8 的 GPTQ 求解 + clip 复验消融

对应 `SPEC.md` §4.2 和 §4.8.1。本脚本只做**权重求解**和**代理损失层面**的结论，
不生成视频（那在 `e2_quality.py`）。

三件事：
  1. on-policy Hessian 收集：H_l = E[fq(x_l) fq(x_l)^T]，x_l 是 layer l 在真实
     4 步 rollout 里看到的输入，且已过部署用的激活伪量化 —— GEMM 看到什么，
     权重就该对着什么优化。协议抄自 `an earlier in-house implementation`。
  2. 逐层求解 rtn / gptq / gptq_cd / gptq_clip_cd 四档
  3. **clip 复验**：报 recovery 与 γ 分布，与前人在 Kolors SR/DiT 上的 220 层表
     对照。`SPEC.md` §4.8.1 要的就是这个在 Wan 上的版本。

用法：
    export PY=<PY>
    export PYTHONPATH=<SF_ROOT>
    $PY benchmarks/e2_gptq_solve.py --device 0 --calib-prompts 16 \\
        --out results/E2/quant_quality.json
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

sys.path.insert(0, REPO)
sys.path.insert(0, SF_ROOT)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def sched_4(dsl=DSL, shift=SHIFT, NT=1000):
    s = torch.linspace(1.0, 0.0, NT + 1)[:-1]
    sig = shift * s / (1 + (shift - 1) * s)
    idx = [NT - x for x in dsl]
    return sig[idx].tolist(), (sig * NT)[idx].tolist()


def rollout_4(w, cond, noise, sigmas, timesteps, dev, state=None):
    lat = noise.to(device=dev, dtype=torch.float32)
    F = lat.shape[1]
    for i in range(len(sigmas)):
        if state is not None:
            state["step"] = i
        ts = torch.tensor(timesteps[i], device=dev).float().view(1, 1).expand(1, F)
        v, _ = w(lat.to(torch.bfloat16), cond, ts)
        v = v.float()
        x0 = lat - sigmas[i] * v
        lat = x0 + sigmas[i + 1] * v if i < len(sigmas) - 1 else x0
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--calib-prompts", type=int, default=16)
    ap.add_argument("--rows", type=int, default=8192,
                    help="每次 hook 从 32760 行里子采样多少行进 H（抄前人 8192）")
    ap.add_argument("--budget-gb", type=float, default=6.0)
    ap.add_argument("--wfmt", default="fp8_e4m3:-1:fp32",
                    help="per-tensor weight scale：sm_89 上 _scaled_mm 唯一原生支持的粒度")
    ap.add_argument("--afmt", default="fp8_e4m3:tensor")
    ap.add_argument("--rollout", default="rtn", choices=["rtn", "bf16"])
    ap.add_argument("--methods", default="rtn,gptq,gptq_cd,gptq_clip_cd")
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--hdir", default="results/E2/hessians")
    ap.add_argument("--wdir", default="results/E2/solved")
    ap.add_argument("--out", default="results/E2/quant_quality.json")
    ap.add_argument("--reuse-h", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(args.device)
    dev = f"cuda:{args.device}"
    p = torch.cuda.get_device_properties(args.device)

    hdir = os.path.join(REPO, args.hdir)
    wdir = os.path.join(REPO, args.wdir)
    out_path = os.path.join(REPO, args.out)
    for d in (hdir, wdir, os.path.dirname(out_path)):
        os.makedirs(d, exist_ok=True)
    os.chdir(SF_ROOT)   # wan 的 wan_models/... 相对路径依赖 cwd

    from quant.gptq import formats, solvers
    from quant.fp8 import wan_fp8
    from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder

    wfmt = formats.parse_wfmt(args.wfmt)
    afmt = formats.parse_afmt(args.afmt)
    sigmas, timesteps = sched_4()

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": p.name, "sm": f"{p.major}{p.minor}", "torch": torch.__version__,
        "wfmt": args.wfmt, "afmt": args.afmt, "rollout": args.rollout,
        "calib": {"prompts": args.calib_prompts, "rows_per_hook": args.rows,
                  "steps": len(DSL), "denoising_step_list": DSL, "shift": SHIFT,
                  "prompt_file": PROMPT_FILE, "seed_base": 1234},
        "seq_len": 32760,
    }

    # ---------------- 校准 prompt 的 text embedding ----------------------
    emb_path = os.path.join(hdir, f"calib_embeds_{args.calib_prompts}.pt")
    if not os.path.exists(emb_path):
        prompts = [l.strip() for l in open(PROMPT_FILE) if l.strip()][:args.calib_prompts]
        print(f"[calib] encoding {len(prompts)} prompts", flush=True)
        enc = WanTextEncoder().eval().requires_grad_(False).to(torch.bfloat16).to(dev)
        embs = []
        for pr in prompts:
            embs.append(enc(text_prompts=[pr])["prompt_embeds"].to(torch.bfloat16).cpu())
        torch.save({"prompts": prompts, "embeds": embs}, emb_path)
        del enc
        free()
    calib = torch.load(emb_path, weights_only=False)
    payload["calib"]["prompts_used"] = calib["prompts"]
    print(f"[calib] {len(calib['embeds'])} embeds ready", flush=True)

    # ---------------- 建模型 + 换 FakeQuantLinear ------------------------
    w = WanDiffusionWrapper(is_causal=False)
    sd = torch.load(DISTILL_CKPT, map_location="cpu", weights_only=False)
    miss, unexp = w.model.load_state_dict(sd, strict=False)
    assert len(miss) == 0 and len(unexp) == 0, (len(miss), len(unexp))
    del sd
    w = w.to(dev).to(torch.bfloat16).eval()
    w.model.requires_grad_(False)

    names = wan_fp8.convert_fakequant(w.model, afmt=afmt)
    print(f"[conv] {len(names)} linears -> FakeQuantLinear (afmt={args.afmt})", flush=True)
    payload["n_quantized_linears"] = len(names)

    # 原始 bf16 权重先存一份，solve 要从它出发
    W0 = {n: w.model.get_submodule(n).weight.data.clone() for n in names}

    # rollout 权重：RTN 量化后的（on-policy）
    if args.rollout == "rtn":
        t0 = time.time()
        for n in names:
            m = w.model.get_submodule(n)
            m.weight.data.copy_(wfmt.rtn(W0[n].float()).to(m.weight.dtype))
        print(f"[H] rollout weights = RTN({wfmt.name}) in {time.time()-t0:.0f}s", flush=True)

    # ---------------- H 收集（分组，每组一次校准 pass） -------------------
    groups = wan_fp8.plan_groups(w.model, names, args.budget_gb)
    payload["h_groups"] = [len(g) for g in groups]
    print(f"[H] {len(names)} layers -> {len(groups)} passes "
          f"(sizes {[len(g) for g in groups]})", flush=True)

    have_all = all(os.path.exists(os.path.join(hdir, n + ".pt")) for n in names)
    if args.reuse_h and have_all:
        print("[H] reusing existing hessians", flush=True)
    else:
        gen = torch.Generator(device="cpu").manual_seed(1234)
        counts = {}
        for gi, group in enumerate(groups):
            acc = {n: torch.zeros(w.model.get_submodule(n).in_features,
                                  w.model.get_submodule(n).in_features,
                                  device=dev, dtype=torch.float32) for n in group}
            num = {n: 0 for n in group}

            def mk(n):
                def hook(xq):
                    x2 = xq.reshape(-1, xq.shape[-1])
                    if x2.shape[0] > args.rows:
                        idx = torch.randint(0, x2.shape[0], (args.rows,), generator=gen)
                        x2 = x2[idx.to(x2.device)]
                    xf = x2.float()
                    acc[n].addmm_(xf.T, xf)
                    num[n] += x2.shape[0]
                return hook

            for n in names:
                w.model.get_submodule(n).collect = mk(n) if n in acc else None

            t0 = time.time()
            for ri, emb in enumerate(calib["embeds"]):
                cond = {"prompt_embeds": emb.to(dev).to(torch.bfloat16)}
                g = torch.Generator(device=dev).manual_seed(1000 + ri)
                noise = torch.randn(1, *LATENT_SHAPE, device=dev,
                                    dtype=torch.float32, generator=g)
                rollout_4(w, cond, noise, sigmas, timesteps, dev)
                del cond, noise
                if ri % 4 == 0:
                    print(f"[H] pass {gi+1}/{len(groups)} case {ri+1}/"
                          f"{len(calib['embeds'])} ({time.time()-t0:.0f}s)", flush=True)
            for n in group:
                torch.save((acc[n] / max(num[n], 1)).cpu(),
                           os.path.join(hdir, n + ".pt"))
                counts[n] = num[n]
            del acc
            free()
            print(f"[H] pass {gi+1}/{len(groups)} done in {time.time()-t0:.0f}s", flush=True)
        for n in names:
            w.model.get_submodule(n).collect = None
        payload["calib"]["rows_accumulated"] = counts
        # 每维样本数：H 是否良态的唯一相关量
        payload["calib"]["samples_per_dim"] = {
            n: counts[n] / w.model.get_submodule(n).in_features
            for n in list(counts)[:3]}

    # DiT 权重恢复成 bf16 原值，solve 从原始权重出发
    for n in names:
        w.model.get_submodule(n).weight.data.copy_(W0[n])
    free()

    # ---------------- 逐层求解 ------------------------------------------
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    results = {}
    for method in methods:
        t0 = time.time()
        infos, packed = {}, {}
        for i, n in enumerate(names):
            H = torch.load(os.path.join(hdir, n + ".pt"), weights_only=False).to(dev)
            Wl = W0[n].float().to(dev)
            Q, info = solvers.solve_layer(Wl, H, wfmt, method=method,
                                          passes=args.passes)
            infos[n] = info
            packed[n] = Q.to(torch.bfloat16).cpu()
            del H, Wl, Q
            if i % 60 == 0:
                print(f"[solve {method}] {i+1}/{len(names)} "
                      f"({time.time()-t0:.0f}s) rec={info['recovery']*100:.1f}% "
                      f"gamma={info['gamma']}", flush=True)
                free()
        torch.save({"weights": packed, "infos": infos, "method": method,
                    "wfmt": args.wfmt},
                   os.path.join(wdir, f"w8_{method}.pt"))
        rec = [v["recovery"] for v in infos.values()]
        gam = {}
        for v in infos.values():
            gam[str(v["gamma"])] = gam.get(str(v["gamma"]), 0) + 1
        nfb = sum(1 for v in infos.values() if v["fallback"])
        results[method] = {
            "recovery_mean": sum(rec) / len(rec),
            "recovery_min": min(rec), "recovery_max": max(rec),
            "gamma_hist": gam,
            "n_fallback_to_rtn": nfb,
            "n_layers": len(names),
            "solve_s": time.time() - t0,
            "per_layer": {n: {k: (float(v[k]) if isinstance(v[k], (int, float)) else v[k])
                              for k in ("recovery", "gamma", "e_rtn", "e", "fallback")}
                          for n, v in infos.items()},
        }
        print(f"[solve {method}] DONE rec_mean={results[method]['recovery_mean']*100:.2f}% "
              f"gamma={gam} fallback={nfb}/{len(names)} "
              f"({results[method]['solve_s']:.0f}s)", flush=True)
        del packed, infos
        free()

    payload["solve"] = results

    # ---------------- clip 复验结论 -------------------------------------
    if "gptq_cd" in results and "gptq_clip_cd" in results:
        a, b = results["gptq_cd"], results["gptq_clip_cd"]
        nz = sum(v for k, v in b["gamma_hist"].items() if float(k) != 1.0)
        payload["clip_verdict"] = {
            "recovery_cd_only": a["recovery_mean"],
            "recovery_clip_cd": b["recovery_mean"],
            "delta": b["recovery_mean"] - a["recovery_mean"],
            "n_layers_picking_gamma_lt_1": nz,
            "n_layers": b["n_layers"],
            "frac_layers_picking_gamma_lt_1": nz / b["n_layers"],
            "prior_work_reference": {
                "source": "一份更早的实验记录 (Kolors SR/DiT, 220 layers)",
                "w8_gptq": 0.761, "w8_gptqcd": 0.803, "w8_gptqclipcd": 0.803,
                "gamma_hist_clipcd": {"1.0": 219, "0.94": 1},
                "e2e_lpips_cd": 0.0245, "e2e_lpips_clipcd": 0.0247,
                "int4_control": {"no_clip": 0.578, "with_clip": 0.629},
            },
            "note": "proxy-loss level only; end-to-end video quality is in e2_quality.py",
        }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwritten: {out_path}")
    if "clip_verdict" in payload:
        cv = payload["clip_verdict"]
        print(f"CLIP VERDICT: cd-only {cv['recovery_cd_only']*100:.2f}% vs "
              f"clip+cd {cv['recovery_clip_cd']*100:.2f}%  "
              f"(delta {cv['delta']*100:+.2f} pp); "
              f"{cv['n_layers_picking_gamma_lt_1']}/{cv['n_layers']} layers picked gamma<1")


if __name__ == "__main__":
    main()
