#!/usr/bin/env python3
"""e1_stage_profile.py — E1: 50 步 baseline vs 4 步蒸馏的全链路 per-stage profile

报告第一张图。论点：**蒸馏不只是步数变少，它重排了瓶颈结构。**

两个配置（**两个独立 checkpoint，不是同一权重 ± adapter**）：

  50step : 原始 Wan2.1-T2V-1.3B 官方权重 + UniPC + 外部 CFG(5.0)
           -> 每步 2 次前向（cond + uncond），50 步 = **100 次前向**
  4step  : lightx2v Wan2.1-T2V-1.3B-Distill-Models 完整 BF16 权重
           denoising_step_list=[1000,750,500,250], shift=5, 无外部 CFG
           -> 每步 1 次前向，4 步 = **4 次前向**

→ 步数轴的朴素倍数是 **25×**（100/4），不是 10×。

用法：
    export PY=<PY>
    $PY benchmarks/e1_stage_profile.py --device 0 --out results/E1/stage_profile.json

显存约束（24 GB 单卡）决定了本脚本的结构：T5 / DiT / VAE **分阶段加载并释放**，
不同时驻留。这也是单卡上唯一可行的跑法，报告里按此口径写。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch

SF_ROOT = os.environ.get("SF_ROOT", "")   # 上游 Wan 推理工作目录（含 wan/、utils/、prompts/）
assert SF_ROOT, "请先 export SF_ROOT=<上游工作目录>"
DISTILL_CKPT = os.environ.get("DISTILL_CKPT", "")   # lightx2v 4 步蒸馏权重 (rev ef72050)
assert DISTILL_CKPT, "请先 export DISTILL_CKPT=<distill_native.pt 的路径>"

LATENT_SHAPE = (21, 16, 60, 104)

# ---- 口径常量（报告里要逐条写出来的就是这些） -----------------------------
CFG_50STEP = dict(steps=50, solver="unipc", shift=5.0, guide_scale=5.0,
                  forwards_per_step=2, total_forwards=100,
                  weights="official Wan2.1-T2V-1.3B (WanModel.from_pretrained)")
CFG_4STEP = dict(steps=4, solver="step-distill", shift=5.0, guide_scale=None,
                 denoising_step_list=[1000, 750, 500, 250],
                 forwards_per_step=1, total_forwards=4,
                 weights=f"lightx2v distilled, {DISTILL_CKPT}")

PROMPT = "Aerial drone shot flying over a dense green forest"
SEED = 242

# ---- op 分类规则（数据驱动：原始 per-op 表也一并存进 json，分类可审计） ----
BUCKETS = [
    ("attention", ("flash_attention", "_flash_", "flash_fwd", "scaled_dot_product",
                   "_efficient_attention", "cudnn_attention", "mem_efficient",
                   "fmha", "sdpa")),
    ("gemm",      ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm",
                   "aten::matmul", "aten::linear", "gemm", "cutlass", "cublas")),
    ("norm",      ("layer_norm", "native_layer_norm", "rms_norm", "group_norm",
                   "aten::var_mean", "aten::rsqrt", "aten::pow", "aten::mean")),
    ("rope",      ("view_as_complex", "view_as_real", "aten::polar", "aten::angle",
                   "complex")),
    ("gelu",      ("gelu", "aten::silu", "aten::sigmoid", "aten::tanh")),
    ("elementwise", ("aten::mul", "aten::add", "aten::div", "aten::sub", "aten::copy_",
                     "aten::to", "aten::clamp", "aten::neg",
                     "aten::_to_copy", "aten::contiguous", "aten::clone", "memcpy")),
    ("reshape",   ("aten::cat", "aten::view", "aten::reshape", "aten::permute",
                   "aten::transpose", "aten::flatten", "aten::unsqueeze",
                   "aten::squeeze", "aten::expand", "aten::slice", "aten::select",
                   "aten::stack", "aten::split", "aten::chunk", "aten::index")),
    ("conv",      ("aten::conv", "convolution", "cudnn_convolution")),
]


def classify(name: str) -> str:
    low = name.lower()
    for bucket, pats in BUCKETS:
        for p in pats:
            if p.lower() in low:
                return bucket
    return "other"


def dev_time(evt):
    """torch 2.11 把 self_cuda_time_total 改名为 self_device_time_total。"""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(evt, attr, None)
        if v is not None:
            return v
    return 0.0


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def sync_time(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return r, time.perf_counter() - t0


# =========================== stage 1: text encode ===========================

def stage_text_encode(dev, out_dir):
    """T5 UMT5-XXL。

    ⚠️ 参考实现把 T5 构造成 **float32**（`utils/wan_wrapper.py:21`），
    而 checkpoint 本身是 bf16（`models_t5_umt5-xxl-enc-bf16.pth`, 11.36 GB）。
    5.68 B 参数 × 4 B = 22.7 GB = 21.1 GiB，对 22.2 GiB 可用显存**只剩 ~1.1 GiB
    给激活**。所以本脚本**实测**三件事而不是假设：
      1. fp32 权重能否上卡（t5_fp32_fits_in_24gb）
      2. fp32 能否真正跑完一次 encode（t5_fp32_forward_ok）—— 权重装得下不等于跑得动
      3. bf16 的耗时与显存（实际采用的那档）
    """
    sys.path.insert(0, SF_ROOT)
    from utils.wan_wrapper import WanTextEncoder
    from wan.configs.shared_config import wan_shared_cfg

    res = {"t5_fp32_fits_in_24gb": None, "t5_fp32_err": None}

    enc = WanTextEncoder().eval().requires_grad_(False)          # CPU, fp32
    n_param = sum(p.numel() for p in enc.parameters())
    res["t5_params"] = n_param
    res["t5_fp32_bytes"] = n_param * 4
    res["t5_bf16_bytes"] = n_param * 2

    # 如实试一次 fp32：先上卡，再真跑一次 forward（权重装得下 != 跑得动）
    res["t5_fp32_forward_ok"] = None
    try:
        enc = enc.to(dev)
        torch.cuda.synchronize()
        res["t5_fp32_fits_in_24gb"] = True
        res["t5_fp32_weights_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        try:
            with torch.no_grad():
                _ = enc(text_prompts=[PROMPT])
            torch.cuda.synchronize()
            res["t5_fp32_forward_ok"] = True
            res["t5_fp32_forward_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        except Exception as e2:
            res["t5_fp32_forward_ok"] = False
            res["t5_fp32_forward_err"] = f"{type(e2).__name__}: {str(e2)[:200]}"
        enc = enc.cpu()
        free()
    except Exception as e:
        res["t5_fp32_fits_in_24gb"] = False
        res["t5_fp32_err"] = f"{type(e).__name__}: {str(e)[:200]}"
        enc = enc.cpu()
        free()

    enc = enc.to(torch.bfloat16).to(dev)
    neg = wan_shared_cfg.sample_neg_prompt

    with torch.no_grad():
        # warmup（tokenizer 首次调用有一次性开销，不该计进 stage 时间）
        _ = enc(text_prompts=[PROMPT])
        torch.cuda.synchronize()
        (cond, uncond), t = sync_time(
            lambda: (enc(text_prompts=[PROMPT]), enc(text_prompts=[neg])))

    res["ms_cond_plus_uncond"] = t * 1e3
    with torch.no_grad():
        _, t1 = sync_time(lambda: enc(text_prompts=[PROMPT]))
    res["ms_single_prompt"] = t1 * 1e3
    res["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
    res["dtype_used"] = "bfloat16"
    res["embed_shape"] = tuple(cond["prompt_embeds"].shape)

    torch.save({"cond": cond["prompt_embeds"].to(torch.bfloat16).cpu(),
                "uncond": uncond["prompt_embeds"].to(torch.bfloat16).cpu()},
               os.path.join(out_dir, "prompt_embeds.pt"))
    del enc, cond, uncond
    free()
    return res


# =========================== stage 2: denoise ===============================

def build_dit(dev, distilled: bool):
    sys.path.insert(0, SF_ROOT)
    from utils.wan_wrapper import WanDiffusionWrapper
    w = WanDiffusionWrapper(is_causal=False)
    info = {}
    if distilled:
        sd = torch.load(DISTILL_CKPT, map_location="cpu", weights_only=False)
        miss, unexp = w.model.load_state_dict(sd, strict=False)
        info = {"missing": len(miss), "unexpected": len(unexp),
                "missing_examples": list(miss)[:5], "unexpected_examples": list(unexp)[:5]}
        del sd
    w = w.to(dev).to(torch.bfloat16).eval()
    w.model.requires_grad_(False)
    return w, info


def make_sched_50(dev, steps, shift):
    sys.path.insert(0, SF_ROOT)
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1,
                                      use_dynamic_shifting=False)
    sch.set_timesteps(steps, device=dev, shift=shift)
    return sch


def make_sched_4(dsl, shift=5.0, NT=1000):
    """WanStepDistillScheduler 的数学，抄自 <SF_ROOT>/scripts/nvfp4/phase1_distill4.py"""
    s = torch.linspace(1.0, 0.0, NT + 1)[:-1]
    sig_full = shift * s / (1 + (shift - 1) * s)
    ts_full = sig_full * NT
    idx = [NT - x for x in dsl]
    return sig_full[idx].tolist(), ts_full[idx].tolist()


def denoise_50(w, sch, cond, uncond, lat, guide_scale, dev, n_steps_limit=None):
    F = lat.shape[1]
    ts_list = sch.timesteps
    for i, t in enumerate(ts_list):
        if n_steps_limit is not None and i >= n_steps_limit:
            break
        ts = t.to(dev).float().view(1, 1).expand(1, F)
        mi = lat.to(torch.bfloat16)
        v_c, _ = w(mi, cond, ts)
        v_u, _ = w(mi, uncond, ts)
        v = v_u.float() + guide_scale * (v_c.float() - v_u.float())
        lat = sch.step(v, t, lat, return_dict=False)[0].float()
    return lat


def denoise_4(w, sigmas, timesteps, cond, lat, dev):
    F = lat.shape[1]
    for i in range(len(sigmas)):
        ts = torch.tensor(timesteps[i], device=dev).float().view(1, 1).expand(1, F)
        v, _ = w(lat.to(torch.bfloat16), cond, ts)
        v = v.float()
        x0 = lat - sigmas[i] * v
        lat = x0 + sigmas[i + 1] * v if i < len(sigmas) - 1 else x0
    return lat


def profile_forwards(fn, n_forwards):
    """对 fn 做一次 torch profiler 采样，按 op 桶统计 self device time。

    ⚠️ **踩过的坑**：`prof.key_averages()` 同时返回 aten 层 op（device_type=CPU，
    device time 由其子 kernel 汇总而来）和 CUDA kernel 条目（device_type=CUDA）。
    两层都累加会**恰好双计一遍** —— 第一版就是这么错的，总时长正好是 wall clock
    的 2 倍。这里分开统计两个视图：

      op_view     : 只取 aten 层（device_type=CPU），self_device_time_total
      kernel_view : 只取 CUDA kernel 条目

    两个视图应该各自 ≈ wall clock。它们互为交叉验证：
    对不上说明有 device 时间没被任何 aten op 认领（异步 memcpy 之类）。
    """
    from torch.profiler import profile, ProfilerActivity
    from torch.autograd import DeviceType

    fn()  # warmup，别把 autotune/首次 kernel 编译计进去
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        fn()
        torch.cuda.synchronize()

    views = {}
    for view, want_cpu in (("op_view", True), ("kernel_view", False)):
        per_op, buckets = {}, {}
        for e in prof.key_averages():
            is_cpu = getattr(e, "device_type", DeviceType.CPU) == DeviceType.CPU
            if is_cpu != want_cpu:
                continue
            us = dev_time(e)
            if us <= 0:
                continue
            per_op[e.key] = per_op.get(e.key, 0.0) + us
            b = classify(e.key)
            buckets[b] = buckets.get(b, 0.0) + us
        tot = sum(buckets.values())
        views[view] = {
            "total_device_ms": tot / 1e3,
            "buckets_ms": {k: v / 1e3 for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])},
            "buckets_share": {k: v / tot for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])} if tot else {},
            "top_ops_ms": {k[:110]: v / 1e3 for k, v in sorted(per_op.items(), key=lambda kv: -kv[1])[:25]},
        }
    out = {"n_forwards_profiled": n_forwards}
    out.update(views["op_view"])          # 主视图放顶层，向后兼容
    out["kernel_view"] = views["kernel_view"]
    out["view_agreement"] = (views["kernel_view"]["total_device_ms"]
                             / max(views["op_view"]["total_device_ms"], 1e-9))
    return out


def stage_denoise(dev, out_dir, distilled, embeds, prof_steps):
    cfg = dict(CFG_4STEP if distilled else CFG_50STEP)
    w, load_info = build_dit(dev, distilled)
    cfg["ckpt_load"] = load_info
    free()

    g = torch.Generator(device=dev).manual_seed(SEED)
    noise = torch.randn(1, *LATENT_SHAPE, device=dev, dtype=torch.float32, generator=g)
    cond = {"prompt_embeds": embeds["cond"].to(dev).to(torch.bfloat16)}
    uncond = {"prompt_embeds": embeds["uncond"].to(dev).to(torch.bfloat16)}

    if distilled:
        sig, tsl = make_sched_4(cfg["denoising_step_list"], cfg["shift"])
        cfg["sigmas"] = [round(x, 5) for x in sig]
        cfg["timesteps"] = [round(x, 2) for x in tsl]
        run_full = lambda: denoise_4(w, sig, tsl, cond, noise.clone(), dev)
        run_prof = run_full                      # 4 步全测
        n_fw_prof = 4
    else:
        sch = make_sched_50(dev, cfg["steps"], cfg["shift"])
        cfg["timesteps_head"] = [round(float(x), 2) for x in sch.timesteps[:5]]
        run_full = lambda: denoise_50(w, make_sched_50(dev, cfg["steps"], cfg["shift"]),
                                      cond, uncond, noise.clone(), cfg["guide_scale"], dev)
        run_prof = lambda: denoise_50(w, make_sched_50(dev, cfg["steps"], cfg["shift"]),
                                      cond, uncond, noise.clone(), cfg["guide_scale"],
                                      dev, n_steps_limit=prof_steps)
        n_fw_prof = prof_steps * 2

    # 先 warmup 一次（DiT 首次前向有 cudnn/sdpa autotune）
    with torch.no_grad():
        if distilled:
            _ = run_full()
        else:
            _ = denoise_50(w, make_sched_50(dev, cfg["steps"], cfg["shift"]),
                           cond, uncond, noise.clone(), cfg["guide_scale"], dev,
                           n_steps_limit=1)
    free()

    with torch.no_grad():
        lat, t_full = sync_time(run_full)
    peak = torch.cuda.max_memory_allocated() / 2**30

    with torch.no_grad():
        prof = profile_forwards(lambda: run_prof(), n_fw_prof)

    res = {
        "config": cfg,
        "denoise_wall_s": t_full,
        "ms_per_forward": t_full * 1e3 / cfg["total_forwards"],
        "peak_alloc_gib": peak,
        "profile": prof,
        "profile_scope": ("all 4 steps" if distilled
                          else f"first {prof_steps} of {cfg['steps']} steps "
                               f"({n_fw_prof} of {cfg['total_forwards']} forwards)"),
        "latent_stats": {"mean": float(lat.mean()), "std": float(lat.std()),
                         "min": float(lat.min()), "max": float(lat.max()),
                         "finite": bool(torch.isfinite(lat).all())},
    }
    tag = "4step" if distilled else "50step"
    torch.save(lat.cpu(), os.path.join(out_dir, f"latent_{tag}.pt"))
    del w, cond, uncond, noise, lat
    free()
    return res


# =========================== stage 3: VAE decode ============================

VAE_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
VAE_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
           3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


def stage_vae(dev, out_dir, latents):
    """在**真实模型输出的 latent** 上量 decode 时间 + BF16/FP32 保真度。

    T1 的 PSNR 56.8 dB 来自随机 latent；T3_HANDOFF §1.2 要求在真实 latent 上复核。
    """
    sys.path.insert(0, SF_ROOT)
    from wan.modules.vae import _video_vae
    ck = os.path.join(SF_ROOT, "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")

    out = {}
    for tag, lat in latents.items():
        # wrapper 约定 [1,F,C,H,W] -> VAE 要 [1,C,F,H,W]
        z = lat.permute(0, 2, 1, 3, 4).contiguous()
        pix = {}
        for dt in (torch.float32, torch.bfloat16):
            m = _video_vae(pretrained_path=ck, z_dim=16).eval().requires_grad_(False)
            m = m.to(device=dev, dtype=dt)
            scale = [torch.tensor(VAE_MEAN, device=dev, dtype=dt),
                     1.0 / torch.tensor(VAE_STD, device=dev, dtype=dt)]
            zz = z.to(device=dev, dtype=dt)
            with torch.no_grad():
                _ = m.decode(zz, scale)            # warmup
                torch.cuda.synchronize()
                free()
                o, t = sync_time(lambda: m.decode(zz, scale))
                o = o.float().clamp_(-1, 1)
            key = str(dt).replace("torch.", "")
            out[f"{tag}/{key}"] = {
                "decode_s": t,
                "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                "out_shape": tuple(o.shape),
                "has_nan": bool(torch.isnan(o).any()),
            }
            pix[key] = o.cpu()
            del m, o, zz
            free()
        a, b = pix["float32"], pix["bfloat16"]
        d = (a - b).abs()
        mse = float(((a - b) ** 2).mean())
        # max|Δ| 由极少数像素决定，单看它会把结论带偏；给分位数谱和越界像素比例。
        qs = [0.5, 0.9, 0.99, 0.999, 0.9999]
        flat = d.flatten().float()[::11].contiguous()      # 降采样，quantile 有元素上限
        dq = torch.quantile(flat, torch.tensor(qs))
        out[f"{tag}/fidelity_bf16_vs_fp32"] = {
            "latent_source": f"real model output ({tag})",
            "mse": mse,
            "psnr_db": float(10 * torch.log10(torch.tensor(4.0 / mse))) if mse > 0 else None,
            "mean_abs_diff": float(d.mean()),
            "max_abs_diff": float(d.max()),
            "mean_abs_diff_8bit": float(d.mean()) * 127.5,
            "max_abs_diff_8bit": float(d.max()) * 127.5,
            "abs_diff_8bit_quantiles": {f"p{q*100:g}": float(v) * 127.5
                                        for q, v in zip(qs, dq)},
            "frac_pixels_over_2_255": float((d > 2 / 127.5).float().mean()),
            "frac_pixels_over_8_255": float((d > 8 / 127.5).float().mean()),
            "frac_pixels_over_16_255": float((d > 16 / 127.5).float().mean()),
        }
        # 存视频供肉眼抽查
        try:
            from wan.utils.utils import cache_video
            for key, o in pix.items():
                cache_video(o, save_file=os.path.join(out_dir, f"vae_{tag}_{key}.mp4"),
                            fps=16, normalize=True, value_range=(-1, 1))
        except Exception as e:
            out[f"{tag}/video_dump_err"] = f"{type(e).__name__}: {e}"
        del pix
        free()
    return out


# =============================== driver =====================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="results/E1/stage_profile.json")
    ap.add_argument("--prof-steps", type=int, default=2,
                    help="50 步配置下用前 N 步做 profiler 采样（100 次前向全采会产生巨大 trace）")
    ap.add_argument("--skip-50step", action="store_true")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(args.device)
    dev = f"cuda:{args.device}"
    p = torch.cuda.get_device_properties(args.device)
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(SF_ROOT)   # wan 的相对路径（wan_models/...）依赖 cwd
    out_dir = os.path.abspath(os.path.join(
        REPO, out_dir))
    out_path = os.path.join(out_dir, os.path.basename(args.out))

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": p.name, "sm": f"{p.major}{p.minor}",
        "total_mem_gib": p.total_memory / 2**30,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "prompt": PROMPT, "seed": SEED,
        "pixel_geometry": "832x480, 81 frames, 16 fps",
        "seq_len": 32760,
        "attention_backend": "torch SDPA fallback (flash_attn package NOT installed in this env)",
        "structure_note": "T5 / DiT / VAE are loaded and freed sequentially; they never "
                          "co-reside. This is the only feasible layout on a single 24 GB "
                          "card and is the accounting used in the report.",
    }

    print(f"device: {p.name}  torch {torch.__version__}\n")

    print("[stage 1] text encode (T5 UMT5-XXL)")
    payload["text_encode"] = stage_text_encode(dev, out_dir)
    r = payload["text_encode"]
    print(f"  fp32 weights fit: {r['t5_fp32_fits_in_24gb']}  "
          f"fp32 forward ok: {r['t5_fp32_forward_ok']}")
    print(f"  bf16 encode: {r['ms_single_prompt']:.0f} ms/prompt, "
          f"{r['ms_cond_plus_uncond']:.0f} ms for cond+uncond, peak {r['peak_alloc_gib']:.2f} GiB")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    embeds = torch.load(os.path.join(out_dir, "prompt_embeds.pt"), weights_only=False)

    print("\n[stage 2a] denoise 4-step (distilled)")
    payload["denoise_4step"] = stage_denoise(dev, out_dir, True, embeds, args.prof_steps)
    d = payload["denoise_4step"]
    print(f"  wall {d['denoise_wall_s']:.2f} s  ({d['ms_per_forward']:.0f} ms/forward)  "
          f"peak {d['peak_alloc_gib']:.2f} GiB")
    print(f"  buckets: " + "  ".join(f"{k}={v*100:.1f}%" for k, v in
                                     list(d["profile"]["buckets_share"].items())[:6]))
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    if not args.skip_50step:
        print("\n[stage 2b] denoise 50-step + CFG (official weights)")
        payload["denoise_50step"] = stage_denoise(dev, out_dir, False, embeds, args.prof_steps)
        d = payload["denoise_50step"]
        print(f"  wall {d['denoise_wall_s']:.2f} s  ({d['ms_per_forward']:.0f} ms/forward)  "
              f"peak {d['peak_alloc_gib']:.2f} GiB")
        print(f"  buckets: " + "  ".join(f"{k}={v*100:.1f}%" for k, v in
                                         list(d["profile"]["buckets_share"].items())[:6]))
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

    print("\n[stage 3] VAE decode on REAL latents")
    lats = {}
    for tag in ("4step", "50step"):
        fp = os.path.join(out_dir, f"latent_{tag}.pt")
        if os.path.exists(fp):
            lats[tag] = torch.load(fp, weights_only=False)
    payload["vae"] = stage_vae(dev, out_dir, lats)
    for k, v in payload["vae"].items():
        if "fidelity" in k:
            print(f"  {k}: PSNR {v['psnr_db']:.2f} dB  "
                  f"mean|d| {v['mean_abs_diff_8bit']:.2f}/255  max|d| {v['max_abs_diff_8bit']:.2f}/255")
            print(f"      quantiles(8bit) {v['abs_diff_8bit_quantiles']}  "
                  f">2/255: {v['frac_pixels_over_2_255']*100:.3f}%  "
                  f">8/255: {v['frac_pixels_over_8_255']*100:.4f}%")
        elif "decode_s" in v:
            print(f"  {k}: {v['decode_s']:.2f} s  peak {v['peak_alloc_gib']:.2f} GiB")

    # ---- 端到端合成 + VAE 占比实测 -------------------------------------
    e2e = {}
    t_text = payload["text_encode"]["ms_cond_plus_uncond"] / 1e3
    t_text_1 = payload["text_encode"]["ms_single_prompt"] / 1e3
    for tag, key in (("4step", "denoise_4step"), ("50step", "denoise_50step")):
        if key not in payload:
            continue
        t_den = payload[key]["denoise_wall_s"]
        # 4 步无 CFG 只需 cond；50 步需 cond+uncond
        t_txt = t_text_1 if tag == "4step" else t_text
        for vdt in ("float32", "bfloat16"):
            vk = f"{tag}/{vdt}"
            if vk not in payload["vae"]:
                continue
            t_vae = payload["vae"][vk]["decode_s"]
            tot = t_txt + t_den + t_vae
            e2e[f"{tag}/vae_{vdt}"] = {
                "text_s": t_txt, "denoise_s": t_den, "vae_s": t_vae, "total_s": tot,
                "share": {"text": t_txt / tot, "denoise": t_den / tot, "vae": t_vae / tot},
            }
    payload["e2e"] = e2e

    # 蒸馏轴的实测倍数（同一 VAE dtype 下比较，VAE 是固定成本）
    if "denoise_50step" in payload and "denoise_4step" in payload:
        d50 = payload["denoise_50step"]["denoise_wall_s"]
        d4 = payload["denoise_4step"]["denoise_wall_s"]
        payload["distill_axis"] = {
            "naive_forward_ratio": 100 / 4,
            "denoise_only_speedup": d50 / d4,
            "e2e_speedup_vae_fp32": (e2e["50step/vae_float32"]["total_s"]
                                     / e2e["4step/vae_float32"]["total_s"])
            if "50step/vae_float32" in e2e and "4step/vae_float32" in e2e else None,
            "e2e_speedup_vae_bf16": (e2e["50step/vae_bfloat16"]["total_s"]
                                     / e2e["4step/vae_bfloat16"]["total_s"])
            if "50step/vae_bfloat16" in e2e and "4step/vae_bfloat16" in e2e else None,
            "note": "denoise-only speedup should approach 25x (forward-count ratio); "
                    "the e2e numbers show how much Amdahl (VAE + text) eats.",
        }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*70}")
    for k, v in e2e.items():
        print(f"{k:22s} total {v['total_s']:7.2f} s   "
              f"text {v['share']['text']*100:4.1f}%  "
              f"denoise {v['share']['denoise']*100:5.1f}%  "
              f"VAE {v['share']['vae']*100:5.1f}%")
    if "distill_axis" in payload:
        da = payload["distill_axis"]
        print(f"\ndistill axis: naive {da['naive_forward_ratio']:.0f}x  "
              f"denoise-only {da['denoise_only_speedup']:.2f}x  "
              f"e2e(VAE fp32) {da['e2e_speedup_vae_fp32']:.2f}x  "
              f"e2e(VAE bf16) {da['e2e_speedup_vae_bf16']:.2f}x")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
