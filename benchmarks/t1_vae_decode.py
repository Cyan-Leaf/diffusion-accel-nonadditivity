#!/usr/bin/env python3
"""t1_vae_decode.py — T1: Wan2.1 VAE decode 单次耗时与显存峰值

为什么 T1 就要量这个数（`research/MODELS.md` §VAE decode 成本）：
VAE decode 是**固定成本**，与 denoising 步数无关。50 步时它被摊薄到看不见，
4 步时占比可能暴涨。若 4 步下 VAE 占到 20%+，则「attention 是瓶颈」这个前提
在蒸馏模型上已不成立 —— 这直接是 E5 不可加性的第一个成因（H1 Amdahl）。

本脚本只量 VAE，不跑 DiT，不跑 T5。E1 会把三段拼起来。

用法:
    $SFPY benchmarks/t1_vae_decode.py --out results/T1/vae_decode.json
    ($SFPY = <PY>，
     那个环境里有 easydict/einops 等 wan 依赖)

注意 dtype：`utils/wan_wrapper.py:decode_to_pixel` 被 `inference_wan.py:181`
以 **float32** latent 调用，而 VAE 权重本身也是 fp32 加载的。
所以「as-shipped」这一档是 FP32，不是 BF16。MODELS.md 的表头写的是 BF16，
两档都量，报告里按实际链路用 FP32 那档。
"""
import argparse
import json
import os
import sys
import time

import torch

SF_ROOT = os.environ.get("SF_ROOT", "")   # 上游 Wan 推理工作目录（含 wan/、utils/、prompts/）
assert SF_ROOT, "请先 export SF_ROOT=<上游工作目录>"
VAE_CKPT = os.path.join(SF_ROOT, "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")

# (F_latent, C, H_latent, W_latent) —— wrapper 约定的 latent 布局
# 来源 <SF_ROOT>/inference_wan.py:33
LATENT_SHAPE = (21, 16, 60, 104)

VAE_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
VAE_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
           3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


def load_vae(dev, dtype):
    sys.path.insert(0, SF_ROOT)
    from wan.modules.vae import _video_vae
    m = _video_vae(pretrained_path=VAE_CKPT, z_dim=16).eval().requires_grad_(False)
    return m.to(device=dev, dtype=dtype)


def bench_decode(m, dev, dtype, reps, use_cache=False):
    f, c, h, w = LATENT_SHAPE
    z = torch.randn(1, c, f, h, w, device=dev, dtype=dtype)
    scale = [torch.tensor(VAE_MEAN, device=dev, dtype=dtype),
             1.0 / torch.tensor(VAE_STD, device=dev, dtype=dtype)]
    fn = m.cached_decode if use_cache else m.decode

    with torch.no_grad():
        out = fn(z, scale)          # warmup + 拿输出形状
        out_shape = tuple(out.shape)
        del out
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(z, scale)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
            del out
        peak = torch.cuda.max_memory_allocated(dev)
    ts.sort()
    return {
        "dtype": str(dtype), "use_cache": use_cache,
        "reps": reps,
        "ms_median": ts[len(ts) // 2], "ms_min": ts[0], "ms_max": ts[-1],
        "peak_alloc_bytes": peak, "peak_alloc_gib": peak / 2**30,
        "out_shape": out_shape,
    }


def fidelity_bf16_vs_fp32(dev, seed=1234):
    """BF16 decode 相对 FP32 decode 的数值差。

    为什么要量：FP32 那档是链路 as-shipped 的形态，但 FP32 decode 是 4 步流水线里
    最大的单项成本。若 BF16 数值上无损，这是一笔免费的 2×。
    ⚠️ 这里用的是**随机 latent**，不是真实模型输出。真实 latent 在分布内，
    行为可能不同（大概率更好）。E1 要在真实 latent 上复核。
    """
    f, c, h, w = LATENT_SHAPE
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(1, c, f, h, w, generator=g).to(dev)
    outs = {}
    for dt in (torch.float32, torch.bfloat16):
        m = load_vae(dev, dt)
        scale = [torch.tensor(VAE_MEAN, device=dev, dtype=dt),
                 1.0 / torch.tensor(VAE_STD, device=dev, dtype=dt)]
        with torch.no_grad():
            o = m.decode(z.to(dt), scale).float().clamp_(-1, 1)
        outs[str(dt)] = o.cpu()
        del m, o
        torch.cuda.empty_cache()
    a, b = outs["torch.float32"], outs["torch.bfloat16"]
    mse = float(((a - b) ** 2).mean())
    # 输出域是 [-1,1]，峰峰值 2 -> PSNR = 10*log10(2^2 / mse)
    psnr = float(10 * torch.log10(torch.tensor(4.0 / mse))) if mse > 0 else float("inf")
    return {
        "seed": seed, "latent": "random normal (NOT real model output)",
        "mean_abs_diff": float((a - b).abs().mean()),
        "max_abs_diff": float((a - b).abs().max()),
        "mse": mse, "psnr_db": psnr,
        "fp32_has_nan": bool(torch.isnan(a).any()),
        "bf16_has_nan": bool(torch.isnan(b).any()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="results/T1/vae_decode.json")
    ap.add_argument("--skip-fidelity", action="store_true")
    args = ap.parse_args()

    dev = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)
    p = torch.cuda.get_device_properties(args.device)
    print(f"device: {p.name} sm_{p.major}{p.minor}  torch {torch.__version__}")
    print(f"latent: {LATENT_SHAPE} (F,C,H,W) -> 832x480x81 pixels\n")

    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        m = load_vae(dev, dtype)
        wbytes = sum(t.numel() * t.element_size() for t in m.parameters())
        for use_cache in (False, True):
            try:
                r = bench_decode(m, dev, dtype, args.reps, use_cache)
                r["weight_bytes"] = wbytes
                r["ok"] = True
            except Exception as e:
                r = {"dtype": str(dtype), "use_cache": use_cache, "ok": False,
                     "err": f"{type(e).__name__}: {e}"}
            rows.append(r)
            tag = f"{str(dtype).replace('torch.',''):9s} cache={str(use_cache):5s}"
            if r["ok"]:
                print(f"  {tag}  {r['ms_median']:9.1f} ms  "
                      f"peak_alloc {r['peak_alloc_gib']:.2f} GiB  out {r['out_shape']}")
            else:
                print(f"  {tag}  FAILED: {r['err'][:80]}")
        del m
        torch.cuda.empty_cache()

    fid = None
    if not args.skip_fidelity:
        fid = fidelity_bf16_vs_fp32(dev)
        print(f"\nbf16 vs fp32 decode: PSNR {fid['psnr_db']:.2f} dB  "
              f"mean|d| {fid['mean_abs_diff']:.5f}  max|d| {fid['max_abs_diff']:.5f}  "
              f"nan {fid['fp32_has_nan']}/{fid['bf16_has_nan']}")

    # 拿 E0 的 DiT GEMM 成本给 VAE 做个粗占比参照（只算 GEMM，不含 attention/norm，
    # 所以这是 DiT 侧的**下界**，VAE 占比是**上界**）。
    ref = None
    e0 = "results/E0/gemm_bench.json"
    if os.path.exists(e0):
        with open(e0) as f:
            d = json.load(f)
        per_step_ms = 0.0
        for r in d["rows"]:
            c = r.get("per_step_calls") or 0
            if c and r["name"] != "self_qkv_fused" and r["bf16"].get("ok"):
                per_step_ms += r["bf16"]["ms"] * c
        ref = {"dit_gemm_only_bf16_ms_per_step_lb": per_step_ms,
               "note": "GEMM only (no attention/norm/rope), so this is a lower bound "
                       "on DiT per-step cost; the derived VAE share is an upper bound."}
        ok = [r for r in rows if r.get("ok") and r["dtype"] == "torch.float32"
              and not r["use_cache"]]
        if ok and per_step_ms > 0:
            vae_ms = ok[0]["ms_median"]
            for steps in (4, 50):
                ref[f"vae_share_ub_at_{steps}steps"] = vae_ms / (vae_ms + steps * per_step_ms)

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": p.name, "sm": f"{p.major}{p.minor}",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "vae_ckpt": VAE_CKPT,
        "latent_shape_FCHW": LATENT_SHAPE,
        "pixel_geometry": "832x480, 81 frames, 16 fps",
        "rows": rows,
        "bf16_vs_fp32_fidelity": fid,
        "dit_reference": ref,
        "caveat": "decode() is a per-latent-frame causal loop with a feature cache "
                  "(wan/modules/vae.py:545-569); it is NOT spatially tiled. Peak memory "
                  "reported is torch allocator peak during decode only, VAE weights included.",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    if ref:
        print(f"\nDiT GEMM-only BF16: {ref['dit_gemm_only_bf16_ms_per_step_lb']:.1f} ms/step (lower bound)")
        for k in ("vae_share_ub_at_4steps", "vae_share_ub_at_50steps"):
            if k in ref:
                print(f"  {k}: {ref[k]*100:.1f}%")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
