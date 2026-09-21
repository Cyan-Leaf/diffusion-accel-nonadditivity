#!/usr/bin/env python3
"""e3_fusion_latency.py — E3 P0：融合前后的四点折线

`SPEC.md` §4.3 要的那张图。四个点全部实测，全部真 `torch._scaled_mm`：

    朴素 0.836×  →  compiled quant 1.009×  →  **融合后 ? ×**  →  Amdahl 上界 1.105×

消融是增量的，每一格只开一个开关，这样每一段差额都有明确归属：

    bf16                      基线
    fp8_naive                 eager quant（T3 已测，复跑确认）
    fp8_compiled              torch.compile 的 quant（T3 已测）
    +F1 bias_epilogue         bias 走 _scaled_mm 原生 epilogue
    +F2 share_qkv             q/k/v 共享一次量化（数值逐位不变）
    +F3 fuse_epilogue         norm+mod+quant / gelu+quant 折成一遍
    上界                       由 E1 占比 + E0 GEMM 上界算出

同时报 **latent 数值偏差**：F1/F2 应逐位不变，F3 会变（中间张量存储精度变了）。
T4-A 给了判读这个偏差的尺子：两个数学等价的真 kernel 配置已相差 0.1204。

用法：
    $PY benchmarks/e3_fusion_latency.py --device 0 --out results/E3/fusion_latency.json
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
LATENT_SHAPE = (21, 16, 60, 104)
DSL = [1000, 750, 500, 250]
SHIFT = 5.0
PROMPT_SEED = 242

sys.path.insert(0, REPO)
sys.path.insert(0, SF_ROOT)


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def sched_4():
    s = torch.linspace(1.0, 0.0, 1001)[:-1]
    sig = 5.0 * s / (1 + 4.0 * s)
    idx = [1000 - x for x in DSL]
    return sig[idx].tolist(), (sig * 1000)[idx].tolist()


def build(dev):
    from utils.wan_wrapper import WanDiffusionWrapper
    w = WanDiffusionWrapper(is_causal=False)
    sd = torch.load(DISTILL_CKPT, map_location="cpu", weights_only=False)
    miss, unexp = w.model.load_state_dict(sd, strict=False)
    assert not miss and not unexp
    del sd
    w = w.to(dev).to(torch.bfloat16).eval()
    w.model.requires_grad_(False)
    return w


def run4(w, cond, noise, sigmas, timesteps, dev):
    lat = noise.to(device=dev, dtype=torch.float32)
    F = lat.shape[1]
    for i in range(len(sigmas)):
        ts = torch.tensor(timesteps[i], device=dev).float().view(1, 1).expand(1, F)
        v, _ = w(lat.to(torch.bfloat16), cond, ts)
        v = v.float()
        x0 = lat - sigmas[i] * v
        lat = x0 + sigmas[i + 1] * v if i < len(sigmas) - 1 else x0
    return lat


def profile_buckets(fn, dev):
    """profile 一次前向，按 op 桶统计 self device time。

    只取 aten 层（device_type=CPU），**不能同时累加 CUDA kernel 层**——
    那会恰好双计一遍（T3 §7(1) 的教训，`CLAUDE.md` 通则 1）。
    分类规则与 `benchmarks/e1_stage_profile.py` 保持一致，直接复用。
    """
    from torch.profiler import profile, ProfilerActivity
    from torch.autograd import DeviceType
    sys.path.insert(0, os.path.join(REPO, "benchmarks"))
    from e1_stage_profile import classify, dev_time
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    buckets, per_op = {}, {}
    for e in prof.key_averages():
        if getattr(e, "device_type", DeviceType.CPU) != DeviceType.CPU:
            continue
        us = dev_time(e)
        if us <= 0:
            continue
        buckets[classify(e.key)] = buckets.get(classify(e.key), 0.0) + us
        per_op[e.key] = per_op.get(e.key, 0.0) + us
    tot = sum(buckets.values())
    return {"total_device_ms": tot / 1e3,
            "buckets_share": {k: v / tot for k, v in
                              sorted(buckets.items(), key=lambda kv: -kv[1])},
            "top_ops_ms": {k[:100]: v / 1e3 for k, v in
                           sorted(per_op.items(), key=lambda kv: -kv[1])[:15]}}


def bench(fn, reps):
    fn()
    torch.cuda.synchronize()
    ts = []
    r = None
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], ts, r


ARMS = [
    ("bf16_eager",      dict(kind="bf16")),
    ("bf16_compiled",   dict(kind="bf16c")),
    ("fp8_naive",       dict(kind="old", mode="pertensor_naive")),
    ("fp8_compiled",    dict(kind="old", mode="pertensor")),
    ("fp8_F1_bias",     dict(kind="fused", share=False, bias=True,  epi=False)),
    ("fp8_F1F2_shareq", dict(kind="fused", share=True,  bias=True,  epi=False)),
    ("fp8_F1F2F3_full", dict(kind="fused", share=True,  bias=True,  epi=True)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="results/E3/fusion_latency.json")
    ap.add_argument("--arms", default="")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = max(
        getattr(torch._dynamo.config, "cache_size_limit", 8), 128)
    torch.cuda.set_device(args.device)
    dev = f"cuda:{args.device}"
    p = torch.cuda.get_device_properties(args.device)
    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.chdir(SF_ROOT)

    from quant.fp8 import wan_fp8
    from fusion import wan_fused

    cond_cpu = torch.load(os.path.join(REPO, "results/E1/prompt_embeds.pt"),
                          weights_only=False)["cond"]
    cond = {"prompt_embeds": cond_cpu.to(dev).to(torch.bfloat16)}
    sigmas, timesteps = sched_4()
    g = torch.Generator(device=dev).manual_seed(PROMPT_SEED)
    noise = torch.randn(1, *LATENT_SHAPE, device=dev, dtype=torch.float32, generator=g)

    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    rows, lats = {}, {}
    for name, cfg in ARMS:
        if want and name not in want:
            continue
        try:
            w = build(dev)
            if cfg["kind"] == "bf16c":
                n = wan_fused.convert_bf16_fused(w.model)
            elif cfg["kind"] == "old":
                n = len(wan_fp8.convert_fp8(w.model, mode=cfg["mode"]))
            elif cfg["kind"] == "fused":
                n = len(wan_fused.convert_fused(
                    w.model, share_quant=cfg["share"], fuse_bias=cfg["bias"],
                    fuse_epilogue=cfg["epi"]))
            else:
                n = 0
            free()
            med, ts, lat = bench(
                lambda: run4(w, cond, noise, sigmas, timesteps, dev), args.reps)
            rows[name] = {"ok": True, "n_linears": n, "denoise_s": med,
                          "denoise_s_all": ts, "ms_per_forward": med * 1e3 / 4,
                          "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                          "latent_finite": bool(torch.isfinite(lat).all())}
            lats[name] = lat.float().cpu()
            print(f"[{name:18s}] {med:7.3f} s  {med*1e3/4:6.0f} ms/fwd  "
                  f"peak {rows[name]['peak_alloc_gib']:.2f} GiB  ({n} linears)",
                  flush=True)
            if name.startswith("bf16"):
                # Amdahl 上界由 GEMM 占比推出，基线一换占比就可能变 —— 必须重测
                one = lambda: run4(w, cond, noise, sigmas, timesteps, dev)
                rows[name]["profile"] = profile_buckets(one, dev)
                bs = rows[name]["profile"]["buckets_share"]
                print(f"{'':20s} buckets: " + "  ".join(
                    f"{k}={v*100:.1f}%" for k, v in list(bs.items())[:5]), flush=True)
            del w, lat
            free()
        except Exception as e:
            rows[name] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"[{name:18s}] FAILED: {type(e).__name__}: {e}", flush=True)
            free()

    # ⚠️ 两套基线都报。主表用 compiled 基线——那才是「编译」这个变量被控住之后
    # 的 FP8 净收益。eager 基线保留，因为它是 T3/T4 报过的那个口径。
    be = rows.get("bf16_eager", {}).get("denoise_s")
    bc = rows.get("bf16_compiled", {}).get("denoise_s")
    for k, v in rows.items():
        if not v.get("ok"):
            continue
        if be:
            v["speedup_vs_bf16_eager"] = be / v["denoise_s"]
        if bc:
            v["speedup_vs_bf16_compiled"] = bc / v["denoise_s"]
        v["denoise_speedup_vs_bf16"] = v.get("speedup_vs_bf16_compiled",
                                             v.get("speedup_vs_bf16_eager"))

    # 数值偏差：以 fp8_compiled 为参照（它是 T3 报的那一档）
    if "fp8_compiled" in lats:
        ref = lats["fp8_compiled"]
        for k, v in lats.items():
            if k.startswith("bf16"):
                continue
            rows[k]["latent_rel_l2_vs_fp8_compiled"] = float(
                (v.double() - ref.double()).norm() / ref.double().norm())
    if "bf16_eager" in lats:
        rb = lats["bf16_eager"]
        for k, v in lats.items():
            rows[k]["latent_rel_l2_vs_bf16"] = float(
                (v.double() - rb.double()).norm() / rb.double().norm())

    # 四点折线 + 增量归因
    sp = {k: rows[k].get("denoise_speedup_vs_bf16") for k in rows if rows[k].get("ok")}
    FP8_GEMM = 1.895            # E0 实测
    ceilings = {}
    for b in ("bf16_eager", "bf16_compiled"):
        pr = rows.get(b, {}).get("profile")
        if pr:
            g = pr["buckets_share"].get("gemm", 0.0)
            ceilings[b] = {"gemm_share": g,
                           "attention_share": pr["buckets_share"].get("attention", 0.0),
                           "denoise_ceiling_fp8_only": 1.0 / ((1 - g) + g / FP8_GEMM)}
    ceiling = (ceilings.get("bf16_compiled") or ceilings.get("bf16_eager") or {}
               ).get("denoise_ceiling_fp8_only")
    line = {
        "1_naive": sp.get("fp8_naive"),
        "2_compiled_quant": sp.get("fp8_compiled"),
        "3_fused_F1F2F3": sp.get("fp8_F1F2F3_full"),
        "4_amdahl_ceiling": ceiling,
        "0_bare_gemm_E0": 1.895,
    }
    inc = {}
    order = [("bf16_eager", "bf16_compiled", "BASELINE: compile the BF16 side too"),
             ("fp8_naive", "fp8_compiled", "compiled quant (3 passes -> 1)"),
             ("fp8_compiled", "fp8_F1_bias", "F1 bias into _scaled_mm epilogue"),
             ("fp8_F1_bias", "fp8_F1F2_shareq", "F2 share q/k/v quantization"),
             ("fp8_F1F2_shareq", "fp8_F1F2F3_full", "F3 fuse norm/gelu epilogue")]
    for a, b, label in order:
        if sp.get(a) and sp.get(b):
            inc[label] = {"from": sp[a], "to": sp[b], "delta_x": sp[b] - sp[a],
                          "delta_pct_of_denoise": 100 * (1 / sp[a] - 1 / sp[b])}
    remaining = None
    if ceiling and sp.get("fp8_F1F2F3_full"):
        remaining = 100 * (1 / sp["fp8_F1F2F3_full"] - 1 / ceiling)

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": p.name, "torch": torch.__version__,
        "config": {"steps": 4, "forwards": 4, "seq_len": 32760, "seed": PROMPT_SEED},
        "kernel": "torch._scaled_mm, use_fast_accum=True, out_dtype=bf16",
        "rows": rows,
        "four_point_line": line,
        "baselines": {"note": "two baselines are reported. The main table uses the "
                              "COMPILED bf16 baseline: only then is 'compiled vs eager' "
                              "controlled for, and the remaining delta attributable to FP8. "
                              "The eager baseline is what T3/T4 reported.",
                      "bf16_eager_s": rows.get("bf16_eager", {}).get("denoise_s"),
                      "bf16_compiled_s": rows.get("bf16_compiled", {}).get("denoise_s"),
                      "compile_only_speedup_on_bf16":
                          (rows["bf16_eager"]["denoise_s"] / rows["bf16_compiled"]["denoise_s"])
                          if rows.get("bf16_eager", {}).get("ok") and
                             rows.get("bf16_compiled", {}).get("ok") else None},
        "amdahl_ceilings_recomputed": ceilings,
        "incremental_attribution": inc,
        "remaining_gap_to_ceiling_pct_of_denoise": remaining,
        "numerics_note": "F1 (native bias) and F2 (shared q/k/v quant) do not change the "
                         "math; F3 changes the storage precision of the norm/GELU "
                         "intermediates and therefore changes the result. Read the "
                         "latent deltas against the T4-A resolution floor: two "
                         "mathematically equivalent real-kernel configs already differ "
                         "by 0.1204 (results/E4A/path_parity.json).",
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*72}")
    if "bf16_eager" in rows and "bf16_compiled" in rows and rows["bf16_compiled"].get("ok"):
        print(f"BASELINE: bf16 eager {rows['bf16_eager']['denoise_s']:.3f}s -> "
              f"compiled {rows['bf16_compiled']['denoise_s']:.3f}s = "
              f"{rows['bf16_eager']['denoise_s']/rows['bf16_compiled']['denoise_s']:.3f}x "
              f"(compile alone, on BF16)")
    for b, c in ceilings.items():
        print(f"  ceiling[{b:14s}] gemm={c['gemm_share']*100:.1f}% "
              f"attn={c['attention_share']*100:.1f}% -> "
              f"{c['denoise_ceiling_fp8_only']:.3f}x")
    print("four-point line (denoise speedup vs COMPILED bf16):")
    for k in ("0_bare_gemm_E0", "1_naive", "2_compiled_quant", "3_fused_F1F2F3",
              "4_amdahl_ceiling"):
        v = line.get(k)
        print(f"    {k:22s} {v if v is None else f'{v:.3f}x'}")
    print("incremental attribution:")
    for k, v in inc.items():
        print(f"    {k:38s} {v['from']:.3f} -> {v['to']:.3f}  "
              f"({v['delta_pct_of_denoise']:+.2f}% of denoise time)")
    if remaining is not None:
        print(f"remaining gap to Amdahl ceiling: {remaining:.2f}% of denoise time")
    print("latent deltas (vs fp8_compiled):")
    for k, v in rows.items():
        if v.get("ok") and "latent_rel_l2_vs_fp8_compiled" in v:
            print(f"    {k:20s} {v['latent_rel_l2_vs_fp8_compiled']:.5f}")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
