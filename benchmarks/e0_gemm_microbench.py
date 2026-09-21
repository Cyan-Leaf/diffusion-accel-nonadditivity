#!/usr/bin/env python3
"""e0_gemm_microbench.py — E0: FP8 相对 BF16 的可得吞吐（4090 / sm_89）

这不是选型实验。FP8 已定（SPEC §2.2）。E0 只回答：4090 上 FP8 相对 BF16 能拿到多少。
INT8 那一列保留，作为 SPEC §4.8「格式轴」的第一个数据点，不是候选方案。

关键：必须用 Wan2.1-1.3B 的真实 GEMM 形状。DiT 的 GEMM 常是瘦长的，
用 4096^3 方阵测出来的 roofline 位置和真实情况完全不同。

用法:
    python benchmarks/e0_gemm_microbench.py --shapes research/gemm_shapes.json
    python benchmarks/e0_gemm_microbench.py --shapes ... --device 0 --out results/E0/gemm_bench.json

---------------------------------------------------------------------------
对交接骨架的一处更正（2026-09-18，执行者）
---------------------------------------------------------------------------
骨架里把 `torch._scaled_mm(out_dtype=...)` 当成「accumulate 精度」来测
（列名 fp8_acc_bf16 / fp8_acc_fp32）。**这是错的。**

FP8 输入走 Tensor Core 时，MMA 累加器在 Ada 上恒为 FP32，`out_dtype` 只决定
epilogue 往 HBM 写出的元素类型。所以原来那两列测的不是 accumulate 阉割，
而是「写出 2 字节 vs 4 字节」的访存差。

真正控制累加行为的开关是 `use_fast_accum`：开启后 cuBLASLt 走 FP8 快速累加
路径（降低向 FP32 提升的频率）。所以本脚本把两个轴分开测：
    out_dtype      ∈ {bfloat16, float32}   ← epilogue 写出宽度
    use_fast_accum ∈ {False, True}         ← 累加路径

L4 参考表里的 "FP16/FP32-acc" vs "FP16/FP16-acc" 说的是 MMA 累加器，
和 `out_dtype` 不是一回事，对比时不要混。

同时新增 roofline 侧信息（bytes / 算术强度 / 实测带宽 / %of peak），
因为 CP-0 的第三档判据（"形状太瘦长、memory-bound 主导"）必须靠这几个数才能判，
只看 TFLOPS 判不出来。

参考（L4, Ada 数据中心卡, NVIDIA 开发者论坛实测）:
    FP8  / FP32-acc : 188 TFLOPS
    FP16 / FP32-acc :  87 TFLOPS
    FP16 / FP16-acc :  85 TFLOPS
    INT8 / INT32-acc: 165 TOPS
    !! L4 不是 4090。本脚本就是为了填 4090 这个空。
"""
import argparse
import json
import os
import sys
import time

import torch

WARMUP = 10
ITERS = 50

# RTX 4090 (AD102, sm_89) 峰值，来源 NVIDIA Ada GPU Architecture Whitepaper v2.1
# "Appendix A - GeForce RTX 4090 Full Specifications"，格式 dense/sparse：
#   Peak FP8  Tensor TFLOPS with FP16 Accumulate : 660.6/1321.2
#   Peak FP8  Tensor TFLOPS with FP32 Accumulate : 330.3/660.6    <- 脚注 4
#   Peak FP16 Tensor TFLOPS with FP16 Accumulate : 330.3/660.6
#   Peak FP16 Tensor TFLOPS with FP32 Accumulate : 165.2/330.4
#   Peak BF16 Tensor TFLOPS with FP32 Accumulate : 165.2/330.4
#   Peak INT8 Tensor TOPS                        : 660.6/1321.2
# 白皮书脚注 4 原文：
#   "Peak FP8 Tensor TFLOPS with FP32 Accumulate changed to proper number
#    in v.2.02 of this whitepaper"
# 即 NVIDIA 自己把 FP8/FP32-acc 从 660.6 更正为 330.3。HANDOFF 里引的「白皮书标 660」
# 是 FP8/**FP16**-Accumulate 那一行（或更正前的旧值），与 FP32-acc 实测不同档，不可直接比。
PEAK = {
    "bf16_fp32acc_dense_tflops": 165.2,
    "fp8_fp32acc_dense_tflops": 330.3,
    "fp8_fp16acc_dense_tflops": 660.6,
    "int8_dense_tops": 660.6,
    "hbm_gbps": 1008.0,
    "source": "NVIDIA Ada GPU Architecture Whitepaper v2.1, Appendix A",
}

DTYPE_BYTES = {
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
    "fp32": 4,
}


def make_l2_flusher(dev):
    """4090 有 72MB L2。小形状的输入能整个躺在 L2 里，测出来是缓存带宽不是 HBM 带宽。
    每次计时前冲掉 L2，让不同形状之间可比。"""
    buf = torch.empty(int(128e6 // 4), device=dev, dtype=torch.float32)
    return lambda: buf.zero_()


def bench(fn, flush, warmup=WARMUP, iters=ITERS):
    """返回中位数耗时(ms)。用 median 而非 mean，抗时钟抖动。
    用 cuda event 计时；每次迭代前冲 L2 并同步，把缓存效应从形状间的比较里去掉。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        flush()
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        end.synchronize()
        ts.append(start.elapsed_time(end))
    ts.sort()
    return ts[len(ts) // 2]


def flops(m, n, k):
    return 2.0 * m * n * k


def gemm_bytes(m, n, k, in_bytes, out_bytes):
    """最优情况下的访存量：读 A、读 B、写 C 各一遍。
    真实 kernel 因 tiling 会多读，所以这是下界 —— 用它算出的『实测带宽』
    是下界，用它算出的算术强度是上界。"""
    return m * k * in_bytes + k * n * in_bytes + m * n * out_bytes


def _metrics(m, n, k, ms, in_bytes, out_bytes, unit="tflops"):
    f = flops(m, n, k)
    b = gemm_bytes(m, n, k, in_bytes, out_bytes)
    sec = ms * 1e-3
    return {
        "ms": ms,
        unit: f / sec / 1e12,
        "bytes_lb": b,
        "arith_intensity_ub": f / b,
        "eff_bw_gbps_lb": b / sec / 1e9,
    }


def run_bf16(m, n, k, dev, flush):
    try:
        a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=dev, dtype=torch.bfloat16)
        fn = lambda: torch.mm(a, b)
        fn()
        ms = bench(fn, flush)
        r = {"ok": True}
        r.update(_metrics(m, n, k, ms, DTYPE_BYTES["bf16"], DTYPE_BYTES["bf16"]))
        r["pct_of_peak"] = r["tflops"] / PEAK["bf16_fp32acc_dense_tflops"]
        return r
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}"}


def run_fp8(m, n, k, dev, flush, out_dtype, fast_accum):
    """FP8 E4M3 via torch._scaled_mm。

    layout 约束：b 必须是列主序，所以构造成 (n,k) contiguous 再 .t()。
    形状约束：K 需 16 的倍数；M 太小（<16）cuBLASLt 会直接拒。
    out_dtype 只是 epilogue 写出类型，不是累加精度（见文件头更正）。
    """
    if not hasattr(torch, "float8_e4m3fn"):
        return {"ok": False, "err": "torch.float8_e4m3fn missing"}
    try:
        a = torch.randn(m, k, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = torch.randn(n, k, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
        sa = torch.tensor(1.0, device=dev)
        sb = torch.tensor(1.0, device=dev)
        fn = lambda: torch._scaled_mm(a, b, scale_a=sa, scale_b=sb,
                                      out_dtype=out_dtype, use_fast_accum=fast_accum)
        fn()  # 先跑一次暴露错误
        ms = bench(fn, flush)
        ob = 2 if out_dtype in (torch.bfloat16, torch.float16) else 4
        r = {"ok": True}
        r.update(_metrics(m, n, k, ms, DTYPE_BYTES["fp8"], ob))
        r["pct_of_peak"] = r["tflops"] / PEAK["fp8_fp32acc_dense_tflops"]
        return r
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}"}


def run_int8(m, n, k, dev, flush):
    """INT8 / INT32 acc。格式轴对照点，不是候选方案。

    torch._int_mm 在 sm_89 上走 cuBLASLt IMMA，对形状有约束
    （经验上 M>=16、K 与 N 为 8/16 的倍数）。跑不了就记 ok=False + 原因。
    """
    try:
        a = torch.randint(-127, 127, (m, k), device=dev, dtype=torch.int8)
        b = torch.randint(-127, 127, (k, n), device=dev, dtype=torch.int8)
        fn = lambda: torch._int_mm(a, b)
        fn()
        ms = bench(fn, flush)
        r = {"ok": True}
        r.update(_metrics(m, n, k, ms, DTYPE_BYTES["int8"], 4, unit="tops"))
        r["pct_of_peak"] = r["tops"] / PEAK["int8_dense_tops"]
        return r
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}"}


CONFIGS = [
    ("fp8_e4m3_out_bf16_slowacc", dict(out_dtype=torch.bfloat16, fast_accum=False)),
    ("fp8_e4m3_out_bf16_fastacc", dict(out_dtype=torch.bfloat16, fast_accum=True)),
    ("fp8_e4m3_out_fp32_slowacc", dict(out_dtype=torch.float32, fast_accum=False)),
    ("fp8_e4m3_out_fp32_fastacc", dict(out_dtype=torch.float32, fast_accum=True)),
]
FP8_KEYS = [c[0] for c in CONFIGS]


def main():
    global ITERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="research/gemm_shapes.json",
                    help="T1 产出的真实 GEMM 形状表")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="results/E0/gemm_bench.json")
    ap.add_argument("--iters", type=int, default=ITERS)
    args = ap.parse_args()

    if not os.path.exists(args.shapes):
        print(f"FATAL: {args.shapes} not found.", file=sys.stderr)
        print("T1 必须先产出真实 GEMM 形状表。不要用方阵凑数——", file=sys.stderr)
        print("DiT 的 GEMM 是瘦长的，方阵测出来的 roofline 位置是错的。", file=sys.stderr)
        sys.exit(2)

    ITERS = args.iters

    with open(args.shapes) as f:
        meta = json.load(f)
    shapes = meta["shapes"]

    dev = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)
    p = torch.cuda.get_device_properties(args.device)
    flush = make_l2_flusher(dev)
    print(f"device: {p.name} sm_{p.major}{p.minor}  torch {torch.__version__}")
    print(f"shapes: {args.shapes}  ({meta.get('note','')})\n")

    rows = []
    for s in shapes:
        m, n, k = s["m"], s["n"], s["k"]
        name = s.get("name", f"{m}x{n}x{k}")
        print(f"[{name}] M={m} N={n} K={k}   GFLOP={flops(m,n,k)/1e9:.1f}")

        r = {"name": name, "m": m, "n": n, "k": k,
             "per_step_calls": s.get("per_step_calls"), "role": s.get("role"),
             "gflop": flops(m, n, k) / 1e9}
        r["bf16"] = run_bf16(m, n, k, dev, flush)
        for key, kw in CONFIGS:
            r[key] = run_fp8(m, n, k, dev, flush, **kw)
        r["int8_acc_int32"] = run_int8(m, n, k, dev, flush)

        base = r["bf16"].get("tflops")
        if base:
            print(f"    {'bf16':28s} {base:8.1f} TF  1.00x"
                  f"  ({r['bf16']['pct_of_peak']*100:.0f}% peak,"
                  f" AI={r['bf16']['arith_intensity_ub']:.0f},"
                  f" BW>={r['bf16']['eff_bw_gbps_lb']:.0f}GB/s)")
        else:
            print(f"    {'bf16':28s} FAILED: {r['bf16']['err'][:70]}")
        for key in FP8_KEYS + ["int8_acc_int32"]:
            v = r[key]
            if v.get("ok"):
                perf = v.get("tflops", v.get("tops"))
                if base:
                    v["speedup_vs_bf16"] = perf / base
                    tag = f"{perf/base:.2f}x"
                else:
                    tag = "n/a"
                print(f"    {key:28s} {perf:8.1f}    {tag}"
                      f"  ({v['pct_of_peak']*100:.0f}% peak,"
                      f" BW>={v['eff_bw_gbps_lb']:.0f}GB/s)")
            else:
                print(f"    {key:28s} FAILED: {v['err'][:70]}")
        rows.append(r)
        torch.cuda.empty_cache()

    # ---- CP-0 判定 -------------------------------------------------------
    # 只用真正进 DiT 前向、且每步都跑的 GEMM 加权（per_step_calls>0），
    # 按 FLOPs×调用次数加权。假想的 fused QKV 和一次性的 text_embed 不计入。
    def weight_of(r):
        c = r.get("per_step_calls") or 0
        if r["name"] == "self_qkv_fused":
            return 0.0
        return flops(r["m"], r["n"], r["k"]) * c

    best = []
    for r in rows:
        w = weight_of(r)
        if w <= 0:
            continue
        cands = [(r[kk]["speedup_vs_bf16"], kk) for kk in FP8_KEYS
                 if r[kk].get("ok") and "speedup_vs_bf16" in r[kk]]
        if cands:
            sp, kk = max(cands)
            best.append({"name": r["name"], "w": w, "speedup": sp, "config": kk})
    if best:
        tot = sum(b["w"] for b in best)
        agg = sum(b["w"] * b["speedup"] for b in best) / tot
    else:
        agg = 0.0

    if agg >= 1.5:
        verdict, action = "A", "按计划走，E2 加速比目标 = 本上界 * 0.70"
    elif agg >= 1.2:
        verdict, action = "B", "仍走 FP8，报告须解释 GeForce accumulate 阉割"
    else:
        verdict, action = "C", ("E2 叙事改为『compute 侧量化在此规模下收益有限』，"
                                "重心前移到 E3(访存) 与 E4(稀疏)。这不是失败。")

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": p.name, "sm": f"{p.major}{p.minor}",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "iters": ITERS, "warmup": WARMUP,
        "shapes_file": args.shapes,
        "shapes_note": meta.get("note"),
        "peak_reference_4090": PEAK,
        "rows": rows,
        "cp0_weighting": "FLOPs x per_step_calls, DiT per-step GEMMs only; self_qkv_fused excluded (hypothetical)",
        "cp0_per_shape_best": best,
        "aggregate_fp8_speedup_vs_bf16": agg,
        "cp0_verdict": verdict,
        "cp0_action": action,
        "semantics_note": (
            "out_dtype is the epilogue output element type, NOT the MMA accumulator. "
            "FP8 MMA on Ada always accumulates in FP32. use_fast_accum is the actual "
            "accumulate-path switch. The L4 reference table's FP32-acc/FP16-acc rows "
            "refer to the MMA accumulator and are therefore NOT comparable to out_dtype."
        ),
        "reference_L4": {
            "measured_fp8_tflops": 188, "measured_fp16_fp32acc_tflops": 87,
            "measured_fp16_fp16acc_tflops": 85, "measured_int8_tops": 165,
            "spec_dense_bf16_tflops": 121, "spec_dense_fp16_tflops": 121,
            "spec_dense_fp8_tflops": 242, "spec_dense_int8_tops": 242,
            "spec_source": "NVIDIA Ada GPU Architecture Whitepaper v2.1, L4 spec table (dense | sparse)",
            "pct_of_dense_peak": {"fp8": 188 / 242, "fp16_fp32acc": 87 / 121,
                                  "fp16_fp16acc": 85 / 121, "int8": 165 / 242},
            "note": "L4 is Ada datacenter (AD104), NOT 4090 (AD102). The forum's "
                    "'whitepaper says 660' refers to the FP8-with-FP16-Accumulate row "
                    "(or the pre-v2.02 erroneous FP32-acc value), not to the accumulate "
                    "mode their 188 TFLOPS was measured under.",
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*70}")
    print(f"aggregate FP8 vs BF16 (FLOPs x calls weighted): {agg:.3f}x")
    for b in sorted(best, key=lambda d: -d["w"]):
        print(f"  {b['name']:16s} w={b['w']/1e12:8.1f} TFLOP/step  "
              f"{b['speedup']:.2f}x  via {b['config']}")
    print(f"CP-0 verdict: {verdict}  ->  {action}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
