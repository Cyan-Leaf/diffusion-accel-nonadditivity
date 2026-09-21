#!/usr/bin/env python3
"""e2_scale_granularity.py — scale 粒度对 FP8 与 INT8 的价值差多少（格式轴的直接证据）

**动机（T3 期间浮现，不在原 SPEC 里）**：

E2 里出现两个乍看矛盾的观察：
  1. `clip` 在 Wan 上有 225/300 层选了 γ<1，但 recovery 只涨 +0.03 pp
  2. 把权重 scale 从 per-tensor 换成 per-channel，端到端 PSNR **反而略差**

如果 per-channel 真的更细，权重误差应该单调下降。观察 2 与此矛盾，
除非 **scale 粒度对 FP8 本身就几乎没价值** —— 那与 `clip` 无效是同一个机制：

> 浮点码本的格距是**相对**的（E4M3 在任意量级上都保留约 3 位尾数），
> 所以把 scale 调细或调紧只是整体平移，不改变有效位数；
> 整数码本的格距是**绝对**的，scale 直接决定 bulk 的分辨率。
> —— 一份更早的实验记录

端到端 PSNR 是「对参照样本的偏离度」，对混沌采样器不单调，**不能用来判这件事**。
本脚本绕开采样器，直接量**纯权重量化误差**：在真实 Wan 权重上，
per-tensor / per-channel / per-group 三种粒度 × FP8 / INT8 两种码本。

若「FP8 上粒度不值钱、INT8 上值钱」成立，则 `SPEC.md` §4.8.1 的结论可以从
「clip 的必要性是格式属性」升级为更一般的
**「scale 自由度（粒度与 clip 都算）的价值是格式属性，不是算法属性」**。

用法：
    $PY benchmarks/e2_scale_granularity.py --out results/E2/scale_granularity.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

REPO = os.environ.get("GENMODEL_ACCEL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DISTILL_CKPT = os.environ.get("DISTILL_CKPT", "")   # lightx2v 4 步蒸馏权重 (rev ef72050)
assert DISTILL_CKPT, "请先 export DISTILL_CKPT=<distill_native.pt 的路径>"

sys.path.insert(0, REPO)

# ---------- T6-B：按指数位数造 minifloat 码本 ------------------------------
# 目的：把 E7 的结论从「NVFP4 更像整数」这个**二分**，升级成
#       「scale 粒度的价值随**指数位数**单调递减」这条**带自变量的规律**。
# 规律能外推到没测过的格式，二分不能。
#
# E2M3 与 E2M1 同为 2 指数位、尾数位不同 —— 若两者落在相近位置，
# 说明决定因素是**指数位**而不是总位宽，规律更硬。

def minifloat_levels(e_bits, m_bits, reserve_inf_nan):
    """minifloat 的全部有限可表示值（含次正规）。bias = 2^(e-1) - 1。

    ⚠️ `reserve_inf_nan` 必须按**各格式的实际标准**设：
      - OCP FP6 (E3M2/E2M3) 与 FP4 (E2M1) **不保留** inf/nan，指数码全用满
      - IEEE 风格的 FP8 E5M2 **保留**指数全 1 给 inf/nan
    第一版我对所有合成格式统一保留了 inf/nan，结果 E2M1 的 max 会变成 3
    而真实 NVFP4 是 6 —— **那样 E2M3 与 E2M1 的对比就被约定差异污染了**。
    """
    bias = 2 ** (e_bits - 1) - 1
    vals = set()
    emax = 2 ** e_bits - 1
    for e in range(0, emax + 1):
        if reserve_inf_nan and e == emax:
            continue
        for m in range(2 ** m_bits):
            v = ((m / 2 ** m_bits) * 2.0 ** (1 - bias) if e == 0
                 else (1 + m / 2 ** m_bits) * 2.0 ** (e - bias))
            vals.add(v)
            vals.add(-v)
    return sorted(vals)


def _register_minifloats():
    """按各格式的实际标准注册。expected_max 是写死的自检值（通则 2）。"""
    from quant.gptq.formats import Codebook, CODEBOOKS
    specs = {                      # name: (e, m, reserve_inf_nan, expected_max)
        "fp8_e5m2": (5, 2, True,  57344.0),    # IEEE 风格
        "fp6_e3m2": (3, 2, False, 28.0),       # OCP FP6
        "fp6_e2m3": (2, 3, False, 7.5),        # OCP FP6
    }
    for name, (e, m, r, want) in specs.items():
        lv = minifloat_levels(e, m, r)
        assert abs(max(lv) - want) < 1e-9, (name, max(lv), want)
        if name not in CODEBOOKS:
            CODEBOOKS[name] = Codebook(lv, name)
    # NVFP4 的 E2M1 已在 CODEBOOKS 里（max 6，OCP 约定），顺带核一下
    assert abs(CODEBOOKS["nvfp4"].absmax - 6.0) < 1e-9, CODEBOOKS["nvfp4"].absmax
    return specs


BLOCK_SUFFIX = ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
                "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
                "ffn.0", "ffn.2"]

SPECS = [
    ("fp8_pertensor",  "fp8_e4m3:-1:fp32"),
    ("fp8_perchannel", "fp8_e4m3:0:fp32"),
    ("fp8_group128",   "fp8_e4m3:128:fp32"),
    ("fp8_group16",    "fp8_e4m3:16:fp32"),
    ("int8_pertensor",  "int8:-1:fp32"),
    ("int8_perchannel", "int8:0:fp32"),
    ("int8_group128",   "int8:128:fp32"),
    ("int8_group16",    "int8:16:fp32"),
    # ---- T5-C：E7 的头条。T3 §5.4 留下的可证伪预测是
    #      「NVFP4 E2M1 在这条轴上应更像**整数**码本」——因为 ±6 的动态范围下
    #      「格距相对于量级」这个性质不再成立。加这四行就能验。
    #      注意 nvfp4 的 scale_fmt 用 e4m3（两级 scale，与部署一致）。
    ("nvfp4_pertensor",  "nvfp4:-1:e4m3"),
    ("nvfp4_perchannel", "nvfp4:0:e4m3"),
    ("nvfp4_group128",   "nvfp4:128:e4m3"),
    ("nvfp4_group16",    "nvfp4:16:e4m3"),
    # ---- 隔离 SPEC §4.7 的 D2：NVFP4 的 block scale 自身被量化成 E4M3。
    #      同样的 E2M1 网格 + **未量化的 FP32 scale** 作对照，差额就是 D2 的代价。
    ("nvfp4grid_fp32scale_group16",  "nvfp4:16:fp32"),
    ("nvfp4grid_fp32scale_group128", "nvfp4:128:fp32"),
    ("nvfp4grid_fp32scale_perchan",  "nvfp4:0:fp32"),
    ("nvfp4grid_fp32scale_pertensor","nvfp4:-1:fp32"),
    # 对照：同为 4 bit 的整数码本，用来判断 nvfp4 落在 fp8 那一侧还是 int 那一侧
    ("int4_pertensor",  "int4:-1:fp32"),
    ("int4_perchannel", "int4:0:fp32"),
    ("int4_group128",   "int4:128:fp32"),
    ("int4_group16",    "int4:16:fp32"),
    # ---- T6-B：指数位曲线的中间点 ----
    ("fp8e5m2_pertensor",  "fp8_e5m2:-1:fp32"),
    ("fp8e5m2_perchannel", "fp8_e5m2:0:fp32"),
    ("fp8e5m2_group128",   "fp8_e5m2:128:fp32"),
    ("fp8e5m2_group16",    "fp8_e5m2:16:fp32"),
    ("fp6e3m2_pertensor",  "fp6_e3m2:-1:fp32"),
    ("fp6e3m2_perchannel", "fp6_e3m2:0:fp32"),
    ("fp6e3m2_group128",   "fp6_e3m2:128:fp32"),
    ("fp6e3m2_group16",    "fp6_e3m2:16:fp32"),
    ("fp6e2m3_pertensor",  "fp6_e2m3:-1:fp32"),
    ("fp6e2m3_perchannel", "fp6_e2m3:0:fp32"),
    ("fp6e2m3_group128",   "fp6_e2m3:128:fp32"),
    ("fp6e2m3_group16",    "fp6_e2m3:16:fp32"),
]

# 格式 -> (指数位, 尾数位, 总位宽)。整数码本的指数位记为 0。
FORMAT_BITS = {
    "int8":      (0, 7, 8),
    "int4":      (0, 3, 4),
    "nvfp4":     (2, 1, 4),
    "fp6e2m3":   (2, 3, 6),
    "fp6e3m2":   (3, 2, 6),
    "fp8":       (4, 3, 8),      # e4m3
    "fp8e5m2":   (5, 2, 8),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="results/E2/scale_granularity.json")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    dev = f"cuda:{args.device}"
    from quant.gptq import formats
    _register_minifloats()

    sd = torch.load(DISTILL_CKPT, map_location="cpu", weights_only=True)
    names = [f"blocks.{i}.{s}.weight" for i in range(30) for s in BLOCK_SUFFIX]
    names = [n for n in names if n in sd]
    print(f"{len(names)} weight tensors from {os.path.basename(DISTILL_CKPT)}")

    fmts = {tag: formats.parse_wfmt(spec) for tag, spec in SPECS}
    acc = {tag: [] for tag in fmts}
    per_role = {}

    t0 = time.time()
    for i, n in enumerate(names):
        W = sd[n].to(dev).float()
        role = ".".join(n.split(".")[2:-1])          # e.g. self_attn.q
        nrm = W.norm()
        for tag, wf in fmts.items():
            Q = wf.rtn(W)
            rel = float((W - Q).norm() / nrm)
            acc[tag].append(rel)
            per_role.setdefault(role, {}).setdefault(tag, []).append(rel)
        del W
        if i % 60 == 0:
            print(f"  {i+1}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)

    summary = {tag: {"rel_fro_err_mean": sum(v) / len(v),
                     "rel_fro_err_max": max(v),
                     "rel_fro_err_min": min(v)}
               for tag, v in acc.items()}

    # 「把粒度调细一档能省多少误差」——这是格式轴要的那个比较
    gain = {}
    for fam in FORMAT_BITS:
        base = summary[f"{fam}_pertensor"]["rel_fro_err_mean"]
        gain[fam] = {
            "pertensor": 1.0,
            "perchannel": base / summary[f"{fam}_perchannel"]["rel_fro_err_mean"],
            "group128": base / summary[f"{fam}_group128"]["rel_fro_err_mean"],
            "group16": base / summary[f"{fam}_group16"]["rel_fro_err_mean"],
        }

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": torch.cuda.get_device_name(args.device),
        "ckpt": DISTILL_CKPT,
        "n_tensors": len(names),
        "metric": "relative Frobenius error ||W - RTN(W)||_F / ||W||_F, "
                  "pure weight quantization, no sampler involved",
        "summary": summary,
        "error_reduction_vs_pertensor": gain,
        "per_role_mean": {r: {t: sum(v) / len(v) for t, v in d.items()}
                          for r, d in per_role.items()},
        "interpretation": (
            "If finer scale granularity reduces error much more for int8 than for fp8, "
            "then scale-side degrees of freedom (granularity AND clipping) are worth "
            "little on a floating-point codebook, because E4M3 already keeps ~3 mantissa "
            "bits at every magnitude. That is the same mechanism behind the clip result."),
    }
    out = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'granularity':13s}" + "".join(f"{f:>12s}" for f in
          ("FP8 err", "INT8 err", "NVFP4 err", "INT4 err")) +
          "".join(f"{f:>11s}" for f in ("FP8 g", "INT8 g", "NVFP4 g", "INT4 g")))
    for g in ("pertensor", "perchannel", "group128", "group16"):
        errs = [summary[f"{f}_{g}"]["rel_fro_err_mean"] for f in
                ("fp8", "int8", "nvfp4", "int4")]
        gs = [gain[f][g] for f in ("fp8", "int8", "nvfp4", "int4")]
        print(f"{g:13s}" + "".join(f"{e:12.6f}" for e in errs) +
              "".join(f"{x:11.3f}" for x in gs))

    # ---- E7 的可证伪预测：nvfp4 落在 fp8 那一侧还是整数那一侧？----------
    pred = {}
    for g in ("perchannel", "group128", "group16"):
        pred[g] = {"fp8_gain": gain["fp8"][g], "int8_gain": gain["int8"][g],
                   "int4_gain": gain["int4"][g], "nvfp4_gain": gain["nvfp4"][g]}
        # nvfp4 更接近哪一侧（对数距离，因为这些是比值）
        import math
        dn = abs(math.log(gain["nvfp4"][g]) - math.log(gain["fp8"][g]))
        di = abs(math.log(gain["nvfp4"][g]) - math.log(gain["int4"][g]))
        pred[g]["closer_to"] = "fp8(float-like)" if dn < di else "int4(integer-like)"
        pred[g]["log_dist_to_fp8"] = dn
        pred[g]["log_dist_to_int4"] = di
    # ---- D2 的代价：E4M3 scale vs FP32 scale，同一 E2M1 网格 ----------
    d2 = {}
    for g, tag in (("group16", "group16"), ("group128", "group128"),
                   ("perchannel", "perchan"), ("pertensor", "pertensor")):
        a = summary[f"nvfp4_{g}"]["rel_fro_err_mean"]
        b = summary[f"nvfp4grid_fp32scale_{tag}"]["rel_fro_err_mean"]
        d2[g] = {"e4m3_scale_err": a, "fp32_scale_err": b,
                 "d2_cost_ratio": a / b, "d2_cost_pct": 100 * (a / b - 1)}
    payload["D2_scale_quantization_cost"] = {
        "claim": "SPEC 4.7 D2: NVFP4's block scale is itself quantized to FP8 E4M3, so "
                 "the true grid is shifted relative to the FP32-optimal one. This block "
                 "isolates that cost by holding the E2M1 element grid fixed and only "
                 "changing the scale format.",
        "per_granularity": d2,
    }
    # ---- T6-B：指数位 -> scale 粒度收益 的曲线 -------------------------
    curve = []
    for fam, (eb, mb, tb) in FORMAT_BITS.items():
        if fam not in gain:
            continue
        curve.append({"format": fam, "exp_bits": eb, "mant_bits": mb, "total_bits": tb,
                      "absmax": float(max(abs(x) for x in
                          __import__("quant.gptq.formats", fromlist=["CODEBOOKS"])
                          .CODEBOOKS[{"fp8":"fp8_e4m3","fp8e5m2":"fp8_e5m2",
                                      "fp6e3m2":"fp6_e3m2","fp6e2m3":"fp6_e2m3"}
                                     .get(fam, fam)].levels.tolist())),
                      "gain_perchannel": gain[fam]["perchannel"],
                      "gain_group128": gain[fam]["group128"],
                      "gain_group16": gain[fam]["group16"],
                      "err_group16": summary[f"{fam}_group16"]["rel_fro_err_mean"]})
    curve.sort(key=lambda r: (r["exp_bits"], r["total_bits"]))
    # 单调性检验：按指数位分组取平均，看是否随指数位递减
    bye = {}
    for r in curve:
        bye.setdefault(r["exp_bits"], []).append(r["gain_group16"])
    means = {e: sum(v) / len(v) for e, v in sorted(bye.items())}
    ks = sorted(means)
    mono = all(means[ks[i]] >= means[ks[i + 1]] for i in range(len(ks) - 1))
    # 严格单调会被平台期上的数值噪声打破，所以同时给**容差版**与**饱和点**。
    TOL = 0.02          # 2%：低于此的"上升"视为噪声
    mono_tol = all(means[ks[i]] >= means[ks[i + 1]] * (1 - TOL)
                   for i in range(len(ks) - 1))
    # 饱和点：从哪个指数位起，后续各点彼此相差 < TOL
    sat = None
    for i in range(len(ks)):
        tail = [means[k] for k in ks[i:]]
        if len(tail) >= 2 and (max(tail) - min(tail)) / max(tail) < TOL:
            sat = ks[i]
            break
    plateau = {str(k): means[k] for k in ks if sat is not None and k >= sat}
    # 「决定因素是指数位还是总位宽」：同指数位不同位宽的组内离散 vs 组间跨度
    within = max((max(v) - min(v)) for v in bye.values() if len(v) > 1) if any(
        len(v) > 1 for v in bye.values()) else 0.0
    between = max(means.values()) - min(means.values())
    payload["T6B_exponent_bit_curve"] = {
        "claim_under_test": "the value of finer scale granularity decreases monotonically "
                            "with the number of exponent bits in the codebook",
        "curve": curve,
        "mean_gain_group16_by_exp_bits": means,
        "monotone_decreasing_strict": mono,
        "monotone_decreasing_within_2pct_tolerance": mono_tol,
        "saturation_exp_bits": sat,
        "plateau_values": plateau,
        "reading": ("the gain decreases with exponent bits and then SATURATES: e=0 -> "
                    f"{means[ks[0]]:.3f}, e=2 -> {means.get(2, float('nan')):.3f}, and "
                    f"e>={sat} is a plateau at ~{sum(plateau.values())/len(plateau):.3f} "
                    "if plateau is non-empty. The strict-monotonicity flag can be False "
                    "purely from sub-1% noise on the plateau -- read the plateau, not the flag."),
        "within_exp_bit_spread": within,
        "between_exp_bit_spread": between,
        "driver_is_exponent_bits_not_total_width": bool(within < 0.5 * between),
        "convention_note": "FP8 E4M3 levels come from the hardware dtype (finite-only, "
                           "max 448). FP8 E5M2 follows IEEE (inf/nan reserved, max 57344). "
                           "FP6 E3M2/E2M3 and FP4 E2M1 follow OCP (no inf/nan; max 28 / "
                           "7.5 / 6). Each format uses ITS OWN standard convention.",
    }
    payload["E7_prediction_test"] = {
        "prediction": "NVFP4 E2M1 should behave MORE LIKE AN INTEGER codebook on this "
                      "axis, because at a dynamic range of +-6 the 'step size is relative "
                      "to magnitude' property no longer holds (T3 section 5.4).",
        "per_granularity": pred,
        "verdict": ("CONFIRMED" if all(v["closer_to"].startswith("int") for v in pred.values())
                    else "REFUTED" if all(v["closer_to"].startswith("fp8") for v in pred.values())
                    else "MIXED"),
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print("\nE7 prediction test (is NVFP4 float-like or integer-like on this axis?):")
    for g, v in pred.items():
        print(f"    {g:12s} nvfp4 gain {v['nvfp4_gain']:6.3f}  "
              f"(fp8 {v['fp8_gain']:.3f} / int4 {v['int4_gain']:.3f})  -> {v['closer_to']}")
    print(f"    VERDICT: {payload['E7_prediction_test']['verdict']}")
    c = payload["T6B_exponent_bit_curve"]
    print(f"\nT6-B exponent-bit curve (gain from per-tensor -> group-16):")
    print(f"    {'format':10s}{'exp':>5s}{'mant':>6s}{'bits':>6s}{'absmax':>10s}"
          f"{'perchan':>9s}{'g128':>8s}{'g16':>8s}")
    for r in c["curve"]:
        print(f"    {r['format']:10s}{r['exp_bits']:5d}{r['mant_bits']:6d}"
              f"{r['total_bits']:6d}{r['absmax']:10.1f}"
              f"{r['gain_perchannel']:9.3f}{r['gain_group128']:8.3f}{r['gain_group16']:8.3f}")
    print(f"    mean g16 gain by exp bits: " +
          "  ".join(f"{e}->{v:.3f}" for e, v in c["mean_gain_group16_by_exp_bits"].items()))
    print(f"    MONOTONE (strict): {c['monotone_decreasing_strict']}   "
          f"(within 2% tol): {c['monotone_decreasing_within_2pct_tolerance']}")
    print(f"    SATURATES at exp_bits >= {c['saturation_exp_bits']}, plateau = "
          + ", ".join(f"e{k}:{v:.3f}" for k, v in c["plateau_values"].items()))
    print(f"    within-exp-bit spread {c['within_exp_bit_spread']:.3f} vs "
          f"between {c['between_exp_bit_spread']:.3f} -> "
          f"driver is exponent bits: {c['driver_is_exponent_bits_not_total_width']}")
    print("\nD2 cost (same E2M1 grid, E4M3 scale vs FP32 scale):")
    for g, v in payload["D2_scale_quantization_cost"]["per_granularity"].items():
        print(f"    {g:12s} e4m3 {v['e4m3_scale_err']:.6f}  fp32 {v['fp32_scale_err']:.6f}"
              f"  -> D2 costs {v['d2_cost_pct']:+.2f}%")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
