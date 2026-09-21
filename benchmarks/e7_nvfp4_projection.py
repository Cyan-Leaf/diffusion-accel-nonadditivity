#!/usr/bin/env python3
"""e7_nvfp4_projection.py — NVFP4 的加速比**预估**（不是实测），以及显存收益（是实测）

## 为什么是预估而不是实测

NVFP4 的硬件加速只在 Blackwell 上有（sm_100/sm_120），**Ada 没有**。
在 4090 上做 fake quant，量化/反量化的额外访存必然让端到端比 BF16 还慢 ——
**那个延迟数字不具备加速比语义，本项目不报**（`SPEC.md` §4.8.5）。

**但「不报实测」不等于「什么都说不了」。** 本文的瀑布方法（§7.6）恰好就是
用来回答「一个算子级的加速比落到端到端还剩多少」的 —— 把 NVFP4 的算子级数字
代进同一套已实测的占比，就得到一个**带明确假设的预估**。

## 三个上界，必须分清（这是本脚本存在的主要理由）

| 上界 | 公式 | 它回答的问题 |
|---|---|---|
| FP8 零开销 | `1/((1-p) + p/1.895)` | 把 FP8 的 GEMM 加速无损耗落进模型 |
| **NVFP4 零开销** | `1/((1-p) + p/g_nvfp4)` | 同上，换成 NVFP4 的 GEMM 加速 |
| **绝对天花板** | `1/(1-p)` | **任何**只加速 GEMM 的手段（含 INT4、FP4、未来格式） |

⚠️ **本项目中间稿曾把「FP8 零开销上界」误标为「GEMM 无穷快的上界」。**
若照那个说法，NVFP4 的空间是「几乎没有」（FP8 已吃掉 97.9%）；
**订正后的答案是还有 13.5% 的格式侧空间。一个被叫错名字的界会让下一个决策问错问题。**

## 输入的来源（哪些是实测、哪些是假设，逐项标注）

| 量 | 值 | 来源 |
|---|---|---|
| GEMM 占 denoise | 0.2043 | **本机实测**（E3 compiled profile） |
| denoise 占端到端 | 0.4958 / 0.6597 | **本机实测**（E1，VAE fp32 / bf16） |
| FP8 裸 GEMM 加速 | 1.8951× | **本机实测**（E0） |
| FP8 的兑现率（实测/零开销上界） | 0.9794 | **本机实测**（E3 的 1.0840 / 1.1068） |
| **NVFP4 : FP8 的 GEMM 吞吐比** | **2.0×（假设）** | ⚠️ **不是本机实测** —— Blackwell 的 NVFP4 张量核标称是 FP8 的 2 倍。脚本对 1.5/2.0/3.0 都给一列 |
| NVFP4 的兑现率 | 取 FP8 的 0.9794 为**乐观**上限 | ⚠️ 假设。NVFP4 用 group-16 scale，每个 GEMM 要多读 2048 倍的 scale，**实际兑现率只会更低** |

用法：
    python3 benchmarks/e7_nvfp4_projection.py --out results/E7/nvfp4_projection.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

REPO = os.environ.get("GENMODEL_ACCEL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 权重字节数（本机实测，来自 E2/E3 的落盘）
W_BF16_GIB = 2.64
W_FP8_GIB = 1.35


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/E7/nvfp4_projection.json")
    ap.add_argument("--ratios", default="1.5,2.0,3.0",
                    help="假设的 NVFP4:FP8 GEMM 吞吐比")
    args = ap.parse_args()

    E1 = json.load(open(os.path.join(REPO, "results/E1/stage_profile.json")))
    E3 = json.load(open(os.path.join(REPO, "results/E3/fusion_latency.json")))
    L = json.load(open(os.path.join(REPO, "results/E5/ladder.json")))
    E7 = json.load(open(os.path.join(REPO, "results/E7/scale_granularity_formats.json")))

    p = L["H2"]["4step"]["disjoint_amdahl_model"]["gemm_share_p"]     # GEMM 占 denoise
    g_fp8 = 1.8950877155587238                                        # E0 实测
    a_fp8 = E3["rows"]["fp8_F1F2F3_full"]["speedup_vs_bf16_compiled"]  # 1.0840
    ceil_fp8 = 1.0 / ((1 - p) + p / g_fp8)                            # 1.1068
    ceil_abs = 1.0 / (1 - p)                                          # 1.2568
    realise = a_fp8 / ceil_fp8                                        # 0.9794

    shares = {k: E1["e2e"][f"4step/vae_{k}"]["share"]["denoise"]
              for k in ("float32", "bfloat16")}

    rows = []
    for r in [float(x) for x in args.ratios.split(",")]:
        g = g_fp8 * r
        ceil = 1.0 / ((1 - p) + p / g)
        proj = ceil * realise                       # 乐观：兑现率与 FP8 相同
        row = {"nvfp4_over_fp8_gemm_ratio": r, "assumed_bare_gemm_speedup": g,
               "denoise_ceiling_zero_overhead": ceil,
               "denoise_projected_at_fp8_realisation_rate": proj,
               "gain_over_measured_fp8_denoise": proj / a_fp8,
               "fraction_of_absolute_ceiling": ceil / ceil_abs,
               "e2e_projected": {}}
        for k, sd in shares.items():
            for nm, sp in (("ceiling", ceil), ("projected", proj)):
                row["e2e_projected"][f"vae_{k}_{nm}"] = 1.0 / ((1 - sd) + sd / sp)
        rows.append(row)

    # 显存：这一侧是真实可得的，且与延迟无关
    d2 = E7.get("D2_scale_quantization_cost", {})
    mem = {
        "weights_bf16_gib": W_BF16_GIB, "weights_fp8_gib": W_FP8_GIB,
        "weights_nvfp4_gib_naive_half_of_fp8": W_FP8_GIB / 2,
        "scale_overhead_note": ("NVFP4 是 group-16 + E4M3 的二级 scale：每 16 个 4-bit 权重"
                                "配 1 个 8-bit scale = 每权重多 0.5 bit，即 4 -> 4.5 bit，"
                                "相对 FP8 是 4.5/8 = 0.5625"),
        "weights_nvfp4_gib_with_scales": W_FP8_GIB * 4.5 / 8,
        "vs_bf16": None, "vs_fp8": None,
        "D2_quality_cost_of_quantising_the_scale": d2,
    }
    mem["vs_bf16"] = mem["weights_nvfp4_gib_with_scales"] / W_BF16_GIB
    mem["vs_fp8"] = mem["weights_nvfp4_gib_with_scales"] / W_FP8_GIB

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "THIS_IS_A_PROJECTION_NOT_A_MEASUREMENT": (
            "NVFP4 has no hardware path on Ada (sm_89). Fake-quant latency on a 4090 "
            "carries no speedup semantics and is not reported. What follows substitutes an "
            "ASSUMED operator-level ratio into THIS PROJECT'S MEASURED operator mix."),
        "measured_inputs": {
            "gemm_share_of_denoise": p,
            "denoise_share_of_e2e": shares,
            "fp8_bare_gemm_speedup_E0": g_fp8,
            "fp8_measured_denoise_speedup_E3": a_fp8,
            "fp8_zero_overhead_ceiling": ceil_fp8,
            "fp8_realisation_rate": realise,
        },
        "three_ceilings": {
            "fp8_zero_overhead": ceil_fp8,
            "absolute_any_gemm_only_method": ceil_abs,
            "headroom_left_for_ANY_better_format": ceil_abs / ceil_fp8,
            "reading": ("FP8 already captured 97.9% of ITS OWN ceiling. The room a better "
                        "weight format can still address is the gap between the two "
                        "ceilings, not the gap to 1.107."),
        },
        "assumed_inputs": {
            "nvfp4_over_fp8_gemm_throughput": "1.5 / 2.0 / 3.0 (Blackwell nominal is ~2x)",
            "nvfp4_realisation_rate": ("taken equal to FP8's 0.9794 -- OPTIMISTIC. NVFP4 "
                                       "uses group-16 scales, so a GEMM reads 2048x more "
                                       "scale values than per-tensor FP8; the realisation "
                                       "rate can only be lower."),
        },
        "projection": rows,
        "memory": mem,
        "verdict": None,
    }
    best = max(rows, key=lambda r: r["denoise_projected_at_fp8_realisation_rate"])
    at2 = [r for r in rows if abs(r["nvfp4_over_fp8_gemm_ratio"] - 2.0) < 1e-9]
    at2 = at2[0] if at2 else best
    out["verdict"] = {
        "headline": (
            "Even at the nominal 2x NVFP4:FP8 GEMM ratio and an optimistic realisation rate, "
            "the projected denoise speedup is %.3fx against FP8's measured %.3fx -- a gain of "
            "%.1f%%, and the end-to-end gain (VAE fp32) is %.1f%%."
            % (at2["denoise_projected_at_fp8_realisation_rate"], a_fp8,
               (at2["gain_over_measured_fp8_denoise"] - 1) * 100,
               (at2["e2e_projected"]["vae_float32_projected"]
                / (1 / ((1 - shares["float32"]) + shares["float32"] / a_fp8)) - 1) * 100)),
        "why_so_small": ("GEMM is only %.1f%% of denoise and denoise is only %.1f%% of "
                         "end-to-end. The format axis is bounded by the operator mix, not "
                         "by the format." % (p * 100, shares["float32"] * 100)),
        "where_nvfp4_does_pay": ("memory: weights %.2f -> %.2f GiB including the group-16 "
                                 "E4M3 scales (%.0f%% of FP8, %.0f%% of BF16). That is a "
                                 "deployment benefit, and it is NOT a speed benefit."
                                 % (W_FP8_GIB, mem["weights_nvfp4_gib_with_scales"],
                                    mem["vs_fp8"] * 100, mem["vs_bf16"] * 100)),
        "this_is_the_paper_s_own_thesis_applied_forward": (
            "cause (1): a headline operator-level ratio does not transfer to the stage or "
            "the chain. We do not need a Blackwell to say that -- the operator mix already "
            "bounds it."),
    }

    path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)

    print("=== 三个上界（p = GEMM 占 denoise = %.4f）===" % p)
    print(f"  FP8 零开销上界            {ceil_fp8:.4f}x   （FP8 实测 {a_fp8:.4f}x = 其 {realise*100:.1f}%）")
    print(f"  绝对天花板 1/(1-p)        {ceil_abs:.4f}x   <- 任何只加速 GEMM 的手段")
    print(f"  留给更好格式的空间         {ceil_abs/ceil_fp8:.4f}x  ({(ceil_abs/ceil_fp8-1)*100:.1f}%)")
    print("\n=== NVFP4 预估（⚠️ 吞吐比是假设，占比是实测）===")
    print(f"{'NVFP4:FP8':>10s} {'裸GEMM':>9s} {'denoise上界':>11s} {'denoise预估':>11s} "
          f"{'vs FP8实测':>10s} {'e2e预估(fp32)':>13s}")
    base_e2e = 1 / ((1 - shares["float32"]) + shares["float32"] / a_fp8)
    for r in rows:
        print(f"{r['nvfp4_over_fp8_gemm_ratio']:10.1f}x {r['assumed_bare_gemm_speedup']:8.3f}x "
              f"{r['denoise_ceiling_zero_overhead']:10.4f}x "
              f"{r['denoise_projected_at_fp8_realisation_rate']:10.4f}x "
              f"{r['gain_over_measured_fp8_denoise']:9.4f}x "
              f"{r['e2e_projected']['vae_float32_projected']:12.4f}x")
    print(f"{'':10s} {'':9s} {'':11s} {'FP8 实测':>11s} {a_fp8:9.4f}x {base_e2e:12.4f}x")
    print("\n=== 显存（这一侧是真实收益）===")
    print(f"  BF16 {W_BF16_GIB:.2f} GiB -> FP8 {W_FP8_GIB:.2f} -> "
          f"NVFP4(含 group-16 E4M3 scale) {mem['weights_nvfp4_gib_with_scales']:.2f} GiB"
          f"   = FP8 的 {mem['vs_fp8']*100:.0f}%，BF16 的 {mem['vs_bf16']*100:.0f}%")
    print(f"\n{out['verdict']['headline']}")
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
