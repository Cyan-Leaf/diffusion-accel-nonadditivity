#!/usr/bin/env python3
"""e5_fullchain.py — 把全链路加速比的算式落盘（纯算术，不跑模型）

**为什么要有这个脚本**：全链路数字（「155.9× 的朴素乘积 vs 16.3× 的实测」）是本文
成因 (二) 的全部内容，但它**不是任何一次测量的直接输出** —— 它是把四个不同实验的
落盘数字组合起来的。中间稿里流传过 16.48× / 23.63×，**复算不出来**
（正确值是 16.30× / 23.26×）。

**一个没有算式、只有结论的承重数字，是下一次错误的入口。** 所以把组合写成代码。

组合规则（每一项都注明来源文件）：
- text encode、VAE decode          <- results/E1/stage_profile.json（与配置无关）
- 50 步基线的 denoise              <- results/E1/stage_profile.json
- 4 步各配置的 denoise             <- results/E5/ladder.json（每 arm 独立进程实测）
- 朴素乘积的三个因子                <- E1 蒸馏轴 / E0 加权 GEMM / E4 稀疏 kernel

⚠️ **VAE dtype 必须成对**：50 步基线与 4 步配置要用同一个 VAE dtype，
否则会把 VAE 的 1.98× 混进链路加速比里。本脚本两种 dtype 各算一套。

⚠️ **50 步基线有两个候选，必须说明选了哪个（通则 8）**：

| 候选 | denoise | 是什么 |
|---|---|---|
| **E1 的 50 步**（主用） | **240.82 s** | **官方 Wan2.1-T2V-1.3B 权重**，UniPC + 外部 CFG，**eager** —— 「今天用户实际跑的那条流水线」 |
| ladder 的 `50step/bf16_dense` | 238.84 s | **蒸馏权重**跑 50 步，compiled —— 它只为检验 I 是否随步数变化而存在，**不是蒸馏基线** |

**选 E1 那个**，理由：全链路要回答「不做蒸馏时这条流水线要多久」，
而 ladder 的 50 步臂用的是蒸馏后的权重，**拿它当分母会把蒸馏轴的收益抹掉一部分**。
（两者差 0.8%：100 次前向的耗时几乎不依赖权重内容，所以 238.84 可作为
「若把基线也 compile」的敏感性参照 —— 本脚本把这一行也算出来。）

用法：
    python3 benchmarks/e5_fullchain.py --out results/E5/fullchain.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

REPO = os.environ.get("GENMODEL_ACCEL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# E0 的加权 FP8/BF16 GEMM 比值（results/E0/gemm_bench_torch211.json 的加权汇总）
FP8_GEMM_SPEEDUP = 1.8950877155587238


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/E5/fullchain.json")
    args = ap.parse_args()

    E1 = json.load(open(os.path.join(REPO, "results/E1/stage_profile.json")))
    L = json.load(open(os.path.join(REPO, "results/E5/ladder.json")))
    K = json.load(open(os.path.join(REPO, "results/E4/svg_latency.json")))["kernel"]

    dist = E1["distill_axis"]["denoise_only_speedup"]
    sparse_kernel = K["kernel_speedup_svg_half_temporal"]
    naive = dist * FP8_GEMM_SPEEDUP * sparse_kernel

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pure_arithmetic": "no model is run here; this composes numbers already on disk",
        "naive_product": {
            "factors": {
                "step_distillation_denoise_stage": dist,
                "fp8_bare_gemm_operator": FP8_GEMM_SPEEDUP,
                "sparse_attention_kernel_operator": sparse_kernel,
            },
            "scopes": ["stage (denoise)", "operator (bare GEMM)",
                       "operator (attention kernel)"],
            "product": naive,
            "why_it_is_wrong": ("the three factors are measured at three different scopes; "
                                "multiplying them bounds nothing"),
        },
        "chains": {},
    }

    for vae_dtype in ["float32", "bfloat16"]:
        base = E1["e2e"][f"50step/vae_{vae_dtype}"]
        four = E1["e2e"][f"4step/vae_{vae_dtype}"]
        text_s, vae_s = four["text_s"], four["vae_s"]
        rows = {}
        for arm, r in L["rows"]["4step"].items():
            den = r["denoise_s"]
            total = text_s + den + vae_s
            rows[arm] = {"text_s": text_s, "denoise_s": den, "vae_s": vae_s,
                         "total_s": total,
                         "fullchain_speedup_vs_50step": base["total_s"] / total}
        best = max(rows, key=lambda a: rows[a]["fullchain_speedup_vs_50step"])
        # 敏感性：若把 50 步基线也 compile（用 ladder 的 50step/bf16_dense 的 denoise）
        compiled_base = (L["rows"]["50step"]["bf16_dense"]["denoise_s"]
                         + base["text_s"] + base["vae_s"])
        sens = {
            "compiled_50step_baseline_total_s": compiled_base,
            "note": ("ladder's 50step/bf16_dense runs the DISTILLED weights for 50 steps "
                     "under compile; timing of 100 forwards barely depends on which weights, "
                     "so this serves as a 'what if the baseline were compiled too' check. It "
                     "is NOT used as the headline denominator -- see module docstring."),
            "fullchain_speedup_if_baseline_compiled":
                compiled_base / rows[best]["total_s"],
        }
        out["chains"][f"vae_{vae_dtype}"] = {
            "baseline_50step": {"total_s": base["total_s"], "denoise_s": base["denoise_s"],
                                "note": "official weights, UniPC + CFG, 100 forwards"},
            "arms": rows,
            "best_arm": best,
            "best_fullchain_speedup": rows[best]["fullchain_speedup_vs_50step"],
            "fraction_of_naive_product": rows[best]["fullchain_speedup_vs_50step"] / naive,
            "baseline_sensitivity": sens,
        }

    # ---- T11 §1：把 155.86 -> 42.86 -> 16.30 逐项拆到零残差 -------------
    #
    # ⚠️ 残差恒等于零，而且这**不是一个检验**。
    #    stage = den50 / d4_best 可以电报式地拆成
    #        (den50 / d4_eager) x (d4_eager / d4_comp) x (d4_comp / d4_best)
    #      =        D           x      compile        x          c
    #    而 c = a * b * I **按 I 的定义**恒成立。所以无论把某一项叫成什么、
    #    归给哪个成因，残差都是零。
    #    → **闭合不构成正确性的证据；本分解的内容在于每一项的归因，不在于它闭合。**
    #    （这正是通则 6 的同类：先想清楚读出量在零假设下取什么值 —— 这里恒为 0。）
    den50 = E1["denoise_50step"]["denoise_wall_s"]
    d4_eager = E1["e2e"]["4step/vae_float32"]["denoise_s"]
    d4_comp = L["rows"]["4step"]["bf16_dense"]["denoise_s"]
    d4_best = L["rows"]["4step"]["fp8_sparse"]["denoise_s"]
    H = L["H2"]["4step"]
    a, b, I = H["a_fp8_only"], H["b_sparse_only"], H["interaction_I"]
    compile_gain = d4_eager / d4_comp
    stage = den50 / d4_best
    fc32 = out["chains"]["vae_float32"]["best_fullchain_speedup"]

    steps = [
        {"n": 1, "what": "FP8: bare GEMM %.3fx -> denoise %.4fx" % (FP8_GEMM_SPEEDUP, a),
         "factor": a / FP8_GEMM_SPEEDUP, "direction": "down",
         "attribution": "cause (2) implementation overhead (naive W8A8 is 0.831x, a NEGATIVE "
                        "return) + cause (1) scope (GEMM is 20.4% of denoise)"},
        {"n": 2, "what": "sparse: mask kernel %.3fx -> denoise %.4fx" % (sparse_kernel, b),
         "factor": b / sparse_kernel, "direction": "down",
         "attribution": "cause (1) scope: attention is 52.5% of denoise"},
        {"n": 3, "what": "torch.compile gain already present in the BF16 baseline",
         "factor": compile_gain, "direction": "up",
         "attribution": "NOT one of the three causes -- a calibration term. The naive "
                        "product's distillation factor was measured against EAGER 4-step "
                        "denoise (9.592 s) while the ladder arms are COMPILED (9.465 s)."},
        {"n": 4, "what": "super-multiplicativity  I = c/(a*b)",
         "factor": I, "direction": "up",
         "attribution": "THE REFUTATION -- the only factor that adds back. The two axes are "
                        "exactly disjoint, so composing them beats the product of their "
                        "individual speedups."},
    ]
    cum = naive
    for st in steps:
        cum *= st["factor"]
        st["cumulative"] = cum
    residual = cum / stage - 1.0

    out["waterfall"] = {
        "levels": [
            {"level": "naive product", "value": naive,
             "how": "25.11 (stage) x 1.895 (operator) x 3.28 (operator)"},
            {"level": "measured denoise stage", "value": stage,
             "how": f"{den50:.3f} s (50-step denoise, official weights, eager) / "
                    f"{d4_best:.3f} s (4-step FP8+sparse denoise, compiled)",
             "loss_vs_previous": naive / stage},
            {"level": "measured full chain", "value": fc32,
             "how": "see chains.vae_float32",
             "loss_vs_previous": stage / fc32,
             "attribution": "structural fact: VAE decode + text encode lie outside every axis"},
        ],
        "itemised_naive_to_stage": steps,
        "residual_after_four_items": residual,
        "residual_is_an_identity_not_a_check": (
            "stage = (den50/d4_eager) x (d4_eager/d4_comp) x (d4_comp/d4_best) telescopes, "
            "and the last factor equals a*b*I BY THE DEFINITION of I. The residual is "
            "therefore identically zero no matter how each factor is labelled or attributed. "
            "Closure is NOT evidence that the attribution is right."),
        "distillation_passes_through_untouched": {
            "factor": dist,
            "forward_count_ratio": 25.0,
            "deviation_pct": (dist / 25.0 - 1) * 100,
            "why_it_matters": ("the distillation factor appears in the naive product AND in "
                               "the measured stage speedup with no loss term between them. It "
                               "is the only one of the three factors measured at the scope it "
                               "is claimed at (a stage), and it is the only one that is "
                               "delivered in full. Cause (1) stated positively: a speedup is "
                               "only realised at the level at which it was measured."),
        },
        "vae_step": {"n": 5, "what": "VAE decode + text encode outside every axis",
                     "factor": fc32 / stage, "direction": "down",
                     "cumulative": fc32},
        "per_axis_path": {
            "fp8": {"headline_bare_gemm": FP8_GEMM_SPEEDUP,
                    "after_scope_denoise": a, "denoise_ceiling": 1.107,
                    "note": "no separate 'after overhead' rung exists on this axis: the naive "
                            "implementation is 0.831x, i.e. negative return"},
            "sparse": {"headline_bare_kernel": K["kernel_speedup_flex_vs_dense"],
                       "after_implementation_overhead": sparse_kernel,
                       "after_scope_denoise": b,
                       "note": "4.13 -> 3.28 is the token reordering (cause 2); "
                               "3.28 -> 1.489 is attention's share of denoise (cause 1)"},
        },
        "which_sparse_factor_in_the_naive_product": {
            "chosen": sparse_kernel,
            "alternative": K["kernel_speedup_flex_vs_dense"],
            "why_chosen": ("3.28x is the kernel speedup SVG actually delivers once the token "
                           "reordering it requires is included. Using 4.13x would count "
                           "cause (2) twice: once inside the naive product and again as an "
                           "explanation of the shortfall."),
        },
    }
    alt = dist * FP8_GEMM_SPEEDUP * K["kernel_speedup_flex_vs_dense"]
    out["waterfall"]["which_sparse_factor_in_the_naive_product"]["if_alternative"] = {
        "naive_product": alt, "fraction_of_naive_product": fc32 / alt}

    out["superseded"] = {
        "values": "16.48x / 23.63x",
        "status": "do NOT use -- they do not reproduce from the on-disk numbers",
        "correct": {k: round(v["best_fullchain_speedup"], 3)
                    for k, v in out["chains"].items()},
        "note": ("these appeared in intermediate drafts without a recorded composition. "
                 "That is precisely why this script exists."),
    }

    p = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=2)

    print(f"朴素乘积 = {dist:.4f} (阶段) x {FP8_GEMM_SPEEDUP:.4f} (算子) x "
          f"{sparse_kernel:.4f} (算子) = {naive:.2f}x")
    for dt, c in out["chains"].items():
        print(f"\n--- {dt} (50 步基线 {c['baseline_50step']['total_s']:.2f} s) ---")
        for arm, r in c["arms"].items():
            print(f"   {arm:14s} {r['text_s']:.3f} + {r['denoise_s']:6.3f} + "
                  f"{r['vae_s']:.3f} = {r['total_s']:7.3f} s  ->  "
                  f"{r['fullchain_speedup_vs_50step']:6.2f}x")
        print(f"   最好 = {c['best_arm']} {c['best_fullchain_speedup']:.2f}x "
              f"= 朴素乘积的 {c['fraction_of_naive_product']*100:.1f}%")
    w = out["waterfall"]
    print("\n=== T11 §1  itemised waterfall (residual is an IDENTITY, not a check) ===")
    print(f"    {'naive product':44s} {naive:9.3f}x")
    for st in w["itemised_naive_to_stage"]:
        sign = "x" if st["direction"] == "up" else "/"
        f = st["factor"] if st["direction"] == "up" else 1 / st["factor"]
        print(f"  {st['n']}. {st['what']:42s} {sign}{f:7.4f}  -> {st['cumulative']:9.3f}x")
    print(f"    {'= measured denoise stage':44s} {stage:9.3f}x   "
          f"residual {w['residual_after_four_items']*100:+.5f}%")
    v = w["vae_step"]
    print(f"  {v['n']}. {v['what']:42s} /{1/v['factor']:7.4f}  -> {v['cumulative']:9.3f}x")
    d = w["distillation_passes_through_untouched"]
    print(f"\n    distillation {d['factor']:.4f}x passes through with NO loss term "
          f"({d['deviation_pct']:+.2f}% vs the 100/4 forward-count ratio)")
    print(f"\nwritten: {p}")


if __name__ == "__main__":
    main()
