#!/usr/bin/env python3
"""e1_amdahl.py — 用 E1 的实测占比 + E0 的实测 GEMM 上界，算各轴的 Amdahl 上界

这不是新实验，是对 E0/E1 已有实测数字的算术合成。存成产物是为了让报告里
每个「上界」都能指回一个可复算的 json，而不是手算的数。

核心问题：E0 测出 FP8 在裸 GEMM 上有 1.895×。**这个数在端到端能剩多少？**
E1 测出 4 步 DiT 里 GEMM 只占 20.2%，且 VAE 占端到端 50%。两个 Amdahl 串起来。

用法：
    python benchmarks/e1_amdahl.py --out results/E1/amdahl.json
"""
from __future__ import annotations

import argparse
import json
import os

REPO = os.environ.get("GENMODEL_ACCEL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 未实测的轴用文献值，**必须在报告里标成 projection**
LIT_SVG = 2.3   # CogVideoX-v1.5 2.28x / HunyuanVideo 2.33x；Wan2.1 上待 E4 自测


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="results/E1/stage_profile.json")
    ap.add_argument("--e0", default="results/E0/gemm_bench.json")
    ap.add_argument("--out", default="results/E1/amdahl.json")
    args = ap.parse_args()

    e1 = json.load(open(os.path.join(REPO, args.e1)))
    e0 = json.load(open(os.path.join(REPO, args.e0)))
    fp8_gemm = e0["aggregate_fp8_speedup_vs_bf16"]

    out = {"inputs": {
        "fp8_gemm_speedup_measured_E0": fp8_gemm,
        "svg_attention_speedup_LITERATURE_not_measured": LIT_SVG,
        "note": "SVG factor is literature (CogVideoX-v1.5 2.28x / HunyuanVideo 2.33x). "
                "Everything else on this page is measured on this machine. Any row that "
                "uses the SVG factor MUST be labelled a projection in the report.",
    }}

    for tag in ("4step", "50step"):
        key = f"denoise_{tag}"
        if key not in e1:
            continue
        sh = e1[key]["profile"]["buckets_share"]
        gemm = sh.get("gemm", 0.0)
        attn = sh.get("attention", 0.0)
        t_den = e1[key]["denoise_wall_s"]
        t_txt = (e1["text_encode"]["ms_single_prompt"] if tag == "4step"
                 else e1["text_encode"]["ms_cond_plus_uncond"]) / 1e3

        row = {"denoise_s": t_den, "text_s": t_txt,
               "share_gemm": gemm, "share_attention": attn,
               "share_other": 1.0 - gemm - attn}

        # 单轴：只加速 GEMM
        f_gemm = (1 - gemm) + gemm / fp8_gemm
        row["denoise_ceiling_fp8_only"] = 1.0 / f_gemm
        # 单轴：只加速 attention
        f_attn = (1 - attn) + attn / LIT_SVG
        row["denoise_ceiling_svg_only_PROJECTION"] = 1.0 / f_attn
        # 双轴
        f_both = (1 - gemm - attn) + gemm / fp8_gemm + attn / LIT_SVG
        row["denoise_ceiling_fp8_plus_svg_PROJECTION"] = 1.0 / f_both
        row["naive_product_fp8_x_svg"] = fp8_gemm * LIT_SVG

        row["vae"] = {}
        for vdt in ("float32", "bfloat16"):
            vk = f"{tag}/{vdt}"
            if vk not in e1.get("vae", {}):
                continue
            t_vae = e1["vae"][vk]["decode_s"]
            base = t_txt + t_den + t_vae
            r = {"vae_s": t_vae, "total_s_bf16_baseline": base,
                 "share_vae": t_vae / base, "share_denoise": t_den / base}
            for name, f in (("fp8_only", f_gemm),
                            ("svg_only_PROJECTION", f_attn),
                            ("fp8_plus_svg_PROJECTION", f_both)):
                tot = t_txt + t_den * f + t_vae
                r[f"e2e_ceiling_{name}"] = base / tot
                r[f"e2e_total_s_{name}"] = tot
            # 「收益被吃掉多少」：超出 1× 的部分保留了几成
            r["excess_retained_fp8_only"] = ((r["e2e_ceiling_fp8_only"] - 1)
                                             / (fp8_gemm - 1))
            r["excess_retained_fp8_plus_svg_PROJECTION"] = (
                (r["e2e_ceiling_fp8_plus_svg_PROJECTION"] - 1)
                / (fp8_gemm * LIT_SVG - 1))
            row["vae"][vdt] = r
        out[tag] = row

    # 三轴全链路（含蒸馏）：以 50 步 BF16 + FP32 VAE 为分母
    if "50step" in out and "4step" in out:
        for vdt in ("float32", "bfloat16"):
            if vdt not in out["50step"]["vae"] or vdt not in out["4step"]["vae"]:
                continue
            base50 = out["50step"]["vae"][vdt]["total_s_bf16_baseline"]
            f_both4 = ((1 - out["4step"]["share_gemm"] - out["4step"]["share_attention"])
                       + out["4step"]["share_gemm"] / fp8_gemm
                       + out["4step"]["share_attention"] / LIT_SVG)
            t4 = (out["4step"]["text_s"] + out["4step"]["denoise_s"] * f_both4
                  + out["4step"]["vae"][vdt]["vae_s"])
            naive = (out["50step"]["denoise_s"] / out["4step"]["denoise_s"]) \
                * fp8_gemm * LIT_SVG
            out.setdefault("full_chain_PROJECTION", {})[vdt] = {
                "baseline_50step_bf16_vaefp32_s": base50,
                "target_4step_fp8_svg_s": t4,
                "e2e_speedup": base50 / t4,
                "naive_product": naive,
                "naive_product_breakdown": {
                    "distill_measured": out["50step"]["denoise_s"] / out["4step"]["denoise_s"],
                    "fp8_gemm_measured": fp8_gemm,
                    "svg_literature": LIT_SVG},
                "fraction_of_naive_product": (base50 / t4) / naive,
            }

    os.makedirs(os.path.dirname(os.path.join(REPO, args.out)), exist_ok=True)
    with open(os.path.join(REPO, args.out), "w") as f:
        json.dump(out, f, indent=2)

    print(f"FP8 GEMM (E0, measured):            {fp8_gemm:.3f}x")
    for tag in ("4step", "50step"):
        if tag not in out:
            continue
        r = out[tag]
        print(f"\n[{tag}]  GEMM share {r['share_gemm']*100:.1f}%  "
              f"attention share {r['share_attention']*100:.1f}%")
        print(f"  denoise ceiling, FP8 only            {r['denoise_ceiling_fp8_only']:.3f}x")
        print(f"  denoise ceiling, +SVG (projection)   "
              f"{r['denoise_ceiling_fp8_plus_svg_PROJECTION']:.3f}x  "
              f"(naive product {r['naive_product_fp8_x_svg']:.2f}x)")
        for vdt, v in r["vae"].items():
            print(f"  e2e (VAE {vdt:9s}) VAE={v['share_vae']*100:4.1f}%  "
                  f"FP8 only {v['e2e_ceiling_fp8_only']:.3f}x "
                  f"(retains {v['excess_retained_fp8_only']*100:.1f}% of the excess)  "
                  f"+SVG {v['e2e_ceiling_fp8_plus_svg_PROJECTION']:.3f}x")
    if "full_chain_PROJECTION" in out:
        print()
        for vdt, v in out["full_chain_PROJECTION"].items():
            print(f"full chain (VAE {vdt:9s}): {v['e2e_speedup']:.2f}x  vs naive "
                  f"{v['naive_product']:.1f}x  = {v['fraction_of_naive_product']*100:.1f}%")
    print(f"\nwritten: {os.path.join(REPO, args.out)}")


if __name__ == "__main__":
    main()
