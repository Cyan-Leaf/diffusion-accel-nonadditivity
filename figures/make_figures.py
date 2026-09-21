#!/usr/bin/env python3
"""make_figures.py — 从落盘 json 生成报告用的四张图（T10 §4）

**没有新测量**：四张图全部读 `results/` 下已有的 json。

| 图 | 内容 | 数据源 |
|---|---|---|
| F1 | 50 步 vs 4 步的阶段占比（denoise 96.1% → 49.6%，VAE 3.9% → 50.1%） | `E1/stage_profile.json` |
| F2 | 访存轴四点折线（0.831 → 0.999 → 1.043 → 1.052 → 1.084，虚线标上界 1.107） | `E3/fusion_latency.json` |
| **F3** | **瀑布：155.86 → 42.86 → 16.30，每段标归因（全文论点，一张图讲完）** | `E5/fullchain.json` |
| F4 | 指数位数 → scale 粒度收益（7 个格式，标 e=3 饱和点） | `E7/scale_granularity_formats.json` |

⚠️ **图注全部用英文**：本机 matplotlib 没有任何中文字体
（`font_manager` 里 Hei / Song / Noto CJK 一个都没有），
中文标签会渲染成方框。**与其让图上出现方框，不如统一英文** ——
报告正文是中文，图注英文，这在国内的技术报告里是常见且可接受的。

⚠️ **F2 的对照组必须画在图上**：纵轴是「相对 compiled BF16 的 denoise 加速」，
不是相对 eager。这是通则 5 —— 对照组配置要出现在图里，不能只在正文。

用法：
    <PY> figures/make_figures.py --out figures
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = os.environ.get("GENMODEL_ACCEL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})

C_DEN = "#3b6fb6"
C_VAE = "#d1603d"
C_TXT = "#8a8a8a"
C_OK = "#2e7d4f"
C_WARN = "#b03a2e"


def load(p):
    return json.load(open(os.path.join(REPO, p)))


# ---------------------------------------------------------------- F1
def fig1(out):
    E1 = load("results/E1/stage_profile.json")
    rows = [("50 steps\n(official weights,\nUniPC + CFG, 100 fwd)", "50step/vae_float32"),
            ("4 steps\n(distilled,\nno CFG, 4 fwd)", "4step/vae_float32")]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.6),
                                   gridspec_kw={"width_ratios": [1, 1]})

    # 左：绝对秒数（对数，因为差 13 倍）
    for i, (lbl, k) in enumerate(rows):
        e = E1["e2e"][k]
        b = 0.0
        for val, c, nm in [(e["text_s"], C_TXT, "text encode"),
                           (e["denoise_s"], C_DEN, "denoise"),
                           (e["vae_s"], C_VAE, "VAE decode")]:
            axL.bar(i, val, bottom=b, color=c, width=0.55,
                    label=nm if i == 0 else None, edgecolor="white", linewidth=0.6)
            b += val
        axL.text(i, b * 1.04, f"{b:.1f} s", ha="center", fontsize=10, weight="bold")
    axL.set_xticks(range(len(rows)))
    axL.set_xticklabels([r[0] for r in rows], fontsize=8)
    axL.set_ylabel("wall clock (s)")
    axL.set_title("(a) absolute cost", fontsize=10)
    axL.legend(fontsize=8, loc="upper right", frameon=False)
    axL.set_ylim(0, 290)

    # 右：占比（论点所在）
    for i, (lbl, k) in enumerate(rows):
        e = E1["e2e"][k]["share"]
        b = 0.0
        for key, c in [("text", C_TXT), ("denoise", C_DEN), ("vae", C_VAE)]:
            v = e[key] * 100
            axR.bar(i, v, bottom=b, color=c, width=0.55, edgecolor="white", linewidth=0.6)
            if v > 4:
                axR.text(i, b + v / 2, f"{v:.1f}%", ha="center", va="center",
                         color="white", fontsize=10, weight="bold")
            b += v
    axR.set_xticks(range(len(rows)))
    axR.set_xticklabels([r[0] for r in rows], fontsize=8)
    axR.set_ylabel("share of end-to-end (%)")
    axR.set_title("(b) share -- the addressable surface of all four axes\n"
                  "shrinks 96.1% -> 49.6%", fontsize=10)
    axR.set_ylim(0, 108)
    axR.annotate("", xy=(1, 49.6), xytext=(0, 96.1),
                 arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.6))
    axR.text(0.5, 78, "all four axes\nact only here", ha="center", fontsize=8,
             color=C_WARN, weight="bold")
    fig.suptitle("F1  Bottleneck migration: distillation moves the cost outside every axis",
                 fontsize=11, weight="bold", y=1.04)
    fig.savefig(os.path.join(out, "F1_bottleneck_migration.png"))
    plt.close(fig)
    return "F1_bottleneck_migration.png"


# ---------------------------------------------------------------- F2
def fig2(out):
    F = load("results/E3/fusion_latency.json")["rows"]
    order = [("naive\nW8A8", "fp8_naive"),
             ("+ quant folded\ninto compile", "fp8_compiled"),
             ("+ F1\nbias epilogue", "fp8_F1_bias"),
             ("+ F2\nshared q/k/v quant", "fp8_F1F2_shareq"),
             ("+ F3\nnorm/GELU epilogue", "fp8_F1F2F3_full")]
    y = [F[k]["speedup_vs_bf16_compiled"] for _, k in order]
    x = list(range(len(y)))
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.axhline(1.0, color="#999", lw=1.0, ls="-")
    ax.axhline(1.107, color=C_OK, lw=1.4, ls="--")
    ax.axhline(1.2568, color="#7a7a7a", lw=1.1, ls=":")
    ax.text(-0.05, 1.2600, "absolute ceiling 1.257x = 1/(1-p) -- ANY GEMM-only method",
            ha="left", va="bottom", color="#666", fontsize=8.2)
    ax.text(-0.05, 1.1095, "Amdahl ceiling 1.107x   (FP8's measured 1.895x GEMM, zero overhead)",
            ha="left", va="bottom", color=C_OK, fontsize=8.5, weight="bold")
    ax.text(-0.05, 1.003, "compiled BF16 baseline = 1.000x", ha="left", va="bottom",
            color="#666", fontsize=8.5)
    ax.plot(x, y, "-o", color=C_DEN, lw=1.8, ms=7, zorder=3)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.3f}x", (xi, yi), textcoords="offset points",
                    xytext=(0, 11 if yi > 1 else -17), ha="center",
                    fontsize=9.5, weight="bold",
                    color=C_WARN if yi < 1 else "black")
    ax.add_patch(Rectangle((x[-1] - 0.16, y[-1]), 0.32, 1.107 - y[-1],
                           facecolor=C_OK, alpha=0.18, zorder=1))
    ax.annotate("1.91 pp\nunrecovered", (x[-1], (y[-1] + 1.107) / 2),
                textcoords="offset points", xytext=(-52, -34), ha="center", va="center",
                fontsize=8.5, color=C_OK,
                arrowprops=dict(arrowstyle="-", color=C_OK, lw=0.8,
                                connectionstyle="arc3,rad=0.2"))
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in order], fontsize=8)
    ax.set_ylabel("denoise speedup  vs  COMPILED BF16")
    ax.set_ylim(0.78, 1.30)
    ax.set_title("F2  The memory-traffic axis: a naive W8A8 implementation is SLOWER than\n"
                 "BF16; three fusion levels recover 28.2 pp, all of it memory traffic",
                 fontsize=10.5, weight="bold")
    fig.savefig(os.path.join(out, "F2_fusion_ladder.png"))
    plt.close(fig)
    return "F2_fusion_ladder.png"


# ---------------------------------------------------------------- F3 (主图)
def fig3(out):
    """五段瀑布，八个横轴位置（三个端点柱 + 五个浮动段）。

    ⚠️ **对数轴上 x1.0435 必然是一根极矮的柱子**（|log 1.0435| 只有 |log 1.7475| 的 1/13）,
    所以「视觉上唯一向上的那根」不能靠尺寸做出来，只能靠**方向**：
    每一段上方画一个三角（▲ 向上 / ▼ 向下），第 4 项用饱和红 + 粗边 + 引注。
    **把一个 4% 的效应画得和 43% 的一样大才是撒谎。**"""
    W = load("results/E5/fullchain.json")["waterfall"]
    naive = W["levels"][0]["value"]
    stage = W["levels"][1]["value"]
    final = W["levels"][2]["value"]
    steps = W["itemised_naive_to_stage"] + [W["vae_step"]]

    # 八个位置：端点 / 五段 / 端点 / 端点
    X_NAIVE, X_STAGE, X_FINAL = 0, 5, 7
    xs = [1, 2, 3, 4, 6]                       # 五个浮动段
    cums = [naive] + [st["cumulative"] for st in steps]

    labels = ["naive\nproduct", "1. FP8\nGEMM 1.895x\n-> denoise",
              "2. sparse\nkernel 3.28x\n-> denoise", "3. compile in\nbaseline\n(calibration)",
              "4. SUPER-\nMULTIPLICATIVE\nI = c/(a*b)", "measured\nDENOISE\nSTAGE",
              "5. VAE + text\noutside\nevery axis", "measured\nFULL\nCHAIN"]

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.set_yscale("log")
    ax.set_ylim(7, 640)
    ax.set_yticks([10, 20, 50, 100, 200, 400])
    ax.set_yticklabels(["10x", "20x", "50x", "100x", "200x", "400x"])

    for x, v, c in [(X_NAIVE, naive, "#9e9e9e"), (X_STAGE, stage, C_DEN),
                    (X_FINAL, final, C_OK)]:
        ax.bar(x, v, color=c, width=0.56, edgecolor="white", linewidth=1, zorder=2)
        ax.text(x, v * 1.11, f"{v:.2f}x", ha="center", fontsize=12.5, weight="bold")

    for x, st, c_prev, c_now in zip(xs, steps, cums[:-1], cums[1:]):
        lo, hi = sorted((c_prev, c_now))
        up = st["direction"] == "up"
        if st["n"] == 4:
            col, ec, lw, z = "#c0392b", "#7b241c", 1.8, 4
        elif up:
            col, ec, lw, z = "#b0b0b0", "#7a7a7a", 1.0, 3
        else:
            col, ec, lw, z = C_WARN, "white", 0.8, 3
        ax.bar(x, hi - lo, bottom=lo, color=col, alpha=1.0 if st["n"] == 4 else
               (0.9 if up else 0.5), width=0.56, edgecolor=ec, linewidth=lw, zorder=z)
        # 方向三角：尺寸恒定，只表方向
        ax.plot(x, hi * 1.30, marker="^" if up else "v", ms=13,
                color="#c0392b" if st["n"] == 4 else ("#7a7a7a" if up else C_WARN),
                zorder=5)
        f = st["factor"] if up else 1 / st["factor"]
        ax.text(x, hi * 1.62, ("$\\times$" if up else "$\\div$") + f"{f:.4f}",
                ha="center", fontsize=11, weight="bold",
                color="#c0392b" if st["n"] == 4 else ("#666" if up else C_WARN))
        ax.text(x, lo * 0.84, f"{c_now:.2f}x", ha="center", va="top",
                fontsize=8.8, color="#444")

    # 连接虚线：8 个位置 -> 7 段，每段画在两者之间的那个累积水平上
    order = [X_NAIVE, xs[0], xs[1], xs[2], xs[3], X_STAGE, xs[4], X_FINAL]
    levels = [cums[0], cums[1], cums[2], cums[3], cums[4], cums[4], cums[5]]
    assert len(levels) == len(order) - 1
    for i, lv in enumerate(levels):
        ax.plot([order[i] + 0.28, order[i + 1] - 0.28], [lv] * 2, ls=":",
                color="#999", lw=1, zorder=1)

    ax.annotate("the ONLY factor that adds back.\n"
                "The two axes are exactly disjoint, so\n"
                "composing them BEATS the product of\n"
                "their individual speedups (I = 1.0435\n"
                "vs 1.0431 from the analytic model).\n"
                "It is small -- and it is the wrong SIGN\n"
                "for the explanation everyone assumes.",
                xy=(4.0, stage * 1.55), xytext=(4.72, 132),
                fontsize=8.3, color="#c0392b", linespacing=1.45, va="bottom",
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.5,
                                connectionstyle="arc3,rad=0.28"),
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#c0392b", lw=1.1))

    d = W["distillation_passes_through_untouched"]
    ax.text(0.015, 0.035,
            "the 25.11x distillation factor appears on BOTH ends with no loss term:\n"
            "it is the only factor measured at the scope it is claimed at (a stage),\n"
            f"and the only one delivered in full ({d['deviation_pct']:+.2f}% vs the 100/4 "
            "forward-count ratio)",
            transform=ax.transAxes, fontsize=8.2, color=C_OK, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.32", fc="#f2f8f4", ec=C_OK, lw=0.9))

    ax.set_xticks(range(8))
    ax.set_xticklabels(labels, fontsize=8.0)
    ax.set_xlim(-0.6, 7.6)
    ax.set_ylabel("speedup over the as-shipped\n50-step pipeline (log scale)")
    ax.set_title("F3  Where the shortfall goes, itemised:  155.86x  ->  42.86x  ->  16.30x\n"
                 "residual after the four items is +0.00000% -- but that is an IDENTITY, "
                 "not a check",
                 fontsize=10.5, weight="bold")
    fig.text(0.5, -0.055,
             "naive product = 25.11x (denoise STAGE)  x  1.895x (bare GEMM OPERATOR)  x  "
             "3.28x (attention-kernel OPERATOR)   --   three different scopes",
             ha="center", fontsize=8.5, color="#444")
    fig.savefig(os.path.join(out, "F3_waterfall.png"))
    plt.close(fig)
    return "F3_waterfall.png"


# ---------------------------------------------------------------- F4
def fig4(out):
    """直接读 `T6B_exponent_bit_curve.curve` —— 它自带 exp_bits / total_bits / gain_group16，
    不需要在这里重建任何映射，也不需要兜底常量（兜底常量会让图与 json 悄悄脱钩）。"""
    curve = load("results/E7/scale_granularity_formats.json")["T6B_exponent_bit_curve"]["curve"]
    pretty = {"int4": "INT4 (4 bit)", "int8": "INT8 (8 bit)", "nvfp4": "NVFP4 E2M1 (4 bit)",
              "fp6e2m3": "FP6 E2M3 (6 bit)", "fp6e3m2": "FP6 E3M2 (6 bit)",
              "fp8": "FP8 E4M3 (8 bit)", "fp8e5m2": "FP8 E5M2 (8 bit)"}
    fig, ax = plt.subplots(figsize=(7.8, 4.3))

    byb = {}
    for r in curve:
        byb.setdefault(r["exp_bits"], []).append(r["gain_group16"])
    bs = sorted(byb)
    ax.plot(bs, [sum(byb[b]) / len(byb[b]) for b in bs], "-", color="#c4c4c4", lw=7,
            zorder=1, solid_capstyle="round", label="mean per exponent-bit count")

    mk = {0: "s", 2: "o", 3: "^", 4: "D", 5: "v"}
    for r in sorted(curve, key=lambda r: (r["exp_bits"], -r["gain_group16"])):
        b, g = r["exp_bits"], r["gain_group16"]
        ax.plot(b, g, mk[b], ms=9, color=C_DEN if b <= 2 else C_WARN, zorder=3)
        if b >= 3:                      # 三个饱和点挤在同一高度 -> 标在下方并缩短
            short = {"fp6e3m2": "FP6\nE3M2\n6 bit", "fp8": "FP8\nE4M3\n8 bit",
                     "fp8e5m2": "FP8\nE5M2\n8 bit"}[r["format"]]
            ax.annotate(short, (b, g), textcoords="offset points", xytext=(0, -15),
                        ha="center", va="top", fontsize=8, color=C_WARN,
                        linespacing=1.25)
        else:
            ax.annotate(pretty.get(r["format"], r["format"]), (b, g),
                        textcoords="offset points", xytext=(11, 0), ha="left",
                        fontsize=8.5, va="center")

    ax.axvspan(2.5, 5.6, color=C_WARN, alpha=0.07, zorder=0)
    ax.axvline(3, color=C_WARN, ls="--", lw=1.3)
    sat = [r for r in curve if r["exp_bits"] >= 3]
    ax.text(3.10, 5.95,
            "saturation at 3 exponent bits:\n"
            + " / ".join(f"{r['gain_group16']:.3f}" for r in sorted(sat, key=lambda r: r["exp_bits"]))
            + "  (spread < 0.4%),\nwhile total width spans "
            + f"{min(r['total_bits'] for r in sat)}-{max(r['total_bits'] for r in sat)} bits and\n"
            + f"max representable spans {min(r['absmax'] for r in sat):.0f}"
            + f" -> {max(r['absmax'] for r in sat):.0f}",
            fontsize=8, color=C_WARN, va="top")
    # 同指数位、总位宽差一倍的两组对照（这是「决定因素是指数位」的直接证据）
    for b in (0, 2):
        gs = sorted(byb[b])
        ax.annotate("", xy=(b - 0.10, gs[0]), xytext=(b - 0.10, gs[-1]),
                    arrowprops=dict(arrowstyle="<->", color=C_DEN, lw=1.2))
        ax.annotate(f"2x the bit width,\nonly {gs[-1] - gs[0]:.2f} apart",
                    (b - 0.13, (gs[0] + gs[-1]) / 2), textcoords="offset points",
                    xytext=(-6, 0), ha="right", va="center", fontsize=7.5, color=C_DEN)

    ax.set_xticks(sorted(byb))
    ax.set_xlabel("number of exponent bits in the codebook")
    ax.set_ylabel("error improvement,\nper-tensor $\\rightarrow$ group-16 scale")
    ax.set_ylim(-0.15, 6.35)
    ax.set_xlim(-1.35, 6.25)
    ax.set_title("F4  The value of finer scale granularity is a property of the FORMAT,\n"
                 "and the deciding variable is exponent bits -- not total bit width",
                 fontsize=10.5, weight="bold")
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(os.path.join(out, "F4_exponent_bits.png"))
    plt.close(fig)
    return "F4_exponent_bits.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)
    for fn in (fig1, fig2, fig3, fig4):
        name = fn(out)
        print(f"  wrote {args.out}/{name}")
    print("\n⚠️ 图注为英文：本机 matplotlib 无任何中文字体，中文会渲染成方框。")


if __name__ == "__main__":
    main()
