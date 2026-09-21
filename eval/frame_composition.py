#!/usr/bin/env python3
"""frame_composition.py — T8：稀疏臂的样本身份由谁决定 + 一个失败的构图代理量

⚠️ **先读这段**：本脚本里的 `center_border_hf_ratio` 是我为「构图坍缩到特写」设计的
代理量，**它失败了**（见文末 verdict）。脚本保留它与它的否证，不保留它的结论。
真正站得住的是第二个量 `cross_arm_video_psnr`。

**这不是新实验**：只读已生成的 mp4（与 §1.4 同一批），不生成任何视频。

## 为什么要算这个

T8 §1.4 要求把 VBench imaging 的逐视频分摊开。摊开后出现了一件 handoff 没预期的事：

| prompt | bf16 | bf16_sparse | delta |
|---|---|---|---|
| p0 | 0.7632 | 0.7517 | −0.0115 |
| p1 | 0.7319 | **0.5948** | **−0.1371** |
| p2 | 0.7145 | **0.7617** | **+0.0472（稀疏更高）** |

均值 gap（−0.0338）**几乎全部来自 p1**，而 p2 上稀疏臂**反超**。
抽帧（`results/frames/compare_p{0,1,2}_midframe.png`）给出了解释：
**三条 prompt 上，稀疏无预热臂都坍缩成同一种构图 —— 人物特写**
（主体占满画面、背景浅景深虚化），无论 prompt 要求的是雪地广角还是集市场景。

这在机制上讲得通：SVG 的两个 mask 是「首帧 sink + 带状空间邻域」，
第一步（σ=1.0）丢掉长程依赖后，模型无法建立全画幅的全局布局，
**退化到一个居中主体填满画面**。

**关键后果**：同一个失效模式在 p1 上被 MUSIQ 记为大幅扣分、在 p2 上记为**加分**。
→ **无参照 IQA 在稀疏臂上不是一个有效的裁判**，因为它评的是「这一帧好不好看」，
而稀疏改变的是「生成了哪一个视频」。这与 T8 §2 指出的高频能量比混淆**是同一个混淆**，
只是 §2 只点到了高频比，**它同样命中 VBench imaging**。

## 怎么操作化「特写」

`center_border_hf_ratio` = 中心 50% 区域的高频能量密度 / 边缘区域的高频能量密度。
- 特写（主体居中锐利 + 背景虚化）→ **比值高**
- 广角（细节铺满画面）→ **比值接近 1**

高频用 3×3 Laplacian 的能量，在灰度上算，逐帧再对帧取中位数。

### ⚠️ 这个代理量 REFUTED（实测后写回）

    arm                  p0     p1     p2   mean
    bf16               1.28   2.43   1.49   1.74
    bf16_sparse        1.18   1.59   2.60   1.79      <-- 与 dense 无法区分

**不但均值几乎相同（1.79 vs 1.74），逐 prompt 的方向还相反**（p1 稀疏更低、p2 更高）。
原因是这个代理量把**取景**（主体占画面多少）和**背景虚化程度**混在了一起：
p1 的稀疏臂虽是特写，背景却是一片高频的彩旗/风车，比值反而被压低。

→ **「构图坍缩到特写」这个观察在抽帧上是清楚的，但本代理量不能支撑它。
保留为定性观察 + 帧图，不给数字。**（通则 3：没测准的量不给点估。）

## ⚠️ T9 §2：「支配」不是「不相交」，而且输出域里「不相交」没有定义

T8 写过「这是 H2 两条轴不相交在像素域的对应观察」。**那一步不成立，已删除。**
「A 的扰动比 B 大」是**支配**关系，不是独立性 —— 两个强相互作用的因素也可以一大一小。

负责人提出输出域的正确形式应是比较**同一份 FP8 在两种条件下的扰动**：
`d(dense, fp8_dense)` vs `d(sparse, fp8_sparse)`，相等则不相交。**这个方向对，但有两重障碍。**

### 障碍一：本项目没有配置对齐的 `fp8_dense` 视频（已查）

| 臂 | FP8 代码路径 |
|---|---|
| 质量表的 `rtn_tensor` / `gptq*` | `wan_fp8.convert_fakequant` —— **伪量化** |
| `fp8_sparse` | `fusion.wan_fused.convert_fused` —— **真 `_scaled_mm` + E3 全融合** |

**算法相同（RTN per-tensor），代码路径不同。** 而 T4-A 已经量过这条路径差：
伪量化对 FP64 参照是 1e-6，`_scaled_mm` 是 2.5e-3~3.3e-3；
**两个数学等价的真 kernel 配置之间的 latent L2 是 0.1204**，
而量化臂自身对 bf16 的偏离只有 0.31–0.39 —— **混淆量约为待测效应的三分之一，不可忽略。**
→ **严格检验未执行。** 补它需要一个配对的 `fp8_dense` 臂（3 条视频），本轮不跑模型。

### 障碍二（更根本）：即使配置对齐，该检验也不能证明「耦合」

延迟域的不相交是关于**资源**的陈述，建立在一个**线性可加**的量（时间）上。
输出域没有对应的陈述可做，**因为采样器是非线性的**：
同样大小的扰动施加在一条**已经偏离**的轨迹上（稀疏臂的 latent L2 = 1.13，已超过信号本身），
其效果本来就与施加在原轨迹上不同。这是 ODE 求解器的性质，**不是两条轴在争什么**。

→ **正确的结论比原先那句强**：
> 「不相交」不能外推到输出域 —— **不是因为我们没测，而是因为输出域没有一个可加的基底，
> 让「不相交」这个概念有定义。** 延迟域之所以能问这个问题，是因为时间可加。

**下面的数字因此只作为「输出域的复合不保幅」的例示，不作为耦合的证据。**

## 支配关系：样本身份由哪条轴决定

`cross_arm_video_psnr` 逐 prompt 算各臂之间的全片平均像素 PSNR。**3/3 prompt 一致**：

    sparse~fp8_sparse (10.10-12.90 dB)  >  dense~sparse (7.47-8.74)  且  > dense~fp8_sparse

→ **两个稀疏臂彼此的相似度，高于它们各自与 dense 的相似度。**
即**稀疏这条轴决定了生成的是哪一个样本，叠在其上的 FP8 只是一个更小的扰动。**
这是 H2「两条轴不相交」在**像素域**的对应观察（H2 是在延迟上测的）。

同样 3/3 一致：`dense~warm1`（11.70-15.47）稳定落在 `dense~gptq_cd`（15.14-20.99）
与 `dense~sparse`（7.47-8.74）之间 → **一步 dense 预热找回了大部分距离。**

⚠️ **n=3，描述性观察。** 但「3/3 同向」比「均值差」可靠，因为设计是配对的。

用法：
    <PY> eval/frame_composition.py --out results/E9/frame_composition.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import av
import numpy as np

REPO = os.environ.get("GENMODEL_ACCEL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = ["bf16", "gptq_cd", "gptq_clip_cd", "rtn_tensor",
        "bf16_sparse", "bf16_sparse_warm1", "fp8_sparse"]

# T9 §2：同一份 FP8（算法同为 RTN per-tensor）在两种条件下的扰动幅度。
# ⚠️ 两者代码路径不同（伪量化 vs 真 kernel + E3 融合），见文件头「障碍一」。
FP8_PERTURBATION = {
    "fp8_on_dense__CONFOUNDED_pseudoquant_path": ("bf16", "rtn_tensor"),
    "fp8_on_sparse__real_kernel_path": ("bf16_sparse", "fp8_sparse"),
}
PROMPTS = ["p0", "p1", "p2"]


def read_frames(path):
    c = av.open(path)
    fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    return fr


def lap_energy(gray):
    """3x3 Laplacian 的能量图（不引 scipy，直接切片做卷积）。"""
    g = gray.astype(np.float64)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    return lap ** 2


def center_border_ratio(rgb):
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    e = lap_energy(gray)
    H, W = e.shape
    h0, h1 = H // 4, H - H // 4          # 中心 50%（面积 25%）
    w0, w1 = W // 4, W - W // 4
    center = e[h0:h1, w0:w1]
    mask = np.ones_like(e, dtype=bool)
    mask[h0:h1, w0:w1] = False
    border = e[mask]
    cd = center.mean()
    bd = border.mean()
    return float(cd / bd) if bd > 0 else float("nan")


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    m = ((a - b) ** 2).mean()
    return 99.0 if m == 0 else float(10 * np.log10(255.0 ** 2 / m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="results/E9/vbench_work")
    ap.add_argument("--out", default="results/E9/frame_composition.json")
    args = ap.parse_args()

    vd = os.path.join(REPO, args.videos)
    cache, comp = {}, {}
    for arm in ARMS:
        comp[arm] = {}
        for p in PROMPTS:
            f = os.path.join(vd, arm, f"{arm}_{p}.mp4")
            if not os.path.exists(f):
                continue
            fr = read_frames(f)
            cache[(arm, p)] = fr
            r = [center_border_ratio(x) for x in fr]
            comp[arm][p] = {"median": float(np.median(r)),
                            "mean": float(np.mean(r)), "n_frames": len(fr)}

    # 跨臂视频距离：稀疏两臂彼此 vs 各自对 dense
    dist = {}
    for p in PROMPTS:
        if ("bf16", p) not in cache:
            continue
        d = cache[("bf16", p)]
        pairs = {"dense~rtn_tensor": ("bf16", "rtn_tensor"),
                 "dense~gptq_cd": ("bf16", "gptq_cd"),
                 "dense~sparse": ("bf16", "bf16_sparse"),
                 "dense~fp8_sparse": ("bf16", "fp8_sparse"),
                 "dense~warm1": ("bf16", "bf16_sparse_warm1"),
                 "sparse~fp8_sparse": ("bf16_sparse", "fp8_sparse")}
        row = {}
        for name, (a, b) in pairs.items():
            if (a, p) not in cache or (b, p) not in cache:
                continue
            A, B = cache[(a, p)], cache[(b, p)]
            n = min(len(A), len(B))
            row[name] = float(np.mean([psnr(A[i], B[i]) for i in range(n)]))
        dist[p] = row
        del d

    # ---- T9 §2 的那两个量（带混淆说明，见文件头）-------------------------
    perturb = {}
    for name, (a, b) in FP8_PERTURBATION.items():
        vals = {}
        for p in PROMPTS:
            if (a, p) in cache and (b, p) in cache:
                A, B = cache[(a, p)], cache[(b, p)]
                n = min(len(A), len(B))
                vals[p] = float(np.mean([psnr(A[i], B[i]) for i in range(n)]))
        perturb[name] = {"pair": [a, b], "per_prompt": vals,
                         "mean_db": float(np.mean(list(vals.values()))) if vals else None}

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "not_a_new_experiment": ("reads the already-generated mp4s only; no sampling, "
                                 "no new arm, no new prompt"),
        "metric": ("center_border_hf_ratio = Laplacian energy density in the central 50% "
                   "box divided by that in the surrounding border. Close-up (sharp centred "
                   "subject, shallow-DOF background) -> high; wide shot (detail spread "
                   "across the frame) -> near 1."),
        "composition": comp,
        "cross_arm_video_psnr": dist,
        "fp8_perturbation_under_two_conditions": {
            "values": perturb,
            "question_it_would_answer": ("is the perturbation FP8 causes the same with and "
                                         "without sparsity? equal => the two axes compose "
                                         "additively in the output domain"),
            "obstacle_1_configuration": (
                "NOT a matched pair. The dense side uses the PSEUDO-QUANT path "
                "(wan_fp8.convert_fakequant); fp8_sparse uses the REAL _scaled_mm path with "
                "E3 fusion (fusion.wan_fused.convert_fused). Same algorithm (RTN "
                "per-tensor), different code path. T4-A measured that path difference: two "
                "mathematically equivalent real-kernel configs differ by latent L2 0.1204, "
                "against the quant arms' own deviation from bf16 of only 0.31-0.39 -- the "
                "confound is about a third of the effect. The strict test needs a matched "
                "fp8_dense arm (3 videos); not generated."),
            "obstacle_2_deeper": (
                "EVEN WITH a matched pair this would not establish 'coupling'. The "
                "latency-domain disjointness is a statement about a LINEARLY ADDITIVE "
                "quantity (time). The output domain has no additive basis: the sampler is "
                "nonlinear, so a perturbation of a given size applied to an ALREADY "
                "DIVERGED trajectory (the sparse arm is at latent L2 1.13, past the signal "
                "magnitude) simply does not act the same way. That is a property of the ODE "
                "solver, not of the axes interacting."),
            "correct_conclusion": (
                "'Disjoint' cannot be extrapolated to the output domain -- not because we "
                "did not measure it, but because the output domain lacks an additive basis "
                "in which 'disjoint' is even defined. The latency domain admits the "
                "question only because time is additive."),
            "how_to_read_the_numbers": ("illustration that output-domain composition is not "
                                        "magnitude-preserving; NOT evidence of coupling"),
        },
        "n_prompts": len(PROMPTS),
        "caveat": ("n=3. Descriptive observations reported alongside the extracted frames, "
                   "not statistical claims."),
        "verdict": {
            "center_border_hf_ratio": {
                "status": "REFUTED as a discriminator",
                "why": ("bf16_sparse mean 1.79 vs bf16 1.74, and the per-prompt directions "
                        "disagree (p1 lower, p2 higher). The proxy conflates FRAMING with "
                        "BACKGROUND BLUR: the p1 sparse clip is a close-up but its "
                        "background is high-frequency bunting, which pushes the ratio down."),
                "consequence": ("the 'composition collapses to a close-up' observation stays "
                                "QUALITATIVE, supported by results/frames/ only; no number "
                                "is attached to it in the report"),
            },
            "cross_arm_video_psnr": {
                "status": "holds on 3/3 prompts",
                "claim": ("the two sparse arms resemble EACH OTHER more than either "
                          "resembles dense -> the sparsity axis fixes the sample identity "
                          "and FP8 on top is a smaller perturbation. This is a DOMINANCE "
                          "statement, NOT independence. The T8 sentence calling it 'the "
                          "pixel-domain counterpart of H2' has been deleted -- see the "
                          "module docstring for why the output domain admits no such "
                          "counterpart."),
                "secondary": ("dense~warm1 sits between dense~gptq_cd and dense~sparse on "
                              "3/3 prompts -> one dense warmup step recovers most of the "
                              "distance"),
            },
        },
    }
    out = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)

    print("=== center/border high-frequency ratio (median over frames) ===")
    print("高 = 特写（主体居中锐利、背景虚化）；接近 1 = 细节铺满画面")
    print(f"{'arm':20s} " + " ".join(f"{p:>8s}" for p in PROMPTS) + f" {'mean':>8s}")
    for arm in ARMS:
        if not comp.get(arm):
            continue
        v = [comp[arm][p]["median"] for p in PROMPTS if p in comp[arm]]
        print(f"{arm:20s} " + " ".join(f"{x:8.2f}" for x in v) +
              f" {np.mean(v):8.2f}")
    print("\n=== cross-arm full-clip mean PSNR (dB) ===")
    keys = list(next(iter(dist.values())).keys())
    print(f"{'prompt':8s} " + " ".join(f"{k:>18s}" for k in keys))
    for p, row in dist.items():
        print(f"{p:8s} " + " ".join(f"{row[k]:18.2f}" for k in keys))
    print("\n=== T9 §2: FP8 的扰动幅度（dB，越低 = 扰动越大）===")
    for name, r in perturb.items():
        v = " ".join(f"{r['per_prompt'][q]:7.2f}" for q in PROMPTS if q in r["per_prompt"])
        print(f"    {name:44s} {v}   mean {r['mean_db']:6.2f}")
    print("    ⚠️ 两者代码路径不同（伪量化 vs 真 kernel + E3 融合），且输出域没有可加基底")
    print("       —— 只作『复合不保幅』的例示，不作耦合证据。见文件头。")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
