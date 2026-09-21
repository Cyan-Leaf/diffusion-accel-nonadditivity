#!/usr/bin/env python3
"""vbench_pervideo.py — T8 §1：不做平均，把 VBench 的逐视频分数摊开

**为什么**：T7 报的是每臂 3 条视频的**均值**。于是「稀疏更差」这个结论的**无参照证据**
只剩一个数：imaging 0.7027 vs 0.7365，gap = 0.0338，**没有任何离散度估计**，
读者无法判断它是不是噪声。

MUSIQ 是**逐视频**打分的，而各臂用的是**同一组 prompt、同一组 seed** ——
所以这是一个**配对**设计，只是被平均掉了。把配对恢复出来不需要重跑任何东西。

本脚本做三件事（全部只读 `results/E9/vbench_work/_results/*_eval_results.json`）：

1. **配对符号检验**：对每个臂，逐 prompt 与 `bf16` 比，数有几个 prompt 变差。
   n=3 时 3/3 同向的精确双尾 p = 0.25 —— **报出来，但必须写明 n=3 下它不可能显著**。
   它的价值不是 p 值，是**「有没有交叉」**：若有交叉，那比均值差更值得知道。
2. **两个噪声地板**（T8 §1.1 / §1.2），都从数据自身取：
   - `fp8_sparse` 在 imaging 上反超 `bf16_sparse`，而 PSNR 与 latent L2 都说它更差 ——
     **方向上不该发生**，把这个反超量当作噪声下界；
   - 五个量化臂（质量应当接近）之间的极差。
3. **gap / 噪声地板的比值** —— 结论只能写到这个精度。

⚠️ **不要把本脚本的输出写成「VBench 证实了稀疏更差」。** 见 T8 §1.3：
四条证据（VBench imaging、PSNR、高频能量比、抽帧）方向一致 → 结论成立；
单拎 VBench 出来 → 撑不住。这是通则 3 的同类。

用法（普通 python 即可，纯读 json）：
    python3 eval/vbench_pervideo.py --out results/E9/vbench_pervideo.json
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import time

REPO = os.environ.get("GENMODEL_ACCEL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = "bf16"
QUANT_ARMS = ["rtn_tensor", "rtn_token", "gptq", "gptq_cd", "gptq_clip_cd"]
SPARSE_ARMS = ["bf16_sparse", "bf16_sparse_warm1", "fp8_sparse"]


def load_per_video(results_dir: str) -> dict:
    """{arm: {dim: {prompt_tag: score}}}，MUSIQ 原始分除以 100 与 VBench 均值同尺度。"""
    out = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "*_eval_results.json"))):
        arm = os.path.basename(f)[: -len("_eval_results.json")]
        d = json.load(open(f))
        per = {}
        for dim, payload in d.items():
            mean, rows = payload[0], payload[1]
            scores = {}
            for r in rows:
                tag = os.path.basename(r["video_path"]).rsplit("_p", 1)[1].split(".")[0]
                v = r["video_results"]
                # imaging_quality 的逐视频分是 MUSIQ 原始分（0-100），均值已 /100
                scores["p" + tag] = v / 100.0 if dim == "imaging_quality" else v
            per[dim] = {"mean_reported": mean, "per_video": scores,
                        "mean_recomputed": sum(scores.values()) / len(scores)}
        out[arm] = per
    return out


def sign_test_two_sided(n_down: int, n: int) -> float:
    """精确二项双尾 p（H0: p=0.5）。n=3 时 3/3 给 0.25 —— 这正是要写明的那件事。"""
    k = min(n_down, n - n_down)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/E9/vbench_work/_results")
    ap.add_argument("--out", default="results/E9/vbench_pervideo.json")
    args = ap.parse_args()

    rd = os.path.join(REPO, args.results)
    data = load_per_video(rd)
    assert REF in data, f"reference arm {REF} not found in {rd}"
    dims = sorted(data[REF].keys())

    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "source": args.results, "reference_arm": REF,
               "why": ("VBench means over n=3 hide the fact that the design is PAIRED "
                       "(same prompts, same seeds across arms). Recovering the pairing "
                       "costs nothing and turns a suspicious mean difference into a "
                       "directional observation."),
               "per_video": data, "paired": {}, "noise_floors": {}, "verdict": {}}

    # ---- 1. 配对符号检验 -------------------------------------------------
    for dim in dims:
        ref = data[REF][dim]["per_video"]
        prompts = sorted(ref)
        rows = {}
        for arm in data:
            if arm == REF:
                continue
            cur = data[arm][dim]["per_video"]
            deltas = {p: cur[p] - ref[p] for p in prompts}
            n_down = sum(1 for p in prompts if deltas[p] < 0)
            rows[arm] = {
                "per_prompt": {p: {"ref": ref[p], "arm": cur[p], "delta": deltas[p]}
                               for p in prompts},
                "n_prompts": len(prompts),
                "n_worse_than_ref": n_down,
                "all_same_direction": n_down in (0, len(prompts)),
                "sign_test_p_two_sided": sign_test_two_sided(n_down, len(prompts)),
                "mean_delta": sum(deltas.values()) / len(prompts),
                "min_delta": min(deltas.values()), "max_delta": max(deltas.values()),
            }
        payload["paired"][dim] = rows

    # ---- 2. 两个噪声地板（都取自数据自身）-------------------------------
    dim = "imaging_quality"
    m = {a: data[a][dim]["mean_reported"] for a in data}

    # (a) 方向矛盾：fp8_sparse 在 imaging 上反超 bf16_sparse，而 PSNR/latent 都说更差
    contradiction = m["fp8_sparse"] - m["bf16_sparse"]
    # (b) 五个量化臂（质量应当接近）的极差
    qv = [m[a] for a in QUANT_ARMS if a in m]
    quant_range = max(qv) - min(qv)
    # (c) 顺带：同一臂内跨 prompt 的离散（这是 prompt 难度，不是方法差异，仅作尺度参照）
    within = {a: max(data[a][dim]["per_video"].values()) - min(data[a][dim]["per_video"].values())
              for a in data}

    payload["noise_floors"][dim] = {
        "a_direction_contradiction": {
            "value": contradiction,
            "what": ("fp8_sparse beats bf16_sparse on VBench imaging by this much, while "
                     "PSNR (8.12 vs 8.26 dB) and latent L2 (1.1453 vs 1.1273) both say it "
                     "is WORSE. Stacking a lossy quantiser on top of sparsity should not "
                     "improve the image. Treat this as a lower bound on the noise."),
        },
        "b_quant_arm_range": {
            "value": quant_range, "arms": {a: m[a] for a in QUANT_ARMS if a in m},
            "what": ("range across five quantisation arms whose quality ought to be close; "
                     "their VBench ordering also disagrees with their PSNR ordering"),
        },
        "c_within_arm_across_prompts": {
            "value": within,
            "what": ("spread across the 3 prompts WITHIN one arm -- this is prompt "
                     "difficulty, not method noise; listed only to show that the paired "
                     "design is what removes it"),
        },
    }

    # ---- 3. gap / 噪声地板 ------------------------------------------------
    gap = m[REF] - m["bf16_sparse"]
    floor = max(abs(contradiction), quant_range)
    payload["verdict"][dim] = {
        "gap_sparse_vs_ref": gap,
        "conservative_noise_floor": floor,
        "floor_source": ("direction contradiction" if abs(contradiction) >= quant_range
                         else "quant arm range"),
        "ratio": gap / floor if floor > 0 else float("inf"),
        "reading": ("The drop on VBench imaging is about this many times the noise floor of "
                    "the metric in THIS setting. Direction supported, magnitude NOT strong "
                    "enough to carry the conclusion alone. Its value is that it agrees with "
                    "PSNR, high-frequency energy ratio and the extracted frames -- and "
                    "unlike those three, it is reference-free."),
        "do_not_write": "VBench confirms that sparsity is worse (it does not, at n=3)",
        "SUPERSEDES_the_ratio_above": {
            "finding": ("the paired breakdown makes the ratio the WEAKER statement. The "
                        "mean gap is not a uniform shift: p1 -0.1371, p0 -0.0115, "
                        "p2 +0.0472 (sparse BETTER). One prompt carries 135% of the mean "
                        "gap and another cancels 47% of it."),
            "why_it_matters": ("this is the signature of 'a different video was generated', "
                               "not 'the same video was degraded': a no-reference IQA score "
                               "of a DIFFERENT scene is a draw from a different "
                               "distribution. On p2 the sparse arm's close-up framing "
                               "scores HIGHER than dense's wide shot -- the metric rewards "
                               "the failure mode."),
            "contrast_with_quantisation": ("the five quantisation arms give 14 of 15 paired "
                                           "comparisons negative, each small. So the SMALLER "
                                           "effect is the BETTER established one -- exactly "
                                           "inverted from what the means suggest."),
            "correct_wording": ("VBench imaging cannot arbitrate the sparse arm, because "
                                "sparsity changes which video is generated rather than "
                                "degrading a fixed one. It CAN arbitrate the quantisation "
                                "arms (composition preserved, 14/15 paired comparisons "
                                "negative). The evidence that sparsity costs quality is the "
                                "latent/pixel divergence plus the frames, not VBench."),
        },
        "warm1_between": {
            "bf16_sparse": m["bf16_sparse"], "bf16_sparse_warm1": m["bf16_sparse_warm1"],
            "quant_arm_band": [min(qv), max(qv)], "ref": m[REF],
            "what": ("warm1 lands between the no-warmup sparse arm and the quantisation "
                     "band: dense warmup recovers most of the VBench gap too, giving the "
                     "'first step sets the composition' mechanism a FOURTH independent "
                     "line of evidence alongside PSNR, HF ratio and the frames"),
        },
    }

    out = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)

    # ---- 打印 -------------------------------------------------------------
    print(f"=== per-video {dim} (MUSIQ/100), reference = {REF} ===")
    prompts = sorted(data[REF][dim]["per_video"])
    print(f"{'arm':22s} " + " ".join(f"{p:>8s}" for p in prompts) + f" {'mean':>8s}  vs ref")
    for arm in [REF] + QUANT_ARMS + SPARSE_ARMS:
        if arm not in data:
            continue
        pv = data[arm][dim]["per_video"]
        line = f"{arm:22s} " + " ".join(f"{pv[p]:8.4f}" for p in prompts)
        line += f" {data[arm][dim]['mean_reported']:8.4f}"
        if arm != REF:
            r = payload["paired"][dim][arm]
            marks = "".join("-" if r["per_prompt"][p]["delta"] < 0 else "+" for p in prompts)
            line += f"  {marks}  {r['n_worse_than_ref']}/{r['n_prompts']} worse"
            if r["all_same_direction"]:
                line += f"  (sign p={r['sign_test_p_two_sided']:.2f})"
        print(line)
    v = payload["verdict"][dim]
    print(f"\ngap(ref - bf16_sparse) = {v['gap_sparse_vs_ref']:.4f}")
    print(f"noise floor            = {v['conservative_noise_floor']:.4f} "
          f"({v['floor_source']})")
    print(f"ratio                  = {v['ratio']:.2f}x")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
