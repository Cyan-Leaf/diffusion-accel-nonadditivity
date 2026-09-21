#!/usr/bin/env python3
"""e7_nvfp4_parity.py — NVFP4 伪量化的**码本层** parity gate（T13 §2.5(c)）

## ⚠️ 为什么不能直接复用 `FakeQuantLinearParity`

T13 §2.5 要求「先过 `FakeQuantLinearParity` 对 FP64 参照，再跑端到端」。
**那个类做不到，而原因本身是结论的一部分**：它的第 1 条纪律是
「量化决策用**硬件 cast**，不用 `bucketize`」—— 而 Ada（sm_89）上
`torch.float8_e4m3fn` 存在，**`float4` 不存在**。NVFP4 没有硬件 cast 可对。

→ 于是 NVFP4 的 parity 只能是**码本层**的：拿一个**独立的 FP64 参照实现**
（穷举最近电平，不用 `bucketize`）去对我们的实现。

**这正好对应 T4-A 的裁定**：「伪量化在**格式层面**有效、在 **kernel 输出预测**上无效」。
对 NVFP4，本机**根本没有 kernel**，所以格式层就是全部 ——
**也正因如此，本文这一节只能叫「NVFP4 码本的算法质量」，不能叫「NVFP4 的质量」。**

## ⚠️ 第一版把这个 gate 设计错了，记在这里（通则 6 第三次重犯）

第一版的检查 3 是「`bucketize` 选出的电平 vs 穷举最近邻，**必须 100% 一致，否则是 bug**」，
检查 2 的阈值设成 `rel-L2 < 1e-5`。**跑出来 FAIL（2.06e-2 / 0.47%），而实现是对的。**

原因：我的参照**在 fp64 里算 scale**，被测实现**在 fp32 里算**。两者差 ~1e-7，
而 **0.20–0.44% 的权重元素恰好落在量化判决边界（中点）上** ——
`bf16` 权重只有 8 位尾数、E4M3 scale 也是离散的，两个粗网格相除，
**精确命中中点的概率远高于连续分布下的直觉**。scale 动 1e-7，这些元素就翻到另一侧。

**我设阈值时没有先推导「实现完全正确时这个读出量会是多少」** ——
答案不是 0，是「边界元素比例 × 一个电平步长」。
**这正是通则 6，而它已经是第三次被重犯（通则 4 之于 T5、通则 6 之于 T11、这次之于我）**，
也正是通则 9 说的那件事：**通则不会因为写下来就生效。**

→ 改法：**把「算法」与「精度」两个变量分开测**。

## 六项检查

1. **E2M1 电平表**：必须恰好是 ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}，absmax = 6
2. **算法一致性（同一个 x）**：`bucketize` vs 穷举最近邻，**必须逐位相同（rel-L2 == 0）**
   —— 这才是「实现有没有 bug」的那个检验，零假设下它恰好是 0
3. **精度敏感性（fp32 vs fp64 的 scale）**：报翻转比例，**并与边界元素比例对照** ——
   零假设下它 ≈ 边界元素比例，不是 0
4. **边界元素比例**：本身是一个值得报的量（bf16 网格 × 离散 scale 的后果）
5. **无 bf16 往返**（T4-A bug1 同类）：dtype 必须全程 fp32
6. **二级 scale（E4M3）**：必须真的改变 scale，且代价可量

用法：
    <PY> benchmarks/e7_nvfp4_parity.py --out results/E7/nvfp4_parity.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

REPO = os.environ.get("GENMODEL_ACCEL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
DISTILL = os.environ.get("DISTILL", "")   # lightx2v 4 步蒸馏权重 (rev ef72050)
assert DISTILL, "请先 export DISTILL=<distill_native.pt 的路径>"


def ref_nvfp4_fp64(W64, group=16, scale_e4m3=True):
    """独立的 FP64 参照：不用 bucketize，穷举最近电平。

    刻意用和被测实现不同的写法（cdist 式穷举 vs 中点 bucketize），
    这样两者一致才有意义 —— 否则只是同一段代码跑两遍。
    """
    levels = torch.tensor([s * v for v in E2M1 for s in (1.0, -1.0)],
                          dtype=torch.float64, device=W64.device)
    levels = torch.unique(levels)
    absmax_cb = float(levels.abs().max())
    N, K = W64.shape
    amax = W64.reshape(N, K // group, group).abs().amax(-1)          # [N, G]
    s = amax / absmax_cb
    if scale_e4m3:
        enc = 448.0 * absmax_cb / W64.abs().max().clamp_min(1e-30)
        s = (s.clamp_min(1e-30) * enc).to(torch.float8_e4m3fn).double() / enc
    s = s.clamp_min(1e-30)
    x = W64.reshape(N, K // group, group) / s.unsqueeze(-1)
    # 穷举最近电平
    idx = (x.unsqueeze(-1) - levels.view(1, 1, 1, -1)).abs().argmin(-1)
    q = levels[idx]
    return (q * s.unsqueeze(-1)).reshape(N, K), s, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--n-tensors", type=int, default=12)
    ap.add_argument("--out", default="results/E7/nvfp4_parity.json")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    dev = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)
    import sys
    sys.path.insert(0, REPO)
    from quant.gptq import formats

    res = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "why_not_FakeQuantLinearParity": (
               "that harness quantises via a HARDWARE CAST (torch.float8_e4m3fn). Ada has no "
               "float4 dtype, so there is no hardware cast to compare against for NVFP4. The "
               "parity that CAN be run is codebook-level: our implementation against an "
               "independent FP64 exhaustive-nearest-level reference."),
           "scope_discipline": ("this gate covers the FORMAT MODEL only. There is no NVFP4 "
                                "kernel on this machine, so no kernel-output claim is made. "
                                "The report section must say 'the algorithmic quality of the "
                                "NVFP4 codebook', not 'the quality of NVFP4'."),
           "checks": {}}

    # ---- 1. 电平表 -------------------------------------------------------
    cb = formats.CODEBOOKS["nvfp4"]
    expect = sorted(set([s * v for v in E2M1 for s in (1.0, -1.0)]))
    got = [round(float(x), 6) for x in cb.levels.tolist()]
    ok1 = got == [round(x, 6) for x in expect] and abs(cb.absmax - 6.0) < 1e-9
    res["checks"]["1_e2m1_levels"] = {
        "pass": bool(ok1), "n_levels": cb.n, "absmax": cb.absmax,
        "levels": got,
        "note": "OCP E2M1 不保留 inf/nan，故 absmax = 6（若实现保留了就会是 3，那是 T6 抓过的坑）"}
    print(f"[1] 电平表 n={cb.n} absmax={cb.absmax}  -> {'PASS' if ok1 else 'FAIL'}")

    # ---- 取真实 Wan 权重 -------------------------------------------------
    sd = torch.load(DISTILL, map_location="cpu", weights_only=True)
    names = [f"blocks.{i}.{s}.weight" for i in range(30)
             for s in ("self_attn.q", "ffn.0", "cross_attn.v")]
    names = [n for n in names if n in sd][:args.n_tensors]

    wf = formats.parse_wfmt("nvfp4:16:e4m3")
    lv64 = torch.unique(torch.tensor([sg * v for v in E2M1 for sg in (1.0, -1.0)],
                                     dtype=torch.float64, device=dev))
    mids64 = (lv64[1:] + lv64[:-1]) / 2
    alg, rel, flip, tief, nontie, dt_ok = [], [], [], [], [], []
    for n in names:
        W = sd[n].to(dev).float()
        if W.shape[1] % 16:
            continue
        s_ours = wf.col_scales(W)
        x = (W / s_ours).contiguous()
        lv_bucket = cb.value(cb.index(x)).double()
        lv_exhaust = lv64[(x.double().unsqueeze(-1) - lv64.view(1, 1, -1)).abs().argmin(-1)]
        # 2) 算法一致性：同一个 x
        alg.append(float((lv_bucket - lv_exhaust).abs().max()))
        dt_ok.append(wf.rtn(W).dtype == torch.float32)
        # 3/4) 精度敏感性：翻转必须能被 scale 扰动**精确解释**
        #     判据不用任意带宽：一个元素翻转是「可解释的」当且仅当判决边界
        #     落在 x_ours 与 x_ref 之间。带宽由扰动自己定，不由我拍。
        q_ref, s_ref, _ = ref_nvfp4_fp64(W.double(), 16, True)
        lv_ref = q_ref / s_ref.repeat_interleave(16, 1)
        x_ref = W.double() / s_ref.repeat_interleave(16, 1)
        d = (lv_bucket - lv_ref).abs().gt(1e-9)
        lo_x = torch.minimum(x.double(), x_ref)
        hi_x = torch.maximum(x.double(), x_ref)
        # 是否存在某个中点 m 满足 lo <= m <= hi
        m = mids64.view(1, 1, -1)
        straddles = ((m >= lo_x.unsqueeze(-1)) & (m <= hi_x.unsqueeze(-1))).any(-1)
        flip.append(float(d.float().mean()))
        tief.append(float(straddles.float().mean()))
        nontie.append(float((d & ~straddles).float().mean()))
        rel.append(float((wf.rtn(W).double() - q_ref).norm() / q_ref.norm().clamp_min(1e-30)))
        del W, q_ref
    assert alg, "no tensor evaluated"

    ok2 = max(alg) == 0.0
    res["checks"]["2_algorithm_identical_same_input"] = {
        "pass": bool(ok2), "n_tensors": len(alg),
        "rel_l2_max": max(alg),
        "null_hypothesis_value": 0.0,
        "what_it_tests": ("bucketize-against-midpoints vs exhaustive nearest-level, fed the "
                          "SAME fp32 normalised input. If the implementation is correct this "
                          "is EXACTLY zero -- both are nearest-level with the same tie rule."),
        "contrast_with_T4A": ("T4-A measured up to 3.99% of elements choosing different "
                              "levels on FP8, but there the two sides were a HARDWARE CAST "
                              "and bucketize -- different rounding rules. Here both sides are "
                              "software nearest-level, so any difference would be a bug.")}
    print(f"[2] 算法一致性（同一 x）rel-L2 max={max(alg):.3e}  -> {'PASS' if ok2 else 'FAIL'}")

    ok3 = max(nontie) == 0.0
    res["checks"]["3_precision_sensitivity_fp32_vs_fp64_scale"] = {
        "pass": bool(ok3),
        "frac_flipped_max": max(flip), "frac_flipped_mean": sum(flip) / len(flip),
        "frac_straddling_a_boundary_max": max(tief),
        "frac_flipped_NOT_explained_by_the_perturbation": max(nontie),
        "null_hypothesis_value": ("NOT zero for the flip rate -- it equals the fraction of "
                                  "elements whose decision boundary lies between the fp32 and "
                                  "fp64 normalised values. The TESTABLE quantity is the "
                                  "fraction of flips that CANNOT be explained that way, whose "
                                  "null value IS exactly zero."),
        "what_it_tests": ("our scale is computed in fp32, the reference in fp64; they differ "
                          "by ~1e-7. Elements exactly on a midpoint flip. Off-boundary "
                          "elements must not."),
        "rel_l2_including_boundary_flips": max(rel),
        "note": ("the 1e-5 threshold in the first version of this gate was wrong: it assumed "
                 "the null value was 0 for a quantity whose null value is the boundary "
                 "fraction. See the module docstring.")}
    print(f"[3] fp32/fp64 scale 翻转 max={max(flip)*100:.4f}%  其中**不能被扰动解释的** "
          f"{max(nontie)*100:.5f}%  -> {'PASS' if ok3 else 'FAIL'}")

    res["checks"]["4_decision_boundary_fraction"] = {
        "pass": True,
        "frac_max": max(tief), "frac_mean": sum(tief) / len(tief),
        "why_it_is_high": ("bf16 weights carry only 8 mantissa bits and the E4M3 scale is "
                           "itself discrete; dividing one coarse grid by another lands on a "
                           "midpoint far more often than a continuous distribution would. "
                           "This is a property of the data, not a defect."),
        "consequence": ("any two NVFP4 implementations that differ in tie rule or in scale "
                        "precision will disagree on this fraction of weights. Reporting a "
                        "bit-exact match between two such implementations without stating "
                        "the tie rule would be meaningless.")}
    print(f"[4] 判决边界落在 fp32/fp64 两值之间的元素 max={max(tief)*100:.4f}% "
          f"mean={sum(tief)/len(tief)*100:.4f}%   （bf16 网格 × 离散 scale 的后果，不是缺陷）")

    ok5 = all(dt_ok)
    res["checks"]["5_no_bf16_roundtrip_bug1_analogue"] = {
        "pass": bool(ok5), "all_fp32": bool(ok5),
        "note": "T4-A bug1：中间结果落 bf16 贡献 0.7–4.7pp 的误差"}
    print(f"[5] 全程 fp32（无 bf16 往返） -> {'PASS' if ok5 else 'FAIL'}")

    # ---- 5. 二级 scale 的效果 -------------------------------------------
    W = sd[names[0]].to(dev).float()
    wf_fp32 = formats.parse_wfmt("nvfp4:16:fp32")
    e_fp32 = float((W - wf_fp32.rtn(W)).norm() / W.norm())
    e_e4m3 = float((W - wf.rtn(W)).norm() / W.norm())
    s_a = wf_fp32.group_scales(W)
    s_b = wf.group_scales(W)
    changed = float((s_a - s_b).abs().gt(0).float().mean())
    ok5 = changed > 0.5 and e_e4m3 >= e_fp32
    res["checks"]["6_two_level_scale"] = {
        "pass": bool(ok5), "frac_scales_changed_by_e4m3": changed,
        "rel_err_scale_fp32": e_fp32, "rel_err_scale_e4m3": e_e4m3,
        "cost_pct": (e_e4m3 / e_fp32 - 1) * 100,
        "note": "§8.3(b) 报的 D2 代价是 +1.05%（300 张量均值）；此处是单张量抽查"}
    print(f"[6] 二级 scale：{changed*100:.0f}% 的 scale 被改变，误差 "
          f"{e_fp32:.5f} -> {e_e4m3:.5f} ({(e_e4m3/e_fp32-1)*100:+.2f}%)"
          f"  -> {'PASS' if ok5 else 'FAIL'}")

    allok = all(c["pass"] for c in res["checks"].values())
    res["GATE"] = {"pass": bool(allok),
                   "meaning": ("PASS 只意味着格式模型可信，**不意味着任何 kernel 结论**。"
                               "按 T13 §2.5，PASS 才允许跑质量臂。")}
    path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"\n=== GATE: {'PASS -> 允许跑质量臂' if allok else 'FAIL -> 停下来报告，不要跑质量'} ===")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
