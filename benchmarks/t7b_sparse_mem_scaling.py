#!/usr/bin/env python3
"""t7b_sparse_mem_scaling.py — T7 §2：稀疏 attention 的显存是 O(S) 还是 O(S²)

**为什么非查不可**：T6 报 `bf16_sparse` 峰值 20.31 GiB 对 `bf16_dense` 的 4.79 GiB，
我把它归因为「FlexAttention 后端的代价，SVG 的 flashinfer 应该更省」。
**那个归因没有证据，而且有一个更可能的解释。**

块稀疏 attention 的显存应该是 **O(S)**（只存被保留的块）。
若实测是 **O(S²)**，说明 score 矩阵被物化了 ——
单 head 的 [32760, 32760] bf16 就是 2.1 GB，几个 head 并发就是 15 GiB 量级。
**那不是"后端的代价"，是 BlockMask 没生效、退化成 dense mask + masked softmax。**

| 观测 | 结论 | 报告怎么写 |
|---|---|---|
| 增量 ∝ S | 真是块稀疏 | 20.31 GiB 是真实部署代价，照原样写 |
| 增量 ∝ S² | score 矩阵被物化 | **必须改口径**：不能说成 FlexAttention 的固有代价 |

⚠️ **无论结果如何都不修**（那是 kernel 工程，不服务论点）。只把归因写对。

这是通则 5 的精神：**对照组的配置要检查，被你批评的那个对象的配置同样要检查。**

用法：
    $PY benchmarks/t7b_sparse_mem_scaling.py --device 0 --out results/E4/sparse_mem_scaling.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

import torch

REPO = os.environ.get("GENMODEL_ACCEL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

H, D = 12, 128
FRAME_SIZE = 1560          # Wan2.1 @ 832x480 的每帧 patch 数


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--frames", default="5,10,21",
                    help="latent 帧数；S = frames * 1560")
    ap.add_argument("--out", default="results/E4/sparse_mem_scaling.json")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.cuda.set_device(args.device)
    dev = f"cuda:{args.device}"
    p = torch.cuda.get_device_properties(args.device)
    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    from sparse import wan_svg as S

    rows = []
    for nf in [int(x) for x in args.frames.split(",")]:
        Sn = nf * FRAME_SIZE
        q = torch.randn(1, H, Sn, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, H, Sn, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, H, Sn, D, device=dev, dtype=torch.bfloat16)
        base = torch.cuda.memory_allocated()

        r = {"num_frame": nf, "S": Sn, "qkv_bytes": 3 * H * Sn * D * 2,
             "score_matrix_bytes_if_materialised": H * Sn * Sn * 2}

        # dense 参照
        free()
        torch.cuda.reset_peak_memory_stats()
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        r["dense_peak_bytes"] = torch.cuda.max_memory_allocated()
        del _
        free()

        # 稀疏
        try:
            bm = S.get_block_mask(Sn, dev, num_frame=nf, frame_size=FRAME_SIZE)
            _ = S.flex(q, k, v, bm)          # warmup（编译）
            torch.cuda.synchronize()
            free()
            torch.cuda.reset_peak_memory_stats()
            _ = S.flex(q, k, v, bm)
            torch.cuda.synchronize()
            r["sparse_peak_bytes"] = torch.cuda.max_memory_allocated()
            r["ok"] = True
            del _
        except Exception as e:
            r["ok"] = False
            r["err"] = f"{type(e).__name__}: {str(e)[:200]}"
        free()
        del q, k, v
        free()

        if r.get("ok"):
            r["delta_bytes"] = r["sparse_peak_bytes"] - r["dense_peak_bytes"]
            r["delta_gib"] = r["delta_bytes"] / 2**30
            r["delta_over_S"] = r["delta_bytes"] / Sn
            r["delta_over_S2"] = r["delta_bytes"] / Sn ** 2
            r["delta_as_frac_of_full_score_matrix"] = (
                r["delta_bytes"] / r["score_matrix_bytes_if_materialised"])
            print(f"[S={Sn:6d}] dense {r['dense_peak_bytes']/2**30:6.2f} GiB  "
                  f"sparse {r['sparse_peak_bytes']/2**30:6.2f} GiB  "
                  f"delta {r['delta_gib']:6.2f} GiB  "
                  f"= {r['delta_as_frac_of_full_score_matrix']*100:5.1f}% of a full "
                  f"[H,S,S] score matrix", flush=True)
        else:
            print(f"[S={Sn:6d}] FAILED: {r['err'][:80]}", flush=True)
        rows.append(r)

    # ---- 若 kernel 本身是 O(S)，那 20.31 GiB 在哪？逐段隔离 ----------------
    Sn = 21 * FRAME_SIZE
    q = torch.randn(1, H, Sn, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, H, Sn, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, H, Sn, D, device=dev, dtype=torch.bfloat16)
    bm = S.get_block_mask(Sn, dev, num_frame=21, frame_size=FRAME_SIZE)
    dm = S.build_dense_masks(device=dev, max_row=10000)
    idx = torch.tensor([[0] * 6 + [1] * 6], device=dev)
    stages = {}

    def peak(fn, warm=1):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        free()
        torch.cuda.reset_peak_memory_stats()
        r = fn()
        torch.cuda.synchronize()
        pk = torch.cuda.max_memory_allocated()
        del r
        free()
        return pk

    stages["a_bare_flex"] = peak(lambda: S.flex(q, k, v, bm))
    stages["b_svg_attention_with_reorder"] = peak(
        lambda: S.svg_attention(q, k, v, idx))
    stages["c_sample_mse_only"] = peak(lambda: S.sample_mse(q, k, v, dm))
    stages["dense_masks_resident_bytes"] = sum(
        t.numel() * t.element_size() for t in dm)
    payload_stages = {kk: (vv / 2**30 if "bytes" not in kk else vv / 2**30)
                      for kk, vv in stages.items()}
    print("\nstage isolation at S=32760 (peak GiB):")
    for kk, vv in payload_stages.items():
        print(f"    {kk:34s} {vv:6.3f}")
    del q, k, v, dm
    free()

    ok = [r for r in rows if r.get("ok")]
    verdict = {}
    if len(ok) >= 2:
        a, b = ok[0], ok[-1]
        rs = b["S"] / a["S"]
        rd = b["delta_bytes"] / max(a["delta_bytes"], 1)
        # 拟合指数：delta ~ S^alpha
        alpha = math.log(rd) / math.log(rs) if rd > 0 and rs > 1 else float("nan")
        verdict = {
            "S_ratio": rs, "delta_ratio": rd, "fitted_exponent_alpha": alpha,
            "interpretation": ("delta ~ S^alpha. alpha ~1 => genuinely block-sparse "
                               "(memory scales with kept blocks). alpha ~2 => the score "
                               "matrix is being materialised, i.e. the BlockMask is not "
                               "delivering its memory advantage."),
            "conclusion": ("block-sparse (O(S))" if alpha < 1.4 else
                           "score matrix materialised (O(S^2))" if alpha > 1.7 else
                           "ambiguous"),
            "delta_as_frac_of_full_score_matrix": {
                str(r["S"]): r["delta_as_frac_of_full_score_matrix"] for r in ok},
        }

    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "device": p.name, "torch": torch.__version__,
               "purpose": "decide whether the 20.31 GiB sparse peak reported in T6 is a "
                          "genuine block-sparse deployment cost or an artefact of the "
                          "score matrix being materialised",
               "rows": rows, "verdict": verdict,
               "stage_isolation_gib_at_S32760": payload_stages,
               "stage_isolation_note": (
                   "if the bare FlexAttention call is small but svg_attention or "
                   "sample_mse is large, the 20.31 GiB reported in T6 belongs to OUR "
                   "integration layer, not to the FlexAttention backend -- the "
                   "attribution in the report must say so."),
               "do_not_fix": "regardless of the outcome we do not optimise this; only the "
                             "attribution in the report is corrected."}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    if verdict:
        print(f"\nS ratio {verdict['S_ratio']:.2f}x -> delta ratio "
              f"{verdict['delta_ratio']:.2f}x -> alpha = "
              f"{verdict['fitted_exponent_alpha']:.2f}")
        print(f"CONCLUSION: {verdict['conclusion']}")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
