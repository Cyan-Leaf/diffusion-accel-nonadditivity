#!/usr/bin/env python3
"""t6a_attn_flops_audit.py — T6-A：dense SDPA 的 FLOPs / 计时口径核对

**为什么要做**：E4 报 dense SDPA 跑到 164.8 TFLOPS，是 4090 BF16(FP32-acc)
标称峰值 165.2 的 **99.8%**。按 `CLAUDE.md` 通则 4，**贴着峰值的数字先怀疑测量**。

这个数不影响 4.13× 那个加速比（那是 40.0/9.7 的时间比，与 FLOPs 口径无关），
但「dense 那一边没被浪费」这个健全性检查是 E4 叙事的承重墙 ——
FLOPs 口径若错，这个检查就是空的。

核对四件事：
  1. FLOPs 公式：非因果前向应为 `4·B·H·S²·D`（QK^T 与 P@V 各 2·B·H·S²·D）
  2. 计时：CUDA event vs perf_counter，warmup 是否够，有没有漏 sync
  3. **同 session 内的 GEMM 对照** —— 这才是决定性的。
     标称峰值是按 nominal boost clock 算的，而 E0 里 BF16 GEMM 实测到过
     167.1 TFLOPS（= 标称的 101%），说明卡能跑在标称之上。
     所以正确的问题不是「attention 是否超过标称峰值」，
     而是「**attention 相对同一 session 里测到的 GEMM 上限是多少**」。
  4. 正确性：SDPA 的输出与手写参照是否一致（别在测一个退化的 kernel）

用法：
    $PY benchmarks/t6a_attn_flops_audit.py --device 0 --out results/E4/attn_flops_audit.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

REPO = os.environ.get("GENMODEL_ACCEL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S_FULL = 32760
H, D = 12, 128
PEAK_BF16_FP32ACC = 165.2      # Ada 白皮书 v2.1 dense，nominal boost clock
PEAK_BF16_FP16ACC = 330.3      # 同表，FP16 累加那一档


def bench_event(fn, warmup=5, iters=30, flush=None):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        if flush is not None:
            flush()
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def bench_perf(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="results/E4/attn_flops_audit.json")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(args.device)
    dev = f"cuda:{args.device}"
    p = torch.cuda.get_device_properties(args.device)
    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "device": p.name, "torch": torch.__version__,
               "trigger": "E4 reported dense SDPA at 164.8 TFLOPS = 99.8% of the 165.2 "
                          "spec peak. Rule 4 says a number pinned to a boundary is "
                          "suspect until the measurement is audited."}

    # ---- 1. FLOPs 公式 -------------------------------------------------
    payload["flops_formula"] = {
        "expression": "4 * B * H * S^2 * D",
        "derivation": "QK^T is [S,D]x[D,S] = 2*S^2*D MACs->FLOPs per head; "
                      "P@V is [S,S]x[S,D] = 2*S^2*D. Non-causal, no masking, "
                      "so no factor of 1/2.",
        "B": 1, "H": H, "S": S_FULL, "D": D,
        "total_tflop": 4 * 1 * H * S_FULL ** 2 * D / 1e12,
        "matches_e4_script": True,
        "note": "e4_svg.py used `2 * 2 * Sn * Sn * H * D` which is the same expression.",
    }
    print(f"[1] FLOPs = 4*B*H*S^2*D = {payload['flops_formula']['total_tflop']:.3f} TFLOP")

    # ---- 4. 正确性（先做，小规模） --------------------------------------
    Ss = 2048
    q = torch.randn(1, 2, Ss, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, 2, Ss, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, 2, Ss, D, device=dev, dtype=torch.bfloat16)
    ref = torch.matmul(F.softmax(
        (q.double() @ k.double().transpose(-2, -1)) / D ** 0.5, dim=-1), v.double())
    got = F.scaled_dot_product_attention(q, k, v).double()
    payload["correctness"] = {
        "S_checked": Ss,
        "rel_l2_vs_fp64_reference": float((got - ref).norm() / ref.norm()),
        "note": "confirms SDPA is computing full non-causal attention, not a degenerate "
                "or masked variant",
    }
    print(f"[4] SDPA vs fp64 reference (S={Ss}): "
          f"{payload['correctness']['rel_l2_vs_fp64_reference']:.2e}")
    del q, k, v, ref, got
    torch.cuda.empty_cache()

    # ---- 3. 同 session 的 GEMM 对照（决定性的一步）---------------------
    # 用 E0 的真实 DiT 形状里最大的那个，以及一个方阵，取同 session 的上限
    gemms = {}
    for tag, (m, n, kk) in {
            "ffn_up_32760x8960x1536": (32760, 8960, 1536),
            "ffn_down_32760x1536x8960": (32760, 1536, 8960),
            "square_8192": (8192, 8192, 8192),
            "square_16384": (16384, 16384, 16384)}.items():
        try:
            a = torch.randn(m, kk, device=dev, dtype=torch.bfloat16)
            b = torch.randn(kk, n, device=dev, dtype=torch.bfloat16)
            ms = bench_event(lambda: torch.mm(a, b), warmup=5, iters=20)
            gemms[tag] = {"ms": ms, "tflops": 2.0 * m * n * kk / (ms * 1e-3) / 1e12}
            del a, b
            torch.cuda.empty_cache()
        except Exception as e:
            gemms[tag] = {"err": f"{type(e).__name__}: {e}"}
    gmax = max(v["tflops"] for v in gemms.values() if "tflops" in v)
    payload["same_session_gemm"] = {"rows": gemms, "max_tflops": gmax,
                                    "pct_of_spec_peak": gmax / PEAK_BF16_FP32ACC}
    print(f"[3] same-session BF16 GEMM max: {gmax:.1f} TFLOPS "
          f"({gmax/PEAK_BF16_FP32ACC*100:.1f}% of the 165.2 spec)")
    for t, v in gemms.items():
        if "tflops" in v:
            print(f"      {t:28s} {v['tflops']:7.1f} TFLOPS")

    # ---- 2. attention 的计时，两种计时器 + 各 backend -------------------
    q = torch.randn(1, H, S_FULL, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, H, S_FULL, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, H, S_FULL, D, device=dev, dtype=torch.bfloat16)
    fl = payload["flops_formula"]["total_tflop"]

    rows = {}
    rows["default_event"] = bench_event(
        lambda: F.scaled_dot_product_attention(q, k, v), warmup=5, iters=20)
    rows["default_perf_counter"] = bench_perf(
        lambda: F.scaled_dot_product_attention(q, k, v), warmup=5, iters=20)
    # 大量 warmup / 大量 iters，看有没有时钟爬升效应
    rows["default_event_long"] = bench_event(
        lambda: F.scaled_dot_product_attention(q, k, v), warmup=20, iters=50)

    from torch.nn.attention import sdpa_kernel, SDPBackend
    for name, be in (("flash", SDPBackend.FLASH_ATTENTION),
                     ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                     ("cudnn", SDPBackend.CUDNN_ATTENTION)):
        try:
            with sdpa_kernel([be]):
                F.scaled_dot_product_attention(q, k, v)
                rows[f"backend_{name}"] = bench_event(
                    lambda: F.scaled_dot_product_attention(q, k, v),
                    warmup=5, iters=20)
        except Exception as e:
            rows[f"backend_{name}"] = None
            payload.setdefault("backend_errors", {})[name] = f"{type(e).__name__}"

    payload["attention_timing"] = {
        k2: ({"ms": v2, "tflops": fl / (v2 * 1e-3),
              "pct_of_spec_peak": (fl / (v2 * 1e-3)) / PEAK_BF16_FP32ACC,
              "pct_of_same_session_gemm": (fl / (v2 * 1e-3)) / gmax}
             if v2 else None)
        for k2, v2 in rows.items()}
    print("[2] attention timing:")
    for k2, v2 in payload["attention_timing"].items():
        if v2:
            print(f"      {k2:24s} {v2['ms']:6.2f} ms  {v2['tflops']:6.1f} TFLOPS  "
                  f"{v2['pct_of_spec_peak']*100:5.1f}% of spec  "
                  f"{v2['pct_of_same_session_gemm']*100:5.1f}% of same-session GEMM")
        else:
            print(f"      {k2:24s} unavailable")

    ok = [v for v in payload["attention_timing"].values() if v]
    best = min(v["ms"] for v in ok)
    tf = fl / (best * 1e-3)
    payload["verdict"] = {
        "attention_tflops": tf,
        "pct_of_spec_peak_165_2": tf / PEAK_BF16_FP32ACC,
        "pct_of_same_session_gemm_max": tf / gmax,
        "same_session_gemm_max_tflops": gmax,
        "exceeds_fp16acc_peak": tf > PEAK_BF16_FP16ACC,
        "interpretation": (
            "The right comparison is against the GEMM ceiling measured in the SAME "
            "session, not against the spec number: the spec is computed at nominal boost "
            "clock, and E0 already observed BF16 GEMM at 167.1 TFLOPS (101% of spec). "
            "If attention lands at or below the same-session GEMM max, the measurement is "
            "sound and the 'dense side is not being wasted' check stands."),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    v = payload["verdict"]
    print(f"\n{'='*70}")
    print(f"attention {tf:.1f} TFLOPS = {v['pct_of_spec_peak_165_2']*100:.1f}% of spec, "
          f"{v['pct_of_same_session_gemm_max']*100:.1f}% of same-session GEMM max "
          f"({gmax:.1f})")
    print(f"exceeds the FP16-accumulate peak ({PEAK_BF16_FP16ACC})? "
          f"{v['exceeds_fp16acc_peak']}")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
