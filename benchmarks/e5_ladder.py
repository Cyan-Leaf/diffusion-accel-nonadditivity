#!/usr/bin/env python3
"""e5_ladder.py — T6-C：E5 的 2×2 阶梯，回答 H2（量化与稀疏是否争同一份收益）

## 预注册判据（**写在测量之前**，`T6_HANDOFF.md` §4 要求）

设单独开 FP8 的 denoise 加速为 `a`，单独开稀疏为 `b`，两者同开为 `c`。
定义 **交互比 I = c / (a·b)**。

  I ≳ 0.95        -> 作用在不同瓶颈，**可乘**
  I ≲ 0.90        -> **争同一份收益**，且要给出争的是哪一份
  0.90 < I < 0.95 -> 不下结论，报数

**两种结果都是结论，不预期哪一个。**

⚠️ 判据用 **denoise 段**，不是端到端：两条轴都只作用在 denoise 上，
端到端还叠了 VAE 这个固定成本，会把 I 机械地推向 1（Amdahl 的算术效应，
不是「不争收益」的证据）。端到端另报，但不用来判 H2。

## 结构

  bf16_dense    : compiled BF16 基线（T5-A 定的公平基线）
  fp8_dense     : E3 的全融合 FP8
  bf16_sparse   : SVG-mask 稀疏 attention
  fp8_sparse    : 两者同开

4 步与 50 步各跑一遍 —— **若 I 在两种步数下不同，那本身就是论点**。

用法：
    $PY benchmarks/e5_ladder.py --device 0 --out results/E5/ladder.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch

REPO = os.environ.get("GENMODEL_ACCEL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SF_ROOT = os.environ.get("SF_ROOT", "")   # 上游 Wan 推理工作目录（含 wan/、utils/、prompts/）
assert SF_ROOT, "请先 export SF_ROOT=<上游工作目录>"
DISTILL_CKPT = os.environ.get("DISTILL_CKPT", "")   # lightx2v 4 步蒸馏权重 (rev ef72050)
assert DISTILL_CKPT, "请先 export DISTILL_CKPT=<distill_native.pt 的路径>"
LATENT_SHAPE = (21, 16, 60, 104)
DSL = [1000, 750, 500, 250]
SEED = 242

# 预注册判据
I_MULTIPLICATIVE = 0.95
I_COMPETING = 0.90

sys.path.insert(0, REPO)
sys.path.insert(0, SF_ROOT)


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def sched_4():
    s = torch.linspace(1.0, 0.0, 1001)[:-1]
    sig = 5.0 * s / (1 + 4.0 * s)
    idx = [1000 - x for x in DSL]
    return sig[idx].tolist(), (sig * 1000)[idx].tolist()


# ---------------- SVG 接进 WanSelfAttention ---------------------------------

# SVG 的 dense warmup：前 N 个 timestep / 前 N 层走 dense（`wan_t2v_inference.py:88-94`）
#   num_fp_timesteps = floor(first_times_fp * num_inference_steps)
#   num_fp_layers    = floor(first_layers_fp * num_layers)
# SVG 自己的 Wan 脚本用 first_times_fp=0.2、first_layers_fp=0.03。
# ⚠️ **在 4 步下 floor(0.2*4) = 0** —— 即 SVG 自己的比例在少步设定下给出**零** dense 预热；
# 50 步下同一比例给 10 步。这正是 T5 对 H3 的更正里说的「粒度变粗」的实例。
SPARSE_STATE = {"step": 0, "n_fp_steps": 0, "n_fp_layers": 0}


def patch_self_attn_sparse(model, n_layers=30, dmasks=None, num_sampled_rows=64,
                           n_fp_steps=0, n_fp_layers=0):
    """把每个 block 的 `self_attn.forward` 换成 SVG 路径。

    ⚠️ 与 E3 的融合**不冲突**：E3 换的是 `WanAttentionBlock.forward`，
    这里换的是 `WanSelfAttention.forward`，是两个不同对象上的方法。
    q/k/v 仍然走 `Fp8LinearFused`（若已替换），只是 attention 本身变稀疏。
    """
    from sparse import wan_svg as S
    from wan.modules.model import rope_apply
    SPARSE_STATE["n_fp_steps"] = n_fp_steps
    SPARSE_STATE["n_fp_layers"] = n_fp_layers
    n = 0
    for i in range(n_layers):
        sa = model.get_submodule(f"blocks.{i}.self_attn")

        def mk(sa, layer_idx=i):
            def fwd(x, seq_lens, grid_sizes, freqs):
                b, s, nh, d = (x.shape[0], x.shape[1], sa.num_heads, sa.head_dim)
                q = sa.norm_q(sa.q(x)).view(b, s, nh, d)
                k = sa.norm_k(sa.k(x)).view(b, s, nh, d)
                v = sa.v(x).view(b, s, nh, d)
                q = rope_apply(q, grid_sizes, freqs)
                k = rope_apply(k, grid_sizes, freqs)
                # [B,S,H,D] -> [B,H,S,D]
                qh = q.permute(0, 2, 1, 3).contiguous()
                kh = k.permute(0, 2, 1, 3).contiguous()
                vh = v.permute(0, 2, 1, 3).contiguous()
                # SVG 的 dense warmup：高噪声的前几步 / 前几层走全 attention
                if (SPARSE_STATE["step"] < SPARSE_STATE["n_fp_steps"]
                        or layer_idx < SPARSE_STATE["n_fp_layers"]):
                    out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh)
                else:
                    mse = S.sample_mse(qh, kh, vh, dmasks,
                                       num_sampled_rows=num_sampled_rows)
                    best = torch.argmin(mse, dim=0)
                    out = S.svg_attention(qh, kh, vh, best)
                out = out.permute(0, 2, 1, 3).reshape(b, s, nh * d)
                return sa.o(out)
            return fwd
        sa.forward = mk(sa)
        n += 1
    return n


def build(dev, fp8: bool, sparse: bool, dmasks=None):
    from utils.wan_wrapper import WanDiffusionWrapper
    from fusion import wan_fused
    w = WanDiffusionWrapper(is_causal=False)
    sd = torch.load(DISTILL_CKPT, map_location="cpu", weights_only=False)
    miss, unexp = w.model.load_state_dict(sd, strict=False)
    assert not miss and not unexp
    del sd
    w = w.to(dev).to(torch.bfloat16).eval()
    w.model.requires_grad_(False)
    info = {}
    if fp8:
        info["fp8_linears"] = len(wan_fused.convert_fused(
            w.model, share_quant=True, fuse_bias=True, fuse_epilogue=True))
    else:
        info["bf16_compiled_blocks"] = wan_fused.convert_bf16_fused(w.model)
    if sparse:
        info["sparse_attn_layers"] = patch_self_attn_sparse(w.model, dmasks=dmasks)
    free()
    return w, info


def run_steps(w, cond, uncond, noise, dev, steps, guide):
    """steps=4 走 step-distill 闭式；steps=50 走 UniPC + 外部 CFG。"""
    if steps == 4:
        sig, ts = sched_4()
        lat = noise.to(device=dev, dtype=torch.float32)
        F = lat.shape[1]
        for i in range(4):
            t = torch.tensor(ts[i], device=dev).float().view(1, 1).expand(1, F)
            v, _ = w(lat.to(torch.bfloat16), cond, t)
            v = v.float()
            x0 = lat - sig[i] * v
            lat = x0 + sig[i + 1] * v if i < 3 else x0
        return lat
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1,
                                      use_dynamic_shifting=False)
    sch.set_timesteps(steps, device=dev, shift=5.0)
    lat = noise.to(device=dev, dtype=torch.float32)
    F = lat.shape[1]
    for i, t in enumerate(sch.timesteps):
        tt = t.to(dev).float().view(1, 1).expand(1, F)
        mi = lat.to(torch.bfloat16)
        vc, _ = w(mi, cond, tt)
        vu, _ = w(mi, uncond, tt)
        v = vu.float() + guide * (vc.float() - vu.float())
        lat = sch.step(v, t, lat, return_dict=False)[0].float()
    return lat


def bench(fn, reps):
    fn()
    torch.cuda.synchronize()
    ts = []
    r = None
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], ts, r


ARMS = [("bf16_dense", False, False), ("fp8_dense", True, False),
        ("bf16_sparse", False, True), ("fp8_sparse", True, True)]


def _disjoint_amdahl_check(a, b, c, st):
    """两条轴若作用在**不相交**的时间分数上，联合加速比是多少？

    设分数 p、q 不相交，单轴局部加速 s_p、s_q，记 A=p/s_p、B=q/s_q：
        1/a = (1-p) + A
        1/b = (1-q) + B
        1/c = (1-p-q) + A + B
    则 (1/a)(1/b) - (1/c) = (p-A)(q-B) >= 0
    ⟹ **c >= a*b 恒成立**：不相交的两条优化，联合加速比**必然不低于**两者之积。

    所以 I > 1 不是异常，是不相交的**签名**。
    真正会把 I 压到 1 以下的，是两条轴作用在**重叠**的分数上（SPEC 的 H2 假设）。

    这里用 E1/E3 实测的占比（GEMM 0.204、attention 0.525）反解 s_p、s_q，
    再算出「若完全不相交，I 应该是多少」，与实测的 I 比。
    """
    import json as _j
    try:
        e3 = _j.load(open(os.path.join(REPO, "results/E3/fusion_latency.json")))
        sh = e3["rows"]["bf16_compiled"]["profile"]["buckets_share"]
        p_, q_ = sh["gemm"], sh["attention"]
    except Exception:
        return {"disjoint_model": "unavailable (need results/E3/fusion_latency.json)"}
    A = 1 / a - (1 - p_)
    B = 1 / b - (1 - q_)
    if A <= 0 or B <= 0:
        return {"disjoint_model": "inconsistent (implied local speedup <= 0)"}
    c_pred = 1 / ((1 - p_ - q_) + A + B)
    I_pred = c_pred / (a * b)
    return {"disjoint_amdahl_model": {
        "gemm_share_p": p_, "attention_share_q": q_,
        "implied_local_speedup_gemm": p_ / A,
        "implied_local_speedup_attention": q_ / B,
        "predicted_c_if_perfectly_disjoint": c_pred,
        "predicted_I_if_perfectly_disjoint": I_pred,
        "measured_I": c / (a * b),
        "measured_minus_predicted": c / (a * b) - I_pred,
        "reading": "I>1 is the SIGNATURE of disjointness, not an anomaly: for two "
                   "optimisations on disjoint time fractions, c >= a*b always holds "
                   "((1/a)(1/b) - 1/c = (p-A)(q-B) >= 0). A measured I that matches the "
                   "disjoint prediction means the two axes do not overlap at all; "
                   "SPEC's H2 (they compete for the same attention time) is then refuted.",
    }}


def merge_and_judge(paths, out_path):
    """把多个单-arm 进程的结果合并，再判 H2。"""
    merged = None
    for pth in paths:
        d = json.load(open(pth))
        rows = d.pop("rows")          # ⚠️ 先取出来再当 merged —— `merged = d` 是别名，
        if merged is None:            #    直接 merged["rows"]={} 会把第一个文件自己的
            merged = d                #    rows 清空（第一版就是这么丢掉 bf16_dense 的）
            merged["rows"] = {}
            merged["merged_from"] = []
        merged["merged_from"].append(os.path.basename(pth))
        for st, r in rows.items():
            merged["rows"].setdefault(st, {}).update(r)
    merged["one_process_per_arm"] = True
    merged["why_one_process_per_arm"] = (
        "torch.compile guards and the CUDA caching allocator carry state across arms. "
        "Running all four in one process put bf16_sparse at a 20.31 GiB peak on a "
        "22.16 GiB card; memory pressure there would understate b and inflate I. "
        "Separate processes remove both the allocator and the compile-cache coupling.")
    for st, res in merged["rows"].items():
        if all(res.get(n[0], {}).get("ok") for n in ARMS):
            base = res["bf16_dense"]["denoise_s"]
            a = base / res["fp8_dense"]["denoise_s"]
            b = base / res["bf16_sparse"]["denoise_s"]
            c = base / res["fp8_sparse"]["denoise_s"]
            I = c / (a * b)
            merged.setdefault("H2", {})[st] = {
                "a_fp8_only": a, "b_sparse_only": b, "c_both": c,
                "product_ab": a * b, "interaction_I": I,
                "verdict": ("multiplicative" if I >= I_MULTIPLICATIVE else
                            "competing" if I <= I_COMPETING else "inconclusive"),
                "loss_vs_product_pct": 100 * (1 - I),
                "peak_gib": {n[0]: res[n[0]]["peak_alloc_gib"] for n in ARMS},
            }
            merged["H2"][st].update(_disjoint_amdahl_check(a, b, c, st))
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    for st, h in merged.get("H2", {}).items():
        print(f"H2 [{st}]  a={h['a_fp8_only']:.4f}  b={h['b_sparse_only']:.4f}  "
              f"a*b={h['product_ab']:.4f}  c={h['c_both']:.4f}  "
              f"I={h['interaction_I']:.4f}  => {h['verdict']}")
        print(f"        peaks: " + "  ".join(f"{k}={v:.2f}GiB" for k, v in h["peak_gib"].items()))
        dm = h.get("disjoint_amdahl_model")
        if isinstance(dm, dict):
            print(f"        disjoint model: implied local speedups "
                  f"gemm {dm['implied_local_speedup_gemm']:.3f}x / "
                  f"attn {dm['implied_local_speedup_attention']:.3f}x  -> "
                  f"predicted I={dm['predicted_I_if_perfectly_disjoint']:.4f} "
                  f"vs measured {dm['measured_I']:.4f} "
                  f"(diff {dm['measured_minus_predicted']:+.4f})")
    print(f"written: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--steps", default="4,50")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--reps-50", type=int, default=1)
    ap.add_argument("--out", default="results/E5/ladder.json")
    ap.add_argument("--arms", default="", help="只跑这些 arm（逗号分隔）。"
                    "⚠️ 强烈建议**每个 arm 一个独立进程**：torch.compile 的 guard "
                    "与 allocator 状态会跨 arm 污染。首次同进程跑 4 个 arm 时，"
                    "bf16_sparse 峰值 20.31 GiB（卡共 22.16），几乎必然有显存压力，"
                    "而那会**低估 b、抬高 I** —— 正是通则 4 要防的方向。")
    ap.add_argument("--merge", default="", help="把多个单-arm 的 json 合并并判 H2")
    args = ap.parse_args()

    if args.merge:
        merge_and_judge([x.strip() for x in args.merge.split(",") if x.strip()],
                        os.path.join(REPO, args.out))
        return

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = max(
        getattr(torch._dynamo.config, "cache_size_limit", 8), 256)
    torch.cuda.set_device(args.device)
    dev = f"cuda:{args.device}"
    p = torch.cuda.get_device_properties(args.device)
    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.chdir(SF_ROOT)

    from sparse import wan_svg as S
    dmasks = S.build_dense_masks(device=dev, max_row=10000)

    emb = torch.load(os.path.join(REPO, "results/E1/prompt_embeds.pt"),
                     weights_only=False)
    cond = {"prompt_embeds": emb["cond"].to(dev).to(torch.bfloat16)}
    uncond = {"prompt_embeds": emb["uncond"].to(dev).to(torch.bfloat16)}
    g = torch.Generator(device=dev).manual_seed(SEED)
    noise = torch.randn(1, *LATENT_SHAPE, device=dev, dtype=torch.float32, generator=g)

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": p.name, "torch": torch.__version__,
        "preregistered_criterion": {
            "quantity": "I = c / (a*b) on the DENOISE stage, where a = FP8-only "
                        "speedup, b = sparse-only speedup, c = both-on speedup",
            "multiplicative_if": f"I >= {I_MULTIPLICATIVE}",
            "competing_if": f"I <= {I_COMPETING}",
            "inconclusive_between": [I_COMPETING, I_MULTIPLICATIVE],
            "why_denoise_not_e2e": "both axes act only on denoise; the end-to-end number "
                                   "additionally carries the VAE fixed cost, which pushes "
                                   "I mechanically toward 1 by Amdahl arithmetic and would "
                                   "manufacture a false 'multiplicative' verdict.",
            "written_before_measurement": True,
        },
        "baseline_config": {
            "bf16_dense": "COMPILED bf16 (fusion.wan_fused.convert_bf16_fused) -- the "
                          "fair baseline established in T5-A, not eager",
            "fp8": "full E3 fusion (F1 bias epilogue + F2 shared q/k/v quant + F3 "
                   "norm/GELU epilogue), real torch._scaled_mm",
            "sparse": "SVG mask + sample_mse classification + FlexAttention backend, "
                      "online classification每层每步 (no caching), as SVG ships it",
        },
        "rows": {},
    }

    want = [x.strip() for x in args.arms.split(",") if x.strip()]
    for steps in [int(x) for x in args.steps.split(",")]:
        reps = args.reps if steps == 4 else args.reps_50
        res = {}
        for name, fp8, sparse in ARMS:
            if want and name not in want:
                continue
            key = f"{steps}step/{name}"
            try:
                w, info = build(dev, fp8, sparse, dmasks)
                med, ts, lat = bench(
                    lambda: run_steps(w, cond, uncond, noise, dev, steps, 5.0), reps)
                res[name] = {"ok": True, "denoise_s": med, "all_s": ts, "info": info,
                             "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                             "latent_finite": bool(torch.isfinite(lat).all()),
                             "latent_std": float(lat.std())}
                print(f"[{steps:2d}step {name:12s}] {med:8.3f} s  peak "
                      f"{res[name]['peak_alloc_gib']:.2f} GiB  {info}", flush=True)
                del w, lat
                free()
            except Exception as e:
                res[name] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:400]}"}
                print(f"[{steps:2d}step {name:12s}] FAILED: {type(e).__name__}: {e}",
                      flush=True)
                free()
        payload["rows"][f"{steps}step"] = res

        if all(res.get(n[0], {}).get("ok") for n in ARMS):
            base = res["bf16_dense"]["denoise_s"]
            a = base / res["fp8_dense"]["denoise_s"]
            b = base / res["bf16_sparse"]["denoise_s"]
            c = base / res["fp8_sparse"]["denoise_s"]
            I = c / (a * b)
            verdict = ("multiplicative" if I >= I_MULTIPLICATIVE else
                       "competing" if I <= I_COMPETING else "inconclusive")
            payload.setdefault("H2", {})[f"{steps}step"] = {
                "a_fp8_only": a, "b_sparse_only": b, "c_both": c,
                "product_ab": a * b, "interaction_I": I, "verdict": verdict,
                "loss_vs_product_pct": 100 * (1 - I),
            }
            print(f"   -> H2 [{steps} step]  a={a:.4f}  b={b:.4f}  a*b={a*b:.4f}  "
                  f"c={c:.4f}  I={I:.4f}  => {verdict}", flush=True)

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
