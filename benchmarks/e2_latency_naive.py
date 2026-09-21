#!/usr/bin/env python3
"""e2_latency_naive.py — E2: 朴素（未融合）W8A8 的实测延迟

填 `SPEC.md` §4.2 三数结构的中间那格：

    天花板 1.895×   (E0, 裸 GEMM)                      ← 已有
        ↓  差额 = quant/dequant 的访存代价
    朴素      ?  ×  (本脚本)                            ← 要测的
        ↓  差额 = E3 消除掉的部分
    融合      ?  ×  (E3, 下一轮)

测四档，全部真 `torch._scaled_mm`（不是伪量化）：

  bf16               基线
  fp8 pertensor_naive  quant 用朴素 eager 算子（3 遍 activation）—— 真正的「朴素」
  fp8 pertensor        quant 用 torch.compile 融成 1 遍 —— 前人 ship 的形态，
                       已经是部分融合，所以是朴素与 E3 之间的中间档
  fp8 pertok           per-token scale，sm_89 上只能 out_dtype=fp32 再乘 scale

同时报 per-step 和端到端（含 VAE 固定成本），因为 E1 已经证明
4 步下 VAE 占 50%，只报 per-step 会把端到端收益夸大一倍。

用法：
    $PY benchmarks/e2_latency_naive.py --device 0 --out results/E2/latency_naive.json
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
SHIFT = 5.0
PROMPT = "Aerial drone shot flying over a dense green forest"
SEED = 242

sys.path.insert(0, REPO)
sys.path.insert(0, SF_ROOT)


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def sched_4(dsl=DSL, shift=SHIFT, NT=1000):
    s = torch.linspace(1.0, 0.0, NT + 1)[:-1]
    sig = shift * s / (1 + (shift - 1) * s)
    idx = [NT - x for x in dsl]
    return sig[idx].tolist(), (sig * NT)[idx].tolist()


def build(dev, mode, n_layers=30):
    from utils.wan_wrapper import WanDiffusionWrapper
    from quant.fp8 import wan_fp8
    w = WanDiffusionWrapper(is_causal=False)
    sd = torch.load(DISTILL_CKPT, map_location="cpu", weights_only=False)
    miss, unexp = w.model.load_state_dict(sd, strict=False)
    assert not miss and not unexp
    del sd
    w = w.to(dev).to(torch.bfloat16).eval()
    w.model.requires_grad_(False)
    n = 0
    if mode != "bf16":
        n = len(wan_fp8.convert_fp8(w.model, mode=mode, n_layers=n_layers))
    free()
    return w, n


def run4(w, cond, noise, sigmas, timesteps, dev):
    lat = noise.to(device=dev, dtype=torch.float32)
    F = lat.shape[1]
    for i in range(len(sigmas)):
        ts = torch.tensor(timesteps[i], device=dev).float().view(1, 1).expand(1, F)
        v, _ = w(lat.to(torch.bfloat16), cond, ts)
        v = v.float()
        x0 = lat - sigmas[i] * v
        lat = x0 + sigmas[i + 1] * v if i < len(sigmas) - 1 else x0
    return lat


def bench(fn, reps):
    fn()  # warmup（torch.compile 的那两档第一次要编译，必须排除）
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], ts, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--modes", default="bf16,pertensor_naive,pertensor,pertok")
    ap.add_argument("--out", default="results/E2/latency_naive.json")
    ap.add_argument("--e1", default="results/E1/stage_profile.json")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = max(
        getattr(torch._dynamo.config, "cache_size_limit", 8), 64)
    torch.cuda.set_device(args.device)
    dev = f"cuda:{args.device}"
    p = torch.cuda.get_device_properties(args.device)

    out_path = os.path.join(REPO, args.out)
    e1_path = os.path.join(REPO, args.e1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.chdir(SF_ROOT)

    # 复用 E1 的 text embedding，保证同一 prompt / 同一口径
    emb_file = os.path.join(REPO, "results/E1/prompt_embeds.pt")
    if os.path.exists(emb_file):
        cond_cpu = torch.load(emb_file, weights_only=False)["cond"]
        emb_src = "results/E1/prompt_embeds.pt"
    else:
        from utils.wan_wrapper import WanTextEncoder
        enc = WanTextEncoder().eval().requires_grad_(False).to(torch.bfloat16).to(dev)
        cond_cpu = enc(text_prompts=[PROMPT])["prompt_embeds"].to(torch.bfloat16).cpu()
        del enc
        free()
        emb_src = "freshly encoded"

    sigmas, timesteps = sched_4()
    g = torch.Generator(device=dev).manual_seed(SEED)
    noise = torch.randn(1, *LATENT_SHAPE, device=dev, dtype=torch.float32, generator=g)
    cond = {"prompt_embeds": cond_cpu.to(dev).to(torch.bfloat16)}

    rows = {}
    lat_ref = None
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        try:
            w, nlin = build(dev, mode)
        except Exception as e:
            rows[mode] = {"ok": False, "err": f"{type(e).__name__}: {e}"}
            print(f"[{mode}] BUILD FAILED: {e}", flush=True)
            continue
        wbytes = sum(b.numel() * b.element_size()
                     for b in list(w.model.parameters()) + list(w.model.buffers()))
        try:
            med, ts, lat = bench(lambda: run4(w, cond, noise, sigmas, timesteps, dev),
                                 args.reps)
            r = {"ok": True, "n_fp8_linears": nlin,
                 "denoise_s": med, "denoise_s_all": ts,
                 "ms_per_forward": med * 1e3 / 4,
                 "weight_bytes": wbytes, "weight_gib": wbytes / 2**30,
                 "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "latent_stats": {"mean": float(lat.mean()), "std": float(lat.std()),
                                  "finite": bool(torch.isfinite(lat).all())}}
            if mode == "bf16":
                lat_ref = lat.float().cpu()
            elif lat_ref is not None:
                d = (lat.float().cpu() - lat_ref)
                r["latent_rel_l2_vs_bf16"] = float(d.norm() / lat_ref.norm())
            rows[mode] = r
            print(f"[{mode}] denoise {med:.3f}s  {med*1e3/4:.0f} ms/fwd  "
                  f"weights {wbytes/2**30:.2f} GiB  peak "
                  f"{r['peak_alloc_gib']:.2f} GiB  ({nlin} fp8 linears)", flush=True)
        except Exception as e:
            rows[mode] = {"ok": False, "n_fp8_linears": nlin,
                          "err": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"[{mode}] BENCH FAILED: {type(e).__name__}: {e}", flush=True)
        del w
        free()

    # ---- 加速比与三数结构 ----------------------------------------------
    base = rows.get("bf16", {}).get("denoise_s")
    for k, v in rows.items():
        if v.get("ok") and base:
            v["denoise_speedup_vs_bf16"] = base / v["denoise_s"]

    # 端到端：denoise + VAE（固定成本，取自 E1 实测）+ text
    e2e = {}
    if os.path.exists(e1_path):
        e1 = json.load(open(e1_path))
        t_text = e1["text_encode"]["ms_single_prompt"] / 1e3
        for vdt in ("float32", "bfloat16"):
            key = f"4step/{vdt}"
            if key not in e1.get("vae", {}):
                continue
            t_vae = e1["vae"][key]["decode_s"]
            tot_b = None
            for mode, v in rows.items():
                if not v.get("ok"):
                    continue
                tot = t_text + v["denoise_s"] + t_vae
                if mode == "bf16":
                    tot_b = tot
                e2e[f"{mode}/vae_{vdt}"] = {"total_s": tot, "vae_s": t_vae,
                                            "text_s": t_text,
                                            "denoise_s": v["denoise_s"]}
            if tot_b:
                for k in list(e2e):
                    if k.endswith(f"vae_{vdt}"):
                        e2e[k]["e2e_speedup_vs_bf16"] = tot_b / e2e[k]["total_s"]

    three = None
    if base:
        cand = {m: rows[m].get("denoise_speedup_vs_bf16")
                for m in ("pertensor_naive", "pertensor", "pertok")
                if rows.get(m, {}).get("ok")}
        three = {
            "ceiling_gemm_only_E0": 1.895,
            "naive_w8a8_denoise": cand.get("pertensor_naive"),
            "compiled_quant_denoise": cand.get("pertensor"),
            "pertok_denoise": cand.get("pertok"),
            "fused_E3": None,
            "gap_ceiling_to_naive": (1.895 - cand["pertensor_naive"])
            if cand.get("pertensor_naive") else None,
            "gap_naive_to_compiled": (cand["pertensor"] - cand["pertensor_naive"])
            if cand.get("pertensor") and cand.get("pertensor_naive") else None,
            "note": "gap_ceiling_to_naive is the memory cost of un-fused quant/dequant; "
                    "gap_naive_to_compiled is the part torch.compile already recovers by "
                    "collapsing 3 activation passes into 1. E3 attacks the remainder.",
        }

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": p.name, "sm": f"{p.major}{p.minor}",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "config": {"steps": 4, "denoising_step_list": DSL, "shift": SHIFT,
                   "cfg": None, "forwards": 4, "seq_len": 32760,
                   "prompt": PROMPT, "seed": SEED, "embeds_from": emb_src},
        "kernel": "torch._scaled_mm (real FP8 GEMM), use_fast_accum=True",
        "rows": rows,
        "e2e": e2e,
        "three_number_structure": three,
        "caveat": "Weights here are RTN-quantized inside Fp8Linear (latency does not "
                  "depend on which solver produced them). Quality is in "
                  "results/E2/quant_quality.json.",
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*70}")
    for m, v in rows.items():
        if v.get("ok"):
            print(f"{m:20s} {v['denoise_s']:7.3f} s  "
                  f"{v.get('denoise_speedup_vs_bf16', 1):.3f}x (denoise)")
        else:
            print(f"{m:20s} FAILED: {v['err'][:70]}")
    for k, v in e2e.items():
        print(f"  e2e {k:28s} {v['total_s']:7.2f} s  "
              f"{v.get('e2e_speedup_vs_bf16', 1):.3f}x")
    if three:
        print(f"\nthree-number: ceiling {three['ceiling_gemm_only_E0']:.3f}x -> "
              f"naive {three['naive_w8a8_denoise']}x -> "
              f"compiled {three['compiled_quant_denoise']}x -> fused(E3) TBD")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
