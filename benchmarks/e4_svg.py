#!/usr/bin/env python3
"""e4_svg.py — E4：SVG-mask 结构化稀疏 attention 的实测

**判据按 `T5_HANDOFF.md` §2.1(c) 的新版**（因为 kernel 后端不是 SVG 默认的
flashinfer，端到端不与 SVG 报告的数字直接比）：

  主读出   : attention kernel 自身加速比  vs  实测稀疏率
  交叉验证 : kernel 加速比应与稀疏率倒数**同量级**（通则 1）
  端到端   : 由 Amdahl 算术合成，并与实测对照

必测两项（handoff 反复强调，容易漏）：
  1. **online 分类（sample_mse）的单独耗时** —— H3 的摊销分母在 4 步下只有 50 步的 1/12.5
  2. **head 分类占比** —— 自测，不引用 HunyuanVideo 的 29.2/66.7/4.1

用法：
    $PY benchmarks/e4_svg.py --device 0 --out results/E4/svg_latency.json
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

sys.path.insert(0, REPO)
sys.path.insert(0, SF_ROOT)


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def bench(fn, warmup=3, iters=10):
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


def sched_4():
    s = torch.linspace(1.0, 0.0, 1001)[:-1]
    sig = 5.0 * s / (1 + 4.0 * s)
    idx = [1000 - x for x in DSL]
    return sig[idx].tolist(), (sig * 1000)[idx].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="results/E4/svg_latency.json")
    ap.add_argument("--layers", default="0,7,15,22,29",
                    help="在哪些 block 上抓真实 q/k/v 做 head 分类")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(args.device)
    dev = f"cuda:{args.device}"
    p = torch.cuda.get_device_properties(args.device)
    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.chdir(SF_ROOT)

    from sparse import wan_svg as S

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": p.name, "torch": torch.__version__,
        "provenance": {
            "algorithm": "SVG official implementation (svg-project/Sparse-VideoGen @ main): "
                         "mask design (svg/models/wan/utils.py::get_attention_mask), "
                         "online head classification (svg/models/wan/attention.py::sample_mse), "
                         "temporal token reorder (svg/models/wan/placement.py)",
            "kernel_backend": "torch.nn.attention.flex_attention -- which is SVG's OWN "
                              "second backend (sparse_flex_attention); flashinfer is not "
                              "installed in this environment",
            "naming": "report as 'SVG's mask design and head-classification criterion with "
                      "FlexAttention backend'. Do NOT claim to have reproduced SVG, and do "
                      "NOT compare directly against SVG's reported end-to-end numbers.",
            "how_obtained": "repo could not be cloned on this host (any transfer >3MB is "
                            "truncated by the proxy); fetched via GitHub tree API + "
                            "raw.githubusercontent.com file by file, 171/178 source files",
        },
        "geometry": {"num_frame": S.NUM_FRAME, "frame_size": S.FRAME_SIZE,
                     "seq_len": S.SEQ_LEN, "heads": 12, "head_dim": 128,
                     "context_length": 0,
                     "note": "Wan self-attn does not concatenate text; text goes through "
                             "cross-attn. Same as SVG's diffusers-Wan path."},
    }

    # ---------------- 1. 稀疏率（通则 1 的交叉验证量）--------------------
    sp = S.measured_sparsity()
    payload["sparsity"] = sp
    print(f"[sparsity] kept {sp['kept_frac']*100:.2f}%  "
          f"-> ideal kernel speedup ~{1/sp['kept_frac']:.2f}x  "
          f"(band half-width {sp['band_half_width_blocks']} blocks of 128)", flush=True)

    # ---------------- 2. mask 一致性断言 ---------------------------------
    # online 分类用的 dense mask 与 FlexAttention 跑的 mask_mod 必须逐位一致，
    # 否则「分类」和「执行」用的不是同一个东西。**这是通则 1 的自检。**
    nrow = 512
    dm_s, dm_t = S.build_dense_masks(device=dev, max_row=nrow)
    mod = S.svg_mask_mod()
    qi = torch.arange(nrow, device=dev).view(-1, 1)
    ki = torch.arange(S.SEQ_LEN, device=dev).view(1, -1)
    flexmask = mod(None, None, qi, ki)
    agree = bool(torch.equal(flexmask, dm_s))
    payload["mask_consistency"] = {
        "flex_mask_mod_equals_dense_spatial_mask": agree,
        "rows_checked": nrow,
        "mismatch_frac": float((flexmask != dm_s).float().mean()),
    }
    print(f"[mask] flex mask_mod == dense spatial mask: {agree} "
          f"(mismatch {payload['mask_consistency']['mismatch_frac']*100:.4f}%)", flush=True)

    # ---------------- 3. kernel 级加速比 ---------------------------------
    B, H, Sn, D = 1, 12, S.SEQ_LEN, 128
    q = torch.randn(B, H, Sn, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, H, Sn, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, H, Sn, D, device=dev, dtype=torch.bfloat16)
    bm = S.get_block_mask(Sn, dev)

    t_dense = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
    t_flex = bench(lambda: S.flex(q, k, v, bm))
    idx_all_sp = torch.zeros(B, H, dtype=torch.long, device=dev)
    idx_half = torch.tensor([[0] * (H // 2) + [1] * (H - H // 2)], device=dev)
    t_svg_sp = bench(lambda: S.svg_attention(q, k, v, idx_all_sp))
    t_svg_mix = bench(lambda: S.svg_attention(q, k, v, idx_half))

    attn_flops = 2 * 2 * Sn * Sn * H * D          # QK^T + AV
    payload["kernel"] = {
        "shape": {"B": B, "H": H, "S": Sn, "D": D},
        "dense_sdpa_ms": t_dense,
        "flex_sparse_ms": t_flex,
        "svg_all_spatial_ms": t_svg_sp,
        "svg_half_temporal_ms": t_svg_mix,
        "kernel_speedup_flex_vs_dense": t_dense / t_flex,
        "kernel_speedup_svg_all_spatial": t_dense / t_svg_sp,
        "kernel_speedup_svg_half_temporal": t_dense / t_svg_mix,
        "ideal_from_sparsity": 1 / sp["kept_frac"],
        "efficiency_vs_ideal": (t_dense / t_flex) / (1 / sp["kept_frac"]),
        "dense_tflops": attn_flops / (t_dense * 1e-3) / 1e12,
        "reorder_overhead_ms": t_svg_mix - t_flex,
    }
    kk = payload["kernel"]
    print(f"[kernel] dense {t_dense:.1f} ms ({kk['dense_tflops']:.0f} TFLOPS)  "
          f"flex-sparse {t_flex:.1f} ms  -> {kk['kernel_speedup_flex_vs_dense']:.2f}x", flush=True)
    print(f"         ideal from sparsity {kk['ideal_from_sparsity']:.2f}x  "
          f"-> kernel efficiency {kk['efficiency_vs_ideal']*100:.0f}%", flush=True)
    print(f"         svg(all spatial) {t_svg_sp:.1f} ms  svg(half temporal) "
          f"{t_svg_mix:.1f} ms  reorder overhead {kk['reorder_overhead_ms']:.1f} ms", flush=True)

    # ---------------- 4. online 分类的单独耗时（H3）----------------------
    dmask = S.build_dense_masks(device=dev, max_row=10000)
    t_prof = bench(lambda: S.sample_mse(q, k, v, dmask), warmup=2, iters=5)
    payload["online_profiling"] = {
        "sample_mse_ms_per_layer_per_step": t_prof,
        "num_sampled_rows": 64, "sample_mse_max_row": 10000,
        "cost_per_step_all_30_layers_ms": t_prof * 30,
        "dense_attn_ms_per_layer": t_dense,
        "profiling_as_frac_of_dense_attn": t_prof / t_dense,
        "profiling_as_frac_of_sparse_attn": t_prof / t_flex,
    }
    op = payload["online_profiling"]
    print(f"[H3] sample_mse {t_prof:.1f} ms/layer/step  "
          f"= {op['profiling_as_frac_of_sparse_attn']*100:.1f}% of the sparse attn it enables",
          flush=True)
    del q, k, v
    free()

    # ---------------- 5. 真实 q/k/v 上的 head 分类 -----------------------
    from utils.wan_wrapper import WanDiffusionWrapper
    w = WanDiffusionWrapper(is_causal=False)
    sd = torch.load(DISTILL_CKPT, map_location="cpu", weights_only=False)
    miss, unexp = w.model.load_state_dict(sd, strict=False)
    assert not miss and not unexp
    del sd
    w = w.to(dev).to(torch.bfloat16).eval()
    w.model.requires_grad_(False)

    cond = {"prompt_embeds": torch.load(
        os.path.join(REPO, "results/E1/prompt_embeds.pt"),
        weights_only=False)["cond"].to(dev).to(torch.bfloat16)}
    g = torch.Generator(device=dev).manual_seed(SEED)
    noise = torch.randn(1, *LATENT_SHAPE, device=dev, dtype=torch.float32, generator=g)
    sigmas, timesteps = sched_4()

    want = [int(x) for x in args.layers.split(",")]
    cls = {}
    gen = torch.Generator().manual_seed(0)

    # 在 4 个 step 上都抓，这样能看跨 step 的稳定性（E6 的起点）
    lat = noise.clone()
    for si in range(4):
        cap = {}
        hooks = []

        def mk(li):
            def h(mod, inp, outp):
                # WanSelfAttention.forward 的 q/k/v 在内部，改抓 q/k/v Linear 的输出
                pass
            return h
        # 直接抓三个 linear 的输出，自己拼 [B,H,S,D]
        store = {}

        def mk_lin(li, which):
            def h(mod, inp, outp):
                store[(li, which)] = outp.detach()
            return h
        for li in want:
            blk = w.model.get_submodule(f"blocks.{li}")
            for which in ("q", "k", "v"):
                hooks.append(getattr(blk.self_attn, which)
                             .register_forward_hook(mk_lin(li, which)))
        ts = torch.tensor(timesteps[si], device=dev).float().view(1, 1).expand(1, 21)
        vpred, _ = w(lat.to(torch.bfloat16), cond, ts)
        for h in hooks:
            h.remove()
        vpred = vpred.float()
        x0 = lat - sigmas[si] * vpred
        lat = x0 + sigmas[si + 1] * vpred if si < 3 else x0

        for li in want:
            qh = store[(li, "q")].reshape(1, S.SEQ_LEN, 12, 128).permute(0, 2, 1, 3)
            kh = store[(li, "k")].reshape(1, S.SEQ_LEN, 12, 128).permute(0, 2, 1, 3)
            vh = store[(li, "v")].reshape(1, S.SEQ_LEN, 12, 128).permute(0, 2, 1, 3)
            mse = S.sample_mse(qh.contiguous(), kh.contiguous(), vh.contiguous(),
                               dmask, gen=gen)
            best = torch.argmin(mse, dim=0)[0]          # [H]
            # 「ambiguous」的操作化：两个 mask 的 MSE 相差不到 10% 就算难分
            m0, m1 = mse[0, 0].float(), mse[1, 0].float()
            ratio = torch.maximum(m0, m1) / torch.minimum(m0, m1).clamp_min(1e-30)
            cls[f"step{si}_layer{li}"] = {
                "best": best.tolist(),
                "n_spatial": int((best == 0).sum()), "n_temporal": int((best == 1).sum()),
                "mse_spatial": m0.tolist(), "mse_temporal": m1.tolist(),
                "mse_ratio": ratio.tolist(),
                "n_ambiguous_within_10pct": int((ratio < 1.10).sum()),
            }
        store.clear()
        free()
    del w
    free()

    tot_s = sum(c["n_spatial"] for c in cls.values())
    tot_t = sum(c["n_temporal"] for c in cls.values())
    tot_a = sum(c["n_ambiguous_within_10pct"] for c in cls.values())
    tot = tot_s + tot_t
    payload["head_classification"] = {
        "per_step_layer": cls,
        "layers_probed": want, "steps_probed": 4, "heads_per_layer": 12,
        "total_head_step_pairs": tot,
        "spatial_frac": tot_s / tot, "temporal_frac": tot_t / tot,
        "ambiguous_frac_mse_within_10pct": tot_a / tot,
        "note": "SVG's shipped code has only TWO classes (assert len(attention_masks)==2); "
                "'ambiguous' is not a class it emits. We operationalise it as 'the two "
                "masks' MSE differ by less than 10%', i.e. the argmin is not decisive. "
                "Do NOT compare these numbers to HunyuanVideo's 29.2/66.7/4.1.",
    }
    hc = payload["head_classification"]
    print(f"[heads] spatial {hc['spatial_frac']*100:.1f}%  temporal "
          f"{hc['temporal_frac']*100:.1f}%  (of {tot} head-step pairs)", flush=True)
    print(f"        not decisive (MSE within 10%): "
          f"{hc['ambiguous_frac_mse_within_10pct']*100:.1f}%", flush=True)

    # 跨 step 的一致率（E6 的起点）
    consist = {}
    for li in want:
        seqs = [cls[f"step{si}_layer{li}"]["best"] for si in range(4)]
        same = [sum(1 for a, b in zip(seqs[i], seqs[i + 1]) if a == b) / len(seqs[i])
                for i in range(3)]
        consist[f"layer{li}"] = {"adjacent_step_agreement": same,
                                 "mean": sum(same) / len(same)}
    payload["cross_step_consistency"] = {
        "per_layer": consist,
        "mean_over_layers": sum(c["mean"] for c in consist.values()) / len(consist),
        "note": "E6's starting point: how stable is the head classification across the "
                "4 denoising steps. High agreement => a static table could replace online "
                "profiling.",
    }
    print(f"[E6 pilot] adjacent-step classification agreement: "
          f"{payload['cross_step_consistency']['mean_over_layers']*100:.1f}%", flush=True)

    # ---------------- 6. 端到端的 Amdahl 合成 ---------------------------
    e1 = os.path.join(REPO, "results/E1/stage_profile.json")
    if os.path.exists(e1):
        d = json.load(open(e1))
        attn_share = d["denoise_4step"]["profile"]["buckets_share"]["attention"]
        t_den = d["denoise_4step"]["denoise_wall_s"]
        t_txt = d["text_encode"]["ms_single_prompt"] / 1e3
        ksp = kk["kernel_speedup_svg_half_temporal"]
        # profiling 开销要加回去：每层每步一次
        prof_frac = op["cost_per_step_all_30_layers_ms"] / 1e3 * 4 / t_den
        f = (1 - attn_share) + attn_share / ksp + prof_frac
        comp = {"attention_share_of_denoise": attn_share,
                "kernel_speedup_used": ksp,
                "profiling_overhead_frac_of_denoise": prof_frac,
                "denoise_ceiling_with_profiling": 1 / f,
                "denoise_ceiling_without_profiling":
                    1 / ((1 - attn_share) + attn_share / ksp)}
        for vdt in ("float32", "bfloat16"):
            key = f"4step/{vdt}"
            if key in d.get("vae", {}):
                t_vae = d["vae"][key]["decode_s"]
                base = t_txt + t_den + t_vae
                comp[f"e2e_{vdt}"] = base / (t_txt + t_den * f + t_vae)
        payload["e2e_amdahl_composition"] = comp
        print(f"[e2e] attention is {attn_share*100:.1f}% of denoise; kernel {ksp:.2f}x "
              f"-> denoise {comp['denoise_ceiling_with_profiling']:.3f}x "
              f"(without profiling overhead {comp['denoise_ceiling_without_profiling']:.3f}x)")
        for vdt in ("float32", "bfloat16"):
            if f"e2e_{vdt}" in comp:
                print(f"       e2e (VAE {vdt}) {comp[f'e2e_{vdt}']:.3f}x")

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
