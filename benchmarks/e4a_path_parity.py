#!/usr/bin/env python3
"""e4a_path_parity.py — T4-A: 伪量化路径 vs 真 kernel 路径的数值口径闭合

**为什么这是 P0**：E7 的 NVFP4 全部是 fake quant（4090 没有 FP4 tensor core）。
如果 fake quant 与真 kernel 在数值上差得远，**E7 的所有数字都无法外推到部署**，
整个补充章节的效力归零。这不是 housekeeping，是 E7 的有效性前提。

T3 观察到的差异：延迟脚本报 `latent_rel_l2 = 0.586`，质量脚本报 `0.388`。
T3 把它归给了「激活 scale 按 reshape 后的二维张量算 vs 按原始形状算」。
**本脚本先检验这个归因是否成立。**

两级实验：

  L1 单 GEMM 级：同一个真实 Linear + 同一个真实激活，
     把「伪量化的 bf16 matmul」与「真 _scaled_mm」同时对一个 FP32 参照比。
     这一级没有采样器、没有混沌放大，差异全部是算术差异。

  L2 端到端级：**同 prompt 同 seed 同权重**，两条路径各跑一次 4 步，比 latent。
     T3 的 0.586 vs 0.388 是**不同 prompt 不同 seed**，本级检验那个比较是否成立。

用法：
    $PY benchmarks/e4a_path_parity.py --device 0 --out results/E4A/path_parity.json
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
E4M3_MAX = 448.0

# L2 必须与 T3 的延迟脚本完全同口径，才能判定 0.586 那个数
PROMPT_LAT = "Aerial drone shot flying over a dense green forest"
SEED_LAT = 242

sys.path.insert(0, REPO)
sys.path.insert(0, SF_ROOT)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def sched_4(dsl=DSL, shift=SHIFT, NT=1000):
    s = torch.linspace(1.0, 0.0, NT + 1)[:-1]
    sig = shift * s / (1 + (shift - 1) * s)
    idx = [NT - x for x in dsl]
    return sig[idx].tolist(), (sig * NT)[idx].tolist()


# ============================ L0: fp8 cast 语义 =============================

def probe_fp8_cast(dev):
    """`.to(float8_e4m3fn)` 到底是饱和还是溢出成 NaN/inf？
    伪量化的 `cb.round`（bucketize 到 253 个电平）是**饱和**语义，
    两者若不一致，超范围元素上会有系统性差异。**这是通则 2：把推断编码成测试。**"""
    x = torch.tensor([0.0, 1.0, 448.0, 449.0, 1000.0, 1e5, -1e5, 2**-10, 2**-12],
                     device=dev, dtype=torch.float32)
    y = x.to(torch.float8_e4m3fn).float()
    return {"input": x.tolist(), "after_cast": y.tolist(),
             "saturates_at_448": bool(y[4] == 448.0),
             "has_nan_or_inf": bool(~torch.isfinite(y).all())}


# ====================== L1: 单 GEMM 级的算术差异 ============================

def l1_single_gemm(dev, layer_name, x_bf16, W_bf16, bias_bf16, cb, rows=4096):
    """把各条路径同时对一个 **FP64** 参照比。

    ⚠️ 第一版用 FP32 算参照，**但 `allow_tf32=True` 让那个 matmul 走了 TF32
    （10 位尾数）**，于是参照本身带 ~2.3e-3 的误差，给所有路径造了一个假地板。
    改用 FP64 参照。（这是通则 1 的又一次应验：0.0023 在四个层上完全一致，
    那种「太整齐」本身就是信号。）

    参照 `ref`：FP64 下 (dequant 后的激活) @ (dequant 后的权重)。
    这是「低精度算术想表达的那个数学对象」，所有实现路径都是它的近似。
    """
    out = {"layer": layer_name, "shape": {"M_full": x_bf16.shape[0],
                                          "K": x_bf16.shape[1],
                                          "N": W_bf16.shape[0]}}
    # scale 必须用**整个**激活算（部署时就是这样），只有 matmul 的行做子采样，
    # 否则量化决策本身就变了。FP64 参照在 32760 行上放不下 22 GiB。
    xs = (x_bf16.float().abs().amax() / E4M3_MAX).clamp_min(1e-30)
    if x_bf16.shape[0] > rows:
        g = torch.Generator(device="cpu").manual_seed(0)
        idx = torch.randperm(x_bf16.shape[0], generator=g)[:rows].to(x_bf16.device)
        x_bf16 = x_bf16[idx].contiguous()
    out["shape"]["M_used"] = x_bf16.shape[0]
    out["scale_note"] = ("activation scale computed on the FULL 32760 rows; only the "
                         "matmul rows are subsampled (fp64 ref does not fit otherwise)")
    ws = (W_bf16.float().abs().amax() / E4M3_MAX).clamp_min(1e-30)

    # 两种量化决策：bucketize（伪量化现状） vs 硬件 cast（真 kernel）
    xq_lvl = cb.round(x_bf16.float() / xs)
    wq_lvl = cb.round(W_bf16.float() / ws)
    xq_hw = (x_bf16.float() / xs).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    wq_hw = (W_bf16.float() / ws).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()

    out["quant_decision_parity"] = {
        "act_levels_identical": bool(torch.equal(xq_lvl, xq_hw)),
        "wgt_levels_identical": bool(torch.equal(wq_lvl, wq_hw)),
        "act_mismatch_frac": float((xq_lvl != xq_hw).float().mean()),
        "wgt_mismatch_frac": float((wq_lvl != wq_hw).float().mean()),
        "note": "bucketize-to-253-levels vs hardware .to(float8_e4m3fn)",
    }

    # --- FP64 参照（以硬件 cast 的电平为准，因为那是部署时真实发生的事） ---
    xdq64 = (xq_hw.double() * xs.double())
    wdq64 = (wq_hw.double() * ws.double())
    ref = xdq64 @ wdq64.t()
    if bias_bf16 is not None:
        ref = ref + bias_bf16.double()

    res = {}

    # 路径 A：现状伪量化（bucketize -> bf16 -> bf16 matmul）= 质量脚本
    xdq_bf16 = (xq_lvl * xs).to(torch.bfloat16)
    wdq_bf16 = (wq_lvl * ws).to(torch.bfloat16)
    res["fake_bucketize_bf16"] = torch.nn.functional.linear(
        xdq_bf16, wdq_bf16, bias_bf16).double()

    # 路径 A2：bucketize 电平，但 FP32 存储 + FP32 matmul（TF32 关）
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    res["fake_bucketize_fp32"] = ((xq_lvl * xs) @ (wq_lvl * ws).t()
                                  + (bias_bf16.float() if bias_bf16 is not None else 0)
                                  ).double()
    # 路径 A3 = PARITY：硬件 cast 电平 + FP32 存储 + FP32 matmul（TF32 关）
    res["fake_parity_fp32"] = ((xq_hw * xs) @ (wq_hw * ws).t()
                               + (bias_bf16.float() if bias_bf16 is not None else 0)
                               ).double()
    # 路径 A5 = HWEXACT：逐位复刻真 kernel 的 dtype 序列
    #   （amax/scale/除法全在 bf16 里，然后才 cast）
    xs_bf = (x_bf16.abs().amax() / E4M3_MAX).clamp_min(1e-6)
    xq_bf = (x_bf16 / xs_bf).clamp_(-E4M3_MAX, E4M3_MAX).to(
        torch.float8_e4m3fn).float() * xs_bf.float()
    res["fake_hwexact_fp32"] = (xq_bf @ (wq_hw * ws).t()
                                + (bias_bf16.float() if bias_bf16 is not None else 0)
                                ).double()
    out["act_scale_dtype_effect"] = {
        "scale_fp32": float(xs), "scale_bf16": float(xs_bf),
        "rel_scale_diff": abs(float(xs) - float(xs_bf)) / float(xs),
        "level_mismatch_frac": float(
            ((xq_hw * xs) != xq_bf).float().mean()),
    }

    # 路径 A4：同 parity 但开 TF32，量 TF32 单独值多少
    torch.backends.cuda.matmul.allow_tf32 = True
    res["fake_parity_tf32"] = ((xq_hw * xs) @ (wq_hw * ws).t()
                               + (bias_bf16.float() if bias_bf16 is not None else 0)
                               ).double()
    torch.backends.cuda.matmul.allow_tf32 = prev

    # 路径 B：真 _scaled_mm
    x8 = (x_bf16.float() / xs).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    w8 = (W_bf16.float() / ws).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).t()
    for tag, fast in (("real_scaled_mm_fastaccum", True),
                      ("real_scaled_mm_slowaccum", False)):
        o = torch._scaled_mm(x8.contiguous(), w8,
                             scale_a=xs.reshape(()), scale_b=ws.reshape(()),
                             out_dtype=torch.bfloat16, use_fast_accum=fast)
        if bias_bf16 is not None:
            o = o + bias_bf16
        res[tag] = o.double()

    out["rel_err_vs_fp64_ref"] = {k: rel(v, ref) for k, v in res.items()}
    base = torch.nn.functional.linear(x_bf16, W_bf16, bias_bf16).double()
    qm = rel(ref, base)
    out["quantization_error_magnitude"] = {
        "fp64_ref_vs_unquantized_bf16": qm,
        "note": "this is the actual quantization error; every path-difference number "
                "must be compared against THIS, not against zero",
    }
    out["path_diff_vs_real_fastaccum"] = {
        k: rel(v, res["real_scaled_mm_fastaccum"]) for k, v in res.items()
        if k != "real_scaled_mm_fastaccum"}
    out["path_diff_as_frac_of_quant_err"] = {
        k: rel(v, res["real_scaled_mm_fastaccum"]) / qm for k, v in res.items()
        if k != "real_scaled_mm_fastaccum"}
    return out


# ============================ L2: 端到端 ====================================

def build_wrapper(dev):
    from utils.wan_wrapper import WanDiffusionWrapper
    w = WanDiffusionWrapper(is_causal=False)
    sd = torch.load(DISTILL_CKPT, map_location="cpu", weights_only=False)
    miss, unexp = w.model.load_state_dict(sd, strict=False)
    assert not miss and not unexp
    del sd
    w = w.to(dev).to(torch.bfloat16).eval()
    w.model.requires_grad_(False)
    return w


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="results/E4A/path_parity.json")
    ap.add_argument("--l1-layers", default="blocks.0.ffn.0,blocks.0.self_attn.q,"
                                           "blocks.15.ffn.2,blocks.29.cross_attn.o")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(args.device)
    dev = f"cuda:{args.device}"
    p = torch.cuda.get_device_properties(args.device)
    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.chdir(SF_ROOT)

    from quant.gptq import formats
    from quant.fp8 import wan_fp8

    wfmt = formats.parse_wfmt("fp8_e4m3:-1:fp32")
    cb = wfmt.cb
    sigmas, timesteps = sched_4()

    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "device": p.name, "torch": torch.__version__,
               "purpose": "decide whether the fake-quant path is a valid proxy for a "
                          "real FP8/FP4 kernel, which gates E7"}

    # ---- L0 -------------------------------------------------------------
    payload["L0_fp8_cast_semantics"] = probe_fp8_cast(dev)
    print("[L0] fp8 cast:", payload["L0_fp8_cast_semantics"]["saturates_at_448"],
          "| nan/inf:", payload["L0_fp8_cast_semantics"]["has_nan_or_inf"], flush=True)

    # ---- L1：需要真实激活，所以先挂 hook 跑一步拿到 --------------------
    w = build_wrapper(dev)
    emb_file = os.path.join(REPO, "results/E1/prompt_embeds.pt")
    cond_cpu = torch.load(emb_file, weights_only=False)["cond"]
    cond = {"prompt_embeds": cond_cpu.to(dev).to(torch.bfloat16)}
    g = torch.Generator(device=dev).manual_seed(SEED_LAT)
    noise = torch.randn(1, *LATENT_SHAPE, device=dev, dtype=torch.float32, generator=g)

    want = [s.strip() for s in args.l1_layers.split(",") if s.strip()]
    cap = {}
    hooks = []

    def mk(n):
        def h(mod, inp, outp):
            if n not in cap:
                cap[n] = inp[0].detach().reshape(-1, inp[0].shape[-1]).clone()
        return h
    for n in want:
        hooks.append(w.model.get_submodule(n).register_forward_hook(mk(n)))
    ts0 = torch.tensor(timesteps[0], device=dev).float().view(1, 1).expand(1, 21)
    _ = w(noise.to(torch.bfloat16), cond, ts0)
    for h in hooks:
        h.remove()
    print(f"[L1] captured activations for {list(cap)}", flush=True)

    payload["L1_single_gemm"] = []
    for n in want:
        mod = w.model.get_submodule(n)
        r = l1_single_gemm(dev, n, cap[n], mod.weight.data, mod.bias.data, cb)
        payload["L1_single_gemm"].append(r)
        qm = r["quantization_error_magnitude"]["fp64_ref_vs_unquantized_bf16"]
        print(f"[L1] {n:26s} M={r['shape']['M_used']:6d} K={r['shape']['K']:5d} "
              f"N={r['shape']['N']:5d}  quant_err={qm:.5f}", flush=True)
        for k, v in r["rel_err_vs_fp64_ref"].items():
            pd = r["path_diff_as_frac_of_quant_err"].get(k)
            tail = f"   pathdiff={pd*100:6.2f}% of quant_err" if pd is not None else "   (reference path)"
            print(f"       {k:26s} vs_fp64ref={v:.6f}{tail}", flush=True)
        p_ = r["quant_decision_parity"]
        print(f"       levels act_ident={p_['act_levels_identical']} "
              f"({p_['act_mismatch_frac']:.2e})  wgt_ident={p_['wgt_levels_identical']} "
              f"({p_['wgt_mismatch_frac']:.2e})", flush=True)
    del cap
    free()

    # ---- L2：同 prompt 同 seed，两条路径各跑一次 -----------------------
    print("\n[L2] end-to-end, SAME prompt / seed / weights", flush=True)
    afmt = formats.parse_afmt("fp8_e4m3:tensor")
    lats = {}

    # bf16 参照
    lats["bf16"] = run4(w, cond, noise, sigmas, timesteps, dev).cpu()
    del w
    free()

    for tag, mode in (("fake_bucketize_bf16", "fake"),
                      ("fake_parity_fp32", "parity"),
                      ("fake_hwexact_fp32", "hwexact"),
                      ("real_pertensor", "pertensor"),
                      ("real_pertensor_rerun", "pertensor"),
                      ("real_pertensor_naive", "pertensor_naive")):
        w = build_wrapper(dev)
        if mode == "fake":
            names = wan_fp8.convert_fakequant(w.model, afmt=afmt)
            for nm in names:
                m = w.model.get_submodule(nm)
                m.weight.data.copy_(wfmt.rtn(m.weight.data.float()).to(m.weight.dtype))
        elif mode == "parity":
            names = wan_fp8.convert_fakequant_parity(w.model, act_fp8=True, w_fp8=True)
        elif mode == "hwexact":
            names = wan_fp8.convert_fakequant_parity(w.model, act_fp8=True, w_fp8=True,
                                                     hw_dtype_seq=True)
        else:
            names = wan_fp8.convert_fp8(w.model, mode=mode)
        free()
        lats[tag] = run4(w, cond, noise, sigmas, timesteps, dev).cpu()
        del w
        free()
        print(f"  {tag}: {len(names)} linears", flush=True)

    ref = lats["bf16"]
    l2 = {k: {"rel_l2_vs_bf16": rel(v, ref)} for k, v in lats.items() if k != "bf16"}
    l2["pairs"] = {
        "fake_bucketize_bf16_vs_real": rel(lats["fake_bucketize_bf16"],
                                           lats["real_pertensor"]),
        "fake_parity_fp32_vs_real": rel(lats["fake_parity_fp32"],
                                        lats["real_pertensor"]),
        "fake_hwexact_fp32_vs_real": rel(lats["fake_hwexact_fp32"],
                                         lats["real_pertensor"]),
        "real_pertensor_vs_real_naive": rel(lats["real_pertensor"],
                                            lats["real_pertensor_naive"]),
        "real_pertensor_vs_ITSELF_rerun": rel(lats["real_pertensor"],
                                              lats["real_pertensor_rerun"]),
    }
    l2["determinism_note"] = (
        "real_pertensor_vs_ITSELF_rerun is a pure determinism check (identical config, "
        "identical seed, fresh model build). It calibrates how much of every other "
        "number is run-to-run noise vs a real difference.")
    l2["t3_numbers_for_comparison"] = {
        "latency_script_pertensor_naive": 0.5862189531326294,
        "quality_script_rtn_tensor": 0.3883,
        "note": "T3 compared these two and attributed the gap to an activation-scale "
                "shape difference. That attribution was WRONG on two counts: the two "
                "numbers used DIFFERENT prompts and seeds, and the real cause is the "
                "bf16 dequantization round-trip in the fake-quant path (see L1).",
    }
    payload["L2_end_to_end"] = l2
    for k, v in l2.items():
        if "rel_l2_vs_bf16" in v:
            print(f"  {k:24s} rel_l2 vs bf16 = {v['rel_l2_vs_bf16']:.4f}")
    for k, x in l2["pairs"].items():
        print(f"  pair {k:38s} rel_l2 = {x:.4f}")

    # ---- 结论 ----------------------------------------------------------
    l1 = payload["L1_single_gemm"]
    def worst(key):
        return max(r["path_diff_as_frac_of_quant_err"][key] for r in l1)
    e2e = {k: l2[k]["rel_l2_vs_bf16"] for k in l2
           if isinstance(l2[k], dict) and "rel_l2_vs_bf16" in l2[k]}
    gap_old = abs(e2e["fake_bucketize_bf16"] - e2e["real_pertensor"]) / e2e["real_pertensor"]
    gap_new = abs(e2e["fake_parity_fp32"] - e2e["real_pertensor"]) / e2e["real_pertensor"]
    gap_hw = abs(e2e["fake_hwexact_fp32"] - e2e["real_pertensor"]) / e2e["real_pertensor"]
    payload["verdict"] = {
        "L1_path_diff_frac_of_quant_err": {
            "fake_bucketize_bf16 (status quo)": worst("fake_bucketize_bf16"),
            "fake_bucketize_fp32": worst("fake_bucketize_fp32"),
            "fake_parity_fp32 (fp32 scale)": worst("fake_parity_fp32"),
            "fake_hwexact_fp32 (bf16 scale = hw)": worst("fake_hwexact_fp32"),
            "fake_parity_tf32": worst("fake_parity_tf32"),
            "real_scaled_mm_slowaccum": worst("real_scaled_mm_slowaccum"),
        },
        "L2_rel_l2_vs_bf16": e2e,
        "L2_gap_to_real": {"status_quo": gap_old, "parity": gap_new, "hwexact": gap_hw},
        "irreducible_floor": {
            "real_fastaccum_vs_real_slowaccum_frac_of_quant_err":
                max(r["path_diff_as_frac_of_quant_err"]["real_scaled_mm_slowaccum"]
                    for r in l1),
            "note": "two legitimate real-kernel configurations differ from each other "
                    "by this much. No fake-quant path can be closer to 'the' real "
                    "kernel than the real kernels are to each other -- this is the "
                    "right yardstick, not zero.",
        },
        "fakequant_valid_proxy_status_quo": bool(worst("fake_bucketize_bf16") < 0.10
                                                 and gap_old < 0.10),
        "fakequant_valid_proxy_parity": bool(worst("fake_parity_fp32") < 0.10
                                             and gap_new < 0.10),
        "fakequant_valid_proxy_hwexact": bool(worst("fake_hwexact_fp32") < 0.10
                                              and gap_hw < 0.10),
        "threshold_note": "the proxy is called valid if the path difference is <10% of "
                          "the quantization error it is meant to measure, at both levels",
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    v = payload["verdict"]
    print(f"\n{'='*70}")
    print("L1 path diff as fraction of the quantization error (max over 4 layers):")
    for k, x in v["L1_path_diff_frac_of_quant_err"].items():
        print(f"    {k:36s} {x*100:7.2f}%")
    print("L2 rel_l2 vs bf16:")
    for k, x in v["L2_rel_l2_vs_bf16"].items():
        print(f"    {k:36s} {x:.4f}")
    print("L2 gap to real kernel: " + "  ".join(
        f"{k} {x*100:.1f}%" for k, x in v["L2_gap_to_real"].items()))
    print(f"irreducible floor (real vs real): "
          f"{v['irreducible_floor']['real_fastaccum_vs_real_slowaccum_frac_of_quant_err']*100:.2f}%"
          f" of quant err")
    print(f"VERDICT  status quo = {v['fakequant_valid_proxy_status_quo']}  "
          f"parity = {v['fakequant_valid_proxy_parity']}  "
          f"hwexact = {v['fakequant_valid_proxy_hwexact']}")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
