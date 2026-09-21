#!/usr/bin/env python3
"""preflight.py — 可选工具。证明这台机器算得对，再谈算得快。

【什么时候跑】
本项目的机器已在上一个项目验证过，**默认不必跑**。
只在以下情况跑：换卡、换机器、或出现无法解释的数值异常。

【为什么存在】
上一个项目在一块有缺陷 SM 的 RTX 4090 上，y=2x+1 在 1.678e7 个元素里
错了 8070 个。不报错、不崩溃，只是安静地算错（Xid 13），静默污染了 0.32% 的数据。
性能测试本身不会暴露算术错误——这就是它存在的理由。

用法:
    $PY scripts/preflight.py
    $PY scripts/preflight.py --devices 0,1,2,3 --out results/T0/preflight.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

try:
    import torch
except ImportError:
    print("FATAL: torch not installed", file=sys.stderr)
    sys.exit(2)


N_ELEM = 1 << 24          # 1.678e7，与上次踩坑时同量级
GEMM_N = 4096
RTOL_BF16 = 2e-2


def _res(name, ok, detail):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    return {"check": name, "ok": bool(ok), "detail": detail}


def check_arith(dev):
    """逐元素算术正确性。这是抓缺陷 SM 的主力。"""
    out = []
    torch.cuda.set_device(dev)
    d = f"cuda:{dev}"

    # 1) y = 2x + 1，fp32，精确可比
    x = torch.arange(N_ELEM, dtype=torch.float32, device=d)
    y = 2 * x + 1
    expect = torch.arange(N_ELEM, dtype=torch.float32, device=d) * 2 + 1
    bad = int((y != expect).sum().item())
    out.append(_res(f"dev{dev}.elementwise_exact", bad == 0,
                    f"{bad}/{N_ELEM} mismatched"))

    # 2) 重复执行的自一致性（缺陷 SM 往往是间歇性的）
    flaky = 0
    for _ in range(8):
        y2 = 2 * x + 1
        flaky += int((y2 != expect).sum().item())
    out.append(_res(f"dev{dev}.elementwise_repeat", flaky == 0,
                    f"{flaky} mismatches over 8 reps"))

    # 3) GEMM 对拍：fp32 vs fp64 参考
    a = torch.randn(GEMM_N, GEMM_N, device=d, dtype=torch.float32)
    b = torch.randn(GEMM_N, GEMM_N, device=d, dtype=torch.float32)
    c32 = (a @ b).double()
    c64 = a.double() @ b.double()
    rel = ((c32 - c64).abs() / (c64.abs() + 1e-6)).max().item()
    out.append(_res(f"dev{dev}.gemm_fp32_vs_fp64", rel < 1e-3,
                    f"max rel err {rel:.3e}"))

    # 4) reduction 一致性
    s1 = x.sum().item()
    s2 = x.double().sum().item()
    rel_s = abs(s1 - s2) / abs(s2)
    out.append(_res(f"dev{dev}.reduction", rel_s < 1e-3,
                    f"rel err {rel_s:.3e}"))

    del x, y, expect, a, b, c32, c64
    torch.cuda.empty_cache()
    return out


def check_cross_device(devs):
    """同一输入在多卡上应逐位相同。不同 => 至少一张卡有问题。"""
    if len(devs) < 2:
        return [_res("cross_device", True, "single device, skipped")]
    torch.manual_seed(0)
    ref_cpu = torch.randn(2048, 2048)
    results = {}
    for dv in devs:
        x = ref_cpu.to(f"cuda:{dv}")
        results[dv] = (x @ x).cpu()
    base = results[devs[0]]
    out = []
    for dv in devs[1:]:
        same = torch.equal(base, results[dv])
        diff = int((base != results[dv]).sum().item())
        out.append(_res(f"cross_device.{devs[0]}_vs_{dv}", same,
                        f"{diff} elements differ"))
    return out


def check_fp8(dev):
    """FP8 可用性 —— 本项目主线精度载体。"""
    out = []
    d = f"cuda:{dev}"
    has_e4m3 = hasattr(torch, "float8_e4m3fn")
    out.append(_res(f"dev{dev}.fp8_dtype", has_e4m3,
                    "torch.float8_e4m3fn " + ("available" if has_e4m3 else "MISSING")))
    if not has_e4m3:
        return out

    try:
        x = torch.randn(512, 512, device=d, dtype=torch.bfloat16)
        xq = x.to(torch.float8_e4m3fn)
        xb = xq.to(torch.bfloat16)
        rel = ((x - xb).abs() / (x.abs() + 1e-3)).mean().item()
        # E4M3 有 3 位尾数，平均相对误差应在百分位量级
        out.append(_res(f"dev{dev}.fp8_cast", rel < 0.15,
                        f"mean rel err {rel:.4f}"))
    except Exception as e:
        out.append(_res(f"dev{dev}.fp8_cast", False, f"{type(e).__name__}: {e}"))
        return out

    # scaled_mm：FP8 GEMM 的入口。失败不致命，但 E0 要知道。
    try:
        a = torch.randn(512, 512, device=d, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = torch.randn(512, 512, device=d, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
        sa = torch.tensor(1.0, device=d)
        sb = torch.tensor(1.0, device=d)
        _ = torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        out.append(_res(f"dev{dev}.fp8_scaled_mm", True, "torch._scaled_mm OK"))
    except Exception as e:
        out.append(_res(f"dev{dev}.fp8_scaled_mm", False,
                        f"{type(e).__name__}: {e} "
                        f"(NOT fatal for preflight, but E0 must note it)"))
    torch.cuda.empty_cache()
    return out


def check_env(devs):
    out = []
    out.append(_res("torch_version", True, torch.__version__))
    out.append(_res("cuda_version", True, str(torch.version.cuda)))
    for dv in devs:
        p = torch.cuda.get_device_properties(dv)
        ok_arch = p.major == 8 and p.minor == 9   # Ada = sm_89
        out.append(_res(f"dev{dv}.arch", ok_arch,
                        f"{p.name} sm_{p.major}{p.minor}, "
                        f"{p.total_memory/2**30:.1f} GiB, {p.multi_processor_count} SMs"))
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,"
             "clocks_throttle_reasons.active,ecc.errors.uncorrected.volatile.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        out.append(_res("nvidia_smi", q.returncode == 0, q.stdout.strip() or q.stderr.strip()))
    except Exception as e:
        out.append(_res("nvidia_smi", False, str(e)))
    return out


def dev_str(d):
    return str(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default=None, help="e.g. 0,1,2,3")
    ap.add_argument("--out", default="results/T0/preflight.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: CUDA not available", file=sys.stderr)
        sys.exit(2)

    devs = ([int(x) for x in args.devices.split(",")] if args.devices
            else list(range(torch.cuda.device_count())))

    print(f"preflight: {len(devs)} device(s) {devs}\n")
    results = []

    print("[env]")
    results += check_env(devs)
    print("\n[arithmetic correctness]  <- this is the one that matters")
    for dv in devs:
        results += check_arith(dv)
    print("\n[cross-device consistency]")
    results += check_cross_device(devs)
    print("\n[fp8 availability]")
    for dv in devs:
        results += check_fp8(dv)

    n_fail = sum(1 for r in results if not r["ok"])
    # scaled_mm 失败单独归类，不阻塞
    hard_fail = [r for r in results if not r["ok"] and "scaled_mm" not in r["check"]]

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "devices": devs,
        "results": results,
        "n_fail": n_fail,
        "n_hard_fail": len(hard_fail),
        "verdict": "PASS" if not hard_fail else "FAIL",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*60}")
    print(f"VERDICT: {payload['verdict']}  ({n_fail} fail, {len(hard_fail)} hard)")
    print(f"written: {args.out}")
    if hard_fail:
        print("\nHARD FAILURES — do NOT run perf tests on this config:")
        for r in hard_fail:
            print(f"  - {r['check']}: {r['detail']}")
        print("\nIf arithmetic checks failed on one device, exclude that device")
        print("and rerun with --devices. Report which one to the owner.")
    sys.exit(0 if not hard_fail else 1)


if __name__ == "__main__":
    main()
