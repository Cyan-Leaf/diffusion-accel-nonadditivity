#!/usr/bin/env python3
"""vbench_run.py — 在本项目的各 arm 视频上跑官方 VBench（T7 §7.5.2 的 gate + T8 的执行体）

## 权重来源与可得性（gate 的结论）

VBench 的检测器权重**不需要下载** —— 前一个项目已经落盘在
`<SF_ROOT>/dumps/vbench_assets/`，本脚本用软链把它们摆成 VBench 期望的布局。
本机 >3 MB 的网络传输会被代理截断，所以这条是 T8 能否进行的**唯一**决定因素。

| 维度 | 检测器 | 本地权重 | 可用 |
|---|---|---|---|
| `imaging_quality` | MUSIQ (SPAQ) | `musiq_spaq.pth` (108 MB) | ✅ |
| `aesthetic_quality` | CLIP ViT-L/14 + LAION linear | `clip_vit_l14.pt` (932 MB) + `aesthetic_linear.pth` | ✅ |
| `subject_consistency` | DINO **ViT-B/16** | 只有 **ViT-S/16**（`dino_deitsmall16`，384 维） | ❌ |

⚠️ **`subject_consistency` 不评**：磁盘上的是 deit-small16（embed_dim 384），
而 VBench 本地模式要的是 `dino_vitb16`（768）。用小模型顶替会得到一个
**不是 VBench 官方定义的 subject_consistency 分数**，那比不报更糟。
非本地模式要从 GitHub 拉 DINO，本机拉不动。**如实写"该维度因检测器不可得而未评"。**

## ⚠️ 分数的可比性边界（报告里必须写）

**拉过来的是基础设施（权重 / 代码），不是可比的分数。**
前一个项目的 VBench 数来自另一个模型、另一组 prompt、另一种视频规格。
VBench 分数的可比性依赖**完全相同的 prompt 集 + seed + 视频规格**，
**跨项目直接比分数是错的**。本项目只在**自己的 arm 之间**比。

用法（必须用 venv_vbench，它里面 pin 了 transformers 4.33.2）：
    VBENCH_CACHE_DIR=<repo>/.vbench_cache \\
    <sf>/venv_vbench/bin/python eval/vbench_run.py --videos results/E2/videos --out ...

⚠️ **编号**：本实验是 **E9**，产物在 `results/E9/`。
原先落在 `results/E8/`，但 `SPEC.md` §3 的 E8 是「INT8 质量对照」，**两个不同的东西共用了
一个编号**（T8 §4 指出）。已迁到 E9 并在 SPEC §3 矩阵补行。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time

REPO = os.environ.get("GENMODEL_ACCEL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SF = os.environ.get("SF", "")   # 上游 Wan 推理工作目录（含 wan/、utils/、prompts/）
assert SF, "请先 export SF=<上游工作目录>"
ASSETS = f"{SF}/dumps/vbench_assets"
CACHE = f"{REPO}/.vbench_cache"

# 只列**权重齐全**的维度。subject_consistency 见文件头。
DIMS_AVAILABLE = ["aesthetic_quality", "imaging_quality"]
DIMS_UNAVAILABLE = {
    "subject_consistency": "needs DINO ViT-B/16 (768-dim); only ViT-S/16 "
                           "(dino_deitsmall16, 384-dim) is on disk, and the non-local "
                           "path needs a GitHub fetch that this host cannot complete",
}
# ⚠️ `aesthetic_linear.pth` in the prior project's assets is a DIFFERENT predictor
# (an MLP 768->1024->128->...), not VBench's `sa_0_4_vit_l_14_linear.pth` (a single
# Linear 768->1). Loading it raises a state_dict mismatch. The correct head is 4 KB and
# DID download successfully -- it is under the ~3 MB truncation threshold of this host.
AES_HEAD_NOTE = ("the LAION linear head (sa_0_4_vit_l_14_linear.pth, 4071 B) was fetched "
                 "directly; the aesthetic_linear.pth in the prior project's assets is a "
                 "different (MLP) predictor and is NOT interchangeable")


def setup_cache():
    """把已落盘的权重摆成 VBench 期望的布局（软链，不复制）。"""
    links = {
        f"{CACHE}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth": f"{ASSETS}/musiq_spaq.pth",
        f"{CACHE}/clip_model/ViT-L-14.pt": f"{ASSETS}/clip_vit_l14.pt",
    }
    # aesthetic 的 linear 头单独处理，见 AES_HEAD_NOTE
    aes = f"{CACHE}/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"
    missing = []
    for dst, src in links.items():
        if not os.path.exists(src):
            missing.append(src)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    if not os.path.exists(aes):
        missing.append(aes + " (LAION sa_0_4_vit_l_14_linear.pth, 4 KB)")
    return links, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="results/E2/videos",
                    help="含 <arm>_p<i>.mp4 的目录")
    ap.add_argument("--extra-videos", default="results/E4/videos",
                    help="第二个视频目录（稀疏臂），一并纳入")
    ap.add_argument("--out", default="results/E9/vbench.json")
    ap.add_argument("--work", default="results/E9/vbench_work")
    ap.add_argument("--dims", default=",".join(DIMS_AVAILABLE))
    ap.add_argument("--gate-only", action="store_true",
                    help="只查权重可得性，不跑评测（T7 §7.5.2）")
    args = ap.parse_args()

    links, missing = setup_cache()
    gate = {
        "weights_source": ASSETS,
        "cache_dir": CACHE,
        "links": {os.path.relpath(k, REPO): v for k, v in links.items()},
        "missing_sources": missing,
        "dimensions_available": DIMS_AVAILABLE,
        "dimensions_unavailable": DIMS_UNAVAILABLE,
        "aesthetic_head_note": AES_HEAD_NOTE,
        "comparability_note": (
            "Infrastructure (weights, code) was reused from the author's prior project. "
            "SCORES ARE NOT COMPARABLE ACROSS PROJECTS: VBench scores depend on the exact "
            "prompt set, seeds and video specification. We compare only across arms "
            "within this project's own run."),
    }
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    if args.gate_only:
        out = os.path.join(REPO, args.out.replace(".json", "_gate.json"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(gate, open(out, "w"), indent=2)
        print(f"\ngate written: {out}")
        return
    if missing:
        sys.exit(f"FATAL: missing weight sources: {missing}")

    # VBench 的 custom_input 模式要求「一个目录 = 一个 arm」，所以按 arm 分目录
    work = os.path.join(REPO, args.work)
    os.makedirs(work, exist_ok=True)
    arms = {}
    for vd in [args.videos, args.extra_videos]:
        d = os.path.join(REPO, vd)
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "*.mp4"))):
            base = os.path.basename(f)
            arm = base.rsplit("_p", 1)[0]
            ad = os.path.join(work, arm)
            os.makedirs(ad, exist_ok=True)
            dst = os.path.join(ad, base)
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
            arms.setdefault(arm, []).append(dst)
    print(f"\n{len(arms)} arms: " + ", ".join(f"{k}({len(v)})" for k, v in arms.items()),
          flush=True)

    import torch
    from vbench import VBench
    import vbench as _vb
    info_json = os.path.join(os.path.dirname(_vb.__file__), "VBench_full_info.json")
    out_dir = os.path.join(work, "_results")
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device("cuda:0")
    dims = [d.strip() for d in args.dims.split(",") if d.strip()]

    scores = {}
    for arm, files in sorted(arms.items()):
        rj = os.path.join(out_dir, f"{arm}_eval_results.json")
        if not os.path.exists(rj):
            print(f"===== {arm} ({len(files)} videos) =====", flush=True)
            vb = VBench(dev, info_json, out_dir)
            vb.evaluate(videos_path=os.path.join(work, arm), name=arm,
                        dimension_list=dims, mode="custom_input")
        res = json.load(open(rj))
        scores[arm] = {d: res[d][0] for d in dims if d in res}
        print(f"{arm}: {scores[arm]}", flush=True)

    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "gate": gate, "dimensions_run": dims, "n_videos_per_arm":
                   {k: len(v) for k, v in arms.items()},
               "scores": scores}
    out = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
