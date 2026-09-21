#!/usr/bin/env python3
"""make_contact_sheets.py — 给眼评用的带标注对照图（T12，负责人反馈第 1 条）

**为什么重做**：`results/frames/compare_p*_midframe.png` 是无标注的横向拼图，
**看图的人无法知道哪一格是哪个 arm**，于是 `REPORT.md` 附 A 第 1 条那 10 分钟眼评
根本没法做。这个脚本把标注补上。

每一格标出：
- **arm 名**，参照臂显式标 `REFERENCE`
- **该 arm 在这条 prompt 上的三个数**：像素 PSNR / latent 相对 L2 / VBench imaging
  —— 让眼评可以直接回答「数字说它差，眼睛同不同意」

**不做任何图像处理**：只抽中间帧、缩放、拼接、写字。不调色、不增强、不裁剪。

⚠️ **标签文字用英文**：PIL 的内置位图字体只有 ASCII；本机也没有中文 TTF
（与 `figures/make_figures.py` 同一个限制）。arm 名本来就是英文标识符。

用法：
    <PY> eval/make_contact_sheets.py --out results/frames
"""
from __future__ import annotations

import argparse
import json
import os

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.environ.get("GENMODEL_ACCEL_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDEOS = "results/E9/vbench_work"
REF = "bf16"

# 分两行，每行四格；顺序按「参照 → 量化（构图保持）→ 稀疏（构图可能改变）」
# 3 行 × 3 列：参照+量化 / 量化+NVFP4 / 稀疏三臂
ROWS = [
    ["bf16", "rtn_tensor", "gptq"],
    ["gptq_cd", "gptq_clip_cd", "nvfp4_w4a4"],
    ["bf16_sparse", "bf16_sparse_warm1", "fp8_sparse"],
]
GROUP = {"bf16": "REFERENCE", "rtn_tensor": "FP8 W8A8", "gptq": "FP8 W8A8",
         "gptq_cd": "FP8 W8A8", "gptq_clip_cd": "FP8 W8A8",
         "nvfp4_w4a4": "NVFP4 W4A4 (4 bit)",
         "bf16_sparse": "SPARSE, no warmup", "bf16_sparse_warm1": "SPARSE + 1 dense step",
         "fp8_sparse": "SPARSE + FP8, no warmup"}
# T14 §1：NVFP4 臂的指标存在另一个 json 里，且那里的 arm 名是 `gptq_cd`（与 FP8 臂撞名）
ALIAS = {"nvfp4_w4a4": ("results/E7/quality_nvfp4.json", "gptq_cd")}

TILE_W = 416            # 832 / 2
LABEL_H = 54
PAD = 6


def load_metrics():
    """{arm: {prompt_index: {psnr, l2, imaging}}}，缺的留 None。"""
    m = {}
    for f in ["results/E2/quality_pixel.json", "results/E2/quality_clip_ablation.json",
              "results/E4/sparse_quality.json"]:
        p = os.path.join(REPO, f)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for arm, r in d.get("metrics_vs_bf16", {}).items():
            for i, pr in enumerate(r.get("per_prompt", [])):
                m.setdefault(arm, {}).setdefault(i, {})
                m[arm][i]["psnr"] = pr.get("pixel_psnr_db")
                m[arm][i]["l2"] = pr.get("latent_rel_l2")
    # 别名臂（指标在别的 json 里、且 arm 名撞名）
    for arm, (path, key) in ALIAS.items():
        p2 = os.path.join(REPO, path)
        if not os.path.exists(p2):
            continue
        r = json.load(open(p2)).get("metrics_vs_bf16", {}).get(key, {})
        for i, pr in enumerate(r.get("per_prompt", [])):
            m.setdefault(arm, {}).setdefault(i, {})
            m[arm][i]["psnr"] = pr.get("pixel_psnr_db")
            m[arm][i]["l2"] = pr.get("latent_rel_l2")
    vb = os.path.join(REPO, "results/E9/vbench_pervideo.json")
    if os.path.exists(vb):
        d = json.load(open(vb))["per_video"]
        for arm, dims in d.items():
            pv = dims.get("imaging_quality", {}).get("per_video", {})
            for k, v in pv.items():
                i = int(k[1:])
                m.setdefault(arm, {}).setdefault(i, {})["imaging"] = v
    # VBench 逐视频：不在 pervideo.json 里的臂，直接读 VBench 的 eval_results
    for arm in ALIAS:
        f = os.path.join(REPO, f"results/E9/vbench_work/_results/{arm}_eval_results.json")
        if not os.path.exists(f):
            continue
        for row in json.load(open(f)).get("imaging_quality", [None, []])[1]:
            i = int(os.path.basename(row["video_path"]).rsplit("_p", 1)[1].split(".")[0])
            m.setdefault(arm, {}).setdefault(i, {})["imaging"] = row["video_results"] / 100.0
    return m


def mid_frame(arm: str, pi: int):
    f = os.path.join(REPO, VIDEOS, arm, f"{arm}_p{pi}.mp4")
    if not os.path.exists(f):
        return None
    c = av.open(f)
    fr = [x.to_ndarray(format="rgb24") for x in c.decode(video=0)]
    c.close()
    return fr[len(fr) // 2]


def tile(arm, pi, metrics, font, font_b):
    img = mid_frame(arm, pi)
    if img is None:
        return None
    h, w, _ = img.shape
    th = int(TILE_W * h / w)
    im = Image.fromarray(img).resize((TILE_W, th), Image.LANCZOS)

    canvas = Image.new("RGB", (TILE_W, th + LABEL_H), "white")
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)

    is_ref = arm == REF
    is_sparse = "sparse" in arm
    is_nvfp4 = arm.startswith("nvfp4")
    col = ((20, 90, 30) if is_ref else (150, 40, 30) if is_sparse
           else (140, 90, 10) if is_nvfp4 else (30, 60, 130))
    d.rectangle([0, th, TILE_W, th + LABEL_H], fill=(248, 248, 248))
    d.line([0, th, TILE_W, th], fill=col, width=3)
    d.text((6, th + 4), arm, fill=col, font=font_b)
    d.text((6, th + 20), GROUP.get(arm, ""), fill=col, font=font)

    mm = metrics.get(arm, {}).get(pi, {})
    bits = []
    if mm.get("psnr") is not None:
        bits.append(f"PSNR {mm['psnr']:.2f} dB")
    if mm.get("l2") is not None:
        bits.append(f"latent L2 {mm['l2']:.3f}")
    if mm.get("imaging") is not None:
        bits.append(f"VBench img {mm['imaging']:.4f}")
    txt = "  |  ".join(bits) if bits else "(reference: metrics are computed against it)"
    d.text((6, th + 36), txt, fill=(60, 60, 60), font=font)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/frames")
    ap.add_argument("--prompts", default="0,1,2")
    args = ap.parse_args()
    out = os.path.join(REPO, args.out)
    os.makedirs(out, exist_ok=True)
    metrics = load_metrics()
    try:
        font = ImageFont.load_default(13)
        font_b = ImageFont.load_default(15)
    except TypeError:                      # 老版本 PIL 的 load_default 不收尺寸
        font = font_b = ImageFont.load_default()

    for pi in [int(x) for x in args.prompts.split(",")]:
        grid_rows = []
        for row in ROWS:
            tiles = [t for t in (tile(a, pi, metrics, font, font_b) for a in row)
                     if t is not None]
            if not tiles:
                continue
            w = sum(t.width for t in tiles) + PAD * (len(tiles) - 1)
            h = max(t.height for t in tiles)
            strip = Image.new("RGB", (w, h), "white")
            x = 0
            for t in tiles:
                strip.paste(t, (x, 0))
                x += t.width + PAD
            grid_rows.append(strip)
        if not grid_rows:
            continue
        W = max(r.width for r in grid_rows)
        H = sum(r.height for r in grid_rows) + PAD * (len(grid_rows) - 1) + 30
        sheet = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(sheet)
        d.text((6, 8), f"prompt p{pi}  --  middle frame of each arm  "
                       f"(reference = {REF};  green = reference, "
                       f"blue = FP8 W8A8, orange = NVFP4 W4A4, red = sparse)", fill=(0, 0, 0), font=font_b)
        y = 30
        for r in grid_rows:
            sheet.paste(r, (0, y))
            y += r.height + PAD
        p = os.path.join(out, f"labelled_p{pi}_midframe.png")
        sheet.save(p)
        print(f"  wrote {os.path.relpath(p, REPO)}   ({sheet.width}x{sheet.height})")

    print("\n眼评要回答的三件事（REPORT.md 附 A 第 1 条）：")
    print("  (a) bf16 与四个 FP8 臂是否视觉上不可区分？")
    print("  (a2) ⭐ T14 §1：**NVFP4 W4A4（橙框）的构图是否保持？** "
          "决定 §8.6.3 读法 3（高频比 0.8585）可读还是不可读")
    print("  (b) bf16_sparse / fp8_sparse 是否是『另一个构图』而不是『画坏了』？")
    print("  (c) bf16_sparse_warm1 是否回到了与 bf16 相同的构图？")


if __name__ == "__main__":
    main()
