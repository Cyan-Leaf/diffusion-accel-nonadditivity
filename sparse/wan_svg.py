#!/usr/bin/env python3
"""wan_svg.py — E4：SVG 的 spatial/temporal 结构化稀疏，接到 Wan2.1 上

## 血统与命名（重要，报告里要照抄）

本文件的**算法**（两个 mask 的定义、online 的 head 分类准则、temporal head 的
token 重排）**逐条抄自 SVG 官方实现**：
  `svg-project/Sparse-VideoGen` @ main
  - `svg/models/wan/utils.py::get_attention_mask`      -> 两个 mask
  - `svg/models/wan/attention.py::sample_mse`          -> online 分类
  - `svg/models/wan/placement.py::wan_token_reorder_*` -> temporal 的重排
（该仓库本机无法 clone；用 GitHub tree API + `raw.githubusercontent.com`
 逐文件拉下来的，171/178 个源文件，见 `results/T5_REPORT.md`。）

**换掉的只有 kernel 后端**：SVG 默认走 `flashinfer`（本环境未装），
本文件走 **`torch.nn.attention.flex_attention`** —— 那是 **SVG 自己的另一个后端**
（`svg/models/wan/attention.py::sparse_flex_attention`），不是我另找的实现。

→ 报告里的措辞：**"SVG's mask design and head-classification criterion, with
  FlexAttention as the kernel backend (one of SVG's own two backends)"**。
  不写「我们复现了 SVG」，也不与 SVG 报告的端到端数字直接比较。

## Wan2.1-1.3B @ 832x480x81 的几何

  num_frame  = 21          (latent frames)
  frame_size = 30*52 = 1560 (每帧的 patch 数)
  seq_len    = 21*1560 = 32760
  context_length = 0        (Wan 的 self-attn 不拼 text，text 走 cross-attn)

⚠️ SVG 的 Wan 路径是给 **diffusers 的 `WanTransformer3DModel`** 写的，那边
self-attn 也不拼 text，所以 `context_length=0` 这条与其一致。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Wan2.1-1.3B @ 832x480, 81 帧（出处见 research/gemm_shapes.json）
NUM_FRAME = 21
FRAME_SIZE = 30 * 52          # 1560
SEQ_LEN = NUM_FRAME * FRAME_SIZE
BLOCK = 128


# ============================ SVG 的两个 mask ================================

def build_dense_masks(num_frame=NUM_FRAME, frame_size=FRAME_SIZE, device="cuda",
                      max_row=None):
    """SVG `get_attention_mask` 的逐条复制（`svg/models/wan/utils.py:60-110`）。

    spatial：first-frame sink + 128 粒度的块带状（|i-j| < 2*frame_size/128 块）
    temporal：同一个带状 mask，但在 **token-major** 顺序下 ——
              即把 (frame, token) 两个轴换过来再看，等价于「重排后的滑窗」

    只用于 online 分类里的采样行，所以可以只建前 max_row 行。
    """
    n = num_frame * frame_size
    max_row = n if max_row is None else min(max_row, n)
    m = torch.zeros((n, n), dtype=torch.bool, device="cpu")
    m[:, :frame_size] = 1                                    # first frame sink
    block_thres = frame_size * 2
    nb = math.ceil(n / BLOCK)
    for i in range(nb):
        for j in range(nb):
            if abs(i - j) < block_thres // BLOCK:
                m[i * BLOCK:(i + 1) * BLOCK, j * BLOCK:(j + 1) * BLOCK] = 1
    spatial = m
    # temporal = 同一 mask 在 token-major 下的样子
    temporal = (m.reshape(frame_size, num_frame, frame_size, num_frame)
                .permute(1, 0, 3, 2)
                .reshape(n, n))
    return (spatial[:max_row].to(device).contiguous(),
            temporal[:max_row].to(device).contiguous())


def measured_sparsity(num_frame=NUM_FRAME, frame_size=FRAME_SIZE):
    """mask 的真实稀疏率（保留元素占比）。

    **通则 1 要求的交叉验证量**：稀疏 attention 的 kernel 加速比
    应与这个数的倒数同量级。对不上就是 kernel 效率问题，不是稀疏率问题。
    """
    n = num_frame * frame_size
    nb = math.ceil(n / BLOCK)
    band_blocks = 0
    thres = (frame_size * 2) // BLOCK
    for i in range(nb):
        for j in range(nb):
            if abs(i - j) < thres:
                band_blocks += 1
    band = band_blocks * BLOCK * BLOCK
    sink = n * frame_size
    # sink 与 band 有重叠：前 frame_size 列里落在带内的部分
    overlap = 0
    for i in range(nb):
        for j in range(nb):
            if abs(i - j) < thres and j * BLOCK < frame_size:
                lo, hi = j * BLOCK, min((j + 1) * BLOCK, frame_size)
                overlap += BLOCK * max(0, hi - lo)
    kept = band + sink - overlap
    return {"kept_frac": kept / (n * n), "n": n, "band_blocks": band_blocks,
            "blocks_total": nb * nb, "band_half_width_blocks": thres,
            "note": "block-granular (128) count of the SVG mask, sink+band minus overlap"}


# ======================= SVG 的 online head 分类 =============================

@torch.no_grad()
def sample_mse(q, k, v, masks, num_sampled_rows=64, sample_mse_max_row=10000,
               gen=None):
    """SVG `sample_mse` 的逐条复制（`svg/models/wan/attention.py:213-237`）。

    q/k/v: [B, H, S, D]。返回 [n_masks, B, H] 的 MSE。
    对每个 head，argmin 就是它的类别（0=spatial, 1=temporal）。
    """
    B, H, S, D = q.shape
    nrows = min(num_sampled_rows, S)
    hi = min(sample_mse_max_row, masks[0].shape[0], S)
    rows = torch.randint(low=0, high=hi, size=(nrows,), generator=gen)
    rows = rows.to(q.device)
    sq = q[:, :, rows, :]
    scores = torch.matmul(sq, k.transpose(-2, -1)) / (D ** 0.5)
    golden = torch.matmul(F.softmax(scores, dim=-1), v)
    out = torch.zeros(len(masks), B, H, device=q.device, dtype=q.dtype)
    for i, m in enumerate(masks):
        sm = m[rows, :]
        s = scores.masked_fill(sm == 0, float("-inf"))
        h = torch.matmul(F.softmax(s, dim=-1), v)
        out[i] = torch.mean((h - golden) ** 2, dim=(2, 3))
    return out


# ==================== temporal head 的 token 重排 ============================

def to_token_major(t, num_frame=NUM_FRAME, frame_size=FRAME_SIZE):
    """frame-major -> token-major（SVG `wan_token_reorder_to_token_major`，
    去掉了 fix_len 分支，因为 Wan 的 self-attn 里 context_length=0）。"""
    B, H, S, D = t.shape
    return (t.reshape(B, H, num_frame, frame_size, D)
             .transpose(2, 3).reshape(B, H, S, D))


def to_frame_major(t, num_frame=NUM_FRAME, frame_size=FRAME_SIZE):
    B, H, S, D = t.shape
    return (t.reshape(B, H, frame_size, num_frame, D)
             .transpose(2, 3).reshape(B, H, S, D))


# ========================== FlexAttention 后端 ===============================

def svg_mask_mod(num_frame=NUM_FRAME, frame_size=FRAME_SIZE):
    """与上面 dense mask 同一条规则，写成 FlexAttention 的 mask_mod。

    ⚠️ 必须与 `build_dense_masks` **逐位一致**，否则 online 分类用的 mask
    和实际跑的 mask 就不是一个东西。`e4_svg.py` 里有一个显式的一致性断言。
    """
    thres_blocks = (frame_size * 2) // BLOCK

    def mod(b, h, q_idx, kv_idx):
        sink = kv_idx < frame_size
        band = torch.abs(q_idx // BLOCK - kv_idx // BLOCK) < thres_blocks
        return sink | band
    return mod


_BM_CACHE = {}


def get_block_mask(S, device, num_frame=NUM_FRAME, frame_size=FRAME_SIZE):
    from torch.nn.attention.flex_attention import create_block_mask
    key = (S, str(device), num_frame, frame_size)
    if key not in _BM_CACHE:
        _BM_CACHE[key] = create_block_mask(
            svg_mask_mod(num_frame, frame_size), B=None, H=None, Q_LEN=S, KV_LEN=S,
            device=device)
    return _BM_CACHE[key]


_FLEX = None


def flex(q, k, v, block_mask):
    global _FLEX
    from torch.nn.attention.flex_attention import flex_attention
    if _FLEX is None:
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX(q, k, v, block_mask=block_mask)


@torch.no_grad()
def svg_attention(q, k, v, best_mask_idx, num_frame=NUM_FRAME,
                  frame_size=FRAME_SIZE):
    """一次 FlexAttention 跑完两类 head：temporal 的 q/k/v 先重排到 token-major，
    这样两类 head 在各自的顺序下用的是**同一个**带状 mask（SVG 的做法）。
    输出再把 temporal 的部分排回来。
    """
    B, H, S, D = q.shape
    idx = best_mask_idx.reshape(B, H)
    tmask = (idx == 1)
    qq, kk, vv = q.clone(), k.clone(), v.clone()
    if tmask.any():
        for b in range(B):
            sel = tmask[b]
            if sel.any():
                qq[b, sel] = to_token_major(q[b, sel].unsqueeze(0), num_frame,
                                            frame_size).squeeze(0)
                kk[b, sel] = to_token_major(k[b, sel].unsqueeze(0), num_frame,
                                            frame_size).squeeze(0)
                vv[b, sel] = to_token_major(v[b, sel].unsqueeze(0), num_frame,
                                            frame_size).squeeze(0)
    out = flex(qq, kk, vv, get_block_mask(S, q.device, num_frame, frame_size))
    if tmask.any():
        for b in range(B):
            sel = tmask[b]
            if sel.any():
                out[b, sel] = to_frame_major(out[b, sel].unsqueeze(0), num_frame,
                                             frame_size).squeeze(0)
    return out
