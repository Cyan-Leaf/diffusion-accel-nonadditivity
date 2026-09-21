#!/usr/bin/env python3
"""wan_fused.py — E3 P0：把 quant 融进上游 epilogue、把 dequant/bias 融进 GEMM epilogue

`SPEC.md` §4.3 的 P0 两项。目标：把 T3 测到的 **1.009×** 往 Amdahl 上界 **1.105×** 推。
（P1 五项——adaLN+LN 合并、QKV 合一、RoPE prologue 等——本轮不做。）

## 朴素 W8A8 在这个模型上到底多读写了几遍

以一个 block 为例（M=32760，dim=1536，ffn_dim=8960，bf16）：

  self-attn 入口   norm1(x)*(1+e1)+e0  ->  写 100 MB bf16
                   q/k/v 三个 Fp8Linear 各自：
                     读 100 MB 求 amax        x3
                     读 100 MB 除+cast 写 33 MB  x3
                   -> **同一个张量被量化了三遍，amax 也算了三遍**
  ffn 中间         GELU(ffn.0 out)     ->  写 **587 MB** bf16
                   ffn.2 读 587 MB 求 amax + 再读 587 MB 除+cast
  每个 linear 的 bias  ->  额外一遍 [M,N] 的读+写

## 本文件做的三件事

  F1  **bias 融进 `_scaled_mm` 的 epilogue**（原生支持），省掉每个 linear 一遍
      [M,N] 的读+写。ffn.0 的 [32760,8960] 一遍就是 587 MB。
  F2  **q/k/v 共享一次量化**：三者输入是同一个张量对象，用一个挂在张量上的
      cache 让 amax + cast 只做一次。省 2/3 的入口量化开销，且不改 block 结构。
  F3  **上游 epilogue 融合**：把 `norm+modulate+quant` 与 `gelu+quant` 各自
      用 `torch.compile` 折成一遍，直接产出 fp8，不再落 bf16 中间张量。
      这一项需要替换 `WanAttentionBlock.forward`。

F1/F2 不改数值语义（F2 是把重复计算去掉，结果逐位相同）。
F3 改变的是中间张量的存储精度（norm 的输出不再落 bf16 而是直接 fp8），
**这会让数值与未融合版不同**——量级在 T4-A 测出的分辨率下限之内，但必须报。
"""
from __future__ import annotations

import torch
import torch.nn as nn

E4M3_MAX = 448.0


# ----------------------------- F1 + F2 --------------------------------------

@torch.compile(dynamic=True)
def _quant_pertensor(x):
    sc = (x.abs().amax() / E4M3_MAX).clamp_min(1e-6)
    return (x / sc).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), sc.float()


class Fp8Carry:
    """只带 shape 和已量化好的 (x8, scale) 的轻量载体。

    为什么要它：`WanSelfAttention.forward` 对输入只用了 `x.shape[:2]`，
    然后把 x 原样递给 q/k/v（`model.py:142-148`）。
    如果上游已经把量化做完了，就没必要再物化一个 [32760,1536] 的 bf16 中间张量
    ——那正是 F3 想省掉的那一遍写。
    """
    __slots__ = ("shape", "fp8", "dtype", "device")

    def __init__(self, shape, x8, xs, dtype, device):
        self.shape = shape
        self.fp8 = (x8, xs)
        self.dtype = dtype
        self.device = device


class Fp8LinearFused(nn.Module):
    """F1：bias 走 `_scaled_mm` 的原生 epilogue。
    F2：q/k/v 共享一次量化（三者拿到的是**同一个张量对象**，按 id 缓存）。
    F3：若上游递来的是 `Fp8Carry`，直接用它的 fp8，完全跳过入口量化。

    F1/F2 不改数值语义（F2 只是去掉重复计算，逐位相同）。
    F3 改变中间张量的存储精度，数值会变——量级见 T4-A 的分辨率下限。
    """

    _cache_key = None
    _cache_val = None

    def __init__(self, lin, share_quant=True, fuse_bias=True):
        super().__init__()
        W = lin.weight.data.float()
        ws = (W.abs().max() / E4M3_MAX).clamp_min(1e-12)
        self.register_buffer("w_scale", ws.reshape(1))
        # _scaled_mm 要 B 列主序：W 是 [N,K] 行主序 -> W.t() 已是列主序，
        # **不能 .contiguous()**，那会变回行主序，cuBLASLt 直接拒
        self.register_buffer("w8", (W / ws).clamp(-E4M3_MAX, E4M3_MAX)
                             .to(torch.float8_e4m3fn).t())
        self.register_buffer("bias8", None if lin.bias is None
                             else lin.bias.data.to(torch.bfloat16))
        self.out_features = lin.out_features
        self.share_quant = share_quant
        self.fuse_bias = fuse_bias

    def _mm(self, x8, xs):
        if self.fuse_bias and self.bias8 is not None:
            return torch._scaled_mm(x8, self.w8, scale_a=xs.reshape(()),
                                    scale_b=self.w_scale.reshape(()),
                                    bias=self.bias8, out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
        out = torch._scaled_mm(x8, self.w8, scale_a=xs.reshape(()),
                               scale_b=self.w_scale.reshape(()),
                               out_dtype=torch.bfloat16, use_fast_accum=True)
        if self.bias8 is not None:
            out = out + self.bias8
        return out

    def forward(self, x):
        if isinstance(x, Fp8Carry):                       # F3
            x8, xs = x.fp8
            return self._mm(x8, xs).reshape(*x.shape[:-1], self.out_features)
        shp = x.shape
        # ⚠️ 缓存必须挂在**张量对象本身**上。初版用类级 `id(x)` 做 key，
        # 而 CPython 会复用已释放对象的 id —— ffn 的 [32760,8960] 量化结果
        # 被喂给了 q/k/v，直接 shape mismatch 崩掉。挂属性没有这个问题。
        cached = getattr(x, "_fp8c", None) if self.share_quant else None
        if cached is not None and cached[0].shape[-1] == shp[-1]:
            x8, xs = cached
        else:
            x8, xs = _quant_pertensor(x.reshape(-1, shp[-1]).contiguous())
            if self.share_quant:
                try:
                    x._fp8c = (x8, xs)
                except Exception:
                    pass
        return self._mm(x8, xs).reshape(*shp[:-1], self.out_features)


# ------------------------------- F3 -----------------------------------------

@torch.compile(dynamic=True)
def fused_norm_mod_quant(x, scale, shift, eps: float = 1e-6):
    """LayerNorm(无仿射) -> *(1+scale) + shift -> fp8。

    Wan 的 `WanLayerNorm` 默认 `elementwise_affine=False`
    （`model.py:309,320`，norm1/norm2 都是），所以这里不带仿射参数。
    """
    xf = x.float()
    mu = xf.mean(-1, keepdim=True)
    var = xf.var(-1, keepdim=True, unbiased=False)
    y = (xf - mu) * torch.rsqrt(var + eps) * (1 + scale.float()) + shift.float()
    y = y.reshape(-1, y.shape[-1])          # _scaled_mm 只吃 2D
    sc = (y.abs().amax() / E4M3_MAX).clamp_min(1e-6)
    return (y / sc).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), sc.float()


@torch.compile(dynamic=True)
def fused_gelu_quant(x):
    """GELU(tanh) -> fp8。ffn.0 的输出是 [32760, 8960]（bf16 587 MB），
    不融合的话这里要多走三遍。"""
    y = torch.nn.functional.gelu(x.float(), approximate="tanh")
    y = y.reshape(-1, y.shape[-1])          # _scaled_mm 只吃 2D
    sc = (y.abs().amax() / E4M3_MAX).clamp_min(1e-6)
    return (y / sc).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), sc.float()


def make_fused_block_forward(block):
    """替换 `WanAttentionBlock.forward`，在两个入口 + ffn 中间直接产出 fp8。

    cross-attn 的 norm3 保持原样：它带仿射，且 k/v 来自 text（M=512），
    占每步 FLOPs < 0.3%，融合收益可忽略。
    """
    dim = block.dim
    ffn_dim = block.ffn_dim

    def fwd(x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        e = (block.modulation + e).chunk(6, dim=1)
        B, L = x.shape[0], x.shape[1]

        # self-attn 入口：norm1 + modulate + quant，一次出 fp8，不落中间张量
        x8, xs = fused_norm_mod_quant(x, e[1], e[0], block.norm1.eps)
        y = block.self_attn(Fp8Carry((B, L, dim), x8, xs, x.dtype, x.device),
                            seq_lens, grid_sizes, freqs)
        x = x + y * e[2]

        # cross-attn：不融合
        x = x + block.cross_attn(block.norm3(x), context, context_lens)

        # ffn 入口：norm2 + modulate + quant
        f8, fs = fused_norm_mod_quant(x, e[4], e[3], block.norm2.eps)
        h = block.ffn[0](Fp8Carry((B, L, dim), f8, fs, x.dtype, x.device))
        # ffn 中间：gelu + quant（这里的张量最大）
        g8, gs = fused_gelu_quant(h)
        y = block.ffn[2](Fp8Carry((B, L, ffn_dim), g8, gs, x.dtype, x.device))
        x = x + y * e[5]
        return x
    return fwd


# ------------------------------ 装配 -----------------------------------------

def convert_fused(model, n_layers=30, share_quant=True, fuse_bias=True,
                  fuse_epilogue=True):
    """把 30 个 block 的 10 个 linear 换成 Fp8LinearFused，并按需替换 forward。"""
    from quant.fp8.wan_fp8 import target_names, set_submodule
    names = target_names(model, n_layers)
    for n in names:
        m = model.get_submodule(n)
        set_submodule(model, n, Fp8LinearFused(m, share_quant, fuse_bias)
                      .to(m.weight.device))
    if fuse_epilogue:
        for i in range(n_layers):
            blk = model.get_submodule(f"blocks.{i}")
            blk.forward = make_fused_block_forward(blk)
    torch.cuda.empty_cache()
    return names


# ===================== BF16 侧的同档编译（T5-A，P0） =========================
#
# ⚠️ **为什么必须有这一节**：T4 的四点折线里，FP8 侧的 F3 用了 `torch.compile`
# 折叠 norm/GELU 的 epilogue，而 **BF16 基线是纯 eager**。
# 那么 1.094× 里就混进了「compiled vs eager」，不全是「FP8 vs BF16」。
# 更糟的是 Amdahl 上界 1.105× 本身由 eager BF16 的算子占比推出 —— 基线一换，
# 上界也会动。
#
# 本节给 BF16 基线**同档**的编译：把 `norm+modulate` 与 `GELU` 各自折成一遍，
# 但不做任何量化。这样「编译」这个变量在两侧被控住，折线的差额才归 FP8。


@torch.compile(dynamic=True)
def fused_norm_mod_bf16(x, scale, shift, eps: float = 1e-6):
    """与 `fused_norm_mod_quant` 同档，只是输出 bf16 而不是 fp8。"""
    xf = x.float()
    mu = xf.mean(-1, keepdim=True)
    var = xf.var(-1, keepdim=True, unbiased=False)
    y = (xf - mu) * torch.rsqrt(var + eps) * (1 + scale.float()) + shift.float()
    return y.to(x.dtype)


@torch.compile(dynamic=True)
def fused_gelu_bf16(x):
    return torch.nn.functional.gelu(x.float(), approximate="tanh").to(x.dtype)


def make_bf16_fused_block_forward(block):
    """BF16 版的 block forward，融合点与 FP8 版**一一对应**。"""
    def fwd(x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        e = (block.modulation + e).chunk(6, dim=1)
        y = block.self_attn(fused_norm_mod_bf16(x, e[1], e[0], block.norm1.eps),
                            seq_lens, grid_sizes, freqs)
        x = x + y * e[2]
        x = x + block.cross_attn(block.norm3(x), context, context_lens)
        h = block.ffn[0](fused_norm_mod_bf16(x, e[4], e[3], block.norm2.eps))
        y = block.ffn[2](fused_gelu_bf16(h))
        x = x + y * e[5]
        return x
    return fwd


def convert_bf16_fused(model, n_layers=30):
    """只替换 forward，不换任何 linear —— 纯 BF16 + 同档编译。"""
    for i in range(n_layers):
        blk = model.get_submodule(f"blocks.{i}")
        blk.forward = make_bf16_fused_block_forward(blk)
    return n_layers
