#!/usr/bin/env python3
"""wan_fp8.py — Wan2.1 DiT 的 FP8 W8A8 接入层（本项目新写的部分）

`quant/gptq/` 里的 `formats.py` / `solvers.py` 是 vendored 的（见那两个文件的头注）。
**本文件是 Wan 特有的部分**：哪些 linear 要量化、怎么换、以及三档 kernel 形态。

三档 kernel 形态（对应 `SPEC.md` §4.2 的三数结构）：

  FakeQuantLinear   伪量化，bf16 GEMM。**只用于质量评测**，不报延迟。
  Fp8LinearNaive    真 `_scaled_mm`，quant 用朴素 eager 算子
                    （3 遍 activation：abs/amax → div → cast）
                    = 「朴素 W8A8」那一格，E3 要消除的就是这些多余往返
  Fp8LinearCompiled 真 `_scaled_mm`，quant 用 `torch.compile` 融成一遍
                    = 前人 `an earlier in-house implementation` 的形态
                    严格说这已经是「部分融合」，所以它是朴素与 E3 之间的中间档

E0 已经证明 epilogue 的写出宽度值 9–16%，所以三档之间的差额是可预期的、
且**都在 quant/dequant 的访存上，不在算力上**。
"""
from __future__ import annotations

import torch
import torch.nn as nn

E4M3_MAX = 448.0

# 一个 block 里可量化的 10 个 linear。patch_embedding 是 Conv3d、head 只有 N=64、
# time_embedding/text_embedding 每 prompt 或每步只跑一次 —— 都不在每步热路径上，
# 不量化（理由和数字见 research/gemm_shapes.json 的 per_step_calls 列）。
BLOCK_LINEARS = [
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
]


def target_names(model, n_layers=30):
    """要量化的 linear 的全限定名，顺序固定（决定 H 收集的分组顺序）。"""
    names = []
    for i in range(n_layers):
        for suf in BLOCK_LINEARS:
            n = f"blocks.{i}.{suf}"
            try:
                m = model.get_submodule(n)
            except AttributeError:
                continue
            if isinstance(m, nn.Linear):
                names.append(n)
    return names


def set_submodule(model, name, new):
    parts = name.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = new
    else:
        setattr(obj, last, new)


# ============================== 质量侧 ======================================

class FakeQuantLinear(nn.Module):
    """伪量化 linear：权重已被替换为反量化后的值，激活按 afmt 做动态伪量化。

    `collect` 是 Hessian 收集钩子：拿到的是**激活伪量化之后**的张量，
    也就是 GEMM 真正看到的东西 —— 与前人的 on-policy 协议一致
    （`an earlier in-house implementation`）。
    """

    def __init__(self, lin: nn.Linear, afmt=None):
        super().__init__()
        self.weight = nn.Parameter(lin.weight.data, requires_grad=False)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data,
                                                               requires_grad=False)
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.afmt = afmt
        self.collect = None

    def forward(self, x):
        xq = self.afmt(x) if self.afmt is not None else x
        if self.collect is not None:
            self.collect(xq)
        return torch.nn.functional.linear(xq, self.weight.to(xq.dtype),
                                          None if self.bias is None
                                          else self.bias.to(xq.dtype))


def convert_fakequant(model, afmt=None, n_layers=30):
    names = target_names(model, n_layers)
    for n in names:
        m = model.get_submodule(n)
        set_submodule(model, n, FakeQuantLinear(m, afmt).to(m.weight.device))
    return names


# ------------------------- parity 版伪量化（T4-A 产物） ---------------------

def fp8_quant_dequant_hw(x, scale):
    """用**硬件 cast** 做量化决策，而不是 bucketize。

    T4-A 实测：`Codebook.round`（bucketize 到 253 个电平）与
    `.to(torch.float8_e4m3fn)` 在**最多 4% 的元素上给出不同的电平**
    （见 `results/E4A/path_parity.json` 的 `quant_decision_parity`）。
    伪量化要当真 kernel 的代理，量化决策必须逐位一致 —— 用硬件自己的 cast。

    ⚠️ `.to(float8_e4m3fn)` **不饱和**：>448 的值变成 NaN（L0 实测）。
    所以先 clamp 到 ±448 再 cast，把「饱和」这个语义显式补上。
    """
    return (x / scale).clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float() * scale


class FakeQuantLinearParity(nn.Module):
    """与 `Fp8Linear(mode='pertensor')` **数值口径对齐**的伪量化 linear。

    与 `FakeQuantLinear` 的四处差别，每一处都是 T4-A 定位出来的误差源：

      1. 量化决策用硬件 cast，不用 bucketize      -> 消除电平不一致（最多 4% 元素）
      2. 反量化与 matmul 全程 **FP32**，不落 bf16 -> 消除激活侧 bf16 往返
         （T4-A 实测：bf16 往返贡献的误差是量化误差本身的 72–81%）
      3. **权重以 FP32 存**                        -> 消除权重侧 bf16 往返。
         `Fp8Linear` 存的是精确 FP8 + FP32 scale；若把反量化值写回 bf16 张量，
         就又引入一次舍入。代价是权重显存 2.64 -> 5.68 GiB
      4. matmul 显式关 TF32                        -> `_scaled_mm` 的累加是真 FP32，
         而 torch 的 fp32 matmul 默认走 TF32（10 位尾数），不关掉就不是同一个东西

    **它只用于质量评测和 E7 的 NVFP4 代理，不用于任何延迟数字。**
    """

    def __init__(self, lin: nn.Linear, act_fp8=True, w_fp8=True, hw_dtype_seq=False):
        super().__init__()
        W = lin.weight.data.float()
        if w_fp8:
            ws = (W.abs().amax() / E4M3_MAX).clamp_min(1e-30)
            W = fp8_quant_dequant_hw(W, ws)
        self.register_buffer("w32", W)
        self.register_buffer("b32", None if lin.bias is None
                             else lin.bias.data.float())
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.act_fp8 = act_fp8
        # hw_dtype_seq=True 逐位复刻 Fp8Linear 的 dtype 序列：
        #   amax 在 **bf16** 上算（输入张量就是 bf16），scale 也是 bf16，
        #   除法在 bf16 里做，然后才 cast 到 fp8。
        # T4-A 实测这一条是 fake/real 在端到端上系统性分叉的主因。
        self.hw_dtype_seq = hw_dtype_seq
        self.collect = None

    def forward(self, x):
        shp = x.shape
        if self.act_fp8 and self.hw_dtype_seq:
            xb = x.reshape(-1, shp[-1])
            xs = (xb.abs().amax() / E4M3_MAX).clamp_min(1e-6)
            x2 = (xb / xs).clamp_(-E4M3_MAX, E4M3_MAX).to(
                torch.float8_e4m3fn).float() * xs.float()
        else:
            x2 = x.reshape(-1, shp[-1]).float()
            if self.act_fp8:
                xs = (x2.abs().amax() / E4M3_MAX).clamp_min(1e-30)
                x2 = fp8_quant_dequant_hw(x2, xs)
        if self.collect is not None:
            self.collect(x2)
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            out = x2 @ self.w32.t()
            if self.b32 is not None:
                out = out + self.b32
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        return out.reshape(*shp[:-1], self.out_features).to(x.dtype)


def convert_fakequant_parity(model, n_layers=30, act_fp8=True, w_fp8=True,
                             hw_dtype_seq=False):
    names = target_names(model, n_layers)
    for n in names:
        m = model.get_submodule(n)
        set_submodule(model, n,
                      FakeQuantLinearParity(m, act_fp8, w_fp8,
                                            hw_dtype_seq).to(m.weight.device))
    torch.cuda.empty_cache()
    return names


def load_parity_weights(model, packed: dict):
    """把 solve 出来的反量化权重塞进 parity 模块的 fp32 buffer（不经 bf16）。"""
    n = 0
    for name, W in packed.items():
        m = model.get_submodule(name)
        m.w32.copy_(W.to(m.w32.device, torch.float32))
        n += 1
    return n


def load_dequant_weights(model, packed: dict):
    """把 solve 出来的反量化权重塞回去（key = target_names 里的名字）。"""
    n = 0
    for name, W in packed.items():
        m = model.get_submodule(name)
        m.weight.data.copy_(W.to(m.weight.device, m.weight.dtype))
        n += 1
    return n


# ============================== 延迟侧 ======================================

def _quant_pertensor_naive(x):
    """朴素 eager：三遍 activation。这正是 E3 要消掉的东西。"""
    amax = x.abs().amax()                      # pass 1：读一遍
    sc = (amax.float() / E4M3_MAX).clamp_min(1e-6)
    xf = x.float() / sc                        # pass 2：读一遍 + 写 fp32（4 字节！）
    return xf.to(torch.float8_e4m3fn), sc      # pass 3：读 fp32 + 写 fp8


@torch.compile(dynamic=True)
def _quant_pertensor_compiled(x):
    sc = (x.abs().amax() / E4M3_MAX).clamp_min(1e-6)
    return (x / sc).to(torch.float8_e4m3fn), sc.float()


@torch.compile(dynamic=True)
def _quant_pertok_compiled(x):
    sc = (x.abs().amax(-1, keepdim=True) / E4M3_MAX).clamp_min(1e-6)
    return (x / sc).to(torch.float8_e4m3fn), sc.float()


class Fp8Linear(nn.Module):
    """真 `torch._scaled_mm`。mode ∈ {pertensor_naive, pertensor, pertok}

    layout：`_scaled_mm` 要 A 行主序 [M,K]、B 列主序 [K,N]。
    W 是 [N,K] 行主序，所以 `W.t()` 已经是列主序 —— **不能 `.contiguous()`**，
    那会变回行主序，cuBLASLt 直接拒（`an earlier in-house implementation`）。

    sm_89 上 rowwise scaled_mm 会 abort，所以 per-token 只能走
    `out_dtype=fp32` 再乘 scale —— 多一遍 elementwise。
    这条正好呼应 E0 的副产物 2（FP32 输出比 BF16 输出慢 9–16%）。
    """

    def __init__(self, lin, mode="pertensor"):
        super().__init__()
        self.mode = mode
        W = lin.weight.data.float()
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data,
                                                               requires_grad=False)
        if mode.startswith("pertensor"):
            ws = (W.abs().max() / E4M3_MAX).clamp_min(1e-12)
            self.register_buffer("w_scale", ws.reshape(1))
        else:
            ws = (W.abs().amax(1, keepdim=True) / E4M3_MAX).clamp_min(1e-12)
            self.register_buffer("w_scale", ws.reshape(1, -1))
        self.register_buffer("w8", (W / ws).to(torch.float8_e4m3fn).t())
        self.out_features = lin.out_features

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).contiguous()
        if self.mode == "pertensor_naive":
            x8, xs = _quant_pertensor_naive(x2)
            out = torch._scaled_mm(x8, self.w8, scale_a=xs.reshape(()),
                                   scale_b=self.w_scale.reshape(()),
                                   out_dtype=torch.bfloat16, use_fast_accum=True)
        elif self.mode == "pertensor":
            x8, xs = _quant_pertensor_compiled(x2)
            out = torch._scaled_mm(x8, self.w8, scale_a=xs.reshape(()),
                                   scale_b=self.w_scale.reshape(()),
                                   out_dtype=torch.bfloat16, use_fast_accum=True)
        else:
            x8, xs = _quant_pertok_compiled(x2)
            one = torch.ones((), device=x.device)
            out = torch._scaled_mm(x8, self.w8, scale_a=one, scale_b=one,
                                   out_dtype=torch.float32, use_fast_accum=True)
            out = (out * xs * self.w_scale).to(torch.bfloat16)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shp[:-1], self.out_features)


def convert_fp8(model, mode="pertensor", n_layers=30):
    names = target_names(model, n_layers)
    for n in names:
        m = model.get_submodule(n)
        set_submodule(model, n, Fp8Linear(m, mode).to(m.weight.device))
    torch.cuda.empty_cache()
    return names


# ============================ H 收集的分组 ==================================

def plan_groups(model, names, budget_gb):
    """按 K² 装箱：每组的 H 总大小不超过 budget_gb，每组跑一次校准 pass。

    抄 `an earlier in-house implementation` 的做法。Wan 上的账：
      K=1536 的 linear -> H 9.4 MB；K=8960 的 ffn.2 -> H 321 MB
      300 个 linear 全放一起 = 11.3 GB，单卡放不下（还要留给权重和激活）
    """
    sizes = []
    for n in names:
        k = model.get_submodule(n).in_features
        sizes.append((n, k * k * 4 / 1e9))
    groups, cur, cur_sz = [], [], 0.0
    for n, sz in sorted(sizes, key=lambda t: -t[1]):
        if cur and cur_sz + sz > budget_gb:
            groups.append(cur)
            cur, cur_sz = [], 0.0
        cur.append(n)
        cur_sz += sz
    if cur:
        groups.append(cur)
    return groups
