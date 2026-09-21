# ---------------------------------------------------------------------------
# VENDORED — 逐字复制自 formats.py
#   源仓库：<HOME>/an earlier in-house library/formats.py
#   原始血统：<SF_ROOT>/scripts/nvfp4/{gptq_nvfp4.py,gptq_variants.py}
#             -> an earlier in-house library（格式解耦重写）-> 本文件
#   报告中标注：adapted from author's prior work
#
# **本文件不做任何修改**，Wan 特有的东西全部放在 quant/fp8/wan_fp8.py。
# 这样「复用了什么」与「本项目加了什么」在文件边界上就分清了。
# vendored at 2026-09-18 by T3
# ---------------------------------------------------------------------------

"""Pluggable quantization grids.

Everything downstream (RTN / GPTQ / clip / coordinate descent) only ever talks to
a `Codebook`, so switching between NVFP4, FP8-E4M3, INT8 and INTk is a one-line
change and the solver comparison stays apples-to-apples across formats.

A codebook is a sorted 1-D table of levels for the *normalized* operand
(i.e. w / scale).  Rounding is `bucketize` against the midpoints, which is exact
round-to-nearest with ties-to-even handled explicitly where the hardware does so.
Coordinate descent needs to enumerate candidates, and doing that over a full
253-entry FP8 table is wasteful, so candidates are taken as a +-radius window
around the current level index; for 4-bit tables the window covers everything.
"""
from __future__ import annotations

import torch

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _e4m3_levels():
    b = torch.arange(256, dtype=torch.uint8)
    v = b.view(torch.float8_e4m3fn).float()
    v = v[torch.isfinite(v)]
    return torch.unique(v)


class Codebook:
    """A symmetric, sorted level table plus the machinery solvers need."""

    def __init__(self, levels, name, cd_radius=None):
        lv = torch.as_tensor(sorted(set(float(x) for x in levels)), dtype=torch.float32)
        self.levels = lv
        self.name = name
        self.n = lv.numel()
        self.absmax = float(lv.abs().max())
        self.mids = (lv[1:] + lv[:-1]) / 2
        # full enumeration is cheap below ~24 levels; otherwise use a local window
        self.cd_radius = cd_radius if cd_radius is not None else (
            self.n if self.n <= 24 else 3)
        self._cache = {}

    def _dev(self, device):
        c = self._cache.get(device)
        if c is None:
            c = (self.levels.to(device), self.mids.to(device))
            self._cache[device] = c
        return c

    def index(self, x):
        """Nearest level index for normalized values x."""
        lv, mids = self._dev(x.device)
        return torch.bucketize(x.contiguous(), mids)

    def value(self, idx):
        lv, _ = self._dev(idx.device)
        return lv[idx]

    def round(self, x):
        return self.value(self.index(x))

    def candidates(self, idx):
        """[..., C] candidate level values around each current index."""
        lv, _ = self._dev(idx.device)
        if self.cd_radius >= self.n:
            return lv.view(*([1] * idx.dim()), self.n).expand(*idx.shape, self.n)
        r = self.cd_radius
        off = torch.arange(-r, r + 1, device=idx.device)
        cand_idx = (idx.unsqueeze(-1) + off).clamp_(0, self.n - 1)
        return lv[cand_idx]


def _int_levels(bits, symmetric=True):
    m = 2 ** (bits - 1)
    lo = -(m - 1) if symmetric else -m
    return list(range(lo, m))


CODEBOOKS = {
    "nvfp4": Codebook([s * v for v in _E2M1 for s in (1, -1)], "nvfp4"),
    "e2m1": Codebook([s * v for v in _E2M1 for s in (1, -1)], "e2m1"),
    "int4": Codebook(_int_levels(4), "int4"),
    "int8": Codebook(_int_levels(8), "int8"),
    "fp8_e4m3": Codebook(_e4m3_levels().tolist(), "fp8_e4m3"),
}


class WeightFormat:
    """Codebook + scale granularity + optional scale quantization.

    group: 0 -> one scale per output channel (row); k>0 -> one scale per
           (row, k contiguous input channels).
    scale_fmt: 'fp32' or 'e4m3' (NVFP4's two-level scheme: block E4M3 scale
           normalized by an FP32 per-tensor global encode factor).
    """

    def __init__(self, codebook, group=0, scale_fmt="fp32", name=None):
        self.cb = CODEBOOKS[codebook] if isinstance(codebook, str) else codebook
        self.group = group
        self.scale_fmt = scale_fmt
        self.name = name or f"{self.cb.name}_g{group}_{scale_fmt}"

    def group_scales(self, W, gamma=1.0):
        """[N, G] dequant scales for weight [N, K].

        group = -1 -> a single per-tensor scale (what torch._scaled_mm can take
        natively on sm_89; anything finer needs an extra epilogue pass).
        """
        N, K = W.shape
        Wf = W.float()
        if self.group and self.group > 0:
            assert K % self.group == 0, (K, self.group)
            amax = Wf.reshape(N, K // self.group, self.group).abs().amax(-1)
        elif self.group == -1:
            amax = Wf.abs().amax().reshape(1, 1).expand(N, 1)
        else:
            amax = Wf.abs().amax(-1, keepdim=True)
        s = amax * (gamma / self.cb.absmax)
        if self.scale_fmt == "e4m3":
            enc = 448.0 * self.cb.absmax / Wf.abs().max().clamp_min(1e-30)
            s = (s.clamp_min(1e-30) * enc).to(torch.float8_e4m3fn).float() / enc
        return s.clamp_min(1e-30)

    def col_scales(self, W, gamma=1.0):
        """[N, K] per-column broadcast of the group scales (what GPTQ needs)."""
        N, K = W.shape
        s = self.group_scales(W, gamma)
        if s.shape[1] == 1:
            return s.expand(N, K)
        g = self.group
        return s.repeat_interleave(g, dim=1)

    def rtn(self, W, gamma=1.0, s_col=None):
        s = self.col_scales(W, gamma) if s_col is None else s_col
        return self.cb.round(W.float() / s) * s


class ActFormat:
    """Dynamic activation fake-quant (per-token / per-tensor / per-group).

    `axis="group:g"` gives one scale per (row, contiguous block of g input
    channels) -- the granularity the upstream fp8-emitting operator uses.  It
    sits between per-tensor and per-token in cost and below both in error.  All
    three of the DiT's K values (3072 / 12288 / 15360) divide by 64 and 128, so
    no padding is needed.

    Memory note: at the 4k tier one activation is [38k, 15360] = 1.2 GB in bf16.
    A naive `x.float()` plus a bucketize against the midpoints allocates several
    fp32 temporaries of that size and OOMs a 24 GB card, so BOTH the reduction
    and the apply are chunked over rows, and no full-size fp32 copy is ever made.
    """

    def __init__(self, codebook="fp8_e4m3", axis="token", chunk_rows=4096,
                 mul=1.0):
        self.cb = CODEBOOKS[codebook] if isinstance(codebook, str) else codebook
        self.axis = axis
        self.chunk_rows = chunk_rows
        # `mul` scales the dynamic range the codebook is asked to cover:
        # mul < 1 clips the outliers and spends the mantissa on the bulk,
        # mul > 1 leaves headroom.  It is the knob PACT/LSQ learn, exposed here
        # so it can be searched per step (see qat/lsq_scan.py).
        self.mul = float(mul)

    def __call__(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        n, cr = x2.shape[0], self.chunk_rows

        if isinstance(self.axis, str) and self.axis.startswith("group"):
            g = int(self.axis.split(":")[1]) if ":" in self.axis else 128
            k = x2.shape[-1]
            assert k % g == 0, (k, g)
            out = torch.empty_like(x2)
            for i in range(0, n, cr):
                j = min(i + cr, n)
                c = x2[i:j].float().reshape(-1, k // g, g)
                s = (c.abs().amax(-1, keepdim=True) * self.mul
                     / self.cb.absmax).clamp_min(1e-30)
                out[i:j] = (self.cb.round(c / s) * s).reshape(j - i, k).to(x.dtype)
                del c
            return out.reshape(shp)

        gs = None
        if self.axis == "tensor":
            amax = x2.new_zeros((), dtype=torch.float32)
            for i in range(0, n, cr):
                amax = torch.maximum(amax, x2[i:i + cr].abs().amax().float())
            gs = (amax * self.mul / self.cb.absmax).clamp_min(1e-30)

        out = torch.empty_like(x2)
        for i in range(0, n, cr):
            j = min(i + cr, n)
            c = x2[i:j].float()
            if gs is None:
                s = (c.abs().amax(-1, keepdim=True) * self.mul
                     / self.cb.absmax).clamp_min(1e-30)
            else:
                s = gs
            out[i:j] = (self.cb.round(c / s) * s).to(x.dtype)
            del c
        return out.reshape(shp)


def parse_wfmt(spec):
    """'nvfp4:16:e4m3' | 'fp8_e4m3:0:fp32' | 'int4:128:fp32'"""
    parts = spec.split(":")
    cb = parts[0]
    group = int(parts[1]) if len(parts) > 1 else 0
    sf = parts[2] if len(parts) > 2 else "fp32"
    return WeightFormat(cb, group, sf, name=spec)


def parse_afmt(spec):
    """'none' | 'fp8_e4m3:token' | 'fp8_e4m3:tensor' | 'fp8_e4m3:group:128'"""
    if not spec or spec == "none":
        return None
    parts = spec.split(":")
    axis = ":".join(parts[1:]) if len(parts) > 1 else "token"
    return ActFormat(parts[0], axis)
