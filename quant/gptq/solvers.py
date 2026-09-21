# ---------------------------------------------------------------------------
# VENDORED — 逐字复制自 solvers.py
#   源仓库：<HOME>/an earlier in-house library/solvers.py
#   原始血统：<SF_ROOT>/scripts/nvfp4/{gptq_nvfp4.py,gptq_variants.py}
#             -> an earlier in-house library（格式解耦重写）-> 本文件
#   报告中标注：adapted from author's prior work
#
# **本文件不做任何修改**，Wan 特有的东西全部放在 quant/fp8/wan_fp8.py。
# 这样「复用了什么」与「本项目加了什么」在文件边界上就分清了。
# vendored at 2026-09-18 by T3
# ---------------------------------------------------------------------------

"""Training-free weight-quantization solvers, ported from the Wan2.1 NVFP4 work
(<SF_ROOT>: scripts/nvfp4/gptq_nvfp4.py + gptq_variants.py) and made
format-agnostic.

Ladder of local optimality on the same objective  min_Q (W-Q) H (W-Q)^T :

    rtn      no coupling; optimal only if H is diagonal
    gptq     sequential greedy over columns with error feedback; never revisits
    +clip    per-layer H-aware scale clip gamma (trade 1 saturated element for a
             finer bulk grid); only pays off *together with* error feedback
    +cd      exact per-element coordinate descent -> 1-opt local minimum,
             monotone from the GPTQ solution

Per-layer fallback to RTN whenever the solved proxy loss is worse than RTN's.
"""
from __future__ import annotations

import torch


def _proxy(D, H):
    return torch.einsum("nk,kl,nl->", D, H, D).item()


def prepare_hessian(H, percdamp=0.01):
    H = H.clone()
    dead = torch.diag(H) <= 0
    H[dead, dead] = 1.0
    damp = percdamp * torch.diag(H).mean()
    H.diagonal().add_(damp)
    return H, dead


def _chol_inv_upper(H, percdamp, dead_fix=True, max_tries=6):
    """Cholesky of H^-1 with damping escalation.

    Real DiT Hessians are occasionally rank-deficient (single-block proj_out sees
    a GELU output whose negative tail is nearly constant, and padded text tokens
    are exactly constant), so a fixed 1% damp is not always enough.  Escalate
    x10 until the factorization succeeds and report the damp actually used.
    """
    damp = percdamp
    for _ in range(max_tries):
        Hd = H.clone()
        Hd.diagonal().add_(damp * torch.diag(H).mean().clamp_min(1e-12))
        try:
            Hc = torch.linalg.cholesky(Hd)
            Hinv = torch.cholesky_inverse(Hc)
            return torch.linalg.cholesky(Hinv, upper=True), Hd, damp
        except Exception:
            damp *= 10.0
    raise RuntimeError("cholesky failed even at damp=%g" % damp)


def gptq_solve(W0, H0, s_col, cb, blocksize=128, percdamp=0.01, zero_dead=False):
    """GPTQ with explicit per-column scales. Returns (Q, H_damped, damp_used).

    ⚠️ `zero_dead` (the textbook `W[:, dead] = 0`) defaults to FALSE here, which
    differs from every GPTQ reference implementation.  Reason, measured on the
    omni-v2 DiT (kolors_dit/dead_channels.py):

      * 13 of 512 layers have channels whose second moment is exactly 0 -- all of
        them `transformer_blocks.*.ff_context.net.2`, the text-stream FFN
        down-projection, whose input is a GELU output with dead units.
      * The scorer, `prepare_hessian`, sets those diagonal entries to 1.0 -- on
        the affected layers that is ~36x the MEDIAN diagonal.  So zeroing the
        column and then scoring it at weight 1.0 charges an enormous fake price
        for a channel the objective does not actually constrain.
      * Consequence: the solved proxy loss came out worse than RTN and all 12
        layers fell back.  Keeping the weights turns 0.00% recovery into
        +40% .. +84%, and is bit-identical on every layer with no dead channels
        (verified on 2 such layers: 99.70% and 99.98% both ways).

    A channel with zero second moment is simply unconstrained; leaving its
    weight alone (RTN handles it) is right, throwing it away is not.
    """
    W = W0.float().clone()
    N, K = W.shape
    H = H0.clone()
    dead = torch.diag(H) <= 0
    H[dead, dead] = 1.0
    if zero_dead:
        W[:, dead] = 0
    Hinv, Hd, damp = _chol_inv_upper(H, percdamp)
    Q = torch.zeros_like(W)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = i1 + j
            w = W1[:, j]
            d = Hinv1[j, j]
            sc = s_col[:, col]
            q = cb.round(w / sc) * sc
            Q[:, col] = q
            err = (w - q) / d
            W1[:, j + 1:] -= err.unsqueeze(1) * Hinv1[j, j + 1:].unsqueeze(0)
            Err1[:, j] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q, Hd, damp


def cd_polish(W0, Q0, Hd, s_col, cb, passes=4, tol=-1e-9, block=128):
    """Exact per-element coordinate descent (Gauss-Seidel over columns).

    Fixing every other element, the loss is a 1-D quadratic in the chosen
    element, so the best level among the candidates is closed-form; only
    strictly-improving moves are accepted => monotone, cannot diverge.

    Blocking: G = D @ Hd is the gradient we need at column j.  Updating all K
    columns of G after every single element move costs O(N*K) per column and
    dominates.  Instead we keep only G[:, block] exact via small rank-1 updates
    inside the block and flush the accumulated change to the rest of G with one
    matmul per block.  Columns outside the block are not read until their own
    block starts, so this is *bit-for-bit the same sequential algorithm*, just
    ~K/block times less memory traffic.
    """
    W = W0.float()
    N, K = W.shape
    Q = Q0.clone()
    D = W - Q
    G = D @ Hd
    hdiag = torch.diag(Hd)
    rows = torch.arange(N, device=W.device)
    hist = []
    for _ in range(passes):
        changed = 0
        for i1 in range(0, K, block):
            i2 = min(i1 + block, K)
            Gb = G[:, i1:i2].clone()
            Hbb = Hd[i1:i2, i1:i2]
            dD = torch.zeros(N, i2 - i1, device=W.device, dtype=W.dtype)
            for j in range(i2 - i1):
                col = i1 + j
                sj = s_col[:, col]
                idx = cb.index(Q[:, col] / sj)
                cand = cb.candidates(idx) * sj.unsqueeze(-1)       # [N, C]
                dcur = D[:, col].unsqueeze(-1)
                delta = (W[:, col].unsqueeze(-1) - cand) - dcur
                dl = 2 * delta * Gb[:, j].unsqueeze(-1) + delta * delta * hdiag[col]
                b = dl.argmin(1)
                bd = torch.where(dl[rows, b] < tol, delta[rows, b],
                                 torch.zeros_like(delta[rows, 0]))
                nz = bd != 0
                if nz.any():
                    changed += int(nz.sum())
                    D[:, col] += bd
                    Q[:, col] = W[:, col] - D[:, col]
                    dD[:, j] += bd
                    if j + 1 < i2 - i1:
                        Gb[:, j + 1:] += bd.unsqueeze(-1) * Hbb[j, j + 1:].unsqueeze(0)
            if dD.abs().sum() > 0:
                G += dD @ Hd[i1:i2, :]
        hist.append((changed, _proxy(D, Hd)))
        if changed == 0:
            break
    return Q, hist


def solve_layer(W0, H, wfmt, method="gptq_clip_cd", gammas=(1.0, 0.97, 0.94, 0.90),
                passes=4, percdamp=0.01, blocksize=128, clip_select="gptq",
                verbose=False):
    """Returns (Q, info). W0 [N,K] float32 cuda, H [K,K] float32 cuda.

    clip_select='gptq' picks the clip gamma on the GPTQ-only proxy loss and then
    runs CD once on the winner (4x cheaper than polishing every gamma).  This is
    consistent with the source recipe, which already selects gamma by the H proxy
    -- and with its finding that a *denser* proxy-selected gamma grid makes
    end-to-end worse, i.e. the proxy is only trusted as a coarse screen here.
    clip_select='full' polishes every gamma and picks on the final loss.
    """
    cb = wfmt.cb
    W0 = W0.float()
    s1 = wfmt.col_scales(W0, 1.0)
    rtn = cb.round(W0 / s1) * s1
    Hd0, _ = prepare_hessian(H, percdamp)
    e_rtn = _proxy(W0 - rtn, Hd0)

    if method == "rtn":
        return rtn, {"recovery": 0.0, "gamma": 1.0, "e_rtn": e_rtn, "e": e_rtn,
                     "fallback": False}

    use_clip = "clip" in method
    use_cd = "cd" in method and passes > 0
    gl = list(gammas) if use_clip else [1.0]

    damp_used = percdamp
    if use_clip and use_cd and clip_select == "gptq":
        best_g, best_e_pre = 1.0, float("inf")
        for g in gl:
            s_col = wfmt.col_scales(W0, g)
            Q, Hd, dmp = gptq_solve(W0, H, s_col, cb, blocksize=blocksize,
                                    percdamp=percdamp)
            damp_used = max(damp_used, dmp)
            e = _proxy(W0 - Q, Hd0)
            if e < best_e_pre:
                best_g, best_e_pre = g, e
        gl = [best_g]

    best_Q, best_e, best_g = rtn, e_rtn, 1.0
    for g in gl:
        s_col = wfmt.col_scales(W0, g)
        Q, Hd, dmp = gptq_solve(W0, H, s_col, cb, blocksize=blocksize,
                                percdamp=percdamp)
        damp_used = max(damp_used, dmp)
        if use_cd:
            Q, _ = cd_polish(W0, Q, Hd, s_col, cb, passes=passes)
        e = _proxy(W0 - Q, Hd0)
        if e < best_e:
            best_Q, best_e, best_g = Q, e, g

    fallback = best_e >= e_rtn
    if fallback:
        best_Q, best_e, best_g = rtn, e_rtn, 1.0
    info = {"recovery": 1.0 - best_e / max(e_rtn, 1e-30), "gamma": best_g,
            "e_rtn": e_rtn, "e": best_e, "fallback": bool(fallback),
            "damp": damp_used}
    if verbose:
        print("   ", info, flush=True)
    return best_Q, info
