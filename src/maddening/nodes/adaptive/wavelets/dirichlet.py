"""Boundary-adapted Deslauriers-Dubuc wavelets for homogeneous Dirichlet BCs.

Production geometry (MIME vessel walls) is no-slip Dirichlet, which breaks the
translation symmetry the periodic transform relies on.  This module builds a
boundary-adapted DD basis on the interval (0,1) with implied zero data outside,
ported from the spike's validated construction
(``dd_wavelets.synthesis_matrix(boundary="dirichlet")``, FINDINGS Inv 2A:
1D κ≈3.8, better than periodic; wrong-sign safe).

Unlike the periodic transform (matrix-free, JIT hot path), the Dirichlet basis
is built as a **dense synthesis matrix at construction time** (eager), exactly
as the spike validated and as the ``hierarchical_hat`` toy does for its local
basis.  Multi-dimensional Dirichlet uses the **separable tensor product** of the
1D Dirichlet basis (FINDINGS Inv 2A used this; it needs Jacobi rather than DK
scaling per Correction C1 -- and Jacobi is the production default).  A
matrix-free separable Dirichlet transform is a later optimisation (tracked with
the gather-to-K perf work); the dense path is correct and sufficient at the
validated cavity sizes.

The grid convention: ``side`` interior nodes (boundary nodes are not DOFs);
``side = (n_coarse + 1) * 2**n_levels - 1`` for the dyadic refinement
``cur -> 2*cur + 1``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from maddening.nodes.adaptive.wavelets.transform import _DD_FILTERS

__all__ = [
    "dirichlet_side",
    "synthesis_matrix_dirichlet",
    "levels_dirichlet",
]


def dirichlet_side(n_levels: int, n_coarse: int) -> int:
    """Interior-node count after ``n_levels`` Dirichlet refinements."""
    cur = n_coarse
    for _ in range(n_levels):
        cur = 2 * cur + 1
    return cur


def _refine_dirichlet(coarse: np.ndarray, order: int) -> np.ndarray:
    """One Dirichlet refinement step ``cur -> 2*cur + 1`` (numpy, eager).

    Coarse samples are interior nodal values with implied zero data just outside
    the domain.  Midpoints (including the two ends, between the boundary zero and
    the first/last coarse sample) are predicted with the widest centred DD
    stencil that fits, shrinking to linear near the ends -- the spike's
    reduced-order boundary treatment that avoids fabricating data past the
    boundary.
    """
    nc = coarse.shape[0]
    fine = np.zeros(2 * nc + 1, dtype=float)
    fine[1::2] = coarse                      # coarse -> odd interior nodes
    padded = np.concatenate(([0.0], coarse, [0.0]))  # zeros at the boundary
    for j in range(nc + 1):                  # midpoint between padded[j], padded[j+1]
        o = order
        while o > 2:                         # shrink until the stencil fits
            half = o // 2
            if j - (half - 1) >= 0 and j + half + 1 <= len(padded):
                break
            o -= 2
        offsets, weights = _DD_FILTERS[o]
        val = 0.0
        for off, w in zip(offsets, weights):
            idx = j + off
            if 0 <= idx < len(padded):
                val += w * padded[idx]
        fine[2 * j] = val                    # midpoints -> even nodes
    return fine


def _synth_dirichlet_1d(coeffs: np.ndarray, n_levels: int, n_coarse: int,
                        order: int) -> np.ndarray:
    """Inverse 1D Dirichlet transform (numpy, eager) for a single coeff vector."""
    idx = 0
    vals = coeffs[idx:idx + n_coarse].astype(float).copy()
    idx += n_coarse
    cur = n_coarse
    for _ in range(n_levels):
        refined = _refine_dirichlet(vals, order)     # size 2*cur+1
        n_detail = cur + 1
        detail = coeffs[idx:idx + n_detail]
        idx += n_detail
        refined[0::2] += detail                       # details at midpoints
        vals = refined
        cur = 2 * cur + 1
    return vals


def levels_dirichlet_1d(n_levels: int, n_coarse: int) -> np.ndarray:
    labs = [0] * n_coarse
    cur = n_coarse
    for lvl in range(n_levels):
        labs += [lvl] * (cur + 1)
        cur = 2 * cur + 1
    return np.asarray(labs, dtype=int)


def _synthesis_matrix_dirichlet_1d(n_levels: int, n_coarse: int, order: int):
    N = dirichlet_side(n_levels, n_coarse)
    W = np.zeros((N, N))
    e = np.zeros(N)
    for j in range(N):
        e[:] = 0.0
        e[j] = 1.0
        W[:, j] = _synth_dirichlet_1d(e, n_levels, n_coarse, order)
    x = (np.arange(1, N + 1)) / (N + 1)
    return W, levels_dirichlet_1d(n_levels, n_coarse), x


def synthesis_matrix_dirichlet(n_levels: int, n_coarse: int, order: int = 4,
                               dim: int = 1):
    """Dense Dirichlet synthesis matrix ``W`` and per-DOF level labels.

    1D: boundary-adapted DD basis.  Multi-D: separable tensor product (kron) of
    the 1D Dirichlet basis; the per-DOF level is the max over axes (so the
    coarse block is level 0 for CDD's coarse-seeding).  Returns ``(W, levels,
    side)`` as ``jnp`` arrays / int.
    """
    W1, lev1, _ = _synthesis_matrix_dirichlet_1d(n_levels, n_coarse, order)
    side = W1.shape[0]
    if dim == 1:
        return jnp.asarray(W1), jnp.asarray(lev1), side
    if dim == 2:
        W = np.kron(W1, W1)
        L = np.maximum.outer(lev1, lev1).reshape(-1)
        return jnp.asarray(W), jnp.asarray(L), side
    if dim == 3:
        W = np.kron(np.kron(W1, W1), W1)
        lv = lev1
        L = (np.maximum.outer(np.maximum.outer(lv, lv).reshape(-1),
                              lv).reshape(-1))
        return jnp.asarray(W), jnp.asarray(L), side
    raise ValueError(f"dim must be 1, 2, or 3; got {dim}")


def levels_dirichlet(n_levels: int, n_coarse: int, dim: int = 1) -> jnp.ndarray:
    _, L, _ = synthesis_matrix_dirichlet(n_levels, n_coarse, order=4, dim=dim)
    return L
