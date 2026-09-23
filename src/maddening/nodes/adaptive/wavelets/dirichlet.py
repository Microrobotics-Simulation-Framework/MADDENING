"""Boundary-adapted Deslauriers-Dubuc basis for homogeneous Dirichlet conditions.

A wall breaks the translation symmetry the periodic lifting scheme in
:mod:`.transform` relies on.  This module builds the interpolating basis
on the open interval with the boundary values pinned to zero: the
coarse samples are interior nodal values, the data just outside the
domain is implied zero, and each refinement predicts the new midpoints
-- including the two next to the walls -- with the widest centred
Deslauriers-Dubuc stencil that fits, shrinking to linear interpolation
at the ends rather than inventing data past the boundary.

Unlike the periodic transform this basis is built as a **dense synthesis
matrix at construction time**, in NumPy, and multi-dimensional bases are
the separable tensor (Kronecker) product of the 1-D one.  That is
correct and cheap at the sizes the node validates
(``side <= 383`` in 1-D, ``47**2`` in 2-D); a matrix-free separable
transform is the obvious optimisation if larger Dirichlet problems are
ever needed.

Grid convention: ``side`` **interior** nodes at ``x_i = i / (side + 1)``,
``i = 1 .. side``; the boundary nodes are not degrees of freedom.  One
refinement takes ``cur -> 2 cur + 1`` interior nodes, so
``side = (n_coarse + 1) * 2**n_levels - 1``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.nodes.adaptive.wavelets.transform import _DD_FILTERS, _check_dim, _check_order

__all__ = [
    "dirichlet_side",
    "synthesis_matrix_dirichlet",
]


@stability(StabilityLevel.EXPERIMENTAL)
def dirichlet_side(n_levels: int, n_coarse: int) -> int:
    """Interior-node count per axis after ``n_levels`` Dirichlet refinements.

    ``(n_coarse + 1) * 2**n_levels - 1``.

    >>> dirichlet_side(n_levels=2, n_coarse=2)
    11
    """
    cur = int(n_coarse)
    for _ in range(int(n_levels)):
        cur = 2 * cur + 1
    return cur


def _refine_dirichlet(coarse: np.ndarray, order: int) -> np.ndarray:
    """One refinement ``cur -> 2 cur + 1`` (NumPy, eager).

    Midpoints (even output indices, including the two next to the walls)
    are predicted from the zero-padded coarse samples with the widest
    centred stencil that fits.
    """
    nc = coarse.shape[0]
    fine = np.zeros(2 * nc + 1, dtype=coarse.dtype)
    fine[1::2] = coarse                                  # coarse -> odd nodes
    padded = np.concatenate(([0.0], coarse, [0.0]))      # zero data at the walls
    for j in range(nc + 1):                              # midpoint j between padded[j], padded[j+1]
        o = order
        while o > 2:                                     # shrink until the stencil fits
            half = o // 2
            if j - (half - 1) >= 0 and j + half + 1 <= len(padded):
                break
            o -= 2
        offsets, weights = _DD_FILTERS[o]
        val = 0.0
        for off, w in zip(offsets, weights):
            k = j + off
            if 0 <= k < len(padded):
                val += w * padded[k]
        fine[2 * j] = val
    return fine


def _synthesis_1d(coeffs: np.ndarray, n_levels: int, n_coarse: int, order: int) -> np.ndarray:
    idx = 0
    vals = np.array(coeffs[idx:idx + n_coarse], dtype=np.float64)
    idx += n_coarse
    cur = n_coarse
    for _ in range(n_levels):
        refined = _refine_dirichlet(vals, order)          # 2 cur + 1
        n_detail = cur + 1
        refined[0::2] += coeffs[idx:idx + n_detail]
        idx += n_detail
        vals = refined
        cur = 2 * cur + 1
    return vals


def _level_labels_1d(n_levels: int, n_coarse: int) -> np.ndarray:
    labs = [0] * n_coarse
    cur = n_coarse
    for lvl in range(n_levels):
        labs += [lvl] * (cur + 1)
        cur = 2 * cur + 1
    return np.asarray(labs, dtype=np.int32)


def _level_labels_nd(n_levels: int, n_coarse: int, dim: int) -> np.ndarray:
    """Level label per function of the ``dim``-D tensor-product basis.

    The maximum over axes of the 1-D labels, in the Kronecker (row-major)
    ordering of :func:`synthesis_matrix_dirichlet`, so the coarse block is
    exactly level 0 and level 0 as a whole -- the coarse block plus the
    first detail band per axis -- has ``(2 n_coarse + 1) ** dim`` entries.
    Host ``int32`` NumPy: what the assembly and the node's seed-size
    validation read, concrete inside a trace.
    """
    lev1 = _level_labels_1d(int(n_levels), int(n_coarse))
    lev = lev1
    for _ in range(int(dim) - 1):
        lev = np.maximum.outer(lev, lev1).reshape(-1)
    return np.asarray(lev, dtype=np.int32)


def _synthesis_matrix_1d(n_levels: int, n_coarse: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    n = dirichlet_side(n_levels, n_coarse)
    W = np.zeros((n, n))
    e = np.zeros(n)
    for j in range(n):
        e[:] = 0.0
        e[j] = 1.0
        W[:, j] = _synthesis_1d(e, n_levels, n_coarse, order)
    return W, _level_labels_1d(n_levels, n_coarse)


@stability(StabilityLevel.EXPERIMENTAL)
def synthesis_matrix_dirichlet(n_levels: int, n_coarse: int, *, order: int = 4,
                               dim: int = 1, dtype: Any = None):
    """Dense Dirichlet synthesis matrix ``W`` with per-function level labels.

    Parameters
    ----------
    n_levels, n_coarse : int
        Refinements and coarse interior points per axis.
    order : {2, 4, 6}
        Interpolating order away from the walls.
    dim : {1, 2, 3}
        Spatial dimension; the multi-D basis is the Kronecker product of
        the 1-D one and a function's level is the maximum over axes, so
        the coarse block is exactly level ``0``.
    dtype : optional
        Floating dtype of ``W``; JAX's canonical float when omitted.

    Returns
    -------
    (W, levels, side)
        ``W`` of shape ``(side**dim, side**dim)``, ``levels`` an ``int32``
        vector, and the interior-node count ``side`` per axis.
    """
    _check_order(order)
    _check_dim(dim)
    W1, _ = _synthesis_matrix_1d(int(n_levels), int(n_coarse), int(order))
    side = W1.shape[0]
    W = W1
    for _ in range(dim - 1):
        W = np.kron(W, W1)
    lev = _level_labels_nd(n_levels, n_coarse, dim)
    return jnp.asarray(W, dtype=dtype), jnp.asarray(lev, dtype=jnp.int32), side
