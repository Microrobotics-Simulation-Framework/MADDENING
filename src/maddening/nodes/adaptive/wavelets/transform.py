"""Deslauriers-Dubuc interpolating wavelet transforms (matrix-free, JAX).

Production reimplementation of the spike's wavelet construction
(``spikes/wavelet_derisking/dd_wavelets.py`` and ``dd_jax_poc.py``).  The
spike materialised the synthesis matrix ``W`` column-by-column in numpy; this
module is matrix-free and JAX-native: the forward (analysis) and inverse
(synthesis) transforms are implemented as lifting steps using ``jnp.roll``
prediction, with all shapes static so the per-level Python loop unrolls cleanly
under ``jax.jit``.

The transforms are interpolating (Deslauriers-Dubuc) of order ``2``, ``4``, or
``6``.  ``order=4`` (DD-4) is the production default: approximation order 4 and
4 vanishing moments, validated by the derisking spike as well-conditioned under
Dahmen-Kunoth / hybrid-Jacobi scaling.

Three spatial dimensionalities are supported, all on the **isotropic Mallat**
multiresolution (the basis the spike's Correction C1 established as the correct
one for an isotropic PDE operator):

* 1D: coefficient layout ``[coarse(n_coarse), detail_0(n_coarse),
  detail_1(2 n_coarse), ...]``.
* 2D: three detail subbands per level (LH, HL, HH), all at one resolution level.
* 3D: seven detail subbands per level (all L/H parities except LLL).

Only periodic boundaries are implemented here; the boundary-adapted Dirichlet
basis is a later milestone.

A dense ``synthesis_matrix`` materialiser is provided (via ``jax.vmap`` over the
identity) for operator assembly and for tests that compare against a dense
reference -- it is not used on the matrix-free hot path.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "DD_ORDERS",
    "synthesis_1d",
    "analysis_1d",
    "synthesis_2d",
    "analysis_2d",
    "synthesis_3d",
    "analysis_3d",
    "synthesis_transpose",
    "structural_blocks",
    "synthesis_matrix",
    "levels_1d",
    "levels_2d",
    "levels_3d",
    "n_dofs",
]

# DD interpolating-subdivision prediction filters.  A midpoint between coarse
# samples ``0`` and ``1`` is predicted from the ``2N`` nearest coarse samples;
# offsets are relative to the left neighbour.  Symmetric, sum to 1.
_DD_FILTERS: dict[int, Tuple[Tuple[int, ...], Tuple[float, ...]]] = {
    2: ((0, 1), (0.5, 0.5)),
    4: ((-1, 0, 1, 2), (-1.0 / 16, 9.0 / 16, 9.0 / 16, -1.0 / 16)),
    6: ((-2, -1, 0, 1, 2, 3),
        (3.0 / 256, -25.0 / 256, 150.0 / 256, 150.0 / 256, -25.0 / 256, 3.0 / 256)),
}

DD_ORDERS: tuple[int, ...] = tuple(sorted(_DD_FILTERS))


def _check_order(order: int) -> None:
    if order not in _DD_FILTERS:
        raise ValueError(f"unsupported DD order {order!r}; supported: {DD_ORDERS}")


def n_dofs(n_levels: int, n_coarse: int, dim: int) -> int:
    """Total DOF count ``N**dim`` where ``N = n_coarse * 2**n_levels``."""
    side = n_coarse * (2 ** n_levels)
    return side ** dim


# ----------------------------------------------------------------------
# Prediction (vectorised, periodic, roll-based).
# ----------------------------------------------------------------------

def _predict_axis(arr: jax.Array, axis: int, order: int) -> jax.Array:
    """Periodic DD midpoint prediction along one axis.

    Returns an array the same shape as ``arr``; entry ``i`` is the predicted
    midpoint between samples ``i`` and ``i+1`` along ``axis``.
    """
    offsets, weights = _DD_FILTERS[order]
    out = jnp.zeros_like(arr)
    for off, w in zip(offsets, weights):
        out = out + w * jnp.roll(arr, -off, axis=axis)
    return out


# ----------------------------------------------------------------------
# 1D transform.
# ----------------------------------------------------------------------

def synthesis_1d(coeffs: jax.Array, n_levels: int, n_coarse: int,
                 order: int = 4) -> jax.Array:
    """Inverse interpolating-wavelet transform: coeffs -> grid values.

    >>> import jax.numpy as jnp
    >>> c = jnp.zeros(8).at[0].set(1.0)  # single coarse coeff, n_coarse=2
    >>> u = synthesis_1d(c, n_levels=2, n_coarse=2, order=4)
    >>> u.shape
    (8,)
    """
    _check_order(order)
    idx = 0
    vals = jax.lax.dynamic_slice_in_dim(coeffs, idx, n_coarse)
    idx += n_coarse
    cur = n_coarse
    for _ in range(n_levels):
        detail = jax.lax.dynamic_slice_in_dim(coeffs, idx, cur)
        idx += cur
        mids = _predict_axis(vals, 0, order) + detail
        fine = jnp.zeros(2 * cur, dtype=coeffs.dtype)
        fine = fine.at[0::2].set(vals)
        fine = fine.at[1::2].set(mids)
        vals = fine
        cur *= 2
    return vals


def analysis_1d(values: jax.Array, n_levels: int, n_coarse: int,
                order: int = 4) -> jax.Array:
    """Forward transform: grid values -> coeffs (inverse of ``synthesis_1d``)."""
    _check_order(order)
    blocks = []
    vals = values
    for _ in range(n_levels):
        coarse = vals[0::2]
        mids = vals[1::2]
        blocks.append(mids - _predict_axis(coarse, 0, order))
        vals = coarse
    return jnp.concatenate([vals] + blocks[::-1])


# ----------------------------------------------------------------------
# 2D isotropic Mallat transform (3 subbands/level: LH, HL, HH).
# ----------------------------------------------------------------------

def synthesis_2d(coeffs: jax.Array, n_levels: int, n_coarse: int,
                 order: int = 4) -> jax.Array:
    """Inverse 2D isotropic Mallat transform; returns the flattened grid.

    Subband-to-parity map (matches the spike): LH -> (even,odd) [y-midpoints],
    HL -> (odd,even) [x-midpoints], HH -> (odd,odd) [xy-midpoints].
    """
    _check_order(order)
    idx = 0
    sz = n_coarse
    img = jax.lax.dynamic_slice_in_dim(coeffs, idx, sz * sz).reshape(sz, sz)
    idx += sz * sz
    cur = n_coarse
    for _ in range(n_levels):
        dLH = jax.lax.dynamic_slice_in_dim(coeffs, idx, cur * cur).reshape(cur, cur)
        idx += cur * cur
        dHL = jax.lax.dynamic_slice_in_dim(coeffs, idx, cur * cur).reshape(cur, cur)
        idx += cur * cur
        dHH = jax.lax.dynamic_slice_in_dim(coeffs, idx, cur * cur).reshape(cur, cur)
        idx += cur * cur
        py = _predict_axis(img, 1, order)
        px = _predict_axis(img, 0, order)
        pxy = _predict_axis(py, 0, order)
        fine = jnp.zeros((2 * cur, 2 * cur), dtype=coeffs.dtype)
        fine = fine.at[0::2, 0::2].set(img)
        fine = fine.at[0::2, 1::2].set(py + dLH)
        fine = fine.at[1::2, 0::2].set(px + dHL)
        fine = fine.at[1::2, 1::2].set(pxy + dHH)
        img = fine
        cur *= 2
    return img.reshape(-1)


def analysis_2d(values: jax.Array, n_levels: int, n_coarse: int,
                order: int = 4) -> jax.Array:
    """Forward 2D isotropic Mallat transform (inverse of ``synthesis_2d``)."""
    _check_order(order)
    side = int(round(values.shape[0] ** 0.5))
    img = values.reshape(side, side)
    blocks = []  # finest-first; reversed at the end
    for _ in range(n_levels):
        coarse = img[0::2, 0::2]
        py = _predict_axis(coarse, 1, order)
        px = _predict_axis(coarse, 0, order)
        pxy = _predict_axis(py, 0, order)
        dLH = img[0::2, 1::2] - py
        dHL = img[1::2, 0::2] - px
        dHH = img[1::2, 1::2] - pxy
        blocks.append((dLH, dHL, dHH))
        img = coarse
    out = [img.reshape(-1)]
    for dLH, dHL, dHH in blocks[::-1]:
        out += [dLH.reshape(-1), dHL.reshape(-1), dHH.reshape(-1)]
    return jnp.concatenate(out)


# ----------------------------------------------------------------------
# 3D isotropic Mallat transform (7 subbands/level).
# ----------------------------------------------------------------------

# Parity order (matches spike dd_wavelets._PARITIES7).
_PAR3 = ((0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))


def _predict_parity(coarse: jax.Array, parity: Tuple[int, int, int],
                    order: int) -> jax.Array:
    """Predict the sub-lattice of a 3D parity by composing axis predictions."""
    cur = coarse
    for ax in range(3):
        if parity[ax]:
            cur = _predict_axis(cur, ax, order)
    return cur


def synthesis_3d(coeffs: jax.Array, n_levels: int, n_coarse: int,
                 order: int = 4) -> jax.Array:
    """Inverse 3D isotropic Mallat transform; returns the flattened grid."""
    _check_order(order)
    idx = 0
    sz = n_coarse
    img = jax.lax.dynamic_slice_in_dim(coeffs, idx, sz ** 3).reshape(sz, sz, sz)
    idx += sz ** 3
    cur = n_coarse
    for _ in range(n_levels):
        details = []
        for _par in _PAR3:
            d = jax.lax.dynamic_slice_in_dim(coeffs, idx, cur ** 3).reshape(cur, cur, cur)
            idx += cur ** 3
            details.append(d)
        fine = jnp.zeros((2 * cur, 2 * cur, 2 * cur), dtype=coeffs.dtype)
        fine = fine.at[0::2, 0::2, 0::2].set(img)
        for par, d in zip(_PAR3, details):
            pred = _predict_parity(img, par, order)
            fine = fine.at[par[0]::2, par[1]::2, par[2]::2].set(pred + d)
        img = fine
        cur *= 2
    return img.reshape(-1)


def analysis_3d(values: jax.Array, n_levels: int, n_coarse: int,
                order: int = 4) -> jax.Array:
    """Forward 3D isotropic Mallat transform (inverse of ``synthesis_3d``)."""
    _check_order(order)
    side = int(round(values.shape[0] ** (1.0 / 3.0)))
    # guard against float rounding
    while side ** 3 < values.shape[0]:
        side += 1
    img = values.reshape(side, side, side)
    blocks = []
    for _ in range(n_levels):
        coarse = img[0::2, 0::2, 0::2]
        dets = []
        for par in _PAR3:
            sub = img[par[0]::2, par[1]::2, par[2]::2]
            dets.append(sub - _predict_parity(coarse, par, order))
        blocks.append(dets)
        img = coarse
    out = [img.reshape(-1)]
    for dets in blocks[::-1]:
        out += [d.reshape(-1) for d in dets]
    return jnp.concatenate(out)


# ----------------------------------------------------------------------
# Level labels (for Dahmen-Kunoth / hybrid-Jacobi scaling and CDD).
# ----------------------------------------------------------------------

def levels_1d(n_levels: int, n_coarse: int) -> jax.Array:
    labs = [0] * n_coarse
    cur = n_coarse
    for lvl in range(n_levels):
        labs += [lvl] * cur
        cur *= 2
    return jnp.asarray(labs, dtype=jnp.int32)


def levels_2d(n_levels: int, n_coarse: int) -> jax.Array:
    labs = [0] * (n_coarse * n_coarse)
    cur = n_coarse
    for lvl in range(n_levels):
        labs += [lvl] * (3 * cur * cur)
        cur *= 2
    return jnp.asarray(labs, dtype=jnp.int32)


def levels_3d(n_levels: int, n_coarse: int) -> jax.Array:
    labs = [0] * (n_coarse ** 3)
    cur = n_coarse
    for lvl in range(n_levels):
        labs += [lvl] * (7 * cur ** 3)
        cur *= 2
    return jnp.asarray(labs, dtype=jnp.int32)


# ----------------------------------------------------------------------
# Dense materialiser (operator assembly + tests; not on the hot path).
# ----------------------------------------------------------------------

_SYNTH = {1: synthesis_1d, 2: synthesis_2d, 3: synthesis_3d}


def structural_blocks(n_levels: int, n_coarse: int, dim: int):
    """Per-DOF structural block id and the first index of each block.

    The synthesis coefficient layout is: one **coarse** block (``n_coarse**dim``
    DOFs), then per level ``n_subband`` blocks (1 in 1D, 3 in 2D, 7 in 3D) of
    size ``cur**dim`` with ``cur`` doubling each level.  By periodic translation
    invariance, every column *within one block* has the **same** ``L²`` norm, so
    one representative per block reconstructs all N norms exactly (derisk D4,
    validated to machine zero).

    **This is the correct key — NOT ``levels_*()``.**  ``levels_*()`` labels the
    coarse block and the first detail level both ``0`` and carries no subband
    distinction, so a level-only representative is wrong by O(0.15) (D4).

    Returns
    -------
    ids : ``(N,)`` int -- block id per DOF (``0 .. 1 + n_levels*n_subband - 1``).
    reps : ``(n_blocks,)`` int -- the first DOF index of each block.
    """
    n_sub = {1: 1, 2: 3, 3: 7}[dim]
    ids: list[int] = []
    reps: list[int] = []
    bid = 0
    pos = 0
    csz = n_coarse ** dim
    reps.append(pos)
    ids += [bid] * csz
    pos += csz
    cur = n_coarse
    for _ in range(n_levels):
        block = cur ** dim
        for _s in range(n_sub):
            bid += 1
            reps.append(pos)
            ids += [bid] * block
            pos += block
        cur *= 2
    return np.asarray(ids, dtype=np.int64), np.asarray(reps, dtype=np.int64)


def synthesis_transpose(u: jax.Array, n_levels: int, n_coarse: int,
                        order: int = 4, dim: int = 1) -> jax.Array:
    """Apply ``Wᵀ`` (the exact adjoint of synthesis) matrix-free.

    **This is NOT ``analysis``.**  DD interpolating wavelets are non-orthogonal,
    so ``W⁻¹ ≠ Wᵀ``: ``analysis_*`` is the *inverse*, and substituting it for the
    transpose gives an order-unity wrong answer with no error and no shape
    mismatch (derisk D3 measured the mismatch at ~0.9, and
    ``jax.linear_transpose`` matching the dense ``Wᵀ`` to machine zero).  The
    transpose is what appears throughout the *adjoint* path — the RHS projection
    ``Wᵀ f``, the sensor row, ``dJ/da`` — so it must be exact.

    Implemented as ``jax.linear_transpose`` of the (linear) synthesis, which is
    the transpose by construction; no matrix is materialised.
    """
    synth = _SYNTH[dim]
    z = jnp.zeros_like(u)
    (out,) = jax.linear_transpose(
        lambda c: synth(c, n_levels, n_coarse, order), z)(u)
    return out


def synthesis_matrix(n_levels: int, n_coarse: int, order: int = 4,
                     dim: int = 1) -> jax.Array:
    """Materialise the synthesis matrix ``W`` (column ``j`` = basis fn ``j``).

    Built by ``jax.vmap`` of the matrix-free synthesis over the identity --
    used for Galerkin operator assembly and for dense reference checks, not on
    the solve hot path.
    """
    N = n_dofs(n_levels, n_coarse, dim)
    synth = _SYNTH[dim]
    eye = jnp.eye(N)
    cols = jax.vmap(lambda e: synth(e, n_levels, n_coarse, order))(eye)
    # ``cols[j]`` is synthesis(e_j); W[:, j] = cols[j]  =>  W = cols.T
    return cols.T
