"""Interpolating (Deslauriers-Dubuc) wavelet transforms, matrix-free in JAX.

The basis :class:`~maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode`
solves in.  Both directions are lifting schemes built from one primitive,
the periodic midpoint *prediction*: a value halfway between two coarse
samples is interpolated from the ``order`` nearest coarse samples with
the Deslauriers-Dubuc filter [DeslauriersDubuc1989]_.  Synthesis
(coefficients to grid values) predicts and adds the detail; analysis
(grid values to coefficients) predicts and subtracts it.  Every shape is
static, the per-level Python loop unrolls at trace time, and the
prediction is a sum of ``jnp.roll`` calls, so both transforms run under
``jax.jit`` / ``jax.vmap`` unchanged.

Three spatial dimensionalities, all on the **isotropic Mallat**
multiresolution (one resolution level per refinement step, ``2**dim - 1``
detail subbands per level):

* 1-D: ``[coarse(n_coarse), detail_0(n_coarse), detail_1(2 n_coarse), ...]``.
* 2-D: three subbands per level (LH, HL, HH).
* 3-D: seven subbands per level (every parity except LLL).

Only periodic boundaries live here; the boundary-adapted basis for
homogeneous Dirichlet conditions is :mod:`.dirichlet`.

Precision
---------
Nothing here fixes a dtype.  The transforms inherit the dtype of the
array they are given, and :func:`synthesis_matrix` takes an explicit
``dtype`` so that the node can build its operator at the precision the
:class:`~maddening.nodes.adaptive.base.AdaptiveNode` base class resolved
(float32 by default, float64 under ``jax_enable_x64``, or whatever
``dtype=`` the node was constructed with).

.. [DeslauriersDubuc1989] Deslauriers, G., Dubuc, S. (1989).  Symmetric
   iterative interpolation processes.  *Constructive Approximation* 5,
   49-68.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

__all__ = [
    "DD_ORDERS",
    "n_dofs",
    "side_length",
    "synthesis",
    "analysis",
    "synthesis_matrix",
    "level_labels",
]

# Deslauriers-Dubuc interpolating-subdivision prediction filters.  A
# midpoint between coarse samples ``0`` and ``1`` is predicted from the
# ``order`` nearest coarse samples; offsets are relative to the left
# neighbour.  Symmetric, and they sum to one, so a constant is
# reproduced exactly; DD-``2N`` reproduces polynomials of degree
# ``2N - 1``.
_DD_FILTERS: dict[int, tuple[tuple[int, ...], tuple[float, ...]]] = {
    2: ((0, 1), (0.5, 0.5)),
    4: ((-1, 0, 1, 2), (-1.0 / 16, 9.0 / 16, 9.0 / 16, -1.0 / 16)),
    6: ((-2, -1, 0, 1, 2, 3),
        (3.0 / 256, -25.0 / 256, 150.0 / 256, 150.0 / 256, -25.0 / 256, 3.0 / 256)),
}

#: The interpolating orders implemented: ``(2, 4, 6)``.  ``4`` is the
#: node's default (approximation order 4, four vanishing moments).
DD_ORDERS: tuple[int, ...] = tuple(sorted(_DD_FILTERS))

# Detail-subband parities in 3-D, in coefficient-layout order.
_PARITIES_3D = (
    (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
)


def _check_order(order: int) -> None:
    if order not in _DD_FILTERS:
        raise ValueError(
            f"unsupported Deslauriers-Dubuc order {order!r}; supported: {DD_ORDERS}"
        )


def _check_dim(dim: int) -> None:
    if dim not in (1, 2, 3):
        raise ValueError(f"dim must be 1, 2 or 3, got {dim!r}")


@stability(StabilityLevel.EXPERIMENTAL)
def side_length(n_levels: int, n_coarse: int) -> int:
    """Grid points per axis after ``n_levels`` periodic refinements.

    ``n_coarse * 2**n_levels``.
    """
    return int(n_coarse) * (2 ** int(n_levels))


@stability(StabilityLevel.EXPERIMENTAL)
def n_dofs(n_levels: int, n_coarse: int, dim: int) -> int:
    """Total number of basis functions, ``side_length(...) ** dim``."""
    _check_dim(dim)
    return side_length(n_levels, n_coarse) ** int(dim)


# ---------------------------------------------------------------------------
# Prediction (vectorised, periodic, roll-based)
# ---------------------------------------------------------------------------

def _predict_axis(arr: jax.Array, axis: int, order: int) -> jax.Array:
    """Periodic Deslauriers-Dubuc midpoint prediction along one axis.

    Entry ``i`` of the result is the predicted value halfway between
    samples ``i`` and ``i + 1`` along ``axis``; the result has the same
    shape as ``arr``.
    """
    offsets, weights = _DD_FILTERS[order]
    out = jnp.zeros_like(arr)
    for off, w in zip(offsets, weights):
        out = out + w * jnp.roll(arr, -off, axis=axis)
    return out


def _predict_parity(coarse: jax.Array, parity: tuple[int, ...], order: int) -> jax.Array:
    """Predict one sub-lattice by composing axis predictions."""
    cur = coarse
    for ax, odd in enumerate(parity):
        if odd:
            cur = _predict_axis(cur, ax, order)
    return cur


# ---------------------------------------------------------------------------
# Per-dimension transforms
# ---------------------------------------------------------------------------

def _synthesis_1d(coeffs: jax.Array, n_levels: int, n_coarse: int, order: int) -> jax.Array:
    idx = 0
    vals = coeffs[idx:idx + n_coarse]
    idx += n_coarse
    cur = n_coarse
    for _ in range(n_levels):
        detail = coeffs[idx:idx + cur]
        idx += cur
        mids = _predict_axis(vals, 0, order) + detail
        fine = jnp.zeros(2 * cur, dtype=coeffs.dtype)
        fine = fine.at[0::2].set(vals)
        fine = fine.at[1::2].set(mids)
        vals = fine
        cur *= 2
    return vals


def _analysis_1d(values: jax.Array, n_levels: int, n_coarse: int, order: int) -> jax.Array:
    del n_coarse  # implied by values.shape and n_levels
    blocks = []
    vals = values
    for _ in range(n_levels):
        coarse = vals[0::2]
        mids = vals[1::2]
        blocks.append(mids - _predict_axis(coarse, 0, order))
        vals = coarse
    return jnp.concatenate([vals] + blocks[::-1])


def _synthesis_nd(coeffs: jax.Array, n_levels: int, n_coarse: int, order: int, dim: int) -> jax.Array:
    """Isotropic Mallat synthesis in 2-D or 3-D; returns the flattened grid."""
    parities = [p for p in _all_parities(dim) if any(p)]
    idx = 0
    block = n_coarse ** dim
    img = coeffs[idx:idx + block].reshape((n_coarse,) * dim)
    idx += block
    cur = n_coarse
    for _ in range(n_levels):
        details = []
        for _p in parities:
            d = coeffs[idx:idx + cur ** dim].reshape((cur,) * dim)
            idx += cur ** dim
            details.append(d)
        fine = jnp.zeros((2 * cur,) * dim, dtype=coeffs.dtype)
        fine = fine.at[tuple(slice(0, None, 2) for _ in range(dim))].set(img)
        for par, d in zip(parities, details):
            pred = _predict_parity(img, par, order)
            sl = tuple(slice(p, None, 2) for p in par)
            fine = fine.at[sl].set(pred + d)
        img = fine
        cur *= 2
    return img.reshape(-1)


def _analysis_nd(values: jax.Array, n_levels: int, n_coarse: int, order: int, dim: int) -> jax.Array:
    parities = [p for p in _all_parities(dim) if any(p)]
    side = side_length(n_levels, n_coarse)
    img = values.reshape((side,) * dim)
    blocks = []  # finest first; reversed at the end
    for _ in range(n_levels):
        coarse = img[tuple(slice(0, None, 2) for _ in range(dim))]
        dets = []
        for par in parities:
            sub = img[tuple(slice(p, None, 2) for p in par)]
            dets.append(sub - _predict_parity(coarse, par, order))
        blocks.append(dets)
        img = coarse
    out = [img.reshape(-1)]
    for dets in blocks[::-1]:
        out += [d.reshape(-1) for d in dets]
    return jnp.concatenate(out)


def _all_parities(dim: int) -> tuple[tuple[int, ...], ...]:
    if dim == 2:
        # LH -> (even, odd), HL -> (odd, even), HH -> (odd, odd)
        return ((0, 0), (0, 1), (1, 0), (1, 1))
    return ((0, 0, 0),) + _PARITIES_3D


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

@stability(StabilityLevel.EXPERIMENTAL)
def synthesis(coeffs: jax.Array, n_levels: int, n_coarse: int, *,
              order: int = 4, dim: int = 1) -> jax.Array:
    """Inverse transform: wavelet coefficients to grid values.

    Parameters
    ----------
    coeffs : jax.Array
        Coefficient vector of length :func:`n_dofs`.
    n_levels, n_coarse : int
        Refinement levels and coarse points per axis.
    order : {2, 4, 6}
        Interpolating order.
    dim : {1, 2, 3}
        Spatial dimension.

    Returns
    -------
    jax.Array
        Grid values, flattened row-major, length :func:`n_dofs`.

    Examples
    --------
    A single unit coarse coefficient synthesises to a bump that is exactly
    one at its own node and interpolates smoothly between:

    >>> import jax.numpy as jnp
    >>> c = jnp.zeros(8).at[0].set(1.0)          # n_coarse=2, n_levels=2
    >>> u = synthesis(c, n_levels=2, n_coarse=2, order=4)
    >>> u.shape
    (8,)
    >>> bool(jnp.isclose(u[0], 1.0)), bool(jnp.isclose(u[4], 0.0))
    (True, True)
    """
    _check_order(order)
    _check_dim(dim)
    if dim == 1:
        return _synthesis_1d(coeffs, n_levels, n_coarse, order)
    return _synthesis_nd(coeffs, n_levels, n_coarse, order, dim)


@stability(StabilityLevel.EXPERIMENTAL)
def analysis(values: jax.Array, n_levels: int, n_coarse: int, *,
             order: int = 4, dim: int = 1) -> jax.Array:
    """Forward transform: grid values to wavelet coefficients.

    The exact inverse of :func:`synthesis` (a lifting scheme is
    invertible by construction, independent of the filter).

    Parameters
    ----------
    values : jax.Array
        Grid values, flattened row-major, length :func:`n_dofs`.
    n_levels, n_coarse, order, dim
        As for :func:`synthesis`.

    Returns
    -------
    jax.Array
        Coefficients in the layout described in the module docstring.
    """
    _check_order(order)
    _check_dim(dim)
    if dim == 1:
        return _analysis_1d(values, n_levels, n_coarse, order)
    return _analysis_nd(values, n_levels, n_coarse, order, dim)


@stability(StabilityLevel.EXPERIMENTAL)
def level_labels(n_levels: int, n_coarse: int, dim: int = 1) -> jax.Array:
    """Resolution level of every basis function, ``int32`` of length :func:`n_dofs`.

    The coarse block is level ``0`` and the details added by refinement
    ``j`` (``j = 0, ..., n_levels - 1``) are level ``j``, so the coarse
    block and the first detail band share a label.  This is what the
    diagonal preconditioners and the CDD coarse seed read.
    """
    _check_dim(dim)
    per_level = 2 ** dim - 1
    labs = [0] * (n_coarse ** dim)
    cur = n_coarse
    for lvl in range(n_levels):
        labs += [lvl] * (per_level * cur ** dim)
        cur *= 2
    return jnp.asarray(labs, dtype=jnp.int32)


@stability(StabilityLevel.EXPERIMENTAL)
def synthesis_matrix(n_levels: int, n_coarse: int, *, order: int = 4,
                     dim: int = 1, dtype: Any = None) -> jax.Array:
    """Materialise the synthesis matrix ``W`` (column ``j`` is basis function ``j``).

    Built by ``jax.vmap`` of the matrix-free :func:`synthesis` over the
    identity.  Used for Galerkin operator assembly and for dense reference
    checks -- it is not on the solve hot path.

    Parameters
    ----------
    dtype : optional
        Floating dtype of the result; JAX's canonical float when omitted.
    """
    _check_order(order)
    _check_dim(dim)
    n = n_dofs(n_levels, n_coarse, dim)
    eye = jnp.eye(n, dtype=dtype)
    synth: Callable[[jax.Array], jax.Array] = (
        lambda e: synthesis(e, n_levels, n_coarse, order=order, dim=dim)
    )
    # ``jit`` around the ``vmap``: a bare vmap executes op by op and every
    # batched scatter and roll is compiled separately on first use -- about
    # 60 compiles and 11 s for a 128-point basis.  One fused compile is a
    # fraction of a second, and this runs once per node.
    cols = jax.jit(jax.vmap(synth))(eye)
    # ``cols[j]`` is synthesis(e_j) = W[:, j], so W = cols.T
    return cols.T
