"""Diagonal preconditioners for the wavelet Galerkin operator.

Port of the spike's ``hybrid_jacobi.py::precond``.  The derisking spike
established (FINDINGS continuation Inv 1) that **hybrid** Jacobi -- per-entry at
the coarse level, level-mean at fine levels -- matches full-Jacobi condition
number to four significant figures at O(log N) assembly cost, and is the
production default.  ``full``, ``level``, and ``dk`` (Dahmen-Kunoth ``2^{tj}``,
matrix-free) are provided as alternatives.

The scaling vector ``D`` is computed once at node construction (eager, not on
the JIT hot path), so the per-level reductions use concrete level labels.
``A`` is then preconditioned symmetrically as ``Â = D⁻¹ A D⁻¹``.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["diagonal_scaling", "Kind"]

Kind = Literal["hybrid", "full", "level", "dk"]


def diagonal_scaling(diag, levels, kind: Kind = "hybrid",
                     *, t: float = 1.0) -> jnp.ndarray:
    """Return the symmetric diagonal scaling ``D`` (so ``Â = D⁻¹ A D⁻¹``).

    Fully **JAX-traceable in ``diag``**: the level labels fix a static
    segmentation of the DOFs (computed with NumPy), while every ``diag``-dependent
    reduction uses :func:`jax.ops.segment_sum`, so ``jax.grad`` / ``jax.jit`` flow
    through when ``diag`` comes from an operator assembled in-trace from a
    coefficient field ``a(x)`` (required by the θ→A path).  Called eagerly at node
    construction it returns the same values as the previous NumPy implementation
    (to floating-point round-off; the per-level mean now sums via ``segment_sum``
    rather than a Python loop).

    Parameters
    ----------
    diag : the diagonal of the (unscaled) wavelet operator ``A_wave``.
    levels : per-DOF level label (from ``transform.levels_*``); treated as static.
    kind : ``"hybrid"`` (default) | ``"full"`` | ``"level"`` | ``"dk"``.
    t : elliptic order for ``"dk"`` (1 for Laplacian, 2 for biharmonic).
    """
    d = jnp.abs(jnp.asarray(diag))
    lev_np = np.asarray(levels).astype(int)          # static structure

    if kind == "full":
        D = jnp.sqrt(d)
    elif kind == "dk":
        # Purely structural (no diag dependence): 2^{t·level}.
        D = jnp.asarray(2.0 ** (t * lev_np.astype(np.float64)))
    elif kind in ("level", "hybrid"):
        uniq = sorted(set(lev_np.tolist()))
        # Static segment id per DOF (0..L-1), ascending level order.
        seg_np = np.searchsorted(np.asarray(uniq), lev_np).astype(np.int32)
        seg = jnp.asarray(seg_np)
        n_seg = len(uniq)
        counts = jnp.asarray(np.bincount(seg_np, minlength=n_seg)
                             .astype(np.float64))
        sums = jax.ops.segment_sum(d, seg, num_segments=n_seg)
        level_mean = sums / counts                   # per-segment mean of d
        level_mean_D = jnp.sqrt(level_mean[seg])     # broadcast back per DOF
        if kind == "hybrid":
            # per-entry at the coarse level (min label), level-mean elsewhere
            is_coarse = jnp.asarray(lev_np == uniq[0])
            D = jnp.where(is_coarse, jnp.sqrt(d), level_mean_D)
        else:  # "level"
            D = level_mean_D
    else:
        raise ValueError(f"unknown preconditioner kind {kind!r}")

    return jnp.where(D > 0, D, 1.0)
