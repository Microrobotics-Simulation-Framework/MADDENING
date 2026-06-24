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

import jax.numpy as jnp
import numpy as np

__all__ = ["diagonal_scaling", "Kind"]

Kind = Literal["hybrid", "full", "level", "dk"]


def diagonal_scaling(diag: np.ndarray, levels: np.ndarray, kind: Kind = "hybrid",
                     *, t: float = 1.0) -> jnp.ndarray:
    """Return the symmetric diagonal scaling ``D`` (so ``Â = D⁻¹ A D⁻¹``).

    Parameters
    ----------
    diag : the diagonal of the (unscaled) wavelet operator ``A_wave``.
    levels : per-DOF level label (from ``transform.levels_*``).
    kind : ``"hybrid"`` (default) | ``"full"`` | ``"level"`` | ``"dk"``.
    t : elliptic order for ``"dk"`` (1 for Laplacian, 2 for biharmonic).
    """
    d = np.abs(np.asarray(diag, dtype=np.float64))
    lev = np.asarray(levels).astype(int)
    uniq = sorted(set(lev.tolist()))

    if kind == "full":
        D = np.sqrt(d)
    elif kind == "dk":
        D = 2.0 ** (t * lev)
    elif kind in ("level", "hybrid"):
        D = np.zeros_like(d)
        for i, l in enumerate(uniq):
            m = lev == l
            if kind == "hybrid" and i == 0:
                D[m] = np.sqrt(d[m])            # per-entry at the coarse level
            else:
                D[m] = np.sqrt(d[m].mean())     # level-mean at fine levels
    else:
        raise ValueError(f"unknown preconditioner kind {kind!r}")

    D = np.where(D > 0, D, 1.0)
    return jnp.asarray(D)
