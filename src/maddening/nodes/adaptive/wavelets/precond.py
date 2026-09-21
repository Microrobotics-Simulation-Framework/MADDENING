"""Diagonal preconditioners for the wavelet Galerkin operator.

The wavelet operator ``A`` is preconditioned symmetrically,
``A_hat = D^-1 A D^-1``, with a diagonal ``D`` chosen once at node
construction from the level structure of the basis:

``"hybrid"`` (default)
    Per-entry ``sqrt(A_jj)`` on the coarse level, the level mean of
    ``sqrt(A_jj)`` on every finer level.  Matches full Jacobi's
    condition number to four significant figures while needing only one
    representative per level -- the choice the node's validated regime
    was measured with.
``"full"``
    Jacobi, ``D_j = sqrt(A_jj)``.
``"level"``
    Level-mean Jacobi on every level, the coarse one included.
``"dk"``
    The Dahmen-Kunoth scaling ``D_j = 2**(t * level_j)`` for an operator
    of elliptic order ``2t`` [DahmenKunoth1992]_; matrix-free, and the
    classical choice, but for this isotropic operator it is weaker than
    ``"hybrid"`` at every size measured.

``D`` is computed eagerly in NumPy from concrete level labels; it is a
constant of the node, not part of the traced step.

.. [DahmenKunoth1992] Dahmen, W., Kunoth, A. (1992).  Multilevel
   preconditioning.  *Numerische Mathematik* 63, 315-344.
"""

from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

__all__ = ["PRECONDITIONERS", "Kind", "diagonal_scaling"]

Kind = Literal["hybrid", "full", "level", "dk"]

#: The accepted ``kind`` values, in the order the module docstring lists them.
PRECONDITIONERS: tuple[str, ...] = ("hybrid", "full", "level", "dk")


@stability(StabilityLevel.EXPERIMENTAL)
def diagonal_scaling(diag: Any, levels: Any, kind: str = "hybrid", *,
                     t: float = 1.0, dtype: Any = None) -> jax.Array:
    """The symmetric diagonal scaling ``D`` (so that ``A_hat = D^-1 A D^-1``).

    Parameters
    ----------
    diag : array-like
        Diagonal of the unscaled wavelet operator ``A``.
    levels : array-like of int
        Per-function level label, from
        :func:`~maddening.nodes.adaptive.wavelets.transform.level_labels`.
    kind : {"hybrid", "full", "level", "dk"}
        See the module docstring.
    t : float
        Elliptic half-order for ``"dk"`` (``1`` for a second-order operator).
    dtype : optional
        Floating dtype of the result; JAX's canonical float when omitted.

    Returns
    -------
    jax.Array
        Strictly positive vector of the same length as ``diag``; any
        entry that would be zero is replaced by one.

    Raises
    ------
    ValueError
        For a ``kind`` outside :data:`PRECONDITIONERS`.
    """
    if kind not in PRECONDITIONERS:
        raise ValueError(
            f"unknown preconditioner kind {kind!r}; expected one of {PRECONDITIONERS}"
        )
    d = np.abs(np.asarray(diag, dtype=np.float64))
    lev = np.asarray(levels).astype(int)
    uniq = sorted(set(lev.tolist()))

    if kind == "full":
        D = np.sqrt(d)
    elif kind == "dk":
        D = 2.0 ** (float(t) * lev)
    else:  # "level" or "hybrid"
        D = np.zeros_like(d)
        for i, lvl in enumerate(uniq):
            m = lev == lvl
            if kind == "hybrid" and i == 0:
                D[m] = np.sqrt(d[m])            # per entry on the coarse level
            else:
                D[m] = np.sqrt(d[m].mean())     # level mean elsewhere

    D = np.where(D > 0, D, 1.0)
    return jnp.asarray(D, dtype=dtype)
