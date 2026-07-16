"""Swappable preconditioner protocol for the wavelet Galerkin solve.

The preconditioner **owns the coordinate system** in which CDD selection and the
frozen solve operate.  This is the seam that lets a contrast-robust
preconditioner (BPX / AMG, roadmap R1) drop in for high-contrast problems
without a rewrite of the node -- see ``FINDINGS_D5`` for why diagonal scaling
alone is not contrast-robust (κ ∝ contrast).

Two structurally different modes sit behind one protocol:

* **Diagonal (Jacobi family, the default and only implementation today).**
  Solves in *scaled coordinates* ``ĉ = D c`` with ``Â = D⁻¹ A D⁻¹``.  The
  marking indicator is ``|r|``, correct where diagonal scaling makes the basis
  (near-)Riesz-stable in ℓ² -- true for the Laplacian, and the reason it
  degrades at high contrast.  ``inner_precond`` is ``None``: the scaling *is* the
  preconditioner, and the frozen K×K solve is direct.

* **Operator (BPX / AMG, R1, not implemented).**  Leaves coordinates unchanged
  (identity ``to_scaled`` / ``from_scaled``, ``A`` unscaled), supplies an
  ``inner_precond`` ``v -> M⁻¹v`` for the (matrix-free, M16) Krylov solve, and
  marks on ``|M⁻¹ r|``.  A two-sided factor ``Â = LᵀAL`` cannot unify the two
  modes: BPX's natural additive form ``M⁻¹ = Σ_j 2^{-2j} I_j I_jᵀ`` has no cheap
  square root, so ``ĉ`` would not live in ``c``'s space.  Two modes, one
  protocol, is the honest resolution.

Note the two modes do **not** share a marking indicator (``|D⁻¹r|`` vs
``|M⁻¹r|``); they are different error indicators, not two spellings of one.  R1
must validate its own indicator -- it is not inherited (see the plan / D5).
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from maddening.nodes.adaptive.wavelets.precond import diagonal_scaling, Kind

__all__ = ["Preconditioner", "DiagonalScaling"]


@runtime_checkable
class Preconditioner(Protocol):
    """Coordinate-owning preconditioner interface.

    The solve pipeline is:  ``b̂ = scale_rhs(b)`` → solve ``Â ĉ = b̂`` in the
    preconditioner's coordinates (``Â`` from :meth:`scale_operator_dense` or, in
    the matrix-free path, the raw operator plus :attr:`inner_precond`) → recover
    ``c = from_scaled(ĉ)``.  CDD marks on :meth:`indicator` of the scaled
    residual.
    """

    #: ``v -> M⁻¹v`` for a Krylov inner solve, or ``None`` when the scaling
    #: itself is the preconditioner (diagonal mode, direct frozen solve).
    inner_precond: Optional[Callable[[jax.Array], jax.Array]]

    def to_scaled(self, c: jax.Array) -> jax.Array: ...
    def from_scaled(self, c_hat: jax.Array) -> jax.Array: ...
    def scale_operator_dense(self, A: jax.Array) -> jax.Array: ...
    def scale_rhs(self, b: jax.Array) -> jax.Array: ...
    def indicator(self, r_scaled: jax.Array) -> jax.Array: ...


class DiagonalScaling:
    """Two-sided diagonal (Jacobi-family) preconditioner: ``Â = D⁻¹ A D⁻¹``.

    Works in scaled coordinates ``ĉ = D c`` throughout.  This reproduces the
    node's original hand-rolled scaling exactly; it is the default.
    """

    inner_precond: Optional[Callable[[jax.Array], jax.Array]] = None

    def __init__(self, D: jax.Array):
        self.D = D

    def to_scaled(self, c: jax.Array) -> jax.Array:
        return c * self.D

    def from_scaled(self, c_hat: jax.Array) -> jax.Array:
        return c_hat / self.D

    def scale_operator_dense(self, A: jax.Array) -> jax.Array:
        return (A / self.D[:, None]) / self.D[None, :]

    def scale_rhs(self, b: jax.Array) -> jax.Array:
        return b / self.D

    def indicator(self, r_scaled: jax.Array) -> jax.Array:
        return jnp.abs(r_scaled)

    @classmethod
    def from_operator(cls, diag: jax.Array, levels, kind: Kind = "hybrid",
                      *, t: float = 1.0) -> "DiagonalScaling":
        """Build from the (unscaled) operator diagonal and per-DOF level labels.

        ``diagonal_scaling`` is JAX-traceable (M4), so this may be called inside
        a trace when the operator is assembled from ``a(x)`` (the θ→A path).
        """
        return cls(diagonal_scaling(diag, levels, kind, t=t))
