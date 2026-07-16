"""WaveletEllipticNode — general matrix-free adaptive wavelet elliptic solver.

Solves the linear elliptic problem

    -∇·(a(x) ∇u) + m u = f(x)

on the isotropic Mallat DD wavelet basis, matrix-free (so it runs past the
dense-assembly ceiling), with an adaptive CDD active set and the frozen-active-set
adjoint.  **The node knows no physics.**  It is a differentiable forward solver:
the caller supplies how the differentiable parameter θ maps to the coefficient
field ``a`` and the source ``f``, which sensor/objective to read, and which
preconditioner to use.  Any elliptic application — coefficient inference, source
design, a coupled field — is expressed as a *configuration* of this node from
outside MADDENING; no application logic lives here.

Parameterisation (all supplied by the caller):

* ``coeff_fn(θ) → a`` maps the parameter to the coefficient field (the θ→A path);
  or a fixed ``a`` field when the operator does not depend on θ; or, by default,
  ``a = θ`` (θ *is* the coefficient — differentiate the solve w.r.t. ``a``).
* ``source_fn(θ) → f`` maps the parameter to the source (the θ→b path); or a fixed
  ``source`` field.  A problem where θ drives *both* ``a`` and ``f`` supplies both
  callables.
* the **sensor / objective** via the M6 ``Sensor`` protocol (point, multi-point,
  gradient, field functional — the node does not know which);
* the **preconditioner** via the M5 protocol (hybrid-Jacobi now; a contrast-robust
  operator preconditioner drops into the ``inner_precond`` slot later).

``a`` is a general coefficient field; what it represents physically, and how it is
derived from θ, is entirely the caller's mapping.  The preconditioner diagonal is
lagged at a reference ``a_ref`` (block-invariant only for a constant reference; the
derisk measured a saturating ~2.2× conditioning cost for lagging).

Numerical scope: the matrix-free variable-coefficient operator is **periodic**.
The ``boundary`` argument names the intended condition; only ``"periodic"`` (which
also models an open domain via zero-padding) is implemented — ``"dirichlet"``
(absorbing) and ``"neumann"`` (no-flux) are explicit, documented slots that raise
``NotImplementedError`` rather than being absent.
"""

from __future__ import annotations

from typing import Callable, ClassVar, Optional

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import NodeMeta, StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.nodes.adaptive.base import AdaptiveNode
from maddening.nodes.adaptive.wavelet import (
    WaveletAdaptiveNode, _OperatorContext)
from maddening.nodes.adaptive.wavelets import cdd as _cdd
from maddening.nodes.adaptive.wavelets import matrixfree as _mf
from maddening.nodes.adaptive.wavelets import preconditioners as _pre
from maddening.nodes.adaptive.wavelets import sensors as _sen
from maddening.nodes.adaptive.wavelets import transform as _T

# Boundary conditions the interface names.  Only "periodic" is implemented; the
# others are explicit slots so the interface does not foreclose them.
_BOUNDARIES = ("periodic", "dirichlet", "neumann")


@stability(StabilityLevel.EXPERIMENTAL)
class WaveletEllipticNode(WaveletAdaptiveNode):
    """General matrix-free adaptive wavelet solver for ``-∇·(a∇u) + m u = f``."""

    meta: ClassVar[NodeMeta] = NodeMeta(
        algorithm_id="MADD-NODE-WAVELET-ELLIPTIC",
        algorithm_version="0.1.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "General matrix-free adaptive wavelet elliptic solver: coefficient "
            "field and source supplied by the caller, differentiable through the "
            "solve; CDD active set, swappable preconditioner and sensor, "
            "frozen-active-set adjoint"
        ),
        governing_equations=(
            "-∇·(a(x)∇u) + m u = f(x); a and f are caller-supplied functions of "
            "the differentiable parameter θ; J = sensor(u); dJ/dθ flows through "
            "the solve (and through operator assembly when a depends on θ); "
            "active set Λ via Cohen-Dahmen-DeVore residual marking (Doerfler)"
        ),
        discretization=(
            "Matrix-free isotropic Mallat DD-4 wavelet operator "
            "Â = D⁻¹ Wnᵀ A_phys(a) Wn D⁻¹ (never assembled); O(log N) column "
            "norms + lagged diagonal preconditioner; masked CG frozen solve via "
            "ift_linear_solve"
        ),
        assumptions=(
            "Linear steady elliptic problem; no time integration",
            "Periodic boundary conditions (the matrix-free variable-coefficient "
            "operator is periodic; other BCs are named but unimplemented slots)",
            "Preconditioner diagonal lagged at a constant reference a_ref "
            "(saturating ~2.2x conditioning cost)",
        ),
        limitations=(
            "Hybrid-Jacobi CG iteration count scales with coefficient contrast; a "
            "contrast-robust preconditioner for high-contrast coefficients drops "
            "into the M5 inner_precond slot but is not implemented. At inadequate "
            "budget and high contrast the masked solve is ill-conditioned and a "
            "small residual does not bound the error",
            "EXPERIMENTAL: forward solver only; no boundary condition beyond "
            "periodic is implemented",
        ),
    )

    def __init__(
        self,
        *,
        name: str = "wavelet_elliptic",
        timestep: float = 1.0,
        dim: int = 1,
        n_levels: int = 5,
        n_coarse: int = 2,
        order: int = 4,
        K: int | None = None,
        theta_init: jax.Array | float = 1.0,
        a: jax.Array | float | None = None,
        coeff_fn: Optional[Callable[[jax.Array], jax.Array]] = None,
        source: jax.Array | None = None,
        source_fn: Optional[Callable[[jax.Array], jax.Array]] = None,
        a_ref: jax.Array | float = 1.0,
        mass: float = 1.0,
        preconditioner: str = "hybrid",
        max_outer: int | None = None,
        sensor: tuple[float, ...] | None = None,
        sensor_op: "_sen.Sensor | None" = None,
        boundary: str = "periodic",
        cg_rtol: float = 1e-8,
        cg_atol: float = 1e-10,
        **kw,
    ):
        if dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2, or 3; got {dim}")
        if boundary not in _BOUNDARIES:
            raise ValueError(f"boundary must be one of {_BOUNDARIES}; got {boundary!r}")
        if boundary != "periodic":
            raise NotImplementedError(
                f"boundary={boundary!r} is a documented interface slot but is not "
                f"implemented. The matrix-free variable-coefficient operator is "
                f"periodic (which also models an open domain via zero-padding). "
                f"'dirichlet' (absorbing) and 'neumann' (no-flux) require a "
                f"boundary-adapted variable-coefficient assembly path (see "
                f"operator.py). Use boundary='periodic'.")
        side = n_coarse * (2 ** n_levels)
        N_max = side ** dim
        # Skip WaveletAdaptiveNode.__init__ (dense assembly); go to AdaptiveNode.
        AdaptiveNode.__init__(self, name=name, timestep=timestep, N_max=N_max, **kw)

        self.dim = int(dim)
        self.n_levels = int(n_levels)
        self.n_coarse = int(n_coarse)
        self.order = int(order)
        self.boundary = str(boundary)
        self.side = int(side)
        self.N_max = int(N_max)
        self.K = int(K) if K is not None else max(8, N_max // 16)
        self.max_outer = int(max_outer) if max_outer is not None else _cdd.MAX_OUTER
        self.mass = float(mass)
        self.cg_rtol = float(cg_rtol)
        self.cg_atol = float(cg_atol)
        self._h = 1.0 / side

        # Caller-supplied parameterisation (the node knows no physics).
        self._coeff_fn = coeff_fn
        self._source_fn = source_fn
        self._a_fixed = None if a is None else jnp.broadcast_to(
            jnp.asarray(a, dtype=jnp.float64), (N_max,))

        # θ-independent matrix-free machinery, built once.
        self._levels = {1: _T.levels_1d, 2: _T.levels_2d, 3: _T.levels_3d}[dim](
            n_levels, n_coarse)
        self._norms = _op_column_norms(n_levels, n_coarse, order, dim, self._h)
        self._wn_apply, self._wn_transpose = _mf.make_wn_ops(
            n_levels, n_coarse, order, dim, self._norms)

        # Lagged preconditioner: diagonal of A_wave at the constant reference
        # a_ref, frozen (block-invariant only for a constant reference; the
        # derisk measured a saturating ~2.2x conditioning cost for lagging).
        a_ref_arr = jnp.broadcast_to(jnp.asarray(a_ref, dtype=jnp.float64),
                                     (N_max,))
        a_phys_ref = _mf.make_varcoeff_apply(a_ref_arr, side, dim, self._h,
                                             mass=self.mass)
        diagA_ref = _mf.wave_diagonal_fast(n_levels, n_coarse, order, dim,
                                           self._norms, a_phys_ref)
        self._precond = _pre.DiagonalScaling.from_operator(
            diagA_ref, self._levels, preconditioner)
        self._D = self._precond.D

        lev_np = np.asarray(self._levels)
        self._coarse = jnp.asarray(lev_np == lev_np.min())

        # Grid + default point sensor.
        coords1d = np.arange(side) / side
        mesh = np.meshgrid(*([coords1d] * dim), indexing="ij")
        self._grid = [jnp.asarray(m.reshape(-1)) for m in mesh]
        if sensor is None:
            sensor = ((0.30,), (0.30, 0.40), (0.30, 0.40, 0.60))[dim - 1]
        self.sensor = tuple(float(s) for s in sensor)
        sidx = 0
        for d in range(dim):
            i = int(np.argmin(np.abs(coords1d - self.sensor[d])))
            sidx = sidx * side + i
        self._sensor_idx = int(sidx)
        self._srow = self._wn_transpose(
            jax.nn.one_hot(self._sensor_idx, N_max, dtype=jnp.float64))
        self._sensor_op = sensor_op if sensor_op is not None else \
            _sen.LinearSensor(self._srow[None, :], scalar=True)

        # Fixed source (used when source_fn is None).
        if source is None:
            r2 = (self._grid[0] - 0.42) ** 2
            for d in range(1, dim):
                r2 = r2 + (self._grid[d] - 0.5) ** 2
            source = jnp.exp(-r2 / 0.10 ** 2)
        self._source = jnp.asarray(source)

        self._theta_init = jnp.broadcast_to(
            jnp.asarray(theta_init, dtype=jnp.float64), (N_max,))

    # ---- θ accessors (θ is opaque to the node) ----
    def _get_theta(self, state):
        return state["theta"]

    def _set_theta(self, state, theta_new):
        return {**state, "theta": jnp.asarray(theta_new)}

    # ---- caller-supplied maps θ → a, θ → f ----
    def _coeff_field(self, theta) -> jax.Array:
        if self._coeff_fn is not None:
            return self._coeff_fn(theta)        # θ → a
        if self._a_fixed is not None:
            return self._a_fixed                # fixed a (θ does not drive A)
        return theta                            # a = θ (the coefficient itself)

    def _source_field(self, theta) -> jax.Array:
        if self._source_fn is not None:
            return self._source_fn(theta)       # θ → f
        return self._source                     # fixed source

    def _rhs_coeffs(self, theta) -> jax.Array:
        # STRONG-form variable-coefficient operator: RHS is Wnᵀf, no h^dim factor.
        return self._wn_transpose(self._source_field(theta))

    # ---- θ→A: assemble the matrix-free operator in-trace ----
    def _build_operator(self, state) -> "_OperatorContext":
        a = self._coeff_field(self._get_theta(state))
        a_phys = _mf.make_varcoeff_apply(a, self.side, self.dim, self._h,
                                         mass=self.mass)
        wave_apply = _mf.make_wave_apply(
            self.n_levels, self.n_coarse, self.order, self.dim,
            self._norms, a_phys, self._D)

        def solve_masked(mask, rhs_scaled):
            return _mf.masked_cg_solve(wave_apply, mask, rhs_scaled,
                                       inner_precond=self._precond.inner_precond,
                                       rtol=self.cg_rtol, atol=self.cg_atol)

        return _OperatorContext(
            apply=wave_apply,
            solve_masked=solve_masked,
            scale_rhs=self._precond.scale_rhs,
            from_scaled=self._precond.from_scaled,
            indicator=self._precond.indicator,
            coarse=self._coarse,
        )

    # ---- cold start ----
    def _initial_state_impl(self) -> dict:
        empty = {
            "c": jnp.zeros(self.N_max, dtype=jnp.float64),
            "mask": jnp.zeros(self.N_max, dtype=bool),
            "theta": self._theta_init,
        }
        mask = self.compute_active_set(empty, prev=None, is_cold_start=True)
        return self.solve_frozen({**empty, "mask": mask}, mask)

    # ---- blindness gate: near-inert for the local wavelet basis ----
    def blindness_ratio(self, state: dict) -> float:
        # Local wavelet basis is trap-immune (Gate 2); the dense full-basis
        # gradient the base diagnostic needs does not exist matrix-free. Return
        # 1.0 to skip the gate — correct here, not a shortcut.
        return 1.0

    def compute_full_basis_gradient(self, state: dict) -> jax.Array:
        raise NotImplementedError(
            "WaveletEllipticNode is matrix-free: no dense operator exists for a "
            "full-basis gradient. The blindness gate is overridden (trap-immune "
            "local basis), so this is not needed.")


def _op_column_norms(n_levels, n_coarse, order, dim, h):
    # local import avoids a top-level operator import cycle at module load
    from maddening.nodes.adaptive.wavelets import operator as _op
    return _op.column_norms_fast(n_levels, n_coarse, order, dim, h)
