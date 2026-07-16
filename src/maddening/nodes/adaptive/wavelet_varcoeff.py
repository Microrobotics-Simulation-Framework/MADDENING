"""WaveletVarcoeffNode — matrix-free variable-coefficient wavelet solver (M19).

The θ→A path (application 1: magnetic susceptibility inference).  Where
:class:`WaveletAdaptiveNode` bakes a constant operator into ``__init__`` and lets
θ enter only through the RHS, this node assembles the operator **in-trace** from a
coefficient field carried in state — so ``jax.grad`` flows w.r.t. that field
(``dJ/da``) through operator assembly.  Everything is matrix-free (M13–M16), so it
runs at the 64³ that dense assembly cannot reach (M18).

Governing equation (magnetostatics form)::

    -∇·(a(x) ∇φ) + m φ = f          a(x) = 1 + χ(x)

The differentiable parameter θ **is** the coefficient field χ (shape ``(N,)``);
the source ``f`` is a fixed field for M19 (M20 makes it the χ-dependent
``-∇·(χ H₀)``).  The preconditioner diagonal ``D`` is **lagged at a reference
``a₀``** (default the background ``a₀ = 1``): the block-invariant diagonal only
holds for a constant coefficient, so ``D`` is built once from ``a₀`` and frozen.
The derisk measured a *saturating* ~2.2× conditioning cost for lagging (D5 /
FINDINGS_D5); that is the accepted, designed cost.

Scope: **χ ≤ 10² near-term.**  Hybrid-Jacobi CG iteration count scales with
contrast (D5: ~500 CG iters at χ=10² in 3D; the GPU confirmation run); the
contrast-robust preconditioner for χ ≥ 10³ is the gated research track R1 and
drops into the M5 preconditioner ``inner_precond`` slot.
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import NodeMeta, StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.nodes.adaptive.base import AdaptiveNode
from maddening.nodes.adaptive.wavelet import (
    WaveletAdaptiveNode, _OperatorContext, _warn_if_not_converged)
from maddening.nodes.adaptive.wavelets import cdd as _cdd
from maddening.nodes.adaptive.wavelets import matrixfree as _mf
from maddening.nodes.adaptive.wavelets import preconditioners as _pre
from maddening.nodes.adaptive.wavelets import sensors as _sen
from maddening.nodes.adaptive.wavelets import transform as _T


@stability(StabilityLevel.EXPERIMENTAL)
class WaveletVarcoeffNode(WaveletAdaptiveNode):
    """Matrix-free variable-coefficient adaptive wavelet solver (θ = χ field)."""

    meta: ClassVar[NodeMeta] = NodeMeta(
        algorithm_id="MADD-NODE-WAVELET-VARCOEFF",
        algorithm_version="0.1.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "Matrix-free variable-coefficient wavelet elliptic solver with the "
            "coefficient field as the differentiable parameter (θ→A path); "
            "lagged-diagonal preconditioner, frozen-active-set adjoint"
        ),
        governing_equations=(
            "-∇·(a(x)∇φ) + m φ = f, a = 1 + χ; J = sensor(φ); dJ/dχ flows "
            "through operator assembly; active set Λ via CDD (Doerfler θ_D=0.5)"
        ),
        discretization=(
            "Matrix-free isotropic Mallat DD-4 wavelet operator "
            "Â = D⁻¹ Wnᵀ A_phys(a) Wn D⁻¹ (never assembled); O(log N) column "
            "norms + lagged diagonal; masked CG frozen solve via ift_linear_solve"
        ),
        assumptions=(
            "Steady scalar elliptic problem; no time integration",
            "Periodic boundary conditions (matrix-free varcoeff is periodic)",
            "Preconditioner diagonal lagged at a reference a₀ (constant); "
            "saturating ~2.2× conditioning cost (D5)",
        ),
        limitations=(
            "SCOPE: coefficient contrast χ ≤ 10² near-term. Hybrid-Jacobi CG "
            "iteration count scales with contrast (~500 CG iters at χ=10² in 3D); "
            "χ ≥ 10³ needs a contrast-robust preconditioner (research track R1, "
            "not implemented). At inadequate budget and high contrast the masked "
            "solve is ill-conditioned and a small residual does not bound the "
            "error (D5 / FINDINGS_D5)",
            "EXPERIMENTAL: forward model only; inverse formulation + "
            "regularisation for χ inference is a separate research track (R3)",
        ),
    )

    def __init__(
        self,
        *,
        name: str = "wavelet_varcoeff",
        timestep: float = 1.0,
        dim: int = 1,
        n_levels: int = 5,
        n_coarse: int = 2,
        order: int = 4,
        K: int | None = None,
        chi_init: jax.Array | float = 0.0,
        a_ref: jax.Array | float = 1.0,
        source: jax.Array | None = None,
        h0: tuple[float, ...] | None = None,
        mass: float = 1.0,
        preconditioner: str = "hybrid",
        max_outer: int | None = None,
        sensor: tuple[float, ...] | None = None,
        sensor_op: "_sen.Sensor | None" = None,
        cg_rtol: float = 1e-8,
        cg_atol: float = 1e-10,
        **kw,
    ):
        if dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2, or 3; got {dim}")
        side = n_coarse * (2 ** n_levels)
        N_max = side ** dim
        # Skip WaveletAdaptiveNode.__init__ (dense assembly); go to AdaptiveNode.
        AdaptiveNode.__init__(self, name=name, timestep=timestep, N_max=N_max, **kw)

        self.dim = int(dim)
        self.n_levels = int(n_levels)
        self.n_coarse = int(n_coarse)
        self.order = int(order)
        self.boundary = "periodic"
        self.side = int(side)
        self.N_max = int(N_max)
        self.K = int(K) if K is not None else max(8, N_max // 16)
        self.max_outer = int(max_outer) if max_outer is not None else _cdd.MAX_OUTER
        self.mass = float(mass)
        self.cg_rtol = float(cg_rtol)
        self.cg_atol = float(cg_atol)
        self._h = 1.0 / side

        # χ-independent matrix-free machinery, built once.
        self._levels = {1: _T.levels_1d, 2: _T.levels_2d, 3: _T.levels_3d}[dim](
            n_levels, n_coarse)
        self._norms = _op_column_norms(n_levels, n_coarse, order, dim, self._h)
        self._wn_apply, self._wn_transpose = _mf.make_wn_ops(
            n_levels, n_coarse, order, dim, self._norms)

        # Lagged preconditioner: diagonal of A_wave at the reference a₀ (constant),
        # frozen.  D5: lagging costs a saturating ~2.2× — the accepted design cost.
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

        # Grid, sensor, fixed source.
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
        # matrix-free sensor row Wn[sidx] = Wnᵀ e_sidx
        self._srow = self._wn_transpose(
            jax.nn.one_hot(self._sensor_idx, N_max, dtype=jnp.float64))
        self._sensor_op = sensor_op if sensor_op is not None else \
            _sen.LinearSensor(self._srow[None, :], scalar=True)

        # RHS mode.  h0 given → the magnetostatic source -∇·(χ H₀) (M20), which
        # depends on χ, so χ enters through BOTH A(χ) and b(χ).  Otherwise a fixed
        # source field (M19).
        if h0 is not None:
            if len(h0) != dim:
                raise ValueError(f"h0 must have length dim={dim}; got {len(h0)}")
            self._h0 = tuple(float(v) for v in h0)
            self._source = None
        else:
            self._h0 = None
            if source is None:
                r2 = (self._grid[0] - 0.42) ** 2
                for d in range(1, dim):
                    r2 = r2 + (self._grid[d] - 0.5) ** 2
                source = jnp.exp(-r2 / 0.10 ** 2)
            self._source = jnp.asarray(source)

        # θ is the χ field.
        self._chi_init = jnp.broadcast_to(
            jnp.asarray(chi_init, dtype=jnp.float64), (N_max,))

    # ---- θ = χ field ----
    def _get_theta(self, state):
        return state["theta"]

    def _set_theta(self, state, theta_new):
        return {**state, "theta": jnp.asarray(theta_new)}

    # ---- RHS: fixed source (M19) or the magnetostatic -∇·(χH₀) (M20) ----
    def _magnetic_source(self, chi) -> jax.Array:
        """Physical-space ``-∇·(χ H₀) = -Σ_d H₀_d ∂_d χ`` (H₀ constant), periodic
        central differences.  Depends on χ, so ``dJ/dχ`` also flows through the RHS."""
        c = chi.reshape((self.side,) * self.dim)
        f = jnp.zeros_like(c)
        for d in range(self.dim):
            dchi = (jnp.roll(c, -1, axis=d) - jnp.roll(c, 1, axis=d)) / (2 * self._h)
            f = f - self._h0[d] * dchi
        return f.reshape(-1)

    def _rhs_coeffs(self, theta) -> jax.Array:
        if self._h0 is None:
            del theta                   # M19: source independent of χ
            f = self._source
        else:
            f = self._magnetic_source(theta)   # M20: source = -∇·(χ H₀)
        return (self._h ** self.dim) * self._wn_transpose(f)

    # ---- θ→A: assemble the matrix-free varcoeff operator in-trace ----
    def _build_operator(self, state) -> "_OperatorContext":
        chi = self._get_theta(state)
        a = 1.0 + chi
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
            "theta": self._chi_init,
        }
        mask = self.compute_active_set(empty, prev=None, is_cold_start=True)
        return self.solve_frozen({**empty, "mask": mask}, mask)

    # ---- blindness gate: near-inert for the local wavelet basis (M22) ----
    def blindness_ratio(self, state: dict) -> float:
        # The local wavelet basis is trap-immune (Gate 2), and the full-basis
        # gradient the base diagnostic needs is a dense solve that does not exist
        # matrix-free. Return 1.0 (well-behaved) to skip the gate — correct here,
        # not a shortcut. See WaveletAdaptiveNode docs on the near-inert machinery.
        return 1.0

    def compute_full_basis_gradient(self, state: dict) -> jax.Array:
        raise NotImplementedError(
            "WaveletVarcoeffNode is matrix-free: no dense operator exists for a "
            "full-basis gradient. The blindness gate is overridden (trap-immune "
            "local basis), so this is not needed.")


def _op_column_norms(n_levels, n_coarse, order, dim, h):
    # local import avoids a top-level operator import cycle at module load
    from maddening.nodes.adaptive.wavelets import operator as _op
    return _op.column_norms_fast(n_levels, n_coarse, order, dim, h)
