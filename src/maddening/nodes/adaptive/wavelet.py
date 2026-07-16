"""WaveletAdaptiveNode — adaptive Deslauriers-Dubuc wavelet PDE solver.

The production adaptive-wavelet node the derisking spike
(``spikes/wavelet_derisking/``, closed 2026-06-22) was built toward.  It solves

    (-Δ + m) u = f(x; θ)            [or -∇·(a(x)∇u) + m u = f]

on an isotropic Mallat Deslauriers-Dubuc wavelet basis, selecting an adaptive
active set per step with Cohen-Dahmen-DeVore (CDD) residual marking and a
hybrid-Jacobi preconditioner, and exposing exact gradients through the frozen
active-set solve (the ``AdaptiveNode`` frozen-set adjoint).

Structure mirrors :class:`maddening.nodes.adaptive.hierarchical_hat.\
HierarchicalHatAdaptiveNode` (the local-basis BCOO template): the masked
operator is a :class:`jax.experimental.sparse.BCOO` passed to
:func:`maddening.core.solver_utils.ift_linear_solve`.  The wavelet numerics live
in :mod:`maddening.nodes.adaptive.wavelets`.

Spike-confirmed design choices carried in:

* **Isotropic Mallat basis + hybrid-Jacobi** preconditioner (Correction C1).
* **Wrong-sign safety via coarse-inclusion in CDD** (FINDINGS §3), not pure
  locality — CDD always retains the coarse level.
* The inherited blindness / ``symmetry_break`` machinery is **near-inert** here:
  the local wavelet basis is trap-immune (Gate 2), so the cold-start gate is a
  cheap safety net, not an active mechanism.
* **No custom_vjp / stop_gradient mitigation** on the adjoint — autodiff is
  exact between active-set changes, Clarke subgradient at kinks (FINDINGS §6).

Dimensionality is a constructor parameter (``dim`` ∈ {1, 2, 3}); the source
moves along axis 0 (``θ``) with the other axes centred.

Examples
--------

>>> from maddening.nodes.adaptive.wavelet import WaveletAdaptiveNode
>>> node = WaveletAdaptiveNode(dim=1, n_levels=6, theta_init=0.42)
>>> state = node.initial_state()
>>> bool(state['mask'][0])  # coarse DOF always active
True
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np

import warnings

from maddening.core.compliance.metadata import NodeMeta, StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.solver_utils import ift_linear_solve
from maddening.nodes.adaptive.base import AdaptiveNode
from maddening.nodes.adaptive.wavelets import cdd as _cdd
from maddening.nodes.adaptive.wavelets import operator as _op
from maddening.nodes.adaptive.wavelets import precond as _pc
from maddening.nodes.adaptive.wavelets import preconditioners as _pre
from maddening.warnings import ConvergenceWarning


def _warn_if_not_converged(converged: jax.Array) -> None:
    """Emit a ConvergenceWarning (host-side, JIT-safe) when ``converged`` is
    false.  ``jax.debug.callback`` runs on the host at execution time, so the
    warning fires under ``jax.jit`` and ``lax.scan`` alike; it is
    autodiff-transparent, so it does not perturb any gradient."""
    def _cb(conv) -> None:
        if not bool(conv):
            warnings.warn(
                "WaveletAdaptiveNode: CDD active-set selection did not reach "
                "tolerance (budget K or MAX_OUTER exhausted first). The returned "
                "solution may be inaccurate; at high coefficient contrast it can "
                "be worse than zero. Increase K, raise MAX_OUTER, or use a "
                "contrast-robust preconditioner.",
                ConvergenceWarning, stacklevel=2,
            )
    jax.debug.callback(_cb, converged)


@stability(StabilityLevel.EXPERIMENTAL)
class WaveletAdaptiveNode(AdaptiveNode):
    """Adaptive DD-wavelet PDE solver (CDD selection, hybrid-Jacobi, frozen adjoint).

    Parameters
    ----------
    dim : int, default 1
        Spatial dimension (1, 2, or 3).
    n_levels : int, default 6
        Refinement levels; grid side ``= n_coarse * 2**n_levels``,
        ``N_max = side**dim``.
    n_coarse : int, default 2
        Coarse-grid points per axis.
    order : int, default 4
        Deslauriers-Dubuc order (DD-4 is the validated production default).
    K : int, optional
        Active-set budget for CDD.  Defaults to ``N_max // 16`` (the spike's
        sparsity target).
    sigma : float, default 0.10
        Gaussian source width.
    sensor : tuple[float, ...], optional
        Sensor location; defaults to an off-axis point per ``dim``.
    theta_init : float, default 0.42
        Initial source position along axis 0.
    mass : float, default 1.0
        Zeroth-order term coefficient.
    preconditioner : str, default "hybrid"
        ``"hybrid"`` | ``"full"`` | ``"level"`` | ``"dk"``.
    """

    meta: ClassVar[NodeMeta] = NodeMeta(
        algorithm_id="MADD-NODE-WAVELET",
        algorithm_version="0.1.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "Adaptive Deslauriers-Dubuc wavelet PDE solver: isotropic Mallat "
            "basis, CDD residual-driven active set, hybrid-Jacobi "
            "preconditioner, frozen-active-set adjoint"
        ),
        governing_equations=(
            "(-Δ + m) u(x) = f(x; θ)  [or -∇·(a(x)∇u) + m u = f]; "
            "J = u(x_sensor); active set Λ via Cohen-Dahmen-DeVore residual "
            "marking (Doerfler θ_D=0.5)"
        ),
        discretization=(
            "Isotropic Mallat Deslauriers-Dubuc (DD-4) interpolating wavelet "
            "basis; Galerkin BCOO operator A_wave = Wn^T A_phys Wn; "
            "hybrid-Jacobi diagonal scaling; CG inner solve via ift_linear_solve"
        ),
        assumptions=(
            "Steady scalar elliptic problem: the node solves a single "
            "boundary-value problem per update() and has no time integration",
            "Periodic or homogeneous-Dirichlet boundary conditions "
            "(variable-coefficient assembly is periodic-only)",
            "H^1 solution regularity (bounded/step RHS); Besov regime untested",
            "Source localised along axis 0 (theta); other axes centred",
        ),
        limitations=(
            "SCOPE: this is a steady scalar elliptic solver. It has no "
            "convection, no velocity, no pressure and no time integration, and "
            "is therefore NOT a fluid/Navier-Stokes solver. The lid-driven "
            "cavity under benchmarks/ is a NumPy/SciPy finite-difference "
            "reference solver; this node does not participate in it "
            "(MADD-VER-CAVITY-FD-100 is registered against that FD reference, "
            "not against this node)",
            "Cross-validated as an elliptic solver only: manufactured solutions "
            "in 1D/2D/3D and cross-code vs MIME's FFT-spectral Helmholtz solver "
            "in 2D/3D (MADD-VER-WAVELET-001..005). No validation exists against "
            "a flow solver or on any time-dependent problem",
            "Operator assembly is dense O(N^2) in memory with N = side**dim; the "
            "designed matrix-free matvec is not implemented. Validated sizes are "
            "~64^2 in 2D and 8^3-16^3 in 3D; the node does not scale past them",
            "CDD outer loop is unrolled to MAX_OUTER=30 (round-6 decision); "
            "no multi-GPU sharding (single-device only)",
            "EXPERIMENTAL: see spikes/wavelet_derisking/KNOWN_LIMITATIONS.md",
        ),
    )

    def __init__(
        self,
        *,
        name: str = "wavelet_adaptive",
        timestep: float = 1.0,
        dim: int = 1,
        n_levels: int = 6,
        n_coarse: int = 2,
        order: int = 4,
        K: int | None = None,
        sigma: float = 0.10,
        sensor: tuple[float, ...] | None = None,
        theta_init: float = 0.42,
        mass: float = 1.0,
        preconditioner: str = "hybrid",
        boundary: str = "periodic",
        max_outer: int | None = None,
        **kw,
    ):
        if dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2, or 3; got {dim}")
        if boundary not in ("periodic", "dirichlet"):
            raise ValueError(f"boundary must be 'periodic' or 'dirichlet'; got {boundary!r}")
        if boundary == "dirichlet":
            from maddening.nodes.adaptive.wavelets.dirichlet import dirichlet_side
            side = dirichlet_side(n_levels, n_coarse)
        else:
            side = n_coarse * (2 ** n_levels)
        N_max = side ** dim
        super().__init__(name=name, timestep=timestep, N_max=N_max, **kw)

        self.dim = int(dim)
        self.n_levels = int(n_levels)
        self.n_coarse = int(n_coarse)
        self.order = int(order)
        self.boundary = str(boundary)
        self.side = int(side)
        self.N_max = int(N_max)
        self.K = int(K) if K is not None else max(8, N_max // 16)
        self.max_outer = int(max_outer) if max_outer is not None else _cdd.MAX_OUTER
        self.sigma = float(sigma)
        self.mass = float(mass)
        self._theta_init = float(theta_init)

        if sensor is None:
            sensor = ((0.30,), (0.30, 0.40), (0.30, 0.40, 0.60))[dim - 1]
        self.sensor = tuple(float(s) for s in sensor)

        # Assemble the Galerkin operator (constant-coefficient) once.
        res = _op.assemble_wave_operator(
            self.n_levels, self.n_coarse, order=self.order, dim=self.dim,
            mass=self.mass, boundary=self.boundary,
        )
        self._A = res["A_dense"]          # unscaled (for full-basis gradient)
        self._Wn = res["Wn"]              # L2-normalised synthesis
        self._levels = res["levels"]
        self._h = res["h"]

        # Preconditioner seam (M5): the preconditioner owns the coordinate
        # system.  DiagonalScaling reproduces the original hybrid-Jacobi path
        # exactly; a contrast-robust operator preconditioner (R1) drops into the
        # same protocol.  Computed once, eager.
        self._precond = _pre.DiagonalScaling.from_operator(
            jnp.diag(self._A), self._levels, preconditioner)
        self._D = self._precond.D                       # kept for compatibility
        self._Ah = self._precond.scale_operator_dense(self._A)
        # Scaled operator as a constant BCOO (assembled once, static nse) for an
        # O(nnz) jit-safe matvec on the masked-solve / CDD hot path.
        self._Ah_bcoo = jsparse.BCOO.fromdense(self._Ah)

        lev_np = np.asarray(self._levels)
        self._coarse = jnp.asarray(lev_np == lev_np.min())

        # Grid coordinates and sensor index (flattened, row-major).
        # Periodic: x_i = i/side.  Dirichlet: interior nodes x_i = i/(side+1).
        if self.boundary == "dirichlet":
            coords1d = np.arange(1, self.side + 1) / (self.side + 1)
        else:
            coords1d = np.arange(self.side) / self.side
        mesh = np.meshgrid(*([coords1d] * self.dim), indexing="ij")
        self._grid = [jnp.asarray(m.reshape(-1)) for m in mesh]
        sidx = 0
        for d in range(self.dim):
            i = int(np.argmin(np.abs(coords1d - self.sensor[d])))
            sidx = sidx * self.side + i
        self._sensor_idx = int(sidx)
        self._srow = self._Wn[self._sensor_idx]

    # ---- theta accessors ----
    def _get_theta(self, state):
        return state["theta"]

    def _set_theta(self, state, theta_new):
        return {**state, "theta": jnp.atleast_1d(theta_new)}

    # ---- RHS: project the moving Gaussian source onto the wavelet basis ----
    def _rhs_coeffs(self, theta) -> jax.Array:
        theta_s = jnp.squeeze(theta)
        r2 = (self._grid[0] - theta_s) ** 2
        for d in range(1, self.dim):
            r2 = r2 + (self._grid[d] - 0.5) ** 2
        f = jnp.exp(-r2 / self.sigma ** 2)
        return (self._h ** self.dim) * (self._Wn.T @ f)

    def _scaled_rhs(self, theta) -> jax.Array:
        return self._precond.scale_rhs(self._rhs_coeffs(theta))

    # ---- selection: CDD residual marking (returns a frozen mask) ----
    def _solve_masked(self, mask, rhs_scaled):
        # Gather the K active DOFs into a fixed-size buffer and solve the dense
        # K×K system directly (O(K^3), realises the adaptivity speedup) instead
        # of an O(N) iterative solve on the full masked operator.  CDD caps the
        # active set at K, so buf=K suffices.
        return _op.gather_solve(self._Ah, mask, rhs_scaled, self.K)

    def compute_active_set(self, state, *, prev=None, is_cold_start=False):
        del prev, is_cold_start
        bh = self._scaled_rhs(self._get_theta(state))
        mask, _, _conv = _cdd.cdd_select(
            lambda v: self._Ah_bcoo @ v, self._solve_masked, bh,
            self._coarse, self.K, max_outer=self.max_outer,
            indicator=self._precond.indicator,
        )
        return jax.lax.stop_gradient(mask)

    def update(self, state, boundary_inputs, dt):
        """One adaptive step, surfacing CDD non-convergence.

        Identical to the base ``compute_active_set`` → ``solve_frozen`` flow,
        but captures CDD's ``converged`` flag and emits a
        :class:`~maddening.warnings.ConvergenceWarning` (JIT-safe, via
        ``jax.debug.callback``) when the active-set solve exhausted its budget
        or iteration ceiling before reaching tolerance.  This is what stops a
        high-contrast solve from silently returning a worse-than-zero result
        (see ``spikes/wavelet_apps/FINDINGS_D5``).  The frozen re-solve in
        :meth:`solve_frozen` is unchanged, so the gradient path is unaffected.
        """
        del boundary_inputs, dt
        bh = self._scaled_rhs(self._get_theta(state))
        mask, _, converged = _cdd.cdd_select(
            lambda v: self._Ah_bcoo @ v, self._solve_masked, bh,
            self._coarse, self.K, max_outer=self.max_outer,
            indicator=self._precond.indicator,
        )
        _warn_if_not_converged(converged)
        mask = jax.lax.stop_gradient(mask)
        return self.solve_frozen(state, mask)

    # ---- frozen inner solve (the adjoint flows through here) ----
    def solve_frozen(self, state, mask):
        bh = self._scaled_rhs(self._get_theta(state))
        c_hat = self._solve_masked(mask, bh)
        c = self._precond.from_scaled(c_hat)   # physical wavelet coefficients
        return {**state, "c": c, "mask": mask}

    # ---- sensor functional J = u(x_sensor) ----
    def _sensor(self, state) -> jax.Array:
        return self._srow @ state["c"]

    # ---- full-basis gradient (no mask) for the blindness diagnostic ----
    def compute_full_basis_gradient(self, state) -> jax.Array:
        def J_full(theta):
            b = self._rhs_coeffs(theta)
            c = jnp.linalg.solve(self._A, b)
            return self._srow @ c

        return jax.grad(lambda t: jnp.squeeze(J_full(t)))(self._get_theta(state))

    # ---- cold-start state ----
    def _initial_state_impl(self) -> dict:
        theta = jnp.atleast_1d(jnp.asarray(self._theta_init, dtype=jnp.float64))
        empty = {
            "c": jnp.zeros(self.N_max, dtype=jnp.float64),
            "mask": jnp.zeros(self.N_max, dtype=bool),
            "theta": theta,
        }
        mask = self.compute_active_set(empty, prev=None, is_cold_start=True)
        return self.solve_frozen({**empty, "mask": mask}, mask)
