"""TopKAdaptiveNode — toy 1, non-local sine basis.

1D Poisson + Gaussian source + sine eigenbasis + top-K selection.
This is the first concrete :class:`~maddening.nodes.adaptive.AdaptiveNode`
subclass.  It serves three purposes:

1. **Self-test for the framework.**  Exercises the full lifecycle —
   ``__init__`` → ``initial_state`` (with cold-start blindness gate)
   → ``update`` (with ``compute_active_set`` + ``solve_frozen``) —
   end-to-end.
2. **Demonstrates the framework in a non-local basis.**  Sine modes
   ``phi_k(x) = sqrt(2) * sin(k * pi * x)`` have oscillating
   ``phi_k(x_sensor)``; selecting modes by ``|b_k|`` can produce
   wrong-sign solutions near domain boundaries (spike round-4
   Investigation 1).  This subclass reproduces that failure mode
   with ``selection_quantity='b'`` and avoids it with the default
   ``selection_quantity='c'``.
3. **Documents the ift_linear_solve usage pattern.**  Although the
   sine basis has a diagonal operator (so a direct ``c = b / lambda``
   suffices), the implementation calls
   :func:`~maddening.core.solver_utils.ift_linear_solve` to exercise
   that wiring under autodiff.

Mathematical setup
------------------

Solves the steady-state Helmholtz equation

.. math::
    (-\\partial_x^2 + 1) u(x) = f(x; \\theta)

on the open interval ``(0, 1)`` with Dirichlet boundaries
``u(0) = u(1) = 0`` and Gaussian source
``f(x; theta) = exp(-((x - theta) / sigma)^2)``.

The sine basis ``phi_k(x) = sqrt(2) * sin(k * pi * x)`` is a complete
orthonormal eigenbasis of ``A = -d^2/dx^2 + I`` on this domain with
eigenvalues ``lambda_k = (k * pi)^2 + 1``.  In this basis the
operator is diagonal: ``A_kk = lambda_k``, so the masked solve is
``c_k = b_k / lambda_k`` for ``k`` in the active set, ``c_k = 0``
otherwise.

The sensor objective is ``J = u(x_sensor) = sum_k c_k phi_k(x_sensor)``.

Selection criteria
------------------

* ``selection_quantity='c'`` (default).  Selects the top-K basis modes
  by ``|b_k / lambda_k|`` — i.e., by the actual coefficient magnitude
  in the solution.  Avoids the round-4 wrong-sign failure because the
  ``1/lambda_k = 1/((k pi)^2 + 1)`` weighting upweights the low-k
  modes that dominate the solution at any sensor.
* ``selection_quantity='b'``.  Selects by ``|b_k|`` only.
  Reproduces the spike's round-4 boundary failure mode: at
  ``theta ~ 0.04``, near the domain boundary, top-|b| selects high-k
  modes whose ``phi_k(x_sensor)`` alternates in sign, producing a
  wrong-sign sensor reading.  Documented for educational value and
  as a regression guard.

Examples
--------

>>> from maddening.nodes.adaptive.topk import TopKAdaptiveNode
>>> node = TopKAdaptiveNode(theta_init=0.42)
>>> state = node.initial_state()
>>> bool(state['mask'].sum() == node.K)
True
"""

from __future__ import annotations

from typing import Literal, Optional

import jax
import jax.numpy as jnp

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.solver_utils import ift_linear_solve
from maddening.nodes.adaptive.base import AdaptiveNode


@stability(StabilityLevel.STABLE)
class TopKAdaptiveNode(AdaptiveNode):
    """1D Poisson + Gaussian source on the sine eigenbasis, top-K selection.

    Parameters
    ----------
    N : int, default 256
        Number of sine basis modes.
    K : int, default 16
        Active-set budget.
    sigma : float, default 0.04
        Gaussian source width.
    sensor_idx : int or None
        Index into the internal grid at which the sensor reads.
        Default: ``N // 3`` (≈ ``x = 1/3``).
    selection_quantity : {"b", "c"}, default "c"
        See module docstring.
    theta_init : float, default 0.42
        Initial parameter.  ``0.42`` is a known good point (no Palais
        trap, blindness ratio ≈ 0.86).
    """

    def __init__(
        self,
        *,
        name: str = "topk_adaptive",
        timestep: float = 1.0,
        N: int = 256,
        K: int = 16,
        sigma: float = 0.04,
        sensor_idx: Optional[int] = None,
        selection_quantity: Literal["b", "c"] = "c",
        theta_init: float = 0.42,
        **kw,
    ):
        super().__init__(name=name, timestep=timestep, N_max=N, **kw)
        if selection_quantity not in ("b", "c"):
            raise ValueError(
                f"selection_quantity must be 'b' or 'c'; got "
                f"{selection_quantity!r}"
            )
        self.N = int(N)
        self.K = int(K)
        self.sigma = float(sigma)
        self.selection_quantity = selection_quantity
        self._theta_init = float(theta_init)

        # Precompute the basis-evaluation matrix and eigenvalues.
        n_grid = 2 * self.N
        ks = jnp.arange(1, self.N + 1, dtype=jnp.float64)
        x_grid = jnp.linspace(0.0, 1.0, n_grid, dtype=jnp.float64)
        self._ks = ks
        self._x_grid = x_grid
        self._dx = float(x_grid[1] - x_grid[0])
        self._phi = jnp.sin(jnp.pi * jnp.outer(x_grid, ks))   # (n_grid, N)
        self._lambdas = (ks * jnp.pi) ** 2 + 1.0
        self._sensor_idx = (
            int(sensor_idx) if sensor_idx is not None
            else n_grid // 3
        )

    # ---- theta accessors ----
    def _get_theta(self, state):
        return state["theta"]

    def _set_theta(self, state, theta_new):
        return {**state, "theta": jnp.atleast_1d(theta_new)}

    # ---- RHS in the basis ----
    def _rhs_coeffs(self, theta):
        theta_s = jnp.squeeze(theta)
        f = jnp.exp(-((self._x_grid - theta_s) / self.sigma) ** 2)
        return 2.0 * self._dx * (self._phi.T @ f)

    # ---- selection ----
    def compute_active_set(self, state, *, prev=None, is_cold_start=False):
        del prev, is_cold_start
        theta = self._get_theta(state)
        b = self._rhs_coeffs(theta)
        if self.selection_quantity == "b":
            score = jnp.abs(b)
        else:  # "c": magnitude-of-solution coefficient
            score = jnp.abs(b) / self._lambdas
        threshold = jnp.sort(score)[-self.K]
        return jax.lax.stop_gradient(score >= threshold)

    # ---- inner solve ----
    def solve_frozen(self, state, mask):
        theta = self._get_theta(state)
        b = self._rhs_coeffs(theta)
        # Even though A is diagonal here, route through ift_linear_solve
        # to exercise the public solver primitive.  Build a masked
        # operator that returns A_kk * v_k on active indices and v_k on
        # inactive (the identity-on-inactive trick: the solution has
        # c_k = 0 outside the mask, so the inactive-row part is
        # cosmetic but lets us use a single dense solver path).
        diag = jnp.where(mask, self._lambdas, 1.0)
        rhs = jnp.where(mask, b, 0.0)

        def operator_fn(v):
            return diag * v

        # Jacobi preconditioner here is the identity for this diagonal
        # operator (M = diag(A) so M^{-1} A = I), so CG converges in 1
        # iteration.  We still pass it through to exercise the
        # preconditioner code path.
        precond = lambda v: v / diag
        c = ift_linear_solve(
            operator_fn, rhs, solver="cg",
            preconditioner=precond, rtol=1e-12, atol=1e-14,
        )
        return {**state, "c": c, "mask": mask}

    # ---- sensor functional ----
    def _sensor(self, state) -> jax.Array:
        return self._phi[self._sensor_idx] @ state["c"]

    # ---- full-basis gradient ----
    def compute_full_basis_gradient(self, state) -> jax.Array:
        def J_full(theta):
            b = self._rhs_coeffs(theta)
            c = b / self._lambdas
            return self._phi[self._sensor_idx] @ c

        return jax.grad(lambda t: jnp.squeeze(J_full(t)))(self._get_theta(state))

    # ---- cold-start state ----
    def _initial_state_impl(self) -> dict:
        theta = jnp.atleast_1d(jnp.asarray(self._theta_init, dtype=jnp.float64))
        empty = {
            "c": jnp.zeros(self.N, dtype=jnp.float64),
            "mask": jnp.zeros(self.N, dtype=bool),
            "theta": theta,
        }
        mask = self.compute_active_set(empty, prev=None, is_cold_start=True)
        return self.solve_frozen({**empty, "mask": mask}, mask)
