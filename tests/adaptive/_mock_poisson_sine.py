"""Mock ``AdaptiveNode`` subclass used by M3 and M4 tests.

Reproduces the 1D Poisson + Gaussian source + sine eigenbasis toy
from the spike rounds:

    (-d^2/dx^2 + 1) u(x) = exp(-((x - theta) / sigma)^2)
    on (0, 1), Dirichlet BCs

with sine basis ``phi_k(x) = sqrt(2) * sin(k * pi * x)`` and
eigenvalues ``lambda_k = (k * pi)^2 + 1``.  ``A`` is diagonal in
this basis, so the masked solve is trivial.  Sensor objective
``J = u(x_sensor)`` with default ``x_sensor = 0.333``.

This is the same problem the spike's ``q2_frozen_set_gradient.py``,
``trap_characterisation.py`` etc. used.  Numerical expectations at
known theta values are documented in the spike findings memo
(round-3 prevalence sweep, round-6 threshold calibration).
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from maddening.nodes.adaptive import AdaptiveNode


# Constants chosen to match the spike toys.
N_BASIS = 256
N_GRID = 512
SIGMA = 0.04
K_ACTIVE = 16


def _build_static():
    ks = jnp.arange(1, N_BASIS + 1, dtype=jnp.float64)
    lambdas = (ks * jnp.pi) ** 2 + 1.0
    x_grid = jnp.linspace(0.0, 1.0, N_GRID, dtype=jnp.float64)
    phi = jnp.sin(jnp.pi * jnp.outer(x_grid, ks))
    return ks, lambdas, x_grid, phi


_KS, _LAMBDAS, _X_GRID, _PHI = _build_static()
_DX = float(_X_GRID[1] - _X_GRID[0])
_SENSOR_IDX = N_GRID // 3   # x_sensor ~ 0.333, matches spike toys


class MockPoissonSineNode(AdaptiveNode):
    """1D Poisson sine-basis AdaptiveNode with top-|b| selection.

    Top-|b| is chosen (rather than top-|c|) because the spike's round-3
    blindness data was generated with top-|b|; this lets the M3 tests
    assert on the exact numerical values from
    ``trap_characterisation.py``.
    """

    def __init__(self, theta_init: float = 0.42, **kw):
        super().__init__(N_max=N_BASIS, **kw)
        self._theta_init = float(theta_init)

    # ---- theta accessors ----
    def _get_theta(self, state):
        return state["theta"]

    def _set_theta(self, state, theta_new):
        return {**state, "theta": jnp.atleast_1d(theta_new)}

    # ---- selection ----
    def _rhs_coeffs(self, theta) -> jax.Array:
        """Project the Gaussian source onto the sine basis."""
        theta_s = jnp.squeeze(theta)
        f = jnp.exp(-((_X_GRID - theta_s) / SIGMA) ** 2)
        return 2.0 * _DX * (_PHI.T @ f)

    def compute_active_set(self, state, *, prev=None, is_cold_start=False):
        del prev, is_cold_start
        theta = self._get_theta(state)
        b = self._rhs_coeffs(theta)
        mag = jnp.abs(b)
        threshold = jnp.sort(mag)[-K_ACTIVE]
        return jax.lax.stop_gradient(mag >= threshold)

    # ---- inner solve ----
    def solve_frozen(self, state, mask):
        theta = self._get_theta(state)
        b = self._rhs_coeffs(theta)
        # A is diagonal in this basis -- no need to call
        # ift_linear_solve here; the masked solve is c = b / lambda on
        # the mask.  ift_linear_solve is exercised in M5 (TopK toy).
        c = jnp.where(mask, b / _LAMBDAS, 0.0)
        return {**state, "c": c, "mask": mask}

    # ---- sensor objective ----
    def _sensor(self, state) -> jax.Array:
        c = state["c"]
        # u(x_sensor) = phi_sensor . c
        return _PHI[_SENSOR_IDX] @ c

    # ---- full-basis gradient ----
    def compute_full_basis_gradient(self, state) -> jax.Array:
        """∇_θ J_full: the gradient under no masking.

        For this 1D scalar-theta problem the gradient is a scalar
        (returned as shape ``(1,)`` for consistency with multi-D
        subclasses).
        """
        def J_full(theta):
            b = self._rhs_coeffs(theta)
            c = b / _LAMBDAS
            return _PHI[_SENSOR_IDX] @ c

        return jax.grad(lambda t: jnp.squeeze(J_full(t)))(self._get_theta(state))

    # ---- cold-start state ----
    def _initial_state_impl(self) -> dict:
        theta = jnp.atleast_1d(jnp.asarray(self._theta_init, dtype=jnp.float64))
        state = {
            "c": jnp.zeros(N_BASIS, dtype=jnp.float64),
            "mask": jnp.zeros(N_BASIS, dtype=bool),
            "theta": theta,
        }
        # Materialise a real state via one solve so blindness checks
        # have a meaningful mask to read.
        mask = self.compute_active_set(state, prev=None, is_cold_start=True)
        return self.solve_frozen({**state, "mask": mask}, mask)


def make_node(theta: float = 0.42, **kw) -> MockPoissonSineNode:
    """Convenience factory used by M3/M4 tests."""
    node = MockPoissonSineNode(theta_init=theta, **kw)
    return node


def state_at(theta: float, **kw) -> dict:
    """Build the ready-to-test state at the given theta."""
    node = make_node(theta=theta, **kw)
    # Skip the cold-start blindness gate -- M3 tests want the raw state.
    return node._initial_state_impl()
