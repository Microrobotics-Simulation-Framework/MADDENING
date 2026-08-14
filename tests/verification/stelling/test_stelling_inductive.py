"""Inductive step verification (stelling 0.2 — check_inductive_step).

Proves loop invariants automatically: one iteration of a body function
preserves the declared state bounds. VERIFIED means the invariant holds
for ALL iterations by induction.
"""

import jax.numpy as jnp

from stelling.inductive import check_inductive_step


SOLVER_TIMEOUT = 15000


class TestHeatPositivityInductive:
    """Explicit Euler heat equation preserves T >= 0 when CFL-safe."""

    def test_cfl_safe_preserves_positivity(self):
        """With dt = 0.4 * dx^2 / (2*alpha), positivity is preserved."""
        def heat_body(state, constants):
            T_left = state["T_left"]
            T_center = state["T_center"]
            T_right = state["T_right"]
            alpha = 0.01
            dx = 0.1
            dt = 0.4 * dx ** 2 / (2.0 * alpha)
            lap = (T_right - 2.0 * T_center + T_left) / (dx * dx)
            T_new = T_center + alpha * dt * lap
            return {"T_left": T_left, "T_center": T_new, "T_right": T_right}

        v = check_inductive_step(
            heat_body,
            state_bounds={
                "T_left": ((0.0, 1000.0), "float64"),
                "T_center": ((0.0, 1000.0), "float64"),
                "T_right": ((0.0, 1000.0), "float64"),
            },
            solver_timeout_ms=SOLVER_TIMEOUT,
        )
        assert v.status == "VERIFIED"


class TestAdaptiveDtInductive:
    """The adaptive PI controller preserves dt bounds across iterations."""

    def test_dt_stays_bounded(self):
        """clip(dt * factor, dt_min, dt_max) stays in [dt_min, dt_max]."""
        def pi_body(state, constants):
            error = state["error"]
            dt = state["dt"]
            factor = 0.9 / error
            factor = jnp.clip(factor, 0.2, 5.0)
            dt_new = jnp.clip(dt * factor, 1e-8, 0.1)
            return {"error": error, "dt": dt_new}

        v = check_inductive_step(
            pi_body,
            state_bounds={
                "error": ((0.1, 10.0), "float64"),
                "dt": ((1e-8, 0.1), "float64"),
            },
            solver_timeout_ms=SOLVER_TIMEOUT,
        )
        assert v.status == "VERIFIED"

    def test_dt_stays_bounded_with_sqrt(self):
        """The full PI formula with sqrt preserves dt bounds."""
        def pi_sqrt_body(state, constants):
            error = state["error"]
            dt = state["dt"]
            factor = 0.9 * jnp.power(1.0 / error, 0.5)
            factor = jnp.clip(factor, 0.2, 5.0)
            dt_new = jnp.clip(dt * factor, 1e-8, 0.1)
            return {"error": error, "dt": dt_new}

        v = check_inductive_step(
            pi_sqrt_body,
            state_bounds={
                "error": ((0.01, 10.0), "float64"),
                "dt": ((1e-8, 0.1), "float64"),
            },
            solver_timeout_ms=SOLVER_TIMEOUT,
        )
        assert v.status == "VERIFIED"


class TestAitkenOmegaInductive:
    """Aitken omega bounds are preserved by the clip operation."""

    def test_omega_invariant(self):
        """clip(raw, 0.01, 2.0) always stays in [0.01, 2.0]."""
        def aitken_body(state, constants):
            omega = state["omega"]
            new_omega = jnp.clip(omega * 0.5 + 0.3, 0.01, 2.0)
            return {"omega": new_omega}

        v = check_inductive_step(
            aitken_body,
            state_bounds={"omega": ((0.01, 2.0), "float64")},
        )
        assert v.status == "VERIFIED"


class TestExponentialDecayInductive:
    """Exponential decay x_{n+1} = (1-dt)*x_n preserves bounds.
    Requires solver (dependency problem: x appears twice)."""

    def test_decay_preserves_bounds(self):
        def decay_body(state, constants):
            x = state["x"]
            return {"x": x + 0.1 * (-x)}

        v = check_inductive_step(
            decay_body,
            state_bounds={"x": ((-10.0, 10.0), "float64")},
            solver_timeout_ms=SOLVER_TIMEOUT,
        )
        assert v.status == "VERIFIED"


class TestCouplingNormInductive:
    """The coupling norm computation preserves state bounds."""

    def test_norm_preserves_bounds(self):
        """After one coupling pass, a_old := a_new stays bounded."""
        def coupling_body(state, constants):
            a_new = state["a_new"]
            a_old = state["a_old"]
            diff = jnp.abs(a_new - a_old)
            scale = 1e-8 + 1e-6 * jnp.maximum(
                jnp.abs(a_new), jnp.abs(a_old)
            )
            jnp.where(scale > 0, diff / jnp.maximum(scale, 1e-300), 0.0)
            return {"a_new": a_new, "a_old": a_new}

        v = check_inductive_step(
            coupling_body,
            state_bounds={
                "a_new": ((-1000.0, 1000.0), "float64"),
                "a_old": ((-1000.0, 1000.0), "float64"),
            },
            solver_timeout_ms=SOLVER_TIMEOUT,
        )
        assert v.status == "VERIFIED"
