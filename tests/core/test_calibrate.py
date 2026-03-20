"""Tests for Phase 4: Differentiable calibration tooling.

Verifies that:
1. calibrate() can recover simple scalar parameters
2. calibrate() works with multi-parameter optimisation
3. calibrate() reports convergence correctly
4. calibrate() can recover physics parameters from simulation data
5. Custom loss functions work
"""

import jax
import jax.numpy as jnp
import pytest

from maddening.core.simulation.calibration import CalibrateResult, calibrate
from maddening.core.simulation.integrators import rk4_step
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode


class TestCalibrateBasic:
    """Basic calibrate() tests."""

    def test_scalar_recovery(self):
        """Should recover a simple scalar parameter."""
        # f(a) = a^2, target = 9.0, so a should converge to 3.0
        def forward(params):
            return params["a"] ** 2

        result = calibrate(
            forward_fn=forward,
            initial_params={"a": jnp.array(1.0)},
            reference_trajectory=jnp.array(9.0),
            n_iters=500,
            learning_rate=0.005,  # small LR for nonlinear problem
        )
        assert isinstance(result, CalibrateResult)
        assert float(result.params["a"]) == pytest.approx(3.0, abs=0.2)
        assert result.loss_history[-1] < result.loss_history[0]

    def test_returns_calibrate_result(self):
        """Should return a CalibrateResult."""
        def forward(params):
            return params["x"]

        result = calibrate(
            forward_fn=forward,
            initial_params={"x": jnp.array(0.0)},
            reference_trajectory=jnp.array(5.0),
            n_iters=10,
        )
        assert isinstance(result, CalibrateResult)
        assert isinstance(result.loss_history, list)
        assert len(result.loss_history) <= 10

    def test_multi_parameter(self):
        """Should recover multiple parameters."""
        # f(a, b) = a*x + b, target at x=2: 7.0 (a=3, b=1)
        def forward(params):
            return params["a"] * 2.0 + params["b"]

        result = calibrate(
            forward_fn=forward,
            initial_params={
                "a": jnp.array(1.0),
                "b": jnp.array(0.0),
            },
            reference_trajectory=jnp.array(7.0),
            n_iters=1000,
            learning_rate=0.05,
        )
        # a*2 + b = 7; many solutions. Check the loss is low.
        predicted = result.params["a"] * 2.0 + result.params["b"]
        assert float(predicted) == pytest.approx(7.0, abs=0.1)

    def test_convergence_flag(self):
        """Should report convergence when loss is below tolerance."""
        def forward(params):
            return params["x"]

        result = calibrate(
            forward_fn=forward,
            initial_params={"x": jnp.array(4.9)},
            reference_trajectory=jnp.array(5.0),
            n_iters=100,
            learning_rate=0.1,
            tolerance=1e-4,
        )
        assert result.converged

    def test_custom_loss(self):
        """Should work with a custom loss function."""
        def forward(params):
            return params["x"]

        def my_loss(pred, ref):
            return jnp.abs(pred - ref)

        result = calibrate(
            forward_fn=forward,
            initial_params={"x": jnp.array(0.0)},
            reference_trajectory=jnp.array(3.0),
            loss_fn=my_loss,
            n_iters=200,
            learning_rate=0.1,
        )
        assert float(result.params["x"]) == pytest.approx(3.0, abs=0.5)


class TestCalibratePhysics:
    """Calibrate physics parameters from simulation data."""

    def test_recover_gravity(self):
        """Recover gravity from a free-fall trajectory."""
        true_g = -9.81

        # Generate reference trajectory using true gravity
        def run_sim(params):
            g = params["gravity"]
            state = {"position": jnp.array(0.0), "velocity": jnp.array(0.0)}

            def step(s, _):
                # Manual Euler integration with traced gravity
                v_new = s["velocity"] + g * 0.01
                p_new = s["position"] + v_new * 0.01
                new_s = {"position": p_new, "velocity": v_new}
                return new_s, p_new

            final, trajectory = jax.lax.scan(step, state, jnp.arange(100))
            return trajectory

        ref = run_sim({"gravity": jnp.array(true_g)})

        result = calibrate(
            forward_fn=run_sim,
            initial_params={"gravity": jnp.array(-5.0)},
            reference_trajectory=ref,
            n_iters=300,
            learning_rate=0.1,
        )

        recovered_g = float(result.params["gravity"])
        assert recovered_g == pytest.approx(true_g, abs=0.5), (
            f"Recovered gravity {recovered_g}, expected {true_g}"
        )

    def test_recover_spring_stiffness(self):
        """Recover spring stiffness from oscillation data."""
        true_k = 50.0

        def run_sim(params):
            k = params["stiffness"]
            state = {"position": jnp.array(1.0), "velocity": jnp.array(0.0)}

            def step(s, _):
                force = -k * s["position"] - 1.0 * s["velocity"]
                v_new = s["velocity"] + force * 0.01
                p_new = s["position"] + v_new * 0.01
                new_s = {"position": p_new, "velocity": v_new}
                return new_s, p_new

            final, trajectory = jax.lax.scan(step, state, jnp.arange(200))
            return trajectory

        ref = run_sim({"stiffness": jnp.array(true_k)})

        result = calibrate(
            forward_fn=run_sim,
            initial_params={"stiffness": jnp.array(40.0)},
            reference_trajectory=ref,
            n_iters=1000,
            learning_rate=0.1,
        )

        recovered_k = float(result.params["stiffness"])
        assert recovered_k == pytest.approx(true_k, abs=10.0), (
            f"Recovered stiffness {recovered_k}, expected {true_k}"
        )

    def test_recover_thermal_diffusivity(self):
        """Recover thermal diffusivity from heat evolution."""
        true_alpha = 0.05

        def run_sim(params):
            alpha = params["alpha"]
            n = 10
            dx = 1.0 / n
            T = jnp.zeros(n)

            def step(T, _):
                T_padded = jnp.concatenate([
                    jnp.array([100.0]), T, jnp.array([0.0])
                ])
                laplacian = T_padded[2:] - 2.0 * T_padded[1:-1] + T_padded[:-2]
                coeff = alpha * 0.0001 / (dx * dx)
                T_new = T + coeff * laplacian
                return T_new, T_new

            final, history = jax.lax.scan(step, T, jnp.arange(500))
            return final

        ref = run_sim({"alpha": jnp.array(true_alpha)})

        result = calibrate(
            forward_fn=run_sim,
            initial_params={"alpha": jnp.array(0.03)},
            reference_trajectory=ref,
            n_iters=500,
            learning_rate=0.0001,  # small LR for stiff heat equation
        )

        recovered = float(result.params["alpha"])
        # Just verify it moved closer to the true value
        initial_error = abs(0.03 - true_alpha)
        final_error = abs(recovered - true_alpha)
        assert final_error < initial_error, (
            f"Calibration should improve: initial error {initial_error}, "
            f"final error {final_error}, recovered={recovered}"
        )

    def test_loss_decreases(self):
        """Loss should monotonically decrease (mostly)."""
        def forward(params):
            return params["x"] ** 2

        result = calibrate(
            forward_fn=forward,
            initial_params={"x": jnp.array(5.0)},
            reference_trajectory=jnp.array(4.0),
            n_iters=100,
            learning_rate=0.01,
        )
        # First loss should be larger than last
        assert result.loss_history[0] > result.loss_history[-1]
