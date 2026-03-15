"""Tests for Phase 3: Node derivatives() and pluggable integrators.

Verifies that:
1. derivatives() is implemented for BallNode, SpringDamperNode, HeatNode
2. Euler, Heun, and RK4 integrators work correctly
3. RK4 is more accurate than Euler for the same step size
4. integrate_node() convenience function works
5. All integrators are JIT-compatible and differentiable
6. derivatives() raises NotImplementedError for base class
"""

import jax
import jax.numpy as jnp
import pytest

from maddening.core.integrators import (
    euler_step,
    heun_step,
    integrate_node,
    rk4_step,
)
from maddening.core.node import SimulationNode
from maddening.nodes.ball import BallNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode


# ==================================================================
# Test derivatives() on nodes
# ==================================================================

class TestNodeDerivatives:
    """Test that nodes implement derivatives() correctly."""

    def test_base_class_raises(self):
        """SimulationNode.derivatives() raises NotImplementedError."""
        table = TableNode(name="table", timestep=0.01)
        with pytest.raises(NotImplementedError):
            table.derivatives({}, {})

    def test_ball_derivatives(self):
        """BallNode derivatives: dx/dt = v, dv/dt = g."""
        ball = BallNode(name="ball", timestep=0.01, gravity=-10.0)
        state = {"position": jnp.array(5.0), "velocity": jnp.array(2.0)}
        derivs = ball.derivatives(state, {})

        assert float(derivs["position"]) == pytest.approx(2.0)  # dx/dt = v
        assert float(derivs["velocity"]) == pytest.approx(-10.0)  # dv/dt = g

    def test_spring_derivatives(self):
        """SpringDamperNode derivatives: dx/dt = v, dv/dt = F/m."""
        spring = SpringDamperNode(
            name="spring", timestep=0.01,
            stiffness=100.0, damping=2.0, mass=1.0,
            rest_length=1.0,
        )
        state = {"position": jnp.array(2.0), "velocity": jnp.array(0.0)}
        bi = {"anchor_position": jnp.array(0.0)}
        derivs = spring.derivatives(state, bi)

        # F = -100*(2-0-1) - 2*0 = -100
        assert float(derivs["position"]) == pytest.approx(0.0)  # v=0
        assert float(derivs["velocity"]) == pytest.approx(-100.0)  # F/m

    def test_heat_derivatives(self):
        """HeatNode derivatives: dT/dt = alpha * laplacian + source."""
        n = 5
        heat = HeatNode(
            name="rod", timestep=0.001, n_cells=n,
            thermal_diffusivity=0.1, length=1.0,
        )
        # Linear profile: T = [0, 25, 50, 75, 100]
        state = {"temperature": jnp.linspace(0.0, 100.0, n)}
        bi = {
            "left_temperature": jnp.array(0.0),
            "right_temperature": jnp.array(100.0),
        }
        derivs = heat.derivatives(state, bi)

        assert "temperature" in derivs
        assert derivs["temperature"].shape == (n,)
        # For a linear profile, the interior Laplacian should be ~0
        # (boundary cells may differ due to ghost cell arrangement)
        assert jnp.allclose(derivs["temperature"][1:-1], 0.0, atol=1.0)
        # All derivatives should be finite
        assert jnp.all(jnp.isfinite(derivs["temperature"]))

    def test_derivatives_jit(self):
        """derivatives() should be JIT-compilable."""
        ball = BallNode(name="ball", timestep=0.01)
        state = ball.initial_state()

        @jax.jit
        def get_derivs(s):
            return ball.derivatives(s, {})

        derivs = get_derivs(state)
        assert jnp.isfinite(derivs["position"])
        assert jnp.isfinite(derivs["velocity"])


# ==================================================================
# Test integrators
# ==================================================================

class TestEulerIntegrator:
    """Tests for the Euler integrator."""

    def test_euler_ball(self):
        """Euler should approximate ballistic motion."""
        ball = BallNode(name="ball", timestep=0.01, gravity=-10.0)
        state = {"position": jnp.array(0.0), "velocity": jnp.array(10.0)}

        new_state = euler_step(ball.derivatives, state, {}, 0.01)
        # x += v * dt = 0 + 10*0.01 = 0.1
        assert float(new_state["position"]) == pytest.approx(0.1, abs=1e-5)
        # v += g * dt = 10 + (-10)*0.01 = 9.9
        assert float(new_state["velocity"]) == pytest.approx(9.9, abs=1e-5)


class TestRK4Integrator:
    """Tests for the RK4 integrator."""

    def test_rk4_ball(self):
        """RK4 should be more accurate than Euler for free fall."""
        ball = BallNode(name="ball", timestep=0.1, gravity=-10.0)
        state = {"position": jnp.array(0.0), "velocity": jnp.array(0.0)}
        dt = 1.0

        # Exact solution: x(1) = 0.5*g*t^2 = -5.0, v(1) = g*t = -10.0
        state_euler = euler_step(ball.derivatives, state, {}, dt)
        state_rk4 = rk4_step(ball.derivatives, state, {}, dt)

        # For constant acceleration, RK4 should be exact
        assert float(state_rk4["position"]) == pytest.approx(-5.0, abs=1e-4)
        assert float(state_rk4["velocity"]) == pytest.approx(-10.0, abs=1e-4)

        # Euler is less accurate (x = v*dt = 0, since v starts at 0)
        assert abs(float(state_euler["position"]) - (-5.0)) > 0.1

    def test_rk4_spring_accuracy(self):
        """RK4 should be significantly more accurate for spring oscillation."""
        spring = SpringDamperNode(
            name="spring", timestep=0.01,
            stiffness=100.0, damping=0.0, mass=1.0,
            rest_length=0.0, initial_position=1.0,
        )
        state0 = spring.initial_state()
        bi = {"anchor_position": jnp.array(0.0)}

        # omega = sqrt(k/m) = 10, period = 2*pi/10 ~ 0.628
        # Run for one full period
        n_steps = 100
        dt = 0.00628  # period / n_steps

        # Euler
        state_euler = state0
        for _ in range(n_steps):
            state_euler = euler_step(
                spring.derivatives, state_euler, bi, dt
            )

        # RK4
        state_rk4 = state0
        for _ in range(n_steps):
            state_rk4 = rk4_step(
                spring.derivatives, state_rk4, bi, dt
            )

        # After one period, both should return near (1, 0).
        # RK4 should be much closer.
        err_euler_pos = abs(float(state_euler["position"]) - 1.0)
        err_rk4_pos = abs(float(state_rk4["position"]) - 1.0)

        assert err_rk4_pos < err_euler_pos, (
            f"RK4 error {err_rk4_pos} should be less than Euler {err_euler_pos}"
        )
        # RK4 should be very close
        assert err_rk4_pos < 0.01, f"RK4 error {err_rk4_pos} should be < 0.01"

    def test_rk4_heat(self):
        """RK4 should work with HeatNode."""
        n = 10
        heat = HeatNode(
            name="rod", timestep=0.001, n_cells=n,
            thermal_diffusivity=0.01, initial_temperature=0.0,
        )
        state = heat.initial_state()
        bi = {
            "left_temperature": jnp.array(100.0),
            "right_temperature": jnp.array(0.0),
        }

        new_state = rk4_step(heat.derivatives, state, bi, 0.001)
        assert new_state["temperature"].shape == (n,)
        assert jnp.all(jnp.isfinite(new_state["temperature"]))


class TestHeunIntegrator:
    """Tests for the Heun integrator."""

    def test_heun_accuracy(self):
        """Heun should be between Euler and RK4 in accuracy."""
        ball = BallNode(name="ball", timestep=0.1, gravity=-10.0)
        state = {"position": jnp.array(0.0), "velocity": jnp.array(0.0)}

        state_euler = euler_step(ball.derivatives, state, {}, 1.0)
        state_heun = heun_step(ball.derivatives, state, {}, 1.0)
        state_rk4 = rk4_step(ball.derivatives, state, {}, 1.0)

        # Exact: -5.0
        err_euler = abs(float(state_euler["position"]) - (-5.0))
        err_heun = abs(float(state_heun["position"]) - (-5.0))
        err_rk4 = abs(float(state_rk4["position"]) - (-5.0))

        # For constant acceleration, Heun and RK4 are both exact
        assert err_heun < 0.01
        assert err_rk4 < 0.01


class TestIntegrateNode:
    """Tests for the integrate_node convenience function."""

    def test_euler_method(self):
        """integrate_node with 'euler' method."""
        ball = BallNode(name="ball", timestep=0.01)
        state = ball.initial_state()
        new_state = integrate_node(ball, state, {}, 0.01, method="euler")
        assert "position" in new_state
        assert "velocity" in new_state

    def test_rk4_method(self):
        """integrate_node with 'rk4' method."""
        ball = BallNode(name="ball", timestep=0.01)
        state = ball.initial_state()
        new_state = integrate_node(ball, state, {}, 0.01, method="rk4")
        assert "position" in new_state

    def test_heun_method(self):
        """integrate_node with 'heun' method."""
        ball = BallNode(name="ball", timestep=0.01)
        state = ball.initial_state()
        new_state = integrate_node(ball, state, {}, 0.01, method="heun")
        assert "position" in new_state

    def test_unknown_method_raises(self):
        """Should raise for unknown method."""
        ball = BallNode(name="ball", timestep=0.01)
        state = ball.initial_state()
        with pytest.raises(ValueError, match="Unknown"):
            integrate_node(ball, state, {}, 0.01, method="adams")


class TestIntegratorJAXCompat:
    """Test JAX compatibility of integrators."""

    def test_rk4_jit(self):
        """RK4 should JIT compile."""
        ball = BallNode(name="ball", timestep=0.01, gravity=-10.0)
        state = ball.initial_state()

        @jax.jit
        def step(s):
            return rk4_step(ball.derivatives, s, {}, 0.01)

        new_state = step(state)
        assert jnp.isfinite(new_state["position"])

    def test_rk4_grad(self):
        """RK4 should be differentiable."""
        spring = SpringDamperNode(
            name="spring", timestep=0.01,
            stiffness=100.0, damping=1.0, mass=1.0,
            rest_length=1.0,
        )

        def loss_fn(pos0):
            state = {"position": pos0, "velocity": jnp.array(0.0)}
            bi = {"anchor_position": jnp.array(0.0)}
            for _ in range(10):
                state = rk4_step(spring.derivatives, state, bi, 0.01)
            return state["position"]

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(2.0))
        assert jnp.isfinite(g)

    def test_rk4_scan(self):
        """RK4 should work inside lax.scan."""
        ball = BallNode(name="ball", timestep=0.01, gravity=-10.0)
        state = ball.initial_state()

        def scan_body(carry, _):
            s = rk4_step(ball.derivatives, carry, {}, 0.01)
            return s, s["position"]

        final, trajectory = jax.lax.scan(
            scan_body, state, jnp.arange(100)
        )
        assert jnp.isfinite(final["position"])
        assert trajectory.shape == (100,)

    def test_rk4_vmap(self):
        """RK4 should work with vmap."""
        ball = BallNode(name="ball", timestep=0.01, gravity=-10.0)

        def step(v0):
            state = {"position": jnp.array(0.0), "velocity": v0}
            return rk4_step(ball.derivatives, state, {}, 0.1)

        batched = jax.vmap(step)
        v0s = jnp.array([1.0, 2.0, 3.0, 4.0])
        results = batched(v0s)
        assert results["position"].shape == (4,)
        assert jnp.all(jnp.isfinite(results["position"]))


class TestRK4vsEulerAccuracy:
    """Quantitative accuracy comparison between RK4 and Euler."""

    def test_spring_energy_conservation(self):
        """RK4 should conserve energy much better than Euler for
        an undamped spring over many steps."""
        spring = SpringDamperNode(
            name="spring", timestep=0.01,
            stiffness=100.0, damping=0.0, mass=1.0,
            rest_length=0.0, initial_position=1.0,
            initial_velocity=0.0,
        )
        state0 = spring.initial_state()
        bi = {"anchor_position": jnp.array(0.0)}

        # Initial energy: 0.5*k*x^2 + 0.5*m*v^2 = 50.0
        k = 100.0
        def energy(s):
            return 0.5 * k * float(s["position"])**2 + 0.5 * float(s["velocity"])**2

        E0 = energy(state0)

        # Run 1000 steps
        state_euler = state0
        state_rk4 = state0
        dt = 0.01
        for _ in range(1000):
            state_euler = euler_step(spring.derivatives, state_euler, bi, dt)
            state_rk4 = rk4_step(spring.derivatives, state_rk4, bi, dt)

        E_euler = energy(state_euler)
        E_rk4 = energy(state_rk4)

        # RK4 should conserve energy much better
        err_euler = abs(E_euler - E0) / E0
        err_rk4 = abs(E_rk4 - E0) / E0

        assert err_rk4 < err_euler, (
            f"RK4 energy error {err_rk4} should be < Euler {err_euler}"
        )
        assert err_rk4 < 0.01, f"RK4 energy error {err_rk4} should be < 1%"
