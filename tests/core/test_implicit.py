"""Tests for Phase 5: Implicit node support.

Verifies that:
1. implicit_residual() is implemented for SpringDamperNode and HeatNode
2. implicit_residual() raises NotImplementedError for base class
3. implicit_euler_step() solves the backward Euler equation
4. Implicit integration is stable for stiff problems
5. Implicit solver is JIT-compatible and differentiable
"""

import jax
import jax.numpy as jnp
import pytest

from maddening.core.implicit import implicit_euler_step
from maddening.core.integrators import euler_step, rk4_step
from maddening.core.node import SimulationNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode


class TestImplicitResidual:
    """Test implicit_residual() on nodes."""

    def test_base_class_raises(self):
        """SimulationNode.implicit_residual() raises NotImplementedError."""
        table = TableNode(name="table", timestep=0.01)
        with pytest.raises(NotImplementedError):
            table.implicit_residual({}, {}, {}, 0.01)

    def test_spring_residual(self):
        """SpringDamperNode should implement implicit_residual."""
        spring = SpringDamperNode(
            name="spring", timestep=0.01,
            stiffness=100.0, damping=1.0, mass=1.0,
            rest_length=0.0,
        )
        state_old = {"position": jnp.array(1.0), "velocity": jnp.array(0.0)}
        state_new = {"position": jnp.array(1.0), "velocity": jnp.array(0.0)}
        bi = {"anchor_position": jnp.array(0.0)}

        res = spring.implicit_residual(state_new, state_old, bi, 0.01)
        assert "position" in res
        assert "velocity" in res
        # At the exact solution, residual should be zero
        # state_new = state_old means no change, residual = -dt*f(x)
        assert jnp.all(jnp.isfinite(res["position"]))

    def test_heat_residual(self):
        """HeatNode should implement implicit_residual."""
        n = 5
        heat = HeatNode(
            name="rod", timestep=0.001, n_cells=n,
            thermal_diffusivity=0.01,
        )
        state_old = {"temperature": jnp.ones(n) * 50.0}
        state_new = {"temperature": jnp.ones(n) * 50.0}
        bi = {
            "left_temperature": jnp.array(100.0),
            "right_temperature": jnp.array(0.0),
        }

        res = heat.implicit_residual(state_new, state_old, bi, 0.001)
        assert "temperature" in res
        assert res["temperature"].shape == (n,)

    def test_residual_zero_at_solution(self):
        """Residual should be zero when state_new is the exact solution."""
        spring = SpringDamperNode(
            name="spring", timestep=0.01,
            stiffness=100.0, damping=0.0, mass=1.0,
            rest_length=0.0,
        )
        state_old = {"position": jnp.array(0.0), "velocity": jnp.array(0.0)}
        bi = {"anchor_position": jnp.array(0.0)}
        dt = 0.01

        # For zero initial state with no force, the solution is zero
        state_exact = {"position": jnp.array(0.0), "velocity": jnp.array(0.0)}
        res = spring.implicit_residual(state_exact, state_old, bi, dt)
        assert jnp.allclose(res["position"], 0.0, atol=1e-10)
        assert jnp.allclose(res["velocity"], 0.0, atol=1e-10)


class TestImplicitEulerStep:
    """Tests for the implicit Euler solver."""

    def test_spring_implicit(self):
        """Implicit Euler should solve the spring equation."""
        spring = SpringDamperNode(
            name="spring", timestep=0.01,
            stiffness=100.0, damping=1.0, mass=1.0,
            rest_length=0.0,
        )
        state = {"position": jnp.array(1.0), "velocity": jnp.array(0.0)}
        bi = {"anchor_position": jnp.array(0.0)}
        dt = 0.01

        new_state = implicit_euler_step(
            spring.implicit_residual, state, bi, dt, n_newton=5,
        )

        assert jnp.isfinite(new_state["position"])
        assert jnp.isfinite(new_state["velocity"])
        # Spring should start moving (position should change)
        assert float(new_state["position"]) != pytest.approx(1.0, abs=1e-3)

    def test_heat_implicit(self):
        """Implicit Euler should solve the heat equation."""
        n = 5
        heat = HeatNode(
            name="rod", timestep=0.001, n_cells=n,
            thermal_diffusivity=0.01,
        )
        state = {"temperature": jnp.zeros(n)}
        bi = {
            "left_temperature": jnp.array(100.0),
            "right_temperature": jnp.array(0.0),
        }
        dt = 0.001

        new_state = implicit_euler_step(
            heat.implicit_residual, state, bi, dt, n_newton=5,
        )

        assert new_state["temperature"].shape == (n,)
        assert jnp.all(jnp.isfinite(new_state["temperature"]))
        # Temperature near left should have increased
        assert float(new_state["temperature"][0]) > 0.0

    def test_stiff_spring_stability(self):
        """Implicit Euler should be stable for very stiff springs
        where explicit Euler diverges."""
        k_stiff = 10000.0
        spring = SpringDamperNode(
            name="spring", timestep=0.01,
            stiffness=k_stiff, damping=1.0, mass=1.0,
            rest_length=0.0,
        )
        state = {"position": jnp.array(1.0), "velocity": jnp.array(0.0)}
        bi = {"anchor_position": jnp.array(0.0)}
        dt = 0.1  # Large dt for stiff system

        # Explicit Euler with this dt should diverge
        state_explicit = state
        for _ in range(20):
            state_explicit = euler_step(
                spring.derivatives, state_explicit, bi, dt
            )

        # Implicit Euler should stay stable
        state_implicit = state
        for _ in range(20):
            state_implicit = implicit_euler_step(
                spring.implicit_residual, state_implicit, bi, dt,
                n_newton=10,
            )

        # Explicit should have diverged (huge values or NaN)
        explicit_energy = (
            float(state_explicit["position"]) ** 2
            + float(state_explicit["velocity"]) ** 2
        )

        # Implicit should be bounded
        implicit_energy = (
            float(state_implicit["position"]) ** 2
            + float(state_implicit["velocity"]) ** 2
        )

        assert jnp.isfinite(implicit_energy)
        assert implicit_energy < 10.0, (
            f"Implicit energy {implicit_energy} should be bounded"
        )

    def test_implicit_jit(self):
        """Implicit Euler should JIT compile."""
        spring = SpringDamperNode(
            name="spring", timestep=0.01,
            stiffness=100.0, damping=1.0, mass=1.0,
            rest_length=0.0,
        )
        state = spring.initial_state()
        bi = {"anchor_position": jnp.array(0.0)}

        @jax.jit
        def step(s):
            return implicit_euler_step(
                spring.implicit_residual, s, bi, 0.01, n_newton=3,
            )

        new_state = step(state)
        assert jnp.isfinite(new_state["position"])

    def test_implicit_grad(self):
        """Implicit Euler should be differentiable."""
        spring = SpringDamperNode(
            name="spring", timestep=0.01,
            stiffness=100.0, damping=1.0, mass=1.0,
            rest_length=0.0,
        )
        bi = {"anchor_position": jnp.array(0.0)}

        def loss_fn(x0):
            state = {"position": x0, "velocity": jnp.array(0.0)}
            for _ in range(5):
                state = implicit_euler_step(
                    spring.implicit_residual, state, bi, 0.01,
                    n_newton=3,
                )
            return state["position"]

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(1.0))
        assert jnp.isfinite(g)

    def test_newton_convergence(self):
        """More Newton iterations should give a more accurate solution."""
        spring = SpringDamperNode(
            name="spring", timestep=0.01,
            stiffness=100.0, damping=1.0, mass=1.0,
            rest_length=0.0,
        )
        state = {"position": jnp.array(1.0), "velocity": jnp.array(0.0)}
        bi = {"anchor_position": jnp.array(0.0)}
        dt = 0.01

        # Reference with many Newton iterations
        ref = implicit_euler_step(
            spring.implicit_residual, state, bi, dt, n_newton=20,
        )

        # Fewer iterations
        few = implicit_euler_step(
            spring.implicit_residual, state, bi, dt, n_newton=2,
        )

        # More iterations should be closer to the reference
        more = implicit_euler_step(
            spring.implicit_residual, state, bi, dt, n_newton=5,
        )

        err_few = abs(float(few["position"]) - float(ref["position"]))
        err_more = abs(float(more["position"]) - float(ref["position"]))

        assert err_more <= err_few + 1e-10, (
            f"More Newton iters error {err_more} should be <= "
            f"fewer iters error {err_few}"
        )
