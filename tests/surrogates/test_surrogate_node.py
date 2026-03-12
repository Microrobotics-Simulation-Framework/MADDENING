"""Tests for SurrogateNode -- direct/derivative modes, integrators, jit/scan/grad/vmap."""

import pytest
import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.surrogates.architecture import SurrogateArchitecture
from maddening.surrogates.node import SurrogateNode, euler_integrator, rk4_integrator


# ------------------------------------------------------------------
# Minimal test architecture (no equinox dependency)
# ------------------------------------------------------------------

class IdentityDirect(SurrogateArchitecture):
    """Direct-mode architecture: returns state unchanged (identity)."""
    mode = "direct"

    def init_params(self, rng_key, state_spec, boundary_spec):
        return {}

    def forward(self, params, state, boundary_inputs, dt):
        return {k: v for k, v in state.items()}


class LinearDerivative(SurrogateArchitecture):
    """Derivative-mode: d(state)/dt = -state (exponential decay)."""
    mode = "derivative"

    def init_params(self, rng_key, state_spec, boundary_spec):
        return {}

    def forward(self, params, state, boundary_inputs, dt):
        return {k: -v for k, v in state.items()}


class ScaledDirect(SurrogateArchitecture):
    """Direct-mode with weights: new_state = state * weights["scale"]."""
    mode = "direct"

    def init_params(self, rng_key, state_spec, boundary_spec):
        return {"scale": jnp.array(0.5)}

    def forward(self, params, state, boundary_inputs, dt):
        return {k: v * params["scale"] for k, v in state.items()}


class GravityDerivative(SurrogateArchitecture):
    """Derivative-mode: mimics gravity (dv/dt = -9.81, dp/dt = v)."""
    mode = "derivative"

    def init_params(self, rng_key, state_spec, boundary_spec):
        return {}

    def forward(self, params, state, boundary_inputs, dt):
        return {
            "position": state["velocity"],
            "velocity": jnp.array(-9.81, dtype=jnp.float32),
        }


# ------------------------------------------------------------------
# Construction and basic update
# ------------------------------------------------------------------

class TestSurrogateNodeBasic:
    def test_identity_direct(self):
        arch = IdentityDirect()
        node = SurrogateNode(
            name="test", timestep=0.01, architecture=arch,
            weights={}, state_spec={"x": ()}, boundary_spec={},
            initial_values={"x": 1.0},
        )
        state = node.initial_state()
        assert float(state["x"]) == 1.0
        new_state = node.update(state, {}, 0.01)
        assert float(new_state["x"]) == pytest.approx(1.0)

    def test_scaled_direct_with_weights(self):
        arch = ScaledDirect()
        weights = {"scale": jnp.array(2.0)}
        node = SurrogateNode(
            name="test", timestep=0.01, architecture=arch,
            weights=weights, state_spec={"x": ()}, boundary_spec={},
            initial_values={"x": 3.0},
        )
        state = node.initial_state()
        new_state = node.update(state, {}, 0.01)
        assert float(new_state["x"]) == pytest.approx(6.0)

    def test_derivative_mode_euler(self):
        arch = LinearDerivative()
        node = SurrogateNode(
            name="test", timestep=0.1, architecture=arch,
            weights={}, state_spec={"x": ()}, boundary_spec={},
            initial_values={"x": 1.0},
            integrator=euler_integrator,
        )
        state = node.initial_state()
        new_state = node.update(state, {}, 0.1)
        # x' = -x, Euler: x_new = x + dt * (-x) = 1 + 0.1*(-1) = 0.9
        assert float(new_state["x"]) == pytest.approx(0.9)

    def test_derivative_mode_rk4(self):
        arch = LinearDerivative()
        node = SurrogateNode(
            name="test", timestep=0.1, architecture=arch,
            weights={}, state_spec={"x": ()}, boundary_spec={},
            initial_values={"x": 1.0},
            integrator=rk4_integrator,
        )
        state = node.initial_state()
        new_state = node.update(state, {}, 0.1)
        # Exact: exp(-0.1) ≈ 0.904837
        # RK4 should be very close
        import math
        assert float(new_state["x"]) == pytest.approx(math.exp(-0.1), rel=1e-4)

    def test_default_integrator_is_euler(self):
        arch = LinearDerivative()
        node = SurrogateNode(
            name="test", timestep=0.1, architecture=arch,
            weights={}, state_spec={"x": ()}, boundary_spec={},
            initial_values={"x": 1.0},
        )
        state = node.initial_state()
        new_state = node.update(state, {}, 0.1)
        assert float(new_state["x"]) == pytest.approx(0.9)

    def test_multi_field_state(self):
        arch = GravityDerivative()
        node = SurrogateNode(
            name="ball", timestep=0.01, architecture=arch,
            weights={}, state_spec={"position": (), "velocity": ()},
            boundary_spec={},
            initial_values={"position": 10.0, "velocity": 0.0},
            integrator=euler_integrator,
        )
        state = node.initial_state()
        new = node.update(state, {}, 0.01)
        # dp/dt = v = 0, dv/dt = -9.81
        # Euler: p = 10 + 0.01*0 = 10, v = 0 + 0.01*(-9.81) = -0.0981
        assert float(new["position"]) == pytest.approx(10.0)
        assert float(new["velocity"]) == pytest.approx(-0.0981)

    def test_state_fields(self):
        arch = IdentityDirect()
        node = SurrogateNode(
            name="test", timestep=0.01, architecture=arch,
            weights={}, state_spec={"a": (), "b": ()}, boundary_spec={},
            initial_values={"a": 0.0, "b": 1.0},
        )
        fields = node.state_fields()
        assert set(fields) == {"a", "b"}

    def test_to_dict(self):
        arch = IdentityDirect()
        node = SurrogateNode(
            name="test", timestep=0.01, architecture=arch,
            weights={}, state_spec={"x": ()}, boundary_spec={},
            initial_values={"x": 0.0},
        )
        d = node.to_dict()
        assert d["type"] == "SurrogateNode"
        assert d["architecture"] == "IdentityDirect"
        assert d["mode"] == "direct"


# ------------------------------------------------------------------
# SurrogateNode in GraphManager
# ------------------------------------------------------------------

class TestSurrogateInGraph:
    def test_add_to_graph_and_step(self):
        gm = GraphManager()
        arch = IdentityDirect()
        node = SurrogateNode(
            name="s", timestep=0.01, architecture=arch,
            weights={}, state_spec={"x": ()}, boundary_spec={},
            initial_values={"x": 5.0},
        )
        gm.add_node(node)
        gm.compile()
        state = gm.step()
        assert float(state["s"]["x"]) == pytest.approx(5.0)

    def test_run_scan(self):
        gm = GraphManager()
        arch = LinearDerivative()
        node = SurrogateNode(
            name="decay", timestep=0.01, architecture=arch,
            weights={}, state_spec={"x": ()}, boundary_spec={},
            initial_values={"x": 1.0},
            integrator=euler_integrator,
        )
        gm.add_node(node)
        gm.compile()
        final = gm.run_scan(100)
        # After 100 steps of Euler with dt=0.01 for x'=-x:
        # x = (1 - 0.01)^100 ≈ 0.366
        assert float(final["decay"]["x"]) == pytest.approx(0.99 ** 100, rel=1e-3)


# ------------------------------------------------------------------
# JIT / scan / grad / vmap compatibility
# ------------------------------------------------------------------

class TestJAXCompatibility:
    def _make_node(self, arch=None, integrator=None):
        if arch is None:
            arch = ScaledDirect()
        return SurrogateNode(
            name="s", timestep=0.01, architecture=arch,
            weights=arch.init_params(jax.random.PRNGKey(0), {"x": ()}, {}),
            state_spec={"x": ()}, boundary_spec={},
            initial_values={"x": 1.0},
            integrator=integrator,
        )

    def test_jit_update(self):
        node = self._make_node()
        state = node.initial_state()
        jitted = jax.jit(node.update, static_argnames=())
        result = jitted(state, {}, 0.01)
        assert jnp.isfinite(result["x"])

    def test_grad_through_update(self):
        arch = ScaledDirect()
        weights = {"scale": jnp.array(2.0)}
        node = SurrogateNode(
            name="s", timestep=0.01, architecture=arch,
            weights=weights, state_spec={"x": ()}, boundary_spec={},
            initial_values={"x": 1.0},
        )

        def loss(x0):
            state = {"x": x0}
            new_state = node.update(state, {}, 0.01)
            return new_state["x"] ** 2

        grad_fn = jax.grad(loss)
        g = grad_fn(jnp.array(3.0))
        # new_x = x0 * scale = x0 * 2, loss = (2*x0)^2 = 4*x0^2
        # d(loss)/d(x0) = 8*x0 = 24
        assert float(g) == pytest.approx(24.0)

    def test_vmap_update(self):
        arch = IdentityDirect()
        node = SurrogateNode(
            name="s", timestep=0.01, architecture=arch,
            weights={}, state_spec={"x": ()}, boundary_spec={},
            initial_values={"x": 0.0},
        )
        batch_states = {"x": jnp.array([1.0, 2.0, 3.0])}
        vmapped = jax.vmap(lambda s: node.update(s, {}, 0.01))
        result = vmapped(batch_states)
        assert jnp.allclose(result["x"], jnp.array([1.0, 2.0, 3.0]))

    def test_scan_with_surrogate(self):
        arch = LinearDerivative()
        node = SurrogateNode(
            name="s", timestep=0.01, architecture=arch,
            weights={}, state_spec={"x": ()}, boundary_spec={},
            initial_values={"x": 1.0},
            integrator=euler_integrator,
        )
        state = node.initial_state()

        def step(s, _):
            ns = node.update(s, {}, 0.01)
            return ns, ns["x"]

        final, trace = jax.lax.scan(step, state, None, length=10)
        assert trace.shape == (10,)
        assert float(final["x"]) == pytest.approx(0.99 ** 10, rel=1e-5)

    def test_grad_through_graph_with_surrogate(self):
        """Grad flows through a GraphManager containing a SurrogateNode."""
        arch = ScaledDirect()
        weights = {"scale": jnp.array(0.9)}
        node = SurrogateNode(
            name="s", timestep=0.01, architecture=arch,
            weights=weights, state_spec={"x": ()}, boundary_spec={},
            initial_values={"x": 1.0},
        )
        gm = GraphManager()
        gm.add_node(node)
        gm.compile()

        step_fn = gm._build_step_fn()

        def loss(init_state):
            final = step_fn(init_state, {})
            return final["s"]["x"] ** 2

        init = gm._state
        g = jax.grad(loss)(init)
        assert jnp.isfinite(g["s"]["x"])
