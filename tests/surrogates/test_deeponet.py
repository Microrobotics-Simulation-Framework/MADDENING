"""Tests for DeepONet and S-DeepONet surrogate architectures."""

import pytest
import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.surrogates.node import SurrogateNode, euler_integrator, rk4_integrator
from maddening.surrogates.dataset import DatasetGenerator
from maddening.surrogates.training.trainer import SurrogateTrainer
from maddening.surrogates.architectures.deeponet import (
    DeepONetDirect,
    DeepONetDerivative,
    SDeepONetDirect,
    SDeepONetDerivative,
)


# ------------------------------------------------------------------
# DeepONet: init_params + forward
# ------------------------------------------------------------------

class TestDeepONetDirect:
    def test_init_and_forward(self):
        arch = DeepONetDirect(n_basis=8, branch_hidden=(8, 8))
        state_spec = {"position": (), "velocity": ()}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        params = arch.init_params(key, state_spec, boundary_spec)

        state = {"position": jnp.array(5.0), "velocity": jnp.array(0.0)}
        out = arch.forward(params, state, {}, 0.01)
        assert "position" in out
        assert "velocity" in out
        assert jnp.isfinite(out["position"])
        assert jnp.isfinite(out["velocity"])

    def test_mode_is_direct(self):
        arch = DeepONetDirect()
        assert arch.mode == "direct"

    def test_with_boundary_inputs(self):
        arch = DeepONetDirect(n_basis=8, branch_hidden=(8,))
        state_spec = {"x": ()}
        boundary_spec = {"force": ()}
        key = jax.random.PRNGKey(1)
        params = arch.init_params(key, state_spec, boundary_spec)

        state = {"x": jnp.array(1.0)}
        boundary = {"force": jnp.array(2.0)}
        out = arch.forward(params, state, boundary, 0.01)
        assert jnp.isfinite(out["x"])

    def test_in_surrogate_node(self):
        arch = DeepONetDirect(n_basis=8, branch_hidden=(8,))
        state_spec = {"x": ()}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="test", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"x": 1.0},
        )
        state = node.initial_state()
        new_state = node.update(state, {}, 0.01)
        assert jnp.isfinite(new_state["x"])

    def test_jit_compatible(self):
        arch = DeepONetDirect(n_basis=8, branch_hidden=(8,))
        state_spec = {"x": ()}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="test", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"x": 1.0},
        )
        state = node.initial_state()
        jitted = jax.jit(node.update)
        result = jitted(state, {}, 0.01)
        assert jnp.isfinite(result["x"])


class TestDeepONetDerivative:
    def test_init_and_forward(self):
        arch = DeepONetDerivative(n_basis=8, branch_hidden=(8,))
        state_spec = {"position": (), "velocity": ()}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        params = arch.init_params(key, state_spec, boundary_spec)

        state = {"position": jnp.array(5.0), "velocity": jnp.array(0.0)}
        deriv = arch.forward(params, state, {}, 0.01)
        assert "position" in deriv
        assert "velocity" in deriv
        assert jnp.isfinite(deriv["position"])
        assert jnp.isfinite(deriv["velocity"])

    def test_mode_is_derivative(self):
        arch = DeepONetDerivative()
        assert arch.mode == "derivative"

    def test_euler_integration(self):
        arch = DeepONetDerivative(n_basis=8, branch_hidden=(8,))
        state_spec = {"x": ()}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="test", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"x": 1.0},
            integrator=euler_integrator,
        )
        state = node.initial_state()
        new_state = node.update(state, {}, 0.01)
        assert jnp.isfinite(new_state["x"])

    def test_rk4_integration(self):
        arch = DeepONetDerivative(n_basis=8, branch_hidden=(8,))
        state_spec = {"x": ()}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="test", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"x": 1.0},
            integrator=rk4_integrator,
        )
        state = node.initial_state()
        new_state = node.update(state, {}, 0.01)
        assert jnp.isfinite(new_state["x"])


# ------------------------------------------------------------------
# DeepONet training convergence
# ------------------------------------------------------------------

class TestDeepONetTraining:
    def _make_ball_dataset(self, n_steps=200):
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
        gm.compile()
        return DatasetGenerator.from_graph(gm, "ball", n_steps=n_steps)

    def test_direct_training_reduces_loss(self):
        ds = self._make_ball_dataset(n_steps=200)
        arch = DeepONetDirect(n_basis=16, branch_hidden=(16, 16))
        trainer = SurrogateTrainer(arch, ds)
        result = trainer.train(
            n_epochs=30, batch_size=32,
            rng_key=jax.random.PRNGKey(42),
        )
        assert result.train_losses[-1] < result.train_losses[0]

    def test_derivative_training_reduces_loss(self):
        ds = self._make_ball_dataset(n_steps=200)
        arch = DeepONetDerivative(n_basis=16, branch_hidden=(16, 16))
        trainer = SurrogateTrainer(arch, ds)
        result = trainer.train(
            n_epochs=30, batch_size=32,
            rng_key=jax.random.PRNGKey(42),
        )
        assert result.train_losses[-1] < result.train_losses[0]


# ------------------------------------------------------------------
# S-DeepONet: init_params + forward + GRU hidden state
# ------------------------------------------------------------------

class TestSDeepONetDirect:
    def test_init_and_forward(self):
        arch = SDeepONetDirect(n_basis=8, gru_hidden_size=8, proj_hidden=(8,))
        # Physical state fields + _gru_hidden
        state_spec = {"x": (), "_gru_hidden": (8,)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        params = arch.init_params(key, state_spec, boundary_spec)

        state = {
            "x": jnp.array(1.0),
            "_gru_hidden": jnp.zeros(8),
        }
        out = arch.forward(params, state, {}, 0.01)
        assert "x" in out
        assert "_gru_hidden" in out
        assert jnp.isfinite(out["x"])
        assert out["_gru_hidden"].shape == (8,)

    def test_mode_is_direct(self):
        arch = SDeepONetDirect()
        assert arch.mode == "direct"

    def test_hidden_size(self):
        arch = SDeepONetDirect(gru_hidden_size=16)
        assert arch.hidden_size() == 16

    def test_hidden_state_evolves(self):
        """GRU hidden state should change across consecutive calls."""
        arch = SDeepONetDirect(n_basis=8, gru_hidden_size=8, proj_hidden=(8,))
        state_spec = {"x": (), "_gru_hidden": (8,)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        params = arch.init_params(key, state_spec, boundary_spec)

        state = {"x": jnp.array(1.0), "_gru_hidden": jnp.zeros(8)}
        out1 = arch.forward(params, state, {}, 0.01)
        out2 = arch.forward(params, out1, {}, 0.01)

        # Hidden state should differ between steps
        assert not jnp.allclose(out1["_gru_hidden"], out2["_gru_hidden"])

    def test_in_surrogate_node(self):
        arch = SDeepONetDirect(n_basis=8, gru_hidden_size=8, proj_hidden=(8,))
        state_spec = {"x": (), "_gru_hidden": (8,)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="test", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"x": 1.0, "_gru_hidden": jnp.zeros(8)},
        )
        state = node.initial_state()
        new_state = node.update(state, {}, 0.01)
        assert jnp.isfinite(new_state["x"])
        assert new_state["_gru_hidden"].shape == (8,)

    def test_jit_compatible(self):
        arch = SDeepONetDirect(n_basis=8, gru_hidden_size=8, proj_hidden=(8,))
        state_spec = {"x": (), "_gru_hidden": (8,)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="test", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"x": 1.0, "_gru_hidden": jnp.zeros(8)},
        )
        state = node.initial_state()
        jitted = jax.jit(node.update)
        result = jitted(state, {}, 0.01)
        assert jnp.isfinite(result["x"])

    def test_scan_multi_step(self):
        """S-DeepONet GRU hidden state persists across scan steps."""
        arch = SDeepONetDirect(n_basis=8, gru_hidden_size=8, proj_hidden=(8,))
        state_spec = {"x": (), "_gru_hidden": (8,)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="test", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"x": 1.0, "_gru_hidden": jnp.zeros(8)},
        )
        state = node.initial_state()

        def step(s, _):
            ns = node.update(s, {}, 0.01)
            return ns, ns["x"]

        final, trace = jax.lax.scan(step, state, None, length=10)
        assert trace.shape == (10,)
        assert jnp.all(jnp.isfinite(trace))
        # Hidden state should have evolved from zeros
        assert not jnp.allclose(final["_gru_hidden"], jnp.zeros(8))


class TestSDeepONetDerivative:
    def test_init_and_forward(self):
        arch = SDeepONetDerivative(n_basis=8, gru_hidden_size=8, proj_hidden=(8,))
        state_spec = {"x": (), "_gru_hidden": (8,)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        params = arch.init_params(key, state_spec, boundary_spec)

        state = {"x": jnp.array(1.0), "_gru_hidden": jnp.zeros(8)}
        out = arch.forward(params, state, {}, 0.01)
        assert "x" in out
        assert "_gru_hidden" in out
        assert jnp.isfinite(out["x"])

    def test_mode_is_derivative(self):
        arch = SDeepONetDerivative()
        assert arch.mode == "derivative"

    def test_hidden_size(self):
        arch = SDeepONetDerivative(gru_hidden_size=24)
        assert arch.hidden_size() == 24
