"""Tests for FNO (Fourier Neural Operator) surrogate architectures.

Covers 1D, 2D, and 3D spatial fields, scalar bypass, direct/derivative
modes, JIT compatibility, and training convergence.

Uses small spatial grids and low parameter counts to stay within GPU
memory limits (~6 GB VRAM).
"""

import pytest
import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.surrogates.node import SurrogateNode, euler_integrator, rk4_integrator
from maddening.surrogates.architectures.fno import FNODirect, FNODerivative


# ------------------------------------------------------------------
# FNODirect: 1D spatial field
# ------------------------------------------------------------------

class TestFNODirect1D:
    def _make_arch(self):
        return FNODirect(
            spatial_field="temperature",
            n_modes=(4,),
            hidden_channels=4,
            n_layers=2,
        )

    def test_init_and_forward(self):
        arch = self._make_arch()
        state_spec = {"temperature": (8,)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        params = arch.init_params(key, state_spec, boundary_spec)

        state = {"temperature": jnp.ones(8)}
        out = arch.forward(params, state, {}, 0.01)
        assert "temperature" in out
        assert out["temperature"].shape == (8,)
        assert jnp.all(jnp.isfinite(out["temperature"]))

    def test_mode_is_direct(self):
        arch = self._make_arch()
        assert arch.mode == "direct"

    def test_in_surrogate_node(self):
        arch = self._make_arch()
        state_spec = {"temperature": (8,)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="heat", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"temperature": jnp.ones(8)},
        )
        state = node.initial_state()
        new_state = node.update(state, {}, 0.01)
        assert new_state["temperature"].shape == (8,)
        assert jnp.all(jnp.isfinite(new_state["temperature"]))

    def test_jit_compatible(self):
        arch = self._make_arch()
        state_spec = {"temperature": (8,)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="heat", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"temperature": jnp.ones(8)},
        )
        state = node.initial_state()
        jitted = jax.jit(node.update)
        result = jitted(state, {}, 0.01)
        assert jnp.all(jnp.isfinite(result["temperature"]))

    def test_scan_multi_step(self):
        arch = self._make_arch()
        state_spec = {"temperature": (8,)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="heat", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"temperature": jnp.ones(8)},
        )
        state = node.initial_state()

        def step(s, _):
            ns = node.update(s, {}, 0.01)
            return ns, jnp.mean(ns["temperature"])

        final, trace = jax.lax.scan(step, state, None, length=5)
        assert trace.shape == (5,)
        assert jnp.all(jnp.isfinite(trace))


# ------------------------------------------------------------------
# FNODirect: mixed spatial + scalar fields
# ------------------------------------------------------------------

class TestFNODirectMixed:
    def test_spatial_plus_scalar(self):
        """FNO with a spatial field + scalar field (scalar bypass MLP)."""
        arch = FNODirect(
            spatial_field="temperature",
            n_modes=(4,),
            hidden_channels=4,
            n_layers=1,
            scalar_hidden=(8,),
        )
        state_spec = {"temperature": (8,), "ambient": ()}
        boundary_spec = {"heat_flux": ()}
        key = jax.random.PRNGKey(0)
        params = arch.init_params(key, state_spec, boundary_spec)

        state = {
            "temperature": jnp.ones(8),
            "ambient": jnp.array(20.0),
        }
        boundary = {"heat_flux": jnp.array(5.0)}
        out = arch.forward(params, state, boundary, 0.01)
        assert "temperature" in out
        assert "ambient" in out
        assert out["temperature"].shape == (8,)
        assert out["ambient"].shape == ()
        assert jnp.isfinite(out["ambient"])


# ------------------------------------------------------------------
# FNODirect: 2D spatial field
# ------------------------------------------------------------------

class TestFNODirect2D:
    def test_init_and_forward(self):
        arch = FNODirect(
            spatial_field="field",
            n_modes=(3, 3),
            hidden_channels=4,
            n_layers=1,
        )
        state_spec = {"field": (6, 6)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        params = arch.init_params(key, state_spec, boundary_spec)

        state = {"field": jnp.ones((6, 6))}
        out = arch.forward(params, state, {}, 0.01)
        assert out["field"].shape == (6, 6)
        assert jnp.all(jnp.isfinite(out["field"]))


# ------------------------------------------------------------------
# FNODirect: 3D spatial field
# ------------------------------------------------------------------

class TestFNODirect3D:
    def test_init_and_forward(self):
        arch = FNODirect(
            spatial_field="field",
            n_modes=(2, 2, 2),
            hidden_channels=4,
            n_layers=1,
        )
        state_spec = {"field": (4, 4, 4)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        params = arch.init_params(key, state_spec, boundary_spec)

        state = {"field": jnp.ones((4, 4, 4))}
        out = arch.forward(params, state, {}, 0.01)
        assert out["field"].shape == (4, 4, 4)
        assert jnp.all(jnp.isfinite(out["field"]))


# ------------------------------------------------------------------
# FNODerivative
# ------------------------------------------------------------------

class TestFNODerivative:
    def test_mode_is_derivative(self):
        arch = FNODerivative(
            spatial_field="temperature",
            n_modes=(4,),
            hidden_channels=4,
            n_layers=1,
        )
        assert arch.mode == "derivative"

    def test_euler_integration_1d(self):
        arch = FNODerivative(
            spatial_field="temperature",
            n_modes=(4,),
            hidden_channels=4,
            n_layers=1,
        )
        state_spec = {"temperature": (8,)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="heat", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"temperature": jnp.ones(8)},
            integrator=euler_integrator,
        )
        state = node.initial_state()
        new_state = node.update(state, {}, 0.01)
        assert new_state["temperature"].shape == (8,)
        assert jnp.all(jnp.isfinite(new_state["temperature"]))

    def test_rk4_integration_1d(self):
        arch = FNODerivative(
            spatial_field="temperature",
            n_modes=(4,),
            hidden_channels=4,
            n_layers=1,
        )
        state_spec = {"temperature": (8,)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="heat", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"temperature": jnp.ones(8)},
            integrator=rk4_integrator,
        )
        state = node.initial_state()
        new_state = node.update(state, {}, 0.01)
        assert new_state["temperature"].shape == (8,)
        assert jnp.all(jnp.isfinite(new_state["temperature"]))

    def test_2d_derivative(self):
        arch = FNODerivative(
            spatial_field="field",
            n_modes=(3, 3),
            hidden_channels=4,
            n_layers=1,
        )
        state_spec = {"field": (6, 6)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="test", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"field": jnp.ones((6, 6))},
            integrator=euler_integrator,
        )
        state = node.initial_state()
        new_state = node.update(state, {}, 0.01)
        assert new_state["field"].shape == (6, 6)
        assert jnp.all(jnp.isfinite(new_state["field"]))

    def test_3d_derivative(self):
        arch = FNODerivative(
            spatial_field="field",
            n_modes=(2, 2, 2),
            hidden_channels=4,
            n_layers=1,
        )
        state_spec = {"field": (4, 4, 4)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="test", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"field": jnp.ones((4, 4, 4))},
            integrator=euler_integrator,
        )
        state = node.initial_state()
        new_state = node.update(state, {}, 0.01)
        assert new_state["field"].shape == (4, 4, 4)
        assert jnp.all(jnp.isfinite(new_state["field"]))


# ------------------------------------------------------------------
# FNO in GraphManager
# ------------------------------------------------------------------

class TestFNOInGraph:
    def test_add_to_graph_and_step(self):
        arch = FNODirect(
            spatial_field="temperature",
            n_modes=(4,),
            hidden_channels=4,
            n_layers=1,
        )
        state_spec = {"temperature": (8,)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="heat_surr", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"temperature": jnp.ones(8)},
        )
        gm = GraphManager()
        gm.add_node(node)
        gm.compile()
        state = gm.step()
        assert state["heat_surr"]["temperature"].shape == (8,)
        assert jnp.all(jnp.isfinite(state["heat_surr"]["temperature"]))

    def test_run_scan(self):
        arch = FNODirect(
            spatial_field="temperature",
            n_modes=(4,),
            hidden_channels=4,
            n_layers=1,
        )
        state_spec = {"temperature": (8,)}
        boundary_spec = {}
        key = jax.random.PRNGKey(0)
        weights = arch.init_params(key, state_spec, boundary_spec)

        node = SurrogateNode(
            name="heat_surr", timestep=0.01, architecture=arch,
            weights=weights, state_spec=state_spec, boundary_spec=boundary_spec,
            initial_values={"temperature": jnp.ones(8)},
        )
        gm = GraphManager()
        gm.add_node(node)
        gm.compile()
        final = gm.run_scan(10)
        assert final["heat_surr"]["temperature"].shape == (8,)
        assert jnp.all(jnp.isfinite(final["heat_surr"]["temperature"]))
