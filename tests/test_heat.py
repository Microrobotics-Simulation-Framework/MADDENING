"""Tests for HeatNode -- 1D heat diffusion on a rod."""

import warnings

import pytest
import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode


class TestHeatNodeUnit:
    """Unit tests for HeatNode in isolation."""

    def test_uniform_temperature_stays_uniform(self):
        """A rod at uniform temperature with matching BCs should not change."""
        h = HeatNode(name="rod", timestep=0.001, n_cells=20,
                     initial_temperature=100.0)
        state = h.initial_state()
        # BCs match the uniform temperature
        bi = {"left_temperature": jnp.array(100.0),
              "right_temperature": jnp.array(100.0)}
        for _ in range(100):
            state = h.update(state, bi, 0.001)
        assert jnp.allclose(state["temperature"],
                            jnp.full(20, 100.0), atol=1e-5)

    def test_dirichlet_bcs_converge_to_linear(self):
        """Hot ends on a cold rod should converge toward a linear profile."""
        n = 50
        h = HeatNode(name="rod", timestep=0.0001, n_cells=n, length=1.0,
                     thermal_diffusivity=0.1, initial_temperature=0.0)

        # Use GraphManager + run_scan for speed (avoids Python loop)
        gm = GraphManager()
        gm.add_node(h)
        gm.add_external_input("rod", "left_temperature", shape=())
        gm.add_external_input("rod", "right_temperature", shape=())
        gm.compile()

        ext = {
            "rod": {
                "left_temperature": jnp.array(100.0),
                "right_temperature": jnp.array(200.0),
            }
        }
        gm.run_scan(50000, external_inputs=ext)
        T = gm.get_node_state("rod")["temperature"]

        # Steady-state for 1D heat eq with Dirichlet BCs is linear
        # from T_left=100 at cell 0 to T_right=200 at cell n-1
        T_expected = jnp.linspace(100.0, 200.0, n)
        assert jnp.allclose(T, T_expected, atol=5.0), \
            f"Max deviation: {float(jnp.max(jnp.abs(T - T_expected)))}"

    def test_heat_source_increases_temperature_uniformly(self):
        """Uniform source with matching BCs should increase temperature uniformly
        in the interior (boundaries are pinned by Dirichlet BCs)."""
        n = 20
        h = HeatNode(name="rod", timestep=0.001, n_cells=n,
                     initial_temperature=0.0, thermal_diffusivity=0.0)
        state = h.initial_state()
        # Zero diffusivity means only the source term matters in the interior.
        # BCs pin the boundaries at 0.
        source = jnp.ones(n) * 10.0
        bi = {"left_temperature": jnp.array(0.0),
              "right_temperature": jnp.array(0.0),
              "heat_source": source}
        state = h.update(state, bi, 0.001)
        T = state["temperature"]
        # Boundary cells are pinned at 0 by Dirichlet BCs
        assert float(T[0]) == pytest.approx(0.0)
        assert float(T[-1]) == pytest.approx(0.0)
        # Interior cells should increase by source * dt = 10 * 0.001 = 0.01
        for i in range(1, n - 1):
            assert float(T[i]) == pytest.approx(0.01, abs=1e-6)

    def test_energy_conservation_insulated(self):
        """With insulated boundaries (BCs match endpoints) and no source,
        total thermal energy should be conserved."""
        n = 30
        # Non-uniform initial temperature
        T0 = jnp.sin(jnp.linspace(0, jnp.pi, n))
        h = HeatNode(name="rod", timestep=0.0001, n_cells=n, length=1.0,
                     thermal_diffusivity=0.01,
                     initial_temperature=T0.tolist())
        state = h.initial_state()
        initial_energy = float(jnp.sum(state["temperature"]))

        # Run with BCs matching endpoints (insulated-like)
        for _ in range(500):
            T = state["temperature"]
            bi = {"left_temperature": T[0],
                  "right_temperature": T[-1]}
            state = h.update(state, bi, 0.0001)

        final_energy = float(jnp.sum(state["temperature"]))
        # Dirichlet BCs (even set to endpoint values) cause small energy
        # drift because boundary cells are overwritten after diffusion.
        assert final_energy == pytest.approx(initial_energy, rel=0.05)

    def test_step_function_diffuses(self):
        """A step-function initial condition should diffuse (become smoother)."""
        n = 40
        T0 = jnp.concatenate([jnp.zeros(20), jnp.ones(20)]) * 100.0
        h = HeatNode(name="rod", timestep=0.0001, n_cells=n, length=1.0,
                     thermal_diffusivity=0.01,
                     initial_temperature=T0.tolist())
        state = h.initial_state()

        initial_gradient = jnp.max(jnp.abs(jnp.diff(state["temperature"])))

        # Run with insulated BCs
        for _ in range(1000):
            T = state["temperature"]
            bi = {"left_temperature": T[0],
                  "right_temperature": T[-1]}
            state = h.update(state, bi, 0.0001)

        final_gradient = jnp.max(jnp.abs(jnp.diff(state["temperature"])))
        # The maximum gradient should decrease as the step diffuses
        assert float(final_gradient) < float(initial_gradient)

    def test_state_fields(self):
        h = HeatNode(name="rod", timestep=0.01, n_cells=10)
        assert h.state_fields() == ["temperature"]

    def test_to_dict_round_trip(self):
        h = HeatNode(name="rod", timestep=0.01, n_cells=15, length=2.0,
                     thermal_diffusivity=0.05, initial_temperature=50.0)
        d = h.to_dict()
        assert d["type"] == "HeatNode"
        assert d["name"] == "rod"
        assert d["timestep"] == 0.01
        assert d["params"]["n_cells"] == 15
        assert d["params"]["length"] == 2.0
        assert d["params"]["thermal_diffusivity"] == 0.05
        assert d["params"]["initial_temperature"] == 50.0

        # Reconstruct from dict
        h2 = HeatNode(name=d["name"], timestep=d["timestep"], **d["params"])
        state2 = h2.initial_state()
        assert state2["temperature"].shape == (15,)
        assert jnp.allclose(state2["temperature"], jnp.full(15, 50.0))

    def test_non_uniform_initial_temperature(self):
        """Array initial temperature should be used directly."""
        T0 = [10.0, 20.0, 30.0, 40.0, 50.0]
        h = HeatNode(name="rod", timestep=0.01, n_cells=5,
                     initial_temperature=T0)
        state = h.initial_state()
        assert state["temperature"].shape == (5,)
        assert jnp.allclose(state["temperature"],
                            jnp.array(T0, dtype=jnp.float32))


class TestHeatNodeJAX:
    """JAX compilation and differentiation tests."""

    def test_jit_compilation(self):
        """HeatNode update should JIT-compile without errors."""
        h = HeatNode(name="rod", timestep=0.001, n_cells=20,
                     thermal_diffusivity=0.01)
        state = h.initial_state()
        bi = {"left_temperature": jnp.array(100.0),
              "right_temperature": jnp.array(0.0),
              "heat_source": jnp.zeros(20)}

        jit_update = jax.jit(h.update)
        new_state = jit_update(state, bi, 0.001)
        assert new_state["temperature"].shape == (20,)
        assert jnp.all(jnp.isfinite(new_state["temperature"]))

    def test_grad_through_update(self):
        """jax.grad should work through the heat update."""
        h = HeatNode(name="rod", timestep=0.001, n_cells=10,
                     thermal_diffusivity=0.01)

        def loss_fn(left_bc):
            state = h.initial_state()
            bi = {"left_temperature": left_bc,
                  "right_temperature": jnp.array(0.0)}
            for _ in range(10):
                state = h.update(state, bi, 0.001)
            return jnp.sum(state["temperature"])

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(100.0))
        assert jnp.isfinite(g)
        # Increasing left BC should increase total temperature
        assert float(g) > 0.0


class TestHeatNodeInGraph:
    """Tests using the GraphManager infrastructure."""

    def test_run_scan_with_history(self):
        """run_scan_with_history should work with HeatNode."""
        n = 10
        gm = GraphManager()
        h = HeatNode(name="rod", timestep=0.001, n_cells=n, length=1.0,
                     thermal_diffusivity=0.01, initial_temperature=0.0)
        gm.add_node(h)
        gm.add_external_input("rod", "left_temperature", shape=())
        gm.add_external_input("rod", "right_temperature", shape=())
        gm.compile()

        ext = {
            "rod": {
                "left_temperature": jnp.array(100.0),
                "right_temperature": jnp.array(0.0),
            }
        }

        final, history = gm.run_scan_with_history(200, external_inputs=ext)
        assert final["rod"]["temperature"].shape == (n,)
        assert history["rod"]["temperature"].shape == (200, n)
        # Left end should be hot
        assert float(final["rod"]["temperature"][0]) == pytest.approx(100.0)
        # Right end should be cold
        assert float(final["rod"]["temperature"][-1]) == pytest.approx(0.0)

    def test_two_rods_connected_by_edge(self):
        """Temperature from one heat node drives the BC of another via an edge."""
        n = 10
        gm = GraphManager()
        rod_a = HeatNode(name="rod_a", timestep=0.001, n_cells=n,
                         thermal_diffusivity=0.01,
                         initial_temperature=100.0)
        rod_b = HeatNode(name="rod_b", timestep=0.001, n_cells=n,
                         thermal_diffusivity=0.01,
                         initial_temperature=0.0)
        gm.add_node(rod_a)
        gm.add_node(rod_b)

        # rod_a's temperature field drives rod_b's left BC.
        # We use a transform to extract the rightmost cell of rod_a.
        gm.add_edge(
            source="rod_a", target="rod_b",
            source_field="temperature", target_field="left_temperature",
            transform=lambda T: T[-1],  # rightmost cell of rod_a
        )

        # Give rod_a fixed BCs via external inputs
        gm.add_external_input("rod_a", "left_temperature", shape=())
        gm.add_external_input("rod_a", "right_temperature", shape=())
        gm.add_external_input("rod_b", "right_temperature", shape=())

        gm.compile()

        ext = {
            "rod_a": {
                "left_temperature": jnp.array(100.0),
                "right_temperature": jnp.array(100.0),
            },
            "rod_b": {
                "right_temperature": jnp.array(0.0),
            },
        }

        final, history = gm.run_scan_with_history(500, external_inputs=ext)

        # rod_a stays at ~100 (uniform, matching BCs)
        assert jnp.allclose(final["rod_a"]["temperature"],
                            jnp.full(n, 100.0), atol=1.0)
        # rod_b's left end should be warm (driven by rod_a's right end ~100)
        assert float(final["rod_b"]["temperature"][0]) > 50.0
        # rod_b's right end pinned at 0
        assert float(final["rod_b"]["temperature"][-1]) == pytest.approx(0.0)

    def test_graph_serialization_round_trip(self):
        """GraphManager.to_dict / from_dict should work with HeatNode."""
        gm = GraphManager()
        h = HeatNode(name="rod", timestep=0.01, n_cells=8, length=0.5,
                     thermal_diffusivity=0.02, initial_temperature=25.0)
        gm.add_node(h)
        gm.add_external_input("rod", "left_temperature", shape=())
        gm.compile()

        config = gm.to_dict()

        registry = {"HeatNode": HeatNode}
        gm2 = GraphManager.from_dict(config, registry)
        state = gm2.get_node_state("rod")
        assert state["temperature"].shape == (8,)
        assert jnp.allclose(state["temperature"], jnp.full(8, 25.0))
