"""Tests for v0.2 #3 static_data channel on SimulationNode."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode


# ---------------------------------------------------------------------------
# Helper nodes
# ---------------------------------------------------------------------------


class _PointwiseNode(SimulationNode):
    """A pointwise node with no static_data (the default)."""

    def initial_state(self) -> dict:
        return {"x": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] + dt}


class _StaticDataNode(SimulationNode):
    """A node carrying an N-length lookup table in static_data.

    The table size is parameterised so tests can swap a node with a
    different-shape table and verify the JIT cache invalidates.
    """

    def __init__(self, name, timestep, n: int = 4, dtype=jnp.float32):
        super().__init__(name, timestep, n=n)
        self._lut = jnp.arange(n, dtype=dtype)

    @property
    def static_data(self) -> dict:
        return {"lookup": self._lut}

    def initial_state(self):
        return {"y": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt):
        # Use the LUT inside update so its shape matters for JIT.
        first = self._lut[0]
        return {"y": state["y"] + dt + first}


class _ScalarStaticDataNode(SimulationNode):
    """Node whose static_data is a non-array scalar dict."""

    def __init__(self, name, timestep, mode="A"):
        super().__init__(name, timestep, mode=mode)
        self._mode = mode

    @property
    def static_data(self) -> dict:
        return {"mode": self._mode, "version": 1}

    def initial_state(self):
        return {"z": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt):
        return {"z": state["z"] + dt}


# ---------------------------------------------------------------------------
# Default behaviour: empty static_data on SimulationNode
# ---------------------------------------------------------------------------


class TestDefaultStaticData:
    def test_default_static_data_is_empty(self):
        n = _PointwiseNode("p", timestep=0.01)
        assert n.static_data == {}

    def test_default_hash_is_zero(self):
        n = _PointwiseNode("p", timestep=0.01)
        assert n.static_data_hash() == 0

    def test_default_does_not_break_existing_nodes(self):
        # Sanity check: an unmodified concrete node still works in a graph
        gm = GraphManager()
        gm.add_node(_PointwiseNode("a", timestep=0.01))
        gm.compile()
        state = gm.step()
        assert "a" in state
        assert "x" in state["a"]


# ---------------------------------------------------------------------------
# Override semantics
# ---------------------------------------------------------------------------


class TestOverrideStaticData:
    def test_can_carry_jax_array(self):
        n = _StaticDataNode("s", timestep=0.01, n=8)
        sd = n.static_data
        assert "lookup" in sd
        assert sd["lookup"].shape == (8,)

    def test_can_carry_scalar_values(self):
        n = _ScalarStaticDataNode("s", timestep=0.01, mode="X")
        sd = n.static_data
        assert sd == {"mode": "X", "version": 1}

    def test_hash_nonzero_when_static_data_present(self):
        n = _StaticDataNode("s", timestep=0.01, n=8)
        assert n.static_data_hash() != 0


# ---------------------------------------------------------------------------
# Hash semantics: by shape+dtype, not by content
# ---------------------------------------------------------------------------


class TestStaticDataHash:
    def test_same_shape_dtype_yields_same_hash_even_if_values_differ(self):
        # Two nodes whose static_data has the same (key, shape, dtype)
        # must hash identically — JIT cache key tracks shape, not values.
        n1 = _StaticDataNode("a", timestep=0.01, n=4)
        n2 = _StaticDataNode("b", timestep=0.01, n=4)
        # Force-mutate one node's lookup contents
        n2._lut = jnp.array([100.0, 200.0, 300.0, 400.0], dtype=jnp.float32)
        assert n1.static_data_hash() == n2.static_data_hash()

    def test_different_shape_yields_different_hash(self):
        n1 = _StaticDataNode("a", timestep=0.01, n=4)
        n2 = _StaticDataNode("b", timestep=0.01, n=8)
        assert n1.static_data_hash() != n2.static_data_hash()

    def test_different_dtype_yields_different_hash(self):
        n1 = _StaticDataNode("a", timestep=0.01, n=4, dtype=jnp.float32)
        n2 = _StaticDataNode("b", timestep=0.01, n=4, dtype=jnp.int32)
        assert n1.static_data_hash() != n2.static_data_hash()

    def test_different_scalar_value_yields_different_hash(self):
        n1 = _ScalarStaticDataNode("a", timestep=0.01, mode="A")
        n2 = _ScalarStaticDataNode("b", timestep=0.01, mode="B")
        assert n1.static_data_hash() != n2.static_data_hash()

    def test_hash_is_int(self):
        n = _StaticDataNode("a", timestep=0.01, n=4)
        assert isinstance(n.static_data_hash(), int)

    def test_hash_stable_across_calls(self):
        n = _StaticDataNode("a", timestep=0.01, n=4)
        h1 = n.static_data_hash()
        h2 = n.static_data_hash()
        h3 = n.static_data_hash()
        assert h1 == h2 == h3

    def test_hash_with_mixed_array_and_scalar(self):
        class Mixed(SimulationNode):
            def initial_state(self): return {"x": jnp.array(0.0)}
            def update(self, s, b, dt): return s
            @property
            def static_data(self):
                return {"arr": jnp.zeros(3), "tag": "v1", "n": 7}

        n = Mixed("m", timestep=0.01)
        # Just verify it doesn't crash and yields a stable int
        h = n.static_data_hash()
        assert isinstance(h, int)
        assert h == n.static_data_hash()


# ---------------------------------------------------------------------------
# GraphManager integration: snapshot + dirty detection
# ---------------------------------------------------------------------------


class TestGraphManagerSnapshot:
    def test_compile_records_per_node_hashes(self):
        gm = GraphManager()
        gm.add_node(_PointwiseNode("p", timestep=0.01))
        gm.add_node(_StaticDataNode("s", timestep=0.01, n=4))
        gm.compile()
        assert gm._static_data_hashes["p"] == 0
        assert gm._static_data_hashes["s"] != 0

    def test_check_static_data_dirty_false_after_fresh_compile(self):
        gm = GraphManager()
        gm.add_node(_StaticDataNode("s", timestep=0.01, n=4))
        gm.compile()
        assert gm._check_static_data_dirty() is False
        assert gm._dirty is False

    def test_check_static_data_dirty_true_when_static_data_changes(self):
        gm = GraphManager()
        node = _StaticDataNode("s", timestep=0.01, n=4)
        gm.add_node(node)
        gm.compile()
        # Mutate the node's static_data to a different shape
        node._lut = jnp.arange(8, dtype=jnp.float32)
        assert gm._check_static_data_dirty() is True
        assert gm._dirty is True

    def test_step_auto_recompiles_after_static_data_change(self):
        gm = GraphManager()
        node = _StaticDataNode("s", timestep=0.01, n=4)
        gm.add_node(node)
        gm.compile()
        # First step works
        gm.step()
        # Mutate static_data shape
        node._lut = jnp.arange(8, dtype=jnp.float32)
        # The next step should silently recompile and succeed
        state = gm.step()
        assert "s" in state
        # Hash snapshot should now reflect the new shape
        new_hash = node.static_data_hash()
        assert gm._static_data_hashes["s"] == new_hash

    def test_replace_node_with_different_static_data_recompiles(self):
        gm = GraphManager()
        gm.add_node(_StaticDataNode("s", timestep=0.01, n=4))
        gm.compile()
        h_old = gm._static_data_hashes["s"]

        # Remove and re-add with different size — emulates the
        # replace_node pattern from surrogates/replace.py.
        gm.remove_node("s")
        gm.add_node(_StaticDataNode("s", timestep=0.01, n=8))
        # add_node sets _dirty=True so compile() will refresh hashes
        gm.compile()
        h_new = gm._static_data_hashes["s"]
        assert h_old != h_new

    def test_static_data_constant_does_not_trigger_recompile(self):
        gm = GraphManager()
        gm.add_node(_StaticDataNode("s", timestep=0.01, n=4))
        gm.compile()
        # Step multiple times; no recompile should happen.
        compile_count = [0]
        original_compile = gm.compile

        def counting_compile():
            compile_count[0] += 1
            return original_compile()

        gm.compile = counting_compile
        for _ in range(5):
            gm.step()
        assert compile_count[0] == 0


# ---------------------------------------------------------------------------
# Closure behaviour: static_data is reachable from update()
# ---------------------------------------------------------------------------


class TestStaticDataInUpdate:
    def test_static_data_value_used_in_step(self):
        # _StaticDataNode.update returns state["y"] + dt + lut[0].
        # With LUT = arange(n) → lut[0] = 0 → state increases by dt each step.
        gm = GraphManager()
        gm.add_node(_StaticDataNode("s", timestep=0.1, n=4))
        gm.compile()
        for _ in range(5):
            gm.step()
        # 5 steps of dt=0.1 → expect y ≈ 0.5
        state = gm._state
        assert abs(float(state["s"]["y"]) - 0.5) < 1e-5

    def test_static_data_can_carry_large_array(self):
        # 1 million-element LUT must not break compilation or stepping.
        class BigStatic(SimulationNode):
            def __init__(self, name, timestep):
                super().__init__(name, timestep)
                self._big = jnp.arange(1_000_000, dtype=jnp.float32)

            @property
            def static_data(self): return {"big": self._big}

            def initial_state(self): return {"y": jnp.array(0.0)}

            def update(self, state, boundary_inputs, dt):
                return {"y": state["y"] + self._big[42] * dt}

        gm = GraphManager()
        gm.add_node(BigStatic("b", timestep=0.01))
        gm.compile()
        gm.step()
        # 42 * 0.01 = 0.42
        assert abs(float(gm._state["b"]["y"]) - 0.42) < 1e-4


# ---------------------------------------------------------------------------
# Static data and surrogate replace_node
# ---------------------------------------------------------------------------


class TestReplaceNodeIntegration:
    def test_replace_with_static_data_node_dirties_graph(self):
        gm = GraphManager()
        gm.add_node(_PointwiseNode("x", timestep=0.01))
        gm.compile()
        assert gm._dirty is False
        # remove + add (the pattern replace_node uses) sets _dirty=True
        gm.remove_node("x")
        gm.add_node(_StaticDataNode("x", timestep=0.01, n=4))
        assert gm._dirty is True
        gm.compile()
        assert gm._static_data_hashes["x"] != 0


# ---------------------------------------------------------------------------
# HeatNode migration (v0.2 #3 follow-up)
# ---------------------------------------------------------------------------


class TestHeatNodeStaticData:
    """HeatNode is the first in-tree consumer of the static_data channel.
    Migrating ``grid_x`` here pins the API contract the v0.2 brief
    called out: 'at least one node uses static_data for real'."""

    def test_heatnode_exposes_grid_x_via_static_data(self):
        from maddening.nodes.heat import HeatNode

        node = HeatNode("rod", timestep=0.01, n_cells=8, length=2.0)
        sd = node.static_data
        assert "grid_x" in sd
        assert sd["grid_x"].shape == (8,)

    def test_heatnode_uniform_grid_matches_linspace(self):
        from maddening.nodes.heat import HeatNode
        import numpy as np

        node = HeatNode("rod", timestep=0.01, n_cells=10, length=1.0)
        x = np.asarray(node.static_data["grid_x"])
        # Cell centres: dx/2, dx + dx/2, ..., L - dx/2
        expected = np.linspace(0.05, 0.95, 10, dtype=np.float32)
        assert np.allclose(x, expected, atol=1e-6)

    def test_heatnode_nonuniform_grid_round_trips(self):
        from maddening.nodes.heat import HeatNode
        import numpy as np

        custom = [0.0, 0.1, 0.3, 0.6, 1.0]
        node = HeatNode(
            "rod", timestep=0.01, n_cells=5, length=1.0,
            grid_points=custom,
        )
        x = np.asarray(node.static_data["grid_x"])
        assert np.allclose(x, custom, atol=1e-6)

    def test_static_data_is_stable_across_calls(self):
        from maddening.nodes.heat import HeatNode

        node = HeatNode("rod", timestep=0.01, n_cells=8, length=2.0)
        # Same object identity → JAX won't retrace
        assert node.static_data["grid_x"] is node.static_data["grid_x"]

    def test_grid_x_property_aliases_static_data(self):
        from maddening.nodes.heat import HeatNode

        node = HeatNode("rod", timestep=0.01, n_cells=8, length=2.0)
        assert node._grid_x is node.static_data["grid_x"]

    def test_hash_differs_between_uniform_grid_sizes(self):
        from maddening.nodes.heat import HeatNode

        a = HeatNode("a", timestep=0.01, n_cells=8, length=2.0)
        b = HeatNode("b", timestep=0.01, n_cells=10, length=2.0)
        assert a.static_data_hash() != b.static_data_hash()

    def test_hash_same_for_same_shape_and_dtype(self):
        # Two nodes with the same n_cells + dtype hash identically even
        # if the actual grid point *values* differ.  This is the
        # documented contract: shape+dtype only, not contents.
        from maddening.nodes.heat import HeatNode

        a = HeatNode("a", timestep=0.01, n_cells=5, length=1.0)
        b = HeatNode("b", timestep=0.01, n_cells=5, length=1.0,
                     grid_points=[0.0, 0.1, 0.3, 0.6, 1.0])
        assert a.static_data_hash() == b.static_data_hash()

    def test_checkpoint_roundtrip_with_static_data(self, tmp_path):
        """The static_data contract says: arrays reconstruct from
        self.params on load.  HeatNode demonstrates this — saving +
        loading a graph preserves its grid_x without serialising the
        array itself."""
        from maddening.core.simulation.checkpoint import (
            save_state, load_state,
        )
        from maddening.nodes.heat import HeatNode
        import numpy as np

        # Source graph with a non-uniform grid (so reconstruction is
        # not trivially the linspace default)
        gm_src = GraphManager()
        gm_src.add_node(HeatNode(
            "rod", timestep=0.01, n_cells=5, length=1.0,
            grid_points=[0.0, 0.1, 0.3, 0.6, 1.0],
        ))
        gm_src.compile()
        gm_src.step()

        snap = tmp_path / "heat.npz"
        save_state(gm_src, snap)

        # Reconstruct from params alone (the user's responsibility).
        gm_dst = GraphManager()
        gm_dst.add_node(HeatNode(
            "rod", timestep=0.01, n_cells=5, length=1.0,
            grid_points=[0.0, 0.1, 0.3, 0.6, 1.0],
        ))
        gm_dst.compile()
        load_state(gm_dst, snap)

        # The state was restored
        assert np.allclose(
            np.asarray(gm_dst.get_node_state("rod")["temperature"]),
            np.asarray(gm_src.get_node_state("rod")["temperature"]),
        )
        # The static_data was rebuilt from params (not from the .npz)
        assert np.allclose(
            np.asarray(gm_dst._nodes["rod"].node.static_data["grid_x"]),
            [0.0, 0.1, 0.3, 0.6, 1.0],
            atol=1e-6,
        )

    def test_heatnode_step_runs_with_uniform_grid(self):
        # End-to-end: a graph with a HeatNode actually steps.  This
        # exercises the static_data → JIT closure path for real.
        from maddening.nodes.heat import HeatNode

        gm = GraphManager()
        gm.add_node(HeatNode(
            "rod", timestep=0.01, n_cells=10, length=1.0,
            initial_temperature=100.0,
        ))
        gm.compile()
        for _ in range(5):
            gm.step()
        # Should not have NaN-ed; temperature still finite
        import numpy as np
        T = np.asarray(gm.get_node_state("rod")["temperature"])
        assert np.all(np.isfinite(T))
