"""Tests for v0.2 #3 static_data channel on SimulationNode."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.params import ParamSpec
from maddening.core.simulation.hybrid_node import HybridNode
from maddening.core.static_data import StaticArray


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
        return {"lookup": StaticArray(self._lut)}

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
        assert isinstance(sd["lookup"], StaticArray)
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
                return {
                    "arr": StaticArray(jnp.zeros(3)),
                    "tag": "v1",
                    "n": 7,
                }

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
            def static_data(self): return {"big": StaticArray(self._big)}

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
        assert isinstance(sd["grid_x"], StaticArray)
        assert sd["grid_x"].shape == (8,)

    def test_heatnode_grid_x_declares_shard_along_axis_0(self):
        """v0.2 #3 follow-up: HeatNode opts into per-shard slicing
        along the cell axis."""
        from maddening.nodes.heat import HeatNode
        node = HeatNode("rod", timestep=0.01, n_cells=8, length=2.0)
        sd_arr = node.static_data["grid_x"]
        assert sd_arr.replication == "shard"
        assert sd_arr.shard_axis == 0

    def test_heatnode_uniform_grid_matches_linspace(self):
        from maddening.nodes.heat import HeatNode
        import numpy as np

        node = HeatNode("rod", timestep=0.01, n_cells=10, length=1.0)
        x = np.asarray(node.static_data["grid_x"].value)
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
        x = np.asarray(node.static_data["grid_x"].value)
        assert np.allclose(x, custom, atol=1e-6)

    def test_static_data_is_stable_across_calls(self):
        from maddening.nodes.heat import HeatNode

        node = HeatNode("rod", timestep=0.01, n_cells=8, length=2.0)
        # Underlying array is the same object on repeat access (so JAX
        # won't retrace), even though the StaticArray wrapper is rebuilt.
        assert node.static_data["grid_x"].value is node.static_data["grid_x"].value

    def test_grid_x_property_aliases_static_data(self):
        from maddening.nodes.heat import HeatNode

        node = HeatNode("rod", timestep=0.01, n_cells=8, length=2.0)
        assert node._grid_x is node.static_data["grid_x"].value

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
            np.asarray(gm_dst._nodes["rod"].node.static_data["grid_x"].value),
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


# ---------------------------------------------------------------------------
# A wrapper reports the statics of the node it wraps.
#
# ``SimulationNode.static_data`` defaults to the merged static_data of
# every node held as an attribute, so a wrapper that declares none of its
# own still hashes what it wraps.  Before that, every wrapper hashed to
# ``0`` forever and ``_check_static_data_dirty`` could not fire for the
# nodes most likely to need it.
# ---------------------------------------------------------------------------


class _Wrapper(SimulationNode):
    """A wrapper that declares no static_data of its own."""

    def __init__(self, inner: SimulationNode):
        super().__init__(inner.name, inner.delta_t)
        self.inner = inner

    def initial_state(self):
        return self.inner.initial_state()

    def update(self, state, boundary_inputs, dt):
        return self.inner.update(state, boundary_inputs, dt)


class _WrapperWithOwnStatics(_Wrapper):
    """A wrapper that adds a static of its own and merges the rest in."""

    @property
    def static_data(self) -> dict:
        return {"own": StaticArray(jnp.zeros(3, dtype=jnp.float32)),
                **super().static_data}


class _TwoNodeHolder(SimulationNode):
    """Holds two nodes that both declare a ``lookup``."""

    def __init__(self, name, timestep, first, second):
        super().__init__(name, timestep)
        self.first = first
        self.second = second

    def initial_state(self):
        return {"y": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt):
        return {"y": state["y"] + dt}


class TestWrapperStaticDataProxy:
    def test_a_node_that_wraps_nothing_reports_nothing(self):
        """The forwarding default must not invent statics for a leaf."""
        assert _PointwiseNode("p", timestep=0.01).static_data == {}
        assert _PointwiseNode("p", timestep=0.01).static_data_hash() == 0

    def test_one_wrapper_reports_the_wrapped_nodes_statics(self):
        inner = _StaticDataNode("s", timestep=0.01, n=8)
        wrapper = _Wrapper(inner)
        assert set(wrapper.static_data) == {"lookup"}
        assert wrapper.static_data["lookup"].shape == (8,)
        assert wrapper.static_data_hash() == inner.static_data_hash()
        assert wrapper.static_data_hash() != 0

    def test_a_production_wrapper_proxies_too(self):
        """``HybridNode`` declares no statics, so it inherits the default."""
        inner = _StaticDataNode("s", timestep=0.01, n=8)
        hybrid = HybridNode(inner, lambda state, boundary_inputs, dt: {})
        assert hybrid.static_data_hash() == inner.static_data_hash() != 0

    def test_the_proxy_composes_through_a_nested_pair(self):
        """Two levels of wrapping, because one level is not the real case.

        ``HybridNode(ShardedStencilNode(node))`` is the shape the sharded
        statics cache lives in; a proxy that only covered the outermost
        wrapper would report ``0`` for it.
        """
        inner = _StaticDataNode("s", timestep=0.01, n=8)
        nested = HybridNode(_Wrapper(inner), lambda s, b, d: {})
        assert set(nested.static_data) == {"lookup"}
        assert nested.static_data["lookup"].shape == (8,)
        assert nested.static_data_hash() == inner.static_data_hash() != 0

    def test_hash_moves_when_the_wrapped_nodes_statics_change(self):
        inner = _StaticDataNode("s", timestep=0.01, n=4)
        nested = HybridNode(_Wrapper(inner), lambda s, b, d: {})
        before = nested.static_data_hash()
        inner._lut = jnp.arange(8, dtype=jnp.float32)      # a remesh
        assert nested.static_data_hash() != before

    def test_hash_holds_when_the_wrapped_nodes_statics_do_not(self):
        """A check that always fires is as useless as one that never does.

        The hash is over shape/dtype/replication/shard_axis by contract,
        never over contents, so rebuilding the same-shaped array must
        leave it alone.
        """
        inner = _StaticDataNode("s", timestep=0.01, n=4)
        nested = HybridNode(_Wrapper(inner), lambda s, b, d: {})
        before = nested.static_data_hash()
        assert nested.static_data_hash() == before          # stable re-read
        inner._lut = jnp.arange(4, dtype=jnp.float32) * 3.0  # same shape
        assert nested.static_data_hash() == before

    def test_drift_check_fires_for_a_changed_wrapped_node(self):
        gm = GraphManager()
        inner = _StaticDataNode("s", timestep=0.01, n=4)
        gm.add_node(HybridNode(_Wrapper(inner), lambda s, b, d: {}))
        gm.compile()
        assert gm._static_data_hashes["s"] != 0
        inner._lut = jnp.arange(8, dtype=jnp.float32)
        assert gm._check_static_data_dirty() is True
        assert gm._dirty is True

    def test_drift_check_stays_quiet_for_an_unchanged_wrapped_node(self):
        gm = GraphManager()
        inner = _StaticDataNode("s", timestep=0.01, n=4)
        gm.add_node(HybridNode(_Wrapper(inner), lambda s, b, d: {}))
        gm.compile()
        for _ in range(3):
            gm.step()
            assert gm._check_static_data_dirty() is False
            assert gm._dirty is False

    def test_a_wrapper_can_add_its_own_statics_and_keep_the_wrapped_ones(self):
        inner = _StaticDataNode("s", timestep=0.01, n=4)
        wrapper = _WrapperWithOwnStatics(inner)
        assert set(wrapper.static_data) == {"own", "lookup"}
        assert wrapper.static_data_hash() not in (
            0, inner.static_data_hash(),
        )

    def test_two_wrapped_nodes_sharing_a_key_both_reach_the_hash(self):
        """Neither may silently displace the other and go unhashed."""
        first = _StaticDataNode("a", timestep=0.01, n=4)
        second = _StaticDataNode("b", timestep=0.01, n=8)
        holder = _TwoNodeHolder("h", 0.01, first, second)
        assert len(holder.static_data) == 2
        before = holder.static_data_hash()
        second._lut = jnp.arange(16, dtype=jnp.float32)
        after = holder.static_data_hash()
        assert after != before
        first._lut = jnp.arange(2, dtype=jnp.float32)
        assert holder.static_data_hash() != after

    def test_a_cycle_between_two_nodes_terminates(self):
        class _Holder(SimulationNode):
            def __init__(self, name):
                super().__init__(name, timestep=0.01)
                self.other: SimulationNode | None = None

            def initial_state(self):
                return {"x": jnp.array(0.0)}

            def update(self, state, boundary_inputs, dt):
                return {"x": state["x"] + dt}

        a, b = _Holder("a"), _Holder("b")
        a.other, b.other = b, a
        assert a.static_data == {}          # must return, not recurse
        assert a.static_data_hash() == 0


# ---------------------------------------------------------------------------
# D10 steps 2 and 3: declaring where a static comes from, and refusing the
# one provenance that cannot work.
#
# A static built in ``__init__`` from a parameter goes stale invisibly: a
# live parameter write does not dirty the graph, and ``static_data_hash``
# covers shape and dtype but never contents.  ``static_data_deps`` is the
# node's declaration of that link; ``compile()`` refuses the case no
# rebuild could fix, a static derived from a *trainable* parameter, since
# the static is baked into the HLO as a constant and the gradient would
# silently omit the term through it.
# ---------------------------------------------------------------------------


class _DerivedStaticNode(SimulationNode):
    """A node whose static table is built in ``__init__`` from ``scale``.

    ``scale`` is a float (a leaf of ``params_pytree``) whose trainability
    the test chooses; ``n`` is an ``int``, so it is structural and can
    never be a violation however it is declared.
    """

    def __init__(self, name, timestep, *, scale=2.0, n=4, trainable=True):
        super().__init__(name, timestep, scale=scale, n=n)
        self._trainable = trainable
        self._table = jnp.arange(n, dtype=jnp.float32) * scale

    @property
    def static_data(self) -> dict:
        return {"table": StaticArray(self._table)}

    def static_data_deps(self) -> dict:
        return {"table": ("scale", "n")}

    def param_specs(self) -> dict:
        return {**super().param_specs(),
                "scale": ParamSpec(trainable=self._trainable)}

    def initial_state(self):
        return {"y": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        # ``scale`` reaches the step twice: traced through ``params``,
        # and baked through the table built from it in ``__init__``.
        # That split is exactly what the refusal is about -- a gradient
        # through the first alone is missing the second.
        p = {**self.params, **(params or {})}
        return {"y": state["y"] + (self._table[0] + p["scale"]) * dt}


def _compiled_graph(node):
    gm = GraphManager()
    gm.add_node(node)
    gm.compile()
    return gm


class TestStaticDataDepsDeclaration:
    def test_a_node_declares_no_dependencies_by_default(self):
        """The declaration is opt-in: a node that says nothing owes nothing."""
        assert _PointwiseNode("p", timestep=0.01).static_data_deps() == {}
        assert _StaticDataNode("s", timestep=0.01).static_data_deps() == {}

    def test_a_declaration_keys_on_the_nodes_static_data_keys(self):
        node = _DerivedStaticNode("d", timestep=0.01, trainable=False)
        assert set(node.static_data_deps()) <= set(node.static_data)

    def test_a_wrapper_forwards_the_declaration_as_it_forwards_the_statics(self):
        """Both forward, so a wrapper's deps still key on its own statics.

        If only ``static_data`` forwarded, a wrapper would publish a
        static whose provenance had vanished -- the same blind spot the
        static-data hash had before wrappers proxied it.
        """
        inner = _DerivedStaticNode("d", timestep=0.01, trainable=False)
        wrapper = _Wrapper(inner)
        assert wrapper.static_data_deps() == inner.static_data_deps()
        assert set(wrapper.static_data_deps()) <= set(wrapper.static_data)

    def test_the_forwarded_declaration_composes_through_a_nested_pair(self):
        inner = _DerivedStaticNode("d", timestep=0.01, trainable=False)
        nested = HybridNode(_Wrapper(inner), lambda s, b, dd: {})
        assert nested.static_data_deps() == {"table": ("scale", "n")}

    def test_a_cycle_between_two_nodes_terminates(self):
        """The forwarding guard is the one ``static_data`` already uses."""

        class _Holder(SimulationNode):
            def __init__(self, name):
                super().__init__(name, timestep=0.01)
                self.other = None

            def initial_state(self):
                return {"x": jnp.array(0.0)}

            def update(self, state, boundary_inputs, dt):
                return {"x": state["x"] + dt}

        a, b = _Holder("a"), _Holder("b")
        a.other, b.other = b, a
        assert a.static_data_deps() == {}


class TestStaticDataDepsRefusedAtCompile:
    def test_a_non_trainable_dependency_compiles(self):
        """The legal case: the parameter is frozen, so nothing is lost."""
        gm = _compiled_graph(
            _DerivedStaticNode("d", timestep=0.01, trainable=False)
        )
        gm.step()

    def test_a_trainable_dependency_is_refused(self):
        gm = GraphManager()
        gm.add_node(_DerivedStaticNode("d", timestep=0.01, trainable=True))
        with pytest.raises(ValueError) as excinfo:
            gm.compile()
        message = str(excinfo.value)
        # The three things a reader needs to locate the problem ...
        assert "'d'" in message
        assert "'table'" in message
        assert "'scale'" in message
        # ... why it cannot be supported ...
        assert "constant" in message
        assert "differentiate" in message
        # ... and both ways out.
        assert "trainable=False" in message
        assert "update()" in message

    def test_a_structural_dependency_is_not_a_violation(self):
        """``n`` is an ``int``: nothing differentiates through it.

        ``_DerivedStaticNode`` declares ``n`` alongside ``scale`` on both
        sides of the test, so the passing case above proves a structural
        dependency is allowed and this pins the reason: it is absent from
        the parameter pytree, not merely absent from ``param_specs``.
        """
        node = _DerivedStaticNode("d", timestep=0.01, trainable=False)
        assert "n" in node.static_data_deps()["table"]
        assert "n" not in node.params_pytree()

    def test_an_undeclared_derivation_still_compiles(self):
        """A guard rail, not an inference engine.

        The same node without the declaration compiles happily -- which
        is the latent hazard D10 describes, and the reason the
        declaration has to be written by hand.
        """

        class _Undeclared(_DerivedStaticNode):
            def static_data_deps(self):
                return {}

        _compiled_graph(_Undeclared("d", timestep=0.01, trainable=True))

    def test_the_refusal_fires_through_a_wrapper(self):
        """A wrapper must not be able to hide an inner node's declaration."""
        inner = _DerivedStaticNode("d", timestep=0.01, trainable=True)
        gm = GraphManager()
        gm.add_node(_Wrapper(inner))
        with pytest.raises(ValueError, match="'scale'"):
            gm.compile()

    def test_the_refusal_names_the_wrapped_node_when_the_names_differ(self):
        """The graph knows the wrapper's name; the fix is in the inner node."""
        inner = _DerivedStaticNode("inner", timestep=0.01, trainable=True)
        gm = GraphManager()
        gm.add_node(HybridNode(inner, lambda s, b, d: {}, name="outer"))
        with pytest.raises(ValueError) as excinfo:
            gm.compile()
        assert "'outer'" in str(excinfo.value)
        assert "'inner'" in str(excinfo.value)

    def test_a_wrapper_over_a_frozen_dependency_still_compiles(self):
        inner = _DerivedStaticNode("d", timestep=0.01, trainable=False)
        _compiled_graph(HybridNode(inner, lambda s, b, dd: {}))

    def test_freezing_the_parameter_is_a_working_fix(self):
        """The message's first remedy has to actually work."""
        gm = GraphManager()
        gm.add_node(_DerivedStaticNode("d", timestep=0.01, trainable=True))
        with pytest.raises(ValueError):
            gm.compile()
        gm2 = GraphManager()
        gm2.add_node(_DerivedStaticNode("d", timestep=0.01, trainable=False))
        gm2.compile()


class TestHeatNodeStaticDataDeps:
    """``HeatNode`` is the only shipped node deriving a static from params.

    Its provenance differs by grid, and so does its declaration: the
    non-uniform grid reads ``grid_x`` and derives it from the frozen
    ``grid_points``, while the uniform grid derives it from the trainable
    ``length`` but never reads it -- ``_compute_laplacian`` recomputes
    ``dx`` from the traced parameter.  Declaring ``length`` there anyway
    would trip the refusal, which the last test here pins.
    """

    def test_the_uniform_grid_declares_nothing(self):
        from maddening.nodes.heat import HeatNode

        node = HeatNode("rod", timestep=0.01, n_cells=8, length=2.0)
        assert node.static_data_deps() == {}

    def test_the_non_uniform_grid_declares_grid_points(self):
        from maddening.nodes.heat import HeatNode

        node = HeatNode("rod", timestep=0.01, n_cells=4,
                        grid_points=[0.1, 0.2, 0.35, 0.8])
        assert node.static_data_deps() == {"grid_x": ("grid_points",)}

    def test_grid_points_is_frozen_so_the_declaration_is_legal(self):
        from maddening.nodes.heat import HeatNode

        node = HeatNode("rod", timestep=0.01, n_cells=4,
                        grid_points=[0.1, 0.2, 0.35, 0.8])
        assert node.param_specs()["grid_points"].trainable is False
        assert "grid_points" in node.params_pytree()   # it *is* a pytree leaf
        _compiled_graph(node).step()

    def test_the_uniform_grid_still_compiles_with_length_trainable(self):
        from maddening.nodes.heat import HeatNode

        node = HeatNode("rod", timestep=0.01, n_cells=8, length=2.0)
        assert node.param_specs()["length"].trainable is True
        _compiled_graph(node).step()

    def test_declaring_length_would_be_refused(self):
        """The guard rail is live for the exact case HeatNode avoids.

        If the uniform Laplacian ever started reading ``grid_x``, the
        honest declaration would name ``length`` -- and this is what
        ``compile()`` does about it.
        """
        from maddening.nodes.heat import HeatNode

        class _ReadsGridXOnTheUniformPath(HeatNode):
            def static_data_deps(self):
                return {"grid_x": ("grid_points", "length", "n_cells")}

        gm = GraphManager()
        gm.add_node(_ReadsGridXOnTheUniformPath("rod", timestep=0.01, n_cells=8))
        with pytest.raises(ValueError) as excinfo:
            gm.compile()
        assert "'length'" in str(excinfo.value)
        assert "'grid_x'" in str(excinfo.value)
