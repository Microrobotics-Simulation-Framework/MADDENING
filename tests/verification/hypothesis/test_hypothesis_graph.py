"""Property-based tests for graph topology validation.

Tests that GraphManager correctly handles various graph topologies,
rejects invalid configurations, and does not crash on valid random graphs.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, assume, note
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.heat import HeatNode
from maddening.warnings import ExceptionGroup


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

node_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=12,
)

# Strategy for generating a random valid node
def _make_ball(name, dt=0.01):
    return BallNode(name=name, timestep=dt, initial_position=1.0, initial_velocity=0.0)


def _make_spring(name, dt=0.01):
    return SpringDamperNode(name=name, timestep=dt, stiffness=10.0, damping=0.5, mass=1.0)


def _make_heat(name, dt=0.001, n_cells=5):
    return HeatNode(name=name, timestep=dt, n_cells=n_cells, thermal_diffusivity=0.01)


# ---------------------------------------------------------------------------
# Test: random valid topologies do not crash
# ---------------------------------------------------------------------------

class TestRandomValidTopologies:
    """Random valid graph topologies should compile and step without error."""

    @given(
        n_nodes=st.integers(min_value=1, max_value=6),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(max_examples=100, deadline=None)
    def test_random_ball_chain_compiles_and_steps(self, n_nodes, seed):
        """Build a chain of BallNode objects and step without crash."""
        rng = np.random.default_rng(seed)
        gm = GraphManager()

        names = [f"ball_{i}" for i in range(n_nodes)]
        for name in names:
            gm.add_node(BallNode(
                name=name, timestep=0.01,
                initial_position=float(rng.uniform(0, 10)),
                initial_velocity=float(rng.uniform(-5, 5)),
            ))

        # Add edges between consecutive nodes (position -> table_position)
        for i in range(n_nodes - 1):
            gm.add_edge(names[i], names[i + 1], "position", "table_position")

        gm.compile()
        state = gm.step()

        # All outputs should be finite
        for name in names:
            assert jnp.isfinite(state[name]["position"]), (
                f"Non-finite position for {name}"
            )
            assert jnp.isfinite(state[name]["velocity"]), (
                f"Non-finite velocity for {name}"
            )

    @given(
        n_nodes=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(max_examples=80, deadline=None)
    def test_random_spring_ball_coupled_steps(self, n_nodes, seed):
        """Build a random mix of springs and balls, edge them, step."""
        rng = np.random.default_rng(seed)
        gm = GraphManager()

        names = []
        for i in range(n_nodes):
            if rng.random() < 0.5:
                node = BallNode(
                    name=f"n_{i}", timestep=0.01,
                    initial_position=float(rng.uniform(0, 10)),
                    initial_velocity=float(rng.uniform(-2, 2)),
                )
            else:
                node = SpringDamperNode(
                    name=f"n_{i}", timestep=0.01,
                    stiffness=float(rng.uniform(1, 100)),
                    damping=float(rng.uniform(0.1, 5)),
                    mass=float(rng.uniform(0.1, 10)),
                    initial_position=float(rng.uniform(-5, 5)),
                )
            gm.add_node(node)
            names.append(f"n_{i}")

        # Add a random subset of valid edges
        for i in range(n_nodes - 1):
            src_name = names[i]
            tgt_name = names[i + 1]
            # Check what edge is valid: position -> table_position (ball)
            # or position -> anchor_position (spring)
            tgt_node = gm._nodes[tgt_name].node
            if isinstance(tgt_node, BallNode):
                gm.add_edge(src_name, tgt_name, "position", "table_position")
            else:
                gm.add_edge(src_name, tgt_name, "position", "anchor_position")

        gm.compile()
        state = gm.step()

        for name in names:
            assert jnp.isfinite(state[name]["position"])
            assert jnp.isfinite(state[name]["velocity"])

    @settings(max_examples=30, deadline=None)
    @given(seed=st.integers(min_value=0, max_value=2**31))
    def test_single_node_graph_steps(self, seed):
        """A single isolated node should compile and step cleanly."""
        rng = np.random.default_rng(seed)
        gm = GraphManager()
        gm.add_node(BallNode(
            name="solo", timestep=0.01,
            initial_position=float(rng.uniform(-100, 100)),
            initial_velocity=float(rng.uniform(-50, 50)),
        ))
        gm.compile()
        state = gm.step()
        assert "solo" in state
        assert jnp.isfinite(state["solo"]["position"])

    @settings(max_examples=20, deadline=None)
    @given(n_steps=st.integers(min_value=1, max_value=50))
    def test_empty_boundary_inputs_multi_step(self, n_steps):
        """Multiple steps with no boundary inputs should not crash."""
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01, initial_position=10.0))
        gm.compile()
        for _ in range(n_steps):
            state = gm.step()
        assert jnp.isfinite(state["b"]["position"])


# ---------------------------------------------------------------------------
# Test: self-edges should be rejected or handled gracefully
# ---------------------------------------------------------------------------

class TestSelfEdgeHandling:
    """Self-edges (source==target same node) should not cause undefined behavior."""

    def test_self_edge_ball_position_to_table(self):
        """A ball whose position feeds back to its own table_position."""
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01, initial_position=5.0))
        # This is technically a self-loop; it should either be rejected
        # or produce a valid (non-crashing) result
        gm.add_edge("b", "b", "position", "table_position")
        # Either compile raises or step works without crash
        try:
            gm.compile()
            state = gm.step()
            # If it doesn't raise, output must still be finite
            assert jnp.isfinite(state["b"]["position"])
            assert jnp.isfinite(state["b"]["velocity"])
        except (RuntimeError, ValueError):
            pass  # rejection is also acceptable

    def test_self_edge_spring_position_to_anchor(self):
        """A spring whose position feeds back to its own anchor."""
        gm = GraphManager()
        gm.add_node(SpringDamperNode(
            name="s", timestep=0.01, stiffness=10.0, damping=1.0,
            initial_position=2.0,
        ))
        gm.add_edge("s", "s", "position", "anchor_position")
        try:
            gm.compile()
            state = gm.step()
            # Self-referencing spring: force depends on (pos - pos - rest_length)
            # = -rest_length, which is constant. Should be finite.
            assert jnp.isfinite(state["s"]["position"])
            assert jnp.isfinite(state["s"]["velocity"])
        except (RuntimeError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Test: duplicate node names are rejected
# ---------------------------------------------------------------------------

class TestDuplicateNodeNames:
    """Adding a node with an existing name should raise ValueError."""

    def test_duplicate_name_raises(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="dup", timestep=0.01))
        with pytest.raises(ValueError, match="already exists"):
            gm.add_node(BallNode(name="dup", timestep=0.02))

    @given(name=node_name_st)
    @settings(max_examples=50, deadline=None)
    def test_duplicate_name_raises_any_name(self, name):
        """Any name used twice should be rejected."""
        gm = GraphManager()
        gm.add_node(BallNode(name=name, timestep=0.01))
        with pytest.raises(ValueError, match="already exists"):
            gm.add_node(SpringDamperNode(name=name, timestep=0.01))


# ---------------------------------------------------------------------------
# Test: missing edge targets raise at compile time
# ---------------------------------------------------------------------------

class TestMissingEdgeTargets:
    """Edges referencing non-existent nodes should raise on compile."""

    def test_missing_source_node_raises(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("ghost", "b", "position", "table_position")
        with pytest.raises(RuntimeError, match="non-existent source"):
            gm.compile()

    def test_missing_target_node_raises(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("b", "ghost", "position", "table_position")
        with pytest.raises(RuntimeError, match="non-existent.*target"):
            gm.compile()

    def test_missing_source_field_raises(self):
        """An edge referencing a non-existent source field should error at compile."""
        gm = GraphManager()
        gm.add_node(BallNode(name="a", timestep=0.01))
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("a", "b", "nonexistent_field", "table_position")
        with pytest.raises(RuntimeError, match="source field"):
            gm.compile()

    @given(ghost_name=node_name_st)
    @settings(max_examples=30, deadline=None)
    def test_missing_target_any_name(self, ghost_name):
        """Any non-existent target should raise."""
        assume(ghost_name != "b")
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("b", ghost_name, "position", "table_position")
        with pytest.raises(RuntimeError):
            gm.compile()


# ---------------------------------------------------------------------------
# Test: shape/dtype mismatches raise ExceptionGroup at compile
# ---------------------------------------------------------------------------

class TestShapeDtypeMismatch:
    """Shape or dtype mismatches on edges with BoundaryInputSpec should raise."""

    def test_shape_mismatch_raises_exception_group(self):
        """Connecting a vector field to a scalar boundary input should raise."""
        gm = GraphManager()
        # HeatNode has temperature of shape (n_cells,) but BallNode expects
        # table_position of shape ()
        gm.add_node(HeatNode(name="h", timestep=0.001, n_cells=5))
        gm.add_node(BallNode(name="b", timestep=0.01))
        # temperature is shape (5,), table_position expects ()
        gm.add_edge("h", "b", "temperature", "table_position")
        with pytest.raises(ExceptionGroup):
            gm.compile()

    def test_compatible_shapes_compile_ok(self):
        """Scalar-to-scalar edges with matching types should compile fine."""
        gm = GraphManager()
        gm.add_node(BallNode(name="a", timestep=0.01, initial_position=5.0))
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("a", "b", "position", "table_position")
        # Should not raise
        gm.compile()
        state = gm.step()
        assert jnp.isfinite(state["b"]["position"])


# ---------------------------------------------------------------------------
# Test: removing nodes cleans up edges
# ---------------------------------------------------------------------------

class TestNodeRemoval:
    """Removing a node should also remove all connected edges."""

    def test_remove_node_removes_edges(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="a", timestep=0.01))
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("a", "b", "position", "table_position")
        gm.remove_node("a")
        # Edge should be gone; compiling with just b should work
        gm.compile()
        state = gm.step()
        assert "b" in state

    def test_remove_nonexistent_node_raises(self):
        gm = GraphManager()
        with pytest.raises(KeyError):
            gm.remove_node("ghost")


# ---------------------------------------------------------------------------
# Test: cyclic graphs compile with staggering
# ---------------------------------------------------------------------------

class TestCyclicGraphs:
    """Cycles in the graph should be handled (staggering or coupling)."""

    @given(n_nodes=st.integers(min_value=2, max_value=4))
    @settings(max_examples=20, deadline=None)
    def test_cycle_compiles_with_staggering(self, n_nodes):
        """A simple cycle should compile (back-edges use prev timestep)."""
        gm = GraphManager()
        names = [f"s_{i}" for i in range(n_nodes)]
        for name in names:
            gm.add_node(SpringDamperNode(name=name, timestep=0.01))

        # Create a cycle: s_0 -> s_1 -> ... -> s_{n-1} -> s_0
        for i in range(n_nodes):
            src = names[i]
            tgt = names[(i + 1) % n_nodes]
            gm.add_edge(src, tgt, "position", "anchor_position")

        # Should compile without error (cycle handled by staggering)
        gm.compile()
        state = gm.step()
        for name in names:
            assert jnp.isfinite(state[name]["position"])

    def test_bidirectional_edge_pair(self):
        """Two nodes with edges in both directions form a cycle."""
        gm = GraphManager()
        gm.add_node(SpringDamperNode(name="a", timestep=0.01))
        gm.add_node(SpringDamperNode(name="b", timestep=0.01))
        gm.add_edge("a", "b", "position", "anchor_position")
        gm.add_edge("b", "a", "position", "anchor_position")
        gm.compile()
        state = gm.step()
        assert jnp.isfinite(state["a"]["position"])
        assert jnp.isfinite(state["b"]["position"])
