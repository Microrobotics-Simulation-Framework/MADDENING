"""Tests for replace_node -- edge/external input preservation after swap."""

import pytest
import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.surrogates.architecture import SurrogateArchitecture
from maddening.surrogates.node import SurrogateNode
from maddening.surrogates.replace import replace_node


class ConstantDirect(SurrogateArchitecture):
    """Returns constant state for testing."""
    mode = "direct"

    def init_params(self, rng_key, state_spec, boundary_spec):
        return {}

    def forward(self, params, state, boundary_inputs, dt):
        return {k: jnp.zeros_like(v) for k, v in state.items()}


class PassthroughDirect(SurrogateArchitecture):
    """Returns state unchanged -- useful for testing wiring."""
    mode = "direct"

    def init_params(self, rng_key, state_spec, boundary_spec):
        return {}

    def forward(self, params, state, boundary_inputs, dt):
        return {k: v for k, v in state.items()}


def _make_surrogate_ball(arch=None):
    if arch is None:
        arch = ConstantDirect()
    return SurrogateNode(
        name="ball", timestep=0.01, architecture=arch,
        weights={},
        state_spec={"position": (), "velocity": ()},
        boundary_spec={"table_position": ()},
        initial_values={"position": 10.0, "velocity": 0.0},
    )


class TestReplaceNode:
    def test_basic_replace(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
        gm.compile()

        surrogate = _make_surrogate_ball()
        replace_node(gm, "ball", surrogate)

        assert "ball" in gm.node_names
        assert isinstance(gm._nodes["ball"].node, SurrogateNode)

    def test_preserves_edges(self):
        gm = GraphManager()
        gm.add_node(TableNode("table", timestep=0.01))
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=5.0))
        gm.add_edge("table", "ball", "position", "table_position")
        gm.compile()

        surrogate = _make_surrogate_ball()
        replace_node(gm, "ball", surrogate)

        # Edge should be preserved
        assert len(gm._edges) == 1
        edge = gm._edges[0]
        assert edge.source_node == "table"
        assert edge.target_node == "ball"
        assert edge.source_field == "position"
        assert edge.target_field == "table_position"

    def test_preserves_external_inputs(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01))
        gm.add_external_input("ball", "force", shape=())
        gm.compile()

        surrogate = _make_surrogate_ball()
        replace_node(gm, "ball", surrogate)

        assert len(gm._external_inputs) == 1
        ei = gm._external_inputs[0]
        assert ei.target_node == "ball"
        assert ei.target_field == "force"

    def test_graph_runs_after_replace(self):
        gm = GraphManager()
        gm.add_node(TableNode("table", timestep=0.01))
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=5.0))
        gm.add_edge("table", "ball", "position", "table_position")
        gm.compile()

        surrogate = _make_surrogate_ball()
        replace_node(gm, "ball", surrogate)
        gm.compile()

        state = gm.step()
        assert "ball" in state
        assert "table" in state
        assert jnp.isfinite(state["ball"]["position"])

    def test_scan_after_replace(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
        gm.compile()

        arch = PassthroughDirect()
        surrogate = SurrogateNode(
            name="ball", timestep=0.01, architecture=arch,
            weights={},
            state_spec={"position": (), "velocity": ()},
            boundary_spec={},
            initial_values={"position": 10.0, "velocity": 0.0},
        )
        replace_node(gm, "ball", surrogate)
        gm.compile()

        final = gm.run_scan(50)
        # Passthrough should keep state unchanged
        assert float(final["ball"]["position"]) == pytest.approx(10.0)

    def test_name_mismatch_raises(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01))

        bad_surrogate = SurrogateNode(
            name="wrong_name", timestep=0.01,
            architecture=ConstantDirect(), weights={},
            state_spec={"x": ()}, boundary_spec={},
            initial_values={"x": 0.0},
        )
        with pytest.raises(ValueError, match="must match"):
            replace_node(gm, "ball", bad_surrogate)

    def test_missing_node_raises(self):
        gm = GraphManager()
        surrogate = _make_surrogate_ball()
        with pytest.raises(KeyError, match="No node"):
            replace_node(gm, "ball", surrogate)

    def test_preserves_outgoing_edges(self):
        """Edges where the replaced node is the SOURCE should also be preserved."""
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=5.0))
        gm.add_node(TableNode("table", timestep=0.01))
        # Ball position feeds into table as an external-like edge
        gm.add_edge("ball", "table", "position", "ball_height")
        gm.compile()

        surrogate = SurrogateNode(
            name="ball", timestep=0.01, architecture=PassthroughDirect(),
            weights={},
            state_spec={"position": (), "velocity": ()},
            boundary_spec={},
            initial_values={"position": 5.0, "velocity": 0.0},
        )
        replace_node(gm, "ball", surrogate)

        assert len(gm._edges) == 1
        assert gm._edges[0].source_node == "ball"
