"""Tests for replace_node -- edge/external input preservation after swap."""

from dataclasses import fields, replace
from inspect import signature

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from maddening.core import edge as edge_module
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import ExternalInputSpec, GraphManager
from maddening.core.params import ParamSpec
from maddening.core.transforms import lbm_to_si_length
from maddening.nodes.ball import BallNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode
from maddening.surrogates.architecture import SurrogateArchitecture
from maddening.surrogates.node import SurrogateNode
from maddening.surrogates.replace import _core, replace_node


#: A 5x3 interface operator, so a dropped mapping cannot even broadcast.
_H = np.array(
    [[1.0, 0.0, 0.0],
     [0.5, 0.5, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.5, 0.5],
     [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def _mapped_graph():
    """A compiled graph whose one edge maps ``coarse.temperature`` (3) onto
    ``fine.heat_source`` (5).  Returns the graph and the edge key."""
    gm = GraphManager()
    gm.add_node(HeatNode("coarse", timestep=0.01, n_cells=3,
                         initial_temperature=jnp.array([1.0, 2.0, 3.0])))
    gm.add_node(HeatNode("fine", timestep=0.01, n_cells=5))
    gm.add_edge("coarse", "fine", "temperature", "heat_source",
                mapping=matrix_mapping(_H))
    gm.compile()
    return gm, gm._edges[0].key


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


# ---------------------------------------------------------------------------
# Edge attributes survive the swap (regression: every one of these was
# silently reset to its default by the positional re-add).
# ---------------------------------------------------------------------------


def _passthrough_surrogate(name, state, boundary_spec=None):
    """A surrogate that stands in for a node, holding ``state`` unchanged."""
    return SurrogateNode(
        name=name, timestep=0.01, architecture=PassthroughDirect(), weights={},
        state_spec={k: jnp.shape(v) for k, v in state.items()},
        boundary_spec=dict(boundary_spec or {}),
        initial_values=dict(state),
    )


class TestReplaceNodePreservesEdgeAttributes:
    """Each test asserts on the *values* the graph computes, not on the
    ``EdgeSpec`` fields: an attribute test would pass on an edge whose
    attribute is restored but whose effect is not, and a user only ever
    sees the numbers."""

    def test_additive_edge_still_adds_after_replace(self):
        """Two additive edges into one boundary input summed to 3.0 before
        the swap and, with ``additive`` reset to ``False``, to 1.0 after."""
        gm = GraphManager()
        gm.add_node(TableNode("t1", timestep=0.01, position=1.0))
        gm.add_node(TableNode("t2", timestep=0.01, position=2.0))
        gm.add_node(SpringDamperNode("s", timestep=0.01, initial_position=0.0))
        gm.add_edge("t1", "s", "position", "anchor_position", additive=True)
        gm.add_edge("t2", "s", "position", "anchor_position", additive=True)
        gm.compile()
        before = float(gm.resolve_boundary_inputs("s")["anchor_position"])
        assert before == pytest.approx(3.0)

        replace_node(gm, "t1", _passthrough_surrogate("t1", {"position": 1.0}))
        gm.compile()

        assert float(gm.resolve_boundary_inputs("s")["anchor_position"]) == pytest.approx(3.0)

    def test_transform_and_units_survive_replace(self):
        """The transform scales the value and the declared units are what
        ``validate()`` checks the target against; both must come back."""
        gm = GraphManager()
        gm.add_node(TableNode("t", timestep=0.01, position=4.0))
        gm.add_node(SpringDamperNode("s", timestep=0.01))
        # 4.0 lattice lengths -> 4.0 * dx metres, the scale the spring's
        # anchor_position is declared in.
        gm.add_edge(
            "t", "s", "position", "anchor_position",
            transform=lbm_to_si_length(dx_physical=0.25),
            source_units="lattice", target_units="m",
        )
        gm.compile()
        before = float(gm.resolve_boundary_inputs("s")["anchor_position"])
        assert before == pytest.approx(1.0)
        assert not [i for i in gm.validate() if "units" in i]

        replace_node(gm, "t", _passthrough_surrogate("t", {"position": 4.0}))
        gm.compile()

        assert float(gm.resolve_boundary_inputs("s")["anchor_position"]) == pytest.approx(1.0)
        edge = gm._edges[0]
        # The units are declarative -- they do not scale anything, they are
        # what validate() compares against the target's expected_units --
        # so their loss shows up as a diagnostic that stops firing.
        assert (edge.source_units, edge.target_units) == ("lattice", "m")
        gm._edges[0] = replace(edge, target_units="K")
        assert [i for i in gm.validate() if "units" in i]

    def test_mapped_edge_transfers_the_same_values_after_replace(self):
        """A dropped ``mapping`` removes the whole interface transfer."""
        gm, _ = _mapped_graph()
        before = np.asarray(gm.resolve_boundary_inputs("fine")["heat_source"])

        state = dict(gm._state["coarse"])
        replace_node(gm, "coarse", _passthrough_surrogate("coarse", state))
        gm.compile()

        after = np.asarray(gm.resolve_boundary_inputs("fine")["heat_source"])
        assert gm._edges[0].mapping is not None
        np.testing.assert_allclose(after, before, rtol=1e-6)

    def test_tuned_mapping_weights_survive_replace(self):
        """``remove_node`` drops ``params["mappings"][key]`` and the next
        compile re-snapshots it from the mapping object, which would undo
        weights a sysid fit had moved."""
        gm, key = _mapped_graph()
        slot = gm.params["mappings"][key]
        slot["H"] = (slot["H"] * 3.0).astype(slot["H"].dtype)
        before = np.asarray(gm.resolve_boundary_inputs("fine")["heat_source"])

        state = dict(gm._state["coarse"])
        replace_node(gm, "coarse", _passthrough_surrogate("coarse", state))
        gm.compile()

        after = np.asarray(gm.resolve_boundary_inputs("fine")["heat_source"])
        np.testing.assert_allclose(after, before, rtol=1e-6)

    def test_trainable_mapping_weights_stay_trainable_after_replace(self):
        gm, key = _mapped_graph()
        gm.set_param_spec(key, "H", ParamSpec(trainable=True))
        assert gm.trainable_mask()["mappings"][key]["H"] is True

        state = dict(gm._state["coarse"])
        replace_node(gm, "coarse", _passthrough_surrogate("coarse", state))
        gm.compile()

        assert gm.trainable_mask()["mappings"][key]["H"] is True

    def test_second_mapped_edge_on_a_field_pair_keeps_its_ordinal(self):
        """``ordinal`` -- and with it the ``params["mappings"]`` slot each
        edge's weights live in -- is recomputed by ``add_edge``; it must
        come out the same or the two edges swap weights."""
        gm = GraphManager()
        gm.add_node(HeatNode("coarse", timestep=0.01, n_cells=3,
                             initial_temperature=jnp.array([1.0, 2.0, 3.0])))
        gm.add_node(HeatNode("fine", timestep=0.01, n_cells=5))
        for scale in (1.0, 10.0):
            gm.add_edge("coarse", "fine", "temperature", "heat_source",
                        additive=True,
                        mapping=matrix_mapping(scale * _H))
        gm.compile()
        keys_before = [e.key for e in gm._edges]
        before = np.asarray(gm.resolve_boundary_inputs("fine")["heat_source"])
        assert keys_before[1].endswith("#1")

        state = dict(gm._state["coarse"])
        replace_node(gm, "coarse", _passthrough_surrogate("coarse", state))
        gm.compile()

        assert [e.key for e in gm._edges] == keys_before
        np.testing.assert_allclose(
            np.asarray(gm.resolve_boundary_inputs("fine")["heat_source"]),
            before, rtol=1e-6,
        )

    def test_external_input_dtype_survives_replace(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01))
        gm.add_external_input("ball", "force", shape=(2,), dtype=jnp.int32)
        gm.compile()

        replace_node(gm, "ball", _make_surrogate_ball())

        assert gm._external_inputs[0].dtype == jnp.int32
        assert gm._default_external_inputs()["ball"]["force"].dtype == jnp.int32


class TestEdgeSpecCannotSilentlyDropAField:
    """The durable guard.  Every re-add of a saved edge in the tree --
    ``replace_node`` and ``POST /surrogate/deactivate`` -- goes through
    ``EdgeSpec.add_edge_kwargs``, which derives the call from the
    dataclass's own fields, so a new field on ``EdgeSpec`` fails here
    instead of being reset to its default in silence."""

    def test_every_edge_spec_field_is_carried_across(self):
        carried = set(edge_module._ADD_EDGE_KWARGS)
        derived = set(edge_module._DERIVED_BY_ADD_EDGE)
        assert {f.name for f in fields(EdgeSpec)} == carried | derived
        assert not (carried & derived)

    def test_every_edge_keyword_is_an_add_edge_parameter(self):
        params = signature(GraphManager.add_edge).parameters
        assert set(edge_module._ADD_EDGE_KWARGS.values()) <= set(params)

    def test_add_edge_kwargs_round_trips_an_edge(self):
        """The kwargs really do rebuild the edge -- including ``ordinal``,
        which ``add_edge`` recomputes rather than being handed."""
        gm, _ = _mapped_graph()
        saved = gm._edges[0]
        gm.remove_edge("coarse", "fine", "temperature", "heat_source")
        gm.add_edge(**saved.add_edge_kwargs())
        assert gm._edges[0] == saved

    def test_every_external_input_field_is_carried_across(self):
        carried = set(_core.EXTERNAL_INPUT_FIELD_TO_ADD_KWARG)
        assert {f.name for f in fields(ExternalInputSpec)} == carried
        params = signature(GraphManager.add_external_input).parameters
        assert set(_core.EXTERNAL_INPUT_FIELD_TO_ADD_KWARG.values()) <= set(params)

    def test_an_uncarried_field_is_refused_by_name(self, monkeypatch):
        """Stands in for a future field: with ``additive`` off the table,
        building the kwargs must raise rather than return eight of nine."""
        table = dict(edge_module._ADD_EDGE_KWARGS)
        table.pop("additive")
        monkeypatch.setattr(edge_module, "_ADD_EDGE_KWARGS", table)

        with pytest.raises(RuntimeError, match="additive"):
            EdgeSpec("a", "b", "x", "y", additive=True).add_edge_kwargs()

    def test_an_uncarried_field_aborts_the_replacement(self, monkeypatch):
        """And ``replace_node`` raises before touching the graph, rather
        than leaving it half-rewired."""
        table = dict(edge_module._ADD_EDGE_KWARGS)
        table.pop("additive")
        monkeypatch.setattr(edge_module, "_ADD_EDGE_KWARGS", table)

        gm = GraphManager()
        gm.add_node(TableNode("t", timestep=0.01, position=1.0))
        gm.add_node(SpringDamperNode("s", timestep=0.01))
        gm.add_edge("t", "s", "position", "anchor_position", additive=True)
        gm.compile()

        with pytest.raises(RuntimeError, match="additive"):
            replace_node(gm, "t", _passthrough_surrogate("t", {"position": 1.0}))

        assert isinstance(gm._nodes["t"].node, TableNode)
        assert len(gm._edges) == 1 and gm._edges[0].additive is True
