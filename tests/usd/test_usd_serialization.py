"""
Tests for USD graph serialization (Phase 7).

Tests save_graph_to_usd and load_graph_from_usd round-trip,
including edge transforms, coupling groups, and external inputs.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from pxr import Usd, Sdf, Vt

import maddening.usd  # noqa: F401 (schema registration)
from maddening.usd.serialization import (
    save_graph_to_usd,
    load_graph_from_usd,
    register_node_class,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.transforms import register_transform, get_transform_name
from maddening.nodes.heat import HeatNode
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode


@register_transform("test_negate_for_usd", "Negate for USD test")
def _negate_for_usd(x):
    return -x


class TestSaveGraphToUSD:
    """Test serialization of GraphManager to USD."""

    def test_save_simple_graph(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", 0.01, initial_position=5.0))
        gm.add_node(TableNode("table", 0.01, position=0.0))
        gm.add_edge("table", "ball", "position", "table_position")
        gm.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm, stage)

        # Check root
        root = stage.GetPrimAtPath("/Simulation")
        assert root.IsValid()
        assert root.GetTypeName() == "MaddeningSimulationGraph"

        # Check nodes
        ball_prim = stage.GetPrimAtPath("/Simulation/nodes/ball")
        assert ball_prim.IsValid()
        assert "BallNode" in ball_prim.GetAttribute("maddening:nodeType").Get()

        table_prim = stage.GetPrimAtPath("/Simulation/nodes/table")
        assert table_prim.IsValid()

        # Check edge
        edge_prim = stage.GetPrimAtPath("/Simulation/edges/e0")
        assert edge_prim.IsValid()
        assert edge_prim.GetAttribute("maddening:sourceNode").Get() == "table"
        assert edge_prim.GetAttribute("maddening:targetNode").Get() == "ball"
        assert edge_prim.GetAttribute("maddening:sourceField").Get() == "position"
        assert edge_prim.GetAttribute("maddening:targetField").Get() == "table_position"

    def test_save_with_registered_transform(self):
        gm = GraphManager()
        gm.add_node(HeatNode("rod_a", 0.01, n_cells=5))
        gm.add_node(HeatNode("rod_b", 0.01, n_cells=5))
        gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                     transform="extract_last")
        gm.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm, stage)

        edge = stage.GetPrimAtPath("/Simulation/edges/e0")
        assert edge.GetAttribute("maddening:transformName").Get() == "extract_last"

    def test_save_unregistered_transform_raises(self):
        gm = GraphManager()
        gm.add_node(HeatNode("rod_a", 0.01, n_cells=5))
        gm.add_node(HeatNode("rod_b", 0.01, n_cells=5))
        gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                     transform=lambda T: T[-1])  # unregistered lambda
        gm.compile()

        stage = Usd.Stage.CreateInMemory()
        from maddening.core.transforms import UnregisteredTransformError
        with pytest.raises(UnregisteredTransformError):
            save_graph_to_usd(gm, stage)

    def test_save_coupling_group(self):
        gm = GraphManager()
        gm.add_node(HeatNode("rod_a", 0.01, n_cells=5))
        gm.add_node(HeatNode("rod_b", 0.01, n_cells=5))
        gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                     transform="extract_last")
        gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                     transform="extract_first")
        gm.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=15,
            tolerance=1e-7,
            acceleration="aitken",
            diagnostics=True,
        )
        gm.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm, stage)

        cg = stage.GetPrimAtPath("/Simulation/coupling_groups/cg0")
        assert cg.IsValid()
        nodes = sorted(cg.GetAttribute("maddening:nodes").Get())
        assert nodes == ["rod_a", "rod_b"]
        assert cg.GetAttribute("maddening:maxIterations").Get() == 15
        assert abs(cg.GetAttribute("maddening:tolerance").Get() - 1e-7) < 1e-15
        assert cg.GetAttribute("maddening:acceleration").Get() == "aitken"
        assert cg.GetAttribute("maddening:diagnostics").Get() is True

    def test_save_external_input(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", 0.01, initial_position=5.0))
        gm.add_external_input("ball", "external_force", shape=(3,))
        gm.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm, stage)

        ext = stage.GetPrimAtPath("/Simulation/external_inputs/ext0")
        assert ext.IsValid()
        assert ext.GetAttribute("maddening:targetNode").Get() == "ball"
        assert ext.GetAttribute("maddening:targetField").Get() == "external_force"
        assert list(ext.GetAttribute("maddening:shape").Get()) == [3]

    def test_save_additive_edge(self):
        gm = GraphManager()
        gm.add_node(HeatNode("rod_a", 0.01, n_cells=5))
        gm.add_node(HeatNode("rod_b", 0.01, n_cells=5))
        gm.add_edge("rod_a", "rod_b", "temperature", "heat_source",
                     transform="extract_last", additive=True)
        gm.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm, stage)

        edge = stage.GetPrimAtPath("/Simulation/edges/e0")
        assert edge.GetAttribute("maddening:additive").Get() is True


class TestLoadGraphFromUSD:
    """Test deserialization of GraphManager from USD."""

    def test_round_trip_simple(self):
        """Save and reload a simple graph, verify structure matches."""
        gm1 = GraphManager()
        gm1.add_node(BallNode("ball", 0.01, initial_position=5.0,
                               initial_velocity=0.0, elasticity=0.7))
        gm1.add_node(TableNode("table", 0.01, position=0.0))
        gm1.add_edge("table", "ball", "position", "table_position")
        gm1.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)

        gm2 = load_graph_from_usd(stage)

        assert set(gm2._nodes.keys()) == {"ball", "table"}
        assert len(gm2._edges) == 1
        assert gm2._edges[0].source_node == "table"
        assert gm2._edges[0].target_node == "ball"

    def test_round_trip_heat_nodes(self):
        """Save and reload HeatNodes, verify params preserved."""
        gm1 = GraphManager()
        gm1.add_node(HeatNode("rod", 0.005, n_cells=20, length=2.0,
                               thermal_diffusivity=0.05))
        gm1.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)

        gm2 = load_graph_from_usd(stage)
        rod = gm2._nodes["rod"].node
        assert isinstance(rod, HeatNode)
        assert rod.params["n_cells"] == 20
        assert rod.params["length"] == 2.0
        assert abs(rod.params["thermal_diffusivity"] - 0.05) < 1e-10

    def test_round_trip_with_transform(self):
        """Save and reload edges with registered transforms."""
        gm1 = GraphManager()
        gm1.add_node(HeatNode("rod_a", 0.01, n_cells=5))
        gm1.add_node(HeatNode("rod_b", 0.01, n_cells=5))
        gm1.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                      transform="extract_last")
        gm1.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)

        gm2 = load_graph_from_usd(stage)
        assert len(gm2._edges) == 1
        edge = gm2._edges[0]
        assert edge.transform is not None
        # Verify the transform works
        arr = jnp.array([1.0, 2.0, 3.0])
        assert float(edge.transform(arr)) == 3.0

    # ``tolerance`` is dead under ``convergence_norm="mixed"``; both are
    # set here to prove the stage carries them, not to configure a solve.
    @pytest.mark.filterwarnings("ignore:CouplingGroup.tolerance:UserWarning")
    def test_round_trip_coupling_group(self):
        """Save and reload coupling groups."""
        gm1 = GraphManager()
        gm1.add_node(HeatNode("rod_a", 0.01, n_cells=5))
        gm1.add_node(HeatNode("rod_b", 0.01, n_cells=5))
        gm1.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                      transform="extract_last")
        gm1.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                      transform="extract_first")
        gm1.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=25,
            tolerance=1e-9,
            acceleration="aitken",
            diagnostics=True,
            convergence_norm="mixed",
        )
        gm1.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)

        gm2 = load_graph_from_usd(stage)
        assert len(gm2._coupling_groups) == 1
        cg = gm2._coupling_groups[0]
        assert cg.nodes == frozenset(["rod_a", "rod_b"])
        assert cg.max_iterations == 25
        assert abs(cg.tolerance - 1e-9) < 1e-15
        assert cg.acceleration == "aitken"
        assert cg.diagnostics is True
        assert cg.convergence_norm == "mixed"

    def test_round_trip_external_input(self):
        """Save and reload external inputs."""
        gm1 = GraphManager()
        gm1.add_node(BallNode("ball", 0.01, initial_position=5.0))
        gm1.add_external_input("ball", "force", shape=(3,))
        gm1.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)

        gm2 = load_graph_from_usd(stage)
        assert len(gm2._external_inputs) == 1
        ext = gm2._external_inputs[0]
        assert ext.target_node == "ball"
        assert ext.target_field == "force"
        assert ext.shape == (3,)

    def test_round_trip_file_persistence(self, tmp_path):
        """Full round-trip through a .usda file."""
        filepath = str(tmp_path / "graph.usda")

        gm1 = GraphManager()
        gm1.add_node(HeatNode("rod_a", 0.01, n_cells=8))
        gm1.add_node(HeatNode("rod_b", 0.01, n_cells=8))
        gm1.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                      transform="extract_last")
        gm1.compile()

        # Save to file
        stage = Usd.Stage.CreateNew(filepath)
        save_graph_to_usd(gm1, stage)
        stage.Save()

        # Load from file
        stage2 = Usd.Stage.Open(filepath)
        gm2 = load_graph_from_usd(stage2)

        assert set(gm2._nodes.keys()) == {"rod_a", "rod_b"}
        assert len(gm2._edges) == 1

        # Verify the loaded graph can compile and step
        gm2.compile()
        state = gm2.step()
        assert "rod_a" in state
        assert "rod_b" in state

    def test_round_trip_additive_edge(self):
        """Save and reload additive edges."""
        gm1 = GraphManager()
        gm1.add_node(HeatNode("rod_a", 0.01, n_cells=5))
        gm1.add_node(HeatNode("rod_b", 0.01, n_cells=5))
        gm1.add_edge("rod_a", "rod_b", "temperature", "heat_source",
                      transform="identity", additive=True)
        gm1.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)

        gm2 = load_graph_from_usd(stage)
        assert gm2._edges[0].additive is True

    def test_load_nonexistent_path_raises(self):
        stage = Usd.Stage.CreateInMemory()
        with pytest.raises(ValueError, match="No prim"):
            load_graph_from_usd(stage, root_path="/NonExistent")

    def test_round_trip_multirate(self):
        """Save and reload a multi-rate graph."""
        gm1 = GraphManager()
        gm1.add_node(HeatNode("fast", 0.001, n_cells=5))
        gm1.add_node(HeatNode("slow", 0.01, n_cells=5))
        gm1.add_edge("fast", "slow", "temperature", "left_temperature",
                      transform="extract_last")
        gm1.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)

        root = stage.GetPrimAtPath("/Simulation")
        assert root.GetAttribute("maddening:isMultirate").Get() is True

        gm2 = load_graph_from_usd(stage)
        fast_node = gm2._nodes["fast"].node
        slow_node = gm2._nodes["slow"].node
        assert abs(fast_node.delta_t - 0.001) < 1e-10
        assert abs(slow_node.delta_t - 0.01) < 1e-10


#: ``NON_DEFAULT`` is deliberately an inconsistent configuration, and has
#: to be: no group a user would write can hold every field away from its
#: default at once, because ``convergence_norm``, ``acceleration`` and
#: ``solver`` each read one of two knobs and ignore the other.
#: ``CouplingGroup`` warns about every knob its configuration ignores,
#: which is the point of the warning and exactly wrong for a fixture
#: whose job is to differ from the default in all eighteen fields.  The
#: opt-out names the five, and only the tests built from the fixture
#: carry it: a *sixth* inert knob is a real finding and still fails.
inert_knobs_are_the_point = pytest.mark.filterwarnings(
    "ignore:CouplingGroup.tolerance:UserWarning",
    "ignore:CouplingGroup.relaxation:UserWarning",
    "ignore:CouplingGroup.jacobian_reuse:UserWarning",
    "ignore:CouplingGroup.linear_solver:UserWarning",
    "ignore:CouplingGroup.strict_convergence:UserWarning",
)


class TestCouplingGroupFields:
    """A coupling group on a stage carries the same nineteen fields as the
    config does.

    The stage used to store eleven of them, so a graph saved with an IQN
    acceleration, a jacobian-reuse window and a strict-convergence
    contract came back as a plain fixed-point iteration -- the same
    silent reconfiguration the config's missing ``coupling_groups`` key
    caused, with more of the settings surviving.
    """

    #: Everything but ``nodes``, each different from the dataclass default.
    NON_DEFAULT = {
        "max_iterations": 17,
        "tolerance": 3e-7,
        "convergence_norm": "mixed",
        "atol": 2e-8,
        "rtol": 3e-6,
        "diagnostics": True,
        "acceleration": "iqn-ils",
        "relaxation": 0.75,
        "iteration_mode": "jacobi",
        "accelerated_fields": {"rod_a": ("temperature",), "rod_b": ("temperature",)},
        "subcycling": True,
        "boundary_interpolation": "quadratic",
        "jacobian_reuse": 3,
        "waveform_iterations": 2,
        "predictor": "quadratic",
        # Deprecated but still legal, and a stage that loses it silently
        # changes which solver the group runs.
        "solver": "fori",
        "strict_convergence": True,
        "linear_solver": "dense",
    }

    #: The attributes this branch added -- what an older stage does *not*
    #: author, and what the backward-compatibility test clears.
    ADDED_ATTRS = (
        "maddening:atol",
        "maddening:rtol",
        "maddening:acceleratedFieldsJson",
        "maddening:jacobianReuse",
        "maddening:waveformIterations",
        "maddening:solver",
        "maddening:strictConvergence",
        "maddening:linearSolver",
    )

    @staticmethod
    def _two_rods():
        gm = GraphManager()
        gm.add_node(HeatNode("rod_a", 0.01, n_cells=5))
        gm.add_node(HeatNode("rod_b", 0.01, n_cells=5))
        gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                    transform="extract_last")
        gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                    transform="extract_first")
        return gm

    def test_the_attribute_table_covers_every_field_of_the_group(self):
        """``_COUPLING_GROUP_ATTRS`` plus the two fields USD cannot express
        as a scalar *is* the field set.  A field added to ``CouplingGroup``
        fails here until the stage carries it too."""
        from dataclasses import fields

        from maddening.core.coupling.group import CouplingGroup
        from maddening.usd.serialization import _COUPLING_GROUP_ATTRS

        covered = {field for _attr, field, _convert in _COUPLING_GROUP_ATTRS}
        covered |= {"nodes", "accelerated_fields"}
        assert covered == {f.name for f in fields(CouplingGroup)}

    @inert_knobs_are_the_point
    def test_every_field_round_trips_through_a_stage(self):
        from dataclasses import fields

        from maddening.core.coupling.group import CouplingGroup

        gm1 = self._two_rods()
        gm1.add_coupling_group(["rod_a", "rod_b"], **self.NON_DEFAULT)

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)
        gm2 = load_graph_from_usd(stage)

        before, after = gm1._coupling_groups[0], gm2._coupling_groups[0]
        for f in fields(CouplingGroup):
            assert getattr(after, f.name) == getattr(before, f.name), f.name
        # Non-default in every field, or the comparison proves nothing.
        default = CouplingGroup(nodes=before.nodes)
        for name, value in self.NON_DEFAULT.items():
            assert getattr(default, name) != value, name

    @inert_knobs_are_the_point
    def test_the_stage_and_the_config_describe_the_same_group(self):
        """Two spellings of one graph: a field that only one of them keeps
        is a field the other silently drops."""
        gm1 = self._two_rods()
        gm1.add_coupling_group(["rod_a", "rod_b"], **self.NON_DEFAULT)

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)
        from_usd = load_graph_from_usd(stage)
        from_config = GraphManager.from_dict(gm1.to_dict(), {"HeatNode": HeatNode})

        assert (from_usd._coupling_groups[0].to_dict()
                == from_config._coupling_groups[0].to_dict())

    def test_two_groups_keep_their_own_settings(self):
        gm1 = GraphManager()
        for name in ("a1", "a2", "b1", "b2"):
            gm1.add_node(HeatNode(name, 0.01, n_cells=4))
        # Each group sets only knobs its own acceleration and solver read
        # -- ``jacobian_reuse`` under ``"fixed"`` was a dead setting, and
        # the point here is that two *live* configurations stay apart.
        gm1.add_coupling_group(["a1", "a2"], max_iterations=4,
                               acceleration="fixed", relaxation=0.4)
        gm1.add_coupling_group(["b1", "b2"], max_iterations=30,
                               acceleration="iqn-imvj", jacobian_reuse=2,
                               linear_solver="dense")

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)
        gm2 = load_graph_from_usd(stage)

        by_nodes = {g.nodes: g for g in gm2._coupling_groups}
        assert by_nodes[frozenset({"a1", "a2"})] == gm1._coupling_groups[0]
        assert by_nodes[frozenset({"b1", "b2"})] == gm1._coupling_groups[1]

    def test_a_stage_written_before_these_attributes_still_loads(self, tmp_path):
        """Backward compatibility against a real file: a stage authoring
        only the attributes the previous writer emitted loads with the new
        fields at their defaults, i.e. exactly as it did before."""
        from dataclasses import fields

        from maddening.core.coupling.group import CouplingGroup

        path = tmp_path / "old_stage.usda"
        gm1 = self._two_rods()
        # ``acceleration="fixed"`` rather than ``"aitken"``: ``relaxation``
        # has to be authored on the stage for this test to prove the old
        # attributes still load, and only ``"fixed"`` reads it.
        gm1.add_coupling_group(["rod_a", "rod_b"], max_iterations=25,
                               convergence_norm="mixed", acceleration="fixed",
                               relaxation=0.8, iteration_mode="jacobi",
                               subcycling=True, boundary_interpolation="quadratic",
                               diagnostics=True, predictor="linear")

        stage = Usd.Stage.CreateNew(str(path))
        save_graph_to_usd(gm1, stage)
        cg_prim = stage.GetPrimAtPath("/Simulation/coupling_groups/cg0")
        for attr_name in self.ADDED_ATTRS:
            # RemoveProperty, not Clear: an older stage has no opinion at
            # all about these, not an opinion-less declaration of them.
            assert cg_prim.RemoveProperty(attr_name), attr_name
        stage.Save()

        text = path.read_text()
        for attr_name in self.ADDED_ATTRS:
            assert attr_name not in text, f"{attr_name} is still authored"

        gm2 = load_graph_from_usd(Usd.Stage.Open(str(path)))

        group = gm2._coupling_groups[0]
        assert group.max_iterations == 25
        assert group.acceleration == "fixed"
        assert group.relaxation == pytest.approx(0.8)
        assert group.iteration_mode == "jacobi"
        assert group.boundary_interpolation == "quadratic"
        assert group.predictor == "linear"
        default = CouplingGroup(nodes=group.nodes)
        added_fields = ("atol", "rtol", "accelerated_fields", "jacobian_reuse",
                        "waveform_iterations", "solver", "strict_convergence",
                        "linear_solver")
        assert len(added_fields) == len(self.ADDED_ATTRS)
        for name in added_fields:
            assert getattr(group, name) == getattr(default, name), name
        assert {f.name for f in fields(CouplingGroup)} >= set(added_fields)

    def test_a_hand_edited_enum_on_the_stage_names_the_prim(self):
        """A `.usda` is as hand-editable as a config, and the constructor's
        complaint names the field but not where it came from.  The reader
        names the prim, as the config loader names the group index."""
        gm1 = self._two_rods()
        gm1.add_coupling_group(["rod_a", "rod_b"])
        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)
        prim = stage.GetPrimAtPath("/Simulation/coupling_groups/cg0")
        prim.GetAttribute("maddening:acceleration").Set("aitkin")

        with pytest.raises(ValueError) as exc:
            load_graph_from_usd(stage)

        msg = str(exc.value)
        assert "/Simulation/coupling_groups/cg0" in msg, msg
        assert "acceleration" in msg and "'aitkin'" in msg, msg
        assert "expected one of" in msg, msg

    def test_a_group_naming_a_node_the_stage_does_not_have_is_rejected(self):
        gm1 = self._two_rods()
        gm1.add_coupling_group(["rod_a", "rod_b"])
        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)
        prim = stage.GetPrimAtPath("/Simulation/coupling_groups/cg0")
        prim.GetAttribute("maddening:nodes").Set(Vt.StringArray(["rod_a", "ghost"]))

        with pytest.raises(ValueError, match="ghost"):
            load_graph_from_usd(stage)
