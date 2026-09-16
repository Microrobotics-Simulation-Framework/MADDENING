"""Edge interface mappings round-trip through USD as their MappingSpec
(``maddening:mappingSpecJson``): point references, never weights; asset
references resolve relative to the stage file."""

import json

import numpy as np
import pytest
from pxr import Usd

from maddening.core.coupling.mapping import matrix_mapping, rbf_mapping
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.usd.serialization import load_graph_from_usd, save_graph_to_usd

C2F = "coarse.temperature->fine.heat_source"


def _grid(gm, name):
    return gm._nodes[name].node.static_data["grid_x"].value


def _rods(mapping_factory):
    gm = GraphManager()
    gm.add_node(HeatNode("coarse", 1e-4, n_cells=6, thermal_diffusivity=0.1))
    gm.add_node(HeatNode("fine", 1e-4, n_cells=12, thermal_diffusivity=0.1))
    gm.add_edge("coarse", "fine", "temperature", "heat_source",
                mapping=mapping_factory(gm))
    gm.compile()
    return gm


def test_node_field_referenced_mapping_round_trips_in_memory():
    gm = _rods(lambda g: rbf_mapping(
        _grid(g, "coarse"), _grid(g, "fine"), kernel="multiquadric", epsilon=2.5,
        source_ref={"node": "coarse", "field": "grid_x"},
        target_ref={"node": "fine", "field": "grid_x"}))
    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)

    prim = stage.GetPrimAtPath("/Simulation/edges/e0")
    stored = json.loads(prim.GetAttribute("maddening:mappingSpecJson").Get())
    assert stored["kind"] == "rbf" and stored["epsilon"] == 2.5
    assert stored["points"]["source_points"] == {"node": "coarse", "field": "grid_x"}
    assert "H" not in stored

    gm2 = load_graph_from_usd(stage)
    gm2.compile()
    assert list(gm2.params["mappings"]) == [C2F]
    np.testing.assert_array_equal(np.asarray(gm2.params["mappings"][C2F]["H"]),
                                  np.asarray(gm.params["mappings"][C2F]["H"]))
    np.testing.assert_array_equal(np.asarray(gm2.step()["fine"]["temperature"]),
                                  np.asarray(gm.step()["fine"]["temperature"]))


def test_asset_referenced_matrix_resolves_relative_to_the_stage_file(tmp_path):
    H = np.zeros((12, 6), np.float32)
    H[::2, :] = np.eye(6)
    H[1::2, :] = np.eye(6)
    np.save(tmp_path / "weights.npy", H)
    gm = _rods(lambda g: matrix_mapping(H, asset="weights.npy"))

    stage = Usd.Stage.CreateNew(str(tmp_path / "graph.usda"))
    save_graph_to_usd(gm, stage)
    stage.GetRootLayer().Save()

    gm2 = load_graph_from_usd(Usd.Stage.Open(str(tmp_path / "graph.usda")))
    gm2.compile()
    np.testing.assert_array_equal(np.asarray(gm2.params["mappings"][C2F]["H"]), H)
    # an explicit base_dir overrides the stage directory
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(ValueError, match="missing point asset 'weights.npy'"):
        load_graph_from_usd(Usd.Stage.Open(str(tmp_path / "graph.usda")), base_dir=other)


def test_save_refuses_a_mapping_without_point_references():
    gm = _rods(lambda g: matrix_mapping(np.ones((12, 6), np.float32)))
    with pytest.raises(ValueError, match=r"asset=") as exc:
        save_graph_to_usd(gm, Usd.Stage.CreateInMemory())
    assert C2F in str(exc.value)


def test_edge_without_mapping_carries_no_spec_attribute():
    gm = GraphManager()
    gm.add_node(HeatNode("coarse", 1e-4, n_cells=6))
    gm.add_node(HeatNode("fine", 1e-4, n_cells=6))
    gm.add_edge("coarse", "fine", "temperature", "heat_source")
    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    attr = stage.GetPrimAtPath("/Simulation/edges/e0").GetAttribute("maddening:mappingSpecJson")
    assert not attr or attr.Get() in (None, "")
    assert load_graph_from_usd(stage).edges[0].mapping is None
