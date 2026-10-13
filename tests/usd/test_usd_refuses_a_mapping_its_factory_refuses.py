"""A USD stage whose mapping spec its factory refuses does not load.

``load_graph_from_usd`` rebuilds an edge's mapping by calling the factory
on the referenced points, exactly as ``GraphManager.from_dict`` does, so a
spec the factory refuses -- cell boundaries out of order, a NaN among the
points, a non-finite matrix (MADD-ANO-192) -- must come back as a
``MappingRebuildError`` naming the edge, not as a graph with a wrong
operator on it.
"""

import json

import numpy as np
import pytest
from pxr import Usd

from maddening.core.coupling.mapping import projection_1d_mapping
from maddening.core.coupling.mapping_spec import MappingRebuildError
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.usd.serialization import load_graph_from_usd, save_graph_to_usd

SOURCE, TARGET = np.linspace(0.0, 1.0, 7), np.linspace(0.0, 1.0, 4)
EDGE = "edge coarse.temperature -> fine.heat_source"


def _stage(tmp_path):
    """A stage holding one projection edge from a 6-cell rod to a 3-cell rod."""
    gm = GraphManager()
    gm.add_node(HeatNode("coarse", 1e-4, n_cells=6, thermal_diffusivity=0.1))
    gm.add_node(HeatNode("fine", 1e-4, n_cells=3, thermal_diffusivity=0.1))
    gm.add_edge("coarse", "fine", "temperature", "heat_source",
                mapping=projection_1d_mapping(SOURCE, TARGET))
    gm.compile()
    path = str(tmp_path / "graph.usda")
    stage = Usd.Stage.CreateNew(path)
    save_graph_to_usd(gm, stage)
    stage.GetRootLayer().Save()
    return gm, path


def _rewrite_spec(path, spec):
    stage = Usd.Stage.Open(path)
    stage.GetPrimAtPath("/Simulation/edges/e0").GetAttribute(
        "maddening:mappingSpecJson").Set(json.dumps(spec))
    stage.GetRootLayer().Save()


def test_the_stage_as_written_loads_the_same_operator(tmp_path):
    """The control: nothing below fails for a reason other than its spec."""
    gm, path = _stage(tmp_path)
    loaded = load_graph_from_usd(Usd.Stage.Open(path))
    np.testing.assert_array_equal(np.asarray(loaded.edges[0].mapping.H),
                                  np.asarray(gm.edges[0].mapping.H))


@pytest.mark.parametrize("spec,says", [
    ({"kind": "projection_1d", "points": {
        "source_boundaries": SOURCE[::-1].tolist(), "target_boundaries": TARGET.tolist()}},
     r"source_boundaries must be strictly increasing, but source_boundaries\[1\]"),
    ({"kind": "projection_1d", "points": {
        "source_boundaries": SOURCE.tolist(), "target_boundaries": [0.0, 0.7, 0.3, 1.0]}},
     r"target_boundaries must be strictly increasing, but target_boundaries\[2\]"),
    ({"kind": "projection_1d", "points": {
        "source_boundaries": {"asset": "bad.npy"}, "target_boundaries": TARGET.tolist()}},
     "source_boundaries holds a non-finite value at index 2 "),
    ({"kind": "nearest_neighbor", "mode": "consistent", "points": {
        "source_points": {"asset": "bad_points.npy"},
        "target_points": [0.2, 0.5, 0.8]}},
     "source_points holds a non-finite coordinate at index 2 "),
    ({"kind": "rbf", "mode": "consistent", "points": {
        "source_points": {"asset": "bad_points.npy"},
        "target_points": [0.2, 0.5, 0.8]}},
     "source_points holds a non-finite coordinate at index 2 "),
    ({"kind": "matrix", "points": {"H": {"asset": "bad_matrix.npy"}}},
     r"H holds a non-finite value at index \(0, 2\)"),
], ids=["descending source", "non-monotone target", "NaN boundary",
        "NaN point (nearest)", "NaN point (rbf)", "NaN matrix"])
def test_a_stage_whose_mapping_spec_the_factory_refuses_does_not_load(tmp_path, spec, says):
    _, path = _stage(tmp_path)
    bad = SOURCE.copy()
    bad[2] = np.nan
    np.save(tmp_path / "bad.npy", bad)
    np.save(tmp_path / "bad_points.npy", bad[:6])
    H = np.zeros((3, 6), np.float32)
    H[0, 2] = np.nan
    np.save(tmp_path / "bad_matrix.npy", H)
    _rewrite_spec(path, spec)
    with pytest.raises(MappingRebuildError, match=says) as caught:
        load_graph_from_usd(Usd.Stage.Open(path))
    assert str(caught.value).startswith(EDGE)
    assert type(caught.value.__cause__) is ValueError
