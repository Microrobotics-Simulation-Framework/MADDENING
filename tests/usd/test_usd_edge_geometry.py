"""A USD stage carries an edge's geometry, and only for an edge that has one.

An edge with a geometry-dependent mapping (experimental) names the state
field its mapping reads: ``add_edge(..., geometry=(anchor, field))``.  A
stage holds it as two string attributes on the edge prim,
``maddening:geometryAnchor`` and ``maddening:geometryField``, written
only for such an edge, so the stage of any other graph is the one it was.
A stage with one of the two and not the other is refused, naming the
edge; any other malformed pair is refused by ``add_edge``'s own rule.
"""

import numpy as np
import pytest
from pxr import Usd

from maddening.core.coupling.mapping import nearest_neighbor_mapping
from maddening.usd.serialization import load_graph_from_usd, save_graph_to_usd

from tests.core import geometry_surface_graphs as G

ANCHOR, FIELD = "maddening:geometryAnchor", "maddening:geometryField"


def _saved(tmp_path, gm, name="graph.usda"):
    path = str(tmp_path / name)
    stage = Usd.Stage.CreateNew(path)
    save_graph_to_usd(gm, stage)
    stage.GetRootLayer().Save()
    return path


def _load(path):
    return load_graph_from_usd(Usd.Stage.Open(path), node_registry=G.REGISTRY)


def _edge_prims(path):
    stage = Usd.Stage.Open(path)
    prims = {}
    for prim in stage.GetPrimAtPath("/Simulation/edges").GetChildren():
        key = (f"{prim.GetAttribute('maddening:sourceNode').Get()}."
               f"{prim.GetAttribute('maddening:sourceField').Get()}->"
               f"{prim.GetAttribute('maddening:targetNode').Get()}."
               f"{prim.GetAttribute('maddening:targetField').Get()}")
        prims[key] = prim
    return stage, prims


@pytest.mark.parametrize("group", [False, True], ids=["plain step", "group"])
def test_a_stage_carries_each_edge_s_geometry_and_the_loaded_graph_steps_the_same(
        tmp_path, group):
    kw = dict(group=True, diagnostics=False) if group else {}
    gm = G.graph(**kw)
    path = _saved(tmp_path, gm)
    stage, prims = _edge_prims(path)
    assert prims[G.GATHER].GetAttribute(ANCHOR).Get() == "target"
    assert prims[G.SCATTER].GetAttribute(ANCHOR).Get() == "source"
    assert {p.GetAttribute(FIELD).Get() for p in prims.values()} == {"pos"}

    loaded = _load(path)
    assert {e.key: e.geometry for e in loaded.edges} == {
        G.GATHER: ("target", "pos"), G.SCATTER: ("source", "pos")}
    assert loaded.to_dict()["edges"] == gm.to_dict()["edges"]
    loaded.compile()
    for _ in range(3):
        gm.step()
        loaded.step()
    G.assert_same_states(gm, loaded, "loaded from the stage")
    # Written again, the loaded graph gives the same two attributes.
    stage_again, again = _edge_prims(_saved(tmp_path, loaded, "again.usda"))
    assert stage_again is not stage
    for key, prim in prims.items():
        for name in (ANCHOR, FIELD):
            assert again[key].GetAttribute(name).Get() == prim.GetAttribute(name).Get()


def test_a_stage_of_a_graph_without_a_geometry_edge_has_neither_attribute(tmp_path):
    """Nothing is written for an edge without one, a mapped edge included."""
    grid_points = G.SPACING * np.arange(G.N_GRID)
    marker_points = 0.3 + 0.4 * np.arange(G.N_MARKERS)
    gm = G.graph(compile=False, edges=False)
    gm.add_edge("grid", "markers", "x", "sampled",
                mapping=nearest_neighbor_mapping(grid_points, marker_points))
    gm.add_edge("markers", "grid", "x", "deposit",
                mapping=nearest_neighbor_mapping(marker_points, grid_points))
    gm.compile()
    path = _saved(tmp_path, gm)
    stage, prims = _edge_prims(path)
    assert stage and len(prims) == 2
    for prim in prims.values():
        assert prim.GetAttribute("maddening:mappingSpecJson").Get()
        for name in (ANCHOR, FIELD):
            assert not prim.HasAttribute(name), (prim.GetPath(), name)
    with open(path) as text:
        assert "maddening:geometry" not in text.read()
    assert all(e.geometry is None for e in _load(path).edges)


@pytest.mark.parametrize("dropped,kept", [(ANCHOR, FIELD), (FIELD, ANCHOR)])
def test_a_stage_with_half_a_geometry_is_refused_naming_the_edge(tmp_path, dropped, kept):
    path = _saved(tmp_path, G.graph())
    stage, prims = _edge_prims(path)
    prims[G.SCATTER].RemoveProperty(dropped)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError) as refused:
        _load(path)
    text = str(refused.value)
    assert "markers.x -> grid.deposit" in text and f"{kept} without {dropped}" in text


@pytest.mark.parametrize("name,value,says", [
    (ANCHOR, "markers", r'geometry must be \("source", <field>\) or \("target", <field>\)'),
    (ANCHOR, "", r"geometry must be"),
    (FIELD, "", r"geometry must be"),
])
def test_a_malformed_geometry_in_a_stage_is_add_edge_s_refusal(tmp_path, name, value, says):
    path = _saved(tmp_path, G.graph())
    stage, prims = _edge_prims(path)
    prims[G.SCATTER].GetAttribute(name).Set(value)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match=says) as refused:
        _load(path)
    assert G.SCATTER in str(refused.value)


def test_a_stage_whose_geometry_dependent_mapping_lost_both_attributes_is_refused(tmp_path):
    """G3 at this door: the mapping reads a geometry and the stage names none."""
    path = _saved(tmp_path, G.graph())
    stage, prims = _edge_prims(path)
    for name in (ANCHOR, FIELD):
        prims[G.GATHER].RemoveProperty(name)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="reads a moving geometry and none was given"):
        _load(path)


def test_a_stage_that_names_a_field_the_anchor_does_not_hold_fails_at_compile(tmp_path):
    """G5 is the graph's, at compile, whichever door the edge came through."""
    path = _saved(tmp_path, G.graph())
    stage, prims = _edge_prims(path)
    prims[G.SCATTER].GetAttribute(FIELD).Set("x_not_held")
    stage.GetRootLayer().Save()
    loaded = _load(path)
    assert loaded.edges[1].geometry == ("source", "x_not_held")
    with pytest.raises(RuntimeError, match="geometry field 'x_not_held' is not in the state"):
        loaded.compile()
    assert np.asarray(G.graph().get_node_state("markers")["pos"]).shape == (G.N_MARKERS, 1)
