"""Interface mappings round-trip through configs as a ``MappingSpec``:
kind + hyper-parameters + point *references* (node field, asset file,
small inline list), never the weights.  ``from_dict`` rebuilds the
weights bitwise and registers ``params["mappings"]`` as ``add_edge``
does; a checkpoint loaded afterwards wins over the rebuilt weights."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from maddening.core.coupling.mapping import (
    StaticLinearMapping,
    matrix_mapping,
    nearest_neighbor_mapping,
    projection_1d_mapping,
    rbf_mapping,
)
from maddening.core.coupling.mapping_spec import (
    INLINE_POINT_LIMIT,
    MappingSpec,
    PointReferenceError,
    make_point_resolver,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.simulation.checkpoint import load_state, save_state
from maddening.nodes.heat import HeatNode
from maddening.serialization import config as cfg

N_COARSE, N_FINE = 6, 12
C2F = "coarse.temperature->fine.heat_source"
F2C = "fine.temperature->coarse.heat_source"


class Vec(SimulationNode):
    """n-vector integrating its boundary input (a stand-in interface)."""

    def __init__(self, name, timestep, n=3):
        super().__init__(name, timestep, n=n)

    def initial_state(self):
        return {"v": jnp.arange(1, self.params["n"] + 1, dtype=jnp.float32)}

    def update(self, s, bi, dt):
        return {"v": s["v"] + dt * bi.get("inp", jnp.zeros_like(s["v"]))}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(self.params["n"],), description="i")}


REGISTRY = {"HeatNode": HeatNode, "Vec": Vec}


def _rods():
    gm = GraphManager()
    gm.add_node(HeatNode("coarse", 1e-4, n_cells=N_COARSE, thermal_diffusivity=0.1,
                         initial_temperature=300.0))
    gm.add_node(HeatNode("fine", 1e-4, n_cells=N_FINE, thermal_diffusivity=0.1,
                         initial_temperature=350.0))
    return gm


def _grid(gm, name):
    return gm._nodes[name].node.static_data["grid_x"].value


def _two_rods_with_node_refs(kernel="thin_plate_spline"):
    """Mappings built from the rods' own ``grid_x`` static data and
    referenced by node field, so the config carries no coordinates.
    (No coupling group: ``to_dict`` does not carry those, and the
    trajectories are compared exactly.)"""
    gm = _rods()
    xc, xf = _grid(gm, "coarse"), _grid(gm, "fine")
    gm.add_edge("coarse", "fine", "temperature", "heat_source",
                mapping=rbf_mapping(xc, xf, epsilon=2.0, kernel=kernel,
                                    source_ref={"node": "coarse", "field": "grid_x"},
                                    target_ref={"node": "fine", "field": "grid_x"}))
    gm.add_edge("fine", "coarse", "temperature", "heat_source",
                mapping=rbf_mapping(xf, xc, epsilon=2.0, kernel=kernel, mode="conservative",
                                    source_ref={"node": "fine", "field": "grid_x"},
                                    target_ref={"node": "coarse", "field": "grid_x"}))
    gm.compile()
    gm.set_node_state("coarse", {"temperature": jnp.asarray(300 + 50 * xc ** 2, jnp.float32)})
    gm.set_node_state("fine", {"temperature": jnp.asarray(350 - 40 * xf, jnp.float32)})
    return gm


def _same_weights_and_trajectory(gm, gm2, keys):
    gm2.compile()
    assert set(gm2.params["mappings"]) == set(keys)
    for k in keys:
        np.testing.assert_array_equal(np.asarray(gm2.params["mappings"][k]["H"]),
                                      np.asarray(gm.params["mappings"][k]["H"]))
        assert gm2.params["mappings"][k]["H"].dtype == gm.params["mappings"][k]["H"].dtype
    for name in gm.node_names:
        gm2.set_node_state(name, gm.get_node_state(name))
    a, b = gm.run_scan(20), gm2.run_scan(20)
    for name in gm.node_names:
        for field in a[name]:
            np.testing.assert_array_equal(np.asarray(a[name][field]),
                                          np.asarray(b[name][field]))


# ---------------------------------------------------------------- round trips

@pytest.mark.parametrize("codec", ["json", "yaml"])
def test_rbf_node_field_references_round_trip(codec):
    gm = _two_rods_with_node_refs()
    d = cfg.to_dict(gm)
    text = json.dumps(d) if codec == "json" else yaml.safe_dump(d)
    assert "grid_x" in text and "\"H\"" not in text and "'H'" not in text
    back = json.loads(text) if codec == "json" else yaml.safe_load(text)
    m = back["edges"][0]["mapping"]
    assert m["points"] == {"source_points": {"node": "coarse", "field": "grid_x"},
                           "target_points": {"node": "fine", "field": "grid_x"}}
    assert m["kernel"] == "thin_plate_spline" and m["mode"] == "consistent"
    assert back["edges"][1]["mapping"]["mode"] == "conservative"
    gm2 = cfg.from_dict(back, REGISTRY)
    assert gm2.edges[0].mapping.spec == gm.edges[0].mapping.spec
    _same_weights_and_trajectory(gm, gm2, [C2F, F2C])


def test_inline_small_point_sets_round_trip_bitwise():
    """No reference given and <= INLINE_POINT_LIMIT points: the factory
    inlines them (with their dtype), and the rebuild is bitwise equal."""
    gm = GraphManager()
    gm.add_node(Vec("a", 1.0, n=5))
    gm.add_node(Vec("b", 1.0, n=3))
    src = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    tgt = np.array([0.1, 0.5, 0.9], np.float32)
    gm.add_edge("a", "b", "v", "inp",
                mapping=nearest_neighbor_mapping(src, tgt, mode="conservative"))
    gm.add_edge("a", "b", "v", "inp", additive=True,
                mapping=rbf_mapping(src, tgt, kernel="gaussian", epsilon=3.0,
                                    polynomial=False, ridge=1e-6))
    gm.compile()
    d = json.loads(json.dumps(gm.to_dict()))
    assert d["edges"][0]["mapping"]["points"]["source_points"] == {
        "inline": src.tolist(), "dtype": "float32"}
    assert d["edges"][1]["ordinal"] == 1
    gm2 = GraphManager.from_dict(d, REGISTRY)
    _same_weights_and_trajectory(gm, gm2, ["a.v->b.inp", "a.v->b.inp#1"])
    assert gm2.edges[1].mapping.describe()["polynomial"] is False


def test_projection_1d_asset_references_round_trip(tmp_path):
    """Boundaries saved as .npy / .npz assets next to the config, loaded
    back relative to ``base_dir``."""
    sb = np.linspace(0.0, 1.0, 4)
    tb = np.linspace(0.0, 1.0, 6)
    np.save(tmp_path / "source_bounds.npy", sb)
    np.savez(tmp_path / "grids.npz", coarse=sb, fine=tb)
    gm = GraphManager()
    gm.add_node(Vec("a", 1.0, n=3))
    gm.add_node(Vec("b", 1.0, n=5))
    gm.add_edge("a", "b", "v", "inp",
                mapping=projection_1d_mapping(
                    sb, tb, source_ref={"asset": "source_bounds.npy"},
                    target_ref={"asset": "grids.npz", "key": "fine"}))
    gm.compile()
    d = gm.to_dict()
    (tmp_path / "graph.json").write_text(json.dumps(d))
    back = json.loads((tmp_path / "graph.json").read_text())
    assert back["edges"][0]["mapping"]["points"]["target_boundaries"] == {
        "asset": "grids.npz", "key": "fine"}
    gm2 = cfg.from_dict(back, REGISTRY, base_dir=tmp_path)
    _same_weights_and_trajectory(gm, gm2, ["a.v->b.inp"])
    # an .npz with a single member needs no key
    np.savez(tmp_path / "one.npz", tb)
    resolve = make_point_resolver(base_dir=tmp_path)
    np.testing.assert_array_equal(resolve({"asset": "one.npz"}), tb)
    with pytest.raises(PointReferenceError, match="members"):
        resolve({"asset": "grids.npz"})
    with pytest.raises(PointReferenceError, match="no member"):
        resolve({"asset": "grids.npz", "key": "medium"})


def test_node_reference_falls_back_to_array_valued_params():
    """A non-uniform HeatNode's ``grid_points`` constructor parameter is a
    valid point field too."""
    gm = GraphManager()
    xs = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
    gm.add_node(HeatNode("rod", 1e-4, n_cells=5, grid_points=xs))
    gm.add_node(Vec("b", 1e-4, n=3))
    gm.add_edge("rod", "b", "temperature", "inp",
                mapping=nearest_neighbor_mapping(
                    xs, [0.0, 0.5, 1.0], source_ref={"node": "rod", "field": "grid_points"}))
    gm.compile()
    gm2 = GraphManager.from_dict(json.loads(json.dumps(gm.to_dict())), REGISTRY)
    _same_weights_and_trajectory(gm, gm2, ["rod.temperature->b.inp"])


def test_matrix_mapping_requires_an_asset_and_round_trips_through_it(tmp_path):
    gm = GraphManager()
    gm.add_node(Vec("a", 1.0))
    gm.add_node(Vec("b", 1.0))
    H = np.array([[1, 0, 0], [0, 0.5, 0.5], [0, 0, 1]], np.float32)
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(H, kind="supermesh"))
    gm.compile()
    with pytest.raises(ValueError, match=r"asset=") as exc:
        gm.to_dict()
    assert "a.v->b.inp" in str(exc.value) and "['H']" in str(exc.value)
    # display path still works and shows the gap
    shown = gm.to_dict(strict_mappings=False)["edges"][0]["mapping"]
    assert shown["points"] == {"H": None} and shown["label"] == "supermesh"

    np.save(tmp_path / "H.npy", H)
    gm.remove_edge("a", "b", "v", "inp")
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(H, kind="supermesh", asset="H.npy"))
    gm.compile()
    d = gm.to_dict()
    assert d["edges"][0]["mapping"] == {
        "kind": "matrix", "mode": "consistent", "shape": [3, 3], "label": "supermesh",
        "points": {"H": {"asset": "H.npy"}}}
    gm2 = GraphManager.from_dict(d, REGISTRY, base_dir=tmp_path)
    assert gm2.edges[0].mapping.kind == "supermesh"
    _same_weights_and_trajectory(gm, gm2, ["a.v->b.inp"])
    with pytest.raises(ValueError, match="matrix_mapping: asset="):
        matrix_mapping(H, asset={"node": "a", "field": "v"})


def test_large_point_set_without_reference_is_usable_but_not_serialisable():
    n = INLINE_POINT_LIMIT + 1
    pts = np.linspace(0.0, 1.0, n)
    m = rbf_mapping(pts, [0.0, 1.0])
    assert m.spec.missing_points() == ["source_points"]
    assert m.n_source == n                                   # still a working mapping
    gm = GraphManager()
    gm.add_node(Vec("a", 1.0, n=n))
    gm.add_node(Vec("b", 1.0, n=2))
    gm.add_edge("a", "b", "v", "inp", mapping=m)
    with pytest.raises(ValueError, match=r"source_ref=.*node.*asset"):
        gm.to_dict()
    with pytest.raises(PointReferenceError, match="no reference for \\['source_points'\\]"):
        m.spec.build(make_point_resolver())
    with pytest.raises(PointReferenceError, match="exceed INLINE_POINT_LIMIT"):
        rbf_mapping(pts, [0.0, 1.0], source_ref={"inline": pts.tolist()})


def test_add_edge_accepts_a_spec_and_its_dict_form():
    gm = _rods()
    spec = MappingSpec("rbf", {"kernel": "gaussian", "epsilon": 2.0, "mode": "consistent"},
                       {"source_points": {"node": "coarse", "field": "grid_x"},
                        "target_points": {"node": "fine", "field": "grid_x"}})
    gm.add_edge("coarse", "fine", "temperature", "heat_source", mapping=spec)
    gm.add_edge("fine", "coarse", "temperature", "heat_source", mapping={
        "kind": "nearest_neighbor", "mode": "conservative",
        "points": {"source_points": {"node": "fine", "field": "grid_x"},
                   "target_points": {"node": "coarse", "field": "grid_x"}}})
    gm.compile()
    expected = rbf_mapping(_grid(gm, "coarse"), _grid(gm, "fine"), kernel="gaussian",
                           epsilon=2.0)
    np.testing.assert_array_equal(np.asarray(gm.params["mappings"][C2F]["H"]),
                                  np.asarray(expected.H))
    rebuilt = gm.edges[0].mapping.spec
    assert rebuilt.kind == "rbf" and rebuilt.points == spec.points
    assert rebuilt.hyperparameters == {**spec.hyperparameters, "polynomial": True,
                                       "ridge": 1e-8}                # defaults filled in
    assert gm.edges[1].mapping.kind == "nearest_neighbor"


# ---------------------------------------------------------------- checkpoints

def test_checkpoint_weights_win_over_the_rebuilt_spec(tmp_path):
    """Config carries the recipe, the checkpoint the (possibly trained)
    weights: loading both restores the checkpoint's."""
    gm = _two_rods_with_node_refs()
    trained = 1.5 * gm.params["mappings"][C2F]["H"]
    gm.params["mappings"][C2F]["H"] = trained
    ck = save_state(gm, tmp_path / "ck")
    config = json.loads(json.dumps(gm.to_dict()))
    assert "H" not in config["edges"][0]["mapping"]

    gm2 = cfg.from_dict(config, REGISTRY)
    gm2.compile()
    geometric = rbf_mapping(_grid(gm2, "coarse"), _grid(gm2, "fine"), epsilon=2.0,
                            kernel="thin_plate_spline").H
    np.testing.assert_array_equal(np.asarray(gm2.params["mappings"][C2F]["H"]),
                                  np.asarray(geometric))                 # rebuilt
    load_state(gm2, ck)
    np.testing.assert_array_equal(np.asarray(gm2.params["mappings"][C2F]["H"]),
                                  np.asarray(trained))                   # checkpoint wins
    np.testing.assert_array_equal(np.asarray(gm2.step()["fine"]["temperature"]),
                                  np.asarray(gm.step()["fine"]["temperature"]))


# ---------------------------------------------------------------- errors

def _edge_dict(mapping):
    return {"nodes": [{"type": "Vec", "name": "a", "timestep": 1.0, "params": {"n": 3}},
                      {"type": "Vec", "name": "b", "timestep": 1.0, "params": {"n": 3}}],
            "edges": [{"source_node": "a", "target_node": "b", "source_field": "v",
                       "target_field": "inp", "mapping": mapping}],
            "external_inputs": []}


INLINE3 = {"inline": [0.0, 0.5, 1.0], "dtype": "float64"}


@pytest.mark.parametrize("mapping, message", [
    ({"kind": "supermesh", "points": {}}, "unknown mapping kind 'supermesh'"),
    ({"kind": "rbf", "mode": "consistent"}, "no 'points'"),
    ({"kind": "rbf", "sigma": 2.0,
      "points": {"source_points": INLINE3, "target_points": INLINE3}},
     "no hyper-parameter\\(s\\) \\['sigma'\\]"),
    ({"kind": "projection_1d", "points": {"source_points": INLINE3}},
     "takes point sets \\['source_boundaries', 'target_boundaries'\\]"),
    ({"kind": "nearest_neighbor",
      "points": {"source_points": {"node": "zed", "field": "v"}, "target_points": INLINE3}},
     "unknown node 'zed'"),
    ({"kind": "nearest_neighbor",
      "points": {"source_points": {"node": "a", "field": "grid"}, "target_points": INLINE3}},
     "no point field 'grid'"),
    ({"kind": "nearest_neighbor",
      "points": {"source_points": {"asset": "missing.npy"}, "target_points": INLINE3}},
     "missing point asset 'missing.npy'"),
    ({"kind": "nearest_neighbor",
      "points": {"source_points": {"asset": "../escape.npy"}, "target_points": INLINE3}},
     "must not contain '..'"),
    ({"kind": "nearest_neighbor",
      "points": {"source_points": {"asset": "/etc/passwd"}, "target_points": INLINE3}},
     "must be relative"),
    ({"kind": "nearest_neighbor",
      "points": {"source_points": {"mesh": "a"}, "target_points": INLINE3}},
     "unknown point reference"),
    ({"kind": "nearest_neighbor",
      "points": {"source_points": {"node": "a"}, "target_points": INLINE3}},
     "a node reference is"),
    ({"kind": "nearest_neighbor",
      "points": {"source_points": 7, "target_points": INLINE3}},
     "must be a dict"),
    ({"kind": "nearest_neighbor", "shape": [3, 9],
      "points": {"source_points": INLINE3, "target_points": INLINE3}},
     "recorded \\[3, 9\\]"),
])
def test_from_dict_names_the_broken_edge_and_the_problem(tmp_path, mapping, message):
    with pytest.raises(ValueError, match=message) as exc:
        GraphManager.from_dict(_edge_dict(mapping), REGISTRY, base_dir=tmp_path)
    assert "edge a.v -> b.inp" in str(exc.value)


def test_hand_built_or_custom_mapping_is_refused_by_the_config_writer():
    gm = GraphManager()
    gm.add_node(Vec("a", 1.0))
    gm.add_node(Vec("b", 1.0))
    gm.add_edge("a", "b", "v", "inp", mapping=StaticLinearMapping(jnp.eye(3)))
    with pytest.raises(ValueError, match="carries no MappingSpec"):
        gm.to_dict()
    assert gm.to_dict(strict_mappings=False)["edges"][0]["mapping"]["kind"] == "matrix"


def test_spec_validates_its_own_fields():
    with pytest.raises(ValueError, match="unknown mapping kind"):
        MappingSpec("spline", {}, {})
    with pytest.raises(ValueError, match="no hyper-parameter"):
        MappingSpec("nearest_neighbor", {"epsilon": 1.0},
                    {"source_points": INLINE3, "target_points": INLINE3})
    spec = MappingSpec("rbf", {"epsilon": 2.0}, {"source_points": [0.0, 1.0],
                                                 "target_points": [[0.5]]})
    assert spec.points["source_points"] == {"inline": [0.0, 1.0], "dtype": "float64"}
    assert MappingSpec.from_dict(spec.to_dict()) == spec
    assert hash(spec) == hash(MappingSpec.from_dict(spec.to_dict()))
    assert spec.to_dict() == {"kind": "rbf", "epsilon": 2.0, "points": spec.points}
