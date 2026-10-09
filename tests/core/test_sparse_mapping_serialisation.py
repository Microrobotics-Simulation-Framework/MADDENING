"""A sparse mapping round-trips through a config as its recipe.

A config carries the kind, the hyper-parameters and *references* to the
arrays the mapping was built from -- never the index, never the weights.
``from_dict`` calls the same builder on the same arrays, so the reloaded
mapping has the same index (bit for bit, hence the same structure digest)
and the same weights, and steps exactly as the graph it was saved from.

Run over every sparse case (``tests/sparse_mapping_support.py``): nearest
neighbour in both modes and both conservative forms and the 1-D projection
from node-field references, and ``sparse_matrix`` from two members of one
``.npz`` saved next to the config.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json

import numpy as np
import pytest
import yaml

from maddening.core.coupling.mapping import projection_1d_mapping
from maddening.core.coupling.mapping_spec import (
    INLINE_POINT_LIMIT,
    MappingRebuildError,
    MappingSpec,
    PointReferenceError,
    point_array_digest,
)
from maddening.core.coupling.sparse_mapping import (
    StaticSparseMapping,
    sparse_matrix_mapping,
    sparse_nearest_neighbor_mapping,
    sparse_projection_1d_mapping,
)
from maddening.core.graph_manager import GraphManager
from tests.sparse_mapping_support import (
    A2B,
    B2A,
    CASES,
    REGISTRY,
    Interface,
    edge_mapping,
    mapped_pair,
    node_points,
    x64,
)


def _structure(mapping: StaticSparseMapping) -> tuple:
    counts = None if mapping.counts is None else np.asarray(mapping.counts).tobytes()
    return (mapping.layout, mapping.n_source, mapping.n_target, mapping.indices.shape,
            mapping.indices.tobytes(), counts, mapping.structure_digest())


def _assert_same_mapping(got, expected) -> None:
    """The same sparse mapping: class, kind, mode, spec, index and every
    weight bit."""
    assert type(got) is type(expected) is StaticSparseMapping
    assert (got.kind, got.mode) == (expected.kind, expected.mode)
    assert got.spec == expected.spec
    assert _structure(got) == _structure(expected)
    a, b = np.asarray(got.weights), np.asarray(expected.weights)
    assert a.dtype == b.dtype and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("codec", ["json", "yaml"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_a_sparse_mapping_round_trips_through_a_config_as_its_recipe(name, codec, tmp_path):
    case = CASES[name]
    gm = mapped_pair(case, base_dir=tmp_path)
    config = gm.to_dict()
    text = json.dumps(config) if codec == "json" else yaml.safe_dump(config)
    written = json.loads(text) if codec == "json" else yaml.safe_load(text)

    for edge, stored in zip(gm.edges, written["edges"]):
        mapping = stored["mapping"]
        assert mapping["kind"] == case.kind and mapping["mode"] == case.mode
        assert mapping["shape"] == [edge.mapping.n_target, edge.mapping.n_source]
        # the recipe: hyper-parameters and references, nothing of the structure
        assert set(mapping) == {"kind", "mode", "shape", "points",
                                *edge.mapping.spec.hyperparameters}
        assert set(mapping["points"]) == set(case.arrays)
        for array, reference in mapping["points"].items():
            assert set(reference) <= {"node", "field", "asset", "key", "sha256"}
            assert len(reference["sha256"]) == 64
        for absent in ("W", "indices", "counts", "k", "nnz", "layout", "weights"):
            assert absent not in mapping or absent in case.arrays
    assert "inline" not in text
    if case.kind == "sparse_nearest_neighbor":
        assert written["edges"][0]["mapping"]["transpose"] == (
            "scatter" if case.layout == "scatter" else "gather")
    if case.kind == "sparse_matrix":
        stored = written["edges"][0]["mapping"]
        assert stored["n_source"] == 4 and type(stored["n_source"]) is int
        assert stored["name"] == "stencil"
        assert stored["points"]["indices"] == {
            "asset": "a_to_b.npz", "key": "indices",
            "sha256": point_array_digest(case.assets(node_points(gm, "a", "b"))["indices"])}

    reloaded = GraphManager.from_dict(written, REGISTRY, base_dir=tmp_path)
    reloaded.compile()
    for edge, again in zip(gm.edges, reloaded.edges):
        _assert_same_mapping(again.mapping, edge.mapping)
    for key in (A2B, B2A):
        assert list(reloaded.params["mappings"][key]) == ["W"]
    a, b = gm.run_scan(8), reloaded.run_scan(8)
    for node in ("a", "b"):
        np.testing.assert_array_equal(np.asarray(a[node]["x"]), np.asarray(b[node]["x"]))
    # and the reloaded graph writes the config it was read from
    assert json.loads(json.dumps(reloaded.to_dict())) == json.loads(json.dumps(config))


@pytest.mark.parametrize("name", sorted(CASES))
def test_add_edge_takes_a_sparse_spec_and_its_dict_form(name, tmp_path):
    """A hand-written spec without content hashes or optional
    hyper-parameters, and its dict: rebuilt by the graph, recording the
    hashes of what it resolved."""
    case = CASES[name]
    built = mapped_pair(case, base_dir=tmp_path, compile=False).edges[0].mapping
    hyper = dict(built.spec.hyperparameters)
    points = {array: {k: v for k, v in reference.items() if k != "sha256"}
              for array, reference in built.spec.points.items()}

    for given in (MappingSpec(case.kind, hyper, points),
                  {"kind": case.kind, **hyper, "points": points}):
        gm = GraphManager()
        gm.add_node(Interface("a", 0.125, n=4))
        gm.add_node(Interface("b", 0.125, n=6))
        if case.kind == "sparse_matrix":
            # asset paths given to add_edge are relative to the working directory
            previous = os.getcwd()
            os.chdir(tmp_path)
            try:
                gm.add_edge("a", "b", "x", "inp", mapping=given)
            finally:
                os.chdir(previous)
        else:
            gm.add_edge("a", "b", "x", "inp", mapping=given)
        _assert_same_mapping(gm.edges[0].mapping, built)


def test_the_two_conservative_forms_are_two_recipes_of_one_operator(tmp_path):
    """``transpose`` is recorded, so a reload applies the operator the way
    it was built to be applied; the two differ in layout and agree in what
    they compute."""
    forms = {}
    for transpose in ("gather", "scatter"):
        gm = mapped_pair(CASES[f"nearest-conservative-{transpose}"])
        config = json.loads(json.dumps(gm.to_dict()))
        assert [e["mapping"]["transpose"] for e in config["edges"]] == [transpose] * 2
        reloaded = GraphManager.from_dict(config, REGISTRY)
        assert [e.mapping.layout for e in reloaded.edges] == [transpose] * 2
        reloaded.compile()
        forms[transpose] = np.asarray(reloaded.run_scan(6)["b"]["x"])
    np.testing.assert_allclose(forms["gather"], forms["scatter"], rtol=1e-5)

    # a spec that leaves the form out is the default one
    points = {"source_points": [0.0, 0.5, 1.0], "target_points": [0.2, 0.8]}
    default = MappingSpec.from_dict({"kind": "sparse_nearest_neighbor",
                                     "mode": "conservative", "points": points})
    assert default.build(lambda ref: np.asarray(ref["inline"])).layout == "gather"
    with pytest.raises(ValueError, match="takes only transpose='gather'"):
        MappingSpec.from_dict({"kind": "sparse_nearest_neighbor", "transpose": "scatter",
                               "points": points}).build(
            lambda ref: np.asarray(ref["inline"]))


def test_a_small_point_set_is_inlined_and_a_large_one_needs_a_reference():
    small = sparse_nearest_neighbor_mapping(np.linspace(0.0, 1.0, 5), [0.1, 0.9])
    assert small.spec.points["source_points"] == {
        "inline": np.linspace(0.0, 1.0, 5).tolist(), "dtype": "float64"}
    rebuilt = small.spec.build(lambda ref: np.asarray(ref["inline"], dtype=ref["dtype"]))
    _assert_same_mapping(rebuilt, small)

    big = np.linspace(0.0, 1.0, INLINE_POINT_LIMIT + 1)
    for mapping, missing in (
            (sparse_nearest_neighbor_mapping(big, [0.1, 0.9]), "source_points"),
            (sparse_projection_1d_mapping([0.0, 0.5, 1.0], big), "target_boundaries")):
        assert mapping.spec.missing_points() == [missing]
        gm = GraphManager()
        gm.add_node(Interface("a", 0.125, n=mapping.n_source))
        gm.add_node(Interface("b", 0.125, n=mapping.n_target))
        gm.add_edge("a", "b", "x", "inp", mapping=mapping)
        gm.compile()
        gm.step()                                            # usable ...
        with pytest.raises(ValueError, match=rf"point set\(s\) \['{missing}'\] were not "
                                             r"recorded"):
            gm.to_dict()                                     # ... not serialisable
        assert gm.to_dict(strict_mappings=False)["edges"][0]["mapping"]["kind"] == mapping.kind


def test_sparse_matrix_needs_both_assets_and_is_never_inlined(tmp_path):
    indices = np.array([[0, 1], [1, 2], [2, -1]])
    values = np.array([[0.5, 0.5], [0.25, 0.75], [1.0, 0.0]], np.float32)
    np.save(tmp_path / "indices.npy", indices)
    np.save(tmp_path / "values.npy", values)

    def graph(**assets):
        gm = GraphManager()
        gm.add_node(Interface("a", 0.125, n=3))
        gm.add_node(Interface("b", 0.125, n=3))
        gm.add_edge("a", "b", "x", "inp", mapping=sparse_matrix_mapping(
            indices, values, n_source=3, **assets))
        return gm

    for assets, missing in (({}, "['indices', 'values']"),
                            (dict(indices_asset="indices.npy"), "['values']"),
                            (dict(values_asset="values.npy"), "['indices']")):
        with pytest.raises(ValueError, match="cannot be serialised") as refused:
            graph(**assets).to_dict()
        assert missing in str(refused.value) and "_asset=" in str(refused.value)
    shown = graph().to_dict(strict_mappings=False)["edges"][0]["mapping"]
    assert shown["points"] == {"indices": None, "values": None} and shown["n_source"] == 3

    gm = graph(indices_asset="indices.npy", values_asset="values.npy")
    config = json.loads(json.dumps(gm.to_dict()))
    assert "inline" not in json.dumps(config)
    reloaded = GraphManager.from_dict(config, REGISTRY, base_dir=tmp_path)
    _assert_same_mapping(reloaded.edges[0].mapping, gm.edges[0].mapping)
    # the asset must hold exactly the array the mapping was built from
    with pytest.raises(PointReferenceError, match="carries sha256"):
        sparse_matrix_mapping(indices, values, n_source=3, indices_asset={
            "asset": "indices.npy", "sha256": "0" * 64})


@pytest.mark.parametrize("changed", ["indices", "values"])
def test_an_asset_that_changed_since_the_save_is_refused_naming_the_edge(changed, tmp_path):
    """The reference records the content hash of each array: a pattern or a
    weight edited on disk is not silently rebuilt into another operator."""
    gm = mapped_pair(CASES["matrix"], base_dir=tmp_path)
    config = json.loads(json.dumps(gm.to_dict()))
    with np.load(tmp_path / "a_to_b.npz") as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if changed == "indices":
        used = np.argwhere(arrays["indices"] >= 0)[0]
        arrays["indices"][tuple(used)] = (arrays["indices"][tuple(used)] + 1) % 4
    else:
        arrays["values"][0, 0] += 0.125
    np.savez(tmp_path / "a_to_b.npz", **arrays)
    with pytest.raises(MappingRebuildError, match="differs from the points") as refused:
        GraphManager.from_dict(config, REGISTRY, base_dir=tmp_path)
    assert refused.value.edge == "a.x -> b.inp" and refused.value.kind == "sparse_matrix"


def test_a_node_field_that_moved_is_refused_when_the_config_is_written():
    gm = mapped_pair(CASES["nearest-consistent"])
    gm.to_dict()
    gm.get_node("a")._points[0] += 0.25              # the field no longer hashes as recorded
    with pytest.raises(ValueError, match="no longer describes the points"):
        gm.to_dict()


@pytest.mark.parametrize("enabled", [False, True], ids=["x64-off", "x64-on"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_the_rebuild_gives_the_same_index_and_weights_under_either_x64_setting(
        name, enabled, tmp_path):
    """The index is int32 and the nearest-neighbour and projection weights
    float32 whatever ``jax_enable_x64`` says; ``sparse_matrix`` keeps the
    dtype of the values it was given."""
    case = CASES[name]
    with x64(enabled):
        gm = mapped_pair(case, base_dir=tmp_path)
        config = json.loads(json.dumps(gm.to_dict()))
        reloaded = GraphManager.from_dict(config, REGISTRY, base_dir=tmp_path)
        for edge, again in zip(gm.edges, reloaded.edges):
            _assert_same_mapping(again.mapping, edge.mapping)
            assert edge.mapping.indices.dtype == np.int32
            expected = ("float64" if enabled and case.kind == "sparse_matrix" else "float32")
            assert str(edge.mapping.weights.dtype) == expected


_FLOAT32_KINDS = sorted(name for name, case in CASES.items() if case.kind != "sparse_matrix")


@pytest.mark.parametrize("written_under", [False, True], ids=["written-x64-off", "written-x64-on"])
@pytest.mark.parametrize("name", _FLOAT32_KINDS + ["dense-projection"])
def test_a_config_written_under_one_x64_setting_rebuilds_the_same_weights_under_the_other(
        name, written_under, tmp_path):
    """The nearest-neighbour and projection kinds hold float32 weights
    under either setting (the node fields they are built from are
    float64), so the recipe a config carries rebuilds the operator that
    was saved, bit for bit, in a process with the other setting: the
    sparse kinds and the dense projection they are held to."""
    def pair():
        if name != "dense-projection":
            return mapped_pair(CASES[name], base_dir=tmp_path)
        gm = mapped_pair(CASES["projection"], compile=False)
        for edge in list(gm.edges):
            gm.remove_edge(edge.source_node, edge.target_node, edge.source_field,
                           edge.target_field)
        for source, target in (("a", "b"), ("b", "a")):
            points = node_points(gm, source, target)
            gm.add_edge(source, target, "x", "inp", mapping=projection_1d_mapping(
                points["source_boundaries"], points["target_boundaries"],
                source_ref={"node": source, "field": "boundaries"},
                target_ref={"node": target, "field": "boundaries"}))
        gm.compile()
        return gm

    def weights(mapping):
        return np.asarray(mapping.H if name == "dense-projection" else mapping.weights)

    with x64(written_under):
        gm = pair()
        config = json.loads(json.dumps(gm.to_dict()))
        saved = [weights(edge.mapping) for edge in gm.edges]
        index = [None if name == "dense-projection" else _structure(edge.mapping)
                 for edge in gm.edges]
    with x64(not written_under):
        reloaded = GraphManager.from_dict(config, REGISTRY, base_dir=tmp_path)
        for edge, expected, structure in zip(reloaded.edges, saved, index):
            again = weights(edge.mapping)
            assert again.dtype == expected.dtype == np.float32
            assert again.tobytes() == expected.tobytes()
            if structure is not None:
                assert _structure(edge.mapping) == structure
        # (The step itself is not compared: its own arithmetic differs by a
        # rounding between the two settings, whatever the weights.)
        reloaded.step()


def test_a_sparse_mapping_built_by_hand_has_no_recipe_and_is_refused_by_the_writers():
    gm = GraphManager()
    gm.add_node(Interface("a", 0.125, n=3))
    gm.add_node(Interface("b", 0.125, n=2))
    gm.add_edge("a", "b", "x", "inp", mapping=StaticSparseMapping(
        np.array([[0, 1], [2, 2]]), np.ones((2, 2), np.float32), n_source=3))
    with pytest.raises(ValueError, match="carries no MappingSpec"):
        gm.to_dict()
    shown = gm.to_dict(strict_mappings=False)["edges"][0]["mapping"]
    assert shown == {"kind": "sparse_matrix", "mode": "consistent", "shape": [2, 3]}


def test_edge_mapping_helper_references_every_array_of_every_case(tmp_path):
    """The fixture can express a dropped reference: every array of every
    case is referenced by node field or by asset, never inlined."""
    gm = GraphManager()
    gm.add_node(Interface("a", 0.125, n=4))
    gm.add_node(Interface("b", 0.125, n=6))
    for case in CASES.values():
        mapping = edge_mapping(case, gm, "a", "b", base_dir=tmp_path)
        forms = {next(iter(reference)) for reference in mapping.spec.points.values()}
        assert forms == ({"asset"} if case.kind == "sparse_matrix" else {"node"})
        assert mapping.spec.missing_points() == []
