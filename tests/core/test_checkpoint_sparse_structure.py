"""A checkpoint's sparse mapping weights are refused on another pattern.

A sparse mapping's weights are one number per slot of an index the
checkpoint does not carry.  Restored onto a different index of the same
shape they would run an operator nobody computed, and nothing about their
shape or values could tell.  So ``save_state`` writes the digest of the
pattern beside the weights, and ``load_state`` -- and ``POST
/checkpoint/load``, which calls it -- refuses weights whose digest is not
the live mapping's, before anything is read into the graph:

* another pattern of the same shape (other points, another index);
* the same index read in the other layout;
* weights saved for a dense mapping onto a sparse one, and the reverse;
* an archive whose digest member was removed, or is not a digest.

A graph reloads its own checkpoint, a config rebuilt from the same points
reloads it, and a dense mapping's checkpoint is what it was.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.mapping import StaticLinearMapping, matrix_mapping
from maddening.core.coupling.sparse_mapping import (
    StaticSparseMapping,
    sparse_matrix_mapping,
    sparse_nearest_neighbor_mapping,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.simulation.checkpoint import (
    _STRUCTURE_MEMBER,
    load_state,
    save_state,
)
from tests.registered_mapping_kinds import INVERSE_DISTANCE, KINDS
from tests.sparse_mapping_support import (
    A2B,
    B2A,
    CASES,
    REGISTRY,
    Interface,
    mapped_pair,
)

DIGEST = "structure.sha256"


def _members(path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _snapshot(gm: GraphManager) -> dict:
    return {"state": {n: {f: np.asarray(v).copy() for f, v in fields.items()}
                      for n, fields in gm._state.items()},
            "params": {section: {o: {k: np.asarray(v).copy() for k, v in leaves.items()}
                                 for o, leaves in owners.items()}
                       for section, owners in gm.params.items()}}


def _flat(tree: dict, prefix: tuple = ()) -> dict:
    out = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(_flat(value, prefix + (key,)))
        else:
            out[prefix + (key,)] = (str(value.dtype), value.shape, value.tobytes())
    return out


def _assert_untouched(gm: GraphManager, before: dict) -> None:
    """A refused load left every state field and every parameter leaf as
    it was: names, dtypes, shapes and bytes."""
    assert _flat(_snapshot(gm)) == _flat(before), "a refused load changed the graph"


def _train(gm: GraphManager, factor: float = 1.25) -> dict:
    """Every mapping weight moved, as a fit would leave it; the new values."""
    moved = {}
    for key, slot in gm.params["mappings"].items():
        for name in slot:
            slot[name] = (slot[name] * factor).astype(slot[name].dtype)
        moved[key] = {name: np.asarray(leaf).copy() for name, leaf in slot.items()}
    return moved


def _pair(points_b=None, *, mode="consistent", transpose="gather") -> GraphManager:
    """``a`` (4) onto ``b`` (6) by sparse nearest neighbour from explicit
    point sets, so that two graphs of one shape can differ in pattern."""
    gm = GraphManager()
    gm.add_node(Interface("a", 0.125, n=4))
    gm.add_node(Interface("b", 0.125, n=6, rate=0.25))
    xa = np.linspace(0.0, 1.0, 4)
    xb = np.linspace(0.0, 1.0, 6) if points_b is None else np.asarray(points_b)
    gm.add_edge("a", "b", "x", "inp", mapping=sparse_nearest_neighbor_mapping(
        xa, xb, mode=mode, transpose=transpose))
    gm.compile()
    return gm


# ---------------------------------------------------------------------------
# What a checkpoint carries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CASES))
def test_a_checkpoint_carries_each_sparse_edges_weights_and_its_patterns_digest(name, tmp_path):
    """Beside ``W``, under the same edge: 32 bytes, the SHA-256 of the
    mapping's structure.  Not the index."""
    gm = mapped_pair(CASES[name], base_dir=tmp_path)
    members = _members(save_state(gm, tmp_path / "ck"))
    for key, edge in zip((A2B, B2A), gm.edges):
        assert f"_params_mappings/{key}/W" in members
        digest = members[f"_params_mappings/{key}/{DIGEST}"]
        assert digest.dtype == np.uint8 and digest.shape == (32,)
        assert digest.tobytes().hex() == edge.mapping.structure_digest()
        assert members[f"_params_mappings/{key}/W"].shape == edge.mapping.indices.shape
    mapping_members = sorted(m for m in members if m.startswith("_params_mappings/"))
    assert mapping_members == sorted(
        f"_params_mappings/{key}/{leaf}" for key in (A2B, B2A) for leaf in ("W", DIGEST))
    assert not any(np.issubdtype(m.dtype, np.integer) and m.dtype != np.uint8
                   for name_, m in members.items() if name_.startswith("_params_mappings/"))
    assert _STRUCTURE_MEMBER == DIGEST and not DIGEST.isidentifier()


def test_a_dense_mappings_checkpoint_has_the_members_it_always_had(tmp_path):
    """Nothing is added for a mapping whose weights are the whole operator:
    the built-in dense kinds, and a registered kind of another class."""
    gm = GraphManager()
    gm.add_node(Interface("a", 0.125, n=4))
    gm.add_node(Interface("b", 0.125, n=6))
    xa, xb = np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 6)
    gm.add_edge("a", "b", "x", "inp", mapping=matrix_mapping(np.ones((6, 4), np.float32)))
    gm.add_edge("b", "a", "x", "inp", mapping=KINDS[INVERSE_DISTANCE].build(xb, xa))
    gm.compile()
    members = _members(save_state(gm, tmp_path / "ck"))
    assert sorted(m for m in members if m.startswith("_params_mappings/")) == [
        f"_params_mappings/{A2B}/H", f"_params_mappings/{B2A}/W",
        f"_params_mappings/{B2A}/gain"]
    assert not any(DIGEST in m for m in members)


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_graph_reloads_its_own_checkpoint_and_so_does_its_config(name, tmp_path):
    """The trained weights come back bit for bit into the graph that saved
    them and into one rebuilt from its config: the same points give the
    same pattern, so the same digest."""
    gm = mapped_pair(CASES[name], base_dir=tmp_path)
    moved = _train(gm)
    gm.run(3)
    path = save_state(gm, tmp_path / "ck")
    expected = _snapshot(gm)

    gm.reset_state()
    gm.reset_params()
    load_state(gm, path)
    with pytest.warns(UserWarning, match="live mapping weights"):
        config = json.loads(json.dumps(gm.to_dict()))
    rebuilt = GraphManager.from_dict(config, REGISTRY, base_dir=tmp_path)
    rebuilt.compile()
    assert np.asarray(rebuilt.params["mappings"][A2B]["W"]).tobytes() != \
        moved[A2B]["W"].tobytes()
    load_state(rebuilt, path)
    for graph in (gm, rebuilt):
        for key in (A2B, B2A):
            got = np.asarray(graph.params["mappings"][key]["W"])
            assert got.dtype == moved[key]["W"].dtype
            assert got.tobytes() == moved[key]["W"].tobytes()
        for node in ("a", "b"):
            np.testing.assert_array_equal(np.asarray(graph.get_node_state(node)["x"]),
                                          expected["state"][node]["x"])
    gm.run(2)
    rebuilt.run(2)
    np.testing.assert_array_equal(np.asarray(rebuilt.get_node_state("b")["x"]),
                                  np.asarray(gm.get_node_state("b")["x"]))


# ---------------------------------------------------------------------------
# Another pattern of the same shape
# ---------------------------------------------------------------------------

#: Six targets that each pick another source than the uniform ones do.
OTHER_POINTS = np.linspace(0.0, 1.0, 6)[::-1].copy()


def test_the_two_fixtures_differ_in_pattern_and_in_nothing_a_shape_check_sees():
    """The fixture can express the defect: same edge key, same leaf name,
    shape and dtype -- and, for nearest neighbour, the same weights (all
    one) -- over a different index."""
    a, b = _pair(), _pair(OTHER_POINTS)
    wa, wb = a.params["mappings"][A2B]["W"], b.params["mappings"][A2B]["W"]
    assert wa.shape == wb.shape and wa.dtype == wb.dtype
    np.testing.assert_array_equal(np.asarray(wa), np.asarray(wb))
    ia, ib = a.edges[0].mapping.indices, b.edges[0].mapping.indices
    assert ia.shape == ib.shape and not np.array_equal(ia, ib)
    assert a.edges[0].mapping.structure_digest() != b.edges[0].mapping.structure_digest()


def test_weights_saved_for_another_pattern_of_the_same_shape_are_refused(tmp_path):
    saved = _pair()
    _train(saved)
    saved.run(2)
    path = save_state(saved, tmp_path / "ck")

    other = _pair(OTHER_POINTS)
    other.run(1)
    before = _snapshot(other)
    with pytest.raises(ValueError) as refused:
        load_state(other, path)
    message = str(refused.value)
    for part in ("Checkpoint mapping weights 'W' for edge 'a.x->b.inp'",
                 "were saved for a different sparsity pattern",
                 saved.edges[0].mapping.structure_digest()[:12],
                 other.edges[0].mapping.structure_digest()[:12], "Nothing was loaded."):
        assert part in message, part
    _assert_untouched(other, before)
    # and the graph it was saved from still takes it
    load_state(_pair(), path)


def test_a_refused_pattern_installs_nothing_of_any_other_edge_or_node(tmp_path):
    """Atomic: the dense edge beside the sparse one keeps its weights, the
    nodes keep their parameters and their state."""
    def graph(points_b=None):
        gm = GraphManager()
        gm.add_node(Interface("a", 0.125, n=4))
        gm.add_node(Interface("b", 0.125, n=6, rate=0.25))
        xa = np.linspace(0.0, 1.0, 4)
        xb = np.linspace(0.0, 1.0, 6) if points_b is None else points_b
        gm.add_edge("a", "b", "x", "inp", mapping=sparse_nearest_neighbor_mapping(xa, xb))
        gm.add_edge("b", "a", "x", "inp", mapping=matrix_mapping(
            np.full((4, 6), 0.125, np.float32)))
        gm.compile()
        return gm

    saved = graph()
    saved.params["mappings"][B2A]["H"] = saved.params["mappings"][B2A]["H"] * 3.0
    saved.params["nodes"]["a"]["rate"] = jnp.asarray(0.875, jnp.float32)
    saved.run(2)
    path = save_state(saved, tmp_path / "ck")

    other = graph(OTHER_POINTS)
    before = _snapshot(other)
    with pytest.raises(ValueError, match="a different sparsity pattern"):
        load_state(other, path)
    _assert_untouched(other, before)
    same = graph()
    load_state(same, path)
    np.testing.assert_array_equal(np.asarray(same.params["mappings"][B2A]["H"]),
                                  np.asarray(saved.params["mappings"][B2A]["H"]))
    assert float(same.params["nodes"]["a"]["rate"]) == 0.875


@pytest.mark.parametrize("live_transpose, saved_transpose",
                         [("gather", "scatter"), ("scatter", "gather")])
def test_the_two_conservative_forms_do_not_exchange_weights(live_transpose, saved_transpose,
                                                            tmp_path):
    """One operator in two layouts is two structures: their weights are
    indexed differently (and here differ in shape too, which the shape
    check reports first when it can)."""
    saved = _pair(mode="conservative", transpose=saved_transpose)
    path = save_state(saved, tmp_path / "ck")
    live = _pair(mode="conservative", transpose=live_transpose)
    before = _snapshot(live)
    with pytest.raises(ValueError, match="a different sparsity pattern|has shape"):
        load_state(live, path)
    _assert_untouched(live, before)


def test_the_same_index_in_the_other_layout_is_another_pattern(tmp_path):
    """Hand-built: a square index means one operator read by rows and its
    transpose read by columns.  Same edge, leaf, shape and values."""
    index = np.array([[1, 0], [2, 2], [0, 1]])
    weights = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], np.float32)

    def graph(layout):
        gm = GraphManager()
        gm.add_node(Interface("a", 0.125, n=3))
        gm.add_node(Interface("b", 0.125, n=3))
        extra = {} if layout == "gather" else {"n_target": 3, "layout": "scatter"}
        gm.add_edge("a", "b", "x", "inp", mapping=StaticSparseMapping(
            index, jnp.asarray(weights), n_source=3, **extra))
        gm.compile()
        return gm

    path = save_state(graph("gather"), tmp_path / "ck")
    live = graph("scatter")
    before = _snapshot(live)
    with pytest.raises(ValueError, match="a different sparsity pattern"):
        load_state(live, path)
    _assert_untouched(live, before)
    load_state(graph("gather"), path)


# ---------------------------------------------------------------------------
# Dense weights onto a sparse mapping, and the reverse
# ---------------------------------------------------------------------------

class _DenseW:
    """A mapping class of the caller's own whose one weight is a dense
    matrix called ``W``: the leaf name a sparse mapping uses."""

    kind = "dense_w"
    mode = "consistent"
    spec = None

    def __init__(self, weight):
        self._weight = weight

    n_source = property(lambda self: int(self._weight.shape[1]))
    n_target = property(lambda self: int(self._weight.shape[0]))

    def params_pytree(self):
        return {"W": self._weight}

    def apply(self, field, weights=None, geom=None):
        return (self._weight if weights is None else weights["W"]) @ field

    def apply_T(self, field, weights=None, geom=None):
        return (self._weight if weights is None else weights["W"]).T @ field


def _full_rows(dense: bool) -> GraphManager:
    """``a`` (3) onto ``b`` (4) with a weight ``W`` of shape ``(4, 3)``:
    dense, or sparse with every entry (reversed within each row)."""
    gm = GraphManager()
    gm.add_node(Interface("a", 0.125, n=3))
    gm.add_node(Interface("b", 0.125, n=4))
    weights = jnp.asarray(np.arange(1.0, 13.0, dtype=np.float32).reshape(4, 3) / 16.0)
    mapping = _DenseW(weights) if dense else sparse_matrix_mapping(
        np.tile(np.array([2, 1, 0]), (4, 1)), np.asarray(weights), n_source=3)
    gm.add_edge("a", "b", "x", "inp", mapping=mapping)
    gm.compile()
    return gm


def test_dense_weights_are_refused_on_a_sparse_mapping_of_their_shape(tmp_path):
    """The same key, leaf name, shape and dtype; the dense ``W[i, j]`` is
    the weight of source ``j`` and the sparse one of source ``2 - j``."""
    dense, sparse = _full_rows(True), _full_rows(False)
    assert dense.params["mappings"][A2B]["W"].shape == sparse.params["mappings"][A2B]["W"].shape
    path = save_state(dense, tmp_path / "dense")
    assert not any(DIGEST in member for member in _members(path))
    before = _snapshot(sparse)
    with pytest.raises(ValueError) as refused:
        load_state(sparse, path)
    assert "do not say which sparsity pattern they were saved for" in str(refused.value)
    assert sparse.edges[0].mapping.structure_digest()[:12] in str(refused.value)
    _assert_untouched(sparse, before)


def test_sparse_weights_are_refused_on_a_dense_mapping_of_their_shape(tmp_path):
    dense, sparse = _full_rows(True), _full_rows(False)
    path = save_state(sparse, tmp_path / "sparse")
    before = _snapshot(dense)
    with pytest.raises(ValueError) as refused:
        load_state(dense, path)
    assert "were saved for a sparse mapping" in str(refused.value)
    assert "the mapping on this edge now is not one" in str(refused.value)
    _assert_untouched(dense, before)
    assert isinstance(matrix_mapping(np.eye(2, dtype=np.float32)), StaticLinearMapping)


# ---------------------------------------------------------------------------
# An archive that was edited
# ---------------------------------------------------------------------------

def _edited(path, tmp_path, edit) -> str:
    members = _members(path)
    edit(members)
    out = tmp_path / "edited.npz"
    np.savez(out, **members)
    return str(out)


MEMBER = f"_params_mappings/{A2B}/{DIGEST}"


def _drop(members):
    del members[MEMBER]


def _flip(members):
    digest = members[MEMBER].copy()
    digest[0] ^= 1
    members[MEMBER] = digest


EDITS = {
    "the digest removed": (_drop, "do not say which sparsity pattern"),
    "one bit of the digest flipped": (_flip, "a different sparsity pattern"),
    "a digest of 31 bytes": (lambda m: m.__setitem__(MEMBER, m[MEMBER][:31]),
                             r"is uint8\[31\] data, not a 32-byte digest"),
    "a digest of another dtype": (lambda m: m.__setitem__(MEMBER, m[MEMBER].astype(np.int32)),
                                  r"is int32\[32\] data, not a 32-byte digest"),
    "a digest that is text": (lambda m: m.__setitem__(MEMBER, np.array(["ab"] * 32)),
                              "not a 32-byte digest"),
    "a two-dimensional digest": (lambda m: m.__setitem__(MEMBER, m[MEMBER].reshape(4, 8)),
                                 r"is uint8\[4, 8\] data, not a 32-byte digest"),
}


@pytest.mark.parametrize("name", sorted(EDITS))
def test_an_archive_whose_digest_is_missing_or_not_one_is_refused(name, tmp_path):
    edit, message = EDITS[name]
    gm = _pair()
    path = _edited(save_state(gm, tmp_path / "ck"), tmp_path, edit)
    fresh = _pair()
    before = _snapshot(fresh)
    with pytest.raises(ValueError, match=message):
        load_state(fresh, path)
    _assert_untouched(fresh, before)


def test_a_digest_without_weights_installs_nothing_and_is_not_judged(tmp_path):
    """The digest answers for the weights beside it.  An archive that
    carries none for an edge installs none there, whatever digest it has."""
    gm = _pair()

    def only_the_digest(members):
        del members[f"_params_mappings/{A2B}/W"]
        members[MEMBER] = np.zeros(32, np.uint8)

    path = _edited(save_state(gm, tmp_path / "ck"), tmp_path, only_the_digest)
    fresh = _pair(OTHER_POINTS)
    before = np.asarray(fresh.params["mappings"][A2B]["W"]).copy()
    load_state(fresh, path)
    np.testing.assert_array_equal(np.asarray(fresh.params["mappings"][A2B]["W"]), before)


def test_a_mapping_that_reports_no_valid_digest_cannot_be_saved(tmp_path):
    class Broken(_DenseW):
        def structure_digest(self):
            return "not a digest"

    gm = GraphManager()
    gm.add_node(Interface("a", 0.125, n=3))
    gm.add_node(Interface("b", 0.125, n=4))
    gm.add_edge("a", "b", "x", "inp", mapping=Broken(jnp.ones((4, 3))))
    gm.compile()
    with pytest.raises(ValueError, match="structure_digest\\(\\) must return a SHA-256"):
        save_state(gm, tmp_path / "ck")


def test_any_mapping_class_can_declare_a_structure(tmp_path):
    """The check is the protocol's, not the sparse class's: a mapping of
    another class that defines ``structure_digest()`` is held to it."""
    class Structured(_DenseW):
        def __init__(self, weight, tag):
            super().__init__(weight)
            self._tag = tag

        def structure_digest(self):
            import hashlib
            return hashlib.sha256(self._tag).hexdigest()

    def graph(tag):
        gm = GraphManager()
        gm.add_node(Interface("a", 0.125, n=3))
        gm.add_node(Interface("b", 0.125, n=4))
        gm.add_edge("a", "b", "x", "inp", mapping=Structured(jnp.ones((4, 3)), tag))
        gm.compile()
        return gm

    path = save_state(graph(b"one"), tmp_path / "ck")
    load_state(graph(b"one"), path)
    with pytest.raises(ValueError, match="a different sparsity pattern"):
        load_state(graph(b"two"), path)


# ---------------------------------------------------------------------------
# The REST door
# ---------------------------------------------------------------------------

def test_post_checkpoint_load_refuses_weights_saved_for_another_pattern(tmp_path):
    """``POST /checkpoint/load`` calls ``load_state``: a 400 with the reason,
    and the graph as it was."""
    pytest.importorskip("fastapi", reason="the REST door needs the server extra")
    from maddening.api.server import SimulationServer
    from tests._loopback_client import LoopbackTestClient as TestClient

    saved = _pair()
    _train(saved)
    saved.run(2)
    saved.save_state(str(tmp_path / "trained.npz"))

    def client(gm):
        server = SimulationServer(REGISTRY, graph_manager=gm, checkpoint_root=str(tmp_path))
        return TestClient(server.create_app(), raise_server_exceptions=False)

    other = _pair(OTHER_POINTS)
    before = _snapshot(other)
    refused = client(other).post("/checkpoint/load", params={"path": "trained.npz"})
    assert refused.status_code == 400, refused.text
    detail = refused.json()["detail"]
    assert "were saved for a different sparsity pattern" in detail and A2B in detail
    assert str(tmp_path) not in detail
    _assert_untouched(other, before)

    same = _pair()
    loaded = client(same).post("/checkpoint/load", params={"path": "trained.npz"})
    assert loaded.status_code == 200, loaded.text
    np.testing.assert_array_equal(np.asarray(same.params["mappings"][A2B]["W"]),
                                  np.asarray(saved.params["mappings"][A2B]["W"]))
