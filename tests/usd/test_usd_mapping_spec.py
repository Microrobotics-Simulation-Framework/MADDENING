"""Edge interface mappings round-trip through USD as their MappingSpec
(``maddening:mappingSpecJson``): point references, never weights; asset
references resolve relative to the stage file.

The round trips run over the built-in RBF kind and over each kind
registered the way another library registers one
(``tests/registered_mapping_kinds.py``); a stage, like a config, can only
*name* a kind."""

import json
import sys

import numpy as np
import pytest
from pxr import Usd

from maddening.core.coupling.mapping import matrix_mapping, rbf_mapping
from maddening.core.coupling.mapping_spec import MappingRebuildError, point_array_digest
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.usd.serialization import load_graph_from_usd, save_graph_to_usd
from tests.core.builtin_mapping_pins import PINS
from tests.registered_mapping_kinds import KINDS as REGISTERED_KINDS
from tests.registered_mapping_kinds import assert_same_weights, temporary_kind
from tests.usd.builtin_mapping_usd_pins import capture_usd

C2F = "coarse.temperature->fine.heat_source"

#: ``make(source, target, source_ref=, target_ref=)`` per kind, and the
#: weights each puts into ``params["mappings"]``.
POINT_KINDS = {
    "rbf": lambda source, target, **refs: rbf_mapping(
        source, target, kernel="multiquadric", epsilon=2.5, **refs),
    **{name: kind.build for name, kind in REGISTERED_KINDS.items()},
}
WEIGHTS = {"rbf": ("H",), **{name: kind.weights for name, kind in REGISTERED_KINDS.items()}}


def _node_referenced(kind):
    return lambda g: POINT_KINDS[kind](
        _grid(g, "coarse"), _grid(g, "fine"),
        source_ref={"node": "coarse", "field": "grid_x"},
        target_ref={"node": "fine", "field": "grid_x"})


def _grid(gm, name):
    return gm.get_node(name).static_data["grid_x"].value


def _refs(points):
    """``points`` without the content hashes (asserted separately)."""
    return {n: None if r is None else {k: v for k, v in r.items() if k != "sha256"}
            for n, r in points.items()}


def _rods(mapping_factory):
    gm = GraphManager()
    gm.add_node(HeatNode("coarse", 1e-4, n_cells=6, thermal_diffusivity=0.1))
    gm.add_node(HeatNode("fine", 1e-4, n_cells=12, thermal_diffusivity=0.1))
    gm.add_edge("coarse", "fine", "temperature", "heat_source",
                mapping=mapping_factory(gm))
    gm.compile()
    return gm


@pytest.mark.parametrize("kind", sorted(POINT_KINDS))
def test_node_field_referenced_mapping_round_trips_in_memory(kind):
    gm = _rods(_node_referenced(kind))
    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)

    prim = stage.GetPrimAtPath("/Simulation/edges/e0")
    stored = json.loads(prim.GetAttribute("maddening:mappingSpecJson").Get())
    assert stored["kind"] == kind
    if kind == "rbf":
        assert stored["epsilon"] == 2.5
    assert _refs(stored["points"])["source_points"] == {"node": "coarse", "field": "grid_x"}
    assert stored["points"]["source_points"]["sha256"] == point_array_digest(
        np.asarray(_grid(gm, "coarse")))
    assert not set(WEIGHTS[kind]) & set(stored)
    # the stage and the config write the same thing for the edge
    assert stored == json.loads(json.dumps(gm.to_dict()["edges"][0]["mapping"]))

    gm2 = load_graph_from_usd(stage)
    gm2.compile()
    assert list(gm2.params["mappings"]) == [C2F]
    assert gm2.edges[0].mapping.spec == gm.edges[0].mapping.spec
    assert_same_weights(gm2.params["mappings"][C2F], gm.params["mappings"][C2F])
    np.testing.assert_array_equal(np.asarray(gm2.step()["fine"]["temperature"]),
                                  np.asarray(gm.step()["fine"]["temperature"]))


@pytest.mark.parametrize("kind", sorted(REGISTERED_KINDS))
def test_an_asset_referenced_registered_kind_resolves_relative_to_the_stage_file(
        tmp_path, kind):
    """The stage directory is the ``base_dir`` of a registered kind's
    asset references too, and the reference limits apply there."""
    source = np.linspace(0.0, 1.0, 6)
    target = np.linspace(0.05, 0.95, 12)
    np.save(tmp_path / "source.npy", source)
    np.savez(tmp_path / "grids.npz", target=target, other=source)
    gm = _rods(lambda g: REGISTERED_KINDS[kind].build(
        source, target, source_ref={"asset": "source.npy"},
        target_ref={"asset": "grids.npz", "key": "target"}))
    stage = Usd.Stage.CreateNew(str(tmp_path / "graph.usda"))
    save_graph_to_usd(gm, stage)
    stage.GetRootLayer().Save()

    gm2 = load_graph_from_usd(Usd.Stage.Open(str(tmp_path / "graph.usda")))
    gm2.compile()
    assert_same_weights(gm2.params["mappings"][C2F], gm.params["mappings"][C2F])
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(MappingRebuildError, match="missing point asset 'source.npy'"):
        load_graph_from_usd(Usd.Stage.Open(str(tmp_path / "graph.usda")), base_dir=other)
    np.save(tmp_path / "source.npy", source[::-1].copy())       # changed since the save
    with pytest.raises(MappingRebuildError, match="differs from the points") as changed:
        load_graph_from_usd(Usd.Stage.Open(str(tmp_path / "graph.usda")))
    assert changed.value.kind == kind


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


@pytest.mark.parametrize("kind", [k for k in sorted(POINT_KINDS) if WEIGHTS[k]])
def test_trainable_mapping_param_spec_survives_usd_round_trip(tmp_path, kind):
    """A mapped edge whose weights were made trainable for sysid
    (``set_param_spec(edge.key, "H", ParamSpec())``) keeps that spec
    through a USD round trip, for every weight its kind exposes: the
    override belongs to the edge, so it is written on the edge prim and
    applied after the edge is re-created."""
    from maddening.core.params import ParamSpec

    gm = _rods(_node_referenced(kind))
    for weight in WEIGHTS[kind]:
        gm.set_param_spec(C2F, weight, ParamSpec(trainable=True, description="learned"))

    stage = Usd.Stage.CreateNew(str(tmp_path / "graph.usda"))
    save_graph_to_usd(gm, stage)
    stage.GetRootLayer().Save()

    gm2 = load_graph_from_usd(Usd.Stage.Open(str(tmp_path / "graph.usda")))
    gm2.compile()
    for weight in WEIGHTS[kind]:
        assert gm2.param_spec_overrides()[C2F][weight].trainable is True
        assert gm2.param_specs()["mappings"][C2F][weight].description == "learned"
        assert gm2.trainable_mask()["mappings"][C2F][weight] is True


@pytest.mark.parametrize("kind", sorted(POINT_KINDS))
def test_a_corrupt_mapping_spec_attribute_names_its_edge(kind):
    """A hand-edited / truncated ``maddening:mappingSpecJson`` fails like
    every other rebuild problem: a ``MappingRebuildError`` naming the
    edge, not a bare ``JSONDecodeError``."""
    gm = _rods(_node_referenced(kind))
    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    prim = stage.GetPrimAtPath("/Simulation/edges/e0")
    prim.GetAttribute("maddening:mappingSpecJson").Set('{"kind": "%s", "poi' % kind)

    with pytest.raises(MappingRebuildError) as exc:
        load_graph_from_usd(stage)
    assert exc.value.edge == "coarse.temperature -> fine.heat_source"
    assert isinstance(exc.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize("kind", sorted(POINT_KINDS))
def test_usd_save_refuses_a_reference_to_a_removed_node(kind):
    """The USD writer resolves node references too, so a stale one is
    refused at save time rather than on the next load."""
    gm = GraphManager()
    gm.add_node(HeatNode("coarse", 1e-4, n_cells=6, thermal_diffusivity=0.1))
    gm.add_node(HeatNode("fine", 1e-4, n_cells=12, thermal_diffusivity=0.1))
    gm.add_node(HeatNode("spare", 1e-4, n_cells=6, thermal_diffusivity=0.1))
    # the mapping's source points come from a third node, which then goes
    gm.add_edge("coarse", "fine", "temperature", "heat_source",
                mapping=POINT_KINDS[kind](
                    _grid(gm, "spare"), _grid(gm, "fine"),
                    source_ref={"node": "spare", "field": "grid_x"},
                    target_ref={"node": "fine", "field": "grid_x"}))
    save_graph_to_usd(gm, Usd.Stage.CreateInMemory())          # fine while it exists
    gm.remove_node("spare")
    with pytest.raises(ValueError, match="unknown node 'spare'"):
        save_graph_to_usd(gm, Usd.Stage.CreateInMemory())


# ---------------------------------------------------------------------------
# The built-in kinds write what they wrote before the kinds became a registry
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def usd_pins() -> dict:
    return {"pinned": json.loads(PINS.read_text(encoding="utf-8"))["usd"],
            "captured": capture_usd()}


@pytest.mark.parametrize("graph", ["rods_with_node_references",
                                   "vectors_with_inline_and_asset_references"])
def test_a_built_in_kind_writes_the_stage_attribute_it_wrote_before_the_registry(
        graph, usd_pins):
    """The ``maddening:mappingSpecJson`` text of every mapped edge, and the
    weights a stage rebuilds, against the capture taken on the tree before
    the registry (``tests/usd/builtin_mapping_usd_pins.py``)."""
    pinned, captured = usd_pins["pinned"][graph], usd_pins["captured"][graph]
    assert len(pinned["attributes"]) in (2, 7)
    assert captured["attributes"] == pinned["attributes"]
    assert captured["rebuilt_weights"] == pinned["rebuilt_weights"]


# ---------------------------------------------------------------------------
# A stage can only name a kind
# ---------------------------------------------------------------------------

def _renamed_kind(stage, kind) -> None:
    prim = stage.GetPrimAtPath("/Simulation/edges/e0")
    attr = prim.GetAttribute("maddening:mappingSpecJson")
    attr.Set(json.dumps({**json.loads(attr.Get()), "kind": kind}))


@pytest.mark.parametrize("kind", ["never_registered", "os.system", "usd_kind_sentinel",
                                  "usd_kind_sentinel.build", "usd_kind_sentinel:build",
                                  "tests.registered_mapping_kinds.inverse_distance_mapping"])
def test_a_stage_naming_an_unregistered_kind_is_refused_and_nothing_is_imported(
        tmp_path, monkeypatch, kind):
    """The kind on a stage is a name looked up in the registry.  One that
    spells an importable module, a dotted path or an entry point is
    refused like any unknown name, and the module is not imported."""
    marker = tmp_path / "imported"
    (tmp_path / "usd_kind_sentinel.py").write_text(
        f"open({str(marker)!r}, 'w').write('imported')\n"
        "def build(*args, **kwargs): raise SystemExit(9)\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    gm = _rods(_node_referenced("inverse_distance"))
    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    _renamed_kind(stage, kind)
    with pytest.raises(MappingRebuildError, match="unknown mapping kind") as refused:
        load_graph_from_usd(stage)
    assert refused.value.edge == "coarse.temperature -> fine.heat_source"
    assert refused.value.kind == kind
    assert "'inverse_distance'" in str(refused.value)        # the registered kinds
    assert "usd_kind_sentinel" not in sys.modules and not marker.exists()


def test_a_stage_written_with_a_kind_that_is_no_longer_registered_is_refused():
    """Registered when the stage was written, not in the program loading
    it: the stage carries the name only."""
    registered = REGISTERED_KINDS["linear_1d"]

    def factory(source_points, target_points, *, clamp=True, source_points_ref=None,
                target_points_ref=None):
        from maddening.core.coupling.mapping import StaticLinearMapping
        from maddening.core.coupling.mapping_spec import MappingSpec

        built = registered.factory(source_points, target_points, clamp=clamp,
                                   source_points_ref=source_points_ref,
                                   target_points_ref=target_points_ref)
        spec = MappingSpec("passing_through", built.spec.hyperparameters, built.spec.points)
        return StaticLinearMapping(built.H, kind="passing_through", spec=spec)

    stage = Usd.Stage.CreateInMemory()
    with temporary_kind("passing_through", factory,
                        arrays=("source_points", "target_points"),
                        hyperparameters={"clamp": bool}):
        gm = _rods(lambda g: factory(_grid(g, "coarse"), _grid(g, "fine")))
        save_graph_to_usd(gm, stage)
        reloaded = load_graph_from_usd(stage)
        assert_same_weights(reloaded.edges[0].mapping.params_pytree(),
                            gm.edges[0].mapping.params_pytree())
    with pytest.raises(MappingRebuildError,
                       match="unknown mapping kind 'passing_through'"):
        load_graph_from_usd(stage)
