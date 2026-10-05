"""A write that moves the points an interface mapping was built from is refused.

A mapped edge built from a ``{"node": ..., "field": ...}`` point reference
reads the node's field once, when the mapping is built.  A uniform
``HeatNode`` derives its ``grid_x`` from ``length`` when it is constructed,
and its own step reads ``length`` (``dx = length / n_cells``), so a new
``length`` is used by the rod and not by the mapping.  ``PUT
/graph/params`` answered 200 to one and a ``gm.params`` write ran with it:
the rod stepped on the new length while the mapping interpolated from the
old grid, and the config ``to_dict()`` then wrote did not load
(MADD-ANO-063).  Every door that writes a value into the running graph now
asks the same question -- would a node rebuilt with the value read a
referenced field differently? -- and refuses the value when it would.

What stays open is MADD-ANO-022: a traced or explicit ``params=`` pytree
(``sysid.fit``, ``run_scan(params=...)``) still computes with the
constructor's geometry, because no write is made for anything to refuse.

The question is asked of the mapping's ``MappingSpec``, whatever its kind,
so every test here runs three times: with the built-in RBF mapping it was
written for, with a mapping of a kind registered the way another library
registers one (``tests/registered_mapping_kinds.py``; a class of its own
with two weights), and with a registered kind that has no weights at all
(its entry of ``params["mappings"]`` is ``{}``).
"""

from __future__ import annotations

import json
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.coupling.mapping import rbf_mapping
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from tests.registered_mapping_kinds import INVERSE_DISTANCE, KINDS, SELECTION

EDGE = "a.temperature->b.heat_source"
REGISTRY = {"HeatNode": HeatNode}

#: ``make(source, target, source_ref=, target_ref=)`` for each mapping kind
#: the module runs under.
_MAKERS = {"rbf": rbf_mapping, INVERSE_DISTANCE: KINDS[INVERSE_DISTANCE].build,
           SELECTION: KINDS[SELECTION].build}
_MAKE = {"mapping": rbf_mapping}


@pytest.fixture(autouse=True, params=sorted(_MAKERS))
def mapping_kind(request):
    """Every test in the module, under each mapping kind."""
    _MAKE["mapping"] = _MAKERS[request.param]
    yield request.param
    _MAKE["mapping"] = rbf_mapping


def _rods(*, source_ref=None, source_points=None, target_ref=None, wrap=None,
          mapped=True):
    """Two uniform rods, ``a`` mapped onto ``b`` by an RBF mapping whose point
    sets are, by default, references to each rod's own ``grid_x`` (taken as
    the rod holds it, float32, so the recorded hashes describe the field)."""
    a = HeatNode("a", 0.01, n_cells=6, length=1.0, thermal_diffusivity=0.005,
                 initial_temperature=np.linspace(1.0, 2.0, 6).tolist())
    b = HeatNode("b", 0.01, n_cells=5, length=1.0, thermal_diffusivity=0.005,
                 initial_temperature=0.5)
    gm = GraphManager()
    gm.add_node(wrap(a) if wrap is not None else a)
    gm.add_node(b)
    if mapped:
        gm.add_edge("a", "b", "temperature", "heat_source", mapping=_MAKE["mapping"](
            np.asarray(a.static_data["grid_x"].value) if source_points is None
            else source_points(a),
            np.asarray(b.static_data["grid_x"].value),
            source_ref=source_ref if source_ref is not None else {"node": "a", "field": "grid_x"},
            target_ref=target_ref if target_ref is not None else {"node": "b", "field": "grid_x"}))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*is disconnected.*")
        gm.compile()
    return gm


def _client(gm, tmp_path):
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm,
                              checkpoint_root=str(tmp_path))
    return TestClient(server.create_app(), raise_server_exceptions=False)


def _reloads_and_agrees(gm):
    """The graph's config, through JSON, reloads and steps as the graph does."""
    config = json.loads(json.dumps(gm.to_dict()))
    fresh = GraphManager.from_dict(config, REGISTRY)
    fresh.compile()
    for name in gm.node_names:
        fresh.set_node_state(name, gm.get_node_state(name))
    gm.run(3)
    fresh.run(3)
    for name in gm.node_names:
        for field, value in gm.get_node_state(name).items():
            np.testing.assert_array_equal(np.asarray(value),
                                          np.asarray(fresh.get_node_state(name)[field]))


# ---------------------------------------------------------------------------
# gm.params: refused at the next run, save and checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rod", ["a", "b"])
def test_a_length_written_into_gm_params_under_a_reference_is_refused(rod):
    """The source rod's grid and the target rod's grid alike: each is a
    point set the mapping was built from."""
    gm = _rods()
    gm.run(2)
    before = {n: dict(gm.get_node_state(n)) for n in gm.node_names}
    gm.params["nodes"][rod]["length"] = jnp.float32(1.5)
    for call in (lambda: gm.run(1), lambda: gm.run_scan(1), gm.to_dict,
                 lambda: gm.check_params()):
        with pytest.raises(ValueError, match="differs from the node's own value") as exc:
            call()
        assert EDGE in str(exc.value) and f"{rod}.grid_x" in str(exc.value)
    for name, fields in before.items():
        for field, value in fields.items():
            np.testing.assert_array_equal(np.asarray(gm.get_node_state(name)[field]),
                                          np.asarray(value))
    # Restoring the leaf makes the graph whole again.
    gm.params["nodes"][rod]["length"] = jnp.float32(1.0)
    gm.run(1)


def test_a_save_state_with_a_moved_length_is_refused(tmp_path):
    gm = _rods()
    gm.params["nodes"]["a"]["length"] = jnp.float32(1.5)
    with pytest.raises(ValueError, match="interface mapping on edge"):
        gm.save_state(tmp_path / "c.npz")


def test_a_checkpoint_carrying_another_length_is_refused_after_load_state(tmp_path):
    """A checkpoint is a door into ``gm.params`` too: one taken from a graph
    whose rod was built at another length restores the leaf, and the next
    run refuses it rather than stepping the old mapping on the new rod."""
    path = tmp_path / "c.npz"
    _rods().save_state(path)
    data = dict(np.load(path, allow_pickle=False))
    key = next(k for k in data if k.endswith("a/length"))
    data[key] = np.asarray(1.5, dtype=data[key].dtype)
    np.savez(path, **data)
    gm = _rods()
    gm.load_state(path)
    with pytest.raises(ValueError, match="interface mapping on edge"):
        gm.run(1)


def test_a_value_that_moves_no_referenced_point_set_is_still_taken():
    """The neighbour that must keep working: ``thermal_diffusivity`` on the
    mapped rod is a calibration, not geometry.  Taken, and the save reloads."""
    gm = _rods()
    gm.params["nodes"]["a"]["thermal_diffusivity"] = jnp.float32(0.006)
    gm.run(1)
    _reloads_and_agrees(gm)


def test_a_length_under_an_inline_reference_is_taken():
    """A mapping whose points are inlined states its geometry outright: the
    saved graph rebuilds it from the same points, so a length written into
    the rod reloads as it runs."""
    a = HeatNode("a", 0.01, n_cells=6, length=1.0)
    gm = _rods(source_ref={"inline": np.asarray(a.static_data["grid_x"].value,
                                                np.float64).tolist(),
                           "dtype": "float64"})
    gm.params["nodes"]["a"]["length"] = jnp.float32(1.5)
    gm.run(1)
    _reloads_and_agrees(gm)


def test_a_length_on_an_unmapped_rod_is_taken():
    gm = _rods(mapped=False)
    gm.params["nodes"]["a"]["length"] = jnp.float32(1.5)
    gm.run(1)
    _reloads_and_agrees(gm)


def test_an_explicit_params_pytree_is_not_refused():
    """MADD-ANO-022's residual, pinned: a caller's own ``params=`` is never
    saved, so it is not refused, and it still computes with the mapping's
    constructor geometry."""
    gm = _rods()
    p = {s: {o: dict(v) for o, v in owners.items()} for s, owners in gm.params.items()}
    p["nodes"]["a"]["length"] = jnp.float32(1.5)
    gm.run_scan(2, params=p)


def test_a_sharded_rod_under_a_reference_is_refused_the_same_way(tmp_path):
    """A wrapper is answered for by the node it wraps, rebuilt from the
    params they share (``_params_holders``): the same refusal through
    ``gm.params`` and through the route."""
    from maddening.cloud.multigpu.device_mesh import create_device_mesh
    from maddening.cloud.multigpu.sharded_node import ShardedStencilNode

    mesh = create_device_mesh(shape=(1,))
    gm = _rods(wrap=lambda node: ShardedStencilNode(node, mesh, {"devices": 0}))
    resp = _client(gm, tmp_path).put("/graph/params/a", json={"params": {"length": 1.5}})
    assert resp.status_code == 400, resp.text
    assert EDGE in resp.json()["detail"]
    gm.params["nodes"]["a"]["length"] = jnp.float32(1.5)
    with pytest.raises(ValueError, match="interface mapping on edge"):
        gm.run(1)


# ---------------------------------------------------------------------------
# PUT /graph/params and POST /checkpoint/load
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rod", ["a", "b"])
def test_the_route_refuses_a_length_under_a_reference_and_writes_nothing(rod, tmp_path):
    gm = _rods()
    client = _client(gm, tmp_path)
    before = client.get(f"/graph/params/{rod}").json()
    resp = client.put(f"/graph/params/{rod}", json={"params": {"length": 1.5}})
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert EDGE in detail and f"{rod}.grid_x" in detail and "Nothing was written" in detail
    assert client.get(f"/graph/params/{rod}").json() == before
    assert gm.get_node(rod).params["length"] == 1.0


def test_the_route_still_takes_a_calibration_of_the_mapped_rod(tmp_path):
    gm = _rods()
    client = _client(gm, tmp_path)
    resp = client.put("/graph/params/a", json={"params": {"thermal_diffusivity": 0.006}})
    assert resp.status_code == 200, resp.text
    _reloads_and_agrees(gm)


def test_the_route_refuses_a_parameter_field_a_reference_names(tmp_path):
    """A reference may name an array-valued constructor parameter instead of
    a static (the resolver's second rule): an initial condition the route
    otherwise takes, for the next reset, is refused while a mapping was
    built from it -- the save would no longer describe the mapping."""
    new_profile = np.linspace(0.0, 1.0, 6).tolist()
    gm = _rods(source_ref={"node": "a", "field": "initial_temperature"},
               source_points=lambda a: np.asarray(a.params["initial_temperature"]))
    gm.to_dict()                     # the fixture's reference describes its points
    resp = _client(gm, tmp_path).put(
        "/graph/params/a", json={"params": {"initial_temperature": new_profile}})
    assert resp.status_code == 400, resp.text
    assert "a.initial_temperature" in resp.json()["detail"]
    # Unmapped, the same write is an initial condition the route takes.
    plain = _rods(mapped=False)
    resp = _client(plain, tmp_path).put(
        "/graph/params/a", json={"params": {"initial_temperature": new_profile}})
    assert resp.status_code == 200, resp.text


def test_the_checkpoint_route_refuses_and_undoes_a_moved_length(tmp_path):
    gm = _rods()
    path = tmp_path / "c.npz"
    gm.save_state(path)
    data = dict(np.load(path, allow_pickle=False))
    key = next(k for k in data if k.endswith("a/length"))
    data[key] = np.asarray(1.5, dtype=data[key].dtype)
    np.savez(path, **data)
    gm.run(2)
    client = _client(gm, tmp_path)
    before = client.get("/graph/state").json()
    resp = client.post("/checkpoint/load", params={"path": "c.npz"})
    assert resp.status_code == 400, resp.text
    assert "interface mapping on edge" in resp.json()["detail"]
    assert client.get("/graph/state").json() == before
    assert float(gm.params["nodes"]["a"]["length"]) == 1.0


# ---------------------------------------------------------------------------
# An exported FMU: the leaf is not a tunable parameter
# ---------------------------------------------------------------------------

def test_an_fmu_does_not_export_a_length_a_mapping_was_built_from():
    """An FMU writes its parameters straight into the compiled step's pytree,
    where no later check sees them, so the decision is taken at export: the
    rods' lengths are fixed (with the mapping named), their diffusivities
    stay tunable, and a set of a length is refused."""
    from maddening.fmi.model_description import build_model_description
    from maddening.fmi.sidecar import FmuSidecar, SidecarConfig

    gm = _rods()
    md = build_model_description(gm, model_name="rods")
    for rod in ("a", "b"):
        assert f"{rod}.params.length" in md.fixed_parameters
        assert EDGE in md.fixed_parameters[f"{rod}.params.length"]
    tunable = {v.name for v in md.variables if v.causality == "parameter"}
    assert {"a.params.thermal_diffusivity", "b.params.thermal_diffusivity"} <= tunable
    assert not {"a.params.length", "b.params.length"} & tunable
    sc = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,  # noqa: SLF001
        initial_state=gm._state, params=gm.params,                     # noqa: SLF001
        param_specs=gm.param_specs(), fixed_params=md.fixed_parameters))
    with pytest.raises(ValueError, match="not tunable"):
        sc.set_params({"a.params.length": 1.5})


def test_an_fmu_of_unmapped_rods_still_exports_their_lengths():
    from maddening.fmi.model_description import build_model_description

    md = build_model_description(_rods(mapped=False), model_name="rods")
    tunable = {v.name for v in md.variables if v.causality == "parameter"}
    assert "a.params.length" in tunable
