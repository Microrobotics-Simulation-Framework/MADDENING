"""The mapping kinds are one registry, and ``register_mapping`` opens it.

Four things are held here.

* **Nothing that existed changed.**  What the four built-in kinds write,
  rebuild and refuse was captured on the tree before the registry
  (``tests/core/builtin_mapping_pins.py``) and is compared with what they
  do now, text for text and bit for bit.
* **The registry's own rules**: what a kind name and a declaration may be,
  that a kind cannot be registered twice or a built-in replaced, and that
  nothing public removes one.
* **The trust boundary.**  A file can only name a kind.  A clean process
  that has not imported the registering module refuses the kind, imports
  nothing because of the name, and loads the same config once the program
  itself imports the module.  A kind's hyper-parameters and every array it
  is given are checked before its factory runs; whatever the factory
  raises or returns, the loader names the edge.
* **What a mapping may put into ``params["mappings"]``**, enforced where
  the edge is added, and that everything it allows holds on the readers of
  that entry.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import ast
import dataclasses
import functools
import inspect
import json
import subprocess
import sys
import textwrap
import warnings
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from maddening.core import node as node_module
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.coupling import mapping as mapping_module
from maddening.core.coupling import mapping_registry, mapping_spec
from maddening.core.coupling.mapping import (
    Mapping,
    StaticLinearMapping,
    matrix_mapping,
    nearest_neighbor_mapping,
    projection_1d_mapping,
    rbf_mapping,
    register_mapping,
)
from maddening.core.coupling.mapping_spec import (
    INLINE_ELEMENT_LIMIT,
    INLINE_POINT_LIMIT,
    MappingRebuildError,
    MappingSpec,
    PointReferenceError,
    make_point_resolver,
    point_array_digest,
    reference_for_array,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.core.simulation.checkpoint import load_state, save_state
from maddening.nodes.heat import HeatNode
from tests.core.builtin_mapping_pins import (
    PINS,
    REFUSED_SPECS,
    REGISTRY,
    Vec,
    capture,
)
from tests.registered_mapping_kinds import (
    INVERSE_DISTANCE,
    KINDS,
    LINEAR_1D,
    SELECTION,
    InverseDistanceMapping,
    assert_same_weights,
    temporary_kind,
    weights_of,
)
from tests.sparse_mapping_support import SCATTER, SPARSE_POINT_KINDS

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Where the ``maddening`` under test was imported from: a fresh interpreter
#: is given the same tree, whichever tree this run is testing.
SRC = Path(mapping_registry.__file__).resolve().parents[3]
BUILTIN_KINDS = ["matrix", "nearest_neighbor", "projection_1d", "rbf"]
#: The sparse kinds the library registers itself, through the public
#: ``register_mapping`` (``maddening.core.coupling.sparse_mapping``), and
#: every kind a process has before it registers one of its own.
SPARSE_KINDS = ["sparse_matrix", "sparse_nearest_neighbor", "sparse_projection_1d"]
#: The geometry-dependent reference kind (experimental), registered like
#: the sparse ones through the public door.
GEOMETRY_KINDS = ["multilinear_grid"]
LIBRARY_KINDS = sorted(BUILTIN_KINDS + SPARSE_KINDS + GEOMETRY_KINDS)
EDGE = "a.v -> b.inp"
INLINE3 = {"inline": [0.0, 0.5, 1.0], "dtype": "float64"}
PAIR = {"source_points": INLINE3, "target_points": INLINE3}
REGISTERED = sorted(KINDS)
#: Every kind that maps one point set onto another and is held to what a
#: registered kind is held to: the three ``tests/registered_mapping_kinds``
#: registers, and the library's own ``sparse_nearest_neighbor``.
PAIR_KINDS = {**KINDS, **SPARSE_POINT_KINDS}
PAIRS = sorted(PAIR_KINDS)
#: A valid reference for each array of every such kind and of the two
#: sparse kinds that take other arrays (``rows.npy`` and ``vals.npy`` are
#: in the asset directory of the tests that resolve them).
SPEC_POINTS = {
    **{kind: PAIR for kind in PAIRS},
    "sparse_projection_1d": {"source_boundaries": INLINE3, "target_boundaries": INLINE3},
    "sparse_matrix": {"indices": {"asset": "rows.npy"}, "values": {"asset": "vals.npy"}},
}
EVERY_REGISTERED = sorted(SPEC_POINTS)


def _config(mapping, n_source=3, n_target=3) -> dict:
    return {"nodes": [{"type": "Vec", "name": "a", "timestep": 1.0, "params": {"n": n_source}},
                      {"type": "Vec", "name": "b", "timestep": 1.0, "params": {"n": n_target}}],
            "edges": [{"source_node": "a", "target_node": "b", "source_field": "v",
                       "target_field": "inp", "mapping": mapping}],
            "external_inputs": []}


def _load(mapping, base_dir=None, **sizes) -> GraphManager:
    return GraphManager.from_dict(_config(mapping, **sizes), REGISTRY, base_dir=base_dir)


def _vectors(mapping, n_source=3, n_target=3) -> GraphManager:
    gm = GraphManager()
    gm.add_node(Vec("a", 1.0, n=n_source))
    gm.add_node(Vec("b", 1.0, n=n_target))
    gm.add_edge("a", "b", "v", "inp", mapping=mapping)
    return gm


def _spy(monkeypatch, kind: str) -> list:
    """Record every call of *kind*'s factory for the rest of the test."""
    entry = mapping_registry._MAPPING_REGISTRY[kind]
    calls: list = []

    def recording(*args, **kwargs):
        calls.append((args, kwargs))
        return entry.factory(*args, **kwargs)

    monkeypatch.setitem(mapping_registry._MAPPING_REGISTRY, kind,
                        dataclasses.replace(entry, factory=recording))
    return calls


# ===========================================================================
# 1. Nothing that existed changed
# ===========================================================================

GRAPHS = ("rods_with_node_references", "vectors_with_inline_and_asset_references",
          "vectors_with_every_kernel")


@pytest.fixture(scope="module")
def pinned() -> dict:
    return json.loads(PINS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def captured() -> dict:
    return capture()


def test_the_capture_can_express_a_change_to_any_built_in_kind(pinned):
    """A pin that left a kind, a kernel or a mode out would pass whatever
    happened to it."""
    mappings = [e["mapping"] for g in GRAPHS
                for e in json.loads(pinned[g]["config"])["edges"]]
    labelled = {m["label"] for m in mappings if "label" in m}
    assert {m["kind"] for m in mappings} - labelled == set(BUILTIN_KINDS)
    assert labelled == {"supermesh"}
    assert {m.get("kernel") for m in mappings} >= set(mapping_module._KERNELS)
    assert {m["mode"] for m in mappings} == {"consistent", "conservative"}
    forms = {next(iter(ref)) for m in mappings for ref in m["points"].values()}
    assert forms == {"node", "asset", "inline"}
    weighed = [w for g in GRAPHS[:2] for w in pinned[g]["weights"].values()]
    assert len(weighed) == 9 and all(np.size(w["H"]["values"]) for w in weighed)


@pytest.mark.parametrize("graph", GRAPHS)
def test_a_built_in_kind_writes_the_config_it_wrote_before_the_registry(
        graph, pinned, captured):
    """The JSON text, key order included."""
    assert captured[graph]["config"] == pinned[graph]["config"]


@pytest.mark.parametrize("graph", GRAPHS[:2])
def test_a_built_in_kind_builds_and_rebuilds_the_weights_it_did_before_the_registry(
        graph, pinned, captured):
    """Every weight of every edge, exactly; and ``from_dict`` rebuilds the
    same ones and writes the same config again."""
    for part in ("weights", "rebuilt_weights", "rebuilt_config"):
        assert captured[graph][part] == pinned[graph][part], part
    assert captured[graph]["rebuilt_weights"] == captured[graph]["weights"]
    assert captured[graph]["rebuilt_config"] == captured[graph]["config"]


@pytest.mark.parametrize("name", sorted(REFUSED_SPECS))
def test_a_malformed_built_in_spec_is_refused_in_the_words_it_was_before_the_registry(
        name, pinned, captured):
    assert pinned["refusals"][name]["type"] == "MappingRebuildError"
    assert captured["refusals"][name] == pinned["refusals"][name]


@pytest.mark.parametrize("part", ["unserialisable", "spec_forms"])
def test_the_write_side_refusals_and_the_spec_forms_are_what_they_were(
        part, pinned, captured):
    assert captured[part] == pinned[part]


def test_the_built_in_kinds_are_entries_of_the_registry_and_no_table_is_left():
    """One table: the names the closed tables had are gone, and every
    built-in kind is an entry the spec validation and the rebuild read."""
    for gone in ("_FACTORIES", "_HYPER_TYPES", "_REF_KWARG"):
        assert not hasattr(mapping_spec, gone), gone
    entries = mapping_registry._MAPPING_REGISTRY
    assert [k for k in sorted(entries) if entries[k].builtin] == BUILTIN_KINDS
    assert entries["rbf"].factory is rbf_mapping
    assert entries["nearest_neighbor"].factory is nearest_neighbor_mapping
    assert entries["projection_1d"].factory is projection_1d_mapping
    assert list(entries["rbf"].hyperparameters) == [
        "kernel", "epsilon", "polynomial", "ridge", "mode"]
    assert entries["matrix"].references == {"H": "asset"}
    # the matrix entry calls matrix_mapping with the label as its kind
    labelled = entries["matrix"].factory(np.eye(2, dtype=np.float32), label="supermesh")
    assert labelled.kind == "supermesh" and labelled.spec.kind == "matrix"


# ===========================================================================
# 2. A clean process: only the built-in kinds, and a file that names another
# ===========================================================================

_CLEAN_PROCESS = textwrap.dedent('''
    import importlib, json, os, sys
    payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
    out = {}

    def outcome(fn):
        try:
            fn()
        except Exception as exc:
            return {"type": type(exc).__name__, "message": str(exc)}
        return {"type": None}

    # Only mapping_spec: the built-in kinds must be there without anyone
    # having imported the module that defines their factories.
    from maddening.core.coupling.mapping_spec import MappingSpec
    out["spec_from_mapping_spec_alone"] = MappingSpec(
        "nearest_neighbor", {}, payload["pair"]).kind
    from maddening.core.coupling import mapping_registry
    out["kinds"] = mapping_registry._registered_kinds()
    out["unknown_spec"] = outcome(lambda: MappingSpec("spline", {}, {}))
    for label, kind in payload["not_kinds"].items():
        out["not_a_kind:" + label] = outcome(
            lambda: MappingSpec.from_dict({"kind": kind, "points": {}}))

    from maddening.core.graph_manager import GraphManager
    from maddening.nodes.heat import HeatNode
    registry = {"HeatNode": HeatNode}
    out["unregistered"] = outcome(lambda: GraphManager.from_dict(payload["config"], registry))
    out["add_edge_unregistered"] = outcome(lambda: MappingSpec.from_dict(
        payload["config"]["edges"][0]["mapping"]))
    for label, kind in payload["sentinel_kinds"].items():
        config = json.loads(json.dumps(payload["config"]))
        config["edges"][0]["mapping"]["kind"] = kind
        out["sentinel:" + label] = outcome(lambda: GraphManager.from_dict(config, registry))
    out["imported"] = sorted(m for m in payload["must_not_import"] if m in sys.modules)
    out["marker"] = os.path.exists(payload["marker"])

    # The program imports the module that registers the kind: now it loads.
    importlib.import_module(payload["registering_module"])
    out["kinds_after_import"] = mapping_registry._registered_kinds()
    gm = GraphManager.from_dict(payload["config"], registry)
    import numpy as np
    out["rebuilt"] = {name: [str(np.asarray(leaf).dtype), list(np.shape(leaf)),
                             np.asarray(leaf).tobytes().hex()]
                      for name, leaf in gm.edges[0].mapping.params_pytree().items()}
    out["rebuilt_config"] = json.dumps(gm.to_dict())
    print("RESULT" + json.dumps(out))
''')

#: Module names a config tries to have imported by naming them as a kind.
_SENTINEL_KINDS = {
    "a module name": "mapping_kind_sentinel",
    "a dotted path": "mapping_kind_sentinel.build",
    "an entry-point path": "mapping_kind_sentinel:build",
    "a package path": "mapping_kind_sentinel_pkg.factory.build",
    "the registering module": "tests.registered_mapping_kinds",
    "its factory by dotted path": "tests.registered_mapping_kinds.inverse_distance_mapping",
    "a stdlib callable": "os.system",
}


def _rods(make, **hyper) -> GraphManager:
    gm = GraphManager()
    gm.add_node(HeatNode("coarse", 1e-4, n_cells=6, thermal_diffusivity=0.1,
                         initial_temperature=300.0))
    gm.add_node(HeatNode("fine", 1e-4, n_cells=12, thermal_diffusivity=0.1,
                         initial_temperature=350.0))
    xc = gm.get_node("coarse").static_data["grid_x"].value
    xf = gm.get_node("fine").static_data["grid_x"].value
    gm.add_edge("coarse", "fine", "temperature", "heat_source",
                mapping=make(xc, xf, source_ref={"node": "coarse", "field": "grid_x"},
                             target_ref={"node": "fine", "field": "grid_x"}, **hyper))
    return gm


C2F = "coarse.temperature->fine.heat_source"


@pytest.fixture(scope="module")
def clean_process(tmp_path_factory) -> dict:
    """One fresh interpreter that has imported ``maddening`` and nothing
    that registers a kind, handed a config this process wrote with a
    registered kind.  Its report, and what this process built."""
    tmp = tmp_path_factory.mktemp("clean_process")
    marker = tmp / "sentinel_was_imported"
    body = f"open({str(marker)!r}, 'w').write('imported')\ndef build(*a, **k): raise SystemExit(9)\n"
    (tmp / "mapping_kind_sentinel.py").write_text(body, encoding="utf-8")
    pkg = tmp / "mapping_kind_sentinel_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(body, encoding="utf-8")
    (pkg / "factory.py").write_text(body, encoding="utf-8")
    # ... and installed metadata that advertises it as an entry point, in
    # every group a discovery mechanism might plausibly read.
    dist = tmp / "mapping_kind_sentinel-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: mapping-kind-sentinel\nVersion: 1.0\n", encoding="utf-8")
    (dist / "entry_points.txt").write_text("".join(
        f"[{group}]\ninverse_distance = mapping_kind_sentinel:build\n"
        f"mapping_kind_sentinel = mapping_kind_sentinel:build\n"
        for group in ("maddening.mappings", "maddening.mapping_kinds", "maddening.coupling",
                      "maddening.plugins", "maddening")), encoding="utf-8")

    gm = _rods(KINDS[INVERSE_DISTANCE].build, power=3.5, normalise=False,
               mode="conservative", neighbours=4)
    config = json.loads(json.dumps(gm.to_dict()))
    assert type(config["edges"][0]["mapping"]["neighbours"]) is int
    payload = {
        "config": config, "pair": PAIR, "marker": str(marker),
        "registering_module": "tests.registered_mapping_kinds",
        "sentinel_kinds": _SENTINEL_KINDS,
        "must_not_import": ["tests.registered_mapping_kinds", "tests", "mapping_kind_sentinel",
                            "mapping_kind_sentinel_pkg", "mapping_kind_sentinel_pkg.factory"],
        "not_kinds": {"a list": ["rbf"], "a dict": {"kind": "rbf"}, "a number": 7,
                      "null": None, "a bool": True, "the empty string": "",
                      "a near miss": "RBF"},
    }
    (tmp / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "PYTHONPATH": os.pathsep.join([str(SRC), str(REPO_ROOT), str(tmp)])}
    done = subprocess.run([sys.executable, "-c", _CLEAN_PROCESS, str(tmp / "payload.json")],
                          capture_output=True, text=True, timeout=300, env=env, cwd=tmp)
    assert done.returncode == 0, done.stderr[-4000:]
    report = json.loads(done.stdout.rsplit("RESULT", 1)[1])
    return {"report": report, "config": config, "marker": marker,
            "weights": weights_of(gm.edges[0].mapping)}


def test_a_process_that_registered_nothing_has_exactly_the_kinds_the_library_ships(
        clean_process):
    """The four built-in kinds, the three sparse ones and the
    geometry-dependent one, without anyone having imported the modules
    that define their factories."""
    report = clean_process["report"]
    assert report["spec_from_mapping_spec_alone"] == "nearest_neighbor"
    assert report["kinds"] == LIBRARY_KINDS


def test_the_unknown_kind_refusal_lists_the_librarys_kinds_when_only_they_are_registered(
        clean_process):
    """The message a user saw before the registry, with the sparse kinds
    the library has gained since."""
    assert clean_process["report"]["unknown_spec"] == {
        "type": "ValueError",
        "message": f"unknown mapping kind 'spline'; choose from {LIBRARY_KINDS}"}


@pytest.mark.parametrize("label, shown", [
    ("a list", "['rbf']"), ("a dict", "{'kind': 'rbf'}"), ("a number", "7"),
    ("null", "None"), ("a bool", "True"), ("the empty string", "''"),
    ("a near miss", "'RBF'"),
])
def test_a_kind_that_is_not_a_registered_name_is_an_unknown_kind_whatever_its_type(
        clean_process, label, shown):
    """``kind`` comes from a file: a list or a dict must not come back as
    ``TypeError: unhashable type`` out of the table lookup."""
    assert clean_process["report"]["not_a_kind:" + label] == {
        "type": "ValueError",
        "message": f"unknown mapping kind {shown}; choose from {LIBRARY_KINDS}"}


def test_a_kind_registered_in_another_process_is_refused_naming_the_edge_and_the_kinds(
        clean_process):
    refused = clean_process["report"]["unregistered"]
    assert refused["type"] == "MappingRebuildError"
    assert refused["message"] == (
        "edge coarse.temperature -> fine.heat_source: cannot rebuild interface mapping "
        "(kind 'inverse_distance'): ValueError: unknown mapping kind 'inverse_distance'; "
        f"choose from {LIBRARY_KINDS}")
    assert clean_process["report"]["add_edge_unregistered"]["type"] == "ValueError"


@pytest.mark.parametrize("label", sorted(_SENTINEL_KINDS))
def test_a_kind_that_spells_an_importable_path_is_only_an_unknown_name(
        clean_process, label):
    refused = clean_process["report"]["sentinel:" + label]
    assert refused["type"] == "MappingRebuildError"
    assert f"unknown mapping kind {_SENTINEL_KINDS[label]!r}" in refused["message"]


def test_refusing_a_kind_imports_nothing_the_file_named(clean_process):
    """Not the module that registers the kind, not a module on the path
    whose name the config gave as a kind, not one advertised as an entry
    point under the kind's name."""
    assert clean_process["report"]["imported"] == []
    assert clean_process["report"]["marker"] is False
    assert not clean_process["marker"].exists()


def test_the_same_config_loads_once_the_program_imports_the_registering_module(
        clean_process):
    report = clean_process["report"]
    assert report["kinds_after_import"] == sorted(LIBRARY_KINDS + REGISTERED)
    rebuilt = {name: np.frombuffer(bytes.fromhex(data), dtype=dtype).reshape(shape)
               for name, (dtype, shape, data) in report["rebuilt"].items()}
    assert_same_weights(rebuilt, clean_process["weights"],
                        what="weights rebuilt in another process")
    assert json.loads(report["rebuilt_config"]) == clean_process["config"]


def test_a_checkpoint_names_no_kind_and_loading_one_imports_nothing(
        tmp_path, monkeypatch):
    """A checkpoint carries weights under edge keys and weight names, never
    a kind.  One whose member names spell an importable module -- as an
    edge, as a weight, as a kind-like path -- restores the weights it has
    for this graph's edges, ignores the rest, and imports nothing."""
    marker = tmp_path / "imported"
    (tmp_path / "checkpoint_kind_sentinel.py").write_text(
        f"open({str(marker)!r}, 'w').write('imported')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    gm = _rods(KINDS[INVERSE_DISTANCE].build)
    gm.compile()
    trained = {name: 1.5 * leaf for name, leaf in gm.params["mappings"][C2F].items()}
    gm.params["mappings"][C2F].update(trained)
    path = save_state(gm, tmp_path / "ck")
    with np.load(path, allow_pickle=False) as archive:
        members = {k: archive[k] for k in archive.files}
    weight = members[f"_params_mappings/{C2F}/W"]
    members["_params_mappings/checkpoint_kind_sentinel/W"] = weight
    members[f"_params_mappings/{C2F}/checkpoint_kind_sentinel"] = weight
    members["_params_mappings/checkpoint_kind_sentinel.build/kind"] = np.asarray(1.0)
    members["_params/checkpoint_kind_sentinel/kind"] = np.asarray(1.0)
    np.savez(path, **members)

    fresh = _rods(KINDS[INVERSE_DISTANCE].build)
    load_state(fresh, path)
    assert sorted(fresh.params["mappings"]) == [C2F]
    assert_same_weights(fresh.params["mappings"][C2F],
                        {name: np.asarray(leaf) for name, leaf in trained.items()})
    assert "checkpoint_kind_sentinel" not in sys.modules and not marker.exists()


_IMPORT_MACHINERY = ("importlib", "pkgutil", "runpy", "imp", "zipimport", "pkg_resources",
                     "entry_points", "import_module", "__import__", "exec", "eval",
                     "find_spec", "load_module", "globals", "locals", "vars")


@pytest.mark.parametrize("module", [mapping_registry, mapping_spec],
                         ids=["mapping_registry", "mapping_spec"])
def test_the_kind_lookup_has_no_import_machinery_to_reach(module):
    """Read from the source: neither module names anything that imports,
    discovers or evaluates; every ``getattr`` reads a fixed attribute (the
    old lookup found a factory by a name from the spec, on a module); and
    the only imports inside a function have a fixed, literal target."""
    tree = ast.parse(inspect.getsource(module))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom)}
    assert not (names | imported) & set(_IMPORT_MACHINERY)
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "getattr"):
        assert isinstance(call.args[1], ast.Constant), ast.unparse(call)
    deferred = sorted(
        ast.unparse(n) for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
        for n in ast.walk(f) if isinstance(n, (ast.Import, ast.ImportFrom)))
    assert deferred == {
        mapping_registry: [
            "from maddening.serialization.json_codec import NON_FINITE_TOKENS",
            "import maddening.core.coupling.grid_mapping",
            "import maddening.core.coupling.mapping",
            "import maddening.core.coupling.sparse_mapping",
        ],
        mapping_spec: [
            "from maddening.core.coupling.mapping import Mapping, _params_contract_problem",
        ],
    }[module]


# ===========================================================================
# 3. The registry's own rules
# ===========================================================================

def _scaled_identity(kind: str):
    """A well-behaved factory of *kind*, for registration tests."""
    def factory(source_points, target_points, *, scale=1.0, flip=False, name="x",
                source_points_ref=None, target_points_ref=None):
        spec = MappingSpec(kind, {"scale": float(scale), "flip": bool(flip), "name": name}, {
            "source_points": reference_for_array(source_points, source_points_ref,
                                                 name="source_points"),
            "target_points": reference_for_array(target_points, target_points_ref,
                                                 name="target_points"),
        })
        H = scale * np.eye(np.shape(target_points)[0], np.shape(source_points)[0])
        return StaticLinearMapping(jnp.asarray(H[::-1] if flip else H, jnp.float32),
                                   kind=kind, spec=spec)
    return factory


_factory = _scaled_identity("probe")


_DECLARATION = dict(arrays=("source_points", "target_points"),
                    hyperparameters={"scale": float, "flip": bool, "name": str})


def test_register_mapping_is_experimental_and_returns_the_factory_unchanged():
    assert register_mapping._stability_level is StabilityLevel.EXPERIMENTAL
    assert mapping_module.register_mapping is mapping_registry.register_mapping
    with temporary_kind("probe", _factory, **_DECLARATION) as returned:
        assert returned is _factory
        entry = mapping_registry._MAPPING_REGISTRY["probe"]
        assert entry.factory is _factory and entry.builtin is False
        assert entry.references == {"source_points": "source_points_ref",
                                    "target_points": "target_points_ref"}
    assert "probe" not in mapping_registry._registered_kinds()


def test_the_registry_record_is_a_dataclass_that_can_gain_a_defaulted_field():
    """A later change adds what a kind declares (whether it needs a moving
    geometry, for one).  The record is a frozen dataclass built by keyword,
    so a new field with a default changes no entry and no caller."""
    record = mapping_registry._MappingKind
    assert dataclasses.is_dataclass(record) and record.__dataclass_params__.frozen
    fields = {f.name: f for f in dataclasses.fields(record)}
    assert list(fields) == ["kind", "factory", "arrays", "hyperparameters", "references",
                            "builtin", "needs_geometry"]
    assert fields["builtin"].default is False
    # The field this test anticipated: defaulted, so it changed no entry.
    assert fields["needs_geometry"].default is False
    assert {k for k, e in mapping_registry._MAPPING_REGISTRY.items() if e.needs_geometry} \
        >= set(GEOMETRY_KINDS)
    assert not any(mapping_registry._MAPPING_REGISTRY[k].needs_geometry
                   for k in BUILTIN_KINDS + SPARSE_KINDS)

    @dataclasses.dataclass(frozen=True)
    class Grown(record):
        needs_something_later: bool = False

    for entry in mapping_registry._MAPPING_REGISTRY.values():
        grown = Grown(**{name: getattr(entry, name) for name in fields})
        assert grown.needs_something_later is False
        assert dataclasses.replace(grown, needs_something_later=True).kind == entry.kind
    # every construction in the module is by keyword
    source = inspect.getsource(mapping_registry)
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_MappingKind"]
    assert calls and all(not call.args for call in calls)


def test_the_public_signature_is_the_one_documented():
    signature = inspect.signature(register_mapping)
    assert [(p.name, p.kind.name, p.default is inspect.Parameter.empty)
            for p in signature.parameters.values()] == [
        ("kind", "POSITIONAL_OR_KEYWORD", True),
        ("arrays", "KEYWORD_ONLY", True),
        ("hyperparameters", "KEYWORD_ONLY", True),
        ("references", "KEYWORD_ONLY", False),
        ("needs_geometry", "KEYWORD_ONLY", False),
    ]
    assert signature.parameters["references"].default is None
    assert signature.parameters["needs_geometry"].default is False


@pytest.mark.parametrize("kind", [None, 7, 3.0, True, b"rbf", ["rbf"], ("a",), {"a": 1}, ""])
def test_a_kind_name_must_be_a_non_empty_string(kind):
    with pytest.raises(ValueError, match="a mapping kind must be a non-empty string"):
        register_mapping(kind, **_DECLARATION)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_a_kind_name_cannot_spell_a_non_finite_json_token(token):
    """The serialisers reserve these three strings; a config could not
    carry the kind.  A lookalike is an ordinary name."""
    with pytest.raises(ValueError, match="spells a non-finite JSON token"):
        register_mapping(token, **_DECLARATION)
    with temporary_kind(token.lower(), _factory, **_DECLARATION):
        assert token.lower() in mapping_registry._registered_kinds()


@pytest.mark.parametrize("kind", ["with space", "a.b", "a/b", "Ünïcode", "os.system",
                                  "tests.registered_mapping_kinds", "1st", "a-b"])
def test_any_other_non_empty_string_is_a_kind_name_and_only_a_name(kind):
    """A dotted or path-like name is a dictionary key like any other."""
    factory = _scaled_identity(kind)
    with temporary_kind(kind, factory, **_DECLARATION):
        gm = _vectors(factory([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], scale=2.0))
        config = json.loads(json.dumps(gm.to_dict()))
        assert config["edges"][0]["mapping"]["kind"] == kind
        rebuilt = GraphManager.from_dict(config, REGISTRY).edges[0].mapping
        assert_same_weights(weights_of(rebuilt), weights_of(gm.edges[0].mapping))
    with pytest.raises(MappingRebuildError, match="unknown mapping kind"):
        GraphManager.from_dict(config, REGISTRY)


def test_registering_a_kind_twice_with_another_factory_is_an_error():
    def other(source_points, target_points, *, scale=1.0, flip=False, name="x",
              source_points_ref=None, target_points_ref=None):
        raise AssertionError("never registered, never called")

    with temporary_kind("probe", _factory, **_DECLARATION):
        with pytest.raises(ValueError, match=r"Mapping kind 'probe' is already registered "
                                             r"to .*_scaled_identity.*factory\. Cannot "
                                             r"re-register to .*other\."):
            register_mapping("probe", **_DECLARATION)(other)
        assert mapping_registry._MAPPING_REGISTRY["probe"].factory is _factory


def test_registering_the_same_factory_again_is_a_no_op():
    with temporary_kind("probe", _factory, **_DECLARATION):
        before = mapping_registry._MAPPING_REGISTRY["probe"]
        assert register_mapping("probe", **_DECLARATION)(_factory) is _factory
        assert mapping_registry._MAPPING_REGISTRY["probe"] is before
        # ... the default references spelled out are the same declaration
        register_mapping("probe", **_DECLARATION, references={
            "source_points": "source_points_ref",
            "target_points": "target_points_ref"})(_factory)


@pytest.mark.parametrize("changed", [
    dict(arrays=("target_points", "source_points")),
    dict(hyperparameters={"scale": float, "flip": bool}),
    dict(hyperparameters={"scale": float, "flip": bool, "name": bool}),
    dict(hyperparameters={"flip": bool, "scale": float, "name": str}),
], ids=["array order", "a hyper-parameter fewer", "a type", "hyper-parameter order"])
def test_registering_the_same_factory_with_another_declaration_is_an_error(changed):
    """Keeping the first silently would leave the second caller with
    arguments the rebuild never passes."""
    with temporary_kind("probe", _factory, **_DECLARATION):
        with pytest.raises(ValueError, match="with a different declaration"):
            register_mapping("probe", **{**_DECLARATION, **changed})(_factory)


@pytest.mark.parametrize("kind", BUILTIN_KINDS)
def test_a_built_in_kind_can_never_be_replaced(kind):
    entry = mapping_registry._MAPPING_REGISTRY[kind]
    with pytest.raises(ValueError, match=f"Mapping kind '{kind}' is built in .* and "
                                         f"cannot be replaced"):
        register_mapping(kind, **_DECLARATION)(_factory)
    # ... not even by a factory that takes the built-in's own arguments
    with pytest.raises(ValueError, match="built in"):
        register_mapping(kind, arrays=entry.arrays,
                         hyperparameters={n: t for n, t in entry.hyperparameters.items()
                                          if n != "label"},
                         references=entry.references)(lambda **kwargs: None)
    assert mapping_registry._MAPPING_REGISTRY[kind] is entry
    with pytest.raises(ValueError, match="built in and cannot be removed"):
        mapping_registry._unregister(kind)
    assert mapping_registry._MAPPING_REGISTRY[kind] is entry


@pytest.mark.parametrize("kind", SPARSE_KINDS)
def test_a_sparse_kind_of_the_library_is_a_registered_kind_that_cannot_be_replaced(kind):
    """The sparse kinds go through the public ``register_mapping`` -- so
    everything a registered kind is held to is asked of them at every
    rebuild -- and their names are taken: a second factory is refused."""
    entry = mapping_registry._MAPPING_REGISTRY[kind]
    assert entry.builtin is False
    assert entry.factory.__module__ == "maddening.core.coupling.sparse_mapping"
    assert entry.factory._stability_level is StabilityLevel.EXPERIMENTAL
    assert "label" not in entry.hyperparameters, "'label' names the dense matrix kind"
    with pytest.raises(ValueError, match=f"Mapping kind '{kind}' is already registered to "
                                         f"maddening.core.coupling.sparse_mapping"):
        register_mapping(kind, **_DECLARATION)(_factory)
    assert mapping_registry._MAPPING_REGISTRY[kind] is entry


def test_a_sparse_kinds_name_is_taken_before_any_registration_can_claim_it():
    """In a process that imports the registry and nothing else, the first
    ``register_mapping`` call loads the library's sparse kinds before it
    looks -- and a config naming one loads without an import of its own."""
    code = ("from maddening.core.coupling.mapping_registry import register_mapping\n"
            "try:\n"
            "    register_mapping('sparse_matrix', arrays=('x',), hyperparameters={})"
            "(lambda x, x_ref=None: None)\n"
            "except ValueError as exc:\n"
            "    print('REFUSED', exc)\n"
            "import sys\n"
            "print('TREE IMPORTED', 'scipy.spatial' in sys.modules)\n")
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SRC)}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=300, env=env)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "REFUSED Mapping kind 'sparse_matrix' is already registered to" in done.stdout
    # the k-d tree is imported by the builder that needs it, not with the kinds
    assert done.stdout.strip().splitlines()[-1] == "TREE IMPORTED False"


def test_a_built_in_name_is_taken_before_any_registration_can_claim_it():
    """In a process that imports the registry and nothing else, the first
    ``register_mapping`` call loads the built-in kinds before it looks."""
    code = ("from maddening.core.coupling.mapping_registry import register_mapping\n"
            "try:\n"
            "    register_mapping('rbf', arrays=('x',), hyperparameters={})(lambda x, x_ref=None: None)\n"
            "except ValueError as exc:\n"
            "    print('REFUSED', exc)\n")
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SRC)}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=300, env=env)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "REFUSED Mapping kind 'rbf' is built in" in done.stdout


def test_nothing_public_removes_a_kind():
    assert mapping_registry.__all__ == ["register_mapping"]
    for module in (mapping_registry, mapping_module, mapping_spec):
        public = [n for n in dir(module) if not n.startswith("_")]
        assert not [n for n in public if "unregister" in n.lower() or "remove" in n.lower()
                    or "clear" in n.lower() or "deregister" in n.lower()], module.__name__
    with pytest.raises(KeyError, match="is not registered"):
        mapping_registry._unregister("never_registered")


@pytest.mark.parametrize("declaration, message", [
    (dict(arrays="source_points"), "arrays must be a tuple or list"),
    (dict(arrays=("source points", "target_points")), "arrays must be Python identifiers"),
    (dict(arrays=("source_points", 3)), "arrays must be Python identifiers"),
    (dict(arrays=("source_points", "source_points")), "arrays repeats a name"),
    (dict(arrays=None), "arrays must be a tuple or list"),
    (dict(hyperparameters=("scale",)), "hyperparameters must be a dict"),
    (dict(hyperparameters=None), "hyperparameters must be a dict"),
    (dict(hyperparameters={"scale": "real"}), "choose from str, bool, int and float"),
    (dict(hyperparameters={"scale": "int"}), "choose from str, bool, int and float"),
    (dict(hyperparameters={"scale": None}), "choose from str, bool, int and float"),
    (dict(hyperparameters={"scale": complex}), "choose from str, bool, int and float"),
    (dict(hyperparameters={"scale": np.float64}), "choose from str, bool, int and float"),
    (dict(hyperparameters={"scale": np.int64}), "choose from str, bool, int and float"),
    (dict(hyperparameters={"scale": list}), "choose from str, bool, int and float"),
    (dict(hyperparameters={"scale": bytes}), "choose from str, bool, int and float"),
    (dict(hyperparameters={"sc ale": float}), "hyperparameters must be Python identifiers"),
    (dict(hyperparameters={"kind": str}), r"name\(s\) \['kind'\] are reserved"),
    (dict(hyperparameters={"points": str}), r"name\(s\) \['points'\] are reserved"),
    (dict(hyperparameters={"shape": str}), r"name\(s\) \['shape'\] are reserved"),
    (dict(hyperparameters={"label": str}), r"name\(s\) \['label'\] are reserved"),
    (dict(hyperparameters={"source_points": float}), "both an array and a hyper-parameter"),
    (dict(references={"source_points": "source_points_ref"}),
     "references must name exactly the arrays"),
    (dict(references={"source_points": "a", "target_points": "b", "extra": "c"}),
     "references must name exactly the arrays"),
    (dict(references={"source_points": "ref", "target_points": "ref"}),
     "references repeats a name"),
    (dict(references={"source_points": "scale", "target_points": "target_points_ref"}),
     "both a hyper-parameter and a reference keyword"),
    (dict(references={"source_points": "target_points", "target_points": "x"}),
     "both an array and a reference keyword"),
    (dict(references={"source_points": "not an identifier", "target_points": "x"}),
     "references must be Python identifiers"),
    (dict(references=["source_points_ref", "target_points_ref"]),
     "references must be a dict"),
    (dict(hyperparameters={"scale": float, "flip": bool, "name": str, "absent": float}),
     r"takes no keyword argument\(s\) \['absent'\]"),
    (dict(references={"source_points": "source_ref", "target_points": "target_ref"}),
     r"takes no keyword argument\(s\) \['source_ref', 'target_ref'\]"),
    (dict(arrays=("source_points",)), r"requires argument\(s\) \['target_points'\]"),
], ids=lambda value: value if isinstance(value, str) else None)
def test_a_declaration_the_rebuild_could_not_honour_is_refused_at_registration(
        declaration, message):
    """Each of these would otherwise surface while loading someone's
    config, far from the line that wrote it."""
    with pytest.raises(ValueError, match=message):
        register_mapping("probe", **{**_DECLARATION, **declaration})(_factory)
    assert "probe" not in mapping_registry._registered_kinds()


@pytest.mark.parametrize("not_callable", [None, "inverse_distance_mapping", 3, {"a": 1}])
def test_the_registered_object_must_be_callable(not_callable):
    with pytest.raises(TypeError, match="the factory must be callable"):
        register_mapping("probe", **_DECLARATION)(not_callable)
    assert "probe" not in mapping_registry._registered_kinds()


def test_a_factory_taking_keyword_arguments_freely_or_with_no_arrays_registers():
    """``**kwargs`` takes any declared name, and a kind may take no array
    at all (a mapping fixed by its hyper-parameters)."""
    def free(**kwargs):
        raise AssertionError("not called here")

    def no_arrays(*, size=2.0):
        n = int(size)
        return StaticLinearMapping(jnp.eye(n, dtype=jnp.float32), kind="identity",
                                   spec=MappingSpec("identity", {"size": float(size)}, {}))

    with temporary_kind("free", free, **_DECLARATION):
        pass
    with temporary_kind("identity", no_arrays, arrays=(), hyperparameters={"size": float}):
        gm = _vectors(no_arrays(size=3))
        config = json.loads(json.dumps(gm.to_dict()))
        assert config["edges"][0]["mapping"] == {
            "kind": "identity", "mode": "consistent", "shape": [3, 3], "size": 3.0,
            "points": {}}
        rebuilt = GraphManager.from_dict(config, REGISTRY).edges[0].mapping
        np.testing.assert_array_equal(np.asarray(rebuilt.H), np.eye(3))


#: What the rebuild passes a factory declared with ``_DECLARATION``: every
#: declared name, by keyword.
_DECLARED = ("source_points", "target_points", "scale", "flip", "name",
             "source_points_ref", "target_points_ref")


def _taking_each_name(source_points, target_points, *, scale=1.0, flip=False, name="x",
                      source_points_ref=None, target_points_ref=None):
    return "called"


def _needing_a_tolerance(source_points, target_points, tolerance, **hyper_and_references):
    return "called"


def _without_a_name(source_points, target_points, *, scale=1.0, flip=False,
                    source_points_ref=None, target_points_ref=None):
    return "called"


def _positional_arrays(source_points, target_points, /, *, scale=1.0, flip=False, name="x",
                       source_points_ref=None, target_points_ref=None):
    return "called"


def _positional_arrays_beside_kwargs(source_points, target_points, /, **hyper_and_references):
    return "called"


def _optional_positional_only_scale(scale=1.0, /, source_points=None, target_points=None, *,
                                    flip=False, name="x", source_points_ref=None,
                                    target_points_ref=None):
    return "called"


def _name_is_the_rest_of_the_positionals(source_points, target_points, *name, scale=1.0,
                                         flip=False, source_points_ref=None,
                                         target_points_ref=None):
    return "called"


def _forgetting_wraps(fn):
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)
    return inner


class _FactoryOwner:
    def __init__(self, source_points=None, target_points=None, **hyper_and_references):
        pass

    def method(self, source_points, target_points, **hyper_and_references):
        return "called"

    @classmethod
    def build(cls, source_points, target_points, **hyper_and_references):
        return "called"

    def __call__(self, source_points, target_points, *, scale=1.0, flip=False, name="x",
                 source_points_ref=None, target_points_ref=None):
        return "called"


_POSITIONAL = r"requires positional-only argument\(s\) \['source_points', 'target_points'\]"
#: How a factory may be spelled -> the refusal registering it draws, or
#: None when the rebuild's keyword call binds.
_FACTORY_SPELLINGS = {
    "function": (_taking_each_name, None),
    "var_keyword": (lambda **kwargs: "called", None),
    "callable_object": (_FactoryOwner(), None),
    "bound_method": (_FactoryOwner().method, None),
    "classmethod": (_FactoryOwner.build, None),
    "class": (_FactoryOwner, None),
    "no_wraps_decorator": (_forgetting_wraps(_taking_each_name), None),
    "partial_supplying_the_extra_argument":
        (functools.partial(_needing_a_tolerance, tolerance=1), None),
    "required_argument_undeclared":
        (_needing_a_tolerance, r"requires argument\(s\) \['tolerance'\]"),
    "declared_name_not_taken":
        (_without_a_name, r"takes no keyword argument\(s\) \['name'\]"),
    "required_positional_only": (_positional_arrays, _POSITIONAL),
    "required_positional_only_beside_kwargs": (_positional_arrays_beside_kwargs, _POSITIONAL),
    # A declared name the signature holds where no keyword reaches it: an
    # optional positional-only parameter, and the name of ``*args``.  The
    # keyword rule answered for the name alone, so both registered and
    # failed with a TypeError when a config was loaded.
    "optional_positional_only_declared":
        (_optional_positional_only_scale, r"takes no keyword argument\(s\) \['scale'\]"),
    "declared_name_is_var_positional":
        (_name_is_the_rest_of_the_positionals, r"takes no keyword argument\(s\) \['name'\]"),
}


@pytest.mark.parametrize("spelling", sorted(_FACTORY_SPELLINGS))
def test_registration_refuses_exactly_the_factories_the_rebuild_could_not_call(spelling):
    """However the factory is spelled -- a function, a callable object, a
    bound method, a class, a ``functools.partial``, a decorator without
    ``functools.wraps`` -- it registers when ``factory(**declared)`` binds
    and is refused when it does not.  Each body here is trivial, so a
    ``TypeError`` from the call is a failure to bind.

    A required positional-only argument beside ``**kwargs`` registered
    until the check asked for the required arguments apart from the
    keywords taken: the keyword lands in ``**kwargs`` and the positional
    one is still missing."""
    factory, message = _FACTORY_SPELLINGS[spelling]
    try:
        factory(**dict.fromkeys(_DECLARED))
        binds = True
    except TypeError:
        binds = False
    assert binds is (message is None)
    if message is None:
        with temporary_kind("probe", factory, **_DECLARATION):
            pass
    else:
        try:
            with pytest.raises(ValueError, match=message):
                register_mapping("probe", **_DECLARATION)(factory)
            assert "probe" not in mapping_registry._registered_kinds()
        finally:    # a kind that did register must not reach the next test
            if "probe" in mapping_registry._registered_kinds():
                mapping_registry._unregister("probe")
    assert "probe" not in mapping_registry._registered_kinds()


def test_whether_a_factory_takes_a_declared_name_is_the_packages_one_keyword_rule(monkeypatch):
    """The registry asks ``maddening.core.node._signature_takes_keyword``,
    the rule every optional keyword in the package is probed with, about
    each declared name and obeys the answer: it has no rule of its own
    (``tests/core/test_params_probe_agreement.py`` scans for one)."""
    assert mapping_registry._signature_takes_keyword is node_module._signature_takes_keyword
    asked = []

    def rule(fn, keyword):
        asked.append((fn, keyword))
        return keyword != "scale"

    monkeypatch.setattr(mapping_registry, "_signature_takes_keyword", rule)
    with pytest.raises(ValueError, match=r"takes no keyword argument\(s\) \['scale'\], which "
                                         r"the registration declares \(scale: a hyper-parameter"):
        register_mapping("probe", **_DECLARATION)(_factory)
    assert sorted(asked, key=lambda pair: pair[1]) == [(_factory, name)
                                                       for name in sorted(_DECLARED)]
    assert "probe" not in mapping_registry._registered_kinds()


def test_a_factory_whose_signature_cannot_be_read_is_taken_on_trust():
    """Nothing can be checked at the decorator, so nothing is refused
    there; the rebuild's own call is what fails if the declaration is
    wrong."""
    class Opaque:
        __signature__ = "not a signature"

        def __call__(self, **kwargs):
            return "called"

    opaque = Opaque()
    with pytest.raises((TypeError, ValueError)):
        inspect.signature(opaque)
    with temporary_kind("probe", opaque, **_DECLARATION):
        assert mapping_registry._MAPPING_REGISTRY["probe"].factory is opaque
    assert "probe" not in mapping_registry._registered_kinds()


# ===========================================================================
# 4. A registered kind's spec is checked before its factory runs
# ===========================================================================

#: An integer a JSON config can carry and no float64 can hold.
_HUGE = 10 ** 400

#: Values no declaration of that type accepts, with the refusal each draws.
_WRONG = {
    float: [("abc", "must be a real number"), (None, "must be a real number"),
            (True, "must be a real number"), ([1.0], "must be a real number"),
            ({"v": 1.0}, "must be a real number"), (float("inf"), "must be finite"),
            (float("-inf"), "must be finite"), (float("nan"), "must be finite"),
            (_HUGE, "must be finite, got an integer of 401 digits"),
            (-_HUGE, "must be finite, got an integer of 401 digits")],
    int: [("3", "must be an integer"), (None, "must be an integer"),
          (True, "must be an integer"), (False, "must be an integer"),
          (2.0, "must be an integer"), (2.5, "must be an integer"),
          ([2], "must be an integer"), ({"v": 2}, "must be an integer"),
          (float("inf"), "must be an integer"), (float("nan"), "must be an integer"),
          (np.float32(2.0), "must be an integer"),
          (_HUGE, "must be an integer of a magnitude float64 can hold, got one of 401"),
          (-_HUGE, "must be an integer of a magnitude float64 can hold, got one of 401")],
    bool: [(1, "must be a bool"), (0, "must be a bool"), ("true", "must be a bool"),
           (None, "must be a bool"), (1.0, "must be a bool"), ([True], "must be a bool")],
    str: [(3, "must be a str"), (None, "must be a str"), (True, "must be a str"),
          (["a"], "must be a str"), (2.5, "must be a str")],
}

_HYPER_CASES = [
    pytest.param(kind, name, value, message,
                 id=f"{kind}.{name}={str(value)[:12]!r}" if isinstance(value, int)
                 and not isinstance(value, bool) else f"{kind}.{name}={value!r}")
    for kind in EVERY_REGISTERED
    for name, declared in mapping_registry._MAPPING_REGISTRY[kind].hyperparameters.items()
    for value, message in _WRONG[declared]
]


@pytest.mark.parametrize("kind, name, value, message", _HYPER_CASES)
def test_a_registered_hyper_parameter_of_the_wrong_type_is_refused_before_the_factory(
        monkeypatch, kind, name, value, message):
    """Every hyper-parameter of every registered kind, against every value
    its declared type refuses: the loader names the edge and the kind, and
    the factory never sees the value."""
    calls = _spy(monkeypatch, kind)
    with pytest.raises(MappingRebuildError, match=message) as refused:
        _load({"kind": kind, name: value, "points": SPEC_POINTS[kind]})
    assert refused.value.edge == EDGE and refused.value.kind == kind
    assert f"mapping kind {kind!r}: hyper-parameter {name!r}" in str(refused.value)
    assert isinstance(refused.value.__cause__, ValueError)
    assert calls == []
    with pytest.raises(ValueError, match=message):
        MappingSpec(kind, {name: value}, SPEC_POINTS[kind])


def test_the_hyper_parameter_battery_covers_every_declared_type():
    declared = {t for k in REGISTERED
                for t in mapping_registry._MAPPING_REGISTRY[k].hyperparameters.values()}
    assert declared == {str, bool, int, float} == set(_WRONG)
    assert declared == set(mapping_registry._HYPERPARAMETER_TYPES)
    # ... and every hyper-parameter of every sparse kind is in it
    battery = {(case.values[0], case.values[1]) for case in _HYPER_CASES}
    entries = mapping_registry._MAPPING_REGISTRY
    assert {(kind, name) for kind in SPARSE_KINDS for name in entries[kind].hyperparameters} \
        == {("sparse_matrix", "n_source"), ("sparse_matrix", "mode"), ("sparse_matrix", "name"),
            ("sparse_nearest_neighbor", "mode"), ("sparse_nearest_neighbor", "transpose")} \
        <= battery
    assert entries["sparse_matrix"].hyperparameters["n_source"] is int


@pytest.mark.parametrize("kind", REGISTERED)
@pytest.mark.parametrize("given, stored", [(2, 2.0), (np.float32(1.5), 1.5),
                                           (np.int64(3), 3.0), (0.25, 0.25)])
def test_a_real_hyper_parameter_takes_any_finite_number_and_stores_a_float(
        kind, given, stored):
    reals = [n for n, t in mapping_registry._MAPPING_REGISTRY[kind].hyperparameters.items()
             if t is float]
    for name in reals:
        value = MappingSpec(kind, {name: given}, PAIR).hyperparameters[name]
        assert value == stored and type(value) is float


@pytest.mark.parametrize("given, stored", [
    (2, 2), (0, 0), (-3, -3), (np.int32(4), 4), (np.uint8(5), 5), (2 ** 63, 2 ** 63),
    (2 ** 1023, 2 ** 1023),
], ids=["int", "zero", "negative", "numpy int32", "numpy uint8", "past int64",
        "the largest power of two a float64 holds"])
def test_an_integer_hyper_parameter_takes_an_int_and_stores_an_int(given, stored):
    """An ``int``-declared hyper-parameter: any integer that is not a bool,
    up to the magnitude a float64 can hold (the bound a real has), kept
    exactly as a Python ``int`` through the spec and its JSON."""
    spec = MappingSpec(INVERSE_DISTANCE, {"neighbours": given}, PAIR)
    value = spec.hyperparameters["neighbours"]
    assert value == stored and type(value) is int
    back = MappingSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert back == spec and type(back.hyperparameters["neighbours"]) is int
    assert back.hyperparameters["neighbours"] == stored
    back = MappingSpec.from_dict(yaml.safe_load(yaml.safe_dump(spec.to_dict())))
    assert back.hyperparameters["neighbours"] == stored


def test_an_integer_hyper_parameter_reaches_the_factory_and_the_config_as_an_int(
        monkeypatch):
    calls = _spy(monkeypatch, INVERSE_DISTANCE)
    gm = _load({"kind": INVERSE_DISTANCE, "neighbours": 2, "points": PAIR})
    (_, kwargs), = calls
    assert kwargs["neighbours"] == 2 and type(kwargs["neighbours"]) is int
    stored = json.loads(json.dumps(gm.to_dict()))["edges"][0]["mapping"]["neighbours"]
    assert stored == 2 and type(stored) is int
    # ... and it changed what was built: two sources per target, not three
    assert int(np.count_nonzero(np.asarray(gm.edges[0].mapping.W)[0])) == 2


@pytest.mark.parametrize("kind, hyper", [("rbf", "epsilon"), ("rbf", "ridge"),
                                         (INVERSE_DISTANCE, "power")])
@pytest.mark.parametrize("value", [_HUGE, -_HUGE, 2 ** 1024])
def test_an_integer_no_float64_holds_is_a_refused_real_naming_the_edge(kind, hyper, value):
    """JSON integers have no size limit.  One too large to be a float used
    to leave ``MappingSpec`` as an ``OverflowError`` -- which no loader
    wraps, so for a built-in kind the edge was never named.  It is "must
    be finite" now, for a built-in kind and a registered one alike."""
    config = json.loads(json.dumps(_config({"kind": kind, hyper: value, "points": PAIR})))
    with pytest.raises(MappingRebuildError, match="must be finite, got an integer of "
                                                  r"\d+ digits") as refused:
        GraphManager.from_dict(config, REGISTRY)
    assert refused.value.edge == EDGE and refused.value.kind == kind
    assert type(refused.value.__cause__) is ValueError
    with pytest.raises(ValueError, match="must be finite"):
        MappingSpec(kind, {hyper: value}, PAIR)


@pytest.mark.parametrize("argument", ["epsilon", "ridge"])
def test_the_rbf_factory_itself_refuses_an_integer_no_float64_holds(argument):
    with pytest.raises(ValueError, match=f"{argument} must be finite, got an integer of "
                                         "401 digits"):
        rbf_mapping([0.0, 0.5, 1.0], [0.0, 1.0], **{argument: _HUGE})


@pytest.mark.parametrize("kind", EVERY_REGISTERED)
def test_an_unknown_hyper_parameter_of_a_registered_kind_is_refused_before_the_factory(
        monkeypatch, kind):
    calls = _spy(monkeypatch, kind)
    takes = list(mapping_registry._MAPPING_REGISTRY[kind].hyperparameters)
    for stray in ("sigma", "epsilon", "label_text", "kernel", "label", "k", "layout"):
        with pytest.raises(MappingRebuildError) as refused:
            _load({"kind": kind, stray: 2.0, "points": SPEC_POINTS[kind]})
        assert (f"mapping kind {kind!r} has no hyper-parameter(s) [{stray!r}]; it takes "
                f"{takes}") in str(refused.value)
        assert refused.value.edge == EDGE and refused.value.kind == kind
    assert calls == []


@pytest.mark.parametrize("kind", PAIRS)
@pytest.mark.parametrize("points, message", [
    ({"source_points": INLINE3}, "takes point sets"),
    ({"source_points": INLINE3, "target_points": INLINE3, "H": INLINE3}, "takes point sets"),
    ({}, "takes point sets"),
    ({"source_boundaries": INLINE3, "target_boundaries": INLINE3}, "takes point sets"),
    ({"source_points": None, "target_points": INLINE3},
     r"has no reference for \['source_points'\]"),
    ({"source_points": INLINE3, "target_points": None},
     r"has no reference for \['target_points'\]"),
    ({"source_points": None, "target_points": None},
     r"has no reference for \['source_points', 'target_points'\]"),
], ids=["one missing", "one extra", "none", "another kind's", "source not recorded",
        "target not recorded", "neither recorded"])
def test_a_registered_spec_with_the_wrong_or_missing_point_sets_is_refused_before_the_factory(
        monkeypatch, kind, points, message):
    calls = _spy(monkeypatch, kind)
    with pytest.raises(MappingRebuildError, match=message) as refused:
        _load({"kind": kind, "points": points})
    assert refused.value.edge == EDGE and refused.value.kind == kind
    assert calls == []


@pytest.mark.parametrize("kind", PAIRS)
def test_a_missing_reference_names_the_keyword_the_kind_declared(kind):
    """The refusal of a point set that was never recorded says which
    factory argument records it, from the kind's own declaration."""
    described = PAIR_KINDS[kind]
    big = np.linspace(0.0, 1.0, INLINE_POINT_LIMIT + 1)
    mapping = described.build(big, [0.0, 0.5, 1.0])
    assert mapping.spec.missing_points() == ["source_points"]
    with pytest.raises(PointReferenceError, match=f"without {described.source_ref}=$"):
        mapping.spec.build(make_point_resolver())
    gm = _vectors(mapping, n_source=big.size)
    with pytest.raises(ValueError, match=f"Pass {described.source_ref}= to the factory"):
        gm.to_dict()
    # the display writer still describes it
    assert gm.to_dict(strict_mappings=False)["edges"][0]["mapping"]["kind"] == kind


@pytest.mark.parametrize("kind", PAIRS)
@pytest.mark.parametrize("mapping, message", [
    ({"mode": "consistent"}, "has no 'points'"),
    ({"points": [INLINE3, INLINE3]}, "'points' must be a dict"),
    ({"points": PAIR, "shape": [3, 9]}, r"recorded \[3, 9\]"),
    ({"points": PAIR, "shape": 3}, "two-element list of ints"),
    ({"points": PAIR, "shape": [3.0, 3]}, "two-element list of ints"),
    ({"points": PAIR, "shape": [True, 3]}, "two-element list of ints"),
], ids=["no points", "points a list", "another shape", "shape a number",
        "shape of floats", "shape of bools"])
def test_a_malformed_registered_spec_names_its_edge_like_a_built_in_one(
        kind, mapping, message):
    with pytest.raises(MappingRebuildError, match=message) as refused:
        _load({"kind": kind, **mapping})
    assert refused.value.edge == EDGE and refused.value.kind == kind
    assert refused.value.__cause__ is not None


# ---------------------------------------------------------------------------
# The reference limits sit in the resolver, so they hold for every kind
# ---------------------------------------------------------------------------

def _forged_npy(path: Path, shape: tuple, dtype="<f8") -> None:
    """A ``.npy`` that is only a header claiming ``shape``."""
    with open(path, "wb") as fp:
        np.lib.format.write_array_header_1_0(
            fp, {"descr": dtype, "fortran_order": False, "shape": shape})


def _asset_fixtures(tmp: Path) -> None:
    outside = tmp.parent / (tmp.name + "_outside")
    outside.mkdir()
    np.save(outside / "secret.npy", np.array([0.0, 0.5, 1.0]))
    (tmp / "link.npy").symlink_to(outside / "secret.npy")
    (tmp / "dirlink").symlink_to(outside, target_is_directory=True)
    _forged_npy(tmp / "huge.npy", (100_000_000_000,))
    _forged_npy(tmp / "short.npy", (1000,))
    np.save(tmp / "words.npy", np.array(["a", "b", "c"]))
    np.save(tmp / "complex.npy", np.array([0j, 0.5j, 1j]))
    np.save(tmp / "objects.npy", np.array([{"a": 1}, None, 3], dtype=object),
            allow_pickle=True)
    np.save(tmp / "other.npy", np.array([0.0, 0.25, 1.0]))
    np.save(tmp / "rows.npy", np.array([[0], [1], [2]]))
    np.save(tmp / "vals.npy", np.ones((3, 1), np.float32))
    np.savez(tmp / "two.npz", a=np.array([0.0, 0.5, 1.0]), b=np.array([0.0, 0.5, 1.0]))
    (tmp / "bad.npz").write_bytes(b"not a zip archive at all")
    (tmp / "loop.npy").symlink_to(tmp / "loop.npy")
    os.mkfifo(tmp / "pipe.npy")
    if np.dtype(np.longdouble).itemsize > 8:
        np.save(tmp / "extended.npy", np.array([0.0, 0.5, 1.0], dtype=np.longdouble))


_GOOD_DIGEST = point_array_digest(np.array([0.0, 0.5, 1.0]))

#: A reference the resolver refuses, and the words it uses.
_REFUSED_REFERENCES = {
    "asset past the size cap": ({"asset": "huge.npy"}, "MAX_ASSET_BYTES"),
    "asset shorter than its header": ({"asset": "short.npy"}, "the file holds only"),
    "asset of strings": ({"asset": "words.npy"}, "not a bool / integer / float"),
    "asset of complex numbers": ({"asset": "complex.npy"}, "not a bool / integer / float"),
    "asset of pickled objects": ({"asset": "objects.npy"}, "not a bool / integer / float"),
    "asset through a symlink that leaves": ({"asset": "link.npy"},
                                            "outside the config directory"),
    "asset through a directory symlink that leaves": ({"asset": "dirlink/secret.npy"},
                                                      "outside the config directory"),
    "asset by absolute path": ({"asset": "/etc/passwd.npy"}, "must be relative"),
    "asset by a parent path": ({"asset": "../escape.npy"}, r"must not contain '\.\.'"),
    "asset that is missing": ({"asset": "missing.npy"}, "missing point asset"),
    "asset that is a symlink loop": ({"asset": "loop.npy"}, "missing point asset"),
    "asset that is a pipe": ({"asset": "pipe.npy"}, "is not a file"),
    "asset with a NUL in its name": ({"asset": "nul\x00.npy"}, "cannot rebuild"),
    "asset of another suffix": ({"asset": "points.txt"}, "must be a .npy or .npz file"),
    "archive that is not a zip": ({"asset": "bad.npz", "key": "a"}, "not a valid .npz"),
    "archive member not named": ({"asset": "two.npz"}, "add 'key' to the reference"),
    "archive member that is absent": ({"asset": "two.npz", "key": "c"}, "has no member"),
    "asset that changed since the save": ({"asset": "other.npy", "sha256": _GOOD_DIGEST},
                                          "differs from the points"),
    "a malformed hash": ({"asset": "other.npy", "sha256": "abc"},
                         "64-character lowercase hex"),
    "inline past the point limit": ({"inline": [0.0] * (INLINE_POINT_LIMIT + 1)},
                                    "exceed INLINE_POINT_LIMIT"),
    "inline past the element limit": (
        {"inline": [[0.0] * (INLINE_ELEMENT_LIMIT // 2 + 1)] * 2}, "INLINE_ELEMENT_LIMIT"),
    "inline holding a NaN": ({"inline": [0.0, float("nan"), 1.0]}, "must be finite"),
    "inline holding an infinity": ({"inline": [0.0, float("inf"), 1.0]}, "must be finite"),
    "inline of a complex dtype": ({"inline": [0.0, 0.5, 1.0], "dtype": "complex128"},
                                  "not a bool / integer / float"),
    "inline of a string dtype": ({"inline": [0.0, 0.5, 1.0], "dtype": "U3"},
                                 "not a bool / integer / float"),
    "inline of an object dtype": ({"inline": [0.0, 0.5, 1.0], "dtype": "object"},
                                  "not a bool / integer / float"),
    "inline holding complex values": ({"inline": [0.0, 0.5 + 1j, 1.0]},
                                      "hold complex values"),
    "inline holding an inexact integer": ({"inline": [0, 1, 2 ** 53 + 1]},
                                          "cannot represent exactly"),
    "inline that is a scalar": ({"inline": 3.0}, "must be a list"),
    "inline with a stray key": ({"inline": [0.0, 0.5, 1.0], "weights": [1.0]},
                                "an inline reference is"),
    "a node that does not exist": ({"node": "zed", "field": "v"}, "unknown node 'zed'"),
    "a node field that does not exist": ({"node": "a", "field": "grid"},
                                         "no point field 'grid'"),
    "a node reference without a field": ({"node": "a"}, "a node reference is"),
    "an unknown reference form": ({"mesh": "a"}, "unknown point reference"),
    "a reference that is a number": (7, "must be a dict"),
}
if np.dtype(np.longdouble).itemsize > 8:
    _REFUSED_REFERENCES["asset of extended precision"] = (
        {"asset": "extended.npy"}, "extended-precision")
    _REFUSED_REFERENCES["inline of an extended dtype"] = (
        {"inline": [0.0, 0.5, 1.0], "dtype": "longdouble"}, "extended-precision")


@pytest.fixture(scope="module")
def asset_dir(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("config_dir")
    _asset_fixtures(tmp)
    return tmp


#: Every array of the control (``nearest_neighbor``) and of every kind held
#: to the registry's contract, the three sparse kinds' included.
_REFERENCE_SLOTS = [pytest.param(kind, array, id=f"{kind}.{array}")
                    for kind in ["nearest_neighbor", *EVERY_REGISTERED]
                    for array in SPEC_POINTS.get(kind, PAIR)]


@pytest.mark.parametrize("kind, array", _REFERENCE_SLOTS)
@pytest.mark.parametrize("case", sorted(_REFUSED_REFERENCES))
def test_every_reference_limit_holds_for_a_registered_kind_before_its_factory_runs(
        monkeypatch, asset_dir, case, kind, array):
    """The limits on references -- size, dtype, containment, content hash,
    the inline limits -- are the resolver's, so a registered kind gets
    them for each of its arrays exactly as a built-in one does
    (``nearest_neighbor`` is the control), and its factory is not called
    with anything the resolver refused.  The boundary arrays of the sparse
    projection and the index and value arrays of ``sparse_matrix`` are
    arrays like any other."""
    reference, message = _REFUSED_REFERENCES[case]
    calls = _spy(monkeypatch, kind)
    valid = SPEC_POINTS.get(kind, PAIR)
    with pytest.raises(MappingRebuildError, match=message) as refused:
        _load({"kind": kind, "n_source": 3, "points": {**valid, array: reference}}
              if kind == "sparse_matrix" else
              {"kind": kind, "points": {**valid, array: reference}}, base_dir=asset_dir)
    assert refused.value.edge == EDGE and refused.value.kind == kind
    assert isinstance(refused.value.__cause__, (ValueError, OSError))
    assert calls == []


def test_the_valid_references_of_the_reference_battery_build_every_kind(asset_dir):
    """The control: with none of its references replaced, each spec the
    battery above starts from loads -- so each refusal is the replaced
    reference's."""
    assert len(_REFERENCE_SLOTS) == 2 * (1 + len(EVERY_REGISTERED))
    for kind in EVERY_REGISTERED:
        extra = {"n_source": 3} if kind == "sparse_matrix" else {}
        sizes = {"n_source": 2} if kind == "sparse_projection_1d" else {}
        if kind == "sparse_projection_1d":
            sizes["n_target"] = 2
        gm = _load({"kind": kind, **extra, "points": SPEC_POINTS[kind]}, base_dir=asset_dir,
                   **sizes)
        assert gm.edges[0].mapping.kind == kind


@pytest.mark.parametrize("kind", ["nearest_neighbor", *PAIRS])
def test_a_compressed_asset_past_the_cap_is_refused_before_decompression_for_any_kind(
        monkeypatch, tmp_path, kind):
    np.savez_compressed(tmp_path / "bomb.npz", pts=np.zeros(200_000, np.float64))
    monkeypatch.setattr(mapping_spec, "MAX_ASSET_BYTES", 8192)
    calls = _spy(monkeypatch, kind)
    with pytest.raises(MappingRebuildError, match="refused before decompression"):
        _load({"kind": kind, "points": {**PAIR, "source_points": {
            "asset": "bomb.npz", "key": "pts"}}}, base_dir=tmp_path)
    assert calls == []


@pytest.mark.parametrize("kind", PAIRS)
def test_a_registered_factory_is_called_with_checked_arrays_hyper_parameters_and_references(
        monkeypatch, tmp_path, kind):
    """What the factory is handed: ``factory(**arrays, **hyperparameters,
    **references)``, the arrays being the ones the references resolve to."""
    described = PAIR_KINDS[kind]
    source = np.array([0.0, 0.4, 1.0])
    np.save(tmp_path / "source.npy", source)
    hyper = {name: values[-1] for name, values in described.hyper.items()}
    calls = _spy(monkeypatch, kind)
    _load({"kind": kind, **hyper, "points": {
        "source_points": {"asset": "source.npy"}, "target_points": INLINE3}},
        base_dir=tmp_path)
    (args, kwargs), = calls
    assert args == ()
    assert set(kwargs) == {"source_points", "target_points", described.source_ref,
                           described.target_ref, *hyper}
    np.testing.assert_array_equal(kwargs["source_points"], source)
    np.testing.assert_array_equal(kwargs["target_points"], [0.0, 0.5, 1.0])
    assert kwargs[described.source_ref] == {"asset": "source.npy"}
    assert kwargs[described.target_ref] == INLINE3
    assert {name: kwargs[name] for name in hyper} == hyper
    assert {name: type(kwargs[name]) for name in hyper} == {
        name: type(value) for name, value in hyper.items()}


# ===========================================================================
# 5. Whatever a registered factory raises or returns, the loader names the edge
# ===========================================================================

class _Boom(Exception):
    """An exception type no built-in factory raises."""


_RAISED = [
    RuntimeError("solver did not converge"), ZeroDivisionError("division by zero"),
    AttributeError("'NoneType' object has no attribute 'shape'"),
    IndexError("index 3 is out of bounds"), AssertionError("unreachable"),
    LookupError("no such kernel"), NotImplementedError("3-D point sets"),
    StopIteration(), FloatingPointError("underflow"), ArithmeticError("overflow"),
    RecursionError("maximum recursion depth exceeded"), UnicodeError("bad name"),
    np.linalg.LinAlgError("Singular matrix"), _Boom("anything at all"),
    # ... and the types the built-in factories raise, for a registered kind
    ValueError("power must be positive"), TypeError("bad operand"), KeyError("missing"),
    OSError("disk went away"), MemoryError("out of memory"),
    zipfile.BadZipFile("not a zip"), ImportError("cannot import name 'cKDTree'"),
    ModuleNotFoundError("No module named 'scipy'"), OverflowError("int too large"),
]


@pytest.mark.parametrize("raised", _RAISED, ids=lambda exc: type(exc).__name__)
def test_any_exception_from_a_registered_factory_is_a_rebuild_error_naming_the_edge(raised):
    def failing(source_points, target_points, *, source_points_ref=None,
                target_points_ref=None):
        raise raised

    with temporary_kind("failing", failing, arrays=("source_points", "target_points"),
                        hyperparameters={}):
        with pytest.raises(MappingRebuildError) as wrapped:
            _load({"kind": "failing", "points": PAIR})
        assert wrapped.value.edge == EDGE and wrapped.value.kind == "failing"
        assert wrapped.value.__cause__ is raised and wrapped.value.cause is raised
        assert str(wrapped.value).startswith(
            f"edge {EDGE}: cannot rebuild interface mapping (kind 'failing'): "
            f"{type(raised).__name__}")
        # built directly (add_edge with a spec), the factory's own exception
        with pytest.raises(type(raised)) as direct:
            _vectors(MappingSpec("failing", {}, PAIR))
        assert direct.value is raised


@pytest.mark.parametrize("missing", [ImportError("cannot import name 'cKDTree'"),
                                     ModuleNotFoundError("No module named 'scipy'")],
                         ids=lambda exc: type(exc).__name__)
def test_an_import_error_while_rebuilding_a_built_in_kind_names_the_edge_too(missing):
    """A factory that needs a package this environment lacks.  The edge
    that needs it is what the user has to be told, whichever kind it is:
    ``ImportError`` is one of the errors the loader reports for a built-in
    kind as well (it used not to be)."""
    def resolver(reference):
        raise missing

    for kind in ("nearest_neighbor", "rbf", "projection_1d", INVERSE_DISTANCE):
        points = ({"source_boundaries": INLINE3, "target_boundaries": INLINE3}
                  if kind == "projection_1d" else PAIR)
        edge = {"source_node": "a", "target_node": "b", "source_field": "v",
                "target_field": "inp", "mapping": {"kind": kind, "points": points}}
        with pytest.raises(MappingRebuildError) as wrapped:
            GraphManager._rebuild_mapping(edge, resolver)
        assert wrapped.value.edge == EDGE and wrapped.value.kind == kind
        assert wrapped.value.__cause__ is missing
        assert type(missing).__name__ in str(wrapped.value)


def test_a_base_exception_from_a_registered_factory_is_not_swallowed():
    """``KeyboardInterrupt`` and ``SystemExit`` are not failures of the
    edge: only ``Exception`` is wrapped."""
    class Interrupt(BaseException):
        pass

    def interrupted(source_points, target_points, *, source_points_ref=None,
                    target_points_ref=None):
        raise Interrupt()

    with temporary_kind("interrupted", interrupted,
                        arrays=("source_points", "target_points"), hyperparameters={}):
        with pytest.raises(Interrupt):
            _load({"kind": "interrupted", "points": PAIR})


def test_an_unlisted_exception_from_a_built_in_rebuild_still_surfaces_as_itself():
    """The wider net is for factories the library did not write.  What the
    built-in kinds can raise is listed in the loader; anything else from
    them is a defect, and surfaces unwrapped exactly as it did."""
    def boom(reference):
        raise RuntimeError("not one of the listed types")

    edge = {"source_node": "a", "target_node": "b", "source_field": "v",
            "target_field": "inp", "mapping": {"kind": "nearest_neighbor", "points": PAIR}}
    with pytest.raises(RuntimeError, match="not one of the listed types"):
        GraphManager._rebuild_mapping(edge, boom)
    with pytest.raises(MappingRebuildError):
        GraphManager._rebuild_mapping(
            {**edge, "mapping": {"kind": INVERSE_DISTANCE, "points": PAIR}}, boom)


@pytest.mark.parametrize("kind, hyper, points, message", [
    (INVERSE_DISTANCE, {"mode": "sideways"}, PAIR, "mode='sideways' not in"),
    (INVERSE_DISTANCE, {"power": -1.0}, PAIR, "power must be positive"),
    (LINEAR_1D, {}, {"source_points": [1.0, 0.5, 0.0], "target_points": INLINE3},
     "at least two increasing source points"),
], ids=["a mode it does not know", "a power it cannot use", "points it cannot order"])
def test_a_value_the_factory_itself_refuses_names_the_edge(kind, hyper, points, message):
    """A value of the declared type that the factory will not take is the
    factory's own ``ValueError``, wrapped like a built-in one."""
    with pytest.raises(MappingRebuildError, match=message) as refused:
        _load({"kind": kind, **hyper, "points": points})
    assert refused.value.edge == EDGE and refused.value.kind == kind
    assert type(refused.value.__cause__) is ValueError


def test_a_registered_kind_is_not_asked_what_a_built_in_factory_asks_of_its_geometry():
    """The built-in kinds check their coordinates *inside* their factories
    (MADD-ANO-192); neither the registry nor the reference resolver does.
    So a built-in kind's refusal reaches the loader through the registry
    as any factory's ``ValueError`` does, and a registered kind given the
    same references is called with the same arrays: what its formula does
    not cover is its own factory's to refuse."""
    descending = {"inline": [1.0, 0.5, 0.0], "dtype": "float64"}
    points = {"source_boundaries": descending, "target_boundaries": INLINE3}

    with pytest.raises(MappingRebuildError, match="must be strictly increasing") as refused:
        _load({"kind": "projection_1d", "points": points}, n_source=2, n_target=2)
    assert refused.value.edge == EDGE and refused.value.kind == "projection_1d"
    assert type(refused.value.__cause__) is ValueError

    handed = []

    def unchecked(source_boundaries, target_boundaries, *, source_boundaries_ref=None,
                  target_boundaries_ref=None):
        handed.append((np.asarray(source_boundaries).tolist(),
                       np.asarray(target_boundaries).tolist()))
        spec = MappingSpec("unchecked", {}, {
            "source_boundaries": reference_for_array(
                source_boundaries, source_boundaries_ref, name="source_boundaries"),
            "target_boundaries": reference_for_array(
                target_boundaries, target_boundaries_ref, name="target_boundaries")})
        return StaticLinearMapping(jnp.full((2, 2), 0.5, jnp.float32), kind="unchecked",
                                   spec=spec)

    with temporary_kind("unchecked", unchecked,
                        arrays=("source_boundaries", "target_boundaries"),
                        hyperparameters={}):
        gm = _load({"kind": "unchecked", "points": points}, n_source=2, n_target=2)
    assert handed == [([1.0, 0.5, 0.0], [0.0, 0.5, 1.0])]
    assert gm.edges[0].mapping.kind == "unchecked"


def test_a_registered_factory_with_a_required_hyper_parameter_names_it_when_a_spec_omits_it():
    def needs_radius(source_points, target_points, *, radius, source_points_ref=None,
                     target_points_ref=None):
        raise AssertionError("not reached without a radius")

    with temporary_kind("needs_radius", needs_radius,
                        arrays=("source_points", "target_points"),
                        hyperparameters={"radius": float}):
        with pytest.raises(MappingRebuildError, match="radius") as refused:
            _load({"kind": "needs_radius", "points": PAIR})
        assert isinstance(refused.value.__cause__, TypeError)


class _Bare:
    """A ``Mapping`` in every member, to take members away from."""

    kind = "returns"
    mode = "consistent"
    n_source = 3
    n_target = 3
    spec: object = None

    def __init__(self, tree=None, **attributes):
        self._tree = {"H": jnp.eye(3, dtype=jnp.float32)} if tree is None else tree
        self.__dict__.update(attributes)

    def params_pytree(self):
        return self._tree() if callable(self._tree) else self._tree

    def apply(self, field, weights=None, geom=None):
        return field

    def apply_T(self, field, weights=None, geom=None):
        return field

    def __repr__(self):
        return "_Bare()"


def _spec_as_given(source_points, target_points, source_ref, target_ref, kind="returns",
                   **hyper):
    return MappingSpec(kind, hyper, {
        "source_points": reference_for_array(source_points, source_ref, name="source_points"),
        "target_points": reference_for_array(target_points, target_ref, name="target_points"),
    })


class _NoTranspose:
    kind = "returns"
    mode = "consistent"
    n_source = n_target = 3

    def params_pytree(self):
        return {}

    def apply(self, field, weights=None, geom=None):
        return field


def _returns(what):
    """A factory of kind ``returns`` whose result is ``what(spec)``."""
    def factory(source_points, target_points, *, scale=1.0, source_ref=None,
                target_ref=None):
        spec = _spec_as_given(source_points, target_points, source_ref, target_ref,
                              scale=float(scale))
        return what(spec, source_points, target_points)
    return factory


_NODE_PAIR = {"source_points": {"node": "a", "field": "pts"},
              "target_points": {"node": "b", "field": "pts"}}

_BAD_RETURNS = {
    "None": (lambda spec, s, t: None, TypeError, "returned NoneType, which is not a Mapping"),
    "a dict": (lambda spec, s, t: {"H": np.eye(3)}, TypeError,
               "returned dict, which is not a Mapping"),
    "a matrix": (lambda spec, s, t: np.eye(3), TypeError,
                 "returned ndarray, which is not a Mapping"),
    "an object without apply_T": (lambda spec, s, t: _NoTranspose(), TypeError,
                                  r"it lacks \['apply_T'\]"),
    "a mapping of another kind": (
        lambda spec, s, t: _Bare(spec=spec, kind="other"), ValueError,
        "returned a mapping whose kind is 'other'"),
    "a StaticLinearMapping left at its default kind": (
        lambda spec, s, t: StaticLinearMapping(jnp.eye(3), spec=spec), ValueError,
        "returned a mapping whose kind is 'matrix'"),
    "a mapping without a spec": (lambda spec, s, t: _Bare(), ValueError,
                                 "carrying NoneType as its spec"),
    "a mapping whose spec is a dict": (
        lambda spec, s, t: _Bare(spec=spec.to_dict()), ValueError,
        "carrying dict as its spec"),
    "a mapping with a built-in kind's spec": (
        lambda spec, s, t: _Bare(spec=nearest_neighbor_mapping(s, t).spec), ValueError,
        "carrying a MappingSpec of kind 'nearest_neighbor'"),
    "a spec that dropped a node reference": (
        lambda spec, s, t: _Bare(spec=_spec_as_given(s, t, None, None,
                                                     **spec.hyperparameters)),
        ValueError, "did not record the reference it was given for 'source_points'"),
    "a spec that changed a hyper-parameter": (
        lambda spec, s, t: _Bare(spec=MappingSpec("returns", {"scale": 2.0}, spec.points)),
        ValueError, r"did not record hyper-parameter\(s\) \['scale'\] as given"),
    "a spec that dropped a hyper-parameter": (
        lambda spec, s, t: _Bare(spec=MappingSpec("returns", {}, spec.points)),
        ValueError, r"did not record hyper-parameter\(s\) \['scale'\] as given"),
    "weights that are a nested dict": (
        lambda spec, s, t: _Bare({"layer": {"H": jnp.eye(3)}}, spec=spec), ValueError,
        "params_pytree.. entry 'layer' is a nested container"),
    "weights that change between calls": (
        lambda spec, s, t: _Bare(lambda: {"H": next(_CALLS) * jnp.eye(3)}, spec=spec),
        ValueError, "is not the same on every call"),
}


class PVec(Vec):
    """A ``Vec`` that publishes a point set, for node references."""

    def __init__(self, name, timestep, n=3, pts=None):
        super(Vec, self).__init__(name, timestep, n=n, pts=pts)


def _node_config(mapping) -> dict:
    config = _config(mapping)
    for node in config["nodes"]:
        node["type"] = "PVec"
        node["params"]["pts"] = [0.0, 0.5, 1.0]
    return config


@pytest.mark.parametrize("case", sorted(_BAD_RETURNS))
def test_what_a_registered_factory_returns_is_held_to_the_spec_it_was_built_from(case):
    """A reloaded graph is saved again from what the factory returned, so
    a result that is not the mapping the spec describes is refused where
    it is made -- through the loader with the edge, and directly."""
    what, error, message = _BAD_RETURNS[case]
    registry = {**REGISTRY, "PVec": PVec}
    declaration = dict(arrays=("source_points", "target_points"),
                       hyperparameters={"scale": float},
                       references={"source_points": "source_ref",
                                   "target_points": "target_ref"})
    spec = {"kind": "returns", "scale": 1.5, "points": _NODE_PAIR}
    with temporary_kind("returns", _returns(what), **declaration):
        with pytest.raises(MappingRebuildError, match=message) as refused:
            GraphManager.from_dict(_node_config(spec), registry)
        assert refused.value.edge == EDGE and refused.value.kind == "returns"
        assert isinstance(refused.value.__cause__, error)
        assert "the factory registered for mapping kind 'returns'" in str(refused.value)
        gm = GraphManager()
        gm.add_node(PVec("a", 1.0, pts=[0.0, 0.5, 1.0]))
        gm.add_node(PVec("b", 1.0, pts=[0.0, 0.5, 1.0]))
        with pytest.raises(error, match=message):
            MappingSpec.from_dict(spec).build(gm.point_resolver())
        assert gm.edges == []


def test_a_registered_factory_returning_the_mapping_its_spec_describes_is_accepted():
    """The control for the table above: the same factory, returning what
    it should, rebuilds -- so each refusal is of the one thing changed."""
    declaration = dict(arrays=("source_points", "target_points"),
                       hyperparameters={"scale": float},
                       references={"source_points": "source_ref",
                                   "target_points": "target_ref"})
    with temporary_kind("returns", _returns(lambda spec, s, t: _Bare(spec=spec)),
                        **declaration):
        gm = GraphManager.from_dict(
            _node_config({"kind": "returns", "scale": 1.5, "points": _NODE_PAIR}),
            {**REGISTRY, "PVec": PVec})
        rebuilt = gm.edges[0].mapping
        assert rebuilt.spec.hyperparameters == {"scale": 1.5}
        assert rebuilt.spec.points["source_points"]["node"] == "a"
        # a hand-written reference carried no hash; the rebuild records it
        assert rebuilt.spec.points["source_points"]["sha256"] == _GOOD_DIGEST
        assert json.loads(json.dumps(gm.to_dict()))["edges"][0]["mapping"] == {
            "kind": "returns", "scale": 1.5, "shape": [3, 3],
            "points": rebuilt.spec.points}


def test_the_protocol_members_a_result_is_checked_for_are_the_protocols_own():
    """The "is not a Mapping" refusal reads the member list off the
    protocol; if the interpreter stopped providing it the refusal would
    name nothing."""
    assert sorted(Mapping.__protocol_attrs__) == [
        "apply", "apply_T", "kind", "mode", "n_source", "n_target", "params_pytree"]
    assert isinstance(_Bare(), Mapping) and not isinstance(_NoTranspose(), Mapping)


# ===========================================================================
# 6. What a mapping may put into params["mappings"]
# ===========================================================================

_F32 = jnp.float32
#: A call counter for the ``params_pytree()`` results that differ per call.
_CALLS = __import__("itertools").count(1)

#: ``params_pytree()`` results ``add_edge`` refuses, and why.
_REFUSED_TREES = {
    "a list": (lambda: [jnp.eye(3)], "returned list, not a plain dict"),
    "a tuple": (lambda: (jnp.eye(3),), "returned tuple, not a plain dict"),
    "None": (lambda: None, "returned NoneType, not a plain dict"),
    "an array": (lambda: jnp.eye(3), "not a plain dict"),
    "a dict subclass": (lambda: __import__("collections").OrderedDict(H=jnp.eye(3)),
                        "returned OrderedDict, not a plain dict"),
    "a nested dict": (lambda: {"layer": {"H": jnp.eye(3)}},
                      "entry 'layer' is a nested container"),
    "a list of arrays": (lambda: {"H": [jnp.eye(3), jnp.eye(3)]},
                         "entry 'H' is a nested container"),
    "a tuple of arrays": (lambda: {"H": (jnp.eye(3), jnp.eye(3))},
                          "entry 'H' is a nested container"),
    "a key holding a slash": (lambda: {"a/b": jnp.eye(3)}, "has the key 'a/b'"),
    "a key holding a dot": (lambda: {"a.b": jnp.eye(3)}, "has the key 'a.b'"),
    "a key holding a hash": (lambda: {"H#1": jnp.eye(3)}, "has the key 'H#1'"),
    "a key holding an arrow": (lambda: {"a->b": jnp.eye(3)}, "has the key 'a->b'"),
    "a key holding a space": (lambda: {"my weights": jnp.eye(3)}, "has the key 'my weights'"),
    "an empty key": (lambda: {"": jnp.eye(3)}, "has the key ''"),
    "a key starting with a digit": (lambda: {"1st": jnp.eye(3)}, "has the key '1st'"),
    "an integer key": (lambda: {0: jnp.eye(3)}, "has the key 0"),
    "a tuple key": (lambda: {("H", 0): jnp.eye(3)}, r"has the key \('H', 0\)"),
    "a NumPy float64 leaf": (lambda: {"H": np.eye(3)}, "entry 'H' is ndarray, not a JAX array"),
    "a NumPy float32 leaf": (lambda: {"H": np.eye(3, dtype=np.float32)},
                             "entry 'H' is ndarray, not a JAX array"),
    "a Python float leaf": (lambda: {"H": jnp.eye(3), "gain": 0.5},
                            "entry 'gain' is float, not a JAX array"),
    "a NumPy scalar leaf": (lambda: {"H": jnp.eye(3), "gain": np.float32(0.5)},
                            "entry 'gain' is float32, not a JAX array"),
    "a None leaf": (lambda: {"H": None}, "entry 'H' is NoneType, not a JAX array"),
    "a string leaf": (lambda: {"H": "eye"}, "entry 'H' is str, not a JAX array"),
    "an int32 leaf": (lambda: {"H": jnp.eye(3), "index": jnp.arange(3)},
                      "entry 'index' has dtype int32"),
    "an unsigned leaf": (lambda: {"H": jnp.arange(3, dtype=jnp.uint8)},
                         "entry 'H' has dtype uint8"),
    "a bool leaf": (lambda: {"H": jnp.eye(3, dtype=bool)}, "entry 'H' has dtype bool"),
    "a complex leaf": (lambda: {"H": jnp.eye(3, dtype=jnp.complex64)},
                       "entry 'H' has dtype complex64"),
    "a NaN": (lambda: {"H": jnp.eye(3).at[0, 0].set(jnp.nan)}, "holds a non-finite value"),
    "an infinity": (lambda: {"H": jnp.eye(3).at[1, 2].set(jnp.inf)},
                    "holds a non-finite value"),
    "a minus infinity in a scalar": (lambda: {"gain": jnp.asarray(-jnp.inf)},
                                     "holds a non-finite value"),
    "values that change between calls": (
        lambda: {"H": next(_CALLS) * jnp.eye(3)},
        "is not the same on every call: entry 'H' changed"),
    "a shape that changes between calls": (
        lambda: {"H": jnp.zeros((next(_CALLS),), _F32)},
        "is not the same on every call: entry 'H' changed"),
    "a dtype that changes between calls": (
        lambda: {"H": jnp.eye(3, dtype=(_F32, jnp.float16)[next(_CALLS) % 2])},
        "is not the same on every call: entry 'H' changed"),
    "keys that change between calls": (
        lambda: {f"w{next(_CALLS)}": jnp.eye(3)},
        "is not the same on every call: a second call returned the keys"),
    "a tree that stops being a dict": (
        lambda: {"H": jnp.eye(3)} if next(_CALLS) % 2 else None,
        "is not the same on every call: a second call returned"),
}


@pytest.mark.parametrize("case", sorted(_REFUSED_TREES))
def test_add_edge_refuses_a_mapping_whose_params_pytree_a_reader_could_not_walk(case):
    """Asked where the edge is added, before any reader can meet it: on
    the tree before the registry each of these compiled or saved wrongly
    (a ``/`` in a key was silently not restored from a checkpoint, a
    NumPy or Python leaf drew the "live weights differ" warning on a graph
    whose weights nobody had changed, a NaN made the FMU refuse its own
    snapshot) or failed deep inside ``compile()`` with an unrelated
    message."""
    tree, message = _REFUSED_TREES[case]
    gm = GraphManager()
    gm.add_node(Vec("a", 1.0))
    gm.add_node(Vec("b", 1.0))
    with pytest.raises(ValueError, match=message) as refused:
        gm.add_edge("a", "b", "v", "inp", mapping=_Bare(tree))
    assert str(refused.value).startswith("mapping _Bare() on a.v -> b.inp: params_pytree() ")
    assert gm.edges == [] and not gm.params["mappings"]


def test_a_traced_weight_is_refused_as_traced():
    gm = GraphManager()
    gm.add_node(Vec("a", 1.0))
    gm.add_node(Vec("b", 1.0))

    def traced(scale):
        gm.add_edge("a", "b", "v", "inp", mapping=_Bare({"H": scale * jnp.eye(3)}))
        return scale

    with pytest.raises(ValueError, match="entry 'H' is a traced value"):
        jax.jit(traced)(2.0)
    assert gm.edges == []


@pytest.mark.parametrize("matrix", [
    np.eye(3, dtype=np.int32), np.eye(3, dtype=bool), np.eye(3, dtype=np.float64),
    np.eye(3, dtype=np.complex64), np.full((3, 3), np.nan, np.float32),
], ids=["int32", "bool", "float64", "complex64", "nan"])
def test_a_static_linear_mapping_is_not_asked_and_takes_what_it_always_took(matrix):
    """The built-in class is exempt: what its matrix may hold did not
    change with the registry.

    What the ``matrix_mapping`` *factory* takes is the factory's own rule,
    not this check's: it refuses a non-finite ``H`` (MADD-ANO-192), under
    its own message, before there is a mapping to ask.  The class built
    directly is still added whatever its matrix holds."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # complex -> real casts
        gm = _vectors(StaticLinearMapping(jnp.asarray(matrix)))
        assert len(gm.edges) == 1
        if np.all(np.isfinite(matrix)):
            assert len(_vectors(matrix_mapping(matrix)).edges) == 1
        else:
            with pytest.raises(ValueError, match="H holds a non-finite value") as refused:
                matrix_mapping(matrix)
            assert "params_pytree()" not in str(refused.value)


def test_a_subclass_of_the_built_in_class_is_asked():
    class Indexed(StaticLinearMapping):
        def params_pytree(self):
            return {"H": self.H, "index": jnp.arange(3)}

    gm = GraphManager()
    gm.add_node(Vec("a", 1.0))
    gm.add_node(Vec("b", 1.0))
    with pytest.raises(ValueError, match="entry 'index' has dtype int32"):
        gm.add_edge("a", "b", "v", "inp", mapping=Indexed(jnp.eye(3)))


class _Shapes(_Bare):
    """Every leaf shape the contract allows, in one mapping: a scalar, a
    vector, a matrix, a rank-3 array, an array with no element, and two
    floating widths."""

    def __init__(self):
        super().__init__({
            "scalar": jnp.asarray(2.0, _F32),
            "vector": jnp.asarray([1.0, 0.5, 0.25], _F32),
            "matrix": jnp.eye(3, dtype=_F32),
            "cube": jnp.ones((2, 3, 3), _F32),
            "empty": jnp.zeros((0, 3), _F32),
            "half": jnp.asarray([1.0, 1.0, 1.0], jnp.float16),
        })

    def apply(self, field, weights=None, geom=None):
        w = self.params_pytree() if weights is None else weights
        mixed = (w["matrix"] + w["cube"][0] - w["cube"][1]) @ field
        return w["scalar"] * w["vector"] * mixed * w["half"].astype(field.dtype)


@pytest.fixture(scope="module")
def shaped() -> GraphManager:
    gm = _vectors(_Shapes())
    gm.compile()
    return gm


@pytest.fixture(scope="module")
def weightless() -> GraphManager:
    gm = _vectors(KINDS[SELECTION].build([0.0, 0.5, 1.0], [0.9, 0.1, 0.4]))
    gm.compile()
    return gm


def _moved(gm: GraphManager) -> None:
    for slot in gm.params["mappings"].values():
        for name, leaf in list(slot.items()):
            slot[name] = (leaf * 0 + 7).astype(leaf.dtype)


def _snapshot(gm: GraphManager) -> dict:
    return {key: weights_of_tree(slot) for key, slot in gm.params["mappings"].items()}


def weights_of_tree(slot: dict) -> dict:
    return {name: np.asarray(leaf) for name, leaf in slot.items()}


@pytest.mark.parametrize("graph", ["shaped", "weightless"])
def test_every_allowed_weight_shape_steps_and_is_restored_by_a_checkpoint(
        graph, request, tmp_path):
    """A scalar, a vector, a matrix, a rank-3 array, an empty array, a
    16-bit float (float16) -- and no weights at all: each steps, and a
    checkpoint puts every leaf back bit for bit under its own name."""
    gm = request.getfixturevalue(graph)
    gm.reset_state()
    gm.reset_params()
    first = np.asarray(gm.step()["b"]["v"])
    assert np.all(np.isfinite(first))
    before = _snapshot(gm)
    path = save_state(gm, tmp_path / "ck")
    with np.load(path, allow_pickle=False) as archive:
        stored = {k: archive[k] for k in archive.files}       # nothing was pickled
    members = sorted(k for k in stored if k.startswith("_params_mappings/"))
    assert members == sorted(f"_params_mappings/a.v->b.inp/{name}"
                             for name in before["a.v->b.inp"])
    state = {name: gm.get_node_state(name) for name in gm.node_names}
    _moved(gm)
    load_state(gm, path)
    for key, slot in before.items():
        assert_same_weights(gm.params["mappings"][key], slot, what=f"restored {key}")
    for name, fields in state.items():
        for field, value in fields.items():
            np.testing.assert_array_equal(np.asarray(gm.get_node_state(name)[field]),
                                          np.asarray(value))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm._warn_about_unsaved_mapping_weights()             # untouched: no warning
        specs = gm.param_specs()["mappings"]["a.v->b.inp"]
    assert sorted(specs) == sorted(before["a.v->b.inp"])
    assert all(spec.trainable is False for spec in specs.values())


def test_a_moved_weight_of_any_allowed_shape_draws_the_live_weights_warning(shaped):
    for name in ("scalar", "vector", "matrix", "cube", "half"):
        shaped.reset_params()
        slot = shaped.params["mappings"]["a.v->b.inp"]
        slot[name] = (slot[name] * 2).astype(slot[name].dtype)
        with pytest.warns(UserWarning, match=rf"live mapping weights \['{name}'\]"):
            shaped._warn_about_unsaved_mapping_weights()
    shaped.reset_params()


def test_an_edit_of_the_live_weights_does_not_reach_the_table_a_mapping_keeps():
    """``gm.params["mappings"][edge.key]`` is the graph's own copy.  A
    mapping may hand out the dict it keeps (this one does); kept by
    reference, an in-place edit of the live weights rewrote the mapping's
    own, so ``reset_params()`` restored the edit and ``to_dict()`` compared
    the live weights with themselves and never warned."""
    kept = {"H": jnp.eye(3, dtype=_F32), "gain": jnp.asarray(2.0, _F32)}
    mapping = _Bare(kept)
    assert mapping.params_pytree() is kept
    gm = _vectors(mapping)
    gm.compile()
    live = gm.params["mappings"]["a.v->b.inp"]
    assert live is not kept and live == kept
    live["H"] = 5.0 * live["H"]                               # as a fit leaves it
    live["gain"] = jnp.asarray(-1.0, _F32)
    assert float(kept["H"][0, 0]) == 1.0 and float(kept["gain"]) == 2.0
    with pytest.warns(UserWarning, match=r"live mapping weights \['H', 'gain'\]"):
        gm._warn_about_unsaved_mapping_weights()
    gm.reset_params()
    restored = gm.params["mappings"]["a.v->b.inp"]
    assert restored is not kept
    assert_same_weights(restored, weights_of_tree(kept))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm._warn_about_unsaved_mapping_weights()


@pytest.mark.parametrize("graph", ["shaped", "weightless"])
def test_the_fmu_state_archive_carries_every_allowed_weight_and_installs_none(
        graph, request):
    """The FMU's own snapshot restores; an archive that changes any one
    leaf is refused naming that leaf.  With no weights there is nothing an
    archive could install."""
    from maddening.fmi import build_model_description
    from maddening.fmi.fmu_state import serialize_fmu_state
    from maddening.fmi.sidecar import FmuSidecar, SidecarConfig

    gm = request.getfixturevalue(graph)
    gm.reset_state()
    gm.reset_params()
    md = build_model_description(gm, model_name="M")
    assert [v.name for v in md.variables if v.causality == "parameter"] == []
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs()))
    sidecar.set_fmu_state(sidecar.get_fmu_state())
    slot = gm.params["mappings"]["a.v->b.inp"]
    for name, leaf in slot.items():
        if leaf.size == 0:
            continue                                  # no element to forge
        forged = {section: {owner: dict(leaves) for owner, leaves in owners.items()}
                  for section, owners in gm.params.items()}
        forged["mappings"]["a.v->b.inp"][name] = (leaf + 1).astype(leaf.dtype)
        snapshot = serialize_fmu_state(state=gm._state, schema_token=md.instantiation_token,
                                       params=forged)
        with pytest.raises(ValueError, match=f"interface-mapping weights 'a.v->b.inp' / "
                                             f"'{name}'"):
            sidecar.set_fmu_state(snapshot)
    assert_same_weights(sidecar.params["mappings"]["a.v->b.inp"], weights_of_tree(slot))


def test_a_fit_runs_beside_a_weightless_mapped_edge_and_moves_every_trainable_shape():
    """``sysid`` over the entry: with no weights the fit of a node constant
    runs as on an unmapped graph; with every allowed shape trainable the
    optimiser moves each non-empty leaf and returns the tree's shape."""
    from maddening.sysid import fit

    gm = GraphManager()
    gm.add_node(HeatNode("a", 1e-4, n_cells=3, initial_temperature=1.0))
    gm.add_node(HeatNode("b", 1e-4, n_cells=3, initial_temperature=0.0))
    xs = [0.0, 0.5, 1.0]
    gm.add_edge("a", "b", "temperature", "heat_source",
                mapping=KINDS[SELECTION].build(xs, xs))
    gm.add_edge("a", "b", "temperature", "heat_source", additive=True, mapping=_Shapes())
    gm.compile()
    key = "a.temperature->b.heat_source#1"
    assert gm.params["mappings"]["a.temperature->b.heat_source"] == {}
    start = _snapshot(gm)[key]
    for name in start:
        gm.set_param_spec(key, name, ParamSpec())

    step, ext, state0 = gm._build_step_fn(), gm._default_external_inputs(), gm._state

    def loss(p):
        state = step(step(state0, ext, p), ext, p)
        return jnp.sum((state["b"]["temperature"] - 1.0) ** 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = fit(gm, loss, n_iter=3, lr=0.05)
    fitted = result.params["mappings"]
    assert fitted["a.temperature->b.heat_source"] == {}
    assert sorted(fitted[key]) == sorted(start)       # a pytree round trip sorts keys
    for name, leaf in start.items():
        got = np.asarray(fitted[key][name])
        assert got.shape == leaf.shape and got.dtype == leaf.dtype
        if leaf.size:
            assert not np.array_equal(got, leaf), f"{name} was trainable and did not move"
        assert np.all(np.isfinite(got))


# ===========================================================================
# 7. Round trips of a registered kind
# ===========================================================================

def _hyper_points(kind: str):
    """Every valid hyper-parameter combination of *kind*, small enough to list."""
    names = list(PAIR_KINDS[kind].hyper)
    combos = [{}]
    for name in names:
        combos = [{**c, name: value} for c in combos for value in PAIR_KINDS[kind].hyper[name]]
    if kind in SPARSE_POINT_KINDS:
        combos.append(dict(SCATTER))      # the conservative operator as a scatter-add
    return combos


_ROUND_TRIPS = [pytest.param(kind, hyper, id=f"{kind}{sorted(hyper.items())}")
                for kind in PAIRS for hyper in _hyper_points(kind)]


@pytest.mark.parametrize("form", ["node", "asset", "inline"])
@pytest.mark.parametrize("codec", ["json", "yaml"])
@pytest.mark.parametrize("kind, hyper", _ROUND_TRIPS)
def test_a_registered_kind_round_trips_bitwise_through_a_config(
        tmp_path, kind, hyper, codec, form):
    """Every registered kind, every hyper-parameter combination, both text
    codecs and all three reference forms: the config carries the recipe
    and no weight, ``from_dict`` rebuilds every weight bit for bit with an
    equal spec, and writing the reloaded graph gives the same config."""
    described = PAIR_KINDS[kind]
    source, target = np.array([0.0, 0.3, 0.55, 1.0]), np.array([0.1, 0.5, 0.9])
    refs = {"node": ({"node": "a", "field": "pts"}, {"node": "b", "field": "pts"}),
            "asset": ({"asset": "points.npz", "key": "source"},
                      {"asset": "target.npy"}),
            "inline": (None, None)}[form]
    np.savez(tmp_path / "points.npz", source=source, other=target)
    np.save(tmp_path / "target.npy", target)
    gm = GraphManager()
    gm.add_node(PVec("a", 1.0, n=4, pts=source.tolist()))
    gm.add_node(PVec("b", 1.0, n=3, pts=target.tolist()))
    points = (np.asarray(gm.get_node("a").params["pts"]),
              np.asarray(gm.get_node("b").params["pts"])) if form == "node" \
        else (source, target)
    mapping = described.build(*points, source_ref=refs[0], target_ref=refs[1], **hyper)
    gm.add_edge("a", "b", "v", "inp", mapping=mapping)

    # Through JSON first either way: the stand-in nodes keep their constants
    # in the node's own params mapping, which json.dumps writes and
    # yaml.safe_dump (plain dicts only) does not.
    config = json.loads(json.dumps(gm.to_dict()))
    text = json.dumps(config) if codec == "json" else yaml.safe_dump(config)
    written = (json.loads(text) if codec == "json" else yaml.safe_load(text))
    stored = written["edges"][0]["mapping"]
    assert stored["kind"] == kind
    assert {name: stored[name] for name in hyper} == hyper
    assert not set(stored) & set(described.weights)
    assert set(stored) <= {"kind", "points", "shape", "mode", *described.hyper}
    assert form in stored["points"]["source_points"]

    reloaded = GraphManager.from_dict(written, {**REGISTRY, "PVec": PVec}, base_dir=tmp_path)
    rebuilt = reloaded.edges[0].mapping
    assert type(rebuilt) is type(mapping)
    assert rebuilt.spec == mapping.spec and rebuilt.kind == kind
    assert_same_weights(weights_of(rebuilt), weights_of(mapping))
    assert (rebuilt.n_target, rebuilt.n_source) == (3, 4)
    assert json.loads(json.dumps(reloaded.to_dict())) == json.loads(json.dumps(config))


@pytest.mark.parametrize("kind", PAIRS)
def test_a_reloaded_registered_kind_steps_exactly_as_the_graph_it_was_saved_from(kind):
    gm = _rods(PAIR_KINDS[kind].build)
    gm.compile()
    reloaded = GraphManager.from_dict(json.loads(json.dumps(gm.to_dict())),
                                      {"HeatNode": HeatNode})
    reloaded.compile()
    assert_same_weights(reloaded.params["mappings"][C2F], gm.params["mappings"][C2F])
    a, b = gm.run_scan(5), reloaded.run_scan(5)
    for name in gm.node_names:
        for field in a[name]:
            np.testing.assert_array_equal(np.asarray(a[name][field]),
                                          np.asarray(b[name][field]))


@pytest.mark.parametrize("kind", PAIRS)
def test_add_edge_accepts_a_registered_spec_and_its_dict_form(kind):
    described = PAIR_KINDS[kind]
    gm = _rods(described.build)
    spec = gm.edges[0].mapping.spec
    for given in (spec, spec.to_dict(), {**spec.to_dict(), "shape": [12, 6]}):
        other = GraphManager()
        other.add_node(HeatNode("coarse", 1e-4, n_cells=6, thermal_diffusivity=0.1))
        other.add_node(HeatNode("fine", 1e-4, n_cells=12, thermal_diffusivity=0.1))
        other.add_edge("coarse", "fine", "temperature", "heat_source", mapping=given)
        assert other.edges[0].mapping.spec == spec
        assert_same_weights(weights_of(other.edges[0].mapping),
                            weights_of(gm.edges[0].mapping))


@pytest.mark.parametrize("kind", [k for k in PAIRS if PAIR_KINDS[k].weights])
def test_a_checkpoint_beats_the_config_for_a_registered_kinds_trained_weights(
        tmp_path, kind):
    """Config: the recipe.  Checkpoint: the weights, each under its own
    name.  ``to_dict`` says which weights a config will not carry."""
    gm = _rods(PAIR_KINDS[kind].build)
    gm.compile()
    recipe = weights_of_tree(gm.params["mappings"][C2F])
    for name in PAIR_KINDS[kind].weights:
        gm.params["mappings"][C2F][name] = 1.5 * gm.params["mappings"][C2F][name]
    trained = weights_of_tree(gm.params["mappings"][C2F])
    path = save_state(gm, tmp_path / "ck")
    names = "', '".join(sorted(PAIR_KINDS[kind].weights))
    with pytest.warns(UserWarning, match=rf"live mapping weights \['{names}'\]"):
        config = json.loads(json.dumps(gm.to_dict()))
    reloaded = GraphManager.from_dict(config, {"HeatNode": HeatNode})
    reloaded.compile()
    assert_same_weights(reloaded.params["mappings"][C2F], recipe, what="rebuilt")
    load_state(reloaded, path)
    assert_same_weights(reloaded.params["mappings"][C2F], trained, what="checkpoint")
    np.testing.assert_array_equal(np.asarray(reloaded.step()["fine"]["temperature"]),
                                  np.asarray(gm.step()["fine"]["temperature"]))


@pytest.mark.parametrize("kind", [k for k in PAIRS if PAIR_KINDS[k].weights])
def test_a_trainable_spec_on_a_registered_kinds_weight_round_trips(kind):
    gm = _rods(PAIR_KINDS[kind].build)
    for name in PAIR_KINDS[kind].weights:
        gm.set_param_spec(C2F, name, ParamSpec(trainable=True, description=f"learned {name}"))
    with pytest.raises(KeyError, match="no weight 'absent'"):
        gm.set_param_spec(C2F, "absent", ParamSpec())
    config = json.loads(json.dumps(gm.to_dict()))
    reloaded = GraphManager.from_dict(config, {"HeatNode": HeatNode})
    reloaded.compile()
    for name in PAIR_KINDS[kind].weights:
        assert reloaded.param_specs()["mappings"][C2F][name].description == f"learned {name}"
        assert reloaded.trainable_mask()["mappings"][C2F][name] is True


@pytest.mark.parametrize("kind", PAIRS)
def test_a_strict_write_refuses_a_stale_node_reference_of_a_registered_kind(kind):
    """The write-time check of node references reads the spec, whatever
    its kind: a reference to a node that is gone, or whose field moved, is
    refused when the config is written."""
    pts = [0.0, 0.5, 1.0]
    gm = GraphManager()
    for name in ("a", "b", "c"):
        gm.add_node(PVec(name, 1.0, pts=pts))
    gm.add_edge("a", "b", "v", "inp", mapping=PAIR_KINDS[kind].build(
        np.asarray(pts), np.asarray(pts), source_ref={"node": "c", "field": "pts"}))
    gm.to_dict()                                           # fine while it exists
    gm.get_node("c").params["pts"] = [0.0, 0.25, 1.0]
    with pytest.raises(ValueError, match="no longer describes the points") as moved:
        gm.to_dict()
    assert "a.v->b.inp" in str(moved.value)
    with pytest.raises(ValueError, match="Cannot remove node 'c'"):
        gm.remove_node("c")
    # remove_node() refuses this removal while the edge holds the reference
    # (MADD-ANO-214); the write-time check is for a graph that lost the node
    # some other way.
    gm._remove_node("c", replacing=True)  # noqa: SLF001
    with pytest.raises(ValueError, match="unknown node 'c'"):
        gm.to_dict()
    assert gm.to_dict(strict_mappings=False)["edges"][0]["mapping"]["kind"] == kind


# ---------------------------------------------------------------------------
# What an edge writes must read back as the mapping's spec
# ---------------------------------------------------------------------------

def test_a_registered_mapping_whose_description_is_not_its_spec_is_refused_at_write_time():
    """A class of the caller's own may define ``describe()``; an edge
    writes what it returns, so it has to be the spec.  One that reports
    another hyper-parameter would reload as a different mapping."""
    class Describing(InverseDistanceMapping):
        def describe(self):
            return {**self.spec.to_dict(), "power": 9.0}

    honest = KINDS[INVERSE_DISTANCE].build([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], power=2.0)
    lying = Describing(W=honest.W, gain=honest.gain, mode=honest.mode, spec=honest.spec)
    gm = _vectors(lying)
    with pytest.raises(ValueError, match="what it writes to a config is not its "
                                         "MappingSpec") as refused:
        gm.to_dict()
    assert "a.v->b.inp" in str(refused.value) and "'power': 9.0" in str(refused.value)
    # the display writer shows what the mapping says of itself
    assert gm.to_dict(strict_mappings=False)["edges"][0]["mapping"]["power"] == 9.0


@pytest.mark.parametrize("built, message", [
    (lambda spec: StaticLinearMapping(jnp.eye(3), spec=spec),
     "does not read back at all.*mapping kind 'matrix' has no hyper-parameter"),
    (lambda spec: StaticLinearMapping(jnp.eye(3), kind=SELECTION, spec=spec),
     "does not read back at all.*mapping kind 'selection' has no hyper-parameter"),
    (lambda spec: StaticLinearMapping(jnp.eye(3), kind=LINEAR_1D, spec=spec,
                                      meta={"stiffness": 2.0}),
     r"does not read back at all.*no hyper-parameter\(s\) \['stiffness'\]"),
], ids=["the default kind", "another registered kind",
        "a description key that is no hyper-parameter"])
def test_a_static_linear_mapping_of_a_registered_kind_must_describe_its_own_spec(
        built, message):
    """A third party's short way is the built-in class, whose description
    is its spec *plus* its ``kind`` and ``meta``; left at the default kind
    it would be written as a ``matrix`` and rebuilt by the wrong factory."""
    spec = KINDS[LINEAR_1D].build([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]).spec
    gm = _vectors(built(spec))
    with pytest.raises(ValueError, match=message):
        gm.to_dict()


def test_a_describe_method_that_returns_the_spec_is_written_as_it_is():
    class Describing(InverseDistanceMapping):
        def describe(self):
            return {**self.spec.to_dict(), "shape": [self.n_target, self.n_source]}

    honest = KINDS[INVERSE_DISTANCE].build([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    gm = _vectors(Describing(W=honest.W, gain=honest.gain, mode=honest.mode,
                             spec=honest.spec))
    assert gm.to_dict()["edges"][0]["mapping"] == {**honest.spec.to_dict(), "shape": [3, 3]}
