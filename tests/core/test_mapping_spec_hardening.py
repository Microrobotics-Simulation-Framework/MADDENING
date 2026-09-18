"""A serialised interface mapping is untrusted input, and a reference is
only useful if it still describes the points the mapping was built from.

These are the invariants an independent audit of the ``MappingSpec``
feature found unguarded: the loader must apply ``param_specs`` that name
a mapped edge, a reference must be checked against the array it claims
to describe (at write time and at rebuild time), an ``{"asset"}`` path
must stay inside its config directory *after* symlinks are resolved and
may not allocate unbounded memory, every rebuild failure must name its
edge, and writing the recipe of a mapping whose weights were trained
must say so."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import builtins
import io
import json
import tempfile
import warnings
import zipfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from maddening.core.coupling import mapping_spec as ms
from maddening.core.coupling.mapping import (
    matrix_mapping,
    nearest_neighbor_mapping,
    projection_1d_mapping,
    rbf_mapping,
)
from maddening.core.coupling.mapping_spec import (
    MappingRebuildError,
    MappingSpec,
    PointReferenceError,
    make_point_resolver,
    point_array_digest,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.params import ParamSpec
from maddening.nodes.heat import HeatNode
from tests.conftest import EXAMPLES_COSTLY

C2F = "coarse.temperature->fine.heat_source"


class Vec(SimulationNode):
    """n-vector integrating its boundary input (a stand-in interface)."""

    def __init__(self, name, timestep, n=3, pts=None):
        super().__init__(name, timestep, n=n, pts=pts)

    def initial_state(self):
        return {"v": jnp.arange(1, self.params["n"] + 1, dtype=jnp.float32)}

    def update(self, s, bi, dt):
        return {"v": s["v"] + dt * bi.get("inp", jnp.zeros_like(s["v"]))}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(self.params["n"],), description="i")}


REGISTRY = {"HeatNode": HeatNode, "Vec": Vec}
INLINE3 = {"inline": [0.0, 0.5, 1.0], "dtype": "float64"}


def _grid(gm, name):
    return np.asarray(gm.get_node(name).static_data["grid_x"].value)


def _two_rods():
    gm = GraphManager()
    gm.add_node(HeatNode("coarse", 1e-4, n_cells=6, thermal_diffusivity=0.1))
    gm.add_node(HeatNode("fine", 1e-4, n_cells=12, thermal_diffusivity=0.1))
    gm.add_edge("coarse", "fine", "temperature", "heat_source",
                mapping=rbf_mapping(_grid(gm, "coarse"), _grid(gm, "fine"), epsilon=2.0,
                                    source_ref={"node": "coarse", "field": "grid_x"},
                                    target_ref={"node": "fine", "field": "grid_x"}))
    gm.compile()
    return gm


def _config(mapping):
    return {"nodes": [{"type": "Vec", "name": "a", "timestep": 1.0, "params": {"n": 3}},
                      {"type": "Vec", "name": "b", "timestep": 1.0, "params": {"n": 3}}],
            "edges": [{"source_node": "a", "target_node": "b", "source_field": "v",
                       "target_field": "inp", "mapping": mapping}],
            "external_inputs": []}


# --------------------------------------------------- F1: trainable mapping weights

def test_trainable_mapping_param_spec_survives_config_round_trip():
    """``set_param_spec(edge.key, "H", ParamSpec())`` — the documented way
    to calibrate an interface operator — is written to the config and read
    back.  The loader must create the edges (and their
    ``params["mappings"]`` slots) before it applies ``param_specs``,
    which name those slots."""
    gm = _two_rods()
    gm.set_param_spec(C2F, "H", ParamSpec(trainable=True, description="learned"))

    config = json.loads(json.dumps(gm.to_dict()))
    assert config["param_specs"][C2F]["H"]["trainable"] is True

    gm2 = GraphManager.from_dict(config, REGISTRY)
    gm2.compile()
    assert gm2.param_spec_overrides()[C2F]["H"].trainable is True
    # sysid-style access: the spec is in param_specs() and the trainable
    # mask marks exactly this leaf.
    assert gm2.param_specs()["mappings"][C2F]["H"].trainable is True
    assert gm2.trainable_mask()["mappings"][C2F]["H"] is True


def test_param_specs_naming_neither_a_node_nor_a_mapped_edge_is_a_value_error():
    """The loader speaks ``ValueError`` (not ``KeyError``) and names what
    it could not apply."""
    config = _config(None)
    config["param_specs"] = {"ghost": {"n": ParamSpec().to_dict()}}
    with pytest.raises(ValueError, match=r"param_specs\['ghost'\]\['n'\]"):
        GraphManager.from_dict(config, REGISTRY)


# ------------------------------------------- F2 / F7: references must stay truthful

def test_reference_that_does_not_match_the_points_used_is_refused():
    """A ``source_ref`` naming the wrong node field has the same shape but
    different values: the recorded content hash catches it at write time
    instead of silently rebuilding a different operator."""
    gm = GraphManager()
    gm.add_node(HeatNode("a", 1e-4, n_cells=6, length=1.0))
    gm.add_node(HeatNode("b", 1e-4, n_cells=6, length=2.0))
    xa, xb = _grid(gm, "a"), _grid(gm, "b")
    assert xa.shape == xb.shape and not np.array_equal(xa, xb)

    gm.add_edge("a", "b", "temperature", "heat_source",
                mapping=rbf_mapping(xa, xb, epsilon=2.0,
                                    source_ref={"node": "b", "field": "grid_x"},
                                    target_ref={"node": "b", "field": "grid_x"}))
    with pytest.raises(ValueError, match="no longer describes the points"):
        gm.to_dict()

    # ... and a hand-written hash that does not match the points is refused
    # by the factory itself.
    with pytest.raises(PointReferenceError, match="does not describe these points"):
        rbf_mapping(xa, xb, source_ref={"node": "a", "field": "grid_x",
                                        "sha256": point_array_digest(xb)})


def test_strict_to_dict_refuses_a_reference_to_a_removed_node():
    gm = GraphManager()
    for name in ("a", "b", "c"):
        gm.add_node(Vec(name, 1.0, pts=[0.0, 0.5, 1.0]))
    pts = np.array([0.0, 0.5, 1.0])
    gm.add_edge("a", "b", "v", "inp",
                mapping=nearest_neighbor_mapping(
                    pts, pts, source_ref={"node": "c", "field": "pts"}))
    gm.remove_node("c")
    with pytest.raises(ValueError, match="unknown node 'c'") as exc:
        gm.to_dict()
    assert "a.v->b.inp" in str(exc.value)
    # the display writer still describes it (GET /graph must not break)
    assert gm.to_dict(strict_mappings=False)["edges"][0]["mapping"]["kind"] == \
        "nearest_neighbor"


def test_rebuilt_reference_that_changed_since_the_save_is_refused_on_load():
    """The hash is checked on the way in as well: a node field that moved
    between save and load rebuilds a different operator, so it is an error
    naming the edge, not a silent difference."""
    gm = _two_rods()
    config = json.loads(json.dumps(gm.to_dict()))
    for node in config["nodes"]:
        if node["name"] == "fine":
            node["params"]["length"] = 2.0          # the fine grid moves
    with pytest.raises(MappingRebuildError, match="differs from the points") as exc:
        GraphManager.from_dict(config, REGISTRY)
    assert exc.value.edge == "coarse.temperature -> fine.heat_source"


# ------------------------------------------------------- F3: symlinks and base_dir

def test_asset_symlink_pointing_outside_base_dir_is_refused(tmp_path):
    """Symlinks are resolved before the containment check, for a link to a
    file and for a link to a directory."""
    outside = tmp_path / "outside"
    outside.mkdir()
    np.save(outside / "secret.npy", np.array([42.0, 43.0]))
    base = tmp_path / "cfg"
    base.mkdir()
    np.save(base / "own.npy", np.array([0.0, 1.0]))
    (base / "link.npy").symlink_to(outside / "secret.npy")
    (base / "dirlink").symlink_to(outside, target_is_directory=True)

    resolve = make_point_resolver(base_dir=base)
    np.testing.assert_array_equal(resolve({"asset": "own.npy"}), [0.0, 1.0])
    for rel in ("link.npy", "dirlink/secret.npy"):
        with pytest.raises(PointReferenceError, match="outside the config directory"):
            resolve({"asset": rel})


# ----------------------------------------------------------- F4: asset size limits

def _forged_npy(path: Path, shape: tuple, dtype="<f8") -> None:
    """A ``.npy`` file that is only a header: it claims ``shape`` but
    holds no data at all."""
    with open(path, "wb") as fp:
        np.lib.format.write_array_header_1_0(
            fp, {"descr": dtype, "fortran_order": False, "shape": shape})


def test_asset_whose_header_exceeds_the_size_cap_is_refused_before_allocation(tmp_path):
    """numpy trusts the header, so a 100-byte file can ask for 800 GB.  The
    header is read first and refused against ``MAX_ASSET_BYTES``; nothing
    is allocated."""
    _forged_npy(tmp_path / "huge.npy", (100_000_000_000,))
    assert (tmp_path / "huge.npy").stat().st_size < 1024
    resolve = make_point_resolver(base_dir=tmp_path)
    with pytest.raises(PointReferenceError, match="MAX_ASSET_BYTES"):
        resolve({"asset": "huge.npy"})

    # a header that fits the cap but not the file it lives in
    _forged_npy(tmp_path / "short.npy", (1000,))
    with pytest.raises(PointReferenceError, match="the file holds only"):
        resolve({"asset": "short.npy"})


def test_compressed_asset_member_larger_than_the_cap_is_refused_before_decompression(
        tmp_path, monkeypatch):
    """An ``.npz`` member is judged by the uncompressed size in the zip
    directory, so a small highly-compressible archive cannot expand into
    memory first."""
    np.savez_compressed(tmp_path / "bomb.npz", pts=np.zeros(200_000, np.float64))
    assert (tmp_path / "bomb.npz").stat().st_size < 200_000       # compresses away
    monkeypatch.setattr(ms, "MAX_ASSET_BYTES", 8192)
    resolve = make_point_resolver(base_dir=tmp_path)
    with pytest.raises(PointReferenceError, match="refused before decompression"):
        resolve({"asset": "bomb.npz", "key": "pts"})


@pytest.mark.parametrize("rel", ["nul\x00.npy", "missing.npy", "loop.npy"])
def test_an_unresolvable_asset_path_is_a_point_reference_error(tmp_path, rel):
    """Whatever the operating system says about the path — gone, a
    symlink loop, a NUL byte — the caller gets one error type that names
    the asset and says where it was looked for."""
    (tmp_path / "loop.npy").symlink_to(tmp_path / "loop.npy")
    with pytest.raises(PointReferenceError, match="missing point asset"):
        make_point_resolver(base_dir=tmp_path)({"asset": rel})


def test_asset_of_a_non_numeric_dtype_is_refused(tmp_path):
    np.save(tmp_path / "words.npy", np.array(["a", "b"]))
    with pytest.raises(PointReferenceError, match="not a bool / integer / float"):
        make_point_resolver(base_dir=tmp_path)({"asset": "words.npy"})


# ------------------------------------ F3 / F4: the asset file is opened exactly once

def _count_opens_of(monkeypatch, target: Path) -> dict:
    """Count every open of ``target`` *by path*, however it is opened.

    ``os.open``, ``builtins.open`` and ``io.open`` are the three ways
    this code path (and numpy, and zipfile) can name a file; each extra
    one is another window in which the name can come to mean a different
    file than the one that was checked.
    """
    counter = {"n": 0}
    real_os_open, real_builtins_open, real_io_open = os.open, builtins.open, io.open

    def names_target(path) -> bool:
        try:
            return Path(os.fsdecode(path)) == target
        except TypeError:                    # an int fd, or something stranger
            return False

    def counted(wrapped):
        def opener(path, *args, **kwargs):
            if names_target(path):
                counter["n"] += 1
            return wrapped(path, *args, **kwargs)
        return opener

    monkeypatch.setattr(os, "open", counted(real_os_open))
    monkeypatch.setattr(builtins, "open", counted(real_builtins_open))
    monkeypatch.setattr(io, "open", counted(real_io_open))
    return counter


@pytest.mark.parametrize("name, save", [
    ("pts.npy", lambda path: np.save(path, np.array([1.0, 2.0, 3.0]))),
    ("pts.npz", lambda path: np.savez(path, pts=np.array([1.0, 2.0, 3.0]))),
])
def test_an_asset_is_opened_by_path_exactly_once(tmp_path, monkeypatch, name, save):
    """Resolving the path, then opening it again to read the header, then
    a third time to read the data, gives a writer in the config directory
    two windows in which to swap the checked file for a symlink.  The
    loader opens the resolved path once and does everything else on that
    descriptor, so there is nothing to swap in between."""
    save(tmp_path / name)
    real = (tmp_path / name).resolve()
    counter = _count_opens_of(monkeypatch, real)

    points = make_point_resolver(base_dir=tmp_path)({"asset": name})

    np.testing.assert_array_equal(points, [1.0, 2.0, 3.0])
    assert counter["n"] == 1, f"{name} was opened by path {counter['n']} times, not once"


def test_a_final_path_component_that_is_a_symlink_is_refused_at_open_time(tmp_path):
    """``O_NOFOLLOW`` on the open of the *resolved* path.

    A reference that names a symlink is not affected: the link is
    followed by the resolution, and the resolved path — the target — is
    what is opened, so a link inside the config directory to a file
    inside it still loads.  The flag only fires when the last component
    is a symlink at the moment of the open, which after a successful
    ``resolve()`` means it became one since, i.e. the race.
    """
    np.save(tmp_path / "real.npy", np.array([1.0, 2.0]))
    (tmp_path / "link.npy").symlink_to(tmp_path / "real.npy")

    # A symlink reference, resolved the ordinary way, still loads.
    resolve = make_point_resolver(base_dir=tmp_path)
    np.testing.assert_array_equal(resolve({"asset": "link.npy"}), [1.0, 2.0])

    # Handed a path whose final component is a link — what the resolved
    # path becomes if it is swapped after the check — the open refuses.
    with pytest.raises(PointReferenceError, match="cannot open point asset"):
        ms._open_asset(tmp_path / "link.npy", "link.npy")

    fp, _info = ms._open_asset(tmp_path / "real.npy", "real.npy")
    fp.close()


def test_the_asset_size_cap_is_measured_on_the_opened_descriptor(tmp_path):
    """``stat`` on a path answers about whatever that name means *now*;
    ``fstat`` answers about the file that was opened and keeps answering
    after the name has been given to another file.  The size the loader
    checks the header against comes from the descriptor it reads, so a
    forged header cannot be excused by a different file of the same
    name."""
    _forged_npy(tmp_path / "short.npy", (1000,))          # claims 8000 bytes of data
    np.save(tmp_path / "big.npy", np.zeros(1000))         # really holds 8000

    fp, info = ms._open_asset(tmp_path / "short.npy", "short.npy")
    try:
        os.replace(tmp_path / "big.npy", tmp_path / "short.npy")
        assert (tmp_path / "short.npy").stat().st_size > info.st_size, "swap did not take"
        # The descriptor still describes, and still reads, the small file.
        assert info.st_size < 1024
        shape, dtype = ms._read_npy_header(fp, "asset 'short.npy'")
        with pytest.raises(PointReferenceError, match="the file holds only"):
            ms._check_declared_size(shape, dtype, "asset 'short.npy'",
                                    available=info.st_size - fp.tell())
    finally:
        fp.close()


def test_an_asset_that_is_not_a_regular_file_is_refused_without_blocking(tmp_path):
    """A FIFO named ``*.npy`` opens for reading only once someone writes
    to it, which would hang the load.  The open is non-blocking and the
    descriptor's mode decides."""
    os.mkfifo(tmp_path / "pipe.npy")
    with pytest.raises(PointReferenceError, match="is not a file"):
        make_point_resolver(base_dir=tmp_path)({"asset": "pipe.npy"})
    (tmp_path / "dir.npy").mkdir()
    with pytest.raises(PointReferenceError, match="is not a file"):
        make_point_resolver(base_dir=tmp_path)({"asset": "dir.npy"})


# ------------------------------------------- F5: every rebuild failure names its edge

@pytest.mark.parametrize("mapping, message", [
    ({"kind": "rbf", "epsilon": None, "points": {"source_points": INLINE3,
                                                 "target_points": INLINE3}},
     "must be a real number"),
    ({"kind": "rbf", "epsilon": "abc", "points": {"source_points": INLINE3,
                                                  "target_points": INLINE3}},
     "must be a real number"),
    ({"kind": "rbf", "epsilon": float("inf"), "points": {"source_points": INLINE3,
                                                         "target_points": INLINE3}},
     "must be finite"),
    ({"kind": "nearest_neighbor", "shape": 3,
      "points": {"source_points": INLINE3, "target_points": INLINE3}},
     "two-element list of ints"),
    ({"kind": "nearest_neighbor", "points": [INLINE3, INLINE3]},
     "'points' must be a dict"),
    ({"kind": "nearest_neighbor",
      "points": {"source_points": {"asset": "bad.npz", "key": "pts"},
                 "target_points": INLINE3}},
     "not a valid .npz"),
    ({"kind": "nearest_neighbor",
      "points": {"source_points": {"asset": "nul\x00.npy"}, "target_points": INLINE3}},
     "cannot rebuild interface mapping"),
])
def test_from_dict_names_the_broken_edge_whatever_the_failure(tmp_path, mapping, message):
    (tmp_path / "bad.npz").write_bytes(b"not a zip archive at all")
    with pytest.raises(MappingRebuildError, match=message) as exc:
        GraphManager.from_dict(_config(mapping), REGISTRY, base_dir=tmp_path)
    assert "edge a.v -> b.inp" in str(exc.value)
    assert exc.value.__cause__ is not None          # the original is chained


def _numpy_ufunc_type_error() -> TypeError:
    """A real ``UFuncTypeError`` (a private numpy ``TypeError`` subclass),
    as raised when a factory meets a string point set."""
    try:
        np.asarray(["a", "b"]) * 2.0
    except TypeError as exc:
        return exc
    raise AssertionError("numpy no longer refuses str * float")


@pytest.mark.parametrize("exc", [
    TypeError("bad operand"),
    KeyError("missing"),
    OSError("disk went away"),
    MemoryError("out of memory"),
    zipfile.BadZipFile("not a zip"),
    json.JSONDecodeError("nope", "{", 0),
    _numpy_ufunc_type_error(),
])
def test_rebuild_reports_the_edge_for_every_failure_type(exc):
    """``_rebuild_mapping`` is the only place that knows which edge a
    config entry belongs to, so no failure may escape it unlabelled."""
    def boom(ref):
        raise exc

    edge = {"source_node": "a", "target_node": "b",
            "source_field": "v", "target_field": "inp",
            "mapping": {"kind": "nearest_neighbor",
                        "points": {"source_points": INLINE3,
                                   "target_points": INLINE3}}}
    with pytest.raises(MappingRebuildError) as caught:
        GraphManager._rebuild_mapping(edge, boom)
    assert caught.value.edge == "a.v -> b.inp"
    assert caught.value.kind == "nearest_neighbor"
    assert caught.value.__cause__ is exc
    assert type(exc).__name__ in str(caught.value)


# ----------------------------------------------------------- F6: describe()["kind"]

def test_describe_keeps_the_user_facing_kind_label_for_matrix_mappings(tmp_path):
    """``matrix_mapping(kind="supermesh").describe()["kind"]`` is what a
    dashboard (``GET /graph``) shows; the spec keeps ``matrix`` so the
    rebuild finds the factory, and carries the label separately."""
    H = np.eye(3, dtype=np.float32)
    np.save(tmp_path / "H.npy", H)
    m = matrix_mapping(H, kind="supermesh", asset="H.npy")
    described = m.describe()
    assert described["kind"] == "supermesh" == m.kind
    assert described["label"] == "supermesh"
    assert m.spec.kind == "matrix"
    # describe() is what the writers store, so it must load back as a matrix
    rebuilt = MappingSpec.from_dict(described).build(make_point_resolver(base_dir=tmp_path))
    assert rebuilt.kind == "supermesh" and rebuilt.spec.kind == "matrix"
    np.testing.assert_array_equal(np.asarray(rebuilt.H), H)
    # a label that collides with another factory's name is still a matrix
    assert MappingSpec.from_dict({**described, "kind": "rbf", "label": "rbf"}).kind == \
        "matrix"


# ---------------------------------------------------------- F8: inline point limits

@pytest.mark.parametrize("ref, message", [
    ({"inline": [[0.0] * 1000] * 2, "dtype": "float64"}, "INLINE_ELEMENT_LIMIT"),
    ({"inline": [0.0, float("nan")], "dtype": "float64"}, "must be finite"),
    ({"inline": [0.0, 1.0], "dtype": "complex128"}, "not a bool / integer / float"),
    ({"inline": [0.0, 1.0], "dtype": "U3"}, "not a bool / integer / float"),
    ({"inline": [0.0, 1.0], "dtype": "object"}, "not a bool / integer / float"),
    ({"inline": [0.0, 1.0], "dtype": "datetime64[ns]"}, "not a bool / integer / float"),
    ({"inline": 3.0}, "must be a list"),
])
def test_inline_reference_rejects_non_real_dtypes_non_finite_values_and_total_size(
        ref, message):
    with pytest.raises(PointReferenceError, match=message):
        ms.normalise_point_reference(ref)


def test_inline_reference_accepts_small_real_point_sets():
    ref = ms.normalise_point_reference({"inline": [[0.0, 1.0], [1.0, 2.0]],
                                        "dtype": "float32"})
    assert ref == {"inline": [[0.0, 1.0], [1.0, 2.0]], "dtype": "float32"}


# ---------------------------------------------------------- F9: resolver edge cases

def test_node_reference_to_a_scalar_static_field_is_a_point_reference_error():
    """A 0-d array is not a point set; saying so beats an ``IndexError``
    from inside the factory."""
    from maddening.core.static_data import StaticArray

    class WithScalar(Vec):
        @property
        def static_data(self):
            return {"zero_d": StaticArray(np.array(3.0)),
                    "pts": StaticArray(np.array([0.0, 0.5, 1.0]))}

    gm = GraphManager()
    gm.add_node(WithScalar("a", 1.0))
    resolve = gm.point_resolver()
    with pytest.raises(PointReferenceError, match="is a scalar"):
        resolve({"node": "a", "field": "zero_d"})
    np.testing.assert_array_equal(resolve({"node": "a", "field": "pts"}),
                                  [0.0, 0.5, 1.0])
    # the "available fields" hint lists each field once
    with pytest.raises(PointReferenceError, match=r"array fields are \['pts', 'zero_d'\]"):
        resolve({"node": "a", "field": "nope"})


def test_asset_key_is_refused_for_a_single_array_npy_file():
    with pytest.raises(PointReferenceError, match="drop the key"):
        ms.normalise_point_reference({"asset": "pts.npy", "key": "anything"})
    with pytest.raises(PointReferenceError, match="'key' must be a string"):
        ms.normalise_point_reference({"asset": "pts.npz", "key": 3})


def test_matrix_mapping_refuses_a_non_string_kind_label():
    H = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError, match="non-empty string label"):
        matrix_mapping(H, kind=5)
    with pytest.raises(ValueError, match="non-empty string label"):
        matrix_mapping(H, kind="")
    with pytest.raises(ValueError, match="must be a str"):
        MappingSpec("matrix", {"label": 5}, {"H": {"asset": "H.npy"}})


# ---------------------------------------------------- F10: non-finite hyper-parameters

@pytest.mark.parametrize("kwargs", [{"epsilon": float("inf")}, {"epsilon": float("nan")},
                                    {"ridge": float("inf")}, {"epsilon": "2.0"}])
def test_rbf_mapping_refuses_non_finite_hyperparameters(kwargs):
    """``json.dumps`` writes ``Infinity``/``NaN``, which strict JSON
    parsers reject — and a non-finite kernel width is meaningless anyway."""
    pts = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="must be (finite|a real number)"):
        rbf_mapping(pts, pts, **kwargs)


def test_finite_hyperparameters_still_round_trip_through_strict_json():
    pts = np.array([0.0, 0.5, 1.0])
    m = rbf_mapping(pts, pts, epsilon=2.5, ridge=1e-8)
    json.loads(json.dumps(m.describe()), parse_constant=_no_constants)


def _no_constants(name):
    raise AssertionError(f"non-finite JSON constant {name!r} in a mapping spec")


# --------------------------------------------------- F11: trained weights are not lost

def test_to_dict_warns_when_live_mapping_weights_differ_from_the_recipe():
    gm = _two_rods()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.to_dict()                       # untouched weights: no warning

    gm.params["mappings"][C2F]["H"] = 1.5 * gm.params["mappings"][C2F]["H"]
    with pytest.warns(UserWarning, match=r"live mapping weights \['H'\]"):
        config = gm.to_dict()
    assert "H" not in config["edges"][0]["mapping"]
    # display-only writer stays silent: nothing is being persisted
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.to_dict(strict_mappings=False)


# ------------------------------------------------------------- F14: sharded wrappers

def test_node_reference_reaches_through_a_sharded_wrapper():
    """A point reference resolves to the statics of the node a wrapper wraps.

    This inverts what this test used to assert.  ``ShardedStencilNode``
    exposed none of its inner node's ``static_data``, so a reference to
    ``grid_x`` through the wrapper was a documented dead end and the
    resolver's error pointed at the asset form instead.  Now that
    ``SimulationNode.static_data`` forwards to the nodes a node wraps, the
    reference reaches the *declaration* the inner node made -- the full
    ``(8,)`` grid, not a per-device shard -- which is exactly what a point
    set should be.  The asset route still works and is still the right
    answer for points that are not a node's static data at all.
    """
    from maddening.cloud.multigpu.device_mesh import create_device_mesh
    from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
    from maddening.core.static_data import StaticArray

    class Stencil1D(SimulationNode):
        def __init__(self, name, timestep=1e-3, n=8):
            super().__init__(name, timestep, n=n)
            self._x = np.linspace(0.0, 1.0, n, dtype=np.float32)

        def initial_state(self):
            return {"f": jnp.zeros(self.params["n"], jnp.float32)}

        def halo_width(self):
            return {0: 1}

        def update(self, s, bi, dt):
            return {"f": s["f"]}

        def update_padded(self, sp, bi, dt, *, static_padded=None, shard_info=None):
            return {"f": sp["f"]}

        def boundary_input_spec(self):
            return {"inp": BoundaryInputSpec(shape=(self.params["n"],), description="i")}

        @property
        def static_data(self):
            return {"grid_x": StaticArray(self._x)}

    inner = Stencil1D("rod")
    wrapper = ShardedStencilNode(inner, create_device_mesh(shape=(1,)),
                                 axis_map={"devices": 0}, boundary="edge")
    grid = np.asarray(inner.static_data["grid_x"].value)

    # The wrapper now reports the inner node's declaration, full length.
    assert set(wrapper.static_data) == {"grid_x"}
    assert np.asarray(wrapper.static_data["grid_x"].value).shape == grid.shape

    gm = GraphManager()
    gm.add_node(wrapper)
    resolved = gm.point_resolver()({"node": "rod", "field": "grid_x"})
    np.testing.assert_array_equal(resolved, grid)

    # A field that is genuinely absent still fails, and the hint no longer
    # claims a wrapper cannot re-export -- because it now can.
    with pytest.raises(PointReferenceError) as exc:
        gm.point_resolver()({"node": "rod", "field": "not_a_field"})
    assert "does not re-export" not in str(exc.value)

    # the asset route still works
    with tempfile.TemporaryDirectory() as tmp:
        np.save(Path(tmp) / "grid.npy", grid)
        np.testing.assert_array_equal(
            make_point_resolver(base_dir=tmp)({"asset": "grid.npy"}), grid)


# ---------------------------------------------------- F15: property-based round trip

_KERNELS = ("gaussian", "multiquadric", "inverse_multiquadric", "thin_plate_spline")


def _points(data, n, dim, dtype):
    """``n`` distinct points in ``dim`` dimensions, on a coarse integer
    lattice so the coordinates are exact in both float widths."""
    rows = data.draw(st.lists(st.tuples(*[st.integers(-40, 40)] * dim),
                              min_size=n, max_size=n, unique=True))
    return (np.asarray(sorted(rows), dtype=dtype) / 8).astype(dtype)


@pytest.mark.parametrize("codec", ["json", "yaml"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["rbf", "nearest_neighbor", "projection_1d", "matrix"])
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, database=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(data=st.data())
def test_every_factory_round_trips_bitwise_for_random_hyperparameters(
        kind, dtype, codec, data):
    """For all four factories, both float widths, both text codecs and any
    valid hyper-parameters: the spec survives the text round trip and the
    factory rebuilds bitwise-identical weights with an equal
    ``describe()``."""
    dim = data.draw(st.integers(1, 3)) if kind in ("rbf", "nearest_neighbor") else 1
    n_src = data.draw(st.integers(dim + 2, 8))
    n_tgt = data.draw(st.integers(dim + 2, 8))
    mode = data.draw(st.sampled_from(["consistent", "conservative"]))

    with tempfile.TemporaryDirectory() as tmp:
        resolve = make_point_resolver(base_dir=tmp)
        if kind == "rbf":
            src, tgt = _points(data, n_src, dim, dtype), _points(data, n_tgt, dim, dtype)
            mapping = rbf_mapping(
                src, tgt, mode=mode,
                kernel=data.draw(st.sampled_from(_KERNELS)),
                epsilon=data.draw(st.floats(0.25, 8.0, allow_nan=False,
                                            allow_infinity=False)),
                polynomial=data.draw(st.booleans()),
                ridge=data.draw(st.floats(1e-10, 1e-3, allow_nan=False,
                                          allow_infinity=False)))
        elif kind == "nearest_neighbor":
            src, tgt = _points(data, n_src, dim, dtype), _points(data, n_tgt, dim, dtype)
            mapping = nearest_neighbor_mapping(src, tgt, mode=mode)
        elif kind == "projection_1d":
            mapping = projection_1d_mapping(_points(data, n_src, 1, dtype).ravel(),
                                            _points(data, n_tgt, 1, dtype).ravel())
        else:
            flat = data.draw(st.lists(st.integers(-8, 8), min_size=n_tgt * n_src,
                                      max_size=n_tgt * n_src))
            H = (np.asarray(flat, dtype=dtype) / 4).reshape(n_tgt, n_src)
            np.save(Path(tmp) / "H.npy", H)
            mapping = matrix_mapping(H, kind=data.draw(st.sampled_from(
                ["matrix", "supermesh", "mortar"])), mode=mode, asset="H.npy")

        described = mapping.describe()
        text = (json.dumps(described) if codec == "json"
                else yaml.safe_dump(described, default_flow_style=False))
        back = json.loads(text) if codec == "json" else yaml.safe_load(text)
        rebuilt = MappingSpec.from_dict(back).build(resolve)

    np.testing.assert_array_equal(np.asarray(rebuilt.H), np.asarray(mapping.H))
    assert rebuilt.H.dtype == mapping.H.dtype
    assert rebuilt.describe() == described
    assert rebuilt.spec == mapping.spec
