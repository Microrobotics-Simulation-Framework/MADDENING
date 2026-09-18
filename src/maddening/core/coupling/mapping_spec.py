"""MappingSpec: how an interface mapping was built, so it can be rebuilt.

A :class:`MappingSpec` records the *recipe* of a mapping — which factory
(``kind``), its hyper-parameters, and **references** to the point sets
it was built from — never the weights.  The factories in
:mod:`maddening.core.coupling.mapping` attach one to every mapping they
return; ``GraphManager.to_dict`` / ``from_dict`` and the USD writer /
reader carry it, and :meth:`MappingSpec.build` recomputes the mapping by
calling the same factory on the same inputs, so the rebuilt weights are
bitwise equal to the original ones.  Weights that were changed
afterwards (system identification, a hand edit of
``params["mappings"]``) live in checkpoints, which win over the rebuilt
values when loaded on top of a config; ``to_dict`` warns when it is
about to write the recipe of a mapping whose live weights differ.

Point references
----------------
A point set is described by reference, not inlined:

``{"node": "<name>", "field": "<key>", "sha256": "<hex>"}``
    a node's ``static_data[key]`` (a ``StaticArray`` is unwrapped) or,
    failing that, an array-valued constructor parameter
    ``node.params[key]`` — e.g. ``{"node": "rod", "field": "grid_x"}``
    for a ``HeatNode``;
``{"asset": "<relative path>.npy", "sha256": ...}`` /
``{"asset": "<path>.npz", "key": "<k>", "sha256": ...}``
    an external NumPy file, relative to the directory the config / USD
    stage lives in (``base_dir``).  Absolute paths and ``..`` components
    are refused, symlinks are resolved and the resolved file must still
    lie under the resolved ``base_dir``, so a config cannot read outside
    its own directory.  The file's header is read first and an array
    larger than :data:`MAX_ASSET_BYTES` (or larger than the file that
    claims to hold it) is refused before anything is allocated; only
    bool / integer / float arrays are accepted;
``{"inline": [[...], ...], "dtype": "float64"}`` (or a plain list)
    the points themselves, only for small sets — at most
    :data:`INLINE_POINT_LIMIT` points and :data:`INLINE_ELEMENT_LIMIT`
    numbers in total, finite, of a bool / integer / float dtype; the
    factories inline automatically when no reference is given and the
    set is that small.

``sha256`` is the content hash (:func:`point_array_digest`: dtype,
shape and bytes) of the array the factory was actually given.  The
factories record it; :func:`build_mapping` refuses a reference that
resolves to different points, and ``GraphManager.to_dict`` refuses to
write a node reference that no longer resolves to the recorded points.
A hand-written reference may omit it (then nothing is checked, and the
hash is recorded on the first rebuild).

A larger point set without a reference is not serialisable: the
factory leaves the entry ``None`` and ``GraphManager.to_dict`` refuses
with a message naming the argument (``source_ref=`` / ``target_ref=`` /
``asset=``) to pass.  An explicit matrix (``matrix_mapping``) is always
stored as an ``asset`` reference, never inlined.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

#: Point sets with at most this many points are inlined into the spec
#: when the factory is given no reference for them.
INLINE_POINT_LIMIT = 64

#: An inline point set may hold at most this many numbers in total
#: (points × coordinates), whatever its shape.
INLINE_ELEMENT_LIMIT = 16 * INLINE_POINT_LIMIT

#: Largest array (``prod(shape) * itemsize``) an ``{"asset"}`` reference
#: may load; read from the ``.npy`` header / ``.npz`` directory *before*
#: anything is allocated.  Module constant: raise it for genuinely large
#: interfaces (``mapping_spec.MAX_ASSET_BYTES = ...``).
MAX_ASSET_BYTES = 256 * 1024 * 1024

#: Array dtype kinds a point set may have (bool, signed / unsigned
#: integer, float).  Complex, string, object and datetime arrays are
#: refused both inline and from assets.
_NUMERIC_KINDS = "biuf"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# kind -> (factory name, array-argument names, allowed hyper-parameters)
_FACTORIES: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "rbf": ("rbf_mapping", ("source_points", "target_points"),
            ("kernel", "epsilon", "polynomial", "ridge", "mode")),
    "nearest_neighbor": ("nearest_neighbor_mapping", ("source_points", "target_points"),
                         ("mode",)),
    "projection_1d": ("projection_1d_mapping", ("source_boundaries", "target_boundaries"),
                      ()),
    "matrix": ("matrix_mapping", ("H",), ("mode", "label")),
}

# hyper-parameter -> expected type ("real" = finite int/float, not bool)
_HYPER_TYPES: dict[str, Any] = {
    "kernel": str, "mode": str, "label": str, "polynomial": bool,
    "epsilon": "real", "ridge": "real",
}

# array-argument name -> the factory keyword that carries its reference
_REF_KWARG = {
    "source_points": "source_ref", "target_points": "target_ref",
    "source_boundaries": "source_ref", "target_boundaries": "target_ref",
    "H": "asset",
}

_RESERVED_KEYS = ("kind", "points", "shape")


class PointReferenceError(ValueError):
    """A point reference is malformed, cannot be resolved, or resolves to
    different points than the mapping was built from."""


class MappingRebuildError(ValueError):
    """A serialised mapping could not be rebuilt.

    Raised by ``GraphManager.from_dict`` / ``load_graph_from_usd`` for
    *every* failure while rebuilding one edge's mapping — malformed
    spec, unresolvable or mismatching reference, unreadable asset, a
    factory error — with the edge key and the spec kind in the message
    and the original exception chained as ``__cause__``.
    """

    def __init__(self, edge: str, kind: Any, cause: BaseException):
        self.edge = edge
        self.kind = kind
        self.cause = cause
        what = "" if kind is None else f" (kind {kind!r})"
        super().__init__(
            f"edge {edge}: cannot rebuild interface mapping{what}: "
            f"{type(cause).__name__}: {cause}"
        )


# ---------------------------------------------------------------------------
# Content hashes
# ---------------------------------------------------------------------------


@stability(StabilityLevel.EVOLVING)
def point_array_digest(array: Any) -> str:
    """SHA-256 (hex) of an array's canonical bytes: dtype, shape, C-order data.

    Two arrays have the same digest exactly when they have the same
    dtype (including byte order), the same shape and bitwise-equal
    elements; this is what the ``sha256`` entry of a point reference
    records and what the rebuild checks.
    """
    arr = np.ascontiguousarray(np.asarray(array))
    h = hashlib.sha256()
    h.update(arr.dtype.str.encode("ascii"))
    h.update(repr(tuple(int(s) for s in arr.shape)).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _short(digest: Optional[str]) -> str:
    return "none" if digest is None else digest[:12] + "…"


def _without_hash(ref: dict) -> dict:
    return {k: v for k, v in ref.items() if k != "sha256"}


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def _check_sha256(ref: dict, name: str) -> Optional[str]:
    digest = ref.get("sha256")
    if digest is None:
        return None
    if not isinstance(digest, str) or not _SHA256_RE.match(digest):
        raise PointReferenceError(
            f"{name}: 'sha256' must be a 64-character lowercase hex string, got {digest!r}"
        )
    return digest


def normalise_point_reference(ref: Any, *, name: str = "points") -> dict:
    """Validate ``ref`` and return it in canonical dict form.

    Accepts the three reference forms of the module docstring, a plain
    (nested) list for an inline set, or a bare string as an asset path.
    """
    if isinstance(ref, str):
        ref = {"asset": ref}
    if isinstance(ref, (list, tuple)):
        ref = {"inline": ref}
    if not isinstance(ref, dict):
        raise PointReferenceError(
            f"{name}: a point reference must be a dict such as "
            f"{{'node': <name>, 'field': <key>}}, {{'asset': <file>.npy}} or "
            f"{{'inline': [...]}}, got {type(ref).__name__}"
        )
    if "node" in ref:
        if set(ref) - {"node", "field", "sha256"} or "field" not in ref or not all(
                isinstance(ref[k], str) and ref[k] for k in ("node", "field")):
            raise PointReferenceError(
                f"{name}: a node reference is {{'node': <name>, 'field': <key>}} "
                f"with string values (plus an optional 'sha256'), got {ref!r}"
            )
        out = {"node": ref["node"], "field": ref["field"]}
        digest = _check_sha256(ref, name)
        if digest is not None:
            out["sha256"] = digest
        return out
    if "asset" in ref:
        if set(ref) - {"asset", "key", "sha256"} or not isinstance(ref["asset"], str) \
                or not ref["asset"]:
            raise PointReferenceError(
                f"{name}: an asset reference is {{'asset': <relative path>}} "
                f"(plus 'key' for an .npz member and an optional 'sha256'), got {ref!r}"
            )
        _check_relative(ref["asset"], name)
        suffix = Path(ref["asset"]).suffix.lower()
        if suffix not in (".npy", ".npz"):
            raise PointReferenceError(
                f"{name}: asset {ref['asset']!r} must be a .npy or .npz file"
            )
        out = {"asset": ref["asset"]}
        if "key" in ref:
            if not isinstance(ref["key"], str):
                raise PointReferenceError(f"{name}: asset 'key' must be a string")
            if suffix != ".npz":
                raise PointReferenceError(
                    f"{name}: 'key' selects a member of an .npz archive; "
                    f"{ref['asset']!r} is a single-array .npy file, drop the key"
                )
            out["key"] = ref["key"]
        digest = _check_sha256(ref, name)
        if digest is not None:
            out["sha256"] = digest
        return out
    if "inline" in ref:
        if set(ref) - {"inline", "dtype"}:
            raise PointReferenceError(
                f"{name}: an inline reference is {{'inline': [...], 'dtype': <name>}}, "
                f"got keys {sorted(ref)}"
            )
        arr = _inline_array(ref, name)
        return {"inline": arr.tolist(), "dtype": str(arr.dtype)}
    raise PointReferenceError(
        f"{name}: unknown point reference {ref!r}; expected one of 'node'/'field', "
        f"'asset' or 'inline'"
    )


def _inline_array(ref: dict, name: str) -> np.ndarray:
    """The validated array of an inline reference."""
    raw = ref["inline"]
    if isinstance(raw, (list, tuple)) and len(raw) > INLINE_POINT_LIMIT:
        # Before materialising: a million-row list must not be turned
        # into an array just to be refused.
        raise PointReferenceError(
            f"{name}: {len(raw)} inline points exceed INLINE_POINT_LIMIT="
            f"{INLINE_POINT_LIMIT}; save them as an asset or reference a node field"
        )
    dtype_name = ref.get("dtype", "float64")
    try:
        dtype = np.dtype(dtype_name)
    except TypeError as exc:
        raise PointReferenceError(f"{name}: unknown inline dtype {dtype_name!r}") from exc
    if dtype.kind not in _NUMERIC_KINDS:
        raise PointReferenceError(
            f"{name}: inline dtype {dtype_name!r} is not a bool / integer / float dtype"
        )
    try:
        arr = np.asarray(raw, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PointReferenceError(f"{name}: inline points are not numeric: {exc}") from exc
    if arr.ndim == 0:
        raise PointReferenceError(f"{name}: inline points must be a list, got a scalar")
    if arr.shape[0] > INLINE_POINT_LIMIT:
        raise PointReferenceError(
            f"{name}: {arr.shape[0]} inline points exceed INLINE_POINT_LIMIT="
            f"{INLINE_POINT_LIMIT}; save them as an asset or reference a node field"
        )
    if arr.size > INLINE_ELEMENT_LIMIT:
        raise PointReferenceError(
            f"{name}: inline points of shape {arr.shape} hold {arr.size} numbers, more "
            f"than INLINE_ELEMENT_LIMIT={INLINE_ELEMENT_LIMIT}; save them as an asset"
        )
    if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
        raise PointReferenceError(f"{name}: inline points must be finite (no NaN / inf)")
    return arr


def _check_relative(rel: str, name: str) -> None:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise PointReferenceError(
            f"{name}: asset path {rel!r} must be relative to the config directory "
            f"and must not contain '..'"
        )


def _inlineable(arr: np.ndarray) -> bool:
    return (arr.ndim >= 1 and arr.dtype.kind in _NUMERIC_KINDS
            and arr.shape[0] <= INLINE_POINT_LIMIT and arr.size <= INLINE_ELEMENT_LIMIT
            and (arr.dtype.kind != "f" or bool(np.all(np.isfinite(arr)))))


def reference_for_array(array: Any, ref: Any, *, name: str, inline_ok: bool = True) -> Optional[dict]:
    """The reference a factory records for ``array``.

    ``ref`` given: normalised, checked against ``array`` and returned
    with the array's :func:`point_array_digest` as ``sha256`` (an
    inline reference must hold exactly ``array``; a node / asset
    reference that already carries a different hash is refused).
    ``ref`` is ``None``: the points are inlined when there are at most
    :data:`INLINE_POINT_LIMIT` of them (and ``inline_ok``), else
    ``None`` — the mapping is then usable but not serialisable.
    """
    arr = np.asarray(array)
    if ref is None:
        if not inline_ok or not _inlineable(arr):
            return None
        return {"inline": arr.tolist(), "dtype": str(arr.dtype)}
    ref = normalise_point_reference(ref, name=name)
    if "inline" in ref:
        given = np.asarray(ref["inline"], dtype=ref["dtype"])
        if given.shape != arr.shape or not np.array_equal(given, arr):
            raise PointReferenceError(
                f"{name}: the inline reference (shape {given.shape}) does not hold the "
                f"points the mapping was built from (shape {arr.shape}); pass the same "
                f"array or leave the reference out to inline it automatically"
            )
        return ref
    digest = point_array_digest(arr)
    recorded = ref.get("sha256")
    if recorded is not None and recorded != digest:
        raise PointReferenceError(
            f"{name}: reference {_without_hash(ref)!r} carries sha256 {_short(recorded)} "
            f"but the points the mapping was built from hash to {_short(digest)}; the "
            f"reference does not describe these points"
        )
    return {**ref, "sha256": digest}


@stability(StabilityLevel.EVOLVING)
def make_point_resolver(graph=None, base_dir=None) -> Callable[[dict], np.ndarray]:
    """A ``resolve_points`` callable for :meth:`MappingSpec.build`.

    Parameters
    ----------
    graph : GraphManager, optional
        Where ``{"node", "field"}`` references are looked up (its nodes'
        ``static_data`` and array-valued ``params``, through
        ``graph.node_names`` / ``graph.get_node``).  Without it a node
        reference is an error.
    base_dir : path-like, optional
        Directory ``{"asset"}`` paths are relative to; the current
        working directory when omitted.  Symlinks are resolved and the
        file must stay under it.
    """
    base = Path(base_dir) if base_dir is not None else Path.cwd()

    def resolve(ref: dict) -> np.ndarray:
        ref = normalise_point_reference(ref)
        if "inline" in ref:
            return np.asarray(ref["inline"], dtype=ref["dtype"])
        if "asset" in ref:
            return _load_asset(base, ref)
        return _node_field(graph, ref["node"], ref["field"])

    return resolve


# ---------------------------------------------------------------------------
# Asset files
# ---------------------------------------------------------------------------


def _load_asset(base: Path, ref: dict) -> np.ndarray:
    rel = ref["asset"]
    base_real = base.resolve()
    try:
        real = (base / rel).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        # missing, symlink loop, not a directory, embedded NUL byte
        raise PointReferenceError(
            f"missing point asset {rel!r} (looked in {base}): {exc}; pass base_dir= "
            f"pointing at the directory the config was saved in, or save the array "
            f"there with numpy.save"
        ) from exc
    if not real.is_relative_to(base_real):
        raise PointReferenceError(
            f"asset {rel!r} resolves to {real}, outside the config directory "
            f"{base_real}; symlinks are followed and the file must stay under it"
        )
    if not real.is_file():
        raise PointReferenceError(f"point asset {rel!r} ({real}) is not a file")
    where = f"asset {rel!r}"
    if Path(rel).suffix.lower() == ".npz":
        return _load_npz_member(real, ref, where)
    with open(real, "rb") as fp:
        shape, dtype = _read_npy_header(fp, where)
        _check_declared_size(shape, dtype, where, available=real.stat().st_size - fp.tell())
    return np.asarray(np.load(real, allow_pickle=False))


def _read_npy_header(fp, where: str) -> tuple[tuple, np.dtype]:
    """``(shape, dtype)`` from a ``.npy`` header, without touching the data."""
    try:
        version = np.lib.format.read_magic(fp)
        if version == (1, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_1_0(fp)
        elif version == (2, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_2_0(fp)
        else:
            raise PointReferenceError(
                f"{where}: .npy format version {version} is not supported for point "
                f"sets (save a plain numeric array)"
            )
    except PointReferenceError:
        raise
    except ValueError as exc:                    # bad magic, truncated / corrupt header
        raise PointReferenceError(f"{where}: not a valid .npy file: {exc}") from exc
    return tuple(shape), np.dtype(dtype)


def _check_declared_size(shape: tuple, dtype: np.dtype, where: str,
                         *, available: Optional[int] = None) -> None:
    if dtype.kind not in _NUMERIC_KINDS:
        raise PointReferenceError(
            f"{where}: dtype {dtype} is not a bool / integer / float dtype"
        )
    n_bytes = math.prod(int(s) for s in shape) * dtype.itemsize
    if n_bytes > MAX_ASSET_BYTES:
        raise PointReferenceError(
            f"{where}: header declares an array of shape {shape} {dtype} = {n_bytes} bytes, "
            f"more than MAX_ASSET_BYTES={MAX_ASSET_BYTES} ({MAX_ASSET_BYTES >> 20} MiB); "
            f"refused before loading (raise maddening.core.coupling.mapping_spec."
            f"MAX_ASSET_BYTES for a genuinely large interface)"
        )
    if available is not None and n_bytes > available:
        raise PointReferenceError(
            f"{where}: header declares {n_bytes} bytes of data but the file holds only "
            f"{available}; the file is truncated or its header was forged"
        )


def _load_npz_member(real: Path, ref: dict, where: str) -> np.ndarray:
    try:
        with zipfile.ZipFile(real) as zf:
            infos = {}
            for info in zf.infolist():
                name = info.filename[:-4] if info.filename.endswith(".npy") else info.filename
                infos[name] = info
            keys = list(infos)
            key = ref.get("key")
            if key is None:
                if len(keys) != 1:
                    raise PointReferenceError(
                        f"{where} has members {keys}; add 'key' to the reference"
                    )
                key = keys[0]
            if key not in infos:
                raise PointReferenceError(f"{where} has no member {key!r}; it has {keys}")
            info = infos[key]
            member = f"{where} member {key!r}"
            if info.file_size > MAX_ASSET_BYTES:
                raise PointReferenceError(
                    f"{member} is {info.file_size} bytes uncompressed (zip directory), "
                    f"more than MAX_ASSET_BYTES={MAX_ASSET_BYTES} "
                    f"({MAX_ASSET_BYTES >> 20} MiB); refused before decompression"
                )
            with zf.open(info) as fp:
                shape, dtype = _read_npy_header(fp, member)
                _check_declared_size(shape, dtype, member,
                                     available=info.file_size - fp.tell())
    except zipfile.BadZipFile as exc:
        raise PointReferenceError(f"{where} is not a valid .npz (zip) archive: {exc}") from exc
    with np.load(real, allow_pickle=False) as data:
        return np.asarray(data[key])


# ---------------------------------------------------------------------------
# Node fields
# ---------------------------------------------------------------------------


def _node_field(graph, node_name: str, field_name: str) -> np.ndarray:
    if graph is None:
        raise PointReferenceError(
            f"cannot resolve {{'node': {node_name!r}, 'field': {field_name!r}}} without "
            f"a graph"
        )
    names = list(graph.node_names)
    if node_name not in names:
        raise PointReferenceError(
            f"point reference names unknown node {node_name!r}; the graph has "
            f"{sorted(names)}"
        )
    node = graph.get_node(node_name)
    static = getattr(node, "static_data", None) or {}
    if not isinstance(static, dict):
        static = {}
    params = getattr(node, "params", None) or {}
    if not isinstance(params, dict):
        params = {}

    def _as_point_array(value, source: str) -> np.ndarray:
        value = getattr(value, "value", value)     # StaticArray -> array
        arr = np.asarray(value)
        if arr.ndim == 0:
            raise PointReferenceError(
                f"node {node_name!r} {source} {field_name!r} is a scalar "
                f"({arr.dtype}), not a point set"
            )
        if arr.dtype.kind not in _NUMERIC_KINDS:
            raise PointReferenceError(
                f"node {node_name!r} {source} {field_name!r} has dtype {arr.dtype}, "
                f"not a numeric point set"
            )
        return arr

    if field_name in static:
        return _as_point_array(static[field_name], "static_data field")
    value = params.get(field_name)
    if value is not None and not isinstance(value, (str, bytes, dict)):
        arr = np.asarray(value)
        if arr.ndim >= 1:
            return _as_point_array(arr, "parameter")
    available = sorted(set(static) | {
        k for k, v in params.items()
        if v is not None and not isinstance(v, (str, bytes, dict))
        and np.ndim(v) >= 1
    })
    hint = ""
    if not available:
        # Wrapper nodes (ShardedStencilNode, ShardedUnstructuredNode) hold
        # the inner node's static data privately and expose none of their
        # own, so a node reference cannot reach it.
        hint = (" (the node exposes no array field at all — a wrapper such as "
                "ShardedStencilNode / ShardedUnstructuredNode does not re-export the "
                "static_data of the node it wraps; save the points as an "
                "{'asset': '<file>.npy'} next to the config instead)")
    raise PointReferenceError(
        f"node {node_name!r} has no point field {field_name!r}; its array fields are "
        f"{available}{hint}"
    )


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


def _check_hyperparameter(kind: str, key: str, value: Any) -> Any:
    expected = _HYPER_TYPES[key]
    if expected == "real":
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(
                f"mapping kind {kind!r}: hyper-parameter {key!r} must be a real number, "
                f"got {value!r}"
            )
        if not math.isfinite(value):
            raise ValueError(
                f"mapping kind {kind!r}: hyper-parameter {key!r} must be finite, got "
                f"{value!r} (JSON cannot represent it)"
            )
        return float(value)
    if not isinstance(value, expected):
        raise ValueError(
            f"mapping kind {kind!r}: hyper-parameter {key!r} must be a "
            f"{expected.__name__}, got {value!r}"
        )
    return value


@stability(StabilityLevel.EVOLVING)
@dataclass(frozen=True)
class MappingSpec:
    """Recipe of a factory-built mapping: kind, hyper-parameters, point references.

    Parameters
    ----------
    kind : str
        ``"rbf"``, ``"nearest_neighbor"``, ``"projection_1d"`` or ``"matrix"``.
    hyperparameters : dict
        The factory's non-array keyword arguments (``kernel``, ``epsilon``,
        ``mode``, ...), JSON-able and type-checked (``epsilon`` / ``ridge``
        finite reals, ``polynomial`` a bool, the rest strings).  For
        ``"matrix"`` a ``label`` entry is the user-facing ``kind`` label
        of ``matrix_mapping``.
    points : dict
        Array-argument name → point reference (see the module docstring),
        or ``None`` for a set that was not recorded.

    ``to_dict()`` flattens to ``{"kind": ..., <hyper-parameters>...,
    "points": {...}}``; ``from_dict`` accepts that plus the ``shape`` key
    ``StaticLinearMapping.describe`` adds, and the user label under
    ``kind`` when ``label`` repeats it (what ``describe()`` writes for a
    labelled matrix mapping).
    """
    kind: str
    hyperparameters: dict = field(default_factory=dict)
    points: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.kind, str) or self.kind not in _FACTORIES:
            raise ValueError(
                f"unknown mapping kind {self.kind!r}; choose from {sorted(_FACTORIES)}"
            )
        if not isinstance(self.hyperparameters, dict):
            raise ValueError(
                f"mapping kind {self.kind!r}: hyperparameters must be a dict, got "
                f"{type(self.hyperparameters).__name__}"
            )
        if not isinstance(self.points, dict):
            raise ValueError(
                f"mapping kind {self.kind!r}: 'points' must be a dict mapping each "
                f"array argument to a point reference, got {type(self.points).__name__}"
            )
        _, array_names, hyper_names = _FACTORIES[self.kind]
        unknown = set(self.hyperparameters) - set(hyper_names)
        if unknown:
            raise ValueError(
                f"mapping kind {self.kind!r} has no hyper-parameter(s) "
                f"{sorted(unknown)}; it takes {list(hyper_names)}"
            )
        hyper = {k: _check_hyperparameter(self.kind, k, v)
                 for k, v in self.hyperparameters.items()}
        missing = set(array_names) - set(self.points)
        extra = set(self.points) - set(array_names)
        if missing or extra:
            raise ValueError(
                f"mapping kind {self.kind!r} takes point sets {list(array_names)}; "
                f"got {sorted(self.points)}"
            )
        points = {
            n: None if self.points[n] is None
            else normalise_point_reference(self.points[n], name=n)
            for n in array_names
        }
        object.__setattr__(self, "hyperparameters", hyper)
        object.__setattr__(self, "points", points)

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-able ``{"kind", <hyper-parameters>, "points"}`` (no weights)."""
        return {"kind": self.kind, **self.hyperparameters, "points": dict(self.points)}

    @classmethod
    def from_dict(cls, d: dict) -> "MappingSpec":
        """Inverse of :meth:`to_dict`; tolerates the extra keys of
        ``StaticLinearMapping.describe`` (``shape``, and the user label
        of a matrix mapping in place of ``kind``)."""
        if not isinstance(d, dict) or "kind" not in d:
            raise ValueError(f"a mapping spec is a dict with a 'kind' key, got {d!r}")
        kind = d["kind"]
        if "points" not in d:
            raise ValueError(
                f"mapping spec of kind {kind!r} has no 'points'; a mapping saved "
                f"without point references cannot be rebuilt"
            )
        if not isinstance(d["points"], dict):
            raise ValueError(
                f"mapping spec of kind {kind!r}: 'points' must be a dict, got "
                f"{type(d['points']).__name__}"
            )
        label = d.get("label")
        if isinstance(label, str) and kind == label:
            # ``describe()`` of ``matrix_mapping(kind="supermesh")`` reports
            # the user label as ``kind`` and repeats it as ``label``; only
            # ``matrix`` has a ``label``, so the pair identifies the factory
            # even when the user label collides with another kind's name.
            kind = "matrix"
        hyper = {k: v for k, v in d.items() if k not in _RESERVED_KEYS}
        # ``kind`` is untrusted: an unhashable one (a list, a dict) must
        # reach the "unknown mapping kind" ValueError of ``__post_init__``,
        # not come back out of this lookup as ``TypeError: unhashable type``,
        # which names neither the key nor what is wrong with it.
        if isinstance(kind, str) and kind in _FACTORIES \
                and "mode" not in _FACTORIES[kind][2]:
            # ``describe()`` reports the mode of every mapping; a factory
            # with a fixed mode (projection_1d) does not take it back.
            hyper.pop("mode", None)
        return cls(kind=kind, hyperparameters=hyper, points=dict(d["points"]))

    def __hash__(self) -> int:
        return hash(json.dumps(self.to_dict(), sort_keys=True, default=str))

    # -- rebuild --------------------------------------------------------

    def missing_points(self) -> list[str]:
        """Names of the point sets that were not recorded (``None``)."""
        return [n for n, r in self.points.items() if r is None]

    def build(self, resolve_points: Callable[[dict], Any]):
        """Rebuild the mapping by calling its factory on the resolved points.

        ``resolve_points(ref) -> array`` turns each reference into the
        array (see :func:`make_point_resolver`); a reference with a
        ``sha256`` must resolve to exactly the recorded array.  The
        references are passed back to the factory, so the rebuilt
        mapping carries this spec again.
        """
        return build_mapping(self, resolve_points)


def _check_digest(name: str, ref: dict, array: Any) -> None:
    recorded = ref.get("sha256")
    if recorded is None:
        return
    arr = np.asarray(array)
    got = point_array_digest(arr)
    if got != recorded:
        raise PointReferenceError(
            f"{name}: reference {_without_hash(ref)!r} resolves to an array "
            f"({arr.dtype}{arr.shape}, sha256 {_short(got)}) that differs from the points "
            f"the mapping was built from (sha256 {_short(recorded)}); the referenced "
            f"field / file changed since the mapping was saved, or the reference names "
            f"the wrong points"
        )


@stability(StabilityLevel.EVOLVING)
def build_mapping(spec: MappingSpec, resolve_points: Callable[[dict], Any]):
    """Recompute the mapping described by ``spec``; see :meth:`MappingSpec.build`."""
    from maddening.core.coupling import mapping as _factories  # noqa: PLC0415

    factory_name, array_names, _ = _FACTORIES[spec.kind]
    missing = spec.missing_points()
    if missing:
        raise PointReferenceError(
            f"mapping spec {spec.kind!r} has no reference for {missing}; it was built "
            f"from a point set too large to inline and without "
            f"{' / '.join(sorted({_REF_KWARG[n] + '=' for n in missing}))}"
        )
    arrays = {n: resolve_points(spec.points[n]) for n in array_names}
    for n in array_names:
        _check_digest(n, spec.points[n], arrays[n])
    refs = {_REF_KWARG[n]: spec.points[n] for n in array_names}
    hyper = dict(spec.hyperparameters)
    if spec.kind == "matrix" and "label" in hyper:
        hyper["kind"] = hyper.pop("label")
    factory = getattr(_factories, factory_name)
    return factory(**arrays, **hyper, **refs)


def verify_node_references(spec: MappingSpec, resolve_points: Callable[[dict], Any]) -> None:
    """Check that every ``{"node", "field"}`` reference of ``spec`` still
    resolves to the points the mapping was built from.

    A reference that does not resolve (the node was removed, the field
    renamed), or that resolves to an array with a different
    :func:`point_array_digest`, is a :class:`PointReferenceError`.
    Asset and inline references are not checked here — assets are
    verified when they are loaded (reading them at write time would be
    a surprise), inline sets carry their data.
    """
    for name, ref in spec.points.items():
        if ref is None or "node" not in ref:
            continue
        _check_digest(name, ref, resolve_points(ref))


def check_mapping_serialisable(mapping: Any, *, edge_key: str = "",
                               resolve_points: Optional[Callable[[dict], Any]] = None,
                               ) -> MappingSpec:
    """The mapping's complete :class:`MappingSpec`, or a ``ValueError``
    saying why the mapping cannot be written to a config / USD stage.

    With ``resolve_points`` (the graph's resolver) the node references
    are resolved and compared with the recorded hashes, so a stale
    reference — a removed node, a field that changed — is refused at
    write time instead of at the next load.
    """
    where = f" on edge {edge_key}" if edge_key else ""
    spec = getattr(mapping, "spec", None)
    if not isinstance(spec, MappingSpec):
        raise ValueError(
            f"mapping {mapping!r}{where} carries no MappingSpec and cannot be "
            f"serialised; build it with rbf_mapping / nearest_neighbor_mapping / "
            f"projection_1d_mapping / matrix_mapping(asset=...)"
        )
    missing = spec.missing_points()
    if missing:
        hint = ", ".join(sorted({_REF_KWARG[n] + "=" for n in missing}))
        raise ValueError(
            f"mapping {mapping!r}{where} cannot be serialised: point set(s) {missing} "
            f"were not recorded (more than {INLINE_POINT_LIMIT} points, a non-numeric "
            f"or non-finite set, or an explicit matrix).  Pass {hint} to the factory — "
            f"a node field {{'node': <name>, 'field': <key>}} or an asset "
            f"{{'asset': '<file>.npy'}} saved next to the config"
        )
    if resolve_points is not None:
        try:
            verify_node_references(spec, resolve_points)
        except PointReferenceError as exc:
            raise ValueError(
                f"mapping {mapping!r}{where} cannot be serialised: a point reference "
                f"no longer describes the points it was built from — {exc}.  Rebuild "
                f"the mapping from the current graph, or reference the right node field"
            ) from exc
    return spec


__all__ = [
    "INLINE_ELEMENT_LIMIT",
    "INLINE_POINT_LIMIT",
    "MAX_ASSET_BYTES",
    "MappingRebuildError",
    "MappingSpec",
    "PointReferenceError",
    "build_mapping",
    "check_mapping_serialisable",
    "make_point_resolver",
    "normalise_point_reference",
    "point_array_digest",
    "reference_for_array",
    "verify_node_references",
]
