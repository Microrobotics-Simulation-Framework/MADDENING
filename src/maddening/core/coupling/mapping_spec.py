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
    for a ``HeatNode``.  The coordinates are read **once**, when the
    mapping is built, and the weights are fixed from then on — see
    *A node reference does not follow a calibrated parameter* below;
``{"asset": "<relative path>.npy", "sha256": ...}`` /
``{"asset": "<path>.npz", "key": "<k>", "sha256": ...}``
    an external NumPy file, relative to the directory the config / USD
    stage lives in (``base_dir``).  Absolute paths and ``..`` components
    are refused, symlinks are resolved and the resolved file must still
    lie under the resolved ``base_dir``, so a config cannot read outside
    its own directory.  That resolved path is then opened **once**
    (``O_NOFOLLOW``), and the size check, the header and the data all
    come from that one descriptor, so no second lookup of the name can
    land on a different file.  An array larger than
    :data:`MAX_ASSET_BYTES` (or larger than the file that claims to hold
    it) is refused before anything is allocated; only bool / integer /
    float arrays are accepted;
``{"inline": [[...], ...], "dtype": "float64"}`` (or a plain list)
    the points themselves, only for small sets — at most
    :data:`INLINE_POINT_LIMIT` points and :data:`INLINE_ELEMENT_LIMIT`
    numbers in total, finite, of a bool / integer / float dtype; the
    factories inline automatically when no reference is given and the
    set is that small.

A node reference does not follow a calibrated parameter (MADD-ANO-022)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A ``{"node", "field"}`` reference resolves the node's static data at the
moment the mapping is built, and the weights computed from it are
snapshotted into ``gm.params["mappings"]`` at compile time.  If that
static is *derived from a trainable parameter*, calibrating the
parameter through ``gm.params`` moves the node and leaves the mapping at
the constructor's geometry.  The in-tree case is the default, uniform
``HeatNode``: ``grid_x`` is built in ``__init__`` from ``length``, and
``length`` is trainable, because the node's own step reads the traced
``length`` and never ``grid_x``.  So ``static_data_deps`` declares
nothing and ``compile()`` accepts the graph.  A mapped edge on
``grid_x`` then interpolates from the old grid.  Measured on an 8-cell
source rod calibrated from ``length`` 1.0 to 1.25 and mapped onto a
16-cell rod:

* the target moves by about ``4e-4`` where the same graph constructed at
  1.25 moves it by about ``1.2e-2``;
* the gradient of the target with respect to ``length`` has the wrong
  sign (``-6.4e-3`` against ``+2.7e-1``);
* a fit of ``length`` from the target's data converges to about 0.22
  instead of 1.25.

Nothing refuses the graph or warns, and the recorded ``sha256`` still
matches, because the static itself never changed.

Until a fix lands (being scoped for 0.5.0), use one of these:

* declare the parameter non-trainable when a mapped edge references a
  grid it derives, e.g. ``gm.set_param_spec("rod", "length",
  ParamSpec(trainable=False))``, so no fit can move it;
* give the mapping explicit coordinates, as an ``{"asset"}`` or
  ``{"inline"}`` reference, so the config states that the mapping's
  geometry is fixed rather than implying that it follows the node;
* if the geometry must be calibrated, fit it from observations of the
  node itself (not through the mapped edge), then rebuild the graph at
  the fitted value so the mapping is rebuilt from the new grid.

Accepted dtypes
---------------
Whatever the reference form, a point set must be a bool, integer or
float array of at most :data:`_MAX_ELEMENT_BYTES` bytes per element:
``bool``, ``int8``…``int64``, ``uint8``…``uint64``, ``float16``,
``float32``, ``float64``.  Complex, string, object and datetime arrays
are refused, and so is **extended precision** — ``np.longdouble``,
spelled ``float96`` on 32-bit x86 and ``float128`` on x86-64 and
aarch64.  An extended-precision point set is refused with an error that
says so, *not* narrowed to ``float64``: it cannot be written
(``arr.tolist()`` yields ``np.longdouble`` objects and ``json.dumps``
refuses them) and its :func:`point_array_digest` is not stable (the
padding bytes of an 80-bit value in a 16-byte slot are not zeroed, so
arrays that compare equal can hash differently).  Silently narrowing it
would hide a precision loss the caller did not ask for, which is a
well-known source of hard-to-trace numerical bugs; the refusal names the
dtype and the fix (``np.asarray(points, dtype=np.float64)``).

The same holds for the *values* of an inline reference that names no
``"dtype"``.  Such a payload is read as ``float64``, and
``np.longdouble(...).tolist()`` yields ``np.longdouble`` scalars rather
than Python floats, so an extended-precision payload (scalars in a
list, or an array) with no ``"dtype"`` used to be rounded to
``float64`` without a word.  It is refused, and the message says that
no reference form can keep the extra precision and how to go ahead:
convert to ``float64`` first, add ``"dtype": "float64"`` to accept the
rounding explicitly, or save a larger set as a ``.npy`` asset.

This is a limit of the serialised form, not a judgement about extended
precision.  If a real interface ever needs it, it can be added later
behind the same API — a composite inline representation (the value
bytes, or a mantissa / exponent pair, as JSON-safe integers) together
with a canonical digest, or an FFI path that formats and hashes the
value itself.  Nothing in this module assumes 8 bytes is the last word.

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
import os
import re
import stat
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

#: Widest element (bytes) a point set may have.  Every bool / integer /
#: float of at most this width is rendered by ``ndarray.tolist()`` as a
#: *Python* scalar, which ``json.dumps`` can write and which round-trips
#: through a config; an extended-precision float (``np.longdouble``:
#: ``float96`` on 32-bit x86, ``float128`` on x86-64 and aarch64) does
#: not, and its digest is not stable either.  Such point sets are
#: refused — never narrowed behind the user's back — by
#: :func:`_check_element_width`.
_MAX_ELEMENT_BYTES = 8

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
# Accepted element widths
# ---------------------------------------------------------------------------


def _check_element_width(dtype: Any, where: str, *, what: str = "dtype") -> None:
    """Refuse an extended-precision point set, loudly.

    A bool / integer / float of at most :data:`_MAX_ELEMENT_BYTES` bytes
    per element survives everything a spec does with it.  An
    extended-precision float (``np.longdouble``) survives neither step,
    so it is rejected here rather than quietly narrowed to ``float64``:
    a silent narrowing is the kind of precision loss that turns into a
    scientific-computing bug nobody can trace, and an explicit constraint
    is honest about what the format can carry.

    Parameters
    ----------
    dtype : numpy.dtype or dtype-like
        The dtype to check.
    where : str
        What is being checked, used as the prefix of the message (an
        argument name, a node field, an asset path).
    what : str, optional
        How to call the dtype in the message, e.g. ``"inline dtype"``.

    Raises
    ------
    PointReferenceError
        If ``dtype`` is wider than :data:`_MAX_ELEMENT_BYTES`.

    Notes
    -----
    The limit is a property of the *serialised* form, not of the
    physics.  Extended precision could be supported later without
    changing any of this — a composite representation (the value bytes,
    or a mantissa / exponent pair, written as JSON-safe integers) or an
    FFI path that hashes and formats the value itself — if a real
    interface ever needs it.  Nothing here is built on the assumption
    that 8 bytes is the last word.
    """
    dtype = np.dtype(dtype)
    if dtype.itemsize <= _MAX_ELEMENT_BYTES:
        return
    raise PointReferenceError(
        f"{where}: {what} {dtype.name!r} is an extended-precision float "
        f"({dtype.itemsize} bytes per element) and is not supported for point sets, "
        f"which are limited to bool / integer / float dtypes of at most "
        f"{_MAX_ELEMENT_BYTES} bytes (float16 / float32 / float64).  It is refused "
        f"rather than narrowed to float64 behind your back, because it cannot "
        f"round-trip: json.dumps cannot write it, since arr.tolist() on a "
        f"{dtype.name} array yields np.longdouble objects rather than Python floats, "
        f"so the config could not be saved; and point_array_digest is unstable for "
        f"it, because np.longdouble leaves the padding bytes of its slot "
        f"(an 80-bit value in {dtype.itemsize} bytes on x86-64) unzeroed, so two "
        f"arrays that compare equal can hash differently and a reference to them is "
        f"rejected at random.  If you do not need the extra precision, convert the "
        f"points yourself: np.asarray(points, dtype=np.float64)."
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

    An extended-precision array is a :class:`PointReferenceError`: the
    guarantee above does not hold for it (see
    :data:`_MAX_ELEMENT_BYTES`), so returning a digest would be a lie.
    """
    arr = np.ascontiguousarray(np.asarray(array))
    _check_element_width(arr.dtype, "point set")
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


def _is_extended_float(dtype: np.dtype) -> bool:
    """A real float wider than :data:`_MAX_ELEMENT_BYTES` (``np.longdouble``
    where it is wider than ``float64``)."""
    return dtype.kind == "f" and dtype.itemsize > _MAX_ELEMENT_BYTES


def _extended_precision_in(raw: Any) -> Optional[np.dtype]:
    """The first extended-precision float dtype in an inline payload, or ``None``.

    Walks the payload *without converting it*: lists and tuples item by
    item, NumPy arrays by dtype (an object array element by element) and
    NumPy scalars by dtype.  ``np.longdouble.tolist()`` yields
    ``np.longdouble`` scalars, not Python floats, so a payload built that
    way carries the dtype on every element.

    Bounded by the element limit rather than by the payload: once more
    than :data:`INLINE_ELEMENT_LIMIT` numbers have been seen, or a single
    list or array holds more than that, whatever the payload coerces to
    has more numbers than the limit (or is empty, or ragged) and is
    refused after coercion anyway, so nothing past that point can be
    accepted narrowed.
    """
    stack = [raw]
    seen = 0
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            if len(item) > INLINE_ELEMENT_LIMIT:
                return None
            stack.extend(item)
            continue
        if isinstance(item, (np.ndarray, np.generic)):
            if _is_extended_float(item.dtype):
                return item.dtype
            if item.size > INLINE_ELEMENT_LIMIT:
                return None
            if isinstance(item, np.ndarray) and item.dtype.hasobject:
                stack.extend(item.ravel().tolist())
                continue
            seen += item.size
        else:
            seen += 1
        if seen > INLINE_ELEMENT_LIMIT:
            return None
    return None


def _refuse_implicit_narrowing(raw: Any, name: str) -> None:
    """Refuse extended-precision inline values that carry no ``"dtype"``.

    Without a ``"dtype"`` an inline payload is read as ``float64``, and
    ``np.asarray(values, dtype=float64)`` rounds an ``np.longdouble``
    value without a word.  The coercion itself is deliberately left
    alone (it is the property-tested boundary for object and string
    payloads); this runs before it and fires only when the payload holds
    extended-precision values *and* the reference names no dtype.  An
    explicit ``"dtype": "float64"`` is the caller accepting the rounding
    and goes through unchanged.
    """
    found = _extended_precision_in(raw)
    if found is None:
        return
    ours, f64 = np.finfo(found), np.finfo(np.float64)
    raise PointReferenceError(
        f"{name}: the inline points hold extended-precision values ({found.name}, "
        f"about {ours.precision} significant digits) and the reference names no "
        f"'dtype'.  Inline points without a 'dtype' are read as float64 (about "
        f"{f64.precision} digits), so they would be rounded without a word; they "
        f"are refused instead.  Extended precision cannot be kept by any point "
        f"reference: inline, asset and node-field point sets are all limited to "
        f"{_MAX_ELEMENT_BYTES}-byte elements, so these points can be at most float64 "
        f"whichever way you pass them.  To go ahead: convert them yourself first, "
        f"np.asarray(points, dtype=np.float64).tolist() gives plain Python floats; "
        f"or keep the values and add 'dtype': 'float64' to the reference to accept "
        f"the rounding explicitly; or, for a set too large to inline (more than "
        f"{INLINE_POINT_LIMIT} points), numpy.save the float64 array next to the "
        f"config and pass {{'asset': '<file>.npy'}}."
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
    if ref.get("dtype") is None:
        # The default below would narrow an np.longdouble payload to
        # float64 silently; refuse that before the coercion.
        _refuse_implicit_narrowing(raw, name)
    dtype_name = ref.get("dtype", "float64")
    try:
        dtype = np.dtype(dtype_name)
    except TypeError as exc:
        raise PointReferenceError(f"{name}: unknown inline dtype {dtype_name!r}") from exc
    if dtype.kind not in _NUMERIC_KINDS:
        raise PointReferenceError(
            f"{name}: inline dtype {dtype_name!r} is not a bool / integer / float dtype"
        )
    _check_element_width(dtype, name, what="inline dtype")
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
            and arr.dtype.itemsize <= _MAX_ELEMENT_BYTES
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

    An extended-precision point set is a :class:`PointReferenceError`,
    not a quiet ``None``: it can be neither written nor hashed (see
    :data:`_MAX_ELEMENT_BYTES`), and "not serialisable" would leave the
    caller guessing which of the several reasons applied.
    """
    arr = np.asarray(array)
    _check_element_width(arr.dtype, name)
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


#: ``O_NOFOLLOW`` refuses a final path component that is a symlink;
#: ``O_NONBLOCK`` keeps a FIFO with an asset's name from hanging the
#: open until someone writes to it.  Both are POSIX-only: on a platform
#: without them the open degrades to a plain one (and the containment
#: check below still applies), which is why the path handling here is
#: only claimed to hold on POSIX.
_ASSET_OPEN_FLAGS = (os.O_RDONLY
                     | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0))


def _open_asset(real: Path, rel: str) -> tuple[Any, os.stat_result]:
    """The asset file opened **once**, as ``(file object, stat result)``.

    Everything downstream — the size cap, the "is it a regular file"
    check, the header and the data — must come from this one descriptor.
    Resolving the path and then opening it by path again leaves a window
    (time of check to time of use) in which a writer in the config
    directory can swap the checked file for a symlink, so the resolved
    path is opened once and never named again:

    * ``O_NOFOLLOW`` refuses a final component that is a symlink.
      ``real`` came out of :meth:`Path.resolve`, so in the honest case it
      is never one — including when the *reference* named a symlink,
      because that link was already followed to its target.  The flag
      therefore only fires on a component that became a link after the
      resolution, i.e. on the race.
    * :func:`os.fstat` on the descriptor, rather than ``stat`` on the
      path, decides what was actually opened.

    ``O_NOFOLLOW`` covers only the last component, so the containment
    check on the resolved path (which walks the parent directories)
    stays where it is.
    """
    try:
        fd = os.open(real, _ASSET_OPEN_FLAGS)
    except OSError as exc:
        raise PointReferenceError(
            f"cannot open point asset {rel!r} ({real}): {exc}; the file was removed, "
            f"replaced by a symlink or made unreadable after its path was resolved"
        ) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise PointReferenceError(f"point asset {rel!r} ({real}) is not a file")
        fp = os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise
    return fp, info


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
    where = f"asset {rel!r}"
    fp, info = _open_asset(real, rel)
    try:
        if Path(rel).suffix.lower() == ".npz":
            return _load_npz_member(fp, ref, where)
        shape, dtype = _read_npy_header(fp, where)
        _check_declared_size(shape, dtype, where, available=info.st_size - fp.tell())
        fp.seek(0)
        return np.asarray(np.load(fp, allow_pickle=False))
    finally:
        fp.close()


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
    _check_element_width(dtype, where)
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


def _load_npz_member(fp, ref: dict, where: str) -> np.ndarray:
    """A member of a ``.npz`` archive, read from an already-open ``fp``.

    ``fp`` is the descriptor :func:`_open_asset` returned: the zip
    directory and the member data come from the same open file, so the
    uncompressed-size check cannot be aimed at a different archive than
    the one that is decompressed.
    """
    try:
        with zipfile.ZipFile(fp) as zf:
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
            with zf.open(info) as member_fp:
                shape, dtype = _read_npy_header(member_fp, member)
                _check_declared_size(shape, dtype, member,
                                     available=info.file_size - member_fp.tell())
    except zipfile.BadZipFile as exc:
        raise PointReferenceError(f"{where} is not a valid .npz (zip) archive: {exc}") from exc
    fp.seek(0)
    with np.load(fp, allow_pickle=False) as data:
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
        _check_element_width(arr.dtype, f"node {node_name!r} {source} {field_name!r}")
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
        # A node with no array field at all.  This used to be the common
        # case for a wrapper, which held the inner node's static data
        # privately; ``SimulationNode.static_data`` now forwards to the
        # nodes a node wraps, so a reference reaches through a
        # ShardedStencilNode / HybridNode to the statics the wrapped node
        # declares, and reaching this branch means the field really is
        # absent everywhere.
        hint = (" (the node exposes no array field at all, including through "
                "any node it wraps; save the points as an "
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
