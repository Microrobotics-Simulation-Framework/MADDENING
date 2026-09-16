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
values when loaded on top of a config.

Point references
----------------
A point set is described by reference, not inlined:

``{"node": "<name>", "field": "<key>"}``
    a node's ``static_data[key]`` (a ``StaticArray`` is unwrapped) or,
    failing that, an array-valued constructor parameter
    ``node.params[key]`` — e.g. ``{"node": "rod", "field": "grid_x"}``
    for a ``HeatNode``;
``{"asset": "<relative path>.npy"}`` / ``{"asset": "<path>.npz", "key": "<k>"}``
    an external NumPy file, relative to the directory the config / USD
    stage lives in (``base_dir``); absolute paths and ``..`` components
    are refused so a config cannot read outside its own directory;
``{"inline": [[...], ...], "dtype": "float64"}`` (or a plain list)
    the points themselves, only for small sets — at most
    :data:`INLINE_POINT_LIMIT` points; the factories inline
    automatically when no reference is given and the set is that small.

A larger point set without a reference is not serialisable: the
factory leaves the entry ``None`` and ``GraphManager.to_dict`` refuses
with a message naming the argument (``source_ref=`` / ``target_ref=`` /
``asset=``) to pass.  An explicit matrix (``matrix_mapping``) is always
stored as an ``asset`` reference, never inlined.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

#: Point sets with at most this many points are inlined into the spec
#: when the factory is given no reference for them.
INLINE_POINT_LIMIT = 64

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

# array-argument name -> the factory keyword that carries its reference
_REF_KWARG = {
    "source_points": "source_ref", "target_points": "target_ref",
    "source_boundaries": "source_ref", "target_boundaries": "target_ref",
    "H": "asset",
}

_RESERVED_KEYS = ("kind", "points", "shape")


class PointReferenceError(ValueError):
    """A point reference is malformed or cannot be resolved."""


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


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
        if set(ref) != {"node", "field"} or not all(
                isinstance(ref[k], str) and ref[k] for k in ("node", "field")):
            raise PointReferenceError(
                f"{name}: a node reference is {{'node': <name>, 'field': <key>}} "
                f"with string values, got {ref!r}"
            )
        return {"node": ref["node"], "field": ref["field"]}
    if "asset" in ref:
        if set(ref) - {"asset", "key"} or not isinstance(ref["asset"], str) or not ref["asset"]:
            raise PointReferenceError(
                f"{name}: an asset reference is {{'asset': <relative path>}} "
                f"(plus 'key' for an .npz member), got {ref!r}"
            )
        _check_relative(ref["asset"], name)
        out = {"asset": ref["asset"]}
        if "key" in ref:
            if not isinstance(ref["key"], str):
                raise PointReferenceError(f"{name}: asset 'key' must be a string")
            out["key"] = ref["key"]
        return out
    if "inline" in ref:
        if set(ref) - {"inline", "dtype"}:
            raise PointReferenceError(
                f"{name}: an inline reference is {{'inline': [...], 'dtype': <name>}}, "
                f"got keys {sorted(ref)}"
            )
        try:
            arr = np.asarray(ref["inline"], dtype=ref.get("dtype", "float64"))
        except (TypeError, ValueError) as exc:
            raise PointReferenceError(f"{name}: inline points are not numeric: {exc}") from exc
        if arr.ndim == 0:
            raise PointReferenceError(f"{name}: inline points must be a list, got a scalar")
        if arr.shape[0] > INLINE_POINT_LIMIT:
            raise PointReferenceError(
                f"{name}: {arr.shape[0]} inline points exceed INLINE_POINT_LIMIT="
                f"{INLINE_POINT_LIMIT}; save them as an asset or reference a node field"
            )
        return {"inline": arr.tolist(), "dtype": str(arr.dtype)}
    raise PointReferenceError(
        f"{name}: unknown point reference {ref!r}; expected one of 'node'/'field', "
        f"'asset' or 'inline'"
    )


def _check_relative(rel: str, name: str) -> None:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise PointReferenceError(
            f"{name}: asset path {rel!r} must be relative to the config directory "
            f"and must not contain '..'"
        )


def reference_for_array(array: Any, ref: Any, *, name: str, inline_ok: bool = True) -> Optional[dict]:
    """The reference a factory records for ``array``.

    ``ref`` given: normalised and returned as is.  ``ref`` is ``None``:
    the points are inlined when there are at most
    :data:`INLINE_POINT_LIMIT` of them (and ``inline_ok``), else ``None``
    — the mapping is then usable but not serialisable.
    """
    if ref is not None:
        return normalise_point_reference(ref, name=name)
    arr = np.asarray(array)
    if not inline_ok or arr.ndim == 0 or arr.shape[0] > INLINE_POINT_LIMIT:
        return None
    return {"inline": arr.tolist(), "dtype": str(arr.dtype)}


@stability(StabilityLevel.EVOLVING)
def make_point_resolver(graph=None, base_dir=None) -> Callable[[dict], np.ndarray]:
    """A ``resolve_points`` callable for :meth:`MappingSpec.build`.

    Parameters
    ----------
    graph : GraphManager, optional
        Where ``{"node", "field"}`` references are looked up (its nodes'
        ``static_data`` and array-valued ``params``).  Without it a node
        reference is an error.
    base_dir : path-like, optional
        Directory ``{"asset"}`` paths are relative to; the current
        working directory when omitted.
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


def _load_asset(base: Path, ref: dict) -> np.ndarray:
    path = base / ref["asset"]
    if not path.is_file():
        raise PointReferenceError(
            f"missing point asset {ref['asset']!r} (looked in {base}); pass base_dir= "
            f"pointing at the directory the config was saved in, or save the array "
            f"there with numpy.save"
        )
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        with data:
            keys = list(data.files)
            key = ref.get("key")
            if key is None:
                if len(keys) != 1:
                    raise PointReferenceError(
                        f"asset {ref['asset']!r} has members {keys}; add 'key' to the "
                        f"reference"
                    )
                key = keys[0]
            if key not in keys:
                raise PointReferenceError(
                    f"asset {ref['asset']!r} has no member {key!r}; it has {keys}"
                )
            return np.asarray(data[key])
    return np.asarray(data)


def _node_field(graph, node_name: str, field_name: str) -> np.ndarray:
    if graph is None:
        raise PointReferenceError(
            f"cannot resolve {{'node': {node_name!r}, 'field': {field_name!r}}} without "
            f"a graph"
        )
    nodes = getattr(graph, "_nodes", {})
    if node_name not in nodes:
        raise PointReferenceError(
            f"point reference names unknown node {node_name!r}; the graph has "
            f"{sorted(nodes)}"
        )
    node = nodes[node_name].node
    static = {}
    try:
        static = dict(node.static_data or {})
    except Exception:  # noqa: BLE001 - a node may not implement static_data
        static = {}
    if field_name in static:
        value = static[field_name]
        value = getattr(value, "value", value)     # StaticArray -> array
        return np.asarray(value)
    params = getattr(node, "params", {}) or {}
    value = params.get(field_name)
    if value is not None and not isinstance(value, (str, bytes, dict)):
        arr = np.asarray(value)
        if arr.ndim >= 1:
            return arr
    available = sorted(static) + sorted(
        k for k, v in params.items()
        if v is not None and not isinstance(v, (str, bytes, dict))
        and np.ndim(v) >= 1
    )
    raise PointReferenceError(
        f"node {node_name!r} has no point field {field_name!r}; its array fields are "
        f"{available}"
    )


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


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
        ``mode``, ...), JSON-able.  For ``"matrix"`` a ``label`` entry is
        the user-facing ``kind`` label of ``matrix_mapping``.
    points : dict
        Array-argument name → point reference (see the module docstring),
        or ``None`` for a set that was not recorded.

    ``to_dict()`` flattens to ``{"kind": ..., <hyper-parameters>...,
    "points": {...}}``; ``from_dict`` accepts that plus the ``shape`` key
    ``StaticLinearMapping.describe`` adds.
    """
    kind: str
    hyperparameters: dict = field(default_factory=dict)
    points: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in _FACTORIES:
            raise ValueError(
                f"unknown mapping kind {self.kind!r}; choose from {sorted(_FACTORIES)}"
            )
        _, array_names, hyper_names = _FACTORIES[self.kind]
        unknown = set(self.hyperparameters) - set(hyper_names)
        if unknown:
            raise ValueError(
                f"mapping kind {self.kind!r} has no hyper-parameter(s) "
                f"{sorted(unknown)}; it takes {list(hyper_names)}"
            )
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
        object.__setattr__(self, "hyperparameters", dict(self.hyperparameters))
        object.__setattr__(self, "points", points)

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-able ``{"kind", <hyper-parameters>, "points"}`` (no weights)."""
        return {"kind": self.kind, **self.hyperparameters, "points": dict(self.points)}

    @classmethod
    def from_dict(cls, d: dict) -> "MappingSpec":
        """Inverse of :meth:`to_dict`; tolerates the extra keys of
        ``StaticLinearMapping.describe`` (``shape``)."""
        if not isinstance(d, dict) or "kind" not in d:
            raise ValueError(f"a mapping spec is a dict with a 'kind' key, got {d!r}")
        if "points" not in d:
            raise ValueError(
                f"mapping spec of kind {d['kind']!r} has no 'points'; a mapping saved "
                f"without point references cannot be rebuilt"
            )
        hyper = {k: v for k, v in d.items() if k not in _RESERVED_KEYS}
        if d["kind"] in _FACTORIES and "mode" not in _FACTORIES[d["kind"]][2]:
            # ``describe()`` reports the mode of every mapping; a factory
            # with a fixed mode (projection_1d) does not take it back.
            hyper.pop("mode", None)
        return cls(kind=d["kind"], hyperparameters=hyper, points=dict(d["points"] or {}))

    def __hash__(self) -> int:
        return hash(json.dumps(self.to_dict(), sort_keys=True, default=str))

    # -- rebuild --------------------------------------------------------

    def missing_points(self) -> list[str]:
        """Names of the point sets that were not recorded (``None``)."""
        return [n for n, r in self.points.items() if r is None]

    def build(self, resolve_points: Callable[[dict], Any]):
        """Rebuild the mapping by calling its factory on the resolved points.

        ``resolve_points(ref) -> array`` turns each reference into the
        array (see :func:`make_point_resolver`).  The references are
        passed back to the factory, so the rebuilt mapping carries this
        spec again.
        """
        return build_mapping(self, resolve_points)


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
    refs = {_REF_KWARG[n]: spec.points[n] for n in array_names}
    hyper = dict(spec.hyperparameters)
    if spec.kind == "matrix" and "label" in hyper:
        hyper["kind"] = hyper.pop("label")
    factory = getattr(_factories, factory_name)
    return factory(**arrays, **hyper, **refs)


def check_mapping_serialisable(mapping: Any, *, edge_key: str = "") -> MappingSpec:
    """The mapping's complete :class:`MappingSpec`, or a ``ValueError``
    saying why the mapping cannot be written to a config / USD stage."""
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
            f"were not recorded (more than {INLINE_POINT_LIMIT} points, or an explicit "
            f"matrix).  Pass {hint} to the factory — a node field "
            f"{{'node': <name>, 'field': <key>}} or an asset {{'asset': '<file>.npy'}} "
            f"saved next to the config"
        )
    return spec


__all__ = [
    "INLINE_POINT_LIMIT",
    "MappingSpec",
    "PointReferenceError",
    "build_mapping",
    "check_mapping_serialisable",
    "make_point_resolver",
    "normalise_point_reference",
    "reference_for_array",
]
