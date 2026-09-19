"""``fmi3GetFMUState`` / ``fmi3SetFMUState`` round-trip via MADDENING state.

The v0.2 #8 checkpoint manifest already serialises the full graph
state with an integrity manifest; this module re-exposes that same
serialiser behind the FMI 3.0 names so the FMU sidecar can call
``GetFMUState`` / ``SetFMUState`` without rebuilding the round-trip.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability


@stability(StabilityLevel.EVOLVING)
@dataclass(frozen=True)
class FMUState:
    """Opaque handle holding a frozen snapshot of a MADDENING graph state.

    The FMI 3.0 standard treats FMUState as opaque to the importer —
    the importer gets a handle from ``fmi3GetFMUState`` and feeds it
    back via ``fmi3SetFMUState`` later.  We mirror that here: the
    only operations on :class:`FMUState` are produced via
    :func:`serialize_fmu_state` / :func:`deserialize_fmu_state`.

    The internal payload is an ``npz`` archive of plain arrays, loaded
    with ``allow_pickle=False``: restoring a snapshot never executes
    anything the snapshot carries.  It used to be a pickle, which made
    every transport that reached these two functions an arbitrary-code
    door for whoever supplied the bytes.  For on-disk persistence (e.g.
    checkpoint export at session boundaries), use the v0.2 #8 checkpoint
    manifest path instead.

    Attributes
    ----------
    payload : bytes
        Serialised graph state.
    schema_token : str
        The model's instantiation token at the time of serialisation —
        :func:`deserialize_fmu_state` rejects a mismatched token to
        catch the "wrong FMU loaded the wrong snapshot" failure mode.
    """
    payload: bytes
    schema_token: str


@stability(StabilityLevel.EVOLVING)
def serialize_fmu_state(
    *,
    state: dict[str, dict[str, Any]],
    schema_token: str,
    params: Optional[dict] = None,
) -> FMUState:
    """Snapshot a graph state to an opaque :class:`FMUState` handle.

    Parameters
    ----------
    state : dict
        ``{node_name: {field_name: array, ...}, ...}`` — the same
        shape :meth:`GraphManager.step` returns.
    schema_token : str
        The model's instantiation token (from
        :class:`ModelDescription.instantiation_token`).  Recorded so
        deserialisation can catch the wrong-FMU case.
    params : dict, optional
        The graph parameter pytree in force (``GraphManager.params``
        layout).  FMI ``parameter`` variables are part of the FMU state,
        so a snapshot taken after an importer tuned one restores it.

    Raises
    ------
    TypeError
        If the tree has a non-string key or a leaf numpy cannot store as
        a plain array.  Both used to be accepted, by pickling them.

    Notes
    -----
    Implementation detail: the payload is an ``npz`` of plain arrays plus
    the key paths that rebuild the tree.  This is reasonable for in-RAM
    round-trips between ``fmi3GetFMUState`` and ``fmi3SetFMUState`` (the
    typical FMI use case) — it's *not* meant for cross-version
    persistence, which is what the v0.2 #8 manifest path handles with
    proper integrity hashing.
    """
    # Coerce JAX arrays to numpy for portability.
    coerced = {
        node: {field: np.asarray(val) for field, val in fields.items()}
        for node, fields in state.items()
    }
    if params is None:
        payload_obj: Any = coerced
        kind = "state"
    else:
        payload_obj = {
            _STATE_KEY: coerced,
            _PARAMS_KEY: _coerce_tree(params),
        }
        kind = "state+params"
    items: list[tuple[list[str], np.ndarray]] = []
    empty: list[list[str]] = []
    _flatten_tree(payload_obj, [], items, empty)
    members: dict[str, np.ndarray] = {f"a{i}": arr for i, (_, arr) in enumerate(items)}
    members[_KIND_MEMBER] = np.array(kind)
    members[_PATHS_MEMBER] = np.array(json.dumps([path for path, _ in items]))
    members[_EMPTY_MEMBER] = np.array(json.dumps(empty))
    buf = io.BytesIO()
    np.savez(buf, **members)
    return FMUState(payload=buf.getvalue(), schema_token=schema_token)


_STATE_KEY = "__maddening_state__"
_PARAMS_KEY = "__maddening_params__"
_KIND_MEMBER = "_kind"
_PATHS_MEMBER = "_paths"
_EMPTY_MEMBER = "_empty"
_ZIP_MAGIC = b"PK\x03\x04"


def _coerce_tree(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _coerce_tree(v) for k, v in tree.items()}
    return np.asarray(tree)


def _flatten_tree(tree: Any, prefix: list[str],
                  out: list[tuple[list[str], np.ndarray]],
                  empty: list[list[str]]) -> None:
    """Depth-first ``(key path, array)`` pairs, the form an ``npz`` can hold.

    A branch with no leaves under it has no arrays to record, so its path
    is collected separately -- without that, ``{"mappings": {}}`` would
    come back as a tree with no ``"mappings"`` key at all.
    """
    if isinstance(tree, dict):
        if not tree and prefix:
            empty.append(prefix)
        for key, value in tree.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"FMU state keys must be strings, got {key!r} "
                    f"({type(key).__name__}) at {'/'.join(prefix) or '<root>'}",
                )
            _flatten_tree(value, [*prefix, key], out, empty)
        return
    arr = np.asarray(tree)
    if arr.dtype == object:
        raise TypeError(
            f"FMU state leaf {'/'.join(prefix)!r} is not a plain array "
            f"({type(tree).__name__}); only numeric / boolean / string arrays "
            "can be snapshotted",
        )
    out.append((prefix, arr))


def _unflatten(paths: list[list[str]], empty: list[list[str]], data) -> Any:
    tree: dict[str, Any] = {}
    for i, path in enumerate(paths):
        node = tree
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = data[f"a{i}"]
    for path in empty:
        node = tree
        for key in path:
            node = node.setdefault(key, {})
    return tree


@stability(StabilityLevel.EVOLVING)
def deserialize_fmu_state(
    fmu_state: FMUState,
    *,
    expected_schema_token: str,
    return_params: bool = False,
) -> Any:
    """Restore a graph state from an :class:`FMUState` handle.

    Returns the state dict, or ``(state, params)`` when
    ``return_params`` is true (``params`` is ``None`` for a snapshot
    taken without a parameter pytree).

    Nothing in the payload is executed: it is an ``npz`` of plain arrays
    read with ``allow_pickle=False``.

    Raises
    ------
    ValueError
        If the snapshot's ``schema_token`` doesn't match
        ``expected_schema_token`` — protects against an FMU loading
        a snapshot from a structurally different model — or if the
        payload is not a readable archive.  A payload written by a
        MADDENING before 0.4.0 was a pickle and is refused by name
        rather than unpickled.
    """
    if fmu_state.schema_token != expected_schema_token:
        raise ValueError(
            f"FMUState schema mismatch: snapshot was made for "
            f"{fmu_state.schema_token!r}, but the current model is "
            f"{expected_schema_token!r}.  This snapshot is incompatible "
            "with the loaded FMU.",
        )
    payload = fmu_state.payload
    if not isinstance(payload, (bytes, bytearray)) or not bytes(payload[:4]) == _ZIP_MAGIC:
        raise ValueError(
            "FMUState payload is not a MADDENING snapshot archive.  A "
            "snapshot written before 0.4.0 was a pickle; it is refused "
            "rather than unpickled, because unpickling a payload executes "
            "whatever produced it.  Take a fresh snapshot with "
            "serialize_fmu_state().",
        )
    try:
        data = np.load(io.BytesIO(bytes(payload)), allow_pickle=False)
    except (zipfile.BadZipFile, ValueError, OSError) as exc:
        raise ValueError(f"FMUState payload is not a valid archive: {exc}") from exc
    with data:
        try:
            kind = str(data[_KIND_MEMBER])
            paths = json.loads(str(data[_PATHS_MEMBER]))
            empty = (json.loads(str(data[_EMPTY_MEMBER]))
                     if _EMPTY_MEMBER in data.files else [])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"FMUState payload is missing its directory: {exc}",
            ) from exc
        tree = _unflatten(paths, empty, data)
    if kind == "state+params":
        state, params = tree.get(_STATE_KEY, {}), tree.get(_PARAMS_KEY)
    else:
        state, params = tree, None
    return (state, params) if return_params else state


__all__ = [
    "FMUState",
    "deserialize_fmu_state",
    "serialize_fmu_state",
]
