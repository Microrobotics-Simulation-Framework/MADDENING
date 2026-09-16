"""``fmi3GetFMUState`` / ``fmi3SetFMUState`` round-trip via MADDENING state.

The v0.2 #8 checkpoint manifest already serialises the full graph
state with an integrity manifest; this module re-exposes that same
serialiser behind the FMI 3.0 names so the FMU sidecar can call
``GetFMUState`` / ``SetFMUState`` without rebuilding the round-trip.
"""

from __future__ import annotations

import io
import pickle
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

    The internal payload is a numpy-safe pickled bytes blob.  For
    on-disk persistence (e.g. checkpoint export at session
    boundaries), use the v0.2 #8 checkpoint manifest path instead.

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

    Notes
    -----
    Implementation detail: we pickle a dict of numpy arrays.  This
    is reasonable for in-RAM round-trips between
    ``fmi3GetFMUState`` and ``fmi3SetFMUState`` (the typical FMI
    use case) — it's *not* meant for cross-version persistence,
    which is what the v0.2 #8 manifest path handles with proper
    integrity hashing.
    """
    # Coerce JAX arrays to numpy for portability.
    coerced = {
        node: {field: np.asarray(val) for field, val in fields.items()}
        for node, fields in state.items()
    }
    if params is None:
        payload_obj: Any = coerced
    else:
        payload_obj = {
            _STATE_KEY: coerced,
            _PARAMS_KEY: _coerce_tree(params),
        }
    payload = pickle.dumps(payload_obj, protocol=pickle.HIGHEST_PROTOCOL)
    return FMUState(payload=payload, schema_token=schema_token)


_STATE_KEY = "__maddening_state__"
_PARAMS_KEY = "__maddening_params__"


def _coerce_tree(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _coerce_tree(v) for k, v in tree.items()}
    return np.asarray(tree)


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

    Raises
    ------
    ValueError
        If the snapshot's ``schema_token`` doesn't match
        ``expected_schema_token`` — protects against an FMU loading
        a snapshot from a structurally different model.
    """
    if fmu_state.schema_token != expected_schema_token:
        raise ValueError(
            f"FMUState schema mismatch: snapshot was made for "
            f"{fmu_state.schema_token!r}, but the current model is "
            f"{expected_schema_token!r}.  This snapshot is incompatible "
            "with the loaded FMU.",
        )
    obj = pickle.loads(fmu_state.payload)
    if isinstance(obj, dict) and _STATE_KEY in obj:
        state, params = obj[_STATE_KEY], obj.get(_PARAMS_KEY)
    else:
        state, params = obj, None
    return (state, params) if return_params else state


__all__ = [
    "FMUState",
    "deserialize_fmu_state",
    "serialize_fmu_state",
]
