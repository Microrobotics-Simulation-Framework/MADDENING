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
from typing import Any

import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability


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
    payload = pickle.dumps(coerced, protocol=pickle.HIGHEST_PROTOCOL)
    return FMUState(payload=payload, schema_token=schema_token)


@stability(StabilityLevel.EVOLVING)
def deserialize_fmu_state(
    fmu_state: FMUState,
    *,
    expected_schema_token: str,
) -> dict[str, dict[str, Any]]:
    """Restore a graph state from an :class:`FMUState` handle.

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
    return pickle.loads(fmu_state.payload)


__all__ = [
    "FMUState",
    "deserialize_fmu_state",
    "serialize_fmu_state",
]
