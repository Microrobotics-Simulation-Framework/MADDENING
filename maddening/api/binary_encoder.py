"""
BinaryStateEncoder -- efficient binary serialization of simulation state.

Packs state dicts into flat float32 buffers for high-throughput WebSocket
streaming.  Sends a JSON schema once at connection time, then binary
frames containing only raw floats.

Frame format::

    [8 bytes: sim_time as float64]
    [N * 4 bytes: flat float32 values in schema order]

For a typical 4-node graph with a 20-cell heat rod, each frame is
~108 bytes vs ~800 bytes for JSON (7x reduction).
"""

import struct
from typing import Any

import numpy as np


class BinaryStateEncoder:
    """Builds a fixed schema from an example state dict, then packs
    subsequent states into flat binary buffers.

    Parameters
    ----------
    state : dict
        Example state ``{node_name: {field_name: array_or_scalar}}``.
    fields : dict, optional
        ``{node_name: [field1, field2, ...]}`` — include only these
        node/field combinations.  If ``None`` (default), all fields are
        included.  Nodes or fields not present in ``state`` are silently
        ignored.
    """

    def __init__(
        self,
        state: dict[str, dict[str, Any]],
        fields: dict[str, list[str]] | None = None,
    ):
        self._fields: list[tuple[str, str, tuple, int]] = []
        self._total_floats = 0
        self._subscription = fields  # keep for introspection

        for node in sorted(state.keys()):
            if fields is not None and node not in fields:
                continue
            allowed = None
            if fields is not None:
                allowed = set(fields[node])
            for field in sorted(state[node].keys()):
                if allowed is not None and field not in allowed:
                    continue
                val = state[node][field]
                if hasattr(val, "shape"):
                    shape = tuple(int(d) for d in val.shape)
                    n = max(1, int(np.prod(shape))) if shape else 1
                else:
                    shape = ()
                    n = 1
                self._fields.append((node, field, shape, n))
                self._total_floats += n

        self._buf_size = 8 + self._total_floats * 4

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schema(self) -> dict:
        """Return a JSON-serializable schema for the binary layout."""
        fields = []
        offset = 0
        for node, field, shape, n in self._fields:
            fields.append({
                "node": node,
                "field": field,
                "shape": list(shape),
                "offset": offset,
                "count": n,
            })
            offset += n
        return {
            "type": "schema",
            "fields": fields,
            "total_floats": self._total_floats,
            "frame_bytes": self._buf_size,
        }

    def encode(self, sim_time: float, state: dict) -> bytes:
        """Pack ``(sim_time, state)`` into a binary frame."""
        buf = bytearray(self._buf_size)
        struct.pack_into("d", buf, 0, sim_time)
        offset = 8
        for node, field, _shape, n in self._fields:
            val = state[node][field]
            if hasattr(val, "flatten"):
                flat = np.asarray(val.flatten(), dtype=np.float32)
            else:
                flat = np.array([float(val)], dtype=np.float32)
            buf[offset : offset + n * 4] = flat.tobytes()
            offset += n * 4
        return bytes(buf)

    @property
    def total_floats(self) -> int:
        return self._total_floats

    @property
    def frame_bytes(self) -> int:
        return self._buf_size

    @property
    def subscription(self) -> dict[str, list[str]] | None:
        """Return the active field subscription, or ``None`` for all."""
        return self._subscription
