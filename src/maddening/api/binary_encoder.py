"""
BinaryStateEncoder -- efficient binary serialization of simulation state.

Packs state dicts into flat float32 buffers for high-throughput WebSocket
streaming.  Sends a JSON schema once at connection time, then binary
frames containing only raw floats.

Frame format (uncompressed)::

    [8 bytes: sim_time as float64]
    [N * 4 bytes: flat float32 values in schema order]

Optional compression (v0.2 #6) wraps the float-payload portion with
zstd or zstd+xor.  The 8-byte ``sim_time`` header is always plaintext
so clients can demux/seek without decompressing; everything after byte 8
is compressed.

For a typical 4-node graph with a 20-cell heat rod, each frame is
~108 bytes vs ~800 bytes for JSON (7x reduction).  Compression buys an
additional 5-15× on slowly-varying fields and ~2× on noisy ones.
"""

import struct
from typing import Any

import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

try:
    import zstandard as _zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False


VALID_COMPRESSIONS = {"none", "zstd", "zstd+xor"}


@stability(StabilityLevel.EVOLVING)
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
    compression : str, optional
        One of ``"none"`` (default), ``"zstd"``, or ``"zstd+xor"``.

        - ``"none"``: raw float32 payload (matches v0.1 behaviour).
        - ``"zstd"``: zstandard-compress the payload.  Best ratio on
          smooth fields (heat rods, slowly-mixing flows).
        - ``"zstd+xor"``: XOR the new frame against the previous one
          before zstd-compressing.  Best ratio on noisy fields where
          consecutive frames differ in just a few bits per float.
    zstd_level : int, optional
        zstd compression level 1-22 (default 3).  Higher = smaller +
        slower.  Level 3 is the latency-friendly sweet spot.

    Notes
    -----
    Compression requires the ``zstandard`` package (``pip install
    maddening[compression]``).  Asking for a compressed encoder
    without it installed raises ``ImportError`` at construction time
    so the failure is loud rather than silent.
    """

    def __init__(
        self,
        state: dict[str, dict[str, Any]],
        fields: dict[str, list[str]] | None = None,
        *,
        compression: str = "none",
        zstd_level: int = 3,
    ):
        if compression not in VALID_COMPRESSIONS:
            raise ValueError(
                f"compression must be one of {VALID_COMPRESSIONS!r}, "
                f"got {compression!r}"
            )
        if compression != "none" and not _HAS_ZSTD:
            raise ImportError(
                f"compression={compression!r} requires the 'zstandard' "
                "package. Install with: pip install maddening[compression]"
            )

        self._compression = compression
        self._zstd_level = zstd_level
        self._zstd_cctx = (
            _zstd.ZstdCompressor(level=zstd_level) if _HAS_ZSTD and compression != "none"
            else None
        )
        self._prev_payload: bytes | None = None  # for xor-delta

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
            # ``frame_bytes`` is the *uncompressed* size; compressed
            # frames are variable-length so the client should size its
            # decompress buffer from the schema and decompress lazily.
            "frame_bytes": self._buf_size,
            "compression": self._compression,
        }

    def encode(self, sim_time: float, state: dict) -> bytes:
        """Pack ``(sim_time, state)`` into a binary frame.

        The 8-byte ``sim_time`` header is always plaintext so receivers
        can demux without decompressing.  When ``compression`` is
        enabled, the float payload (everything after byte 8) is
        compressed; the on-wire frame is therefore
        ``[8 B sim_time][variable-length compressed payload]`` and the
        receiver must look at the schema's ``compression`` field to
        decide whether to decompress.
        """
        # Build the uncompressed payload (everything after byte 8).
        payload = bytearray(self._total_floats * 4)
        offset = 0
        for node, field, _shape, n in self._fields:
            val = state[node][field]
            if hasattr(val, "flatten"):
                flat = np.asarray(val.flatten(), dtype=np.float32)
            else:
                flat = np.array([float(val)], dtype=np.float32)
            payload[offset : offset + n * 4] = flat.tobytes()
            offset += n * 4
        payload_b = bytes(payload)

        if self._compression == "none":
            wire_payload = payload_b
        elif self._compression == "zstd":
            wire_payload = self._zstd_cctx.compress(payload_b)
        elif self._compression == "zstd+xor":
            # XOR against the previous payload before compressing.
            # On steady-state fields the XOR is mostly zeros which zstd
            # crushes to a few bytes.  First frame has no prev → emit raw.
            if self._prev_payload is None or len(self._prev_payload) != len(payload_b):
                xored = payload_b
            else:
                a = np.frombuffer(payload_b, dtype=np.uint8)
                b = np.frombuffer(self._prev_payload, dtype=np.uint8)
                xored = (a ^ b).tobytes()
            wire_payload = self._zstd_cctx.compress(xored)
            self._prev_payload = payload_b
        else:  # pragma: no cover — guarded by __init__
            raise RuntimeError(f"unknown compression {self._compression!r}")

        out = bytearray(8 + len(wire_payload))
        struct.pack_into("d", out, 0, sim_time)
        out[8:] = wire_payload
        return bytes(out)

    @property
    def total_floats(self) -> int:
        return self._total_floats

    @property
    def frame_bytes(self) -> int:
        """Uncompressed frame size in bytes (8 + 4·total_floats).

        For compressed encoders, the actual on-wire size is variable
        per frame; see :meth:`encode` return value.
        """
        return self._buf_size

    @property
    def subscription(self) -> dict[str, list[str]] | None:
        """Return the active field subscription, or ``None`` for all."""
        return self._subscription

    @property
    def compression(self) -> str:
        """The active compression mode (``"none"``, ``"zstd"``, ``"zstd+xor"``)."""
        return self._compression


def decode_frame(frame: bytes, schema: dict) -> tuple[float, np.ndarray]:
    """Inverse of :meth:`BinaryStateEncoder.encode`.

    Returns ``(sim_time, values)`` where ``values`` is a flat
    ``float32`` array of length ``schema["total_floats"]``.

    For ``compression="zstd+xor"`` the caller must track the previous
    decoded payload externally — that path is for clients that want to
    drive the XOR-delta cheaply on their side too; for stateless
    consumers, use ``compression="zstd"``.
    """
    sim_time = struct.unpack_from("d", frame, 0)[0]
    body = frame[8:]
    comp = schema.get("compression", "none")
    if comp == "none":
        raw = body
    elif comp == "zstd":
        if not _HAS_ZSTD:
            raise ImportError(
                "zstandard not installed; cannot decode zstd-compressed frame"
            )
        raw = _zstd.ZstdDecompressor().decompress(body)
    elif comp == "zstd+xor":
        raise NotImplementedError(
            "zstd+xor decoding is stateful; the caller must maintain the "
            "previous payload and XOR after decompressing"
        )
    else:
        raise ValueError(f"unknown compression {comp!r}")
    values = np.frombuffer(raw, dtype=np.float32)
    return sim_time, values
