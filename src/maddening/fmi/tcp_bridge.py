"""TCP bridge between the FMU C wrapper and :class:`FmuSidecar`.

The compiled FMU (``src/maddening/fmi/c/maddening_fmu.c``) is loaded
into the importer's process and speaks a tiny protocol that needs only
libc on the C side: every message is a 4-byte big-endian length prefix
followed by one frame.  This module is the Python end of that protocol:
it maps FMI value references onto the sidecar's inputs, outputs and
parameters using the :class:`ModelDescription` the FMU was built from,
and runs the master-step loop.

Requests (importer -> sidecar) and responses, JSON form::

    {"op": "hello"}                       -> {"ok": true, "token": ..., "model": ...,
                                              "master_dt": h, "protocol": 2, "binary": b}
    {"op": "set", "vr": [..], "values": [..]}
                                          -> {"ok": true}
    {"op": "get", "vr": [..]}             -> {"ok": true, "values": [..]}
    {"op": "step", "t": t, "dt": h}       -> {"ok": true, "t": t + h}
    {"op": "get_state"}                   -> {"ok": true, "state": "<base64>"}
    {"op": "set_state", "state": ".."}    -> {"ok": true}
    {"op": "reset"}                       -> {"ok": true}
    {"op": "terminate"}                   -> {"ok": true}
    any failure                           -> {"ok": false, "error": "..."}

``values`` are flat numbers in value-reference order; an array variable
contributes ``prod(shape)`` entries in row-major order.  Inputs are held
until the next ``step``; a communication step ``h`` must be a whole
multiple of ``master_dt`` (it runs ``h / master_dt`` graph steps; anything
else is refused, and the FMU advertises a fixed communication step).  A
``set`` is atomic: parameters are bounds-checked by the sidecar and inputs
are committed only when every value in the request was valid.

**Binary frames (protocol 2).**  Bit 31 of the length prefix marks a
*binary* frame; the low 31 bits are the payload length.  A binary
payload is ``[u32 BE header_len][header JSON][raw bytes]``: the header
carries ``op`` and metadata, the raw part carries the data, so bulk
values and state blobs never pass through JSON text::

    set:        {"op":"set","vr":[..],"n":N,"dtype":"f64"}   raw = N little-endian float64
    get reply:  {"ok":true,"n":N,"dtype":"f64"}              raw = N little-endian float64
    set_state:  {"op":"set_state","n":L}                     raw = L bytes of npz
    get_state reply: {"ok":true,"n":L}                       raw = L bytes of npz

A client opts in with ``{"op": "hello", "protocol": 2, "binary": true}``;
only then does the bridge answer ``get`` / ``get_state`` with binary
frames and accept binary ``set`` / ``set_state`` requests.  Every other
op, every error reply and every JSON-only client is unchanged: a client
that sends ``{"op": "hello"}`` gets exactly the protocol-1 behaviour
(``recv_message`` / ``send_message`` here speak both forms).  A client
announcing a protocol this bridge does not know is refused at hello.

The importer is **untrusted**: nothing that arrives on the socket is ever
unpickled or evaluated.  The FMU-state blob is an ``npz`` archive of plain
arrays (``allow_pickle=False`` on load) carrying the schema token, the
time, the pending inputs, the node states, ``_meta`` and the params; on
``set_state`` every array is checked against the live one (token, key set,
shape) before anything is written.  Bind the bridge to ``127.0.0.1``
unless the network is trusted.

ZMQ is not required.  A ZMQ transport with the same frame payloads can be
added later without touching the C wrapper's request format.
"""

from __future__ import annotations

import base64
import io
import json
import socket
import struct
import threading
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.fmi.model_description import FMIVariable, ModelDescription
from maddening.fmi.sidecar import FmuSidecar

_HEADER = struct.Struct(">I")
_MAX_MESSAGE = 64 * 1024 * 1024
_BINARY_FLAG = 0x80000000
_LENGTH_MASK = 0x7FFFFFFF
PROTOCOL_VERSION = 2
"""Highest sidecar protocol this bridge speaks (1 = JSON only, 2 = + binary frames)."""


def _recv_exact(conn: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_raw(conn: socket.socket) -> Optional[tuple[bool, bytes]]:
    """One length-prefixed frame as ``(is_binary, payload)``.

    ``None`` at EOF; ``ValueError`` when the (31-bit) length exceeds the
    64 MiB limit, binary flag or not.
    """
    head = _recv_exact(conn, _HEADER.size)
    if head is None:
        return None
    (word,) = _HEADER.unpack(head)
    n = word & _LENGTH_MASK
    if n > _MAX_MESSAGE:
        raise ValueError(f"message of {n} bytes exceeds the {_MAX_MESSAGE}-byte limit")
    body = _recv_exact(conn, n)
    if body is None:
        return None
    return bool(word & _BINARY_FLAG), body


def recv_frame(conn: socket.socket) -> Optional[bytes]:
    """One length-prefixed frame's payload (``None`` at EOF; ``ValueError``
    over the limit).  Use :func:`recv_raw` to learn whether it was binary."""
    got = recv_raw(conn)
    return None if got is None else got[1]


def encode_binary(header: dict, raw: bytes) -> bytes:
    """Payload of a binary frame: ``[u32 BE header_len][header JSON][raw]``."""
    hdr = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _HEADER.pack(len(hdr)) + hdr + raw


def decode_binary(payload: bytes) -> tuple[dict, bytes]:
    """Split a binary payload into ``(header, raw)``; ``ValueError`` if malformed."""
    if len(payload) < _HEADER.size:
        raise ValueError("binary frame shorter than its header length field")
    (hlen,) = _HEADER.unpack_from(payload)
    if hlen > len(payload) - _HEADER.size:
        raise ValueError(f"binary header of {hlen} bytes exceeds the {len(payload)}-byte payload")
    try:
        header = json.loads(payload[_HEADER.size:_HEADER.size + hlen].decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"binary header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("binary header must be a JSON object")
    return header, payload[_HEADER.size + hlen:]


def recv_message(conn: socket.socket) -> Optional[dict]:
    """One decoded message.  A JSON frame is its object; a binary frame is
    its header with the raw payload under ``"raw"`` (``bytes``)."""
    got = recv_raw(conn)
    if got is None:
        return None
    is_binary, body = got
    if is_binary:
        header, raw = decode_binary(body)
        header["raw"] = raw
        return header
    return json.loads(body.decode("utf-8"))


def send_message(conn: socket.socket, payload: dict) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    conn.sendall(_HEADER.pack(len(body)) + body)


def send_binary(conn: socket.socket, header: dict, raw: bytes) -> None:
    """Send one binary-flagged frame (``header`` JSON + ``raw`` bytes)."""
    body = encode_binary(header, raw)
    if len(body) > _LENGTH_MASK:
        raise ValueError("binary frame exceeds the 31-bit length field")
    conn.sendall(_HEADER.pack(_BINARY_FLAG | len(body)) + body)


def values_of(reply: dict) -> np.ndarray:
    """The ``values`` of a ``get`` reply as float64, whichever form it took."""
    if "raw" in reply:
        return np.frombuffer(reply["raw"], dtype="<f8").astype(np.float64)
    return np.asarray(reply["values"], dtype=np.float64)


def state_of(reply: dict) -> bytes:
    """The npz bytes of a ``get_state`` reply, whichever form it took."""
    if "raw" in reply:
        return reply["raw"]
    return base64.b64decode(reply["state"])


def _copy_tree(tree):
    """Deep copy of a nested dict of arrays without pickle."""
    if tree is None:
        return None
    if isinstance(tree, dict):
        return {k: _copy_tree(v) for k, v in tree.items()}
    return tree if hasattr(tree, "dtype") else np.asarray(tree)


def _size(var: FMIVariable) -> int:
    return int(np.prod(var.shape)) if var.shape else 1


@stability(StabilityLevel.EVOLVING)
class FmuTcpBridge:
    """Serve one :class:`FmuSidecar` to the FMU C wrapper over TCP.

    Parameters
    ----------
    sidecar : FmuSidecar
    model_description : ModelDescription
        The description the FMU was built from; value references are
        resolved against its variables.
    master_dt : float
        The graph's base timestep (one sidecar ``step``).
    host, port : str, int
        Bind address; ``port=0`` picks a free port (see :attr:`endpoint`).
    """

    def __init__(
        self,
        sidecar: FmuSidecar,
        model_description: ModelDescription,
        *,
        master_dt: float,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._sidecar = sidecar
        self._md = model_description
        self._dt = float(master_dt)
        self._vars: dict[int, FMIVariable] = {
            v.value_reference: v for v in model_description.variables
        }
        # Every declared input starts at its advertised start value (0),
        # exactly as gm.step() fills an unset external input, so a node
        # whose update() has a non-zero fallback for a *missing* input
        # (HeatNode's T_left = T[0]) behaves like the graph.
        self._inputs: dict[str, dict[str, Any]] = self._zero_inputs()
        self._time = 0.0
        self._initial_state = _copy_tree(sidecar.state)
        self._initial_params = _copy_tree(sidecar.params)
        self._busy = threading.Lock()                     # one instance per bridge
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(4)
        self._host, self._port = self._server.getsockname()[:2]
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.requests_served = 0
        self.binary_frames_served = 0     # binary replies sent (get / get_state)
        self.binary_frames_received = 0   # binary requests accepted (set / set_state)

    # ----------------------------------------------------------------- server
    @property
    def endpoint(self) -> str:
        return f"{self._host}:{self._port}"

    def start(self) -> "FmuTcpBridge":
        self._thread = threading.Thread(target=self._serve, name="maddening-fmu-bridge",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __enter__(self) -> "FmuTcpBridge":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _serve(self) -> None:
        self._server.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()

    def _serve_conn(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(None)
            if not self._busy.acquire(blocking=False):
                # A bridge holds ONE sidecar state; a second instance
                # would silently share it.  Refuse instead of blocking.
                try:
                    req = recv_message(conn)
                    if req is not None:
                        send_message(conn, {"ok": False, "error":
                                            "bridge already serves an FMU instance; "
                                            "start one FmuTcpBridge per instance"})
                except (OSError, ValueError):
                    pass
                return
            binary = False                # negotiated at hello, per connection
            try:
                while not self._stop.is_set():
                    try:
                        got = recv_raw(conn)
                    except (OSError, ValueError):
                        break                              # socket / framing error
                    if got is None:
                        break
                    is_binary, body = got
                    try:
                        if is_binary:
                            if not binary:
                                raise ValueError("binary frames need a hello with "
                                                 "\"protocol\": 2, \"binary\": true first")
                            req = self._decode_binary_request(body)
                            self.binary_frames_received += 1
                        else:
                            req = json.loads(body.decode("utf-8"))
                            if not isinstance(req, dict):
                                raise ValueError("request must be a JSON object")
                    except (ValueError, UnicodeDecodeError) as exc:
                        # a corrupt request (an FMU-state blob with stray
                        # quotes, say) is an error reply, not a dead instance
                        send_message(conn, {"ok": False, "error": f"malformed request: {exc}"})
                        continue
                    reply = self._dispatch(req)
                    if req.get("op") == "hello" and reply.get("ok"):
                        binary = bool(reply.get("binary"))
                    if binary and reply.get("ok") and ("values" in reply or "state" in reply):
                        if "values" in reply:
                            arr = np.ascontiguousarray(reply["values"], dtype="<f8")
                            header, raw = {"ok": True, "n": int(arr.size), "dtype": "f64"}, arr.tobytes()
                        else:
                            header, raw = {"ok": True, "n": len(reply["state"])}, reply["state"]
                        try:
                            send_binary(conn, header, raw)
                        except ValueError as exc:            # beyond the 31-bit length field
                            send_message(conn, {"ok": False, "error": f"reply too large: {exc}"})
                            continue
                        self.binary_frames_served += 1
                    else:
                        send_message(conn, self._jsonify(reply))
            finally:
                self._busy.release()

    @staticmethod
    def _decode_binary_request(body: bytes) -> dict:
        """A binary ``set`` / ``set_state`` request as the plain request
        dict :meth:`handle` takes (``values`` as an array, ``state`` as
        bytes).  ``ValueError`` on any inconsistency; nothing is trusted."""
        header, raw = decode_binary(body)
        op = header.get("op")
        n = header.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError("binary header needs a non-negative integer \"n\"")
        if op == "set":
            if header.get("dtype") != "f64":
                raise ValueError(f"unsupported binary dtype {header.get('dtype')!r} (only f64)")
            if len(raw) != 8 * n:
                raise ValueError(f"binary set announces {n} float64 but carries {len(raw)} bytes")
            values = np.frombuffer(raw, dtype="<f8").astype(np.float64)
            return {"op": "set", "vr": header.get("vr"), "values": values}
        if op == "set_state":
            if len(raw) != n:
                raise ValueError(f"binary set_state announces {n} bytes but carries {len(raw)}")
            return {"op": "set_state", "state": raw}
        raise ValueError(f"op {op!r} has no binary request form")

    @staticmethod
    def _jsonify(reply: dict) -> dict:
        """The JSON (protocol-1) form of a reply: values as a list, state as base64."""
        if reply.get("ok"):
            if "values" in reply:
                reply = {**reply, "values": np.asarray(reply["values"], dtype=np.float64).tolist()}
            if "state" in reply:
                reply = {**reply, "state": base64.b64encode(reply["state"]).decode("ascii")}
        return reply

    # ---------------------------------------------------------------- handler
    def handle(self, req: dict) -> dict:
        """Serve one decoded request (also usable without a socket) in its
        JSON form: ``values`` come back as a list, ``state`` as base64."""
        return self._jsonify(self._dispatch(req))

    def _dispatch(self, req: dict) -> dict:
        """Serve one request; ``values`` is a float64 array and ``state``
        raw npz bytes, encoded by the caller for the wire in use."""
        self.requests_served += 1
        try:
            op = req.get("op")
            if op == "hello":
                proto = req.get("protocol", 1)
                if not isinstance(proto, int) or isinstance(proto, bool) or proto < 1:
                    raise ValueError(f"protocol must be a positive integer, got {proto!r}")
                if proto > PROTOCOL_VERSION:
                    raise ValueError(f"protocol {proto} is not supported by this bridge "
                                     f"(highest is {PROTOCOL_VERSION})")
                binary = proto >= 2 and req.get("binary") is True
                return {"ok": True, "token": self._md.instantiation_token,
                        "model": self._md.model_name, "master_dt": self._dt,
                        "protocol": PROTOCOL_VERSION, "binary": binary}
            if op == "set":
                self._set(req["vr"], req["values"])
                return {"ok": True}
            if op == "get":
                return {"ok": True, "values": self._get(req["vr"])}
            if op == "step":
                h = float(req["dt"])
                if not np.isfinite(h) or h <= 0:
                    raise ValueError(f"communication step must be positive, got {h!r}")
                ratio = h / self._dt
                n = int(round(ratio))
                if n < 1 or abs(ratio - n) > 1e-6 * max(1.0, n):
                    # The FMU advertises a fixed communication step: the
                    # physics can only advance whole master steps, and
                    # reporting t + h for a different advance would desync
                    # the importer's time from the state.
                    raise ValueError(
                        f"communication step {h!r} is not a whole multiple of the "
                        f"master timestep {self._dt!r}"
                    )
                saved = self._sidecar._state                        # noqa: SLF001
                try:
                    for _ in range(n):
                        self._sidecar.step(self._inputs)
                except Exception:
                    # a failed sub-step must not leave a partial advance
                    # behind: the importer is told nothing happened
                    self._sidecar._state = saved                    # noqa: SLF001
                    raise
                self._time = float(req.get("t", self._time)) + n * self._dt
                return {"ok": True, "t": self._time}
            if op == "get_state":
                return {"ok": True, "state": self._encode_state()}
            if op == "set_state":
                blob = req["state"]
                if not isinstance(blob, (bytes, bytearray)):
                    blob = base64.b64decode(blob)
                self._decode_state(bytes(blob))
                return {"ok": True}
            if op == "reset":
                self._sidecar._state = _copy_tree(self._initial_state)      # noqa: SLF001
                if self._initial_params is not None:
                    self._sidecar._params = _copy_tree(self._initial_params)  # noqa: SLF001
                self._inputs, self._time = self._zero_inputs(), 0.0
                return {"ok": True}
            if op == "terminate":
                return {"ok": True}
            return {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as exc:  # noqa: BLE001 - reported to the importer
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _zero_inputs(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for var in self._md.variables:
            if var.causality == "input" and not var.is_clock:
                node, _, field = var.name.partition(".")
                out.setdefault(node, {})[field] = jnp.zeros(var.shape or (), dtype=var.dtype)
        return out

    # -------------------------------------------------------- FMU state blob
    _META = "_meta"

    def _encode_state(self) -> bytes:
        """``npz`` of plain arrays: token, time, node states, ``_meta``,
        params (nodes + mappings) and the pending inputs.  No pickle."""
        arrays: dict[str, np.ndarray] = {
            "_token": np.array(self._md.instantiation_token),
            "_time": np.array(self._time, dtype=np.float64),
        }
        for node, fields in self._sidecar.state.items():
            for f, v in fields.items():
                arrays[f"s/{node}/{f}"] = np.asarray(v)
        params = self._sidecar.params or {}
        for section in ("nodes", "mappings"):
            for owner, leaves in params.get(section, {}).items():
                for k, v in leaves.items():
                    arrays[f"p/{section}/{owner}/{k}"] = np.asarray(v)
        for node, fields in self._inputs.items():
            for f, v in fields.items():
                arrays[f"i/{node}/{f}"] = np.asarray(v)
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        return buf.getvalue()

    def _decode_state(self, blob: bytes) -> None:
        """Validate against the live state before writing anything."""
        try:
            data = np.load(io.BytesIO(blob), allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"FMU state blob is not a valid archive: {exc}") from exc
        with data:
            keys = set(data.files)
            # Never decompress more than the model can hold: a compressed
            # member must not exceed the live leaf it claims to replace.
            zf = getattr(data, "zip", None)
            live_bytes = {
                f"s/{n}/{f}": int(np.asarray(v).nbytes)
                for n, fields in self._sidecar.state.items() for f, v in fields.items()
            }
            for section in ("nodes", "mappings"):
                for owner, leaves in (self._sidecar.params or {}).get(section, {}).items():
                    for k, v in leaves.items():
                        live_bytes[f"p/{section}/{owner}/{k}"] = int(np.asarray(v).nbytes)
            for var in self._md.variables:
                if var.causality == "input":
                    node, _, field = var.name.partition(".")
                    live_bytes[f"i/{node}/{field}"] = int(np.zeros(var.shape or (), var.dtype).nbytes)
            if zf is not None:
                for k in keys:
                    try:
                        size = zf.getinfo(k + ".npy").file_size
                    except KeyError:
                        continue
                    cap = live_bytes.get(k, 256) + 4096
                    if size > cap:
                        raise ValueError(f"FMU state member {k!r} is {size} bytes, more than the "
                                         f"{cap} the model can hold")
            if "_token" not in keys or str(data["_token"]) != self._md.instantiation_token:
                raise ValueError("FMU state belongs to a different model (schema token mismatch)")
            state = {n: dict(f) for n, f in self._sidecar.state.items()}
            expected = {f"s/{n}/{f}" for n, fields in state.items() for f in fields}
            got = {k for k in keys if k.startswith("s/")}
            if got != expected:
                raise ValueError(f"FMU state fields differ from the model: "
                                 f"missing {sorted(expected - got)}, extra {sorted(got - expected)}")
            new_state: dict[str, dict[str, Any]] = {}
            for k in expected:
                _, node, field = k.split("/", 2)
                live = np.asarray(state[node][field])
                arr = data[k]
                if arr.shape != live.shape:
                    raise ValueError(f"FMU state {node}.{field}: shape {arr.shape} != {live.shape}")
                new_state.setdefault(node, {})[field] = jnp.asarray(arr, dtype=live.dtype)
            params = self._sidecar.params
            new_params = None
            if params is not None:
                new_params = {"nodes": {}, "mappings": {}}
                for section in ("nodes", "mappings"):
                    for owner, leaves in params.get(section, {}).items():
                        new_params[section][owner] = {}
                        for k, v in leaves.items():
                            key = f"p/{section}/{owner}/{k}"
                            if key not in keys:
                                raise ValueError(f"FMU state lacks parameter {key}")
                            live = np.asarray(v)
                            arr = data[key]
                            if arr.shape != live.shape:
                                raise ValueError(f"FMU state param {key}: shape {arr.shape} != {live.shape}")
                            new_params[section][owner][k] = jnp.asarray(arr, dtype=live.dtype)
            inputs: dict[str, dict[str, Any]] = self._zero_inputs()
            for k in keys:
                if k.startswith("i/"):
                    _, node, field = k.split("/", 2)
                    var = next((v for v in self._md.variables if v.name == f"{node}.{field}"
                                and v.causality == "input"), None)
                    if var is None:
                        raise ValueError(f"FMU state carries unknown input {node}.{field}")
                    arr = data[k]
                    if tuple(arr.shape) != tuple(var.shape or ()):
                        raise ValueError(f"FMU state input {node}.{field}: bad shape {arr.shape}")
                    inputs.setdefault(node, {})[field] = jnp.asarray(arr, dtype=var.dtype)
            t = float(data["_time"]) if "_time" in keys else 0.0
            if not np.isfinite(t):
                raise ValueError("FMU state carries a non-finite time")
        # every check passed: commit
        self._sidecar._state = new_state                      # noqa: SLF001
        if new_params is not None:
            self._sidecar._params = new_params                # noqa: SLF001
        self._inputs, self._time = inputs, t

    # ----------------------------------------------------------- vr mapping
    def _set(self, vrs: list[int], values) -> None:
        if not isinstance(vrs, (list, tuple)):
            raise ValueError("vr must be a list of value references")
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("values must be a flat list of numbers")
        pos = 0
        staged: list[tuple[FMIVariable, np.ndarray]] = []
        for vr in vrs:
            var = self._vars.get(int(vr))
            if var is None:
                raise KeyError(f"unknown value reference {vr}")
            n = _size(var)
            chunk = values[pos:pos + n]
            if chunk.size != n:
                raise ValueError(f"vr {vr} ({var.name}) expects {n} values, got {chunk.size}")
            pos += n
            staged.append((var, chunk.reshape(var.shape or ())))
        if pos != len(values):
            raise ValueError(f"{len(values) - pos} trailing values without a value reference")
        param_updates: dict[str, Any] = {}
        input_updates: list[tuple[str, str, np.ndarray]] = []
        for var, arr in staged:
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"variable {var.name!r}: value must be finite")
            if var.causality == "parameter":
                param_updates[var.name] = arr.astype(var.dtype)
            elif var.causality == "input":
                node, _, field = var.name.partition(".")
                input_updates.append((node, field, arr.astype(var.dtype)))
            elif var.is_clock:
                continue                              # clock ticks are informational
            else:
                raise ValueError(f"variable {var.name!r} ({var.causality}) is read-only")
        # Atomic: parameters are validated (bounds) by the sidecar first;
        # inputs are only committed once nothing can fail any more.
        if param_updates:
            self._sidecar.set_params(param_updates)
        for node, field, arr in input_updates:
            self._inputs.setdefault(node, {})[field] = arr

    def _get(self, vrs: list[int]) -> np.ndarray:
        if not isinstance(vrs, (list, tuple)):
            raise ValueError("vr must be a list of value references")
        parts: list[np.ndarray] = []
        params = self._sidecar.get_params()
        for vr in vrs:
            var = self._vars.get(int(vr))
            if var is None:
                raise KeyError(f"unknown value reference {vr}")
            if var.causality == "independent":
                parts.append(np.asarray([self._time], dtype=np.float64))
            elif var.causality == "parameter":
                parts.append(np.asarray(params[var.name], dtype=np.float64).ravel())
            elif var.causality == "input":
                node, _, field = var.name.partition(".")
                val = self._inputs.get(node, {}).get(field)
                if val is None:
                    val = np.zeros(var.shape or (), dtype=np.float64)
                parts.append(np.asarray(val, dtype=np.float64).ravel())
            elif var.is_clock:
                parts.append(np.zeros(1, dtype=np.float64))
            else:
                node, _, field = var.name.partition(".")
                parts.append(np.asarray(self._sidecar.state[node][field],
                                        dtype=np.float64).ravel())
        if not parts:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(parts)


__all__ = ["FmuTcpBridge", "PROTOCOL_VERSION", "decode_binary", "encode_binary",
           "recv_frame", "recv_message", "recv_raw", "send_binary", "send_message",
           "state_of", "values_of"]
