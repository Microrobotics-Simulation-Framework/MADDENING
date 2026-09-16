"""TCP/JSON bridge between the FMU C wrapper and :class:`FmuSidecar`.

The compiled FMU (``src/maddening/fmi/c/maddening_fmu.c``) is loaded
into the importer's process and speaks a tiny protocol that needs only
libc on the C side: every message is a 4-byte big-endian length prefix
followed by one UTF-8 JSON object.  This module is the Python end of
that protocol: it maps FMI value references onto the sidecar's inputs,
outputs and parameters using the :class:`ModelDescription` the FMU was
built from, and runs the master-step loop.

Requests (importer -> sidecar) and responses::

    {"op": "hello"}                       -> {"ok": true, "token": ..., "model": ...}
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

The importer is **untrusted**: nothing that arrives on the socket is ever
unpickled or evaluated.  The FMU-state blob is an ``npz`` archive of plain
arrays (``allow_pickle=False`` on load) carrying the schema token, the
time, the pending inputs, the node states, ``_meta`` and the params; on
``set_state`` every array is checked against the live one (token, key set,
shape) before anything is written.  Bind the bridge to ``127.0.0.1``
unless the network is trusted.

ZMQ is not required.  A ZMQ transport with the same JSON payloads can be
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


def _recv_exact(conn: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(conn: socket.socket) -> Optional[bytes]:
    """One length-prefixed frame (``None`` at EOF; ``ValueError`` over the limit)."""
    head = _recv_exact(conn, _HEADER.size)
    if head is None:
        return None
    (n,) = _HEADER.unpack(head)
    if n > _MAX_MESSAGE:
        raise ValueError(f"message of {n} bytes exceeds the {_MAX_MESSAGE}-byte limit")
    return _recv_exact(conn, n)


def recv_message(conn: socket.socket) -> Optional[dict]:
    body = recv_frame(conn)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))


def send_message(conn: socket.socket, payload: dict) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    conn.sendall(_HEADER.pack(len(body)) + body)


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
            try:
                while not self._stop.is_set():
                    try:
                        body = recv_frame(conn)
                    except (OSError, ValueError):
                        break                              # socket / framing error
                    if body is None:
                        break
                    try:
                        req = json.loads(body.decode("utf-8"))
                        if not isinstance(req, dict):
                            raise ValueError("request must be a JSON object")
                    except (ValueError, UnicodeDecodeError) as exc:
                        # a corrupt request (an FMU-state blob with stray
                        # quotes, say) is an error reply, not a dead instance
                        send_message(conn, {"ok": False, "error": f"malformed request: {exc}"})
                        continue
                    send_message(conn, self.handle(req))
            finally:
                self._busy.release()

    # ---------------------------------------------------------------- handler
    def handle(self, req: dict) -> dict:
        """Serve one decoded request (also usable without a socket)."""
        self.requests_served += 1
        try:
            op = req.get("op")
            if op == "hello":
                return {"ok": True, "token": self._md.instantiation_token,
                        "model": self._md.model_name, "master_dt": self._dt}
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
                return {"ok": True, "state": base64.b64encode(self._encode_state()).decode("ascii")}
            if op == "set_state":
                self._decode_state(base64.b64decode(req["state"]))
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
    def _set(self, vrs: list[int], values: list[float]) -> None:
        pos = 0
        staged: list[tuple[FMIVariable, np.ndarray]] = []
        for vr in vrs:
            var = self._vars.get(int(vr))
            if var is None:
                raise KeyError(f"unknown value reference {vr}")
            n = _size(var)
            chunk = np.asarray(values[pos:pos + n], dtype=np.float64)
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

    def _get(self, vrs: list[int]) -> list[float]:
        out: list[float] = []
        params = self._sidecar.get_params()
        for vr in vrs:
            var = self._vars.get(int(vr))
            if var is None:
                raise KeyError(f"unknown value reference {vr}")
            if var.causality == "independent":
                out.append(self._time)
            elif var.causality == "parameter":
                out.extend(np.asarray(params[var.name], dtype=np.float64).ravel().tolist())
            elif var.causality == "input":
                node, _, field = var.name.partition(".")
                val = self._inputs.get(node, {}).get(field)
                if val is None:
                    val = np.zeros(var.shape or (), dtype=np.float64)
                out.extend(np.asarray(val, dtype=np.float64).ravel().tolist())
            elif var.is_clock:
                out.append(0.0)
            else:
                node, _, field = var.name.partition(".")
                out.extend(np.asarray(self._sidecar.state[node][field],
                                      dtype=np.float64).ravel().tolist())
        return out


__all__ = ["FmuTcpBridge", "recv_frame", "recv_message", "send_message"]
