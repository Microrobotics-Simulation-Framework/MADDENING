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
until the next ``step``; a communication step of ``h`` runs
``round(h / master_dt)`` graph steps.

ZMQ is not required.  A ZMQ transport with the same JSON payloads can be
added later without touching the C wrapper's request format.
"""

from __future__ import annotations

import base64
import json
import pickle
import socket
import struct
import threading
from typing import Any, Optional

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


def recv_message(conn: socket.socket) -> Optional[dict]:
    head = _recv_exact(conn, _HEADER.size)
    if head is None:
        return None
    (n,) = _HEADER.unpack(head)
    if n > _MAX_MESSAGE:
        raise ValueError(f"message of {n} bytes exceeds the {_MAX_MESSAGE}-byte limit")
    body = _recv_exact(conn, n)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))


def send_message(conn: socket.socket, payload: dict) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    conn.sendall(_HEADER.pack(len(body)) + body)


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
        self._inputs: dict[str, dict[str, Any]] = {}      # {node: {field: array}}
        self._time = 0.0
        self._initial_state = pickle.dumps(sidecar.state)
        self._initial_params = pickle.dumps(sidecar.params)
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
            with conn:
                conn.settimeout(None)
                while not self._stop.is_set():
                    try:
                        req = recv_message(conn)
                    except (OSError, ValueError):
                        break
                    if req is None:
                        break
                    send_message(conn, self.handle(req))

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
                n = max(int(round(h / self._dt)), 1)
                for _ in range(n):
                    self._sidecar.step(self._inputs)
                self._time = float(req.get("t", self._time)) + h
                return {"ok": True, "t": self._time}
            if op == "get_state":
                blob = pickle.dumps((self._sidecar.get_fmu_state(), self._time, self._inputs))
                return {"ok": True, "state": base64.b64encode(blob).decode("ascii")}
            if op == "set_state":
                fmu_state, t, inputs = pickle.loads(base64.b64decode(req["state"]))
                self._sidecar.set_fmu_state(fmu_state)
                self._time, self._inputs = float(t), inputs
                return {"ok": True}
            if op == "reset":
                self._sidecar._state = pickle.loads(self._initial_state)   # noqa: SLF001
                params = pickle.loads(self._initial_params)
                if params is not None:
                    self._sidecar._params = params                          # noqa: SLF001
                self._inputs, self._time = {}, 0.0
                return {"ok": True}
            if op == "terminate":
                return {"ok": True}
            return {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as exc:  # noqa: BLE001 - reported to the importer
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

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
        for var, arr in staged:
            if var.causality == "parameter":
                param_updates[var.name] = arr.astype(var.dtype)
            elif var.causality == "input":
                node, _, field = var.name.partition(".")
                self._inputs.setdefault(node, {})[field] = arr.astype(var.dtype)
            elif var.is_clock:
                continue                              # clock ticks are informational
            else:
                raise ValueError(f"variable {var.name!r} ({var.causality}) is read-only")
        if param_updates:
            self._sidecar.set_params(param_updates)   # atomic + bounds

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


__all__ = ["FmuTcpBridge", "recv_message", "send_message"]
