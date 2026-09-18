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
that sends ``{"op": "hello"}`` gets the protocol-1 behaviour (the hello
reply merely gains ``protocol`` and ``binary``; ``recv_message`` /
``send_message`` here speak both forms).  A client announcing a protocol
this bridge does not know is refused at hello.

**Frame limit.**  A frame is at most 64 MiB (``_MAX_MESSAGE``) in both
directions: a longer request drops the connection (its length is not to
be trusted), and a reply that would be longer (a ``get`` of more than
about 8 M values, a huge state) is replaced by a JSON error reply, so the
connection stays in sync and the C wrapper, which refuses to read a
longer frame, never sees one from this bridge.

**Connection lifetime.**  A connection holds the bridge's single FMU
instance for as long as it lives, so no wait on it is unbounded: a peer
has ten seconds to begin its first frame, five minutes of silence
between frames once it has spoken, and two minutes to finish a frame it
has announced the length of.  Overrunning any of them ends the
connection exactly as EOF does, and the instance slot is free again.
The number of live connection threads is capped (16); further
connections are closed on accept.  ``stop()`` shuts every live
connection down, so a parked worker does not outlive the bridge.

The importer is **untrusted**: nothing that arrives on the socket is ever
unpickled or evaluated.  The FMU-state blob is an ``npz`` archive of plain
arrays (``allow_pickle=False`` on load) carrying the schema token, the
time, the pending inputs, the node states and the params; on
``set_state`` the archive directory is checked first (only the expected
member names, each member's declared size capped by the live array it
replaces, and a cap on the total), so nothing is decompressed that the
model could not hold, and then every array is checked against the live
one (token, key set, shape) before anything is written.  Bind the bridge
to ``127.0.0.1`` unless the network is trusted.

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
import time
import zipfile
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.params import check_bounds
from maddening.fmi.model_description import FMIVariable, ModelDescription
from maddening.fmi.sidecar import FmuSidecar

_HEADER = struct.Struct(">I")
_MAX_MESSAGE = 64 * 1024 * 1024
"""Frame limit in bytes, both directions and both frame kinds (the C wrapper's FRAME_MAX)."""
_BINARY_FLAG = 0x80000000
_LENGTH_MASK = 0x7FFFFFFF
_NPY_SLACK = 4096
"""Bytes an ``npz`` member may exceed its array by (the ``.npy`` header)."""
PROTOCOL_VERSION = 2
"""Highest sidecar protocol this bridge speaks (1 = JSON only, 2 = + binary frames)."""

# ---------------------------------------------------------------- timeouts
# A connection holds the bridge's single instance slot (``_busy``) for as
# long as it lives, so every wait on it is bounded.  The three budgets are
# separate because a legitimate importer's silences are of three different
# lengths, and one number generous enough for the longest would leave the
# instance slot parkable by a peer that says nothing at all.
_HANDSHAKE_TIMEOUT = 10.0
"""Seconds a freshly accepted connection has to start its first frame.

The C wrapper sends ``hello`` immediately after ``connect`` (see
``bridge_connect`` in ``c/maddening_fmu.c``), so ten seconds is already
three orders of magnitude more than a healthy importer needs, while a
port scan, a crashed importer or a dropped link is dropped promptly
instead of owning the instance for ever."""
_IDLE_TIMEOUT = 300.0
"""Seconds an established connection may stay silent between frames.

An importer is idle between ``doStep`` calls, and the master may be
waiting on a slow co-simulation partner or on a human at a debugger
prompt, so this one is deliberately generous: five minutes of silence
from a client that has already completed a handshake is a link that is
gone, not a slow one."""
_FRAME_TIMEOUT = 120.0
"""Seconds to finish a frame once its length prefix has arrived.

Bounds the dribbling peer the per-recv timeout alone does not: one byte
every nine seconds would renew a plain socket timeout for ever.  Two
minutes still covers a full 64 MiB frame on a link of about 5 Mbit/s."""
_MAX_CONNECTIONS = 16
"""Live connection threads allowed at once.

The bridge serves one FMU instance, so every connection beyond the first
is refused anyway; the cap exists so that refusing them costs a bounded
number of threads."""


def _json_object(body: bytes, what: str) -> Any:
    """``json.loads`` of ``body``; every failure is a ``ValueError``.

    The importer's bytes may be anything: not UTF-8, not JSON, or nested
    so deeply that the JSON scanner raises ``RecursionError``.  All of
    those must come out as the one exception the framing contract
    promises, so the connection loop answers with an error reply instead
    of dying with a traceback.
    """
    try:
        return json.loads(body.decode("utf-8"))
    except RecursionError as exc:
        raise ValueError(f"{what} is nested too deeply") from exc
    except ValueError as exc:                  # JSONDecodeError, UnicodeDecodeError
        raise ValueError(f"{what} is not JSON: {exc}") from exc


def _recv_exact(conn: socket.socket, n: int,
                deadline: Optional[float] = None) -> Optional[bytes]:
    """``n`` bytes, ``None`` at EOF.

    ``deadline`` (a :func:`time.monotonic` value) bounds the whole read,
    not each ``recv``: a peer that dribbles one byte at a time renews the
    socket's own timeout indefinitely, and the connection it is dribbling
    on holds the bridge's only instance slot.  Overrunning it raises
    :exc:`socket.timeout`, which every caller already treats as a dead
    connection.
    """
    buf = bytearray()
    while len(buf) < n:
        if deadline is not None and time.monotonic() > deadline:
            raise socket.timeout(
                f"frame of {n} bytes was still incomplete after "
                f"{len(buf)} bytes"
            )
        chunk = conn.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_raw(conn: socket.socket, *,
             frame_timeout: Optional[float] = None) -> Optional[tuple[bool, bytes]]:
    """One length-prefixed frame as ``(is_binary, payload)``.

    ``None`` at EOF; ``ValueError`` when the (31-bit) length exceeds the
    64 MiB limit, binary flag or not.  ``frame_timeout`` bounds the body
    once the length prefix has arrived (:exc:`socket.timeout` on
    overrun); the wait for the prefix itself is the socket's own timeout,
    which the caller sets according to what the connection is waiting
    for.
    """
    head = _recv_exact(conn, _HEADER.size)
    if head is None:
        return None
    (word,) = _HEADER.unpack(head)
    n = word & _LENGTH_MASK
    if n > _MAX_MESSAGE:
        raise ValueError(f"message of {n} bytes exceeds the {_MAX_MESSAGE}-byte limit")
    deadline = None if frame_timeout is None else time.monotonic() + frame_timeout
    body = _recv_exact(conn, n, deadline)
    if body is None:
        return None
    return bool(word & _BINARY_FLAG), body


def recv_frame(conn: socket.socket, *,
               frame_timeout: Optional[float] = None) -> Optional[bytes]:
    """One length-prefixed frame's payload (``None`` at EOF; ``ValueError``
    over the limit).  Use :func:`recv_raw` to learn whether it was binary."""
    got = recv_raw(conn, frame_timeout=frame_timeout)
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
    header = _json_object(payload[_HEADER.size:_HEADER.size + hlen], "binary header")
    if not isinstance(header, dict):
        raise ValueError("binary header must be a JSON object")
    return header, payload[_HEADER.size + hlen:]


def recv_message(conn: socket.socket, *,
                 frame_timeout: Optional[float] = None) -> Optional[dict]:
    """One decoded message.  A JSON frame is its object; a binary frame is
    its header with the raw payload under ``"raw"`` (``bytes``), a dict
    :meth:`FmuTcpBridge.handle` accepts as is.  ``ValueError`` on a
    malformed frame of either kind."""
    got = recv_raw(conn, frame_timeout=frame_timeout)
    if got is None:
        return None
    is_binary, body = got
    if is_binary:
        header, raw = decode_binary(body)
        header["raw"] = raw
        return header
    return _json_object(body, "message")


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


def checked_value(arr, dtype, *, what: str) -> np.ndarray:
    """``arr`` in ``dtype``, refused unless the model can hold it.

    The one value check on this module's write paths.  ``set`` and
    ``set_state`` both go through it, so an FMU-state archive cannot
    install a value a ``set`` of the same variable would refuse -- which
    it could until 0.4.0, because the two paths each had their own idea
    of what a valid value was and only one of them had any.

    Parameters
    ----------
    arr : array-like
        The incoming value, in whatever dtype it arrived in (float64 off
        the wire, the archive's own dtype out of an ``npz``).
    dtype : numpy dtype
        The dtype of the live array it would replace.
    what : str
        How to name the value in an error, e.g. ``"variable 'm.params.k'"``.

    Returns
    -------
    numpy.ndarray
        ``arr`` cast to ``dtype``.

    Raises
    ------
    ValueError
        If the incoming value is not finite, or if ``dtype`` cannot hold
        it: a float32 field set to ``1e308`` would be stored (and read
        back) as ``inf``, and an integer would wrap silently.
    """
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.inexact) and not bool(np.all(np.isfinite(a))):
        raise ValueError(f"{what}: value must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        cast = a.astype(dtype)
    if np.issubdtype(cast.dtype, np.floating):
        fits = bool(np.all(np.isfinite(cast)))
    elif np.issubdtype(cast.dtype, np.integer):
        fits = bool(np.array_equal(cast.astype(np.float64), a.astype(np.float64)))
    else:
        fits = True                                       # bool
    if not fits:
        raise ValueError(f"{what}: value does not fit its type {dtype}")
    return cast


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
        self._server.listen(_MAX_CONNECTIONS)
        self._host, self._port = self._server.getsockname()[:2]
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Live accepted connections and the threads serving them.  ``stop()``
        # has to reach both: closing the listening socket says nothing to a
        # connection already accepted, and a worker parked on one of those
        # outlives the bridge that "stopped".
        self._live_lock = threading.Lock()
        self._live_conns: set[socket.socket] = set()
        self._live_workers: set[threading.Thread] = set()
        self.requests_served = 0
        self.binary_frames_served = 0     # binary replies sent (get / get_state)
        self.binary_frames_received = 0   # binary requests accepted (set / set_state)
        self.connections_refused_over_cap = 0

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
        """Stop accepting and end every connection this bridge still holds.

        Closing the listening socket only stops new connections; a worker
        blocked reading an accepted one is untouched by it and used to
        survive ``stop()`` indefinitely, still holding the instance lock.
        Each live connection is therefore shut down here, which turns the
        worker's pending ``recv`` into an EOF, and the workers are joined.
        """
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass
        with self._live_lock:
            conns = list(self._live_conns)
            workers = list(self._live_workers)
        for conn in conns:
            # shutdown, not close: the worker owns the socket object and
            # closes it on its way out, and a half-close is what makes its
            # blocking recv return instead of waiting for a peer that is
            # never going to speak.
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        for worker in workers:
            worker.join(timeout=5.0)

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
            worker = threading.Thread(target=self._serve_conn, args=(conn,),
                                      name="maddening-fmu-conn", daemon=True)
            with self._live_lock:
                if len(self._live_conns) >= _MAX_CONNECTIONS:
                    over_cap = True
                else:
                    over_cap = False
                    self._live_conns.add(conn)
                    self._live_workers.add(worker)
            if over_cap:
                # Nothing is written back: a reply needs a send that a peer
                # which is not reading can stall, and stalling the accept
                # loop is exactly what the cap exists to prevent.
                self.connections_refused_over_cap += 1
                try:
                    conn.close()
                except OSError:
                    pass
                continue
            worker.start()

    def _serve_conn(self, conn: socket.socket) -> None:
        try:
            self._serve_conn_inner(conn)
        finally:
            with self._live_lock:
                self._live_conns.discard(conn)
                self._live_workers.discard(threading.current_thread())

    def _serve_conn_inner(self, conn: socket.socket) -> None:
        with conn:
            if self._stop.is_set():
                return
            binary = False                # negotiated at hello, per connection
            held = False                  # does this connection hold the instance?
            try:
                while not self._stop.is_set():
                    # Every wait on this socket is finite, and the first one
                    # happens before the instance slot is claimed.  Both
                    # matter: a peer that connects and says nothing used to
                    # park the bridge's only FMU instance for ever (a port
                    # scan, a crashed importer or a dropped link was enough),
                    # and a peer that was refused the slot used to park a
                    # thread for ever.
                    conn.settimeout(_IDLE_TIMEOUT if held else _HANDSHAKE_TIMEOUT)
                    try:
                        got = recv_raw(conn, frame_timeout=_FRAME_TIMEOUT)
                    except socket.timeout:
                        break        # a silent peer is a gone peer: same as EOF
                    except (OSError, ValueError):
                        break                              # socket / framing error
                    if got is None:
                        break
                    if not held:
                        # A bridge holds ONE sidecar state; a second instance
                        # would silently share it.  Refuse instead of
                        # blocking -- but only now that this peer has proved
                        # it has something to say.
                        if not self._busy.acquire(blocking=False):
                            try:
                                send_message(conn, {"ok": False, "error":
                                                    "bridge already serves an FMU instance; "
                                                    "start one FmuTcpBridge per instance"})
                            except OSError:
                                pass
                            return
                        held = True
                    is_binary, body = got
                    try:
                        if is_binary:
                            if not binary:
                                raise ValueError("binary frames need a hello with "
                                                 "\"protocol\": 2, \"binary\": true first")
                            req = self._binary_request(*decode_binary(body))
                            self.binary_frames_received += 1
                        else:
                            req = _json_object(body, "request")
                            if not isinstance(req, dict):
                                raise ValueError("request must be a JSON object")
                    except ValueError as exc:
                        # a corrupt request (an FMU-state blob with stray
                        # quotes, say) is an error reply, not a dead instance
                        reply = {"ok": False, "error": f"malformed request: {exc}"}
                    else:
                        reply = self._dispatch(req)
                        if req.get("op") == "hello" and reply.get("ok"):
                            binary = bool(reply.get("binary"))
                    try:
                        if self._send_reply(conn, reply, binary):
                            self.binary_frames_served += 1
                    except OSError:
                        break          # the importer hung up mid-reply: nobody to tell
            finally:
                if held:
                    self._busy.release()

    def _send_reply(self, conn: socket.socket, reply: dict, binary: bool) -> bool:
        """Send one reply; returns whether it went as a binary frame.

        A successful ``get`` / ``get_state`` on a binary connection is a
        binary frame, everything else JSON.  A reply of either kind that
        would exceed the frame limit is replaced by a JSON error reply:
        the bridge never puts a frame on the wire that the C wrapper
        would refuse to read, so the connection stays in sync.
        """
        if binary and reply.get("ok") and ("values" in reply or "state" in reply):
            if "values" in reply:
                arr = np.ascontiguousarray(reply["values"], dtype="<f8")
                header, raw = {"ok": True, "n": int(arr.size), "dtype": "f64"}, arr.tobytes()
            else:
                header, raw = {"ok": True, "n": len(reply["state"])}, reply["state"]
            body = encode_binary(header, raw)
            if len(body) <= _MAX_MESSAGE:
                conn.sendall(_HEADER.pack(_BINARY_FLAG | len(body)) + body)
                return True
        else:
            body = json.dumps(self._jsonify(reply), separators=(",", ":")).encode("utf-8")
            if len(body) <= _MAX_MESSAGE:
                conn.sendall(_HEADER.pack(len(body)) + body)
                return False
        send_message(conn, {"ok": False, "error": f"reply of {len(body)} bytes exceeds the "
                                                   f"{_MAX_MESSAGE}-byte frame limit"})
        return False

    @staticmethod
    def _binary_request(header: dict, raw) -> dict:
        """A binary ``set`` / ``set_state`` request (its decoded header and
        raw part) as the plain request dict :meth:`_dispatch` takes
        (``values`` as an array, ``state`` as bytes).  ``ValueError`` on
        any inconsistency; nothing is trusted."""
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise ValueError("the raw part of a binary request must be bytes")
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
            return {"op": "set_state", "state": bytes(raw)}
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
        JSON form: ``values`` come back as a list, ``state`` as base64.

        ``req`` is either the JSON form or the dict :func:`recv_message`
        returns for a binary frame (the header's keys plus ``"raw"``);
        the latter is validated exactly as on the socket path.
        """
        if "raw" in req:
            try:
                req = self._binary_request({k: v for k, v in req.items() if k != "raw"}, req["raw"])
            except ValueError as exc:
                return {"ok": False, "error": f"malformed request: {exc}"}
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
                # The communication point is parsed *before* anything moves.
                # It used to be parsed after the loop, so a request carrying a
                # ``t`` that is not a number was answered "not ok" with the
                # physics already advanced and ``_time`` left behind it: the
                # importer's clock and the bridge's state desynchronise
                # permanently, and nothing on the wire says so.
                raw_t = req.get("t", self._time)
                try:
                    t0 = float(raw_t)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"communication point must be a number, got {raw_t!r}"
                    ) from exc
                if not np.isfinite(t0):
                    raise ValueError(
                        f"communication point must be finite, got {raw_t!r}"
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
                self._time = t0 + n * self._dt
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
                node, field = var.node_field()
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

    def _member_caps(self) -> dict[str, int]:
        """Every member an FMU-state archive may carry (name without the
        ``.npy`` suffix) and the most bytes it may decompress to: the live
        array it replaces plus the ``.npy`` header."""
        caps = {"_token": 256, "_time": 8}
        for n, fields in self._sidecar.state.items():
            for f, v in fields.items():
                caps[f"s/{n}/{f}"] = int(np.asarray(v).nbytes)
        for section in ("nodes", "mappings"):
            for owner, leaves in (self._sidecar.params or {}).get(section, {}).items():
                for k, v in leaves.items():
                    caps[f"p/{section}/{owner}/{k}"] = int(np.asarray(v).nbytes)
        for var in self._md.variables:
            if var.causality == "input":
                node, field = var.node_field()
                caps[f"i/{node}/{field}"] = int(np.zeros(var.shape or (), var.dtype).nbytes)
        return {k: v + _NPY_SLACK for k, v in caps.items()}

    def _check_archive_directory(self, blob: bytes) -> None:
        """Refuse the archive from its directory alone, before any member
        is decompressed: every member (whatever its name) must be one the
        model expects and declare no more than that member may hold, and
        the total declared size is capped too (a zip bomb is a small blob
        declaring gigabytes; deflate alone gives about 1000:1)."""
        caps = self._member_caps()
        try:
            zf = zipfile.ZipFile(io.BytesIO(blob))
        except Exception as exc:  # noqa: BLE001 - BadZipFile and friends
            raise ValueError(f"FMU state blob is not a valid archive: {exc}") from exc
        with zf:
            total = 0
            for info in zf.infolist():
                name = info.filename
                key = name[:-4] if name.endswith(".npy") else None
                if key is None or key not in caps:
                    kind = {"s": "state field", "p": "parameter", "i": "input"}.get(
                        name.split("/", 1)[0], "member")
                    raise ValueError(f"FMU state carries unknown {kind} {name!r}")
                if info.file_size > caps[key]:
                    raise ValueError(f"FMU state member {key!r} is {info.file_size} bytes, more "
                                     f"than the {caps[key]} the model can hold")
                total += info.file_size
            budget = sum(caps.values())
            if total > budget:
                raise ValueError(f"FMU state declares {total} bytes in total, more than the "
                                 f"{budget} the model can hold")

    def _decode_state(self, blob: bytes) -> None:
        """Validate against the live state before writing anything.

        An archive may only install values a ``set`` of the same variables
        would be allowed to install: every restored array goes through
        :func:`checked_value` (finite, and representable in the live
        array's dtype) and the restored parameter tree through
        ``check_bounds`` against the graph's declared ``ParamSpec``.  A snapshot
        of a diverged model -- one holding ``inf`` or ``NaN`` -- therefore
        does not restore; the error names the field.
        """
        self._check_archive_directory(blob)
        try:
            data = np.load(io.BytesIO(blob), allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"FMU state blob is not a valid archive: {exc}") from exc
        with data:
            keys = set(data.files)
            if "_token" not in keys:
                raise ValueError("FMU state belongs to a different model (schema token mismatch)")
            token = data["_token"]
            if token.dtype.kind != "U" or token.shape != () or str(token) != self._md.instantiation_token:
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
                new_state.setdefault(node, {})[field] = jnp.asarray(
                    checked_value(arr, live.dtype, what=f"FMU state {node}.{field}")
                )
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
                            new_params[section][owner][k] = jnp.asarray(
                                checked_value(arr, live.dtype,
                                              what=f"FMU state param {owner}.params.{k}")
                            )
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
                    inputs.setdefault(node, {})[field] = jnp.asarray(
                        checked_value(arr, var.dtype,
                                      what=f"FMU state input {node}.{field}")
                    )
            t = float(data["_time"]) if "_time" in keys else 0.0
            if not np.isfinite(t):
                raise ValueError("FMU state carries a non-finite time")
        if new_params is not None:
            # The bounds the model description advertises, applied to the
            # archive exactly as ``set`` applies them through
            # ``FmuSidecar.set_params``.  Without this an importer could
            # restore mass = -1.0 against a declared (0.1, 10.0) and the
            # bridge would answer ok -- the documented guarantee is that it
            # cannot silently tune a constant the graph declares invalid,
            # and that has to hold for both doors into the parameter tree.
            check_bounds(new_params, self._sidecar.param_specs or {})
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
            if var.is_clock:
                continue                              # clock ticks are informational
            if var.causality == "parameter":
                param_updates[var.name] = self._in_dtype(var, arr)
            elif var.causality == "input":
                node, field = var.node_field()
                input_updates.append((node, field, self._in_dtype(var, arr)))
            else:
                raise ValueError(f"variable {var.name!r} ({var.causality}) is read-only")
        # Atomic: parameters are validated (bounds) by the sidecar first;
        # inputs are only committed once nothing can fail any more.
        if param_updates:
            self._sidecar.set_params(param_updates)
        for node, field, arr in input_updates:
            self._inputs.setdefault(node, {})[field] = arr

    @staticmethod
    def _in_dtype(var: FMIVariable, arr: np.ndarray) -> np.ndarray:
        """``arr`` in the variable's dtype, refused when the dtype cannot
        hold it.  The shared check :func:`checked_value`, named for the
        FMI variable; ``set_state`` applies the same one."""
        return checked_value(arr, var.dtype, what=f"variable {var.name!r}")

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
                node, field = var.node_field()
                val = self._inputs.get(node, {}).get(field)
                if val is None:
                    val = np.zeros(var.shape or (), dtype=np.float64)
                parts.append(np.asarray(val, dtype=np.float64).ravel())
            elif var.is_clock:
                parts.append(np.zeros(1, dtype=np.float64))
            else:
                node, field = var.node_field()
                parts.append(np.asarray(self._sidecar.state[node][field],
                                        dtype=np.float64).ravel())
        if not parts:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(parts)


__all__ = ["FmuTcpBridge", "PROTOCOL_VERSION", "checked_value", "decode_binary",
           "encode_binary", "recv_frame", "recv_message", "recv_raw", "send_binary",
           "send_message", "state_of", "values_of"]
