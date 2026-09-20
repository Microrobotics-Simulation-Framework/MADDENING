"""
Network transport for remote visualization and command input.

State output (simulation -> visualization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``NetworkRelay`` publishes state on a ZMQ PUB socket.
``NetworkReceiver`` subscribes and exposes ``latest_snapshot()``.

Command input (controller -> simulation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``CommandPublisher`` sends commands on a ZMQ PUB socket.
``CommandReceiver`` subscribes and exposes ``latest_commands()``.

Both directions use the same single-slot-latest-value pattern
with ``ZMQ.CONFLATE`` so only the most recent data matters.

Security
~~~~~~~~
Every socket here defaults to a **loopback** address, so the local
development flow needs no configuration.  Giving any of them an address
another host can reach turns on ZMQ CURVE encryption and
authentication, keyed by the shared ``MADDENING_TRANSPORT_TOKEN``; see
:mod:`maddening.transport_auth`.  The recommended way to reach a remote
simulation remains an SSH tunnel to the loopback port, which needs no
token at all.

Typical SSH-tunnel topology::

    Cloud / HPC                         Local / Robot
    ───────────                         ────────────
    GraphManager                        Controller (ROS2, etc.)
        │ observer                          │
        ▼                                   ▼
    NetworkRelay (PUB :5555)           CommandPublisher (PUB :5556)
        │ state                             │ commands
        ▼                                   ▼
    ─── ZMQ ──────────────────────────── ZMQ ───
        │                                   │
        ▼                                   ▼
    CommandReceiver (SUB :5556)        NetworkReceiver (SUB :5555)
        │                                   │
        ▼                                   ▼
    RealtimeRunner.step(ext_inputs)    Renderer(s)
"""

import json
import threading
from typing import Any, Optional

from maddening.transport_auth import TransportAuth, resolve_security

try:
    import zmq
except ImportError as _exc:
    raise ImportError(
        "Network transport requires 'pyzmq'. "
        "Install with:  pip install maddening[network]"
    ) from _exc


# ------------------------------------------------------------------
# State output: simulation -> visualization
# ------------------------------------------------------------------

class NetworkRelay:
    """Publish simulation state over ZMQ (runs on the simulation side).

    Attaches to a ``GraphManager`` as an observer.  On each step,
    serializes the state dict to JSON and publishes it on a ZMQ PUB
    socket.

    Parameters
    ----------
    address : str
        ZMQ bind address (default ``"tcp://127.0.0.1:5555"``).

        .. versionchanged:: 0.4.0
           The default was ``"tcp://*:5555"``, which published the full
           simulation state on every interface with no authentication.
           It is now loopback-only.  To stream off-box, forward the port
           over SSH (``ssh -L 5555:127.0.0.1:5555 <host>``), or pass an
           explicit non-loopback address, which requires a token.
    fields : dict, optional
        ``{node_name: [field1, field2, ...]}`` — publish only these
        node/field combinations.  If ``None`` (default) the full state
        dict is published.  Nodes or fields not present in the state
        are silently dropped at publish time.  Use this to cut wire
        bandwidth when remote clients only care about a subset
        (e.g. ``fields={"lbm": ["velocity"]}`` to skip the 16-fdist
        f-distribution arrays in a Lattice-Boltzmann graph).
    """

    def __init__(
        self,
        address: str = "tcp://127.0.0.1:5555",
        fields: dict[str, list[str]] | None = None,
        secure: bool | None = None,
        token: str | None = None,
    ):
        self._address = address
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._secure = resolve_security(address, secure)
        self._authenticator = None
        if self._secure:
            auth = TransportAuth(token=token)
            self._authenticator = auth.start_authenticator(self._context)
            auth.secure_server(self._socket)
        self._socket.bind(address)
        self._step_count = 0
        self._timestep = 0.0
        self._fields = fields

    @property
    def secure(self) -> bool:
        """Whether this socket is encrypted and authenticated with CURVE."""
        return self._secure

    @property
    def subscription(self) -> dict[str, list[str]] | None:
        """Return the active field filter, or ``None`` if no filter."""
        return self._fields

    def _filter_state(self, data) -> dict:
        """Apply the field filter (no-op when ``self._fields`` is None)."""
        if self._fields is None:
            return {
                node: {k: _to_json(v) for k, v in fields.items()}
                for node, fields in data.items()
            }
        out: dict[str, dict] = {}
        for node, allowed_fields in self._fields.items():
            node_state = data.get(node)
            if node_state is None:
                continue
            allowed = set(allowed_fields)
            slot = {k: _to_json(v) for k, v in node_state.items() if k in allowed}
            if slot:
                out[node] = slot
        return out

    def attach(self, graph_manager) -> None:
        """Register as an observer on *graph_manager*."""
        self._timestep = graph_manager.timestep
        graph_manager.add_observer(self._on_event)

    def _on_event(self, event: str, data) -> None:
        if event != "step":
            return
        self._step_count += 1
        sim_time = self._step_count * self._timestep

        state = self._filter_state(data)
        payload = json.dumps({"t": sim_time, "state": state}).encode()
        try:
            self._socket.send(payload, zmq.NOBLOCK)
        except zmq.Again:
            pass  # drop frame if send buffer is full

    def close(self) -> None:
        """Shut down the socket and context."""
        self._socket.close()
        if self._authenticator is not None:
            self._authenticator.stop()
            self._authenticator = None
        self._context.term()


def _to_json(v):
    """Convert a JAX/NumPy scalar or array to a JSON-friendly value."""
    if hasattr(v, "shape") and getattr(v, "shape", ()) != ():
        # array — coerce to plain list of floats
        import numpy as _np
        return _np.asarray(v).astype(float).tolist()
    return float(v)


class NetworkReceiver:
    """Receive simulation state over ZMQ (runs on the visualization side).

    Exposes ``latest_snapshot()`` with the same signature as
    ``StateRelay``, so any renderer works as a drop-in.

    Parameters
    ----------
    address : str
        ZMQ connect address (default ``"tcp://localhost:5555"``).
    secure : bool, optional
        Whether to encrypt and authenticate the socket with ZMQ CURVE.
        ``None`` (default) decides from *address*: a loopback endpoint
        is left in cleartext, anything reachable from another host is
        secured.  ``True`` forces CURVE on even for loopback.  ``False``
        is rejected for a non-loopback address -- see
        :mod:`maddening.transport_auth`.
    token : str, optional
        The shared secret both ends derive their CURVE keys from.
        ``None`` reads ``MADDENING_TRANSPORT_TOKEN``, falling back to
        ``MADDENING_API_TOKEN`` -- which is the HTTP API's cleartext
        bearer credential, so prefer the transport variable; see
        :mod:`maddening.transport_auth`.  Only consulted when the
        socket is secured.
    """

    def __init__(
        self,
        address: str = "tcp://localhost:5555",
        secure: bool | None = None,
        token: str | None = None,
    ):
        self._address = address
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._secure = resolve_security(address, secure)
        if self._secure:
            TransportAuth(token=token).secure_client(self._socket)
        self._socket.connect(address)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._lock = threading.Lock()
        self._snapshot: Optional[dict] = None
        self._sim_time: float = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        """Start the background receive thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._stop.is_set():
            socks = dict(poller.poll(timeout=100))
            if self._socket in socks:
                payload = self._socket.recv()
                msg = json.loads(payload)
                with self._lock:
                    self._sim_time = msg["t"]
                    self._snapshot = msg["state"]

    def latest_snapshot(self) -> tuple[float, Optional[dict]]:
        """Same interface as ``StateRelay.latest_snapshot()``."""
        with self._lock:
            return (self._sim_time, self._snapshot)

    def stop(self) -> None:
        """Stop the receive thread and close the socket."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._socket.close()
        self._context.term()


# ------------------------------------------------------------------
# Command input: controller -> simulation
# ------------------------------------------------------------------

class CommandPublisher:
    """Send commands to a remote simulation (runs on the controller side).

    Publishes a JSON-encoded dict of external inputs over ZMQ PUB.
    The simulation's ``CommandReceiver`` picks up the latest value.

    Parameters
    ----------
    address : str
        ZMQ bind address (default ``"tcp://127.0.0.1:5556"``).

        .. versionchanged:: 0.4.0
           The default was ``"tcp://*:5556"``.  This socket carries
           control commands that are passed straight to
           ``GraphManager.step(external_inputs=...)``, so publishing it
           on every interface let anyone who could reach the port both
           read and drive the simulation.  It is now loopback-only.
    secure : bool, optional
        Whether to encrypt and authenticate the socket with ZMQ CURVE.
        ``None`` (default) decides from *address*: a loopback endpoint
        is left in cleartext, anything reachable from another host is
        secured.  ``True`` forces CURVE on even for loopback.  ``False``
        is rejected for a non-loopback address -- see
        :mod:`maddening.transport_auth`.
    token : str, optional
        The shared secret both ends derive their CURVE keys from.
        ``None`` reads ``MADDENING_TRANSPORT_TOKEN``, falling back to
        ``MADDENING_API_TOKEN`` -- which is the HTTP API's cleartext
        bearer credential, so prefer the transport variable; see
        :mod:`maddening.transport_auth`.  Only consulted when the
        socket is secured.

    Example
    -------
    ::

        pub = CommandPublisher()          # loopback, no token needed
        pub.send({"robot": {"joint_torques": [0.1, -0.2, 0.0, ...]}})

        # Off-box, with MADDENING_TRANSPORT_TOKEN set on both sides:
        pub = CommandPublisher("tcp://0.0.0.0:5556")
    """

    def __init__(
        self,
        address: str = "tcp://127.0.0.1:5556",
        secure: bool | None = None,
        token: str | None = None,
    ):
        self._address = address
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._secure = resolve_security(address, secure)
        self._authenticator = None
        if self._secure:
            auth = TransportAuth(token=token)
            self._authenticator = auth.start_authenticator(self._context)
            auth.secure_server(self._socket)
        self._socket.bind(address)

    @property
    def secure(self) -> bool:
        """Whether this socket is encrypted and authenticated with CURVE."""
        return self._secure

    def send(self, external_inputs: dict[str, dict]) -> None:
        """Publish a command dict.

        Structure matches ``GraphManager.step(external_inputs=...)``:
        ``{node_name: {field_name: value, ...}, ...}``.
        Values should be plain Python floats or lists (JSON-serializable).
        """
        payload = json.dumps(external_inputs).encode()
        try:
            self._socket.send(payload, zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self) -> None:
        self._socket.close()
        if self._authenticator is not None:
            self._authenticator.stop()
            self._authenticator = None
        self._context.term()


class CommandReceiver:
    """Receive commands from a remote controller (runs on the simulation side).

    Exposes ``latest_commands()`` which returns the most recent command
    dict, structured as ``{node_name: {field_name: value}}``.  This is
    passed directly to ``GraphManager.step(external_inputs=...)``.

    Parameters
    ----------
    address : str
        ZMQ connect address (default ``"tcp://localhost:5556"``).
    secure : bool, optional
        Whether to encrypt and authenticate the socket with ZMQ CURVE.
        ``None`` (default) decides from *address*: a loopback endpoint
        is left in cleartext, anything reachable from another host is
        secured.  ``True`` forces CURVE on even for loopback.  ``False``
        is rejected for a non-loopback address -- see
        :mod:`maddening.transport_auth`.
    token : str, optional
        The shared secret both ends derive their CURVE keys from.
        ``None`` reads ``MADDENING_TRANSPORT_TOKEN``, falling back to
        ``MADDENING_API_TOKEN`` -- which is the HTTP API's cleartext
        bearer credential, so prefer the transport variable; see
        :mod:`maddening.transport_auth`.  Only consulted when the
        socket is secured.
    """

    def __init__(
        self,
        address: str = "tcp://localhost:5556",
        secure: bool | None = None,
        token: str | None = None,
    ):
        self._address = address
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._secure = resolve_security(address, secure)
        if self._secure:
            TransportAuth(token=token).secure_client(self._socket)
        self._socket.connect(address)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._lock = threading.Lock()
        self._commands: Optional[dict] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        """Start the background receive thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._stop.is_set():
            socks = dict(poller.poll(timeout=100))
            if self._socket in socks:
                payload = self._socket.recv()
                msg = json.loads(payload)
                with self._lock:
                    self._commands = msg

    def latest_commands(self) -> Optional[dict[str, dict]]:
        """Return the most recent command dict, or ``None``."""
        with self._lock:
            return self._commands

    def stop(self) -> None:
        """Stop the receive thread and close the socket."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._socket.close()
        self._context.term()
