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
        ZMQ bind address (default ``"tcp://*:5555"``).
    """

    def __init__(self, address: str = "tcp://*:5555"):
        self._address = address
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(address)
        self._step_count = 0
        self._timestep = 0.0

    def attach(self, graph_manager) -> None:
        """Register as an observer on *graph_manager*."""
        self._timestep = graph_manager.timestep
        graph_manager.add_observer(self._on_event)

    def _on_event(self, event: str, data) -> None:
        if event != "step":
            return
        self._step_count += 1
        sim_time = self._step_count * self._timestep

        # Convert JAX arrays to plain floats for JSON serialization
        state = {
            node: {k: float(v) for k, v in fields.items()}
            for node, fields in data.items()
        }
        payload = json.dumps({"t": sim_time, "state": state}).encode()
        try:
            self._socket.send(payload, zmq.NOBLOCK)
        except zmq.Again:
            pass  # drop frame if send buffer is full

    def close(self) -> None:
        """Shut down the socket and context."""
        self._socket.close()
        self._context.term()


class NetworkReceiver:
    """Receive simulation state over ZMQ (runs on the visualization side).

    Exposes ``latest_snapshot()`` with the same signature as
    ``StateRelay``, so any renderer works as a drop-in.

    Parameters
    ----------
    address : str
        ZMQ connect address (default ``"tcp://localhost:5555"``).
    """

    def __init__(self, address: str = "tcp://localhost:5555"):
        self._address = address
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
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
        ZMQ bind address (default ``"tcp://*:5556"``).

    Example
    -------
    ::

        pub = CommandPublisher("tcp://*:5556")
        pub.send({"robot": {"joint_torques": [0.1, -0.2, 0.0, ...]}})
    """

    def __init__(self, address: str = "tcp://*:5556"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(address)

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
    """

    def __init__(self, address: str = "tcp://localhost:5556"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
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
