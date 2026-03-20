"""ZMQ-based coordinator for multi-job distributed simulation.

The coordinator runs on the rank-0 VM as a separate process. It:
1. Binds a ZMQ ROUTER socket on a known port
2. Accepts worker registrations (subgraph_id → ip:port mappings)
3. Blocks until all expected workers have registered
4. Broadcasts the complete topology to all workers
5. Monitors heartbeats and detects failures

Workers connect to the coordinator during their setup phase and block
until they receive their topology. Once the topology is received, they
create their ZMQ sockets and begin simulation.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class CoordinatorState(Enum):
    WAITING = "waiting"          # Waiting for workers to register
    READY = "ready"              # All workers registered, topology broadcast
    RUNNING = "running"          # Workers executing
    SHUTTING_DOWN = "shutting_down"
    DONE = "done"


@dataclass
class WorkerInfo:
    """Registration info from a worker."""
    subgraph_id: str
    address: str                 # ip:port
    zmq_ports: dict[str, int]    # {service_name: port}
    registered_at: float = 0.0


@dataclass(frozen=True)
class PeerEndpoint:
    """One entry in the topology for a worker."""
    peer_id: str
    address: str                 # "tcp://ip:port"
    role: str                    # "bind" or "connect"
    socket_type: str             # "PUB", "SUB"
    edge_name: str               # MADDENING edge identifier


@dataclass(frozen=True)
class WorkerTopology:
    """Topology broadcast to a single worker."""
    subgraph_id: str
    peers: list[PeerEndpoint]


class Coordinator:
    """ZMQ coordinator for multi-job rendezvous and topology broadcast.

    Parameters
    ----------
    expected_workers : list[str]
        Subgraph IDs of all expected workers.
    edges : list[dict]
        Inter-job edges: ``{source: str, target: str, source_field: str,
        target_field: str}``.  Source/target are subgraph IDs.
    port : int
        Port to bind the ROUTER socket on.
    heartbeat_interval : float
        Seconds between expected heartbeats.
    heartbeat_timeout : float
        Seconds of missed heartbeats before declaring a worker dead.
    """

    def __init__(
        self,
        expected_workers: list[str],
        edges: list[dict],
        port: int = 5580,
        heartbeat_interval: float = 10.0,
        heartbeat_timeout: float = 30.0,
    ) -> None:
        self._expected = set(expected_workers)
        self._edges = edges
        self._port = port
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._workers: dict[str, WorkerInfo] = {}
        self._state = CoordinatorState.WAITING
        self._lock = threading.Lock()
        self._all_registered = threading.Event()
        self._shutdown = threading.Event()
        self._on_worker_dead: Optional[Callable[[str], None]] = None

    @property
    def state(self) -> CoordinatorState:
        return self._state

    @property
    def registered_workers(self) -> dict[str, WorkerInfo]:
        with self._lock:
            return dict(self._workers)

    def start(self, on_worker_dead: Optional[Callable[[str], None]] = None) -> None:
        """Start the coordinator in a background thread.

        Parameters
        ----------
        on_worker_dead : callable, optional
            Called with subgraph_id when a worker is declared dead.
        """
        self._on_worker_dead = on_worker_dead
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="coordinator",
        )
        self._thread.start()

    def wait_for_all(self, timeout: Optional[float] = None) -> bool:
        """Block until all expected workers have registered.

        Returns True if all registered, False on timeout.
        """
        return self._all_registered.wait(timeout=timeout)

    def shutdown(self) -> None:
        """Signal all workers to shut down and stop the coordinator."""
        self._shutdown.set()
        self._state = CoordinatorState.SHUTTING_DOWN

    def build_topology(self) -> dict[str, WorkerTopology]:
        """Build the topology descriptor for all workers.

        For each inter-job edge (A → B):
        - A gets a PeerEndpoint with role="bind", socket_type="PUB"
        - B gets a PeerEndpoint with role="connect", socket_type="SUB"
        """
        topologies: dict[str, WorkerTopology] = {}

        for wid in self._expected:
            topologies[wid] = WorkerTopology(subgraph_id=wid, peers=[])

        for edge in self._edges:
            src = edge["source"]
            tgt = edge["target"]
            src_field = edge.get("source_field", "")
            tgt_field = edge.get("target_field", "")
            edge_name = f"{src_field}->{tgt_field}"

            src_info = self._workers.get(src)
            tgt_info = self._workers.get(tgt)

            if src_info is None or tgt_info is None:
                continue

            # Source binds PUB, target connects SUB
            # Use the source's first ZMQ port, or default 5555
            src_port = next(iter(src_info.zmq_ports.values()), 5555)
            src_addr = f"tcp://{src_info.address.split(':')[0]}:{src_port}"

            if src in topologies:
                topologies[src].peers.append(PeerEndpoint(
                    peer_id=tgt,
                    address=src_addr,
                    role="bind",
                    socket_type="PUB",
                    edge_name=edge_name,
                ))
            if tgt in topologies:
                topologies[tgt].peers.append(PeerEndpoint(
                    peer_id=src,
                    address=src_addr,
                    role="connect",
                    socket_type="SUB",
                    edge_name=edge_name,
                ))

        return topologies

    def _run(self) -> None:
        """Coordinator main loop (runs in background thread)."""
        try:
            import zmq
        except ImportError:
            logger.error("Coordinator requires pyzmq. pip install maddening[network]")
            return

        ctx = zmq.Context()
        sock = ctx.socket(zmq.ROUTER)
        sock.bind(f"tcp://0.0.0.0:{self._port}")
        sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1s poll timeout
        logger.info("Coordinator listening on port %d, expecting %s",
                     self._port, sorted(self._expected))

        last_heartbeat: dict[str, float] = {}

        while not self._shutdown.is_set():
            try:
                # ROUTER gives us [identity, empty, message]
                frames = sock.recv_multipart()
                if len(frames) < 3:
                    continue
                identity = frames[0]
                msg = json.loads(frames[2])

                msg_type = msg.get("type")

                if msg_type == "register":
                    subgraph_id = msg["subgraph_id"]
                    with self._lock:
                        self._workers[subgraph_id] = WorkerInfo(
                            subgraph_id=subgraph_id,
                            address=msg.get("address", ""),
                            zmq_ports=msg.get("zmq_ports", {}),
                            registered_at=time.monotonic(),
                        )
                        last_heartbeat[subgraph_id] = time.monotonic()
                        logger.info("Worker registered: %s (%d/%d)",
                                    subgraph_id, len(self._workers),
                                    len(self._expected))

                        # Send ACK
                        sock.send_multipart([
                            identity, b"",
                            json.dumps({"type": "ack", "status": "registered"}).encode(),
                        ])

                        # Check if all registered
                        if set(self._workers.keys()) >= self._expected:
                            self._state = CoordinatorState.READY
                            self._all_registered.set()
                            # Broadcast topology to all workers
                            self._broadcast_topology(sock)
                            self._state = CoordinatorState.RUNNING

                elif msg_type == "get_topology":
                    subgraph_id = msg.get("subgraph_id", "")
                    topos = getattr(self, "_topologies", None)
                    if topos and subgraph_id in topos:
                        topo = topos[subgraph_id]
                        sock.send_multipart([
                            identity, b"",
                            json.dumps({
                                "type": "topology",
                                "subgraph_id": subgraph_id,
                                "peers": [
                                    {
                                        "peer_id": p.peer_id,
                                        "address": p.address,
                                        "role": p.role,
                                        "socket_type": p.socket_type,
                                        "edge_name": p.edge_name,
                                    }
                                    for p in topo.peers
                                ],
                            }).encode(),
                        ])
                    else:
                        sock.send_multipart([
                            identity, b"",
                            json.dumps({"type": "not_ready"}).encode(),
                        ])

                elif msg_type == "heartbeat":
                    subgraph_id = msg.get("subgraph_id", "")
                    last_heartbeat[subgraph_id] = time.monotonic()
                    sock.send_multipart([
                        identity, b"",
                        json.dumps({"type": "heartbeat_ack"}).encode(),
                    ])

            except zmq.Again:
                pass  # timeout, check for dead workers
            except Exception:
                logger.debug("Coordinator recv error", exc_info=True)

            # Check for dead workers
            if self._state == CoordinatorState.RUNNING:
                now = time.monotonic()
                for wid, last in list(last_heartbeat.items()):
                    if now - last > self._heartbeat_timeout:
                        logger.warning("Worker %s heartbeat timeout", wid)
                        if self._on_worker_dead:
                            self._on_worker_dead(wid)

        # Send SHUTDOWN to all connected workers
        logger.info("Coordinator shutting down")
        self._state = CoordinatorState.DONE
        sock.close()
        ctx.term()

    def _broadcast_topology(self, sock) -> None:
        """Send topology to all registered workers."""
        topologies = self.build_topology()
        # We don't have worker identities stored by ROUTER socket.
        # Instead, workers will poll for their topology after registration.
        # Store it for retrieval.
        self._topologies = topologies
        logger.info("Topology built for %d workers", len(topologies))
