"""Worker-side rendezvous client for multi-job distributed simulation.

Each worker process uses ``WorkerClient`` to:
1. Connect to the coordinator's ROUTER socket
2. Register with its subgraph_id, IP, and ZMQ ports
3. Wait for the topology broadcast
4. Send periodic heartbeats during execution
5. Listen for SHUTDOWN or PEER_DEAD signals

Usage::

    client = WorkerClient(
        coordinator_addr="10.0.0.1:5580",
        subgraph_id="flow",
        address="10.0.0.2:5555",
        zmq_ports={"state_pub": 5555, "cmd_sub": 5556},
    )
    topology = client.register_and_wait(timeout=300)
    # topology is a dict with peer connection info
    client.start_heartbeat()
    # ... run simulation ...
    client.stop()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeerConnection:
    """Resolved peer connection info for this worker."""
    peer_id: str
    address: str        # "tcp://ip:port"
    role: str           # "bind" or "connect"
    socket_type: str    # "PUB", "SUB"
    edge_name: str


class WorkerClient:
    """Client-side rendezvous for a single worker process.

    Parameters
    ----------
    coordinator_addr : str
        ``"ip:port"`` of the coordinator's ROUTER socket.
    subgraph_id : str
        This worker's unique subgraph identifier.
    address : str
        This worker's public ``"ip:port"`` for peer connections.
    zmq_ports : dict[str, int]
        ZMQ ports this worker exposes: ``{service_name: port}``.
    """

    def __init__(
        self,
        coordinator_addr: str,
        subgraph_id: str,
        address: str,
        zmq_ports: Optional[dict[str, int]] = None,
    ) -> None:
        self._coordinator_addr = coordinator_addr
        self._subgraph_id = subgraph_id
        self._address = address
        self._zmq_ports = zmq_ports or {}
        self._topology: Optional[list[PeerConnection]] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._on_shutdown: Optional[Callable[[], None]] = None
        self._on_peer_dead: Optional[Callable[[str], None]] = None

    @property
    def topology(self) -> Optional[list[PeerConnection]]:
        """Peer connections assigned to this worker, or None if not yet received."""
        return self._topology

    def register_and_wait(self, timeout: float = 300.0) -> list[PeerConnection]:
        """Register with the coordinator and wait for topology.

        Blocks until the topology is received or *timeout* expires.

        Returns
        -------
        list[PeerConnection]
            Peer connections for this worker.

        Raises
        ------
        TimeoutError
            If the coordinator doesn't respond within *timeout*.
        ConnectionError
            If the coordinator is unreachable.
        """
        try:
            import zmq
        except ImportError:
            raise ImportError(
                "WorkerClient requires pyzmq. "
                "Install with:  pip install maddening[network]"
            )

        ctx = zmq.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 5000)  # 5s recv timeout
        sock.connect(f"tcp://{self._coordinator_addr}")

        # Send registration
        reg_msg = {
            "type": "register",
            "subgraph_id": self._subgraph_id,
            "address": self._address,
            "zmq_ports": self._zmq_ports,
        }
        sock.send_multipart([b"", json.dumps(reg_msg).encode()])
        logger.info("Registered with coordinator at %s as %s",
                     self._coordinator_addr, self._subgraph_id)

        # Wait for ACK
        deadline = time.monotonic() + timeout
        ack_received = False
        while time.monotonic() < deadline:
            try:
                frames = sock.recv_multipart()
                msg = json.loads(frames[-1])
                if msg.get("type") == "ack":
                    ack_received = True
                    logger.info("Registration ACK received")
                    break
            except zmq.Again:
                continue

        if not ack_received:
            sock.close()
            ctx.term()
            raise TimeoutError(
                f"No ACK from coordinator at {self._coordinator_addr} "
                f"within {timeout}s"
            )

        # Now poll for topology broadcast
        # The coordinator stores topology after all workers register.
        # We poll by sending a "get_topology" request.
        while time.monotonic() < deadline:
            sock.send_multipart([
                b"",
                json.dumps({
                    "type": "get_topology",
                    "subgraph_id": self._subgraph_id,
                }).encode(),
            ])
            try:
                frames = sock.recv_multipart()
                msg = json.loads(frames[-1])
                if msg.get("type") == "topology":
                    peers = [
                        PeerConnection(
                            peer_id=p["peer_id"],
                            address=p["address"],
                            role=p["role"],
                            socket_type=p["socket_type"],
                            edge_name=p["edge_name"],
                        )
                        for p in msg.get("peers", [])
                    ]
                    self._topology = peers
                    logger.info("Topology received: %d peers", len(peers))
                    sock.close()
                    ctx.term()
                    return peers
                elif msg.get("type") == "not_ready":
                    time.sleep(1)
                    continue
            except zmq.Again:
                time.sleep(1)
                continue

        sock.close()
        ctx.term()
        raise TimeoutError(
            f"Topology not received within {timeout}s"
        )

    def start_heartbeat(
        self,
        interval: float = 10.0,
        on_shutdown: Optional[Callable[[], None]] = None,
        on_peer_dead: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Start sending periodic heartbeats to the coordinator.

        Parameters
        ----------
        interval : float
            Seconds between heartbeats.
        on_shutdown : callable, optional
            Called when the coordinator sends a SHUTDOWN signal.
        on_peer_dead : callable, optional
            Called with ``peer_id`` when the coordinator reports a dead peer.
        """
        self._on_shutdown = on_shutdown
        self._on_peer_dead = on_peer_dead
        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(interval,),
            daemon=True,
            name=f"heartbeat-{self._subgraph_id}",
        )
        self._heartbeat_thread.start()

    def stop(self) -> None:
        """Stop heartbeating."""
        self._stop_event.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)

    def _heartbeat_loop(self, interval: float) -> None:
        """Background heartbeat sender."""
        try:
            import zmq
        except ImportError:
            return

        ctx = zmq.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.connect(f"tcp://{self._coordinator_addr}")

        while not self._stop_event.is_set():
            try:
                sock.send_multipart([
                    b"",
                    json.dumps({
                        "type": "heartbeat",
                        "subgraph_id": self._subgraph_id,
                    }).encode(),
                ])
                try:
                    frames = sock.recv_multipart()
                    msg = json.loads(frames[-1])
                    if msg.get("type") == "shutdown":
                        logger.warning("SHUTDOWN received from coordinator")
                        if self._on_shutdown:
                            self._on_shutdown()
                        return
                    elif msg.get("type") == "peer_dead":
                        peer_id = msg.get("peer_id", "")
                        logger.warning("Peer %s declared dead", peer_id)
                        if self._on_peer_dead:
                            self._on_peer_dead(peer_id)
                except zmq.Again:
                    pass
            except Exception:
                logger.debug("Heartbeat send failed", exc_info=True)

            self._stop_event.wait(interval)

        sock.close()
        ctx.term()
