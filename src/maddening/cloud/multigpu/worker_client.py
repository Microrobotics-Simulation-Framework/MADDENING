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
    secure : bool, optional
        Whether to talk to the coordinator over ZMQ CURVE.  ``None``
        (default) decides from *coordinator_addr*: a loopback
        coordinator is contacted in cleartext, a remote one is
        encrypted and authenticated.  ``True`` forces it on.

        **Pass ``True`` explicitly when the coordinator binds a
        non-loopback address but you reach it over a loopback one** --
        rank 0's own worker talking to ``127.0.0.1:5580``, or any worker
        going through an SSH tunnel.  The coordinator's posture is set
        by *its* bind address, which this client cannot see, so the
        address rule gets that case wrong.  It fails loudly rather than
        silently: ``register_and_wait`` raises ``ConnectionError``
        naming the mismatch.
    token : str, optional
        Shared secret the CURVE keys are derived from; ``None`` reads
        ``MADDENING_TRANSPORT_TOKEN``, falling back to
        ``MADDENING_API_TOKEN``.  Must match the coordinator's.
    """

    def __init__(
        self,
        coordinator_addr: str,
        subgraph_id: str,
        address: str,
        zmq_ports: Optional[dict[str, int]] = None,
        secure: Optional[bool] = None,
        token: Optional[str] = None,
    ) -> None:
        from maddening.transport_auth import resolve_security

        self._secure = resolve_security(f"tcp://{coordinator_addr}", secure)
        self._token = token
        if self._secure:
            from maddening.transport_auth import TransportAuth

            TransportAuth(token=token)  # raises now if the token is missing
        self._coordinator_addr = coordinator_addr
        self._subgraph_id = subgraph_id
        self._address = address
        self._zmq_ports = zmq_ports or {}
        self._topology: Optional[list[PeerConnection]] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._on_shutdown: Optional[Callable[[], None]] = None
        self._on_peer_dead: Optional[Callable[[str], None]] = None

    def _secure_socket(self, sock) -> None:
        """Apply CURVE client keys to *sock* before it connects."""
        if not self._secure:
            return
        from maddening.transport_auth import TransportAuth

        TransportAuth(token=self._token).secure_client(sock)

    @property
    def secure(self) -> bool:
        """Whether this client talks to the coordinator over CURVE."""
        return self._secure

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
        # Short, because this is the loop's poll interval: the deadline
        # below is only checked between recv calls, so a 5s RCVTIMEO made
        # register_and_wait(timeout=3) take 5s and overshoot every
        # deadline shorter than itself.
        sock.setsockopt(zmq.RCVTIMEO, 250)
        self._secure_socket(sock)
        # A DEALER whose peer rejects the security handshake enters mute
        # state, and an unbounded send on a mute socket blocks forever --
        # measured, not assumed: a plain DEALER sending to a CURVE ROUTER
        # never returns from the first send_multipart.  That would make
        # `timeout` a lie and hang the worker on the single most likely
        # misconfiguration, a token that does not match the coordinator's.
        sock.setsockopt(zmq.SNDTIMEO, 1000)
        sock.connect(f"tcp://{self._coordinator_addr}")

        # Send registration
        reg_msg = {
            "type": "register",
            "subgraph_id": self._subgraph_id,
            "address": self._address,
            "zmq_ports": self._zmq_ports,
        }
        payload = [b"", json.dumps(reg_msg).encode()]

        # Wait for ACK, re-sending as we go so a coordinator that starts
        # after this worker still gets the registration.
        deadline = time.monotonic() + timeout
        ack_received = False
        deliverable = False
        while time.monotonic() < deadline:
            try:
                sock.send_multipart(payload)
                deliverable = True
            except zmq.Again:
                # No peer will take it. Keep trying until the deadline:
                # the coordinator may still be coming up.
                pass
            try:
                frames = sock.recv_multipart()
                msg = json.loads(frames[-1])
                if msg.get("type") == "ack":
                    ack_received = True
                    logger.info("Registration ACK received")
                    break
            except zmq.Again:
                continue

        if not ack_received and not deliverable:
            sock.close()
            ctx.term()
            raise ConnectionError(
                f"Could not deliver a registration to the coordinator at "
                f"{self._coordinator_addr}: it accepted no message in "
                f"{timeout:.0f}s. This worker has CURVE "
                f"{'on' if self._secure else 'off'}. A coordinator bound to a "
                f"non-loopback address has CURVE on, and a worker reaching it "
                f"over a loopback address (a tunnel, or rank 0's own worker) "
                f"turns it off by default -- pass secure=True to WorkerClient "
                f"in that case, and give both ends the same "
                f"MADDENING_API_TOKEN."
            )

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
        # See register_and_wait: an unbounded send on a mute DEALER never
        # returns, which would wedge this daemon thread past stop().
        sock.setsockopt(zmq.SNDTIMEO, 1000)
        self._secure_socket(sock)
        sock.connect(f"tcp://{self._coordinator_addr}")

        while not self._stop_event.is_set():
            try:
                try:
                    sock.send_multipart([
                        b"",
                        json.dumps({
                            "type": "heartbeat",
                            "subgraph_id": self._subgraph_id,
                        }).encode(),
                    ])
                except zmq.Again:
                    logger.debug("Heartbeat undeliverable to %s",
                                 self._coordinator_addr)
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
