"""Tests for the ZMQ coordinator."""

import json
import threading
import time

import pytest

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

from maddening.cloud.multigpu.coordinator import (
    Coordinator,
    CoordinatorState,
    WorkerTopology,
)


@pytest.mark.skipif(not HAS_ZMQ, reason="pyzmq not installed")
class TestCoordinator:
    def test_start_and_state(self):
        coord = Coordinator(
            expected_workers=["a", "b"],
            edges=[],
            port=15580,
        )
        coord.start()
        assert coord.state == CoordinatorState.WAITING
        coord.shutdown()

    def test_worker_registration(self):
        coord = Coordinator(
            expected_workers=["a", "b"],
            edges=[],
            port=15581,
        )
        coord.start()
        time.sleep(0.2)  # let coordinator bind

        # Simulate worker registration
        ctx = zmq.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect("tcp://127.0.0.1:15581")

        # Register worker "a"
        sock.send_multipart([
            b"",
            json.dumps({
                "type": "register",
                "subgraph_id": "a",
                "address": "127.0.0.1:5555",
                "zmq_ports": {"state": 5555},
            }).encode(),
        ])
        time.sleep(0.3)

        assert "a" in coord.registered_workers
        assert coord.state == CoordinatorState.WAITING  # still waiting for "b"

        # Register worker "b"
        sock.send_multipart([
            b"",
            json.dumps({
                "type": "register",
                "subgraph_id": "b",
                "address": "127.0.0.1:5556",
                "zmq_ports": {"state": 5556},
            }).encode(),
        ])
        time.sleep(0.3)

        assert "b" in coord.registered_workers
        # Should now be RUNNING (all registered → topology broadcast)
        assert coord.state in (CoordinatorState.READY, CoordinatorState.RUNNING)

        sock.close()
        ctx.term()
        coord.shutdown()

    def test_wait_for_all_timeout(self):
        coord = Coordinator(
            expected_workers=["a", "b"],
            edges=[],
            port=15582,
        )
        coord.start()
        result = coord.wait_for_all(timeout=0.5)
        assert result is False  # No workers registered
        coord.shutdown()

    def test_wait_for_all_success(self):
        coord = Coordinator(
            expected_workers=["a"],
            edges=[],
            port=15583,
        )
        coord.start()
        time.sleep(0.2)

        # Register in background
        def register():
            ctx = zmq.Context()
            sock = ctx.socket(zmq.DEALER)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect("tcp://127.0.0.1:15583")
            sock.send_multipart([
                b"",
                json.dumps({
                    "type": "register",
                    "subgraph_id": "a",
                    "address": "127.0.0.1:5555",
                    "zmq_ports": {},
                }).encode(),
            ])
            time.sleep(0.2)
            sock.close()
            ctx.term()

        threading.Thread(target=register, daemon=True).start()
        result = coord.wait_for_all(timeout=5.0)
        assert result is True
        coord.shutdown()

    def test_topology_building(self):
        coord = Coordinator(
            expected_workers=["flow", "structure"],
            edges=[{
                "source": "flow",
                "target": "structure",
                "source_field": "pressure",
                "target_field": "surface_load",
            }],
            port=15584,
        )
        coord.start()
        time.sleep(0.2)

        ctx = zmq.Context()

        # Register both workers
        for wid, addr, port in [("flow", "10.0.0.1", 5555),
                                 ("structure", "10.0.0.2", 5556)]:
            sock = ctx.socket(zmq.DEALER)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect("tcp://127.0.0.1:15584")
            sock.send_multipart([
                b"",
                json.dumps({
                    "type": "register",
                    "subgraph_id": wid,
                    "address": f"{addr}:{port}",
                    "zmq_ports": {"state": port},
                }).encode(),
            ])
            time.sleep(0.2)
            sock.close()

        coord.wait_for_all(timeout=5.0)

        # Check topology
        topos = coord.build_topology()
        assert "flow" in topos
        assert "structure" in topos

        # Flow should have a PUB bind entry
        flow_topo = topos["flow"]
        assert len(flow_topo.peers) == 1
        assert flow_topo.peers[0].role == "bind"
        assert flow_topo.peers[0].socket_type == "PUB"
        assert flow_topo.peers[0].peer_id == "structure"

        # Structure should have a SUB connect entry
        struct_topo = topos["structure"]
        assert len(struct_topo.peers) == 1
        assert struct_topo.peers[0].role == "connect"
        assert struct_topo.peers[0].socket_type == "SUB"
        assert struct_topo.peers[0].peer_id == "flow"

        ctx.term()
        coord.shutdown()


@pytest.mark.skipif(not HAS_ZMQ, reason="pyzmq not installed")
class TestCoordinatorTopology:
    def test_bidirectional_edges(self):
        """Bidirectional coupling: both sides get bind + connect entries."""
        coord = Coordinator(
            expected_workers=["a", "b"],
            edges=[
                {"source": "a", "target": "b", "source_field": "x", "target_field": "y"},
                {"source": "b", "target": "a", "source_field": "y", "target_field": "x"},
            ],
            port=15585,
        )
        coord.start()
        time.sleep(0.2)

        ctx = zmq.Context()
        for wid, addr, port in [("a", "10.0.0.1", 5555), ("b", "10.0.0.2", 5556)]:
            sock = ctx.socket(zmq.DEALER)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect("tcp://127.0.0.1:15585")
            sock.send_multipart([
                b"",
                json.dumps({
                    "type": "register",
                    "subgraph_id": wid,
                    "address": f"{addr}:{port}",
                    "zmq_ports": {"state": port},
                }).encode(),
            ])
            time.sleep(0.2)
            sock.close()

        coord.wait_for_all(timeout=5.0)
        topos = coord.build_topology()

        # Each worker should have 2 peers (one bind, one connect)
        assert len(topos["a"].peers) == 2
        assert len(topos["b"].peers) == 2

        ctx.term()
        coord.shutdown()
