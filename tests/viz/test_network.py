"""Tests for ZMQ network transport (NetworkRelay, NetworkReceiver, CommandPublisher, CommandReceiver).

These tests use loopback connections and short timeouts.
"""

import json
import time
import threading

import pytest
import zmq

from maddening.viz.network import (
    NetworkRelay,
    NetworkReceiver,
    CommandPublisher,
    CommandReceiver,
)
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


# Use unique ports to avoid conflicts between parallel test runs
_STATE_PORT = 15555
_CMD_PORT = 15556


@pytest.fixture
def state_ports():
    """Return (bind_addr, connect_addr) for state channel."""
    p = _STATE_PORT
    return f"tcp://127.0.0.1:{p}", f"tcp://127.0.0.1:{p}"


@pytest.fixture
def cmd_ports():
    """Return (bind_addr, connect_addr) for command channel."""
    p = _CMD_PORT
    return f"tcp://127.0.0.1:{p}", f"tcp://127.0.0.1:{p}"


class TestNetworkRelay:
    def test_publish_receives(self, bouncing_ball_graph, state_ports):
        bind_addr, connect_addr = state_ports
        relay = NetworkRelay(address=bind_addr)
        relay.attach(bouncing_ball_graph)

        receiver = NetworkReceiver(address=connect_addr)
        receiver.start()

        # Give ZMQ time to connect
        time.sleep(0.3)

        # Step the simulation to generate state events
        for _ in range(5):
            bouncing_ball_graph.step()
            time.sleep(0.05)

        time.sleep(0.2)
        sim_time, snap = receiver.latest_snapshot()

        receiver.stop()
        relay.close()

        assert sim_time > 0.0
        assert snap is not None
        assert "ball" in snap

    def test_receiver_initial_state_is_none(self, state_ports):
        _, connect_addr = state_ports
        # Don't bind -- just test initial state
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUB)
        sock.bind(state_ports[0])

        receiver = NetworkReceiver(address=connect_addr)
        sim_time, snap = receiver.latest_snapshot()
        assert sim_time == 0.0
        assert snap is None

        sock.close()
        ctx.term()


class TestNetworkRelayFieldFilter:
    """v0.2 #5: NetworkRelay should publish only subscribed fields when a
    field selector is supplied at construction."""

    def test_filter_to_single_field(self, bouncing_ball_graph):
        bind_addr = f"tcp://127.0.0.1:{_STATE_PORT + 10}"
        relay = NetworkRelay(address=bind_addr, fields={"ball": ["position"]})
        relay.attach(bouncing_ball_graph)
        assert relay.subscription == {"ball": ["position"]}

        receiver = NetworkReceiver(address=bind_addr)
        receiver.start()
        time.sleep(0.3)

        for _ in range(5):
            bouncing_ball_graph.step()
            time.sleep(0.05)
        time.sleep(0.2)
        _, snap = receiver.latest_snapshot()
        receiver.stop()
        relay.close()

        assert snap is not None
        assert "ball" in snap
        # Only position should be in the published frame; velocity dropped.
        assert set(snap["ball"].keys()) == {"position"}

    def test_filter_unknown_field_silently_dropped(self, bouncing_ball_graph):
        bind_addr = f"tcp://127.0.0.1:{_STATE_PORT + 11}"
        relay = NetworkRelay(
            address=bind_addr,
            fields={"ball": ["position", "nonexistent"], "ghost_node": ["x"]},
        )
        relay.attach(bouncing_ball_graph)
        receiver = NetworkReceiver(address=bind_addr)
        receiver.start()
        time.sleep(0.3)

        for _ in range(3):
            bouncing_ball_graph.step()
            time.sleep(0.05)
        time.sleep(0.2)
        _, snap = receiver.latest_snapshot()
        receiver.stop()
        relay.close()

        assert snap is not None
        assert set(snap.keys()) == {"ball"}
        assert set(snap["ball"].keys()) == {"position"}

    def test_default_no_filter_publishes_all(self, bouncing_ball_graph):
        bind_addr = f"tcp://127.0.0.1:{_STATE_PORT + 12}"
        relay = NetworkRelay(address=bind_addr)
        relay.attach(bouncing_ball_graph)
        assert relay.subscription is None

        receiver = NetworkReceiver(address=bind_addr)
        receiver.start()
        time.sleep(0.3)

        for _ in range(3):
            bouncing_ball_graph.step()
            time.sleep(0.05)
        time.sleep(0.2)
        _, snap = receiver.latest_snapshot()
        receiver.stop()
        relay.close()

        assert snap is not None
        # The bouncing ball fixture has both position and velocity
        assert "ball" in snap
        assert "position" in snap["ball"]
        assert "velocity" in snap["ball"]


class TestCommandChannel:
    def test_send_receive(self, cmd_ports):
        bind_addr, connect_addr = cmd_ports
        pub = CommandPublisher(address=bind_addr)
        recv = CommandReceiver(address=connect_addr)
        recv.start()

        # Give ZMQ time to connect
        time.sleep(0.3)

        cmd = {"robot": {"joint_torque": 1.5}}
        for _ in range(5):
            pub.send(cmd)
            time.sleep(0.05)

        time.sleep(0.2)
        result = recv.latest_commands()

        recv.stop()
        pub.close()

        assert result is not None
        assert result["robot"]["joint_torque"] == 1.5

    def test_initial_commands_are_none(self):
        # Without connecting to anything, commands should be None
        recv = CommandReceiver.__new__(CommandReceiver)
        recv._commands = None
        recv._lock = threading.Lock()
        assert recv.latest_commands() is None
