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
