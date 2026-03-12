"""Tests for render stride / decoupled physics-render rates."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time
import threading

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.viz.relay import StateRelay
from maddening.viz.runner import RealtimeRunner


@pytest.fixture
def simple_graph():
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=5.0,
                          initial_velocity=0.0, elasticity=0.7))
    gm.add_node(TableNode("table", timestep=0.01, position=0.0))
    gm.add_edge(source="table", target="ball",
                source_field="position", target_field="table_position")
    gm.compile()
    return gm


class TestRelayStride:
    """StateRelay with stride parameter."""

    def test_default_stride_is_one(self):
        relay = StateRelay()
        assert relay.stride == 1

    def test_custom_stride(self):
        relay = StateRelay(stride=5)
        assert relay.stride == 5

    def test_stride_setter(self):
        relay = StateRelay()
        relay.stride = 10
        assert relay.stride == 10

    def test_stride_clamped_to_one(self):
        relay = StateRelay(stride=0)
        assert relay.stride == 1
        relay.stride = -5
        assert relay.stride == 1

    def test_stride_captures_every_nth(self, simple_graph):
        relay = StateRelay(stride=3)
        relay.attach(simple_graph)

        # Step 5 times
        for _ in range(5):
            simple_graph.step()

        # With stride=3, should capture at step 3 (not 1, 2, 4, 5 unless 6)
        sim_time, snapshot = relay.latest_snapshot()
        assert snapshot is not None
        # Step count 3 → sim_time = 3 * 0.01 = 0.03
        assert sim_time == pytest.approx(0.03, abs=1e-6)

    def test_stride_one_captures_all(self, simple_graph):
        relay = StateRelay(stride=1)
        relay.attach(simple_graph)

        for _ in range(3):
            simple_graph.step()

        sim_time, snapshot = relay.latest_snapshot()
        assert snapshot is not None
        assert sim_time == pytest.approx(0.03, abs=1e-6)

    def test_stride_larger_than_steps(self, simple_graph):
        relay = StateRelay(stride=10)
        relay.attach(simple_graph)

        # Step 5 times — never hits stride
        for _ in range(5):
            simple_graph.step()

        _, snapshot = relay.latest_snapshot()
        assert snapshot is None  # no capture yet

    def test_stride_change_during_sim(self, simple_graph):
        relay = StateRelay(stride=100)
        relay.attach(simple_graph)

        for _ in range(5):
            simple_graph.step()
        _, snap1 = relay.latest_snapshot()
        assert snap1 is None

        # Change stride to 1
        relay.stride = 1
        simple_graph.step()  # step 6
        _, snap2 = relay.latest_snapshot()
        assert snap2 is not None


class TestRunnerStepsPerFrame:
    """RealtimeRunner with steps_per_frame parameter."""

    def test_default_steps_per_frame(self, simple_graph):
        relay = StateRelay()
        relay.attach(simple_graph)
        runner = RealtimeRunner(simple_graph, relay)
        assert runner.steps_per_frame == 1

    def test_custom_steps_per_frame(self, simple_graph):
        relay = StateRelay()
        relay.attach(simple_graph)
        runner = RealtimeRunner(simple_graph, relay, steps_per_frame=10)
        assert runner.steps_per_frame == 10

    def test_steps_per_frame_setter(self, simple_graph):
        relay = StateRelay()
        relay.attach(simple_graph)
        runner = RealtimeRunner(simple_graph, relay)
        runner.steps_per_frame = 20
        assert runner.steps_per_frame == 20

    def test_steps_per_frame_clamped(self, simple_graph):
        relay = StateRelay()
        relay.attach(simple_graph)
        runner = RealtimeRunner(simple_graph, relay, steps_per_frame=0)
        assert runner.steps_per_frame == 1

    def test_batch_stepping(self, simple_graph):
        """Runner with steps_per_frame>1 advances sim time faster."""
        relay = StateRelay()
        relay.attach(simple_graph)
        runner = RealtimeRunner(
            simple_graph, relay,
            steps_per_frame=5,
            time_scale=1000.0,  # run fast
        )
        runner.start()
        time.sleep(0.15)
        runner.stop()

        # Should have advanced by multiples of 5 steps
        sim_time = runner.sim_time
        assert sim_time > 0.0
        # sim_time should be a multiple of 5 * dt = 0.05
        n_frames = round(sim_time / (5 * 0.01))
        assert n_frames >= 1
