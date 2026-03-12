"""Tests for visualization infrastructure (relay, runner, renderer ABC)."""

import threading
import time

import pytest
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.viz.relay import StateRelay
from maddening.viz.renderer import Renderer, GraphInfo
from maddening.viz.runner import RealtimeRunner


# ------------------------------------------------------------------
# GraphInfo
# ------------------------------------------------------------------

class TestGraphInfo:
    def test_from_graph_manager(self, bouncing_ball_graph):
        info = GraphInfo.from_graph_manager(bouncing_ball_graph)
        assert set(info.node_names) == {"ball", "table"}
        assert info.timestep == 0.01
        assert "position" in info.node_state_fields["ball"]
        assert "velocity" in info.node_state_fields["ball"]
        assert "position" in info.node_state_fields["table"]
        assert len(info.edges) == 1

    def test_node_params(self, bouncing_ball_graph):
        info = GraphInfo.from_graph_manager(bouncing_ball_graph)
        assert info.node_params["ball"]["elasticity"] == 0.7
        assert info.node_params["table"]["position"] == 0.0


# ------------------------------------------------------------------
# Renderer ABC
# ------------------------------------------------------------------

class TestRendererABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Renderer()

    def test_concrete_renderer(self):
        class DummyRenderer(Renderer):
            def __init__(self):
                self.setup_called = False
                self.updates = []

            def setup(self, graph_info):
                self.setup_called = True

            def update(self, sim_time, state):
                self.updates.append((sim_time, state))

            def teardown(self):
                pass

        r = DummyRenderer()
        r.setup(None)
        assert r.setup_called
        assert r.requested_fields() is None  # default


# ------------------------------------------------------------------
# StateRelay
# ------------------------------------------------------------------

class TestStateRelay:
    def test_initial_snapshot_is_none(self):
        relay = StateRelay()
        sim_time, snap = relay.latest_snapshot()
        assert sim_time == 0.0
        assert snap is None

    def test_receives_step_events(self, bouncing_ball_graph):
        relay = StateRelay()
        relay.attach(bouncing_ball_graph)
        bouncing_ball_graph.step()
        sim_time, snap = relay.latest_snapshot()
        assert sim_time > 0.0
        assert snap is not None
        assert "ball" in snap

    def test_snapshot_updates_on_each_step(self, bouncing_ball_graph):
        relay = StateRelay()
        relay.attach(bouncing_ball_graph)

        bouncing_ball_graph.step()
        t1, _ = relay.latest_snapshot()

        bouncing_ball_graph.step()
        t2, _ = relay.latest_snapshot()

        assert t2 > t1

    def test_thread_safety(self, bouncing_ball_graph):
        """Concurrent reads from relay shouldn't crash."""
        relay = StateRelay()
        relay.attach(bouncing_ball_graph)

        errors = []

        def reader():
            for _ in range(100):
                try:
                    t, s = relay.latest_snapshot()
                except Exception as e:
                    errors.append(e)

        t = threading.Thread(target=reader)
        t.start()
        bouncing_ball_graph.run(100)
        t.join()
        assert len(errors) == 0


# ------------------------------------------------------------------
# RealtimeRunner
# ------------------------------------------------------------------

class TestRealtimeRunner:
    def test_start_stop(self, bouncing_ball_graph):
        relay = StateRelay()
        relay.attach(bouncing_ball_graph)
        runner = RealtimeRunner(bouncing_ball_graph, relay, time_scale=100.0)
        runner.start()
        time.sleep(0.2)
        runner.stop()
        # Some steps should have been taken
        t, snap = relay.latest_snapshot()
        assert t > 0.0
        assert snap is not None

    def test_pause_resume(self, bouncing_ball_graph):
        relay = StateRelay()
        relay.attach(bouncing_ball_graph)
        runner = RealtimeRunner(bouncing_ball_graph, relay, time_scale=100.0)
        runner.start()
        time.sleep(0.1)

        runner.pause()
        time.sleep(0.05)
        t_paused, _ = relay.latest_snapshot()

        # Give it a moment to confirm paused state
        time.sleep(0.1)
        t_still, _ = relay.latest_snapshot()
        # Should not have advanced much while paused
        # (allow small tolerance for steps in flight)
        assert abs(t_still - t_paused) < 0.02

        runner.resume()
        time.sleep(0.1)
        t_resumed, _ = relay.latest_snapshot()
        assert t_resumed > t_paused

        runner.stop()

    def test_time_scale_property(self, bouncing_ball_graph):
        relay = StateRelay()
        relay.attach(bouncing_ball_graph)
        runner = RealtimeRunner(bouncing_ball_graph, relay, time_scale=1.0)
        assert runner.time_scale == 1.0
        runner.time_scale = 2.0
        assert runner.time_scale == 2.0
        # Minimum clamp
        runner.time_scale = 0.001
        assert runner.time_scale == 0.01

    def test_sim_time_advances(self, bouncing_ball_graph):
        relay = StateRelay()
        relay.attach(bouncing_ball_graph)
        runner = RealtimeRunner(bouncing_ball_graph, relay, time_scale=100.0)
        runner.start()
        time.sleep(0.2)
        runner.stop()
        assert runner.sim_time > 0.0
