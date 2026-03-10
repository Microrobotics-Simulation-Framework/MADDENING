"""Tests for HistoryLogger -- observer-based state recording."""

import pytest
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.core.history_logger import HistoryLogger
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_bouncing_ball_graph():
    """Create a fresh bouncing-ball graph (table -> ball)."""
    gm = GraphManager()
    ball = BallNode(name="ball", timestep=0.01,
                    initial_position=5.0, initial_velocity=0.0,
                    elasticity=0.7)
    table = TableNode(name="table", timestep=0.01, position=0.0)
    gm.add_node(table)
    gm.add_node(ball)
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    return gm


# ------------------------------------------------------------------
# Basic recording
# ------------------------------------------------------------------

class TestBasicRecording:
    def test_records_correct_number_of_steps(self):
        gm = _make_bouncing_ball_graph()
        logger = HistoryLogger()
        gm.add_observer(logger)
        n_steps = 50
        gm.run(n_steps)
        history = logger.history
        assert history["ball"]["position"].shape == (n_steps,)
        assert history["ball"]["velocity"].shape == (n_steps,)
        assert history["table"]["position"].shape == (n_steps,)

    def test_len_matches_steps(self):
        gm = _make_bouncing_ball_graph()
        logger = HistoryLogger()
        gm.add_observer(logger)
        gm.run(30)
        assert len(logger) == 30

    def test_records_from_step_method(self):
        gm = _make_bouncing_ball_graph()
        logger = HistoryLogger()
        gm.add_observer(logger)
        gm.step()
        gm.step()
        gm.step()
        assert len(logger) == 3

    def test_history_contains_all_nodes(self):
        gm = _make_bouncing_ball_graph()
        logger = HistoryLogger()
        gm.add_observer(logger)
        gm.run(10)
        history = logger.history
        assert "ball" in history
        assert "table" in history

    def test_history_contains_all_fields(self):
        gm = _make_bouncing_ball_graph()
        logger = HistoryLogger()
        gm.add_observer(logger)
        gm.run(10)
        history = logger.history
        assert "position" in history["ball"]
        assert "velocity" in history["ball"]
        assert "position" in history["table"]


# ------------------------------------------------------------------
# Values match run_scan_with_history
# ------------------------------------------------------------------

class TestValuesMatch:
    def test_matches_scan_history(self):
        """HistoryLogger output should match run_scan_with_history."""
        n_steps = 200

        # Observer-based recording
        gm_obs = _make_bouncing_ball_graph()
        logger = HistoryLogger()
        gm_obs.add_observer(logger)
        gm_obs.run(n_steps)
        obs_history = logger.history

        # Scan-based recording
        gm_scan = _make_bouncing_ball_graph()
        _, scan_history = gm_scan.run_scan_with_history(n_steps)

        for node_name in scan_history:
            assert node_name in obs_history, (
                f"Node '{node_name}' missing from observer history"
            )
            for field_name in scan_history[node_name]:
                assert jnp.allclose(
                    obs_history[node_name][field_name],
                    scan_history[node_name][field_name],
                    atol=1e-5,
                ), (
                    f"Mismatch in {node_name}.{field_name}"
                )

    def test_last_entry_matches_final_state(self):
        gm = _make_bouncing_ball_graph()
        logger = HistoryLogger()
        gm.add_observer(logger)
        gm.run(100)

        history = logger.history
        final_state_ball = gm.get_node_state("ball")
        assert jnp.allclose(
            history["ball"]["position"][-1],
            final_state_ball["position"],
            atol=1e-6,
        )
        assert jnp.allclose(
            history["ball"]["velocity"][-1],
            final_state_ball["velocity"],
            atol=1e-6,
        )


# ------------------------------------------------------------------
# Field filtering
# ------------------------------------------------------------------

class TestFieldFiltering:
    def test_filter_single_node(self):
        gm = _make_bouncing_ball_graph()
        logger = HistoryLogger(fields={"ball": ["position"]})
        gm.add_observer(logger)
        gm.run(20)
        history = logger.history
        assert "ball" in history
        assert "position" in history["ball"]
        assert "velocity" not in history["ball"]
        assert "table" not in history

    def test_filter_multiple_fields(self):
        gm = _make_bouncing_ball_graph()
        logger = HistoryLogger(fields={"ball": ["position", "velocity"]})
        gm.add_observer(logger)
        gm.run(20)
        history = logger.history
        assert "position" in history["ball"]
        assert "velocity" in history["ball"]
        assert "table" not in history

    def test_filter_multiple_nodes(self):
        gm = _make_bouncing_ball_graph()
        logger = HistoryLogger(fields={
            "ball": ["position"],
            "table": ["position"],
        })
        gm.add_observer(logger)
        gm.run(20)
        history = logger.history
        assert "ball" in history
        assert "table" in history
        assert "position" in history["ball"]
        assert "velocity" not in history["ball"]
        assert history["ball"]["position"].shape == (20,)
        assert history["table"]["position"].shape == (20,)


# ------------------------------------------------------------------
# Reset and reuse
# ------------------------------------------------------------------

class TestResetReuse:
    def test_reset_clears_history(self):
        gm = _make_bouncing_ball_graph()
        logger = HistoryLogger()
        gm.add_observer(logger)
        gm.run(10)
        assert len(logger) == 10

        logger.reset()
        assert len(logger) == 0
        assert logger.history == {}

    def test_record_after_reset(self):
        gm = _make_bouncing_ball_graph()
        logger = HistoryLogger()
        gm.add_observer(logger)

        # First recording
        gm.run(10)
        assert len(logger) == 10

        # Reset and record again
        logger.reset()
        gm.run(20)
        assert len(logger) == 20

        history = logger.history
        assert history["ball"]["position"].shape == (20,)

    def test_reset_preserves_field_filter(self):
        gm = _make_bouncing_ball_graph()
        logger = HistoryLogger(fields={"ball": ["position"]})
        gm.add_observer(logger)

        gm.run(10)
        logger.reset()
        gm.run(5)

        history = logger.history
        assert "ball" in history
        assert "velocity" not in history["ball"]
        assert "table" not in history
        assert history["ball"]["position"].shape == (5,)


# ------------------------------------------------------------------
# Multi-node graph recording
# ------------------------------------------------------------------

class TestMultiNodeGraph:
    def test_two_balls_one_table(self):
        """Two ball nodes coupled to the same table."""
        gm = GraphManager()
        table = TableNode(name="table", timestep=0.01, position=0.0)
        ball_a = BallNode(name="ball_a", timestep=0.01,
                          initial_position=5.0, elasticity=0.7)
        ball_b = BallNode(name="ball_b", timestep=0.01,
                          initial_position=10.0, elasticity=0.9)
        gm.add_node(table)
        gm.add_node(ball_a)
        gm.add_node(ball_b)
        gm.add_edge("table", "ball_a", "position", "table_position")
        gm.add_edge("table", "ball_b", "position", "table_position")
        gm.compile()

        logger = HistoryLogger()
        gm.add_observer(logger)
        n_steps = 50
        gm.run(n_steps)

        history = logger.history
        assert "table" in history
        assert "ball_a" in history
        assert "ball_b" in history
        assert history["ball_a"]["position"].shape == (n_steps,)
        assert history["ball_b"]["position"].shape == (n_steps,)
        assert history["table"]["position"].shape == (n_steps,)

        # ball_b starts higher, so its first position should be higher
        assert float(history["ball_b"]["position"][0]) > float(
            history["ball_a"]["position"][0]
        )

    def test_multi_node_values_match_scan(self):
        """Multi-node observer history matches scan history."""
        def make():
            gm = GraphManager()
            table = TableNode(name="table", timestep=0.01, position=0.0)
            ball_a = BallNode(name="ball_a", timestep=0.01,
                              initial_position=5.0, elasticity=0.7)
            ball_b = BallNode(name="ball_b", timestep=0.01,
                              initial_position=10.0, elasticity=0.9)
            gm.add_node(table)
            gm.add_node(ball_a)
            gm.add_node(ball_b)
            gm.add_edge("table", "ball_a", "position", "table_position")
            gm.add_edge("table", "ball_b", "position", "table_position")
            gm.compile()
            return gm

        n_steps = 100

        gm_obs = make()
        logger = HistoryLogger()
        gm_obs.add_observer(logger)
        gm_obs.run(n_steps)

        gm_scan = make()
        _, scan_history = gm_scan.run_scan_with_history(n_steps)

        for node_name in scan_history:
            for field_name in scan_history[node_name]:
                assert jnp.allclose(
                    logger.history[node_name][field_name],
                    scan_history[node_name][field_name],
                    atol=1e-5,
                ), f"Mismatch in {node_name}.{field_name}"


# ------------------------------------------------------------------
# Empty run (0 steps)
# ------------------------------------------------------------------

class TestEmptyRun:
    def test_zero_steps_empty_history(self):
        gm = _make_bouncing_ball_graph()
        logger = HistoryLogger()
        gm.add_observer(logger)
        gm.run(0)
        assert len(logger) == 0
        assert logger.history == {}

    def test_zero_steps_repr(self):
        logger = HistoryLogger()
        assert "0 steps" in repr(logger)

    def test_no_observer_calls(self):
        """Logger with no events should have empty history."""
        logger = HistoryLogger()
        assert len(logger) == 0
        assert logger.history == {}


# ------------------------------------------------------------------
# Meta stripping
# ------------------------------------------------------------------

class TestMetaStripping:
    def test_no_meta_in_history(self):
        """_meta key should never appear in recorded history."""
        gm = GraphManager()
        # Use two different timesteps to trigger multi-rate (_meta key)
        ball = BallNode(name="ball", timestep=0.01,
                        initial_position=5.0, elasticity=0.7)
        table = TableNode(name="table", timestep=0.01, position=0.0)
        gm.add_node(table)
        gm.add_node(ball)
        gm.add_edge("table", "ball", "position", "table_position")
        gm.compile()

        logger = HistoryLogger()
        gm.add_observer(logger)
        gm.run(10)
        assert "_meta" not in logger.history
