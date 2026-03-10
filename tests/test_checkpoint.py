"""Tests for checkpoint / restore (state serialization)."""

import numpy as np
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.checkpoint import save_state, load_state
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_bouncing_ball_graph():
    """Fresh compiled bouncing-ball graph (table -> ball)."""
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01, position=0.0))
    gm.add_node(BallNode(name="ball", timestep=0.01,
                          initial_position=5.0, initial_velocity=0.0,
                          elasticity=0.7))
    gm.add_edge(source="table", target="ball",
                source_field="position", target_field="table_position")
    gm.compile()
    return gm


def _make_multirate_graph():
    """Compiled graph with different timesteps (multi-rate)."""
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01, position=0.0))
    gm.add_node(BallNode(name="ball", timestep=0.02,
                          initial_position=5.0, initial_velocity=0.0,
                          elasticity=0.7))
    gm.add_edge(source="table", target="ball",
                source_field="position", target_field="table_position")
    gm.compile()
    return gm


def _states_equal(state_a, state_b):
    """Check that two nested state dicts have matching arrays."""
    assert state_a.keys() == state_b.keys()
    for node_name in state_a:
        for field in state_a[node_name]:
            np.testing.assert_array_equal(
                np.asarray(state_a[node_name][field]),
                np.asarray(state_b[node_name][field]),
            )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestSaveLoadRoundTrip:
    """Save state, load into the same graph, verify states match."""

    def test_roundtrip_initial_state(self, tmp_path):
        gm = _make_bouncing_ball_graph()
        original_ball = dict(gm.get_node_state("ball"))
        original_table = dict(gm.get_node_state("table"))

        gm.save_state(tmp_path / "ckpt.npz")
        # Mutate state so we can verify load actually restores
        gm.run(100)

        gm.load_state(tmp_path / "ckpt.npz")
        np.testing.assert_array_equal(
            np.asarray(gm.get_node_state("ball")["position"]),
            np.asarray(original_ball["position"]),
        )
        np.testing.assert_array_equal(
            np.asarray(gm.get_node_state("ball")["velocity"]),
            np.asarray(original_ball["velocity"]),
        )
        np.testing.assert_array_equal(
            np.asarray(gm.get_node_state("table")["position"]),
            np.asarray(original_table["position"]),
        )

    def test_roundtrip_after_stepping(self, tmp_path):
        gm = _make_bouncing_ball_graph()
        gm.run(50)
        saved_ball = dict(gm.get_node_state("ball"))

        gm.save_state(tmp_path / "mid.npz")
        gm.run(50)  # continue running

        gm.load_state(tmp_path / "mid.npz")
        np.testing.assert_array_equal(
            np.asarray(gm.get_node_state("ball")["position"]),
            np.asarray(saved_ball["position"]),
        )
        np.testing.assert_array_equal(
            np.asarray(gm.get_node_state("ball")["velocity"]),
            np.asarray(saved_ball["velocity"]),
        )


class TestSaveRunLoadVerify:
    """Save, run more steps, load -- verify state matches saved point."""

    def test_load_restores_saved_not_current(self, tmp_path):
        gm = _make_bouncing_ball_graph()
        gm.run(30)
        saved_state = {
            "ball": dict(gm.get_node_state("ball")),
            "table": dict(gm.get_node_state("table")),
        }
        gm.save_state(tmp_path / "s30")

        gm.run(200)
        # State should have evolved
        assert not jnp.array_equal(
            gm.get_node_state("ball")["position"],
            saved_state["ball"]["position"],
        )

        gm.load_state(tmp_path / "s30")
        _states_equal(
            {"ball": gm.get_node_state("ball"),
             "table": gm.get_node_state("table")},
            saved_state,
        )


class TestContinueFromCheckpoint:
    """Save, load, continue running -- compare to uninterrupted run."""

    def test_deterministic_continuation(self, tmp_path):
        # Run A: uninterrupted 100 steps
        gm_a = _make_bouncing_ball_graph()
        gm_a.run(100)
        state_a = {
            "ball": dict(gm_a.get_node_state("ball")),
            "table": dict(gm_a.get_node_state("table")),
        }

        # Run B: 50 steps, checkpoint, load, 50 more steps
        gm_b = _make_bouncing_ball_graph()
        gm_b.run(50)
        gm_b.save_state(tmp_path / "half")
        # Mess up state by running extra
        gm_b.run(999)
        # Restore to step 50
        gm_b.load_state(tmp_path / "half")
        gm_b.run(50)

        state_b = {
            "ball": dict(gm_b.get_node_state("ball")),
            "table": dict(gm_b.get_node_state("table")),
        }
        _states_equal(state_a, state_b)


class TestMultiRate:
    """Verify _meta.step_count is saved and restored for multi-rate graphs."""

    def test_meta_step_count_saved(self, tmp_path):
        gm = _make_multirate_graph()
        assert gm.is_multirate
        gm.run(17)

        step_count_before = int(gm._state["_meta"]["step_count"])
        assert step_count_before == 17

        gm.save_state(tmp_path / "mr.npz")
        gm.run(10)
        assert int(gm._state["_meta"]["step_count"]) == 27

        gm.load_state(tmp_path / "mr.npz")
        assert int(gm._state["_meta"]["step_count"]) == step_count_before

    def test_multirate_deterministic_continuation(self, tmp_path):
        # Uninterrupted
        gm_a = _make_multirate_graph()
        gm_a.run(40)
        state_a = {
            "ball": dict(gm_a.get_node_state("ball")),
            "table": dict(gm_a.get_node_state("table")),
        }

        # Interrupted
        gm_b = _make_multirate_graph()
        gm_b.run(20)
        gm_b.save_state(tmp_path / "mr_half")
        gm_b.run(999)
        gm_b.load_state(tmp_path / "mr_half")
        gm_b.run(20)

        state_b = {
            "ball": dict(gm_b.get_node_state("ball")),
            "table": dict(gm_b.get_node_state("table")),
        }
        _states_equal(state_a, state_b)


class TestMismatchDetection:
    """Loading into a graph with different nodes raises ValueError."""

    def test_different_nodes_raises(self, tmp_path):
        gm1 = _make_bouncing_ball_graph()
        gm1.save_state(tmp_path / "v1.npz")

        # Build a graph with only a table -- different structure
        gm2 = GraphManager()
        gm2.add_node(TableNode(name="table", timestep=0.01, position=0.0))
        gm2.compile()

        with pytest.raises(ValueError, match="mismatch"):
            gm2.load_state(tmp_path / "v1.npz")

    def test_different_node_names_raises(self, tmp_path):
        gm1 = _make_bouncing_ball_graph()
        gm1.save_state(tmp_path / "v1.npz")

        # Same node types but different names
        gm2 = GraphManager()
        gm2.add_node(TableNode(name="desk", timestep=0.01, position=0.0))
        gm2.add_node(BallNode(name="sphere", timestep=0.01,
                                initial_position=5.0, initial_velocity=0.0))
        gm2.add_edge(source="desk", target="sphere",
                     source_field="position", target_field="table_position")
        gm2.compile()

        with pytest.raises(ValueError, match="mismatch"):
            gm2.load_state(tmp_path / "v1.npz")


class TestMultiFieldNodes:
    """Ball has position + velocity -- verify all fields are saved."""

    def test_all_ball_fields_present_in_npz(self, tmp_path):
        gm = _make_bouncing_ball_graph()
        gm.run(10)
        gm.save_state(tmp_path / "fields.npz")

        data = np.load(tmp_path / "fields.npz")
        keys = set(data.files)
        assert "ball/position" in keys
        assert "ball/velocity" in keys
        assert "table/position" in keys

    def test_field_values_match(self, tmp_path):
        gm = _make_bouncing_ball_graph()
        gm.run(10)
        pos = np.asarray(gm.get_node_state("ball")["position"])
        vel = np.asarray(gm.get_node_state("ball")["velocity"])
        gm.save_state(tmp_path / "fields.npz")

        data = np.load(tmp_path / "fields.npz")
        np.testing.assert_array_equal(data["ball/position"], pos)
        np.testing.assert_array_equal(data["ball/velocity"], vel)


class TestPlainNumpyLoad:
    """Checkpoint file can be loaded with plain numpy (np.load)."""

    def test_numpy_loadable(self, tmp_path):
        gm = _make_bouncing_ball_graph()
        gm.run(5)
        gm.save_state(tmp_path / "plain.npz")

        data = np.load(tmp_path / "plain.npz")
        assert isinstance(data["ball/position"], np.ndarray)
        assert isinstance(data["ball/velocity"], np.ndarray)
        assert isinstance(data["table/position"], np.ndarray)
        data.close()


class TestPathExtension:
    """Path with .npz extension and without should both work."""

    def test_save_load_with_npz_extension(self, tmp_path):
        gm = _make_bouncing_ball_graph()
        gm.run(5)
        gm.save_state(tmp_path / "with_ext.npz")
        gm.load_state(tmp_path / "with_ext.npz")

    def test_save_load_without_npz_extension(self, tmp_path):
        gm = _make_bouncing_ball_graph()
        gm.run(5)
        gm.save_state(tmp_path / "no_ext")
        # numpy.savez creates "no_ext.npz" -- load should find it
        gm.load_state(tmp_path / "no_ext")

    def test_load_nonexistent_raises(self, tmp_path):
        gm = _make_bouncing_ball_graph()
        with pytest.raises(FileNotFoundError):
            gm.load_state(tmp_path / "does_not_exist")


class TestConvenienceMethods:
    """GraphManager.save_state / load_state delegate correctly."""

    def test_convenience_roundtrip(self, tmp_path):
        gm = _make_bouncing_ball_graph()
        gm.run(25)
        saved_pos = np.asarray(gm.get_node_state("ball")["position"])

        gm.save_state(tmp_path / "conv.npz")
        gm.run(100)

        gm.load_state(tmp_path / "conv.npz")
        np.testing.assert_array_equal(
            np.asarray(gm.get_node_state("ball")["position"]),
            saved_pos,
        )
