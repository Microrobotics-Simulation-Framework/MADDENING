"""Tests for GraphManager -- construction, validation, compilation, execution."""

import pytest
import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager, ExternalInputSpec
from maddening.core.edge import EdgeSpec
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


# ------------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------------

class TestConstruction:
    def test_add_node(self):
        gm = GraphManager()
        ball = BallNode(name="ball", timestep=0.01)
        gm.add_node(ball)
        assert "ball" in gm.node_names

    def test_add_duplicate_node_raises(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01))
        with pytest.raises(ValueError, match="already exists"):
            gm.add_node(BallNode(name="b", timestep=0.01))

    def test_add_edge(self):
        gm = GraphManager()
        gm.add_node(TableNode(name="t", timestep=0.01))
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("t", "b", "position", "table_position")
        assert len(gm._edges) == 1

    def test_remove_node(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.remove_node("b")
        assert "b" not in gm.node_names

    def test_remove_node_removes_connected_edges(self):
        gm = GraphManager()
        gm.add_node(TableNode(name="t", timestep=0.01))
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("t", "b", "position", "table_position")
        gm.remove_node("b")
        assert len(gm._edges) == 0

    def test_remove_node_removes_external_inputs(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_external_input("b", "force")
        gm.remove_node("b")
        assert len(gm._external_inputs) == 0

    def test_remove_nonexistent_node_raises(self):
        gm = GraphManager()
        with pytest.raises(KeyError):
            gm.remove_node("nonexistent")

    def test_remove_edge(self):
        gm = GraphManager()
        gm.add_node(TableNode(name="t", timestep=0.01))
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("t", "b", "position", "table_position")
        gm.remove_edge("t", "b", "position", "table_position")
        assert len(gm._edges) == 0

    def test_marks_dirty_on_mutation(self):
        gm = GraphManager()
        gm.add_node(TableNode(name="t", timestep=0.01))
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("t", "b", "position", "table_position")
        gm.compile()
        assert not gm._dirty
        gm.add_node(TableNode(name="t2", timestep=0.01))
        assert gm._dirty


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

class TestValidation:
    def test_valid_graph_no_issues(self, bouncing_ball_graph):
        # Already compiled, so validation passed. Re-validate:
        issues = bouncing_ball_graph.validate()
        errors = [i for i in issues if i.startswith("ERROR")]
        assert len(errors) == 0

    def test_nonexistent_source_node(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm._edges.append(EdgeSpec("ghost", "b", "x", "y"))
        issues = gm.validate()
        assert any("ghost" in i for i in issues)

    def test_nonexistent_target_node(self):
        gm = GraphManager()
        gm.add_node(TableNode(name="t", timestep=0.01))
        gm._edges.append(EdgeSpec("t", "ghost", "position", "y"))
        issues = gm.validate()
        assert any("ghost" in i for i in issues)

    def test_bad_source_field(self):
        gm = GraphManager()
        gm.add_node(TableNode(name="t", timestep=0.01))
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("t", "b", "nonexistent_field", "table_position")
        issues = gm.validate()
        assert any("nonexistent_field" in i for i in issues)

    def test_disconnected_node_warning(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="lonely", timestep=0.01))
        issues = gm.validate()
        assert any("disconnected" in i.lower() for i in issues)

    def test_mixed_timesteps_info(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="fast", timestep=0.01))
        gm.add_node(BallNode(name="slow", timestep=0.1))
        gm.add_edge("fast", "slow", "position", "table_position")
        issues = gm.validate()
        # Multi-rate is now supported -- should be INFO, not ERROR
        assert any("multi-rate" in i.lower() for i in issues)
        errors = [i for i in issues if i.startswith("ERROR")]
        timestep_errors = [e for e in errors if "timestep" in e.lower()]
        assert len(timestep_errors) == 0

    def test_cycle_detected_as_warning(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="a", timestep=0.01))
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("a", "b", "position", "table_position")
        gm.add_edge("b", "a", "position", "table_position")
        issues = gm.validate()
        assert any("cycle" in i.lower() for i in issues)
        # Should be warning, not error
        errors = [i for i in issues if i.startswith("ERROR")]
        cycle_errors = [e for e in errors if "cycle" in e.lower()]
        assert len(cycle_errors) == 0

    def test_external_input_bad_node(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_external_input("nonexistent", "force")
        issues = gm.validate()
        assert any("nonexistent" in i for i in issues)


# ------------------------------------------------------------------
# Compilation
# ------------------------------------------------------------------

class TestCompilation:
    def test_compile_sets_schedule(self, bouncing_ball_graph):
        assert len(bouncing_ball_graph.schedule) == 2
        # Table should come before ball (table is source)
        assert bouncing_ball_graph.schedule.index("table") < \
               bouncing_ball_graph.schedule.index("ball")

    def test_compile_clears_dirty(self, bouncing_ball_graph):
        assert not bouncing_ball_graph._dirty

    def test_compile_creates_compiled_step(self, bouncing_ball_graph):
        assert bouncing_ball_graph._compiled_step is not None

    def test_compile_with_errors_raises(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01))
        # Add an edge referencing a non-existent node to trigger a real error
        gm._edges.append(EdgeSpec("ghost", "b", "x", "y"))
        with pytest.raises(RuntimeError, match="errors"):
            gm.compile()

    def test_compile_multirate_succeeds(self):
        """Multi-rate graphs should now compile successfully."""
        gm = GraphManager()
        gm.add_node(BallNode(name="fast", timestep=0.01))
        gm.add_node(BallNode(name="slow", timestep=0.1))
        gm.add_edge("fast", "slow", "position", "table_position")
        gm.compile()  # Should not raise
        assert gm.is_multirate

    def test_auto_compile_on_step(self):
        gm = GraphManager()
        gm.add_node(TableNode(name="t", timestep=0.01))
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("t", "b", "position", "table_position")
        # Don't call compile() -- step() should auto-compile
        gm.step()
        assert not gm._dirty


# ------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------

class TestExecution:
    def test_single_step(self, bouncing_ball_graph):
        state_before = bouncing_ball_graph.get_node_state("ball")
        pos_before = float(state_before["position"])

        bouncing_ball_graph.step()

        state_after = bouncing_ball_graph.get_node_state("ball")
        pos_after = float(state_after["position"])
        # Ball should start falling
        assert pos_after < pos_before

    def test_run_n_steps(self, bouncing_ball_graph):
        bouncing_ball_graph.run(100)
        state = bouncing_ball_graph.get_node_state("ball")
        # After 100 steps, ball should still be above table
        assert float(state["position"]) >= 0.0

    def test_run_with_callback(self, bouncing_ball_graph):
        positions = []

        def record(step_idx, state):
            positions.append(float(state["ball"]["position"]))

        bouncing_ball_graph.run(50, callback=record)
        assert len(positions) == 50
        # First position should be below starting height (ball is falling)
        assert positions[0] < 5.0

    def test_ball_bounces(self, bouncing_ball_graph):
        """Ball should bounce off table and stay above it."""
        bouncing_ball_graph.run(1000)
        state = bouncing_ball_graph.get_node_state("ball")
        assert float(state["position"]) >= 0.0

    def test_table_is_static(self, bouncing_ball_graph):
        pos_before = float(bouncing_ball_graph.get_node_state("table")["position"])
        bouncing_ball_graph.run(100)
        pos_after = float(bouncing_ball_graph.get_node_state("table")["position"])
        assert pos_before == pos_after


# ------------------------------------------------------------------
# Scan-based execution
# ------------------------------------------------------------------

class TestRunScan:
    def test_run_scan_returns_final_state(self, bouncing_ball_graph):
        state = bouncing_ball_graph.run_scan(100)
        assert "ball" in state
        assert "table" in state
        assert "position" in state["ball"]
        assert "velocity" in state["ball"]

    def test_run_scan_matches_run(self):
        """run_scan should produce the same final state as run."""
        def make_graph():
            gm = GraphManager()
            ball = BallNode(name="ball", timestep=0.01,
                            initial_position=5.0, elasticity=0.7)
            table = TableNode(name="table", timestep=0.01, position=0.0)
            gm.add_node(table)
            gm.add_node(ball)
            gm.add_edge("table", "ball", "position", "table_position")
            gm.compile()
            return gm

        n_steps = 200

        gm_loop = make_graph()
        gm_loop.run(n_steps)

        gm_scan = make_graph()
        gm_scan.run_scan(n_steps)

        for node_name in gm_loop.node_names:
            loop_state = gm_loop.get_node_state(node_name)
            scan_state = gm_scan.get_node_state(node_name)
            for field in loop_state:
                assert jnp.allclose(
                    loop_state[field], scan_state[field], atol=1e-5
                ), (
                    f"Mismatch in {node_name}.{field}: "
                    f"run={float(loop_state[field])}, "
                    f"scan={float(scan_state[field])}"
                )

    def test_run_scan_updates_internal_state(self, bouncing_ball_graph):
        pos_before = float(bouncing_ball_graph.get_node_state("ball")["position"])
        bouncing_ball_graph.run_scan(100)
        pos_after = float(bouncing_ball_graph.get_node_state("ball")["position"])
        assert pos_after != pos_before

    def test_run_scan_ball_stays_above_table(self, bouncing_ball_graph):
        bouncing_ball_graph.run_scan(1000)
        state = bouncing_ball_graph.get_node_state("ball")
        assert float(state["position"]) >= 0.0

    def test_run_scan_auto_compiles(self):
        gm = GraphManager()
        gm.add_node(TableNode(name="t", timestep=0.01))
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("t", "b", "position", "table_position")
        # Don't call compile() -- run_scan should auto-compile
        gm.run_scan(10)
        assert not gm._dirty


class TestRunScanWithHistory:
    def test_returns_final_state_and_history(self, bouncing_ball_graph):
        final, history = bouncing_ball_graph.run_scan_with_history(100)
        assert "ball" in final
        assert "ball" in history
        assert "position" in history["ball"]
        assert "velocity" in history["ball"]

    def test_history_shape(self, bouncing_ball_graph):
        n_steps = 50
        _, history = bouncing_ball_graph.run_scan_with_history(n_steps)
        assert history["ball"]["position"].shape == (n_steps,)
        assert history["ball"]["velocity"].shape == (n_steps,)
        assert history["table"]["position"].shape == (n_steps,)

    def test_history_last_entry_matches_final_state(self, bouncing_ball_graph):
        n_steps = 100
        final, history = bouncing_ball_graph.run_scan_with_history(n_steps)
        for node_name in ("ball", "table"):
            for field in final[node_name]:
                assert jnp.allclose(
                    final[node_name][field],
                    history[node_name][field][-1],
                    atol=1e-6,
                ), f"History last entry != final state for {node_name}.{field}"

    def test_history_matches_callback_recording(self):
        """History from scan should match step-by-step recording."""
        def make_graph():
            gm = GraphManager()
            ball = BallNode(name="ball", timestep=0.01,
                            initial_position=5.0, elasticity=0.7)
            table = TableNode(name="table", timestep=0.01, position=0.0)
            gm.add_node(table)
            gm.add_node(ball)
            gm.add_edge("table", "ball", "position", "table_position")
            gm.compile()
            return gm

        n_steps = 200

        # Step-by-step with callback
        gm_loop = make_graph()
        positions = []
        velocities = []
        def record(step_idx, state):
            positions.append(float(state["ball"]["position"]))
            velocities.append(float(state["ball"]["velocity"]))
        gm_loop.run(n_steps, callback=record)

        # Scan with history
        gm_scan = make_graph()
        _, history = gm_scan.run_scan_with_history(n_steps)

        for i in range(n_steps):
            assert jnp.allclose(
                jnp.array(positions[i]),
                history["ball"]["position"][i],
                atol=1e-5,
            ), f"Position mismatch at step {i}"
            assert jnp.allclose(
                jnp.array(velocities[i]),
                history["ball"]["velocity"][i],
                atol=1e-5,
            ), f"Velocity mismatch at step {i}"

    def test_scan_with_history_auto_compiles(self):
        gm = GraphManager()
        gm.add_node(TableNode(name="t", timestep=0.01))
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("t", "b", "position", "table_position")
        gm.run_scan_with_history(10)
        assert not gm._dirty

    def test_final_state_matches_run_scan(self):
        """run_scan and run_scan_with_history should give same final state."""
        def make_graph():
            gm = GraphManager()
            ball = BallNode(name="ball", timestep=0.01,
                            initial_position=5.0, elasticity=0.7)
            table = TableNode(name="table", timestep=0.01, position=0.0)
            gm.add_node(table)
            gm.add_node(ball)
            gm.add_edge("table", "ball", "position", "table_position")
            gm.compile()
            return gm

        n_steps = 100

        gm1 = make_graph()
        state1 = gm1.run_scan(n_steps)

        gm2 = make_graph()
        state2, _ = gm2.run_scan_with_history(n_steps)

        for node_name in gm1.node_names:
            for field in state1[node_name]:
                assert jnp.allclose(
                    state1[node_name][field],
                    state2[node_name][field],
                    atol=1e-6,
                )


# ------------------------------------------------------------------
# External inputs
# ------------------------------------------------------------------

class TestExternalInputs:
    def test_add_external_input(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_external_input("b", "force", shape=(), dtype=jnp.float32)
        assert len(gm._external_inputs) == 1

    def test_default_external_inputs(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_external_input("b", "force", shape=())
        ext = gm._default_external_inputs()
        assert "b" in ext
        assert "force" in ext["b"]
        assert float(ext["b"]["force"]) == 0.0

    def test_empty_external_inputs(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01))
        ext = gm._default_external_inputs()
        assert ext == {}

    def test_step_with_external_inputs(self):
        gm = GraphManager()
        ball = BallNode(name="b", timestep=0.01)
        table = TableNode(name="t", timestep=0.01)
        gm.add_node(table)
        gm.add_node(ball)
        gm.add_edge("t", "b", "position", "table_position")
        gm.compile()
        # Step with no external inputs
        gm.step()
        state = gm.get_node_state("b")
        assert state is not None

    def test_external_input_spec_frozen(self):
        spec = ExternalInputSpec("b", "force", (), jnp.float32)
        assert spec.target_node == "b"
        assert spec.target_field == "force"


# ------------------------------------------------------------------
# State access
# ------------------------------------------------------------------

class TestStateAccess:
    def test_get_node_state(self, bouncing_ball_graph):
        state = bouncing_ball_graph.get_node_state("ball")
        assert "position" in state
        assert "velocity" in state

    def test_get_nonexistent_node_raises(self, bouncing_ball_graph):
        with pytest.raises(KeyError):
            bouncing_ball_graph.get_node_state("ghost")

    def test_set_node_state(self, bouncing_ball_graph):
        new_state = {
            "position": jnp.array(10.0),
            "velocity": jnp.array(0.0),
        }
        bouncing_ball_graph.set_node_state("ball", new_state)
        assert float(bouncing_ball_graph.get_node_state("ball")["position"]) == 10.0

    def test_set_nonexistent_node_raises(self, bouncing_ball_graph):
        with pytest.raises(KeyError):
            bouncing_ball_graph.set_node_state("ghost", {})


# ------------------------------------------------------------------
# Observer pattern
# ------------------------------------------------------------------

class TestObserver:
    def test_observer_called_on_step(self, bouncing_ball_graph):
        events = []
        bouncing_ball_graph.add_observer(lambda e, d: events.append(e))
        bouncing_ball_graph.step()
        assert "step" in events

    def test_observer_receives_state(self, bouncing_ball_graph):
        states = []
        bouncing_ball_graph.add_observer(
            lambda e, d: states.append(d) if e == "step" else None
        )
        bouncing_ball_graph.step()
        assert len(states) == 1
        assert "ball" in states[0]

    def test_multiple_observers(self, bouncing_ball_graph):
        counts = [0, 0]

        def obs1(e, d):
            if e == "step":
                counts[0] += 1

        def obs2(e, d):
            if e == "step":
                counts[1] += 1

        bouncing_ball_graph.add_observer(obs1)
        bouncing_ball_graph.add_observer(obs2)
        bouncing_ball_graph.step()
        assert counts == [1, 1]

    def test_observer_on_compile(self):
        gm = GraphManager()
        gm.add_node(TableNode(name="t", timestep=0.01))
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_edge("t", "b", "position", "table_position")
        events = []
        gm.add_observer(lambda e, d: events.append(e))
        gm.compile()
        assert "compiled" in events


# ------------------------------------------------------------------
# Convenience properties
# ------------------------------------------------------------------

class TestConvenience:
    def test_timestep(self, bouncing_ball_graph):
        assert bouncing_ball_graph.timestep == 0.01

    def test_timestep_no_nodes_raises(self):
        gm = GraphManager()
        with pytest.raises(RuntimeError):
            _ = gm.timestep

    def test_timestep_multirate_returns_gcd(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="fast", timestep=0.01))
        gm.add_node(BallNode(name="slow", timestep=0.1))
        # GCD of 0.01 and 0.1 is 0.01
        assert abs(gm.timestep - 0.01) < 1e-12

    def test_base_timestep_alias(self, bouncing_ball_graph):
        assert bouncing_ball_graph.base_timestep == bouncing_ball_graph.timestep

    def test_node_names(self, bouncing_ball_graph):
        names = bouncing_ball_graph.node_names
        assert set(names) == {"ball", "table"}

    def test_repr(self, bouncing_ball_graph):
        r = repr(bouncing_ball_graph)
        assert "2 nodes" in r
        assert "1 edges" in r
        assert "compiled" in r

    def test_repr_multirate(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="fast", timestep=0.01))
        gm.add_node(BallNode(name="slow", timestep=0.1))
        gm.add_edge("fast", "slow", "position", "table_position")
        gm.compile()
        r = repr(gm)
        assert "multi-rate" in r
