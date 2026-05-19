"""Tests for multi-rate timestep scheduling.

Verifies that nodes with different timesteps are handled correctly:
- base timestep computation (GCD)
- rate dividers
- conditional node updates (slow nodes skip sub-steps)
- equivalence with uniform-rate graphs when all timesteps match
- scan-based execution with multi-rate
- history recording with multi-rate
- gradient flow through multi-rate graphs
- edge data delivery between nodes at different rates
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import (
    GraphManager,
    _float_gcd,
    _multi_gcd,
    _META_KEY,
)
from maddening.core.node import SimulationNode
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


# ------------------------------------------------------------------
# Helper nodes for multi-rate tests
# ------------------------------------------------------------------

class CounterNode(SimulationNode):
    """A node that increments a counter by 1 each time it updates.

    Useful for verifying *how many times* a node actually ran.
    """

    def halo_width(self) -> dict[int, int]:
        return {}

    def initial_state(self) -> dict:
        return {"count": jnp.array(0, dtype=jnp.int32)}

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        return {"count": state["count"] + 1}


class AccumulatorNode(SimulationNode):
    """A node that accumulates dt each step.

    After N actual updates, ``total`` should equal N * node_dt.
    """

    def halo_width(self) -> dict[int, int]:
        return {}

    def initial_state(self) -> dict:
        return {"total": jnp.array(0.0, dtype=jnp.float32)}

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        return {"total": state["total"] + dt}


class ReaderNode(SimulationNode):
    """A node that copies an input value into its state.

    Used to verify edge data delivery between multi-rate nodes.
    """

    def halo_width(self) -> dict[int, int]:
        return {}

    def initial_state(self) -> dict:
        return {"value": jnp.array(0.0, dtype=jnp.float32)}

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        inp = boundary_inputs.get("input_value", state["value"])
        return {"value": inp}


# ------------------------------------------------------------------
# GCD utilities
# ------------------------------------------------------------------

class TestFloatGCD:
    def test_same_value(self):
        assert abs(_float_gcd(0.01, 0.01) - 0.01) < 1e-12

    def test_simple_ratio(self):
        # GCD(0.01, 0.1) = 0.01
        assert abs(_float_gcd(0.01, 0.1) - 0.01) < 1e-12

    def test_order_independent(self):
        assert abs(_float_gcd(0.1, 0.01) - _float_gcd(0.01, 0.1)) < 1e-12

    def test_non_trivial(self):
        # GCD(0.002, 0.005) = 0.001
        assert abs(_float_gcd(0.002, 0.005) - 0.001) < 1e-12

    def test_multi_gcd_two(self):
        assert abs(_multi_gcd([0.01, 0.1]) - 0.01) < 1e-12

    def test_multi_gcd_three(self):
        # GCD(0.01, 0.02, 0.05) = 0.01
        assert abs(_multi_gcd([0.01, 0.02, 0.05]) - 0.01) < 1e-12

    def test_multi_gcd_non_trivial(self):
        # GCD(0.004, 0.006) = 0.002
        assert abs(_multi_gcd([0.004, 0.006]) - 0.002) < 1e-12


# ------------------------------------------------------------------
# Graph properties
# ------------------------------------------------------------------

class TestMultirateProperties:
    def test_is_multirate_false_for_uniform(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="a", timestep=0.01))
        gm.add_node(TableNode(name="b", timestep=0.01))
        gm.add_edge("b", "a", "position", "table_position")
        gm.compile()
        assert not gm.is_multirate

    def test_is_multirate_true(self):
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()
        assert gm.is_multirate

    def test_base_timestep(self):
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        assert abs(gm.timestep - 0.01) < 1e-12
        assert abs(gm.base_timestep - 0.01) < 1e-12

    def test_rate_dividers(self):
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()
        rd = gm.rate_dividers
        assert rd["fast"] == 1
        assert rd["slow"] == 10

    def test_rate_dividers_three_rates(self):
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.001))
        gm.add_node(CounterNode(name="mid", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()
        rd = gm.rate_dividers
        assert rd["fast"] == 1
        assert rd["mid"] == 10
        assert rd["slow"] == 100

    def test_meta_key_in_state_when_multirate(self):
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()
        assert _META_KEY in gm._state

    def test_no_meta_key_when_uniform(self):
        gm = GraphManager()
        gm.add_node(CounterNode(name="a", timestep=0.01))
        gm.add_node(CounterNode(name="b", timestep=0.01))
        gm.compile()
        assert _META_KEY not in gm._state

    def test_validate_info_message(self):
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        issues = gm.validate()
        info = [i for i in issues if i.startswith("INFO")]
        assert len(info) == 1
        assert "multi-rate" in info[0].lower()
        # Should NOT contain any errors about timesteps
        errors = [i for i in issues if i.startswith("ERROR")]
        assert len(errors) == 0


# ------------------------------------------------------------------
# Step execution
# ------------------------------------------------------------------

class TestMultirateExecution:
    def test_fast_node_runs_every_step(self):
        """A node at the base rate should update every step."""
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()

        for _ in range(10):
            gm.step()

        assert int(gm.get_node_state("fast")["count"]) == 10

    def test_slow_node_runs_at_its_rate(self):
        """A slow node (10x divider) should update once per 10 base steps."""
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()

        # Run exactly 10 base steps = 1 slow step
        for _ in range(10):
            gm.step()

        assert int(gm.get_node_state("fast")["count"]) == 10
        assert int(gm.get_node_state("slow")["count"]) == 1

    def test_slow_node_after_20_steps(self):
        """After 20 base steps, slow node should have updated twice."""
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()

        for _ in range(20):
            gm.step()

        assert int(gm.get_node_state("fast")["count"]) == 20
        assert int(gm.get_node_state("slow")["count"]) == 2

    def test_slow_node_no_update_between_fires(self):
        """Between firing steps, slow node state should not change."""
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()

        # Step 0 fires the slow node (step_count=0, 0%10==0)
        gm.step()
        count_after_1 = int(gm.get_node_state("slow")["count"])
        assert count_after_1 == 1

        # Steps 1-9: slow node should NOT update
        for _ in range(9):
            gm.step()
            count = int(gm.get_node_state("slow")["count"])
            assert count == 1, f"Slow node updated unexpectedly, count={count}"

        # Step 10: slow node fires again
        gm.step()
        assert int(gm.get_node_state("slow")["count"]) == 2

    def test_run_multirate(self):
        """run() with multi-rate graph."""
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()

        gm.run(100)
        assert int(gm.get_node_state("fast")["count"]) == 100
        assert int(gm.get_node_state("slow")["count"]) == 10

    def test_run_callback_excludes_meta(self):
        """Callbacks should not see _meta in the state dict."""
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()

        states = []
        gm.run(5, callback=lambda i, s: states.append(s))
        for s in states:
            assert _META_KEY not in s

    def test_step_return_excludes_meta(self):
        """step() return value should not contain _meta."""
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()

        result = gm.step()
        assert _META_KEY not in result

    def test_observer_excludes_meta(self):
        """Observer notifications should not contain _meta."""
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()

        step_data = []
        gm.add_observer(
            lambda e, d: step_data.append(d) if e == "step" else None
        )
        gm.step()
        assert _META_KEY not in step_data[0]

    def test_accumulator_multirate(self):
        """AccumulatorNode should accumulate its own dt, not the base dt."""
        gm = GraphManager()
        gm.add_node(AccumulatorNode(name="fast", timestep=0.01))
        gm.add_node(AccumulatorNode(name="slow", timestep=0.1))
        gm.compile()

        # Run 100 base steps = 1.0s of sim time at base rate
        gm.run(100)

        fast_total = float(gm.get_node_state("fast")["total"])
        slow_total = float(gm.get_node_state("slow")["total"])

        # Fast runs 100 times with dt=0.01: total = 100 * 0.01 = 1.0
        assert abs(fast_total - 1.0) < 1e-4

        # Slow runs 10 times with dt=0.1: total = 10 * 0.1 = 1.0
        assert abs(slow_total - 1.0) < 1e-4


# ------------------------------------------------------------------
# Scan execution
# ------------------------------------------------------------------

class TestMultirateScan:
    def test_run_scan_multirate(self):
        """run_scan with multi-rate should give same result as run."""
        def make_graph():
            gm = GraphManager()
            gm.add_node(CounterNode(name="fast", timestep=0.01))
            gm.add_node(CounterNode(name="slow", timestep=0.1))
            gm.compile()
            return gm

        gm_loop = make_graph()
        gm_loop.run(100)

        gm_scan = make_graph()
        gm_scan.run_scan(100)

        for name in ("fast", "slow"):
            loop_count = int(gm_loop.get_node_state(name)["count"])
            scan_count = int(gm_scan.get_node_state(name)["count"])
            assert loop_count == scan_count, (
                f"{name}: run={loop_count}, scan={scan_count}"
            )

    def test_run_scan_excludes_meta(self):
        """run_scan return value should not contain _meta."""
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()

        result = gm.run_scan(10)
        assert _META_KEY not in result

    def test_scan_with_history_multirate(self):
        """run_scan_with_history with multi-rate."""
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()

        n_steps = 20
        final, history = gm.run_scan_with_history(n_steps)

        assert _META_KEY not in final
        assert _META_KEY not in history

        # History shapes
        assert history["fast"]["count"].shape == (n_steps,)
        assert history["slow"]["count"].shape == (n_steps,)

        # Fast count should be 1, 2, 3, ... 20
        for i in range(n_steps):
            assert int(history["fast"]["count"][i]) == i + 1

        # Slow count: step 0 fires (count=1), then stays 1 until step 10 fires (count=2)
        expected_slow = []
        for i in range(n_steps):
            # step_count at step i is i (0-indexed before increment)
            # Slow fires when step_count % 10 == 0 -> step_count = 0, 10
            if i < 10:
                expected_slow.append(1)
            else:
                expected_slow.append(2)

        for i in range(n_steps):
            assert int(history["slow"]["count"][i]) == expected_slow[i], (
                f"Step {i}: expected {expected_slow[i]}, got {int(history['slow']['count'][i])}"
            )

    def test_scan_with_history_last_matches_final(self):
        """Last history entry should match final state."""
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()

        final, history = gm.run_scan_with_history(50)
        for name in ("fast", "slow"):
            assert jnp.allclose(
                final[name]["count"],
                history[name]["count"][-1],
            )


# ------------------------------------------------------------------
# Edge data delivery between multi-rate nodes
# ------------------------------------------------------------------

class TestMultirateEdges:
    def test_fast_to_slow_edge(self):
        """Fast node feeds data to slow node. Slow node reads latest
        fast value on the sub-step it fires."""
        gm = GraphManager()
        gm.add_node(AccumulatorNode(name="fast", timestep=0.01))
        gm.add_node(ReaderNode(name="slow", timestep=0.1))
        gm.add_edge("fast", "slow", "total", "input_value")
        gm.compile()

        # After 10 base steps: fast.total = 0.1, slow reads it on step 0
        gm.run(10)
        # The slow node fires at step 0 (reads fast.total = 0.01 after
        # fast runs at step 0), then doesn't fire again until step 10.
        # Actually, on step 0, fast updates first (accumulator gets 0.01),
        # then slow reads fast.total = 0.01.
        slow_val = float(gm.get_node_state("slow")["value"])
        # slow fired at step 0 (read fast=0.01). Won't fire until step 10.
        assert abs(slow_val - 0.01) < 1e-5

    def test_slow_to_fast_edge(self):
        """Slow node feeds data to fast node. Fast node reads stale slow
        data between slow updates."""
        gm = GraphManager()
        gm.add_node(AccumulatorNode(name="slow", timestep=0.1))
        gm.add_node(ReaderNode(name="fast", timestep=0.01))
        gm.add_edge("slow", "fast", "total", "input_value")
        gm.compile()

        # After 10 base steps: slow fires at step 0 (total=0.1)
        gm.run(10)
        fast_val = float(gm.get_node_state("fast")["value"])
        # Fast reads slow.total every step. Slow updated at step 0 -> total=0.1
        # That value stays until step 10. So fast last read 0.1.
        assert abs(fast_val - 0.1) < 1e-5


# ------------------------------------------------------------------
# Backward compatibility: uniform-rate graphs
# ------------------------------------------------------------------

class TestUniformRateBackwardsCompat:
    def test_uniform_rate_no_overhead(self):
        """Uniform-rate graphs should not have _meta in state."""
        gm = GraphManager()
        gm.add_node(CounterNode(name="a", timestep=0.01))
        gm.add_node(CounterNode(name="b", timestep=0.01))
        gm.compile()
        assert _META_KEY not in gm._state

    def test_uniform_rate_step(self):
        gm = GraphManager()
        gm.add_node(CounterNode(name="a", timestep=0.01))
        gm.add_node(CounterNode(name="b", timestep=0.01))
        gm.compile()
        gm.run(10)
        assert int(gm.get_node_state("a")["count"]) == 10
        assert int(gm.get_node_state("b")["count"]) == 10

    def test_uniform_bouncing_ball_still_works(self):
        """The classic bouncing ball test should be unchanged."""
        gm = GraphManager()
        gm.add_node(TableNode(name="table", timestep=0.01, position=0.0))
        gm.add_node(BallNode(name="ball", timestep=0.01,
                              initial_position=5.0, elasticity=0.7))
        gm.add_edge("table", "ball", "position", "table_position")
        gm.compile()
        gm.run(1000)
        assert float(gm.get_node_state("ball")["position"]) >= 0.0


# ------------------------------------------------------------------
# Bouncing ball multi-rate test
# ------------------------------------------------------------------

class TestMultirateBouncingBall:
    def test_ball_fast_table_slow(self):
        """Ball at 1kHz, table at 100Hz (static anyway, so same result)."""
        gm = GraphManager()
        gm.add_node(TableNode(name="table", timestep=0.01, position=0.0))
        gm.add_node(BallNode(name="ball", timestep=0.001,
                              initial_position=5.0, elasticity=0.7))
        gm.add_edge("table", "ball", "position", "table_position")
        gm.compile()

        assert gm.is_multirate
        assert abs(gm.timestep - 0.001) < 1e-12

        # Run 5000 base steps = 5.0 seconds of simulation
        gm.run(5000)
        pos = float(gm.get_node_state("ball")["position"])
        # Ball should still be above table
        assert pos >= -1e-6

    def test_multirate_ball_scan_matches_run(self):
        """Verify scan and run produce the same results for multi-rate."""
        def make_graph():
            gm = GraphManager()
            gm.add_node(TableNode(name="table", timestep=0.01, position=0.0))
            gm.add_node(BallNode(name="ball", timestep=0.001,
                                  initial_position=5.0, elasticity=0.7))
            gm.add_edge("table", "ball", "position", "table_position")
            gm.compile()
            return gm

        n_steps = 500

        gm_loop = make_graph()
        gm_loop.run(n_steps)

        gm_scan = make_graph()
        gm_scan.run_scan(n_steps)

        for name in ("ball", "table"):
            loop_state = gm_loop.get_node_state(name)
            scan_state = gm_scan.get_node_state(name)
            for field in loop_state:
                assert jnp.allclose(
                    loop_state[field], scan_state[field], atol=1e-5
                ), (
                    f"Mismatch in {name}.{field}: "
                    f"run={float(loop_state[field])}, "
                    f"scan={float(scan_state[field])}"
                )


# ------------------------------------------------------------------
# Autodiff through multi-rate
# ------------------------------------------------------------------

class TestMultirateGrad:
    def test_grad_through_multirate_step(self):
        """Gradients should flow through conditional updates."""
        gm = GraphManager()
        gm.add_node(AccumulatorNode(name="fast", timestep=0.01))
        gm.add_node(AccumulatorNode(name="slow", timestep=0.1))
        gm.compile()

        step_fn = gm._build_step_fn()
        ext = gm._default_external_inputs()

        def loss_fn(init_total):
            state = {
                "fast": {"total": init_total},
                "slow": {"total": jnp.array(0.0)},
                _META_KEY: {"step_count": jnp.array(0, dtype=jnp.int32)},
            }
            for _ in range(10):
                state = step_fn(state, ext)
            return state["fast"]["total"]

        grad_fn = jax.grad(loss_fn)
        grad_val = grad_fn(jnp.array(0.0))
        assert jnp.isfinite(grad_val)
        # fast accumulates dt=0.01 each step, starting from init_total.
        # After 10 steps: total = init_total + 10*0.01 = init_total + 0.1
        # d(total)/d(init_total) = 1.0
        assert abs(float(grad_val) - 1.0) < 1e-5

    def test_grad_through_multirate_bouncing_ball(self):
        """Gradient through a multi-rate bouncing ball graph."""
        gm = GraphManager()
        gm.add_node(TableNode(name="table", timestep=0.01, position=0.0))
        gm.add_node(BallNode(name="ball", timestep=0.001,
                              initial_position=5.0, elasticity=0.7))
        gm.add_edge("table", "ball", "position", "table_position")
        gm.compile()

        step_fn = gm._build_step_fn()
        ext = gm._default_external_inputs()

        def loss_fn(init_pos):
            state = {
                "table": {"position": jnp.array(0.0)},
                "ball": {"position": init_pos, "velocity": jnp.array(0.0)},
                _META_KEY: {"step_count": jnp.array(0, dtype=jnp.int32)},
            }
            for _ in range(10):
                state = step_fn(state, ext)
            return state["ball"]["position"]

        grad_fn = jax.grad(loss_fn)
        grad_val = grad_fn(jnp.array(5.0))
        assert jnp.isfinite(grad_val)
        assert float(grad_val) != 0.0


# ------------------------------------------------------------------
# Three-rate graph
# ------------------------------------------------------------------

class TestThreeRateGraph:
    def test_three_rates(self):
        """Fast (1ms), medium (10ms), slow (100ms) nodes."""
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.001))
        gm.add_node(CounterNode(name="mid", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()

        assert gm.is_multirate
        assert abs(gm.timestep - 0.001) < 1e-12

        # Run 100 base steps = 0.1s
        gm.run(100)
        assert int(gm.get_node_state("fast")["count"]) == 100
        assert int(gm.get_node_state("mid")["count"]) == 10
        assert int(gm.get_node_state("slow")["count"]) == 1

    def test_three_rates_200_steps(self):
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.001))
        gm.add_node(CounterNode(name="mid", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()

        gm.run(200)
        assert int(gm.get_node_state("fast")["count"]) == 200
        assert int(gm.get_node_state("mid")["count"]) == 20
        assert int(gm.get_node_state("slow")["count"]) == 2


# ------------------------------------------------------------------
# Auto-compile
# ------------------------------------------------------------------

class TestMultirateAutoCompile:
    def test_step_auto_compiles_multirate(self):
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        # No explicit compile
        gm.step()
        assert not gm._dirty
        assert gm.is_multirate

    def test_run_scan_auto_compiles_multirate(self):
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        result = gm.run_scan(10)
        assert not gm._dirty
        assert _META_KEY not in result


# ------------------------------------------------------------------
# get/set node state with multi-rate
# ------------------------------------------------------------------

class TestMultirateStateAccess:
    def test_get_node_state_works(self):
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()
        gm.step()
        state = gm.get_node_state("fast")
        assert "count" in state

    def test_set_node_state_works(self):
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()
        gm.set_node_state("fast", {"count": jnp.array(42, dtype=jnp.int32)})
        assert int(gm.get_node_state("fast")["count"]) == 42

    def test_get_meta_raises(self):
        """Users should not be able to access _meta as a node."""
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()
        with pytest.raises(KeyError):
            gm.get_node_state(_META_KEY)

    def test_set_meta_raises(self):
        """Users should not be able to set _meta."""
        gm = GraphManager()
        gm.add_node(CounterNode(name="fast", timestep=0.01))
        gm.add_node(CounterNode(name="slow", timestep=0.1))
        gm.compile()
        with pytest.raises(KeyError):
            gm.set_node_state(_META_KEY, {})


# ------------------------------------------------------------------
# Multi-rate with external inputs
# ------------------------------------------------------------------

class TestMultirateExternalInputs:
    def test_external_input_with_multirate(self):
        """External inputs should work in multi-rate mode."""
        gm = GraphManager()
        gm.add_node(ReaderNode(name="fast", timestep=0.01))
        gm.add_node(ReaderNode(name="slow", timestep=0.1))
        gm.add_external_input("fast", "input_value", shape=())
        gm.add_external_input("slow", "input_value", shape=())
        gm.compile()

        ext = {
            "fast": {"input_value": jnp.array(42.0)},
            "slow": {"input_value": jnp.array(99.0)},
        }

        gm.step(external_inputs=ext)
        assert abs(float(gm.get_node_state("fast")["value"]) - 42.0) < 1e-5
        assert abs(float(gm.get_node_state("slow")["value"]) - 99.0) < 1e-5


# ------------------------------------------------------------------
# Recompile from multirate to uniform and vice versa
# ------------------------------------------------------------------

class TestMultirateRecompile:
    def test_add_node_recompiles_to_multirate(self):
        gm = GraphManager()
        gm.add_node(CounterNode(name="a", timestep=0.01))
        gm.compile()
        assert not gm.is_multirate

        gm.add_node(CounterNode(name="b", timestep=0.1))
        gm.compile()
        assert gm.is_multirate
        assert _META_KEY in gm._state

    def test_remove_node_recompiles_to_uniform(self):
        gm = GraphManager()
        gm.add_node(CounterNode(name="a", timestep=0.01))
        gm.add_node(CounterNode(name="b", timestep=0.1))
        gm.compile()
        assert gm.is_multirate

        gm.remove_node("b")
        gm.compile()
        assert not gm.is_multirate
        assert _META_KEY not in gm._state


# ------------------------------------------------------------------
# Multi-rate with cycles (back-edge staggering)
# ------------------------------------------------------------------

class TestMultirateWithCycles:
    def test_cycle_with_multirate(self):
        """Two nodes at different rates with a mutual dependency."""

        class EchoNode(SimulationNode):
            def halo_width(self) -> dict[int, int]:
                return {}

            def initial_state(self):
                return {"val": jnp.array(1.0)}

            def update(self, state, boundary_inputs, dt):
                inp = boundary_inputs.get("other_val", jnp.array(0.0))
                return {"val": state["val"] + inp * dt}

        gm = GraphManager()
        gm.add_node(EchoNode(name="a", timestep=0.01))
        gm.add_node(EchoNode(name="b", timestep=0.1))
        gm.add_edge("a", "b", "val", "other_val")
        gm.add_edge("b", "a", "val", "other_val")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.compile()

        assert gm.is_multirate
        gm.run(10)
        # Both should have evolved from their initial value
        assert float(gm.get_node_state("a")["val"]) > 1.0
        assert float(gm.get_node_state("b")["val"]) > 1.0
