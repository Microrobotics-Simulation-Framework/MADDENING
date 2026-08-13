"""Property-based tests for determinism.

Tests that MADDENING simulations are fully deterministic: same inputs
always produce exactly the same outputs, JIT does not change results,
and vmap over a batch matches sequential execution.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings, assume, note
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.heat import HeatNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leaves_allclose(tree_a, tree_b, rtol=0.0, atol=0.0):
    """Check all leaves of two pytrees are within tolerance."""
    leaves_a = jax.tree.leaves(tree_a)
    leaves_b = jax.tree.leaves(tree_b)
    assert len(leaves_a) == len(leaves_b), "Pytree structure mismatch"
    for la, lb in zip(leaves_a, leaves_b):
        if not jnp.allclose(la, lb, rtol=rtol, atol=atol):
            return False
    return True


def _leaves_exactly_equal(tree_a, tree_b):
    """Check all leaves are bitwise equal."""
    leaves_a = jax.tree.leaves(tree_a)
    leaves_b = jax.tree.leaves(tree_b)
    if len(leaves_a) != len(leaves_b):
        return False
    for la, lb in zip(leaves_a, leaves_b):
        if not jnp.array_equal(la, lb):
            return False
    return True


# Tolerance for JIT vs eager comparison.
# JAX/XLA may reorder floating-point operations under JIT, causing
# differences at the ULP level. Empirically observed:
# - SpringDamperNode: 1 ULP (~1.2e-7 for values near 1.0)
# - HeatNode (10-cell stencil): 2-3 ULPs (~2.4e-7 near 1.0)
# - For larger values (e.g. 20.0), ULP is proportionally larger
# We use both rtol and atol to cover both regimes.
_JIT_ATOL = 1e-6   # absolute tolerance (covers values near zero)
_JIT_RTOL = 1e-6   # relative tolerance (covers larger values)


# ---------------------------------------------------------------------------
# Test: same inputs produce exactly the same outputs
# ---------------------------------------------------------------------------

class TestRepeatability:
    """Calling update or step with the same inputs must produce identical results."""

    @given(
        position=st.floats(min_value=-100, max_value=100,
                           allow_nan=False, allow_infinity=False),
        velocity=st.floats(min_value=-50, max_value=50,
                           allow_nan=False, allow_infinity=False),
        table_pos=st.floats(min_value=-10, max_value=10,
                            allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=300, deadline=None)
    def test_ball_update_deterministic(self, position, velocity, table_pos):
        """BallNode.update called twice with same args gives same result."""
        node = BallNode(name="b", timestep=0.01, initial_position=0.0)
        state = {
            "position": jnp.array(position, dtype=jnp.float32),
            "velocity": jnp.array(velocity, dtype=jnp.float32),
        }
        bi = {"table_position": jnp.array(table_pos, dtype=jnp.float32)}
        dt = 0.01

        out1 = node.update(state, bi, dt)
        out2 = node.update(state, bi, dt)

        assert _leaves_exactly_equal(out1, out2), (
            f"BallNode not deterministic: {out1} != {out2}"
        )

    @given(
        position=st.floats(min_value=-50, max_value=50,
                           allow_nan=False, allow_infinity=False),
        velocity=st.floats(min_value=-20, max_value=20,
                           allow_nan=False, allow_infinity=False),
        anchor=st.floats(min_value=-10, max_value=10,
                         allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=300, deadline=None)
    def test_spring_update_deterministic(self, position, velocity, anchor):
        """SpringDamperNode.update called twice gives same result."""
        node = SpringDamperNode(name="s", timestep=0.01, stiffness=50.0)
        state = {
            "position": jnp.array(position, dtype=jnp.float32),
            "velocity": jnp.array(velocity, dtype=jnp.float32),
        }
        bi = {"anchor_position": jnp.array(anchor, dtype=jnp.float32)}
        dt = 0.01

        out1 = node.update(state, bi, dt)
        out2 = node.update(state, bi, dt)

        assert _leaves_exactly_equal(out1, out2)

    @given(
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(max_examples=100, deadline=None)
    def test_heat_update_deterministic(self, seed):
        """HeatNode.update called twice gives same result."""
        rng = np.random.default_rng(seed)
        n_cells = 8
        node = HeatNode(name="h", timestep=0.001, n_cells=n_cells,
                        thermal_diffusivity=0.01)

        temp = jnp.array(rng.uniform(0, 100, n_cells), dtype=jnp.float32)
        state = {"temperature": temp}
        bi = {
            "left_temperature": jnp.array(float(rng.uniform(0, 100)), dtype=jnp.float32),
            "right_temperature": jnp.array(float(rng.uniform(0, 100)), dtype=jnp.float32),
        }

        out1 = node.update(state, bi, 0.001)
        out2 = node.update(state, bi, 0.001)

        assert _leaves_exactly_equal(out1, out2)

    @given(
        n_steps=st.integers(min_value=1, max_value=20),
        position=st.floats(min_value=-10, max_value=10,
                           allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_graph_step_deterministic(self, n_steps, position):
        """GraphManager.step called with same state produces same result."""
        gm1 = GraphManager()
        gm1.add_node(BallNode(name="b", timestep=0.01, initial_position=position))
        gm1.compile()

        gm2 = GraphManager()
        gm2.add_node(BallNode(name="b", timestep=0.01, initial_position=position))
        gm2.compile()

        for _ in range(n_steps):
            state1 = gm1.step()
            state2 = gm2.step()
            assert _leaves_exactly_equal(state1, state2), (
                f"GraphManager not deterministic at step: "
                f"{state1} != {state2}"
            )


# ---------------------------------------------------------------------------
# Test: JIT does not change results vs eager
# ---------------------------------------------------------------------------

class TestJITConsistency:
    """JIT compilation must not change numerical results."""

    @given(
        position=st.floats(min_value=-50, max_value=50,
                           allow_nan=False, allow_infinity=False),
        velocity=st.floats(min_value=-20, max_value=20,
                           allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_ball_jit_matches_eager(self, position, velocity):
        """BallNode.update under jit gives same result as eager.

        NOTE: JAX/XLA may reorder FP ops under JIT causing 1-2 ULP diffs
        (confirmed empirically). We allow _JIT_ATOL tolerance.
        """
        node = BallNode(name="b", timestep=0.01, gravity=-9.81)
        state = {
            "position": jnp.array(position, dtype=jnp.float32),
            "velocity": jnp.array(velocity, dtype=jnp.float32),
        }
        bi = {"table_position": jnp.array(0.0, dtype=jnp.float32)}
        dt = 0.01

        # Eager
        eager_out = node.update(state, bi, dt)

        # JIT
        jitted_update = jax.jit(node.update)
        jit_out = jitted_update(state, bi, dt)

        assert _leaves_allclose(eager_out, jit_out, rtol=_JIT_RTOL, atol=_JIT_ATOL), (
            f"JIT != eager beyond ULP tolerance: {eager_out} vs {jit_out}"
        )

    @given(
        position=st.floats(min_value=-20, max_value=20,
                           allow_nan=False, allow_infinity=False),
        velocity=st.floats(min_value=-10, max_value=10,
                           allow_nan=False, allow_infinity=False),
        anchor=st.floats(min_value=-5, max_value=5,
                         allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_spring_jit_matches_eager(self, position, velocity, anchor):
        """SpringDamperNode.update under jit gives same result as eager.

        NOTE: XLA may fuse/reorder FP ops causing 1-ULP jitter.
        Confirmed: stiffness=30, pos=0, vel=1, anchor=1 gives
        velocity=1.5799999 (eager) vs 1.58 (JIT) -- 1 ULP difference.
        """
        node = SpringDamperNode(name="s", timestep=0.01, stiffness=30.0, damping=2.0)
        state = {
            "position": jnp.array(position, dtype=jnp.float32),
            "velocity": jnp.array(velocity, dtype=jnp.float32),
        }
        bi = {"anchor_position": jnp.array(anchor, dtype=jnp.float32)}
        dt = 0.01

        eager_out = node.update(state, bi, dt)
        jit_out = jax.jit(node.update)(state, bi, dt)

        assert _leaves_allclose(eager_out, jit_out, rtol=_JIT_RTOL, atol=_JIT_ATOL), (
            f"Spring JIT != eager beyond ULP tolerance: "
            f"pos diff={float(eager_out['position'])-float(jit_out['position'])}, "
            f"vel diff={float(eager_out['velocity'])-float(jit_out['velocity'])}"
        )

    @given(seed=st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=80, deadline=None)
    def test_heat_jit_matches_eager(self, seed):
        """HeatNode.update under jit gives same result as eager.

        NOTE: XLA's FP-reordering under JIT can cause 1-ULP diffs
        in the Laplacian stencil operations.
        """
        rng = np.random.default_rng(seed)
        n_cells = 10
        node = HeatNode(name="h", timestep=0.001, n_cells=n_cells)

        temp = jnp.array(rng.uniform(0, 50, n_cells), dtype=jnp.float32)
        state = {"temperature": temp}
        bi = {
            "left_temperature": jnp.array(float(rng.uniform(0, 50)), dtype=jnp.float32),
            "right_temperature": jnp.array(float(rng.uniform(0, 50)), dtype=jnp.float32),
        }

        eager_out = node.update(state, bi, 0.001)
        jit_out = jax.jit(node.update)(state, bi, 0.001)

        assert _leaves_allclose(eager_out, jit_out, rtol=_JIT_RTOL, atol=_JIT_ATOL), (
            f"Heat JIT != eager beyond ULP tolerance"
        )

    @given(
        n_steps=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(max_examples=30, deadline=None)
    def test_graph_scan_matches_step_loop(self, n_steps, seed):
        """run_scan should produce the same final state as repeated step().

        Both paths go through JIT, so results should be bitwise identical
        (same compiled code, same execution order).
        """
        rng = np.random.default_rng(seed)
        pos = float(rng.uniform(-10, 10))
        vel = float(rng.uniform(-5, 5))

        # Via step loop
        gm_loop = GraphManager()
        gm_loop.add_node(BallNode(
            name="b", timestep=0.01,
            initial_position=pos, initial_velocity=vel,
        ))
        gm_loop.compile()
        for _ in range(n_steps):
            state_loop = gm_loop.step()

        # Via run_scan
        gm_scan = GraphManager()
        gm_scan.add_node(BallNode(
            name="b", timestep=0.01,
            initial_position=pos, initial_velocity=vel,
        ))
        gm_scan.compile()
        state_scan = gm_scan.run_scan(n_steps)

        # Both go through JIT -- results should match exactly or within
        # a few ULPs if scan fuses differently than repeated jit calls
        assert _leaves_allclose(state_loop, state_scan, rtol=_JIT_RTOL, atol=_JIT_ATOL), (
            f"run_scan != step loop after {n_steps} steps: "
            f"{state_loop} vs {state_scan}"
        )


# ---------------------------------------------------------------------------
# Test: vmap over a batch produces the same results as a loop
# ---------------------------------------------------------------------------

class TestVmapConsistency:
    """vmap over a batch must match sequential execution exactly."""

    @given(
        batch_size=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(max_examples=50, deadline=None)
    def test_ball_vmap_matches_loop(self, batch_size, seed):
        """vmapped BallNode.update matches sequential calls."""
        rng = np.random.default_rng(seed)
        node = BallNode(name="b", timestep=0.01, gravity=-9.81)
        dt = 0.01

        # Generate batch of states
        positions = rng.uniform(-50, 50, batch_size).astype(np.float32)
        velocities = rng.uniform(-20, 20, batch_size).astype(np.float32)

        batched_state = {
            "position": jnp.array(positions),
            "velocity": jnp.array(velocities),
        }
        bi = {"table_position": jnp.array(0.0, dtype=jnp.float32)}

        # Sequential loop
        loop_results = []
        for i in range(batch_size):
            single_state = {
                "position": jnp.array(positions[i]),
                "velocity": jnp.array(velocities[i]),
            }
            out = node.update(single_state, bi, dt)
            loop_results.append(out)

        # vmap
        def update_single(pos, vel):
            s = {"position": pos, "velocity": vel}
            return node.update(s, bi, dt)

        vmap_out = jax.vmap(update_single)(
            batched_state["position"],
            batched_state["velocity"],
        )

        for i in range(batch_size):
            assert jnp.array_equal(
                vmap_out["position"][i], loop_results[i]["position"]
            ), f"vmap position mismatch at index {i}"
            assert jnp.array_equal(
                vmap_out["velocity"][i], loop_results[i]["velocity"]
            ), f"vmap velocity mismatch at index {i}"

    @given(
        batch_size=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(max_examples=50, deadline=None)
    def test_spring_vmap_matches_loop(self, batch_size, seed):
        """vmapped SpringDamperNode.update matches sequential calls."""
        rng = np.random.default_rng(seed)
        node = SpringDamperNode(
            name="s", timestep=0.01,
            stiffness=50.0, damping=2.0, mass=1.0,
        )
        dt = 0.01

        positions = rng.uniform(-10, 10, batch_size).astype(np.float32)
        velocities = rng.uniform(-5, 5, batch_size).astype(np.float32)
        anchors = rng.uniform(-3, 3, batch_size).astype(np.float32)

        # Sequential
        loop_results = []
        for i in range(batch_size):
            s = {
                "position": jnp.array(positions[i]),
                "velocity": jnp.array(velocities[i]),
            }
            bi = {"anchor_position": jnp.array(anchors[i])}
            loop_results.append(node.update(s, bi, dt))

        # vmap over state AND boundary input
        def update_single(pos, vel, anch):
            s = {"position": pos, "velocity": vel}
            bi = {"anchor_position": anch}
            return node.update(s, bi, dt)

        vmap_out = jax.vmap(update_single)(
            jnp.array(positions),
            jnp.array(velocities),
            jnp.array(anchors),
        )

        for i in range(batch_size):
            assert jnp.array_equal(
                vmap_out["position"][i], loop_results[i]["position"]
            ), f"Spring vmap position mismatch at index {i}"
            assert jnp.array_equal(
                vmap_out["velocity"][i], loop_results[i]["velocity"]
            ), f"Spring vmap velocity mismatch at index {i}"

    @given(
        batch_size=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(max_examples=30, deadline=None)
    def test_heat_vmap_matches_loop(self, batch_size, seed):
        """vmapped HeatNode.update matches sequential calls."""
        rng = np.random.default_rng(seed)
        n_cells = 6
        node = HeatNode(name="h", timestep=0.001, n_cells=n_cells,
                        thermal_diffusivity=0.01)
        dt = 0.001

        temps = rng.uniform(0, 100, (batch_size, n_cells)).astype(np.float32)
        left_bcs = rng.uniform(0, 100, batch_size).astype(np.float32)
        right_bcs = rng.uniform(0, 100, batch_size).astype(np.float32)

        # Sequential
        loop_results = []
        for i in range(batch_size):
            s = {"temperature": jnp.array(temps[i])}
            bi = {
                "left_temperature": jnp.array(left_bcs[i]),
                "right_temperature": jnp.array(right_bcs[i]),
            }
            loop_results.append(node.update(s, bi, dt))

        # vmap
        def update_single(temp, l_bc, r_bc):
            s = {"temperature": temp}
            bi = {"left_temperature": l_bc, "right_temperature": r_bc}
            return node.update(s, bi, dt)

        vmap_out = jax.vmap(update_single)(
            jnp.array(temps),
            jnp.array(left_bcs),
            jnp.array(right_bcs),
        )

        for i in range(batch_size):
            assert jnp.array_equal(
                vmap_out["temperature"][i], loop_results[i]["temperature"]
            ), f"Heat vmap mismatch at index {i}"

    @given(
        batch_size=st.integers(min_value=2, max_value=4),
        n_steps=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(max_examples=20, deadline=None)
    def test_graph_sweep_matches_individual_runs(self, batch_size, n_steps, seed):
        """GraphManager.run_sweep matches individual run_scan calls."""
        rng = np.random.default_rng(seed)

        positions = rng.uniform(-10, 10, batch_size).astype(np.float32)
        velocities = rng.uniform(-5, 5, batch_size).astype(np.float32)

        # Individual runs
        individual_results = []
        for i in range(batch_size):
            gm = GraphManager()
            gm.add_node(BallNode(
                name="b", timestep=0.01,
                initial_position=float(positions[i]),
                initial_velocity=float(velocities[i]),
            ))
            gm.compile()
            final = gm.run_scan(n_steps)
            individual_results.append(final)

        # Batch via run_sweep
        gm_sweep = GraphManager()
        gm_sweep.add_node(BallNode(
            name="b", timestep=0.01,
            initial_position=0.0, initial_velocity=0.0,
        ))
        gm_sweep.compile()

        batched_init = {
            "b": {
                "position": jnp.array(positions),
                "velocity": jnp.array(velocities),
            },
        }
        sweep_results = gm_sweep.run_sweep(n_steps, batched_init)

        for i in range(batch_size):
            assert jnp.allclose(
                sweep_results["b"]["position"][i],
                individual_results[i]["b"]["position"],
                atol=0.0, rtol=0.0,
            ), f"Sweep position mismatch at index {i}"
            assert jnp.allclose(
                sweep_results["b"]["velocity"][i],
                individual_results[i]["b"]["velocity"],
                atol=0.0, rtol=0.0,
            ), f"Sweep velocity mismatch at index {i}"


# ---------------------------------------------------------------------------
# Test: no hidden mutable state
# ---------------------------------------------------------------------------

class TestNoHiddenState:
    """Nodes should not accumulate hidden mutable state between calls."""

    @given(
        n_calls=st.integers(min_value=2, max_value=10),
        position=st.floats(min_value=-10, max_value=10,
                           allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_ball_no_hidden_state(self, n_calls, position):
        """Calling update N times with the SAME state gives the SAME output each time."""
        node = BallNode(name="b", timestep=0.01, gravity=-9.81)
        state = {
            "position": jnp.array(position, dtype=jnp.float32),
            "velocity": jnp.array(0.0, dtype=jnp.float32),
        }
        bi = {"table_position": jnp.array(-1.0, dtype=jnp.float32)}
        dt = 0.01

        reference_out = node.update(state, bi, dt)
        for i in range(n_calls - 1):
            out = node.update(state, bi, dt)
            assert _leaves_exactly_equal(out, reference_out), (
                f"Hidden state detected on call {i+2}: output differs from first call"
            )

    @given(seed=st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=50, deadline=None)
    def test_heat_no_hidden_state(self, seed):
        """HeatNode has no hidden mutable state between calls."""
        rng = np.random.default_rng(seed)
        n_cells = 8
        node = HeatNode(name="h", timestep=0.001, n_cells=n_cells)

        temp = jnp.array(rng.uniform(10, 90, n_cells), dtype=jnp.float32)
        state = {"temperature": temp}
        bi = {
            "left_temperature": jnp.array(50.0, dtype=jnp.float32),
            "right_temperature": jnp.array(50.0, dtype=jnp.float32),
        }

        out1 = node.update(state, bi, 0.001)

        # Call with different state in between
        other_state = {"temperature": jnp.zeros(n_cells)}
        _ = node.update(other_state, bi, 0.001)

        # Call again with original state -- should get same result
        out2 = node.update(state, bi, 0.001)
        assert _leaves_exactly_equal(out1, out2), (
            "HeatNode has hidden mutable state"
        )

    @given(
        n_steps=st.integers(min_value=1, max_value=10),
        position=st.floats(min_value=-5, max_value=5,
                           allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_graph_restart_from_same_state_deterministic(self, n_steps, position):
        """Re-setting a graph to the same initial state and re-stepping
        must produce the same trajectory."""
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01, initial_position=position))
        gm.compile()

        # First run
        states_1 = []
        for _ in range(n_steps):
            states_1.append(gm.step())

        # Reset state manually
        gm._state = {"b": BallNode(name="b", timestep=0.01, initial_position=position).initial_state()}

        # Second run
        states_2 = []
        for _ in range(n_steps):
            states_2.append(gm.step())

        for i in range(n_steps):
            assert _leaves_exactly_equal(states_1[i], states_2[i]), (
                f"Trajectory diverged at step {i} after state reset"
            )
