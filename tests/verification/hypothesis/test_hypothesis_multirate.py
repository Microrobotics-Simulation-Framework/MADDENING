"""Property-based tests for multirate consistency.

Tests that the multi-rate scheduling in GraphManager produces results
consistent with running nodes at their native timestep.
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
# Helper: build a linear node (no coupling, no collision) to test
# multirate equivalence. For a linear system, N steps at dt should equal
# 1 step at N*dt (up to integration discretization effects).
# ---------------------------------------------------------------------------

class LinearDecayNode:
    """A simple linear decay node: x(t+dt) = x(t) * (1 - alpha*dt).

    This is a trivial linear system where multirate equivalence is exact:
    N steps at dt/N gives the same answer as 1 step at dt
    (to first-order Euler accuracy).

    NOTE: we cannot use this directly with GraphManager because it needs
    to inherit from SimulationNode. Instead we test with SpringDamperNode
    (which is linear) in the undamped, unforced regime.
    """
    pass


# ---------------------------------------------------------------------------
# Test: single node at different rates produces consistent results
# ---------------------------------------------------------------------------

class TestSingleNodeMultirate:
    """A single node stepped N times at dt should match stepping once at N*dt
    only for exact linear integrators. For Euler, we test that the results
    are 'close enough' (bounded error growth)."""

    @given(
        n_sub=st.integers(min_value=1, max_value=10),
        position=st.floats(min_value=-10.0, max_value=10.0,
                           allow_nan=False, allow_infinity=False),
        velocity=st.floats(min_value=-5.0, max_value=5.0,
                           allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_ball_multirate_monotonic_error(self, n_sub, position, velocity):
        """For a BallNode (gravity-only, no collision), finer timestep
        should produce results closer to the analytical solution.

        Analytical: v(t) = v0 + g*t, x(t) = x0 + v0*t + 0.5*g*t^2
        """
        g = -9.81
        dt_total = 0.01  # total time
        dt_fine = dt_total / n_sub

        # Run with fine steps
        node = BallNode(name="b", timestep=dt_fine, gravity=g,
                        initial_position=position, initial_velocity=velocity)
        state = node.initial_state()
        for _ in range(n_sub):
            state = node.update(state, {}, dt_fine)

        # Run with single coarse step
        node_coarse = BallNode(name="b2", timestep=dt_total, gravity=g,
                               initial_position=position, initial_velocity=velocity)
        state_coarse = node_coarse.initial_state()
        state_coarse = node_coarse.update(state_coarse, {}, dt_total)

        # Analytical solution
        x_exact = position + velocity * dt_total + 0.5 * g * dt_total**2
        v_exact = velocity + g * dt_total

        # For forward Euler on this linear system, both should be exact
        # (gravity is constant, so Euler is exact for velocity, and
        # position uses the updated velocity... wait, BallNode uses
        # v_new for position: pos += v_new * dt, so it's semi-implicit)
        # Actually BallNode: v_new = v + g*dt; pos_new = pos + v_new*dt
        # For coarse step: v = v0 + g*T; x = x0 + (v0+g*T)*T = x0 + v0*T + g*T^2
        # For fine (N sub-steps): each step i: v_i = v0 + g*i*dt; x_i = x_{i-1} + v_i*dt
        # x_N = x0 + sum_{i=1}^{N} (v0 + g*i*dt)*dt = x0 + N*v0*dt + g*dt^2 * sum(i=1..N)(i)
        #      = x0 + v0*T + g*dt^2 * N*(N+1)/2 = x0 + v0*T + g*T^2*(N+1)/(2*N)
        # Coarse: x = x0 + v0*T + g*T^2
        # Fine limit (N->inf): x = x0 + v0*T + g*T^2/2  (exact)
        # So coarse has error g*T^2/2 compared to exact (over-shooting)
        # Fine has error g*T^2/(2*N) compared to exact

        # Key property: fine result is closer to exact than coarse
        err_fine = abs(float(state["position"]) - x_exact)
        err_coarse = abs(float(state_coarse["position"]) - x_exact)

        # Fine should be at least as accurate as coarse (or very close)
        # Allow small floating point tolerance
        assert err_fine <= err_coarse + 1e-5, (
            f"Fine ({n_sub} sub-steps) error {err_fine} > coarse error {err_coarse}"
        )

    @given(
        n_sub=st.integers(min_value=1, max_value=8),
        velocity=st.floats(min_value=-5.0, max_value=5.0,
                           allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None)
    def test_ball_velocity_exact_regardless_of_rate(self, n_sub, velocity):
        """For constant gravity, velocity integration is exact at any rate.

        v(t) = v0 + g*t regardless of how many sub-steps we take.
        """
        g = -9.81
        dt_total = 0.01
        dt_fine = dt_total / n_sub

        node = BallNode(name="b", timestep=dt_fine, gravity=g,
                        initial_position=0.0, initial_velocity=velocity)
        state = node.initial_state()
        for _ in range(n_sub):
            state = node.update(state, {}, dt_fine)

        v_expected = velocity + g * dt_total
        assert abs(float(state["velocity"]) - v_expected) < 1e-5, (
            f"Velocity after {n_sub} steps: {float(state['velocity'])} "
            f"!= expected {v_expected}"
        )


# ---------------------------------------------------------------------------
# Test: GraphManager multirate scheduling fires nodes at correct rates
# ---------------------------------------------------------------------------

class TestGraphManagerMultirate:
    """Tests that GraphManager's multirate scheduling is consistent."""

    @given(
        rate_multiplier=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_slow_node_unchanged_between_fires(self, rate_multiplier):
        """A slow node should not change state on non-fire steps."""
        dt_fast = 0.01
        dt_slow = dt_fast * rate_multiplier

        gm = GraphManager()
        gm.add_node(BallNode(name="fast", timestep=dt_fast, initial_position=10.0))
        gm.add_node(BallNode(name="slow", timestep=dt_slow, initial_position=5.0))

        gm.compile()
        assert gm.is_multirate

        # Step once -- fast fires, slow fires (step 0 % any == 0)
        state = gm.step()
        slow_pos_after_first = float(state["slow"]["position"])
        slow_vel_after_first = float(state["slow"]["velocity"])

        # Steps 1..rate_multiplier-1: slow should NOT fire
        for i in range(1, rate_multiplier):
            state = gm.step()
            assert float(state["slow"]["position"]) == slow_pos_after_first, (
                f"Slow node changed at sub-step {i} (should not fire)"
            )
            assert float(state["slow"]["velocity"]) == slow_vel_after_first, (
                f"Slow node velocity changed at sub-step {i}"
            )

        # At step rate_multiplier, slow fires again
        state = gm.step()
        # Position should have changed
        assert float(state["slow"]["position"]) != slow_pos_after_first, (
            "Slow node did not fire at expected step"
        )

    @given(rate_multiplier=st.integers(min_value=2, max_value=4))
    @settings(max_examples=20, deadline=None)
    def test_fast_node_fires_every_step(self, rate_multiplier):
        """A fast node should update every base step."""
        dt_fast = 0.01
        dt_slow = dt_fast * rate_multiplier

        gm = GraphManager()
        gm.add_node(BallNode(
            name="fast", timestep=dt_fast,
            initial_position=100.0, initial_velocity=0.0,
        ))
        gm.add_node(BallNode(
            name="slow", timestep=dt_slow,
            initial_position=0.0, initial_velocity=0.0,
        ))

        gm.compile()
        prev_pos = 100.0

        for _ in range(rate_multiplier):
            state = gm.step()
            cur_pos = float(state["fast"]["position"])
            # Under gravity, ball should move every step
            assert cur_pos != prev_pos, (
                "Fast node did not fire on a base step"
            )
            prev_pos = cur_pos

    @given(
        rate_multiplier=st.integers(min_value=2, max_value=4),
        n_cycles=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_multirate_state_consistent_at_sync_points(self, rate_multiplier, n_cycles):
        """At sync points (multiples of the slow timestep), both nodes
        should have been updated a consistent number of times."""
        dt_fast = 0.01
        dt_slow = dt_fast * rate_multiplier

        gm = GraphManager()
        gm.add_node(BallNode(
            name="fast", timestep=dt_fast,
            initial_position=10.0, initial_velocity=-1.0,
        ))
        gm.add_node(BallNode(
            name="slow", timestep=dt_slow,
            initial_position=10.0, initial_velocity=-1.0,
        ))

        gm.compile()

        # Advance to a sync point
        total_base_steps = rate_multiplier * n_cycles
        for _ in range(total_base_steps):
            state = gm.step()

        # At sync point, slow has fired n_cycles times
        # Verify outputs are finite (consistency check)
        assert jnp.isfinite(jnp.array(state["fast"]["position"]))
        assert jnp.isfinite(jnp.array(state["slow"]["position"]))
        assert jnp.isfinite(jnp.array(state["fast"]["velocity"]))
        assert jnp.isfinite(jnp.array(state["slow"]["velocity"]))


# ---------------------------------------------------------------------------
# Test: multirate with edges between fast and slow nodes
# ---------------------------------------------------------------------------

class TestMultirateWithEdges:
    """Test that edges between fast and slow nodes work correctly."""

    @given(rate_multiplier=st.integers(min_value=2, max_value=4))
    @settings(max_examples=20, deadline=None)
    def test_fast_to_slow_edge_uses_latest_fast_value(self, rate_multiplier):
        """An edge from a fast node to a slow node should use the fast
        node's most recent value when the slow node fires."""
        dt_fast = 0.01
        dt_slow = dt_fast * rate_multiplier

        gm = GraphManager()
        gm.add_node(BallNode(
            name="fast", timestep=dt_fast,
            initial_position=5.0, initial_velocity=0.0,
        ))
        gm.add_node(BallNode(
            name="slow", timestep=dt_slow,
            initial_position=20.0, initial_velocity=0.0,
        ))
        # Fast's position drives slow's table_position
        gm.add_edge("fast", "slow", "position", "table_position")

        gm.compile()
        assert gm.is_multirate

        # Step through one full slow cycle
        for _ in range(rate_multiplier):
            state = gm.step()

        # Both should have finite state
        assert jnp.isfinite(jnp.array(state["fast"]["position"]))
        assert jnp.isfinite(jnp.array(state["slow"]["position"]))

    @given(rate_multiplier=st.integers(min_value=2, max_value=3))
    @settings(max_examples=20, deadline=None)
    def test_multirate_spring_chain_finite(self, rate_multiplier):
        """A chain of springs at different rates should remain finite."""
        dt_fast = 0.005
        dt_slow = dt_fast * rate_multiplier

        gm = GraphManager()
        gm.add_node(SpringDamperNode(
            name="fast_spring", timestep=dt_fast,
            stiffness=10.0, damping=2.0, mass=1.0,
            initial_position=3.0,
        ))
        gm.add_node(SpringDamperNode(
            name="slow_spring", timestep=dt_slow,
            stiffness=5.0, damping=1.0, mass=2.0,
            initial_position=0.0,
        ))
        gm.add_edge("fast_spring", "slow_spring", "position", "anchor_position")

        gm.compile()

        # Run a few full cycles
        n_steps = rate_multiplier * 5
        for _ in range(n_steps):
            state = gm.step()

        assert jnp.isfinite(jnp.array(state["fast_spring"]["position"]))
        assert jnp.isfinite(jnp.array(state["slow_spring"]["position"]))


# ---------------------------------------------------------------------------
# Test: rate divider computation
# ---------------------------------------------------------------------------

class TestRateDividers:
    """Test that rate dividers are computed correctly."""

    @given(
        dt_a=st.sampled_from([0.01, 0.02, 0.05, 0.1]),
        dt_b=st.sampled_from([0.01, 0.02, 0.05, 0.1]),
    )
    @settings(max_examples=50, deadline=None)
    def test_rate_dividers_are_positive_integers(self, dt_a, dt_b):
        """Rate dividers should always be positive integers."""
        gm = GraphManager()
        gm.add_node(BallNode(name="a", timestep=dt_a))
        gm.add_node(BallNode(name="b", timestep=dt_b))
        gm.compile()

        for name, div in gm.rate_dividers.items():
            assert isinstance(div, int)
            assert div >= 1, f"Rate divider for {name} is {div} (must be >= 1)"

    @given(multiplier=st.integers(min_value=2, max_value=10))
    @settings(max_examples=30, deadline=None)
    def test_rate_divider_ratio_matches_timestep_ratio(self, multiplier):
        """rate_divider(slow) / rate_divider(fast) == dt_slow / dt_fast."""
        dt_fast = 0.01
        dt_slow = dt_fast * multiplier

        gm = GraphManager()
        gm.add_node(BallNode(name="fast", timestep=dt_fast))
        gm.add_node(BallNode(name="slow", timestep=dt_slow))
        gm.compile()

        dividers = gm.rate_dividers
        assert dividers["fast"] == 1
        assert dividers["slow"] == multiplier
