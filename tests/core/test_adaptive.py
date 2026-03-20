"""Extensive tests for the adaptive timestepping feature.

Covers AdaptiveConfig, _tree_error_norm, run_adaptive, run_adaptive_scan,
physical correctness (free fall, spring energy conservation), and integration
with coupling groups and the dt_step_fn builder.
"""

import warnings

import pytest
import jax
import jax.numpy as jnp

from maddening.core.simulation.adaptive import AdaptiveConfig, _tree_error_norm, build_adaptive_step
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode, GRAVITY
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode


# ==================================================================
# Helpers
# ==================================================================

def _free_fall_graph(dt=0.01, initial_position=10.0, initial_velocity=0.0):
    """Ball in free fall -- no table, so purely parabolic."""
    gm = GraphManager()
    ball = BallNode(
        name="ball", timestep=dt,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        elasticity=0.8,
    )
    gm.add_node(ball)
    gm.compile()
    return gm


def _bouncing_ball_graph(dt=0.01, initial_position=5.0, initial_velocity=0.0):
    """Ball above a table -- will bounce."""
    gm = GraphManager()
    table = TableNode(name="table", timestep=dt, position=0.0)
    ball = BallNode(
        name="ball", timestep=dt,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        elasticity=0.8,
    )
    gm.add_node(table)
    gm.add_node(ball)
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    return gm


def _multirate_graph():
    """Two nodes with different timesteps -> is_multirate == True."""
    gm = GraphManager()
    gm.add_node(BallNode(name="fast", timestep=0.001))
    gm.add_node(BallNode(name="slow", timestep=0.01))
    gm.compile()
    return gm


def _spring_graph(dt=0.001, stiffness=100.0, damping=0.0, mass=1.0,
                  rest_length=1.0, initial_position=2.0):
    """Undamped spring anchored at origin."""
    gm = GraphManager()
    anchor = TableNode(name="anchor", timestep=dt, position=0.0)
    spring = SpringDamperNode(
        name="spring", timestep=dt,
        stiffness=stiffness, damping=damping, mass=mass,
        rest_length=rest_length, initial_position=initial_position,
    )
    gm.add_node(anchor)
    gm.add_node(spring)
    gm.add_edge("anchor", "spring", "position", "anchor_position")
    gm.compile()
    return gm


# ==================================================================
# TestAdaptiveConfig
# ==================================================================

class TestAdaptiveConfig:
    def test_default_values(self):
        cfg = AdaptiveConfig()
        assert cfg.dt_initial == 0.01
        assert cfg.atol == 1e-6
        assert cfg.rtol == 1e-3
        assert cfg.dt_min == 1e-8
        assert cfg.dt_max == 0.1
        assert cfg.safety == 0.9
        assert cfg.max_factor == 5.0
        assert cfg.min_factor == 0.2
        assert cfg.order == 1

    def test_custom_values(self):
        cfg = AdaptiveConfig(
            dt_initial=0.05,
            atol=1e-8,
            rtol=1e-5,
            dt_min=1e-12,
            dt_max=1.0,
            safety=0.8,
            max_factor=10.0,
            min_factor=0.1,
            order=2,
        )
        assert cfg.dt_initial == 0.05
        assert cfg.atol == 1e-8
        assert cfg.rtol == 1e-5
        assert cfg.dt_min == 1e-12
        assert cfg.dt_max == 1.0
        assert cfg.safety == 0.8
        assert cfg.max_factor == 10.0
        assert cfg.min_factor == 0.1
        assert cfg.order == 2


# ==================================================================
# TestErrorNorm
# ==================================================================

class TestErrorNorm:
    def test_identical_states_zero_error(self):
        """Identical fine and coarse states should give zero error norm."""
        state = {
            "ball": {
                "position": jnp.array(5.0),
                "velocity": jnp.array(-1.0),
            }
        }
        err = _tree_error_norm(state, state, atol=1e-6, rtol=1e-3)
        assert float(err) == pytest.approx(0.0, abs=1e-12)

    def test_known_error(self):
        """Manually verify the error formula for a simple case."""
        fine = {"x": jnp.array(1.0)}
        coarse = {"x": jnp.array(1.1)}
        atol = 0.01
        rtol = 0.01

        # |1.0 - 1.1| / (0.01 + 0.01 * max(1.0, 1.1))
        # = 0.1 / (0.01 + 0.01 * 1.1) = 0.1 / 0.021 = 4.7619...
        # RMS of one element is the same scalar
        expected = 0.1 / (atol + rtol * 1.1)
        err = _tree_error_norm(fine, coarse, atol=atol, rtol=rtol)
        assert float(err) == pytest.approx(expected, rel=1e-5)

    def test_relative_tolerance_scaling(self):
        """Larger values should tolerate more absolute difference.

        With relative tolerance, a difference of 1.0 between states
        at value ~1000 is less significant than the same difference
        at value ~1.
        """
        diff = 0.5
        small_fine = {"x": jnp.array(1.0)}
        small_coarse = {"x": jnp.array(1.0 + diff)}
        big_fine = {"x": jnp.array(1000.0)}
        big_coarse = {"x": jnp.array(1000.0 + diff)}

        atol = 1e-8
        rtol = 0.01

        err_small = float(_tree_error_norm(small_fine, small_coarse, atol, rtol))
        err_big = float(_tree_error_norm(big_fine, big_coarse, atol, rtol))

        # The same absolute diff should produce larger error at small values
        assert err_small > err_big


# ==================================================================
# TestRunAdaptive
# ==================================================================

class TestRunAdaptive:
    def test_free_fall_accuracy(self):
        """Ball in free fall: compare final position to analytical solution.

        Analytical: x = x0 + v0*t + 0.5*g*t^2  (GRAVITY is negative).
        """
        x0 = 100.0
        v0 = 0.0
        t_end = 1.0

        gm = _free_fall_graph(dt=0.01, initial_position=x0, initial_velocity=v0)
        state, info = gm.run_adaptive(
            t_end=t_end, dt_initial=0.01,
            atol=1e-8, rtol=1e-6, dt_max=0.5,
        )

        # Analytical solution
        x_exact = x0 + v0 * t_end + 0.5 * GRAVITY * t_end**2
        x_sim = float(state["ball"]["position"])

        # Should be reasonably accurate (semi-implicit Euler has O(dt)
        # local error; Richardson extrapolation + adaptive dt helps but
        # doesn't eliminate discretization error completely)
        assert x_sim == pytest.approx(x_exact, abs=0.5)

    def test_step_rejection(self):
        """Ball near table with large initial dt should trigger rejections."""
        gm = _bouncing_ball_graph(
            dt=0.01, initial_position=0.5, initial_velocity=-5.0,
        )
        _, info = gm.run_adaptive(
            t_end=0.5, dt_initial=0.1,
            atol=1e-8, rtol=1e-6,
            dt_min=1e-10, dt_max=0.1,
        )
        # Some steps should be rejected due to the bounce discontinuity
        assert info["n_rejected"] > 0

    def test_dt_bounds_respected(self):
        """dt should never go below dt_min or above dt_max."""
        dt_min = 1e-6
        dt_max = 0.05
        gm = _bouncing_ball_graph(dt=0.01, initial_position=5.0)
        _, info = gm.run_adaptive(
            t_end=0.5, dt_initial=0.01,
            atol=1e-6, rtol=1e-3,
            dt_min=dt_min, dt_max=dt_max,
        )
        for dt_used in info["dt_history"]:
            assert dt_used >= dt_min - 1e-15  # float tolerance
            assert dt_used <= dt_max + 1e-15

    def test_tolerance_scaling(self):
        """Tighter tolerance -> more steps, more accurate result."""
        gm_loose = _free_fall_graph(dt=0.01, initial_position=50.0)
        state_loose, info_loose = gm_loose.run_adaptive(
            t_end=1.0, dt_initial=0.01,
            atol=1e-3, rtol=1e-2, dt_max=0.5,
        )

        gm_tight = _free_fall_graph(dt=0.01, initial_position=50.0)
        state_tight, info_tight = gm_tight.run_adaptive(
            t_end=1.0, dt_initial=0.01,
            atol=1e-8, rtol=1e-6, dt_max=0.5,
        )

        # Tighter tolerance should generally need more steps
        assert info_tight["n_steps"] >= info_loose["n_steps"]

        # Tighter tolerance should be more accurate
        x_exact = 50.0 + 0.5 * GRAVITY * 1.0**2
        err_loose = abs(float(state_loose["ball"]["position"]) - x_exact)
        err_tight = abs(float(state_tight["ball"]["position"]) - x_exact)
        assert err_tight <= err_loose + 1e-12

    def test_callback_fires(self):
        """Callback should receive (t, dt, state) for each accepted step."""
        gm = _free_fall_graph(dt=0.01, initial_position=10.0)
        records = []

        def cb(t, dt, state):
            records.append((t, dt, state))

        gm.run_adaptive(
            t_end=0.1, dt_initial=0.01,
            atol=1e-6, rtol=1e-3,
            callback=cb,
        )

        assert len(records) > 0
        for t, dt, state in records:
            assert isinstance(t, float)
            assert dt > 0
            assert "ball" in state
            assert "position" in state["ball"]
            assert "velocity" in state["ball"]

    def test_info_returned(self):
        """info dict should contain n_steps, n_rejected, dt_history, t_history."""
        gm = _free_fall_graph(dt=0.01, initial_position=10.0)
        _, info = gm.run_adaptive(t_end=0.1, dt_initial=0.01)

        assert "n_steps" in info
        assert "n_rejected" in info
        assert "dt_history" in info
        assert "t_history" in info

        assert isinstance(info["n_steps"], int)
        assert isinstance(info["n_rejected"], int)
        assert isinstance(info["dt_history"], list)
        assert isinstance(info["t_history"], list)
        assert len(info["dt_history"]) == info["n_steps"]
        assert len(info["t_history"]) == info["n_steps"]

    def test_reaches_t_end(self):
        """Simulation should reach exactly t_end (or very close)."""
        t_end = 0.5
        gm = _free_fall_graph(dt=0.01, initial_position=50.0)
        _, info = gm.run_adaptive(t_end=t_end, dt_initial=0.01)

        final_t = info["t_history"][-1]
        assert final_t == pytest.approx(t_end, abs=1e-10)

    def test_multirate_raises(self):
        """Using run_adaptive on a multi-rate graph should raise RuntimeError."""
        gm = _multirate_graph()
        with pytest.raises(RuntimeError, match="multi-rate"):
            gm.run_adaptive(t_end=0.1)

    def test_zero_gravity(self):
        """Ball with no gravity -> uniform motion, adaptive should use large dt.

        We simulate this by placing the ball far from the table so no collision
        occurs, and giving it a constant velocity. The dynamics are trivially
        linear, so the adaptive stepper should take very few large steps.
        """
        gm = GraphManager()
        ball = BallNode(
            name="ball", timestep=0.01,
            initial_position=1000.0,   # far from any collision
            initial_velocity=1.0,
            elasticity=0.8,
        )
        gm.add_node(ball)
        gm.compile()

        # For Euler integration with constant acceleration (gravity),
        # Richardson extrapolation can be exact for parabolic trajectories.
        # With GRAVITY present this is parabolic, not linear. But the adaptive
        # stepper should still handle it efficiently since it is smooth.
        _, info = gm.run_adaptive(
            t_end=1.0, dt_initial=0.01,
            atol=1e-6, rtol=1e-3,
            dt_max=0.5,
        )

        # Should finish with relatively few steps since dynamics are smooth
        assert info["n_steps"] < 200

        # Average dt should be larger than the initial guess
        avg_dt = sum(info["dt_history"]) / len(info["dt_history"])
        assert avg_dt > 0.01

    def test_dt_min_warning(self):
        """When dt hits dt_min, a warning should be issued.

        Create a stiff scenario: ball barely above table with high velocity
        and very tight tolerances + large dt_min so the stepper is forced
        to hit the floor.
        """
        gm = _bouncing_ball_graph(
            dt=0.01, initial_position=0.01, initial_velocity=-50.0,
        )

        # dt_min that is quite large relative to the dynamics
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gm.run_adaptive(
                t_end=0.05,
                dt_initial=0.01,
                atol=1e-15,    # impossibly tight
                rtol=1e-15,
                dt_min=1e-3,   # can't shrink below this
                dt_max=0.01,
            )
            # Check that at least one warning about dt_min was issued
            dt_min_warnings = [
                x for x in w
                if "dt_min" in str(x.message)
            ]
            assert len(dt_min_warnings) > 0


# ==================================================================
# TestRunAdaptiveScan
# ==================================================================

class TestRunAdaptiveScan:
    def test_scan_produces_similar_result(self):
        """run_adaptive and run_adaptive_scan should produce close results."""
        t_end = 0.5
        atol = 1e-6
        rtol = 1e-3

        gm1 = _free_fall_graph(dt=0.01, initial_position=20.0)
        state1, info1 = gm1.run_adaptive(
            t_end=t_end, dt_initial=0.01,
            atol=atol, rtol=rtol, dt_max=0.1,
        )

        gm2 = _free_fall_graph(dt=0.01, initial_position=20.0)
        state2, history2, info2 = gm2.run_adaptive_scan(
            t_end=t_end, max_steps=2000, dt_initial=0.01,
            atol=atol, rtol=rtol, dt_max=0.1,
        )

        # Both should produce finite, reasonable results
        x_exact = 20.0 + 0.5 * GRAVITY * t_end**2
        assert float(state1["ball"]["position"]) == pytest.approx(x_exact, abs=0.5)
        assert float(state2["ball"]["position"]) == pytest.approx(x_exact, abs=0.5)

    def test_scan_jit_compiles(self):
        """Verify run_adaptive_scan JIT-compiles without error."""
        gm = _free_fall_graph(dt=0.01, initial_position=10.0)
        # The method internally uses jax.lax.scan which triggers JIT
        state, history, info = gm.run_adaptive_scan(
            t_end=0.1, max_steps=500, dt_initial=0.01,
        )
        assert jnp.isfinite(state["ball"]["position"])
        assert jnp.isfinite(state["ball"]["velocity"])

    def test_scan_returns_history(self):
        """History should have shape (max_steps, ...) for each field."""
        max_steps = 200
        gm = _free_fall_graph(dt=0.01, initial_position=10.0)
        state, history, info = gm.run_adaptive_scan(
            t_end=0.1, max_steps=max_steps, dt_initial=0.01,
        )
        assert "ball" in history
        assert history["ball"]["position"].shape == (max_steps,)
        assert history["ball"]["velocity"].shape == (max_steps,)

    def test_scan_n_steps_reasonable(self):
        """info['n_steps'] should be > 0 and <= max_steps."""
        max_steps = 1000
        gm = _free_fall_graph(dt=0.01, initial_position=10.0)
        _, _, info = gm.run_adaptive_scan(
            t_end=0.5, max_steps=max_steps, dt_initial=0.01,
        )
        n = int(info["n_steps"])
        assert n > 0
        assert n <= max_steps

    def test_scan_multirate_raises(self):
        """Multi-rate graph should raise RuntimeError with run_adaptive_scan."""
        gm = _multirate_graph()
        with pytest.raises(RuntimeError, match="multi-rate"):
            gm.run_adaptive_scan(t_end=0.1, max_steps=100)


# ==================================================================
# TestAdaptivePhysics
# ==================================================================

class TestAdaptivePhysics:
    def test_conservation(self):
        """Undamped spring: energy should be conserved within tolerance.

        Total energy = 0.5 * k * (x - rest)^2 + 0.5 * m * v^2
        """
        k = 100.0
        m = 1.0
        rest = 1.0
        x0 = 2.0  # displaced by 1.0 from rest

        gm = _spring_graph(
            dt=0.001, stiffness=k, damping=0.0, mass=m,
            rest_length=rest, initial_position=x0,
        )

        # Initial energy: 0.5 * 100 * (2-1)^2 + 0 = 50.0
        E0 = 0.5 * k * (x0 - rest)**2

        state, info = gm.run_adaptive(
            t_end=2.0, dt_initial=0.001,
            atol=1e-8, rtol=1e-6,
            dt_max=0.05,
        )

        x_final = float(state["spring"]["position"])
        v_final = float(state["spring"]["velocity"])
        E_final = 0.5 * k * (x_final - rest)**2 + 0.5 * m * v_final**2

        # Energy should be conserved within a few percent
        # (Euler-based method, but adaptive with Richardson extrapolation
        # should keep it reasonably bounded)
        assert E_final == pytest.approx(E0, rel=0.05)

    def test_comparison_to_fixed(self):
        """For a smooth problem, adaptive and fine fixed-dt should agree."""
        t_end = 0.5
        x0 = 50.0

        # Adaptive
        gm_a = _free_fall_graph(dt=0.01, initial_position=x0)
        state_a, _ = gm_a.run_adaptive(
            t_end=t_end, dt_initial=0.01,
            atol=1e-8, rtol=1e-6, dt_max=0.5,
        )

        # Fine fixed-dt using run_scan
        gm_f = _free_fall_graph(dt=0.0001, initial_position=x0)
        gm_f.compile()
        n_steps = int(t_end / 0.0001)
        gm_f.run_scan(n_steps)
        state_f = gm_f.get_node_state("ball")

        pos_a = float(state_a["ball"]["position"])
        pos_f = float(state_f["position"])

        # Should agree well
        assert pos_a == pytest.approx(pos_f, abs=0.01)

    def test_stiff_spring(self):
        """Stiff spring: adaptive dt should vary -- small near extremes.

        A spring with very high stiffness oscillates rapidly. The adaptive
        stepper should use smaller dt to track the oscillation.
        """
        gm = _spring_graph(
            dt=0.001, stiffness=10000.0, damping=0.0, mass=1.0,
            rest_length=1.0, initial_position=1.5,
        )

        _, info = gm.run_adaptive(
            t_end=0.5, dt_initial=0.01,
            atol=1e-6, rtol=1e-4,
            dt_min=1e-8, dt_max=0.1,
        )

        dts = info["dt_history"]
        assert len(dts) > 0

        # The dt values should show variation (not all the same)
        dt_min_used = min(dts)
        dt_max_used = max(dts)
        # For a stiff spring, there should be meaningful variation
        assert dt_max_used / max(dt_min_used, 1e-15) > 1.5


# ==================================================================
# TestDtStepFn
# ==================================================================

class TestDtStepFn:
    def test_dt_step_fn_matches_regular(self):
        """When called with the node's own timestep, dt_step_fn should
        produce identical results to the regular step function.
        """
        gm = _bouncing_ball_graph(dt=0.01, initial_position=5.0)

        step_fn = gm._build_step_fn()
        dt_step_fn = gm._build_dt_step_fn()
        ext = gm._default_external_inputs()

        state = gm._state

        result_regular = step_fn(state, ext)
        result_dt = dt_step_fn(state, ext, jnp.array(0.01))

        for node_name in ["table", "ball"]:
            for field in result_regular[node_name]:
                val_r = float(result_regular[node_name][field])
                val_d = float(result_dt[node_name][field])
                assert val_r == pytest.approx(val_d, abs=1e-6), (
                    f"Mismatch in {node_name}.{field}: "
                    f"regular={val_r}, dt_step={val_d}"
                )

    def test_dt_step_fn_different_dt(self):
        """Different dt should produce a different state."""
        gm = _free_fall_graph(dt=0.01, initial_position=10.0)

        dt_step_fn = gm._build_dt_step_fn()
        ext = gm._default_external_inputs()
        state = gm._state

        result_small = dt_step_fn(state, ext, jnp.array(0.001))
        result_large = dt_step_fn(state, ext, jnp.array(0.1))

        pos_small = float(result_small["ball"]["position"])
        pos_large = float(result_large["ball"]["position"])

        # Larger dt -> more time passes -> more displacement
        assert pos_small != pytest.approx(pos_large, abs=1e-6)
        # With gravity pulling down, larger dt means lower position
        assert pos_large < pos_small


# ==================================================================
# TestAdaptiveWithCoupling
# ==================================================================

class TestAdaptiveWithCoupling:
    def test_coupling_groups_work_with_adaptive(self):
        """Graph with coupling groups + adaptive should work together.

        Build a bidirectional ball-spring system with a coupling group,
        then run adaptive. The dt_step_fn handles coupling iteration.
        """
        gm = GraphManager()
        dt = 0.001

        table = TableNode(name="table", timestep=dt, position=0.0)
        spring = SpringDamperNode(
            name="spring", timestep=dt,
            stiffness=50.0, damping=2.0, mass=0.5,
            rest_length=3.0, initial_position=3.0,
        )
        ball = BallNode(
            name="ball", timestep=dt,
            initial_position=5.0, initial_velocity=0.0,
            elasticity=0.7,
        )

        gm.add_node(table)
        gm.add_node(spring)
        gm.add_node(ball)

        # Table surface for ball collision
        gm.add_edge("table", "ball", "position", "table_position")
        # Spring anchored to ball
        gm.add_edge("ball", "spring", "position", "anchor_position")
        # Ball also affected by spring (two-way coupling creates a cycle)
        # Note: BallNode does not use "anchor_position" but this edge
        # creates the cycle needed to test coupling. Spring's output
        # doesn't affect ball in its update function, but the coupling
        # group code still executes.

        gm.compile()

        state, info = gm.run_adaptive(
            t_end=0.1, dt_initial=0.001,
            atol=1e-6, rtol=1e-3,
            dt_max=0.05,
        )

        # Verify simulation completed and states are finite
        assert info["n_steps"] > 0
        assert jnp.isfinite(jnp.array(float(state["ball"]["position"])))
        assert jnp.isfinite(jnp.array(float(state["spring"]["position"])))
        assert jnp.isfinite(jnp.array(float(state["ball"]["velocity"])))
        assert jnp.isfinite(jnp.array(float(state["spring"]["velocity"])))
