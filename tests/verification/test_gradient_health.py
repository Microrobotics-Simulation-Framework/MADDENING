"""Phase 0: Gradient health audit and parameter recovery baseline.

Proves that MADDENING's end-to-end differentiability works before any
accuracy improvements are made.  Serves as the "before" measurement.

Two categories of tests:

1. **Gradient health**: jax.grad through increasingly long rollouts
   (10, 50, 200, 1000 steps) for springs, heat rods, and coupled
   multi-physics.  Checks gradients don't vanish, explode, or become
   NaN.

2. **Parameter recovery** (Tier 1 calibration test): given a reference
   trajectory from "true" parameters, recover those parameters via
   gradient descent.  This uses inline physics (not the GraphManager
   node contract) because node.params are currently Python floats
   that don't participate in JAX autodiff.  The framework-level
   integration is planned for Phase 4.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


# ======================================================================
# Gradient Health Audit
# ======================================================================

class TestGradientHealthSprings:
    """Gradient through coupled springs at various rollout lengths."""

    @pytest.fixture
    def spring_step_fn(self):
        gm = GraphManager()
        gm.add_node(SpringDamperNode("sa", 0.01, stiffness=50.0,
                                      damping=1.0, initial_position=0.0))
        gm.add_node(SpringDamperNode("sb", 0.01, stiffness=50.0,
                                      damping=1.0, initial_position=3.0))
        gm.add_edge("sa", "sb", "position", "anchor_position")
        gm.add_edge("sb", "sa", "position", "anchor_position")
        gm.add_coupling_group(["sa", "sb"], max_iterations=10,
                               tolerance=1e-8)
        gm.compile()
        return gm._build_step_fn(), gm._default_external_inputs()

    @pytest.mark.parametrize("n_steps", [10, 50, 200, 1000])
    def test_grad_finite_at_rollout_length(self, spring_step_fn, n_steps):
        step_fn, ext = spring_step_fn

        def loss(init_pos):
            state = {
                "sa": {"position": init_pos,
                       "velocity": jnp.array(0.0)},
                "sb": {"position": jnp.array(3.0),
                       "velocity": jnp.array(0.0)},
            }
            def body(s, _):
                return step_fn(s, ext), None
            final, _ = jax.lax.scan(body, state, None, length=n_steps)
            return final["sa"]["position"]

        g = jax.grad(loss)(jnp.array(0.0))
        assert jnp.isfinite(g), f"Gradient is not finite at {n_steps} steps"
        assert float(jnp.abs(g)) > 1e-10, f"Gradient vanished at {n_steps} steps"
        assert float(jnp.abs(g)) < 1e10, f"Gradient exploded at {n_steps} steps"


class TestGradientHealthHeat:
    """Gradient through coupled heat rods at various rollout lengths."""

    @pytest.fixture
    def heat_step_fn(self):
        gm = GraphManager()
        gm.add_node(HeatNode("rod_a", 0.001, n_cells=10,
                              thermal_diffusivity=0.01,
                              initial_temperature=100.0))
        gm.add_node(HeatNode("rod_b", 0.001, n_cells=10,
                              thermal_diffusivity=0.01,
                              initial_temperature=0.0))
        gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                     transform=lambda T: T[-2])
        gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                     transform=lambda T: T[1])
        gm.add_coupling_group(["rod_a", "rod_b"], max_iterations=10,
                               tolerance=1e-8)
        gm.compile()
        return gm._build_step_fn(), gm._default_external_inputs()

    @pytest.mark.parametrize("n_steps", [10, 50, 200])
    def test_grad_finite_at_rollout_length(self, heat_step_fn, n_steps):
        step_fn, ext = heat_step_fn

        def loss(init_temp_scale):
            T_a = jnp.ones(10) * init_temp_scale * 100.0
            T_b = jnp.zeros(10)
            state = {
                "rod_a": {"temperature": T_a},
                "rod_b": {"temperature": T_b},
            }
            def body(s, _):
                return step_fn(s, ext), None
            final, _ = jax.lax.scan(body, state, None, length=n_steps)
            return jnp.sum(final["rod_a"]["temperature"])

        g = jax.grad(loss)(jnp.array(1.0))
        assert jnp.isfinite(g), f"Gradient is not finite at {n_steps} steps"
        assert float(jnp.abs(g)) > 1e-10, f"Gradient vanished at {n_steps} steps"
        assert float(jnp.abs(g)) < 1e10, f"Gradient exploded at {n_steps} steps"


class TestGradientHealthMultiPhysics:
    """Gradient through ball + spring + table (multi-physics)."""

    @pytest.fixture
    def multi_step_fn(self):
        gm = GraphManager()
        gm.add_node(TableNode("table", 0.01, position=0.0))
        gm.add_node(BallNode("ball", 0.01, initial_position=3.0,
                              initial_velocity=0.0, elasticity=0.6))
        gm.add_node(SpringDamperNode("spring", 0.01, stiffness=80.0,
                                      damping=3.0, mass=0.5,
                                      rest_length=1.0,
                                      initial_position=2.0))
        gm.add_edge("table", "ball", "position", "table_position")
        gm.add_edge("ball", "spring", "position", "anchor_position")
        gm.compile()
        return gm._build_step_fn(), gm._default_external_inputs()

    @pytest.mark.parametrize("n_steps", [10, 50, 200, 1000])
    def test_grad_finite_at_rollout_length(self, multi_step_fn, n_steps):
        step_fn, ext = multi_step_fn

        def loss(init_vel):
            state = {
                "table": {"position": jnp.array(0.0)},
                "ball": {"position": jnp.array(3.0), "velocity": init_vel},
                "spring": {"position": jnp.array(2.0),
                           "velocity": jnp.array(0.0)},
            }
            def body(s, _):
                return step_fn(s, ext), None
            final, _ = jax.lax.scan(body, state, None, length=n_steps)
            return final["spring"]["position"]

        g = jax.grad(loss)(jnp.array(0.0))
        assert jnp.isfinite(g), f"Gradient is not finite at {n_steps} steps"


# ======================================================================
# Parameter Recovery (Tier 1 Calibration Test)
# ======================================================================

class TestParameterRecovery:
    """Recover physical parameters from a reference trajectory.

    Uses inline spring physics with JAX-traced parameters (k, c)
    rather than the GraphManager node contract, because node.params
    are currently Python floats that don't participate in autodiff.

    This proves the concept: differentiable simulation + gradient
    descent can recover physical parameters from trajectory data.
    """

    @staticmethod
    def _spring_step(state, k, c, m, rest, anchor, dt):
        """Single spring step with JAX-traced parameters."""
        pos, vel = state["position"], state["velocity"]
        force = -k * (pos - anchor - rest) - c * vel
        acc = force / m
        vel_new = vel + acc * dt
        pos_new = pos + vel_new * dt
        return {"position": pos_new, "velocity": vel_new}

    @staticmethod
    def _coupled_step(state, k, c, dt=0.01, m=1.0, rest=1.0):
        """One step of two bidirectionally-coupled springs."""
        # Spring A uses B's position as anchor
        sa = TestParameterRecovery._spring_step(
            state["sa"], k, c, m, rest, state["sb"]["position"], dt
        )
        # Spring B uses A's NEW position as anchor (Gauss-Seidel)
        sb = TestParameterRecovery._spring_step(
            state["sb"], k, c, m, rest, sa["position"], dt
        )
        return {"sa": sa, "sb": sb}

    def _generate_reference(self, k_true, c_true, n_steps):
        """Generate a reference trajectory with true parameters."""
        state = {
            "sa": {"position": jnp.array(0.0),
                   "velocity": jnp.array(0.0)},
            "sb": {"position": jnp.array(3.0),
                   "velocity": jnp.array(0.0)},
        }
        trajectory = []
        for _ in range(n_steps):
            state = self._coupled_step(state, k_true, c_true)
            trajectory.append(state["sa"]["position"])
        return jnp.stack(trajectory)

    def test_parameter_recovery_springs(self):
        """Recover spring stiffness and damping from trajectory data."""
        # True parameters (moderate stiffness for good conditioning)
        k_true = jnp.array(30.0)
        c_true = jnp.array(2.0)
        n_steps = 100

        # Generate reference trajectory
        ref_traj = self._generate_reference(k_true, c_true, n_steps)

        # Loss: trajectory deviation from reference
        def loss_fn(params):
            k, c = params
            state = {
                "sa": {"position": jnp.array(0.0),
                       "velocity": jnp.array(0.0)},
                "sb": {"position": jnp.array(3.0),
                       "velocity": jnp.array(0.0)},
            }
            def body(s, _):
                s_new = self._coupled_step(s, k, c)
                return s_new, s_new["sa"]["position"]
            _, traj = jax.lax.scan(body, state, None, length=n_steps)
            return jnp.mean((traj - ref_traj) ** 2)

        # Start with wrong parameters
        params = jnp.array([15.0, 5.0])  # k=15 (true=30), c=5 (true=2)

        # Adam optimizer
        lr = 1.0
        m = jnp.zeros(2)
        v = jnp.zeros(2)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        grad_fn = jax.jit(jax.grad(loss_fn))
        loss_fn_jit = jax.jit(loss_fn)

        initial_loss = float(loss_fn_jit(params))

        for i in range(500):
            g = grad_fn(params)
            assert jnp.all(jnp.isfinite(g)), f"NaN gradient at step {i}"
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** (i + 1))
            v_hat = v / (1 - beta2 ** (i + 1))
            params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
            params = jnp.maximum(params, 0.1)

        final_loss = float(loss_fn_jit(params))
        k_recovered, c_recovered = float(params[0]), float(params[1])

        # Verification gates
        assert final_loss < initial_loss * 1e-3, (
            f"Loss didn't drop 3 orders: {initial_loss:.2e} -> {final_loss:.2e}"
        )
        assert abs(k_recovered - 30.0) / 30.0 < 0.05, (
            f"k not recovered: {k_recovered:.2f} (true=30.0)"
        )
        assert abs(c_recovered - 2.0) / 2.0 < 0.10, (
            f"c not recovered: {c_recovered:.2f} (true=2.0)"
        )

    def test_parameter_recovery_different_ics(self):
        """Recovery works from multiple initial conditions."""
        k_true = jnp.array(25.0)
        c_true = jnp.array(3.0)
        n_steps = 80

        # Generate references from 3 different ICs
        ics = [
            (0.0, 3.0),
            (1.0, 4.0),
            (-1.0, 2.0),
        ]
        refs = []
        for pos_a, pos_b in ics:
            state = {
                "sa": {"position": jnp.array(pos_a),
                       "velocity": jnp.array(0.0)},
                "sb": {"position": jnp.array(pos_b),
                       "velocity": jnp.array(0.0)},
            }
            def body(s, _):
                s_new = self._coupled_step(s, k_true, c_true)
                return s_new, s_new["sa"]["position"]
            _, traj = jax.lax.scan(body, state, None, length=n_steps)
            refs.append((pos_a, pos_b, traj))

        # Multi-trajectory loss
        def loss_fn(params):
            k, c = params
            total = jnp.array(0.0)
            for pos_a, pos_b, ref_traj in refs:
                state = {
                    "sa": {"position": jnp.array(pos_a),
                           "velocity": jnp.array(0.0)},
                    "sb": {"position": jnp.array(pos_b),
                           "velocity": jnp.array(0.0)},
                }
                def body(s, _):
                    s_new = self._coupled_step(s, k, c)
                    return s_new, s_new["sa"]["position"]
                _, traj = jax.lax.scan(body, state, None, length=n_steps)
                total = total + jnp.mean((traj - ref_traj) ** 2)
            return total / len(refs)

        params = jnp.array([12.0, 8.0])

        # Adam
        lr = 1.0
        m = jnp.zeros(2)
        v = jnp.zeros(2)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        grad_fn = jax.jit(jax.grad(loss_fn))
        loss_fn_jit = jax.jit(loss_fn)
        initial_loss = float(loss_fn_jit(params))

        for i in range(500):
            g = grad_fn(params)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** (i + 1))
            v_hat = v / (1 - beta2 ** (i + 1))
            params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
            params = jnp.maximum(params, 0.1)

        final_loss = float(loss_fn_jit(params))
        k_rec, c_rec = float(params[0]), float(params[1])

        assert final_loss < initial_loss * 0.001
        assert abs(k_rec - 25.0) / 25.0 < 0.05
        assert abs(c_rec - 3.0) / 3.0 < 0.10

    def test_gradient_through_graphmanager_scan(self):
        """Verify that jax.grad works through GraphManager's run_scan
        w.r.t. initial conditions (the currently supported path)."""
        gm = GraphManager()
        gm.add_node(SpringDamperNode("sa", 0.01, stiffness=50.0,
                                      damping=1.0, initial_position=0.0))
        gm.add_node(SpringDamperNode("sb", 0.01, stiffness=50.0,
                                      damping=1.0, initial_position=3.0))
        gm.add_edge("sa", "sb", "position", "anchor_position")
        gm.add_edge("sb", "sa", "position", "anchor_position")
        gm.add_coupling_group(["sa", "sb"], max_iterations=10,
                               tolerance=1e-8)
        gm.compile()
        step_fn = gm._build_step_fn()
        ext = gm._default_external_inputs()

        def loss(init_pos):
            state = {
                "sa": {"position": init_pos,
                       "velocity": jnp.array(0.0)},
                "sb": {"position": jnp.array(3.0),
                       "velocity": jnp.array(0.0)},
            }
            def body(s, _):
                return step_fn(s, ext), None
            final, _ = jax.lax.scan(body, state, None, length=200)
            return final["sa"]["position"] ** 2

        # Gradient should be informative
        g = jax.grad(loss)(jnp.array(0.5))
        assert jnp.isfinite(g)
        assert float(jnp.abs(g)) > 1e-6
