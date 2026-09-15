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
   gradient descent through the graph's own step function, with
   ``jax.grad`` taken with respect to ``GraphManager.params`` (node
   constants as a traced pytree) — for a single spring and through a
   coupling group.
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

    Goes through the real graph: ``jax.grad`` of a trajectory loss with
    respect to ``gm.params`` — the third pytree of the compiled step —
    for a single spring and for a two-spring coupling group solved by
    the default early-exit IFT solver.  This is the proof that node
    constants are traced, differentiable inputs rather than closure
    constants baked into the jit.
    """

    K_TRUE, C_TRUE = 30.0, 2.0
    N_STEPS = 100

    @staticmethod
    def _single(k, c):
        gm = GraphManager()
        gm.add_node(SpringDamperNode("s", 0.01, stiffness=k, damping=c,
                                     mass=1.0, rest_length=1.0,
                                     initial_position=0.0))
        gm.compile()
        return gm

    @staticmethod
    def _coupled(k, c):
        gm = GraphManager()
        gm.add_node(SpringDamperNode("sa", 0.01, stiffness=k, damping=c,
                                     mass=1.0, rest_length=1.0,
                                     initial_position=0.0))
        gm.add_node(SpringDamperNode("sb", 0.01, stiffness=k, damping=c,
                                     mass=1.0, rest_length=1.0,
                                     initial_position=3.0))
        gm.add_edge("sa", "sb", "position", "anchor_position")
        gm.add_edge("sb", "sa", "position", "anchor_position")
        gm.add_coupling_group(["sa", "sb"], max_iterations=20, tolerance=1e-8)
        gm.compile()
        return gm

    @staticmethod
    def _positions(gm, params, n_steps, nodes, state0=None):
        """Stacked positions of ``nodes`` over an n-step rollout through
        the graph's step function with an explicit params pytree."""
        step_fn = gm._build_step_fn()
        ext = gm._default_external_inputs()

        def body(s, _):
            s = step_fn(s, ext, params)
            return s, jnp.stack([s[n]["position"] for n in nodes])

        init = gm._state if state0 is None else state0
        _, traj = jax.lax.scan(body, init, None, length=n_steps)
        return traj

    FIT = ("stiffness", "damping")

    @classmethod
    def _adam(cls, loss_fn, params, n_iter, lr=0.3):
        """Adam on the ``FIT`` entries only.  Updating every float in the
        node pytree would also move ``mass``, and (k, c, m) scaled
        together leaves the trajectory unchanged — the unidentifiable
        direction ``maddening.sysid.fim`` reports."""
        grad_fn = jax.jit(jax.grad(loss_fn))
        mask = {n: {k: (k in cls.FIT) for k in node} for n, node in params.items()}
        m = jax.tree.map(jnp.zeros_like, params)
        v = jax.tree.map(jnp.zeros_like, params)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(1, n_iter + 1):
            g = grad_fn(params)
            assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g)), (
                f"non-finite gradient at iteration {i}"
            )
            m = jax.tree.map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
            v = jax.tree.map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ ** 2, v, g)
            params = jax.tree.map(
                lambda p_, m_, v_, fit: jnp.maximum(
                    p_ - lr * (m_ / (1 - b1 ** i)) / (jnp.sqrt(v_ / (1 - b2 ** i)) + eps),
                    0.1,
                ) if fit else p_,
                params, m, v, mask,
            )
        return params

    def _recover(self, build, nodes, k0, c0):
        gm = build(self.K_TRUE, self.C_TRUE)
        ref = self._positions(gm, gm.params, self.N_STEPS, nodes)

        def loss_fn(node_params):
            return jnp.mean(
                (self._positions(gm, {"nodes": node_params, "mappings": {}},
                                 self.N_STEPS, nodes) - ref) ** 2
            )

        start = jax.tree.map(lambda x: x, gm.params["nodes"])
        for n in nodes:
            start[n]["stiffness"] = jnp.asarray(k0, dtype=jnp.float32)
            start[n]["damping"] = jnp.asarray(c0, dtype=jnp.float32)
        loss_jit = jax.jit(loss_fn)
        initial_loss = float(loss_jit(start))
        fitted = self._adam(loss_fn, start, 500)
        final_loss = float(loss_jit(fitted))
        assert final_loss < initial_loss * 1e-3, (initial_loss, final_loss)
        for n in nodes:
            k, c = float(fitted[n]["stiffness"]), float(fitted[n]["damping"])
            assert abs(k - self.K_TRUE) / self.K_TRUE < 0.05, f"{n}: k={k}"
            assert abs(c - self.C_TRUE) / self.C_TRUE < 0.10, f"{n}: c={c}"

    def test_parameter_recovery_single_spring(self):
        """k, c recovered through gm's step with the params pytree."""
        self._recover(self._single, ("s",), 15.0, 5.0)

    def test_parameter_recovery_coupled_springs(self):
        """Same through a coupling group (IFT rule carries d/d params)."""
        self._recover(self._coupled, ("sa", "sb"), 15.0, 5.0)

    def test_params_gradient_matches_float64_finite_differences(self):
        """The float32 params gradient against a float64 central
        difference of the same rollout."""
        def loss_of_k(gm, k, dtype):
            # Nodes hard-code float32 state; promote the initial state
            # (not the node) so the whole rollout runs in ``dtype``.
            params = jax.tree.map(lambda x: jnp.asarray(x, dtype), gm.params)
            params["nodes"]["s"]["stiffness"] = jnp.asarray(k, dtype=dtype)
            state0 = jax.tree.map(
                lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
                gm._state,
            )
            traj = self._positions(gm, params, self.N_STEPS, ("s",), state0)
            return jnp.sum(traj ** 2)

        gm32 = self._single(self.K_TRUE, self.C_TRUE)
        g32 = float(jax.grad(lambda k: loss_of_k(gm32, k, jnp.float32))(
            jnp.asarray(self.K_TRUE, jnp.float32)))

        prev = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        try:
            gm64 = self._single(self.K_TRUE, self.C_TRUE)
            h = 1e-4 * self.K_TRUE
            lp = float(loss_of_k(gm64, self.K_TRUE + h, jnp.float64))
            lm = float(loss_of_k(gm64, self.K_TRUE - h, jnp.float64))
            g_fd = (lp - lm) / (2 * h)
            g64 = float(jax.grad(lambda k: loss_of_k(gm64, k, jnp.float64))(
                jnp.asarray(self.K_TRUE, jnp.float64)))
        finally:
            jax.config.update("jax_enable_x64", prev)

        rel32 = abs(g32 - g_fd) / abs(g_fd)
        rel64 = abs(g64 - g_fd) / abs(g_fd)
        print(f"\n[params grad] float32 AD={g32:.6g}  float64 AD={g64:.6g}  "
              f"float64 FD={g_fd:.6g}  rel err: f32={rel32:.2e} f64={rel64:.2e}")
        assert rel64 < 1e-6, rel64
        assert rel32 < 1e-2, rel32

    def test_jvp_wrt_params_through_coupled_step(self):
        gm = self._coupled(self.K_TRUE, self.C_TRUE)
        compiled = gm._compiled_step
        ext = gm._default_external_inputs()
        state = gm._state

        def f(k):
            params = jax.tree.map(lambda x: x, gm.params)
            params["nodes"]["sa"]["stiffness"] = k
            out = compiled(state, ext, params)
            return out["sb"]["position"]

        k0 = jnp.asarray(self.K_TRUE, jnp.float32)
        _, t = jax.jvp(f, (k0,), (jnp.ones_like(k0),))
        g = jax.grad(f)(k0)
        assert bool(jnp.isfinite(t)) and bool(jnp.isfinite(g))
        assert abs(float(t) - float(g)) < 1e-5 * max(1.0, abs(float(g)))

    def test_params_change_takes_effect_without_recompile(self):
        gm = self._single(self.K_TRUE, self.C_TRUE)
        gm.step()
        compiled = gm._compiled_step
        s_before = dict(gm._state["s"])
        a = gm.step()["s"]["position"]
        gm._state["s"] = s_before
        gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(2 * self.K_TRUE, jnp.float32)
        b = gm.step()["s"]["position"]
        assert gm._compiled_step is compiled
        assert gm._dirty is False
        assert float(a) != float(b)

    def test_nodes_without_params_keyword_keep_working(self):
        class Legacy(SpringDamperNode):
            def update(self, state, boundary_inputs, dt):  # 3-arg contract
                return super().update(state, boundary_inputs, dt)

        gm = GraphManager()
        gm.add_node(Legacy("l", 0.01, stiffness=10.0, damping=1.0,
                           initial_position=0.5))
        gm.compile()
        assert "l" not in gm.params["nodes"]
        out = gm.step()
        assert bool(jnp.isfinite(out["l"]["position"]))


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
