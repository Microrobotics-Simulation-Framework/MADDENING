"""Tests for RigidBody2DNode."""

import pytest
import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.rigid_body_2d import RigidBody2DNode
from maddening.nodes.table import TableNode


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _run_n_steps(node, state, boundary_inputs, n, dt):
    """Run n steps of a node's update, returning the final state."""
    for _ in range(n):
        state = node.update(state, boundary_inputs, dt)
    return state


# ------------------------------------------------------------------
# Basic physics
# ------------------------------------------------------------------

class TestFreeFall:
    """Verify trajectory matches analytical solution for free fall."""

    def test_free_fall_position(self):
        """x = x0 + v0*t + 0.5*g*t^2 (semi-implicit Euler approximation)."""
        dt = 0.001
        n_steps = 1000  # 1 second total
        body = RigidBody2DNode(name="body", timestep=dt,
                               initial_x=0.0, initial_y=10.0)
        state = body.initial_state()
        for _ in range(n_steps):
            state = body.update(state, {}, dt)

        t = dt * n_steps
        # Semi-implicit Euler for constant acceleration is exact for position
        # after n steps: y = y0 + v0*t + g*(t + dt)*t/2
        # (because each step: v += g*dt, y += v*dt, giving a slight lead)
        # For n*dt = t with constant g:
        #   y_n = y0 + sum_{i=1}^{n}(v0 + i*g*dt)*dt
        #       = y0 + v0*t + g*dt^2 * n*(n+1)/2
        n = n_steps
        expected_y = 10.0 + 0.0 * t + (-9.81) * dt**2 * n * (n + 1) / 2
        assert float(state["x"][1]) == pytest.approx(expected_y, rel=1e-4)

    def test_free_fall_velocity(self):
        """v = v0 + g*t."""
        dt = 0.001
        n_steps = 500
        body = RigidBody2DNode(name="body", timestep=dt,
                               initial_y=100.0)
        state = body.initial_state()
        for _ in range(n_steps):
            state = body.update(state, {}, dt)
        t = dt * n_steps
        expected_vy = -9.81 * t
        assert float(state["v"][1]) == pytest.approx(expected_vy, rel=1e-5)

    def test_free_fall_x_unchanged(self):
        """Horizontal position should not change under pure gravity."""
        dt = 0.001
        body = RigidBody2DNode(name="body", timestep=dt,
                               initial_x=5.0, initial_y=10.0)
        state = body.initial_state()
        for _ in range(100):
            state = body.update(state, {}, dt)
        assert float(state["x"][0]) == pytest.approx(5.0, abs=1e-6)


class TestZeroGravity:
    """Verify uniform motion with no gravity."""

    def test_uniform_linear_motion(self):
        dt = 0.01
        n_steps = 100
        body = RigidBody2DNode(name="body", timestep=dt,
                               gravity=(0.0, 0.0),
                               initial_x=1.0, initial_y=2.0,
                               initial_vx=3.0, initial_vy=-1.0)
        state = body.initial_state()
        for _ in range(n_steps):
            state = body.update(state, {}, dt)

        t = dt * n_steps
        assert float(state["x"][0]) == pytest.approx(1.0 + 3.0 * t, rel=1e-5)
        assert float(state["x"][1]) == pytest.approx(2.0 + (-1.0) * t, rel=1e-5)
        assert float(state["v"][0]) == pytest.approx(3.0, abs=1e-6)
        assert float(state["v"][1]) == pytest.approx(-1.0, abs=1e-6)

    def test_uniform_rotation(self):
        dt = 0.01
        n_steps = 100
        body = RigidBody2DNode(name="body", timestep=dt,
                               gravity=(0.0, 0.0),
                               initial_omega=2.0)
        state = body.initial_state()
        for _ in range(n_steps):
            state = body.update(state, {}, dt)

        t = dt * n_steps
        assert float(state["angle"]) == pytest.approx(2.0 * t, rel=1e-5)
        assert float(state["omega"]) == pytest.approx(2.0, abs=1e-6)


class TestAppliedForce:
    """Verify acceleration F/m."""

    def test_horizontal_force(self):
        dt = 0.001
        n_steps = 100
        mass = 2.0
        fx = 10.0
        body = RigidBody2DNode(name="body", timestep=dt,
                               mass=mass, gravity=(0.0, 0.0))
        state = body.initial_state()
        force = jnp.array([fx, 0.0])
        for _ in range(n_steps):
            state = body.update(state, {"force": force}, dt)

        t = dt * n_steps
        expected_vx = fx / mass * t
        assert float(state["v"][0]) == pytest.approx(expected_vx, rel=1e-5)

    def test_force_and_gravity_combine(self):
        dt = 0.001
        n_steps = 200
        mass = 3.0
        body = RigidBody2DNode(name="body", timestep=dt, mass=mass)
        state = body.initial_state()
        # Apply upward force to counteract gravity exactly
        force = jnp.array([0.0, 9.81 * mass])
        for _ in range(n_steps):
            state = body.update(state, {"force": force}, dt)

        # Net acceleration should be zero -> velocity should remain zero
        assert float(state["v"][0]) == pytest.approx(0.0, abs=1e-5)
        assert float(state["v"][1]) == pytest.approx(0.0, abs=1e-5)
        # Position should remain at origin
        assert float(state["x"][0]) == pytest.approx(0.0, abs=1e-5)
        assert float(state["x"][1]) == pytest.approx(0.0, abs=1e-5)


class TestAppliedTorque:
    """Verify angular acceleration tau/I."""

    def test_constant_torque(self):
        dt = 0.001
        n_steps = 500
        inertia = 2.0
        tau = 5.0
        body = RigidBody2DNode(name="body", timestep=dt,
                               inertia=inertia, gravity=(0.0, 0.0))
        state = body.initial_state()
        torque = jnp.array(tau)
        for _ in range(n_steps):
            state = body.update(state, {"torque": torque}, dt)

        t = dt * n_steps
        expected_omega = tau / inertia * t
        assert float(state["omega"]) == pytest.approx(expected_omega, rel=1e-5)

    def test_torque_does_not_affect_translation(self):
        dt = 0.01
        n_steps = 50
        body = RigidBody2DNode(name="body", timestep=dt,
                               gravity=(0.0, 0.0))
        state = body.initial_state()
        torque = jnp.array(10.0)
        for _ in range(n_steps):
            state = body.update(state, {"torque": torque}, dt)

        assert float(state["x"][0]) == pytest.approx(0.0, abs=1e-6)
        assert float(state["x"][1]) == pytest.approx(0.0, abs=1e-6)
        assert float(state["v"][0]) == pytest.approx(0.0, abs=1e-6)
        assert float(state["v"][1]) == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------
# Graph integration
# ------------------------------------------------------------------

class TestInGraph:
    """Test RigidBody2DNode wired in a graph with edges."""

    def test_force_from_another_node(self):
        """Wire a constant-force source node to the rigid body via an edge."""
        gm = GraphManager()
        # Use a TableNode as a constant-value source for one component
        # and apply force via an edge with transform.
        source = TableNode(name="force_src", timestep=0.001, position=5.0)
        body = RigidBody2DNode(name="body", timestep=0.001,
                               mass=1.0, gravity=(0.0, 0.0))
        gm.add_node(source)
        gm.add_node(body)

        # Transform the scalar position into a 2D force vector
        def scalar_to_force(scalar):
            return jnp.array([scalar, 0.0])

        gm.add_edge("force_src", "body", "position", "force",
                     transform=scalar_to_force)
        gm.compile()

        final, history = gm.run_scan_with_history(100)

        # Body should have accelerated in x from the constant force
        assert float(final["body"]["v"][0]) > 0.0
        assert float(final["body"]["x"][0]) > 0.0
        # y should be zero (no gravity, no y-force)
        assert float(final["body"]["x"][1]) == pytest.approx(0.0, abs=1e-5)

    def test_external_input_force(self):
        """Wire force via external inputs."""
        gm = GraphManager()
        body = RigidBody2DNode(name="body", timestep=0.01,
                               mass=2.0, gravity=(0.0, 0.0))
        gm.add_node(body)
        gm.add_external_input("body", "force", shape=(2,))
        gm.compile()

        ext = {"body": {"force": jnp.array([4.0, 0.0])}}
        gm.run(10, external_inputs=ext)
        state = gm.get_node_state("body")
        # a = 4/2 = 2, after 10 steps at dt=0.01: v = 2*0.1 = 0.2
        assert float(state["v"][0]) == pytest.approx(0.2, rel=1e-4)


# ------------------------------------------------------------------
# JIT and differentiation
# ------------------------------------------------------------------

class TestJIT:
    """Verify JIT compilation works."""

    def test_jit_update(self):
        body = RigidBody2DNode(name="body", timestep=0.01)
        state = body.initial_state()
        jitted = jax.jit(body.update, static_argnums=())
        new = jitted(state, {}, 0.01)
        assert jnp.isfinite(new["x"]).all()
        assert jnp.isfinite(new["v"]).all()
        assert jnp.isfinite(new["angle"])
        assert jnp.isfinite(new["omega"])

    def test_jit_in_graph(self):
        """Compiled graph step uses JIT internally."""
        gm = GraphManager()
        body = RigidBody2DNode(name="body", timestep=0.01,
                               gravity=(0.0, 0.0))
        gm.add_node(body)
        gm.compile()
        state = gm.step()
        assert jnp.isfinite(state["body"]["x"]).all()


class TestGrad:
    """Verify jax.grad through update works."""

    def test_grad_wrt_initial_velocity(self):
        body = RigidBody2DNode(name="body", timestep=0.01,
                               gravity=(0.0, 0.0))

        def loss_fn(init_vx):
            state = {
                "x": jnp.array([0.0, 0.0]),
                "angle": jnp.array(0.0),
                "v": jnp.array([init_vx, 0.0]),
                "omega": jnp.array(0.0),
            }
            for _ in range(10):
                state = body.update(state, {}, 0.01)
            return state["x"][0]

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(1.0))
        assert jnp.isfinite(g)
        assert float(g) != 0.0

    def test_grad_wrt_force(self):
        body = RigidBody2DNode(name="body", timestep=0.01,
                               mass=2.0, gravity=(0.0, 0.0))

        def loss_fn(fx):
            state = body.initial_state()
            force = jnp.array([fx, 0.0])
            for _ in range(5):
                state = body.update(state, {"force": force}, 0.01)
            return state["x"][0]

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(1.0))
        assert jnp.isfinite(g)
        assert float(g) > 0.0

    def test_grad_wrt_torque(self):
        body = RigidBody2DNode(name="body", timestep=0.01,
                               inertia=3.0, gravity=(0.0, 0.0))

        def loss_fn(tau):
            state = body.initial_state()
            for _ in range(5):
                state = body.update(state, {"torque": tau}, 0.01)
            return state["angle"]

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(1.0))
        assert jnp.isfinite(g)
        assert float(g) > 0.0


# ------------------------------------------------------------------
# run_scan_with_history
# ------------------------------------------------------------------

class TestScanWithHistory:
    """Verify run_scan_with_history works."""

    def test_history_shapes(self):
        gm = GraphManager()
        body = RigidBody2DNode(name="body", timestep=0.01)
        gm.add_node(body)
        gm.compile()
        n_steps = 200
        final, history = gm.run_scan_with_history(n_steps)

        assert history["body"]["x"].shape == (n_steps, 2)
        assert history["body"]["v"].shape == (n_steps, 2)
        assert history["body"]["angle"].shape == (n_steps,)
        assert history["body"]["omega"].shape == (n_steps,)

    def test_history_final_matches(self):
        gm = GraphManager()
        body = RigidBody2DNode(name="body", timestep=0.01,
                               initial_y=5.0)
        gm.add_node(body)
        gm.compile()
        final, history = gm.run_scan_with_history(100)

        # Final state should match last entry in history
        assert float(final["body"]["x"][0]) == pytest.approx(
            float(history["body"]["x"][-1, 0]), abs=1e-6)
        assert float(final["body"]["x"][1]) == pytest.approx(
            float(history["body"]["x"][-1, 1]), abs=1e-6)
        assert float(final["body"]["angle"]) == pytest.approx(
            float(history["body"]["angle"][-1]), abs=1e-6)


# ------------------------------------------------------------------
# Initial state and introspection
# ------------------------------------------------------------------

class TestInitialState:
    """Verify initial state from parameters."""

    def test_default_initial_state(self):
        body = RigidBody2DNode(name="body", timestep=0.01)
        state = body.initial_state()
        assert jnp.allclose(state["x"], jnp.array([0.0, 0.0]))
        assert float(state["angle"]) == pytest.approx(0.0)
        assert jnp.allclose(state["v"], jnp.array([0.0, 0.0]))
        assert float(state["omega"]) == pytest.approx(0.0)

    def test_custom_initial_state(self):
        body = RigidBody2DNode(name="body", timestep=0.01,
                               initial_x=1.0, initial_y=2.0,
                               initial_vx=3.0, initial_vy=4.0,
                               initial_angle=0.5, initial_omega=1.5)
        state = body.initial_state()
        assert float(state["x"][0]) == pytest.approx(1.0)
        assert float(state["x"][1]) == pytest.approx(2.0)
        assert float(state["v"][0]) == pytest.approx(3.0)
        assert float(state["v"][1]) == pytest.approx(4.0)
        assert float(state["angle"]) == pytest.approx(0.5)
        assert float(state["omega"]) == pytest.approx(1.5)


class TestStateFields:
    """Verify state_fields() returns correct fields."""

    def test_state_fields(self):
        body = RigidBody2DNode(name="body", timestep=0.01)
        fields = body.state_fields()
        assert set(fields) == {"x", "angle", "v", "omega"}


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------

class TestSerialization:
    """Verify to_dict() round-trip."""

    def test_to_dict_type(self):
        body = RigidBody2DNode(name="body", timestep=0.01,
                               mass=2.0, inertia=0.5)
        d = body.to_dict()
        assert d["type"] == "RigidBody2DNode"
        assert d["name"] == "body"
        assert d["timestep"] == 0.01
        assert d["params"]["mass"] == 2.0
        assert d["params"]["inertia"] == 0.5

    def test_to_dict_round_trip(self):
        body = RigidBody2DNode(name="rb", timestep=0.005,
                               mass=3.0, inertia=2.0,
                               gravity=[0.0, -5.0],
                               initial_x=1.0, initial_y=2.0,
                               initial_vx=0.5, initial_vy=-0.5,
                               initial_angle=0.1, initial_omega=0.2)
        d = body.to_dict()

        # Reconstruct from dict
        body2 = RigidBody2DNode(name=d["name"], timestep=d["timestep"],
                                **d["params"])
        d2 = body2.to_dict()

        assert d == d2

        # States should match
        s1 = body.initial_state()
        s2 = body2.initial_state()
        for key in s1:
            assert jnp.allclose(s1[key], s2[key])

    def test_graph_manager_round_trip(self):
        """Round-trip through GraphManager serialization."""
        from maddening.nodes import RigidBody2DNode as RB2D

        gm = GraphManager()
        body = RigidBody2DNode(name="body", timestep=0.01,
                               mass=2.0, initial_x=1.0)
        gm.add_node(body)
        gm.add_external_input("body", "force", shape=(2,))
        gm.compile()

        config = gm.to_dict()
        registry = {"RigidBody2DNode": RB2D}
        gm2 = GraphManager.from_dict(config, registry)
        gm2.compile()

        state = gm2.get_node_state("body")
        assert float(state["x"][0]) == pytest.approx(1.0)
