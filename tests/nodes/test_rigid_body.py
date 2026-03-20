"""Tests for RigidBodyNode (6DOF)."""

import warnings

import pytest
import jax
import jax.numpy as jnp

from maddening.nodes.rigid_body import RigidBodyNode, quat_multiply, quat_normalize, omega_to_quat


# ------------------------------------------------------------------
# Construction and initial state
# ------------------------------------------------------------------

class TestConstruction:
    """Verify construction and default parameters."""

    def test_construction_defaults(self):
        body = RigidBodyNode(name="body", timestep=0.01)
        assert body.name == "body"
        assert body.delta_t == 0.01
        assert body.params["mass"] == 1.0
        assert body.params["inertia"] == [1.0, 1.0, 1.0]
        assert body.params["gravity"] == [0.0, 0.0, -9.81]
        assert body.params["constraints"] == {}
        assert body.params["initial_position"] == [0.0, 0.0, 0.0]
        assert body.params["initial_velocity"] == [0.0, 0.0, 0.0]
        assert body.params["initial_orientation"] == [1.0, 0.0, 0.0, 0.0]
        assert body.params["initial_angular_velocity"] == [0.0, 0.0, 0.0]

    def test_initial_state_shapes(self):
        body = RigidBodyNode(name="body", timestep=0.01)
        state = body.initial_state()
        assert state["position"].shape == (3,)
        assert state["orientation"].shape == (4,)
        assert state["velocity"].shape == (3,)
        assert state["angular_velocity"].shape == (3,)

    def test_initial_state_values(self):
        body = RigidBodyNode(
            name="body", timestep=0.01,
            initial_position=(1.0, 2.0, 3.0),
            initial_velocity=(0.5, -0.5, 1.0),
            initial_orientation=(0.707, 0.707, 0.0, 0.0),
            initial_angular_velocity=(0.1, 0.2, 0.3),
        )
        state = body.initial_state()
        assert jnp.allclose(state["position"], jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(state["velocity"], jnp.array([0.5, -0.5, 1.0]))
        assert jnp.allclose(
            state["orientation"],
            jnp.array([0.707, 0.707, 0.0, 0.0]),
            atol=1e-5,
        )
        assert jnp.allclose(state["angular_velocity"], jnp.array([0.1, 0.2, 0.3]))

    def test_state_fields(self):
        body = RigidBodyNode(name="body", timestep=0.01)
        fields = body.state_fields()
        assert set(fields) == {"position", "orientation", "velocity", "angular_velocity"}


# ------------------------------------------------------------------
# Free fall
# ------------------------------------------------------------------

class TestFreeFall:
    """Verify trajectory under gravity only."""

    def test_free_fall(self):
        """After N steps under gravity, z should decrease."""
        dt = 0.001
        n_steps = 500
        body = RigidBodyNode(
            name="body", timestep=dt,
            initial_position=(0, 0, 10),
        )
        state = body.initial_state()
        for _ in range(n_steps):
            state = body.update(state, {}, dt)
        assert float(state["position"][2]) < 10.0

    def test_free_fall_matches_analytical(self):
        """z = z0 + v0*t + semi-implicit correction."""
        dt = 0.001
        n_steps = 1000
        z0 = 10.0
        g = -9.81
        body = RigidBodyNode(
            name="body", timestep=dt,
            initial_position=(0, 0, z0),
            gravity=(0, 0, g),
        )
        state = body.initial_state()
        for _ in range(n_steps):
            state = body.update(state, {}, dt)

        # Semi-implicit Euler with constant acceleration:
        #   z_n = z0 + g * dt^2 * n*(n+1)/2
        n = n_steps
        expected_z = z0 + g * dt**2 * n * (n + 1) / 2
        assert float(state["position"][2]) == pytest.approx(expected_z, rel=1e-4)

    def test_free_fall_xy_unchanged(self):
        """x and y should not change under z-gravity only."""
        dt = 0.001
        body = RigidBodyNode(
            name="body", timestep=dt,
            initial_position=(5.0, 3.0, 10.0),
        )
        state = body.initial_state()
        for _ in range(100):
            state = body.update(state, {}, dt)
        assert float(state["position"][0]) == pytest.approx(5.0, abs=1e-6)
        assert float(state["position"][1]) == pytest.approx(3.0, abs=1e-6)


# ------------------------------------------------------------------
# Force input
# ------------------------------------------------------------------

class TestForceInput:
    """Verify acceleration from external force."""

    def test_force_accelerates(self):
        dt = 0.001
        n_steps = 100
        mass = 2.0
        fx = 10.0
        body = RigidBodyNode(
            name="body", timestep=dt,
            mass=mass, gravity=(0, 0, 0),
        )
        state = body.initial_state()
        force = jnp.array([fx, 0.0, 0.0])
        for _ in range(n_steps):
            state = body.update(state, {"force": force}, dt)

        t = dt * n_steps
        expected_vx = fx / mass * t
        assert float(state["velocity"][0]) == pytest.approx(expected_vx, rel=1e-5)

    def test_force_and_gravity_cancel(self):
        dt = 0.001
        n_steps = 200
        mass = 3.0
        body = RigidBodyNode(
            name="body", timestep=dt, mass=mass,
            gravity=(0, 0, -9.81),
        )
        state = body.initial_state()
        # Upward force to cancel gravity
        force = jnp.array([0.0, 0.0, 9.81 * mass])
        for _ in range(n_steps):
            state = body.update(state, {"force": force}, dt)
        assert float(state["velocity"][2]) == pytest.approx(0.0, abs=1e-5)
        assert float(state["position"][2]) == pytest.approx(0.0, abs=1e-5)


# ------------------------------------------------------------------
# Torque input
# ------------------------------------------------------------------

class TestTorqueInput:
    """Verify angular acceleration from external torque."""

    def test_torque_increases_angular_velocity(self):
        dt = 0.001
        n_steps = 500
        inertia_z = 2.0
        tau_z = 5.0
        body = RigidBodyNode(
            name="body", timestep=dt,
            inertia=(1, 1, inertia_z), gravity=(0, 0, 0),
        )
        state = body.initial_state()
        torque = jnp.array([0.0, 0.0, tau_z])
        for _ in range(n_steps):
            state = body.update(state, {"torque": torque}, dt)

        t = dt * n_steps
        expected_omega_z = tau_z / inertia_z * t
        assert float(state["angular_velocity"][2]) == pytest.approx(
            expected_omega_z, rel=1e-5
        )

    def test_torque_does_not_affect_translation(self):
        dt = 0.01
        n_steps = 50
        body = RigidBodyNode(name="body", timestep=dt, gravity=(0, 0, 0))
        state = body.initial_state()
        torque = jnp.array([10.0, 0.0, 0.0])
        for _ in range(n_steps):
            state = body.update(state, {"torque": torque}, dt)
        assert jnp.allclose(state["position"], jnp.zeros(3), atol=1e-6)
        assert jnp.allclose(state["velocity"], jnp.zeros(3), atol=1e-6)


# ------------------------------------------------------------------
# Quaternion normalization
# ------------------------------------------------------------------

class TestQuaternion:
    """Verify quaternion stays normalised."""

    def test_quaternion_stays_normalized(self):
        dt = 0.01
        n_steps = 10000
        body = RigidBodyNode(
            name="body", timestep=dt, gravity=(0, 0, 0),
            initial_angular_velocity=(1.0, 0.5, -0.3),
        )
        state = body.initial_state()
        for _ in range(n_steps):
            state = body.update(state, {}, dt)
        q_norm = float(jnp.linalg.norm(state["orientation"]))
        assert q_norm == pytest.approx(1.0, abs=1e-6)

    def test_quaternion_helpers(self):
        """Verify pure quaternion helpers."""
        q_identity = jnp.array([1.0, 0.0, 0.0, 0.0])
        # Identity * identity = identity
        result = quat_multiply(q_identity, q_identity)
        assert jnp.allclose(result, q_identity, atol=1e-6)

        # Normalize already-unit quat
        q_norm = quat_normalize(q_identity)
        assert jnp.allclose(q_norm, q_identity, atol=1e-6)

        # omega_to_quat
        omega = jnp.array([1.0, 2.0, 3.0])
        q_omega = omega_to_quat(omega)
        assert float(q_omega[0]) == pytest.approx(0.0)
        assert jnp.allclose(q_omega[1:], omega)


# ------------------------------------------------------------------
# Constraints
# ------------------------------------------------------------------

class TestConstraints:
    """Verify DOF locking via constraints."""

    def test_constraints_lock_z(self):
        dt = 0.01
        n_steps = 100
        body = RigidBodyNode(
            name="body", timestep=dt,
            constraints={"z": 0.0},
            initial_position=(0, 0, 0),
        )
        state = body.initial_state()
        for _ in range(n_steps):
            state = body.update(state, {}, dt)
        # z should stay locked at 0 even with gravity
        assert float(state["position"][2]) == pytest.approx(0.0, abs=1e-10)
        assert float(state["velocity"][2]) == pytest.approx(0.0, abs=1e-10)

    def test_constraints_2d_mode(self):
        """constraints={"z":0,"rx":0,"ry":0} gives 2D-like behaviour."""
        dt = 0.001
        n_steps = 500
        body = RigidBodyNode(
            name="body", timestep=dt,
            gravity=(0, -9.81, 0),
            constraints={"z": 0.0, "rx": 0.0, "ry": 0.0},
            initial_position=(1, 10, 0),
            initial_velocity=(2, 0, 0),
            initial_angular_velocity=(0, 0, 1.0),
        )
        state = body.initial_state()
        for _ in range(n_steps):
            state = body.update(state, {}, dt)

        # z stays locked
        assert float(state["position"][2]) == pytest.approx(0.0, abs=1e-10)
        # x should have moved (initial velocity)
        assert float(state["position"][0]) > 1.0
        # y should have decreased (gravity)
        assert float(state["position"][1]) < 10.0
        # rx, ry locked
        assert float(state["angular_velocity"][0]) == pytest.approx(0.0, abs=1e-10)
        assert float(state["angular_velocity"][1]) == pytest.approx(0.0, abs=1e-10)
        # rz should still be active
        assert float(state["angular_velocity"][2]) == pytest.approx(1.0, rel=1e-5)

    def test_constraints_lock_nonzero_value(self):
        """Lock z to a non-zero value."""
        dt = 0.01
        body = RigidBodyNode(
            name="body", timestep=dt,
            constraints={"z": 5.0},
        )
        state = body.initial_state()
        for _ in range(100):
            state = body.update(state, {}, dt)
        assert float(state["position"][2]) == pytest.approx(5.0, abs=1e-10)


# ------------------------------------------------------------------
# Derivatives and implicit residual
# ------------------------------------------------------------------

class TestDerivatives:
    """Verify derivatives() returns correct shapes."""

    def test_derivatives_shapes(self):
        body = RigidBodyNode(name="body", timestep=0.01, gravity=(0, 0, 0))
        state = body.initial_state()
        derivs = body.derivatives(state, {})
        assert derivs["position"].shape == (3,)
        assert derivs["orientation"].shape == (4,)
        assert derivs["velocity"].shape == (3,)
        assert derivs["angular_velocity"].shape == (3,)

    def test_derivatives_zero_gravity_zero_force(self):
        """With no gravity and no force, derivatives of velocity should be zero."""
        body = RigidBodyNode(name="body", timestep=0.01, gravity=(0, 0, 0))
        state = body.initial_state()
        derivs = body.derivatives(state, {})
        assert jnp.allclose(derivs["velocity"], jnp.zeros(3), atol=1e-6)
        assert jnp.allclose(derivs["angular_velocity"], jnp.zeros(3), atol=1e-6)

    def test_derivatives_with_constraints(self):
        """Constrained DOFs should have zero derivatives."""
        body = RigidBodyNode(
            name="body", timestep=0.01,
            constraints={"z": 0.0, "rx": 0.0},
        )
        state = body.initial_state()
        derivs = body.derivatives(state, {})
        # z-component of position and velocity derivatives should be zero
        assert float(derivs["position"][2]) == pytest.approx(0.0, abs=1e-10)
        assert float(derivs["velocity"][2]) == pytest.approx(0.0, abs=1e-10)
        # rx component of angular velocity derivative should be zero
        assert float(derivs["angular_velocity"][0]) == pytest.approx(0.0, abs=1e-10)


class TestImplicitResidual:
    """Verify implicit_residual() returns correct shapes and zeros for converged."""

    def test_implicit_residual_shapes(self):
        body = RigidBodyNode(name="body", timestep=0.01, gravity=(0, 0, 0))
        state = body.initial_state()
        residual = body.implicit_residual(state, state, {}, 0.01)
        assert residual["position"].shape == (3,)
        assert residual["orientation"].shape == (4,)
        assert residual["velocity"].shape == (3,)
        assert residual["angular_velocity"].shape == (3,)

    def test_implicit_residual_zero_for_converged(self):
        """When state_new == state_old and derivatives are zero, residual is zero."""
        body = RigidBodyNode(name="body", timestep=0.01, gravity=(0, 0, 0))
        state = body.initial_state()
        residual = body.implicit_residual(state, state, {}, 0.01)
        for key in residual:
            assert jnp.allclose(residual[key], jnp.zeros_like(residual[key]), atol=1e-6), (
                f"Non-zero residual for {key}: {residual[key]}"
            )


# ------------------------------------------------------------------
# Boundary input spec
# ------------------------------------------------------------------

class TestBoundaryInputSpec:
    """Verify boundary_input_spec declares correct inputs."""

    def test_boundary_input_spec(self):
        body = RigidBodyNode(name="body", timestep=0.01)
        spec = body.boundary_input_spec()
        assert "force" in spec
        assert "torque" in spec
        assert spec["force"].shape == (3,)
        assert spec["torque"].shape == (3,)
        assert spec["force"].coupling_type == "additive"
        assert spec["torque"].coupling_type == "additive"


# ------------------------------------------------------------------
# JIT compatibility
# ------------------------------------------------------------------

class TestJIT:
    """Verify JIT compilation works."""

    def test_jit_compatible(self):
        body = RigidBodyNode(name="body", timestep=0.01, gravity=(0, 0, 0))
        state = body.initial_state()
        jitted = jax.jit(body.update)
        new = jitted(state, {}, 0.01)
        assert jnp.isfinite(new["position"]).all()
        assert jnp.isfinite(new["orientation"]).all()
        assert jnp.isfinite(new["velocity"]).all()
        assert jnp.isfinite(new["angular_velocity"]).all()

    def test_jit_with_constraints(self):
        body = RigidBodyNode(
            name="body", timestep=0.01,
            constraints={"z": 0.0, "rx": 0.0, "ry": 0.0},
        )
        state = body.initial_state()
        jitted = jax.jit(body.update)
        new = jitted(state, {}, 0.01)
        assert float(new["position"][2]) == pytest.approx(0.0, abs=1e-10)


# ------------------------------------------------------------------
# Grad compatibility
# ------------------------------------------------------------------

class TestGrad:
    """Verify jax.grad through update works."""

    def test_grad_compatible(self):
        body = RigidBodyNode(
            name="body", timestep=0.01, gravity=(0, 0, 0),
        )

        def loss_fn(init_vz):
            state = {
                "position": jnp.array([0.0, 0.0, 0.0]),
                "orientation": jnp.array([1.0, 0.0, 0.0, 0.0]),
                "velocity": jnp.array([0.0, 0.0, init_vz]),
                "angular_velocity": jnp.array([0.0, 0.0, 0.0]),
            }
            for _ in range(10):
                state = body.update(state, {}, 0.01)
            return state["position"][2]

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(1.0))
        assert jnp.isfinite(g)
        assert float(g) != 0.0

    def test_grad_wrt_force(self):
        body = RigidBodyNode(
            name="body", timestep=0.01,
            mass=2.0, gravity=(0, 0, 0),
        )

        def loss_fn(fz):
            state = body.initial_state()
            force = jnp.array([0.0, 0.0, fz])
            for _ in range(5):
                state = body.update(state, {"force": force}, 0.01)
            return state["position"][2]

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(1.0))
        assert jnp.isfinite(g)
        assert float(g) > 0.0

    def test_grad_wrt_torque(self):
        body = RigidBodyNode(
            name="body", timestep=0.01,
            inertia=(1, 1, 3), gravity=(0, 0, 0),
        )

        def loss_fn(tau_z):
            state = body.initial_state()
            torque = jnp.array([0.0, 0.0, tau_z])
            for _ in range(5):
                state = body.update(state, {"torque": torque}, 0.01)
            return state["angular_velocity"][2]

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(1.0))
        assert jnp.isfinite(g)
        assert float(g) > 0.0


# ------------------------------------------------------------------
# Deprecation warning on RigidBody2DNode
# ------------------------------------------------------------------

class TestDeprecated2D:
    """Verify RigidBody2DNode emits DeprecationWarning."""

    def test_deprecated_2d_warns(self):
        from maddening.nodes.rigid_body_2d import RigidBody2DNode
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = RigidBody2DNode(name="body2d", timestep=0.01)
            assert len(w) >= 1
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "deprecated" in str(deprecation_warnings[0].message).lower()


# ------------------------------------------------------------------
# NodeMeta
# ------------------------------------------------------------------

class TestNodeMeta:
    """Verify meta is properly configured."""

    def test_meta_exists(self):
        assert RigidBodyNode.meta is not None
        assert RigidBodyNode.meta.algorithm_id == "MADD-NODE-007"
        assert RigidBodyNode.meta.stability == StabilityLevel.EXPERIMENTAL

    def test_meta_fields(self):
        meta = RigidBodyNode.meta
        assert len(meta.assumptions) > 0
        assert len(meta.limitations) > 0
        assert len(meta.hazard_hints) > 0
        assert len(meta.validated_regimes) > 0


from maddening.core.compliance.metadata import StabilityLevel
