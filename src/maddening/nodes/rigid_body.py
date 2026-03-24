"""
RigidBodyNode -- a 6DOF rigid body with translational and rotational dynamics.

Models a rigid body in 3D with position (x, y, z), orientation (quaternion),
linear velocity (vx, vy, vz), and angular velocity (wx, wy, wz).  External
forces and torques are accepted as boundary inputs.

Integration uses semi-implicit Euler (velocity updated first, then
position uses the new velocity) for better energy behaviour.  Orientation
is represented as a unit quaternion (w, x, y, z) and re-normalised each
step to prevent drift.

All operations use ``jnp`` so the entire ``update`` is fully
JAX-traceable and JIT-compilable.
"""

from typing import Any

import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.compliance.metadata import NodeMeta, StabilityLevel, ValidatedRegime
from maddening.core.compliance.stability import stability


# ------------------------------------------------------------------
# Quaternion helpers (pure JAX, no side-effects)
# ------------------------------------------------------------------

def quat_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_normalize(q: jnp.ndarray) -> jnp.ndarray:
    """Normalise a quaternion to unit length."""
    return q / jnp.linalg.norm(q)


def omega_to_quat(omega: jnp.ndarray) -> jnp.ndarray:
    """Convert angular velocity vector (3,) to pure quaternion (0, wx, wy, wz)."""
    return jnp.array([0.0, omega[0], omega[1], omega[2]])


# ------------------------------------------------------------------
# Constraint application helpers
# ------------------------------------------------------------------

# Maps constraint keys to (state_field, index_or_None).
# index_or_None is the component index for vector fields, or None for
# fields where the constraint value replaces the whole entry (not used
# here, but kept for extensibility).
_CONSTRAINT_MAP = {
    "x":  ("position", 0),
    "y":  ("position", 1),
    "z":  ("position", 2),
    "rx": ("angular_velocity", 0),
    "ry": ("angular_velocity", 1),
    "rz": ("angular_velocity", 2),
}


@stability(StabilityLevel.EXPERIMENTAL)
class RigidBodyNode(SimulationNode):
    """A 6-DOF rigid body subject to forces and torques.

    State
    -----
    position : jnp.array, shape (3,)
        Position [x, y, z] in world frame.
    orientation : jnp.array, shape (4,)
        Orientation quaternion (w, x, y, z).  Initialised to (1, 0, 0, 0).
    velocity : jnp.array, shape (3,)
        Linear velocity [vx, vy, vz].
    angular_velocity : jnp.array, shape (3,)
        Angular velocity [wx, wy, wz] in rad/s.

    Boundary inputs
    ---------------
    force : jnp.array, shape (3,)
        External force [Fx, Fy, Fz].  Defaults to [0, 0, 0].
    torque : jnp.array, shape (3,)
        External torque [Tx, Ty, Tz].  Defaults to [0, 0, 0].

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    mass : float
        Mass of the body (default 1.0).
    inertia : tuple or list
        Diagonal inertia tensor (Ix, Iy, Iz).  Default (1, 1, 1).
    gravity : tuple or list
        Gravitational acceleration [gx, gy, gz].  Default (0, 0, -9.81).
    constraints : dict or None
        DOF constraints.  Keys are ``"x"``, ``"y"``, ``"z"``, ``"rx"``,
        ``"ry"``, ``"rz"``.  Values are the locked value.  For example
        ``{"z": 0, "rx": 0, "ry": 0}`` locks z-translation and x/y
        rotation, giving 2D behaviour in the x-y plane.
    initial_position : tuple
        Initial position (default (0, 0, 0)).
    initial_velocity : tuple
        Initial velocity (default (0, 0, 0)).
    initial_orientation : tuple
        Initial quaternion (w, x, y, z).  Default (1, 0, 0, 0).
    initial_angular_velocity : tuple
        Initial angular velocity.  Default (0, 0, 0).
    """

    meta = NodeMeta(
        algorithm_id="MADD-NODE-007",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="6-DOF rigid body with translational and rotational dynamics",
        governing_equations=(
            "F = m*a; τ = I*α (diagonal inertia); "
            "dq/dt = 0.5 * ω_quat ⊗ q; semi-implicit Euler integration"
        ),
        discretization="Semi-implicit Euler (1st-order)",
        assumptions=(
            "Rigid body (no deformation)",
            "Constant mass and diagonal inertia tensor",
            "Forces and torques applied at centre of mass",
            "Quaternion orientation (no gimbal lock)",
        ),
        limitations=(
            "1st-order integration — energy drift over long simulations",
            "Diagonal inertia only — no off-diagonal terms",
            "No contact/collision detection with other bodies",
            "No constraint handling (joints, hinges) beyond DOF locking",
        ),
        validated_regimes=(
            ValidatedRegime("mass", 1e-6, 1e6, "kg"),
            ValidatedRegime("inertia", 1e-6, 1e6, "kg·m²"),
        ),
        hazard_hints=(
            "Zero mass or zero inertia component causes division by zero",
            "No collision detection — bodies can overlap without penalty",
            "Quaternion drift accumulates over very long simulations despite renormalization",
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        mass: float = 1.0,
        inertia: tuple | list = (1.0, 1.0, 1.0),
        gravity: tuple | list = (0.0, 0.0, -9.81),
        constraints: dict | None = None,
        initial_position: tuple | list = (0.0, 0.0, 0.0),
        initial_velocity: tuple | list = (0.0, 0.0, 0.0),
        initial_orientation: tuple | list = (1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity: tuple | list = (0.0, 0.0, 0.0),
    ):
        super().__init__(
            name,
            timestep,
            mass=mass,
            inertia=list(inertia),
            gravity=list(gravity),
            constraints=dict(constraints) if constraints else {},
            initial_position=list(initial_position),
            initial_velocity=list(initial_velocity),
            initial_orientation=list(initial_orientation),
            initial_angular_velocity=list(initial_angular_velocity),
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def requires_halo(self) -> bool:
        """Pointwise (no spatial neighbor access)."""
        return False

    def initial_state(self) -> dict:
        p = self.params
        return {
            "position": jnp.array(p["initial_position"], dtype=jnp.float32),
            "orientation": jnp.array(p["initial_orientation"], dtype=jnp.float32),
            "velocity": jnp.array(p["initial_velocity"], dtype=jnp.float32),
            "angular_velocity": jnp.array(p["initial_angular_velocity"], dtype=jnp.float32),
        }

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Semi-implicit Euler integration of 6-DOF rigid-body dynamics.

        If ``force`` or ``torque`` are not supplied in *boundary_inputs*
        they default to zero, so the node still produces sensible
        behaviour when tested in isolation (free fall under gravity).
        """
        mass = self.params["mass"]
        inertia = jnp.array(self.params["inertia"], dtype=jnp.float32)
        gravity = jnp.array(self.params["gravity"], dtype=jnp.float32)
        constraints = self.params["constraints"]

        pos = state["position"]
        orient = state["orientation"]
        vel = state["velocity"]
        ang_vel = state["angular_velocity"]

        force = boundary_inputs.get(
            "force", jnp.zeros(3, dtype=jnp.float32)
        )
        torque = boundary_inputs.get(
            "torque", jnp.zeros(3, dtype=jnp.float32)
        )

        # --- Translational dynamics ---
        total_force = force + gravity * mass
        acceleration = total_force / mass
        vel_new = vel + acceleration * dt
        pos_new = pos + vel_new * dt

        # --- Rotational dynamics (diagonal inertia) ---
        alpha = torque / inertia
        ang_vel_new = ang_vel + alpha * dt

        # --- Quaternion update ---
        omega_q = omega_to_quat(ang_vel_new)
        orient_new = orient + 0.5 * dt * quat_multiply(omega_q, orient)
        orient_new = quat_normalize(orient_new)

        # --- Apply DOF constraints ---
        for key, locked_value in constraints.items():
            field_name, idx = _CONSTRAINT_MAP[key]
            if field_name == "position":
                pos_new = pos_new.at[idx].set(locked_value)
                vel_new = vel_new.at[idx].set(0.0)
            elif field_name == "angular_velocity":
                ang_vel_new = ang_vel_new.at[idx].set(locked_value)

        return {
            "position": pos_new,
            "orientation": orient_new,
            "velocity": vel_new,
            "angular_velocity": ang_vel_new,
        }

    # ------------------------------------------------------------------
    # Derivatives (for higher-order integrators)
    # ------------------------------------------------------------------

    def derivatives(
        self, state: dict, boundary_inputs: dict
    ) -> dict[str, Any]:
        """Compute time derivatives of all state fields.

        Returns {field: d_field/dt}.
        """
        mass = self.params["mass"]
        inertia = jnp.array(self.params["inertia"], dtype=jnp.float32)
        gravity = jnp.array(self.params["gravity"], dtype=jnp.float32)
        constraints = self.params["constraints"]

        vel = state["velocity"]
        ang_vel = state["angular_velocity"]
        orient = state["orientation"]

        force = boundary_inputs.get(
            "force", jnp.zeros(3, dtype=jnp.float32)
        )
        torque = boundary_inputs.get(
            "torque", jnp.zeros(3, dtype=jnp.float32)
        )

        total_force = force + gravity * mass
        acceleration = total_force / mass
        alpha = torque / inertia

        # dq/dt = 0.5 * omega_quat * q
        omega_q = omega_to_quat(ang_vel)
        dq_dt = 0.5 * quat_multiply(omega_q, orient)

        d_pos = vel
        d_vel = acceleration
        d_orient = dq_dt
        d_ang_vel = alpha

        # Zero out constrained DOFs
        for key in constraints:
            field_name, idx = _CONSTRAINT_MAP[key]
            if field_name == "position":
                d_pos = d_pos.at[idx].set(0.0)
                d_vel = d_vel.at[idx].set(0.0)
            elif field_name == "angular_velocity":
                d_ang_vel = d_ang_vel.at[idx].set(0.0)

        return {
            "position": d_pos,
            "orientation": d_orient,
            "velocity": d_vel,
            "angular_velocity": d_ang_vel,
        }

    # ------------------------------------------------------------------
    # Implicit residual (backward Euler)
    # ------------------------------------------------------------------

    def implicit_residual(
        self,
        state_new: dict,
        state_old: dict,
        boundary_inputs: dict,
        dt: float,
    ) -> dict[str, Any]:
        """Backward Euler residual: x_new - x_old - dt * f(x_new)."""
        derivs = self.derivatives(state_new, boundary_inputs)
        return {
            k: state_new[k] - state_old[k] - dt * derivs[k]
            for k in derivs
        }

    # ------------------------------------------------------------------
    # Boundary input introspection
    # ------------------------------------------------------------------

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "force": BoundaryInputSpec(
                shape=(3,),
                description="External force [Fx, Fy, Fz]",
                coupling_type="additive",
                expected_units="N",
            ),
            "torque": BoundaryInputSpec(
                shape=(3,),
                description="External torque [Tx, Ty, Tz]",
                coupling_type="additive",
                expected_units="N*m",
            ),
        }
