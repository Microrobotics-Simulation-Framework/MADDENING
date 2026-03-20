"""
RigidBody2DNode -- a 2D rigid body with translational and rotational dynamics.

.. deprecated::
    Use :class:`~maddening.nodes.rigid_body.RigidBodyNode` with
    ``constraints={"z": 0, "rx": 0, "ry": 0}`` instead.

Models a rigid body in 2D with position (x, y), orientation (angle),
linear velocity (vx, vy), and angular velocity (omega).  External
forces and torques are accepted as boundary inputs.

Integration uses semi-implicit Euler (velocity updated first, then
position uses the new velocity) for better energy behaviour.

All operations use ``jnp`` so the entire ``update`` is fully
JAX-traceable and JIT-compilable.
"""

import warnings

import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.compliance.metadata import NodeMeta, StabilityLevel, ValidatedRegime
from maddening.core.compliance.stability import stability


@stability(StabilityLevel.EXPERIMENTAL)
class RigidBody2DNode(SimulationNode):
    """A 2D rigid body subject to forces and torques.

    .. deprecated::
        Use :class:`~maddening.nodes.rigid_body.RigidBodyNode` with
        ``constraints={"z": 0, "rx": 0, "ry": 0}`` for full 6-DOF
        support with equivalent 2D behaviour.

    State
    -----
    x : jnp.array, shape (2,)
        Position [x, y] in world frame.
    angle : jnp.array, shape ()
        Orientation angle in radians.
    v : jnp.array, shape (2,)
        Linear velocity [vx, vy].
    omega : jnp.array, shape ()
        Angular velocity in rad/s.

    Boundary inputs
    ---------------
    force : jnp.array, shape (2,)
        External force [Fx, Fy].  Defaults to [0, 0].
    torque : jnp.array, shape ()
        External torque.  Defaults to 0.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    mass : float
        Mass of the body (default 1.0).
    inertia : float
        Moment of inertia about the centre of mass (default 1.0).
    gravity : list or tuple of float
        Gravitational acceleration vector [gx, gy] (default [0.0, -9.81]).
    initial_x : float
        Initial x-position (default 0.0).
    initial_y : float
        Initial y-position (default 0.0).
    initial_vx : float
        Initial x-velocity (default 0.0).
    initial_vy : float
        Initial y-velocity (default 0.0).
    initial_angle : float
        Initial orientation angle in radians (default 0.0).
    initial_omega : float
        Initial angular velocity in rad/s (default 0.0).
    """

    meta = NodeMeta(
        algorithm_id="MADD-NODE-004",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="2D rigid body with translational and rotational dynamics",
        governing_equations="F = m*a; τ = I*α; semi-implicit Euler integration",
        discretization="Semi-implicit Euler (1st-order)",
        assumptions=(
            "Rigid body (no deformation)",
            "Constant mass and inertia",
            "2D only — no out-of-plane motion",
            "Forces and torques applied at centre of mass",
        ),
        limitations=(
            "1st-order integration — energy drift over long simulations",
            "No contact/collision detection with other bodies",
            "No constraint handling (joints, hinges)",
        ),
        validated_regimes=(
            ValidatedRegime("mass", 1e-6, 1e6, "kg"),
            ValidatedRegime("inertia", 1e-6, 1e6, "kg·m²"),
        ),
        hazard_hints=(
            "Zero mass or zero inertia causes division by zero",
            "No collision detection — bodies can overlap without penalty",
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        mass: float = 1.0,
        inertia: float = 1.0,
        gravity: tuple | list = (0.0, -9.81),
        initial_x: float = 0.0,
        initial_y: float = 0.0,
        initial_vx: float = 0.0,
        initial_vy: float = 0.0,
        initial_angle: float = 0.0,
        initial_omega: float = 0.0,
    ):
        warnings.warn(
            "RigidBody2DNode is deprecated. Use RigidBodyNode with "
            "constraints={'z': 0, 'rx': 0, 'ry': 0} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            name,
            timestep,
            mass=mass,
            inertia=inertia,
            gravity=list(gravity),
            initial_x=initial_x,
            initial_y=initial_y,
            initial_vx=initial_vx,
            initial_vy=initial_vy,
            initial_angle=initial_angle,
            initial_omega=initial_omega,
        )

    @property
    def requires_halo(self) -> bool:
        """Pointwise (no spatial neighbor access)."""
        return False

    def initial_state(self) -> dict:
        p = self.params
        return {
            "x": jnp.array([p["initial_x"], p["initial_y"]], dtype=jnp.float32),
            "angle": jnp.array(p["initial_angle"], dtype=jnp.float32),
            "v": jnp.array([p["initial_vx"], p["initial_vy"]], dtype=jnp.float32),
            "omega": jnp.array(p["initial_omega"], dtype=jnp.float32),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Semi-implicit Euler integration of 2D rigid-body dynamics.

        If ``force`` or ``torque`` are not supplied in *boundary_inputs*
        they default to zero, so the node still produces sensible
        behaviour when tested in isolation (free fall under gravity).
        """
        mass = self.params["mass"]
        inertia = self.params["inertia"]
        gravity = jnp.array(self.params["gravity"], dtype=jnp.float32)

        x = state["x"]
        angle = state["angle"]
        v = state["v"]
        omega = state["omega"]

        force = boundary_inputs.get(
            "force", jnp.zeros(2, dtype=jnp.float32)
        )
        torque = boundary_inputs.get(
            "torque", jnp.array(0.0, dtype=jnp.float32)
        )

        # Translational dynamics
        acceleration = (force + gravity * mass) / mass
        v_new = v + acceleration * dt

        # Rotational dynamics
        alpha = torque / inertia
        omega_new = omega + alpha * dt

        # Position/angle update (semi-implicit: uses new velocities)
        x_new = x + v_new * dt
        angle_new = angle + omega_new * dt

        return {
            "x": x_new,
            "angle": angle_new,
            "v": v_new,
            "omega": omega_new,
        }

    def boundary_input_spec(self):
        return {
            "force": BoundaryInputSpec(
                shape=(2,), description="External force [Fx, Fy]",
                coupling_type="additive",
            ),
            "torque": BoundaryInputSpec(
                shape=(), description="External torque",
                coupling_type="additive",
            ),
        }
