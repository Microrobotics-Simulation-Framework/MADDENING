"""
BallNode -- a ball under gravity with optional collision against a surface.

Collision detection uses ``jnp.where`` so the entire ``update`` is
JAX-traceable and JIT-compilable.
"""

import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.compliance.metadata import NodeMeta, StabilityLevel, ValidatedRegime
from maddening.core.compliance.stability import stability

GRAVITY = -9.81  # default; use gravity param on BallNode for per-instance control


@stability(StabilityLevel.STABLE)
class BallNode(SimulationNode):
    """A point-mass ball subject to gravity.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    initial_position : float
        Starting height (default 0.0).
    initial_velocity : float
        Starting velocity (default 0.0).
    elasticity : float
        Coefficient of restitution for collisions (default 0.8).
    gravity : float
        Gravitational acceleration (default -9.81 m/s^2).
    """

    meta = NodeMeta(
        algorithm_id="MADD-NODE-001",
        algorithm_version="1.0.0",
        stability=StabilityLevel.STABLE,
        description="Point-mass ball under gravity with optional surface collision",
        governing_equations="dv/dt = g; dx/dt = v; collision: v -> -e*v at x = table_pos",
        discretization="Forward Euler (explicit, 1st-order)",
        assumptions=(
            "Point mass (no rotational dynamics)",
            "Perfectly rigid collision surface",
            "Coefficient of restitution is constant (not velocity-dependent)",
        ),
        limitations=(
            "Forward Euler is only 1st-order — large timesteps cause energy drift",
            "Collision detection is per-step: tunneling possible if v*dt > gap",
            "No air resistance or drag",
        ),
        validated_regimes=(
            ValidatedRegime("elasticity", 0.0, 1.0, notes="e=0 is perfectly inelastic, e=1 is perfectly elastic"),
            ValidatedRegime("timestep", 0.0001, 0.1, "s", "Tested range; smaller is more accurate"),
        ),
        hazard_hints=(
            "Tunneling through collision surface at large dt or high velocity",
            "Energy drift accumulates over long simulations due to 1st-order integration",
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        initial_position: float = 0.0,
        initial_velocity: float = 0.0,
        elasticity: float = 0.8,
        gravity: float = -9.81,
    ):
        super().__init__(
            name,
            timestep,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            elasticity=elasticity,
            gravity=gravity,
        )

    def halo_width(self) -> dict[int, int]:
        """Pointwise (no spatial neighbour access)."""
        return {}

    def initial_state(self) -> dict:
        return {
            "position": jnp.array(self.params["initial_position"], dtype=jnp.float32),
            "velocity": jnp.array(self.params["initial_velocity"], dtype=jnp.float32),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Integrate gravity, then handle collision if table_position is provided."""
        gravity = self.params["gravity"]
        velocity = state["velocity"] + gravity * dt
        position = state["position"] + velocity * dt

        table_pos = boundary_inputs.get("table_position", None)
        if table_pos is not None:
            elasticity = self.params["elasticity"]
            hit = position < table_pos
            position = jnp.where(hit, table_pos, position)
            velocity = jnp.where(
                hit & (jnp.abs(velocity) > 1e-4),
                -velocity * elasticity,
                jnp.where(hit, 0.0, velocity),
            )

        return {"position": position, "velocity": velocity}

    def derivatives(self, state, boundary_inputs):
        """dx/dt = v, dv/dt = g (no collision)."""
        gravity = self.params["gravity"]
        return {
            "position": state["velocity"],
            "velocity": jnp.array(gravity, dtype=jnp.float32),
        }

    def boundary_input_spec(self):
        return {
            "table_position": BoundaryInputSpec(
                shape=(), description="Surface position for collision",
                expected_units="m",
            ),
        }
