"""
BallNode -- a ball under gravity with optional collision against a surface.

Collision detection uses ``jnp.where`` so the entire ``update`` is
JAX-traceable and JIT-compilable.
"""

import jax.numpy as jnp

from maddening.core.node import SimulationNode

GRAVITY = -9.81


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
    """

    def __init__(
        self,
        name: str,
        timestep: float,
        initial_position: float = 0.0,
        initial_velocity: float = 0.0,
        elasticity: float = 0.8,
    ):
        super().__init__(
            name,
            timestep,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            elasticity=elasticity,
        )

    def initial_state(self) -> dict:
        return {
            "position": jnp.array(self.params["initial_position"], dtype=jnp.float32),
            "velocity": jnp.array(self.params["initial_velocity"], dtype=jnp.float32),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Integrate gravity, then handle collision if table_position is provided."""
        velocity = state["velocity"] + GRAVITY * dt
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
