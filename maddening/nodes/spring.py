"""
SpringDamperNode -- a spring-damper connecting two attachment points.

Models a linear spring with viscous damping.  One end of the spring is
the node's own ``position``; the other end arrives via
``boundary_inputs["anchor_position"]``.

The entire ``update`` uses ``jnp`` operations so it is fully
JAX-traceable and JIT-compilable.
"""

import jax.numpy as jnp

from maddening.core.node import SimulationNode


class SpringDamperNode(SimulationNode):
    """A spring-damper connecting two attachment points.

    Models a linear spring with damping::

        F = -k * (position - anchor_position - rest_length) - c * velocity
        acceleration = F / mass

    The node tracks its own position (one end of the spring).
    The other end comes via ``boundary_inputs["anchor_position"]``.

    Integration uses semi-implicit Euler (velocity updated first, then
    position uses the new velocity) for better energy behaviour.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    stiffness : float
        Spring constant *k* (N/m).  Default 100.0.
    damping : float
        Damping coefficient *c* (N s/m).  Default 1.0.
    mass : float
        Point mass at this end of the spring (kg).  Default 1.0.
    rest_length : float
        Natural (unstretched) length of the spring.  Default 1.0.
    initial_position : float
        Starting position of this end.  Default 0.0.
    initial_velocity : float
        Starting velocity of this end.  Default 0.0.
    """

    def __init__(
        self,
        name: str,
        timestep: float,
        stiffness: float = 100.0,
        damping: float = 1.0,
        mass: float = 1.0,
        rest_length: float = 1.0,
        initial_position: float = 0.0,
        initial_velocity: float = 0.0,
    ):
        super().__init__(
            name,
            timestep,
            stiffness=stiffness,
            damping=damping,
            mass=mass,
            rest_length=rest_length,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
        )

    def initial_state(self) -> dict:
        return {
            "position": jnp.array(self.params["initial_position"], dtype=jnp.float32),
            "velocity": jnp.array(self.params["initial_velocity"], dtype=jnp.float32),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Semi-implicit Euler integration of spring-damper dynamics.

        If ``anchor_position`` is not supplied the anchor defaults to the
        origin (0.0), so the node still produces sensible behaviour when
        tested in isolation.
        """
        k = self.params["stiffness"]
        c = self.params["damping"]
        m = self.params["mass"]
        rest = self.params["rest_length"]

        position = state["position"]
        velocity = state["velocity"]

        anchor = boundary_inputs.get("anchor_position", jnp.array(0.0, dtype=jnp.float32))

        # Spring-damper force: F = -k*(x - anchor - rest) - c*v
        force = -k * (position - anchor - rest) - c * velocity

        # Semi-implicit Euler: update velocity first, then use new velocity
        acceleration = force / m
        velocity = velocity + acceleration * dt
        position = position + velocity * dt

        return {"position": position, "velocity": velocity}
