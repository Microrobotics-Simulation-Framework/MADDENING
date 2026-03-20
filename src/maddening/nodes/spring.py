"""
SpringDamperNode -- a spring-damper connecting two attachment points.

Models a linear spring with viscous damping.  One end of the spring is
the node's own ``position``; the other end arrives via
``boundary_inputs["anchor_position"]``.

The entire ``update`` uses ``jnp`` operations so it is fully
JAX-traceable and JIT-compilable.
"""

import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.compliance.metadata import NodeMeta, StabilityLevel, ValidatedRegime
from maddening.core.compliance.stability import stability


@stability(StabilityLevel.STABLE)
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

    meta = NodeMeta(
        algorithm_id="MADD-NODE-003",
        algorithm_version="1.0.0",
        stability=StabilityLevel.STABLE,
        description="Linear spring-damper connecting two attachment points",
        governing_equations="F = -k*(x - anchor - rest) - c*v; a = F/m",
        discretization="Semi-implicit Euler (1st-order, better energy conservation than forward Euler)",
        assumptions=(
            "Linear spring (Hooke's law)",
            "Viscous damping (linear in velocity)",
            "Point mass (no rotational dynamics)",
        ),
        limitations=(
            "1st-order integration — energy drift over long simulations",
            "No nonlinear spring behaviour (hardening, softening)",
            "No collision detection with other objects",
        ),
        validated_regimes=(
            ValidatedRegime("stiffness", 0.01, 1e6, "N/m", "Tested range; very stiff springs need small dt"),
            ValidatedRegime("damping", 0.0, 1e4, "N·s/m"),
        ),
        hazard_hints=(
            "Very stiff springs (k > 1e4) with large dt can cause numerical instability",
            "Zero mass causes division by zero",
        ),
    )

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

    @property
    def requires_halo(self) -> bool:
        """Pointwise (no spatial neighbor access)."""
        return False

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

    def derivatives(self, state, boundary_inputs):
        """dx/dt = v, dv/dt = F/m."""
        k = self.params["stiffness"]
        c = self.params["damping"]
        m = self.params["mass"]
        rest = self.params["rest_length"]
        anchor = boundary_inputs.get(
            "anchor_position", jnp.array(0.0, dtype=jnp.float32)
        )
        force = -k * (state["position"] - anchor - rest) - c * state["velocity"]
        return {
            "position": state["velocity"],
            "velocity": force / m,
        }

    def implicit_residual(self, state_new, state_old, boundary_inputs, dt):
        """Backward Euler residual: x_new - x_old - dt * f(x_new)."""
        derivs = self.derivatives(state_new, boundary_inputs)
        return {
            k: state_new[k] - state_old[k] - dt * derivs[k]
            for k in derivs
        }

    def boundary_input_spec(self):
        return {
            "anchor_position": BoundaryInputSpec(
                shape=(), description="Position of the other end of the spring",
            ),
        }

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        anchor = boundary_inputs.get(
            "anchor_position", jnp.array(0.0, dtype=jnp.float32)
        )
        k = self.params["stiffness"]
        c = self.params["damping"]
        rest = self.params["rest_length"]
        force = -k * (state["position"] - anchor - rest) - c * state["velocity"]
        return {"spring_force": force}
