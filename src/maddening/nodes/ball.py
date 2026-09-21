"""
BallNode -- a ball under gravity with optional collision against a surface.

Collision detection uses ``jnp.where`` so the entire ``update`` is
JAX-traceable and JIT-compilable.
"""

import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.compliance.metadata import (
    DiscretizationOrder,
    NodeMeta,
    StabilityLevel,
    ValidatedRegime,
)
from maddening.core.compliance.stability import stability
from maddening.core.params import ParamSpec

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
        discretization_order=DiscretizationOrder(
            spatial=None,
            temporal=1.0,
            notes=(
                "1st order globally in position and velocity, measured by "
                "MADD-VER-010.  No spatial order -- the node integrates an "
                "ODE.  The order holds, but the scheme ``update()`` "
                "implements is semi-implicit (symplectic) Euler, not the "
                "forward Euler ``discretization`` above names: the position "
                "update uses the already-updated velocity, so it disagrees "
                "with a forward-Euler step built from this node's own "
                "``derivatives()`` at O(dt).  See MADD-ANO-011.  Both "
                "schemes are 1st order, which is why the declared order is "
                "unaffected.  The order claim also covers the smooth "
                "(collision-free) regime only: a ``table_position`` contact "
                "is a non-smooth event and no order is claimed across it."
            ),
        ),
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

    def param_specs(self) -> dict[str, ParamSpec]:
        return {
            **super().param_specs(),
            # Inclusive bounds: a perfectly elastic (1.0) or perfectly
            # inelastic (0.0) ball is a valid model, so no logit.
            "elasticity": ParamSpec(bounds=(0.0, 1.0)),
            "gravity": ParamSpec(units="m/s^2"),
        }

    def initial_state(self) -> dict:
        return {
            "position": jnp.array(self.params["initial_position"], dtype=jnp.float32),
            "velocity": jnp.array(self.params["initial_velocity"], dtype=jnp.float32),
        }

    def update(
        self, state: dict, boundary_inputs: dict, dt: float, *, params=None,
    ) -> dict:
        """Integrate gravity, then handle collision if table_position is provided."""
        p = self.params if params is None else {**self.params, **params}
        gravity = p["gravity"]
        velocity = state["velocity"] + gravity * dt
        position = state["position"] + velocity * dt

        table_pos = boundary_inputs.get("table_position", None)
        if table_pos is not None:
            elasticity = p["elasticity"]
            hit = position < table_pos
            position = jnp.where(hit, table_pos, position)
            velocity = jnp.where(
                hit & (jnp.abs(velocity) > 1e-4),
                -velocity * elasticity,
                jnp.where(hit, 0.0, velocity),
            )

        return {"position": position, "velocity": velocity}

    def derivatives(self, state, boundary_inputs, *, params=None):
        """dx/dt = v, dv/dt = g (no collision).

        ``g`` comes from the injected ``params`` when the caller supplies
        them (``integrate_node(..., params=...)``), by the same
        ``{**self.params, **params}`` rule as ``update``.  ``elasticity``
        is not read here at all: this is the collision-free right-hand
        side, so that parameter has no derivative through this path by
        construction, not by omission.

        ``g`` follows the dtype of the velocity it will be added to,
        rather than being pinned to float32.  Pinning it made this node
        disagree with *itself*: ``update`` reads ``p["gravity"]`` raw,
        so under ``jax_enable_x64`` with a float64 state the explicit
        path integrated -9.81 while ``derivatives`` -- and therefore
        ``integrate_node``, ``euler_step`` and every higher-order
        integrator built on it -- integrated -9.810000419616699, an
        absolute difference of 4.196e-07 in the acceleration.  That is
        the MADD-ANO-011/012 pattern (one node, two answers) arriving
        by a different mechanism.

        Following the state cannot promote anything: at the framework
        default the state is float32 and so is ``g``, exactly as
        before.  It only stops the node from *demoting* a float64 carry
        halfway through a step.
        """
        p = self.params if params is None else {**self.params, **params}
        gravity = p["gravity"]
        velocity = jnp.asarray(state["velocity"])
        return {
            "position": velocity,
            "velocity": jnp.asarray(gravity, dtype=velocity.dtype),
        }

    def boundary_input_spec(self):
        return {
            "table_position": BoundaryInputSpec(
                shape=(), description="Surface position for collision",
                expected_units="m",
            ),
        }
