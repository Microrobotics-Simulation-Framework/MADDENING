"""
TableNode -- a static surface (e.g. a table).

The table does not move; its ``update`` returns the state unchanged.
"""

import jax.numpy as jnp

from maddening.core.node import SimulationNode
from maddening.core.compliance.metadata import NodeMeta, StabilityLevel
from maddening.core.compliance.stability import stability


@stability(StabilityLevel.STABLE)
class TableNode(SimulationNode):
    """A fixed horizontal surface.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    position : float
        Height of the surface (default 0.0).
    """

    meta = NodeMeta(
        algorithm_id="MADD-NODE-002",
        algorithm_version="1.0.0",
        stability=StabilityLevel.STABLE,
        description="Static horizontal surface (collision boundary)",
        assumptions=(
            "Surface is rigid and immovable",
            "Surface is infinite in extent (1D collision model)",
        ),
        limitations=(
            "No deformation or dynamic response to impacts",
        ),
        hazard_hints=(
            "Does not model surface deformation — unsuitable for soft tissue boundaries",
        ),
    )

    def __init__(self, name: str, timestep: float, position: float = 0.0):
        super().__init__(name, timestep, position=position)

    def halo_width(self) -> dict[int, int]:
        """Pointwise (no spatial neighbour access)."""
        return {}

    def initial_state(self) -> dict:
        return {
            "position": jnp.array(self.params["position"], dtype=jnp.float32),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Static: return state unchanged."""
        return state
