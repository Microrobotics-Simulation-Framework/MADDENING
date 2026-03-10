"""
TableNode -- a static surface (e.g. a table).

The table does not move; its ``update`` returns the state unchanged.
"""

import jax.numpy as jnp

from maddening.core.node import SimulationNode


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

    def __init__(self, name: str, timestep: float, position: float = 0.0):
        super().__init__(name, timestep, position=position)

    def initial_state(self) -> dict:
        return {
            "position": jnp.array(self.params["position"], dtype=jnp.float32),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Static: return state unchanged."""
        return state
