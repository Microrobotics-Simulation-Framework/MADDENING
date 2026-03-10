"""
SimulationNode ABC -- the contract every physics node must satisfy.

Nodes are *descriptors*: they carry metadata (name, timestep, parameters)
and expose two pure functions:

    initial_state()  ->  dict of JAX arrays
    update(state, boundary_inputs, dt) -> new state dict

Nodes must NEVER store mutable simulation state.  All state lives in the
GraphManager.
"""

from abc import ABC, abstractmethod


class SimulationNode(ABC):
    """Abstract base class for all simulation nodes.

    Subclasses must implement ``initial_state`` and ``update``.
    ``update`` must be a **pure function** suitable for JAX tracing
    (use ``jnp.where`` instead of Python ``if`` for value-dependent
    branching).
    """

    def __init__(self, name: str, timestep: float, **params):
        self.name = name
        self.delta_t = float(timestep)
        self.params = dict(params)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def initial_state(self) -> dict:
        """Return the initial state as a dict of JAX arrays."""
        ...

    @abstractmethod
    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Pure function: (state, boundary_inputs, dt) -> new_state.

        Must be JAX-traceable.  No Python-level side-effects.
        """
        ...

    # ------------------------------------------------------------------
    # Introspection helpers used by GraphManager
    # ------------------------------------------------------------------

    def state_fields(self) -> list[str]:
        """Return the list of field names produced by ``initial_state``."""
        return list(self.initial_state().keys())

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the node descriptor (not runtime state)."""
        return {
            "type": type(self).__name__,
            "name": self.name,
            "timestep": self.delta_t,
            "params": self.params,
        }
