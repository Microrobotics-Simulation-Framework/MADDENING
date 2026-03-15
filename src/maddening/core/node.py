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
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from maddening.core.metadata import StabilityLevel
from maddening.core.stability import stability


@dataclass(frozen=True)
class BoundaryInputSpec:
    """Descriptor for an expected boundary input.

    Parameters
    ----------
    shape : tuple
        Array shape (empty tuple for scalar).
    dtype : any
        JAX dtype.
    default : any
        Default value if not supplied.
    coupling_type : str
        ``"replacive"`` (last edge wins) or ``"additive"`` (edges sum).
    description : str
        Human-readable description.
    """
    shape: tuple = ()
    dtype: Any = None  # defaults to jnp.float32 at use site
    default: Any = None
    coupling_type: str = "replacive"
    description: str = ""


@stability(StabilityLevel.STABLE)
class SimulationNode(ABC):
    """Abstract base class for all simulation nodes.

    Subclasses must implement ``initial_state`` and ``update``.
    ``update`` must be a **pure function** suitable for JAX tracing
    (use ``jnp.where`` instead of Python ``if`` for value-dependent
    branching).

    Subclasses should attach a ``meta`` ClassVar with a ``NodeMeta`` instance
    providing algorithm identity, stability level, assumptions, limitations,
    hazard hints, and other compliance-relevant metadata.
    """

    meta: ClassVar[Optional["NodeMeta"]] = None  # type: ignore[name-defined]

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
    # UQ interface (Section 9.4)
    # ------------------------------------------------------------------

    def uncertainty_spec(self) -> Optional["UncertaintySpec"]:  # type: ignore[name-defined]
        """Return the UQ specification for this node, or None.

        Override in subclasses that support uncertainty quantification.
        """
        return None

    # ------------------------------------------------------------------
    # Boundary and flux introspection (Phase 6)
    # ------------------------------------------------------------------

    def boundary_input_spec(self) -> dict[str, "BoundaryInputSpec"]:
        """Declare expected boundary inputs with shapes and semantics.

        Returns a dict mapping input names to BoundaryInputSpec
        descriptors.  Default: empty dict (backward compatible).
        Override to enable validation and documentation.
        """
        return {}

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float
    ) -> dict:
        """Compute flux quantities at coupling interfaces.

        Returns a dict of flux values (forces, heat fluxes, etc.)
        that other nodes can consume via edges.  These are NOT part
        of the node's state -- they are derived quantities.

        Must be JAX-traceable (pure function).
        Default: empty dict (no fluxes).
        """
        return {}

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
