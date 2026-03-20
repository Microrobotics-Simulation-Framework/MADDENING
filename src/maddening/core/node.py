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

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability


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
        self.geometry_source: Optional[str] = params.pop("geometry_source", None)
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

    @property
    @abstractmethod
    def requires_halo(self) -> bool:
        """Whether this node's ``update()`` accesses spatial neighbors.

        Returns ``True`` for nodes with stencil operations (e.g. finite
        differences, LBM streaming) where sharding the state along a
        spatial axis requires halo exchange between shards.

        Returns ``False`` for pointwise nodes (e.g. ODE-based, rigid
        body, surrogate models) where each element is independent.

        ``ShardedNode`` checks this property and refuses to shard nodes
        that require halos until halo exchange is implemented.

        Every ``SimulationNode`` subclass **must** override this.
        """
        ...

    def boundary_input_spec(self) -> dict[str, "BoundaryInputSpec"]:
        """Declare expected boundary inputs with shapes and semantics.

        Returns a dict mapping input names to BoundaryInputSpec
        descriptors.  Default: empty dict (backward compatible).
        Override to enable validation and documentation.
        """
        return {}

    def interface_dof_indices(self) -> dict[str, tuple[str, int]]:
        """Map boundary input name to (state_field, index).

        Identifies which boundary inputs correspond to interface DOFs
        where the node's internal BC enforcement may conflict with
        coupled data.  The graph manager uses this together with
        :meth:`compute_interface_correction` to undo internal BC
        enforcement on coupled interface cells.

        Default: ``{}`` (no interface DOFs -- backward compatible).

        Example for a heat rod with Dirichlet BCs at both ends::

            return {
                "left_temperature": ("temperature", 0),
                "right_temperature": ("temperature", -1),
            }
        """
        return {}

    def derivatives(
        self, state: dict, boundary_inputs: dict
    ) -> dict[str, Any]:
        """Compute time derivatives of the state fields.

        Returns ``{field: d_field/dt}`` for each field.  If not all
        fields have continuous derivatives (e.g., collision detection),
        return only the fields that do.

        This enables pluggable integration (RK4, etc.) at the graph
        level.  Default raises ``NotImplementedError`` -- override in
        nodes that have a natural ODE form.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement derivatives(). "
            "Override this method to enable higher-order integration."
        )

    def implicit_residual(
        self,
        state_new: dict,
        state_old: dict,
        boundary_inputs: dict,
        dt: float,
    ) -> dict[str, Any]:
        """Compute the residual for implicit (backward Euler) integration.

        Returns ``{field: R(x_new)}`` where the residual is::

            R(x_new) = x_new - x_old - dt * f(x_new, boundary_inputs)

        Zero residual means x_new satisfies the implicit equation.
        The graph manager solves this with a fixed-count Newton
        iteration via ``jax.lax.fori_loop``.

        Default raises ``NotImplementedError`` -- override in nodes
        that need implicit time integration (e.g., stiff systems).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement implicit_residual()."
        )

    def compute_interface_correction(
        self,
        pre_state: dict,
        boundary_inputs: dict,
        dt: float,
    ) -> dict[str, list[tuple[int, Any]]]:
        """Compute corrected values at interface DOFs.

        After ``update()`` is called, the coupling system calls this
        method to obtain what the interface DOF values *should* be
        without internal BC enforcement.  The corrections are applied
        as ``state[field].at[index].set(value)``.

        Parameters
        ----------
        pre_state : dict
            The node's state **before** ``update()`` was called.
        boundary_inputs : dict
            The boundary inputs that were passed to ``update()``.
        dt : float
            The timestep.

        Returns
        -------
        dict[str, list[tuple[int, value]]]
            ``{field_name: [(index, corrected_value), ...]}``.
            Default: ``{}`` (no corrections -- backward compatible).
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
