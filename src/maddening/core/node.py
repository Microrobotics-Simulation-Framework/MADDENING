"""
SimulationNode ABC -- the contract every physics node must satisfy.

Nodes are *descriptors*: they carry metadata (name, timestep, parameters)
and expose two pure functions:

    initial_state()  ->  dict of JAX arrays
    update(state, boundary_inputs, dt) -> new state dict

Nodes must NEVER store mutable simulation state.  All state lives in the
GraphManager.
"""

import inspect
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.params import ParamSpec


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
    expected_units: str | None = None


@dataclass(frozen=True)
class BoundaryFluxSpec:
    """Descriptor for a declared boundary flux output.

    Parameters
    ----------
    shape : tuple
        Array shape (empty tuple for scalar).
    dtype : any
        JAX dtype.
    description : str
        Human-readable description.
    output_units : str or None
        Physical units of this flux output (e.g. ``"N"``, ``"W/m^2"``).
        Informational -- used for documentation and unit mismatch warnings.
    """
    shape: tuple = ()
    dtype: Any = None
    description: str = ""
    output_units: str | None = None


def _merge_from_wrapped(node: "SimulationNode", getter, guard: str) -> dict:
    """Merge ``getter(inner)`` over every node held as an attribute.

    The shared machinery behind the forwarding defaults of
    :attr:`SimulationNode.static_data` and
    :meth:`SimulationNode.static_data_deps`: both must walk the same
    attributes, qualify colliding keys the same way and terminate on the
    same cycles, or a wrapper's dependency declaration stops lining up
    with the statics it declares for.

    Parameters
    ----------
    node : SimulationNode
        The wrapper (or leaf) to collect from.
    getter : callable
        ``inner -> dict``, applied to each wrapped node.
    guard : str
        Name of the re-entrancy flag set on ``node`` for the duration of
        the walk, so a reference cycle between two nodes terminates.

    Returns
    -------
    dict
        ``{}`` when ``node`` wraps nothing.  Otherwise the union of the
        wrapped nodes' dicts, in attribute order.  Two wrapped nodes
        contributing the same key keep both entries, the second
        qualified by the attribute holding it, rather than one silently
        displacing the other.
    """
    d = getattr(node, "__dict__", None)
    # Fast path: a leaf node wraps nothing, and this runs per node per
    # ``step()``.  Checked before the guard so a leaf never even grows
    # the guard attribute.
    if not d or not any(isinstance(v, SimulationNode) for v in d.values()):
        return {}
    if getattr(node, guard, False):
        return {}
    # ``object.__setattr__`` so a node built as a frozen dataclass can
    # still carry the guard; a node that refuses it outright is only at
    # risk from a reference cycle, which is not a shape the wrappers
    # make.
    try:
        object.__setattr__(node, guard, True)
    except (AttributeError, TypeError):
        pass
    try:
        out: dict = {}
        for attr, value in list(d.items()):
            if not isinstance(value, SimulationNode):
                continue
            for key, item in getter(value).items():
                while key in out:
                    key = f"{attr}.{key}"
                out[key] = item
        return out
    finally:
        try:
            object.__setattr__(node, guard, False)
        except (AttributeError, TypeError):
            pass


def static_data_dep_violations(
    node, spec_overrides: Optional[dict] = None,
) -> list[tuple[str, str, str]]:
    """Declared static-data dependencies that name a trainable parameter.

    The rule
    :meth:`~maddening.core.graph_manager.GraphManager.compile` refuses a
    graph over, factored out so it can be asked of a node on its own.
    See :meth:`SimulationNode.static_data_deps` for why such a
    dependency is an error rather than a supported feature.

    A declared dependency is a violation when the named parameter is
    **both** a leaf of :meth:`SimulationNode.params_pytree` -- a value
    the graph actually differentiates -- and left trainable by
    :meth:`SimulationNode.param_specs`.  A structural parameter (an
    ``int`` such as ``n_cells``, a ``str``, a ``bool``, a nested dict)
    never reaches the parameter pytree, so nothing differentiates
    through it and baking it into a static is the whole point of the
    static channel.

    ``node`` and the nodes it wraps are all walked, each resolved
    against **its own** specs and pytree.  The forwarding default of
    :meth:`SimulationNode.static_data_deps` means a wrapper republishes
    what it wraps, but a wrapper is not obliged to republish
    ``param_specs`` or ``params_pytree`` (``HybridNode`` does; a
    hand-rolled one need not), so resolving an inner declaration against
    the outer node alone would quietly find nothing.  A violation
    visible at more than one level is reported once, attributed to the
    innermost node that declares it -- the one holding both the static
    and the parameter, which is where the fix goes.  The graph supplies
    the outer name it knows the node by.

    ``spec_overrides`` is how the *graph* takes part.  The node's own
    ``param_specs()`` is not the view the optimiser sees:
    :meth:`~maddening.core.graph_manager.GraphManager.param_specs` merges
    :meth:`~maddening.core.graph_manager.GraphManager.set_param_spec`
    overrides over it, and ``trainable_mask``, ``unconstrain``,
    ``check_params`` and ``maddening.sysid`` all read that merged view.
    Resolving the rule against the node alone made it disagree with the
    optimiser in both directions: unfreezing a parameter at graph level
    let a baked static through (a silently wrong gradient), and the
    remedy the error message names -- freeze it -- did not clear the
    refusal when applied through ``set_param_spec``.

    Parameters
    ----------
    node : SimulationNode
        The node to check.  Duck-typed objects missing any of the three
        methods contribute nothing rather than raising, matching how
        ``compile`` probes for the other node-contract hooks.
    spec_overrides : dict of str to ParamSpec, optional
        The graph's overrides for *this* node, applied over each walked
        node's own specs.  They always apply to ``node`` itself, whose
        ``params_pytree`` is the one they were validated against.  They
        reach a wrapped node only when it is the single wrapped holder of
        the parameter name: two wrapped nodes exposing the same name make
        the override ambiguous (the graph sees one of them under a
        qualified key), and leaving their own specs alone can only
        over-refuse, never let a baked static through.

    Returns
    -------
    list of (str, str, str)
        ``(owner_name, static_data_key, param_key)``, in walk order.
    """
    # ``owner_of`` is overwritten as the walk descends, so the deepest
    # node that declares a given violation is the one named; ``order``
    # keeps the report in first-seen (outermost) order.
    owner_of: dict[tuple[str, str], str] = {}
    order: list[tuple[str, str]] = []
    overrides = dict(spec_overrides or {})
    # How many *wrapped* nodes expose each overridden name, so an
    # ambiguous override can be recognised without a second walk.
    holders: dict[str, int] = {}
    if overrides:
        inner_seen: set[int] = {id(node)}
        inner_queue = [
            v for v in getattr(node, "__dict__", {}).values()
            if isinstance(v, SimulationNode)
        ]
        while inner_queue:
            obj = inner_queue.pop(0)
            if id(obj) in inner_seen:
                continue
            inner_seen.add(id(obj))
            pytree_of = getattr(obj, "params_pytree", None)
            if callable(pytree_of):
                for key in pytree_of() or {}:
                    if key in overrides:
                        holders[key] = holders.get(key, 0) + 1
            for value in list(getattr(obj, "__dict__", {}).values()):
                if isinstance(value, SimulationNode):
                    inner_queue.append(value)
    seen: set[int] = set()
    queue = [node]
    while queue:
        obj = queue.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        deps = getattr(obj, "static_data_deps", None)
        specs_of = getattr(obj, "param_specs", None)
        pytree_of = getattr(obj, "params_pytree", None)
        if callable(deps) and callable(specs_of) and callable(pytree_of):
            declared = deps() or {}
            if declared:
                specs = dict(specs_of() or {})
                if overrides:
                    specs.update({
                        k: v for k, v in overrides.items()
                        if obj is node or holders.get(k) == 1
                    })
                leaves = set(pytree_of() or {})
                owner = getattr(obj, "name", repr(obj))
                for static_key in sorted(declared):
                    for param_key in declared[static_key]:
                        if param_key not in leaves:
                            continue
                        if not specs.get(param_key, ParamSpec()).trainable:
                            continue
                        if (static_key, param_key) not in owner_of:
                            order.append((static_key, param_key))
                        owner_of[(static_key, param_key)] = owner
        for value in list(getattr(obj, "__dict__", {}).values()):
            if isinstance(value, SimulationNode):
                queue.append(value)
    return [(owner_of[k], k[0], k[1]) for k in order]


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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject legacy subclasses that override ``requires_halo``.

        Pre-v0.3 ``requires_halo`` was a property derived from
        :meth:`halo_width`.  v0.3.0 removes the property and the
        compat shim; subclasses that override ``requires_halo``
        directly (without overriding ``halo_width``) raise
        :class:`maddening.warnings.MigrationError` at class-definition
        time, naming the migration target.

        See the migration guide at
        ``docs/developer_guide/halo_width_migration.md``.
        """
        super().__init_subclass__(**kwargs)
        overrides_requires_halo = "requires_halo" in cls.__dict__
        overrides_halo_width = "halo_width" in cls.__dict__
        if overrides_requires_halo and not overrides_halo_width:
            from maddening.warnings import MigrationError  # noqa: PLC0415
            raise MigrationError(
                api_name="SimulationNode.requires_halo",
                affected_class=cls,
                replacement="halo_width() -> dict[int, int]",
                migration_guide=(
                    "https://microrobotica.org/maddening/developer_guide/"
                    "halo_width_migration.html"
                ),
            )

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

        A node may declare an optional keyword-only ``params`` argument
        (``update(self, state, boundary_inputs, dt, *, params=None)``).
        When it does, the graph passes the node's entry of the graph
        parameter pytree (see :meth:`params_pytree`) on every call — as
        traced arrays, so ``jax.grad`` reaches them and they can change
        between steps without recompiling.  Read constants from
        ``params`` (falling back to ``self.params`` for structural
        entries) rather than from ``self.params`` directly.
        """
        ...

    def accepts_params(self) -> bool:
        """True when :meth:`update` declares a ``params`` keyword.

        Such nodes receive their entry of the graph parameter pytree on
        every call; the others keep the 3-argument contract and read
        constants from ``self.params`` (baked into the trace, so not
        differentiable through the graph).
        """
        try:
            sig = inspect.signature(self.update)
        except (TypeError, ValueError):
            return False
        return "params" in sig.parameters

    def param_specs(self) -> dict[str, ParamSpec]:
        """Per-parameter :class:`ParamSpec` for the leaves of
        :meth:`params_pytree`.

        Default: every ``initial_*`` entry is ``trainable=False`` (an
        initial condition, not a dynamics constant — ``update`` never
        reads it); everything else is the default spec (trainable,
        unbounded, identity).  Subclasses extend the returned dict with
        bounds and transforms for their constants, e.g.
        ``{"stiffness": ParamSpec(bounds=(0, None), transform="log")}``.
        A graph can override any entry with
        :meth:`GraphManager.set_param_spec`.
        """
        return {
            key: ParamSpec(trainable=False, description="initial condition")
            for key in self.params
            if key.startswith("initial_")
        }

    def params_pytree(self) -> dict:
        """The node's differentiable parameters as a pytree of arrays.

        Default: every float-valued entry of ``self.params`` — Python
        floats, floating-point arrays, and lists/tuples of numbers —
        promoted to float32 arrays.  Ints, bools, strings and nested
        dicts are structural (they change shapes or the trace) and are
        excluded; they stay on the recompile path.

        ``GraphManager.compile`` snapshots this into
        ``GraphManager.params["nodes"][name]`` for nodes whose
        :meth:`update` accepts ``params``.
        """
        out: dict = {}
        for key, value in self.params.items():
            if isinstance(value, (bool, int, str, dict)) or value is None:
                continue
            if isinstance(value, float):
                out[key] = jnp.asarray(value, dtype=jnp.float32)
                continue
            # Arrays, tracers (a node built inside a traced function),
            # and lists/tuples of numbers.  Anything jnp can't turn into
            # a floating array is structural and skipped.
            try:
                arr = jnp.asarray(value)
            except (TypeError, ValueError):
                continue
            if arr.size == 0 or not jnp.issubdtype(arr.dtype, jnp.floating):
                continue
            if isinstance(value, (list, tuple)):
                arr = arr.astype(jnp.float32)
            out[key] = arr
        return out

    # ------------------------------------------------------------------
    # Introspection helpers used by GraphManager
    # ------------------------------------------------------------------

    def state_fields(self) -> list[str]:
        """Return the list of field names produced by ``initial_state``."""
        return list(self.initial_state().keys())

    # ------------------------------------------------------------------
    # Static-data channel (v0.2 #3)
    # ------------------------------------------------------------------

    @property
    def static_data(self) -> dict:
        """Non-state arrays/values closed over by :meth:`update`.

        Default: ``{}``.  Override in subclasses that need to carry
        mesh structures, lookup tables, wall masks, basis functions,
        or any other tensor that participates in ``update()`` but
        does not evolve in time.

        Contract
        --------
        * **Keys** must be strings.
        * **Values** are either Python scalars / strings / tuples (which
          are hashed by value) or JAX/NumPy arrays (which are hashed by
          ``(shape, dtype)`` — the contents themselves are *not* hashed
          because static_data is expected to be much larger than the
          state).
        * Returned dict should be **stable across calls** for a given
          node instance.  Build it once in ``__init__`` and stash on
          ``self``; do not reconstruct per-call.
        * Static data is **not checkpointed**.  See the
          ``static_data_provider`` pattern in ``DESIGN.md`` — store the
          provider config in ``self.params`` so it survives a
          checkpoint/restore round-trip and the static data can be
          reconstructed on ``load_state``.

        Why a separate channel
        ----------------------
        Putting large arrays in :meth:`initial_state` makes them part of
        the JAX state pytree — they get carried through every
        ``fori_loop``, multi-rate step, and gradient pass even though
        they never change.  ``static_data`` lets the node hold them
        outside the state, closing over them in ``update()``.  JAX bakes
        them into the JIT-compiled HLO as constants, which is exactly
        what we want for a 1 GB FVM mesh.

        Derived from a parameter
        ------------------------
        A static built in ``__init__`` from ``self.params`` goes stale
        when that parameter is written: a live write does not mark the
        graph dirty (interactive edits stay cheap that way) and
        :meth:`static_data_hash` covers shape and dtype, never contents,
        so the drift check cannot see it either.  Declare the link in
        :meth:`static_data_deps`; ``compile()`` then refuses the one
        case that is unfixable, a static derived from a *trainable*
        parameter, which no amount of rebuilding could make
        differentiable.

        Wrappers
        --------
        The default is **not** an empty dict for a node that wraps
        others: it forwards, returning the merged ``static_data`` of
        every :class:`SimulationNode` held as an instance attribute.
        A wrapper that declares no statics of its own therefore reports
        the statics of the node it wraps, and
        :meth:`static_data_hash` -- the signal
        :meth:`~maddening.core.graph_manager.GraphManager._check_static_data_dirty`
        watches -- moves when those change.  Without the forwarding a
        wrapped node's statics were invisible to the graph, which sees
        only the outermost object, and the drift check could never fire
        for exactly the nodes (the sharded wrappers) that cache a
        materialisation of them.  Forwarding composes, so
        ``HybridNode(ShardedStencilNode(inner))`` reports ``inner``'s
        statics through both levels; re-entrancy is guarded, so a cycle
        between two nodes terminates (and contributes nothing).

        What is reported is the wrapped node's **declaration** -- the
        :class:`~maddening.core.static_data.StaticArray` as the node
        built it, full shape, ``replication`` and ``shard_axis`` intact
        -- never a wrapper's materialised per-device view.  Three
        reasons, and getting this backwards makes every compile look
        like drift:

        * the hash is over ``(key, shape, dtype, replication,
          shard_axis)``, and a materialised shard is a bare
          ``jax.Array`` that has lost ``replication`` and
          ``shard_axis``; ``coerce_static_data_value`` raises
          ``MigrationError`` on a bare array, so hashing one would not
          merely be wrong, it would fail;
        * materialising costs a ``device_put`` per array, and this
          property is read on every ``step()`` via the drift check --
          precisely the per-frame host overhead the cached
          materialisation exists to avoid;
        * a per-device view is a function of the mesh, not of the
          node's statics.  "The statics changed" is a property of the
          declaration; an unchanged node must hash the same however
          many devices it is spread over.

        A wrapper that does declare statics of its own overrides this
        and should merge ``super().static_data`` in, or the nodes it
        wraps drop out of the hash again.
        """
        return self._wrapped_static_data()

    def _wrapped_static_data(self) -> dict:
        """Merged ``static_data`` of every node held as an attribute.

        The default :attr:`static_data` of a node that declares none of
        its own; see that property for the contract and the reasoning.

        Returns
        -------
        dict
            ``{}`` when this node wraps nothing.  Otherwise the union of
            the wrapped nodes' ``static_data``, in attribute order.  Two
            wrapped nodes declaring the same key keep both entries, the
            second qualified by the attribute holding it, rather than
            one silently displacing the other and going unhashed.
        """
        return _merge_from_wrapped(
            self, lambda inner: inner.static_data, "_collecting_static_data",
        )

    @stability(StabilityLevel.STABLE)
    def static_data_deps(self) -> dict[str, tuple[str, ...]]:
        """Which parameters each entry of :attr:`static_data` derives from.

        Default: ``{}`` for a leaf node, and -- like :attr:`static_data`
        and :meth:`invalidate_static_cache` -- the merged declaration of
        every node this one wraps otherwise, keyed identically to the
        forwarded ``static_data`` so the two line up.

        Returns
        -------
        dict[str, tuple[str, ...]]
            ``{static_data_key: (param_key, ...)}``.  A static with no
            entry declares nothing: one built from literals, read from a
            file, or passed in by the caller has no parameter to name.

        What to declare
        ---------------
        The statics whose **contents the compiled step reads**, and for
        each, the ``self.params`` keys those contents were computed
        from.  Both halves are load-bearing:

        * *contents*, not shape.  :meth:`static_data_hash` covers
          ``(shape, dtype, replication, shard_axis)`` and deliberately
          never the values, so a static rebuilt at the same shape from a
          different parameter value is invisible to it.  A live write to
          a parameter does not mark the graph dirty either -- that is
          what makes interactive slider edits cheap.  This declaration
          is the only place the link is written down.
        * *read by the step*.  A static the node publishes but whose
          values ``update()`` never looks at bakes nothing into the HLO,
          so there is no derivative for the graph to lose and nothing to
          declare.  ``HeatNode`` is exactly that case on its uniform
          grid: ``grid_x`` is built from ``length``, but the uniform
          Laplacian recomputes ``dx = length / n_cells`` from the
          *traced* parameter and never reads ``grid_x``.

        Why a trainable dependency is refused
        -------------------------------------
        :meth:`~maddening.core.graph_manager.GraphManager.compile`
        raises when a declared dependency names a parameter that is both
        a leaf of :meth:`params_pytree` and left trainable by
        :meth:`param_specs`.  A static is baked into the compiled HLO as
        a constant while a trainable parameter is traced; deriving the
        first from the second means the gradient the graph reports is
        missing the term through the static -- silently, and in exactly
        the direction an optimiser is pushing.  You cannot have both, so
        it is an error rather than a feature.  The two ways out are to
        mark the parameter ``ParamSpec(trainable=False)``, or to stop
        deriving the static from it and compute the quantity inside
        ``update()`` from the traced parameter instead.

        A structural parameter is never a violation: an ``int``, ``str``
        or ``bool`` never reaches :meth:`params_pytree`, nothing
        differentiates through it, and baking it is what the static
        channel is for.

        Notes
        -----
        Declaring a dependency does not yet *rebuild* anything.  A write
        to a declared non-trainable dependency still leaves the static
        as ``__init__`` built it; recovering it means reconstructing the
        node.  The rebuild hook and the ``set_node_params`` integration
        are D10 steps 4 and 5, deferred to 0.5.0.
        """
        return _merge_from_wrapped(
            self,
            lambda inner: inner.static_data_deps(),
            "_collecting_static_data_deps",
        )

    def static_data_hash(self) -> int:
        """Stable hash over :attr:`static_data` for JIT cache invalidation.

        For each array value (whether wrapped in
        :class:`~maddening.core.static_data.StaticArray` or a bare
        array), the hash includes ``(key, shape, dtype, replication,
        shard_axis)`` — sharding policy is part of the cache key, so
        a node that switches from replicated to sharded is
        recognised as a recompile-worthy change.

        Non-array scalars hash by ``repr(value)``.

        Returns
        -------
        int
            ``0`` if ``static_data`` is empty.
        """
        # Local import to avoid a circular at module load.
        from maddening.core.static_data import (
            StaticArray, coerce_static_data_value,
        )
        sd = self.static_data
        if not sd:
            return 0
        items = []
        for k in sorted(sd):
            v = coerce_static_data_value(sd[k], node_name=self.name, key=k)
            if isinstance(v, StaticArray):
                items.append((
                    str(k),
                    v.shape, str(v.dtype),
                    v.replication, v.shard_axis,
                ))
            else:
                items.append((str(k), repr(v)))
        return hash(tuple(items))

    @stability(StabilityLevel.STABLE)
    def invalidate_static_cache(self) -> None:
        """Drop any cached materialisation of this node's static data.

        Part of the node contract rather than a duck-typed hook: a node
        that keeps a derived copy of :attr:`static_data` (the sharded
        wrappers cache the per-device placement) overrides this to drop
        it, and :meth:`~maddening.core.graph_manager.GraphManager.compile`
        calls it on every node so that a rebuild is a clean slate.

        The default is a no-op for the node itself and **forwards to every
        node this one wraps**, found by scanning the instance attributes
        for :class:`SimulationNode` values.  Without that, a cache one
        level down -- a ``ShardedStencilNode`` inside a
        :class:`~maddening.core.simulation.hybrid_node.HybridNode`, say --
        is invisible to the graph, which sees only the outermost object
        and would trace a rebuilt step against the previous buffer.
        Re-entrancy is guarded, so a cycle between two nodes terminates.

        A wrapper that overrides this must call ``super()`` so the chain
        keeps going.  A node that holds its inner nodes in a list or dict
        rather than in a plain attribute must forward to them itself.

        Notes
        -----
        Cheap and idempotent by contract: it runs once per node per
        ``compile()``, and the cost of a re-materialisation is paid
        lazily on the next trace, not here.
        """
        if getattr(self, "_invalidating_static_cache", False):
            return
        # ``object.__setattr__`` so a node built as a frozen dataclass
        # can still carry the guard.  A node that refuses the attribute
        # outright (``__slots__``) has no ``__dict__`` for the scan
        # below to walk, so it forwards to nothing and the recursion
        # stops there with or without a guard.
        try:
            object.__setattr__(self, "_invalidating_static_cache", True)
        except (AttributeError, TypeError):
            pass
        try:
            for value in list(getattr(self, "__dict__", {}).values()):
                if isinstance(value, SimulationNode):
                    value.invalidate_static_cache()
        finally:
            try:
                object.__setattr__(self, "_invalidating_static_cache", False)
            except (AttributeError, TypeError):
                pass

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

    def halo_width(self) -> dict[int, int]:
        """Per-axis halo width required by this node's ``update()``.

        Returns a dict mapping spatial axis index to the number of ghost
        cells the node needs on each side of that axis.  Empty dict means
        the node is pointwise (no spatial neighbour access).

        Examples
        --------
        - Pointwise nodes (Ball, Spring, Surrogate): ``{}``
        - 1-D Heat with 2nd-order FD: ``{0: 1}``
        - 1-D Heat with 4th-order FD: ``{0: 2}``
        - 3-D D3Q19 LBM: ``{0: 1, 1: 1, 2: 1}``

        The dict drives pencil-decomposition halo exchange: each entry
        ``axis -> width`` means the sharded state needs ``width`` ghost
        cells on each side of ``axis`` before ``update_padded`` runs.

        Default: ``{}`` (pointwise).  Override in stencil nodes.
        """
        return {}

    def update_padded(
        self,
        state_padded: dict,
        boundary_inputs: dict,
        dt: float,
        *,
        static_padded: dict | None = None,
        shard_info: dict[int, tuple[Any, int]] | None = None,
    ) -> dict:
        """Update from halo-padded state.

        ``ShardedStencilNode`` calls this after exchanging halos: every
        state field listed in :meth:`halo_width` is padded by the
        declared width on each side of the relevant spatial axis.  The
        return value is expected to have the same padded shape; the
        sharding wrapper strips halos afterwards.

        Default: pointwise nodes (empty ``halo_width()``) fall back to
        :meth:`update`.  Stencil nodes that have not been ported must
        override this; calling the default raises
        :class:`NotImplementedError`.

        Parameters
        ----------
        state_padded : dict
            Halo-padded state arrays.
        boundary_inputs : dict
            Boundary inputs (replicated across all shards).
        dt : float
            Timestep.
        static_padded : dict, optional
            ``{static_data_key: halo_padded_slab}`` for each
            :class:`~maddening.core.static_data.StaticArray` declared with
            ``replication="shard"`` on this node.  The wrapper has
            materialised the per-device slice and halo-exchanged it
            (``boundary="edge"`` — statics don't evolve, so periodic
            wrap would be wrong even when state uses periodic).  ``None``
            in the unsharded path and when the node carries no sharded
            statics.
        shard_info : dict[int, tuple[Any, int]], optional
            ``{spatial_axis: (global_offset, local_extent)}`` for every
            spatial axis the wrapping :class:`ShardedStencilNode` shards.
            ``global_offset`` is a **traced JAX scalar**
            (``lax.axis_index * local_extent``) — usable in
            ``jax.lax.dynamic_slice`` but **not** in Python integer
            slicing.  ``None`` in the unsharded path.

        Sharded outputs
        ---------------
        Output keys must be either:

        * a member of :meth:`state_fields` — the wrapper strips halos, or
        * a member of :meth:`domain_integral_fields` — the wrapper
          ``lax.psum``\\ s the value across the device mesh.

        Any other key is a contract violation and the wrapper raises.
        """
        if self.halo_width():
            raise NotImplementedError(
                f"{type(self).__name__} declares halo_width="
                f"{self.halo_width()} but does not override "
                "`update_padded`. Required for sharded stencil execution."
            )
        return self.update(state_padded, boundary_inputs, dt)

    def domain_integral_axes(self) -> dict[str, tuple[str, ...]]:
        """Mesh axes to reduce each domain integral over (C4, v0.4.0).

        Default: empty dict = every key in :meth:`domain_integral_fields`
        is ``psum``-med over the *full* mesh.  A key mapped to a tuple of
        mesh-axis names is reduced over those axes only; the result then
        keeps one leading dimension per *unreduced* mesh axis (in mesh
        order), sharded along it — e.g. a body-surface drag that lives
        on the shards of one pencil row, or a per-slab integral.  An
        empty tuple means no reduction: the per-shard partial values are
        stacked.  Values must be floating-point.
        """
        return {}

    def domain_integral_fields(self) -> set[str]:
        """Output keys that are domain integrals (cross-shard reductions).

        A sharded stencil node may emit small non-spatial outputs that
        are ``jnp.sum``-over-lattice integrals — e.g. drag force /
        torque from an immersed-boundary method.  Each device sees only
        its partial sum; the correct result needs an all-reduce across
        every mesh axis.

        Declaring a key here tells :class:`ShardedStencilNode` to apply
        ``jax.lax.psum`` across the full mesh after
        :meth:`update_padded` returns.  Values must be floating-point
        (``psum`` on integer dtypes risks wrap).

        Default: empty set.  Override in stencil nodes that emit such
        outputs.
        """
        return set()

    def boundary_input_spec(self) -> dict[str, "BoundaryInputSpec"]:
        """Declare expected boundary inputs with shapes and semantics.

        Returns a dict mapping input names to BoundaryInputSpec
        descriptors.  Default: empty dict (backward compatible).
        Override to enable validation and documentation.
        """
        return {}

    def boundary_flux_spec(self) -> dict[str, "BoundaryFluxSpec"]:
        """Declare flux outputs from ``compute_boundary_fluxes``.

        Returns a dict mapping flux field names to BoundaryFluxSpec
        descriptors.  Default: empty dict (backward compatible).
        Override to enable validation and documentation of flux outputs.
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
        *,
        params=None,
    ) -> dict[str, list[tuple[int, Any]]]:
        """Compute corrected values at interface DOFs.

        A node that takes ``params`` in :meth:`update` must take it here
        too (same ``{**self.params, **params}`` rule): the coupling
        system passes the node's ``gm.params`` entry, so a calibrated
        diffusivity also corrects the interface cells.

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
        self, state: dict, boundary_inputs: dict, dt: float, *, params=None,
    ) -> dict:
        """Compute flux quantities at coupling interfaces.

        Returns a dict of flux values (forces, heat fluxes, etc.)
        that other nodes can consume via edges.  These are NOT part
        of the node's state -- they are derived quantities.

        Must be JAX-traceable (pure function).
        Default: empty dict (no fluxes).

        A node that takes ``params`` in :meth:`update` must take it here
        too and read its constants from it (``{**self.params, **params}``):
        the graph passes the node's entry of ``GraphManager.params`` on
        every flux evaluation, so a calibrated stiffness changes the
        force a flux edge delivers, not only the node's own integration.
        A node that declares no ``params`` keyword here is called with
        the 3-argument form and its fluxes use the constructor constants.
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
