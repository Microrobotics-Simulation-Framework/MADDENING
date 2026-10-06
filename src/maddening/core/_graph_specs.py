"""Bookkeeping structs and small helpers of the compiled graph step.

Moved verbatim out of ``maddening.core.graph_manager``: the per-node
records, the hook dispatchers, the reserved state keys and the name
refusals, the multi-rate timestep arithmetic and the device-mesh helpers.
Private: ``GraphManager`` and the coupling machinery read them from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Sequence

import jax
# `jax.core` is not re-exported by `jax/__init__.py`, so the attribute
# only resolves for a type checker when the submodule is imported by name.
import jax.core
import jax.numpy as jnp
import numpy as np

from maddening.core.edge import EdgeSpec, _delivered
from maddening.core.node import SimulationNode, _method_accepts_params


# ------------------------------------------------------------------
# Internal bookkeeping structs
# ------------------------------------------------------------------

@dataclass
class _NodeSpec:
    """Everything the graph manager needs to know about a node."""
    node: SimulationNode          # the descriptor object
    update_fn: Callable           # node.update  (pure function)
    timestep: float
    # True when ``update`` declares a ``params`` keyword: the graph then
    # passes the node's entry of the graph parameter pytree on every
    # call (traced, differentiable).  Nodes that don't opt in keep the
    # 3-argument contract and read constants from ``self.params``.
    accepts_params: bool = False
    # Same for ``compute_boundary_fluxes``: a flux producer that reads
    # its constants from ``params`` gets the node's pytree entry on
    # every flux evaluation (a calibrated stiffness changes the force a
    # flux edge delivers).
    flux_accepts_params: bool = False


@dataclass(frozen=True)
class _StepPlan:
    """The derived values ``_build_step_fn`` closes over.

    ``compile()`` recomputes all of them and can still raise afterwards
    (the ``accelerated_fields`` validation, the static-data refusal,
    ``_build_step_fn`` itself), so they travel as a plan and are written
    onto the graph only at the commit point at the end of a successful
    compile.  A failed ``compile()`` leaves the graph exactly as it was:
    anything reading ``_schedule``, ``_is_multirate``, ``_rate_dividers``
    or :attr:`~GraphManager.params` as a description of the step that is
    actually running would otherwise read a step that was never built.
    """
    schedule: list[str]
    back_edges: list[EdgeSpec]
    is_multirate: bool
    rate_dividers: dict[str, int]
    params: dict


def _correction_accepts_params(node: SimulationNode) -> bool:
    """Does the graph pass ``params=`` to ``compute_interface_correction``?

    The one params rule, :func:`~maddening.core.node._method_accepts_params`:
    an explicit ``params`` keyword *or* a ``**kwargs`` that would forward
    it, asked through the node's own ``accepts_params`` probe when it has
    one.  Until 0.4.0 this probe accepted only the explicit keyword, so a
    ``def compute_interface_correction(self, *args, **kwargs)`` override
    that forwards to ``super()`` was called without ``params`` and
    corrected the interface cells from the constructor's constants while
    ``update`` used the calibrated ones -- and the verification battery,
    which already read the shared rule, disagreed with the graph.
    """
    return _method_accepts_params(node, "compute_interface_correction")


def _flux_accepts_params(node: SimulationNode) -> bool:
    """Does the graph pass ``params=`` to ``compute_boundary_fluxes``?

    Same rule and same history as :func:`_correction_accepts_params`: a
    ``**kwargs``-forwarding flux producer delivered the constructor's
    flux on every flux edge.
    """
    return _method_accepts_params(node, "compute_boundary_fluxes")


def _node_fluxes(spec: _NodeSpec, state, boundary_inputs, dt, node_params):
    """``spec.node.compute_boundary_fluxes`` with the params contract."""
    if spec.flux_accepts_params and node_params is not None:
        return spec.node.compute_boundary_fluxes(
            state, boundary_inputs, dt, params=node_params,
        )
    return spec.node.compute_boundary_fluxes(state, boundary_inputs, dt)


def _update_accepts_params(node: SimulationNode) -> bool:
    """Does the graph pass ``params=`` to ``update``?

    :func:`~maddening.core.node._method_accepts_params`, like every other
    params probe.  Its duck-typed fallback used to accept only an
    explicit ``params`` keyword, so a node object that does not subclass
    :class:`SimulationNode` and forwards ``**kwargs`` was left out of
    ``gm.params`` while the verification battery and a wrapped copy of
    the same node disagreed about it.
    """
    return _method_accepts_params(node, "update")


def _node_update(spec: _NodeSpec, state, boundary_inputs, dt, node_params):
    # named_scope is trace-time metadata only (no runtime cost); the
    # profiler's trace attribution keys device kernels on it.
    with jax.named_scope(f"node:{spec.node.name}"):
        if spec.accepts_params and node_params is not None:
            return spec.update_fn(state, boundary_inputs, dt, params=node_params)
        return spec.update_fn(state, boundary_inputs, dt)


def _is_stencil_wrapper(node: Any) -> bool:
    """A ``ShardedStencilNode`` (or subclass), asked without importing it."""
    return callable(getattr(node, "_xla_scan_hazards", None)) and callable(
        getattr(node, "_statics_replicated_over_devices", None))


def _outermost_stencil_wrapper(node: Any, _depth: int = 0) -> Any:
    """The first ``ShardedStencilNode`` under ``node``, through the wrappers
    that hold their node as ``physics_node`` (``HybridNode``) or ``_inner``;
    ``None`` when there is none."""
    while node is not None and _depth < 64:
        if _is_stencil_wrapper(node):
            return node
        nxt = getattr(node, "physics_node", None)
        node = getattr(node, "_inner", None) if nxt is None else nxt
        _depth += 1
    return None


def _innermost_wrapped(node: Any) -> Any:
    """The node a chain of ``_inner`` wrappers ends at."""
    for _ in range(64):
        nxt = getattr(node, "_inner", None)
        if nxt is None:
            break
        node = nxt
    return node


class _ResolvedParams(NamedTuple):
    """The graph parameter pytree split for the step builders: per-node
    pytrees (``nodes[name]``) and per-edge mapping weights
    (``mappings[edge.key]``).  Both are traced when an explicit ``params``
    is passed, baked constants otherwise."""
    nodes: dict
    mappings: dict


def _strong_typed(tree):
    """Strip JAX weak typing from every array leaf.

    ``jnp.array(0.0)`` is *weak-typed*; the same leaf after one step is
    strongly typed (it is the result of arithmetic with typed arrays),
    and a jitted step keyed on ``(shape, dtype, weak_type)`` retraces —
    once per leaf whose weak type flips, typically on the second and
    third steps of every run (measured: three compiles of the same step
    on the MIME AR4 graph, ~2.6 s).  Normalising the seed state and the
    values callers hand in keeps the trace signature constant.
    """
    def _fix(x):
        if getattr(x, "weak_type", False):
            return x.astype(x.dtype)
        return x
    return jax.tree.map(_fix, tree)


def _edge_geom(edge: EdgeSpec, src_state, consumer_state):
    """The geometry *edge*'s mapping reads at one call site, or ``None``.

    A source-anchored geometry is read from *src_state*, the same state
    dict the edge's value is looked up in, so it has the value's time
    level.  A target-anchored one is the named field of *consumer_state*:
    the ``state`` argument of the hook (``update`` or
    ``compute_boundary_fluxes``) the boundary inputs are being resolved
    for.  *consumer_state* may be a zero-argument callable, called only
    for a target-anchored geometry: a call site whose consumer state is
    built after its boundary inputs then traces nothing new, and nothing
    in another order, for every edge without one.
    """
    if edge.geometry is None:
        return None
    anchor, field = edge.geometry
    if anchor == "source":
        return src_state[edge.source_node][field]
    held: Any = consumer_state() if callable(consumer_state) else consumer_state
    return held[field]


_GEOMETRY_ANCHORS = ("source", "target")


def _checked_geometry(key: str, geometry, mapping) -> Optional[tuple[str, str]]:
    """``add_edge``'s *geometry* as the tuple an :class:`EdgeSpec` holds.

    Raises ``ValueError`` for a malformed value, for a geometry without a
    geometry-dependent mapping, and for a geometry-dependent mapping
    without a geometry.  ``None`` with any other mapping, or with no
    mapping, is the edge every graph had.
    """
    reads = mapping is not None and bool(getattr(mapping, "needs_geometry", False))
    if geometry is None:
        if reads:
            raise ValueError(
                f"add_edge({key}): mapping {mapping!r} reads a moving geometry and "
                f"none was given. Pass geometry=(\"source\", <state field>) or "
                f"geometry=(\"target\", <state field>)."
            )
        return None
    given = geometry
    pair = None
    if isinstance(geometry, dict):
        if set(geometry) == {"anchor", "field"}:
            pair = (geometry["anchor"], geometry["field"])
    elif isinstance(geometry, (list, tuple)) and len(geometry) == 2:
        pair = tuple(geometry)
    if (pair is None or not all(isinstance(x, str) and x for x in pair)
            or pair[0] not in _GEOMETRY_ANCHORS):
        raise ValueError(
            f"add_edge({key}): geometry must be (\"source\", <field>) or "
            f"(\"target\", <field>), a state field of this edge's own source or "
            f"target node; got {given!r}. A geometry held by any other node is not "
            f"supported in 0.4.0: carry it in the source or target node's state."
        )
    if mapping is None:
        raise ValueError(
            f"add_edge({key}): geometry={pair!r} was given without a mapping. A "
            f"geometry is the moving interface an interface mapping reads; pass "
            f"mapping=<a geometry-dependent mapping> or drop geometry=."
        )
    if not reads:
        raise ValueError(
            f"add_edge({key}): mapping {mapping!r} is static (it reads no geometry), "
            f"but geometry={pair!r} was given and would be ignored. Drop geometry=, or "
            f"use a geometry-dependent mapping kind."
        )
    return (pair[0], pair[1])


def _mapping_field_leads(mapping) -> Optional[tuple[tuple, tuple]]:
    """``(source lead, target lead)`` of a mapping that declares
    ``field_shapes()``: the leading axes of the field it reads and of the
    field it delivers.  ``None`` for a mapping without the method, which
    acts on axis 0 (``n_source`` in, ``n_target`` out) as every mapping
    did."""
    shapes = getattr(mapping, "field_shapes", None)
    if shapes is None:
        return None
    source_lead, target_lead = shapes()
    return (tuple(int(n) for n in source_lead), tuple(int(n) for n in target_lead))


_GEOMETRY_DTYPES = ("float32", "float64")


def _geometry_edge_issues(edges, nodes, state) -> list[str]:
    """``validate()``'s issues about the geometry each edge names.

    ``ERROR:`` for a geometry on an edge with a sharded end, one that is
    not a state field of its anchor node (absent, or one of its boundary
    fluxes), one that is not float32 or float64, and one whose shape the
    mapping does not read; plus whatever the mapping itself says about
    positions held in that dtype (``geometry_dtype_problems``, optional:
    errors and ``WARNING:`` advisories).  Empty for a graph without a
    geometry edge.
    """
    from maddening.core.node import SimulationNode  # noqa: PLC0415

    issues: list[str] = []
    for e in edges:
        if e.geometry is None:
            continue
        anchor, field = e.geometry
        for which, name in (("source", e.source_node), ("target", e.target_node)):
            obj = nodes[name].node if name in nodes else None
            for _ in range(64):
                if obj is None:
                    break
                if getattr(obj, "_mesh", None) is not None:
                    issues.append(
                        f"ERROR: edge {e.key}: its {which} node {name!r} is sharded "
                        f"({type(obj).__name__}). A geometry-dependent mapping is not "
                        f"supported on an edge with a sharded end in 0.4.0. Use an "
                        f"unsharded node on both ends.")
                    break
                nxt = getattr(obj, "physics_node", None)
                obj = getattr(obj, "_inner", None) if nxt is None else nxt
        holder = e.source_node if anchor == "source" else e.target_node
        if holder not in state or holder not in nodes:
            continue        # a missing node is reported by the edge checks
        fields = state[holder]
        if field not in fields:
            node = nodes[holder].node
            is_flux = (
                type(node).compute_boundary_fluxes is not SimulationNode.compute_boundary_fluxes
                and field in node.compute_boundary_fluxes(fields, {}, 0.0))
            if is_flux:
                issues.append(
                    f"ERROR: edge {e.key}: geometry field {field!r} of {holder!r} is a "
                    f"boundary flux (compute_boundary_fluxes), not a state field. A "
                    f"geometry must be a state field in 0.4.0: a flux is recomputed "
                    f"within a step and has no single time level. Hold the geometry in "
                    f"{holder!r}'s state.")
            else:
                issues.append(
                    f"ERROR: edge {e.key}: geometry field {field!r} is not in the state "
                    f"of its {anchor} node {holder!r}. State fields: {sorted(fields)}.")
            continue
        value = fields[field]
        dtype = getattr(value, "dtype", None)
        if str(dtype) not in _GEOMETRY_DTYPES:
            issues.append(
                f"ERROR: edge {e.key}: geometry field {holder}.{field} has dtype {dtype}; "
                f"a geometry must be a float32 or float64 array.")
            continue
        shape = tuple(int(n) for n in np.shape(value))
        want = tuple(getattr(e.mapping, "geometry_shape", ()))
        accepts = getattr(e.mapping, "accepts_geometry_shape", None)
        if not (accepts(shape) if callable(accepts) else shape == want):
            issues.append(
                f"ERROR: edge {e.key}: geometry field {holder}.{field} has shape {shape}, "
                f"but mapping {e.mapping!r} reads a geometry of shape {want}.")
            continue
        problems = getattr(e.mapping, "geometry_dtype_problems", None)
        if callable(problems):
            found: Any = problems(dtype)
            errors, advisories = found
            issues.extend(f"ERROR: edge {e.key}: {text}." for text in errors)
            issues.extend(f"WARNING: edge {e.key}: {text}." for text in advisories)
    return issues


def _refuse_unpreserved_geometry(edges, name: str, original, replacement) -> None:
    """Raise ``ValueError`` if replacing node *name* (*original*) by
    *replacement* would drop a geometry an edge reads from it.

    For each edge anchored on *name*, the replacement's
    ``initial_state()`` must hold the geometry field with the same shape
    and a float32 or float64 dtype.  Asked before anything is removed, so
    a refusal leaves the graph as it was.  Replacing the other end of a
    geometry edge is not this function's concern.
    """
    new_state = None
    for e in edges:
        if e.geometry is None:
            continue
        anchor, field = e.geometry
        if (e.source_node if anchor == "source" else e.target_node) != name:
            continue
        old = original.initial_state().get(field)
        if old is None:
            continue        # never valid: ``validate()`` reports it
        if new_state is None:
            new_state = replacement.initial_state()
        new = new_state.get(field)
        if (new is None or tuple(np.shape(new)) != tuple(np.shape(old))
                or str(getattr(new, "dtype", None)) not in _GEOMETRY_DTYPES):
            raise ValueError(
                f"replace_node({name!r}): edge {e.key} reads its geometry from "
                f"{name}.{field} (shape {tuple(np.shape(old))}, {old.dtype}), which the "
                f"replacement {type(replacement).__name__} does not hold. Nothing was "
                f"changed."
            )


def _geometry_edge_keys(edges) -> list[str]:
    """The keys of the edges of *edges* that read a geometry."""
    return [e.key for e in edges if e.geometry is not None]


def _apply_edge(edge: EdgeSpec, value, params, geom=None):
    """Mapping (interface transfer) first, then the scalar transform.

    The step's edge rule (:func:`maddening.core.edge._delivered`) with the
    step's resolved parameters: the mapping weights are ``params``'s
    ``mappings`` entry for the edge, baked or traced as the step has them.
    *geom* is the geometry of a geometry-dependent mapping
    (:func:`_edge_geom`); an edge without one ignores it.
    """
    return _delivered(edge, value, None if params is None else params.mappings, geom)


def _scheduled_timesteps(nodes, coupling_groups) -> dict[str, float]:
    """Each node's timestep as ``compile()`` schedules it.

    A node's own timestep, except that every member of a ``subcycling=True``
    coupling group is scheduled at the group's largest member timestep: the
    group solves once per macro step and sub-steps its faster members inside
    that solve (:func:`_group_dividers`), so from outside the group every
    member advances by the macro timestep at a time.  ``compile()`` derives
    the rate dividers from these and :attr:`GraphManager.timestep` the step,
    so the two cannot disagree.
    """
    scheduled = {name: spec.timestep for name, spec in nodes.items()}
    for group in coupling_groups:
        if not group.subcycling:
            continue
        members = [n for n in group.nodes if n in nodes]
        if not members:
            continue
        macro = max(nodes[n].timestep for n in members)
        for n in members:
            scheduled[n] = macro
    return scheduled


def _step_duration(scheduled) -> float:
    """The simulated time one compiled step advances: the GCD of *scheduled*.

    *scheduled* is :func:`_scheduled_timesteps`'s mapping.  ``RuntimeError``
    when it is empty (a graph with no nodes takes no step).
    """
    timesteps = sorted(set(scheduled.values()))
    if not timesteps:
        raise RuntimeError("No nodes registered.")
    if len(timesteps) == 1:
        return timesteps[0]
    return _multi_gcd(timesteps)


class _StepState(dict):
    """The user state an ``EVENT_STEP`` observer is handed, carrying
    ``advance``: the simulated time that step covered, or ``None`` for a
    step of the graph's own :attr:`GraphManager.timestep`.

    A ``dict`` in every other respect, so an observer that reads the state
    is unaffected.  :meth:`GraphManager.run_adaptive` steps by a varying
    ``dt``, and the relays added ``timestep`` for each of its steps: twelve
    adaptive steps to t = 1 s of a 1/64 s graph were streamed as 0.1875 s.
    """

    __slots__ = ("advance",)

    def __init__(self, state: dict, advance: Optional[float] = None) -> None:
        super().__init__(state)
        self.advance = advance


_EMPTY_EXTERNAL_INPUTS: dict[str, dict] = {}


def _declared_input_dtypes(external_inputs) -> dict[str, dict]:
    """``{node: {field: dtype}}`` of the declared external inputs, each
    dtype as JAX runs it (a ``float64`` declaration is ``float32`` without
    ``jax_enable_x64``, as the zeros of an omitted input already were)."""
    out: dict[str, dict] = {}
    for ei in external_inputs:
        out.setdefault(ei.target_node, {})[ei.target_field] = (
            jax.dtypes.canonicalize_dtype(ei.dtype))
    return out


def _cast_external_inputs(external_inputs, declared: dict[str, dict]):
    """``external_inputs`` with every declared leaf in its declared dtype.

    Called at the top of every step program, so the declaration is what
    runs whichever door the value came in by: ``step``, a scan, an
    ensemble, the raw compiled step, the FMU bridge (which casts on the
    host and meets a no-op here) and the REST server.  The declared dtype
    used to be applied only to the zeros of an omitted input and by the
    FMU export, so an x64 graph ran ``0.1`` as float64 where its FMU ran
    ``float32(0.1)``.

    A leaf that already has the dtype, strongly typed, is passed through
    untouched: the cast adds nothing to the program of a graph stepped
    with arrays of the declared dtype (the committed step programs).  A
    weakly typed one (a Python scalar) becomes strongly typed, like the
    zeros it replaces.
    """
    if not declared or not external_inputs:
        return external_inputs
    out = None
    for node, fields in external_inputs.items():
        want = declared.get(node)
        if not want:
            continue
        for field, value in fields.items():
            dtype = want.get(field)
            if dtype is None or (
                getattr(value, "dtype", None) == dtype
                and not getattr(value, "weak_type", False)
            ):
                continue
            if out is None:
                out = {n: dict(f) for n, f in external_inputs.items()}
            out[node][field] = jnp.asarray(value, dtype=dtype)
    return external_inputs if out is None else out


def _input_cast_changes(value, dtype) -> Optional[str]:
    """What kind of value ``value`` is, when casting it to the declared
    ``dtype`` changes the number the step would otherwise have run; else
    ``None``.

    The comparison is with the value as JAX takes it in (a ``numpy.float64``
    is already float32 without ``jax_enable_x64``), so nothing is reported
    for a narrowing JAX itself always made.  A concrete value is compared
    exactly, on the host: ``0.5`` into a float32 input changes nothing,
    ``0.1`` from float64 does.  A traced value has no number to compare and
    is judged by dtype alone.
    """
    have = getattr(value, "dtype", None)
    if have is not None and have == dtype:
        return None
    if isinstance(value, jax.core.Tracer):
        return None if _holds_every_value(have, dtype) else f"a traced {have}"
    try:
        arr = np.asarray(value)
        have = jax.dtypes.canonicalize_dtype(arr.dtype)
    except Exception:  # noqa: BLE001 - not a numeric leaf: the step will say so
        return None
    if have == dtype or _holds_every_value(have, dtype):
        return None
    # No warnings filter is touched here (they are per-process, and graphs
    # run in threads): NumPy's floating-point complaints about a cast are
    # ``errstate``'s, which is per-thread, and the one cast that warns
    # through ``warnings`` -- complex to real -- is not made.
    with np.errstate(all="ignore"):
        try:
            before = arr.astype(have)
            if have.kind == "c" and np.dtype(dtype).kind != "c":
                if bool(np.any(np.imag(before) != 0)):
                    return f"a {have}"
                before = np.real(before)
                have = before.dtype
            after = before.astype(dtype)
            # Back in the dtype it came from, so the comparison is exact:
            # NumPy would compare an int64 with a float64 as float64s.
            same = bool(np.array_equal(after.astype(have), before, equal_nan=True))
            if same and have.kind in "iu" and dtype.kind == "f":
                # ... and the way back is defined only inside the integer's
                # range (a uint64 2**64 - 1 is the float 2**64).
                info = np.iinfo(have)
                same = bool(np.all(after < float(2 ** info.bits if have.kind == "u"
                                                 else 2 ** (info.bits - 1))))
        except Exception:  # noqa: BLE001
            same = False
    return None if same else f"a {have}"


def _holds_every_value(have, dtype) -> bool:
    """Does ``dtype`` hold every value of ``have``?  NumPy's "safe" cast,
    less the one it gets wrong for this purpose: a 64-bit integer into a
    float64, whose mantissa is 53 bits."""
    have, dtype = np.dtype(have), np.dtype(dtype)
    if have.kind in "iu" and dtype.kind == "f":
        bits = have.itemsize * 8 - (have.kind == "i")
        return bits <= np.finfo(dtype).nmant + 1
    return bool(np.can_cast(have, dtype, "safe"))

# Key for internal multi-rate metadata in the full state dict.
_META_KEY = "_meta"
#: State and checkpoint keys a node may not be named: the graph's own state
#: (``_meta``) and a checkpoint's params prefixes
#: (``maddening.core.simulation.checkpoint``).
_RESERVED_STATE_KEYS = frozenset({_META_KEY, "_params", "_params_mappings"})


def _uncarriable_characters(text: str) -> list[str]:
    """The characters of *text* that some place a name is written cannot
    hold, as code points (``U+0000``), each once, in the order met.

    A name is written to a checkpoint's member names, to a config (JSON,
    and whatever a caller writes ``to_dict()`` as), to a USD stage and to
    an FMU's model description (XML 1.0).  What one of them cannot carry,
    measured on each:

    * a NUL ends a checkpoint member's name, so the archive is written and
      does not load, and a USD string keeps the name only up to it;
    * XML 1.0 has no way to write U+0000 to U+001F (but tab, line feed and
      carriage return) or U+FFFE and U+FFFF, escaped or not: the model
      description is written and no parser reads it;
    * a surrogate (U+D800 to U+DFFF) cannot be encoded as UTF-8, so no
      file holds it, and the tracer refuses it as a name.

    So: U+0000 to U+001F but tab, line feed and carriage return; the
    surrogates; U+FFFE and U+FFFF.  That is exactly what is not a character
    of XML 1.0, the narrowest of the carriers.  Every other character -- a
    space, a dot, a quote, a line break, U+007F to U+009F, any letter of
    any script -- is written to all of them and read back as itself, and is
    taken: a name is not refused for being unusual.
    """
    found: list[str] = []
    for ch in text:
        point = ord(ch)
        if ((point < 0x20 and ch not in "\t\n\r") or 0xD800 <= point <= 0xDFFF
                or point in (0xFFFE, 0xFFFF)):
            label = f"U+{point:04X}"
            if label not in found:
                found.append(label)
    return found


#: What a refusal says of the characters :func:`_uncarriable_characters`
#: finds, after naming them.
_UNCARRIABLE_WHY = (
    "a name must not contain U+0000 to U+001F (but tab, line feed and "
    "carriage return), a surrogate, or U+FFFE / U+FFFF, because not every "
    "place a name is written can hold one (a checkpoint's member names end "
    "at a NUL, an FMU's model description cannot carry these control "
    "characters, and no file can carry a surrogate)"
)


def _node_name_refusal(name: Any) -> Optional[str]:
    """Why ``add_node`` refuses *name*, or ``None``.

    One rule for every door a node's name comes in by -- ``add_node``, and
    so ``from_dict``, a USD stage and ``POST /graph/nodes`` (which asks
    before it builds anything).  A name is refused where it is introduced
    when some place it is later written could not hold it: the checkpoint
    of a node named with a NUL was saved and did not load.
    """
    if not isinstance(name, str):
        return (f"Node name {name!r} is invalid: a name is a string, not "
                f"{type(name).__name__}")
    bad = [t for t in ("/", "#", "->") if t in name]
    if not name or bad:
        # These tokens delimit checkpoint keys, mapping slots and edge
        # keys; a node name containing them corrupts those namespaces.
        return (f"Node name {name!r} is invalid: must be non-empty and must "
                f"not contain {bad or ['/', '#', '->']}")
    uncarriable = _uncarriable_characters(name)
    if uncarriable:
        return (f"Node name {name!r} is invalid: it contains "
                f"{', '.join(uncarriable)}, and {_UNCARRIABLE_WHY}.")
    if name in _RESERVED_STATE_KEYS:
        # The graph's own state lives under ``_meta`` (coupling and
        # multirate carries) and a checkpoint keeps the params under
        # ``_params`` and ``_params_mappings``.  A node named for one of
        # them was taken (POST /graph/nodes answered 201), the next
        # compile dropped its state, every step was a KeyError and a
        # checkpoint save was refused until the node was deleted.
        return (f"Node name {name!r} is invalid: it is a key the graph "
                f"reserves for its own state and checkpoints "
                f"({', '.join(sorted(_RESERVED_STATE_KEYS))}).  A different "
                f"spelling ({name.lstrip('_')!r}, say) is fine.")
    from maddening.serialization.json_codec import (  # noqa: PLC0415
        NON_FINITE_TOKENS,
    )
    if name in NON_FINITE_TOKENS:
        # MADD-ANO-010: the JSON surfaces refuse a string that spells a
        # non-finite token, and a node name is a JSON *value* in
        # ``to_dict`` (``nodes[i]["name"]``) and in any mapping point
        # reference.  It reached the stage untouched, though, because
        # ``save_graph_to_usd`` writes it to a typed USD String
        # attribute that never sees the codec -- so a ``.usda`` could
        # round-trip to a graph that could not be written as a config,
        # and the same graph was refused or accepted depending on which
        # surface it met.  Refused here instead, at the point of entry,
        # which is what the anomaly's own workaround recommends
        # ("validate names ... where they are accepted, not where they
        # are saved") and what makes the three surfaces agree.
        return (f"Node name {name!r} is invalid: it spells a non-finite "
                f"JSON token, which the serialisers reserve (MADD-ANO-010), so "
                f"a graph holding it could not be written as a config or "
                f"referenced from an interface mapping.  A different spelling "
                f"({name.lower()!r}, say) is fine.")
    return None


def _field_name_refusal(name: Any) -> Optional[str]:
    """Why an edge or an external input cannot name the field *name*, or
    ``None``: the rule of :func:`_node_name_refusal` for the names a graph
    is given beside its nodes'.

    A field's name is written where a node's is -- it is a JSON *value* in
    ``to_dict`` (``edges[i]["target_field"]``), half of an edge's key and so
    of a mapping's slot in ``params["mappings"]`` and of a checkpoint
    member -- and the target field of an edge is the caller's to choose:
    ``validate`` passes one the target node does not declare, because a
    node may read an input it does not declare.  So it is asked here.

    * The text ``NaN``, ``Infinity`` or ``-Infinity`` is how the config
      writes a non-finite float, and the reader of a config turns each back
      into one wherever it stands: an edge to a field of that name was
      taken, and ``to_dict`` (``GET /graph``) then raised for as long as
      the edge was there.
    * ``#`` ends the edge's key and starts a mapped edge's ordinal
      (``a.x->b.y#1``), so with one in a field's name two mapped edges on
      the same pair shared one slot of weights, and ``remove_edge`` left
      the slot behind.
    * The characters of :func:`_uncarriable_characters`.

    ``.``, ``->`` and ``/`` are carried: nothing splits an edge's key at
    them (a checkpoint splits its member names at the *last* ``/``, and an
    FMU records the node and the field of a variable beside its name).
    """
    if not isinstance(name, str):
        return f"a field's name is a string, not {type(name).__name__}"
    from maddening.serialization.json_codec import (  # noqa: PLC0415
        NON_FINITE_TOKENS,
    )
    if name in NON_FINITE_TOKENS:
        return ("it spells a non-finite JSON token, which the serialisers "
                "reserve (MADD-ANO-010), so a graph holding it could not be "
                f"written as a config.  A different spelling ({name.lower()!r}, "
                "say) is fine")
    if "#" in name:
        return ("it contains '#', which ends an edge's key and starts a "
                "mapped edge's ordinal")
    uncarriable = _uncarriable_characters(name)
    if uncarriable:
        return f"it contains {', '.join(uncarriable)}, and {_UNCARRIABLE_WHY}"
    return None


def _holds_tracer(state: dict) -> bool:
    """Whether *state* came out of a JAX transform rather than a run.

    Every node is asked, and every field of it.  A transform's output is
    traced only where it depends on what the transform differentiates: a
    node that reads no parameter (a clock, a driver) comes back as
    concrete arrays beside traced ones, and so does an integer field, and
    so do most ``_meta`` slots.  This used to ask the first entry only,
    on the premise that nodes are traced together or not at all, and a
    state that has been through a ``lax.scan`` has its keys sorted -- so
    a first node that reads no parameter, or ``_meta`` itself whenever the
    node names sort after ``"_"``, answered "not traced".  The graph then
    kept the tracers, and the next ``step()`` or ``coupling_diagnostics()``
    after ``jax.grad`` of a ``run_scan`` raised ``UnexpectedTracerError``.
    ``_meta`` is skipped: it is not a node, and a graph whose nodes are
    all concrete is not holding a transform's output.  The scan stops at
    the first tracer, and in the common untraced case it is one
    ``isinstance`` per field, next to the per-node dict copies
    ``_user_state`` already makes every step.
    """
    tracer = jax.core.Tracer
    for key, value in state.items():
        if key == _META_KEY:
            continue
        if isinstance(value, dict):
            for leaf in value.values():
                if isinstance(leaf, tracer):
                    return True
        elif isinstance(value, tracer):
            return True
    return False


def _outside_jax_trace() -> bool:
    """Whether no JAX transform is currently active.

    Best effort, on a private JAX helper: when it is not there the
    answer is "cannot tell", which every caller reads as "do not
    intervene".  Being wrong in that direction costs the old behaviour
    (JAX's own ``UnexpectedTracerError`` later), never a wrong number.
    """
    try:
        from jax._src import core as _jax_core
        return bool(_jax_core.trace_state_clean())
    except Exception:            # pragma: no cover - JAX internals moved
        return False


# ------------------------------------------------------------------
# Floating-point-tolerant GCD
# ------------------------------------------------------------------

#: Relative tolerance of the timestep GCD: a Euclidean remainder at or below
#: this fraction of the largest timestep is representation noise, not a
#: smaller common step.  Decimal timesteps carry a few float64 ulps of noise
#: (``0.3 % 0.1`` is ``0.0999...98``, then ``2.8e-17``), far below it, and a
#: genuine common step is a decimal digit away from the noise, far above it.
#: Until 0.4.0 the tolerance was an *absolute* ``1e-9`` (seconds, in effect),
#: so a graph whose timesteps were themselves near a nanosecond stopped the
#: algorithm before its first step: nodes at ``1e-9`` and ``2e-9`` got a base
#: step of ``2e-9``, the fast node advanced ``1e-9`` per base step while the
#: slow one advanced ``2e-9``, and their clocks drifted apart with no error.
#: Relative to the largest timestep it is the old ``1e-9`` exactly for a
#: graph whose largest timestep is one second.
_GCD_RTOL = 1e-9  # units: dimensionless, a fraction of the largest timestep


def _float_gcd(a: float, b: float, rtol: float = _GCD_RTOL, scale: Optional[float] = None) -> float:
    """GCD of two positive floats: Euclid's algorithm, stopping on a remainder
    at or below ``rtol * scale`` (``scale`` defaults to ``max(a, b)``)."""
    if a < b:
        a, b = b, a
    tol = rtol * (a if scale is None else scale)
    while b > tol:
        a, b = b, a % b
    return a


def _multi_gcd(values: Sequence[float], rtol: float = _GCD_RTOL) -> float:
    """GCD of multiple positive floats, to ``rtol`` of the largest of them."""
    scale = max(values)
    result = values[0]
    for v in values[1:]:
        result = _float_gcd(result, v, rtol, scale)
    return result


#: How closely ``divider * base_dt`` must reproduce a node's scheduled
#: timestep, as a fraction of that timestep, for the multi-rate schedule to
#: be compiled.  The float GCD is exact to a few ulps for timesteps a few
#: decades apart and loses about one digit per decade of their ratio; a
#: node whose clock the schedule would run more than a part in a million
#: fast or slow is refused instead (:func:`_rate_dividers`).
_RATE_RTOL = 1e-6  # units: dimensionless, a fraction of the node's timestep


def _rate_dividers(scheduled) -> dict:
    """``{node: divider}`` of a multi-rate graph: the node is stepped every
    *divider*-th base step, the base step being :func:`_step_duration` of
    *scheduled* (:func:`_scheduled_timesteps`'s mapping).

    ``ValueError`` when the schedule would not keep a node's clock: its
    divider rounds to zero, or ``divider * base_dt`` misses its timestep by
    more than :data:`_RATE_RTOL` of it.  The GCD takes a remainder at or
    below ``1e-9`` of the largest timestep for float noise, so a timestep
    below that fraction of the largest *was* the noise: nodes at ``1.0``
    and ``1e-10`` got a base step of ``1.0`` and dividers of 1 and 0, the
    fast node was stepped once per base step with its own ``1e-10``, and
    after four steps its clock read ``4e-10`` against ``4.0`` -- add,
    compile, validate and run all succeeding.  Timesteps a little closer
    together could leave a divider that was a third off (``1.0`` and
    ``1.5e-9``: a base of ``1e-9`` and a divider of 2).  The dividers of a
    graph whose schedule is kept are computed as they always were.
    """
    base_dt = _step_duration(scheduled)
    dividers = {name: round(dt / base_dt) for name, dt in scheduled.items()}
    slowest = max(scheduled, key=lambda name: scheduled[name])
    for name, dt in scheduled.items():
        divider = dividers[name]
        if divider >= 1 and abs(divider * base_dt - dt) <= _RATE_RTOL * dt:
            continue
        if divider < 1:
            how = (f"it is not above {_GCD_RTOL:g} of the largest timestep, which the "
                   f"common step ({base_dt!r}) is computed to, so it would be stepped "
                   f"once per step of {base_dt!r} and advance its own {dt!r} each time")
        else:
            how = (f"the common step found for them is {base_dt!r}, and stepping it "
                   f"every {divider} of those advances the graph's clock by "
                   f"{divider * base_dt!r} for each {dt!r} of its own")
        raise ValueError(
            f"nodes {name!r} (timestep {dt!r}) and {slowest!r} (timestep "
            f"{scheduled[slowest]!r}) cannot be scheduled together: {how}.  Give the "
            "nodes timesteps that are whole multiples of one common step, no more "
            f"than about {1 / _GCD_RTOL:g} times apart."
        )
    return dividers


def _multi_device_mesh(nodes, state, graph_mesh=None):
    """The device mesh a step spans when it spans more than one device, else ``None``.

    A sharded node's mesh (the ``Sharded*Node`` convention: a ``_mesh``
    attribute), the graph's own (:meth:`GraphManager.enable_multigpu`), or
    the mesh of a state field placed with a ``NamedSharding`` over more than
    one device -- the first found.  :func:`_strict_error_if` raises on every
    device of it.
    """
    from jax.sharding import NamedSharding  # noqa: PLC0415

    candidates = [getattr(spec.node, "_mesh", None) for spec in nodes.values()]
    candidates.append(graph_mesh)
    for fields in state.values():
        for leaf in (fields.values() if isinstance(fields, dict) else ()):
            sharding = getattr(leaf, "sharding", None)
            if isinstance(sharding, NamedSharding):
                candidates.append(sharding.mesh)
    for mesh in candidates:
        if mesh is not None and int(np.prod(mesh.devices.shape)) > 1:
            return mesh
    return None


def _drain_mesh(mesh) -> None:
    """Wait until every device of *mesh* has finished what it was running.

    After :func:`_strict_error_if` raised on every device, Python sees the
    first device's error while the others may still be running their own
    raising callback; a process that exits then can abort in teardown
    ("terminate called without an active exception": 3 runs in 6 of a bare
    four-device CPU program that exited right after catching the error; no
    run of 22 through the graph's entry points, with or without this).  A
    computation over every device of the mesh runs on each after what came
    before it, so blocking on one waits them all out.  Defensive: it costs
    nothing on the path that does not raise.
    """
    from jax.sharding import NamedSharding, PartitionSpec  # noqa: PLC0415

    n = int(np.prod(mesh.devices.shape))
    probe = jax.device_put(np.zeros(n, np.float32),
                           NamedSharding(mesh, PartitionSpec(tuple(mesh.axis_names))))
    try:
        jax.block_until_ready(jax.jit(jnp.sum)(probe))
    except Exception:  # noqa: BLE001  (a drain must not replace the error it follows)
        pass


# How many built scan programs one graph keeps between compiles.
_SCAN_CACHE_MAX = 64
