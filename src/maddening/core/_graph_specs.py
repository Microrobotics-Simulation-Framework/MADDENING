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


def _apply_edge(edge: EdgeSpec, value, params):
    """Mapping (interface transfer) first, then the scalar transform.

    The step's edge rule (:func:`maddening.core.edge._delivered`) with the
    step's resolved parameters: the mapping weights are ``params``'s
    ``mappings`` entry for the edge, baked or traced as the step has them.
    """
    return _delivered(edge, value, None if params is None else params.mappings)


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
