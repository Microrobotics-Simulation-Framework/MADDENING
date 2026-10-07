"""Which graph parameters a compiled step reads, and the probes that
decide whether a parameter write can be observed.

Moved verbatim out of ``maddening.core.graph_manager``.  Private.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, Sequence

import jax
# `jax.core` is not re-exported by `jax/__init__.py`, so the attribute
# only resolves for a type checker when the submodule is imported by name.
import jax.core
import jax.numpy as jnp
import numpy as np

from maddening.core.node import SimulationNode
from maddening.core._graph_specs import (
    _NodeSpec,
    _correction_accepts_params,
    _node_fluxes,
    _node_update,
)


# ------------------------------------------------------------------
# Which graph parameters the compiled step reads (structural liveness)
# ------------------------------------------------------------------

def _live_jaxpr_inputs(jaxpr, live_out: Sequence[bool]) -> list[bool]:
    """Which inputs of ``jaxpr`` can reach one of its live outputs.

    A backward walk over the equations.  It is *conservative*: an
    equation whose structure it does not know marks every input live as
    soon as one output is, and an equation with effects keeps its inputs
    whatever happens to its outputs.  So an input reported dead is
    genuinely never read on the way to a live output; an input reported
    live may still be unused.  The graph uses the dead answer to refuse a
    write, which is the direction a conservative answer can only make
    rarer, never wrong.

    The call-like primitives in :data:`_CALL_PRIMITIVES` (``jit`` /
    ``pjit``, ``closed_call``, ``remat``, ``shard_map``, ...) are followed
    into 1:1; ``while`` and ``scan`` are solved to a fixed point over their
    carries; ``cond`` takes the union of its branches; anything else is
    "every input live".  ``custom_jvp_call`` / ``custom_vjp_call`` are not followed
    into, because the derivative rule may read an input the primal does
    not.
    """
    from jax.extend.core import Literal

    live = {
        v for v, keep in zip(jaxpr.outvars, live_out)
        if keep and not isinstance(v, Literal)
    }
    for eqn in reversed(jaxpr.eqns):
        outs = [o in live for o in eqn.outvars]
        effectful = bool(eqn.effects)
        if not any(outs) and not effectful:
            continue
        for v, keep in zip(eqn.invars, _live_eqn_inputs(eqn, outs, effectful)):
            if keep and not isinstance(v, Literal):
                live.add(v)
    return [v in live for v in jaxpr.invars]


def _live_eqn_inputs(eqn, outs: list[bool], effectful: bool) -> list[bool]:
    """``_live_jaxpr_inputs`` for one equation; see there."""
    n_in = len(eqn.invars)
    name = eqn.primitive.name
    params = eqn.params
    everything = [True] * n_in
    if name.startswith("custom_"):
        return everything
    try:
        if name == "while":
            cond = getattr(params["cond_jaxpr"], "jaxpr", params["cond_jaxpr"])
            body = getattr(params["body_jaxpr"], "jaxpr", params["body_jaxpr"])
            cn, bn = params["cond_nconsts"], params["body_nconsts"]
            cond_in = _live_jaxpr_inputs(cond, [True])
            carry = [a or b for a, b in zip(outs, cond_in[cn:])]
            while True:
                body_in = _live_jaxpr_inputs(body, carry)
                grown = [a or b for a, b in zip(carry, body_in[bn:])]
                if grown == carry:
                    break
                carry = grown
            out = cond_in[:cn] + body_in[:bn] + carry
        elif name == "scan":
            body = getattr(params["jaxpr"], "jaxpr", params["jaxpr"])
            if "num_consts" in params:
                nc, ncar = params["num_consts"], params["num_carry"]
            else:
                # jaxlib 0.11 describes the operands as a flat tree of three
                # groups, (consts, carry, xs); ``len`` of a group is its
                # number of flat inputs.  Checked against the operand count
                # so a changed layout falls back to "everything live".
                groups = getattr(params["ft_in"], "elts", None)
                if groups is None or len(groups) != 3:
                    return everything
                nc, ncar = len(groups[0]), len(groups[1])
                if nc + ncar + len(groups[2]) != n_in:
                    return everything
            carry, ys = list(outs[:ncar]), list(outs[ncar:])
            while True:
                body_in = _live_jaxpr_inputs(body, carry + ys)
                grown = [a or b for a, b in zip(carry, body_in[nc:nc + ncar])]
                if grown == carry:
                    break
                carry = grown
            out = body_in[:nc] + carry + body_in[nc + ncar:]
        elif name == "cond":
            ops = [False] * (n_in - 1)
            for branch in params["branches"]:
                branch_in = _live_jaxpr_inputs(getattr(branch, "jaxpr", branch), outs)
                if len(branch_in) != n_in - 1:
                    return everything
                ops = [a or b for a, b in zip(ops, branch_in)]
            out = [True] + ops
        elif name in _CALL_PRIMITIVES:
            subs = [
                getattr(params[key], "jaxpr", params[key])
                for key in ("jaxpr", "call_jaxpr", "fun_jaxpr") if key in params
            ]
            if len(subs) != 1:
                return everything
            sub = subs[0]
            if len(sub.invars) != n_in or len(sub.outvars) != len(outs):
                return everything
            out = _live_jaxpr_inputs(sub, outs)
        else:
            return everything
    except (KeyError, AttributeError, TypeError):
        return everything
    return out if len(out) == n_in else everything


#: Primitives that call their one sub-jaxpr once, operands in order: the
#: walk follows them 1:1.  An allow-list, not "any equation with one
#: sub-jaxpr of matching arity": a loop-like primitive read as a single call
#: under-approximates what its carries read (a value that reaches a live
#: output only on the second iteration looks dead), which is the direction
#: that would refuse a write the step does read.
_CALL_PRIMITIVES = frozenset({
    "pjit", "jit", "closed_call", "core_call", "named_call",
    "remat", "remat2", "checkpoint", "shard_map", "xla_call", "xla_pmap",
})


def _param_leaves_read(step_fn: Callable, state: dict, ext: dict, params: dict) -> set:
    """``{(owner, key)}`` of ``params["nodes"]`` that ``step_fn`` can read.

    Traces ``step_fn(state, ext, params)`` once (no compile, nothing
    executed) and runs :func:`_live_jaxpr_inputs` over the result, every
    output live.  A leaf of a nested pytree entry counts as read when any
    of its sub-leaves is.
    """
    args = (state, ext, params)
    closed = jax.make_jaxpr(step_fn)(*args)
    paths = [p for p, _ in jax.tree_util.tree_flatten_with_path(args)[0]]
    invars = closed.jaxpr.invars
    if len(paths) != len(invars):
        raise ValueError("step inputs do not line up with the traced jaxpr")
    live = _live_jaxpr_inputs(closed.jaxpr, [True] * len(closed.jaxpr.outvars))
    reads: set = set()
    for path, keep in zip(paths, live):
        if (keep and len(path) >= 4 and getattr(path[0], "idx", None) == 2
                and getattr(path[1], "key", None) == "nodes"):
            reads.add((path[2].key, path[3].key))
    return reads


def _hook_outputs(spec: _NodeSpec, state, bi, p) -> list:
    """The traced outputs of a node's own hooks, called the way the graph
    calls them: ``update`` and, where the node has them,
    ``compute_boundary_fluxes`` and ``compute_interface_correction``.

    Only traced values are returned: only those can depend on an input (an
    index in a correction list is a Python int).
    """
    node = spec.node
    outs = [_node_update(spec, state, bi, spec.timestep, p)]
    if type(node).compute_boundary_fluxes is not SimulationNode.compute_boundary_fluxes:
        outs.append(_node_fluxes(spec, state, bi, spec.timestep, p))
    iface = getattr(node, "interface_dof_indices", None)
    if callable(iface) and iface() and _correction_accepts_params(node):
        outs.append(node.compute_interface_correction(
            state, bi, spec.timestep, params=p))
    return [x for x in jax.tree.leaves(outs) if isinstance(x, jax.core.Tracer)]


def _declared_boundary_zeros(node) -> dict:
    """A zero value for every input ``boundary_input_spec()`` declares."""
    declared = node.boundary_input_spec() or {}
    return {
        name: jnp.zeros(tuple(bspec.shape), dtype=bspec.dtype or jnp.float32)
        for name, bspec in declared.items()
    }


def _leaf_layout(leaf: Any) -> tuple[tuple, Any]:
    """``(shape, dtype)`` of a state leaf -- an array, a Python number, or
    the ``ShapeDtypeStruct`` an abstract trace returns -- with the dtype as
    JAX would hold it (a NumPy ``float64`` is ``float32`` without x64)."""
    shape = getattr(leaf, "shape", None)
    shape = tuple(np.shape(leaf)) if shape is None else tuple(shape)
    dtype = getattr(leaf, "dtype", None)
    if dtype is None:
        dtype = jnp.result_type(leaf)
    if jnp.issubdtype(dtype, jax.dtypes.extended):
        # A PRNG key (``key<fry>``): not a NumPy dtype, and a kind of its own.
        return shape, dtype
    return shape, np.dtype(jax.dtypes.canonicalize_dtype(dtype))


def _dtype_kind(dtype: Any) -> Any:
    """The kind of a :func:`_leaf_layout` dtype: NumPy's one-letter kind
    (boolean, integer, float, complex), and for an extended dtype -- a PRNG
    key, which has none -- the dtype itself, so a key is a key of the same
    implementation and nothing else."""
    return getattr(dtype, "kind", dtype)


def _leaf_path(path: tuple) -> str:
    """``node/field`` for a pytree key path."""
    parts = []
    for entry in path:
        for attr in ("key", "idx", "name"):
            if hasattr(entry, attr):
                parts.append(str(getattr(entry, attr)))
                break
        else:
            parts.append(str(entry))
    return "/".join(parts)


def _prng_key_leaves(state: Any) -> list[str]:
    """The leaves of *state* that are PRNG keys (an extended dtype), by
    path.  The REST server refuses a new node that has one: a key has no
    JSON form, so no reply could carry the node's state."""
    return [_leaf_path(path)
            for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]
            if jnp.issubdtype(_leaf_layout(leaf)[1], jax.dtypes.extended)]


def _state_layout_drift(before: Any, after: Any, *, dtypes: bool = True) -> list[str]:
    """How the state layout *after* differs from *before*: one line per
    leaf that is missing, new, of another shape or (with *dtypes*) of
    another kind of dtype (boolean, integer, float, complex, or a PRNG
    key of one implementation); empty when
    the two trees have the same layout.  A dtype's width is not compared:
    with x64 enabled every stock node's update returns ``float64`` for the
    ``float32`` its ``initial_state()`` builds, and the state reloads.

    The comparison behind the REST server's dry run of a new node
    (``POST /graph/nodes``) and behind the check of a stepped state
    (:meth:`GraphManager.step`, ``FmuSidecar.step``): a node whose
    ``update`` returns a leaf of another shape than its
    ``initial_state()`` built broadcasts it at the first step, which was
    stored (MADD-ANO-220).  Host-side: it reads shapes and dtypes, never values,
    so tracers and the abstract values of :func:`jax.eval_shape` compare
    like arrays.
    """
    old = {_leaf_path(p): leaf for p, leaf in jax.tree_util.tree_flatten_with_path(before)[0]}
    new = {_leaf_path(p): leaf for p, leaf in jax.tree_util.tree_flatten_with_path(after)[0]}
    found = []
    for name in old:
        if name not in new:
            found.append(f"{name!r} is missing after the update")
            continue
        shape_before, dtype_before = _leaf_layout(old[name])
        shape_after, dtype_after = _leaf_layout(new[name])
        if shape_before != shape_after:
            found.append(f"{name!r} has shape {shape_before} before the update "
                         f"and {shape_after} after it")
        elif dtypes and _dtype_kind(dtype_before) != _dtype_kind(dtype_after):
            found.append(f"{name!r} has dtype {dtype_before} before the update "
                         f"and {dtype_after} after it")
    found.extend(f"{name!r} appears only after the update" for name in new if name not in old)
    return found


def _node_update_layout_drift(spec: _NodeSpec, state: Any, node_params: Any,
                              boundary_inputs: Optional[dict] = None, *,
                              dtypes: bool = True) -> list[str]:
    """:func:`_state_layout_drift` of one ``update`` of *spec*'s node on
    *state*, traced abstractly the way the graph calls it
    (:func:`~maddening.core._graph_specs._node_update`: *node_params*
    injected where the node takes them).  *boundary_inputs* is what the
    graph would deliver: ``{}`` for a node with no edges, and by default a
    zero for every input the node declares.  Nothing is computed.  Raises
    whatever the node's code raises while it is traced."""
    if boundary_inputs is None:
        try:
            boundary_inputs = _declared_boundary_zeros(spec.node)
        except Exception:  # noqa: BLE001 - the descriptor is advisory
            boundary_inputs = {}
    after = jax.eval_shape(
        lambda st, b, p: _node_update(spec, st, b, spec.timestep, p),
        state, boundary_inputs, node_params)
    return _state_layout_drift(state, after, dtypes=dtypes)


def _refers_to(value: Any, targets: list, shared: dict, depth: int = 0) -> bool:
    """Does ``value`` reach one of ``targets`` (by identity), the dict
    ``shared``, or an object whose ``params`` is ``shared``?  Followed
    through bound methods, ``functools.partial``, closures, ``__wrapped__``
    chains and containers, two levels deep."""
    if depth > 2:
        return False
    if value is shared or any(value is t for t in targets):
        return True
    if isinstance(value, type):
        return False
    try:
        if getattr(value, "params", None) is shared:
            return True
    except Exception:  # noqa: BLE001 - a property that raises holds nothing
        pass
    # A container is searched only when it is small: a long list is data (a
    # mesh, a table), not a place anyone keeps a bound method.
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value) <= 256 and any(
            _refers_to(v, targets, shared, depth + 1) for v in value)
    if isinstance(value, dict):
        return len(value) <= 256 and any(
            _refers_to(v, targets, shared, depth + 1) for v in value.values())
    if not callable(value):
        return False
    held: list = []
    bound = getattr(value, "__self__", None)
    if bound is not None:
        held.append(bound)
    if isinstance(value, functools.partial):
        held.extend([value.func, *value.args, *value.keywords.values()])
    for cell in getattr(value, "__closure__", None) or ():
        try:
            held.append(cell.cell_contents)
        except ValueError:          # an empty cell
            continue
    wrapped = getattr(value, "__wrapped__", None)
    if wrapped is not None:
        held.append(wrapped)
    return any(_refers_to(v, targets, shared, depth + 1) for v in held)


def _node_with_params(node: Any, params: dict) -> Optional[Any]:
    """A shallow copy of ``node`` that reads ``params`` wherever ``node``
    reads its own ``params`` dict, or ``None`` when no faithful copy can be
    made.

    This answers "what would the node compute after a write into its params
    dict?" without writing to it: the graph may be stepping on a runner
    thread (``POST /sim/start``) that reads the live dict, so the original
    is never mutated.  Every node object reachable through attributes whose
    ``params`` *is* the same dict -- a sharded wrapper and the node it wraps
    share one -- is copied the same way, so the copy sees the new values
    wherever the real write would land; a node holding a *copy* of the dict
    (``HybridNode``'s physics node) keeps its own, because the real write
    does not reach it either.

    Refused (``None``) when an attribute holds a callable bound to, or
    closing over, one of the originals (``self._f = jax.jit(self._impl)``):
    that callable reads the original's params, and a copy that kept it would
    report "not read" for a value the node does read.  Also refused for an
    object without ``__dict__``.
    """
    original = getattr(node, "params", None)
    if not isinstance(original, dict):
        return None
    sharing: list = []

    def collect(obj, depth: int) -> bool:
        if depth > 4:
            return False
        if any(obj is s for s in sharing):
            return True
        try:
            attrs = vars(obj)
        except TypeError:
            return False
        sharing.append(obj)
        for value in attrs.values():
            if value is not obj and getattr(value, "params", None) is original \
                    and not isinstance(value, type):
                if not collect(value, depth + 1):
                    return False
        return True

    if not collect(node, 0):
        return None
    copies = {id(obj): object.__new__(type(obj)) for obj in sharing}
    for obj in sharing:
        attrs = dict(vars(obj))
        for name, value in attrs.items():
            if value is original:
                attrs[name] = params
            elif id(value) in copies and any(value is s for s in sharing):
                attrs[name] = copies[id(value)]
            elif _refers_to(value, sharing, original):
                return None
        copies[id(obj)].__dict__.update(attrs)
    return copies[id(node)]


def _param_probe_pair(node: Any, key: str, value: Any) -> Optional[tuple]:
    """``(old, new, descended)``: two copies of the node that answers for a
    write of ``node.params[key] = value`` -- one reading the current params,
    one reading the new value -- or ``None`` when no faithful copy can be
    made at any level.

    The node itself is copied when it can be (:func:`_node_with_params`).
    A wrapper that closes over the node it wraps (the sharded wrappers keep
    a ``shard_map`` closure over it) cannot be; the node it wraps, which
    shares its params dict -- so it is where the write lands -- and builds
    the state and runs the physics the wrapper distributes, is copied
    instead, and ``descended`` says so.  The walk follows only attributes
    whose ``params`` *is* the shared dict, breadth first, a few nodes deep.
    """
    shared = getattr(node, "params", None)
    if not isinstance(shared, dict):
        return None
    for candidate in _params_holders(node):
        old = _node_with_params(candidate, dict(shared))
        new = _node_with_params(candidate, {**shared, key: value})
        if old is not None and new is not None:
            return old, new, candidate is not node
    return None


def _params_holders(node: Any) -> list:
    """``node``, then every node it wraps that shares its params dict
    (breadth first, a few deep): the objects a ``node.params`` write lands
    on, outermost first."""
    shared = getattr(node, "params", None)
    if not isinstance(shared, dict):
        return []
    out: list = []
    candidates: list = [node]
    while candidates and len(out) < 8:
        candidate = candidates.pop(0)
        if any(candidate is seen for seen in out):
            continue
        out.append(candidate)
        for attr in list(getattr(candidate, "__dict__", {}).values()):
            if attr is not candidate and not isinstance(attr, type) \
                    and getattr(attr, "params", None) is shared:
                candidates.append(attr)
    return out


def _unprobeable_write_reason(node: Any) -> str:
    """The refusal for a write whose use cannot be established by a copy."""
    return (
        f"no copy of {type(node).__name__} that reads the new value can be "
        "made (it, or a node it wraps that shares its params, holds a "
        "callable bound to or closing over itself), so whether anything it "
        "computes while it runs would read the value cannot be told, and a "
        "write that cannot be shown to be used is not accepted"
    )


def _static_deps_reason(node: Any, key: str) -> Optional[str]:
    """Why ``node`` cannot read a new value of ``key`` from its params, by
    its own :meth:`~maddening.core.node.SimulationNode.static_data_deps`
    declaration, or ``None`` when it declares no static built from ``key``."""
    # `Callable[..., Any] | None`, not `Any`: `callable()` narrows a bare
    # `Any` to `(...) -> object`, whose result has no `.items()`.
    deps: Callable[..., Any] | None = getattr(node, "static_data_deps", None)
    declared = (deps() if callable(deps) else None) or {}
    statics = sorted(s for s, names in declared.items() if key in names)
    if not statics:
        return None
    return (
        f"{type(node).__name__} bakes it into static_data "
        f"{statics} when the node is constructed (static_data_deps): "
        "the compiled step reads the static, not the parameter"
    )


def _leaf_values_equal(a, b) -> bool:
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    if len(la) != len(lb):
        return False
    for x, y in zip(la, lb):
        xa, ya = np.asarray(x), np.asarray(y)
        if xa.shape != ya.shape:
            return False
        try:
            if not np.array_equal(xa, ya, equal_nan=True):
                return False
        except TypeError:
            if not np.array_equal(xa, ya):
                return False
    return True


def _short_value(value) -> str:
    """A parameter value for a message: the number(s) if few, else the shape."""
    arr = np.asarray(value)
    if arr.size > 8:
        return f"<array of shape {arr.shape}>"
    return np.array2string(arr, precision=7, separator=", ")


def _write_counts(params) -> Optional[dict]:
    """``{key: writes}`` of a node's params mapping since its lineage began
    (``_ParamsDict._write_counts``: its own counted writes and those of the
    mappings it replaced), or ``None`` for a mapping that does not count."""
    counter = getattr(params, "_write_counts", None)
    if not callable(counter):
        return None
    counts = counter()
    return counts if isinstance(counts, dict) else None
