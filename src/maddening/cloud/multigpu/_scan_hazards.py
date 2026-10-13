"""The kernel pattern XLA miscompiles inside a loop (MADD-ANO-068), found in a jaxpr.

On jaxlib 0.10.2, 0.11.0 and 0.11.2 (CPU), a
:class:`~maddening.cloud.multigpu.sharded_node.ShardedStencilNode` step
traced inside ``lax.scan``, ``fori_loop`` or ``while_loop`` is compiled
wrongly for one kernel pattern: the node reads a sharded ``StaticArray``
in its halo, reads another array through a window at a ``shard_info``
offset, and the static is replicated over a mesh axis of two or more
devices.  On a mesh axis the ``axis_map`` leaves unused every cell comes
back 1e-3 to 1e-2 off, silently; on a ``(2, 2)`` pencil whose static is
split along one axis the step fails to compile.  The same step under
``jit`` alone is right.  Disabling XLA's algebraic simplifier removes it,
and it reproduces without MADDENING.  There is no workaround that does
not rely on XLA declining to fold an expression, so
:class:`~maddening.core.graph_manager.GraphManager` refuses the paths
that put such a step inside a loop.

This module decides, from the node's own ``update_padded``, whether a
static is read that way.  The wrapper traces ``update_padded`` once with
its sharded statics and its ``shard_info`` offsets as explicit inputs
(:func:`analyse_update_padded`) and asks, per static:

* **halo** -- does any live use of the static read outside its interior?
  Only a ``slice`` (or a ``dynamic_slice`` at literal starts) lying
  wholly inside ``[halo, n - halo)`` along the static's shard axis counts
  as an interior read; any other use, a call into a ``jit``-ed helper
  included, counts as a halo read.  A static the step never reads
  (``HeatNode``'s ``grid_x``) is read nowhere.
* **window** -- does a live ``dynamic_slice``, ``dynamic_update_slice``,
  ``gather`` or ``scatter`` take an index derived from a ``shard_info``
  offset (or from ``lax.axis_index``) into an array that is not derived
  from this static alone?  A window into the static itself is not
  counted: the 2-D field of the multi-device validation goals takes its
  own columns out of its static that way, inside ``run_scan`` and a
  coupling group, and is right on every mesh measured.

Both are conservative in the direction of refusing: an operation the
walk does not recognise reads the halo, and a call it cannot follow
argument by argument passes every input's taint to every output.  What
it cannot see is a node that computes a shard offset some other way
(from ``self`` attributes set outside the trace, say); see the anomaly's
registry entry for what is and is not covered.

The probe (:func:`probe_scan_hazards`) is how the graph collects the
answers: while it is active, every ``ShardedStencilNode`` whose
``update`` is traced records its hazards against itself.  It is a trace
of the graph's step under ``jax.eval_shape`` -- the wrappers see the
boundary inputs the step really hands them -- and it never compiles.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Iterator, Optional

#: The registry entry every refusal names.
ANOMALY_ID = "MADD-ANO-068"

_PROBE: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "maddening_scan_hazard_probe", default=None)


@contextlib.contextmanager
def probe_scan_hazards() -> Iterator[dict]:
    """Collect ``{id(wrapper): (wrapper, [ScanHazard, ...])}`` while active.

    A ``ShardedStencilNode`` whose ``update`` is traced inside the block
    adds an entry for itself (an empty list when it found nothing), so a
    wrapper that is absent was never reached.
    """
    found: dict = {}
    token = _PROBE.set(found)
    try:
        yield found
    finally:
        _PROBE.reset(token)


def active_probe() -> Optional[dict]:
    """The collector of the innermost active :func:`probe_scan_hazards`, if any."""
    return _PROBE.get()


@dataclass(frozen=True)
class ScanHazard:
    """One sharded static a wrapped node reads in the miscompiled pattern.

    Attributes
    ----------
    node : str
        The wrapped node's name.
    node_type : str
        The wrapped node's class name.
    static : str
        The ``static_data`` key of the sharded ``StaticArray``.
    shard_axis : int
        The static's own shard axis.
    replicated_over : tuple of (str, int)
        The mesh axes of two or more devices the static is replicated
        over, with their device counts, in mesh order.
    window : str
        The primitive through which ``update_padded`` reads another array
        at a ``shard_info`` offset; ``""`` when the node could not be
        analysed at all (then every static counts).
    """

    node: str
    node_type: str
    static: str
    shard_axis: Optional[int]
    replicated_over: tuple
    window: str

    def describe(self) -> str:
        """One sentence for a refusal message."""
        axes = ", ".join(f"{a!r} ({n} devices)" for a, n in self.replicated_over)
        if not self.window:
            return (f"{self.node_type} {self.node!r} could not be analysed, so its "
                    f"sharded static {self.static!r} (replicated over mesh axis {axes}) "
                    "is taken to be read in the miscompiled pattern")
        return (f"{self.node_type} {self.node!r} reads its sharded static "
                f"{self.static!r} in its halo and another array through a window at a "
                f"shard_info offset ({self.window}), and {self.static!r} is "
                f"replicated over mesh axis {axes}")


# ---------------------------------------------------------------------------
# The jaxpr walk
# ---------------------------------------------------------------------------

#: A taint: (data sources, derived from a shard offset).  Data sources are
#: ``("static", key)`` for a sharded static and ``"other"`` for anything
#: else that is data (state, boundary inputs, params, closed-over arrays).
_NONE: tuple = (frozenset(), False)
_OTHER: tuple = (frozenset({"other"}), False)
_OFFSET: tuple = (frozenset(), True)

#: Call-like primitives whose callee takes the caller's inputs one for one.
_CALLS = frozenset({"jit", "pjit", "closed_call", "core_call", "remat", "checkpoint",
                    "custom_jvp_call", "custom_vjp_call", "custom_vjp_call_jaxpr"})


def _literal_type():
    from jax.extend.core import Literal  # noqa: PLC0415 - jax.extend is lazy
    return Literal


def _union(taints) -> tuple:
    data: frozenset = frozenset()
    off = False
    for d, o in taints:
        data = data | d
        off = off or o
    return data, off


def _open(jaxpr):
    """The open jaxpr of a ``Jaxpr`` or ``ClosedJaxpr``."""
    return getattr(jaxpr, "jaxpr", jaxpr)


def _callee(eqn):
    """The callee of a call-like equation, when it maps inputs one for one."""
    if eqn.primitive.name not in _CALLS:
        return None
    for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        sub = eqn.params.get(key)
        if sub is None:
            continue
        inner = _open(sub)
        if (len(inner.invars) == len(eqn.invars)
                and len(inner.outvars) == len(eqn.outvars)):
            return inner
    return None


def _sub_jaxprs(eqn) -> list:
    from jax.extend.core import jaxprs_in_params  # noqa: PLC0415
    return [_open(j) for j in jaxprs_in_params(eqn.params)]


def _live_vars(jaxpr) -> set:
    """Variables of ``jaxpr`` on a path to an output or an effect.

    Conservative: a live equation keeps every input.
    """
    Literal = _literal_type()
    live = {v for v in jaxpr.outvars if not isinstance(v, Literal)}
    for eqn in reversed(jaxpr.eqns):
        if eqn.effects or any(o in live for o in eqn.outvars):
            live.update(v for v in eqn.invars if not isinstance(v, Literal))
    return live


def _window(eqn, ins) -> Optional[tuple]:
    """``(primitive, operand data)`` when ``eqn`` indexes by a shard offset."""
    name = eqn.primitive.name
    if name == "dynamic_slice":
        operand, index = ins[:1], ins[1:]
    elif name == "dynamic_update_slice":
        operand, index = ins[:2], ins[2:]
    elif name == "gather":
        operand, index = ins[:1], ins[1:2]
    elif name.startswith("scatter"):
        operand, index = ins[:1] + ins[2:], ins[1:2]
    else:
        return None
    if not _union(index)[1]:
        return None
    return name, _union(operand)[0]


def _walk(jaxpr, in_taints, const_taint, live_all: bool, windows: list) -> list:
    """Propagate taints through ``jaxpr``; append every live window found.

    Returns the taints of ``jaxpr.outvars``.  ``live_all`` treats every
    equation as live (inside a callee of a live equation, where the walk
    does not narrow liveness).
    """
    Literal = _literal_type()
    env: dict = {}
    for v, t in zip(jaxpr.invars, in_taints):
        env[v] = t
    for v in jaxpr.constvars:
        env[v] = const_taint
    live = None if live_all else _live_vars(jaxpr)

    def read(v):
        return _NONE if isinstance(v, Literal) else env.get(v, _OTHER)

    for eqn in jaxpr.eqns:
        ins = [read(v) for v in eqn.invars]
        eqn_live = live_all or bool(eqn.effects) or any(
            o in live for o in eqn.outvars)  # type: ignore[operator]
        found = _window(eqn, ins) if eqn_live else None
        if found is not None:
            windows.append(found)
        sink = windows if eqn_live else []
        callee = _callee(eqn)
        if callee is not None:
            outs = _walk(callee, ins, _OTHER, True, sink)
        else:
            u = _union(ins)
            for sub in _sub_jaxprs(eqn):
                _walk(sub, [u] * len(sub.invars), _union([u, _OTHER]), True, sink)
            outs = [u] * len(eqn.outvars)
        if eqn.primitive.name == "axis_index":
            outs = [_OFFSET] * len(eqn.outvars)
        elif not any(not isinstance(v, Literal) for v in eqn.invars):
            # Made in place (an iota, a filled array): data, not an offset.
            outs = [_OTHER] * len(eqn.outvars)
        for o, t in zip(eqn.outvars, outs):
            env[o] = t
    return [read(v) for v in jaxpr.outvars]


def _interior_read(eqn, var, axis: int, halo: int, n: int) -> bool:
    """Is ``eqn`` a slice of ``var`` lying wholly inside its interior along ``axis``?"""
    Literal = _literal_type()
    if eqn.invars[0] is not var or sum(v is var for v in eqn.invars) != 1:
        return False
    name = eqn.primitive.name
    if name == "slice":
        lo = int(eqn.params["start_indices"][axis])
        hi = int(eqn.params["limit_indices"][axis])
    elif name == "dynamic_slice" and all(isinstance(v, Literal) for v in eqn.invars[1:]):
        size = int(eqn.params["slice_sizes"][axis])
        # dynamic_slice clamps its start into [0, n - size].
        lo = min(max(int(eqn.invars[1 + axis].val), 0), n - size)
        hi = lo + size
    else:
        return False
    return halo <= lo and hi <= n - halo


def _reads_halo(jaxpr, var, axis: int, halo: int, n: int, live: set) -> bool:
    """Does a live use of ``var`` read outside ``[halo, n - halo)`` along ``axis``?"""
    if var not in live:
        return False
    if any(o is var for o in jaxpr.outvars):
        return True
    for eqn in jaxpr.eqns:
        if not any(v is var for v in eqn.invars):
            continue
        if not (eqn.effects or any(o in live for o in eqn.outvars)):
            continue
        if not _interior_read(eqn, var, axis, halo, n):
            return True
    return False


def analyse_update_padded(closed, layout) -> dict[str, tuple[bool, list]]:
    """Per sharded static: ``(read in its halo, [windows into other arrays])``.

    Parameters
    ----------
    closed : ClosedJaxpr
        ``update_padded`` traced as a function of ``(statics, offsets)``:
        its first ``len(layout)`` inputs are the halo-padded sharded
        statics, in ``layout`` order, and the rest are the ``shard_info``
        offsets.  Everything else the node reads (state, boundary inputs,
        params, closed-over arrays) is a constant of the jaxpr.
    layout : sequence of (key, axis, halo, n)
        For each static: its key, its shard axis, the halo the wrapper
        padded it with along that axis (``0`` when it was not
        exchanged), and its padded extent there.

    Returns
    -------
    dict
        ``{key: (halo_read, windows)}``; ``windows`` lists
        ``(primitive, data sources)`` for every live offset-indexed read
        of an array not derived from ``key`` alone.
    """
    jaxpr = closed.jaxpr
    n_static = len(layout)
    statics = list(jaxpr.invars[:n_static])
    in_taints = ([(frozenset({("static", key)}), False) for key, *_ in layout]
                 + [_OFFSET] * (len(jaxpr.invars) - n_static))
    windows: list = []
    _walk(jaxpr, in_taints, _OTHER, False, windows)
    live = _live_vars(jaxpr)
    out: dict[str, tuple[bool, list]] = {}
    for (key, axis, halo, n), var in zip(layout, statics):
        halo_read = halo > 0 and _reads_halo(jaxpr, var, axis, halo, n, live)
        own = frozenset({("static", key)})
        foreign = [w for w in windows if not w[1] <= own]
        out[key] = (halo_read, foreign)
    return out


def window_label(windows: list) -> str:
    """A short description of the windows found, for a message."""
    names = sorted({name for name, _ in windows})
    return ", ".join(names) if names else ""
