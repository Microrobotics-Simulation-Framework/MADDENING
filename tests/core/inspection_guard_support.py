"""The read-only guard for ``GraphManager``'s inspection methods.

:func:`assert_read_only` calls a method and proves it changed nothing
and compiled nothing.  "Nothing" is measured, not listed: the guard
fingerprints **every attribute of the graph**, recursively -- the state
and ``_meta`` bit for bit, ``params`` leaf by leaf, every node object
and its ``params`` dict, the dirty flag, the compiled step and scan
cache by identity, the trace counters, the schedule -- so an attribute
nobody thought to list is still covered.  Arrays are compared by
identity *and* by bytes: a method that replaces a leaf with an equal
copy changed the graph as surely as one that wrote a new value.

Compilation and tracing are measured through ``jax.monitoring``: every
JAX trace, lowering and backend compile emits a duration event, and the
guard records them while the method runs.  That covers the step's own
jit (``trace_count`` / ``scan_trace_count`` are checked separately as
well), a fresh ``jax.jit``, and eager ``jax.numpy`` operations, which
JAX compiles once per shape.

Shared by ``tests/core/test_inspection_read_only.py`` and the sharded
case in ``tests/cloud/multigpu/``.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, Callable, Iterator

import jax
import jax.core
import numpy as np

_EVENTS: list[str] = []
_RECORDING = [False]


def _listener(event: str, *_args: Any, **_kwargs: Any) -> None:
    if _RECORDING[0] and ("compile" in event or "trace" in event or "lower" in event):
        _EVENTS.append(event)


# Registered once per process (this module is imported once).
jax.monitoring.register_event_duration_secs_listener(_listener)


@contextlib.contextmanager
def compile_events() -> Iterator[list[str]]:
    """The JAX trace / lowering / compile events fired inside the block."""
    start = len(_EVENTS)
    _RECORDING[0] = True
    seen: list[str] = []
    try:
        yield seen
    finally:
        _RECORDING[0] = False
        seen.extend(_EVENTS[start:])
        del _EVENTS[start:]


def _array_bytes(x: Any) -> bytes:
    if jax.dtypes.issubdtype(x.dtype, jax.dtypes.extended):
        return np.asarray(jax.random.key_data(x)).tobytes()
    return np.asarray(x).tobytes()


def _fp(obj: Any, depth: int, seen: set[int]) -> Any:
    """A comparable fingerprint of ``obj``: identities, values, bytes."""
    if isinstance(obj, jax.core.Tracer):
        return ("tracer", id(obj), tuple(obj.shape), str(obj.dtype))
    if isinstance(obj, jax.Array):
        return ("array", id(obj), tuple(obj.shape), str(obj.dtype), _array_bytes(obj))
    if isinstance(obj, np.ndarray):
        return ("ndarray", id(obj), obj.shape, str(obj.dtype), obj.tobytes())
    if obj is None or isinstance(obj, (bool, int, float, complex, str, bytes, np.generic)):
        return ("value", type(obj).__name__, repr(obj))
    if depth <= 0 or id(obj) in seen:
        return ("ref", type(obj).__name__, id(obj))
    seen = seen | {id(obj)}
    if isinstance(obj, dict):
        return ("dict", id(obj), tuple((repr(k), _fp(v, depth - 1, seen)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, id(obj) if isinstance(obj, list) else 0,
                tuple(_fp(v, depth - 1, seen) for v in obj))
    if isinstance(obj, (set, frozenset)):
        return (type(obj).__name__, id(obj) if isinstance(obj, set) else 0,
                tuple(sorted(repr(v) for v in obj)))
    from maddening.core.node import SimulationNode  # noqa: PLC0415
    if dataclasses.is_dataclass(obj) or isinstance(obj, SimulationNode):
        attrs = getattr(obj, "__dict__", None)
        if attrs is None:
            attrs = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
        # The attribute dict itself is fingerprinted by content only: for a
        # slotted dataclass it is a temporary, whose id means nothing.
        return ("object", type(obj).__name__, id(obj),
                tuple((k, _fp(v, depth - 1, seen)) for k, v in attrs.items()))
    return ("ref", type(obj).__name__, id(obj))


def fingerprint(gm: Any) -> dict[str, Any]:
    """``{attribute: fingerprint}`` over every attribute of ``gm``, plus the
    public counters and the process-wide x64 flag."""
    out = {name: _fp(value, 10, set()) for name, value in sorted(vars(gm).items())}
    out["<trace_count>"] = gm.trace_count
    out["<scan_trace_count>"] = gm.scan_trace_count
    out["<schedule>"] = tuple(gm.schedule)
    out["<x64>"] = bool(jax.config.read("jax_enable_x64"))
    return out


def assert_read_only(gm: Any, call: Callable[[Any], Any], *, allow_eager_compile: bool = False):
    """Call ``call(gm)`` and assert that the graph is exactly as it was and
    that nothing was compiled or traced.

    ``allow_eager_compile`` permits JAX compile events (and nothing else):
    for a method documented to read through code that dispatches eager
    ``jax.numpy`` operations, which JAX compiles once per shape.  The
    graph's trace counters, compiled step and every other attribute are
    still checked.
    """
    before = fingerprint(gm)
    with compile_events() as events:
        result = call(gm)
    after = fingerprint(gm)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    assert not changed, f"the call changed the graph: {changed}"
    if not allow_eager_compile:
        assert not events, f"the call compiled or traced: {sorted(set(events))}"
    return result
