"""A strict fingerprint of everything a graph holds, for "exactly as it was".

``rest_oracle.snapshot`` compares what a client can observe: the config,
the parameters, the state, the files, the clock.  This compares the objects
themselves.  :func:`fingerprint` walks from its roots through every
``dict``, ``list``, ``set`` and ``tuple`` and through the ``__dict__`` of
every object that has one, and records, by path,

* for a container: its identity, its type and what it holds, in order;
* for anything else: its identity and its type -- and for a NumPy array,
  which can change in place, its bytes.

Two fingerprints are equal exactly when every container reachable then is
reachable now, is the same object, and holds the same objects in the same
order.  A JAX array cannot change, so "the same object" is "the same bits".

Written apart from ``maddening.core._graph_transaction`` on purpose, and
stricter: it opens *every* object with a ``__dict__`` (the transaction
opens nodes, this package's classes and dataclasses), so an object the
transaction shares as a leaf is still seen here if a request changes it.

It reads no property of the graph (``gm.params`` takes in pending node
writes, which is a change), only ``__dict__``.
"""

from __future__ import annotations

import hashlib
import types
from typing import Any

import numpy as np

#: Shared code and foreign machinery: identity only, never walked into.
_CLOSED = (type, types.ModuleType, types.FunctionType, types.MethodType,
           types.BuiltinFunctionType)
_CLOSED_MODULES = ("jax", "jaxlib", "numpy", "threading", "_thread", "concurrent",
                   "logging", "weakref", "functools")


class Fingerprint(dict):
    """``{path: record}``, holding every object it saw (so no ``id`` it
    recorded can be handed to another object while it is compared)."""

    def __init__(self) -> None:
        super().__init__()
        self.alive: list = []


def _opens(obj: Any):
    if isinstance(obj, _CLOSED) or callable(obj):
        return None
    module = type(obj).__module__ or ""
    if module.split(".", 1)[0] in _CLOSED_MODULES:
        return None
    try:
        attrs = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        return None
    return attrs if type(attrs) is dict else None


def fingerprint(*roots: Any) -> Fingerprint:
    """The fingerprint of everything reachable from *roots*."""
    out = Fingerprint()
    seen: dict[int, str] = {}
    stack = [(f"<{i}>", root) for i, root in enumerate(roots)]
    while stack:
        path, obj = stack.pop()
        out.alive.append(obj)
        if obj is None or type(obj) in (int, float, str, bool, bytes, complex):
            # Compared by value: the same number is not always one object.
            out[path] = ("value", type(obj).__name__, repr(obj))
            continue
        if id(obj) in seen:
            out[path] = ("again", seen[id(obj)])
            continue
        seen[id(obj)] = path
        record: tuple = (id(obj), type(obj).__qualname__)
        children: list = []
        if isinstance(obj, dict):
            items = list(dict.items(obj))
            out.alive.append(items)
            record += ("dict", tuple(repr(k) for k, _ in items))
            children += [(f"{path}[{k!r}]", v) for k, v in items]
        elif isinstance(obj, (list, tuple)):
            record += ("sequence", len(obj))
            children += [(f"{path}[{i}]", v) for i, v in enumerate(obj)]
        elif isinstance(obj, (set, frozenset)):
            record += ("set", tuple(sorted(repr(x) for x in obj)))
        elif isinstance(obj, np.ndarray):
            record += ("ndarray", obj.dtype.str, obj.shape,
                       hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest())
        attrs = _opens(obj)
        if attrs is not None:
            items = list(attrs.items())
            out.alive.append(items)
            record += ("attrs", tuple(k for k, _ in items))
            children += [(f"{path}.{k}", v) for k, v in items]
        out[path] = record
        stack.extend(children)
    return out


def served_fingerprint(served) -> Fingerprint:
    """Of a served graph, and of what its server keeps about it: the
    surrogates' record, the relay's clock and frame, whether the relay
    observes the graph."""
    server = served.server
    relay = server.relay
    return fingerprint(
        served.gm, server._original_nodes, server._active_surrogates,  # noqa: SLF001
        server._relay_attached,  # noqa: SLF001
        relay._snapshot, relay._sim_time, relay._step_count, relay._timestep,  # noqa: SLF001
        relay._elapsed)  # noqa: SLF001


def fingerprint_differences(before: Fingerprint, after: Fingerprint, limit: int = 12) -> list[str]:
    """One line per path whose record differs, shortest paths first."""
    paths = sorted((p for p in set(before) | set(after) if before.get(p) != after.get(p)),
                   key=lambda p: (p.count("[") + p.count("."), p))
    return [f"{p}: {before.get(p, '<absent>')!r} -> {after.get(p, '<absent>')!r}"[:400]
            for p in paths[:limit]]


def assert_exactly_as_it_was(before: Fingerprint, after: Fingerprint, what: str) -> None:
    found = fingerprint_differences(before, after)
    assert not found, f"{what} did not leave the graph exactly as it was: " + "; ".join(found)
