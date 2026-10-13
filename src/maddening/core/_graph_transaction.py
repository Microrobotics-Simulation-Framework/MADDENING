"""Snapshot and restore of everything a :class:`GraphManager` holds.

Private.  What makes a write "all or nothing" for a caller that cannot see
inside it: take a :class:`_Snapshot` before the write, and ``restore()`` it
if the write does not finish.  The REST server's write routes are its user
(``SimulationServer._graph_transaction``).

What a snapshot is
------------------
Not a list of attributes.  A list goes stale the day an attribute is added,
and a restore that misses one is a silent defect.  The snapshot walks the
object graph from its roots and records every **container** it reaches:

* a ``dict``, ``list`` or ``set`` (or a subclass: a node's ``_ParamsDict``):
  the object and a shallow copy of its contents;
* an **opened** object -- a :class:`~maddening.core.node.SimulationNode`,
  any instance of a class defined in this package, or a dataclass instance
  -- through its ``__dict__``, recorded as a dict is.  The graph manager
  itself, its node specs, the nodes (and the nodes they wrap), the edges,
  their mappings and the coupling groups are all reached this way;
* a ``tuple`` is walked (it can hold a container) and not recorded.

Everything else is a **leaf**, shared and never copied: a JAX array (which
cannot change), a number, a string, a function (the compiled step and every
cached scan), and a NumPy array.  So the cost is the number of containers,
not the size of the fields: a state of a million cells is one reference.

``restore()`` refills each recorded container **in place** with what it
held, through the base class's own methods (so a ``_ParamsDict`` counts no
write, and its own counters are put back with the rest of its
``__dict__``).  In place, because the compiled step, a caller and the
graph's own bookkeeping hold these containers by reference: a copy put in
their place would leave them reading the abandoned one.  An object created
since the snapshot is reachable from nothing afterwards.

What it does not cover
----------------------
* A NumPy array (or ``bytearray``) written **in place**: shared, not
  copied.  No route of the server writes one in place.
* An object of a class from outside this package that is neither a node nor
  a dataclass, changing its own attributes (a user-defined mapping that
  mutates itself): a leaf.
* Anything not reachable from the roots: module-level caches, JAX's own
  compilation caches, files, threads, observers already notified.
"""

from __future__ import annotations

import dataclasses
import types
from typing import Any, Iterable, Optional

from maddening.core.node import SimulationNode

_PACKAGE = __name__.split(".", 1)[0] + "."

#: Never opened, whatever defines them: a function, method, module or class
#: is shared code, not state of the graph.
_NEVER_OPENED = (type, types.ModuleType, types.FunctionType, types.MethodType,
                 types.BuiltinFunctionType)


def _opened_dict(obj: Any) -> Optional[dict]:
    """The ``__dict__`` of an object the snapshot opens, or ``None`` for a
    leaf (see the module docstring)."""
    if isinstance(obj, _NEVER_OPENED):
        return None
    cls = type(obj)
    if not (isinstance(obj, SimulationNode)
            or (cls.__module__ or "").startswith(_PACKAGE)
            or dataclasses.is_dataclass(obj)):
        return None
    try:
        attrs = object.__getattribute__(obj, "__dict__")
    except AttributeError:      # __slots__: nothing to rebind
        return None
    return attrs if type(attrs) is dict else None


class _Snapshot:
    """Every container reachable from *roots*, with what it holds now.

    Parameters
    ----------
    roots : iterable
        The objects to start from (a ``GraphManager``; a server adds the
        containers of its own that describe the graph).
    leaves : dict, optional
        Filled with ``{type: one example}`` for every leaf reached: what a
        test reads to say that no unclassified kind of object is shared.
    """

    __slots__ = ("_records",)

    def __init__(self, roots: Iterable[Any], *, leaves: Optional[dict] = None) -> None:
        # (container, shallow copy of its contents, kind); every container
        # is kept alive by its record, so an id() is never reused.
        records: list[tuple[Any, Any, str]] = []
        seen: set[int] = set()
        stack = list(roots)
        while stack:
            obj = stack.pop()
            if obj is None or type(obj) in (int, float, str, bool, bytes, complex):
                if leaves is not None:
                    leaves.setdefault(type(obj), obj)
                continue
            key = id(obj)
            if key in seen:
                continue
            if isinstance(obj, tuple):
                seen.add(key)
                # Kept alive for the walk: a tuple built on the fly would
                # hand its id to the next one.
                records.append((obj, None, "tuple"))
                stack.extend(obj)
                continue
            if isinstance(obj, dict):
                seen.add(key)
                held = dict.copy(obj) if type(obj) is dict else dict(dict.items(obj))
                records.append((obj, held, "dict"))
                stack.extend(held.values())
            elif isinstance(obj, list):
                seen.add(key)
                held = list.copy(obj)
                records.append((obj, held, "list"))
                stack.extend(held)
            elif isinstance(obj, set):
                seen.add(key)
                # Its elements are hashable, so they are not containers.
                records.append((obj, set(obj), "set"))
            attrs = _opened_dict(obj)
            if attrs is not None:
                seen.add(key)
                # The object, not its ``__dict__``: that is fetched again at
                # the restore, and refilled in place.
                held = dict(attrs)
                records.append((obj, held, "object"))
                stack.extend(held.values())
            elif key not in seen:
                seen.add(key)
                if leaves is not None:
                    leaves.setdefault(type(obj), obj)
        self._records = records

    def __len__(self) -> int:
        """How many containers the snapshot recorded."""
        return sum(1 for _, _, kind in self._records if kind != "tuple")

    def restore(self) -> None:
        """Put every recorded container back as it was, in place."""
        for obj, held, kind in self._records:
            if kind == "dict":
                dict.clear(obj)
                dict.update(obj, held)
            elif kind == "list":
                list.__setitem__(obj, slice(None), held)
            elif kind == "set":
                set.clear(obj)
                set.update(obj, held)
            elif kind == "object":
                attrs = object.__getattribute__(obj, "__dict__")
                attrs.clear()
                attrs.update(held)
