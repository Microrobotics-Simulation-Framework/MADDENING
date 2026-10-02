"""What a node would allocate, told from its constructor arguments alone.

Private on purpose: an internal convention between the built-in nodes, the
REST server (:mod:`maddening.api.server`) and
:meth:`~maddening.core.graph_manager.GraphManager.from_dict`, not yet part
of the node contract.  Promote it (and document it for node authors) only
after a release has shown it earns a public name.

Why it exists
-------------
Every check that refuses a node *by its size* used to run after the node
was built: ``POST /graph/nodes`` measured the initial state it had just
allocated, and ``PUT /graph/params`` called the node's constructor with the
new value to see whether it took it.  A constructor that turns an argument
into an array dimension therefore allocated the whole thing before any
refusal.  A ``WaveletAdaptiveNode`` given ``n_levels=10_000_000`` built a
Python list of ``n_coarse * 2**n_levels`` labels and grew one process to
57.7 GB before the kernel's OOM killer stopped it; a D3Q19 ``LBMNode`` of
140^3 cells took 1.1 GB before the state cap refused it.

The hook
--------
A node class whose array sizes come from its constructor arguments -- a
cell count, a grid shape, a number of refinement levels -- defines::

    @classmethod
    def _allocation_estimate(cls, args: dict) -> AllocationEstimate | None

``args`` holds the constructor's keyword arguments with its defaults
applied (:func:`constructor_arguments`); the constructor has not been
called.  The hook returns what the constructor and ``initial_state()``
would allocate, computed without allocating anything, or ``None`` when the
arguments are not ones the constructor would take (it is left to refuse
them with its own message).  It must be cheap, must not raise, and must
stay cheap for absurd arguments: clamp an exponent rather than compute
``2 ** 10**7``.

A class without the hook is unaffected: every caller treats "no estimate"
as "cannot be told before building", which is the behaviour before 0.4.0.
"""

from __future__ import annotations

import inspect
import math
import os
from collections.abc import Mapping
from typing import Any, NamedTuple, Optional


class AllocationEstimate(NamedTuple):
    """What constructing a node and building its initial state allocates.

    Attributes
    ----------
    state_elements : int
        Scalars across every field ``initial_state()`` returns -- exactly
        what ``POST /graph/nodes`` measures on the built state and bounds
        by :data:`maddening.api.server.MAX_NODE_STATE_ELEMENTS`.
    peak_bytes : int
        The peak memory the constructor and ``initial_state()`` take
        together, estimated from what they build (a dense operator, the
        temporaries of an equilibrium) -- within a small factor, never an
        exact figure.  May be astronomically large; it is a Python int.
    """

    state_elements: int
    peak_bytes: int


#: Exponents above this many bits are clamped by the estimates: anything
#: that large is refused whatever its exact size, and computing it exactly
#: (``(n_coarse * 2**10**7) ** 3``) would itself take seconds.
SATURATION_BITS = 256


def as_count(value: Any) -> Optional[int]:
    """*value* as a non-negative integer count, or ``None``.

    An ``int`` (not a ``bool``), a NumPy integer, or a finite float with no
    fractional part -- a JSON round trip turns ``64`` into ``64.0``, and
    some constructors take that.  Anything else is ``None``: not a count
    this estimate can read, so the constructor decides.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        out = value
    else:
        try:
            import numpy as np  # noqa: PLC0415 - keep this module import-light
        except ImportError:  # pragma: no cover - numpy is a hard dependency
            np = None
        if np is not None and isinstance(value, np.integer):
            out = int(value)
        elif isinstance(value, float) or (
                np is not None and isinstance(value, np.floating)):
            as_float = float(value)
            if not math.isfinite(as_float) or not as_float.is_integer():
                return None
            out = int(as_float)
        else:
            return None
    return out if out >= 0 else None


def constructor_arguments(node_cls: Any, params: Mapping[str, Any]) -> Optional[dict]:
    """The keyword arguments ``node_cls(name=..., timestep=..., **params)``
    would run with, defaults applied, or ``None`` when the call would not
    bind (an unknown keyword: the constructor's own ``TypeError`` decides).

    Arguments collected by a ``**kwargs`` parameter are merged in by name.
    """
    if not isinstance(params, Mapping):
        return None
    try:
        sig = inspect.signature(node_cls)
    except (TypeError, ValueError):
        return None
    try:
        bound = sig.bind_partial(**params)
    except TypeError:
        return None
    bound.apply_defaults()
    out: dict = {}
    for name, value in bound.arguments.items():
        kind = sig.parameters[name].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            out.update(value)
        elif kind is not inspect.Parameter.VAR_POSITIONAL:
            out[name] = value
    return out


def estimate_allocation(node_cls: Any, params: Mapping[str, Any]) -> Optional[AllocationEstimate]:
    """What ``node_cls(name=..., timestep=..., **params)`` and its
    ``initial_state()`` would allocate, or ``None`` when the class does not
    say (no ``_allocation_estimate`` hook), the arguments do not bind, or
    the hook cannot read them.  Nothing is constructed.
    """
    hook = getattr(node_cls, "_allocation_estimate", None)
    if hook is None or not callable(hook):
        return None
    args = constructor_arguments(node_cls, params)
    if args is None:
        return None
    try:
        estimate = hook(args)
    except Exception:  # noqa: BLE001 - a hook must not raise; "cannot tell"
        return None
    if estimate is None:
        return None
    try:
        state, peak = estimate
    except (TypeError, ValueError):
        return None
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (state, peak)):
        return None
    return AllocationEstimate(state, peak)


def physical_memory_bytes() -> Optional[int]:
    """This machine's physical memory in bytes, or ``None`` when the
    platform does not say (``os.sysconf`` is POSIX)."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    if not isinstance(pages, int) or not isinstance(page_size, int) \
            or pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def format_count(n: int) -> str:
    """*n* for a message: digits up to about 10^15, a power of two beyond
    (``str()`` of a huge int is slow, and refused past 4300 digits)."""
    if n < 10 ** 15:
        return str(n)
    return f"at least 2^{n.bit_length() - 1}"


def format_bytes(n: int) -> str:
    """*n* bytes for a message, in TiB / GiB / MiB, or a power of two when huge."""
    if n >= 2 ** 60:
        return f"at least 2^{n.bit_length() - 1} bytes"
    if n >= 2 ** 40:
        return f"{n / 2 ** 40:.1f} TiB"
    if n >= 2 ** 30:
        return f"{n / 2 ** 30:.1f} GiB"
    return f"{n / 2 ** 20:.1f} MiB"


def refuse_beyond_memory(node_cls: Any, name: Any, params: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when building node *name* of *node_cls* with
    *params* would take more memory than this machine has.

    For :meth:`~maddening.core.graph_manager.GraphManager.from_dict`, which
    builds every node a config names: a config is untrusted input, and one
    naming a node no machine can hold used to be found out by the OOM
    killer rather than by an error.  This refuses only what cannot fit
    (the estimate against physical memory), so every graph that loaded
    before still loads.  A class without an estimate, or a platform that
    does not report its memory, is not checked.
    """
    estimate = estimate_allocation(node_cls, params)
    if estimate is None:
        return
    memory = physical_memory_bytes()
    if memory is None or estimate.peak_bytes <= memory:
        return
    cls_name = getattr(node_cls, "__name__", str(node_cls))
    raise ValueError(
        f"node {name!r} ({cls_name}) cannot be built on this machine: with "
        f"these params its constructor and initial state would take about "
        f"{format_bytes(estimate.peak_bytes)} ({format_count(estimate.state_elements)} "
        f"state elements), more than the {format_bytes(memory)} of memory it "
        f"has.  Nothing was constructed; reduce the node's size parameters."
    )
