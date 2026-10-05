"""Silence warnings in one thread: the one place MADDENING touches the filters.

MADDENING runs a node's code a second time in several places only to learn
something about it -- which parameters a compiled step reads, whether a
constructor takes a new value, what shape ``initial_state()`` builds.  What
that code warns was said, or will be said, on the real run; said again from
a probe it is noise, and where warnings are errors it would turn "this
write is fine" into "cannot tell".  So a probe runs with warnings silenced.

Until 0.4.0 shipped, each of those sixteen probes did it the usual way::

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ...

which is wrong in a process with more than one thread, twice over:

* ``warnings.filters`` is process-wide.  For as long as one thread was
  inside a probe, **every** warning of **every** thread was dropped --
  another graph's compile-time advisory, a refused write, a precision
  note -- with nothing to show for it.
* ``catch_warnings`` saves the filter list on the way in and puts it back
  on the way out, and is not thread-safe.  Two probes that overlapped in
  two threads and left in the order they entered put back each other's
  saved list, and the second one's was the first one's ``ignore``: the
  process kept a filter ignoring every warning **for good**
  (MADD-ANO-197).  Independent graphs stepped from a thread each did that
  in 1 to 9 of 40 rounds.

:func:`quiet_warnings` replaces all of them.  It never saves or restores
the filter list, and what it puts in the list ignores nothing by itself:

* one filter, ``("ignore", <per-thread matcher>, Warning, None, 0)``,
  whose *message* slot -- where a filter usually holds a compiled regular
  expression, asked ``.match(text)`` -- holds a ``threading.local``.  In a
  thread inside a ``quiet_warnings()`` block its ``match`` accepts every
  text; in every other thread it accepts none.  So the block silences the
  thread that opened it and no other, and a copy of the filter that
  outlives the block (see "What this cannot do") is inert;
* the filter is inserted at the front of ``warnings.filters`` when a block
  opens (one ``list.insert``) and removed when the last open block of the
  process closes (``list.remove``), both under :data:`_FILTER_LOCK`.  The
  lock is held for those list operations only -- never while the block's
  body runs -- so nothing a probe calls (a node's ``update``, a JAX trace,
  a route holding the REST server's graph lock) can wait on it, and there
  is no lock order to get wrong;
* the matcher runs no Python code (a ``threading.local`` attribute read
  and a C-level callable), so a thread scanning the filters for a warning
  of its own is never suspended inside it, and the insert and the remove
  are each atomic for such a scan.

``tests/core/test_quiet_warnings.py`` pins each of those, and
``test_only_the_helper_touches_the_warnings_filters`` there fails if any
other module of ``src/`` uses ``catch_warnings``, ``simplefilter``,
``filterwarnings``, ``resetwarnings`` or ``warnings.filters``.

What this cannot do
-------------------
It cannot make ``warnings.catch_warnings()`` safe for **other** code that
uses it from several threads of the same process -- your own, or a
library's.  Two such blocks can still put back each other's filters,
whatever MADDENING does.  Where one of them overlaps a ``quiet_warnings()``
block the consequences for MADDENING are bounded:

* a foreign block that opened before a probe and closes during it puts
  back a list without the probe's filter, so the rest of that probe is not
  silenced: what the node's code warns there is shown (and raised, where
  warnings are errors, which a probe reads as "cannot tell");
* a foreign block that opened during a probe and closes after it puts back
  a list that still holds the probe's filter.  That copy ignores nothing
  (no thread is inside a block) and is removed the next time the last
  block closes.

Neither leaves anything that silences a warning.  See "Warnings and
threads" in ``docs/user_guide/troubleshooting.md`` for the libraries on
MADDENING's own paths that use ``catch_warnings``.

Only the default build of CPython is covered: with the global interpreter
lock a ``list.insert`` is atomic.  A free-threaded build, and an
interpreter run with ``-X context_aware_warnings`` (3.14), keep their
filters elsewhere; neither has been tried.
"""

from __future__ import annotations

import contextlib
import functools
import operator
import threading
import warnings
from typing import Iterator

__all__: list[str] = []

#: Compared by identity with a warning's text, which it never is.
_NO_TEXT = object()

#: ``match(text)`` for a thread outside every block: false for any object,
#: a ``str`` or not (``warnings.warn(123)`` is legal).  A C-level callable
#: -- ``functools.partial`` of ``operator.is_`` -- so that asking it runs no
#: Python code; see the module docstring.
_MATCH_NOTHING = functools.partial(operator.is_, _NO_TEXT)

#: ``match(text)`` for a thread inside a block: true for any object.
_MATCH_EVERYTHING = functools.partial(operator.is_not, _NO_TEXT)


class _ThisThread(threading.local):
    """Per-thread state, and the *message* slot of :data:`_FILTER`.

    The warnings machinery asks a filter's message slot ``.match(text)``;
    reading ``match`` off a ``threading.local`` answers for the calling
    thread.  No ``__init__``: one would run, as Python code, on a thread's
    first read.
    """

    #: Blocks this thread is inside.
    depth = 0
    #: What the filter answers for this thread; an instance attribute
    #: (:data:`_MATCH_EVERYTHING`) while ``depth`` is positive.  A
    #: ``staticmethod``, so that reading the class's default through an
    #: instance hands back the callable as it is on every interpreter: a
    #: bare ``functools.partial`` there warns on 3.13 (from inside the
    #: warnings machinery, which then recurses until the stack ends) and is
    #: bound like a method from 3.14.
    match = staticmethod(_MATCH_NOTHING)


_THREAD = _ThisThread()

#: The one filter MADDENING ever puts in ``warnings.filters``.
_FILTER = ("ignore", _THREAD, Warning, None, 0)

#: Serialises MADDENING's own changes to ``warnings.filters`` and
#: :data:`_open_blocks`.  Re-entrant (a finaliser run by an allocation
#: inside the guarded region may open a block of its own), and held for a
#: counter update and a list operation only: never across a block's body.
_FILTER_LOCK = threading.RLock()

#: ``quiet_warnings()`` blocks open in the whole process.
_open_blocks = 0


def _block_opened() -> None:
    """Count the block and see :data:`_FILTER` at the front of the filters."""
    global _open_blocks
    with _FILTER_LOCK:
        _open_blocks += 1
        filters = warnings.filters
        if not filters or filters[0] is not _FILTER:
            # A copy further back (a filter somebody put ahead of it since)
            # stays where it is until the last block closes: taking it out
            # here would leave the threads inside a block uncovered for a
            # moment.
            filters.insert(0, _FILTER)


def _block_closed() -> None:
    """Uncount the block; the last one out removes every copy of the filter."""
    global _open_blocks
    with _FILTER_LOCK:
        _open_blocks -= 1
        if _open_blocks > 0:
            return
        filters = warnings.filters
        while True:
            try:
                filters.remove(_FILTER)
            except ValueError:
                return


@contextlib.contextmanager
def quiet_warnings() -> Iterator[None]:
    """Ignore every warning raised **in this thread** inside the block.

    The replacement for ``warnings.catch_warnings()`` followed by
    ``warnings.simplefilter("ignore")``, for a region that runs a node's
    code only to learn something about it.  Other threads keep their
    warnings, the process's filters are the ones before once no block is
    open, and blocks may nest and overlap across threads in any order.
    Errors are not touched: only warnings are silenced.

    A warning ignored here is not remembered as shown, so the same warning
    raised later, outside the block, is delivered as if for the first time.
    """
    _block_opened()
    _THREAD.depth += 1
    _THREAD.match = _MATCH_EVERYTHING
    try:
        yield
    finally:
        _THREAD.depth -= 1
        if _THREAD.depth == 0:
            _THREAD.match = _MATCH_NOTHING
        _block_closed()
