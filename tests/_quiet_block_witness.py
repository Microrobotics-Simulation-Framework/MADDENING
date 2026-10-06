"""A witness that node code can call to say what a warnings-silencing block
around it does to the rest of the process.

MADDENING runs a node's code a second time, with warnings silenced, in
sixteen places (:mod:`maddening.core._quiet_warnings` says why).  Each is a
``with`` block in ``src/``; :func:`silencing_blocks` finds them by parsing
the source, so a block added later is found too.  A test gives the graph a
node whose constructor, ``initial_state()`` and ``update()`` call
:func:`witness`, and drives the public path that reaches the block.  From
inside the node's code, the witness

* walks the call stack for the blocks it is running inside (by file and
  line range, so no block has to be named by hand);
* raises a warning **in another thread** and notes whether that thread got
  it (:class:`Canary`, which the test has made an error): a block that
  silences the whole process drops it;
* raises a warning of its own and notes whether the block silenced it
  (:class:`SaidInsideAProbe`, also an error outside a block).

Both spellings of a silencing block are found -- ``quiet_warnings()`` and
the ``warnings.catch_warnings()`` it replaced -- so the same tests, run
against a tree that still uses the second, fail on what that spelling does
and not on a name they cannot import.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import pathlib
import sys
import threading
import warnings
from dataclasses import dataclass, field
from typing import Iterator

import maddening

SRC_ROOT = pathlib.Path(maddening.__file__).resolve().parent

#: The helper itself: its own ``with`` blocks are not probes.
HELPER = "core/_quiet_warnings.py"

#: What a ``with`` item calls when it silences warnings.
_SILENCERS = ("quiet_warnings", "catch_warnings")


class Canary(UserWarning):
    """Raised in a thread that is inside no block: it must be delivered."""


class SaidInsideAProbe(UserWarning):
    """Raised by node code inside a block: it must be silenced."""


@dataclass(frozen=True)
class Block:
    """One silencing ``with`` block of ``src/``."""

    path: str            #: relative to the package root, POSIX
    function: str        #: dotted names of the enclosing scopes
    ordinal: int         #: 1-based position among that function's blocks
    first_line: int
    last_line: int

    @property
    def name(self) -> str:
        return f"{self.path}::{self.function}#{self.ordinal}"


def _called_name(node: ast.expr) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)


@functools.lru_cache(maxsize=None)
def _blocks_under(root: pathlib.Path) -> tuple[Block, ...]:
    found: list[Block] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel == HELPER:
            continue
        text = path.read_text()
        if not any(name in text for name in _SILENCERS):
            continue        # a block names what it calls: nothing to parse
        tree = ast.parse(text, filename=str(path))
        scopes: list[str] = []
        counts: dict[str, int] = {}

        class Visitor(ast.NodeVisitor):
            def _scope(self, node):
                scopes.append(node.name)
                self.generic_visit(node)
                scopes.pop()

            visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scope

            def _with(self, node):
                if any(_called_name(item.context_expr) in _SILENCERS for item in node.items):
                    function = ".".join(scopes) or "<module>"
                    counts[function] = counts.get(function, 0) + 1
                    found.append(Block(rel, function, counts[function],
                                       node.lineno, node.end_lineno or node.lineno))
                self.generic_visit(node)

            visit_With = visit_AsyncWith = _with

        Visitor().visit(tree)
    return tuple(found)


def silencing_blocks(root: pathlib.Path = SRC_ROOT) -> list[Block]:
    """Every ``with quiet_warnings():`` (or ``with warnings.catch_warnings():``)
    block under *root*, the helper's own module excepted.  Parsed once per
    process."""
    return list(_blocks_under(root))


@dataclass
class Sighting:
    """What one call of :func:`witness` saw from inside a block."""

    where: str                   #: the node method that called
    another_thread_heard: bool   #: a warning of another thread was delivered
    silenced_here: bool          #: a warning of this thread was not


@dataclass
class Watch:
    """The sightings of one :func:`watching` block, by silencing block."""

    blocks: list[Block]
    seen: dict[str, list[Sighting]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def visited(self) -> set[str]:
        return set(self.seen)

    def deaf(self) -> dict[str, list[str]]:
        """Blocks inside which another thread's warning was dropped."""
        return {name: [s.where for s in sightings if not s.another_thread_heard]
                for name, sightings in self.seen.items()
                if not all(s.another_thread_heard for s in sightings)}

    def loud(self) -> dict[str, list[str]]:
        """Blocks inside which the node's own warning was not silenced."""
        return {name: [s.where for s in sightings if not s.silenced_here]
                for name, sightings in self.seen.items()
                if not all(s.silenced_here for s in sightings)}


_watch: Watch | None = None


def _blocks_on_the_stack(blocks: list[Block]) -> list[Block]:
    inside = []
    frame = sys._getframe(2)
    by_file: dict[str, list[Block]] = {}
    for block in blocks:
        by_file.setdefault(str(SRC_ROOT / block.path), []).append(block)
    while frame is not None:
        for block in by_file.get(frame.f_code.co_filename, ()):
            if (block.first_line <= frame.f_lineno <= block.last_line
                    and frame.f_code.co_name == block.function.rsplit(".", 1)[-1]):
                inside.append(block)
        frame = frame.f_back
    return inside


def another_thread_hears_its_own_warning() -> bool:
    """Raise :class:`Canary` in a new thread; was it delivered there?
    (Call with ``error`` in force for the category.)"""
    heard: list[bool] = []

    def warn():
        try:
            warnings.warn(Canary("a warning raised in another thread"))
        except Canary:
            heard.append(True)

    thread = threading.Thread(target=warn, daemon=True)
    thread.start()
    thread.join(30)
    return bool(heard)


def witness(where: str) -> None:
    """Called by node code.  Inside a silencing block (and a
    :func:`watching` block of the test), record what the block does;
    anywhere else, do nothing."""
    watch = _watch
    if watch is None:
        return
    inside = _blocks_on_the_stack(watch.blocks)
    if not inside:
        return
    heard = another_thread_hears_its_own_warning()
    try:
        warnings.warn(SaidInsideAProbe(f"{where}: said by the node's code"))
        silenced = True
    except SaidInsideAProbe:
        silenced = False
    with watch.lock:
        for block in inside:
            watch.seen.setdefault(block.name, []).append(Sighting(where, heard, silenced))


@contextlib.contextmanager
def watching() -> Iterator[Watch]:
    """Arm :func:`witness`: both of its warnings are errors wherever no
    block silences them.  One at a time, in the thread that runs the test."""
    global _watch
    assert _watch is None, "watching() does not nest"
    watch = Watch(silencing_blocks())
    with warnings.catch_warnings():
        warnings.simplefilter("error", Canary)
        warnings.simplefilter("error", SaidInsideAProbe)
        _watch = watch
        try:
            yield watch
        finally:
            _watch = None
