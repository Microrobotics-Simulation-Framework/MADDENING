"""``quiet_warnings()`` silences the thread that asked, and nothing else.

MADDENING runs a node's code a second time, with warnings silenced, to
learn something about it (which parameters a step reads, whether a
constructor takes a value).  It used to do so with
``warnings.catch_warnings()`` and ``simplefilter("ignore")``.  The filter
list is process-wide and that block saves and restores it, so while one
thread probed every thread's warnings were dropped, and two probes that
overlapped in two threads could leave an ``ignore`` for every warning in
the process for good (MADD-ANO-197).
:func:`maddening.core._quiet_warnings.quiet_warnings` replaces every such
block; this module pins what it promises, one test per promise, and pins
with a source scan that nothing else in ``src/`` touches the filters.

The tests that drive real graphs through the blocks are in
``test_warning_probes_across_threads.py``.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import threading
import warnings

import pytest

import maddening
from maddening.core import _quiet_warnings as qw
from maddening.core._quiet_warnings import quiet_warnings

#: How long a thread of these tests waits for another before it gives up.
WAIT = 30  # units: s


class Loud(UserWarning):
    """A warning of these tests' own."""


@pytest.fixture
def every_warning_is_an_error():
    """A filter list of the test's own: ``error`` for everything, so a
    warning that is not silenced raises where it is given."""
    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.simplefilter("error")
        yield


def _raises_here(message="a warning", category=Loud) -> bool:
    try:
        warnings.warn(message, category)
    except Warning:
        return True
    return False


def _in_a_thread(fn):
    """``fn()`` in a new thread; its result, or the exception it raised."""
    box: list = []

    def run():
        try:
            box.append(("ok", fn()))
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box.append(("raised", exc))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(WAIT)
    assert box, "the thread never finished"
    kind, value = box[0]
    if kind == "raised":
        raise value
    return value


def _copies() -> int:
    return sum(1 for item in warnings.filters if item is qw._FILTER)


# ---------------------------------------------------------------------------
# What a block does
# ---------------------------------------------------------------------------


def test_a_block_silences_the_thread_that_opened_it(every_warning_is_an_error):
    """Inside the block nothing this thread warns is raised -- a message
    that is text, one that is not (``warnings.warn(123)`` is legal), and a
    warning instance -- and after it everything is again."""
    assert _raises_here()
    with quiet_warnings():
        assert not _raises_here()
        assert not _raises_here(123)
        assert not _raises_here(b"bytes")
        assert not _raises_here(Loud(["not", "text"]), None)
        assert not _raises_here("a deprecation", DeprecationWarning)
    assert _raises_here()
    assert _raises_here(123)


def test_another_thread_keeps_its_warnings_while_a_block_is_open(
        every_warning_is_an_error):
    """The defect's first half: a filter is process-wide, so the old block
    dropped every thread's warnings for as long as it was open.  A thread
    outside the block is told what it warns -- text or not."""
    with quiet_warnings():
        assert _in_a_thread(_raises_here)
        assert _in_a_thread(lambda: _raises_here(123))
        assert not _raises_here()
    assert _in_a_thread(_raises_here)


def test_a_thread_started_inside_a_block_is_not_inside_it(every_warning_is_an_error):
    """The silence belongs to the thread that opened the block, not to the
    threads it starts: a worker a node's code spawns answers for itself."""
    with quiet_warnings():
        assert _in_a_thread(_raises_here)

    def opens_its_own():
        with quiet_warnings():
            return _raises_here()

    assert not _in_a_thread(opens_its_own)
    assert _raises_here()


def test_blocks_nest(every_warning_is_an_error):
    with quiet_warnings():
        with quiet_warnings():
            assert not _raises_here()
        assert not _raises_here(), "the inner block's exit ended the outer one's silence"
    assert _raises_here()


def test_a_block_that_raises_closes_like_one_that_returns(every_warning_is_an_error):
    before = list(warnings.filters)
    with pytest.raises(ZeroDivisionError):
        with quiet_warnings():
            1 / 0
    assert _raises_here()
    assert list(warnings.filters) == before
    assert qw._open_blocks == 0 and qw._THREAD.depth == 0


def test_an_error_inside_a_block_is_still_an_error(every_warning_is_an_error):
    """Only warnings are silenced."""
    with pytest.raises(ValueError, match="not a warning"):
        with quiet_warnings():
            raise ValueError("not a warning")


def test_a_warning_ignored_in_a_block_is_delivered_the_next_time():
    """Python remembers a warning it has shown, per place, and shows it
    once.  One ignored inside a block was not shown, so it is not
    remembered: the same warning from the same line, outside the block, is
    delivered."""
    def warn_from_one_line():
        warnings.warn("said twice from one line", Loud)

    with warnings.catch_warnings(record=True) as shown:
        warnings.resetwarnings()
        warnings.simplefilter("default")
        with quiet_warnings():
            warn_from_one_line()
        assert shown == []
        warn_from_one_line()
        assert [str(w.message) for w in shown] == ["said twice from one line"]
        warn_from_one_line()
        assert len(shown) == 1, "the default action shows a warning once per place"


# ---------------------------------------------------------------------------
# What it leaves in the process's filters
# ---------------------------------------------------------------------------


def test_the_filters_are_the_ones_before_once_no_block_is_open(every_warning_is_an_error):
    """The same list object with the same entries: nothing is saved and
    put back, and the one filter a block adds is taken out again."""
    the_list = warnings.filters
    before = list(the_list)
    with quiet_warnings():
        assert warnings.filters is the_list
        assert warnings.filters[0] is qw._FILTER and _copies() == 1
        assert list(warnings.filters[1:]) == before
        with quiet_warnings():
            assert _copies() == 1
        assert _copies() == 1, "a nested block's exit removed the filter under an open one"
    assert warnings.filters is the_list
    assert list(warnings.filters) == before


@pytest.mark.parametrize("first_in_leaves", ["first", "last"])
def test_blocks_that_overlap_in_two_threads_leave_the_filters_alone(
        every_warning_is_an_error, first_in_leaves):
    """The defect's second half.  Two ``catch_warnings`` blocks that
    overlapped in two threads and left in the order they entered each put
    back the other's saved list, and the second one's was the first one's
    ``ignore``.  Both orders of leaving, each thread silenced for exactly
    as long as its own block."""
    before = list(warnings.filters)
    a_inside, b_inside, a_out, b_out = (threading.Event() for _ in range(4))
    seen: dict = {}

    def a():
        with quiet_warnings():
            a_inside.set()
            assert b_inside.wait(WAIT)
            if first_in_leaves == "last":
                assert b_out.wait(WAIT)
            seen["a inside"] = _raises_here()
        seen["a after"] = _raises_here()
        a_out.set()

    def b():
        assert a_inside.wait(WAIT)
        with quiet_warnings():
            b_inside.set()
            if first_in_leaves == "first":
                assert a_out.wait(WAIT)
            seen["b inside"] = _raises_here()
        seen["b after"] = _raises_here()
        b_out.set()

    threads = [threading.Thread(target=a, daemon=True), threading.Thread(target=b, daemon=True)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT)
    assert not any(t.is_alive() for t in threads)
    assert seen == {"a inside": False, "a after": True,
                    "b inside": False, "b after": True}
    assert list(warnings.filters) == before
    assert _raises_here()


def test_many_threads_opening_blocks_at_once_lose_no_warning_and_leave_no_filter(
        every_warning_is_an_error):
    """Eight threads, each in and out of a block a few hundred times,
    released together: inside its block a thread is never told, outside it
    always is, and the filters end as they began.  The interpreter is told
    to switch threads as often as it can, so the blocks really interleave
    (at the default interval the whole test fits in a handful of switches)."""
    before = list(warnings.filters)
    n_threads, rounds = 8, 300
    barrier = threading.Barrier(n_threads)
    wrong: list = []
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # units: s

    def work(i):
        barrier.wait(WAIT)
        for r in range(rounds):
            with quiet_warnings():
                if _raises_here(f"{i}/{r} inside"):
                    wrong.append((i, r, "raised inside its block"))
                if r % 3 == 0:
                    with quiet_warnings():
                        pass
                    if _raises_here(f"{i}/{r} after a nested block"):
                        wrong.append((i, r, "raised after a nested block closed"))
            if not _raises_here(f"{i}/{r} outside"):
                wrong.append((i, r, "not raised outside every block"))

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(n_threads)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(WAIT)
    finally:
        sys.setswitchinterval(interval)
    assert not any(t.is_alive() for t in threads)
    assert not wrong, wrong[:5]
    assert list(warnings.filters) == before
    assert qw._open_blocks == 0


def test_a_filter_put_ahead_of_an_open_block_is_overtaken_by_the_next_block(
        every_warning_is_an_error):
    """A filter is consulted front to back.  One put ahead of the block's
    while it is open wins from then on (that is what its author asked
    for); the next block to open goes to the front again, and the last
    one out removes every copy -- and nothing that was not its own."""
    with quiet_warnings():
        warnings.simplefilter("error", Loud)          # somebody's own filter
        mine = warnings.filters[0]
        assert _raises_here(), "a filter put in front did not take effect"
        with quiet_warnings():
            assert warnings.filters[0] is qw._FILTER
            assert not _raises_here()
            assert _copies() == 2
        assert _copies() == 2, "a copy was taken out from under an open block"
    assert _copies() == 0
    assert warnings.filters[0] is mine, "the block removed a filter that was not its own"


# ---------------------------------------------------------------------------
# The lock, and what runs under it
# ---------------------------------------------------------------------------


def test_the_lock_is_free_while_a_block_runs():
    """The lock guards a counter and a list operation.  It is not held
    while the block's body runs, so nothing the body calls -- a node's
    ``update``, a JAX trace, a route holding the REST server's graph lock
    -- can wait on it, whichever lock that code holds or wants."""
    def take():
        got = qw._FILTER_LOCK.acquire(timeout=5)
        if got:
            qw._FILTER_LOCK.release()
        return got

    with quiet_warnings():
        assert _in_a_thread(take), "the helper's lock is held across the block"
        with quiet_warnings():
            assert _in_a_thread(take)


def test_the_lock_guards_two_list_operations_and_nothing_else():
    """Read off the source: the lock is taken in the two functions that
    open and close a block, and all that is called while it is held is
    ``list.insert`` and ``list.remove`` on the filters.  No call out --
    to a node, to JAX, to anything that takes another lock -- so the lock
    cannot be one half of a deadlock.  Nothing else in ``src/`` names it."""
    tree = ast.parse((SRC_ROOT / HELPER).read_text())
    holders = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.With) and any(
                    isinstance(item.context_expr, ast.Name)
                    and item.context_expr.id == "_FILTER_LOCK" for item in node.items):
                calls = sorted(
                    ast.unparse(call.func) for stmt in node.body for call in ast.walk(stmt)
                    if isinstance(call, ast.Call))
                holders[fn.name] = calls
    assert holders == {"_block_opened": ["filters.insert"],
                       "_block_closed": ["filters.remove"]}, holders
    named = [node for node in ast.walk(tree)
             if isinstance(node, ast.Name) and node.id == "_FILTER_LOCK"]
    assert len(named) == 3, "the lock is named somewhere other than its definition and two holders"
    elsewhere = [path.relative_to(SRC_ROOT).as_posix() for path in sorted(SRC_ROOT.rglob("*.py"))
                 if path.relative_to(SRC_ROOT).as_posix() != HELPER
                 and "_FILTER_LOCK" in path.read_text()]
    assert not elsewhere, elsewhere


def test_a_block_opens_and_closes_while_the_same_thread_holds_the_lock(
        every_warning_is_an_error):
    """Re-entrant: code that runs inside the guarded region on the thread
    that holds the lock (a finaliser an allocation there triggers) may
    open a block of its own."""
    def nested():
        with qw._FILTER_LOCK:
            with quiet_warnings():
                return _raises_here()

    assert _in_a_thread(nested) is False


def test_asking_the_filter_runs_no_python_code(every_warning_is_an_error):
    """A thread looking up the filter for a warning of its own walks the
    list in C.  If the helper's filter ran Python code there, that thread
    could be suspended in the middle of its walk while a block closed and
    the list changed under it -- and skip the filter after ours.  With a
    block open in this thread, another thread's warning is delivered
    without a single Python-level call."""
    def warn_under_a_profiler():
        calls: list = []

        def profiler(frame, event, arg):
            if event == "call":
                calls.append(frame.f_code.co_name)

        sys.setprofile(profiler)
        try:
            try:
                warnings.warn("another thread's", Loud)
            except Loud:
                raised = True
            else:
                raised = False
        finally:
            sys.setprofile(None)
        return raised, calls

    with quiet_warnings():
        raised, calls = _in_a_thread(warn_under_a_profiler)
    assert raised
    assert calls == []


def test_the_per_thread_default_is_read_the_same_way_on_every_interpreter():
    """Two things about the ``threading.local`` the filter holds that only
    a newer interpreter than CI's would show.  Its class default is a
    ``staticmethod``: a bare ``functools.partial`` read through an
    instance warns on Python 3.13 -- from inside the warnings machinery,
    which then recurses until the stack ends -- and is bound like a method
    from 3.14.  And it has no ``__init__``, which would run, as Python
    code, the first time each thread reads it."""
    assert isinstance(vars(qw._ThisThread)["match"], staticmethod)
    assert "__init__" not in vars(qw._ThisThread)
    assert qw._THREAD.match("text") is False
    assert _in_a_thread(lambda: qw._THREAD.match("text")) is False
    with quiet_warnings():
        assert qw._THREAD.match("text") is True and qw._THREAD.match(123) is True
        assert _in_a_thread(lambda: qw._THREAD.match(123)) is False


# ---------------------------------------------------------------------------
# What it cannot do: somebody else's catch_warnings() in another thread
# ---------------------------------------------------------------------------


def _foreign_block(opened: threading.Event, close: threading.Event) -> threading.Thread:
    """A thread that holds a ``warnings.catch_warnings()`` block of its
    own open from when it starts until *close* is set."""
    def run():
        with warnings.catch_warnings():
            opened.set()
            assert close.wait(WAIT)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert opened.wait(WAIT)
    return thread


def test_a_foreign_block_that_closes_after_a_probe_leaves_a_copy_that_ignores_nothing(
        every_warning_is_an_error):
    """``catch_warnings`` in another thread, opened while a block is open
    and closed after it: it puts back the list it saved, which holds the
    block's filter.  With ``simplefilter("ignore")`` that copy silenced
    the process for good.  This one matches no thread outside a block, so
    every warning is still delivered, and the next block to close takes
    it out."""
    before = list(warnings.filters)
    opened, close = threading.Event(), threading.Event()
    with quiet_warnings():
        foreign = _foreign_block(opened, close)
        assert not _raises_here()
    close.set()
    foreign.join(WAIT)
    assert _copies() == 1, "the scenario did not leave a copy: the test is not testing it"
    assert _raises_here() and _in_a_thread(_raises_here)
    assert _raises_here(123)
    with quiet_warnings():
        assert not _raises_here()
    assert list(warnings.filters) == before


def test_a_foreign_block_that_closes_inside_a_probe_ends_its_silence_and_leaves_nothing(
        every_warning_is_an_error):
    """The limit, as documented: ``catch_warnings`` in another thread,
    opened before a block and closed during it, puts back a list from
    before the block -- without its filter.  The rest of that block is not
    silenced (here the warning is raised; shown, without ``error``), and
    that is all: nothing is left behind, and the next block is silenced."""
    before = list(warnings.filters)
    opened, close = threading.Event(), threading.Event()
    foreign = _foreign_block(opened, close)
    with quiet_warnings():
        assert not _raises_here()
        close.set()
        foreign.join(WAIT)
        assert _raises_here(), (
            "a foreign catch_warnings() that closed inside the block did not "
            "end its silence: the documented limit no longer holds as written")
    assert list(warnings.filters) == before
    with quiet_warnings():
        assert not _raises_here()
    assert list(warnings.filters) == before


# ---------------------------------------------------------------------------
# Nothing else in src/ touches the filters
# ---------------------------------------------------------------------------

SRC_ROOT = pathlib.Path(maddening.__file__).resolve().parent
HELPER = "core/_quiet_warnings.py"

#: Calls that change the process's warning filters, or save and restore them.
_FILTER_CALLS = ("catch_warnings", "simplefilter", "filterwarnings", "resetwarnings")
#: Attributes of the ``warnings`` module that are the filter state itself.
_FILTER_STATE = ("filters", "showwarning", "_filters_mutated", "_showwarnmsg_impl",
                 "defaultaction", "onceregistry")
_CALL_IN_TEXT = re.compile(r"\b(" + "|".join(_FILTER_CALLS) + r")\s*\(")

#: Text that *tells a user* how to silence one category for good -- one
#: ``simplefilter`` at start-up, which saves and restores nothing.  Keyed
#: by the line, so a new mention has to be put here on purpose.
_ADVICE_TO_THE_USER = {
    ("warnings.py", 'warnings.simplefilter("ignore", PrecisionLimitWarning)'),
    ("sysid.py", 'f"warnings.simplefilter(\'ignore\', PrecisionLimitWarning), "'),
}


def _filter_uses(source: str, rel: str) -> list[tuple[str, int, str]]:
    """``(file, line, what)`` for every way *source* touches the warning
    filters: in its code (a call, an import, the filter state of the
    ``warnings`` module under any alias) and in its string constants (a
    script an example ships to another machine is a string here and code
    there), less :data:`_ADVICE_TO_THE_USER`."""
    tree = ast.parse(source, filename=rel)
    lines = source.splitlines()
    aliases = {"warnings"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {a.asname or a.name for a in node.names if a.name == "warnings"}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "warnings":
            for a in node.names:
                if a.name in _FILTER_CALLS + _FILTER_STATE or a.name == "*":
                    found.append((rel, node.lineno, f"from warnings import {a.name}"))
        elif isinstance(node, ast.Attribute):
            receiver = node.value.id if isinstance(node.value, ast.Name) else None
            if node.attr in _FILTER_CALLS:
                found.append((rel, node.lineno, f".{node.attr}"))
            elif node.attr in _FILTER_STATE and receiver in aliases:
                found.append((rel, node.lineno, f"{receiver}.{node.attr}"))
        elif isinstance(node, ast.Name) and node.id in _FILTER_CALLS:
            found.append((rel, node.lineno, node.id))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in _FILTER_CALLS:
                # getattr(warnings, "simplefilter"): the name alone, as data.
                found.append((rel, node.lineno, f"the name {node.value!r} as a string"))
                continue
            if not _CALL_IN_TEXT.search(node.value):
                continue
            first, last = node.lineno, node.end_lineno or node.lineno
            for number in range(first, last + 1):
                line = lines[number - 1].strip()
                if _CALL_IN_TEXT.search(line) and (rel, line) not in _ADVICE_TO_THE_USER:
                    found.append((rel, number, f"in a string: {line}"))
    return found


def test_only_the_helper_touches_the_warnings_filters():
    """Every silencing block in ``src/`` goes through ``quiet_warnings()``.
    A ``warnings.catch_warnings()`` written beside it again -- in the
    library, in an example, in a script an example ships -- is the defect
    back, in whichever threads reach it."""
    found = []
    scanned = 0
    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel = path.relative_to(SRC_ROOT).as_posix()
        if rel == HELPER:
            continue
        text = path.read_text()
        # Every use the scan knows names the module or one of the calls;
        # a file that mentions neither is not parsed (most of the tree).
        if "warnings" not in text and not any(call in text for call in _FILTER_CALLS):
            continue
        scanned += 1
        found += _filter_uses(text, rel)
    assert scanned >= 5, f"only {scanned} files of {SRC_ROOT} were scanned"
    assert not found, "\n".join(f"{rel}:{line}: {what}" for rel, line, what in found)


def test_the_advice_the_scan_lets_through_is_still_in_the_source():
    """An entry of the allowance that matches nothing is an allowance for
    a line nobody has read."""
    for rel, line in sorted(_ADVICE_TO_THE_USER):
        text = (SRC_ROOT / rel).read_text()
        assert any(candidate.strip() == line for candidate in text.splitlines()), (rel, line)


def test_the_helper_is_where_the_scan_says_it_is():
    """The one module the scan skips exists, and does touch the filters:
    a scan that skipped a file that is not there would pass on a tree
    whose helper had moved, with every use unchecked at its new home."""
    source = (SRC_ROOT / HELPER).read_text()
    assert _filter_uses(source, HELPER)


def _helper_uses(source: str, rel: str) -> tuple[int, list[tuple[str, int, str]]]:
    """``(calls, odd)``: how many times *source* calls ``quiet_warnings``,
    and every use that is not ``with quiet_warnings():`` under that name
    -- an import under another name, a call that is not a ``with`` item,
    the function handed on as a value."""
    tree = ast.parse(source, filename=rel)
    with_items = {id(item.context_expr) for node in ast.walk(tree)
                  if isinstance(node, (ast.With, ast.AsyncWith)) for item in node.items}
    callees = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    calls, odd = 0, []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if "quiet_warnings" in a.name and a.asname not in (None, a.name):
                    odd.append((rel, node.lineno, f"{a.name} imported as {a.asname}"))
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name == "quiet_warnings":
                calls += 1
                if id(node) not in with_items:
                    odd.append((rel, node.lineno, "called outside a with statement"))
        elif (isinstance(node, (ast.Name, ast.Attribute)) and id(node) not in callees
              and (node.id if isinstance(node, ast.Name) else node.attr) == "quiet_warnings"):
            odd.append((rel, node.lineno, "used as a value, not called"))
    return calls, odd


def test_every_use_of_the_helper_is_a_with_block_under_its_own_name():
    """``tests/_quiet_block_witness.py`` finds the silencing blocks of
    ``src/`` as ``with quiet_warnings():`` statements, and
    ``test_warning_probes_across_threads.py`` drives a node through every
    one it finds.  A use it could not find -- the helper imported under
    another name, entered by hand, or handed on as a value -- would be a
    block no test looks inside."""
    odd, calls = [], 0
    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel = path.relative_to(SRC_ROOT).as_posix()
        text = path.read_text()
        if rel == HELPER or "quiet_warnings" not in text:
            continue
        n, found = _helper_uses(text, rel)
        calls += n
        odd += found
    assert not odd, odd
    assert calls >= 16, f"{calls} uses of quiet_warnings() found under {SRC_ROOT}"


_IMPORT = "from maddening.core._quiet_warnings import quiet_warnings\n"


@pytest.mark.parametrize("source", [
    "from maddening.core._quiet_warnings import quiet_warnings as _quiet\nwith _quiet():\n    pass\n",
    "import maddening.core._quiet_warnings as quiet_warnings_module\n",
    _IMPORT + "import contextlib\nwith contextlib.ExitStack() as stack:\n"
              "    stack.enter_context(quiet_warnings())\n",
    _IMPORT + "block = quiet_warnings()\nblock.__enter__()\n",
    _IMPORT + "silence = quiet_warnings\nwith silence():\n    pass\n",
    _IMPORT + "def probe(cm=quiet_warnings):\n    with cm():\n        pass\n",
    "from maddening.core import _quiet_warnings\nhow = _quiet_warnings.quiet_warnings\n",
])
def test_the_scan_sees_each_use_of_the_helper_a_witness_would_not_find(source):
    assert _helper_uses(source, "scanned.py")[1], source


@pytest.mark.parametrize("source, calls", [
    (_IMPORT + "with quiet_warnings():\n    pass\n", 1),
    (_IMPORT + "with quiet_warnings(), open('f') as f:\n    pass\n", 1),
    (_IMPORT + "def a():\n    with quiet_warnings():\n        pass\n"
               "def b():\n    with quiet_warnings():\n        pass\n", 2),
    ("from maddening.core import _quiet_warnings\nwith _quiet_warnings.quiet_warnings():\n    pass\n", 1),
    ("x = 1\n", 0),
])
def test_the_scan_counts_a_with_block_and_finds_nothing_odd_in_it(source, calls):
    assert _helper_uses(source, "scanned.py") == (calls, []), source


def test_the_scans_allowance_is_for_one_line_of_one_file():
    """The advice the scan lets through is let through where it stands,
    and nowhere else: not another line of the same file, not the same
    line in another file."""
    rel, line = sorted(_ADVICE_TO_THE_USER)[1]
    assert rel == "warnings.py"
    advice = f'"""Advice.\n\n    {line}\n"""\n'
    assert _filter_uses(advice, rel) == []
    assert _filter_uses(advice, "other.py")
    assert _filter_uses('"""Run under ``warnings.simplefilter("always")``."""\n', rel)


@pytest.mark.parametrize("source", [
    "import warnings\nwith warnings.catch_warnings():\n    pass\n",
    "import warnings\nwarnings.simplefilter('ignore')\n",
    "import warnings\nwarnings.filterwarnings('ignore', message='x')\n",
    "import warnings\nwarnings.resetwarnings()\n",
    "import warnings as _w\nwith _w.catch_warnings():\n    _w.simplefilter('ignore')\n",
    "from warnings import catch_warnings\n",
    "from warnings import simplefilter as quiet\nquiet('ignore')\n",
    "from warnings import *\n",
    "import warnings\nwarnings.filters.insert(0, ('ignore', None, Warning, None, 0))\n",
    "import warnings as w\nw.filters[:] = []\n",
    "import warnings\nwarnings.showwarning = print\n",
    "import numpy as np\nwith np.testing.suppress_warnings() as sup:\n    sup.simplefilter('ignore')\n",
    'SCRIPT = r"""\nimport warnings\nwith warnings.catch_warnings():\n    gm.compile()\n"""\n',
    'def f():\n    """Run under ``warnings.simplefilter("ignore")``."""\n',
    "msg = f\"call warnings.filterwarnings('ignore') {1}\"\n",
    "from somewhere import *\nwith catch_warnings():\n    simplefilter('ignore')\n",
    "def run(catch_warnings):\n    with catch_warnings():\n        pass\n",
    "import warnings\ngetattr(warnings, 'simplefilter')('ignore')\n",
    "import warnings\nblock = getattr(warnings, 'catch_warnings')\n",
])
def test_the_scan_sees_each_way_of_touching_the_filters(source):
    assert _filter_uses(source, "scanned.py"), source


@pytest.mark.parametrize("source", [
    "import warnings\nwarnings.warn('x', UserWarning, stacklevel=2)\n",
    "import warnings\nwarnings.warn_explicit('x', UserWarning, 'f.py', 1)\n",
    "from maddening.core._quiet_warnings import quiet_warnings\nwith quiet_warnings():\n    pass\n",
    "class Bank:\n    filters = []\nBank.filters.append(1)\nself = Bank()\nself.filters = []\n",
    "# with warnings.catch_warnings(): a comment is not code\n",
    'text = "the warnings filters are process-wide"\n',
])
def test_the_scan_lets_through_what_does_not_touch_the_filters(source):
    assert _filter_uses(source, "scanned.py") == [], source
