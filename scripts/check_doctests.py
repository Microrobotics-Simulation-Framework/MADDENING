#!/usr/bin/env python
"""Run every ``>>>`` example in ``src/maddening/`` and prove they ran.

Docstring examples are documentation that claims to be executable.  Until
this gate existed nothing executed them: ``--doctest-modules`` appears
neither in ``pyproject.toml`` nor in any CI workflow, so an example could
drift out of date for a whole release and still read as authoritative.
That is the same defect class as an unchecked citation or an unmapped
implementation claim, which is why this lives beside the other
``scripts/check_*.py`` gates and runs in the same CI job.

Three things are checked, because the count alone is not enough:

1. **pytest must pass** every doctest it collects.
2. **The set of examples that ran must not silently shrink.**  A
   doctest job that runs nothing exits 0 and reports success, which is
   worse than no gate: it is cited as coverage.  So the script
   independently scans the package source with :mod:`ast` +
   :mod:`doctest`, finds every file that *contains* an example, and fails
   unless a doctest in each of them actually passed.  On top of that a
   committed floor (``MIN_EXAMPLES``) guards the absolute number of
   *examples executed*.  Both are counted from passing test reports, so
   deselection, a skip and an import error are all failures of this
   gate -- a doctest that does not run is prose again, which is the thing
   the gate exists to prevent.
3. **No example is skipped.**  A pytest doctest item is a whole
   *docstring*; ``# doctest: +SKIP`` on one ``>>>`` inside it leaves the
   item passing.  The floor used to count items, so a wrong example
   marked ``+SKIP`` inside a two-example docstring still reported
   "OK: 31 ... floor 31" (audit_040_phase3_wave_d, D2).  The gate now
   counts the examples the doctest runner actually executed (its
   ``tries``), reports every example it skipped, and fails on any: an
   example that cannot run belongs in a plain code block, not behind
   ``>>>``.

The static scan mirrors :class:`doctest.DocTestFinder`'s own traversal --
module docstring, top-level functions and classes, and class members,
without descending into function bodies -- so it requires exactly the
files pytest would collect from and no others.

``src/maddening/examples/`` is excluded, and must stay excluded.  Those
files are runnable scripts, not library modules: several call
``sys.exit()`` at import time when an optional dependency is missing
(which crashes pytest's collector outright, not merely the file), and six
are named ``*_test.py``, so pytest collects their functions as live tests
that would launch cloud VMs.  Excluding them leaves no hole:
``tests/test_examples_smoke.py`` owns that directory, statically for the
cloud scripts -- which it says the suite must never execute, for the same
reason -- and by running the cheap headless ones in its ``slow`` lane.

Usage:
    python scripts/check_doctests.py [--min N] [-- PYTEST_ARG ...]

Exit codes:
    0 -- every example ran, passed, and the set that ran is the expected size
    1 -- a doctest failed, a file with examples produced no passing doctest,
         an example was skipped, nothing ran, or fewer examples than the
         floor executed
    2 -- the run could not be trusted (pytest could not be run at all, or
         the doctest runner no longer exposes the count this gate reads)
"""

from __future__ import annotations

import argparse
import ast
import doctest
import os
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = Path("src/maddening")
#: Runnable scripts, not library modules.  See the module docstring.
EXCLUDED = (PACKAGE / "examples",)

#: The floor, on ``>>>`` examples *executed* -- not on docstrings.  Raise it
#: when examples are added; never lower it to make a run pass -- a shrinking
#: set of examples is the failure this number exists to catch.
#:
#: History, while it counted docstrings (pytest items): 15 on
#: release/0.4.0 (8 files); 22 after the phase-3 TypedDict pass (15 files);
#: 31 after the IFT gradient bound added three to coupling/acceleration.py.
#: Those 31 docstrings hold MIN_EXAMPLES examples, measured on
#: audit_040_phase3_wave_d's fix branch as the runner's own ``tries`` total
#: (CPython 3.12.3, pytest 9.0.3, jaxlib 0.11.0; the count is a property of
#: the source, not of the environment, and the static scan in
#: ``files_with_examples`` agrees with it).
#:
#: 98 -> 110 when audit_040_phase3_confirm found the floor 12 below the 110
#: examples that run, so deleting the nine in
#: ``acceleration.arnoldi_spectral_radius`` passed -- against a CHANGELOG
#: saying the gate fails if the collection shrinks.  A floor below the count
#: guards only the part of the collection it covers: raise it in the commit
#: that adds examples (the OK line prints the count beside the floor),
#: together with ``COMMITTED_EXAMPLE_FLOOR`` in
#: ``tests/compliance/test_gate_scripts.py``.
MIN_EXAMPLES = 110


def _docstrings(path: Path):
    """Yield the docstrings ``doctest.DocTestFinder`` would look at.

    Module, top-level functions and classes, and class members
    (recursively), which is exactly the finder's traversal: it does not
    descend into function bodies, so a ``>>>`` in a nested function is
    never collected and must not be required here either.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return

    def walk(node):
        text = ast.get_docstring(node, clean=False)
        if text:
            yield text
        for child in node.body:
            if isinstance(child, ast.ClassDef):
                yield from walk(child)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # The finder reads the function's own docstring but does
                # not recurse into its body.
                own = ast.get_docstring(child, clean=False)
                if own:
                    yield own

    yield from walk(tree)


def files_with_examples() -> dict[str, int]:
    """Map ``src/...`` path -> number of docstrings holding examples."""
    return {path: docstrings
            for path, (docstrings, _examples) in _static_counts().items()}


def _static_counts() -> dict[str, tuple[int, int]]:
    """Map ``src/...`` path -> ``(docstrings with examples, examples)``."""
    parser = doctest.DocTestParser()
    found: dict[str, tuple[int, int]] = {}
    for path in sorted((REPO_ROOT / PACKAGE).rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        if any(str(rel).startswith(str(ex)) for ex in EXCLUDED):
            continue
        per_docstring = [len(parser.get_examples(d)) for d in _docstrings(path)]
        per_docstring = [n for n in per_docstring if n]
        if per_docstring:
            found[str(rel)] = (len(per_docstring), sum(per_docstring))
    return found


class _Recorder:
    """Counts the doctests that actually *passed*, and the examples in them
    that actually *ran*, per file.

    Reading this from the pytest run itself -- rather than from a JUnit
    report or by parsing ``--collect-only`` output -- means the gate
    measures the same objects pytest measures, with no report format to
    drift between releases.

    It counts passes, not collected items, and the difference is the
    whole point.  An earlier version of this recorder hooked
    ``pytest_collection_modifyitems``, which runs *before* ``-k`` / ``-m``
    deselection: mutation-testing it with ``-k no_such_doctest`` showed
    it happily reporting 15 doctests when none had run.  A deselection
    that removed only some of them would have left the gate green over a
    suite that executed nothing.  ``pytest_runtest_logreport`` cannot say
    that: a doctest that was deselected, skipped, errored in setup or
    failed never produces a passing call report, so it is simply not
    counted and the file it lives in goes missing.

    A passing item is a whole docstring, though, and an example inside it
    marked ``+SKIP`` does not stop it passing.  So the examples are counted
    too, from the doctest runner's own ``tries`` -- the number of examples
    it executed -- read before and after each item's call phase.  pytest
    shares one runner per module, hence the difference rather than the
    total.  ``tries`` is a CPython ``DocTestRunner`` attribute; if a
    future release drops it the recorder says so (``instrument_error``)
    and the gate exits 2, rather than silently counting zero.
    """

    def __init__(self) -> None:
        self.passed: dict[str, int] = {}
        self.examples_run: dict[str, int] = {}
        #: ``(nodeid, lines of skipped examples, number skipped)``
        self.skipped_examples: list[tuple[str, list[int], int]] = []
        self.collected = 0
        self.instrument_error: str | None = None
        self._before: dict[str, int] = {}
        self._ran: dict[str, tuple[int, int, list[int]]] = {}

    def pytest_collection_finish(self, session):  # noqa: D102 (pytest hook)
        # After every modifyitems hook, so session.items is final.
        self.collected = len(session.items)

    @staticmethod
    def _doctest_parts(item):
        dtest = getattr(item, "dtest", None)
        runner = getattr(item, "runner", None)
        return (dtest, runner) if dtest is not None and runner is not None else None

    def pytest_runtest_setup(self, item):  # noqa: D102 (pytest hook)
        parts = self._doctest_parts(item)
        if parts is None:
            return
        _dtest, runner = parts
        if not hasattr(runner, "tries"):
            self.instrument_error = (
                f"{type(runner).__name__} has no 'tries' attribute, so the "
                f"number of examples executed cannot be read"
            )
            return
        self._before[item.nodeid] = runner.tries

    def pytest_runtest_makereport(self, item, call):  # noqa: D102 (pytest hook)
        if call.when != "call" or item.nodeid not in self._before:
            return None
        dtest, runner = self._doctest_parts(item)
        ran = runner.tries - self._before.pop(item.nodeid)
        skipped_lines = [
            (dtest.lineno or 0) + ex.lineno + 1
            for ex in dtest.examples if ex.options.get(doctest.SKIP)
        ]
        self._ran[item.nodeid] = (ran, len(dtest.examples), skipped_lines)
        return None                          # let pytest build the report

    def pytest_runtest_logreport(self, report):  # noqa: D102 (pytest hook)
        if report.when != "call" or not report.passed:
            return
        path = report.nodeid.split("::", 1)[0]
        self.passed[path] = self.passed.get(path, 0) + 1
        ran, total, skipped_lines = self._ran.pop(report.nodeid, (0, 0, []))
        self.examples_run[path] = self.examples_run.get(path, 0) + ran
        if ran < total:
            self.skipped_examples.append((report.nodeid, skipped_lines,
                                          total - ran))


def pytest_args(extra: list[str]) -> list[str]:
    return [
        "--doctest-modules",
        str(PACKAGE),
        *(f"--ignore={ex}" for ex in EXCLUDED),
        "-p",
        "no:cacheprovider",
        *extra,
    ]


def run_pytest(extra: list[str]) -> tuple[int, "_Recorder"]:
    # Record, rather than assume, the configuration the examples run in:
    # CPU JAX and no plugin autoload, matching every other pytest
    # invocation in this repository.  Set only if the caller left them
    # unset, so CI can still override.  Both are read at import time by
    # the code they configure, so they go in before pytest starts.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    os.chdir(REPO_ROOT)

    import pytest  # imported here so --help works without pytest installed

    args = pytest_args(extra)
    print("equivalent standalone command:")
    print(
        f"  JAX_PLATFORMS={os.environ['JAX_PLATFORMS']} "
        f"PYTEST_DISABLE_PLUGIN_AUTOLOAD={os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']} "
        f"python -m pytest {shlex.join(args)}"
    )
    print(f"  (pytest {pytest.__version__}, cwd {REPO_ROOT})\n", flush=True)

    recorder = _Recorder()
    rc = int(pytest.main(args, plugins=[recorder]))
    return rc, recorder


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min",
        type=int,
        default=MIN_EXAMPLES,
        help=f"minimum number of >>> examples that must execute and pass "
             f"(default {MIN_EXAMPLES}); a run in which nothing executed "
             f"fails whatever this is set to",
    )
    args, extra = ap.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    static = _static_counts()
    expected = {path: docstrings for path, (docstrings, _n) in static.items()}
    print(
        f"python {sys.version.split()[0]}; "
        f"{len(expected)} file(s) under {PACKAGE} contain "
        f"{sum(n for _d, n in static.values())} docstring example(s) in "
        f"{sum(expected.values())} docstring(s)"
    )

    try:
        rc, recorder = run_pytest(extra)
    except ImportError as exc:
        print(
            f"ERROR: could not run pytest ({exc}); the gate reports nothing "
            f"rather than success.",
            file=sys.stderr,
        )
        return 2

    if recorder.instrument_error:
        print(f"ERROR: {recorder.instrument_error}; the gate reports nothing "
              f"rather than success.", file=sys.stderr)
        return 2

    per_file = recorder.passed
    total = sum(per_file.values())
    examples = sum(recorder.examples_run.values())
    n_skipped = sum(n for _nodeid, _lines, n in recorder.skipped_examples)
    print(
        f"\n{recorder.collected} doctest(s) selected, {total} passed, "
        f"from {len(per_file)} file(s); {examples} example(s) executed in "
        f"them, {n_skipped} skipped:"
    )
    for name in sorted(per_file):
        print(f"  {per_file[name]:3d} docstring(s) "
              f"{recorder.examples_run.get(name, 0):4d} example(s)  {name}")

    failures = []
    if recorder.skipped_examples:
        failures.append(
            f"{n_skipped} example(s) inside passing doctests did not run "
            f"(# doctest: +SKIP, or a skip flag in effect).  A skipped "
            f"example is prose that looks like a test; rewrite it so it "
            f"runs, or show it as a plain code block instead of >>>:\n"
            + "\n".join(
                f"    {nodeid}: {n} skipped"
                + (f" (line(s) {', '.join(map(str, lines))})" if lines else "")
                for nodeid, lines, n in recorder.skipped_examples
            )
        )
    if examples == 0:
        failures.append(
            "no docstring example executed.  A gate that runs nothing cannot "
            "fail; this holds whatever --min says."
        )
    missing = sorted(set(expected) - set(per_file))
    if missing:
        failures.append(
            "these files contain docstring examples, and no doctest in them "
            "passed -- they were not collected, were deselected, were "
            "skipped, or they failed:\n"
            + "\n".join(f"    {m}  ({expected[m]} example docstring(s))" for m in missing)
        )
    if examples < args.min:
        failures.append(
            f"{examples} example(s) executed in passing doctests, floor is "
            f"{args.min}.  A gate that runs fewer examples than it used to "
            f"has stopped checking something; find out what, and only then "
            f"raise or lower MIN_EXAMPLES in {Path(__file__).name}."
        )
    if rc != 0:
        failures.append(f"pytest exited {rc}; see the output above.")

    if failures:
        print("\nFAIL: docstring examples")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"\nOK: {examples} docstring example(s) in {total} docstring(s) "
          f"ran and passed, 0 skipped, floor {args.min}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
