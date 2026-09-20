#!/usr/bin/env python
"""Run every ``>>>`` example in ``src/maddening/`` and prove they ran.

Docstring examples are documentation that claims to be executable.  Until
this gate existed nothing executed them: ``--doctest-modules`` appears
neither in ``pyproject.toml`` nor in any CI workflow, so an example could
drift out of date for a whole release and still read as authoritative.
That is the same defect class as an unchecked citation or an unmapped
implementation claim, which is why this lives beside the other
``scripts/check_*.py`` gates and runs in the same CI job.

Two things are checked, because the count alone is not enough:

1. **pytest must pass** every doctest it collects.
2. **The collection must not silently shrink.**  A doctest job that
   collects nothing exits 0 and reports success, which is worse than no
   gate: it is cited as coverage.  So the script independently scans the
   package source with :mod:`ast` + :mod:`doctest`, finds every file that
   *contains* an example, and fails if pytest did not collect a test from
   each of them.  On top of that a committed floor (``MIN_DOCTESTS``)
   guards the absolute count.

The static scan mirrors :class:`doctest.DocTestFinder`'s own traversal --
module docstring, top-level functions and classes, and class members,
without descending into function bodies -- so it requires exactly the
files pytest would collect from and no others.

``src/maddening/examples/`` is excluded, and must stay excluded.  Those
files are runnable scripts, not library modules: several call
``sys.exit()`` at import time when an optional dependency is missing
(which crashes pytest's collector outright, not merely the file), and six
are named ``*_test.py``, so pytest collects their functions as live tests
that would launch cloud VMs.

Usage:
    python scripts/check_doctests.py [--min N] [-- PYTEST_ARG ...]

Exit codes:
    0 -- every example ran and the collection is the expected size
    1 -- a doctest failed, a file with examples was not collected, or
         fewer than the floor were collected
    2 -- the run could not be trusted (pytest could not be run at all)
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

#: The floor.  Raise it when examples are added; never lower it to make a
#: run pass -- a shrinking collection is the failure this number exists to
#: catch.  Measured at 15 on release/0.4.0 (8 files).
MIN_DOCTESTS = 15


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
    parser = doctest.DocTestParser()
    found: dict[str, int] = {}
    for path in sorted((REPO_ROOT / PACKAGE).rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        if any(str(rel).startswith(str(ex)) for ex in EXCLUDED):
            continue
        count = sum(1 for d in _docstrings(path) if parser.get_examples(d))
        if count:
            found[str(rel)] = count
    return found


class _Recorder:
    """Records the file of every item pytest ends up with.

    ``pytest_collection_modifyitems`` is called once, after collection and
    after deselection, so what it sees is exactly what will run.  Reading
    it from the pytest run itself -- rather than from a JUnit report or
    from parsing ``--collect-only`` output -- means the gate measures the
    same objects pytest measures, with no format to drift between
    releases.
    """

    def __init__(self) -> None:
        self.per_file: dict[str, int] = {}

    def pytest_collection_modifyitems(self, items):  # noqa: D102 (pytest hook)
        for item in items:
            try:
                rel = Path(item.path).resolve().relative_to(REPO_ROOT)
            except (AttributeError, ValueError):
                continue
            key = str(rel)
            self.per_file[key] = self.per_file.get(key, 0) + 1


def pytest_args(extra: list[str]) -> list[str]:
    return [
        "--doctest-modules",
        str(PACKAGE),
        *(f"--ignore={ex}" for ex in EXCLUDED),
        "-p",
        "no:cacheprovider",
        *extra,
    ]


def run_pytest(extra: list[str]) -> tuple[int, dict[str, int]]:
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
    return rc, recorder.per_file


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min",
        type=int,
        default=MIN_DOCTESTS,
        help=f"minimum number of doctests that must be collected (default {MIN_DOCTESTS})",
    )
    args, extra = ap.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    expected = files_with_examples()
    print(
        f"python {sys.version.split()[0]}; "
        f"{len(expected)} file(s) under {PACKAGE} contain docstring examples"
    )

    try:
        rc, per_file = run_pytest(extra)
    except ImportError as exc:
        print(
            f"ERROR: could not run pytest ({exc}); the gate reports nothing "
            f"rather than success.",
            file=sys.stderr,
        )
        return 2

    total = sum(per_file.values())
    print(f"\ncollected {total} doctest(s) from {len(per_file)} file(s):")
    for name in sorted(per_file):
        print(f"  {per_file[name]:3d}  {name}")

    failures = []
    missing = sorted(set(expected) - set(per_file))
    if missing:
        failures.append(
            "these files contain docstring examples that pytest did not "
            "collect (an ignore rule, an import error, or a config change "
            "that turned doctest collection off for them):\n"
            + "\n".join(f"    {m}  ({expected[m]} example docstring(s))" for m in missing)
        )
    if total < args.min:
        failures.append(
            f"collected {total} doctests, floor is {args.min}.  A gate that "
            f"collects less than it used to has stopped checking something; "
            f"find out what, and only then raise or lower MIN_DOCTESTS in "
            f"{Path(__file__).name}."
        )
    if rc != 0:
        failures.append(f"pytest exited {rc}; see the output above.")

    if failures:
        print("\nFAIL: docstring examples")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"\nOK: {total} docstring examples ran, floor {args.min}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
