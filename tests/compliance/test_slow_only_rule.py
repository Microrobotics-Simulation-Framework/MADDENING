"""Every slow mark on a framework test says where its property is checked per push.

A slow mark takes a test off every push.  On a release branch it then runs
only when someone dispatches ``slow-tests.yml``, so a regression in what it
checks can sit unseen until then.  ``docs/developer_guide/testing_standards.md``
("What only the slow lane checks") therefore asks every slow-marked test of
a framework property to say, at the mark, one of two things:

* **where the property is still checked on every push**: a comment line
  ``# Per push: <node id>`` among the comments and decorators directly above
  the test's ``def`` (or above its class, or above the module's
  ``pytestmark``, when the slow mark is written there), naming tests the
  default lane runs;
* **or that it is slow-only on purpose**: the test, or its class, is named
  in the table under that heading, with the reason.

Nothing checked that rule, and audits kept finding framework properties
that were slow-only with neither.  This module checks it over
:data:`FRAMEWORK_PATHS`.  Which tests are slow and which run in the default
lane comes from pytest itself (one ``--collect-only`` subprocess), so a slow
mark applied through a parameter list, a class or ``pytestmark`` counts, and
a named witness that is itself slow-marked, misspelt or missing its class
fails.  The comments are read from the source.

Under ``tests/verification/hypothesis/`` the table is no way out: the
``verify-hypothesis`` job runs those slow properties on every push, but on
jax 0.10.2 only, so each needs a per-push sibling the default lane runs on
both JAX lanes.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STANDARDS = REPO_ROOT / "docs" / "developer_guide" / "testing_standards.md"
TABLE_HEADING = "### What only the slow lane checks"

#: Where a slow-marked test is presumed to check a framework property: the
#: coupling solvers and their reports, gradients, the params pytree, sysid,
#: sharding, checkpoints and serialisation, the REST and FMU surfaces, the
#: verification harness and the CI gates.  A new file here is covered
#: without anyone remembering to add it.
FRAMEWORK_PATHS = (
    "tests/core/",
    "tests/property/",
    "tests/verification/",
    "tests/cloud/multigpu/",
    "tests/compliance/",
    "tests/api/",
    "tests/fmi/",
)

#: Directories whose slow tests must name a per-push sibling: the table is
#: not enough (see the module docstring).
SIBLING_REQUIRED = ("tests/verification/hypothesis/",)

#: Files under :data:`FRAMEWORK_PATHS` whose slow tests do not check a
#: framework property, with the reason.  Each must exist and hold a slow
#: test, so an entry cannot outlive its file.
NOT_FRAMEWORK = {
    "tests/cloud/multigpu/test_run_pod_dry_run.py":
        "runs scripts/run_pod.py, the multi-GPU session's checklist, in dry-run mode on CPU: "
        "the session harness itself, a subprocess per goal",
    "tests/cloud/multigpu/test_run_pod_seeded_faults.py":
        "seeds faults into a scratch copy of the library and runs scripts/run_pod.py's goals "
        "against it: the session harness's own mutation test",
    "tests/api/test_binary_encoder.py":
        "a latency budget for one encoder on slow dynamics: a timing claim for a quiet "
        "machine, not a correctness property",
    "tests/fmi/test_binary_frames.py":
        "a throughput comparison of a million-element binary get against JSON: a timing "
        "claim for a quiet machine, not a correctness property",
    "tests/verification/test_gci_order.py":
        "grid-convergence studies of one node (the LBM pipe): a node's convergence, not the "
        "harness, which its per-push tests check on cheap nodes",
    "tests/verification/test_lbm_pressure_poiseuille.py":
        "pressure-driven Poiseuille flow in the LBM node against its analytic profile: a "
        "node's accuracy",
    "tests/verification/test_mms_order.py":
        "the LBM node's manufactured-solution convergence order: a node's accuracy",
    "tests/verification/test_wavelet_mms_order.py":
        "the wavelet node's manufactured-solution convergence order: a node's accuracy",
    "tests/verification/test_builtin_nodes_verified_lbm.py":
        "the verify_node battery on the costly LBM nodes: the battery runs on every push on "
        "the cheap built-in nodes and the surrogate",
}

#: ``tests/<path>.py::<name>[::<name>][<params>]``: a pytest node id below a file.
_NODE_ID = re.compile(r"tests/[\w/.\-]+?\.py(?:::[A-Za-z_]\w*)+(?:\[[^\]\s]*\])?")
_TEST_NAME = re.compile(r"\btest_\w+|\bTest[A-Z0-9_]\w*")
_PER_PUSH = re.compile(r"^\s*#\s*Per push:")


# --------------------------------------------------------------------------
# Reading the source: the comments at a slow mark
# --------------------------------------------------------------------------
def _block_start(lines: list[str], first: int) -> int:
    """The first line (0-based) of the comments directly above line ``first``."""
    while first > 0 and lines[first - 1].lstrip().startswith("#"):
        first -= 1
    return first


def _region(lines: list[str], node: ast.stmt) -> tuple[int, int]:
    """Lines ``[start, end)`` (0-based) of a definition's mark block: the
    comments directly above it, its decorators and any comments between
    them, up to the ``def`` / ``class`` line or the statement itself."""
    decorators = getattr(node, "decorator_list", [])
    first = min([d.lineno for d in decorators] + [node.lineno]) - 1
    return _block_start(lines, first), node.lineno - 1


def per_push_comments(lines: list[str], start: int, end: int) -> list[tuple[int, str]]:
    """Each ``# Per push:`` comment in ``lines[start:end]``, with its line
    number (1-based) and its text: the line and the comment lines directly
    after it, up to the first line that is not a comment."""
    found = []
    for i in range(start, end):
        if not _PER_PUSH.match(lines[i]):
            continue
        text = [lines[i].split("Per push:", 1)[1]]
        j = i + 1
        while j < end and lines[j].lstrip().startswith("#") and not _PER_PUSH.match(lines[j]):
            text.append(lines[j].lstrip().lstrip("#"))
            j += 1
        found.append((i + 1, " ".join(t.strip() for t in text)))
    return found


def witness_ids(text: str) -> tuple[list[str], list[str]]:
    """``(node ids, bare names)`` in a ``# Per push:`` comment's text.

    A bare name is a test or class name written outside a full node id
    (``test_x``, ``TestA::test_x``, or a file with no test named): it
    cannot be resolved without guessing, and a name missing its class is
    not a node id pytest knows.
    """
    ids = [m.group(0) for m in _NODE_ID.finditer(text)]
    rest = _NODE_ID.sub(" ", text)
    bare = [m.group(0) for m in _TEST_NAME.finditer(rest)]
    return ids, bare


@dataclass(frozen=True)
class MarkBlock:
    """Where a test's ``# Per push:`` comment may be written."""

    file: str
    lineno: int  # the def line (1-based)
    comments: tuple[tuple[int, str], ...]


@cache
def _source(rel: str) -> tuple[list[str], ast.Module]:
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    return text.splitlines(), ast.parse(text, filename=rel)


def mark_blocks(lines: list[str], tree: ast.Module) -> dict[str, tuple[int, list[tuple[int, str]]]]:
    """``{"[Class::]test_name": (def line, per-push comments)}`` for every
    test function, where the comments are those at the function, at its
    class and at the module's ``pytestmark`` together (a mark written on
    the class or the module is commented there)."""
    module_comments: list[tuple[int, str]] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets):
            module_comments += per_push_comments(lines, *_region(lines, stmt))
    out = {}

    def add(func, prefix, inherited):
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) and func.name.startswith("test"):
            out[prefix + func.name] = (func.lineno,
                                       inherited + per_push_comments(lines, *_region(lines, func)))

    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef):
            cls_comments = list(module_comments) + per_push_comments(lines, *_region(lines, stmt))
            for s in stmt.body:
                if isinstance(s, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "pytestmark" for t in s.targets):
                    cls_comments += per_push_comments(lines, *_region(lines, s))
            for func in stmt.body:
                add(func, f"{stmt.name}::", cls_comments)
        else:
            add(stmt, "", list(module_comments))
    return out


# --------------------------------------------------------------------------
# Reading the testing standards: the slow-only table
# --------------------------------------------------------------------------
def slow_only_table(doc: str) -> list[str]:
    """Node ids in the "Slow-marked tests" column of the slow-only table.

    A cell lists them in backticks; one that starts with ``::`` continues
    the file of the id before it (``tests/a.py::test_x`` and ``::test_y``).
    """
    if TABLE_HEADING not in doc:
        raise AssertionError(f"{STANDARDS.name} has no {TABLE_HEADING!r} section")
    section = doc.split(TABLE_HEADING, 1)[1].split("\n### ", 1)[0]
    rows = [ln for ln in section.splitlines() if ln.startswith("|")]
    if len(rows) < 3 or "Slow-marked tests" not in rows[0]:
        raise AssertionError("the slow-only table is missing or its columns moved: "
                             f"{rows[:1]}")
    column = [c.strip() for c in rows[0].strip("|").split("|")].index("Slow-marked tests")
    ids = []
    for row in rows[2:]:
        cell = [c.strip() for c in row.strip("|").split("|")][column]
        current = ""
        for span in re.findall(r"`([^`]+)`", cell):
            if span.startswith("tests/"):
                current = span.split("::", 1)[0]
                ids.append(span)
            elif span.startswith("::") and current:
                ids.append(current + span)
            else:
                raise AssertionError(f"slow-only table: {span!r} is not a node id "
                                     "(`tests/<file>.py::<test>` or `::<test>` after one)")
    return ids


def _covers(target: str, nodeid: str) -> bool:
    """Whether ``target`` (a node id, a class, or a test without its
    parameters) names the item ``nodeid``."""
    return nodeid == target or nodeid.startswith(target + "[") or nodeid.startswith(target + "::")


# --------------------------------------------------------------------------
# Asking pytest: what is slow, and what the default lane runs
# --------------------------------------------------------------------------
_COLLECT = r"""
import json, os, sys
import pytest

class Dump:
    items = []

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, items):
        Dump.items = [[i.nodeid, i.get_closest_marker("slow") is not None,
                       i.get_closest_marker("skip") is not None] for i in items]

code = pytest.main(sys.argv[1:], plugins=[Dump()])
with open(os.environ["SLOW_RULE_DUMP"], "w") as f:
    json.dump(Dump.items, f)
sys.exit(code)
"""


@dataclass(frozen=True)
class Collection:
    slow: frozenset[str]
    default_lane: frozenset[str]
    skipped: frozenset[str] = frozenset()


def partition(items: list[list]) -> Collection:
    """``[[node id, slow?, skip-marked?], ...]`` into what the default lane
    runs, what it does not (slow), and what it collects but always skips."""
    return Collection(
        slow=frozenset(n for n, slow, _ in items if slow),
        default_lane=frozenset(n for n, slow, skip in items if not slow and not skip),
        skipped=frozenset(n for n, slow, skip in items if not slow and skip))


def _files_named() -> set[str]:
    """The files the ``# Per push:`` comments and the slow-only table name."""
    texts = list(slow_only_table(STANDARDS.read_text(encoding="utf-8")))
    for rel in _framework_files():
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        if "Per push:" in text:   # no parse for the files that name nothing
            lines = text.splitlines()
            texts += [t for _, t in per_push_comments(lines, 0, len(lines))]
    return {i.split("::", 1)[0] for t in texts for i in _NODE_ID.findall(t)}


def may_hold_a_slow_mark(text: str) -> bool:
    """Whether a test file can carry a slow mark: its text says "slow".

    A mark is spelled ``pytest.mark.slow``, or a name bound to it, in the
    file itself; no conftest or plugin under ``tests/`` adds marks
    (:func:`test_no_conftest_or_plugin_adds_a_mark` holds that).  Collecting
    only these files and the witnesses' (about a third of the framework files)
    keeps this module inside the default lane's 5 s budget.
    """
    return "slow" in text.lower()


@pytest.fixture(scope="module")
def collection(tmp_path_factory) -> Collection:
    """Every item in a framework file that can hold a slow mark, plus the
    files the comments and the table name, collected once with
    ``-m "slow or not slow"``."""
    paths = sorted({rel for rel in _framework_files()
                    if may_hold_a_slow_mark((REPO_ROOT / rel).read_text(encoding="utf-8"))}
                   | {f for f in _files_named() if (REPO_ROOT / f).is_file()})
    dump = tmp_path_factory.mktemp("slow_rule") / "items.json"
    env = {k: v for k, v in os.environ.items()
           if k not in ("MADDENING_TEST_SHARD", "MADDENING_TEST_JAX_TIMING", "PYTEST_ADDOPTS")}
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", JAX_PLATFORMS="cpu", SLOW_RULE_DUMP=str(dump))
    proc = subprocess.run(
        [sys.executable, "-c", _COLLECT, "--collect-only", "-q", "-p", "no:cacheprovider",
         "-m", "slow or not slow", *paths],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return partition(json.loads(dump.read_text()))


@cache
def _framework_files() -> tuple[str, ...]:
    return tuple(sorted(p.relative_to(REPO_ROOT).as_posix()
                        for d in FRAMEWORK_PATHS for p in (REPO_ROOT / d).rglob("test_*.py")))


def _slow_functions(collection: Collection) -> dict[str, list[str]]:
    """``{function node id: [slow items]}``: parameters stripped."""
    out: dict[str, list[str]] = {}
    for nodeid in sorted(collection.slow):
        out.setdefault(nodeid.split("[", 1)[0], []).append(nodeid)
    return out


def _comments_at(function: str) -> tuple[int, list[tuple[int, str]]]:
    rel, name = function.split("::", 1)
    lines, tree = _source(rel)
    blocks = mark_blocks(lines, tree)
    if name in blocks:
        return blocks[name]
    # Inherited from a base class: read the comments where the method is written.
    inherited = [k for k in blocks if k.rsplit("::", 1)[-1] == name.rsplit("::", 1)[-1]]
    if len(inherited) == 1:
        return blocks[inherited[0]]
    raise AssertionError(f"{function}: collected, but its definition is not in {rel}")


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------
def uncovered(function: str, items: list[str], comments: list[tuple[int, str]],
              table: list[str]) -> str:
    """Why the slow test ``function`` breaks the rule, or ``""`` if it does not.

    ``items`` are its slow-marked node ids, ``comments`` the ``# Per push:``
    comments at its mark and ``table`` the slow-only table's node ids.
    """
    if any(witness_ids(text)[0] for _, text in comments):
        return ""
    in_table = any(_covers(t, item) for t in table for item in items)
    if not in_table:
        return "no `# Per push: <node id>` at its mark, not in the table"
    if function.startswith(SIBLING_REQUIRED):
        return ("the slow-only table is not enough here: verify-hypothesis runs it on "
                "jax 0.10.2 only, so it needs a per-push sibling")
    return ""


def witness_problems(rel: str, comments: list[tuple[int, str]], collection: Collection) -> list[str]:
    """What is wrong with the ``# Per push:`` comments at one slow mark."""
    problems = []
    for lineno, text in comments:
        ids, bare = witness_ids(text)
        if not ids:
            problems.append(f"{rel}:{lineno}: names no node id: {text[:100]!r}")
        for name in bare:
            problems.append(f"{rel}:{lineno}: {name!r} is not a node id; write "
                            "tests/<file>.py::[<Class>::]<test>")
        for target in ids:
            if not any(_covers(target, n) for n in collection.default_lane):
                why = ("slow-marked itself" if any(_covers(target, n) for n in collection.slow)
                       else "skip-marked" if any(_covers(target, n) for n in collection.skipped)
                       else "pytest collects no such test")
                problems.append(f"{rel}:{lineno}: {target} does not run in the default lane ({why})")
    return problems


def test_every_slow_framework_test_names_a_per_push_witness_or_is_in_the_slow_only_table(collection):
    table = slow_only_table(STANDARDS.read_text(encoding="utf-8"))
    missing = []
    for function, items in _slow_functions(collection).items():
        rel = function.split("::", 1)[0]
        if not rel.startswith(FRAMEWORK_PATHS) or rel in NOT_FRAMEWORK:
            continue
        lineno, comments = _comments_at(function)
        why = uncovered(function, items, comments, table)
        if why:
            missing.append(f"{rel}:{lineno}: {function.split('::', 1)[1]} ({why})")
    assert not missing, (
        "slow-marked framework tests that say neither where their property is checked on every "
        "push nor that it is slow-only on purpose.  Above the mark, add `# Per push: <node id>` "
        "naming a cheaper test the default lane runs (write one if none exists), or add the "
        f"property to the table under {TABLE_HEADING!r} in {STANDARDS.relative_to(REPO_ROOT)} "
        "with the reason:\n  " + "\n  ".join(missing))


def test_every_named_per_push_witness_runs_in_the_default_lane(collection):
    problems = []
    for function in _slow_functions(collection):
        _, comments = _comments_at(function)
        problems += witness_problems(function.split("::", 1)[0], comments, collection)
    assert not problems, (
        "`# Per push:` comments at slow marks that do not name a test the default lane runs:\n  "
        + "\n  ".join(problems))


def test_no_witness_or_table_row_names_a_test_that_needs_usd_core():
    """A witness is a test the default lane runs, and the collection above
    hands every named file to pytest.  ``tests/usd/`` is neither where
    ``usd-core`` is not installed, which is every sharded lane: its conftest
    skips the directory, and pytest handed one of its files exits with that
    skip as an error.  This says so in every lane, with the reason, where
    the collection would only fail in some."""
    named = sorted(f for f in _files_named() if f.startswith("tests/usd/"))
    assert not named, (
        f"{named}: named by a `# Per push:` comment or the slow-only table, but the default "
        "lane skips tests/usd (no usd-core), so nothing there is a per-push witness.  Name a "
        "test the sharded lanes run")


def test_the_slow_only_table_and_the_exemptions_name_real_slow_tests(collection):
    problems = []
    for target in slow_only_table(STANDARDS.read_text(encoding="utf-8")):
        if not any(_covers(target, n) for n in collection.slow):
            problems.append(f"slow-only table: {target} names no slow-marked test pytest collects")
    for rel, reason in NOT_FRAMEWORK.items():
        if not reason.strip():
            problems.append(f"NOT_FRAMEWORK: {rel} gives no reason")
        if not rel.startswith(FRAMEWORK_PATHS):
            problems.append(f"NOT_FRAMEWORK: {rel} is not under a framework path")
        elif not any(n.startswith(rel + "::") for n in collection.slow):
            problems.append(f"NOT_FRAMEWORK: {rel} holds no slow-marked test (retire the entry)")
    assert not problems, "\n".join(problems)


# --------------------------------------------------------------------------
# The readers find what they are for (no collection: milliseconds)
# --------------------------------------------------------------------------
_SAMPLE = '''
import pytest

# Per push: tests/a/test_x.py::test_cheap
@pytest.mark.slow
def test_commented_above_the_mark():
    pass


@pytest.mark.slow  # a reason
# Costly tier: drawn shapes.
# Per push: tests/a/test_x.py::TestA::test_one and
#     tests/a/test_y.py::test_two[3-x]
@pytest.mark.parametrize("n", [1, 2])
def test_comment_between_decorators(n):
    pass


# Per push: test_bare_name, written without its file
@pytest.mark.slow
def test_bare():
    pass


# Per push: tests/a/test_x.py::test_cheap
def test_other():
    pass


@pytest.mark.slow
def test_none():
    pass


# Per push: tests/a/test_z.py::TestZ::test_cls
@pytest.mark.slow
class TestMarkedClass:
    def test_in_class(self):
        pass


class TestPlain:
    # Per push: tests/a/test_z.py::test_method_witness
    @pytest.mark.slow
    def test_method(self):
        pass


# Per push, in prose with a colon later: tests/a/test_x.py::test_cheap
@pytest.mark.slow
def test_prose_is_not_the_form():
    pass
'''


def test_the_comment_reader_finds_what_it_is_for():
    lines = _SAMPLE.splitlines()
    blocks = {k: [witness_ids(t) for _, t in v[1]] for k, v in mark_blocks(lines, ast.parse(_SAMPLE)).items()}
    assert blocks["test_commented_above_the_mark"] == [(["tests/a/test_x.py::test_cheap"], [])]
    assert blocks["test_comment_between_decorators"] == [
        (["tests/a/test_x.py::TestA::test_one", "tests/a/test_y.py::test_two[3-x]"], [])]
    assert blocks["test_bare"] == [([], ["test_bare_name"])]
    assert blocks["test_none"] == []
    assert blocks["TestMarkedClass::test_in_class"] == [(["tests/a/test_z.py::TestZ::test_cls"], [])]
    assert blocks["TestPlain::test_method"] == [(["tests/a/test_z.py::test_method_witness"], [])]
    # "Per push," is prose, not the form: it is not read as a witness.
    assert blocks["test_prose_is_not_the_form"] == []
    # A comment above one test is not read for the next one.
    assert blocks["test_other"] == [(["tests/a/test_x.py::test_cheap"], [])]
    # A name missing its class, and a file with no test, are bare.
    assert witness_ids("tests/a/test_x.py::test_q (below) and TestA::test_r") == (
        ["tests/a/test_x.py::test_q"], ["TestA", "test_r"])
    assert witness_ids("tests/a/test_x.py, every case") == ([], ["test_x"])


def test_the_table_reader_finds_what_it_is_for():
    doc = (f"{TABLE_HEADING}\n\nintro\n\n| Property | Slow-marked tests | Why |\n|---|---|---|\n"
           "| one | `tests/a.py::test_x` and `::test_y`; `tests/b.py::TestB` | r |\n"
           "| two | `tests/c.py::test_z[p]` | r |\n\nTo add a row ...\n\n### Next\n"
           "| x | `tests/d.py::test_not_in_the_section` | y |\n")
    assert slow_only_table(doc) == ["tests/a.py::test_x", "tests/a.py::test_y", "tests/b.py::TestB",
                                    "tests/c.py::test_z[p]"]
    with pytest.raises(AssertionError, match="not a node id"):
        slow_only_table(doc.replace("`::test_y`", "`test_y`"))
    assert _covers("tests/b.py::TestB", "tests/b.py::TestB::test_q[1]")
    assert _covers("tests/a.py::test_x", "tests/a.py::test_x[2]")
    assert not _covers("tests/a.py::test_x", "tests/a.py::test_xy")
    assert not _covers("tests/a.py::test_q", "tests/a.py::TestA::test_q")


def test_the_rule_takes_a_comment_or_a_table_row_but_not_a_row_alone_for_a_hypothesis_property():
    comment = [(3, "tests/a/test_x.py::test_cheap")]
    table = ["tests/core/test_t.py::TestSlow"]
    assert uncovered("tests/core/test_t.py::test_f", ["tests/core/test_t.py::test_f"], comment, []) == ""
    assert uncovered("tests/core/test_t.py::TestSlow::test_f",
                     ["tests/core/test_t.py::TestSlow::test_f[1]"], [], table) == ""
    assert "not in the table" in uncovered("tests/core/test_t.py::test_f",
                                           ["tests/core/test_t.py::test_f"], [], table)
    # prose naming no test is not a witness
    assert "not in the table" in uncovered("tests/core/test_t.py::test_f",
                                           ["tests/core/test_t.py::test_f"],
                                           [(3, "the seed-0 cases of this test")], [])
    hyp = "tests/verification/hypothesis/test_h.py::TestH::test_p"
    assert "per-push sibling" in uncovered(hyp, [hyp], [], ["tests/verification/hypothesis/test_h.py::TestH"])
    assert uncovered(hyp, [hyp], comment, []) == ""


def test_a_witness_resolves_only_to_an_item_the_default_lane_runs():
    c = partition([["tests/a.py::test_slow", True, False], ["tests/a.py::test_mixed[1]", True, False],
                   ["tests/a.py::TestA::test_fast", False, False],
                   ["tests/a.py::test_mixed[0]", False, False],
                   ["tests/a.py::test_skipped", False, True]])
    assert "skip-marked" in witness_problems("tests/b.py", [(1, "tests/a.py::test_skipped")], c)[0]
    ok = [(1, "tests/a.py::TestA::test_fast and tests/a.py::test_mixed (its fast case)")]
    assert witness_problems("tests/b.py", ok, c) == []
    assert "slow-marked itself" in witness_problems("tests/b.py", [(1, "tests/a.py::test_slow")], c)[0]
    assert "collects no such test" in witness_problems("tests/b.py", [(1, "tests/a.py::test_fast")], c)[0]
    bare = witness_problems("tests/b.py", [(1, "tests/a.py::test_mixed and TestA::test_fast")], c)
    assert len(bare) == 2 and all("is not a node id" in b for b in bare), bare
    assert "names no node id" in witness_problems("tests/b.py", [(1, "below")], c)[0]


def test_the_framework_paths_are_the_ones_the_testing_standards_name():
    """The guide says which directories this rule covers; dropping one here
    would exempt it silently."""
    doc = STANDARDS.read_text(encoding="utf-8")
    para = doc[doc.index("`tests/compliance/test_slow_only_rule.py` checks this rule"):]
    para = para[:para.index("using pytest's own collection")]
    named = set(re.findall(r"`(tests/[\w/]+)`", para))
    assert named == {p.rstrip("/") for p in FRAMEWORK_PATHS}, (named, FRAMEWORK_PATHS)


def test_no_conftest_or_plugin_adds_a_mark():
    """What lets the collection skip files that never say "slow": a mark can
    only come from the file itself.  A conftest or plugin under ``tests/``
    that added one to items would put slow tests where this module does not
    look."""
    found = []
    for path in sorted((REPO_ROOT / "tests").rglob("*.py")):
        if path.name != "conftest.py" and not path.name.startswith("_"):
            continue
        text = path.read_text(encoding="utf-8")
        if "add_marker(" in text or "pytest_itemcollected" in text:
            found.append(path.relative_to(REPO_ROOT).as_posix())
    assert not found, (
        f"{found} add marks to collected items; collect every framework file in this "
        "module's `collection` fixture instead of only those that say \"slow\"")


def test_a_file_that_never_says_slow_is_the_only_one_skipped():
    assert may_hold_a_slow_mark("@pytest.mark.slow\ndef test_x(): pass")
    assert may_hold_a_slow_mark("from tests.helpers import SLOW_CASES")
    assert not may_hold_a_slow_mark("def test_x():\n    pass\n")
