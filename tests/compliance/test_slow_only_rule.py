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

**A witness is a test that runs on every push and can fail there.**  Asking
only for its ``slow`` and ``skip`` marks let through a witness that reports
success when its assertions fail, and one that is collected and never runs.
So a named witness is refused when every item it names is one of these:

* **in a file the per-push lane never collects**.  The collection below
  hands every named file to pytest by path, and pytest collects a file it
  is handed whatever its name and whatever ``--ignore`` says; the lane
  runs ``pytest tests/``, which does neither.  So the lane's own selection
  is read (:func:`lane_exclusions`): the paths and ``--ignore`` /
  ``--deselect`` arguments of its pytest command in
  ``.github/workflows/ci.yml``, read as ``test_ci_sharding.py`` reads them
  (an argument that selects in any other way is refused, not guessed at);
  the paths ``scripts/check_anomalies.py`` records in
  ``PATHS_CI_NEVER_RUNS``; and pytest's ``python_files`` patterns from the
  project's configuration, which a file must match to be collected from a
  directory;
* **skipped by a mark**: ``skip``, or a ``skipif`` whose condition is true
  in this lane.  The condition is the value pytest itself holds for the
  collected item (:data:`_COLLECT`), never the decorator's text: the
  multi-device witnesses under ``tests/cloud/multigpu`` carry
  ``skipif(len(jax.devices()) < 4)``, which is false there because that
  directory's conftest provides the devices, and they stay witnesses.  A
  condition written as a string is evaluated by pytest only when the test
  is set up; it is not decided here and is refused: write the expression.
  A mark whose condition reads a variable this collection takes out of its
  environment (:data:`_UNSET_FOR_COLLECTION`: the sharded lanes set
  ``MADDENING_TEST_SHARD``) is decided here otherwise than in the lane, and
  is refused when the mark, or a module-level name or function it uses,
  spells the variable out;
* **expected to fail**: an ``xfail`` mark that applies to the item in this
  lane -- strict or not, with ``raises=`` or without, ``run=False``, on the
  function, its class, the module or one parameter -- or a ``pytest.xfail()``
  call in its body, or ``unittest.expectedFailure``.  Its conditions are
  decided as a ``skipif``'s are;
* **skipped by its own body**: a ``pytest.skip()`` every run reaches, in the
  test, in a module-level helper it calls, in a fixture of its module it
  requests or in an xunit ``setup_module`` / ``setup_function`` (or a
  ``setup_class`` / ``setup_method`` / ``setUp`` / ``setUpClass`` of its
  class).  ``scripts/check_anomalies.py`` already reads these for the
  registry's evidence, and its reader is the one used here
  (:func:`body_outcomes`).  ``unittest.skip`` on the test or its class, and
  a ``raise unittest.SkipTest`` in its body, are read too.  A skip under a
  condition (``unittest.skipIf`` / ``skipUnless`` included), and a
  ``pytest.importorskip()``, cannot be decided from the source: such a
  witness is refused unless :data:`CONDITIONAL_BODY_SKIPS` names it with
  the reason its condition is false in the default lane.  The test read is
  the one pytest collects: the LAST binding of its name in its module or
  class, and a name whose last binding is not a ``def`` is not read.

What this does not see, as the reader it borrows does not: a skip in a
fixture or helper of another module (a ``conftest.py`` included), in a
method reached through ``self``, behind a ``usefixtures`` mark or in a
collection hook; a skip in a helper nested in the test, one called through
a local alias (``leave = pytest.skip``) or imported from ``_pytest``
(``from _pytest.outcomes import skip``); a mark applied while the test
runs (``request.applymarker``, ``request.node.add_marker`` in the body or
a fixture); a conftest's ``pytest_runtest_setup`` or autouse fixture that
calls ``pytest.skip`` / ``pytest.xfail``; a condition that reads an unset
variable through an imported name, or by building the variable's name;
and a witness that runs and asserts nothing.  Those are the
run's to show, not the source's.  (A conftest that deselects, ignores or
marks items is refused by ``test_ci_sharding.py`` and
:func:`test_no_conftest_or_plugin_adds_a_mark`.)

**The one exception** is an xfail witness of an xfail pin.  A slow test that
is itself the pin of an open anomaly has nothing that passes to point at:
its per-push sibling is the cheaper pin of the same defect, which turns red
on the push that fixes it.  An xfail-marked witness is therefore accepted
for a slow test when all of this holds, and only then
(:func:`shared_open_pin`):

* the witness's mark is ``strict=True`` (a pin: it fails once the defect is
  gone), and so is the mark of **every** slow-marked item of the slow test
  (one parametrisation that is expected to pass needs a witness that can
  fail);
* the witness's mark does not say ``run=False``: pytest never runs such a
  test, so it cannot turn red on the push that fixes the defect;
* **the same anomaly**: one registry id, ``MADD-ANO-<n>``, is written in the
  ``reason=`` of the witness's mark and of each of those slow items' marks.
  The ids are read from the marks pytest collected; nothing is inferred
  from a docstring, a comment or a file name;
* that entry's ``resolution_status`` in
  ``docs/validation/known_anomalies.yaml`` is one of :data:`OPEN_STATUSES`.
  A pin of a resolved entry is not a pin of anything.

A slow test that is expected to pass never takes the exception, whatever
its witness's reason says.

Under ``tests/verification/hypothesis/`` the table is no way out: the
``verify-hypothesis`` job runs those slow properties on every push, but on
jax 0.10.2 only, so each needs a per-push sibling the default lane runs on
both JAX lanes.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STANDARDS = REPO_ROOT / "docs" / "developer_guide" / "testing_standards.md"
TABLE_HEADING = "### What only the slow lane checks"
REGISTRY = REPO_ROOT / "docs" / "validation" / "known_anomalies.yaml"
ANOMALY_GATE = REPO_ROOT / "scripts" / "check_anomalies.py"

#: A registry id as an xfail mark's ``reason=`` writes it.
_ANOMALY_ID = re.compile(r"\bMADD-ANO-\d+\b")

#: The ``resolution_status`` values of a defect that is still in the tree:
#: the only entries an xfail pin can stand for.  Anything else (``resolved``,
#: a status nobody has enumerated) is not open.
OPEN_STATUSES = ("open", "partially_resolved")

#: Witnesses whose body skips under a condition, with the reason the
#: condition is false in the default lane: ``{node id of the test function
#: (no parameters): reason}``.  Nothing reads the condition of a
#: ``pytest.skip()`` under an ``if`` or of a ``pytest.importorskip()`` from
#: the source, so every such witness is refused unless it is named here; an
#: entry whose test no longer skips conditionally, or is no longer a
#: witness, fails too (it would excuse the next skip written there).
_FOUR_DEVICES = ("skips with fewer than four devices; tests/cloud/multigpu/conftest.py gives every "
                 "CPU lane 16 virtual devices before jax starts (test_conftest_device_policy.py)")
_RUN_POD = "tests/cloud/multigpu/test_run_pod_seeded_faults.py::"
CONDITIONAL_BODY_SKIPS: dict[str, str] = {
    "tests/cloud/multigpu/test_run_capacity_rungs_and_ramp.py::"
    "test_one_rung_runs_in_a_process_of_its_own_as_the_session_starts_it": _FOUR_DEVICES,
    "tests/cloud/multigpu/test_run_pod_dry_run.py::"
    "test_exchange_timing_input_is_presharded_on_the_mesh": _FOUR_DEVICES,
    f"{_RUN_POD}test_a_seeded_fault_fails_the_cg_part_of_the_gradient_goal": _FOUR_DEVICES,
    f"{_RUN_POD}test_a_seeded_miscount_fails_the_forward_goals_exact_count": _FOUR_DEVICES,
    f"{_RUN_POD}test_one_step_of_the_stencil_goals_node_shows_each_seeded_fault": _FOUR_DEVICES,
    f"{_RUN_POD}test_the_cg_part_of_the_gradient_goal_passes_on_solves_that_converged": _FOUR_DEVICES,
    f"{_RUN_POD}test_the_forward_goal_counts_a_field_of_ones_exactly_on_both_sides": _FOUR_DEVICES,
    "tests/compliance/test_step_program_digests_are_frozen.py::"
    "test_the_digests_have_only_grown_since_their_capture":
        "skips where the clone is too shallow to hold the commit the digests were captured on, "
        "which is the sharded test jobs (fetch-depth 2); the compliance job fetches the whole "
        "history and runs tests/compliance on every push, and what it checks is the file's git "
        "history, the same under any jax",
}

#: Variables :func:`collected` takes out of its subprocess's environment: the
#: shard would deselect most of the files, the timing plugin and extra
#: arguments would change what is collected.  The sharded lanes set the
#: first, so a mark whose condition reads one of them is decided here
#: otherwise than there (:func:`_marks_read_an_unset_variable`).
_UNSET_FOR_COLLECTION = ("MADDENING_TEST_SHARD", "MADDENING_TEST_JAX_TIMING", "PYTEST_ADDOPTS")

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
#: The parameter id is read up to the bracket that closes the one opening it,
#: written directly after the name: it may hold spaces and any other
#: character but a line break, and brackets nested one deep in balanced
#: pairs (``test_w[the case]``, ``test_w[a[1]-b]``).  An id holding an
#: unbalanced bracket is read up to that bracket and then names no item, so
#: it is refused rather than read as another test.
_NODE_ID = re.compile(r"tests/[\w/.\-]+?\.py(?:::[A-Za-z_]\w*)+"
                      r"(?:\[(?:[^\[\]\n]|\[[^\[\]\n]*\])*\])?")
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

def applies(mark):
    # pytest's own rule (_pytest.skipping): no condition at all, or any
    # condition true.  "string": a condition pytest evaluates in the item's
    # namespace at setup, which is not decided here and is read as applying.
    conditions = (mark.kwargs["condition"],) if "condition" in mark.kwargs else mark.args
    if not conditions:
        return "always"
    held = ""
    for condition in conditions:
        if isinstance(condition, str):
            held = held or "string"
        elif condition:
            return "true"
    return held

def skipif(item):
    return next((a for a in map(applies, item.iter_markers("skipif")) if a), "")

def xfail(item, strict_default):
    for mark in item.iter_markers("xfail"):      # closest first: the one pytest takes
        a = applies(mark)
        if a:
            return [str(mark.kwargs.get("reason", "")),
                    bool(mark.kwargs.get("strict", strict_default)), a,
                    bool(mark.kwargs.get("run", True))]
    return None

def strict_by_default(config):
    # The ini option an xfail mark without strict= falls back to: pytest 9
    # names it strict_xfail and keeps the older name, earlier ones know
    # only xfail_strict.  Neither known: say so, never assume.
    for name in ("strict_xfail", "xfail_strict"):
        try:
            return bool(config.getini(name))
        except (ValueError, KeyError):
            continue
    raise RuntimeError("this pytest has neither the strict_xfail nor the xfail_strict option")

class Dump:
    items = []

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        strict_default = strict_by_default(config)
        Dump.items = [[i.nodeid, i.get_closest_marker("slow") is not None,
                       i.get_closest_marker("skip") is not None, skipif(i),
                       xfail(i, strict_default)] for i in items]

code = pytest.main(sys.argv[1:], plugins=[Dump()])
with open(os.environ["SLOW_RULE_DUMP"], "w") as f:
    json.dump(Dump.items, f)
sys.exit(code)
"""


@dataclass(frozen=True)
class Xfail:
    """The ``xfail`` mark pytest applies to an item in this lane."""

    reason: str
    strict: bool
    #: ``"always"`` (no condition), ``"true"``, or ``"string"`` (a condition
    #: written as a string: not decided here, and read as applying).
    applies: str = "always"
    #: ``False`` for ``run=False``: pytest reports the item xfailed without
    #: running it.
    run: bool = True


@dataclass(frozen=True)
class Collection:
    slow: frozenset[str]
    #: What the default lane runs: not slow-marked, and skipped by no mark.
    default_lane: frozenset[str]
    #: Not slow-marked, and skip-marked.
    skipped: frozenset[str] = frozenset()
    #: Not slow-marked, and skipped by a ``skipif`` in this lane: ``"true"``,
    #: ``"always"`` or ``"string"`` (:class:`Xfail`'s vocabulary).
    skipif: dict[str, str] = field(default_factory=dict)
    #: Every item, slow-marked or not, an ``xfail`` mark applies to.
    xfail: dict[str, Xfail] = field(default_factory=dict)
    #: ``{file or node id: why}`` for what the per-push lane never runs
    #: although this collection, handed the file by path, holds it
    #: (:func:`lane_exclusions`).  Filled by the ``collection`` fixture.
    not_collected: dict[str, str] = field(default_factory=dict)


def partition(items: list[list]) -> Collection:
    """``[[node id, slow?, skip-marked?, skipif, xfail], ...]`` (the rows
    :data:`_COLLECT` writes) into what the default lane runs, what it does
    not (slow), what it collects and a mark always skips, and the items an
    xfail mark applies to."""
    return Collection(
        slow=frozenset(n for n, slow, _skip, _skipif, _xfail in items if slow),
        default_lane=frozenset(n for n, slow, skip, skipif, _xfail in items
                               if not slow and not skip and not skipif),
        skipped=frozenset(n for n, slow, skip, _skipif, _xfail in items if not slow and skip),
        skipif={n: skipif for n, slow, skip, skipif, _xfail in items
                if not slow and not skip and skipif},
        xfail={n: Xfail(*xfail) for n, _slow, _skip, _skipif, xfail in items if xfail})


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
def collection(tmp_path_factory, pytestconfig) -> Collection:
    """Every item in a framework file that can hold a slow mark, plus the
    files the comments and the table name, collected once with
    ``-m "slow or not slow"``; and what of it the per-push lane never runs."""
    paths = sorted({rel for rel in _framework_files()
                    if may_hold_a_slow_mark((REPO_ROOT / rel).read_text(encoding="utf-8"))}
                   | {f for f in _files_named() if (REPO_ROOT / f).is_file()})
    rows = collected(paths, REPO_ROOT, tmp_path_factory.mktemp("slow_rule") / "items.json")
    return dataclasses.replace(partition(rows), not_collected=lane_exclusions(
        paths, per_push_lane_args(), pytestconfig.getini("python_files"),
        _anomaly_gate()._ci_never_runs))  # noqa: SLF001


# --------------------------------------------------------------------------
# What the per-push lane never collects, although a run naming the file would
# --------------------------------------------------------------------------
def per_push_lane_args() -> list[str]:
    """The arguments of the per-push lane's pytest, read from
    ``.github/workflows/ci.yml`` by ``test_ci_sharding.py``'s reader, at
    the job and step it names as the sharded per-push lane."""
    from tests.compliance import test_ci_sharding as sharding  # noqa: PLC0415

    job, step = sharding.SHARD_SETTING["ci.yml"]
    return sharding._pytest_args(sharding._workflow("ci.yml")["jobs"][job], step)  # noqa: SLF001


def lane_selection(args: list[str], reporting=None) -> tuple[list[str], list[str], list[str]]:
    """``(paths, ignores, deselects)`` of a pytest command's arguments.

    ``reporting`` are the arguments that change only what is reported
    (``test_ci_sharding.REPORTING_ARGS`` by default).  Any other argument
    starting with ``-`` selects in a way this reader does not model
    (``-k``, ``-m``, ``--ignore-glob``, ...) and is refused: guessing would
    accept a witness the lane drops.
    """
    if reporting is None:
        from tests.compliance import test_ci_sharding as sharding  # noqa: PLC0415

        reporting = sharding.REPORTING_ARGS
    paths, ignores, deselects = [], [], []
    rest = list(args)
    while rest:
        arg = rest.pop(0)
        if arg in reporting:
            continue
        option, eq, value = arg.partition("=")
        if option in ("--ignore", "--deselect"):
            if not eq:
                if not rest:
                    raise AssertionError(f"the per-push lane's pytest ends with {arg!r} and no value")
                value = rest.pop(0)
            (ignores if option == "--ignore" else deselects).append(value)
        elif arg.startswith("-"):
            raise AssertionError(
                f"the per-push lane's pytest takes {arg!r}, which is neither a path, --ignore, "
                "--deselect nor a reporting argument (test_ci_sharding.REPORTING_ARGS): say in "
                "lane_selection how it selects, or no witness can be judged")
        else:
            paths.append(arg)
    norm = [os.path.normpath(p).replace(os.sep, "/") for p in paths]
    return norm, [os.path.normpath(i).replace(os.sep, "/") for i in ignores], deselects


def _within(rel: str, top: str) -> bool:
    return rel == top or top == "." or rel.startswith(top.rstrip("/") + "/")


def lane_exclusions(files: list[str], args: list[str], python_files: list[str], never_runs, *,
                    root: Path = REPO_ROOT, reporting=None) -> dict[str, str]:
    """``{file or node id: why}`` for what the lane whose pytest takes
    ``args`` never runs, among ``files`` (repository-relative).

    ``python_files`` are pytest's patterns for a test file (the project's
    ``python_files``), matched as pytest matches them
    (``_pytest.pathlib.fnmatch_ex`` on the absolute path); ``never_runs``
    is ``scripts/check_anomalies.py``'s ``_ci_never_runs`` (the entry of
    ``PATHS_CI_NEVER_RUNS`` holding a path, or ``None``).  A file is
    collected by the lane only when it is under one of the command's paths
    (a path that names a file or a node id stands for that file), under
    none of its ``--ignore`` paths, matches a ``python_files`` pattern
    unless the command names it, and is not recorded as never run.  A
    ``--deselect`` drops the node id it names.
    """
    from _pytest.pathlib import fnmatch_ex  # noqa: PLC0415  (pytest's own matcher)

    paths, ignores, deselects = lane_selection(args, reporting)
    out = {}
    for rel in files:
        named = [p for p in paths if p.split("::", 1)[0] == rel]
        under = [p for p in paths if _within(rel, p.split("::", 1)[0])]
        ignored = next((i for i in ignores if _within(rel, i)), None)
        recorded = never_runs(rel)
        if not under:
            out[rel] = (f"the per-push lane runs pytest on {paths} only, and this file is under "
                        "none of them")
        elif ignored:
            out[rel] = f"the per-push lane passes --ignore={ignored}"
        elif recorded:
            out[rel] = (f"scripts/check_anomalies.py records {recorded} in PATHS_CI_NEVER_RUNS: "
                        "no CI lane runs it")
        elif not named and not any(fnmatch_ex(p, root / rel) for p in python_files):
            out[rel] = (f"its name matches none of pytest's python_files {list(python_files)}, so "
                        "`pytest tests/` never collects it; only a run naming the file does")
    for nodeid in deselects:
        out[nodeid] = f"the per-push lane passes --deselect {nodeid}"
    return out


def _lane_drops(nodeid: str, collection: Collection) -> str:
    """Why the per-push lane never runs ``nodeid`` (an item, or a test or
    class above items), or ``""``."""
    return next((why for key, why in collection.not_collected.items() if _covers(key, nodeid)), "")


def collected(paths: list[str], cwd: Path, dump: Path) -> list[list]:
    """The rows :data:`_COLLECT` writes for ``paths``: one pytest
    ``--collect-only`` over them with ``-m "slow or not slow"``, in a process
    of its own started in ``cwd``."""
    env = {k: v for k, v in os.environ.items() if k not in _UNSET_FOR_COLLECTION}
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", JAX_PLATFORMS="cpu", SLOW_RULE_DUMP=str(dump))
    proc = subprocess.run(
        [sys.executable, "-c", _COLLECT, "--collect-only", "-q", "-p", "no:cacheprovider",
         "-m", "slow or not slow", *paths],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(dump.read_text())


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
# Reading a witness: what its own body does, and the registry's statuses
# --------------------------------------------------------------------------
@cache
def _anomaly_gate():
    """``scripts/check_anomalies.py`` as a module (the scripts are not a
    package): its reader of the skips and xfails a test reaches without a
    mark is the one this rule uses."""
    spec = importlib.util.spec_from_file_location("_slow_rule_check_anomalies", ANOMALY_GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def outcomes_in(tree: ast.Module, name: str) -> tuple[str, ...]:
    """What the test ``[<Class>::]<test>`` of a parsed module reaches when
    it runs, as ``scripts/check_anomalies.py`` reads it: ``"skip"`` for a
    ``pytest.skip()`` every run reaches (in the test, a module-level helper
    it calls, a fixture of the module it requests), ``"skipif"`` for one
    under a condition and for any ``pytest.importorskip()``, ``"xfail"``
    for a ``pytest.xfail()`` anywhere.  ``("unread",)`` when the test is
    not written in the module (inherited from, or generated in, another),
    or when the last binding of its name (or its class's) is not a
    definition: nothing can be said of a body that was not read.

    Read here besides (:func:`_more_outcomes`): xunit setups,
    ``unittest.skip`` / ``skipIf`` / ``skipUnless`` / ``expectedFailure``
    on the test or its classes, ``raise unittest.SkipTest`` in its body,
    and ``"unset"`` for a mark whose condition reads a variable of
    :data:`_UNSET_FOR_COLLECTION`."""
    gate = _anomaly_gate()
    scope, classes, node = tree.body, [], None
    for part in name.split("::"):
        node = _last_binding(scope, part)
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return ("unread",)
        if isinstance(node, ast.ClassDef):
            classes.append(node)
            scope = node.body
    if isinstance(node, ast.ClassDef):
        return ("unread",)
    aliases = gate._pytest_aliases(tree)  # noqa: SLF001
    return (tuple(gate._test_function_kinds(node, classes, tree, aliases))  # noqa: SLF001
            + _more_outcomes(node, classes, tree, aliases))


def _last_binding(scope: list[ast.stmt], name: str) -> ast.stmt | None:
    """The last statement of ``scope`` that binds ``name`` (a ``def``, a
    ``class``, an assignment, an import, a ``del``): the one the module or
    class holds when pytest collects it."""
    last = None
    for stmt in scope:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound = {stmt.name}
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
            targets = stmt.targets if isinstance(stmt, (ast.Assign, ast.Delete)) else [stmt.target]
            bound = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            bound = {(a.asname or a.name).split(".")[0] for a in stmt.names}
        else:
            continue
        if name in bound:
            last = stmt
    return last


#: What a ``unittest`` name reached from a test's decorators or body does.
_UNITTEST_OUTCOMES = {"unittest.skip": "skip", "unittest.skipIf": "skipif",
                      "unittest.skipUnless": "skipif", "unittest.expectedFailure": "xfail",
                      "unittest.SkipTest": "skip"}
#: The xunit setups pytest runs before a test: at module level, and in its class.
_MODULE_SETUPS = ("setup_module", "setUpModule")
_FUNCTION_SETUPS = ("setup_function",)
_CLASS_SETUPS = ("setup_class", "setup_method", "setUp", "setUpClass")


def _unittest_name(expr: ast.expr, tree: ast.Module) -> str:
    """``expr`` as ``unittest.<name>`` when it is one, through the module's
    imports (``import unittest as ut``, ``from unittest import skip``,
    ``unittest.case``), or ``""``."""
    names = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            names.update({a.asname or a.name: a.name for a in stmt.names
                          if a.name in ("unittest", "unittest.case")})
        elif isinstance(stmt, ast.ImportFrom) and stmt.module in ("unittest", "unittest.case"):
            names.update({a.asname or a.name: f"unittest.{a.name}" for a in stmt.names})
    parts = []
    while isinstance(expr, ast.Attribute):
        parts.insert(0, expr.attr)
        expr = expr.value
    if not isinstance(expr, ast.Name) or expr.id not in names:
        return ""
    dotted = ".".join([names[expr.id], *parts]).replace("unittest.case.", "unittest.")
    return dotted if dotted in _UNITTEST_OUTCOMES else ""


def _more_outcomes(func, classes: list[ast.ClassDef], tree: ast.Module, aliases) -> tuple[str, ...]:
    """What :func:`outcomes_in` reads beyond the anomaly gate's reader."""
    gate = _anomaly_gate()
    kinds = []
    # unittest's decorators on the test and on each class around it.
    for deco in func.decorator_list + [d for cls in classes for d in cls.decorator_list]:
        kind = _UNITTEST_OUTCOMES.get(_unittest_name(deco.func if isinstance(deco, ast.Call) else deco,
                                                     tree))
        if kind:
            kinds.append(kind)
    # raise unittest.SkipTest(...): every run when straight in the body, else under a condition.
    for node in _own_nodes(func.body):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if _unittest_name(exc, tree) == "unittest.SkipTest":
                kinds.append("skip" if node in func.body else "skipif")
    # xunit setups, read as the test's own body is.
    wanted = set(_MODULE_SETUPS) | (set() if classes else set(_FUNCTION_SETUPS))
    setups = [s for s in tree.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
              and s.name in wanted]
    setups += [s for cls in classes for s in cls.body
               if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name in _CLASS_SETUPS]
    helpers = gate._module_functions(tree)  # noqa: SLF001
    for setup in setups:
        kinds += gate._imperative_kinds(setup.body, aliases, helpers)  # noqa: SLF001
    if _marks_read_an_unset_variable(func, classes, tree):
        kinds.append("unset")
    return tuple(kinds)


def _own_nodes(body: list[ast.stmt]):
    """Every node of ``body`` that runs where it is written: nested
    ``def`` / ``class`` / ``lambda`` bodies are not entered."""
    todo = list(body)
    while todo:
        node = todo.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            todo += ast.iter_child_nodes(node)


def _marks_read_an_unset_variable(func, classes: list[ast.ClassDef], tree: ast.Module) -> list[str]:
    """The variables of :data:`_UNSET_FOR_COLLECTION` that the marks on
    ``func`` read: its decorators, its classes' decorators and
    ``pytestmark``, the module's ``pytestmark``, and every module-level
    name or function those use, followed to the end."""
    def pytestmark(body):
        return [s.value for s in body if isinstance(s, (ast.Assign, ast.AnnAssign)) and s.value
                is not None and any(isinstance(t, ast.Name) and t.id == "pytestmark"
                                    for t in (s.targets if isinstance(s, ast.Assign) else [s.target]))]

    exprs = list(func.decorator_list) + pytestmark(tree.body)
    for cls in classes:
        exprs += list(cls.decorator_list) + pytestmark(cls.body)
    module_names = {}
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_names[stmt.name] = stmt
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
            for t in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]):
                module_names.update({n.id: stmt for n in ast.walk(t) if isinstance(n, ast.Name)})
    texts, seen, todo = [], set(), exprs
    while todo:
        node = todo.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        texts.append(ast.unparse(node))
        todo += [module_names[n.id] for n in ast.walk(node)
                 if isinstance(n, ast.Name) and n.id in module_names]
    return sorted({v for v in _UNSET_FOR_COLLECTION for t in texts if v in t})


@cache
def body_outcomes(function: str) -> tuple[str, ...]:
    """:func:`outcomes_in` for the test function ``tests/<file>.py::[<Class>::]<test>``."""
    rel, name = function.split("::", 1)
    return outcomes_in(_source(rel)[1], name)


@cache
def _registry_statuses() -> dict[str, str]:
    """``{anomaly id: resolution_status}`` of the registry."""
    import yaml  # noqa: PLC0415

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    data = yaml.load(REGISTRY.read_text(encoding="utf-8"), Loader=loader)
    return {str(a["anomaly_id"]): str(a.get("resolution_status", "")) for a in data["anomalies"]}


def registry_status(anomaly_id: str) -> str:
    """The entry's ``resolution_status``, or ``""`` when the registry holds no such entry."""
    return _registry_statuses().get(anomaly_id, "")


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


def shared_open_pin(witness: Xfail, slow_items: list[str], collection: Collection,
                    status=registry_status) -> str:
    """The registry id that lets an xfail-marked witness stand for a slow
    test, or ``""``: the one exception of the module docstring.

    ``witness`` is the mark on the witness, ``slow_items`` the slow-marked
    items of the slow test, ``status`` the registry's status of an id.  The
    id is one that the ``reason=`` of the witness's mark and of every slow
    item's mark all name; every one of those marks is strict and applies
    for certain (no undecided condition); and the entry is open.
    """
    marks = [witness] + [collection.xfail.get(item) for item in slow_items]
    if not slow_items or any(m is None or not m.strict or m.applies == "string" for m in marks):
        return ""
    shared = set.intersection(*(set(_ANOMALY_ID.findall(m.reason)) for m in marks))
    return min((i for i in shared if status(i) in OPEN_STATUSES), default="")


def cannot_fail(nodeid: str, slow_items: list[str], collection: Collection, *,
                outcomes=body_outcomes, status=registry_status,
                excused: dict[str, str] | None = None) -> str:
    """Why the default-lane item ``nodeid`` cannot fail there, or ``""``.

    ``slow_items`` are the slow-marked items of the test it would witness;
    ``outcomes`` reads what a test function's body reaches
    (:func:`body_outcomes`); ``excused`` is :data:`CONDITIONAL_BODY_SKIPS`.
    """
    function = nodeid.split("[", 1)[0]
    excused = CONDITIONAL_BODY_SKIPS if excused is None else excused
    reached = outcomes(function)
    if "unread" in reached:
        return ("its definition is not in its file, so nothing shows its body runs to an "
                "assertion; name a test written where it is collected")
    if "skip" in reached:
        return ("its body skips on every run: a pytest.skip() in the test, a helper it calls "
                "or a fixture it requests")
    if "xfail" in reached:
        return ("its body calls pytest.xfail(), or it is marked unittest.expectedFailure: it "
                "reports success when its assertions fail")
    if "unset" in reached:
        return (f"a skipif or xfail mark on it, its class or its module reads one of "
                f"{list(_UNSET_FOR_COLLECTION)}, which this guard's collection takes out of its "
                "environment and the sharded lanes set: the condition is decided here otherwise "
                "than in the lane; name a test whose marks do not read them")
    if "skipif" in reached and function not in excused:
        return ("its body skips under a condition (a pytest.skip() under an `if`, or a "
                "pytest.importorskip()) that the source does not decide; name a test that "
                "always runs, or add it to CONDITIONAL_BODY_SKIPS with the reason the "
                "condition is false in the default lane")
    mark = collection.xfail.get(nodeid)
    if mark is None:
        return ""
    if mark.applies == "string":
        return ("xfail-marked on a condition written as a string, which is not decided "
                "here; write the expression")
    if not mark.run:
        return ("xfail-marked with run=False: pytest never runs it, so it cannot fail, nor turn "
                "red on the push that fixes a defect it pins; name a test that runs")
    if shared_open_pin(mark, slow_items, collection, status):
        return ""
    pins = sorted(set(_ANOMALY_ID.findall(mark.reason)))
    return ("xfail-marked: it reports success when its assertions fail.  An xfail pin is a "
            "witness only of a slow test that is itself a strict xfail pin of the same open "
            f"registry entry (this mark: strict={mark.strict}, names {pins or 'no entry'}); "
            "name a test that passes and checks the property, or write one")


@cache
def _by_file(items: frozenset[str]) -> dict[str, tuple[str, ...]]:
    """``{file: its items}``: a witness is looked for among its own file's
    items, not among the eight thousand the collection holds."""
    out: dict[str, list[str]] = {}
    for nodeid in sorted(items):
        out.setdefault(nodeid.split("::", 1)[0], []).append(nodeid)
    return {rel: tuple(ids) for rel, ids in out.items()}


def witness_problems(rel: str, comments: list[tuple[int, str]], collection: Collection,
                     slow_items: list[str] | None = None, **readers) -> list[str]:
    """What is wrong with the ``# Per push:`` comments at one slow mark.

    ``slow_items`` are the slow-marked items of the test the comments are
    written at; ``readers`` are :func:`cannot_fail`'s.
    """
    problems = []
    slow_items = list(slow_items or ())
    for lineno, text in comments:
        ids, bare = witness_ids(text)
        if not ids:
            problems.append(f"{rel}:{lineno}: names no node id: {text[:100]!r}")
        for name in bare:
            problems.append(f"{rel}:{lineno}: {name!r} is not a node id; write "
                            "tests/<file>.py::[<Class>::]<test>")
        for target in ids:
            in_file = _by_file(collection.default_lane).get(target.split("::", 1)[0], ())
            named = [n for n in in_file if _covers(target, n)]
            runs = [n for n in named if not _lane_drops(n, collection)]
            if not runs:
                skipif = sorted({v for n, v in collection.skipif.items() if _covers(target, n)})
                dropped = _lane_drops(target, collection) or next(
                    (d for d in map(lambda n: _lane_drops(n, collection), named) if d), "")
                why = (dropped if dropped
                       else "slow-marked itself" if any(_covers(target, n) for n in collection.slow)
                       else "skip-marked" if any(_covers(target, n) for n in collection.skipped)
                       else "skipif-marked on a condition written as a string, which is not "
                            "decided here; write the expression" if skipif == ["string"]
                       else "skipif-marked, and the condition is true in this lane" if skipif
                       else "pytest collects no such test")
                problems.append(f"{rel}:{lineno}: {target} does not run in the default lane ({why})")
                continue
            # One item that can fail is a witness: a test named without its
            # parameters stands for those that run and are expected to pass.
            whys = [cannot_fail(n, slow_items, collection, **readers) for n in runs]
            if all(whys):
                problems.append(f"{rel}:{lineno}: {target} cannot fail in the default lane ({whys[0]})")
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


def test_every_named_per_push_witness_runs_in_the_default_lane_and_can_fail_there(collection):
    problems = []
    for function, items in _slow_functions(collection).items():
        _, comments = _comments_at(function)
        problems += witness_problems(function.split("::", 1)[0], comments, collection, items)
    assert not problems, (
        "`# Per push:` comments at slow marks that do not name a test the default lane runs "
        "and that can fail there:\n  " + "\n  ".join(problems))


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
    # A parameter id is read to the bracket closing it: spaces, and brackets
    # nested in pairs, are part of it (it was cut at the first space, and the
    # comment then named the test's other parameters).
    assert witness_ids("tests/a/test_x.py::test_w[the case] and tests/a/test_x.py::test_v[a[1]-b], "
                       "then tests/a/test_x.py::test_u[] (empty)") == (
        ["tests/a/test_x.py::test_w[the case]", "tests/a/test_x.py::test_v[a[1]-b]",
         "tests/a/test_x.py::test_u[]"], [])
    # A bracket after a space is prose; an unbalanced one ends the id, which
    # then names no item and is refused.
    assert witness_ids("tests/a/test_x.py::test_w [the fast case]") == (["tests/a/test_x.py::test_w"], [])
    assert witness_ids("tests/a/test_x.py::test_w[a]b] x")[0] == ["tests/a/test_x.py::test_w[a]"]
    assert _NODE_ID.fullmatch("tests/a/test_x.py::TestA::test_w[x y-1.5e-3]")


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


def _row(nodeid: str, *, slow=False, skip=False, skipif="", xfail=None) -> list:
    """One row of the collection's dump, as :data:`_COLLECT` writes it."""
    return [nodeid, slow, skip, skipif, xfail]


def _reads_nothing(_function: str) -> tuple[str, ...]:
    return ()


def test_a_witness_resolves_only_to_an_item_the_default_lane_runs():
    c = partition([_row("tests/a.py::test_slow", slow=True), _row("tests/a.py::test_mixed[1]", slow=True),
                   _row("tests/a.py::TestA::test_fast"),
                   _row("tests/a.py::test_mixed[0]"),
                   _row("tests/a.py::test_skipped", skip=True)])

    def problems(text):
        return witness_problems("tests/b.py", [(1, text)], c, ["tests/b.py::test_s"],
                                outcomes=_reads_nothing)

    assert "skip-marked" in problems("tests/a.py::test_skipped")[0]
    assert problems("tests/a.py::TestA::test_fast and tests/a.py::test_mixed (its fast case)") == []
    assert "slow-marked itself" in problems("tests/a.py::test_slow")[0]
    assert "collects no such test" in problems("tests/a.py::test_fast")[0]
    bare = problems("tests/a.py::test_mixed and TestA::test_fast")
    assert len(bare) == 2 and all("is not a node id" in b for b in bare), bare
    assert "names no node id" in problems("below")[0]


def test_a_witness_in_a_file_the_per_push_lane_never_collects_is_refused(tmp_path):
    """The lane runs ``pytest tests/``; the collection here names each file.
    What only a run naming the file collects is no witness: a path the
    lane ignores, one recorded as never run, a file whose name pytest's
    ``python_files`` does not match, an item it deselects."""
    args = ["tests/", "-v", "--ignore=tests/viz", "--deselect", "tests/core/test_d.py::test_gone"]
    never = {"tests/api/test_3d.py": "tests/api/test_3d.py"}.get
    files = ["tests/core/test_ok.py", "tests/core/ok_test.py", "tests/viz/test_v.py",
             "tests/core/zz_cases.py", "tests/api/test_3d.py", "tests/core/test_d.py",
             "docs/test_elsewhere.py"]
    patterns = ["test_*.py", "*_test.py"]
    out = lane_exclusions(files, args, patterns, never, root=tmp_path, reporting={"-v"})
    assert set(out) == {"tests/viz/test_v.py", "tests/core/zz_cases.py", "tests/api/test_3d.py",
                        "docs/test_elsewhere.py", "tests/core/test_d.py::test_gone"}, out
    assert "--ignore=tests/viz" in out["tests/viz/test_v.py"]
    assert "python_files" in out["tests/core/zz_cases.py"]
    assert "PATHS_CI_NEVER_RUNS" in out["tests/api/test_3d.py"]
    assert "under none of them" in out["docs/test_elsewhere.py"]
    assert "--deselect" in out["tests/core/test_d.py::test_gone"]
    # The patterns are the project's: a file one matches is collected, and a
    # file the command names is collected whatever its name.
    assert "tests/core/zz_cases.py" not in lane_exclusions(files, args, ["zz_*.py"], never,
                                                          root=tmp_path, reporting={"-v"})
    assert "tests/core/zz_cases.py" not in lane_exclusions(
        files, [*args, "tests/core/zz_cases.py::test_x"], patterns, never, root=tmp_path,
        reporting={"-v"})
    # A selection the reader does not model is refused, not guessed at.
    for selecting in (["-k", "fast"], ["-m", "fast"], ["--ignore-glob=*viz*"], ["-x"]):
        with pytest.raises(AssertionError, match="neither a path"):
            lane_selection(["tests/", *selecting], reporting=set())
    # And a witness the lane drops is refused, with the reason.
    c = dataclasses.replace(partition([
        _row("tests/viz/test_v.py::TestV::test_w"), _row("tests/core/zz_cases.py::test_w"),
        _row("tests/core/test_d.py::test_gone[0]"), _row("tests/core/test_d.py::test_gone[1]"),
        _row("tests/core/test_d.py::test_mixed[0]"), _row("tests/core/test_ok.py::test_w"),
        _row("tests/b.py::test_s", slow=True)]),
        not_collected={**out, "tests/core/test_d.py::test_mixed[1]": "deselected"})

    def problems(target):
        return witness_problems("tests/b.py", [(1, target)], c, ["tests/b.py::test_s"],
                                outcomes=_reads_nothing)

    assert "--ignore=tests/viz" in problems("tests/viz/test_v.py::TestV::test_w")[0]
    assert "python_files" in problems("tests/core/zz_cases.py::test_w")[0]
    assert "--deselect" in problems("tests/core/test_d.py::test_gone")[0]
    assert "--deselect" in problems("tests/core/test_d.py::test_gone[1]")[0]
    assert problems("tests/core/test_d.py::test_mixed") == []     # one case runs
    assert problems("tests/core/test_ok.py::test_w") == []


def test_the_lane_reader_reads_the_per_push_lane_of_this_tree(pytestconfig):
    """On the real workflow, configuration and record: ``tests/viz`` is
    ignored, a file the lane collects is not dropped, and a file named
    outside ``python_files`` is."""
    args = per_push_lane_args()
    paths, ignores, _deselects = lane_selection(args)
    assert paths == ["tests"] and "tests/viz" in ignores, (paths, ignores)
    patterns = pytestconfig.getini("python_files")
    out = lane_exclusions(["tests/viz/test_serialization.py", "tests/core/test_transforms.py",
                           "tests/core/inspection_graphs.py"], args, patterns,
                          _anomaly_gate()._ci_never_runs)  # noqa: SLF001
    assert set(out) == {"tests/viz/test_serialization.py", "tests/core/inspection_graphs.py"}, out


_PIN = ["MADD-ANO-7: open, pinned", True, "always"]


def test_a_witness_a_mark_skips_or_expects_to_fail_is_refused():
    """The marks, as pytest holds them for the collected items."""
    c = partition([
        _row("tests/a.py::test_pin", xfail=_PIN),
        _row("tests/a.py::test_loose", xfail=["flaky", False, "always"]),
        _row("tests/a.py::test_on_this_lane", xfail=["jax 9 only", True, "true"]),
        _row("tests/a.py::test_cond_text", xfail=["x", True, "string"]),
        _row("tests/a.py::test_never", skipif="true"),
        _row("tests/a.py::test_no_condition", skipif="always"),
        _row("tests/a.py::test_text", skipif="string"),
        # A condition that is false in this lane leaves no trace: the item runs.
        _row("tests/a.py::test_four_devices"),
        _row("tests/a.py::test_cases[pinned]", xfail=_PIN), _row("tests/a.py::test_cases[ok]"),
        _row("tests/a.py::test_all_pinned[a]", xfail=_PIN),
        _row("tests/a.py::test_all_pinned[b]", xfail=_PIN),
        _row("tests/b.py::test_s", slow=True)])

    def problems(target):
        return witness_problems("tests/b.py", [(1, target)], c, ["tests/b.py::test_s"],
                                outcomes=_reads_nothing, status=lambda _id: "open")

    # A slow test that is expected to pass, with an xfail witness: the audited case.
    for target in ("test_pin", "test_loose", "test_on_this_lane", "test_all_pinned",
                   "test_all_pinned[a]", "test_cases[pinned]"):
        (problem,) = problems(f"tests/a.py::{target}")
        assert "cannot fail in the default lane (xfail-marked: it reports success" in problem, problem
    assert "written as a string" in problems("tests/a.py::test_cond_text")[0]
    for target in ("test_never", "test_no_condition"):
        (problem,) = problems(f"tests/a.py::{target}")
        assert "does not run in the default lane (skipif-marked, and the condition is true" in problem
    assert "skipif-marked on a condition written as a string" in problems("tests/a.py::test_text")[0]
    assert problems("tests/a.py::test_four_devices") == []
    # One parametrisation that is expected to pass is a witness.
    assert problems("tests/a.py::test_cases") == [] and problems("tests/a.py::test_cases[ok]") == []


def test_a_witness_whose_own_body_skips_or_expects_to_fail_is_refused():
    c = partition([_row("tests/a.py::test_w[0]"), _row("tests/a.py::test_w[1]"),
                   _row("tests/b.py::test_s", slow=True)])

    def problems(reached, excused=None):
        return witness_problems("tests/b.py", [(1, "tests/a.py::test_w")], c, ["tests/b.py::test_s"],
                                outcomes=lambda function: reached, excused=excused or {})

    assert problems(()) == []
    assert "its body skips on every run" in problems(("skip",))[0]
    assert "its body calls pytest.xfail()" in problems(("xfail",))[0]
    assert "its definition is not in its file" in problems(("unread",))[0]
    assert "MADDENING_TEST_SHARD" in problems(("unset",))[0]
    # The table excuses a condition in the body, never one the lane decides otherwise.
    assert problems(("unset",), {"tests/a.py::test_w": "reason"}) == problems(("unset",))
    conditional = problems(("skipif",))
    assert len(conditional) == 1 and "CONDITIONAL_BODY_SKIPS" in conditional[0], conditional
    # Named with its reason, by the function (its parameters share one body).
    assert problems(("skipif",), {"tests/a.py::test_w": "four devices, which the conftest provides"}) == []
    assert problems(("skipif",), {"tests/a.py::test_other": "another test"}) == conditional
    # The table excuses a condition, never a skip every run reaches.
    assert "its body skips on every run" in problems(("skipif", "skip"),
                                                      {"tests/a.py::test_w": "reason"})[0]


def test_an_xfail_pin_is_a_witness_only_of_a_slow_pin_of_the_same_open_anomaly():
    """The one exception, at each of its edges."""
    status = {"MADD-ANO-7": "open", "MADD-ANO-8": "partially_resolved", "MADD-ANO-9": "resolved",
              "MADD-ANO-70": "open"}

    def accepted(witness, *slow):
        rows = [_row("tests/a.py::test_w", xfail=witness)]
        rows += [_row(f"tests/b.py::test_s[{i}]", slow=True, xfail=mark) for i, mark in enumerate(slow)]
        c = partition(rows)
        problems = witness_problems("tests/b.py", [(1, "tests/a.py::test_w")], c, sorted(c.slow),
                                    outcomes=_reads_nothing, status=lambda i: status.get(i, ""))
        assert len(problems) <= 1
        return not problems

    pin7 = ["MADD-ANO-7: the bound reads low", True, "always"]
    assert accepted(pin7, pin7)
    assert accepted(pin7, ["through another kind (MADD-ANO-7)", True, "always"], pin7)
    assert accepted(["MADD-ANO-8 and MADD-ANO-9", True, "always"], ["MADD-ANO-8", True, "true"])
    # A slow test that is expected to pass, in whole or in one parametrisation.
    assert not accepted(pin7, None)
    assert not accepted(pin7, pin7, None)
    # Another anomaly; an id that only starts the same; no id at all.
    assert not accepted(pin7, ["MADD-ANO-8: another defect", True, "always"])
    assert not accepted(pin7, ["MADD-ANO-70: another defect", True, "always"])
    assert not accepted(["the bound reads low", True, "always"], ["the bound reads low", True, "always"])
    # A resolved entry, and one the registry does not hold.
    assert not accepted(["MADD-ANO-9", True, "always"], ["MADD-ANO-9", True, "always"])
    assert not accepted(["MADD-ANO-6", True, "always"], ["MADD-ANO-6", True, "always"])
    # Not a pin: either side not strict, or on a condition that is not decided.
    assert not accepted(["MADD-ANO-7", False, "always"], pin7)
    assert not accepted(pin7, ["MADD-ANO-7", False, "always"])
    assert not accepted(["MADD-ANO-7", True, "string"], pin7)
    assert not accepted(pin7, ["MADD-ANO-7", True, "string"])
    # A witness pytest never runs (run=False) cannot turn red when the defect goes.
    assert not accepted(["MADD-ANO-7: pinned, never run", True, "always", False], pin7)
    assert accepted(["MADD-ANO-7: pinned", True, "always", True], pin7)
    # No slow items handed in (a caller that forgot them) excuses nothing.
    c = partition([_row("tests/a.py::test_w", xfail=pin7)])
    assert witness_problems("tests/b.py", [(1, "tests/a.py::test_w")], c, outcomes=_reads_nothing,
                            status=lambda i: "open")
    assert set(OPEN_STATUSES) == {"open", "partially_resolved"}


_BODIES = '''
import os
import unittest
import unittest as ut
import pytest
from pytest import skip as leave
from unittest import SkipTest, skip as quietly

SHARDED = "MADDENING_TEST_SHARD" in os.environ


def _need_devices():
    import jax
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")


def _never():
    leave("not yet")


@pytest.fixture
def gone():
    pytest.skip("the fixture skips")


def test_runs():
    assert 1 + 1 == 2


def test_skips_first():
    pytest.skip("seeded")
    assert False


def test_skips_through_a_helper():
    _never()


def test_skips_through_a_fixture(gone):
    assert gone


def test_skips_under_a_condition():
    _need_devices()
    assert True


def test_needs_a_package():
    pytest.importorskip("no_such_package")


def test_expects_to_fail():
    if 1 + 1 == 2:
        pytest.xfail("known")


def test_skips_after_it_returned():
    return
    pytest.skip("dead code")


class TestA:
    def test_method(self):
        pytest.skip("seeded")

    def test_plain(self):
        assert True


@unittest.skip("on a plain function")
def test_unittest_skip():
    assert False


@quietly("imported under another name")
def test_unittest_skip_imported():
    assert False


@ut.skipIf(1 + 1 == 2, "a condition")
def test_unittest_skipif():
    assert False


@unittest.expectedFailure
def test_unittest_expected_failure():
    assert False


@unittest.skip("on the class")
class TestSkippedCase(unittest.TestCase):
    def test_in_it(self):
        assert False


class TestCaseMethods(unittest.TestCase):
    def setUp(self):
        self.skipTest("unittest's own skip, through self: not seen")

    @unittest.skip("on the method")
    def test_skipped(self):
        assert False


def test_raises_skiptest():
    raise unittest.SkipTest("in the body")


def test_raises_skiptest_imported_on_a_condition():
    if len(os.sep) == 1:
        raise SkipTest("under a condition")
    assert False


def test_redefined():
    assert False


def test_redefined():  # noqa: F811  (the one pytest collects)
    pytest.skip("the second definition")


def test_redefined_back():
    pytest.skip("the first definition")


def test_redefined_back():  # noqa: F811
    assert True


def test_rebound():
    assert True


test_rebound = None


@pytest.mark.skipif("MADDENING_TEST_SHARD" in os.environ, reason="sharded lanes only")
def test_skipif_on_the_shard():
    assert False


@pytest.mark.skipif(SHARDED, reason="through a module name")
def test_skipif_on_the_shard_by_name():
    assert False


def _sharded():
    return bool(os.environ.get("MADDENING_TEST_SHARD"))


@pytest.mark.xfail(_sharded(), reason="through a module function")
def test_xfail_on_the_shard_by_function():
    assert False


@pytest.mark.skipif(os.name == "nt", reason="another variable")
def test_skipif_on_another_condition():
    assert True
'''

_MODULE_WIDE = '''
import os
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("MADDENING_TEST_JAX_TIMING") == "1", reason="x")


def setup_module(module):
    pass


def setup_function(function):
    pytest.skip("xunit setup of every module-level test")


def test_plain():
    assert True


class TestC:
    def setup_method(self, method):
        if os.sep == "/":
            pytest.skip("xunit setup of the class's tests, on a condition")

    def test_in_class(self):
        assert True
'''


def test_the_body_reader_finds_what_it_is_for():
    tree = ast.parse(_BODIES)
    assert outcomes_in(tree, "test_runs") == ()
    assert outcomes_in(tree, "test_skips_first") == ("skip",)
    assert outcomes_in(tree, "test_skips_through_a_helper") == ("skip",)
    assert outcomes_in(tree, "test_skips_through_a_fixture") == ("skip",)
    assert outcomes_in(tree, "test_skips_under_a_condition") == ("skipif",)
    assert outcomes_in(tree, "test_needs_a_package") == ("skipif",)
    assert outcomes_in(tree, "test_expects_to_fail") == ("xfail",)
    assert outcomes_in(tree, "test_skips_after_it_returned") == ()
    assert outcomes_in(tree, "TestA::test_method") == ("skip",)
    assert outcomes_in(tree, "TestA::test_plain") == ()
    # Not written in the module: nothing was read, and that is said.
    assert outcomes_in(tree, "test_generated_elsewhere") == ("unread",)
    assert outcomes_in(tree, "TestA::test_inherited") == ("unread",)
    assert outcomes_in(tree, "TestA") == ("unread",)
    # unittest's skips and expected failure, under any import of them.
    assert outcomes_in(tree, "test_unittest_skip") == ("skip",)
    assert outcomes_in(tree, "test_unittest_skip_imported") == ("skip",)
    assert outcomes_in(tree, "test_unittest_skipif") == ("skipif",)
    assert outcomes_in(tree, "test_unittest_expected_failure") == ("xfail",)
    assert outcomes_in(tree, "TestSkippedCase::test_in_it") == ("skip",)
    assert outcomes_in(tree, "TestCaseMethods::test_skipped") == ("skip",)
    assert outcomes_in(tree, "test_raises_skiptest") == ("skip",)
    assert outcomes_in(tree, "test_raises_skiptest_imported_on_a_condition") == ("skipif",)
    # The definition pytest collects is the last one; a name rebound to
    # something else is not read.
    assert outcomes_in(tree, "test_redefined") == ("skip",)
    assert outcomes_in(tree, "test_redefined_back") == ()
    assert outcomes_in(tree, "test_rebound") == ("unread",)
    # A mark whose condition reads the shard variable, spelled out, through a
    # module-level name or a module-level function.
    assert outcomes_in(tree, "test_skipif_on_the_shard") == ("unset",)
    assert outcomes_in(tree, "test_skipif_on_the_shard_by_name") == ("unset",)
    assert outcomes_in(tree, "test_xfail_on_the_shard_by_function") == ("unset",)
    assert outcomes_in(tree, "test_skipif_on_another_condition") == ()
    # xunit setups, and a module mark reading an unset variable.
    wide = ast.parse(_MODULE_WIDE)
    assert outcomes_in(wide, "test_plain") == ("skip", "unset")
    assert outcomes_in(wide, "TestC::test_in_class") == ("skipif", "unset")


_MARKS = '''
import sys
import pytest

DEVICES = 16


@pytest.mark.slow
def test_slow():
    pass


@pytest.mark.skip(reason="x")
def test_skip():
    pass


@pytest.mark.skipif(DEVICES < 4, reason="needs 4 devices")
def test_enough_devices():
    pass


@pytest.mark.skipif(DEVICES < 32, reason="needs 32 devices")
def test_too_few_devices():
    pass


@pytest.mark.skipif(False, DEVICES < 32, reason="any condition true skips")
def test_second_condition():
    pass


@pytest.mark.skipif(reason="no condition: pytest skips")
def test_skipif_without_a_condition():
    pass


@pytest.mark.skipif("DEVICES < 32", reason="a string")
def test_skipif_string():
    pass


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="MADD-ANO-7: pinned")
def test_pin():
    assert False


@pytest.mark.xfail(reason="loose")
def test_loose():
    pass


@pytest.mark.xfail(run=False, reason="never run")
def test_not_run():
    pass


@pytest.mark.xfail(strict=True, run=False, reason="MADD-ANO-7: pinned, never run")
def test_pin_not_run():
    assert False


@pytest.mark.xfail(DEVICES < 4, strict=True, reason="a false condition: the test can fail")
def test_xfail_on_a_false_condition():
    pass


@pytest.mark.xfail(condition=sys.version_info >= (3,), strict=True, reason="true here")
def test_xfail_on_a_true_condition():
    pass


@pytest.mark.parametrize("n", [0, pytest.param(1, marks=pytest.mark.xfail(strict=True, reason="p"))])
def test_cases(n):
    pass


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="MADD-ANO-7: the slow pin")
def test_slow_pin():
    assert False


@pytest.mark.xfail(strict=True, reason="on the class")
class TestMarked:
    def test_in_class(self):
        pass

    @pytest.mark.xfail(strict=False, reason="the closer mark is the one pytest takes")
    def test_closer(self):
        pass
'''


def test_the_collection_holds_the_marks_as_pytest_decides_them(tmp_path):
    """The dump against a file of every form: what pytest holds for each
    collected item, conditions evaluated, never the decorators' text."""
    (tmp_path / "test_marks.py").write_text(_MARKS, encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: slow\n", encoding="utf-8")
    rows = collected(["test_marks.py"], tmp_path, tmp_path / "items.json")
    c = partition(rows)
    name = "test_marks.py::{}".format
    assert c.slow == {name("test_slow"), name("test_slow_pin")}
    assert c.skipped == {name("test_skip")}
    assert c.skipif == {name("test_too_few_devices"): "true", name("test_second_condition"): "true",
                        name("test_skipif_without_a_condition"): "always",
                        name("test_skipif_string"): "string"}
    assert c.default_lane == {name(t) for t in (
        "test_enough_devices", "test_pin", "test_loose", "test_not_run", "test_pin_not_run",
        "test_xfail_on_a_false_condition", "test_xfail_on_a_true_condition", "test_cases[0]",
        "test_cases[1]", "TestMarked::test_in_class", "TestMarked::test_closer")}
    assert c.xfail == {
        name("test_pin"): Xfail("MADD-ANO-7: pinned", True, "always"),
        name("test_loose"): Xfail("loose", False, "always"),
        name("test_not_run"): Xfail("never run", False, "always", run=False),
        name("test_pin_not_run"): Xfail("MADD-ANO-7: pinned, never run", True, "always", run=False),
        name("test_xfail_on_a_true_condition"): Xfail("true here", True, "true"),
        name("test_cases[1]"): Xfail("p", True, "always"),
        name("test_slow_pin"): Xfail("MADD-ANO-7: the slow pin", True, "always"),
        name("TestMarked::test_in_class"): Xfail("on the class", True, "always"),
        name("TestMarked::test_closer"): Xfail("the closer mark is the one pytest takes", False,
                                               "always")}
    # And the rule on what was collected: the pin stands for the slow pin of
    # its anomaly, and for no slow test that is expected to pass.
    readers = dict(outcomes=_reads_nothing, status=lambda i: "open")
    pin = [(1, "tests/test_marks.py::test_pin")]
    under_tests = partition([[f"tests/{row[0]}", *row[1:]] for row in rows])
    assert witness_problems("tests/x.py", pin, under_tests, ["tests/test_marks.py::test_slow_pin"],
                            **readers) == []
    assert "xfail-marked" in witness_problems("tests/x.py", pin, under_tests,
                                              ["tests/test_marks.py::test_slow"], **readers)[0]
    # A strict pin of the same open entry that never runs is no witness.
    (never_run,) = witness_problems("tests/x.py", [(1, "tests/test_marks.py::test_pin_not_run")],
                                    under_tests, ["tests/test_marks.py::test_slow_pin"], **readers)
    assert "run=False: pytest never runs it" in never_run, never_run
    for target in ("test_xfail_on_a_false_condition", "test_enough_devices", "test_cases"):
        assert witness_problems("tests/x.py", [(1, f"tests/test_marks.py::{target}")], under_tests,
                                ["tests/test_marks.py::test_slow"], **readers) == [], target
    # A mark that does not say whether it is strict takes the ini's default, as pytest does.
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nxfail_strict = true\nmarkers =\n    slow: slow\n", encoding="utf-8")
    strict = partition(collected(["test_marks.py"], tmp_path, tmp_path / "strict.json"))
    assert strict.xfail[name("test_loose")].strict
    assert not strict.xfail[name("TestMarked::test_closer")].strict      # it says strict=False


def _witness_functions() -> set[str]:
    """Every test function a ``# Per push:`` comment names, parameters dropped."""
    out = set()
    for rel in _framework_files():
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        if "Per push:" in text:
            lines = text.splitlines()
            for _, comment in per_push_comments(lines, 0, len(lines)):
                out |= {i.split("[", 1)[0] for i in witness_ids(comment)[0]}
    return out


def test_every_excused_conditional_skip_is_a_witness_that_still_skips_under_a_condition():
    """An entry of :data:`CONDITIONAL_BODY_SKIPS` cannot outlive what it is
    for, and excuses a condition only: never a skip every run reaches."""
    named = _witness_functions()
    problems = []
    for function, reason in CONDITIONAL_BODY_SKIPS.items():
        if not reason.strip():
            problems.append(f"{function}: gives no reason")
        if not (REPO_ROOT / function.split("::", 1)[0]).is_file():
            problems.append(f"{function}: its file does not exist")
        elif function not in named and not any(_covers(n, function) for n in named):
            problems.append(f"{function}: no `# Per push:` comment names it (retire the entry)")
        elif set(body_outcomes(function)) != {"skipif"}:
            problems.append(f"{function}: its body reaches {sorted(set(body_outcomes(function)))}, "
                            "not a skip under a condition alone (retire the entry)")
    assert not problems, "CONDITIONAL_BODY_SKIPS:\n  " + "\n  ".join(problems)


def test_the_registry_reader_tells_an_open_entry_from_a_resolved_one():
    """What the exception stands on: the statuses are the registry's own."""
    statuses = _registry_statuses()
    assert len(statuses) > 200 and set(statuses.values()) >= {"open", "resolved"}, len(statuses)
    assert all(_ANOMALY_ID.fullmatch(i) for i in statuses), sorted(statuses)[:3]
    assert registry_status("MADD-ANO-001") == "resolved" and registry_status("MADD-ANO-0") == ""


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
