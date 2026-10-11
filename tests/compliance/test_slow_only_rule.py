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

* **skipped by a mark**: ``skip``, or a ``skipif`` whose condition is true
  in this lane.  The condition is the value pytest itself holds for the
  collected item (:data:`_COLLECT`), never the decorator's text: the
  multi-device witnesses under ``tests/cloud/multigpu`` carry
  ``skipif(len(jax.devices()) < 4)``, which is false there because that
  directory's conftest provides the devices, and they stay witnesses.  A
  condition written as a string is evaluated by pytest only when the test
  is set up; it is not decided here and is refused: write the expression;
* **expected to fail**: an ``xfail`` mark that applies to the item in this
  lane -- strict or not, with ``raises=`` or without, ``run=False``, on the
  function, its class, the module or one parameter -- or a ``pytest.xfail()``
  call in its body.  Its conditions are decided as a ``skipif``'s are;
* **skipped by its own body**: a ``pytest.skip()`` every run reaches, in the
  test, in a module-level helper it calls or in a fixture of its module it
  requests.  ``scripts/check_anomalies.py`` already reads these for the
  registry's evidence, and its reader is the one used here
  (:func:`body_outcomes`).  A skip under a condition, and a
  ``pytest.importorskip()``, cannot be decided from the source: such a
  witness is refused unless :data:`CONDITIONAL_BODY_SKIPS` names it with
  the reason its condition is false in the default lane.

What this does not see, as the reader it borrows does not: a skip in a
fixture or helper of another module (a ``conftest.py`` included), in a
method reached through ``self``, behind a ``usefixtures`` mark or in a
collection hook.  Those are the run's to show, not the source's.

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
                    bool(mark.kwargs.get("strict", strict_default)), a]
    return None

class Dump:
    items = []

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        strict_default = bool(config.getini("xfail_strict"))
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
def collection(tmp_path_factory) -> Collection:
    """Every item in a framework file that can hold a slow mark, plus the
    files the comments and the table name, collected once with
    ``-m "slow or not slow"``."""
    paths = sorted({rel for rel in _framework_files()
                    if may_hold_a_slow_mark((REPO_ROOT / rel).read_text(encoding="utf-8"))}
                   | {f for f in _files_named() if (REPO_ROOT / f).is_file()})
    return partition(collected(paths, REPO_ROOT, tmp_path_factory.mktemp("slow_rule") / "items.json"))


def collected(paths: list[str], cwd: Path, dump: Path) -> list[list]:
    """The rows :data:`_COLLECT` writes for ``paths``: one pytest
    ``--collect-only`` over them with ``-m "slow or not slow"``, in a process
    of its own started in ``cwd``."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("MADDENING_TEST_SHARD", "MADDENING_TEST_JAX_TIMING", "PYTEST_ADDOPTS")}
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
    not written in the module (inherited from, or generated in, another):
    nothing can be said of a body that was not read."""
    gate = _anomaly_gate()
    scope, classes, node = tree.body, [], None
    for part in name.split("::"):
        node = next((n for n in scope if getattr(n, "name", None) == part and isinstance(
            n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if node is None:
            return ("unread",)
        if isinstance(node, ast.ClassDef):
            classes.append(node)
            scope = node.body
    if isinstance(node, ast.ClassDef):
        return ("unread",)
    return tuple(gate._test_function_kinds(node, classes, tree, gate._pytest_aliases(tree)))  # noqa: SLF001


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
        return "its body calls pytest.xfail(): it reports success when its assertions fail"
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
            runs = [n for n in in_file if _covers(target, n)]
            if not runs:
                skipif = sorted({v for n, v in collection.skipif.items() if _covers(target, n)})
                why = ("slow-marked itself" if any(_covers(target, n) for n in collection.slow)
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
    # No slow items handed in (a caller that forgot them) excuses nothing.
    c = partition([_row("tests/a.py::test_w", xfail=pin7)])
    assert witness_problems("tests/b.py", [(1, "tests/a.py::test_w")], c, outcomes=_reads_nothing,
                            status=lambda i: "open")
    assert set(OPEN_STATUSES) == {"open", "partially_resolved"}


_BODIES = '''
import pytest
from pytest import skip as leave


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
        "test_enough_devices", "test_pin", "test_loose", "test_not_run",
        "test_xfail_on_a_false_condition", "test_xfail_on_a_true_condition", "test_cases[0]",
        "test_cases[1]", "TestMarked::test_in_class", "TestMarked::test_closer")}
    assert c.xfail == {
        name("test_pin"): Xfail("MADD-ANO-7: pinned", True, "always"),
        name("test_loose"): Xfail("loose", False, "always"),
        name("test_not_run"): Xfail("never run", False, "always"),
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
    for target in ("test_xfail_on_a_false_condition", "test_enough_devices", "test_cases"):
        assert witness_problems("tests/x.py", [(1, f"tests/test_marks.py::{target}")], under_tests,
                                ["tests/test_marks.py::test_slow"], **readers) == [], target


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
