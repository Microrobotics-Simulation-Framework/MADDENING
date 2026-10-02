"""The coupling claims inventory is well-formed, and every row's tests can fail.

``docs/validation/coupling_claims.yaml`` lists every documented claim about
coupling with its conditions, its oracle, the tests that pin it and a
status.  An inventory nothing checks rots the way the registry did before
its gates failed (``testing_standards.md``, "The compliance gates now
fail"), so this module holds it to four rules:

* **every row is well-formed**: the required fields, a known status, an
  ``id`` of the form ``CPL-NNN`` used once, sources that name files in the
  repository, the extra field each status needs (``proposed_wording`` for
  ``ambiguous``, ``finding`` for ``failing``, ``reason`` for ``untested``)
  and no field the schema does not know -- it fails closed;
* **every cited test exists and runs on every push**: pytest collects it,
  it is not skip-marked, and if it is slow-marked it names a ``# Per
  push: <node id>`` witness at its mark (the slow-only rule's convention,
  ``tests/compliance/test_slow_only_rule.py``);
* **a failing row cites a strict xfail** whose reason starts with the
  row's id, so the row and the test move together when the fix lands;
* **a verified row cites no xfail**, strict or not: a claim the tree does
  not meet is not verified.  And every strict xfail whose reason names a
  row is cited by that row, so a test cannot outlive its row's status.

Which tests are slow, skipped or xfail comes from pytest itself, in one
``--collect-only`` subprocess over the files the inventory cites.  The
rules are pure functions over the rows and that collection, so the
self-tests at the end feed them broken rows and check each rule fires.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from tests.compliance.test_slow_only_rule import _NODE_ID, _comments_at, _covers, witness_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY = REPO_ROOT / "docs" / "validation" / "coupling_claims.yaml"

REQUIRED = ("id", "area", "claim", "sources", "conditions", "oracle", "tests", "status")
OPTIONAL = ("proposed_wording", "finding", "reason", "notes")
STATUSES = ("verified", "failing", "untested", "ambiguous")
#: The field each status needs beside the required ones.
NEEDS = {"ambiguous": "proposed_wording", "failing": "finding", "untested": "reason"}
_ID = re.compile(r"^CPL-\d{3}$")
#: A source is a repository path, then ``:line``, `` (section)`` or ``#anchor``.
_SOURCE_PATH = re.compile(r"^([\w./\-]+?\.(?:py|md|yaml|toml|json|txt))(?=$|[:#\s(])")
#: An xfail reason that names a row.
_REASON_ROW = re.compile(r"^(CPL-\d{3}):")


@dataclass(frozen=True)
class Item:
    """One collected test item, as far as the rules care."""

    nodeid: str
    slow: bool
    skip: bool
    xfail: bool
    strict: bool
    reason: str


def load_rows(path: Path = INVENTORY) -> list[dict]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("claims"), list):
        raise AssertionError(f"{path.name}: expected a mapping with a 'claims' list")
    return doc["claims"]


# --------------------------------------------------------------------------
# Rule 1: the rows themselves
# --------------------------------------------------------------------------
def row_problems(rows: list, repo_root: Path = REPO_ROOT) -> list[str]:
    """What is wrong with the rows, read alone."""
    problems = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        where = f"row {i}"
        if not isinstance(row, dict):
            problems.append(f"{where}: not a mapping")
            continue
        rid = row.get("id")
        where = f"{rid or where}"
        if not isinstance(rid, str) or not _ID.match(rid):
            problems.append(f"{where}: id {rid!r} is not CPL-NNN")
        elif rid in seen:
            problems.append(f"{where}: id used twice")
        else:
            seen.add(rid)
        unknown = sorted(set(row) - set(REQUIRED) - set(OPTIONAL))
        if unknown:
            problems.append(f"{where}: unknown field(s) {unknown}")
        for key in REQUIRED:
            if key not in row:
                problems.append(f"{where}: missing {key!r}")
        for key in ("area", "claim", "conditions", "oracle"):
            if key in row and not (isinstance(row[key], str) and row[key].strip()):
                problems.append(f"{where}: {key!r} must be non-empty text")
        status = row.get("status")
        if status not in STATUSES:
            problems.append(f"{where}: status {status!r} is not one of {STATUSES}")
        need = NEEDS.get(status)
        if need and not (isinstance(row.get(need), str) and row[need].strip()):
            problems.append(f"{where}: a {status} row needs {need!r}")
        sources = row.get("sources")
        if not (isinstance(sources, list) and sources):
            problems.append(f"{where}: 'sources' must be a non-empty list")
        else:
            for src in sources:
                m = _SOURCE_PATH.match(src) if isinstance(src, str) else None
                if m is None:
                    problems.append(f"{where}: source {src!r} names no repository file")
                elif not (repo_root / m.group(1)).is_file():
                    problems.append(f"{where}: source file {m.group(1)} does not exist")
        tests = row.get("tests")
        if not isinstance(tests, list):
            problems.append(f"{where}: 'tests' must be a list")
            continue
        if not tests and status != "untested":
            problems.append(f"{where}: cites no test (only an untested row may)")
        for t in tests:
            if not (isinstance(t, str) and _NODE_ID.fullmatch(t)):
                problems.append(f"{where}: {t!r} is not a node id tests/<file>.py::<test>")
    return problems


# --------------------------------------------------------------------------
# Asking pytest: what is collected, slow, skipped and xfail
# --------------------------------------------------------------------------
_COLLECT = r"""
import json, os, sys
import pytest

class Dump:
    items = []

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        default_strict = bool(config.getini("xfail_strict"))
        out = []
        for i in items:
            xf = i.get_closest_marker("xfail")
            out.append([i.nodeid, i.get_closest_marker("slow") is not None,
                        i.get_closest_marker("skip") is not None, xf is not None,
                        bool(xf.kwargs.get("strict", default_strict)) if xf else False,
                        str(xf.kwargs.get("reason", "")) if xf else ""])
        Dump.items = out

code = pytest.main(sys.argv[1:], plugins=[Dump()])
with open(os.environ["COUPLING_CLAIMS_DUMP"], "w") as f:
    json.dump(Dump.items, f)
sys.exit(code)
"""


def cited_files(rows: list) -> list[str]:
    return sorted({t.split("::", 1)[0] for row in rows if isinstance(row, dict)
                   for t in (row.get("tests") or []) if isinstance(t, str)})


@pytest.fixture(scope="module")
def collection(tmp_path_factory) -> list[Item]:
    """Every item of every cited file, collected once with ``-m "slow or not slow"``."""
    files = [f for f in cited_files(load_rows()) if (REPO_ROOT / f).is_file()]
    dump = tmp_path_factory.mktemp("coupling_claims") / "items.json"
    env = {k: v for k, v in os.environ.items()
           if k not in ("MADDENING_TEST_SHARD", "MADDENING_TEST_JAX_TIMING", "PYTEST_ADDOPTS")}
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", JAX_PLATFORMS="cpu",
               COUPLING_CLAIMS_DUMP=str(dump))
    proc = subprocess.run(
        [sys.executable, "-c", _COLLECT, "--collect-only", "-q", "-p", "no:cacheprovider",
         "-m", "slow or not slow", *files],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return [Item(*row) for row in json.loads(dump.read_text())]


# --------------------------------------------------------------------------
# Rule 2: the cited tests exist and run on every push
# --------------------------------------------------------------------------
def _per_push_witness(function: str) -> bool:
    try:
        _line, comments = _comments_at(function)
    except (AssertionError, FileNotFoundError, SyntaxError):
        return False
    return any(witness_ids(text)[0] for _, text in comments)


def collection_problems(rows: list, items: list[Item], witness=_per_push_witness) -> list[str]:
    """Cited tests pytest does not collect, that are skipped, or slow without a witness."""
    problems = []
    for row in rows:
        for target in row.get("tests") or []:
            covered = [it for it in items if _covers(target, it.nodeid)]
            if not covered:
                problems.append(f"{row.get('id')}: {target} -- pytest collects no such test")
                continue
            if all(it.skip for it in covered):
                problems.append(f"{row.get('id')}: {target} -- skip-marked, it never runs")
            # A parametrised test with some items in the default lane runs on
            # every push; one whose every item is slow needs a witness.
            if all(it.slow or it.skip for it in covered) and any(it.slow for it in covered) \
                    and not witness(target.split("[", 1)[0]):
                problems.append(f"{row.get('id')}: {target} -- slow-marked with no "
                                "'# Per push: <node id>' witness at its mark")
    return problems


# --------------------------------------------------------------------------
# Rule 3 and 4: xfails and statuses agree
# --------------------------------------------------------------------------
def xfail_problems(rows: list, items: list[Item]) -> list[str]:
    """A failing row without its strict xfail; a verified row citing an xfail; an orphan xfail."""
    problems = []
    cited_by: dict[str, set[str]] = {}
    for row in rows:
        rid, status = row.get("id"), row.get("status")
        covered = [it for t in row.get("tests") or [] for it in items if _covers(t, it.nodeid)]
        for it in covered:
            cited_by.setdefault(it.nodeid, set()).add(rid)
        if status == "failing":
            mine = [it for it in covered if it.xfail and it.strict
                    and it.reason.startswith(f"{rid}:")]
            if not mine:
                problems.append(f"{rid}: failing, but cites no strict xfail whose reason "
                                f"starts with '{rid}:'")
        if status == "verified":
            for it in covered:
                if it.xfail:
                    problems.append(f"{rid}: verified, but cites the xfail {it.nodeid}")
    for it in items:
        m = _REASON_ROW.match(it.reason) if it.xfail else None
        if m and m.group(1) not in cited_by.get(it.nodeid, set()):
            problems.append(f"{it.nodeid}: its xfail names {m.group(1)}, which does not cite it")
        if m and not it.strict:
            problems.append(f"{it.nodeid}: an xfail naming {m.group(1)} must be strict")
    return problems


# --------------------------------------------------------------------------
# The rules on the inventory
# --------------------------------------------------------------------------
def test_every_row_is_well_formed():
    problems = row_problems(load_rows())
    assert not problems, "coupling_claims.yaml:\n  " + "\n  ".join(problems)


def test_every_cited_test_exists_and_runs_on_every_push(collection):
    problems = collection_problems(load_rows(), collection)
    assert not problems, "coupling_claims.yaml cites tests that cannot fail on a push:\n  " \
        + "\n  ".join(problems)


def test_statuses_and_xfails_agree(collection):
    problems = xfail_problems(load_rows(), collection)
    assert not problems, "coupling_claims.yaml statuses disagree with the tests:\n  " \
        + "\n  ".join(problems)


def test_the_inventory_counts_what_it_says():
    """Every status is a known one, and the file holds at least one row of each kind it reports."""
    rows = load_rows()
    assert len(rows) >= 100, f"only {len(rows)} rows: the inventory was truncated?"
    statuses = {row["status"] for row in rows}
    assert statuses <= set(STATUSES)


# --------------------------------------------------------------------------
# Self-tests: each rule fires on the defect it exists for
# --------------------------------------------------------------------------
_GOOD = {
    "id": "CPL-900", "area": "a", "claim": "c", "conditions": "k", "oracle": "o",
    "sources": ["src/maddening/core/coupling/group.py:1 (x)"],
    "tests": ["tests/core/test_x.py::test_a"], "status": "verified",
}


def _row(**changes):
    row = dict(_GOOD)
    for k, v in changes.items():
        if v is None:
            row.pop(k, None)
        else:
            row[k] = v
    return row


def _item(nodeid="tests/core/test_x.py::test_a", slow=False, skip=False, xfail=False,
          strict=False, reason=""):
    return Item(nodeid, slow, skip, xfail, strict, reason)


@pytest.mark.parametrize("row, fragment", [
    (_row(id="CPL-1"), "not CPL-NNN"),
    (_row(status="done"), "status"),
    (_row(status="ambiguous"), "needs 'proposed_wording'"),
    (_row(status="failing"), "needs 'finding'"),
    (_row(status="untested", tests=[]), "needs 'reason'"),
    (_row(tests=[]), "cites no test"),
    (_row(tests=["test_a"]), "is not a node id"),
    (_row(sources=[]), "non-empty list"),
    (_row(sources=["src/maddening/no_such_file.py:3"]), "does not exist"),
    (_row(sources=["somewhere in the docs"]), "names no repository file"),
    (_row(oracle=" "), "non-empty text"),
    (_row(claim=None), "missing 'claim'"),
    (_row(extra="x"), "unknown field"),
], ids=["id", "status", "ambiguous", "failing", "untested", "no-test", "bare-name",
        "no-source", "missing-file", "not-a-file", "blank", "missing-key", "unknown-key"])
def test_the_row_rule_fires_on_each_defect(row, fragment):
    problems = row_problems([row])
    assert any(fragment in p for p in problems), problems


def test_the_row_rule_passes_a_good_row_and_refuses_a_duplicate_id():
    assert row_problems([_GOOD]) == []
    assert any("used twice" in p for p in row_problems([_GOOD, dict(_GOOD)]))


def test_the_collection_rule_fires_on_each_defect():
    rows = [_GOOD]
    assert collection_problems(rows, [_item()]) == []
    assert any("collects no such test" in p
               for p in collection_problems(rows, [_item("tests/core/test_x.py::test_b")]))
    assert any("skip-marked" in p for p in collection_problems(rows, [_item(skip=True)]))
    no_witness = collection_problems(rows, [_item(slow=True)], witness=lambda f: False)
    assert any("no '# Per push" in p for p in no_witness)
    assert collection_problems(rows, [_item(slow=True)], witness=lambda f: True) == []
    # Some parameters in the default lane: the test runs on every push.
    mixed = [_item("tests/core/test_x.py::test_a[fast]"),
             _item("tests/core/test_x.py::test_a[slow]", slow=True)]
    assert collection_problems(rows, mixed, witness=lambda f: False) == []
    # A parametrised test is covered by its function's id.
    assert collection_problems(rows, [_item("tests/core/test_x.py::test_a[1]")]) == []


def test_the_xfail_rule_fires_on_each_defect():
    failing = _row(status="failing", finding="f")
    strict = _item(xfail=True, strict=True, reason="CPL-900: broken; pending fix")
    assert xfail_problems([failing], [strict]) == []
    assert any("cites no strict xfail" in p for p in xfail_problems([failing], [_item()]))
    assert any("cites no strict xfail" in p for p in xfail_problems(
        [failing], [_item(xfail=True, strict=False, reason="CPL-900: x")]))
    assert any("cites no strict xfail" in p for p in xfail_problems(
        [failing], [_item(xfail=True, strict=True, reason="CPL-901: another row")]))
    assert any("verified, but cites the xfail" in p for p in xfail_problems([_GOOD], [strict]))
    orphan = _item("tests/core/test_x.py::test_z", xfail=True, strict=True, reason="CPL-900: x")
    assert any("which does not cite it" in p for p in xfail_problems([failing], [strict, orphan]))
    loose = _item(xfail=True, strict=False, reason="CPL-900: x")
    assert any("must be strict" in p for p in xfail_problems([_row(status="ambiguous")], [loose]))
