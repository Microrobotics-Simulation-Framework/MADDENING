"""Every claims inventory is well-formed, and every row's tests can fail.

``docs/validation/*_claims.yaml`` each list the documented claims of one
audit area -- coupling, sysid and FMU export, the REST surface -- with the
conditions a claim was stated for, its oracle, the tests that pin it and a
status.  An inventory nothing checks rots the way the registry did before
its gates failed (``testing_standards.md``, "The compliance gates now
fail"), so this module holds every one of them, found by its file name,
to the same rules:

* **the file is well-formed**: ``schema_version: 1``, a ``prefixes`` list
  of the id prefixes the file owns (two to six capital letters), a
  non-empty ``claims`` list and no other top-level key; no prefix is owned
  by two files, and every prefix a file declares names one of its rows;
* **every row is well-formed**: the required fields, a known status, an
  ``id`` of the form ``<PREFIX>-NNN`` with one of its own file's prefixes,
  used once, sources that name files in the repository, the extra field
  each status needs (``proposed_wording`` for ``ambiguous``, ``finding``
  for ``failing``, ``reason`` for ``untested``) and no field the schema
  does not know -- it fails closed;
* **every cited test exists and runs on every push**: pytest collects it,
  it is not skip-marked, and if it is slow-marked it names a ``# Per
  push: <node id>`` witness at its mark (the slow-only rule's convention,
  ``tests/compliance/test_slow_only_rule.py``);
* **a failing row cites a strict xfail** whose reason starts with the
  row's id, so the row and the test move together when the fix lands;
* **a verified row cites no xfail**, strict or not: a claim the tree does
  not meet is not verified.  And every xfail whose reason names a row is
  strict and cited by that row, and names a row some inventory holds, so a
  test cannot outlive its row's status.

Which tests are slow, skipped or xfail comes from pytest itself, in one
``--collect-only`` subprocess over the files every inventory cites.  The
rules are pure functions over the files, their rows and that collection,
so the self-tests at the end feed them broken files and rows and check
each rule fires.
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
VALIDATION = REPO_ROOT / "docs" / "validation"
#: Every inventory, by its name: a new ``*_claims.yaml`` is checked without
#: anyone remembering to add it here.
INVENTORIES = sorted(VALIDATION.glob("*_claims.yaml"))
#: The inventories the repository must hold; a guard over nothing is no guard.
EXPECTED = ("coupling_claims.yaml", "sysid_fmu_claims.yaml")

SCHEMA_VERSION = 1
TOP_LEVEL = ("schema_version", "prefixes", "claims")
REQUIRED = ("id", "area", "claim", "sources", "conditions", "oracle", "tests", "status")
OPTIONAL = ("proposed_wording", "finding", "reason", "notes")
STATUSES = ("verified", "failing", "untested", "ambiguous")
#: The field each status needs beside the required ones.
NEEDS = {"ambiguous": "proposed_wording", "failing": "finding", "untested": "reason"}
_PREFIX = re.compile(r"^[A-Z]{2,6}$")
_ID = re.compile(r"^([A-Z]{2,6})-\d{3}$")
#: A source is a repository path, then ``:line``, `` (section)`` or ``#anchor``.
_SOURCE_PATH = re.compile(r"^([\w./\-]+?\.(?:py|md|yaml|toml|json|txt|c|h|xml))(?=$|[:#\s(])")
#: An xfail reason that names a row.
_REASON_ROW = re.compile(r"^(([A-Z]{2,6})-\d{3}):")


@dataclass(frozen=True)
class Item:
    """One collected test item, as far as the rules care."""

    nodeid: str
    slow: bool
    skip: bool
    xfail: bool
    strict: bool
    reason: str


@dataclass(frozen=True)
class Inventory:
    """One ``*_claims.yaml``: its file name and its parsed document."""

    name: str
    doc: object

    def _get(self, key):
        value = self.doc.get(key) if isinstance(self.doc, dict) else None
        return value if isinstance(value, list) else []

    @property
    def rows(self) -> list:
        return self._get("claims")

    @property
    def prefixes(self) -> list:
        return self._get("prefixes")


def load(path: Path) -> Inventory:
    return Inventory(path.name, yaml.safe_load(path.read_text(encoding="utf-8")))


def load_all(paths=None) -> list[Inventory]:
    return [load(p) for p in (INVENTORIES if paths is None else paths)]


def every_row(inventories: list[Inventory]) -> list[dict]:
    return [row for inv in inventories for row in inv.rows if isinstance(row, dict)]


# --------------------------------------------------------------------------
# Rule 0: the files, and the prefixes they own
# --------------------------------------------------------------------------
def file_problems(inventories: list[Inventory]) -> list[str]:
    """What is wrong with the files' top level, alone and between files."""
    problems = []
    owner: dict[str, str] = {}
    for inv in inventories:
        doc = inv.doc
        if not isinstance(doc, dict):
            problems.append(f"{inv.name}: not a mapping")
            continue
        unknown = sorted(set(doc) - set(TOP_LEVEL))
        if unknown:
            problems.append(f"{inv.name}: unknown top-level key(s) {unknown}")
        if doc.get("schema_version") != SCHEMA_VERSION:
            problems.append(f"{inv.name}: schema_version {doc.get('schema_version')!r} "
                            f"is not {SCHEMA_VERSION}")
        if not (isinstance(doc.get("claims"), list) and doc["claims"]):
            problems.append(f"{inv.name}: 'claims' must be a non-empty list")
        prefixes = doc.get("prefixes")
        if not (isinstance(prefixes, list) and prefixes):
            problems.append(f"{inv.name}: 'prefixes' must be a non-empty list")
            continue
        used = {m.group(1) for row in every_row([inv])
                for m in [_ID.match(str(row.get("id", "")))] if m}
        for p in prefixes:
            if not (isinstance(p, str) and _PREFIX.match(p)):
                problems.append(f"{inv.name}: prefix {p!r} is not 2-6 capital letters")
                continue
            if owner.get(p, inv.name) != inv.name:
                problems.append(f"{inv.name}: prefix {p} is also declared by {owner[p]}")
            owner.setdefault(p, inv.name)
            if p not in used:
                problems.append(f"{inv.name}: prefix {p} names no row")
    return problems


# --------------------------------------------------------------------------
# Rule 1: the rows themselves
# --------------------------------------------------------------------------
def row_problems(rows: list, prefixes, repo_root: Path = REPO_ROOT) -> list[str]:
    """What is wrong with one file's rows, read alone; *prefixes* are the file's own."""
    problems = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        where = f"row {i}"
        if not isinstance(row, dict):
            problems.append(f"{where}: not a mapping")
            continue
        rid = row.get("id")
        where = f"{rid or where}"
        m = _ID.match(rid) if isinstance(rid, str) else None
        if m is None:
            problems.append(f"{where}: id {rid!r} is not <PREFIX>-NNN")
        elif m.group(1) not in prefixes:
            problems.append(f"{where}: prefix {m.group(1)} is not one of this file's "
                            f"{list(prefixes)}")
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
                sm = _SOURCE_PATH.match(src) if isinstance(src, str) else None
                if sm is None:
                    problems.append(f"{where}: source {src!r} names no repository file")
                elif not (repo_root / sm.group(1)).is_file():
                    problems.append(f"{where}: source file {sm.group(1)} does not exist")
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
with open(os.environ["CLAIMS_INVENTORY_DUMP"], "w") as f:
    json.dump(Dump.items, f)
sys.exit(code)
"""


def cited_files(rows: list) -> list[str]:
    return sorted({t.split("::", 1)[0] for row in rows if isinstance(row, dict)
                   for t in (row.get("tests") or []) if isinstance(t, str)})


@pytest.fixture(scope="module")
def collection(tmp_path_factory) -> list[Item]:
    """Every item of every cited file, collected once with ``-m "slow or not slow"``."""
    files = [f for f in cited_files(every_row(load_all())) if (REPO_ROOT / f).is_file()]
    dump = tmp_path_factory.mktemp("claims") / "items.json"
    env = {k: v for k, v in os.environ.items()
           if k not in ("MADDENING_TEST_SHARD", "MADDENING_TEST_JAX_TIMING", "PYTEST_ADDOPTS")}
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", JAX_PLATFORMS="cpu",
               CLAIMS_INVENTORY_DUMP=str(dump))
    proc = subprocess.run(
        [sys.executable, "-c", _COLLECT, "--collect-only", "-q", "-p", "no:cacheprovider",
         "-m", "slow or not slow", *files],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)
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
    """A failing row without its strict xfail; a verified row citing an xfail; an orphan xfail.

    *rows* are one file's.  An xfail whose reason names one of them must be
    strict and cited by it; an xfail naming another file's row is that
    file's to report, and one naming no row at all is
    :func:`unowned_xfail_problems`'.
    """
    problems = []
    ids = {row.get("id") for row in rows}
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
        if not m or m.group(1) not in ids:
            continue
        if m.group(1) not in cited_by.get(it.nodeid, set()):
            problems.append(f"{it.nodeid}: its xfail names {m.group(1)}, which does not cite it")
        if not it.strict:
            problems.append(f"{it.nodeid}: an xfail naming {m.group(1)} must be strict")
    return problems


def unowned_xfail_problems(inventories: list[Inventory], items: list[Item]) -> list[str]:
    """xfail reasons naming an id, under a declared prefix, that no inventory holds."""
    ids = {row.get("id") for row in every_row(inventories)}
    prefixes = {p for inv in inventories for p in inv.prefixes}
    problems = []
    for it in items:
        m = _REASON_ROW.match(it.reason) if it.xfail else None
        if m and m.group(2) in prefixes and m.group(1) not in ids:
            problems.append(f"{it.nodeid}: its xfail names {m.group(1)}, which no inventory holds")
    return problems


# --------------------------------------------------------------------------
# The rules on the inventories
# --------------------------------------------------------------------------
_NAMES = [p.name for p in INVENTORIES]


def test_every_expected_inventory_exists():
    names = set(_NAMES)
    assert set(EXPECTED) <= names, f"missing from docs/validation: {sorted(set(EXPECTED) - names)}"


def test_the_files_are_well_formed_and_own_disjoint_prefixes():
    problems = file_problems(load_all())
    assert not problems, "claims inventories:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("path", INVENTORIES, ids=_NAMES)
def test_every_row_is_well_formed(path):
    inv = load(path)
    problems = row_problems(inv.rows, inv.prefixes)
    assert not problems, f"{inv.name}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("path", INVENTORIES, ids=_NAMES)
def test_every_cited_test_exists_and_runs_on_every_push(path, collection):
    inv = load(path)
    problems = collection_problems(inv.rows, collection)
    assert not problems, f"{inv.name} cites tests that cannot fail on a push:\n  " \
        + "\n  ".join(problems)


@pytest.mark.parametrize("path", INVENTORIES, ids=_NAMES)
def test_statuses_and_xfails_agree(path, collection):
    inv = load(path)
    problems = xfail_problems(inv.rows, collection)
    assert not problems, f"{inv.name} statuses disagree with the tests:\n  " \
        + "\n  ".join(problems)


def test_no_xfail_names_a_row_no_inventory_holds(collection):
    problems = unowned_xfail_problems(load_all(), collection)
    assert not problems, "\n  ".join(problems)


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


def _inv(name="a_claims.yaml", **doc):
    base = {"schema_version": 1, "prefixes": ["CPL"], "claims": [_GOOD]}
    for k, v in doc.items():
        if v is None:
            base.pop(k, None)
        else:
            base[k] = v
    return Inventory(name, base)


@pytest.mark.parametrize("row, fragment", [
    (_row(id="CPL-1"), "not <PREFIX>-NNN"),
    (_row(id="SYS-001"), "not one of this file's"),
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
], ids=["id", "foreign-prefix", "status", "ambiguous", "failing", "untested", "no-test",
        "bare-name", "no-source", "missing-file", "not-a-file", "blank", "missing-key",
        "unknown-key"])
def test_the_row_rule_fires_on_each_defect(row, fragment):
    problems = row_problems([row], ["CPL"])
    assert any(fragment in p for p in problems), problems


def test_the_row_rule_passes_good_rows_and_refuses_a_duplicate_id():
    assert row_problems([_GOOD], ["CPL"]) == []
    assert row_problems([_row(id="SYS-001"), _row(id="FMU-001")], ["SYS", "FMU"]) == []
    c_source = next(iter(sorted((REPO_ROOT / "src/maddening/fmi/c").glob("*.c"))), None)
    if c_source is not None:     # a C source is a repository file too
        rel = c_source.relative_to(REPO_ROOT).as_posix()
        assert row_problems([_row(sources=[f"{rel}:1 (x)"])], ["CPL"]) == []
    assert any("used twice" in p for p in row_problems([_GOOD, dict(_GOOD)], ["CPL"]))


@pytest.mark.parametrize("inventories, fragment", [
    ([_inv(extra=1)], "unknown top-level key"),
    ([_inv(schema_version=2)], "schema_version"),
    ([_inv(schema_version=None)], "schema_version"),
    ([_inv(claims=[])], "'claims' must be a non-empty list"),
    ([_inv(prefixes=None)], "'prefixes' must be a non-empty list"),
    ([_inv(prefixes=["cpl"])], "2-6 capital letters"),
    ([_inv(prefixes=["CPL", "SYS"])], "prefix SYS names no row"),
    ([_inv("a_claims.yaml"), _inv("b_claims.yaml")], "also declared by a_claims.yaml"),
    ([Inventory("a_claims.yaml", ["not", "a", "mapping"])], "not a mapping"),
], ids=["unknown-key", "version", "no-version", "no-claims", "no-prefixes", "lowercase",
        "unused-prefix", "shared-prefix", "not-a-mapping"])
def test_the_file_rule_fires_on_each_defect(inventories, fragment):
    problems = file_problems(inventories)
    assert any(fragment in p for p in problems), problems


def test_the_file_rule_passes_disjoint_files():
    two = [_inv("a_claims.yaml"),
           _inv("b_claims.yaml", prefixes=["SYS", "FMU"],
                claims=[_row(id="SYS-001"), _row(id="FMU-001")])]
    assert file_problems(two) == []


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


def test_the_xfail_rule_leaves_another_files_rows_to_that_file():
    """SYS-001's xfail is the sysid file's: the coupling file neither flags nor claims it."""
    sys_row = _row(id="SYS-001", status="failing", finding="f",
                   tests=["tests/core/test_x.py::test_s"])
    xf = _item("tests/core/test_x.py::test_s", xfail=True, strict=True, reason="SYS-001: x")
    items = [_item(), xf]
    assert xfail_problems([_GOOD], items) == []
    assert xfail_problems([sys_row], items) == []
    dropped = dict(sys_row, tests=["tests/core/test_x.py::test_a"])
    assert any("which does not cite it" in p for p in xfail_problems([dropped], items))


def test_an_xfail_naming_a_row_nobody_holds_is_reported():
    invs = [_inv("a_claims.yaml"),
            _inv("b_claims.yaml", prefixes=["SYS"], claims=[_row(id="SYS-001")])]
    ghost = _item("tests/core/test_x.py::test_g", xfail=True, strict=True, reason="SYS-777: x")
    assert any("no inventory holds" in p for p in unowned_xfail_problems(invs, [ghost]))
    held = _item("tests/core/test_x.py::test_g", xfail=True, strict=True, reason="SYS-001: x")
    assert unowned_xfail_problems(invs, [held]) == []
    other = _item("tests/core/test_x.py::test_g", xfail=True, strict=True,
                  reason="differential: x; pending fix")
    assert unowned_xfail_problems(invs, [other]) == []
