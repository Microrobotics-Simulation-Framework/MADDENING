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
  ``tests/compliance/test_slow_only_rule.py``).  A test under
  ``tests/usd/`` needs ``usd-core``, which only the ``test-usd`` job
  installs to run tests with, so it must also be in that job's selection
  ("Tests that need an optional extra", below);
* **a failing row cites a strict xfail** whose reason starts with the
  row's id, so the row and the test move together when the fix lands;
* **a verified row cites no xfail**, strict or not: a claim the tree does
  not meet is not verified.  And every xfail whose reason names a row is
  strict and cited by that row, and names a row some inventory holds, so a
  test cannot outlive its row's status;
* **every row of a file with a domain matrix fills it**: a file that
  names a ``domain_set`` (one of :data:`DOMAIN_SETS`, a fixed list) gives
  every row a ``domains`` mapping with exactly that set's domains, each
  one a tested node id (or a list of them), ``narrowed`` or ``n/a``.  A
  ``narrowed`` domain is excluded in the row's ``conditions`` by a
  trailing "Not claimed for ..." clause that names it, and the clause
  names no domain that is not narrowed.  A domain the conditions name
  before that clause, or cover with a phrase such as "any dtype", is
  covered, so it cannot be ``n/a``.  A tested domain's test is collected
  and runs on every push (the rule above), counts for the xfail rules,
  and names its domain in its own source (a float64 cell cites a test
  that mentions x64 or float64): a floor that catches a cell citing a
  test written for another domain, not a proof that the test is apt.

Which tests are slow, skipped or xfail comes from pytest itself, in one
``--collect-only`` subprocess over the files every inventory cites.  The
rules are pure functions over the files, their rows and that collection,
so the self-tests at the end feed them broken files and rows and check
each rule fires.

Tests that need an optional extra
---------------------------------
``tests/usd/conftest.py`` skips its directory when ``pxr`` (``usd-core``)
cannot be imported.  This module runs in two kinds of job: the sharded
lanes install ``.[ci]``, without it, and the ``compliance`` job installs
``.[ci,usd]``.  Where ``pxr`` is importable nothing differs: pytest
collects the cited USD tests like any other.  Where it is not, pytest
cannot import those modules at all, so their items are read from their
source instead (:func:`items_from_source`): one item per test function,
with the ``slow``, ``skip`` and ``xfail`` marks its decorators, its
class's and the module's ``pytestmark`` spell.  Every rule then runs
over them unchanged, so a row citing a USD test that does not exist, is
skip-marked, is slow with no witness or disagrees with its status fails
in both kinds of job.

The source reader is not pytest: it expands no parameters (a cited
``test_x[id]`` is taken to exist when ``test_x`` does) and reads no mark
attached to one parameter or reached through an import.  Two tests keep
it honest.  One collects a module of every spelling it reads with pytest
and compares, in every job.  The other compares it with pytest's own
answer for the cited USD files, in the job that can collect them, on
every push -- so a spelling the reader does not know fails there rather
than passing here unread.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest
import yaml

from tests.compliance.test_ci_workflows import _logical_lines, _usd_job_targets, _workflow
from tests.compliance.test_slow_only_rule import _NODE_ID, _comments_at, _covers, witness_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION = REPO_ROOT / "docs" / "validation"
#: Every inventory, by its name: a new ``*_claims.yaml`` is checked without
#: anyone remembering to add it here.
INVENTORIES = sorted(VALIDATION.glob("*_claims.yaml"))
#: The inventories the repository must hold; a guard over nothing is no guard.
EXPECTED = ("coupling_claims.yaml", "sysid_fmu_claims.yaml", "rest_runpod_claims.yaml")

SCHEMA_VERSION = 1
TOP_LEVEL = ("schema_version", "prefixes", "domain_set", "claims")
REQUIRED = ("id", "area", "claim", "sources", "conditions", "oracle", "tests", "status")
OPTIONAL = ("proposed_wording", "finding", "reason", "notes", "domains")
STATUSES = ("verified", "failing", "untested", "ambiguous")
#: The field each status needs beside the required ones.
NEEDS = {"ambiguous": "proposed_wording", "failing": "finding", "untested": "reason"}
_PREFIX = re.compile(r"^[A-Z]{2,6}$")
_ID = re.compile(r"^([A-Z]{2,6})-\d{3}$")
#: A source is a repository path, then ``:line``, `` (section)`` or ``#anchor``.
_SOURCE_PATH = re.compile(r"^([\w./\-]+?\.(?:py|md|yaml|toml|json|txt|c|h|xml))(?=$|[:#\s(])")
#: An xfail reason that names a row.
_REASON_ROW = re.compile(r"^(([A-Z]{2,6})-\d{3}):")
#: Tests here need ``usd-core`` (imported as ``pxr``): the directory's
#: conftest skips them without it.  The sharded lanes do not install it, so
#: on a push these run in the ``test-usd`` job and nowhere else.
USD_TESTS = "tests/usd/"

# --------------------------------------------------------------------------
# The domain matrix: the fixed vocabularies, and how conditions name them
# --------------------------------------------------------------------------
#: The numeric domains a coupling, sysid or FMU claim can be exercised in
#: (``testing_standards.md``, "The domain matrix", defines each).
NUMERIC_DOMAINS = (
    "f32", "f64", "mixed_dtype", "16bit",
    "jit", "grad", "vmap",
    "multi_rate", "sub_cycled", "predictors_warm_starts",
    "adaptive", "checkpoint_restart", "sharded",
)
#: The domains a REST or run_pod claim can be exercised in
#: (``testing_standards.md``, "The domain matrix", defines each): where the
#: server is bound and whether the request must carry the token; what runs
#: beside the request (simultaneous requests on a real loopback server, the
#: realtime runner, an in-flight ``/sim/run``, a shutdown signal); what the
#: request carries (sizes at and past the caps, hostile values); what it
#: targets (a wrapper node, a graph restored from a checkpoint); and, for
#: ``run_pod.py``, the records ``--summarise`` reads (a dry run on CPU
#: devices, a record relabelled as a GPU run, files of several commits).
SERVER_DOMAINS = (
    "loopback_bind", "non_loopback_bind", "token_enforced", "no_token",
    "concurrent", "runner_active", "sim_run_active",
    "large_payload", "hostile_input", "shutdown",
    "wrapper_nodes", "checkpoint_restore",
    "dry_run_cpu", "relabelled_records", "mixed_commits",
)
#: Every vocabulary a file may name in its top-level ``domain_set``.  A
#: file that names none has no matrix, and its rows may not carry one.
DOMAIN_SETS: dict[str, tuple[str, ...]] = {"numeric": NUMERIC_DOMAINS,
                                           "server": SERVER_DOMAINS}
KNOWN_DOMAINS = frozenset(d for members in DOMAIN_SETS.values() for d in members)
NOT_APPLICABLE = "n/a"
NARROWED = "narrowed"
#: Cells waiting on a parallel branch: an oracle that will cover them, or a
#: fix PR whose tests will.  Temporary while those branches are open; the
#: matrix is complete only when none is left (PENDING_ALLOWED False).
PENDING = ("TODO-oracle", "TODO-fix")
PENDING_ALLOWED = False
#: A narrowed domain is excluded by a clause that starts with this and runs
#: to the end of the row's ``conditions``.
NARROWING_LEAD = "Not claimed for"
#: How a condition names each domain.  Matched in this order, each match
#: blanked before the next domain is looked for, so "float32 leaves under
#: x64" names mixed_dtype and not f32 or f64 as well.
DOMAIN_SPELLINGS: tuple[tuple[str, re.Pattern], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE)) for name, pattern in (
        ("mixed_dtype", r"mixed[- ](?:dtypes?|precisions?)|float32 (?:leaves|fields|parameters|"
                        r"nodes?|members?) (?:in|under|beside) (?:an? )?(?:x64|float64)"),
        ("16bit", r"\bb?float16\b|\bbfloat16\b|\b16-bit\b|\bsixteen-bit\b"),
        ("predictors_warm_starts", r"\bpredictors?\b|\bwarm[- ]starts?\b"),
        ("sub_cycled", r"\bsub-?cycl\w*"),
        ("multi_rate", r"\bmulti-?rate\b"),
        ("adaptive", r"\brun_adaptive\w*|\badaptive step\w*"),
        ("checkpoint_restart", r"\bcheckpoints?\b|\brestarts?\b|\bsave_state\b|\bload_state\b"),
        ("sharded", r"\bshard\w*|\bvirtual devices\b|\bdevice mesh\b"),
        ("vmap", r"\bvmap\w*|\bbatched\b"),
        ("grad", r"\bgrad\b|\bgradients?\b|\bjvp\b|\bvjp\b|\bjacfwd\b|\bjacrev\b|"
                 r"\bderivatives?\b|\bdifferentiat\w*|\b(?:forward|reverse)[- ]mode\b"),
        ("jit", r"\bjit\b|\bjitted\b|\bjax\.jit\b"),
        ("f64", r"\bfloat64\b|\bx64\b"),
        ("f32", r"\bfloat32\b"),
    ))
#: Phrases in a condition that cover several domains at once.
COVERING_PHRASES: tuple[tuple[re.Pattern, tuple[str, ...]], ...] = (
    (re.compile(r"\b(?:any|every|all) (?:float(?:ing)?(?:-point)? )?dtypes?\b", re.IGNORECASE),
     ("f32", "f64", "mixed_dtype", "16bit")),
    (re.compile(r"\b(?:any|every|all) graphs?\b", re.IGNORECASE), ("multi_rate", "sub_cycled")),
)
#: How a REST or run_pod condition names each server domain, in the same
#: way: "non-loopback bind" is consumed before "loopback bind" is looked
#: for, and "without the token" before "with the token".
SERVER_SPELLINGS: tuple[tuple[str, re.Pattern], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE)) for name, pattern in (
        ("non_loopback_bind", r"\bnon-loopback\b|\b0\.0\.0\.0\b|\bpublicly bound\b|"
                              r"\bevery other bind\b"),
        ("loopback_bind", r"\bloopback (?:binds?|bound|server|address(?:es)?|hosts?|"
                          r"spellings?)\b|\bloopback-bound\b|\bloopback keeps\b|"
                          r"\bloopback for\b"),
        ("no_token", r"\bno token\b|\bwithout (?:the|a) token\b"),
        ("token_enforced", r"\bwith the token\b|\bthe token presented\b|"
                           r"\btoken (?:is )?(?:enforced|demanded)\b|\bfor the credential\b|"
                           r"\bthe backstop\b|\banonymous\b|\bno Authorization header\b"),
        ("concurrent", r"\bconcurren\w*|\bsimultaneous\w*|\b\d+ threads\b|"
                       r"\bthreads asking\b"),
        # run_pod's goal runners are not the realtime runner
        ("runner_active", r"(?<!GitHub )\brunners?\b(?! replaced)(?!'s source)|\bsim/start\b"),
        ("sim_run_active", r"/sim/run (?:in progress|in flight|whose)\b|\bmid-run\b|"
                           r"\bin-flight /sim/run\b|"
                           r"\b(?:beside|during|while|slices of) an? /sim/run\b"),
        ("shutdown", r"\bSIGINT\b|\bSIGTERM\b|\bshut(?:s|ting)? ?down\b|\blifespan\b|"
                     r"\brequest_shutdown\b"),
        ("large_payload", r"\boversized?\b|\bthe (?:limit|bound|budget|cap)s? patched\b|"
                          r"\bpatched (?:budget|bound|limit)s?\b|\bthe boundary\b|"
                          r"\beach bound at its edge\b|\bstate caps\b|\bbudgets?\b|"
                          r"\bMAX_[A-Z_]+|\bfar more values\b|\bat its bound\b|"
                          r"\bthe bound(?: \+ 1)?\b|\bover the shipped\b|\babove 1000\b|"
                          r"\bover the cap\b|\btoo long\b"),
        ("hostile_input", r"\bmalformed\b|\bwrong[- ]typed?\b|\bwrong types?\b|"
                          r"\bwrong shape\b|\bnon-finite\b|\bNaN\b|\bInfinity\b|"
                          r"\.\. escapes|\bNUL\b|\bnon-base64\b|\bsurrogate characters\b|"
                          r"\bdamaged\b|\btruncated\b|\btampered\b|\bcorrupt\w*|"
                          r"\bforeign Origin\b|\brebound Host\b|\battacker\.example\b|"
                          r"\bnegative count\b|\bhostile\b|\bnot in UTF-8\b|"
                          r"\babsolute paths\b|\bsymlink out\b|\bnot base64\b"),
        ("wrapper_nodes", r"\bsharded (?:wrappers?|nodes?|body|bodies)\b|\bHybridNode\b|"
                          r"\bunstructured wrappers\b|\bwrapper nodes?\b"),
        # a checkpoint *save* that holds the lock is not a restore
        ("checkpoint_restore", r"\bcheckpoints?\b(?! save\b)|\bsave_state\b|\bload_state\b|"
                               r"\brestor\w*|/checkpoint/load\b"),
        ("dry_run_cpu", r"--dry-run\b|\bdry[- ]runs?\b|\bdry_run\b"),
        ("relabelled_records", r"\brelabell?ed\b|\brelabell?ing\b"),
        ("mixed_commits", r"\bcommits?\b|\bgit_commit\b"),
    ))
#: Phrases in a REST condition that cover several server domains at once:
#: either bind, and the token rule a bind implies (a non-loopback bind
#: always demands the token; a loopback bind demands none of a loopback
#: peer).
SERVER_COVERING_PHRASES: tuple[tuple[re.Pattern, tuple[str, ...]], ...] = (
    (re.compile(r"\b(?:any|every|both kinds of) binds?\b|\bboth binds\b", re.IGNORECASE),
     ("loopback_bind", "non_loopback_bind", "token_enforced", "no_token")),
    (re.compile(r"\bnon-loopback (?:bind|spelling)", re.IGNORECASE), ("token_enforced",)),
    (re.compile(r"(?<!non-)\bloopback bind\b", re.IGNORECASE), ("no_token",)),
)
#: Each domain set's spellings and covering phrases.
SPELLINGS = {"numeric": (DOMAIN_SPELLINGS, COVERING_PHRASES),
             "server": (SERVER_SPELLINGS, SERVER_COVERING_PHRASES)}
#: What a tested cell's test must say somewhere in its source (its node id,
#: its decorators, its body and the definitions it names, three levels
#: deep): the domain's own vocabulary.  A floor, not a proof.
DOMAIN_WITNESS: dict[str, re.Pattern] = {
    name: re.compile(pattern, re.IGNORECASE) for name, pattern in (
        # float64 state exists only under x64; "float64" alone is usually an
        # oracle's precision, not the domain under test.
        ("f64", r"x64"),
        ("16bit", r"float16|bfloat16|sixteen"),
        ("jit", r"jit|compile|run_scan|\.step\(|\.run\(|lax\.|while_loop|fori_loop"),
        # The fitters and fim differentiate the loss or residual they are given.
        ("grad", r"grad|jvp|vjp|jacfwd|jacrev|jacobian|hessian|linearize|derivative|"
                 r"\bfit\w*\(|\bfim\w*\("),
        ("vmap", r"vmap|run_sweep"),
        ("multi_rate", r"multi.?rate|rate.?divider|start_step"),
        ("sub_cycled", r"sub.?cycl"),
        ("predictors_warm_starts", r"predictor|warm|jacobian_reuse"),
        ("adaptive", r"adaptive"),
        ("checkpoint_restart", r"checkpoint|save_state|load_state|restart|snapshot|fmu_state|"
                               r"get_state|set_state|restore|reset|window"),
        ("sharded", r"shard|mesh"),
        # -- the server set --
        ("non_loopback_bind", r"0\.0\.0\.0|non.?loopback|public|bind_host=|203\.0\.113"),
        # a non-loopback bind always demands the token, and a 401 is its refusal
        ("token_enforced", r"Bearer|Authorization|enforced|0\.0\.0\.0|\b401\b"),
        ("runner_active", r"sim/start|runner"),
        ("sim_run_active", r"sim/run"),
        ("large_payload", r"MAX_|limit|budget|\bcaps?\b|413|oversiz|bound|large|huge|"
                          r"too.?long|NAME_MAX|\d+(?:_000){2,}"),
        ("hostile_input", r"malformed|\bnan\b|infinity|non.?finite|traversal|\.\./|\\x00|"
                          r"\bNUL\b|garbage|hostile|invalid|wrong|foreign|attacker|tamper|"
                          r"truncat|corrupt|@given|\bbool|\bstr\b|string|null|base64|"
                          r"cross.?origin|not.?a.?host|negative|out.?of.?range|missing|"
                          r"unknown|fraction|flush|subnormal|empty"),
        ("shutdown", r"SIGINT|SIGTERM|shutdown|signal|lifespan"),
        ("wrapper_nodes", r"Sharded\w*Node|HybridNode|hybrid|wrapper"),
        ("checkpoint_restore", r"checkpoint|save_state|load_state|restore"),
        # run_pod_record/ is a real --dry-run's output, read as _RECORD
        ("dry_run_cpu", r"dry.?run|virtual|\bcpu\b|run_pod_record|_RECORD\b"),
        ("relabelled_records", r"relabel"),
        ("mixed_commits", r"commit"),
    )}
#: A real server's mark, for ``concurrent``: the requests must meet on one
#: server's socket, event loop and worker pool, not each on its own
#: in-process client.  The mark is the call that starts one --
#: ``uvicorn.run(...)`` (a server script's too) or ``uvicorn.Server(...)``,
#: which ``rest_claims_support.loopback_server`` makes -- because an
#: in-process client has no reason to make it.  A URL, a ``127.0.0.1``, the
#: word "socket" (every WebSocket test says it) or "uvicorn" in passing are
#: no evidence: a client that names a loopback Host says all of them.  Nor
#: is the call quoted in backticks, as a docstring quotes it.  (Prose that
#: spells the call unquoted still reads as one: a floor, as every witness is.)
_REAL_SERVER = re.compile(r"(?<!`)\b\w*uvicorn(?:\(\))?\.(?:run|Server)\(")
#: ... and the requests must be simultaneous: the word, a barrier or a
#: gather, or a thread the test starts.  Two things the helper that starts
#: the server says are no evidence, or every test that starts one would be
#: simultaneous by that alone: the thread the server itself runs in
#: (``Thread(target=userver.run)``), and the domain's own name, which the
#: helper passes as a tag (``make_server("concurrent", ...)``).
_SIMULTANEOUS = re.compile(r"simultaneous|barrier|gather|Thread\(\s*target=(?!\w+\.run\b)",
                           re.IGNORECASE)
#: Test modules whose definitions are not read as a test's own words.  The
#: shared in-process client names a loopback Host and peer for every REST
#: test that constructs it (and annotates a ``str``), so read as evidence it
#: would witness ``loopback_bind``, ``no_token`` and ``hostile_input`` for a
#: test of none of them, and its URL once read as a real server.  What a
#: test says of its domain it says itself, or in a helper written for it.
TRANSPORT_MODULES = frozenset({"tests/_loopback_client.py"})


def witnesses(domain: str, text: str) -> bool:
    """Whether *text*, a test's source, names *domain*."""
    if domain == "f32":     # the default precision: anything but an x64-only test
        return bool(re.search(r"float32|bfloat16|float16", text)) or not re.search(
            r"x64", text, re.IGNORECASE)
    if domain == "mixed_dtype":
        return bool(re.search(r"mixed", text, re.IGNORECASE)) or bool(
            re.search(r"x64", text, re.IGNORECASE) and re.search(r"float32", text))
    if domain == "loopback_bind":   # the default bind: anything but a non-loopback-only test
        return bool(re.search(r"loopback|127\.0\.0\.1|localhost", text, re.IGNORECASE)) or \
            not DOMAIN_WITNESS["non_loopback_bind"].search(text)
    if domain == "no_token":        # the default: anything but a test that always sends one
        return bool(re.search(r"anonymous|no.?token|without|not.?challenged|"
                              r"enforced\s*(?:is\s*False|==)|loopback|127\.0\.0\.1",
                              text, re.IGNORECASE)) or \
            not DOMAIN_WITNESS["token_enforced"].search(text)
    if domain == "concurrent":
        return bool(_REAL_SERVER.search(text) and _SIMULTANEOUS.search(text))
    return bool(DOMAIN_WITNESS[domain].search(text))


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

    @property
    def domain_set(self):
        return self.doc.get("domain_set") if isinstance(self.doc, dict) else None

    @property
    def domains(self) -> tuple[str, ...] | None:
        """The domain names this file's rows fill, or None for no matrix."""
        return DOMAIN_SETS.get(self.domain_set) if isinstance(self.domain_set, str) else None


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
        if "domain_set" in doc and not (isinstance(doc["domain_set"], str)
                                        and doc["domain_set"] in DOMAIN_SETS):
            problems.append(f"{inv.name}: domain_set {doc['domain_set']!r} is not one of "
                            f"{sorted(DOMAIN_SETS)}")
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
def row_problems(rows: list, prefixes, repo_root: Path = REPO_ROOT,
                 domains: tuple[str, ...] | None = None) -> list[str]:
    """What is wrong with one file's rows, read alone; *prefixes* are the file's own,
    and *domains* its domain set (None: the file has no matrix)."""
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
        problems += domain_problems(row, domains)
    return problems


# --------------------------------------------------------------------------
# Rule 1b: the domain matrix of a row
# --------------------------------------------------------------------------
def domain_mentions(text: str, domain_set: str = "numeric") -> set[str]:
    """The domains of *domain_set* that *text* names, by its spellings
    (:data:`DOMAIN_SPELLINGS`, :data:`SERVER_SPELLINGS`), most specific first."""
    found: set[str] = set()
    rest = text

    def blank(name):
        def sub(m):
            found.add(name)
            return " " * len(m.group(0))
        return sub

    for name, pattern in SPELLINGS[domain_set][0]:
        rest = pattern.sub(blank(name), rest)
    return found


def split_conditions(text: str) -> tuple[str, str]:
    """``(what the conditions claim, the trailing "Not claimed for" clause or "")``."""
    i = text.find(NARROWING_LEAD)
    return (text, "") if i < 0 else (text[:i], text[i:])


def covered_domains(text: str, domain_set: str = "numeric") -> set[str]:
    """The domains a condition (before its narrowing clause) names or covers."""
    found = domain_mentions(text, domain_set)
    for pattern, names in SPELLINGS[domain_set][1]:
        if pattern.search(text):
            found.update(names)
    return found


def set_of(domains: tuple[str, ...]) -> str:
    """The name of the domain set whose domains are *domains*."""
    return next(name for name, members in DOMAIN_SETS.items() if members == domains)


def cell_tests(value) -> list[str] | None:
    """The node ids a tested cell cites, or None if *value* is not a tested cell."""
    values = value if isinstance(value, list) and value else [value]
    if all(isinstance(v, str) and _NODE_ID.fullmatch(v) for v in values):
        return list(values)
    return None


def domain_problems(row: dict, domains: tuple[str, ...] | None) -> list[str]:
    """What is wrong with one row's ``domains``, against its file's domain set."""
    where = row.get("id") or "a row"
    cells = row.get("domains")
    if domains is None:
        return ([f"{where}: a 'domains' matrix, but its file names no domain_set"]
                if "domains" in row else [])
    if not isinstance(cells, dict):
        return [f"{where}: no 'domains' mapping of each domain to a test, "
                f"'{NARROWED}' or '{NOT_APPLICABLE}'"]
    problems = []
    missing = [d for d in domains if d not in cells]
    if missing:
        problems.append(f"{where}: 'domains' says nothing about {missing}")
    unknown = sorted(str(k) for k in cells if k not in domains)
    if unknown:
        problems.append(f"{where}: {unknown} not in its file's domain set")
    conditions = row.get("conditions") if isinstance(row.get("conditions"), str) else ""
    claimed, clause = split_conditions(conditions)
    vocabulary = set_of(domains)
    excluded = domain_mentions(clause, vocabulary)
    covered = covered_domains(claimed, vocabulary)
    if clause and not excluded:
        problems.append(f"{where}: its '{NARROWING_LEAD}' clause names no domain")
    for d in domains:
        if d not in cells:
            continue
        value = cells[d]
        narrowed = value == NARROWED
        pending = PENDING_ALLOWED and value in PENDING
        if not (narrowed or pending or value == NOT_APPLICABLE or cell_tests(value) is not None):
            problems.append(f"{where}: domain {d}: {value!r} is not a node id, a list of them, "
                            f"'{NARROWED}' or '{NOT_APPLICABLE}'")
        if narrowed and d not in excluded:
            problems.append(f"{where}: domain {d} is narrowed, but its conditions do not "
                            f"exclude it (a trailing '{NARROWING_LEAD} ...' clause naming it)")
        if d in excluded and not narrowed:
            problems.append(f"{where}: its conditions exclude {d}, but the cell is {value!r}")
        if value == NOT_APPLICABLE and d in covered:
            problems.append(f"{where}: domain {d} is n/a, but its conditions cover it")
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


def domain_targets(row: dict) -> list[tuple[str, str]]:
    """``(domain, node id)`` for every test the row's domain matrix cites."""
    cells = row.get("domains")
    if not isinstance(cells, dict):
        return []
    return [(str(d), t) for d, v in cells.items() for t in (cell_tests(v) or [])]


def cited_targets(row: dict) -> list[str]:
    """Every node id a row cites: its tests, then its tested domains'."""
    tests = [t for t in (row.get("tests") or []) if isinstance(t, str)]
    return tests + [t for _, t in domain_targets(row)]


def cited_files(rows: list) -> list[str]:
    return sorted({t.split("::", 1)[0] for row in rows if isinstance(row, dict)
                   for t in cited_targets(row)})


def usd_core_installed() -> bool:
    """Whether pytest can collect ``tests/usd/`` here: ``pxr`` can be found."""
    return importlib.util.find_spec("pxr") is not None


def _collect(files: list[str], dump: Path, cwd: Path = REPO_ROOT) -> list[Item]:
    """What pytest collects from *files*, with ``-m "slow or not slow"``."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("MADDENING_TEST_SHARD", "MADDENING_TEST_JAX_TIMING", "PYTEST_ADDOPTS")}
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", JAX_PLATFORMS="cpu",
               CLAIMS_INVENTORY_DUMP=str(dump))
    proc = subprocess.run(
        [sys.executable, "-c", _COLLECT, "--collect-only", "-q", "-p", "no:cacheprovider",
         "-m", "slow or not slow", *files],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return [Item(*row) for row in json.loads(dump.read_text())]


def split_by_lane(files: list[str], usd_core: bool) -> tuple[list[str], list[str]]:
    """``(the files pytest can collect here, the ones to read from source)``:
    without ``usd-core`` the cited files under :data:`USD_TESTS` are the second."""
    unread = [] if usd_core else [f for f in files if f.startswith(USD_TESTS)]
    return [f for f in files if f not in unread], unread


@pytest.fixture(scope="module")
def collection(tmp_path_factory) -> list[Item]:
    """Every item of every cited file, collected once with ``-m "slow or not
    slow"``; where ``usd-core`` is not installed, the items of the cited
    files under ``tests/usd/`` are read from their source instead."""
    rows = every_row(load_all())
    files, unread = split_by_lane(
        [f for f in cited_files(rows) if (REPO_ROOT / f).is_file()], usd_core_installed())
    items = _collect(files, tmp_path_factory.mktemp("claims") / "items.json")
    cited = [t for row in rows for t in cited_targets(row)]
    return items + [it for f in unread for it in items_from_source(f, cited)]


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
        for target in cited_targets(row):
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


def lane_problems(rows: list, usd_job: list[str]) -> list[str]:
    """Cited tests under ``tests/usd/`` that the ``test-usd`` job does not run.

    *usd_job* is what that job hands pytest (``_usd_job_targets``).  The
    sharded lanes skip the directory for want of ``usd-core``, so a USD
    test outside that selection runs in no per-push job, however it is
    marked."""
    problems = []
    for row in rows:
        for target in cited_targets(row):
            rel = target.split("::", 1)[0]
            if rel.startswith(USD_TESTS) and not any(
                    rel == s or rel.startswith(s.rstrip("/") + "/") or _covers(s, target)
                    for s in usd_job):
                problems.append(f"{row.get('id')}: {target} -- needs usd-core, and the test-usd "
                                "job (the one per-push job that runs tests with it) does not "
                                "select it")
    return problems


# --------------------------------------------------------------------------
# Reading a module's items from its source, where pytest cannot import it
# --------------------------------------------------------------------------
#: The marks the rules read.
_MARKS = ("slow", "skip", "xfail")


def _mark(expr, defs: dict):
    """``(name, the call or None)`` when *expr* spells ``<x>.mark.<name>``,
    called or bare, or is a module-level name bound to one; else None."""
    if isinstance(expr, ast.Name):
        bound = defs.get(expr.id, (None, None))[1]
        expr = bound.value if isinstance(bound, (ast.Assign, ast.AnnAssign)) else None
    call = expr if isinstance(expr, ast.Call) else None
    attr = call.func if call is not None else expr
    if isinstance(attr, ast.Attribute) and attr.attr in _MARKS \
            and isinstance(attr.value, ast.Attribute) and attr.value.attr == "mark":
        return attr.attr, call
    return None


def _literal(node, default):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return default


def items_from_source(rel: str, cited=(), repo_root: Path = REPO_ROOT) -> list[Item]:
    """The items of test module *rel*, read from its source rather than collected.

    One item per ``test*`` function at module level and per ``test*``
    method of a ``Test*`` class, with the closest ``slow``, ``skip`` and
    ``xfail`` mark among the function's decorators, its class's and the
    module's ``pytestmark`` (a module-level name bound to a mark is
    followed).  An xfail's ``strict`` and ``reason`` are its literal
    keywords; ``strict`` defaults to False, as the project's
    ``xfail_strict`` does.

    Parameters are not expanded.  A node id in *cited* that names one
    parameter of a function found here gets an item of its own with the
    function's marks, so the citation is covered; whether that parameter
    exists is pytest's to say, where it can collect the module.  See the
    module docstring for what else this does not read and what holds it
    to pytest's answer.
    """
    mod = _module(rel, repo_root)
    if mod is None:
        return []
    _text, tree, defs = mod

    def marks(exprs, inherited: dict) -> dict:
        found = dict(inherited)
        for expr in exprs:
            got = _mark(expr, defs)
            if got is not None:
                found[got[0]] = got[1]
        return found

    def item(nodeid: str, found: dict) -> Item:
        call = found.get("xfail")
        keywords = {k.arg: k.value for k in call.keywords} if call is not None else {}
        return Item(nodeid, "slow" in found, "skip" in found, "xfail" in found,
                    "strict" in keywords and bool(_literal(keywords["strict"], False)),
                    str(_literal(keywords["reason"], "")) if "reason" in keywords else "")

    def is_test(stmt) -> bool:
        return isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) \
            and stmt.name.startswith("test")

    whole = defs.get("pytestmark", (None, None))[1]
    value = whole.value if isinstance(whole, (ast.Assign, ast.AnnAssign)) else None
    module_marks = marks(value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value], {})
    items = []
    for stmt in tree.body:
        if is_test(stmt):
            items.append(item(f"{rel}::{stmt.name}", marks(stmt.decorator_list, module_marks)))
        elif isinstance(stmt, ast.ClassDef) and stmt.name.startswith("Test"):
            class_marks = marks(stmt.decorator_list, module_marks)
            items += [item(f"{rel}::{stmt.name}::{sub.name}",
                           marks(sub.decorator_list, class_marks))
                      for sub in stmt.body if is_test(sub)]
    by_function = {it.nodeid: it for it in items}
    for target in sorted(set(cited)):
        function = target.split("[", 1)[0]
        if target != function and function in by_function:
            items.append(dataclasses.replace(by_function[function], nodeid=target))
    return items


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
        covered = [it for t in cited_targets(row) for it in items if _covers(t, it.nodeid)]
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
# Rule 5: a tested domain cites a test that names its domain
# --------------------------------------------------------------------------
@cache
def _module(rel: str, repo_root: Path = REPO_ROOT):
    """``(text, tree, top-level definitions by name)`` of a test module, or None."""
    path = repo_root / rel
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=rel)
    defs: dict[str, object] = {}
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs[stmt.name] = (rel, stmt)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    defs[t.id] = (rel, stmt)
        elif isinstance(stmt, ast.ImportFrom) and stmt.module and stmt.module.startswith("tests"):
            for alias in stmt.names:
                as_module = f"{stmt.module}.{alias.name}".replace(".", "/") + ".py"
                if (repo_root / as_module).is_file():     # from tests.x import module
                    defs.setdefault(alias.asname or alias.name, (as_module, None))
                else:
                    defs.setdefault(alias.asname or alias.name,
                                    (stmt.module.replace(".", "/") + ".py", alias.name))
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name.startswith("tests.") and alias.asname:
                    defs.setdefault(alias.asname, (alias.name.replace(".", "/") + ".py", None))
    return text, tree, defs


def _imports_transport(node) -> bool:
    """Whether *node* is an import of (or from) one of :data:`TRANSPORT_MODULES`."""
    def path(dotted: str) -> str:
        return dotted.replace(".", "/") + ".py"
    if isinstance(node, ast.ImportFrom) and node.module:
        return path(node.module) in TRANSPORT_MODULES or any(
            path(f"{node.module}.{a.name}") in TRANSPORT_MODULES for a in node.names)
    return isinstance(node, ast.Import) and any(path(a.name) in TRANSPORT_MODULES
                                                for a in node.names)


def _segment(text: str, node) -> str:
    """*node*'s source lines, decorators included, less any import of a
    transport module inside it: the import names the client, not a domain."""
    lines = text.splitlines()
    first = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
    dropped = {n for imp in ast.walk(node) if _imports_transport(imp)
               for n in range(imp.lineno, imp.end_lineno + 1)}
    return "\n".join(ln for n, ln in enumerate(lines[first - 1:node.end_lineno], first)
                     if n not in dropped)


def _resolve(entry, repo_root: Path):
    """A definition entry ``(file, node)`` or an import ``(file, name)`` -> ``(text, node)``."""
    rel, node = entry
    if node is None:                    # a module alias: nothing to read by itself
        return None
    if rel in TRANSPORT_MODULES:        # the shared client: no test's own words
        return None
    if isinstance(node, str):           # imported from another test module
        mod = _module(rel, repo_root)
        if mod is None or node not in mod[2] or isinstance(mod[2][node][1], str):
            return None
        rel, node = mod[2][node]
    mod = _module(rel, repo_root)
    return None if mod is None else (mod[0], node, mod[2])


def source_of_test(nodeid: str, repo_root: Path = REPO_ROOT, depth: int = 3) -> str | None:
    """The text a tested cell's witness is looked for in, or None if the test
    cannot be found: the node id, the test with its decorators (and its
    class's), the module's ``pytestmark``, and every module-level definition
    the test names -- a helper, a fixture, a constant, one imported from
    another test module, ``alias.name`` of a test module imported as
    ``alias`` -- and the ones those name, *depth* levels deep."""
    rel, *names = nodeid.split("[", 1)[0].split("::")
    mod = _module(rel, repo_root)
    if mod is None or not names:
        return None
    text, tree, defs = mod
    owner = None
    func = defs.get(names[0], (None, None))[1]
    if len(names) == 2 and isinstance(func, ast.ClassDef):
        owner, func = func, next((s for s in func.body if isinstance(
            s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == names[1]), None)
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) or len(names) > 2:
        return None
    pieces = [nodeid, _segment(text, func)]
    scope, members = defs, {}
    if owner is not None:
        pieces += [ast.unparse(d) for d in owner.decorator_list]
        # The class's own helpers and fixtures, reached as ``self.name`` or
        # by a fixture argument, shadow the module's.
        members = {s.name: (rel, s) for s in owner.body
                   if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))}
        scope = {**defs, **members}
    if "pytestmark" in defs and not isinstance(defs["pytestmark"][1], str):
        pieces.append(_segment(text, defs["pytestmark"][1]))
    seen: set[int] = {id(func)}
    frontier = [(text, func, scope)]
    for _ in range(depth):
        nxt = []
        for src, node, scope in frontier:
            names_used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            names_used |= {a.arg for a in ast.walk(node) if isinstance(a, ast.arg)}
            entries = [scope.get(name) for name in sorted(names_used)]
            # ``alias.name`` where ``alias`` is a test module imported here,
            # and ``self.name`` / ``cls.name`` for a method of the test's class
            for n in ast.walk(node):
                if not (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)):
                    continue
                if scope.get(n.value.id, (None, 0))[1] is None:
                    entries.append((scope[n.value.id][0], n.attr))
                elif n.value.id in ("self", "cls"):
                    entries.append(scope.get(n.attr))
            for entry in entries:
                got = _resolve(entry, repo_root) if entry else None
                if got is None or id(got[1]) in seen:
                    continue
                seen.add(id(got[1]))
                pieces.append(_segment(got[0], got[1]))
                # a method of the test's class keeps the class's scope
                nxt.append((got[0], got[1], scope) if entry in members.values() else got)
        frontier = nxt
    return "\n".join(pieces)


def domain_witness_problems(rows: list, source=source_of_test) -> list[str]:
    """Tested domains whose test does not name its domain anywhere in its source."""
    problems = []
    for row in rows:
        for domain, target in domain_targets(row):
            text = source(target)
            if text is None:
                problems.append(f"{row.get('id')}: domain {domain}: cannot find {target}'s source")
            elif domain in KNOWN_DOMAINS and not witnesses(domain, text):
                problems.append(f"{row.get('id')}: domain {domain}: {target} never names "
                                f"its domain (see DOMAIN_WITNESS)")
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
    problems = row_problems(inv.rows, inv.prefixes, domains=inv.domains)
    assert not problems, f"{inv.name}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("path", INVENTORIES, ids=_NAMES)
def test_every_tested_domain_cites_a_test_that_names_its_domain(path):
    inv = load(path)
    problems = domain_witness_problems(inv.rows)
    assert not problems, f"{inv.name}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("path", INVENTORIES, ids=_NAMES)
def test_every_cited_test_exists_and_runs_on_every_push(path, collection):
    inv = load(path)
    problems = collection_problems(inv.rows, collection) \
        + lane_problems(inv.rows, _usd_job_targets())
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


# --------------------------------------------------------------------------
# Self-tests of the domain matrix
# --------------------------------------------------------------------------
_T = "tests/core/test_x.py::test_a"


def _cells(**changes):
    """A full numeric matrix: every domain n/a but the ones changed."""
    cells = {d: NOT_APPLICABLE for d in NUMERIC_DOMAINS}
    cells.update({k.replace("bit16", "16bit"): v for k, v in changes.items()})
    return cells


def _drow(conditions="k", **changes):
    return _row(conditions=conditions, domains=_cells(**changes))


@pytest.mark.parametrize("row, fragment", [
    (_row(), "no 'domains' mapping"),
    (_row(domains=["f32"]), "no 'domains' mapping"),
    (_row(domains={d: NOT_APPLICABLE for d in NUMERIC_DOMAINS if d != "vmap"}),
     "says nothing about ['vmap']"),
    (_row(domains=dict(_cells(), gpu=NOT_APPLICABLE)), "not in its file's domain set"),
    (_drow(f64="maybe"), "is not a node id, a list of them"),
    (_drow(f64=3), "is not a node id, a list of them"),
    (_drow(f64=[]), "is not a node id, a list of them"),
    (_drow(f64=["tests/core/test_x.py"]), "is not a node id, a list of them"),
    (_drow(f64=[_T, "test_b"]), "is not a node id, a list of them"),
    (_drow(f64=NARROWED), "do not exclude it"),
    (_drow("float32 only. Not claimed for vmap.", f64=NARROWED, vmap=NARROWED),
     "domain f64 is narrowed"),
    (_drow("k. Not claimed for float64.", f64=_T), "exclude f64, but the cell is"),
    (_drow("k. Not claimed for float64.", f64=NOT_APPLICABLE), "exclude f64, but the cell is"),
    (_drow("k. Not claimed for the moon.", f64=NARROWED), "names no domain"),
    (_drow("under vmap and jit"), "domain vmap is n/a, but its conditions cover it"),
    (_drow("any dtype", f32=_T), "domain 16bit is n/a, but its conditions cover it"),
    (_drow("every graph", f32=_T), "domain sub_cycled is n/a, but its conditions cover it"),
    (_drow("a sub-cycled group"), "domain sub_cycled is n/a"),
    (_drow("an IFT gradient"), "domain grad is n/a"),
    (_drow("float32 leaves under x64"), "domain mixed_dtype is n/a"),
], ids=["no-matrix", "not-a-mapping", "missing-domain", "unknown-domain", "bad-word",
        "bad-type", "empty-list", "bad-node-id", "bad-node-in-list", "narrowed-unexcluded",
        "narrowed-other", "excluded-but-tested", "excluded-but-na", "empty-clause",
        "named-but-na", "any-dtype", "any-graph", "sub-cycled-named", "gradient-named",
        "mixed-named"])
def test_the_domain_rule_fires_on_each_defect(row, fragment):
    problems = row_problems([row], ["CPL"], domains=NUMERIC_DOMAINS)
    assert any(fragment in p for p in problems), problems


def test_the_domain_rule_passes_a_filled_matrix_and_reads_each_spelling():
    assert row_problems([_drow(f32=_T, jit=[_T, "tests/core/test_x.py::test_b[1]"])], ["CPL"],
                        domains=NUMERIC_DOMAINS) == []
    narrowed = _drow("both solvers. Not claimed for float64, mixed dtypes, 16-bit floats, vmap, "
                     "multi-rate or sub-cycled graphs, predictors, run_adaptive, checkpoints "
                     "or sharded graphs.", f32=_T,
                     **{d: NARROWED for d in ("f64", "mixed_dtype", "bit16", "vmap", "multi_rate",
                                              "sub_cycled", "predictors_warm_starts", "adaptive",
                                              "checkpoint_restart", "sharded")})
    assert row_problems([narrowed], ["CPL"], domains=NUMERIC_DOMAINS) == []
    # The specific spelling is consumed first: this names mixed_dtype only.
    assert domain_mentions("float32 leaves under x64") == {"mixed_dtype"}
    assert domain_mentions("bfloat16 and float16") == {"16bit"}
    assert domain_mentions("a jitted loss; jax.grad; jvp") == {"jit", "grad"}
    assert domain_mentions("WaveletAdaptiveNode") == set()
    # A domain named only after the clause is not covered by the conditions.
    assert covered_domains(split_conditions("k. Not claimed for vmap.")[0]) == set()
    # A file without a domain set: its rows carry no matrix, and need none.
    assert row_problems([_GOOD], ["CPL"]) == []
    assert any("names no domain_set" in p
               for p in row_problems([_drow(f32=_T)], ["CPL"]))


def test_the_file_rule_refuses_an_unknown_domain_set():
    assert any("domain_set 'gpu'" in p for p in file_problems([_inv(domain_set="gpu")]))
    assert any("domain_set 3" in p for p in file_problems([_inv(domain_set=3)]))
    assert file_problems([_inv(domain_set="numeric")]) == []
    assert file_problems([_inv(domain_set="server")]) == []
    assert _inv(domain_set="numeric").domains == NUMERIC_DOMAINS
    assert _inv(domain_set="server").domains == SERVER_DOMAINS
    assert _inv().domains is None


def test_the_collection_and_xfail_rules_read_the_domain_tests():
    row = _drow(f32=_T, vmap="tests/core/test_x.py::test_v")
    assert any("test_v -- pytest collects no such test" in p
               for p in collection_problems([row], [_item()]))
    vm = _item("tests/core/test_x.py::test_v", slow=True)
    assert any("no '# Per push" in p for p in collection_problems(
        [row], [_item(), vm], witness=lambda f: False))
    assert collection_problems([row], [_item(), _item("tests/core/test_x.py::test_v")]) == []
    # A verified row cites no xfail, in its domains either.
    xv = _item("tests/core/test_x.py::test_v", xfail=True, strict=True, reason="CPL-900: x")
    assert any("verified, but cites the xfail" in p for p in xfail_problems([row], [_item(), xv]))
    # A failing row's strict xfail may be the failing domain's test, which cites it.
    failing = dict(row, status="failing", finding="f")
    assert xfail_problems([failing], [_item(), xv]) == []
    assert cited_files([row]) == ["tests/core/test_x.py"]


def test_the_witness_rule_wants_each_domain_named_in_its_test():
    row = _drow(f32=_T, f64="tests/core/test_x.py::test_d")
    sources = {_T: "def test_a(): gm.step()", "tests/core/test_x.py::test_d": "def test_d(): ..."}
    assert any("domain f64: tests/core/test_x.py::test_d never names" in p
               for p in domain_witness_problems([row], source=sources.get))
    sources["tests/core/test_x.py::test_d"] = "def test_d():\n    with _x64(): gm.step()"
    assert domain_witness_problems([row], source=sources.get) == []
    # f32 is the default: any test but one that only runs under x64.
    sources[_T] = "def test_a():\n    jax.config.update('jax_enable_x64', True)"
    assert any("domain f32" in p for p in domain_witness_problems([row], source=sources.get))
    sources[_T] = "def test_a(dtype=jnp.float32):\n    with _x64(): ..."
    assert domain_witness_problems([row], source=sources.get) == []
    assert any("cannot find" in p for p in domain_witness_problems([row], source=lambda t: None))
    assert witnesses("mixed_dtype", "with _x64(): jnp.float32") and not witnesses(
        "mixed_dtype", "with _x64(): jnp.float64")
    for domain in NUMERIC_DOMAINS:
        assert not witnesses(domain, "def test_q(): x64") or domain in ("f64",)


def test_the_source_reader_follows_the_names_a_test_uses(tmp_path):
    (tmp_path / "tests" / "core").mkdir(parents=True)
    (tmp_path / "tests" / "core" / "support.py").write_text(
        "def farthest():\n    return 'jax.vmap'\n\ndef farther():\n    return farthest()\n\n"
        "def far():\n    return farther()\n\ndef helper():\n    return far()\n\n"
        "def by_alias():\n    return 'shard_map'\n")
    (tmp_path / "tests" / "core" / "test_y.py").write_text(
        "from tests.core.support import helper\n"
        "import tests.core.support as sup\n"
        "import pytest\n\n"
        "def _deep():\n    return 'run_adaptive'\n\n"
        "def _build():\n    return _deep()\n\n"
        "@pytest.fixture\ndef graph():\n    return 'subcycling=True'\n\n"
        "@pytest.mark.parametrize('n', [1])\n"
        "def test_one(graph, n):\n    _build(); helper(); sup.by_alias()\n\n"
        "class TestK:\n    def test_two(self):\n        pass\n")
    one = source_of_test("tests/core/test_y.py::test_one[1]", repo_root=tmp_path)
    assert "subcycling=True" in one and "run_adaptive" in one and "parametrize" in one
    assert "helper()" in one and "return far()" in one and "return farther()" in one
    assert "shard_map" in one      # sup.by_alias, through the module alias
    assert "jax.vmap" not in one   # four levels away: not read
    assert source_of_test("tests/core/test_y.py::TestK::test_two", repo_root=tmp_path)
    assert source_of_test("tests/core/test_y.py::test_none", repo_root=tmp_path) is None
    assert source_of_test("tests/core/test_nope.py::test_one", repo_root=tmp_path) is None


def test_a_pending_cell_is_accepted_only_while_pending_cells_are(monkeypatch):
    """``TODO-oracle`` / ``TODO-fix`` mark a cell a parallel branch will fill; the
    guard takes them only while ``PENDING_ALLOWED`` is set, so a release that
    turns it off fails on any left."""
    module = sys.modules[__name__]
    row = _drow(f32=_T, f64="TODO-oracle", vmap="TODO-fix")
    monkeypatch.setattr(module, "PENDING_ALLOWED", True)
    assert row_problems([row], ["CPL"], domains=NUMERIC_DOMAINS) == []
    monkeypatch.setattr(module, "PENDING_ALLOWED", False)
    problems = row_problems([row], ["CPL"], domains=NUMERIC_DOMAINS)
    assert any("domain f64: 'TODO-oracle'" in p for p in problems), problems
    assert any("domain vmap: 'TODO-fix'" in p for p in problems), problems
    # a pending cell carries no test, so nothing is collected for it
    assert domain_targets(row) == [("f32", _T)]


# --------------------------------------------------------------------------
# Self-tests of the server domain set
# --------------------------------------------------------------------------
def _srow(conditions="k", **changes):
    cells = {d: NOT_APPLICABLE for d in SERVER_DOMAINS}
    cells.update(changes)
    return _row(id="REST-900", conditions=conditions, domains=cells)


@pytest.mark.parametrize("text, names", [
    ("a non-loopback bind", {"non_loopback_bind"}),
    ("a loopback bind and a non-loopback bind", {"loopback_bind", "non_loopback_bind"}),
    ("without and with the token", {"token_enforced"}),
    ("loopback bind with no token", {"loopback_bind", "no_token"}),
    ("every runner replaced by a recorder; the runner's source", set()),
    ("the GitHub runners have it", set()),
    ("a checkpoint save slowed to 1.5 s", set()),
    ("a checkpoint saved at a known clock", {"checkpoint_restore"}),
    ("beside a /sim/run whose first slice is held open", {"sim_run_active"}),
    ("GET /graph/state, /sim/run, /sim/reset", set()),
    ("the surrogate routes are out of scope", set()),
    ("surrogate characters", {"hostile_input"}),
    ("virtual CPU devices", set()),
    ("--goal all --dry-run", {"dry_run_cpu"}),
    ("the relabelled record with one goal from another commit",
     {"relabelled_records", "mixed_commits"}),
    ("sharded wrappers; a HybridNode", {"wrapper_nodes"}),
    ("SIGINT to a uvicorn process", {"shutdown"}),
    ("8 threads x 25 steps", {"concurrent"}),
])
def test_the_server_spellings_name_each_domain_and_only_it(text, names):
    """What a REST condition names: the specific spelling first ("non-
    loopback" before "loopback", "without the token" before "with the
    token"), and run_pod's goal runners, a checkpoint save, a route list and
    the surrogate routes name nothing."""
    assert domain_mentions(text, "server") == names


@pytest.mark.parametrize("row, fragment", [
    (_srow("a non-loopback bind"), "domain non_loopback_bind is n/a, but its conditions cover it"),
    (_srow("a non-loopback bind", non_loopback_bind=_T),
     "domain token_enforced is n/a, but its conditions cover it"),
    (_srow("any bind", loopback_bind=_T, non_loopback_bind=_T, token_enforced=_T),
     "domain no_token is n/a, but its conditions cover it"),
    (_srow("k. Not claimed for simultaneous requests."), "its conditions exclude concurrent"),
    (_srow("k", concurrent=NARROWED), "domain concurrent is narrowed, but its conditions do not"),
    (_srow("k. Not claimed for the weather.", shutdown=NARROWED), "names no domain"),
    (_srow("k. Not claimed for float64.", shutdown=NARROWED), "names no domain"),
    (_row(id="REST-900", domains={d: NOT_APPLICABLE for d in NUMERIC_DOMAINS}),
     "says nothing about"),
], ids=["named-but-na", "bind-implies-token", "any-bind", "excluded-not-narrowed",
        "narrowed-not-excluded", "empty-clause", "numeric-word-in-a-server-clause",
        "numeric-matrix-in-a-server-file"])
def test_the_domain_rule_fires_on_each_defect_of_a_server_matrix(row, fragment):
    problems = row_problems([row], ["REST"], domains=SERVER_DOMAINS)
    assert any(fragment in p for p in problems), problems


def test_the_domain_rule_passes_a_filled_server_matrix():
    narrowed = ("loopback_bind", "non_loopback_bind", "token_enforced", "no_token",
                "concurrent", "runner_active", "sim_run_active", "large_payload",
                "hostile_input", "shutdown", "wrapper_nodes", "checkpoint_restore",
                "dry_run_cpu", "relabelled_records", "mixed_commits")
    clause = ("k. Not claimed for a loopback bind, a non-loopback bind, a request the token is "
              "demanded of, a request no token is demanded of, simultaneous requests, the "
              "runner running, an in-flight /sim/run, oversized requests, hostile input, a "
              "server shutting down, wrapper nodes, a graph restored from a checkpoint, dry "
              "runs, relabelled records or mixed commits.")
    assert row_problems([_srow(clause, **{d: NARROWED for d in narrowed})], ["REST"],
                        domains=SERVER_DOMAINS) == []
    assert row_problems([_srow("a loopback bind", loopback_bind=_T, no_token=_T)], ["REST"],
                        domains=SERVER_DOMAINS) == []
    # a numeric file reads "checkpoint" as checkpoint_restart, never the server's word
    assert domain_mentions("a checkpoint", "numeric") == {"checkpoint_restart"}
    assert domain_mentions("a checkpoint", "server") == {"checkpoint_restore"}


def test_the_witness_rule_reads_the_server_vocabulary():
    # concurrent: a call that starts a server, and requests that meet on it
    helper = ("def loopback_server(chk, root):\n    server = make_server('concurrent', chk, root)\n"
              "    userver = _uvicorn().Server(config)\n"
              "    thread = threading.Thread(target=userver.run)  # uvicorn, in a thread\n")
    served = "def test_c():\n    with S.loopback_server(chk, root) as s:\n        %s\n" + helper
    real = served % "simultaneously([job] * 4)"
    in_process = "def test_c():\n    threads = [threading.Thread(target=TestClient(app).get)]"
    assert witnesses("concurrent", real)
    assert witnesses("concurrent", served % "threading.Thread(target=lambda: get(s)).start()")
    assert witnesses("concurrent", "def test_c():\n    Popen([_SERVER]); threading.Thread(target=run)"
                                   "\n_SERVER = 'uvicorn.run(app, port=8000)'")
    assert not witnesses("concurrent", in_process), "TestClient threads are no real server"
    # ... which a URL, a WebSocket, a client named for loopback and the word
    # "uvicorn" do not make them
    assert not witnesses("concurrent", in_process + "\n    TestClient(app, 'http://127.0.0.1')"
                         ".websocket_connect('/ws')  # as uvicorn's proxy_headers would\n"
                         "LOOPBACK_BASE_URL = 'http://127.0.0.1'\nsocket.socket()")
    # ... nor a docstring that quotes the call
    assert not witnesses("concurrent", in_process.replace(
        "\n", '\n    """``uvicorn.run(app, host="0.0.0.0")`` and ``_uvicorn().Server(config)``'
        ' never tell the app."""\n', 1))
    # ... and a server's own thread and domain tag are no simultaneous requests
    assert not witnesses("concurrent", served % "httpx.get(s)  # one request at a time")
    assert not witnesses("concurrent", "def test_c():\n    uvicorn.run(app)  # one request")
    # loopback and no token are the defaults: anything but a test that only
    # runs on another bind / always presents the token
    assert witnesses("loopback_bind", "def test_a(): client.get('/graph')")
    assert not witnesses("loopback_bind", "def test_a(): _server('0.0.0.0')")
    assert witnesses("loopback_bind", "def test_a(): _server('0.0.0.0'); _server('127.0.0.1')")
    assert witnesses("no_token", "def test_a(): client.get('/graph')")
    assert not witnesses("no_token", "def test_a(): c.get('/', headers={'Authorization': x})")
    assert witnesses("no_token", "def test_an_anonymous_caller(): headers={'Authorization': x}")
    assert witnesses("token_enforced", "def test_a(): SimulationServer(bind_host='0.0.0.0')")
    assert not witnesses("token_enforced", "def test_a(): client.get('/graph')")
    for domain, yes, no in (
            ("runner_active", "client.post('/sim/start')", "client.post('/sim/step')"),
            ("sim_run_active", "client.post('/sim/run')", "client.post('/sim/step')"),
            ("shutdown", "signal.raise_signal(signal.SIGTERM)", "client.post('/sim/stop')"),
            ("wrapper_nodes", "HybridNode(spring, f)", "SpringDamperNode('s', 0.01)"),
            ("checkpoint_restore", "post('/checkpoint/load')", "post('/sim/reset')"),
            ("dry_run_cpu", "main(['--goal', 'all', '--dry-run'])", "rp.recommend(docs)"),
            ("relabelled_records", "_as_real_gpu_run(recorded)  # relabelled", "docs"),
            ("mixed_commits", "doc['environment']['git_commit'] = 'f' * 40", "doc['goal']"),
            ("large_payload", "MAX_REQUEST_BODY_BYTES", "client.get('/graph')"),
            ("hostile_input", "Origin: evil.example  # foreign", "client.get('/graph')")):
        assert witnesses(domain, f"def test_x():\n    {yes}"), domain
        assert not witnesses(domain, f"def test_x():\n    {no}"), domain
    assert {d for d in SERVER_DOMAINS} <= KNOWN_DOMAINS
    # ... and the rule asks it of every server cell, as of every numeric one
    row = _srow(concurrent="tests/api/test_x.py::test_c", shutdown="tests/api/test_x.py::test_s")
    sources = {"tests/api/test_x.py::test_c": in_process,
               "tests/api/test_x.py::test_s": "def test_s(): client.post('/sim/step')"}
    problems = domain_witness_problems([row], source=sources.get)
    assert any("domain concurrent" in p for p in problems), problems
    assert any("domain shutdown" in p for p in problems), problems
    sources["tests/api/test_x.py::test_c"] = real
    sources["tests/api/test_x.py::test_s"] = "def test_s(): signal.raise_signal(SIGTERM)"
    assert domain_witness_problems([row], source=sources.get) == []


def test_the_shared_in_process_client_is_no_tests_own_words(tmp_path, monkeypatch):
    """The client every REST test constructs names a loopback Host and peer.
    Read as a test's own words it witnesses a loopback bind, a tokenless
    request and hostile input for a test of none of them (and its URL once
    read as a real server), so neither its source nor the line that imports
    it is read -- here a test of a non-loopback bind that always presents
    the token, from threads, through the real client."""
    client = "tests/_loopback_client.py"
    assert client in TRANSPORT_MODULES and (REPO_ROOT / client).is_file()
    (tmp_path / "tests" / "api").mkdir(parents=True)
    (tmp_path / client).write_text((REPO_ROOT / client).read_text(encoding="utf-8"),
                                   encoding="utf-8")
    (tmp_path / "tests" / "api" / "test_z.py").write_text(
        "import threading\n"
        "from tests._loopback_client import LoopbackTestClient as TestClient\n\n\n"
        "def test_public():\n"
        "    from tests._loopback_client import (\n        LoopbackTestClient as Client,\n    )\n"
        "    app = SimulationServer({}, bind_host='0.0.0.0').create_app()\n"
        "    Client(app, headers={'Authorization': 'Bearer t'}).get('/graph')\n"
        "    jobs = [threading.Thread(target=TestClient(app).get) for _ in range(8)]\n",
        encoding="utf-8")
    nodeid = "tests/api/test_z.py::test_public"
    text = source_of_test(nodeid, repo_root=tmp_path)
    assert "bind_host='0.0.0.0'" in text and "Loopback" not in text and "127.0.0.1" not in text
    for domain in ("loopback_bind", "no_token", "hostile_input", "concurrent"):
        assert not witnesses(domain, text), domain
    assert witnesses("non_loopback_bind", text) and witnesses("token_enforced", text)
    # What that keeps out: the same test, the client read as its own words.
    monkeypatch.setattr(sys.modules[__name__], "TRANSPORT_MODULES", frozenset())
    read = source_of_test(nodeid, repo_root=tmp_path)
    assert "LOOPBACK_BASE_URL" in read
    for domain in ("loopback_bind", "no_token", "hostile_input"):
        assert witnesses(domain, read), domain


# --------------------------------------------------------------------------
# Self-tests: tests that need an optional extra
# --------------------------------------------------------------------------
_U = "tests/usd/test_x.py::test_a"


def test_only_the_usd_files_are_read_from_source_and_only_without_usd_core():
    files = ["tests/core/test_x.py", "tests/usd/test_x.py", "tests/usd_like/test_x.py"]
    assert split_by_lane(files, usd_core=True) == (files, [])
    assert split_by_lane(files, usd_core=False) == (
        ["tests/core/test_x.py", "tests/usd_like/test_x.py"], ["tests/usd/test_x.py"])


def test_the_lane_rule_wants_a_usd_test_in_the_usd_jobs_selection():
    row = _row(tests=[_T, _U])
    for selection in (["tests/usd/", "tests/property/test_round_trips.py"], ["tests/usd"],
                      ["tests/usd/test_x.py"], [_U], ["tests/usd/test_x.py::test_b", _U]):
        assert lane_problems([row], selection) == [], selection
    for selection in ([], ["tests/property/test_round_trips.py"], ["tests/usd/test_y.py"],
                      ["tests/usd/test_x.py::test_b"], ["tests/usd_like/"], ["tests/core/"]):
        problems = lane_problems([row], selection)
        assert len(problems) == 1 and f"{_U} -- needs usd-core" in problems[0], selection
    # a class the job selects runs its methods; a domain cell's test is read too
    in_class = _row(tests=["tests/usd/test_x.py::TestK::test_a"])
    assert lane_problems([in_class], ["tests/usd/test_x.py::TestK"]) == []
    assert lane_problems([in_class], ["tests/usd/test_x.py::TestJ"])
    assert lane_problems([_drow(f32=_T, jit=_U)], [])
    # a test that needs no extra is not this rule's
    assert lane_problems([_GOOD], []) == []
    # ... and the shipped job runs the directory
    assert lane_problems([row], _usd_job_targets()) == []


_MARK_SPELLINGS = '''
import pytest

slow = pytest.mark.slow
quiet = pytest.mark.filterwarnings("ignore")


def helper():
    pass


@pytest.fixture
def thing():
    return 1


def test_plain(thing):
    pass


@pytest.mark.slow
def test_slow():
    pass


@slow
def test_slow_by_name():
    pass


@quiet
def test_a_mark_the_rules_do_not_read():
    pass


@pytest.mark.skip(reason="never")
def test_skipped():
    pass


@pytest.mark.skipif(False, reason="conditional")
def test_a_skipif_is_not_a_skip_mark():
    pass


@pytest.mark.xfail(strict=True, reason="SYS-900: broken; pending fix")
def test_strict_xfail():
    assert False


@pytest.mark.xfail(reason="loose")
def test_loose_xfail():
    assert False


@pytest.mark.xfail
def test_bare_xfail():
    assert False


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.slow
def test_slow_parametrised(n):
    pass


@pytest.mark.parametrize("n", [1, 2])
def test_parametrised(n):
    pass


@pytest.mark.slow
class TestMarked:
    def test_inherits(self):
        pass

    @pytest.mark.xfail(strict=True, reason="CPL-900: x")
    def test_adds(self):
        assert False

    def helper(self):
        pass


class TestPlain:
    def test_method(self):
        pass

    @pytest.mark.skip
    def test_bare_skip(self):
        pass


class NotCollected:
    def test_method(self):
        pass
'''
_MODULE_MARKS = '''
import pytest

pytestmark = [pytest.mark.filterwarnings("ignore"),
              pytest.mark.xfail(strict=True, reason="REST-900: y")]


def test_under_the_module_mark():
    assert False


@pytest.mark.xfail(reason="its own")
def test_with_a_closer_mark():
    assert False
'''
_ONE_MODULE_MARK = "import pytest\n\npytestmark = pytest.mark.slow\n\n\ndef test_a():\n    pass\n"


def _marks_by_function(items: list[Item]) -> dict[str, set[tuple]]:
    """``{function node id: the (slow, skip, xfail, strict, reason) of its items}``."""
    out: dict[str, set[tuple]] = {}
    for it in items:
        out.setdefault(it.nodeid.split("[", 1)[0], set()).add(
            (it.slow, it.skip, it.xfail, it.strict, it.reason))
    return out


def test_the_source_reader_agrees_with_pytest_on_every_mark_it_reads(tmp_path):
    """One module of every spelling, collected by pytest and read from its
    source: the same test functions, each with the same marks.  This is
    what lets the reader stand in for pytest where a module cannot be
    imported."""
    sample = tmp_path / "tests" / "sample"
    sample.mkdir(parents=True)
    texts = {"test_spellings.py": _MARK_SPELLINGS, "test_module_marks.py": _MODULE_MARKS,
             "test_one_module_mark.py": _ONE_MODULE_MARK}
    for name, text in texts.items():
        (sample / name).write_text(text, encoding="utf-8")
    files = sorted(f"tests/sample/{name}" for name in texts)
    collected = _collect(files, tmp_path / "items.json", cwd=tmp_path)
    read = [it for f in files for it in items_from_source(f, repo_root=tmp_path)]
    assert _marks_by_function(read) == _marks_by_function(collected)
    # ... and the sample holds every case: each mark, and a test with none
    flags = {flag for marks in _marks_by_function(read).values() for flag in marks}
    assert {f[:4] for f in flags} == {
        (False, False, False, False), (True, False, False, False), (False, True, False, False),
        (False, False, True, True), (False, False, True, False), (True, False, True, True)}
    assert len(read) == 18 and len(collected) == 20   # two tests of two parameters


def test_a_cited_parameter_is_covered_when_its_function_is_found(tmp_path):
    (tmp_path / "tests" / "usd").mkdir(parents=True)
    (tmp_path / "tests" / "usd" / "test_x.py").write_text(
        "import pytest\n\n@pytest.mark.slow\n@pytest.mark.parametrize('n', [1])\n"
        "def test_a(n):\n    pass\n", encoding="utf-8")
    cited = [f"{_U}[1]", "tests/usd/test_x.py::test_gone[1]", "tests/core/test_x.py::test_a[1]"]
    items = items_from_source("tests/usd/test_x.py", cited, repo_root=tmp_path)
    assert items == [Item(_U, True, False, False, False, ""),
                     Item(f"{_U}[1]", True, False, False, False, "")]
    assert items_from_source("tests/usd/test_gone.py", cited, repo_root=tmp_path) == []


def test_every_rule_fires_on_items_read_from_source(tmp_path):
    """What a lane without usd-core is left with: a row citing a USD test
    that does not exist, never runs, is slow with no witness or disagrees
    with its status still fails there."""
    (tmp_path / "tests" / "usd").mkdir(parents=True)
    (tmp_path / "tests" / "usd" / "test_x.py").write_text(
        "import pytest\n\n\ndef test_a():\n    pass\n\n\n"
        "@pytest.mark.skip(reason='x')\ndef test_never():\n    pass\n\n\n"
        "@pytest.mark.slow\ndef test_slow():\n    pass\n\n\n"
        "@pytest.mark.xfail(strict=True, reason='CPL-900: broken; pending fix')\n"
        "def test_broken():\n    assert False\n", encoding="utf-8")
    items = items_from_source("tests/usd/test_x.py", repo_root=tmp_path)

    def cites(name, **changes):
        return [_row(tests=[f"tests/usd/test_x.py::{name}"], **changes)]

    assert collection_problems(cites("test_a"), items) == []
    assert any("collects no such test" in p for p in collection_problems(cites("test_b"), items))
    assert any("skip-marked" in p for p in collection_problems(cites("test_never"), items))
    assert any("no '# Per push" in p for p in collection_problems(
        cites("test_slow"), items, witness=lambda f: False))
    assert any("verified, but cites the xfail" in p
               for p in xfail_problems(cites("test_broken"), items))
    failing = dict(status="failing", finding="f")
    assert xfail_problems(cites("test_broken", **failing), items) == []
    assert any("cites no strict xfail" in p for p in xfail_problems(cites("test_a", **failing), items))
    assert any("which does not cite it" in p for p in xfail_problems(cites("test_a"), items))
    ghost = [dataclasses.replace(it, reason="CPL-777: x") for it in items if it.xfail]
    assert any("no inventory holds" in p for p in unowned_xfail_problems([_inv()], ghost))


def test_one_ungated_job_runs_this_module_with_usd_core():
    """What holds the source reader to pytest's answer on every push: the
    ``compliance`` job installs the ``usd`` extra, runs all of
    ``tests/compliance/`` and is gated on nothing a push changed.  Without
    it the comparison below would be skipped in every job."""
    job = _workflow("ci.yml")["jobs"]["compliance"]
    assert "if" not in job and "needs" not in job, "the compliance job is gated"
    scripts = [s.get("run", "") for s in job["steps"]]
    assert any('pip install -e ".[ci,usd]"' in _logical_lines(script) for script in scripts), \
        "the compliance job no longer installs the usd extra"
    runs = [shlex.split(line) for script in scripts for line in _logical_lines(script)
            if "-m pytest" in line]
    assert len(runs) == 1, runs
    args = runs[0][runs[0].index("pytest") + 1:]
    assert not any(a in ("-m", "-k") or a.startswith(("-m=", "-k=", "--deselect", "--ignore"))
                   for a in args), f"the compliance job narrows its selection: {args}"
    assert [a for a in args if not a.startswith("-")] == ["tests/compliance/"], args


def test_the_source_reader_agrees_with_pytest_on_the_cited_usd_tests(collection):
    """Where usd-core is installed, ``collection`` holds pytest's own items
    for the cited files under ``tests/usd/``.  The source reader, which
    stands in for pytest in the lanes without it, must give each of those
    test functions the same marks -- so a mark spelled in a way the reader
    does not know fails here, on every push, rather than passing unread
    there."""
    if not usd_core_installed():
        pytest.skip("usd-core is not installed, so pytest cannot collect tests/usd here and "
                    "there is nothing to compare the source reader with; the compliance job "
                    "installs the usd extra and runs this comparison on every push")
    files = [f for f in cited_files(every_row(load_all())) if f.startswith(USD_TESTS)]
    for rel in files:
        collected = [it for it in collection if it.nodeid.startswith(rel + "::")]
        assert collected, f"{rel}: cited, but pytest collected nothing from it"
        assert _marks_by_function(items_from_source(rel)) == _marks_by_function(collected), rel
