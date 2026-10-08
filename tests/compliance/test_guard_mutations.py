"""Every CI guard must fail when the thing it guards is broken.

The CI lanes, the per-test time budget, the shard split, the JAX timing
plugin, the compilation-cache pruner and the duration allowlist are each
pinned by guard tests (``tests/compliance/test_ci_workflows.py``,
``test_ci_sharding.py``, ``test_report_test_durations.py``,
``test_prune_jax_cache.py`` and ``tests/core/test_compile_cache.py``), and
the rule that a slow mark names its per-push witness by
``tests/compliance/test_slow_only_rule.py``.  A
guard that still passes once the workflow, script or config it reads is
broken protects nothing, and nothing else notices: CI stays green either
way.  Audits found such gaps by hand, by seeding faults and checking
whether any guard failed.  This module does that mechanically.

Each :class:`Mutant` in :data:`MUTANTS` is one seeded fault: replace
``anchor`` with ``replacement`` in ``file``.  The slow test
:func:`test_every_seeded_ci_fault_fails_a_guard` unpacks ``git archive
HEAD`` into a fresh temporary directory for each mutant (never the working
tree), applies the edit there, and runs the mutant's guard tests against
that copy, with the copy's own ``src`` on ``PYTHONPATH`` and the CI-only
environment variables removed.  The mutant is *caught* when pytest exits 1
(at least one guard test failed).  Exit 0 means it *survived*.  Any other
exit (a usage error, an internal error, nothing collected) is a broken run
and fails the test: it is never counted as a catch.

Expectations:

* A plain mutant must be caught.
* ``equivalent="<why>"`` marks a mutant that changes nothing CI relies on.
  It is asserted to *survive*, so if a guard starts catching it, the test
  fails and the entry has to be moved deliberately.
* ``gap="<what is unguarded>"`` marks a live guard gap: a mutant that should
  be caught and is not.  It runs as ``xfail(strict=True)``, so closing the
  gap fails the test until the marker is removed.

The per-push sibling, :func:`test_every_mutant_anchor_matches_the_tree_exactly_once`,
checks the table against the working tree in a few milliseconds: every
anchor must occur exactly once in its file, and every guard file and test
must exist.  Without it, a refactor that moves an anchor would turn a
mutant into a no-op that only the next slow-lane run could notice.

Adding a mutant is one entry in :data:`MUTANTS`; see "Guard mutations" in
``docs/developer_guide/testing_standards.md``.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# Guard selections: pytest arguments, run in order with ``-x``, so the file
# most likely to catch a mutant goes first.
# --------------------------------------------------------------------------
_DURATIONS = "tests/compliance/test_report_test_durations.py"
_COLLECTS = f"{_DURATIONS}::test_every_shipped_allowlist_entry_is_a_node_id_pytest_collects"
_REASONS = f"{_DURATIONS}::test_the_shipped_allowlist_names_real_tests_with_reasons"
_SHARDING = "tests/compliance/test_ci_sharding.py"
_WORKFLOWS = "tests/compliance/test_ci_workflows.py"
_PRUNE = "tests/compliance/test_prune_jax_cache.py"
_COMPILE_CACHE = "tests/core/test_compile_cache.py"
_SLOW_RULE = "tests/compliance/test_slow_only_rule.py"

#: The budget script and the timing plugin.  The allowlist-collection test
#: is left out: it collects the whole tree (seconds), and only the allowlist
#: mutants need it.
DUR = (_DURATIONS, "--deselect", _COLLECTS)
SHARD = (_SHARDING,)
WF = (_WORKFLOWS,)
PRUNE = (_PRUNE,)
CC = (_COMPILE_CACHE,)
SLOW_RULE = (_SLOW_RULE,)
_CLAIMS = "tests/compliance/test_claims_inventories.py"
CLAIMS = (_CLAIMS,)
_PLAN_SCAN = "tests/core/test_interface_plan_is_the_only_enumeration.py"
PLAN_SCAN = (_PLAN_SCAN,)
#: The witness rule alone (no collection of the tree), and its self-tests.
WITNESS = (f"{_CLAIMS}::test_every_tested_domain_cites_a_test_that_names_its_domain",)
SERVER_WORDS = (f"{_CLAIMS}::test_the_witness_rule_reads_the_server_vocabulary",)
TRANSPORT = (f"{_CLAIMS}::test_the_shared_in_process_client_is_no_tests_own_words",)
CPL = "docs/validation/coupling_claims.yaml"
RST = "docs/validation/rest_runpod_claims.yaml"
SYS = "docs/validation/sysid_fmu_claims.yaml"
MAP = "docs/validation/mapping_claims.yaml"
#: The shipped allowlist: the reason check (fast) first, then collection.
ALLOW = (_REASONS, _COLLECTS)
#: Workflows, the root conftest and the pytest configuration.
CI_ALL = DUR + SHARD + WF

#: Every guard file, run whole on the unmutated copy before any mutant.
GUARD_FILES = (_DURATIONS, _SHARDING, _WORKFLOWS, _PRUNE, _COMPILE_CACHE, _SLOW_RULE, _PLAN_SCAN)


@dataclass(frozen=True)
class Mutant:
    """One seeded fault: replace ``anchor`` (exactly once) in ``file``."""

    id: str
    file: str
    anchor: str
    replacement: str
    guards: tuple[str, ...]
    lets_through: str
    equivalent: str = ""
    gap: str = ""


_M = Mutant
RTD = "scripts/report_test_durations.py"
SH = "tests/_sharding.py"
CF = "tests/conftest.py"
JT = "tests/_jax_timing.py"
PR = "scripts/prune_jax_cache.py"
CI = ".github/workflows/ci.yml"
SL = ".github/workflows/slow-tests.yml"
SWT = "tests/slow_lane_weights.json"
SLW = "scripts/slow_lane_weights.py"
PP = "pyproject.toml"
AL = "tests/duration_allowlist.txt"
TC = "tests/core/test_compile_cache.py"
SR = _SLOW_RULE
TS = "docs/developer_guide/testing_standards.md"
_LM = "test_fit_lm_through_ift_coupled_group_recovers_stiffness_from_nearby"
_XF = "tests/core/test_cross_feature_edge_cases.py"

# Lines reused by several allowlist mutants (an existing entry, and a real
# parametrised test).
_AL_LAST = ("tests/core/test_compile_counts.py::"
            "test_the_committed_baseline_matches_what_the_code_compiles_to # kept")
_KEPT = ("tests/core/test_coupling_diagnostics_leave_the_state_alone.py::"
         "test_diagnostics_leave_the_returned_state_bit_identical")
_ZMQ = "tests/security/test_zmq_transport_auth.py"
_API = "tests/api/"
_BEARER = f"{_API}test_bearer_auth.py::"
_MG = "tests/cloud/multigpu/"
_EDGES = f"{_MG}test_run_pod_documented_edges.py::"
_VERDICT = f"{_MG}test_run_pod_verdict_integrity.py::"
#: A real uvicorn server on loopback, spoken to one connection at a time.
_CLEARTEXT = (f"{_API}test_rest_claims_at_their_domain_edges.py::"
              "test_the_server_speaks_plain_http_and_the_token_crosses_in_cleartext[127.0.0.1]")
#: REST-040's tokenless and concurrent cells (the tokenless one makes the anchor unique).
_STEPS_CELLS = ("    no_token: tests/api/test_concurrent_requests_are_serialised.py::"
                "test_concurrent_steps_are_all_taken_bit_identical_to_serial\n"
                "    concurrent: %s\n")
_STEPS_SERVED = ("tests/api/test_rest_claims_under_concurrent_requests.py::"
                 "test_simultaneous_steps_are_all_taken_bit_identical_to_serial")


def _cell(domain: str, test: str) -> str:
    """One domain cell of a claims row, as the inventories write it."""
    return f"    {domain}: {test!r}\n" if "[" in test else f"    {domain}: {test}\n"


def _wrong_cell(id: str, domain: str, cited: str, wrong: str, lets_through: str) -> Mutant:
    """A REST or run_pod row's *domain* cell made to cite *wrong*, a test of another domain."""
    return _M(id, RST, _cell(domain, cited), _cell(domain, wrong), WITNESS, lets_through)


# A test under tests/usd that a claims row cites.
_USD_SPEC = "tests/usd/test_usd_mapping_spec.py"
_USD_GONE = "test_a_stage_written_with_a_kind_that_is_no_longer_registered_is_refused"

MUTANTS: tuple[Mutant, ...] = (
    # --- R: the time budget, scripts/report_test_durations.py --------------
    _M("R1", RTD, '    return 1 if verdict["failing"] else 0', "    return 0", DUR,
       "the budget never fails a job"),
    _M("R2", RTD, '"failing": [t for t in ranked if t.seconds > fail_over and t.nodeid not in allow],',
       '"failing": [t for t in ranked if t.seconds > fail_over],', DUR,
       "allowlisted tests fail the job anyway (allowlist ignored)"),
    _M("R3", RTD, '"failing": [t for t in ranked if t.seconds > fail_over and t.nodeid not in allow],',
       '"failing": [t for t in ranked if t.seconds > fail_over'
       ' and not any(t.nodeid.startswith(a) for a in allow)],', DUR,
       "an entry exempts every test it is a prefix of (parametrised ids, whole files)"),
    _M("R4", RTD, "if t.nodeid not in by_id or t.seconds > by_id[t.nodeid].seconds:",
       "if t.nodeid not in by_id or t.seconds < by_id[t.nodeid].seconds:", DUR,
       "a test is judged on its fastest report"),
    _M("R5", RTD, 'p.add_argument("--fail-over", type=float, default=20.0,',
       'p.add_argument("--fail-over", type=float, default=200.0,', DUR, "hard line moved to 200 s"),
    _M("R6", RTD, "        if not tests:\n            raise ReportError(",
       "        if False:\n            raise ReportError(", DUR, "an empty report passes"),
    _M("R7", RTD, "broken = [r for r, cases in reports if cases and all(t.collection_error for t in cases)]",
       "broken = []", DUR, "a collection-error-only report (no test ran) passes"),
    _M("R8", RTD, 'parts = [file, *[c for c in classes.split(".") if c], name]', "parts = [file, name]", DUR,
       "class tests get the wrong node id (the allowlist never matches them)"),
    _M("R9", RTD, "    if not file:\n        raise ReportError(",
       "    if not file:\n        file = 'unknown.py'\n    if False:\n        raise ReportError(", DUR,
       "an xunit2 report (no file attribute) is judged with made-up node ids"),
    _M("R10", RTD, "    if args.no_removable:\n", "    if False:\n", DUR,
       "--no-removable ignored: a partial lane lists entries as removable"),
    _M("R11", RTD, 'if removable and (cache_mode in ("warm", "mixed") or any(s.restored for s in shards)):',
       "if removable and False:", DUR, "a warm run lists allowlist entries as removable"),
    _M("R12", RTD, 'and t.outcome == "passed"]', "]", DUR,
       "a skipped or failed allowlisted test is called removable"),
    _M("R13", RTD, "self.hits <= WARM_HIT_RATE * self.lookups", "self.hits < WARM_HIT_RATE * self.lookups", DUR,
       "a shard whose lookups hit 50% is called warm"),
    _M("R14", RTD, "if mode_file is not None and mode_file.is_file():", "if False:", DUR,
       "per-shard cache-mode files ignored"),
    _M("R15", RTD, "        if not reason.strip():\n            raise ReportError(",
       "        if False:\n            raise ReportError(", DUR, "an allowlist entry without a reason is accepted"),
    _M("R16", RTD, "            return started > 0", "            return False", DUR,
       "subprocess tests are called slow even with a warm cache"),
    _M("R17", RTD, 'return max(0.0, self.jax["jax_compile_s"] - self.cache_read) if self.jax else 0.0',
       'return self.jax["jax_compile_s"] if self.jax else 0.0', DUR,
       "cache reads counted as compile (and twice in running)"),
    _M("R18", RTD, '"failing": [t for t in ranked if t.seconds > fail_over and',
       '"failing": [t for t in ranked if t.seconds >= fail_over and', DUR,
       "an unlisted test timed at exactly 20.000 s fails",
       equivalent="moves only the boundary point: JUnit records milliseconds, so a test at exactly "
                  "the hard line is a measure-zero event and no guard pins which side it falls on"),
    _M("R19", RTD, 'return [s for s in shards if s.claimed == "off" and s.looked_up]', "return []", DUR,
       "a cache left switched on in the slow lane is not flagged"),
    _M("R20", RTD, 'fail_over = args.fail_over or float("inf")', "fail_over = args.fail_over or 1e-9", DUR,
       "--fail-over 0 fails every test instead of reporting only"),
    _M("R21", RTD, 'collection_error=(outcome == "error" and detail is not None\n'
                   '                              and detail.get("message") == COLLECTION_FAILURE),',
       "collection_error=False,", DUR, "collection errors not recognised"),
    _M("R22", RTD, "line=int(line) + 1 if line is not None else None,",
       "line=int(line) if line is not None else None,", DUR, "annotations point one line off"),
    _M("R23", RTD, '"failing": [t for t in ranked if t.seconds > fail_over and t.nodeid not in allow]',
       '"failing": [t for t in ranked if t.seconds > 2 * fail_over and t.nodeid not in allow]', DUR,
       "the hard line is doubled to 40 s"),
    _M("R24", RTD, 'cache_mode in ("warm", "mixed")', 'cache_mode in ("warm",)', DUR,
       "a mixed-cache lane lists allowlist entries as removable"),
    _M("R25", RTD, '"slow_warm": [t for t in ranked if t.jax and t.uncacheable > slow_over',
       '"slow_warm": [t for t in ranked if t.jax and t.seconds > slow_over', DUR,
       "'slow even with a warm cache' counts the compile a warm cache removes"),
    _M("R26", RTD, '            seconds=float(case.get("time") or 0.0), outcome=outcome,',
       '            seconds=float(int(float(case.get("time") or 0.0))), outcome=outcome,', DUR,
       "times truncated to whole seconds: an unlisted 20.9 s test passes the hard line, 5.9 s escapes the warning"),
    _M("R27", RTD, '        "allow_fast": [t for t in ranked if t.nodeid in allow and t.seconds <= slow_over',
       '        "allow_fast": [t for t in ranked if t.nodeid in allow and t.seconds <= fail_over', DUR,
       "a kept test that passed at 6-20 s is offered as a removable allowlist entry"),
    # --- S: the shard split, tests/_sharding.py ------------------------------
    _M("S1", SH, 'return int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % n + 1',
       'return int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % n', SHARD,
       "0-based shards: every file hashing to 0 runs nowhere"),
    _M("S2", SH, "    if n == PINS_FOR and path in PINS:", "    if False:", SHARD, "pins ignored"),
    _M("S3", SH, "    if n == PINS_FOR and path in PINS:", "    if path in PINS:", SHARD,
       "pins leak into other job counts"),
    _M("S4", SH, "(keep if self.shard_of(path) == self.shard else drop).append(item)", "keep.append(item)",
       SHARD, "every shard runs the whole suite"),
    _M("S5", SH, 'path = item.nodeid.split("::", 1)[0]', "path = item.nodeid", SHARD,
       "a file's tests split across shards (hashed by node id)"),
    _M("S6", SH, "        if not 1 <= pinned <= PINS_FOR:", "        if False:", SHARD,
       "a pin to a shard that does not exist is accepted at registration",
       equivalent="defence in depth: test_every_pin_names_a_real_file_and_a_real_shard checks every "
                  "shipped pin directly, so an out-of-range pin is still caught (mutant S8)"),
    _M("S7", SH, "    if not 1 <= i <= n:", "    if not 0 <= i <= n:", SHARD, "MADDENING_TEST_SHARD=0/4 accepted"),
    _M("S8", SH, '"tests/core/test_coupling_while_default.py": 1,',
       '"tests/core/test_coupling_while_default.py": 5,', SHARD, "a file pinned to shard 5 of 4 runs nowhere"),
    _M("S9", SH, '"tests/core/test_coupling_while_default.py": 1,',
       '"tests/core/test_coupling_while_defaults.py": 1,', SHARD, "a pin naming a renamed file"),
    _M("S10", SH, 'return int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % n + 1',
       'return int(hashlib.md5(path.encode("utf-8")).hexdigest(), 16) % n + 1', SHARD,
       "every file re-dealt (every saved base cache is for the wrong files)"),
    _M("S11", SH, "        if drop:\n            config.hook.pytest_deselected(items=drop)\n", "        if drop:\n",
       SHARD, "deselected tests not reported as deselected"),
    _M("S12", SH, "            items[:] = keep", "            pass", SHARD,
       "deselection reported, but every shard still runs everything"),
    # --- S, continued: the slow lane's weighted split --------------------------
    _M("S13", SH, "    return dealt[path] if path in dealt else _by_hash(path, n)", "    return _by_hash(path, n)",
       SHARD, "the weighted split ignores its table: the slow lane is back on the hash split, one shard at "
              "its timeout"),
    _M("S14", SH, "    return dealt[path] if path in dealt else _by_hash(path, n)",
       "    return dealt[path] if path in dealt else shard_of(path, n)", SHARD,
       "the per-push pins leak into the weighted split"),
    _M("S15", SH, "        lightest = min(range(n), key=lambda k: (load[k], k))", "        lightest = 0", SHARD,
       "every measured file dealt to shard 1"),
    _M("S16", SH, "sorted(weights.items(), key=lambda kv: (-kv[1], kv[0])):",
       "sorted(weights.items(), key=lambda kv: (kv[1], kv[0])):", SHARD,
       "shortest first: the 48-minute file lands last, on top of a full shard"),
    _M("S17", SH, "    return i, n, bool(colon)", "    return i, n, False", SHARD,
       "the ':weighted' suffix is parsed and ignored"),
    _M("S18", SH, "    if colon and plan != WEIGHTED:", "    if False:", SHARD,
       "any suffix is taken for ':weighted', a misspelt one included"),
    _M("S19", SH, "    weights = load_weights() if weighted else None", "    weights = None", SHARD,
       "a weighted spec registers the hash split"),
    _M("S20", SH, "        if self.weighted:\n            return weighted_shard_of(path, self.of, self._dealt)\n",
       "", SHARD, "the plugin holds the dealt table and never reads it"),
    _M("S21", SH, "if type(v) is not int or v <= 0}", "if type(v) is not int or v < 0}", SHARD,
       "a zero weight accepted: the file is dealt as if it cost nothing"),
    _M("S22", SH, "    if not isinstance(seconds, dict) or not seconds:", "    if not isinstance(seconds, dict):",
       SHARD, "an empty table accepted: a weighted run that is a hash split"),
    _M("S23", SH, '    if not re.fullmatch(r"[0-9]+/[0-9]+", counts):', '    if not re.fullmatch(r".+/.+", counts):',
       SHARD, "a malformed spec dies in int() with a traceback instead of a usage error, or (' 1/4') is accepted"),
    # --- SW: the slow lane's table, tests/slow_lane_weights.json ----------------
    _M("SW1", SWT, '  "tests/property/test_sysid_truth_recovery.py": ',
       '  "tests/property/test_sysid_truth_recovered.py": ', SHARD,
       "the table's heaviest entry names a file that does not exist: its shard is 48 minutes lighter than dealt"),
    _M("SW2", SWT, ' "min_seconds": ', ' "min_seconds": 1000', SHARD,
       "the table claims a floor its entries are under"),
    _M("SW3", SWT, ' "runs": [', ' "runs": [], "earlier": [', SHARD, "weights with no CI run to trace them to"),
    # --- G: the table's generator, scripts/slow_lane_weights.py ----------------
    _M("G1", SLW, "            slowest[path] = max(slowest[path], seconds)",
       "            slowest[path] = seconds", SHARD, "a file's weight is whichever lane was read last"),
    _M("G2", SLW, "            merged[lane].update(files)", "            merged[lane] = {**files, **merged[lane]}",
       SHARD, "the first run's measurement wins over a later one"),
    _M("G3", SLW, "             if round(s) >= min_seconds and p not in gone}",
       "             if round(s) >= min_seconds}", SHARD, "a measured file that no longer exists stays in the table"),
    _M("G4", SLW, "            lanes[lane][path] += float(case.get(\"time\") or 0.0)",
       "            lanes[lane][path] = float(case.get(\"time\") or 0.0)", SHARD,
       "a file weighs what its last test took"),
    # --- C: the root conftest -------------------------------------------------
    _M("C1", CF, "    if spec:\n        from tests import _sharding", "    if False:\n        from tests import _sharding",
       CI_ALL, "MADDENING_TEST_SHARD ignored: every shard runs everything"),
    _M("C2", CF, '    if os.environ.get("MADDENING_TEST_JAX_TIMING") == "1":', "    if False:", CI_ALL,
       "JAX timing never recorded: no compile/trace split, no hit rates, warm labels unchecked"),
    # --- J: the JAX timing plugin, tests/_jax_timing.py -----------------------
    _M("J1", JT, '        if call.when == "teardown":', '        if call.when == "call":', DUR,
       "properties attached after the report is built (JUnit gets none)"),
    _M("J2", JT, "        self.timing.reset()", "        pass", DUR, "totals accumulate across tests"),
    _M("J3", JT, '    "os.posix_spawn", "os.spawn", "os.exec",', '    "os.spawn", "os.exec",', DUR,
       "posix_spawn children not counted"),
    _M("J4", JT, '"/jax/compilation_cache/cache_hits": "jax_cache_hits",',
       '"/jax/compilation_cache/cache_hit": "jax_cache_hits",', DUR, "cache hits read zero"),
    _M("J5", JT, '"/jax/core/compile/backend_compile_duration": "jax_compile_s",',
       '"/jax/core/compile/backend_compile": "jax_compile_s",', DUR, "compile time reads zero"),
    # --- P: the cache pruner, scripts/prune_jax_cache.py ----------------------
    _M("P1", PR, "    if len(payload) <= TIME_BYTES:", "    if False:", PRUNE, "header-only entries kept"),
    _M("P2", PR, '        except zlib.error as exc:\n            return f"zlib: {exc}"',
       "        except zlib.error as exc:\n            return None", PRUNE, "truncated zlib entries kept"),
    _M("P3", PR, "        path.unlink()", "        pass", PRUNE, "nothing deleted"),
    _M("P4", PR, "path.name.endswith(NOT_ENTRY_SUFFIXES)", "False", PRUNE, "atime files checked and deleted"),
    _M("P5", PR, '            return f"zstd: {exc}"', "            return None", PRUNE, "truncated zstd entries kept"),
    _M("P6", PR, "    if len(payload) <= TIME_BYTES:\n", "    if len(payload) < TIME_BYTES:\n", PRUNE,
       "an entry holding only the compile-time header is kept"),
    # --- Y: the default lanes, .github/workflows/ci.yml -----------------------
    _M("Y1", CI, '          MADDENING_TEST_SHARD: "${{ matrix.shard }}/4"\n        run: |',
       '          MADDENING_TEST_SHARD: "${{ matrix.shard }}/4"\n'
       '          PYTEST_ADDOPTS: "--deselect tests/core"\n        run: |', CI_ALL,
       "every shard silently drops tests/core (selection moved into the step's env)"),
    _M("Y2", CI, "        if: ${{ !cancelled() && github.event_name == 'push' && steps.tests.outcome == 'success' "
                 "&& steps.prune-cache.outcome == 'success' }}\n        uses: actions/cache/save@v4",
       "        if: ${{ !cancelled() && github.event_name != 'schedule' && steps.tests.outcome == 'success' "
       "&& steps.prune-cache.outcome == 'success' }}\n        uses: actions/cache/save@v4", CI_ALL,
       "pull requests save caches"),
    _M("Y3", CI, 'case "$msg" in *"[cold-ci]"*) mode=cold ;; esac',
       'case "$msg" in *"[cold-ci-off]"*) mode=cold ;; esac', CI_ALL, "a [cold-ci] request is ignored"),
    _M("Y4", CI, '-shard${{ matrix.shard }}of4" >> "$GITHUB_OUTPUT"', '" >> "$GITHUB_OUTPUT"', CI_ALL,
       "one compilation cache shared by all shards"),
    _M("Y5", CI, "mark\\.slow|^-", "mark\\.slowXX|^-", CI_ALL, "a slow-mark edit no longer forces a cold run"),
    _M("Y7", CI, '--pytest-arg=--tb=short \\\n            --pytest-arg=-m "--pytest-arg=slow or not slow"',
       "--pytest-arg=--tb=short", CI_ALL, "verify-hypothesis drops the slow-marked properties"),
    _M("Y8", CI, "      - name: Run tests\n        id: tests\n",
       "      - name: Run tests\n        id: tests\n        continue-on-error: true\n", CI_ALL,
       "a red test step leaves the job green (every failing test merges)"),
    _M("Y12", CI, 'if [ "$n" -ne 4 ]; then', 'if [ "$n" -lt 3 ]; then', CI_ALL,
       "a lane with three of four shard reports lists removable entries"),
    _M("Y14", CI, '          MADDENING_TEST_JAX_TIMING: "1"\n', "", CI_ALL, "JAX timing not recorded in CI"),
    _M("Y15", CI, "        shard: [1, 2, 3, 4]\n", "        shard: [1, 2, 3]\n", CI_ALL,
       "the default lane runs three of four shards"),
    _M("Y16", CI, "        shard: [1, 2, 3, 4]\n",
       "        shard: [1, 2, 3, 4]\n        exclude:\n          - shard: 4\n", CI_ALL,
       "the default lane's matrix excludes shard 4: a quarter of the suite never runs"),
    _M("Y17", CI, "--ignore=tests/viz \\\n", "--ignore=tests/viz --ignore=tests/core \\\n", CI_ALL,
       "the default lane stops collecting a whole package"),
    _M("Y18", CI, "python -m pytest tests/ -v -rs --tb=short --ignore=tests/viz \\",
       "python -m pytest tests/ -v -rs --tb=short --ignore=tests/viz -k 'not property' \\", CI_ALL,
       "the default lane deselects tests by keyword"),
    _M("Y19", CI, "--allowlist tests/duration_allowlist.txt --markdown /dev/null\n",
       "--allowlist tests/duration_allowlist.txt --markdown /dev/null --fail-over 0\n", CI_ALL,
       "the budget step switches its hard line off"),
    _M("Y20", CI, "      - name: Test time budget\n",
       "      - name: Test time budget\n        continue-on-error: true\n", CI_ALL,
       "the budget step can no longer fail the job"),
    _M("Y21", CI, "--allowlist tests/duration_allowlist.txt --markdown /dev/null\n",
       "--allowlist tests/duration_allowlist.txt --markdown /dev/null || true\n", CI_ALL,
       "the budget step swallows the gate's exit code with `|| true`"),
    _M("Y22", CI, "--allowlist tests/duration_allowlist.txt --markdown /dev/null\n",
       "--allowlist tests/duration_allowlist.txt --markdown /dev/null; true\n", CI_ALL,
       "the budget step swallows the gate's exit code with `; true`"),
    _M("Y23", CI, '          if [ "${{ github.event_name }}" != "pull_request" ]; then\n            mode=cold',
       '          if [ "${{ github.event_name }}" != "pull_request" ]; then\n            mode=warm', CI_ALL,
       "pushes restore the base cache instead of starting cold, then save it"),
    _M("Y24", CI, '            extra="--no-removable"\n', '            extra=""\n', CI_ALL,
       "the lane summary lists removable entries when a shard report is missing"),
    _M("Y25", CI, "if: ${{ !cancelled() && github.event_name == 'push' && steps.tests.outcome == 'success' "
                  "&& steps.prune-cache.outcome == 'success' }}",
       "if: ${{ !cancelled() && github.event_name == 'push' && steps.prune-cache.outcome == 'success' }}", CI_ALL,
       "the cache is saved even when the test step failed"),
    _M("Y26", CI, "key=jaxcc-v1-${{ runner.os }}-$cpu-py", "key=jaxcc-v1-${{ runner.os }}-py", CI_ALL,
       "the CPU model is dropped from the compilation-cache key",
       equivalent="the CPU model only raises the hit rate: JAX's own cache key includes the CPU's "
                  "features, so a cache from another model is never used wrongly, only missed"),
    # A job skipped by `if:` reports success, even to a required check, so
    # each of these is a green CI that runs no test.
    _M("Y27", CI, "  test:\n    needs: changes\n    if: needs.changes.outputs.code == 'true'\n",
       "  test:\n    needs: changes\n    if: needs.changes.outputs.code == 'True'\n", CI_ALL,
       "the eight default-lane shards never run"),
    _M("Y28", CI, "      code: ${{ steps.classify.outputs.code }}\n",
       "      code: ${{ steps.classify.outputs.cold }}\n", CI_ALL,
       "every lane gated on the code verdict is skipped unless the diff also asks for a cold run"),
    _M("Y29", CI, "  verify-hypothesis:\n    needs: changes\n    if: needs.changes.outputs.code == 'true'\n",
       "  verify-hypothesis:\n    needs: changes\n    if: needs.changes.outputs.code == 'True'\n", CI_ALL,
       "verify-hypothesis, the only ci-depth run of the property tests, never runs"),
    _M("Y30", CI, "  test-usd:\n    needs: changes\n    if: needs.changes.outputs.code == 'true'\n",
       "  test-usd:\n    needs: changes\n    if: needs.changes.outputs.code == 'True'\n", CI_ALL,
       "the only job that runs the usd-core tests never runs"),
    _M("Y31", CI, "    needs: [changes, test]\n    if: ${{ !cancelled() && needs.changes.outputs.code == 'true' }}\n",
       "    needs: [changes, test]\n    if: ${{ !cancelled() && needs.changes.outputs.code == 'True' }}\n", CI_ALL,
       "the lane summary (removable entries, slow-even-warm list, the no-shard-ran failure) never runs"),
    _M("Y32", CI, "        id: classify\n", "        id: classify-files\n", CI_ALL,
       "steps.classify.outputs.code reads empty: every gated lane is skipped"),
    _M("Y33", CI, "both are visible in the summary.\n    needs: changes\n    if: needs.changes.outputs.code == 'true'\n",
       "both are visible in the summary.\n    needs: changes\n    if: needs.changes.outputs.code == 'True'\n",
       CI_ALL, "the blocking typecheck never runs"),
    _M("Y34", CI, 'the change needs the test lanes"\n            exit 1\n',
       'the change needs the test lanes"\n            exit 0\n', CI_ALL,
       "a lane none of whose shards ran passes its summary: the skipped test job's only trace goes green"),
    _M("Y35", CI, "          python -m pytest tests/ -v -rs --tb=short --ignore=tests/viz \\\n",
       "          MADDENING_TEST_SHARD=1/4 python -m pytest tests/ -v -rs --tb=short --ignore=tests/viz \\\n",
       CI_ALL, "all four default-lane shards run shard 1's files: three quarters of the suite runs nowhere"),
    _M("Y36", CI, '"jaxlib==${{ matrix.jax-version }}"\n          pip install -e ".[ci]"\n',
       '"jaxlib==${{ matrix.jax-version }}"\n          pip install -e ".[ci]"\n'
       '          echo "MADDENING_TEST_SHARD=1/4" >> "$GITHUB_ENV"\n', CI_ALL,
       "an earlier step points every later step of every shard at shard 1"),
    _M("Y37", CI, "          JAX_COMPILATION_CACHE_DIR: ${{ runner.temp }}/jax-cache\n",
       "          JAX_COMPILATION_CACHE_DIR: ${{ runner.temp }}/jax-cache-2\n", CI_ALL,
       "the tests write a cache the prune and save steps never see: nothing is saved, every PR runs cold"),
    _M("Y38", CI, "          key: ${{ steps.cc.outputs.key }}-${{ github.sha }}\n\n      - name: Upload test durations",
       "          key: ${{ steps.cc.outputs.key }}\n\n      - name: Upload test durations", CI_ALL,
       "the first cache saved per key is kept for good (keys are immutable): PRs restore an ever older base"),
    _M("Y39", CI, "        timeout-minutes: 10\n        run: sudo apt-get update",
       "        run: sudo apt-get update", CI_ALL,
       "a silent package mirror holds a per-push runner for six hours (it did for 1 h 50 min on 2026-10-07)"),
    _M("Y40", CI, "        timeout-minutes: 10\n        run: sudo apt-get update",
       "        timeout-minutes: 600\n        run: sudo apt-get update", CI_ALL,
       "the install step has a limit in name only"),
    # --- Z: the slow lane, .github/workflows/slow-tests.yml -------------------
    _M("Z1", SL, '          echo "pytest exited with code ${{ steps.pytest.outputs.exit_code }}"\n          exit 1',
       '          echo "pytest exited with code ${{ steps.pytest.outputs.exit_code }}"', CI_ALL,
       "the slow lane goes green on ordinary test failures"),
    _M("Z2", SL, "sudo apt-get install -y -qq valgrind clang", "sudo apt-get install -y -qq clang", CI_ALL,
       "the valgrind tests skip in the only lane that runs them"),
    _M("Z10", SL, "        timeout-minutes: 10\n        run: sudo apt-get update",
       "        run: sudo apt-get update", CI_ALL,
       "a silent package mirror holds a slow-lane runner for the job's 180 minutes"),
    _M("Z3", SL, "      - name: Run full suite (slow lane)\n        id: pytest\n        continue-on-error: true\n"
                 "        env:\n",
       "      - name: Run full suite (slow lane)\n        id: pytest\n        continue-on-error: true\n"
       "        env:\n          PYTEST_ADDOPTS: \"--deselect tests/core\"\n", CI_ALL,
       "every slow-lane shard silently drops tests/core (selection moved into the step's env)"),
    _M("Z4", SL, "        shard: [1, 2, 3, 4, 5, 6, 7, 8]\n",
       "        shard: [1, 2, 3, 4, 5, 6, 7, 8]\n        exclude:\n          - shard: 8\n", CI_ALL,
       "the slow lane's matrix excludes shard 8"),
    _M("Z5", SL, "          timeout --kill-after=60s 175m python -m pytest tests/ \\\n",
       "          MADDENING_TEST_SHARD=1/8 timeout --kill-after=60s 175m python -m pytest tests/ \\\n", CI_ALL,
       "all eight slow-lane shards run shard 1's files"),
    _M("Z6", SL, '          MADDENING_TEST_SHARD: "${{ matrix.shard }}/8:weighted"',
       '          MADDENING_TEST_SHARD: "${{ matrix.shard }}/8"', SHARD,
       "the slow lane runs eight shards split by hash: every test once, the balance gone"),
    _M("Z7", SL, "        shard: [1, 2, 3, 4, 5, 6, 7, 8]\n", "        shard: [1, 2, 3, 4, 5, 6, 7]\n", SHARD,
       "the slow lane runs seven of eight shards"),
    _M("Z8", SL, "-shard${{ matrix.shard }}of8", "-shard${{ matrix.shard }}of4", SHARD,
       "the slow lane's artifacts are named for four shards: a download by name misses four"),
    _M("Z9", SL, "shard ${{ matrix.shard }} of 8)", "shard ${{ matrix.shard }} of 4)", SHARD,
       "a shard's summary is titled 'shard 5 of 4'"),
    # --- PY: the pytest configuration, pyproject.toml -------------------------
    _M("PY1", PP, "addopts = \"-m 'not slow'\"", "addopts = \"-m 'not slow' --ignore=tests/fmi\"", CI_ALL,
       "every lane silently stops collecting tests/fmi (selection moved into addopts)"),
    _M("PY2", PP, "addopts = \"-m 'not slow'\"", "addopts = \"-m 'not slow'\"\njunit_duration_report = \"call\"",
       CI_ALL, "the budget stops seeing setup and teardown (module-fixture compiles)"),
    # --- A: the shipped allowlist, tests/duration_allowlist.txt ---------------
    _M("A1", AL, "[gauss-seidel-interface-ift] # kept", "[gauss_seidel-interface-ift] # kept", ALLOW,
       "a typo'd entry (silently exempts nothing)"),
    _M("A2", AL, _AL_LAST, f"{_KEPT} # kept: prefix\n{_AL_LAST}", ALLOW,
       "an entry naming a parametrised test without its parameters"),
    _M("A3", AL, _AL_LAST,
       "tests/core/test_calibrate.py::TestCalibratePhysics::test_recover_gravity # kept: stale entry for a "
       f"slow-marked test\n{_AL_LAST}", ALLOW,
       "an entry for a slow-marked test",
       equivalent="exempts nothing while the test stays slow-marked, since no budgeted lane runs it; "
                  "it would matter only if the mark were removed"),
    _M("A4", AL, _AL_LAST,
       f"{_ZMQ}::TestNoSuchClass::test_a_worker_without_curve_does_not_hang_on_a_curve_coordinator "
       f"# kept: wrong class\n{_AL_LAST}", ALLOW, "an entry naming a class that does not exist"),
    _M("A5", AL, _AL_LAST,
       f"tests/core/test_coupling_diagnostics_leave_the_state_alone.py # kept: whole file\n{_AL_LAST}", ALLOW,
       "an entry naming a whole file"),
    _M("A6", AL, _AL_LAST,
       f"tests/core/test_compile_counts.py::test_a_test_that_was_renamed # kept: stale\n{_AL_LAST}", ALLOW,
       "an entry for a test that was renamed away"),
    _M("A7", AL, _AL_LAST, f"{_KEPT}[jacobi-l2-ift] # because\n{_AL_LAST}", ALLOW,
       "an entry whose reason is neither `kept:` nor `pending triage`"),
    # --- T: the compilation-cache isolation, tests/core/test_compile_cache.py -
    _M("T1", TC, "        cc._enabled_dir = enabled  # noqa: SLF001 -- restoring what enable() set\n"
                 "        jax_cache.reset_cache()",
       "        cc._enabled_dir = enabled  # noqa: SLF001 -- restoring what enable() set", CC,
       "JAX's cache object left pointing at the test's directory"),
    _M("T2", TC, "    settings, enabled = _snapshot()\n    jax_cache.reset_cache()\n    try:",
       "    settings, enabled = _snapshot()\n    try:", CC,
       "enable(tmp) silently keeps writing to the outer cache"),
    _M("T3", TC, "            jax.config.update(key, value)", "            pass", CC,
       "the three JAX settings left changed"),
    # --- W: the slow-only rule, tests/compliance/test_slow_only_rule.py -------
    _M("W1", _XF, f"# Per push: {_XF}::{_LM}\n@pytest.mark.slow", "@pytest.mark.slow", SLOW_RULE,
       "a slow framework test with no per-push witness and no table row: its property is off "
       "every push and nothing says so"),
    _M("W2", _XF, f"\n\ndef {_LM}():", f"\n\n@pytest.mark.slow\ndef {_LM}():", SLOW_RULE,
       "the named witness slow-marked itself: the property it stands for runs on no push"),
    _M("W3", "tests/property/test_sysid_contract.py",
       "tests/property/test_sysid_contract.py::TestBoundsAndTransforms::"
       "test_a_bound_no_float32_can_hold_is_met_at_the_leafs_precision",
       "tests/property/test_sysid_contract.py::test_a_bound_no_float32_can_hold_is_met_at_the_leafs_precision",
       SLOW_RULE, "a witness named without its class: not a node id, so nothing shows it runs"),
    _M("W4", TS, "`tests/core/test_calibrate.py::TestCalibratePhysics`, ", "", SLOW_RULE,
       "slow tests dropped from the slow-only table with no witness named"),
    _M("W5", TS, "`::test_every_recorded_fixture_still_measures_its_baseline_row`",
       "`::test_every_recorded_fixture_measures_its_row`", SLOW_RULE,
       "a table row naming a renamed test: the row exempts nothing and the real test is uncovered"),
    _M("W6", SR, 'SIBLING_REQUIRED = ("tests/verification/hypothesis/",)', "SIBLING_REQUIRED = ()",
       SLOW_RULE, "a slow hypothesis property listed only in the table: verify-hypothesis runs it on "
       "jax 0.10.2 only, so jax 0.11.2 sees it in the slow lane alone"),
    _M("W7", SR, "default_lane=frozenset(n for n, slow, skip in items if not slow and not skip),",
       "default_lane=frozenset(n for n, slow, skip in items if not skip),", SLOW_RULE,
       "a slow-marked test accepted as a per-push witness"),
    _M("W8", SR, '    "tests/core/",\n', "", SLOW_RULE,
       "tests/core dropped from the covered directories: its slow marks go unchecked"),
    _M("W9", SR, 'return nodeid == target or nodeid.startswith(target + "[") or',
       "return nodeid == target or nodeid.startswith(target) or", SLOW_RULE,
       "a witness name that is only a prefix of a real test (test_x for test_xy) accepted"),
    _M("W10", SR, r'_PER_PUSH = re.compile(r"^\s*#\s*Per push:")', r'_PER_PUSH = re.compile(r"^\s*#\s*Per push")',
       SLOW_RULE, "prose such as 'Per push, the forward tests ...' read as a witness"),
    _M("W11", SR, '    "tests/fmi/test_binary_frames.py":\n',
       '    "tests/core/test_transforms.py": "stale: this file holds no slow test",\n'
       '    "tests/fmi/test_binary_frames.py":\n', SLOW_RULE,
       "an exemption that names a file with no slow test: it outlives what it was for and would "
       "exempt the next slow test written there"),
    _M("W12", _XF, f"\n\ndef {_LM}():", f"\n\n@pytest.mark.skip(reason='x')\ndef {_LM}():", SLOW_RULE,
       "the named witness skip-marked: it is collected and never runs"),
    # --- K: the claims inventories' domain matrix, tests/compliance/test_claims_inventories.py
    _M("K1", CPL, "    float32, a rate near 1. Not claimed for float64 (x64), mixed dtypes,\n",
       "    float32, a rate near 1. Not claimed for float64 (x64),\n", CLAIMS,
       "a narrowed domain whose conditions no longer exclude it: the row reads as claimed there"),
    _M("K2", CPL, "    f64: tests/core/test_coupling_accelerators_under_x64.py::"
       "test_a_float64_group_keeps_float64_carries_under_x64\n",
       "    f64: tests/core/test_coupling_claims_graphs.py::test_imvj_without_reuse_is_iqn_ils\n",
       CLAIMS, "a float64 cell citing a test that never runs under x64"),
    _M("K3", _CLAIMS, "    return tests + [t for _, t in domain_targets(row)]\n", "    return tests\n",
       CLAIMS, "a domain cell's test left out of the collection and xfail rules: a strict xfail "
       "cited only in a domain would not make its row failing"),
    _M("K4", _CLAIMS, "        if value == NOT_APPLICABLE and d in covered:\n", "        if False:\n",
       CLAIMS, "n/a accepted for a domain the conditions name: an untested domain hidden as "
       "inapplicable"),
    _M("K5", RST,
       "    second slice fails, held to a replay of the counted steps. Not claimed\n"
       "    for a server shutting down.\n",
       "    second slice fails, held to a replay of the counted steps.\n", CLAIMS,
       "a narrowed server domain whose conditions no longer exclude it: REST-051 reads as "
       "claimed while the server shuts down"),
    _M("K6", RST, _STEPS_CELLS % _STEPS_SERVED,
       _STEPS_CELLS % "tests/api/test_concurrent_requests_are_serialised.py::"
       "test_concurrent_steps_are_all_taken_bit_identical_to_serial", CLAIMS,
       "a concurrent cell citing in-process TestClient threads, not a real server"),
    _M("K7", _CLAIMS, "        return bool(_REAL_SERVER.search(text) and _SIMULTANEOUS.search(text))\n",
       "        return bool(_SIMULTANEOUS.search(text))\n", CLAIMS,
       "the concurrent witness satisfied by threads alone: in-process clients read as a real "
       "server"),
    _M("K8", _CLAIMS, '                                           "server": SERVER_DOMAINS}\n',
       "                                           }\n", CLAIMS,
       "the server domain set dropped: the REST inventory's matrix checked by nothing"),
    _M("K9", _CLAIMS, '    (re.compile(r"\\bnon-loopback (?:bind|spelling)", re.IGNORECASE), '
       '("token_enforced",)),\n', "", CLAIMS,
       "a non-loopback bind no longer covers token_enforced: the token cell of a row about "
       "such a bind could hide as n/a"),
    # A server cell citing a test of another domain, one per domain a shared helper's words
    # could witness (K10-K13: the ones the in-process client's did) or a bind or record names.
    _M("K10", RST, _STEPS_CELLS % _STEPS_SERVED, _STEPS_CELLS % repr(_CLEARTEXT), WITNESS,
       "a concurrent cell citing a real server spoken to one connection at a time: the server's "
       "own thread read as simultaneous requests"),
    _wrong_cell("K11", "loopback_bind", f"{_BEARER}test_loopback_bind_serves_every_route_without_a_token",
                f"{_BEARER}test_interactive_docs_are_not_served_when_the_token_is_enforced",
                "a loopback cell citing a test that only binds 0.0.0.0"),
    _wrong_cell("K12", "no_token", f"{_BEARER}test_a_loopback_websocket_still_gets_its_subprotocol_echoed",
                f"{_BEARER}test_a_websocket_authenticates_with_the_authorization_header",
                "a tokenless cell citing a test that always presents the token"),
    _wrong_cell("K13", "hostile_input",
                f"{_API}test_rest_claims_in_every_domain.py::"
                "test_the_claim_holds_on_a_non_loopback_bind_with_the_token[REST-006]",
                f"{_BEARER}test_the_backstop_explains_the_misconfiguration",
                "a hostile-input cell citing a test that sends one well-formed GET"),
    _wrong_cell("K14", "non_loopback_bind",
                f"{_API}test_auth_and_host_rules_hold_at_their_edges.py::"
                "test_an_anonymous_oversized_body_is_refused_for_its_credential_first",
                f"{_API}test_request_memory_is_bounded.py::test_a_body_over_the_limit_is_a_413_before_it_is_parsed",
                "a non-loopback cell citing a test on the default loopback bind"),
    _wrong_cell("K15", "token_enforced",
                f"{_API}test_auth_and_host_rules_hold_at_their_edges.py::"
                "test_every_state_changing_route_refuses_a_foreign_origin",
                f"{_API}test_cross_origin_requests.py::test_a_cross_origin_state_change_is_refused",
                "a token cell citing a test that presents no token to a loopback bind"),
    _wrong_cell("K16", "shutdown",
                f"{_API}test_rest_claims_in_every_domain.py::"
                "test_the_claim_holds_when_sigterm_arrives_mid_request[REST-029]",
                f"{_API}test_request_bounds_hold_at_their_boundaries.py::"
                "test_a_body_of_exactly_the_limit_is_read_and_one_byte_more_is_a_413",
                "a shutdown cell citing a test that raises no signal"),
    _wrong_cell("K17", "dry_run_cpu",
                f"{_MG}test_run_pod_claims_across_records.py::"
                "test_a_dry_run_record_whose_commit_is_not_a_sha_counts_as_none",
                f"{_EDGES}test_a_repository_with_no_commit_records_no_commit",
                "a dry-run cell citing a test that reads no dry run's record"),
    _wrong_cell("K18", "relabelled_records",
                f"{_VERDICT}test_summarise_lists_a_record_that_cannot_decide_and_exits_3",
                f"{_EDGES}test_a_truncated_goal_file_reads_invalid_and_the_summary_exits_3",
                "a relabelled-record cell citing a test that relabels nothing"),
    _wrong_cell("K19", "mixed_commits",
                f"{_MG}test_run_pod_claims_across_records.py::"
                "test_a_goal_that_raised_among_files_of_two_commits_fails_its_item_and_exits_3",
                f"{_VERDICT}test_a_tampered_record_of_a_goal_that_raised_cannot_decide",
                "a mixed-commit cell citing a test whose files record one commit"),
    # ... and the witness rule's own repairs undone
    _M("K20", _CLAIMS, "    if rel in TRANSPORT_MODULES:        # the shared client: no test's own words\n"
       "        return None\n", "", TRANSPORT,
       "the in-process client's source read as every REST test's own words: its loopback Host and "
       "peer witness a loopback bind and a tokenless request, its annotations hostile input"),
    _M("K21", _CLAIMS, "    dropped = {n for imp in ast.walk(node) if _imports_transport(imp)\n",
       "    dropped = {n for imp in ast.walk(node) if False\n", TRANSPORT,
       "the line that imports the in-process client inside a test read as the test naming loopback"),
    _M("K22", _CLAIMS, '_REAL_SERVER = re.compile(r"(?<!`)\\b\\w*uvicorn(?:\\(\\))?\\.(?:run|Server)\\(")\n',
       '_REAL_SERVER = re.compile(r"uvicorn|socket|http://127\\.0\\.0\\.1", re.IGNORECASE)\n', SERVER_WORDS,
       "a real server read from a loopback URL, a WebSocket or the word uvicorn: in-process "
       "threads through a client that names a loopback Host witness concurrent"),
    _M("K23", _CLAIMS, '_SIMULTANEOUS = re.compile(r"simultaneous|barrier|gather|Thread\\(\\s*target=(?!\\w+\\.run\\b)",\n'
       "                           re.IGNORECASE)\n",
       '_SIMULTANEOUS = re.compile(r"concurren|thread|simultaneous|barrier|gather", re.IGNORECASE)\n',
       SERVER_WORDS, "the server's own thread and domain tag read as simultaneous requests: every test "
       "that starts a real server witnesses concurrent"),
    # A cited test under tests/usd needs usd-core.  These must be caught where pytest collects
    # the file (usd-core installed) and where its items are read from source (it is not).
    _M("K24", MAP, f"    - {_USD_SPEC}::{_USD_GONE}\n", f"    - {_USD_SPEC}::{_USD_GONE}_too\n", CLAIMS,
       "a row citing a tests/usd test that does not exist: unverifiable where usd-core is "
       "absent, so taken on trust there"),
    _M("K25", CI, "            tests/usd/ \\\n", "", CLAIMS,
       "the one job that installs usd-core no longer runs tests/usd: a row's cited USD test "
       "runs in no per-push job"),
    _M("K26", _USD_SPEC, f"\n\ndef {_USD_GONE}():", f"\n\n@pytest.mark.skip(reason='x')\ndef {_USD_GONE}():",
       CLAIMS, "a cited tests/usd test skip-marked: it is collected and never runs"),
    _M("K27", CI, "      # running them locally relies on.\n      - name: Install dependencies\n"
       "        run: |\n          python -m pip install --upgrade pip\n"
       '          pip install "jax==0.10.2" "jaxlib==0.10.2"\n          pip install -e ".[ci,usd]"\n',
       "      # running them locally relies on.\n      - name: Install dependencies\n"
       "        run: |\n          python -m pip install --upgrade pip\n"
       '          pip install "jax==0.10.2" "jaxlib==0.10.2"\n          pip install -e ".[ci]"\n',
       CLAIMS, "the compliance job without usd-core: no per-push job compares the claims "
       "guard's source reader with what pytest collects from tests/usd"),
    # --- IP: a coupling group's edges are enumerated in one module only, -----
    # --- tests/core/test_interface_plan_is_the_only_enumeration.py -----------
    _M("IP1", "src/maddening/core/coupling/_group_layout.py",
       '    return group.convergence_norm == "interface" and plan.norm_reads_mapping_weights()\n',
       '    return group.convergence_norm == "interface" and any(\n'
       "        e.mapping is not None and e.source_field for e in plan.declared_edges())\n",
       PLAN_SCAN, "a second enumeration of what the interface norm reads, in the layout module"),
    _M("IP2", "src/maddening/core/coupling/_coupled_block.py",
       "                read = plan.source_fields()\n",
       "                read = {(e.source_node, e.source_field) for e in plan.declared_edges()}\n",
       PLAN_SCAN, "the spectrum's weights looping over the edges themselves again"),
    _M("IP3", _PLAN_SCAN,
       'EDGE_ATTRIBUTES = ("source_node", "target_node", "source_field", "target_field", "geometry")',
       'EDGE_ATTRIBUTES = ("source_field", "target_field")',
       PLAN_SCAN, "a loop that tests only the edges' nodes, or reads their geometry"),
    _M("IP4", _PLAN_SCAN,
       '           if not where.startswith(f"{DESCRIPTION}::") and where not in allowed]',
       '           if not where.startswith("core/coupling/") and where not in allowed]',
       PLAN_SCAN, "any enumeration written in the coupling package: the whole package exempt"),
    _M("IP5", _PLAN_SCAN,
       '    return [*sorted((SRC / "core" / "coupling").glob("*.py")), SRC / "core" / "graph_manager.py"]',
       '    return sorted((SRC / "core" / "coupling").glob("*.py"))',
       PLAN_SCAN, "an enumeration written in graph_manager.py, which the scan no longer reads"),
    _M("IP6", _PLAN_SCAN,
       "            for where in sorted(allowed) if where not in found]",
       "            for where in sorted(allowed) if False]",
       PLAN_SCAN, "an allowance that outlives its function, inherited by the next one of that name"),
)


# --------------------------------------------------------------------------
# Per push: the table still matches the tree.
# --------------------------------------------------------------------------
@cache
def _tree_text(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _guard_targets(guards: tuple[str, ...]) -> list[str]:
    """The files and node ids a guard selection names (option values included)."""
    return [g for g in guards if not g.startswith("-")]


def test_every_mutant_anchor_matches_the_tree_exactly_once():
    """A missing or ambiguous anchor would make its mutant a silent no-op."""
    problems = []
    for m in MUTANTS:
        path = REPO_ROOT / m.file
        if not path.is_file():
            problems.append(f"{m.id}: {m.file} does not exist")
            continue
        found = _tree_text(m.file).count(m.anchor)
        if found != 1:
            problems.append(f"{m.id}: anchor found {found} times in {m.file} (need exactly 1): {m.anchor[:80]!r}")
    assert not problems, (
        "guard mutants out of step with the tree; update the anchor in MUTANTS to the code's new "
        "spelling (keep the fault the same), or retire the mutant on purpose:\n  " + "\n  ".join(problems))


def test_the_mutant_table_is_well_formed():
    """Unique ids, real guard tests, a real edit, and a reason for every exception."""
    problems = []
    ids = [m.id for m in MUTANTS]
    problems += [f"duplicate id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for m in MUTANTS:
        if not re.fullmatch(r"[A-Z]{1,2}[0-9]+", m.id):
            problems.append(f"{m.id}: ids are a group letter and a number, e.g. R26")
        if m.anchor == m.replacement or not m.anchor:
            problems.append(f"{m.id}: the replacement must change the anchor")
        if m.equivalent and m.gap:
            problems.append(f"{m.id}: a mutant is either equivalent or a guard gap, not both")
        if not m.lets_through.strip():
            problems.append(f"{m.id}: say what the fault would let through in CI")
        targets = _guard_targets(m.guards)
        if not targets:
            problems.append(f"{m.id}: no guard tests")
        for target in targets:
            file, _, name = target.partition("::")
            func = name.rsplit("::", 1)[-1].split("[", 1)[0]
            if not (REPO_ROOT / file).is_file():
                problems.append(f"{m.id}: guard file {file} does not exist")
            elif func and not re.search(rf"^\s*(async\s+)?def {re.escape(func)}\(", _tree_text(file), re.M):
                problems.append(f"{m.id}: guard test {target} does not exist")
        for i, arg in enumerate(m.guards):
            if arg == "--deselect" and (i + 1 == len(m.guards) or m.guards[i + 1].startswith("-")):
                problems.append(f"{m.id}: --deselect without a node id")
    assert not problems, "\n".join(problems)


def test_every_seeded_inventory_fault_fails_a_rule_that_reads_only_the_tree():
    """The mutants that edit a claims inventory, caught on every push.

    The row rule and the witness rule are pure functions of an inventory's
    rows and the cited tests' sources, so such a mutant needs no copy of the
    tree: it is applied in memory and the two rules are asked of the rows it
    changed.  The slow lane runs on a schedule, on the default branch; a
    change that blinds a rule -- a helper every test shares starting to say
    a domain's words -- would otherwise pass every push of its own PR."""
    import yaml

    from tests.compliance import test_claims_inventories as claims

    row_start = "\n- id: "     # a row starts at the left margin, in every inventory

    @cache
    def head(file: str):
        """The file's top level (its prefixes and domain set), without its rows."""
        return claims.Inventory(Path(file).name, yaml.safe_load(_tree_text(file).split(row_start)[0]))

    def problems(row_text: str, file: str) -> list[str]:
        rows = yaml.safe_load(row_text)
        assert isinstance(rows, list) and len(rows) == 1, row_text[:200]
        return (claims.row_problems(rows, head(file).prefixes, domains=head(file).domains)
                + claims.domain_witness_problems(rows))

    checked, survivors = [], []
    for m in MUTANTS:
        if m.file not in (CPL, RST) or m.equivalent or m.gap:
            continue
        text = _tree_text(m.file)
        at = text.index(m.anchor)
        end = text.find(row_start, at + len(m.anchor))
        row = text[text.rindex(row_start, 0, at) + 1:len(text) if end < 0 else end + 1]
        # The row as it stands passes both rules, or a catch below means nothing.
        assert problems(row, m.file) == [], f"{m.id}: its row fails unmutated"
        checked.append(m.id)
        if not problems(row.replace(m.anchor, m.replacement), m.file):
            survivors.append(f"{m.id}: {m.lets_through}")
    assert len(checked) >= 14, f"inventory mutants checked: {checked}"
    assert not survivors, (
        "seeded inventory faults that neither the row rule nor the witness rule of "
        f"{_CLAIMS} catches on this tree:\n  " + "\n  ".join(survivors))


# --------------------------------------------------------------------------
# Slow lane: every mutant, run.
# --------------------------------------------------------------------------
class GuardGap(AssertionError):
    """A mutant that should be caught survived every guard test."""


#: Unset for the guard runs: CI-only switches and anything that would add
#: pytest arguments, change the cache or write into the outer job.
_UNSET_PREFIXES = ("MADDENING_", "PYTEST_", "JAX_")
_UNSET = {"GITHUB_STEP_SUMMARY", "GITHUB_OUTPUT", "GITHUB_ENV", "GITHUB_PATH", "COV_CORE_SOURCE"}
#: One guard run's ceiling.  The slowest selection takes under a minute.
_RUN_TIMEOUT_S = 600


@dataclass(frozen=True)
class GuardRun:
    exit_code: int
    first_failure: str
    tail: str


def _run_guards(tree: Path, args: tuple[str, ...], basetemp: Path, *, stop_at_first: bool = True) -> GuardRun:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_UNSET_PREFIXES) and k not in _UNSET}
    env.update(PYTHONPATH=str(tree / "src"), JAX_PLATFORMS="cpu", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "pytest", *args, "-q", "-p", "no:cacheprovider", f"--basetemp={basetemp}"]
    if stop_at_first:
        cmd.append("-x")
    try:
        proc = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=_RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        pytest.fail(f"the guard run timed out after {_RUN_TIMEOUT_S} s: {' '.join(args)}\n{out[-2000:]}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    failures = [ln for ln in lines if ln.startswith(("FAILED ", "ERROR "))]
    tail = lines[-1] if lines else proc.stderr.strip()[-500:]
    return GuardRun(proc.returncode, failures[0] if failures else "", tail)


def _unpack(archive: bytes, dest: Path) -> Path:
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return dest


@pytest.fixture(scope="module")
def head_archive() -> bytes:
    """``git archive HEAD``: the committed tree, whatever the working tree holds."""
    proc = subprocess.run(["git", "-C", str(REPO_ROOT), "archive", "--format=tar", "HEAD"],
                          capture_output=True, timeout=300)
    if proc.returncode != 0:
        pytest.fail("the guard mutations run on `git archive HEAD` and need a git checkout: "
                    + proc.stderr.decode(errors="replace")[-500:])
    return proc.stdout


@pytest.fixture(scope="module")
def guards_pass_unmutated(head_archive, tmp_path_factory) -> None:
    """Every guard file passes on the unmutated copy, or no verdict below means anything."""
    root = tmp_path_factory.mktemp("guards_unmutated")
    tree = _unpack(head_archive, root / "tree")
    run = _run_guards(tree, GUARD_FILES, root / "basetemp", stop_at_first=False)
    if run.exit_code != 0:
        pytest.fail(f"the guard tests fail on the unmutated tree (exit {run.exit_code}: {run.tail}); "
                    f"first failure: {run.first_failure or 'none reported'}.  Fix that first.")


def _params():
    for m in MUTANTS:
        marks = [pytest.mark.xfail(strict=True, raises=GuardGap, reason=f"guard gap: {m.gap}")] if m.gap else []
        yield pytest.param(m, id=m.id, marks=marks)


# Slow-only on purpose: every mutant re-runs guard test files in a fresh copy
# of the tree, minutes in all (docs/developer_guide/testing_standards.md,
# "What only the slow lane checks").
# Per push: tests/compliance/test_guard_mutations.py::test_every_mutant_anchor_matches_the_tree_exactly_once
@pytest.mark.slow
@pytest.mark.parametrize("mutant", list(_params()))
def test_every_seeded_ci_fault_fails_a_guard(mutant, head_archive, guards_pass_unmutated, tmp_path,
                                             record_property):
    tree = _unpack(head_archive, tmp_path / "tree")
    target = tree / mutant.file
    text = target.read_text(encoding="utf-8")
    assert text.count(mutant.anchor) == 1, f"{mutant.id}: anchor not found exactly once in HEAD's {mutant.file}"
    target.write_text(text.replace(mutant.anchor, mutant.replacement), encoding="utf-8")

    run = _run_guards(tree, mutant.guards, tmp_path / "basetemp")
    if run.exit_code not in (0, 1):
        pytest.fail(f"{mutant.id}: the guard run broke (exit {run.exit_code}: {run.tail}); a broken run is "
                    "not a catch -- check the replacement is valid in its file")
    caught = run.exit_code == 1
    record_property("caught_by", run.first_failure)
    if mutant.equivalent:
        assert not caught, (
            f"{mutant.id} is listed as equivalent ({mutant.equivalent}) but is now caught by "
            f"{run.first_failure or run.tail}.  If that is right, drop `equivalent=` from its entry.")
    elif not caught:
        raise GuardGap(f"guard gap: {mutant.id} survived every guard test in {_guard_targets(mutant.guards)} "
                       f"({run.tail}); it would let through: {mutant.lets_through}")
