"""Every CI guard must fail when the thing it guards is broken.

The CI lanes, the per-test time budget, the shard split, the JAX timing
plugin, the compilation-cache pruner and the duration allowlist are each
pinned by guard tests (``tests/compliance/test_ci_workflows.py``,
``test_ci_sharding.py``, ``test_report_test_durations.py``,
``test_prune_jax_cache.py`` and ``tests/core/test_compile_cache.py``).  A
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
mutant into a no-op that only the slow lane, days later, could notice.

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

#: The budget script and the timing plugin.  The allowlist-collection test
#: is left out: it collects the whole tree (seconds), and only the allowlist
#: mutants need it.
DUR = (_DURATIONS, "--deselect", _COLLECTS)
SHARD = (_SHARDING,)
WF = (_WORKFLOWS,)
PRUNE = (_PRUNE,)
CC = (_COMPILE_CACHE,)
#: The shipped allowlist: the reason check (fast) first, then collection.
ALLOW = (_REASONS, _COLLECTS)
#: Workflows, the root conftest and the pytest configuration.
CI_ALL = DUR + SHARD + WF

#: Every guard file, run whole on the unmutated copy before any mutant.
GUARD_FILES = (_DURATIONS, _SHARDING, _WORKFLOWS, _PRUNE, _COMPILE_CACHE)


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
PP = "pyproject.toml"
AL = "tests/duration_allowlist.txt"
TC = "tests/core/test_compile_cache.py"

# Lines reused by several allowlist mutants (an existing entry, and a real
# parametrised test).
_AL_LAST = ("tests/core/test_compile_counts.py::"
            "test_the_committed_baseline_matches_what_the_code_compiles_to # kept")
_KEPT = ("tests/core/test_coupling_diagnostics_leave_the_state_alone.py::"
         "test_diagnostics_leave_the_returned_state_bit_identical")
_ZMQ = "tests/security/test_zmq_transport_auth.py"

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
    # --- S: the shard split, tests/_sharding.py ------------------------------
    _M("S1", SH, 'return int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % n + 1',
       'return int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % n', SHARD,
       "0-based shards: every file hashing to 0 runs nowhere"),
    _M("S2", SH, "    if n == PINS_FOR and path in PINS:", "    if False:", SHARD, "pins ignored"),
    _M("S3", SH, "    if n == PINS_FOR and path in PINS:", "    if path in PINS:", SHARD,
       "pins leak into other job counts"),
    _M("S4", SH, "(keep if shard_of(path, self.of) == self.shard else drop).append(item)", "keep.append(item)",
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
    # --- Z: the slow lane, .github/workflows/slow-tests.yml -------------------
    _M("Z1", SL, '          echo "pytest exited with code ${{ steps.pytest.outputs.exit_code }}"\n          exit 1',
       '          echo "pytest exited with code ${{ steps.pytest.outputs.exit_code }}"', CI_ALL,
       "the slow lane goes green on ordinary test failures"),
    _M("Z2", SL, "sudo apt-get install -y -qq valgrind clang", "sudo apt-get install -y -qq clang", CI_ALL,
       "the valgrind tests skip in the only lane that runs them"),
    _M("Z3", SL, "      - name: Run full suite (slow lane)\n        id: pytest\n        continue-on-error: true\n"
                 "        env:\n",
       "      - name: Run full suite (slow lane)\n        id: pytest\n        continue-on-error: true\n"
       "        env:\n          PYTEST_ADDOPTS: \"--deselect tests/core\"\n", CI_ALL,
       "every slow-lane shard silently drops tests/core (selection moved into the step's env)"),
    _M("Z4", SL, "        shard: [1, 2, 3, 4]\n",
       "        shard: [1, 2, 3, 4]\n        exclude:\n          - shard: 4\n", CI_ALL,
       "the slow lane's matrix excludes shard 4"),
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
            if not (REPO_ROOT / file).is_file():
                problems.append(f"{m.id}: guard file {file} does not exist")
            elif name and not re.search(rf"^def {re.escape(name)}\(", _tree_text(file), re.M):
                problems.append(f"{m.id}: guard test {target} does not exist")
        for i, arg in enumerate(m.guards):
            if arg == "--deselect" and (i + 1 == len(m.guards) or m.guards[i + 1].startswith("-")):
                problems.append(f"{m.id}: --deselect without a node id")
    assert not problems, "\n".join(problems)


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
        pytest.fail(f"the guard run timed out after {_RUN_TIMEOUT_S} s: {' '.join(args)}\n"
                    f"{(exc.stdout or '')[-2000:] if isinstance(exc.stdout, str) else ''}")
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
