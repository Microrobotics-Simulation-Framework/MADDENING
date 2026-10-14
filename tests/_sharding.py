"""Split the suite across CI runners by test file, stably.

Each CI test lane runs as N parallel jobs; ``MADDENING_TEST_SHARD=i/N``
tells a job to keep only the files of shard ``i`` (1-based).  The whole
suite is still *collected* in every job -- so every ``conftest.py`` is
imported exactly as in a single-process run, including ones that change
the environment at import time -- and the other files' tests are
deselected.

The assignment is a pure function of the file's path, never of the test
list:

* adding or removing a test, or a whole file, moves no other file;
* a file's tests stay together, so module and class fixtures build once;
* shard ``i`` of a pull request therefore holds the same files as shard
  ``i`` of the base branch, and the per-shard XLA compilation cache the
  base branch saves is the right one for it.

Duration-balanced splitters (``pytest-split``) do not have that property:
they cut the suite into chunks by measured time, so one new test shifts
every boundary after it, and they split files across shards.

Both lanes ask for a weighted split, by a suffix on the spec, because a
hash leaves the shards far from equal: one file takes 48 minutes on the
slow lane, and by hash a single shard held nearly half of it.  A weighted
split deals the files of a table of measured seconds per file, longest
first, each onto the shard with the least time so far; a file the table
does not list goes by the hash above.  It is still a function of the path
and of committed data, and a file's tests still stay together.

* ``i/N:weighted`` deals ``slow_lane_weights.json`` (``slow-tests.yml``).
  That lane restores no compilation cache, so nothing there depends on a
  file staying on its shard.
* ``i/N:weighted-push`` deals ``per_push_weights.json`` (``ci.yml``).  The
  per-push lane does restore a cache per shard, saved by the base branch's
  run of the same shard, so a pull request must see the same deal as its
  base.  It does: the deal is a function of the committed table alone, never
  of the test list, so a change to the table is the one thing that moves a
  file, and ``ci.yml``'s ``changes`` job runs a pull request cold when its
  diff touches the table.

A plain ``i/N`` is the hash split alone.

GitHub's runners differ in speed: on the slow lane of 2026-10-11 to 13
one runner took 1.6 to 1.9 times as long as another over the same files.
A table measured on fast runners under-weighs what they ran (shard 4
ran 95 to 108 minutes against 68 predicted: its heaviest file had
been weighed on the two fastest runners of sixty).  So each shard
records how fast its runner was, as properties of its JUnit report
(:data:`RUNNER_PROPERTIES`): the seconds pytest took to collect the
suite, which is the same work on every shard and comes before any test,
and the runner's CPU.  The report is the one place these are written and
read.  The collection time is the speed measure: ``scripts/slow_lane_weights.py``
scales each shard's times to a runner of median speed by
:func:`runner_scale`, and ``scripts/slow_lane_drift.py`` scales its
prediction by the same function.  The CPU model is recorded; the weights
script uses it only for a report that has no collection time.

A few files must not share a shard (:data:`APART`): the deal puts each
file of a group on a shard that holds no other file of the group.  Each
group names the defect it works around and is removed when that is fixed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import pytest

ENV_VAR = "MADDENING_TEST_SHARD"

#: The job count ``ci.yml`` runs per JAX version.  Six, from the per-file
#: times of the cold push runs of 2026-10-11 to 13 (``tests/per_push_weights.json``):
#: the weighted split predicts 27.6 minutes of test time on the slowest of six
#: shards, 23.4 of seven and 20.8 of eight.  Six keeps a push at 19 jobs, so
#: three runs at once fit the organisation's 60.
PER_PUSH_SHARDS = 6

#: The suffix of a shard spec that asks for the slow lane's weighted split.
WEIGHTED = "weighted"
#: ... and for the per-push lane's.
WEIGHTED_PUSH = "weighted-push"
#: The job count ``slow-tests.yml`` runs per JAX version.  Ten, not the
#: six of the per-push lane: the measured lane is 635 minutes of tests
#: (each file's slower JAX version, 2026-10-10; 477 three days earlier),
#: so eight balanced shards sit at 79 minutes each and ten at 64.  Twelve
#: would sit at 53 (the slowest predicted at 56 with a job's fixed cost),
#: and the organisation's plan runs that many jobs at once (60 against the
#: lane's 24): they are the next step.  The heaviest file alone is 49
#: minutes, so past twelve the step is to split that file.
SLOW_LANE_SHARDS = 10
#: Measured seconds per test file for each weighted split, next to this
#: module.  ``scripts/slow_lane_weights.py`` writes them from the JUnit
#: artifacts of a run and prints the minutes it predicts per shard.
WEIGHTS_FILE = Path(__file__).with_name("slow_lane_weights.json")
PER_PUSH_WEIGHTS_FILE = Path(__file__).with_name("per_push_weights.json")
#: ``{suffix of a shard spec: its table}``.
TABLES = {WEIGHTED: WEIGHTS_FILE, WEIGHTED_PUSH: PER_PUSH_WEIGHTS_FILE}

#: The JUnit report's property holding the seconds from the session's start
#: to the end of collection: the runner's speed, measured on the same work
#: on every shard (every test module of the suite is imported; the other
#: shards' tests are only deselected).
COLLECT_SECONDS_PROPERTY = "runner_collect_seconds"
#: ... the runner's CPU model (metadata; see :func:`runner_scale`).
CPU_PROPERTY = "runner_cpu"
#: Every property :class:`Plugin` adds to the report (``testsuite`` level).
RUNNER_PROPERTIES = (COLLECT_SECONDS_PROPERTY, CPU_PROPERTY, "runner_cpus")
#: How a runner's collection time scales its test time: a test file's
#: seconds go as the collection seconds to this power.  Fitted on the
#: sixty runners of the slow-lane runs 38182068116, 38206585198 and
#: 38332673699 (2026-10-11 to 13): the collection time and each runner's
#: speed over the files it ran agree with r = 0.86, slope 1.54 in logs, and
#: scaling by it halves the spread of one file's time between runners (the
#: standard deviation of the log, over 146 files, from 0.17 to 0.09).
SPEED_EXPONENT = 1.5

#: Groups of test files the weighted deal never puts on one shard.
#: Temporary by rule: each group says which defect it works around, and is
#: removed when that defect is fixed.
APART: tuple[tuple[str, ...], ...] = (
    # A JAX trace leaked by the checkpoint-load file's costly-kind tests
    # (29k DynamicJaxprTracer still alive after jax.clear_caches() and
    # gc.collect()) makes the REST file's costly-node tests about ten times
    # slower when the two share a process: on shard 7 of slow-lane run
    # 38504812929 they did, and the shard ran 131.7 / 142.3 minutes
    # (bisect: benchmarks/results/050_p0_3c_shard4/local/shard7_bisect.txt).
    # The leak is being investigated; remove this group when it is fixed.
    ("tests/api/test_a_checkpoint_load_answers_as_a_params_write_on_every_parameter.py",
     "tests/property/test_rest_requests_generated_from_the_schema.py"),
)


def runner_fields(report) -> dict[str, str]:
    """The :data:`RUNNER_PROPERTIES` of a parsed JUnit report (an ``Element``), those it has."""
    suites = [report] if report.tag == "testsuite" else list(report.iter("testsuite"))
    found: dict[str, str] = {}
    for suite in suites:
        for prop in suite.findall("properties/property"):
            if prop.get("name") in RUNNER_PROPERTIES and prop.get("value"):
                found.setdefault(prop.get("name"), prop.get("value"))
    return found


def collect_seconds(report) -> float | None:
    """A parsed report's :data:`COLLECT_SECONDS_PROPERTY` in seconds, or None without a usable one."""
    try:
        seconds = float(runner_fields(report).get(COLLECT_SECONDS_PROPERTY, ""))
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def runner_scale(seconds: float, reference_seconds: float, exponent: float = SPEED_EXPONENT) -> float:
    """What one second on a runner that collected in ``seconds`` is on one that collected in ``reference_seconds``.

    ``(reference_seconds / seconds) ** exponent``: below 1 for a runner slower
    than the reference, above 1 for a faster one.  The one rule both the
    weights table (each report times this) and the drift check (its
    prediction divided by this) use.
    """
    return (reference_seconds / seconds) ** exponent


def cpu_model() -> str:
    """The CPU's model name from ``/proc/cpuinfo``, or ``"unknown"``."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as info:
            for line in info:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def usable_cpus() -> int:
    """The CPUs this process may run on (its affinity), else the machine's count."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 0


def record_suite_property(config, name: str, value: str) -> bool:
    """Add a ``testsuite`` property to the ``--junitxml`` report, if one is written.

    The same call pytest's ``record_testsuite_property`` fixture makes; it
    returns False when no report is being written.
    """
    try:
        from _pytest.junitxml import xml_key
    except ImportError:  # checked against a real report in tests/compliance/test_ci_sharding.py
        return False
    xml = config.stash.get(xml_key, None)
    if xml is None:
        return False
    xml.add_global_property(name, value)
    return True


def parse(spec: str) -> tuple[int, int, str | None]:
    """``"i/N"`` or ``"i/N:<table>"`` -> ``(i, N, table suffix or None)``, 1 <= i <= N.

    The suffix is one of ``TABLES``.  Anything else is an error, an unknown
    suffix included: a misspelt ``:weighted`` must not fall back to the hash
    split unnoticed.
    """
    counts, colon, plan = spec.partition(":")
    if colon and plan not in TABLES:
        raise pytest.UsageError(f"{ENV_VAR}={spec!r}: the suffixes are {', '.join(':' + t for t in TABLES)}")
    # Digits only: int() would also take " 6", "+6" and "1_0".
    if not re.fullmatch(r"[0-9]+/[0-9]+", counts):
        raise pytest.UsageError(f"{ENV_VAR}={spec!r}: expected 'i/N' or 'i/N:{WEIGHTED}', e.g. '2/4'")
    i, n = (int(x) for x in counts.split("/"))
    if not 1 <= i <= n:
        raise pytest.UsageError(f"{ENV_VAR}={spec!r}: need 1 <= i <= N")
    return i, n, plan if colon else None


def _by_hash(path: str, n: int) -> int:
    # sha256, not hash(): Python's string hash is salted per process.
    return int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % n + 1


def shard_of(path: str, n: int) -> int:
    """The 1-based shard of a test file, out of ``n``, by the hash of its path alone."""
    return _by_hash(path, n)


def load_weights(path: Path = WEIGHTS_FILE) -> dict[str, int]:
    """``{test file: seconds}`` from the committed table; a bad table is an error."""
    try:
        seconds = json.loads(path.read_text(encoding="utf-8"))["seconds"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise pytest.UsageError(f"{path}: cannot read the weighted split's table ({exc!r})") from None
    if not isinstance(seconds, dict) or not seconds:
        raise pytest.UsageError(f"{path}: 'seconds' must be a non-empty {{test file: seconds}} table")
    bad = {k: v for k, v in seconds.items() if type(v) is not int or v <= 0}
    if bad:
        raise pytest.UsageError(f"{path}: every weight must be a positive whole number of seconds: {bad}")
    return seconds


def deal(weights: dict[str, int], n: int,
         apart: tuple[tuple[str, ...], ...] = APART) -> dict[str, int]:
    """``{file: shard}`` for the listed files: longest first, each onto the lightest shard.

    A file in a group of ``apart`` goes onto the lightest shard that holds
    no other file of its group (any shard, when the group outnumbers the
    shards).  Ties (equal seconds, equal loads) break on the path and on the
    lower shard number, so the result does not depend on the table's order.
    """
    group = {path: g for g, files in enumerate(apart) for path in files}
    taken: dict[int, set[int]] = {}
    load = [0] * n
    shards = {}
    for path, seconds in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0])):
        held = taken.setdefault(group[path], set()) if path in group else set()
        free = [k for k in range(n) if k not in held] or list(range(n))
        lightest = min(free, key=lambda k: (load[k], k))
        shards[path] = lightest + 1
        load[lightest] += seconds
        held.add(lightest)
    return shards


def weighted_shard_of(path: str, n: int, dealt: dict[str, int]) -> int:
    """The 1-based shard of a test file under the weighted split.

    ``dealt`` is ``deal(weights, n)``.  A file it does not list -- a new
    one, or one too light to be in the table -- goes by the hash of its
    path.
    """
    return dealt[path] if path in dealt else _by_hash(path, n)


class Plugin:
    def __init__(self, shard: int, of: int, weights: dict[str, int] | None = None):
        self.shard, self.of = shard, of
        self.weighted = weights is not None
        self._dealt = deal(weights, of) if weights is not None else {}
        self._session_start: float | None = None

    def pytest_sessionstart(self, session):
        self._session_start = time.perf_counter()

    @pytest.hookimpl(trylast=True)
    def pytest_collection_finish(self, session):
        # trylast: after every other plugin's work on the collected items.
        if self._session_start is None:
            return
        seconds = time.perf_counter() - self._session_start
        for name, value in zip(RUNNER_PROPERTIES,
                               (f"{seconds:.3f}", cpu_model(), str(usable_cpus()))):
            record_suite_property(session.config, name, value)

    def shard_of(self, path: str) -> int:
        if self.weighted:
            return weighted_shard_of(path, self.of, self._dealt)
        return shard_of(path, self.of)

    def pytest_collection_modifyitems(self, config, items):
        keep, drop = [], []
        for item in items:
            # The node id's path part is relative to the root dir with "/"
            # separators on every platform, so the hash is too.
            path = item.nodeid.split("::", 1)[0]
            (keep if self.shard_of(path) == self.shard else drop).append(item)
        if drop:
            config.hook.pytest_deselected(items=drop)
            items[:] = keep

    def pytest_report_header(self, config):
        how = "by file, weighted" if self.weighted else "by file"
        return f"test shard: {self.shard}/{self.of} ({how}; see tests/_sharding.py)"


def register(config, spec: str):
    shard, of, plan = parse(spec)
    weights = load_weights(TABLES[plan]) if plan else None
    config.pluginmanager.register(Plugin(shard, of, weights), "maddening-test-shard")
