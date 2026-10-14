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

``PINS`` places a file on a shard explicitly, overriding the hash, to fix
an imbalance the hash leaves.  Pinning moves that file and nothing else.
Pins apply only when the job count equals ``PINS_FOR``; changing the job
count is a deliberate rebalance that re-deals every file anyway.

The slow lane (``slow-tests.yml``) asks for ``i/N:weighted`` instead.  It
restores no compilation cache, so nothing there depends on a file staying
on its shard, and its files are far from equal: one takes 48 minutes, and
by hash a single shard held nearly half of the lane.  A weighted split
deals the files of ``slow_lane_weights.json`` (measured seconds per file),
longest first, each onto the shard with the least time so far; a file the
table does not list goes by the hash above.  It is still a function of the
path and of committed data, and a file's tests still stay together.

GitHub's runners differ in speed: on the slow lane of 2026-10-11 to 13
one runner took 1.6 to 1.9 times as long as another over the same files.
A table measured on fast runners under-weighs what they ran (shard 4
ran 95 to 108 minutes against 68 predicted: its heaviest file had
been weighed on the two fastest runners of sixty).  So each shard
records how fast its runner was, as properties of its JUnit report
(:data:`RUNNER_PROPERTIES`): the seconds pytest took to collect the
suite, which is the same work on every shard and comes before any test,
and the runner's CPU.  ``scripts/slow_lane_weights.py`` scales each
shard's times to a runner of median speed by it.
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

#: ``{test file path relative to the repository root: shard}``.
#:
#: The six pins that level the six shards, from the per-file times of three
#: cold push runs on the release branch (38089224978, 38093868119, 38096508004;
#: 2026-10-10 and 11).  By hash alone the slowest of six jobs is predicted
#: at 33 to 37 minutes on those runs; with these, at 28 to 30 against a
#: mean of 24 to 30 (four shards ran 43 to 48).  Each was chosen as the one
#: file whose move off the slowest shard lowers the slowest shard most, over
#: the runs and both JAX lanes; every other file stays where its hash puts
#: it.  The minutes are the slower JAX lane's, the mean of the runs.
#: Recompute from the "Test durations" lane summaries of several runs when
#: a shard runs persistently long (one run carries its runners' speeds, a
#: factor of up to 1.6); moving a pin moves only that file.
PINS: dict[str, int] = {
    "tests/property/test_differential_coupling_topologies.py": 3,               # 5.1 min
    "tests/nodes/adaptive/test_wavelet_node.py": 3,                             # 2.4 min
    "tests/core/test_coupling_claims_in_every_domain.py": 6,                    # 2.4 min
    "tests/core/test_coupling_claims_graphs.py": 2,                             # 2.2 min
    "tests/test_examples_smoke.py": 6,                                          # 1.6 min
    "tests/verification/test_verify_node_harness.py": 1,                        # 1.2 min
}
#: The job count the pins were balanced for.
PINS_FOR = 6

#: The suffix of a shard spec that asks for the weighted split.
WEIGHTED = "weighted"
#: The job count ``slow-tests.yml`` runs per JAX version.  Ten, not the
#: six of the per-push lane: the measured lane is 635 minutes of tests
#: (each file's slower JAX version, 2026-10-10; 477 three days earlier),
#: so eight balanced shards sit at 79 minutes each and ten at 64.  Twelve
#: would sit at 53 (the slowest predicted at 56 with a job's fixed cost),
#: and the organisation's plan runs that many jobs at once (60 against the
#: lane's 24): they are the next step.  The heaviest file alone is 49
#: minutes, so past twelve the step is to split that file.
SLOW_LANE_SHARDS = 10
#: Measured seconds per test file for the weighted split, next to this
#: module.  ``scripts/slow_lane_weights.py`` writes it from the JUnit
#: artifacts of a slow-lane run and prints the minutes it predicts per shard.
WEIGHTS_FILE = Path(__file__).with_name("slow_lane_weights.json")

#: The JUnit report's property holding the seconds from the session's start
#: to the end of collection: the runner's speed, measured on the same work
#: on every shard (every test module of the suite is imported; the other
#: shards' tests are only deselected).
COLLECT_SECONDS_PROPERTY = "runner_collect_seconds"
#: Every property :class:`Plugin` adds to the report (``testsuite`` level).
RUNNER_PROPERTIES = (COLLECT_SECONDS_PROPERTY, "runner_cpu", "runner_cpus")


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


def parse(spec: str) -> tuple[int, int, bool]:
    """``"i/N"`` or ``"i/N:weighted"`` -> ``(i, N, weighted)``, 1 <= i <= N.

    Anything else is an error, an unknown suffix included: a misspelt
    ``:weighted`` must not fall back to the hash split unnoticed.
    """
    counts, colon, plan = spec.partition(":")
    if colon and plan != WEIGHTED:
        raise pytest.UsageError(f"{ENV_VAR}={spec!r}: the only suffix is ':{WEIGHTED}'")
    # Digits only: int() would also take " 6", "+6" and "1_0".
    if not re.fullmatch(r"[0-9]+/[0-9]+", counts):
        raise pytest.UsageError(f"{ENV_VAR}={spec!r}: expected 'i/N' or 'i/N:{WEIGHTED}', e.g. '2/4'")
    i, n = (int(x) for x in counts.split("/"))
    if not 1 <= i <= n:
        raise pytest.UsageError(f"{ENV_VAR}={spec!r}: need 1 <= i <= N")
    return i, n, bool(colon)


def _by_hash(path: str, n: int) -> int:
    # sha256, not hash(): Python's string hash is salted per process.
    return int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % n + 1


def shard_of(path: str, n: int) -> int:
    """The 1-based shard of a test file, out of ``n`` (the per-push split)."""
    if n == PINS_FOR and path in PINS:
        return PINS[path]
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


def deal(weights: dict[str, int], n: int) -> dict[str, int]:
    """``{file: shard}`` for the listed files: longest first, each onto the lightest shard.

    Ties (equal seconds, equal loads) break on the path and on the lower
    shard number, so the result does not depend on the table's order.
    """
    load = [0] * n
    shards = {}
    for path, seconds in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0])):
        lightest = min(range(n), key=lambda k: (load[k], k))
        shards[path] = lightest + 1
        load[lightest] += seconds
    return shards


def weighted_shard_of(path: str, n: int, dealt: dict[str, int]) -> int:
    """The 1-based shard of a test file under the weighted split.

    ``dealt`` is ``deal(weights, n)``.  A file it does not list -- a new
    one, or one too light to be in the table -- goes by the hash of its
    path, never by a pin: the pins balance the per-push lane.
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
    shard, of, weighted = parse(spec)
    for path, pinned in PINS.items():
        if not 1 <= pinned <= PINS_FOR:
            raise pytest.UsageError(f"tests/_sharding.py pins {path} to shard {pinned} of {PINS_FOR}")
    weights = load_weights() if weighted else None
    config.pluginmanager.register(Plugin(shard, of, weights), "maddening-test-shard")
