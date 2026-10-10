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
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ENV_VAR = "MADDENING_TEST_SHARD"

#: ``{test file path relative to the repository root: shard}``.
#:
#: The eleven heaviest files, placed longest first onto the lightest shard,
#: from the per-file times of the first CI run after the slow-test triage
#: (PR #147's cold run, 2026-09-24, both JAX lanes, the slower lane per
#: file).  The previous eight pins, set before the triage moved about 180
#: tests to the slow lane, had drifted to 11.3 / 6.0 / 7.7 / 9.8 minutes
#: (slowest 30% over the mean); these give 8.8 / 8.8 / 8.8 / 8.5 (1.3%).
#: Recompute from the "Test durations" lane summary when a shard runs
#: persistently long; moving a pin moves only that file.
PINS: dict[str, int] = {
    "tests/nodes/adaptive/test_wavelet_node.py": 3,                 # 2.4 min
    "tests/core/test_coupling_convergence_reporting.py": 2,         # 1.8 min
    "tests/property/test_coupling_error_bound.py": 1,               # 1.8 min
    "tests/compliance/test_gate_scripts.py": 4,                     # 1.5 min
    "tests/core/test_coupling_error_bound.py": 2,                   # 1.3 min
    "tests/core/test_coupling_solver_equivalence.py": 1,            # 1.2 min
    "tests/core/test_coupling_non_finite_state.py": 3,              # 0.9 min
    "tests/verification/test_verify_node_harness.py": 4,            # 0.9 min
    "tests/nodes/adaptive/test_wavelet_engine.py": 3,               # 0.7 min
    "tests/core/test_coupling_while_default.py": 1,                 # 0.6 min
    "tests/core/test_coupling_precision_floor.py": 2,               # 0.6 min
}
#: The job count the pins were balanced for.
PINS_FOR = 4

#: The suffix of a shard spec that asks for the weighted split.
WEIGHTED = "weighted"
#: The job count ``slow-tests.yml`` runs per JAX version.  Ten, not the
#: four of the per-push lane: the measured lane is 635 minutes of tests
#: (each file's slower JAX version, 2026-10-10; 477 three days earlier),
#: so eight balanced shards sit at 79 minutes each and ten at 64.  Ten is
#: also the ceiling: with two JAX versions that is twenty jobs, the most
#: the repository runs at once.  The heaviest file alone is 49 minutes, so
#: the next step after ten is to split that file.
SLOW_LANE_SHARDS = 10
#: Measured seconds per test file for the weighted split, next to this
#: module.  ``scripts/slow_lane_weights.py`` writes it from the JUnit
#: artifacts of a slow-lane run and prints the minutes it predicts per shard.
WEIGHTS_FILE = Path(__file__).with_name("slow_lane_weights.json")


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
