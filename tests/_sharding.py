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
"""

from __future__ import annotations

import hashlib

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


def parse(spec: str) -> tuple[int, int]:
    """``"i/N"`` -> ``(i, N)``, 1 <= i <= N; anything else is an error."""
    try:
        i, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise pytest.UsageError(f"{ENV_VAR}={spec!r}: expected 'i/N', e.g. '2/4'") from None
    if not 1 <= i <= n:
        raise pytest.UsageError(f"{ENV_VAR}={spec!r}: need 1 <= i <= N")
    return i, n


def shard_of(path: str, n: int) -> int:
    """The 1-based shard of a test file, out of ``n``."""
    if n == PINS_FOR and path in PINS:
        return PINS[path]
    # sha256, not hash(): Python's string hash is salted per process.
    return int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % n + 1


class Plugin:
    def __init__(self, shard: int, of: int):
        self.shard, self.of = shard, of

    def pytest_collection_modifyitems(self, config, items):
        keep, drop = [], []
        for item in items:
            # The node id's path part is relative to the root dir with "/"
            # separators on every platform, so the hash is too.
            path = item.nodeid.split("::", 1)[0]
            (keep if shard_of(path, self.of) == self.shard else drop).append(item)
        if drop:
            config.hook.pytest_deselected(items=drop)
            items[:] = keep

    def pytest_report_header(self, config):
        return f"test shard: {self.shard}/{self.of} (by file; see tests/_sharding.py)"


def register(config, spec: str):
    shard, of = parse(spec)
    for path, pinned in PINS.items():
        if not 1 <= pinned <= PINS_FOR:
            raise pytest.UsageError(f"tests/_sharding.py pins {path} to shard {pinned} of {PINS_FOR}")
    config.pluginmanager.register(Plugin(shard, of), "maddening-test-shard")
