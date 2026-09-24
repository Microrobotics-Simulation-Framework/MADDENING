"""The CI test shards must be a stable partition of the suite.

Shard ``i`` of a pull request has to hold the same test files as shard
``i`` of the base branch, because the XLA compilation cache a shard
restores was saved by that shard on the base branch.  These tests pin the
properties that make that true: the assignment is a function of the file
path alone, every test lands on exactly one shard, the pins are real and
in range, and CI runs the job count the pins were balanced for.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import _sharding

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_assignment_is_a_fixed_function_of_the_path():
    # Golden values: a change here re-deals every file, which invalidates
    # every shard's cache on the base branch.  It has to be deliberate.
    assert _sharding.shard_of("tests/core/test_graph_manager.py", 4) == 3
    assert _sharding.shard_of("tests/nodes/test_spring.py", 4) == 4
    assert _sharding.shard_of("tests/fmi/test_c_unit.py", 4) == 1
    assert _sharding.shard_of("tests/nodes/test_spring.py", 3) == 2


def test_pins_apply_only_at_the_job_count_they_were_balanced_for():
    path, pinned = next(iter(_sharding.PINS.items()))
    assert _sharding.shard_of(path, _sharding.PINS_FOR) == pinned
    other = _sharding.PINS_FOR + 1
    assert 1 <= _sharding.shard_of(path, other) <= other


def test_every_pin_names_a_real_file_and_a_real_shard():
    # A pin to a renamed file silently stops balancing anything.
    for path, shard in _sharding.PINS.items():
        assert (REPO_ROOT / path).is_file(), f"pinned file {path} does not exist"
        assert 1 <= shard <= _sharding.PINS_FOR, (path, shard)


@pytest.mark.parametrize("spec", ["0/4", "5/4", "a/b", "4", "1/4/2", ""])
def test_a_malformed_shard_spec_is_refused(spec):
    with pytest.raises(pytest.UsageError):
        _sharding.parse(spec)


def test_the_shards_partition_the_tests_and_keep_files_together():
    nodeids = [f"tests/{d}/test_{f}.py::test_{t}" for d in "abc" for f in range(7) for t in range(3)]
    nodeids += list(_sharding.PINS)  # pinned files too
    n = _sharding.PINS_FOR
    seen = {}
    for shard in range(1, n + 1):
        items = [SimpleNamespace(nodeid=nid) for nid in nodeids]
        dropped = []
        config = SimpleNamespace(hook=SimpleNamespace(
            pytest_deselected=lambda items: dropped.extend(items)))
        _sharding.Plugin(shard, n).pytest_collection_modifyitems(config, items)
        assert len(items) + len(dropped) == len(nodeids)
        for item in items:
            assert item.nodeid not in seen, f"{item.nodeid} on shards {seen[item.nodeid]} and {shard}"
            seen[item.nodeid] = shard
    assert set(seen) == set(nodeids)
    by_file = {}
    for nid, shard in seen.items():
        by_file.setdefault(nid.split("::")[0], set()).add(shard)
    assert all(len(s) == 1 for s in by_file.values()), "a file was split across shards"


def test_ci_runs_the_job_count_the_pins_were_balanced_for():
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    shards = re.search(r"^\s*shard:\s*\[([0-9, ]+)\]", ci, re.M)
    assert shards, "the test matrix has no shard axis"
    listed = [int(x) for x in shards.group(1).split(",")]
    n = _sharding.PINS_FOR
    assert listed == list(range(1, n + 1))
    assert f'MADDENING_TEST_SHARD: "${{{{ matrix.shard }}}}/{n}"' in ci
