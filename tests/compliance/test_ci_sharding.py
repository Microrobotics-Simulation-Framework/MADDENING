"""The CI test shards must be a stable partition of the suite.

Shard ``i`` of a pull request has to hold the same test files as shard
``i`` of the base branch, because the XLA compilation cache a shard
restores was saved by that shard on the base branch.  These tests pin the
properties that make that true: the assignment is a function of the file
path alone, every test lands on exactly one shard, the pins are real and
in range, and CI runs the job count the pins were balanced for -- every
shard of it, each over the whole suite (no matrix leg excluded, no extra
``--ignore`` or selection flag), since either would drop tests from CI
while every other check here still passed.
"""

import hashlib
import re
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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
    # At any other count a pinned file hashes like every other file: a pin
    # that leaked would put it on a shard the count may not even have.
    for path, pinned in _sharding.PINS.items():
        assert _sharding.shard_of(path, _sharding.PINS_FOR) == pinned
        for n in (1, 2, 3, 5, 8):
            if n == _sharding.PINS_FOR:
                continue
            by_hash = int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % n + 1
            assert _sharding.shard_of(path, n) == by_hash, (path, n)


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


@pytest.mark.parametrize("workflow", ["ci.yml", "slow-tests.yml"])
def test_ci_runs_the_job_count_the_pins_were_balanced_for(workflow):
    ci = (REPO_ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    shards = re.search(r"^\s*shard:\s*\[([0-9, ]+)\]", ci, re.M)
    assert shards, f"{workflow}: the test matrix has no shard axis"
    listed = [int(x) for x in shards.group(1).split(",")]
    n = _sharding.PINS_FOR
    assert listed == list(range(1, n + 1)), workflow
    assert f'MADDENING_TEST_SHARD: "${{{{ matrix.shard }}}}/{n}"' in ci, workflow


def _workflow(name):
    return yaml.safe_load((REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def _pytest_args(job, step_name):
    (step,) = [s for s in job["steps"] if s.get("name") == step_name]
    # Continuation lines joined, so the invocation reads as one command.
    lines = [ln.strip() for ln in re.sub(r"\\\n", " ", step["run"]).splitlines()
             if "-m pytest" in ln and not ln.strip().startswith("#")]
    assert len(lines) == 1, lines
    # `${{ matrix.shard }}` -> `{{matrix.shard}}`, one shell word.
    tokens = shlex.split(re.sub(r"\$\{\{\s*(.*?)\s*\}\}", r"{{\1}}", lines[0]))
    return tokens[tokens.index("pytest") + 1:]


@pytest.mark.parametrize("workflow, job", [("ci.yml", "test"), ("slow-tests.yml", "slow")])
def test_no_matrix_leg_drops_or_adds_a_shard(workflow, job):
    # `exclude: [{shard: 4}]` would leave the shard axis intact -- so the
    # count check above passes -- while a quarter of the suite runs nowhere.
    matrix = _workflow(workflow)["jobs"][job]["strategy"]["matrix"]
    assert matrix["shard"] == list(range(1, _sharding.PINS_FOR + 1))
    for key in ("exclude", "include"):
        for leg in matrix.get(key) or []:
            assert "shard" not in leg, f"{workflow}: matrix {key} touches a shard: {leg}"


#: Every argument the default lane's pytest takes that does not change
#: which tests run.  Anything else fails the test below: an added
#: ``--ignore``, ``-k``, ``-m`` or ``--deselect`` would silently drop tests
#: from every shard at once.
REPORTING_ARGS = {"-v", "-rs", "--tb=short", "--durations=0", "--durations-min=1.0",
                  "-o", "junit_family=xunit1",
                  "--junitxml=test-results-shard{{matrix.shard}}.xml"}


@pytest.mark.parametrize("workflow, job, step, selection", [
    ("ci.yml", "test", "Run tests", ["tests/", "--ignore=tests/viz"]),
    ("slow-tests.yml", "slow", "Run full suite (slow lane)",
     ["tests/", "-m", "slow or not slow", "--ignore=tests/viz"]),
])
def test_the_sharded_lanes_run_the_whole_suite_but_viz(workflow, job, step, selection):
    args = _pytest_args(_workflow(workflow)["jobs"][job], step)
    chosen = [a for a in args if a not in REPORTING_ARGS]
    assert chosen == selection, (
        f"{workflow} {step!r}: pytest selects {chosen}, expected exactly {selection}; "
        "if a new argument only changes reporting, add it to REPORTING_ARGS")
    assert len(args) == len(set(args)), args
