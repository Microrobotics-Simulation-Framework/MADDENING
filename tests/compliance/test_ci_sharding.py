"""The CI test shards must be a stable partition of the suite.

Shard ``i`` of a pull request has to hold the same test files as shard
``i`` of the base branch, because the XLA compilation cache a shard
restores was saved by that shard on the base branch.  These tests pin the
properties that make that true: the assignment is a function of the file
path and of a committed table alone, every test lands on exactly one
shard, and CI runs the job count each lane declares -- every shard of it
(no matrix leg excluded or added).

Both lanes split by measured time (``i/N:weighted``, from
``tests/slow_lane_weights.json``, for the slow lane; ``i/N:weighted-push``,
from ``tests/per_push_weights.json``, for the per-push lane, whose shards
restore a compilation cache saved by the same shard of the base branch).
For each the tests pin that the split is still a partition by file, that the
table names real files and is the one the generator writes, that the deal is
balanced and under the lane's time target, and that the workflow asks for it
on every shard of its own count.

They also pin what the sharded lanes select, in each place a selection can
be written, since a narrower one drops tests from CI while every other
check here still passes:

* the pytest command line of both lanes (no extra ``--ignore``, ``-k``,
  ``-m`` or ``--deselect``);
* the environment pytest reads more arguments or plugins from
  (``PYTEST_ADDOPTS``, ``PYTEST_PLUGINS``), at workflow, job and step
  level, and in any ``run`` script;
* the shard itself: ``MADDENING_TEST_SHARD`` is named once per workflow,
  in the pytest step's env with the matrix value, and nowhere else (an
  inline ``MADDENING_TEST_SHARD=1/4`` in a ``run`` script would point every
  shard at one);
* ``[tool.pytest.ini_options]`` in ``pyproject.toml`` (``addopts`` exactly
  ``-m 'not slow'``, no key this file does not know) and the absence of
  any other pytest configuration file;
* every ``conftest.py`` under ``tests/`` (no collection hook, no
  ``collect_ignore``);
* the root ``conftest.py`` in action: a real pytest run through it, with
  ``MADDENING_TEST_SHARD=1/2``, keeps one of two files and deselects the
  other.

What none of it sees: a test that skips itself, and a ``pytestmark`` or
fixture in a test module that deselects or skips its own tests.
"""

import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests import _jax_timing, _sharding

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Each sharded lane's job count, and the suffix of its shard spec.
LANES = {
    "ci.yml": (_sharding.PER_PUSH_SHARDS, ":" + _sharding.WEIGHTED_PUSH),
    "slow-tests.yml": (_sharding.SLOW_LANE_SHARDS, ":" + _sharding.WEIGHTED),
}

#: Each weighted split: its spec suffix, table, lane count, the prefix of the
#: artifacts the table's runs came from, and the minutes of measured test
#: time the heaviest of its shards may be predicted to hold.  The slow lane's
#: wrapper kills a shard at 175; the per-push lane's target is the 30 minutes
#: a push (cold) is meant to take beyond its setup.
SPLITS = {
    _sharding.WEIGHTED: (_sharding.WEIGHTS_FILE, _sharding.SLOW_LANE_SHARDS, "slow-durations-", 100),
    _sharding.WEIGHTED_PUSH: (_sharding.PER_PUSH_WEIGHTS_FILE, _sharding.PER_PUSH_SHARDS,
                              "test-durations-", 30),
}


def test_the_assignment_is_a_fixed_function_of_the_path():
    # Golden values: a change here re-deals every file, which invalidates
    # every shard's cache on the base branch.  It has to be deliberate.
    assert _sharding.shard_of("tests/core/test_graph_manager.py", 4) == 3
    assert _sharding.shard_of("tests/nodes/test_spring.py", 4) == 4
    assert _sharding.shard_of("tests/fmi/test_c_unit.py", 4) == 1
    assert _sharding.shard_of("tests/nodes/test_spring.py", 3) == 2
    # ...and at the count the per-push lane runs.
    assert _sharding.shard_of("tests/core/test_graph_manager.py", 6) == 3
    assert _sharding.shard_of("tests/nodes/test_spring.py", 6) == 2
    assert _sharding.shard_of("tests/fmi/test_c_unit.py", 6) == 5


@pytest.mark.parametrize("spec", ["0/4", "5/4", "a/b", "4", "1/4/2", "", " 1/4", "1/ 4", "+1/4", "1/1_0",
                                  "0/6:weighted", "7/6:weighted", "1/6:", "1/6:weigthed",
                                  "1/6:weighted:weighted", ":weighted", "1/6 :weighted",
                                  "1/6:weighted-pushed", "1/6:weighted-push:weighted-push", "1/6:Weighted-push",
                                  "1/6:push", ":weighted-push"])
def test_a_malformed_shard_spec_is_refused(spec):
    # A misspelt suffix must not fall back to the hash split: the slow lane
    # would run, every test once, with one shard back at its timeout.
    with pytest.raises(pytest.UsageError):
        _sharding.parse(spec)


def test_a_shard_spec_says_which_split_it_asks_for():
    assert _sharding.parse("2/4") == (2, 4, None)
    assert _sharding.parse("6/6:weighted") == (6, 6, "weighted")
    assert _sharding.parse("6/6:weighted-push") == (6, 6, "weighted-push")


def _tree_test_files() -> list[str]:
    """Every file pytest collects tests from under ``tests/`` (its default patterns)."""
    files = {p.relative_to(REPO_ROOT).as_posix()
             for pattern in ("test_*.py", "*_test.py") for p in (REPO_ROOT / "tests").rglob(pattern)}
    assert len(files) > 500, len(files)
    return sorted(files)


@pytest.mark.parametrize("n, table", [
    (_sharding.PER_PUSH_SHARDS, None), (_sharding.SLOW_LANE_SHARDS, _sharding.WEIGHTS_FILE),
    (_sharding.PER_PUSH_SHARDS, _sharding.PER_PUSH_WEIGHTS_FILE),
    # The weighted splits at other counts: a table is dealt for whatever
    # count the spec names.
    (_sharding.PER_PUSH_SHARDS, _sharding.WEIGHTS_FILE), (1, _sharding.WEIGHTS_FILE),
    (8, _sharding.WEIGHTS_FILE), (8, _sharding.PER_PUSH_WEIGHTS_FILE)])
def test_the_shards_partition_the_tests_and_keep_files_together(n, table):
    weights = _sharding.load_weights(table) if table else None
    nodeids = [f"tests/{d}/test_{f}.py::test_{t}" for d in "abc" for f in range(7) for t in range(3)]
    # Every real test file, two tests each: the ones the table lists and the
    # ones it leaves to the hash.
    nodeids += [f"{path}::test_{t}" for path in _tree_test_files() for t in "ab"]
    seen = {}
    for shard in range(1, n + 1):
        items = [SimpleNamespace(nodeid=nid) for nid in nodeids]
        dropped = []
        config = SimpleNamespace(hook=SimpleNamespace(
            pytest_deselected=lambda items: dropped.extend(items)))
        _sharding.Plugin(shard, n, weights).pytest_collection_modifyitems(config, items)
        assert len(items) + len(dropped) == len(nodeids)
        for item in items:
            assert item.nodeid not in seen, f"{item.nodeid} on shards {seen[item.nodeid]} and {shard}"
            seen[item.nodeid] = shard
    assert set(seen) == set(nodeids)
    by_file = {}
    for nid, shard in seen.items():
        by_file.setdefault(nid.split("::")[0], set()).add(shard)
    assert all(len(s) == 1 for s in by_file.values()), "a file was split across shards"


#: --- the weighted splits -----------------------------------------------------

#: Kept under its old name: the slow lane's wrapper kills a shard at 175; the
#: lane is rebalanced (more shards, or a file split) long before that, when a
#: refreshed table predicts over this.
SLOW_SHARD_TARGET_MIN = SPLITS[_sharding.WEIGHTED][3]

_SPLIT_IDS = sorted(SPLITS)


def _table(split: str = _sharding.WEIGHTED) -> dict:
    return json.loads(SPLITS[split][0].read_text(encoding="utf-8"))


@pytest.mark.parametrize("split", _SPLIT_IDS)
def test_every_weighted_file_is_a_real_test_file(split):
    """An entry for a deleted or renamed file is dealt a slot nothing runs in.

    Every test still runs once, but the shard that "holds" the ghost is
    lighter than the table says and the others heavier; a heavy file's
    rename would quietly undo the balance.
    """
    file = SPLITS[split][0]
    weights = _sharding.load_weights(file)
    real = set(_tree_test_files())
    ghosts = sorted(set(weights) - real)
    assert not ghosts, (
        f"tests/{file.name} lists files that are not test files in the tree: {ghosts}; "
        "regenerate it (scripts/slow_lane_weights.py) or remove the entries")


@pytest.mark.parametrize("split", _SPLIT_IDS)
def test_the_weights_table_says_where_its_numbers_came_from(split):
    table = _table(split)
    prefix = SPLITS[split][2]
    assert table["runs"], "no run recorded"
    for run in table["runs"]:
        assert re.fullmatch(r"[0-9]+", run["run"]), run
        assert re.fullmatch(r"[0-9a-f]{7,40}", run["commit"]), run
        assert run["artifacts"] and all(a.startswith(prefix) for a in run["artifacts"]), run
    floor = table["min_seconds"]
    assert type(floor) is int and floor >= 1
    light = {p: s for p, s in table["seconds"].items() if s < floor}
    assert not light, f"entries under the table's own {floor} s floor: {light}"
    assert list(table["seconds"]) == sorted(table["seconds"]), "the table is written sorted by path"
    if split != _sharding.WEIGHTED:
        return
    # The slow lane's weights are for a runner of median speed: every report
    # they were taken from carried its runner's collection time, so none is
    # weighed by the luck of the runner that ran it, and none by its CPU
    # model (the fallback for a report without one).
    speed = table["runner_speed"]
    assert speed["property"] == _sharding.COLLECT_SECONDS_PROPERTY, speed
    assert speed["reference_seconds"] > 0 and speed["exponent"] == _sharding.SPEED_EXPONENT, speed
    reports = sum(len(run["artifacts"]) for run in table["runs"])
    assert (speed["reports_scaled"], speed["reports_by_cpu_model"], speed["reports_as_measured"]) \
        == (reports, 0, 0), speed
    assert speed["cpu_factors"] == {} and speed["cpu_reference"] is None, speed


@pytest.mark.parametrize("table", [
    {}, {"seconds": {}}, {"seconds": {"tests/test_a.py": 0}}, {"seconds": {"tests/test_a.py": -3}},
    {"seconds": {"tests/test_a.py": 2.5}}, {"seconds": {"tests/test_a.py": "12"}},
    {"seconds": {"tests/test_a.py": True}}, {"seconds": ["tests/test_a.py"]}, "not json"])
def test_a_weights_table_that_cannot_be_dealt_is_refused(tmp_path, table):
    # Fail closed: a weighted run with no usable table must not quietly
    # become a hash split (or, with a zero or negative weight, a lopsided one).
    path = tmp_path / "weights.json"
    path.write_text(table if isinstance(table, str) else json.dumps(table), encoding="utf-8")
    with pytest.raises(pytest.UsageError):
        _sharding.load_weights(path)
    with pytest.raises(pytest.UsageError):
        _sharding.load_weights(tmp_path / "missing.json")


def test_the_deal_puts_the_longest_files_on_the_lightest_shards():
    # By hand, loads after each file: a 50 -> shard 1 (50, 0, 0); b 30 -> 2
    # (50, 30, 0); c 20 -> 3 (50, 30, 20); d 20 -> 3 (50, 30, 40); e 10 -> 2.
    weights = {"t/e.py": 10, "t/a.py": 50, "t/d.py": 20, "t/b.py": 30, "t/c.py": 20}
    assert _sharding.deal(weights, 3) == {"t/a.py": 1, "t/b.py": 2, "t/c.py": 3, "t/d.py": 3, "t/e.py": 2}
    # The table's order is not an input.
    assert _sharding.deal(dict(reversed(list(weights.items()))), 3) == _sharding.deal(weights, 3)
    # Equal weights tie on the path, equal loads on the lower shard.
    assert _sharding.deal({"t/y.py": 7, "t/x.py": 7}, 4) == {"t/x.py": 1, "t/y.py": 2}
    assert _sharding.deal(weights, 1) == dict.fromkeys(weights, 1)


@pytest.mark.parametrize("split", _SPLIT_IDS)
def test_a_file_the_table_does_not_list_goes_by_the_hash_of_its_path(split):
    file, n = SPLITS[split][:2]
    dealt = _sharding.deal(_sharding.load_weights(file), n)
    for path in ("tests/core/test_not_measured_yet.py", "tests/new/test_added_since.py"):
        by_hash = int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % n + 1
        assert path not in dealt
        assert _sharding.weighted_shard_of(path, n, dealt) == by_hash
    for path, shard in dealt.items():
        assert _sharding.weighted_shard_of(path, n, dealt) == shard


@pytest.mark.parametrize("split", _SPLIT_IDS)
def test_the_weighted_shards_are_balanced_and_under_the_target(split):
    """The table's own prediction for the count the workflow runs.

    Longest-first dealing leaves any two shards within the heaviest file of
    each other; and the heaviest shard has to be under the lane's target,
    or the table was refreshed into a lane that needs more shards.
    """
    file, n, _, target = SPLITS[split]
    weights = _sharding.load_weights(file)
    dealt = _sharding.deal(weights, n)
    assert set(dealt) == set(weights) and set(dealt.values()) == set(range(1, n + 1))
    load = [0] * n
    for path, shard in dealt.items():
        load[shard - 1] += weights[path]
    assert max(load) - min(load) <= max(weights.values()), load
    assert max(load) / 60 < target, (
        f"the weighted split predicts {max(load) / 60:.0f} min on the heaviest of {n} {split} shards "
        f"(target: under {target}); raise the shard count or split the heaviest file")
    # One file alone must leave room on its shard, or no count can balance it.
    assert max(weights.values()) / 60 < target


def test_a_weighted_spec_registers_the_weighted_split_from_the_shipped_table():
    registered = []
    config = SimpleNamespace(pluginmanager=SimpleNamespace(
        register=lambda plugin, name: registered.append((plugin, name))))
    n = _sharding.SLOW_LANE_SHARDS
    m = _sharding.PER_PUSH_SHARDS
    _sharding.register(config, f"2/{n}:weighted")
    _sharding.register(config, f"2/{n}")
    _sharding.register(config, f"2/{m}:weighted-push")
    (weighted, name), (plain, _), (push, _) = registered
    assert name == "maddening-test-shard"
    assert (weighted.shard, weighted.of, weighted.weighted) == (2, n, True)
    assert (plain.shard, plain.of, plain.weighted) == (2, n, False)
    assert (push.shard, push.of, push.weighted) == (2, m, True)
    heaviest = max(_sharding.load_weights().items(), key=lambda kv: (kv[1], kv[0]))[0]
    # Longest first onto the lightest shard: the heaviest file opens shard 1.
    assert weighted.shard_of(heaviest) == 1
    heaviest_push = max(_sharding.load_weights(_sharding.PER_PUSH_WEIGHTS_FILE).items(),
                        key=lambda kv: (kv[1], kv[0]))[0]
    assert push.shard_of(heaviest_push) == 1
    # Each suffix reads its own table: the per-push deal is not the hash split.
    listed = _sharding.load_weights(_sharding.PER_PUSH_WEIGHTS_FILE)
    assert any(push.shard_of(p) != _sharding.shard_of(p, m) for p in listed)
    assert weighted.pytest_report_header(config) == (
        f"test shard: 2/{n} (by file, weighted; see tests/_sharding.py)")
    assert plain.pytest_report_header(config) == f"test shard: 2/{n} (by file; see tests/_sharding.py)"


def _write_report(tmp_path, run, artifact, cases):
    directory = tmp_path / run / artifact
    directory.mkdir(parents=True)
    body = "".join(f'<testcase classname="x" name="t{i}" file="{f}" time="{s}"/>'
                   for i, (f, s) in enumerate(cases))
    (directory / "test-results.xml").write_text(
        f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>', encoding="utf-8")


def test_the_weights_generator_writes_a_table_the_split_reads(tmp_path):
    """``scripts/slow_lane_weights.py`` on two small runs: slower lane, later run, floor, ghosts."""
    report = lambda *a: _write_report(tmp_path, *a)  # noqa: E731

    real_a, real_b, real_c = _tree_test_files()[:3]
    # First run: both lanes, both shards.  a: 30 + 31 s on the old lane, 40 s on the new.
    report("r1", "slow-durations-py3.12-jax0.10.2-shard1of2", [(real_a, 30.0), (real_a, 31.0), (real_b, 4.0)])
    report("r1", "slow-durations-py3.12-jax0.10.2-shard2of2", [(real_c, 99.0), ("tests/test_gone.py", 500.0)])
    report("r1", "slow-durations-py3.12-jax0.11.2-shard1of2", [(real_a, 40.0), (real_b, 3.0)])
    report("r1", "slow-durations-py3.12-jax0.11.2-shard2of2", [(real_c, 90.0)])
    # Second run lost its first shard on one lane; it re-measured c there at 20 s.
    report("r2", "slow-durations-py3.12-jax0.10.2-shard2of2", [(real_c, 20.4)])
    out = tmp_path / "weights.json"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "slow_lane_weights.py"),
         "--run", f"11={tmp_path / 'r1'}", "--run", f"22={tmp_path / 'r2'}",
         "--commit", "11=abc1234", "--commit", "22=def5678", "--shards", "2",
         "--aggregate", "last", "--write", "--output", str(out)],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    table = json.loads(out.read_text(encoding="utf-8"))
    # a: its slower lane (61 s, two tests summed).  b: under the floor.  c: the
    # later run's 20 s on one lane, the first run's 90 s on the other.
    assert table["seconds"] == {real_a: 61, real_c: 90}
    # No --lane: the slow lane's table; --aggregate last: the last run wins.
    assert "slow lane" in table["what"] and "the later" in table["measured"], table
    assert [(r["run"], r["commit"]) for r in table["runs"]] == [("11", "abc1234"), ("22", "def5678")]
    assert table["runs"][1]["artifacts"] == ["slow-durations-py3.12-jax0.10.2-shard2of2"]
    assert "not a file in the tree, left out: tests/test_gone.py" in proc.stdout
    # Reports written before the runner's fields were recorded are taken as measured, and said to be.
    assert table["runner_speed"]["reports_as_measured"] == 5, table["runner_speed"]
    assert (f"5 reports taken as measured (no {_sharding.COLLECT_SECONDS_PROPERTY} and no "
            f"{_sharding.CPU_PROPERTY} in the report)") in proc.stdout, proc.stdout
    assert _sharding.load_weights(out) == table["seconds"]
    # The prediction, per lane: c (90) opens shard 1, a (61) shard 2; b is
    # unlisted and goes by hash.
    b_shard = _sharding.weighted_shard_of(real_b, 2, {})
    lane = [90.0, 40.0]
    lane[b_shard - 1] += 3.0
    assert f"{'weighted, 2 shards':<34}{lane[0] / 60:6.1f} {lane[1] / 60:6.1f}" in proc.stdout, proc.stdout


def test_the_weights_generator_scales_each_shard_to_a_runner_of_median_speed(tmp_path):
    """A shard's times are scaled by its runner's collection time before they are weighed.

    One run, one lane, three shards on runners that collected in 20, 40 and
    80 s (the median 40).  The same 100 s file weighs 100 s from the median
    runner, 100 * (40 / 80) ** 1.5 from the slow one and 100 * (40 / 20) **
    1.5 from the fast one; with ``--no-normalise`` all three weigh 100.  The
    reports name a CPU too, and it changes nothing: a report with a
    collection time is scaled by that alone, never also by its CPU model.
    On 2026-10-11 the heaviest file was weighed on the two fastest runners
    of sixty, at 30 minutes against 47 on a median one, and its shard ran
    95 to 108 minutes against 68 predicted.
    """
    real_a, real_b, real_c = _tree_test_files()[:3]
    run = tmp_path / "r1"
    for shard, (path, collect) in enumerate([(real_a, 20.0), (real_b, 40.0), (real_c, 80.0)], 1):
        directory = run / f"slow-durations-py3.12-jax0.10.2-shard{shard}of3"
        directory.mkdir(parents=True)
        (directory / "report.xml").write_text(
            '<testsuites><testsuite name="pytest"><properties>'
            f'<property name="{_sharding.COLLECT_SECONDS_PROPERTY}" value="{collect}"/>'
            '<property name="runner_cpu" value="a CPU"/></properties>'
            f'<testcase classname="x" name="t" file="{path}" time="100.0"/>'
            '</testsuite></testsuites>', encoding="utf-8")

    def generate(*extra):
        out = tmp_path / f"weights{len(extra)}.json"
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "slow_lane_weights.py"),
             "--run", f"11={run}", "--commit", "11=abc1234", "--shards", "3",
             "--write", "--output", str(out), *extra],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return json.loads(out.read_text(encoding="utf-8")), proc.stdout

    table, out = generate()
    assert table["seconds"] == {real_a: round(100 * 2 ** 1.5), real_b: 100, real_c: round(100 * 0.5 ** 1.5)}
    assert table["runner_speed"] == {"property": _sharding.COLLECT_SECONDS_PROPERTY, "reference_seconds": 40.0,
                                     "exponent": 1.5, "reports_scaled": 3, "reports_by_cpu_model": 0,
                                     "cpu_reference": None, "cpu_factors": {}, "reports_as_measured": 0}
    assert "3 reports scaled to a runner that collects in 40.0 s" in out, out
    assert "scaled to a runner of median speed by the time it took to collect" in table["measured"], table
    assert "CPU model" not in table["measured"] and "CPU model" not in out, out
    table, out = generate("--no-normalise")
    assert table["seconds"] == {real_a: 100, real_b: 100, real_c: 100}
    assert table["runner_speed"]["reports_as_measured"] == 3, table["runner_speed"]
    assert "every report taken as measured (--no-normalise)" in out, out


def test_the_per_push_lane_table_is_the_median_of_the_runs_it_names(tmp_path):
    """``--lane per-push`` (median by default): one slow runner in three does not set a file's weight.

    Three cold runs measured a on one JAX lane at 60, 600 and 70 s -- the
    middle one is 70 -- and b at 100 s on every run; c was added after the
    first run and was measured twice (the median of two is their mean).
    """
    report = lambda *a: _write_report(tmp_path, *a)  # noqa: E731
    real_a, real_b, real_c = _tree_test_files()[:3]
    lane = "test-durations-py3.12-jax0.10.2-shard1of1"
    report("r1", lane, [(real_a, 60.0), (real_b, 100.0)])
    report("r2", lane, [(real_a, 600.0), (real_b, 100.0), (real_c, 30.0)])
    report("r3", lane, [(real_a, 70.0), (real_b, 100.0), (real_c, 50.0)])
    out = tmp_path / "per_push.json"
    command = [sys.executable, str(REPO_ROOT / "scripts" / "slow_lane_weights.py"), "--lane", "per-push",
               *(a for r in ("r1", "r2", "r3") for a in ("--run", f"{r[1]}={tmp_path / r}")),
               "--write", "--output", str(out)]
    proc = subprocess.run([*command, "--aggregate", "median"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    table = json.loads(out.read_text(encoding="utf-8"))
    assert table["seconds"] == {real_a: 70, real_b: 100, real_c: 40}
    assert "median" in table["measured"]
    assert "per-push lane" in table["what"], table
    # The lane's own shard count is the default prediction.
    assert f"weighted, {_sharding.PER_PUSH_SHARDS} shards" in proc.stdout
    # The median is the default now, for both lanes; the last run's is --aggregate last.
    proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["seconds"] == {real_a: 70, real_b: 100, real_c: 40}
    proc = subprocess.run([*command, "--aggregate", "last"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["seconds"] == {real_a: 70, real_b: 100, real_c: 50}


@pytest.mark.parametrize("split", _SPLIT_IDS)
def test_the_shipped_weights_table_is_what_the_generator_writes(split):
    # "do not edit by hand": keys in the generator's order and nothing else.
    # The per-push table was written before the generator recorded the
    # runners' speed and has no ``runner_speed`` until it is next written.
    keys = ["what", "measured", "runs", "runner_speed", "min_seconds", "seconds"]
    if split != _sharding.WEIGHTED and "runner_speed" not in _table(split):
        keys.remove("runner_speed")
    assert list(_table(split)) == keys


#: Every way a workflow writes a lane's shard count besides the matrix: the
#: shard spec, artifact and cache names (``of6``), step titles (``of 6``,
#: ``of 6 shard``) and the slow lane's issue titles (``shard ${shard}/10``).
_COUNT_MENTIONS = re.compile(
    r"matrix\.shard \}\}/([0-9]+)|matrix\.shard \}\}of([0-9]+)|matrix\.shard \}\} of ([0-9]+)"
    r"|\$\{shard\}/([0-9]+)|\bof ([0-9]+) shards?\b|-ne ([0-9]+) \]")


@pytest.mark.parametrize("workflow", sorted(LANES))
def test_each_lane_runs_every_shard_of_its_own_count(workflow):
    ci = (REPO_ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    shards = re.search(r"^\s*shard:\s*\[([0-9, ]+)\]", ci, re.M)
    assert shards, f"{workflow}: the test matrix has no shard axis"
    listed = [int(x) for x in shards.group(1).split(",")]
    n, suffix = LANES[workflow]
    assert listed == list(range(1, n + 1)), workflow
    assert f'MADDENING_TEST_SHARD: "${{{{ matrix.shard }}}}/{n}{suffix}"' in ci, workflow
    # The count written anywhere else: artifact names, titles, the issue
    # titles.  A stale one mislabels a shard's report ("shard 5 of 4").
    mentions = [(m.group(0), int(next(g for g in m.groups() if g))) for m in _COUNT_MENTIONS.finditer(ci)]
    assert len(mentions) >= 4, mentions
    stale = [text for text, count in mentions if count != n]
    assert not stale, f"{workflow} runs {n} shards but also says: {stale}"


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
    # `exclude: [{shard: 6}]` would leave the shard axis intact -- so the
    # count check above passes -- while a sixth of the suite runs nowhere.
    matrix = _workflow(workflow)["jobs"][job]["strategy"]["matrix"]
    assert matrix["shard"] == list(range(1, LANES[workflow][0] + 1))
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


# ---------------------------------------------------------------------------
# Selection written anywhere but the command line
# ---------------------------------------------------------------------------

#: Environment variables pytest takes more arguments (``PYTEST_ADDOPTS``) or
#: plugins (``PYTEST_PLUGINS``) from.  Set in a workflow, either narrows
#: every shard's selection where no check of the command line looks.
PYTEST_ARGUMENT_ENV = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")


def _env_blocks(workflow: dict):
    """``(where, env)`` for the workflow, each job and each step."""
    yield "the workflow", workflow.get("env") or {}
    for job_id, job in workflow["jobs"].items():
        yield f"job {job_id}", job.get("env") or {}
        for step in job.get("steps", []):
            yield f"job {job_id} step {step.get('name')!r}", step.get("env") or {}


@pytest.mark.parametrize("workflow", ["ci.yml", "slow-tests.yml"])
def test_no_workflow_environment_adds_pytest_arguments(workflow):
    """``PYTEST_ADDOPTS: "--deselect tests/core"`` in a step's env drops tests/core
    from every shard, and the command-line check above never sees it.

    Every env block is read (workflow, job, step), and every ``run`` script
    too: ``echo PYTEST_ADDOPTS=... >> "$GITHUB_ENV"`` or an ``export`` sets
    it for the steps after.
    """
    wf = _workflow(workflow)
    offenders = [f"{where}: {key}" for where, env in _env_blocks(wf)
                 for key in env if key in PYTEST_ARGUMENT_ENV]
    for job_id, job in wf["jobs"].items():
        for step in job.get("steps", []):
            offenders += [f"job {job_id} step {step.get('name')!r}: {var} in its script"
                          for var in PYTEST_ARGUMENT_ENV if var in step.get("run", "")]
    assert not offenders, (
        f"{workflow} sets pytest arguments outside the command line, where the selection "
        f"checks do not look: {offenders}")


#: The one place each sharded lane may name ``MADDENING_TEST_SHARD``: the
#: env of its pytest step, set from the matrix.
SHARD_SETTING = {
    "ci.yml": ("test", "Run tests"),
    "slow-tests.yml": ("slow", "Run full suite (slow lane)"),
}


def _mentions(node, path=()):
    """``(path, text)`` for every mapping key and string in a parsed workflow that names
    ``MADDENING_TEST_SHARD``."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and "MADDENING_TEST_SHARD" in key:
                yield (*path, key), key
            yield from _mentions(value, (*path, key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _mentions(value, (*path, i))
    elif isinstance(node, str) and "MADDENING_TEST_SHARD" in node:
        yield path, node


@pytest.mark.parametrize("workflow", sorted(SHARD_SETTING))
def test_each_lane_sets_its_shard_once_from_the_matrix(workflow):
    """``MADDENING_TEST_SHARD=1/6 python -m pytest ...`` points all six shards at one.

    Five sixths of the suite then runs nowhere, and the checks above still
    pass: they read the step's env, and the pytest arguments after
    ``pytest``.  So the variable may be named once per workflow -- in the
    pytest step's env, with the matrix value -- and nowhere else: no
    ``run:`` script (an inline prefix, an ``export``, a write to
    ``$GITHUB_ENV``), no other env block, no other input.
    """
    wf = _workflow(workflow)
    job_id, step_name = SHARD_SETTING[workflow]
    steps = wf["jobs"][job_id]["steps"]
    (index,) = [i for i, s in enumerate(steps) if s.get("name") == step_name]
    allowed = ("jobs", job_id, "steps", index, "env", "MADDENING_TEST_SHARD")
    n, suffix = LANES[workflow]
    assert steps[index]["env"]["MADDENING_TEST_SHARD"] == f"${{{{ matrix.shard }}}}/{n}{suffix}"
    elsewhere = [f"{'.'.join(map(str, where))}: {text.strip()[:120]!r}"
                 for where, text in _mentions(wf) if where != allowed]
    assert not elsewhere, (
        f"{workflow} names MADDENING_TEST_SHARD outside {step_name!r}'s env, where it can override "
        f"the matrix's shard: {elsewhere}")


def test_the_shard_scan_sees_every_place_the_variable_can_be_written():
    # The scan above is only a guard if it can fire.
    wf = {"env": {"MADDENING_TEST_SHARD": "1/4"},
          "jobs": {"j": {"steps": [
              {"name": "s", "env": {"MADDENING_TEST_SHARD": "${{ matrix.shard }}/4"},
               "run": "MADDENING_TEST_SHARD=1/4 python -m pytest tests/"},
              {"run": 'echo "MADDENING_TEST_SHARD=1/4" >> "$GITHUB_ENV"'},
              {"uses": "x", "with": {"script": "process.env.MADDENING_TEST_SHARD = '1/4'"}}]}}}
    assert [where for where, _ in _mentions(wf)] == [
        ("env", "MADDENING_TEST_SHARD"),
        ("jobs", "j", "steps", 0, "env", "MADDENING_TEST_SHARD"),
        ("jobs", "j", "steps", 0, "run"),
        ("jobs", "j", "steps", 1, "run"),
        ("jobs", "j", "steps", 2, "with", "script")]


def test_the_environment_check_sees_every_place_a_variable_can_be_set():
    # The scan above is only a guard if it can fire.
    wf = {"env": {"PYTEST_ADDOPTS": "-x"},
          "jobs": {"j": {"env": {"PYTEST_PLUGINS": "p"},
                         "steps": [{"name": "s", "env": {"PYTEST_ADDOPTS": "-k x"}}]}}}
    found = [(where, key) for where, env in _env_blocks(wf) for key in env
             if key in PYTEST_ARGUMENT_ENV]
    assert found == [("the workflow", "PYTEST_ADDOPTS"), ("job j", "PYTEST_PLUGINS"),
                     ("job j step 's'", "PYTEST_ADDOPTS")]


#: ``[tool.pytest.ini_options]``, key by key.  ``addopts`` reaches every
#: lane's pytest, so anything added to it (``--ignore=tests/fmi``, ``-k``, a
#: wider ``-m``) changes what every shard runs; a new key (``norecursedirs``,
#: ``python_files``, ``junit_duration_report``) can do the same, so an
#: unknown one fails here until it is added with what it is for.
EXPECTED_INI = {
    "addopts": "-m 'not slow'",
    "testpaths": ["tests"],
    "filterwarnings": None,   # reporting, not selection: any value
    "markers": None,          # reporting, not selection: any value
}
#: Keys that may be absent, with the one value each may take when present.
#: ``junit_duration_report = "call"`` would drop setup and teardown -- where
#: a module fixture's compile is paid -- from every time the budget judges.
OPTIONAL_INI = {"junit_duration_report": "total"}


def test_the_pytest_configuration_selects_nothing_beyond_the_slow_mark():
    ini = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "tool"]["pytest"]["ini_options"]
    assert set(ini) - set(OPTIONAL_INI) == set(EXPECTED_INI), (
        f"[tool.pytest.ini_options] keys are {sorted(ini)}; a new key can change what every "
        "lane selects or what the time budget measures -- add it to EXPECTED_INI with why")
    for key, want in EXPECTED_INI.items():
        if want is not None:
            assert ini[key] == want, f"[tool.pytest.ini_options] {key} = {ini[key]!r}, expected {want!r}"
    for key, want in OPTIONAL_INI.items():
        assert ini.get(key, want) == want, f"[tool.pytest.ini_options] {key} = {ini[key]!r}, expected {want!r}"
    # pytest reads the first of these it finds before pyproject.toml, and a
    # `pytest.ini` wins even when it is empty: every setting above would go.
    for name, section in (("pytest.ini", ""), (".pytest.ini", ""), ("tox.ini", "[pytest]"),
                          ("setup.cfg", "[tool:pytest]")):
        path = REPO_ROOT / name
        if path.is_file():
            assert section and section not in path.read_text(encoding="utf-8"), (
                f"{name} holds pytest configuration, which pytest reads instead of pyproject.toml")


#: Names that, defined in a ``conftest.py``, choose which tests are collected
#: or kept.  The root conftest registers ``tests/_sharding.py``'s
#: ``pytest_collection_modifyitems`` as a plugin; it defines none itself.
COLLECTION_HOOKS = {"collect_ignore", "collect_ignore_glob", "pytest_ignore_collect",
                    "pytest_collection_modifyitems", "pytest_deselected", "pytest_collect_file",
                    "pytest_pycollect_makemodule", "pytest_plugins"}


def _collection_hooks(source: str) -> list[str]:
    found = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in COLLECTION_HOOKS:
            found.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            found += [t.id for t in targets if isinstance(t, ast.Name) and t.id in COLLECTION_HOOKS]
    return found


def test_no_conftest_chooses_which_tests_are_collected():
    """``collect_ignore = ["fmi"]`` in a conftest drops a directory from every shard."""
    offenders = {}
    for path in sorted((REPO_ROOT / "tests").rglob("conftest.py")):
        hooks = _collection_hooks(path.read_text(encoding="utf-8"))
        if hooks:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = hooks
    assert not offenders, (
        f"these conftests choose which tests run, where no selection check looks: {offenders}")


def test_the_conftest_scan_finds_what_it_is_for():
    source = ("collect_ignore = ['fmi']\n"
              "collect_ignore_glob: list = ['*.py']\n"
              "def pytest_collection_modifyitems(config, items):\n    pass\n"
              "def pytest_configure(config):\n    pass\n"
              "def ball_node():\n    pass\n")
    assert _collection_hooks(source) == [
        "collect_ignore", "collect_ignore_glob", "pytest_collection_modifyitems"]


# ---------------------------------------------------------------------------
# The root conftest switches on what CI asks for
# ---------------------------------------------------------------------------

#: Time the kept test spends in a fixture's setup, which the JUnit time the
#: budget judges must include (``junit_duration_report = "total"``, the
#: default; ``"call"`` would report the test body alone).
SETUP_SECONDS = 0.2


def _files_on_each_shard_of_two() -> tuple[str, str]:
    """A test file path on shard 1 of 2 and one on shard 2 of 2."""
    found: dict[int, str] = {}
    for i in range(100):
        rel = f"tests/test_scratch_{i}.py"
        found.setdefault(_sharding.shard_of(rel, 2), rel)
        if len(found) == 2:
            return found[1], found[2]
    raise AssertionError("no two of 100 names hash to different shards of 2")


@pytest.fixture(scope="module")
def shard_run(tmp_path_factory):
    """One pytest run of a two-file tree through the real root conftest and ini.

    The scratch tree holds copies of ``tests/conftest.py`` (and the two
    modules it registers) and of ``pyproject.toml``, and runs as one CI
    shard does: ``MADDENING_TEST_SHARD=1/2`` and ``MADDENING_TEST_JAX_TIMING=1``.
    """
    root = tmp_path_factory.mktemp("shard_run")
    (root / "tests").mkdir()
    for name in ("__init__.py", "conftest.py", "_sharding.py", "_jax_timing.py"):
        shutil.copy(REPO_ROOT / "tests" / name, root / "tests" / name)
    shutil.copy(REPO_ROOT / "pyproject.toml", root / "pyproject.toml")
    kept, dropped = _files_on_each_shard_of_two()
    (root / kept).write_text(
        "import time\n\nimport jax\nimport jax.numpy as jnp\nimport pytest\n\n\n"
        "@pytest.fixture\n"
        "def compiled():\n"
        f"    time.sleep({SETUP_SECONDS})\n"
        "    return jax.jit(lambda x: x * 1.6180339 + 0.5772156)(jnp.ones(3))\n\n\n"
        "def test_kept(compiled):\n"
        "    assert compiled.shape == (3,)\n")
    (root / dropped).write_text("def test_dropped():\n    pass\n")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("MADDENING_TEST_", "PYTEST_", "JAX_COMPILATION_CACHE",
                                "JAX_PERSISTENT_CACHE"))}
    env.update(MADDENING_TEST_SHARD="1/2", MADDENING_TEST_JAX_TIMING="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", JAX_PLATFORMS="cpu")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-p", "no:cacheprovider",
         "--junitxml=report.xml", "-o", "junit_family=xunit1"],
        cwd=root, env=env, capture_output=True, text=True, timeout=300)
    report = ET.parse(root / "report.xml").getroot() if (root / "report.xml").is_file() \
        else ET.Element("testsuites")
    cases = list(report.iter("testcase"))
    return SimpleNamespace(proc=proc, cases=cases, kept=kept, dropped=dropped, report=report)


def test_the_root_conftest_keeps_only_this_shards_files(shard_run):
    """``MADDENING_TEST_SHARD`` must reach ``tests/_sharding.py`` through the conftest.

    If the conftest stopped registering the plugin, every CI shard would
    run the whole suite: six times the CI time, no test dropped, and every
    check of ``_sharding.py`` itself still green.
    """
    out = shard_run.proc.stdout + shard_run.proc.stderr
    assert shard_run.proc.returncode == 0, out[-3000:]
    assert "test shard: 1/2 (by file; see tests/_sharding.py)" in out, out[-3000:]
    assert re.search(r"\b1 passed, 1 deselected\b", out), out[-3000:]
    assert [c.get("file") for c in shard_run.cases] == [shard_run.kept], (
        f"the run kept {[c.get('file') for c in shard_run.cases]}, expected only "
        f"{shard_run.kept} ({shard_run.dropped} is on shard 2)")


def test_the_root_conftest_records_jax_timing_into_the_junit_report(shard_run):
    """``MADDENING_TEST_JAX_TIMING=1`` must attach the timing properties.

    Without them the budget's summary has no compile / tracing split, no
    cache hit rates, and cannot check a shard's warm label.  The kept test
    compiles a program in its fixture, so its trace and compile times are
    non-zero -- the listeners are really registered, not just the names.
    """
    assert shard_run.proc.returncode == 0, shard_run.proc.stdout[-3000:]
    (case,) = shard_run.cases
    props = {p.get("name"): float(p.get("value")) for p in case.iter("property")}
    assert set(props) == set(_jax_timing.PROPERTIES), sorted(props)
    assert props["jax_trace_s"] > 0 and props["jax_compile_s"] > 0, props
    assert props["subprocesses"] == 0, props


def test_a_shards_junit_report_records_how_fast_its_runner_was(shard_run):
    """Each shard's report carries its runner's collection time and CPU.

    ``scripts/slow_lane_weights.py`` scales each shard's times to a runner
    of median speed by the collection time; without it the next table is
    weighed by the luck of the runners its run drew (the same files take
    1.6 to 1.9 times as long on one hosted runner as on another).
    """
    assert shard_run.proc.returncode == 0, shard_run.proc.stdout[-3000:]
    (suite,) = shard_run.report.iter("testsuite")
    props = {p.get("name"): p.get("value") for p in suite.findall("properties/property")}
    assert set(props) == set(_sharding.RUNNER_PROPERTIES), props
    # A number of seconds (the root conftest has imported JAX before the
    # session starts, so two small files collect in milliseconds).
    assert 0 <= float(props[_sharding.COLLECT_SECONDS_PROPERTY]) < 300, props
    assert props["runner_cpu"] and int(props["runner_cpus"]) >= 1, props


def test_a_shards_junit_time_includes_fixture_setup(shard_run):
    """The budget judges setup + call + teardown, as it says.

    A module fixture's compile is charged to the first test that needs it,
    in its setup.  ``junit_duration_report = "call"`` would report the test
    body alone, and every such compile would vanish from the budget.
    """
    assert shard_run.proc.returncode == 0, shard_run.proc.stdout[-3000:]
    (case,) = shard_run.cases
    assert float(case.get("time")) >= SETUP_SECONDS, (
        f"the kept test's JUnit time is {case.get('time')} s, less than the "
        f"{SETUP_SECONDS} s its fixture's setup took")
