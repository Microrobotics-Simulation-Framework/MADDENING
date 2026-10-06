"""The CI test shards must be a stable partition of the suite.

Shard ``i`` of a pull request has to hold the same test files as shard
``i`` of the base branch, because the XLA compilation cache a shard
restores was saved by that shard on the base branch.  These tests pin the
properties that make that true: the assignment is a function of the file
path alone, every test lands on exactly one shard, the pins are real and
in range, and CI runs the job count the pins were balanced for -- every
shard of it (no matrix leg excluded or added).

The slow lane restores no cache and splits by measured time instead
(``i/N:weighted``, from ``tests/slow_lane_weights.json``).  For it the tests
pin that the split is still a partition by file, that the table names real
files and is the one the generator writes, that the deal is balanced and
under the lane's time target, that the per-push split does not move, and
that the workflow asks for it on every shard of its own count.

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
    "ci.yml": (_sharding.PINS_FOR, ""),
    "slow-tests.yml": (_sharding.SLOW_LANE_SHARDS, ":" + _sharding.WEIGHTED),
}


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


@pytest.mark.parametrize("spec", ["0/4", "5/4", "a/b", "4", "1/4/2", "", " 1/4", "1/ 4", "+1/4", "1/1_0",
                                  "0/6:weighted", "7/6:weighted", "1/6:", "1/6:weigthed",
                                  "1/6:weighted:weighted", ":weighted", "1/6 :weighted"])
def test_a_malformed_shard_spec_is_refused(spec):
    # A misspelt suffix must not fall back to the hash split: the slow lane
    # would run, every test once, with one shard back at its timeout.
    with pytest.raises(pytest.UsageError):
        _sharding.parse(spec)


def test_a_shard_spec_says_which_split_it_asks_for():
    assert _sharding.parse("2/4") == (2, 4, False)
    assert _sharding.parse("6/6:weighted") == (6, 6, True)


def _tree_test_files() -> list[str]:
    """Every file pytest collects tests from under ``tests/`` (its default patterns)."""
    files = {p.relative_to(REPO_ROOT).as_posix()
             for pattern in ("test_*.py", "*_test.py") for p in (REPO_ROOT / "tests").rglob(pattern)}
    assert len(files) > 500, len(files)
    return sorted(files)


@pytest.mark.parametrize("n, weighted", [
    (_sharding.PINS_FOR, False), (_sharding.SLOW_LANE_SHARDS, True),
    # The weighted split at other counts, the per-push lane's included: the
    # table is dealt for whatever count the spec names.
    (_sharding.PINS_FOR, True), (1, True), (8, True)])
def test_the_shards_partition_the_tests_and_keep_files_together(n, weighted):
    weights = _sharding.load_weights() if weighted else None
    nodeids = [f"tests/{d}/test_{f}.py::test_{t}" for d in "abc" for f in range(7) for t in range(3)]
    nodeids += list(_sharding.PINS)  # pinned files too
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


#: --- the slow lane's weighted split -------------------------------------------

#: Minutes of measured test time the heaviest slow-lane shard may hold.  The
#: wrapper kills a shard at 175; the lane is rebalanced (more shards, or a
#: file split) long before that, when a refreshed table predicts over this.
SLOW_SHARD_TARGET_MIN = 100


def _table() -> dict:
    return json.loads(_sharding.WEIGHTS_FILE.read_text(encoding="utf-8"))


def test_every_weighted_file_is_a_real_test_file():
    """An entry for a deleted or renamed file is dealt a slot nothing runs in.

    Every test still runs once, but the shard that "holds" the ghost is
    lighter than the table says and the others heavier; a heavy file's
    rename would quietly undo the balance.
    """
    weights = _sharding.load_weights()
    real = set(_tree_test_files())
    ghosts = sorted(set(weights) - real)
    assert not ghosts, (
        f"tests/slow_lane_weights.json lists files that are not test files in the tree: {ghosts}; "
        "regenerate it (scripts/slow_lane_weights.py) or remove the entries")


def test_the_weights_table_says_where_its_numbers_came_from():
    table = _table()
    assert table["runs"], "no run recorded"
    for run in table["runs"]:
        assert re.fullmatch(r"[0-9]+", run["run"]), run
        assert re.fullmatch(r"[0-9a-f]{7,40}", run["commit"]), run
        assert run["artifacts"] and all(a.startswith("slow-durations-") for a in run["artifacts"]), run
    floor = table["min_seconds"]
    assert type(floor) is int and floor >= 1
    light = {p: s for p, s in table["seconds"].items() if s < floor}
    assert not light, f"entries under the table's own {floor} s floor: {light}"
    assert list(table["seconds"]) == sorted(table["seconds"]), "the table is written sorted by path"


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


def test_a_file_the_table_does_not_list_goes_by_the_hash_of_its_path():
    n = _sharding.SLOW_LANE_SHARDS
    dealt = _sharding.deal(_sharding.load_weights(), n)
    for path in ("tests/core/test_not_measured_yet.py", "tests/new/test_added_since.py"):
        by_hash = int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % n + 1
        assert path not in dealt
        assert _sharding.weighted_shard_of(path, n, dealt) == by_hash
    for path, shard in dealt.items():
        assert _sharding.weighted_shard_of(path, n, dealt) == shard


def test_the_weighted_split_ignores_the_per_push_pins():
    # The pins balance the per-push lane's times.  At a weighted count equal
    # to PINS_FOR an unlisted pinned file must still go by hash, or the two
    # rules would both claim it.
    n = _sharding.PINS_FOR
    for path in _sharding.PINS:
        by_hash = int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % n + 1
        assert _sharding.weighted_shard_of(path, n, {}) == by_hash
        assert _sharding.Plugin(1, n, {"tests/test_other.py": 5}).shard_of(path) == by_hash


def test_the_per_push_split_does_not_read_the_weights():
    """The per-push lane's shards hold the files their saved caches were built from.

    A plain ``i/N`` spec must give ``shard_of`` for every real file -- the
    table's files included -- whatever the table says.
    """
    n = _sharding.PINS_FOR
    plain = _sharding.Plugin(1, n)
    assert not plain.weighted
    moved = [p for p in _tree_test_files() if plain.shard_of(p) != _sharding.shard_of(p, n)]
    assert not moved, moved
    weighted = _sharding.Plugin(1, n, _sharding.load_weights())
    assert weighted.weighted
    assert any(weighted.shard_of(p) != _sharding.shard_of(p, n) for p in _tree_test_files()), (
        "the weighted split at four shards equals the hash split: the table is not being used")


def test_the_slow_lanes_weighted_shards_are_balanced_and_under_the_target():
    """The table's own prediction for the count the workflow runs.

    Longest-first dealing leaves any two shards within the heaviest file of
    each other; and the heaviest shard has to be under the lane's target,
    or the table was refreshed into a lane that needs more shards.
    """
    n = _sharding.SLOW_LANE_SHARDS
    weights = _sharding.load_weights()
    dealt = _sharding.deal(weights, n)
    assert set(dealt) == set(weights) and set(dealt.values()) == set(range(1, n + 1))
    load = [0] * n
    for path, shard in dealt.items():
        load[shard - 1] += weights[path]
    assert max(load) - min(load) <= max(weights.values()), load
    assert max(load) / 60 < SLOW_SHARD_TARGET_MIN, (
        f"the weighted split predicts {max(load) / 60:.0f} min on the heaviest of {n} slow-lane shards "
        f"(target: under {SLOW_SHARD_TARGET_MIN}); raise SLOW_LANE_SHARDS or split the heaviest file")
    # One file alone must leave room on its shard, or no count can balance it.
    assert max(weights.values()) / 60 < SLOW_SHARD_TARGET_MIN


def test_a_weighted_spec_registers_the_weighted_split_from_the_shipped_table():
    registered = []
    config = SimpleNamespace(pluginmanager=SimpleNamespace(
        register=lambda plugin, name: registered.append((plugin, name))))
    n = _sharding.SLOW_LANE_SHARDS
    _sharding.register(config, f"2/{n}:weighted")
    _sharding.register(config, f"2/{n}")
    (weighted, name), (plain, _) = registered
    assert name == "maddening-test-shard"
    assert (weighted.shard, weighted.of, weighted.weighted) == (2, n, True)
    assert (plain.shard, plain.of, plain.weighted) == (2, n, False)
    heaviest = max(_sharding.load_weights().items(), key=lambda kv: (kv[1], kv[0]))[0]
    # Longest first onto the lightest shard: the heaviest file opens shard 1.
    assert weighted.shard_of(heaviest) == 1
    assert weighted.pytest_report_header(config) == (
        f"test shard: 2/{n} (by file, weighted; see tests/_sharding.py)")
    assert plain.pytest_report_header(config) == f"test shard: 2/{n} (by file; see tests/_sharding.py)"


def test_the_weights_generator_writes_a_table_the_split_reads(tmp_path):
    """``scripts/slow_lane_weights.py`` on two small runs: slower lane, later run, floor, ghosts."""
    def report(run, artifact, cases):
        directory = tmp_path / run / artifact
        directory.mkdir(parents=True)
        body = "".join(f'<testcase classname="x" name="t{i}" file="{f}" time="{s}"/>'
                       for i, (f, s) in enumerate(cases))
        (directory / "test-results.xml").write_text(
            f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>', encoding="utf-8")

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
         "--write", "--output", str(out)],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    table = json.loads(out.read_text(encoding="utf-8"))
    # a: its slower lane (61 s, two tests summed).  b: under the floor.  c: the
    # later run's 20 s on one lane, the first run's 90 s on the other.
    assert table["seconds"] == {real_a: 61, real_c: 90}
    assert [(r["run"], r["commit"]) for r in table["runs"]] == [("11", "abc1234"), ("22", "def5678")]
    assert table["runs"][1]["artifacts"] == ["slow-durations-py3.12-jax0.10.2-shard2of2"]
    assert "no longer in the tree, left out: tests/test_gone.py" in proc.stdout
    assert _sharding.load_weights(out) == table["seconds"]
    # The prediction, per lane: c (90) opens shard 1, a (61) shard 2; b is
    # unlisted and goes by hash.
    b_shard = _sharding.weighted_shard_of(real_b, 2, {})
    lane = [90.0, 40.0]
    lane[b_shard - 1] += 3.0
    assert f"{'weighted, 2 shards':<34}{lane[0] / 60:6.1f} {lane[1] / 60:6.1f}" in proc.stdout, proc.stdout


def test_the_shipped_weights_table_is_what_the_generator_writes():
    # "do not edit by hand": keys in the generator's order and nothing else.
    assert list(_table()) == ["what", "measured", "runs", "min_seconds", "seconds"]


#: Every way a workflow writes a lane's shard count besides the matrix: the
#: shard spec, artifact and cache names (``of4``), step titles (``of 4``,
#: ``of 4 shard``) and the slow lane's issue titles (``shard ${shard}/4``).
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
    # `exclude: [{shard: 4}]` would leave the shard axis intact -- so the
    # count check above passes -- while a quarter of the suite runs nowhere.
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
    """``MADDENING_TEST_SHARD=1/4 python -m pytest ...`` points all four shards at one.

    Three quarters of the suite then runs nowhere, and the checks above still
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
    cases = list(ET.parse(root / "report.xml").getroot().iter("testcase")) \
        if (root / "report.xml").is_file() else []
    return SimpleNamespace(proc=proc, cases=cases, kept=kept, dropped=dropped)


def test_the_root_conftest_keeps_only_this_shards_files(shard_run):
    """``MADDENING_TEST_SHARD`` must reach ``tests/_sharding.py`` through the conftest.

    If the conftest stopped registering the plugin, every CI shard would
    run the whole suite: four times the CI time, no test dropped, and every
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
