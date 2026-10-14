"""``scripts/slow_lane_weights.py``: the CPU-model fallback of the runner-speed scaling.

GitHub's pool hands a job any of several runners, 1.6 to 1.9 times apart
on this suite.  The weights script scales every shard's report by one
measure of its runner's speed, the time pytest took to collect the suite
(the report's ``runner_collect_seconds``; ``tests/compliance/test_ci_sharding.py``
pins that scaling).  The CPU model, the report's ``runner_cpu``, is recorded
metadata, and is used only for a report that has no collection time: such a
report is divided by its model's speed factor, estimated from the files it
shares with the scaled reports.  These tests pin that on small, hand-made
runs: a report is never corrected twice, the factor and its reference, the
reports that cannot be scaled (no runner fields, too little overlap), the
cold-only filter of the per-push lane, and ``--latest``'s download, with
``gh`` replaced.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests import _sharding  # noqa: E402

_spec = importlib.util.spec_from_file_location("slow_lane_weights", REPO_ROOT / "scripts" / "slow_lane_weights.py")
slw = importlib.util.module_from_spec(_spec)
sys.modules["slow_lane_weights"] = slw
_spec.loader.exec_module(slw)

FAST = "AMD EPYC 9V45 96-Core Processor"
MID = "AMD EPYC 9V74 80-Core Processor"
SLOW = "AMD EPYC 7763 64-Core Processor"
LANE = "slow-durations-py3.12-jax0.10.2"


def _tree_files(n: int) -> list[str]:
    return sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "tests" / "core").glob("test_*.py"))[:n]


def _shard(root: Path, run: str, shard: int, cases: list[tuple[str, float]], cpu: str | None,
           cache: str | None = None, lane: str = LANE, collect: float | None = None) -> None:
    """One downloaded shard artifact: its JUnit report, with the runner's fields it is given."""
    directory = root / run / f"{lane}-shard{shard}of3"
    directory.mkdir(parents=True)
    props = "".join(f'<property name="{name}" value="{value}"/>' for name, value in (
        (_sharding.COLLECT_SECONDS_PROPERTY, collect), (_sharding.CPU_PROPERTY, cpu)) if value is not None)
    body = "".join(f'<testcase classname="x" name="t{i}" file="{f}" time="{s}"/>' for i, (f, s) in enumerate(cases))
    (directory / "test-results.xml").write_text(
        f'<testsuites><testsuite name="pytest"><properties>{props}</properties>{body}</testsuite></testsuites>',
        encoding="utf-8")
    if cache is not None:
        (directory / f"cache-mode-shard{shard}.txt").write_text(cache + "\n", encoding="utf-8")


def _unit(cpu, files, lane=LANE, run="r", collect=None):
    return slw.Unit(lane=lane, artifact=f"{lane}-shard1of1", cpu=cpu, files=files, run=run, collect=collect)


# ---------------------------------------------------------------------------
# one rule per report
# ---------------------------------------------------------------------------

def test_a_report_with_a_collection_time_is_scaled_by_it_alone():
    """Never corrected twice: a timed report's CPU model changes nothing.

    Two reports on SLOW collected in 60 s, one on FAST in 30 s; the median
    is 60.  Each is scaled by (60 / its own) ** 1.5 and by nothing else,
    whatever its CPU.
    """
    files = {"a": 100.0, "b": 60.0, "c": 40.0}
    units = [_unit(SLOW, files, collect=60.0), _unit(SLOW, files, collect=60.0),
             _unit(FAST, {p: s / 1.5 for p, s in files.items()}, collect=30.0)]
    scales, record, notes = slw.runner_scales(units)
    assert scales == [1.0, 1.0, pytest.approx(2 ** 1.5)]
    assert record["reports_scaled"] == 3 and record["reports_by_cpu_model"] == 0, record
    assert record["cpu_factors"] == {} and record["cpu_reference"] is None and notes == []


def test_a_report_without_a_collection_time_falls_back_to_its_cpu_models_factor():
    """Against the scaled reports, so the table stays on a runner of median speed.

    Three timed reports put a, b, c, d at 100, 60, 40, 30 s on the median
    runner.  An untimed SLOW report took 1.5 times that: its factor is 1.5
    against the scaled reports, and it is divided by it.  An untimed report
    with no CPU either is taken as measured.
    """
    files = {"a": 100.0, "b": 60.0, "c": 40.0, "d": 30.0}
    units = [_unit(FAST, files, collect=40.0) for _ in range(3)]
    units.append(_unit(SLOW, {p: s * 1.5 for p, s in files.items()}))
    units.append(_unit(slw.UNKNOWN_CPU, {p: s * 3 for p, s in files.items()}))
    scales, record, _notes = slw.runner_scales(units)
    assert scales == [1.0, 1.0, 1.0, pytest.approx(1 / 1.5), 1.0]
    assert record == {"property": _sharding.COLLECT_SECONDS_PROPERTY, "reference_seconds": 40.0,
                      "exponent": _sharding.SPEED_EXPONENT, "reports_scaled": 3, "reports_by_cpu_model": 1,
                      "cpu_reference": slw.MEDIAN_RUNNER, "cpu_factors": {SLOW: 1.5}, "reports_as_measured": 1}
    fixed = slw.normalise(units, scales)
    assert fixed[3].files == {p: pytest.approx(s) for p, s in files.items()}


def test_with_no_collection_time_anywhere_the_fallback_is_against_the_commonest_model():
    files = {"a": 100.0, "b": 60.0, "c": 40.0}
    units = [_unit(SLOW, {p: s * 1.5 for p, s in files.items()}) for _ in range(2)] + [_unit(FAST, files)]
    scales, record, _notes = slw.runner_scales(units)
    assert record["cpu_reference"] == SLOW and record["reference_seconds"] is None and record["exponent"] == 0
    assert scales == [1.0, 1.0, pytest.approx(1.5)]       # FAST brought up to SLOW's time
    assert record["reports_scaled"] == 0 and record["reports_by_cpu_model"] == 3


# ---------------------------------------------------------------------------
# the factor
# ---------------------------------------------------------------------------

def test_a_models_factor_is_the_median_of_its_files_ratios_to_the_reference():
    # Three files ran on both models: 1.5x, 1.5x and 3.0x slower on SLOW.  The
    # median ratio is 1.5, not the 3.0 of the worst file or the mean 2.0.
    units = [
        _unit(FAST, {"a": 100.0, "b": 50.0, "c": 20.0}),
        _unit(FAST, {"a": 100.0, "b": 50.0, "c": 20.0}),
        _unit(SLOW, {"a": 150.0, "b": 75.0, "c": 60.0}),
    ]
    factors, reference, notes = slw.speed_factors(units)
    assert reference == FAST and factors == {FAST: 1.0, SLOW: pytest.approx(1.5)}
    assert notes == []


def test_the_ratio_is_of_medians_when_a_model_ran_a_file_several_times():
    units = [_unit(FAST, {f: 10.0 * (i + 1)}) for i, f in enumerate("abc")] * 2   # 2 shards each
    units += [_unit(SLOW, {f: 30.0 * (i + 1)}) for i, f in enumerate("abc")]
    units += [_unit(SLOW, {f: 90.0 * (i + 1)}) for i, f in enumerate("abc")]      # one run drew a bad day
    factors, _ref, _notes = slw.speed_factors(units, reference=FAST)
    assert factors[SLOW] == pytest.approx(6.0)        # median(30, 90) / 10 = 6, for every file


def test_the_reference_is_the_model_with_most_shards_unless_one_is_named():
    units = [_unit(SLOW, {"a": 30.0}) for _ in range(3)] + [_unit(FAST, {"a": 20.0})]
    assert slw.speed_factors(units)[1] == SLOW
    assert slw.speed_factors(units, reference=FAST)[1] == FAST
    assert slw.speed_factors(units, reference="no such model")[1] == SLOW


def test_files_too_light_to_measure_do_not_set_a_factor():
    # Under the floor on the reference model the ratio is noise: ignored, so
    # three such files leave the model without a factor.
    units = [_unit(FAST, {"a": 1.0, "b": 2.0, "c": 3.0}), _unit(SLOW, {"a": 3.0, "b": 6.0, "c": 9.0})]
    factors, _ref, notes = slw.speed_factors(units, reference=FAST)
    assert factors[SLOW] == 1.0
    assert any("only 0 file(s)" in n for n in notes)


def test_a_model_with_too_little_overlap_stays_at_one_and_says_so():
    units = [_unit(FAST, {"a": 50.0, "b": 50.0}), _unit(FAST, {"a": 50.0, "b": 50.0}),
             _unit(SLOW, {"a": 100.0, "b": 100.0})]      # two shared files, three are needed
    factors, _ref, notes = slw.speed_factors(units)
    assert factors[SLOW] == 1.0
    assert any(SLOW in n and "not scaled" in n for n in notes), notes


def test_units_with_no_runner_fields_are_not_scaled_and_do_not_count_as_a_model():
    units = [_unit(slw.UNKNOWN_CPU, {"a": 50.0}), _unit(FAST, {"a": 40.0})]
    factors, reference, notes = slw.speed_factors(units)
    assert factors == {FAST: 1.0} and reference == FAST
    assert slw.runner_scales(units)[0][0] == 1.0
    assert any("nothing to normalise" in n for n in notes)


def test_a_run_on_a_slow_runner_no_longer_skews_the_median():
    """The point of it.  Three runs of one file: 100 s on FAST twice, and 150 s on SLOW.

    Unnormalised the median is 100 only by luck of the draw (it is 150 when
    two runs drew SLOW).  Scaled by each report's collection time (FAST
    collects in 30 s, SLOW in 30 * 1.5 ** (1 / 1.5)), every report agrees
    with the others, so no draw can set one file's weight against the rest;
    the table is on the median runner of the reports given, which its
    ``runner_speed`` records for the drift check (with three reports that
    median is itself a draw; a real table has sixty or more).
    """
    slow_collect = 30.0 * 1.5 ** (1 / _sharding.SPEED_EXPONENT)

    def runs(models):
        return [_unit(m, {f: s * (1.5 if m == SLOW else 1.0) for f, s in
                          {"a": 100.0, "b": 60.0, "c": 40.0, "d": 20.0}.items()},
                      run=str(i), collect=slow_collect if m == SLOW else 30.0)
                for i, m in enumerate(models)]
    tables = []
    for models in ([FAST, FAST, SLOW], [FAST, SLOW, SLOW]):
        units = runs(models)
        raw = slw.merge([slw.lanes_of([u]) for u in units], "median")[LANE]
        scales, _record, _n = slw.runner_scales(units)
        tables.append(slw.merge([slw.lanes_of([u]) for u in slw.normalise(units, scales)], "median")[LANE])
        if models.count(SLOW) == 2:
            assert raw["a"] == 150.0      # without it, the unlucky draw sets the weight
    # The reference is the median runner of each set, so the two tables differ
    # by the ratio of their medians and no more: the shape is the same.
    assert tables[0]["a"] / tables[0]["d"] == pytest.approx(5.0) == tables[1]["a"] / tables[1]["d"]
    assert tables[0] == {f: pytest.approx(s) for f, s in {"a": 100.0, "b": 60.0, "c": 40.0, "d": 20.0}.items()}
    assert tables[1] == {f: pytest.approx(s * 1.5) for f, s in {"a": 100.0, "b": 60.0, "c": 40.0, "d": 20.0}.items()}


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def _run(args, tmp_path):
    out = tmp_path / "weights.json"
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "slow_lane_weights.py"),
                           *args, "--shards", "1", "--write", "--output", str(out)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(out.read_text(encoding="utf-8")), proc.stdout


def _three_runs(tmp_path, models, timed=()):
    """Three one-shard runs; a run whose index is in ``timed`` carries a collection time of 40 s."""
    a, b, c, d = _tree_files(4)
    for i, model in enumerate(models):
        k = 1.5 if model == SLOW else 1.0
        _shard(tmp_path, f"r{i}", 1, [(a, 100 * k), (b, 60 * k), (c, 40 * k), (d, 30 * k)], model,
               collect=40.0 if i in timed else None)
    return (a, b, c, d), [x for i in range(len(models)) for x in ("--run", f"{i}={tmp_path / f'r{i}'}")]


def test_the_command_falls_back_to_cpu_factors_only_where_there_is_no_collection_time(tmp_path):
    # Run 0 (FAST) carries a collection time; runs 1 and 2 (SLOW) do not, and
    # are brought to the scaled run's time by SLOW's factor, 1.5.
    files, runs = _three_runs(tmp_path, [FAST, SLOW, SLOW], timed=(0,))
    table, stdout = _run(runs, tmp_path)
    assert table["seconds"] == {files[0]: 100, files[1]: 60, files[2]: 40, files[3]: 30}
    assert "1 reports scaled to a runner that collects in 40.0 s" in stdout, stdout
    assert f"2 reports without {_sharding.COLLECT_SECONDS_PROPERTY} divided by their CPU model's factor " \
           f"against the scaled reports: {SLOW} 1.50" in stdout, stdout
    assert "2 report(s) without a collection time are divided by their CPU model's speed factor" in table["measured"]
    assert table["runner_speed"]["cpu_factors"] == {SLOW: 1.5}, table["runner_speed"]
    # --no-normalise: the median of the raw seconds, which the two slow draws set.
    raw, stdout = _run([*runs, "--no-normalise"], tmp_path)
    assert raw["seconds"][files[0]] == 150 and "speed factor" not in raw["measured"]
    assert raw["runner_speed"]["reports_as_measured"] == 3


def test_with_no_collection_time_the_table_is_on_the_commonest_model(tmp_path):
    files, runs = _three_runs(tmp_path, [FAST, SLOW, SLOW])
    table, stdout = _run(runs, tmp_path)
    assert table["seconds"] == {files[0]: 150, files[1]: 90, files[2]: 60, files[3]: 45}
    assert f"against {SLOW}: " in stdout and f"{FAST} 0.67" in stdout, stdout
    assert table["runner_speed"]["reference_seconds"] is None and table["runner_speed"]["cpu_reference"] == SLOW


def test_a_report_with_no_runner_fields_is_used_as_it_is_and_the_output_says_so(tmp_path):
    a, b, c = _tree_files(3)
    _shard(tmp_path, "r0", 1, [(a, 80.0), (b, 60.0), (c, 40.0)], None)
    _shard(tmp_path, "r1", 1, [(a, 100.0), (b, 60.0), (c, 40.0)], None)
    table, stdout = _run(["--run", f"0={tmp_path / 'r0'}", "--run", f"1={tmp_path / 'r1'}"], tmp_path)
    assert table["seconds"][a] == 90              # the median of two: their mean
    assert "2 reports taken as measured" in stdout and "speed factor" not in table["measured"]


def test_the_per_push_lane_leaves_out_a_warm_shard(tmp_path):
    a, b, c = _tree_files(3)
    lane = "test-durations-py3.12-jax0.10.2"
    _shard(tmp_path, "r0", 1, [(a, 100.0), (b, 50.0), (c, 40.0)], FAST, cache="cold", lane=lane)
    _shard(tmp_path, "r1", 1, [(a, 10.0), (b, 5.0), (c, 4.0)], FAST, cache="warm", lane=lane)
    table, stdout = _run(["--lane", "per-push", "--run", f"0={tmp_path / 'r0'}", "--run", f"1={tmp_path / 'r1'}"],
                         tmp_path)
    assert table["seconds"][a] == 100 and "warm compilation cache, left out" in stdout


def test_the_default_aggregate_is_the_median(tmp_path):
    a, b, c = _tree_files(3)
    for i, t in enumerate((60.0, 600.0, 70.0)):
        _shard(tmp_path, f"r{i}", 1, [(a, t), (b, 50.0), (c, 40.0)], None)
    runs = [x for i in range(3) for x in ("--run", f"{i}={tmp_path / f'r{i}'}")]
    table, _ = _run(runs, tmp_path)
    assert table["seconds"][a] == 70 and "median" in table["measured"]


def test_the_speed_exponent_must_be_positive(tmp_path):
    files, runs = _three_runs(tmp_path, [FAST, FAST, FAST], timed=(0, 1, 2))
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "slow_lane_weights.py"), *runs,
                           "--speed-exponent", "0"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2 and "--no-normalise" in proc.stderr


# ---------------------------------------------------------------------------
# --latest, with gh replaced
# ---------------------------------------------------------------------------

def _fake_gh(tmp_path, listing, without=()):
    calls = []

    def run(cmd, capture_output=True, text=True, check=False):
        calls.append(cmd)
        if cmd[:3] == ["gh", "run", "list"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps(listing), stderr="")
        run_id = cmd[3]
        if run_id in without:
            return SimpleNamespace(returncode=1, stdout="", stderr="no artifact")
        (a,) = _tree_files(1)
        _shard(Path(cmd[cmd.index("--dir") + 1]).parent, run_id, 1, [(a, 50.0)], FAST)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return run, calls


def test_latest_takes_the_newest_runs_that_left_artifacts(tmp_path):
    listing = [{"databaseId": i, "headSha": f"sha{i}" * 3, "event": "schedule", "conclusion": c}
               for i, c in ((9, "cancelled"), (8, "failure"), (7, "success"), (6, "success"), (5, "success"))]
    run, calls = _fake_gh(tmp_path, listing, without=("7",))
    found = slw.fetch_latest("slow", 2, tmp_path / "dl", run=run)
    # 9 was cancelled (no artifacts to trust), 7 had expired; 8 and 6 are the two newest left.
    assert [r for r, _sha, _d in found] == ["8", "6"]
    assert calls[0][:6] == ["gh", "run", "list", "--workflow", "slow-tests.yml", "--status"]
    assert any(c[3] == "7" for c in calls[1:])                      # tried, and passed over


def test_latest_for_the_per_push_lane_reads_pushes_only(tmp_path):
    listing = [{"databaseId": 3, "headSha": "a" * 40, "event": "pull_request", "conclusion": "success"},
               {"databaseId": 2, "headSha": "b" * 40, "event": "push", "conclusion": "success"}]
    run, calls = _fake_gh(tmp_path, listing)
    found = slw.fetch_latest("per-push", 3, tmp_path / "dl", run=run)
    assert [r for r, _s, _d in found] == ["2"] and found[0][1] == "b" * 12
    assert calls[0][4] == "ci.yml" and calls[1][-1] == "test-durations-*"


def test_a_run_whose_shards_were_all_warm_is_refused_not_written_as_an_empty_table(tmp_path):
    a, b = _tree_files(2)
    lane = "test-durations-py3.12-jax0.10.2"
    _shard(tmp_path, "r0", 1, [(a, 10.0), (b, 5.0)], FAST, cache="warm", lane=lane)
    out = tmp_path / "weights.json"
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "slow_lane_weights.py"), "--lane", "per-push",
                           "--run", f"0={tmp_path / 'r0'}", "--write", "--output", str(out)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0 and "warm compilation cache" in proc.stderr
    assert not out.exists()
