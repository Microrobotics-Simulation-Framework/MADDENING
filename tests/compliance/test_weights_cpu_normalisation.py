"""``scripts/slow_lane_weights.py`` normalises each shard's seconds by its CPU's speed.

GitHub's pool hands a job any of several CPU generations, about 1.5x apart
on this suite.  A table made from the median of several runs is only as good
as the runs' luck unless each shard's seconds are first divided by the speed
factor of the CPU it ran on, estimated from the files that ran on several
models.  These tests pin that on small, hand-made runs: the factor, the
reference model, the shards that cannot be scaled (no sidecar, too little
overlap), the cold-only filter of the per-push lane, and ``--latest``'s
download, with ``gh`` replaced.
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
           cache: str | None = None, lane: str = LANE) -> None:
    directory = root / run / f"{lane}-shard{shard}of3"
    directory.mkdir(parents=True)
    body = "".join(f'<testcase classname="x" name="t{i}" file="{f}" time="{s}"/>' for i, (f, s) in enumerate(cases))
    (directory / "test-results.xml").write_text(
        f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>', encoding="utf-8")
    if cpu is not None:
        (directory / f"runner-shard{shard}.json").write_text(json.dumps({"cpu_model": cpu}), encoding="utf-8")
    if cache is not None:
        (directory / f"cache-mode-shard{shard}.txt").write_text(cache + "\n", encoding="utf-8")


def _unit(cpu, files, lane=LANE, run="r"):
    return slw.Unit(lane=lane, artifact=f"{lane}-shard1of1", cpu=cpu, files=files, run=run)


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
    factors, _ref, _notes = slw.speed_factors(units, reference="9v45")
    assert factors[SLOW] == pytest.approx(6.0)        # median(30, 90) / 10 = 6, for every file


def test_the_reference_is_the_model_with_most_shards_unless_one_is_named():
    units = [_unit(SLOW, {"a": 30.0}) for _ in range(3)] + [_unit(FAST, {"a": 20.0})]
    assert slw.speed_factors(units)[1] == SLOW
    assert slw.speed_factors(units, reference="9v45")[1] == FAST     # a substring, any case
    with pytest.raises(SystemExit, match="matches none"):
        slw.speed_factors(units, reference="xeon")


def test_files_too_light_to_measure_do_not_set_a_factor():
    # Under the floor on the reference model the ratio is noise: ignored, so
    # three such files leave the model without a factor.
    units = [_unit(FAST, {"a": 1.0, "b": 2.0, "c": 3.0}), _unit(SLOW, {"a": 3.0, "b": 6.0, "c": 9.0})]
    factors, _ref, notes = slw.speed_factors(units, reference="9v45")
    assert factors[SLOW] == 1.0
    assert any("only 0 file(s)" in n for n in notes)


def test_a_model_with_too_little_overlap_stays_at_one_and_says_so():
    units = [_unit(FAST, {"a": 50.0, "b": 50.0}), _unit(FAST, {"a": 50.0, "b": 50.0}),
             _unit(SLOW, {"a": 100.0, "b": 100.0})]      # two shared files, three are needed
    factors, _ref, notes = slw.speed_factors(units)
    assert factors[SLOW] == 1.0
    assert any(SLOW in n and "not scaled" in n for n in notes), notes


def test_units_with_no_sidecar_are_not_scaled_and_do_not_count_as_a_model():
    units = [_unit(slw.UNKNOWN_CPU, {"a": 50.0}), _unit(FAST, {"a": 40.0})]
    factors, reference, notes = slw.speed_factors(units)
    assert factors == {FAST: 1.0} and reference == FAST
    assert slw.normalise(units, factors)[0].files == {"a": 50.0}
    assert any("nothing to normalise" in n for n in notes)


def test_a_run_on_a_slow_runner_no_longer_skews_the_median():
    """The point of it.  Three runs of one file: 100 s on FAST twice, and 150 s on SLOW.

    Unnormalised the median is 100 only by luck of the draw (it is 150 when
    two runs drew SLOW); normalised it is 100 either way.
    """
    def runs(models):
        return [_unit(m, {"a": 100.0 * (1.5 if m == SLOW else 1.0), "b": 60.0 * (1.5 if m == SLOW else 1.0),
                          "c": 40.0 * (1.5 if m == SLOW else 1.0), "d": 20.0 * (1.5 if m == SLOW else 1.0)}, run=str(i))
                for i, m in enumerate(models)]
    for models in ([FAST, FAST, SLOW], [FAST, SLOW, SLOW]):
        units = runs(models)
        raw = slw.merge([slw.lanes_of([u]) for u in units], "median")[LANE]
        factors, _ref, _n = slw.speed_factors(units, reference="9v45")
        fixed = slw.merge([slw.lanes_of([u]) for u in slw.normalise(units, factors)], "median")[LANE]
        assert fixed == {"a": pytest.approx(100.0), "b": pytest.approx(60.0),
                         "c": pytest.approx(40.0), "d": pytest.approx(20.0)}
        if models.count(SLOW) == 2:
            assert raw["a"] == 150.0      # without it, the unlucky draw sets the weight


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


def _three_runs(tmp_path, models):
    a, b, c, d = _tree_files(4)
    for i, model in enumerate(models):
        k = 1.5 if model == SLOW else 1.0
        _shard(tmp_path, f"r{i}", 1, [(a, 100 * k), (b, 60 * k), (c, 40 * k), (d, 30 * k)], model)
    return (a, b, c, d), [x for i in range(len(models)) for x in ("--run", f"{i}={tmp_path / f'r{i}'}")]


def test_the_command_writes_the_table_on_the_reference_model_and_says_how(tmp_path):
    files, runs = _three_runs(tmp_path, [FAST, SLOW, SLOW])
    # The default reference is the model with the most shards: SLOW, twice.
    table, stdout = _run(runs, tmp_path)
    assert table["seconds"] == {files[0]: 150, files[1]: 90, files[2]: 60, files[3]: 45}
    assert f"relative to {SLOW}" in stdout and "speed factors" in stdout
    assert "divided by its CPU model's speed factor" in table["measured"]
    # --reference names the FAST model: the same runs, on its time.
    table, _ = _run([*runs, "--reference", "9v45"], tmp_path)
    assert table["seconds"] == {files[0]: 100, files[1]: 60, files[2]: 40, files[3]: 30}
    # --no-normalise: the median of the raw seconds, which the two slow draws set.
    raw, _ = _run([*runs, "--no-normalise"], tmp_path)
    assert raw["seconds"][files[0]] == 150 and "speed factor" not in raw["measured"]
    # The same runs with two fast draws instead of two slow ones: the normalised
    # table on the FAST model does not move, the raw one does.
    other = tmp_path / "other"
    other.mkdir()
    files2, runs2 = _three_runs(other, [FAST, FAST, SLOW])
    table, _ = _run([*runs2, "--reference", "9v45"], other)
    assert table["seconds"] == {files2[0]: 100, files2[1]: 60, files2[2]: 40, files2[3]: 30}


def test_the_reference_flag_picks_the_models_the_seconds_are_on(tmp_path):
    files, runs = _three_runs(tmp_path, [FAST, SLOW, SLOW])
    table, _ = _run([*runs, "--reference", "7763"], tmp_path)
    assert table["seconds"][files[0]] == 150       # SLOW is the reference: the table is its time
    table, _ = _run([*runs, "--reference", "9v45"], tmp_path)
    assert table["seconds"][files[0]] == 100


def test_a_report_with_no_sidecar_is_used_as_it_is_and_the_output_says_so(tmp_path):
    a, b, c = _tree_files(3)
    _shard(tmp_path, "r0", 1, [(a, 80.0), (b, 60.0), (c, 40.0)], None)
    _shard(tmp_path, "r1", 1, [(a, 100.0), (b, 60.0), (c, 40.0)], None)
    table, stdout = _run(["--run", f"0={tmp_path / 'r0'}", "--run", f"1={tmp_path / 'r1'}"], tmp_path)
    assert table["seconds"][a] == 90              # the median of two: their mean
    assert "no runner sidecar" in stdout and "speed factor" not in table["measured"]


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
