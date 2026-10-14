"""The slow lane's limits, its issues and its drift warning.

Three things ``slow-tests.yml`` got wrong on 2026-10-11 to 13, pinned here
so that none returns without a test failing:

* **The inner time limit cannot fire before the job's.**  ``timeout 175m``
  started with the pytest step and the job's ``timeout-minutes: 180`` with
  the job; on a degraded runner that took 11 minutes to set up, the job's
  limit fired first, the classifier and issue steps never ran and the result
  read "The operation was canceled".  pytest's limit is now the job's,
  less what the clock step saw used, less an allowance for the setup before
  that step, less a margin for the steps after pytest.
* **A run on a throwaway branch opens no public issue.**
* **A shard that outruns its weights says so**, naming the files the table
  has not measured (``scripts/slow_lane_drift.py``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.compliance import test_ci_workflows as wf

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "slow_lane_drift.py"


def _job() -> dict:
    return wf._workflow("slow-tests.yml")["jobs"]["slow"]


def _pytest_step() -> dict:
    return wf._step(_job(), wf.SLOW_PYTEST)


# ---------------------------------------------------------------------------
# The time limit
# ---------------------------------------------------------------------------

def test_the_clock_is_read_in_the_first_step():
    first = _job()["steps"][0]
    assert first.get("id") == "clock" and "date +%s" in first["run"], first
    assert "job_start=" in first["run"] and "$GITHUB_OUTPUT" in first["run"]


def test_the_inner_limit_plus_a_setup_allowance_stays_under_the_jobs():
    """Whatever pytest may be given, the steps after it still fit inside the job's limit."""
    job, env = _job(), _pytest_step()["env"]
    minutes = {k: int(env[k]) for k in ("JOB_TIMEOUT_MIN", "PYTEST_MAX_MIN", "UNSEEN_SETUP_MIN",
                                        "CLASSIFY_MARGIN_MIN")}
    assert minutes["JOB_TIMEOUT_MIN"] == job["timeout-minutes"], (
        "the job's own limit and the number pytest's limit is worked out from must be the same")
    assert env["JOB_START"] == "${{ steps.clock.outputs.job_start }}"
    assert (minutes["PYTEST_MAX_MIN"] + minutes["UNSEEN_SETUP_MIN"] + minutes["CLASSIFY_MARGIN_MIN"]
            < minutes["JOB_TIMEOUT_MIN"]), minutes
    # The setup before the first step took 11 minutes on 2026-10-13: the allowance covers it, and
    # the steps after pytest (classifier, issue, re-raise) have their five minutes.
    assert minutes["UNSEEN_SETUP_MIN"] >= 11 and minutes["CLASSIFY_MARGIN_MIN"] >= 5, minutes
    # The command takes the computed limit, not a literal.
    (command,) = [ln.strip() for ln in re.sub(r"\\\n", " ", _pytest_step()["run"]).splitlines()
                  if "-m pytest" in ln and not ln.strip().startswith("#")]
    assert 'timeout --kill-after=60s "${limit}s" python -m pytest' in command, command
    assert not re.search(r"\b\d+m\b", command.split("python -m pytest")[0]), command


def _run_pytest_step(tmp_path: Path, *, used_sec: int | None, extra_env: dict | None = None) -> list[str]:
    """Run the pytest step under bash with a ``timeout`` that records its arguments; returns them."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    args_file = tmp_path / "timeout-args"
    (bin_dir / "timeout").write_text(f'#!/bin/sh\necho "$@" > {args_file}\nexit 0\n')
    (bin_dir / "timeout").chmod(0o755)
    import time
    job_start = "" if used_sec is None else str(int(time.time()) - used_sec)
    proc, outputs = wf._run_step(_pytest_step(), {"matrix.shard": "1", "steps.clock.outputs.job_start": job_start},
                                 tmp_path, extra_env=extra_env, path=f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    given = args_file.read_text().split()
    assert outputs["limit_sec"] == given[1].removesuffix("s"), (outputs, given)
    return given


@pytest.mark.parametrize("used_min", [0, 3, 5, 11, 60, 170])
def test_pytest_is_given_what_the_job_has_left(tmp_path, used_min):
    if shutil.which("bash") is None:
        pytest.fail("bash is needed to run the step; CI runners have it")
    used = used_min * 60 + 20                  # the clock step's reading, 20 s before this step started
    given = _run_pytest_step(tmp_path, used_sec=used)
    assert given[0] == "--kill-after=60s" and given[2:5] == ["python", "-m", "pytest"], given
    limit = int(given[1].removesuffix("s"))
    env = _pytest_step()["env"]
    room = (int(env["JOB_TIMEOUT_MIN"]) - int(env["UNSEEN_SETUP_MIN"]) - int(env["CLASSIFY_MARGIN_MIN"])) * 60
    # What the job has left, up to the maximum (a few seconds of the step's own running are allowed for).
    expected = min(int(env["PYTEST_MAX_MIN"]) * 60, room - used)
    assert abs(limit - expected) <= 10 or (limit == int(env["PYTEST_MAX_MIN"]) * 60 == expected), (limit, expected)
    # A normal start is not shortened: the maximum is what it always was.
    if used_min <= 3:
        assert limit == 175 * 60
    # The invariant the job's own limit rests on: what pytest may use, plus what the job used before it, plus
    # the setup the clock did not see and the margin after pytest, is within the job's limit.
    assert limit + used + (room_gap := int(env["UNSEEN_SETUP_MIN"]) * 60 + int(env["CLASSIFY_MARGIN_MIN"]) * 60) \
        <= int(env["JOB_TIMEOUT_MIN"]) * 60 + 10, (limit, used, room_gap)


@pytest.mark.parametrize("used_min", [0, 4, 12, 40, 100, 150, 179, 185, 260])
def test_the_limit_never_outlasts_the_job_and_is_never_zero(tmp_path, used_min):
    # `timeout 0` means no limit at all: a job that is nearly out of time still gets a floor.
    given = _run_pytest_step(tmp_path, used_sec=used_min * 60)
    limit = int(given[1].removesuffix("s"))
    assert 60 <= limit <= int(_pytest_step()["env"]["PYTEST_MAX_MIN"]) * 60
    env = _pytest_step()["env"]
    room = (int(env["JOB_TIMEOUT_MIN"]) - int(env["UNSEEN_SETUP_MIN"]) - int(env["CLASSIFY_MARGIN_MIN"])) * 60
    assert limit <= max(60, room - used_min * 60 + 5), (limit, used_min)


def test_a_missing_clock_reading_is_not_a_shorter_limit(tmp_path):
    # If the first step wrote nothing, the step assumes none of the job was used rather than failing.
    given = _run_pytest_step(tmp_path, used_sec=None)
    assert int(given[1].removesuffix("s")) == int(_pytest_step()["env"]["PYTEST_MAX_MIN"]) * 60


@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the classifier's JavaScript")
def test_the_classifier_reads_the_limit_the_step_gave_pytest():
    # A kill at or after the limit is a timeout, an earlier one is memory: with a shorter limit
    # (a slow setup used some of the job's minutes) the dividing line moves with it.
    (timeout,) = wf._classify_failure("137", 100 * 60, limit_sec=100 * 60)
    assert "timeout" in timeout["title"] and "100-minute" in timeout["body"]
    (oom,) = wf._classify_failure("137", 99 * 60, limit_sec=100 * 60)
    assert "out-of-memory" in oom["title"] and "100-minute" in oom["body"]
    (timeout,) = wf._classify_failure("124", 5 * 60, limit_sec=100 * 60)
    assert "timeout" in timeout["title"]


# ---------------------------------------------------------------------------
# Tracking issues
# ---------------------------------------------------------------------------

def _holds(expression: str, ref: str, exit_code: str) -> bool:
    """Evaluate a step condition of the forms the slow lane uses, for one ref and exit code."""
    text = re.sub(r"^\$\{\{\s*(.*?)\s*\}\}$", r"\1", expression.strip())
    text = re.sub(r"startsWith\(github\.ref,\s*('[^']*')\)", r"ref.startswith(\1)", text)
    text = text.replace("github.ref", "ref").replace("steps.pytest.outputs.exit_code", "code")
    text = text.replace("&&", " and ").replace("||", " or ")
    text = re.sub(r"!(?!=)", " not ", text)
    assert re.fullmatch(r"[\w .()'/!=]*", text), f"a condition this test does not know how to read: {text!r}"
    return bool(eval(text, {}, {"ref": ref, "code": exit_code}))        # noqa: S307 - the repository's own text


ISSUE_STEP = wf.CLASSIFY_FAILURE
NO_ISSUE_STEP = "Say that no tracking issue is opened"


@pytest.mark.parametrize("ref, issue", [
    ("refs/heads/main", True),
    ("refs/heads/release/0.4.2", True),
    ("refs/heads/release/0.5.0", True),
    ("refs/heads/ci/slow-lane-hygiene", False),
    ("refs/heads/feat/anything", False),
    ("refs/heads/releases/x", False),
    ("refs/heads/not-release/0.4.2", False),
    ("refs/pull/12/merge", False),
    ("refs/tags/v0.4.2", False),
])
def test_an_issue_is_opened_only_for_a_run_on_main_or_a_release_branch(ref, issue):
    job = _job()
    condition = wf._step(job, ISSUE_STEP)["if"]
    assert _holds(condition, ref, "124") is issue, (ref, condition)
    # A clean run opens none anywhere, and the notice that says why is its exact complement.
    assert _holds(condition, ref, "0") is False
    notice = wf._step(job, NO_ISSUE_STEP)
    assert _holds(notice["if"], ref, "124") is (not issue), (ref, notice["if"])
    assert _holds(notice["if"], ref, "0") is False
    assert "continue-on-error" not in notice


def test_the_condition_reader_tells_a_throwaway_branch_from_main():
    assert _holds("steps.pytest.outputs.exit_code != '0' && github.ref == 'refs/heads/main'",
                  "refs/heads/main", "1")
    assert not _holds("steps.pytest.outputs.exit_code != '0' && github.ref == 'refs/heads/main'",
                      "refs/heads/x", "1")


def test_the_re_raise_still_fails_the_job_on_every_branch():
    # No issue on a throwaway branch is not a green run: the job is red and GitHub emails it.
    reraise = wf._step(_job(), wf.RERAISE)
    assert reraise["if"] == "steps.pytest.outputs.exit_code != '0'"
    assert "exit 1" in reraise["run"]


# ---------------------------------------------------------------------------
# The drift warning
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def drift():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        spec = importlib.util.spec_from_file_location("_slow_lane_drift_under_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(REPO_ROOT))
    return module


def _report(tmp_path: Path, cases: list[tuple[str, float]], name="results.xml") -> Path:
    body = "".join(f'<testcase classname="x" name="t{i}" file="{f}" time="{s}"/>' for i, (f, s) in enumerate(cases))
    path = tmp_path / name
    path.write_text(f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>', encoding="utf-8")
    return path


def _weights_of_shard(shard: int, n: int):
    from tests import _sharding
    weights = _sharding.load_weights()
    dealt = _sharding.deal(weights, n)
    return weights, [p for p in weights if dealt[p] == shard]


def test_the_step_runs_the_drift_script_on_the_shards_own_report():
    step = wf._step(_job(), "Weights against this run")
    assert "scripts/slow_lane_drift.py test-results-shard${{ matrix.shard }}.xml" in step["run"]
    assert '--shard "${{ matrix.shard }}/10:weighted"' in step["run"]
    assert "continue-on-error" not in step and "||" not in step["run"]
    # Only when pytest wrote a report: a timed-out shard writes none.
    assert "hashFiles('test-results-shard*.xml') != ''" in step["if"] and "!cancelled()" in step["if"]


def test_a_shard_on_its_prediction_is_quiet_and_one_far_past_it_names_the_files_nothing_measured(drift, tmp_path):
    n, shard = 10, 3
    weights, listed = _weights_of_shard(shard, n)
    predicted = sum(weights[p] for p in listed)
    # The shard ran every listed file at its weight (a runner as fast as the one the table came from) ...
    quiet = _report(tmp_path, [(p, float(weights[p])) for p in listed], "quiet.xml")
    # ... or those, plus two files the table has never seen, 12 and 25 minutes, and one small one.
    drifting = _report(tmp_path, [(p, float(weights[p])) for p in listed] + [
        ("tests/core/test_added_since_a.py", 1500.0), ("tests/core/test_added_since_b.py", 720.0),
        ("tests/core/test_added_since_c.py", 4.0)], "drifting.xml")
    out = tmp_path / "summary.md"
    env = {**os.environ, "GITHUB_STEP_SUMMARY": str(out), "PYTHONPATH": str(REPO_ROOT)}
    run = lambda report: subprocess.run(  # noqa: E731
        [sys.executable, str(SCRIPT), str(report), "--shard", f"{shard}/{n}:weighted"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    proc = run(quiet)
    assert proc.returncode == 0, proc.stderr
    assert "::warning" not in proc.stdout and f"{predicted / 60:.1f} min predicted (+0%)" in proc.stdout, proc.stdout
    proc = run(drifting)
    assert proc.returncode == 0, proc.stderr                       # it reports, it never fails the job
    (warning,) = [ln for ln in proc.stdout.splitlines() if ln.startswith("::warning")]
    assert warning.startswith("::warning title=Slow-lane weights are stale::")
    first, second = warning.index("test_added_since_a.py"), warning.index("test_added_since_b.py")
    assert first < second, "heaviest first"
    assert "25.0 min, not in the table" in warning and "test_added_since_c.py" not in warning, warning
    summary = out.read_text()
    assert "`tests/core/test_added_since_a.py` | 1500" in summary and "Refresh the table" in summary


def test_a_listed_file_that_grew_is_named_and_a_slow_runner_alone_is_not(drift, tmp_path):
    n, shard = 10, 3
    weights, listed = _weights_of_shard(shard, n)
    grew = max(listed, key=lambda p: weights[p])
    cases = [(p, float(weights[p]) * (6 if p == grew else 1)) for p in listed]
    result = drift.assess(dict(cases), weights, __import__("tests._sharding", fromlist=["x"]).deal(weights, n), shard, 10)
    assert [g[0] for g in result["grown"]] == [grew] and result["unmeasured"] == []
    assert drift.drifted(result, 0.5, 300)
    # Every file 1.4 times slower (a slow runner: the spread between runners is up to 1.6): under the margin.
    slow = {p: float(weights[p]) * 1.4 for p in listed}
    result = drift.assess(slow, weights, __import__("tests._sharding", fromlist=["x"]).deal(weights, n), shard, 10)
    assert result["grown"] == [] and result["unmeasured"] == [] and not drift.drifted(result, 0.5, 300)


def test_the_prediction_is_scaled_to_the_runner_the_report_says_it_ran_on(drift, tmp_path):
    """A shard 1.76 times its weights on a runner 1.6 times the table's is on its prediction.

    Shard 4 ran 95 to 108 minutes against 68 predicted on 2026-10-12 and 13,
    on runners that collected the suite in 37 to 41 s, where its heaviest
    file had been weighed on runners that took 26 and 28: "stale table or
    slow runner?" is answered by the report itself.  The same report
    without the runner's collection time warns.
    """
    from tests import _sharding
    n, shard = 10, 3
    weights, listed = _weights_of_shard(shard, n)
    predicted = sum(weights[p] for p in listed)
    speed = json.loads(_sharding.WEIGHTS_FILE.read_text(encoding="utf-8"))["runner_speed"]
    collect = speed["reference_seconds"] * 1.6 ** (1 / speed["exponent"])
    body = "".join(f'<testcase classname="x" name="t{i}" file="{p}" time="{weights[p] * 1.76}"/>'
                   for i, p in enumerate(listed))
    props = (f'<properties><property name="{_sharding.COLLECT_SECONDS_PROPERTY}" value="{collect:.4f}"/>'
             '<property name="runner_cpu" value="A Slow CPU"/></properties>')
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_STEP_SUMMARY"}
    env["PYTHONPATH"] = str(REPO_ROOT)
    outs = []
    for name, head in (("timed.xml", props), ("untimed.xml", "")):
        report = tmp_path / name
        report.write_text(f'<testsuites><testsuite name="pytest">{head}{body}</testsuite></testsuites>',
                          encoding="utf-8")
        proc = subprocess.run([sys.executable, str(SCRIPT), str(report), "--shard", f"{shard}/{n}:weighted"],
                              cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        outs.append(proc.stdout)
    timed, untimed = outs
    assert "::warning" not in timed, timed
    assert "this runner (A Slow CPU) collected the suite in" in timed and "so x1.60" in timed, timed
    assert f"{predicted * 1.6 / 60:.1f} min expected (+10%)" in timed, timed
    assert "::warning title=Slow-lane weights are stale::" in untimed, untimed
    result = drift.assess({p: weights[p] * 1.76 for p in listed}, weights, _sharding.deal(weights, n),
                          shard, 10, speed=1.6)
    assert result["predicted"] == predicted and result["expected"] == pytest.approx(predicted * 1.6)


@pytest.mark.parametrize("measured, predicted, margin, excess", [
    (3000, 2000, 0.5, 300),        # +50% exactly is not past the margin
    (3001, 2000, 0.5, 300),        # just past it
    (700, 400, 0.5, 300),          # +75% and exactly 300 s: both passed
    (650, 400, 0.5, 300),          # +62%, but only 250 s: a small shard that is relatively far off
    (10000, 10000, 0.5, 300),
])
def test_the_margin_and_the_floor_both_have_to_be_passed(drift, measured, predicted, margin, excess):
    result = {"measured": float(measured), "predicted": float(predicted), "unmeasured": [], "grown": []}
    expected = measured > predicted * (1 + margin) and measured - predicted >= excess
    assert drift.drifted(result, margin, excess) is expected


def test_the_drift_script_refuses_what_it_cannot_read(tmp_path):
    report = _report(tmp_path, [("tests/a/test_x.py", 1.0)])
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    for argv, why in (([str(tmp_path / "missing.xml"), "--shard", "1/10:weighted"], "no report"),
                      ([str(report), "--shard", "1/10"], "not the weighted split"),
                      ([str(report), "--shard", "11/10:weighted"], "no such shard")):
        proc = subprocess.run([sys.executable, str(SCRIPT), *argv], cwd=tmp_path, env=env,
                              capture_output=True, text=True, timeout=120)
        assert proc.returncode != 0, (why, proc.stdout, proc.stderr)
    empty = tmp_path / "empty.xml"
    empty.write_text("<testsuites/>")
    proc = subprocess.run([sys.executable, str(SCRIPT), str(empty), "--shard", "1/10:weighted"], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0 and "no test cases" in proc.stderr


def test_yaml_of_the_slow_lane_still_loads():
    assert yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "slow-tests.yml").read_text(encoding="utf-8"))
