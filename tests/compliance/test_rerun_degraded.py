"""``.github/workflows/rerun-degraded.yml``: the decision and the safety of the workflow.

The workflow re-runs the failed jobs of a ``CI`` or ``Slow tests`` run once,
and only when every failed job stopped at the degraded-runner canary.  It
cannot run until it is on ``main`` (``workflow_run`` workflows trigger from
the default branch's copy only), and a real degraded runner cannot be
produced on demand, so what can be pinned is:

* the decision, on canned API output: a marker present, absent, mixed with a
  real test failure, on a second attempt, with an expired artifact, with a
  cancelled job beside it;
* the paging of ``gh api --paginate`` output that the script parses;
* the script's calls: ``gh run rerun <id> --failed`` and nothing else that
  writes, with the run id read from the environment;
* what the workflow may do: ``actions: write`` and nothing else, no checkout,
  no ``${{ }}`` expression inside a shell command, triggered by the two
  workflows by their real names, on completion.

The script is extracted from the workflow file and run: nothing here
re-types the logic.
"""

from __future__ import annotations

import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
CANARY_STEP = "Runner canary (degraded runner check)"


def _workflow() -> dict:
    return yaml.safe_load((WORKFLOWS / "rerun-degraded.yml").read_text(encoding="utf-8"))


def _step_run() -> str:
    (step,) = _workflow()["jobs"]["rerun"]["steps"]
    return step["run"]


def _script_source() -> str:
    body = _step_run()
    match = re.fullmatch(r"python3 - <<'PY'\n(.*)\nPY\n?", body, re.DOTALL)
    assert match, "the step is `python3 - <<'PY'` ... `PY`: that is how this test finds the script"
    return match.group(1)


@pytest.fixture(scope="module")
def script() -> types.ModuleType:
    module = types.ModuleType("rerun_script")
    module.__dict__["__name__"] = "rerun_script"          # not __main__: nothing runs on import
    exec(compile(_script_source(), "rerun-degraded.yml:script", "exec"), module.__dict__)   # noqa: S102
    return module


def _job(job_id, name, conclusion="failure", canary="failure"):
    steps = [{"name": "Install dependencies", "conclusion": "success"}]
    if canary is not None:
        steps.append({"name": CANARY_STEP, "conclusion": canary})
    steps.append({"name": "Run tests", "conclusion": "skipped" if canary == "failure" else "failure"})
    return {"id": job_id, "name": name, "conclusion": conclusion, "steps": steps}


def _marker(job_id, expired=False):
    return {"name": f"degraded-runner-{job_id}", "expired": expired}


RUN = {"conclusion": "failure", "run_attempt": 1}
GOOD = _job(11, "test (3.12, 0.10.2, 3)", "success", None)


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------

def test_a_failed_job_that_stopped_at_the_canary_and_left_the_marker_is_rerun(script):
    rerun, why = script.decide(RUN, [GOOD, _job(12, "test (3.12, 0.10.2, 4)")], [_marker(12), {"name": "test-durations-x"}])
    assert rerun is True and "test (3.12, 0.10.2, 4)" in why


def test_every_failed_job_must_be_a_degraded_runner_not_just_one(script):
    jobs = [_job(12, "test (3.12, 0.10.2, 4)"), _job(13, "test (3.12, 0.11.2, 5)")]
    assert script.decide(RUN, jobs, [_marker(12), _marker(13)])[0] is True
    rerun, why = script.decide(RUN, jobs, [_marker(12)])               # the second has no artifact
    assert rerun is False and "test (3.12, 0.11.2, 5)" in why and "no degraded-runner artifact" in why


def test_a_job_that_failed_in_the_tests_is_not_a_degraded_runner_even_beside_one(script):
    real = _job(13, "test (3.12, 0.11.2, 5)", "failure", "success")    # canary passed, tests failed
    rerun, why = script.decide(RUN, [_job(12, "test (3.12, 0.10.2, 4)"), real], [_marker(12)])
    assert rerun is False and "canary step did not fail" in why


def test_a_marker_without_a_failed_canary_step_is_not_enough(script):
    # An artifact of that name next to a job whose tests failed (a test that planted it, say).
    real = _job(13, "test (3.12, 0.11.2, 5)", "failure", "success")
    assert script.decide(RUN, [real], [_marker(13)])[0] is False


def test_a_failed_canary_step_without_the_marker_is_not_enough(script):
    # The canary script itself crashed (exit 1): no marker was written, the step failed.
    assert script.decide(RUN, [_job(12, "test (3.12, 0.10.2, 4)")], [])[0] is False


def test_the_second_attempt_is_never_rerun(script):
    jobs = [_job(12, "test (3.12, 0.10.2, 4)")]
    rerun, why = script.decide({"conclusion": "failure", "run_attempt": 2}, jobs, [_marker(12)])
    assert rerun is False and "attempt 2" in why
    assert script.decide({"conclusion": "failure", "run_attempt": "1"}, jobs, [_marker(12)])[0] is True  # env strings


@pytest.mark.parametrize("conclusion", ["success", "cancelled", "timed_out", None, "skipped"])
def test_only_a_failed_run_is_considered(script, conclusion):
    jobs = [_job(12, "test (3.12, 0.10.2, 4)")]
    assert script.decide({"conclusion": conclusion, "run_attempt": 1}, jobs, [_marker(12)])[0] is False


def test_an_expired_marker_does_not_count(script):
    assert script.decide(RUN, [_job(12, "t")], [_marker(12, expired=True)])[0] is False


def test_a_timed_out_or_cancelled_job_beside_the_degraded_one_blocks_the_rerun(script):
    degraded = _job(12, "test (3.12, 0.10.2, 4)")
    for conclusion in ("timed_out", "cancelled"):
        other = _job(13, "slow (3.12, 0.10.2, 9)", conclusion, "success")
        assert script.decide(RUN, [degraded, other], [_marker(12)])[0] is False, conclusion


def test_skipped_and_successful_jobs_are_ignored(script):
    jobs = [GOOD, {"id": 14, "name": "docs", "conclusion": "skipped", "steps": []}, _job(12, "test (3.12, 0.10.2, 4)")]
    assert script.decide(RUN, jobs, [_marker(12)])[0] is True


def test_a_failed_run_with_no_failed_job_is_not_rerun(script):
    # The conclusion can lag the jobs; the rerun would have nothing to rerun.
    assert script.decide(RUN, [GOOD], [])[0] is False


def test_the_lane_summary_failing_because_every_shard_was_degraded_does_not_block_the_rerun(script):
    # When all six shards of a lane stop at the canary, "Test durations (<jax>)" finds no report and
    # fails: a consequence of the degraded jobs, re-run with them.  A test job that failed for real
    # beside them still blocks, and the summary alone never causes a re-run.
    summary = {"id": 30, "name": "Test durations (0.10.2)", "conclusion": "failure", "steps": []}
    degraded = [_job(i, f"test (3.12, 0.10.2, {i})") for i in range(1, 4)]
    markers = [_marker(i) for i in range(1, 4)]
    assert script.decide(RUN, [*degraded, summary], markers)[0] is True
    real = _job(13, "test (3.12, 0.11.2, 5)", "failure", "success")
    assert script.decide(RUN, [*degraded, summary, real], markers)[0] is False
    assert script.decide(RUN, [GOOD, summary], [])[0] is False


def test_a_marker_for_another_job_does_not_mark_this_one(script):
    assert script.decide(RUN, [_job(12, "t")], [_marker(21)])[0] is False


def test_the_names_the_decision_reads_are_the_ones_the_lanes_write():
    for workflow in ("ci.yml", "slow-tests.yml"):
        steps = [s for job in yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))["jobs"].values()
                 for s in job.get("steps", [])]
        assert any(s.get("name") == CANARY_STEP for s in steps), workflow
        marker = [s for s in steps if s.get("name") == "Mark a degraded runner"]
        assert [m["with"]["name"] for m in marker] == ["degraded-runner-${{ job.check_run_id }}"], workflow


# ---------------------------------------------------------------------------
# paging, and the calls
# ---------------------------------------------------------------------------

def test_paginated_gh_output_is_every_page_back_to_back(script):
    pages = script._documents('{"jobs": [{"id": 1}], "total_count": 2}\n{"jobs": [{"id": 2}], "total_count": 2}\n')
    assert [j["id"] for page in pages for j in page["jobs"]] == [1, 2]
    assert script._documents("") == [] and script._documents(" \n ") == []


def _fake_gh(monkeypatch, script, jobs, artifacts, calls):
    import json

    def run(cmd, capture_output=False, text=False, check=False):
        calls.append(list(cmd))
        if cmd[:3] == ["gh", "api", "--paginate"]:
            path = cmd[3]
            if "/jobs" in path:
                return types.SimpleNamespace(stdout=json.dumps({"jobs": jobs[:1]}) + json.dumps({"jobs": jobs[1:]}))
            if "/artifacts" in path:
                return types.SimpleNamespace(stdout=json.dumps({"artifacts": artifacts}))
        return types.SimpleNamespace(stdout="", returncode=0)
    monkeypatch.setattr(script.subprocess, "run", run)


def test_main_reruns_the_failed_jobs_once_by_run_id_and_reads_the_api_for_this_run(script, monkeypatch, tmp_path, capsys):
    calls = []
    _fake_gh(monkeypatch, script, [GOOD, _job(12, "test (3.12, 0.10.2, 4)")], [_marker(12)], calls)
    monkeypatch.setenv("RUN_ID", "555")
    monkeypatch.setenv("RUN_ATTEMPT", "1")
    monkeypatch.setenv("RUN_CONCLUSION", "failure")
    monkeypatch.setenv("GH_REPO", "owner/name")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    assert script.main() == 0
    assert ["gh", "run", "rerun", "555", "--failed"] in calls
    assert sum(1 for c in calls if c[:3] == ["gh", "run", "rerun"]) == 1
    paths = [c[3] for c in calls if c[:3] == ["gh", "api", "--paginate"]]
    assert paths == ["repos/owner/name/actions/runs/555/jobs?per_page=100&filter=latest",
                     "repos/owner/name/actions/runs/555/artifacts?per_page=100"]
    assert "re-running the failed jobs" in capsys.readouterr().out
    assert "re-ran the failed jobs" in (tmp_path / "summary").read_text()


def test_main_does_not_rerun_when_the_decision_is_no(script, monkeypatch, tmp_path, capsys):
    calls = []
    _fake_gh(monkeypatch, script, [_job(12, "test (3.12, 0.10.2, 4)"), _job(13, "t2", "failure", "success")],
             [_marker(12)], calls)
    for k, v in dict(RUN_ID="556", RUN_ATTEMPT="1", RUN_CONCLUSION="failure", GH_REPO="o/n").items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert script.main() == 0
    assert not [c for c in calls if c[:3] == ["gh", "run", "rerun"]]
    assert "not re-running" in capsys.readouterr().out


def test_the_script_writes_through_gh_run_rerun_only():
    source = _script_source()
    gh_calls = re.findall(r'\["gh", "([a-z]+)"(?:, "([a-z-]+)")?', source)
    assert set(gh_calls) == {("api", "--paginate"), ("run", "rerun")}, gh_calls
    assert "--method" not in source and "-X" not in source


# ---------------------------------------------------------------------------
# what the workflow may do
# ---------------------------------------------------------------------------

def test_the_workflow_is_triggered_by_the_two_lanes_by_name_on_completion():
    workflow = _workflow()
    on = workflow["on"] if "on" in workflow else workflow[True]
    assert set(on) == {"workflow_run"}
    names = {yaml.safe_load((WORKFLOWS / f).read_text(encoding="utf-8"))["name"] for f in ("ci.yml", "slow-tests.yml")}
    assert set(on["workflow_run"]["workflows"]) == names == {"CI", "Slow tests (scheduled)"}
    assert on["workflow_run"]["types"] == ["completed"]


def test_the_token_is_actions_write_and_nothing_else_at_every_level():
    workflow = _workflow()
    assert workflow["permissions"] == {"actions": "write"}
    for job in workflow["jobs"].values():
        assert job.get("permissions", workflow["permissions"]) == {"actions": "write"}


def test_it_never_checks_out_or_executes_the_runs_code():
    text = (WORKFLOWS / "rerun-degraded.yml").read_text(encoding="utf-8")
    assert "checkout" not in re.sub(r"(?m)^\s*#.*$", "", text)                   # not even a comment's-worth of step
    (step,) = _workflow()["jobs"]["rerun"]["steps"]
    assert "uses" not in step
    assert "head_sha" not in text and "head_branch" not in text and "pull_request" not in re.sub(r"(?m)^\s*#.*$", "", text)


def test_no_expression_is_interpolated_into_the_shell_script():
    """The script's inputs reach it through ``env:``, never through ``${{ }}`` in the command."""
    (step,) = _workflow()["jobs"]["rerun"]["steps"]
    assert "${{" not in step["run"]
    assert set(step["env"]) == {"GH_TOKEN", "GH_REPO", "RUN_ID", "RUN_ATTEMPT", "RUN_CONCLUSION"}
    # And the quoted heredoc: the shell expands nothing in it.
    assert step["run"].startswith("python3 - <<'PY'\n")


def test_the_job_runs_only_for_a_first_attempt_that_failed_and_has_a_limit():
    job = _workflow()["jobs"]["rerun"]
    condition = job["if"]
    assert "github.event.workflow_run.conclusion == 'failure'" in condition
    assert "github.event.workflow_run.run_attempt == 1" in condition
    assert job["runs-on"] == "ubuntu-24.04" and job["timeout-minutes"] <= 15


def test_the_script_compiles_and_runs_under_python_3(tmp_path):
    """The extracted script as ``python3 -`` runs it, with nothing to read: it must fail on the missing
    environment, not on a syntax error."""
    proc = subprocess.run([sys.executable, "-I", "-"], input=_script_source(), capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin"}, timeout=60)
    assert proc.returncode != 0 and "KeyError" in proc.stderr and "RUN_ID" in proc.stderr
