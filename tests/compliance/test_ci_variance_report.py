"""``scripts/ci_variance_report.py`` on canned ``gh`` JSON and job logs.

The report is how the CI lanes' spread by CPU is read, and how a degraded
runner shows up beside its siblings.  These tests feed it hand-made jobs in
the shape of ``gh run view --json jobs`` (no network) and pin: what a job
record is read from, the CPU model from either log format, the tables, and
the flag rule -- including the two cases it exists for, the 3.8x per-push
job of 2026-10-13 and a job on the CPU that sets the baseline, which must
not be flagged for being slow when its own CPU is.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("ci_variance_report", REPO_ROOT / "scripts" / "ci_variance_report.py")
cvr = importlib.util.module_from_spec(_spec)
sys.modules["ci_variance_report"] = cvr
_spec.loader.exec_module(cvr)

EPYC7763 = "AMD EPYC 7763 64-Core Processor"
EPYC9V45 = "AMD EPYC 9V45 96-Core Processor"
T0 = datetime(2026, 10, 14, 1, 0, tzinfo=timezone.utc)


def _step(name, start_min, minutes):
    start = T0 + timedelta(minutes=start_min)
    return {"name": name, "startedAt": start.isoformat().replace("+00:00", "Z"),
            "completedAt": (start + timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z"),
            "conclusion": "success"}


def _job(name, job_id, test_min, install_min=1.0, test_step="Run tests", conclusion="success"):
    return {"name": name, "databaseId": job_id, "conclusion": conclusion,
            "steps": [_step("Install dependencies", 0, install_min), _step(test_step, 2, test_min)]}


def _log(cpu, style="per-push", canary=None):
    lines = ["2026-10-14T01:00:00Z other output"]
    if style == "per-push":
        lines.append(f"2026-10-14T01:03:00Z compilation cache: cold on '{cpu}'")
    else:
        lines.append(f"2026-10-14T01:00:30Z runner: cpu '{cpu}', 4 vCPU, memory 15.6 GiB total, 14.9 GiB available")
    if canary:
        lines.append(f"runner: canary numpy matmul {canary / 2:.2f} s + jit compile {canary / 2:.2f} s = {canary:.2f} s "
                     f"(limit 12.0 s), import 0.50 s; install 60 s (limit 190 s)")
    return "\n".join(lines)


class FakeGh:
    """Runs, jobs and logs from dictionaries."""

    def __init__(self, runs, logs, listing=None):
        self.runs, self.logs, self.listing, self.asked = runs, logs, listing or {}, []

    def run_jobs(self, run_id):
        return self.runs[run_id]

    def job_log(self, job_id):
        return self.logs.get(job_id, "")

    def list_runs(self, workflow, n, event=None, status=None):
        self.asked.append((workflow, n, event, status))
        return self.listing[workflow][:n]


# ---------------------------------------------------------------------------
# one job
# ---------------------------------------------------------------------------

def test_a_record_reads_the_test_step_minutes_the_install_minutes_and_the_cpu():
    record = cvr.parse_job("7", _job("test (3.12, 0.10.2, 3)", 11, 23.5, install_min=1.25),
                           _log(EPYC7763, canary=3.0))
    assert (record.run, record.lane, record.shard, record.job_id) == ("7", "per-push", "test (3.12, 0.10.2, 3)", 11)
    assert record.test_min == pytest.approx(23.5) and record.install_min == pytest.approx(1.25)
    assert record.cpu == EPYC7763 and record.canary_s == pytest.approx(3.0)


def test_the_slow_lane_is_read_from_its_own_step_and_its_new_log_line():
    job = _job("slow (3.12, 0.11.2, 4)", 12, 61.0, test_step="Run full suite (slow lane)")
    record = cvr.parse_job("8", job, _log(EPYC9V45, style="health"))
    assert record.lane == "slow" and record.test_min == pytest.approx(61.0) and record.cpu == EPYC9V45


def test_a_log_that_names_no_cpu_is_unknown_not_an_error():
    # The slow lane logged none before scripts/ci_runner_health.py.
    assert cvr.cpu_from_log("nothing of the sort") == "unknown"
    assert cvr.cpu_from_log(None) == "unknown"
    record = cvr.parse_job("8", _job("slow (3.12, 0.11.2, 4)", 12, 61.0, test_step="Run full suite (slow lane)"), "")
    assert record.cpu == "unknown" and record.canary_s is None


@pytest.mark.parametrize("job", [
    {"name": "compliance", "databaseId": 1, "conclusion": "success",
     "steps": [_step("Run tests", 0, 5)]},                                    # not a test job
    {"name": "test (3.12, 0.10.2, 1)", "databaseId": 2, "conclusion": "skipped", "steps": []},
    {"name": "test (3.12, 0.10.2, 2)", "databaseId": 3, "conclusion": "failure",
     "steps": [_step("Install dependencies", 0, 1), {"name": "Run tests", "startedAt": None, "completedAt": None}]},
])
def test_a_job_with_no_finished_test_step_is_not_a_record(job):
    assert cvr.parse_job("9", job, "") is None


def test_a_degraded_job_that_stopped_at_the_canary_has_no_test_minutes_and_is_left_out():
    job = {"name": "test (3.12, 0.10.2, 5)", "databaseId": 5, "conclusion": "failure",
           "steps": [_step("Install dependencies", 0, 8), _step("Runner canary (degraded runner check)", 9, 0.1),
                     {"name": "Run tests", "conclusion": "skipped"}]}
    assert cvr.parse_job("9", job, _log(EPYC7763)) is None


# ---------------------------------------------------------------------------
# the tables and the flags
# ---------------------------------------------------------------------------

def _records(spec):
    """``[(run, shard, cpu, minutes)]`` -> records."""
    return [cvr.JobRecord(run=run, lane="per-push", shard=shard, job_id=i, conclusion="success", test_min=minutes,
                          install_min=1.0, cpu=cpu)
            for i, (run, shard, cpu, minutes) in enumerate(spec)]


SHARD = "test (3.12, 0.10.2, 3)"


def test_a_job_3_8_times_its_siblings_on_the_same_cpu_is_flagged_and_they_are_not():
    records = _records([("1", SHARD, EPYC7763, 30.0), ("2", SHARD, EPYC7763, 31.0), ("3", SHARD, EPYC7763, 29.5),
                        ("4", SHARD, EPYC7763, 30.5), ("5", SHARD, EPYC7763, 114.0)])
    flagged = cvr.flag_outliers(records)
    assert [r.run for r in flagged] == ["5"]
    assert "3.8x its CPU's median of 4" in flagged[0].flag


def test_a_job_is_judged_against_the_others_so_it_cannot_pull_its_own_baseline():
    # Three fast jobs at 30 and three slow ones at 90.  Each slow job is compared with the other
    # five (median 30): flagged.  Counting itself the median would be 60 and 90 is not over 1.5x60.
    records = _records([("1", SHARD, EPYC7763, 30.0), ("2", SHARD, EPYC7763, 30.0), ("3", SHARD, EPYC7763, 30.0),
                        ("4", SHARD, EPYC7763, 90.0), ("5", SHARD, EPYC7763, 90.0), ("6", SHARD, EPYC7763, 90.0)])
    assert {r.run for r in cvr.flag_outliers(records)} == {"4", "5", "6"}


def test_the_generation_spread_alone_is_not_an_outlier():
    # 19.6 min on the newest CPU and 30 on the oldest is the 1.5x of the pool, in both directions.
    records = _records([("1", SHARD, EPYC9V45, 19.6), ("2", SHARD, EPYC9V45, 20.0), ("3", SHARD, EPYC9V45, 19.8),
                        ("4", SHARD, EPYC7763, 30.0), ("5", SHARD, EPYC7763, 30.4), ("6", SHARD, EPYC7763, 29.9)])
    assert cvr.flag_outliers(records) == []


def test_a_cpu_with_too_few_peers_falls_back_to_the_shard_with_the_spread_allowed():
    base = [("1", SHARD, EPYC7763, 30.0), ("2", SHARD, EPYC7763, 31.0), ("3", SHARD, EPYC7763, 29.0)]
    # Alone on its CPU at 50 (1.67x the shard's median, over 1.5 but under 1.5 * 1.5): not flagged.
    assert cvr.flag_outliers(_records(base + [("4", SHARD, EPYC9V45, 50.0)])) == []
    # Alone on its CPU at 100 (3.3x): flagged, against the shard's median.
    flagged = cvr.flag_outliers(_records(base + [("4", SHARD, EPYC9V45, 100.0)]))
    assert [r.run for r in flagged] == ["4"] and "over all CPUs" in flagged[0].flag


def test_nothing_is_flagged_without_enough_peers():
    assert cvr.flag_outliers(_records([("1", SHARD, EPYC7763, 30.0), ("2", SHARD, EPYC7763, 300.0)])) == []


def test_shards_are_judged_apart_a_heavy_shard_is_not_an_outlier_of_a_light_one():
    other = "test (3.12, 0.10.2, 4)"
    records = _records([(str(i), SHARD, EPYC7763, 20.0) for i in range(4)]
                       + [(str(i), other, EPYC7763, 45.0) for i in range(4)])
    assert cvr.flag_outliers(records) == []


def test_the_tables_name_the_cpu_ratio_and_the_shard_spread():
    records = _records([("1", SHARD, EPYC9V45, 20.0), ("2", SHARD, EPYC7763, 30.0), ("3", SHARD, EPYC7763, 32.0)])
    rows = cvr.by_cpu(records)
    assert [(r[1], r[2], r[3]) for r in rows] == [(EPYC9V45, 1, 20.0), (EPYC7763, 2, 31.0)]
    assert rows[1][6] == pytest.approx(1.55)
    (shard_row,) = cvr.by_shard(records)
    assert shard_row[1] == SHARD and shard_row[3] == pytest.approx(1.6)
    text = cvr.render(records)
    assert "20.0(9V45)  30.0(7763)  32.0(7763)   max/min 1.60" in text
    assert "1.55x" in text and "none" in text


def test_the_install_and_canary_block_is_what_the_limits_are_derived_from():
    records = _records([("1", SHARD, EPYC7763, 30.0), ("2", SHARD, EPYC7763, 31.0)])
    records[0].install_min, records[0].canary_s = 1.0, 2.0
    records[1].install_min, records[1].canary_s = 1.5, 3.0
    (row,) = cvr.install_and_canary(records)
    assert row == (EPYC7763, 2, 75.0, 90.0, 2.5, 3.0)


# ---------------------------------------------------------------------------
# the command, with gh replaced
# ---------------------------------------------------------------------------

def test_the_command_reads_the_runs_it_is_given_and_the_last_green_pushes(capsys):
    runs = {"10": [_job("test (3.12, 0.10.2, 1)", 100, 25.0), _job("compliance", 101, 3.0)],
            "11": [_job("test (3.12, 0.10.2, 1)", 110, 26.0),
                   _job("slow (3.12, 0.10.2, 1)", 111, 70.0, test_step="Run full suite (slow lane)")]}
    logs = {100: _log(EPYC7763), 110: _log(EPYC7763), 111: _log(EPYC7763, style="health")}
    gh = FakeGh(runs, logs, listing={"ci.yml": ["10"], "slow-tests.yml": ["11"]})
    assert cvr.main(["--last-push", "1", "--last-slow", "1"], fetcher=gh) == 0
    out = capsys.readouterr().out
    assert gh.asked == [("ci.yml", 1, "push", "success"), ("slow-tests.yml", 1, None, "completed")]
    assert "3 test jobs from 2 runs: 10, 11" in out
    assert "per-push" in out and "slow" in out


def test_the_command_says_so_when_no_job_finished(capsys):
    gh = FakeGh({"10": [{"name": "test (3.12, 0.10.2, 1)", "databaseId": 1, "conclusion": "cancelled", "steps": []}]}, {})
    assert cvr.main(["--run", "10"], fetcher=gh) == 1
    assert "no finished test jobs" in capsys.readouterr().err


def test_the_command_needs_a_source_of_runs():
    with pytest.raises(SystemExit):
        cvr.main([], fetcher=FakeGh({}, {}))
