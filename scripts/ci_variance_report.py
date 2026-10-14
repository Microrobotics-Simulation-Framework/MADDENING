#!/usr/bin/env python3
"""How much the test-step minutes of the CI lanes vary, by CPU model and by shard.

The jobs of both test lanes run on whichever CPU GitHub's pool gives them
(about 1.5x apart on this suite), and now and then on a degraded machine
that is 4-10x worse.  This tabulates, from runs already made::

    python scripts/ci_variance_report.py --last-push 6 --last-slow 3
    python scripts/ci_variance_report.py --run 38437636720 --run 38439179642
    python scripts/ci_variance_report.py --last-push 6 --flag-ratio 1.3

* ``--run ID``: a run of either workflow, by id (repeatable);
* ``--last-push N``: the last N *green push* runs of ``ci.yml`` (pushes run
  cold, so their numbers are comparable; pull requests are warm);
* ``--last-slow N``: the last N completed runs of ``slow-tests.yml``.

For every test job it reads the minutes of the test step (``Run tests`` of
the per-push lane, ``Run full suite (slow lane)`` of the slow lane) and the
install step from ``gh run view --json jobs``, and the CPU model and the
canary figures from the job log (``compilation cache: <mode> on '<cpu>'`` or
``runner: cpu '<cpu>'``).  A log with neither names no CPU (the slow lane
logged none before ``scripts/ci_runner_health.py``): those jobs are listed
under ``unknown``.

Three tables:

1. by lane and CPU model: jobs, median, min and max test minutes, and the
   median's ratio to the fastest model's;
2. by shard: every run's minutes with the CPU it drew, and the max/min
   ratio -- the part of a shard's spread that the CPU explains is visible
   next to it;
3. jobs flagged as far outside their distribution: a job is flagged when its
   minutes exceed ``--flag-ratio`` (default 1.5) times the median of the
   *other* jobs of the same shard on the same CPU model, when there are at
   least ``--min-group`` (3) of them; otherwise when they exceed
   ``--flag-ratio`` times ``CPU_SPREAD`` (1.5, the generation-to-generation
   spread) times the median of the other jobs of that shard on any CPU.  With
   fewer than that nothing can be said and the job is not flagged.

A fourth block lists the install and canary figures by CPU model: the data
the limits in ``scripts/ci_runner_health.py`` come from.

Needs the ``gh`` CLI, logged in.  ``--repo OWNER/NAME`` (default: the
current repository).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

#: Step names, per lane, whose duration is the test minutes and the install minutes.
LANES = {
    "per-push": {"test_step": "Run tests", "job_prefix": "test ("},
    "slow": {"test_step": "Run full suite (slow lane)", "job_prefix": "slow ("},
}
INSTALL_STEP = "Install dependencies"
#: Generation-to-generation spread of this suite's time, used when a CPU's own group is too small.
CPU_SPREAD = 1.5

_CPU_PER_PUSH = re.compile(r"compilation cache: \w+ on '([^']+)'")
_CPU_HEALTH = re.compile(r"runner: cpu '([^']+)', (\d+) vCPU")
_CANARY = re.compile(r"runner: canary numpy matmul ([\d.]+) s \+ jit compile ([\d.]+) s = ([\d.]+) s")


@dataclass
class JobRecord:
    run: str
    lane: str
    shard: str            # the job's name without the run: "test (3.12, 0.10.2, 3)"
    job_id: int
    conclusion: str
    test_min: float
    install_min: float | None
    cpu: str
    canary_s: float | None = None
    flag: str = ""


# ---------------------------------------------------------------------------
# Reading one job
# ---------------------------------------------------------------------------

def _minutes(step: dict) -> float | None:
    if not step or not step.get("startedAt") or not step.get("completedAt"):
        return None
    t = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))   # noqa: E731
    return (t(step["completedAt"]) - t(step["startedAt"])).total_seconds() / 60.0


def cpu_from_log(log: str | None) -> str:
    """The CPU model a job's log names, or ``unknown``."""
    for pattern in (_CPU_HEALTH, _CPU_PER_PUSH):
        m = pattern.search(log or "")
        if m:
            return m.group(1)
    return "unknown"


def canary_from_log(log: str | None) -> float | None:
    m = _CANARY.search(log or "")
    return float(m.group(3)) if m else None


def short_cpu(cpu: str) -> str:
    """``7763`` for an EPYC 7763, ``8573C`` for a Xeon Platinum 8573C: enough to tell models apart in a table."""
    m = re.search(r"EPYC\s+(\S+)", cpu) or re.search(r"\b(\d{4}\w*(?:-\w)?)\b", cpu)
    return m.group(1) if m else (cpu.split()[0] if cpu.split() else "?")


def lane_of(job_name: str) -> str | None:
    for lane, info in LANES.items():
        if job_name.startswith(info["job_prefix"]):
            return lane
    return None


def parse_job(run_id: str, job: dict, log: str | None) -> JobRecord | None:
    """A ``JobRecord`` from one entry of ``gh run view --json jobs``, or ``None`` when it is not a test job
    or its test step never finished (skipped, cancelled before it, still running)."""
    lane = lane_of(job.get("name", ""))
    if lane is None:
        return None
    steps = {s.get("name"): s for s in job.get("steps", [])}
    test_min = _minutes(steps.get(LANES[lane]["test_step"], {}))
    if test_min is None:
        return None
    return JobRecord(run=str(run_id), lane=lane, shard=job["name"], job_id=int(job.get("databaseId") or 0),
                     conclusion=str(job.get("conclusion")), test_min=test_min,
                     install_min=_minutes(steps.get(INSTALL_STEP, {})),
                     cpu=cpu_from_log(log), canary_s=canary_from_log(log))


# ---------------------------------------------------------------------------
# The analysis (pure)
# ---------------------------------------------------------------------------

def flag_outliers(records: list[JobRecord], flag_ratio: float = 1.5, min_group: int = 3) -> list[JobRecord]:
    """Set ``flag`` on each job far outside its distribution; return the flagged ones.

    See the module docstring for the rule.  Each job is compared with the
    *others*: a degraded job must not pull the median it is judged against.
    """
    for r in records:
        r.flag = ""
        others = [o for o in records if o is not r and o.lane == r.lane and o.shard == r.shard]
        same_cpu = [o.test_min for o in others if o.cpu == r.cpu and r.cpu != "unknown"]
        if len(same_cpu) >= min_group:
            reference, limit, basis = statistics.median(same_cpu), flag_ratio, f"its CPU's median of {len(same_cpu)}"
        elif len(others) >= min_group:
            reference = statistics.median(o.test_min for o in others)
            limit, basis = flag_ratio * CPU_SPREAD, f"the shard's median over all CPUs of {len(others)}"
        else:
            continue
        if reference > 0 and r.test_min > limit * reference:
            r.flag = (f"{r.test_min:.1f} min is {r.test_min / reference:.1f}x {basis} "
                      f"({reference:.1f} min; flagged over {limit:.2f}x)")
    return [r for r in records if r.flag]


def _stats(values: list[float]) -> tuple[int, float, float, float]:
    return len(values), statistics.median(values), min(values), max(values)


def by_cpu(records: list[JobRecord]) -> list[tuple[str, str, int, float, float, float, float]]:
    """``[(lane, cpu, jobs, median, min, max, median / the lane's fastest median)]``."""
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        groups[(r.lane, r.cpu)].append(r.test_min)
    fastest: dict[str, float] = {}
    for (lane, cpu), v in groups.items():
        if cpu != "unknown":
            fastest[lane] = min(fastest.get(lane, float("inf")), statistics.median(v))
    rows = []
    for (lane, cpu), v in sorted(groups.items(), key=lambda kv: (kv[0][0], statistics.median(kv[1]))):
        n, med, lo, hi = _stats(v)
        base = fastest.get(lane)
        rows.append((lane, cpu, n, med, lo, hi, (med / base) if base and cpu != "unknown" else float("nan")))
    return rows


def by_shard(records: list[JobRecord]) -> list[tuple[str, str, list[tuple[float, str]], float]]:
    """``[(lane, shard, [(minutes, cpu)], max/min)]``, shards in name order, runs by minutes."""
    groups: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
    for r in records:
        groups[(r.lane, r.shard)].append((r.test_min, r.cpu))
    rows = []
    for (lane, shard), v in sorted(groups.items()):
        v.sort()
        rows.append((lane, shard, v, (v[-1][0] / v[0][0]) if v[0][0] > 0 else float("nan")))
    return rows


def install_and_canary(records: list[JobRecord]) -> list[tuple[str, int, float | None, float | None, float | None, float | None]]:
    """``[(cpu, jobs, median install s, max install s, median canary s, max canary s)]`` over every lane."""
    groups: dict[str, list[JobRecord]] = defaultdict(list)
    for r in records:
        groups[r.cpu].append(r)
    rows = []
    for cpu, rs in sorted(groups.items()):
        ins = [r.install_min * 60 for r in rs if r.install_min is not None]
        can = [r.canary_s for r in rs if r.canary_s is not None]
        rows.append((cpu, len(rs), statistics.median(ins) if ins else None, max(ins) if ins else None,
                     statistics.median(can) if can else None, max(can) if can else None))
    return rows


def render(records: list[JobRecord], flag_ratio: float = 1.5, min_group: int = 3) -> str:
    """The report, as text."""
    out: list[str] = []
    flagged = flag_outliers(records, flag_ratio, min_group)
    runs = sorted({r.run for r in records})
    out.append(f"{len(records)} test jobs from {len(runs)} runs: {', '.join(runs)}")
    out.append("\nTest-step minutes by lane and CPU model")
    out.append(f"  {'lane':<9}{'cpu model':<46}{'jobs':>5}{'median':>8}{'min':>7}{'max':>7}{'vs fastest':>11}")
    for lane, cpu, n, med, lo, hi, rel in by_cpu(records):
        out.append(f"  {lane:<9}{cpu:<46}{n:>5}{med:>8.1f}{lo:>7.1f}{hi:>7.1f}"
                   f"{('%.2fx' % rel) if rel == rel else '-':>11}")
    out.append("\nTest-step minutes by shard (minutes, CPU: shortest to longest)")
    for lane, shard, v, ratio in by_shard(records):
        cells = "  ".join(f"{m:.1f}({short_cpu(cpu)})" for m, cpu in v)
        out.append(f"  {shard:<26}{cells}   max/min {ratio:.2f}")
    out.append(f"\nJobs far outside their distribution (over {flag_ratio}x their CPU's median, "
               f"or {flag_ratio * CPU_SPREAD:.2f}x the shard's median when the CPU has fewer than {min_group} peers)")
    if flagged:
        for r in sorted(flagged, key=lambda r: -r.test_min):
            out.append(f"  run {r.run} {r.shard} [{r.cpu}] job {r.job_id}: {r.flag}")
    else:
        out.append("  none")
    out.append("\nInstall and canary seconds by CPU model (the data behind scripts/ci_runner_health.py's limits)")
    out.append(f"  {'cpu model':<46}{'jobs':>5}{'install med':>12}{'max':>6}{'canary med':>11}{'max':>6}")
    f = lambda v, p: "-" if v is None else f"{v:.{p}f}"           # noqa: E731
    for cpu, n, im, ix, cm, cx in install_and_canary(records):
        out.append(f"  {cpu:<46}{n:>5}{f(im, 0):>12}{f(ix, 0):>6}{f(cm, 2):>11}{f(cx, 2):>6}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

class GhFetcher:
    """Reads runs, jobs and logs with the ``gh`` CLI."""

    def __init__(self, repo: str | None = None):
        self.repo = repo or self._gh(["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"]).strip()

    @staticmethod
    def _gh(args: list[str]) -> str:
        return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout

    def list_runs(self, workflow: str, n: int, event: str | None = None, status: str | None = None) -> list[str]:
        args = ["run", "list", "--repo", self.repo, "--workflow", workflow, "--limit", str(max(n * 5, 30)),
                "--json", "databaseId,conclusion,event,status"]
        if event:
            args += ["--event", event]
        rows = json.loads(self._gh(args))
        keep = [str(r["databaseId"]) for r in rows
                if (status != "success" or r.get("conclusion") == "success")
                and (status != "completed" or r.get("status") == "completed")]
        return keep[:n]

    def run_jobs(self, run_id: str) -> list[dict]:
        return json.loads(self._gh(["run", "view", run_id, "--repo", self.repo, "--json", "jobs"]))["jobs"]

    def job_log(self, job_id: int) -> str:
        done = subprocess.run(["gh", "api", f"repos/{self.repo}/actions/jobs/{job_id}/logs"],
                              capture_output=True, text=True)
        return done.stdout if done.returncode == 0 else ""


def collect(fetcher, run_ids: list[str]) -> list[JobRecord]:
    """A ``JobRecord`` for every finished test job of the runs."""
    records = []
    for run_id in run_ids:
        for job in fetcher.run_jobs(run_id):
            if lane_of(job.get("name", "")) is None or not job.get("steps"):
                continue
            record = parse_job(run_id, job, fetcher.job_log(job["databaseId"]))
            if record:
                records.append(record)
    return records


def main(argv: list[str] | None = None, fetcher=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", default=[], metavar="ID", help="a run id (repeatable)")
    ap.add_argument("--last-push", type=int, default=0, metavar="N", help="the last N green push runs of ci.yml")
    ap.add_argument("--last-slow", type=int, default=0, metavar="N", help="the last N completed runs of slow-tests.yml")
    ap.add_argument("--flag-ratio", type=float, default=1.5,
                    help="flag a job over this many times its CPU's median (default 1.5)")
    ap.add_argument("--min-group", type=int, default=3,
                    help="peers on the same CPU and shard a job needs to be judged by them (default 3)")
    ap.add_argument("--repo", default=None, help="OWNER/NAME (default: the current repository)")
    args = ap.parse_args(argv)
    if not (args.run or args.last_push or args.last_slow):
        ap.error("give --run, --last-push or --last-slow")
    fetcher = fetcher or GhFetcher(args.repo)
    ids = list(args.run)
    if args.last_push:
        ids += fetcher.list_runs("ci.yml", args.last_push, event="push", status="success")
    if args.last_slow:
        ids += fetcher.list_runs("slow-tests.yml", args.last_slow, status="completed")
    records = collect(fetcher, list(dict.fromkeys(ids)))
    if not records:
        print("no finished test jobs in those runs", file=sys.stderr)
        return 1
    print(render(records, args.flag_ratio, args.min_group))
    return 0


if __name__ == "__main__":
    sys.exit(main())
