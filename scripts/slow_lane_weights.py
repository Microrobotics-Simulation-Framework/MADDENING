#!/usr/bin/env python3
"""Write a lane's per-file weights from the JUnit reports of its runs.

Both CI lanes split the suite across runners with the weighted split of
``tests/_sharding.py``, which deals the files of a table by their measured
seconds: ``tests/slow_lane_weights.json`` for ``slow-tests.yml``
(``--lane slow``, the default) and ``tests/per_push_weights.json`` for
``ci.yml`` (``--lane per-push``).  This script writes a table from the
lane's own artifacts and prints the minutes per shard it predicts, so a
rebalance is one download and one command::

    gh run download <run id> --dir /tmp/slow --pattern 'slow-durations-*'
    python scripts/slow_lane_weights.py --run <run id>=/tmp/slow --write

    # the same, without downloading by hand: the lane's last three runs
    python scripts/slow_lane_weights.py --latest --write
    python scripts/slow_lane_weights.py --lane per-push --latest 5 --write

    # the per-push lane from runs you downloaded: several COLD push runs
    for id in <id> <id> <id>; do
      gh run download $id --dir /tmp/push/$id --pattern 'test-durations-*'; done
    python scripts/slow_lane_weights.py --lane per-push --write
        --run <id>=/tmp/push/<id> --run <id>=/tmp/push/<id>   (one command)

Each ``--run ID=DIR`` names a directory laid out as ``gh run download``
leaves it: one sub-directory per artifact
(``slow-durations-py3.12-jax0.10.2-shard1of4``), holding that shard's
JUnit XML.  The artifact's name without its ``-shardKofN`` suffix is the
*lane*.  Give several runs to cover a shard that one of them lost (a
timed-out shard writes no report).  For a file and a lane, with
``--aggregate median`` (the default) the median of the runs that measured
it; with ``--aggregate last`` the last run given that measured it.
``--latest [N]`` (three by default) downloads the lane's last N runs that
left artifacts, or as many as there are.  Use only runs whose shards all
ran cold (a push to ``main`` or ``release/**``, or a pull request run with
``[cold-ci]``): a warm shard's compile times are the cache's.  A per-push
shard whose ``cache-mode-shardK.txt`` says warm is left out, with a
message.

Speed of the runner.  The jobs of one lane run on whatever runner the
pool gives them, and one runner takes 1.6 to 1.9 times as long as another
over the same files.  A table taken as measured weighs each file by the
luck of its runner: on 2026-10-11 the heaviest file was weighed on the two
fastest runners of sixty, at 30 minutes against 47 on a median one, and
the shard it was dealt to ran 95 to 108 minutes against 68 predicted.  So
the script scales every shard's report before it takes the median across
runs, by one measure of its runner's speed: the seconds pytest took to
collect the suite, the same work on every shard before any test.
``tests/_sharding.py`` writes it into the JUnit report as the testsuite
property ``runner_collect_seconds``, beside ``runner_cpu``, and the report
is the only place this script reads either from.  Each report's seconds
are multiplied by ``(median / its own) ** 1.5`` (``_sharding.runner_scale``;
the median is over every report given): the table is the time on a runner
of median speed, and its ``runner_speed`` says so.

The CPU model is recorded, not used, except for a report that has no
collection time: that one is divided by its CPU model's speed factor,
estimated from the files it shares with the scaled reports (the median,
over those files, of the ratio of the medians; when no report has a
collection time, relative to the model with the most shards).  A report
with neither, or whose model shares too few files, is taken as measured.
The script says how many reports went each way, and ``--no-normalise``
takes every report as measured.  A report written before the property
existed (before 2026-10-14) can be given its collection time from the job
log: the seconds from pytest's "test session starts" line to its
"collected N items" line, which agree with the property to a few
milliseconds.

A file's weight is its slower lane's total (setup, call and teardown of
every test in it), in whole seconds.  Files under ``--min-seconds`` in
every lane are left out -- the split places those by the hash of their
path -- and so is a measured file that no longer exists.

Without ``--write`` the table is printed nowhere and only the prediction
is shown, which also answers "what would N shards give?" (``--shards``).
The prediction is test time only: a job adds about three minutes of
checkout and installation.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests import _sharding  # noqa: E402

_SHARD_SUFFIX = re.compile(r"-shard\d+of\d+$")
AGGREGATES = ("last", "median")
UNKNOWN_CPU = "unknown"
#: The reference the CPU-model fallback is estimated against when some
#: reports have a collection time: those reports, scaled.  Not a CPU name.
MEDIAN_RUNNER = "(a runner of median speed)"
#: A speed factor needs this many files measured on both models.
MIN_FILES_FOR_FACTOR = 3
#: Files lighter than this on the reference model are too noisy to set a factor.
FACTOR_MIN_SECONDS = 5.0
#: Which workflow, and which artifacts, ``--latest`` reads for each lane.
LATEST = {
    "slow": dict(workflow="slow-tests.yml", pattern="slow-durations-*", events=None),
    # Only a push runs cold; a pull request's shards are warm unless `[cold-ci]`.
    "per-push": dict(workflow="ci.yml", pattern="test-durations-*", events=("push",)),
}

#: What each lane's table is called, where it lives, and how many shards it deals.
LANES = {
    "slow": dict(file=_sharding.WEIGHTS_FILE, shards=_sharding.SLOW_LANE_SHARDS,
                 name="Seconds per test file on the slow lane",
                 measured="Each file's slower lane, setup + call + teardown of every test in it, on "
                          "GitHub ubuntu-24.04 runners with no compilation cache."),
    "per-push": dict(file=_sharding.PER_PUSH_WEIGHTS_FILE, shards=_sharding.PER_PUSH_SHARDS,
                     name="Seconds per test file on the per-push lane",
                     measured="Each file's slower JAX lane, setup + call + teardown of every test in it, on "
                              "GitHub ubuntu-24.04 runners in cold runs (no compilation cache)."),
}


@dataclass
class Unit:
    """One shard's report of one run: its lane, its runner's fields and its seconds per file."""
    lane: str
    artifact: str
    cpu: str
    files: dict[str, float]
    cache: str | None = None
    run: str = ""
    collect: float | None = None


def _artifact_cache_mode(directory: Path) -> str | None:
    for mode_file in sorted(directory.glob("cache-mode-shard*.txt")):
        return mode_file.read_text(encoding="utf-8").strip() or None
    return None


def read_units(directory: Path, run: str = "") -> list[Unit]:
    """Every shard report of one downloaded run, with the runner fields the report carries."""
    reports = sorted(directory.glob("*/*.xml"))
    if not reports:
        raise SystemExit(f"{directory}: no <artifact>/<report>.xml below it")
    units = []
    for report in reports:
        root = ET.parse(report).getroot()
        files: dict[str, float] = defaultdict(float)
        for case in root.iter("testcase"):
            path = case.get("file") or case.get("classname", "").replace(".", "/") + ".py"
            files[path] += float(case.get("time") or 0.0)
        units.append(Unit(lane=_SHARD_SUFFIX.sub("", report.parent.name), artifact=report.parent.name,
                          cpu=_sharding.runner_fields(root).get(_sharding.CPU_PROPERTY, UNKNOWN_CPU),
                          files=dict(files), cache=_artifact_cache_mode(report.parent), run=run,
                          collect=_sharding.collect_seconds(root)))
    return units


def lanes_of(units: list[Unit]) -> dict[str, dict[str, float]]:
    """``{lane: {test file: seconds}}`` of one run's shard reports."""
    lanes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for unit in units:
        for path, seconds in unit.files.items():
            lanes[unit.lane][path] += seconds
    return {lane: dict(files) for lane, files in lanes.items()}


def read_run(directory: Path) -> dict[str, dict[str, float]]:
    """``{lane: {test file: seconds}}`` from one downloaded run, as measured."""
    return lanes_of(read_units(directory))


def speed_factors(units: list[Unit], reference: str | None = None,
                  min_seconds: float = FACTOR_MIN_SECONDS) -> tuple[dict[str, float], str | None, list[str]]:
    """``({cpu: factor}, reference, notes)``: how much slower each model ran than the reference.

    The CPU-model fallback of :func:`runner_scales`, for reports with no
    collection time.  A factor of 1.5 means the model took 1.5 times the
    reference's seconds on the same files.  For every (lane, file) measured
    on both a model and the reference, the ratio of the median seconds on
    each; the model's factor is the median of those ratios.  A model with
    fewer than ``MIN_FILES_FOR_FACTOR`` such files, and the ``unknown``
    model, stay at 1.0 and are named in ``notes``.  ``reference`` is the
    exact name of one of the units' models; by default the model with the
    most shards.
    """
    known = [u for u in units if u.cpu != UNKNOWN_CPU]
    notes: list[str] = []
    count: dict[str, int] = defaultdict(int)
    for u in known:
        count[u.cpu] += 1
    if len(count) < 2:
        if len(count) == 1:
            notes.append(f"every shard with a known CPU ran on {next(iter(count))}: nothing to normalise")
        return {c: 1.0 for c in count}, (next(iter(count)) if count else None), notes
    ref = reference if reference in count else max(sorted(count), key=lambda c: count[c])
    seen: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for u in known:
        for path, seconds in u.files.items():
            seen[(u.lane, path)][u.cpu].append(seconds)
    factors = {ref: 1.0}
    for cpu in sorted(count):
        if cpu == ref:
            continue
        ratios = []
        for by_cpu in seen.values():
            if cpu in by_cpu and ref in by_cpu:
                base = statistics.median(by_cpu[ref])
                if base >= min_seconds:
                    ratios.append(statistics.median(by_cpu[cpu]) / base)
        if len(ratios) >= MIN_FILES_FOR_FACTOR:
            factors[cpu] = statistics.median(ratios)
        else:
            factors[cpu] = 1.0
            notes.append(f"{cpu}: only {len(ratios)} file(s) also ran on {ref} "
                         f"(need {MIN_FILES_FOR_FACTOR}); its shards are not scaled")
    return factors, ref, notes


def runner_scales(units: list[Unit], exponent: float = _sharding.SPEED_EXPONENT,
                  ) -> tuple[list[float], dict, list[str]]:
    """``(one scale per unit, the table's runner_speed record, notes)``: one rule per report.

    * A report with a collection time is scaled to a runner that collects
      in the median of them: ``_sharding.runner_scale(its own, median)``.
    * One without, that names its CPU, is divided by that model's factor
      (:func:`speed_factors`) against the scaled reports, or, when no report
      has a collection time, against the model with the most shards.
    * One with neither stays as measured.
    """
    timed = [u.collect for u in units if u.collect]
    reference = statistics.median(timed) if timed else None
    scales = [_sharding.runner_scale(u.collect, reference, exponent) if u.collect else 1.0 for u in units]
    fallback = [i for i, u in enumerate(units) if not u.collect and u.cpu != UNKNOWN_CPU]
    factors: dict[str, float] = {}
    cpu_reference, notes = None, []
    if fallback:
        pool = [Unit(u.lane, u.artifact, MEDIAN_RUNNER, {p: s * k for p, s in u.files.items()})
                for u, k in zip(units, scales) if u.collect]
        pool += [units[i] for i in fallback]
        factors, cpu_reference, notes = speed_factors(pool, MEDIAN_RUNNER if timed else None)
        factors.pop(MEDIAN_RUNNER, None)
        for i in fallback:
            scales[i] = 1.0 / factors.get(units[i].cpu, 1.0)
    record = {
        "property": _sharding.COLLECT_SECONDS_PROPERTY,
        "reference_seconds": None if reference is None else round(reference, 2),
        "exponent": exponent if reference is not None else 0,
        "reports_scaled": len(timed),
        "reports_by_cpu_model": len(fallback),
        "cpu_reference": cpu_reference if fallback else None,
        "cpu_factors": {cpu: round(f, 3) for cpu, f in sorted(factors.items())},
        "reports_as_measured": len(units) - len(timed) - len(fallback),
    }
    return scales, record, notes


def normalise(units: list[Unit], scales: list[float]) -> list[Unit]:
    """The same units with every file's seconds multiplied by the unit's scale."""
    return [Unit(u.lane, u.artifact, u.cpu, {p: s * k for p, s in u.files.items()}, u.cache, u.run, u.collect)
            for u, k in zip(units, scales)]


def as_measured(units: list[Unit]) -> dict:
    """The ``runner_speed`` record of a table that scaled nothing (``--no-normalise``)."""
    return {"property": _sharding.COLLECT_SECONDS_PROPERTY, "reference_seconds": None, "exponent": 0,
            "reports_scaled": 0, "reports_by_cpu_model": 0, "cpu_reference": None, "cpu_factors": {},
            "reports_as_measured": len(units)}


def fetch_latest(lane: str, count: int, destination: Path, run=subprocess.run) -> list[tuple[str, str, Path]]:
    """Download the lane's last ``count`` runs that left artifacts: ``[(run id, commit, directory)]``.

    Newest first among the runs GitHub lists; a run with no artifact left
    (expired, or none uploaded) is passed over.  ``run`` is
    ``subprocess.run``, replaceable by a test.
    """
    info = LATEST[lane]
    listing = run(["gh", "run", "list", "--workflow", info["workflow"], "--status", "completed",
                   "--limit", str(max(30, count * 8)), "--json", "databaseId,headSha,event,conclusion"],
                  capture_output=True, text=True, check=True)
    found: list[tuple[str, str, Path]] = []
    for entry in json.loads(listing.stdout):
        if len(found) >= count:
            break
        if entry.get("conclusion") not in ("success", "failure"):
            continue
        if info["events"] and entry.get("event") not in info["events"]:
            continue
        run_id = str(entry["databaseId"])
        target = destination / run_id
        got = run(["gh", "run", "download", run_id, "--dir", str(target), "--pattern", info["pattern"]],
                  capture_output=True, text=True)
        if got.returncode == 0 and any(target.glob("*/*.xml")):
            found.append((run_id, str(entry.get("headSha", "unknown"))[:12], target))
        else:
            print(f"  run {run_id}: no {info['pattern']} artifacts left, passed over", file=sys.stderr)
    return found


def merge(runs: list[dict[str, dict[str, float]]], aggregate: str = "last") -> dict[str, dict[str, float]]:
    """One ``{lane: {file: seconds}}``.

    ``"last"``: a later run's measurement of a file wins.  ``"median"``: the
    median over the runs that measured it (the mean of the middle two when
    that is an even number).
    """
    if aggregate not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}, not {aggregate!r}")
    merged: dict[str, dict[str, float]] = defaultdict(dict)
    if aggregate == "last":
        for run in runs:
            for lane, files in run.items():
                merged[lane].update(files)
        return dict(merged)
    seen: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for lane, files in run.items():
            for path, seconds in files.items():
                seen[lane][path].append(seconds)
    for lane, files in seen.items():
        merged[lane] = {path: statistics.median(times) for path, times in files.items()}
    return dict(merged)


def weights(lanes: dict[str, dict[str, float]], min_seconds: int, root: Path = REPO_ROOT,
            ) -> tuple[dict[str, int], list[str]]:
    """``({file: whole seconds on its slower lane}, [measured files that are gone])``."""
    slowest: dict[str, float] = defaultdict(float)
    for files in lanes.values():
        for path, seconds in files.items():
            slowest[path] = max(slowest[path], seconds)
    gone = sorted(p for p in slowest if not (root / p).is_file())
    table = {p: round(s) for p, s in sorted(slowest.items())
             if round(s) >= min_seconds and p not in gone}
    return table, gone


def predict(files: dict[str, float], shard_of, n: int) -> list[float]:
    """Minutes of measured test time on each of ``n`` shards under ``shard_of(path)``."""
    load = [0.0] * n
    for path, seconds in files.items():
        load[shard_of(path) - 1] += seconds
    return [s / 60.0 for s in load]


def _row(label: str, minutes: list[float]) -> str:
    return f"  {label:<34}" + " ".join(f"{m:6.1f}" for m in minutes) + f"   worst {max(minutes):.1f}"


def _speed_summary(units: list[Unit], record: dict, scales: list[float]) -> tuple[list[str], str]:
    """``(lines to print, the clause for the table's "measured")``."""
    lines, clause = [], ""
    timed = [(u.collect, k) for u, k in zip(units, scales) if u.collect]
    if timed:
        seconds = [c for c, _k in timed]
        factors = [k for _c, k in timed]
        lines.append(f"runner speed: {len(timed)} reports scaled to a runner that collects in "
                     f"{record['reference_seconds']:.1f} s (the median; theirs {min(seconds):.1f} to "
                     f"{max(seconds):.1f} s, so x{min(factors):.2f} to x{max(factors):.2f}, "
                     f"exponent {record['exponent']:g})")
        clause = (" Each shard's seconds are scaled to a runner of median speed by the time it took to "
                  "collect the suite (runner_speed).")
    if record["reports_by_cpu_model"]:
        against = ("the scaled reports" if record["cpu_reference"] == MEDIAN_RUNNER
                   else record["cpu_reference"])
        lines.append(f"runner speed: {record['reports_by_cpu_model']} reports without "
                     f"{_sharding.COLLECT_SECONDS_PROPERTY} divided by their CPU model's factor against "
                     f"{against}: " + ", ".join(f"{cpu} {f:.2f}" for cpu, f in record["cpu_factors"].items()))
        clause += (f" {record['reports_by_cpu_model']} report(s) without a collection time are divided by "
                   "their CPU model's speed factor (runner_speed.cpu_factors).")
    if record["reports_as_measured"]:
        lines.append(f"runner speed: {record['reports_as_measured']} reports taken as measured (no "
                     f"{_sharding.COLLECT_SECONDS_PROPERTY} and no {_sharding.CPU_PROPERTY} in the report)")
    return lines, clause


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lane", choices=sorted(LANES), default="slow",
                    help="which lane's table: sets the default --shards and --output (default slow)")
    ap.add_argument("--run", action="append", default=[], metavar="ID=DIR",
                    help="a run id and the directory `gh run download` filled; repeatable (with --aggregate last, later wins)")
    ap.add_argument("--latest", nargs="?", type=int, const=3, default=None, metavar="N",
                    help="instead of --run: download the lane's last N runs that left artifacts "
                         "(default 3; as many as exist if fewer)")
    ap.add_argument("--download-dir", type=Path, default=None,
                    help="where --latest puts the runs (default: a new temporary directory)")
    ap.add_argument("--aggregate", choices=AGGREGATES, default="median",
                    help="a file's seconds across runs: the median, or the last run's (default median)")
    ap.add_argument("--speed-exponent", type=float, default=_sharding.SPEED_EXPONENT,
                    help="scale each report by (median collection time / its own) to this power "
                         f"(default {_sharding.SPEED_EXPONENT})")
    ap.add_argument("--no-normalise", action="store_true",
                    help="take every report as measured: no scaling by the runner's speed")
    ap.add_argument("--commit", action="append", default=[], metavar="ID=SHA",
                    help="the commit a run measured, recorded in the table")
    ap.add_argument("--min-seconds", type=int, default=10,
                    help="leave out files lighter than this on every lane (default 10)")
    ap.add_argument("--shards", type=int, default=None,
                    help="shard count to predict for (default: the --lane's)")
    ap.add_argument("--top", type=int, default=15, help="heaviest files to list (default 15)")
    ap.add_argument("--write", action="store_true", help="write the lane's table")
    ap.add_argument("--output", type=Path, default=None, help="where (default: the --lane's table)")
    args = ap.parse_args(argv)
    if args.speed_exponent <= 0:
        ap.error("--speed-exponent must be positive (--no-normalise takes the reports as measured)")
    lane_info = LANES[args.lane]
    n = args.shards if args.shards is not None else lane_info["shards"]
    output = args.output if args.output is not None else lane_info["file"]

    commits = dict(c.split("=", 1) for c in args.commit)
    specs: list[tuple[str, Path]] = []
    if args.latest is not None:
        if args.run:
            ap.error("give --run or --latest, not both")
        if args.latest < 1:
            ap.error("--latest needs at least one run")
        destination = args.download_dir or Path(tempfile.mkdtemp(prefix=f"weights-{args.lane}-"))
        found = fetch_latest(args.lane, args.latest, destination)
        if not found:
            raise SystemExit(f"no run of {LATEST[args.lane]['workflow']} has {LATEST[args.lane]['pattern']} artifacts")
        print(f"downloaded {len(found)} of the last runs into {destination}: "
              + ", ".join(run_id for run_id, _sha, _dir in found))
        for run_id, sha, directory in reversed(found):          # oldest first, so "last" is the newest
            commits.setdefault(run_id, sha)
            specs.append((run_id, directory))
    else:
        for spec in args.run:
            run_id, _, directory = spec.partition("=")
            if not directory:
                ap.error(f"--run {spec!r}: expected ID=DIR")
            specs.append((run_id, Path(directory)))
        if not specs:
            ap.error("give --run ID=DIR (one or more) or --latest")

    run_units: list[list[Unit]] = []
    for run_id, directory in specs:
        units = read_units(directory, run_id)
        if args.lane == "per-push":
            warm = [u for u in units if u.cache == "warm"]
            for u in warm:
                print(f"  run {run_id}: {u.artifact} ran with a warm compilation cache, left out")
            units = [u for u in units if u.cache != "warm"]
        run_units.append(units)
    all_units = [u for units in run_units for u in units]
    if not all_units:
        raise SystemExit("no shard report left to read: every one ran with a warm compilation cache "
                         "(the per-push table is made from cold runs: pushes to main or release/**, "
                         "or a pull request with [cold-ci])")
    if args.no_normalise:
        speed = as_measured(all_units)
        print("runner speed: every report taken as measured (--no-normalise)")
        clause = ""
    else:
        scales, speed, notes = runner_scales(all_units, args.speed_exponent)
        lines, clause = _speed_summary(all_units, speed, scales)
        for line in lines:
            print(line)
        for note in notes:
            print(f"  note: {note}")
        it = iter(scales)
        run_units = [normalise(units, [next(it) for _ in units]) for units in run_units]
    runs = [lanes_of(units) for units in run_units]
    sources = [{"run": run_id, "commit": commits.get(run_id, "unknown"),
                "artifacts": sorted({u.artifact for u in units})}
               for (run_id, _directory), units in zip(specs, run_units)]
    lanes = merge(runs, args.aggregate)
    table, gone = weights(lanes, args.min_seconds)
    dealt = _sharding.deal(table, n)

    print(f"{sum(len(f) for f in lanes.values())} file measurements on {len(lanes)} lanes; "
          f"{len(table)} files at or over {args.min_seconds} s in the table")
    for path in gone:
        print(f"  measured but not a file in the tree, left out: {path}")
    print(f"\nheaviest files (minutes per lane: {', '.join(sorted(lanes))})")
    for path in sorted(table, key=lambda p: (-table[p], p))[:args.top]:
        per_lane = " ".join(f"{lanes[lane].get(path, 0.0) / 60:6.1f}" for lane in sorted(lanes))
        print(f"  {per_lane}  {path}")
    print("\npredicted minutes of test time per shard")
    for lane in sorted(lanes):
        files = {p: s for p, s in lanes[lane].items() if p not in gone}
        print(f" {lane}")
        print(_row(f"by hash, {n} shards", predict(files, lambda p: _sharding.shard_of(p, n), n)))
        print(_row(f"weighted, {n} shards",
                   predict(files, lambda p: _sharding.weighted_shard_of(p, n, dealt), n)))

    if args.write:
        document = {
            "what": f"{lane_info['name']}, for the weighted split of tests/_sharding.py. "
                    "Written by scripts/slow_lane_weights.py; do not edit by hand.",
            "measured": f"{lane_info['measured']}" + clause + " "
                        + ("Where several runs measured a file, the median of them."
                           if args.aggregate == "median" else
                           "Where two runs measured a file, the later."),
            "runs": sources,
            "runner_speed": speed,
            "min_seconds": args.min_seconds,
            "seconds": table,
        }
        output.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
