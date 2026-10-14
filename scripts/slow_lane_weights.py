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
it, which averages out the speed of the runners each run drew (a factor of
up to 1.6 between two runs of one file); with ``--aggregate last`` the last
run given that measured it.  ``--latest [N]`` (three by default) downloads
the lane's last N runs that left artifacts, or as many as there are.  Use
only runs whose shards all ran cold (a push to ``main`` or ``release/**``, or
a pull request run with ``[cold-ci]``): a warm shard's compile times are
the cache's.  A per-push shard whose ``cache-mode-shardK.txt`` says warm is
left out, with a message.

Speed of the runner.  The jobs of one lane run on whatever CPU the pool
gives them, and generations differ by about 1.5x, so a run that drew slow
runners would raise every file's weight.  Each shard's artifact carries a
``runner-shardK.json`` sidecar (``scripts/ci_runner_health.py``) naming its
CPU model.  From the files that ran on several models the script estimates
each model's speed factor relative to a reference model (the ratio of the
medians, over those files, the median of the ratios), divides every
measurement by its model's factor, and only then takes the median across
runs: the table is the time on the reference model.  The reference is the
model with the most shards (``--reference SUBSTRING`` names another).  A
shard with no sidecar (runs before the sidecar existed) is not scaled, and
a model sharing no file with the reference stays at 1.0, both said in the
output.  ``--no-normalise`` turns it off.

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
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests import _sharding  # noqa: E402

_SHARD_SUFFIX = re.compile(r"-shard\d+of\d+$")
AGGREGATES = ("last", "median")
UNKNOWN_CPU = "unknown"
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
    """One shard's report of one run: the lane it belongs to, its CPU and its seconds per file."""
    lane: str
    artifact: str
    cpu: str
    files: dict[str, float]
    cache: str | None = None
    run: str = ""


def _artifact_cpu(directory: Path) -> str:
    """The CPU model the sidecar of a downloaded artifact names, or ``unknown``."""
    for sidecar in sorted(directory.glob("runner-*.json")):
        try:
            model = json.loads(sidecar.read_text(encoding="utf-8")).get("cpu_model")
        except (OSError, ValueError):
            continue
        if isinstance(model, str) and model:
            return model
    return UNKNOWN_CPU


def _artifact_cache_mode(directory: Path) -> str | None:
    for mode_file in sorted(directory.glob("cache-mode-shard*.txt")):
        return mode_file.read_text(encoding="utf-8").strip() or None
    return None


def read_units(directory: Path, run: str = "") -> list[Unit]:
    """Every shard report of one downloaded run, with the CPU its sidecar names."""
    reports = sorted(directory.glob("*/*.xml"))
    if not reports:
        raise SystemExit(f"{directory}: no <artifact>/<report>.xml below it")
    units = []
    for report in reports:
        files: dict[str, float] = defaultdict(float)
        for case in ET.parse(report).getroot().iter("testcase"):
            path = case.get("file") or case.get("classname", "").replace(".", "/") + ".py"
            files[path] += float(case.get("time") or 0.0)
        units.append(Unit(lane=_SHARD_SUFFIX.sub("", report.parent.name), artifact=report.parent.name,
                          cpu=_artifact_cpu(report.parent), files=dict(files),
                          cache=_artifact_cache_mode(report.parent), run=run))
    return units


def lanes_of(units: list[Unit]) -> dict[str, dict[str, float]]:
    """``{lane: {test file: seconds}}`` of one run's shard reports."""
    lanes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for unit in units:
        for path, seconds in unit.files.items():
            lanes[unit.lane][path] += seconds
    return {lane: dict(files) for lane, files in lanes.items()}


def read_run(directory: Path) -> dict[str, dict[str, float]]:
    """``{lane: {test file: seconds}}`` from one downloaded run."""
    return lanes_of(read_units(directory))


def speed_factors(units: list[Unit], reference: str | None = None,
                  min_seconds: float = FACTOR_MIN_SECONDS) -> tuple[dict[str, float], str | None, list[str]]:
    """``({cpu: factor}, reference, notes)``: how much slower each model ran than the reference.

    A factor of 1.5 means the model took 1.5 times the reference's seconds
    on the same files.  For every (lane, file) measured on both a model and
    the reference, the ratio of the median seconds on each; the model's
    factor is the median of those ratios.  A model with fewer than
    ``MIN_FILES_FOR_FACTOR`` such files, and the ``unknown`` model, stay at
    1.0 and are named in ``notes``.  ``reference`` is a substring of a model
    name; by default the model with the most shards.
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
    if reference:
        matches = sorted(c for c in count if reference.lower() in c.lower())
        if not matches:
            raise SystemExit(f"--reference {reference!r} matches none of: {', '.join(sorted(count))}")
        ref = max(matches, key=lambda c: (count[c], c))
    else:
        ref = max(sorted(count), key=lambda c: count[c])
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


def normalise(units: list[Unit], factors: dict[str, float]) -> list[Unit]:
    """The same units with every file's seconds divided by its model's factor."""
    return [Unit(u.lane, u.artifact, u.cpu,
                 {p: s / factors.get(u.cpu, 1.0) for p, s in u.files.items()}, u.cache, u.run)
            for u in units]


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
    ap.add_argument("--reference", default=None, metavar="CPU",
                    help="a substring of the CPU model the table is normalised to "
                         "(default: the model with the most shards)")
    ap.add_argument("--no-normalise", action="store_true",
                    help="do not scale each shard's seconds by its CPU model's speed factor")
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
    clause = ""
    if not args.no_normalise:
        factors, reference, notes = speed_factors(all_units, args.reference)
        unscaled = sum(1 for u in all_units if u.cpu == UNKNOWN_CPU)
        if len(factors) > 1:
            print(f"speed factors, seconds on each CPU model relative to {reference}:")
            counts = defaultdict(int)
            for u in all_units:
                counts[u.cpu] += 1
            for cpu in sorted(factors, key=lambda c: (factors[c], c)):
                print(f"  {factors[cpu]:5.2f}  {cpu}  ({counts[cpu]} shard reports)")
            clause = (f" Each shard's seconds are divided by its CPU model's speed factor, so the table is "
                      f"the time on {reference}: " + ", ".join(
                          f"{cpu} {factors[cpu]:.2f}" for cpu in sorted(factors, key=lambda c: (factors[c], c))) + ".")
            run_units = [normalise(units, factors) for units in run_units]
        for note in notes:
            print(f"  note: {note}")
        if unscaled:
            print(f"  note: {unscaled} shard report(s) have no runner sidecar (a run from before "
                  f"scripts/ci_runner_health.py) and are not scaled")
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
            "measured": f"{lane_info['measured']} "
                        + ("Where several runs measured a file, the median of them."
                           if args.aggregate == "median" else
                           "Where two runs measured a file, the later.") + clause,
            "runs": sources,
            "min_seconds": args.min_seconds,
            "seconds": table,
        }
        output.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
