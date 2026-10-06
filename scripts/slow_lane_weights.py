#!/usr/bin/env python3
"""Write the slow lane's per-file weights from the JUnit reports of a run.

``slow-tests.yml`` splits the suite across runners with the weighted split
of ``tests/_sharding.py``, which deals the files of
``tests/slow_lane_weights.json`` by their measured seconds.  This script
writes that table from the lane's own artifacts and prints the minutes per
shard it predicts, so a rebalance is one download and one command::

    gh run download <run id> --dir /tmp/slow --pattern 'slow-durations-*'
    python scripts/slow_lane_weights.py --run <run id>=/tmp/slow --write

Each ``--run ID=DIR`` names a directory laid out as ``gh run download``
leaves it: one sub-directory per artifact
(``slow-durations-py3.12-jax0.10.2-shard1of4``), holding that shard's
JUnit XML.  The artifact's name without its ``-shardKofN`` suffix is the
*lane*.  Give several runs to cover a shard that one of them lost (a
timed-out shard writes no report): for a file and a lane, the last run
given that measured it is the one used.

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
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests import _sharding  # noqa: E402

_SHARD_SUFFIX = re.compile(r"-shard\d+of\d+$")


def read_run(directory: Path) -> dict[str, dict[str, float]]:
    """``{lane: {test file: seconds}}`` from one downloaded run."""
    lanes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    reports = sorted(directory.glob("*/*.xml"))
    if not reports:
        raise SystemExit(f"{directory}: no <artifact>/<report>.xml below it")
    for report in reports:
        lane = _SHARD_SUFFIX.sub("", report.parent.name)
        for case in ET.parse(report).getroot().iter("testcase"):
            path = case.get("file") or case.get("classname", "").replace(".", "/") + ".py"
            lanes[lane][path] += float(case.get("time") or 0.0)
    return {lane: dict(files) for lane, files in lanes.items()}


def merge(runs: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """One ``{lane: {file: seconds}}``; a later run's measurement of a file wins."""
    merged: dict[str, dict[str, float]] = defaultdict(dict)
    for run in runs:
        for lane, files in run.items():
            merged[lane].update(files)
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
    ap.add_argument("--run", action="append", required=True, metavar="ID=DIR",
                    help="a run id and the directory `gh run download` filled; repeatable, later wins")
    ap.add_argument("--commit", action="append", default=[], metavar="ID=SHA",
                    help="the commit a run measured, recorded in the table")
    ap.add_argument("--min-seconds", type=int, default=10,
                    help="leave out files lighter than this on every lane (default 10)")
    ap.add_argument("--shards", type=int, default=_sharding.SLOW_LANE_SHARDS,
                    help="shard count to predict for (default: the slow lane's)")
    ap.add_argument("--top", type=int, default=15, help="heaviest files to list (default 15)")
    ap.add_argument("--write", action="store_true", help=f"write {_sharding.WEIGHTS_FILE.name}")
    ap.add_argument("--output", type=Path, default=_sharding.WEIGHTS_FILE)
    args = ap.parse_args(argv)

    commits = dict(c.split("=", 1) for c in args.commit)
    runs, sources = [], []
    for spec in args.run:
        run_id, _, directory = spec.partition("=")
        if not directory:
            ap.error(f"--run {spec!r}: expected ID=DIR")
        runs.append(read_run(Path(directory)))
        sources.append({
            "run": run_id, "commit": commits.get(run_id, "unknown"),
            "artifacts": sorted(p.name for p in Path(directory).iterdir() if any(p.glob("*.xml")))})
    lanes = merge(runs)
    table, gone = weights(lanes, args.min_seconds)
    n = args.shards
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
        print(_row(f"by hash and pins, {_sharding.PINS_FOR} shards",
                   predict(files, lambda p: _sharding.shard_of(p, _sharding.PINS_FOR), _sharding.PINS_FOR)))
        print(_row(f"weighted, {n} shards",
                   predict(files, lambda p: _sharding.weighted_shard_of(p, n, dealt), n)))

    if args.write:
        document = {
            "what": "Seconds per test file on the slow lane, for the weighted split of tests/_sharding.py. "
                    "Written by scripts/slow_lane_weights.py; do not edit by hand.",
            "measured": "Each file's slower lane, setup + call + teardown of every test in it, on "
                        "ubuntu-latest with no compilation cache. Where two runs measured a file, the later.",
            "runs": sources,
            "min_seconds": args.min_seconds,
            "seconds": table,
        }
        args.output.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
