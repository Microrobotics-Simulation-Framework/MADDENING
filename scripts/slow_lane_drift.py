#!/usr/bin/env python3
"""Warn when a slow-lane shard ran far past what the weights table predicted.

``slow-tests.yml`` deals the suite onto its shards by the measured seconds of
``tests/slow_lane_weights.json`` (``tests/_sharding.py``).  A table goes
stale without a sound: a file added since is not in it and goes by the hash
of its path, a file that grew keeps its old weight, and the shard that gets
them runs long.  On 2026-10-11 shard 10 ran 95 to 153 minutes against 63
predicted, and nobody knew until a run timed out.

This reads one shard's JUnit XML and the table, adds up what the table
predicts for that shard, and when the measured time exceeds it by more than
``--margin`` (default 50%) *and* by at least ``--min-excess`` seconds
(default 300), prints a GitHub warning that names

* the files the table has not measured (not in it, at ``min_seconds`` or
  more here), heaviest first, and
* the files it has, that took more than twice their weight and at least a
  minute more,

so the refresh -- ``scripts/slow_lane_weights.py``, one command -- is
prompted by the run and not noticed by the next timeout.  A run is the
shard's own, on one runner: runners differ by a factor of up to 1.6 between
runs (the reason for a margin of half), so a warning is a prompt to look
at the named files, not a verdict on the runner.

It reports and exits 0.  Exit 2: the report or the table cannot be read.

Usage::

    python scripts/slow_lane_drift.py test-results-shard3.xml --shard 3/10:weighted
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests import _sharding  # noqa: E402

#: How many files a warning names of each kind; the summary lists them all.
NAMED = 8
#: A listed file counts as grown past this multiple of its weight, and by this many seconds.
GROWN_FACTOR = 2.0
GROWN_SECONDS = 60


def measured_per_file(report: Path) -> dict[str, float]:
    """``{test file: seconds}`` from one pytest JUnit XML (setup, call and teardown of every test)."""
    try:
        root = ET.parse(report).getroot()
    except (OSError, ET.ParseError) as exc:
        raise SystemExit(f"slow_lane_drift: {report}: cannot read the report ({exc})") from None
    seconds: dict[str, float] = defaultdict(float)
    for case in root.iter("testcase"):
        path = case.get("file") or case.get("classname", "").replace(".", "/") + ".py"
        seconds[path] += float(case.get("time") or 0.0)
    if not seconds:
        raise SystemExit(f"slow_lane_drift: {report}: no test cases")
    return dict(seconds)


def assess(measured: dict[str, float], weights: dict[str, int], dealt: dict[str, int], shard: int,
           min_seconds: int) -> dict:
    """What the table predicted for ``shard`` against what ran; pure."""
    predicted = float(sum(w for path, w in weights.items() if dealt[path] == shard))
    total = sum(measured.values())
    unmeasured = sorted(((p, s) for p, s in measured.items() if p not in weights and s >= min_seconds),
                        key=lambda kv: (-kv[1], kv[0]))
    grown = sorted(((p, weights[p], s) for p, s in measured.items()
                    if p in weights and s > GROWN_FACTOR * weights[p] and s - weights[p] >= GROWN_SECONDS),
                   key=lambda t: (-(t[2] - t[1]), t[0]))
    return {"predicted": predicted, "measured": total, "unmeasured": unmeasured, "grown": grown}


def drifted(result: dict, margin: float, min_excess: float) -> bool:
    return (result["measured"] > result["predicted"] * (1 + margin)
            and result["measured"] - result["predicted"] >= min_excess)


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min"


def render(result: dict, shard_spec: str, margin: float) -> tuple[str, str]:
    """``(one-line summary, markdown)``."""
    pred, meas = result["predicted"], result["measured"]
    line = (f"shard {shard_spec}: {_minutes(meas)} of test time against {_minutes(pred)} predicted "
            f"({(meas / pred - 1) * 100:+.0f}%)" if pred else
            f"shard {shard_spec}: {_minutes(meas)} of test time; the table predicts nothing for it")
    md = [f"### Slow-lane weights against this run, shard {shard_spec}", "", line + ".", ""]
    if result["unmeasured"]:
        md += ["Not in the weights table (placed by the hash of their path), heaviest first:", "",
               "| file | seconds here |", "|---|---|"]
        md += [f"| `{p}` | {s:.0f} |" for p, s in result["unmeasured"]]
        md.append("")
    if result["grown"]:
        md += ["In the table, and more than twice their weight here:", "",
               "| file | weight | seconds here |", "|---|---|---|"]
        md += [f"| `{p}` | {w} | {s:.0f} |" for p, w, s in result["grown"]]
        md.append("")
    md += [f"The warning fires past +{margin * 100:.0f}%.  Refresh the table with "
           "`scripts/slow_lane_weights.py` (see `docs/developer_guide/testing_standards.md`, "
           '"Sharded lanes").', ""]
    return line, "\n".join(md)


def warning(result: dict, shard_spec: str) -> str:
    named = [f"{p} ({s / 60:.1f} min, not in the table)" for p, s in result["unmeasured"][:NAMED]]
    named += [f"{p} ({s / 60:.1f} min against {w / 60:.1f})" for p, w, s in result["grown"][:NAMED]]
    pred, meas = result["predicted"], result["measured"]
    files = "; ".join(named) if named else "no file stands out: look at the runner (speeds differ by up to 1.6x)"
    return ("::warning title=Slow-lane weights are stale::" + (
        f"shard {shard_spec} ran {meas / 60:.0f} min against {pred / 60:.0f} predicted. Files the "
        f"weights table has not measured, or has far too low: {files}. Refresh tests/slow_lane_weights.json "
        "with scripts/slow_lane_weights.py.").replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report", type=Path, help="the shard's pytest --junitxml file")
    ap.add_argument("--shard", required=True, metavar="i/N:weighted",
                    help="the shard spec the job ran (MADDENING_TEST_SHARD)")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="warn when measured exceeds predicted by this share (default 0.5)")
    ap.add_argument("--min-excess", type=float, default=300.0,
                    help="... and by at least this many seconds (default 300)")
    ap.add_argument("--markdown", type=Path,
                    default=Path(os.environ["GITHUB_STEP_SUMMARY"]) if os.environ.get("GITHUB_STEP_SUMMARY") else None)
    args = ap.parse_args(argv)
    try:
        shard, count, plan = _sharding.parse(args.shard)
    except Exception as exc:  # pytest.UsageError: a malformed spec
        print(f"slow_lane_drift: {exc}", file=sys.stderr)
        return 2
    # `parse` says True, or the suffix ("weighted"), for the slow lane's split;
    # this reads the slow lane's table and nothing else.
    if plan not in (True, "weighted"):
        print(f"slow_lane_drift: {args.shard!r} is not the slow lane's ':weighted' split", file=sys.stderr)
        return 2
    try:
        weights = _sharding.load_weights(_sharding.WEIGHTS_FILE)
    except Exception as exc:
        print(f"slow_lane_drift: {exc}", file=sys.stderr)
        return 2
    min_seconds = json.loads(_sharding.WEIGHTS_FILE.read_text(encoding="utf-8")).get("min_seconds", 10)
    result = assess(measured_per_file(args.report), weights, _sharding.deal(weights, count), shard, min_seconds)
    line, md = render(result, args.shard, args.margin)
    print(line)
    if args.markdown:
        with open(args.markdown, "a", encoding="utf-8") as fh:
            fh.write(md + "\n")
    if drifted(result, args.margin, args.min_excess):
        print(warning(result, args.shard))
    return 0


if __name__ == "__main__":
    sys.exit(main())
