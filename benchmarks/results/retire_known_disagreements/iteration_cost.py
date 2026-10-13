#!/usr/bin/env python
"""What a tighter fixture criterion costs across the whole sweep.

``diff_sweep.py`` in
``benchmarks/results/d2_converged_returns_measured_iterate/`` answers
"did the returned states move"; this answers "what was paid for it", in
the only two currencies that mean anything on a shared machine:
iterations per step and the fraction of steps that reported
``converged=True``.  Both are read straight out of the ``replay_sweep``
JSONL records, so no graph is re-run.

An at-cap step is counted the way the profiler counts one, ``iters >=
cap - 1``: ``cond`` stops at ``i >= max_iter - 1``, so a group meeting
its criterion on the last pass the cap allows lands in this bucket too.
That overcounts slightly and in a fixed direction, which is why the
converged fraction beside it is the number to read.

Deliberately **no timings**.

Usage
-----
Replay the sweep on the unchanged tree and on one with the tightened
fixtures, then::

    python iteration_cost.py "rtol 1e-4 -> 3e-7" before.jsonl after.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Row slices worth separating.  ``l2`` is the control: the fixtures'
#: ``rtol`` is not read under that norm, so those rows must not move,
#: and a change there means the experiment leaked.
_SLICES = (
    ("all", lambda row: True),
    ("interface", lambda row: row["convergence_norm"] != "l2"),
    ("l2 (control)", lambda row: row["convergence_norm"] == "l2"),
    ("interface + iqn",
     lambda row: (row["convergence_norm"] != "l2"
                  and row["acceleration"].startswith("iqn"))),
)


def _load(path: Path) -> dict:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("error") or not record.get("ok"):
            continue
        rows[(record["fixture"], record["label"])] = record
    return rows


def _stats(rows: dict, keep) -> dict:
    iterations, converged, at_cap = [], [], 0
    for record in rows.values():
        if not keep(record):
            continue
        caps = record["caps"]
        for key, group in record["groups"].items():
            cap = caps.get(key) if isinstance(caps, dict) else None
            for used, flag in zip(group["iterations"], group["converged"]):
                iterations.append(used)
                converged.append(bool(flag))
                if cap is not None and used >= cap - 1:
                    at_cap += 1
    if not iterations:
        return {}
    return {
        "steps": len(iterations),
        "mean_iters": sum(iterations) / len(iterations),
        "conv_frac": sum(converged) / len(converged),
        "at_cap": at_cap,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("title")
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    before, after = _load(args.before), _load(args.after)
    print(f"### {args.title}   ({len(before)} / {len(after)} rows)\n")
    print("| slice | steps | iters before | iters after | change "
          "| converged before | converged after | at-cap before | at-cap after |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    summary = {}
    for name, keep in _SLICES:
        b, a = _stats(before, keep), _stats(after, keep)
        if not b or not a:
            continue
        change = (a["mean_iters"] - b["mean_iters"]) / b["mean_iters"] * 100.0
        summary[name] = {"before": b, "after": a, "pct_iterations": change}
        print(f"| `{name}` | {b['steps']} | {b['mean_iters']:.3f} "
              f"| {a['mean_iters']:.3f} | {change:+.1f}% "
              f"| {b['conv_frac']:.4f} | {a['conv_frac']:.4f} "
              f"| {b['at_cap']} | {a['at_cap']} |")
    if args.out:
        args.out.write_text(json.dumps(
            {"title": args.title, "slices": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
