#!/usr/bin/env python
"""Compare two ``replay_sweep.py`` runs: pre-fix tree against fixed tree.

    python diff_sweep.py raw/before.jsonl raw/after.jsonl \\
        --out comparison.json --markdown comparison.md

Answers the three questions D2 is blocked on:

1. Do iteration counts and converged fractions move?  The fix touches
   only which of two adjacent iterates leaves the loop, never ``cond``,
   so for a *single* step nothing here may move.  Over a driven window
   the returned state feeds the next step, so later steps may.  Both are
   reported: ``step0_*`` is the untainted single-step comparison,
   ``window_*`` the compounded one.
2. How far does the returned state move?  ``rel_step1`` is the
   uncompounded relative shift after one step from the identical initial
   state; ``rel_final`` is the same after the whole window.
3. Are ``at_cap_fraction == 1.0`` rows bit-identical?  A row that never
   exits on its criterion never reaches the changed line.

No timings anywhere: this machine is shared.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[int, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["row"]] = r
    return rows


def _state(rec: dict, key: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(rec[key]), dtype=np.float32)


def _rel(a: np.ndarray, b: np.ndarray) -> dict:
    """Relative shift of *b* from *a*, in L2 and worst-element terms."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    finite = np.isfinite(a64) & np.isfinite(b64)
    out = {"n": int(a64.size), "n_nonfinite": int((~finite).sum())}
    if not finite.any():
        out.update(l2=float("nan"), max_elem=float("nan"), identical=False)
        return out
    a64, b64 = a64[finite], b64[finite]
    d = b64 - a64
    den = float(np.sqrt(np.sum(a64 * a64)))
    out["l2"] = float(np.sqrt(np.sum(d * d)) / den) if den > 0 else float("nan")
    scale = np.maximum(np.abs(a64), 1e-30)
    out["max_elem"] = float(np.max(np.abs(d) / scale))
    out["identical"] = bool(np.array_equal(a, b))
    return out


def _at_cap_fraction(rec: dict) -> float:
    """Profiler rule, maximised over the row's groups."""
    caps = rec.get("caps") or {}
    best = 0.0
    for key, g in rec["groups"].items():
        cap = caps.get(key)
        if cap is None:
            continue
        it = np.asarray(g["iterations"])
        best = max(best, float(np.mean(it >= cap - 1)))
    return best


def _converged_fraction(rec: dict) -> float:
    """Fraction of steps on which every group reported converged."""
    per = [np.asarray(g["converged"], dtype=bool)
           for g in rec["groups"].values()]
    if not per:
        return float("nan")
    return float(np.mean(np.logical_and.reduce(per)))


def _iters_mean(rec: dict) -> float:
    per = [float(np.mean(g["iterations"])) for g in rec["groups"].values()]
    return max(per) if per else float("nan")


def compare(b: dict, a: dict) -> dict:
    """One comparison record for a (fixture, configuration) row."""
    out = {k: b[k] for k in ("row", "fixture", "label", "iteration_mode",
                             "acceleration", "convergence_norm")}
    if not (b.get("ok") and a.get("ok")):
        out.update(ok=False,
                   before_error=b.get("error"), after_error=a.get("error"))
        return out
    out["ok"] = True
    out["n_steps"] = b["n_steps"]

    # --- iterations / convergence, step 0 only (no feedback yet) ------
    step0_iters_same = True
    step0_conv_same = True
    window_iters_same = True
    window_conv_same = True
    first_iter_diff = None
    for key in b["group_keys"]:
        gb, ga = b["groups"][key], a["groups"][key]
        ib, ia = np.asarray(gb["iterations"]), np.asarray(ga["iterations"])
        cb, ca = (np.asarray(gb["converged"], dtype=bool),
                  np.asarray(ga["converged"], dtype=bool))
        step0_iters_same &= bool(ib[0] == ia[0])
        step0_conv_same &= bool(cb[0] == ca[0])
        window_iters_same &= bool(np.array_equal(ib, ia))
        window_conv_same &= bool(np.array_equal(cb, ca))
        where = np.flatnonzero(ib != ia)
        if where.size:
            cand = int(where[0])
            first_iter_diff = (cand if first_iter_diff is None
                               else min(first_iter_diff, cand))
    out.update(
        step0_iterations_identical=step0_iters_same,
        step0_converged_identical=step0_conv_same,
        window_iterations_identical=window_iters_same,
        window_converged_identical=window_conv_same,
        first_step_iterations_differ=first_iter_diff,
        iterations_mean_before=_iters_mean(b),
        iterations_mean_after=_iters_mean(a),
        converged_fraction_before=_converged_fraction(b),
        converged_fraction_after=_converged_fraction(a),
        at_cap_fraction_before=_at_cap_fraction(b),
        at_cap_fraction_after=_at_cap_fraction(a),
    )

    # --- state ---------------------------------------------------------
    out["rel_step1"] = _rel(_state(b, "step1_state"), _state(a, "step1_state"))
    out["rel_final"] = _rel(_state(b, "final_state"), _state(a, "final_state"))
    same = [hb == ha for hb, ha in zip(b["step_hashes"], a["step_hashes"])]
    out["bit_identical_all_steps"] = all(same)
    out["first_differing_step"] = (None if all(same)
                                   else int(same.index(False)))
    # A row that never exits on its criterion never reaches the changed
    # line; the residual it reports is the only thing the fix could have
    # touched, and it does not touch it.
    out["never_converges"] = (out["converged_fraction_before"] == 0.0)
    return out


def summarise(cmps: list[dict]) -> dict:
    ok = [c for c in cmps if c["ok"]]
    at_cap = [c for c in ok if c["at_cap_fraction_before"] == 1.0]
    conv = [c for c in ok if c["converged_fraction_before"] > 0.0]
    moved = [c for c in ok if not c["bit_identical_all_steps"]]

    def _pct(xs, q):
        return float(np.percentile(xs, q)) if xs else float("nan")

    r1 = [c["rel_step1"]["l2"] for c in conv
          if math.isfinite(c["rel_step1"]["l2"])]
    rf = [c["rel_final"]["l2"] for c in conv
          if math.isfinite(c["rel_final"]["l2"])]
    worst_step1 = max(conv, key=lambda c: (c["rel_step1"]["l2"]
                                           if math.isfinite(c["rel_step1"]["l2"])
                                           else -1), default=None)
    worst_final = max(conv, key=lambda c: (c["rel_final"]["l2"]
                                           if math.isfinite(c["rel_final"]["l2"])
                                           else -1), default=None)
    return {
        "rows_compared": len(cmps),
        "rows_ok": len(ok),
        "rows_failed": len(cmps) - len(ok),
        # Prediction 1
        "rows_with_step0_iterations_identical": sum(
            c["step0_iterations_identical"] for c in ok),
        "rows_with_step0_converged_identical": sum(
            c["step0_converged_identical"] for c in ok),
        "rows_with_window_iterations_identical": sum(
            c["window_iterations_identical"] for c in ok),
        "rows_with_window_converged_identical": sum(
            c["window_converged_identical"] for c in ok),
        "prediction1_single_step_holds": all(
            c["step0_iterations_identical"] and c["step0_converged_identical"]
            for c in ok),
        # Prediction 2
        "rows_at_cap_fraction_1": len(at_cap),
        "rows_at_cap_fraction_1_bit_identical": sum(
            c["bit_identical_all_steps"] for c in at_cap),
        "prediction2_at_cap_bit_identical_holds": all(
            c["bit_identical_all_steps"] for c in at_cap),
        "at_cap_violations": [
            {"fixture": c["fixture"], "label": c["label"],
             "first_differing_step": c["first_differing_step"],
             "rel_step1_l2": c["rel_step1"]["l2"],
             "converged_fraction_before": c["converged_fraction_before"]}
            for c in at_cap if not c["bit_identical_all_steps"]],
        # Magnitude
        "rows_that_move": len(moved),
        "rows_bit_identical": len(ok) - len(moved),
        "rows_with_a_converged_exit": len(conv),
        "rel_step1_l2_median": _pct(r1, 50),
        "rel_step1_l2_p90": _pct(r1, 90),
        "rel_step1_l2_max": max(r1) if r1 else float("nan"),
        "rel_final_l2_median": _pct(rf, 50),
        "rel_final_l2_p90": _pct(rf, 90),
        "rel_final_l2_max": max(rf) if rf else float("nan"),
        "worst_step1_row": (None if worst_step1 is None else
                            {"fixture": worst_step1["fixture"],
                             "label": worst_step1["label"],
                             "rel_l2": worst_step1["rel_step1"]["l2"]}),
        "worst_final_row": (None if worst_final is None else
                            {"fixture": worst_final["fixture"],
                             "label": worst_final["label"],
                             "rel_l2": worst_final["rel_final"]["l2"]}),
    }


def _by_group(cmps: list[dict], key) -> list[dict]:
    """Median / max relative shift bucketed by *key*."""
    buckets: dict = {}
    for c in cmps:
        if not c["ok"] or c["converged_fraction_before"] == 0.0:
            continue
        buckets.setdefault(key(c), []).append(c)
    out = []
    for name, rows in sorted(buckets.items()):
        s1 = [r["rel_step1"]["l2"] for r in rows
              if math.isfinite(r["rel_step1"]["l2"])]
        sf = [r["rel_final"]["l2"] for r in rows
              if math.isfinite(r["rel_final"]["l2"])]
        out.append({
            "key": name, "rows": len(rows),
            "bit_identical": sum(r["bit_identical_all_steps"] for r in rows),
            "rel_step1_median": float(np.median(s1)) if s1 else float("nan"),
            "rel_step1_max": max(s1) if s1 else float("nan"),
            "rel_final_median": float(np.median(sf)) if sf else float("nan"),
            "rel_final_max": max(sf) if sf else float("nan"),
        })
    return out


def _table(rows: list[dict], title: str) -> list[str]:
    out = [f"### {title}", "",
           "| key | rows | bit-identical | rel step1 median | rel step1 max "
           "| rel final median | rel final max |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        out.append(
            f"| `{r['key']}` | {r['rows']} | {r['bit_identical']} "
            f"| {r['rel_step1_median']:.2e} | {r['rel_step1_max']:.2e} "
            f"| {r['rel_final_median']:.2e} | {r['rel_final_max']:.2e} |")
    out.append("")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()

    b, a = _load(args.before), _load(args.after)
    shared = sorted(set(b) & set(a))
    cmps = [compare(b[i], a[i]) for i in shared]
    summary = summarise(cmps)
    summary["rows_only_in_before"] = sorted(set(b) - set(a))
    summary["rows_only_in_after"] = sorted(set(a) - set(b))
    by_accel = _by_group(cmps, lambda c: c["acceleration"])
    by_norm = _by_group(cmps, lambda c: c["convergence_norm"])
    by_fixture = _by_group(cmps, lambda c: c["fixture"])

    doc = {"summary": summary, "by_acceleration": by_accel,
           "by_norm": by_norm, "by_fixture": by_fixture, "rows": cmps}
    if args.out:
        args.out.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    if args.markdown:
        lines = [f"## Sweep comparison ({summary['rows_compared']} rows"
                 f" of 350)", ""]
        for k, v in summary.items():
            if k in ("at_cap_violations", "worst_step1_row",
                     "worst_final_row", "rows_only_in_before",
                     "rows_only_in_after"):
                lines.append(f"- `{k}`: `{json.dumps(v)}`")
            elif isinstance(v, float):
                lines.append(f"- `{k}`: {v:.3e}")
            else:
                lines.append(f"- `{k}`: {v}")
        lines.append("")
        lines += _table(by_accel, "By acceleration")
        lines += _table(by_norm, "By convergence norm")
        lines += _table(by_fixture, "By fixture")
        args.markdown.write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
