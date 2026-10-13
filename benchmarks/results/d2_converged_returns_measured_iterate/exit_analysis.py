#!/usr/bin/env python
"""Why the returned state moves as far as it does, per exit kind.

``diff_sweep.py`` says how far each row moved.  This says what predicts
it, which is what decides whether a downstream experiment has to be
re-run: the shift on a criterion exit is *one residual*, so it is large
exactly when the criterion is loose **relative to the magnitude of the
state**, not when the residual is large in absolute terms.

    python exit_analysis.py raw/before.jsonl raw/after.jsonl \\
        --out exit_analysis.json --markdown exit_analysis.md

Three cuts:

* **by exit kind at step 0** -- criterion exit (the changed line ran) or
  cap exit (it did not).  Cap-exit rows must be bit-identical.
* **shift against one residual** -- for ``convergence_norm="l2"``, where
  the reported residual is ``||F(x) - x||_2`` in the state's own units,
  the predicted relative shift is ``residual / ||x||``.  The ratio of
  measured to predicted says whether "one residual" is the whole story.
* **pass-one exits** -- rows that meet their criterion on the first
  pass.  That is the configuration MIME's D2 group is in (a 1.7e-05 N
  drag force against ``atol=1e-8``), and it is where the shift is a
  whole update rather than a converged tail.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[int, dict]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["row"]] = r
    return out


def _state(rec: dict, key: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(rec[key]), dtype=np.float32)


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    ok = np.isfinite(a64) & np.isfinite(b64)
    if not ok.any():
        return float("nan")
    a64, b64 = a64[ok], b64[ok]
    den = float(np.sqrt(np.sum(a64 * a64)))
    if den == 0.0:
        return float("nan")
    return float(np.sqrt(np.sum((b64 - a64) ** 2)) / den)


def _by_field(rb: dict, ra: dict, key: str) -> dict[str, float]:
    """Relative shift split by state field name.

    The spring fixtures' interface field is ``position``; ``velocity``
    is state the convergence criterion never looks at and (under
    ``iqn-*``) the accelerator never touches.  Splitting the shift that
    way is what separates "the coupled quantity moved" from "a field
    nothing was checking moved".
    """
    b, a = _state(rb, key), _state(ra, key)
    out: dict[str, list] = {}
    off = 0
    for name, n in zip(rb["state_fields"], rb["state_sizes"]):
        field = name.split(".", 1)[1]
        out.setdefault(field, [[], []])
        out[field][0].append(b[off:off + n])
        out[field][1].append(a[off:off + n])
        off += n
    return {f: _rel_l2(np.concatenate(bs), np.concatenate(as_))
            for f, (bs, as_) in out.items()}


def _step0(rec: dict) -> dict:
    """What the first step did, per the pre-fix tree."""
    caps = rec.get("caps") or {}
    iters, res, conv, at_cap = [], [], [], []
    for key in rec["group_keys"]:
        g = rec["groups"][key]
        iters.append(int(g["iterations"][0]))
        res.append(float(g["residual"][0]))
        conv.append(bool(g["converged"][0]))
        cap = caps.get(key)
        at_cap.append(cap is not None and iters[-1] >= cap - 1)
    return {
        "iterations": max(iters),
        "min_iterations": min(iters),
        "residual": max(res),
        "all_converged": all(conv),
        "any_at_cap": any(at_cap),
        "state_l2": float(rec["step_l2"][0]),
    }


def _stats(xs: list[float]) -> dict:
    xs = [x for x in xs if math.isfinite(x)]
    if not xs:
        return {"n": 0}
    a = np.asarray(xs)
    return {"n": int(a.size), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max()),
            "min": float(a.min())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()

    b, a = _load(args.before), _load(args.after)
    rows = []
    for i in sorted(set(b) & set(a)):
        rb, ra = b[i], a[i]
        if not (rb.get("ok") and ra.get("ok")):
            continue
        s0 = _step0(rb)
        rel1 = _rel_l2(_state(rb, "step1_state"), _state(ra, "step1_state"))
        relf = _rel_l2(_state(rb, "final_state"), _state(ra, "final_state"))
        pred = (s0["residual"] / s0["state_l2"]
                if s0["state_l2"] > 0 else float("nan"))
        rows.append({
            "row": i, "fixture": rb["fixture"], "label": rb["label"],
            "acceleration": rb["acceleration"],
            "convergence_norm": rb["convergence_norm"],
            "step0_iterations": s0["iterations"],
            "step0_min_iterations": s0["min_iterations"],
            "step0_residual": s0["residual"],
            "step0_state_l2": s0["state_l2"],
            "step0_all_converged": s0["all_converged"],
            "step0_any_at_cap": s0["any_at_cap"],
            "rel_step1": rel1, "rel_final": relf,
            "residual_over_norm": pred,
            "measured_over_predicted": (rel1 / pred
                                        if pred not in (0.0,) and
                                        math.isfinite(pred) and pred > 0
                                        else float("nan")),
            "bit_identical": rb["step_hashes"] == ra["step_hashes"],
            "rel_step1_by_field": _by_field(rb, ra, "step1_state"),
            "rel_final_by_field": _by_field(rb, ra, "final_state"),
        })

    crit = [r for r in rows if r["step0_all_converged"]
            and not r["step0_any_at_cap"]]
    cap = [r for r in rows if r["step0_any_at_cap"]]
    unconv = [r for r in rows if not r["step0_all_converged"]]
    l2rows = [r for r in crit if r["convergence_norm"] == "l2"]
    pass_one = [r for r in crit
                if r["step0_iterations"] <= min(
                    (x["step0_iterations"] for x in crit), default=0)]

    doc = {
        "criterion_exit_at_step0": {
            "rows": len(crit),
            "rel_step1": _stats([r["rel_step1"] for r in crit]),
            "rel_final": _stats([r["rel_final"] for r in crit]),
            "bit_identical": sum(r["bit_identical"] for r in crit),
        },
        "cap_exit_at_step0": {
            "rows": len(cap),
            "bit_identical_whole_window": sum(r["bit_identical"] for r in cap),
            "rel_step1": _stats([r["rel_step1"] for r in cap]),
        },
        "unconverged_at_step0": {
            "rows": len(unconv),
            "bit_identical_whole_window": sum(
                r["bit_identical"] for r in unconv),
            "rel_step1": _stats([r["rel_step1"] for r in unconv]),
        },
        "one_residual_check_l2_norm": {
            "rows": len(l2rows),
            "measured_over_predicted": _stats(
                [r["measured_over_predicted"] for r in l2rows]),
            "predicted_residual_over_norm": _stats(
                [r["residual_over_norm"] for r in l2rows]),
        },
        "fewest_pass_criterion_exits": {
            "iterations": (min((r["step0_iterations"] for r in crit),
                               default=None)),
            "rows": len(pass_one),
            "rel_step1": _stats([r["rel_step1"] for r in pass_one]),
        },
        "rows": rows,
    }

    # Where the shift lands: the criterion (and, under ``iqn-*``, the
    # accelerator) sees only the interface fields, so a field outside
    # that set can move by much more than the residual.
    fields = sorted({f for r in rows for f in r["rel_step1_by_field"]})
    doc["by_state_field"] = {
        f: {
            "criterion_exit_rows": _stats(
                [r["rel_step1_by_field"].get(f, float("nan")) for r in crit]),
            "criterion_exit_rows_iqn": _stats(
                [r["rel_step1_by_field"].get(f, float("nan")) for r in crit
                 if r["acceleration"].startswith("iqn")
                 and r["convergence_norm"] == "interface"]),
        }
        for f in fields
    }
    if args.out:
        args.out.write_text(json.dumps(doc, indent=1) + "\n")
    if args.markdown:
        lines = ["## What predicts the shift", ""]
        for name in ("criterion_exit_at_step0", "cap_exit_at_step0",
                     "unconverged_at_step0", "one_residual_check_l2_norm",
                     "fewest_pass_criterion_exits", "by_state_field"):
            lines.append(f"### `{name}`")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(doc[name], indent=1))
            lines.append("```")
            lines.append("")
        worst = sorted((r for r in rows if math.isfinite(r["rel_step1"])),
                       key=lambda r: -r["rel_step1"])[:15]
        lines += ["### Fifteen largest single-step shifts", "",
                  "| fixture | config | step0 iters | converged | "
                  "rel shift after 1 step | rel shift after the window |",
                  "|---|---|---:|---|---:|---:|"]
        for r in worst:
            lines.append(
                f"| `{r['fixture']}` | `{r['label']}` | "
                f"{r['step0_iterations']} | "
                f"{'yes' if r['step0_all_converged'] else 'no'} | "
                f"{r['rel_step1']:.2e} | {r['rel_final']:.2e} |")
        lines.append("")
        args.markdown.write_text("\n".join(lines) + "\n")

    print(json.dumps({k: v for k, v in doc.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
