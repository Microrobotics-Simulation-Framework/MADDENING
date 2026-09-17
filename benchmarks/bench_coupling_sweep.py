#!/usr/bin/env python
"""Sweep the coupling options over the fixtures in ``coupling_fixtures``.

For every fixture and every configuration this records the per-step
cost, the iterations actually used against the cap, how often the group
converged, the final residual, and — where the platform can report it —
kernels per step and the device-busy fraction, so each row says whether
it was launch-bound or compute-bound.

    JAX_PLATFORMS=cpu python benchmarks/bench_coupling_sweep.py \\
        --json benchmarks/results/coupling_sweep_cpu.json

Useful flags::

    --fixtures chain-5,ring-8   only these fixtures (default: the fast set)
    --include-slow              also run the 1e5-cell fixtures
    --fields                    add the accelerated_fields variants
    --steps N                   override every fixture's timed-step count
    --trace / --no-trace        force the jax.profiler trace on or off
                                (default: on only when the device is not CPU)

The JSON follows the conventions of ``bench_coupling.py``: a header with
the device and JAX version, then one object per measured variant.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from coupling_fixtures import (  # noqa: E402
    FIXTURES,
    CouplingConfig,
    ScopeNotApplicable,
    fixture_names,
    sweep_configs,
)
from maddening.core.simulation.profiler import profile_graph  # noqa: E402


#: A step whose dispatch floor is at least this fraction of its wall time
#: is spending most of its time getting work to the device rather than
#: doing it.  On a GPU the device-busy fraction from the trace is the
#: better signal and takes precedence; on CPU, where the trace reports no
#: kernels, this ratio is all there is.
_LAUNCH_BOUND_AT = 0.33
_COMPUTE_BOUND_AT = 0.10


def _regime(mean_ms: float, floor_ms: float, trace) -> tuple[str, float]:
    """Classify a row as launch- or compute-bound, and say on what basis."""
    if trace is not None and trace.n_kernels_per_step > 0:
        busy = trace.device_busy_fraction
        if busy >= 0.7:
            return "compute-bound", busy
        if busy <= 0.3:
            return "launch-bound", busy
        return "mixed", busy
    if mean_ms <= 0:
        return "unknown", 0.0
    frac = floor_ms / mean_ms
    if frac >= _LAUNCH_BOUND_AT:
        return "launch-bound", frac
    if frac <= _COMPUTE_BOUND_AT:
        return "compute-bound", frac
    return "mixed", frac


def _state_signature(gm) -> dict[str, list]:
    """A small, order-stable fingerprint of the graph's state.

    Used to check that every configuration of a fixture lands on the
    same fixed point; a full state dump would bloat the JSON for the
    1e5-cell fixtures.  Each ``"<node>.<field>"`` entry carries
    ``[sum, l2, max_abs]`` — the sum and the norm are what a
    disagreement shows up in, and ``max_abs`` is the scale to measure
    that disagreement against.
    """
    out: dict[str, list] = {}
    for name in sorted(gm.node_names):
        state = gm.get_node_state(name)
        for field in sorted(state):
            arr = np.asarray(state[field], dtype=np.float64).ravel()
            if arr.size == 0:
                continue
            out[f"{name}.{field}"] = [
                float(arr.sum()),
                float(np.sqrt(np.sum(arr * arr))),
                float(np.max(np.abs(arr))),
            ]
    return out


def _measure(spec, config: CouplingConfig, *, steps: int, warmup: int,
             trace: bool, trace_steps: int) -> dict:
    """Build, profile and diagnose one (fixture, configuration) pair."""
    row: dict = {
        "fixture": spec.name,
        "family": spec.family,
        "label": config.label,
        "iteration_mode": config.iteration_mode,
        "acceleration": config.acceleration,
        "relaxation": config.relaxation,
        "convergence_norm": config.convergence_norm,
        "accel_scope": config.accel_scope,
        "jacobian_reuse": config.jacobian_reuse,
    }
    t0 = time.perf_counter()
    try:
        built = spec.build(config)
    except ScopeNotApplicable as exc:
        row["ok"] = False
        row["skipped"] = True
        row["error"] = str(exc)
        return row
    except Exception as exc:  # a fixture that cannot be built is a result
        row["ok"] = False
        row["skipped"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    row["build_s"] = time.perf_counter() - t0

    gm = built.gm
    rep = profile_graph(
        gm, n_steps=steps, n_warmup=warmup, measure_coupling=False,
        trace=trace, trace_steps=trace_steps,
    )
    diag = gm.coupling_diagnostics()

    row.update({
        "ok": True,
        "error": None,
        "n_nodes": rep.n_nodes,
        "state_elements": rep.total_state_elements,
        "jit_compile_ms": rep.jit_compile_ms,
        "mean_step_ms": rep.mean_step_ms,
        "median_step_ms": rep.median_step_ms,
        "p95_step_ms": rep.p95_step_ms,
        "dispatch_floor_ms": rep.dispatch_floor_ms,
    })

    groups = {}
    for key, st in rep.coupling_iter_stats.items():
        groups[key] = {
            "iterations_mean": st["mean"],
            "iterations_min": st["min"],
            "iterations_max": st["max"],
            "cap": st["cap"],
            "at_cap_fraction": st["at_cap_fraction"],
            "converged_fraction": st["converged_fraction"],
            "final_residual": float(diag.get(key, {}).get("residual", float("nan"))),
        }
    row["groups"] = groups
    if groups:
        row["iterations_mean"] = max(g["iterations_mean"] for g in groups.values())
        row["iterations_max"] = max(g["iterations_max"] for g in groups.values())
        row["cap"] = max(g["cap"] for g in groups.values())
        row["converged_fraction"] = min(
            g["converged_fraction"] for g in groups.values())
        row["at_cap_fraction"] = max(g["at_cap_fraction"] for g in groups.values())
        residuals = [g["final_residual"] for g in groups.values()]
        finite = [r for r in residuals if math.isfinite(r)]
        row["final_residual"] = max(finite) if finite else residuals[0]

    tr = rep.trace
    row["n_kernels_per_step"] = tr.n_kernels_per_step if tr else 0.0
    row["device_busy_fraction"] = tr.device_busy_fraction if tr else 0.0
    row["device_busy_ms_per_step"] = tr.device_busy_ms_per_step if tr else 0.0
    row["host_dispatch_ms_per_step"] = tr.host_dispatch_ms_per_step if tr else 0.0
    regime, basis = _regime(rep.mean_step_ms, rep.dispatch_floor_ms, tr)
    row["regime"] = regime
    row["regime_basis"] = basis
    row["regime_from"] = "device_busy_fraction" if (
        tr is not None and tr.n_kernels_per_step > 0
    ) else "dispatch_floor_over_step"

    row["state_signature"] = _state_signature(gm)
    row["predicted_rho"] = dict(built.predicted_rho)
    return row


def _field_scales(sig: dict) -> dict:
    """Largest magnitude of each *field name*, across the nodes carrying it.

    A single global scale hides the thing this check exists to catch: on
    the heterogeneous fixture the grid's temperature is ~1 and the
    scalar nodes' positions are ~1e-2, so dividing everything by the
    largest entry lets an accelerator move a scalar node by its whole
    value and still score 1e-2.  Normalising each entry against *itself*
    has the opposite failure — an oscillating trajectory passing through
    zero explodes, and on the spring fixtures every field is a single
    scalar, so per-entry is exactly that case.  Taking the scale of a
    field name across all the nodes carrying it is the measure that
    survives both.
    """
    scales: dict[str, float] = {}
    for key, (_s, _l2, max_abs) in sig.items():
        field = key.split(".", 1)[1]
        scales[field] = max(scales.get(field, 0.0), abs(max_abs))
    return scales


def _same_fixed_point(rows: list[dict]) -> dict:
    """Largest deviation between converged rows' fixed points.

    An accelerator that converges somewhere else is a bug, not a
    speed-up, so the sweep reports this next to the timings rather than
    leaving it to the test suite alone.
    """
    ref = None
    scales: dict = {}
    worst = 0.0
    worst_label = ""
    for row in rows:
        if not row.get("ok") or row.get("converged_fraction", 0.0) < 1.0:
            continue
        sig = row["state_signature"]
        if ref is None:
            ref, scales = sig, _field_scales(sig)
            continue
        if set(sig) != set(ref):
            continue
        for key, values in sig.items():
            scale = max(scales[key.split(".", 1)[1]], 1e-12)
            dev = max(abs(v - r) for v, r in zip(values, ref[key])) / scale
            if dev > worst:
                worst, worst_label = dev, row["label"]
    return {"max_relative_deviation": worst, "worst_config": worst_label}


#: Preference order when two configurations time the same: the simpler
#: one wins, because every extra knob is a thing that can be wrong.
_SIMPLICITY = {"none": 0, "fixed": 1, "aitken": 2, "iqn-ils": 3, "iqn-imvj": 4}


def _best(rows: list[dict]) -> dict:
    """Best fully converged configuration of a fixture.

    Not simply the fastest row: on the launch-bound fixtures the spread
    between configurations is a few tens of microseconds, well inside
    the run-to-run noise, so a bare ``min`` on ``mean_step_ms`` reports
    whichever row happened to get a quiet scheduler slice.  Anything
    within 10% of the fastest is treated as tied and broken on iteration
    count — which *is* measured exactly — and then on simplicity.
    """
    ok = [r for r in rows
          if r.get("ok") and r.get("converged_fraction", 0.0) >= 1.0]
    if not ok:
        return {}
    floor = min(r["mean_step_ms"] for r in ok)
    tied = [r for r in ok if r["mean_step_ms"] <= floor * 1.10]
    best = min(tied, key=lambda r: (r.get("iterations_mean", 1e9),
                                    _SIMPLICITY.get(r["acceleration"], 9),
                                    r["mean_step_ms"]))
    base = next(
        (r for r in ok
         if r["iteration_mode"] == "gauss-seidel"
         and r["acceleration"] == "none" and r["convergence_norm"] == "l2"),
        None,
    )
    out = {
        "label": best["label"],
        "mean_step_ms": best["mean_step_ms"],
        "iterations_mean": best.get("iterations_mean"),
        "n_tied_within_10pct": len(tied),
        "fastest_mean_step_ms": floor,
    }
    if base is not None:
        out["baseline_label"] = base["label"]
        out["baseline_mean_step_ms"] = base["mean_step_ms"]
        out["speedup_vs_baseline"] = base["mean_step_ms"] / best["mean_step_ms"]
    return out


#: The combinations the brief singles out, each reported explicitly
#: whether or not it wins.
_HIGHLIGHTS = {
    "jacobi+aitken": lambda r: (r["iteration_mode"] == "jacobi"
                                and r["acceleration"] == "aitken"),
    "jacobi+iqn": lambda r: (r["iteration_mode"] == "jacobi"
                             and r["acceleration"].startswith("iqn")),
    "gs+under-relaxation": lambda r: (r["iteration_mode"] == "gauss-seidel"
                                      and r["acceleration"] == "fixed"
                                      and r["relaxation"] < 1.0),
    "interface-norm+interface-fields": lambda r: (
        r["convergence_norm"] == "interface"
        and r["acceleration"].startswith("iqn")
        and r["accel_scope"] in ("auto", "interface")),
}


def _highlights(rows: list[dict]) -> dict:
    out = {}
    for name, pred in _HIGHLIGHTS.items():
        picked = [r for r in rows if r.get("ok") and pred(r)]
        if not picked:
            continue
        conv = [r for r in picked if r.get("converged_fraction", 0.0) >= 1.0]
        pool = conv or picked
        best = min(pool, key=lambda r: r["mean_step_ms"])
        out[name] = {
            "label": best["label"],
            "mean_step_ms": best["mean_step_ms"],
            "iterations_mean": best.get("iterations_mean"),
            "converged_fraction": best.get("converged_fraction"),
            "n_configs": len(picked),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", default="",
                    help="comma-separated fixture names (default: fast set)")
    ap.add_argument("--include-slow", action="store_true",
                    help="also run the 1e5-cell fixtures (minutes, not seconds)")
    ap.add_argument("--fields", action="store_true",
                    help="add the accelerated_fields (accel_scope) variants")
    ap.add_argument("--norms", default="l2,interface")
    ap.add_argument("--steps", type=int, default=0,
                    help="override every fixture's timed-step count")
    ap.add_argument("--max-iterations", type=int, default=0,
                    help="override every group's iteration cap.  IQN "
                         "allocates max_iterations-1 secant columns, so "
                         "this is also the knob that decides how big its "
                         "least-squares problem is")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--trace", dest="trace", action="store_true", default=None)
    ap.add_argument("--no-trace", dest="trace", action="store_false")
    ap.add_argument("--trace-steps", type=int, default=10)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    device = str(jax.devices()[0])
    trace = args.trace
    if trace is None:
        trace = "cpu" not in device.lower()

    if args.fixtures:
        names = [n.strip() for n in args.fixtures.split(",") if n.strip()]
        unknown = [n for n in names if n not in FIXTURES]
        if unknown:
            raise SystemExit(f"unknown fixture(s): {unknown}; "
                             f"known: {sorted(FIXTURES)}")
    else:
        names = fixture_names(include_slow=args.include_slow)

    norms = tuple(n.strip() for n in args.norms.split(",") if n.strip())
    configs = sweep_configs(norms, extra_fields=args.fields)
    if args.max_iterations:
        configs = [replace(c, max_iterations=args.max_iterations)
                   for c in configs]

    rows: list[dict] = []
    per_fixture: dict[str, dict] = {}
    t_start = time.perf_counter()
    for name in names:
        spec = FIXTURES[name]
        # ``mixed-modes`` pins each group's iteration mode itself, so
        # sweeping that axis would only duplicate rows.
        cfgs = ([c for c in configs if c.iteration_mode == "gauss-seidel"]
                if spec.mode_fixed else configs)
        steps = args.steps or spec.steps
        warmup = args.warmup or spec.warmup
        fixture_rows: list[dict] = []
        for cfg in cfgs:
            row = _measure(spec, cfg, steps=steps, warmup=warmup,
                           trace=trace, trace_steps=args.trace_steps)
            fixture_rows.append(row)
            rows.append(row)
            if not args.quiet:
                if row["ok"]:
                    print(f"  {name:18s} {row['label']:36s} "
                          f"{row['mean_step_ms']:8.3f} ms  "
                          f"it {row.get('iterations_mean', float('nan')):5.1f}"
                          f"/{row.get('cap', 0):<3d} "
                          f"conv {row.get('converged_fraction', 0) * 100:3.0f}%  "
                          f"{row['regime']}")
                elif row.get("skipped"):
                    print(f"  {name:18s} {row['label']:36s}  skipped "
                          f"({row['error']})")
                else:
                    print(f"  {name:18s} {row['label']:36s}  FAILED "
                          f"{row['error']}")
                sys.stdout.flush()
        per_fixture[name] = {
            "summary": spec.summary,
            "family": spec.family,
            "slow": spec.slow,
            "steps": steps,
            "warmup": warmup,
            "expect": spec.expect,
            "predicted_rho": next(
                (r["predicted_rho"] for r in fixture_rows
                 if r.get("ok") and r.get("predicted_rho")), {}),
            "fixed_point_agreement": _same_fixed_point(fixture_rows),
            "best": _best(fixture_rows),
            "highlights": _highlights(fixture_rows),
        }

    out = {
        "benchmark": "coupling-sweep",
        "generated_by": "benchmarks/bench_coupling_sweep.py",
        "label": args.label,
        "device": device,
        "jax": jax.__version__,
        "trace": trace,
        "max_iterations_override": args.max_iterations or None,
        "n_rows": len(rows),
        "wall_s": time.perf_counter() - t_start,
        "fixtures": per_fixture,
        "rows": rows,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {args.json}  ({len(rows)} rows, "
              f"{out['wall_s']:.0f} s)")

    print("\n=== Best converged configuration per fixture ===")
    for name in names:
        best = per_fixture[name].get("best") or {}
        if not best:
            print(f"  {name:18s} (nothing converged)")
            continue
        speed = best.get("speedup_vs_baseline")
        print(f"  {name:18s} {best['label']:36s} "
              f"{best['mean_step_ms']:8.3f} ms  "
              f"it {best.get('iterations_mean', float('nan')):5.1f}"
              + (f"   {speed:.2f}x vs gs/none/l2" if speed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
