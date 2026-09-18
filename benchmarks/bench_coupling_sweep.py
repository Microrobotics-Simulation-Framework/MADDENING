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
    --stat-steps N              steps in the iteration-statistics pass
                                (default: the fixture's own step count,
                                capped at 50).  Independent of --steps, so
                                a shorter timing run does not also shorten
                                and shift the window the iteration counts
                                are averaged over
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

#: Cap on the iteration-statistics window.
#:
#: The window itself is the *fixture's* declared step count, not the
#: run's.  The statistics pass used to be ``min(n_steps, 50)`` steps
#: taken from wherever the timed run stopped, so ``--steps 10`` halved
#: the iteration-count sample and moved its window earlier in the
#: trajectory; on fixtures driven by a 44-step oscillator the mean
#: iteration count moved with it (``jac/iqn-imvj5/l2`` on ``chain-5``:
#: 4.00 at ten steps, 3.00 at the default).  Taking it from the fixture
#: makes it a property of the graph — ``stiff-pair-1.2`` is declared
#: short because a divergent group overflows, ``slow-drift`` is declared
#: long because its transient is — while ``--steps`` changes only how
#: many timings are averaged.
_STAT_CAP = 50


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


def _coupled_nodes(built) -> frozenset:
    """The nodes inside some coupling group of *built*."""
    out: set[str] = set()
    for key in built.group_keys:
        out.update(key.split("+"))
    return frozenset(out)


def _state_signature(gm, nodes=None) -> dict[str, list]:
    """A small, order-stable fingerprint of the graph's state.

    Used to check that every configuration of a fixture lands on the
    same fixed point; a full state dump would bloat the JSON for the
    1e5-cell fixtures.  Each ``"<node>.<field>"`` entry carries
    ``[sum, l2, max_abs, n]``.

    ``n`` is the entry's element count and it is load-bearing rather
    than informational.  ``sum`` and ``l2`` are *extensive* — they grow
    with the array — while ``max_abs`` is *intensive*, so comparing the
    raw triple against a scale taken from ``max_abs`` reports a
    per-cell disagreement of 7e-7 on a 60 000-cell grid as a "relative
    deviation" of 4e-2, and a fixture's score then depends on its state
    size rather than on its physics.  With ``n`` recorded,
    :func:`_same_fixed_point` can divide the sum by ``n`` and the norm
    by ``sqrt(n)`` and compare three intensive quantities.

    *nodes* restricts the signature to a subset — the driver nodes sit
    outside every coupling group and have no business setting the scale
    that a coupled node's disagreement is measured against.
    """
    out: dict[str, list] = {}
    for name in sorted(gm.node_names):
        if nodes is not None and name not in nodes:
            continue
        state = gm.get_node_state(name)
        for field in sorted(state):
            arr = np.asarray(state[field], dtype=np.float64).ravel()
            if arr.size == 0:
                continue
            out[f"{name}.{field}"] = [
                float(arr.sum()),
                float(np.sqrt(np.sum(arr * arr))),
                float(np.max(np.abs(arr))),
                int(arr.size),
            ]
    return out


def _measure(spec, config: CouplingConfig, *, steps: int, warmup: int,
             stat_steps: int, trace: bool, trace_steps: int) -> dict:
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
        n_stat_steps=stat_steps, trace=trace, trace_steps=trace_steps,
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
    # ``expect`` used to be recorded per fixture and never compared to
    # anything, so a fixture could declare itself launch-bound and
    # measure compute-bound for its whole sweep without a word.
    row["expect"] = spec.expect
    row["regime_matches_expect"] = regime == spec.expect
    row["regime_from"] = "device_busy_fraction" if (
        tr is not None and tr.n_kernels_per_step > 0
    ) else "dispatch_floor_over_step"

    coupled = _coupled_nodes(built)
    row["state_signature"] = _state_signature(gm, coupled)
    # Kept for diagnostics, deliberately not part of the comparison:
    # these are the driver nodes, which sit outside every group.
    row["context_signature"] = _state_signature(
        gm, frozenset(gm.node_names) - coupled)
    row["predicted_rho"] = dict(built.predicted_rho)
    return row


def _intensive(entry: list) -> tuple[float, float, float]:
    """``[sum, l2, max_abs, n]`` as three size-independent quantities.

    The mean and the root-mean-square are what the sum and the L2 norm
    become once the element count is divided out; ``max_abs`` already
    is one.  All three then mean the same thing on a four-element
    interface and on a 60 000-cell grid, which is what makes deviations
    comparable across fixtures.
    """
    total, l2, max_abs, n = entry[0], entry[1], entry[2], max(int(entry[3]), 1)
    return total / n, l2 / math.sqrt(n), max_abs


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

    This only works if *sig* is already restricted to the coupled
    nodes.  ``heterogeneous``'s driver is a spring, so it carries a
    ``position`` of ~0.88 while the coupled probes carry ~0.011; with
    the driver in the signature every probe disagreement was divided by
    eighty, and a 20.6% error scored 2.6e-3.
    """
    scales: dict[str, float] = {}
    for key, entry in sig.items():
        field = key.split(".", 1)[1]
        scales[field] = max(scales.get(field, 0.0), abs(entry[2]))
    return scales


#: An entry whose own amplitude is below this fraction of its field's
#: scale is measured against that fraction instead, so a node passing
#: through zero cannot divide a finite disagreement by nothing.
_NODE_SCALE_FLOOR = 1e-2


def _same_fixed_point(rows: list[dict]) -> dict:
    """Largest deviation between converged rows' fixed points.

    An accelerator that converges somewhere else is a bug, not a
    speed-up, so the sweep reports this next to the timings rather than
    leaving it to the test suite alone.

    Two numbers, because one is not enough on a mixed-scale fixture.
    ``max_relative_deviation`` measures every entry against its *field's*
    scale, which is the right question for "did the graph land
    somewhere else".  ``max_node_relative_deviation`` measures each
    entry against its own amplitude (floored, so a zero crossing cannot
    divide by nothing), which is the right question for "did any one
    node move", and is the one that catches a small-amplitude node
    hiding behind a large-amplitude one that shares its field name.
    """
    ref = None
    scales: dict = {}
    worst = 0.0
    worst_label = ""
    worst_node = 0.0
    worst_node_label = ""
    worst_node_key = ""
    for row in rows:
        if not row.get("ok") or row.get("converged_fraction", 0.0) < 1.0:
            continue
        sig = row["state_signature"]
        if ref is None:
            ref, scales = sig, _field_scales(sig)
            continue
        if set(sig) != set(ref):
            continue
        for key, entry in sig.items():
            field_scale = max(scales[key.split(".", 1)[1]], 1e-12)
            got, want = _intensive(entry), _intensive(ref[key])
            gap = max(abs(v - r) for v, r in zip(got, want))
            dev = gap / field_scale
            if dev > worst:
                worst, worst_label = dev, row["label"]
            own = max(abs(ref[key][2]), _NODE_SCALE_FLOOR * field_scale, 1e-12)
            dev_node = gap / own
            if dev_node > worst_node:
                worst_node = dev_node
                worst_node_label, worst_node_key = row["label"], key
    return {
        "max_relative_deviation": worst,
        "worst_config": worst_label,
        "max_node_relative_deviation": worst_node,
        "worst_node_config": worst_node_label,
        "worst_node_entry": worst_node_key,
    }


#: Floor on the tie band, as a fraction of the fastest median.  A
#: fixture whose rows all sampled cleanly still cannot resolve
#: differences this small between separate timing runs.
_MIN_TIE_BAND = 0.10

#: Preference order when two configurations time the same: the simpler
#: one wins, because every extra knob is a thing that can be wrong.
_SIMPLICITY = {"none": 0, "fixed": 1, "aitken": 2, "iqn-ils": 3, "iqn-imvj": 4}


def _best(rows: list[dict]) -> dict:
    """Best fully converged configuration of a fixture.

    Not simply the fastest row: on the launch-bound fixtures the spread
    between configurations is a few tens of microseconds, well inside
    the run-to-run noise, so a bare ``min`` on ``mean_step_ms`` reports
    whichever row happened to get a quiet scheduler slice.  Rows that
    time the same are treated as tied and broken on iteration count —
    which *is* measured exactly — and then on simplicity.

    How wide "the same" is comes from the measurement, not from a fixed
    percentage.  A 10% band around the fastest *mean* was far inside
    what these rows actually resolve: ``star-8``'s
    ``gs/fixed0.8/interface`` recorded a median of 0.299 ms against a
    p95 of 0.549 ms, an 84% swing within one row, so the band excluded
    ``gs/aitken/interface`` — not measurably slower — and the fixture's
    reported best became the row with 2.6x the iterations.  Ranking on a
    metric whose noise exceeds the differences being ranked is how that
    happens.

    Two things set the band.  The first is the **dispatch floor**, the
    time a jitted identity on the same pytree takes, which the profiler
    measures per row: it is the part of a step that is getting work to
    the device rather than doing it, and a difference smaller than it is
    not a statement about the algorithm.  On ``star-8`` that floor is
    0.217 ms against a fastest step of 0.288 ms — most of the step — so
    a 29% gap between two configurations there is inside the dispatch
    and means nothing, while the 3x gap in their iteration counts is
    exact.  The median floor over the fixture's rows is used, so one
    row's bad sample cannot set it.  The second is the sampling spread,
    the median over the rows of ``(p95 - median) / median``, again a
    median over rows so that one row's tail does not widen the band for
    everything else, and floored at 10%.  The band is the fastest
    median plus whichever of the two is larger.

    Rows inside it are ranked on iteration count — which is measured
    exactly — and then on simplicity.  Rows outside it really are
    slower: IQN on ``star-8`` costs 1.2-1.4 ms against a 0.5 ms band and
    is not in the running whatever its iteration count says, which is
    the whole reason this function does not simply rank on iterations.
    Medians rather than means throughout: one slow sample moves a mean
    and not a median.
    """
    ok = [r for r in rows
          if r.get("ok") and r.get("converged_fraction", 0.0) >= 1.0]
    if not ok:
        return {}

    def _median(r):
        return r.get("median_step_ms", r["mean_step_ms"])

    def _spread(r):
        m = _median(r)
        return (r.get("p95_step_ms", m) - m) / m if m > 0 else 0.0

    def _mid(values):
        values = sorted(values)
        return values[len(values) // 2] if values else 0.0

    noise = max(_mid([_spread(r) for r in ok]), _MIN_TIE_BAND)
    dispatch = _mid([r.get("dispatch_floor_ms", 0.0) for r in ok])
    fastest = min(ok, key=_median)
    floor = _median(fastest)
    band = floor + max(dispatch, noise * floor)
    tied = [r for r in ok if _median(r) <= band]
    best = min(tied, key=lambda r: (r.get("iterations_mean", 1e9),
                                    _SIMPLICITY.get(r["acceleration"], 9),
                                    _median(r)))
    base = next(
        (r for r in ok
         if r["iteration_mode"] == "gauss-seidel"
         and r["acceleration"] == "none" and r["convergence_norm"] == "l2"),
        None,
    )
    fewest = min(ok, key=lambda r: (r.get("iterations_mean", 1e9),
                                    _SIMPLICITY.get(r["acceleration"], 9)))
    out = {
        "label": best["label"],
        "mean_step_ms": best["mean_step_ms"],
        "median_step_ms": best.get("median_step_ms"),
        "iterations_mean": best.get("iterations_mean"),
        "tie_rule": "median <= fastest median + max(dispatch floor, "
                    "spread x fastest median)",
        "tie_noise": noise,
        "tie_dispatch_floor_ms": dispatch,
        "n_tied": len(tied),
        "fastest_label": fastest["label"],
        "fastest_median_step_ms": floor,
        # Recorded next to the pick rather than folded into it: where
        # these two disagree, the reader is looking at a fixture whose
        # step time and iteration count do not point the same way, and
        # that is worth seeing rather than resolving silently.
        "fewest_iterations_label": fewest["label"],
        "fewest_iterations_mean": fewest.get("iterations_mean"),
        "tie_band_ms": band,
    }
    if base is not None:
        out["baseline_label"] = base["label"]
        out["baseline_mean_step_ms"] = base["mean_step_ms"]
        out["baseline_median_step_ms"] = base.get("median_step_ms")
        out["speedup_vs_baseline"] = (
            _median(base) / _median(best) if _median(best) > 0 else float("nan"))
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
        # Median for the same reason as in ``_best``: on a launch-bound
        # row a single slow sample moves the mean and not the median.
        best = min(pool, key=lambda r: r.get("median_step_ms",
                                             r["mean_step_ms"]))
        out[name] = {
            "label": best["label"],
            "mean_step_ms": best["mean_step_ms"],
            "median_step_ms": best.get("median_step_ms"),
            "iterations_mean": best.get("iterations_mean"),
            "converged_fraction": best.get("converged_fraction"),
            "n_configs": len(picked),
        }
    return out


def _resummarise(path: Path) -> int:
    """Recompute a recorded file's derived summaries in place."""
    data = json.loads(path.read_text())
    for name, summary in data["fixtures"].items():
        fixture_rows = [r for r in data["rows"] if r["fixture"] == name]
        summary["fixed_point_agreement"] = _same_fixed_point(fixture_rows)
        summary["best"] = _best(fixture_rows)
        summary["highlights"] = _highlights(fixture_rows)
        summary["regime_matches_expect_fraction"] = (
            sum(1 for r in fixture_rows if r.get("regime_matches_expect"))
            / max(sum(1 for r in fixture_rows if r.get("ok")), 1))
    data["resummarised_by"] = "benchmarks/bench_coupling_sweep.py --resummarise"
    path.write_text(json.dumps(data, indent=2, default=float))
    print(f"resummarised {path} ({len(data['rows'])} rows)")
    for name, summary in data["fixtures"].items():
        best = summary.get("best") or {}
        print(f"  {name:18s} {best.get('label', '(nothing converged)'):36s} "
              f"{best.get('mean_step_ms', float('nan')):8.3f} ms  "
              f"it {best.get('iterations_mean', float('nan')):5.1f}")
    return 0


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
    ap.add_argument("--stat-steps", type=int, default=0,
                    help="steps in the iteration-statistics pass "
                         "(default: the fixture's own step count, capped "
                         f"at {_STAT_CAP}).  Independent of --steps, so "
                         "reducing the timing run does not also shorten "
                         "and shift the window the iteration counts are "
                         "measured over")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--trace", dest="trace", action="store_true", default=None)
    ap.add_argument("--no-trace", dest="trace", action="store_false")
    ap.add_argument("--trace-steps", type=int, default=10)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--resummarise", type=Path, default=None,
                    help="recompute the per-fixture summary blocks of an "
                         "existing results file from its own recorded rows "
                         "and write it back.  The rows are the measurement; "
                         "`best`, `highlights` and `fixed_point_agreement` "
                         "are derived from them, so correcting how one is "
                         "derived does not need the machine time again")
    ap.add_argument("--label", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.resummarise:
        return _resummarise(args.resummarise)

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
        stat_steps = args.stat_steps or min(spec.steps, _STAT_CAP)
        fixture_rows: list[dict] = []
        for cfg in cfgs:
            row = _measure(spec, cfg, steps=steps, warmup=warmup,
                           stat_steps=stat_steps,
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
            "stat_steps": stat_steps,
            "expect": spec.expect,
            "regime_matches_expect_fraction": (
                sum(1 for r in fixture_rows if r.get("regime_matches_expect"))
                / max(sum(1 for r in fixture_rows if r.get("ok")), 1)),
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
        "stat_steps_override": args.stat_steps or None,
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
