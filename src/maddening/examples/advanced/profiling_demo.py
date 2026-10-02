#!/usr/bin/env python3
"""Profile a coupled graph: where a step's time goes, and what it compiles.

Builds two heat rods that meet end to end and solve their shared
boundary in a coupling group, then calls
:func:`~maddening.core.simulation.profiler.profile_graph` with every
measurement switched on and reads the :class:`ProfileReport` back:

* **timings** -- JIT compile, mean / median / p95 step, and the
  *dispatch floor* (a jitted identity on the same state: what a step
  costs before any physics runs);
* **coupling** (``measure_coupling=True``) -- iterations per step from
  the always-on ``_meta`` diagnostics, and the overhead *measured* by
  recompiling the graph with the group capped at one pass, with its
  standard error and the cost per extra iteration;
* **bottleneck and recommendations** -- the profiler's verdict;
* **compile counts** (``counts=True``, ``count_scan_steps=N``) --
  retraces, jaxpr primitives and lowered HLO ops for the step and for a
  ``run_scan`` program.  Integers, not timings: the same graph gives the
  same numbers on any machine, which this example checks by counting a
  second, freshly built graph;
* **a trace** (``trace=True``) -- a short ``jax.profiler`` recording,
  summarised as a :class:`TraceSummary`.  On a GPU it attributes kernel
  time to the graph's ``node:<name>`` / ``coupling:*`` scopes; the CPU
  backend records host events only, so there the device columns are
  zero and the host dispatch time is the number it measures;
* **a Perfetto export** -- :func:`profile_report_to_perfetto`, written
  as JSON to drag into https://ui.perfetto.dev.

Everything is written to a temporary directory, removed on exit, unless
``--out-dir`` names one to keep.  Nothing is written into the package.

The grid spacing is fixed at 1 mm, so every ``--n-cells`` runs the same
Fourier number and the group needs a similar number of passes (four to
seven here); what grows with the size is the work in each pass.  At the
default size the extra passes are most of the step, and the overhead is
tens of standard errors from zero.  At ``--n-cells 64`` a pass costs
about as much as dispatching it, and the overhead is a few standard
errors from zero at most, or below resolution: the report says which.

See also ``profile_lbm_step.py``, which saves the same Perfetto JSON from
an uncoupled graph and can capture an XLA-level trace for TensorBoard,
and ``docs/developer_guide/profiling.md`` for what each number means.

Usage
-----
    python -m maddening.examples.advanced.profiling_demo
    python -m maddening.examples.advanced.profiling_demo --n-cells 64 --n-steps 20
    python -m maddening.examples.advanced.profiling_demo --out-dir ./profile_out
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from maddening.core.graph_manager import GraphManager
from maddening.core.simulation.profiler import (
    compile_counts,
    profile_graph,
    profile_report_to_perfetto,
)
from maddening.nodes.heat import HeatNode

DX = 1e-3            # grid spacing [m], fixed for every --n-cells
ALPHA = 1e-4         # thermal diffusivity [m^2/s]
FOURIER = 0.2        # dt * alpha / dx^2: inside the explicit stencil's limit
DT = FOURIER * DX ** 2 / ALPHA   # 2 ms
GROUP = "cold+hot"


def build_graph(n_cells: int) -> GraphManager:
    """Two rods of *n_cells* 1-mm cells joined end to end in a coupling group.

    Each rod's free end is held by the other's interface cell: ``hot``'s
    right end reads ``cold``'s first cell and ``cold``'s left end reads
    ``hot``'s last, iterated to ``tolerance`` inside every step.
    """
    gm = GraphManager()
    for name, temperature in (("hot", 100.0), ("cold", 0.0)):
        gm.add_node(HeatNode(
            name, timestep=DT, n_cells=n_cells, length=n_cells * DX,
            thermal_diffusivity=ALPHA, initial_temperature=temperature,
        ))
    gm.add_edge("hot", "cold", "temperature", "left_temperature",
                transform=lambda T: T[-1])
    gm.add_edge("cold", "hot", "temperature", "right_temperature",
                transform=lambda T: T[0])
    gm.add_coupling_group(["hot", "cold"], max_iterations=30, tolerance=1e-7)
    gm.compile()
    return gm


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-cells", type=int, default=32768,
                        help="Cells per rod (default 32768: the coupling cost is "
                             "resolved; 64 is dispatch-bound)")
    parser.add_argument("--n-steps", type=int, default=100,
                        help="Timed steps per window (default 100)")
    parser.add_argument("--trace-steps", type=int, default=10,
                        help="Steps recorded by jax.profiler (default 10)")
    parser.add_argument("--scan-steps", type=int, default=16,
                        help="Length of the run_scan program whose compile "
                             "counts are taken (default 16)")
    parser.add_argument("--out-dir", default=None,
                        help="Keep the trace and the Perfetto JSON here "
                             "(default: a temporary directory, removed on exit)")
    args = parser.parse_args(argv)

    if args.out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="maddening_profiling_demo_"))
        keep = False
    else:
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        keep = True
    try:
        run(args, out_dir)
    finally:
        if not keep:
            shutil.rmtree(out_dir, ignore_errors=True)
    if not keep:
        print(f"(Trace and JSON were written to a temporary directory, now "
              f"removed; pass --out-dir DIR to keep them.)")
    print()
    print("All checks passed.")
    return 0


def run(args, out_dir: Path) -> None:
    section(f"The graph: two {args.n_cells}-cell rods in one coupling group")
    gm = build_graph(args.n_cells)
    gm.print_graph()

    section("profile_graph with every measurement on")
    trace_dir = out_dir / "jax_trace"
    report = profile_graph(
        gm,
        n_steps=args.n_steps,
        n_warmup=3,
        measure_coupling=True,          # recompile capped at one pass, time it
        counts=True,                    # retraces, jaxpr primitives, HLO ops
        count_scan_steps=args.scan_steps,
        trace=True,                     # a short jax.profiler recording
        trace_steps=args.trace_steps,
        trace_dir=str(trace_dir),
    )
    print(report)

    # ------------------------------------------------------------------
    section("Reading the report: coupling")
    stats = report.coupling_iter_stats[GROUP]
    extra = stats["mean"] - 1.0
    print(f"  {GROUP}: {stats['mean']:.2f} passes per step on average "
          f"(min {stats['min']}, max {stats['max']}, cap {stats['cap']}), "
          f"over {stats['n']} steps")
    print(f"  converged on {stats['converged_fraction']:.0%} of steps, "
          f"at the cap on {stats['at_cap_fraction']:.0%}")
    assert stats["min"] >= 2, "the coupling group should iterate, not stop after one pass"
    assert stats["converged_fraction"] == 1.0 and stats["at_cap_fraction"] == 0.0

    overhead, se = report.coupling_overhead_ms, report.coupling_overhead_se_ms
    print()
    print(f"  real step:            {report.mean_step_ms:8.3f} ms")
    print(f"  capped at one pass:   {report.one_iteration_step_ms:8.3f} ms")
    print(f"  coupling overhead:    {overhead:8.3f} +- {se:.3f} ms  "
          f"(the difference; {report.coupling_overhead_method})")
    print(f"  per extra iteration:  {report.coupling_per_iteration_ms:8.3f} ms  "
          f"(overhead / {extra:.2f} extra passes)")
    assert report.coupling_overhead_method == "measured"
    # The per-iteration cost is the overhead shared over the extra passes.
    assert abs(report.coupling_per_iteration_ms * extra - overhead) <= 1e-9 + 1e-6 * abs(overhead)
    if abs(overhead) > se:
        print(f"  The overhead is {abs(overhead) / se:.1f} standard errors from zero, "
              f"{overhead / report.mean_step_ms:.0%} of the step.  The standard "
              f"error is a lower bound on the uncertainty -- it ignores drift "
              f"between the two windows -- so on a shared machine repeat the run "
              f"before trusting a small multiple.")
    else:
        print("  The overhead is within one standard error of zero: below this "
              "run's resolution (the report says so too).  The step is "
              "dominated by dispatch, not by the extra passes; raise --n-cells "
              "to see them.")

    section("Reading the report: bottleneck and recommendations")
    print(f"  bottleneck: {report.bottleneck or '(none named)'}")
    for rec in report.recommendations or ["(none)"]:
        print(f"  - {rec}")
    # The verdicts follow from the timings above by stated rules
    # (profile_graph's source); check the coupling one is consistent.
    if report.bottleneck.startswith("Coupling overhead"):
        assert overhead > 0.3 * report.mean_step_ms
    print(f"  dispatch floor {report.dispatch_floor_ms:.3f} ms of a "
          f"{report.mean_step_ms:.3f} ms step: "
          + ("more than half -- the step is launch-bound, batch steps with run_scan"
             if report.dispatch_floor_ms > 0.5 * report.mean_step_ms
             else "under half -- the step is doing real work"))

    # ------------------------------------------------------------------
    section("Compile counts: integers, the same on every machine")
    counts = report.counts
    assert counts is not None
    print(f"  step:  {counts.retrace_count} trace, "
          f"{counts.jaxpr_primitive_count} jaxpr primitives, "
          f"{counts.hlo_op_count} lowered HLO ops")
    print(f"  run_scan({counts.scan_steps}): {counts.scan_retrace_count} trace, "
          f"{counts.scan_jaxpr_primitive_count} jaxpr primitives, "
          f"{counts.scan_hlo_op_count} lowered HLO ops")
    assert counts.retrace_count == 1, "an unexpected retrace costs a full compile per run"
    assert counts.scan_retrace_count == 1 and counts.scan_steps == args.scan_steps
    # A second graph, built from scratch: the step's counts must agree
    # exactly.  (compile_counts warms it 4 steps first, the floor at which
    # a late retrace would have shown up.)
    fresh = compile_counts(build_graph(args.n_cells))
    same = (fresh.retrace_count, fresh.jaxpr_primitive_count, fresh.hlo_op_count) == (
        counts.retrace_count, counts.jaxpr_primitive_count, counts.hlo_op_count)
    print(f"  a freshly built graph counts {fresh.retrace_count} / "
          f"{fresh.jaxpr_primitive_count} / {fresh.hlo_op_count}: "
          + ("identical" if same else "DIFFERENT"))
    assert same, (fresh, counts)
    print("  This is why the compile-time regression gate "
          "(scripts/compile_counts.py) reads these and never the clock.")

    # ------------------------------------------------------------------
    section("The trace summary")
    tr = report.trace
    assert tr is not None
    traces = glob.glob(str(trace_dir / "**" / "perfetto_trace.json.gz"), recursive=True)
    print(f"  {tr.n_steps} steps recorded into {tr.log_dir}")
    print(f"  host dispatch of the compiled step: {tr.host_dispatch_ms_per_step:.3f} ms/step")
    print(f"  device kernels: {tr.n_kernels_per_step:.0f}/step, "
          f"busy {tr.device_busy_ms_per_step:.3f} ms/step "
          f"({tr.device_busy_fraction:.0%} of the wall step)")
    assert tr.n_steps == args.trace_steps and traces, "the trace was not written"
    assert Path(tr.log_dir) == trace_dir
    assert tr.host_dispatch_ms_per_step > 0.0
    if "cpu" in report.device.lower():
        print("  The CPU backend records host events only, so the device columns "
              "are zero here.  On a GPU the same call fills them, and the kernel "
              "time by scope separates node:hot / node:cold from the coupling loop.")
    else:
        for scope, ms in sorted(tr.device_ms_by_scope.items(), key=lambda kv: -kv[1]):
            print(f"    {scope:32s} {ms:8.3f} ms/step")

    # ------------------------------------------------------------------
    section("Perfetto export")
    perfetto = profile_report_to_perfetto(report)
    json_path = out_dir / "profile.json"
    json_path.write_text(json.dumps(perfetto, indent=2), encoding="utf-8")
    names = [e["name"] for e in json.loads(json_path.read_text())["traceEvents"]]
    print(f"  wrote {json_path} ({len(names)} events: {', '.join(names)})")
    assert names[0] == f"run x{args.n_steps}"
    assert {"hot.update", "cold.update"} <= set(names)
    # A coupling slice is drawn only for a positive overhead: a negative
    # duration is not a valid Perfetto slice.
    assert ("coupling_overhead" in names) == (overhead > 0)
    print("  Open it at https://ui.perfetto.dev (drag and drop the file).  The "
          "per-node bars are each node timed alone and laid end to end: a "
          "reconstruction, not a recording.")


if __name__ == "__main__":
    sys.exit(main())
