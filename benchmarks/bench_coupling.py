#!/usr/bin/env python
"""Coupled-step benchmark: coupling group vs staggered baseline.

Measures the warm per-step cost of a graph with its coupling group(s)
enabled and with them disabled (back-edge staggering), reports the
iterations the groups actually use against ``max_iterations``, the
measured per-iteration coupling cost, the dispatch floor, and — with
``--trace`` — device kernel time by named scope and the device-busy
fraction (launch-bound vs compute-bound).  Checks the TODO_perf.md
PERF-1 acceptance (``--acceptance-ms``, default 30 ms/step).

Graphs::

    --graph mime-ar4      MIME AR4 + helical-UMR experiment
                          (``--experiment DIR`` with physics/params.py and
                          physics/setup.py; needs the ``mime`` package)
    --graph springs       two coupled SpringDamperNodes (MADDENING only)
    --graph heat-chain    N coupled HeatNodes with interface overrides
                          (``--n-nodes``, ``--n-cells``)

Examples::

    python benchmarks/bench_coupling.py --graph mime-ar4 \\
        --experiment ../MIME/experiments/ar4_helical_drive --trace --json out.json
    JAX_PLATFORMS=cpu python benchmarks/bench_coupling.py --graph springs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# XLA setup before JAX is imported (mirrors MIME's AR4 driver): don't
# grab the whole GPU, skip autotuning for the tiny matrices these graphs
# have.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from maddening.core.graph_manager import GraphManager  # noqa: E402
from maddening.core.simulation.profiler import ProfileReport, profile_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Graph factories: each returns (gm, external_inputs)
# ---------------------------------------------------------------------------


def _springs(coupling: bool, **_):
    from maddening.nodes.spring import SpringDamperNode

    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 1e-3, stiffness=30.0, damping=2.0, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 1e-3, stiffness=30.0, damping=2.0, initial_position=3.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    if coupling:
        gm.add_coupling_group(["a", "b"], max_iterations=20, tolerance=1e-8)
    gm.compile()
    return gm, None


def _heat_chain(coupling: bool, n_nodes: int = 4, n_cells: int = 64, **_):
    from maddening.nodes.heat import HeatNode

    gm = GraphManager()
    names = [f"rod{i}" for i in range(n_nodes)]
    for i, n in enumerate(names):
        gm.add_node(HeatNode(n, 1e-5, n_cells=n_cells, thermal_diffusivity=0.1,
                             initial_temperature=300.0 + 20.0 * i))
    from maddening.core.transforms import extract_first, extract_last
    for a, b in zip(names[:-1], names[1:]):
        gm.add_edge(a, b, "temperature", "left_temperature", transform=extract_last)
        gm.add_edge(b, a, "temperature", "right_temperature", transform=extract_first)
    if coupling:
        gm.add_coupling_group(names, max_iterations=20, tolerance=1e-6)
    gm.compile()
    return gm, None


def _mime_ar4(coupling: bool, experiment: Path | None = None, **_):
    if experiment is None:
        raise SystemExit("--graph mime-ar4 needs --experiment DIR")
    experiment = experiment.resolve()
    ns: dict = {}
    exec((experiment / "physics" / "params.py").read_text(), ns)
    params = {k: v for k, v in ns.items() if not k.startswith("_")}
    if not coupling:
        params["USE_COUPLING_GROUP"] = False
    import importlib.util
    sys.path.insert(0, str(experiment / "physics"))
    try:
        spec = importlib.util.spec_from_file_location(
            "_bench_setup", str(experiment / "physics" / "setup.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    import inspect
    sig = inspect.signature(mod.build_graph)
    gm = mod.build_graph(params, str(experiment)) if len(sig.parameters) >= 2 \
        else mod.build_graph(params)
    gm.compile()
    arm = gm._nodes["arm"].node
    ext = {
        "motor": {"commanded_velocity": jnp.float32(2.0 * np.pi * 10.0)},
        "arm": {"commanded_joint_torques": jnp.zeros((arm._num_joints,), jnp.float32)},
    }
    return gm, ext


GRAPHS = {"springs": _springs, "heat-chain": _heat_chain, "mime-ar4": _mime_ar4}


# ---------------------------------------------------------------------------


def _row(label: str, rep: ProfileReport) -> dict:
    return {
        "variant": label,
        "device": rep.device,
        "jit_compile_ms": rep.jit_compile_ms,
        "mean_step_ms": rep.mean_step_ms,
        "median_step_ms": rep.median_step_ms,
        "p95_step_ms": rep.p95_step_ms,
        "dispatch_floor_ms": rep.dispatch_floor_ms,
        "one_iteration_step_ms": rep.one_iteration_step_ms,
        "coupling_overhead_ms": rep.coupling_overhead_ms,
        "coupling_overhead_se_ms": rep.coupling_overhead_se_ms,
        "coupling_overhead_method": rep.coupling_overhead_method,
        "coupling_per_iteration_ms": rep.coupling_per_iteration_ms,
        "coupling_iter_stats": rep.coupling_iter_stats,
        "node_times_ms": rep.node_times_ms,
        "trace": None if rep.trace is None else {
            "device_busy_ms_per_step": rep.trace.device_busy_ms_per_step,
            "device_busy_fraction": rep.trace.device_busy_fraction,
            "n_kernels_per_step": rep.trace.n_kernels_per_step,
            "host_dispatch_ms_per_step": rep.trace.host_dispatch_ms_per_step,
            "device_ms_by_scope": rep.trace.device_ms_by_scope,
            "top_kernels": rep.trace.top_kernels,
        },
        "recommendations": rep.recommendations,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", choices=sorted(GRAPHS), default="springs")
    ap.add_argument("--experiment", type=Path, default=None)
    ap.add_argument("--n-nodes", type=int, default=4)
    ap.add_argument("--n-cells", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--trace", action="store_true", help="jax.profiler scope attribution")
    ap.add_argument("--trace-steps", type=int, default=20)
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the coupling-disabled variant")
    ap.add_argument("--acceptance-ms", type=float, default=30.0)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--label", default="", help="free-text tag stored in the JSON")
    args = ap.parse_args()

    factory = GRAPHS[args.graph]
    kw = dict(experiment=args.experiment, n_nodes=args.n_nodes, n_cells=args.n_cells)
    variants = [("coupled", True)] + ([] if args.no_baseline else [("baseline", False)])
    rows = []
    reports = {}
    for label, coupling in variants:
        t0 = time.time()
        gm, ext = factory(coupling, **kw)
        build_s = time.time() - t0
        rep = profile_graph(gm, n_steps=args.steps, n_warmup=args.warmup,
                            external_inputs=ext, trace=args.trace,
                            trace_steps=args.trace_steps)
        rep.graph_name = f"{args.graph} [{label}]"
        reports[label] = rep
        row = _row(label, rep)
        row["build_s"] = build_s
        rows.append(row)
        print(rep)
        print()

    coupled = reports["coupled"]
    summary = {
        "graph": args.graph, "label": args.label, "steps": args.steps,
        "device": coupled.device, "jax": jax.__version__,
        "acceptance_ms": args.acceptance_ms,
        "coupled_mean_step_ms": coupled.mean_step_ms,
        "accepted": coupled.mean_step_ms <= args.acceptance_ms,
        "variants": rows,
    }
    if "baseline" in reports:
        base = reports["baseline"]
        summary["baseline_mean_step_ms"] = base.mean_step_ms
        summary["coupling_cost_vs_baseline_ms"] = coupled.mean_step_ms - base.mean_step_ms
    print("=== Summary ===")
    print(f"  coupled:  {coupled.mean_step_ms:8.2f} ms/step  "
          f"(dispatch floor {coupled.dispatch_floor_ms:.2f}, "
          f"one-iteration {coupled.one_iteration_step_ms:.2f})")
    if "baseline" in reports:
        print(f"  baseline: {base.mean_step_ms:8.2f} ms/step  "
              f"-> coupling costs {summary['coupling_cost_vs_baseline_ms']:.2f} ms/step")
    for key, st in coupled.coupling_iter_stats.items():
        print(f"  {key}: {st['mean']:.2f} iterations mean of cap {st['cap']} "
              f"(at cap {st['at_cap_fraction'] * 100:.0f}%, "
              f"converged {st['converged_fraction'] * 100:.0f}%)")
    verdict = "PASS" if summary["accepted"] else "FAIL"
    print(f"  PERF-1 acceptance <= {args.acceptance_ms:.0f} ms/step: {verdict}")
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2, default=float))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
