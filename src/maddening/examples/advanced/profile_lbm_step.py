#!/usr/bin/env python3
"""v0.2 #9 demo: profile a simulation step and save a Perfetto JSON.

Runs the v0.2 profiler against a small Heat+Coupling graph (a real
graph rather than a toy single-node smoke), saves the result as
``profile.json``, and prints how to open it.

The profile contains:
  * a top-level event for the run (n_steps × mean step time)
  * one event per node (in-isolation update timing)
  * a coupling-overhead event for the residual time
  * bottleneck + recommendation summary in ``otherData``

Drag-and-drop the saved JSON into https://ui.perfetto.dev to see
the flame-graph view.  The Perfetto UI is the same one TensorBoard's
"Trace Viewer" plugin uses, so you don't need TensorBoard for this
file format.

Usage:
    python profile_lbm_step.py
    python profile_lbm_step.py --n-steps 200 --out my_profile.json
    python profile_lbm_step.py --jax-trace      # also capture the
                                                # XLA-level trace dir

For real LBM you can swap _build_graph for your own graph; the
profiler is graph-shape-agnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.core.simulation.profiler import (
    JaxProfilerSession,
    profile_graph,
    profile_report_to_perfetto,
)
from maddening.nodes.heat import HeatNode


def _build_graph(n_cells: int) -> GraphManager:
    """Build a two-node Heat–Heat coupled graph at the requested size."""
    gm = GraphManager()
    gm.add_node(HeatNode(
        "rod_a", timestep=0.001, n_cells=n_cells, length=1.0,
        initial_temperature=100.0,
    ))
    gm.add_node(HeatNode(
        "rod_b", timestep=0.001, n_cells=n_cells, length=1.0,
        initial_temperature=0.0,
    ))
    gm.add_edge(
        "rod_a", "rod_b", "temperature", "left_temperature",
        transform=lambda T: T[-1],
    )
    gm.compile()
    return gm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-cells", type=int, default=64,
                        help="Cells per heat rod (more = more compute per step)")
    parser.add_argument("--n-steps", type=int, default=100,
                        help="Steps to benchmark (after 3-step warmup)")
    parser.add_argument("--out", default="profile.json",
                        help="Output Perfetto JSON path")
    parser.add_argument("--jax-trace", action="store_true",
                        help="Also capture an XLA-level jax.profiler trace "
                             "into a temp directory and print its path")
    args = parser.parse_args()

    gm = _build_graph(args.n_cells)

    if args.jax_trace:
        with JaxProfilerSession() as jax_sess:
            print(f"Capturing JAX XLA trace into {jax_sess.log_dir} ...")
            report = profile_graph(gm, n_steps=args.n_steps, n_warmup=3)
        jax_dir = jax_sess.log_dir
    else:
        report = profile_graph(gm, n_steps=args.n_steps, n_warmup=3)
        jax_dir = None

    perfetto = profile_report_to_perfetto(report)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(perfetto, indent=2), encoding="utf-8")

    print()
    print(report)
    print()
    print(f"Perfetto JSON written to:  {out_path.resolve()}")
    print(f"Open in:                   https://ui.perfetto.dev "
          f"(drag-and-drop the file)")
    if jax_dir is not None:
        print(f"JAX/XLA trace directory:   {jax_dir}")
        print(f"View with TensorBoard:     tensorboard --logdir={jax_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
