#!/usr/bin/env python3
"""Profile a simulation step and save a Perfetto JSON.

Runs :func:`~maddening.core.simulation.profiler.profile_graph` on a
small graph of two heat rods joined by a one-way edge, saves the
result as ``profile.json`` in the current directory, and prints the
report and how to open the file.  (The file name is historical: the
graph it builds is not a Lattice-Boltzmann one.)

The profile contains:
  * a top-level ``run xN`` event spanning the timed steps, whose
    ``args`` carry the mean step time, the throughput and the
    bottleneck verdict;
  * one ``<node>.update`` event per node -- each node's update timed in
    isolation and laid out in series, so the bars are a reconstruction,
    not a recording;
  * a ``coupling_overhead`` event only when the graph has coupling
    groups and the measured overhead is positive (this graph has none,
    so its trace has none);
  * the recommendations and the JIT compile time in ``otherData``.

Drag-and-drop the saved JSON into https://ui.perfetto.dev to see the
timeline.

Usage:
    python -m maddening.examples.advanced.profile_lbm_step
    python -m maddening.examples.advanced.profile_lbm_step --n-steps 200 --out my_profile.json
    python -m maddening.examples.advanced.profile_lbm_step --jax-trace
        # also capture an XLA-level jax.profiler trace directory

To profile your own graph, swap ``_build_graph`` for it; the profiler
does not depend on the graph's shape.

See also ``profiling_demo.py``, which profiles a *coupled* graph and
reads the rest of the report: the measured coupling overhead and cost per
iteration, the deterministic compile counts, and the ``trace=True``
summary.
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
