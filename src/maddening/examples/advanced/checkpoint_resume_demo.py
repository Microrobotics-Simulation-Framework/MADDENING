#!/usr/bin/env python3
"""Checkpoint a coupled run, resume it in a fresh graph, get the same numbers.

``gm.save_state(path)`` writes one ``.npz`` holding three things:

* every node's state;
* ``_meta``: the graph's internal bookkeeping -- here a coupling group's
  warm starts (the predictor's history of past solutions and the
  IQN-IMVJ secant columns carried across timesteps), plus its last
  diagnostics;
* ``gm.params``: the parameter values in force, so a calibrated graph
  resumes calibrated.

``load_state`` into a freshly built graph of the same structure restores
all three, and the resumed run is **bitwise identical** to the run that
never stopped -- the same positions and the same iteration count on
every step.  This example checks that, and checks that the resumed
graph's ``_meta`` is the saved one.  Then it shows what the ``_meta``
part is for: a "cold" restart that restores only the node states starts
its coupling group with an empty predictor history and no secant
columns.  It solves the same problem to the same tolerance, but it is a
different run -- in a transient it typically takes more passes on its
first step and stops at a different point inside the tolerance (both are
printed).  Only the full checkpoint is guaranteed to reproduce the run.

It also shows that a checkpoint refuses a graph it does not fit, and
changes nothing when it does.

For a checkpoint that travels (spot preemption, a shared filesystem),
``maddening.core.simulation.checkpoint.save_state_with_manifest`` adds a
sha256 manifest that ``load_state_with_manifest`` verifies; see
``docs/user_guide/cloud_resume.md``.

The checkpoint is written to a temporary directory, removed on exit.

Usage
-----
    python -m maddening.examples.advanced.checkpoint_resume_demo
    python -m maddening.examples.advanced.checkpoint_resume_demo --warmup 20 --steps 20
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

DT = 0.005
K = 4000.0          # k * dt^2 / m = 0.1: the coupled pair needs a few passes
GROUP = "A+B"
CALIBRATED_DAMPING = 30.0   # written into gm.params; the constructor says 25


def build() -> GraphManager:
    """Two masses joined by one spring, solved in a coupling group whose
    warm starts live in ``_meta``: a linear predictor and IQN-IMVJ
    acceleration reusing its secant columns for 3 timesteps."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("A", DT, stiffness=K, damping=25.0, mass=1.0,
                                 rest_length=1.0, initial_position=0.0))
    gm.add_node(SpringDamperNode("B", DT, stiffness=K, damping=25.0, mass=1.0,
                                 rest_length=-1.0, initial_position=5.0))
    gm.add_edge("A", "B", "position", "anchor_position")
    gm.add_edge("B", "A", "position", "anchor_position")
    gm.add_coupling_group(["A", "B"], max_iterations=20, tolerance=1e-6,
                          acceleration="iqn-imvj", jacobian_reuse=3,
                          predictor="linear")
    gm.compile()
    return gm


def advance(gm: GraphManager, n: int) -> tuple[np.ndarray, list[int]]:
    """Step *n* times; the positions after each step and the passes each took."""
    positions, passes = [], []
    for _ in range(n):
        gm.step()
        positions.append([float(gm.get_node_state(name)["position"]) for name in ("A", "B")])
        passes.append(int(gm.coupling_diagnostics()[GROUP]["iterations"]))
    return np.asarray(positions, dtype=np.float64), passes


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--warmup", type=int, default=100,
                        help="Steps run before the checkpoint (default 100)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Steps run after it, by each graph (default 50)")
    args = parser.parse_args(argv)
    if args.warmup < 2 or args.steps < 1:
        parser.error("--warmup must be at least 2 (the predictor keeps two past "
                     "solutions) and --steps at least 1")

    with tempfile.TemporaryDirectory(prefix="maddening_checkpoint_demo_") as tmp:
        run(args, Path(tmp))
    print()
    print("All checks passed.")
    return 0


def meta_rows(gm: GraphManager) -> dict[str, dict]:
    """``_meta`` as ``state_summary(include_meta=True)`` reports it: per
    entry its shape, dtype and min / max / mean (read-only)."""
    return {row["field"]: row for row in gm.state_summary(include_meta=True)
            if row["node"] == "_meta"}


def same_rows(a: dict, b: dict) -> bool:
    keys = ("shape", "dtype", "min", "max", "mean")
    return a.keys() == b.keys() and all(
        all(str(a[f][k]) == str(b[f][k]) for k in keys) for f in a)


def run(args, tmp: Path) -> None:
    section(f"Run {args.warmup} steps, calibrate a parameter, save a checkpoint")
    gm = build()
    gm.params["nodes"]["A"]["damping"] = jnp.float32(CALIBRATED_DAMPING)
    gm.run(args.warmup)
    path = gm.save_state(tmp / "run.npz")
    saved_meta = meta_rows(gm)
    print(f"  saved {path.name}: the node states, gm.params and "
          f"{len(saved_meta)} _meta entries:")
    for field, row in saved_meta.items():
        print(f"    {field:30s} {row['dtype']}{list(row['shape'])}  "
              f"max {row['max']:.4g}")
    pred_count = f"coupling_{GROUP}_pred_count"
    secants = [f"coupling_{GROUP}_V", f"coupling_{GROUP}_W"]
    assert saved_meta[pred_count]["max"] == 2           # two past solutions held

    section(f"Continue the original for {args.steps} steps")
    reference, ref_passes = advance(gm, args.steps)
    print(f"  passes per step: {ref_passes[:12]}{' ...' if args.steps > 12 else ''}")

    section("Resume from the checkpoint in a freshly built graph")
    resumed = build()
    damping_before = float(resumed.params["nodes"]["A"]["damping"])
    resumed.load_state(path)
    damping_after = float(resumed.params["nodes"]["A"]["damping"])
    print(f"  A.damping in gm.params: {damping_before:g} as built, "
          f"{damping_after:g} after load_state (the calibrated value)")
    assert damping_before == 25.0 and damping_after == CALIBRATED_DAMPING
    meta_restored = same_rows(meta_rows(resumed), saved_meta)
    print(f"  _meta restored as saved (warm starts included): {meta_restored}")
    assert meta_restored
    # Snapshot the restored node states, for the cold restart below.
    restored = {name: resumed.get_node_state(name) for name in resumed.node_names}
    trajectory, passes = advance(resumed, args.steps)
    identical = np.array_equal(trajectory, reference) and passes == ref_passes
    print(f"  passes per step: {passes[:12]}{' ...' if args.steps > 12 else ''}")
    print(f"  positions and passes bitwise identical to the uninterrupted run: {identical}")
    assert identical

    section("A cold restart: the node states only, no _meta")
    cold = build()
    cold.params = resumed.params              # the same calibrated values
    for name, fields in restored.items():
        cold.set_node_state(name, fields)
    cold_meta = meta_rows(cold)
    print(f"  its warm starts: predictor history {cold_meta[pred_count]['max']} "
          f"solutions (saved: 2), secant columns all zero: "
          f"{all(cold_meta[f]['max'] == cold_meta[f]['min'] == 0 for f in secants)}")
    assert cold_meta[pred_count]["max"] == 0
    assert all(cold_meta[f]["max"] == cold_meta[f]["min"] == 0 for f in secants)
    cold_trajectory, cold_passes = advance(cold, args.steps)
    gap = float(np.abs(cold_trajectory - reference).max())
    first = min(4, args.steps)
    print(f"  passes on the first {first} steps: cold {cold_passes[:first]}, "
          f"uninterrupted {ref_passes[:first]}")
    print(f"  largest position difference from the uninterrupted run: {gap:.2e} m")
    assert gap < 1e-3, gap          # the same problem, to the same tolerance
    if gap > 0 or cold_passes != ref_passes:
        print("  The same problem solved to the same tolerance, but a different run:"
              "\n  only the full checkpoint reproduces the uninterrupted one.")
    else:
        print("  This time the cold restart matched: the pair has settled, and the"
              "\n  warm starts change little.  Try a shorter --warmup.")

    section("A checkpoint refuses a graph it does not fit")
    other = GraphManager()
    other.add_node(SpringDamperNode("A", DT, stiffness=K, damping=25.0))
    other.compile()
    before = other.get_node_state("A")
    try:
        other.load_state(path)
    except ValueError as exc:
        print(f"  ValueError: {str(exc)[:160]}")
    else:
        raise AssertionError("a checkpoint loaded into a graph with a missing node")
    unchanged = all(np.array_equal(np.asarray(before[f]), np.asarray(v))
                    for f, v in other.get_node_state("A").items())
    print(f"  the graph is as it was: {unchanged}")
    assert unchanged


if __name__ == "__main__":
    sys.exit(main())
