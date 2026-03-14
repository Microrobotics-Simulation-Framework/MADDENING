#!/usr/bin/env python
"""
Subcycling demo: mixed-timestep coupling groups.

**The problem**: different physics run at different timescales.  A stiff
spring needs a small dt (0.001s) for numerical stability, but a soft
spring can use a larger dt (0.01s).  Without subcycling, you'd either
run everything at the fast rate (wasting compute on the slow physics)
or put them in separate coupling groups (losing tight coupling).

**The solution**: ``subcycling=True`` lets fast and slow nodes coexist
in one coupling group.  The fast node takes 10 sub-steps per coupling
iteration while the slow node takes 1 step.  Both advance the same
amount of simulation time per pass.

**Boundary interpolation**: during the fast node's sub-steps, its
boundary conditions from the slow node are interpolated in time:
- ``"linear"``: smooth ramp from previous to current iteration values
- ``"constant"``: step function (uses end-of-iteration values throughout)

Usage
-----
    python -m maddening.examples.coupling.subcycling_demo
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


def build_uniform(dt=0.001):
    """Reference: both springs at the fast rate (expensive but accurate)."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode(
        name="stiff", timestep=dt,
        stiffness=200.0, damping=2.0, mass=0.5,
        rest_length=1.0, initial_position=0.0,
    ))
    gm.add_node(SpringDamperNode(
        name="soft", timestep=dt,
        stiffness=20.0, damping=1.0, mass=2.0,
        rest_length=1.0, initial_position=4.0,
    ))
    gm.add_edge("stiff", "soft", "position", "anchor_position")
    gm.add_edge("soft", "stiff", "position", "anchor_position")
    gm.add_coupling_group(
        ["stiff", "soft"],
        max_iterations=20, tolerance=1e-10,
    )
    gm.compile()
    return gm


def build_subcycled(dt_fast=0.001, dt_slow=0.01,
                    interpolation="linear"):
    """Subcycled: stiff spring at fast rate, soft at slow rate."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode(
        name="stiff", timestep=dt_fast,
        stiffness=200.0, damping=2.0, mass=0.5,
        rest_length=1.0, initial_position=0.0,
    ))
    gm.add_node(SpringDamperNode(
        name="soft", timestep=dt_slow,
        stiffness=20.0, damping=1.0, mass=2.0,
        rest_length=1.0, initial_position=4.0,
    ))
    gm.add_edge("stiff", "soft", "position", "anchor_position")
    gm.add_edge("soft", "stiff", "position", "anchor_position")
    gm.add_coupling_group(
        ["stiff", "soft"],
        max_iterations=20, tolerance=1e-10,
        subcycling=True,
        boundary_interpolation=interpolation,
    )
    gm.compile()
    return gm


def build_staggered(dt_fast=0.001, dt_slow=0.01):
    """Staggered: no coupling group, just back-edge lag.  Cheapest but least accurate."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode(
        name="stiff", timestep=dt_fast,
        stiffness=200.0, damping=2.0, mass=0.5,
        rest_length=1.0, initial_position=0.0,
    ))
    gm.add_node(SpringDamperNode(
        name="soft", timestep=dt_slow,
        stiffness=20.0, damping=1.0, mass=2.0,
        rest_length=1.0, initial_position=4.0,
    ))
    gm.add_edge("stiff", "soft", "position", "anchor_position")
    gm.add_edge("soft", "stiff", "position", "anchor_position")
    # No coupling group -- staggered (back-edge lag)
    gm.compile()
    return gm


def main() -> None:
    dt_fast = 0.001
    dt_slow = 0.01
    sim_time = 1.0
    n_fast = int(sim_time / dt_fast)
    n_slow = int(sim_time / dt_slow)
    subcycle_ratio = round(dt_slow / dt_fast)

    print("Subcycling Demo: Mixed-Timestep Coupling")
    print("=" * 65)
    print()
    print("  Stiff spring: k=200, c=2, m=0.5 -- needs dt=0.001 for stability")
    print("  Soft spring:  k=20,  c=1, m=2   -- can use dt=0.01")
    print(f"  Subcycle ratio: {subcycle_ratio}x (stiff takes {subcycle_ratio}")
    print(f"    sub-steps for every 1 step of soft)")
    print(f"  Simulation time: {sim_time}s")
    print()

    # Reference
    print("--- Reference: both at dt=0.001 ({} steps) ---".format(n_fast))
    gm_ref = build_uniform(dt=dt_fast)
    s_ref = gm_ref.run_scan(n_fast)
    ref_stiff = float(s_ref["stiff"]["position"])
    ref_soft = float(s_ref["soft"]["position"])
    print(f"  Stiff: {ref_stiff:.6f},  Soft: {ref_soft:.6f}")
    print(f"  (This is the 'ground truth' -- expensive but accurate)")
    print()

    # Staggered (no coupling)
    print("--- Staggered: no coupling, multi-rate ({} slow steps) ---".format(n_slow))
    gm_stag = build_staggered(dt_fast, dt_slow)
    s_stag = gm_stag.run_scan(n_fast)  # runs at base dt
    stag_stiff = float(s_stag["stiff"]["position"])
    stag_soft = float(s_stag["soft"]["position"])
    diff_stag = abs(stag_stiff - ref_stiff) + abs(stag_soft - ref_soft)
    print(f"  Stiff: {stag_stiff:.6f},  Soft: {stag_soft:.6f}")
    print(f"  Diff from reference: {diff_stag:.4f}")
    print(f"  (Cheap but inaccurate -- back-edge lag introduces error)")
    print()

    # Subcycled linear
    print("--- Subcycled: linear interpolation ({} macro steps) ---".format(n_slow))
    gm_lin = build_subcycled(dt_fast, dt_slow, "linear")
    s_lin = gm_lin.run_scan(n_slow)
    lin_stiff = float(s_lin["stiff"]["position"])
    lin_soft = float(s_lin["soft"]["position"])
    diff_lin = abs(lin_stiff - ref_stiff) + abs(lin_soft - ref_soft)
    print(f"  Stiff: {lin_stiff:.6f},  Soft: {lin_soft:.6f}")
    print(f"  Diff from reference: {diff_lin:.4f}")
    print(f"  (Good balance of accuracy and cost)")
    print()

    # Subcycled constant
    print("--- Subcycled: constant interpolation ({} macro steps) ---".format(n_slow))
    gm_const = build_subcycled(dt_fast, dt_slow, "constant")
    s_const = gm_const.run_scan(n_slow)
    const_stiff = float(s_const["stiff"]["position"])
    const_soft = float(s_const["soft"]["position"])
    diff_const = abs(const_stiff - ref_stiff) + abs(const_soft - ref_soft)
    print(f"  Stiff: {const_stiff:.6f},  Soft: {const_soft:.6f}")
    print(f"  Diff from reference: {diff_const:.4f}")
    print()

    # Summary
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"{'Method':<30} {'Steps':>6} {'Diff':>10}")
    print("-" * 48)
    print(f"{'Reference (both fast)' :<30} {n_fast:6d} {'--':>10}")
    print(f"{'Staggered (no coupling)' :<30} {n_fast:6d} {diff_stag:10.4f}")
    print(f"{'Subcycled (linear)' :<30} {n_slow:6d} {diff_lin:10.4f}")
    print(f"{'Subcycled (constant)' :<30} {n_slow:6d} {diff_const:10.4f}")
    print()
    print("Takeaway:")
    print(f"  - Subcycling uses {n_slow} macro steps instead of {n_fast} base steps")
    print(f"    ({subcycle_ratio}x fewer graph-level steps).")
    print("  - The error comes from the slow node's time discretisation,")
    print("    not from the coupling iteration (which fully converges).")
    print("  - Linear interpolation gives smoother boundary conditions")
    print("    during sub-steps; constant is simpler but less accurate.")


if __name__ == "__main__":
    main()
