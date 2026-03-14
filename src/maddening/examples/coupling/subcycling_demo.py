#!/usr/bin/env python
"""
Subcycling demo: mixed-timestep coupling groups.

Demonstrates coupling between nodes that run at different timesteps.
Without subcycling, all nodes in a coupling group must share the same
timestep.  With ``subcycling=True``, fast nodes take multiple sub-steps
per coupling iteration while slow nodes take one step.

Setup: a stiff spring at 1 kHz coupled to a soft spring at 100 Hz.
The fast spring needs a small dt for stability; the slow spring can
use a larger dt.  Subcycling lets them coexist in one coupling group.

Compares:
1. Both at fast rate (reference)
2. Subcycled coupling (mixed rates)
3. Subcycled with linear vs constant boundary interpolation

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python -m maddening.examples.coupling.subcycling_demo
"""

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


def build_uniform(dt=0.001, n_steps=1000):
    """Reference: both springs at the fast rate."""
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
        diagnostics=True,
    )
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
    print("=" * 60)
    print(f"  Stiff spring: k=200, c=2, m=0.5, dt={dt_fast}")
    print(f"  Soft spring:  k=20,  c=1, m=2,   dt={dt_slow}")
    print(f"  Subcycle ratio: {subcycle_ratio}x")
    print(f"  Simulation time: {sim_time}s")
    print()

    # Reference: both at fast rate
    print("--- Reference (both at dt={}) ---".format(dt_fast))
    gm_ref = build_uniform(dt=dt_fast)
    s_ref = gm_ref.run_scan(n_fast)
    ref_stiff = float(s_ref["stiff"]["position"])
    ref_soft = float(s_ref["soft"]["position"])
    print(f"  Stiff: {ref_stiff:.6f}")
    print(f"  Soft:  {ref_soft:.6f}")
    print()

    # Subcycled with linear interpolation
    print("--- Subcycled (linear interpolation) ---")
    gm_lin = build_subcycled(dt_fast, dt_slow, "linear")
    print(f"  Multi-rate: {gm_lin.is_multirate}")
    print(f"  Rate dividers: {gm_lin.rate_dividers}")
    s_lin = gm_lin.run_scan(n_slow)
    lin_stiff = float(s_lin["stiff"]["position"])
    lin_soft = float(s_lin["soft"]["position"])
    print(f"  Stiff: {lin_stiff:.6f}")
    print(f"  Soft:  {lin_soft:.6f}")
    diff_lin = abs(lin_stiff - ref_stiff) + abs(lin_soft - ref_soft)
    print(f"  Diff from reference: {diff_lin:.6f}")
    print()

    # Subcycled with constant interpolation
    print("--- Subcycled (constant interpolation) ---")
    gm_const = build_subcycled(dt_fast, dt_slow, "constant")
    s_const = gm_const.run_scan(n_slow)
    const_stiff = float(s_const["stiff"]["position"])
    const_soft = float(s_const["soft"]["position"])
    print(f"  Stiff: {const_stiff:.6f}")
    print(f"  Soft:  {const_soft:.6f}")
    diff_const = abs(const_stiff - ref_stiff) + abs(const_soft - ref_soft)
    print(f"  Diff from reference: {diff_const:.6f}")
    print()

    # Comparison
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Method':<30} {'Stiff':>10} {'Soft':>10} {'Diff':>10}")
    print("-" * 62)
    print(f"{'Reference (uniform dt)' :<30} {ref_stiff:10.4f} {ref_soft:10.4f} {'--':>10}")
    print(f"{'Subcycled (linear)' :<30} {lin_stiff:10.4f} {lin_soft:10.4f} {diff_lin:10.4f}")
    print(f"{'Subcycled (constant)' :<30} {const_stiff:10.4f} {const_soft:10.4f} {diff_const:10.4f}")
    print()

    # Verify finite results
    for name, val in [("lin_stiff", lin_stiff), ("lin_soft", lin_soft),
                      ("const_stiff", const_stiff), ("const_soft", const_soft)]:
        assert jnp.isfinite(jnp.array(val)), f"{name} is not finite!"

    # Linear interpolation should generally be closer to reference
    # than constant (but we don't assert this strictly)
    if diff_lin < diff_const:
        print("Check: linear interpolation is closer to reference (as expected).")
    else:
        print(f"Note: constant interpolation is closer to reference "
              f"(unusual but possible for this problem).")

    print("\nDone.")


if __name__ == "__main__":
    main()
