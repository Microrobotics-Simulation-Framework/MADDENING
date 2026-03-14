#!/usr/bin/env python
"""
Subcycling demo: mixed-timestep coupling groups.

**The problem**: different physics run at different timescales.  A stiff
spring needs a small dt (0.001s) for numerical stability, but a soft
spring can use a larger dt (0.01s).  Without subcycling, you can't put
them in the same coupling group (different timesteps are rejected).

**The solution**: ``subcycling=True`` lets fast and slow nodes coexist
in one coupling group.  The fast node takes 10 sub-steps per coupling
iteration while the slow node takes 1 step.  Both advance the same
amount of simulation time per pass (the slow node's dt).

**Key tradeoff**: subcycling exchanges coupling information once per
macro step (at the slow rate), while multi-rate staggering exchanges
it at every fast step.  For well-conditioned problems with small dt,
staggering may actually be more accurate because it communicates more
frequently.  Subcycling's advantage is that it CONVERGES the coupling
within each macro step (via iteration), which matters for strongly
coupled or marginally stable problems.

Usage
-----
    python -m maddening.examples.coupling.subcycling_demo
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


def build_coupled(dt, coupling=True, subcycling=False):
    """Build a bidirectional spring graph."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode(
        name="stiff", timestep=0.001 if subcycling else dt,
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
    if coupling:
        kwargs = dict(max_iterations=20, tolerance=1e-10)
        if subcycling:
            kwargs["subcycling"] = True
        gm.add_coupling_group(["stiff", "soft"], **kwargs)
    gm.compile()
    return gm


def main() -> None:
    dt_fast = 0.001
    dt_slow = 0.01
    sim_time = 1.0
    n_fast = int(sim_time / dt_fast)
    n_slow = int(sim_time / dt_slow)
    ratio = round(dt_slow / dt_fast)

    print("Subcycling Demo: Mixed-Timestep Coupling")
    print("=" * 65)
    print()
    print("  Stiff spring: k=200, c=2, m=0.5")
    print("  Soft spring:  k=20,  c=1, m=2")
    print(f"  Subcycle ratio: {ratio}x (stiff takes {ratio} sub-steps")
    print(f"  for every 1 step of soft)")
    print(f"  Simulation time: {sim_time}s")
    print()

    # 1. Reference: both at fast rate, with coupling
    print("1. Reference: both at dt=0.001, with coupling ({} steps)".format(
        n_fast))
    gm_ref = build_coupled(dt=dt_fast, coupling=True)
    s_ref = gm_ref.run_scan(n_fast)
    ref_stiff = float(s_ref["stiff"]["position"])
    ref_soft = float(s_ref["soft"]["position"])
    print(f"   Stiff={ref_stiff:.6f}  Soft={ref_soft:.6f}")
    print(f"   (Ground truth: both nodes at fast rate, coupling converged)")
    print()

    # 2. Both at slow rate, with coupling (the baseline subcycling improves on)
    print("2. Coarse baseline: both at dt=0.01, with coupling ({} steps)".format(
        n_slow))
    gm_coarse = build_coupled(dt=dt_slow, coupling=True)
    s_coarse = gm_coarse.run_scan(n_slow)
    coarse_stiff = float(s_coarse["stiff"]["position"])
    coarse_soft = float(s_coarse["soft"]["position"])
    diff_coarse = abs(coarse_stiff - ref_stiff) + abs(coarse_soft - ref_soft)
    print(f"   Stiff={coarse_stiff:.6f}  Soft={coarse_soft:.6f}")
    print(f"   Diff from reference: {diff_coarse:.4f}")
    print(f"   (The stiff spring loses accuracy at the slow dt)")
    print()

    # 3. Subcycled: stiff at fast dt, soft at slow dt
    print("3. Subcycled: stiff at dt=0.001, soft at dt=0.01 ({} macro steps)".format(
        n_slow))
    gm_sub = build_coupled(dt=dt_slow, coupling=True, subcycling=True)
    s_sub = gm_sub.run_scan(n_slow)
    sub_stiff = float(s_sub["stiff"]["position"])
    sub_soft = float(s_sub["soft"]["position"])
    diff_sub = abs(sub_stiff - ref_stiff) + abs(sub_soft - ref_soft)
    print(f"   Stiff={sub_stiff:.6f}  Soft={sub_soft:.6f}")
    print(f"   Diff from reference: {diff_sub:.4f}")
    print(f"   (Stiff node uses its own small dt via sub-stepping)")
    print()

    # Summary
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"{'Method':<45} {'Macro steps':>12} {'Diff':>8}")
    print("-" * 67)
    print(f"{'1. Reference (both fast, coupled)' :<45} {n_fast:12d} {'--':>8}")
    print(f"{'2. Coarse (both slow, coupled)' :<45} {n_slow:12d} {diff_coarse:8.4f}")
    print(f"{'3. Subcycled (mixed dt, coupled)' :<45} {n_slow:12d} {diff_sub:8.4f}")
    print()

    if diff_sub < diff_coarse:
        print("Result: subcycling IMPROVES accuracy over the coarse baseline")
        print(f"  by {diff_coarse - diff_sub:.4f} ({diff_coarse/max(diff_sub,1e-12):.1f}x).")
        print("  The stiff spring benefits from its native small dt while")
        print("  the soft spring runs at its natural slow rate.")
    else:
        print(f"Result: subcycled diff ({diff_sub:.4f}) vs coarse ({diff_coarse:.4f}).")
        print("  For this well-conditioned problem, the coarse baseline")
        print("  is already accurate.  Subcycling's value is in problems")
        print("  where the fast node is UNSTABLE at the slow dt.")

    print()
    print("When to use subcycling:")
    print("  - When fast physics REQUIRES a small dt (stability constraint)")
    print("    but slow physics works fine at a larger dt.")
    print("  - When you want converged coupling (via iteration) between")
    print("    nodes with different timesteps.")
    print("  - When the alternative (running everything at the fast dt)")
    print("    is too expensive.")


if __name__ == "__main__":
    main()
