#!/usr/bin/env python
"""
Multi-rate timestep scheduling demo for MADDENING.

Demonstrates how nodes with different timesteps coexist in a single
graph.  The GraphManager automatically derives the base timestep
(GCD of all node timesteps) and assigns rate dividers so that each
node fires at its declared rate.

Setup:
- "fast_ball" runs at dt=0.001 (1 kHz) -- fine-grained dynamics
- "slow_spring" runs at dt=0.01 (100 Hz) -- coarser spring response
- "table" runs at dt=0.001 (matches fast ball for collision accuracy)

The compiled step function advances at the base rate (0.001s).
The slow spring only updates every 10th base step, but its update
is always computed (for JAX traceability) and conditionally applied
via jnp.where.

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python maddening/examples/multirate_demo.py
"""

import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode


def main() -> None:
    # ---- build the multi-rate graph -------------------------------------
    gm = GraphManager()

    # Fast nodes (1 kHz)
    table = TableNode(name="table", timestep=0.001, position=0.0)
    fast_ball = BallNode(
        name="fast_ball",
        timestep=0.001,
        initial_position=3.0,
        initial_velocity=0.0,
        elasticity=0.8,
    )

    # Slow node (100 Hz)
    slow_spring = SpringDamperNode(
        name="slow_spring",
        timestep=0.01,
        stiffness=40.0,
        damping=2.0,
        mass=0.5,
        rest_length=1.0,
        initial_position=2.0,
        initial_velocity=0.0,
    )

    gm.add_node(table)
    gm.add_node(fast_ball)
    gm.add_node(slow_spring)

    # Wire table -> fast_ball (collision surface)
    gm.add_edge(
        source="table", target="fast_ball",
        source_field="position", target_field="table_position",
    )

    # Wire fast_ball -> slow_spring (spring tracks ball)
    gm.add_edge(
        source="fast_ball", target="slow_spring",
        source_field="position", target_field="anchor_position",
    )

    # ---- validate and compile -------------------------------------------
    issues = gm.validate()
    for issue in issues:
        print(issue)

    gm.compile()

    # ---- report multi-rate info -----------------------------------------
    print(f"\n{'=' * 55}")
    print("MULTI-RATE SCHEDULING INFO")
    print(f"{'=' * 55}")
    print(f"  Is multi-rate:   {gm.is_multirate}")
    print(f"  Base timestep:   {gm.base_timestep}")
    print(f"  Rate dividers:   {gm.rate_dividers}")
    print(f"  Schedule:        {gm.schedule}")
    print(f"  Graph:           {gm}")

    for name, divider in gm.rate_dividers.items():
        effective_dt = gm.base_timestep * divider
        effective_hz = 1.0 / effective_dt
        print(f"  Node '{name}': divider={divider}, "
              f"effective dt={effective_dt:.4f}s ({effective_hz:.0f} Hz)")

    # ---- run simulation -------------------------------------------------
    # Run for 2 seconds of simulation time.
    # Base timestep is 0.001, so 2000 base steps = 2 seconds.
    sim_time = 2.0
    n_base_steps = int(sim_time / gm.base_timestep)

    print(f"\nRunning {n_base_steps} base steps "
          f"(base_dt={gm.base_timestep}, total time={sim_time:.1f}s)...")

    final_state, history = gm.run_scan_with_history(n_base_steps)

    # ---- extract trajectories -------------------------------------------
    ball_pos = history["fast_ball"]["position"]
    ball_vel = history["fast_ball"]["velocity"]
    spring_pos = history["slow_spring"]["position"]
    spring_vel = history["slow_spring"]["velocity"]

    # ---- results --------------------------------------------------------
    print(f"\n{'=' * 55}")
    print("SIMULATION RESULTS")
    print(f"{'=' * 55}")

    print(f"\n  Fast ball (dt={0.001}):")
    print(f"    Final position:  {float(final_state['fast_ball']['position']):.6f}")
    print(f"    Final velocity:  {float(final_state['fast_ball']['velocity']):.6f}")
    print(f"    Min position:    {float(jnp.min(ball_pos)):.6f}")
    print(f"    Max position:    {float(jnp.max(ball_pos)):.6f}")

    print(f"\n  Slow spring (dt={0.01}):")
    print(f"    Final position:  {float(final_state['slow_spring']['position']):.6f}")
    print(f"    Final velocity:  {float(final_state['slow_spring']['velocity']):.6f}")
    print(f"    Min position:    {float(jnp.min(spring_pos)):.6f}")
    print(f"    Max position:    {float(jnp.max(spring_pos)):.6f}")

    # ---- verify multi-rate behaviour ------------------------------------
    # The slow spring should update less frequently than the fast ball.
    # Check that the spring position has "staircase" patterns: it holds
    # constant for ~10 base steps, then jumps.  We count the number of
    # unique consecutive differences.
    spring_diffs = jnp.diff(spring_pos)
    n_zero_diffs = int(jnp.sum(spring_diffs == 0.0))
    n_total_diffs = len(spring_diffs)
    pct_held = 100.0 * n_zero_diffs / n_total_diffs

    print(f"\n  Multi-rate verification:")
    print(f"    Spring held constant for {n_zero_diffs}/{n_total_diffs} "
          f"consecutive steps ({pct_held:.1f}%)")
    print(f"    Expected ~90% (updates every 10th step)")

    # Ball should change almost every step (gravity always acts)
    ball_diffs = jnp.diff(ball_pos)
    n_ball_changes = int(jnp.sum(ball_diffs != 0.0))
    pct_ball_active = 100.0 * n_ball_changes / len(ball_diffs)
    print(f"    Ball changed on {n_ball_changes}/{len(ball_diffs)} "
          f"steps ({pct_ball_active:.1f}%)")

    # ---- sanity checks --------------------------------------------------
    assert float(final_state["fast_ball"]["position"]) >= 0.0, \
        "Ball fell through the table!"
    assert pct_held > 80.0, \
        f"Spring should be held constant ~90% of steps, got {pct_held:.1f}%"
    assert gm.is_multirate, "Graph should be multi-rate!"

    print("\nSanity checks passed.")

    # ---- plot -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        base_dt = gm.base_timestep
        t = jnp.arange(n_base_steps) * base_dt

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # Positions
        axes[0].plot(t, ball_pos, "b-", linewidth=0.5, label="Fast ball (1 kHz)")
        axes[0].plot(t, spring_pos, "r-", linewidth=0.8, label="Slow spring (100 Hz)")
        axes[0].axhline(0, color="k", linewidth=0.5, linestyle="--", label="Table")
        axes[0].set_ylabel("Position (m)")
        axes[0].set_title("Multi-Rate Timestep Scheduling (MADDENING)")
        axes[0].legend()

        # Velocities
        axes[1].plot(t, ball_vel, "b-", linewidth=0.5, label="Fast ball")
        axes[1].plot(t, spring_vel, "r-", linewidth=0.8, label="Slow spring")
        axes[1].set_ylabel("Velocity (m/s)")
        axes[1].legend()

        # Zoom into a short window to show staircase pattern
        zoom_start = 0
        zoom_end = min(500, n_base_steps)  # first 0.5s
        t_zoom = t[zoom_start:zoom_end]
        sp_zoom = spring_pos[zoom_start:zoom_end]
        bp_zoom = ball_pos[zoom_start:zoom_end]

        axes[2].plot(t_zoom, bp_zoom, "b.-", linewidth=0.5, markersize=1,
                     label="Fast ball")
        axes[2].plot(t_zoom, sp_zoom, "r.-", linewidth=0.8, markersize=2,
                     label="Slow spring (staircase)")
        axes[2].set_ylabel("Position (m)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title("Zoomed: first 0.5s (spring staircase pattern)")
        axes[2].legend()

        plt.tight_layout()
        out_path = os.path.join(_project_root, "multirate_demo_result.png")
        plt.savefig(out_path, dpi=150)
        print(f"\nPlot saved to {out_path}")
    except ImportError:
        print("matplotlib not available, skipping plot.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
