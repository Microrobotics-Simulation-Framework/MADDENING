#!/usr/bin/env python
"""
Coupled spring-ball system using the MADDENING GraphManager.

Demonstrates multi-node graph construction with two-way data coupling:
- A table provides a collision surface at height 0.
- A ball starts at height 5 and bounces on the table under gravity.
- A spring-damper has its anchor wired to the ball's position, so
  the free end of the spring tracks and responds to the ball's motion.

The graph wiring is:
    table.position  -> ball.table_position   (collision surface)
    ball.position   -> spring.anchor_position (spring follows ball)

This produces oscillatory behavior in both the ball (bouncing with
energy loss from elasticity) and the spring (oscillating around
the ball's position with damping).

Uses run_scan_with_history() for efficient trajectory collection
and matplotlib (Agg backend) for plotting.

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python maddening/examples/coupled_spring_ball.py
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
    # ---- build the graph ------------------------------------------------
    gm = GraphManager()

    table = TableNode(name="table", timestep=0.01, position=0.0)
    ball = BallNode(
        name="ball",
        timestep=0.01,
        initial_position=5.0,
        initial_velocity=0.0,
        elasticity=0.7,
    )
    spring = SpringDamperNode(
        name="spring",
        timestep=0.01,
        stiffness=50.0,
        damping=2.0,
        mass=0.5,
        rest_length=1.0,
        initial_position=4.0,  # starts 1m below ball (at rest length)
        initial_velocity=0.0,
    )

    gm.add_node(table)
    gm.add_node(ball)
    gm.add_node(spring)

    # Wire table -> ball (collision surface)
    gm.add_edge(
        source="table",
        target="ball",
        source_field="position",
        target_field="table_position",
    )

    # Wire ball -> spring (spring anchor follows ball position)
    gm.add_edge(
        source="ball",
        target="spring",
        source_field="position",
        target_field="anchor_position",
    )

    # ---- validate and compile -------------------------------------------
    issues = gm.validate()
    for issue in issues:
        print(issue)

    gm.compile()
    print(f"Schedule: {gm.schedule}")
    print(f"Graph: {gm}")

    # ---- run with full history ------------------------------------------
    n_steps = 2000
    dt = 0.01
    print(f"\nRunning {n_steps} steps (dt={dt}, total time={n_steps * dt:.1f}s)...")

    final_state, history = gm.run_scan_with_history(n_steps)

    # Extract trajectories
    ball_pos = history["ball"]["position"]
    ball_vel = history["ball"]["velocity"]
    spring_pos = history["spring"]["position"]
    spring_vel = history["spring"]["velocity"]

    # ---- summary statistics ---------------------------------------------
    print("\n--- Ball trajectory ---")
    print(f"  Min position:  {float(jnp.min(ball_pos)):.4f}")
    print(f"  Max position:  {float(jnp.max(ball_pos)):.4f}")
    print(f"  Mean position: {float(jnp.mean(ball_pos)):.4f}")
    print(f"  Final position: {float(final_state['ball']['position']):.4f}")
    print(f"  Final velocity: {float(final_state['ball']['velocity']):.4f}")

    print("\n--- Spring trajectory ---")
    print(f"  Min position:  {float(jnp.min(spring_pos)):.4f}")
    print(f"  Max position:  {float(jnp.max(spring_pos)):.4f}")
    print(f"  Mean position: {float(jnp.mean(spring_pos)):.4f}")
    print(f"  Final position: {float(final_state['spring']['position']):.4f}")
    print(f"  Final velocity: {float(final_state['spring']['velocity']):.4f}")

    # ---- sanity checks --------------------------------------------------
    assert float(final_state["ball"]["position"]) >= 0.0, "Ball fell through the table!"
    print("\nSanity check: ball did not fall through the table.")

    # Spring should have oscillated (check that it didn't stay at initial position)
    spring_range = float(jnp.max(spring_pos) - jnp.min(spring_pos))
    assert spring_range > 0.1, f"Spring did not oscillate (range={spring_range:.4f})"
    print(f"Sanity check: spring oscillated with range {spring_range:.4f}.")

    # ---- plot -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = jnp.arange(n_steps) * dt

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        # Positions
        axes[0].plot(t, ball_pos, "b-", linewidth=0.7, label="Ball")
        axes[0].plot(t, spring_pos, "r-", linewidth=0.7, label="Spring end")
        axes[0].axhline(0.0, color="k", linewidth=1, linestyle="--", label="Table")
        axes[0].set_ylabel("Position (m)")
        axes[0].set_title("Coupled Spring-Ball System (MADDENING)")
        axes[0].legend()

        # Velocities
        axes[1].plot(t, ball_vel, "b-", linewidth=0.7, label="Ball")
        axes[1].plot(t, spring_vel, "r-", linewidth=0.7, label="Spring end")
        axes[1].set_ylabel("Velocity (m/s)")
        axes[1].legend()

        # Separation (spring stretch)
        separation = ball_pos - spring_pos
        axes[2].plot(t, separation, "g-", linewidth=0.7)
        axes[2].axhline(1.0, color="k", linewidth=1, linestyle="--", label="Rest length")
        axes[2].set_ylabel("Spring stretch (m)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend()

        plt.tight_layout()
        out_path = os.path.join(_project_root, "coupled_spring_ball_result.png")
        plt.savefig(out_path, dpi=150)
        print(f"\nPlot saved to {out_path}")
    except ImportError:
        print("matplotlib not available, skipping plot.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
