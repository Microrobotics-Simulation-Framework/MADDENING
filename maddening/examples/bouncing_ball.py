#!/usr/bin/env python
"""
Bouncing-ball demo using the MADDENING GraphManager.

A ball starts at height 5 and bounces on a table at height 0.
The simulation is fully JIT-compiled through JAX.

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python maddening/examples/bouncing_ball.py
"""

import sys
import os

# Ensure the project root is on the path so ``import maddening`` works
# even when invoked directly.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


def main() -> None:
    # ---- build the graph ------------------------------------------------
    gm = GraphManager()

    ball = BallNode(
        name="ball",
        timestep=0.01,
        initial_position=5.0,
        initial_velocity=0.0,
        elasticity=0.7,
    )
    table = TableNode(name="table", timestep=0.01, position=0.0)

    gm.add_node(table)
    gm.add_node(ball)

    # Wire table.position -> ball.table_position
    gm.add_edge(
        source="table",
        target="ball",
        source_field="position",
        target_field="table_position",
    )

    # ---- validate and compile -------------------------------------------
    issues = gm.validate()
    for issue in issues:
        print(issue)

    gm.compile()
    print(f"Schedule: {gm.schedule}")
    print(f"Graph: {gm}")

    # ---- run the simulation ---------------------------------------------
    n_steps = 1000
    positions = []
    velocities = []

    def record(step_idx, state):
        positions.append(float(state["ball"]["position"]))
        velocities.append(float(state["ball"]["velocity"]))

    gm.run(n_steps, callback=record)

    # ---- report ----------------------------------------------------------
    final = gm.get_node_state("ball")
    print(f"\nAfter {n_steps} steps (dt=0.01, t={n_steps * 0.01:.1f}s):")
    print(f"  Ball position = {float(final['position']):.6f}")
    print(f"  Ball velocity = {float(final['velocity']):.6f}")

    # Quick sanity check: ball should have settled near the table
    assert float(final["position"]) >= 0.0, "Ball fell through the table!"
    print("\nSanity check passed: ball did not fall through the table.")

    # ---- test serialization round-trip -----------------------------------
    config = gm.to_dict()
    registry = {"BallNode": BallNode, "TableNode": TableNode}
    gm2 = GraphManager.from_dict(config, registry)
    # Edge is already restored by from_dict, just compile and step.
    gm2.compile()
    gm2.step()
    final2 = gm2.get_node_state("ball")
    print(f"Serialization round-trip: OK (ball position after 1 step = {float(final2['position']):.6f})")

    # ---- test JIT + grad ------------------------------------------------
    # Demonstrate that we can differentiate through the compiled step
    empty_ext = gm._default_external_inputs()

    def loss_fn(init_pos):
        """Compute final ball position as a function of initial position."""
        state = {
            "table": {"position": jnp.array(0.0)},
            "ball": {"position": init_pos, "velocity": jnp.array(0.0)},
        }
        # Run a few steps through the compiled function
        for _ in range(10):
            state = gm._compiled_step(state, empty_ext)
        return state["ball"]["position"]

    grad_fn = jax.grad(loss_fn)
    init_pos = jnp.array(5.0)
    grad_val = grad_fn(init_pos)
    print(f"d(final_position)/d(init_position) after 10 steps = {float(grad_val):.6f}")
    print("JIT + grad: OK")

    # ---- optional matplotlib plot ----------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        t = [i * 0.01 for i in range(n_steps)]

        ax1.plot(t, positions, "b-", linewidth=0.5)
        ax1.set_ylabel("Position")
        ax1.set_title("Bouncing Ball (MADDENING GraphManager)")
        ax1.axhline(0.0, color="k", linewidth=1, label="Table")
        ax1.legend()

        ax2.plot(t, velocities, "r-", linewidth=0.5)
        ax2.set_ylabel("Velocity")
        ax2.set_xlabel("Time (s)")

        plt.tight_layout()
        plt.savefig(os.path.join(_project_root, "bouncing_ball_result.png"), dpi=150)
        print(f"Plot saved to {os.path.join(_project_root, 'bouncing_ball_result.png')}")
    except ImportError:
        print("matplotlib not available, skipping plot.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
