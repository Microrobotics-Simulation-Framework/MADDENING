#!/usr/bin/env python
"""
Parameter-sweep demo using MADDENING's ``run_sweep`` (jax.vmap).

Creates a bouncing-ball graph (ball + table) and sweeps over 20
different initial positions from 1.0 to 10.0.  Optionally plots the
trajectories if matplotlib is available.

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python maddening/examples/parameter_sweep_demo.py
"""

import sys
import os

# Ensure project root is on the path.
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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
        initial_position=5.0,   # default; overridden by sweep
        initial_velocity=0.0,
        elasticity=0.7,
    )
    table = TableNode(name="table", timestep=0.01, position=0.0)

    gm.add_node(table)
    gm.add_node(ball)
    gm.add_edge(
        source="table", target="ball",
        source_field="position", target_field="table_position",
    )
    gm.compile()
    print(f"Schedule: {gm.schedule}")

    # ---- set up the sweep -----------------------------------------------
    n_sims = 20
    n_steps = 500  # 5 seconds at dt=0.01

    initial_positions = jnp.linspace(1.0, 10.0, n_sims)

    batched_initial_states = {
        "table": {
            "position": jnp.zeros(n_sims, dtype=jnp.float32),
        },
        "ball": {
            "position": initial_positions.astype(jnp.float32),
            "velocity": jnp.zeros(n_sims, dtype=jnp.float32),
        },
    }

    # ---- run the sweep with history -------------------------------------
    print(f"\nRunning parameter sweep: {n_sims} simulations x {n_steps} steps ...")
    finals, histories = gm.run_sweep(
        n_steps=n_steps,
        initial_states=batched_initial_states,
        return_history=True,
    )
    print("Sweep complete.")

    # ---- print results ---------------------------------------------------
    print(f"\n{'Initial Pos':>12s}  {'Final Pos':>12s}  {'Final Vel':>12s}")
    print("-" * 42)
    for i in range(n_sims):
        print(
            f"{float(initial_positions[i]):12.4f}  "
            f"{float(finals['ball']['position'][i]):12.6f}  "
            f"{float(finals['ball']['velocity'][i]):12.6f}"
        )

    # ---- verification ----------------------------------------------------
    # All final positions should be >= 0 (ball cannot fall through table)
    assert jnp.all(finals["ball"]["position"] >= -1e-5), \
        "FAIL: a ball fell through the table!"
    print("\nVerification: no ball fell through the table.")

    # History shapes should be (n_sims, n_steps)
    assert histories["ball"]["position"].shape == (n_sims, n_steps), \
        f"FAIL: unexpected history shape {histories['ball']['position'].shape}"
    print(f"History shape: {histories['ball']['position'].shape} -- correct.")

    # Final states should match last history entry
    assert jnp.allclose(
        finals["ball"]["position"],
        histories["ball"]["position"][:, -1],
        atol=1e-5,
    ), "FAIL: final state does not match last history entry!"
    print("Final states match last history entry.")

    # ---- optional plot ---------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        t = jnp.arange(n_steps) * 0.01

        for i in range(n_sims):
            alpha = 0.3 + 0.7 * (i / (n_sims - 1))
            ax.plot(
                t,
                histories["ball"]["position"][i],
                linewidth=0.8,
                alpha=alpha,
                label=f"h0={float(initial_positions[i]):.1f}" if i % 5 == 0 else None,
            )

        ax.axhline(0.0, color="k", linewidth=1.5, linestyle="--", label="Table")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Ball position (height)")
        ax.set_title(f"Parameter Sweep: {n_sims} bouncing balls with different initial heights")
        ax.legend(loc="upper right")
        ax.set_xlim(0, n_steps * 0.01)
        plt.tight_layout()

        out_path = os.path.join(_project_root, "parameter_sweep_result.png")
        plt.savefig(out_path, dpi=150)
        print(f"\nPlot saved to {out_path}")
    except ImportError:
        print("\nmatplotlib not available, skipping plot.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
