#!/usr/bin/env python
"""
External inputs (controller injection) demo for MADDENING.

Demonstrates how to inject time-varying external inputs into a
simulation graph using the external_inputs API.  This is the
mechanism for controller commands, sensor data, or user inputs
that come from outside the graph.

Setup:
- Table at height 0 (collision surface).
- Ball starts at height 2, subject to gravity.
- An external input "external_force" is declared on the ball.

Three scenarios are simulated and compared:
1. No external force (free bouncing).
2. Constant upward force (partially counteracts gravity).
3. Periodic (sinusoidal) upward force (creates interesting dynamics).

Each scenario uses a step-by-step loop with changing external_inputs,
since run() and run_scan() only support static external inputs.

Note: BallNode does not natively consume "external_force" from
boundary_inputs.  We use a custom BallWithForceNode that extends
BallNode to accept an additional force term.

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python maddening/examples/external_inputs_demo.py
"""

import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.table import TableNode


# ---- Custom node that accepts external force ----------------------------

GRAVITY = -9.81


class BallWithForceNode(SimulationNode):
    """A ball under gravity that also accepts an external force.

    Extends the standard BallNode concept by reading an optional
    ``external_force`` from boundary_inputs.  This force is added
    to the gravitational acceleration before integration.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    initial_position : float
        Starting height (default 0.0).
    initial_velocity : float
        Starting velocity (default 0.0).
    elasticity : float
        Coefficient of restitution (default 0.8).
    mass : float
        Mass of the ball in kg (default 1.0).
    """

    def __init__(
        self,
        name: str,
        timestep: float,
        initial_position: float = 0.0,
        initial_velocity: float = 0.0,
        elasticity: float = 0.8,
        mass: float = 1.0,
    ):
        super().__init__(
            name,
            timestep,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            elasticity=elasticity,
            mass=mass,
        )

    def initial_state(self) -> dict:
        return {
            "position": jnp.array(self.params["initial_position"], dtype=jnp.float32),
            "velocity": jnp.array(self.params["initial_velocity"], dtype=jnp.float32),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        mass = self.params["mass"]
        ext_force = boundary_inputs.get(
            "external_force", jnp.array(0.0, dtype=jnp.float32)
        )

        # Gravity + external force
        acceleration = GRAVITY + ext_force / mass
        velocity = state["velocity"] + acceleration * dt
        position = state["position"] + velocity * dt

        # Collision with table
        table_pos = boundary_inputs.get("table_position", None)
        if table_pos is not None:
            elasticity = self.params["elasticity"]
            hit = position < table_pos
            position = jnp.where(hit, table_pos, position)
            velocity = jnp.where(
                hit & (jnp.abs(velocity) > 1e-4),
                -velocity * elasticity,
                jnp.where(hit, 0.0, velocity),
            )

        return {"position": position, "velocity": velocity}


def run_scenario(
    scenario_name: str,
    force_fn,
    n_steps: int,
    dt: float,
) -> tuple[list[float], list[float], list[float]]:
    """Run a scenario with a given force function.

    Parameters
    ----------
    scenario_name : str
        Label for printing.
    force_fn : callable
        Maps step index to external force value (float).
    n_steps : int
        Number of steps to simulate.
    dt : float
        Timestep.

    Returns
    -------
    (positions, velocities, forces) : tuple of lists
    """
    gm = GraphManager()

    table = TableNode(name="table", timestep=dt, position=0.0)
    ball = BallWithForceNode(
        name="ball",
        timestep=dt,
        initial_position=2.0,
        initial_velocity=0.0,
        elasticity=0.75,
        mass=1.0,
    )

    gm.add_node(table)
    gm.add_node(ball)

    gm.add_edge(
        source="table", target="ball",
        source_field="position", target_field="table_position",
    )

    # Declare external input
    gm.add_external_input(
        target_node="ball",
        target_field="external_force",
        shape=(),
        dtype=jnp.float32,
    )

    gm.compile()

    positions = []
    velocities = []
    forces = []

    for i in range(n_steps):
        f = force_fn(i)
        ext = {"ball": {"external_force": jnp.array(float(f), dtype=jnp.float32)}}
        state = gm.step(external_inputs=ext)

        positions.append(float(state["ball"]["position"]))
        velocities.append(float(state["ball"]["velocity"]))
        forces.append(float(f))

    return positions, velocities, forces


def main() -> None:
    n_steps = 1500
    dt = 0.01
    total_time = n_steps * dt

    print(f"External Inputs Demo: {n_steps} steps, dt={dt}, "
          f"total time={total_time:.1f}s")
    print("=" * 60)

    # ---- Scenario 1: No external force ----------------------------------
    print("\nScenario 1: No external force (free bouncing)")
    pos1, vel1, f1 = run_scenario(
        "No force",
        force_fn=lambda i: 0.0,
        n_steps=n_steps,
        dt=dt,
    )
    print(f"  Final position: {pos1[-1]:.4f}")
    print(f"  Final velocity: {vel1[-1]:.4f}")

    # ---- Scenario 2: Constant upward force ------------------------------
    print("\nScenario 2: Constant upward force (5 N, partially counteracts gravity)")
    pos2, vel2, f2 = run_scenario(
        "Constant force",
        force_fn=lambda i: 5.0,
        n_steps=n_steps,
        dt=dt,
    )
    print(f"  Final position: {pos2[-1]:.4f}")
    print(f"  Final velocity: {vel2[-1]:.4f}")
    print(f"  Effective gravity: {GRAVITY + 5.0:.2f} m/s^2 (reduced)")

    # ---- Scenario 3: Periodic sinusoidal force --------------------------
    print("\nScenario 3: Periodic sinusoidal force (amplitude=15 N, freq=1 Hz)")

    def sinusoidal_force(step_idx):
        t = step_idx * dt
        return 15.0 * float(jnp.sin(2.0 * jnp.pi * 1.0 * t))

    pos3, vel3, f3 = run_scenario(
        "Sinusoidal force",
        force_fn=sinusoidal_force,
        n_steps=n_steps,
        dt=dt,
    )
    print(f"  Final position: {pos3[-1]:.4f}")
    print(f"  Final velocity: {vel3[-1]:.4f}")

    # ---- comparison -----------------------------------------------------
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Scenario':<30} {'Mean Pos':>10} {'Max Pos':>10} {'Mean |Vel|':>10}")
    print("-" * 62)

    import numpy as np
    for label, pos, vel in [
        ("No force", pos1, vel1),
        ("Constant 5N upward", pos2, vel2),
        ("Sinusoidal 15N @ 1Hz", pos3, vel3),
    ]:
        mean_p = np.mean(pos)
        max_p = np.max(pos)
        mean_v = np.mean(np.abs(vel))
        print(f"{label:<30} {mean_p:10.4f} {max_p:10.4f} {mean_v:10.4f}")

    # ---- sanity checks --------------------------------------------------
    # Constant upward force should keep ball higher on average
    mean_pos1 = np.mean(pos1)
    mean_pos2 = np.mean(pos2)
    assert mean_pos2 > mean_pos1, \
        f"Constant force should raise mean position ({mean_pos2:.4f} vs {mean_pos1:.4f})"
    print(f"\nSanity check: constant force raises mean position "
          f"({mean_pos2:.4f} > {mean_pos1:.4f}).")

    # Sinusoidal force should produce higher max position than no force
    max_pos1 = max(pos1)
    max_pos3 = max(pos3)
    assert max_pos3 > max_pos1, \
        f"Sinusoidal force should produce higher max ({max_pos3:.4f} vs {max_pos1:.4f})"
    print(f"Sanity check: sinusoidal force produces higher peaks "
          f"({max_pos3:.4f} > {max_pos1:.4f}).")

    # Ball should never fall below table
    assert min(pos1) >= -0.01, "Ball fell through table (scenario 1)!"
    assert min(pos2) >= -0.01, "Ball fell through table (scenario 2)!"
    assert min(pos3) >= -0.01, "Ball fell through table (scenario 3)!"
    print("Sanity check: ball never fell through the table in any scenario.")

    # ---- plot -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = [i * dt for i in range(n_steps)]

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # Positions
        axes[0].plot(t, pos1, "b-", linewidth=0.6, label="No force", alpha=0.8)
        axes[0].plot(t, pos2, "r-", linewidth=0.6, label="Constant 5N", alpha=0.8)
        axes[0].plot(t, pos3, "g-", linewidth=0.6, label="Sinusoidal 15N", alpha=0.8)
        axes[0].axhline(0, color="k", linewidth=0.5, linestyle="--")
        axes[0].set_ylabel("Position (m)")
        axes[0].set_title("External Inputs Demo: Ball Position Under Different Forces")
        axes[0].legend()

        # Velocities
        axes[1].plot(t, vel1, "b-", linewidth=0.4, label="No force", alpha=0.7)
        axes[1].plot(t, vel2, "r-", linewidth=0.4, label="Constant 5N", alpha=0.7)
        axes[1].plot(t, vel3, "g-", linewidth=0.4, label="Sinusoidal 15N", alpha=0.7)
        axes[1].set_ylabel("Velocity (m/s)")
        axes[1].legend()

        # Applied forces
        axes[2].plot(t, f1, "b-", linewidth=0.6, label="No force")
        axes[2].plot(t, f2, "r-", linewidth=0.6, label="Constant 5N")
        axes[2].plot(t, f3, "g-", linewidth=0.6, label="Sinusoidal 15N")
        axes[2].set_ylabel("External Force (N)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend()

        plt.tight_layout()
        out_path = os.path.join(_project_root, "external_inputs_demo_result.png")
        plt.savefig(out_path, dpi=150)
        print(f"\nPlot saved to {out_path}")
    except ImportError:
        print("matplotlib not available, skipping plot.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
