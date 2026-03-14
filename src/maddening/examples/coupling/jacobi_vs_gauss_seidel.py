#!/usr/bin/env python
"""
Jacobi vs Gauss-Seidel iteration mode comparison.

Demonstrates the two iteration modes available for coupling groups:

- **Gauss-Seidel** (default): nodes are updated sequentially within
  each iteration.  Each node sees the latest results from nodes
  earlier in the schedule.
- **Jacobi**: all nodes read from the frozen previous-iteration state
  and are updated independently.  Results are swapped in after all
  nodes have been evaluated.

Uses a 3-node cycle (A -> B -> C -> A) to highlight the difference
between sequential and parallel updates.

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python -m maddening.examples.coupling.jacobi_vs_gauss_seidel
"""

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


def build_triangle(mode="gauss-seidel", acceleration="none"):
    """Build a 3-node cycle: A -> B -> C -> A."""
    gm = GraphManager()
    dt = 0.005

    # Three springs with different stiffnesses (asymmetric)
    gm.add_node(SpringDamperNode(
        name="A", timestep=dt, stiffness=80.0, damping=1.0,
        mass=1.0, rest_length=1.0, initial_position=0.0,
    ))
    gm.add_node(SpringDamperNode(
        name="B", timestep=dt, stiffness=50.0, damping=1.5,
        mass=0.8, rest_length=1.5, initial_position=3.0,
    ))
    gm.add_node(SpringDamperNode(
        name="C", timestep=dt, stiffness=30.0, damping=2.0,
        mass=1.2, rest_length=2.0, initial_position=6.0,
    ))

    gm.add_edge("A", "B", "position", "anchor_position")
    gm.add_edge("B", "C", "position", "anchor_position")
    gm.add_edge("C", "A", "position", "anchor_position")

    gm.add_coupling_group(
        ["A", "B", "C"],
        max_iterations=15,
        tolerance=1e-10,
        iteration_mode=mode,
        acceleration=acceleration,
        diagnostics=True,
    )
    gm.compile()
    return gm


def run_and_report(label, gm, n_steps):
    """Run simulation and report results."""
    total_iters = 0
    for _ in range(n_steps):
        gm.step()
        diag = gm.coupling_diagnostics()
        key = list(diag.keys())[0]
        total_iters += diag[key]["iterations"]

    pos = {name: float(gm.get_node_state(name)["position"])
           for name in ["A", "B", "C"]}
    avg_iters = total_iters / n_steps
    return pos, avg_iters, total_iters


def main() -> None:
    n_steps = 300

    print("Jacobi vs Gauss-Seidel: 3-Node Cycle")
    print("=" * 65)
    print(f"  Cycle: A(k=80) -> B(k=50) -> C(k=30) -> A")
    print(f"  Initial: A=0, B=3, C=6")
    print(f"  dt=0.005, max_iterations=15, tol=1e-10")
    print(f"  Running {n_steps} steps each")
    print()

    configs = [
        ("GS (plain)", "gauss-seidel", "none"),
        ("GS + Aitken", "gauss-seidel", "aitken"),
        ("Jacobi (plain)", "jacobi", "none"),
        ("Jacobi + Aitken", "jacobi", "aitken"),
    ]

    all_results = {}
    for label, mode, accel in configs:
        gm = build_triangle(mode=mode, acceleration=accel)
        pos, avg_it, total_it = run_and_report(label, gm, n_steps)
        all_results[label] = (pos, avg_it, total_it)

    # Print comparison
    print(f"{'Method':<20} {'Avg iters':>10} {'Total':>8} "
          f"{'A':>8} {'B':>8} {'C':>8}")
    print("-" * 65)
    for label, (pos, avg_it, total_it) in all_results.items():
        print(f"{label:<20} {avg_it:10.1f} {total_it:8d} "
              f"{pos['A']:8.3f} {pos['B']:8.3f} {pos['C']:8.3f}")
    print()

    # All methods should reach the same equilibrium
    ref_pos = list(all_results.values())[0][0]
    for label, (pos, _, _) in all_results.items():
        diff = sum(abs(pos[n] - ref_pos[n]) for n in ["A", "B", "C"])
        assert diff < 0.5, f"{label} equilibrium differs: diff={diff:.4f}"
    print("Check: all methods converge to the same equilibrium.")

    # Gauss-Seidel and Jacobi should give different trajectories
    gs_pos = all_results["GS (plain)"][0]
    jac_pos = all_results["Jacobi (plain)"][0]
    diff = sum(abs(gs_pos[n] - jac_pos[n]) for n in ["A", "B", "C"])
    print(f"Check: GS vs Jacobi position difference = {diff:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
