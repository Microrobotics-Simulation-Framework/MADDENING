#!/usr/bin/env python
"""
Jacobi vs Gauss-Seidel iteration mode comparison.

**When to use each mode:**

- **Gauss-Seidel** (default): nodes update sequentially.  Each node
  sees the latest results from earlier nodes in the schedule.  This
  gives faster convergence for most problems because information
  propagates within a single pass.

- **Jacobi**: all nodes read from the frozen previous-iteration state.
  Updates are computed independently and swapped in afterward.  Jacobi
  is useful when:
  1. You want ORDER-INDEPENDENT results (GS depends on schedule order)
  2. The coupling is symmetric and GS introduces artificial asymmetry
  3. You plan to parallelize node updates across devices (future)

The key demonstration: with INSUFFICIENT iterations (max_iterations=2),
GS and Jacobi produce DIFFERENT results because information propagates
differently.  With sufficient iterations, both converge to the same
fixed point.

Uses a 3-node cycle (A -> B -> C -> A) with asymmetric stiffness.

Usage
-----
    python -m maddening.examples.coupling.jacobi_vs_gauss_seidel
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


def build_triangle(mode="gauss-seidel", max_iters=15, acceleration="none"):
    """Build a 3-node cycle: A -> B -> C -> A."""
    gm = GraphManager()
    dt = 0.005

    # Three springs with DIFFERENT stiffnesses -- asymmetric coupling
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
        max_iterations=max_iters,
        tolerance=1e-12,
        iteration_mode=mode,
        acceleration=acceleration,
        diagnostics=True,
    )
    gm.compile()
    return gm


def run_and_collect(gm, n_steps):
    """Run and return final positions and average iteration count."""
    total_iters = 0
    for _ in range(n_steps):
        gm.step()
        diag = gm.coupling_diagnostics()
        key = list(diag.keys())[0]
        total_iters += diag[key]["iterations"]

    pos = {name: float(gm.get_node_state(name)["position"])
           for name in ["A", "B", "C"]}
    return pos, total_iters / n_steps


def main() -> None:
    n_steps = 300

    print("Jacobi vs Gauss-Seidel: 3-Node Cycle")
    print("=" * 70)
    print()
    print("  Cycle: A(k=80) -> B(k=50) -> C(k=30) -> A")
    print("  Asymmetric stiffness so iteration order matters.")
    print()

    # ---- Part 1: Converged results (sufficient iterations) ----
    print("Part 1: CONVERGED results (max_iterations=15, tol=1e-12)")
    print("-" * 70)
    print()
    print("  With enough iterations, both methods reach the SAME fixed point.")
    print("  The number of iterations may differ (GS usually needs fewer).")
    print()

    configs_converged = [
        ("GS (plain)", "gauss-seidel", 15, "none"),
        ("Jacobi (plain)", "jacobi", 15, "none"),
        ("GS + Aitken", "gauss-seidel", 15, "aitken"),
        ("Jacobi + Aitken", "jacobi", 15, "aitken"),
    ]

    print(f"  {'Method':<20} {'Avg iters':>10} "
          f"{'A':>8} {'B':>8} {'C':>8}")
    print(f"  {'-'*58}")
    converged_results = {}
    for label, mode, mi, accel in configs_converged:
        gm = build_triangle(mode=mode, max_iters=mi, acceleration=accel)
        pos, avg_it = run_and_collect(gm, n_steps)
        converged_results[label] = (pos, avg_it)
        print(f"  {label:<20} {avg_it:10.1f} "
              f"{pos['A']:8.3f} {pos['B']:8.3f} {pos['C']:8.3f}")

    print()

    # ---- Part 2: Under-converged results (too few iterations) ----
    print("Part 2: UNDER-CONVERGED results (max_iterations=2)")
    print("-" * 70)
    print()
    print("  With only 2 iterations, GS and Jacobi take DIFFERENT paths")
    print("  because information propagates differently within each pass.")
    print("  GS: A updates, then B sees A's new value, then C sees B's new value.")
    print("  Jacobi: A, B, C all see each other's PREVIOUS-iteration values.")
    print()

    configs_underconv = [
        ("GS (2 iters)", "gauss-seidel", 2, "none"),
        ("Jacobi (2 iters)", "jacobi", 2, "none"),
    ]

    print(f"  {'Method':<20} "
          f"{'A':>10} {'B':>10} {'C':>10}")
    print(f"  {'-'*52}")
    underconv_results = {}
    for label, mode, mi, accel in configs_underconv:
        gm = build_triangle(mode=mode, max_iters=mi, acceleration=accel)
        pos, _ = run_and_collect(gm, n_steps)
        underconv_results[label] = pos
        print(f"  {label:<20} "
              f"{pos['A']:10.4f} {pos['B']:10.4f} {pos['C']:10.4f}")

    gs_pos = underconv_results["GS (2 iters)"]
    jac_pos = underconv_results["Jacobi (2 iters)"]
    diff = sum(abs(gs_pos[n] - jac_pos[n]) for n in ["A", "B", "C"])
    print()
    print(f"  Total position difference: {diff:.6f}")
    if diff > 1e-6:
        print("  -> GS and Jacobi give DIFFERENT results with limited iterations.")
        print("     This is expected: the iteration ORDER matters when you")
        print("     haven't converged to the fixed point.")
    else:
        print("  -> Results are very similar (problem may be weakly coupled).")

    print()
    print("Takeaway:")
    print("  - Use Gauss-Seidel (default) for fastest convergence.")
    print("  - Use Jacobi when you need order-independent, reproducible results")
    print("    (e.g., for verification or parallel execution).")
    print("  - Both reach the same answer with sufficient iterations.")


if __name__ == "__main__":
    main()
