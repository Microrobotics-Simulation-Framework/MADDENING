#!/usr/bin/env python
"""
Coupling convergence diagnostics demo.

**Why diagnostics matter**: the coupling solver iterates inside each
timestep, but by default you can't see what's happening.  With
``diagnostics=True``, you get iteration counts and residuals, which
tell you:

- Is the solver converging?  (residual should decrease)
- Is max_iterations sufficient?  (if avg_iters == max_iters, you need more)
- Would acceleration help?  (if plain needs >5 iters, try Aitken)

**Mixed norm**: the default L2 norm is scale-dependent -- a temperature
field at 300 K dominates a displacement at 0.001 m.  The ``"mixed"``
norm uses per-field atol/rtol scaling, like ODE solvers do.  Converged
when norm <= 1.0.

Usage
-----
    python -m maddening.examples.coupling.convergence_diagnostics_demo
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


def demo_step_by_step():
    """Track convergence step-by-step to understand solver behavior."""
    print("=" * 65)
    print("Part 1: Step-by-Step Diagnostics")
    print("=" * 65)
    print()
    print("  Two stiff springs (k=500) with very tight tolerance (1e-12).")
    print("  Diagnostics show how many coupling iterations each timestep needs.")
    print()

    gm = GraphManager()
    dt = 0.005
    gm.add_node(SpringDamperNode(
        name="A", timestep=dt, stiffness=500.0, damping=0.5,
        mass=1.0, rest_length=1.0, initial_position=0.0,
    ))
    gm.add_node(SpringDamperNode(
        name="B", timestep=dt, stiffness=500.0, damping=0.5,
        mass=1.0, rest_length=1.0, initial_position=5.0,
    ))
    gm.add_edge("A", "B", "position", "anchor_position")
    gm.add_edge("B", "A", "position", "anchor_position")
    gm.add_coupling_group(
        ["A", "B"],
        max_iterations=30,
        tolerance=1e-12,
        diagnostics=True,
    )
    gm.compile()

    print(f"  {'Step':>5} {'Iters':>6} {'Residual':>12} {'A pos':>10} {'B pos':>10}")
    print(f"  {'-'*46}")

    for i in range(15):
        gm.step()
        diag = gm.coupling_diagnostics()
        info = diag["A+B"]
        pos_a = float(gm.get_node_state("A")["position"])
        pos_b = float(gm.get_node_state("B")["position"])
        print(f"  {i+1:5d} {info['iterations']:6d} {info['residual']:12.2e} "
              f"{pos_a:10.4f} {pos_b:10.4f}")

    print()
    print("  Notice: the iteration count and residual tell you whether the")
    print("  solver is working hard or coasting.  If 'Iters' equals")
    print("  max_iterations (30), you need more iterations or acceleration.")
    print()


def demo_insufficient_iterations():
    """Show the effect of insufficient max_iterations."""
    print("=" * 65)
    print("Part 2: Diagnosing Insufficient Iterations")
    print("=" * 65)
    print()
    print("  Same stiff problem, but with different max_iterations limits.")
    print("  Too few iterations means the solver CAN'T converge to tolerance.")
    print()

    configs = [
        ("max_iter=2", 2, "none"),
        ("max_iter=5", 5, "none"),
        ("max_iter=15", 15, "none"),
        ("max_iter=5 + Aitken", 5, "aitken"),
    ]

    print(f"  {'Config':<22} {'Avg iters':>10} {'Avg residual':>14} "
          f"{'Status':>20}")
    print(f"  {'-'*68}")

    for label, max_it, accel in configs:
        gm = GraphManager()
        dt = 0.005
        gm.add_node(SpringDamperNode(
            name="A", timestep=dt, stiffness=500.0, damping=0.5,
            initial_position=0.0,
        ))
        gm.add_node(SpringDamperNode(
            name="B", timestep=dt, stiffness=500.0, damping=0.5,
            initial_position=5.0,
        ))
        gm.add_edge("A", "B", "position", "anchor_position")
        gm.add_edge("B", "A", "position", "anchor_position")
        gm.add_coupling_group(
            ["A", "B"],
            max_iterations=max_it,
            tolerance=1e-12,
            acceleration=accel,
            diagnostics=True,
        )
        gm.compile()

        total_iters = 0
        total_residual = 0.0
        n_steps = 50
        for _ in range(n_steps):
            gm.step()
            diag = gm.coupling_diagnostics()["A+B"]
            total_iters += diag["iterations"]
            total_residual += diag["residual"]

        avg_it = total_iters / n_steps
        avg_res = total_residual / n_steps
        if avg_res < 1e-10:
            status = "CONVERGED"
        elif avg_it >= max_it - 0.5:
            status = f"NOT CONVERGED (capped)"
        else:
            status = f"PARTIAL (res={avg_res:.1e})"
        print(f"  {label:<22} {avg_it:10.1f} {avg_res:14.2e} {status:>20}")

    print()
    print("  Rule of thumb:")
    print("  - If avg_iters == max_iterations, you're iteration-starved.")
    print("    Increase max_iterations or add acceleration='aitken'.")
    print("  - If avg_residual >> tolerance, the solver isn't converging.")
    print("  - Aitken can recover convergence even with fewer iterations.")
    print()


def demo_acceleration_effect():
    """Show how acceleration reduces iteration count."""
    print("=" * 65)
    print("Part 3: Acceleration Effect on Iteration Count")
    print("=" * 65)
    print()
    print("  Running the same stiff problem with different acceleration")
    print("  methods, all with max_iterations=30 and tolerance=1e-12.")
    print()

    methods = [
        ("Plain", "none"),
        ("Aitken", "aitken"),
        ("IQN-ILS", "iqn-ils"),
    ]

    print(f"  {'Method':<15} {'Avg iters':>10} {'Total (50 steps)':>18}")
    print(f"  {'-'*45}")

    for label, accel in methods:
        gm = GraphManager()
        dt = 0.005
        gm.add_node(SpringDamperNode(
            name="A", timestep=dt, stiffness=500.0, damping=0.5,
            initial_position=0.0,
        ))
        gm.add_node(SpringDamperNode(
            name="B", timestep=dt, stiffness=500.0, damping=0.5,
            initial_position=5.0,
        ))
        gm.add_edge("A", "B", "position", "anchor_position")
        gm.add_edge("B", "A", "position", "anchor_position")
        gm.add_coupling_group(
            ["A", "B"],
            max_iterations=30,
            tolerance=1e-12,
            acceleration=accel,
            diagnostics=True,
        )
        gm.compile()

        total_iters = 0
        n_steps = 50
        for _ in range(n_steps):
            gm.step()
            total_iters += gm.coupling_diagnostics()["A+B"]["iterations"]

        avg_it = total_iters / n_steps
        print(f"  {label:<15} {avg_it:10.1f} {total_iters:18d}")

    print()
    print("  Use diagnostics to measure, then choose the best strategy.")
    print()


def main() -> None:
    demo_step_by_step()
    demo_insufficient_iterations()
    demo_acceleration_effect()
    print("All demos complete.")


if __name__ == "__main__":
    main()
