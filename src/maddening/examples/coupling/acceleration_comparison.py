#!/usr/bin/env python
"""
Coupling acceleration methods comparison.

Compares plain fixed-point, Aitken, fixed under-relaxation, and
IQN-ILS quasi-Newton acceleration on the same bidirectional spring
problem.  Uses coupling diagnostics to report iteration counts
and residuals, showing how acceleration reduces the number of
iterations needed for convergence.

Setup: two spring-dampers connected in a loop.  Each spring's anchor
is the other spring's position, creating a bidirectional cycle.

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python -m maddening.examples.coupling.acceleration_comparison
"""

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


def build_graph(acceleration="none", relaxation=1.0):
    """Build a bidirectional spring graph with the given acceleration method."""
    gm = GraphManager()
    dt = 0.01

    gm.add_node(SpringDamperNode(
        name="spring_a", timestep=dt,
        stiffness=100.0, damping=0.5, mass=1.0,
        rest_length=1.0, initial_position=0.0,
    ))
    gm.add_node(SpringDamperNode(
        name="spring_b", timestep=dt,
        stiffness=100.0, damping=0.5, mass=1.0,
        rest_length=1.0, initial_position=5.0,
    ))
    gm.add_edge("spring_a", "spring_b", "position", "anchor_position")
    gm.add_edge("spring_b", "spring_a", "position", "anchor_position")

    gm.add_coupling_group(
        ["spring_a", "spring_b"],
        max_iterations=30,
        tolerance=1e-10,
        acceleration=acceleration,
        relaxation=relaxation,
        diagnostics=True,
    )
    gm.compile()
    return gm


def main() -> None:
    n_steps = 200

    methods = [
        ("Plain fixed-point", "none", 1.0),
        ("Aitken", "aitken", 1.0),
        ("Fixed (omega=0.7)", "fixed", 0.7),
        ("IQN-ILS", "iqn-ils", 1.0),
    ]

    print("Coupling Acceleration Comparison")
    print("=" * 70)
    print(f"  Springs: k=100, c=0.5, m=1, rest=1")
    print(f"  Initial positions: A=0, B=5")
    print(f"  dt=0.01, max_iterations=30, tol=1e-10")
    print(f"  Running {n_steps} steps each")
    print()

    results = {}
    for label, accel, omega in methods:
        gm = build_graph(acceleration=accel, relaxation=omega)

        # Run and collect diagnostics from each step
        total_iters = 0
        max_iters_used = 0
        for _ in range(n_steps):
            gm.step()
            diag = gm.coupling_diagnostics()
            info = diag["spring_a+spring_b"]
            total_iters += info["iterations"]
            max_iters_used = max(max_iters_used, info["iterations"])

        state = gm.get_node_state("spring_a")
        pos_a = float(state["position"])
        state_b = gm.get_node_state("spring_b")
        pos_b = float(state_b["position"])
        avg_iters = total_iters / n_steps

        results[label] = {
            "pos_a": pos_a, "pos_b": pos_b,
            "avg_iters": avg_iters, "max_iters": max_iters_used,
            "total_iters": total_iters,
        }

    # Print comparison table
    print(f"{'Method':<25} {'Avg iters':>10} {'Max iters':>10} "
          f"{'Total iters':>12} {'Final A':>10} {'Final B':>10}")
    print("-" * 80)
    for label, r in results.items():
        print(f"{label:<25} {r['avg_iters']:10.1f} {r['max_iters']:10d} "
              f"{r['total_iters']:12d} {r['pos_a']:10.4f} {r['pos_b']:10.4f}")
    print()

    # Verify all methods reach the same equilibrium
    positions = [(r["pos_a"], r["pos_b"]) for r in results.values()]
    ref_a, ref_b = positions[0]
    for label, r in results.items():
        diff = abs(r["pos_a"] - ref_a) + abs(r["pos_b"] - ref_b)
        assert diff < 0.1, f"{label} diverged from reference: diff={diff:.4f}"
    print("Check: all methods converge to the same equilibrium.")

    # Verify acceleration methods use fewer total iterations
    plain_total = results["Plain fixed-point"]["total_iters"]
    for label, r in results.items():
        if label != "Plain fixed-point":
            ratio = r["total_iters"] / max(plain_total, 1)
            status = "fewer" if ratio < 1.0 else "same/more"
            print(f"  {label}: {ratio:.2f}x iterations vs plain ({status})")

    print("\nDone.")


if __name__ == "__main__":
    main()
