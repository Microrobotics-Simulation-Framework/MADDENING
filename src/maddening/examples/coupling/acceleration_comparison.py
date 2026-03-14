#!/usr/bin/env python
"""
Coupling acceleration methods comparison.

Compares convergence behavior of four coupling acceleration strategies
on a STIFF problem where plain fixed-point iteration struggles.

**When to use each method:**

- **Plain (none)**: Default. Fine for weakly-coupled or well-conditioned
  problems where convergence happens in 2-5 iterations. Zero overhead.

- **Aitken**: Best general-purpose accelerator. Automatically adapts the
  relaxation factor each iteration. Use when plain iteration needs >5
  iterations or when you're unsure. Minimal overhead (one dot product).

- **Fixed relaxation (omega < 1)**: Under-relaxation. Use when the solver
  DIVERGES — oscillates with growing amplitude. Omega=0.5 halves each
  correction, stabilizing otherwise-divergent problems. Slows convergence
  on well-conditioned problems.

- **IQN-ILS**: Quasi-Newton. Builds a low-rank Jacobian approximation
  from iteration history. Best for strongly-coupled problems (FSI,
  thermal-structural). Higher per-iteration cost (linear algebra) but
  dramatically fewer iterations on hard problems.

Setup: Two spring-dampers with HIGH stiffness (k=1000) and very tight
tolerance (1e-12).  This creates a stiff coupling problem where the
solver needs many iterations to converge.

Usage
-----
    python -m maddening.examples.coupling.acceleration_comparison
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


def build_graph(acceleration="none", relaxation=1.0):
    """Build a stiff bidirectional spring graph."""
    gm = GraphManager()
    dt = 0.005

    # Stiff springs with moderate damping.  The coupling spectral
    # radius per step is k*dt^2/m ~ 0.25.  With explicit integration
    # the per-step coupling is moderate, so the solver converges
    # in ~5-10 iterations -- enough to show differences between methods.
    gm.add_node(SpringDamperNode(
        name="spring_a", timestep=dt,
        stiffness=500.0, damping=2.0, mass=0.5,
        rest_length=1.0, initial_position=0.0,
    ))
    gm.add_node(SpringDamperNode(
        name="spring_b", timestep=dt,
        stiffness=500.0, damping=2.0, mass=0.5,
        rest_length=1.0, initial_position=5.0,
    ))
    gm.add_edge("spring_a", "spring_b", "position", "anchor_position")
    gm.add_edge("spring_b", "spring_a", "position", "anchor_position")

    gm.add_coupling_group(
        ["spring_a", "spring_b"],
        max_iterations=50,
        tolerance=1e-12,
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
        ("Fixed (omega=0.3)", "fixed", 0.3),
        ("IQN-ILS", "iqn-ils", 1.0),
    ]

    print("Coupling Acceleration Comparison")
    print("=" * 72)
    print()
    print("  Problem: two springs with stiff coupling (k=500, m=0.5, dt=0.005).")
    print("  Coupling spectral radius ~ k*dt^2/m = 0.025 per step.")
    print()
    print("  Note: with EXPLICIT integration (forward/semi-implicit Euler),")
    print("  per-step coupling is inherently moderate.  Acceleration methods")
    print("  show their full power with implicit schemes or large timesteps")
    print("  where the coupling spectral radius approaches 1.0.")
    print()
    print("  dt=0.005, max_iterations=50, tol=1e-12")
    print(f"  Running {n_steps} steps each")
    print()

    results = {}
    for label, accel, omega in methods:
        gm = build_graph(acceleration=accel, relaxation=omega)

        total_iters = 0
        max_iters_used = 0
        for _ in range(n_steps):
            gm.step()
            diag = gm.coupling_diagnostics()
            info = diag["spring_a+spring_b"]
            total_iters += info["iterations"]
            max_iters_used = max(max_iters_used, info["iterations"])

        pos_a = float(gm.get_node_state("spring_a")["position"])
        pos_b = float(gm.get_node_state("spring_b")["position"])
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
    print("All methods converge to the same equilibrium -- the acceleration")
    print("method affects HOW FAST you get there, not WHERE you end up.")
    print()

    # Analysis
    plain_total = results["Plain fixed-point"]["total_iters"]
    print("Analysis:")
    for label, r in results.items():
        if label == "Plain fixed-point":
            continue
        ratio = r["total_iters"] / max(plain_total, 1)
        if ratio < 0.8:
            print(f"  {label}: {ratio:.2f}x iterations -- "
                  f"FASTER than plain ({r['total_iters']} vs {plain_total})")
        elif ratio > 1.2:
            print(f"  {label}: {ratio:.2f}x iterations -- "
                  f"SLOWER (overhead exceeds benefit for this problem)")
        else:
            print(f"  {label}: {ratio:.2f}x iterations -- "
                  f"similar to plain")

    print()
    print("Takeaway:")
    print("  - For weakly-coupled problems, plain iteration is hard to beat.")
    print("  - For stiff coupling (high k, tight tolerance), Aitken or IQN-ILS")
    print("    can significantly reduce iteration count.")
    print("  - Under-relaxation (omega<1) is for STABILIZATION, not speed.")
    print("    Use it when plain iteration diverges, then switch to Aitken.")


if __name__ == "__main__":
    main()
