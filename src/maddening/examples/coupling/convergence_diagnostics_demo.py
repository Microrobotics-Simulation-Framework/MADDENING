#!/usr/bin/env python
"""
Coupling convergence diagnostics demo.

Demonstrates the ``diagnostics=True`` feature that reports iteration
counts and final residuals from each coupling step.  Also shows
how to use the mixed atol/rtol convergence norm for scale-invariant
convergence checking.

Useful for tuning ``max_iterations`` and ``tolerance`` parameters,
and for detecting problems where the coupling solver struggles to
converge.

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python -m maddening.examples.coupling.convergence_diagnostics_demo
"""

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode


def demo_spring_diagnostics():
    """Track convergence of coupled springs step-by-step."""
    print("=" * 60)
    print("Spring Coupling: Step-by-Step Diagnostics")
    print("=" * 60)

    gm = GraphManager()
    dt = 0.01
    gm.add_node(SpringDamperNode(
        name="A", timestep=dt, stiffness=100.0, damping=0.5,
        mass=1.0, rest_length=1.0, initial_position=0.0,
    ))
    gm.add_node(SpringDamperNode(
        name="B", timestep=dt, stiffness=100.0, damping=0.5,
        mass=1.0, rest_length=1.0, initial_position=5.0,
    ))
    gm.add_edge("A", "B", "position", "anchor_position")
    gm.add_edge("B", "A", "position", "anchor_position")
    gm.add_coupling_group(
        ["A", "B"],
        max_iterations=25,
        tolerance=1e-10,
        diagnostics=True,
    )
    gm.compile()

    print(f"\n  dt={dt}, max_iterations=25, tolerance=1e-10")
    print(f"  First 20 steps:\n")
    print(f"  {'Step':>5} {'Iters':>6} {'Residual':>12} {'A pos':>10} {'B pos':>10}")
    print(f"  {'-'*46}")

    for i in range(20):
        gm.step()
        diag = gm.coupling_diagnostics()
        info = diag["A+B"]
        pos_a = float(gm.get_node_state("A")["position"])
        pos_b = float(gm.get_node_state("B")["position"])
        print(f"  {i+1:5d} {info['iterations']:6d} {info['residual']:12.2e} "
              f"{pos_a:10.4f} {pos_b:10.4f}")
    print()


def demo_mixed_norm():
    """Compare L2 norm vs mixed norm on a multi-scale problem."""
    print("=" * 60)
    print("Mixed Norm: Multi-Scale Convergence")
    print("=" * 60)

    gm_l2 = GraphManager()
    gm_mixed = GraphManager()
    dt = 0.001

    for gm, norm_type in [(gm_l2, "l2"), (gm_mixed, "mixed")]:
        # Heat rod coupled to a spring (different scales)
        gm.add_node(HeatNode(
            name="rod", timestep=dt, n_cells=10,
            thermal_diffusivity=0.01, length=1.0,
            initial_temperature=100.0,
        ))
        gm.add_node(SpringDamperNode(
            name="spring", timestep=dt, stiffness=50.0, damping=1.0,
            mass=1.0, rest_length=0.0, initial_position=0.0,
        ))
        # Rod's rightmost temperature -> spring's anchor
        gm.add_edge("rod", "spring", "temperature", "anchor_position",
                    transform=lambda T: T[-1] / 100.0)
        # Spring position -> rod's right BC
        gm.add_edge("spring", "rod", "position", "right_temperature",
                    transform=lambda p: p * 100.0)

        kwargs = dict(
            max_iterations=20,
            diagnostics=True,
        )
        if norm_type == "l2":
            kwargs["tolerance"] = 1e-8
        else:
            kwargs["convergence_norm"] = "mixed"
            kwargs["atol"] = 1e-8
            kwargs["rtol"] = 1e-4

        gm.add_coupling_group(["rod", "spring"], **kwargs)
        gm.compile()

    # Run both for 50 steps and compare
    print(f"\n  Heat rod (T~100) coupled to spring (x~0.01-1)")
    print(f"  Scale mismatch: temperature ~100x larger than position")
    print()

    total_l2 = 0
    total_mixed = 0
    for _ in range(50):
        gm_l2.step()
        gm_mixed.step()
        total_l2 += gm_l2.coupling_diagnostics()["rod+spring"]["iterations"]
        total_mixed += gm_mixed.coupling_diagnostics()["rod+spring"]["iterations"]

    print(f"  L2 norm:    {total_l2:5d} total iterations (50 steps)")
    print(f"  Mixed norm: {total_mixed:5d} total iterations (50 steps)")
    print(f"  Ratio: {total_l2 / max(total_mixed, 1):.2f}x")
    print()

    # Both should produce finite results
    s_l2 = gm_l2.get_node_state("rod")
    s_mixed = gm_mixed.get_node_state("rod")
    assert jnp.all(jnp.isfinite(s_l2["temperature"]))
    assert jnp.all(jnp.isfinite(s_mixed["temperature"]))
    print("  Check: both norms produce finite results.")
    print()


def demo_insufficient_iterations():
    """Show what happens when max_iterations is too low."""
    print("=" * 60)
    print("Diagnosing Insufficient Iterations")
    print("=" * 60)

    configs = [
        ("max_iter=2", 2),
        ("max_iter=5", 5),
        ("max_iter=15", 15),
    ]

    print(f"\n  Stiff springs (k=500), tight tolerance (1e-12)")
    print()
    print(f"  {'Config':<15} {'Avg iters':>10} {'Avg residual':>14} "
          f"{'Converged?':>12}")
    print(f"  {'-'*53}")

    for label, max_it in configs:
        gm = GraphManager()
        dt = 0.01
        gm.add_node(SpringDamperNode(
            name="A", timestep=dt, stiffness=500.0, damping=0.1,
            initial_position=0.0,
        ))
        gm.add_node(SpringDamperNode(
            name="B", timestep=dt, stiffness=500.0, damping=0.1,
            initial_position=10.0,
        ))
        gm.add_edge("A", "B", "position", "anchor_position")
        gm.add_edge("B", "A", "position", "anchor_position")
        gm.add_coupling_group(
            ["A", "B"],
            max_iterations=max_it,
            tolerance=1e-12,
            diagnostics=True,
        )
        gm.compile()

        total_iters = 0
        total_residual = 0.0
        n_steps = 30
        for _ in range(n_steps):
            gm.step()
            diag = gm.coupling_diagnostics()["A+B"]
            total_iters += diag["iterations"]
            total_residual += diag["residual"]

        avg_it = total_iters / n_steps
        avg_res = total_residual / n_steps
        converged = avg_res < 1e-12
        status = "yes" if converged else f"no (res={avg_res:.1e})"
        print(f"  {label:<15} {avg_it:10.1f} {avg_res:14.2e} {status:>12}")

    print()
    print("  Takeaway: if avg residual >> tolerance, increase max_iterations")
    print("  or use acceleration='aitken' or 'iqn-ils'.")
    print()


def main() -> None:
    demo_spring_diagnostics()
    demo_mixed_norm()
    demo_insufficient_iterations()
    print("All demos complete.")


if __name__ == "__main__":
    main()
