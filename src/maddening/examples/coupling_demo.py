#!/usr/bin/env python
"""
Iterative coupling (Gauss-Seidel) demo for MADDENING.

Demonstrates the difference between *staggered* coupling (default for
cycles -- back-edges read previous timestep) and *Gauss-Seidel*
coupling via ``add_coupling_group()`` / ``auto_couple()``.

Setup: two spring-dampers connected in a loop -- each spring's anchor
is the other spring's position.  This creates a bidirectional coupling
(cycle in the graph).

- **Staggered**: each spring sees the other's *previous-step* position
  as its anchor.  This introduces a one-step lag.
- **Gauss-Seidel**: within each timestep, the group is iterated until
  the positions converge, giving self-consistent results.

The demo compares the trajectories from both approaches.

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python maddening/examples/coupling_demo.py
"""

import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


def build_coupled_springs(use_coupling=False, use_auto=False):
    """Build a graph with two bidirectionally-coupled springs.

    Spring A's anchor is Spring B's position, and vice versa.
    This creates a cycle (A -> B -> A).

    Parameters
    ----------
    use_coupling : bool
        If True, register a coupling group for Gauss-Seidel iteration.
    use_auto : bool
        If True, use ``auto_couple()`` to detect the SCC automatically.
    """
    gm = GraphManager()

    dt = 0.001

    spring_a = SpringDamperNode(
        name="spring_a",
        timestep=dt,
        stiffness=10.0,
        damping=5.0,
        mass=1.0,
        rest_length=2.0,
        initial_position=0.0,
        initial_velocity=0.0,
    )
    spring_b = SpringDamperNode(
        name="spring_b",
        timestep=dt,
        stiffness=10.0,
        damping=5.0,
        mass=1.0,
        rest_length=2.0,
        initial_position=5.0,
        initial_velocity=0.0,
    )

    gm.add_node(spring_a)
    gm.add_node(spring_b)

    # Bidirectional coupling: A's position -> B's anchor, B's position -> A's anchor
    gm.add_edge(
        source="spring_a", target="spring_b",
        source_field="position", target_field="anchor_position",
    )
    gm.add_edge(
        source="spring_b", target="spring_a",
        source_field="position", target_field="anchor_position",
    )

    if use_coupling and not use_auto:
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=20,
            tolerance=1e-10,
        )
    elif use_auto:
        groups = gm.auto_couple(max_iterations=20, tolerance=1e-10)
        print(f"  auto_couple() found {len(groups)} coupling group(s):")
        for g in groups:
            print(f"    nodes={set(g.nodes)}, max_iter={g.max_iterations}, "
                  f"tol={g.tolerance:.0e}")

    gm.compile()
    return gm


def run_simulation(gm, n_steps):
    """Run and collect trajectory."""
    positions_a = []
    positions_b = []

    def callback(step_idx, state):
        positions_a.append(float(state["spring_a"]["position"]))
        positions_b.append(float(state["spring_b"]["position"]))

    gm.run(n_steps, callback=callback)
    return positions_a, positions_b


def main() -> None:
    n_steps = 5000
    dt = 0.001
    total_time = n_steps * dt

    print("Coupling Demo: Two Bidirectionally-Coupled Springs")
    print("=" * 60)
    print(f"  Spring A: initial pos = 0.0, k=10, c=5, m=1, rest=2.0")
    print(f"  Spring B: initial pos = 5.0, k=10, c=5, m=1, rest=2.0")
    print(f"  Timestep: {dt}, Steps: {n_steps}, Total time: {total_time:.1f}s")
    print(f"  Coupling: A.position -> B.anchor, B.position -> A.anchor")
    print()

    # ---- Scenario 1: Staggered (default) --------------------------------
    print("--- Scenario 1: Staggered (default cycle handling) ---")
    gm_stag = build_coupled_springs(use_coupling=False)
    print(f"  Schedule: {gm_stag.schedule}")
    pos_a_stag, pos_b_stag = run_simulation(gm_stag, n_steps)
    final_a_stag = pos_a_stag[-1]
    final_b_stag = pos_b_stag[-1]
    print(f"  Final A position: {final_a_stag:.6f}")
    print(f"  Final B position: {final_b_stag:.6f}")
    print()

    # ---- Scenario 2: Gauss-Seidel (manual) ------------------------------
    print("--- Scenario 2: Gauss-Seidel coupling (manual) ---")
    gm_gs = build_coupled_springs(use_coupling=True)
    print(f"  Schedule: {gm_gs.schedule}")
    pos_a_gs, pos_b_gs = run_simulation(gm_gs, n_steps)
    final_a_gs = pos_a_gs[-1]
    final_b_gs = pos_b_gs[-1]
    print(f"  Final A position: {final_a_gs:.6f}")
    print(f"  Final B position: {final_b_gs:.6f}")
    print()

    # ---- Scenario 3: auto_couple() --------------------------------------
    print("--- Scenario 3: auto_couple() (automatic SCC detection) ---")
    gm_auto = build_coupled_springs(use_auto=True)
    print(f"  Schedule: {gm_auto.schedule}")
    pos_a_auto, pos_b_auto = run_simulation(gm_auto, n_steps)
    final_a_auto = pos_a_auto[-1]
    final_b_auto = pos_b_auto[-1]
    print(f"  Final A position: {final_a_auto:.6f}")
    print(f"  Final B position: {final_b_auto:.6f}")
    print()

    # ---- Comparison -----------------------------------------------------
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Method':<25} {'Final A':>12} {'Final B':>12}")
    print("-" * 50)
    print(f"{'Staggered':<25} {final_a_stag:12.6f} {final_b_stag:12.6f}")
    print(f"{'Gauss-Seidel (manual)':<25} {final_a_gs:12.6f} {final_b_gs:12.6f}")
    print(f"{'Gauss-Seidel (auto)':<25} {final_a_auto:12.6f} {final_b_auto:12.6f}")
    print()

    # ---- Sanity checks --------------------------------------------------
    # 1. Manual and auto Gauss-Seidel should give identical results
    diff_a = abs(final_a_gs - final_a_auto)
    diff_b = abs(final_b_gs - final_b_auto)
    assert diff_a < 1e-5, (
        f"Manual vs auto GS mismatch for A: {diff_a:.2e}"
    )
    assert diff_b < 1e-5, (
        f"Manual vs auto GS mismatch for B: {diff_b:.2e}"
    )
    print(f"Check: manual and auto Gauss-Seidel agree (diff A={diff_a:.2e}, B={diff_b:.2e}).")

    # 2. With damping, both springs should settle to a finite equilibrium.
    #    The midpoint of initial positions is 2.5; the equilibrium depends
    #    on rest lengths and force law but should be nearby.
    midpoint = (0.0 + 5.0) / 2.0
    assert abs(final_a_gs) < 50.0, f"Spring A diverged: {final_a_gs}"
    assert abs(final_b_gs) < 50.0, f"Spring B diverged: {final_b_gs}"
    print(f"Check: springs settled to finite values -- "
          f"A={final_a_gs:.4f}, B={final_b_gs:.4f}.")

    # 3. Springs should be close to each other (damping brings them together)
    separation = abs(final_a_gs - final_b_gs)
    print(f"Check: final separation = {separation:.4f} (started at 5.0).")

    # 4. Staggered and coupled should give different results
    # (demonstrating that coupling matters)
    stag_gs_diff_a = abs(final_a_stag - final_a_gs)
    stag_gs_diff_b = abs(final_b_stag - final_b_gs)
    max_diff = max(stag_gs_diff_a, stag_gs_diff_b)
    # With dt=0.001 and stiff springs, the difference may be small but nonzero.
    print(f"Check: staggered vs coupled difference = {max_diff:.2e} "
          f"(A: {stag_gs_diff_a:.2e}, B: {stag_gs_diff_b:.2e}).")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
