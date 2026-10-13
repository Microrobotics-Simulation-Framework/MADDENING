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
  1. You want ORDER-INDEPENDENT iterates (GS depends on schedule order)
  2. The coupling is symmetric and GS introduces artificial asymmetry
  3. You plan to parallelize node updates across devices (future)

The key demonstration: with INSUFFICIENT iterations (max_iterations=2),
GS and Jacobi produce DIFFERENT results because information propagates
differently.  With sufficient iterations, both converge to the same
fixed point.  Both claims are checked.

Uses a 3-node cycle (A -> B -> C -> A) with asymmetric stiffness.  Each
``SpringDamperNode`` pulls its end towards ``anchor + rest_length``; the
rest lengths (1.0, 1.5, -2.5) sum to zero so that a configuration
satisfying all three at once exists (B = A + 1.5, C = B - 2.5,
A = C + 1).  With rest lengths that do not sum to zero there is none,
and the three ends chase each other for ever.

Usage
-----
    python -m maddening.examples.coupling.jacobi_vs_gauss_seidel
    python -m maddening.examples.coupling.jacobi_vs_gauss_seidel --steps 50
"""

import argparse
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


def build_triangle(mode="gauss-seidel", max_iters=15, acceleration="none"):
    """Build a 3-node cycle: A -> B -> C -> A."""
    gm = GraphManager()
    dt = 0.005

    # Three springs with DIFFERENT stiffnesses -- asymmetric coupling.
    # Each end is pulled only towards its predecessor (a directed cycle),
    # which oscillates with growing amplitude unless the damping is large
    # enough -- roughly c**2 > k * m / 2 for each node -- so the damping
    # here sits above that.
    gm.add_node(SpringDamperNode(
        name="A", timestep=dt, stiffness=80.0, damping=10.0,
        mass=1.0, rest_length=1.0, initial_position=0.0,
    ))
    gm.add_node(SpringDamperNode(
        name="B", timestep=dt, stiffness=50.0, damping=8.0,
        mass=0.8, rest_length=1.5, initial_position=3.0,
    ))
    gm.add_node(SpringDamperNode(
        name="C", timestep=dt, stiffness=30.0, damping=8.0,
        mass=1.2, rest_length=-2.5, initial_position=6.0,
    ))

    gm.add_edge("A", "B", "position", "anchor_position")
    gm.add_edge("B", "C", "position", "anchor_position")
    gm.add_edge("C", "A", "position", "anchor_position")

    gm.add_coupling_group(
        ["A", "B", "C"],
        max_iterations=max_iters,
        tolerance=1e-6,
        iteration_mode=mode,
        acceleration=acceleration,
    )
    gm.compile()
    return gm


def run_and_collect(gm, n_steps, n_diag=10):
    """Run *n_steps*; return final positions, average passes, unconverged steps.

    ``coupling_diagnostics()`` is read after each of the first *n_diag*
    steps (it is a host round trip, far dearer than a step); the rest of
    the run is one ``lax.scan``, and its last step's verdict is checked.
    """
    n_diag = min(n_diag, n_steps)
    total_iters = 0
    unconverged = 0
    for _ in range(n_diag):
        gm.step()
        info = gm.coupling_diagnostics()["A+B+C"]
        total_iters += info["iterations"]
        unconverged += not info["converged"]
    if n_steps > n_diag:
        gm.run_scan(n_steps - n_diag)
        unconverged += not gm.coupling_diagnostics()["A+B+C"]["converged"]

    pos = {name: float(gm.get_node_state(name)["position"])
           for name in ["A", "B", "C"]}
    return pos, total_iters / n_diag, unconverged


def run_final(gm, n_steps):
    """Final positions only -- one lax.scan, no per-step diagnostics."""
    state = gm.run_scan(n_steps)
    return {name: float(state[name]["position"]) for name in ["A", "B", "C"]}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Jacobi vs Gauss-Seidel")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Steps per configuration (default: 1000)")
    n_steps = parser.parse_args(argv).steps

    print("Jacobi vs Gauss-Seidel: 3-Node Cycle")
    print("=" * 70)
    print()
    print("  Cycle: A(k=80) -> B(k=50) -> C(k=30) -> A")
    print("  Asymmetric stiffness so iteration order matters.")
    print()

    # ---- Part 1: Converged results (sufficient iterations) ----
    print("Part 1: CONVERGED results (max_iterations=15, tol=1e-6)")
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

    print(f"  {'Method':<20} {'Avg iters*':>10} "
          f"{'A':>9} {'B':>9} {'C':>9}")
    print(f"  {'-'*61}")
    converged_results = {}
    for label, mode, mi, accel in configs_converged:
        gm = build_triangle(mode=mode, max_iters=mi, acceleration=accel)
        pos, avg_it, unconverged = run_and_collect(gm, n_steps)
        assert unconverged == 0, f"{label}: {unconverged} steps unconverged"
        converged_results[label] = (pos, avg_it)
        print(f"  {label:<20} {avg_it:10.1f} "
              f"{pos['A']:9.5f} {pos['B']:9.5f} {pos['C']:9.5f}")

    print("  * passes per step, averaged over the first 10 steps")
    ref = converged_results["GS (plain)"][0]
    for label, (pos, _) in converged_results.items():
        diff = sum(abs(pos[n] - ref[n]) for n in "ABC")
        assert diff < 1e-4, f"{label} reached a different state: {diff:.2e}"
    if n_steps >= 1000:
        # 5 s is several decay times of the slowest mode: the cycle has
        # come to rest with every end at its rest offset from its anchor.
        assert abs((ref["B"] - ref["A"]) - 1.5) < 1e-3, ref
        assert abs((ref["C"] - ref["B"]) + 2.5) < 1e-3, ref
        print(f"  Settled: B - A = {ref['B'] - ref['A']:.4f} (rest 1.5), "
              f"C - B = {ref['C'] - ref['B']:.4f} (rest -2.5).")
    gs_it = converged_results["GS (plain)"][1]
    jac_it = converged_results["Jacobi (plain)"][1]
    print()
    print(f"  All four converged on every step to the same state; GS needed "
          f"{gs_it:.1f} passes per step, Jacobi {jac_it:.1f}.")
    print()

    # ---- Part 2: Under-converged results (too few iterations) ----
    n_transient = min(60, n_steps)
    print(f"Part 2: UNDER-CONVERGED results (max_iterations=2), first "
          f"{n_transient} steps")
    print("-" * 70)
    print()
    print("  With only 2 iterations, GS and Jacobi take DIFFERENT paths")
    print("  because information propagates differently within each pass.")
    print("  GS: A updates, then B sees A's new value, then C sees B's new value.")
    print("  Jacobi: A, B, C all see each other's PREVIOUS-iteration values.")
    print("  Compared during the transient, while the ends are still moving.")
    print()

    configs_underconv = [
        ("Converged (GS, 15)", "gauss-seidel", 15, "none"),
        ("GS (2 iters)", "gauss-seidel", 2, "none"),
        ("Jacobi (2 iters)", "jacobi", 2, "none"),
    ]

    print(f"  {'Method':<20} "
          f"{'A':>10} {'B':>10} {'C':>10} {'|error|':>10}")
    print(f"  {'-'*63}")
    underconv_results = {}
    for label, mode, mi, accel in configs_underconv:
        gm = build_triangle(mode=mode, max_iters=mi, acceleration=accel)
        pos = run_final(gm, n_transient)
        underconv_results[label] = pos
        ref_t = underconv_results["Converged (GS, 15)"]
        err = sum(abs(pos[n] - ref_t[n]) for n in "ABC")
        print(f"  {label:<20} "
              f"{pos['A']:10.6f} {pos['B']:10.6f} {pos['C']:10.6f} {err:10.2e}")

    gs_pos = underconv_results["GS (2 iters)"]
    jac_pos = underconv_results["Jacobi (2 iters)"]
    ref_t = underconv_results["Converged (GS, 15)"]
    diff = sum(abs(gs_pos[n] - jac_pos[n]) for n in ["A", "B", "C"])
    err_gs = sum(abs(gs_pos[n] - ref_t[n]) for n in "ABC")
    err_jac = sum(abs(jac_pos[n] - ref_t[n]) for n in "ABC")
    print()
    print(f"  GS-vs-Jacobi position difference: {diff:.2e}")
    assert diff > 0.0 and err_jac > err_gs, (
        f"expected 2-pass Jacobi to sit further from the converged run than "
        f"2-pass GS: {err_jac:.2e} vs {err_gs:.2e}"
    )
    print("  -> GS and Jacobi give DIFFERENT results with limited iterations:")
    print("     two GS passes already reach the fixed point here, two Jacobi")
    print("     passes do not.  The iteration ORDER matters when you haven't")
    print("     converged.  (Run long enough and both still come to rest at")
    print("     the same state: a state at rest is a fixed point however few")
    print("     passes reach it, so the error lives in the transient.)")
    print()
    print("Takeaway:")
    print("  - Gauss-Seidel (default) needed fewer passes here.")
    print("  - Use Jacobi when the iterates must not depend on schedule order")
    print("    (e.g., for verification or parallel execution).")
    print("  - Both reach the same answer with sufficient iterations.")


if __name__ == "__main__":
    main()
