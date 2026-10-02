#!/usr/bin/env python
"""
Coupling acceleration methods comparison.

Compares four ways of iterating a coupling group to its fixed point:

- **Plain (none)**: the default.  Each pass feeds the latest values
  straight back.  Zero overhead, and hard to beat when the coupling is
  weak and it converges in a few passes.

- **Aitken**: adapts a relaxation factor every pass from the last two
  residuals.  Cheap (one dot product) and a good first thing to try when
  plain iteration needs many passes.

- **Fixed relaxation (omega < 1)**: under-relaxation.  It damps an
  iteration whose error *alternates* in sign, and it always costs passes
  on one that would have converged anyway.  It cannot rescue an
  iteration that diverges *monotonically*: Part 2 shows one.

- **IQN-ILS**: interface quasi-Newton.  Builds a low-rank Jacobian from
  the iteration history -- more linear algebra per pass, far fewer
  passes on strongly-coupled problems, and the only one of the four that
  converges where plain iteration diverges in Part 2.

Setup: two masses joined by one spring, modelled as two
``SpringDamperNode``\\ s anchored to each other (``rest_length`` +1 on one
end and -1 on the other, so the pair is a single spring with equal and
opposite forces).  Within a step each node's new position responds to
its partner's with gain ``r = k * dt**2 / m``, so a Gauss-Seidel pass
scales the error by ``r**2``: the coupling strength is set by ``r``.

- **Part 1** runs 60 steps (``--steps``) at ``r = 0.025``, a weakly-coupled and
  dynamically stable setting, and checks that every method reaches the
  same trajectory.
- **Part 2** solves *one* step at ``r = 0.5`` and at ``r = 1.5``.  At
  those strengths the explicit springs are not stable over many steps --
  a property of the node's integrator, not of the coupling solver -- but
  a single step's fixed point is well defined, and it is all the solver
  sees.  At ``r = 1.5`` a Gauss-Seidel pass *amplifies* the error by
  ``r**2 = 2.25``.

Every iteration count printed is measured, and the conclusions the demo
prints are asserted.

Usage
-----
    python -m maddening.examples.coupling.acceleration_comparison
    python -m maddening.examples.coupling.acceleration_comparison --steps 400
"""

import argparse
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

DT = 0.005
MASS = 0.5
TOLERANCE = 1e-6
GROUP = "spring_a+spring_b"

METHODS = [
    ("Plain fixed-point", "none", 1.0),
    ("Aitken", "aitken", 1.0),
    ("Fixed (omega=0.5)", "fixed", 0.5),
    ("IQN-ILS", "iqn-ils", 1.0),
]


def build_graph(stiffness, damping, acceleration="none", relaxation=1.0,
                max_iterations=50):
    """Two masses on one spring, iterated as a coupling group."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode(
        name="spring_a", timestep=DT,
        stiffness=stiffness, damping=damping, mass=MASS,
        rest_length=1.0, initial_position=0.0,
    ))
    gm.add_node(SpringDamperNode(
        name="spring_b", timestep=DT,
        stiffness=stiffness, damping=damping, mass=MASS,
        rest_length=-1.0, initial_position=5.0,
    ))
    gm.add_edge("spring_a", "spring_b", "position", "anchor_position")
    gm.add_edge("spring_b", "spring_a", "position", "anchor_position")

    kwargs = dict(max_iterations=max_iterations, tolerance=TOLERANCE,
                  acceleration=acceleration)
    if acceleration == "fixed":
        kwargs["relaxation"] = relaxation
    gm.add_coupling_group(["spring_a", "spring_b"], **kwargs)
    gm.compile()
    return gm


def part1_weak_coupling(n_steps=60):
    """All four methods on a weakly-coupled, dynamically stable pair."""
    # r = k dt^2 / m = 500 * 0.005^2 / 0.5 = 0.025.  Damping 5 keeps the
    # coupled explicit scheme stable (it needs damping > k * dt = 2.5).
    k, c = 500.0, 5.0
    r = k * DT ** 2 / MASS
    print(f"Part 1: weak coupling, {n_steps} steps")
    print("-" * 72)
    print(f"  k={k:g}, c={c:g}, m={MASS:g}, dt={DT:g}: r = k*dt^2/m = {r:.3f},"
          f" so a pass scales the error by r^2 = {r * r:.1e}.")
    print()

    results = {}
    for label, accel, omega in METHODS:
        gm = build_graph(k, c, acceleration=accel, relaxation=omega)
        total = 0
        worst = 0
        unconverged = 0
        for _ in range(n_steps):
            gm.step()
            info = gm.coupling_diagnostics()[GROUP]
            total += info["iterations"]
            worst = max(worst, info["iterations"])
            unconverged += not info["converged"]
        results[label] = {
            "avg": total / n_steps, "max": worst, "total": total,
            "unconverged": unconverged,
            "a": float(gm.get_node_state("spring_a")["position"]),
            "b": float(gm.get_node_state("spring_b")["position"]),
        }

    print(f"  {'Method':<20} {'Avg iters':>10} {'Max':>5} {'Total':>7} "
          f"{'Unconv.':>8} {'Final A':>10} {'Final B':>10}")
    print(f"  {'-' * 74}")
    for label, res in results.items():
        print(f"  {label:<20} {res['avg']:10.1f} {res['max']:5d} "
              f"{res['total']:7d} {res['unconverged']:8d} "
              f"{res['a']:10.5f} {res['b']:10.5f}")
    print()

    ref = results["Plain fixed-point"]
    for label, res in results.items():
        assert res["unconverged"] == 0, f"{label} left steps unconverged"
        diff = abs(res["a"] - ref["a"]) + abs(res["b"] - ref["b"])
        assert diff < 1e-4, f"{label} reached a different state: diff={diff:.2e}"
    print("  Every method converged on every step and reached the same state:")
    print("  the acceleration changes how fast each step's fixed point is")
    print("  found, not which fixed point it is.")
    plain = ref["total"]
    for label, res in results.items():
        if label != "Plain fixed-point":
            print(f"    {label}: {res['total'] / plain:.2f}x the passes of plain")
    assert results["Fixed (omega=0.5)"]["total"] > plain, (
        "under-relaxation was expected to cost passes on a problem plain "
        "iteration already solves quickly"
    )
    print("  Under-relaxation only costs passes here.")
    print()


def solve_one_step(stiffness, accel, omega):
    """Iterate a single step's coupling solve and report it."""
    gm = build_graph(stiffness, 0.0, acceleration=accel, relaxation=omega)
    gm.step()
    return gm.coupling_diagnostics()[GROUP]


def part2_strong_coupling():
    """One step's coupling solve at two strengths where plain struggles."""
    print("Part 2: one step at strong coupling (max_iterations=50)")
    print("-" * 72)
    summary = {}
    for r in (0.5, 1.5):
        k = r * MASS / DT ** 2
        print(f"  r = {r} (k = {k:g}): a Gauss-Seidel pass scales the error "
              f"by r^2 = {r * r:g}")
        print(f"    {'Method':<20} {'Iters':>6} {'Converged':>10} {'Residual':>10}")
        for label, accel, omega in METHODS:
            info = solve_one_step(k, accel, omega)
            summary[(r, label)] = info
            print(f"    {label:<20} {info['iterations']:6d} "
                  f"{str(bool(info['converged'])):>10} {info['residual']:10.2e}")
        print()

    plain = summary[(0.5, "Plain fixed-point")]
    for label in ("Aitken", "IQN-ILS"):
        info = summary[(0.5, label)]
        assert info["converged"] and info["iterations"] < plain["iterations"], (
            f"{label} was expected to converge in fewer passes than plain at "
            f"r=0.5 ({info['iterations']} vs {plain['iterations']})"
        )
    print("  r = 0.5: plain iteration converges, slowly; Aitken and IQN-ILS")
    print(f"  need {summary[(0.5, 'Aitken')]['iterations']} and "
          f"{summary[(0.5, 'IQN-ILS')]['iterations']} passes against plain's "
          f"{plain['iterations']}.")

    assert not summary[(1.5, "Plain fixed-point")]["converged"]
    assert not summary[(1.5, "Fixed (omega=0.5)")]["converged"]
    assert summary[(1.5, "IQN-ILS")]["converged"]
    print("  r = 1.5: plain iteration diverges, and under-relaxation cannot")
    print("  help -- the error grows without changing sign, and relaxing a")
    print("  growth factor above 1 still leaves it above 1.  IQN-ILS, which")
    print("  solves for the fixed point from the iteration history instead")
    print("  of waiting for the iteration to contract, converges.")
    print()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Coupling acceleration comparison")
    parser.add_argument("--steps", type=int, default=60,
                        help="Steps in the Part 1 trajectory (default: 60)")
    args = parser.parse_args(argv)

    print("Coupling Acceleration Comparison")
    print("=" * 72)
    print(f"  tolerance = {TOLERANCE:g} (relative L2 norm)")
    print()
    part1_weak_coupling(args.steps)
    part2_strong_coupling()
    print("Takeaway (from the measurements above):")
    print("  - Weak coupling: plain iteration is hard to beat.")
    print("  - Strong coupling: Aitken or IQN-ILS cut the passes per step.")
    print("  - Monotone divergence: IQN-ILS; relaxation will not converge it.")
    print("  - Use coupling_diagnostics() to measure before choosing.")


if __name__ == "__main__":
    main()
