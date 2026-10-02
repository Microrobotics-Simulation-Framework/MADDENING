#!/usr/bin/env python
"""
Coupling convergence diagnostics demo.

**Why diagnostics matter**: a coupling group iterates inside each
timestep, and ``GraphManager.coupling_diagnostics()`` tells you what
that iteration did on the last step:

- ``iterations`` -- passes used.  ``iterations >= max_iterations`` means
  the group ran out of budget.
- ``converged`` -- whether the returned state met the group's criterion.
- ``residual`` / ``error_estimate`` -- how far the last pass moved, and
  an estimate of how far the state still is from the fixed point.

Under the default ``solver="ift"`` these are recorded on every step
whatever ``diagnostics`` says.  ``diagnostics=True`` adds the spectral
keys (``rho_spectral``, the estimated contraction rate per pass, and its
bounds), which cost extra Jacobian-vector products per step -- Part 4
shows them.

**Convergence norms**: all three scale each field's change by that
field's own magnitude, so a group's verdict does not depend on the units
its fields are written in.  ``"l2"`` (the default) compares a global
norm with ``tolerance``; ``"mixed"`` takes a per-field RMS of
``|dx| / (rtol * |x|)`` and converges when that is <= 1; ``"interface"``
does the same over the coupling-edge fields only.  Part 3 runs all
three.

**Reading it, and enforcing it**: ``gm.print_coupling_report()`` prints
the same diagnostics as a table, one row per group, with the documented
caveats flagged beneath it -- a group that hit ``max_iterations``,
``converged=False``, a residual at its float floor.  And
``strict_convergence=True`` turns an unconverged exit into an error
instead of a report: the step raises, and the graph keeps the state it
had.  Part 5 shows a converged group, a capped one and a strict one.

Setup: two masses joined by one spring (two ``SpringDamperNode`` nodes
anchored to each other, ``rest_length`` +1 and -1).  A Gauss-Seidel pass
scales the error by ``(k * dt**2 / m)**2 = 0.01`` here, so the iteration
needs about four passes to reach ``tolerance=1e-6``.

Usage
-----
    python -m maddening.examples.coupling.convergence_diagnostics_demo
    python -m maddening.examples.coupling.convergence_diagnostics_demo --steps 100
"""

import argparse
import logging
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

DT = 0.005
K = 4000.0     # k * dt^2 / m = 0.1, so a GS pass scales the error by 0.01
DAMPING = 25.0  # > k * dt = 20, which the coupled explicit pair needs to be stable
MASS = 1.0
GROUP = "A+B"


def build(max_iterations=15, **group_kwargs):
    """The two-mass spring with a coupling group configured as given."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode(
        name="A", timestep=DT, stiffness=K, damping=DAMPING, mass=MASS,
        rest_length=1.0, initial_position=0.0,
    ))
    gm.add_node(SpringDamperNode(
        name="B", timestep=DT, stiffness=K, damping=DAMPING, mass=MASS,
        rest_length=-1.0, initial_position=5.0,
    ))
    gm.add_edge("A", "B", "position", "anchor_position")
    gm.add_edge("B", "A", "position", "anchor_position")
    gm.add_coupling_group(["A", "B"], max_iterations=max_iterations,
                          **group_kwargs)
    gm.compile()
    return gm


def run(gm, n_steps):
    """Step *n_steps* times; return per-step diagnostics and the final state."""
    infos = []
    for _ in range(n_steps):
        gm.step()
        infos.append(gm.coupling_diagnostics()[GROUP])
    final = (float(gm.get_node_state("A")["position"]),
             float(gm.get_node_state("B")["position"]))
    return infos, final


def demo_step_by_step():
    print("=" * 65)
    print("Part 1: Step-by-Step Diagnostics (tolerance=1e-6)")
    print("=" * 65)
    print()
    gm = build(max_iterations=15, tolerance=1e-6)
    print(f"  {'Step':>5} {'Iters':>6} {'Converged':>10} {'Residual':>10} "
          f"{'Error est.':>11} {'A pos':>8} {'B pos':>8}")
    print(f"  {'-' * 62}")
    for i in range(10):
        gm.step()
        info = gm.coupling_diagnostics()[GROUP]
        pos_a = float(gm.get_node_state("A")["position"])
        pos_b = float(gm.get_node_state("B")["position"])
        print(f"  {i + 1:5d} {info['iterations']:6d} "
              f"{str(bool(info['converged'])):>10} {info['residual']:10.2e} "
              f"{info['error_estimate']:11.2e} {pos_a:8.4f} {pos_b:8.4f}")
        assert info["converged"], f"step {i + 1} did not converge"
    print()
    print("  Every step converged well inside the 15-pass budget.")
    print()


def demo_insufficient_iterations(n_steps=20):
    print("=" * 65)
    print("Part 2: Diagnosing Insufficient Iterations")
    print("=" * 65)
    print()
    print(f"  The same problem with different max_iterations, {n_steps} steps each.")
    print()
    print(f"  {'max_iterations':>14} {'Avg iters':>10} {'Converged':>10} "
          f"{'Final A':>9} {'Final B':>9}  Status")
    print(f"  {'-' * 66}")
    results = {}
    for max_it in (2, 3, 4, 15):
        infos, final = run(build(max_iterations=max_it, tolerance=1e-6), n_steps)
        n_conv = sum(bool(i["converged"]) for i in infos)
        avg_it = sum(i["iterations"] for i in infos) / n_steps
        capped = all(i["iterations"] >= max_it for i in infos)
        if n_conv == n_steps:
            status = "converged"
        elif capped:
            status = "capped every step: iteration-starved"
        else:
            status = "partly converged"
        results[max_it] = (n_conv, final)
        print(f"  {max_it:14d} {avg_it:10.1f} {n_conv:6d}/{n_steps:<3d} "
              f"{final[0]:9.5f} {final[1]:9.5f}  {status}")
    print()

    assert results[2][0] == 0 and results[3][0] == 0
    assert results[4][0] == n_steps and results[15][0] == n_steps
    drift = abs(results[2][1][0] - results[15][1][0])
    assert drift > 1e-3, drift
    print("  With 2 or 3 passes no step converges, and the unconverged steps")
    print(f"  add up: after {n_steps} steps max_iterations=2 ends {drift:.4f} m")
    print("  from the converged run.  With 4 or more every step converges and")
    print("  the extra budget is never used.")
    print("  Rule of thumb: iterations == max_iterations with converged=False")
    print("  means raise max_iterations or add acceleration.")
    print()


def demo_norms(n_steps=20):
    print("=" * 65)
    print("Part 3: Convergence Norms")
    print("=" * 65)
    print()
    configs = [
        ("l2, tolerance=1e-6", dict(convergence_norm="l2", tolerance=1e-6)),
        ("mixed, rtol=1e-6", dict(convergence_norm="mixed", rtol=1e-6)),
        ("interface, rtol=1e-6", dict(convergence_norm="interface", rtol=1e-6)),
    ]
    print(f"  {'Norm':<24} {'Avg iters':>10} {'Converged':>10} {'Last residual':>14}")
    print(f"  {'-' * 61}")
    for label, kwargs in configs:
        infos, _ = run(build(max_iterations=15, **kwargs), n_steps)
        n_conv = sum(bool(i["converged"]) for i in infos)
        avg_it = sum(i["iterations"] for i in infos) / n_steps
        print(f"  {label:<24} {avg_it:10.1f} {n_conv:6d}/{n_steps:<3d} "
              f"{infos[-1]['residual']:14.2e}")
        assert n_conv == n_steps, f"{label}: {n_steps - n_conv} steps unconverged"
    print()
    print("  'l2' compares its residual with tolerance; 'mixed' and")
    print("  'interface' report a scaled residual and converge at <= 1, so")
    print("  their residual column is on a different scale.  'interface'")
    print("  looks only at the fields the coupling edges carry -- here the")
    print("  positions, not the velocities -- so it can stop a pass earlier.")
    print()


def demo_spectral_keys():
    print("=" * 65)
    print("Part 4: What diagnostics=True adds")
    print("=" * 65)
    print()
    gm = build(max_iterations=15, tolerance=1e-6, diagnostics=True)
    gm.step()
    info = gm.coupling_diagnostics()[GROUP]
    predicted = (K * DT ** 2 / MASS) ** 2
    print(f"  rho_spectral (measured contraction per pass): {info['rho_spectral']:.5f}")
    print(f"  predicted (k*dt^2/m)^2:                       {predicted:.5f}")
    print(f"  spectral_error_bound:                         {info['spectral_error_bound']:.2e}")
    assert abs(info["rho_spectral"] - predicted) < 1e-3, info["rho_spectral"]
    print()
    print("  The spectral estimate matches the analytic rate of this linear")
    print("  problem.  It costs extra Jacobian-vector products every step, so")
    print("  switch it on while investigating, not for production runs.")
    print()


def demo_report_and_strict():
    print("=" * 65)
    print("Part 5: print_coupling_report() and strict_convergence")
    print("=" * 65)
    print()
    print("  A converged group (max_iterations=15), after one step:")
    print()
    gm = build(max_iterations=15, tolerance=1e-6)
    gm.step()
    gm.print_coupling_report()
    (row,) = gm.coupling_report()
    assert row["converged"] and row["iterations"] < row["max_iterations"]
    assert not any(f.startswith("hit max_iterations") for f in row["flags"])
    print()
    print(f"  Converged in {row['iterations']} of {row['max_iterations']} passes, "
          f"with no 'hit max_iterations' flag.")
    if any(f.startswith("precision_limited=True") for f in row["flags"]):
        print("  Its precision_limited flag says the residual is at its float floor:")
        print("  the solve went as far as float32 can measure, which is not a failure.")
    print()
    print("  The same graph capped at max_iterations=2:")
    print()
    gm = build(max_iterations=2, tolerance=1e-6)
    gm.step()
    gm.print_coupling_report()
    (row,) = gm.coupling_report()
    assert not row["converged"] and row["iterations"] == row["max_iterations"] == 2
    assert any(f.startswith("hit max_iterations") for f in row["flags"])
    assert any(f.startswith("converged=False") for f in row["flags"])
    print()
    print("  Both caveats are flagged.  Nothing stopped the run: the step was")
    print("  taken with the unconverged state, which is what a report can do.")
    print()
    print("  With strict_convergence=True the same step raises instead:")
    gm = build(max_iterations=2, tolerance=1e-6, strict_convergence=True)
    before = {n: gm.get_node_state(n) for n in ("A", "B")}
    # The check runs inside the compiled step (equinox.error_if); JAX also
    # logs the callback's traceback, silenced here to keep the output short.
    callback_log = logging.getLogger("jax._src.callback")
    level = callback_log.level
    callback_log.setLevel(logging.CRITICAL)
    try:
        gm.step()
    except RuntimeError as exc:
        reason = next(line for line in str(exc).splitlines() if "without converging" in line)
        print(f"    {type(exc).__name__}: ...{reason.split('Error: ', 1)[-1][:150]}...")
    else:
        raise AssertionError("strict_convergence let an unconverged step through")
    finally:
        callback_log.setLevel(level)
    after = {n: gm.get_node_state(n) for n in ("A", "B")}
    unchanged = all(float(after[n][f]) == float(before[n][f])
                    for n in before for f in before[n])
    print(f"  The graph keeps the state it had before the step: {unchanged}")
    assert unchanged
    print("  Use it for calibration and training runs, where a gradient through")
    print("  an unconverged step would be silently wrong.")
    print()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Coupling convergence diagnostics")
    parser.add_argument("--steps", type=int, default=20,
                        help="Steps per configuration in Parts 2 and 3 (default: 20)")
    args = parser.parse_args(argv)
    demo_step_by_step()
    demo_insufficient_iterations(args.steps)
    demo_norms(args.steps)
    demo_spectral_keys()
    demo_report_and_strict()
    print("All demos complete.")


if __name__ == "__main__":
    main()
