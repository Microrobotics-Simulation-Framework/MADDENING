#!/usr/bin/env python
"""Act on a coupling report: its reason codes, what to change, and ``strict_convergence``.

``convergence_diagnostics_demo.py`` reads the numbers
``GraphManager.coupling_diagnostics()`` returns.  This example acts on
them, on a pair of masses joined by one spring, built as two
``SpringDamperNode`` nodes anchored on each other in a coupling group.
One plain Gauss-Seidel pass scales the coupling error by
``(k * dt**2 / m)**2``.

A report's ``spectral_usable`` says whether ``spectral_error_bound``, a
bound on the distance to the fixed point, can be read;
``gradient_bound_usable`` says the same of
``gradient_relative_error_bound``, a bound on the relative error of the
IFT gradient.  Where a flag is ``False`` the report lists why in
``reason_codes`` (EXPERIMENTAL in 0.4.0), as constants of
``maddening.core.coupling.reason_codes``, sorted into three kinds:
``EXPECTED`` (what the group is), ``CONFIGURATION`` (what the caller
chose) and ``WORRY`` (what deserves a look).  A program branches on the
codes, not on the sentence in ``not_usable_reason``.

1. **The default group, read by its codes.**  Every step converges, and
   ``spectral_usable`` is ``False`` for two reasons, both of the
   configuration kind: ``diagnostics_off`` (the default: the bounds are
   not computed) and ``at_float_floor`` (the residual is at its float32
   rounding, where the bound would rest on a count of evaluations that no
   stock node declares).
2. **Acting on them.**  The group built as a calibration run would have
   it: ``gauss-seidel`` / ``iqn-ils`` / ``l2`` (the coupling guide's row
   for a contraction that may be high), ``strict_convergence=True``,
   ``diagnostics=True``, and ``tolerance=1e-4``, which leaves the
   residual above its float floor.  Both flags are ``True``, and the two
   bounds can be read.  The floor is measured at each step's state: as
   the pair settles and its velocities shrink, the floor of the velocity
   rises, and the same residual is at its floor again.  The flags are per
   step.
3. **A worry that ``converged`` does not show.**  The stiffness written
   up to 48 000 through ``gm.params``, as a fit's step writes it, with no
   recompile: a plain pass now *grows* the error 1.44 times, which
   ``rho_spectral`` measures.  IQN-ILS converges anyway, so
   ``strict_convergence`` lets each step through, and the report says
   ``not_contracting``, a worry.  For this pair it is a real one: a pass
   that does not contract means a step that grows, and the pair's common
   velocity is multiplied by ``g = (m - c*dt) / (m - k*dt**2) = -1.25``
   every step while every step converges (MADD-ANO-098).  Without the
   accelerator, ``strict_convergence`` raises instead, and the graph keeps
   the state it had.
4. **What to change for that worry.**  Make one plain pass contract: for
   this pair, halve the timestep, so a pass scales the error by 0.09 and
   ``k*dt <= c`` holds too.  The worry is gone, and a common velocity
   given to the pair decays by ``g = 0.893`` a step.

The coupling guide's table "The reason codes of a report"
(``docs/developer_guide/coupling_algorithm_guide.md``) says what to do
about every code.

Usage
-----
    python -m maddening.examples.coupling.acting_on_diagnostics_demo
    python -m maddening.examples.coupling.acting_on_diagnostics_demo --parts 1,2,3
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from maddening.core.coupling import reason_codes
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

DT = 0.005
K = 4000.0          # k * dt**2 / m = 0.1: a plain pass scales the error by 0.01
STIFF = 48000.0     # k * dt**2 / m = 1.2: a plain pass grows it 1.44 times
# The pair is stable for k*dt <= c <= (2*m - k*dt**2)/dt (MADD-ANO-098):
# 20 <= c <= 380 at K and DT, and 120 <= c <= 680 at STIFF and DT / 2.
DAMPING = 150.0
MASS = 1.0
GROUP = "A+B"
#: A calibration run's settings (part 2).
CALIBRATION = dict(iteration_mode="gauss-seidel", acceleration="iqn-ils",
                   convergence_norm="l2", tolerance=1e-4, max_iterations=20,
                   diagnostics=True, strict_convergence=True)

#: What to change, for the codes this example meets.  The coupling guide's
#: table "The reason codes of a report" has every code.
ACTIONS = {
    reason_codes.DIAGNOSTICS_OFF:
        "build the group with diagnostics=True for a diagnostic run",
    reason_codes.AT_FLOAT_FLOOR:
        "loosen the tolerance until the residual stands off its float "
        "floor, or run in float64, or declare update_evaluations() on "
        "every member",
    reason_codes.NOT_CONTRACTING:
        "make one plain pass contract (here: a smaller timestep); the "
        "accelerator is carrying the solve, and nothing bounds it",
    reason_codes.GRADIENT_BOUND_NOT_COMPUTED:
        "here it follows from the spectral flag's reasons",
}


def kind_of(code: str) -> str:
    """``expected``, ``configuration`` or ``worry``: the code's kind."""
    for kind, codes in (("expected", reason_codes.EXPECTED),
                        ("configuration", reason_codes.CONFIGURATION),
                        ("worry", reason_codes.WORRY)):
        if code in codes:
            return kind
    return "unknown"            # a code newer than this example


def act_on(entry: dict) -> set:
    """Print why each usable flag of one report entry is ``False``, the kind
    of each reason and what to change; return the kinds met."""
    kinds = set()
    for flag in ("spectral_usable", "gradient_bound_usable"):
        codes = entry["reason_codes"][flag]
        print(f"    {flag}={entry[flag]}" + ("" if entry[flag] else ", because:"))
        assert bool(entry[flag]) == (not codes), "a False flag has a code, a True one none"
        for code in codes:
            kinds.add(kind_of(code))
            print(f"      {code:28s} {kind_of(code):13s} -> "
                  f"{ACTIONS.get(code, 'see the guide table')}")
    return kinds


def build(dt=DT, stiffness=K, **group_kwargs) -> GraphManager:
    """The two masses and their spring, with a coupling group configured as given."""
    gm = GraphManager()
    for name, rest, start in (("A", 1.0, 0.0), ("B", -1.0, 5.0)):
        gm.add_node(SpringDamperNode(name, dt, stiffness=stiffness, damping=DAMPING,
                                     mass=MASS, rest_length=rest, initial_position=start))
    gm.add_edge("A", "B", "position", "anchor_position")
    gm.add_edge("B", "A", "position", "anchor_position")
    gm.add_coupling_group(["A", "B"], **group_kwargs)
    gm.compile()
    return gm


@functools.lru_cache(maxsize=None)
def calibration_graph() -> GraphManager:
    """Parts 2 and 3 share one compiled graph: part 3 only writes a parameter."""
    return build(**CALIBRATION)


def report(gm: GraphManager, n_steps: int = 3) -> dict:
    """Step *n_steps* times and print the last step's report in one line."""
    gm.run(n_steps)
    e = gm.coupling_diagnostics()[GROUP]
    print(f"    converged={bool(e['converged'])} in {e['iterations']} passes; residual "
          f"{e['residual']:.2e}, its float floor {e['residual_precision_floor']:.2e}; "
          f"rho_spectral {e['rho_spectral']:.4g}")
    return e


def common_velocity(gm: GraphManager) -> float:
    """The pair's centre-of-mass velocity."""
    return (float(gm.get_node_state("A")["velocity"])
            + float(gm.get_node_state("B")["velocity"])) / 2


def predicted_growth(dt: float, stiffness: float) -> float:
    """``g = (m - c*dt) / (m - k*dt**2)``: what one converged step of this
    pair does to its common velocity (MADD-ANO-098)."""
    return (MASS - DAMPING * dt) / (MASS - stiffness * dt ** 2)


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def part_1() -> None:
    section("1. The default group, read by its codes")
    print("  add_coupling_group(['A', 'B']) with the defaults (tolerance=1e-6):")
    entry = report(build())
    kinds = act_on(entry)
    codes = entry["reason_codes"]["spectral_usable"]
    assert entry["converged"] and not entry["spectral_usable"]
    assert codes == [reason_codes.DIAGNOSTICS_OFF, reason_codes.AT_FLOAT_FLOOR], codes
    assert kinds == {"configuration"}
    print("  Every reason is a setting: nothing is wrong with the solve.  The same")
    print(f"  in words (not_usable_reason): \"{entry['not_usable_reason'][:140]}...\"")


def part_2() -> None:
    section("2. Acting on them: the group as a calibration run would have it")
    print("  " + ", ".join(f"{k}={v!r}" for k, v in CALIBRATION.items()) + ":")
    gm = calibration_graph()
    entry = report(gm)
    act_on(entry)
    assert entry["converged"] and entry["spectral_usable"] and entry["gradient_bound_usable"]
    assert entry["residual"] > entry["residual_precision_floor"]
    print("  The residual stands off its floor, so both bounds can be read:")
    print(f"    distance to the fixed point <= {entry['spectral_error_bound']:.2e} "
          f"(spectral_error_bound, in the group's norm)")
    print(f"    relative error of the IFT gradient <= "
          f"{entry['gradient_relative_error_bound']:.2e} (gradient_relative_error_bound)")
    print("  No node in maddening.nodes declares update_evaluations() in 0.4.0, so with")
    print("  stock nodes the ways off the floor are a looser tolerance, as here, or float64.")

    usable, floors = [], []
    for _ in range(17):
        gm.step()
        e = gm.coupling_diagnostics()[GROUP]
        usable.append(bool(e["spectral_usable"]))
        floors.append(e["residual_precision_floor"])
    first_down = usable.index(False) if False in usable else None
    print("  The floor is measured at each step's state.  As the pair settles its velocities")
    print("  shrink, while the spring forces that set them, and their rounding, do not; and")
    print(f"  the norm measures each field against its own size.  Over the next {len(usable)} steps the")
    print(f"  floor rose from {floors[0]:.1e} to {max(floors):.1e}, and from step "
          f"{4 + first_down if first_down is not None else '-'} the same residual was")
    print("  at its floor again (at_float_floor).  The flags are per step: read them on every")
    print("  step you rely on.")
    assert first_down is not None, "the floor stayed below the residual"


def part_3() -> None:
    section("3. A worry that converged=True does not show")
    gm = calibration_graph()
    gm.reset_state()            # back to the start; the compiled step is kept
    gm.run(3)
    for name in ("A", "B"):
        gm.params["nodes"][name]["stiffness"] = jnp.float32(STIFF)
    print(f"  The stiffness written up to {STIFF:g} through gm.params, as a fit's step")
    print("  writes it; the same compiled step:")
    entry = report(gm)
    kinds = act_on(entry)
    codes = entry["reason_codes"]["spectral_usable"]
    pass_gain = (STIFF * DT ** 2 / MASS) ** 2
    worries = set(codes) & reason_codes.WORRY
    assert entry["converged"] and not entry["spectral_usable"] and "worry" in kinds
    assert worries == {reason_codes.NOT_CONTRACTING}, codes
    assert abs(entry["rho_spectral"] - pass_gain) < 1e-2 * pass_gain, entry["rho_spectral"]
    print(f"  rho_spectral {entry['rho_spectral']:.3f} against (k*dt^2/m)^2 = {pass_gain:.3f}: "
          f"a plain pass")
    print("  grows the error.  IQN-ILS converged anyway, and strict_convergence let the")
    print("  step through: it checks convergence, not the bounds.")

    n = 40
    velocities, converged = [common_velocity(gm)], []
    for _ in range(n):
        gm.step()
        converged.append(bool(gm.coupling_diagnostics()[GROUP]["converged"]))
        velocities.append(common_velocity(gm))
    per_step = velocities[-1] / velocities[-2]
    g = predicted_growth(DT, STIFF)
    print(f"  And the step itself grows.  Over {n} more steps, each converged="
          f"{all(converged)}, the pair's")
    print(f"  common velocity went from {velocities[0]:.1e} to {velocities[-1]:.1e} m/s, "
          f"{per_step:.3f} times per step")
    print(f"  at the end, against g = (m - c*dt)/(m - k*dt^2) = {g:.3f} (MADD-ANO-098).  The")
    print("  graph was compiled with a stable stiffness, so compile() had nothing to warn")
    print("  of; not_contracting is what says so.  A calibration loop stops on a worry:")
    print(f"      if set(codes) & reason_codes.WORRY: raise ...     # here {sorted(worries)}")
    assert all(converged) and abs(per_step - g) < 1e-2 * abs(g), (per_step, g)

    print()
    print("  The same stiffness without the accelerator, strict_convergence=True:")
    plain = build(stiffness=STIFF, max_iterations=20, tolerance=1e-4,
                  strict_convergence=True)
    before = {n: plain.get_node_state(n) for n in ("A", "B")}
    # The check runs inside the compiled step (equinox.error_if); JAX also
    # logs the callback's traceback, silenced here to keep the output short.
    callback_log = logging.getLogger("jax._src.callback")
    level = callback_log.level
    callback_log.setLevel(logging.CRITICAL)
    try:
        plain.step()
    except RuntimeError as exc:
        reason = next(line for line in str(exc).splitlines() if "without converging" in line)
        print(f"    {type(exc).__name__}: ...{reason.split('Error: ', 1)[-1][:110]}...")
    else:
        raise AssertionError("strict_convergence let an unconverged step through")
    finally:
        callback_log.setLevel(level)
    after = {n: plain.get_node_state(n) for n in ("A", "B")}
    unchanged = all(float(after[n][f]) == float(before[n][f]) for n in before for f in before[n])
    print(f"    the graph keeps the state it had before the step: {unchanged}")
    assert unchanged


def part_4() -> None:
    section("4. What to change for that worry: make the pass contract")
    dt = DT / 2
    print(f"  The same stiffness at dt={dt:g}: a plain pass scales the error by "
          f"(k*dt^2/m)^2 = {(STIFF * dt ** 2 / MASS) ** 2:.2f},")
    print(f"  and k*dt = {STIFF * dt:g} <= c = {DAMPING:g}, inside the pair's stable range:")
    gm = build(dt=dt, stiffness=STIFF, **CALIBRATION)
    entry = report(gm)
    kinds = act_on(entry)
    codes = entry["reason_codes"]["spectral_usable"]
    assert entry["converged"] and entry["rho_spectral"] < 1.0
    assert reason_codes.NOT_CONTRACTING not in codes and kinds <= {"configuration"}, codes
    print("  No worry is left" + ("." if not codes else
          ": what remains is a setting (IQN-ILS took the residual to its float floor)."))
    n = 40
    for name in ("A", "B"):             # give the pair a common velocity of 1 m/s
        state = gm.get_node_state(name)
        gm.set_node_state(name, {**state, "velocity": state["velocity"] + 1.0})
    velocities = [common_velocity(gm)]
    for _ in range(n):
        gm.step()
        velocities.append(common_velocity(gm))
    per_step = velocities[-1] / velocities[-2]
    g = predicted_growth(dt, STIFF)
    print(f"  Given a common velocity of {velocities[0]:.2f} m/s, the pair has "
          f"{velocities[-1]:.1e} m/s {n} steps later:")
    print(f"  {per_step:.3f} times per step at the end, against g = {g:.3f}.  It decays.")
    assert 0.0 < g < 1.0 and abs(per_step - g) < 1e-2 * g, (per_step, g)
    assert abs(velocities[-1]) < abs(velocities[0])


PARTS = {"1": part_1, "2": part_2, "3": part_3, "4": part_4}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", default="1,2,3,4",
                        help="Comma-separated parts to run, in order (default 1,2,3,4)")
    args = parser.parse_args(argv)
    chosen = [p.strip() for p in args.parts.split(",") if p.strip()]
    unknown = sorted(set(chosen) - set(PARTS))
    if unknown:
        parser.error(f"unknown parts {unknown}; choose from {sorted(PARTS)}")
    for part in chosen:
        PARTS[part]()
    print()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
