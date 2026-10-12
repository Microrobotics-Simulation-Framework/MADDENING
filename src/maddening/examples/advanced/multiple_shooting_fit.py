#!/usr/bin/env python3
"""Multiple shooting on a coupled graph: weigh a mass you cannot reach.

Two masses joined by one spring, as two ``SpringDamperNode`` nodes whose
anchors are each other's position, solved together in a coupling group
(the pair of ``coupling.coupling_demo``).  Mass A is known (1 kg); mass B
is not.  Both positions are recorded, with noise, and B's mass is
recovered from the record:

1. **The objective.**  :func:`~maddening.sysid.windowed_loss` cuts the
   record into windows and runs the coupled step over each.  On the clean
   record, at the true mass, it is exactly zero both teacher-forced (each
   window restarts from the recorded state) and under multiple shooting
   (each window restarts from a free state of its own, here seeded from
   the record by :func:`~maddening.sysid.init_window_states`).
2. **Teacher forcing on noisy data.**  :func:`~maddening.sysid.fit` on the
   teacher-forced loss restarts every window from a noisy recorded state,
   and the fitted mass absorbs that noise: 8.9% off on this record.
3. **Multiple shooting.**  :func:`~maddening.sysid.fit_multiple_shooting`
   fits the mass and the window starts together, with a continuity
   penalty that joins each window's end to the next window's start, so
   the answer is one continuous trajectory through the noisy samples.  The
   mass comes back 0.4% off, and the positions of the fitted window
   starts lie about three times closer to the clean positions than the
   noisy samples they started from.  (The velocities, which are not
   measured, are not all brought closer: a 20-sample window pins B's
   velocity at its start only loosely.)

The tolerance the example asserts, 5%, comes from the spread of the
answer over noise draws.  Over 20 draws of the noise on this record (200
steps, sigma 1 cm and 1 cm/s, 100 iterations) the multiple-shooting
estimate had an RMS error of 1.5% and a largest error of 3.5%, so 5% is
over three RMS errors; teacher forcing had an RMS error of 6.5% and a
largest of 13%, and was the closer of the two in 2 of the 20.  Neither is
biased on average (both means within 0.4% of the truth): what noisy
window starts cost teacher forcing is precision.

The gradient of every window passes through the coupling group's solve,
which ``solver="ift"`` (the default) differentiates with the implicit
function theorem.  The interface here has two unknowns, so the group uses
``linear_solver="dense"``, which solves that 2 x 2 system directly and
compiles faster than the default GMRES.  ``strict_convergence=True`` makes
a step that leaves the group unconverged raise instead of returning a
gradient that is not the derivative of a fixed point.

See ``docs/user_guide/parameters.md`` ("System identification") for the
fitters and what they return.

Usage
-----
    python -m maddening.examples.advanced.multiple_shooting_fit
    python -m maddening.examples.advanced.multiple_shooting_fit --n-iter 100
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import (
    fit,
    fit_multiple_shooting,
    init_window_states,
    observations_from_history,
    windowed_loss,
)

DT = 0.01            # [s]
STEPS = 200          # recorded steps (2 s, a little over two periods)
WINDOW = 20          # samples per window: 10 windows
MASS_B = 2.0         # [kg], the value to recover
START_B = 1.4        # [kg], the fits' starting guess (30% low)
NOISE_STD = 0.01     # [m] on positions, [m/s] on velocities
TOLERANCE = 0.05     # relative; see the module docstring
LR = 0.1


def build() -> GraphManager:
    """One spring between mass A (top) and mass B (bottom), coupled."""
    gm = GraphManager()
    # The pair is one spring when the two rest lengths are equal and
    # opposite: A rests 2 m above B, B 2 m below A.  Started 3 m apart.
    gm.add_node(SpringDamperNode("a", DT, stiffness=10.0, damping=0.5, mass=1.0,
                                 rest_length=2.0, initial_position=3.0))
    gm.add_node(SpringDamperNode("b", DT, stiffness=10.0, damping=0.5, mass=MASS_B,
                                 rest_length=-2.0, initial_position=0.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=20, tolerance=1e-6,
                          linear_solver="dense", strict_convergence=True)
    gm.compile()
    # Everything is known except B's mass: freeze the rest, and give the
    # mass a lower bound and a log transform so the fit keeps it positive.
    for node in ("a", "b"):
        for key in ("stiffness", "damping", "mass", "rest_length"):
            gm.set_param_spec(node, key, ParamSpec(trainable=False))
    gm.set_param_spec("b", "mass", ParamSpec(bounds=(0.0, None), transform="log"))
    return gm


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def mass_b(params: dict) -> float:
    return float(params["nodes"]["b"]["mass"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-iter", type=int, default=100,
                        help="Adam iterations for each fit (default 100)")
    args = parser.parse_args(argv)
    if args.n_iter < 100:
        # Fewer iterations stop the multiple-shooting fit before it reaches
        # the noise floor (at 10 it is still 11% off), which says nothing
        # about the method.
        parser.error("--n-iter must be at least 100")

    gm = build()
    print("Trainable parameters:", [name for name, on in
                                    zip(*_flat_mask(gm.trainable_mask())) if on])

    # -- the record ---------------------------------------------------------
    # Sample 0 is the state the record starts from (compile() leaves the
    # graph there); run_scan_with_history records the state after each step.
    init = {name: gm.get_node_state(name) for name in gm.node_names}
    _, history = gm.run_scan_with_history(STEPS)
    clean = observations_from_history(init, history)        # T = STEPS + 1
    rng = np.random.default_rng(0)
    noisy = jax.tree.map(
        lambda x: x + jnp.asarray(rng.normal(0.0, NOISE_STD, x.shape), x.dtype), clean)

    def positions(h):
        """What is measured: both positions, over the time axis."""
        return h["a"]["position"], h["b"]["position"]

    truth = gm.params
    start = jax.tree.map(lambda x: x, truth)
    start["nodes"]["b"]["mass"] = jnp.asarray(START_B, jnp.float32)
    print(f"Recorded {STEPS} steps ({STEPS * DT:.0f} s) of both masses, "
          f"noise sigma {NOISE_STD}; B's true mass {MASS_B} kg, the fits start "
          f"at {START_B} kg.")

    # -- 1. the objective -----------------------------------------------------
    section("1. windowed_loss: zero at the truth on the clean record")
    states = init_window_states(clean, WINDOW)     # one free start per window
    teacher = windowed_loss(gm, truth, clean, obs_fn=positions, window=WINDOW)
    # The same loss at two parameter values: jitted, it compiles once.
    shooting_loss = jax.jit(lambda p: windowed_loss(
        gm, p, clean, obs_fn=positions, window=WINDOW,
        window_states=states, continuity_weight=1.0))
    shooting, off = shooting_loss(truth), shooting_loss(start)
    print(f"  teacher-forced, true mass:     {float(teacher):.3g}")
    print(f"  multiple shooting, true mass:  {float(shooting):.3g}")
    print(f"  multiple shooting, mass {START_B}:  {float(off):.3g}")
    n_windows = jax.tree.leaves(states)[0].shape[0]
    print(f"  {n_windows} windows of {WINDOW} samples; each window's free start "
          f"is a full state of both nodes.")
    assert float(teacher) == 0.0 and float(shooting) == 0.0
    assert float(off) > 0.0 and n_windows == STEPS // WINDOW

    # -- 2. teacher forcing on the noisy record --------------------------------
    section("2. fit on the teacher-forced loss, noisy record")
    tf = fit(gm, lambda p: windowed_loss(gm, p, noisy, obs_fn=positions, window=WINDOW),
             params=start, n_iter=args.n_iter, lr=LR)
    tf_error = mass_b(tf.params) / MASS_B - 1.0
    print(f"  mass B = {mass_b(tf.params):.4f} kg ({100 * tf_error:+.1f}%)")
    print("  Every window restarts from a noisy recorded state, and the mass "
          "absorbs the noise.")

    # -- 3. multiple shooting ----------------------------------------------
    section("3. fit_multiple_shooting, noisy record")
    ms, fitted_states = fit_multiple_shooting(
        gm, noisy, obs_fn=positions, window=WINDOW, params=start,
        continuity_weight=1.0, n_iter=args.n_iter, lr=LR)
    print(ms)
    ms_error = mass_b(ms.params) / MASS_B - 1.0
    print(f"  mass B = {mass_b(ms.params):.4f} kg ({100 * ms_error:+.2f}%), "
          f"tolerance {100 * TOLERANCE:.0f}%")
    # How far the window starts' positions are from the clean ones, RMS over
    # both nodes and every window: as seeded from the noisy record, and fitted.
    def rms_from_clean(starts):
        errors = [starts[n]["position"] - clean[n]["position"][:-1:WINDOW] for n in ("a", "b")]
        return float(jnp.sqrt(jnp.mean(jnp.concatenate(errors) ** 2)))
    seeded = rms_from_clean(init_window_states(noisy, WINDOW))
    fitted = rms_from_clean(fitted_states)
    print(f"  window starts' positions, RMS distance from the clean record: "
          f"{seeded:.4f} m as seeded from the noisy record, {fitted:.4f} m fitted")
    assert abs(ms_error) < TOLERANCE, ms_error
    assert abs(ms_error) < abs(tf_error), (ms_error, tf_error)
    assert ms.best_loss < ms.losses[0]
    assert ms.excited_rank == 1 and ms.hold_declined is False
    assert jax.tree.leaves(fitted_states)[0].shape == (n_windows,)
    assert fitted < 0.5 * seeded, (fitted, seeded)
    print("  The window starts moved off the noisy samples towards one continuous "
          "trajectory, and the mass came back.")

    print()
    print("All checks passed.")
    return 0


def _flat_mask(mask: dict):
    """(names, flags) of a params-shaped mask, for printing."""
    names, flags = [], []
    for node, leaves in mask["nodes"].items():
        for key, flag in leaves.items():
            names.append(f"{node}.{key}")
            flags.append(bool(flag))
    return names, flags


if __name__ == "__main__":
    sys.exit(main())
