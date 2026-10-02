#!/usr/bin/env python3
"""Calibrate a graph's parameters from data, and ask what the data can tell.

A spring-damper (stiffness ``k``, damping ``c``, mass ``m``) is run at its
true parameters and its position recorded with measurement noise.  Then:

1. **What can the data identify?**  :func:`~maddening.sysid.fim` at the
   starting guess, over ``(k, c, m)``.  Position data determines only
   ``k/m`` and ``c/m``: scaling all three together changes nothing, so the
   Fisher matrix has rank 2 of 3 and every Cramér-Rao bound is infinite.
2. **Adam with the identifiability guard.**  :func:`~maddening.sysid.fit`
   over all three, after giving ``c`` and ``m`` ``transform="log"`` in
   their :class:`~maddening.core.params.ParamSpec`.  The ratios are
   recovered; the undetermined scale is *held* where the start put it,
   which ``FitResult.excited_rank`` (2) and ``hold_declined`` (False)
   report.
3. **Freeze the mass and recover the rest.**  ``ParamSpec(trainable=
   False)`` on ``m``, a ``mask=`` that selects ``k`` and ``c``, and
   :func:`~maddening.sysid.fit_lm` (Levenberg-Marquardt) from a perturbed
   start, with ``params_table()`` before and after.  Its ``step_tol`` is
   set to ``1e-5``, far below the parameters' statistical uncertainty, and
   ``converged`` reports that this stopping test was met.
4. **How well?**  ``fim`` at the fitted values with ``noise_std`` gives
   each parameter's Cramér-Rao bound in its own units; the fit lands
   within a few standard deviations of the truth.

The residual is a rollout through ``gm.run_sweep``, which takes
``params=`` and -- unlike every other ``run_*`` method -- does not
advance the graph's state, so it can be differentiated and called
repeatedly.  It is wrapped in ``jax.jit``: the fitters and ``fim``
differentiate it, and a jitted rollout is one compiled program rather
than a scan dispatched op by op.  All public API.

See ``docs/user_guide/parameters.md`` for the whole story: which leaves
are parameters, ``ParamSpec``, the guard, ``fit_multiple_shooting`` for
noisy data, and persistence.

Usage
-----
    python -m maddening.examples.advanced.sysid_demo
    python -m maddening.examples.advanced.sysid_demo --samples 100 --n-iter 150
"""

from __future__ import annotations

import argparse
import math
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import fim, fit, fit_lm

TRUE = {"stiffness": 30.0, "damping": 2.0, "mass": 1.0}
START = {"stiffness": 45.0, "damping": 3.0}      # the perturbed guess
NOISE_STD = 0.01                                  # measurement noise [m]
# fit_lm's stopping test: converged once a proposed step moves every fitted
# parameter by at most this fraction of itself.  1e-5 is a choice about the
# answer -- hundreds of times below the Cramer-Rao sigma this data gives
# (0.2-1%) -- where the default (16 float32 ulps, 1.9e-6) asks for the
# parameters' own float resolution.  On this data the default is met at
# some record lengths and not at others (measured: met at 200 and 300
# samples, not at 50, 100 or 500, where the damping retries are exhausted
# first), so ``converged`` would depend on --samples.  At 1e-5 it is met at
# every one of them.
STEP_TOL = 1e-5
DT = 0.01


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def leaf(params: dict, key: str) -> float:
    return float(params["nodes"]["spring"][key])


def with_values(params: dict, **values) -> dict:
    """A copy of *params* with the spring's leaves set to *values*."""
    out = jax.tree.map(lambda x: x, params)
    for key, value in values.items():
        out["nodes"]["spring"][key] = jnp.asarray(value, dtype=jnp.float32)
    return out


def select(params: dict, *keys: str) -> dict:
    """A mask (params-shaped pytree of bools) selecting the spring's *keys*."""
    mask = jax.tree.map(lambda _: False, params)
    for key in keys:
        mask["nodes"]["spring"][key] = True
    return mask


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=300,
                        help="Recorded samples, one per 10 ms step (default 300)")
    parser.add_argument("--n-iter", type=int, default=300,
                        help="Adam iterations in part 2 (default 300)")
    args = parser.parse_args(argv)
    if args.samples < 100 or args.n_iter < 150:
        # A shorter record (under one second of the oscillation) or a smaller
        # Adam budget does not pin k/m and c/m to the tolerances part 2
        # asserts: measured at 50 samples and 100 iterations, c/m = 1.75.
        parser.error("--samples must be at least 100 and --n-iter at least 150")
    n = args.samples

    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", DT, initial_position=0.5, **TRUE))
    gm.compile()

    section("The parameters, as the node declares them")
    gm.print_params_table()

    # -- the data ---------------------------------------------------------
    # run_sweep runs a batch of simulations from the states it is given
    # (a batch of one here) and writes nothing back to the graph.
    init = {"spring": {k: v[None] for k, v in gm.get_node_state("spring").items()}}

    def positions(params: dict) -> jnp.ndarray:
        _, history = gm.run_sweep(n, init, return_history=True, params=params)
        return history["spring"]["position"][0]

    truth = gm.params
    rng = np.random.default_rng(0)
    measured = positions(truth) + jnp.asarray(
        rng.normal(0.0, NOISE_STD, n), dtype=jnp.float32)

    @jax.jit
    def residual(params: dict) -> jnp.ndarray:
        return positions(params) - measured

    print()
    print(f"Recorded {n} samples of the position ({n * DT:.1f} s) at k={TRUE['stiffness']}, "
          f"c={TRUE['damping']}, m={TRUE['mass']}, with noise sigma={NOISE_STD} m.")
    start = with_values(truth, **START)

    # -- 1. identifiability -------------------------------------------------
    section("1. What can position data identify?  fim over (k, c, m)")
    report = fim(residual, start, mask=select(start, "stiffness", "damping", "mass"))
    print(report)
    weakest = np.abs(np.asarray(report.eigvecs[:, 0]))
    print(f"  weakest direction (relative coordinates): {np.round(weakest, 3)}; "
          f"(1, 1, 1)/sqrt(3) = {1 / math.sqrt(3):.3f} each")
    assert report.rank == 2 and len(report.param_names) == 3
    assert np.all(np.isinf(np.asarray(report.crb)))
    assert np.allclose(weakest, 1 / math.sqrt(3), atol=0.02)
    print("  Rank 2 of 3: the data determines k/m and c/m, not the common scale "
          "of (k, c, m), so no parameter is separately determined.")

    # -- 2. Adam with the identifiability guard -----------------------------
    section("2. fit (Adam) over all three, with the guard")
    # The guard holds a direction that is fixed in the optimiser's
    # (unconstrained) coordinates.  The scale degeneracy is a straight line
    # in log coordinates, so give damping a log transform too (the node
    # declares it with none, to keep zero damping representable).
    gm.set_param_spec("spring", "damping", ParamSpec(bounds=(0.0, None), transform="log"))
    gm.set_param_spec("spring", "mass", ParamSpec(bounds=(0.0, None), transform="log"))
    adam = fit(gm, lambda p: 0.5 * jnp.sum(residual(p) ** 2), params=start,
               mask=select(start, "stiffness", "damping", "mass"),
               n_iter=args.n_iter, lr=0.1)
    print(adam)
    print(f"  ('not converged' because tol=0, the default: fit's only stopping "
          f"test is the loss reaching tol, so Adam runs its whole budget.  It "
          f"returns its lowest-loss iterate, number {adam.best_iteration}, not "
          f"necessarily its last.)")
    assert adam.converged is False and adam.n_iter == args.n_iter
    k, c, m = (leaf(adam.params, key) for key in ("stiffness", "damping", "mass"))
    held = (k * c * m) ** (1 / 3)
    held_start = (START["stiffness"] * START["damping"] * 1.0) ** (1 / 3)
    print(f"  k/m = {k / m:.3f} (true {TRUE['stiffness'] / TRUE['mass']:.0f}), "
          f"c/m = {c / m:.4f} (true {TRUE['damping'] / TRUE['mass']:.0f})")
    print(f"  common scale (k*c*m)^(1/3) = {held:.4f}; at the start {held_start:.4f}")
    print(f"  excited rank {adam.excited_rank} of 3, hold declined: {adam.hold_declined}")
    assert adam.excited_rank == 2 and adam.hold_declined is False
    assert abs(k / m - 30.0) < 0.5 and abs(c / m - 2.0) < 0.1
    assert abs(held - held_start) < 1e-3 * held_start
    print("  The ratios are recovered.  The scale the data cannot see was held at "
          "the value supplied, rather than left wherever Adam's steps wandered.")

    # -- 3. freeze the mass, recover k and c --------------------------------
    section("3. Freeze the mass and recover k and c with fit_lm")
    # set_param_spec replaces the whole spec, so restate the bounds.
    gm.set_param_spec("spring", "mass",
                      ParamSpec(trainable=False, bounds=(0.0, None), transform="log"))
    gm.params = with_values(start)            # the perturbed guess, in the graph
    print("Before (gm.params holds the guess; mass is now frozen):")
    gm.print_params_table()

    # ParamSpec bounds are enforced: a negative stiffness is refused.
    try:
        gm.check_params(with_values(start, stiffness=-5.0))
    except ValueError as exc:
        print(f"\n  check_params refuses a negative stiffness: {exc}")
    else:
        raise AssertionError("a stiffness below its bound was accepted")

    # A mask may narrow the trainable set (rest_length is trainable too,
    # and stays fixed here) but never widen it onto a frozen leaf.
    try:
        fit_lm(gm, residual, params=start, mask=select(start, "stiffness", "mass"))
    except ValueError as exc:
        print(f"  a mask naming the frozen mass is refused: {str(exc).splitlines()[0][:110]}...")
    else:
        raise AssertionError("a mask widened onto a frozen leaf was accepted")

    lm = fit_lm(gm, residual, params=start, mask=select(start, "stiffness", "damping"),
                noise_std=NOISE_STD, step_tol=STEP_TOL)
    print()
    print(lm)
    # The loss is 0.5 * sum((r / sigma)^2): pure noise at sigma gives about
    # n / 2, so a fit at the noise floor lands there and goes no lower.
    floor = 2.0 * lm.best_loss / n
    print(f"  best loss {lm.best_loss:.1f} = {floor:.2f} x n/2: at the noise floor.")
    print(f"  'converged' is True: an iteration proposed a step that moved both k "
          f"and c by at most step_tol={STEP_TOL:g} of their values.  That is fit_lm's "
          f"stopping test, and it counts the proposal whether or not it lowered "
          f"the loss -- at the floor a step that small cannot strictly lower a "
          f"loss that is rounding noise.  Like every fit_lm run, its last iterate "
          f"is its lowest (iterate {lm.best_iteration} of {len(lm.losses)} "
          f"evaluated).")
    assert 0.7 < floor < 1.3, floor
    assert lm.converged is True and lm.n_iter < 50          # stopped on the test
    assert lm.best_iteration in (len(lm.losses), len(lm.losses) - 1)
    assert lm.excited_rank == 2 and lm.hold_declined is False
    assert leaf(lm.params, "mass") == TRUE["mass"]            # frozen: untouched
    assert leaf(lm.params, "rest_length") == leaf(start, "rest_length")
    gm.params = lm.params
    print()
    print("After (gm.params holds the fit):")
    gm.print_params_table()

    # -- 4. how well -------------------------------------------------------
    section("4. Cramér-Rao bounds at the fit, in the parameters' own units")
    bounds = fim(residual, lm.params, mask=select(lm.params, "stiffness", "damping"),
                 noise_std=NOISE_STD, scale=None)
    print(bounds)
    sigma = dict(zip(bounds.param_names, np.sqrt(np.asarray(bounds.crb))))
    for key in ("stiffness", "damping"):
        s = sigma[f"['nodes']['spring']['{key}']"]
        got, true = leaf(lm.params, key), TRUE[key]
        print(f"  {key:9s} fitted {got:8.4f} +- {s:.4f} (1 sigma), true {true}: "
              f"{abs(got - true) / s:.1f} sigma away")
        assert math.isfinite(s) and abs(got - true) < 4 * s, (key, got, s)
    print("  Both within 4 sigma of the truth, as the noise level predicts.")

    print()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
