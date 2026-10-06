"""What a change of units can and cannot do to a coupling group's verdict (CPL-009).

Every convergence norm divides a field's change by the field's own
magnitude, so the criterion is dimensionless.  What that buys depends on
the factor:

* a **power of two** rescales every float exactly, so the run is the same
  run: state bits (after undoing the factor), pass count, residual, error
  estimate and verdict;
* a **decimal** factor rounds the inputs differently.  The iterates then
  differ by rounding, the residual (a difference of nearly equal numbers)
  by a visible fraction of itself once it is near its float floor, and the
  error estimate with it.  The pass count can move.  The verdict can
  differ too, and only in one place: a group that stops at its iteration
  cap with the error estimate within that rounding of the threshold.

The rounding is stated as a band and tested as stated: the error estimate
at a decimal factor is within ``1% of itself + 16 float floors``
(:func:`~maddening.core.coupling.acceleration.residual_precision_floor`) of
the unscaled one.  Measured on the three groups below at eleven decimal
factors from 1e-20 to 1e20 (float32, CPU, jaxlib 0.11.0): at most 4.7
floors where the residual is within ten floors of its resolution, and
0.17% where it is far above.  On the first group, which stops at its cap
of 18 passes with the estimate 1% above ``tolerance=1e-5``, the estimate
read 0.85 to 1.51 thresholds and two factors (1e3, 1e6) read
``converged=True`` where the unscaled group reads ``False``: the row used
to say a decimal factor moves "the pass count, not the verdict".
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening import GraphManager, SimulationNode
from maddening.core.coupling.acceleration import residual_precision_floor
from maddening.core.node import BoundaryInputSpec

#: The band the claim states for a decimal factor: a fraction of the
#: estimate, plus a count of the residual's float floors.
BAND_RELATIVE = 0.01
BAND_FLOORS = 16.0

SIZES = (2, 1, 2)
NAMES = ("n0", "n1", "n2")
POWERS_OF_TWO = (2.0 ** -10, 2.0 ** 40)
DECIMALS = (1e-3, 1e3, 1e6, 0.1, 3.0, 9.81, 1e-20)


class _Affine(SimulationNode):
    """``x_new = a x + b + sum_s G_s @ u_s``: equivariant under one factor on ``b`` and ``x``."""

    def __init__(self, name, sources, gains, b, a, x0):
        params = {f"G_{s}": np.asarray(g, np.float32) for s, g in gains.items()}
        super().__init__(name, 1.0, b=np.asarray(b, np.float32), a=np.float32(a), **params)
        self._sources, self._x0 = dict(sources), np.asarray(x0, np.float32)

    def update_evaluations(self):
        return 1.0

    def initial_state(self):
        return {"x": jnp.asarray(self._x0)}

    def boundary_input_spec(self):
        return {f"u_{s}": BoundaryInputSpec(shape=(m,), dtype=jnp.float32,
                                            default=jnp.zeros((m,), jnp.float32))
                for s, m in self._sources.items()}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        acc = p["a"] * state["x"] + p["b"]
        for s in self._sources:
            acc = acc + p[f"G_{s}"] @ boundary_inputs[f"u_{s}"]
        return {"x": acc.astype(jnp.float32)}


def _problem(seed, rho=0.9):
    """Three nodes, each reading the other two; the coupling blocks' spectral radius is *rho*."""
    rng = np.random.default_rng(seed)
    offsets = np.concatenate([[0], np.cumsum(SIZES)])
    n = int(offsets[-1])
    A = np.zeros((n, n))
    for d in range(3):
        for s in range(3):
            if s != d:
                A[offsets[d]:offsets[d + 1], offsets[s]:offsets[s + 1]] = rng.standard_normal(
                    (SIZES[d], SIZES[s]))
    A *= rho / max(abs(np.linalg.eigvals(A)))
    # Representable in float32, so a power of two rescales them exactly.
    b = (rng.standard_normal(n) + 2.0 * (rng.random(n) > 0.5)).astype(np.float32)
    x0 = (rng.standard_normal(n) + 1.0).astype(np.float32)
    return A.astype(np.float32), b, x0, offsets


def _run(problem, scale, *, max_iterations, tolerance=1e-5):
    """Two steps of the group with ``b`` and the initial state multiplied by *scale*."""
    A, b, x0, off = problem
    gm = GraphManager()
    for d, name in enumerate(NAMES):
        sources = {NAMES[s]: SIZES[s] for s in range(3) if s != d}
        gains = {NAMES[s]: A[off[d]:off[d + 1], off[s]:off[s + 1]] for s in range(3) if s != d}
        gm.add_node(_Affine(name, sources, gains,
                            (b[off[d]:off[d + 1]].astype(np.float64) * scale),
                            0.5, (x0[off[d]:off[d + 1]].astype(np.float64) * scale)))
    for d, name in enumerate(NAMES):
        for s in range(3):
            if s != d:
                gm.add_edge(NAMES[s], name, "x", f"u_{NAMES[s]}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_coupling_group(list(NAMES), iteration_mode="gauss-seidel", acceleration="fixed",
                              relaxation=0.7, convergence_norm="l2", tolerance=tolerance,
                              max_iterations=max_iterations)
        gm.compile()
        gm.step()
        gm.step()
        report = dict(gm.coupling_diagnostics()["+".join(NAMES)])
    state = np.concatenate([np.asarray(gm.get_node_state(nm)["x"]).ravel() for nm in NAMES])
    floor = float(residual_precision_floor(gm._state, list(NAMES), "l2"))
    return state, report, floor


#: (seed, cap): a group at its cap with the estimate 1% above the
#: threshold (1.011e-5 for 1e-5); the same group stopped far from its
#: fixed point; a group that converges before its cap (16 passes of 25).
GROUPS = {"marginal-at-the-cap": (3, 18), "far-at-the-cap": (3, 10), "converged": (7, 25)}


@pytest.fixture(scope="module", params=sorted(GROUPS))
def runs(request):
    seed, cap = GROUPS[request.param]
    problem = _problem(seed)
    scales = (1.0,) + POWERS_OF_TWO + DECIMALS
    return request.param, cap, {s: _run(problem, s, max_iterations=cap) for s in scales}


def test_a_power_of_two_change_of_units_is_the_same_run(runs):
    _, _, by_scale = runs
    state, report, _ = by_scale[1.0]
    for scale in POWERS_OF_TWO:
        scaled_state, scaled_report, _ = by_scale[scale]
        assert np.array_equal((scaled_state.astype(np.float64) / scale).astype(np.float32), state)
        for key in ("converged", "iterations", "residual", "error_estimate"):
            assert scaled_report[key] == report[key], (scale, key)


def test_a_decimal_change_of_units_moves_the_estimate_by_no_more_than_the_stated_rounding(runs):
    _, _, by_scale = runs
    _, report, floor = by_scale[1.0]
    assert floor > 0.0
    for scale in DECIMALS:
        _, scaled, _ = by_scale[scale]
        band = BAND_RELATIVE * report["error_estimate"] + BAND_FLOORS * floor
        assert abs(scaled["error_estimate"] - report["error_estimate"]) <= band, (
            scale, scaled["error_estimate"], report["error_estimate"], floor)


def test_a_decimal_change_of_units_moves_the_verdict_only_at_the_cap_within_that_rounding(runs):
    """The verdict at a decimal factor is the unscaled one unless both runs
    stopped at the cap with the estimate inside the band about the threshold."""
    name, cap, by_scale = runs
    _, report, floor = by_scale[1.0]
    tolerance = 1e-5
    band = BAND_RELATIVE * report["error_estimate"] + BAND_FLOORS * floor
    marginal = report["iterations"] == cap and abs(report["error_estimate"] - tolerance) <= band
    # The three groups are the three cases, so neither branch is vacuous.
    assert marginal == (name == "marginal-at-the-cap")
    assert (report["iterations"] < cap) == (name == "converged")
    for scale in DECIMALS:
        _, scaled, _ = by_scale[scale]
        if scaled["converged"] != report["converged"]:
            assert marginal, (name, scale, scaled, report)
            assert scaled["iterations"] == cap
            assert abs(scaled["error_estimate"] - tolerance) <= band
        if name == "converged":
            # Below the cap rounding moves the pass on which the estimate
            # first meets the threshold, never whether it does.
            assert scaled["converged"] is True and scaled["iterations"] < cap
        if name == "far-at-the-cap":
            assert scaled["converged"] is False and scaled["iterations"] == cap
