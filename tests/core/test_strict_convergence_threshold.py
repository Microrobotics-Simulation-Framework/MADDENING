"""``strict_convergence`` raises exactly where ``converged`` is ``False``, at the threshold.

``strict_convergence=True`` refuses a step whose kept solve did not meet
the group's criterion: the *error estimate* ``omega * r / (1 - rho)``
(MADD-ANO-005), not the raw residual ``r``, against the threshold
(``tolerance`` under the L2 norm), compared in the residual's dtype.
Every raising test elsewhere sits orders of magnitude past that boundary
(a NaN residual, ``tolerance=1e-12`` at a cap of two), so loosening the
predicate to ``est <= 4 * threshold`` -- or ``1000 *`` -- or testing the raw
residual instead of the estimate went unnoticed.  These pin the boundary
from both sides on one group, run to its cap so the state the strict
check sees is the same at every threshold:

* the estimate at twice the threshold raises;
* the raw residual within the threshold but the estimate above it raises;
* the estimate exactly at the threshold does not (``<=``, not ``<``).

The same boundary is held for the adaptive steppers' strict check, which
collects the verdicts of the solves an accepted attempt keeps and raises
on the host (``run_adaptive``) or in-graph (``run_adaptive_scan``) -- a
separate predicate from the one ``step()`` raises through.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

KEY = "a+b"
#: Gauss-Seidel rate ``0.8 * 0.8 = 0.64``: the estimate is ~2.8x the residual,
#: so a threshold between them separates the two tests.
GAIN = 0.8
CAP = 6


class _Relay(SimulationNode):
    """``x <- gain * u + bias``: affine, timestep-independent."""

    def __init__(self, name, *, gain, bias):
        super().__init__(name, 1.0, gain=jnp.float32(gain),
                         bias=jnp.asarray(bias, jnp.float32))

    def initial_state(self):
        return {"x": jnp.zeros(2, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), dtype=jnp.float32,
                                       default=jnp.zeros(2, jnp.float32))}

    def update(self, state, boundary_inputs, dt):
        return {"x": self.params["gain"] * boundary_inputs["u"] + self.params["bias"]}


def _graph(tolerance, strict, cap=CAP):
    gm = GraphManager()
    gm.add_node(_Relay("a", gain=GAIN, bias=[1.0, 2.0]))
    gm.add_node(_Relay("b", gain=GAIN, bias=[0.0, 1.0]))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=cap, tolerance=tolerance,
                          strict_convergence=strict)
    gm.compile()
    return gm


@functools.lru_cache(maxsize=None)
def _at_the_cap():
    """The report of the group run to its cap: ``(residual, error_estimate)``.

    At ``tolerance=1e-12`` nothing short of the cap meets the criterion, and
    neither does any threshold below the cap's estimate, so every graph
    below stops on the same pass and hands the strict check the same state.
    """
    gm = _graph(1e-12, strict=False)
    gm.step()
    d = gm.coupling_diagnostics()[KEY]
    assert d["iterations"] == CAP and not d["converged"] and d["ratio_usable"], dict(d)
    assert d["error_estimate"] > 2.0 * d["residual"], (
        "fixture premise: the estimate is well above the raw residual", dict(d))
    return float(d["residual"]), float(d["error_estimate"])


def _report(tolerance):
    gm = _graph(tolerance, strict=False)
    gm.step()
    return gm.coupling_diagnostics()[KEY]


def test_strict_convergence_raises_at_twice_the_threshold():
    """An estimate of 2x the threshold is unconverged, and strict refuses it.

    Caught the mutants ``est <= 4 * threshold`` and ``est <= 1000 * threshold``.
    """
    _res, est = _at_the_cap()
    tol = est / 2.0  # exact: a power of two
    d = _report(tol)
    assert d["iterations"] == CAP and d["error_estimate"] == est, dict(d)
    assert d["error_estimate"] / tol == 2.0 and not d["converged"], dict(d)
    with pytest.raises(Exception, match="without converging"):
        _graph(tol, strict=True).step()


def test_strict_convergence_tests_the_error_estimate_not_the_raw_residual():
    """A residual inside the threshold with the estimate outside it still raises.

    The pre-0.4.0 criterion compared the raw residual, which says only how
    far the last pass moved; MADD-ANO-005 replaced it with the estimated
    distance to the fixed point.  Caught the mutant ``final_est = final_res``.
    """
    res, est = _at_the_cap()
    tol = 1.5 * res
    d = _report(tol)
    assert d["iterations"] == CAP, dict(d)
    assert d["residual"] <= tol < d["error_estimate"] and not d["converged"], dict(d)
    with pytest.raises(Exception, match="without converging"):
        _graph(tol, strict=True).step()


def test_strict_convergence_accepts_an_estimate_exactly_at_the_threshold():
    """``est <= threshold`` is converged, so strict lets the step through.

    The other side of the boundary: a predicate made strict (``est <
    threshold``) or shifted down would raise here.  The state is the
    cap's, the report says ``converged=True`` and agrees with the strict
    check, which is the "one verdict" the report promises.
    """
    _res, est = _at_the_cap()
    d = _report(est)
    assert d["iterations"] == CAP and d["error_estimate"] == est, dict(d)
    assert d["converged"], dict(d)
    gm = _graph(est, strict=True)
    gm.step()
    assert gm.coupling_diagnostics()[KEY]["converged"]


# ---------------------------------------------------------------------------
# The adaptive steppers' strict check: the same boundary through the sink
# ---------------------------------------------------------------------------

#: One accepted attempt of the whole interval: ``dt_min = dt_max = t_end``.
_ADAPTIVE = dict(dt_initial=0.1, dt_min=0.1, dt_max=0.1)


def _adaptive(entry, gm):
    if entry == "run_adaptive":
        return gm.run_adaptive(0.1, **_ADAPTIVE)
    return gm.run_adaptive_scan(0.1, max_steps=1, **_ADAPTIVE)


@functools.lru_cache(maxsize=None)
def _first_half_at_the_cap():
    """The first kept half step's estimate, read off the folded report.

    The relay ignores ``dt``, so the first half step is the same solve as
    ``step()``'s from the same start; it runs to the cap, and the report of
    the accepted attempt carries its (failing) verdict.
    """
    gm = _graph(1e-12, strict=False)
    _adaptive("run_adaptive", gm)
    d = gm.coupling_diagnostics()[KEY]
    assert d["iterations"] == CAP and not d["converged"], dict(d)
    return float(d["residual"]), float(d["error_estimate"])


@pytest.mark.parametrize("entry", ["run_adaptive", "run_adaptive_scan"])
def test_the_adaptive_strict_check_raises_at_twice_the_threshold(entry):
    """The kept first half step at 2x the threshold raises under both steppers."""
    _res, est = _first_half_at_the_cap()
    tol = est / 2.0
    with pytest.raises(Exception, match="without converging"):
        _adaptive(entry, _graph(tol, strict=True))


@pytest.mark.parametrize("entry", ["run_adaptive", "run_adaptive_scan"])
def test_the_adaptive_strict_check_tests_the_error_estimate(entry):
    """Residual inside, estimate outside the threshold: the adaptive check raises too."""
    res, est = _first_half_at_the_cap()
    tol = 1.5 * res
    assert res <= tol < est
    with pytest.raises(Exception, match="without converging"):
        _adaptive(entry, _graph(tol, strict=True))


def test_the_adaptive_report_agrees_with_its_strict_check_at_the_threshold():
    """At ``tolerance`` = the first half's estimate both halves converge: no raise.

    The second half starts from the first's result, nearer the fixed
    point, so it meets any threshold the first met.
    """
    _res, est = _first_half_at_the_cap()
    gm = _graph(est, strict=True)
    _adaptive("run_adaptive", gm)
    d = gm.coupling_diagnostics()[KEY]
    assert d["converged"], dict(d)
    assert np.isfinite(d["residual"])
