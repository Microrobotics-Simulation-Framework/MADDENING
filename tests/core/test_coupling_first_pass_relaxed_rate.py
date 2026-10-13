"""Under ``acceleration="fixed"`` the first loop pass reads the relaxed rate.

Every coupling loop starts with one unrelaxed pass, and the first loop pass
takes its rate from that pass's residual: the ratio spans an *unrelaxed*
step, so it is the map's own rate ``rho`` (as ``sqrt(rho)``), while the
step scale ``omega`` assumes the relaxed rate ``1 - omega (1 - rho)``.  So
``omega r / (1 - sqrt(rho))`` understated the distance ``r / (1 - rho)``
whenever ``omega < 1 / (1 + sqrt(rho))``: at ``rho = 0.5``, ``omega = 0.3``
a group stopped there with ``converged=True`` 1.41 thresholds from its
fixed point.  The first pass now translates the rate it read
(``first_pass_relaxed_amplification``); over-relaxation and every other
acceleration are untouched.
"""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import (
    first_pass_relaxed_amplification,
    relaxes_first_pass,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.core.test_coupling_solver_equivalence import residual_noise_floor

#: ``a: x = RHO u + 1``, ``b: x = u``: one mode, rate ``RHO``, fixed point ``1 / (1 - RHO)``.
RHO = 0.5
X_STAR = 1.0 / (1.0 - RHO)
TOL = 1e-4


class Relay(SimulationNode):
    def __init__(self, name, gain, bias, x0):
        super().__init__(name, 1.0, g=gain, b=bias)
        self._x0 = x0

    def initial_state(self):
        return {"x": jnp.asarray(self._x0, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.asarray(0.0, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": (p["g"] * boundary_inputs["u"] + p["b"]).astype(jnp.float32)}


def _solve(solver, relaxation, delta):
    """One step from ``x* (1 + delta)`` on both nodes: ``(diagnostics, distance)``.

    The distance is the group's relative L2 norm to the exact fixed point,
    ``sqrt(sum ((x - x*) / max|x|)**2)``, the units ``tolerance`` is in.
    """
    start = np.float32(X_STAR * (1.0 + delta))
    gm = GraphManager()
    gm.add_node(Relay("a", RHO, 1.0, start))
    gm.add_node(Relay("b", 1.0, 0.0, start))
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "CouplingGroup solver='fori' is deprecated",
                                DeprecationWarning)
        gm.add_coupling_group(["a", "b"], acceleration="fixed", relaxation=relaxation,
                              tolerance=TOL, max_iterations=60, solver=solver,
                              diagnostics=True)
    gm.compile()
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    x = np.array([float(gm.get_node_state(n)["x"]) for n in ("a", "b")], np.float64)
    dist = float(np.sqrt(np.sum(((x - X_STAR) / np.maximum(np.abs(x), X_STAR)) ** 2)))
    return d, dist


@pytest.mark.parametrize("solver", ["ift", "fori"])
@pytest.mark.parametrize("relaxation", [0.1, 0.3, 0.5, 0.6])
@pytest.mark.parametrize("delta", [1.2e-4, 2.0e-4, 4.0e-4, 1e-3])
def test_converged_is_within_the_threshold_under_under_relaxation(solver, relaxation, delta):
    """Wherever it stops, the one-mode estimate is not below the distance.

    Up to the estimate's own float32 resolution, derived as in
    ``tests/property/test_differential_fixed_point.py``: a residual
    carries ``floor`` of rounding, which moves the distance by
    ``floor * amp`` and the ratio by ``2 floor / r``, i.e. the estimate by
    ``2 omega floor amp**2``.
    """
    d, dist = _solve(solver, relaxation, delta)
    assert d["converged"] and d["ratio_usable"]
    floor = residual_noise_floor("l2", 1e-6, 2)
    amp = d["amplification"]
    slack = floor * amp + 2.0 * relaxation * floor * amp ** 2
    assert dist <= TOL + slack, (d["iterations"], dist / TOL, d["error_estimate"] / TOL)
    assert d["error_estimate"] >= dist - slack


@pytest.mark.parametrize("relaxation", [0.3, 0.6])
def test_a_first_pass_exit_still_happens_when_the_estimate_allows_it(relaxation):
    """Close enough, the group stops on its first loop pass -- and is right to.

    The translated estimate there is ``r / (1 - sqrt(rho))``, which bounds
    ``r / (1 - rho)``.
    """
    for solver in ("ift", "fori"):
        d, dist = _solve(solver, relaxation, 2e-5)
        assert d["iterations"] == 1 and d["converged"] and d["ratio_usable"]
        assert d["error_estimate"] >= dist
        # The amplification reported is the relaxed iteration's,
        # ``1 / (omega (1 - sqrt(rho)))``, to the ratio's float resolution.
        assert d["amplification"] == pytest.approx(
            1.0 / (relaxation * (1.0 - RHO ** 0.5)), rel=2e-2)


@pytest.mark.parametrize("relaxation", [0.3, 0.7, 1.0, 1.4])
def test_both_solvers_stop_on_the_same_pass_and_report_the_same_verdict(relaxation):
    for delta in (2e-5, 2e-4, 1e-3):
        di, _ = _solve("ift", relaxation, delta)
        df, _ = _solve("fori", relaxation, delta)
        assert di["iterations"] == df["iterations"], delta
        assert di["converged"] == df["converged"]
        assert di["amplification"] == pytest.approx(df["amplification"], rel=1e-5)


def test_the_estimate_reported_is_the_one_the_loop_compared():
    """``error_estimate = omega * residual * amplification`` on a first-pass exit."""
    d, _ = _solve("ift", 0.3, 2e-5)
    assert d["iterations"] == 1
    assert d["error_estimate"] == pytest.approx(
        0.3 * d["residual"] * d["amplification"], rel=1e-6)


@pytest.mark.parametrize("acceleration, relaxation", [
    ("none", 1.0), ("aitken", 1.0), ("iqn-ils", 1.0), ("iqn-imvj", 1.0),
    ("fixed", 1.0), ("fixed", 1.3), ("fixed", 1.95),
])
def test_nothing_else_is_rescaled_and_nothing_else_reads_the_pass(acceleration, relaxation):
    """Statically the identity: the very object back, the predicate never read."""
    amp = jnp.float32(3.0)

    class Unreadable:
        def __bool__(self):
            raise AssertionError("the first-pass predicate was read")

    assert not relaxes_first_pass(acceleration, relaxation)
    assert first_pass_relaxed_amplification(amp, acceleration, relaxation, Unreadable()) is amp


def test_under_relaxation_divides_the_first_pass_only_and_keeps_a_rejection():
    assert relaxes_first_pass("fixed", 0.4)
    first = first_pass_relaxed_amplification(jnp.float32(3.0), "fixed", 0.4, jnp.bool_(True))
    later = first_pass_relaxed_amplification(jnp.float32(3.0), "fixed", 0.4, jnp.bool_(False))
    rejected = first_pass_relaxed_amplification(jnp.float32(0.0), "fixed", 0.4, jnp.bool_(True))
    assert float(first) == pytest.approx(7.5)
    assert float(later) == 3.0
    assert float(rejected) == 0.0
