"""A coupling group's ``converged`` has one definition, in the residual's dtype.

The loops, ``strict_convergence`` and the sysid mask decide
``estimated_error(residual, amplification) <= threshold`` in-graph, in
float32 for a float32 group, with the threshold rounded to float32.
``coupling_diagnostics()`` and the profiler used to re-derive it on the
host in float64, so within half a float32 ulp of the threshold the report
said "not converged" about a step the loop had stopped on, strict had let
through and sysid had kept.

The tolerance only decides where the loop stops, never the iterates, so
the test runs once, reads the residual and amplification the loop stopped
on, and picks a tolerance whose float32 rounding equals the float32
estimate while it sits below the float64 product of the same two numbers.
"""

import math
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import acceleration
from maddening.core.coupling.acceleration import (
    convergence_criterion,
    estimated_error,
    reported_error_estimate,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.simulation.profiler import _meta_converged
from maddening.sysid import observations_from_history, windowed_loss

KEY = "a+b"


class _Affine(SimulationNode):
    def __init__(self, name, gain, bias):
        super().__init__(name=name, timestep=0.01, gain=gain, bias=bias)

    def initial_state(self):
        return {"x": jnp.float32(0.0)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}

    def update(self, state, boundary_inputs, dt):
        u = boundary_inputs.get("u", jnp.float32(0.0))
        return {"x": self.params["gain"] * u + self.params["bias"]}


def _pair(tolerance, solver="ift", **kw):
    gm = GraphManager()
    gm.add_node(_Affine("a", 0.9, 1.0))
    gm.add_node(_Affine("b", 0.9, 0.0))
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    # ``"ift"`` records its verdict always; ``"fori"`` only with diagnostics.
    gm.add_coupling_group(["a", "b"], max_iterations=200, tolerance=tolerance,
                          solver=solver, diagnostics=solver == "fori", **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # solver="fori" is deprecated
        gm.compile()
    return gm


def _slots(gm):
    meta = gm._state["_meta"]
    return (np.asarray(meta[f"coupling_{KEY}_residual"]),
            np.asarray(meta[f"coupling_{KEY}_amplification"]),
            int(meta[f"coupling_{KEY}_iterations"]))


@pytest.fixture(scope="module")
def edge_tolerance():
    """A tolerance at which the two precisions disagree on the stopping pass."""
    reference = _pair(1e-5)
    reference.step()
    residual, amplification, passes = _slots(reference)
    assert residual.dtype == np.float32
    est32 = np.float32(residual * amplification)
    est64 = float(residual) * float(amplification)
    tolerance = math.nextafter(est64, -math.inf)
    # The construction, checked rather than assumed: float32 cannot tell
    # the tolerance from the estimate, float64 puts it below.
    assert np.float32(tolerance) == est32 and tolerance < est64, (
        tolerance, est32, est64)
    return tolerance, passes


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_every_reader_gives_the_loops_verdict(edge_tolerance, solver):
    tolerance, passes = edge_tolerance
    gm = _pair(tolerance, solver=solver)
    gm.step()
    residual, amplification, iterations = _slots(gm)
    # The loop stopped on its criterion, on the same pass as at 1e-5:
    # the verdict it acted on is "converged".
    assert iterations == passes < 200
    group = gm._coupling_groups[0]
    threshold, scale = convergence_criterion(group)
    in_graph = bool(estimated_error(jnp.asarray(residual), jnp.asarray(amplification),
                                    scale) <= threshold)
    assert in_graph
    report = gm.coupling_diagnostics()[KEY]
    assert report["converged"] is True, report
    assert _meta_converged(gm._state["_meta"], f"coupling_{KEY}_residual",
                           f"coupling_{KEY}_amplification", threshold, scale) is True
    assert acceleration.reported_converged(residual, amplification, scale,
                                           threshold) is True
    # ``error_estimate`` is the estimate the loop compared, in float32,
    # so ``converged`` is its float32 comparison with the threshold.
    assert report["error_estimate"] == float(np.float32(residual * amplification))
    assert np.float32(report["error_estimate"]) <= np.float32(threshold)


def test_strict_convergence_agrees_with_the_report(edge_tolerance):
    tolerance, _ = edge_tolerance
    gm = _pair(tolerance, strict_convergence=True)
    gm.step()                           # raises if strict disagrees
    assert gm.coupling_diagnostics()[KEY]["converged"] is True


def test_the_sysid_mask_agrees_with_the_report(edge_tolerance):
    tolerance, _ = edge_tolerance
    gm = _pair(tolerance)
    init = {n: dict(gm.get_node_state(n)) for n in ("a", "b")}
    truth = {n: {f: jnp.asarray(v)[None] + 1.0 for f, v in init[n].items()}
             for n in ("a", "b")}
    obs = observations_from_history(init, truth)
    kw = dict(obs_fn=lambda s: s["a"]["x"], window=1)
    unmasked = float(windowed_loss(gm, gm.params, obs, mask_unconverged=False, **kw))
    masked = float(windowed_loss(gm, gm.params, obs, mask_unconverged=True, **kw))
    assert unmasked > 0.0 and masked == unmasked    # sysid: converged
    gm.step()                                       # the same step, reported
    assert gm.coupling_diagnostics()[KEY]["converged"] is True


def _draws(dtype, n=200):
    rng = np.random.default_rng(0)
    for _ in range(n):
        yield (dtype(10.0 ** rng.uniform(-9, 2)),
               dtype(rng.choice([0.0, 1.0, 10.0 ** rng.uniform(0, 4)])),
               float(rng.choice([1.0, 0.7, 1.5])))


def test_the_host_verdict_is_the_in_graph_verdict_on_float32_slots():
    """``estimated_error(...) <= threshold`` as JAX computes it, rounding included.

    Each threshold is the estimate itself and its float64 neighbours,
    which float32 cannot tell from it: the comparison must round the
    threshold as the in-graph one does.
    """
    for residual, amplification, scale in _draws(np.float32):
        expected = estimated_error(jnp.asarray(residual), jnp.asarray(amplification),
                                   scale)
        assert np.asarray(expected).dtype == np.float32
        assert reported_error_estimate(residual, amplification, scale) == float(expected)
        for threshold in (float(expected), math.nextafter(float(expected), 0.0),
                          math.nextafter(float(expected), math.inf)):
            verdict = bool(expected <= threshold)
            # A float64 *scalar* threshold too: NumPy 2 promotes a bare
            # Python float to the array's dtype, but not a np.float64.
            for spelled in (threshold, np.float64(threshold)):
                assert acceleration.reported_converged(
                    residual, amplification, scale, spelled) is verdict, (
                    residual, amplification, scale, spelled)


def test_float64_slots_keep_the_float64_arithmetic():
    """A float64 group (x64) is reported exactly as before: nothing to round."""
    for residual, amplification, scale in _draws(np.float64):
        a, r = float(amplification), float(residual)
        expected = r * max(scale * a, 1.0) if a >= 1.0 else r
        assert reported_error_estimate(residual, amplification, scale) == expected
        for threshold in (expected, math.nextafter(expected, 0.0)):
            assert acceleration.reported_converged(
                residual, amplification, scale, threshold) is \
                (expected <= threshold)
