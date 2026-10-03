"""Truth recovery: a fitter that says ``converged=True`` has found the truth.

The oracle, over generated problems: draw a true parameter anywhere inside
its bounds (near them too) and a start anywhere in the box, generate
noiseless data, run the fitter, and require **either recovery of the truth
to a stated tolerance, or ``converged=False``** -- never ``converged=True``
at a point that is not the constrained optimum.

The problems are built so that the truth is the *only* constrained
stationary point in the box: each trainable coordinate has its own block of
residuals, ``g(p) - g(p*)`` with ``g`` strictly monotone and nonlinear, so the
loss is a sum of one-dimensional unimodal losses.  Nonlinear so that a
Gauss-Newton step can overshoot -- past a bound, which is how
``fit_lm`` used to stop on the bound of a clipped coordinate and call it
converged (audit_040_p4_5/fmu-sysid/repro_fit_lm_stuck_at_bound.py).  Every
transform takes part: a bounded identity leaf (``transform=None``, clipped),
``log`` and ``logit``.

Per push: ``fit_lm`` on the closed-form problem.  Slow: the same for ``fit``
(Adam, with a ``tol``) and ``fit_lm`` with more draws, and the spring graph
itself under all three fitters.
"""

from __future__ import annotations

import contextlib
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, note, settings
from hypothesis.errors import InvalidArgument
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import fit, fit_lm, fit_multiple_shooting, observations_from_history

from tests.conftest import EXAMPLES_COSTLY

T = np.linspace(0.1, 2.0, 8)

#: Where in its range a drawn value sits: anywhere, and near either edge.
_FRACTION = st.one_of(
    st.floats(0.02, 0.98),
    st.sampled_from([1e-4, 1e-3, 1e-2, 0.99, 0.999, 0.9999]),
)

#: The three ways a trainable leaf can be bounded, and its range.
_SPECS = {
    "clip": (ParamSpec(bounds=(0.0, 5.0)), (0.0, 5.0)),
    "log": (ParamSpec(bounds=(0.0, None), transform="log"), (0.05, 20.0)),
    "logit": (ParamSpec(bounds=(0.5, 2.0), transform="logit"), (0.5, 2.0)),
}
KEYS = ("stiffness", "damping", "rest_length")


_ONE = (1.0, 1.0, 1.0)


def _spec(kind, unit=1.0):
    """The coordinate's spec with its parameter measured in a unit ``unit``
    times smaller: bounds scale with it; a ``log`` leaf's floor 0 does not."""
    spec, _ = _SPECS[kind]
    if spec.transform == "log":
        return spec
    lo, hi = spec.bounds
    return ParamSpec(bounds=(lo * unit, hi * unit), transform=spec.transform)


def _graph(kinds, units=_ONE):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.0,
                                 rest_length=1.0, initial_position=0.2))
    for key, kind, unit in zip(KEYS, kinds, units):
        gm.set_param_spec("s", key, _spec(kind, unit))
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
    gm.compile()
    return gm


def _value(kind, fraction):
    lo, hi = _SPECS[kind][1]
    return lo + fraction * (hi - lo)


def _blocks(p, units=_ONE):
    """One strictly monotone, nonlinear block of residuals per coordinate,
    with a finite, nowhere-vanishing derivative over every range drawn (an
    ``exp(-c t)`` block underflowed to an exactly flat plateau that a fit
    then rightly called stationary, which is not what this oracle is
    about).  The concave blocks make a Gauss-Newton step from above
    overshoot below the truth -- past a lower bound of 0 for a clipped
    coordinate -- and the convex one overshoots from below.  ``units``
    measure each parameter in another unit: the model reads ``p / unit``.

    The division is kept apart from what follows by an optimisation barrier:
    under ``jax.jit`` XLA reassociates ``0.2 * (x / 1e-20) ** 2`` into
    ``x * x * (0.2 * 1e40)``, whose constant overflows float32 while ``x * x``
    flushes, and the block is NaN -- a property of that rewrite and of no
    fitter, which a model written for parameters at extreme scales has to
    keep out of the way."""
    q = p["nodes"]["s"]
    k, c, rest = jax.lax.optimization_barrier(
        tuple(q[key] / unit for key, unit in zip(KEYS, units)))
    t = jnp.asarray(T, jnp.float32)
    return jnp.concatenate([
        jnp.log1p(c * t),
        jnp.sqrt(k + 0.1) * t,
        rest * t + 0.2 * rest ** 2 * t,
    ])


def _start(gm, kinds, fractions, units=_ONE, dtype=jnp.float32):
    p = jax.tree.map(lambda x: x, gm.params)
    for key, kind, fraction, unit in zip(KEYS, kinds, fractions, units):
        p["nodes"]["s"][key] = jnp.asarray(_value(kind, fraction) * unit, dtype)
    return p


def _mask(gm):
    mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
    for key in KEYS:
        mask["nodes"]["s"][key] = True
    return mask


def _recovered(res, truth, kinds, units=_ONE) -> tuple[bool, list]:
    errors = []
    ok = True
    for key, kind, unit in zip(KEYS, kinds, units):
        lo, hi = _SPECS[kind][1]
        got = float(res.params["nodes"]["s"][key]) / unit
        want = float(truth["nodes"]["s"][key]) / unit
        # 1e-3 of the coordinate's range: a fit at its float32 floor is far
        # inside this, and a fit stuck on a bound is far outside it.
        tolerance = 1e-3 * (hi - lo)
        errors.append((key, got, want))
        ok &= abs(got - want) <= tolerance
    return ok, errors


_PROBLEM = dict(
    kinds=st.tuples(*[st.sampled_from(sorted(_SPECS))] * 3),
    truth_at=st.tuples(_FRACTION, _FRACTION, _FRACTION),
    start_at=st.tuples(_FRACTION, _FRACTION, _FRACTION),
)


def _fit(kinds, truth_at, start_at, units=_ONE, dtype=jnp.float32, scale=1.0):
    """``fit_lm`` on the closed-form problem; ``scale`` multiplies the
    residual (its units), ``units`` the parameters'."""
    gm = _graph(kinds, units)
    truth = _start(gm, kinds, truth_at, units, dtype)
    data = _blocks(truth, units)
    s = jnp.asarray(scale, data.dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a declined hold says so
        res = fit_lm(gm, lambda p: s * (_blocks(p, units) - data),
                     params=_start(gm, kinds, start_at, units, dtype), mask=_mask(gm),
                     n_iter=60)
    ok, errors = _recovered(res, truth, kinds, units)
    _note(f"units={units} scale={scale} kinds={kinds} converged={res.converged} "
          f"n_iter={res.n_iter} best_loss={res.best_loss} (key, got, truth)={errors}")
    return res, ok, errors


def _note(message):
    """``hypothesis.note`` inside a property, nothing in a plain test (whose
    assertion messages carry the same numbers)."""
    try:
        note(message)
    except InvalidArgument:
        pass


def _check_fit_lm(kinds, truth_at, start_at):
    res, ok, errors = _fit(kinds, truth_at, start_at)
    assert ok or not res.converged, errors


@given(**_PROBLEM)
# Found by the broad draw below: a logit stiffness started near the bottom of
# (0.5, 2) is thrown to the top edge, where the sigmoid is flat, and its
# value moves by under ``step_tol`` for any step in ``u`` -- it stopped there,
# "converged", at 1.999997 with the truth at 1.25.  Pinned per push.
@example(kinds=("logit", "clip", "clip"), truth_at=(0.5, 0.5, 0.75),
         start_at=(0.0625, 0.5, 0.5))
# Found here: a logit damping driven past the sigmoid's edge sat on the
# clamp (derivative 0) at 1.999998 of (0.5, 2), "converged".
@example(kinds=("clip", "logit", "clip"), truth_at=(0.5, 0.5, 0.5),
         start_at=(0.5, 0.0001, 0.5))
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
def test_fit_lm_converged_means_the_truth(kinds, truth_at, start_at):
    _check_fit_lm(kinds, truth_at, start_at)


@contextlib.contextmanager
def _x64():
    """Run the body in double precision, restoring the global setting."""
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


#: Under x64 every trainable leaf at float64 (``"float64"``), or the three
#: trainable leaves at float32 beside the graph's float64 ones (``"mixed"``:
#: a value with its own float dtype keeps it, and ``ravel_pytree`` promotes
#: the optimiser's vector to float64 around it).
_X64_DTYPES = {"float64": jnp.float64, "mixed": jnp.float32}


def _check_fit_lm_x64(kinds, truth_at, start_at, leaves):
    """The oracle under x64: x64 is a supported mode (``fim`` recommends it),
    and the identifiability guard, the step tolerance and the floor rule
    all read their precision from the leaves -- the guard used to take it
    from the promoted vector, and the floor rule's ladder ended a rung
    short at float64's resolution."""
    with _x64():
        res, ok, errors = _fit(kinds, truth_at, start_at, dtype=_X64_DTYPES[leaves])
    assert ok or not res.converged, (leaves, errors)
    return res


#: Per push: each transform, near and away from its bounds, both ways round.
_X64_CASES = [
    (("clip", "log", "logit"), (0.5, 0.3, 0.6), (0.9, 0.7, 0.2)),
    (("logit", "clip", "clip"), (0.5, 0.5, 0.75), (0.0625, 0.5, 0.5)),
    (("clip", "logit", "clip"), (0.5, 0.5, 0.5), (0.5, 0.0001, 0.5)),
    (("log", "log", "logit"), (1e-3, 0.999, 0.9999), (0.5, 0.02, 0.1)),
]


@pytest.mark.parametrize("leaves", sorted(_X64_DTYPES))
@pytest.mark.parametrize("kinds, truth_at, start_at", _X64_CASES)
def test_fit_lm_converged_means_the_truth_under_x64(kinds, truth_at, start_at, leaves):
    res = _check_fit_lm_x64(kinds, truth_at, start_at, leaves)
    if (kinds, truth_at, start_at) == _X64_CASES[0]:
        # Non-vacuity: away from every edge the fit does say converged, so
        # the oracle above is tested, not satisfied by a run that never is.
        assert res.converged, (leaves, res.n_iter, res.best_loss)


# Per push: tests/property/test_sysid_truth_recovery.py::test_fit_lm_converged_means_the_truth_under_x64
@pytest.mark.slow  # a fit compiled per example under x64: over 5 s on CI
@given(**_PROBLEM, leaves=st.sampled_from(sorted(_X64_DTYPES)))
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
def test_fit_lm_converged_means_the_truth_under_x64_broadly(kinds, truth_at, start_at, leaves):
    _check_fit_lm_x64(kinds, truth_at, start_at, leaves)


def _check_units(kinds, truth_at, start_at, which, unit):
    """Parameter-unit invariance: measure one parameter in a unit up to 1e6
    times larger or smaller (its bounds, start and truth scale with it) and
    the fit, as called by default, must give the same answer -- the same
    returned point, ``excited_rank`` and ``hold_declined`` -- in about as
    many iterations, and each run must still recover the truth or say
    ``converged=False``.  A Marquardt floor of ``eps`` times the *mean* of
    ``diag(JᵀJ)`` failed exactly this: the coordinate whose column the
    rescaling shrank had its step crushed and stopped short, "converged".
    So did the identifiability guard asking its questions in the
    optimiser's coordinates: an identity coordinate in units 1e-4 was
    called undetermined and the hold declined (rank 2 of 3), and in units
    1e-5 or 1e6 it was held at its start, "converged", far from the truth.
    (Residual-scale invariance:
    ``tests/core/test_sysid_bounded_coordinates_and_residual_scale.py``.)"""
    units = tuple(unit if i == which else 1.0 for i in range(3))
    base, base_ok, base_errors = _fit(kinds, truth_at, start_at)
    scaled, scaled_ok, scaled_errors = _fit(kinds, truth_at, start_at, units)
    assert base_ok or not base.converged, base_errors
    assert scaled_ok or not scaled.converged, scaled_errors
    if base.converged and scaled.converged:
        slack = max(3, (base.n_iter + scaled.n_iter) // 4)
        assert abs(base.n_iter - scaled.n_iter) <= slack, (base.n_iter, scaled.n_iter)
        assert (scaled.excited_rank, scaled.hold_declined) == (
            base.excited_rank, base.hold_declined), (base, scaled)
        for (key, got, _), (_, got_scaled, _), kind in zip(base_errors, scaled_errors, kinds):
            lo, hi = _SPECS[kind][1]
            assert abs(got_scaled - got) <= 1e-3 * (hi - lo), (key, got, got_scaled)


#: Per push, each transform's coordinate rescaled both ways (two fits a case,
#: each compiled, so a fixed table rather than a draw; the draw is slow below).
_UNIT_CASES = [
    (("clip", "log", "logit"), (0.5, 0.3, 0.6), (0.9, 0.7, 0.2), 0, 1e4),
    (("clip", "log", "logit"), (0.5, 0.3, 0.6), (0.9, 0.7, 0.2), 0, 1e-4),
    (("log", "clip", "logit"), (0.2, 0.01, 0.5), (0.8, 0.9, 0.9), 1, 1e4),
    (("log", "clip", "logit"), (0.2, 0.01, 0.5), (0.8, 0.9, 0.9), 1, 1e-4),
    # The units the 0.4.0-dev guard held a determined identity coordinate in.
    (("log", "clip", "logit"), (0.6, 0.6, 0.5), (0.4, 0.4, 0.9), 1, 1e-5),
    (("log", "clip", "logit"), (0.6, 0.6, 0.5), (0.4, 0.4, 0.9), 1, 1e6),
    (("logit", "log", "clip"), (0.5, 0.5, 0.75), (0.1, 0.9, 0.2), 2, 1e4),
    (("logit", "log", "clip"), (0.5, 0.5, 0.75), (0.1, 0.9, 0.2), 2, 1e-4),
]


@pytest.mark.parametrize("kinds, truth_at, start_at, which, unit", _UNIT_CASES)
def test_fit_lm_answers_the_same_in_any_units(kinds, truth_at, start_at, which, unit):
    _check_units(kinds, truth_at, start_at, which, unit)


# ---------------------------------------------------------------------------
# The whole float32 range: a residual and a parameter of any normal size
# ---------------------------------------------------------------------------

#: Residual scales across float32's range.  Below ``1e-19`` (``sqrt(tiny)``)
#: ``r * r`` flushes to zero and above ``1e19`` it overflows; below about
#: ``1e-19`` the products of ``JᵀJ`` and ``Jᵀr`` flush too.  ``fit_lm`` used to
#: report ``converged=True`` at a point 2-7% off, or at its unmoved start,
#: from ``1e-18`` down (audit_040_p4_10/fmu-sysid/
#: repro_fitters_tiny_residual_units.py).
_RESIDUAL_SCALES = [1e-30, 1e-25, 1e-20, 1e-19, 1e-18, 1e-10, 1e10, 1e18, 1e19, 1e20,
                    1e25, 1e30]

#: Parameter natural scales across float32's range, for the identity
#: (``clip``) and the ``logit`` coordinate (a ``log`` coordinate is
#: unit-free by construction).  An identity parameter whose natural scale
#: is ``1e-23`` -- a 10 nm particle's volume in cubic metres, the example
#: the parameter guide gives -- or ``1e23`` came back unmoved, "converged"
#: (repro_fit_lm_extreme_parameter_units.py).
_PARAMETER_SCALES = [1e-30, 1e-23, 1e-20, 1e20, 1e23, 1e30]

_RANGE_CASE = (("clip", "log", "logit"), (0.5, 0.3, 0.6), (0.9, 0.7, 0.2))


@pytest.mark.parametrize("scale", _RESIDUAL_SCALES)
def test_fit_lm_recovers_the_truth_at_every_residual_scale(scale):
    """A residual of any normal float32 size: ``fit_lm`` converges, at the
    truth, at the point and in about the iterations of the unscaled fit."""
    kinds, truth_at, start_at = _RANGE_CASE
    base, base_ok, base_errors = _fit(kinds, truth_at, start_at)
    res, ok, errors = _fit(kinds, truth_at, start_at, scale=scale)
    assert base.converged and base_ok, base_errors
    assert res.converged and ok, (scale, res.n_iter, res.best_loss, errors)
    assert abs(res.n_iter - base.n_iter) <= 3, (base.n_iter, res.n_iter)
    for (key, got, _), (_, got_scaled, _), kind in zip(base_errors, errors, kinds):
        lo, hi = _SPECS[kind][1]
        assert abs(got_scaled - got) <= 1e-4 * (hi - lo), (key, got, got_scaled)
    # The loss is the scaled one, read without flushing or overflowing.
    assert np.isfinite(res.losses).all() and res.losses[0] > 0.0
    assert res.losses[0] == pytest.approx(base.losses[0] * scale ** 2, rel=1e-5)


@pytest.mark.parametrize("which", [0, 2], ids=["clip", "logit"])
@pytest.mark.parametrize("unit", _PARAMETER_SCALES)
def test_fit_lm_recovers_the_truth_at_every_parameter_scale(which, unit):
    """One parameter at a natural scale anywhere in float32's range: the
    same answer as at unit scale, converged (``_check_units``), and never
    ``converged=True`` away from the truth."""
    kinds, truth_at, start_at = _RANGE_CASE
    _check_units(kinds, truth_at, start_at, which, unit)
    units = tuple(unit if i == which else 1.0 for i in range(3))
    res, ok, errors = _fit(kinds, truth_at, start_at, units)
    assert res.converged and ok, (unit, res.n_iter, res.best_loss, errors)


def _adam_fit(kinds, truth_at, start_at, units=_ONE, scale=1.0, n_iter=400):
    gm = _graph(kinds, units)
    truth = _start(gm, kinds, truth_at, units)
    data = _blocks(truth, units)
    s = jnp.asarray(scale, data.dtype)
    start = _start(gm, kinds, start_at, units)
    loss = jax.jit(lambda p: 0.5 * jnp.sum((s * (_blocks(p, units) - data)) ** 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a flushed loss says so
        res = fit(gm, loss, params=start, mask=_mask(gm), n_iter=n_iter, lr=0.05)
    ok, errors = _recovered(res, truth, kinds, units)
    return res, ok, errors


#: For ``fit`` the loss is the caller's, so it can only be evaluated where
#: ``0.5 * ||s * r||²`` is finite: up to ``s`` near ``1e18`` here.  Above, the
#: loss itself overflows and ``fit`` refuses (FloatingPointError), loudly.
_ADAM_RESIDUAL_SCALES = [1e-30, 1e-20, 1e-19, 1e-18, 1e-10, 1.0, 1e10, 1e18]


@pytest.mark.parametrize("scale", _ADAM_RESIDUAL_SCALES)
def test_fit_recovers_the_truth_at_every_residual_scale(scale):
    """Adam on a loss of any representable size: its gradient is taken clear
    of the flush (``_gradient_lift``) and its ``eps`` is relative, so it
    returns the unscaled fit's answer.  At ``1e-19`` it used to return its
    start, and at ``1e-18`` a point 1% off."""
    kinds, truth_at, start_at = _RANGE_CASE
    base, base_ok, base_errors = _adam_fit(kinds, truth_at, start_at)
    res, ok, errors = _adam_fit(kinds, truth_at, start_at, scale=scale)
    assert base_ok, base_errors
    assert ok, (scale, errors)
    for (key, got, _), (_, got_scaled, _), kind in zip(base_errors, errors, kinds):
        lo, hi = _SPECS[kind][1]
        assert abs(got_scaled - got) <= 1e-4 * (hi - lo), (key, got, got_scaled)


def test_fit_refuses_a_loss_that_overflows():
    """Above the representable range the caller's loss is ``inf`` and the
    refusal is loud, never a result."""
    kinds, truth_at, start_at = _RANGE_CASE
    with pytest.raises(FloatingPointError, match="non-finite loss or gradient"):
        _adam_fit(kinds, truth_at, start_at, scale=1e30, n_iter=5)


@pytest.mark.parametrize("unit", _PARAMETER_SCALES)
def test_fit_recovers_the_truth_at_every_parameter_scale(unit):
    """Adam on transformed coordinates, which are unit-free: a ``log`` and a
    ``logit`` parameter at a natural scale anywhere in float32's range.  (An
    identity coordinate is not: Adam's step is about ``lr`` in it, so its
    path depends on its units by design -- MADD-ANO-135's residual risk.)"""
    kinds, truth_at, start_at = (("log", "log", "logit"), (0.3, 0.6, 0.6),
                                 (0.7, 0.2, 0.2))
    base, base_ok, base_errors = _adam_fit(kinds, truth_at, start_at)
    res, ok, errors = _adam_fit(kinds, truth_at, start_at, units=(unit, 1.0, unit))
    assert base_ok, base_errors
    assert ok, (unit, errors)


# Per push: tests/property/test_sysid_truth_recovery.py::test_fit_lm_answers_the_same_in_any_units
@pytest.mark.slow  # two fits compiled per example: over 5 s on CI
@given(**_PROBLEM, which=st.integers(0, 2),
       unit=st.sampled_from([1e-6, 1e-5, 1e-4, 1e-2, 1e2, 1e4, 1e5, 1e6]))
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
def test_fit_lm_answers_the_same_in_any_units_broadly(kinds, truth_at, start_at, which, unit):
    _check_units(kinds, truth_at, start_at, which, unit)


# Per push: tests/property/test_sysid_truth_recovery.py::test_fit_lm_converged_means_the_truth
@pytest.mark.slow  # a fit compiled per example, four times the per-push draws: over 5 s on CI
@given(**_PROBLEM)
@settings(max_examples=4 * EXAMPLES_COSTLY, deadline=None, derandomize=True)
def test_fit_lm_converged_means_the_truth_broadly(kinds, truth_at, start_at):
    _check_fit_lm(kinds, truth_at, start_at)


# Per push: tests/property/test_sysid_truth_recovery.py::test_fit_lm_converged_means_the_truth
#   (Adam's ``converged`` is the ``tol`` test, which the per-push oracle's
#   fitter shares; the clipped-coordinate projection is pinned per push by
#   tests/core/test_sysid_bounded_coordinates_and_residual_scale.py::test_fit_brings_a_clipped_coordinate_back_into_its_range.)
@pytest.mark.slow  # hundreds of Adam steps per example: over 5 s on CI
@given(**_PROBLEM)
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
def test_fit_converged_means_the_truth(kinds, truth_at, start_at):
    gm = _graph(kinds)
    truth = _start(gm, kinds, truth_at)
    data = _blocks(truth)
    start = _start(gm, kinds, start_at)
    loss = jax.jit(lambda p: 0.5 * jnp.sum((_blocks(p) - data) ** 2))
    tol = 1e-10 * max(float(loss(start)), 1e-30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = fit(gm, loss, params=start, mask=_mask(gm), n_iter=400, lr=0.05, tol=tol)
    ok, errors = _recovered(res, truth, kinds)
    note(f"kinds={kinds} converged={res.converged} best_loss={res.best_loss} {errors}")
    assert ok or not res.converged, errors


# ---------------------------------------------------------------------------
# The spring graph itself: damping (clipped) from anywhere to anywhere
# ---------------------------------------------------------------------------

N_STEPS = 60


def _spring(damping):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=damping, mass=1.0,
                                 rest_length=1.0, initial_position=0.2))
    gm.compile()
    return gm


def _only_damping(gm):
    mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
    mask["nodes"]["s"]["damping"] = True
    return mask


# Per push: tests/core/test_sysid_bounded_coordinates_and_residual_scale.py::test_fit_lm_brings_a_clipped_coordinate_back_into_its_range
# Per push: tests/core/test_sysid_bounded_coordinates_and_residual_scale.py::test_fit_multiple_shooting_brings_a_clipped_coordinate_back_into_its_range
@pytest.mark.slow  # a graph rollout per residual and a fit per example: over 5 s on CI
@given(truth=st.floats(0.01, 6.0), start=st.floats(0.0, 15.0),
       fitter=st.sampled_from(["fit_lm", "fit", "fit_multiple_shooting"]))
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
def test_every_fitter_on_the_spring_converged_means_the_truth(truth, start, fitter):
    source = _spring(truth)
    s0 = source._user_state(source._state)  # noqa: SLF001
    obs = observations_from_history(s0, source.run_scan_with_history(N_STEPS)[1])
    position = obs["s"]["position"][1:]
    gm = _spring(start)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if fitter == "fit_lm":
            res = fit_lm(gm, lambda p: _spring(start).run_scan_with_history(
                N_STEPS, params=p)[1]["s"]["position"] - position,
                mask=_only_damping(gm), n_iter=40)
        elif fitter == "fit":
            loss = jax.jit(lambda p: 0.5 * jnp.sum((_spring(start).run_scan_with_history(
                N_STEPS, params=p)[1]["s"]["position"] - position) ** 2))
            res = fit(gm, loss, mask=_only_damping(gm), n_iter=200, lr=0.2,
                      tol=1e-12 * max(float(loss(gm.params)), 1e-30))
        else:
            res, _ = fit_multiple_shooting(
                gm, obs, obs_fn=lambda h: h["s"]["position"], window=20,
                mask=_only_damping(gm), n_iter=200, lr=0.2, lr_states=1e-6, tol=1e-12)
    got = float(res.params["nodes"]["s"]["damping"])
    note(f"{fitter}: truth={truth} start={start} got={got} converged={res.converged}")
    assert abs(got - truth) <= 1e-3 * (1.0 + truth) or not res.converged
