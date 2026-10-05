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
#: repro_fitters_tiny_residual_units.py).  At ``1e37`` the sum of even the
#: column-framed ``Jᵀr`` overflows unless the residual is framed too.  Below
#: about ``1e-31`` the residual's own rounding at the optimum (``s * eps``)
#: is not a normal number: the model flushes it to an exact zero, which no
#: fitter can tell from an exact fit, so the range ends there.
_RESIDUAL_SCALES = [1e-30, 1e-25, 1e-20, 1e-19, 1e-18, 1e-10, 1e10, 1e18, 1e19,
                    1e20, 1e25, 1e30, 1e37]

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
    # The identifiability guard sees the same gradients (its tracker reads
    # ``Jᵀr`` framed; bare, it flushed to zero and the rank read None).
    assert (res.excited_rank, res.hold_declined) == (base.excited_rank, base.hold_declined)
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
    assert res.excited_rank == base.excited_rank, (res.excited_rank, base.excited_rank)
    for (key, got, _), (_, got_scaled, _), kind in zip(base_errors, errors, kinds):
        lo, hi = _SPECS[kind][1]
        assert abs(got_scaled - got) <= 1e-4 * (hi - lo), (key, got, got_scaled)


def test_fit_does_not_count_a_flushed_loss_as_reaching_tol():
    """``loss_fn`` returning exactly 0.0 with a gradient that is not zero has
    flushed its own value; with ``tol > 0`` it used to stop at its first
    evaluation, ``converged=True``, unfitted.  It is warned about and fitted."""
    kinds, truth_at, start_at = _RANGE_CASE
    gm = _graph(kinds)
    truth = _start(gm, kinds, truth_at)
    data = _blocks(truth)
    s = jnp.asarray(1e-20, data.dtype)
    loss = jax.jit(lambda p: 0.5 * jnp.sum((s * (_blocks(p) - data)) ** 2))
    start = _start(gm, kinds, start_at)
    assert float(loss(start)) == 0.0                      # the caller's loss flushed
    with pytest.warns(RuntimeWarning, match="returned exactly 0.0 at iteration 1"):
        res = fit(gm, loss, params=start, mask=_mask(gm), n_iter=400, lr=0.05, tol=1e-30)
    assert not res.converged and res.n_iter == 400
    assert _recovered(res, truth, kinds)[0]


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


# ---------------------------------------------------------------------------
# The transform grid: wherever the identity control recovers the truth, the
# transformed fit does
# ---------------------------------------------------------------------------
#
# Four audit rounds found a defect where a bounded transform (``log``,
# ``logit``) meets the floating-point limits inside a fitter, each the
# neighbour of the one fixed before it.  The last: ``fit_lm`` could not bring
# a ``logit`` coordinate back from the edge of its range, so a fit that an
# early step carried there ended on its bound, ``converged=False``, with no
# warning (audit_040_p4_12/fmu-sysid/repro_F1_logit_edge_trap.py).  So this
# oracle is over the whole class rather than a case: the guide's spring, its
# damping under each transform, at every position of truth and start --
# interior, and a few float spacings from each edge -- beside the same fit
# with the damping under the identity transform and the same bounds, which
# clips: the control (``tests/property/sysid_transform_grid.py``).  Wherever
# the control recovers the truth the transformed fit must, or it must say,
# naming the parameter, that it ended on the edge of its transform's range.

from tests.property import sysid_transform_grid as grid  # noqa: E402

#: Cells of the audit's reproducer that ended on a bound, as ``(precision,
#: the damping the data hold, bounds, the damping's start, the stiffness's
#: start over its truth)``.  Under x64 the trap needed nothing but the
#: guide's own start; in float32 it needed the stiffness started further
#: off.  ``mixed`` is float32 leaves in an x64 graph.
_EDGE_TRAP_CELLS = {
    # The traced one: 2.0 for a truth of 1.9 in (0.5, 2), n_iter 4.
    "x64-upper-edge-from-the-guides-start": ("x64", 1.9, (0.5, 2.0), 0.6, 1.5),
    "mixed-upper-edge-from-the-guides-start": ("mixed", 1.9, (0.5, 2.0), 0.6, 1.5),
    # Every start in (0, 10) ended on 10 for a truth of 9.5.
    "x64-upper-edge-from-mid-range": ("x64", 9.5, (0.0, 10.0), 5.0, 1.5),
    # The lower edge: 3.6e-15 (1.9e-6 in float32) for a truth of 1.2 in (0, 4).
    "x64-lower-edge": ("x64", 1.2, (0.0, 4.0), 0.2, 5.0 / 30.0),
    "float32-lower-edge": ("float32", 1.2, (0.0, 4.0), 0.2, 5.0 / 30.0),
    "float32-upper-edge": ("float32", 5.0, (0.0, 10.0), 9.5, 5.0 / 30.0),
    "float32-upper-edge-stiffness-from-1000": ("float32", 5.0, (0.0, 10.0), 5.0,
                                               1000.0 / 30.0),
}


@pytest.mark.parametrize("cell", sorted(_EDGE_TRAP_CELLS))
def test_fit_lm_comes_back_from_the_edge_of_a_logit_range(cell):
    """The reproducer's cells.  Each ended with the damping on a bound of
    its ``logit`` range, ``converged=False``, from a start and for a truth
    well inside it, where the same fit with the bounds clipped recovered the
    truth.  It comes back -- converged, at the truth -- and, not having
    ended on an edge, says nothing about one."""
    kind, truth, bounds, start, other = _EDGE_TRAP_CELLS[cell]
    with grid.precision(kind != "float32"):
        problem = grid.build_problem(masked=False, damping=truth,
                                     float32_leaves=kind == "mixed")
        result = grid.run_cell(problem, "fit_lm", "logit", "-", "-", other, bounds=bounds,
                               start_damping=start)
        if kind == "mixed":
            assert problem.truth["nodes"]["spring"]["damping"].dtype == np.float32
            assert problem.truth["nodes"]["spring"]["mass"].dtype == np.float64
    assert result.control_ok, f"the control no longer recovers this cell: {result}"
    assert result.fit_ok and result.converged and not result.edge_warned, str(result)
    assert not result.on_edge, str(result)


#: Per push, the fitter and precision of the two tests below; the grid
#: (slow) runs every one.  Multiple shooting compiles a windowed loss and
#: its gradient per fit, which is the grid's to pay for.
_PER_PUSH = [("fit_lm", False), ("fit_lm", True), ("fit", False)]


@pytest.mark.parametrize("fitter, x64", _PER_PUSH,
                         ids=[f"{f}-{'x64' if x else 'float32'}" for f, x in _PER_PUSH])
def test_a_start_on_the_edge_of_a_logit_range_comes_back(fitter, x64):
    """A start the documented interior allows, a few float spacings inside a
    ``logit`` bound, with the truth at the far side of the range: the fit
    leaves the edge and recovers it, as its control does.  (``fit`` by the
    grid's annealed schedule: Adam's step is about ``lr`` whatever the
    gradient, and the edge is 8 units of the coordinate out in float32, 18
    in float64.)"""
    with grid.precision(x64):
        problem = grid.build_problem(masked=False)
        with grid.shared_programs({}, fitter):          # one compile for the four fits
            for start_at, truth_at in (("lower-edge", "95%"), ("upper-edge", "5%")):
                result = grid.run_cell(problem, fitter, "logit", truth_at, start_at, 1.5)
                assert result.control_ok, str(result)
                assert result.fit_ok and not result.edge_warned, str(result)
                assert not result.on_edge, str(result)


_SAYS_SO = [("fit_lm", False), ("fit_lm", True), ("fit", False),
            ("fit_multiple_shooting", False)]


@pytest.mark.parametrize("fitter, x64", _SAYS_SO,
                         ids=[f"{f}-{'x64' if x else 'float32'}" for f, x in _SAYS_SO])
def test_a_fit_that_ends_on_the_edge_of_its_transform_says_so(fitter, x64):
    """The other ending.  A ``logit`` parameter whose optimum is nearer its
    bound than the transform resolves (``sqrt(eps)`` of the bounds' size)
    stops on that margin, strictly inside the bounds, and a
    ``RuntimeWarning`` names the parameter, the edge and the remedy; it
    ended there in silence.  The control, which clips, sits on the bound
    itself and has nothing to say.

    ``fit_lm``: the truth a few float spacings inside the upper bound.
    ``fit``: a loss that pushes at the bound with Adam steps of 50 (Adam
    alone closes on a flat edge too slowly to arrive: its step shrinks with
    the gradient it remembers).  ``fit_multiple_shooting``: started on the
    edge with the truth there, which it says twice -- when it starts and
    when it ends."""
    with grid.precision(x64):
        problem = grid.build_problem(masked=False)
        gm, only_damping = problem.gm, problem.only("damping")
        bounds = grid.bounds_around("logit", "upper-edge")
        start = problem.start(grid.STIFFNESS, grid.value_at("logit", bounds, "50%"))
        problem.set_damping_spec(grid.spec_for("logit", bounds))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if fitter == "fit_lm":
                res = fit_lm(gm, problem.residual, params=start, mask=only_damping)
            elif fitter == "fit":
                res = fit(gm, lambda p: -50.0 * p["nodes"]["spring"]["damping"],
                          params=start, mask=only_damping, lr=50.0, n_iter=40)
            else:
                on_edge = problem.start(grid.STIFFNESS,
                                        grid.value_at("logit", bounds, "upper-edge"))
                res, _ = fit_multiple_shooting(
                    gm, problem.observations, obs_fn=lambda h: h["spring"]["position"],
                    window=20, params=on_edge, mask=only_damping, lr=1e-3, lr_states=1e-6,
                    n_iter=3)
        assert all(issubclass(w.category, RuntimeWarning) for w in caught), caught
        texts = [str(w.message) for w in caught]
        assert any(grid.EDGE_WARNING in t for t in texts), texts
        said = [t for t in texts if grid.EDGE_WARNING in t][-1]
        for part in (f"{fitter}:", "node 'spring', parameter 'damping'", "the upper edge",
                     "'logit'", "transform=None"):
            assert part in said, (part, said)
        if fitter == "fit_multiple_shooting":
            started = [t for t in texts if grid.EDGE_START_WARNING in t]
            assert started and "node 'spring', parameter 'damping'" in started[0], texts
        else:
            assert not any(grid.EDGE_START_WARNING in t for t in texts), texts
            value = res.params["nodes"]["spring"]["damping"]
            margin = float(np.sqrt(np.finfo(np.asarray(value).dtype).eps)) * bounds[1]
            assert bounds[0] < float(value) < bounds[1]
            assert bounds[1] - float(value) == pytest.approx(margin, rel=1e-2), float(value)
        problem.set_damping_spec(grid.control_for("logit", bounds))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            control = fit_lm(gm, problem.residual, params=start, mask=only_damping)
        assert float(control.params["nodes"]["spring"]["damping"]) == pytest.approx(
            bounds[1], abs=2e-3 * (bounds[1] - bounds[0]))


#: Compiled model-side programs shared across the cells of one block
#: (``grid.shared_programs``), and the spring of each precision and mask.
_GRID_PROGRAMS: dict = {}
_GRID_PROBLEMS: dict = {}

#: Where the differential oracle holds for the Adam fitters: a stiffness
#: started within a factor of ten below, or two either side, of its truth.
#: From ten or thirty times too stiff the loss has other basins (a fast
#: oscillation is best fitted by damping it out), and Adam in a ``log``
#: coordinate, whose step multiplies the value, walks into them where Adam
#: in the value's own coordinate does not: a difference between two
#: parametrisations of one problem, with no edge in it.  Those cells are
#: still run, and held to what no parametrisation excuses: never on an
#: edge of the transform's range without the warning.
_ADAM_NEAR = (0.1, 0.5, 1.5)


def _grid_block(fitter, transform, x64, masked):
    key = (x64, masked)
    # A fit compiles a handful of small programs of its own (its Adam update,
    # the guard's products); thousands of fits in one process keep them all.
    for stale in [slot for slot in _GRID_PROGRAMS if slot[0] != (fitter, x64, masked)]:
        del _GRID_PROGRAMS[stale]
    jax.clear_caches()
    with grid.precision(x64):
        if key not in _GRID_PROBLEMS:
            _GRID_PROBLEMS[key] = grid.build_problem(masked)
        problem = _GRID_PROBLEMS[key]
        with grid.shared_programs(_GRID_PROGRAMS, (fitter, x64, masked)):
            return [grid.run_cell(problem, fitter, transform, truth_at, start_at, other)
                    for truth_at in grid.POSITIONS for start_at in grid.POSITIONS
                    for other in grid.OTHER_STARTS]


# Per push: tests/property/test_sysid_truth_recovery.py::test_fit_lm_comes_back_from_the_edge_of_a_logit_range
# Per push: tests/property/test_sysid_truth_recovery.py::test_a_start_on_the_edge_of_a_logit_range_comes_back
# Per push: tests/property/test_sysid_truth_recovery.py::test_a_fit_that_ends_on_the_edge_of_its_transform_says_so
@pytest.mark.slow  # 125 cells, two fits each, a block: minutes
@pytest.mark.parametrize("masked", [False, True], ids=["frozen-by-spec", "left-out-by-mask"])
@pytest.mark.parametrize("x64", [False, True], ids=["float32", "x64"])
@pytest.mark.parametrize("transform", grid.TRANSFORMS)
@pytest.mark.parametrize("fitter", grid.FITTERS)
def test_the_transformed_fit_recovers_the_truth_wherever_its_control_does(
        fitter, transform, x64, masked):
    """The grid: transform x precision x where the truth sits (5%, 50%, 95%
    of the range, a few float spacings from each edge) x where the start
    sits (likewise) x the other parameter's start (0.1, 0.5, 1.5, 10 and 30
    times its truth) x the third leaf frozen by its spec or left out by
    ``mask=`` under a ``logit`` a million times wider than its value x the
    three fitters.  Wherever the identity-transform control recovers the
    truth, the transformed fit does, or it warns by name that it ended on
    the edge of its transform's range; where truth and start are both well
    inside the range ``fit_lm`` has no such excuse; and no fit, in any
    cell, ends on an edge without the warning."""
    cells = _grid_block(fitter, transform, x64, masked)
    assert len(cells) == 125
    silent = [str(c) for c in cells if c.silent_on_an_edge]
    assert not silent, "\n".join(silent)
    judged = [c for c in cells if fitter == "fit_lm" or c.other in _ADAM_NEAR]
    violations = [str(c) for c in judged if c.violates]
    assert not violations, "\n".join(violations)
    if fitter == "fit_lm":
        # Truth and start both well inside: recovery, not a warning.
        inside = ("5%", "50%", "95%")
        excused = [str(c) for c in cells if c.control_ok and not c.fit_ok
                   and c.truth_at in inside and c.start_at in inside]
        assert not excused, "\n".join(excused)
        # ``converged=True`` is the truth, or an edge the fit warned about.
        wrong = [str(c) for c in cells if c.converged and not c.fit_ok and not c.edge_warned]
        assert not wrong, "\n".join(wrong)
    # Non-vacuity: the control recovers the truth in most of what is judged,
    # and the transformed fit recovers it (rather than warning) in most of that.
    recovered = [c for c in judged if c.control_ok]
    assert len(recovered) >= len(judged) // 2, (len(recovered), len(judged))
    assert sum(c.fit_ok for c in recovered) >= (2 * len(recovered)) // 3
