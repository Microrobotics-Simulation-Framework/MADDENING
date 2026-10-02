"""The identifiability guard decides the same in any units.

``hold_undetermined`` (on by default in every fitter) asks two questions of
each direction -- did any gradient of the run point along it, and does the
objective curve along it where the fit ended -- then holds the directions
that pass both at their start, unless that costs more loss than rounding
can.  Every one of those steps compares coordinates with each other.  Asked
in the optimiser's coordinates ``theta``, where an identity-transform
parameter is in its own units, the answers depended on the units: ``fit_lm``
on ``r = A diag(u) x - b``, damping (an identity coordinate) in units ``u``,
reached ``1.6e-11`` and then held the damping direction -- which the data
determines -- at its start, returning a loss of 13.9 at ``u = 1e-5`` (and
0.1 against ``3.1e-12`` at ``1e6``), with ``hold_declined`` False and no
warning; at ``u = 1e-4`` to ``1e-2`` and ``1e4`` it declined with a
``RuntimeWarning`` and reported ``excited_rank`` 2 of 3 for data that
determines all three.  The guard now asks in coordinates measured relative
to each identity parameter's size (``sysid._relative_scale``), which no
change of units moves.

Here: ``fit_lm``'s answer, rank and verdict over seven decades of units; a
hold that must happen holding the same physical combination in every unit;
a hold that must be declined declined in every unit; the shared guard
itself, for both curvature tests, given the same problem in two unit
systems; and the coordinate with no size of its own.  The truth-recovery
oracle's unit table (``tests/property/test_sysid_truth_recovery.py``)
asserts the same over generated problems.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import (
    _ExcitationTracker,
    _SelectedObjective,
    _gauss_newton_flatness,
    _hessian_flatness,
    _hold_undetermined_directions,
    _relative_scale,
    fit_lm,
)

KEYS = ("stiffness", "damping", "rest_length")
#: Seven decades either side of 1, including every unit the 0.4.0-dev guard
#: got wrong (held: 1e-5, 1e6; declined: 1e-4, 1e-3, 1e-2, 1e4).
UNITS = [1e-5, 1e-4, 1e-3, 1e-2, 1e4, 1e6]


def _spring():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.0,
                                 rest_length=1.0, initial_position=0.2))
    gm.compile()
    return gm


def _mask(gm):
    mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
    for key in KEYS:
        mask["nodes"]["s"][key] = True
    return mask


def _x(p, unit):
    """``(stiffness, damping, rest_length)`` in the model's units: the
    damping leaf (identity transform, bounds ``(0, None)``) is in units
    ``unit``, so the model reads ``unit`` times it."""
    q = p["nodes"]["s"]
    return jnp.stack([q["stiffness"], q["damping"] * unit, q["rest_length"]])


def _start(gm, unit, damping=2.0, **values):
    """The start, physical values; the damping leaf holds ``damping / unit``."""
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s"]["damping"] = jnp.asarray(damping / unit, jnp.float32)
    for key, value in values.items():
        p["nodes"]["s"][key] = jnp.asarray(value, jnp.float32)
    return p


def _physical(res, unit):
    q = res.params["nodes"]["s"]
    return np.array([float(q["stiffness"]), float(q["damping"]) * unit,
                     float(q["rest_length"])])


def _fit_lm(residual, gm, start, **kw):
    """``fit_lm`` with warnings as errors: a declined hold warns, and a
    decline that differs between units is the defect this file pins."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        return fit_lm(gm, residual, params=start, mask=_mask(gm), n_iter=50, **kw)


# ---------------------------------------------------------------------------
# A well-posed fit: nothing to hold, in any units
# ---------------------------------------------------------------------------

_A = np.random.default_rng(0).normal(size=(12, 3))
_TRUTH = np.array([40.0, 3.0, 1.2])


def _determined(unit):
    """The 0.4.0-dev reproducer: ``A x - b`` determines every coordinate."""
    gm = _spring()
    b = jnp.asarray(_A @ _TRUTH, jnp.float32)
    A = jnp.asarray(_A, jnp.float32)
    res = _fit_lm(lambda p: A @ _x(p, unit) - b, gm, _start(gm, unit))
    return res, _physical(res, unit)


@pytest.fixture(scope="module")
def determined_in_si():
    return _determined(1.0)


@pytest.mark.parametrize("unit", UNITS)
def test_fit_lm_with_its_defaults_answers_the_same_in_any_units(determined_in_si, unit):
    """Full rank, nothing held, nothing declined and no warning, the truth
    to float32's floor -- in every unit, as in SI.  0.4.0-dev: the hold
    returned damping 2.0 (its start) at 1e-5 and 3.085 at 1e6, and declined
    with a warning at 1e-4 to 1e-2 and 1e4, reporting rank 2 of 3."""
    base, base_x = determined_in_si
    res, x = _determined(unit)
    assert (base.excited_rank, base.hold_declined) == (3, False)
    assert (res.excited_rank, res.hold_declined, res.undetermined_drift) == (3, False, 0.0)
    np.testing.assert_allclose(base_x, _TRUTH, rtol=1e-5)
    np.testing.assert_allclose(x, base_x, rtol=1e-4)
    assert res.converged and abs(res.n_iter - base.n_iter) <= 2, (base.n_iter, res.n_iter)


# ---------------------------------------------------------------------------
# A degenerate fit: the hold is made, on the same combination, in any units
# ---------------------------------------------------------------------------

_B = np.random.default_rng(1).normal(size=(8, 2))
#: What the data determine: ``log k + 0.5 c`` and ``rest_length``.
_COMBINATION, _REST = float(np.log(20.0) + 0.5), 1.2
#: The start: damping 4, which the fit lowers (to about 3.1), so the
#: guard's scale for it is the start's: 1 / 4.
_K0, _C0 = 30.0, 4.0
#: Along the undetermined direction ``(d log k, d c) ~ (-0.5, 1)``, which in
#: the guard's coordinates ``(log k, c / 4)`` is ``~ (-2, 1)``, the
#: coordinate the hold puts back at its start is ``-2 log k + c / 4``.
_S0 = float(-2.0 * np.log(_K0) + _C0 / 4.0)


def _degenerate_residual(unit, amplitude=0.0):
    """The data read ``log k + 0.5 c`` and ``rest_length`` only: one
    direction, mixing the ``log`` stiffness with the identity damping, is
    exactly undetermined, and Levenberg-Marquardt drifts along it.  With an
    ``amplitude``, a ``bump`` read through ``stop_gradient`` (no gradient,
    no curvature) is ``amplitude`` where the held coordinate is at its
    start and vanishes a few widths away: holding the flat direction then
    puts the loss up by about ``amplitude**2``, which no rounding explains."""
    B = jnp.asarray(_B, jnp.float32)
    y = B @ jnp.asarray([_COMBINATION, _REST], jnp.float32)

    def residual(p):
        x = _x(p, unit)
        rows = B @ jnp.stack([jnp.log(x[0]) + 0.5 * x[1], x[2]]) - y
        if not amplitude:
            return rows
        s = -2.0 * jnp.log(x[0]) + x[1] / 4.0
        bump = jax.lax.stop_gradient(amplitude * jnp.exp(-((s - _S0) / 1e-3) ** 2))
        return jnp.concatenate([rows, bump[None]])

    return residual


def _degenerate(unit):
    gm = _spring()
    residual = _degenerate_residual(unit)
    start = _start(gm, unit, damping=_C0, stiffness=_K0)
    res = _fit_lm(residual, gm, start)
    raw = fit_lm(gm, residual, params=start, mask=_mask(gm), n_iter=50,
                 hold_undetermined=False)
    return res, _physical(res, unit), _physical(raw, unit)


def _held_coordinate(x):
    return -2.0 * np.log(x[0]) + x[1] / 4.0


@pytest.fixture(scope="module")
def degenerate_in_si():
    return _degenerate(1.0)


@pytest.mark.parametrize("unit", [1e-5, 1e6])
def test_the_hold_holds_the_same_combination_in_any_units(degenerate_in_si, unit):
    """Rank 2 of 3, held, not declined, the determined combinations where
    the fit put them and the undetermined one back at its start -- the same
    physical parameters in every unit.  The hold removes the move along the
    undetermined direction orthogonally in coordinates relative to each
    identity parameter's size, so the combination it holds at its start is
    a property of the parameters, not of the units they are written in."""
    base, base_x, base_raw = degenerate_in_si
    res, x, raw = _degenerate(unit)
    assert (base.excited_rank, base.hold_declined) == (2, False)
    assert (res.excited_rank, res.hold_declined) == (2, False)
    assert abs(_held_coordinate(base_raw) - _S0) > 1e-2      # unguarded, it drifted
    for got in (base_x, x):
        assert np.log(got[0]) + 0.5 * got[1] == pytest.approx(_COMBINATION, rel=1e-5)
        assert got[2] == pytest.approx(_REST, rel=1e-5)
        assert _held_coordinate(got) == pytest.approx(_S0, abs=1e-5)
    np.testing.assert_allclose(x, base_x, rtol=1e-5)


@pytest.mark.parametrize("unit", [1.0, 1e-5, 1e6])
def test_a_hold_that_would_raise_the_loss_is_declined_in_any_units(unit):
    """The degenerate fit with a bump of 0.03 where the hold would land: a
    rise of about ``9e-4`` from a fit at the float32 floor, declined in
    every unit, the selected iterate returned bit for bit.  Measured in
    ``theta``, the tolerance's quantisation term admitted rises of this
    size at ``1e-5`` (the damping coordinate is 4e5, and the curvature of
    the others multiplied its rounding) and at ``1e6`` (the floor of 1
    claimed a rounding of a percent of a coordinate of 4e-6)."""
    gm = _spring()
    residual = _degenerate_residual(unit, amplitude=0.03)
    start = _start(gm, unit, damping=_C0, stiffness=_K0)
    with pytest.warns(RuntimeWarning, match="would raise the loss"):
        res = fit_lm(gm, residual, params=start, mask=_mask(gm), n_iter=50)
    raw = fit_lm(gm, residual, params=start, mask=_mask(gm), n_iter=50,
                 hold_undetermined=False)
    assert res.hold_declined is True and res.excited_rank == 2
    assert [np.asarray(v).tobytes() for v in jax.tree.leaves(res.params)] == \
        [np.asarray(v).tobytes() for v in jax.tree.leaves(raw.params)]


# ---------------------------------------------------------------------------
# The shared guard, given the same problem in two unit systems
# ---------------------------------------------------------------------------

#: The physical problem: ``0.5 ||M p - y||²`` over three identity
#: parameters, ``M``'s second and third columns parallel, so ``(0, 1, -2)``
#: (scaled) is exactly undetermined.
_M = np.array([[1.0, 2.0, 1.0], [0.5, -1.0, -0.5], [2.0, 0.5, 0.25], [1.0, 1.0, 0.5]])
_P0 = np.array([1.0, 2.0, 0.5])            # the start
_P = np.array([1.3, 2.4, 0.3])             # the selected iterate
_GRADS = [np.array([1.0, -0.5, 2.0, 0.3]), np.array([0.2, 1.0, -1.0, 0.7]),
          np.array([-0.4, 0.3, 0.8, 1.1]), np.array([0.6, -0.2, 0.1, -0.9])]


def _guard(units, curvature, loss_rise=0.0):
    """``_hold_undetermined_directions`` on the physical problem written in
    ``units`` (``theta = p / units``): gradients ``units * Mᵀ r``, Jacobian
    ``M diag(units)``, Hessian ``diag(units) MᵀM diag(units)``."""
    u = np.asarray(units, dtype=np.float64)
    J = _M * u[None, :]
    tracker = _ExcitationTracker(3, np.float32)
    for r in _GRADS:
        tracker.observe((J.T @ r).astype(np.float32))
    theta0 = jnp.asarray(_P0 / u, jnp.float32)
    theta = jnp.asarray(_P / u, jnp.float32)

    def loss(th):
        p = np.asarray(th, np.float64) * u
        r = _M @ p - _M @ _P
        return 0.5 * float(r @ r) + loss_rise

    def flatness(candidates, spanned, scale):
        if curvature == "gauss_newton":
            return _gauss_newton_flatness(J, candidates, np.float32, scale=scale)
        return _hessian_flatness(lambda V: (J.T @ J) @ V, candidates, spanned,
                                 np.float32, "test", scale)

    objective = _SelectedObjective(
        loss=loss, reference=lambda: (0.0, np.zeros(3)), flatness=flatness,
        transformed=np.zeros(3, dtype=bool), columns=tracker.gradient_scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out, rank, _, declined = _hold_undetermined_directions(
            tracker, theta, theta0, objective, "test")
    return np.asarray(out, np.float64) * u, rank, declined


@pytest.mark.parametrize("curvature", ["gauss_newton", "hessian"])
@pytest.mark.parametrize("units", [(1.0, 1e-5, 1.0), (1e6, 1.0, 1e-3), (1e-4, 1e4, 1e2)])
def test_the_guard_decides_the_same_in_any_units(curvature, units):
    """Rank, verdict and held point (in physical units) are the same as in
    SI, for a hold that is made and for one whose held loss is far outside
    the tolerance and so is declined."""
    for rise, declined_want in ((0.0, False), (1e-3, True)):
        base, base_rank, base_declined = _guard((1.0, 1.0, 1.0), curvature, rise)
        held, rank, declined = _guard(units, curvature, rise)
        assert (base_rank, base_declined) == (2, declined_want)
        assert (rank, declined) == (base_rank, base_declined)
        np.testing.assert_allclose(held, base, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# The scale itself
# ---------------------------------------------------------------------------


def test_the_scale_is_one_for_a_transformed_coordinate_and_relative_otherwise():
    c = _relative_scale(np.array([0.5, -2.0, 3.0]), np.array([0.7, -4.0, 1.0]),
                        np.array([True, False, False]))
    np.testing.assert_array_equal(c, [1.0, 0.25, 1.0 / 3.0])
    assert np.array_equal(_relative_scale([1.0, 2.0], [3.0, 4.0], None), [1.0, 1.0])


def test_a_coordinate_with_no_size_of_its_own_is_matched_to_the_others():
    """An identity coordinate that started and ended at exactly 0 takes the
    scale that makes its column as large as the largest of the others in
    the scaled coordinates -- which a change of its units leaves alone --
    and one the objective does not read at all keeps 1."""
    transformed = np.array([True, False, False, False])
    theta = np.array([0.3, 2.0, 0.0, 0.0])
    for unit in (1.0, 1e-5, 1e6):
        columns = np.array([4.0, 6.0, 5.0 * unit, 0.0])
        c = _relative_scale(theta, theta, transformed, lambda: columns)
        np.testing.assert_allclose(columns[:3] / c[:3], [4.0, 12.0, 12.0])
        assert c[3] == 1.0


# ---------------------------------------------------------------------------
# fim's own rank verdict: invariant under its default scale, not under None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit, raw_rank", [(1.0, 3), (1e-5, 2), (1e6, 1)])
def test_fims_rank_is_the_same_in_any_units_under_its_default_scale(unit, raw_rank):
    """``scale="relative"`` (the default) multiplies each column by its
    parameter's value, which a change of units leaves alone, so ``rank`` and
    ``cond`` are those of SI.  ``scale=None`` is raw sensitivities, as its
    documentation says, and its verdicts compare columns in their own units:
    the damping column in units 1e-5 reads as unresolved, and in units 1e6
    it drowns the other two."""
    from maddening.sysid import fim

    A = jnp.asarray(_A, jnp.float32)

    def problem(u):
        def residual(q):
            return A @ jnp.stack([q["k"], q["c"] * u, q["r"]])
        return residual, {"k": jnp.float32(40.0), "c": jnp.float32(3.0 / u),
                          "r": jnp.float32(1.2)}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # a precision-limited verdict
        relative, raw = fim(*problem(unit)), fim(*problem(unit), scale=None)
        si = fim(*problem(1.0))
    assert int(relative.rank) == 3 and int(raw.rank) == raw_rank
    assert float(relative.cond) == pytest.approx(float(si.cond), rel=1e-4)
