"""No fitter may move along directions the data cannot determine.

The spring's ``(k, c, m)`` common-scale degeneracy is exact: position-only
data sees ``k/m`` and ``c/m``, so multiplying all three by the same factor
leaves the trajectory bit-identical.  In relative coordinates that is the
direction ``(1, 1, 1)/sqrt(3)``, which
``tests/verification/hypothesis/test_hypothesis_sysid.py`` and
``FIMReport.least_identifiable`` both name, and here it is the direction
along which the *geometric mean* of ``(k, c, m)`` is the coordinate.

Every fitter here moves along it, each by its own route.  ``g . v`` is
exactly zero because every gradient is ``J^T r``, but no step rule in this
module is the gradient:

* ``fit`` and ``fit_multiple_shooting`` take ``-lr * D g`` for a diagonal
  Adam preconditioner ``D``, and ``(D g) . v`` is not zero.  The drift does
  not converge, so the fitted scale is decided by the iteration budget and
  the learning rate rather than by the data.  That is a reproducibility
  defect.
* ``fit_lm`` solves ``(A + lam * diag(A))^-1 g``, which is orthogonal to
  ``null(A)`` only where ``diag(A)`` is isotropic there.  Its step vanishes
  with the gradient, so its drift *converges*: the answer is the same for
  every budget, it is simply neither the caller's value nor one the data
  chose.  That is a consistency defect and not the same thing, and the
  tests below assert the difference rather than eliding it.

What is measured is that the guard removes that, and that it removes
nothing else.  Since the guard learned to confirm each direction its
gradient test raises against the objective's curvature at the returned
point, and to refuse a hold that would raise the loss, every guarded
result here also asserts ``hold_declined is False``: the spring's scale is
exactly flat, so a decline on it would be the guard failing at its one
job.  The cases where it must decline, and the ones where 0.4.0-dev held
directions the data determined, are ``test_sysid_hold_never_raises_the_loss.py``.
"""

import contextlib
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening import sysid
from maddening.sysid import (
    _ExcitationTracker,
    fim,
    fit,
    fit_lm,
    fit_multiple_shooting,
    init_window_states,
    observations_from_history,
    windowed_loss,
)

DT = 0.01
N_STEPS, WINDOW = 120, 20
TRUTH = {"stiffness": 30.0, "damping": 2.0, "mass": 1.0}
START = {"stiffness": 45.0, "damping": 3.0, "mass": 1.0}
#: Noise on the observations, so the residual does not vanish and Adam
#: keeps taking (sign-like, ~lr) steps after the fit has converged.  That
#: is where the unbounded part of the drift lives: with noiseless data the
#: float32 loss reaches exactly 0.0, the gradient with it, and Adam stops.
NOISE_STD = 0.02


def _spring(all_log=True):
    """Spring graph with ``(k, c, m)`` trainable and ``rest_length`` frozen.

    ``damping`` is given ``transform="log"``, which the shipped
    :class:`SpringDamperNode` spec does not: it declares ``bounds=(0, None)``
    with the identity transform so that a damping of exactly ``0.0`` stays
    representable (``log`` gives ``p = lo + exp(u) > lo`` strictly).  The
    log transform is what makes the scale degeneracy a *fixed* direction in
    the optimiser's coordinates -- the same coordinates
    ``fim(scale="relative")`` works in -- and
    ``test_a_rotating_null_direction_is_reported_as_full_rank`` covers the
    shipped spec, where it is not.
    """
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, rest_length=1.0, initial_position=0.5,
                                 **TRUTH))
    gm.compile()
    gm.set_param_spec("s", "rest_length", ParamSpec(trainable=False, units="m"))
    if all_log:
        gm.set_param_spec("s", "damping", ParamSpec(
            bounds=(0.0, None), transform="log", units="N*s/m"))
    return gm


def _with(gm, values):
    p = jax.tree.map(lambda x: x, gm.params)
    for k, v in values.items():
        p["nodes"]["s"][k] = jnp.asarray(v, dtype=jnp.float32)
    return p


def _observations(gm, noisy):
    init = {n: gm.get_node_state(n) for n in gm.node_names}
    _, hist = gm.run_scan_with_history(N_STEPS, params=_with(gm, TRUTH))
    obs = observations_from_history(init, hist)
    if noisy:
        rng = np.random.default_rng(20260920)
        draw = rng.normal(0.0, NOISE_STD, size=obs["s"]["position"].shape)
        obs["s"]["position"] = obs["s"]["position"] + jnp.asarray(
            draw, dtype=jnp.float32)
    return obs


def _loss(gm, obs):
    return jax.jit(lambda p: windowed_loss(
        gm, p, obs, obs_fn=lambda h: h["s"]["position"], window=WINDOW))


def _scale(params):
    """The null coordinate: the geometric mean of ``(k, c, m)``.

    Multiplying all three by ``a`` multiplies this by ``a`` and leaves every
    ratio the data can see alone, so it is exactly the coordinate along the
    degenerate direction and nothing else.
    """
    s = params["nodes"]["s"]
    return float(np.cbrt(float(s["stiffness"]) * float(s["damping"])
                         * float(s["mass"])))


START_SCALE = float(np.cbrt(45.0 * 3.0 * 1.0))


@pytest.fixture(scope="module")
def noisy_fit():
    gm = _spring()
    return gm, _loss(gm, _observations(gm, noisy=True))


@pytest.fixture(scope="module")
def clean_fit():
    gm = _spring()
    return gm, _loss(gm, _observations(gm, noisy=False))


# ---------------------------------------------------------------------------
# The degeneracy this file is about is the one ``fim`` names
# ---------------------------------------------------------------------------


def test_the_held_direction_is_the_one_fim_calls_least_identifiable(clean_fit):
    """Ties the rest of this file to the tree's own diagnostic: the
    direction the guard holds must be the direction ``fim`` reports as
    unresolved, not a degeneracy this test invented."""
    gm, _ = clean_fit
    obs = _observations(gm, noisy=False)
    step_fn = gm._build_step_fn()                     # noqa: SLF001
    ext = gm._default_external_inputs()               # noqa: SLF001
    init = jax.tree.map(lambda x: x[0], obs)
    truth = obs["s"]["position"][1:]
    base = _with(gm, TRUTH)
    names = ("stiffness", "damping", "mass")

    def residual(sub):
        p = jax.tree.map(lambda x: x, base)
        for k in names:
            p["nodes"]["s"][k] = sub[k]

        def body(state, _):
            state = step_fn(state, ext, p)
            return state, state["s"]["position"]

        _, pos = jax.lax.scan(body, init, None, length=N_STEPS)
        return pos - truth

    sub = {k: base["nodes"]["s"][k] for k in names}
    rep = fim(residual, sub)
    assert rep.rank == 2, (rep.rank, rep.eigvals)
    v = np.asarray(rep.eigvecs[:, 0], dtype=np.float64)
    ones = np.ones(3) / np.sqrt(3.0)
    assert abs(float(v @ ones)) > 0.99, v


# ---------------------------------------------------------------------------
# The defect: the fitted scale is decided by the budget, not by the data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lr", [0.05, 0.2])
@pytest.mark.parametrize("n_iter", [200, 1200])
def test_fitted_scale_is_the_starting_scale_for_every_budget(noisy_fit, lr, n_iter):
    """The fit must return the scale it was given.

    The data determines ``k/m`` and ``c/m`` and says nothing at all about
    the overall scale, so the only defensible value for it is the one the
    caller supplied.  Without the guard this lands somewhere between -1.7%
    and +46% of the starting scale depending on ``lr`` and ``n_iter``, at a
    loss identical in its first six digits.
    """
    gm, loss = noisy_fit
    start = _with(gm, START)
    res = fit(gm, loss, params=start, n_iter=n_iter, lr=lr, notify_every=0)
    assert res.excited_rank == 2, res.excited_rank
    assert res.hold_declined is False
    assert res.undetermined_drift is not None and res.undetermined_drift > 1e-3
    # 1e-4 relative is ~3 float32 ulps on a quantity of order 5; the guard
    # zeroes this exactly in exact arithmetic.
    assert abs(_scale(res.params) / START_SCALE - 1.0) < 1e-4, _scale(res.params)


def test_the_scale_is_held_at_the_budget_the_guard_was_built_for(noisy_fit):
    """The guard was introduced after measuring the unguarded drift at
    ``lr=0.2`` out to 10,000 iterations.  Now that each candidate must also
    pass the curvature test and the loss check, it must still hold there.
    Measured: the scale lands +7.6e-8 from the start, against -5.43%
    unguarded, with ``undetermined_drift`` 0.097 removed and the loss of
    the selected iterate unchanged."""
    gm, loss = noisy_fit
    held = fit(gm, loss, params=_with(gm, START), n_iter=10_000, lr=0.2,
               notify_every=0)
    assert held.excited_rank == 2 and held.hold_declined is False
    assert held.undetermined_drift > 1e-2
    assert abs(_scale(held.params) / START_SCALE - 1.0) < 1e-4, _scale(held.params)
    assert float(loss(held.params)) <= held.best_loss * (1 + 2**-13) + 1e-12


def test_the_fitted_scale_does_not_depend_on_the_optimiser_schedule(noisy_fit):
    """The reproducibility statement: runs that differ only in their Adam
    schedule must return the same parameters, because the data they were
    given is the same.

    The ``hold_undetermined=False`` half is what makes this a regression
    test rather than a tautology -- it asserts the defect is still there
    when the guard is off, so a guard that silently stopped running could
    not make both halves pass.  Unguarded, this grid spans -7.5% to -5.4%
    of the starting scale; the schedule, not the data, picks the value.
    """
    gm, loss = noisy_fit
    start = _with(gm, START)
    schedule = [(0.01, 200), (0.01, 1200), (0.2, 200), (0.2, 1200)]

    held = [fit(gm, loss, params=start, lr=lr, n_iter=n, notify_every=0)
            for lr, n in schedule]
    for res, (lr, n) in zip(held, schedule):
        assert res.excited_rank == 2, (lr, n, res.excited_rank)
        assert res.hold_declined is False, (lr, n)
    scales = [_scale(r.params) for r in held]
    assert max(scales) / min(scales) - 1.0 < 1e-4, dict(zip(map(str, schedule),
                                                            scales))

    raw = [_scale(fit(gm, loss, params=start, lr=lr, n_iter=n, notify_every=0,
                      hold_undetermined=False).params)
           for lr, n in schedule]
    assert max(raw) / min(raw) - 1.0 > 1e-2, (
        dict(zip(map(str, schedule), raw)),
        "the unguarded schedule dependence this test exists for did not "
        "reproduce",
    )


def test_holding_the_scale_does_not_cost_loss_or_the_identifiable_ratios(noisy_fit):
    """The guard moves along a flat direction, so it must change neither
    the loss nor ``k/m`` and ``c/m`` -- the two combinations the data does
    determine."""
    gm, loss = noisy_fit
    start = _with(gm, START)
    kw = dict(params=start, n_iter=600, lr=0.2, notify_every=0)
    held = fit(gm, loss, **kw)
    raw = fit(gm, loss, hold_undetermined=False, **kw)

    l_held, l_raw = float(loss(held.params)), float(loss(raw.params))
    assert held.hold_declined is False
    assert l_held <= l_raw * 1.001 + 1e-9, (l_held, l_raw)
    for num in ("stiffness", "damping"):
        a = (float(held.params["nodes"]["s"][num])
             / float(held.params["nodes"]["s"]["mass"]))
        b = (float(raw.params["nodes"]["s"][num])
             / float(raw.params["nodes"]["s"]["mass"]))
        assert abs(a / b - 1.0) < 1e-3, (num, a, b)
    # ``losses`` is the record of the iterates, which the guard does not
    # touch; only the returned ``params`` differ.
    np.testing.assert_array_equal(held.losses, raw.losses)
    assert held.n_iter == raw.n_iter and held.converged == raw.converged


# ---------------------------------------------------------------------------
# It must not fire on anything else
# ---------------------------------------------------------------------------


def test_a_well_posed_fit_gets_its_iterate_back_bit_for_bit(clean_fit):
    """With ``mass`` frozen the remaining ``(k, c)`` are identifiable, so
    the guard has nothing to remove and must return the same bits as a run
    without it -- not merely the same value to a tolerance."""
    gm, loss = clean_fit
    gm = _spring()                       # own copy: this one freezes mass
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False, units="kg"))
    loss = _loss(gm, _observations(gm, noisy=True))
    start = _with(gm, START)
    kw = dict(params=start, n_iter=300, lr=0.1, notify_every=0)
    held = fit(gm, loss, **kw)
    raw = fit(gm, loss, hold_undetermined=False, **kw)
    assert held.excited_rank == 2, held.excited_rank
    assert held.undetermined_drift == 0.0 and held.hold_declined is False
    for key, value in held.params["nodes"]["s"].items():
        assert float(value) == float(raw.params["nodes"]["s"][key]), key
    assert float(held.params["nodes"]["s"]["mass"]) == 1.0


def test_a_rotating_null_direction_is_reported_as_full_rank(noisy_fit):
    """The guard's one real limit, pinned rather than left to be discovered.

    With the shipped :class:`SpringDamperNode` specs, ``damping`` uses the
    identity transform while ``stiffness`` and ``mass`` use ``log``, so the
    scale direction in the optimiser's coordinates is ``(c, 1, 1)`` -- it
    *rotates* as ``c`` moves.  No direction is null for the whole run, the
    accumulated matrix has full rank, and the guard correctly holds nothing
    rather than removing a direction the data partly saw.  It reports that:
    ``excited_rank`` is full and ``undetermined_drift`` is exactly 0.0.
    """
    gm = _spring(all_log=False)
    loss = _loss(gm, _observations(gm, noisy=True))
    start = _with(gm, START)
    kw = dict(params=start, n_iter=400, lr=0.2, notify_every=0)
    held = fit(gm, loss, **kw)
    raw = fit(gm, loss, hold_undetermined=False, **kw)
    assert held.excited_rank == 3, held.excited_rank
    assert held.undetermined_drift == 0.0 and held.hold_declined is False
    for key, value in held.params["nodes"]["s"].items():
        assert float(value) == float(raw.params["nodes"]["s"][key]), key


def test_fewer_iterations_than_parameters_answers_none_not_full_rank(noisy_fit):
    """Two gradients cannot tell an unobserved direction from an
    unobservable one, so the guard declines to answer.  ``None`` and not
    ``3``: a full-rank verdict would read as "nothing undetermined was
    found" when nothing looked."""
    gm, loss = noisy_fit
    start = _with(gm, START)
    kw = dict(params=start, n_iter=2, lr=0.2, notify_every=0)
    held = fit(gm, loss, **kw)
    raw = fit(gm, loss, hold_undetermined=False, **kw)
    assert held.excited_rank is None
    assert held.undetermined_drift is None and held.hold_declined is None
    for key, value in held.params["nodes"]["s"].items():
        assert float(value) == float(raw.params["nodes"]["s"][key]), key


def test_above_the_parameter_cap_the_guard_declines_and_says_so(
        noisy_fit, monkeypatch):
    """Over :data:`_EXCITATION_MAX_PARAMS` trainable leaves the ``n x n``
    accumulation is not worth its memory, so the guard is not built.  The
    cap is exercised by lowering it rather than by building a graph with
    513 parameters, which would measure pytest's patience and not this."""
    gm, loss = noisy_fit
    monkeypatch.setattr(sysid, "_EXCITATION_MAX_PARAMS", 2)
    start = _with(gm, START)
    held = fit(gm, loss, params=start, n_iter=200, lr=0.2, notify_every=0)
    assert held.excited_rank is None and held.undetermined_drift is None
    assert held.hold_declined is None


def test_hold_undetermined_refuses_a_non_bool(noisy_fit):
    """A truthy non-bool would silently decide whether the returned
    parameters are reproducible."""
    gm, loss = noisy_fit
    with pytest.raises(ValueError, match="hold_undetermined must be a bool"):
        fit(gm, loss, params=_with(gm, START), n_iter=2,
            hold_undetermined="yes")            # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ``fit_lm``: the same drift by a different route, and now the same guard
# ---------------------------------------------------------------------------


def _lm_residual(gm, obs):
    """``params -> residual`` for the whole trajectory, for ``fit_lm``.

    ``fit_lm`` wants the residual itself rather than a scalar loss, so it
    cannot share ``_loss``; everything else -- graph, observations,
    degeneracy -- is the one the rest of this file measures.
    """
    step_fn = gm._build_step_fn()                     # noqa: SLF001
    ext = gm._default_external_inputs()               # noqa: SLF001
    init = jax.tree.map(lambda x: x[0], obs)
    truth = obs["s"]["position"][1:]

    def residual(p):
        def body(state, _):
            state = step_fn(state, ext, p)
            return state, state["s"]["position"]

        _, pos = jax.lax.scan(body, init, None, length=N_STEPS)
        return pos - truth

    return residual


@pytest.mark.parametrize("noisy", [False, True])
def test_fit_lm_unguarded_moves_along_the_scale_direction_but_converges(noisy):
    """Pins the correction PR 100 made to the finding it came from, and
    keeps it visible now that the default hides it.

    The claim was that ``fit_lm`` *cannot* move in an exactly-null
    direction, because ``J v = 0`` makes ``(J^T r) . v = 0``.  The gradient
    is indeed orthogonal to ``v``, but the step is not the gradient: the
    Marquardt solve is ``(A + lam * diag(A))^-1 g``, and that is orthogonal
    to ``null(A)`` only when ``diag(A)`` is isotropic on the relevant
    subspace.  Counterexample in two dimensions: ``A = [[1, 2], [2, 4]]``
    and ``g = (1, 2)`` give a step ``prop (2, 1)`` for every ``lam``, while
    ``null(A)`` is spanned by ``(2, -1)``.

    What is different in kind, and why this is a consistency fix rather
    than the reproducibility defect ``fit`` had, is that the step vanishes
    as the gradient does: the drift converges and stops.  Both halves are
    asserted.  A future change that made LM's drift budget-dependent would
    be the same defect arriving by a different route, and a change that
    silently stopped LM drifting at all would make the guarded tests below
    tautologies -- this is what stops either passing unnoticed.
    """
    gm = _spring()
    residual = _lm_residual(gm, _observations(gm, noisy=noisy))
    start = _with(gm, START)
    kw = dict(params=start, notify_every=0, hold_undetermined=False)

    budgets = [5, 10, 25, 60, 200]
    raw = [fit_lm(gm, residual, n_iter=n, **kw) for n in budgets]
    drifts = [_scale(r.params) / START_SCALE - 1.0 for r in raw]
    # It moves: not the "cannot move" the finding claimed.  Measured on
    # this fixture: -0.849% noiseless, +0.429% at sigma = 0.02 -- four
    # decades above the 1e-7 the guarded runs below leave behind.
    assert all(abs(d) > 1e-3 for d in drifts), dict(zip(budgets, drifts))
    # And it has stopped, so the budget does not decide the answer.  From
    # the second budget on the scale is identical to the last bit; the
    # first is included to show the convergence, not asserted equal.
    settled = [_scale(r.params) for r in raw[1:]]
    assert len(set(settled)) == 1, dict(zip(budgets[1:], settled))
    # LM reports the guard it did not run.
    assert raw[0].excited_rank is None and raw[0].undetermined_drift is None
    assert raw[0].hold_declined is None


@pytest.mark.parametrize("n_iter", [10, 60, 200])
def test_fit_lm_returns_the_starting_scale_for_every_budget(n_iter):
    """Guarded, ``fit_lm`` must return the scale it was given.

    The data determines ``k/m`` and ``c/m`` and says nothing at all about
    the overall scale, so the only defensible value for it is the caller's.
    """
    gm = _spring()
    residual = _lm_residual(gm, _observations(gm, noisy=True))
    res = fit_lm(gm, residual, params=_with(gm, START), n_iter=n_iter,
                 notify_every=0)
    assert res.excited_rank == 2, res.excited_rank
    assert res.hold_declined is False
    assert res.undetermined_drift is not None and res.undetermined_drift > 1e-3
    assert abs(_scale(res.params) / START_SCALE - 1.0) < 1e-4, _scale(res.params)


def test_fit_lm_holding_costs_neither_loss_nor_the_identifiable_ratios():
    """The guard moves along a flat direction, so ``0.5 ||r||^2`` and the
    two combinations the data does determine must be unchanged."""
    gm = _spring()
    obs = _observations(gm, noisy=True)
    residual = _lm_residual(gm, obs)
    kw = dict(params=_with(gm, START), n_iter=40, notify_every=0)
    held = fit_lm(gm, residual, **kw)
    raw = fit_lm(gm, residual, hold_undetermined=False, **kw)

    def sse(p):
        r = np.asarray(residual(p))
        return 0.5 * float(r @ r)

    assert held.hold_declined is False
    assert sse(held.params) <= sse(raw.params) * 1.001 + 1e-9, (
        sse(held.params), sse(raw.params))
    for num in ("stiffness", "damping"):
        a = (float(held.params["nodes"]["s"][num])
             / float(held.params["nodes"]["s"]["mass"]))
        b = (float(raw.params["nodes"]["s"][num])
             / float(raw.params["nodes"]["s"]["mass"]))
        assert abs(a / b - 1.0) < 1e-3, (num, a, b)
    # The iterates are untouched; only the returned params differ.
    np.testing.assert_array_equal(held.losses, raw.losses)
    assert held.n_iter == raw.n_iter and held.converged == raw.converged


def test_a_well_posed_fit_lm_gets_its_iterate_back_bit_for_bit():
    """With ``mass`` frozen the remaining ``(k, c)`` are identifiable, so
    the guard has nothing to remove and must return the same bits as a run
    without it -- not merely the same value to a tolerance.

    Two trainable parameters, not one: a one-parameter fixture cannot
    express a rank deficiency at all, so it would pass on a guard that had
    been broken into never firing.  ``excited_rank == 2`` is asserted for
    the same reason -- it says the guard ran and found full rank, rather
    than declining and returning the iterate by the other route.
    """
    gm = _spring()
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False, units="kg"))
    residual = _lm_residual(gm, _observations(gm, noisy=True))
    kw = dict(params=_with(gm, START), n_iter=40, notify_every=0)
    held = fit_lm(gm, residual, **kw)
    raw = fit_lm(gm, residual, hold_undetermined=False, **kw)
    assert held.excited_rank == 2, held.excited_rank
    assert held.undetermined_drift == 0.0 and held.hold_declined is False
    for key, value in held.params["nodes"]["s"].items():
        assert float(value) == float(raw.params["nodes"]["s"][key]), key
    assert float(held.params["nodes"]["s"]["mass"]) == 1.0


def test_fit_lm_refuses_a_non_bool_hold_undetermined():
    """Refused before any model evaluation, like the other hyper-parameters."""
    gm = _spring()
    residual = _lm_residual(gm, _observations(gm, noisy=False))
    with pytest.raises(ValueError, match="hold_undetermined must be a bool"):
        fit_lm(gm, residual, params=_with(gm, START), n_iter=2,
               hold_undetermined="yes")         # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ``fit_multiple_shooting``: Adam again, so the budget decides
# ---------------------------------------------------------------------------


def _ms_kwargs(gm, obs):
    return dict(observations=obs, obs_fn=lambda h: h["s"]["position"],
                window=WINDOW, params=_with(gm, START), notify_every=0)


#: ``lr`` / ``n_iter`` pairs whose *unguarded* answers span -4.80% to
#: -1.97% of the starting scale -- a 3.0% spread, which is the thing the
#: guard has to remove.  The same shape of schedule ``fit``'s own
#: reproducibility test uses, and for the same reason: on this fixture the
#: dependence is mostly on ``lr``, so a grid that varied only ``n_iter``
#: would understate it by a factor of ten.
_MS_SCHEDULE = [(0.01, 200), (0.01, 1200), (0.2, 200), (0.2, 1200)]


def test_multiple_shooting_scale_does_not_depend_on_the_schedule():
    """``fit_multiple_shooting`` is Adam, so it drifts for ``fit``'s reason
    and, like ``fit`` on this fixture, the drift depends on the schedule.
    Measured unguarded: -4.35% at ``lr=0.01, n_iter=200``, -4.80% at 1200,
    -1.97% at ``lr=0.2, n_iter=200`` and -2.02% at 1200 -- a 3.0% spread in
    the answer for a loss that agrees to four digits.  Most of it is the
    learning rate; extending the budget to 4,000 moved the last iterate at
    ``lr=0.2`` on to -2.31%, so it had not settled either.  Those are last
    iterates; the lowest-loss iterates the fitter returns since it selects
    them land -4.35%, -4.80%, -1.97% and -1.97%, the same 3.0% spread.

    The ``hold_undetermined=False`` half is what makes this a regression
    test rather than a tautology: it asserts the defect is still there when
    the guard is off, so a guard that silently stopped running could not
    make both halves pass.
    """
    gm = _spring()
    obs = _observations(gm, noisy=True)
    kw = _ms_kwargs(gm, obs)

    held = [fit_multiple_shooting(gm, lr=lr, n_iter=n, **kw)[0]
            for lr, n in _MS_SCHEDULE]
    for res, (lr, n) in zip(held, _MS_SCHEDULE):
        assert res.excited_rank == 2, (lr, n, res.excited_rank)
        assert res.undetermined_drift is not None
        assert res.hold_declined is False, (lr, n)
    scales = [_scale(r.params) for r in held]
    assert max(scales) / min(scales) - 1.0 < 1e-4, dict(
        zip(map(str, _MS_SCHEDULE), scales))
    for s in scales:
        assert abs(s / START_SCALE - 1.0) < 1e-4, s

    # The unguarded half needs only to show the dependence is there, and the
    # two 200-iteration schedules already span -4.35% to -1.97%; the two
    # 1,200-iteration fits it skips were a third of this test's time once
    # every guarded fit also compiled the curvature test.
    unguarded = [s for s in _MS_SCHEDULE if s[1] == 200]
    raw = [_scale(fit_multiple_shooting(gm, lr=lr, n_iter=n,
                                        hold_undetermined=False, **kw)[0].params)
           for lr, n in unguarded]
    assert max(raw) / min(raw) - 1.0 > 1e-2, (
        dict(zip(map(str, unguarded), raw)),
        "the unguarded schedule dependence this test exists for did not "
        "reproduce",
    )


def test_multiple_shooting_holding_the_scale_leaves_the_window_states_alone():
    """The guard covers the parameters only.

    The window starts are decision variables of this fit, not constants a
    caller records as provenance, and a caller warm-starting from them
    needs the values the optimiser actually reached.  They must come back
    identical to the unguarded run, and so must ``losses``.

    "Identical to the unguarded run" is not enough on its own: a
    ``fit_multiple_shooting`` that reset the window states to their seed
    would satisfy it in *both* runs, and this test passed on exactly that
    seeded fault until the non-vacuity assertion below was added.  So the
    states are also required to have moved off ``init_window_states``.
    """
    gm = _spring()
    obs = _observations(gm, noisy=True)
    kw = dict(_ms_kwargs(gm, obs), n_iter=200, lr=0.2)
    held, ws_held = fit_multiple_shooting(gm, **kw)
    raw, ws_raw = fit_multiple_shooting(gm, hold_undetermined=False, **kw)

    assert held.excited_rank == 2 and held.undetermined_drift > 1e-3
    assert held.hold_declined is False
    assert _scale(held.params) != _scale(raw.params)      # the guard did fire
    seed = init_window_states(obs, WINDOW)
    assert any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(ws_held), jax.tree.leaves(seed))
    ), "the window states never left their seed; the comparison below is vacuous"
    held_leaves = jax.tree.leaves_with_path(ws_held)
    raw_leaves = dict(jax.tree.leaves_with_path(ws_raw))
    assert held_leaves, "the window-state tree is empty; this asserts nothing"
    for path, value in held_leaves:
        np.testing.assert_array_equal(
            np.asarray(value), np.asarray(raw_leaves[path]),
            err_msg=jax.tree_util.keystr(path))
    np.testing.assert_array_equal(held.losses, raw.losses)
    assert held.n_iter == raw.n_iter and held.converged == raw.converged


def test_a_well_posed_multiple_shooting_fit_gets_its_iterate_back_bit_for_bit():
    """Two identifiable parameters, so the guard removes nothing and the
    returned bits are those of a run without it."""
    gm = _spring()
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False, units="kg"))
    obs = _observations(gm, noisy=True)
    kw = dict(_ms_kwargs(gm, obs), n_iter=300, lr=0.1)
    held, _ = fit_multiple_shooting(gm, **kw)
    raw, _ = fit_multiple_shooting(gm, hold_undetermined=False, **kw)
    assert held.excited_rank == 2, held.excited_rank
    assert held.undetermined_drift == 0.0 and held.hold_declined is False
    for key, value in held.params["nodes"]["s"].items():
        assert float(value) == float(raw.params["nodes"]["s"][key]), key


def test_multiple_shooting_refuses_a_non_bool_hold_undetermined():
    gm = _spring()
    obs = _observations(gm, noisy=False)
    with pytest.raises(ValueError, match="hold_undetermined must be a bool"):
        fit_multiple_shooting(gm, n_iter=2, hold_undetermined="yes",
                              **_ms_kwargs(gm, obs))   # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The cutoff, from both sides
# ---------------------------------------------------------------------------

#: ``(max(n, sqrt(T)) * eps)**2`` for ``n = 2`` parameters, ``T = 100``
#: gradients and float32 gradients: ``(10 * 1.1920928955078125e-07)**2``.
#: Written out rather than recomputed from the implementation's formula,
#: so that changing the formula fails these two tests instead of moving
#: them with it.
_CUTOFF_N2_T100_F32 = 1.4210854715202004e-12


def _tracker_with_second_direction(energy_ratio):
    """A 2-D tracker fed 99 gradients along ``e0`` and one along ``e1``
    carrying ``energy_ratio`` of the cutoff's share of the total."""
    tracker = _ExcitationTracker(2, np.float32)
    for _ in range(99):
        tracker.observe(np.array([1.0, 0.0], dtype=np.float32))
    top = 99.0
    amplitude = float(np.sqrt(energy_ratio * _CUTOFF_N2_T100_F32 * top))
    tracker.observe(np.array([0.0, amplitude], dtype=np.float64))
    assert tracker.count == 100
    return tracker


def test_a_direction_just_above_the_cutoff_is_kept():
    rank, projector = _tracker_with_second_direction(1.0 + 1e-6).projector()
    assert rank == 2
    assert projector is None          # full rank: the iterate is not touched


def test_a_direction_just_below_the_cutoff_is_held():
    rank, projector = _tracker_with_second_direction(1.0 - 1e-6).projector()
    assert rank == 1
    assert projector is not None
    # The projector keeps e0 and discards e1, so a displacement with both
    # comes back with only its e0 part.
    kept = projector @ np.array([3.0, 5.0])
    np.testing.assert_allclose(kept, [3.0, 0.0], atol=1e-9)


def test_the_cutoff_is_numerical_and_keeps_a_weakly_excited_direction():
    """A direction at 1e-3 of the top -- where real weak identifiability
    sits -- is kept, not held.  The guard answers "can the data see this at
    all", never "does it see it well enough"; that second question is
    ``FIMReport.crb``'s, and answering it here would discard real
    information under the name of a rounding threshold."""
    rank, projector = _tracker_with_second_direction(
        1e-3 / _CUTOFF_N2_T100_F32).projector()
    assert rank == 2 and projector is None


# ---------------------------------------------------------------------------
# Under jax_enable_x64, and with float32 constants in an x64 graph
# ---------------------------------------------------------------------------
#
# x64 is a supported mode -- ``fim`` recommends it for an ill-conditioned
# problem -- and the guard used to miss the scale degeneracy there in most
# runs (audit_040_p4_8/fmu-sysid/repro_hold_x64.py): the tracker decomposed
# ``G = sum g g^T`` with ``eigh``, which resolves ``G``'s eigenvalues only to
# about ``eps64`` of the largest, while an exactly null direction of float64
# gradients carries ``eps64**2`` of it.  It came back at ``5e-17`` against a
# cutoff of ``4e-31``: ``excited_rank=3``, nothing held, the scale 1-2% off
# its start.  A float32 constant in an x64 graph failed in every run
# (repro_hold_x64_mixed_dtypes.py), because the tracker and the curvature
# tests took ``eps`` from the raveled vector -- float64 -- while the
# gradient of a float32 leaf carries float32 rounding.
#
# The graph's scans refuse a float32 seed state under x64 (MADD-ANO-017),
# so these records start from the seed cast to float64 and are made with
# the step function directly, which leaves the graph's state alone.


@contextlib.contextmanager
def _x64():
    """Run the body in double precision, restoring the global setting."""
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


#: ``"float64"``: every constant at x64's canonical precision.  ``"mixed"``:
#: ``(k, c, m)`` given as float32 values, which keep their dtype in an x64
#: graph while ``rest_length`` and the state are float64.
_PRECISIONS = ["float64", "mixed"]

#: How close to its start a held scale must land: rounding of the held
#: point in the trainable leaves' own precision, with room.
_HELD = {"float64": 1e-12, "mixed": 1e-5}


def _spring_x64(leaves, freeze_mass=False):
    """``_spring()`` built under x64 (call inside ``_x64()``)."""
    constants = (TRUTH if leaves == "float64"
                 else {k: np.float32(v) for k, v in TRUTH.items()})
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, rest_length=1.0, initial_position=0.5,
                                 **constants))
    gm.compile()
    gm.set_param_spec("s", "rest_length", ParamSpec(trainable=False, units="m"))
    gm.set_param_spec("s", "damping", ParamSpec(
        bounds=(0.0, None), transform="log", units="N*s/m"))
    if freeze_mass:
        gm.set_param_spec("s", "mass", ParamSpec(trainable=False, units="kg"))
    want = jnp.float64 if leaves == "float64" else jnp.float32
    for key in TRUTH:
        assert jnp.result_type(gm.params["nodes"]["s"][key]) == want, key
    return gm


def _with_x64(gm, values):
    """``_with`` keeping each leaf's own dtype."""
    p = jax.tree.map(lambda x: x, gm.params)
    for k, v in values.items():
        p["nodes"]["s"][k] = jnp.asarray(v, dtype=jnp.result_type(p["nodes"]["s"][k]))
    return p


def _observations_x64(gm, noisy):
    step_fn = gm._build_step_fn()                     # noqa: SLF001
    ext = gm._default_external_inputs()               # noqa: SLF001
    init = {n: {k: jnp.asarray(v, jnp.float64) for k, v in gm.get_node_state(n).items()}
            for n in gm.node_names}
    truth = _with_x64(gm, TRUTH)

    def body(state, _):
        state = step_fn(state, ext, truth)
        return state, state

    _, hist = jax.lax.scan(body, init, None, length=N_STEPS)
    obs = observations_from_history(init, hist)
    if noisy:
        rng = np.random.default_rng(20260920)
        draw = rng.normal(0.0, NOISE_STD, size=obs["s"]["position"].shape)
        obs["s"]["position"] = obs["s"]["position"] + jnp.asarray(draw, jnp.float64)
    return obs


def _start_scale_x64(gm):
    s = _with_x64(gm, START)["nodes"]["s"]
    return float(np.cbrt(float(s["stiffness"]) * float(s["damping"]) * float(s["mass"])))


@pytest.mark.parametrize("leaves", _PRECISIONS)
def test_fit_holds_the_scale_under_x64(leaves):
    """The schedules the audit found unheld under x64 (``lr=0.01`` at 10
    and 200 iterations, ``lr=0.2`` at every budget) and the mixed graph's.
    The ``hold_undetermined=False`` half shows the drift the guard removes
    is there, so a guard that stopped running could not pass."""
    with _x64():
        gm = _spring_x64(leaves)
        loss = _loss(gm, _observations_x64(gm, noisy=True))
        start, s0 = _with_x64(gm, START), _start_scale_x64(gm)
        drift = []
        for lr, n_iter in [(0.01, 10), (0.2, 50), (0.2, 200)]:
            res = fit(gm, loss, params=start, n_iter=n_iter, lr=lr, notify_every=0)
            assert res.excited_rank == 2, (leaves, lr, n_iter, res.excited_rank)
            assert res.hold_declined is False, (leaves, lr, n_iter)
            assert abs(_scale(res.params) / s0 - 1.0) < _HELD[leaves], (
                leaves, lr, n_iter, _scale(res.params) / s0 - 1.0)
            raw = fit(gm, loss, params=start, n_iter=n_iter, lr=lr, notify_every=0,
                      hold_undetermined=False)
            drift.append(abs(_scale(raw.params) / s0 - 1.0))
        assert max(drift) > 1e-3, (leaves, drift, "no drift for the guard to remove")


@pytest.mark.parametrize("leaves", _PRECISIONS)
@pytest.mark.parametrize("noisy", [False, True])
def test_fit_lm_holds_the_scale_under_x64(leaves, noisy):
    """Noiselessly the audit's x64 ``fit_lm`` reached loss ``4e-30`` with
    ``excited_rank=3`` and the scale 0.98% off its start."""
    with _x64():
        gm = _spring_x64(leaves)
        residual = _lm_residual(gm, _observations_x64(gm, noisy=noisy))
        s0 = _start_scale_x64(gm)
        for n_iter in (10, 60):
            res = fit_lm(gm, residual, params=_with_x64(gm, START), n_iter=n_iter,
                         notify_every=0)
            assert res.excited_rank == 2, (leaves, noisy, n_iter, res.excited_rank)
            assert res.hold_declined is False, (leaves, noisy, n_iter)
            assert res.undetermined_drift is not None and res.undetermined_drift > 1e-3
            assert abs(_scale(res.params) / s0 - 1.0) < _HELD[leaves], (
                leaves, noisy, n_iter, _scale(res.params) / s0 - 1.0)


@pytest.mark.parametrize("leaves", _PRECISIONS)
def test_multiple_shooting_holds_the_scale_under_x64(leaves):
    with _x64():
        gm = _spring_x64(leaves)
        obs = _observations_x64(gm, noisy=True)
        s0 = _start_scale_x64(gm)
        kw = dict(observations=obs, obs_fn=lambda h: h["s"]["position"],
                  window=WINDOW, params=_with_x64(gm, START), notify_every=0,
                  n_iter=200, lr=0.2)
        held, _ = fit_multiple_shooting(gm, **kw)
        assert held.excited_rank == 2 and held.hold_declined is False, held
        assert held.undetermined_drift > 1e-3, held.undetermined_drift
        assert abs(_scale(held.params) / s0 - 1.0) < _HELD[leaves], (
            leaves, _scale(held.params) / s0 - 1.0)


@pytest.mark.parametrize("leaves", _PRECISIONS)
def test_a_well_posed_fit_under_x64_gets_its_iterate_back_bit_for_bit(leaves):
    """The other side of the tracker change: with ``mass`` frozen nothing
    is undetermined, so a guard resolving float64 spectra more finely must
    still find full rank and return the unguarded bits -- for ``fit`` and
    ``fit_lm``."""
    with _x64():
        gm = _spring_x64(leaves, freeze_mass=True)
        obs = _observations_x64(gm, noisy=True)
        start = _with_x64(gm, START)
        for runner, objective, kw in (
                (fit, _loss(gm, obs), dict(n_iter=300, lr=0.1)),
                (fit_lm, _lm_residual(gm, obs), dict(n_iter=40))):
            held = runner(gm, objective, params=start, notify_every=0, **kw)
            raw = runner(gm, objective, params=start, notify_every=0,
                         hold_undetermined=False, **kw)
            assert held.excited_rank == 2, (runner.__name__, held.excited_rank)
            assert held.undetermined_drift == 0.0 and held.hold_declined is False
            for key, value in held.params["nodes"]["s"].items():
                assert float(value) == float(raw.params["nodes"]["s"][key]), (
                    runner.__name__, key)


def test_fit_lm_holds_a_float32_residuals_degeneracy_in_an_x64_process():
    """The curvature test reads its precision from the leaves too.  A
    residual simulated in float32 from float32 constants has a Jacobian whose
    null combination is float32 rounding -- ``||J v|| = 8e-8`` of ``||J||``
    here, the force's ``k * x`` divided by ``m`` -- so by the float64 cutoff
    of the promoted vector it is not flat and the degeneracy went unheld;
    by float32's it is.  (The graph's own scans need a float64 state under
    x64, MADD-ANO-017, which puts the spring's arithmetic in float64; this
    residual is the spring's update, written out in float32.)"""
    def simulate(k, c, m):
        dt = jnp.float32(DT)

        def body(state, _):
            x, v = state
            a = (-k * (x - jnp.float32(1.0)) - c * v) / m
            v = v + dt * a
            x = x + dt * v
            return (x, v), x

        _, xs = jax.lax.scan(body, (jnp.float32(0.5), jnp.float32(0.0)), None,
                             length=60)
        return xs

    with _x64():
        gm = _spring_x64("mixed")
        data = simulate(*(jnp.float32(TRUTH[k]) for k in ("stiffness", "damping", "mass")))

        def residual(p):
            s = p["nodes"]["s"]
            return simulate(s["stiffness"], s["damping"], s["mass"]) - data

        assert residual(gm.params).dtype == jnp.float32
        s0 = _start_scale_x64(gm)
        res = fit_lm(gm, residual, params=_with_x64(gm, START), n_iter=30,
                     notify_every=0)
        raw = fit_lm(gm, residual, params=_with_x64(gm, START), n_iter=30,
                     notify_every=0, hold_undetermined=False)
    assert res.excited_rank == 2 and res.hold_declined is False, res
    assert abs(_scale(res.params) / s0 - 1.0) < _HELD["mixed"], _scale(res.params) / s0 - 1.0
    # Unguarded it drifts 2.3e-4 (5.2e-3 before a ``log`` coordinate's step was
    # read on its tangent): twenty times what the guard leaves.
    assert abs(_scale(raw.params) / s0 - 1.0) > 1e-4, "no drift for the guard to remove"


def _orthogonal_gradients(dtype, n_grad=20):
    """``n_grad`` gradients exactly orthogonal to ``(1, 1, 1)`` in exact
    arithmetic, rounded to ``dtype`` (which leaves ``~eps`` along it)."""
    rng = np.random.default_rng(0)
    null = np.ones(3) / np.sqrt(3.0)
    out = []
    for _ in range(n_grad):
        g = rng.normal(size=3)
        out.append((g - null * (null @ g)).astype(dtype))
    return out


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_tracker_resolves_an_exact_null_direction_at_either_precision(dtype):
    """The mechanism on the tracker alone.  The ``eigh`` of ``G`` put the
    null energy of float64 gradients at ``5e-17`` of the largest against a
    cutoff of ``1e-30`` and reported 3 of 3; the factor's singular values
    put it at about one ``eps`` of the largest singular value, under the
    cutoff of ``max(n, sqrt(T)) * eps``, at both precisions."""
    tracker = _ExcitationTracker(3, np.dtype(dtype))
    for g in _orthogonal_gradients(dtype):
        tracker.observe(g)
    split = tracker.split()
    assert split is not None
    evecs, excited = split
    assert int(excited.sum()) == 2, excited
    held = evecs[:, ~excited][:, 0]
    assert abs(float(held @ (np.ones(3) / np.sqrt(3.0)))) > 1 - 1e-9, held


def test_the_tracker_folds_a_long_run_without_moving_its_verdict():
    """Past the fold buffer the gradients are folded into the triangular
    factor by QR; the verdict and the spanned subspace must be those of the
    whole stack, decomposed at once."""
    grads = _orthogonal_gradients(np.float64, n_grad=1000)
    tracker = _ExcitationTracker(3, np.dtype(np.float64))
    for g in grads:
        tracker.observe(g)
    evecs, excited = tracker.split()
    assert int(excited.sum()) == 2
    stack = np.stack(grads)
    np.testing.assert_allclose(tracker._gram, stack.T @ stack, rtol=1e-12,  # noqa: SLF001
                               atol=1e-12 * float(np.abs(stack.T @ stack).max()))
    np.testing.assert_allclose(tracker.gradient_scale(),
                               np.linalg.norm(stack, axis=0), rtol=1e-12)


def test_the_guard_reads_its_precision_from_the_coarsest_trainable_leaf():
    """``ravel_pytree`` promotes a float32 leaf beside a float64 one, but the
    gradient of the float32 leaf carries float32 rounding: the guard's
    ``eps`` is the coarsest *trainable* leaf's, and a float32 leaf outside
    the trainable set does not coarsen it.  (Under x64: without it there is
    no float64 leaf to be finer than float32.)"""
    with _x64():
        tree = {"a": np.zeros(2, np.float64), "b": np.float32(1.0), "c": np.float64(2.0)}
        coarsest = sysid._coarsest_dtype                      # noqa: SLF001
        assert coarsest(tree, np.arange(4)) == np.float32
        assert coarsest(tree, np.array([0, 1, 3])) == np.float64
        assert coarsest(tree, np.array([2])) == np.float32
