"""``hold_undetermined`` never returns parameters above the fit's own loss.

The guard (on by default in ``fit``, ``fit_lm`` and
``fit_multiple_shooting``) exists to stop an optimiser wandering along a
direction the data cannot determine -- the spring's ``(k, c, m)`` common
scale, along which Adam drifts by an amount the learning rate and the
budget choose.  Earlier 0.4.0 development builds held every direction the
run's *gradients* had not spanned.  That is necessary for a direction the
data cannot see, and not sufficient: a short or fast-converging run's
gradients need not span the data-determined space, and the guard then
took real progress back, silently:

* ``fit_lm`` on a closed-form four-parameter bowl, every direction
  determined, reached a loss of 0.0; the guard reported ``excited_rank``
  1 of 4 and returned parameters at 0.22 (masked to two: 0.279 to 0.372);
* ``fit_lm`` on the spring, ``(k, c, m, rest_length)`` from stiffness 1%
  off: 0.0 became 1.2e-4;
* ten Adam steps on the bowl: ``excited_rank`` 3 of 4, and the result moved
  off the iterate the fit selected.

A direction is now held only if it passes two tests -- no gradient of the
run pointed along it, *and* the objective has no curvature along it at the
returned point -- and then only if the held point's loss is within a
stated tolerance of the selected iterate's.  This file pins the
regressions, the loss check and its tolerance, the curvature cutoffs from
both sides, and the paths that check newly decides: a hold through a
bound, a loss JAX cannot differentiate twice, and the wiring each fitter
gives the shared machinery.  That the guard still holds the spring's
scale is ``test_sysid_undetermined_directions.py``.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening import sysid
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import (
    _ExcitationTracker,
    _SelectedObjective,
    _gauss_newton_flatness,
    _hessian_flatness,
    _hold_undetermined_directions,
    fit,
    fit_lm,
    fit_multiple_shooting,
    observations_from_history,
    windowed_loss,
)

DT = 0.01
TRUTH = {"stiffness": 30.0, "damping": 2.0, "mass": 1.0, "rest_length": 1.0}
#: Minimum of :func:`_bowl`: every one of the spring's four constants is
#: determined, in the coordinate its shipped spec optimises in.
TARGET = {"stiffness": 20.0, "damping": 1.5, "mass": 1.3, "rest_length": 1.2}
BOWL_START = {"stiffness": 33.0, "damping": 2.2, "rest_length": 1.05}


def _spring(**overrides):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, initial_position=0.5,
                                 **{**TRUTH, **overrides}))
    gm.compile()
    return gm


def _with(gm, values):
    p = jax.tree.map(lambda x: x, gm.params)
    for key, value in values.items():
        p["nodes"]["s"][key] = jnp.asarray(value, dtype=jnp.float32)
    return p


def _only(gm, *keys):
    mask = jax.tree.map(lambda _: False, gm.params)
    for key in keys:
        mask["nodes"]["s"][key] = True
    return mask


def _bowl_residual(p):
    s = p["nodes"]["s"]
    return jnp.stack([
        jnp.log(s["stiffness"] / TARGET["stiffness"]),
        s["damping"] - TARGET["damping"],
        jnp.log(s["mass"] / TARGET["mass"]),
        2.0 * (s["rest_length"] - TARGET["rest_length"]),
    ])


def _bowl(p):
    r = _bowl_residual(p)
    return jnp.sum(r * r)


def _half_sse(residual_fn, params):
    r = np.asarray(residual_fn(params), dtype=np.float64).ravel()
    return 0.5 * float(r @ r)


def _bits(tree):
    return [np.asarray(x).tobytes() for x in jax.tree.leaves(tree)]


def _trajectory_residual(gm, n_steps):
    """Position residual of a rollout from ``gm``'s own state against the
    rollout at ``gm.params`` -- noiseless, so the minimum is exactly 0."""
    step_fn = gm._build_step_fn()                     # noqa: SLF001
    ext = gm._default_external_inputs()               # noqa: SLF001
    init = {n: gm.get_node_state(n) for n in gm.node_names}

    def roll(p):
        def body(state, _):
            state = step_fn(state, ext, p)
            return state, state["s"]["position"]
        _, pos = jax.lax.scan(body, init, None, length=n_steps)
        return pos

    truth = roll(gm.params)
    return jax.jit(lambda p: roll(p) - truth)


# ---------------------------------------------------------------------------
# The defect: a well-posed fit returned above the loss it had reached
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("masked", [False, True])
def test_fit_lm_on_a_well_posed_bowl_returns_the_minimum_it_reached(masked):
    """0.4.0-dev returned 0.22 from a fit at 0.0 (masked: 0.372 from
    0.279).  The bowl determines every direction, so the curvature test
    rejects every candidate the gradient test raised: full rank, nothing
    held, the selected iterate bit for bit."""
    gm = _spring()
    mask = _only(gm, "stiffness", "rest_length") if masked else None
    kw = dict(params=_with(gm, BOWL_START), mask=mask, n_iter=6)
    res = fit_lm(gm, _bowl_residual, **kw)
    raw = fit_lm(gm, _bowl_residual, hold_undetermined=False, **kw)
    # Recomputed in float64 from the physical parameters, so a few float32
    # ulps from ``best_loss``; 0.4.0-dev was 33% (masked) or 0.22 above it.
    assert _half_sse(_bowl_residual, res.params) <= res.best_loss * (1 + 1e-6) + 1e-9, (
        _half_sse(_bowl_residual, res.params), res.best_loss)
    assert res.excited_rank == (2 if masked else 4)
    assert res.undetermined_drift == 0.0 and res.hold_declined is False
    assert _bits(res.params) == _bits(raw.params)


def test_fit_lm_on_the_spring_one_percent_off_stays_at_its_minimum():
    """0.4.0-dev: loss 0.0, ``excited_rank`` 2 of 4, returned parameters at
    1.2e-4.  With the shipped specs the spring's scale direction rotates
    (``damping`` is the identity), so whether the local scale direction
    passes both tests here depends on how far ``c`` moved; measured, it
    does, and the hold costs 4.5e-11, inside the quantisation term of the
    tolerance.  Either way the loss stays where the fit put it."""
    gm = _spring()
    residual = _trajectory_residual(gm, 200)
    res = fit_lm(gm, residual, params=_with(gm, {"stiffness": 1.01 * 30.0}),
                 n_iter=50)
    assert res.best_loss == 0.0
    assert _half_sse(residual, res.params) <= 1e-9, _half_sse(residual, res.params)
    assert res.excited_rank in (3, 4) and res.hold_declined is False


def test_a_short_adam_run_on_the_bowl_returns_its_selected_iterate():
    """Ten Adam steps: 0.4.0-dev's gradients spanned 3 of 4 directions and
    the guard moved the result off the selected iterate, raising the loss
    from 0.32342 to 0.32367.  The bowl's Hessian is full rank, so nothing
    is held."""
    gm = _spring()
    res = fit(gm, _bowl, n_iter=10, lr=0.01)
    raw = fit(gm, _bowl, n_iter=10, lr=0.01, hold_undetermined=False)
    assert res.excited_rank == 4 and res.hold_declined is False
    assert res.undetermined_drift == 0.0
    assert _bits(res.params) == _bits(raw.params)


# ---------------------------------------------------------------------------
# The loss check: a hold both tests allow is still refused if it costs loss
# ---------------------------------------------------------------------------
#
# ``log k`` and ``log m`` are trainable and the loss reads only
# ``d = log k - 2 log m``: the direction ``v = (2, 1)/sqrt(5)`` is exactly
# flat, no gradient ever points along it, and Adam's diagonal preconditioner
# and LM's ``lam * diag(A)`` both drift along it.  The ``bump`` term reads
# ``s = 2 log k + log m`` through ``stop_gradient``: it adds no gradient and
# no curvature, so both tests still call ``v`` flat, but its value is 1 at
# the start's ``s`` and vanishes a few widths away -- exactly where the
# hold would put ``s`` back.  It is the case the loss check exists for: a
# direction flat where the fit ended and not on the way back.

_SHIFT = 0.5
_WIDTH = 1e-3


def _d(p):
    s = p["nodes"]["s"]
    return jnp.log(s["stiffness"]) - 2.0 * jnp.log(s["mass"])


def _s(p):
    s = p["nodes"]["s"]
    return 2.0 * jnp.log(s["stiffness"]) + jnp.log(s["mass"])


def _flat_problem(fitter, bumped):
    gm = _spring()
    d0, s0 = float(_d(gm.params)), float(_s(gm.params))

    def bump(p):
        return jax.lax.stop_gradient(jnp.exp(-((_s(p) - s0) / _WIDTH) ** 2))

    def residual(p):
        rows = [_d(p) - (d0 - _SHIFT)]
        if bumped:
            rows.append(bump(p))
        return jnp.stack(rows)

    def loss(p):
        r = residual(p)
        return jnp.sum(r * r)

    mask = _only(gm, "stiffness", "mass")
    if fitter == "fit":
        def call(hold):
            return fit(gm, loss, mask=mask, n_iter=60, lr=0.05,
                       hold_undetermined=hold)
    else:
        def call(hold):
            return fit_lm(gm, residual, mask=mask, n_iter=10,
                          hold_undetermined=hold)
    return gm, s0, call


@pytest.mark.parametrize("fitter", ["fit", "fit_lm"])
def test_a_hold_that_would_raise_the_loss_is_declined_and_said_so(fitter):
    _, s0, call = _flat_problem(fitter, bumped=True)
    with pytest.warns(RuntimeWarning, match="would raise the loss"):
        res = call(True)
    raw = call(False)
    assert res.hold_declined is True
    assert res.excited_rank == 1                      # found, and reported
    assert res.undetermined_drift > 1e-2              # there was drift to hold
    assert abs(float(_s(res.params)) - s0) > 10 * _WIDTH
    # Declined means untouched: the selected iterate, bit for bit.
    assert _bits(res.params) == _bits(raw.params)
    assert (res.best_iteration, res.best_loss) == (raw.best_iteration, raw.best_loss)


@pytest.mark.parametrize("fitter", ["fit", "fit_lm"])
def test_a_flat_hold_is_made_and_keeps_the_determined_combination(fitter):
    """The same problem without the bump: the hold costs nothing, so it is
    made -- ``s`` back at the start, ``d`` where the fit put it."""
    _, s0, call = _flat_problem(fitter, bumped=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = call(True)
    raw = call(False)
    assert res.hold_declined is False and res.excited_rank == 1
    assert abs(float(_s(raw.params)) - s0) > 1e-2     # unguarded, it drifted
    assert abs(float(_s(res.params)) - s0) < 1e-5
    assert abs(float(_d(res.params)) - float(_d(raw.params))) < 1e-5


def test_a_hold_through_a_bound_is_declined():
    """``damping`` is the identity clipped at 0.  The fit reads only
    ``d = log k - 2c``; it ends with ``c`` near 0.05, and holding the flat
    direction would put ``c`` at -0.04, which ``constrain`` clips to 0 --
    moving ``d``, which the data determines.  The loss check sees the
    clipped point, because it evaluates the same ``constrain`` the
    returned parameters go through, and refuses.  0.4.0-dev returned the
    clipped point."""
    gm = _spring(damping=0.2)
    d0 = float(jnp.log(gm.params["nodes"]["s"]["stiffness"]) - 0.4)

    def residual(p):
        s = p["nodes"]["s"]
        return jnp.stack([jnp.log(s["stiffness"]) - 2.0 * s["damping"] - (d0 + 0.6)])

    kw = dict(mask=_only(gm, "stiffness", "damping"), n_iter=10)
    with pytest.warns(RuntimeWarning, match="would raise the loss"):
        res = fit_lm(gm, residual, **kw)
    raw = fit_lm(gm, residual, hold_undetermined=False, **kw)
    assert float(raw.params["nodes"]["s"]["damping"]) > 0.0
    assert res.hold_declined is True and res.excited_rank == 1
    assert _bits(res.params) == _bits(raw.params)
    gm.check_params(res.params)


# ---------------------------------------------------------------------------
# The tolerance, from both sides
# ---------------------------------------------------------------------------

#: ``2**10 * eps32 * |1.0| + 0.5 * 2.0 * (16 * |spacing([1, 1])|)**2`` --
#: ``2**-13 + 2**-37``.  Written out rather than recomputed from the
#: implementation's formula, so that changing the formula fails these
#: tests instead of moving them with it.
_TOL_L1_SCALE2_F32 = 0.00012207031977595761


def _helper_case(loss_held, *, flat=True):
    """The shared hold on a 2-D problem with known numbers: gradients only
    along ``e0``, so ``e1`` is the one candidate; the selected iterate is
    ``(1, 1.25)`` from a start at ``(1, 1)``, so holding ``e1`` lands
    exactly on the start, whose float32 spacing is ``2**-23`` per
    coordinate; ``loss_sel = 1``, no gradient term, curvature scale 2."""
    tracker = _ExcitationTracker(2, np.float32)
    for _ in range(4):
        tracker.observe(np.array([1.0, 0.0], dtype=np.float32))
    theta0 = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
    theta = jnp.asarray([1.0, 1.25], dtype=jnp.float32)
    objective = _SelectedObjective(
        loss=lambda th: loss_held,
        reference=lambda: (1.0, None),
        flatness=lambda candidates, spanned: (np.eye(1), np.array([flat]), 2.0),
    )
    return theta, theta0, _hold_undetermined_directions(
        tracker, theta, theta0, objective, "test")


def test_a_rise_just_inside_the_tolerance_is_held():
    _, theta0, (out, rank, drift, declined) = _helper_case(
        1.0 + _TOL_L1_SCALE2_F32 * (1.0 - 1e-6))
    assert declined is False and rank == 1 and drift == 0.25
    np.testing.assert_array_equal(np.asarray(out), np.asarray(theta0))


def test_a_rise_just_outside_the_tolerance_is_declined():
    with pytest.warns(RuntimeWarning, match="would raise the loss"):
        theta, _, (out, rank, drift, declined) = _helper_case(
            1.0 + _TOL_L1_SCALE2_F32 * (1.0 + 1e-6))
    assert declined is True and rank == 1 and drift == 0.25
    assert np.asarray(out).tobytes() == np.asarray(theta).tobytes()


def test_a_non_finite_held_loss_is_declined():
    with pytest.warns(RuntimeWarning, match="would raise the loss"):
        theta, _, (out, _, _, declined) = _helper_case(float("nan"))
    assert declined is True
    assert np.asarray(out).tobytes() == np.asarray(theta).tobytes()


def test_a_candidate_with_curvature_is_neither_held_nor_counted():
    """The curvature test rejecting the only candidate: full rank, no
    drift reported, the loss never consulted -- the bowl's case."""
    calls = []
    tracker = _ExcitationTracker(2, np.float32)
    for _ in range(4):
        tracker.observe(np.array([1.0, 0.0], dtype=np.float32))
    theta = jnp.asarray([1.0, 1.25], dtype=jnp.float32)
    objective = _SelectedObjective(
        loss=lambda th: calls.append("loss") or 0.0,
        reference=lambda: calls.append("reference") or (0.0, None),
        flatness=lambda c, s: (np.eye(1), np.array([False]), 2.0),
    )
    out, rank, drift, declined = _hold_undetermined_directions(
        tracker, theta, jnp.asarray([1.0, 1.0], jnp.float32), objective, "test")
    assert (rank, drift, declined) == (2, 0.0, False)
    assert out is theta and calls == []


# ---------------------------------------------------------------------------
# The curvature cutoffs, from both sides
# ---------------------------------------------------------------------------

#: ``sqrt(eps32)``, the Hessian test's relative cutoff.
_SQRT_EPS32 = 0.00034526698300124393
#: ``max(n, sqrt(m)) * eps32`` at ``n = m = 2``: ``fim``'s rank cutoff.
_FIM_RTOL_N2_M2_F32 = 2.384185791015625e-07


@pytest.mark.parametrize("factor, flat", [(1.0 - 1e-6, True), (1.0 + 1e-6, False)])
def test_the_hessian_cutoff(factor, flat):
    H = np.diag([1.0, _SQRT_EPS32 * factor])
    e0, e1 = np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])
    W, flags, scale = _hessian_flatness(lambda V: H @ V, e1, e0, np.float32, "test")
    assert scale == 1.0 and bool(flags[0]) is flat


def test_the_hessian_test_reads_hu_not_the_quadratic_form():
    """Off a minimum the Hessian need not be positive: here ``e1ᵀ H e1 = 0``
    but ``H e1 = (0.5, 0)``.  A quadratic-form test would call ``e1`` flat
    and hold a direction the loss bends along."""
    H = np.array([[1.0, 0.5], [0.5, 0.0]])
    e0, e1 = np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])
    _, flags, _ = _hessian_flatness(lambda V: H @ V, e1, e0, np.float32, "test")
    assert not bool(flags[0])


@pytest.mark.parametrize("factor, flat", [(1.0 - 1e-6, True), (1.0 + 1e-6, False)])
def test_the_gauss_newton_cutoff_is_fims_rank_rule(factor, flat):
    J = np.diag([1.0, np.sqrt(_FIM_RTOL_N2_M2_F32 * factor)])
    W, flags, scale = _gauss_newton_flatness(J, np.array([[0.0], [1.0]]), np.float32)
    assert scale == 1.0 and bool(flags[0]) is flat


# ---------------------------------------------------------------------------
# A loss JAX cannot differentiate twice: the gradient test stands alone
# ---------------------------------------------------------------------------


@jax.custom_vjp
def _opaque(x):
    return x


def _opaque_fwd(x):
    return x, None


def _opaque_bwd(_, g):
    # A host callback in the backward pass: reverse mode is fine, but the
    # Hessian-vector product differentiates this rule forward, and a
    # ``pure_callback`` has no forward-mode rule.
    return (jax.pure_callback(lambda a: np.asarray(a), jax.ShapeDtypeStruct(g.shape, g.dtype), g),)


_opaque.defvjp(_opaque_fwd, _opaque_bwd)


def test_without_a_hessian_the_flat_hold_is_still_checked_against_the_loss():
    gm = _spring()
    d0, s0 = float(_d(gm.params)), float(_s(gm.params))

    def loss(p):
        return (_opaque(_d(p)) - (d0 - _SHIFT)) ** 2

    mask = _only(gm, "stiffness", "mass")
    with pytest.warns(RuntimeWarning, match="Hessian-vector product"):
        res = fit(gm, loss, mask=mask, n_iter=60, lr=0.05)
    assert res.hold_declined is False and res.excited_rank == 1
    assert abs(float(_s(res.params)) - s0) < 1e-5


# ---------------------------------------------------------------------------
# Wiring: each fitter hands the shared hold its own objective
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def degenerate():
    """``(k, c, m)`` trainable, all three under ``log``, noisy observations:
    the scale direction is exactly flat and every fitter's gradient test
    leaves it as a candidate, so each one's objective is consulted."""
    gm = _spring()
    gm.set_param_spec("s", "rest_length", ParamSpec(trainable=False))
    gm.set_param_spec("s", "damping", ParamSpec(bounds=(0.0, None), transform="log"))
    init = {n: gm.get_node_state(n) for n in gm.node_names}
    _, hist = gm.run_scan_with_history(60)
    obs = observations_from_history(init, hist)
    rng = np.random.default_rng(20261001)
    obs["s"]["position"] = obs["s"]["position"] + jnp.asarray(
        rng.normal(0.0, 0.02, obs["s"]["position"].shape), jnp.float32)
    return gm, obs


def _spy(monkeypatch, name):
    """Record what a fitter hands one of the guard's helpers, then call the
    real helper."""
    seen = {}
    real = getattr(sysid, name)

    def spy(*args, **kwargs):
        seen.setdefault("calls", []).append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(sysid, name, spy)
    return seen


@pytest.mark.parametrize("fitter", ["fit", "fit_lm", "fit_multiple_shooting"])
def test_each_fitter_hands_the_hold_its_selected_point(degenerate, monkeypatch, fitter):
    """Each fitter's loss, gradient and curvature, as the guard receives
    them, are those of its own objective at the iterate it selected --
    checked against an independent ``jax.grad`` / ``jax.hessian`` /
    ``jax.jacfwd`` there.  For ``fit_multiple_shooting`` that is the joint
    objective in ``theta`` at the *selected* window states.  The scale
    degeneracy is flat everywhere, so a curvature test taken at the start,
    or at the seed window states, would hold it all the same and only this
    comparison would notice."""
    from jax.flatten_util import ravel_pytree

    gm, obs = degenerate
    hold = _spy(monkeypatch, "_hold_undetermined_directions")
    curvature = _spy(monkeypatch, "_gauss_newton_flatness" if fitter == "fit_lm"
                     else "_hessian_flatness")
    obs_fn = lambda h: h["s"]["position"]           # noqa: E731
    start = _with(gm, {"stiffness": 45.0, "damping": 3.0})
    flat_u, unravel = ravel_pytree(gm.unconstrain(start))
    idx = sysid._masked_indices(start, sysid._resolve_mask(gm, start, None))  # noqa: SLF001

    def physical(t):
        return gm.constrain(unravel(flat_u.at[idx].set(t)))

    if fitter == "fit":
        loss = jax.jit(lambda p: windowed_loss(gm, p, obs, obs_fn=obs_fn, window=10))
        res = fit(gm, loss, params=start, n_iter=300, lr=0.2, notify_every=0)
        objective = lambda t: loss(physical(t))          # noqa: E731
    elif fitter == "fit_lm":
        residual = _trajectory_residual(gm, 60)
        res = fit_lm(gm, residual, params=start, n_iter=20, notify_every=0)
        objective = lambda t: 0.5 * jnp.sum(residual(physical(t)) ** 2)   # noqa: E731
    else:
        res, ws = fit_multiple_shooting(gm, obs, obs_fn=obs_fn, window=10,
                                        params=start, n_iter=200, lr=0.2,
                                        notify_every=0)
        objective = lambda t: windowed_loss(           # noqa: E731
            gm, physical(t), obs, obs_fn=obs_fn, window=10, window_states=ws,
            continuity_weight=1.0)

    assert res.excited_rank == 2 and res.hold_declined is False
    (_, theta, _, selected_objective, method), = hold["calls"]
    assert method == fitter
    loss_sel, grad_sel = selected_objective.reference()
    if fitter == "fit_lm":
        assert loss_sel == pytest.approx(res.best_loss, rel=1e-5, abs=1e-12)
    else:
        assert loss_sel == res.best_loss            # same compiled function, same point
    assert selected_objective.loss(theta) == loss_sel
    scale = np.abs(np.asarray(jax.grad(objective)(theta))).max() + 1e-12
    np.testing.assert_allclose(grad_sel, np.asarray(jax.grad(objective)(theta)),
                               rtol=1e-3, atol=1e-3 * scale)
    (args,) = curvature["calls"]
    if fitter == "fit_lm":
        J = np.asarray(args[0], dtype=np.float64)
        J_ref = np.asarray(jax.jacfwd(lambda t: residual(physical(t)))(theta), np.float64)
        np.testing.assert_allclose(J, J_ref, rtol=1e-4, atol=1e-6 * np.abs(J_ref).max())
    else:
        HV = args[0](np.eye(3))
        H = np.asarray(jax.hessian(objective)(theta), dtype=np.float64)
        np.testing.assert_allclose(HV, H, rtol=1e-3, atol=1e-5 * np.abs(H).max())
