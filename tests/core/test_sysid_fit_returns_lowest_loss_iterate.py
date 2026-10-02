"""Every fitter returns the lowest-loss iterate it evaluated, not its last.

``fit`` used to return Adam's last iterate.  Adam's step is about ``lr`` in
size whatever the gradient's, so a fit started at -- or reaching -- a point
whose gradient is rounding noise walks away from it, and the run can end
above where it began.  The parameters guide's own ball-and-spring graph,
started at its truth, began at loss 2.7e-8 and returned parameters at
1.9e-4; ``fit_multiple_shooting``, which also steps every window state at
``lr``, went from 1e-13 to 2e-1.  Nothing in the result said so.

The rule, shared by ``fit`` and ``fit_multiple_shooting`` and true of
``fit_lm`` by construction (it accepts only a step that lowers the loss):

* ``params`` is the iterate with the lowest loss the fitter evaluated;
  ``best_iteration`` counts the updates that produced it, and so indexes
  ``losses``, and ``best_loss`` is its loss.
* A tie goes to the later iterate, so a run whose loss never rose returns its
  last iterate bit for bit -- what every fitter returned before.
* The iterate the last update produced, which the loop never evaluated, is
  evaluated once more and takes part.  Nothing else about the run changes:
  ``losses``, ``callback``, the observers and ``excited_rank`` are as they
  were.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import EVENT_FIT_PROGRESS, GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import (
    fit,
    fit_lm,
    fit_multiple_shooting,
    init_window_states,
    observations_from_history,
    windowed_loss,
)

DT = 0.01
N_STEPS, WINDOW = 200, 20
TRUTH = {"stiffness": 30.0, "damping": 2.0, "mass": 1.0, "rest_length": 1.0}

#: The minimum of :func:`_bowl`, a closed-form loss over the spring's four
#: constants: no rollout, so a fit compiles and runs in milliseconds and
#: Adam's trajectory is fully controlled.  Each term is in the coordinate
#: its spec optimises in where that is a transform (``log`` for stiffness
#: and mass), so the bowl is round where Adam steps.
TARGET = {"stiffness": 20.0, "damping": 1.5, "mass": 1.3, "rest_length": 1.2}


def _spring(variant="shipped"):
    """The spring, under the shipped specs or with every constant transformed.

    ``shipped``: ``stiffness`` and ``mass`` are ``log``, ``damping`` is the
    identity clipped to ``(0, None)``, ``rest_length`` the identity with no
    bounds.  ``transformed``: ``damping`` becomes ``log`` and
    ``rest_length`` a ``logit`` on ``(0.5, 2.0)`` -- so between them the two
    variants put every transform and the clip under the selection.
    """
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, initial_position=0.5, **TRUTH))
    gm.compile()
    if variant == "transformed":
        gm.set_param_spec("s", "damping", ParamSpec(bounds=(0.0, None), transform="log"))
        gm.set_param_spec("s", "rest_length", ParamSpec(bounds=(0.5, 2.0), transform="logit"))
    return gm


def _with(gm, values):
    p = jax.tree.map(lambda x: x, gm.params)
    for key, value in values.items():
        p["nodes"]["s"][key] = jnp.asarray(value, dtype=jnp.float32)
    return p


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


def _bits(tree):
    return [np.asarray(x).tobytes() for x in jax.tree.leaves(tree)]


def _recorder():
    """A ``callback`` keeping every ``(i, loss, params)`` it is shown."""
    seen = {}

    def callback(i, loss, params):
        seen[i] = (loss, params)
    return seen, callback


def _straight_through(p):
    """Exactly 1.0 everywhere, with a gradient of 1 in stiffness.

    ``k - stop_gradient(k)`` is exactly zero for every finite ``k`` and has
    derivative 1, so Adam keeps stepping while every loss it records is the
    same bits: the one way to build a plateau the iterates actually cross.
    """
    k = p["nodes"]["s"]["stiffness"]
    return 1.0 + (k - jax.lax.stop_gradient(k))


def _only(gm, *keys):
    mask = jax.tree.map(lambda _: False, gm.params)
    for key in keys:
        mask["nodes"]["s"][key] = True
    return mask


@pytest.fixture(scope="module")
def recorded():
    gm = _spring()
    init = {n: gm.get_node_state(n) for n in gm.node_names}
    _, hist = gm.run_scan_with_history(N_STEPS)
    obs = observations_from_history(init, hist)
    loss = jax.jit(lambda p: windowed_loss(
        gm, p, obs, obs_fn=lambda h: h["s"]["position"], window=WINDOW))
    return gm, obs, loss


# ---------------------------------------------------------------------------
# The defect: a run that ends above an earlier iterate returned the worse one
# ---------------------------------------------------------------------------


def test_a_fit_that_walks_away_from_its_start_returns_its_start(recorded):
    """Started 0.1% off the truth, the gradient is tiny but Adam's step is
    still ~``lr``, so every update moves *away* from the minimum: the last
    iterate's loss is a hundred times the first.  The start is the lowest
    iterate, so it is what comes back -- bit for bit, through the same
    "unmoved leaves are the input" path a fit that takes no step uses."""
    gm, _, loss = recorded
    start = _with(gm, {"stiffness": 1.001 * TRUTH["stiffness"]})
    res = fit(gm, loss, params=start, n_iter=60, lr=0.05)
    assert float(loss(res.params)) <= 2.0 * res.losses[0], (
        float(loss(res.params)), res.losses[0])
    assert res.losses[-1] > 10.0 * res.losses[0], res.losses   # it did walk away
    assert res.best_iteration == 0
    assert res.best_loss == res.losses[0]
    assert _bits(res.params) == _bits(start)


def test_multiple_shooting_that_walks_away_returns_its_start_and_seed_states(recorded):
    """The same for ``fit_multiple_shooting``, which steps every window
    state at ``lr`` as well: the parameters *and* the window states come
    back from the start, together, because they are one point of the joint
    objective."""
    gm, obs, _ = recorded
    start = _with(gm, {"stiffness": 1.001 * TRUTH["stiffness"]})
    res, ws = fit_multiple_shooting(
        gm, obs, obs_fn=lambda h: h["s"]["position"], window=WINDOW,
        params=start, n_iter=60, lr=0.05)
    assert res.losses[-1] > 10.0 * res.losses[0], res.losses
    assert res.best_iteration == 0
    assert res.best_loss == res.losses[0]
    assert _bits(res.params) == _bits(start)
    assert _bits(ws) == _bits(init_window_states(obs, WINDOW))


def test_the_returned_params_are_the_iterate_with_the_lowest_loss():
    """Adam overshoots the bowl and climbs back out: the lowest loss is in
    the middle of the run, below both its start and its end.  The result
    must be *that* iterate -- the exact parameters the callback was shown
    at it -- not the first, not the last, not a blend."""
    gm = _spring()
    seen, callback = _recorder()
    res = fit(gm, _bowl, n_iter=12, lr=0.1, callback=callback)
    losses = res.losses
    k = res.best_iteration
    assert 0 < k < len(losses) - 1, (k, losses)
    assert losses[-1] > 1.1 * losses[k] and losses[0] > 1.1 * losses[k], losses
    assert res.best_loss == losses[k] == losses.min()
    assert res.excited_rank == 4                 # full rank: nothing projected
    assert _bits(res.params) == _bits(seen[k + 1][1])


# ---------------------------------------------------------------------------
# What must not change
# ---------------------------------------------------------------------------


def test_a_run_whose_loss_never_rose_returns_its_last_iterate_bit_for_bit():
    """The bit-identity claim.  A small step down a bowl never overshoots,
    so the iterate the last update produced is the lowest; it is the one
    returned, and it is exactly the iterate a run one iteration longer
    shows its callback next (Adam's iterates do not depend on ``n_iter``).

    ``hold_undetermined=False``: the selection is what is measured here.
    On this very run the guard of earlier 0.4.0 development builds moved
    the result off the selected iterate, uphill;
    ``test_sysid_hold_never_raises_the_loss.py`` pins that it no longer
    does."""
    gm = _spring()
    n = 10
    res = fit(gm, _bowl, n_iter=n, lr=0.01, hold_undetermined=False)
    assert np.all(np.diff(res.losses) < 0), res.losses
    assert res.best_iteration == n == len(res.losses)
    seen, callback = _recorder()
    fit(gm, _bowl, n_iter=n + 1, lr=0.01, callback=callback, hold_undetermined=False)
    loss_next, params_next = seen[n + 1]
    assert res.best_loss == loss_next < res.losses[-1]
    assert _bits(res.params) == _bits(params_next)


def test_a_multiple_shooting_run_whose_loss_never_rose_returns_its_last_iterate(graphs):
    """The same claim for ``fit_multiple_shooting``: small steps on the
    parameters and smaller ones on the window states, so the joint loss
    falls at every iterate, and the result is the iterate the last update
    produced -- which only the extra evaluation can know is the lowest."""
    gm, obs = graphs["shipped"]
    kw = dict(obs_fn=lambda h: h["s"]["position"], window=10,
              params=_with(gm, {"stiffness": 36.0}),
              mask=_only(gm, "stiffness", "damping"), lr=0.01, lr_states=1e-4,
              hold_undetermined=False)
    n = 6
    res, _ = fit_multiple_shooting(gm, obs, n_iter=n, **kw)
    assert np.all(np.diff(res.losses) < 0), res.losses
    assert res.best_iteration == n == len(res.losses)
    seen, callback = _recorder()
    fit_multiple_shooting(gm, obs, n_iter=n + 1, callback=callback, **kw)
    loss_next, params_next = seen[n + 1]
    assert res.best_loss == loss_next < res.losses[-1]
    assert _bits(res.params) == _bits(params_next)


def test_a_tie_goes_to_the_later_iterate():
    """On a plateau the iterates cross, the last of them is returned.  Ties
    to the *earlier* iterate would hand back the start here, and would
    break the bit-identity claim for every run whose loss went flat."""
    gm = _spring()
    mask = _only(gm, "stiffness")
    n = 6
    res = fit(gm, _straight_through, mask=mask, n_iter=n, lr=0.1)
    assert np.all(res.losses == 1.0)
    assert res.best_iteration == n and res.best_loss == 1.0
    seen, callback = _recorder()
    fit(gm, _straight_through, mask=mask, n_iter=n + 1, lr=0.1, callback=callback)
    assert _bits(res.params) == _bits(seen[n + 1][1])
    assert float(res.params["nodes"]["s"]["stiffness"]) < TRUTH["stiffness"]


def test_a_non_finite_final_loss_is_never_returned():
    """The loop raises on a non-finite loss because it cannot step from
    one.  Nothing is stepped from the final iterate, so a non-finite loss
    there does not stop the run -- but that iterate is not returned either,
    and a ``RuntimeWarning`` says which iterate was returned instead."""
    gm = _spring()
    mask = _only(gm, "stiffness")
    n = 5
    seen, callback = _recorder()
    fit(gm, _straight_through, mask=mask, n_iter=n + 1, lr=0.1, callback=callback)
    k_prev = float(seen[n][1]["nodes"]["s"]["stiffness"])      # iterate n - 1
    k_last = float(seen[n + 1][1]["nodes"]["s"]["stiffness"])  # iterate n
    assert k_last < k_prev
    threshold = 0.5 * (k_prev + k_last)

    def poisoned(p):
        k = p["nodes"]["s"]["stiffness"]
        return _straight_through(p) + jnp.where(k < threshold, jnp.nan, 0.0)

    with pytest.warns(RuntimeWarning, match="non-finite loss"):
        res = fit(gm, poisoned, mask=mask, n_iter=n, lr=0.1)
    assert res.best_iteration == n - 1 and res.best_loss == 1.0
    assert _bits(res.params) == _bits(seen[n][1])


def test_a_run_stopped_by_tol_returns_the_iterate_that_met_it():
    gm = _spring()
    seen, callback = _recorder()
    kw = dict(n_iter=40, lr=0.01, hold_undetermined=False)    # as above
    probe = fit(gm, _bowl, callback=callback, **kw)
    tol = 0.8 * probe.losses[0]
    res = fit(gm, _bowl, tol=tol, **kw)
    assert res.converged and res.losses[-1] <= tol
    assert np.all(res.losses[:-1] > tol)
    assert res.best_iteration == len(res.losses) - 1
    assert res.best_loss == res.losses[-1]
    assert _bits(res.params) == _bits(seen[len(res.losses)][1])


@pytest.mark.parametrize("fitter", ["fit", "fit_multiple_shooting"])
def test_the_final_evaluation_is_invisible_but_for_the_best_fields(recorded, fitter):
    """The iterate the last update produced is evaluated once more, and
    that evaluation must not leak into anything the run already reported:
    not ``losses``, not ``callback``, not the observers -- and not the
    identifiability guard, whose verdict it would change.  Fewer gradients
    than trainable coordinates is "not measured" (``None``); feeding the
    final gradient to the tracker would make it one gradient more and turn
    that into a rank."""
    gm, obs, loss = recorded
    events = []
    gm.add_observer(lambda ev, data: events.append(data["iteration"])
                    if ev == EVENT_FIT_PROGRESS else None)
    seen, callback = _recorder()
    mask = _only(gm, "stiffness", "damping", "mass")
    start = _with(gm, {"stiffness": 36.0})
    try:
        if fitter == "fit":
            res = fit(gm, loss, params=start, mask=mask, n_iter=2, lr=0.05,
                      callback=callback)
        else:
            res, _ = fit_multiple_shooting(
                gm, obs, obs_fn=lambda h: h["s"]["position"], window=WINDOW,
                params=start, mask=mask, n_iter=2, lr=0.05, callback=callback)
    finally:
        gm._observers.clear()  # noqa: SLF001 - module-scoped graph
    assert len(res.losses) == 2 and res.n_iter == 2
    assert sorted(seen) == [1, 2] and events == [1, 2]
    assert res.excited_rank is None and res.undetermined_drift is None
    assert res.best_iteration in (0, 1, 2) and res.best_loss is not None


@pytest.mark.parametrize("fitter", ["fit", "fit_lm", "fit_multiple_shooting"])
def test_n_iter_zero_returns_the_start_having_evaluated_nothing(recorded, fitter):
    gm, obs, loss = recorded
    start = _with(gm, {"stiffness": 36.0})
    if fitter == "fit":
        res = fit(gm, loss, params=start, n_iter=0)
    elif fitter == "fit_lm":
        res = fit_lm(gm, _bowl_residual, params=start, n_iter=0)
    else:
        res, _ = fit_multiple_shooting(
            gm, obs, obs_fn=lambda h: h["s"]["position"], window=WINDOW,
            params=start, n_iter=0)
    assert res.best_iteration == 0 and res.best_loss is None
    assert len(res.losses) == 0
    assert _bits(res.params) == _bits(start)


# ---------------------------------------------------------------------------
# fit_lm: its last iterate is its lowest, and the fields say which it is
# ---------------------------------------------------------------------------


def _half_sse(residual_fn, params):
    r = np.asarray(residual_fn(params), dtype=np.float64)
    return 0.5 * float(r @ r)


def test_fit_lm_ending_on_an_accepted_step_reports_that_step():
    """Two iterations from far away, both accepted: the returned iterate is
    the one the second step produced, which no entry of ``losses`` covers,
    and ``best_loss`` is the loss the acceptance test computed for it."""
    gm = _spring()
    start = _with(gm, {"stiffness": 60.0, "damping": 4.0})
    res = fit_lm(gm, _bowl_residual, params=start, n_iter=2)
    assert len(res.losses) == 2 and not res.converged
    assert res.best_iteration == 2
    assert res.best_loss < res.losses[-1] < res.losses[0]
    assert _half_sse(_bowl_residual, res.params) == pytest.approx(res.best_loss, rel=1e-4)


@jax.custom_jvp
def _reversed_slope(x):
    return x


@_reversed_slope.defjvp
def _reversed_slope_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    return x, -t


def _uphill_residual(params):
    """A residual whose Jacobian has the wrong sign: every Marquardt
    candidate climbs the loss however it is damped, and the proposal is a
    full-sized step (stiffness 30 to 21.5), nowhere near ``step_tol``."""
    return _reversed_slope(params["nodes"]["s"]["stiffness"])[None] - 40.0


def test_fit_lm_ending_on_a_rejected_step_reports_the_iterate_it_kept():
    gm = _spring()
    res = fit_lm(gm, _uphill_residual, n_iter=4)
    assert res.n_iter == 1 and not res.converged
    # 50 from ``exp(log(30))``, one float32 rounding away from 30.
    assert res.best_iteration == 0 and res.best_loss == res.losses[0]
    assert res.best_loss == pytest.approx(50.0, rel=1e-5)
    assert _bits(res.params) == _bits(gm.params)


def test_a_shorter_damped_retry_does_not_make_a_rejected_run_converged():
    """Twelve rejections shrink the candidate by ``lam_up`` each, to far
    below ``step_tol`` (``lam`` reaches 1e9: a step of ~3e-10 relative),
    whatever the loss is doing.  Only the proposal is evidence about the
    iterate, so a run whose proposal was large and every retry rejected is
    not converged -- here the Jacobian is simply wrong."""
    gm = _spring()
    for step_tol in (None, 1e-3):
        res = fit_lm(gm, _uphill_residual, n_iter=4, step_tol=step_tol)
        assert not res.converged, step_tol
        assert res.n_iter == 1 and res.best_iteration == 0


def test_a_residual_that_reads_no_trainable_parameter_is_converged_at_its_start():
    """``J = 0``: every point is stationary, the proposal is exactly zero, and
    the run stops at its start as converged -- ``converged`` says the
    iteration stopped moving.  That the data determined nothing is
    ``excited_rank``'s to say (``None``: every gradient was exactly zero)."""
    gm = _spring()
    res = fit_lm(gm, lambda p: jnp.ones(3, jnp.float32), n_iter=4)
    assert res.n_iter == 1 and res.converged
    assert res.best_iteration == 0 and res.best_loss == res.losses[0] == 1.5
    assert res.excited_rank is None
    assert _bits(res.params) == _bits(gm.params)


def test_fit_lm_stopped_by_tol_or_step_tol_reports_the_right_iterate():
    gm = _spring()
    start = _with(gm, {"stiffness": 60.0})
    by_tol = fit_lm(gm, _bowl_residual, params=start, n_iter=10, tol=1e6)
    assert by_tol.converged and by_tol.best_iteration == 0 == len(by_tol.losses) - 1
    assert by_tol.best_loss == by_tol.losses[0]
    by_step = fit_lm(gm, _bowl_residual, params=start, n_iter=50, step_tol=1e-3)
    assert by_step.converged and by_step.n_iter < 50
    # The step that met ``step_tol`` was accepted, so it is the iterate returned.
    assert by_step.best_iteration == len(by_step.losses)
    assert by_step.best_loss <= by_step.losses[-1]


# ---------------------------------------------------------------------------
# Neighbours: every fitter, masked or not, under every transform
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graphs():
    out = {}
    for variant in ("shipped", "transformed"):
        gm = _spring(variant)
        init = {n: gm.get_node_state(n) for n in gm.node_names}
        _, hist = gm.run_scan_with_history(60)
        out[variant] = (gm, observations_from_history(init, hist))
    return out


@pytest.mark.parametrize("variant", ["shipped", "transformed"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("fitter", ["fit", "fit_lm", "fit_multiple_shooting"])
def test_every_fitter_returns_an_iterate_no_worse_than_any_it_evaluated(
        graphs, fitter, masked, variant):
    """The invariants of the selection, on every fitter, with and without a
    mask, and with the parameters under ``log``, ``logit``, the identity
    and the identity clipped to bounds:

    * ``params`` is inside its bounds, and every leaf outside the mask is
      the input, bit for bit;
    * ``best_iteration`` indexes ``losses`` (or is one past it), and
      ``best_loss`` is at or below every loss recorded;
    * the parameters returned reproduce ``best_loss`` -- for
      ``fit_multiple_shooting``, together with the window states returned;
    * and the Adam fitters really did select: their last iterate is not the
      one returned, so the invariants above are tested on a selection and
      not on the old behaviour.
    """
    gm, obs = graphs[variant]
    mask = _only(gm, "stiffness", "rest_length") if masked else None
    start = _with(gm, {"stiffness": 33.0, "damping": 2.2, "rest_length": 1.05})
    obs_fn = lambda h: h["s"]["position"]          # noqa: E731

    def run(hold):
        kw = dict(params=start, mask=mask, hold_undetermined=hold)
        if fitter == "fit":
            res = fit(gm, _bowl, n_iter=12, lr=0.1, **kw)
            return res, float(_bowl(res.params))
        if fitter == "fit_lm":
            res = fit_lm(gm, _bowl_residual, n_iter=6, **kw)
            return res, _half_sse(_bowl_residual, res.params)
        res, ws = fit_multiple_shooting(
            gm, obs, obs_fn=obs_fn, window=10, n_iter=30, lr=0.05, **kw)
        return res, float(windowed_loss(
            gm, res.params, obs, obs_fn=obs_fn, window=10,
            window_states=ws, continuity_weight=1.0))

    # Unguarded, so that "the parameters returned reproduce best_loss" is a
    # statement about the selection alone: the guard may move params after
    # it, along directions it finds flat, and only when the loss agrees
    # (``test_sysid_hold_never_raises_the_loss.py``).
    res, reproduced = run(False)
    # The guard runs after the selection and never feeds back into it.
    held, held_reproduced = run(True)
    np.testing.assert_array_equal(held.losses, res.losses)
    assert (held.best_iteration, held.best_loss) == (res.best_iteration, res.best_loss)
    gm.check_params(held.params)
    # ... and never returns parameters above the selection's loss: within
    # ``2**-13`` (its relative tolerance in float32) and a rounding-level
    # floor.  Three of these cells went uphill under 0.4.0-dev's guard
    # (fit_lm, unmasked and masked, shipped specs: 0.0 -> 0.22 and
    # 0.279 -> 0.372; unmasked, transformed: 0.0 -> 3.2e-10).
    assert held.hold_declined is False
    assert held_reproduced <= reproduced * (1 + 2**-13) + 1e-12, (
        held_reproduced, reproduced, held.excited_rank)

    gm.check_params(res.params)
    selected = (jax.tree.leaves(mask) if mask is not None
                else jax.tree.leaves(gm.trainable_mask()))
    for flag, before, after in zip(selected, jax.tree.leaves(start),
                                   jax.tree.leaves(res.params)):
        if not bool(flag):
            assert np.asarray(after).tobytes() == np.asarray(before).tobytes()

    losses = res.losses
    assert 0 <= res.best_iteration <= len(losses)
    if res.best_iteration < len(losses):
        assert res.best_loss == losses[res.best_iteration]
    assert res.best_loss <= losses.min()
    assert reproduced == pytest.approx(res.best_loss, rel=1e-3, abs=1e-9)
    if fitter == "fit_lm":
        assert res.best_iteration >= len(losses) - 1
    else:
        assert res.best_iteration < len(losses), (res.best_iteration, losses)


# ---------------------------------------------------------------------------
# The guard after the selection: it may not move a well-posed fit uphill
# ---------------------------------------------------------------------------


def test_the_guard_does_not_raise_the_loss_of_a_well_posed_fit():
    """Every direction of the bowl is determined by the data, so the guard
    should hold nothing, and whatever it does hold must leave the loss
    where the fit left it.

    Found by this file's first version as a strict xfail: the guard of
    earlier 0.4.0 development builds held every direction the run's
    gradients had not spanned, reported ``excited_rank`` 1 of 4 here, and
    returned parameters at 0.22 where ``fit_lm`` had reached 0.0.  The
    fuller set of regressions is ``test_sysid_hold_never_raises_the_loss.py``.
    """
    gm = _spring()
    start = _with(gm, {"stiffness": 33.0, "damping": 2.2, "rest_length": 1.05})
    res = fit_lm(gm, _bowl_residual, params=start, n_iter=6)
    assert _half_sse(_bowl_residual, res.params) <= res.best_loss + 1e-6, (
        res.excited_rank, res.undetermined_drift, res.best_loss)
    assert res.excited_rank == 4 and res.hold_declined is False
