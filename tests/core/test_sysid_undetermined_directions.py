"""``fit`` must not move along directions the data cannot determine.

The spring's ``(k, c, m)`` common-scale degeneracy is exact: position-only
data sees ``k/m`` and ``c/m``, so multiplying all three by the same factor
leaves the trajectory bit-identical.  In relative coordinates that is the
direction ``(1, 1, 1)/sqrt(3)``, which
``tests/verification/hypothesis/test_hypothesis_sysid.py`` and
``FIMReport.least_identifiable`` both name, and here it is the direction
along which the *geometric mean* of ``(k, c, m)`` is the coordinate.

Adam's diagonal preconditioner moves along it.  ``g . v`` is exactly zero
because every gradient is ``J^T r``, but the *update* is ``-lr * D g`` and
``(D g) . v`` is not, so the fitted scale is decided by the iteration
budget and the learning rate rather than by the data.  What is measured
below is that the guard removes that, and that it removes nothing else.
"""

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
    assert res.undetermined_drift is not None and res.undetermined_drift > 1e-3
    # 1e-4 relative is ~3 float32 ulps on a quantity of order 5; the guard
    # zeroes this exactly in exact arithmetic.
    assert abs(_scale(res.params) / START_SCALE - 1.0) < 1e-4, _scale(res.params)


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
    assert held.undetermined_drift == 0.0
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
    assert held.undetermined_drift == 0.0
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
    assert held.undetermined_drift is None
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


def test_hold_undetermined_refuses_a_non_bool(noisy_fit):
    """A truthy non-bool would silently decide whether the returned
    parameters are reproducible."""
    gm, loss = noisy_fit
    with pytest.raises(ValueError, match="hold_undetermined must be a bool"):
        fit(gm, loss, params=_with(gm, START), n_iter=2,
            hold_undetermined="yes")            # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ``fit_lm`` for comparison -- it drifts too, which is not what was reported
# ---------------------------------------------------------------------------


def test_fit_lm_also_moves_along_the_scale_direction_but_stops(clean_fit):
    """Pins a correction to the finding this guard came from.

    The claim was that ``fit_lm`` *cannot* move in an exactly-null
    direction, because ``J v = 0`` makes ``(J^T r) . v = 0``.  The gradient
    is indeed orthogonal to ``v``, but the step is not the gradient: the
    Marquardt solve is ``(A + lam * diag(A))^-1 g``, and that is orthogonal
    to ``null(A)`` only when ``diag(A)`` is isotropic on the relevant
    subspace.  ``fit_lm`` does move along the scale direction.

    What is different in kind, and why it is not the defect ``fit`` has, is
    that the step vanishes as the gradient does: the drift converges and
    stops, so the answer does not depend on the budget.  Both halves are
    asserted, because a future change that made LM's drift budget-dependent
    would be the same defect arriving by a different route.
    """
    gm = _spring()
    obs = _observations(gm, noisy=False)
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

    start = _with(gm, START)
    short = fit_lm(gm, residual, params=start, n_iter=10, notify_every=0)
    long_ = fit_lm(gm, residual, params=start, n_iter=60, notify_every=0)
    drift = _scale(short.params) / START_SCALE - 1.0
    # It moves: not the "cannot move" the finding claimed.
    assert abs(drift) > 1e-3, drift
    # But by well under Adam's, and it has stopped: the budget does not
    # decide the answer.
    assert abs(drift) < 0.05, drift
    assert _scale(short.params) == _scale(long_.params)
    # LM does not run the guard, and must say so rather than claim a rank.
    assert short.excited_rank is None and short.undetermined_drift is None


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
