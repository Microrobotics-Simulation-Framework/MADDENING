"""``fit_lm`` on a residual that is not differentiable in the trained
parameters, and what its verdicts say there (SYS-080, SYS-086).

A node with an event whose timing depends on a trained parameter has a
residual that jumps when the event moves by one time step (MADD-ANO-021).
Levenberg-Marquardt follows a smooth piece of such a loss to its edge, where
every candidate the tolerance resolves crosses the jump and is rejected; the
floor rule read that as the rounding floor and reported ``converged=True``
at losses far above the minimum.  It now tells the two apart from the
rejected candidates themselves (``sysid._one_sided_excess``).

The stock non-smooth graph is the parameter guide's own: ``TableNode ->
BallNode``.  A second ball, on the same table and trained by nothing, is the
record, so one rollout gives the residual and the truth is known exactly.

Also here, because it is the same kind of statement -- a verdict of
``fit_lm`` that the returned point's own Jacobian contradicts: the warning
when ``excited_rank`` reads the full count on a fit whose ``JᵀJ`` does not
(``HeartPumpNode`` with its default specs).
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening import sysid
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes import BallNode, HeartPumpNode, SpringDamperNode, TableNode
from maddening.sysid import fit, fit_lm

DT = 0.01
JUMP = "not differentiable"
RANK = "resolves only"
#: A loss of the ball's residual at the truth is rounding, below 1e-10; the
#: endings beside a jump measured 0.02 to 3.7.
AT_THE_MINIMUM = 1e-9


def _messages(caught, text):
    return [str(w.message) for w in caught if text in str(w.message)]


def _twin(gm, node, n_steps, field):
    """``residual(p)``: ``node``'s ``field`` minus that of ``record``, the
    same node with the generating constants, over one rollout."""
    for leaf in gm.params["nodes"]["record"]:
        gm.set_param_spec("record", leaf, ParamSpec(trainable=False))
    init = {name: {k: v[None] for k, v in gm.get_node_state(name).items()}
            for name in gm.node_names}

    def residual(p):
        history = gm.run_sweep(n_steps, init, return_history=True, params=p)[1]
        return history[node][field][0] - history["record"][field][0]

    return residual


def _started(gm, node, **values):
    start = jax.tree.map(lambda x: x, gm.params)
    for leaf, value in values.items():
        start["nodes"][node][leaf] = jnp.asarray(value, start["nodes"][node][leaf].dtype)
    return start


# ---------------------------------------------------------------------------
# The bouncing ball: a residual with jumps
# ---------------------------------------------------------------------------

#: ``(elasticity, gravity over -9.81)`` to start from; the truth is
#: ``(0.7, 1)``.  Before the one-sided test the first four ended
#: ``converged=True`` at losses of 0.02 to 3.7 (CPU, jaxlib 0.11.0).
BALL_STARTS = [(0.6, 1.1), (0.65, 0.9), (0.75, 1.25), (0.8, 0.8),
               (0.55, 0.9), (0.6, 0.9), (0.75, 0.9), (0.85, 0.9)]


@pytest.fixture(scope="module")
def ball():
    gm = GraphManager()
    gm.add_node(TableNode("table", DT))
    for name in ("ball", "record"):
        gm.add_node(BallNode(name, DT, initial_position=1.0, elasticity=0.7))
        gm.add_edge("table", name, "position", "table_position")
    gm.compile()
    return gm, _twin(gm, "ball", 200, "position")


@pytest.fixture(scope="module")
def ball_fits(ball):
    gm, residual = ball
    fits = []
    for elasticity, gravity in BALL_STARTS:
        start = _started(gm, "ball", elasticity=elasticity, gravity=-9.81 * gravity)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fits.append((fit_lm(gm, residual, params=start), _messages(caught, JUMP)))
    return fits


def test_fit_lm_is_converged_on_the_ball_only_at_the_minimum(ball_fits):
    """``converged=True`` is never reported beside a jump: every converged
    fit of the ball is at the truth's loss."""
    high = [(start, res.best_loss) for start, (res, _) in zip(BALL_STARTS, ball_fits)
            if res.converged and not res.best_loss <= AT_THE_MINIMUM]
    assert not high, high
    # The grid holds both kinds of ending, or it shows nothing.
    assert sum(res.converged for res, _ in ball_fits) >= 2
    assert sum(not res.converged for res, _ in ball_fits) >= 2


def test_fit_lm_beside_a_jump_is_unconverged_and_says_why(ball_fits):
    """An ending beside a jump is ``converged=False`` with one warning that
    names a residual that is not differentiable, the registry entry and
    the guide; a converged fit carries none."""
    stopped = [(res, texts) for res, texts in ball_fits if not res.converged]
    for res, texts in ball_fits:
        assert len(texts) == (0 if res.converged else len(texts))
        assert not (res.converged and texts)
    warned = [texts for _, texts in stopped if texts]
    assert len(warned) >= 2, [res.best_loss for res, _ in stopped]
    for texts in warned:
        assert len(texts) == 1
        assert "converged=False" in texts[0] and "MADD-ANO-021" in texts[0]
        # That it names the guide is read in
        # tests/compliance/test_sysid_non_differentiable_residual_docs.py.
    # Each of them is above the minimum: the run was stopped by a jump, not
    # by the rounding floor.
    for res, texts in stopped:
        if texts:
            assert res.best_loss > AT_THE_MINIMUM
            assert res.best_loss == res.losses[res.best_iteration]


def test_the_loss_beside_a_warned_ending_does_fall_across_the_jump(ball, ball_fits):
    """The warning is true of the point: within a hundredth of the returned
    parameters the loss is lower by more than rounding explains."""
    gm, residual = ball
    program = jax.jit(lambda p: 0.5 * jnp.sum(residual(p) ** 2))
    checked = 0
    for res, texts in ball_fits:
        if not texts:
            continue
        leaves = res.params["nodes"]["ball"]
        lowest = np.inf
        for de in (-1e-2, -1e-3, 1e-3, 1e-2):
            for dg in (-1e-2, -1e-3, 0.0, 1e-3, 1e-2):
                near = dict(leaves, elasticity=leaves["elasticity"] * (1.0 + de),
                            gravity=leaves["gravity"] * (1.0 + dg))
                tree = dict(res.params, nodes=dict(res.params["nodes"], ball=near))
                lowest = min(lowest, float(program(tree)))
        assert lowest < res.best_loss * (1.0 - 1e-3), (res.best_loss, lowest)
        checked += 1
    assert checked >= 2


def test_fit_makes_no_convergence_statement_on_the_ball(ball):
    """The sibling: Adam's only stopping test is ``tol``, so on the same
    graph it reports ``converged=False`` and no warning about jumps."""
    gm, residual = ball
    start = _started(gm, "ball", elasticity=0.6, gravity=-9.81 * 1.1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fit(gm, lambda p: 0.5 * jnp.sum(residual(p) ** 2), params=start, n_iter=30,
                  lr=0.01)
    assert res.converged is False
    assert not _messages(caught, JUMP)


# ---------------------------------------------------------------------------
# Smooth residuals: the floor rule's verdict is unchanged
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def noisy_spring():
    gm = GraphManager()
    for name in ("spring", "record"):
        gm.add_node(SpringDamperNode(name, DT, stiffness=30.0, damping=2.0,
                                     initial_position=0.5))
    gm.compile()
    clean = _twin(gm, "spring", 200, "position")
    noise = jnp.asarray(np.random.default_rng(0).normal(0.0, 3e-3, 200), jnp.float32)
    mask = jax.tree.map(lambda _: False, gm.params)
    for key in ("stiffness", "damping"):
        mask["nodes"]["spring"][key] = True
    return gm, (lambda p: clean(p) + noise), mask


def test_a_smooth_fit_stopped_by_the_floor_rule_is_still_converged(noisy_spring, monkeypatch):
    """A noisy float32 fit at its rounding floor: the one-sided test is
    asked, reads a ratio of order one, and the fit converges with no
    warning -- the same answer from every start."""
    gm, residual, mask = noisy_spring
    real, read = sysid._one_sided_excess, []  # noqa: SLF001

    def watched(*args):
        out = real(*args)
        read.append(out[0])
        return out

    monkeypatch.setattr(sysid, "_one_sided_excess", watched)
    rng = np.random.default_rng(3)
    answers = []
    for _ in range(12):
        k0, c0 = 30.0 * np.exp(rng.uniform(-0.3, 0.3)), 2.0 * np.exp(rng.uniform(-0.3, 0.3))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = fit_lm(gm, residual, mask=mask,
                         params=_started(gm, "spring", stiffness=k0, damping=c0))
        assert res.converged and not _messages(caught, JUMP)
        answers.append([float(res.params["nodes"]["spring"][k])
                        for k in ("stiffness", "damping")])
    assert np.ptp(np.asarray(answers), axis=0).max() < 1e-3
    assert read, "premise: at least one of the fits was stopped by the floor rule"
    # Of order one; the threshold is 2**10, and nothing is read a second
    # time below 2**5.
    assert max(read) < sysid._JUMP_SUSPECT  # noqa: SLF001


# ---------------------------------------------------------------------------
# The one-sided test itself
# ---------------------------------------------------------------------------

def _reading(fn, theta, steps, lo=-np.inf, hi=np.inf, asked=None):
    """``_one_sided_excess`` of ``fn`` at ``theta`` for candidates
    ``theta + step``, with its Jacobian by ``jacfwd``: ``(excess, move,
    jumped)``.  ``asked`` collects the points it evaluated (mirror images
    and halvings)."""
    theta = jnp.asarray(theta, jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    r, J = fn(theta), jax.jacfwd(fn)(theta)
    rejected = []
    for step in steps:
        cand = theta + jnp.asarray(step, theta.dtype)
        rejected.append((cand, fn(cand), float(np.max(np.abs(step)))))

    def mirror(th):
        th = jnp.clip(th, lo, hi)
        if asked is not None:
            asked.append(np.asarray(th))
        return th, fn(th)

    return sysid._one_sided_excess(theta, r, J, rejected, mirror)  # noqa: SLF001


def _excess(fn, theta, steps, **kwargs):
    """``(excess, move)`` of :func:`_reading`, the verdict being the
    threshold's: a jump exactly where ``excess`` is over ``2**10``."""
    excess, move, jumped = _reading(fn, theta, steps, **kwargs)
    assert jumped == (excess > sysid._JUMP_EXCESS)  # noqa: SLF001
    return excess, move


def _smooth(th):
    return jnp.stack([jnp.sin(th[0]) + th[1] ** 2, jnp.exp(th[0] * th[1]), th[0] - th[1]])


def _jumping(th):
    # A step of 0.25 in the second entry as the first coordinate passes 1.
    return _smooth(th) + jnp.where(th[0] > 1.0, 0.25, 0.0) * jnp.asarray([0.0, 1.0, 0.0])


def test_the_one_sided_excess_of_a_smooth_residual_is_small():
    excess, _ = _excess(_smooth, [1.0, 0.5], [[1e-4, 0.0], [-2e-4, 1e-4], [3e-5, 3e-5]])
    assert excess < 2.0


def test_the_one_sided_excess_of_a_jump_is_the_jump_over_the_linear_change():
    """Sitting on the edge of a piece, a candidate across it departs from
    the linearisation by the jump; its mirror image does not."""
    excess, move = _excess(_jumping, [1.0, 0.5], [[0.0, 1e-4], [1e-4, 0.0]])
    assert move == pytest.approx(1e-4)
    # The jump, 0.25, over the linear change of a step of 1e-4 (of order
    # 1e-4): thousands.
    assert excess > 1e3 > sysid._JUMP_EXCESS / 2.0  # noqa: SLF001
    # From the far side of the same point the step back is the one that jumps.
    inside, _ = _excess(_jumping, [1.0, 0.5], [[-1e-4, 0.0]])
    assert inside < 2.0


def test_a_candidate_with_no_mirror_image_or_no_finite_residual_says_nothing():
    # On its lower bound, the mirror image of a step up is the point itself.
    on_bound = jnp.asarray([1.0, -np.inf])
    assert _excess(_jumping, [1.0, 0.5], [[1e-4, 0.0]], lo=on_bound)[0] == 0.0
    # A candidate that did not move.
    assert _excess(_jumping, [1.0, 0.5], [[0.0, 0.0]])[0] == 0.0

    def to_nan(th):
        return _smooth(th) + jnp.where(th[0] > 1.0, jnp.nan, 0.0)

    assert _excess(to_nan, [1.0, 0.5], [[1e-4, 0.0]])[0] == 0.0
    assert _excess(to_nan, [1.0, 0.5], [[-1e-4, 0.0]])[0] == 0.0


# ---------------------------------------------------------------------------
# A small jump read from a long candidate: the second reading
# ---------------------------------------------------------------------------

#: A step of 0.02 in the second entry, read from a candidate of 2e-4: the
#: jump over the linear change of the whole candidate (2.8e-4) is about 70,
#: between the two thresholds of the first reading.
_SMALL_JUMP, _LONG_STEP = 0.02, [2e-4, 0.0]


def _small_jump(at):
    def fn(th):
        return _smooth(th) + (jnp.where(th[0] > at, _SMALL_JUMP, 0.0)
                              * jnp.asarray([0.0, 1.0, 0.0]))
    return fn


def _steep(th):
    # Smooth, with a slope that grows by exp(6.5) over a step of 2e-4.
    return jnp.stack([jnp.exp(3.25e4 * (th[0] - 1.0)), th[1]])


def _first_reading(monkeypatch, fn, steps, **kwargs):
    """The reading with no second one: ``(excess, move, jumped)``, and the
    premise that the largest ratio is between the two thresholds."""
    with monkeypatch.context() as patch:
        patch.setattr(sysid, "_JUMP_SUSPECT", np.inf)
        whole = _reading(fn, [1.0, 0.5], steps, **kwargs)
    assert sysid._JUMP_EXCESS > whole[0] > sysid._JUMP_SUSPECT  # noqa: SLF001
    assert not whole[2]
    return whole


# The edge beside the iterate, and inside the candidate's step (the fit of
# the ball that showed this ended 22 float spacings from its edge).
@pytest.mark.parametrize("at", [1.0, 1.00005, 1.00019])
def test_a_small_jump_inside_a_long_candidate_is_read_across_one_spacing(at, monkeypatch):
    whole = _first_reading(monkeypatch, _small_jump(at), [_LONG_STEP])
    excess, move, jumped = _reading(_small_jump(at), [1.0, 0.5], [_LONG_STEP])
    assert jumped and move == pytest.approx(2e-4)
    # The jump over the mirror image's departure and one float32 spacing's
    # linear change: tens of thousands.
    assert excess > 2.0 ** 4 * sysid._JUMP_ACROSS > whole[0]  # noqa: SLF001


def test_a_jump_read_the_second_time_is_the_one_reported(monkeypatch):
    """Beside a candidate whose first reading is larger and which is no
    jump, whichever of the two comes first."""
    def fn(th):
        # Along the first coordinate the small jump, met on the other side
        # by one 400 times smaller; along the second a smooth residual
        # whose slope grows by exp(9) over a step of 1e-4.
        met = jnp.where(2.0 - th[0] > 1.00005, _SMALL_JUMP / 400.0, 0.0)
        steep = jnp.exp(9e4 * (th[1] - 0.5))
        return _small_jump(1.00005)(th) + jnp.stack([met, 0.0 * met, steep])

    jump, smooth = _LONG_STEP, [0.0, 1e-4]
    only_jump = _reading(fn, [1.0, 0.5], [jump])
    only_smooth = _reading(fn, [1.0, 0.5], [smooth])
    assert only_jump[2] and not only_smooth[2]
    # Premise: the smooth candidate's reading is the larger number.
    assert sysid._JUMP_EXCESS > only_smooth[0] > only_jump[0] > sysid._JUMP_ACROSS  # noqa: SLF001
    assert only_jump[1] == pytest.approx(2e-4) and only_smooth[1] == pytest.approx(1e-4)
    for steps in ([jump, smooth], [smooth, jump]):
        assert _reading(fn, [1.0, 0.5], steps) == only_jump


def test_a_small_jump_met_by_one_on_the_other_side_is_not_one_sided(monkeypatch):
    """A second, smaller jump where the mirror image of the candidate
    falls: the second reading is the ratio of the two jumps, under its
    threshold, and the candidate keeps its first."""
    def fn(th):
        return _small_jump(1.00005)(th) + (jnp.where(2.0 - th[0] > 1.00005, 3e-4, 0.0)
                                           * jnp.asarray([1.0, 0.0, 0.0]))

    asked = []
    reading = _reading(fn, [1.0, 0.5], [_LONG_STEP], asked=asked)
    assert len(asked) > 4, "premise: the candidate was read a second time"
    assert reading == _first_reading(monkeypatch, fn, [_LONG_STEP])


def test_a_smooth_residual_read_a_second_time_is_still_not_a_jump(monkeypatch):
    """A residual curved enough over the candidate to be read again: one
    spacing holds a small part of the departure, and the mirror image's is
    of its size."""
    asked, across = [], []
    real = sysid._excess_across_a_spacing  # noqa: SLF001
    monkeypatch.setattr(sysid, "_excess_across_a_spacing",
                        lambda *args: across.append(real(*args)) or across[-1])
    reading = _reading(_steep, [1.0, 0.5], [_LONG_STEP], asked=asked)
    assert len(asked) > 4 and len(across) == 1
    assert 0.0 < across[0] < 1.0
    assert reading == _first_reading(monkeypatch, _steep, [_LONG_STEP])


def test_a_candidate_outside_the_two_thresholds_is_read_once():
    """A ratio of order one costs the mirror image and nothing more, and
    one over the upper threshold needs no second reading."""
    for fn, steps in ((_smooth, [[1e-4, 0.0], [-2e-4, 1e-4], [3e-5, 3e-5]]),
                      (_jumping, [[1e-4, 0.0]])):
        asked = []
        _excess(fn, [1.0, 0.5], steps, asked=asked)
        assert len(asked) == len(steps)


def test_a_second_reading_through_a_residual_that_is_not_finite_says_nothing(monkeypatch):
    def fn(th):
        gap = (th[0] > 1.00004) & (th[0] < 1.00016)
        return _small_jump(1.00005)(th) + jnp.where(gap, jnp.nan, 0.0)

    asked = []
    reading = _reading(fn, [1.0, 0.5], [_LONG_STEP], asked=asked)
    assert len(asked) > 4, "premise: the candidate was read a second time"
    assert reading == _first_reading(monkeypatch, fn, [_LONG_STEP])


# ---------------------------------------------------------------------------
# A ladder of candidates that never moved is not a ladder of rejections
# ---------------------------------------------------------------------------

def test_an_iteration_whose_damped_candidates_never_moved_is_asked_again(monkeypatch):
    """The floor rule's "every candidate was rejected" needs candidates.
    Simulated by a damping so high that the twelve of one iteration round to
    no move, at an iterate whose undamped Gauss-Newton step overshoots and
    raises the loss: nothing between the two was tried, so the iterate is
    run once more from ``lam0`` instead of being reported converged, and
    the fit goes on to the minimum."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, stiffness=30.0, damping=1.75))
    gm.compile()
    mask = jax.tree.map(lambda _: False, gm.params)
    mask["nodes"]["s"]["damping"] = True

    def residual(p):
        # ``arctan``: the Gauss-Newton step overshoots from beyond 1.39.
        return jnp.stack([jnp.arctan(4.0 * (p["nodes"]["s"]["damping"] - 1.0))])

    real, calls = sysid._marquardt_step, []  # noqa: SLF001

    def stalls_for_one_ladder(th, r, J, lam, lo, hi, held):
        calls.append(1)
        if 2 <= len(calls) <= 13:
            return th, True
        return real(th, r, J, lam, lo, hi, held)

    monkeypatch.setattr(sysid, "_marquardt_step", stalls_for_one_ladder)
    res = fit_lm(gm, residual, mask=mask, lam0=10.0, hold_undetermined=False)
    assert len(calls) > 13
    # The iterate the stalled ladder left is evaluated twice: once by the
    # iteration that stalled, once by the one that asked again.
    assert res.losses[1] == res.losses[2] < res.losses[0]
    assert res.converged
    assert float(res.params["nodes"]["s"]["damping"]) == pytest.approx(1.0, abs=1e-5)
    assert res.best_loss < 1e-10


def test_candidates_that_never_move_are_asked_again_once_only(monkeypatch):
    """Asked again once per iterate: a ladder that never moves from there
    either does not keep the run going for its whole budget."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, stiffness=30.0, damping=1.75))
    gm.compile()
    mask = jax.tree.map(lambda _: False, gm.params)
    mask["nodes"]["s"]["damping"] = True

    def residual(p):
        return jnp.stack([jnp.arctan(4.0 * (p["nodes"]["s"]["damping"] - 1.0))])

    real, calls = sysid._marquardt_step, []  # noqa: SLF001

    def stalls_after_the_first(th, r, J, lam, lo, hi, held):
        calls.append(1)
        return real(th, r, J, lam, lo, hi, held) if len(calls) == 1 else (th, True)

    monkeypatch.setattr(sysid, "_marquardt_step", stalls_after_the_first)
    res = fit_lm(gm, residual, mask=mask, lam0=10.0, hold_undetermined=False, n_iter=40)
    assert res.n_iter == 3, res.n_iter
    assert res.losses[1] == res.losses[2]


# ---------------------------------------------------------------------------
# ``excited_rank`` reads full on a fit ``JᵀJ`` does not resolve
# ---------------------------------------------------------------------------

HEART_KEYS = ("resistance", "compliance", "stroke_volume")


def _heart(log_stroke_volume: bool):
    gm = GraphManager()
    for name in ("heart", "record"):
        gm.add_node(HeartPumpNode(name, DT))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    if log_stroke_volume:
        gm.set_param_spec("heart", "stroke_volume",
                          ParamSpec(bounds=(0.0, None), transform="log"))
    residual = _twin(gm, "heart", 240, "arterial_pressure")
    mask = jax.tree.map(lambda _: False, gm.params)
    for key in HEART_KEYS:
        mask["nodes"]["heart"][key] = True
    truth = gm.params["nodes"]["heart"]
    # The truth up to the scale the data cannot see, then 15% off in the
    # two ratios they can.
    scale = 1.3
    start = _started(gm, "heart", resistance=float(truth["resistance"]) / scale * 1.15,
                     compliance=float(truth["compliance"]) * scale,
                     stroke_volume=float(truth["stroke_volume"]) * scale * 0.85)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fit_lm(gm, residual, params=start, mask=mask)
    return gm, residual, mask, res, _messages(caught, RANK)


def test_fit_lm_warns_when_the_guard_reads_full_and_the_jacobian_does_not():
    """``HeartPumpNode`` with its default specs: the data fix ``R C`` and
    ``SV / C`` but not the scale, the identity ``stroke_volume`` turns the
    null direction in the optimiser's coordinates, and the guard finds
    nothing.  ``fit_lm`` says that its own ``JᵀJ`` resolves fewer."""
    gm, residual, mask, res, texts = _heart(log_stroke_volume=False)
    assert res.excited_rank == 3 and res.hold_declined is False
    assert len(texts) == 1, texts
    assert "excited_rank is 3 of 3" in texts[0] and "resolves only 2" in texts[0]
    assert "stroke_volume" in texts[0] and "transform='log'" in texts[0]
    # It is fim's verdict at the returned point.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert sysid.fim(residual, res.params, mask=mask).rank == 2


def test_fit_lm_warns_on_the_springs_scale_under_its_default_specs():
    """The other stock instance: ``SpringDamperNode``'s identity ``damping``
    beside a ``log`` stiffness and mass, the scale of ``(k, c, m)`` free."""
    gm = GraphManager()
    for name in ("spring", "record"):
        gm.add_node(SpringDamperNode(name, DT, stiffness=30.0, damping=2.0,
                                     initial_position=0.5))
    gm.compile()
    residual = _twin(gm, "spring", 200, "position")
    mask = jax.tree.map(lambda _: False, gm.params)
    for key in ("stiffness", "damping", "mass"):
        mask["nodes"]["spring"][key] = True
    start = _started(gm, "spring", stiffness=30.0 * 1.3 * 1.15, damping=2.0 * 1.3 * 0.85,
                     mass=1.3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fit_lm(gm, residual, params=start, mask=mask)
    texts = _messages(caught, RANK)
    assert res.excited_rank == 3, res.excited_rank
    assert len(texts) == 1 and "resolves only 2" in texts[0], texts


def test_fit_lm_is_silent_where_the_guard_held_the_direction():
    """The documented remedy: under ``log`` the direction is fixed, the
    guard holds it, ``excited_rank`` is 2 and there is nothing to warn of."""
    _, _, _, res, texts = _heart(log_stroke_volume=True)
    assert res.excited_rank == 2
    assert not texts


def test_fit_lm_is_silent_on_a_fit_every_direction_of_which_is_resolved(noisy_spring):
    gm, residual, mask = noisy_spring
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fit_lm(gm, residual, mask=mask,
                     params=_started(gm, "spring", stiffness=24.0, damping=2.5))
    assert res.converged and res.excited_rank == 2
    assert not _messages(caught, RANK)


def test_the_curvature_rank_is_fims_rank_in_fims_coordinates():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(40, 3)).astype(np.float32)
    ones = np.ones(3)
    rank = sysid._curvature_rank  # noqa: SLF001
    assert rank(J, ones) == 3
    # The third column a combination of the others: one direction fewer.
    flat = np.column_stack([J[:, 0], J[:, 1], 2.0 * J[:, 0] - J[:, 1]])
    assert rank(flat, ones) == 2
    # A column that is small because of its coordinate (a parameter in
    # small units, a transform that is flat there) is read by its scale.
    small = J * np.asarray([1.0, 1.0, 1e-9], np.float32)
    assert rank(small, np.asarray([1.0, 1.0, 1e-9])) == 3
    assert rank(small, ones) == 2
    # No statement without a scale, or without a finite J.
    assert rank(J, np.asarray([1.0, 0.0, 1.0])) is None
    assert rank(J, np.asarray([1.0, np.inf, 1.0])) is None
    J[3, 1] = np.nan
    assert rank(J, ones) is None


def test_the_warning_does_not_fire_on_a_coordinate_its_transform_flattens():
    """A ``logit`` damping whose truth sits 0.2% below the upper end of its
    range: in the optimiser's coordinates its column is small because the
    transform is flat there (a singular-value ratio of 5e-4, under float32's
    cutoff of 1e-3), which says nothing about the data (0.26 in the
    parameter relative to itself).  The fit recovers both parameters, and
    the warning, read as fim reads such a parameter, is silent.  (A
    parameter whose value is zero is the other case of a column with no
    scale of its own:
    ``test_sysid_fit_lm_step_tol.py::test_a_truth_of_exactly_zero_has_no_relative_resolution``.)"""
    gm = GraphManager()
    for name in ("spring", "record"):
        gm.add_node(SpringDamperNode(name, DT, stiffness=30.0, damping=1.9,
                                     initial_position=0.5))
    gm.compile()
    gm.set_param_spec("spring", "damping", ParamSpec(bounds=(0.0, 1.9038), transform="logit"))
    residual = _twin(gm, "spring", 100, "position")
    mask = jax.tree.map(lambda _: False, gm.params)
    for key in ("stiffness", "damping"):
        mask["nodes"]["spring"][key] = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fit_lm(gm, residual, mask=mask,
                     params=_started(gm, "spring", stiffness=24.0, damping=1.2))
    assert res.converged and res.excited_rank == 2
    assert float(res.params["nodes"]["spring"]["damping"]) == pytest.approx(1.9, rel=1e-4)
    assert not _messages(caught, RANK)


def test_the_rank_reading_makes_no_statement_where_its_scaled_jacobian_overflows():
    """A finite Jacobian column over a tiny scale overflows in the guard's
    coordinates; the reading then says nothing instead of raising out of a
    diagnostic (it used to reach the SVD with an infinite matrix)."""
    import numpy as np

    from maddening.sysid import _curvature_rank

    J = np.array([[1e300, 1.0], [1.0, 1.0]])
    assert _curvature_rank(J, np.array([1e-300, 1.0])) is None
    assert _curvature_rank(np.array([[2.0, 1.0], [1.0, 3.0]]), np.array([1.0, 1.0])) == 2
