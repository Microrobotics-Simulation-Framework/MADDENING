"""Edge tests for rows of ``docs/validation/sysid_fmu_claims.yaml`` (SYS-NNN).

Each test is cited by a row of the inventory and sits at an edge of the
conditions the row's claim was stated for: the ends of a window's range, a
residual rescaled by many decades, a parameter in other units, a coupled
step under ``jacfwd``, an exact zero loss.  Where the tree does not meet a
claim the test is a strict xfail whose reason starts with the row's id;
``tests/compliance/test_claims_inventories.py`` holds the two together.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import (
    fim,
    fim_core,
    fit,
    fit_lm,
    fit_multiple_shooting,
    init_window_states,
    observations_from_history,
    windowed_loss,
)

N, WINDOW = 40, 10


def _spring(*, external=False, **kw):
    p = dict(stiffness=30.0, damping=2.0, mass=1.0, rest_length=1.0, initial_position=0.5)
    p.update(kw)
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, **p))
    if external:
        gm.add_external_input("s", "anchor_position")
    gm.compile()
    return gm


def _record(gm, n=N, **run):
    init = {name: gm.get_node_state(name) for name in gm.node_names}
    _, hist = gm.run_scan_with_history(n, **run)
    return observations_from_history(init, hist)


def _position(h):
    return h["s"]["position"]


def _only(gm, *keys):
    mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
    for key in keys:
        mask["nodes"]["s"][key] = True
    return mask


@pytest.fixture(scope="module")
def recorded():
    gm = _spring()
    return gm, _record(gm)


# ---------------------------------------------------------------------------
# windowed_loss and its helpers
# ---------------------------------------------------------------------------


def test_observations_from_history_prepends_the_initial_state():
    """SYS-001: sample 0 is the state the record starts from, then one sample per step."""
    gm = _spring()
    init = {name: gm.get_node_state(name) for name in gm.node_names}
    _, hist = gm.run_scan_with_history(7)
    obs = observations_from_history(init, hist)
    assert obs["s"]["position"].shape == (8,)
    assert float(obs["s"]["position"][0]) == float(init["s"]["position"])
    np.testing.assert_array_equal(np.asarray(obs["s"]["position"][1:]),
                                  np.asarray(hist["s"]["position"]))


@pytest.mark.parametrize("window, ok", [(0, False), (1, True), (3, False), (N, True),
                                        (N + 1, False)])
def test_a_window_lies_in_one_to_t_minus_one_and_tiles_the_record(recorded, window, ok):
    """SYS-003: ``window`` divides ``T - 1`` and lies in ``[1, T - 1]``; both ends are legal."""
    gm, obs = recorded
    if ok:
        assert float(windowed_loss(gm, gm.params, obs, obs_fn=_position, window=window)) == 0.0
    else:
        with pytest.raises(ValueError, match=r"must divide T-1=40 and lie in \[1, T-1\]"):
            windowed_loss(gm, gm.params, obs, obs_fn=_position, window=window)


def test_sample_every_and_a_one_sample_record_are_refused(recorded):
    """SYS-003: ``sample_every >= 1``, and ``T == 1`` has nothing to fit."""
    gm, obs = recorded
    with pytest.raises(ValueError, match="sample_every=0 must be >= 1"):
        windowed_loss(gm, gm.params, obs, obs_fn=_position, window=WINDOW, sample_every=0)
    one = jax.tree.map(lambda x: x[:1], obs)
    with pytest.raises(ValueError, match=r"T-1=0"):
        windowed_loss(gm, gm.params, one, obs_fn=_position, window=1)


def test_windowed_loss_leaves_the_graphs_state_untouched(recorded):
    """SYS-004: "``gm._state`` is not modified" -- by a value, a gradient or a masked loss."""
    gm, obs = recorded
    before = [np.array(x) for x in jax.tree.leaves(gm._state)]
    windowed_loss(gm, gm.params, obs, obs_fn=_position, window=WINDOW)
    jax.grad(lambda p: windowed_loss(gm, p, obs, obs_fn=_position, window=WINDOW))(gm.params)
    after = [np.array(x) for x in jax.tree.leaves(gm._state)]
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_windowed_loss_completes_and_checks_external_inputs():
    """SYS-005: an omitted declared input is zero, as ``gm.step`` takes it; an undeclared
    ``node.field`` is a ``ValueError``."""
    gm = _spring(external=True)
    obs = _record(gm, external_inputs={"s": {"anchor_position": jnp.float32(0.25)}})
    with_input = windowed_loss(gm, gm.params, obs, obs_fn=_position, window=WINDOW,
                               external_inputs={"s": {"anchor_position": jnp.float32(0.25)}})
    assert float(with_input) == 0.0
    omitted = windowed_loss(gm, gm.params, obs, obs_fn=_position, window=WINDOW)
    zeros = windowed_loss(gm, gm.params, obs, obs_fn=_position, window=WINDOW,
                          external_inputs={"s": {"anchor_position": jnp.float32(0.0)}})
    assert float(omitted) == float(zeros) > 0.0
    for bad in ({"s": {"nope": 1.0}}, {"zz": {"anchor_position": 1.0}}):
        with pytest.raises(ValueError, match="which this graph does not declare"):
            windowed_loss(gm, gm.params, obs, obs_fn=_position, window=WINDOW,
                          external_inputs=bad)


def test_the_teacher_forced_loss_is_the_sum_of_independent_windows(recorded):
    """SYS-006: every window restarts from the measured state and is its own scan, so the
    loss over the record is the sum of the losses over each window's slice of it."""
    gm, obs = recorded
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s"]["stiffness"] = jnp.float32(41.0)
    whole = float(windowed_loss(gm, p, obs, obs_fn=_position, window=WINDOW))
    parts = sum(float(windowed_loss(gm, p, jax.tree.map(
        lambda x, w=w: x[w * WINDOW:(w + 1) * WINDOW + 1], obs),
        obs_fn=_position, window=WINDOW)) for w in range(N // WINDOW))
    assert whole > 0.0
    assert whole == pytest.approx(parts, rel=1e-6)


def test_window_states_with_another_window_count_are_refused(recorded):
    """SYS-007: every ``window_states`` leaf has leading axis ``n_windows``."""
    gm, obs = recorded
    ws = init_window_states(obs, WINDOW)
    assert jax.tree.leaves(ws)[0].shape[0] == N // WINDOW
    for bad in (jax.tree.map(lambda x: jnp.concatenate([x, x[:1]]), ws),
                jax.tree.map(lambda x: x[:-1], ws)):
        with pytest.raises(ValueError, match="expected n_windows=4"):
            windowed_loss(gm, gm.params, obs, obs_fn=_position, window=WINDOW,
                          window_states=bad)


# ---------------------------------------------------------------------------
# fim
# ---------------------------------------------------------------------------


def _coupled():
    gm = GraphManager()
    for name, x0 in (("s", 0.5), ("t", 2.0)):
        gm.add_node(SpringDamperNode(name, 0.01, stiffness=30.0, damping=2.0, mass=1.0,
                                     rest_length=1.0, initial_position=x0))
    gm.add_edge("s", "t", "position", "anchor_position")
    gm.add_edge("t", "s", "position", "anchor_position")
    gm.add_coupling_group(["s", "t"], max_iterations=50, tolerance=1e-7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the coupled-pair stability advisory
        gm.compile()
    return gm


def test_fim_reaches_through_a_coupled_step():
    """SYS-021: ``jacfwd`` works through a coupled step "because the IFT rule is a
    custom_jvp": the matrix matches a central difference of the same rollout."""
    gm = _coupled()
    step_fn = gm._build_step_fn()
    ext = gm._default_external_inputs()
    init = gm._state
    names = ("damping", "stiffness")

    @jax.jit
    def residual(sub):
        p = jax.tree.map(lambda x: x, gm.params)
        for key, value in sub.items():
            p["nodes"]["s"][key] = value

        def body(s, _):
            s = step_fn(s, ext, p)
            return s, s["t"]["position"]

        return jax.lax.scan(body, init, None, length=20)[1]

    base = {k: gm.params["nodes"]["s"][k] for k in names}
    F = np.asarray(fim(residual, base, scale=None).fim, np.float64)
    jac = np.zeros((20, 2))
    for j, key in enumerate(names):
        h = 1e-2 * (1.0 + abs(float(base[key])))
        plus = {**base, key: jnp.float32(float(base[key]) + h)}
        minus = {**base, key: jnp.float32(float(base[key]) - h)}
        step = 0.5 * (float(plus[key]) - float(minus[key]))
        jac[:, j] = (np.asarray(residual(plus), np.float64)
                     - np.asarray(residual(minus), np.float64)) / (2 * step)
    F_fd = jac.T @ jac
    assert np.abs(F).max() > 0.0
    assert np.abs(F - F_fd).max() <= 2e-2 * np.abs(F).max(), (F, F_fd)


def _linear(scale=1.0):
    A = jnp.asarray([[1.0, 2.0], [0.5, -1.0], [2.0, 0.25]], jnp.float32)

    def residual(p):
        return scale * (A @ jnp.stack([p["a"], p["b"]]) - 1.0)

    return residual


def test_the_rank_verdict_does_not_move_when_the_residual_is_rescaled():
    """SYS-023: "Unlike ``cond`` the verdict does not move when the residual is rescaled
    (by ``noise_std``, say)" -- over thirty decades of sigma in float32, full rank and rank
    deficient alike."""
    full = {"a": jnp.float32(1.5), "b": jnp.float32(-0.5)}
    A = jnp.asarray([[1.0, 2.0], [2.0, 4.0], [0.5, 1.0]], jnp.float32)   # rank 1
    for sigma in (1e-15, 1e-6, 1.0, 1e6, 1e15):
        assert fim(_linear(), full, scale=None, noise_std=sigma).rank == 2, sigma
        r = fim(lambda p: A @ jnp.stack([p["a"], p["b"]]), full, scale=None, noise_std=sigma)
        assert r.rank == 1 and np.isinf(np.asarray(r.crb)).all(), sigma


def test_the_bound_is_in_the_parameters_units():
    """SYS-026: with ``noise_std`` the Cramer-Rao bound is "in the parameters' own units":
    sigma**2 times the unit-noise bound, ``c**2`` times as large for a parameter measured in
    units ``c`` times smaller under ``scale=None``, and the same under ``"relative"``."""
    p = {"a": jnp.float32(1.5), "b": jnp.float32(-0.5)}
    unit = np.asarray(fim(_linear(), p, scale=None).crb, np.float64)
    noisy = np.asarray(fim(_linear(), p, scale=None, noise_std=0.01).crb, np.float64)
    np.testing.assert_allclose(noisy, unit * 1e-4, rtol=1e-4)
    c = 1e3                                           # "a" now in units c times smaller

    def rescaled(q):
        return _linear()({"a": q["a"] / c, "b": q["b"]})

    q = {"a": jnp.float32(1.5 * c), "b": jnp.float32(-0.5)}
    np.testing.assert_allclose(np.asarray(fim(rescaled, q, scale=None).crb, np.float64),
                               unit * np.array([c ** 2, 1.0]), rtol=1e-3)
    np.testing.assert_allclose(np.asarray(fim(rescaled, q).crb, np.float64),
                               np.asarray(fim(_linear(), p).crb, np.float64), rtol=1e-3)


def test_fim_may_linearise_a_frozen_leaf():
    """SYS-033: "Unlike the fitters' ``mask`` this one is free to name a leaf the specs
    freeze" -- while ``fit`` refuses the same mask."""
    gm = _spring()
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
    obs = _record(gm, 20)
    mask = _only(gm, "mass")

    def residual(p):
        return _spring().run_scan_with_history(20, params=p)[1]["s"]["position"] \
            - obs["s"]["position"][1:]

    report = fim(residual, gm.params, mask=mask, scale=None)
    assert report.param_names == ("['nodes']['s']['mass']",) and report.rank == 1
    with pytest.raises(ValueError, match="trainable"):
        fit(gm, lambda p: jnp.sum(residual(p) ** 2), mask=mask, n_iter=1)


def test_fim_core_fails_closed_on_a_nan_matrix():
    """SYS-039: ``cond`` is ``+inf`` "when the smallest eigenvalue is not positive --
    including when it is NaN", ``finite`` is False, ``rank`` 0 and every ``crb`` ``+inf``;
    ``deciding_ratio`` is ``0.0`` (not NaN) when the verdict is not precision limited."""
    p = {"a": jnp.float32(1.0), "b": jnp.float32(2.0)}
    core = fim_core(lambda q: jnp.stack([q["a"] * jnp.nan, q["b"]]), p)
    assert np.isinf(float(core.cond)) and not bool(core.finite)
    assert int(core.rank) == 0 and np.isinf(np.asarray(core.crb)).all()
    ok = fim_core(_linear(), p)
    assert not bool(ok.precision_limited) and float(ok.deciding_ratio) == 0.0


def test_fim_core_takes_a_noise_model_closed_over_inside_jit():
    """SYS-041: inside ``jax.jit`` sigma "has to be a value the trace already holds -- a
    constant closed over, not an argument of the jitted function"."""
    p = {"a": jnp.float32(1.5), "b": jnp.float32(-0.5)}
    closed = jax.jit(lambda q: fim_core(_linear(), q, noise_std=0.5).crb)(p)
    eager = fim(_linear(), p, noise_std=0.5).crb
    np.testing.assert_allclose(np.asarray(closed), np.asarray(eager), rtol=1e-5)
    with pytest.raises(Exception):
        jax.jit(lambda q, s: fim_core(_linear(), q, noise_std=s).crb)(p, jnp.float32(0.5))


# ---------------------------------------------------------------------------
# The fitters
# ---------------------------------------------------------------------------


_RNG_A = np.random.default_rng(0).normal(size=(12, 3))


def _units_fit(unit, **kw):
    """The linear problem ``A x - b`` over (stiffness, damping, rest_length), damping (an
    identity-transform coordinate, bounds ``(0, None)``) measured in units ``unit``, from
    the same physical start; ``(result, physical damping, physical stiffness, loss at the
    returned params)``."""
    gm = _spring(initial_position=0.2)
    mask = _only(gm, "stiffness", "damping", "rest_length")
    b = _RNG_A @ np.array([40.0, 3.0, 1.2])
    u = jnp.asarray([1.0, unit, 1.0], jnp.float32)

    def residual(p):
        q = p["nodes"]["s"]
        x = jnp.stack([q["stiffness"], q["damping"], q["rest_length"]]) * u
        return jnp.asarray(_RNG_A, jnp.float32) @ x - jnp.asarray(b, jnp.float32)

    start = jax.tree.map(lambda x: x, gm.params)
    start["nodes"]["s"]["damping"] = jnp.asarray(2.0 / unit, jnp.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a declined hold says so
        res = fit_lm(gm, residual, params=start, mask=mask, n_iter=50, **kw)
    q = res.params["nodes"]["s"]
    return (res, float(q["damping"]) * unit, float(q["stiffness"]),
            0.5 * float(jnp.sum(residual(res.params) ** 2)))


@pytest.mark.parametrize("unit", [1e-5, 1e-3, 1e4, 1e6])
def test_the_marquardt_step_does_not_depend_on_the_units_of_a_parameter(unit):
    """SYS-084: the floor is ``eps`` times each column's own ``diag(JᵀJ)``, "so the step is
    the same for every scaling of the residual and of each parameter": the same problem,
    damping in other units, reaches the same answer in about as many iterations (the
    identifiability guard off -- it is SYS-063's)."""
    base, damping, stiffness, _ = _units_fit(1.0, hold_undetermined=False)
    res, damping_u, stiffness_u, _ = _units_fit(unit, hold_undetermined=False)
    assert base.converged and res.converged, (base, res)
    assert damping == pytest.approx(3.0, rel=1e-5) and stiffness == pytest.approx(40.0, rel=1e-5)
    assert damping_u == pytest.approx(3.0, rel=1e-4) and stiffness_u == pytest.approx(40.0, rel=1e-4)
    assert abs(res.n_iter - base.n_iter) <= 2, (base.n_iter, res.n_iter)


#: 1e-5 and 1e6 were held at a loss of 13.9 and 0.1 against 1e-11 while the guard asked
#: its questions in the optimiser's coordinates (SYS-063, MADD-ANO-131).
_GUARD_UNITS = [pytest.param(1e-3, id="1e-3"), pytest.param(1e-5, id="1e-5"),
                pytest.param(1e6, id="1e6")]


@pytest.mark.parametrize("unit", _GUARD_UNITS)
def test_the_hold_leaves_the_loss_where_the_fit_left_it_in_any_units(unit):
    """SYS-063: "So the loss is unaffected, beyond rounding, whatever the guard decides" --
    with damping in units 1e-5 or 1e6, where the data determines every direction."""
    res, _, _, returned = _units_fit(unit)
    assert res.best_loss < 1e-9, res
    assert returned <= res.best_loss + 1e-9, (returned, res.best_loss, res.excited_rank,
                                               res.hold_declined, res.undetermined_drift)


#: Damping came back at 2 (its start) and 3.085 against 3 at 1e-5 and 1e6 while the
#: guard asked its questions in the optimiser's coordinates (SYS-088, MADD-ANO-131).
_ANSWER_UNITS = [pytest.param(1e-3, id="1e-3"), pytest.param(1e-5, id="1e-5"),
                 pytest.param(1e6, id="1e6")]


@pytest.mark.parametrize("unit", _ANSWER_UNITS)
def test_fit_lm_answers_the_same_in_any_units_with_its_defaults(unit):
    """SYS-088: "The solve's floor is eps times each column's own curvature, so the answer
    depends neither on the residual's units nor on any parameter's" -- fit_lm as called by
    default, guard included."""
    _, damping, stiffness, _ = _units_fit(1.0)
    _, damping_u, stiffness_u, _ = _units_fit(unit)
    assert damping == pytest.approx(3.0, rel=1e-5)
    assert damping_u == pytest.approx(damping, rel=1e-4), (damping_u, damping)
    assert stiffness_u == pytest.approx(stiffness, rel=1e-4), (stiffness_u, stiffness)


def test_with_the_default_tol_fit_never_reports_converged():
    """SYS-053: "``tol=0.0``, the default, turns this test off" -- even at a loss of exactly
    zero, where ``losses[-1] <= tol`` would hold."""
    gm = _spring()
    mask = _only(gm, "stiffness")
    res = fit(gm, lambda p: 0.0 * p["nodes"]["s"]["stiffness"], mask=mask, n_iter=3)
    assert [float(x) for x in res.losses] == [0.0, 0.0, 0.0] and res.converged is False


def test_losses_zero_is_the_starting_points_loss():
    """SYS-050: ``losses[0]`` is the loss of the start, to the last bits a ``log`` leaf's
    round trip through the optimiser's coordinates moves it."""
    gm = _spring()
    mask = _only(gm, "stiffness")

    def loss(p):
        return (p["nodes"]["s"]["stiffness"] - 25.0) ** 2

    res = fit(gm, loss, mask=mask, n_iter=4, lr=0.1)
    assert float(res.losses[0]) == pytest.approx(float(loss(gm.params)), rel=4e-6)

    def residual(p):
        return jnp.atleast_1d(p["nodes"]["s"]["stiffness"] - 25.0)

    lm = fit_lm(gm, residual, mask=mask, n_iter=4)
    assert float(lm.losses[0]) == pytest.approx(0.5 * float(residual(gm.params)[0]) ** 2,
                                                rel=4e-6)


def test_multiple_shooting_best_loss_is_the_returned_pairs_within_the_guards_tolerance():
    """SYS-071: ``best_loss`` is the loss of the window states and parameters returned --
    exactly without the guard, and within the guard's tolerance when it held a direction."""
    gm = _spring()
    gm.set_param_spec("s", "damping", ParamSpec(bounds=(0.0, None), transform="log"))
    gm.set_param_spec("s", "mass", ParamSpec(bounds=(0.0, None), transform="log"))
    obs = _record(gm)
    start = jax.tree.map(lambda x: x, gm.params)
    start["nodes"]["s"]["stiffness"] = jnp.float32(36.0)
    mask = _only(gm, "stiffness", "damping", "mass")
    for hold in (False, True):
        res, ws = fit_multiple_shooting(gm, obs, obs_fn=_position, window=WINDOW, params=start,
                                        mask=mask, n_iter=60, lr=0.05, hold_undetermined=hold)
        again = float(windowed_loss(gm, res.params, obs, obs_fn=_position, window=WINDOW,
                                    window_states=ws, continuity_weight=1.0))
        if hold:
            assert res.excited_rank == 2 and res.undetermined_drift > 0.0, res
            assert again == pytest.approx(res.best_loss, rel=2.0 ** 10 * 1.2e-7)
        else:
            assert again == pytest.approx(res.best_loss, rel=1e-6)


def test_the_spring_declares_its_parameter_specs():
    """SYS-106: "``SpringDamperNode``: stiffness and mass are ``log``-positive, damping is
    ``>= 0``"."""
    specs = _spring().param_specs()["nodes"]["s"]
    assert specs["stiffness"].transform == "log" and specs["stiffness"].bounds == (0.0, None)
    assert specs["mass"].transform == "log" and specs["mass"].bounds == (0.0, None)
    assert specs["damping"].transform is None and specs["damping"].bounds == (0.0, None)
    assert specs["initial_position"].trainable is False


def test_fim_core_cannot_be_built_positionally():
    """SYS-037: ``FIMCore`` is "Keyword-only for the reason :class:`FIMReport` is"."""
    from maddening.sysid import FIMCore

    core = fim_core(_linear(), {"a": jnp.float32(1.5), "b": jnp.float32(-0.5)})
    fields = [getattr(core, name) for name in ("fim", "eigvals", "eigvecs", "rank", "cond")]
    with pytest.raises(TypeError):
        FIMCore(*fields)


# ---------------------------------------------------------------------------
# ParamSpec bounds at zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [-1e-45, -1e-40, -1e-38])
def test_a_value_below_a_zero_bound_is_refused_however_small(value):
    """SYS-109: ``check`` raises "if a concrete value ... violates bounds", and
    ``gm.check_params`` names "the first leaf out of range".  A negative float32 subnormal
    is below the lower bound 0 of ``SpringDamperNode``'s ``damping``.  (It passed while
    ``check`` compared through ``jnp``, which flushes it to zero on CPU: MADD-ANO-132.)"""
    with pytest.raises(ValueError, match="below bound"):
        ParamSpec(bounds=(0.0, None)).check(np.float32(value), name="damping")


def test_a_value_below_a_zero_bound_by_a_normal_amount_is_refused():
    """SYS-105's edge on the normal side: the smallest negative normal float32 is refused,
    by ``check`` and by ``gm.check_params``."""
    with pytest.raises(ValueError, match="below bound"):
        ParamSpec(bounds=(0.0, None)).check(-np.finfo(np.float32).tiny, name="damping")
    gm = _spring()
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s"]["damping"] = jnp.float32(-np.finfo(np.float32).tiny)
    with pytest.raises(ValueError, match=r"\['damping'\].*below bound 0.0"):
        gm.check_params(p)


# ---------------------------------------------------------------------------
# fim and fim_core near the rank cutoff
# ---------------------------------------------------------------------------


def _near_cutoff(rng, factor, n=3, m=8):
    """A residual ``J p`` whose ``JᵀJ`` has eigenvalues from 1 down to 0.5 and one at
    ``factor`` times the default rank cutoff, in float32."""
    eps = float(np.finfo(np.float32).eps)
    cutoff = max(n, np.sqrt(m)) * eps
    eig = np.concatenate([np.linspace(1.0, 0.5, n - 1), [factor * cutoff]])
    U, _ = np.linalg.qr(rng.normal(size=(m, n)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    J = jnp.asarray(U @ np.diag(np.sqrt(eig)) @ V.T, jnp.float32)
    return lambda p: J @ p["p"]


# Per push: tests/core/test_sysid_fim_core.py::TestCoreAgreesWithReport::test_rank_and_crb_infinities_match_fim
@pytest.mark.slow
def test_fim_and_fim_core_disagree_only_within_five_times_the_cutoff():
    """SYS-043: "they reach a different (rank, precision_limited) on 0.39% of them, never
    above five times the rank cutoff" -- on matrices whose deciding eigenvalue is spread
    log-uniformly from a hundredth to a hundred times the cutoff."""
    rng = np.random.default_rng(11)
    disagreements, limited_count = [], 0
    for n, m in ((3, 8), (3, 512), (6, 512), (6, 4096)):
        params = {"p": jnp.ones(n, jnp.float32)}
        for _ in range(150):
            factor = float(10.0 ** rng.uniform(-2.0, 2.0))
            residual = _near_cutoff(rng, factor, n, m)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                report = fim(residual, params, scale=None)
            limited = any("PrecisionLimit" in type(w.message).__name__ for w in caught)
            limited_count += limited
            core = fim_core(residual, params, scale=None)
            if (report.rank, limited) != (int(core.rank), bool(core.precision_limited)):
                disagreements.append((n, m, factor))
    assert limited_count > 0, "the population must reach the precision band"
    assert all(0.2 <= f <= 5.0 for _, _, f in disagreements), disagreements
