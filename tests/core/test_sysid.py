"""maddening.sysid: windowed teacher-forced loss and Fisher information."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import fim, fit, observations_from_history, windowed_loss

K_TRUE, C_TRUE = 30.0, 2.0
N_STEPS, WINDOW = 200, 20


def _spring_gm(k=K_TRUE, c=C_TRUE, coupled=False):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=k, damping=c, mass=1.0,
                                 rest_length=1.0, initial_position=0.5))
    if coupled:
        gm.add_node(SpringDamperNode("t", 0.01, stiffness=k, damping=c, mass=1.0,
                                     rest_length=1.0, initial_position=3.0))
        gm.add_edge("s", "t", "position", "anchor_position")
        gm.add_edge("t", "s", "position", "anchor_position")
        gm.add_coupling_group(["s", "t"], max_iterations=20, tolerance=1e-6)
    gm.compile()
    return gm


def _observations(gm):
    init = {n: gm.get_node_state(n) for n in gm.node_names}
    _, hist = gm.run_scan_with_history(N_STEPS)
    return observations_from_history(init, hist)


@pytest.fixture(scope="module")
def spring():
    gm = _spring_gm()
    obs = _observations(gm)
    return gm, obs


def _loss_fn(gm, obs, **kw):
    return jax.jit(lambda p: windowed_loss(
        gm, p, obs, obs_fn=lambda h: h["s"]["position"], window=WINDOW, **kw,
    ))


def _with_k(params, k):
    p = jax.tree.map(lambda x: x, params)
    p["nodes"]["s"]["stiffness"] = jnp.asarray(k, dtype=jnp.float32)
    return p


def test_loss_zero_at_truth_positive_elsewhere(spring):
    gm, obs = spring
    loss = _loss_fn(gm, obs)
    assert float(loss(gm.params)) < 1e-9
    assert float(loss(_with_k(gm.params, 1.2 * K_TRUE))) > 1e-4


def test_loss_gradient_finite_and_nonzero(spring):
    gm, obs = spring
    loss = _loss_fn(gm, obs)
    g = jax.grad(loss)(_with_k(gm.params, 1.2 * K_TRUE))["nodes"]["s"]["stiffness"]
    assert bool(jnp.isfinite(g)) and float(g) > 0.0  # k too high -> increase loss


def test_recovers_stiffness_from_20_percent_perturbation(spring):
    gm, obs = spring
    loss = _loss_fn(gm, obs)
    grad = jax.jit(jax.grad(loss))
    k = jnp.asarray(1.2 * K_TRUE, dtype=jnp.float32)
    m = v = jnp.zeros(())
    b1, b2, lr = 0.9, 0.999, 0.5
    for i in range(1, 301):
        g = grad(_with_k(gm.params, k))["nodes"]["s"]["stiffness"]
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        k = k - lr * (m / (1 - b1 ** i)) / (jnp.sqrt(v / (1 - b2 ** i)) + 1e-8)
    assert abs(float(k) - K_TRUE) / K_TRUE < 0.05, float(k)


def test_windows_must_tile_the_observations(spring):
    gm, obs = spring
    with pytest.raises(ValueError, match="must divide"):
        windowed_loss(gm, gm.params, obs, obs_fn=lambda h: h["s"]["position"], window=7)


def _truncated(obs, field, n):
    return {"s": {**obs["s"], field: obs["s"][field][:n]}}


@pytest.mark.parametrize("short_field", ["velocity", "position"])
def test_observation_leaves_of_unequal_length_are_refused(spring, short_field):
    """``T`` used to come from the first leaf alone.  A *later* leaf that
    was shorter was read past its end -- ``dynamic_slice`` clamps, so the
    late windows compared against its last sample repeated and the loss
    was finite and non-zero at the true parameters -- and a *first* leaf
    that was shorter set ``T`` and silently dropped the others' tail.
    ``position`` sorts before ``velocity``, so the two cases are the two
    orders.  Both lengths below tile with ``WINDOW``, so only the length
    check can refuse them."""
    gm, obs = spring
    ragged = _truncated(obs, short_field, N_STEPS - WINDOW + 1)
    with pytest.raises(ValueError, match="disagree on the leading axis") as exc:
        windowed_loss(gm, gm.params, ragged, obs_fn=lambda h: h["s"]["position"],
                      window=WINDOW)
    msg = str(exc.value)
    assert f"{N_STEPS + 1}: " in msg and f"{N_STEPS - WINDOW + 1}: " in msg, msg
    assert "'s.position'" in msg and "'s.velocity'" in msg, msg
    with pytest.raises(ValueError, match="disagree on the leading axis"):
        init_window_states(ragged, WINDOW)


def test_window_states_of_unequal_length_are_refused(spring):
    """The multiple-shooting starts are indexed per window with the same
    clamping ``dynamic_index``: a leaf with too few entries would restart
    its late windows from its last entry."""
    gm, obs = spring
    ws = init_window_states(obs, WINDOW)
    ragged = {"s": {**ws["s"], "velocity": ws["s"]["velocity"][:-1]}}
    with pytest.raises(ValueError,
                       match="window_states leaves disagree on the leading axis"):
        windowed_loss(gm, gm.params, obs, obs_fn=lambda h: h["s"]["position"],
                      window=WINDOW, window_states=ragged)


def test_a_scalar_observation_leaf_is_refused_by_name(spring):
    gm, obs = spring
    bad = {"s": {**obs["s"], "velocity": obs["s"]["velocity"][0]}}
    with pytest.raises(ValueError, match=r"'s\.velocity' is a scalar"):
        windowed_loss(gm, gm.params, bad, obs_fn=lambda h: h["s"]["position"],
                      window=WINDOW)


def test_mask_unconverged_through_coupled_group():
    gm = _spring_gm(coupled=True)
    obs = _observations(gm)
    loss = jax.jit(lambda p: windowed_loss(
        gm, p, obs, obs_fn=lambda h: (h["s"]["position"], h["t"]["position"]),
        window=WINDOW, mask_unconverged=True,
    ))
    assert float(loss(gm.params)) < 1e-6
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s"]["stiffness"] = jnp.asarray(1.3 * K_TRUE, dtype=jnp.float32)
    g = jax.grad(loss)(p)["nodes"]["s"]["stiffness"]
    assert bool(jnp.isfinite(g)) and float(g) != 0.0


def _residual_fn(gm, obs, names):
    """Residual over a sub-pytree of the spring's params."""
    step_fn = gm._build_step_fn()
    ext = gm._default_external_inputs()
    init = jax.tree.map(lambda x: x[0], obs)
    truth = obs["s"]["position"][1:]

    def residual(sub):
        p = jax.tree.map(lambda x: x, gm.params)
        for n in names:
            p["nodes"]["s"][n] = sub[n]

        def body(s, _):
            s = step_fn(s, ext, p)
            return s, s["s"]["position"]

        _, pos = jax.lax.scan(body, init, None, length=N_STEPS)
        return pos - truth

    return residual


def test_fim_identifiable_pair_has_finite_condition_number(spring):
    gm, obs = spring
    names = ("stiffness", "damping")
    sub = {n: gm.params["nodes"]["s"][n] for n in names}
    rep = fim(_residual_fn(gm, obs, names), sub)
    assert rep.param_names == ("['damping']", "['stiffness']")
    assert np.isfinite(rep.cond) and rep.cond < 1e6, rep.cond
    assert rep.rank == 2, rep.eigvals
    assert bool(jnp.all(jnp.isfinite(rep.crb)))


def test_fim_flags_unidentifiable_scale_direction(spring):
    """With position-only data, k, c, m enter only as k/m and c/m: scaling
    all three together is invisible.  In relative coordinates that is the
    direction (1, 1, 1)/sqrt(3), and its eigenvalue must be ~0.

    All three parameters lie partly along that direction, so none of
    them has a finite Cramer-Rao bound: the data pins two combinations
    of k, c and m and leaves the overall scale free.
    """
    gm, obs = spring
    names = ("stiffness", "damping", "mass")
    sub = {n: gm.params["nodes"]["s"][n] for n in names}
    rep = fim(_residual_fn(gm, obs, names), sub)
    ratio = float(rep.eigvals[0] / rep.eigvals[-1])
    assert ratio < 1e-4, ratio
    v = np.asarray(rep.eigvecs[:, 0])
    assert abs(abs(v @ np.ones(3) / np.sqrt(3.0))) > 0.99, v
    assert rep.cond > 1e4
    assert rep.rank == 2, rep.eigvals
    assert bool(jnp.all(jnp.isinf(rep.crb))), rep.crb


# ---------------------------------------------------------------------------
# fit / fim under ParamSpec
# ---------------------------------------------------------------------------


def _perturbed(gm, k_factor, c_factor):
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s"]["stiffness"] = jnp.asarray(k_factor * K_TRUE, jnp.float32)
    p["nodes"]["s"]["damping"] = jnp.asarray(c_factor * C_TRUE, jnp.float32)
    return p


def test_fit_recovers_k_c_with_mass_frozen_by_spec(spring):
    gm, obs = spring
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
    loss = _loss_fn(gm, obs)
    seen = []
    res = fit(gm, loss, params=_perturbed(gm, 1.5, 2.5), n_iter=300, lr=0.1,
              callback=lambda i, l, p: seen.append(float(p["nodes"]["s"]["stiffness"])))
    s = res.params["nodes"]["s"]
    assert abs(float(s["stiffness"]) - K_TRUE) / K_TRUE < 0.05, float(s["stiffness"])
    assert abs(float(s["damping"]) - C_TRUE) / C_TRUE < 0.10, float(s["damping"])
    assert float(s["mass"]) == 1.0                       # frozen, bit-identical
    assert float(s["initial_position"]) == 0.5           # initial_* never move
    assert res.losses[-1] < 1e-3 * res.losses[0]
    assert res.n_iter == 300 and not res.converged       # tol=0: ran to n_iter
    assert min(seen) > 0.0                               # log transform: never <= 0


def test_fit_mask_overrides_specs_and_tol_stops_early(spring):
    gm, obs = spring
    loss = _loss_fn(gm, obs)
    mask = jax.tree.map(lambda _: False, gm.params)
    mask["nodes"]["s"]["stiffness"] = True
    start = _perturbed(gm, 1.3, 1.0)
    res = fit(gm, loss, params=start, mask=mask, n_iter=400, lr=0.1, tol=1e-6)
    s = res.params["nodes"]["s"]
    assert float(s["damping"]) == float(start["nodes"]["s"]["damping"])
    assert abs(float(s["stiffness"]) - K_TRUE) / K_TRUE < 0.02
    assert res.converged and res.n_iter < 400
    assert res.losses[-1] <= 1e-6


def test_fit_rejects_start_outside_bounds(spring):
    gm, obs = spring
    bad = _perturbed(gm, -1.0, 1.0)      # stiffness < 0 with a log spec
    with pytest.raises(ValueError, match="stiffness"):
        fit(gm, _loss_fn(gm, obs), params=bad, n_iter=1)


def test_fit_raises_on_non_finite_gradient(spring):
    gm, obs = spring
    with pytest.raises(FloatingPointError):
        fit(gm, lambda p: jnp.sqrt(p["nodes"]["s"]["stiffness"] - 30.0), n_iter=3)


def test_fim_mask_restricts_to_selected_leaves(spring):
    """``fim`` over the full pytree with the graph's trainable mask equals
    ``fim`` over the hand-built sub-pytree of the same leaves."""
    gm, obs = spring
    gm._param_spec_overrides.clear()  # noqa: SLF001 - module-scoped fixture
    for key in gm.params["nodes"]["s"]:
        if key not in ("stiffness", "damping"):
            gm.set_param_spec("s", key, ParamSpec(trainable=False))
    step_fn = gm._build_step_fn()
    ext = gm._default_external_inputs()
    init = jax.tree.map(lambda x: x[0], obs)
    truth = obs["s"]["position"][1:]

    def residual_full(p):
        def body(s, _):
            s = step_fn(s, ext, p)
            return s, s["s"]["position"]
        return jax.lax.scan(body, init, None, length=N_STEPS)[1] - truth

    rep = fim(residual_full, gm.params, mask=gm.trainable_mask())
    ref = fim(_residual_fn(gm, obs, ("stiffness", "damping")),
              {n: gm.params["nodes"]["s"][n] for n in ("stiffness", "damping")})
    assert rep.fim.shape == (2, 2)
    assert rep.param_names == ("['nodes']['s']['damping']", "['nodes']['s']['stiffness']")
    assert np.allclose(np.asarray(rep.fim), np.asarray(ref.fim), rtol=1e-5)
    with pytest.raises(ValueError, match="selects no parameters"):
        fim(residual_full, gm.params, mask=jax.tree.map(lambda _: False, gm.params))
    gm._param_spec_overrides.clear()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Items 9-11: multiple shooting, noise model + LM, progress events
# ---------------------------------------------------------------------------

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from maddening.sysid import (  # noqa: E402
    fit_lm, fit_multiple_shooting, init_window_states,
)


def test_init_window_states_shape_and_values(spring):
    gm, obs = spring
    ws = init_window_states(obs, WINDOW)
    n_w = N_STEPS // WINDOW
    assert ws["s"]["position"].shape == (n_w,)
    np.testing.assert_array_equal(np.asarray(ws["s"]["position"]),
                                  np.asarray(obs["s"]["position"][: n_w * WINDOW : WINDOW]))
    with pytest.raises(ValueError, match="must divide"):
        init_window_states(obs, 7)


def test_multiple_shooting_loss_zero_at_truth_and_penalises_gaps(spring):
    gm, obs = spring
    ws = init_window_states(obs, WINDOW)
    loss = jax.jit(lambda p, w: windowed_loss(
        gm, p, obs, obs_fn=lambda h: h["s"]["position"], window=WINDOW,
        window_states=w, continuity_weight=1.0))
    assert float(loss(gm.params, ws)) < 1e-9
    # A wrong free start on window 1 costs both data misfit and continuity.
    bad = jax.tree.map(lambda x: x, ws)
    bad["s"]["position"] = bad["s"]["position"].at[1].add(0.5)
    assert float(loss(gm.params, bad)) > 1e-2
    # the continuity term alone (data term unaffected by a moved *end*):
    g = jax.grad(lambda w: loss(gm.params, w))(ws)
    assert bool(jnp.all(jnp.isfinite(g["s"]["position"])))
    with pytest.raises(ValueError, match="leading axis"):
        windowed_loss(gm, gm.params, obs, obs_fn=lambda h: h["s"]["position"],
                      window=WINDOW, window_states=jax.tree.map(lambda x: x[:2], ws))


def test_fit_multiple_shooting_recovers_from_noisy_window_starts(spring):
    """With noise on the observations, teacher forcing seeds every window
    with the noisy state; multiple shooting lets the starts move."""
    gm, obs = spring
    rng = np.random.default_rng(0)
    noisy = jax.tree.map(lambda x: x + jnp.asarray(rng.normal(0, 0.01, x.shape), x.dtype), obs)
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
    gm.set_param_spec("s", "rest_length", ParamSpec(trainable=False))
    res, ws = fit_multiple_shooting(
        gm, noisy, obs_fn=lambda h: h["s"]["position"], window=WINDOW,
        params=_perturbed(gm, 1.3, 1.5), continuity_weight=1.0, n_iter=250, lr=0.1,
        lr_states=0.01,
    )
    k = float(res.params["nodes"]["s"]["stiffness"])
    assert abs(k - K_TRUE) / K_TRUE < 0.05, k
    assert ws["s"]["position"].shape == (N_STEPS // WINDOW,)
    assert res.losses[-1] < res.losses[0]
    gm._param_spec_overrides.clear()  # noqa: SLF001


def test_fim_noise_std_scales_information(spring):
    gm, obs = spring
    names = ("stiffness", "damping")
    sub = {n: gm.params["nodes"]["s"][n] for n in names}
    base = fim(_residual_fn(gm, obs, names), sub)
    scaled = fim(_residual_fn(gm, obs, names), sub, noise_std=2.0)
    np.testing.assert_allclose(np.asarray(scaled.fim), np.asarray(base.fim) / 4.0, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(scaled.crb), np.asarray(base.crb) * 4.0, rtol=1e-4)
    # per-leaf pytree sigma (residual is a single array here)
    per = fim(_residual_fn(gm, obs, names), sub, noise_std=jnp.full((N_STEPS,), 2.0))
    np.testing.assert_allclose(np.asarray(per.fim), np.asarray(scaled.fim), rtol=1e-5)


def _full_residual(gm, obs):
    step_fn = gm._build_step_fn()
    ext = gm._default_external_inputs()
    init = jax.tree.map(lambda x: x[0], obs)
    truth = obs["s"]["position"][1:]

    def residual(p):
        def body(s, _):
            s = step_fn(s, ext, p)
            return s, s["s"]["position"]
        return jax.lax.scan(body, init, None, length=N_STEPS)[1] - truth
    return residual


@pytest.mark.slow  # an LM fit over a rollout: 7-10 s on CI
@given(kf=st.floats(0.5, 2.0), cf=st.floats(0.5, 2.0))
# Absolute at the house floor, not ``EXAMPLES_COSTLY``: one example is
# a 30-iteration Levenberg-Marquardt fit over a full rollout, seconds
# rather than milliseconds.  Tiering it would let the ``ci`` profile
# turn this single test into minutes; depth here is bought from the
# cheaper properties instead.
@settings(max_examples=20, deadline=None)
def test_fit_lm_recovers_k_c_in_few_iterations(kf, cf):
    gm = _spring_gm()
    obs = _observations(gm)
    for key in ("mass", "rest_length"):
        gm.set_param_spec("s", key, ParamSpec(trainable=False))
    res = fit_lm(gm, _full_residual(gm, obs), params=_perturbed(gm, kf, cf), n_iter=30,
                 tol=1e-10)
    s = res.params["nodes"]["s"]
    assert abs(float(s["stiffness"]) - K_TRUE) / K_TRUE < 0.02, float(s["stiffness"])
    assert abs(float(s["damping"]) - C_TRUE) / C_TRUE < 0.05, float(s["damping"])
    assert res.n_iter <= 30 and res.losses[-1] <= res.losses[0]
    assert float(s["mass"]) == 1.0


def test_fit_lm_beats_adam_at_equal_budget(spring):
    gm, obs = spring
    for key in ("mass", "rest_length"):
        gm.set_param_spec("s", key, ParamSpec(trainable=False))
    start = _perturbed(gm, 1.5, 2.5)
    lm = fit_lm(gm, _full_residual(gm, obs), params=start, n_iter=15)
    adam = fit(gm, _loss_fn(gm, obs), params=start, n_iter=15, lr=0.1)
    k_lm = float(lm.params["nodes"]["s"]["stiffness"])
    k_adam = float(adam.params["nodes"]["s"]["stiffness"])
    assert abs(k_lm - K_TRUE) < abs(k_adam - K_TRUE)
    gm._param_spec_overrides.clear()  # noqa: SLF001


def test_fit_progress_events_reach_observers(spring):
    from maddening.core.graph_manager import EVENT_FIT_PROGRESS

    gm, obs = spring
    seen = []
    gm.add_observer(lambda ev, data: seen.append((ev, data)) if ev == EVENT_FIT_PROGRESS else None)
    fit(gm, _loss_fn(gm, obs), params=_perturbed(gm, 1.2, 1.0), n_iter=6, lr=0.1,
        notify_every=2)
    gm._observers.clear()  # noqa: SLF001
    its = [d["iteration"] for _, d in seen]
    assert its == [2, 4, 6]
    assert all(d["method"] == "adam" and d["n_iter"] == 6 for _, d in seen)
    assert all(np.isfinite(d["loss"]) and "s" in d["params"]["nodes"] for _, d in seen)
    # notify_every=0 disables; fit_lm reports method "lm"
    seen.clear()
    gm.add_observer(lambda ev, data: seen.append(data["method"]) if ev == EVENT_FIT_PROGRESS else None)
    fit(gm, _loss_fn(gm, obs), n_iter=2, notify_every=0)
    assert seen == []
    fit_lm(gm, _full_residual(gm, obs), n_iter=2)
    gm._observers.clear()  # noqa: SLF001
    assert seen == ["lm", "lm"]


def test_an_nd_leaf_is_labelled_by_the_index_of_the_element():
    """A 6x6 leaf whose only unobserved element is ``H[0, 5]``.  The
    label used to be the flat position, ``['H'][5]`` -- which, read as
    the NumPy index it looks like, is the whole of row 5.  Each label
    must now *be* the index of its element, in the row-major order the
    columns are in."""
    H = jnp.ones((6, 6), jnp.float32)
    W = jnp.ones((6, 6)).at[0, 5].set(0.0)
    X = jnp.asarray(np.random.default_rng(0).standard_normal((36, 50)),
                    jnp.float32)
    rep = fim(lambda p: (p["H"] * W).reshape(-1) @ X, {"H": H}, scale=None)
    name, weight = rep.least_identifiable()
    assert name == "['H'][0, 5]"
    assert weight == pytest.approx(1.0)
    assert rep.param_names[:7] == tuple(
        f"['H'][0, {j}]" for j in range(6)) + ("['H'][1, 0]",)
    assert len(set(rep.param_names)) == 36
    # A 3-D leaf and a 1-D one side by side: every label is the index.
    params = {"T": jnp.zeros((2, 1, 2)), "v": jnp.zeros(3)}
    rep = fim(lambda p: jnp.concatenate([p["T"].reshape(-1), p["v"]]) * 2.0,
              params, scale=None)
    assert rep.param_names == (
        "['T'][0, 0, 0]", "['T'][0, 0, 1]", "['T'][1, 0, 0]", "['T'][1, 0, 1]",
        "['v'][0]", "['v'][1]", "['v'][2]")
