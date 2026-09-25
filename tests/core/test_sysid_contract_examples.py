"""The sysid contract on fixed examples, on every push.

``tests/property/test_sysid_contract.py`` states the contract of
:mod:`maddening.sysid` (and of the deprecated calibration path) over
generated inputs.  Most of those properties compile a graph, a fit or a
Jacobian per example and are slow-marked, so they run only in the slow
lane.  Where a per-push test elsewhere already checks the same property,
the slow mark names it; this module holds a fixed example of each of the
rest, so that breaking one fails the push that breaks it.  The generated
property stays the stronger check: these pin one case each.

Not here: the two ``tune_coupling_params`` properties.  That function is
an eager grid search that builds and compiles a coupled graph per
configuration (a two-point grid took 29 s cold), and it is deprecated for
removal in 0.5.0.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import (
    fim,
    fit,
    fit_multiple_shooting,
    init_window_states,
    observations_from_history,
    windowed_loss,
)

DT = 0.01
REST = 1.0


def _spring_gm(stiffness=30.0, damping=2.0, mass=1.0, position=0.5):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, stiffness=stiffness, damping=damping,
                                 mass=mass, rest_length=REST,
                                 initial_position=position))
    gm.compile()
    return gm


def _with_params(gm, values):
    p = jax.tree.map(lambda x: x, gm.params)
    for k, v in values.items():
        p["nodes"]["s"][k] = jnp.asarray(v, dtype=jnp.float32)
    return p


def _observe(gm, n_steps, params, sample_every=1):
    init = {n: gm.get_node_state(n) for n in gm.node_names}
    _, hist = gm.run_scan_with_history(n_steps, params=params)
    obs = observations_from_history(init, hist)
    return jax.tree.map(lambda x: x[::sample_every], obs)


def _position(h):
    return h["s"]["position"]


@contextlib.contextmanager
def _x64():
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


# ---------------------------------------------------------------------------
# The trainable contract and the bounds
# ---------------------------------------------------------------------------


def test_multiple_shooting_leaves_a_spec_frozen_leaf_bit_identical():
    """``test_every_fitter_honours_the_same_frozen_set``, for the fitter with
    no per-push check of it: ``fit_multiple_shooting`` shares
    ``_masked_indices`` with ``fit`` and ``fit_lm`` and must leave a leaf
    whose spec says ``trainable=False`` exactly where it started."""
    gm = _spring_gm()
    frozen = ("damping", "rest_length")
    for key in frozen:
        gm.set_param_spec("s", key, ParamSpec(trainable=False))
    obs = _observe(gm, 12, gm.params)
    start = _with_params(gm, {"stiffness": 45.0, "damping": 3.0})
    res, _ = fit_multiple_shooting(gm, obs, obs_fn=_position, window=4, params=start,
                                   n_iter=3, lr=0.05)
    before = {k: float(v) for k, v in start["nodes"]["s"].items()}
    after = {k: float(v) for k, v in res.params["nodes"]["s"].items()}
    for key in (*frozen, "initial_position"):   # the last is frozen by the node itself
        assert after[key] == before[key], (key, before[key], after[key])
    assert after["stiffness"] != before["stiffness"], "the fit moved nothing"


def test_a_logit_leaf_pushed_either_way_finishes_inside_its_bounds():
    """``test_a_fit_that_starts_inside_the_bounds_finishes_inside_them`` for
    the ``logit`` transform, pushed at both bounds.  (An untransformed leaf
    pushed at its bounds is checked on every push in
    ``tests/property/test_sysid_contract.py``.)"""
    lo, hi = 2.0, 10.0
    gm = _spring_gm(stiffness=0.5 * (lo + hi))
    gm.set_param_spec("s", "stiffness", ParamSpec(bounds=(lo, hi), transform="logit"))
    for key in ("damping", "mass", "rest_length"):
        gm.set_param_spec("s", key, ParamSpec(trainable=False))
    for direction in (-1.0, 1.0):
        # Linear: the gradient never vanishes, so Adam keeps pushing.
        res = fit(gm, lambda p: direction * 50.0 * p["nodes"]["s"]["stiffness"],
                  n_iter=6, lr=0.5)
        gm.check_params(res.params)
        value = float(res.params["nodes"]["s"]["stiffness"])
        assert lo <= value <= hi, (direction, value)
        # Non-vacuity: it moved towards the bound it was pushed at.
        assert (value > 6.0) if direction < 0 else (value < 6.0), (direction, value)


# ---------------------------------------------------------------------------
# The Fisher matrix against a finite difference
# ---------------------------------------------------------------------------

_A = np.array([[0.7, -0.3], [0.2, 0.9], [-0.5, 0.4]])
_B = np.array([[1.1, 0.6], [0.8, 1.3], [1.4, 0.9]])


def _analytic(p):
    th = jnp.stack([p["p0"], p["p1"]])
    return jnp.sum(jnp.asarray(_A) * jnp.sin(jnp.asarray(_B) * th[None, :]), axis=1)


def test_fim_matches_a_central_finite_difference_and_its_bound_is_the_inverse():
    """``test_fim_matches_a_central_finite_difference`` and
    ``test_crb_is_consistent_with_the_matrix_it_came_from`` on one smooth
    residual in float64, where a central difference is a precise oracle
    (see that module's docstring for the tolerance)."""
    with _x64():
        params = {"p0": jnp.float64(1.2), "p1": jnp.float64(0.8)}
        report = fim(_analytic, params, scale=None)
        jac = np.zeros((3, 2))
        for j, key in enumerate(("p0", "p1")):           # ``ravel_pytree`` order
            h = 1e-5 * (1.0 + abs(float(params[key])))
            plus = {**params, key: params[key] + h}
            minus = {**params, key: params[key] - h}
            jac[:, j] = (np.asarray(_analytic(plus)) - np.asarray(_analytic(minus))) / (2 * h)
        F = np.asarray(report.fim, dtype=np.float64)
        F_fd = jac.T @ jac
        assert np.allclose(F, F_fd, rtol=1e-7, atol=1e-7 * max(1.0, float(np.abs(F).max()))), (
            F, F_fd)
        # Full rank here, so the bound is the plain inverse, and a parameter
        # is never easier to estimate jointly than alone.
        crb = np.asarray(report.crb, dtype=np.float64)
        assert report.rank == 2
        assert np.allclose(crb, np.diag(np.linalg.inv(F)), rtol=1e-6, atol=1e-9)
        assert np.all(crb >= (1.0 / np.diag(F)) * (1 - 1e-6)), (crb, 1.0 / np.diag(F))


def test_fim_matches_a_finite_difference_of_a_spring_rollout():
    """``test_fim_matches_a_finite_difference_of_a_rollout`` on one spring,
    at the tolerance float32 supports there (2e-2 of the matrix's scale)."""
    gm = _spring_gm()
    truth = _with_params(gm, {"stiffness": 60.0, "damping": 1.5, "mass": 1.2})
    obs = _observe(gm, 20, truth)
    assert float(jnp.var(obs["s"]["position"])) > 1e-2, "the rollout barely moved"
    names = ("damping", "stiffness")                      # ``ravel_pytree`` order
    base = {k: truth["nodes"]["s"][k] for k in names}
    step_fn = gm._build_step_fn()          # noqa: SLF001
    ext = gm._default_external_inputs()    # noqa: SLF001
    init = jax.tree.map(lambda x: x[0], obs)
    measured = obs["s"]["position"][1:]

    @jax.jit
    def residual(sub):
        p = jax.tree.map(lambda x: x, truth)
        for key, value in sub.items():
            p["nodes"]["s"][key] = value

        def body(s, _):
            s = step_fn(s, ext, p)
            return s, s["s"]["position"]

        return jax.lax.scan(body, init, None, length=20)[1] - measured

    F = np.asarray(fim(residual, base, scale=None).fim, dtype=np.float64)
    jac = np.zeros((20, 2))
    for j, key in enumerate(names):
        h = 3e-3 * (1.0 + abs(float(base[key])))
        plus = {**base, key: jnp.float32(float(base[key]) + h)}
        minus = {**base, key: jnp.float32(float(base[key]) - h)}
        step = 0.5 * (float(plus[key]) - float(minus[key]))   # the step float32 took
        jac[:, j] = (np.asarray(residual(plus), np.float64)
                     - np.asarray(residual(minus), np.float64)) / (2 * step)
    F_fd = jac.T @ jac
    scale = float(np.abs(F).max())
    assert scale > 0.0
    assert np.abs(F - F_fd).max() <= 2e-2 * scale, (F, F_fd)


# ---------------------------------------------------------------------------
# The multiple-shooting loss
# ---------------------------------------------------------------------------


def test_the_continuity_penalty_is_affine_in_its_weight_and_charges_a_bumped_start():
    """``test_the_continuity_penalty_is_affine_in_its_weight`` and the second
    half of ``test_a_discontinuous_window_start_costs_the_penalty`` (a moved
    window start costs more as the weight grows) on one tiling.  The first
    half -- a moved start costs something at all -- is checked on every push
    by ``tests/core/test_sysid.py::test_multiple_shooting_loss_zero_at_truth_and_penalises_gaps``."""
    gm = _spring_gm()
    obs = _observe(gm, 12, gm.params)
    ws = init_window_states(obs, 4)
    off = _with_params(gm, {"stiffness": 48.0})

    def loss(params, weight, states=ws):
        return float(windowed_loss(gm, params, obs, obs_fn=_position, window=4,
                                   window_states=states, continuity_weight=weight))

    data_only = loss(off, 0.0)
    teacher = float(windowed_loss(gm, off, obs, obs_fn=_position, window=4))
    # Seeded from the measured window starts, multiple shooting is teacher
    # forcing plus the penalty.
    assert np.isclose(data_only, teacher, rtol=1e-5, atol=1e-8)
    one, two = loss(off, 10.0), loss(off, 20.0)
    penalty = one - data_only
    assert penalty > 0.0, penalty                        # off the truth: a real term
    assert np.isclose(two - data_only, 2.0 * penalty, rtol=1e-4, atol=1e-6 * (1.0 + abs(two)))

    bumped = jax.tree.map(lambda x: x, ws)
    bumped["s"]["position"] = bumped["s"]["position"].at[1].add(0.5)
    assert loss(gm.params, 10.0, bumped) > loss(gm.params, 1.0, bumped) > loss(gm.params, 1.0)
