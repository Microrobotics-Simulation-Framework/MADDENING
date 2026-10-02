"""Two ways a fitter's answer depended on something it should not.

**A clipped coordinate past its bound.**  A trainable leaf with bounds and
``transform=None`` is optimised in its own coordinate and *clipped* by
``constrain``.  The clip's derivative is 0 strictly outside the bounds, so
once a step carried the coordinate past its bound nothing could pull it
back: a spring's damping started at 4 against a truth of 0.05 landed on 0
after one Levenberg-Marquardt step, stayed there, and ``fit_lm`` reported
``converged=True`` with a derivative of -1.0 into the range
(audit_040_p4_5/fmu-sysid/repro_fit_lm_stuck_at_bound.py).  Every fitter
now projects the coordinate back onto its bounds after each update, and
``fit_lm`` holds a coordinate on its bound only while moving it into the
range would raise the loss.

**The residual's units.**  ``fit_lm``'s solve added an absolute
``1e-12 * I`` to ``JᵀJ``; for a residual at or below 1e-7 in its own units
that floor was not negligible and the fit stopped being Gauss-Newton
(repro_fit_lm_residual_scale.py).  The floor is now relative.
"""

import contextlib
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening import sysid
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import fit, fit_lm, fit_multiple_shooting, observations_from_history

DT, N = 0.01, 100
TRUE_C = 0.05


@contextlib.contextmanager
def _precision(x64):
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", x64)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


class _Spring(SpringDamperNode):
    """The stock spring with its state at the working float precision, so
    the same graph runs at float32 and float64."""

    def initial_state(self):
        ft = jnp.zeros(()).dtype
        return {"position": jnp.asarray(self.params["initial_position"], ft),
                "velocity": jnp.asarray(self.params["initial_velocity"], ft)}


def _spring(**kw):
    p = dict(stiffness=30.0, damping=2.0, mass=1.0, rest_length=1.0,
             initial_position=0.2, initial_velocity=0.0)
    p.update(kw)
    gm = GraphManager()
    gm.add_node(_Spring(name="s", timestep=DT, **p))
    gm.compile()
    return gm


def _only(gm, *keys):
    mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
    for key in keys:
        mask["nodes"]["s"][key] = True
    return mask


def _record(**kw):
    return _spring(**kw).run_scan_with_history(N)[1]["s"]["position"]


def _residual_against(obs):
    def residual(p):
        # A fresh graph per evaluation: ``run_scan*`` stores its final state.
        return _spring().run_scan_with_history(N, params=p)[1]["s"]["position"] - obs
    return residual


def _damping(res):
    return float(res.params["nodes"]["s"]["damping"])


# ---------------------------------------------------------------------------
# A coordinate a step carries past its bound comes back
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
def test_fit_lm_brings_a_clipped_coordinate_back_into_its_range(x64):
    """The audit's case: one step carries damping from 4 past its bound 0.
    The fit now reaches the interior truth instead of stopping on the bound."""
    with _precision(x64):
        spec = _spring().param_specs()["nodes"]["s"]["damping"]
        assert spec.transform is None and spec.bounds == (0.0, None)
        obs = _record(damping=TRUE_C)
        gm = _spring(damping=4.0)
        res = fit_lm(gm, _residual_against(obs), mask=_only(gm, "damping"), n_iter=50)
        assert res.converged
        assert _damping(res) == pytest.approx(TRUE_C, rel=1e-4)
        assert res.best_loss <= 1e-10 * res.losses[0]


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
def test_fit_lm_brings_it_back_beside_a_log_coordinate(x64):
    with _precision(x64):
        obs = _record(damping=TRUE_C)
        gm = _spring(damping=15.0)
        res = fit_lm(gm, _residual_against(obs), mask=_only(gm, "stiffness", "damping"),
                     n_iter=50)
        assert res.converged
        assert _damping(res) == pytest.approx(TRUE_C, rel=1e-4)
        assert float(res.params["nodes"]["s"]["stiffness"]) == pytest.approx(30.0, rel=1e-5)


def test_fit_brings_a_clipped_coordinate_back_into_its_range():
    """Adam: its step is about ``lr`` whatever the gradient, so it crosses the
    bound too, and stayed clipped (c = 0 after 300 iterations)."""
    obs = _record(damping=TRUE_C)
    residual = _residual_against(obs)
    gm = _spring(damping=2.0)
    res = fit(gm, jax.jit(lambda p: 0.5 * jnp.sum(residual(p) ** 2)),
              mask=_only(gm, "damping"), n_iter=150, lr=0.5)
    assert _damping(res) == pytest.approx(TRUE_C, rel=2e-2)


def test_fit_multiple_shooting_brings_a_clipped_coordinate_back_into_its_range():
    gm_truth = _spring(damping=TRUE_C)
    s0 = gm_truth._user_state(gm_truth._state)  # noqa: SLF001
    obs = observations_from_history(s0, gm_truth.run_scan_with_history(40)[1])
    gm = _spring(damping=2.0)
    res, _ = fit_multiple_shooting(gm, obs, obs_fn=lambda h: h["s"]["position"], window=10,
                                   mask=_only(gm, "damping"), n_iter=150, lr=0.5,
                                   lr_states=1e-6)
    assert _damping(res) == pytest.approx(TRUE_C, rel=5e-2)


@pytest.fixture(scope="module")
def record_past_the_bound():
    """Data from a damping the bounds exclude (-0.3): the constrained
    optimum is *on* the bound, where the gradient points out of the range."""
    gm_data = GraphManager()
    gm_data.add_node(_Spring(name="s", timestep=DT, stiffness=30.0, damping=-0.3, mass=1.0,
                             rest_length=1.0, initial_position=0.2, initial_velocity=0.0))
    gm_data.set_param_spec("s", "damping", ParamSpec())          # unbounded, for the record
    gm_data.compile()
    return gm_data.run_scan_with_history(N)[1]["s"]["position"]


def test_a_fit_whose_optimum_is_past_the_bound_converges_on_the_bound(record_past_the_bound):
    """A constrained stationary point converges there."""
    gm = _spring(damping=2.0)
    res = fit_lm(gm, _residual_against(record_past_the_bound), mask=_only(gm, "damping"),
                 n_iter=50)
    assert res.converged
    assert _damping(res) == 0.0


def test_the_active_bound_is_held_out_of_the_coupled_step(record_past_the_bound):
    """Stiffness free, damping on its bound with the gradient pointing out:
    the step for stiffness is solved with damping held.  Solved as if
    damping could follow its own (clipped) step, stiffness was moved by the
    coupled amount, the loss rose, and the run crept for all 50 iterations
    without converging; held, it converges in a few, at the stiffness that
    is optimal with damping at 0."""
    residual = _residual_against(record_past_the_bound)
    gm = _spring(damping=0.0)
    on_bound = fit_lm(gm, residual, mask=_only(gm, "stiffness"), n_iter=50)
    assert on_bound.converged
    k_constrained = float(on_bound.params["nodes"]["s"]["stiffness"])
    gm = _spring(damping=2.0)
    res = fit_lm(gm, residual, mask=_only(gm, "stiffness", "damping"), n_iter=50)
    assert res.converged and res.n_iter <= 20, (res.converged, res.n_iter)
    assert _damping(res) == 0.0
    assert float(res.params["nodes"]["s"]["stiffness"]) == pytest.approx(k_constrained, rel=1e-4)


def test_a_coordinate_with_a_descent_direction_into_the_range_never_converges_on_the_bound(
        monkeypatch):
    """The guard behind the hold: while a coordinate on its bound could lower
    the loss by moving into the range, no proposal converges the run, however
    small.  Pinned by making every iterate look that way."""
    obs = _record(damping=TRUE_C)
    gm = _spring(damping=0.5)
    monkeypatch.setattr(sysid._CoordinateBounds, "inward_descent",  # noqa: SLF001
                        lambda self, *args, **kwargs: True)
    res = fit_lm(gm, _residual_against(obs), mask=_only(gm, "damping"), n_iter=12)
    assert not res.converged


def test_the_inward_descent_predicate():
    gm = _spring()
    bounds = sysid._CoordinateBounds(  # noqa: SLF001
        gm, gm.params, np.asarray([0]), jnp.float32)
    names = [jax.tree_util.keystr(p) for p, _ in
             jax.tree_util.tree_flatten_with_path(gm.params)[0]]
    assert names[0] == "['nodes']['s']['damping']"
    assert bounds.active
    on_bound, inside = jnp.asarray([0.0]), jnp.asarray([0.3])
    assert bounds.inward_descent(on_bound, jnp.asarray([-1.0]))      # into the range: descent
    assert not bounds.inward_descent(on_bound, jnp.asarray([1.0]))   # out of it: constraint active
    assert not bounds.inward_descent(inside, jnp.asarray([-1.0]))    # not on a bound at all
    np.testing.assert_array_equal(np.asarray(bounds.project(jnp.asarray([-2.0]))), [0.0])
    np.testing.assert_array_equal(np.asarray(bounds.project(inside)), np.asarray(inside))


def test_an_unbounded_fit_is_not_projected():
    """An identity leaf with no bounds: the projection is the identity and
    the predicate never fires (the same arithmetic as before)."""
    gm = _spring()
    assert gm.param_specs()["nodes"]["s"]["rest_length"].bounds == (None, None)
    rest_length_only = np.asarray([[jax.tree_util.keystr(p) for p, _ in
                                    jax.tree_util.tree_flatten_with_path(gm.params)[0]]
                                   .index("['nodes']['s']['rest_length']")])
    bounds = sysid._CoordinateBounds(gm, gm.params, rest_length_only, jnp.float32)  # noqa: SLF001
    assert not bounds.active
    theta = jnp.asarray([-50.0])
    assert bounds.project(theta) is theta


def test_a_log_and_a_logit_coordinate_are_kept_where_constrain_is_not_clamped():
    """Past the point where ``constrain`` clamps a ``log`` / ``logit`` leaf
    its derivative is 0 too, so those coordinates are projected onto the
    range where it is the transform itself; at its ends the value maps
    strictly inside the clamp, with a non-zero derivative."""
    for spec in (ParamSpec(bounds=(0.5, 2.0), transform="logit"),
                 ParamSpec(bounds=(0.0, None), transform="log")):
        u_lo, u_hi = spec._optimiser_interval(np.float32)  # noqa: SLF001
        for u in (u_lo, u_hi):
            assert np.isfinite(u)
            slope = float(jax.grad(spec.to_constrained)(jnp.float32(u)))
            assert slope > 0.0, (spec, u, slope)
        # Beyond the ends the clamp is active and the slope is gone: the
        # dead zone the projection keeps the optimiser out of (past the
        # ``log`` leaf's upper end ``exp`` overflows and the slope is not
        # even finite).
        beyond = u_lo - 50.0 if spec.transform == "log" else u_hi + 50.0
        assert float(jax.grad(spec.to_constrained)(jnp.float32(beyond))) == 0.0, spec


# ---------------------------------------------------------------------------
# fit_lm does not depend on the residual's units
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def linear_problem():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 3))
    b = A @ np.array([40.0, 3.0, 1.2])
    gm = _spring()
    mask = _only(gm, "stiffness", "damping", "rest_length")
    return gm, mask, A, b


@pytest.mark.parametrize("scale", [1e-10, 1e-8, 1e-7, 1e-4, 1.0, 1e4, 1e10])
def test_fit_lm_answers_the_same_for_every_scaling_of_the_residual(linear_problem, scale):
    """``r = s (A x - b)`` has the optimum of ``A x - b`` for every ``s``.  At
    1e-7 the absolute floor left the fit unconverged after 50 iterations; at
    1e-8 it barely moved."""
    gm, mask, A, b = linear_problem
    ft = jnp.float32

    def residual(p):
        q = p["nodes"]["s"]
        x = jnp.stack([q["stiffness"], q["damping"], q["rest_length"]])
        return scale * (jnp.asarray(A, ft) @ x - jnp.asarray(b, ft))

    res = fit_lm(gm, residual, mask=mask, n_iter=50)
    q = res.params["nodes"]["s"]
    assert res.converged and res.n_iter <= 6, (res.converged, res.n_iter)
    np.testing.assert_allclose([float(q["stiffness"]), float(q["damping"]),
                                float(q["rest_length"])], [40.0, 3.0, 1.2], rtol=2e-6)


def test_a_noise_std_is_the_same_scaling(linear_problem):
    """Weighting by ``noise_std`` divides the residual by it, so a large σ
    is a small residual: the same answer for σ = 1 and σ = 1e8."""
    gm, mask, A, b = linear_problem

    def residual(p):
        q = p["nodes"]["s"]
        x = jnp.stack([q["stiffness"], q["damping"], q["rest_length"]])
        return jnp.asarray(A, jnp.float32) @ x - jnp.asarray(b, jnp.float32)

    for sigma in (1.0, 1e8):
        res = fit_lm(gm, residual, mask=mask, n_iter=50, noise_std=sigma)
        assert res.converged, sigma
        assert float(res.params["nodes"]["s"]["damping"]) == pytest.approx(3.0, rel=2e-6)


# ---------------------------------------------------------------------------
# fit_lm does not depend on the parameters' units
# (audit_040_p4_6/fmu-sysid/repro_q3_marquardt_floor_units.py)
# ---------------------------------------------------------------------------

#: One physical spring in two unit systems: positions identical to rounding.
_UNITS = {"SI": (1e6, 3e7, 2e4), "tonnes": (1e3, 3e4, 20.0)}


def _unit_spring(m, k, c):
    gm = GraphManager()
    gm.add_node(_Spring(name="s", timestep=0.01, stiffness=k, damping=c, mass=m,
                        rest_length=1.0, initial_position=1.2, initial_velocity=0.0))
    gm.compile()
    return gm


def _unit_fit(m, k, c_true, **kw):
    obs = _unit_spring(m, k, c_true).run_scan_with_history(200)[1]["s"]["position"]

    def residual(p):
        return _unit_spring(m, k, 1.0).run_scan_with_history(200, params=p)[1]["s"]["position"] - obs

    gm = _unit_spring(m, k, 3.0 * c_true)
    return fit_lm(gm, residual, mask=_only(gm, "stiffness", "damping"), n_iter=50, **kw)


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
def test_fit_lm_answers_the_same_in_every_unit_system(x64):
    """A floor of ``eps`` times the *mean* of ``diag(JᵀJ)`` crushed the step of
    damping in SI units (its column is small beside the log stiffness'):
    two iterations, ``converged=True``, damping at 3x its truth, where the
    same spring in tonnes reached the truth.  Per column, both do, in about
    the same number of iterations."""
    with _precision(x64):
        runs = {}
        for name, (m, k, c_true) in _UNITS.items():
            res = _unit_fit(m, k, c_true)
            assert res.converged, name
            assert _damping(res) == pytest.approx(c_true, rel=2e-4), name
            assert float(res.params["nodes"]["s"]["stiffness"]) == pytest.approx(k, rel=1e-5)
            runs[name] = res.n_iter
        assert abs(runs["SI"] - runs["tonnes"]) <= 2, runs


def test_a_shrunken_step_never_reads_as_stationary(monkeypatch):
    """Defence in depth.  Whatever shrinks a step -- a mis-scaled floor, a
    large damping -- ``converged`` also needs the undamped Gauss-Newton step
    from the iterate to be within ``step_tol``, which nothing in the
    Marquardt solve can shrink.  Simulated here by a step that moves a
    millionth of the way."""
    from maddening import sysid

    real = sysid._marquardt_step  # noqa: SLF001

    def crushed(th, r, J, lam, lo, hi):
        return th + 1e-6 * (real(th, r, J, lam, lo, hi) - th)

    monkeypatch.setattr(sysid, "_marquardt_step", crushed)
    m, k, c_true = _UNITS["tonnes"]
    res = _unit_fit(m, k, c_true)
    assert not res.converged
    assert _damping(res) == pytest.approx(3.0 * c_true, rel=1e-2)    # it really was stuck
