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


@pytest.mark.parametrize("scale", [1e-30, 1e-20, 1e-19, 1e-18, 1e-10, 1e-8, 1e-7, 1e-4,
                                   1.0, 1e4, 1e10, 1e18, 1e19, 1e20, 1e30])
def test_fit_lm_answers_the_same_for_every_scaling_of_the_residual(linear_problem, scale):
    """``r = s (A x - b)`` has the optimum of ``A x - b`` for every ``s``.  At
    1e-7 the absolute floor left the fit unconverged after 50 iterations; at
    1e-8 it barely moved.  From ``1e-18`` down, ``r * r``, ``JᵀJ`` and ``Jᵀr``
    flushed in float32 and the fit reported ``converged=True`` 2-7% off or at
    its start; from ``1e19`` up they overflowed (MADD-ANO-174)."""
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

    for sigma in (1.0, 1e8, 1e20, 1e-20):
        res = fit_lm(gm, residual, mask=mask, n_iter=50, noise_std=sigma)
        assert res.converged, sigma
        assert float(res.params["nodes"]["s"]["damping"]) == pytest.approx(3.0, rel=2e-6)


def test_fit_lm_never_converges_on_a_solve_it_could_not_form(linear_problem, monkeypatch):
    """Defence in depth for MADD-ANO-174: whatever frames the solve, a step
    whose live columns were not representable (a ``diag(JᵀJ)`` of 0 or
    ``inf`` for a column ``J`` does not leave at zero) never carries a
    converged verdict -- the run stops unconverged and says why."""
    from maddening import sysid

    gm, mask, A, b = linear_problem
    real = sysid._marquardt_step  # noqa: SLF001

    def unrepresentable(th, r, J, lam, lo, hi, held):
        cand, _ = real(th, r, J, lam, lo, hi, held)
        return cand, False

    def residual(p):
        q = p["nodes"]["s"]
        x = jnp.stack([q["stiffness"], q["damping"], q["rest_length"]])
        return jnp.asarray(A, jnp.float32) @ x - jnp.asarray(b, jnp.float32)

    assert fit_lm(gm, residual, mask=mask, n_iter=50).converged       # non-vacuity
    monkeypatch.setattr(sysid, "_marquardt_step", unrepresentable)
    with pytest.warns(RuntimeWarning, match="stationarity could not be read"):
        res = fit_lm(gm, residual, mask=mask, n_iter=50)
    assert not res.converged


def test_the_frames_reach_the_gauss_newton_test_too(linear_problem):
    """``_gauss_newton_step`` takes its column norms of framed columns: a
    column of ``1e-23`` used to have a norm that flushed to zero, was left
    out, and the iterate read as stationary with that parameter unmoved;
    bare and framed agree bit for bit where the bare norms are normal."""
    from maddening.sysid import _gauss_newton_step

    rng = np.random.default_rng(3)
    J = jnp.asarray(rng.normal(size=(20, 2)), jnp.float32)
    r = jnp.asarray(rng.normal(size=20), jnp.float32)
    th = jnp.asarray([0.3, 0.7], jnp.float32)
    lo, hi = jnp.full(2, -jnp.inf, jnp.float32), jnp.full(2, jnp.inf, jnp.float32)
    held = jnp.zeros(2, bool)
    base, ok = _gauss_newton_step(th, r, J, lo, hi, held)
    assert bool(ok)
    # The second parameter in units 2**-80 smaller: its column 2**-80 the
    # size (norm ~1e-24, flushed bare), its step 2**80 larger.
    c = jnp.asarray([1.0, 2.0 ** -80], jnp.float32)
    th_c = th / c
    cand, ok = _gauss_newton_step(th_c, r, J * c, lo, hi, held)
    assert bool(ok)
    np.testing.assert_array_equal(np.asarray(cand * c), np.asarray(base))
    # ... and the residual 2**-70 smaller, which flushed the norms' products.
    cand, ok = _gauss_newton_step(th, r * 2.0 ** -70, J * 2.0 ** -70, lo, hi, held)
    np.testing.assert_array_equal(np.asarray(cand), np.asarray(base))


def test_the_marquardt_step_scales_exactly_with_its_residual():
    """``_marquardt_step`` frames the residual as well as ``J``'s columns, so
    its step is linear in ``r`` to the bit for a power-of-two scaling --
    down to a residual of ``2**-100`` (every entry still a normal number)
    against a Jacobian whose columns hold entries down to ``1e-8`` of their
    largest, where the products of a bare ``Jᵀr`` flush."""
    from maddening.sysid import _marquardt_step

    rng = np.random.default_rng(4)
    J = jnp.asarray(rng.normal(size=(20, 3)) * np.array([1.0, 1e-2, 1e-4])
                    * np.logspace(-8.0, 0.0, 20)[:, None], jnp.float32)
    r = jnp.asarray(rng.normal(size=20), jnp.float32)
    th = jnp.zeros(3, jnp.float32)
    lo, hi = jnp.full(3, -jnp.inf, jnp.float32), jnp.full(3, jnp.inf, jnp.float32)
    held = jnp.zeros(3, bool)
    lam = jnp.asarray(1e-2, jnp.float32)
    base, ok = _marquardt_step(th, r, J, lam, lo, hi, held)
    assert bool(ok)
    for k in (-100, -60, 60, 100):
        assert float(jnp.min(jnp.abs(r * 2.0 ** k))) >= float(jnp.finfo(jnp.float32).tiny)
        cand, ok = _marquardt_step(th, r * 2.0 ** k, J, lam, lo, hi, held)
        assert bool(ok)
        np.testing.assert_array_equal(np.asarray(cand) * 2.0 ** -k, np.asarray(base))
    # At the top of the range ``Jᵀr`` itself overflows unless ``r`` is framed:
    # twenty residual entries near ``1e38`` summed against framed columns.
    Jw = jnp.asarray(rng.normal(size=(20, 3)), jnp.float32)
    rw = jnp.asarray(np.sign(np.asarray(Jw[:, 0])) * 1.5, jnp.float32)
    base_w, _ = _marquardt_step(th, rw, Jw, lam, lo, hi, held)
    cand, ok = _marquardt_step(th, rw * 2.0 ** 126, Jw, lam, lo, hi, held)
    assert bool(ok) and np.all(np.isfinite(np.asarray(cand)))
    np.testing.assert_array_equal(np.asarray(cand, np.float64) * 2.0 ** -126,
                                  np.asarray(base_w, np.float64))


def test_a_loss_that_underflows_float64_is_never_converged():
    """Under x64 a residual near ``1e-170`` has a framed loss whose unframing
    underflows float64 to ``0.0`` while ``r`` is not zero; no step can lower
    it, and a run started at the optimum is not called converged on it."""
    from tests.core.test_sysid_claims_under_x64 import _x64

    rng = np.random.default_rng(5)
    A = rng.normal(size=(30, 2))
    y = A @ np.array([0.3, 0.7]) + 1e-3 * rng.normal(size=30)
    optimum = np.linalg.lstsq(A, y, rcond=None)[0]
    with _x64():
        gm = GraphManager()
        gm.add_node(_Spring(name="s", timestep=0.01, stiffness=float(optimum[0]),
                            damping=float(optimum[1])))
        gm.compile()
        mask = _only(gm, "stiffness", "damping")

        def residual(p):
            q = p["nodes"]["s"]
            return 1e-170 * (jnp.asarray(A) @ jnp.stack([q["stiffness"], q["damping"]])
                             - jnp.asarray(y))

        with pytest.warns(RuntimeWarning, match="underflows float64 although r is not zero"):
            res = fit_lm(gm, residual, mask=mask, n_iter=10, hold_undetermined=False)
        assert not res.converged and res.best_loss == 0.0


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
    the same number of iterations: counted to the truth's neighbourhood (the
    loss under ``64 eps`` of its start in the working precision), not to the
    run's end.  The iterations
    after that are the rounding floor's, and their number is not a property
    of the units: since the fitters evaluate the leaves no step moves as
    they went in (SYS-071), the unmasked mass is the recording's to the bit,
    the float64 loss can reach exactly 0.0, and the two systems take 6 and
    10 iterations to get there, where both used to stop near 2.7e-29."""
    with _precision(x64):
        runs = {}
        for name, (m, k, c_true) in _UNITS.items():
            res = _unit_fit(m, k, c_true)
            assert res.converged, name
            assert _damping(res) == pytest.approx(c_true, rel=2e-4), name
            assert float(res.params["nodes"]["s"]["stiffness"]) == pytest.approx(k, rel=1e-5)
            losses = np.asarray(res.losses, np.float64)
            eps = float(np.finfo(np.float64 if x64 else np.float32).eps)
            near = losses <= 64 * eps * losses[0]
            assert near.any(), (name, losses)
            runs[name] = int(np.argmax(near))
        assert abs(runs["SI"] - runs["tonnes"]) <= 2, runs


#: The two runs a shrunken Marquardt step leaves stuck, as ``(x64, the
#: fraction of the step taken, step_tol)``.
_SHRUNKEN = {
    # No candidate moves, so none lowers the loss: the run ends at its start
    # on a rejected proposal that is within ``step_tol``.
    "to-nothing": (False, 0.0, None),
    # Float64 resolves the loss's decrease over a millionth of the step, so
    # every proposal is accepted, each within a ``step_tol`` of ``1e-5``.
    "to-a-millionth": (True, 1e-6, 1e-5),
}


@pytest.mark.parametrize("shrunk", sorted(_SHRUNKEN))
def test_a_shrunken_step_never_reads_as_stationary(monkeypatch, shrunk):
    """Defence in depth.  Whatever shrinks a step -- a mis-scaled floor, a
    large damping -- ``converged`` also needs the undamped Gauss-Newton step
    from the iterate to be within ``step_tol``, which nothing in the
    Marquardt solve can shrink.  Simulated by a Marquardt step that goes
    only a fraction of its way, in the two runs that leaves stuck
    (:data:`_SHRUNKEN`): the proposal is within ``step_tol`` and rejected
    from the start, or within ``step_tol`` and accepted every time.  Both
    are 3x the true damping away, and neither may read as converged.

    Between the two -- a shrunken candidate rejected *after* the run has
    lowered the loss -- the fit is not stuck: the floor rule takes the
    Gauss-Newton step itself, which is the next test.  A millionth of the
    step in float32 is that case or the first by the rounding of one loss
    comparison: its first candidate tied with the start while the fitters
    evaluated a leaf they do not move at its ``exp(log(p))`` round trip,
    and lowered the loss in its sixth digit once they evaluated it as it
    went in (SYS-071), so the run went on to the truth.  Neither run here
    rests on a comparison that close."""
    from maddening import sysid

    x64, fraction, step_tol = _SHRUNKEN[shrunk]
    marquardt, gauss_newton = sysid._marquardt_step, sysid._gauss_newton_step  # noqa: SLF001
    proposed, undamped = [], []

    def moved(th, cand) -> float:
        """The largest move of a coordinate, relative to itself."""
        return float(jnp.max(jnp.abs(cand - th) / jnp.abs(th)))

    def shrunken(th, r, J, lam, lo, hi, held):
        cand, ok = marquardt(th, r, J, lam, lo, hi, held)
        cand = th + fraction * (cand - th)
        proposed.append(moved(th, cand))
        return cand, ok

    def watched(th, r, J, lo, hi, held):
        cand, ok = gauss_newton(th, r, J, lo, hi, held)
        undamped.append(moved(th, cand))
        return cand, ok

    monkeypatch.setattr(sysid, "_marquardt_step", shrunken)
    monkeypatch.setattr(sysid, "_gauss_newton_step", watched)
    m, k, c_true = _UNITS["tonnes"]
    with _precision(x64):
        res = _unit_fit(m, k, c_true, step_tol=step_tol)
    assert not res.converged
    # The verdict asked the undamped step, and from every iterate it was
    # asked at that step would have moved the damping by more than half.
    assert undamped and min(undamped) > 0.5
    if fraction == 0.0:
        # It really was stuck: no candidate moved, and it is at its start,
        # to the bit.
        assert proposed and max(proposed) == 0.0
        assert _damping(res) == 3.0 * c_true
        assert float(res.params["nodes"]["s"]["stiffness"]) == k
    else:
        # It really was stuck: every proposal was within ``step_tol`` and
        # accepted -- one Marquardt solve an iteration, each lowering the
        # loss, to the end of the budget -- and fifty of them went nowhere.
        assert 0.0 < min(proposed) and max(proposed) <= step_tol
        assert len(proposed) == res.n_iter == 50
        assert np.all(np.diff(res.losses) < 0.0)
        assert 3.0 * c_true * (1.0 - 1e-4) < _damping(res) < 3.0 * c_true


def test_a_run_of_stalled_candidates_is_not_the_rounding_floor(monkeypatch):
    """The floor rule's defence: after progress, every candidate rejected
    down to one within ``step_tol`` reads as the rounding floor only if the
    undamped Gauss-Newton step does not lower the loss either.  Simulated by
    candidates that stop moving after the first (accepted) step: they all
    tie and are rejected, but the fit is nowhere near its floor.

    The Gauss-Newton step that lowers the loss is now taken as the iterate
    (MADD-ANO-174) rather than ending the run unconverged, so a run whose
    damped candidates all stall is carried on by it, and converges only
    where that step is within ``step_tol`` -- at the truth."""
    from maddening import sysid

    real = sysid._marquardt_step  # noqa: SLF001
    calls = []

    def stalls(th, r, J, lam, lo, hi, held):
        calls.append(1)
        return real(th, r, J, lam, lo, hi, held) if len(calls) == 1 else (th, True)

    monkeypatch.setattr(sysid, "_marquardt_step", stalls)
    m, k, c_true = _UNITS["tonnes"]
    res = _unit_fit(m, k, c_true)
    assert len(res.losses) >= 2 and res.losses[1] < res.losses[0]     # it progressed
    assert len(calls) > 2                                              # and stalled
    # Only the first Marquardt candidate moved, so every update after it is
    # a Gauss-Newton step the floor rule took.
    assert res.best_iteration > 1
    assert res.converged
    assert _damping(res) == pytest.approx(c_true, rel=2e-4)
    assert float(res.params["nodes"]["s"]["stiffness"]) == pytest.approx(k, rel=1e-5)


@pytest.mark.parametrize("n_iter", [2, 10], ids=["its-last-act", "then-converged"])
def test_a_gauss_newton_iterate_is_evaluated_as_it_is_returned(monkeypatch, n_iter):
    """Where the floor rule takes the Gauss-Newton step as the iterate, that
    candidate is what the run evaluates, records and returns, as an accepted
    Marquardt candidate is: ``best_loss`` is the loss of exactly the
    parameters returned (SYS-071), and ``converged`` is only ever reported
    at an iterate a loss-lowering step produced (SYS-080).

    ``stiffness`` is not fitted, and its ``exp(log(30.0))`` round trip is not
    ``30.0`` in float32: a candidate evaluated at the round trip reports
    ``12.50001`` for a returned point whose loss is ``12.5``.  With two
    iterations the Gauss-Newton step is the run's last act and ``best_loss``
    is that candidate's own evaluation; with more, the run converges on it
    at the next iteration, which forms its loss again."""
    from maddening import sysid

    marquardt, gauss_newton = sysid._marquardt_step, sysid._gauss_newton_step  # noqa: SLF001
    calls, undamped = [], []

    def stalls(th, r, J, lam, lo, hi, held):
        calls.append(1)
        return marquardt(th, r, J, lam, lo, hi, held) if len(calls) == 1 else (th, True)

    def watched(th, r, J, lo, hi, held):
        cand, ok = gauss_newton(th, r, J, lo, hi, held)
        undamped.append(np.asarray(cand).tobytes())
        return cand, ok

    def residual(p):
        q = p["nodes"]["s"]
        return jnp.stack([q["stiffness"] - 25.0, q["damping"] - 1.0])

    gm = _spring()
    k = np.asarray(gm.params["nodes"]["s"]["stiffness"])
    assert float(jnp.exp(jnp.log(jnp.asarray(k)))) != float(k), "premise: the round trip moves it"
    monkeypatch.setattr(sysid, "_marquardt_step", stalls)
    monkeypatch.setattr(sysid, "_gauss_newton_step", watched)
    res = fit_lm(gm, residual, mask=_only(gm, "damping"), n_iter=n_iter, hold_undetermined=False)

    # Two updates, each of which lowered the loss: the one Marquardt step,
    # then -- every later candidate stalled -- the Gauss-Newton step, which
    # is the iterate returned, to the bit.
    assert len(calls) > 2 and res.best_iteration == 2
    assert np.all(np.diff(res.losses) < 0.0)
    assert np.asarray(res.params["nodes"]["s"]["damping"]).tobytes() in undamped
    assert np.asarray(res.params["nodes"]["s"]["stiffness"]).tobytes() == k.tobytes()
    again = float(0.5 * jnp.sum(residual(res.params) ** 2))
    assert again == float(res.best_loss) == 12.5, (again, res.best_loss)
    if n_iter == 2:
        assert not res.converged and len(res.losses) == 2       # out of budget on that step
    else:
        assert res.converged and res.n_iter == 3
        assert float(res.losses[2]) == float(res.best_loss)
