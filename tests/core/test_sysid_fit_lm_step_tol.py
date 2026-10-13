"""``fit_lm``'s ``converged`` is reachable at the precision the fit runs in.

Two things made a float32 fit that had reached its exact optimum report
``converged=False``:

* only an *accepted* step could set it, and at the float floor the step a
  fit proposes rounds to nothing, which cannot strictly lower a loss that
  is already rounding noise -- so the run ended on twelve rejections;
* the default ``step_tol`` was an absolute ``1e-8`` in the optimiser's
  coordinates, below one float32 ulp of almost any coordinate (2.4e-7 for
  ``log(40)``), so no step at all could meet it.

Now the iteration's *proposal* is put to ``step_tol`` whether or not it was
accepted, and the default is ``2**4`` ulps of each parameter's own dtype,
measured on the physical parameters.
"""

import contextlib
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import fit_lm

DT, N = 0.01, 80


@contextlib.contextmanager
def _x64():
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


def _spring(**kw):
    p = dict(stiffness=30.0, damping=2.0, mass=1.0, rest_length=1.0,
             initial_position=0.2, initial_velocity=0.0)
    p.update(kw)
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="s", timestep=DT, **p))
    gm.compile()
    return gm


def _only(gm, *keys):
    mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
    for key in keys:
        mask["nodes"]["s"][key] = True
    return mask


def test_a_noiseless_float32_fit_at_its_optimum_is_converged():
    """The audit's case: noiseless data, stiffness and damping trainable.
    The fit reaches loss ~1e-13 in four iterations and its next proposal
    moves neither parameter by a bit; before, that read ``converged=False``
    at the default ``step_tol`` and at 1e-5 alike."""
    obs = _spring(stiffness=40.0, damping=3.0).run_scan_with_history(N)[1]["s"]["position"]

    def residual(p):
        # A fresh graph per evaluation: ``run_scan*`` stores its final state.
        return _spring().run_scan_with_history(N, params=p)[1]["s"]["position"] - obs

    gm = _spring()
    res = fit_lm(gm, residual, mask=_only(gm, "stiffness", "damping"), n_iter=50)
    assert res.converged
    assert res.n_iter < 50
    assert res.best_loss < 1e-10
    assert float(res.params["nodes"]["s"]["stiffness"]) == pytest.approx(40.0, rel=1e-5)
    assert float(res.params["nodes"]["s"]["damping"]) == pytest.approx(3.0, rel=1e-5)


def _linear_residual(target):
    """``r = 1e3 (rest_length - target)``: linear in an identity-transform
    leaf, so Levenberg-Marquardt's proposal from ``p0`` is exactly
    ``(p0 - target) / (1 + lam)`` -- a step of a chosen number of ulps."""
    return lambda p: 1e3 * (p["nodes"]["s"]["rest_length"] - target)[None]


def _ulps_away(value, k, dtype):
    x = np.asarray(value, dtype=dtype)
    for _ in range(k):
        x = np.nextafter(x, np.asarray(np.inf, dtype=dtype))
    return x


@pytest.mark.parametrize("k, stops_on_the_first_proposal", [(8, True), (64, False)])
def test_the_default_step_tol_is_sixteen_ulps_of_the_parameter(k, stops_on_the_first_proposal):
    """A proposal of ``k / 1.01`` ulps (``lam0 = 1e-2``): within ``2**4``
    ulps it converges the run on the spot, beyond it the run goes on to
    the next proposal (``k / 101`` ulps) and converges there.  Under the
    old absolute ``1e-8`` neither ever converged: one ulp of 1.0 is
    1.2e-7, and the proposal after the first rounds to nothing and is
    rejected."""
    gm = _spring()
    target = jnp.asarray(_ulps_away(1.0, k, np.float32))
    res = fit_lm(gm, _linear_residual(target), mask=_only(gm, "rest_length"), n_iter=20)
    assert res.converged
    assert res.n_iter == (1 if stops_on_the_first_proposal else 2)


def test_the_default_step_tol_follows_the_working_precision():
    """Under x64 the same eight-ulp proposal is eight *float64* ulps, and the
    default is measured in float64's resolution: it converges on the spot.
    A float32 leaf in an x64 graph is still held to float32's resolution,
    though ``ravel_pytree`` promotes the optimiser's vector to float64."""
    with _x64():
        gm = _spring()
        assert gm.params["nodes"]["s"]["rest_length"].dtype == jnp.float64
        target = jnp.asarray(_ulps_away(1.0, 8, np.float64))
        res = fit_lm(gm, _linear_residual(target), mask=_only(gm, "rest_length"), n_iter=20)
        assert res.converged and res.n_iter == 1

        mixed = _spring(rest_length=np.float32(1.0))
        assert mixed.params["nodes"]["s"]["rest_length"].dtype == jnp.float32
        assert mixed.params["nodes"]["s"]["stiffness"].dtype == jnp.float64
        target = jnp.asarray(_ulps_away(1.0, 8, np.float32))
        res = fit_lm(mixed, _linear_residual(target), mask=_only(mixed, "rest_length"),
                     n_iter=20)
        assert res.converged and res.n_iter == 1


def test_step_tol_zero_counts_only_a_proposal_of_exactly_nothing():
    """``0.0`` is not "off": a proposal that leaves every parameter bit for
    bit where it was is a fixed point of the iteration, and is converged."""
    gm = _spring()
    res = fit_lm(gm, lambda p: jnp.ones(3, jnp.float32), n_iter=4, step_tol=0.0)
    assert res.converged and res.n_iter == 1
    target = jnp.asarray(_ulps_away(1.0, 8, np.float32))
    res = fit_lm(gm, _linear_residual(target), mask=_only(gm, "rest_length"), n_iter=20,
                 step_tol=0.0)
    # Eight ulps is a real step at 0.0; the next proposal rounds to nothing.
    assert res.converged and res.n_iter > 1


def test_an_explicit_step_tol_is_relative_to_each_parameter():
    """``step_tol=1e-3`` means a 0.1% change, under every transform: a
    proposal of 0.05% stops the run, one of 0.2% does not."""
    gm = _spring()
    for change, stops in ((5e-4, True), (2e-3, False)):
        target = jnp.float32(1.0 + change)
        res = fit_lm(gm, _linear_residual(target), mask=_only(gm, "rest_length"),
                     n_iter=20, step_tol=1e-3)
        assert res.converged
        assert (res.n_iter == 1) is stops, (change, res.n_iter)


@pytest.mark.parametrize("bad", [-1e-6, float("nan"), float("inf")])
def test_a_step_tol_with_no_reading_is_refused(bad):
    gm = _spring()
    with pytest.raises(ValueError, match="step_tol"):
        fit_lm(gm, lambda p: jnp.ones(3, jnp.float32), n_iter=1, step_tol=bad)


# ---------------------------------------------------------------------------
# The floor rule's guards (audit_040_p4_6/fmu-sysid/repro_q3_floor_rule_guard_unpinned.py)
# ---------------------------------------------------------------------------


@jax.custom_jvp
def _wrong_sign(x):
    return x


@_wrong_sign.defjvp
def _wrong_sign_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return x, -dx                  # the derivative a buggy custom rule reports


def _damping_record(n=80):
    return _spring(damping=0.5).run_scan_with_history(n)[1]["s"]["position"]


def test_a_run_that_never_lowered_the_loss_is_not_converged_by_the_floor_rule():
    """Every candidate of a wrong-signed Jacobian climbs, down to one far
    inside ``step_tol``.  The floor rule reads "every candidate rejected"
    as the rounding floor only after the run has lowered the loss once;
    without that guard this run was ``converged=True`` at its start."""
    obs = _damping_record()

    def residual(p):
        q = {**p, "nodes": {"s": {**p["nodes"]["s"],
                                  "damping": _wrong_sign(p["nodes"]["s"]["damping"])}}}
        return _spring().run_scan_with_history(80, params=q)[1]["s"]["position"] - obs

    gm = _spring(damping=2.0)
    res = fit_lm(gm, residual, mask=_only(gm, "damping"), n_iter=20)
    assert not res.converged
    assert res.n_iter == 1 and float(res.params["nodes"]["s"]["damping"]) == 2.0


def test_the_floor_rule_never_fires_while_a_bound_coordinate_could_descend(monkeypatch):
    """The other guard: no floor-rule verdict while a coordinate on its bound
    could lower the loss by moving into the range.  A weakly determined
    damping ends its fit on the floor rule (control); told every iterate has
    such a coordinate, the same run must not converge."""
    from maddening import sysid

    obs = _damping_record(100)

    def residual(p):
        return _spring().run_scan_with_history(100, params=p)[1]["s"]["position"] - obs

    gm = _spring(damping=2.0)
    control = fit_lm(gm, residual, mask=_only(gm, "damping"), n_iter=50)
    assert control.converged
    monkeypatch.setattr(sysid._CoordinateBounds, "inward_descent",  # noqa: SLF001
                        lambda self, *args, **kwargs: True)
    res = fit_lm(gm, residual, mask=_only(gm, "damping"), n_iter=50)
    assert not res.converged


# ---------------------------------------------------------------------------
# Under x64: the floor rule's ladder reaches the default tolerance
# ---------------------------------------------------------------------------


def _x64_noiseless_fit(damping_truth, **kw):
    """The audit's x64 case (audit_040_p4_8/fmu-sysid/repro_fit_lm_converged_x64.py):
    noiseless spring data, stiffness and damping fitted from (45, 4), mass
    frozen.  Call inside ``_x64()``: the record starts from the seed cast to
    float64 (the graph's scans refuse a float32 seed under x64,
    MADD-ANO-017)."""
    n = 150

    def build(c):
        gm = GraphManager()
        gm.add_node(SpringDamperNode("spring", 0.01, initial_position=0.5,
                                     stiffness=30.0, damping=c, mass=1.0))
        gm.compile()
        return gm

    truth = build(damping_truth)
    init = {"spring": {k: jnp.asarray(v, jnp.float64)[None]
                       for k, v in truth.get_node_state("spring").items()}}
    data = truth.run_sweep(n, init, return_history=True)[1]["spring"]["position"][0]
    gm = build(2.0)
    residual = jax.jit(lambda p: gm.run_sweep(n, init, return_history=True, params=p)[1]
                       ["spring"]["position"][0] - data)
    start = jax.tree.map(lambda x: x, gm.params)
    start["nodes"]["spring"]["stiffness"] = jnp.asarray(45.0, jnp.float64)
    start["nodes"]["spring"]["damping"] = jnp.asarray(4.0, jnp.float64)
    mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
    mask["nodes"]["spring"]["stiffness"] = True
    mask["nodes"]["spring"]["damping"] = True
    return fit_lm(gm, residual, params=start, mask=mask, n_iter=100, **kw)


def test_a_noiseless_float64_fit_at_its_floor_is_converged():
    """Twelve damped candidates ended one rung short of the default
    ``step_tol`` (a relative step of ``4.4e-15`` against ``3.6e-15`` at
    ``lam = 1e5``), so the floor rule could not fire and the fit, at loss
    ``2.3e-30`` after 8 of 100 iterations, read ``converged=False``.  The
    ladder now goes on to a candidate within the tolerance."""
    with _x64():
        res = _x64_noiseless_fit(1e-6)
    assert res.converged, (res.n_iter, res.best_loss)
    assert res.n_iter < 100 and res.best_loss < 1e-25
    assert float(res.params["nodes"]["spring"]["damping"]) == pytest.approx(1e-6, rel=1e-6)


def test_a_truth_of_exactly_zero_has_no_relative_resolution():
    """The decision, pinned: a damping whose truth is exactly 0 is fitted to
    the residual's rounding noise around 0 (``1.3e-15`` in float64), and a
    step relative to rounding noise is never within ``step_tol`` -- so the
    run stops at its floor with ``converged=False``, at a loss of ``1e-30``.
    The flag is exact about what it tests; a ``tol`` at the noise floor is
    how such a fit says it is done, and the ladder's extension must not
    have bought a looser reading."""
    with _x64():
        stopped = _x64_noiseless_fit(0.0)
        told = _x64_noiseless_fit(0.0, tol=1e-25)
    assert not stopped.converged and stopped.n_iter < 100
    assert stopped.best_loss < 1e-28
    assert abs(float(stopped.params["nodes"]["spring"]["damping"])) < 1e-12
    assert told.converged and told.best_loss <= 1e-25


def test_the_ladder_extension_does_not_depend_on_lam_up(monkeypatch):
    """A small ``lam_up`` damps the twelve rungs by little (``1.5**12`` is
    130); the extension grows ``lam`` by at least a decade a rung, so it
    still reaches a candidate within ``step_tol`` -- in at most 24 more
    rungs, rather than the ``log(1e24) / log(lam_up)`` a ladder of
    ``lam_up`` rungs would need -- and the floor rule fires.  The
    candidates are counted per iteration (``callback`` marks each one's
    start): measured 22 in the last, 12 and 10; a ladder extended by
    ``lam_up`` would have needed 69."""
    from maddening import sysid

    per_iteration: dict[int, int] = {}
    current = {"i": 0}
    step = sysid._marquardt_step                      # noqa: SLF001

    def counted(*args):
        per_iteration[current["i"]] = per_iteration.get(current["i"], 0) + 1
        return step(*args)

    monkeypatch.setattr(sysid, "_marquardt_step", counted)
    with _x64():
        res = _x64_noiseless_fit(1e-6, lam_up=1.5,
                                 callback=lambda i, *_: current.__setitem__("i", i))
    assert res.converged, (res.n_iter, res.best_loss)
    assert res.best_loss < 1e-25
    assert max(per_iteration.values()) > 12, ("the extension never ran", per_iteration)
    assert max(per_iteration.values()) <= 12 + 24, per_iteration


# ---------------------------------------------------------------------------
# At the float floor the verdict depends on neither a parameter's units nor
# the sign of an on-bound gradient's rounding (MADD-ANO-174)
# ---------------------------------------------------------------------------


class _Pair(SimulationNode):
    """A node that only carries the constants ``a`` and ``b`` and their
    specs; the residuals below read them through ``gm.params``."""

    def __init__(self, name, specs, **params):
        self._specs = specs
        super().__init__(name, 0.01, **params)

    def param_specs(self):
        return {**super().param_specs(), **self._specs}

    def initial_state(self):
        return {"x": jnp.zeros(())}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": state["x"] + 0.0 * (p["a"] + p["b"])}


def _pair(specs, **values):
    gm = GraphManager()
    gm.add_node(_Pair("h", specs, **values))
    gm.compile()
    return gm


@pytest.mark.parametrize("unit", [1e-8, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e8])
def test_the_floor_verdict_does_not_depend_on_a_parameters_units(unit):
    """A noisy linear fit in float32, ``a`` an identity parameter in units
    ``unit``: the loss's own rounding (its residual is noise of 0.01, whose
    float32 rounding moves the loss by more than a few ulps of ``a`` do)
    rejected every damped candidate 40 ulps short of the optimum in some
    units -- ``converged=False`` there, ``True`` a few ulps away in others
    (audit_040_p4_10/fmu-sysid/repro_fit_lm_floor_verdict_units.py, part A).
    The floor rule now takes the undamped Gauss-Newton step that still
    lowers the loss, and every unit converges within ``step_tol``."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(40, 2)).astype(np.float32)
    y = (A @ np.array([1.3, 0.7]) + 0.01 * rng.normal(size=40)).astype(np.float32)
    optimum = np.linalg.lstsq(A.astype(np.float64), y.astype(np.float64), rcond=None)[0]
    Aj, yj = jnp.asarray(A), jnp.asarray(y)
    from maddening.core.params import ParamSpec

    gm = _pair({"a": ParamSpec(), "b": ParamSpec()}, a=float(2.0 / unit), b=1.0)
    res = fit_lm(gm, lambda p: Aj @ jnp.stack([p["nodes"]["h"]["a"] * unit,
                                               p["nodes"]["h"]["b"]]) - yj, n_iter=50)
    a = float(res.params["nodes"]["h"]["a"]) * unit
    assert res.converged, (unit, res.n_iter, list(res.losses))
    assert abs(a - optimum[0]) <= 2.0 ** 4 * float(np.finfo(np.float32).eps) * abs(optimum[0])


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
@pytest.mark.parametrize("truth", [0.5, 2.0], ids=["lower", "upper"])
def test_a_truth_on_either_bound_converges(truth, x64):
    """Noiseless, the truth exactly on a bound of an identity parameter: at
    the end of the fit the on-bound gradient is the residual's rounding, and
    its sign decided whether "it could lower the loss by moving into the
    range" -- converged on one bound and not the other in float32, the
    opposite pair under x64 (part B of the reproducer).  A pull whose own
    Newton step is within ``step_tol`` counts as zero."""
    from maddening.core.params import ParamSpec

    rng = np.random.default_rng(1)
    B = rng.normal(size=(30, 2))
    with (_x64() if x64 else contextlib.nullcontext()):
        ft = jnp.float64 if x64 else jnp.float32
        gm = _pair({"a": ParamSpec(bounds=(0.5, 2.0)), "b": ParamSpec()}, a=1.25, b=1.0)
        Bj, yb = jnp.asarray(B, ft), jnp.asarray(B @ np.array([truth, 0.7]), ft)
        res = fit_lm(gm, lambda p: Bj @ jnp.stack([p["nodes"]["h"]["a"],
                                                   p["nodes"]["h"]["b"]]) - yb, n_iter=100)
    assert res.converged, (truth, x64, res.n_iter, res.best_loss)
    assert float(res.params["nodes"]["h"]["a"]) == truth
    assert float(res.params["nodes"]["h"]["b"]) == pytest.approx(0.7, rel=1e-5)


def test_only_an_on_bound_pull_within_step_tol_counts_as_zero(monkeypatch):
    """What the on-bound test is handed, both ways: a coordinate started on
    its bound with the truth well inside keeps its inward gradient (the
    pull's own Newton step is far beyond ``step_tol``), and at a truth
    exactly on the bound the pull left at the end is zeroed."""
    from maddening import sysid
    from maddening.core.params import ParamSpec

    seen = []
    real = sysid._CoordinateBounds.inward_descent

    def spy(self, theta, g, physical=None, rel_tol=0.0):
        seen.append((np.asarray(theta, np.float64).copy(), np.asarray(g, np.float64).copy()))
        return real(self, theta, g, physical, rel_tol)

    monkeypatch.setattr(sysid._CoordinateBounds, "inward_descent", spy)
    rng = np.random.default_rng(1)
    B = rng.normal(size=(30, 2))
    for truth, start, expect_pull in ((1.25, 0.5, True), (2.0, 1.25, False)):
        seen.clear()
        gm = _pair({"a": ParamSpec(bounds=(0.5, 2.0)), "b": ParamSpec()}, a=start, b=1.0)
        Bj, yb = jnp.asarray(B, jnp.float32), jnp.asarray(B @ np.array([truth, 0.7]), jnp.float32)
        res = fit_lm(gm, lambda p: Bj @ jnp.stack([p["nodes"]["h"]["a"],
                                                   p["nodes"]["h"]["b"]]) - yb, n_iter=100)
        assert res.converged
        if expect_pull:
            theta, g = seen[0]
            assert theta[0] == 0.5 and g[0] < 0.0          # on the bound, pulled inward
        else:
            theta, g = seen[-1]
            assert theta[0] == 2.0 and g[0] == 0.0         # rounding, zeroed
