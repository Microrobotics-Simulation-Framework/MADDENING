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
