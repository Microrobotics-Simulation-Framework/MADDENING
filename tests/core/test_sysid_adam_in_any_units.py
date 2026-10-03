"""Adam's ``eps`` is relative, so a fit does not depend on the loss's units.

Adam's step ``m_hat / (sqrt(v_hat) + eps)`` is invariant to the gradient's
scale except through ``eps``, which :func:`~maddening.sysid.fit` and
:func:`~maddening.sysid.fit_multiple_shooting` applied in the loss's own units
(``1e-8``).  A loss near ``1e-12`` -- a position residual in metres at
micrometre scale, squared -- has gradients far below that, the denominator was
``eps``, and the step shrank with it: a spring's damping fitted from 2.0
towards 0.5 on such a loss did not move in 60 iterations, where the same loss
at scale one recovered it.  Both fitters now take their gradients in the frame
of the run's first gradient (an exact power of two), so the iterates are the
same, to the bit, for the loss scaled by any power of two.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import fit, fit_multiple_shooting, observations_from_history

DT, N = 0.01, 40


def _spring(**kw):
    p = dict(stiffness=30.0, damping=2.0, mass=1.0, rest_length=1.0,
             initial_position=0.2, initial_velocity=0.0)
    p.update(kw)
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="s", timestep=DT, **p))
    gm.compile()
    return gm


def _only_damping(gm):
    mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
    mask["nodes"]["s"]["damping"] = True
    return mask


@pytest.fixture(scope="module")
def record():
    return _spring(damping=0.5).run_scan_with_history(N)[1]["s"]["position"]


def _fit_at(scale, record, hold):
    obs = record

    def loss(p):
        # A fresh graph per evaluation: ``run_scan*`` stores its final state.
        x = _spring().run_scan_with_history(N, params=p)[1]["s"]["position"]
        return scale * jnp.sum((x - obs) ** 2)

    gm = _spring()
    return fit(gm, jax.jit(loss), mask=_only_damping(gm), n_iter=40, lr=0.05,
               hold_undetermined=hold)


@pytest.mark.parametrize("hold", [False, True])
def test_fit_takes_the_same_steps_for_a_loss_in_any_units(record, hold):
    ref = _fit_at(1.0, record, hold)
    damping = float(ref.params["nodes"]["s"]["damping"])
    assert abs(damping - 2.0) > 0.5, "the reference fit itself did not move"
    for k in (-40, 30):
        s = 2.0 ** k
        res = _fit_at(s, record, hold)
        assert np.array_equal(np.asarray(res.params["nodes"]["s"]["damping"]),
                              np.asarray(ref.params["nodes"]["s"]["damping"])), k
        assert np.array_equal(res.losses, (ref.losses * s).astype(res.losses.dtype)), k
        assert res.best_iteration == ref.best_iteration


@pytest.mark.parametrize("k", [-100, -120])
def test_fit_takes_the_same_steps_for_a_loss_whose_gradient_flushes(record, k):
    """Smaller still: at ``2**-100`` the first gradient is below ``tiny / eps``
    and is taken with a power-of-two cotangent (``_gradient_lift``), and at
    ``2**-120`` the loss's own value flushes to zero as well.  The cotangent
    is exact, so every iterate is the reference's, to the bit, either way;
    the gradient used to read zero there and the fit did not move
    (MADD-ANO-167).  (At ``2**-120`` the losses all read 0.0, so the returned
    iterate is the last -- a tie goes to the later one -- not the reference's
    lowest; the iterates themselves are compared.)"""
    import warnings

    def iterates(scale):
        seen = []
        obs = record

        def loss(p):
            x = _spring().run_scan_with_history(N, params=p)[1]["s"]["position"]
            return scale * jnp.sum((x - obs) ** 2)

        gm = _spring()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # a flushed loss says so
            res = fit(gm, jax.jit(loss), mask=_only_damping(gm), n_iter=40, lr=0.05,
                      hold_undetermined=False,
                      callback=lambda i, l, p: seen.append(np.asarray(p["nodes"]["s"]["damping"])))
        return res, np.stack(seen)

    ref, ref_steps = iterates(1.0)
    res, steps = iterates(2.0 ** k)
    assert abs(float(ref_steps[-1]) - 2.0) > 0.5, "the reference fit itself did not move"
    assert np.array_equal(steps, ref_steps), k
    if k == -100:
        assert np.array_equal(res.losses, (ref.losses * 2.0 ** k).astype(res.losses.dtype))
        assert np.array_equal(np.asarray(res.params["nodes"]["s"]["damping"]),
                              np.asarray(ref.params["nodes"]["s"]["damping"]))


def test_fit_multiple_shooting_takes_the_same_steps_for_a_loss_in_any_units():
    gm_truth = _spring(damping=0.5)
    s0 = gm_truth._user_state(gm_truth._state)  # noqa: SLF001
    obs = observations_from_history(s0, gm_truth.run_scan_with_history(N)[1])

    def run(scale):
        # The observation scaled by ``scale`` and the continuity weight by
        # ``scale**2``: the whole loss is the reference loss times ``scale**2``.
        gm = _spring()
        res, ws = fit_multiple_shooting(
            gm, obs, obs_fn=lambda h: h["s"]["position"] * scale, window=10,
            mask=_only_damping(gm), n_iter=30, lr=0.05,
            continuity_weight=scale * scale, hold_undetermined=False)
        return res, ws

    ref, ref_ws = run(1.0)
    assert abs(float(ref.params["nodes"]["s"]["damping"]) - 2.0) > 0.2
    s = 2.0 ** -20
    res, ws = run(s)
    assert np.array_equal(np.asarray(res.params["nodes"]["s"]["damping"]),
                          np.asarray(ref.params["nodes"]["s"]["damping"]))
    for a, b in zip(jax.tree.leaves(ws), jax.tree.leaves(ref_ws)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_the_guards_curvature_is_the_losss_whatever_the_lift(record, monkeypatch):
    """The identifiability guard's tolerance pairs the objective's largest
    curvature with each coordinate's rounding, so it is in the loss's units.
    A lifted run takes its Hessian-vector products from the lifted gradient,
    a power of two larger, and must unlift them: the spring's ``(k, c, m)``
    scale degeneracy fitted on the loss scaled by ``2**-120`` reports a
    curvature ``2**-120`` times the unscaled fit's (to within a factor of
    two, the selected iterates differing), not the lift times that."""
    import warnings

    from maddening import sysid

    curvatures = []
    real = sysid._hold_tolerance

    def spy(loss_sel, grad_sel, curvature, *args, **kwargs):
        curvatures.append(float(curvature))
        return real(loss_sel, grad_sel, curvature, *args, **kwargs)

    monkeypatch.setattr(sysid, "_hold_tolerance", spy)

    def fit_at(scale):
        obs = record

        def loss(p):
            x = _spring().run_scan_with_history(N, params=p)[1]["s"]["position"]
            return scale * jnp.sum((x - obs) ** 2)

        from maddening.core.params import ParamSpec

        gm = _spring()
        # ``log`` on all three, so the degeneracy is a fixed direction the
        # guard can hold (with an identity damping it curves, and the guard
        # rightly holds nothing).
        gm.set_param_spec("s", "damping", ParamSpec(bounds=(0.0, None), transform="log"))
        mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
        for key in ("stiffness", "damping", "mass"):
            mask["nodes"]["s"][key] = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return fit(gm, jax.jit(loss), mask=mask, n_iter=60, lr=0.05)

    fit_at(1.0)
    fit_at(2.0 ** -120)
    assert len(curvatures) == 2, curvatures
    ratio = curvatures[1] / curvatures[0] / 2.0 ** -120
    assert 0.5 <= ratio <= 2.0, ratio
