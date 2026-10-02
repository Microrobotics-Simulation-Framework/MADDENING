"""Truth recovery: a fitter that says ``converged=True`` has found the truth.

The oracle, over generated problems: draw a true parameter anywhere inside
its bounds (near them too) and a start anywhere in the box, generate
noiseless data, run the fitter, and require **either recovery of the truth
to a stated tolerance, or ``converged=False``** -- never ``converged=True``
at a point that is not the constrained optimum.

The problems are built so that the truth is the *only* constrained
stationary point in the box: each trainable coordinate has its own block of
residuals, ``g(p) - g(p*)`` with ``g`` strictly monotone and nonlinear, so the
loss is a sum of one-dimensional unimodal losses.  Nonlinear so that a
Gauss-Newton step can overshoot -- past a bound, which is how
``fit_lm`` used to stop on the bound of a clipped coordinate and call it
converged (audit_040_p4_5/fmu-sysid/repro_fit_lm_stuck_at_bound.py).  Every
transform takes part: a bounded identity leaf (``transform=None``, clipped),
``log`` and ``logit``.

Per push: ``fit_lm`` on the closed-form problem.  Slow: the same for ``fit``
(Adam, with a ``tol``) and ``fit_lm`` with more draws, and the spring graph
itself under all three fitters.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import fit, fit_lm, fit_multiple_shooting, observations_from_history

from tests.conftest import EXAMPLES_COSTLY

T = np.linspace(0.1, 2.0, 8)

#: Where in its range a drawn value sits: anywhere, and near either edge.
_FRACTION = st.one_of(
    st.floats(0.02, 0.98),
    st.sampled_from([1e-4, 1e-3, 1e-2, 0.99, 0.999, 0.9999]),
)

#: The three ways a trainable leaf can be bounded, and its range.
_SPECS = {
    "clip": (ParamSpec(bounds=(0.0, 5.0)), (0.0, 5.0)),
    "log": (ParamSpec(bounds=(0.0, None), transform="log"), (0.05, 20.0)),
    "logit": (ParamSpec(bounds=(0.5, 2.0), transform="logit"), (0.5, 2.0)),
}
KEYS = ("stiffness", "damping", "rest_length")


def _graph(kinds):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.0,
                                 rest_length=1.0, initial_position=0.2))
    for key, kind in zip(KEYS, kinds):
        gm.set_param_spec("s", key, _SPECS[kind][0])
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
    gm.compile()
    return gm


def _value(kind, fraction):
    lo, hi = _SPECS[kind][1]
    return lo + fraction * (hi - lo)


def _blocks(p):
    """One strictly monotone, nonlinear block of residuals per coordinate,
    with a finite, nowhere-vanishing derivative over every range drawn (an
    ``exp(-c t)`` block underflowed to an exactly flat plateau that a fit
    then rightly called stationary, which is not what this oracle is
    about).  The concave blocks make a Gauss-Newton step from above
    overshoot below the truth -- past a lower bound of 0 for a clipped
    coordinate -- and the convex one overshoots from below."""
    q = p["nodes"]["s"]
    t = jnp.asarray(T, jnp.float32)
    return jnp.concatenate([
        jnp.log1p(q["damping"] * t),
        jnp.sqrt(q["stiffness"] + 0.1) * t,
        q["rest_length"] * t + 0.2 * q["rest_length"] ** 2 * t,
    ])


def _start(gm, kinds, fractions):
    p = jax.tree.map(lambda x: x, gm.params)
    for key, kind, fraction in zip(KEYS, kinds, fractions):
        p["nodes"]["s"][key] = jnp.asarray(_value(kind, fraction), jnp.float32)
    return p


def _mask(gm):
    mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
    for key in KEYS:
        mask["nodes"]["s"][key] = True
    return mask


def _recovered(res, truth, kinds) -> tuple[bool, list]:
    errors = []
    ok = True
    for key, kind in zip(KEYS, kinds):
        lo, hi = _SPECS[kind][1]
        got = float(res.params["nodes"]["s"][key])
        want = float(truth["nodes"]["s"][key])
        # 1e-3 of the coordinate's range: a fit at its float32 floor is far
        # inside this, and a fit stuck on a bound is far outside it.
        tolerance = 1e-3 * (hi - lo)
        errors.append((key, got, want))
        ok &= abs(got - want) <= tolerance
    return ok, errors


_PROBLEM = dict(
    kinds=st.tuples(*[st.sampled_from(sorted(_SPECS))] * 3),
    truth_at=st.tuples(_FRACTION, _FRACTION, _FRACTION),
    start_at=st.tuples(_FRACTION, _FRACTION, _FRACTION),
)


def _check_fit_lm(kinds, truth_at, start_at):
    gm = _graph(kinds)
    truth = _start(gm, kinds, truth_at)
    data = _blocks(truth)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a declined hold says so
        res = fit_lm(gm, lambda p: _blocks(p) - data, params=_start(gm, kinds, start_at),
                     mask=_mask(gm), n_iter=60)
    ok, errors = _recovered(res, truth, kinds)
    note(f"kinds={kinds} converged={res.converged} n_iter={res.n_iter} "
         f"best_loss={res.best_loss} (key, got, truth)={errors}")
    assert ok or not res.converged, errors


@given(**_PROBLEM)
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
def test_fit_lm_converged_means_the_truth(kinds, truth_at, start_at):
    _check_fit_lm(kinds, truth_at, start_at)


# Per push: tests/property/test_sysid_truth_recovery.py::test_fit_lm_converged_means_the_truth
@pytest.mark.slow  # a fit compiled per example, four times the per-push draws: over 5 s on CI
@given(**_PROBLEM)
@settings(max_examples=4 * EXAMPLES_COSTLY, deadline=None, derandomize=True)
def test_fit_lm_converged_means_the_truth_broadly(kinds, truth_at, start_at):
    _check_fit_lm(kinds, truth_at, start_at)


# Per push: tests/property/test_sysid_truth_recovery.py::test_fit_lm_converged_means_the_truth
#   (Adam's ``converged`` is the ``tol`` test, which the per-push oracle's
#   fitter shares; the clipped-coordinate projection is pinned per push by
#   tests/core/test_sysid_bounded_coordinates_and_residual_scale.py::test_fit_brings_a_clipped_coordinate_back_into_its_range.)
@pytest.mark.slow  # hundreds of Adam steps per example: over 5 s on CI
@given(**_PROBLEM)
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
def test_fit_converged_means_the_truth(kinds, truth_at, start_at):
    gm = _graph(kinds)
    truth = _start(gm, kinds, truth_at)
    data = _blocks(truth)
    start = _start(gm, kinds, start_at)
    loss = jax.jit(lambda p: 0.5 * jnp.sum((_blocks(p) - data) ** 2))
    tol = 1e-10 * max(float(loss(start)), 1e-30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = fit(gm, loss, params=start, mask=_mask(gm), n_iter=400, lr=0.05, tol=tol)
    ok, errors = _recovered(res, truth, kinds)
    note(f"kinds={kinds} converged={res.converged} best_loss={res.best_loss} {errors}")
    assert ok or not res.converged, errors


# ---------------------------------------------------------------------------
# The spring graph itself: damping (clipped) from anywhere to anywhere
# ---------------------------------------------------------------------------

N_STEPS = 60


def _spring(damping):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=damping, mass=1.0,
                                 rest_length=1.0, initial_position=0.2))
    gm.compile()
    return gm


def _only_damping(gm):
    mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
    mask["nodes"]["s"]["damping"] = True
    return mask


# Per push: tests/core/test_sysid_bounded_coordinates_and_residual_scale.py::test_fit_lm_brings_a_clipped_coordinate_back_into_its_range
#   and ::test_fit_multiple_shooting_brings_a_clipped_coordinate_back_into_its_range
@pytest.mark.slow  # a graph rollout per residual and a fit per example: over 5 s on CI
@given(truth=st.floats(0.01, 6.0), start=st.floats(0.0, 15.0),
       fitter=st.sampled_from(["fit_lm", "fit", "fit_multiple_shooting"]))
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
def test_every_fitter_on_the_spring_converged_means_the_truth(truth, start, fitter):
    source = _spring(truth)
    s0 = source._user_state(source._state)  # noqa: SLF001
    obs = observations_from_history(s0, source.run_scan_with_history(N_STEPS)[1])
    position = obs["s"]["position"][1:]
    gm = _spring(start)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if fitter == "fit_lm":
            res = fit_lm(gm, lambda p: _spring(start).run_scan_with_history(
                N_STEPS, params=p)[1]["s"]["position"] - position,
                mask=_only_damping(gm), n_iter=40)
        elif fitter == "fit":
            loss = jax.jit(lambda p: 0.5 * jnp.sum((_spring(start).run_scan_with_history(
                N_STEPS, params=p)[1]["s"]["position"] - position) ** 2))
            res = fit(gm, loss, mask=_only_damping(gm), n_iter=200, lr=0.2,
                      tol=1e-12 * max(float(loss(gm.params)), 1e-30))
        else:
            res, _ = fit_multiple_shooting(
                gm, obs, obs_fn=lambda h: h["s"]["position"], window=20,
                mask=_only_damping(gm), n_iter=200, lr=0.2, lr_states=1e-6, tol=1e-12)
    got = float(res.params["nodes"]["s"]["damping"])
    note(f"{fitter}: truth={truth} start={start} got={got} converged={res.converged}")
    assert abs(got - truth) <= 1e-3 * (1.0 + truth) or not res.converged
