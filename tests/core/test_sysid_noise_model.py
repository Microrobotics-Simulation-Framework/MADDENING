"""``sysid.fim`` / ``fit_lm`` with a per-observation noise model given as a
pytree (the documented form), and ``fit_lm`` returning cleanly when no
step is ever accepted.

Originally written from the independent audit of 2026-09-16 (round 1; report and
reproducers under ``benchmarks/results/audit1/``).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode


def _spring_gm(**kw):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0,
                                 initial_position=1.0, **kw))
    gm.compile()
    return gm


def _pytree_residual(gm):
    step, ext = gm._compiled_step, gm._default_external_inputs()
    target = step(gm._state, ext, gm.params)["s"]

    def residual(p):
        s = step(gm._state, ext, p)["s"]
        return {"pos": s["position"] - target["position"],
                "vel": s["velocity"] - target["velocity"]}
    return residual


def test_fim_noise_std_pytree_dict():
    from maddening.sysid import fim
    gm = _spring_gm()
    residual = _pytree_residual(gm)
    unit = fim(residual, gm.params, mask=gm.trainable_mask(), noise_std=1.0)
    per = fim(residual, gm.params, mask=gm.trainable_mask(),
              noise_std={"pos": 0.1, "vel": 1.0})
    # The velocity residual keeps its weight; the position row is 10x more
    # informative, so per-leaf noise changes the matrix.
    assert np.all(np.isfinite(np.asarray(per.fim)))
    assert not np.allclose(np.asarray(per.fim), np.asarray(unit.fim))
    # scalar forms that must keep working
    for sd in (2.0, np.float32(2.0), jnp.float32(2.0)):
        fim(residual, gm.params, mask=gm.trainable_mask(), noise_std=sd)


def test_fit_lm_noise_std_pytree_dict():
    from maddening.sysid import fit_lm
    gm = _spring_gm()
    residual = _pytree_residual(gm)
    start = jax.tree.map(lambda x: x, gm.params)
    start["nodes"]["s"]["stiffness"] = jnp.float32(45.0)
    res = fit_lm(gm, residual, params=start, n_iter=5,
                 noise_std={"pos": 0.1, "vel": 1.0})
    gm.check_params(res.params)
    assert res.losses[-1] <= res.losses[0]


def test_fit_lm_never_accepting_a_step_returns_cleanly():
    """A residual no step can lower: the loop must exit as not converged
    rather than touch an unset ``step_norm``."""
    from maddening.sysid import fit_lm
    gm = _spring_gm()
    res = fit_lm(gm, lambda p: jnp.ones(3, jnp.float32), n_iter=4)
    assert res.converged is False and res.n_iter == 1


# ---------------------------------------------------------------------------
# ``fit_lm`` must not evaluate the residual for a structure it can trace
# ---------------------------------------------------------------------------


def _counting_residual(gm):
    """``(residual_fn, log)``; ``log`` gains one entry per call, recording
    whether the parameters it was handed were tracers.

    Counting *entries* alone cannot separate the two things that matter: a
    ``jax.jit`` or ``jax.eval_shape`` trace enters the function and runs no
    rollout, while a plain call runs one.  The tracer flag is what makes
    "this entry cost a rollout" assertable.
    """
    log: list[bool] = []
    inner = _pytree_residual(gm)

    def residual(p):
        log.append(isinstance(p["nodes"]["s"]["stiffness"], jax.core.Tracer))
        return inner(p)

    return residual, log


def test_fit_lm_without_a_noise_model_never_evaluates_the_residual_eagerly():
    """``fit_lm`` computed ``r_probe = residual_fn(...)`` once per call and
    used it only for its pytree structure -- which
    ``_inverse_noise_std`` discards on its first line when ``noise_std`` is
    ``None``.  A whole extra rollout per call, buying nothing.

    Every entry into ``residual_fn`` must now be a trace.  The count is
    pinned too, at 3 (the ``jacfwd`` pair inside ``residual_and_jac`` plus
    ``residual_only``), because "all entries are traces" is also true of a
    ``fit_lm`` that stopped calling the residual altogether.
    """
    from maddening.sysid import fit_lm
    gm = _spring_gm()
    residual, log = _counting_residual(gm)
    fit_lm(gm, residual, n_iter=3, notify_every=0)
    assert all(log), f"{log.count(False)} of {len(log)} entries ran a rollout"
    assert len(log) == 3, len(log)


def test_fit_lm_at_zero_iterations_does_not_touch_the_residual_at_all():
    """The sharpest form of the same statement, and the one that cannot be
    confused with a trace: with no iterations to run there is nothing to
    trace either, so the count is exactly zero.  It was 1."""
    from maddening.sysid import fit_lm
    gm = _spring_gm()
    residual, log = _counting_residual(gm)
    res = fit_lm(gm, residual, n_iter=0, notify_every=0)
    assert log == [], log
    assert res.n_iter == 0 and len(res.losses) == 0


def test_a_noise_model_costs_fit_lm_a_trace_not_a_rollout():
    """``noise_std`` genuinely needs the residual's structure, and
    ``jax.eval_shape`` gets it by tracing: one more entry, still no
    rollout.  Pinned so that a future "optimisation" cannot reintroduce
    the eager call under cover of needing the value."""
    from maddening.sysid import fit_lm
    gm = _spring_gm()
    residual, log = _counting_residual(gm)
    fit_lm(gm, residual, n_iter=3, noise_std={"pos": 0.1, "vel": 1.0},
           notify_every=0)
    assert all(log), f"{log.count(False)} of {len(log)} entries ran a rollout"
    assert len(log) == 4, len(log)


def test_removing_the_probe_left_the_noise_weighting_alone():
    """The structure ``jax.eval_shape`` reports must be the structure the
    evaluated residual had, or the weights land on the wrong entries.

    Asserted against the arithmetic rather than against another call of
    the same code: a per-leaf sigma of ``{"pos": 0.1, "vel": 1.0}`` must
    multiply the ``pos`` residual by exactly 10 and leave ``vel``, so the
    weighted ``0.5||r||^2`` at the starting point is
    ``0.5 * (100 * pos^2 + vel^2)``.
    """
    from maddening.sysid import fit_lm
    # ``rest_length=0.5`` against the fixture's ``initial_position=1.0``:
    # the shared fixture starts *at* the rest length with zero velocity, so
    # one step of ``F = -k(x - L) - c v`` moves nothing whatever ``k`` is
    # and both residual leaves are exactly 0.0.  A zero residual cannot
    # show a weight landing on the wrong leaf, which is what this test is
    # for; the assertion below is what caught it.
    gm = _spring_gm(rest_length=0.5)
    residual = _pytree_residual(gm)
    start = jax.tree.map(lambda x: x, gm.params)
    start["nodes"]["s"]["stiffness"] = jnp.float32(45.0)
    r0 = residual(start)
    expected = 0.5 * (100.0 * float(r0["pos"]) ** 2 + float(r0["vel"]) ** 2)
    assert float(r0["vel"]) != 0.0, (
        "the starting residual is zero; this asserts nothing")
    res = fit_lm(gm, residual, params=start, n_iter=1, notify_every=0,
                 noise_std={"pos": 0.1, "vel": 1.0})
    assert float(res.losses[0]) == pytest.approx(expected, rel=1e-5), (
        float(res.losses[0]), expected)
