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
