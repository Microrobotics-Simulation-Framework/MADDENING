"""``run_adaptive`` and ``run_adaptive_scan`` accept the same attempts.

Both call :func:`maddening.core.simulation.adaptive.step_decision`: an
attempt is accepted within tolerance or when it was already made at
``dt_min``, and the next timestep is the PI controller's, clipped to
``[dt_min, dt_max]``.  ``run_adaptive`` used to carry its own copy, which
accepted a *rejected* attempt larger than ``dt_min`` whenever shrinking it
would reach ``dt_min``: on an explicit-Euler decay held at ``dt_min`` it
took a failed first step of 0.008 instead of 0.005, overshot ``t_end`` to
0.103, and ended 16% away from the scan's state.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode

KW = dict(dt_initial=0.04, atol=1e-9, rtol=1e-9, dt_min=0.005, dt_max=0.04)
T_END = 0.1


class _Decay(SimulationNode):
    """Explicit Euler on ``x' = -k x``: its step-doubling error grows with ``dt``."""

    def __init__(self, name, k):
        super().__init__(name, 0.01, k=jnp.float32(k))

    def initial_state(self):
        return {"x": jnp.ones(1, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] - jnp.asarray(dt, jnp.float32) * self.params["k"] * state["x"]}


def _graph():
    gm = GraphManager()
    gm.add_node(_Decay("d", 50.0))
    gm.compile()
    return gm


def test_both_steppers_take_the_same_steps_at_dt_min():
    """A tolerance no step meets: every accepted step is at ``dt_min``, in both."""
    gm = _graph()
    with pytest.warns(UserWarning, match="hit dt_min"):
        state, info = gm.run_adaptive(T_END, **KW)
    assert info["dt_history"] == [KW["dt_min"]] * 20, info["dt_history"]
    assert info["t_history"][-1] == pytest.approx(T_END, abs=1e-12)
    scan = _graph()
    final, _history, sinfo = scan.run_adaptive_scan(T_END, max_steps=60, **KW)
    assert int(sinfo["n_steps"]) == info["n_steps"]
    assert float(sinfo["final_t"]) == pytest.approx(T_END, rel=1e-6)
    np.testing.assert_allclose(np.asarray(final["d"]["x"]), np.asarray(state["d"]["x"]),
                               rtol=1e-6)


def test_an_attempt_within_tolerance_is_accepted_by_both():
    """The ordinary regime is unchanged: the same accepted steps, no warning."""
    loose = dict(KW, atol=1e-1, rtol=1e-1)
    gm = _graph()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        state, info = gm.run_adaptive(T_END, **loose)
    scan = _graph()
    final, _history, sinfo = scan.run_adaptive_scan(T_END, max_steps=60, **loose)
    assert int(sinfo["n_steps"]) == info["n_steps"]
    np.testing.assert_allclose(np.asarray(final["d"]["x"]), np.asarray(state["d"]["x"]),
                               rtol=1e-5)
