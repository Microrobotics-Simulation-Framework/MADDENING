"""Three user-facing defaults stay absolute in their quantity's units, on purpose.

The numeric-constants gate classifies every small constant in the numerical
core.  Most were relative, and the absolute ones that could change a result
silently were made relative.  These three are kept and registered instead:
they are user-facing tolerances that scipy's ``solve_ivp`` and the old
calibration loops share, and changing them would move every run.

* ``run_adaptive`` / ``run_adaptive_scan`` / ``AdaptiveConfig``: ``atol=1e-6``
  in state units and ``dt_min=1e-8`` in seconds.
* ``calibrate``'s ``tolerance=1e-6`` in loss units and
  ``tune_coupling_params``' ``accuracy_threshold=1e-3`` in metric units
  (both deprecated, removed in 0.5.0).

These tests pin the documented behaviour, and the workaround each entry names,
so that a fix (or a drifted figure) fails here and the entry gets updated.
"""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode


class _Decay(SimulationNode):
    """Explicit Euler on ``x' = -x``: the step-doubling error is ``x dt**2 / 4``."""

    def __init__(self, x0):
        super().__init__("d", 0.01)
        self._x0 = x0

    def halo_width(self):
        return {}

    def initial_state(self):
        return {"x": jnp.asarray([self._x0], jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] - dt * state["x"]}


def _steps(x0, **kw):
    gm = GraphManager()
    gm.add_node(_Decay(x0))
    gm.compile()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _state, info = gm.run_adaptive(1.0, dt_initial=0.01, dt_max=1.0, **kw)
    return info["n_steps"]


def test_the_adaptive_default_atol_is_absolute_in_state_units():
    """At scale one the controller works to ``rtol``; at ``2**-30`` (~9e-10)
    the default ``atol=1e-6`` is a thousand times the state, so every step is
    accepted and grows by ``max_factor`` until ``t_end``.  Writing ``atol`` in
    the state's units (``1e-6 * 2**-30``) restores the scale-one sequence."""
    s = 2.0 ** -30
    at_one = _steps(1.0)
    small_default = _steps(s)
    small_scaled = _steps(s, atol=1e-6 * s)
    assert at_one > 2 * small_default, (at_one, small_default)
    assert small_default <= 6           # 0.01, 0.05, 0.25, then t_end: nothing rejected
    assert small_scaled == at_one


def test_calibrate_stops_on_an_absolute_loss_threshold():
    """A loss already below ``tolerance=1e-6`` in its own units is
    "converged" before a single step; a tolerance in the loss's units fits."""
    from maddening.core.simulation.calibration import calibrate

    def forward(p):
        return p["a"] * 1e-4                      # a quantity in small units

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        res = calibrate(forward, {"a": jnp.asarray(3.0)}, jnp.asarray(0.0),
                        n_iters=50, learning_rate=1e6)
        assert res.converged and len(res.loss_history) == 1
        assert float(res.params["a"]) == pytest.approx(3.0)
        fitted = calibrate(forward, {"a": jnp.asarray(3.0)}, jnp.asarray(0.0),
                           n_iters=200, learning_rate=1e6, tolerance=1e-6 * 1e-8)
    assert abs(float(fitted.params["a"])) < 0.3
