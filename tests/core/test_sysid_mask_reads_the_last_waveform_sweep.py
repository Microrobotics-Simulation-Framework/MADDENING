"""``windowed_loss(mask_unconverged=True)`` reads the last waveform sweep's verdict, as documented.

On a sub-cycling group with ``waveform_iterations > 1`` the report slots
the mask reads hold the last sweep's solve.  A window in which an earlier
sweep exited at ``max_iterations`` unconverged while the last converged is
therefore kept -- the same step on which ``strict_convergence=True``, which
checks every sweep, raises, and which ``coupling_diagnostics()`` reports
with ``iterations == max_iterations`` beside ``converged=True``.  The
docstring did not say so (the round-5 audit); it does now, and this pins
what it says.  (Masking on every sweep is a decided 0.5.0 item.)
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.sysid import observations_from_history, windowed_loss


class _Relay(SimulationNode):
    def __init__(self, name, timestep, g, b):
        super().__init__(name, timestep, g=jnp.float32(g), b=jnp.float32(b))

    def initial_state(self):
        return {"x": jnp.zeros(1, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(1,), dtype=jnp.float32,
                                       default=jnp.zeros(1, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["g"] * boundary_inputs["u"] + p["b"]}


def _graph(**group):
    gm = GraphManager()
    gm.add_node(_Relay("A", 0.1, 0.7, 1.0))
    gm.add_node(_Relay("B", 0.05, 0.7, 0.5))
    gm.add_edge("A", "B", "x", "u")
    gm.add_edge("B", "A", "x", "u")
    gm.add_coupling_group(["A", "B"], max_iterations=8, tolerance=1e-4, subcycling=True,
                          waveform_iterations=2, **group)
    gm.compile()
    return gm


def test_a_window_whose_first_sweep_hit_its_cap_is_kept_by_the_mask():
    gm = _graph()
    gm.step()
    d = gm.coupling_diagnostics()["A+B"]
    assert d["iterations"] == 8 and d["converged"], (
        "fixture premise: the first sweep exhausts its cap, the last converges", dict(d))
    with pytest.raises(Exception, match="without converging"):
        _graph(strict_convergence=True).step()
    gm2 = _graph()
    init = {n: dict(v) for n, v in gm2._state.items() if n != "_meta"}
    _, hist = gm2.run_scan_with_history(4)
    obs = observations_from_history(init, hist)
    gm3 = _graph()
    p = jax.tree.map(lambda v: v, gm3.params)
    p["nodes"]["A"]["b"] = p["nodes"]["A"]["b"] + 0.1
    masked = float(windowed_loss(gm3, p, obs, obs_fn=lambda s: s["A"]["x"], window=1,
                                 mask_unconverged=True))
    plain = float(windowed_loss(gm3, p, obs, obs_fn=lambda s: s["A"]["x"], window=1,
                                mask_unconverged=False))
    assert masked == plain > 0.0, (masked, plain)
