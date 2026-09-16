"""Audit: sysid on multirate graphs, IFT gradient vs finite differences, adaptive+params."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.ball import BallNode


def _multirate():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, stiffness=30.0, damping=2.0, initial_position=1.5))
    gm.add_node(SpringDamperNode("b", 0.02, stiffness=10.0, damping=1.0, initial_position=0.5))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.compile()
    return gm


def test_windowed_loss_zero_at_truth_on_multirate_graph():
    from maddening.sysid import observations_from_history, windowed_loss
    gm = _multirate()
    assert gm.is_multirate
    init = {n: gm.get_node_state(n) for n in gm.node_names}
    _, hist = gm.run_scan_with_history(40)
    obs = observations_from_history(init, hist)
    gm.reset_state()
    loss = windowed_loss(gm, gm.params, obs, obs_fn=lambda h: h["b"]["position"], window=8)
    assert float(loss) == pytest.approx(0.0, abs=1e-9)
    # and positive elsewhere
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["b"]["stiffness"] = jnp.float32(20.0)
    assert float(windowed_loss(gm, p, obs, obs_fn=lambda h: h["b"]["position"], window=8)) > 0


def test_windowed_loss_with_sample_every_on_multirate_graph():
    from maddening.sysid import observations_from_history, windowed_loss
    gm = _multirate()
    init = {n: gm.get_node_state(n) for n in gm.node_names}
    _, hist = gm.run_scan_with_history(40)
    obs_full = observations_from_history(init, hist)
    obs = jax.tree.map(lambda x: x[::2], obs_full)          # T = 21, sample_every = 2
    gm.reset_state()
    loss = windowed_loss(gm, gm.params, obs, obs_fn=lambda h: h["b"]["position"],
                         window=5, sample_every=2)
    assert float(loss) == pytest.approx(0.0, abs=1e-9)


def _coupled(**kw):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, stiffness=30.0, damping=2.0, initial_position=1.5))
    gm.add_node(SpringDamperNode("b", 0.01, stiffness=10.0, damping=1.0, initial_position=0.5))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=30, tolerance=1e-10, **kw)
    gm.compile()
    return gm


@pytest.mark.parametrize("kw", [dict(), dict(acceleration="aitken"), dict(acceleration="iqn-ils"),
                                dict(iteration_mode="jacobi")])
def test_ift_gradient_matches_finite_differences_over_scan(kw):
    gm = _coupled(**kw)
    step, ext = gm._build_step_fn(), gm._default_external_inputs()

    def loss(p):
        final, _ = jax.lax.scan(lambda s, _: (step(s, ext, p), None), gm._state, None, length=20)
        return final["a"]["position"] ** 2 + final["b"]["velocity"] ** 2

    g = jax.grad(loss)(gm.params)["nodes"]["b"]["stiffness"]
    h = 1e-2
    pp = jax.tree.map(lambda x: x, gm.params); pm = jax.tree.map(lambda x: x, gm.params)
    pp["nodes"]["b"]["stiffness"] = jnp.float32(10.0 + h)
    pm["nodes"]["b"]["stiffness"] = jnp.float32(10.0 - h)
    fd = (float(loss(pp)) - float(loss(pm))) / (2 * h)
    assert float(g) == pytest.approx(fd, rel=2e-2, abs=1e-4), (float(g), fd)


def test_run_adaptive_uses_params():
    gm = _coupled()
    s0 = gm._state
    r1 = gm.run_adaptive(0.05, dt_initial=0.01)
    a = gm._state["a"]["position"]
    gm._state = s0
    gm.params["nodes"]["b"]["stiffness"] = jnp.float32(50.0)
    gm.run_adaptive(0.05, dt_initial=0.01)
    b = gm._state["a"]["position"]
    assert float(a) != float(b)
