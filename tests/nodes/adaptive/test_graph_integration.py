"""AdaptiveNode inside a GraphManager: params pytree, scan, checkpoints."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.adaptive import AdaptiveNodeBlindnessError

from tests.nodes.adaptive._toys import PoissonSineTopKNode


def _graph(theta=0.42, **kw):
    gm = GraphManager()
    gm.add_node(PoissonSineTopKNode("adaptive", 1.0, theta=theta, n=64, k=16, **kw))
    gm.compile()
    return gm


def test_add_node_compile_and_run_scan():
    gm = _graph()
    state = gm.run_scan(3)["adaptive"]
    assert state["c"].shape == (64,) and state["mask"].dtype == jnp.bool_
    assert int(state["mask"].sum()) == 16
    assert bool(jnp.all(state["c"][~state["mask"]] == 0.0))


def test_step_loop_matches_run_scan():
    a = _graph()
    b = _graph()
    for _ in range(3):
        a.step()
    sa, sb = a.get_node_state("adaptive"), b.run_scan(3)["adaptive"]
    assert jnp.allclose(sa["c"], sb["c"], atol=1e-12)
    assert bool(jnp.array_equal(sa["mask"], sb["mask"]))


def test_params_pytree_exposes_theta_and_drives_the_active_set():
    gm = _graph()
    leaves = gm.params["nodes"]["adaptive"]
    assert set(leaves) == {"theta", "sigma", "sensor_x"}
    assert float(leaves["theta"]) == pytest.approx(0.42)
    before = gm.run_scan(1)["adaptive"]
    gm.params["nodes"]["adaptive"]["theta"] = 0.8
    after = gm.run_scan(1)["adaptive"]
    assert not bool(jnp.array_equal(before["mask"], after["mask"]))
    assert not jnp.allclose(before["c"], after["c"])


def test_param_spec_bounds_and_override_survive_the_graph():
    gm = _graph()
    specs = gm.param_specs()["nodes"]["adaptive"]
    assert specs["theta"].bounds == (0.0, 1.0)
    assert not specs["sigma"].trainable
    gm.set_param_spec("adaptive", "theta", ParamSpec(trainable=False))
    assert not gm.param_specs()["nodes"]["adaptive"]["theta"].trainable


def test_gradient_through_the_compiled_step_reaches_theta():
    gm = _graph()
    node = gm._nodes["adaptive"].node

    def loss(p):
        out = gm._compiled_step(gm._state, gm._default_external_inputs(), p)
        return node.objective(out["adaptive"], {})

    p = jax.tree.map(lambda x: x, gm.params)
    g = jax.grad(loss)(p)["nodes"]["adaptive"]["theta"]
    assert bool(jnp.isfinite(g)) and float(g) != 0.0


def test_checkpoint_round_trip_restores_coefficients_and_mask(tmp_path):
    gm = _graph()
    gm.run_scan(2)
    saved = gm.get_node_state("adaptive")
    gm.save_state(tmp_path / "adaptive.npz")
    gm.params["nodes"]["adaptive"]["theta"] = 0.8
    moved = gm.run_scan(2)["adaptive"]
    assert not bool(jnp.array_equal(moved["mask"], saved["mask"]))

    gm.load_state(tmp_path / "adaptive.npz")
    restored = gm.get_node_state("adaptive")
    assert restored["mask"].dtype == jnp.bool_
    assert bool(jnp.array_equal(restored["mask"], saved["mask"]))
    assert np.array_equal(np.asarray(restored["c"]), np.asarray(saved["c"]))


def test_add_node_at_a_trap_fails_loudly():
    gm = GraphManager()
    with pytest.raises(AdaptiveNodeBlindnessError):
        gm.add_node(PoissonSineTopKNode("adaptive", 1.0, theta=0.5, n=64, k=16))


def test_cold_start_params_can_seed_the_graph():
    node = PoissonSineTopKNode("adaptive", 1.0, theta=0.5, n=64, k=16, blindness_gate=False)
    _, params = node.cold_start()
    gm = GraphManager()
    gm.add_node(node)
    gm.compile()
    gm.params["nodes"]["adaptive"].update(params)
    state = gm.run_scan(1)["adaptive"]
    assert node.blindness_ratio(state, gm.params["nodes"]["adaptive"]) > node.blindness_threshold
