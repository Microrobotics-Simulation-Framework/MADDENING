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


def test_add_node_at_an_established_trap_fails_loudly():
    gm = GraphManager()
    with pytest.raises(AdaptiveNodeBlindnessError, match="Palais fixed point"):
        gm.add_node(PoissonSineTopKNode("adaptive", 1.0, theta=0.5, n=64, k=16))


def test_a_refused_add_node_leaves_the_graph_addable_under_the_same_name():
    """The developer guide's recovery from a trap is to perturb the
    parameters and re-add.  ``add_node`` used to register the node before
    calling ``initial_state()``, so the refusal left a ghost and the name
    was taken for good.  (The graph-level invariant lives in
    ``tests/core/test_graph_mutation_atomicity.py``; this pins the path the
    guide actually documents.)"""
    gm = GraphManager()
    with pytest.raises(AdaptiveNodeBlindnessError):
        gm.add_node(PoissonSineTopKNode("adaptive", 1.0, theta=0.5, n=64, k=16))
    assert list(gm.node_names) == []
    node = PoissonSineTopKNode("adaptive", 1.0, theta=0.5, n=64, k=16,
                               blindness_gate=False)
    _, params = node.cold_start()
    gm.add_node(PoissonSineTopKNode("adaptive", 1.0, theta=float(params["theta"]),
                                    n=64, k=16))
    assert list(gm.node_names) == ["adaptive"]
    gm.compile()
    gm.step()


def test_add_node_at_a_budget_limited_point_succeeds_with_a_warning():
    """A small active-set budget is the whole point of an adaptive solver:
    it must not be a construction-time failure (audit A3)."""
    gm = GraphManager()
    with pytest.warns(UserWarning, match="rules a symmetry trap out"):
        gm.add_node(PoissonSineTopKNode("adaptive", 1.0, theta=0.42, n=64, k=4))
    gm.compile()
    assert int(gm.run_scan(1)["adaptive"]["mask"].sum()) == 4


def test_cold_start_params_can_seed_the_graph():
    node = PoissonSineTopKNode("adaptive", 1.0, theta=0.5, n=64, k=16, blindness_gate=False)
    _, params = node.cold_start()
    gm = GraphManager()
    gm.add_node(node)
    gm.compile()
    gm.params["nodes"]["adaptive"].update(params)
    state = gm.run_scan(1)["adaptive"]
    assert (node.gradient_capture_ratio(state, gm.params["nodes"]["adaptive"])
            > node.gradient_capture_threshold)


def test_the_diagnostic_can_be_run_at_the_live_graph_parameters_at_a_trap():
    """The cold-start check sees the constructor parameters; a graph moved
    onto a trap through ``gm.params`` (what ``sysid.fit`` does) used to run
    silently, and the stale mask even reported a healthy ratio (audit A5)."""
    gm = _graph(theta=0.42)
    node = gm._nodes["adaptive"].node
    live = gm.params["nodes"]["adaptive"]
    assert node.check_gradient_capture(live) > node.gradient_capture_threshold

    live["theta"] = jnp.asarray(0.5)          # the graph now sits on the trap
    gm.run_scan(1)
    assert node.gradient_capture_ratio(gm.get_node_state("adaptive"), live) < 0.01
    with pytest.raises(AdaptiveNodeBlindnessError, match="Palais fixed point"):
        node.check_gradient_capture(live)


def test_reset_state_does_not_re_pay_for_the_diagnostic():
    """``reset_state`` re-runs ``initial_state`` on every node; the
    diagnostic is a function of the parameters, so it is evaluated once."""
    calls = {"n": 0}

    class Counting(PoissonSineTopKNode):
        def compute_full_basis_gradient(self, state, params=None):
            calls["n"] += 1
            return super().compute_full_basis_gradient(state, params)

    gm = GraphManager()
    gm.add_node(Counting("adaptive", 1.0, theta=0.42, n=64, k=16))
    gm.compile()
    before = calls["n"]
    gm.reset_state()
    gm.reset_state()
    assert before == 1 and calls["n"] == 1, calls
