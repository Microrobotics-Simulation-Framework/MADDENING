"""The compiled step must trace once per run, not once per weak-typed leaf.

A node that seeds its state with ``jnp.array(0.0)`` (weak-typed) used to
force a retrace on the second step (the leaf comes back strongly typed)
and again on the third for leaves that only change later — three
compiles of the same function on every run.  ``compile`` and
``set_node_state`` now normalise the state's weak types.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.spring import SpringDamperNode


class WeakSeed(SimulationNode):
    """Weak-typed seeds, one of which only changes from the 2nd step on."""
    def initial_state(self):
        return {"x": jnp.array(1.0), "y": jnp.array(0.0), "n": jnp.array(0)}

    def update(self, s, bi, dt, *, params=None):
        x = s["x"] * (1.0 - dt)
        # ``y`` stays a pass-through on the first step (still weak if not
        # normalised), and becomes arithmetic output only afterwards.
        y = jnp.where(s["n"] > 0, s["y"] + x * dt, s["y"])
        return {"x": x, "y": y, "n": s["n"] + 1}


def _cache_size(gm):
    return gm._compiled_step._cache_size()


def test_weak_typed_seed_state_traces_once():
    gm = GraphManager()
    gm.add_node(WeakSeed("w", 0.01))
    gm.compile()
    assert all(not l.weak_type for l in jax.tree.leaves(gm._state))
    for _ in range(4):
        gm.step()
    assert _cache_size(gm) == 1
    assert float(gm._state["w"]["n"]) == 4


def test_coupled_group_meta_residual_is_strong_and_traces_once():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.01, initial_position=3.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=10, tolerance=1e-8)
    gm.compile()
    meta = gm._state["_meta"]
    assert not meta["coupling_a+b_residual"].weak_type
    for _ in range(3):
        gm.step()
    assert _cache_size(gm) == 1


def test_set_node_state_normalises_weak_types():
    gm = GraphManager()
    gm.add_node(WeakSeed("w", 0.01))
    gm.compile()
    gm.step()
    gm.set_node_state("w", {"x": jnp.array(2.0), "y": jnp.array(0.0), "n": jnp.array(0)})
    assert all(not l.weak_type for l in jax.tree.leaves(gm._state["w"]))
    for _ in range(3):
        gm.step()
    assert _cache_size(gm) == 1


def test_default_external_inputs_cached_but_dicts_fresh():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01))
    gm.add_external_input("s", "anchor_position", shape=())
    gm.compile()
    a = gm._default_external_inputs()
    b = gm._default_external_inputs()
    assert a is not b and a["s"] is not b["s"]              # callers may mutate
    assert a["s"]["anchor_position"] is b["s"]["anchor_position"]   # zeros shared
    assert float(a["s"]["anchor_position"]) == 0.0
    a["s"]["anchor_position"] = jnp.asarray(5.0)
    assert float(gm._default_external_inputs()["s"]["anchor_position"]) == 0.0
    # Adding an input after compile still yields a correct dict.
    gm.add_external_input("s", "extra", shape=(3,))
    gm.compile()
    c = gm._default_external_inputs()
    assert c["s"]["extra"].shape == (3,)


def test_results_unchanged_by_normalisation():
    """Strong typing is a trace-signature change only."""
    def run(normalise):
        gm = GraphManager()
        gm.add_node(WeakSeed("w", 0.01))
        gm.compile()
        if not normalise:
            gm._state["w"] = {"x": jnp.array(1.0), "y": jnp.array(0.0), "n": jnp.array(0)}
        for _ in range(5):
            gm.step()
        return np.asarray(gm._state["w"]["y"])
    np.testing.assert_array_equal(run(True), run(False))


def test_reset_state_keeps_the_compiled_step_and_meta():
    """A reset must not reintroduce weak types (retrace) or break _meta."""
    gm = GraphManager()
    gm.add_node(WeakSeed("w", 0.01))
    gm.add_node(SpringDamperNode("a", 0.01, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.02, initial_position=3.0))     # multi-rate
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.compile()
    for _ in range(3):
        gm.step()
    assert int(gm._state["_meta"]["step_count"]) == 3
    gm.reset_state()
    assert int(gm._state["_meta"]["step_count"]) == 0
    assert all(not l.weak_type for l in jax.tree.leaves(gm._state))
    np.testing.assert_array_equal(np.asarray(gm._state["w"]["n"]), 0)
    for _ in range(3):
        gm.step()
    assert _cache_size(gm) == 1
    assert int(gm._state["w"]["n"]) == 3


def test_reset_state_zeroes_coupling_diagnostics_and_warm_start():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.01, initial_position=3.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=10, tolerance=1e-8,
                          acceleration="iqn-imvj", jacobian_reuse=2)
    gm.compile()
    for _ in range(4):
        gm.step()
    meta = gm._state["_meta"]
    assert any(float(jnp.max(jnp.abs(meta[k]))) > 0 for k in meta if k.endswith("_V"))
    gm.reset_state()
    meta = gm._state["_meta"]
    assert all(float(jnp.max(jnp.abs(meta[k]))) == 0 for k in meta if k.endswith("_V"))
    assert gm.coupling_diagnostics()["a+b"]["iterations"] == 0
    for _ in range(3):
        gm.step()
    assert _cache_size(gm) == 1 and gm.coupling_diagnostics()["a+b"]["converged"]
