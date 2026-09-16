"""Shape and namespace guards around persistence and the params pytree.

* ``load_state`` refuses a state field of the wrong shape and coerces
  dtype; a checkpoint without ``_meta`` keeps the freshly compiled
  ``_meta`` (a multirate graph no longer raises ``KeyError``);
* a wrong-shape params leaf is refused by ``step(params=)`` and by
  ``gm.params[...] =`` (it used to broadcast the node's state for good);
* node names that would corrupt checkpoint keys, mapping slots or edge
  keys are refused at ``add_node``.

Originally written from the independent audit of 2026-09-16 (round 4; report and
reproducers under ``benchmarks/results/audit4/``).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

DT = 0.01


def _spring():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.compile()
    return gm




# ------------------------------------------------------------ checkpoints

def test_load_state_refuses_wrong_shape_state_and_coerces_dtype(tmp_path):
    from maddening.core.simulation.checkpoint import load_state, save_state

    gm = _spring()
    path = save_state(gm, tmp_path / "ck")
    data = dict(np.load(path))
    bad = dict(data)
    bad["s/position"] = np.ones(3, np.float32)
    np.savez(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError, match="shape"):
        load_state(_spring(), tmp_path / "bad.npz")
    odd = dict(data)
    odd["s/position"] = np.array(7, np.int64)
    np.savez(tmp_path / "odd.npz", **odd)
    fresh = _spring()
    load_state(fresh, tmp_path / "odd.npz")
    assert fresh._state["s"]["position"].dtype == jnp.float32
    fresh.step()
    fresh.step()
    assert fresh.trace_count == 1


def test_checkpoint_without_meta_into_multirate_graph_keeps_compiled_meta(tmp_path):
    from maddening.core.simulation.checkpoint import load_state, save_state

    def build(slow_dt):
        gm = GraphManager()
        gm.add_node(SpringDamperNode("s", DT, initial_position=1.0))
        gm.add_node(SpringDamperNode("b", slow_dt, initial_position=0.5))
        gm.add_edge("s", "b", "position", "anchor_position")
        gm.compile()
        return gm

    single = build(DT)
    single.run(2)
    path = save_state(single, tmp_path / "ck")
    assert not any(k.startswith("_meta/") for k in np.load(path).files)
    multi = build(2 * DT)
    load_state(multi, path)
    multi.run(3)                                    # used to raise KeyError: '_meta'
    assert int(multi._state["_meta"]["step_count"]) == 3


# ------------------------------------------------------------- params / names

def test_wrong_shape_params_leaf_is_refused_everywhere():
    gm = _spring()
    with pytest.raises(ValueError, match="shape"):
        gm.step(params={"nodes": {"s": {"stiffness": jnp.ones(3, jnp.float32) * 30}}})
    gm.params["nodes"]["s"]["stiffness"] = jnp.ones(3, jnp.float32) * 30
    with pytest.raises(ValueError, match="shape"):
        gm.step()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(30.0, jnp.float32)
    gm.step()
    assert gm._state["s"]["position"].shape == ()


@pytest.mark.parametrize("name", ["a/b", "a#1", "a->b", ""])
def test_node_names_that_break_key_namespaces_are_refused(name):
    gm = GraphManager()
    with pytest.raises(ValueError, match="invalid"):
        gm.add_node(SpringDamperNode(name, DT))
