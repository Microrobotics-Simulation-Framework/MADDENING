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
from maddening.core.node import SimulationNode
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


# --------------------------------------------------- atomicity of a restore

class _RelaxWithVectorParam(SimulationNode):
    """``x <- x + rate*dt*(sum(gainvec) - x)``, with a vector parameter.

    ``gainvec``'s *shape* is fixed at construction, so two graphs built
    with different ``gain_len`` disagree on a params leaf while agreeing
    on every node name, field name and state shape -- the redeploy after
    a code change that the cloud resume path exists for.
    """

    def __init__(self, name="relax", n=4, rate=0.5, gain_len=3, timestep=DT):
        super().__init__(name=name, timestep=timestep, rate=float(rate),
                         gainvec=[1.0] * int(gain_len), n=int(n))

    def halo_width(self):
        return {}

    def state_fields(self):
        return ["x"]

    def initial_state(self):
        return {"x": jnp.zeros(int(self.params["n"]), jnp.float32)}

    def boundary_input_spec(self):
        return {}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        gain = jnp.sum(jnp.asarray(p["gainvec"]))
        return {"x": state["x"] + p["rate"] * dt * (gain - state["x"])}


def _relax_graph(gain_len, *, names=("relax",)):
    gm = GraphManager()
    for name in names:
        gm.add_node(_RelaxWithVectorParam(name=name, gain_len=gain_len))
    gm.compile()
    return gm


def test_a_restore_that_fails_on_a_params_leaf_applies_no_node_state(tmp_path):
    """A failed restore leaves the graph exactly as it found it.

    The states used to be applied before the params were validated, so a
    checkpoint the graph *rejected* still ended up in it -- and
    ``cloud.entrypoint`` reported that as a fresh start.
    """
    from maddening.core.simulation.checkpoint import load_state, save_state

    src = _relax_graph(3)
    src.set_node_state("relax", {"x": jnp.asarray([7., 8., 9., 10.], jnp.float32)})
    src.params["nodes"]["relax"]["gainvec"] = jnp.full((3,), 2.0, jnp.float32)
    path = save_state(src, tmp_path / "ck")

    dst = _relax_graph(4)
    before_x = np.asarray(dst.get_node_state("relax")["x"]).copy()
    before_gain = np.asarray(dst.params["nodes"]["relax"]["gainvec"]).copy()

    with pytest.raises(ValueError, match="gainvec"):
        load_state(dst, path)

    np.testing.assert_array_equal(
        np.asarray(dst.get_node_state("relax")["x"]), before_x)
    np.testing.assert_array_equal(
        np.asarray(dst.params["nodes"]["relax"]["gainvec"]), before_gain)
    dst.step()          # and the graph is still runnable, not half-restored


def test_a_restore_that_fails_on_one_node_writes_no_other_nodes_params(tmp_path):
    """Params are staged whole: one bad leaf cancels the others too.

    ``_restore`` wrote each leaf as it validated it, so a graph with two
    nodes could keep the first node's checkpointed parameter and the
    second node's fresh one -- a combination that was never saved.
    """
    from maddening.core.simulation.checkpoint import load_state, save_state

    src = GraphManager()
    src.add_node(_RelaxWithVectorParam(name="a", gain_len=3))
    src.add_node(_RelaxWithVectorParam(name="b", gain_len=3))
    src.compile()
    for name in ("a", "b"):
        src.params["nodes"][name]["gainvec"] = jnp.full((3,), 2.0, jnp.float32)
    path = save_state(src, tmp_path / "ck")

    dst = GraphManager()
    dst.add_node(_RelaxWithVectorParam(name="a", gain_len=3))   # leaf fits
    dst.add_node(_RelaxWithVectorParam(name="b", gain_len=4))   # leaf does not
    dst.compile()

    with pytest.raises(ValueError, match=r"nodes\['b'\]\['gainvec'\]"):
        load_state(dst, path)

    np.testing.assert_array_equal(
        np.asarray(dst.params["nodes"]["a"]["gainvec"]), np.ones(3, np.float32))


def test_a_successful_restore_still_applies_state_meta_and_params(tmp_path):
    """The staging must not cost the restore itself: everything lands."""
    from maddening.core.simulation.checkpoint import load_state, save_state

    src = _relax_graph(3)
    src.run(2)
    src.params["nodes"]["relax"]["gainvec"] = jnp.full((3,), 2.0, jnp.float32)
    src.step()
    path = save_state(src, tmp_path / "ck")

    dst = _relax_graph(3)
    load_state(dst, path)
    np.testing.assert_allclose(
        np.asarray(dst.get_node_state("relax")["x"]),
        np.asarray(src.get_node_state("relax")["x"]))
    np.testing.assert_allclose(
        np.asarray(dst.params["nodes"]["relax"]["gainvec"]),
        np.full(3, 2.0, np.float32))
    # ...and the restored graph carries on exactly where the saved one did.
    dst.step()
    src.step()
    np.testing.assert_allclose(
        np.asarray(dst.get_node_state("relax")["x"]),
        np.asarray(src.get_node_state("relax")["x"]))
