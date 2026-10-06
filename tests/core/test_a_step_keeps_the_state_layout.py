"""A step leaves the graph's state with the layout it had.

A node whose ``update`` returns a leaf of another shape than its
``initial_state()`` built -- a list given for a scalar constant, a vector
delivered into a scalar field -- broadcasts the leaf at its first step.
``GraphManager.step``, ``run`` and ``run_adaptive`` stored the result
without a word, in every release since 0.1.0: a checkpoint saved after the
step did not load after ``reset_state`` or into the graph rebuilt from
``to_dict``.  ``run_scan`` and its siblings have always refused it (a scan
carry of another type).  The step entry points now refuse it by name and
store nothing; the comparison (``_param_probes._state_layout_drift``) is
the one the REST server's ``POST /graph/nodes`` dry run uses.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core import _param_probes
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes import BallNode, SpringDamperNode, TableNode

DT = 0.01


class _Adder(SimulationNode):
    """A scalar state plus whatever arrives on ``drive`` (no declared
    shape, so the edge is not held to one)."""

    def initial_state(self):
        return {"total": jnp.zeros((), jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"total": state["total"] + boundary_inputs.get("drive", 0.0)}


class _Vector(SimulationNode):
    def initial_state(self):
        return {"value": jnp.ones((3,), jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return state


def _broadcasting_ball() -> GraphManager:
    gm = GraphManager()
    gm.add_node(BallNode("b", DT, initial_velocity=[1.0, 2.0]))
    return gm


def _shapes(gm: GraphManager) -> dict:
    return {f"{node}/{key}": np.shape(value) for node, fields in gm._state.items()
            if isinstance(fields, dict) for key, value in fields.items()}


@pytest.mark.parametrize("entry", ["step", "run", "run_adaptive"])
def test_a_step_that_changes_a_leafs_shape_is_refused_by_name_and_stores_nothing(entry):
    gm = _broadcasting_ball()
    shapes = _shapes(gm)
    call = {"step": gm.step, "run": lambda: gm.run(3),
            "run_adaptive": lambda: gm.run_adaptive(5 * DT)}[entry]
    with pytest.raises(ValueError, match=r"'b/position' has shape \(\) before the update "
                                         r"and \(2,\) after it"):
        call()
    assert _shapes(gm) == shapes and float(gm._state["b"]["position"]) == 0.0
    # ... and again at the next call: the refusal is not spent by raising.
    with pytest.raises(ValueError, match="b/position"):
        call()


@pytest.mark.parametrize("entry", ["run_scan", "run_scan_with_history", "run_adaptive_scan"])
def test_the_scan_entry_points_refuse_the_same_graph(entry):
    gm = _broadcasting_ball()
    shapes = _shapes(gm)
    call = {"run_scan": lambda: gm.run_scan(3),
            "run_scan_with_history": lambda: gm.run_scan_with_history(3),
            "run_adaptive_scan": lambda: gm.run_adaptive_scan(5 * DT)}[entry]
    with pytest.raises(TypeError, match="carry"):
        call()
    assert _shapes(gm) == shapes


def test_a_vector_delivered_into_a_scalar_field_is_refused_at_the_step():
    gm = GraphManager()
    gm.add_node(_Vector("v", DT))
    gm.add_node(_Adder("a", DT))
    gm.add_edge("v", "a", "value", "drive")
    with pytest.raises(ValueError, match=r"'a/total' has shape \(\) .* \(3,\)"):
        gm.step()
    assert np.shape(gm._state["a"]["total"]) == ()


def test_the_layout_is_asked_once_per_compile_and_adds_no_trace():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, stiffness=30.0, initial_position=1.0))
    gm.add_node(TableNode("t", DT))
    gm.compile()
    assert gm._layout_check_pending
    gm.step()
    assert not gm._layout_check_pending and gm.trace_count == 1
    gm.step()
    gm.run(2)
    assert gm.trace_count == 1
    # A node added later is asked at the first step of the graph it joins.
    gm.add_node(BallNode("b", DT, initial_velocity=[1.0, 2.0]))
    with pytest.raises(ValueError, match="b/position"):
        gm.step()
    gm.remove_node("b")
    gm.step()
    assert not gm._layout_check_pending


def test_a_refused_step_inside_a_transform_is_refused_too():
    gm = _broadcasting_ball()

    def loss(g):
        return jnp.sum(gm.step(params={"nodes": {"b": {"gravity": g}}})["b"]["position"])

    with pytest.raises(ValueError, match="b/position"):
        jax.grad(loss)(jnp.float32(-9.81))


def test_the_comparison_names_every_kind_of_difference():
    before = {"n": {"a": jnp.zeros(()), "b": jnp.zeros((2,)), "c": jnp.zeros((), jnp.int32)}}
    assert _param_probes._state_layout_drift(before, before) == []
    after = {"n": {"a": jnp.zeros((2,)), "c": jnp.zeros(()), "d": jnp.zeros(())}}
    found = _param_probes._state_layout_drift(before, after)
    assert len(found) == 4
    assert any("'n/a' has shape () before the update and (2,) after it" in f for f in found)
    assert any("'n/b' is missing" in f for f in found)
    assert any("'n/c' has dtype int32" in f and "float32" in f for f in found)
    assert any("'n/d' appears only after" in f for f in found)
    # Shapes and names only, where the caller asks for that ...
    assert len(_param_probes._state_layout_drift(before, after, dtypes=False)) == 3
    # ... and a dtype's width is never a difference: with x64 on, every
    # stock node's update returns float64 for a float32 state.
    wide = {"n": {"a": np.zeros((), np.float16), "b": np.zeros((2,), np.float32),
                  "c": np.zeros((), np.int8)}}
    assert _param_probes._state_layout_drift(before, wide) == []
    # The abstract values of a trace compare like arrays.
    abstract = jax.eval_shape(lambda: before)
    assert _param_probes._state_layout_drift(before, abstract) == []
