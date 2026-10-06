"""A step should leave the graph's state with the layout it had.

A node whose ``update`` returns a leaf of another shape than its
``initial_state()`` built -- a list given for a scalar constant, a vector
delivered into a scalar field -- broadcasts the leaf at its first step.
``run_scan`` and its siblings refuse such a graph (a scan carry of another
type).  ``GraphManager.step``, ``run`` and ``run_adaptive`` store the
result without a word, as they have since 0.1.0 (MADD-ANO-218, open): a
checkpoint saved after the step does not load after ``reset_state``.  The
REST server refuses such a node where it is added (``POST /graph/nodes``),
with the comparison tested here (``_param_probes._state_layout_drift``).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core import _param_probes
from maddening.core._graph_specs import _NodeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.nodes import BallNode, HeartPumpNode, SpringDamperNode

DT = 0.01


def _broadcasting_ball() -> GraphManager:
    gm = GraphManager()
    gm.add_node(BallNode("b", DT, initial_velocity=[1.0, 2.0]))
    return gm


def _shapes(gm: GraphManager) -> dict:
    return {f"{node}/{key}": np.shape(value) for node, fields in gm._state.items()
            if isinstance(fields, dict) for key, value in fields.items()}


@pytest.mark.parametrize("entry", ["run_scan", "run_scan_with_history", "run_adaptive_scan"])
def test_the_scan_entry_points_refuse_a_step_that_changes_a_leafs_shape(entry):
    gm = _broadcasting_ball()
    shapes = _shapes(gm)
    call = {"run_scan": lambda: gm.run_scan(3),
            "run_scan_with_history": lambda: gm.run_scan_with_history(3),
            "run_adaptive_scan": lambda: gm.run_adaptive_scan(5 * DT)}[entry]
    with pytest.raises(TypeError, match="carry"):
        call()
    assert _shapes(gm) == shapes


@pytest.mark.xfail(strict=True, reason=(
    "MADD-ANO-218 (open): step, run and run_adaptive store a stepped state whose leaf "
    "has another shape than the state it replaces; the state no longer has the layout "
    "of initial_state(), and its checkpoint does not load after a reset"))
@pytest.mark.parametrize("entry", ["step", "run", "run_adaptive"])
def test_the_step_entry_points_keep_the_shape_of_every_state_leaf(entry):
    gm = _broadcasting_ball()
    shapes = _shapes(gm)
    try:
        {"step": gm.step, "run": lambda: gm.run(3),
         "run_adaptive": lambda: gm.run_adaptive(5 * DT)}[entry]()
    except (ValueError, TypeError):
        pass                    # a refusal that stores nothing would do
    assert _shapes(gm) == shapes


def _drift(node: SimulationNode, boundary_inputs=None, **kw) -> list:
    spec = _NodeSpec(node=node, update_fn=node.update, timestep=node.delta_t,
                     accepts_params=True)
    return _param_probes._node_update_layout_drift(
        spec, node.initial_state(), node.params_pytree(), boundary_inputs, **kw)


def test_one_update_traced_as_the_graph_calls_it_is_compared_with_the_initial_state():
    assert _drift(BallNode("b", DT)) == []
    assert _drift(SpringDamperNode("s", DT, stiffness=3.0)) == []
    (found,) = _drift(BallNode("b", DT, initial_velocity=[1.0, 2.0]))
    assert found == "'position' has shape () before the update and (2,) after it"
    # With no boundary input, as the graph steps a node that has no edge:
    # the pump then reads the constant an input would have replaced.
    pump = HeartPumpNode("h", DT, venous_pressure=[0.0, 0.0])
    assert _drift(pump) == []
    assert any("'arterial_pressure' has shape ()" in f for f in _drift(pump, {}))
    # Nothing is computed: the trace is abstract, so no value is read.
    assert _drift(BallNode("b", DT, initial_position=float("nan"))) == []


class _Adder(SimulationNode):
    def initial_state(self):
        return {"total": jnp.zeros((), jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"total": state["total"] + boundary_inputs["drive"]}

    def boundary_input_spec(self):
        return {"drive": BoundaryInputSpec(shape=(3,), description="a vector")}


def test_the_declared_boundary_inputs_are_delivered_at_their_declared_shapes():
    """A vector delivered into a scalar field is found from the node's own
    declaration, and a trace failure is the caller's to judge."""
    (found,) = _drift(_Adder("a", DT))
    assert "'total' has shape () before the update and (3,) after it" == found
    with pytest.raises(KeyError):
        _drift(_Adder("a", DT), {})


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
