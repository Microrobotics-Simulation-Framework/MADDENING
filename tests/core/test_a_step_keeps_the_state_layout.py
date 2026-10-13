"""A step leaves the graph's state with the layout it had.

A node whose ``update`` returns a leaf of another shape than its
``initial_state()`` built -- a list given for a scalar constant, a vector
delivered into a scalar field -- broadcasts the leaf at its first step.
``run_scan`` and its siblings refuse such a graph (a scan carry of another
type).  ``GraphManager.step``, ``run`` and ``run_adaptive`` stored the
result without a word from 0.1.0 (MADD-ANO-220): a checkpoint saved after
the step did not load after ``reset_state``.  They now refuse it, by the
name of the node and the leaf, and store nothing: the stepped state is
compared with the one it replaces once per trace of the step, host-side
(``GraphManager._store_stepped_state``).  The REST server refuses such a
node where it is added (``POST /graph/nodes``), with the same comparison
(``_param_probes._state_layout_drift``).
"""
import warnings

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


class _Recorder:
    """An observer and a run callback: what a refused step must not reach."""

    def __init__(self, gm: GraphManager):
        self.events, self.calls = [], 0
        gm.add_observer(lambda event, data: self.events.append(event))

    def callback(self, *args):
        self.calls += 1


def _entry_points(gm: GraphManager, seen: _Recorder, **kw) -> dict:
    return {"step": lambda: gm.step(**kw),
            "run": lambda: gm.run(3, callback=seen.callback, **kw),
            "run_adaptive": lambda: gm.run_adaptive(5 * DT, callback=seen.callback, **kw)}


ENTRIES = ["step", "run", "run_adaptive"]


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_step_entry_points_keep_the_shape_of_every_state_leaf(entry):
    """MADD-ANO-220: the stepped state was stored.  Refused, by the name of
    the node and the leaf, and the graph is as it was: the very state
    object (its clock and step count are in it), no observer told of a
    step, no callback called."""
    gm = _broadcasting_ball()
    gm.compile()
    held, shapes = gm._state, _shapes(gm)
    seen = _Recorder(gm)
    with pytest.raises(ValueError, match=r"'b/position' has shape \(\) before the update "
                                         r"and \(2,\) after it") as refusal:
        _entry_points(gm, seen)[entry]()
    assert "Nothing was stored" in str(refusal.value)
    assert gm._state is held and _shapes(gm) == shapes
    assert "step" not in seen.events and seen.calls == 0
    # ... and it is refused again: a refusal does not use the comparison up.
    with pytest.raises(ValueError, match="changes the layout"):
        _entry_points(gm, seen)[entry]()
    assert gm._state is held and "step" not in seen.events and seen.calls == 0


class _Halves(SimulationNode):
    """An integer count its update returns as a float."""

    def initial_state(self):
        return {"n": jnp.zeros((), jnp.int32), "x": jnp.zeros((), jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"n": state["n"] + 0.5, "x": state["x"] + dt}


class _Widens(SimulationNode):
    """An ``int16`` count its update returns as ``int32``: another width of
    the same kind of dtype."""

    def initial_state(self):
        return {"n": jnp.zeros((), jnp.int16)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"n": state["n"].astype(jnp.int32) + 1}


class _DropsAField(SimulationNode):
    def initial_state(self):
        return {"x": jnp.zeros((), jnp.float32), "y": jnp.zeros((), jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": state["x"] + dt}


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("node, needle", [
    (_Halves, "'h/n' has dtype int32 before the update and float32 after it"),
    (_DropsAField, "'h/y' is missing after the update"),
], ids=["kind-of-dtype", "missing-field"])
def test_a_step_that_changes_a_leafs_kind_of_dtype_or_the_fields_is_refused(entry, node, needle):
    """The same loss by another route: the scans refuse a carry of another
    dtype, and a checkpoint of the float count does not load into the
    integer state of the graph after a reset."""
    gm = GraphManager()
    gm.add_node(node("h", DT))
    gm.compile()
    held, seen = gm._state, _Recorder(gm)
    with pytest.raises(ValueError) as refusal:
        _entry_points(gm, seen)[entry]()
    assert needle in str(refusal.value)
    assert gm._state is held and "step" not in seen.events and seen.calls == 0


def test_a_dtype_of_another_width_is_not_what_the_comparison_refuses():
    """The rule's edge, stated: only the kind of dtype is compared.  With
    x64 enabled a stock node returns ``float64`` for the ``float32`` its
    ``initial_state()`` builds, and a comparison of widths would refuse
    every one of them."""
    gm = GraphManager()
    gm.add_node(_Widens("w", DT))
    gm.step()
    gm.step()
    assert gm._state["w"]["n"].dtype == jnp.int32 and int(gm._state["w"]["n"]) == 2


class _Adds(SimulationNode):
    """Adds whatever its ``drive`` input delivers to a scalar."""

    def initial_state(self):
        return {"total": jnp.zeros((), jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"total": state["total"] + boundary_inputs["drive"]}

    def boundary_input_spec(self):
        return {"drive": BoundaryInputSpec(shape=(), description="what it adds")}


def _adder() -> GraphManager:
    gm = GraphManager()
    gm.add_node(_Adds("a", DT))
    gm.add_external_input("a", "drive")
    return gm


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_later_step_that_is_traced_again_is_compared_again(entry):
    """The comparison is once per trace of the step, not once per graph:
    an external input of another shape retraces the step, and the leaf it
    broadcasts is refused as it is at a first step.  The graph is where
    the last good step left it, and goes on from there to the bits."""
    one, vector = {"a": {"drive": 1.0}}, {"a": {"drive": jnp.ones(3, jnp.float32)}}
    gm = _adder()
    gm.step(external_inputs=one)
    gm.step(external_inputs=one)
    assert gm.trace_count == 1
    held, seen = gm._state, _Recorder(gm)
    with pytest.raises(ValueError, match=r"'a/total' has shape \(\) before the update "
                                         r"and \(3,\) after it"):
        _entry_points(gm, seen, external_inputs=vector)[entry]()
    assert gm._state is held and "step" not in seen.events and seen.calls == 0
    gm.step(external_inputs=one)
    reference = _adder()
    for _ in range(3):
        reference.step(external_inputs=one)
    assert np.asarray(gm._state["a"]["total"]).tobytes() == \
        np.asarray(reference._state["a"]["total"]).tobytes()
    assert np.shape(gm._state["a"]["total"]) == ()


def test_the_step_of_a_graph_compiled_again_is_compared_again():
    """Each compile is a new program.  A graph that has stepped is given a
    node that broadcasts: its next step is the first of the new compile."""
    gm = GraphManager()
    gm.add_node(BallNode("ok", DT))
    for _ in range(3):
        gm.step()
    gm.add_node(BallNode("b", DT, initial_velocity=[1.0, 2.0]))
    gm.compile()
    assert gm.trace_count == 0
    held = gm._state
    with pytest.raises(ValueError, match="'b/position' has shape"):
        gm.step()
    assert gm._state is held
    # The user's way on: the node replaced by one of the right rank.
    gm.remove_node("b")
    gm.add_node(BallNode("b", DT, initial_velocity=1.0))
    gm.step()
    assert np.shape(gm._state["b"]["position"]) == ()


class _CountsItsTraces(SimulationNode):
    traces = 0

    def initial_state(self):
        return {"x": jnp.zeros((2,), jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        type(self).traces += 1
        return {"x": state["x"] + dt}


def test_the_comparison_is_made_once_per_trace_and_calls_no_update(monkeypatch):
    """It reads the shapes of the state the step returned, host-side: the
    node's ``update`` runs as often as the step's one trace runs it, and
    nine steps of one program are compared once.  (The patch is of the
    name the graph reads: the count below shows it is in effect.)"""
    compared = []
    real = _param_probes._state_layout_drift
    monkeypatch.setattr(_param_probes, "_state_layout_drift",
                        lambda *a, **kw: compared.append(1) or real(*a, **kw))
    gm = GraphManager()
    gm.add_node(_CountsItsTraces("c", DT))
    gm.compile()
    _CountsItsTraces.traces = 0
    for _ in range(5):
        gm.step()
    gm.run(4)
    assert gm.trace_count == 1 and _CountsItsTraces.traces == 1
    assert len(compared) == 1
    gm.compile()
    gm.step()
    gm.step()
    assert len(compared) == 2


def test_a_graph_with_a_coupling_group_is_held_to_its_layout_too():
    """The group's own keys (its iteration count, its residual) are leaves
    of the stepped state like any other, and a healthy group steps; a
    node beside the group that broadcasts is refused, and the group's
    bookkeeping is where it was."""
    def build(velocity):
        gm = GraphManager()
        gm.add_node(SpringDamperNode("p", DT, stiffness=30.0, damping=2.0, initial_position=1.0))
        gm.add_node(SpringDamperNode("q", DT, stiffness=20.0, damping=2.0, initial_position=0.5))
        gm.add_edge("p", "q", "position", "anchor_position")
        gm.add_edge("q", "p", "position", "anchor_position")
        gm.add_coupling_group(["p", "q"], max_iterations=8, tolerance=1e-6)
        gm.add_node(BallNode("b", DT, initial_velocity=velocity))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.compile()
        return gm

    healthy = build(1.0)
    before = _param_probes._state_layout_drift(healthy._state, healthy._state)
    layout = jax.tree.map(lambda leaf: (np.shape(leaf), np.asarray(leaf).dtype.kind),
                          healthy._state)
    for _ in range(3):
        healthy.step()
    assert before == [] and layout == jax.tree.map(
        lambda leaf: (np.shape(leaf), np.asarray(leaf).dtype.kind), healthy._state)

    gm = build([1.0, 2.0])
    held = gm._state
    for call in (gm.step, lambda: gm.run(2), lambda: gm.run_adaptive(5 * DT)):
        with pytest.raises(ValueError, match="'b/position' has shape"):
            call()
        assert gm._state is held


def test_a_step_inside_a_transform_is_compared_on_the_traced_state():
    """Under ``jax.grad`` the stepped state is a tree of tracers, whose
    shapes are as real as an array's: a healthy graph differentiates, and
    one that broadcasts is refused inside the trace as outside it."""
    def loss(gm):
        def f(v):
            params = gm.params
            params["nodes"]["b"]["gravity"] = v
            gm.step(params=params)
            return jnp.sum(gm.step(params=params)["b"]["position"])
        return f

    healthy = GraphManager()
    healthy.add_node(BallNode("b", DT, initial_position=5.0))
    healthy.compile()
    assert np.isfinite(float(jax.grad(loss(healthy))(jnp.float32(-9.81))))
    with pytest.raises(ValueError, match="'b/position' has shape"):
        broadcasting = _broadcasting_ball()
        broadcasting.compile()
        jax.grad(loss(broadcasting))(jnp.float32(-9.81))


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


class _Draws(SimulationNode):
    """Carries a PRNG key in its state, as a stochastic node does."""

    def initial_state(self):
        return {"key": jax.random.key(0), "x": jnp.zeros((), jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        key, sub = jax.random.split(state["key"])
        return {"key": key, "x": state["x"] + dt * jax.random.normal(sub)}


class _SpendsItsKey(_Draws):
    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"key": jax.random.key_data(state["key"]), "x": state["x"]}


def test_a_prng_key_in_the_state_is_a_leaf_of_a_kind_of_its_own():
    """A key's dtype (``key<fry>``) is not a NumPy dtype and has no kind
    letter: it is compared as itself.  The comparison used to raise on it
    (``canonicalize_dtype called on extended dtype``), which refused every
    step of a graph that carries a key."""
    gm = GraphManager()
    gm.add_node(_Draws("d", DT))
    gm.step()
    gm.run(2)
    assert gm._state["d"]["key"].dtype == jax.random.key(0).dtype
    assert _drift(_Draws("d", DT)) == []
    key, other = jax.random.key(0), jax.random.key(0, impl="rbg")
    assert _param_probes._state_layout_drift({"k": key}, {"k": jax.random.split(key)[0]}) == []
    for after in (other, jnp.zeros((), jnp.uint32)):
        (found,) = _param_probes._state_layout_drift({"k": key}, {"k": after})
        assert "'k' has dtype key<fry> before the update" in found
    assert _param_probes._state_layout_drift({"k": key}, {"k": other}, dtypes=False) == []
    # ... and a node that returns the key's bits for the key is refused.
    spent = GraphManager()
    spent.add_node(_SpendsItsKey("d", DT))
    spent.compile()
    held = spent._state
    with pytest.raises(ValueError, match="'d/key' has shape"):
        spent.step()
    assert spent._state is held
