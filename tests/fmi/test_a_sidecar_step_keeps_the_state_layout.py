"""The FMU sidecar and bridge hold a step to the layout of the state it
was given, as ``GraphManager.step`` does.

The sidecar runs a graph's compiled step on a state of its own, so the
graph's comparison never sees it.  A node that broadcasts a state leaf at
its first step (``BallNode(initial_velocity=[1.0, 2.0])``: the scalar
``position`` becomes ``(2,)``) left the sidecar holding a state that was
not the one its model description declares (MADD-ANO-220).  The step is
refused by the name of the node and the leaf, and nothing is committed.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.fmi import sidecar as sidecar_module
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge
from maddening.nodes import BallNode, SpringDamperNode

DT = 0.01
NEEDLE = r"'b/position' has shape \(\) before the update and \(2,\) after it"


def _graph(velocity) -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.add_node(BallNode("b", DT, initial_velocity=velocity))
    gm.compile()
    return gm


def _sidecar(gm: GraphManager, md) -> FmuSidecar:
    return FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
        input_resolver=gm._resolve_external_inputs))


def test_a_sidecar_step_that_reshapes_a_leaf_is_refused_and_nothing_is_committed():
    gm = _graph([1.0, 2.0])
    sidecar = _sidecar(gm, build_model_description(gm, model_name="m"))
    held = sidecar._state
    for _ in range(2):          # a refusal does not use the comparison up
        with pytest.raises(ValueError, match=NEEDLE) as refusal:
            sidecar.step(None)
        assert "Nothing was advanced" in str(refusal.value)
        assert sidecar._state is held
        assert np.shape(sidecar.state["b"]["position"]) == ()


def test_a_bridge_step_that_reshapes_a_leaf_is_an_error_reply_and_the_clock_stays():
    gm = _graph([1.0, 2.0])
    md = build_model_description(gm, model_name="m")
    bridge = FmuTcpBridge(_sidecar(gm, md), md, master_dt=DT)
    held = bridge._sidecar._state
    reply = bridge.handle({"op": "step", "t": 0.0, "dt": md.default_step_size})
    assert reply["ok"] is False and "'b/position' has shape ()" in reply["error"], reply
    assert bridge._sidecar._state is held and bridge._time == 0.0


def test_a_graphs_step_is_compared_once_per_trace(monkeypatch):
    """Not at every step: the layout a traced program returns is fixed by
    the layout it is given.  The patch is of the name the sidecar reads."""
    calls = []
    real = sidecar_module._state_layout_drift
    monkeypatch.setattr(sidecar_module, "_state_layout_drift",
                        lambda *a, **kw: calls.append(1) or real(*a, **kw))
    gm = _graph(1.0)
    sidecar = _sidecar(gm, build_model_description(gm, model_name="m"))
    for _ in range(5):
        sidecar.step(None)
    assert len(calls) == 1 and gm.trace_count == 1
    # The sidecar's five steps are the graph's five, to the bits.
    reference = _graph(1.0)
    reference.run(5)
    for node in ("s", "b"):
        assert np.asarray(sidecar.state[node]["position"]).tobytes() == \
            np.asarray(reference._state[node]["position"]).tobytes()


def test_a_step_that_is_not_a_graphs_is_compared_at_every_step(monkeypatch):
    """Nothing says when a wrapper or a double is a new program, so it is
    never assumed to be the old one: a leaf that grows at the third step
    is refused at the third step."""
    calls = []
    real = sidecar_module._state_layout_drift
    monkeypatch.setattr(sidecar_module, "_state_layout_drift",
                        lambda *a, **kw: calls.append(1) or real(*a, **kw))
    taken = []

    def step_fn(state, external_inputs):
        taken.append(1)
        x = state["n"]["x"] + 1.0
        return {"n": {"x": jnp.broadcast_to(x, (2,)) if len(taken) == 3 else x}}

    sidecar = FmuSidecar(SidecarConfig(
        schema_token="t", step_fn=step_fn,
        initial_state={"n": {"x": jnp.zeros((), jnp.float32)}}))
    sidecar.step(None)
    sidecar.step(None)
    held = sidecar._state
    with pytest.raises(ValueError, match=r"'n/x' has shape \(\) before the update"):
        sidecar.step(None)
    assert sidecar._state is held and float(held["n"]["x"]) == 2.0 and len(calls) == 3


def test_the_step_of_a_graph_compiled_again_since_is_compared_at_every_step(monkeypatch):
    """Its trace count is the new compile's, not this step's."""
    gm = _graph(1.0)
    sidecar = _sidecar(gm, build_model_description(gm, model_name="m"))
    sidecar.step(None)
    calls = []
    real = sidecar_module._state_layout_drift
    monkeypatch.setattr(sidecar_module, "_state_layout_drift",
                        lambda *a, **kw: calls.append(1) or real(*a, **kw))
    sidecar.step(None)
    assert calls == []
    gm.compile()
    sidecar._advanced(sidecar._state, None)
    sidecar._advanced(sidecar._state, None)
    assert len(calls) == 2
