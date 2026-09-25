"""A node whose state has no fields survives an FMU-state round trip.

The bridge rebuilt the restored state from the archive's members, and a
node with no state fields has none -- so ``get_state`` then ``set_state``
of the bridge's *own* snapshot answered ``ok`` and dropped the node, and
every step after it failed with ``KeyError`` for that node.  Both restore
paths now rebuild on the live model's skeleton.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.fmi import build_model_description
from maddening.fmi.fmu_state import serialize_fmu_state
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge
from maddening.nodes.spring import SpringDamperNode

DT = 1e-2


class _NoState(SimulationNode):
    """A node that keeps no state between steps."""

    def initial_state(self):
        return {}

    def update(self, state, boundary_inputs, dt, **kw):
        return {}


@pytest.fixture(scope="module")
def graph():
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="spring", timestep=DT, stiffness=30.0))
    gm.add_node(_NoState(name="probe", timestep=DT))
    gm.add_external_input("spring", "anchor_position")
    gm.compile()
    assert gm._state["probe"] == {}
    return gm


def _sidecar(gm, md):
    return FmuSidecar(SidecarConfig(schema_token=md.instantiation_token,
                                    step_fn=gm._compiled_step, initial_state=gm._state,
                                    params=gm.params, param_specs=gm.param_specs()))


def test_the_bridge_restores_its_own_snapshot_of_a_node_without_fields(graph):
    md = build_model_description(graph, model_name="m")
    bridge = FmuTcpBridge(_sidecar(graph, md), md, master_dt=DT)
    try:
        snap = bridge.handle({"op": "get_state"})["state"]
        assert bridge.handle({"op": "set_state", "state": snap}) == {"ok": True}
        assert bridge._sidecar.state["probe"] == {}
        reply = bridge.handle({"op": "step", "dt": DT})
        assert reply == {"ok": True, "t": pytest.approx(DT)}, reply
    finally:
        bridge.stop()


def test_the_sidecar_restores_a_snapshot_of_a_node_without_fields(graph):
    md = build_model_description(graph, model_name="m")
    sc = _sidecar(graph, md)
    sc.set_fmu_state(sc.get_fmu_state())
    assert sc.state["probe"] == {}
    sc.step(graph._default_external_inputs())


def test_a_snapshot_that_omits_a_node_without_fields_still_restores_it(graph):
    """No key names a node with no fields, so the key-set check cannot see
    it missing; the restore keeps the live node instead of dropping it."""
    md = build_model_description(graph, model_name="m")
    sc = _sidecar(graph, md)
    state = {"spring": {f: np.asarray(v) for f, v in sc.state["spring"].items()}}
    sc.set_fmu_state(serialize_fmu_state(state=state, schema_token=md.instantiation_token,
                                         params=sc.params))
    assert sc.state["probe"] == {}
    sc.step(graph._default_external_inputs())
