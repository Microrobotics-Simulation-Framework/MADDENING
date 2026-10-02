"""A node whose name holds ``.params.`` has parameters an importer can set.

``add_node`` allows a node named ``rig.params.v2``; its parameters are FMI
variables named ``rig.params.v2.params.elasticity``.  The bridge and the
sidecar split that name at the *first* ``.params.``, found no node ``rig``,
and so every set of such a parameter was refused as unknown, and the bridge
silently left out the bounds and start value its description advertises.
The variable now carries its ``(node, key)`` (``FMIVariable.node`` /
``field``), and the sidecar resolves a parameter name by exact match.

Why not refuse such a name at ``add_node`` instead: the name is legal and
works everywhere else -- the REST API, USD, checkpoints, the graph itself --
and only the FMU export spelled it ambiguously.  Refusing it would break
graphs that never export an FMU, to fix a defect of the export.  The one
truly ambiguous case, two parameters spelling one name (node ``a.params.b``
with key ``c`` beside node ``a`` with key ``b.params.c``), is refused by
name where it would matter: ``build_model_description``'s uniqueness
check, and the sidecar's name lookup.
"""

from __future__ import annotations

import base64
import io

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, state_of, values_of
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode

NODE = "rig.params.v2"
ELASTICITY = f"{NODE}.params.elasticity"


def _graph(elasticity=0.7):
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01))
    gm.add_node(BallNode(name=NODE, timestep=0.01, initial_position=1.0, elasticity=elasticity))
    gm.add_edge("table", NODE, "position", "table_position")
    gm.compile()
    return gm


@pytest.fixture(scope="module")
def gm():
    return _graph()


@pytest.fixture
def served(gm):
    md = build_model_description(gm, model_name="P")
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params))      # no param_specs: the bridge's own
    bridge = FmuTcpBridge(sidecar, md, master_dt=0.01)
    yield md, sidecar, bridge
    bridge.stop()


def _vr(md, name):
    return next(v.value_reference for v in md.variables if v.name == name)


def test_the_variable_carries_its_node_and_key(served):
    md, _, _ = served
    var = next(v for v in md.variables if v.name == ELASTICITY)
    assert (var.node, var.field) == (NODE, "elasticity")


def test_its_parameter_is_set_and_read_back(served):
    md, sidecar, bridge = served
    vr = _vr(md, ELASTICITY)
    assert bridge.handle({"op": "set", "vr": [vr], "values": [0.5]}) == {"ok": True}
    assert values_of(bridge.handle({"op": "get", "vr": [vr]})).tolist() == [0.5]
    sidecar.set_params({ELASTICITY: 0.25})
    assert float(sidecar.get_params()[ELASTICITY]) == 0.25


def test_its_advertised_bounds_hold(served):
    """Without param_specs the sidecar enforces only what the bridge adds
    from the description's min / max -- which it used to leave out for this
    node, so an archive could install 1.5 against a declared [0, 1]."""
    md, sidecar, bridge = served
    reply = bridge.handle({"op": "set", "vr": [_vr(md, ELASTICITY)], "values": [1.5]})
    assert reply["ok"] is False and "above bound" in reply["error"], reply
    blob = state_of(bridge.handle({"op": "get_state"}))
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays[f"p/nodes/{NODE}/elasticity"] = np.asarray(1.5, np.float32)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    reply = bridge.handle({"op": "set_state",
                           "state": base64.b64encode(buf.getvalue()).decode("ascii")})
    assert reply["ok"] is False and "above bound" in reply["error"], reply


def test_its_start_value_is_where_an_instance_starts(gm):
    md = build_model_description(gm, model_name="P")
    other = _graph(elasticity=0.4)
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=other._compiled_step,
        initial_state=other._state, params=other.params))
    with pytest.warns(UserWarning, match=r"rig\.params\.v2\.params\.elasticity"):
        bridge = FmuTcpBridge(sidecar, md, master_dt=0.01)
    try:
        got = values_of(bridge.handle({"op": "get", "vr": [_vr(md, ELASTICITY)]}))
        assert got[0] == pytest.approx(0.7)
    finally:
        bridge.stop()


def test_its_fixed_parameter_is_refused_a_new_value(served):
    """``initial_position`` is not a variable the step reads; the fixed rule
    applied by name used to miss this node's."""
    md, sidecar, _ = served
    name = f"{NODE}.params.initial_position"
    assert name in md.fixed_parameters
    with pytest.raises(ValueError, match="is not tunable"):
        sidecar.set_params({name: 2.0})


def test_two_parameters_spelling_one_name_are_refused_by_that_name():
    sidecar = FmuSidecar(SidecarConfig(
        schema_token="t", step_fn=lambda s, e, p: s, initial_state={"n": {}},
        params={"nodes": {"a.params.b": {"c": jnp.asarray(1.0)},
                          "a": {"b.params.c": jnp.asarray(2.0)}}, "mappings": {}}))
    with pytest.raises(KeyError, match="is ambiguous"):
        sidecar.set_params({"a.params.b.params.c": 3.0})
