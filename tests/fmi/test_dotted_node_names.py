"""A ``.`` in a node name must not change what an FMU variable addresses.

FMI names inputs and outputs ``<node>.<field>`` and reads a dot as a
name-space separator, but a graph node name may contain one --  ``tank.1``,
``zone.A`` and ``hx.hot`` are ordinary engineering names, and the property
suite builds graphs with ``a.b`` in them.  The bridge used to recover the
pair by splitting the variable name at the *first* dot, so a ``set`` on
``rod.A.left_temperature`` was acknowledged and filed under a node ``rod``
the graph does not have (the value silently never arrived), and a ``get``
on an output raised ``KeyError: 'rod'`` -- after the co-simulation had
already run wrong (whole-tree audit W5).

``FMIVariable`` now carries the ``(node, field)`` pair the description was
built from, so nothing has to guess where the name splits.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.fmi import build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge
from maddening.nodes.heat import HeatNode

DT = 0.01


def _bridge_for(node_name):
    gm = GraphManager()
    gm.add_node(HeatNode(node_name, DT, n_cells=4, initial_temperature=0.0))
    gm.add_external_input(target_node=node_name, target_field="left_temperature",
                          shape=())
    gm.compile()
    md = build_model_description(gm, model_name="m")
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token,
        step_fn=lambda s, e, p: gm._compiled_step(s, e, p),
        initial_state=gm._state, params=gm.params,
        param_specs=gm.param_specs(),
    ))
    return md, FmuTcpBridge(sidecar, md, master_dt=DT), sidecar


def _drive(node_name):
    """``set`` the input, ``step``, then read the output back."""
    md, bridge, sidecar = _bridge_for(node_name)
    try:
        variables = {v.causality: v for v in md.variables
                     if v.causality in ("input", "output")}
        assert variables["input"].name == f"{node_name}.left_temperature"
        set_reply = bridge.handle({"op": "set",
                                   "vr": [variables["input"].value_reference],
                                   "values": [500.0]})
        step_reply = bridge.handle({"op": "step", "t": 0.0, "dt": 5 * DT})
        get_reply = bridge.handle({"op": "get",
                                   "vr": [variables["output"].value_reference]})
        state = np.asarray(sidecar.state[node_name]["temperature"])
        return set_reply, step_reply, get_reply, state
    finally:
        bridge.stop()


@pytest.mark.parametrize("node_name", ["rod", "rod.A", "tank.1", "a.b.c"])
def test_a_node_name_containing_a_dot_routes_fmu_inputs_to_that_node(node_name):
    """The same graph reaches the same state whatever the name's dots.

    ``rod`` is the reference: every other name differs only in characters
    the physics does not see, so the trajectory must be identical.
    """
    _, _, _, reference = _drive("rod")
    set_reply, step_reply, get_reply, state = _drive(node_name)

    assert set_reply == {"ok": True}
    assert step_reply["ok"] is True
    assert get_reply["ok"] is True, get_reply
    # The input arrived: the driven cell has moved off its initial zero
    # towards the 500 imposed at the rod end.  It does not reach 500 --
    # the cell centre is half a cell inside the boundary and is advanced
    # by the stencil, not overwritten (MADD-ANO-007).
    assert 0.0 < float(state[0]) < 500.0
    assert float(state[0]) > float(state[1])
    np.testing.assert_allclose(state, reference, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(get_reply["values"]), reference,
                               rtol=1e-6, atol=1e-6)


def test_a_dotted_node_keeps_its_fmu_parameters_addressable():
    """``<node>.params.<key>`` still resolves for a dotted node name."""
    md, bridge, _ = _bridge_for("rod.A")
    try:
        param = next(v for v in md.variables if v.causality == "parameter")
        assert param.name.startswith("rod.A.params.")
        vr = param.value_reference
        assert bridge.handle({"op": "set", "vr": [vr], "values": [0.25]}) == {"ok": True}
        reply = bridge.handle({"op": "get", "vr": [vr]})
        assert reply["ok"] is True
        assert reply["values"][0] == pytest.approx(0.25)
    finally:
        bridge.stop()


# Tagged STABLE so the FMU export admits their state fields as outputs
# (``build_model_description`` exports only stable surfaces).
@stability(StabilityLevel.STABLE)
class _DottedFieldNode(SimulationNode):
    """A node whose *field* name carries the dot instead of the node name."""

    def initial_state(self):
        return {"b.c": jnp.zeros((), dtype=jnp.float32)}

    def state_fields(self):
        return ["b.c"]

    def update(self, state, boundary_inputs, dt):
        return {"b.c": state["b.c"]}


@stability(StabilityLevel.STABLE)
class _PlainFieldNode(SimulationNode):
    def initial_state(self):
        return {"c": jnp.zeros((), dtype=jnp.float32)}

    def state_fields(self):
        return ["c"]

    def update(self, state, boundary_inputs, dt):
        return {"c": state["c"]}


def test_two_nodes_that_spell_the_same_fmu_variable_are_refused_by_name():
    """The one ambiguity the explicit pair cannot resolve is a hard error.

    ``a`` with field ``b.c`` and ``a.b`` with field ``c`` both spell
    ``a.b.c``.  FMI requires unique variable names, so the export refuses
    rather than writing an FMU whose variables an importer cannot tell
    apart.
    """
    gm = GraphManager()
    gm.add_node(_DottedFieldNode(name="a", timestep=DT))
    gm.add_node(_PlainFieldNode(name="a.b", timestep=DT))
    gm.compile()

    with pytest.raises(ValueError, match="FMU variable name clash") as excinfo:
        build_model_description(gm, model_name="m")
    message = str(excinfo.value)
    assert "'a.b.c'" in message
    assert "'a'" in message and "'a.b'" in message
