"""An FMU-state archive cannot install new interface-mapping weights.

The bridge's contract: an archive may only install what a ``set`` of the
same variables could.  Mapping weights (``params["mappings"]``) are never
FMI variables, so no ``set`` reaches them, but the archive carries them as
``p/mappings/<edge>/<key>`` and ``set_state`` installed any finite array of
the right shape.  The step reads them, so the FMU then computed a coupling
its description does not describe: identity weights forged to
``[[0, 10], [-3, 0]]`` turned ``b.x`` from ``[1, 2]`` into ``[20, -3]``,
every call ``ok``.  Every mapping leaf is now fixed, on the bridge and in
``FmuSidecar.set_fmu_state``, with the same message.
"""

from __future__ import annotations

import base64
import io

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.fmi import build_model_description
from maddening.fmi.fmu_state import serialize_fmu_state
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, state_of, values_of

DT = 0.01
FORGED = np.asarray([[0.0, 10.0], [-3.0, 0.0]], np.float32)


@stability(StabilityLevel.STABLE)
class _Source(SimulationNode):
    def __init__(self, name, timestep):
        super().__init__(name, timestep)

    def initial_state(self):
        return {"v": jnp.asarray([1.0, 2.0], jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"v": state["v"]}


@stability(StabilityLevel.STABLE)
class _Sink(SimulationNode):
    def __init__(self, name, timestep):
        super().__init__(name, timestep)

    def initial_state(self):
        return {"x": jnp.zeros(2, jnp.float32)}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(2,), dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": boundary_inputs.get("inp", jnp.zeros(2, jnp.float32))}


@pytest.fixture(scope="module")
def gm():
    gm = GraphManager()
    gm.add_node(_Source("a", DT))
    gm.add_node(_Sink("b", DT))
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(np.eye(2, dtype=np.float32)))
    gm.compile()
    return gm


def _sidecar(gm, md):
    return FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs()))


@pytest.fixture
def served(gm):
    md = build_model_description(gm, model_name="M")
    sidecar = _sidecar(gm, md)
    bridge = FmuTcpBridge(sidecar, md, master_dt=DT)
    yield md, sidecar, bridge
    bridge.stop()


def _bx(md, bridge):
    vr = next(v.value_reference for v in md.variables if v.name == "b.x")
    return values_of(bridge.handle({"op": "get", "vr": [vr]})).tolist()


def _forged_archive(bridge):
    blob = state_of(bridge.handle({"op": "get_state"}))
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    key = next(k for k in arrays if k.startswith("p/mappings/"))
    arrays[key] = FORGED.astype(arrays[key].dtype)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_an_archive_with_new_mapping_weights_is_refused(served):
    md, sidecar, bridge = served
    assert [v.name for v in md.variables if v.causality == "parameter"] == []
    bridge.handle({"op": "step", "t": 0.0, "dt": DT})
    assert _bx(md, bridge) == [1.0, 2.0]
    before = bridge.handle({"op": "get_state"})["state"]
    reply = bridge.handle({"op": "set_state", "state": _forged_archive(bridge)})
    assert reply["ok"] is False, reply
    assert "interface-mapping weights 'a.v->b.inp' / 'H'" in reply["error"], reply
    assert bridge.handle({"op": "get_state"})["state"] == before            # nothing restored
    bridge.handle({"op": "step", "t": DT, "dt": DT})
    assert _bx(md, bridge) == [1.0, 2.0]                                      # still identity
    np.testing.assert_array_equal(sidecar.params["mappings"]["a.v->b.inp"]["H"], np.eye(2))


def test_the_fmus_own_snapshot_still_restores(served):
    md, sidecar, bridge = served
    snap = bridge.handle({"op": "get_state"})["state"]
    bridge.handle({"op": "step", "t": 0.0, "dt": DT})
    assert bridge.handle({"op": "set_state", "state": snap})["ok"] is True


def test_the_sidecars_own_restore_door_refuses_the_same_weights(gm, served):
    md, _, bridge = served
    sidecar = _sidecar(gm, md)
    params = {section: {owner: dict(leaves) for owner, leaves in owners.items()}
              for section, owners in gm.params.items()}
    params["mappings"]["a.v->b.inp"]["H"] = jnp.asarray(FORGED)
    snap = serialize_fmu_state(state=gm._state, schema_token=md.instantiation_token,
                               params=params)
    with pytest.raises(ValueError) as refused:
        sidecar.set_fmu_state(snap)
    reply = bridge.handle({"op": "set_state", "state": _forged_archive(bridge)})
    assert reply["error"] == f"ValueError: {refused.value}"                 # one message
    # and the unedited snapshot restores through the same door
    sidecar.set_fmu_state(sidecar.get_fmu_state())
