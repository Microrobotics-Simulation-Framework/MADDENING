"""A Boolean leaf takes a boolean, or exactly 0 or 1: never a number by its
truthiness.

The one value check on every write path into a sidecar
(``sidecar._checked_value``: the bridge's ``set`` and ``set_state``,
``FmuSidecar.set_params`` and ``FmuSidecar.set_fmu_state``) let any
integer or float through to a boolean target and cast it with
``astype(bool)``.  So ``set gate.open = 0.5`` (or ``2.0``, or ``-3.0``)
answered ``ok`` and read back as 1, and an FMU-state archive holding
``0.25`` in a boolean state field restored it as ``True``.  On the wire
every value is a float64 -- ``fmi3SetBoolean`` sends 1.0 or 0.0 -- so
exactly 0 and 1 stay accepted, as does a boolean array.
"""

from __future__ import annotations

import base64
import io

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.fmi import build_model_description
from maddening.fmi.fmu_state import serialize_fmu_state
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig, _checked_value
from maddening.fmi.tcp_bridge import FmuTcpBridge, state_of

DT = 1e-2


class _Gate(SimulationNode):
    """Integrates its rate while an external boolean ``open`` is set, and
    remembers whether it was open."""

    def __init__(self, name, timestep):
        super().__init__(name, timestep, rate=1.0)

    def initial_state(self):
        return {"level": jnp.asarray(0.0, jnp.float32), "was_open": jnp.asarray(False)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        is_open = boundary_inputs.get("open", jnp.asarray(False))
        return {"level": state["level"] + jnp.where(is_open, p["rate"] * dt, 0.0),
                "was_open": jnp.asarray(is_open, bool)}


@pytest.fixture(scope="module")
def gate_graph():
    gm = GraphManager()
    gm.add_node(_Gate("gate", DT))
    gm.add_external_input("gate", "open", dtype=jnp.bool_)
    gm.compile()
    return gm


@pytest.fixture
def served(gate_graph):
    md = build_model_description(gate_graph, model_name="G", include_evolving=True)
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gate_graph._compiled_step,
        initial_state=gate_graph._state, params=gate_graph.params,
        param_specs=gate_graph.param_specs()))
    bridge = FmuTcpBridge(sidecar, md, master_dt=DT)
    yield md, bridge
    bridge.stop()


def _vr(md, name):
    return next(v.value_reference for v in md.variables if v.name == name)


@pytest.mark.parametrize("value", [0.5, 2.0, -3.0, 1e-300, 0.9999999])
def test_a_set_of_a_boolean_input_to_a_non_0_1_number_is_refused(served, value):
    md, bridge = served
    vr = _vr(md, "gate.open")
    assert next(v.dtype for v in md.variables if v.name == "gate.open") == "bool"
    assert bridge.handle({"op": "set", "vr": [vr], "values": [1.0]}) == {"ok": True}
    reply = bridge.handle({"op": "set", "vr": [vr], "values": [value]})
    assert reply["ok"] is False and "a Boolean takes true / false or exactly 1 / 0" in reply["error"]
    assert bridge.handle({"op": "get", "vr": [vr]})["values"] == [1.0]      # nothing written


def test_exactly_0_and_1_set_a_boolean_input_and_reach_the_step(served):
    md, bridge = served
    vr = _vr(md, "gate.open")
    for value, want in ((1.0, 1.0), (0.0, 0.0), (-0.0, 0.0), (1, 1.0)):
        assert bridge.handle({"op": "set", "vr": [vr], "values": [value]}) == {"ok": True}
        assert bridge.handle({"op": "get", "vr": [vr]})["values"] == [want]
    # the binary path carries the same float64s
    raw = np.asarray([1.0], "<f8").tobytes()
    assert bridge.handle({"op": "set", "vr": [vr], "n": 1, "dtype": "f64", "raw": raw}) == {"ok": True}
    bad = np.asarray([0.5], "<f8").tobytes()
    assert bridge.handle({"op": "set", "vr": [vr], "n": 1, "dtype": "f64", "raw": bad})["ok"] is False
    assert bridge.handle({"op": "step", "t": 0.0, "dt": DT})["ok"]
    sidecar_state = bridge._sidecar.state["gate"]                            # noqa: SLF001
    assert bool(sidecar_state["was_open"]) and float(sidecar_state["level"]) == pytest.approx(DT)


def _archive_with(bridge, **members):
    blob = state_of(bridge.handle({"op": "get_state"}))
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays.update(members)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.parametrize("member", ["s/gate/was_open", "i/gate/open"])
def test_an_archive_with_a_fraction_in_a_boolean_member_does_not_restore(served, member):
    md, bridge = served
    before = bridge.handle({"op": "get_state"})["state"]
    reply = bridge.handle({"op": "set_state", "state": _archive_with(bridge, **{member: np.asarray(0.25)})})
    assert reply["ok"] is False and "exactly 1 / 0" in reply["error"], reply
    assert bridge.handle({"op": "get_state"})["state"] == before
    # an archive holding the same member as 1.0, 1 or True restores
    for good in (np.asarray(1.0), np.asarray(1, np.int64), np.asarray(True)):
        reply = bridge.handle({"op": "set_state", "state": _archive_with(bridge, **{member: good})})
        assert reply["ok"] is True, (good, reply)


def test_the_sidecars_own_restore_door_refuses_it_too(gate_graph):
    sidecar = FmuSidecar(SidecarConfig(schema_token="tok", step_fn=gate_graph._compiled_step,
                                       initial_state=gate_graph._state))
    state = {"gate": {"level": np.float32(0.0), "was_open": np.asarray(0.25)}}
    with pytest.raises(ValueError, match="exactly 1 / 0"):
        sidecar.set_fmu_state(serialize_fmu_state(state=state, schema_token="tok"))
    state["gate"]["was_open"] = np.asarray(1.0)
    sidecar.set_fmu_state(serialize_fmu_state(state=state, schema_token="tok"))
    assert bool(sidecar.state["gate"]["was_open"])


def test_set_params_holds_a_boolean_parameter_to_the_same_rule():
    sidecar = FmuSidecar(SidecarConfig(
        schema_token="t", step_fn=lambda s, e, p: s, initial_state={"n": {}},
        params={"nodes": {"n": {"enabled": jnp.asarray(False)}}, "mappings": {}}))
    with pytest.raises(ValueError, match="exactly 1 / 0"):
        sidecar.set_params({"n.params.enabled": 0.5})
    sidecar.set_params({"n.params.enabled": 1.0})
    assert bool(sidecar.get_params()["n.params.enabled"])
    sidecar.set_params({"n.params.enabled": False})
    assert not bool(sidecar.get_params()["n.params.enabled"])


def test_the_value_check_itself():
    """``_checked_value`` is the one check behind every door above."""
    for ok, want in ((np.asarray([0.0, 1.0]), [False, True]), (np.asarray(1), [True]),
                     (np.asarray([True, False]), [True, False]), (np.asarray(-0.0), [False])):
        assert list(np.ravel(_checked_value(ok, np.bool_, what="b"))) == want
    for bad in (np.asarray(0.5), np.asarray([1.0, 2.0]), np.asarray(-1), np.asarray(np.nan)):
        with pytest.raises(ValueError):
            _checked_value(bad, np.bool_, what="b")
    # numeric targets are unchanged: a bool is still refused for a float leaf
    with pytest.raises(ValueError, match="must be a number"):
        _checked_value(np.asarray(True), np.float32, what="f")
    assert _checked_value(np.asarray(0.5), np.float32, what="f") == np.float32(0.5)
