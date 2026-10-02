"""The FMU boundary takes numbers, and refuses what it used to coerce.

``_value_reference`` already refused a value reference spelt ``"10"``,
``true`` or ``10.9``: nothing on the wire is coerced.  The values next to
it were: a ``set`` of the JSON string ``"45"`` or the boolean ``true`` was
stored as 45.0 and 1.0, a ``step`` whose ``dt`` was ``"0.01"`` or ``true``
(a hundred master steps of 0.01 s) ran, ``t="5"`` moved the clock, an
FMU-state archive with no ``_time`` restored at t = 0, a state member
stored as the *string* ``"1.5"`` restored as the number 1.5, and
``FmuSidecar.set_params`` took ``"45"`` and ``True``.  Each is refused now,
with nothing written, and the lookalike that *is* a number still passes.

The two restore doors also disagreed on a snapshot carrying parameters
into a model with no parameter tree: the bridge refused it and
``FmuSidecar.set_fmu_state`` dropped them in silence.  Both refuse it, with
the same words.
"""

import base64
import io

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.fmi import build_model_description
from maddening.fmi.fmu_state import serialize_fmu_state
from maddening.fmi.model_description import FMIVariable, ModelDescription
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig, _checked_value, _restored_leaf
from maddening.fmi.tcp_bridge import FmuTcpBridge, state_of
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr


@pytest.fixture(scope="module")
def gm():
    return _graph()


@pytest.fixture
def served(gm):
    md, bridge = _bridge(gm)
    yield md, bridge
    bridge.stop()


def _snapshot(bridge, md):
    names = ("time", "spring.position", "spring.anchor_position", "spring.params.stiffness")
    return bridge.handle({"op": "get", "vr": [_vr(md, n) for n in names]})["values"]


# ------------------------------------------------------------------- set

@pytest.mark.parametrize("bad", ["45", True, False, None, [45.0], {"v": 45.0}, "NaNo"],
                         ids=["string", "true", "false", "null", "nested", "object", "token-ish"])
def test_set_refuses_a_value_that_is_not_a_number(served, bad):
    md, bridge = served
    k, anchor = _vr(md, "spring.params.stiffness"), _vr(md, "spring.anchor_position")
    before = _snapshot(bridge, md)
    # a valid value first in the same request: the set is atomic
    reply = bridge.handle({"op": "set", "vr": [anchor, k], "values": [0.75, bad]})
    assert reply["ok"] is False and "number" in reply["error"], reply
    assert _snapshot(bridge, md) == before


@pytest.mark.parametrize("bad", ["45", True, None], ids=["string", "bool", "null"])
def test_set_refuses_a_values_field_that_is_not_a_list(served, bad):
    md, bridge = served
    reply = bridge.handle({"op": "set", "vr": [_vr(md, "spring.params.stiffness")], "values": bad})
    assert reply["ok"] is False and "flat list of numbers" in reply["error"], reply


def test_set_takes_a_json_integer_and_the_binary_floats(served):
    """The lookalikes that are numbers: an integer, a float, an array."""
    md, bridge = served
    k = _vr(md, "spring.params.stiffness")
    assert bridge.handle({"op": "set", "vr": [k], "values": [45]}) == {"ok": True}
    assert bridge.handle({"op": "get", "vr": [k]})["values"] == [45.0]
    raw = np.asarray([40.0], "<f8").tobytes()
    assert bridge.handle({"op": "set", "vr": [k], "n": 1, "dtype": "f64", "raw": raw}) == {"ok": True}
    assert bridge.handle({"op": "get", "vr": [k]})["values"] == [40.0]


# ------------------------------------------------------------------ step

@pytest.mark.parametrize("field,bad", [
    ("dt", "0.01"), ("dt", True), ("dt", None), ("dt", [0.01]), ("dt", 10 ** 400),
    ("t", "0"), ("t", False), ("t", None), ("t", 10 ** 400),
])
def test_step_refuses_a_time_that_is_not_a_number(served, field, bad):
    md, bridge = served
    before = _snapshot(bridge, md)
    request = {"op": "step", "t": 0.0, "dt": DT, field: bad}
    reply = bridge.handle(request)
    assert reply["ok"] is False, reply
    assert ("must be a number" in reply["error"] or "must be finite" in reply["error"]), reply
    assert _snapshot(bridge, md) == before


def test_step_takes_integer_times(served):
    md, bridge = served
    assert bridge.handle({"op": "step", "t": 0, "dt": DT}) == {"ok": True, "t": DT}


# ------------------------------------------------------------ set_state

def _archive_members(bridge):
    blob = state_of(bridge.handle({"op": "get_state"}))
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _restore(bridge, members):
    buf = io.BytesIO()
    np.savez(buf, **members)
    return bridge.handle({"op": "set_state", "state": base64.b64encode(buf.getvalue()).decode()})


@pytest.mark.parametrize("edit", ["no time", "time as a string", "time as an array",
                                  "state as a string", "input as a string", "param as a bool"])
def test_set_state_refuses_an_archive_the_bridge_never_writes(served, edit):
    md, bridge = served
    assert bridge.handle({"op": "step", "t": 0.0, "dt": 2 * DT})["ok"]
    members = _archive_members(bridge)
    assert bridge.handle({"op": "step", "t": 2 * DT, "dt": DT})["ok"]
    before = _snapshot(bridge, md)
    if edit == "no time":
        members.pop("_time")
    elif edit == "time as a string":
        members["_time"] = np.array("0.02")
    elif edit == "time as an array":
        members["_time"] = np.array([0.02])
    elif edit == "state as a string":
        members["s/spring/position"] = np.array("1.5")
    elif edit == "input as a string":
        members["i/spring/anchor_position"] = np.array("1.5")
    else:
        members["p/nodes/spring/stiffness"] = np.array(True)
    reply = _restore(bridge, members)
    assert reply["ok"] is False, (edit, reply)
    assert _snapshot(bridge, md) == before
    # the unedited archive still restores
    members = _archive_members(bridge)
    assert _restore(bridge, members) == {"ok": True}


# --------------------------------------------------- the sidecar's door

@pytest.mark.parametrize("bad", ["45", True, None, b"45", 45 + 0j],
                         ids=["string", "bool", "none", "bytes", "complex"])
def test_set_params_refuses_what_set_refuses(gm, bad):
    md = build_model_description(gm, model_name="Plant")
    sidecar = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token,
                                       step_fn=gm._compiled_step, initial_state=gm._state,
                                       params=gm.params, param_specs=gm.param_specs()))
    with pytest.raises(ValueError, match="must be a number"):
        sidecar.set_params({"spring.params.stiffness": bad})
    assert sidecar.get_params()["spring.params.stiffness"] == 30.0
    sidecar.set_params({"spring.params.stiffness": 45})              # an int is a number
    sidecar.set_params({"spring.params.stiffness": np.float64(44.0)})
    assert sidecar.get_params()["spring.params.stiffness"] == 44.0


def test_the_value_check_keeps_the_kinds_a_model_holds():
    """Neighbouring cases of the new kind rule, which every restore runs:
    a boolean leaf restores from a boolean, a numeric wire value still sets
    a boolean variable, integers and complex values reach leaves of their
    own kind -- only a kind the leaf cannot be is refused."""
    assert _checked_value(np.array(True), np.bool_, what="b").dtype == np.bool_
    assert bool(_checked_value(np.array(1.0), np.bool_, what="b"))
    assert _checked_value(np.array(7), np.int32, what="i") == 7
    assert _checked_value(np.array(1 + 2j), np.complex64, what="c") == np.complex64(1 + 2j)
    assert bool(_restored_leaf(np.array(True), jnp.asarray(False), what="flag"))
    for value, dtype in ((np.array(True), np.float32), (np.array(1 + 2j), np.float32),
                         (np.array("1"), np.float32), (np.array([None]), np.float32)):
        with pytest.raises(ValueError, match="must be a number"):
            _checked_value(value, dtype, what="x")


def test_an_fmu_state_with_boolean_and_integer_fields_round_trips():
    """End to end on both restore paths: a model whose state holds a flag
    and a counter hands out a snapshot it then restores."""
    state = {"n": {"x": jnp.asarray(1.5, jnp.float32), "flag": jnp.asarray(True),
                   "count": jnp.asarray(3, jnp.int32)}}
    md = ModelDescription(model_name="m", instantiation_token="tok", variables=[FMIVariable(
        name="time", value_reference=1, dtype="float64", causality="independent",
        variability="continuous")], default_step_size=DT)
    sidecar = FmuSidecar(SidecarConfig(schema_token="tok", step_fn=lambda s, e: s,
                                       initial_state=state))
    bridge = FmuTcpBridge(sidecar, md, master_dt=DT)
    try:
        blob = bridge.handle({"op": "get_state"})["state"]
        assert bridge.handle({"op": "set_state", "state": blob}) == {"ok": True}
    finally:
        bridge.stop()
    sidecar.set_fmu_state(sidecar.get_fmu_state())
    assert bool(sidecar.state["n"]["flag"]) and int(sidecar.state["n"]["count"]) == 3


# ------------------------------------------------- the two restore doors

def test_both_restore_doors_refuse_parameters_into_a_model_with_none(gm):
    md = build_model_description(gm, model_name="Plant", include_parameters=False)
    full = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token,
                                    step_fn=lambda s, e, p: s, initial_state=gm._state,
                                    params=gm.params))
    carrying = FmuTcpBridge(full, md, master_dt=DT)
    plain = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token,
                                     step_fn=lambda s, e: s, initial_state=gm._state))
    bridge = FmuTcpBridge(FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=lambda s, e: s,
        initial_state=gm._state)), md, master_dt=DT)
    try:
        wire = carrying.handle({"op": "get_state"})["state"]
        reply = bridge.handle({"op": "set_state", "state": wire})
        with pytest.raises(ValueError) as refused:
            plain.set_fmu_state(serialize_fmu_state(
                state=gm._state, schema_token=md.instantiation_token, params=gm.params))
        assert reply == {"ok": False, "error": f"ValueError: {refused.value}"}
        assert "parameters differ from the model" in str(refused.value)
        # without parameters, both restore; an empty tree carries none either
        plain.set_fmu_state(serialize_fmu_state(state=gm._state,
                                                schema_token=md.instantiation_token))
        plain.set_fmu_state(serialize_fmu_state(state=gm._state,
                                                schema_token=md.instantiation_token,
                                                params={"nodes": {}, "mappings": {}}))
        own = bridge.handle({"op": "get_state"})["state"]
        assert bridge.handle({"op": "set_state", "state": own}) == {"ok": True}
    finally:
        carrying.stop()
        bridge.stop()
