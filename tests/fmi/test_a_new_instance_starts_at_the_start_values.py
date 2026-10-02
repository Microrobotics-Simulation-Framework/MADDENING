"""A new FMU instance starts where FMI says it does, not where the last one ended.

``fmi3FreeInstance`` followed by ``fmi3InstantiateCoSimulation`` is what a
master does between runs, and the bridge supports it (``_HANDOVER_GRACE``).
FMI 3.0 starts every instance at the start values of its
``modelDescription.xml``.  The bridge's instance slot used to reset nothing
when a new connection claimed it, so the second instance inherited the
first one's state, tuned parameters, pending inputs and time: FMPy's
``simulate_fmu`` called twice with the same arguments gave two answers,
every call returning ``fmi3OK``.

Now a claim of the slot is a reset to the instantiation point -- the state
the bridge was built over, every parameter at the description's ``start``,
every input at zero, the time at zero -- which is also where ``reset``
(``fmi3Reset``) goes.
"""

import socket

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.fmi import MODEL_IDENTIFIER, build_model_description
from maddening.fmi.package import find_c_compiler
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, recv_message, send_message, values_of
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr


@pytest.fixture(scope="module")
def gm():
    return _graph()


def _connect(bridge):
    host, port = bridge.endpoint.split(":")
    conn = socket.create_connection((host, int(port)), timeout=30)
    send_message(conn, {"op": "hello"})
    reply = recv_message(conn)
    assert reply["ok"], reply
    return conn


def _call(conn, request):
    send_message(conn, request)
    reply = recv_message(conn)
    assert reply is not None, "the bridge closed the connection"
    return reply


_NAMES = ("time", "spring.position", "spring.velocity", "ball.position",
          "spring.anchor_position", "spring.params.stiffness", "ball.params.elasticity")


def _everything(conn, md):
    reply = _call(conn, {"op": "get", "vr": [_vr(md, n) for n in _NAMES]})
    assert reply["ok"], reply
    return dict(zip(_NAMES, values_of(reply).tolist()))


def _drive(conn, md):
    """What an importer does to an instance: tune, drive an input, step."""
    k, el, anchor = (_vr(md, "spring.params.stiffness"), _vr(md, "ball.params.elasticity"),
                     _vr(md, "spring.anchor_position"))
    assert _call(conn, {"op": "set", "vr": [k, el, anchor],
                        "values": [45.0, 0.5, 0.25]}) == {"ok": True}
    assert _call(conn, {"op": "initialize", "t": 2.0})["ok"]
    for i in range(3):
        assert _call(conn, {"op": "step", "t": 2.0 + i * DT, "dt": DT})["ok"]


def test_a_connection_that_claims_the_slot_starts_from_the_start_values(gm):
    md, bridge = _bridge(gm)
    with bridge:
        conn = _connect(bridge)
        with conn:
            fresh = _everything(conn, md)
            _drive(conn, md)
            driven = _everything(conn, md)
        # the importer frees the instance and instantiates a new one
        conn = _connect(bridge)
        with conn:
            again = _everything(conn, md)
    assert driven != fresh                      # the first instance did move
    assert again == fresh
    # and "fresh" is what the description advertises
    starts = {v.name: float(v.start) for v in md.variables if v.start is not None
              and not v.shape}
    assert fresh["spring.params.stiffness"] == starts["spring.params.stiffness"] == 30.0
    assert fresh["ball.params.elasticity"] == starts["ball.params.elasticity"]
    assert fresh["spring.anchor_position"] == starts["spring.anchor_position"] == 0.0
    assert fresh["time"] == 0.0
    assert fresh["spring.position"] == 0.5 and fresh["ball.position"] == 1.0


def test_reset_and_a_new_instance_land_on_the_same_point(gm):
    """``fmi3Reset`` is defined as "the state after instantiation"; the two
    share one commit, and every value -- the full state included -- agrees."""
    md, bridge = _bridge(gm)
    with bridge:
        conn = _connect(bridge)
        with conn:
            _drive(conn, md)
            assert _call(conn, {"op": "reset"}) == {"ok": True}
            after_reset = _everything(conn, md)
            reset_state = {n: dict(f) for n, f in bridge._sidecar.state.items()}
            _drive(conn, md)
        conn = _connect(bridge)
        with conn:
            after_claim = _everything(conn, md)
            claim_state = {n: dict(f) for n, f in bridge._sidecar.state.items()}
            # the new instance is in FMI's Instantiated state: it may initialize
            assert _call(conn, {"op": "initialize", "t": 0.0})["ok"]
    assert after_claim == after_reset
    for node, fields in reset_state.items():
        for field, value in fields.items():
            np.testing.assert_array_equal(np.asarray(claim_state[node][field]),
                                          np.asarray(value), err_msg=f"{node}.{field}")


def test_a_refused_second_connection_resets_nothing(gm):
    """The reset belongs to a *successful* claim: a peer that is refused the
    slot -- while the importer still holds it -- must not wipe the importer's
    instance out from under it."""
    md, bridge = _bridge(gm)
    with bridge:
        conn = _connect(bridge)
        with conn:
            _drive(conn, md)
            before = _everything(conn, md)
            host, port = bridge.endpoint.split(":")
            with socket.create_connection((host, int(port)), timeout=30) as other:
                send_message(other, {"op": "hello"})
                refused = recv_message(other)
            assert refused["ok"] is False and "already serves" in refused["error"]
            assert _everything(conn, md) == before


def test_an_in_process_caller_is_not_a_new_instance(gm):
    """``handle()`` serves the instance that is there; it claims no slot, so
    a Python client driving the bridge in process keeps its state across
    calls (the many tests that drive a bridge this way rely on it)."""
    md, bridge = _bridge(gm)
    try:
        k = _vr(md, "spring.params.stiffness")
        assert bridge.handle({"op": "set", "vr": [k], "values": [40.0]}) == {"ok": True}
        assert bridge.handle({"op": "get", "vr": [k]})["values"] == [40.0]
    finally:
        bridge.stop()


def test_a_sidecar_configured_away_from_the_description_starts_at_the_description():
    """The description is the FMU's contract: a sidecar built with other
    parameter values than the description advertises is brought into line,
    with a warning naming them, so no instance computes with a value its
    XML denies -- and the restore check's "value it was instantiated with"
    exemption follows."""
    gm = _graph()
    md = build_model_description(gm, model_name="Plant", model_identifier=MODEL_IDENTIFIER)
    params = {s: {o: dict(v) for o, v in owners.items()} for s, owners in gm.params.items()}
    params["nodes"]["spring"]["stiffness"] = jnp.asarray(12.0, jnp.float32)
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=params, param_specs=gm.param_specs()))
    with pytest.warns(UserWarning, match=r"spring\.params\.stiffness"):
        bridge = FmuTcpBridge(sidecar, md, master_dt=DT)
    try:
        k = _vr(md, "spring.params.stiffness")
        assert bridge.handle({"op": "get", "vr": [k]})["values"] == [30.0]
        assert sidecar.get_params()["spring.params.stiffness"] == 30.0
        assert float(sidecar._initial_params["nodes"]["spring"]["stiffness"]) == 30.0
    finally:
        bridge.stop()


def test_a_sidecar_matching_its_description_is_not_warned_about(gm, recwarn):
    md, bridge = _bridge(gm)
    bridge.stop()
    assert not [w for w in recwarn if "start values" in str(w.message)]


@pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler")
def test_simulate_fmu_twice_with_the_same_arguments_gives_the_same_result(gm, tmp_path):
    """The reproducer of the finding: FMPy's ``simulate_fmu`` against one
    running bridge, three times.  Runs 1 and 2 have identical arguments and
    must agree with each other and with the graph; run 3 sets no start
    value and must compute with the advertised stiffness of 30, not run 1's
    45."""
    fmpy = pytest.importorskip("fmpy")
    from maddening.fmi.package import build_fmu_binary, write_fmu

    so = build_fmu_binary(tmp_path)
    md, bridge = _bridge(gm)
    sig = np.array([(0.0, 0.25), (1.0, 0.25)],
                   dtype=[("time", np.float64), ("spring.anchor_position", np.float64)])
    out = ["spring.position", "ball.position", "spring.params.stiffness"]
    kw = dict(start_time=0.0, stop_time=0.2, step_size=DT, output_interval=DT,
              input=sig, output=out)
    with bridge:
        fmu = str(write_fmu(md, tmp_path / "plant.fmu", binary=so, endpoint=bridge.endpoint))
        r1 = fmpy.simulate_fmu(fmu, start_values={"spring.params.stiffness": 45.0}, **kw)
        r2 = fmpy.simulate_fmu(fmu, start_values={"spring.params.stiffness": 45.0}, **kw)
        r3 = fmpy.simulate_fmu(fmu, **kw)
    for name in out:
        np.testing.assert_array_equal(r1[name], r2[name], err_msg=name)
    ext = {"spring": {"anchor_position": jnp.asarray(0.25, jnp.float32)}}
    ref = _graph()                       # run_scan moves its state: reset between runs
    p45 = ref.params
    p45["nodes"]["spring"]["stiffness"] = jnp.asarray(45.0, jnp.float32)
    ref45 = ref.run_scan(20, external_inputs=ext, params=p45)
    ref.reset_state()
    ref.params = _graph().params
    ref30 = ref.run_scan(20, external_inputs=ext)
    assert r1["spring.position"][-1] == pytest.approx(float(ref45["spring"]["position"]), rel=1e-5)
    assert r3["spring.params.stiffness"][0] == 30.0
    assert r3["spring.position"][0] == 0.5
    assert r3["spring.position"][-1] == pytest.approx(float(ref30["spring"]["position"]), rel=1e-5)
