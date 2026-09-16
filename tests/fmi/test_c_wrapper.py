"""The FMU C wrapper end to end: build the binary, package an FMU, drive it
through FMPy against a :class:`FmuTcpBridge`, and compare with the graph.

Skipped when no C compiler or FMPy is available.  The bridge's request
handler and the wire framing are tested without a compiler too.
"""

import os
import socket
import threading

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.fmi.package import (
    MODEL_IDENTIFIER, build_fmu_binary, find_c_compiler, write_fmu,
)
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, recv_message, send_message
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

DT = 1e-2


def _graph():
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=DT))
    gm.add_node(BallNode(name="ball", timestep=DT, initial_position=1.0, elasticity=0.7))
    gm.add_node(SpringDamperNode(name="spring", timestep=DT, stiffness=30.0, damping=2.0,
                                 initial_position=0.5))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.add_external_input("spring", "anchor_position")
    gm.compile()
    return gm


def _bridge(gm, **md_kw):
    md = build_model_description(gm, model_name="Plant", model_identifier=MODEL_IDENTIFIER,
                                 **md_kw)
    sc = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
    ))
    return md, FmuTcpBridge(sc, md, master_dt=DT)


def _vr(md, name):
    return next(v.value_reference for v in md.variables if v.name == name)


# ---------------------------------------------------------------- handler

def test_bridge_handler_set_get_step_matches_graph():
    gm = _graph()
    md, bridge = _bridge(gm)
    k, anchor, pos = _vr(md, "spring.params.stiffness"), _vr(md, "spring.anchor_position"), \
        _vr(md, "spring.position")
    assert bridge.handle({"op": "hello"})["token"] == md.instantiation_token
    assert bridge.handle({"op": "set", "vr": [k, anchor], "values": [45.0, 0.25]}) == {"ok": True}
    for i in range(5):
        r = bridge.handle({"op": "step", "t": i * 3 * DT, "dt": 3 * DT})     # 3 graph steps each
        assert r["ok"] and r["t"] == pytest.approx((i + 1) * 3 * DT)
    got = bridge.handle({"op": "get", "vr": [pos, k, _vr(md, "time")]})["values"]

    ref = _graph()
    p = ref.params
    p["nodes"]["spring"]["stiffness"] = jnp.asarray(45.0, jnp.float32)
    ext = {"spring": {"anchor_position": jnp.asarray(0.25, jnp.float32)}}
    out = ref.run_scan(15, external_inputs=ext, params=p)
    assert got[0] == pytest.approx(float(out["spring"]["position"]), rel=1e-6)
    assert got[1] == 45.0 and got[2] == pytest.approx(15 * DT)


def test_bridge_handler_errors_and_state_round_trip():
    gm = _graph()
    md, bridge = _bridge(gm)
    pos, el = _vr(md, "ball.position"), _vr(md, "ball.params.elasticity")
    r = bridge.handle({"op": "set", "vr": [pos], "values": [3.0]})
    assert not r["ok"] and "read-only" in r["error"]
    r = bridge.handle({"op": "set", "vr": [el], "values": [1.5]})           # out of bounds
    assert not r["ok"] and "above bound" in r["error"]
    assert not bridge.handle({"op": "get", "vr": [9999]})["ok"]
    assert not bridge.handle({"op": "nope"})["ok"]
    bridge.handle({"op": "step", "t": 0.0, "dt": DT})
    snap = bridge.handle({"op": "get_state"})["state"]
    before = bridge.handle({"op": "get", "vr": [pos]})["values"]
    bridge.handle({"op": "step", "t": DT, "dt": 5 * DT})
    assert bridge.handle({"op": "set_state", "state": snap})["ok"]
    assert bridge.handle({"op": "get", "vr": [pos]})["values"] == before
    assert bridge.handle({"op": "reset"})["ok"]
    assert bridge.handle({"op": "get", "vr": [pos, _vr(md, "time")]})["values"] == [1.0, 0.0]


def test_wire_framing_over_a_real_socket():
    gm = _graph()
    md, bridge = _bridge(gm)
    with bridge:
        host, port = bridge.endpoint.split(":")
        with socket.create_connection((host, int(port)), timeout=5) as conn:
            send_message(conn, {"op": "hello"})
            assert recv_message(conn)["model"] == "Plant"
            send_message(conn, {"op": "step", "t": 0.0, "dt": DT})
            assert recv_message(conn)["ok"]
            send_message(conn, {"op": "get", "vr": [_vr(md, "ball.position")]})
            assert recv_message(conn)["values"][0] < 1.0
    assert bridge.requests_served == 3


# ------------------------------------------------------------ compiled FMU

needs_cc = pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler")


@needs_cc
def test_binary_builds_and_exports_the_fmi3_symbols(tmp_path):
    import ctypes

    so = build_fmu_binary(tmp_path)
    lib = ctypes.CDLL(str(so))
    lib.fmi3GetVersion.restype = ctypes.c_char_p
    assert lib.fmi3GetVersion() == b"3.0"
    for name in ("fmi3InstantiateCoSimulation", "fmi3DoStep", "fmi3GetFloat32",
                 "fmi3SetFloat64", "fmi3GetFMUState", "fmi3SetFMUState", "fmi3Reset"):
        assert hasattr(lib, name)


@needs_cc
def test_fmpy_drives_the_compiled_fmu_against_the_graph(tmp_path):
    fmpy = pytest.importorskip("fmpy")
    from fmpy import simulate_fmu

    gm = _graph()
    md, bridge = _bridge(gm)
    so = build_fmu_binary(tmp_path)
    with bridge:
        fmu = write_fmu(md, tmp_path / "plant.fmu", binary=so, endpoint=bridge.endpoint)
        anchor_signal = np.array([(0.0, 0.25), (0.5, 0.25)],
                                 dtype=[("time", np.float64), ("spring.anchor_position", np.float64)])
        res = simulate_fmu(
            str(fmu), start_time=0.0, stop_time=0.3, step_size=DT, output_interval=DT,
            start_values={"spring.params.stiffness": 45.0}, input=anchor_signal,
            output=["ball.position", "spring.position", "spring.params.stiffness"],
        )
    assert bridge.requests_served > 30
    # the wrapper negotiated protocol 2: get replies and set requests were binary frames
    assert bridge.binary_frames_served > 20 and bridge.binary_frames_received > 0

    ref = _graph()
    p = ref.params
    p["nodes"]["spring"]["stiffness"] = jnp.asarray(45.0, jnp.float32)
    ext = {"spring": {"anchor_position": jnp.asarray(0.25, jnp.float32)}}
    n = int(round(0.3 / DT))
    out = ref.run_scan(n, external_inputs=ext, params=p)
    assert res["time"][-1] == pytest.approx(0.3)
    assert res["spring.params.stiffness"][-1] == 45.0
    assert res["spring.position"][-1] == pytest.approx(float(out["spring"]["position"]), rel=1e-5)
    assert res["ball.position"][-1] == pytest.approx(float(out["ball"]["position"]), rel=1e-5)
    # the trajectory really moved through the FMU
    assert abs(res["spring.position"][-1] - res["spring.position"][0]) > 1e-3


@needs_cc
def test_low_level_fmi3_api_two_instances_state_and_reset(tmp_path):
    """Use the FMU the way a master algorithm does, through FMPy's FMI 3
    binding rather than ``simulate_fmu``: two instances against two
    bridges in the same process, parameters set before initialisation,
    get/set FMU state, serialize/deserialize, reset, terminate."""
    fmpy = pytest.importorskip("fmpy")
    from fmpy import extract, read_model_description
    from fmpy.fmi3 import FMU3Slave

    gm_a, gm_b = _graph(), _graph()
    md_a, bridge_a = _bridge(gm_a)
    md_b, bridge_b = _bridge(gm_b)
    so = build_fmu_binary(tmp_path)
    with bridge_a, bridge_b:
        fmu_a = write_fmu(md_a, tmp_path / "a.fmu", binary=so, endpoint=bridge_a.endpoint)
        fmu_b = write_fmu(md_b, tmp_path / "b.fmu", binary=so, endpoint=bridge_b.endpoint)
        insts = []
        for fmu, md in ((fmu_a, md_a), (fmu_b, md_b)):
            unz = extract(str(fmu))
            desc = read_model_description(unz)
            inst = FMU3Slave(guid=desc.guid, unzipDirectory=unz,
                             modelIdentifier=desc.coSimulation.modelIdentifier, instanceName="i")
            inst.instantiate(loggingOn=True)
            insts.append((inst, desc))
        vr = {v.name: v.valueReference for v in insts[0][1].modelVariables}
        k, pos, anchor = vr["spring.params.stiffness"], vr["spring.position"], vr["spring.anchor_position"]
        (ia, _), (ib, _) = insts
        ia.setFloat64([k], [45.0])                          # before initialisation
        ib.setFloat64([k], [30.0])
        for inst in (ia, ib):
            inst.enterInitializationMode(startTime=0.0)
            inst.exitInitializationMode()
        ia.setFloat64([anchor], [0.25])
        t = 0.0
        for _ in range(5):
            ia.doStep(currentCommunicationPoint=t, communicationStepSize=DT)
            ib.doStep(currentCommunicationPoint=t, communicationStepSize=DT)
            t += DT
        assert ia.getFloat64([k])[0] == 45.0 and ib.getFloat64([k])[0] == 30.0
        pa, pb = ia.getFloat64([pos])[0], ib.getFloat64([pos])[0]
        assert pa != pb                                     # instances are independent
        # state round trip, both in memory and serialized
        st = ia.getFMUState()
        blob = ia.serializeFMUState(st)
        for _ in range(5):
            ia.doStep(currentCommunicationPoint=t, communicationStepSize=DT)
            t += DT
        assert ia.getFloat64([pos])[0] != pa
        ia.setFMUState(st)
        assert ia.getFloat64([pos])[0] == pa
        st2 = ia.deserializeFMUState(blob)
        ia.setFMUState(st2)
        assert ia.getFloat64([pos])[0] == pa
        ia.freeFMUState(st)
        ia.freeFMUState(st2)
        # reset returns to the initial state and constructor params
        ia.reset()
        assert ia.getFloat64([pos])[0] == 0.5 and ia.getFloat64([k])[0] == 30.0
        for inst in (ia, ib):
            inst.terminate()
            inst.freeInstance()
    # the reference graph agrees with instance a's first 5 steps
    ref = _graph()
    p = ref.params
    p["nodes"]["spring"]["stiffness"] = jnp.asarray(45.0, jnp.float32)
    out = ref.run_scan(5, external_inputs={"spring": {"anchor_position": jnp.asarray(0.25, jnp.float32)}},
                       params=p)
    assert pa == pytest.approx(float(out["spring"]["position"]), rel=1e-6)


@needs_cc
def test_fmpy_validate_fmu_reports_no_problems(tmp_path):
    fmpy = pytest.importorskip("fmpy")
    from fmpy.validation import validate_fmu

    gm = _graph()
    md, _ = _bridge(gm)
    fmu = write_fmu(md, tmp_path / "plant.fmu", binary=build_fmu_binary(tmp_path),
                    endpoint="127.0.0.1:1")
    assert validate_fmu(str(fmu)) == []
