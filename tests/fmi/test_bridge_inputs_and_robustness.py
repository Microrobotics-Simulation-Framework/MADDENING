"""``FmuTcpBridge`` behaves like the graph and survives bad clients.

* an input the importer never set is the advertised zero (a ``HeatNode``
  FMU no longer runs adiabatic until the first ``fmi3Set*``), also after
  ``reset`` and ``set_state``;
* a multi-sub-step ``step`` that fails leaves state and time untouched;
* a malformed request gets an error reply instead of a dropped socket;
* a state-blob member larger than the live leaf is refused before it is
  decompressed, and a non-finite time is refused.

Originally written from the independent audit of 2026-09-16 (round 4; report and
reproducers under ``benchmarks/results/audit4/``).
"""

import base64
import io
import socket
import struct

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import MODEL_IDENTIFIER, build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, recv_message, send_message
from maddening.nodes.heat import HeatNode
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr


def _heat_graph():
    gm = GraphManager()
    gm.add_node(HeatNode("h", 1e-3, n_cells=8, thermal_diffusivity=0.5, initial_temperature=100.0))
    gm.add_external_input("h", "left_temperature")
    gm.compile()
    return gm


def _heat_bridge(gm):
    md = build_model_description(gm, model_name="Heat", model_identifier=MODEL_IDENTIFIER)
    sc = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token, step_fn=gm._compiled_step,
                                  initial_state=gm._state, params=gm.params))
    return md, FmuTcpBridge(sc, md, master_dt=1e-3)


def test_unset_input_is_the_advertised_zero_like_the_graph():
    gm = _heat_graph()
    md, bridge = _heat_bridge(gm)
    temp = _vr(md, "h.temperature")
    for i in range(5):
        assert bridge.handle({"op": "step", "t": i * 1e-3, "dt": 1e-3})["ok"]
    got = np.asarray(bridge.handle({"op": "get", "vr": [temp]})["values"])
    ref = np.asarray(_heat_graph().run_scan(5)["h"]["temperature"])
    np.testing.assert_allclose(got, ref, rtol=1e-5)
    assert got[0] < 50.0                                   # the zero left boundary acted
    # after reset and after set_state from a snapshot taken before any set
    snap = bridge.handle({"op": "get_state"})["state"]
    assert bridge.handle({"op": "reset"})["ok"]
    bridge.handle({"op": "step", "t": 0.0, "dt": 1e-3})
    assert np.asarray(bridge.handle({"op": "get", "vr": [temp]})["values"])[0] < 100.0
    assert bridge.handle({"op": "set_state", "state": snap})["ok"]
    bridge.handle({"op": "step", "t": 5e-3, "dt": 1e-3})
    ref6 = np.asarray(_heat_graph().run_scan(6)["h"]["temperature"])
    np.testing.assert_allclose(np.asarray(bridge.handle({"op": "get", "vr": [temp]})["values"]),
                               ref6, rtol=1e-5)


def test_failed_sub_step_leaves_state_and_time_untouched():
    gm = _graph()
    md, bridge = _bridge(gm)
    pos, t = _vr(md, "spring.position"), _vr(md, "time")
    calls = {"n": 0}
    real = bridge._sidecar._config.step_fn

    def flaky(state, ext, params=None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return real(state, ext, params) if params is not None else real(state, ext)

    bridge._sidecar._config = bridge._sidecar._config.__class__(
        **{**bridge._sidecar._config.__dict__, "step_fn": flaky})
    before = bridge.handle({"op": "get", "vr": [pos, t]})["values"]
    r = bridge.handle({"op": "step", "t": 0.0, "dt": 5 * DT})
    assert not r["ok"] and "boom" in r["error"]
    assert bridge.handle({"op": "get", "vr": [pos, t]})["values"] == before


def test_malformed_request_gets_an_error_reply_not_a_disconnect():
    gm = _graph()
    md, bridge = _bridge(gm)
    with bridge:
        host, port = bridge.endpoint.split(":")
        with socket.create_connection((host, int(port)), timeout=5) as conn:
            send_message(conn, {"op": "hello"})
            assert recv_message(conn)["ok"]
            body = b'{"op":"set_state","state":"abc"def"}'
            conn.sendall(struct.pack(">I", len(body)) + body)
            r = recv_message(conn)
            assert r is not None and not r["ok"] and "malformed" in r["error"]
            conn.sendall(struct.pack(">I", 3) + b"[1]")
            assert not recv_message(conn)["ok"]
            send_message(conn, {"op": "step", "t": 0.0, "dt": DT})     # still alive
            assert recv_message(conn)["ok"]


def test_state_member_larger_than_the_model_is_refused_before_loading():
    gm = _graph()
    md, bridge = _bridge(gm)
    raw = base64.b64decode(bridge.handle({"op": "get_state"})["state"])
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    bomb = dict(arrays)
    bomb["s/spring/position"] = np.zeros(50_000_000, np.float32)        # 200 MB, compresses to ~200 KB
    buf = io.BytesIO()
    np.savez_compressed(buf, **bomb)
    blob = base64.b64encode(buf.getvalue()).decode("ascii")
    assert len(blob) < 2_000_000
    r = bridge.handle({"op": "set_state", "state": blob})
    assert not r["ok"] and "more than" in r["error"]
    nan_t = dict(arrays)
    nan_t["_time"] = np.array(float("nan"))
    buf = io.BytesIO()
    np.savez(buf, **nan_t)
    r = bridge.handle({"op": "set_state", "state": base64.b64encode(buf.getvalue()).decode("ascii")})
    assert not r["ok"] and "finite" in r["error"]
