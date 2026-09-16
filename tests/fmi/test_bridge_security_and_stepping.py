"""``FmuTcpBridge`` at its trust boundary and in its stepping contract.

* ``set_state`` never unpickles importer bytes: a pickle payload is
  refused without executing anything, and the arrays-only ``npz`` blob is
  validated against the schema token and every shape before a write;
* a communication step that is not a whole multiple of the master
  timestep is refused (the FMU advertises a fixed step, no event mode);
* ``set`` is atomic across parameters and inputs and refuses non-finite
  inputs;
* a second FMU instance on one bridge gets a clear error, not a hang.

Originally written from the independent audit of 2026-09-16 (round 3; report and
reproducers under ``benchmarks/results/audit3/``).
"""

import base64
import io
import os
import pickle
import socket
import zipfile

import jax.numpy as jnp
import numpy as np
import pytest

from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr
from maddening.fmi.tcp_bridge import recv_message, send_message


class _Evil:
    """pickle payload whose deserialisation would create a file."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


def test_set_state_never_unpickles(tmp_path):
    gm = _graph()
    md, bridge = _bridge(gm)
    marker = tmp_path / "pwned"
    blob = base64.b64encode(pickle.dumps(_Evil(str(marker)))).decode("ascii")
    r = bridge.handle({"op": "set_state", "state": blob})
    assert not r["ok"] and "archive" in r["error"]
    assert not marker.exists()
    # a valid-looking npz that carries a pickled object is refused too
    buf = io.BytesIO()
    np.savez(buf, _token=np.array(md.instantiation_token), evil=np.array(_Evil(str(marker)), dtype=object))
    r = bridge.handle({"op": "set_state", "state": base64.b64encode(buf.getvalue()).decode("ascii")})
    assert not r["ok"] and not marker.exists()


def test_state_blob_is_arrays_only_and_validated():
    gm = _graph()
    md, bridge = _bridge(gm)
    pos = _vr(md, "spring.position")
    bridge.handle({"op": "step", "t": 0.0, "dt": DT})
    snap = bridge.handle({"op": "get_state"})["state"]
    raw = base64.b64decode(snap)
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        assert str(data["_token"]) == md.instantiation_token
        assert "s/spring/position" in data.files and "p/nodes/spring/stiffness" in data.files
    before = bridge.handle({"op": "get", "vr": [pos]})["values"]
    bridge.handle({"op": "step", "t": DT, "dt": 3 * DT})
    assert bridge.handle({"op": "set_state", "state": snap})["ok"]
    assert bridge.handle({"op": "get", "vr": [pos, _vr(md, "time")]})["values"] == before + [DT]
    # wrong token, wrong shape, missing field: refused, state untouched
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    for mutate, needle in (
        (lambda a: a.__setitem__("_token", np.array("other")), "token"),
        (lambda a: a.__setitem__("s/spring/position", np.zeros(3, np.float32)), "shape"),
        (lambda a: a.pop("s/ball/velocity"), "missing"),
        (lambda a: a.__setitem__("i/spring/nope", np.zeros(())), "unknown input"),
    ):
        bad = dict(arrays)
        mutate(bad)
        buf = io.BytesIO()
        np.savez(buf, **bad)
        r = bridge.handle({"op": "set_state", "state": base64.b64encode(buf.getvalue()).decode("ascii")})
        assert not r["ok"] and needle in r["error"], r
        assert bridge.handle({"op": "get", "vr": [pos]})["values"] == before


def test_step_must_be_a_whole_multiple_of_master_dt():
    gm = _graph()
    md, bridge = _bridge(gm)
    pos = _vr(md, "spring.position")
    r = bridge.handle({"op": "step", "t": 0.0, "dt": 0.4 * DT})
    assert not r["ok"] and "multiple" in r["error"]
    r = bridge.handle({"op": "step", "t": 0.0, "dt": 1.5 * DT})
    assert not r["ok"]
    for bad in (0.0, -DT, float("nan")):
        assert not bridge.handle({"op": "step", "t": 0.0, "dt": bad})["ok"]
    assert bridge.handle({"op": "get", "vr": [_vr(md, "time")]})["values"] == [0.0]
    r = bridge.handle({"op": "step", "t": 0.0, "dt": 3 * DT * (1 + 1e-9)})   # within tolerance
    assert r["ok"] and r["t"] == pytest.approx(3 * DT)
    ref = _graph().run_scan(3)
    assert bridge.handle({"op": "get", "vr": [pos]})["values"][0] == pytest.approx(
        float(ref["spring"]["position"]), rel=1e-6)


def test_set_is_atomic_and_refuses_non_finite_inputs():
    gm = _graph()
    md, bridge = _bridge(gm)
    anchor, el = _vr(md, "spring.anchor_position"), _vr(md, "ball.params.elasticity")
    r = bridge.handle({"op": "set", "vr": [anchor, el], "values": [0.9, 1.5]})
    assert not r["ok"] and "above bound" in r["error"]
    # the input was not committed: still the advertised zero start value
    assert float(bridge._inputs["spring"]["anchor_position"]) == 0.0
    r = bridge.handle({"op": "set", "vr": [anchor], "values": [float("inf")]})
    assert not r["ok"] and "finite" in r["error"]
    assert bridge.handle({"op": "set", "vr": [anchor, el], "values": [0.9, 0.5]})["ok"]
    assert float(bridge._inputs["spring"]["anchor_position"]) == pytest.approx(0.9)


def test_second_instance_on_one_bridge_is_refused_not_blocked():
    gm = _graph()
    md, bridge = _bridge(gm)
    with bridge:
        host, port = bridge.endpoint.split(":")
        with socket.create_connection((host, int(port)), timeout=5) as first:
            send_message(first, {"op": "hello"})
            assert recv_message(first)["ok"]
            with socket.create_connection((host, int(port)), timeout=5) as second:
                send_message(second, {"op": "hello"})
                r = recv_message(second)
                assert not r["ok"] and "one FmuTcpBridge per instance" in r["error"]
            send_message(first, {"op": "step", "t": 0.0, "dt": DT})
            assert recv_message(first)["ok"]
        # after the first disconnects, a new instance may connect
        with socket.create_connection((host, int(port)), timeout=5) as third:
            send_message(third, {"op": "hello"})
            assert recv_message(third)["ok"]


def test_model_description_advertises_fixed_step():
    gm = _graph()
    md, _ = _bridge(gm)
    assert 'canHandleVariableCommunicationStepSize="false"' in md.to_xml()
    assert 'hasEventMode="false"' in md.to_xml()
