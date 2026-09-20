"""``FmuTcpBridge`` at its trust boundary and in its stepping contract.

* ``set_state`` never unpickles importer bytes: a pickle payload is
  refused without executing anything, and the arrays-only ``npz`` blob is
  validated against the schema token and every shape before a write;
* a communication step that is not a whole multiple of the master
  timestep is refused (the FMU advertises a fixed step, no event mode);
* ``set`` is atomic across parameters and inputs and refuses non-finite
  inputs;
* a second FMU instance on one bridge gets a clear error, not a hang,
  while a *reconnecting* instance waits out the departing one's
  hand-over instead of being refused a slot nobody holds.

Originally written from the independent audit of 2026-09-16 (round 3; report and
reproducers under ``benchmarks/results/audit3/``).
"""

import base64
import io
import os
import pickle
import socket
import time
import zipfile

import jax.numpy as jnp
import numpy as np
import pytest

from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr
from maddening.fmi.tcp_bridge import recv_message, send_message, values_of


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
    # The socket budget is well clear of ``_HANDOVER_GRACE``: the refusal
    # below is deliberately made to wait out a possible hand-over, so a
    # timeout of a couple of seconds would be measuring the grace.
    gm = _graph()
    md, bridge = _bridge(gm)
    with bridge:
        host, port = bridge.endpoint.split(":")
        with socket.create_connection((host, int(port)), timeout=30) as first:
            send_message(first, {"op": "hello"})
            assert recv_message(first)["ok"]
            with socket.create_connection((host, int(port)), timeout=30) as second:
                send_message(second, {"op": "hello"})
                r = recv_message(second)
                assert not r["ok"] and "one FmuTcpBridge per instance" in r["error"]
            send_message(first, {"op": "step", "t": 0.0, "dt": DT})
            assert recv_message(first)["ok"]
        # after the first disconnects, a new instance may connect
        with socket.create_connection((host, int(port)), timeout=30) as third:
            send_message(third, {"op": "hello"})
            assert recv_message(third)["ok"]


class _SlowHandover:
    """The bridge's instance lock with its hand-over gap made deterministic.

    The real gap is the scheduler's: the departing connection's worker
    releases the slot in a ``finally`` that runs after its socket has
    reached EOF, so a reconnect can arrive while the slot is still held
    by a connection that no longer exists.  Racing it is not a test --
    it passed 500 times out of 500 on an idle box and failed 21-28% of
    the time on one CPU under load, which is how it reached CI as a
    flake.  Delaying the release by a fixed amount is the same situation
    with the timing pinned.
    """

    def __init__(self, inner, delay):
        self._inner, self._delay = inner, delay

    def acquire(self, blocking=True, timeout=-1):
        return self._inner.acquire(blocking, timeout)

    def release(self):
        time.sleep(self._delay)
        self._inner.release()


def test_a_reconnecting_instance_waits_out_the_previous_ones_hand_over():
    """An importer that frees an instance and immediately makes another is
    not refused a slot nobody holds.

    ``fmi3FreeInstance`` followed by ``fmi3InstantiateCoSimulation`` is
    what every co-simulation master does between runs, and the two are
    one round trip apart.  Before ``_HANDOVER_GRACE`` the second
    ``hello`` was answered "bridge already serves an FMU instance"
    whenever the old worker had not been scheduled yet -- a refusal with
    no second instance anywhere, and nothing for the importer to do
    about it but retry.
    """
    gm = _graph()
    md, bridge = _bridge(gm)
    # Far longer than a scheduler slice, far shorter than the grace.
    bridge._busy = _SlowHandover(bridge._busy, 0.3)
    with bridge:
        host, port = bridge.endpoint.split(":")
        with socket.create_connection((host, int(port)), timeout=30) as first:
            send_message(first, {"op": "hello"})
            assert recv_message(first)["ok"]
        # ``first`` has hung up; its worker is still inside the release.
        with socket.create_connection((host, int(port)), timeout=30) as second:
            send_message(second, {"op": "hello"})
            r = recv_message(second)
            assert r["ok"], f"a reconnect was refused a slot nobody holds: {r}"


def test_model_description_advertises_fixed_step():
    gm = _graph()
    md, _ = _bridge(gm)
    assert 'canHandleVariableCommunicationStepSize="false"' in md.to_xml()
    assert 'hasEventMode="false"' in md.to_xml()


# ------------------------------------------------ set_state: the archive directory

def _zip_bomb(member: str, declared: int) -> bytes:
    """A deflated archive whose one member declares ``declared`` bytes of
    zeros: a few hundred to one at deflate level 1 (kept fast here), about
    1000:1 at the default level, at which a 64 MiB frame declares some
    60 GiB."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        with zf.open(member, "w") as f:
            chunk = b"\0" * (1 << 20)
            for _ in range(declared >> 20):
                f.write(chunk)
    return buf.getvalue()


@pytest.fixture
def no_np_load(monkeypatch):
    """``np.load`` must not run on an archive the directory check has not
    passed: decompression happens there, so a refusal after it is too late."""
    def refuse(*args, **kwargs):
        raise AssertionError("np.load ran on an unchecked FMU-state archive")
    monkeypatch.setattr(np, "load", refuse)


def test_set_state_refuses_a_zip_bomb_before_decompressing_on_both_paths(no_np_load):
    """A member WITHOUT the ``.npy`` suffix used to escape the per-member
    size cap (numpy lists it, ``getinfo(k + ".npy")`` misses it) and was
    then decompressed in full (measured: 1 MiB on the wire declaring 1 GiB,
    +4.7 GB RSS).  Now every member of the directory, whatever its name,
    is checked before any byte is inflated, on the base64 (JSON) and on
    the raw (binary) path."""
    from maddening.fmi.tcp_bridge import send_binary
    from tests.fmi.test_binary_frames import _connect
    gm = _graph()
    md, bridge = _bridge(gm)
    pos = _vr(md, "spring.position")
    bomb = _zip_bomb("_token", 256 << 20)                  # declares 256 MiB
    assert len(bomb) < 2 * (1 << 20)
    before = bridge.handle({"op": "get", "vr": [pos]})["values"]
    t0 = time.perf_counter()
    r = bridge.handle({"op": "set_state", "state": base64.b64encode(bomb).decode("ascii")})
    assert not r["ok"] and "unknown member '_token'" in r["error"], r
    assert time.perf_counter() - t0 < 1.0
    # the same member with the suffix: refused by its declared size
    r = bridge.handle({"op": "set_state", "state": _zip_bomb("_token.npy", 16 << 20)})
    assert not r["ok"] and "16777216 bytes, more than the" in r["error"], r
    # a real state field, declared far larger than the live array
    r = bridge.handle({"op": "set_state", "state": _zip_bomb("s/ball/position.npy", 64 << 20)})
    assert not r["ok"] and "'s/ball/position' is 67108864 bytes" in r["error"], r
    # a member the model does not have at all, however small
    r = bridge.handle({"op": "set_state", "state": _zip_bomb("i/spring/nope.npy", 1 << 20)})
    assert not r["ok"] and "unknown input 'i/spring/nope.npy'" in r["error"], r
    r = bridge.handle({"op": "set_state", "state": _zip_bomb("evil.npy", 0)})
    assert not r["ok"] and "unknown member 'evil.npy'" in r["error"], r
    # binary path: the raw npz bytes go through the very same check
    with bridge:
        conn, _ = _connect(bridge)
        with conn:
            t0 = time.perf_counter()
            send_binary(conn, {"op": "set_state", "n": len(bomb)}, bomb)
            r = recv_message(conn)
            assert not r["ok"] and "unknown member '_token'" in r["error"], r
            assert time.perf_counter() - t0 < 1.0
            send_message(conn, {"op": "get", "vr": [pos]})
            assert values_of(recv_message(conn)).tolist() == before
    assert bridge.handle({"op": "get", "vr": [pos]})["values"] == before


def test_set_state_total_declared_size_is_capped(no_np_load):
    """Members that each fit their own cap must not add up to more than
    the model can hold: duplicate names in the directory are the way to
    multiply an archive's declared size without exceeding any one cap."""
    import warnings
    gm = _graph()
    md, bridge = _bridge(gm)
    caps = bridge._member_caps()
    key = "s/ball/position"
    body = b"\0" * caps[key]                              # exactly at its own cap
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)        # zipfile: duplicate name
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for _ in range(sum(caps.values()) // caps[key] + 2):
                zf.writestr(key + ".npy", body)
    r = bridge.handle({"op": "set_state", "state": buf.getvalue()})
    assert not r["ok"] and "in total, more than the" in r["error"], r


def test_set_state_round_trip_still_passes_the_directory_check():
    """The bridge's own snapshot carries exactly the expected members and
    sizes, so the new check is transparent for a legitimate restore."""
    gm = _graph()
    md, bridge = _bridge(gm)
    pos = _vr(md, "spring.position")
    snap = bridge.handle({"op": "get_state"})["state"]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(snap))) as zf:
        names = {i.filename[:-4] for i in zf.infolist()}
    assert names <= set(bridge._member_caps()) and "_token" in names
    before = bridge.handle({"op": "get", "vr": [pos]})["values"]
    bridge.handle({"op": "step", "t": 0.0, "dt": 3 * DT})
    assert bridge.handle({"op": "set_state", "state": snap}) == {"ok": True}
    assert bridge.handle({"op": "get", "vr": [pos]})["values"] == before


# ---------------------------------------------------- set: the variable's dtype

def test_set_refuses_values_that_overflow_the_variable_dtype():
    """The finiteness check runs on the value in the variable's dtype, not
    on the float64 wire value: a float32 input set to 1e308 used to be
    accepted and read back as inf.  Inputs and parameters alike, and the
    largest float32 still goes through."""
    gm = _graph()
    md, bridge = _bridge(gm)
    anchor, k = _vr(md, "spring.anchor_position"), _vr(md, "spring.params.stiffness")
    assert md.variables[[v.value_reference for v in md.variables].index(anchor)].dtype == "float32"
    for bad in (1e308, -1e308, 3.5e38):
        r = bridge.handle({"op": "set", "vr": [anchor], "values": [bad]})
        assert not r["ok"] and "does not fit its type float32" in r["error"], (bad, r)
        assert bridge.handle({"op": "get", "vr": [anchor]})["values"] == [0.0]
    r = bridge.handle({"op": "set", "vr": [k], "values": [1e308]})
    assert not r["ok"] and "does not fit its type float32" in r["error"], r
    assert bridge.handle({"op": "set", "vr": [anchor], "values": [3.0e38]})["ok"]
    got = bridge.handle({"op": "get", "vr": [anchor]})["values"][0]
    assert np.isfinite(got) and got == pytest.approx(3.0e38, rel=1e-6)
    # an atomic set: the bad value refuses the whole request
    r = bridge.handle({"op": "set", "vr": [anchor, k], "values": [0.1, 1e308]})
    assert not r["ok"]
    assert bridge.handle({"op": "get", "vr": [anchor]})["values"][0] == pytest.approx(3.0e38, rel=1e-6)
