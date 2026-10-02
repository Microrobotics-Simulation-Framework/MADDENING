"""The compiled C wrapper, driven through ``ctypes`` against a live bridge:
what it used to coerce, drop or wait on for ever.

* ``fmi3SetBoolean`` on a Float32 parameter was stored as 1.0, and
  ``fmi3GetInt32`` on a Float32 output truncated 0.5 to 0, both with
  ``fmi3OK``: the wrapper carried every width as a double and the bridge
  could not tell the calls apart.  Each request now names its FMI type.
* ``fmi3SetFloat64`` on a ``<Clock>`` answered ``fmi3OK`` and the value was
  dropped.
* ``fmi3SetFMUState`` with a blob of exactly the frame limit passed the
  wrapper's check on the blob alone; the frame with its header did not fit,
  the bridge dropped the connection, and the instance was dead.
* A step size the bridge accepted made the next legal ``fmi3DoStep`` fail.
* An endpoint that accepted and never answered kept
  ``fmi3InstantiateCoSimulation`` blocked for ever.

Skipped without a C compiler.  The wrapper's own checks are unit-tested in
``tests/fmi/c/test_maddening_fmu.c``; this is the same behaviour end to end.
"""

from __future__ import annotations

import ctypes
import socket
import threading
import time

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.fmi.package import MODEL_IDENTIFIER, build_fmu_binary, find_c_compiler
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge
from tests.fmi.test_a_boolean_takes_true_false_or_exactly_0_1 import _Gate
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr

pytestmark = pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler")

OK, ERROR = 0, 3
FRAME_MAX = 64 * 1024 * 1024
VR = ctypes.c_uint32
LOG = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)


class _Wrapper:
    """The shared library with the prototypes this file calls, and a log."""

    def __init__(self, path):
        self.lib = lib = ctypes.CDLL(str(path))
        self.logs: list[str] = []
        self._log_cb = LOG(lambda env, st, cat, msg: self.logs.append(msg.decode(errors="replace")))
        lib.fmi3InstantiateCoSimulation.restype = ctypes.c_void_p
        lib.fmi3InstantiateCoSimulation.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool,
            ctypes.c_bool, ctypes.c_bool, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, LOG,
            ctypes.c_void_p]
        self.ctypes_of = {"Float64": ctypes.c_double, "Float32": ctypes.c_float,
                          "Int32": ctypes.c_int32, "Boolean": ctypes.c_bool}
        for name, ctype in self.ctypes_of.items():
            for op in ("Get", "Set"):
                f = getattr(lib, f"fmi3{op}{name}")
                f.restype = ctypes.c_int
                f.argtypes = [ctypes.c_void_p, ctypes.POINTER(VR), ctypes.c_size_t,
                              ctypes.POINTER(ctype), ctypes.c_size_t]
        lib.fmi3DoStep.restype = ctypes.c_int
        lib.fmi3DoStep.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double,
                                   ctypes.c_bool, *[ctypes.POINTER(ctypes.c_bool)] * 3,
                                   ctypes.POINTER(ctypes.c_double)]
        lib.fmi3EnterInitializationMode.restype = ctypes.c_int
        lib.fmi3EnterInitializationMode.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_double,
                                                    ctypes.c_double, ctypes.c_bool, ctypes.c_double]
        lib.fmi3DeserializeFMUState.restype = ctypes.c_int
        lib.fmi3DeserializeFMUState.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                                ctypes.POINTER(ctypes.c_void_p)]
        lib.fmi3SetFMUState.restype = ctypes.c_int
        lib.fmi3SetFMUState.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.fmi3FreeFMUState.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.fmi3FreeInstance.argtypes = [ctypes.c_void_p]

    def instantiate(self, token: str):
        return self.lib.fmi3InstantiateCoSimulation(
            b"i", token.encode(), b"", False, True, False, False, None, 0, None,
            self._log_cb, None)

    def call(self, op: str, fmi_type: str, inst, vrs, values):
        ctype = self.ctypes_of[fmi_type]
        arr = (ctype * len(values))(*values)
        status = getattr(self.lib, f"fmi3{op}{fmi_type}")(inst, (VR * len(vrs))(*vrs), len(vrs),
                                                          arr, len(values))
        return status, list(arr)

    def step(self, inst, t, h):
        flags = [ctypes.c_bool() for _ in range(3)]
        last = ctypes.c_double()
        status = self.lib.fmi3DoStep(inst, t, h, False, *[ctypes.byref(f) for f in flags],
                                     ctypes.byref(last))
        return status, last.value


@pytest.fixture(scope="module")
def wrapper(tmp_path_factory):
    return _Wrapper(build_fmu_binary(tmp_path_factory.mktemp("wrapper")))


@pytest.fixture(scope="module")
def gm():
    return _graph()


@pytest.fixture
def instance(wrapper, gm, monkeypatch):
    """One instance of the wrapper against a started bridge over the plant."""
    md, bridge = _bridge(gm)
    bridge.start()
    monkeypatch.setenv("MADDENING_FMU_ENDPOINT", bridge.endpoint)
    inst = wrapper.instantiate(md.instantiation_token)
    assert inst, wrapper.logs
    yield md, bridge, inst
    wrapper.lib.fmi3FreeInstance(inst)
    bridge.stop()


def test_a_boolean_setter_cannot_write_a_float32_parameter(wrapper, instance):
    md, bridge, inst = instance
    el = _vr(md, "ball.params.elasticity")
    status, _ = wrapper.call("Set", "Boolean", inst, [el], [True])
    assert status == ERROR
    assert "is Float32, so fmi3SetBoolean cannot address it" in wrapper.logs[-1]
    assert wrapper.call("Get", "Float32", inst, [el], [0.0]) == (OK, [pytest.approx(0.7)])
    assert wrapper.call("Set", "Float32", inst, [el], [0.5])[0] == OK
    assert wrapper.call("Get", "Float32", inst, [el], [0.0]) == (OK, [0.5])


def test_each_variable_is_read_only_through_the_getter_of_its_type(wrapper, instance):
    md, bridge, inst = instance
    pos, t = _vr(md, "spring.position"), _vr(md, "time")
    assert wrapper.call("Get", "Int32", inst, [pos], [0])[0] == ERROR       # used to be 0
    assert wrapper.call("Get", "Float64", inst, [pos], [0.0])[0] == ERROR
    assert wrapper.call("Get", "Boolean", inst, [pos], [False])[0] == ERROR
    assert wrapper.call("Get", "Float32", inst, [pos], [0.0]) == (OK, [0.5])
    assert wrapper.call("Get", "Float64", inst, [t], [1.0]) == (OK, [0.0])
    assert wrapper.call("Get", "Float32", inst, [t], [1.0])[0] == ERROR
    # a refusal is an error reply, not a broken connection
    assert wrapper.call("Get", "Float32", inst, [pos], [0.0])[0] == OK


def test_a_boolean_variable_is_set_and_read_through_the_boolean_functions(wrapper, monkeypatch):
    gm = GraphManager()
    gm.add_node(_Gate("gate", DT))
    gm.add_external_input("gate", "open", dtype=jnp.bool_)
    gm.compile()
    md = build_model_description(gm, model_name="G", model_identifier=MODEL_IDENTIFIER,
                                 include_evolving=True)
    sidecar = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token,
                                       step_fn=gm._compiled_step, initial_state=gm._state,
                                       params=gm.params))
    with FmuTcpBridge(sidecar, md, master_dt=DT) as bridge:
        monkeypatch.setenv("MADDENING_FMU_ENDPOINT", bridge.endpoint)
        inst = wrapper.instantiate(md.instantiation_token)
        assert inst, wrapper.logs
        try:
            gate = _vr(md, "gate.open")
            assert wrapper.call("Set", "Boolean", inst, [gate], [True])[0] == OK
            assert wrapper.call("Get", "Boolean", inst, [gate], [False]) == (OK, [True])
            assert wrapper.call("Set", "Float64", inst, [gate], [0.5])[0] == ERROR
            assert wrapper.call("Get", "Boolean", inst, [gate], [False]) == (OK, [True])
        finally:
            wrapper.lib.fmi3FreeInstance(inst)


def test_a_numeric_setter_cannot_write_a_clock(wrapper, gm, monkeypatch):
    md, bridge = _bridge(gm, multi_clock=True)
    with bridge:
        monkeypatch.setenv("MADDENING_FMU_ENDPOINT", bridge.endpoint)
        inst = wrapper.instantiate(md.instantiation_token)
        assert inst, wrapper.logs
        try:
            clock = next(v for v in md.variables if v.is_clock)
            assert wrapper.call("Set", "Float64", inst, [clock.value_reference], [0.0])[0] == ERROR
            assert "is Clock" in wrapper.logs[-1]
        finally:
            wrapper.lib.fmi3FreeInstance(inst)


def test_a_step_size_off_by_more_than_the_tolerance_is_refused_not_the_step_after(wrapper,
                                                                                    instance):
    md, bridge, inst = instance
    assert wrapper.lib.fmi3EnterInitializationMode(inst, False, 0.0, 0.0, False, 0.0) == OK
    assert wrapper.step(inst, 0.0, 3 * DT * (1 + 5e-7))[0] == ERROR
    assert "is not a whole multiple" in wrapper.logs[-1]
    h = 3 * DT + 0.9e-6 * DT                       # inside the one tolerance
    status, last = wrapper.step(inst, 0.0, h)
    assert status == OK and last == h
    assert wrapper.step(inst, last, h)[0] == OK    # the next legal point: accepted


def _fmu_state(wrapper, inst, blob: bytes):
    state = ctypes.c_void_p()
    assert wrapper.lib.fmi3DeserializeFMUState(inst, blob, len(blob), ctypes.byref(state)) == OK
    return state


def test_an_fmu_state_frame_over_the_limit_is_refused_and_the_instance_lives(wrapper, instance):
    md, bridge, inst = instance
    pos = _vr(md, "spring.position")
    header = b'{"op":"set_state","n":%d}'
    fits = FRAME_MAX - 4 - len(header % (FRAME_MAX - 40))
    assert 4 + len(header % fits) + fits == FRAME_MAX
    for size, refused_by_the_wrapper in ((FRAME_MAX, True), (fits + 1, True), (fits, False)):
        state = _fmu_state(wrapper, inst, b"\0" * size)
        try:
            n_logs = len(wrapper.logs)
            assert wrapper.lib.fmi3SetFMUState(inst, state) == ERROR
            said = wrapper.logs[n_logs:]
            if refused_by_the_wrapper:
                assert said == ["maddening_fmu: FMU state exceeds the frame limit"], said
            else:                                  # the bridge read it whole, and refused it
                assert len(said) == 1 and "not a valid archive" in said[0], said
        finally:
            wrapper.lib.fmi3FreeFMUState(inst, ctypes.byref(state))
        # the instance is alive and in step either way
        assert wrapper.call("Get", "Float32", inst, [pos], [0.0]) == (OK, [0.5])


class _Silent:
    """Accepts one connection, reads, and never answers."""

    def __init__(self):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.endpoint = "127.0.0.1:%d" % self.srv.getsockname()[1]
        self.hung_up = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        conn, _ = self.srv.accept()
        with conn:
            conn.settimeout(30)
            try:
                while conn.recv(4096):
                    pass
                self.hung_up.set()                 # the wrapper closed its end
            except OSError:
                pass

    def close(self):
        self.srv.close()


def test_an_endpoint_that_never_answers_fails_instantiation_by_the_deadline(wrapper, gm,
                                                                           monkeypatch):
    md, _ = _bridge(gm)
    silent = _Silent()
    try:
        monkeypatch.setenv("MADDENING_FMU_ENDPOINT", silent.endpoint)
        monkeypatch.setenv("MADDENING_FMU_TIMEOUT", "0.5")
        started = time.monotonic()
        assert not wrapper.instantiate(md.instantiation_token)
        elapsed = time.monotonic() - started
        assert 0.4 < elapsed < 5.0, elapsed
        assert "did not answer within the deadline" in wrapper.logs[-1]
        assert silent.hung_up.wait(5.0)
    finally:
        silent.close()


@pytest.mark.parametrize("bad", ["soon", "-1", "nan", "1e9"])
def test_a_deadline_that_is_not_a_number_of_seconds_fails_instantiation(wrapper, instance, bad,
                                                                       monkeypatch):
    md, bridge, _ = instance
    monkeypatch.setenv("MADDENING_FMU_TIMEOUT", bad)
    assert not wrapper.instantiate(md.instantiation_token)
    assert "MADDENING_FMU_TIMEOUT must be a number of seconds" in wrapper.logs[-1]
