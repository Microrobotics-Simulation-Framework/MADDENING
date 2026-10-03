"""The compiled C wrapper, through ``ctypes`` against a live bridge: the FMI
3.0 state machine, the wrapper's own clock, and numbers under an importer's
locale.

* ``fmi3DoStep`` after ``fmi3Terminate`` advanced the model and answered
  ``fmi3OK``.  FMI's Terminated state allows reading, the FMU-state
  functions and ``fmi3Reset``; the bridge now refuses ``doStep``, ``set``
  and ``initialize`` there until ``fmi3Reset``.
* The wrapper now holds FMI 3.0's co-simulation states itself: before,
  ``fmi3DoStep`` before ``fmi3EnterInitializationMode`` advanced the model,
  and ``fmi3ExitInitializationMode``, ``fmi3EnterStepMode`` and a
  zero-length ``fmi3Set*`` answered ``fmi3OK`` after ``fmi3Terminate``
  (claim FMU-025).
* ``fmi3GetClock`` / ``fmi3SetClock`` answered ``fmi3OK`` for any value
  reference.  FMI allows them only in Event Mode, which this FMU does not
  have, so both are ``fmi3Error``.
* ``lastSuccessfulTime`` of a refused ``fmi3DoStep`` after
  ``fmi3SetFMUState`` was the time before the restore.
* Under a locale whose decimal point is ``','``, every ``fmi3DoStep`` sent
  ``"dt":0,01`` and failed.  That half runs in a subprocess, which sets its
  own locale, so this process's locale is never touched.

Skipped without a C compiler.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys

import pytest

from maddening.fmi.package import build_fmu_binary, find_c_compiler
from tests.fmi.test_c_unit import _comma_locale
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr
from tests.fmi.test_c_wrapper_refuses_what_it_used_to_coerce import ERROR, OK, VR, _Wrapper

pytestmark = pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler")


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    return build_fmu_binary(tmp_path_factory.mktemp("wrapper"))


@pytest.fixture(scope="module")
def wrapper(binary):
    w = _Wrapper(binary)
    lib = w.lib
    lib.fmi3Terminate.restype = ctypes.c_int
    lib.fmi3Terminate.argtypes = [ctypes.c_void_p]
    lib.fmi3Reset.restype = ctypes.c_int
    lib.fmi3Reset.argtypes = [ctypes.c_void_p]
    lib.fmi3GetFMUState.restype = ctypes.c_int
    lib.fmi3GetFMUState.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    for name in ("fmi3EnterConfigurationMode", "fmi3ExitConfigurationMode"):
        getattr(lib, name).restype = ctypes.c_int
        getattr(lib, name).argtypes = [ctypes.c_void_p]
    lib.fmi3SetTime.restype = ctypes.c_int
    lib.fmi3SetTime.argtypes = [ctypes.c_void_p, ctypes.c_double]
    for name in ("fmi3GetClock", "fmi3SetClock"):
        f = getattr(lib, name)
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_void_p, ctypes.POINTER(VR), ctypes.c_size_t,
                      ctypes.POINTER(ctypes.c_bool)]
    return w


@pytest.fixture(scope="module")
def gm():
    return _graph()


@pytest.fixture
def instance(wrapper, gm, monkeypatch):
    md, bridge = _bridge(gm)
    bridge.start()
    monkeypatch.setenv("MADDENING_FMU_ENDPOINT", bridge.endpoint)
    inst = wrapper.instantiate(md.instantiation_token)
    assert inst, wrapper.logs
    yield md, bridge, inst
    wrapper.lib.fmi3FreeInstance(inst)
    bridge.stop()


def test_a_terminated_instance_refuses_to_step_set_and_initialize_until_reset(wrapper, instance):
    md, bridge, inst = instance
    lib = wrapper.lib
    pos, anchor = _vr(md, "spring.position"), _vr(md, "spring.anchor_position")
    wrapper.to_step_mode(inst)
    assert wrapper.step(inst, 0.0, DT)[0] == OK
    before = wrapper.call("Get", "Float32", inst, [pos], [0.0])
    assert lib.fmi3Terminate(inst) == OK
    status, last = wrapper.step(inst, DT, DT)
    assert status == ERROR and last == pytest.approx(DT)
    assert "not allowed in the Terminated state" in wrapper.logs[-1]
    assert wrapper.call("Set", "Float32", inst, [anchor], [0.3])[0] == ERROR
    assert lib.fmi3EnterInitializationMode(inst, False, 0.0, 0.0, False, 0.0) == ERROR
    # the three that answered fmi3OK here: a zero-length set, and the two
    # mode changes that have nowhere to go from Terminated
    assert wrapper.call("Set", "Float32", inst, [], [])[0] == ERROR
    assert lib.fmi3ExitInitializationMode(inst) == ERROR
    assert lib.fmi3EnterStepMode(inst) == ERROR
    assert lib.fmi3Terminate(inst) == ERROR
    # and the bridge, which keeps its own check, still says Terminated
    reply = bridge.handle({"op": "step", "t": DT, "dt": DT})
    assert reply["ok"] is False and "has been terminated" in reply["error"]
    # reading, and saving the FMU state, are what Terminated is for
    assert wrapper.call("Get", "Float32", inst, [pos], [0.0]) == before
    state = ctypes.c_void_p()
    assert lib.fmi3GetFMUState(inst, ctypes.byref(state)) == OK
    lib.fmi3FreeFMUState(inst, ctypes.byref(state))
    assert wrapper.call("Get", "Float32", inst, [anchor], [1.0]) == (OK, [0.0])
    # fmi3Reset starts the instance again, in the Instantiated state
    assert lib.fmi3Reset(inst) == OK
    assert wrapper.call("Set", "Float32", inst, [anchor], [0.3])[0] == OK
    wrapper.to_step_mode(inst)
    assert wrapper.step(inst, 0.0, DT)[0] == OK


def test_no_step_before_initialization_and_no_mode_change_out_of_turn(wrapper, instance):
    """FMI 3.0's co-simulation states, held by the wrapper: ``fmi3DoStep``
    used to advance the model straight after instantiation, before any
    ``fmi3EnterInitializationMode`` (and report time 0.01, position
    0.5015, ``fmi3OK``)."""
    md, bridge, inst = instance
    lib = wrapper.lib
    served = bridge.requests_served
    status, last = wrapper.step(inst, 0.0, DT)
    assert status == ERROR and last == 0.0
    assert "fmi3DoStep is not allowed in the Instantiated state" in wrapper.logs[-1]
    assert wrapper.call("Get", "Float64", inst, [_vr(md, "time")], [9.0]) == (ERROR, [9.0])
    assert lib.fmi3ExitInitializationMode(inst) == ERROR
    assert lib.fmi3EnterStepMode(inst) == ERROR
    assert lib.fmi3Terminate(inst) == ERROR
    assert bridge.requests_served == served                # all answered locally
    # nothing moved: initialize and read the start
    assert lib.fmi3EnterInitializationMode(inst, False, 0.0, 0.0, False, 0.0) == OK
    assert wrapper.call("Get", "Float32", inst, [_vr(md, "spring.position")], [0.0]) == (OK, [0.5])
    assert wrapper.step(inst, 0.0, DT)[0] == ERROR          # Initialization Mode: not yet
    assert lib.fmi3EnterInitializationMode(inst, False, 0.0, 0.0, False, 0.0) == ERROR
    assert lib.fmi3ExitInitializationMode(inst) == OK
    assert lib.fmi3EnterStepMode(inst) == ERROR             # Event Mode's, never this FMU's
    assert "Event Mode" in wrapper.logs[-1]
    assert lib.fmi3ExitInitializationMode(inst) == ERROR
    assert wrapper.step(inst, 0.0, DT)[0] == OK


def test_what_this_fmu_never_has_is_refused_in_every_state(wrapper, instance):
    """No structural parameters, so no Configuration Mode; a co-simulation
    instance, so no ``fmi3SetTime``.  All three answered ``fmi3OK`` (and
    ``fmi3SetTime`` moved the wrapper's clock)."""
    md, bridge, inst = instance
    lib = wrapper.lib
    for state in ("Instantiated", "Step Mode"):
        assert lib.fmi3EnterConfigurationMode(inst) == ERROR, state
        assert "no structural parameters" in wrapper.logs[-1]
        assert lib.fmi3ExitConfigurationMode(inst) == ERROR, state
        assert lib.fmi3SetTime(inst, 3.0) == ERROR, state
        assert "model-exchange" in wrapper.logs[-1]
        if state == "Instantiated":
            wrapper.to_step_mode(inst)
    status, last = wrapper.step(inst, 1.0, DT)              # a refused step reports the clock
    assert status == ERROR and last == 0.0


def test_the_wrapper_provides_no_directional_derivative(wrapper, instance):
    """FMU-017: the sidecar's Python API computes directional derivatives with ``jax.jvp``;
    the FMU binary does not, and says so: ``fmi3GetDirectionalDerivative`` is ``fmi3Error``
    in Step Mode, with the sensitivity untouched and nothing asked of the bridge."""
    md, bridge, inst = instance
    f = wrapper.lib.fmi3GetDirectionalDerivative
    f.restype = ctypes.c_int
    f.argtypes = [ctypes.c_void_p, ctypes.POINTER(VR), ctypes.c_size_t, ctypes.POINTER(VR),
                  ctypes.c_size_t, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                  ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    wrapper.to_step_mode(inst)
    served = bridge.requests_served
    unknowns = (VR * 1)(_vr(md, "spring.position"))
    knowns = (VR * 1)(_vr(md, "spring.params.stiffness"))
    seed, sensitivity = (ctypes.c_double * 1)(1.0), (ctypes.c_double * 1)(42.0)
    assert f(inst, unknowns, 1, knowns, 1, seed, 1, sensitivity, 1) == ERROR
    assert sensitivity[0] == 42.0
    assert "Python sidecar API" in wrapper.logs[-1]
    assert bridge.requests_served == served


def test_the_clock_functions_are_refused_for_any_value_reference(wrapper, instance):
    md, bridge, inst = instance
    served = bridge.requests_served
    vals = (ctypes.c_bool * 2)(True, True)
    refs = (VR * 2)(_vr(md, "spring.position"), 99_999)
    assert wrapper.lib.fmi3GetClock(inst, refs, 2, vals) == ERROR
    assert list(vals) == [True, True]                    # nothing written
    assert "Event Mode" in wrapper.logs[-1]
    assert wrapper.lib.fmi3SetClock(inst, refs, 2, vals) == ERROR
    assert bridge.requests_served == served              # answered locally


def test_a_refused_step_after_a_restore_reports_the_restored_time(wrapper, instance):
    md, bridge, inst = instance
    lib = wrapper.lib
    wrapper.to_step_mode(inst)
    t = 0.0
    for _ in range(5):
        assert wrapper.step(inst, t, DT)[0] == OK
        t += DT
    snap = ctypes.c_void_p()
    assert lib.fmi3GetFMUState(inst, ctypes.byref(snap)) == OK
    for _ in range(5):
        assert wrapper.step(inst, t, DT)[0] == OK
        t += DT
    assert lib.fmi3SetFMUState(inst, snap) == OK
    status, last = wrapper.step(inst, 5 * DT, 1.5 * DT)   # not a whole number of steps
    assert status == ERROR
    assert last == pytest.approx(5 * DT)                  # it was 0.1, the time before
    assert wrapper.step(inst, last, DT)[0] == OK          # and the FMU is there
    lib.fmi3FreeFMUState(inst, ctypes.byref(snap))


_LOCALE_DRIVER = r"""
import ctypes, ctypes.util, json, locale, os, sys
so, token = sys.argv[1], sys.argv[2]
locale.setlocale(locale.LC_ALL, "")          # the comma locale from the environment
lib = ctypes.CDLL(so)
LOG = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)
logs = []
cb = LOG(lambda e, s, c, m: logs.append(m.decode(errors="replace")))
lib.fmi3InstantiateCoSimulation.restype = ctypes.c_void_p
lib.fmi3InstantiateCoSimulation.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.c_void_p, LOG, ctypes.c_void_p]
lib.fmi3EnterInitializationMode.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_double,
    ctypes.c_double, ctypes.c_bool, ctypes.c_double]
lib.fmi3ExitInitializationMode.argtypes = [ctypes.c_void_p]
lib.fmi3DoStep.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_bool,
    *[ctypes.POINTER(ctypes.c_bool)] * 3, ctypes.POINTER(ctypes.c_double)]
lib.fmi3GetFloat64.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.fmi3FreeInstance.argtypes = [ctypes.c_void_p]
inst = lib.fmi3InstantiateCoSimulation(b"i", token.encode(), b"", False, True, False, False,
                                       None, 0, None, cb, None)
out = {"decimal_point": locale.localeconv()["decimal_point"], "instantiated": bool(inst)}
if inst:
    out["initialize"] = lib.fmi3EnterInitializationMode(inst, False, 0.0, 0.5, False, 0.0)
    out["exit"] = lib.fmi3ExitInitializationMode(inst)
    flags = [ctypes.c_bool() for _ in range(3)]
    last = ctypes.c_double()
    steps = []
    t = 0.5
    for _ in range(2):
        steps.append(lib.fmi3DoStep(inst, t, 0.01, False, *[ctypes.byref(f) for f in flags],
                                    ctypes.byref(last)))
        t = last.value
    out["steps"], out["last"] = steps, last.value
    v = (ctypes.c_double * 1)()
    out["get_time"] = lib.fmi3GetFloat64(inst, (ctypes.c_uint32 * 1)(int(sys.argv[3])), 1, v, 1)
    out["time"] = v[0]
    lib.fmi3FreeInstance(inst)
out["logs"] = logs
print("RESULT " + json.dumps(out))
"""


def test_an_importer_under_a_comma_decimal_locale_steps_the_fmu(binary, gm, tmp_path):
    found = _comma_locale(tmp_path)
    if found is None:
        pytest.skip("no locale with a ',' decimal point is installed and localedef "
                    "could not build one (glibc's locale sources are missing)")
    _, env = found
    md, bridge = _bridge(gm)
    script = tmp_path / "drive.py"
    script.write_text(_LOCALE_DRIVER)
    with bridge:
        proc = subprocess.run(
            [sys.executable, str(script), str(binary), md.instantiation_token,
             str(_vr(md, "time"))],
            capture_output=True, text=True, timeout=120,
            env={**{k: v for k, v in os.environ.items() if not k.startswith("LC_")},
                 **env, "MADDENING_FMU_ENDPOINT": bridge.endpoint})
    assert proc.returncode == 0, proc.stderr[-3000:]
    got = json.loads(next(line for line in proc.stdout.splitlines()
                          if line.startswith("RESULT "))[7:])
    assert got["decimal_point"] == ",", got        # the locale really took
    assert got["instantiated"], got
    assert got["initialize"] == got["exit"] == OK and got["steps"] == [OK, OK], got
    assert got["last"] == pytest.approx(0.52) and got["time"] == pytest.approx(0.52), got
    assert not got["logs"], got
