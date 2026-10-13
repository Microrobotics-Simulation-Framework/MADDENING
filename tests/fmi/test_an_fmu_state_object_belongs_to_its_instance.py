"""An ``fmi3FMUState`` is reused when the importer hands it back, and freed
with its instance: the compiled wrapper, through ``ctypes``, against a live
bridge.

FMI 3.0.1, "Getting and Setting the Complete FMU State": on entry to
``fmi3GetFMUState`` a non-NULL ``*FMUState`` "points to a previously
returned FMUState that is no longer needed and can be overwritten", and the
function "typically reuses the memory of this FMUState in this case and
returns the same pointer".  An importer that rolls back keeps one variable
and calls ``fmi3GetFMUState(&state)`` before every step.  The wrapper
allocated a new object over it on every call and returned another pointer,
so nothing could free the old one: one whole state blob leaked per call
(2.8 kB for one spring, 82 kB for a 20000-cell rod).  And
``fmi3FreeInstance`` "frees all the allocated memory ... allocated by the
functions of the FMU interface", which includes a state the importer never
freed.

The allocation counts themselves are asserted in the C unit test
(``tests/fmi/c/test_maddening_fmu.c``, under the sanitizers and valgrind
too); here the same rules are checked end to end, on states a real bridge
produced, over both wire forms.

Skipped without a C compiler.
"""

from __future__ import annotations

import ctypes

import pytest

from maddening.fmi.package import build_fmu_binary, find_c_compiler
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr
from tests.fmi.test_c_wrapper_refuses_what_it_used_to_coerce import ERROR, OK, _Wrapper

pytestmark = pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler")


@pytest.fixture(scope="module")
def wrapper(tmp_path_factory):
    w = _Wrapper(build_fmu_binary(tmp_path_factory.mktemp("wrapper")))
    lib = w.lib
    lib.fmi3GetFMUState.restype = ctypes.c_int
    lib.fmi3GetFMUState.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.fmi3FreeFMUState.restype = ctypes.c_int
    lib.fmi3SerializedFMUStateSize.restype = ctypes.c_int
    lib.fmi3SerializedFMUStateSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_size_t)]
    lib.fmi3SerializeFMUState.restype = ctypes.c_int
    lib.fmi3SerializeFMUState.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p,
                                          ctypes.c_size_t]
    lib.fmi3Reset.restype = ctypes.c_int
    lib.fmi3Reset.argtypes = [ctypes.c_void_p]
    lib.fmi3Terminate.restype = ctypes.c_int
    lib.fmi3Terminate.argtypes = [ctypes.c_void_p]
    return w


@pytest.fixture(scope="module")
def gm():
    return _graph()


@pytest.fixture(params=["binary frames", "JSON only"])
def served(request, wrapper, gm, monkeypatch):
    """The plant behind a started bridge, on each wire form: protocol 2
    with binary frames (a state is raw npz bytes) and a bridge that only
    speaks JSON (a state is base64 text)."""
    md, bridge = _bridge(gm)
    if request.param == "JSON only":
        dispatch = bridge._dispatch                                    # noqa: SLF001

        def json_only(req):
            reply = dispatch(req)
            if req.get("op") == "hello" and reply.get("ok"):
                reply = {**reply, "binary": False}
            return reply

        monkeypatch.setattr(bridge, "_dispatch", json_only)
    bridge.start()
    monkeypatch.setenv("MADDENING_FMU_ENDPOINT", bridge.endpoint)
    yield md, bridge
    bridge.stop()


def _instance(wrapper, md):
    inst = wrapper.instantiate(md.instantiation_token)
    assert inst, wrapper.logs
    wrapper.to_step_mode(inst)
    return inst


def _blob(wrapper, inst, state) -> bytes:
    size = ctypes.c_size_t()
    assert wrapper.lib.fmi3SerializedFMUStateSize(inst, state, ctypes.byref(size)) == OK
    buf = ctypes.create_string_buffer(size.value)
    assert wrapper.lib.fmi3SerializeFMUState(inst, state, buf, size.value) == OK
    return buf.raw


def test_get_fmu_state_into_a_held_variable_returns_the_same_object(wrapper, served):
    """The rollback pattern: one variable, a get before every step.  Every
    call returns the object it was handed, now holding the current state --
    which restores to exactly where it was taken."""
    md, bridge = served
    lib = wrapper.lib
    inst = _instance(wrapper, md)
    pos, vel = _vr(md, "spring.position"), _vr(md, "spring.velocity")
    try:
        state = ctypes.c_void_p()                       # NULL: the first call allocates
        assert lib.fmi3GetFMUState(inst, ctypes.byref(state)) == OK
        first = state.value
        assert first
        t, taken = 0.0, []
        for k in range(12):
            assert lib.fmi3GetFMUState(inst, ctypes.byref(state)) == OK
            assert state.value == first, f"call {k} returned another object"
            taken.append((t, wrapper.call("Get", "Float32", inst, [pos, vel], [0.0, 0.0])[1],
                          _blob(wrapper, inst, state)))
            status, t = wrapper.step(inst, t, DT)
            assert status == OK
        # the object holds the last state taken (before the last step), not
        # the first: a restore goes back one step
        here = wrapper.call("Get", "Float32", inst, [pos, vel], [0.0, 0.0])[1]
        assert lib.fmi3SetFMUState(inst, state) == OK
        assert wrapper.call("Get", "Float32", inst, [pos, vel], [0.0, 0.0])[1] == taken[-1][1]
        assert taken[-1][1] != here != taken[0][1]
        assert len({blob for _, _, blob in taken}) == len(taken)     # twelve different states
        # and a step from there is the step that was taken before
        status, _ = wrapper.step(inst, taken[-1][0], DT)
        assert status == OK
        assert wrapper.call("Get", "Float32", inst, [pos, vel], [0.0, 0.0])[1] == here
        assert lib.fmi3FreeFMUState(inst, ctypes.byref(state)) == OK and state.value is None
    finally:
        lib.fmi3FreeInstance(inst)


def test_a_failed_get_leaves_the_held_state_as_it_was(wrapper, served, monkeypatch):
    """``fmi3GetFMUState`` that fails -- here the bridge refuses the request
    -- returns ``fmi3Error`` with the variable and the state it holds
    untouched: the importer can still restore what it saved."""
    md, bridge = served
    lib = wrapper.lib
    inst = _instance(wrapper, md)
    pos = _vr(md, "spring.position")
    try:
        state = ctypes.c_void_p()
        assert lib.fmi3GetFMUState(inst, ctypes.byref(state)) == OK
        held, saved = state.value, _blob(wrapper, inst, state)
        at_save = wrapper.call("Get", "Float32", inst, [pos], [0.0])[1]
        assert wrapper.step(inst, 0.0, DT)[0] == OK
        dispatch = bridge._dispatch                                    # noqa: SLF001

        def refusing(req):
            if req.get("op") == "get_state":
                return {"ok": False, "error": "RuntimeError: no state today"}
            return dispatch(req)

        monkeypatch.setattr(bridge, "_dispatch", refusing)
        assert lib.fmi3GetFMUState(inst, ctypes.byref(state)) == ERROR
        assert "no state today" in wrapper.logs[-1]
        assert state.value == held and _blob(wrapper, inst, state) == saved
        assert lib.fmi3SetFMUState(inst, state) == OK
        assert wrapper.call("Get", "Float32", inst, [pos], [0.0])[1] == at_save
        lib.fmi3FreeFMUState(inst, ctypes.byref(state))
    finally:
        lib.fmi3FreeInstance(inst)


def test_deserialize_always_makes_a_new_state_and_set_changes_none(wrapper, served):
    """``fmi3DeserializeFMUState`` "constructs a copy": a second object,
    even into a variable that still names a live state (the standard gives
    ``*FMUState`` no meaning on entry there, so it is not read).  And
    ``fmi3SetFMUState`` leaves the state it restores unchanged, so it
    restores again."""
    md, bridge = served
    lib = wrapper.lib
    inst = _instance(wrapper, md)
    pos = _vr(md, "spring.position")
    try:
        assert wrapper.step(inst, 0.0, DT)[0] == OK
        state = ctypes.c_void_p()
        assert lib.fmi3GetFMUState(inst, ctypes.byref(state)) == OK
        blob = _blob(wrapper, inst, state)
        at_save = wrapper.call("Get", "Float32", inst, [pos], [0.0])[1]
        alias = ctypes.c_void_p(state.value)
        assert lib.fmi3DeserializeFMUState(inst, blob, len(blob), ctypes.byref(alias)) == OK
        assert alias.value and alias.value != state.value
        assert _blob(wrapper, inst, state) == blob == _blob(wrapper, inst, alias)
        t = DT
        for handle in (state, alias, state, alias):
            status, t = wrapper.step(inst, t, DT)
            assert status == OK
            assert wrapper.call("Get", "Float32", inst, [pos], [0.0])[1] != at_save
            assert lib.fmi3SetFMUState(inst, handle) == OK
            assert wrapper.call("Get", "Float32", inst, [pos], [0.0])[1] == at_save
            assert _blob(wrapper, inst, handle) == blob
            t = DT
        lib.fmi3FreeFMUState(inst, ctypes.byref(alias))
        assert _blob(wrapper, inst, state) == blob            # freeing one leaves the other
        lib.fmi3FreeFMUState(inst, ctypes.byref(state))
    finally:
        lib.fmi3FreeInstance(inst)


def test_states_survive_terminate_and_reset_and_are_freed_with_the_instance(wrapper, served):
    """A state saved before ``fmi3Terminate`` and ``fmi3Reset`` restores
    after them; and ``fmi3FreeInstance`` with states still live -- which the
    standard has it free -- leaves the wrapper able to serve a new instance
    whose own states are its own."""
    md, bridge = served
    lib = wrapper.lib
    inst = _instance(wrapper, md)
    pos = _vr(md, "spring.position")
    for k in range(3):
        assert wrapper.step(inst, k * DT, DT)[0] == OK
    at_save = wrapper.call("Get", "Float32", inst, [pos], [0.0])[1]
    kept = [ctypes.c_void_p() for _ in range(4)]
    for handle in kept:
        assert lib.fmi3GetFMUState(inst, ctypes.byref(handle)) == OK
    assert len({h.value for h in kept}) == 4                  # NULL on entry: four objects
    blob = _blob(wrapper, inst, kept[0])
    assert lib.fmi3Terminate(inst) == OK
    assert _blob(wrapper, inst, kept[1]) == blob
    assert lib.fmi3Reset(inst) == OK
    assert lib.fmi3SetFMUState(inst, kept[2]) == OK           # allowed in every state
    wrapper.to_step_mode(inst, 3 * DT)
    assert wrapper.call("Get", "Float32", inst, [pos], [0.0])[1] == at_save
    lib.fmi3FreeFMUState(inst, ctypes.byref(kept[1]))         # one by the importer
    lib.fmi3FreeInstance(inst)                                # the other three with the instance
    again = _instance(wrapper, md)
    try:
        fresh = ctypes.c_void_p()
        assert lib.fmi3GetFMUState(again, ctypes.byref(fresh)) == OK
        assert _blob(wrapper, again, fresh) != blob           # the new instance's own state
        assert wrapper.call("Get", "Float32", again, [pos], [0.0])[1] == [0.5]
        lib.fmi3FreeFMUState(again, ctypes.byref(fresh))
    finally:
        lib.fmi3FreeInstance(again)
