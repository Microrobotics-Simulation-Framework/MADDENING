"""Drive the compiled wrapper through ctypes against the real bridge."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import ctypes, socket, struct, threading, json
import numpy as np, pytest
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode
from maddening.fmi import build_model_description
from maddening.fmi.package import MODEL_IDENTIFIER
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge

SO = os.environ.get("FMU_SO", "/home/nick/.claude/jobs/d22809a4/tmp/audit4/cbuild/maddening_fmu_plain.so")
DT = 1e-2
LOGS = []
LOGCB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)

@LOGCB
def _log(env, st, cat, msg):
    LOGS.append((st, msg.decode()))


def _graph():
    gm = GraphManager()
    gm.add_node(HeatNode(name="rod", timestep=DT, n_cells=4, thermal_diffusivity=0.1, initial_temperature=50.0, length=1.0))
    gm.add_node(SpringDamperNode(name="spring", timestep=DT, stiffness=30.0, damping=2.0, initial_position=0.5))
    gm.add_external_input("spring", "anchor_position")
    gm.compile()
    return gm


def _bridge(gm):
    md = build_model_description(gm, model_name="P", model_identifier=MODEL_IDENTIFIER, include_evolving=True)
    sc = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token, step_fn=gm._compiled_step,
                                  initial_state=gm._state, params=gm.params, param_specs=gm.param_specs()))
    return md, FmuTcpBridge(sc, md, master_dt=DT)


def _vr(md, name):
    return next(v.value_reference for v in md.variables if v.name == name)


def _lib():
    lib = ctypes.CDLL(SO)
    lib.fmi3InstantiateCoSimulation.restype = ctypes.c_void_p
    lib.fmi3InstantiateCoSimulation.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool,
                                               ctypes.c_bool, ctypes.c_bool, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, LOGCB, ctypes.c_void_p]
    lib.fmi3DoStep.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_bool, ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_double)]
    lib.fmi3GetFloat64.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
    lib.fmi3GetFloat32.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    lib.fmi3SetFloat32.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    lib.fmi3DeserializeFMUState.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
    lib.fmi3SetFMUState.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.fmi3GetFMUState.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.fmi3FreeFMUState.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.fmi3FreeInstance.argtypes = [ctypes.c_void_p]
    lib.fmi3Reset.argtypes = [ctypes.c_void_p]
    return lib


def _step(lib, inst, t, h):
    b1, b2, b3 = ctypes.c_bool(), ctypes.c_bool(), ctypes.c_bool(); last = ctypes.c_double()
    return lib.fmi3DoStep(inst, t, h, False, b1, b2, b3, last), last.value


def _get(lib, inst, vrs, n):
    arr = (ctypes.c_uint32 * len(vrs))(*vrs); out = (ctypes.c_double * n)()
    st = lib.fmi3GetFloat64(inst, arr, len(vrs), out, n)
    return st, list(out)


def test_set_fmu_state_with_corrupt_bytes_does_not_kill_the_instance():
    gm = _graph(); md, bridge = _bridge(gm)
    with bridge:
        os.environ["MADDENING_FMU_ENDPOINT"] = bridge.endpoint
        lib = _lib()
        inst = lib.fmi3InstantiateCoSimulation(b"i", md.instantiation_token.encode(), b"", False, True, False, False, None, 0, None, _log, None)
        assert inst
        st, _ = _step(lib, inst, 0.0, DT); assert st == 0
        # array output: temperature (4,)
        st, vals = _get(lib, inst, [_vr(md, "rod.temperature")], 4); assert st == 0 and len(vals) == 4
        # corrupt serialized state (opaque bytes from the importer's point of view)
        h = ctypes.c_void_p()
        blob = b'abc"def\\x'
        assert lib.fmi3DeserializeFMUState(inst, blob, len(blob), ctypes.byref(h)) == 0
        st = lib.fmi3SetFMUState(inst, h)
        print("SetFMUState corrupt ->", st, LOGS[-1:])
        assert st != 0
        # the instance must still be usable
        st2, last = _step(lib, inst, DT, DT)
        print("DoStep after corrupt SetFMUState ->", st2, LOGS[-1:])
        assert st2 == 0, "connection was killed by a malformed SetFMUState"
        lib.fmi3FreeInstance(inst)


def test_get_state_set_state_round_trip_and_reset_via_c():
    gm = _graph(); md, bridge = _bridge(gm)
    with bridge:
        os.environ["MADDENING_FMU_ENDPOINT"] = bridge.endpoint
        lib = _lib()
        inst = lib.fmi3InstantiateCoSimulation(b"i", md.instantiation_token.encode(), b"", False, True, False, False, None, 0, None, _log, None)
        anchor = _vr(md, "spring.anchor_position"); pos = _vr(md, "spring.position")
        a = (ctypes.c_uint32 * 1)(anchor); v = (ctypes.c_float * 1)(0.3)
        assert lib.fmi3SetFloat32(inst, a, 1, v, 1) == 0
        for i in range(3):
            assert _step(lib, inst, i * DT, DT)[0] == 0
        h = ctypes.c_void_p(); assert lib.fmi3GetFMUState(inst, ctypes.byref(h)) == 0
        _, before = _get(lib, inst, [pos], 1)
        for i in range(3, 6):
            assert _step(lib, inst, i * DT, DT)[0] == 0
        assert lib.fmi3SetFMUState(inst, h) == 0
        _, after = _get(lib, inst, [pos, anchor], 2)
        assert after[0] == pytest.approx(before[0]) and after[1] == pytest.approx(0.3, rel=1e-6)
        assert lib.fmi3Reset(inst) == 0
        _, r = _get(lib, inst, [pos, anchor, _vr(md, "time")], 3)
        assert r == [0.5, 0.0, 0.0]
        lib.fmi3FreeFMUState(inst, ctypes.byref(h))
        lib.fmi3FreeInstance(inst)


def test_wrapper_against_server_that_closes_mid_body_then_recovers():
    """A sidecar that dies mid-reply must surface fmi3Error and never crash."""
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    port = srv.getsockname()[1]
    def serve():
        conn, _ = srv.accept()
        # hello
        n, = struct.unpack(">I", conn.recv(4)); conn.recv(n)
        body = json.dumps({"ok": True, "token": "T", "model": "m", "master_dt": DT}).encode()
        conn.sendall(struct.pack(">I", len(body)) + body)
        # step: announce 100 bytes, send 10, close
        n, = struct.unpack(">I", conn.recv(4)); conn.recv(n)
        conn.sendall(struct.pack(">I", 100) + b'{"ok":true')
        conn.close()
    th = threading.Thread(target=serve, daemon=True); th.start()
    os.environ["MADDENING_FMU_ENDPOINT"] = f"127.0.0.1:{port}"
    lib = _lib()
    inst = lib.fmi3InstantiateCoSimulation(b"i", b"T", b"", False, True, False, False, None, 0, None, _log, None)
    assert inst
    st, last = _step(lib, inst, 0.0, DT)
    print("step vs dying server ->", st, last, LOGS[-1:])
    assert st != 0
    st2, _ = _step(lib, inst, 0.0, DT)
    assert st2 != 0
    st3, vals = _get(lib, inst, [1], 1)
    assert st3 != 0
    lib.fmi3FreeInstance(inst)


def test_wrapper_against_server_replying_huge_length():
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    port = srv.getsockname()[1]
    def serve():
        conn, _ = srv.accept()
        n, = struct.unpack(">I", conn.recv(4)); conn.recv(n)
        conn.sendall(struct.pack(">I", 0xFFFFFFFF) + b'{"ok":true')
        conn.close()
    threading.Thread(target=serve, daemon=True).start()
    os.environ["MADDENING_FMU_ENDPOINT"] = f"127.0.0.1:{port}"
    lib = _lib()
    inst = lib.fmi3InstantiateCoSimulation(b"i", b"", b"", False, True, False, False, None, 0, None, _log, None)
    print("huge length hello ->", inst, LOGS[-1:])
    assert not inst
