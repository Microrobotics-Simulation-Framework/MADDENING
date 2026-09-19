"""Attacker-style probe of FmuTcpBridge over a real socket."""
import io, json, socket, struct, sys, time, zipfile
import numpy as np
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.fmi.package import MODEL_IDENTIFIER
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, recv_raw, _MAX_MESSAGE, _BINARY_FLAG
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

gm = _graph()
md = build_model_description(gm, model_name="Plant", model_identifier=MODEL_IDENTIFIER)
sc = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token, step_fn=gm._compiled_step,
                              initial_state=gm._state, params=gm.params,
                              param_specs=gm.param_specs()))
bridge = FmuTcpBridge(sc, md, master_dt=DT).start()
host, port = bridge.endpoint.split(":")

def connect():
    s = socket.create_connection((host, int(port)), timeout=10)
    s.settimeout(10)
    return s

def send_json(s, obj):
    b = json.dumps(obj).encode()
    s.sendall(struct.pack(">I", len(b)) + b)

def send_bin(s, header, raw):
    h = json.dumps(header, separators=(",", ":")).encode()
    body = struct.pack(">I", len(h)) + h + raw
    s.sendall(struct.pack(">I", _BINARY_FLAG | len(body)) + body)

def read(s):
    try:
        return recv_raw(s)
    except Exception as e:
        return ("ERR", repr(e))

def show(label, got):
    if got is None:
        print(f"  {label:55s} -> EOF (connection dropped)")
    elif isinstance(got, tuple) and got[0] == "ERR":
        print(f"  {label:55s} -> {got[1]}")
    else:
        isb, body = got
        print(f"  {label:55s} -> binary={isb} {body[:150]!r}")

vr = {v.name: v.value_reference for v in md.variables}
K = vr["spring.params.stiffness"]
POS = vr["spring.position"]

print("=== 1. malformed / hostile JSON requests ===")
s = connect()
for label, payload in [
    ("hello", {"op": "hello"}),
    ('set vr not a list', {"op": "set", "vr": K, "values": [1.0]}),
    ('set values nested ragged', {"op": "set", "vr": [K], "values": [[1.0, 2.0]]}),
    ('set values string', {"op": "set", "vr": [K], "values": ["../../etc/passwd"]}),
    ('set vr huge int', {"op": "set", "vr": [2**70], "values": [1.0]}),
    ('set vr float', {"op": "set", "vr": [1.5], "values": [1.0]}),
    ('get vr = dict', {"op": "get", "vr": {"a": 1}}),
    ('step dt nan', {"op": "step", "dt": float("nan"), "t": 0.0}),
    ('step t inf', {"op": "step", "dt": DT, "t": 1e400}),
    ('unknown op', {"op": "wat"}),
    ('op missing', {}),
    ('op is a list', {"op": ["hello"]}),
]:
    try:
        send_json(s, payload)
    except Exception as e:
        print(f"  {label:55s} -> send failed {e!r}"); break
    show(label, read(s))
s.close()

print()
print("=== 2. framing attacks ===")
for label, blob in [
    ("length 0", struct.pack(">I", 0)),
    ("length says 100, sends 4", struct.pack(">I", 100) + b"aaaa"),
    ("length > 64MiB", struct.pack(">I", 0x7FFFFFFF) + b"{}"),
    ("binary flag + length > 64MiB", struct.pack(">I", 0xFFFFFFFF) + b"{}"),
    ("binary without hello negotiation", struct.pack(">I", _BINARY_FLAG | 6) + struct.pack(">I", 2) + b"{}"),
]:
    s = connect()
    s.sendall(blob)
    show(label, read(s))
    s.close()

print()
print("=== 3. binary frame attacks (after negotiating) ===")
def neg():
    s = connect()
    send_json(s, {"op": "hello", "protocol": 2, "binary": True})
    read(s)
    return s

for label, hdr, raw in [
    ("set n=0 raw empty",              {"op": "set", "vr": [K], "n": 0, "dtype": "f64"}, b""),
    ("set n huge, raw tiny",           {"op": "set", "vr": [K], "n": 2**40, "dtype": "f64"}, b"\x00"*8),
    ("set n negative",                 {"op": "set", "vr": [K], "n": -1, "dtype": "f64"}, b"\x00"*8),
    ("set n bool True",                {"op": "set", "vr": [K], "n": True, "dtype": "f64"}, b"\x00"*8),
    ("set dtype f32",                  {"op": "set", "vr": [K], "n": 1, "dtype": "f32"}, b"\x00"*4),
    ("set_state n mismatch",           {"op": "set_state", "n": 10}, b"PK\x03\x04"),
    ("set_state zip bomb decl",        {"op": "set_state", "n": 4}, b"PK\x03\x04"),
]:
    s = neg()
    send_bin(s, hdr, raw)
    show(label, read(s))
    s.close()

print()
print("=== 4. binary header_len lies ===")
for label, body in [
    ("hlen > payload", struct.pack(">I", 1000) + b'{"op":"set"}'),
    ("hlen = 0", struct.pack(">I", 0) + b'{"op":"set"}'),
    ("payload < 4", b"ab"),
    ("header not an object", struct.pack(">I", 2) + b"[]" ),
]:
    s = neg()
    s.sendall(struct.pack(">I", _BINARY_FLAG | len(body)) + body)
    show(label, read(s))
    s.close()

print()
print("=== 5. set_state archive attacks ===")
def npz_bytes(members):
    buf = io.BytesIO()
    np.savez(buf, **members)
    return buf.getvalue()

live_state = {n: dict(f) for n, f in sc.state.items()}
good = bridge.handle({"op": "get_state"})
import base64
good_blob = base64.b64decode(good["state"])
print("  good state size:", len(good_blob))

def try_state(label, blob):
    s = connect()
    send_json(s, {"op": "hello"})
    read(s)
    send_json(s, {"op": "set_state", "state": base64.b64encode(blob).decode()})
    show(label, read(s))
    s.close()

try_state("well-formed", good_blob)
try_state("truncated archive", good_blob[: len(good_blob)//2])
try_state("empty", b"")
try_state("not a zip", b"hello world" * 100)
# member with traversal name
zf_buf = io.BytesIO()
with zipfile.ZipFile(zf_buf, "w") as z:
    z.writestr("../../../../tmp/pwned.npy", b"x" * 10)
try_state("path traversal member", zf_buf.getvalue())
# declared size lie: declare tiny, actually huge
zf_buf = io.BytesIO()
with zipfile.ZipFile(zf_buf, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("s/ball/position.npy", b"\x00" * (50 * 1024 * 1024))
raw = bytearray(zf_buf.getvalue())
print("  bomb compressed size:", len(raw))
try_state("50MB member declared honestly", bytes(raw))

bridge.stop()
