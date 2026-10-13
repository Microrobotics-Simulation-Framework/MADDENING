"""Even a bare TCP connect that sends nothing wedges the bridge (a port scan is enough)."""
import json, socket, struct, time
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

idle = socket.create_connection((host, port))      # connect, send nothing, keep open
while not bridge._busy.locked():
    time.sleep(0.05)
s = socket.create_connection((host, port), timeout=3); s.settimeout(3)
b = json.dumps({"op": "hello"}).encode(); s.sendall(struct.pack(">I", len(b)) + b)
print("real importer's hello ->", recv_raw(s))
s.close(); bridge.stop(); idle.close()
