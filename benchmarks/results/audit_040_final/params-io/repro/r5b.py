import json, socket, struct, threading, time, traceback
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

atk = socket.create_connection((host, port))
atk.sendall(struct.pack(">I", 100))
time.sleep(1.0)
print("busy locked?", bridge._busy.locked())
print("threads:", [t.name for t in threading.enumerate()])
for t in threading.enumerate():
    if t.name.startswith("Thread"):
        fr = sys._current_frames().get(t.ident) if False else None
import sys
for tid, frame in sys._current_frames().items():
    st = "".join(traceback.format_stack(frame))
    if "tcp_bridge" in st:
        print("---- thread", tid, "----"); print(st[-800:])
