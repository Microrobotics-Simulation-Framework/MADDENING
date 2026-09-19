"""Second half of the same defect: every idle connection also leaks a thread.

`_serve_conn` sets `conn.settimeout(None)` before it even tries the busy lock,
so a connection that is refused ("bridge already serves an FMU instance") still
blocks forever in `recv_message`.  Nothing bounds the number of connections.
"""
import socket, threading, time
from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.fmi.package import MODEL_IDENTIFIER
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge
from maddening.nodes.spring import SpringDamperNode

DT = 1e-2
gm = GraphManager(); gm.add_node(SpringDamperNode("s", DT, stiffness=30.0)); gm.compile()
md = build_model_description(gm, model_name="P", model_identifier=MODEL_IDENTIFIER)
sc = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token, step_fn=gm._compiled_step,
                              initial_state=gm._state, params=gm.params, param_specs=gm.param_specs()))
br = FmuTcpBridge(sc, md, master_dt=DT).start()
host, port = br.endpoint.split(":"); port = int(port)

base = threading.active_count()
socks = [socket.create_connection((host, port)) for _ in range(300)]
t0 = time.time()
while threading.active_count() - base < 300 and time.time() - t0 < 30:
    time.sleep(0.2)
print(f"300 idle connections -> live threads: {threading.active_count() - base} "
      f"(bridge threads before: {base})")
print("every one is parked in recv_message with no timeout")
for s in socks: s.close()
br.stop()
