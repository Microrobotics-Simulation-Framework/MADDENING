"""Reachability of the _send_reply ValueError: the hello reply carries free text."""
import os, socket, struct, threading, time, traceback, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.environ["WT"])
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode
from maddening.fmi import build_model_description
from maddening.fmi.package import MODEL_IDENTIFIER
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, send_message, recv_raw

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

def bridge_for(model_name):
    gm = _graph()
    md = build_model_description(gm, model_name=model_name,
                                 model_identifier=MODEL_IDENTIFIER)
    sc = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
    ))
    return md, FmuTcpBridge(sc, md, master_dt=DT)

for name in ("Plant", "Infinity", "NaN", "-Infinity"):
    print("="*70)
    print(f"model_name = {name!r}")
    md, br = bridge_for(name)
    # 1. in-process handle() -- what the sidecar/handle path does
    try:
        r = br.handle({"op": "hello"})
        print("  handle() ->", {k: v for k, v in r.items() if k in ("ok","model","token")})
    except Exception as e:
        print("  handle() raised", type(e).__name__, e)
    # 2. over the socket
    br.start()
    host, port = br.endpoint.rsplit(":", 1)
    excs = []
    def hook(args): excs.append(args)
    threading.excepthook = hook
    s = socket.create_connection((host, int(port)), timeout=5)
    t0 = time.monotonic()
    try:
        send_message(s, {"op": "hello", "protocol": 2, "binary": True})
        s.settimeout(5.0)
        try:
            got = recv_raw(s)
            print(f"  reply after {time.monotonic()-t0:.3f}s:", got)
        except socket.timeout:
            print(f"  TIMEOUT after {time.monotonic()-t0:.3f}s waiting for hello reply")
    finally:
        s.close()
        time.sleep(0.3)
        for a in excs:
            print("  worker thread died:", a.exc_type.__name__, a.exc_value)
        br.stop()
