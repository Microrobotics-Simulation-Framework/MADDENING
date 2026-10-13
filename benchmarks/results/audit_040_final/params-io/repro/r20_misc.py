import json
import numpy as np, jax.numpy as jnp
from pxr import Usd
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.usd.serialization import save_graph_to_usd, load_graph_from_usd, register_node_class

print("=== A. USD silently stringifies an unserialisable node param (default=str) ===")
class Provider:
    def __init__(self, path): self.path = path
    def __repr__(self): return f"<Provider {self.path}>"

@register_node_class
class P(SimulationNode):
    def __init__(self, name, timestep=0.01, **kw):
        super().__init__(name, timestep); self.params = dict(kw)
    def initial_state(self): return {"x": jnp.zeros(())}
    def update(self, s, b, dt, params=None): return {"x": s["x"] + dt}

gm = GraphManager(); gm.add_node(P("p", 0.01, provider=Provider("/data/mesh.vtu"), k=3))
stage = Usd.Stage.CreateInMemory()
save_graph_to_usd(gm, stage)
print("  stored:", stage.GetPrimAtPath("/Simulation/nodes/p")
      .GetAttribute("maddening:paramsJson").Get())
gm2 = load_graph_from_usd(stage)
print("  reloaded provider:", repr(gm2.get_node("p").params["provider"]),
      type(gm2.get_node("p").params["provider"]).__name__)
print("  config to_dict for the same graph:")
try:
    print("   ", json.dumps(gm.to_dict())[:120])
except TypeError as e:
    print("    json.dumps(to_dict()) raises TypeError:", e)

print()
print("=== B. a non-finite value makes the bridge emit non-standard JSON ===")
from maddening.fmi.tcp_bridge import FmuTcpBridge
reply = FmuTcpBridge._jsonify({"ok": True, "values": np.array([1.0, np.inf, np.nan])})
body = json.dumps(reply, separators=(",", ":"))
print("  wire bytes:", body)
try:
    json.loads(body, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    print("  strict JSON: OK")
except ValueError as e:
    print("  strict JSON: FAILS ->", e)
