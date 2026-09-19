"""USD params JSON round-trip: awkward leaves."""
import json
import numpy as np
import jax.numpy as jnp
from pxr import Usd
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.usd.serialization import (save_graph_to_usd, load_graph_from_usd,
                                         register_node_class, _params_to_serializable)


@register_node_class
class WeirdNode(SimulationNode):
    def __init__(self, name, timestep=0.01, **kw):
        super().__init__(name, timestep)
        self.params = dict(kw)

    def initial_state(self):
        return {"x": jnp.zeros(())}

    def update(self, state, boundary_inputs, dt, params=None):
        return {"x": state["x"] + dt}


print("--- _params_to_serializable on awkward leaves ---")
weird = {
    "empty_1d": np.zeros((0,), dtype=np.float32),
    "empty_2d": np.zeros((0, 3), dtype=np.float32),
    "nan": float("nan"),
    "inf": float("inf"),
    "tiny": 5e-324,
    "huge": 1.7976931348623157e308,
    "big_int": 2**62 + 1,
    "bool": True,
    "unicode": "µ✓",
    "nested": {"a": 1},
    "tuple": (1.0, 2.0),
    "f32arr": np.array([1.1, 2.2], dtype=np.float32),
}
ser = _params_to_serializable(weird)
txt = json.dumps(ser, default=str)
print("  json text:", txt[:400])
back = json.loads(txt)
for k in weird:
    a, b = weird[k], back[k]
    same = (np.asarray(a).shape == np.asarray(b).shape) if hasattr(a, "shape") else (a == b or (isinstance(a,float) and np.isnan(a) and np.isnan(b)))
    print(f"    {k:10s}: {type(a).__name__:10s} -> {b!r:40.40s} same={same}")

print()
print("--- shape loss for zero-size arrays ---")
print("  empty_2d in:", weird["empty_2d"].shape, " out:", np.asarray(back["empty_2d"]).shape)

print()
print("--- default=str silently stringifies unserialisable params ---")
class Opaque:
    def __repr__(self): return "<Opaque object>"
p2 = _params_to_serializable({"provider": Opaque()})
print("  ", json.dumps(p2, default=str))

print()
print("--- full graph round trip with NaN / inf params ---")
gm = GraphManager()
gm.add_node(WeirdNode("w", 0.01, gain=float("nan"), cap=float("inf"), arr=np.zeros((0,3))))
stage = Usd.Stage.CreateInMemory()
save_graph_to_usd(gm, stage)
raw = stage.GetPrimAtPath("/Simulation/nodes/w").GetAttribute("maddening:paramsJson").Get()
print("  stored JSON:", raw)
try:
    json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"strict JSON forbids {c}")))
    print("  strict-JSON parse: OK")
except ValueError as e:
    print("  strict-JSON parse FAILS:", e)
gm2 = load_graph_from_usd(stage)
print("  reloaded params:", gm2.get_node("w").params)
