import os, json, math
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode
from maddening.serialization.json_codec import loads as jloads

def strict(t):
    def refuse(x): raise ValueError(f"bare token {x!r}")
    return json.loads(t, parse_constant=refuse)

# array param carrying NaN / inf, through the CONFIG surface
gm = GraphManager()
gm.add_node(HeatNode(name="h", timestep=0.01, n_cells=4,
                     initial_temperature=np.array([1.0, np.nan, np.inf, -0.0])))
gm.compile()
d = gm.to_dict()
print("params value:", repr(d["nodes"][0]["params"]["initial_temperature"])[:120])
try:
    t = json.dumps(d)
    print("json.dumps OK; strict reparse:", strict(t) is not None)
    back = GraphManager.from_dict(jloads(t), {"HeatNode": HeatNode})
    v = back._nodes["h"].node.params["initial_temperature"]
    print("reloaded:", np.asarray(v))
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:200])

# exotic finite floats through the config surface
vals = [-0.0, 5e-324, 1.7976931348623157e308, 0.1234567890123456789, -1e-309]
gm2 = GraphManager()
gm2.add_node(SpringDamperNode(name="s", timestep=0.01, stiffness=vals[0], damping=vals[1],
                              mass=vals[3], rest_length=vals[4], initial_position=vals[2]))
gm2.compile()
d2 = gm2.to_dict()
t2 = json.dumps(d2)
b2 = GraphManager.from_dict(jloads(t2), {"SpringDamperNode": SpringDamperNode})
p_in = gm2.to_dict()["nodes"][0]["params"]
p_out = b2.to_dict()["nodes"][0]["params"]
bad = []
for k in p_in:
    a, b = p_in[k], p_out[k]
    if isinstance(a, float) and isinstance(b, float):
        if not (a == b and math.copysign(1, a) == math.copysign(1, b)):
            bad.append((k, a, b))
    elif a != b:
        bad.append((k, a, b))
print("\nexotic finite floats not preserved:", bad or "none")
print("  values in:", {k: p_in[k] for k in p_in if isinstance(p_in[k], float)})
