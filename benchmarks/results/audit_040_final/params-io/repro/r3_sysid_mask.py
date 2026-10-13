import numpy as np, jax, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening import sysid

def gm_():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.5,
                                 initial_position=0.5))
    gm.compile()
    return gm

gm = gm_()
print("specs:", {k: (v.trainable, v.bounds, v.transform) for k, v in gm.param_specs()["nodes"]["s"].items()})
p0 = gm.params
print("params:", jax.tree.map(lambda x: (x.dtype, float(x)), p0))

# 1. frozen leaf must not move, bit for bit
gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
gm.set_param_spec("s", "damping", ParamSpec(trainable=True))   # trainable but we'll mask it out
def loss(p):
    return jnp.sum((gm.run_scan(30, params=p)["s"]["position"] - 0.1) ** 2)

mask = gm.trainable_mask(gm.params)
# narrow: only stiffness
def narrow(t, specs=None):
    out = jax.tree.map(lambda x: False, t)
    out["nodes"]["s"]["stiffness"] = True
    return out
m = narrow(mask)
res = sysid.fit(gm, loss, mask=m, n_iter=5, lr=0.1)
print("\nfit with mask={stiffness}")
for k in sorted(p0["nodes"]["s"]):
    a = np.asarray(p0["nodes"]["s"][k]); b = np.asarray(res.params["nodes"]["s"][k])
    print(f"  {k:18s} {a.item():.10g} -> {b.item():.10g}  bitwise_same={a.tobytes()==b.tobytes()}  dtype {a.dtype}->{b.dtype}")

# 2. default mask (graph declarations): frozen leaves must not move
gm2 = gm_()
gm2.set_param_spec("s", "mass", ParamSpec(trainable=False))
p0b = gm2.params
def loss2(p):
    return jnp.sum((gm2.run_scan(30, params=p)["s"]["position"] - 0.1) ** 2)
res2 = sysid.fit(gm2, loss2, n_iter=5, lr=0.1)
print("\nfit with default mask (mass frozen)")
for k in sorted(p0b["nodes"]["s"]):
    a = np.asarray(p0b["nodes"]["s"][k]); b = np.asarray(res2.params["nodes"]["s"][k])
    print(f"  {k:18s} {a.item():.10g} -> {b.item():.10g}  bitwise_same={a.tobytes()==b.tobytes()}")
