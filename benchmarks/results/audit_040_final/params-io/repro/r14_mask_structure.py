"""FINDING: sysid checks the mask's leaf COUNT, not its structure, so a mask
whose keys differ from `params` is accepted and fits a DIFFERENT parameter
from the one the caller marked."""
import numpy as np, jax, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening import sysid

def build():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.5,
                                 initial_position=0.5))
    gm.compile(); return gm

gm = build()
step, ext, s0 = gm._build_step_fn(), gm._default_external_inputs(), gm._state
def roll(p, n=30):
    def one(s, _): s = step(s, ext, p); return s, s["s"]["position"]
    return jax.lax.scan(one, s0, None, length=n)[1]
target = jnp.asarray(np.asarray(roll(gm.params)) * 1.3)
def loss(p): return jnp.sum((roll(p) - target) ** 2)

keys = sorted(gm.params["nodes"]["s"])
print("params leaves, flatten order:", keys, "\n")

# The caller means: fit 'damping', freeze the rest.  But the mask they built
# is keyed by their own display labels (same six entries, different spellings).
# (Greek symbol names, as a user might spell them in their own notation.)
labels = {"zeta": True, "alpha": False, "beta": False,
          "gamma": False, "delta": False, "epsilon": False}   # zeta == damping
print("caller's mask (by their own symbol names, zeta == damping):", labels)
print("sorted label order              :", sorted(labels),
      "\n-> the single True sits at index", sorted(labels).index("zeta"),
      "which in params is", keys[sorted(labels).index("zeta")], "\n")

for tag, m in (("correct mask", {"nodes": {"s": {k: (k == "damping") for k in keys}},
                                 "mappings": {}}),
               ("mislabelled mask", {"nodes": {"s": labels}, "mappings": {}})):
    g = build()
    r = sysid.fit(g, loss, mask=m, n_iter=10, lr=0.1)
    moved = {k: (float(g.params["nodes"]["s"][k]), float(r.params["nodes"]["s"][k]))
             for k in keys
             if np.asarray(r.params["nodes"]["s"][k]).tobytes()
                != np.asarray(g.params["nodes"]["s"][k]).tobytes()}
    print(f"{tag:18s} accepted, moved {moved}")

print()
print("The mislabelled mask names no parameter this graph has, yet it is accepted and")
print("silently fits a different constant.  The only check is on the leaf count:")
print("  sysid.py:351 _masked_indices, :409 _resolve_mask, :474 _physical_params")
print("all say 'mask must have the same tree structure as params' but test len() only.")
