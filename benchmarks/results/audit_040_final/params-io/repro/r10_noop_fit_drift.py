"""FINDING: a fit that takes no step still moves the masked parameters.

FitResult's docstring: "Every leaf outside the resolved mask is the value that
went in, bit for bit -- not merely close -- so comparing a fit's input and
output leaf by leaf says exactly which constants the calibration touched."
The leaves *inside* the mask get no such protection, so a zero-iteration fit,
or one that stops at `tol` before its first update, reports every trainable
constant as "touched".
"""
import numpy as np, jax, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening import sysid

def build():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.5,
                                 initial_position=0.5))
    gm.compile()
    return gm

gm = build()
step, ext, s0 = gm._build_step_fn(), gm._default_external_inputs(), gm._state
def roll(p, n=30):
    def one(s, _):
        s = step(s, ext, p); return s, s["s"]["position"]
    return jax.lax.scan(one, s0, None, length=n)[1]
target = jnp.asarray(np.asarray(roll(gm.params)))
def loss(p):  return jnp.sum((roll(p) - target) ** 2)

def report(tag, before, after):
    print(f"  {tag}")
    for k in sorted(before):
        a, b = np.asarray(before[k]), np.asarray(after[k])
        print(f"    {k:18s} {a.item()!r:22} -> {b.item()!r:22} bitwise_same={a.tobytes()==b.tobytes()}")

print("1. fit(n_iter=0) -- no gradient is ever evaluated:")
r = sysid.fit(gm, loss, n_iter=0)
print("   losses:", r.losses, " n_iter:", r.n_iter, " converged:", r.converged)
report("", gm.params["nodes"]["s"], r.params["nodes"]["s"])

print("\n2. fit(tol=1e30) -- stops on the first loss, before any update:")
g2 = build()
r2 = sysid.fit(g2, loss, n_iter=100, tol=1e30)
print("   converged:", r2.converged, " n_iter:", r2.n_iter, " losses:", r2.losses)
report("", g2.params["nodes"]["s"], r2.params["nodes"]["s"])

print("\n3. does the drift accumulate over repeated no-op fits?")
g3 = build()
p = g3.params
for i in range(5):
    p = sysid.fit(g3, loss, params=p, n_iter=0).params
    print(f"   after {i+1} no-op fits: stiffness={float(p['nodes']['s']['stiffness'])!r} "
          f"mass={float(p['nodes']['s']['mass'])!r}")
