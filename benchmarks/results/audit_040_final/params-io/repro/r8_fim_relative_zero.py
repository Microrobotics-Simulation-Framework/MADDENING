"""fim(scale='relative') -- the default -- reports a parameter whose current
value happens to be 0.0 as unidentifiable (crb = +inf), even when the data
determine it perfectly."""
import numpy as np, jax, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening import sysid

gm = GraphManager()
gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.5,
                             initial_position=0.5, initial_velocity=0.0))
gm.compile()
step = gm._build_step_fn()
ext = gm._default_external_inputs()
s0 = gm._state

def residual(p):
    def one(s, _):
        s = step(s, ext, p)
        return s, s["s"]["position"]
    return jax.lax.scan(one, s0, None, length=40)[1]

# A synthetic residual whose sensitivity to every leaf is exactly 1, so the
# only thing that varies between leaves is the leaf's numeric value.
def linear(p):
    return jnp.stack([p["nodes"]["s"][k] for k in sorted(p["nodes"]["s"])])

for name, fn in (("graph rollout", residual), ("J = I (exactly identifiable)", linear)):
    print(f"### {name}")
    for scale in ("relative", None):
        r = sysid.fim(fn, gm.params, scale=scale)
        crb = dict(zip([n.split("']['")[-1].rstrip("']") for n in r.param_names],
                       np.asarray(r.crb)))
        print(f"  scale={scale!r:10}  rank={r.rank}/{len(r.param_names)}  cond={r.cond:g}")
        print(f"      crb={ {k: float(v) for k, v in crb.items()} }")
    print()
print("initial_velocity is exactly 0.0.  With J = I every parameter is perfectly")
print("identifiable, yet scale='relative' (the DEFAULT) zeroes its column and reports")
print("crb = +inf for it.  Nothing in the report says the cause was the value, not the data.")
