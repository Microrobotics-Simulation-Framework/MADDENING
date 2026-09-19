"""Gradient through a checkpointed segment: restore then differentiate."""
import numpy as np, jax, jax.numpy as jnp, tempfile, os
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
TMP = tempfile.mkdtemp()

def build():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.001, stiffness=50.0, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.001, stiffness=50.0, initial_position=2.0))
    gm.add_edge("a","b","position","anchor_position")
    gm.add_edge("b","a","position","anchor_position")
    gm.add_coupling_group(["a","b"], tolerance=1e-10, max_iterations=30,
                          solver="ift", diagnostics=True)
    gm.compile()
    return gm

def grad_of_next(gm, n):
    """d(final a.position)/d(a.stiffness) over n steps from gm's current state,
    using the raw step fn so gm._state is never written with tracers."""
    step = gm._build_step_fn()
    s0 = jax.tree.map(lambda x: x, gm._state)
    ext = gm._default_external_inputs()
    def loss(p):
        s = s0
        for _ in range(n):
            s = step(s, ext, p)
        return s["a"]["position"]
    p = jax.tree.map(lambda x: x, gm.params)
    return float(jax.grad(loss)(p)["nodes"]["a"]["stiffness"])

# uninterrupted
g1 = build(); g1.run(20)
ref_grad = grad_of_next(g1, 10)

# via a checkpoint
g2 = build(); g2.run(20); g2.save_state(os.path.join(TMP,"s.npz"))
g3 = build(); g3.load_state(os.path.join(TMP,"s.npz"))
ck_grad = grad_of_next(g3, 10)
print("grad, never saved :", ref_grad)
print("grad, after resume:", ck_grad)
print("identical:", ref_grad == ck_grad)

# finite-difference sanity on the resumed graph
def f(k):
    g = build(); g.load_state(os.path.join(TMP,"s.npz"))
    p = jax.tree.map(lambda x: x, g.params)
    p["nodes"]["a"]["stiffness"] = jnp.asarray(k, jnp.float32)
    return float(g.run_scan(10, params=p)["a"]["position"])
h = 1e-2
fd = (f(50.0+h) - f(50.0-h)) / (2*h)
print("finite difference  :", fd, " rel err:", abs(fd-ck_grad)/max(abs(fd),1e-30))
