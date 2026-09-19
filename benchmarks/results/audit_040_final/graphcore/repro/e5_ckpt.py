"""Non-trivial checkpoint round trip: coupled graph + params + external inputs."""
import numpy as np, jax, jax.numpy as jnp, tempfile, os
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

TMP = tempfile.mkdtemp()

def build(k=50.0):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.001, stiffness=k, damping=1.0, mass=1.0,
                                 rest_length=1.0, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.001, stiffness=k, damping=1.0, mass=1.0,
                                 rest_length=1.0, initial_position=2.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], tolerance=1e-8, max_iterations=20,
                          diagnostics=True)
    gm.add_external_input("a", "external_force", shape=(), dtype=jnp.float32)
    gm.compile()
    return gm

ext = {"a": {"external_force": jnp.array(0.5)}}

gm = build()
# calibrate-ish: move a param off its constructor value
gm.params["nodes"]["a"]["stiffness"] = jnp.array(37.5, dtype=jnp.float32)
gm.run(20, external_inputs=ext)
saved_state = jax.tree.map(lambda x: np.asarray(x).copy(), gm._state)
saved_params = jax.tree.map(lambda x: np.asarray(x).copy(), gm.params)
gm.save_state(os.path.join(TMP, "c.npz"))

# continue on the original
for _ in range(10): gm.step(ext)
ref_after = jax.tree.map(lambda x: np.asarray(x).copy(), gm._state)

# restore into a *fresh* graph
gm2 = build()
gm2.load_state(os.path.join(TMP, "c.npz"))

def cmp(tag, a, b):
    bad = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad.append(f"{k}: present in only one ({k in a},{k in b})"); continue
        for f in sorted(set(a[k]) | set(b[k])):
            if f not in a[k] or f not in b[k]:
                bad.append(f"{k}/{f}: present in only one"); continue
            x, y = np.asarray(a[k][f]), np.asarray(b[k][f])
            if x.shape != y.shape:
                bad.append(f"{k}/{f}: shape {x.shape} vs {y.shape}")
            elif not np.array_equal(x, y):
                bad.append(f"{k}/{f}: {x} vs {y}")
    print(f"{tag}: {'BIT-IDENTICAL' if not bad else 'DIFFER'}")
    for b_ in bad: print("   ", b_)

cmp("state after restore", saved_state, jax.tree.map(lambda x: np.asarray(x), gm2._state))
cmp("params after restore", saved_params, jax.tree.map(lambda x: np.asarray(x), gm2.params))

# does the restored graph step identically?
for _ in range(10): gm2.step(ext)
cmp("state after 10 more steps", ref_after, jax.tree.map(lambda x: np.asarray(x), gm2._state))
print("diag orig:", gm.coupling_diagnostics())
print("diag rest:", gm2.coupling_diagnostics())
