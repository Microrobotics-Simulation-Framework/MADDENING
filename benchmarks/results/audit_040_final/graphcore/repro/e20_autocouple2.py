"""Same auto_couple hole, with a group whose settings change the numbers."""
import numpy as np, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

def build(group):
    gm = GraphManager()
    gm.add_node(TableNode("t", 0.01, position=1.0))
    gm.add_node(SpringDamperNode("a", 0.01, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.01, initial_position=2.0))
    gm.add_edge("t","a","position","anchor_position")
    gm.add_edge("a","b","position","anchor_position")
    if group:
        gm.add_coupling_group(["a","b"], max_iterations=1, tolerance=1e-12,
                              relaxation=0.3, acceleration="fixed", strict_convergence=False,
                              diagnostics=True)
    gm.compile()
    return gm

gm = build(True); gm.run(5)
snap = {n: dict(d) for n, d in gm._state.items() if n != "_meta"}
print("coupled(relax .3, 1 iter) after 5:", gm._state["b"])

print("auto_couple ->", gm.auto_couple(), " _dirty =", gm._dirty,
      " groups =", gm._coupling_groups)
gm.run(5)
got = {n: dict(d) for n, d in gm._state.items() if n != "_meta"}

ref = build(False)
for n, d in snap.items(): ref.set_node_state(n, d)
ref.run(5)
print("what the graph now says it is (no groups):", ref._state["b"])
print("what it actually computed              :", got["b"])
print("agree:", all(np.array_equal(np.asarray(got["b"][k]), np.asarray(ref._state["b"][k]))
                    for k in got["b"]))
