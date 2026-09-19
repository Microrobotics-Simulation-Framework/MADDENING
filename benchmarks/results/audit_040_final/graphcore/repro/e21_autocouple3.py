"""auto_couple() drops the groups without dirtying the graph -> the step that
keeps running is the coupled one, and it is NOT the graph the object describes."""
import numpy as np, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

def build(group_kw):
    gm = GraphManager()
    gm.add_node(TableNode("t", 0.01, position=1.0))
    gm.add_node(SpringDamperNode("a", 0.005, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.005, initial_position=2.0))
    gm.add_edge("t","a","position","anchor_position")
    gm.add_edge("a","b","position","anchor_position")
    if group_kw is not None:
        gm.add_coupling_group(["a","b"], **group_kw)
    gm.compile()
    return gm

KW = dict(max_iterations=8, tolerance=1e-10, subcycling=True, diagnostics=True)
gm = build(KW); gm.run(4)
snap = {n: dict(d) for n, d in gm._state.items() if n != "_meta"}

print("auto_couple ->", gm.auto_couple(), "| _dirty =", gm._dirty,
      "| gm._coupling_groups =", gm._coupling_groups)
print("repr says:", repr(gm))
gm.run(4)
got = gm._state["b"]

ref = build(None)                       # the graph gm now claims to be
for n, d in snap.items(): ref.set_node_state(n, d)
ref.run(4)

still = build(KW)                       # the graph gm actually still runs
for n, d in snap.items(): still.set_node_state(n, d)
still.run(4)

print("gm after auto_couple :", got)
print("no-group reference   :", ref._state["b"])
print("coupled reference    :", still._state["b"])
print("gm == no-group :", all(np.array_equal(np.asarray(got[k]), np.asarray(ref._state['b'][k])) for k in got))
print("gm == coupled  :", all(np.array_equal(np.asarray(got[k]), np.asarray(still._state['b'][k])) for k in got))
print("coupling_diagnostics() now reports:", gm.coupling_diagnostics())
print("_meta still carries:", sorted(gm._state.get("_meta", {})))
