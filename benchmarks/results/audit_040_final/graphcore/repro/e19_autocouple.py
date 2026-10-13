"""auto_couple() clears the coupling groups but only marks the graph dirty via
add_coupling_group().  When it creates none, the compiled step keeps coupling."""
import numpy as np, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

# acyclic chain, but the user pinned a coupling group over it by hand
gm = GraphManager()
gm.add_node(TableNode("t", 0.01, position=1.0))
gm.add_node(SpringDamperNode("a", 0.01, initial_position=0.0))
gm.add_node(SpringDamperNode("b", 0.01, initial_position=2.0))
gm.add_edge("t","a","position","anchor_position")
gm.add_edge("a","b","position","anchor_position")
try:
    gm.add_coupling_group(["a","b"], max_iterations=12, tolerance=1e-10,
                          diagnostics=True)
    print("acyclic coupling group accepted")
except Exception as e:
    print("acyclic group rejected:", type(e).__name__, str(e)[:120]); raise SystemExit

gm.compile()
gm.run(5)
print("groups:", [sorted(g.nodes) for g in gm._coupling_groups],
      " meta:", sorted(gm._state.get("_meta", {})))
coupled_state = {n: dict(d) for n, d in gm._state.items() if n != "_meta"}

groups = gm.auto_couple()
print("after auto_couple: groups =", groups, " _dirty =", gm._dirty)
gm.run(5)
after_auto = {n: dict(d) for n, d in gm._state.items() if n != "_meta"}
print("meta after 5 more steps:", sorted(gm._state.get("_meta", {})))

# ground truth: the same graph with no coupling group at all
ref = GraphManager()
ref.add_node(TableNode("t", 0.01, position=1.0))
ref.add_node(SpringDamperNode("a", 0.01, initial_position=0.0))
ref.add_node(SpringDamperNode("b", 0.01, initial_position=2.0))
ref.add_edge("t","a","position","anchor_position")
ref.add_edge("a","b","position","anchor_position")
ref.compile()
for n, d in coupled_state.items(): ref.set_node_state(n, d)
ref.run(5)
print("uncoupled truth  b:", ref._state["b"])
print("after auto_couple b:", after_auto["b"])
print("agree:", all(np.array_equal(np.asarray(after_auto["b"][k]),
                                   np.asarray(ref._state["b"][k]))
                    for k in after_auto["b"]))
# forcing the recompile the graph should have done
gm._dirty = True; gm.compile()
for n, d in coupled_state.items(): gm.set_node_state(n, d)
gm.run(5)
print("after forced recompile b:", gm._state["b"])
