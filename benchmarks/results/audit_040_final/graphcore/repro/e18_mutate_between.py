"""Graph mutated between calls: the scan cache must not serve the old program."""
import numpy as np, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode

def base():
    gm = GraphManager()
    gm.add_node(TableNode("table", 0.01, position=0.0))
    gm.add_node(BallNode("ball", 0.01, initial_position=0.05, initial_velocity=-1.0))
    gm.compile(); return gm

# A: run_scan, then add the table->ball edge, then run_scan again
a = base()
a.run_scan(3)
a.add_edge("table","ball","position","table_position")   # collision floor at 0.0
a.run_scan(20)
# B: the same graph built with the edge from the start, fed A's mid state
b = base()
b.run_scan(3)
mid = {n: dict(d) for n, d in b._state.items()}
b2 = GraphManager()
b2.add_node(TableNode("table", 0.01, position=0.0))
b2.add_node(BallNode("ball", 0.01, initial_position=0.05, initial_velocity=-1.0))
b2.add_edge("table","ball","position","table_position")
b2.compile()
for n, d in mid.items(): b2.set_node_state(n, d)
b2.run_scan(20)
print("mutated-then-scan :", a._state["ball"])
print("built-with-edge   :", b2._state["ball"])
print("agree:", all(np.array_equal(np.asarray(a._state["ball"][k]),
                                   np.asarray(b2._state["ball"][k]))
                    for k in a._state["ball"]))
print("scan cache size after mutation:", len(a._scan_cache),
      "generation:", a._compile_generation)

# C: same n_steps before and after the mutation (worst case for the key)
c = base(); c.run_scan(5)
c.add_edge("table","ball","position","table_position")
c.run_scan(5)
d = GraphManager()
d.add_node(TableNode("table", 0.01, position=0.0))
d.add_node(BallNode("ball", 0.01, initial_position=0.05, initial_velocity=-1.0))
d.add_edge("table","ball","position","table_position"); d.compile()
e = base(); e.run_scan(5)
for n, s in e._state.items(): d.set_node_state(n, dict(s))
d.run_scan(5)
print("same-n_steps case agree:",
      all(np.array_equal(np.asarray(c._state["ball"][k]), np.asarray(d._state["ball"][k]))
          for k in c._state["ball"]), c._state["ball"], d._state["ball"])
