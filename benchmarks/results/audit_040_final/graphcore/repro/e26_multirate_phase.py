"""compile() resets _meta['step_count'] to 0 on a multi-rate graph, so any
recompile mid-run re-phases every node whose rate divider is > 1."""
import numpy as np, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode

def build(with_ext):
    gm = GraphManager()
    gm.add_node(TableNode("table", 0.01, position=0.0))
    gm.add_node(BallNode("ball", 0.03, initial_position=1.0))   # divider 3
    gm.add_edge("table", "ball", "position", "table_position")
    if with_ext:
        gm.add_external_input("table", "unused", shape=())
    gm.compile()
    return gm

# uninterrupted: the (zero-valued, unread) external input exists from the start
ref = build(True); ref.run(8)

# interrupted: declare the same input after 3 steps -> recompile
gm = build(False); gm.run(4)
print("step_count before the edit:", int(gm._state["_meta"]["step_count"]))
gm.add_external_input("table", "unused", shape=())      # dirties the graph
gm.run(4)
print("step_count after 4 more   :", int(gm._state["_meta"]["step_count"]),
      "(expected 8)")

print("uninterrupted ball:", ref._state["ball"])
print("interrupted   ball:", gm._state["ball"])
print("agree:", all(np.array_equal(np.asarray(ref._state["ball"][k]),
                                   np.asarray(gm._state["ball"][k]))
                    for k in ref._state["ball"]))

# Control: same interrupted run, but restore the counter the recompile wiped.
gm2 = build(False); gm2.run(4)
n = int(gm2._state["_meta"]["step_count"])
gm2.add_external_input("table", "unused", shape=())
gm2.compile()                                   # wipes step_count to 0
gm2._state["_meta"]["step_count"] = jnp.array(n, dtype=jnp.int32)
gm2.run(4)
print("control (counter restored):", gm2._state["ball"])
print("control == uninterrupted  :",
      all(np.array_equal(np.asarray(ref._state["ball"][k]),
                         np.asarray(gm2._state["ball"][k]))
          for k in ref._state["ball"]))
