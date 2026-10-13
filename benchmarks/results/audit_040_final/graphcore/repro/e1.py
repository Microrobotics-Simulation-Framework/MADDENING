import jax, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode

def build():
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=1.0))
    gm.compile()
    return gm

# --- 1. aliasing of returned state ---
gm = build()
s = gm.step()
print("returned dict is internal dict?", s is gm._state, "inner alias?", s["ball"] is gm._state["ball"])
s["ball"]["position"] = jnp.array(999.0)
print("after caller mutates returned state, internal position =", gm._state["ball"]["position"])
print("next step position =", gm.step()["ball"]["position"])

# get_node_state
gm2 = build()
ns = gm2.get_node_state("ball")
print("get_node_state is live ref?", ns is gm2._state["ball"])
ns["position"] = jnp.array(-42.0)
print("internal after mutating get_node_state:", gm2._state["ball"]["position"])

# run_scan_with_history history aliasing
gm3 = build()
final, hist = gm3.run_scan_with_history(3)
print("hist inner is final inner?", hist["ball"] is final["ball"])
