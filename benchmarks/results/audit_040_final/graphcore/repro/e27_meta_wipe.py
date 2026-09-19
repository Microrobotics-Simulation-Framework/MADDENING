"""compile() wipes _meta on BOTH branches: multi-rate replaces it with a fresh
step_count, uniform-rate pops it outright.  Coupling warm starts go with it."""
import numpy as np, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

gm = GraphManager()
gm.add_node(SpringDamperNode("a", 0.001, initial_position=0.0))
gm.add_node(SpringDamperNode("b", 0.001, initial_position=2.0))
gm.add_edge("a","b","position","anchor_position")
gm.add_edge("b","a","position","anchor_position")
gm.add_coupling_group(["a","b"], tolerance=1e-10, max_iterations=20,
                      predictor="quadratic", acceleration="iqn-imvj",
                      diagnostics=True)
gm.compile()
gm.run(30)
before = {k: np.asarray(v).copy() for k, v in gm._state["_meta"].items()}
print("pred_count before recompile:", int(before["coupling_a+b_pred_count"]))
print("IQN V norm before          :", float(np.linalg.norm(before["coupling_a+b_V"])))
node_state = {n: dict(d) for n, d in gm._state.items() if n != "_meta"}

gm.add_external_input("a", "unused", shape=())   # unrelated edit -> recompile
gm.compile()
after = {k: np.asarray(v) for k, v in gm._state["_meta"].items()}
print("pred_count after recompile :", int(after["coupling_a+b_pred_count"]))
print("IQN V norm after           :", float(np.linalg.norm(after["coupling_a+b_V"])))
print("node state preserved       :",
      all(np.array_equal(np.asarray(gm._state[n][f]), np.asarray(v))
          for n, d in node_state.items() for f, v in d.items()))
