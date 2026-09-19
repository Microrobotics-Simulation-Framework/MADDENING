"""Gradients through run_scan: the entry point writes the traced result back
into gm._state, so the graph is left holding tracers after jax.grad returns."""
import jax, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode

gm = GraphManager()
gm.add_node(BallNode("ball", 0.01, initial_position=1.0))
gm.compile()

def loss(p):
    return gm.run_scan(5, params=p)["ball"]["position"]

g = jax.grad(loss)(jax.tree.map(lambda x: x, gm.params))
print("grad d position / d gravity:", float(g["nodes"]["ball"]["gravity"]))
print("type of gm._state['ball']['position'] after grad:",
      type(gm._state["ball"]["position"]).__name__)
try:
    s = gm.step()
    print("step after grad:", s["ball"]["position"])
except Exception as e:
    print("step after grad FAILED:", type(e).__name__, str(e)[:250])
