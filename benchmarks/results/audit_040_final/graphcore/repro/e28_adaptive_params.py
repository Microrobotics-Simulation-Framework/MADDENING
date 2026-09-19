"""docs/user_guide/parameters.md: 'Every run method takes an optional params='.
run_adaptive and run_adaptive_scan do not."""
import inspect, jax, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode

for m in ("step","run","run_scan","run_scan_with_history","run_sweep",
          "run_adaptive","run_adaptive_scan"):
    has = "params" in inspect.signature(getattr(GraphManager, m)).parameters
    print(f"{m:24} {'params=' if has else 'NO params='}")

gm = GraphManager(); gm.add_node(BallNode("ball", 0.01, initial_position=1.0)); gm.compile()
try:
    gm.run_adaptive_scan(t_end=0.05, max_steps=8, params=gm.params)
except TypeError as e:
    print("run_adaptive_scan(params=...) ->", type(e).__name__, str(e)[:90])

# the only route: write the tracer into gm.params
def loss(g):
    gm.params["nodes"]["ball"]["gravity"] = g
    return gm.run_adaptive_scan(t_end=0.05, max_steps=8)[0]["ball"]["position"]
print("grad via gm.params mutation:", float(jax.grad(loss)(jnp.array(-9.81))))
print("gm.params left holding a", type(gm.params["nodes"]["ball"]["gravity"]).__name__)
