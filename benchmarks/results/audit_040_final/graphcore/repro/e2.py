import jax, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode

# step vs run vs run_scan agreement
def build():
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=1.0))
    gm.compile()
    return gm

a = build()
for _ in range(5): a.step()
b = build(); b.run(5)
c = build(); c.run_scan(5)
print("step ", a._state["ball"])
print("run  ", b._state["ball"])
print("scan ", c._state["ball"])

# Now: change gm.params between calls, no recompile
d = build()
print("params", d.params)
d.params["nodes"]["ball"]["gravity"] = jnp.array(-1.0)
print("scan after live param edit:", d.run_scan(5)["ball"])
e = build()
e.params["nodes"]["ball"]["gravity"] = jnp.array(-1.0)
for _ in range(5): e.step()
print("step after live param edit:", e._state["ball"])
