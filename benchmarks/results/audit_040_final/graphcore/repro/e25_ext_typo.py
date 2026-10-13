"""external_inputs is neither completed from the declarations nor validated."""
import jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode, BoundaryInputSpec

class Sink(SimulationNode):
    def initial_state(self): return {"y": jnp.array(0.0)}
    def boundary_input_spec(self):
        return {"f": BoundaryInputSpec(shape=(), description="force")}
    def update(self, state, bi, dt, *, params=None):
        return {"y": state["y"] + bi.get("f", jnp.array(0.0))}

gm = GraphManager(); gm.add_node(Sink("s", 1.0))
gm.add_external_input("s", "f"); gm.compile()
for tag, ext in (("correct      ", {"s": {"f": jnp.array(7.0)}}),
                 ("field typo   ", {"s": {"force": jnp.array(7.0)}}),
                 ("node typo    ", {"S": {"f": jnp.array(7.0)}}),
                 ("undeclared    ", {"s": {"f": jnp.array(7.0), "zzz": jnp.array(1.0)}})):
    gm.reset_state()
    print(tag, "->", gm.step(ext)["s"]["y"])

# set_node_state accepts any pytree
gm.set_node_state("s", {"typo": jnp.array(1.0)})
print("set_node_state({'typo':...}) accepted; state =", gm.get_node_state("s"))
try:
    gm.step()
    print("step after bad set_node_state: OK (!)")
except Exception as e:
    print("step after bad set_node_state:", type(e).__name__, str(e).splitlines()[0][:100])
