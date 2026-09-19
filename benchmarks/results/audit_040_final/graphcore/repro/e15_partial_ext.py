"""Partially supplied external_inputs: the omitted ones are NOT zero-filled."""
import jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode, BoundaryInputSpec

class Sink(SimulationNode):
    def initial_state(self): return {"y": jnp.array(0.0)}
    def boundary_input_spec(self):
        return {"f": BoundaryInputSpec(shape=(), description="force"),
                "g": BoundaryInputSpec(shape=(), description="other")}
    def update(self, state, boundary_inputs, dt, *, params=None):
        f = boundary_inputs.get("f", jnp.array(-99.0))
        g = boundary_inputs.get("g", jnp.array(-99.0))
        return {"y": f + g}

gm = GraphManager()
gm.add_node(Sink("s", 1.0))
gm.add_external_input("s", "f")
gm.add_external_input("s", "g")
gm.compile()
print("external_inputs=None      ->", gm.step()["s"]["y"])           # expect 0
gm.reset_state()
print("only f given ({'s':{'f':1}}) ->", gm.step({"s": {"f": jnp.array(1.0)}})["s"]["y"])
gm.reset_state()
print("empty dict {}             ->", gm.step({})["s"]["y"])
