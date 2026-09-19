"""The documented escape hatch -- freeze the parameter -- does not work
when it is done with the graph-level API gm.set_param_spec()."""
import jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.params import ParamSpec
from maddening.core.static_data import StaticArray


class ScaledNode(SimulationNode):
    def __init__(self, name, timestep, alpha=1.0):
        super().__init__(name, timestep, alpha=alpha)
        self._table = jnp.array([alpha], dtype=jnp.float32)

    @property
    def static_data(self):
        return {"table": StaticArray(self._table)}

    def static_data_deps(self):
        return {"table": ("alpha",)}
    # NOTE: no param_specs() override -- alpha is trainable by default

    def initial_state(self):
        return {"y": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = {**self.params, **(params or {})}
        return {"y": state["y"] + (p["alpha"] + self._table[0]) * dt}


gm = GraphManager()
gm.add_node(ScaledNode("n", 1.0, alpha=1.0))
try:
    gm.compile()
    print("compile: accepted")
except ValueError as e:
    print("compile REFUSES (expected):", str(e)[:70], "...")

# Documented way out #1: "declare alpha as ParamSpec(trainable=False)".
gm.set_param_spec("n", "alpha", ParamSpec(trainable=False))
print("graph param_specs says trainable:", gm.param_specs()["nodes"]["n"]["alpha"].trainable)
print("trainable_mask says             :", gm.trainable_mask()["nodes"]["n"]["alpha"])
try:
    gm.compile()
    print("compile after graph-level freeze: accepted")
except ValueError as e:
    print("compile after graph-level freeze STILL REFUSES:", str(e)[:70], "...")
