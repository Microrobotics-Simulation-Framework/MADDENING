"""Static-data CONTENTS change at the same shape: what each path does."""
import jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.static_data import StaticArray

class TableLookup(SimulationNode):
    def __init__(self, name, timestep):
        super().__init__(name, timestep)
        self._lut = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

    @property
    def static_data(self):
        return {"lut": StaticArray(self._lut)}

    def initial_state(self):
        return {"y": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"y": state["y"] + self._lut[0] * dt}

gm = GraphManager(); gm.add_node(TableLookup("n", 1.0)); gm.compile()
print("step 1 (lut[0]=1):", gm.step()["n"]["y"])

node = gm.get_node("n")
node._lut = jnp.array([10.0, 2.0, 3.0], dtype=jnp.float32)   # same shape/dtype
print("dirty after content change?", gm._dirty,
      "static hash changed?", node.static_data_hash() != gm._static_data_hashes["n"])
print("step 2 (should be +10):", gm.step()["n"]["y"])

gm.compile()                       # explicit 'rebuild everything'
print("step 3 after explicit compile():", gm.step()["n"]["y"])

# and run_scan on the same graph
gm2 = GraphManager(); gm2.add_node(TableLookup("n", 1.0)); gm2.compile()
gm2.run_scan(1)
gm2.get_node("n")._lut = jnp.array([10.0, 2.0, 3.0], dtype=jnp.float32)
print("run_scan after content change:", gm2.run_scan(1)["n"]["y"])
gm2.compile()
print("run_scan after explicit compile():", gm2.run_scan(1)["n"]["y"])
