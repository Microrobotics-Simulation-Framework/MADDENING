"""A wrapper's forwarded static_data and static_data_deps are documented to be
'keyed identically ... so the two line up'.  They are not, when only one of the
two dicts has a key collision."""
import jax.numpy as jnp
from maddening.core.node import SimulationNode
from maddening.core.static_data import StaticArray

class Leaf(SimulationNode):
    def __init__(self, name, deps):
        super().__init__(name, 1.0, alpha=1.0)
        self._deps = deps
    @property
    def static_data(self): return {"table": StaticArray(jnp.zeros(2))}
    def static_data_deps(self): return dict(self._deps)
    def initial_state(self): return {"y": jnp.array(0.0)}
    def update(self, state, bi, dt, *, params=None): return state

class Wrapper(SimulationNode):
    def __init__(self, name, inner_a, inner_b):
        super().__init__(name, 1.0)
        self.inner_a = inner_a
        self.inner_b = inner_b
    def initial_state(self): return {"y": jnp.array(0.0)}
    def update(self, state, bi, dt, *, params=None): return state

# only inner_b declares a dependency, and both publish a static called "table"
w = Wrapper("w", Leaf("A", {}), Leaf("B", {"table": ("alpha",)}))
print("static_data keys     :", sorted(w.static_data))
print("static_data_deps keys:", sorted(w.static_data_deps()))
print("-> the dep is filed under 'table' (A's static); B's static is "
      "'inner_b.table' and has no entry")
