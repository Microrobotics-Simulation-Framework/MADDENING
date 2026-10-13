"""FINDING: the params pytree silently narrows float64 scalars to float32,
even with jax_enable_x64 on -- and only for *some* spellings of the same value."""
import warnings, jax
print("jax_enable_x64:", jax.config.jax_enable_x64)
import numpy as np, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode

VAL = 0.1234567890123456789

class N(SimulationNode):
    def __init__(self, name, timestep=0.01, **kw):
        super().__init__(name, timestep); self.params = dict(kw)
    def initial_state(self): return {"x": jnp.zeros((), jnp.float64)}
    def update(self, state, boundary_inputs, dt, params=None):
        return {"x": state["x"] + dt * jnp.asarray((params or self.params)["c"]).sum()}

print("isinstance(np.float64(x), float) ->", isinstance(np.float64(VAL), float))
print()
spellings = {
    "python float          ": VAL,
    "np.float64 scalar     ": np.float64(VAL),
    "np.float64 0-d array  ": np.array(VAL, dtype=np.float64),
    "np.float64 1-d array  ": np.array([VAL], dtype=np.float64),
    "list of python floats ": [VAL],
    "jnp float64 array     ": jnp.asarray(VAL, jnp.float64),
}
for label, v in spellings.items():
    gm = GraphManager()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gm.add_node(N("n", 0.01, c=v)); gm.compile()
    leaf = gm.params["nodes"]["n"]["c"]
    got = float(np.asarray(leaf).ravel()[0])
    print(f"  {label} -> dtype={np.dtype(leaf.dtype).name:8s} value={got!r} "
          f"rel_err={abs(got-VAL)/VAL:.2e} warnings={[str(x.message)[:40] for x in w]}")
print()
print("Same numeric value, two dtypes and two answers, with nothing said.")
print("src/maddening/core/node.py:341  `if isinstance(value, float): jnp.asarray(value, jnp.float32)`")
print("src/maddening/core/node.py:354  `if isinstance(value, (list, tuple)): arr.astype(jnp.float32)`")
print("np.float64 IS a subclass of float, so it takes the first branch; an ndarray does not.")
