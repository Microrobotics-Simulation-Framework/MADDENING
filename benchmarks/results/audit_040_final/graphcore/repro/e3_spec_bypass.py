"""compile() refuses a static derived from a TRAINABLE param.
set_param_spec() can flip a param to trainable AFTER compile without dirtying
the graph, so the refusal never re-runs and the gradient is silently wrong."""
import jax, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.params import ParamSpec
from maddening.core.static_data import StaticArray


class ScaledNode(SimulationNode):
    """y' = y + (alpha + table[0]) * dt ; table = [alpha] built in __init__."""
    def __init__(self, name, timestep, alpha=1.0):
        super().__init__(name, timestep, alpha=alpha)
        self._table = jnp.array([alpha], dtype=jnp.float32)   # derived from alpha

    @property
    def static_data(self):
        return {"table": StaticArray(self._table)}

    def static_data_deps(self):
        return {"table": ("alpha",)}

    def param_specs(self):
        return {**super().param_specs(), "alpha": ParamSpec(trainable=False)}

    def initial_state(self):
        return {"y": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = {**self.params, **(params or {})}
        return {"y": state["y"] + (p["alpha"] + self._table[0]) * dt}


def make(alpha):
    gm = GraphManager()
    gm.add_node(ScaledNode("n", 1.0, alpha=alpha))
    gm.compile()
    return gm

gm = make(1.0)
print("compile with alpha frozen: OK")

# Legitimate user action: unfreeze alpha so sysid can fit it.
gm.set_param_spec("n", "alpha", ParamSpec())   # trainable=True (default)
print("trainable_mask after unfreeze:", gm.trainable_mask()["nodes"]["n"]["alpha"])
print("graph dirty?", gm._dirty)

# --- the gradient the graph now reports ---
def loss(params):
    gm.reset_state()
    return gm.run_scan(1, params=params)["n"]["y"]

p = jax.tree.map(lambda x: x, gm.params)
g = jax.grad(loss)(p)["nodes"]["n"]["alpha"]
print("d y / d alpha reported :", float(g))

# --- the true derivative (finite difference over rebuilt graphs) ---
h = 1e-3
def f(a):
    g2 = make(a); g2.reset_state()
    return float(g2.run_scan(1)["n"]["y"])
fd = (f(1.0 + h) - f(1.0 - h)) / (2 * h)
print("d y / d alpha truth    :", fd)

# --- and what compile() would have said ---
gm._dirty = True
try:
    gm.compile()
    print("recompile: accepted (!)")
except ValueError as e:
    print("recompile REFUSES:", str(e)[:90], "...")
