"""Shared fixture: a two-node affine cycle, configurable."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode


class Flow(SimulationNode):
    """tau <- gain*scale + disp  (optionally with an int leaf)."""
    def __init__(self, name, timestep, gain=1.0, scale=1.0, int_leaf=False):
        super().__init__(name, timestep, gain=gain)
        self._scale = scale
        self._int_leaf = int_leaf

    def initial_state(self):
        s = {"tau": jnp.asarray(0.0)}
        if self._int_leaf:
            s["i_step"] = jnp.asarray(0, jnp.int32)
        return s

    def boundary_input_spec(self):
        return {"disp": BoundaryInputSpec(shape=(), description="d")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        disp = boundary_inputs.get("disp", jnp.asarray(0.0))
        out = {"tau": p["gain"] * self._scale + disp}
        if self._int_leaf:
            out["i_step"] = state["i_step"] + jnp.asarray(1, jnp.int32)
        return out


class Struct(SimulationNode):
    def __init__(self, name, timestep, b=0.5):
        super().__init__(name, timestep, b=b)

    def initial_state(self):
        return {"disp": jnp.asarray(0.0)}

    def boundary_input_spec(self):
        return {"tin": BoundaryInputSpec(shape=(), description="t")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        tin = boundary_inputs.get("tin", jnp.asarray(0.0))
        return {"disp": p["b"] * tin}


def graph(solver="ift", *, b=0.5, scale=1.0, int_leaf=False, **kw):
    gm = GraphManager()
    gm.add_node(Flow("flow", 0.01, scale=scale, int_leaf=int_leaf))
    gm.add_node(Struct("struct", 0.01, b=b))
    gm.add_edge("flow", "struct", "tau", "tin")
    gm.add_edge("struct", "flow", "disp", "disp")
    kw.setdefault("diagnostics", True)
    gm.add_coupling_group(["flow", "struct"], solver=solver, **kw)
    gm.compile()
    return gm
