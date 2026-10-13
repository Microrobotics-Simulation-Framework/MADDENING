"""Minimal reproducer: jax.grad through the DEFAULT coupling solver
(solver="ift", linear_solver="gmres") raises a runtime error on a
4-DOF stiff group.  jax.jvp on the same graph is fine, and
linear_solver="dense" is fine, so it is the adjoint (transpose) GMRES
solve that breaks down.

  PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu python repro_gmres_breakdown.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

RHO = np.array([0.999, 0.2])        # two contraction modes
C = np.array([1e-5, 1.0])


class A(SimulationNode):
    def __init__(self, name, dt, gain=1.0):
        super().__init__(name, dt, gain=gain)
    def initial_state(self):
        return {"x": jnp.zeros(2)}
    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), description="u")}
    def update(self, state, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": jnp.asarray(RHO) * bi.get("u", jnp.zeros(2))
                + p["gain"] * jnp.asarray(C)}


class B(SimulationNode):
    def initial_state(self):
        return {"y": jnp.zeros(2)}
    def boundary_input_spec(self):
        return {"v": BoundaryInputSpec(shape=(2,), description="v")}
    def update(self, state, bi, dt, *, params=None):
        return {"y": bi.get("v", jnp.zeros(2))}


def build(linear_solver="gmres"):
    gm = GraphManager()
    gm.add_node(A("a", 0.01)); gm.add_node(B("b", 0.01))
    gm.add_edge("a", "b", "x", "v")
    gm.add_edge("b", "a", "y", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=60, tolerance=1e-4,
                          diagnostics=True, solver="ift",
                          linear_solver=linear_solver)
    gm.compile()
    return gm


EXACT = float(np.sum(C / (1.0 - RHO)))
for ls in ("dense", "gmres"):
    def loss(p, ls=ls):
        return jnp.sum(build(ls).run_scan(1, params=p)["a"]["x"])
    base = build(ls).params
    try:
        g = float(jax.grad(loss)(base)["nodes"]["a"]["gain"])
        print(f"linear_solver={ls:<6} jax.grad -> {g:.6f}  (exact {EXACT:.6f})")
    except Exception as e:
        print(f"linear_solver={ls:<6} jax.grad -> RAISED {type(e).__name__}")
        print("   ", [l for l in str(e).splitlines()
                      if "breakdown" in l.lower()][:1])
