"""F2c: `gradient_error_bound` does not bound the adjoint/FD gap on the
two-mode contraction, exactly as `error_estimate` does not bound the
distance.  Same assertion as
tests/core/test_coupling_error_bound.py::test_the_gradient_trust_bound_bounds_the_adjoint_finite_difference_gap
"""
import os, warnings
os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.simplefilter("ignore")
import jax, jax.numpy as jnp, numpy as np
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

RHO = np.array([0.999, 0.2])
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
        u = bi.get("u", jnp.zeros(2))
        return {"x": jnp.asarray(RHO) * u + p["gain"] * jnp.asarray(C)}


class B(SimulationNode):
    def initial_state(self):
        return {"y": jnp.zeros(2)}
    def boundary_input_spec(self):
        return {"v": BoundaryInputSpec(shape=(2,), description="v")}
    def update(self, state, bi, dt, *, params=None):
        return {"y": bi.get("v", jnp.zeros(2))}


def build(tol=1e-4):
    gm = GraphManager()
    gm.add_node(A("a", 0.01)); gm.add_node(B("b", 0.01))
    gm.add_edge("a", "b", "x", "v"); gm.add_edge("b", "a", "y", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=60, tolerance=tol,
                          diagnostics=True, solver="ift")
    gm.compile()
    return gm


TOL = 1e-4
def loss(p):
    return jnp.sum(build(TOL).run_scan(1, params=p)["a"]["x"])

base = build(TOL).params
analytic = float(jax.grad(loss)(base)["nodes"]["a"]["gain"])
h = 1e-2
def sh(d):
    p = {"nodes": {n: dict(v) for n, v in base["nodes"].items()}}
    p["nodes"]["a"]["gain"] = base["nodes"]["a"]["gain"] + d
    return float(loss(p))
fd = (sh(h) - sh(-h)) / (2 * h)

gm = build(TOL); gm.step()
d = gm.coupling_diagnostics()["a+b"]
exact = float(np.sum(C / (1.0 - RHO)))
print(f"exact d sum(x*)/d gain            = {exact:.6f}")
print(f"IFT adjoint                       = {analytic:.6f}")
print(f"finite difference of the forward  = {fd:.6f}")
print(f"|adjoint - fd|                    = {abs(analytic-fd):.6f}")
print(f"bound_valid                       = {d['bound_valid']}")
print(f"converged                         = {d['converged']}")
print(f"gradient_error_bound              = {d['gradient_error_bound']:.6e}")
ok = abs(analytic - fd) <= max(d["gradient_error_bound"], 1e-5)
print(f"assertion |adjoint-fd| <= bound   = {ok}   "
      f"(exceeded by {abs(analytic-fd)/max(d['gradient_error_bound'],1e-5):.1f}x)")
