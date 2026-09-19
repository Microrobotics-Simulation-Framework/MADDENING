import os, warnings
os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.simplefilter("ignore")
import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/benchmarks/results/audit_040_final/coupling")
import jax, jax.numpy as jnp, numpy as np
from harness import graph
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

print("### C2. do fori and ift also return the same STATE where `iterations` differs?")
for acc in ("none", "iqn-ils"):
    out = {}
    for s in ("fori", "ift"):
        gm = graph(s, b=0.99, max_iterations=6, tolerance=1e-12, acceleration=acc)
        gm.step()
        d = gm.coupling_diagnostics()["flow+struct"]
        out[s] = (float(gm.get_node_state("flow")["tau"]), d["iterations"],
                  d["residual"], d["converged"])
    print(f"    {acc:<9} fori={out['fori']}\n              ift ={out['ift']}")

print()
print("### D. N=40 coupling group: IFT adjoint, gmres vs dense vs FD")
N = 40
KMAT = np.zeros((N, N))
rng = np.random.default_rng(0)
KMAT = 0.9 * rng.random((N, N)) / N     # spectral radius ~0.45, dense


class Big(SimulationNode):
    def __init__(self, name, dt, gain=1.0):
        super().__init__(name, dt, gain=gain)
    def initial_state(self):
        return {"x": jnp.zeros(N)}
    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(N,), description="u")}
    def update(self, state, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = bi.get("u", jnp.zeros(N))
        return {"x": jnp.asarray(KMAT) @ u + p["gain"] * jnp.ones(N)}


class Pass(SimulationNode):
    def initial_state(self):
        return {"y": jnp.zeros(N)}
    def boundary_input_spec(self):
        return {"v": BoundaryInputSpec(shape=(N,), description="v")}
    def update(self, state, bi, dt, *, params=None):
        return {"y": bi.get("v", jnp.zeros(N))}


def big(linear_solver="gmres", solver="ift"):
    gm = GraphManager()
    gm.add_node(Big("big", 0.01)); gm.add_node(Pass("p", 0.01))
    gm.add_edge("big", "p", "x", "v"); gm.add_edge("p", "big", "y", "u")
    gm.add_coupling_group(["big", "p"], max_iterations=80, tolerance=1e-9,
                          diagnostics=True, solver=solver,
                          linear_solver=linear_solver)
    gm.compile()
    return gm


# analytic: x* = (I - K)^-1 * gain * 1 ; d sum(x*)/d gain = sum((I-K)^-1 1)
exact = float(np.sum(np.linalg.solve(np.eye(N) - KMAT, np.ones(N))))
print(f"    analytic d sum(x*)/d gain = {exact:.6f}")
for ls in ("gmres", "dense"):
    def loss(p, ls=ls):
        return jnp.sum(big(ls).run_scan(1, params=p)["big"]["x"])
    base = big(ls).params
    g = float(jax.grad(loss)(base)["nodes"]["big"]["gain"])
    h = 1e-2
    def sh(d):
        p = {"nodes": {n: dict(v) for n, v in base["nodes"].items()}}
        p["nodes"]["big"]["gain"] = base["nodes"]["big"]["gain"] + d
        return float(loss(p))
    fd = (sh(h) - sh(-h)) / (2 * h)
    print(f"    linear_solver={ls:<6} grad={g:12.6f}  fd={fd:12.6f}  "
          f"|grad-exact|/exact={abs(g-exact)/abs(exact):.3e}")
