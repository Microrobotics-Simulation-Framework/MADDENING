import os, warnings
os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.simplefilter("ignore")
import jax, jax.numpy as jnp, numpy as np
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode


def make(RHO, CC, ls="gmres"):
    n = len(RHO)
    class A(SimulationNode):
        def __init__(s, name, dt, gain=1.0): super().__init__(name, dt, gain=gain)
        def initial_state(s): return {"x": jnp.zeros(n)}
        def boundary_input_spec(s):
            return {"u": BoundaryInputSpec(shape=(n,), description="u")}
        def update(s, st, bi, dt, *, params=None):
            p = s.params if params is None else {**s.params, **params}
            return {"x": jnp.asarray(RHO) * bi.get("u", jnp.zeros(n))
                    + p["gain"] * jnp.asarray(CC)}
    class B(SimulationNode):
        def initial_state(s): return {"y": jnp.zeros(n)}
        def boundary_input_spec(s):
            return {"v": BoundaryInputSpec(shape=(n,), description="v")}
        def update(s, st, bi, dt, *, params=None):
            return {"y": bi.get("v", jnp.zeros(n))}
    gm = GraphManager()
    gm.add_node(A("a", 0.01)); gm.add_node(B("b", 0.01))
    gm.add_edge("a", "b", "x", "v"); gm.add_edge("b", "a", "y", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=60, tolerance=1e-4,
                          diagnostics=True, solver="ift", linear_solver=ls)
    gm.compile()
    return gm


def try_grad(RHO, CC, ls="gmres", mode="grad"):
    def loss(p):
        return jnp.sum(make(RHO, CC, ls).run_scan(1, params=p)["a"]["x"])
    base = make(RHO, CC, ls).params
    try:
        if mode == "grad":
            v = float(jax.grad(loss)(base)["nodes"]["a"]["gain"])
        else:
            t = jax.tree.map(jnp.zeros_like, base)
            t["nodes"]["a"]["gain"] = jnp.asarray(1.0)
            v = float(jax.jvp(loss, (base,), (t,))[1])
        return f"{v:.6g}"
    except Exception as e:
        return "BREAKDOWN"


CC = np.array([1e-5, 1.0])
print("forward step always fine; only the tangent/adjoint solve breaks.")
print(f"{'rho_slow':>9} {'cond~':>8} {'grad/gmres':>12} {'jvp/gmres':>12} "
      f"{'grad/dense':>12} {'env dense':>10}")
for rs in (0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995):
    RHO = np.array([rs, 0.2])
    cond = 0.8 / (1 - rs)
    g = try_grad(RHO, CC, "gmres", "grad")
    j = try_grad(RHO, CC, "gmres", "jvp")
    d = try_grad(RHO, CC, "dense", "grad")
    os.environ["MADDENING_IFT_DENSE_SOLVE"] = "1"
    e = try_grad(RHO, CC, "gmres", "grad")
    del os.environ["MADDENING_IFT_DENSE_SOLVE"]
    print(f"{rs:9} {cond:8.0f} {g:>12} {j:>12} {d:>12} {e:>10}")

print()
print("forward only, rho_slow=0.999 (no derivative):")
gm = make(np.array([0.999, 0.2]), CC); gm.step()
print("   state x =", np.asarray(gm.get_node_state("a")["x"]),
      " diagnostics =", gm.coupling_diagnostics()["a+b"])
