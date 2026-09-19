"""F2: the 0.4.0 'error bound' is not a bound on a two-mode contraction.

A linear Gauss-Seidel map with two eigenvalues (a fast one and a slow
one).  While the fast mode still dominates the *step*, rho reads the
fast rate, so 1/(1-rho) is small -- but the remaining distance is
already dominated by the slow mode.  Both the one-step ratio and the
two-step sqrt guard read the fast rate, so the guard does not help.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

RHO = np.array([0.999, 0.2])     # slow mode, fast mode
C   = np.array([1e-5, 1.0])      # so y* = (1e-2, 1.25)
YSTAR = C / (1.0 - RHO)


class A(SimulationNode):
    def initial_state(self):
        return {"x": jnp.zeros(2)}
    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), description="u")}
    def update(self, state, bi, dt, *, params=None):
        u = bi.get("u", jnp.zeros(2))
        return {"x": jnp.asarray(RHO) * u + jnp.asarray(C)}


class B(SimulationNode):
    def initial_state(self):
        return {"y": jnp.zeros(2)}
    def boundary_input_spec(self):
        return {"v": BoundaryInputSpec(shape=(2,), description="v")}
    def update(self, state, bi, dt, *, params=None):
        return {"y": bi.get("v", jnp.zeros(2))}


def scaled_l2(new, old):
    """The group's own l2 residual measure, for two (x, y) state pairs."""
    tot = 0.0
    for n, o in zip(new, old):
        ref = max(np.max(np.abs(n)), np.max(np.abs(o)))
        if ref <= 1e-8:
            continue
        tot += np.sum((np.abs(n - o) / ref) ** 2)
    return float(np.sqrt(tot))


def build(tol, max_iter=60, solver="ift"):
    gm = GraphManager()
    gm.add_node(A("a", 0.01)); gm.add_node(B("b", 0.01))
    gm.add_edge("a", "b", "x", "v")
    gm.add_edge("b", "a", "y", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=max_iter,
                          tolerance=tol, diagnostics=True, solver=solver)
    gm.compile()
    return gm


print(f"exact fixed point y* = {YSTAR}")
print()
hdr = ("tol        iters  residual     amp       error_estimate "
       "bound_valid conv | TRUE dist    est/true")
print(hdr)
for tol in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-7):
    gm = build(tol)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    x = np.asarray(gm.get_node_state("a")["x"])
    y = np.asarray(gm.get_node_state("b")["y"])
    true_d = scaled_l2((x, y), (YSTAR, YSTAR))
    print(f"{tol:<10.0e} {d['iterations']:<6d} {d['residual']:<12.4e} "
          f"{d['amplification']:<9.4g} {d['error_estimate']:<14.4e} "
          f"{str(d['bound_valid']):<11} {str(d['converged']):<4} | "
          f"{true_d:<12.4e} {d['error_estimate']/true_d if true_d else float('nan'):.4g}")
