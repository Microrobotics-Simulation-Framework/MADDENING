"""How much does a grid-sized delivered value loosen the interface norm?

A "robot" with m markers coupled two ways to a 1-D "grid" of N cells:
  gather  grid.u (N)  --G-->  robot.u_at (m)      (linear interpolation at the markers)
  scatter robot.f (m) --G^T--> grid.F (N)         (its transpose)
  robot: f <- a * u_at + b          grid: u <- u0 + c * F
The exact fixed point is a small linear solve.  The same coupled problem is
built three ways and solved with the same tolerance:
  edge-mapped   both mappings on the edges; the interface norm reads what
                each edge DELIVERS (m numbers for the gather, N for the scatter)
  marker-side   the scatter lives inside the grid node, so both internal
                edges carry m numbers
  mixed         edge-mapped, convergence_norm="mixed" (whole state)
Reported: passes taken, the residual the solve reports, and the true relative
error of the marker forces at exit, over the tolerance asked for.
"""
import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from maddening import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.coupling.mapping import matrix_mapping

F = jnp.float64

def interp_matrix(N, m, rng):
    pos = np.sort(rng.uniform(0.2 * N, 0.8 * N, m))
    G = np.zeros((m, N))
    i = np.floor(pos).astype(int); w = pos - i
    G[np.arange(m), i] = 1 - w; G[np.arange(m), i + 1] = w
    return G

class Robot(SimulationNode):
    def __init__(self, name, dt, m, a, b):
        super().__init__(name, dt); self.m, self.a, self.b = m, a, np.asarray(b)
    def initial_state(self): return {"f": jnp.zeros(self.m, F)}
    def boundary_input_spec(self):
        return {"u_at": BoundaryInputSpec(shape=(self.m,), dtype=F, default=jnp.zeros(self.m, F))}
    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"f": self.a * boundary_inputs["u_at"] + jnp.asarray(self.b, F)}

class Grid(SimulationNode):
    """u <- u0 + c * F, with F delivered (edge-mapped) or scattered here from marker forces."""
    def __init__(self, name, dt, N, c, u0, scatter=None):
        super().__init__(name, dt); self.N, self.c, self.u0 = N, c, np.asarray(u0)
        self.S = None if scatter is None else jnp.asarray(scatter, F)
    def initial_state(self): return {"u": jnp.asarray(self.u0, F)}
    def boundary_input_spec(self):
        if self.S is None:
            return {"F": BoundaryInputSpec(shape=(self.N,), dtype=F, default=jnp.zeros(self.N, F))}
        m = self.S.shape[1]
        return {"f_markers": BoundaryInputSpec(shape=(m,), dtype=F, default=jnp.zeros(m, F))}
    def update(self, state, boundary_inputs, dt, *, params=None):
        Ff = boundary_inputs["F"] if self.S is None else self.S @ boundary_inputs["f_markers"]
        return {"u": jnp.asarray(self.u0, F) + self.c * Ff}

def build(kind, N, m, G, a, b, c, u0, norm, schedule, rtol, cap=400):
    gm = GraphManager()
    gm.add_node(Robot("robot", 0.01, m, a, b))
    if kind == "marker-side":
        gm.add_node(Grid("grid", 0.01, N, c, u0, scatter=G.T))
        gm.add_edge("robot", "grid", "f", "f_markers")
    else:
        gm.add_node(Grid("grid", 0.01, N, c, u0))
        gm.add_edge("robot", "grid", "f", "F", mapping=matrix_mapping(G.T))
    gm.add_edge("grid", "robot", "u", "u_at", mapping=matrix_mapping(G))
    gm.add_coupling_group(["robot", "grid"], max_iterations=cap, rtol=rtol,
                          convergence_norm=norm, iteration_mode=schedule, solver="ift")
    gm.compile()
    return gm

rows = []
rng = np.random.default_rng(0)
m, a, c, rtol = 30, 0.9, 1.0, 1e-4          # contraction rate about a*c*||G G^T|| < 1
for N in (10**3, 10**4, 10**5, 10**6):
    G = interp_matrix(N, m, rng)
    scale = 0.6 / np.linalg.norm(G @ G.T, 2)     # the loop's gain: 0.6 * a
    Gs = G * np.sqrt(scale)
    b = rng.uniform(0.5, 1.5, m); u0 = rng.uniform(0.5, 1.5, N)
    f_exact = np.linalg.solve(np.eye(m) - a * c * Gs @ Gs.T, a * Gs @ u0 + b)
    for schedule in ("gauss-seidel", "jacobi"):
        for kind, norm in (("edge-mapped", "interface"), ("marker-side", "interface"), ("edge-mapped", "mixed")):
            try:
                gm = build(kind, N, m, Gs, a, b, c, u0, norm, schedule, rtol)
                gm.step()
                (key, d), = gm.coupling_diagnostics().items()
                f = np.asarray(gm._state["robot"]["f"])
                err = float(np.max(np.abs(f - f_exact) / np.abs(f_exact)))
                rows.append((N, schedule, kind, norm, int(d["iterations"]), float(d["residual"]), bool(d["converged"]), err, err / rtol))
            except Exception as e:
                rows.append((N, schedule, kind, norm, -1, float("nan"), False, float("nan"), float("nan")))
                print("FAILED", N, schedule, kind, norm, type(e).__name__, str(e)[:200], file=sys.stderr)
            print(rows[-1], flush=True)
import json
json.dump([dict(zip(("N", "schedule", "build", "norm", "passes", "residual", "converged", "max_rel_error", "error_over_tolerance"), r)) for r in rows], open("result.json", "w"), indent=1)
