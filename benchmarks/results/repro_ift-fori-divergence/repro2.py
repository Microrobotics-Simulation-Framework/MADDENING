"""fori vs ift on a slowly-contracting group with SMALL interface values.

Hypothesis under test: the interface norm's threshold is hard-coded to
1.0 and its scale is ``atol + rtol*|v|``, so for interface fields whose
magnitude is near ``atol`` the criterion is an ABSOLUTE one and
``tolerance`` is inert.  Both solvers then stop at the same pass, but
``fori`` returns the iterate whose residual passed and ``ift`` returns
one further update -- a difference of a whole residual, which in
relative terms can be percent-sized.
"""
import os, itertools
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

F = jnp.float64


class Flow(SimulationNode):
    """tau <- gain * (SCALE + a * disp)   (interface field, tiny magnitude)"""
    def __init__(self, name, timestep, gain, a, scale):
        super().__init__(name, timestep, gain=jnp.asarray(gain, F),
                         a=jnp.asarray(a, F))
        self._scale = scale

    def initial_state(self):
        return {"tau": jnp.array(0.0, F)}

    def boundary_input_spec(self):
        return {"disp": BoundaryInputSpec(shape=(), description="disp")}

    def update(self, s, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        disp = bi.get("disp", jnp.array(0.0, F))
        return {"tau": p["gain"] * self._scale + p["a"] * disp}


class Struct(SimulationNode):
    """disp <- b * tau"""
    def __init__(self, name, timestep, b):
        super().__init__(name, timestep, b=jnp.asarray(b, F))

    def initial_state(self):
        return {"disp": jnp.array(0.0, F)}

    def boundary_input_spec(self):
        return {"tin": BoundaryInputSpec(shape=(), description="traction")}

    def update(self, s, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        tin = bi.get("tin", jnp.array(0.0, F))
        return {"disp": p["b"] * tin}


def build(solver, *, rho=0.9, scale=1e-6, norm="interface", accel="none",
          max_iterations=8, tolerance=1e-4, atol=1e-8, rtol=1e-6,
          accel_fields=False):
    # a*b = rho  ->  Gauss-Seidel contraction factor rho
    a, b = 1.0, rho
    gm = GraphManager()
    gm.add_node(Flow("flow", 0.01, gain=1.0, a=a, scale=scale))
    gm.add_node(Struct("struct", 0.01, b=b))
    gm.add_edge("flow", "struct", "tau", "tin")
    gm.add_edge("struct", "flow", "disp", "disp")
    kw = dict(convergence_norm=norm, acceleration=accel, solver=solver,
              strict_convergence=False, atol=atol, rtol=rtol,
              diagnostics=True)
    if accel_fields:
        kw["accelerated_fields"] = {"flow": ("tau",), "struct": ("disp",)}
    gm.add_coupling_group(["flow", "struct"], max_iterations=max_iterations,
                          tolerance=tolerance, **kw)
    gm.compile()
    return gm


def measure(solver, n_steps=1, **kw):
    gm = build(solver, **kw)
    base = gm.params
    init = {k: dict(v) for k, v in gm._state.items()}

    def loss(p):
        gm._state = {k: dict(v) for k, v in init.items()}
        out = gm.run_scan(n_steps, params=p)["flow"]["tau"]
        return jnp.sum(out)

    fwd = float(loss(base))
    diag = gm.coupling_diagnostics()
    gm._state = {k: dict(v) for k, v in init.items()}
    g = float(jax.grad(loss)(base)["nodes"]["flow"]["gain"])

    h = 1e-7
    def bump(d):
        p2 = {"nodes": {n: dict(v) for n, v in base["nodes"].items()}}
        p2["nodes"]["flow"]["gain"] = base["nodes"]["flow"]["gain"] + d
        return float(loss(p2))
    fd = (bump(h) - bump(-h)) / (2 * h)
    return fwd, g, fd, diag


def row(label, **kw):
    ff, fg, ffd, fdiag = measure("fori", **kw)
    iff, ig, ifd, idiag = measure("ift", **kw)
    rel = abs(iff - ff) / max(abs(ff), 1e-300)
    k = next(iter(fdiag))
    print(f"{label:<44} rel_fwd={rel:9.3e}  "
          f"fori={ff:.17g} ift={iff:.17g}\n"
          f"{'':44} fori g/fd={abs(fg-ffd)/max(abs(ffd),1e-300):8.2e} "
          f"ift g/fd={abs(ig-ifd)/max(abs(ifd),1e-300):8.2e}  "
          f"fori_diag={fdiag[k]} ift_diag={idiag[k]}")


if __name__ == "__main__":
    print("# analytic fixed point: tau* = scale/(1-rho)")
    for scale in (1e-6, 1e-4, 1.0):
        for rho in (0.9, 0.99):
            row(f"scale={scale:g} rho={rho} interface m=8 tol=1e-4",
                scale=scale, rho=rho)
    print()
    print("# tolerance / iteration invariance at the worst point")
    for tol in (1e-4, 1e-14):
        for m in (8, 40):
            row(f"interface scale=1e-6 rho=0.99 tol={tol:g} m={m}",
                scale=1e-6, rho=0.99, tolerance=tol, max_iterations=m)
    print()
    print("# same graph, l2 norm (tolerance is live there)")
    for tol in (1e-4, 1e-14):
        row(f"l2 scale=1e-6 rho=0.99 tol={tol:g} m=40",
            scale=1e-6, rho=0.99, norm="l2", tolerance=tol, max_iterations=40)
    print()
    print("# atol sweep under the interface norm (the live knob)")
    for atol in (1e-8, 1e-14, 0.0):
        row(f"interface scale=1e-6 rho=0.99 atol={atol:g} m=40",
            scale=1e-6, rho=0.99, atol=atol, max_iterations=40)
    print()
    print("# with iqn-ils acceleration")
    for af in (False, True):
        row(f"interface scale=1e-6 rho=0.99 iqn-ils af={af} m=8",
            scale=1e-6, rho=0.99, accel="iqn-ils", accel_fields=af)
