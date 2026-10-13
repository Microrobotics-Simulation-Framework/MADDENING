"""Ingredient minimality: which of the suspected ingredients are needed?"""
import os, itertools
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

F = jnp.float64


class Flow(SimulationNode):
    def __init__(self, name, timestep, gain, a, scale, int_leaf):
        super().__init__(name, timestep, gain=jnp.asarray(gain, F),
                         a=jnp.asarray(a, F))
        self._scale, self._int = scale, int_leaf

    def initial_state(self):
        s = {"tau": jnp.array(0.0, F)}
        if self._int:
            s["i_step"] = jnp.array(0, jnp.int32)
        return s

    def boundary_input_spec(self):
        return {"disp": BoundaryInputSpec(shape=(), description="disp")}

    def update(self, s, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        disp = bi.get("disp", jnp.array(0.0, F))
        out = {"tau": p["gain"] * self._scale + p["a"] * disp}
        if self._int:
            out["i_step"] = s["i_step"] + jnp.int32(1)
        return out


class Struct(SimulationNode):
    def __init__(self, name, timestep, b):
        super().__init__(name, timestep, b=jnp.asarray(b, F))

    def initial_state(self):
        return {"disp": jnp.array(0.0, F)}

    def boundary_input_spec(self):
        return {"tin": BoundaryInputSpec(shape=(), description="traction")}

    def update(self, s, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"disp": p["b"] * bi.get("tin", jnp.array(0.0, F))}


def run(solver, *, int_leaf, accel_fields, norm, accel, mode="gauss-seidel",
        rho=0.0214, scale=1e-9, m=8, tol=1e-4):
    gm = GraphManager()
    gm.add_node(Flow("flow", 0.01, gain=1.0, a=1.0, scale=scale, int_leaf=int_leaf))
    gm.add_node(Struct("struct", 0.01, b=rho))
    gm.add_edge("flow", "struct", "tau", "tin")
    gm.add_edge("struct", "flow", "disp", "disp")
    kw = dict(convergence_norm=norm, acceleration=accel, solver=solver,
              strict_convergence=False, diagnostics=True, iteration_mode=mode)
    if accel_fields:
        kw["accelerated_fields"] = {"flow": ("tau",), "struct": ("disp",)}
    gm.add_coupling_group(["flow", "struct"], max_iterations=m, tolerance=tol, **kw)
    gm.compile()
    out = gm.run_scan(1)
    d = gm.coupling_diagnostics()
    return float(out["flow"]["tau"]), d[next(iter(d))]


print(f"{'int':>3} {'af':>2} {'norm':<10} {'accel':<8} {'mode':<13} "
      f"{'fori':>22} {'ift':>22} {'rel':>10} conv")
for int_leaf, af, norm, accel, mode in itertools.product(
        [True, False], [True, False], ["interface", "l2", "mixed"],
        ["none", "iqn-ils", "aitken", "fixed"], ["gauss-seidel", "jacobi"]):
    if af and accel not in ("iqn-ils", "iqn-imvj"):
        continue
    kw = dict(int_leaf=int_leaf, accel_fields=af, norm=norm, accel=accel, mode=mode)
    # l2 needs a tolerance that fires early on this tiny-magnitude state
    if norm == "l2":
        kw["tol"] = 1e-8
    try:
        f, fd = run("fori", **kw)
        i, idg = run("ift", **kw)
    except Exception as e:
        print(f"{int(int_leaf):>3} {int(af):>2} {norm:<10} {accel:<8} {mode:<13} ERROR {e!r:.70}")
        continue
    rel = abs(i - f) / max(abs(f), 1e-300)
    print(f"{int(int_leaf):>3} {int(af):>2} {norm:<10} {accel:<8} {mode:<13} "
          f"{f:22.17g} {i:22.17g} {rel:10.4%} {fd['converged']}/{idg['converged']}")
