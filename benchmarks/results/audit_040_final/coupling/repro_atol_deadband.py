"""Reproducer: the 0.4.0 `atol` dead band is READ under
convergence_norm="l2" (the default), where CouplingGroup's docstring
and its own inert-knob UserWarning both say it is ignored.

Two consequences, both silent:
  (1) a group all of whose float fields sit below the DEFAULT
      atol=1e-8 converges after one pass with residual 0.0;
  (2) a group with one large and one small field converges on the
      large one while the small one is still 50% from its fixed point.

  PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu python repro_atol_deadband.py
"""
import os, warnings
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax.numpy as jnp, numpy as np
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

B = 0.5


class Flow(SimulationNode):
    """tau_big <- 1 + disp_big ;  tau_small <- SMALL + disp_small"""
    def __init__(self, name, dt, small=1e-9, two_fields=True):
        super().__init__(name, dt)
        self._small = small
        self._two = two_fields
    def initial_state(self):
        s = {"tau_small": jnp.asarray(0.0)}
        if self._two:
            s["tau_big"] = jnp.asarray(0.0)
        return s
    def boundary_input_spec(self):
        d = {"d_small": BoundaryInputSpec(shape=(), description="s")}
        if self._two:
            d["d_big"] = BoundaryInputSpec(shape=(), description="b")
        return d
    def update(self, state, bi, dt, *, params=None):
        out = {"tau_small": self._small + bi.get("d_small", jnp.asarray(0.0))}
        if self._two:
            out["tau_big"] = 1.0 + bi.get("d_big", jnp.asarray(0.0))
        return out


class Struct(SimulationNode):
    def __init__(self, name, dt, b_small=B, b_big=0.0, two_fields=True):
        super().__init__(name, dt)
        self._bs, self._bb, self._two = b_small, b_big, two_fields
    def initial_state(self):
        s = {"disp_small": jnp.asarray(0.0)}
        if self._two:
            s["disp_big"] = jnp.asarray(0.0)
        return s
    def boundary_input_spec(self):
        d = {"t_small": BoundaryInputSpec(shape=(), description="s")}
        if self._two:
            d["t_big"] = BoundaryInputSpec(shape=(), description="b")
        return d
    def update(self, state, bi, dt, *, params=None):
        out = {"disp_small": self._bs * bi.get("t_small", jnp.asarray(0.0))}
        if self._two:
            out["disp_big"] = self._bb * bi.get("t_big", jnp.asarray(0.0))
        return out


def build(small, two_fields, **group_kw):
    gm = GraphManager()
    gm.add_node(Flow("flow", 0.01, small=small, two_fields=two_fields))
    gm.add_node(Struct("struct", 0.01, two_fields=two_fields))
    gm.add_edge("flow", "struct", "tau_small", "t_small")
    gm.add_edge("struct", "flow", "disp_small", "d_small")
    if two_fields:
        gm.add_edge("flow", "struct", "tau_big", "t_big")
        gm.add_edge("struct", "flow", "disp_big", "d_big")
    gm.add_coupling_group(["flow", "struct"], max_iterations=40,
                          tolerance=1e-10, diagnostics=True, **group_kw)
    gm.compile()
    return gm


print("(1) every field below the default atol=1e-8, default l2 norm")
print(f"{'field scale':>12} {'exact':>12} {'returned':>12} {'rel err':>9} "
      f"{'iters':>5} {'residual':>9} conv")
for small in (1e-6, 1e-8, 1e-9, 1e-12):
    gm = build(small, two_fields=False)
    gm.step()
    got = float(gm.get_node_state("flow")["tau_small"])
    exact = small / (1 - B)
    d = gm.coupling_diagnostics()["flow+struct"]
    print(f"{small:12g} {exact:12.4g} {got:12.4g} "
          f"{abs(got-exact)/exact:9.2%} {d['iterations']:5d} "
          f"{d['residual']:9.2e} {d['converged']}")

print()
print("(2) one O(1) field and one O(1e-9) field in the same group")
gm = build(1e-9, two_fields=True)
gm.step()
d = gm.coupling_diagnostics()["flow+struct"]
big = float(gm.get_node_state("flow")["tau_big"])
sml = float(gm.get_node_state("flow")["tau_small"])
print(f"    tau_big   = {big:.6g}  (exact 1.0,       rel err {abs(big-1.0):.2e})")
print(f"    tau_small = {sml:.6g}  (exact {1e-9/(1-B):.4g}, "
      f"rel err {abs(sml-1e-9/(1-B))/(1e-9/(1-B)):.2%})")
print(f"    diagnostics = {d}")

print()
print("(3) the warning the library emits for the knob that would fix it")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    build(1e-9, two_fields=False, atol=1e-12)
for x in w:
    if "atol" in str(x.message):
        print("   ", " ".join(str(x.message).split()))
print()
print("(3b) and with atol lowered, the same group converges correctly:")
gm = build(1e-9, two_fields=False, atol=1e-12)
gm.step()
got = float(gm.get_node_state("flow")["tau_small"])
print(f"    tau_small = {got:.6g}  (exact {1e-9/(1-B):.6g})  "
      f"diagnostics={gm.coupling_diagnostics()['flow+struct']}")
