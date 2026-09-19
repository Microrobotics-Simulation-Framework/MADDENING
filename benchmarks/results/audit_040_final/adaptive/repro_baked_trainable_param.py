"""Repro: an AdaptiveNode subclass that precomputes its basis in __init__ from a
TRAINABLE parameter gets a silently-zero gradient, and compile() accepts it.

The framework's guard against this (SimulationNode.static_data_deps, extended
this release and used by HeatNode) only fires for arrays published through
`static_data`.  AdaptiveNode's documented pattern -- and both of its toys --
hold the precomputed basis as a bare instance attribute, so nothing checks it.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.core.solver_utils import ift_linear_solve
from maddening.nodes.adaptive import AdaptiveNode

N = 32
KS = jnp.arange(1.0, N + 1.0)
LAM = (KS * jnp.pi) ** 2 + 1.0
XG = jnp.linspace(0.0, 1.0, 2 * N)
PHI = jnp.sqrt(2.0) * jnp.sin(jnp.pi * jnp.outer(XG, KS))
DX = XG[1] - XG[0]


class BakedSensor(AdaptiveNode):
    """`sensor_x` is trainable, but the sensor row is baked in __init__."""
    def __init__(self, bake: bool, **kw):
        super().__init__("baked", 1.0, n_max=N, theta=0.42, sigma=0.08,
                         sensor_x=1.0 / 3.0, blindness_gate=False, **kw)
        self._bake = bake
        # <-- the offending line: an array derived from a trainable parameter
        self._phi_sensor = jnp.sqrt(2.0) * jnp.sin(jnp.pi * KS * self.params["sensor_x"])

    def param_specs(self):
        return {**super().param_specs(),
                "theta": ParamSpec(), "sigma": ParamSpec(trainable=False),
                "sensor_x": ParamSpec()}          # trainable

    def rhs(self, p):
        f = jnp.exp(-((XG - p["theta"]) / p["sigma"]) ** 2)
        return DX * (PHI.T @ f)

    def compute_active_set(self, state, params, *, prev=None, is_cold_start=False):
        s = jnp.abs(self.rhs(params))
        return s >= jnp.sort(s)[-8]

    def solve_frozen(self, state, mask, params):
        diag = jnp.where(mask, LAM, 1.0)
        return {"c": ift_linear_solve(lambda v: diag * v,
                                      jnp.where(mask, self.rhs(params), 0.0),
                                      solver="dense")}

    def objective(self, state, params):
        if self._bake:                       # reads the baked array
            return self._phi_sensor @ state["c"]
        row = jnp.sqrt(2.0) * jnp.sin(jnp.pi * KS * params["sensor_x"])
        return row @ state["c"]              # recomputed from the traced param


def dJ_dsensor(bake):
    node = BakedSensor(bake)
    st = node.initial_state()
    def J(sx):
        p = {**node.params, "theta": jnp.asarray(0.42), "sensor_x": sx}
        return node.objective(node.update(st, {}, 1.0, params={"theta": jnp.asarray(0.42),
                                                               "sensor_x": sx}), p)
    x0 = jnp.asarray(1.0 / 3.0)
    g = float(jax.grad(J)(x0))
    h = 1e-6
    fd = float((J(x0 + h) - J(x0 - h)) / (2 * h))
    return g, fd

print("objective reads the BAKED array (the documented AdaptiveNode pattern):")
g, fd = dJ_dsensor(True)
print(f"   jax.grad = {g: .12e}    central FD(h=1e-6) = {fd: .12e}")
print("objective recomputes from the traced parameter:")
g2, fd2 = dJ_dsensor(False)
print(f"   jax.grad = {g2: .12e}    central FD(h=1e-6) = {fd2: .12e}")

print("\ncompile() verdict on the baked node:")
gm = GraphManager()
node = BakedSensor(True)
gm.add_node(node)
try:
    gm.compile()
    print("   compile() ACCEPTED the graph (no static_data_deps violation seen)")
except Exception as e:
    print("   compile() refused:", type(e).__name__, e)

print("\nsame array, but published through static_data and declared:")
class Declared(BakedSensor):
    @property
    def static_data(self):
        return {"phi_sensor": self._phi_sensor}
    def static_data_deps(self):
        return {"phi_sensor": ("sensor_x",)}
gm2 = GraphManager()
gm2.add_node(Declared(True, name_suffix=None) if False else Declared(True))
try:
    gm2.compile()
    print("   compile() ACCEPTED")
except Exception as e:
    print("   compile() REFUSED:", type(e).__name__, str(e)[:300])
