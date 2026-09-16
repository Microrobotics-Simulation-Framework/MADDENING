"""Run with JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu."""
import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np, pytest
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

BIG = 2**40 + 987654321

class N(SimulationNode):
    def __init__(self, name, dt, fdt=jnp.float32, ints=False):
        super().__init__(name, dt); self._f = fdt; self._ints = ints
    def initial_state(self):
        s = {"x": jnp.array(1.0, self._f)}
        if self._ints:
            s["n"] = jnp.array(BIG, jnp.int64); s["u"] = jnp.array(2**63 - 5, jnp.uint64)
        return s
    def update(self, s, bi, dt):
        other = bi.get("other", jnp.array(0.0, self._f))
        out = {"x": (s["x"] + dt * (other - s["x"])).astype(self._f)}
        if self._ints:
            out["n"] = s["n"] + 1; out["u"] = s["u"]
        return out
    def boundary_input_spec(self):
        return {"other": BoundaryInputSpec(shape=(), description="o")}

def _gm(solver, **kw):
    gm = GraphManager(); gm.add_node(N("a", 0.1, **kw)); gm.add_node(N("b", 0.1, **kw))
    gm.add_edge("a", "b", "x", "other"); gm.add_edge("b", "a", "x", "other")
    gm.add_coupling_group(["a", "b"], solver=solver, max_iterations=5)
    gm.compile(); return gm

def test_float64_state_ift_step_does_not_retrace_and_run_scan_works():
    gm = _gm("ift", fdt=jnp.float64)
    gm.step(); gm.step(); n = gm._compiled_step._cache_size()
    gm.run_scan(2)
    assert n == 1, n

@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_int64_leaves_exact_with_float32_state(solver):
    gm = _gm(solver, ints=True)
    out = gm.run_scan(3)
    assert out["a"]["n"].dtype == jnp.int64 and int(out["a"]["n"]) == BIG + 3
    assert int(out["a"]["u"]) == 2**63 - 5

def test_grad_through_ift_scan_with_int64_leaf():
    gm = _gm("ift", ints=True)
    step = gm._build_step_fn()
    def f(x0):
        s = jax.tree.map(lambda v: v, gm._state); s["a"]["x"] = x0
        def body(c, _): return step(c, gm._default_external_inputs(), gm.params), None
        s, _ = jax.lax.scan(body, s, None, length=3)
        return s["b"]["x"]
    assert bool(jnp.isfinite(jax.grad(f)(jnp.array(1.0, jnp.float32))))
