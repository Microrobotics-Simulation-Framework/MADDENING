"""Params reaching a wrapped node; nesting; **kwargs signatures; unstructured."""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.join(os.environ["WT"], "tests", "cloud", "multigpu"))
from property_support import StencilDiffusion1D                       # noqa: E402
from maddening.cloud.multigpu.device_mesh import create_device_mesh   # noqa: E402
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode  # noqa: E402
from maddening.core.simulation.hybrid_node import HybridNode          # noqa: E402
from maddening.core.graph_manager import GraphManager                 # noqa: E402

ND = len(jax.devices()); print(f"devices: {ND}")
def mesh(n): return create_device_mesh(n, shape=(n,), axis_names=("x",))

n_cells = 16
bi = {"source": jnp.asarray(np.random.default_rng(3).standard_normal(n_cells),
                            jnp.float32),
      "gain": jnp.asarray(1.25, jnp.float32)}

# ---- 1. injected params reach the wrapped node -----------------------------
print("\n== 1. params= injected through the wrapper reaches inner.update_padded ==")
ref = StencilDiffusion1D(n_cells=n_cells)
st = ref.initial_state()
for rate in (0.25, 0.75):
    want = ref.update(st, bi, 0.05, params={"rate": jnp.float32(rate)})["f"]
    for nd in (1, 2, 4):
        if nd > ND: continue
        w = ShardedStencilNode(StencilDiffusion1D(n_cells=n_cells), mesh(nd),
                               {"x": 0}, boundary="periodic")
        got = w.update(st, bi, 0.05, params={"rate": jnp.float32(rate)})["f"]
        e = float(jnp.max(jnp.abs(np.asarray(got) - np.asarray(want))))
        print(f"   rate={rate} dev={nd}: err={e:.2e} {'OK' if e<1e-5 else '*** MISMATCH ***'}")

# ---- 2. a node whose update_padded uses **kwargs ---------------------------
print("\n== 2. inner.update_padded declared with **kwargs ==")
class KwargsDiffusion(StencilDiffusion1D):
    def update_padded(self, state_padded, boundary_inputs, dt, **kw):
        return StencilDiffusion1D.update_padded(
            self, state_padded, boundary_inputs, dt,
            static_padded=kw.get("static_padded"),
            shard_info=kw.get("shard_info"),
            params=kw.get("params"))
k = KwargsDiffusion(n_cells=n_cells)
wk = ShardedStencilNode(k, mesh(min(2, ND)), {"x": 0}, boundary="periodic")
print(f"   _inner_accepts_static_padded = {wk._inner_accepts_static_padded}")
print(f"   _inner_accepts_shard_info    = {wk._inner_accepts_shard_info}")
print(f"   _inner_accepts_params        = {wk._inner_accepts_params}"
      "   <-- var-keyword NOT recognised (sharded_node.py:284)")
want = StencilDiffusion1D(n_cells=n_cells).update(st, bi, 0.05,
                                                  params={"rate": jnp.float32(0.9)})["f"]
got = wk.update(st, bi, 0.05, params={"rate": jnp.float32(0.9)})["f"]
e = float(jnp.max(jnp.abs(np.asarray(got) - np.asarray(want))))
print(f"   injected rate=0.9 honoured? err vs reference = {e:.3e} "
      f"{'OK' if e < 1e-5 else '*** PARAMS SILENTLY DROPPED ***'}")
got_default = wk.update(st, bi, 0.05)["f"]
same_as_default = float(jnp.max(jnp.abs(np.asarray(got) - np.asarray(got_default))))
print(f"   with params vs without params: max diff = {same_as_default:.3e} "
      f"{'(params had NO effect)' if same_as_default < 1e-9 else '(params took effect)'}")

# ---- 3. HybridNode(ShardedStencilNode(inner)) ------------------------------
print("\n== 3. HybridNode wrapping a sharded node ==")
inner = StencilDiffusion1D(n_cells=n_cells)
sh = ShardedStencilNode(inner, mesh(min(4, ND)), {"x": 0}, boundary="periodic")
h = HybridNode(sh, lambda state, boundary_inputs, dt: {})
try:
    out = h.update(st, bi, 0.05)
    want = StencilDiffusion1D(n_cells=n_cells).update(st, bi, 0.05)["f"]
    e = float(jnp.max(jnp.abs(np.asarray(out["f"]) - np.asarray(want))))
    print(f"   HybridNode(Sharded(inner)).update err = {e:.2e} "
          f"{'OK' if e<1e-5 else '*** MISMATCH ***'}")
except Exception as exc:
    print(f"   RAISED {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
print(f"   accepts_params={h.accepts_params()}  params_pytree keys="
      f"{sorted(h.params_pytree())}")

# ---- 4. params pytree through a GraphManager -------------------------------
print("\n== 4. sharded node inside a GraphManager: does params reach it? ==")
gm = GraphManager()
gm.add_node(ShardedStencilNode(StencilDiffusion1D(n_cells=n_cells),
                               mesh(min(4, ND)), {"x": 0}, boundary="periodic"))
gm.compile()
print(f"   gm.params['nodes'] keys: {sorted(gm.params.get('nodes', {}))}")
print(f"   diff params: {sorted(gm.params.get('nodes', {}).get('diff', {}))}")
s0 = np.asarray(gm.get_node_state("diff")["f"]).copy()
gm.step(); a = np.asarray(gm.get_node_state("diff")["f"]).copy()
gm.set_node_state("diff", {"f": jnp.asarray(s0)})
if "diff" in gm.params.get("nodes", {}):
    gm.params["nodes"]["diff"]["rate"] = jnp.asarray(2.0, jnp.float32)
gm.step(); b = np.asarray(gm.get_node_state("diff")["f"])
print(f"   step with rate=0.5 vs rate=2.0 differ? max diff = "
      f"{float(np.max(np.abs(a-b))):.3e} "
      f"{'OK (params reach the sharded node)' if np.max(np.abs(a-b)) > 1e-6 else '*** PARAM WRITE IGNORED ***'}")

# ---- 5. gradient consequence of the dropped params -------------------------
print("\n== 5. d/d(rate) through a **kwargs inner node under the sharded wrapper ==")
f0 = jnp.asarray(np.sin(np.linspace(0, 2*np.pi, n_cells, endpoint=False)), jnp.float32)
def loss_for(node):
    def loss(rate):
        return jnp.sum(node.update({"f": f0}, bi, 0.05, params={"rate": rate})["f"] ** 2)
    return loss
r0 = jnp.asarray(0.5, jnp.float32)

g_ref = float(jax.grad(loss_for(StencilDiffusion1D(n_cells=n_cells)))(r0))
w_ok = ShardedStencilNode(StencilDiffusion1D(n_cells=n_cells), mesh(min(4, ND)),
                          {"x": 0}, boundary="periodic")
g_ok = float(jax.grad(loss_for(w_ok))(r0))
w_kw = ShardedStencilNode(KwargsDiffusion(n_cells=n_cells), mesh(min(4, ND)),
                          {"x": 0}, boundary="periodic")
g_kw = float(jax.grad(loss_for(w_kw))(r0))
print(f"   unsharded reference           d/d rate = {g_ref:.8f}")
print(f"   sharded, explicit params kwarg d/d rate = {g_ok:.8f}")
print(f"   sharded, **kwargs inner        d/d rate = {g_kw:.8f}"
      f"   {'<-- SILENTLY ZERO' if abs(g_kw) < 1e-12 else ''}")
