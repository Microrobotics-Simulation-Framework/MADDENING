"""Gradient of a sharded step vs the unsharded adjoint, + pointwise/no-static paths.

All on CPU virtual devices: this tests the *logic* of the sharded adjoint,
not the real multi-GPU collective path.
"""
import os, sys
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.join(os.environ["WT"], "tests", "cloud", "multigpu"))
from property_support import StencilDiffusion1D, PointwiseRelaxNode   # noqa: E402
from maddening.cloud.multigpu.device_mesh import create_device_mesh   # noqa: E402
from maddening.cloud.multigpu.sharded_node import (                   # noqa: E402
    ShardedStencilNode, ShardedPointwiseNode)

ND = len(jax.devices())
print(f"jax devices: {ND}\n")

def mesh_for(n):
    return create_device_mesh(n, shape=(n,), axis_names=("x",))

# ---- 1. Pointwise wrapper, divisible and not -------------------------------
print("== ShardedPointwiseNode: sharded vs unsharded ==")
for n_dev in (1, 2, 3, 4):
    if n_dev > ND: continue
    for n_cells in (12, 17):
        try:
            inner = PointwiseRelaxNode(n_cells=n_cells)
            ref = PointwiseRelaxNode(n_cells=n_cells)
            w = ShardedPointwiseNode(inner, mesh_for(n_dev))
            st = ref.initial_state()
            bi = {"source": jnp.asarray(
                np.random.default_rng(1).standard_normal(n_cells), jnp.float32)}
            want = ref.update(st, bi, 0.1)["x"]
            got = w.update(st, bi, 0.1)["x"]
            err = float(jnp.max(jnp.abs(np.asarray(got) - np.asarray(want))))
            print(f"  dev={n_dev} n={n_cells} divides={n_cells%n_dev==0}: err={err:.3e}"
                  f"  {'OK' if err < 1e-5 else '*** MISMATCH ***'}")
        except Exception as exc:
            print(f"  dev={n_dev} n={n_cells} divides={n_cells%n_dev==0}: "
                  f"RAISED {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")

# ---- 2. Gradient through the sharded stencil step --------------------------
print("\n== Gradient: d(sum(f_new)^2)/d(rate) and d/d(f0), sharded vs unsharded ==")
n_cells = 16
rng = np.random.default_rng(7)
bi = {"source": jnp.asarray(rng.standard_normal(n_cells), jnp.float32),
      "gain": jnp.asarray(1.25, jnp.float32)}

def loss_factory(node, use_params):
    def loss(f0, rate):
        st = {"f": f0}
        out = node.update(st, bi, 0.05, params={"rate": rate})
        return jnp.sum(out["f"] ** 2)
    return loss

f0 = jnp.asarray(np.sin(np.linspace(0, 2*np.pi, n_cells, endpoint=False)), jnp.float32)
rate0 = jnp.asarray(0.5, jnp.float32)

ref = StencilDiffusion1D(n_cells=n_cells)
g_ref = jax.grad(loss_factory(ref, True), argnums=(0, 1))(f0, rate0)
print(f"  unsharded: d/d rate = {float(g_ref[1]):.8f}  |d/d f0| = {float(jnp.linalg.norm(g_ref[0])):.8f}")

for n_dev in (1, 2, 4):
    if n_dev > ND: continue
    inner = StencilDiffusion1D(n_cells=n_cells)
    w = ShardedStencilNode(inner, mesh_for(n_dev), {"x": 0}, boundary="periodic")
    g = jax.grad(loss_factory(w, True), argnums=(0, 1))(f0, rate0)
    d_rate = float(g[1]); d_f0 = np.asarray(g[0])
    e_rate = abs(d_rate - float(g_ref[1]))
    e_f0 = float(np.max(np.abs(d_f0 - np.asarray(g_ref[0]))))
    ok = "OK" if (e_rate < 1e-4 and e_f0 < 1e-4) else "*** MISMATCH ***"
    zero = " (ZERO!)" if abs(d_rate) < 1e-12 else ""
    print(f"  dev={n_dev}: d/d rate = {d_rate:.8f}{zero}  err_rate={e_rate:.2e} "
          f"err_f0={e_f0:.2e}  {ok}")
