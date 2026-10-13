"""Sharded vs unsharded stencil step across device counts and cell counts.

Reuses the repo's own toy stencil node (tests/cloud/multigpu/property_support.py).
"""
import os, sys, traceback
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, os.path.join(os.environ["WT"], "tests", "cloud", "multigpu"))
from property_support import StencilDiffusion1D          # noqa: E402
from maddening.cloud.multigpu.device_mesh import create_device_mesh  # noqa: E402
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode  # noqa: E402

ND = len(jax.devices())
print(f"jax devices: {ND}")

def one(n_cells, n_dev):
    inner = StencilDiffusion1D(n_cells=n_cells)
    ref = StencilDiffusion1D(n_cells=n_cells)
    mesh = create_device_mesh(n_dev, shape=(n_dev,), axis_names=("x",))
    w = ShardedStencilNode(inner, mesh, {"x": 0}, boundary="periodic")
    st = ref.initial_state()
    rng = np.random.default_rng(0)
    bi = {"source": jnp.asarray(rng.standard_normal(n_cells), jnp.float32),
          "gain": jnp.asarray(1.25, jnp.float32)}
    dt = 0.05
    want = ref.update(st, bi, dt)["f"]
    got = w.update({k: jnp.asarray(v) for k, v in st.items()}, bi, dt)["f"]
    return float(jnp.max(jnp.abs(np.asarray(got) - np.asarray(want))))

for n_dev in (1, 2, 3, 4):
    if n_dev > ND:
        continue
    for n_cells in (12, 16, 17, 18, 20):
        tag = f"devices={n_dev} n_cells={n_cells} (divides={n_cells % n_dev == 0})"
        try:
            err = one(n_cells, n_dev)
            verdict = "OK " if err < 1e-5 else "*** MISMATCH ***"
            print(f"{tag:48s} max|sharded-unsharded| = {err:.3e}  {verdict}")
        except Exception as exc:
            print(f"{tag:48s} RAISED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:110]}")
