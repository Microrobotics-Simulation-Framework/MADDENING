"""ShardedPointwiseNode: is `shard_axes` honoured, and does update() shard?"""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.join(os.environ["WT"], "tests", "cloud", "multigpu"))
from property_support import PointwiseRelaxNode                      # noqa: E402
from maddening.cloud.multigpu.device_mesh import create_device_mesh  # noqa: E402
from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode  # noqa: E402
from maddening.core.node import BoundaryInputSpec, SimulationNode    # noqa: E402

print(f"devices: {len(jax.devices())}")

# --- A. correctness on the default ("devices") mesh -------------------------
print("\n== A. pointwise sharded vs unsharded (default axis name) ==")
for n_dev in (1, 2, 3, 4):
    for n_cells in (12, 17):
        try:
            m = create_device_mesh(n_dev)
            ref = PointwiseRelaxNode(n_cells=n_cells)
            w = ShardedPointwiseNode(PointwiseRelaxNode(n_cells=n_cells), m)
            bi = {"source": jnp.asarray(
                np.random.default_rng(1).standard_normal(n_cells), jnp.float32)}
            st = w.initial_state()          # this is where device_put happens
            err = float(jnp.max(jnp.abs(
                np.asarray(w.update(st, bi, 0.1)["x"])
                - np.asarray(ref.update(ref.initial_state(), bi, 0.1)["x"]))))
            print(f"  dev={n_dev} n={n_cells} divides={n_cells%n_dev==0}: "
                  f"err={err:.2e} {'OK' if err<1e-5 else '*** MISMATCH ***'}")
        except Exception as exc:
            print(f"  dev={n_dev} n={n_cells} divides={n_cells%n_dev==0}: "
                  f"RAISED {type(exc).__name__}: {str(exc).splitlines()[0][:80]}")

# --- B. is shard_axes honoured? ---------------------------------------------
# A 2-D pointwise node whose state is (3, 8).  Axis 0 (=3) is NOT divisible by
# 4 devices; axis 1 (=8) IS.  Asking for shard_axes=(1,) must therefore work.
class TwoDPointwise(SimulationNode):
    def __init__(self, name="p2d", rows=3, cols=8, timestep=0.1):
        super().__init__(name=name, timestep=timestep, rows=int(rows), cols=int(cols))
    def halo_width(self): return {}
    def state_fields(self): return ["x"]
    def initial_state(self):
        r, c = int(self.params["rows"]), int(self.params["cols"])
        return {"x": jnp.reshape(jnp.arange(r*c, dtype=jnp.float32), (r, c))}
    def boundary_input_spec(self): return {}
    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": state["x"] * 1.5}

print("\n== B. shard_axes=(1,) on a (3, 8) state, 4 devices ==")
print("   axis 0 = 3 (NOT divisible by 4); axis 1 = 8 (divisible by 4)")
m4 = create_device_mesh(4)
for axes in ((0,), (1,)):
    try:
        w = ShardedPointwiseNode(TwoDPointwise(), m4, shard_axes=axes)
        st = w.initial_state()
        sh = st["x"].sharding
        print(f"  shard_axes={axes}: OK, resulting sharding spec = {sh.spec}")
    except Exception as exc:
        print(f"  shard_axes={axes}: RAISED {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:100]}")

# --- C. does update() preserve sharding / shard at all? ---------------------
print("\n== C. does ShardedPointwiseNode.update() apply any sharding? ==")
import inspect
src = inspect.getsource(ShardedPointwiseNode.update)
print("   shard_map in update():", "shard_map" in src)
print("   device_put in update():", "device_put" in src)
w = ShardedPointwiseNode(PointwiseRelaxNode(n_cells=16), create_device_mesh(4))
st = w.initial_state()
bi = {"source": jnp.zeros(16, jnp.float32)}
print("   initial_state sharding:", st["x"].sharding.spec)
unsharded_in = {"x": jnp.asarray(np.arange(16, dtype=np.float32))}
out = w.update(unsharded_in, bi, 0.1)
print("   update() on an UNSHARDED input returns sharding:", out["x"].sharding)
