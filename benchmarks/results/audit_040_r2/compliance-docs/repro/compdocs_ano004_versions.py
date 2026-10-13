"""MADD-ANO-004 claims affected_versions "0.3.0, 0.3.1".

The mechanism the entry describes -- ShardedPointwiseNode keeps a private copy
of the inner node's params that update() never reads, so the write that
PUT /graph/params performs (``node.params[key] = value``) changes nothing --
is exercised directly here at each tag.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
tree = sys.argv[1]
sys.path.insert(0, os.path.join(tree, "src"))
import jax, jax.numpy as jnp
import maddening
print("  maddening from:", maddening.__file__)
from maddening.nodes.spring import SpringDamperNode
from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode
from maddening.cloud.multigpu.device_mesh import create_device_mesh

inner = SpringDamperNode("s", timestep=0.01, stiffness=10.0)
mesh = create_device_mesh(shape=(1,))
w = ShardedPointwiseNode(inner, mesh)

def trajectory(node):
    st = node.initial_state()
    for _ in range(5):
        st = node.update(st, {}, 0.01)
    return float(jnp.ravel(st["position"])[0])

before = trajectory(w)
# exactly what v0.2.x/v0.3.x PUT /graph/params/{node} does:
w.params["stiffness"] = 1000.0
after = trajectory(w)
direct = trajectory(SpringDamperNode("s2", timestep=0.01, stiffness=1000.0))
print(f"  wrapper position after 5 steps, k=10   : {before:.6f}")
print(f"  wrapper position after params['k']=1000: {after:.6f}")
print(f"  a genuinely k=1000 node                : {direct:.6f}")
print(f"  VERDICT: param write {'IGNORED (defect present)' if after == before else 'applied'}")
