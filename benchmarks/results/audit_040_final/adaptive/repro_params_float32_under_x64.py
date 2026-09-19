"""Probe: params_pytree() hard-codes float32 (core/node.py:342), so under
jax_enable_x64 an AdaptiveNode's diagnostics and graph gradients run at
float32 parameters while its own solve runs at float64."""
import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING-wt/audit/adaptive/tests")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from maddening.core.graph_manager import GraphManager
from nodes.adaptive._toys import PoissonSineTopKNode

node = PoissonSineTopKNode(n=128, k=32, theta=0.42, sigma=0.04, blindness_gate=False)
print("jax_enable_x64          :", jax.config.read("jax_enable_x64"))
print("node.dtype (cold buffer):", node.dtype)
pt = node.params_pytree()
for k, v in sorted(pt.items()):
    print(f"   params_pytree[{k!r}] = {v!r}")
print("self.params['sigma']    :", repr(node.params["sigma"]),
      " -> pytree value", float(pt["sigma"]),
      f" (relative shift {abs(float(pt['sigma'])-node.params['sigma'])/node.params['sigma']:.2e})")

st = node.initial_state()
print("state c dtype           :", st["c"].dtype)
print("cold-start solve used sigma =", node.params["sigma"], "(float64 python float)")
print("the diagnostic evaluates at sigma =", float(pt["sigma"]), "(float32-rounded)")

gm = GraphManager(); gm.add_node(PoissonSineTopKNode("a", n=128, k=32, theta=0.42, blindness_gate=False))
gm.compile()
print("\ngm.params['nodes']['a'] dtypes:",
      {k: str(v.dtype) for k, v in gm.params["nodes"]["a"].items()})
def loss(p):
    s = gm.run_scan(3, params=p)
    return jnp.sum(s["a"]["c"] ** 2)
g = jax.grad(loss)(gm.params)
print("gradient dtype returned to the user:",
      {k: str(jnp.asarray(v).dtype) for k, v in g["nodes"]["a"].items()})
