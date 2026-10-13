"""Probes: (a) mask_safe on a non-1-D operand; (b) vmap with genuinely
different active sets per batch element."""
import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING-wt/audit/adaptive/tests")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from maddening.nodes.adaptive import AdaptiveNode
from nodes.adaptive._toys import MaskedDenseNode

print("=== (a) mask_safe on a matrix operand ===")
mask = jnp.array([True, False, True, False])
A = jnp.arange(16.0).reshape(4, 4)
out = AdaptiveNode.mask_safe(mask, A, fill=1.0)
print("   A =\n", np.asarray(A))
print("   mask_safe(mask, A) =\n", np.asarray(out))
print("   -> broadcast over the LAST axis: columns are filled, rows are not.")
print("      A subclass sanitising a masked operator row-wise gets the wrong axis,")
print("      silently, with no error.  Docstring says only 'Operand to sanitise'.")
print("   mask.reshape(-1,1) variant =\n", np.asarray(AdaptiveNode.mask_safe(mask[:, None], A, 1.0)))

print("\n=== (b) vmap over theta with genuinely different active sets ===")
node = MaskedDenseNode(n=24, k=6, theta=0.3, blindness_gate=False)
st = node.initial_state()
def step(t):
    return node.update(st, {}, 1.0, params={"theta": t})
thetas = jnp.array([0.05, 0.3, 3.0, 30.0])
vm = jax.vmap(step)(thetas)
allsame = True
for i, t in enumerate(thetas):
    ref = step(t)
    same = bool(jnp.all(vm["mask"][i] == ref["mask"]))
    allsame &= same
    print(f"   theta={float(t):6.2f}  active idx vmap={np.flatnonzero(np.asarray(vm['mask'][i]))}"
          f"  loop={np.flatnonzero(np.asarray(ref['mask']))}  max|dc|={float(jnp.max(jnp.abs(vm['c'][i]-ref['c']))):.2e}")
print("   masks all match the per-element loop:", allsame)
print("   distinct active sets across the batch:",
      len({tuple(np.flatnonzero(np.asarray(m))) for m in vm["mask"]}))

print("\n=== (c) vmap of the gradient with different active sets ===")
def J(t):
    out = step(t)
    return node.objective(out, {**node.params, "theta": t})
gv = jax.vmap(jax.grad(J))(thetas)
gl = jnp.array([jax.grad(J)(t) for t in thetas])
print("   vmap(grad) =", np.asarray(gv))
print("   loop grad  =", np.asarray(gl))
print("   max abs diff =", float(jnp.max(jnp.abs(gv - gl))))
