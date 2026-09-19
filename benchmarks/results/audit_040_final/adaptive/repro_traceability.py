"""Probe: jit / vmap / scan / grad traceability of AdaptiveNode.update."""
import sys, time, warnings
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING-wt/audit/adaptive/tests")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from nodes.adaptive._toys import PoissonSineTopKNode

node = PoissonSineTopKNode(n=64, k=8, theta=0.42, blindness_gate=False)
st = node.initial_state()

def step(state, theta):
    return node.update(state, {}, 1.0, params={"theta": theta})

print("== eager vs jit ==")
e = step(st, jnp.asarray(0.42))
j = jax.jit(step)(st, jnp.asarray(0.42))
print(" max |c_eager - c_jit| =", float(jnp.max(jnp.abs(e["c"] - j["c"]))))
print(" masks equal:", bool(jnp.all(e["mask"] == j["mask"])))

print("\n== same jit, different active set (theta far apart) ==")
f = jax.jit(step)
a = f(st, jnp.asarray(0.10)); b = f(st, jnp.asarray(0.90))
print(" n_active:", int(a['mask'].sum()), int(b['mask'].sum()),
      " masks differ:", bool(jnp.any(a['mask'] != b['mask'])))
ea = step(st, jnp.asarray(0.10)); eb = step(st, jnp.asarray(0.90))
print(" jit matches eager:", float(jnp.max(jnp.abs(a['c']-ea['c']))), float(jnp.max(jnp.abs(b['c']-eb['c']))))

print("\n== vmap over theta with different active sets per batch element ==")
thetas = jnp.array([0.10, 0.42, 0.90])
vm = jax.vmap(lambda t: step(st, t))(thetas)
for i, t in enumerate(thetas):
    ref = step(st, t)
    print(f"  theta={float(t):.2f} n_active vmap={int(vm['mask'][i].sum())} loop={int(ref['mask'].sum())}"
          f" max|dc|={float(jnp.max(jnp.abs(vm['c'][i]-ref['c']))):.3e}"
          f" masks_equal={bool(jnp.all(vm['mask'][i]==ref['mask']))}")

print("\n== scan ==")
def body(state, t):
    out = step(state, t)
    return out, out["c"][0]
ts = jnp.linspace(0.2, 0.8, 7)
final, ys = jax.lax.scan(body, st, ts)
loop_state, loop_ys = st, []
for t in ts:
    loop_state = step(loop_state, t); loop_ys.append(loop_state["c"][0])
print(" scan vs python loop max|dy| =", float(jnp.max(jnp.abs(ys - jnp.array(loop_ys)))))

print("\n== grad through scan ==")
def total(theta):
    def bd(state, i):
        out = step(state, theta)
        return out, None
    fin, _ = jax.lax.scan(bd, st, jnp.arange(3))
    return node.objective(fin, {**node.params, "theta": theta})
print(" grad =", float(jax.grad(total)(jnp.asarray(0.42))))

print("\n== eager host-sync cost of the off-mask finite check ==")
big = PoissonSineTopKNode(n=512, k=32, theta=0.42, blindness_gate=False)
bst = big.initial_state()
import maddening.nodes.adaptive.base as B
def timeit(fn, n=30):
    fn(); t0=time.perf_counter()
    for _ in range(n): jax.block_until_ready(fn()["c"])
    return (time.perf_counter()-t0)/n
on = timeit(lambda: big.update(bst, {}, 1.0, params={"theta": jnp.asarray(0.42)}))
prev = B.set_adaptive_diagnostics(False)
off = timeit(lambda: big.update(bst, {}, 1.0, params={"theta": jnp.asarray(0.42)}))
B.set_adaptive_diagnostics(prev)
print(f" eager update with diagnostics ON  {on*1e3:.2f} ms")
print(f" eager update with diagnostics OFF {off*1e3:.2f} ms  (wall-clock, shared machine)")
