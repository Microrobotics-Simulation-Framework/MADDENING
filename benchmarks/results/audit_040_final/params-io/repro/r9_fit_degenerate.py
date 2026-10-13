"""Degenerate inputs to fit / fit_lm / fim: rejected, or a silent number?"""
import numpy as np, jax, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening import sysid

def build():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.5,
                                 initial_position=0.5))
    gm.compile()
    return gm

gm = build()
step, ext, s0 = gm._build_step_fn(), gm._default_external_inputs(), gm._state
def roll(p, n=30):
    def one(s, _):
        s = step(s, ext, p); return s, s["s"]["position"]
    return jax.lax.scan(one, s0, None, length=n)[1]
target = np.asarray(roll(gm.params))
def loss(p):
    return jnp.sum((roll(p) - jnp.asarray(target)) ** 2)
def resid(p):
    return roll(p) - jnp.asarray(target)

print("=== fit hyper-parameter validation ===")
for kw in ({"n_iter": 0}, {"n_iter": -5}, {"lr": 0.0}, {"lr": -0.1}, {"lr": float('nan')},
           {"betas": (1.0, 0.999)}, {"betas": (0.9, 1.0)}, {"eps": 0.0}, {"eps": -1e-8},
           {"tol": float('nan')}, {"notify_every": -1}):
    g = build()
    try:
        r = sysid.fit(g, loss, n_iter=kw.pop("n_iter", 3), **kw)
        moved = {k: float(v) for k, v in r.params["nodes"]["s"].items()
                 if not np.array_equal(np.asarray(v), np.asarray(g.params["nodes"]["s"][k]))}
        print(f"  {str(kw):32s} -> n_iter={r.n_iter} losses={len(r.losses)} converged={r.converged} moved={list(moved)}")
    except Exception as e:
        print(f"  {str(kw):32s} -> {type(e).__name__}: {str(e)[:90]}")

print()
print("=== fit_lm hyper-parameters ===")
for kw in ({"lam0": 0.0}, {"lam0": -1.0}, {"lam_up": 0.5}, {"lam_down": 2.0},
           {"step_tol": -1.0}, {"n_iter": 0}):
    g = build()
    try:
        r = sysid.fit_lm(g, resid, n_iter=kw.pop("n_iter", 3), **kw)
        print(f"  {str(kw):24s} -> n_iter={r.n_iter} converged={r.converged} losses={np.asarray(r.losses)[:3]}")
    except Exception as e:
        print(f"  {str(kw):24s} -> {type(e).__name__}: {str(e)[:90]}")

print()
print("=== n_iter=0: what does FitResult say? ===")
g = build()
r = sysid.fit(g, loss, n_iter=0)
print("  n_iter:", r.n_iter, "losses:", r.losses, "converged:", r.converged)
print("  params identical to input:",
      all(np.asarray(v).tobytes() == np.asarray(g.params['nodes']['s'][k]).tobytes()
          for k, v in r.params["nodes"]["s"].items()))

print()
print("=== mask that selects nothing ===")
g = build()
m = jax.tree.map(lambda x: False, g.trainable_mask(g.params))
try:
    sysid.fit(g, loss, mask=m, n_iter=2)
except Exception as e:
    print("  ->", type(e).__name__, e)

print()
print("=== mask that widens (marks a frozen leaf) ===")
g = build()
m = jax.tree.map(lambda x: True, g.trainable_mask(g.params))
try:
    sysid.fit(g, loss, mask=m, n_iter=2)
except Exception as e:
    print("  ->", type(e).__name__, str(e)[:160], "...")
