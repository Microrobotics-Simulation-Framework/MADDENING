"""Are DEFAULT_ORDER_SHORTFALL / DEFAULT_ORDER_EXCESS supported by their
own stated justifications, and does the band still do its job?

DEFAULT_ORDER_EXCESS = 1.0 is justified in mms.py by:
  "the corrected fourth-order stencil measures 5.02 over one pair of a
   fourth-order ladder"
DEFAULT_ORDER_SHORTFALL = 0.25 is justified by:
  "HeatNode 1.982 against 2, LBMNode 1.998, RigidBodyNode 0.999 ...
   the coarsest pair ... sits as much as 0.16 low (1.847)"

This measures the largest pairwise order the corrected 4th-order stencil
actually reaches, and the coarsest-pair shortfall, over several
manufactured solutions -- and then checks what the band would and would
not catch.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import maddening.nodes.heat as heat
from maddening.nodes.heat import HeatNode
from maddening.testing.mms import DEFAULT_ORDER_EXCESS, DEFAULT_ORDER_SHORTFALL

L, ALPHA = 1.0, 1.0
PROFILES = {
    "release  (sin + 0.5x + 1 + 0.4x^2)":
        lambda x: jnp.sin(2*jnp.pi*x/L) + 0.5*x + 1.0 + 0.4*x*x,
    "flat-ends (sin + 0.5x + 1)  [pre-0.4.0]":
        lambda x: jnp.sin(2*jnp.pi*x/L) + 0.5*x + 1.0,
    "exp_sin":
        lambda x: jnp.exp(0.7*x)*jnp.sin(3.0*x + 0.4) + 0.3*x*x + 0.2*x + 1.0,
    "tanh_bump":
        lambda x: jnp.tanh(3.0*(x-0.35)) + 0.6*x*x + 1.0,
}

CUBIC = heat._dirichlet_ghosts_4th_order

def linear_ghosts(T, Tb):
    """Genuinely 2nd-order linear extrapolation through the rod end."""
    near = 2.0*Tb - T[0]
    far = 4.0*Tb - 3.0*T[0]
    return far, near

def quadratic_ghosts(T, Tb):
    """Quadratic through (0,Tb),(dx/2,T0),(3dx/2,T1)."""
    near = (8.0*Tb - 10.0*T[0] + 3.0*T[1]) / 1.0 * 0 + (8.0*Tb - 10.0*T[0] + 3.0*T[1])/1.0
    # p(x) = Tb + a x + b x^2 with p(dx/2)=T0, p(3dx/2)=T1 (units of dx)
    # a/2 + b/4 = T0-Tb ; 3a/2 + 9b/4 = T1-Tb
    # -> b = (T1 - 3*T0 + 2*Tb)/1.5 ... solve numerically instead:
    A = np.array([[0.5, 0.25], [1.5, 2.25]])
    inv = np.linalg.inv(A)
    d0 = T[0] - Tb; d1 = T[1] - Tb
    a = inv[0, 0]*d0 + inv[0, 1]*d1
    b = inv[1, 0]*d0 + inv[1, 1]*d1
    p = lambda xx: Tb + a*xx + b*xx*xx
    return p(-1.5), p(-0.5)


def err(n_cells, f, fourier=0.3, stencil_order=4, decay=20.0):
    d2f = jax.vmap(jax.grad(jax.grad(f)))
    dx = L/n_cells
    dt = fourier*dx*dx/ALPHA
    x = np.linspace(dx/2, L-dx/2, n_cells)
    xj = jnp.asarray(x, dtype=jnp.float64)
    exact = np.asarray(jax.vmap(f)(xj), dtype=np.float64)
    source = -ALPHA*np.asarray(d2f(xj), dtype=np.float64)
    node = HeatNode("h", timestep=dt, n_cells=n_cells, length=L,
                    thermal_diffusivity=ALPHA, stencil_order=stencil_order)
    bc = {"left_temperature": jnp.asarray(float(f(jnp.float64(0.0)))),
          "right_temperature": jnp.asarray(float(f(jnp.float64(L)))),
          "heat_source": jnp.asarray(source)}
    step = jax.jit(lambda T: node.update({"temperature": T}, bc, dt)["temperature"])
    nsteps = int(decay*n_cells**2/(fourier*np.pi**2)) + 50
    T = jax.lax.fori_loop(0, nsteps, lambda _, t: step(t), jnp.asarray(exact))
    T = np.asarray(jax.device_get(T), dtype=np.float64)
    return float(np.sqrt(np.mean((T-exact)**2))/np.sqrt(np.mean(exact**2)))


def pairwise(levels, f, **kw):
    e = [err(n, f, **kw) for n in levels]
    return e, [np.log(e[i]/e[i+1])/np.log(levels[i+1]/levels[i])
               for i in range(len(e)-1)]


LV = (10, 20, 40, 80, 160)
print(f"DEFAULT_ORDER_SHORTFALL={DEFAULT_ORDER_SHORTFALL}  "
      f"DEFAULT_ORDER_EXCESS={DEFAULT_ORDER_EXCESS}  "
      f"-> band for a declared 4 is [{4-DEFAULT_ORDER_SHORTFALL}, "
      f"{4+DEFAULT_ORDER_EXCESS}]\n")

print("--- CORRECT (cubic) ghost closure: what is the largest pairwise order? ---")
allmax = -np.inf
for name, f in PROFILES.items():
    e, o = pairwise(LV, f)
    allmax = max(allmax, max(o))
    print(f"  {name:42s} orders={[f'{x:.3f}' for x in o]}  max={max(o):.3f}")
print(f"  >>> largest pairwise order anywhere: {allmax:.4f}  "
      f"(mms.py justifies EXCESS=1.0 with 5.02)")

print("\n--- MUTATION: linear (genuinely 2nd-order) ghost closure ---")
heat._dirichlet_ghosts_4th_order = linear_ghosts
for name, f in PROFILES.items():
    e, o = pairwise(LV, f)
    verdict = ("PASS" if 4-DEFAULT_ORDER_SHORTFALL <= o[-1] <= 4+DEFAULT_ORDER_EXCESS
               else "FAIL")
    tight = ("PASS" if 3.75 <= o[-1] <= 4.05 else "FAIL")
    print(f"  {name:42s} orders={[f'{x:.3f}' for x in o]}  observed={o[-1]:.3f}"
          f"  band[3.75,5.0]={verdict}  band[3.75,4.05]={tight}")

print("\n--- MUTATION: quadratic ghost closure ---")
heat._dirichlet_ghosts_4th_order = quadratic_ghosts
for name, f in PROFILES.items():
    e, o = pairwise(LV, f)
    verdict = ("PASS" if 4-DEFAULT_ORDER_SHORTFALL <= o[-1] <= 4+DEFAULT_ORDER_EXCESS
               else "FAIL")
    print(f"  {name:42s} orders={[f'{x:.3f}' for x in o]}  observed={o[-1]:.3f}  "
          f"band[3.75,5.0]={verdict}")
heat._dirichlet_ghosts_4th_order = CUBIC

print("\n--- SHORTFALL justification: coarsest-pair order, 2nd-order stencil ---")
for name, f in PROFILES.items():
    e, o = pairwise(LV, f, stencil_order=2, fourier=0.4)
    print(f"  {name:42s} orders={[f'{x:.3f}' for x in o]}  coarsest={o[0]:.3f}")
