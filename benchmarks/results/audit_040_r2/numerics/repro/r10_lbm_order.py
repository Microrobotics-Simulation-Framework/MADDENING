"""Independent spatial-order measurement for LBMNode (declared 2.0).

Kolmogorov-type steady shear, but a TWO-mode profile the repo's study
does not use, with the driving force written in closed form (no
ManufacturedSolution / diffusion_operator / measure_order), my own L2
norm and my own ladder.

    u_x(y) = U [ sin(k y) + 0.4 sin(2 k y) ],  u_y = 0,  k = 2 pi / N

u depends on y only and u_y = 0, so (u.grad)u = 0 exactly: this is an
exact steady solution of incompressible NS driven by
    F_x = rho nu U [ k^2 sin(k y) + 0.4 (2k)^2 sin(2 k y) ]
Diffusive scaling: lattice viscosity fixed, U ~ 1/N.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from maddening.nodes.lbm import LBMNode
from maddening.testing.mms import declared_order

NU = 0.05
NX = 4
U_REF, N_REF = 0.02, 16


def err(n):
    U = U_REF * N_REF / n
    k = 2.0*np.pi/n
    y = np.arange(n, dtype=np.float64)
    exact = U*(np.sin(k*y) + 0.4*np.sin(2*k*y))
    fx = NU*U*(k*k*np.sin(k*y) + 0.4*(2*k)**2*np.sin(2*k*y))
    force = np.zeros((NX, n, 2), dtype=np.float64)
    force[:, :, 0] = fx
    node = LBMNode("lbm", timestep=1.0, grid_shape=(NX, n), viscosity=NU,
                   lattice="D2Q9")
    state = {kk: (v if v.dtype == jnp.uint8 else jnp.asarray(v, jnp.float64))
             for kk, v in node.initial_state().items()}
    step = jax.jit(lambda s: node.update(s, {"body_force": jnp.asarray(force)}, 1.0))
    steps = int(14.0/(NU*k*k)) + 100
    state = jax.lax.fori_loop(0, steps, lambda _, s: step(s), state)
    ux = np.asarray(jax.device_get(state["velocity"]), np.float64)[..., 0].mean(axis=0)
    return float(np.sqrt(np.mean((ux-exact)**2))/np.sqrt(np.mean(exact**2)))


lv = (16, 32, 64, 128)
e = [err(n) for n in lv]
o = [np.log(e[i]/e[i+1])/np.log(lv[i+1]/lv[i]) for i in range(len(e)-1)]
node = LBMNode("lbm", timestep=1.0, grid_shape=(4, 16), viscosity=NU, lattice="D2Q9")
print(f"LBMNode declared spatial order = {declared_order(node).spatial}, "
      f"temporal = {declared_order(node).temporal}")
for i, n in enumerate(lv):
    s = "" if i == 0 else f"{o[i-1]:8.3f}"
    print(f"   N={n:5d}  err={e[i]:12.5e} {s}")
print(f"   observed (finest pair) = {o[-1]:.4f}  max pairwise = {max(o):.4f}  "
      f"monotone={all(e[i+1] < e[i] for i in range(len(e)-1))}")
