"""HeatNode.compute_boundary_fluxes: which face does it report?

boundary_flux_spec() calls these 'Heat flux at left boundary' /
'...at right boundary'.  Since 0.4.0 the node's Dirichlet datum is the
temperature AT THE ROD END (x=0, x=L) and the cell centres are at
dx/2 ... L-dx/2.  The rod-end flux is  -alpha*(T[0]-T_left)/(dx/2).
The code returns  -alpha*(T[1]-T[0])/dx, the flux midway between the
first two CELL CENTRES, i.e. at x = dx.

Test: on a field with a known analytic gradient, compare what the node
reports with the true flux at x=0 and with the true flux at x=dx.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from maddening.nodes.heat import HeatNode

L, ALPHA = 1.0, 1.0

# T(x) = exp(x) -> dT/dx = exp(x); flux q = -alpha dT/dx
T_of = lambda x: np.exp(x)
q_of = lambda x: -ALPHA * np.exp(x)

print(f"{'n':>5} {'reported_left':>15} {'true q(0)':>13} {'true q(dx)':>13} "
      f"{'err vs q(0)':>13} {'err vs q(dx)':>13}")
for n in (10, 20, 40, 80, 160, 320, 640):
    dx = L / n
    x = np.linspace(dx/2, L - dx/2, n)
    T = T_of(x)
    node = HeatNode("h", timestep=0.4*dx*dx/ALPHA, n_cells=n, length=L,
                    thermal_diffusivity=ALPHA)
    bi = {"left_temperature": jnp.asarray(T_of(0.0)),
          "right_temperature": jnp.asarray(T_of(L))}
    out = node.compute_boundary_fluxes({"temperature": jnp.asarray(T)}, bi, 1e-4)
    rep = float(out["left_heat_flux"])
    e0 = abs(rep - q_of(0.0)) / abs(q_of(0.0))
    edx = abs(rep - q_of(dx)) / abs(q_of(dx))
    print(f"{n:5d} {rep:15.8f} {q_of(0.0):13.8f} {q_of(dx):13.8f} "
          f"{e0:13.3e} {edx:13.3e}")

print("\nConvergence of the reported left flux to the TRUE ROD-END flux:")
errs = []
ns = (10, 20, 40, 80, 160, 320)
for n in ns:
    dx = L/n
    x = np.linspace(dx/2, L-dx/2, n)
    T = T_of(x)
    node = HeatNode("h", timestep=0.4*dx*dx, n_cells=n, length=L, thermal_diffusivity=ALPHA)
    out = node.compute_boundary_fluxes({"temperature": jnp.asarray(T)},
        {"left_temperature": jnp.asarray(T_of(0.0))}, 1e-4)
    errs.append(abs(float(out["left_heat_flux"]) - q_of(0.0)))
for i in range(len(errs)-1):
    print(f"  n={ns[i]:4d}->{ns[i+1]:4d}  err {errs[i]:.4e} -> {errs[i+1]:.4e}  "
          f"order {np.log(errs[i]/errs[i+1])/np.log(2):.3f}")

print("\nWhat the correct rod-end reconstruction would give "
      "(one-sided, -alpha*(T[0]-T_left)/(dx/2)):")
errs2 = []
for n in ns:
    dx = L/n
    x = np.linspace(dx/2, L-dx/2, n)
    T = T_of(x)
    q = -ALPHA*(T[0] - T_of(0.0))/(dx/2)
    errs2.append(abs(q - q_of(0.0)))
for i in range(len(errs2)-1):
    print(f"  n={ns[i]:4d}->{ns[i+1]:4d}  err {errs2[i]:.4e} -> {errs2[i+1]:.4e}  "
          f"order {np.log(errs2[i]/errs2[i+1])/np.log(2):.3f}")

print("\nGlobal energy balance check on a steady linear profile T(x)=x "
      "(exact for the scheme; true flux is -1 at BOTH ends, so in == out):")
for n in (10, 40, 160):
    dx = L/n
    x = np.linspace(dx/2, L-dx/2, n)
    T = x.copy()
    node = HeatNode("h", timestep=0.4*dx*dx, n_cells=n, length=L, thermal_diffusivity=ALPHA)
    out = node.compute_boundary_fluxes({"temperature": jnp.asarray(T)},
        {"left_temperature": jnp.asarray(0.0), "right_temperature": jnp.asarray(1.0)}, 1e-4)
    print(f"  n={n:4d}  left={float(out['left_heat_flux']):+.6f} "
          f"right={float(out['right_heat_flux']):+.6f}  (true -1.000000 at both ends)")
