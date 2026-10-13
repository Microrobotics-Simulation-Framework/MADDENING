"""Remaining HeatNode docstring claims."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
import maddening.nodes.heat as heat
from maddening.nodes.heat import HeatNode, _laplacian_4th_order_pure, _dirichlet_ghosts_4th_order

print("CLAIM A (_dirichlet_ghosts_4th_order): under the cubic closure the 5-point")
print("  and 3-point forms are algebraically identical at cells 0 and n-1, both")
print("  reducing to (16 Tb - 25 T0 + 10 T1 - T2) / (5 dx^2).")
rng = np.random.default_rng(3)
n, dx = 9, 0.125
for trial in range(3):
    T = rng.normal(size=n); Tb = float(rng.normal())
    far, near = _dirichlet_ghosts_4th_order(jnp.asarray(T), jnp.asarray(Tb))
    fr2, nr1 = _dirichlet_ghosts_4th_order(jnp.asarray(T[::-1]), jnp.asarray(0.0))
    pad = jnp.concatenate([jnp.asarray([far, near]), jnp.asarray(T),
                           jnp.asarray([nr1, fr2])])
    five = np.asarray(_laplacian_4th_order_pure(pad, dx), np.float64)
    three = np.asarray((pad[3:-1] - 2*pad[2:-2] + pad[1:-3])/(dx*dx), np.float64)
    closed = (16*Tb - 25*T[0] + 10*T[1] - T[2])/(5*dx*dx)
    print(f"  trial {trial}: 5pt[0]={five[0]:.10f} 3pt[0]={three[0]:.10f} "
          f"closed={closed:.10f}  agree={np.allclose([five[0],three[0]],closed)}")
    print(f"           cells 1..n-2 differ between 5pt and 3pt: "
          f"{not np.allclose(five[1:-1], three[1:-1])}")

print("\nCLAIM B (_laplacian_4th_order_uniform docstring): 'Boundary cells")
print("  (i=0,1,n-2,n-1): fall back to 2nd-order.'  The code's mask is")
print("  use_4th = (idx >= 1) & (idx <= n-2), i.e. only i=0 and i=n-1.")
import inspect
src = inspect.getsource(heat._laplacian_4th_order_uniform)
print("  code:", [l.strip() for l in src.splitlines() if "use_4th" in l])

print("\nCLAIM C (compute_interface_correction): 'the value this returns is the")
print("  one update() already produced and applying it is an identity'.")
node = HeatNode("h", timestep=1e-4, n_cells=12, length=1.0, thermal_diffusivity=0.01)
T = jnp.asarray(rng.normal(size=12))
bi = {"left_temperature": jnp.asarray(3.0), "right_temperature": jnp.asarray(-1.0)}
upd = node.update({"temperature": T}, bi, 1e-4)["temperature"]
corr = node.compute_interface_correction({"temperature": T}, bi, 1e-4)
for idx, val in corr["temperature"]:
    print(f"  index {idx!s:>3}: correction={float(val):.15f}  "
          f"update()={float(upd[idx]):.15f}  identical={float(val)==float(upd[idx])}")

print("\nCLAIM D (boundary_flux_spec): output_units='W/m^2'.  The formula is")
print("  -alpha * dT/dx with alpha in m^2/s and T in K, i.e. K*m/s, not W/m^2:")
print("  the true conductive flux is -k dT/dx = -rho*c_p*alpha*dT/dx and")
print("  rho*c_p is not a parameter of this node.")
spec = node.boundary_flux_spec()
out = node.compute_boundary_fluxes({"temperature": jnp.asarray(np.linspace(0,1,12))},
                                   bi, 1e-4)
print(f"  declared units      : {spec['left_heat_flux'].output_units}")
print(f"  dimensional value   : {float(out['left_heat_flux']):.6g}  "
      f"[alpha=0.01 m^2/s, dT/dx~1 K/m -> -0.01 K*m/s]")
print(f"  factor missing (rho*c_p for water ~4.18e6 J/(m^3 K)): "
      f"{float(out['left_heat_flux'])*4.18e6:.4g} W/m^2 would be the real flux")
