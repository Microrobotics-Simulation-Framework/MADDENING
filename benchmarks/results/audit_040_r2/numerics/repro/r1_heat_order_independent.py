"""Independent spatial-order study for HeatNode.

Deliberately does NOT use maddening.testing.mms: the point is to measure
the node's order with a harness the release did not write, on a
manufactured solution the release did not choose.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from maddening.nodes.heat import HeatNode

L = 1.0
ALPHA = 1.0


def make_profile(kind):
    if kind == "release":          # the profile tests/verification uses
        f = lambda x: jnp.sin(2*jnp.pi*x/L) + 0.5*x + 1.0 + 0.4*x*x
    elif kind == "exp_sin":        # an independent one, curved at both ends
        f = lambda x: jnp.exp(0.7*x)*jnp.sin(3.0*x + 0.4) + 0.3*x*x + 0.2*x + 1.0
    elif kind == "flat_ends":      # u''=0 at both ends (the disarmed profile)
        f = lambda x: jnp.sin(2*jnp.pi*x/L) + 0.5*x + 1.0
    else:
        raise ValueError(kind)
    d2 = jax.grad(jax.grad(f))
    return f, jax.vmap(d2)


def steady_error(n_cells, stencil_order, fourier, kind, decay=20.0):
    f, d2f = make_profile(kind)
    dx = L / n_cells
    dt = fourier * dx * dx / ALPHA
    x = np.linspace(dx/2, L - dx/2, n_cells)
    xj = jnp.asarray(x, dtype=jnp.float64)
    exact = np.asarray(jax.vmap(f)(xj), dtype=np.float64)
    # steady: 0 = alpha*u'' + S  ->  S = -alpha*u''
    source = -ALPHA * np.asarray(d2f(xj), dtype=np.float64)
    t_left = float(f(jnp.float64(0.0)))
    t_right = float(f(jnp.float64(L)))
    node = HeatNode("h", timestep=dt, n_cells=n_cells, length=L,
                    thermal_diffusivity=ALPHA, initial_temperature=0.0,
                    stencil_order=stencil_order)
    bc = {"left_temperature": jnp.asarray(t_left),
          "right_temperature": jnp.asarray(t_right),
          "heat_source": jnp.asarray(source)}
    step = jax.jit(lambda T: node.update({"temperature": T}, bc, dt)["temperature"])
    nsteps = int(decay * n_cells**2 / (fourier * np.pi**2)) + 50
    T = jax.lax.fori_loop(0, nsteps, lambda _, t: step(t), jnp.asarray(exact))
    T = np.asarray(jax.device_get(T), dtype=np.float64)
    return float(np.sqrt(np.mean((T-exact)**2)) / np.sqrt(np.mean(exact**2)))


def ladder(levels, **kw):
    errs = [steady_error(n, **kw) for n in levels]
    orders = [np.log(errs[i]/errs[i+1])/np.log(levels[i+1]/levels[i])
              for i in range(len(errs)-1)]
    return errs, orders


def show(tag, levels, errs, orders):
    print(f"\n{tag}")
    print(f"  {'n':>6} {'err':>13} {'order':>8}")
    for i, n in enumerate(levels):
        o = "" if i == 0 else f"{orders[i-1]:8.3f}"
        print(f"  {n:6d} {errs[i]:13.6e} {o}")
    print(f"  observed(finest pair) = {orders[-1]:.4f}   max pairwise = {max(orders):.4f}")


if __name__ == "__main__":
    print("x64 =", jax.config.jax_enable_x64)
    for kind in ("release", "exp_sin"):
        lv = (10, 20, 40, 80, 160)
        e, o = ladder(lv, stencil_order=2, fourier=0.4, kind=kind)
        show(f"HeatNode stencil_order=2, profile={kind}, Fo=0.4  [declared 2]", lv, e, o)
        e, o = ladder(lv, stencil_order=4, fourier=0.3, kind=kind)
        show(f"HeatNode stencil_order=4, profile={kind}, Fo=0.3  [declared 4]", lv, e, o)
    # extend the 4th-order ladder to see whether any pair reaches 5.02
    lv = (5, 10, 20, 40, 80, 160, 320)
    e, o = ladder(lv, stencil_order=4, fourier=0.3, kind="release")
    show("HeatNode stencil_order=4, extended ladder, profile=release", lv, e, o)
