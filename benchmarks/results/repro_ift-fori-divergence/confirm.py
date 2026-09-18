"""Confirm: on a criterion-met exit, ift's state == one more pass of fori's."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro2 import Flow, Struct, build, measure, F
from maddening.core.graph_manager import GraphManager


def run(solver, **kw):
    gm = build(solver, **kw)
    init = {k: dict(v) for k, v in gm._state.items()}
    gm._state = {k: dict(v) for k, v in init.items()}
    out = gm.run_scan(1)
    return {"tau": float(out["flow"]["tau"]), "disp": float(out["struct"]["disp"])}, gm.coupling_diagnostics()


def one_gs_pass(st, rho, scale):
    """The group's own Gauss-Seidel pass F: flow then struct."""
    tau = 1.0 * scale + 1.0 * st["disp"]
    disp = rho * tau
    return {"tau": tau, "disp": disp}


print("=== A. interface norm, tiny interface values (MIME's configuration) ===")
for rho in (0.0214, 0.5, 0.99):
    kw = dict(scale=1e-9, rho=rho, norm="interface", accel="none",
              max_iterations=8, tolerance=1e-4)
    f, fd_ = run("fori", **kw)
    i, id_ = run("ift", **kw)
    pred = one_gs_pass(f, rho, 1e-9)
    print(f"rho={rho:<7} fori={f['tau']:.17g} ift={i['tau']:.17g} "
          f"rel={abs(i['tau']-f['tau'])/abs(f['tau']):.4%}")
    print(f"          F(fori)={pred['tau']:.17g}  ift-F(fori)={i['tau']-pred['tau']:.3e}"
          f"   fori_conv={fd_[next(iter(fd_))]} ift_conv={id_[next(iter(id_))]}")

print()
print("=== B. invariance of the gap to max_iterations and tolerance ===")
for m in (2, 8, 40, 200):
    for tol in (1e-4, 1e-14):
        kw = dict(scale=1e-9, rho=0.5, norm="interface", accel="none",
                  max_iterations=m, tolerance=tol)
        f, _ = run("fori", **kw); i, _ = run("ift", **kw)
        print(f"  m={m:<4} tol={tol:<7g} fori={f['tau']:.17g} ift={i['tau']:.17g} "
              f"rel={abs(i['tau']-f['tau'])/abs(f['tau']):.4%}")

print()
print("=== C. the live knob: atol (interface norm ignores `tolerance`) ===")
for atol in (1e-8, 1e-12, 1e-16, 0.0):
    kw = dict(scale=1e-9, rho=0.5, norm="interface", accel="none",
              max_iterations=200, tolerance=1e-4, atol=atol, rtol=1e-14)
    f, fdg = run("fori", **kw); i, _ = run("ift", **kw)
    print(f"  atol={atol:<8g} fori={f['tau']:.17g} ift={i['tau']:.17g} "
          f"rel={abs(i['tau']-f['tau'])/max(abs(f['tau']),1e-300):.4e}  "
          f"iters={fdg[next(iter(fdg))]['iterations']}")

print()
print("=== D. gradients: analytic vs each path's own central FD ===")
for rho, atol in ((0.0214, 1e-8), (0.5, 1e-8), (0.5, 0.0)):
    kw = dict(scale=1e-9, rho=rho, norm="interface", accel="none",
              max_iterations=200, tolerance=1e-4, atol=atol, rtol=1e-14)
    ff, fg, ffd, _ = measure("fori", **kw)
    iff, ig, ifd, _ = measure("ift", **kw)
    print(f"  rho={rho:<7} atol={atol:<7g} "
          f"fori |g-fd|/fd={abs(fg-ffd)/max(abs(ffd),1e-300):.2e}  "
          f"ift |g-fd|/fd={abs(ig-ifd)/max(abs(ifd),1e-300):.2e}")
