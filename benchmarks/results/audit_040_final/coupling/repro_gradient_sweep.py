"""F5: IFT custom_jvp gradient vs analytic / FD / fori, across configs."""
import sys, warnings, itertools
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/benchmarks/results/audit_040_final/coupling")
warnings.simplefilter("ignore")
import jax, jax.numpy as jnp, numpy as np
from harness import graph

B = 0.5
SCALE = 1.0
EXACT_DTAU_DGAIN = SCALE / (1.0 - B)     # d tau*/d gain


def loss(p, **kw):
    gm = graph(**kw)
    return jnp.sum(gm.run_scan(1, params=p)["flow"]["tau"])


def probe(**kw):
    base = graph(**kw).params
    g = float(jax.grad(lambda p: loss(p, **kw))(base)["nodes"]["flow"]["gain"])
    # forward mode
    tang = jax.tree.map(jnp.zeros_like, base)
    tang["nodes"]["flow"]["gain"] = jnp.asarray(1.0)
    try:
        _, fwd = jax.jvp(lambda p: loss(p, **kw), (base,), (tang,))
        fwd = float(fwd)
    except Exception as e:
        fwd = f"ERR:{type(e).__name__}"
    # central difference
    h = 1e-2
    def sh(d):
        p = {"nodes": {n: dict(v) for n, v in base["nodes"].items()}}
        p["nodes"]["flow"]["gain"] = base["nodes"]["flow"]["gain"] + d
        return float(loss(p, **kw))
    fd = (sh(h) - sh(-h)) / (2 * h)
    gm = graph(**kw); gm.step()
    d = gm.coupling_diagnostics()["flow+struct"]
    return g, fwd, fd, d


print(f"analytic d(tau*)/d(gain) = {EXACT_DTAU_DGAIN}")
print()
print(f"{'config':<58} {'grad':>12} {'jvp':>12} {'fd':>12} "
      f"{'|g-exact|':>11} {'conv':>5} {'res':>10}")
cfgs = []
for solver in ("ift", "fori"):
    for acc in ("none", "aitken", "fixed", "iqn-ils", "iqn-imvj"):
        for mode in ("gauss-seidel", "jacobi"):
            kw = dict(solver=solver, acceleration=acc, iteration_mode=mode,
                      max_iterations=40, tolerance=1e-12, b=B, scale=SCALE)
            if acc == "fixed":
                kw["relaxation"] = 0.8
            if acc == "iqn-imvj":
                kw["jacobian_reuse"] = 2
            cfgs.append(kw)
for ls in ("gmres", "dense"):
    cfgs.append(dict(solver="ift", acceleration="none", max_iterations=40,
                     tolerance=1e-12, b=B, scale=SCALE, linear_solver=ls))
for norm in ("mixed", "interface"):
    cfgs.append(dict(solver="ift", acceleration="none", max_iterations=40,
                     convergence_norm=norm, atol=1e-12, rtol=1e-9,
                     b=B, scale=SCALE))

for kw in cfgs:
    label = ",".join(f"{k}={v}" for k, v in kw.items()
                     if k not in ("b", "scale", "max_iterations"))
    try:
        g, fwd, fd, d = probe(**kw)
        bad = abs(g - EXACT_DTAU_DGAIN)
        fs = f"{fwd:12.6f}" if isinstance(fwd, float) else f"{fwd:>12}"
        print(f"{label:<58} {g:12.6f} {fs} {fd:12.6f} {bad:11.2e} "
              f"{str(d['converged']):>5} {d['residual']:10.2e}")
    except Exception as e:
        print(f"{label:<58} EXCEPTION {type(e).__name__}: {str(e)[:70]}")
