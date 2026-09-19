"""Probe: does the documented 'top-K sensor error decreases monotonically'
claim (MADD-VER-004 / algorithm guide 'Validated Physical Regimes') hold away
from the single (theta=0.42, sigma=0.04, x_s=1/3) point it is asserted at?"""
import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING-wt/audit/adaptive/tests")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from nodes.adaptive._toys import PoissonSineTopKNode

def greens(x, theta, sigma, n_quad=20001):
    s = np.linspace(0.0, 1.0, n_quad)
    f = np.exp(-((s - theta) / sigma) ** 2)
    lo = np.minimum.outer(x, s); hi = np.maximum.outer(x, s)
    G = np.sinh(lo) * np.sinh(1.0 - hi) / np.sinh(1.0)
    return np.trapezoid(G * f, s, axis=1)

KS = (4, 8, 16, 32)
n = 256
print(f"{'theta':>6} {'x_s':>6} {'sel':>3} | " + " ".join(f"K={k:<10}" for k in KS) + " monotone?")
bad = 0; total = 0
for sel in ("b", "c"):
    for theta in (0.20, 0.30, 0.42, 0.50, 0.55, 0.70):
        for xs in (1/3, 0.25, 0.5, 0.75):
            jref = float(greens(np.array([xs]), theta, 0.04)[0])
            errs = []
            for k in KS:
                nd = PoissonSineTopKNode(n=n, k=k, theta=theta, sigma=0.04,
                                         sensor_x=xs, selection=sel, blindness_gate=False)
                errs.append(abs(float(nd.objective(nd.initial_state(), nd.params)) - jref))
            mono = all(a > b for a, b in zip(errs, errs[1:]))
            total += 1
            if not mono: bad += 1
            print(f"{theta:6.2f} {xs:6.3f} {sel:>3} | " + " ".join(f"{e:<12.3e}" for e in errs)
                  + ("  yes" if mono else "  NO"))
print(f"\nnon-monotone in {bad}/{total} configurations")
