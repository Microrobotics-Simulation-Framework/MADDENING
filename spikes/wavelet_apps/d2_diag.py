"""D2 diagnosis — is the 32^2 miss a wrong gradient or FD truncation?

Re-probe 32^2 with (a) multiple central-FD steps, (b) a 4th-order FD stencil as
a better reference, (c) absolute error and |fd| magnitude reported so we can see
whether the large *relative* error is just small-denominator inflation.
"""
from __future__ import annotations
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from d2_djda_nd import build

J, N, side = build(2, 4, 2)   # 32^2
a0 = jnp.ones(N)
g = jax.grad(J)(a0)
rng = np.random.default_rng(0)
probes = rng.choice(N, size=6, replace=False)


def fd2(k, e):
    return float((J(a0.at[k].add(e)) - J(a0.at[k].add(-e))) / (2 * e))


def fd4(k, e):
    p1 = float(J(a0.at[k].add(e))); m1 = float(J(a0.at[k].add(-e)))
    p2 = float(J(a0.at[k].add(2 * e))); m2 = float(J(a0.at[k].add(-2 * e)))
    return (-p2 + 8 * p1 - 8 * m1 + m2) / (12 * e)


print(f"{'k':>5} {'analytic':>12} {'|analytic|':>11} "
      f"{'relerr e=1e-4':>13} {'relerr e=1e-6':>13} {'relerr fd4':>11}")
worst4 = 0.0
for k in probes:
    ga = float(g[k])
    r_e4 = abs(ga - fd2(k, 1e-4)) / (abs(fd2(k, 1e-4)) + 1e-30)
    r_e6 = abs(ga - fd2(k, 1e-6)) / (abs(fd2(k, 1e-6)) + 1e-30)
    ref4 = fd4(k, 1e-4)
    r_4 = abs(ga - ref4) / (abs(ref4) + 1e-30)
    worst4 = max(worst4, r_4)
    print(f"{int(k):>5} {ga:>12.4e} {abs(ga):>11.3e} {r_e4:>13.3e} {r_e6:>13.3e} {r_4:>11.3e}")
print()
print(f"worst relerr vs 4th-order FD: {worst4:.3e}  ->",
      "PASS (FD truncation was the culprit)" if worst4 < 1e-3 else "still FAIL")
