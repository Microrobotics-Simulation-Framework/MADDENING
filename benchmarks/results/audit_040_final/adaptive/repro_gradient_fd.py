"""Independent gradient verification for the frozen-active-set adjoint.

Step size: central differences in float64.  Truncation error ~ h^2 |J'''|/6,
round-off ~ eps_mach |J| / h with eps_mach = 2.2e-16, so the optimum is
h ~ (3 eps |J| / |J'''|)^(1/3) ~ 1e-5 for O(1) quantities.  The sweep below
shows the expected V: agreement improves to ~1e-9 relative around h=1e-5 and
degrades again for smaller h.  Every step is checked for an active-set change
over [theta-h, theta+h]; where the set changes, FD is not a valid oracle
(MADD-ANO-003) and that is reported separately.
"""
import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING-wt/audit/adaptive/tests")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from nodes.adaptive._toys import MaskedDenseNode, PoissonSineTopKNode


def study(node, theta0, label):
    st = node.initial_state()
    def J(t):
        p = {"theta": t}
        out = node.update(st, {}, 1.0, params=p)
        return node.objective(out, {**node.params, "theta": t})
    def sel(t):
        return np.asarray(node.compute_active_set(st, {**node.params, "theta": t}))
    g = float(jax.grad(J)(jnp.asarray(theta0)))
    print(f"\n== {label}  theta0={theta0}  jax.grad = {g:.15e}")
    m0 = sel(theta0)
    for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        same = np.array_equal(sel(theta0 - h), m0) and np.array_equal(sel(theta0 + h), m0)
        fd = float((J(jnp.asarray(theta0 + h)) - J(jnp.asarray(theta0 - h))) / (2 * h))
        rel = abs(fd - g) / max(abs(g), 1e-30)
        print(f"   h={h:<8.0e} FD={fd: .15e}  rel.err={rel:9.2e}  "
              f"{'active set constant' if same else '*** ACTIVE SET CHANGES ***'}")


for solver in ("gmres", "cg", "dense"):
    study(MaskedDenseNode(n=24, k=6, theta=0.3, solver=solver, blindness_gate=False),
          0.3, f"MaskedDenseNode solver={solver}")

study(PoissonSineTopKNode(n=128, k=32, theta=0.42, solver="cg", blindness_gate=False),
      0.42, "PoissonSineTopKNode k=32 solver=cg")

# Now deliberately near a switch of the top-K set.
node = PoissonSineTopKNode(n=128, k=16, theta=0.42, blindness_gate=False)
st = node.initial_state()
def sel(t):
    return np.asarray(node.compute_active_set(st, {**node.params, "theta": t}))
ts = np.linspace(0.40, 0.44, 4001)
prev = sel(ts[0]); switches = []
for t in ts[1:]:
    m = sel(t)
    if not np.array_equal(m, prev):
        switches.append(float(t)); prev = m
print(f"\n== switches of the top-16 set in theta in [0.40, 0.44]: {len(switches)}")
if switches:
    study(node, switches[0], "PoissonSineTopKNode exactly AT a switch")
    study(node, switches[0] + 2e-5, "PoissonSineTopKNode 2e-5 past the switch")
