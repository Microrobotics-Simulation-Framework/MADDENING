"""LEVER 3: does an fp32-IR solve preserve the adjoint guarantees?

The node's adjoint is IMPLICIT (lineax native linear-solve autodiff via
ift_linear_solve): grad solves a second system with A^T, it does NOT backprop
through CG iterations. A hand-rolled IR fori_loop is OUTSIDE that path, so
jax.grad would unroll through every fp32 inner iteration.

Measure, in 2D (small enough to differentiate both ways):
  (1) grad through fp64 ift_linear_solve            <- the production adjoint
  (2) grad through a hand-rolled fp32-IR loop       <- the naive mixed-precision drop-in
  (3) jit-vs-eager for each
"""
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")
from maddening.nodes.adaptive.wavelets import transform as T
from maddening.nodes.adaptive.wavelets import operator as op
from maddening.nodes.adaptive.wavelets import matrixfree as mf
from maddening.core.solver_utils import ift_linear_solve

NL, NC, DIM, CONTRAST = 3, 4, 2, 10.0
side = NC * 2 ** NL
h = 1.0 / side
N = T.n_dofs(NL, NC, DIM)
norms64 = op.column_norms_fast(NL, NC, 4, DIM, h)
D64 = jnp.ones(N)
srow = jnp.asarray(np.random.RandomState(5).randn(N))
rhs = jnp.asarray(np.random.RandomState(6).randn(N))
rhs = rhs / jnp.linalg.norm(rhs)


def a_of(theta):
    return 1.0 + (CONTRAST - 1.0) * jax.nn.sigmoid(theta)


def J_ift(theta):
    """Production path: implicit adjoint via ift_linear_solve, all fp64."""
    a_phys = mf.make_varcoeff_apply(a_of(theta), side, DIM, h, mass=1.0)
    apply = mf.make_wave_apply(NL, NC, 4, DIM, norms64, a_phys, D64)
    x = ift_linear_solve(apply, rhs, solver="cg", rtol=1e-10, atol=1e-12)
    return jnp.dot(srow, x)


def J_ir(theta, outers=4, inner=60):
    """Naive mixed-precision drop-in: hand-rolled fp32 IR. grad unrolls it."""
    a32 = a_of(theta).astype(jnp.float32)
    ap32 = mf.make_wave_apply(NL, NC, 4, DIM, norms64.astype(jnp.float32),
                              mf.make_varcoeff_apply(a32, side, DIM, h, mass=1.0),
                              D64.astype(jnp.float32))
    ap64 = mf.make_wave_apply(NL, NC, 4, DIM, norms64,
                              mf.make_varcoeff_apply(a_of(theta), side, DIM, h,
                                                     mass=1.0), D64)
    x = jnp.zeros(N)
    for _ in range(outers):
        r = rhs - ap64(x)
        scale = jnp.linalg.norm(r)
        r32 = (r / scale).astype(jnp.float32)
        xi = jnp.zeros(N, jnp.float32); p = r32; rr = r32
        rs = jnp.dot(rr, rr)
        for _ in range(inner):
            Ap = ap32(p)
            alpha = rs / jnp.dot(p, Ap)
            xi = xi + alpha * p
            rr = rr - alpha * Ap
            rs_new = jnp.dot(rr, rr)
            p = rr + (rs_new / rs) * p
            rs = rs_new
        x = x + scale * xi.astype(jnp.float64)
    return jnp.dot(srow, x)


theta = jnp.asarray(np.random.RandomState(7).randn(N) * 0.5)

print("device:", jax.devices()[0], f" grid={side}^{DIM} N={N}")

# --- forward agreement ---
v_ift = float(J_ift(theta)); v_ir = float(J_ir(theta))
print(f"\n  forward: J_ift={v_ift:.12e}  J_ir={v_ir:.12e}  rel diff "
      f"{abs(v_ir-v_ift)/abs(v_ift):.3e}")

# --- gradients ---
g_ift = jax.grad(J_ift)(theta)
g_ift_jit = jax.jit(jax.grad(J_ift))(theta)
g_ir = jax.grad(J_ir)(theta)
g_ir_jit = jax.jit(jax.grad(J_ir))(theta)

def rel(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))

print(f"\n  jit-vs-eager  (fp64 ift, PRODUCTION) : {rel(g_ift_jit, g_ift):.3e}")
print(f"  jit-vs-eager  (fp32 IR, hand-rolled) : {rel(g_ir_jit, g_ir):.3e}")
print(f"\n  grad(fp32 IR) vs grad(fp64 ift)      : {rel(g_ir, g_ift):.3e}")
print(f"  cosine                               : "
      f"{float(jnp.dot(g_ir,g_ift)/(jnp.linalg.norm(g_ir)*jnp.linalg.norm(g_ift))):.10f}")

# --- FD reference for the production path ---
eps = 1e-6
errs = []
for i in [0, N // 3, N // 2, N - 1]:
    e = jnp.zeros(N).at[i].set(1.0)
    fd = float((J_ift(theta + eps * e) - J_ift(theta - eps * e)) / (2 * eps))
    errs.append(abs(fd - float(g_ift[i])) / max(abs(float(g_ift[i])), 1e-30))
print(f"\n  grad-vs-FD (fp64 ift, production)    : "
      + ", ".join(f"{e:.2e}" for e in errs))
