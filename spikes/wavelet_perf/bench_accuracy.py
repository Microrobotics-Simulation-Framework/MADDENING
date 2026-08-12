"""LEVER 3: accuracy cost of fp32. Round-trip, CG convergence, adjoint."""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")
from bench_precision import build
from maddening.nodes.adaptive.wavelets import transform as T
from maddening.nodes.adaptive.wavelets import matrixfree as mf


def roundtrip(dim, nl, nc, order, dt):
    N = T.n_dofs(nl, nc, dim)
    synth = T._SYNTH[dim]
    ana = {1: T.analysis_1d, 2: T.analysis_2d, 3: T.analysis_3d}[dim]
    c = jax.random.normal(jax.random.PRNGKey(1), (N,), dtype=dt)
    u = synth(c, nl, nc, order)
    c2 = ana(u, nl, nc, order)
    return float(jnp.linalg.norm(c2 - c) / jnp.linalg.norm(c))


def cg_history(dt, nl=4, nc=4, dim=3, contrast=100.0, maxit=800):
    """Plain CG in dtype dt on the masked (full-mask) scaled operator.
    Track TRUE relative residual recomputed in fp64 each iter."""
    b = build(nl, nc, order=4, dim=dim, contrast=contrast, dt=dt)
    apply = b["apply"]
    N = b["N"]
    rhs = jax.random.normal(jax.random.PRNGKey(2), (N,), dtype=dt)
    rhs = rhs / jnp.linalg.norm(rhs)

    @jax.jit
    def step(carry):
        x, r, p, rs = carry
        Ap = apply(p)
        alpha = rs / jnp.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = jnp.dot(r, r)
        p = r + (rs_new / rs) * p
        return (x, r, p, rs_new)

    x = jnp.zeros(N, dtype=dt)
    r = rhs - apply(x)
    p = r
    rs = jnp.dot(r, r)
    carry = (x, r, p, rs)
    nb = float(jnp.linalg.norm(rhs.astype(jnp.float64)))
    hist = []
    # true residual in fp64 using an fp64 operator
    b64 = build(nl, nc, order=4, dim=dim, contrast=contrast, dt=jnp.float64)
    ap64 = jax.jit(b64["apply"])
    for k in range(maxit):
        carry = step(carry)
        if k % 25 == 0 or k == maxit - 1:
            xt = carry[0].astype(jnp.float64)
            true_r = float(jnp.linalg.norm(rhs.astype(jnp.float64) - ap64(xt)) / nb)
            hist.append((k + 1, true_r))
            if true_r < 1e-10:
                break
    return hist


def adjoint_checks(dt, nl=2, nc=4, dim=2, contrast=10.0):
    """J(theta) = <s, solve-free proxy>: use a fixed-iteration CG on the scaled op
    with theta driving the coefficient. Compare jit(grad) vs eager grad, and
    grad vs central FD."""
    side = nc * 2 ** nl
    h = 1.0 / side
    N = T.n_dofs(nl, nc, dim)
    from maddening.nodes.adaptive.wavelets import operator as op
    norms = op.column_norms_fast(nl, nc, 4, dim, h).astype(dt)
    D = jnp.ones(N, dtype=dt)
    srow = jax.random.normal(jax.random.PRNGKey(5), (N,), dtype=dt)
    rhs = jax.random.normal(jax.random.PRNGKey(6), (N,), dtype=dt)

    def J(theta):
        a = 1.0 + (contrast - 1.0) * jax.nn.sigmoid(theta)
        a_phys = mf.make_varcoeff_apply(a, side, dim, h, mass=1.0)
        apply = mf.make_wave_apply(nl, nc, 4, dim, norms, a_phys, D)
        # fixed 40-step CG (differentiable, unrolled) - deterministic graph
        x = jnp.zeros(N, dtype=dt)
        r = rhs - apply(x)
        p = r
        rs = jnp.dot(r, r)
        for _ in range(40):
            Ap = apply(p)
            alpha = rs / jnp.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = jnp.dot(r, r)
            p = r + (rs_new / rs) * p
            rs = rs_new
        return jnp.dot(srow, x)

    theta = jax.random.normal(jax.random.PRNGKey(7), (N,), dtype=dt) * 0.5
    g_eager = jax.grad(J)(theta)
    g_jit = jax.jit(jax.grad(J))(theta)
    jit_vs_eager = float(jnp.max(jnp.abs(g_jit.astype(jnp.float64)
                                         - g_eager.astype(jnp.float64)))
                         / jnp.max(jnp.abs(g_eager.astype(jnp.float64))))
    # FD on a few coords, step tuned per dtype
    eps = 1e-6 if dt == jnp.float32 else 1e-6
    idxs = [0, N // 3, N // 2, N - 1]
    errs = []
    for i in idxs:
        e = jnp.zeros(N, dtype=dt).at[i].set(1.0)
        fp = J(theta + eps * e)
        fm = J(theta - eps * e)
        fd = float((fp - fm) / (2 * eps))
        an = float(g_eager[i])
        errs.append(abs(fd - an) / max(abs(an), 1e-30))
    return jit_vs_eager, errs


if __name__ == "__main__":
    print("device:", jax.devices()[0])
    print("\n1) DD-4 lifting round-trip (analysis(synthesis(c)) vs c), rel L2:")
    for dim, nl, nc in [(1, 6, 4), (2, 4, 4), (3, 4, 4)]:
        for dt, name in [(jnp.float64, "fp64"), (jnp.float32, "fp32")]:
            e = roundtrip(dim, nl, nc, 4, dt)
            print(f"    dim={dim} n_levels={nl} {name}: {e:.3e}")

    print("\n2) CG on scaled op, 64^3, contrast=100 (TRUE rel residual in fp64):")
    for dt, name in [(jnp.float64, "fp64"), (jnp.float32, "fp32")]:
        h = cg_history(dt)
        tail = ", ".join(f"{k}:{v:.2e}" for k, v in h[-6:])
        best = min(v for _, v in h)
        print(f"    {name}: best true rel resid {best:.3e}  | tail {tail}")

    print("\n3) Adjoint (2D, 40-step unrolled CG):")
    for dt, name in [(jnp.float64, "fp64"), (jnp.float32, "fp32")]:
        jve, errs = adjoint_checks(dt)
        print(f"    {name}: jit-vs-eager rel {jve:.3e} | grad-vs-FD rel "
              + ", ".join(f"{e:.2e}" for e in errs))
