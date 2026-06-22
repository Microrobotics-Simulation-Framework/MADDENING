"""Investigation 2 -- DD-4 wavelet operator construction in JAX (PoC).

SPIKE CODE. Engineering proof-of-concept, NOT production. Goal: surface any
JAX show-stoppers before committing to the BCOO sparse-tree implementation.

Part A: 1D DD-4 forward/inverse transform in JAX (lifting via jnp.roll,
        periodic), JIT-compilable, matches numpy ref to 1e-12; jax.grad
        through synthesis matches FD.
Part B: BCOO stiffness A_wave = W^T A_phys W; lineax.linear_solve through a
        FunctionLinearOperator over the BCOO matvec; jax.grad vs FD to 1e-9.
Part C: 2D isotropic Mallat (3 subbands) transform in JAX, JIT; BCOO nnz.
"""

from __future__ import annotations

import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np
import lineax

import dd_wavelets as dd

# DD-4 4-point prediction filter (offsets relative to left coarse nbr).
_OFF = jnp.array([-1, 0, 1, 2])
_W = jnp.array([-1 / 16, 9 / 16, 9 / 16, -1 / 16])


def _predict_mid(coarse):
    """Periodic DD-4 midpoint prediction (vectorised, roll-based)."""
    out = jnp.zeros_like(coarse)
    for off, w in zip([-1, 0, 1, 2], [-1/16, 9/16, 9/16, -1/16]):
        out = out + w * jnp.roll(coarse, -off)
    return out


def synthesis_1d(coeffs, n_levels, n_coarse):
    """Inverse interpolating-wavelet transform (coeffs -> grid values).

    Layout matches dd_wavelets.synthesis_matrix (periodic): coarse(n_coarse),
    then detail blocks of sizes n_coarse, 2 n_coarse, ... All static shapes,
    so the Python loop unrolls cleanly under jit.
    """
    idx = 0
    vals = jax.lax.dynamic_slice_in_dim(coeffs, 0, n_coarse)
    idx += n_coarse
    cur = n_coarse
    for _ in range(n_levels):
        detail = jax.lax.dynamic_slice_in_dim(coeffs, idx, cur)
        idx += cur
        mids = _predict_mid(vals) + detail
        fine = jnp.zeros(2 * cur, dtype=coeffs.dtype)
        fine = fine.at[0::2].set(vals)
        fine = fine.at[1::2].set(mids)
        vals = fine
        cur *= 2
    return vals


def analysis_1d(values, n_levels, n_coarse):
    """Forward transform (grid values -> coeffs), inverse of synthesis_1d."""
    blocks = []
    vals = values
    cur = values.shape[0]
    for _ in range(n_levels):
        coarse = vals[0::2]
        mids = vals[1::2]
        detail = mids - _predict_mid(coarse)
        blocks.append(detail)
        vals = coarse
        cur //= 2
    # vals now is the coarse scaling block; assemble [coarse, d0, d1, ...]
    out = [vals] + blocks  # blocks are finest-first; synthesis consumed coarse-first
    out = [vals] + blocks[::-1]
    return jnp.concatenate(out)


def part_A():
    print("=" * 78)
    print("PART A -- 1D DD-4 transform in JAX")
    print("=" * 78)
    n_levels, n_coarse = 7, 2
    N = n_coarse * 2 ** n_levels  # 256
    Wnp, levels, x = dd.synthesis_matrix(n_levels, n_coarse, 4, "periodic")

    synth = jax.jit(lambda c: synthesis_1d(c, n_levels, n_coarse))
    t0 = time.time()
    _ = synth(jnp.zeros(N)).block_until_ready()
    t_compile = time.time() - t0

    # match numpy reference (W @ coeffs) to 1e-12
    rng = np.random.default_rng(0)
    c = rng.standard_normal(N)
    ref = Wnp @ c
    got = np.asarray(synth(jnp.asarray(c)))
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"  N={N}: synthesis vs numpy W@c rel err = {rel:.2e}  "
          f"({'MATCH' if rel < 1e-12 else 'MISMATCH'})")
    print(f"  JIT compile time: {t_compile*1000:.0f} ms")

    # forward/inverse roundtrip
    rt = np.asarray(jax.jit(lambda v: synthesis_1d(analysis_1d(v, n_levels, n_coarse),
                                                   n_levels, n_coarse))(jnp.asarray(ref)))
    rt_err = np.linalg.norm(rt - ref) / np.linalg.norm(ref)
    print(f"  forward∘inverse roundtrip rel err = {rt_err:.2e}")

    # exec time at several N
    for nl in (7, 9, 11):
        Nn = n_coarse * 2 ** nl
        s = jax.jit(lambda c: synthesis_1d(c, nl, n_coarse))
        cc = jnp.asarray(rng.standard_normal(Nn))
        s(cc).block_until_ready()
        t0 = time.time()
        for _ in range(100):
            s(cc).block_until_ready()
        print(f"  N={Nn}: {(time.time()-t0)/100*1e6:.1f} us/call")

    # jax.grad through synthesis + a dense masked solve
    A_phys = dd.laplacian_periodic(N, mass=True)
    Wn = Wnp / np.sqrt((1.0/N) * np.sum(Wnp**2, axis=0))[None, :]
    A_wave = jnp.asarray(Wn.T @ A_phys @ Wn)
    srow = jnp.asarray(Wn[N // 3])

    def objective(theta):
        # source -> coeffs (smooth), solve, sensor reading
        f = jnp.exp(-((jnp.asarray(x) - theta) / 0.06) ** 2)
        b = (1.0 / N) * (jnp.asarray(Wn).T @ f)
        c = jnp.linalg.solve(A_wave, b)
        return srow @ c

    g = float(jax.grad(objective)(jnp.asarray(0.3)))
    eps = 1e-6
    fd = float((objective(jnp.asarray(0.3 + eps)) - objective(jnp.asarray(0.3 - eps))) / (2 * eps))
    print(f"  jax.grad through solve vs FD: grad={g:.6e} fd={fd:.6e} "
          f"rel={abs(g-fd)/(abs(fd)+1e-30):.2e}")


def part_B():
    print("=" * 78)
    print("PART B -- BCOO stiffness + lineax solve + autodiff")
    print("=" * 78)
    for N, nl in [(256, 7), (1024, 9)]:
        Wnp, levels, x = dd.synthesis_matrix(nl, 2, 4, "periodic")
        Wn = Wnp / np.sqrt((1.0/N) * np.sum(Wnp**2, axis=0))[None, :]
        A_wave = Wn.T @ dd.laplacian_periodic(N, mass=True) @ Wn
        A_wave = 0.5 * (A_wave + A_wave.T)
        thresh = 1e-12 * np.abs(A_wave).max()
        A_sp = np.where(np.abs(A_wave) >= thresh, A_wave, 0.0)
        nnz = int(np.count_nonzero(A_sp))
        A_bcoo = jsparse.BCOO.fromdense(jnp.asarray(A_sp))
        mem_kb = (A_bcoo.data.nbytes + A_bcoo.indices.nbytes) / 1024
        print(f"\n  N={N}: nnz={nnz} ({100*nnz/N**2:.1f}% dense), "
              f"BCOO mem={mem_kb:.0f} KB (dense would be {N*N*8/1024:.0f} KB)")

        # lineax solve through BCOO matvec
        D = jnp.asarray(2.0 ** levels.astype(float))
        def make_solve(rhs_scale):
            def solve_for(theta):
                f = jnp.exp(-((jnp.asarray(x) - theta) / 0.06) ** 2)
                b = (1.0 / N) * (jnp.asarray(Wn).T @ f)
                op = lineax.FunctionLinearOperator(
                    lambda v: A_bcoo @ v,
                    jax.ShapeDtypeStruct((N,), jnp.float64),
                    tags=lineax.positive_semidefinite_tag,
                )
                sol = lineax.linear_solve(
                    op, b,
                    solver=lineax.GMRES(rtol=1e-12, atol=1e-14,
                                        restart=N, stagnation_iters=50),
                )
                return jnp.asarray(Wn[N // 3]) @ sol.value
            return solve_for
        solve_for = make_solve(D)
        try:
            g = float(jax.grad(solve_for)(jnp.asarray(0.3)))
            eps = 1e-6
            fd = float((solve_for(jnp.asarray(0.3+eps)) - solve_for(jnp.asarray(0.3-eps)))/(2*eps))
            rel = abs(g - fd) / (abs(fd) + 1e-30)
            print(f"  lineax+BCOO jax.grad vs FD: rel={rel:.2e}  "
                  f"({'PASS' if rel < 1e-9 else 'CHECK'})")
        except Exception as e:
            print(f"  ERROR in lineax+BCOO autodiff: {type(e).__name__}: {e}")

        # masked matvec closure pattern (round-6)
        mask = jnp.zeros(N, bool).at[jnp.arange(N // 8)].set(True)
        masked_mv = lambda v: jnp.where(mask, A_bcoo @ jnp.where(mask, v, 0.0), 0.0)
        try:
            _ = jax.jit(masked_mv)(jnp.ones(N)).block_until_ready()
            print("  masked matvec closure: JIT OK")
        except Exception as e:
            print(f"  masked matvec ERROR: {type(e).__name__}: {e}")


def part_C():
    print("=" * 78)
    print("PART C -- 2D isotropic Mallat in JAX (32x32)")
    print("=" * 78)
    nl, nc = 4, 2
    Nside = nc * 2 ** nl  # 32
    N = Nside * Nside

    def synth2d(flat):
        idx = 0
        sz = nc
        img = jax.lax.dynamic_slice_in_dim(flat, idx, sz*sz).reshape(sz, sz)
        idx += sz*sz
        cur = nc
        for _ in range(nl):
            dLH = jax.lax.dynamic_slice_in_dim(flat, idx, cur*cur).reshape(cur, cur); idx += cur*cur
            dHL = jax.lax.dynamic_slice_in_dim(flat, idx, cur*cur).reshape(cur, cur); idx += cur*cur
            dHH = jax.lax.dynamic_slice_in_dim(flat, idx, cur*cur).reshape(cur, cur); idx += cur*cur
            # predict along axes
            py = jnp.apply_along_axis(_predict_mid, 1, img)  # midpoints along y
            px = jnp.apply_along_axis(_predict_mid, 0, img)
            pxy = jnp.apply_along_axis(_predict_mid, 0, py)
            fine = jnp.zeros((2*cur, 2*cur), dtype=flat.dtype)
            fine = fine.at[0::2, 0::2].set(img)
            fine = fine.at[0::2, 1::2].set(py + dLH)
            fine = fine.at[1::2, 0::2].set(px + dHL)
            fine = fine.at[1::2, 1::2].set(pxy + dHH)
            img = fine
            cur *= 2
        return img.reshape(-1)

    Wnp, levels, n = dd.synthesis_matrix_2d_isotropic(nl, nc, 4)
    try:
        s2 = jax.jit(synth2d)
        t0 = time.time(); _ = s2(jnp.zeros(N)).block_until_ready()
        tc = time.time() - t0
        rng = np.random.default_rng(0); c = rng.standard_normal(N)
        rel = np.linalg.norm(np.asarray(s2(jnp.asarray(c))) - Wnp @ c) / np.linalg.norm(Wnp @ c)
        print(f"  2D synth JIT compile {tc*1000:.0f} ms; vs numpy rel={rel:.2e} "
              f"({'MATCH' if rel < 1e-10 else 'MISMATCH'})")
    except Exception as e:
        print(f"  2D synth ERROR: {type(e).__name__}: {e}")

    # BCOO stiffness nnz at 2D
    h = 1.0 / Nside
    Wn = Wnp / np.sqrt((h*h) * np.sum(Wnp**2, axis=0))[None, :]
    S = dd.laplacian_periodic(Nside, mass=False); Mm = h*np.eye(Nside)
    A2 = np.kron(S, Mm) + np.kron(Mm, S) + np.kron(Mm, Mm)
    Aw = Wn.T @ A2 @ Wn
    nnz = int(np.count_nonzero(np.abs(Aw) >= 1e-12 * np.abs(Aw).max()))
    print(f"  2D BCOO nnz at N={N}: {nnz} ({100*nnz/N**2:.1f}% dense, "
          f"{nnz/N:.0f} nnz/row)")


if __name__ == "__main__":
    part_A()
    part_B()
    part_C()
