"""Round-7 Investigation 3: adjoint correctness under multi-mask CDD.

Question: when jax.grad differentiates through the CDD outer loop
that changes the mask at each iteration, is the resulting gradient
still the frozen-set adjoint of the final mask?

Empirical test: 1D sine basis (A diagonal, frozen-set adjoint
well-validated in round-2).  CDD with MAX_OUTER in {1, 3, 5, 10}.
Compare jax.grad(J_cdd) to FD of J_cdd at theta = 0.42 (no kink
nearby) and theta = 0.35 (kink-prone per round-2).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from trap_characterisation import (
    N_BASIS, LAMBDAS, x_grid, PHI, SENSOR_IDX,
    rhs_coeffs, J_full, grad_full,
)


def cdd_step(b, mask, c, theta_D):
    """One CDD outer iteration (APPLY+GROW+SOLVE) on the
    1D sine (A diagonal) problem.  No COARSE here."""
    # APPLY: r = b - A c, with A = diag(LAMBDAS)
    r = b - LAMBDAS * c
    # GROW (vectorized Doerfler bulk-chasing)
    r_sq = r * r
    target = theta_D * jnp.sum(r_sq)
    cand_sq = jnp.where(mask, 0.0, r_sq)
    sorted_sq = jnp.sort(cand_sq)[::-1]
    cumsum = jnp.cumsum(sorted_sq)
    n_add = jnp.searchsorted(cumsum, target) + 1
    rank = jnp.argsort(jnp.argsort(-cand_sq))
    added = (rank < n_add) & ~mask
    new_mask_pre_sg = mask | added
    new_mask = jax.lax.stop_gradient(new_mask_pre_sg)
    # SOLVE (diagonal trivial)
    c_new = jnp.where(new_mask, b / LAMBDAS, 0.0)
    return new_mask, c_new


def J_cdd_jax(theta, max_outer, with_intermediate_sg=False):
    """Python-unrolled CDD loop.  Returns u(x_sensor)."""
    b = rhs_coeffs(theta)
    mask = jnp.zeros(N_BASIS, dtype=bool)
    c = jnp.zeros(N_BASIS)
    for k in range(max_outer):
        mask, c = cdd_step(b, mask, c, theta_D=0.5)
        if with_intermediate_sg and k < max_outer - 1:
            c = jax.lax.stop_gradient(c)
    u = PHI @ c
    return u[SENSOR_IDX]


def J_cdd_no_sg(theta, max_outer):
    """Variant: CDD WITHOUT stop_gradient on mask.  Should still
    be valid because top-k via cumsum is non-differentiable to JAX.
    (Round-1 Q4 found this: stop_gradient is documentation when
    the selector is `>=` or top-k.)"""
    b = rhs_coeffs(theta)
    mask = jnp.zeros(N_BASIS, dtype=bool)
    c = jnp.zeros(N_BASIS)
    for k in range(max_outer):
        r = b - LAMBDAS * c
        r_sq = r * r
        target = 0.5 * jnp.sum(r_sq)
        cand_sq = jnp.where(mask, 0.0, r_sq)
        sorted_sq = jnp.sort(cand_sq)[::-1]
        cumsum = jnp.cumsum(sorted_sq)
        n_add = jnp.searchsorted(cumsum, target) + 1
        rank = jnp.argsort(jnp.argsort(-cand_sq))
        added = (rank < n_add) & ~mask
        mask = mask | added   # NO stop_gradient
        c = jnp.where(mask, b / LAMBDAS, 0.0)
    u = PHI @ c
    return u[SENSOR_IDX]


def fd_grad(fn, theta, h=1e-5, *args):
    return float((fn(theta + h, *args) - fn(theta - h, *args)) / (2 * h))


def find_kink_theta(max_outer, search_range=(0.30, 0.40), n=2001):
    """Find a theta where the CDD mask changes between theta-h and theta+h."""
    h = 1e-5
    thetas = np.linspace(*search_range, n)
    prev_mask = None

    def mask_at(theta):
        b = rhs_coeffs(jnp.asarray(theta))
        mask = jnp.zeros(N_BASIS, dtype=bool)
        c = jnp.zeros(N_BASIS)
        for _ in range(max_outer):
            mask, c = cdd_step(b, mask, c, 0.5)
        return np.asarray(mask)

    # binary search for a flip
    masks = [mask_at(t) for t in [thetas[0]]]
    for t in thetas[1:]:
        m = mask_at(t)
        if not np.array_equal(m, masks[-1]):
            # flip found near t
            return float(t - (thetas[1] - thetas[0]) / 2)
        masks = [m]
    return None


def part_b():
    print("# Part B -- jax.grad vs FD at non-kink and kink theta")
    print()
    print("## At theta = 0.42 (no kink in [theta+/-1e-5] -- baseline)")
    theta = 0.42
    print(f"  {'MAX_OUTER':>10} {'jax.grad':>14} {'FD':>14} {'rel_err':>12}")
    for max_outer in [1, 3, 5, 10]:
        g_jit = jax.jit(jax.grad(J_cdd_jax),
                        static_argnums=(1, 2))
        g_auto = float(g_jit(jnp.asarray(theta), max_outer, False))
        # FD
        h = 1e-5
        J_plus = float(J_cdd_jax(jnp.asarray(theta + h), max_outer))
        J_minus = float(J_cdd_jax(jnp.asarray(theta - h), max_outer))
        g_fd = (J_plus - J_minus) / (2 * h)
        rel = abs(g_auto - g_fd) / (abs(g_fd) + 1e-30)
        print(f"  {max_outer:>10d} {g_auto:>+14.6e} {g_fd:>+14.6e} {rel:>12.3e}")

    print()
    print("## Search for a kink theta with MAX_OUTER = 5")
    th_kink = find_kink_theta(5, (0.30, 0.40), n=2001)
    if th_kink is None:
        print("  No kink found in [0.30, 0.40] -- search wider")
        th_kink = find_kink_theta(5, (0.10, 0.90), n=8001)
    print(f"  Kink at theta ≈ {th_kink}" if th_kink else
          "  No kink found in [0.10, 0.90] for MAX_OUTER=5")

    if th_kink is not None:
        print()
        print(f"## At theta ≈ {th_kink:.5f} (kink in [theta+/-h])")
        for max_outer in [1, 3, 5, 10]:
            g_jit = jax.jit(jax.grad(J_cdd_jax),
                            static_argnums=(1, 2))
            g_auto = float(g_jit(jnp.asarray(th_kink), max_outer, False))
            h = 5e-4   # large enough to span the flip
            J_plus = float(J_cdd_jax(jnp.asarray(th_kink + h), max_outer))
            J_minus = float(J_cdd_jax(jnp.asarray(th_kink - h), max_outer))
            g_fd = (J_plus - J_minus) / (2 * h)
            rel = abs(g_auto - g_fd) / (abs(g_fd) + 1e-30)
            print(f"  MAX_OUTER={max_outer}: jax.grad={g_auto:+.4e}, "
                  f"FD={g_fd:+.4e}, rel_err={rel:.2e}")
        print()
        print("  Expected: at a kink, jax.grad returns Clarke subgradient")
        print("  (the value on the current side); FD picks up the jump.")


def part_c():
    print("\n\n# Part C -- stop_gradient on intermediate c (mitigation test)")
    print()
    print("## At theta = 0.42")
    theta = 0.42
    print(f"  {'MAX_OUTER':>10} {'without sg':>14} {'with sg':>14} "
          f"{'no_sg_on_mask':>16}")
    for max_outer in [1, 3, 5, 10]:
        g_auto = float(jax.jit(jax.grad(J_cdd_jax),
                                static_argnums=(1, 2))(
                                    jnp.asarray(theta), max_outer, False))
        g_with_sg = float(jax.jit(jax.grad(J_cdd_jax),
                                   static_argnums=(1, 2))(
                                       jnp.asarray(theta), max_outer, True))
        # also test without stop_gradient on mask
        g_no_mask_sg = float(jax.jit(jax.grad(J_cdd_no_sg),
                                      static_argnums=(1,))(
                                          jnp.asarray(theta), max_outer))
        print(f"  {max_outer:>10d} {g_auto:>+14.6e} {g_with_sg:>+14.6e} "
              f"{g_no_mask_sg:>+16.6e}")
    print()
    print("  Expected: all three columns identical because c only enters")
    print("  the next iteration via the (stop_gradient'd or non-diff) mask")
    print("  construction.  jnp.searchsorted+rank operations are")
    print("  non-differentiable to JAX, so the mask construction blocks")
    print("  c-gradient flow naturally -- stop_gradient is just documentation.")


if __name__ == "__main__":
    part_b()
    part_c()
