"""M2 tests — WaveletAdaptiveNode in 2D and 3D (isotropic Mallat + BCOO autodiff).

Retires the spike's largest timeline risk (Limitation 11: 3D Mallat BCOO never
built/autodiff'd in JAX) and reproduces the FINDINGS closeout 3D numbers.  The
node is dimension-general, so these exercise dim=2/3 directly.

3D autodiff/JIT checks use a small grid (N=512) for speed; the production
numbers (kappa, sparsity) use the gate-2 grid (16^3 = 4096).  The conftest
autouse fixture provides float64.
"""

from __future__ import annotations

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.solver_utils import ift_linear_solve
from maddening.nodes.adaptive import WaveletAdaptiveNode
from maddening.nodes.adaptive.wavelets import operator as OP
from maddening.nodes.adaptive.wavelets import precond as PC


def _J_of_theta(node, theta):
    empty = {
        "c": jnp.zeros(node.N_max, dtype=jnp.float64),
        "mask": jnp.zeros(node.N_max, dtype=bool),
        "theta": jnp.atleast_1d(theta),
    }
    mask = node.compute_active_set(empty, is_cold_start=True)
    st = node.solve_frozen({**empty, "mask": mask}, mask)
    return jnp.squeeze(node._sensor(st))


# ----------------------------------------------------------------------
# 2D
# ----------------------------------------------------------------------

def test_2d_cold_start_and_grad():
    node = WaveletAdaptiveNode(dim=2, n_levels=3, n_coarse=2, theta_init=0.42)
    s = node.initial_state()
    assert bool(jnp.all(s["mask"][jnp.asarray(node._coarse)]))
    th = jnp.asarray(0.42)
    g = float(jax.grad(lambda t: _J_of_theta(node, t))(th))
    e = 1e-5
    fd = float((_J_of_theta(node, th + e) - _J_of_theta(node, th - e)) / (2 * e))
    assert abs(g - fd) / (abs(fd) + 1e-30) < 1e-6


def test_2d_update_jittable_no_recompile():
    node = WaveletAdaptiveNode(dim=2, n_levels=3, n_coarse=2)
    s = node.initial_state()
    count = {"n": 0}

    @jax.jit
    def step(state):
        count["n"] += 1
        return node.update(state, {}, 1.0)

    s1 = step(s)
    step(s1)
    assert count["n"] == 1


# ----------------------------------------------------------------------
# 3D — the BCOO autodiff risk gate
# ----------------------------------------------------------------------

def test_3d_isolated_bcoo_solve_grad_matches_fd():
    """3D Mallat BCOO solve: jax.grad through a single masked BCOO solve = FD to
    ~1e-9 (the spike's Inv 2 standard, now at 3D — Limitation 11 gate)."""
    nl, nc = 3, 1                       # 8^3 = 512
    res = OP.assemble_wave_operator(nl, nc, order=4, dim=3, mass=1.0)
    A, Wn, levels = res["A_dense"], res["Wn"], res["levels"]
    side, h, N = res["side"], res["h"], res["N"]
    D = PC.diagonal_scaling(jnp.diag(A), levels, "hybrid")
    Ah_bcoo = jsparse.BCOO.fromdense((A / D[:, None]) / D[None, :])
    coords = np.arange(side) / side
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    Xj, Yj, Zj = (jnp.asarray(g.reshape(-1)) for g in (X, Y, Z))
    sidx = int(np.argmin(np.abs(coords - 0.3)))
    srow = Wn[(sidx * side + sidx) * side + sidx] / D
    full = jnp.ones(N, dtype=bool)
    op = OP.make_masked_operator(Ah_bcoo, full)

    def J(theta):
        f = jnp.exp(-(((Xj - theta) ** 2 + (Yj - 0.5) ** 2 + (Zj - 0.5) ** 2) / 0.1 ** 2))
        b = (h ** 3 * (Wn.T @ f)) / D
        c = ift_linear_solve(op, b, solver="cg", rtol=1e-12, atol=1e-14)
        return srow @ c

    th = jnp.asarray(0.42)
    g = float(jax.grad(J)(th))
    e = 1e-5
    fd = float((J(th + e) - J(th - e)) / (2 * e))
    # Absolute agreement is ~4e-11 (machine-level); the relative figure is
    # FD-truncation-limited because the functional here is O(1e-3).  Both the
    # absolute and relative bounds below confirm BCOO autodiff is correct in 3D.
    assert abs(g - fd) < 1e-9, f"3D BCOO grad abs err {abs(g - fd)}"
    assert abs(g - fd) / (abs(fd) + 1e-30) < 1e-7


def test_3d_full_node_cold_start_and_grad():
    node = WaveletAdaptiveNode(dim=3, n_levels=3, n_coarse=1, theta_init=0.42)
    s = node.initial_state()
    assert bool(jnp.all(s["mask"][jnp.asarray(node._coarse)]))
    th = jnp.asarray(0.42)
    g = float(jax.grad(lambda t: _J_of_theta(node, t))(th))
    e = 1e-5
    fd = float((_J_of_theta(node, th + e) - _J_of_theta(node, th - e)) / (2 * e))
    assert abs(g - fd) / (abs(fd) + 1e-30) < 1e-6


def test_3d_update_jittable_no_recompile():
    node = WaveletAdaptiveNode(dim=3, n_levels=3, n_coarse=1)
    s = node.initial_state()
    count = {"n": 0}

    @jax.jit
    def step(state):
        count["n"] += 1
        return node.update(state, {}, 1.0)

    s1 = step(s)
    step(s1)
    assert count["n"] == 1


# ----------------------------------------------------------------------
# 3D production numbers (16^3 = 4096) — reproduce FINDINGS closeout
# ----------------------------------------------------------------------

@pytest.mark.slow
def test_3d_kappa_matches_findings():
    """Hybrid-Jacobi kappa at 16^3 ≈ 158 (FINDINGS closeout Inv 1)."""
    res = OP.assemble_wave_operator(4, 1, order=4, dim=3, mass=1.0)
    A, levels = res["A_dense"], res["levels"]
    D = PC.diagonal_scaling(jnp.diag(A), levels, "hybrid")
    Ah = (A / D[:, None]) / D[None, :]
    ev = np.linalg.eigvalsh(np.asarray(Ah))
    kappa = ev[-1] / ev[0]
    assert 120.0 < kappa < 200.0, f"3D hybrid kappa={kappa}"


@pytest.mark.slow
def test_3d_cdd_sparsity_jerr():
    """CDD at k=N/16 in 3D reaches J_err < 0.5% vs the full solve (FINDINGS
    closeout 1A reported ~0.1%)."""
    node = WaveletAdaptiveNode(dim=3, n_levels=4, n_coarse=1, sigma=0.10,
                               theta_init=0.42)
    s = node.initial_state()
    J_cdd = float(node._sensor(s))
    b = node._rhs_coeffs(jnp.atleast_1d(0.42))
    J_full = float(node._srow @ jnp.linalg.solve(node._A, b))
    assert abs(J_cdd - J_full) / (abs(J_full) + 1e-30) < 5e-3
