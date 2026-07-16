"""M19 tests — WaveletVarcoeffNode (matrix-free θ→A, coefficient gradient).

The θ→A path: the coefficient field χ is the differentiable parameter, the
operator is assembled matrix-free in-trace, and dJ/dχ flows through assembly.
The conftest.py autouse fixture provides float64.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.nodes.adaptive import WaveletVarcoeffNode


def _frozen_J(node, chi0):
    """Sensor as a function of χ at a mask frozen at χ0 (the production adjoint)."""
    N = node.N_max
    empty = {"c": jnp.zeros(N), "mask": jnp.zeros(N, bool), "theta": chi0}
    mask = jax.lax.stop_gradient(node.compute_active_set(empty, is_cold_start=True))

    def J(chi):
        st = node.solve_frozen({**empty, "theta": chi, "mask": mask}, mask)
        return jnp.squeeze(node._sensor(st))

    return J


def test_construct_cold_start_and_update():
    node = WaveletVarcoeffNode(dim=2, n_levels=3, n_coarse=2, mass=1.0)
    s = node.initial_state()
    assert s["c"].shape == (node.N_max,)
    assert int(s["mask"].sum()) >= int(jnp.asarray(node._coarse).sum())
    s2 = node.update(s, {}, 1.0)
    assert s2["c"].shape == (node.N_max,)
    assert bool(jnp.all(jnp.isfinite(s2["c"])))


def test_operator_depends_on_chi():
    """The θ→A seam is real: changing χ changes the operator, hence the solve."""
    node = WaveletVarcoeffNode(dim=2, n_levels=3, n_coarse=2, mass=1.0)
    N = node.N_max
    empty = {"c": jnp.zeros(N), "mask": jnp.zeros(N, bool),
             "theta": jnp.zeros(N)}
    mask = jax.lax.stop_gradient(node.compute_active_set(empty, is_cold_start=True))
    c0 = node.solve_frozen({**empty, "mask": mask}, mask)["c"]
    chi = jnp.asarray(50.0 * np.exp(
        -((np.arange(N) - N / 2) / (N / 8)) ** 2))          # contrast ~50 inclusion
    c1 = node.solve_frozen({**empty, "theta": chi, "mask": mask}, mask)["c"]
    assert float(jnp.linalg.norm(c1 - c0)) > 1e-8           # operator moved


def test_djdchi_grad_vs_fd_and_jit_2d():
    """dJ/dχ flows through matrix-free operator assembly; grad matches FD (FD-
    truncation-limited) and jit(grad) equals eager grad to machine precision —
    the adjoint is exact and jit-stable (2D)."""
    node = WaveletVarcoeffNode(dim=2, n_levels=3, n_coarse=2, mass=1.0)
    N = node.N_max
    chi0 = jnp.asarray(0.3 * np.cos(2 * np.pi * np.arange(N) / N))
    J = _frozen_J(node, chi0)
    g = jax.grad(J)(chi0)
    gj = jax.jit(jax.grad(J))(chi0)
    # exactness + jit stability
    assert float(jnp.max(jnp.abs(g - gj)) / (jnp.max(jnp.abs(g)) + 1e-30)) < 1e-9
    # correctness vs FD (central, e=1e-4; coefficient gradients are FD-limited, D2)
    rng = np.random.default_rng(1)
    e = 1e-4
    for k in rng.choice(N, 5, replace=False):
        k = int(k)
        fd = float((J(chi0.at[k].add(e)) - J(chi0.at[k].add(-e))) / (2 * e))
        assert abs(float(g[k]) - fd) / (abs(fd) + 1e-30) < 1e-3


@pytest.mark.slow
def test_djdchi_grad_vs_fd_and_jit_3d():
    """Same, 3D (8³) — the gate-2 grid.  Slow lane (jit-through-CG compile)."""
    node = WaveletVarcoeffNode(dim=3, n_levels=3, n_coarse=1, mass=1.0)
    N = node.N_max
    chi0 = jnp.asarray(0.3 * np.cos(2 * np.pi * np.arange(N) / N))
    J = _frozen_J(node, chi0)
    g = jax.grad(J)(chi0)
    gj = jax.jit(jax.grad(J))(chi0)
    assert float(jnp.max(jnp.abs(g - gj)) / (jnp.max(jnp.abs(g)) + 1e-30)) < 1e-9
    rng = np.random.default_rng(2)
    e = 1e-4
    for k in rng.choice(N, 5, replace=False):
        k = int(k)
        fd = float((J(chi0.at[k].add(e)) - J(chi0.at[k].add(-e))) / (2 * e))
        assert abs(float(g[k]) - fd) / (abs(fd) + 1e-30) < 1e-3


def test_metadata_and_stability():
    from maddening.core.compliance.metadata import StabilityLevel
    meta = WaveletVarcoeffNode.meta
    assert meta.algorithm_id == "MADD-NODE-WAVELET-VARCOEFF"
    assert "χ ≤ 10" in " ".join(meta.limitations)          # scope cap stated
    assert getattr(WaveletVarcoeffNode, "_stability_level", None) == \
        StabilityLevel.EXPERIMENTAL


# ----------------------------------------------------------------------
# M20 — magnetostatic RHS -∇·(χH₀): χ enters through both A(χ) and b(χ)
# ----------------------------------------------------------------------

def test_magnetic_rhs_matches_direct_fd():
    """The node's -∇·(χH₀) source equals a direct central-difference divergence."""
    node = WaveletVarcoeffNode(dim=1, n_levels=6, n_coarse=2, mass=0.5, h0=(1.0,))
    side, h = node.side, node._h
    x = np.arange(side) / side
    chi = jnp.asarray(0.5 * np.exp(-((x - 0.5) / 0.1) ** 2))
    f_node = np.asarray(node._magnetic_source(chi))
    f_direct = -(np.roll(np.asarray(chi), -1) - np.roll(np.asarray(chi), 1)) / (2 * h)
    assert np.max(np.abs(f_node - f_direct)) < 1e-12


def test_magnetics_full_basis_matches_dense_fd():
    """Full-basis magnetics solve reproduces a dense FD solve of the same PDE
    -∇·((1+χ)∇φ)+mφ = -∇·(χH₀) — an assembly-consistency check on the χ-dependent
    RHS and operator (NOT an independent physical validation; that is M23)."""
    from maddening.nodes.adaptive.wavelets import operator as OP
    dim, nl, nc, m = 1, 6, 2, 0.5
    node = WaveletVarcoeffNode(dim=dim, n_levels=nl, n_coarse=nc, mass=m, h0=(1.0,))
    side, h = node.side, node._h
    x = np.arange(side) / side
    chi = jnp.asarray(0.5 * np.exp(-((x - 0.5) / 0.1) ** 2))
    a = jnp.asarray(1.0 + np.asarray(chi))
    f = node._magnetic_source(chi)
    res = OP.assemble_wave_operator(nl, nc, 4, dim, mass=m, a_grid=a)
    Awave, Wn = res["A_dense"], res["Wn"]
    phi_wav = np.asarray(Wn @ jnp.linalg.solve(Awave, (h ** dim) * (Wn.T @ f)))
    A_phys = np.asarray(OP.physical_varcoeff(a, dim, h, mass=m))
    phi_fd = np.linalg.solve(A_phys, (h ** dim) * np.asarray(f))
    assert np.linalg.norm(phi_wav - phi_fd) / np.linalg.norm(phi_fd) < 1e-10


@pytest.mark.parametrize("dim,nl,nc", [(1, 6, 2), (2, 3, 2)])
def test_djdchi_through_A_and_b(dim, nl, nc):
    """dJ/dχ with the magnetics RHS: χ enters BOTH the operator A(χ) and the RHS
    b(χ), and the gradient through both matches FD and is jit-stable."""
    h0 = (1.0,) if dim == 1 else (0.0, 1.0)
    node = WaveletVarcoeffNode(dim=dim, n_levels=nl, n_coarse=nc, mass=0.5, h0=h0)
    N = node.N_max
    chi0 = jnp.asarray(0.3 * np.cos(2 * np.pi * np.arange(N) / N))
    J = _frozen_J(node, chi0)
    g = jax.grad(J)(chi0)
    gj = jax.jit(jax.grad(J))(chi0)
    assert float(jnp.max(jnp.abs(g - gj)) / (jnp.max(jnp.abs(g)) + 1e-30)) < 1e-9
    rng = np.random.default_rng(3)
    e = 1e-4
    for k in rng.choice(N, 5, replace=False):
        k = int(k)
        fd = float((J(chi0.at[k].add(e)) - J(chi0.at[k].add(-e))) / (2 * e))
        assert abs(float(g[k]) - fd) / (abs(fd) + 1e-30) < 2e-3
