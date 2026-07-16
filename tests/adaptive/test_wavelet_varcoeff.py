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
