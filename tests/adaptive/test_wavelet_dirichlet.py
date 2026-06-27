"""M4 tests — Dirichlet boundary conditions (production geometry).

The boundary-adapted DD basis (:mod:`maddening.nodes.adaptive.wavelets.dirichlet`)
for homogeneous Dirichlet BCs, validated against the spike's parity numbers
(FINDINGS Inv 2A: 1D κ≈3.8, better than periodic; 2D tensor ≈150; wrong-sign
safe).  The conftest autouse fixture provides float64.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.nodes.adaptive import WaveletAdaptiveNode
from maddening.nodes.adaptive.wavelets import operator as OP
from maddening.nodes.adaptive.wavelets import precond as PC


def _kappa(A):
    ev = np.linalg.eigvalsh(np.asarray(A))
    return ev[-1] / ev[0]


def test_dirichlet_kappa_1d_better_than_periodic():
    """1D Dirichlet hybrid-Jacobi κ ≈ 3.8 (FINDINGS Inv 2A) — better than the
    periodic ≈ 20 because the +mass term removes the periodic near-null mode."""
    res = OP.assemble_wave_operator(4, 2, order=4, dim=1, mass=1.0,
                                    boundary="dirichlet")
    A, levels = res["A_dense"], res["levels"]
    D = PC.diagonal_scaling(jnp.diag(A), levels, "hybrid")
    k = _kappa((A / D[:, None]) / D[None, :])
    assert 2.5 < k < 6.0, f"1D Dirichlet κ={k}"


def test_dirichlet_kappa_2d_workable():
    """2D tensor Dirichlet κ is workable with Jacobi (FINDINGS Inv 2A ≈150)."""
    res = OP.assemble_wave_operator(4, 2, order=4, dim=2, mass=1.0,
                                    boundary="dirichlet")
    A, levels = res["A_dense"], res["levels"]
    D = PC.diagonal_scaling(jnp.diag(A), levels, "hybrid")
    k = _kappa((A / D[:, None]) / D[None, :])
    assert 80.0 < k < 250.0, f"2D Dirichlet κ={k}"


def test_dirichlet_node_cold_start_and_grad():
    node = WaveletAdaptiveNode(dim=1, n_levels=6, boundary="dirichlet",
                               theta_init=0.42)
    assert node.boundary == "dirichlet"
    s = node.initial_state()
    assert bool(jnp.all(s["mask"][jnp.asarray(node._coarse)]))

    def J(theta):
        e = {"c": jnp.zeros(node.N_max), "mask": jnp.zeros(node.N_max, bool),
             "theta": jnp.atleast_1d(theta)}
        m = node.compute_active_set(e, is_cold_start=True)
        st = node.solve_frozen({**e, "mask": m}, m)
        return jnp.squeeze(node._sensor(st))

    th = jnp.asarray(0.42)
    g = float(jax.grad(J)(th))
    e = 1e-5
    fd = float((J(th + e) - J(th - e)) / (2 * e))
    assert abs(g - fd) / (abs(fd) + 1e-30) < 1e-5


@pytest.mark.parametrize("theta", [0.05, 0.1, 0.5, 0.9, 0.95])
def test_dirichlet_wrong_sign_safe(theta):
    node = WaveletAdaptiveNode(dim=1, n_levels=6, boundary="dirichlet",
                               theta_init=theta)
    s = node._initial_state_impl()
    J_cdd = float(node._sensor(s))
    b = node._rhs_coeffs(jnp.atleast_1d(theta))
    J_full = float(node._srow @ jnp.linalg.solve(node._A, b))
    assert J_cdd * J_full >= -1e-20, f"wrong sign at theta={theta}"


def test_dirichlet_update_jittable_no_recompile():
    node = WaveletAdaptiveNode(dim=1, n_levels=6, boundary="dirichlet")
    s = node.initial_state()
    count = {"n": 0}

    @jax.jit
    def step(state):
        count["n"] += 1
        return node.update(state, {}, 1.0)

    s1 = step(s)
    step(s1)
    assert count["n"] == 1


def test_dirichlet_mms_converges():
    """MMS with u=sin(πx) (satisfies homogeneous Dirichlet): the Dirichlet
    Galerkin solve converges O(h²) to the analytic solution."""
    errs = []
    for nl in (4, 5):                       # side 47, 95
        res = OP.assemble_wave_operator(nl, 2, order=4, dim=1, mass=1.0,
                                        boundary="dirichlet")
        A, Wn, side, h = res["A_dense"], res["Wn"], res["side"], res["h"]
        x = np.arange(1, side + 1) / (side + 1)
        u_ex = np.sin(np.pi * x)
        f = (np.pi ** 2 + 1.0) * u_ex
        b = h * (Wn.T @ jnp.asarray(f))
        u_h = np.asarray(Wn @ jnp.linalg.solve(A, b))
        errs.append(np.linalg.norm(u_h - u_ex) / np.linalg.norm(u_ex))
    rate = np.log2(errs[0] / errs[1])
    assert errs[-1] < 5e-3, f"Dirichlet MMS relL2={errs[-1]}"
    assert rate > 1.8, f"Dirichlet MMS rate={rate}"
