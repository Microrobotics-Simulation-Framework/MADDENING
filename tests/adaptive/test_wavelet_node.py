"""M1 tests — WaveletAdaptiveNode (1D).

Mirrors ``tests/adaptive/test_hierarchical_hat_node.py`` (the local-basis
template) and adds the production JIT-path checks (the spike validated gradients
eager-only).  The ``conftest.py`` autouse fixture provides float64.
"""

from __future__ import annotations

import doctest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.compliance.metadata import StabilityLevel
from maddening.nodes.adaptive import WaveletAdaptiveNode
from maddening.nodes.adaptive import wavelet as wavelet_module


# ----------------------------------------------------------------------
# 1. construction
# ----------------------------------------------------------------------

def test_construct_with_defaults():
    node = WaveletAdaptiveNode(dim=1, n_levels=6)
    assert node.dim == 1
    assert node.side == 2 * 2 ** 6           # n_coarse=2, n_levels=6 -> 128
    assert node.N_max == node.side
    assert node.K == max(8, node.N_max // 16)
    assert node.state_fields() == ["c", "mask", "theta"]


def test_invalid_dim_raises():
    with pytest.raises(ValueError, match="dim"):
        WaveletAdaptiveNode(dim=4)


# ----------------------------------------------------------------------
# 2. cold-start
# ----------------------------------------------------------------------

def test_cold_start_succeeds_and_retains_coarse():
    node = WaveletAdaptiveNode(dim=1, n_levels=6, theta_init=0.42)
    s = node.initial_state()
    assert jnp.allclose(s["theta"], jnp.atleast_1d(0.42), atol=1e-12)
    # coarse level always active (wrong-sign-safety mechanism)
    assert bool(jnp.all(s["mask"][jnp.asarray(node._coarse)]))
    assert bool(s["mask"][0])               # index 0 is a coarse DOF


def test_cold_start_budget():
    node = WaveletAdaptiveNode(dim=1, n_levels=6, K=12, theta_init=0.42)
    s = node.initial_state()
    n_active = int(s["mask"].sum())
    # CDD grows to ~K (a small Doerfler overshoot is allowed)
    assert 12 <= n_active <= 12 + 8


# ----------------------------------------------------------------------
# 3. update invariants
# ----------------------------------------------------------------------

def test_update_shapes_and_mask():
    node = WaveletAdaptiveNode(dim=1, n_levels=6)
    s = node.initial_state()
    new_s = node.update(s, {}, 1.0)
    assert new_s["c"].shape == (node.N_max,)
    assert new_s["mask"].shape == (node.N_max,)
    assert new_s["mask"].dtype == jnp.bool_
    assert bool(jnp.all(new_s["mask"][jnp.asarray(node._coarse)]))


# ----------------------------------------------------------------------
# 4. differentiability (eager)
# ----------------------------------------------------------------------

def _J_of_theta(node, theta):
    empty = {
        "c": jnp.zeros(node.N_max, dtype=jnp.float64),
        "mask": jnp.zeros(node.N_max, dtype=bool),
        "theta": jnp.atleast_1d(theta),
    }
    mask = node.compute_active_set(empty, is_cold_start=True)
    st = node.solve_frozen({**empty, "mask": mask}, mask)
    return jnp.squeeze(node._sensor(st))


def test_autodiff_against_fd_at_non_kink():
    node = WaveletAdaptiveNode(dim=1, n_levels=6, theta_init=0.42)
    th = jnp.asarray(0.42)
    g = float(jax.grad(lambda t: _J_of_theta(node, t))(th))
    h = 1e-5
    g_fd = float((_J_of_theta(node, th + h) - _J_of_theta(node, th - h)) / (2 * h))
    rel = abs(g - g_fd) / (abs(g_fd) + 1e-30)
    assert rel < 1e-5, f"rel={rel}, g={g}, fd={g_fd}"


# ----------------------------------------------------------------------
# 5. production JIT path (spike validated eager-only)
# ----------------------------------------------------------------------

def test_update_is_jittable_no_recompile():
    node = WaveletAdaptiveNode(dim=1, n_levels=6)
    s = node.initial_state()
    count = {"n": 0}

    @jax.jit
    def step(state):
        count["n"] += 1
        return node.update(state, {}, 1.0)

    s1 = step(s)
    s2 = step(s1)
    step(s2)
    assert count["n"] == 1                   # compiled once, no silent recompile
    assert s1["c"].shape == (node.N_max,)


def test_jit_grad_matches_fd():
    node = WaveletAdaptiveNode(dim=1, n_levels=6, theta_init=0.42)
    Jc = jax.jit(lambda t: _J_of_theta(node, t))
    th = jnp.asarray(0.42)
    g = float(jax.jit(jax.grad(lambda t: _J_of_theta(node, t)))(th))
    h = 1e-5
    g_fd = float((Jc(th + h) - Jc(th - h)) / (2 * h))
    assert abs(g - g_fd) / (abs(g_fd) + 1e-30) < 1e-5


# ----------------------------------------------------------------------
# 6. accuracy vs full solve
# ----------------------------------------------------------------------

def test_cdd_accuracy_vs_full_solve():
    node = WaveletAdaptiveNode(dim=1, n_levels=7, theta_init=0.42)
    s = node.initial_state()
    J_cdd = float(node._sensor(s))
    # full (unmasked) reference in the same basis
    b = node._rhs_coeffs(jnp.atleast_1d(0.42))
    c_full = jnp.linalg.solve(node._A, b)
    J_full = float(node._srow @ c_full)
    assert abs(J_cdd - J_full) / (abs(J_full) + 1e-30) < 1e-2


# ----------------------------------------------------------------------
# 7. wrong-sign safety of the production selection (CDD)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("theta", [0.04, 0.08, 0.5, 0.92, 0.96])
def test_cdd_wrong_sign_safe(theta):
    """CDD (coarse-guaranteed) never produces a wrong-sign sensor reading vs
    the full solve, across a boundary sweep (FINDINGS §3 production claim)."""
    node = WaveletAdaptiveNode(dim=1, n_levels=6, theta_init=theta)
    s = node._initial_state_impl()           # bypass cold-start gate
    J_cdd = float(node._sensor(s))
    b = node._rhs_coeffs(jnp.atleast_1d(theta))
    J_full = float(node._srow @ jnp.linalg.solve(node._A, b))
    assert J_cdd * J_full >= -1e-20, f"wrong sign at theta={theta}"


# ----------------------------------------------------------------------
# 8. trap-immunity (local basis): blindness ratio ~ 1
# ----------------------------------------------------------------------

def test_blindness_ratio_near_one():
    node = WaveletAdaptiveNode(dim=1, n_levels=6, theta_init=0.42)
    r = node.blindness_ratio(node._initial_state_impl())
    # local wavelet basis is trap-immune (Gate 2): ratio comfortably above the
    # 0.7 blindness threshold
    assert r > 0.7


# ----------------------------------------------------------------------
# 9. preconditioner variants
# ----------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["hybrid", "full", "dk"])
def test_preconditioner_kinds_construct_and_solve(kind):
    node = WaveletAdaptiveNode(dim=1, n_levels=6, preconditioner=kind)
    s = node.initial_state()
    assert int(s["mask"].sum()) >= int(jnp.asarray(node._coarse).sum())


# ----------------------------------------------------------------------
# 10. metadata / compliance
# ----------------------------------------------------------------------

def test_node_metadata_present():
    meta = WaveletAdaptiveNode.meta
    assert meta.algorithm_id == "MADD-NODE-WAVELET"
    assert meta.description
    assert meta.governing_equations
    assert meta.assumptions and meta.limitations


def test_stability_tag_present():
    assert getattr(WaveletAdaptiveNode, "_stability_level", None) == \
        StabilityLevel.EXPERIMENTAL


# ----------------------------------------------------------------------
# 11. doctest
# ----------------------------------------------------------------------

def test_doctest_in_module_docstring():
    results = doctest.testmod(wavelet_module, verbose=False)
    assert results.failed == 0, f"doctest failed: {results}"
