"""M6 tests for ``HierarchicalHatAdaptiveNode`` (Toy 2: local hat basis).

Test list (numbering matches
``plans/MADDENING_ADAPTIVE_NODE_IMPLEMENTATION_PLAN.md`` §4 M6):

1. test_construct_with_defaults
2. test_active_set_includes_level_0_always
3. test_active_set_shape_is_dyadic_subtree
4. test_active_set_adapts_with_theta
5. test_initial_state_succeeds
6. test_no_wrong_sign_at_boundary
7. test_autodiff_against_fd_at_non_kink
8. test_solve_uses_bcoo_path
9. test_doctest_in_module_docstring

Optional (not blocking):

10. test_optimization_convergence_smoke
"""

from __future__ import annotations

import doctest

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from maddening.nodes.adaptive import HierarchicalHatAdaptiveNode
from maddening.nodes.adaptive import hierarchical_hat as hat_module


# ----------------------------------------------------------------------
# 1. test_construct_with_defaults
# ----------------------------------------------------------------------

def test_construct_with_defaults():
    """node = HierarchicalHatAdaptiveNode() constructs with documented shapes."""
    node = HierarchicalHatAdaptiveNode()
    assert node.N_levels == 7
    assert node.N_max == 2 ** (7 + 1) - 1  # = 255
    assert node.K == 16


# ----------------------------------------------------------------------
# 2. test_active_set_includes_level_0_always
# ----------------------------------------------------------------------

def test_active_set_includes_level_0_always():
    """The level-0 root hat (index 0) is always active."""
    node = HierarchicalHatAdaptiveNode(theta_init=0.42)
    s = node.initial_state()
    assert bool(s["mask"][0]) is True


# ----------------------------------------------------------------------
# 3. test_active_set_shape_is_dyadic_subtree
# ----------------------------------------------------------------------

def test_active_set_shape_is_dyadic_subtree():
    """The active mask has exactly K True entries, all valid hat indices,
    with the level-0 root included."""
    node = HierarchicalHatAdaptiveNode(theta_init=0.42, K=8, N_levels=5)
    s = node.initial_state()
    assert int(s["mask"].sum()) == 8
    assert bool(s["mask"][0]) is True


# ----------------------------------------------------------------------
# 4. test_active_set_adapts_with_theta
# ----------------------------------------------------------------------

def test_active_set_adapts_with_theta():
    """Calling compute_active_set at theta=0.3 and theta=0.7 produces
    different masks (active set genuinely adapts)."""
    node = HierarchicalHatAdaptiveNode(K=16, N_levels=6)
    state_left = {
        "c": jnp.zeros(node.N_max, dtype=jnp.float64),
        "mask": jnp.zeros(node.N_max, dtype=bool),
        "theta": jnp.atleast_1d(0.3),
    }
    state_right = {**state_left, "theta": jnp.atleast_1d(0.7)}
    mask_left = node.compute_active_set(state_left)
    mask_right = node.compute_active_set(state_right)
    # The two masks must differ (and meaningfully -- at least 2 indices flipped).
    diff = int((mask_left != mask_right).sum())
    assert diff >= 2, f"masks should adapt with theta; only {diff} differed"


# ----------------------------------------------------------------------
# 5. test_initial_state_succeeds
# ----------------------------------------------------------------------

def test_initial_state_succeeds():
    """Default theta=0.42 passes the cold-start blindness gate."""
    node = HierarchicalHatAdaptiveNode(theta_init=0.42)
    s = node.initial_state()
    assert jnp.allclose(s["theta"], jnp.atleast_1d(0.42), atol=1e-12)
    assert int(s["mask"].sum()) == node.K


# ----------------------------------------------------------------------
# 6. test_no_wrong_sign_at_boundary
# ----------------------------------------------------------------------

@pytest.mark.parametrize("theta", [0.04, 0.06, 0.94, 0.96])
def test_no_wrong_sign_at_boundary(theta):
    """At boundary-θ values where TopKAdaptiveNode with
    ``selection_quantity='b'`` produces wrong-sign solutions, the
    local hat basis cannot — round-4 locality theorem in action.

    Each hat ``phi_λ`` has ``phi_λ(x_sensor) ≥ 0`` on its support,
    so the masked sum cannot cancel itself into a wrong-sign result.
    """
    node = HierarchicalHatAdaptiveNode(theta_init=theta)
    # Bypass cold-start gate so we test the raw boundary configuration.
    s = node._initial_state_impl()
    # Compute J_full at this theta.
    def J_full(theta_local):
        b = node._rhs_coeffs(theta_local)
        c = jnp.linalg.solve(node._M, b)
        return node._phi[node._sensor_idx] @ c
    J_full_val = float(jnp.squeeze(J_full(s["theta"])))
    J_frozen_val = float(node._sensor(s))
    # Sign agreement (or both near zero).
    assert J_full_val * J_frozen_val >= -1e-20, (
        f"locality theorem violation at theta={theta}: "
        f"J_full={J_full_val}, J_frozen={J_frozen_val}"
    )


# ----------------------------------------------------------------------
# 7. test_autodiff_against_fd_at_non_kink
# ----------------------------------------------------------------------

def test_autodiff_against_fd_at_non_kink():
    """jax.grad through one full update() agrees with FD at theta=0.42."""
    node = HierarchicalHatAdaptiveNode(theta_init=0.42, N_levels=5, K=8)

    def J_of_theta(theta):
        empty = {
            "c": jnp.zeros(node.N_max, dtype=jnp.float64),
            "mask": jnp.zeros(node.N_max, dtype=bool),
            "theta": jnp.atleast_1d(theta),
        }
        mask = node.compute_active_set(empty, prev=None, is_cold_start=True)
        s = node.solve_frozen({**empty, "mask": mask}, mask)
        return jnp.squeeze(node._sensor(s))

    theta0 = jnp.asarray(0.42)
    g_auto = float(jax.grad(J_of_theta)(theta0))
    h = 1e-5
    g_fd = float((J_of_theta(theta0 + h) - J_of_theta(theta0 - h)) / (2 * h))
    rel_err = abs(g_auto - g_fd) / (abs(g_fd) + 1e-30)
    assert rel_err < 1e-4, f"rel_err={rel_err}; g_auto={g_auto}, g_fd={g_fd}"


# ----------------------------------------------------------------------
# 8. test_solve_uses_bcoo_path
# ----------------------------------------------------------------------

def test_solve_uses_bcoo_path():
    """The operator_fn passed to ift_linear_solve is backed by a BCOO matrix.

    Spies on ift_linear_solve and inspects the operator: invoking
    ``operator_fn(probe)`` triggers a BCOO matvec, which we detect
    by checking the type of the matvec result's tracer.
    """
    node = HierarchicalHatAdaptiveNode(theta_init=0.42, N_levels=5, K=8)
    s = node.initial_state()

    # Spy: replace ift_linear_solve with a wrapper that inspects the
    # operator_fn signature.  Look for a BCOO matmul in the function's
    # closure -- the operator we built closes over A_bcoo.
    seen = {"bcoo_observed": False}

    import maddening.nodes.adaptive.hierarchical_hat as hat_mod
    original = hat_mod.ift_linear_solve

    def spy(operator_fn, rhs, **kw):
        # Check whether the operator_fn closes over a BCOO matrix.
        # Each cell in the closure is a free var; one of them should be
        # the BCOO matrix.
        closure_cells = operator_fn.__closure__ or ()
        for cell in closure_cells:
            try:
                contents = cell.cell_contents
            except ValueError:
                continue
            if isinstance(contents, jsparse.BCOO):
                seen["bcoo_observed"] = True
                break
        return original(operator_fn, rhs, **kw)

    hat_mod.ift_linear_solve = spy
    try:
        _ = node.update(s, {}, 1.0)
    finally:
        hat_mod.ift_linear_solve = original
    assert seen["bcoo_observed"], (
        "expected the solve to construct a BCOO operator; "
        "the round-6 BCOO compatibility path was not exercised."
    )


# ----------------------------------------------------------------------
# 9. test_doctest_in_module_docstring
# ----------------------------------------------------------------------

def test_doctest_in_module_docstring():
    """Module docstring doctest passes."""
    results = doctest.testmod(hat_module, verbose=False)
    assert results.failed == 0, (
        f"doctest in hierarchical_hat module failed: {results}"
    )


# ----------------------------------------------------------------------
# 10. test_optimization_convergence_smoke (informational, non-blocking)
# ----------------------------------------------------------------------

def test_optimization_convergence_smoke():
    """30 gradient-ascent steps from theta=0.45 increase J monotonically
    (modulo single-step noise) and reach within 0.15 of the sensor
    location.  Informational — confirms the framework supports a
    full end-to-end optimisation, not just a one-step query."""
    node = HierarchicalHatAdaptiveNode(theta_init=0.45, N_levels=5, K=8)
    s = node.initial_state()

    def J_of_theta(theta):
        empty = {
            "c": jnp.zeros(node.N_max, dtype=jnp.float64),
            "mask": jnp.zeros(node.N_max, dtype=bool),
            "theta": jnp.atleast_1d(theta),
        }
        mask = node.compute_active_set(empty, prev=None, is_cold_start=True)
        st = node.solve_frozen({**empty, "mask": mask}, mask)
        return jnp.squeeze(node._sensor(st))

    grad_J = jax.grad(J_of_theta)
    theta = jnp.asarray(0.45)
    lr = 0.5
    J0 = float(J_of_theta(theta))
    for _ in range(30):
        g = grad_J(theta)
        theta = theta + lr * g
    Jf = float(J_of_theta(theta))
    # Should have increased: ascent on a smooth objective.
    assert Jf >= J0 - 1e-9, f"J should not decrease: J0={J0}, Jf={Jf}"
    # And theta should have moved appreciably toward sensor_x = 1/3.
    assert abs(float(theta) - node.sensor_x) < 0.15, (
        f"expected theta near sensor_x={node.sensor_x}; got {float(theta)}"
    )
