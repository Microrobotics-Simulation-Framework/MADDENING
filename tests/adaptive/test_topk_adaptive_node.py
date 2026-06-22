"""M5 tests for ``TopKAdaptiveNode`` (Toy 1: non-local sine basis).

Test list (numbering matches
``plans/MADDENING_ADAPTIVE_NODE_IMPLEMENTATION_PLAN.md`` §4 M5):

1. test_construct_with_defaults
2. test_initial_state_at_default_theta_succeeds
3. test_initial_state_at_trap_perturbs
4. test_update_produces_correct_shape
5. test_topk_b_reproduces_wrong_sign_at_boundary
6. test_topk_c_avoids_wrong_sign_at_boundary
7. test_blindness_ratio_at_known_points
8. test_autodiff_against_fd_at_non_kink
9. test_doctest_in_module_docstring
10. test_invalid_selection_quantity_raises
"""

from __future__ import annotations

import doctest

import jax
import jax.numpy as jnp
import numpy as np
import pytest


from maddening.nodes.adaptive import TopKAdaptiveNode
from maddening.nodes.adaptive import topk as topk_module


# ----------------------------------------------------------------------
# 1. test_construct_with_defaults
# ----------------------------------------------------------------------

def test_construct_with_defaults():
    """node = TopKAdaptiveNode() constructs and reports the documented state shapes."""
    node = TopKAdaptiveNode()
    assert node.N == 256
    assert node.K == 16
    assert node.selection_quantity == "c"
    assert node.state_fields() == ["c", "mask", "theta"]


# ----------------------------------------------------------------------
# 2. test_initial_state_at_default_theta_succeeds
# ----------------------------------------------------------------------

def test_initial_state_at_default_theta_succeeds():
    """Default theta=0.42 passes the cold-start blindness gate."""
    node = TopKAdaptiveNode(theta_init=0.42)
    s = node.initial_state()
    assert jnp.allclose(s["theta"], jnp.atleast_1d(0.42), atol=1e-12)
    assert int(s["mask"].sum()) == node.K


# ----------------------------------------------------------------------
# 3. test_initial_state_at_trap_perturbs
# ----------------------------------------------------------------------

def test_initial_state_at_trap_perturbs():
    """At theta_init=0.5, the gate perturbs theta away."""
    # Use selection_quantity='b' so the trap reproduces (top-|c| at
    # theta=0.5 doesn't necessarily trap because c=b/lambda still has
    # the symmetry but the |c| ordering favours low-k which dominate
    # the sensor).  The Palais trap structure that the round-3 spike
    # documented is specific to top-|b| selection.
    node = TopKAdaptiveNode(theta_init=0.5, selection_quantity="b")
    s = node.initial_state()
    assert abs(float(s["theta"][0]) - 0.5) > 1e-3


# ----------------------------------------------------------------------
# 4. test_update_produces_correct_shape
# ----------------------------------------------------------------------

def test_update_produces_correct_shape():
    """update() returns a dict with c shape (N,), mask shape (N,) bool,
    exactly K True entries."""
    node = TopKAdaptiveNode()
    s = node.initial_state()
    new_s = node.update(s, {}, 1.0)
    assert new_s["c"].shape == (node.N,)
    assert new_s["mask"].shape == (node.N,)
    assert new_s["mask"].dtype == jnp.bool_
    assert int(new_s["mask"].sum()) == node.K


# ----------------------------------------------------------------------
# 5. test_topk_b_reproduces_wrong_sign_at_boundary
# ----------------------------------------------------------------------

def test_topk_b_reproduces_wrong_sign_at_boundary():
    """selection_quantity='b' at theta near boundary produces a
    wrong-sign sensor reading.  Spike round-4 Inv 1 finding."""
    # Construct at theta=0.04 (a spike-tested boundary failure point).
    # Note: the cold-start blindness gate may want to perturb here;
    # bypass it by reading the raw _initial_state_impl().  At
    # theta=0.04 with top-|b|, the spike measured J_frozen with wrong
    # sign relative to J_full.
    node = TopKAdaptiveNode(theta_init=0.04, selection_quantity="b")
    s = node._initial_state_impl()
    # Compute J_full at this state.
    def J_full(theta):
        b = node._rhs_coeffs(theta)
        c = b / node._lambdas
        return node._phi[node._sensor_idx] @ c
    J_full_val = float(jnp.squeeze(J_full(s["theta"])))
    J_frozen_val = float(node._sensor(s))
    # The signs should disagree at this boundary configuration.
    assert J_full_val * J_frozen_val < 0, (
        f"expected sign disagreement (round-4 wrong-sign failure); "
        f"got J_full={J_full_val}, J_frozen={J_frozen_val}"
    )


# ----------------------------------------------------------------------
# 6. test_topk_c_avoids_wrong_sign_at_boundary
# ----------------------------------------------------------------------

def test_topk_c_avoids_wrong_sign_at_boundary():
    """With selection_quantity='c' at the same boundary theta, the
    sensor reading has the correct sign.  Spike round-4 Inv 1: top-|c|
    avoids the failure because the 1/lambda weighting promotes the
    low-k modes that dominate the solution."""
    node = TopKAdaptiveNode(theta_init=0.04, selection_quantity="c")
    s = node._initial_state_impl()
    def J_full(theta):
        b = node._rhs_coeffs(theta)
        c = b / node._lambdas
        return node._phi[node._sensor_idx] @ c
    J_full_val = float(jnp.squeeze(J_full(s["theta"])))
    J_frozen_val = float(node._sensor(s))
    assert J_full_val * J_frozen_val > 0, (
        f"expected sign agreement under top-|c|; "
        f"got J_full={J_full_val}, J_frozen={J_frozen_val}"
    )


# ----------------------------------------------------------------------
# 7. test_blindness_ratio_at_known_points
# ----------------------------------------------------------------------

def test_blindness_ratio_at_known_points():
    """blindness_ratio matches the spike's round-3 prevalence-sweep numbers."""
    # Use selection_quantity='b' to match the round-3 data exactly.
    for theta, lo, hi in [
        (0.42, 0.7, 1.5),    # round-3 ratio ~0.857
        (0.5, 0.0, 0.01),    # exact Palais trap, ratio ~0
        (0.48, 0.05, 0.5),   # partial-blindness band, ratio ~0.171
    ]:
        node = TopKAdaptiveNode(theta_init=theta, selection_quantity="b")
        s = node._initial_state_impl()  # bypass cold-start gate
        r = node.blindness_ratio(s)
        assert lo <= r <= hi, (
            f"theta={theta}: expected ratio in [{lo}, {hi}]; got {r}"
        )


# ----------------------------------------------------------------------
# 8. test_autodiff_against_fd_at_non_kink
# ----------------------------------------------------------------------

def test_autodiff_against_fd_at_non_kink():
    """jax.grad through one full update() agrees with FD at theta=0.42 to 1e-5."""
    node = TopKAdaptiveNode(theta_init=0.42)

    def J_of_theta(theta):
        # Build a state at this theta (bypass the gate) and run update().
        empty = {
            "c": jnp.zeros(node.N, dtype=jnp.float64),
            "mask": jnp.zeros(node.N, dtype=bool),
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
    assert rel_err < 1e-5, f"rel_err={rel_err}; g_auto={g_auto}, g_fd={g_fd}"


# ----------------------------------------------------------------------
# 9. test_doctest_in_module_docstring
# ----------------------------------------------------------------------

def test_doctest_in_module_docstring():
    """The module-level docstring's doctest example runs without failure."""
    results = doctest.testmod(topk_module, verbose=False)
    assert results.failed == 0, (
        f"doctest in topk module failed: {results}"
    )


# ----------------------------------------------------------------------
# 10. test_invalid_selection_quantity_raises
# ----------------------------------------------------------------------

def test_invalid_selection_quantity_raises():
    """An unknown selection_quantity raises ValueError at construction."""
    with pytest.raises(ValueError, match=r"selection_quantity"):
        TopKAdaptiveNode(selection_quantity="z")  # type: ignore[arg-type]
