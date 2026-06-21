"""M3 tests for the blindness diagnostic + symmetry_break protocol.

Test list (numbering matches
``plans/MADDENING_ADAPTIVE_NODE_IMPLEMENTATION_PLAN.md`` §4 M3):

1. test_blindness_at_known_good_point
2. test_blindness_at_known_trap
3. test_blindness_at_partial_blind
4. test_is_trapped_at_detects_exact_trap
5. test_is_trapped_at_misses_partial
6. test_is_trapped_at_negative_at_good
7. test_symmetry_break_direction
8. test_symmetry_break_recovers_from_trap
9. test_symmetry_break_zero_delta_is_identity
10. test_blindness_ratio_handles_zero_full_gradient
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tests.adaptive._mock_poisson_sine import make_node, state_at


# ----------------------------------------------------------------------
# 1. test_blindness_at_known_good_point
# ----------------------------------------------------------------------

def test_blindness_at_known_good_point():
    """At theta=0.42, blindness ratio is in [0.7, 1.5].

    Round-3 trap_characterisation sweep: blindness_ratio at theta=0.42
    measured as 0.857 with top-|b| selection at K=16.
    """
    node = make_node(theta=0.42)
    state = state_at(theta=0.42)
    r = node.blindness_ratio(state)
    assert 0.7 < r < 1.5, f"expected ~0.857, got {r}"


# ----------------------------------------------------------------------
# 2. test_blindness_at_known_trap
# ----------------------------------------------------------------------

def test_blindness_at_known_trap():
    """At theta=0.5, blindness ratio < 0.01 (Palais trap).

    Round-3 trap_characterisation: ratio measured as ~0.0 (exact zero
    in floating-point) because top-|b| selects only odd-k modes, all
    of which have db_k/dtheta = 0 at theta=0.5 by symmetry.
    """
    node = make_node(theta=0.5)
    state = state_at(theta=0.5)
    r = node.blindness_ratio(state)
    assert r < 0.01, f"expected ~0.0 at trap, got {r}"


# ----------------------------------------------------------------------
# 3. test_blindness_at_partial_blind
# ----------------------------------------------------------------------

def test_blindness_at_partial_blind():
    """At theta=0.48 (just off the trap), ratio in [0.05, 0.5].

    Round-3 prevalence sweep: ratio at theta=0.48 was 0.171.  Allow
    a wide bracket for floating-point drift across JAX versions.
    """
    node = make_node(theta=0.48)
    state = state_at(theta=0.48)
    r = node.blindness_ratio(state)
    assert 0.05 < r < 0.5, f"expected ~0.17, got {r}"


# ----------------------------------------------------------------------
# 4. test_is_trapped_at_detects_exact_trap
# ----------------------------------------------------------------------

def test_is_trapped_at_detects_exact_trap():
    """is_trapped_at returns True at the exact trap (theta=0.5).

    Round-7 Inv 1: proxy = 0.0 across all (r, eps) at the trap.
    """
    node = make_node(theta=0.5)
    state = state_at(theta=0.5)
    assert node.is_trapped_at(state) is True


# ----------------------------------------------------------------------
# 5. test_is_trapped_at_misses_partial
# ----------------------------------------------------------------------

def test_is_trapped_at_misses_partial():
    """At theta=0.48 (partial blindness, ratio ~0.17), the binary check
    returns False.  Round-6 Inv 1: the proxy distinguishes exact-trap
    from non-trap but is unreliable for continuous partial-blindness."""
    node = make_node(theta=0.48)
    state = state_at(theta=0.48)
    assert node.is_trapped_at(state) is False


# ----------------------------------------------------------------------
# 6. test_is_trapped_at_negative_at_good
# ----------------------------------------------------------------------

def test_is_trapped_at_negative_at_good():
    """At theta=0.42 (good point), the check returns False."""
    node = make_node(theta=0.42)
    state = state_at(theta=0.42)
    assert node.is_trapped_at(state) is False


# ----------------------------------------------------------------------
# 7. test_symmetry_break_direction
# ----------------------------------------------------------------------

def test_symmetry_break_direction():
    """symmetry_break perturbs theta by exactly delta along the unit
    g_full direction."""
    node = make_node(theta=0.4)
    state = state_at(theta=0.4)
    g_full = node.compute_full_basis_gradient(state)
    n_full = float(jnp.linalg.norm(g_full))
    assert n_full > 1e-6, "this test requires a non-zero g_full"
    unit = g_full / n_full

    delta = 0.05
    new_state = node.symmetry_break(state, delta)
    theta_old = state["theta"]
    theta_new = new_state["theta"]
    diff = theta_new - theta_old
    # diff should equal delta * unit
    assert jnp.allclose(diff, delta * unit, atol=1e-12)


# ----------------------------------------------------------------------
# 8. test_symmetry_break_recovers_from_trap
# ----------------------------------------------------------------------

def test_symmetry_break_recovers_from_trap():
    """One symmetry_break perturbation at delta=0.05 from theta=0.5
    moves the state into a region with blindness_ratio > 0.7.

    Round-7 Inv 2 result: in 1D the minimum delta is 0.03; the
    default 0.05 has margin.
    """
    node = make_node(theta=0.5)
    state = state_at(theta=0.5)
    new_state = node.symmetry_break(state, 0.05)
    # Update the mask for the new theta before measuring blindness.
    new_mask = node.compute_active_set(new_state, prev=state, is_cold_start=False)
    new_state = node.solve_frozen({**new_state, "mask": new_mask}, new_mask)
    ratio = node.blindness_ratio(new_state)
    assert ratio > 0.7, f"expected >0.7 after symmetry_break, got {ratio}"


# ----------------------------------------------------------------------
# 9. test_symmetry_break_zero_delta_is_identity
# ----------------------------------------------------------------------

def test_symmetry_break_zero_delta_is_identity():
    """symmetry_break(state, 0.0) returns a state with the same theta."""
    node = make_node(theta=0.42)
    state = state_at(theta=0.42)
    new_state = node.symmetry_break(state, 0.0)
    assert jnp.allclose(new_state["theta"], state["theta"], atol=1e-12)


# ----------------------------------------------------------------------
# 10. test_blindness_ratio_handles_zero_full_gradient
# ----------------------------------------------------------------------

def test_blindness_ratio_handles_zero_full_gradient():
    """At an interior extremum of J (both g_frozen and g_full near zero),
    the implementation returns 1.0 as a sentinel rather than dividing
    by zero.

    Build a synthetic state where the full-basis gradient is forced to
    be zero by monkeypatching compute_full_basis_gradient.
    """
    node = make_node(theta=0.42)
    state = state_at(theta=0.42)
    node.compute_full_basis_gradient = lambda s: jnp.zeros_like(s["theta"])
    r = node.blindness_ratio(state)
    assert r == 1.0
