"""M4 tests for the ``initial_state`` cold-start blindness gate.

Test list (numbering matches
``plans/MADDENING_ADAPTIVE_NODE_IMPLEMENTATION_PLAN.md`` §4 M4):

1. test_initial_state_at_good_theta_passes
2. test_initial_state_at_trap_perturbs_and_passes
3. test_initial_state_at_persistent_trap_raises
4. test_blindness_threshold_configurable
5. test_blindness_break_delta_configurable
6. test_initial_state_does_not_call_solve_frozen_when_good
7. test_initial_state_invokes_blindness_check_exactly_twice_at_perturbation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


from maddening.nodes.adaptive import AdaptiveNode, AdaptiveNodeBlindnessError
from tests.adaptive._mock_poisson_sine import (
    MockPoissonSineNode, make_node, K_ACTIVE, N_BASIS,
)


# ----------------------------------------------------------------------
# 1. test_initial_state_at_good_theta_passes
# ----------------------------------------------------------------------

def test_initial_state_at_good_theta_passes():
    """At theta=0.42, the cold-start gate accepts the state unchanged."""
    node = make_node(theta=0.42)
    s = node.initial_state()
    # The returned theta should equal the constructor's theta_init (no
    # perturbation needed at a non-trap point).
    assert jnp.allclose(s["theta"], jnp.atleast_1d(0.42), atol=1e-12)


# ----------------------------------------------------------------------
# 2. test_initial_state_at_trap_perturbs_and_passes
# ----------------------------------------------------------------------

def test_initial_state_at_trap_perturbs_and_passes():
    """At theta=0.5 (Palais trap), initial_state perturbs to theta!=0.5
    and the returned state passes the blindness threshold."""
    node = make_node(theta=0.5)
    s = node.initial_state()
    # Perturbation magnitude is blindness_break_delta = 0.05.
    theta_new = float(s["theta"][0])
    assert abs(theta_new - 0.5) > 1e-3, (
        f"expected perturbation away from 0.5, got theta={theta_new}"
    )
    # And the returned state should pass the threshold check.
    assert node.blindness_ratio(s) >= node.blindness_threshold


# ----------------------------------------------------------------------
# 3. test_initial_state_at_persistent_trap_raises
# ----------------------------------------------------------------------

def test_initial_state_at_persistent_trap_raises():
    """When compute_full_basis_gradient is forced to always return zero
    (synthetic persistent trap), initial_state raises AdaptiveNodeBlindnessError."""
    class PersistentTrap(MockPoissonSineNode):
        def compute_full_basis_gradient(self, state):
            return jnp.zeros_like(state["theta"])

        # Force blindness_ratio to return 0 (not the 1.0 sentinel) so
        # the gate triggers.  We accomplish this by overriding
        # blindness_ratio rather than fighting the sentinel.
        def blindness_ratio(self, state):
            return 0.0

    node = PersistentTrap(theta_init=0.5)
    with pytest.raises(AdaptiveNodeBlindnessError,
                        match=r"Palais fixed point"):
        node.initial_state()


# ----------------------------------------------------------------------
# 4. test_blindness_threshold_configurable
# ----------------------------------------------------------------------

def test_blindness_threshold_configurable():
    """A subclass with blindness_threshold = 0.5 accepts states the
    default 0.7 would reject."""
    class LowerThreshold(MockPoissonSineNode):
        blindness_threshold = 0.05  # accept anything above near-zero

    # At theta=0.48 (partial blindness, round-3 ratio ~0.17) the
    # default-threshold subclass would perturb, but the lower-threshold
    # subclass accepts as-is.
    node = LowerThreshold(theta_init=0.48)
    s = node.initial_state()
    # The theta was NOT perturbed (lower threshold accepted partial).
    assert jnp.allclose(s["theta"], jnp.atleast_1d(0.48), atol=1e-12)


# ----------------------------------------------------------------------
# 5. test_blindness_break_delta_configurable
# ----------------------------------------------------------------------

def test_blindness_break_delta_configurable():
    """A subclass that sets blindness_break_delta = 0.08 perturbs by
    0.08, not the default 0.05.

    Round-3 data: theta=0.5-0.08=0.42 lands in good-gradient
    territory (ratio ~0.857), so the gate accepts after one
    perturbation and we can verify the magnitude on the returned state.
    """
    class CustomDelta(MockPoissonSineNode):
        blindness_break_delta = 0.08

    node = CustomDelta(theta_init=0.5)
    s = node.initial_state()
    theta_new = float(s["theta"][0])
    delta_observed = abs(theta_new - 0.5)
    assert 0.075 < delta_observed < 0.085, (
        f"expected perturbation magnitude ~0.08, got {delta_observed}"
    )


# ----------------------------------------------------------------------
# 6. test_initial_state_no_redundant_solve_at_good_theta
# ----------------------------------------------------------------------

def test_initial_state_no_redundant_solve_at_good_theta():
    """At a non-trap state, the gate accepts immediately.  The only
    ``solve_frozen`` calls after ``_initial_state_impl`` are the ones
    that ``blindness_ratio`` itself triggers via ``jax.grad`` through
    the frozen objective (1 call for the gradient closure).  There is
    NO additional re-solve.

    Guards against a regression that would re-solve the system after
    blindness check passes, doubling cold-start cost.
    """
    class CountingSolves(MockPoissonSineNode):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.solve_calls_after_impl = 0
            self._past_impl = False

        def _initial_state_impl(self):
            s = super()._initial_state_impl()
            self._past_impl = True
            return s

        def solve_frozen(self, state, mask):
            if self._past_impl:
                self.solve_calls_after_impl += 1
            return super().solve_frozen(state, mask)

    node = CountingSolves(theta_init=0.42)
    _ = node.initial_state()
    # 1 call for blindness_ratio's gradient closure.  No additional
    # re-solve from a perturbation branch.
    assert node.solve_calls_after_impl == 1, (
        f"expected 1 extra solve (from blindness_ratio's grad closure) "
        f"at non-trap; got {node.solve_calls_after_impl}"
    )


# ----------------------------------------------------------------------
# 7. test_initial_state_invokes_blindness_check_exactly_twice_at_perturbation
# ----------------------------------------------------------------------

def test_initial_state_invokes_blindness_check_exactly_twice_at_perturbation():
    """Under the perturbation branch, blindness_ratio is called exactly
    twice: pre- and post-perturbation."""
    class CountingBlindness(MockPoissonSineNode):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.blindness_calls = 0

        def blindness_ratio(self, state):
            self.blindness_calls += 1
            return super().blindness_ratio(state)

    node = CountingBlindness(theta_init=0.5)  # trap -> triggers perturbation
    _ = node.initial_state()
    assert node.blindness_calls == 2, (
        f"expected exactly 2 blindness_ratio calls; got {node.blindness_calls}"
    )
