"""M7 cross-cutting integration tests.

These tests exercise the entire AdaptiveNode framework against both
toy subclasses together, verifying that the public surface is
consistent and the stability decorations are present.

Test list (numbering matches
``plans/MADDENING_ADAPTIVE_NODE_IMPLEMENTATION_PLAN.md`` §4 M7):

1. test_both_toys_pass_blindness_at_default_theta
2. test_both_toys_trap_at_theta_0_5
3. test_jax_grad_through_full_node_lifecycle
4. test_blindness_constants_are_public
5. test_stability_tags_present
6. test_AdaptiveNodeBlindnessError_is_public
7. test_solver_utils_importable_from_public_path
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


from maddening.core.compliance.metadata import StabilityLevel
from maddening.nodes.adaptive import (
    AdaptiveNode,
    AdaptiveNodeBlindnessError,
    HierarchicalHatAdaptiveNode,
    TopKAdaptiveNode,
)


# ----------------------------------------------------------------------
# 1. test_both_toys_pass_blindness_at_default_theta
# ----------------------------------------------------------------------

@pytest.mark.parametrize("ctor", [
    lambda: TopKAdaptiveNode(theta_init=0.42),
    lambda: HierarchicalHatAdaptiveNode(theta_init=0.42),
])
def test_both_toys_pass_blindness_at_default_theta(ctor):
    """Both toy subclasses construct and pass the cold-start blindness
    check at the default theta=0.42."""
    node = ctor()
    s = node.initial_state()
    assert jnp.allclose(s["theta"], jnp.atleast_1d(0.42), atol=1e-12)
    assert int(s["mask"].sum()) == node.K


# ----------------------------------------------------------------------
# 2. test_both_toys_trap_at_theta_0_5
# ----------------------------------------------------------------------

def test_topk_b_trap_at_theta_0_5_recovers():
    """TopKAdaptiveNode with selection_quantity='b' at theta=0.5
    is at a Palais trap; the cold-start gate perturbs and recovers."""
    node = TopKAdaptiveNode(theta_init=0.5, selection_quantity="b")
    s = node.initial_state()
    assert abs(float(s["theta"][0]) - 0.5) > 1e-3, (
        "expected the trap to trigger a perturbation"
    )


# ----------------------------------------------------------------------
# 3. test_jax_grad_through_full_node_lifecycle
# ----------------------------------------------------------------------

def test_jax_grad_through_topk_lifecycle():
    """jax.grad through TopK's full update() agrees with FD at theta=0.42."""
    node = TopKAdaptiveNode(theta_init=0.42)

    def lifecycle(theta):
        empty = {
            "c": jnp.zeros(node.N, dtype=jnp.float64),
            "mask": jnp.zeros(node.N, dtype=bool),
            "theta": jnp.atleast_1d(theta),
        }
        mask = node.compute_active_set(empty, prev=None, is_cold_start=True)
        s = node.solve_frozen({**empty, "mask": mask}, mask)
        # Run one update on top of the cold-start state -- exercises the
        # full lifecycle, not just initial_state.
        s2 = node.update(s, {}, 1.0)
        return jnp.squeeze(node._sensor(s2))

    theta0 = jnp.asarray(0.42)
    g_auto = float(jax.grad(lifecycle)(theta0))
    h = 1e-5
    g_fd = float((lifecycle(theta0 + h) - lifecycle(theta0 - h)) / (2 * h))
    rel_err = abs(g_auto - g_fd) / (abs(g_fd) + 1e-30)
    assert rel_err < 1e-5, f"rel_err={rel_err}; g_auto={g_auto}, g_fd={g_fd}"


def test_jax_grad_through_hat_lifecycle():
    """jax.grad through HierarchicalHat's full update() agrees with FD."""
    node = HierarchicalHatAdaptiveNode(theta_init=0.42, N_levels=5, K=8)

    def lifecycle(theta):
        empty = {
            "c": jnp.zeros(node.N_max, dtype=jnp.float64),
            "mask": jnp.zeros(node.N_max, dtype=bool),
            "theta": jnp.atleast_1d(theta),
        }
        mask = node.compute_active_set(empty, prev=None, is_cold_start=True)
        s = node.solve_frozen({**empty, "mask": mask}, mask)
        s2 = node.update(s, {}, 1.0)
        return jnp.squeeze(node._sensor(s2))

    theta0 = jnp.asarray(0.42)
    g_auto = float(jax.grad(lifecycle)(theta0))
    h = 1e-5
    g_fd = float((lifecycle(theta0 + h) - lifecycle(theta0 - h)) / (2 * h))
    rel_err = abs(g_auto - g_fd) / (abs(g_fd) + 1e-30)
    assert rel_err < 1e-4, f"rel_err={rel_err}; g_auto={g_auto}, g_fd={g_fd}"


# ----------------------------------------------------------------------
# 4. test_blindness_constants_are_public
# ----------------------------------------------------------------------

def test_blindness_constants_are_public():
    """The round-7 finalised values are the documented defaults."""
    assert AdaptiveNode.blindness_threshold == 0.7
    assert AdaptiveNode.blindness_break_delta == 0.05
    assert AdaptiveNode.D_threshold == 5


# ----------------------------------------------------------------------
# 5. test_stability_tags_present
# ----------------------------------------------------------------------

@pytest.mark.parametrize("symbol", [
    AdaptiveNode,
    AdaptiveNodeBlindnessError,
    TopKAdaptiveNode,
    HierarchicalHatAdaptiveNode,
])
def test_stability_tag_present_on_public_symbol(symbol):
    """Every public symbol carries @stability(STABLE)."""
    level = getattr(symbol, "_stability_level", None)
    assert level == StabilityLevel.STABLE, (
        f"{symbol!r} missing @stability(STABLE); level={level!r}"
    )


def test_stability_tag_present_on_ift_linear_solve():
    """The ift_linear_solve primitive is STABLE-tagged."""
    from maddening.core.solver_utils import ift_linear_solve
    level = getattr(ift_linear_solve, "_stability_level", None)
    assert level == StabilityLevel.STABLE


# ----------------------------------------------------------------------
# 6. test_AdaptiveNodeBlindnessError_is_public
# ----------------------------------------------------------------------

def test_AdaptiveNodeBlindnessError_is_public():
    """Importable from the public path and inherits RuntimeError."""
    from maddening.nodes.adaptive import AdaptiveNodeBlindnessError as Err
    assert issubclass(Err, RuntimeError)
    # And the class is in __all__
    import maddening.nodes.adaptive as pkg
    assert "AdaptiveNodeBlindnessError" in pkg.__all__


# ----------------------------------------------------------------------
# 7. test_solver_utils_importable_from_public_path
# ----------------------------------------------------------------------

def test_solver_utils_importable_from_public_path():
    """ift_linear_solve is reachable through the documented public path."""
    from maddening.core.solver_utils import ift_linear_solve
    assert callable(ift_linear_solve)
