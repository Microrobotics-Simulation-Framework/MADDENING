"""blindness_ratio / is_trapped_at / symmetry_break / cold-start gate on
the spike's constructed cases (top-|b|, K=16, sensor at x=1/3).

Known points from ``plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md``:
theta=0.42 ratio ~0.86 (good), theta=0.48 ~0.17 (partially blind),
theta=0.5 exactly 0 (Palais trap of the reflection x -> 1 - x).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from maddening.nodes.adaptive import AdaptiveNodeBlindnessError

from tests.nodes.adaptive._toys import MaskedDenseNode, PoissonSineTopKNode


def _node_and_state(theta, **kw):
    node = PoissonSineTopKNode(theta=theta, blindness_gate=False, **kw)
    return node, node.initial_state()


# -- blindness_ratio -------------------------------------------------------------

@pytest.mark.parametrize("theta, lo, hi", [
    (0.42, 0.7, 1.5),    # spike: 0.857
    (0.48, 0.05, 0.5),   # spike: 0.171
    (0.5, 0.0, 0.01),    # spike: 0.0 (trap)
])
def test_blindness_ratio_at_known_points(theta, lo, hi):
    node, s = _node_and_state(theta)
    r = node.blindness_ratio(s)
    assert lo <= r < hi, f"theta={theta}: ratio {r} outside [{lo}, {hi})"


def test_blindness_ratio_accepts_a_parameter_pytree():
    """``params`` overrides the constructor constants: the same state
    evaluated at another theta gives another ratio."""
    node, s = _node_and_state(0.42)
    r_default = node.blindness_ratio(s)
    r_same = node.blindness_ratio(s, {"theta": jnp.asarray(0.42)})
    r_other = node.blindness_ratio(s, {"theta": jnp.asarray(0.6)})
    assert r_same == pytest.approx(r_default, rel=1e-6)
    assert r_other != pytest.approx(r_default, rel=1e-3)


def test_blindness_ratio_sentinel_when_full_gradient_vanishes():
    node, s = _node_and_state(0.42)
    node.compute_full_basis_gradient = lambda state, params=None: {
        k: jnp.zeros_like(v) for k, v in node.params_pytree().items()
    }
    assert node.blindness_ratio(s) == 1.0


def test_blindness_ratio_on_a_dense_operator_is_finite():
    node = MaskedDenseNode(blindness_gate=False)
    r = node.blindness_ratio(node.initial_state())
    assert 0.0 <= r < 10.0


# -- is_trapped_at ----------------------------------------------------------------

def test_is_trapped_at_fires_only_at_the_exact_trap():
    for theta, expected in [(0.5, True), (0.42, False), (0.48, False)]:
        node, s = _node_and_state(theta)
        assert node.is_trapped_at(s) is expected, theta


# -- symmetry_break -----------------------------------------------------------------

def test_symmetry_break_moves_delta_along_unit_full_gradient():
    node, s = _node_and_state(0.4)
    g = node.compute_full_basis_gradient(s)
    n = float(jnp.linalg.norm(g["theta"]))
    assert n > 1e-6
    new = node.symmetry_break(s, delta=0.05)
    assert new["theta"] == pytest.approx(0.4 + 0.05 * float(g["theta"]) / n, abs=1e-6)


def test_symmetry_break_defaults_to_break_delta_and_respects_trainability():
    node, s = _node_and_state(0.5, blindness_break_delta=0.02)
    new = node.symmetry_break(s)
    assert abs(float(new["theta"]) - 0.5) == pytest.approx(0.02, abs=1e-6)
    assert float(new["sigma"]) == pytest.approx(0.04)          # trainable=False
    assert float(new["sensor_x"]) == pytest.approx(1.0 / 3.0)  # trainable=False
    same = node.symmetry_break(s, delta=0.0)
    assert float(same["theta"]) == pytest.approx(0.5)


def test_symmetry_break_escapes_the_trap_in_one_step():
    node, s = _node_and_state(0.5)
    assert node.blindness_ratio(s) < 0.01
    new_params = node.symmetry_break(s)
    s2 = node._cold_start_state({**node.params, **new_params})
    assert node.blindness_ratio(s2, new_params) > node.blindness_threshold


# -- cold-start gate ------------------------------------------------------------------

def test_initial_state_raises_at_a_trap_and_passes_at_a_good_point():
    PoissonSineTopKNode(theta=0.42).initial_state()
    with pytest.raises(AdaptiveNodeBlindnessError, match="Palais fixed point"):
        PoissonSineTopKNode(theta=0.5).initial_state()
    PoissonSineTopKNode(theta=0.5, blindness_gate=False).initial_state()


def test_gate_threshold_is_configurable():
    # ratio ~0.17 at 0.48: fails at the default 0.7, passes at 0.1
    with pytest.raises(AdaptiveNodeBlindnessError):
        PoissonSineTopKNode(theta=0.48).initial_state()
    PoissonSineTopKNode(theta=0.48, blindness_threshold=0.1).initial_state()


def test_cold_start_returns_unperturbed_params_at_a_good_point():
    node = PoissonSineTopKNode(theta=0.42)
    state, params = node.cold_start()
    assert float(params["theta"]) == pytest.approx(0.42)
    assert bool(jnp.array_equal(state["mask"], node.initial_state()["mask"]))


def test_cold_start_perturbs_once_at_the_trap():
    node = PoissonSineTopKNode(theta=0.5)
    state, params = node.cold_start()
    assert abs(float(params["theta"]) - 0.5) == pytest.approx(node.blindness_break_delta, abs=1e-6)
    assert node.blindness_ratio(state, params) >= node.blindness_threshold
    assert bool(jnp.all(state["c"][~state["mask"]] == 0.0))


def test_cold_start_raises_on_a_persistent_trap():
    class Persistent(PoissonSineTopKNode):
        def compute_full_basis_gradient(self, state, params=None):
            return {k: jnp.zeros_like(v) for k, v in self.params_pytree().items()}

        def blindness_ratio(self, state, params=None):
            return 0.0  # blind everywhere

    with pytest.raises(AdaptiveNodeBlindnessError, match="after one symmetry_break"):
        Persistent(theta=0.5, blindness_gate=False).cold_start()
