"""gradient_capture_ratio / frozen_gradient_vanishes_at / symmetry_break / cold-start
diagnostic on the spike's constructed cases (top-|b|, K=16, sensor at x=1/3).

Known points from ``plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md``:
theta=0.42 ratio ~0.86 (good), theta=0.48 ~0.17 (partially blind),
theta=0.5 exactly 0 (Palais trap of the reflection x -> 1 - x).

The ratio is a budget-adequacy measurement, not a symmetry test, and
``frozen_gradient_vanishes_at`` is not one either: a Palais trap implies
a vanishing frozen gradient, so a ``False`` rules a trap out, but a
``True`` is equally consistent with an ordinary stationary point.  The
tests below pin what each of them does and does not establish.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from maddening.nodes.adaptive import AdaptiveNodeBlindnessError

from tests.nodes.adaptive._toys import MaskedDenseNode, PoissonSineTopKNode


def _node_and_state(theta, **kw):
    node = PoissonSineTopKNode(theta=theta, blindness_gate=False, **kw)
    return node, node.initial_state()


def _ratio(n, k, theta=0.42):
    node, state = _node_and_state(theta, n=n, k=k)
    return node, state, node.gradient_capture_ratio(state)


# -- gradient_capture_ratio -------------------------------------------------------------

@pytest.mark.parametrize("theta, lo, hi", [
    (0.42, 0.7, 1.5),    # spike: 0.857
    (0.48, 0.05, 0.5),   # spike: 0.171
    (0.5, 0.0, 0.01),    # spike: 0.0 (trap)
])
def test_gradient_capture_ratio_at_known_points(theta, lo, hi):
    node, s = _node_and_state(theta)
    r = node.gradient_capture_ratio(s)
    assert lo <= r < hi, f"theta={theta}: ratio {r} outside [{lo}, {hi})"


def test_gradient_capture_ratio_accepts_a_parameter_pytree():
    """``params`` overrides the constructor constants: the same state
    evaluated at another theta gives another ratio."""
    node, s = _node_and_state(0.42)
    r_default = node.gradient_capture_ratio(s)
    r_same = node.gradient_capture_ratio(s, {"theta": jnp.asarray(0.42)})
    r_other = node.gradient_capture_ratio(s, {"theta": jnp.asarray(0.6)})
    assert r_same == pytest.approx(r_default, rel=1e-6)
    assert r_other != pytest.approx(r_default, rel=1e-3)


def test_gradient_capture_ratio_sentinel_when_full_gradient_vanishes():
    node, s = _node_and_state(0.42)
    node.compute_full_basis_gradient = lambda state, params=None: {
        k: jnp.zeros_like(v) for k, v in node.params_pytree().items()
    }
    assert node.gradient_capture_ratio(s) == 1.0


def test_gradient_capture_ratio_on_a_dense_operator_is_finite():
    node = MaskedDenseNode(blindness_gate=False)
    r = node.gradient_capture_ratio(node.initial_state())
    assert 0.0 <= r < 10.0


# -- frozen_gradient_vanishes_at --------------------------------------------------

def test_the_frozen_gradient_vanishes_at_the_trap_and_not_at_healthy_points():
    for theta, expected in [(0.5, True), (0.42, False), (0.48, False)]:
        node, s = _node_and_state(theta)
        assert node.frozen_gradient_vanishes_at(s) is expected, theta


def test_a_false_rules_a_trap_out_but_a_true_does_not_establish_one():
    """The claim the check used to make -- "the one diagnostic that can
    *establish* a symmetry trap" -- fails in the direction it was used in.

    At the frozen objective's interior stationary point it returns ``True``
    on a toy whose only reflection fixed point is ``theta = 0.5``.  That
    point is where a *successful* optimisation ends, and the algorithm guide
    tells users to run this between optimiser steps, so the false positive
    sits on the happy path.  Pinned so nobody restores the stronger wording.
    """
    # Located once with brentq on dJ_frozen/dtheta over [0.30, 0.37] (0.4.0
    # adaptive audit).  Pinned rather than re-solved: the test then needs no
    # root finder and no scipy.
    theta_star = 0.343973370458
    node = PoissonSineTopKNode(n=256, k=64, theta=0.30, sigma=0.04,
                               sensor_x=1.0 / 3.0, blindness_gate=False)
    state = node.initial_state()

    def frozen_objective(x):
        out = node.update(state, {}, 1.0, params={"theta": x})
        return node.objective(out, {**node.params, "theta": x})

    g = float(jax.grad(frozen_objective)(jnp.asarray(theta_star)))
    assert abs(g) < 1e-10, f"theta* is not stationary: dJ/dtheta = {g:.3e}"
    assert node.frozen_gradient_vanishes_at(
        state, {"theta": jnp.asarray(theta_star)}) is True


def test_is_trapped_at_is_a_deprecated_alias_that_warns_and_delegates():
    node, s = _node_and_state(0.5)
    with pytest.warns(DeprecationWarning, match="frozen_gradient_vanishes_at"):
        old = node.is_trapped_at(s)
    assert old is node.frozen_gradient_vanishes_at(s)


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
    assert node.gradient_capture_ratio(s) < 0.01
    new_params = node.symmetry_break(s)
    s2 = node._cold_start_state({**node.params, **new_params})
    assert node.gradient_capture_ratio(s2, new_params) > node.gradient_capture_threshold


# -- cold-start gate ------------------------------------------------------------------

def test_initial_state_raises_only_at_an_established_trap():
    """A confirmed Palais fixed point is the one cause the diagnostics can
    establish, so it is the one that still fails construction."""
    PoissonSineTopKNode(theta=0.42).initial_state()
    with pytest.raises(AdaptiveNodeBlindnessError, match="Palais fixed point"):
        PoissonSineTopKNode(theta=0.5).initial_state()
    PoissonSineTopKNode(theta=0.5, blindness_gate=False).initial_state()
    PoissonSineTopKNode(theta=0.5, on_blind="ignore").initial_state()


def test_a_low_ratio_that_is_not_a_trap_only_warns_and_names_the_remedies():
    """theta=0.48 measures ~0.17 but the frozen gradient does not vanish
    there: a partially-blind point is a legitimate construction, not a
    failure."""
    node = PoissonSineTopKNode(theta=0.48)
    with pytest.warns(UserWarning) as record:
        state = node.initial_state()
    message = str(record[0].message)
    assert state["c"].shape == (256,)
    assert "0.1" in message                      # the measured ratio
    assert "0.700" in message                    # the threshold
    assert "rules a symmetry trap out" in message
    assert "raise the active-set budget" in message
    assert "cold_start() / symmetry_break() do" in message  # and why they do not help
    assert "gradient_capture_threshold" in message and "on_blind" in message


def test_gate_threshold_is_configurable():
    # ratio ~0.17 at 0.48: warns at the default 0.7, silent at 0.1
    with pytest.warns(UserWarning, match="gradient-capture ratio"):
        PoissonSineTopKNode(theta=0.48).initial_state()
    PoissonSineTopKNode(theta=0.48, gradient_capture_threshold=0.1).initial_state()


def test_opt_in_raising_turns_a_low_ratio_into_an_error():
    with pytest.raises(AdaptiveNodeBlindnessError, match="rules a symmetry trap out"):
        PoissonSineTopKNode(theta=0.48, on_blind="raise").initial_state()


def test_cold_start_returns_unperturbed_params_at_a_good_point():
    node = PoissonSineTopKNode(theta=0.42)
    state, params = node.cold_start()
    assert float(params["theta"]) == pytest.approx(0.42)
    assert bool(jnp.array_equal(state["mask"], node.initial_state()["mask"]))


def test_cold_start_perturbs_once_at_the_trap():
    node = PoissonSineTopKNode(theta=0.5)
    state, params = node.cold_start()
    assert abs(float(params["theta"]) - 0.5) == pytest.approx(node.blindness_break_delta, abs=1e-6)
    assert node.gradient_capture_ratio(state, params) >= node.gradient_capture_threshold
    assert bool(jnp.all(state["c"][~state["mask"]] == 0.0))


def test_cold_start_raises_on_a_persistent_trap():
    class Persistent(PoissonSineTopKNode):
        def compute_full_basis_gradient(self, state, params=None):
            return {k: jnp.zeros_like(v) for k, v in self.params_pytree().items()}

        def gradient_capture_ratio(self, state, params=None):
            return 0.0  # blind everywhere

    with pytest.raises(AdaptiveNodeBlindnessError, match="after one symmetry_break"):
        Persistent(theta=0.5, blindness_gate=False).cold_start()


# -- what the ratio actually measures (audit A3) ----------------------------------

@pytest.mark.parametrize("n", [64, 256])
def test_gradient_capture_ratio_tracks_the_active_set_budget_not_symmetry(n):
    """At a fixed, entirely non-symmetric theta the ratio is a function of
    the budget k and is essentially independent of the basis size n, while
    ``frozen_gradient_vanishes_at`` correctly reports no trap.  Pinning
    this stops the two questions being conflated again."""
    ratios = {}
    for k in (4, 8, 16, 32):
        node, state, r = _ratio(n, k)
        ratios[k] = r
        assert not node.frozen_gradient_vanishes_at(state), (n, k)
    assert ratios[4] < ratios[8] < ratios[16] < ratios[32]
    # The spike/audit numbers, reproduced at every n.
    assert ratios[4] == pytest.approx(0.163, abs=0.01)
    assert ratios[8] == pytest.approx(0.565, abs=0.01)
    assert ratios[16] == pytest.approx(0.855, abs=0.01)
    assert ratios[32] == pytest.approx(1.0, abs=0.01)


def test_the_documented_remedy_raising_the_budget_lowers_the_shortfall():
    """The warning tells the user to raise the budget; that must work."""
    _, _, low = _ratio(256, 8)
    _, _, high = _ratio(256, 32)
    assert low < PoissonSineTopKNode.gradient_capture_threshold <= high
    with pytest.warns(UserWarning, match="gradient-capture ratio"):
        PoissonSineTopKNode(n=256, k=8).initial_state()
    PoissonSineTopKNode(n=256, k=32).initial_state()  # no warning: remedy works


def test_symmetry_break_is_not_a_remedy_for_a_budget_limited_ratio():
    """The old error message prescribed cold_start()/symmetry_break() for
    every low ratio.  At a budget-limited point they make it worse, which
    is why the message no longer names them there."""
    node, state, before = _ratio(64, 8)
    moved = node.symmetry_break(state)
    after = node.gradient_capture_ratio(node._cold_start_state({**node.params, **moved}), moved)
    assert before < node.gradient_capture_threshold
    assert after < before


def test_documented_validated_active_fraction_range_is_constructible():
    """The guide lists K/n_max down to 0.016 as validated; with the default
    policy that configuration must be reachable (it warns, it does not
    reject)."""
    node = PoissonSineTopKNode(n=256, k=4)
    with pytest.warns(UserWarning, match="gradient-capture ratio"):
        state = node.initial_state()
    assert int(state["mask"].sum()) == 4


# -- the ratio follows the parameters, not a stale mask (audit A5) ----------------

def test_gradient_capture_ratio_is_a_function_of_the_parameters_not_a_stale_mask():
    """A mask selected at a healthy theta used to make the exact trap
    report a healthy ratio (measured 1.005).  The ratio re-selects."""
    node = PoissonSineTopKNode(theta=0.42, n=64, k=16, blindness_gate=False)
    healthy_state = node.initial_state()          # mask chosen at theta=0.42
    trapped = {"theta": jnp.asarray(0.5), "sigma": jnp.asarray(0.04),
               "sensor_x": jnp.asarray(1.0 / 3.0)}
    stale_mask_ratio = node.gradient_capture_ratio(healthy_state, trapped)
    assert stale_mask_ratio < 0.01, stale_mask_ratio
    fresh = node._cold_start_state({**node.params, **trapped})
    assert node.gradient_capture_ratio(fresh, trapped) == pytest.approx(
        stale_mask_ratio, abs=1e-9,
    )


def test_check_gradient_capture_evaluates_the_parameters_it_is_handed():
    """The cold-start call sees the constructor parameters; the public
    check must accept the ones actually in use."""
    node = PoissonSineTopKNode(theta=0.42, n=64, k=16)
    assert node.check_gradient_capture() > node.gradient_capture_threshold
    with pytest.raises(AdaptiveNodeBlindnessError, match="Palais fixed point"):
        node.check_gradient_capture({"theta": jnp.asarray(0.5)})
    assert node.check_gradient_capture(
        {"theta": jnp.asarray(0.5)}, on_blind="ignore",
    ) is None
