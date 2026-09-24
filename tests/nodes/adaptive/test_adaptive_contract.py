"""AdaptiveNode public contract: hooks, state schema, constants, mask semantics."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jax
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.solver_utils import ift_linear_solve
from maddening.nodes.adaptive import AdaptiveNode, AdaptiveNodeBlindnessError

from tests.nodes.adaptive._toys import PoissonSineTopKNode


class _Bare(AdaptiveNode):
    """Implements only the two required hooks: everything else must fail loudly."""

    def compute_active_set(self, state, params, *, prev=None, is_cold_start=False):
        return jnp.ones(self.n_max, dtype=bool)

    def solve_frozen(self, state, mask, params):
        return {"c": jnp.zeros(self.n_max)}


def _sine(**kw):
    return PoissonSineTopKNode(**kw)


# -- abstract surface --------------------------------------------------------

def test_subclass_missing_a_required_hook_fails_at_instantiation():
    """``compute_active_set`` and ``solve_frozen`` are abstract: a subclass
    that forgets one must fail at construction, not at trace time."""

    class MissingBoth(AdaptiveNode):
        pass

    class MissingSolve(AdaptiveNode):
        def compute_active_set(self, state, params, *, prev=None, is_cold_start=False):
            return jnp.ones(self.n_max, dtype=bool)

    for cls in (MissingBoth, MissingSolve):
        with pytest.raises(TypeError, match="abstract"):
            cls("bare", 1.0, n_max=4, blindness_gate=False)


def test_objective_stays_optional_and_raises_only_when_used():
    """``objective`` is deliberately not abstract: a node that runs with
    the diagnostics off never needs it."""
    node = _Bare("bare", 1.0, n_max=4, blindness_gate=False)
    state = {"c": jnp.zeros(4), "mask": jnp.zeros(4, dtype=bool)}
    node.initial_state()
    with pytest.raises(NotImplementedError, match="objective"):
        node.objective(state, {})


def test_n_max_is_required_and_positive():
    with pytest.raises(TypeError):
        _Bare("bare", 1.0)
    with pytest.raises(ValueError, match="n_max"):
        _Bare("bare", 1.0, n_max=0)


def test_public_symbols_are_evolving_until_the_api_freeze():
    """Nothing has shipped: the adaptive surfaces advertise EVOLVING (and
    ``ift_linear_solve`` its pre-merge EXPERIMENTAL) until the 0.4.0 API
    freeze decides, informed by the open questions in the developer guide."""
    import maddening.nodes.adaptive as pkg
    for sym in (AdaptiveNode, AdaptiveNodeBlindnessError):
        assert sym._stability_level == StabilityLevel.EVOLVING, sym
    assert ift_linear_solve._stability_level == StabilityLevel.EXPERIMENTAL
    assert AdaptiveNode.meta.stability == StabilityLevel.EVOLVING
    assert issubclass(AdaptiveNodeBlindnessError, RuntimeError)
    # The wavelet subclass is the package's one concrete node and is
    # EXPERIMENTAL, one level below the base class it is built on.
    from maddening.nodes.adaptive import WaveletAdaptiveNode
    assert WaveletAdaptiveNode._stability_level == StabilityLevel.EXPERIMENTAL
    assert set(pkg.__all__) == {
        "AdaptiveNode", "AdaptiveNodeBlindnessError", "WaveletAdaptiveNode",
        "adaptive_diagnostics_enabled", "set_adaptive_diagnostics",
    }


def test_node_meta_is_filled_in():
    m = AdaptiveNode.meta
    assert m.algorithm_id == "MADD-NODE-009"
    assert m.stability == StabilityLevel.EVOLVING
    assert m.description and m.assumptions and m.limitations and m.hazard_hints
    assert m.implementation_map


# -- constants -----------------------------------------------------------------

def test_spike_constants_are_the_class_defaults():
    assert AdaptiveNode.gradient_capture_threshold == 0.7
    assert AdaptiveNode.blindness_break_delta == 0.05
    assert AdaptiveNode.D_threshold == 5


def test_constants_overridable_per_instance_and_per_subclass():
    node = _sine(gradient_capture_threshold=0.5, blindness_break_delta=0.1, D_threshold=3)
    assert (node.gradient_capture_threshold, node.blindness_break_delta,
            node.D_threshold) == (0.5, 0.1, 3)
    assert AdaptiveNode.gradient_capture_threshold == 0.7  # class default untouched

    class Lax(PoissonSineTopKNode):
        gradient_capture_threshold = 0.2

    assert Lax().gradient_capture_threshold == 0.2


def test_deprecated_blindness_threshold_alias_still_works():
    with pytest.warns(DeprecationWarning, match="gradient_capture_threshold"):
        node = _sine(blindness_threshold=0.5)
    assert node.gradient_capture_threshold == 0.5
    assert node.blindness_threshold == 0.5  # alias kept in sync

    class OldStyle(PoissonSineTopKNode):
        blindness_threshold = 0.2

    assert OldStyle().gradient_capture_threshold == 0.2


def test_deprecated_blindness_ratio_alias_warns_and_delegates():
    node = _sine()
    state = node.initial_state()
    with pytest.warns(DeprecationWarning, match="gradient_capture_ratio"):
        old = node.blindness_ratio(state)
    assert old == node.gradient_capture_ratio(state)


def test_constants_are_not_parameter_leaves():
    """Diagnostic knobs steer host-side checks; a fit must never see them."""
    node = _sine(gradient_capture_threshold=0.5)
    leaves = node.params_pytree()
    assert not {"gradient_capture_threshold", "blindness_threshold",
                "blindness_break_delta", "D_threshold", "blindness_gate",
                "on_blind", "n_max"} & set(leaves)
    assert set(leaves) == {"theta", "sigma", "sensor_x"}
    assert node.param_specs()["theta"].trainable
    assert not node.param_specs()["sigma"].trainable


# -- state schema and mask semantics --------------------------------------------

def test_state_schema_is_fixed_size_c_and_bool_mask():
    node = _sine(n=64, k=16)
    s = node.initial_state()
    assert node.state_fields() == ["c", "mask"]
    assert set(s) == {"c", "mask"}
    assert s["c"].shape == (64,) and s["mask"].shape == (64,)
    assert s["mask"].dtype == jnp.bool_
    assert jnp.issubdtype(s["c"].dtype, jnp.floating)
    assert int(s["mask"].sum()) == 16


def test_extra_initial_state_fields_are_carried():
    class WithCounter(PoissonSineTopKNode):
        def extra_initial_state(self):
            return {"n_solves": jnp.zeros((), dtype=jnp.int32)}

        def solve_frozen(self, state, mask, params):
            out = super().solve_frozen(state, mask, params)
            return {**out, "n_solves": state["n_solves"] + 1}

    node = WithCounter()
    assert node.state_fields() == ["c", "mask", "n_solves"]
    s = node.initial_state()
    s = node.update(s, {}, 1.0)
    assert int(s["n_solves"]) == 2  # cold start + one update


def test_coefficients_are_zero_off_the_mask():
    node = _sine()
    s = node.initial_state()
    assert bool(jnp.all(s["c"][~s["mask"]] == 0.0))
    assert bool(jnp.all(s["c"][s["mask"]] != 0.0))
    s2 = node.update(s, {}, 1.0)
    assert bool(jnp.all(s2["c"][~s2["mask"]] == 0.0))


def test_mask_in_state_is_the_set_the_coefficients_were_solved_on():
    """After ``update`` at a new theta the mask moved with it and ``c`` is
    the frozen solve on *that* mask, not the previous one."""
    node = _sine(theta=0.42)
    s = node.initial_state()
    p = {**node.params, "theta": 0.8}
    s2 = node.update(s, {}, 1.0, params={"theta": 0.8})
    expected_mask = node.compute_active_set(s, p)
    assert bool(jnp.array_equal(s2["mask"], expected_mask))
    assert not bool(jnp.array_equal(s2["mask"], s["mask"]))
    c_ref = jnp.where(expected_mask, node.full_solution_coefficients(p), 0.0)
    assert jnp.allclose(s2["c"], c_ref, atol=1e-10)


def test_base_class_zeroes_c_off_mask_even_if_subclass_does_not():
    class Sloppy(PoissonSineTopKNode):
        def solve_frozen(self, state, mask, params):
            return {"c": self.full_solution_coefficients(params)}  # no masking

    s = Sloppy().initial_state()
    assert bool(jnp.all(s["c"][~s["mask"]] == 0.0))
    assert int((s["c"] != 0).sum()) == 16


def test_wrong_shape_mask_or_c_is_rejected():
    class BadMask(PoissonSineTopKNode):
        def compute_active_set(self, state, params, *, prev=None, is_cold_start=False):
            return jnp.ones(3, dtype=bool)

    class BadC(PoissonSineTopKNode):
        def solve_frozen(self, state, mask, params):
            return {"c": jnp.zeros(3)}

    with pytest.raises(ValueError, match="compute_active_set returned shape"):
        BadMask().initial_state()
    with pytest.raises(ValueError, match="solve_frozen returned c of shape"):
        BadC().initial_state()


def test_update_ignores_boundary_inputs_and_dt():
    node = _sine()
    s = node.initial_state()
    a = node.update(s, {}, 1.0)
    b = node.update(s, {"anything": jnp.ones(3)}, 1e-3)
    assert jnp.array_equal(a["c"], b["c"]) and jnp.array_equal(a["mask"], b["mask"])


def test_selection_by_c_and_by_b_differ_near_the_boundary():
    """Both selections are valid rules; they pick different active sets
    when the source sits near x=0 (the spike's wrong-sign regime)."""
    s_b = _sine(theta=0.04, selection="b", blindness_gate=False).initial_state()
    s_c = _sine(theta=0.04, selection="c", blindness_gate=False).initial_state()
    assert not bool(jnp.array_equal(s_b["mask"], s_c["mask"]))
    with pytest.raises(ValueError, match="selection"):
        _sine(selection="x")


# -- constructor validation (audit A11) ------------------------------------------

@pytest.mark.parametrize("bad", [2.7, "8", True, 0, -3, None])
def test_constructor_rejects_an_n_max_that_is_not_a_positive_whole_number(bad):
    """A computed budget that comes out fractional must not silently
    truncate the basis, and a string must not be coerced."""
    with pytest.raises((ValueError, TypeError), match="n_max"):
        _Bare("bare", 1.0, n_max=bad, blindness_gate=False)


def test_constructor_accepts_an_integral_float_n_max_from_a_json_round_trip():
    assert _Bare("bare", 1.0, n_max=8.0, blindness_gate=False).n_max == 8


@pytest.mark.parametrize("kw", [
    {"gradient_capture_threshold": -1.0},
    {"blindness_break_delta": -5.0},
    {"D_threshold": -2},
    {"D_threshold": 2.5},
])
def test_constructor_rejects_negative_diagnostic_constants(kw):
    with pytest.raises(ValueError, match=next(iter(kw))):
        _sine(**kw)


def test_constructor_rejects_an_unknown_on_blind_policy():
    with pytest.raises(ValueError, match="on_blind"):
        _sine(on_blind="explode")


def test_dtype_is_honoured_by_the_coefficients_or_rejected():
    """``dtype`` types ``c``: the base class enforces it on whatever
    ``solve_frozen`` returns, and a non-floating dtype is refused."""
    node = _sine(n=32, k=8, dtype=jnp.float32, blindness_gate=False)
    assert node.initial_state()["c"].dtype == jnp.float32
    with pytest.raises(ValueError, match="floating"):
        _sine(dtype=jnp.int32)


# -- diagnostics reject typos rather than ignoring them (audit A12) ---------------

def test_diagnostics_reject_a_parameter_key_that_is_not_in_the_params_pytree():
    node = _sine(blindness_gate=False)
    state = node.initial_state()
    with pytest.raises(ValueError, match="bogus"):
        node.gradient_capture_ratio(state, {"bogus": 1.0, "theta": 0.9})
    with pytest.raises(ValueError, match="bogus"):
        node.symmetry_break(state, {"bogus": 1.0})
    # A structural constructor entry is not a leaf but is not a typo either.
    node.gradient_capture_ratio(state, {"theta": 0.9, "k": 16})


def test_update_refuses_an_injected_key_that_is_not_a_constructor_parameter():
    """The base ``update`` path, for every subclass, not only the wavelet node.

    ``{"thetta": 0.9}`` used to be merged into the parameter dict and then
    never read: ``update`` returned the constructor-``theta`` answer, eagerly
    and under ``jit`` alike, on this top-k toy as on any other subclass.  The
    refusal names every unknown key; a real key still moves the answer, and
    a structural constructor entry is accepted as before.
    """
    node = _sine(blindness_gate=False)
    state = node.initial_state()
    with pytest.raises(ValueError, match=r"PoissonSineTopKNode 'adaptive': unknown parameter key\(s\) \['thetta'\]"):
        node.update(state, {}, 1.0, params={"thetta": 0.9})
    with pytest.raises(ValueError, match=r"\['thetta'\]"):
        jax.jit(lambda s, t: node.update(s, {}, 1.0, params={"thetta": t}))(state, 0.9)
    with pytest.raises(ValueError, match=r"\['bogus', 'thetta'\]"):
        node.update(state, {}, 1.0, params={"theta": 0.9, "thetta": 0.9, "bogus": 1.0})

    base = np.asarray(node.update(state, {}, 1.0)["c"])
    moved = np.asarray(node.update(state, {}, 1.0, params={"theta": 0.9})["c"])
    assert not np.array_equal(base, moved)
    same = node.update(state, {}, 1.0, params={"theta": node.params["theta"], "k": 16})
    np.testing.assert_array_equal(np.asarray(same["c"]), base)


# -- cost of the cold-start diagnostic (audit A13) --------------------------------

def test_repeated_initial_state_calls_evaluate_the_diagnostic_once():
    """``initial_state()`` is called from add_node, reset_state, the
    profiler, the REST API, the FMI description and the hypothesis
    strategies; the diagnostic is a function of the parameters, so the
    framework must not pay for it once per call."""
    calls = {"n": 0}

    class Counting(PoissonSineTopKNode):
        def compute_full_basis_gradient(self, state, params=None):
            calls["n"] += 1
            return super().compute_full_basis_gradient(state, params)

    node = Counting(n=32, k=8, on_blind="ignore")
    for _ in range(3):
        node.initial_state()
    assert calls["n"] == 0, "policy 'ignore' must not evaluate the diagnostic"

    node = Counting(n=32, k=32)
    for _ in range(3):
        node.initial_state()
    assert calls["n"] == 1, calls


def test_diagnostics_can_be_disabled_globally():
    from maddening.nodes.adaptive import (
        adaptive_diagnostics_enabled, set_adaptive_diagnostics,
    )

    previous = set_adaptive_diagnostics(False)
    try:
        assert not adaptive_diagnostics_enabled()
        # A trap that would otherwise warn constructs silently.
        state = PoissonSineTopKNode(theta=0.5, n=64, k=16).initial_state()
        assert state["c"].shape == (64,)
    finally:
        set_adaptive_diagnostics(previous)
    assert adaptive_diagnostics_enabled() is previous


def test_solve_and_pack_keeps_working_when_numpy_conversion_is_impossible():
    """The non-finite guard must not fire (or fail) under ``jit``."""
    node = _sine(n=32, k=8, blindness_gate=False)
    s = node.initial_state()
    out = jax.jit(lambda st: node.update(st, {}, 1.0))(s)
    assert bool(np.all(np.isfinite(np.asarray(out["c"]))))
