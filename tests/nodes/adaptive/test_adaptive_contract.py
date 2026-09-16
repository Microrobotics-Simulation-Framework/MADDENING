"""AdaptiveNode public contract: hooks, state schema, constants, mask semantics."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.solver_utils import ift_linear_solve
from maddening.nodes.adaptive import AdaptiveNode, AdaptiveNodeBlindnessError

from tests.nodes.adaptive._toys import PoissonSineTopKNode


class _Bare(AdaptiveNode):
    """Overrides nothing: every hook must fail loudly."""


def _sine(**kw):
    return PoissonSineTopKNode(**kw)


# -- abstract surface --------------------------------------------------------

def test_abstract_hooks_raise_not_implemented():
    node = _Bare("bare", 1.0, n_max=4, blindness_gate=False)
    state = {"c": jnp.zeros(4), "mask": jnp.zeros(4, dtype=bool)}
    with pytest.raises(NotImplementedError, match="compute_active_set"):
        node.update(state, {}, 1.0)
    with pytest.raises(NotImplementedError, match="compute_active_set"):
        node.initial_state()
    with pytest.raises(NotImplementedError, match="solve_frozen"):
        node.solve_frozen(state, state["mask"], {})
    with pytest.raises(NotImplementedError, match="objective"):
        node.objective(state, {})


def test_n_max_is_required_and_positive():
    with pytest.raises(TypeError):
        _Bare("bare", 1.0)
    with pytest.raises(ValueError, match="n_max"):
        _Bare("bare", 1.0, n_max=0)


def test_public_symbols_are_stable_and_error_is_runtime_error():
    import maddening.nodes.adaptive as pkg
    for sym in (AdaptiveNode, AdaptiveNodeBlindnessError, ift_linear_solve):
        assert sym._stability_level == StabilityLevel.STABLE, sym
    assert issubclass(AdaptiveNodeBlindnessError, RuntimeError)
    assert set(pkg.__all__) == {"AdaptiveNode", "AdaptiveNodeBlindnessError"}


def test_node_meta_is_filled_in():
    m = AdaptiveNode.meta
    assert m.algorithm_id == "MADD-NODE-009"
    assert m.stability == StabilityLevel.STABLE
    assert m.description and m.assumptions and m.limitations and m.hazard_hints
    assert m.implementation_map


# -- constants -----------------------------------------------------------------

def test_spike_constants_are_the_class_defaults():
    assert AdaptiveNode.blindness_threshold == 0.7
    assert AdaptiveNode.blindness_break_delta == 0.05
    assert AdaptiveNode.D_threshold == 5


def test_constants_overridable_per_instance_and_per_subclass():
    node = _sine(blindness_threshold=0.5, blindness_break_delta=0.1, D_threshold=3)
    assert (node.blindness_threshold, node.blindness_break_delta, node.D_threshold) == (0.5, 0.1, 3)
    assert AdaptiveNode.blindness_threshold == 0.7  # class default untouched

    class Lax(PoissonSineTopKNode):
        blindness_threshold = 0.2

    assert Lax().blindness_threshold == 0.2


def test_constants_are_not_parameter_leaves():
    """Diagnostic knobs steer host-side checks; a fit must never see them."""
    node = _sine(blindness_threshold=0.5)
    leaves = node.params_pytree()
    assert not {"blindness_threshold", "blindness_break_delta", "D_threshold",
                "blindness_gate", "n_max"} & set(leaves)
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
