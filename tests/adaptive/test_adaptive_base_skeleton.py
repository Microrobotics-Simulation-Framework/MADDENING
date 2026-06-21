"""M2 tests for the ``AdaptiveNode`` skeleton.

Test list (numbering matches
``plans/MADDENING_ADAPTIVE_NODE_IMPLEMENTATION_PLAN.md`` §4 M2):

1. test_abstract_methods_raise
2. test_state_field_contract
3. test_blindness_constants_overridable
4. test_update_dispatch_flow
5. test_AdaptiveNodeBlindnessError_is_runtime_error
6. test_N_max_required
7. test_subclass_does_not_break_SimulationNode_contract
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from maddening.core.node import SimulationNode
from maddening.nodes.adaptive import AdaptiveNode, AdaptiveNodeBlindnessError


# ----------------------------------------------------------------------
# 1. test_abstract_methods_raise
# ----------------------------------------------------------------------

def test_abstract_methods_raise():
    """A subclass that does not override the abstract hooks raises
    NotImplementedError on the first relevant call."""
    class Bare(AdaptiveNode):
        pass

    node = Bare(N_max=16)
    fake_state = {"c": jnp.zeros(16), "mask": jnp.zeros(16, dtype=bool),
                  "theta": jnp.zeros(1)}
    with pytest.raises(NotImplementedError, match=r"compute_active_set"):
        node.update(fake_state, {}, 1.0)


# ----------------------------------------------------------------------
# 2. test_state_field_contract
# ----------------------------------------------------------------------

def test_state_field_contract():
    """A minimal subclass returns state_fields() == ['c', 'mask', 'theta']."""
    class Minimal(AdaptiveNode):
        pass

    node = Minimal(N_max=16)
    assert node.state_fields() == ["c", "mask", "theta"]
    assert node.N_max == 16


# ----------------------------------------------------------------------
# 3. test_blindness_constants_overridable
# ----------------------------------------------------------------------

def test_blindness_constants_overridable_classattr():
    """A subclass that sets blindness_threshold = 0.5 as class-attr has
    that value read by the base class."""
    class TighterThreshold(AdaptiveNode):
        blindness_threshold = 0.5

    node = TighterThreshold(N_max=16)
    assert node.blindness_threshold == 0.5
    assert node.blindness_break_delta == 0.05  # default unchanged
    assert node.D_threshold == 5               # default unchanged


def test_blindness_constants_overridable_kwarg():
    """Constructor kwargs override the class defaults."""
    class Minimal(AdaptiveNode):
        pass

    node = Minimal(
        N_max=16,
        blindness_threshold=0.5,
        blindness_break_delta=0.10,
        D_threshold=8,
    )
    assert node.blindness_threshold == 0.5
    assert node.blindness_break_delta == 0.10
    assert node.D_threshold == 8


# ----------------------------------------------------------------------
# 4. test_update_dispatch_flow
# ----------------------------------------------------------------------

def test_update_dispatch_flow():
    """update() invokes compute_active_set then solve_frozen, returning
    what solve_frozen produces."""
    class Mock(AdaptiveNode):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.compute_calls = 0
            self.solve_calls = 0

        def compute_active_set(self, state, *, prev=None,
                                is_cold_start=False):
            self.compute_calls += 1
            mask = jnp.zeros(self.N_max, dtype=bool)
            mask = mask.at[:4].set(True)
            return mask

        def solve_frozen(self, state, mask):
            self.solve_calls += 1
            new_c = state["c"] + jnp.where(mask, 1.0, 0.0)
            return {**state, "c": new_c, "mask": mask}

    node = Mock(N_max=8)
    state0 = {"c": jnp.zeros(8), "mask": jnp.zeros(8, dtype=bool),
              "theta": jnp.zeros(1)}
    new_state = node.update(state0, {}, 1.0)
    assert node.compute_calls == 1
    assert node.solve_calls == 1
    # The mock added 1 on the first 4 entries.
    expected_c = jnp.asarray([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(new_state["c"], expected_c)
    assert bool(jnp.all(new_state["mask"][:4]))
    assert bool(jnp.all(~new_state["mask"][4:]))


# ----------------------------------------------------------------------
# 5. test_AdaptiveNodeBlindnessError_is_runtime_error
# ----------------------------------------------------------------------

def test_AdaptiveNodeBlindnessError_is_runtime_error():
    """AdaptiveNodeBlindnessError is catchable as RuntimeError."""
    err = AdaptiveNodeBlindnessError("synthetic")
    assert isinstance(err, RuntimeError)
    with pytest.raises(RuntimeError):
        raise err


# ----------------------------------------------------------------------
# 6. test_N_max_required
# ----------------------------------------------------------------------

def test_N_max_required():
    """N_max is a required kwarg-only argument."""
    class Minimal(AdaptiveNode):
        pass

    with pytest.raises(TypeError):
        Minimal()  # type: ignore[call-arg]


# ----------------------------------------------------------------------
# 7. test_subclass_does_not_break_SimulationNode_contract
# ----------------------------------------------------------------------

def test_subclass_does_not_break_SimulationNode_contract():
    """A minimal subclass remains a SimulationNode and exposes the
    SimulationNode interface (name, delta_t, state_fields)."""
    class Minimal(AdaptiveNode):
        pass

    node = Minimal(N_max=16, name="adapt_1", timestep=0.1)
    assert isinstance(node, SimulationNode)
    assert node.name == "adapt_1"
    assert node.delta_t == 0.1
    # state_fields is callable and returns a list of strings.
    fields = node.state_fields()
    assert isinstance(fields, list)
    assert all(isinstance(f, str) for f in fields)
