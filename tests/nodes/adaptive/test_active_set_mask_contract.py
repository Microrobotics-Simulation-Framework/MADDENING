"""What ``compute_active_set`` is allowed to return, and what it is told
when it returns something else.

Two of the three ways a selection rule goes wrong are invisible afterwards:

* an **empty** active set solves to ``c = 0`` with an exactly zero
  gradient, and the diagnostics then report a Palais symmetry trap on a
  problem that has no symmetry;
* a **non-boolean** mask (a score array, an ``argsort`` permutation) is
  truthy almost everywhere, so the node silently becomes a full-basis
  solver while ``gradient_capture_ratio`` reports 1.00 -- the frozen set
  *is* the full set.

Both are the mistakes the second subclass is most likely to make:
textbook coefficient-magnitude thresholding selects the empty set at a
cold start, because ``c`` is all zeros there.  So both are refused, with
a message that names the remedy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from tests.conftest import EXAMPLES_COSTLY
from tests.nodes.adaptive._toys import PoissonSineTopKNode

N = 16


def _node_returning(mask, **kw):
    """A working node whose selection rule returns ``mask`` verbatim."""

    class Canned(PoissonSineTopKNode):
        def compute_active_set(self, state, params, *, prev=None, is_cold_start=False):
            return mask

    return Canned(n=N, k=4, **kw)


# -- empty active set --------------------------------------------------------

def test_an_empty_active_set_is_refused_at_the_cold_start():
    node = _node_returning(jnp.zeros(N, dtype=bool))
    with pytest.raises(ValueError) as exc:
        node.initial_state()
    assert "empty active set" in str(exc.value)


def test_an_empty_active_set_is_refused_during_update():
    """Not only at the cold start: a hysteresis rule that drains the set
    mid-run has to be caught where it happens."""
    good = PoissonSineTopKNode(n=N, k=4, blindness_gate=False)
    state = good.initial_state()
    node = _node_returning(jnp.zeros(N, dtype=bool), blindness_gate=False)
    with pytest.raises(ValueError, match="empty active set"):
        node.update(state, {}, 1.0)


def test_the_empty_set_message_names_is_cold_start_and_the_prev_idiom():
    """The remedy is the whole point of the error: an author who meets it
    has to be told to special-case ``is_cold_start`` and to carry ``prev``."""
    node = _node_returning(jnp.zeros(N, dtype=bool))
    with pytest.raises(ValueError) as exc:
        node.initial_state()
    message = str(exc.value)
    assert "is_cold_start" in message
    assert "prev" in message
    assert "hysteresis" in message


def test_an_empty_set_is_refused_rather_than_diagnosed_as_a_symmetry_trap():
    """The regression the audit found: an empty set used to solve to c = 0,
    and the blindness gate then reported a Palais trap on a problem with no
    symmetry -- naming remedies (cold_start, symmetry_break, perturbation)
    that cannot work, because the set is empty at every theta."""
    from maddening.nodes.adaptive import AdaptiveNodeBlindnessError

    node = _node_returning(jnp.zeros(N, dtype=bool))
    with pytest.raises(ValueError) as exc:
        node.initial_state()
    assert not isinstance(exc.value, AdaptiveNodeBlindnessError)
    message = str(exc.value)
    assert "cold_start()" not in message
    assert "symmetry_break()" not in message


def test_a_single_active_mode_is_accepted():
    """The floor is one mode, not a budget: only the *empty* set is refused."""
    mask = jnp.zeros(N, dtype=bool).at[3].set(True)
    state = _node_returning(mask, blindness_gate=False).initial_state()
    assert int(jnp.sum(state["mask"])) == 1
    assert bool(jnp.all(jnp.isfinite(state["c"])))


# -- dtype -------------------------------------------------------------------

# NumPy, not ``jnp``: the x64 fixture is per test and these are built at
# collection time.
@pytest.mark.parametrize("bad", [
    np.arange(N, dtype=np.float64) / N,
    np.argsort(np.arange(N, dtype=np.float64)),
    np.ones(N, dtype=np.int32),
    np.eye(1, N, 2, dtype=np.float32).ravel(),
], ids=["a score array", "an argsort permutation",
        "a 0/1 integer mask", "a 0/1 float mask"])
def test_a_non_boolean_mask_is_refused_rather_than_silently_cast(bad):
    node = _node_returning(jnp.asarray(bad))
    with pytest.raises(ValueError, match="expected a boolean array"):
        node.initial_state()


def test_the_dtype_message_names_the_two_ways_to_build_a_boolean_mask():
    node = _node_returning(jnp.arange(N, dtype=float) / N)
    with pytest.raises(ValueError) as exc:
        node.initial_state()
    message = str(exc.value)
    assert "scores >= threshold" in message
    assert ".at[idx].set(True)" in message
    assert "full-basis" in message  # says what would otherwise happen


def test_a_non_boolean_mask_is_refused_under_jit_too():
    """The dtype is static, so the check holds inside a trace, where the
    emptiness check cannot look at values."""
    node = _node_returning(jnp.ones(N, dtype=jnp.int32), blindness_gate=False)
    state = {"c": jnp.zeros(N), "mask": jnp.ones(N, dtype=bool)}
    with pytest.raises(ValueError, match="expected a boolean array"):
        jax.jit(lambda s: node.update(s, {}, 1.0))(state)


# -- the properties ----------------------------------------------------------

@settings(max_examples=EXAMPLES_COSTLY)  # a node is built and cold-started per draw
@given(values=arrays(np.float64, N, elements=st.floats(-10, 10, width=32)))
def test_any_non_boolean_mask_is_refused_with_a_message_naming_the_remedy(values):
    """Whatever a float selection rule returns -- all zeros, all ones, a
    genuine score array -- it is refused, and the refusal says what to
    return instead.  A bool cast would have accepted every one of them."""
    node = _node_returning(jnp.asarray(values))
    with pytest.raises(ValueError) as exc:
        node.initial_state()
    message = str(exc.value)
    assert "expected a boolean array" in message
    assert "scores >= threshold" in message


@settings(max_examples=EXAMPLES_COSTLY)  # a node is built and cold-started per draw
@given(flags=arrays(np.bool_, N))
def test_a_boolean_mask_is_accepted_exactly_when_it_is_not_empty(flags):
    """The only value-level constraint on a boolean mask: at least one
    entry true.  Anything else the rule picks is the subclass's business."""
    node = _node_returning(jnp.asarray(flags), blindness_gate=False)
    if not flags.any():
        with pytest.raises(ValueError) as exc:
            node.initial_state()
        assert "empty active set" in str(exc.value)
        assert "is_cold_start" in str(exc.value)
    else:
        state = node.initial_state()
        assert np.array_equal(np.asarray(state["mask"]), flags)
        # and the coefficients are zero off the mask, as always
        assert not np.any(np.asarray(state["c"])[~flags])


def test_the_shipped_toys_still_satisfy_the_contract():
    """The guard must not have narrowed the contract past what the tree
    already does."""
    from tests.nodes.adaptive._toys import MaskedDenseNode

    for node in (PoissonSineTopKNode(n=64, k=8, blindness_gate=False),
                 MaskedDenseNode(n=16, k=4, blindness_gate=False)):
        state = node.initial_state()
        assert state["mask"].dtype == jnp.bool_
        assert bool(jnp.any(state["mask"]))
