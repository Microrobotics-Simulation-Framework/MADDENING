"""``verify_node`` on fields with constraints.

The battery draws every state field on its own, in a box.  A unit
quaternion, positive populations, a counter used as an index or two fields
that must agree do not survive that: the node is judged on states it is
never given.  ``constrain_state`` / ``constrain_boundary`` map every draw
before any check uses it, and ``state_strategy`` / ``boundary_strategy``
replace the sampling for what a map cannot express.

What is held here:

* a node that is only right on its constraint set fails without the option
  and passes with it (a quaternion, positivity, an index);
* the map reaches **every** check: each one, run alone, hands the node
  constrained examples only (a fault that skips the map in one check makes
  that check's case fail);
* a map that changes a shape, a dtype or the set of fields is refused, by
  name, at the first draw it does so on, and never reported as the node's
  failure;
* a supplied strategy's examples reach the node as drawn, shrink, and are
  reported in the usual form;
* a caller who passes none of this draws what they drew before.
"""

import hashlib
import inspect
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import Phase, assume, given, settings
from hypothesis import strategies as st

import maddening.testing.verification as ver
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.nodes.rigid_body import RigidBodyNode
from maddening.testing.verification import (
    DEFAULT_CHECKS,
    assert_node_verified,
    make_inputs,
    node_finite,
    verify_node,
)

# ``verify_node`` takes its own example count (the Hypothesis profile of
# tests/conftest.py does not reach it).  25 is above the house floor of 20;
# every fault these tests look for shows on the first, all-zero example, so
# depth buys nothing here.
KW = dict(max_examples=25, derandomize=True)



@pytest.fixture(scope="module", autouse=True)
def without_hypothesis_explain_phase():
    """Run this module's batteries without Hypothesis's ``explain`` phase.

    The phase re-runs a failing test with other values to annotate the
    failure, and ``verify_node`` keeps none of it (a result holds the last
    example and the message).  Measured here (hypothesis 6.165.3, cores
    16-23): a check that fails on its first example runs 113 examples with
    the phase and 13 without on the node below, 477 and 13 on
    ``RigidBodyNode``.  A dozen tests of this module make a check fail on
    purpose; the verdict and the shrunk counterexample they assert on are
    the same either way.  ``_run`` builds its settings from the profile in
    force, so the profile is where the phase is switched off.
    """
    before = settings.get_current_profile_name()
    settings.register_profile(
        "verify_node_constrained_inputs", parent=settings.default,
        phases=[phase for phase in Phase if phase is not Phase.explain],
    )
    settings.load_profile("verify_node_constrained_inputs")
    yield
    settings.load_profile(before)


SCHEDULE = jnp.asarray([0.5, 1.0, 1.5, 2.0, 2.5], jnp.float32)
#: How sharply ``Constrained`` leaves the unit sphere: its orientation
#: factor is NaN once ``| |q|^2 - 1 | > 1e-3``.
SHARPNESS = 1e6
FLOOR = 1e-3


class Constrained(SimulationNode):
    """A node whose step is right only on its constraint set.

    * ``orientation`` must be a unit quaternion: the factor
      ``sqrt(1 - SHARPNESS (|q|^2 - 1)^2)`` is 1, with zero gradient, on the
      sphere and NaN (value and gradient) off it;
    * ``f`` must be positive (``log``);
    * ``step`` is an ``int32`` counter read as an index into ``SCHEDULE``
      with ``mode="fill"``: out of range, the read is NaN;
    * the boundary input ``gain`` must be positive (``log``).

    ``seen`` holds every concrete value ``update`` was called with (a
    field that is being traced -- under ``jax.jit``, or a float field under
    ``jax.grad`` -- has no value to record).
    """

    def __init__(self, name, timestep, rate=0.5):
        super().__init__(name, timestep, rate=rate)
        self.seen = []

    def initial_state(self):
        return {
            "orientation": jnp.asarray([1.0, 0.0, 0.0, 0.0], jnp.float32),
            "f": jnp.full(3, 0.25, jnp.float32),
            "step": jnp.asarray(0, jnp.int32),
        }

    def boundary_input_spec(self):
        return {"gain": BoundaryInputSpec(shape=(), description="a positive gain")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        concrete = {
            name: np.asarray(value)
            for name, value in {**state, **boundary_inputs}.items()
            if not isinstance(value, jax.core.Tracer)
        }
        if concrete:        # nothing to record while jax.jit traces the call
            self.seen.append(concrete)
        q, f = state["orientation"], state["f"]
        on_sphere = jnp.sqrt(1.0 - SHARPNESS * (jnp.sum(q * q) - 1.0) ** 2)
        setpoint = jnp.take(SCHEDULE, state["step"], mode="fill")
        drive = setpoint * jnp.log(boundary_inputs["gain"])
        return {
            "orientation": q * on_sphere,
            "f": f * (1.0 + dt * p["rate"] * jnp.log(f)) + dt * drive,
            "step": state["step"],
        }


def unit_quaternion(state):
    """Normalise in float64: the square of a tiny float32 draw underflows,
    and the zero quaternion has no direction (the identity stands in)."""
    q = np.asarray(state["orientation"], dtype=np.float64)
    norm = np.linalg.norm(q)
    q = q / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0])
    return {**state, "orientation": jnp.asarray(q, dtype=state["orientation"].dtype)}


def positive_populations(state):
    return {**state, "f": jnp.abs(state["f"]) + FLOOR}


def step_in_schedule(state):
    return {**state, "step": jnp.clip(state["step"], 0, SCHEDULE.shape[0] - 1)}


STATE_CONSTRAINTS = {
    "orientation": unit_quaternion,
    "f": positive_populations,
    "step": step_in_schedule,
}


def constrain_state(state, *, leave_out=None):
    for name, fn in STATE_CONSTRAINTS.items():
        if name != leave_out:
            state = fn(state)
    return state


def constrain_boundary(boundary_inputs):
    return {**boundary_inputs, "gain": jnp.abs(boundary_inputs["gain"]) + FLOOR}


#: Whether a recorded value satisfies its field's constraint.
HOLDS = {
    "orientation": lambda q: abs(float(np.sum(q.astype(np.float64) ** 2)) - 1.0) < 1e-5,
    "f": lambda f: bool(np.all(f > 0)),
    "step": lambda step: 0 <= int(step) < SCHEDULE.shape[0],
    "gain": lambda gain: float(gain) > 0,
}


def violations(records):
    return [
        (name, value.tolist())
        for record in records for name, value in record.items()
        if not HOLDS[name](value)
    ]


def _node():
    return Constrained("c", 0.01)


# ---------------------------------------------------------------------------
# Fails without the option, passes with it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", sorted(STATE_CONSTRAINTS))
def test_a_field_drawn_off_its_constraint_fails_the_battery(field):
    """Each constraint, left out on its own, is what fails: the node is
    judged on a state it is never given."""
    result = verify_node(
        _node(), checks=["finite"],
        constrain_state=lambda s: constrain_state(s, leave_out=field),
        constrain_boundary=constrain_boundary, **KW,
    )["finite"]
    assert result.failed, str(result)
    state = result.counterexample["state"]
    assert not HOLDS[field](np.asarray(state[field])), state
    assert not violations([{k: np.asarray(v) for k, v in state.items() if k != field}])


def test_an_unconstrained_boundary_input_fails_the_battery():
    result = verify_node(
        _node(), checks=["finite"], constrain_state=constrain_state, **KW,
    )["finite"]
    assert result.failed, str(result)
    assert not HOLDS["gain"](np.asarray(result.counterexample["boundary_inputs"]["gain"]))


def test_assert_node_verified_passes_the_options_on():
    assert_node_verified(
        _node(), checks=["finite"],
        constrain_state=constrain_state, constrain_boundary=constrain_boundary, **KW,
    )
    with pytest.raises(AssertionError, match="finite: FAIL"):
        assert_node_verified(_node(), checks=["finite"], **KW)


def test_bounds_still_set_the_box_the_constrained_draw_comes_from():
    """``constrain_state`` maps the draw; it does not replace ``bounds``."""
    node = _node()
    box = {"orientation": (0.5, 1.0), "f": (2.0, 3.0), "step": (1, 3)}
    result = verify_node(
        node, box, checks=["finite"],
        constrain_state=unit_quaternion, constrain_boundary=constrain_boundary, **KW,
    )["finite"]
    assert result.status == "PASS", str(result)
    assert len(node.seen) == result.n_examples
    for record in node.seen:
        assert np.all(record["orientation"] > 0)          # the positive orthant drawn ...
        assert HOLDS["orientation"](record["orientation"])   # ... and normalised
        assert np.all((record["f"] >= 2.0) & (record["f"] <= 3.0))
        assert 1 <= int(record["step"]) <= 3


# ---------------------------------------------------------------------------
# The map reaches every check
# ---------------------------------------------------------------------------

#: Every check that draws examples: the default battery and the three a
#: caller opts into.
OPT_IN = {
    "boundedness": dict(output_bounds={"step": (0, SCHEDULE.shape[0] - 1)}),
    "energy_monotone": dict(energy_fn=lambda state: 0.0),
    "constrained_in": dict(invariants={
        "constrained_in": lambda state, out, boundary_inputs, dt: not violations(
            [{k: np.asarray(v) for k, v in {**state, **boundary_inputs}.items()}]),
    }),
}
EVERY_CHECK = (*DEFAULT_CHECKS, *OPT_IN)


@pytest.mark.parametrize("check", EVERY_CHECK)
def test_every_check_hands_the_node_constrained_examples_only(check):
    """One check at a time, so no other check's evaluations stand in for
    it: every concrete value the node was called with satisfies its
    constraint, and the check passes -- which, for the values a gradient
    or a compiled call traces, it could not on a raw draw (the first is
    the zero quaternion: NaN, value and gradient)."""
    node = _node()
    results = verify_node(
        node, checks=[check] if check in DEFAULT_CHECKS else [],
        constrain_state=constrain_state, constrain_boundary=constrain_boundary,
        **OPT_IN.get(check, {}), **KW,
    )
    assert set(results) == {check}
    result = results[check]
    # PASS, not SKIP: the node takes params, so the params checks drew too.
    assert result.status == "PASS", str(result)
    assert result.n_examples > 0
    assert node.seen, "the check never called the node with a concrete value"
    assert not violations(node.seen)
    # The index field and the boundary input are concrete in every check
    # (under jax.grad too), so each check's own evaluations were seen.
    assert all("step" in record and "gain" in record for record in node.seen)


def test_the_inputs_bundle_draws_constrained_examples():
    """What a standalone ``node_*`` check, or a caller's own ``@given``,
    draws from ``make_inputs(...).strategy()``."""
    inputs = make_inputs(
        _node(), constrain_state=constrain_state, constrain_boundary=constrain_boundary,
    )
    seen = []

    @given(inputs.strategy())
    def draws(example):
        state, boundary_inputs, _ = example
        seen.append({k: np.asarray(v) for k, v in {**state, **boundary_inputs}.items()})

    draws()
    assert seen and not violations(seen)
    node = _node()
    standalone = node_finite(make_inputs(
        node, constrain_state=constrain_state, constrain_boundary=constrain_boundary,
    ), **KW)
    assert standalone.status == "PASS", str(standalone)
    assert node.seen and not violations(node.seen)


def test_a_refused_map_fails_a_caller_s_own_given_test_with_a_value_error():
    """Outside the battery the refusal is an ordinary exception of the
    caller's test, which Hypothesis reports the way it reports any other."""
    fn, message = BAD_STATE_MAPS["dtype"]

    @given(make_inputs(_node(), constrain_state=fn).strategy())
    def draws(example):
        pass

    with pytest.raises(ValueError, match=message):
        draws()


def test_the_reported_counterexample_is_the_constrained_state():
    """A failure on a constrained draw reports the state the node was given,
    not the raw draw behind it."""
    class FailsEverywhere(Constrained):
        def update(self, state, boundary_inputs, dt, *, params=None):
            out = super().update(state, boundary_inputs, dt, params=params)
            return {**out, "f": out["f"] * jnp.nan}

    result = verify_node(
        FailsEverywhere("c", 0.01), checks=["finite"],
        constrain_state=constrain_state, constrain_boundary=constrain_boundary, **KW,
    )["finite"]
    assert result.failed
    assert set(result.counterexample) == {"state", "boundary_inputs", "dt"}
    shown = {**result.counterexample["state"], **result.counterexample["boundary_inputs"]}
    assert not violations([{k: np.asarray(v) for k, v in shown.items()}])
    # Shrunk: the simplest draw, mapped (the zero quaternion's stand-in).
    assert np.array_equal(np.asarray(shown["orientation"]), [1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# A map that changes the layout is refused, by name
# ---------------------------------------------------------------------------


def _in_place_dtype(state):
    state["step"] = state["step"].astype(jnp.float32)
    return state


BAD_STATE_MAPS = {
    "shape": (lambda s: {**s, "orientation": s["orientation"][None]},
              r"constrain_state changed the shape of 'orientation': \(4,\) -> \(1, 4\)"),
    "dtype": (lambda s: {**s, "step": s["step"].astype(jnp.float32)},
              r"constrain_state changed the dtype of 'step': int32 -> float32"),
    "dtype, edited in place": (
        _in_place_dtype, r"constrain_state changed the dtype of 'step': int32 -> float32"),
    "dtype, a numpy double": (
        lambda s: {**s, "f": np.asarray(s["f"], np.float64)},
        r"constrain_state changed the dtype of 'f': float32 -> float64"),
    "a Python scalar": (
        lambda s: {**s, "step": 2},
        r"constrain_state changed the dtype of 'step': int32 -> int64"),
    "dropped field": (lambda s: {k: v for k, v in s.items() if k != "f"},
                      r"constrain_state changed the set of fields: dropped \['f'\]"),
    "added field": (lambda s: {**s, "norm": s["f"]},
                    r"constrain_state changed the set of fields: added \['norm'\]"),
    "not a dict": (lambda s: s["orientation"],
                   r"constrain_state must return the dict it maps .*'orientation'.*, got "),
    "raises": (lambda s: {**s, "orientation": s["orientatoin"]},
               r"constrain_state raised KeyError on a drawn example: 'orientatoin'"),
}
BAD_BOUNDARY_MAPS = {
    "shape": (lambda b: {**b, "gain": b["gain"][None]},
              r"constrain_boundary changed the shape of 'gain': \(\) -> \(1,\)"),
    "dtype": (lambda b: {**b, "gain": b["gain"] > 0},
              r"constrain_boundary changed the dtype of 'gain': float32 -> bool"),
    "dropped field": (lambda b: {},
                      r"constrain_boundary changed the set of fields: dropped \['gain'\]"),
    "added field": (lambda b: {**b, "offset": b["gain"]},
                    r"constrain_boundary changed the set of fields: added \['offset'\]"),
    "not a dict": (lambda b: None, r"constrain_boundary must return the dict it maps .*'gain'"),
    "raises": (lambda b: {"gain": b["gian"]},
               r"constrain_boundary raised KeyError on a drawn example: 'gian'"),
}


@pytest.mark.parametrize("kind", sorted(BAD_STATE_MAPS))
def test_a_state_map_that_changes_the_layout_is_refused_by_name(kind):
    fn, message = BAD_STATE_MAPS[kind]
    with pytest.raises(ValueError, match=message):
        verify_node(_node(), checks=["finite"], constrain_state=fn,
                    constrain_boundary=constrain_boundary, **KW)


@pytest.mark.parametrize("kind", sorted(BAD_BOUNDARY_MAPS))
def test_a_boundary_map_that_changes_the_layout_is_refused_by_name(kind):
    fn, message = BAD_BOUNDARY_MAPS[kind]
    with pytest.raises(ValueError, match=message):
        verify_node(_node(), checks=["finite"], constrain_state=constrain_state,
                    constrain_boundary=fn, **KW)


@pytest.mark.parametrize("check", EVERY_CHECK)
def test_the_refusal_is_raised_from_every_check_never_reported_as_a_failure(check):
    """A refused map says nothing about the node: no check may package it
    as a ``FAIL`` (which ``update`` raising on some input is)."""
    fn, message = BAD_STATE_MAPS["dtype"]
    with pytest.raises(ValueError, match=message):
        verify_node(
            _node(), checks=[check] if check in DEFAULT_CHECKS else [],
            constrain_state=fn, constrain_boundary=constrain_boundary,
            **OPT_IN.get(check, {}), **KW,
        )


def test_the_other_entry_points_refuse_it_too():
    fn, message = BAD_STATE_MAPS["shape"]
    with pytest.raises(ValueError, match=message):
        assert_node_verified(_node(), checks=["finite"], constrain_state=fn, **KW)
    with pytest.raises(ValueError, match=message):
        node_finite(make_inputs(_node(), constrain_state=fn), **KW)


def test_a_map_that_misbehaves_on_some_draws_only_is_still_refused():
    """The layout is checked on every draw, not on the first: here the
    simplest example (all zeros) is mapped properly and evaluated, and the
    map goes wrong on any other."""
    def beyond_the_first(state):
        if np.any(np.asarray(state["f"])):
            return {**constrain_state(state), "step": state["step"].astype(jnp.float32)}
        return constrain_state(state)

    node = _node()
    with pytest.raises(ValueError, match="changed the dtype of 'step'"):
        verify_node(node, checks=["finite"], constrain_state=beyond_the_first,
                    constrain_boundary=constrain_boundary, **KW)
    assert node.seen and not violations(node.seen)


def test_a_refusal_ends_the_check_at_the_first_refused_draw():
    """Nothing is evaluated after it, and it is the refusal that is raised
    even when the node has already failed: Hypothesis is not left to
    shrink a refusal, to draw two hundred more examples "explaining" it,
    or to report it in a group beside the node's failure."""
    class FailsEverywhere(Constrained):
        def update(self, state, boundary_inputs, dt, *, params=None):
            out = super().update(state, boundary_inputs, dt, params=params)
            return {**out, "f": out["f"] * jnp.nan}

    calls = []

    def from_the_second_call(state):
        calls.append(None)
        if len(calls) > 1:
            return {**constrain_state(state), "step": state["step"].astype(jnp.float32)}
        return constrain_state(state)

    node = FailsEverywhere("c", 0.01)
    with pytest.raises(ValueError, match="changed the dtype of 'step'") as caught:
        verify_node(node, checks=["finite"], constrain_state=from_the_second_call,
                    constrain_boundary=constrain_boundary, **KW)
    assert len(calls) == 2 and len(node.seen) == 1
    assert type(caught.value) is ValueError and caught.value.__cause__ is None
    assert not getattr(caught.value, "__notes__", [])
    # Not derandomised: the same.
    calls.clear()
    with pytest.raises(ValueError, match="changed the dtype of 'step'"):
        verify_node(FailsEverywhere("c", 0.01), checks=["finite"],
                    constrain_state=from_the_second_call,
                    constrain_boundary=constrain_boundary, max_examples=25)
    assert len(calls) == 2


@pytest.mark.parametrize("kwargs, error, message", [
    (dict(state_strategy=st.just({}), bounds={"f": (0.0, 1.0)}), ValueError,
     r"state_strategy replaces the sampling of every state field, so bounds for \['f'\]"),
    (dict(boundary_strategy=st.just({}), boundary_bounds={"gain": (0.0, 1.0)}), ValueError,
     r"boundary_strategy replaces .* boundary_bounds for \['gain'\] would be ignored"),
    (dict(boundary_strategy=st.just({}), boundary_inputs={"gain": 1.0}), ValueError,
     "boundary_inputs is one fixed dict and boundary_strategy draws the dict"),
    (dict(constrain_boundary=constrain_boundary, boundary_inputs={"gain": 1.0}), ValueError,
     "constrain_boundary maps drawn boundary inputs, and boundary_inputs is one fixed dict"),
    (dict(constrain_state={"f": (0.0, 1.0)}), TypeError,
     "constrain_state must be a function .* got dict"),
    (dict(constrain_boundary=1.0), TypeError, "constrain_boundary must be a function .* got float"),
    (dict(state_strategy=constrain_state), TypeError,
     "state_strategy must be a Hypothesis strategy .* got function"),
    (dict(boundary_strategy={"gain": 1.0}), TypeError,
     "boundary_strategy must be a Hypothesis strategy .* got dict"),
])
def test_sampling_options_that_contradict_each_other_are_refused(kwargs, error, message):
    node = _node()
    for call in (verify_node, assert_node_verified, make_inputs):
        with pytest.raises(error, match=message):
            call(node, **kwargs)
    assert not node.seen       # refused before anything is drawn


# ---------------------------------------------------------------------------
# A supplied strategy
# ---------------------------------------------------------------------------

N_CELLS = 8


class Located(SimulationNode):
    """A position and the index of the cell it lies in -- two fields that
    must agree.  ``sqrt`` of the offsets into the cell is NaN when they do
    not."""

    def __init__(self, name, timestep, limit=None):
        super().__init__(name, timestep)
        self.limit = limit
        self.seen = []

    def initial_state(self):
        return {"x": jnp.asarray(0.0, jnp.float32), "cell": jnp.asarray(0, jnp.int32)}

    def update(self, state, boundary_inputs, dt):
        self.seen.append((state, boundary_inputs))
        x, cell = state["x"], state["cell"]
        inside = jnp.sqrt(x * N_CELLS - cell) + jnp.sqrt(cell + 1 - x * N_CELLS)
        if self.limit is not None:
            inside = jnp.where(cell > self.limit, jnp.nan, inside)
        return {"x": x + dt * 0.0 * inside, "cell": cell}


@st.composite
def located_states(draw):
    """The cell is drawn; the position is placed inside it (sixteenths of a
    cell: exact in float32, so the two fields agree exactly)."""
    cell = draw(st.integers(0, N_CELLS - 1))
    sixteenths = draw(st.integers(0, 15))
    return {"x": jnp.asarray((cell + sixteenths / 16) / N_CELLS, jnp.float32),
            "cell": jnp.asarray(cell, jnp.int32)}


def _agree(state):
    return int(np.floor(float(state["x"]) * N_CELLS)) == int(state["cell"])


def test_fields_that_must_agree_fail_when_drawn_on_their_own():
    result = verify_node(Located("l", 0.01), {"x": (0.0, 1.0), "cell": (0, N_CELLS - 1)},
                         checks=["finite"], **KW)["finite"]
    assert result.failed and not _agree(result.counterexample["state"])


def test_a_supplied_state_strategy_is_used_as_given():
    """The node is called with the very objects the strategy yielded, and
    with nothing else."""
    drawn = []
    strategy = located_states().map(lambda state: drawn.append(state) or state)
    node = Located("l", 0.01)
    result = verify_node(node, state_strategy=strategy, checks=["finite"], **KW)["finite"]
    assert result.status == "PASS", str(result)
    assert len(node.seen) == result.n_examples > 1
    for state, _ in node.seen:
        assert any(state is yielded for yielded in drawn)
        assert _agree(state)
    assert len({(float(s["x"]), int(s["cell"])) for s, _ in node.seen}) > 1


def test_a_supplied_strategy_shrinks_and_reports_in_the_usual_form():
    """The node fails above cell 2: the counterexample is the strategy's
    simplest failing example (cell 3, at its left edge), under the keys a
    box-drawn counterexample has."""
    result = verify_node(Located("l", 0.01, limit=2), state_strategy=located_states(),
                         checks=["finite"], **KW)["finite"]
    assert result.failed, str(result)
    usual = verify_node(Located("l", 0.01), checks=["finite"], **KW)["finite"]
    assert usual.failed
    assert list(result.counterexample) == list(usual.counterexample) == [
        "state", "boundary_inputs", "dt"]
    state = result.counterexample["state"]
    assert int(state["cell"]) == 3 and float(state["x"]) == 3 / N_CELLS
    assert str(result).startswith("finite: FAIL\n  counterexample: {'state': {'x': ")
    assert "non-finite value in 'x'" in result.detail


def test_a_supplied_boundary_strategy_is_used_as_given():
    drawn = []
    strategy = st.fixed_dictionaries({
        "gain": st.sampled_from([0.5, 1.0, 2.0]).map(lambda g: jnp.asarray(g, jnp.float32)),
    }).map(lambda inputs: drawn.append(inputs) or inputs)
    node = _node()
    result = verify_node(node, checks=["finite"], constrain_state=constrain_state,
                         boundary_strategy=strategy, **KW)["finite"]
    assert result.status == "PASS", str(result)
    assert {float(record["gain"]) for record in node.seen} <= {0.5, 1.0, 2.0}
    assert len(node.seen) == result.n_examples > 1
    located = Located("l", 0.01)
    verify_node(located, state_strategy=located_states(), boundary_strategy=strategy,
                checks=["finite"], **KW)
    assert all(any(inputs is yielded for yielded in drawn) for _, inputs in located.seen)


def test_a_map_is_applied_to_a_supplied_strategy_s_examples():
    """Both given: the strategy draws, the map constrains."""
    raw = st.fixed_dictionaries({
        "orientation": st.sampled_from([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 4.0]]).map(
            lambda q: jnp.asarray(q, jnp.float32)),
        "f": st.just(jnp.full(3, 0.25, jnp.float32)),
        "step": st.integers(0, 4).map(lambda i: jnp.asarray(i, jnp.int32)),
    })
    node = _node()
    result = verify_node(node, state_strategy=raw, constrain_state=unit_quaternion,
                         constrain_boundary=constrain_boundary, checks=["finite"], **KW)["finite"]
    assert result.status == "PASS", str(result)
    assert node.seen and not violations(node.seen)
    unmapped = verify_node(_node(), state_strategy=raw, constrain_boundary=constrain_boundary,
                           checks=["finite"], **KW)["finite"]
    assert unmapped.failed


def test_a_strategy_that_raises_while_generating_is_refused_not_reported_as_a_failure():
    """Otherwise the battery would report the node as failed, with the
    previous example as its counterexample."""
    @st.composite
    def broken(draw):
        cell = draw(st.integers(0, N_CELLS - 1))
        if cell > 2:
            raise RuntimeError("no such cell")
        return {"x": jnp.asarray(cell / N_CELLS, jnp.float32), "cell": jnp.asarray(cell, jnp.int32)}

    with pytest.raises(ValueError, match="state_strategy raised RuntimeError while generating "
                                         "an example: no such cell") as caught:
        verify_node(Located("l", 0.01), state_strategy=broken(), checks=["finite"], **KW)
    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.parametrize("option, message", [
    ("state_strategy", "state_strategy must yield the dict of fields the node is given, got tuple"),
    ("boundary_strategy", "boundary_strategy must yield the dict of fields the node is given, got tuple"),
])
def test_a_strategy_that_does_not_yield_a_dict_is_refused(option, message):
    node = Located("l", 0.01)
    kwargs = {"state_strategy": located_states(), option: st.just((0.5, 4))}
    with pytest.raises(ValueError, match=message):
        verify_node(node, checks=["finite"], **kwargs, **KW)
    assert not node.seen


def test_a_strategy_that_rejects_examples_is_not_a_refusal():
    """``assume`` inside a supplied strategy, or inside a map, is
    Hypothesis's own control flow: the example is discarded, nothing is
    refused."""
    @st.composite
    def not_in_cell_two(draw):
        state = draw(located_states())
        assume(int(state["cell"]) != 2)
        return state

    def not_in_cell_one(state):
        assume(int(state["cell"]) != 1)
        return state

    node = Located("l", 0.01)
    result = verify_node(node, state_strategy=not_in_cell_two(),
                         constrain_state=not_in_cell_one, checks=["finite"], **KW)["finite"]
    assert result.status == "PASS", str(result)
    cells = {int(state["cell"]) for state, _ in node.seen}
    assert cells and cells.isdisjoint({1, 2})


# ---------------------------------------------------------------------------
# A real node: the rigid body's orientation
# ---------------------------------------------------------------------------

BODY_BOX = {
    "position": (-10.0, 10.0), "velocity": (-10.0, 10.0),
    "orientation": (-1.0, 1.0), "angular_velocity": (-5.0, 5.0),
}
BODY_INPUTS = {"force": (-50.0, 50.0), "torque": (-50.0, 50.0)}


def test_the_rigid_body_needs_its_orientation_constrained():
    """A box around the origin holds the zero quaternion, which
    ``RigidBodyNode`` cannot normalise: without the option the battery
    fails for a reason that is not a defect, and the only way round it was
    a box in the positive orthant.  (``finite`` only: one eager
    ``jax.grad`` of this node is a third of a second, and the whole
    battery runs on it, constrained, in ``test_builtin_nodes_verified``.)"""
    def body():
        return RigidBodyNode("r", 0.01, mass=2.0, inertia=(1.0, 2.0, 3.0))

    raw = verify_node(body(), BODY_BOX, boundary_bounds=BODY_INPUTS,
                      checks=["finite"], **KW)["finite"]
    assert raw.failed and "non-finite value in 'orientation'" in raw.detail
    assert not np.any(np.asarray(raw.counterexample["state"]["orientation"]))
    seen = []

    def recording(state):
        state = unit_quaternion(state)
        seen.append(np.asarray(state["orientation"]))
        return state

    constrained = verify_node(body(), BODY_BOX, boundary_bounds=BODY_INPUTS,
                              checks=["finite"], constrain_state=recording, **KW)["finite"]
    assert constrained.status == "PASS", str(constrained)
    # (Hypothesis draws a few examples it does not go on to run.)
    assert len(seen) >= constrained.n_examples > 1
    assert all(HOLDS["orientation"](q) for q in seen)
    # What the positive-orthant box never drew: components of either sign.
    assert any(np.any(q < 0) for q in seen) and any(np.any(q > 0) for q in seen)


# ---------------------------------------------------------------------------
# Nothing changes for a caller who passes none of it
# ---------------------------------------------------------------------------

#: sha256 of the text of ``prop`` in ``verification._run``, decorators
#: included, as it stood before these options existed.
PROP_TEXT = "fa8c050c5a21592a5aedac6a71dd296f71cb26d6bfd63748bfa50a6aa412fcab"


def _prop_text():
    source = inspect.getsource(ver._run)
    start = source.index("    @settings(\n")
    end = source.index("    try:\n        prop()\n")
    return source[start:end]


def test_the_text_a_derandomised_battery_is_seeded_from_is_unchanged():
    """``derandomize=True`` seeds Hypothesis from the source text of the
    function it runs -- ``prop`` in ``_run`` -- so editing that function
    changes the examples of every derandomised battery, here and in every
    package that pins one.  The constraint options were added without
    touching it (they act in ``_Inputs.strategy``).  If this fails, the
    edit moved every derandomised example: put the change somewhere else,
    or change the digest knowingly and say so in the changelog."""
    text = _prop_text()
    assert "def prop(args):" in text and "@given(inputs.strategy())" in text
    assert hashlib.sha256(text.encode()).hexdigest() == PROP_TEXT


def test_without_the_options_the_strategy_is_the_one_it_was():
    """No map and no wrapper are put in the way of a caller who asks for
    none: the bundle's strategy is the three it always was."""
    node = _node()
    plain = make_inputs(node, {"f": (0.0, 1.0)}, boundary_bounds={"gain": (1.0, 2.0)})
    assert (plain.constrain_state, plain.constrain_boundary,
            plain.state_strategy, plain.boundary_strategy) == (None, None, None, None)
    assert repr(plain.strategy()) == repr(st.tuples(
        ver.node_states(node, {"f": (0.0, 1.0)}, dtype=np.dtype(np.float32)),
        ver.boundary_inputs_for(node, {"gain": (1.0, 2.0)}, dtype=np.dtype(np.float32)),
        ver.bounded_dt(1e-4, 0.01),
    ))
    mapped = make_inputs(node, constrain_state=constrain_state)
    assert repr(mapped.strategy()) != repr(make_inputs(node).strategy())
