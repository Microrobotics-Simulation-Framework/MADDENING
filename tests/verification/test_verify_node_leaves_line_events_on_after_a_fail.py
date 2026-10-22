"""``verify_node`` with a FAIL verdict leaves Hypothesis's LINE events on in the caller's process.

Once an example fails, Hypothesis shrinks it under its branch tracer
(``hypothesis.internal.scrutineer.Tracer``) for the ``explain`` phase.  On
Python 3.12 and 3.13 the tracer turns ``sys.monitoring`` LINE events on
for its tool id and, on exit, frees the id and drops its callback but
never turns the events off (``free_tool_id`` clears them only from Python
3.14).  ``verify_node`` runs its checks under Hypothesis with the
profile's phases, ``explain`` included, and a check that finds a
counterexample fails an example on purpose.  So after a FAIL verdict every
line of pure Python the caller runs goes through the line
instrumentation, until the process exits
(MADD-ANO-NEW-records-known-anomalies-0-4-2-5, open).  No value changes;
everything is slower: FastAPI's ``jsonable_encoder`` on about 300,000
floats took 0.11 s before one failing ``finite`` check and 3.6 to 3.8 s
after it (hypothesis 6.165.3 and 6.169.0; jax 0.10.2, 0.11.0 and 0.11.2).

The suite turns the events off after every test (``tests/conftest.py``,
``tests/_hypothesis_line_events.py``).  That protects the suite, not a
user: the tests below look inside the test, before that teardown.

What is held here: the premise as what happens today, the workaround (the
three lines of the suite's helper turn them off), and a strict xfail of the
claim "a FAIL verdict leaves no events on".
"""

from __future__ import annotations

import sys

import jax.numpy as jnp
import pytest

from maddening.core.node import SimulationNode
from maddening.testing.verification import verify_node
from tests._hypothesis_line_events import TOOL_ID, clear_line_events_hypothesis_left

_LABEL = "MADD-ANO-NEW-records-known-anomalies-0-4-2-5"
MONITORING = sys.monitoring


class ForwardNaN(SimulationNode):
    """``x + dt / x``: not finite at ``x == 0``, which the search reaches."""

    def initial_state(self):
        return {"x": jnp.array(0.5, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] + dt / state["x"]}


def _fail_a_verification() -> None:
    """A ``verify_node`` call whose ``finite`` check fails, from a process
    with no events left on Hypothesis's tool id."""
    clear_line_events_hypothesis_left()
    assert MONITORING.get_events(TOOL_ID) == MONITORING.events.NO_EVENTS
    result = verify_node(ForwardNaN(name="n", timestep=0.01), bounds={"x": (0.0, 1.0)},
                         checks=["finite"], max_examples=50, derandomize=True)
    assert result["finite"].status == "FAIL", result["finite"]


def test_a_fail_verdict_leaves_line_events_on_a_freed_tool_id():
    """A premise, as what happens today: the tool id is free (nothing holds
    it, so nothing will turn its events off) and its LINE events are on."""
    try:
        _fail_a_verification()
        assert MONITORING.get_tool(TOOL_ID) is None
        assert MONITORING.get_events(TOOL_ID) & MONITORING.events.LINE
    finally:
        clear_line_events_hypothesis_left()


def test_the_workaround_turns_the_events_off():
    """The workaround: claim the free tool id, turn its events off, free it
    again -- what the suite's helper does."""
    try:
        _fail_a_verification()
        if MONITORING.get_tool(TOOL_ID) is None:
            MONITORING.use_tool_id(TOOL_ID, "clear the events Hypothesis left")
            MONITORING.set_events(TOOL_ID, MONITORING.events.NO_EVENTS)
            MONITORING.free_tool_id(TOOL_ID)
        assert MONITORING.get_events(TOOL_ID) == MONITORING.events.NO_EVENTS
    finally:
        clear_line_events_hypothesis_left()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    f"{_LABEL} (open): a FAIL verdict from verify_node leaves Hypothesis's sys.monitoring "
    "LINE events on for the rest of the process (Python 3.12 and 3.13)"))
def test_a_fail_verdict_leaves_no_events_on():
    """The claim that does not hold today."""
    try:
        _fail_a_verification()
        assert MONITORING.get_events(TOOL_ID) == MONITORING.events.NO_EVENTS
    finally:
        clear_line_events_hypothesis_left()
