"""Every test leaves the process without the LINE events Hypothesis left on.

``tests/_hypothesis_line_events.py`` turns off the ``sys.monitoring``
events Hypothesis's branch tracer leaves on its freed tool id once an
example has failed, and ``tests/conftest.py`` runs it after every test:
without it, one ``find`` or one ``verify_node`` made to fail slowed every
later test of the process several times over.  These tests hold what it
turns off, what it leaves alone, and that the conftest calls it.
"""

from __future__ import annotations

import random
import sys

from hypothesis import find, settings, strategies as st

import tests.conftest as conftest
from tests import _hypothesis_line_events as line_events

MONITORING = sys.monitoring
TOOL = line_events.TOOL_ID


def _leave_line_events_on() -> None:
    """The state Hypothesis's tracer leaves behind: the tool id freed, its
    LINE events still on."""
    MONITORING.use_tool_id(TOOL, "a tracer that exits as Hypothesis's does")
    MONITORING.set_events(TOOL, MONITORING.events.LINE)
    MONITORING.free_tool_id(TOOL)


def test_the_events_left_on_a_freed_tool_id_are_turned_off():
    _leave_line_events_on()
    assert MONITORING.get_events(TOOL) == MONITORING.events.LINE

    assert line_events.clear_line_events_hypothesis_left() is True

    assert MONITORING.get_events(TOOL) == MONITORING.events.NO_EVENTS
    assert MONITORING.get_tool(TOOL) is None
    assert line_events.clear_line_events_hypothesis_left() is False


def test_a_tracer_that_holds_the_tool_id_keeps_its_events():
    MONITORING.use_tool_id(TOOL, "a tracer that is running")
    try:
        MONITORING.set_events(TOOL, MONITORING.events.LINE)
        assert line_events.clear_line_events_hypothesis_left() is False
        assert MONITORING.get_events(TOOL) == MONITORING.events.LINE
        assert MONITORING.get_tool(TOOL) == "a tracer that is running"
    finally:
        MONITORING.set_events(TOOL, MONITORING.events.NO_EVENTS)
        MONITORING.free_tool_id(TOOL)


def test_no_events_are_left_after_a_found_example():
    """A ``find`` whose example is shrunk -- on Python 3.12 and 3.13
    Hypothesis 6.165-6.169 leave LINE events on after it."""
    find(st.integers(0, 100), lambda _: True, settings=settings(database=None),
         random=random.Random(0))
    line_events.clear_line_events_hypothesis_left()
    assert MONITORING.get_events(TOOL) == MONITORING.events.NO_EVENTS


def test_the_conftest_turns_them_off_after_every_test():
    _leave_line_events_on()
    conftest.pytest_runtest_teardown(item=None, nextitem=None)
    assert MONITORING.get_events(TOOL) == MONITORING.events.NO_EVENTS
