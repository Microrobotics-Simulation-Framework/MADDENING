"""Turn off the ``sys.monitoring`` LINE events Hypothesis leaves on.

Once an example fails, Hypothesis runs the rest of the search (the shrink)
under its branch tracer, ``hypothesis.internal.scrutineer.Tracer``, for the
``explain`` phase.  On Python 3.12 and 3.13 the tracer turns LINE events
on for its tool id and, on exit, frees the id and drops its callback but
never turns the events off: ``sys.monitoring.free_tool_id`` does not clear
a tool's events before Python 3.14.  Every line of Python the process
runs after that, in every thread, goes through the line instrumentation,
with no callback to tell it to stop.  Pure-Python code runs several times
slower until the process exits: FastAPI's ``jsonable_encoder``, which
every REST reply goes through, took 24 s instead of 2.3 s per 30 MB state
reply (hypothesis 6.165.3; upstream unchanged at 6.169.1).

A failing example is routine in this suite: ``hypothesis.find`` is a
search whose condition holding *is* the failure, and the verification
tests make ``verify_node`` fail on purpose.  One such test made every
later test of its process slower: after
``tests/api/test_a_checkpoint_load_answers_as_a_params_write_on_every_parameter.py``,
the slow lane's ``[LBMNode]`` case of
``tests/property/test_rest_requests_generated_from_the_schema.py`` took
73 s instead of 6.7 s locally, and 2804 s instead of 200-300 s on a
hosted runner (its ``[LBMPipeNode]`` case).

``tests/conftest.py`` calls :func:`clear_line_events_hypothesis_left`
after every test.  It touches the tool id only when nothing holds it, so a
tracer that is running keeps its events.
"""

from __future__ import annotations

import sys

try:
    from hypothesis.internal.scrutineer import MONITORING_TOOL_ID as TOOL_ID
except ImportError:  # pragma: no cover - the id Hypothesis has always used
    TOOL_ID = 3


def clear_line_events_hypothesis_left() -> bool:
    """Turn off the events left on Hypothesis's tool id once the id is
    free.  ``True`` when there were any."""
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is None:  # pragma: no cover - Python < 3.12 uses settrace
        return False
    if monitoring.get_tool(TOOL_ID) is not None or not monitoring.get_events(TOOL_ID):
        return False
    monitoring.use_tool_id(TOOL_ID, "maddening-tests-clear-line-events")
    try:
        monitoring.set_events(TOOL_ID, monitoring.events.NO_EVENTS)
    finally:
        monitoring.free_tool_id(TOOL_ID)
    return True
