"""A parameter write that overlaps FastAPI's own ``catch_warnings`` block
leaves no warning silenced.

``PUT /graph/params`` runs the node's code several times, warnings
silenced, before it writes (does the state keep its layout?  does the
constructor take the value?).  It holds the server's graph lock while it
does, so two writes to one server never overlap.  FastAPI holds no such
lock: the first time an app is asked for ``/openapi.json`` it builds the
schema's fields, each inside ``warnings.catch_warnings()`` -- 44 blocks for
this server -- which saves the process's filter list on the way in and puts
it back on the way out.

So within **one** server a write's probe and FastAPI's block could overlap.
When FastAPI's block opened while a probe was open and closed after it, the
list it put back was the one it had saved: the probe's.  That used to be
``simplefilter("ignore")`` -- an ``ignore`` for every warning, left in the
process for good (MADD-ANO-204).  The probe's filter now matches only the
thread inside the probe, so the copy FastAPI puts back ignores nothing, and
the next probe to close takes it out.

The overlap is arranged, not waited for: the node's ``initial_state()``,
called inside the write's first probe, holds the probe open until FastAPI's
first block is about to put its list back, and that block is then held
until the write has returned.
"""

from __future__ import annotations

import os
import threading
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode

#: How long a thread waits for the other before the test gives up.
WAIT = 30  # units: s


class _Overlap:
    """The meeting point of the write's probe and FastAPI's block."""

    def __init__(self):
        self.armed = False
        self.probe_open = threading.Event()
        self.foreign_block_closing = threading.Event()
        self.write_returned = threading.Event()
        #: The thread FastAPI builds the schema on (the event loop's).
        self.schema_thread: int | None = None
        self.held = 0
        self.met = False


class _Rod(HeatNode):
    """A rod whose ``initial_state()`` holds the first probe of a write
    open until FastAPI's block has opened and is about to close."""

    overlap = _Overlap()

    def initial_state(self):
        overlap = type(self).overlap
        if overlap.armed and not overlap.probe_open.is_set():
            overlap.probe_open.set()
            overlap.met = overlap.foreign_block_closing.wait(WAIT)
        return super().initial_state()


@pytest.fixture
def overlap(monkeypatch):
    """A fresh meeting point, and the first ``catch_warnings`` block to
    open after the write's probe has -- FastAPI's, on whichever thread
    runs the schema request -- held open, after its body and before it
    puts the filters back, until the write has returned."""
    meeting = _Overlap()
    monkeypatch.setattr(_Rod, "overlap", meeting)
    real_enter = warnings.catch_warnings.__enter__
    real_exit = warnings.catch_warnings.__exit__

    def enter(self):
        if (meeting.armed and meeting.schema_thread is None
                and meeting.probe_open.is_set()):
            meeting.schema_thread = threading.get_ident()
        return real_enter(self)

    def exit_when_the_write_has_returned(self, *exc_info):
        if (meeting.armed and threading.get_ident() == meeting.schema_thread
                and not meeting.foreign_block_closing.is_set()):
            meeting.held += 1
            meeting.foreign_block_closing.set()
            assert meeting.write_returned.wait(WAIT), "the write never returned"
        return real_exit(self, *exc_info)

    monkeypatch.setattr(warnings.catch_warnings, "__enter__", enter)
    monkeypatch.setattr(warnings.catch_warnings, "__exit__", exit_when_the_write_has_returned)
    yield meeting
    meeting.armed = False
    for event in (meeting.probe_open, meeting.foreign_block_closing, meeting.write_returned):
        event.set()


def _server():
    gm = GraphManager()
    gm.add_node(_Rod("rod", 0.01, n_cells=6, length=1.0, thermal_diffusivity=0.005,
                     initial_temperature=1.0))
    gm.compile()
    gm.step()
    return SimulationServer(node_registry={"_Rod": _Rod}, graph_manager=gm).create_app()


def _undamped_pair() -> GraphManager:
    """A graph ``compile()`` warns about (MADD-ANO-098), with a warning
    this suite does not filter."""
    gm = GraphManager()
    for name, rest, start in (("a", 1.0, 0.0), ("b", -1.0, 5.0)):
        gm.add_node(SpringDamperNode(name=name, timestep=0.005, stiffness=10000.0,
                                     damping=0.0, mass=0.5, rest_length=rest,
                                     initial_position=start))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=20, tolerance=1e-6)
    return gm


def _silences_everything(item) -> bool:
    action, message, category, module, lineno = item
    return action == "ignore" and message is None and category is Warning and module is None


def test_a_write_that_overlaps_the_first_openapi_request_leaves_no_warning_silenced(overlap):
    app = _server()
    before = list(warnings.filters)
    replies: dict = {}
    errors: list = []

    def write():
        try:
            client = TestClient(app, raise_server_exceptions=False)
            replies["write"] = client.put(
                "/graph/params/rod", json={"params": {"thermal_diffusivity": 0.004}}).status_code
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)
        finally:
            overlap.write_returned.set()

    def schema():
        try:
            assert overlap.probe_open.wait(WAIT), "the write's probe never opened"
            client = TestClient(app, raise_server_exceptions=False)
            replies["schema"] = client.get("/openapi.json").status_code
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    overlap.armed = True
    threads = [threading.Thread(target=write, daemon=True), threading.Thread(target=schema, daemon=True)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(3 * WAIT)
    finally:
        overlap.armed = False
    assert not any(t.is_alive() for t in threads)
    assert not errors, [f"{type(e).__name__}: {e}" for e in errors]
    assert replies == {"write": 200, "schema": 200}
    assert overlap.met and overlap.held == 1, (
        "the write's probe and FastAPI's block did not overlap as arranged "
        f"(met={overlap.met}, held={overlap.held}): this test is not testing it")

    # Nothing left in the filters silences a warning, in this thread or
    # another, and compile()'s own advisory is delivered -- as the error a
    # filter at the *back* makes of it, behind anything left in front.
    assert not [item for item in warnings.filters if _silences_everything(item)], (
        warnings.filters[:3])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UserWarning, append=True)
        with pytest.raises(UserWarning, match="MADD-ANO-098"):
            _undamped_pair().compile()

    # The copy FastAPI put back is taken out by the next probe to close.
    client = TestClient(app, raise_server_exceptions=False)
    assert client.put("/graph/params/rod",
                      json={"params": {"thermal_diffusivity": 0.0045}}).status_code == 200
    assert list(warnings.filters) == before
