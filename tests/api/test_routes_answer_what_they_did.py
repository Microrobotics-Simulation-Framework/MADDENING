"""Graph, simulation and checkpoint routes answer what they did, at the edges
of what the REST guide and the route docstrings say.

Rows of ``docs/validation/rest_runpod_claims.yaml`` this module pins:

* ``POST /sim/step`` and ``POST /sim/run`` answer a graph that cannot step
  with a 400 naming why; a single step that raises stores nothing, and a
  run that raises at its first step moves nothing.  A run that raises
  part-way has moved the graph by every step before, and its 400 says how
  many (``steps_run``);
* the runner routes are a 409 when there is no runner to act on;
* ``POST /sim/reset`` without a runner resets and says it was not running;
* ``DELETE /graph/nodes/{name}`` removes every edge that touches the node;
* ``POST /graph/nodes`` answers each input class of the 0.4.0 release
  notes' status table with its documented status, a repeat included;
* ``PUT /graph/params`` replies with what a ``GET`` then reads, before the
  first compile and after;
* ``POST /checkpoint/load`` of a file it cannot read, or of another
  graph's checkpoint, is a 400 that names no parser internals and loads
  nothing (a file NumPy refuses as a pickle has its message echoed: a
  strict xfail);
* ``POST /checkpoint/save`` refused on the way leaves the earlier file of
  that name as it was, and says "nothing was written" -- a save whose final
  move fails has already written the manifest (a strict xfail);
* a JAX trace stops itself at its step budget; past its time budget it is
  stopped only by the next step it records (a strict xfail).

Nothing here can reach a cloud provider
(:func:`tests.property.differential.no_cloud_launch`).
"""

from __future__ import annotations

import os
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api import server as server_module
from maddening.api.server import MAX_NODE_PARAM_INT, SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode, HeatNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode
from tests.property.differential import no_cloud_launch

DT = 1.0 / 64.0
REGISTRY = {"BallNode": BallNode, "HeatNode": HeatNode, "TableNode": TableNode,
            "SpringDamperNode": SpringDamperNode}


@pytest.fixture(scope="module", autouse=True)
def _offline():
    with no_cloud_launch():
        yield


def _compiled(gm: GraphManager) -> GraphManager:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


def _served(gm: GraphManager, root=None) -> tuple[SimulationServer, TestClient]:
    server = SimulationServer(REGISTRY, graph_manager=gm,
                              checkpoint_root=None if root is None else str(root))
    return server, TestClient(server.create_app(), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# A step that raises
# ---------------------------------------------------------------------------

class _GuardedCounter(BallNode):
    """A ball moving at 1 per unit time that raises at run time once its
    position passes 0.5 (``equinox.error_if``): the 33rd step from 0."""

    def update(self, state, boundary_inputs, dt, *, params=None):
        out = super().update(state, boundary_inputs, dt, params=params)
        return {**out, "position": eqx.error_if(out["position"], out["position"] > 0.5,
                                                "counter passed 0.5")}


def _guarded(start: float) -> tuple[SimulationServer, TestClient]:
    gm = GraphManager()
    gm.add_node(_GuardedCounter("c", timestep=DT, initial_position=start,
                                initial_velocity=1.0, gravity=0.0))
    return _served(_compiled(gm))


def test_a_single_step_that_raises_is_a_400_and_stores_nothing():
    server, client = _guarded(start=0.5)
    resp = client.post("/sim/step")
    assert resp.status_code == 400, resp.text
    assert "counter passed 0.5" in resp.json()["detail"]
    assert client.get("/graph/state/c").json()["position"] == 0.5


def test_a_run_that_raises_at_its_first_step_is_a_400_and_moves_nothing():
    server, client = _guarded(start=0.5)
    resp = client.post("/sim/run", params={"n_steps": 10})
    assert resp.status_code == 400, resp.text
    assert "counter passed 0.5" in resp.json()["detail"]
    assert client.get("/graph/state/c").json()["position"] == 0.5
    assert server.relay.step_count == 0


def test_a_run_that_raises_part_way_says_how_many_steps_it_took(monkeypatch):
    """Slices that always double (1, 2, 4, ... steps): the 33rd step is the
    second of the sixth slice.  The 400 carries the 32 steps taken, the
    stored ones of the failing slice included, and the graph is after them."""
    monkeypatch.setattr(server_module, "_RUN_SLICE_SECONDS", 1e9)
    server, client = _guarded(start=0.0)
    resp = client.post("/sim/run", params={"n_steps": 100})
    assert resp.status_code == 400, resp.text
    body = resp.json()
    moved = client.get("/graph/state/c").json()["position"]
    assert "nothing was stepped" not in body["detail"], (moved, body)
    assert body["steps_run"] == 32 and body["n_steps"] == 100, body
    assert moved == 32 * DT == 0.5
    assert server.relay.step_count == 32


# ---------------------------------------------------------------------------
# The runner routes with no runner
# ---------------------------------------------------------------------------

def _ball_server(root=None) -> tuple[SimulationServer, TestClient]:
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=3.0, gravity=-1.0))
    return _served(_compiled(gm), root)


@pytest.mark.parametrize("route", ["/sim/stop", "/sim/pause", "/sim/resume"])
def test_a_runner_route_with_no_runner_is_a_409(route):
    server, client = _ball_server()
    resp = client.post(route)
    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == "Runner is not started."
    assert server.runner is None


def test_a_reset_with_no_runner_resets_and_says_it_was_not_running():
    server, client = _ball_server()
    assert client.post("/sim/run", params={"n_steps": 7}).status_code == 200
    resp = client.post("/sim/reset")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["was_running"] is False
    assert body["state"]["ball"] == {"position": 3.0, "velocity": 0.0}
    assert client.get("/graph/state/ball").json() == {"position": 3.0, "velocity": 0.0}


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_removing_a_node_removes_every_edge_that_touches_it():
    gm = GraphManager()
    gm.add_node(TableNode("table", timestep=0.01))
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=1.0))
    gm.add_node(BallNode("other", timestep=0.01, initial_position=2.0))
    gm.add_edge(source="table", target="ball", source_field="position",
                target_field="table_position")
    gm.add_edge(source="table", target="other", source_field="position",
                target_field="table_position")
    server, client = _served(_compiled(gm))
    assert len(server.gm._edges) == 2
    resp = client.delete("/graph/nodes/table")
    assert resp.status_code == 200, resp.text
    assert server.gm._edges == []
    assert "table" not in client.get("/graph/state").json()
    assert client.post("/graph/compile").status_code == 200
    assert client.post("/sim/step").status_code == 200


#: The 0.4.0 release notes' table of ``POST /graph/nodes`` input classes,
#: as (label, params of the request, the status, a fragment of the detail).
NODE_INPUTS = [
    ("slash in name", {"type": "BallNode", "name": "a/b"}, 400, "invalid"),
    ("hash in name", {"type": "BallNode", "name": "a#b"}, 400, "invalid"),
    ("arrow in name", {"type": "BallNode", "name": "a->b"}, 400, "invalid"),
    ("empty name", {"type": "BallNode", "name": ""}, 400, "invalid"),
    ("NaN name", {"type": "BallNode", "name": "NaN"}, 400, "NaN"),
    ("Infinity name", {"type": "BallNode", "name": "Infinity"}, 400, "Infinity"),
    ("-Infinity name", {"type": "BallNode", "name": "-Infinity"}, 400, "-Infinity"),
    ("unstable timestep", {"type": "HeatNode", "name": "h", "timestep": 10.0,
                           "params": {"n_cells": 8}}, 400, "Fourier"),
    ("non-finite constant", {"type": "BallNode", "name": "b",
                             "params": {"gravity": 10 ** 400}}, 400, "value must be finite"),
    ("dimension over the cap", {"type": "HeatNode", "name": "h2",
                                "params": {"n_cells": MAX_NODE_PARAM_INT + 1}}, 422, None),
]


@pytest.mark.parametrize("label, body, status, fragment", NODE_INPUTS,
                         ids=[row[0] for row in NODE_INPUTS])
def test_each_input_class_of_the_release_notes_table_gets_its_status_and_so_does_a_repeat(
        label, body, status, fragment):
    server, client = _ball_server()
    request = {"timestep": 0.01, "params": {}, **body}
    for attempt in ("first", "repeat"):
        resp = client.post("/graph/nodes", json=request)
        assert resp.status_code == status, (label, attempt, resp.status_code, resp.text)
        if fragment is not None:
            assert fragment in str(resp.json()["detail"]), (label, attempt, resp.text)
    assert list(server.gm._nodes) == ["ball"]


@pytest.mark.parametrize("compile_first", [False, True], ids=["before compile",
                                                              "after compile"])
def test_a_params_reply_is_what_a_get_then_reads(compile_first):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", timestep=0.01, initial_position=1.0))
    server, client = _served(_compiled(gm) if compile_first else gm)
    resp = client.put("/graph/params/spring", json={"params": {"stiffness": 42.0,
                                                               "damping": 3}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["params"] == client.get("/graph/params/spring").json()
    assert resp.json()["params"]["stiffness"] == 42.0
    assert resp.json()["params"]["damping"] == 3.0


# ---------------------------------------------------------------------------
# Checkpoints that do not fit
# ---------------------------------------------------------------------------

#: What NumPy's and zipfile's own messages say, which a 400 must not echo.
_INTERNALS = ("pickle", "allow_pickle", "zip", "Zip", "magic", "EOF", "unsafely")


def _names_no_internals(detail: str) -> bool:
    return not any(word in detail for word in _INTERNALS)


def test_a_checkpoint_that_is_not_an_archive_is_a_400_naming_no_internals(tmp_path):
    server, client = _ball_server(tmp_path)
    assert client.post("/sim/run", params={"n_steps": 3}).status_code == 200
    before = client.get("/graph/state").json()
    (tmp_path / "bad.npz").write_bytes(b"PK\x03\x04 this is not an archive")
    resp = client.post("/checkpoint/load", params={"path": "bad.npz"})
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("could not load checkpoint 'bad.npz'"), resp.text
    assert _names_no_internals(detail) and str(tmp_path) not in detail, detail
    assert client.get("/graph/state").json() == before


@pytest.mark.parametrize("content", [b"{}", b"", b"\x93NUMPY not really"],
                         ids=["json-text", "empty", "npy-magic"])
def test_a_checkpoint_numpy_refuses_as_a_pickle_is_a_400_naming_no_internals(tmp_path, content):
    """The release notes: "load errors no longer echo parser internals".
    A file that is not an ``.npz`` (``{}``) is refused by ``numpy.load``
    with a ``ValueError`` about pickled data, and the route answers every
    ``ValueError`` with its message -- the branch meant for the graph's
    own mismatch messages."""
    server, client = _ball_server(tmp_path)
    before = client.get("/graph/state").json()
    (tmp_path / "text.npz").write_bytes(content)
    resp = client.post("/checkpoint/load", params={"path": "text.npz"})
    assert resp.status_code == 400, resp.text
    assert client.get("/graph/state").json() == before
    detail = resp.json()["detail"]
    assert detail == ("could not load checkpoint 'text.npz': it is not an .npz archive of "
                      "plain arrays saved by this API"), detail
    assert _names_no_internals(detail)


def test_a_checkpoint_with_a_member_holding_python_objects_is_refused_by_name(tmp_path):
    """REST-095's neighbour: an ``.npz`` that is an archive, one of whose members NumPy
    could only read by unpickling.  Saved with ``allow_pickle=True``, as no MADDENING
    writer does."""
    server, client = _ball_server(tmp_path)
    before = client.get("/graph/state").json()
    np.savez(tmp_path / "objects.npz", **{"ball/position": np.array([{"x": 1}], dtype=object)})
    resp = client.post("/checkpoint/load", params={"path": "objects.npz"})
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("could not load checkpoint 'objects.npz': it is not an .npz")
    assert _names_no_internals(detail)
    assert client.get("/graph/state").json() == before


def test_a_checkpoint_of_another_graph_is_a_400_and_nothing_is_loaded(tmp_path):
    # Saved by a graph with a node of another name, and by one whose node
    # has a field of another shape.
    other = GraphManager()
    other.add_node(BallNode("elsewhere", timestep=0.01, initial_position=9.0))
    _compiled(other).save_state(str(tmp_path / "other_nodes.npz"))
    reshaped = GraphManager()
    reshaped.add_node(HeatNode("ball", timestep=0.01, n_cells=8))
    _compiled(reshaped).save_state(str(tmp_path / "other_fields.npz"))

    server, client = _ball_server(tmp_path)
    assert client.post("/sim/run", params={"n_steps": 3}).status_code == 200
    before = client.get("/graph/state").json()
    params_before = client.get("/graph/params/ball").json()
    for name, fragment in (("other_nodes.npz", "node mismatch"),
                           ("other_fields.npz", "Field mismatch")):
        resp = client.post("/checkpoint/load", params={"path": name})
        assert resp.status_code == 400, resp.text
        assert fragment in resp.json()["detail"], resp.text
        assert client.get("/graph/state").json() == before
        assert client.get("/graph/params/ball").json() == params_before
    assert np.isfinite(before["ball"]["position"])


# ---------------------------------------------------------------------------
# Shutdown signals
# ---------------------------------------------------------------------------

def test_the_shutdown_signals_are_chained_only_from_the_main_thread_over_a_python_handler():
    """``_chain_shutdown_signals``: SIGINT and SIGTERM call
    ``request_shutdown`` ahead of the handler already installed -- only from
    the main thread, and only over a Python handler (a default or ignored
    signal is left alone)."""
    import signal
    import threading

    server, _client = _ball_server()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    seen: list = []
    try:
        signal.signal(signal.SIGINT, lambda signum, frame: seen.append(signum))
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        restore = server._chain_shutdown_signals()
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
        chained = signal.getsignal(signal.SIGINT)
        chained(signal.SIGINT, None)
        assert server._shutdown.is_set() and seen == [signal.SIGINT]
        restore()
        assert signal.getsignal(signal.SIGINT) is not chained
        # From another thread nothing is installed.
        server._shutdown.clear()
        installed = signal.getsignal(signal.SIGINT)
        out: list = []
        worker = threading.Thread(target=lambda: out.append(server._chain_shutdown_signals()))
        worker.start()
        worker.join(10)
        assert signal.getsignal(signal.SIGINT) is installed
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
        out[0]()
        assert signal.getsignal(signal.SIGINT) is installed
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def test_the_checkpoint_root_defaults_to_a_checkpoints_directory_under_the_working_one(
        tmp_path, monkeypatch):
    """The release notes: paths are relative to
    ``SimulationServer(checkpoint_root=)`` (default ``./checkpoints``)."""
    monkeypatch.chdir(tmp_path)
    server, client = _ball_server()
    assert server.checkpoint_root == (tmp_path / "checkpoints").resolve()
    resp = client.post("/checkpoint/save", params={"path": "c.npz"})
    assert resp.status_code == 200, resp.text
    assert (tmp_path / "checkpoints" / "c.npz").is_file()
    assert (tmp_path / "checkpoints" / "c.npz.manifest.json").is_file()


def test_a_stride_call_that_names_one_value_keeps_the_other():
    """REST-099: "A value left out of the query keeps its current value" -- each way
    round, and a call naming neither changes nothing.  Until 0.4.0 the one left out was
    reset to 1."""
    server, client = _ball_server()
    resp = client.put("/sim/stride", params={"steps_per_frame": 5, "relay_stride": 3})
    assert resp.json() == {"steps_per_frame": 5, "relay_stride": 3}
    resp = client.put("/sim/stride", params={"steps_per_frame": 7})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"steps_per_frame": 7, "relay_stride": 3}
    assert server.relay.stride == 3
    resp = client.put("/sim/stride", params={"relay_stride": 2})
    assert resp.json() == {"steps_per_frame": 7, "relay_stride": 2}
    assert server._steps_per_frame == 7
    assert client.put("/sim/stride").json() == {"steps_per_frame": 7, "relay_stride": 2}
    for bad in ({"steps_per_frame": 0}, {"relay_stride": 0}):
        assert client.put("/sim/stride", params=bad).status_code == 422


# ---------------------------------------------------------------------------
# A save refused on the way, and a JAX trace's budgets
# ---------------------------------------------------------------------------

def test_a_save_refused_at_its_final_move_writes_nothing(tmp_path):
    """The REST guide: the checkpoint "is written under a temporary name,
    the manifest next, and the file moved into place last: a save refused
    on the way (400, 'nothing was written') leaves any earlier file of that
    name as it was".  A name that is a directory -- one a save of
    ``d.npz/inner.npz`` made through the API -- refuses the final move,
    after the manifest beside it was written."""
    server, client = _ball_server(tmp_path)
    assert client.post("/checkpoint/save", params={"path": "d.npz/inner.npz"}).status_code == 200
    before = sorted(p.name for p in tmp_path.iterdir())
    resp = client.post("/checkpoint/save", params={"path": "d.npz"})
    assert resp.status_code == 400, resp.text
    assert "nothing was written" in resp.json()["detail"]
    assert str(tmp_path) not in resp.json()["detail"], resp.text
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_a_save_whose_manifest_name_is_taken_by_a_directory_writes_nothing(tmp_path):
    """REST-102: the manifest's name, not the checkpoint's, is the directory -- an earlier
    checkpoint of that name stays as it was, manifest included."""
    server, client = _ball_server(tmp_path)
    assert client.post("/checkpoint/save", params={"path": "c.npz"}).status_code == 200
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    (tmp_path / "c.npz.manifest.json").unlink()
    (tmp_path / "c.npz.manifest.json").mkdir()
    del before["c.npz.manifest.json"]
    assert client.post("/sim/step").status_code == 200
    resp = client.post("/checkpoint/save", params={"path": "c.npz"})
    assert resp.status_code == 400, resp.text
    assert "nothing was written" in resp.json()["detail"]
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before


def test_a_save_whose_manifest_cannot_be_moved_says_the_checkpoint_was_written(
        tmp_path, monkeypatch):
    """REST-102: the one failure left after the checkpoint is moved into place -- its
    manifest's own move -- is answered as what it is, and the checkpoint loads (with
    sim_time counted from zero, the manifest not being there to restore it)."""
    server, client = _ball_server(tmp_path)
    real = server_module.os.replace

    def second_move_fails(src, dst):          # the manifest's move into place
        if str(dst) == str(tmp_path / "c.npz.manifest.json"):
            raise PermissionError(13, "Permission denied", str(dst))
        return real(src, dst)

    monkeypatch.setattr(server_module.os, "replace", second_move_fails)
    resp = client.post("/checkpoint/save", params={"path": "c.npz"})
    monkeypatch.setattr(server_module.os, "replace", real)
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("checkpoint 'c.npz' was written, but its manifest could not be")
    assert str(tmp_path) not in detail, detail
    assert (tmp_path / "c.npz").is_file() and not (tmp_path / "c.npz.manifest.json").exists()
    assert not [p for p in tmp_path.iterdir() if "partial" in p.name]
    loaded = client.post("/checkpoint/load", params={"path": "c.npz"})
    assert loaded.status_code == 200 and loaded.json()["sim_time_from_checkpoint"] is False


@pytest.mark.parametrize("name", ["../outside.npz", "a\x00b.npz", "x" * 300 + ".npz",
                                  "d.npz"], ids=["outside", "nul", "too-long", "directory"])
def test_no_checkpoint_refusal_names_an_absolute_server_path(tmp_path, name):
    """REST-102: a 4xx detail names the file as the client named it, never the server's
    absolute paths (the checkpoint root's used to be in the refusal of a path outside
    it, and an OSError's text in the others)."""
    server, client = _ball_server(tmp_path)
    (tmp_path / "d.npz").mkdir()
    for route in ("/checkpoint/save", "/checkpoint/load"):
        resp = client.post(route, params={"path": name})
        assert 400 <= resp.status_code < 500, resp.text
        detail = resp.json()["detail"]
        assert str(tmp_path) not in detail and os.getcwd() not in detail, (route, detail)


@pytest.fixture
def traced_client():
    from maddening.core.simulation import profiler

    server, client = _ball_server()
    assert client.post("/sim/step").status_code == 200          # compiled before tracing
    yield server, client
    if profiler.jax_trace_active():
        profiler.stop_jax_trace()


def test_a_trace_records_exactly_its_step_budget(monkeypatch, traced_client):
    """At a budget of 4 steps: the 4th step stops the trace, and a run that
    goes on is not recorded past it."""
    monkeypatch.setattr(server_module, "MAX_JAX_TRACE_STEPS", 4)
    server, client = traced_client
    assert client.post("/sim/profile/jax/start").status_code == 200
    assert client.post("/sim/run", params={"n_steps": 3}).status_code == 200
    assert client.get("/sim/profile/jax/status").json()["active"] is True
    assert client.post("/sim/step").status_code == 200
    status = client.get("/sim/profile/jax/status").json()
    assert status["active"] is False and status["steps"] == 4, status
    assert client.post("/sim/run", params={"n_steps": 5}).status_code == 200
    assert client.get("/sim/profile/jax/status").json()["steps"] == 4


def test_a_trace_past_its_time_budget_is_stopped_without_waiting_for_a_step(
        monkeypatch, traced_client):
    """The REST guide: a trace runs "for at most MAX_JAX_TRACE_STEPS (10 000)
    steps or MAX_JAX_TRACE_SECONDS (600 s): past either it stops itself and
    writes its files".  The time is read only when a step is recorded, so a
    trace left with the simulation idle runs on.  At a budget of 0 s."""
    monkeypatch.setattr(server_module, "MAX_JAX_TRACE_SECONDS", 0.0)
    server, client = traced_client
    assert client.post("/sim/profile/jax/start").status_code == 200
    time.sleep(0.05)
    status = client.get("/sim/profile/jax/status").json()
    assert status["active"] is False, status
    assert status["stopped_by"] == "its time budget, 0 s", status


def test_the_time_budget_stops_a_trace_with_no_request_at_all(monkeypatch, traced_client):
    """REST-104: the budget has its own clock -- a timer set when the trace starts -- so
    the trace stops with nothing stepping and nothing asking."""
    from maddening.core.simulation import profiler

    monkeypatch.setattr(server_module, "MAX_JAX_TRACE_SECONDS", 0.1)
    server, client = traced_client
    assert client.post("/sim/profile/jax/start").status_code == 200
    deadline = time.monotonic() + 10.0
    while profiler.jax_trace_active() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not profiler.jax_trace_active()
    assert server._trace_stopped_by == "its time budget, 0.1 s"
    assert server._last_jax_trace_dir is not None


def test_a_status_read_stops_a_trace_whose_timer_has_not_fired(monkeypatch, traced_client):
    """REST-104: the status, stop and step routes read the clock too, so a late timer
    does not leave a trace reported active past its budget."""
    class Late:                               # a timer that never fires
        def __init__(self, *a, **k):
            self.daemon = True

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(server_module.threading, "Timer", Late)
    monkeypatch.setattr(server_module, "MAX_JAX_TRACE_SECONDS", 0.0)
    server, client = traced_client
    assert client.post("/sim/profile/jax/start").status_code == 200
    status = client.get("/sim/profile/jax/status").json()
    assert status["active"] is False and status["stopped_by"] == "its time budget, 0 s"
    stop = client.post("/sim/profile/jax/stop")
    assert stop.status_code == 409 and "stopped itself" in stop.json()["detail"]
    assert "/" not in stop.json()["detail"].split("GET")[0], stop.json()["detail"]


def test_a_timer_left_from_an_earlier_trace_stops_nothing(monkeypatch, traced_client):
    """REST-104: a trace stopped by a request, then a new one with a longer budget: the
    first trace's timer, firing later, leaves the second alone."""
    from maddening.core.simulation import profiler

    server, client = traced_client
    monkeypatch.setattr(server_module, "MAX_JAX_TRACE_SECONDS", 0.2)
    assert client.post("/sim/profile/jax/start").status_code == 200
    first = server._trace_timer
    assert client.post("/sim/profile/jax/stop").status_code == 200
    monkeypatch.setattr(server_module, "MAX_JAX_TRACE_SECONDS", 30.0)
    assert client.post("/sim/profile/jax/start").status_code == 200
    first.function(*first.args)                # what the first timer would do on firing
    assert profiler.jax_trace_active()
    assert client.get("/sim/profile/jax/status").json()["active"] is True


# ---------------------------------------------------------------------------
# The small end of a float32, on both write routes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [1e-50, -1e-50, 1e-46])
def test_a_state_value_float32_flushes_to_zero_is_refused_and_nothing_is_written(value):
    """The REST twin of the FMU's value check, on ``PUT /graph/state``: a
    non-zero value its type stores as 0 is refused in the overflow's words."""
    server, client = _ball_server()
    before = client.get("/graph/state/ball").json()
    resp = client.put("/graph/state/ball", json={"state": {"position": value, "velocity": 0.0}})
    assert resp.status_code == 400, resp.text
    assert "does not fit its type float32" in resp.json()["detail"]
    assert client.get("/graph/state/ball").json() == before


@pytest.mark.parametrize("value", [1e-45, -1e-40, 1e-38])
def test_a_value_that_rounds_to_a_subnormal_keeps_its_sign_and_magnitude_on_both_routes(value):
    server, client = _ball_server()
    want = float(np.float32(value))
    resp = client.put("/graph/state/ball", json={"state": {"position": value, "velocity": 0.0}})
    assert resp.status_code == 200, resp.text
    assert client.get("/graph/state/ball").json()["position"] == want != 0.0
    resp = client.put("/graph/params/ball", json={"params": {"gravity": value}})
    assert resp.status_code == 200, resp.text
    assert client.get("/graph/params/ball").json()["gravity"] == want != 0.0
