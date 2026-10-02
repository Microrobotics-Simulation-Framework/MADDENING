"""Graph, simulation and checkpoint routes answer what they did, at the edges
of what the REST guide and the route docstrings say.

Rows of ``docs/validation/rest_runpod_claims.yaml`` this module pins:

* ``POST /sim/step`` and ``POST /sim/run`` answer a graph that cannot step
  with a 400 naming why; a single step that raises stores nothing, and a
  run that raises at its first step moves nothing.  A run that raises
  part-way has moved the graph by every step before, and still says
  "nothing was stepped" (a strict xfail);
* the runner routes are a 409 when there is no runner to act on;
* ``POST /sim/reset`` without a runner resets and says it was not running;
* ``DELETE /graph/nodes/{name}`` removes every edge that touches the node;
* ``POST /graph/nodes`` answers each input class of the 0.4.0 release
  notes' status table with its documented status, a repeat included;
* ``PUT /graph/params`` replies with what a ``GET`` then reads, before the
  first compile and after;
* ``POST /checkpoint/load`` of a file it cannot read, or of another
  graph's checkpoint, is a 400 that names no parser internals and loads
  nothing.

Nothing here can reach a cloud provider
(:func:`tests.property.differential.no_cloud_launch`).
"""

from __future__ import annotations

import os
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


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="REST-058: a run that raises part-way says nothing was stepped "
                          "after stepping; pending fix")
def test_a_run_that_raises_part_way_does_not_say_nothing_was_stepped(monkeypatch):
    """Slices that always double (1, 2, 4, ... steps): the 33rd step is
    inside the sixth slice, after 31 steps were stored."""
    monkeypatch.setattr(server_module, "_RUN_SLICE_SECONDS", 1e9)
    server, client = _guarded(start=0.0)
    resp = client.post("/sim/run", params={"n_steps": 100})
    assert resp.status_code == 400, resp.text
    moved = client.get("/graph/state/c").json()["position"]
    assert moved > 0.0
    assert "nothing was stepped" not in resp.json()["detail"], (moved, resp.json())


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

def test_a_checkpoint_that_is_not_an_archive_is_a_400_naming_no_internals(tmp_path):
    server, client = _ball_server(tmp_path)
    assert client.post("/sim/run", params={"n_steps": 3}).status_code == 200
    before = client.get("/graph/state").json()
    (tmp_path / "bad.npz").write_bytes(b"PK\x03\x04 this is not an archive")
    resp = client.post("/checkpoint/load", params={"path": "bad.npz"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "could not load checkpoint 'bad.npz'", resp.text
    assert client.get("/graph/state").json() == before


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="REST-095: a file NumPy refuses with a ValueError has NumPy's "
                          "message echoed in the 400; pending fix")
def test_a_checkpoint_numpy_refuses_as_a_pickle_is_a_400_naming_no_internals(tmp_path):
    """The release notes: "load errors no longer echo parser internals".
    A file that is not an ``.npz`` (``{}``) is refused by ``numpy.load``
    with a ``ValueError`` about pickled data, and the route answers every
    ``ValueError`` with its message -- the branch meant for the graph's
    own mismatch messages."""
    server, client = _ball_server(tmp_path)
    before = client.get("/graph/state").json()
    (tmp_path / "text.npz").write_text("{}")
    resp = client.post("/checkpoint/load", params={"path": "text.npz"})
    assert resp.status_code == 400, resp.text
    assert client.get("/graph/state").json() == before
    assert resp.json()["detail"] == "could not load checkpoint 'text.npz'", resp.text


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


def test_a_stride_call_that_names_one_value_sets_the_other_to_one():
    """``PUT /sim/stride`` sets both values on every call; one left out is
    set to its default, 1 (REST-099 is ``ambiguous``: no document says
    what an omitted value means)."""
    server, client = _ball_server()
    resp = client.put("/sim/stride", params={"steps_per_frame": 5, "relay_stride": 3})
    assert resp.json() == {"steps_per_frame": 5, "relay_stride": 3}
    resp = client.put("/sim/stride", params={"steps_per_frame": 7})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"steps_per_frame": 7, "relay_stride": 1}
    assert server.relay.stride == 1
