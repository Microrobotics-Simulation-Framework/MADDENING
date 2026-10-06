"""A REST write that fails, at any point, leaves the graph exactly as it was.

Every route that can change the graph runs inside one transaction
(``SimulationServer._graph_transaction``): the graph is recorded when the
lock is taken and put back when the route raises -- a refusal of its own,
or anything it did not expect, which is answered 500 with a generic detail.
Each route used to keep the promise by itself, validating before it wrote
or undoing what it had written, and a failure after its first write left
what was written: a node added before the reply could not be encoded, a
parameter moved before the trace that refused it.

Here each route is sent a request that succeeds, with a failure injected at
every point of its body in turn (``tests/property/injected_failures.py``:
before and after each step that mutates the graph, publishes to the streams
or writes the reply).  After each failure the graph must be what it was --
by what a client can observe and object for object
(``tests/property/graph_fingerprint.py``) -- and then, with no failure, the
request must do what it says.  The sequence oracle
(``tests/property/test_rest_write_sequences_leave_a_graph_that_reloads.py``)
asks the same of whatever sequence it generates; this module is the fixed
list, one route at a time, with the experimental surrogate routes and the
runner's start, which the oracle does not drive.

Nothing here can reach a cloud provider: no request is sent to ``/cloud/*``
and the module runs under ``no_cloud_launch``.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode, HeatNode, SpringDamperNode

from tests.property import rest_oracle as O
from tests.property.differential import no_cloud_launch, quiet
from tests.property.graph_fingerprint import assert_exactly_as_it_was, served_fingerprint
from tests.property.injected_failures import (
    InjectedFailure,
    Injector,
    fail_at_every_point,
    injection_points,
)

DT = 0.01
N_CELLS = 8
ALPHA = 0.0625 * (1.0 / N_CELLS) ** 2 / DT      # Fourier number 1/16, float32-exact
EDGE = {"source_node": "ball", "source_field": "position",
        "target_node": "spring", "target_field": "anchor_position"}


@pytest.fixture(scope="module", autouse=True)
def _points():
    with no_cloud_launch(), injection_points():
        yield


def _graph() -> GraphManager:
    gm = GraphManager()
    gm.add_node(HeatNode("rod", DT, n_cells=N_CELLS, length=1.0,
                         thermal_diffusivity=ALPHA, initial_temperature=1.0))
    gm.add_node(SpringDamperNode("spring", DT, stiffness=40.0, damping=0.5,
                                 initial_position=0.5))
    gm.add_node(BallNode("ball", DT, initial_position=3.0))
    gm.add_edge("spring", "ball", "position", "table_position")
    with quiet():
        gm.compile()
        gm.step()
    return gm


@pytest.fixture
def served():
    s = O.serve(_graph())
    assert s.client.post("/checkpoint/save", params={"path": "start.npz"}).status_code == 200
    with quiet():
        yield s
    s.close()


def _steps_run(resp) -> int:
    try:
        body = resp.json()
    except ValueError:
        return 0
    return int(body.get("steps_run") or 0) if isinstance(body, dict) else 0


def _position(served, node="spring") -> float:
    return float(np.asarray(served.gm.get_node_state(node)["position"]))


# ---------------------------------------------------------------------------
# One route at a time
# ---------------------------------------------------------------------------
# name: (prepare, send, the status of the request itself, steps a failure
# must have been injected around, what the request did)

def _dirty(served):
    """Leave the graph waiting for a compile, so the route's own compile is
    one of the points."""
    assert served.client.put("/graph/params/rod",
                             json={"params": {"stencil_order": 4}}).status_code == 200
    assert served.gm._dirty  # noqa: SLF001


def _moved(served):
    assert served.client.post("/sim/run", params={"n_steps": 2}).status_code == 200
    assert served.client.put("/graph/params/spring",
                             json={"params": {"stiffness": 20.0}}).status_code == 200


def _with_edge(served):
    assert served.client.post("/graph/edges", json=EDGE).status_code == 201


ROUTES = {
    "add a node": (
        None,
        lambda c: c.post("/graph/nodes", json={
            "type": "HeatNode", "name": "extra", "timestep": DT,
            "params": {"n_cells": N_CELLS, "thermal_diffusivity": ALPHA}}),
        201, ("GraphManager.add_node", "SimulationServer._publish_state", "_json_reply"),
        lambda s: "extra" in s.gm._nodes),  # noqa: SLF001
    "remove a node": (
        None, lambda c: c.delete("/graph/nodes/rod"),
        200, ("GraphManager.remove_node", "StateRelay.restore"),
        lambda s: "rod" not in s.gm._nodes),  # noqa: SLF001
    "add an edge": (
        None, lambda c: c.post("/graph/edges", json=EDGE),
        201, ("GraphManager.add_edge",),
        lambda s: len(s.gm._edges) == 2),  # noqa: SLF001
    "remove an edge": (
        _with_edge, lambda c: c.request("DELETE", "/graph/edges", json=EDGE),
        200, ("GraphManager.remove_edge",),
        lambda s: len(s.gm._edges) == 1),  # noqa: SLF001
    "compile": (
        _dirty, lambda c: c.post("/graph/compile"),
        200, ("GraphManager.compile",),
        lambda s: not s.gm._dirty),  # noqa: SLF001
    "write a state": (
        None, lambda c: c.put("/graph/state/ball",
                              json={"state": {"position": 2.0, "velocity": 0.5}}),
        200, ("GraphManager.set_node_state", "SimulationServer._publish_state"),
        lambda s: _position(s, "ball") == 2.0),
    "write live params": (
        None, lambda c: c.put("/graph/params/spring",
                              json={"params": {"stiffness": 20.0, "damping": 1.0}}),
        200, ("_ParamsDict.__setitem__", "_json_reply"),
        lambda s: float(s.gm.params["nodes"]["spring"]["damping"]) == 1.0
        and s.gm._nodes["spring"].node.params["stiffness"] == 20.0),  # noqa: SLF001
    "write a structural param with a live one": (
        None, lambda c: c.put("/graph/params/rod", json={
            "params": {"thermal_diffusivity": ALPHA / 2, "stencil_order": 4}}),
        200, ("_ParamsDict.__setitem__", "_json_reply"),
        lambda s: s.gm._nodes["rod"].node.params["stencil_order"] == 4  # noqa: SLF001
        and s.gm._dirty),  # noqa: SLF001
    "save a checkpoint of a graph waiting for a compile": (
        _dirty, lambda c: c.post("/checkpoint/save", params={"path": "later.npz"}),
        200, ("GraphManager.save_state",),
        lambda s: (s.root / "later.npz").is_file()),
    "load a checkpoint": (
        _moved, lambda c: c.post("/checkpoint/load", params={"path": "start.npz"}),
        200, ("GraphManager.load_state", "GraphManager.set_node_state",
              "_restore_state_and_params", "StateRelay.restore", "_json_reply"),
        lambda s: float(s.gm.params["nodes"]["spring"]["stiffness"]) == 40.0),
    "load a checkpoint into a graph waiting for a compile": (
        _dirty, lambda c: c.post("/checkpoint/load", params={"path": "start.npz"}),
        200, ("GraphManager.compile", "GraphManager.load_state"),
        lambda s: not s.gm._dirty),  # noqa: SLF001
    "step": (
        None, lambda c: c.post("/sim/step"),
        200, ("GraphManager.step", "GraphManager._store_state", "_json_reply"),
        lambda s: s.server.relay.step_count == 1),
    "step a graph waiting for a compile": (
        _dirty, lambda c: c.post("/sim/step"),
        200, ("GraphManager.compile", "GraphManager._store_state"),
        lambda s: not s.gm._dirty),  # noqa: SLF001
    "run": (
        None, lambda c: c.post("/sim/run", params={"n_steps": 3}),
        200, ("GraphManager.run", "GraphManager._store_state", "SimulationServer._state_json"),
        lambda s: s.server.relay.step_count >= 3),
    "run no steps on a graph waiting for a compile": (
        _dirty, lambda c: c.post("/sim/run", params={"n_steps": 0}),
        200, ("GraphManager.run", "GraphManager.compile"),
        lambda s: not s.gm._dirty),  # noqa: SLF001
    "reset": (
        _moved, lambda c: c.post("/sim/reset"),
        200, ("GraphManager.reset_state", "StateRelay.reset", "StateRelay.restore",
              "_json_reply"),
        lambda s: _position(s) == 0.5 and s.server.relay.step_count == 0),
    "profile": (
        _moved, lambda c: c.post("/sim/profile", params={"n_steps": 2, "n_warmup": 0}),
        200, ("GraphManager.step", "_restore_state_and_params"),
        lambda s: s.server.relay.step_count == 2),
}


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_a_failure_at_any_point_of_a_write_route_changes_nothing(served, route):
    prepare, send, status, steps, done = ROUTES[route]
    if prepare is not None:
        prepare(served)
    resp, tried, _, _ = fail_at_every_point(
        served, lambda: send(served.client), route, partial=_steps_run)
    assert resp.status_code == status, resp.text
    for step in steps:
        # Before the step and after it: a failure on either side of every
        # mutation the route makes.
        assert any(f"before {step}" in point for point in tried), (step, tried)
        assert any(f"after {step}" in point for point in tried), (step, tried)
    assert done(served), f"{route} answered {status} and did not do it"


def test_every_route_that_takes_the_graph_to_write_is_in_the_list_or_named_here():
    """The list above against the server's source: each use of
    ``_graph_transaction`` is a route this module (or the sequence oracle)
    drives with injected failures, and the only uses of the plain
    ``_graph_access`` left are reads and the one route that stays outside
    (``POST /surrogate/train``: it starts a job and changes nothing of the
    graph).  A write route added with the plain lock fails here."""
    import inspect
    import re

    import maddening.api.server as server_module

    source = inspect.getsource(server_module.SimulationServer)
    transactions = set(re.findall(r'_graph_transaction\("([^"]+)"', source))
    plain = set(re.findall(r'_graph_access\("([^"]+)"', source))
    assert transactions == {
        "add a node", "remove a node", "add an edge", "remove an edge",
        "compile the graph", "write a node's state", "write a node's params",
        "save a checkpoint", "load a checkpoint", "step the graph", "run the graph",
        "read the run's final state", "start the runner", "reset the graph",
        "activate a surrogate", "deactivate a surrogate", "profile the graph"}
    assert plain == {"read the graph", "validate the graph", "read the state",
                     "read a node's params", "start a surrogate job"}


# ---------------------------------------------------------------------------
# The reply of an unexpected failure
# ---------------------------------------------------------------------------

def test_an_unexpected_failure_is_a_500_that_names_nothing_and_is_logged(served, caplog):
    """The 500 says the request failed and the graph was put back, and
    nothing of what failed; the traceback is in the log."""
    injector = Injector(served)
    before = served_fingerprint(served)
    injector.arm(2)        # after the node joined the graph and before it is published
    with caplog.at_level(logging.ERROR, logger="maddening.api.server"):
        try:
            resp = served.client.post("/graph/nodes", json={
                "type": "BallNode", "name": "second", "timestep": DT, "params": {}})
        finally:
            fired = injector.disarm()
    assert fired is not None and "_publish_state" in fired
    assert resp.status_code == 500
    assert resp.json() == {"detail": (
        "The request failed unexpectedly (the server's log says how). The graph was "
        "put back exactly as it was before the request.")}
    assert "second" not in served.gm._nodes  # noqa: SLF001
    assert_exactly_as_it_was(before, served_fingerprint(served), "the failed POST /graph/nodes")
    logged = [r for r in caplog.records if r.exc_info and r.exc_info[0] is InjectedFailure]
    assert logged and "add a node" in logged[0].getMessage()
    # And the name is free: the same request, with no failure, is taken.
    assert served.client.post("/graph/nodes", json={
        "type": "BallNode", "name": "second", "timestep": DT,
        "params": {}}).status_code == 201


def test_a_failure_outside_a_write_route_is_the_same_generic_500(served):
    """A read that fails unexpectedly has no transaction (it changes
    nothing); its 500 is JSON with the generic detail, where Starlette's
    default is a plain-text body."""
    injector = Injector(served)
    injector.arm(0)
    try:
        resp = served.client.get("/graph/params/spring")
    finally:
        fired = injector.disarm()
    assert fired is not None and "_json_reply" in fired
    assert resp.status_code == 500
    assert resp.json() == {
        "detail": "The request failed unexpectedly (the server's log says how)."}


def test_a_refusal_of_the_routes_own_is_answered_as_it_was_and_changes_nothing(served):
    """The transaction puts the graph back and lets a refusal through
    unchanged: the status and detail are the route's."""
    before = served_fingerprint(served)
    resp = served.client.put("/graph/params/spring", json={"params": {"stiffness": -1.0}})
    assert resp.status_code == 400 and "stiffness" in resp.json()["detail"]
    resp = served.client.post("/graph/nodes", json={
        "type": "BallNode", "name": "ball", "timestep": DT, "params": {}})
    assert resp.status_code == 409
    resp = served.client.post("/checkpoint/load", params={"path": "never-saved.npz"})
    assert resp.status_code == 404
    assert_exactly_as_it_was(before, served_fingerprint(served), "three refusals")


def test_a_refused_step_of_an_edited_graph_leaves_it_waiting_for_its_compile(served):
    """A step the configuration does not allow is a 400 and nothing was
    stepped -- nor compiled: the graph is the edited one, to the object,
    not one half-way into a compile."""
    assert served.client.post("/graph/edges", json={
        "source_node": "rod", "source_field": "temperature",
        "target_node": "ball", "target_field": "table_position"}).status_code == 201
    before = served_fingerprint(served)
    for url in ("/sim/step", "/graph/compile"):
        resp = served.client.post(url)
        assert resp.status_code == 400, resp.text
        assert_exactly_as_it_was(before, served_fingerprint(served), f"the refused POST {url}")


# ---------------------------------------------------------------------------
# POST /sim/run: the steps before the failure stay, and the reply says so
# ---------------------------------------------------------------------------

def test_a_run_that_fails_unexpectedly_says_how_many_steps_it_took(served):
    """A run steps in slices, releasing the graph between them (a ``PUT
    /graph/params`` is taken there), so its transaction is a slice: a
    failure in the second slice puts that slice back and leaves the first,
    and the 500 carries ``steps_run``.  The graph is then the one a twin
    reaches in that many steps."""
    twin = _graph()
    with quiet():
        twin.step()
    injector = Injector(served)
    # Points 0-1: the relay; 2-5: the first slice (one step); 6: before the
    # second slice's run; 7: before its first store.
    injector.arm(8)
    try:
        resp = served.client.post("/sim/run", params={"n_steps": 3})
    finally:
        fired = injector.disarm()
    assert fired is not None and "after GraphManager._store_state" in fired, fired
    assert resp.status_code == 500
    body = resp.json()
    assert body["steps_run"] == 1 and body["n_steps"] == 3
    assert "failed unexpectedly" in body["detail"] and "after 1 of the run's 3" in body["detail"]
    assert served.server.relay.step_count == 1
    for name in ("rod", "spring", "ball"):
        for field, value in twin.get_node_state(name).items():
            assert np.array_equal(np.asarray(served.gm.get_node_state(name)[field]),
                                  np.asarray(value)), (name, field)
    # A later write is not refused as "a run is in progress".
    assert served.client.post("/sim/step").status_code == 200


def test_a_params_write_between_two_slices_of_a_run_survives_the_runs_failure(served):
    """Why a run's transaction is not the whole run: a parameter written
    between two slices is another request's accepted write, and a failure
    in a later slice must not take it back."""
    gm = served.gm
    original = type(gm).run
    calls = []

    def run(self, n, **kwargs):
        if self is gm:
            calls.append(n)
            if len(calls) == 2:
                # As PUT /graph/params does between two slices, under the lock.
                self.params["nodes"]["spring"]["damping"] = jnp.asarray(2.0, jnp.float32)
                self._nodes["spring"].node.params["damping"] = 2.0  # noqa: SLF001
                self.params
            if len(calls) == 3:
                raise OSError("the third slice fails")
        return original(self, n, **kwargs)

    type(gm).run = run
    try:
        resp = served.client.post("/sim/run", params={"n_steps": 7})
    finally:
        type(gm).run = original
    assert resp.status_code == 500 and resp.json()["steps_run"] == 3, resp.text
    assert "OSError" not in resp.text and "third slice" not in resp.text
    assert float(gm.params["nodes"]["spring"]["damping"]) == 2.0


# ---------------------------------------------------------------------------
# The runner's start
# ---------------------------------------------------------------------------

def test_a_start_that_fails_leaves_the_graph_and_starts_no_runner(served):
    """``POST /sim/start`` compiles a graph waiting for one, under the
    lock, before the runner's thread exists: a failure there leaves the
    graph as it was and no runner started."""
    _dirty(served)
    resp, tried, _, _ = fail_at_every_point(
        served, lambda: served.client.post("/sim/start"), "POST /sim/start",
        on_failure=lambda fired, failed: _not_started(served, fired))
    try:
        assert resp.status_code == 200, resp.text
        assert any("after GraphManager.compile" in point for point in tried), tried
        # The relay observes the graph the runner steps, although the first
        # start created the runner and its failure detached the relay.
        assert served.server.relay._on_event in served.gm._observers  # noqa: SLF001
    finally:
        assert served.client.post("/sim/stop").status_code == 200


def _not_started(served, fired):
    assert not served.server._runner_started, fired  # noqa: SLF001
    assert served.server.runner is None or not served.server.runner.is_alive, fired


# ---------------------------------------------------------------------------
# The surrogate routes (experimental)
# ---------------------------------------------------------------------------

class _TrainedRod:
    """What ``POST /surrogate/activate`` reads of a finished job's result."""

    state_spec = {"temperature": (N_CELLS,)}

    def to_node(self, name, timestep, initial_values):
        return HeatNode(name, timestep, n_cells=N_CELLS, length=1.0,
                        thermal_diffusivity=ALPHA / 4,
                        initial_temperature=np.asarray(initial_values["temperature"]).tolist())


def test_a_failure_at_any_point_of_a_surrogate_activation_or_revert_changes_nothing(served):
    """Activation had no undo of its own: a failure after the node was
    replaced left the surrogate in the graph and the server not knowing it
    was active.  The transaction covers the graph and the server's record
    of the surrogates (``_original_nodes``, ``_active_surrogates``)."""
    server = served.server
    server._surrogate_jobs["job"] = {  # noqa: SLF001
        "status": "done", "node_name": "rod", "result": _TrainedRod()}
    original = served.gm._nodes["rod"].node  # noqa: SLF001

    resp, tried, _, _ = fail_at_every_point(
        served, lambda: served.client.post("/surrogate/activate/job"), "activate")
    assert resp.status_code == 200, resp.text
    for step in ("replace_node", "GraphManager.compile", "GraphManager.reset_state"):
        assert any(f"after {step}" in point for point in tried), (step, tried)
    assert served.gm._nodes["rod"].node is not original  # noqa: SLF001
    assert server._active_surrogates == {"rod"}  # noqa: SLF001

    resp, tried, _, _ = fail_at_every_point(
        served, lambda: served.client.post("/surrogate/deactivate/rod"), "deactivate")
    assert resp.status_code == 200, resp.text
    for step in ("GraphManager.remove_node", "GraphManager.add_node", "GraphManager.compile",
                 "GraphManager.reset_state"):
        assert any(f"after {step}" in point for point in tried), (step, tried)
    assert served.gm._nodes["rod"].node is original  # noqa: SLF001
    assert not server._active_surrogates and not server._original_nodes  # noqa: SLF001
