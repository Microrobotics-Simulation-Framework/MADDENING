"""Each request bound of the REST guide admits its limit and refuses one past it.

``tests/api/test_request_bounds.py`` and
``tests/api/test_request_memory_is_bounded.py`` check a value well over
each bound and a shipped value well under it.  The rows of
``docs/validation/rest_runpod_claims.yaml`` state the bounds as "at most",
so these check the boundary itself: the limit is accepted and the limit
plus one is refused, for

* the request body (``MAX_REQUEST_BODY_BYTES``), declared and streamed;
* ``POST /sim/run?n_steps=`` (``MAX_RUN_STEPS``), with the bound read
  when the app is built;
* an integer parameter's magnitude (``MAX_NODE_PARAM_INT``) and the values
  in one request's params (``MAX_NODE_PARAM_ELEMENTS``), on both request
  models;
* the whole graph's state (``MAX_GRAPH_STATE_ELEMENTS``).

The bounds that allocate are exercised at a small value
(``monkeypatch``), so no test allocates what the shipped limits admit.
Nothing here can reach a cloud provider
(:func:`tests.property.differential.no_cloud_launch`).
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from maddening.api import server as server_module
from maddening.api.server import (
    MAX_GRAPH_STATE_ELEMENTS,
    MAX_NODE_PARAM_ELEMENTS,
    MAX_NODE_PARAM_INT,
    AddNodeRequest,
    SetNodeParamsRequest,
    SimulationServer,
    _graph_budget_refusal,
)
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from tests.property.differential import no_cloud_launch

REGISTRY = {"SpringDamperNode": SpringDamperNode}


@pytest.fixture(scope="module", autouse=True)
def _offline():
    with no_cloud_launch():
        yield


def _client() -> tuple[SimulationServer, TestClient]:
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="spring", timestep=0.01, initial_position=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer(REGISTRY, graph_manager=gm)
    return server, TestClient(server.create_app(), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# The body
# ---------------------------------------------------------------------------

LIMIT = 1000


@pytest.mark.parametrize("streamed", [False, True], ids=["declared", "streamed"])
def test_a_body_of_exactly_the_limit_is_read_and_one_byte_more_is_a_413(monkeypatch,
                                                                        streamed):
    """At the limit the body reaches the route (an unparsable body is the
    route's 422, not the limit's 413); one byte more never does."""
    monkeypatch.setattr(server_module, "MAX_REQUEST_BODY_BYTES", LIMIT)
    server, client = _client()
    before = client.get("/graph/state/spring").json()
    for size, want in ((LIMIT, 422), (LIMIT + 1, 413)):
        body = b"x" * size
        content = (iter([body[: size // 2], body[size // 2:]]) if streamed else body)
        resp = client.put("/graph/state/spring", content=content,
                          headers={"Content-Type": "application/json"})
        assert resp.status_code == want, (size, resp.status_code, resp.text)
    assert client.get("/graph/state/spring").json() == before


# ---------------------------------------------------------------------------
# POST /sim/run
# ---------------------------------------------------------------------------

def test_a_run_of_exactly_the_step_bound_runs_and_one_more_is_a_422(monkeypatch):
    """``Query(le=MAX_RUN_STEPS)`` is read when the app is built, so the
    bound is exercised at 12 steps, not 100 000."""
    monkeypatch.setattr(server_module, "MAX_RUN_STEPS", 12)
    server, client = _client()
    resp = client.post("/sim/run", params={"n_steps": 12})
    assert resp.status_code == 200, resp.text
    assert server.relay.step_count == 12
    resp = client.post("/sim/run", params={"n_steps": 13})
    assert resp.status_code == 422 and "n_steps" in resp.text
    assert server.relay.step_count == 12


# ---------------------------------------------------------------------------
# Integer parameters and value counts, on both request models
# ---------------------------------------------------------------------------

MODELS = {
    "post": lambda params: AddNodeRequest(type="SpringDamperNode", name="n", timestep=0.01,
                                          params=params),
    "put": lambda params: SetNodeParamsRequest(params=params),
}


@pytest.mark.parametrize("model", sorted(MODELS))
@pytest.mark.parametrize("sign", [1, -1])
def test_an_integer_of_exactly_the_bound_is_taken_and_one_more_is_refused(model, sign):
    make = MODELS[model]
    assert make({"n": sign * MAX_NODE_PARAM_INT}).params["n"] == sign * MAX_NODE_PARAM_INT
    assert make({"n": {"deep": [sign * MAX_NODE_PARAM_INT]}})
    with pytest.raises(ValidationError, match="integer magnitude"):
        make({"n": sign * (MAX_NODE_PARAM_INT + 1)})
    with pytest.raises(ValidationError, match="integer magnitude"):
        make({"n": {"deep": [0, sign * (MAX_NODE_PARAM_INT + 1)]}})
    # A float is a constant, not a dimension; a boolean is not a number.
    assert make({"n": float(10 * MAX_NODE_PARAM_INT), "b": True})


@pytest.mark.parametrize("model", sorted(MODELS))
def test_exactly_the_value_bound_is_taken_and_one_value_more_is_refused(model):
    make = MODELS[model]
    assert make({"v": [0.0] * MAX_NODE_PARAM_ELEMENTS})
    assert make({"v": [0.0] * (MAX_NODE_PARAM_ELEMENTS - 1), "w": 1.0})
    with pytest.raises(ValidationError, match="values in total"):
        make({"v": [0.0] * MAX_NODE_PARAM_ELEMENTS, "w": 1.0})


def test_the_integer_bound_is_a_422_on_both_routes():
    server, client = _client()
    over = MAX_NODE_PARAM_INT + 1
    resp = client.post("/graph/nodes", json={"type": "SpringDamperNode", "name": "n2",
                                             "timestep": 0.01, "params": {"mass": over}})
    assert resp.status_code == 422, resp.text
    resp = client.put("/graph/params/spring", json={"params": {"mass": over}})
    assert resp.status_code == 422, resp.text
    assert "n2" not in server.gm._nodes


# ---------------------------------------------------------------------------
# The whole graph's state
# ---------------------------------------------------------------------------

def test_the_graph_may_hold_exactly_its_state_budget_and_not_one_element_more():
    assert _graph_budget_refusal("n", MAX_GRAPH_STATE_ELEMENTS - 7, 7) is None
    assert _graph_budget_refusal("n", 0, MAX_GRAPH_STATE_ELEMENTS) is None
    refusal = _graph_budget_refusal("n", MAX_GRAPH_STATE_ELEMENTS - 7, 8)
    assert refusal is not None and str(MAX_GRAPH_STATE_ELEMENTS) in refusal


def test_a_node_that_fills_the_graph_budget_exactly_is_added_and_one_more_element_is_refused(
        monkeypatch):
    """Through the route, at a budget of 6: the graph holds the spring's 2
    elements, two more 2-element springs fill it exactly, and a third is
    refused with nothing added."""
    monkeypatch.setattr(server_module, "MAX_GRAPH_STATE_ELEMENTS", 6)
    server, client = _client()
    for name in ("a", "b"):
        resp = client.post("/graph/nodes", json={"type": "SpringDamperNode", "name": name,
                                                 "timestep": 0.01, "params": {}})
        assert resp.status_code == 201, resp.text
    resp = client.post("/graph/nodes", json={"type": "SpringDamperNode", "name": "c",
                                             "timestep": 0.01, "params": {}})
    assert resp.status_code == 400 and "whole graph" in resp.json()["detail"], resp.text
    assert set(server.gm._nodes) == {"spring", "a", "b"}
