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
* an integer parameter's magnitude (``MAX_NODE_PARAM_INT``), on both
  routes, for a parameter that is an integer (an integral JSON number for
  a float parameter is a float), and the values in one request's params
  (``MAX_NODE_PARAM_ELEMENTS``), on both request models;
* a new node's ``ParamSpec`` bounds, as ``PUT /graph/params`` applies them;
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

import jax.numpy as jnp
import pytest
from tests._loopback_client import LoopbackTestClient as TestClient
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
from maddening.core.node import SimulationNode
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


class Counted(SimulationNode):
    """Integer, integer-list and float parameters, none of which allocates:
    the integer bound is exercised at its value without building anything
    of that size."""

    def __init__(self, name, timestep, count: int = 3, dims: tuple = (2, 2),
                 gain: float = 1.0, label: str = "x", flag: bool = False):
        super().__init__(name, timestep, count=count, dims=list(dims), gain=gain,
                         label=label, flag=flag)

    def initial_state(self):
        return {"x": jnp.zeros((), jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] + self.params["gain"] * dt}


def _counted_client() -> tuple[SimulationServer, TestClient]:
    gm = GraphManager()
    gm.add_node(Counted("c", 0.01))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({"Counted": Counted, "SpringDamperNode": SpringDamperNode},
                              graph_manager=gm)
    return server, TestClient(server.create_app(), raise_server_exceptions=False)


@pytest.mark.parametrize("model", sorted(MODELS))
def test_the_request_models_bound_the_value_count_not_an_integers_magnitude(model):
    """A request model knows no parameter's type, and a JSON number does
    not say whether it is an integer, so the models bound only the number
    of values; the routes bound an integer where its parameter is one."""
    make = MODELS[model]
    assert make({"n": 10 * MAX_NODE_PARAM_INT}).params["n"] == 10 * MAX_NODE_PARAM_INT
    assert make({"n": {"deep": [-10 * MAX_NODE_PARAM_INT]}})


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("key, shape", [("count", lambda v: v), ("dims", lambda v: [2, v])])
def test_an_integer_parameter_takes_exactly_the_bound_and_refuses_one_more(sign, key, shape):
    """``POST /graph/nodes`` and ``PUT /graph/params``, top-level and
    inside a list: the bound itself is taken and one more is a 422 that
    adds or writes nothing.  An integral float for an integer parameter is
    the integer it spells, and bounded the same."""
    server, client = _counted_client()
    at, over = sign * MAX_NODE_PARAM_INT, sign * (MAX_NODE_PARAM_INT + 1)
    resp = client.post("/graph/nodes", json={"type": "Counted", "name": "a", "timestep": 0.01,
                                             "params": {key: shape(at)}})
    assert resp.status_code == 201, resp.text
    for i, value in enumerate((over, float(over))):
        resp = client.post("/graph/nodes", json={"type": "Counted", "name": f"b{i}",
                                                 "timestep": 0.01, "params": {key: shape(value)}})
        assert resp.status_code == 422 and "integer magnitude must be at most" in resp.text, \
            resp.text
        assert f"b{i}" not in server.gm._nodes
        resp = client.put("/graph/params/c", json={"params": {key: shape(value)}})
        assert resp.status_code == 422 and "integer magnitude must be at most" in resp.text, \
            resp.text
        assert server.gm.get_node("c").params[key] == ([2, 2] if key == "dims" else 3)


@pytest.mark.parametrize("body", ['{"params": {"gain": 20000000}}',
                                  '{"params": {"gain": 20000000.0}}',
                                  '{"params": {"gain": 2e7}}'])
def test_an_integral_number_for_a_float_parameter_is_a_float_on_both_routes(body):
    """A browser's ``JSON.stringify(2e7)`` is ``20000000``: one JSON number
    for a float parameter, written however the client spells it.  The
    integer bound refused it with a 422 on both routes (it is a constant,
    not a dimension)."""
    server, client = _counted_client()
    resp = client.put("/graph/params/c", content=body,
                      headers={"content-type": "application/json"})
    assert resp.status_code == 200, resp.text
    assert server.gm.get_node("c").params["gain"] == 2e7
    assert type(server.gm.get_node("c").params["gain"]) is float
    post = body.replace('{"params"', '{"type": "Counted", "name": "n", "timestep": 0.01, "params"')
    resp = client.post("/graph/nodes", content=post,
                       headers={"content-type": "application/json"})
    assert resp.status_code == 201, resp.text
    assert server.gm.get_node("n").params["gain"] == 2e7


def test_a_float_leaf_takes_an_integral_json_number_past_the_integer_bound():
    """The auditor's case: ``SpringDamperNode.stiffness`` (a params-pytree
    leaf) of 2e7 N/m, sent as ``20000000``."""
    server, client = _client()
    resp = client.put("/graph/params/spring", content='{"params": {"stiffness": 20000000}}',
                      headers={"content-type": "application/json"})
    assert resp.status_code == 200, resp.text
    assert float(server.gm.params["nodes"]["spring"]["stiffness"]) == 2e7
    resp = client.post("/graph/nodes", content=(
        '{"type": "SpringDamperNode", "name": "s2", "timestep": 0.01, '
        '"params": {"stiffness": 20000000}}'), headers={"content-type": "application/json"})
    assert resp.status_code == 201, resp.text


@pytest.mark.parametrize("model", sorted(MODELS))
def test_exactly_the_value_bound_is_taken_and_one_value_more_is_refused(model):
    make = MODELS[model]
    assert make({"v": [0.0] * MAX_NODE_PARAM_ELEMENTS})
    assert make({"v": [0.0] * (MAX_NODE_PARAM_ELEMENTS - 1), "w": 1.0})
    with pytest.raises(ValidationError, match="values in total"):
        make({"v": [0.0] * MAX_NODE_PARAM_ELEMENTS, "w": 1.0})


def test_the_integer_bound_is_a_422_on_both_routes():
    """For an integer parameter (``mass`` is a float, and takes any
    integral number since the bound reads the parameter's type)."""
    server, client = _counted_client()
    over = MAX_NODE_PARAM_INT + 1
    resp = client.post("/graph/nodes", json={"type": "Counted", "name": "n2",
                                             "timestep": 0.01, "params": {"count": over}})
    assert resp.status_code == 422, resp.text
    resp = client.put("/graph/params/c", json={"params": {"count": over}})
    assert resp.status_code == 422, resp.text
    assert "n2" not in server.gm._nodes


def test_a_parameter_whose_default_says_nothing_is_bounded_as_written():
    """Fail closed: a parameter with no default (or ``None``) could be a
    dimension, so an integer for it is bounded."""
    server, client = _counted_client()
    resp = client.post("/graph/nodes", json={"type": "Counted", "name": "n3", "timestep": 0.01,
                                             "params": {"extra": MAX_NODE_PARAM_INT + 1}})
    assert resp.status_code == 422, resp.text


@pytest.mark.parametrize("params, refusal", [
    ({"damping": -5.0}, "damping=-5.0 below bound 0.0"),
    ({"stiffness": 0.0}, "stiffness"),
    ({"mass": -1.0}, "mass"),
])
def test_a_new_node_outside_its_param_spec_bounds_is_refused_as_a_params_write_is(params,
                                                                                  refusal):
    """``POST /graph/nodes`` applied no ``ParamSpec`` bounds: a spring with
    a damping of -5 was added, which ``PUT /graph/params`` refuses to
    write, and a checkpoint of the graph then carried the value.  Both
    routes refuse it now, naming the bound, and nothing is added."""
    server, client = _client()
    resp = client.post("/graph/nodes", json={"type": "SpringDamperNode", "name": "n4",
                                             "timestep": 0.01, "params": params})
    assert resp.status_code == 400, resp.text
    assert refusal in resp.json()["detail"]
    assert "n4" not in server.gm._nodes
    put = client.put("/graph/params/spring", json={"params": params})
    assert put.status_code == 400 and refusal in put.json()["detail"], put.text
    ok = client.post("/graph/nodes", json={"type": "SpringDamperNode", "name": "n5",
                                           "timestep": 0.01, "params": {"damping": 0.0}})
    assert ok.status_code == 201, ok.text


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


@pytest.mark.parametrize("route, key, value", [
    ("put", "gain", "1.5"), ("put", "count", "3"), ("put", "dims", ["2", 2]),
    ("post", "gain", "1.5"), ("post", "count", " 3 "), ("post", "dims", [2, "2"]),
])
def test_text_for_a_numeric_parameter_is_refused_on_both_routes(route, key, value):
    """N1: ``PUT /graph/params`` stored the string ``"1.5"`` as 1.5 (NumPy
    parses it), where every FMU door refuses a string; ``POST /graph/nodes``
    handed it to the constructor.  Both refuse text for a numeric parameter
    now, and a text parameter still takes text."""
    server, client = _counted_client()
    if route == "put":
        resp = client.put("/graph/params/c", json={"params": {key: value}})
        assert server.gm.get_node("c").params[key] == {"gain": 1.0, "count": 3,
                                                       "dims": [2, 2]}[key]
    else:
        resp = client.post("/graph/nodes", json={"type": "Counted", "name": "n",
                                                 "timestep": 0.01, "params": {key: value}})
        assert "n" not in server.gm._nodes
    assert resp.status_code == 400 and "expected a number, got a string" in resp.text, resp.text
    ok = client.post("/graph/nodes", json={"type": "Counted", "name": "t", "timestep": 0.01,
                                           "params": {"label": "y"}})
    assert ok.status_code == 201, ok.text


def test_a_numeric_string_for_a_float_leaf_is_refused_and_nothing_is_written():
    """N1 on a params-pytree leaf (the oracle's case)."""
    server, client = _client()
    before = float(server.gm.params["nodes"]["spring"]["stiffness"])
    resp = client.put("/graph/params/spring", json={"params": {"stiffness": "1.5"}})
    assert resp.status_code == 400 and "expected a number, got a string" in resp.text
    assert float(server.gm.params["nodes"]["spring"]["stiffness"]) == before


@pytest.mark.parametrize("key, value", [("gain", True), ("count", False), ("dims", [2, True])])
def test_a_boolean_for_a_numeric_parameter_is_refused_on_construction(key, value):
    """N3: ``POST /graph/nodes`` built a node with ``damping: true`` (201),
    and the value dropped out of the params pytree, where ``PUT`` refuses a
    boolean for a numeric parameter.  A boolean parameter takes one."""
    server, client = _counted_client()
    resp = client.post("/graph/nodes", json={"type": "Counted", "name": "n", "timestep": 0.01,
                                             "params": {key: value}})
    assert resp.status_code == 400 and "expected a number, got a boolean" in resp.text, resp.text
    assert "n" not in server.gm._nodes
    spring = client.post("/graph/nodes", json={"type": "SpringDamperNode", "name": "s2",
                                               "timestep": 0.01, "params": {"damping": True}})
    assert spring.status_code == 400 and "got a boolean" in spring.text
    ok = client.post("/graph/nodes", json={"type": "Counted", "name": "b", "timestep": 0.01,
                                           "params": {"flag": True}})
    assert ok.status_code == 201, ok.text


@pytest.mark.parametrize("value", [1e-50, -1e-50, 1e39])
def test_a_value_the_leaf_cannot_hold_is_refused_on_construction_as_on_a_write(value):
    """``POST /graph/nodes`` built a spring with ``damping: 1e-50`` as 0.0
    (float32 flushes it) and ``1e39`` as an infinity, where ``PUT`` refuses
    both as "does not fit its type"; the route refuses them too now, and
    builds nothing.  The smallest normal float32 still builds."""
    server, client = _client()
    put = client.put("/graph/params/spring", json={"params": {"damping": value}})
    assert put.status_code == 400 and "does not fit its type" in put.text, put.text
    resp = client.post("/graph/nodes", json={"type": "SpringDamperNode", "name": "s2",
                                             "timestep": 0.01, "params": {"damping": value}})
    assert resp.status_code == 400 and "does not fit its type" in resp.text, resp.text
    assert "s2" not in server.gm._nodes
    ok = client.post("/graph/nodes", json={"type": "SpringDamperNode", "name": "s3",
                                           "timestep": 0.01, "params": {"damping": 1.2e-38}})
    assert ok.status_code == 201, ok.text
