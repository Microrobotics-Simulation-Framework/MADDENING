"""A 422 echoes the value it refused only when that value is small, and
lists a bounded number of errors.

FastAPI's validation error carries the refused value as ``input``, and for
a body that is the body, or a field of it as large as the body.  The app's
handler encoded the errors whole: a ``PUT /graph/params`` 16 bytes under
``MAX_REQUEST_BODY_BYTES`` (11.2 million ``[],``) was answered, after 22 s,
with a 422 of 32 MiB, encoded on the event loop.  A missing field echoes
the whole body the same way, on every route that takes one.

A refused value of at most ``_MAX_ECHOED_INPUT_CHARS`` characters of JSON
is echoed as before (a number, a name, the text ``NaN``, a short body).  A
larger one is left out, the error says so under ``input_omitted``, and a
reply lists at most ``_MAX_ERRORS_LISTED`` errors, since one error per
element of a typed list would echo a body element by element.

No request here goes to a path under ``/cloud``, ``/surrogate`` or ``/ws``:
the body routes are read from the app with those filtered out first, and
the one typed list of the request models, which is a surrogate route's, is
held to the rule through the model and the handler, without a request.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import ValidationError

from maddening.api import server as server_module
from maddening.api.server import (
    _MAX_ECHOED_INPUT_CHARS,
    _MAX_ERRORS_LISTED,
    SimulationServer,
    TrainSurrogateRequest,
    _echoed_input,
    _small_enough_to_echo,
    _validation_detail,
)
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode
from tests._loopback_client import LoopbackTestClient

LIMIT = _MAX_ECHOED_INPUT_CHARS
NOTE = f"the refused value is longer than {LIMIT} characters as JSON and is not echoed"
#: A 422 of this app: at most this many bytes, whatever the body was.  64
#: errors of a kilobyte of echo each, and their ``loc`` and ``msg``.
REPLY_BOUND = 128 * 1024
NOT_REQUESTED = ("/cloud", "/surrogate", "/ws")


class Unwalkable(list):
    """A list that fails the test if anything visits its elements."""

    def __iter__(self):
        raise AssertionError("a list too long to echo was walked")


class UnwalkableDict(dict):
    def items(self):
        raise AssertionError("an object too large to echo was walked")


def _written(value) -> str:
    """*value* as a reply writes it."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------- the rule


def test_the_bounds_are_the_documented_ones():
    assert (_MAX_ECHOED_INPUT_CHARS, _MAX_ERRORS_LISTED) == (1024, 64)


@pytest.mark.parametrize("value", [
    0, -1.5, True, None, "abc", "NaN", float("nan"), [], {}, [1, 2, 3],
    {"type": "BallNode", "name": "b", "timestep": 0.01, "params": {"x": [0.5] * 20}},
    "x" * (LIMIT - 2), [0] * ((LIMIT - 2 + 1) // 2),
])
def test_a_small_value_is_echoed_as_it_was(value):
    error = {"type": "t", "loc": ("body", "f"), "msg": "m", "input": value, "ctx": {"gt": 0}}
    echoed = _echoed_input(error)
    assert list(echoed) == ["type", "loc", "msg", "input", "ctx"]
    assert _written(echoed["input"]) == _written(value) or value != value
    assert len(_written(echoed["input"])) <= LIMIT
    assert echoed["loc"] == ["body", "f"] and echoed["ctx"] == {"gt": 0}


@pytest.mark.parametrize("value", [
    "x" * (LIMIT - 1),                       # one character over, with its quotes
    [0] * (LIMIT // 2 + 1),
    [[]] * 400,                              # 400 * 3 + 1 characters
    {"k": "v" * LIMIT},
    {str(i): i for i in range(300)},
    "\n" * 600,                              # 600 characters, 1202 as JSON
    "\u0001" * 200,                          # 200 characters, 1202 as JSON
    10 ** 2000,                              # an integer of 2001 digits
    10 ** 5000,                              # one Python does not print
    [[[]] * 20] * 20,
], ids=["text_one_over", "numbers", "empty_lists", "one_long_field", "many_keys",
        "newlines", "control_characters", "a_long_integer", "an_unprintable_integer",
        "lists_of_lists"])
def test_a_value_over_the_limit_is_left_out_and_the_error_says_so(value):
    error = {"type": "t", "loc": ("body",), "msg": "m", "input": value, "ctx": {"gt": 0}}
    echoed = _echoed_input(error)
    assert list(echoed) == ["type", "loc", "msg", "input_omitted", "ctx"]
    assert echoed["input_omitted"] == NOTE
    assert len(_written(echoed)) < 300


def test_the_limit_is_of_the_json_text_to_the_character():
    def kept(value):
        return "input" in _echoed_input({"type": "t", "loc": (), "msg": "m", "input": value})

    assert kept("x" * (LIMIT - 2)) and not kept("x" * (LIMIT - 1))
    assert kept([0] * (LIMIT // 2)) is (len(_written([0] * (LIMIT // 2))) <= LIMIT)
    # [0,0,...,0] of n zeros is 2n + 1 characters
    assert kept([0] * ((LIMIT - 1) // 2)) and not kept([0] * ((LIMIT - 1) // 2 + 1))


def test_a_value_too_long_to_echo_is_not_walked():
    """What made the refusal cost what the body weighs: 11 million elements
    encoded to be sent back.  A container longer than the characters left
    is refused by its length."""
    assert not _small_enough_to_echo(Unwalkable([[]] * 2000), LIMIT)
    assert not _small_enough_to_echo({"params": {"x": Unwalkable([0] * 2000)}}, LIMIT)
    assert not _small_enough_to_echo(UnwalkableDict((str(i), i) for i in range(2000)), LIMIT)
    echoed = _echoed_input({"type": "t", "loc": (), "msg": "m",
                            "input": {"params": {"x": Unwalkable([[]] * 2000)}}})
    assert echoed["input_omitted"] == NOTE and "input" not in echoed


def test_the_walk_stops_within_the_limit_whatever_the_value_holds():
    visited = []

    class Counted(str):
        def __len__(self):
            visited.append(1)
            return 1

    assert not _small_enough_to_echo([Counted("a") for _ in range(LIMIT)] * 1, LIMIT)
    assert len(visited) <= LIMIT
    visited.clear()
    nested: list = []
    for _ in range(5000):
        nested = [nested]
    assert not _small_enough_to_echo(nested, LIMIT)


def test_a_value_nested_deeper_than_it_can_be_encoded_is_left_out_or_echoed_whole():
    nested: list = []
    for _ in range(400):                     # 802 characters, 400 levels
        nested = [nested]
    echoed = _echoed_input({"type": "t", "loc": (), "msg": "m", "input": nested})
    assert ("input" in echoed) != ("input_omitted" in echoed)
    assert len(_written(echoed)) <= LIMIT + 300


def test_an_error_without_an_input_and_an_error_that_is_no_dict_are_held_to_the_rule():
    assert _echoed_input({"type": "t", "loc": ("query", "n"), "msg": "m"}) == {
        "type": "t", "loc": ["query", "n"], "msg": "m"}
    assert _echoed_input("a text") == "a text"
    assert _echoed_input(["x" * LIMIT]) == {"type": "error_not_echoed", "loc": [], "msg": NOTE}


def test_a_reply_lists_a_bounded_number_of_errors_and_counts_the_rest():
    errors = [{"type": "t", "loc": ("body", "hidden_sizes", i), "msg": "m", "input": "a"}
              for i in range(_MAX_ERRORS_LISTED + 1)]
    assert len(_validation_detail(errors[:-1])) == _MAX_ERRORS_LISTED
    detail = _validation_detail(errors)
    assert len(detail) == _MAX_ERRORS_LISTED + 1
    assert [e["loc"][-1] for e in detail[:-1]] == list(range(_MAX_ERRORS_LISTED))
    assert detail[-1] == {
        "type": "errors_not_listed", "loc": [],
        "msg": f"1 more errors are not listed: a reply lists the first {_MAX_ERRORS_LISTED} "
               f"of {_MAX_ERRORS_LISTED + 1}"}
    many = _validation_detail(errors * 1000)
    assert len(many) == _MAX_ERRORS_LISTED + 1
    assert many[-1]["msg"].startswith(f"{len(errors) * 1000 - _MAX_ERRORS_LISTED} more errors")


# ----------------------------------------------------------------- the app


@pytest.fixture(scope="module")
def app():
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=3.0))
    return SimulationServer({"BallNode": BallNode}, graph_manager=gm).create_app()


@pytest.fixture(scope="module")
def client(app):
    with LoopbackTestClient(app, raise_server_exceptions=False) as client:
        yield client


def _body_routes(app):
    """``(method, url)`` of every route of the app that takes a JSON body,
    with the paths this file must not request filtered out first."""
    routes = [route for route in app.routes if isinstance(route, APIRoute)
              and not route.path.startswith(NOT_REQUESTED)]
    assert not [route.path for route in routes if route.path.startswith(NOT_REQUESTED)]
    return sorted((method, route.path.replace("{node_name}", "ball").replace("{name}", "ball"))
                  for route in routes if route.body_field is not None
                  for method in route.methods)


#: Bodies no request model takes, each far over the echo's limit: a list
#: where an object is expected, an object with one long field and none of
#: the model's own, an object of many keys, and a long text.
LARGE_BODIES = {
    "a_list_of_empty_lists": [[]] * 50_000,
    "one_long_field": {"junk": "x" * 150_000},
    "many_keys": {f"k{i}": i for i in range(20_000)},
    "a_long_text": "y" * 150_000,
    "numbers": [0.5] * 50_000,
}


def test_the_body_routes_are_read_from_the_app_and_none_is_one_this_file_must_not_request(app):
    routes = _body_routes(app)
    assert {("POST", "/graph/nodes"), ("POST", "/graph/edges"), ("DELETE", "/graph/edges"),
            ("PUT", "/graph/state/ball"), ("PUT", "/graph/params/ball")} <= set(routes)
    assert not [url for _method, url in routes if url.startswith(NOT_REQUESTED)]


@pytest.mark.parametrize("shape", sorted(LARGE_BODIES))
def test_no_body_route_answers_a_refused_body_with_the_body(app, client, shape):
    body = json.dumps(LARGE_BODIES[shape]).encode()
    assert len(body) > 100 * LIMIT
    before = client.get("/graph").json()
    for method, url in _body_routes(app):
        assert not url.startswith(NOT_REQUESTED)
        resp = client.request(method, url, content=body,
                              headers={"content-type": "application/json"})
        assert resp.status_code == 422, (method, url, resp.status_code, resp.text[:200])
        assert len(resp.content) < REPLY_BOUND, (method, url, len(resp.content))
        detail = resp.json()["detail"]
        assert detail and all("input" not in error or len(_written(error["input"])) <= LIMIT
                              for error in detail), (method, url)
        assert any(error.get("input_omitted") == NOTE for error in detail), (method, url)
    assert client.get("/graph").json() == before


def test_a_params_body_refused_for_its_element_count_is_not_sent_back(client, monkeypatch):
    """The reported request at a size a test can afford: more lists than
    the element cap admits (the cap lowered, as the body-limit tests lower
    theirs), refused by the model's own validator, whose error carries
    the whole ``params`` object."""
    monkeypatch.setattr(server_module, "MAX_NODE_PARAM_ELEMENTS", 1000)
    body = ('{"params": {"elasticity": [' + "[]," * 300_000 + "0]}}").encode()
    resp = client.put("/graph/params/ball", content=body,
                      headers={"content-type": "application/json"})
    assert resp.status_code == 422, resp.text[:200]
    (error,) = resp.json()["detail"]
    assert error["loc"] == ["body", "params"]
    assert "at most 1000 lists and objects in total" in error["msg"]
    assert error["input_omitted"] == NOTE and "input" not in error
    assert len(resp.content) < 1024 < len(body) // 800


def test_a_small_refused_value_is_still_echoed_by_every_kind_of_parameter(client):
    """What a caller reads a 422 for: the value, where it is one."""
    resp = client.put("/graph/params/ball", json={"params": 5})
    assert resp.status_code == 422
    assert [error["input"] for error in resp.json()["detail"]] == [5]
    resp = client.post("/graph/nodes", json={"type": "BallNode", "name": "n"})
    assert resp.status_code == 422
    (error,) = resp.json()["detail"]
    assert error["type"] == "missing" and error["loc"] == ["body", "timestep"]
    assert error["input"] == {"type": "BallNode", "name": "n"}
    resp = client.post("/sim/run?n_steps=abc")
    assert resp.status_code == 422
    assert [error["input"] for error in resp.json()["detail"]] == ["abc"]


def test_a_long_refused_query_value_is_not_echoed(client):
    resp = client.post("/sim/run?n_steps=" + "9" * 4000)
    assert resp.status_code == 422
    (error,) = resp.json()["detail"]
    assert error["loc"] == ["query", "n_steps"] and error["input_omitted"] == NOTE
    assert "input" not in error and len(resp.content) < 1024


def test_the_one_typed_list_of_the_request_models_is_not_echoed_either(app):
    """A surrogate route's ``hidden_sizes``.  Its errors are made here by
    the model and handed to the app's handler, so the rule is held without
    a request to that route."""
    with pytest.raises(ValidationError) as refused:
        TrainSurrogateRequest.model_validate({"node_name": "ball", "hidden_sizes": ["a"] * 5000})
    errors = [dict(error, loc=("body", *error["loc"]))
              for error in refused.value.errors(include_url=False)]
    assert any(len(_written(error["input"])) > 10 * LIMIT for error in errors)
    handler = app.exception_handlers[RequestValidationError]
    reply = asyncio.run(handler(None, RequestValidationError(errors)))
    assert reply.status_code == 422 and len(reply.body) < REPLY_BOUND
    detail = json.loads(reply.body)["detail"]
    assert len(detail) == min(len(errors), _MAX_ERRORS_LISTED + 1)
    assert all("input" not in error or len(_written(error["input"])) <= LIMIT
               for error in detail)
    assert any(error.get("input_omitted") == NOTE for error in detail)


def test_the_handler_cuts_a_list_of_errors_however_it_came_about(app):
    """No request model of the app can make more errors than it has fields
    today (its one typed list is bounded at 16 items); a model that could
    would echo a body one element per error.  The handler is held to the
    bound on errors it is handed."""
    errors = [{"type": "int_type", "loc": ("body", "sizes", i), "msg": "Input should be a "
               "valid integer", "input": "a"} for i in range(70_000)]
    handler = app.exception_handlers[RequestValidationError]
    reply = asyncio.run(handler(None, RequestValidationError(errors)))
    assert reply.status_code == 422 and len(reply.body) < REPLY_BOUND
    detail = json.loads(reply.body)["detail"]
    assert len(detail) == _MAX_ERRORS_LISTED + 1
    assert [error["input"] for error in detail[:-1]] == ["a"] * _MAX_ERRORS_LISTED
    assert detail[-1]["type"] == "errors_not_listed"
    assert f"of {len(errors)}" in detail[-1]["msg"]
