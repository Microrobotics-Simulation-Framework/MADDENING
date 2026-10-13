"""Every JSON reply of the app is written by one encoder, and it cannot fail
on what a request can put in a reply.

A refusal echoes what it refused.  Three encoders stood between that echo
and the wire, and each raised for something a caller can send, inside the
code that was writing the refusal -- so the refusal became a 500:

* Starlette's ``JSONResponse`` refuses a non-finite float, and a string
  that holds a surrogate (it has no UTF-8 bytes): ``{"type": "\\ud800"}``
  on ``POST /graph/nodes`` was a 500 where ``"abc"`` was a 400;
* the handler written for the first of those encoded the 422 with the
  encoder of *data* (``_json_reply``), which refuses the *text* ``NaN``,
  ``Infinity`` and ``-Infinity``: ``POST /sim/run?n_steps=NaN`` -- what a
  browser sends for ``parseInt("")`` -- was a 500 where ``n_steps=abc`` was
  a 422.

``_Reply`` is the response class of every route, of every refusal the app
builds itself, and of the handlers of ``RequestValidationError`` and
``HTTPException`` (Starlette's own 404 and 405 with them).  The doors the
request battery met are cases of
``tests/property/test_rest_requests_generated_from_the_schema.py``; this
file holds the encoder to its contract and the app to using no other.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException

from maddening.api import server as server_module
from maddening.api.server import _UNWRITABLE_REPLY, SimulationServer, _Reply
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode
from tests._loopback_client import LoopbackTestClient

NAN, INF = float("nan"), float("inf")


def strict(body: bytes):
    """*body* as a strict parser reads it: UTF-8, and no bare ``NaN``."""
    def refuse(token):
        raise AssertionError(f"the reply holds the bare token {token}")
    return json.loads(body.decode("utf-8"), parse_constant=refuse)


# ---------------------------------------------------------------------------
# The encoder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("content", [
    {"detail": "Not Found"}, {"a": [1, 2.5, True, None, "x"], "b": {"c": "é名😀"}},
    [], {}, "text", 3, None, {"NaN": "nan", "n": 1e308}, [[[]]], {"k": "a\x00b\n"},
], ids=repr)
def test_a_reply_starlette_can_encode_is_encoded_as_starlette_does(content):
    """Byte for byte: nothing about an ordinary reply changed."""
    assert _Reply(content=content).body == JSONResponse(content=content).body


class _Unknown:
    def __str__(self) -> str:
        return "an object"


@pytest.mark.parametrize("content, written", [
    ({"v": NAN}, {"v": "NaN"}),
    ([INF, -INF, 1.5], ["Infinity", "-Infinity", 1.5]),
    # The text of a token stays the text it was, beside a float that is one.
    ({"input": "NaN", "also": [NAN, "Infinity", "-Infinity"]},
     {"input": "NaN", "also": ["NaN", "Infinity", "-Infinity"]}),
    ({"detail": "a\ud800b"}, {"detail": "a\ufffdb"}),
    ({"a\udfff": ["\ud800\udfff", NAN]}, {"a\ufffd": ["\ufffd\ufffd", "NaN"]}),
    ({5: NAN, (1, 2): "x", None: 1}, {"5": "NaN", "(1, 2)": "x", "None": 1}),
    ({"v": NAN, "o": _Unknown(), "p": Path("a/b"), "t": (1, NAN)},
     {"v": "NaN", "o": "an object", "p": "a/b", "t": [1, "NaN"]}),
    ({"o": _Unknown()}, {"o": "an object"}),
    ({"detail": [{"type": "int_parsing", "loc": ["query", "n_steps"], "input": "NaN",
                  "ctx": {"error": NAN}}]},
     {"detail": [{"type": "int_parsing", "loc": ["query", "n_steps"], "input": "NaN",
                  "ctx": {"error": "NaN"}}]}),
], ids=lambda value: repr(value).encode("ascii", "replace").decode()[:50])
def test_a_reply_is_written_for_what_the_default_encoder_refuses(content, written):
    with pytest.raises((ValueError, TypeError)):
        JSONResponse(content=content)
    reply = _Reply(status_code=422, content=content)
    assert reply.status_code == 422
    assert strict(reply.body) == written


def test_a_reply_nested_past_the_interpreters_depth_keeps_its_status_and_says_so():
    """The last resort: the encoder raises for nothing, so a refusal is
    never turned into a 500 by its own echo."""
    deep: list = [NAN]
    for _ in range(5000):
        deep = [deep]
    reply = _Reply(status_code=422, content={"detail": deep})
    assert reply.status_code == 422
    assert reply.body == _UNWRITABLE_REPLY
    assert "could not be written" in strict(reply.body)["detail"]


# ---------------------------------------------------------------------------
# The app uses no other
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=3.0))
    return SimulationServer({"BallNode": BallNode}, graph_manager=gm).create_app()


@pytest.fixture(scope="module")
def client(app):
    with LoopbackTestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_no_reply_of_the_server_module_is_built_with_the_default_encoder():
    """A reply path added later with ``JSONResponse(...)`` would bring the
    defect back for whatever it echoes; the module builds none."""
    source = inspect.getsource(server_module)
    assert re.findall(r"\bJSONResponse\(", source) == []
    assert "class _Reply(JSONResponse):" in source
    assert len(re.findall(r"\b_Reply\(", source)) >= 9


def test_every_route_and_both_handlers_reply_through_the_one_encoder(app):
    routes = [route for route in app.routes if isinstance(route, APIRoute)]
    assert len(routes) > 30
    other = {f"{sorted(route.methods)} {route.path}": route.response_class.__name__
             for route in routes
             if route.response_class not in (_Reply, HTMLResponse, PlainTextResponse)}
    assert not other, other
    assert sum(route.response_class is _Reply for route in routes) > 25
    for raised in (RequestValidationError, StarletteHTTPException):
        handler = app.exception_handlers[raised]
        assert handler.__module__ == server_module.__name__, raised


def test_a_route_that_returns_what_the_default_encoder_refuses_still_answers():
    """The route's own return value, not only a refusal: a route that
    forgot ``_json_reply`` used to be a 500 after it had done its work."""
    probe = SimulationServer({"BallNode": BallNode}).create_app()

    @probe.get("/probe", response_model=None)
    def _probe():
        return {"value": NAN, "bound": -INF, "text": "NaN", "echo": "a\ud800b"}

    with LoopbackTestClient(probe, raise_server_exceptions=False) as client:
        resp = client.get("/probe")
    assert resp.status_code == 200
    assert strict(resp.content) == {"value": "NaN", "bound": "-Infinity", "text": "NaN",
                                    "echo": "a\ufffdb"}


@pytest.mark.parametrize("detail, written", [
    ("NaN", "NaN"), ("Infinity", "Infinity"), ("-Infinity", "-Infinity"),
    ("a\ud800b", "a\ufffdb"), (NAN, "NaN"),
    ({"got": INF, "text": "NaN", "name": "\udfff"},
     {"got": "Infinity", "text": "NaN", "name": "\ufffd"}),
], ids=lambda value: repr(value).encode("ascii", "replace").decode()[:40])
@pytest.mark.parametrize("raised", [HTTPException, StarletteHTTPException],
                         ids=["fastapi", "starlette"])
def test_an_http_exception_is_answered_whatever_its_detail_echoes(app, raised, detail, written):
    """Every ``HTTPException(detail=...)`` a route raises goes through the
    app's handler, with its status and its headers."""
    handler = app.exception_handlers[StarletteHTTPException]
    reply = asyncio.run(handler(None, raised(status_code=409, detail=detail,
                                             headers={"Retry-After": "1"})))
    assert isinstance(reply, _Reply) and reply.status_code == 409
    assert reply.headers["retry-after"] == "1"
    assert strict(reply.body) == {"detail": written}


@pytest.mark.parametrize("status", [204, 205, 304])
def test_an_http_exception_of_a_status_without_a_body_has_none(app, status):
    handler = app.exception_handlers[StarletteHTTPException]
    reply = asyncio.run(handler(None, HTTPException(status_code=status)))
    assert reply.status_code == status and reply.body == b""


def test_starlettes_own_refusals_keep_their_status_body_and_headers(client):
    missing = client.get("/no/such/route/NaN")
    assert missing.status_code == 404 and strict(missing.content) == {"detail": "Not Found"}
    wrong = client.delete("/graph")
    assert wrong.status_code == 405
    assert strict(wrong.content) == {"detail": "Method Not Allowed"}
    assert wrong.headers["allow"] == "GET"


@pytest.mark.parametrize("url", ["/sim/run?n_steps=NaN", "/sim/run?n_steps=Infinity",
                                 "/sim/run?n_steps=-Infinity"])
def test_a_count_that_is_the_text_of_a_non_finite_number_is_a_422_that_echoes_it(client, url):
    """The minimal request, and what the 422 says: the text, as text."""
    resp = client.post(url)
    assert resp.status_code == 422, resp.text
    (error,) = strict(resp.content)["detail"]
    assert error["loc"] == ["query", "n_steps"] and error["input"] == url.split("=")[1]
    assert client.post("/sim/run?n_steps=abc").status_code == 422


def test_a_422_still_writes_a_refused_number_as_its_token(client):
    """What the handler was written for (a bare NaN in a body) still holds."""
    resp = client.post("/graph/nodes", content=b'{"type": "BallNode", "name": "n", '
                       b'"timestep": NaN}', headers={"content-type": "application/json"})
    assert resp.status_code == 422, resp.text
    assert any(error.get("input") == "NaN" for error in strict(resp.content)["detail"])
    assert client.get("/graph").status_code == 200
