"""Requests generated from the server's own schema are served or refused whole.

Three of the last audit round's REST findings were one malformed value on a
documented field: a NaN ``timestep`` on ``POST /graph/nodes`` answered 500
after adding the node, an integer out of its range on ``PUT /graph/state``
answered 500, and a non-ASCII digit in a ``Host`` port answered 500.  No
test generated such values for *every* field.  This module does, from the
application itself:

* the routes are read from the app's OpenAPI document (FastAPI builds it
  from the route signatures and the pydantic request models), with each
  path parameter, query parameter and request body, and from
  ``app.routes``; a documented route with no request generator here fails
  ``test_every_documented_route_has_a_request_generator``, so a new route
  cannot arrive untested.  The surrogate-training routes, the streams and
  every ``/cloud/*`` route are out of scope and listed
  (:data:`OUT_OF_SCOPE`); no request is ever sent to them;
* each route has one or more *seeds* (:data:`SEEDS`): a request the served
  graph takes -- which node, which parameters -- the part a schema cannot
  say.  Every seed is first sent as it is and must be answered as it says;
* **the battery**: every field of every seed -- each path parameter, each
  query parameter, each member of the JSON body down to an array's
  elements, and the body itself -- is replaced in turn by every malformed
  value of its kind.  For a number: NaN, both infinities, zero, a negative,
  integers of 2**63 and 10**400, a fraction where an integer is expected, a
  value float32 cannot hold, text, a boolean, null, a list, an object.
  For text: empty, very long, the reserved names (``_meta``, ``_params``),
  a path with ``/`` and ``..``, non-ASCII digits, a NUL, and each
  non-string JSON type.  A field is also left out, and one is added.  The
  same is done to the headers a proxy or the ASGI server interprets
  (``Host``, ``Origin``, ``X-Forwarded-For``, ``Forwarded``,
  ``Content-Length``, ``Content-Type``), in process and, for the ones the
  HTTP server itself reads, over a real uvicorn server on loopback;
* **the fuzzer**: Hypothesis draws a seed, replaces up to three of its
  fields with drawn values of any JSON type, and may add a header.

Nothing is predicted of a malformed request.  Whatever the server answers:

1. it is not a 5xx;
2. a refusal changed nothing -- the config, the parameters, the state, the
   files under the checkpoint root and the streams' clock
   (``rest_oracle.assert_nothing_changed``) -- and neither did a ``GET``;
3. a 4xx detail names no absolute server path the request did not carry;
4. a 2xx reply of a JSON route is strict JSON: no ``NaN`` or ``Infinity``
   token; and what a write route accepted is a graph whose save reloads,
   inside its stability limits, that can still step
   (``rest_oracle.check_accepted_graph``);

and, of the three header rules, which are refusals by their definition:

5. a request that carries a forwarding header, a ``Host`` that is not a
   name of this machine, or (on a state-changing method) a foreign
   ``Origin``, and no token, is never served.

Sizes are kept harmless: the longest text is 20 000 characters, and no
drawn count is between 16 and 2**40, so nothing here asks the server for a
long run or a large node.  Nothing here can reach a cloud provider
(:func:`tests.property.differential.no_cloud_launch`; :func:`exchange`
refuses to send to an out-of-scope path).
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import os
import posixpath
import socket
from typing import Any, Callable, Iterator, Optional
from urllib.parse import quote, unquote, urlencode

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from fastapi.routing import APIRoute
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.api import server as server_module
from maddening.api.auth import UNAUTHENTICATED_PATHS
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode, HeatNode, SpringDamperNode
from maddening.nodes.lbm import LBMNode

from tests.conftest import EXAMPLES_FLOOR
from tests.property import rest_oracle as O
from tests.property.differential import no_cloud_launch, note, quiet


@pytest.fixture(scope="module", autouse=True)
def _no_cloud():
    with no_cloud_launch():
        yield


# ---------------------------------------------------------------------------
# The routes, from the application
# ---------------------------------------------------------------------------

#: Route prefixes no request here is sent to: the surrogate-training routes
#: and the streams (EXPERIMENTAL in 0.4.0, hardened in 0.5.0), and
#: ``/cloud/*`` (which could reach a provider).
OUT_OF_SCOPE = ("/surrogate/", "/ws/", "/cloud/")
def framework_routes(app) -> tuple:
    """The routes FastAPI itself adds on a loopback bind -- the schema, its
    two viewers and the first one's redirect page -- as *app* names them.
    They take no input; the header battery is asked of them."""
    return tuple(url for url in (app.openapi_url, app.docs_url,
                                 app.swagger_ui_oauth2_redirect_url, app.redoc_url) if url)


@dataclasses.dataclass(frozen=True)
class Operation:
    """One documented route, as the OpenAPI document states it."""

    method: str
    path: str
    path_params: tuple
    #: ``((name, JSON schema), ...)``
    query: tuple
    #: the request body's JSON schema, ``$ref``s resolved, or ``None``
    body: Optional[dict]
    #: whether a 2xx reply is declared ``application/json``
    json_reply: bool

    @property
    def key(self) -> str:
        return f"{self.method} {self.path}"


def _resolved(schema: Any, components: dict) -> Any:
    if isinstance(schema, dict):
        if "$ref" in schema:
            return _resolved(components[schema["$ref"].rsplit("/", 1)[1]], components)
        return {k: _resolved(v, components) for k, v in schema.items()}
    if isinstance(schema, list):
        return [_resolved(v, components) for v in schema]
    return schema


def documented_operations(app) -> dict[str, Operation]:
    """Every operation of *app*'s OpenAPI document, by ``"METHOD /path"``."""
    spec = app.openapi()
    components = spec.get("components", {}).get("schemas", {})
    out = {}
    for path, item in spec["paths"].items():
        for method, op in item.items():
            params = op.get("parameters", [])
            body = op.get("requestBody", {}).get("content", {}).get("application/json")
            replies = [r for code, r in op.get("responses", {}).items() if code.startswith("2")]
            operation = Operation(
                method=method.upper(), path=path,
                path_params=tuple(p["name"] for p in params if p["in"] == "path"),
                query=tuple((p["name"], _resolved(p["schema"], components))
                            for p in params if p["in"] == "query"),
                body=None if body is None else _resolved(body["schema"], components),
                json_reply=all("application/json" in r.get("content", {}) for r in replies))
            out[operation.key] = operation
    return out


def _in_scope(path: str) -> bool:
    return not path.startswith(OUT_OF_SCOPE)


#: The documented routes of a loopback-bound server, read once at import to
#: parametrise the tests (the fail-closed test builds its own app).
_APP = SimulationServer({}).create_app()
OPERATIONS = documented_operations(_APP)
IN_SCOPE = {key: op for key, op in OPERATIONS.items() if _in_scope(op.path)}
FRAMEWORK = {f"GET {path}": Operation("GET", path, (), (), None, path == _APP.openapi_url)
             for path in framework_routes(_APP)}


# ---------------------------------------------------------------------------
# The served graph, and a seed request per route
# ---------------------------------------------------------------------------

ROD_CELLS = 4


def standard_graph() -> GraphManager:
    """A rod (an array field, an integer and a list parameter), a spring
    and a ball (scalar fields and float parameters), one edge, compiled."""
    gm = GraphManager()
    gm.add_node(HeatNode("rod", 0.01, n_cells=ROD_CELLS, length=1.0, thermal_diffusivity=0.01,
                         initial_temperature=[0.0, 1.0, 2.0, 3.0]))
    gm.add_node(SpringDamperNode("spring", 0.01, stiffness=40.0, damping=0.5,
                                 initial_position=0.5))
    gm.add_node(BallNode("ball", 0.01, initial_position=3.0))
    gm.add_edge("spring", "ball", "position", "table_position")
    with quiet():
        gm.compile()
    return gm


def lattice_graph() -> GraphManager:
    """The standard graph and a 4 x 3 ``LBMNode``, whose ``wall_mask`` is
    the stock ``uint8`` state field.  Only the state route's seed uses it:
    nothing steps it (its step is seconds to compile)."""
    gm = GraphManager()
    gm.add_node(LBMNode("lattice", 1.0, grid_shape=(4, 3), lattice="D2Q9", viscosity=0.1))
    gm.add_node(BallNode("ball", 0.01, initial_position=3.0))
    with quiet():
        gm.compile()
    return gm


GRAPHS: dict[str, Callable[[], GraphManager]] = {"standard": standard_graph,
                                                 "lattice": lattice_graph}
#: The checkpoint every served graph has saved, for the load route.
SAVED = "saved.npz"


def serve(graph: str = "standard") -> O.Served:
    registry = dict(O.REGISTRY)
    if graph == "lattice":
        registry["LBMNode"] = LBMNode        # so the graph's save reloads
    served = O.serve(GRAPHS[graph](), registry=registry)
    resp = served.client.post("/checkpoint/save", params={"path": SAVED})
    assert resp.status_code == 200, resp.text
    return served


@dataclasses.dataclass(frozen=True)
class Seed:
    """A request the served graph takes: the part of a valid request the
    schema does not say.  *body* may be a function of the served graph (a
    state write names every field the node has)."""

    path: dict = dataclasses.field(default_factory=dict)
    query: dict = dataclasses.field(default_factory=dict)
    body: Any = None
    graph: str = "standard"
    #: what the plain seed is answered: 409 for a runner route with no
    #: runner (starting one would step the graph under every comparison)
    status: int = 200
    label: str = ""


def _whole_state(node: str) -> Callable[[O.Served], dict]:
    def body(served: O.Served) -> dict:
        state = served.gm.get_node_state(node)
        return {"state": {f: np.asarray(v).tolist() for f, v in state.items()}}
    return body


_EDGE = {"source_node": "spring", "target_node": "ball", "source_field": "position",
         "target_field": "table_position"}

SEEDS: dict[str, tuple] = {
    "GET /healthz": (Seed(),),
    "GET /viz/auth.js": (Seed(),),
    "GET /viz/graph": (Seed(),),
    "GET /viz/app": (Seed(),),
    "GET /viz/render": (Seed(),),
    "GET /graph": (Seed(),),
    "POST /graph/nodes": (
        Seed(body={"type": "SpringDamperNode", "name": "new", "timestep": 0.01,
                   "params": {"stiffness": 20.0, "damping": 0.5}}, status=201, label="a spring"),
        Seed(body={"type": "HeatNode", "name": "new", "timestep": 0.01,
                   "params": {"n_cells": ROD_CELLS, "thermal_diffusivity": 0.01,
                              "initial_temperature": [0.0, 1.0, 2.0, 3.0]}},
             status=201, label="a rod"),
    ),
    "DELETE /graph/nodes/{name}": (Seed(path={"name": "ball"}),),
    "POST /graph/edges": (Seed(body={"source_node": "ball", "target_node": "spring",
                                     "source_field": "position",
                                     "target_field": "anchor_position"}, status=201),),
    "DELETE /graph/edges": (Seed(body=dict(_EDGE)),),
    "POST /graph/compile": (Seed(),),
    "POST /graph/validate": (Seed(),),
    "GET /graph/state": (Seed(),),
    "GET /graph/state/{node_name}": (Seed(path={"node_name": "ball"}),),
    "PUT /graph/state/{node_name}": (
        Seed(path={"node_name": "ball"}, body={"state": {"position": 2.0, "velocity": 0.5}},
             label="scalar fields"),
        Seed(path={"node_name": "rod"}, body={"state": {"temperature": [1.0, 2.0, 3.0, 4.0]}},
             label="an array field"),
        Seed(path={"node_name": "lattice"}, body=_whole_state("lattice"), graph="lattice",
             label="an integer field"),
    ),
    "GET /graph/params/{node_name}": (Seed(path={"node_name": "spring"}),),
    "PUT /graph/params/{node_name}": (
        Seed(path={"node_name": "spring"}, body={"params": {"stiffness": 41.0, "damping": 0.25}},
             label="float leaves"),
        Seed(path={"node_name": "rod"},
             body={"params": {"thermal_diffusivity": 0.02, "n_cells": ROD_CELLS,
                              "initial_temperature": [3.0, 2.0, 1.0, 0.0]}},
             label="an integer and a list"),
    ),
    "POST /checkpoint/save": (Seed(query={"path": "new.npz"}),),
    "POST /checkpoint/load": (Seed(query={"path": SAVED}),),
    "POST /sim/step": (Seed(),),
    "POST /sim/run": (Seed(query={"n_steps": 2}),),
    "POST /sim/start": (Seed(),),
    "POST /sim/pause": (Seed(status=409),),
    "POST /sim/resume": (Seed(status=409),),
    "POST /sim/stop": (Seed(status=409),),
    "POST /sim/reset": (Seed(),),
    "PUT /sim/stride": (Seed(query={"steps_per_frame": 2, "relay_stride": 3}),),
    "POST /sim/profile": (Seed(query={"n_steps": 1, "n_warmup": 0}),),
    "POST /sim/profile/jax/start": (Seed(),),
    "POST /sim/profile/jax/stop": (Seed(status=409),),
    "GET /sim/profile/jax/status": (Seed(),),
}
#: Routes whose acceptance leaves their own seed valid (a parameter or a
#: state written again, a step, a save over a save): the graph is kept for
#: the next variant.  After any other accepted write a fresh graph is served.
KEEPS_ITS_SEED = frozenset({
    "PUT /graph/state/{node_name}", "PUT /graph/params/{node_name}", "POST /graph/compile",
    "POST /graph/validate", "POST /checkpoint/save", "POST /checkpoint/load", "POST /sim/step",
    "POST /sim/run", "POST /sim/reset", "PUT /sim/stride", "POST /sim/profile"})


# ---------------------------------------------------------------------------
# A request
# ---------------------------------------------------------------------------

#: A body that is not sent at all, and one sent as these bytes.
ABSENT = object()


@dataclasses.dataclass(frozen=True)
class Raw:
    data: bytes


@dataclasses.dataclass
class Request:
    method: str
    template: str
    path: dict
    #: ``[(name, text), ...]``: a list, so a name can be repeated
    query: list
    body: Any
    headers: dict

    def url(self) -> str:
        url = self.template
        for name, value in self.path.items():
            # Quoted whole, so a ``/`` or ``..`` stays inside its segment.
            url = url.replace("{" + name + "}", quote(str(value), safe=""))
        if self.query:
            url += "?" + urlencode(self.query)
        return url

    def content(self) -> Optional[bytes]:
        if self.body is ABSENT:
            return None
        if isinstance(self.body, Raw):
            return self.body.data
        # NaN and Infinity as Python's encoder (and a careless client)
        # spells them: the server's parser reads them.
        return json.dumps(self.body, allow_nan=True).encode()

    def sent(self) -> str:
        """Everything the request carried, as text: a path in a reply that
        is also here was the client's own."""
        body = self.content() or b""
        return " ".join([unquote(self.url()), body.decode("utf-8", "replace"),
                         json.dumps(self.headers)])

    def describe(self) -> str:
        body = self.content()
        text = "" if body is None else " " + body[:200].decode("utf-8", "replace")
        headers = {k: v for k, v in self.headers.items() if k.lower() != "content-type"}
        return f"{self.method} {self.url()[:200]}{text}{' ' + str(headers)[:200] if headers else ''}"


def request_of(op: Operation, seed: Seed, served: O.Served) -> Request:
    body = seed.body(served) if callable(seed.body) else copy.deepcopy(seed.body)
    return Request(op.method, op.path, dict(seed.path),
                   [(k, str(v)) for k, v in seed.query.items()],
                   ABSENT if body is None else body,
                   {} if body is None else {"content-type": "application/json"})


class NotSent(Exception):
    """A request :func:`exchange` would not send."""


#: The three strings ``maddening.serialization.json_codec`` writes a
#: non-finite float as.
TOKEN_TEXTS = ("NaN", "Infinity", "-Infinity")
def left_running(served: O.Served) -> list[str]:
    """What a request has started that goes on by itself: the runner,
    whose thread steps the graph, and a JAX trace."""
    from maddening.core.simulation import profiler

    running = []
    runner = served.server.runner
    if runner is not None and runner.is_alive:
        running.append("the runner")
    if profiler.jax_trace_active():
        running.append("a JAX trace")
    return running


def settle(served: O.Served) -> None:
    """Stop what an accepted request may have started: the runner, a trace."""
    from maddening.core.simulation import profiler

    with served.server._runner_lock:  # noqa: SLF001
        served.server._stop_runner()  # noqa: SLF001
    if profiler.jax_trace_active():
        profiler.stop_jax_trace()


def exchange(served: O.Served, op: Operation, req: Request, *,
             must_refuse: bool = False) -> tuple[list, Any]:
    """Send *req* and hold the reply to the invariants: ``(problems, reply)``.

    Nothing is sent to an out-of-scope route (``/cloud/*`` above all).  A
    path parameter is quoted whole, so the path the server routes is the
    route's own with one opaque segment; a request whose path would name
    an out-of-scope route if something on the way decoded and normalised
    it (``../../cloud/status``) is not sent either (:class:`NotSent`).

    Every comparison is of a graph nothing is stepping.  ``POST /sim/start``
    starts the runner, whose thread steps the graph from then on: a
    checkpoint saved while it runs and the live state read a moment later
    differ by the steps taken between them, so "the state the reload
    restores" failed or passed by the scheduler (it failed on the CI
    runners and passed on a workstation).  So nothing is running when the
    request is sent, and what it started is stopped, its thread joined,
    before anything is read -- and a request that was *refused* must have
    started nothing, which is asked first."""
    url = req.url()
    assert _in_scope(url) and _in_scope(req.template), f"{url} is out of scope"
    if not _in_scope(posixpath.normpath(unquote(url.split("?", 1)[0])) + "/"):
        raise NotSent(url)
    assert not left_running(served), f"{left_running(served)} before {req.describe()}"
    before = O.snapshot(served)
    # Header values as latin-1 bytes, the way they travel: a non-ASCII
    # digit in a port is one byte the client library would not write as text.
    headers = [(name.encode("ascii"), value.encode("latin-1"))
               for name, value in req.headers.items()]
    resp = served.client.request(req.method, url, content=req.content(), headers=headers,
                                 follow_redirects=False)
    status, problems = resp.status_code, []
    started = left_running(served)
    settle(served)
    assert not left_running(served), f"{started} could not be stopped after {req.describe()}"
    if started and status >= 400:
        problems.append(f"{status}, and the request left {' and '.join(started)} running")
    if status >= 500:
        changed = O.differences(before, O.snapshot(served))
        problems.append(f"{status}: {resp.text[:300]}" + (
            " -- and the server changed: " + "; ".join(changed)[:400] if changed
            else " (nothing was changed)"))
    refused = 400 <= status < 500
    if refused or req.method == "GET":
        found = O.differences(before, O.snapshot(served))
        if found:
            problems.append(f"{status} and the server changed: " + "; ".join(found)[:600])
    if refused:
        named = O.server_paths_named(O.detail_text(resp), req.sent(), served)
        if named:
            problems.append(f"{status} names the server path(s) {named}: {resp.text[:300]}")
    if 200 <= status < 300 and op.json_reply:
        try:
            O.assert_strict_json(resp, req.describe())
        except AssertionError as exc:
            problems.append(str(exc)[:400])
    if must_refuse and status < 400:
        problems.append(f"{status}: served, where the Host, Origin or forwarding rule refuses")
    if 200 <= status < 300 and req.method != "GET":
        # What the server accepted is a graph its save reloads, inside its
        # stability limits, that can still step (rest_oracle, 3 and 5; the
        # sequence oracle also steps it beside its reload).
        try:
            O.check_accepted_graph(served, req.describe(), step=False,
                                   shapes_may_differ=op.key == "POST /graph/edges")
        except AssertionError as exc:
            problems.append(f"{status}: {str(exc)[:500]}")
    return problems, resp


# ---------------------------------------------------------------------------
# Malformed values, by kind of field
# ---------------------------------------------------------------------------

LONG = "x" * 20_000
NUMBER_VALUES: tuple = (
    ("NaN", float("nan")), ("+Infinity", float("inf")), ("-Infinity", float("-inf")),
    ("zero", 0), ("a negative", -1.5), ("a negative integer", -1),
    ("an integer of 2**63", 2 ** 63), ("an integer of 10**400", 10 ** 400),
    ("a fraction", 2.5), ("a float below float32", 1e-50), ("a float above float32", 1e39),
    ("a numeric string", "1.5"), ("text", "abc"), ("the text NaN", "NaN"),
    ("the text Infinity", "Infinity"), ("the text -Infinity", "-Infinity"),
    ("a boolean", True), ("null", None), ("a list", [1.0, 2.0]), ("an object", {"x": 1.0}),
)
STRING_VALUES: tuple = (
    ("empty", ""), ("very long", LONG), ("_meta", "_meta"), ("_params", "_params"),
    ("_params_mappings", "_params_mappings"), ("a path", "../../etc/passwd"),
    ("a slash", "a/b"), ("two dots", ".."), ("non-ASCII digits", "١٢٣"),
    ("a superscript digit", "²"), ("a NUL", "a\x00b"), ("spaces", "  "),
    ("the text NaN", "NaN"), ("the text Infinity", "Infinity"),
    ("the text -Infinity", "-Infinity"), ("a number", 5), ("a boolean", True), ("null", None),
    ("a list", ["a"]), ("an object", {"a": "b"}),
    # What not every carrier of a name can hold (a surrogate, an escape,
    # U+FFFE) and what an edge's key is made with (a hash): refused where
    # a name is introduced, echoed without a 500 everywhere else.  A line
    # break and a dot are unusual and carried: taken, and the graph must
    # reload.  The surrogate is a body's alone (a URL cannot spell one:
    # see can_be_in_a_url).
    ("a lone surrogate", "a\ud800b"), ("a line break", "a\nb"),
    ("an escape character", "a\x1bb"), ("U+FFFE", "a\ufffeb"), ("a hash", "a#b"),
    ("a dot", "a.b"),
)


def can_be_in_a_url(value: Any) -> bool:
    """Whether *value* is text a path or a query can spell: a lone
    surrogate has no UTF-8 bytes to percent-encode."""
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


OBJECT_VALUES: tuple = (
    ("an empty object", {}), ("text", "abc"), ("a number", 1), ("null", None), ("a list", []),
    ("a reserved key", {"_meta": 1.0}), ("an empty key", {"": 1.0}),
    ("a very long key", {LONG: 1.0}), ("a nested object", {"a": {"b": {"c": 1.0}}}),
    ("NaN under an unknown key", {"x": float("nan")}),
)
#: Query values are text: the same kinds, as a URL spells them.
QUERY_NUMBER_VALUES: tuple = (
    ("NaN", "NaN"), ("+Infinity", "Infinity"), ("-Infinity", "-Infinity"), ("zero", "0"),
    ("a negative", "-1"), ("an integer of 2**63", str(2 ** 63)),
    ("an integer of 400 digits", "9" * 400), ("a fraction", "2.5"), ("an exponent", "1e400"),
    ("text", "abc"), ("a boolean", "true"), ("null", "null"), ("empty", ""),
    ("non-ASCII digits", "١٢"), ("a superscript digit", "²"),
    ("hexadecimal", "0x10"), ("spaces around it", " 1 "), ("a list", "[1]"),
)
QUERY_STRING_VALUES: tuple = tuple((label, value) for label, value in STRING_VALUES
                                   if can_be_in_a_url(value)) + (
    ("an absolute path", "/etc/hostname"), ("a path out of the root", "a/../../x.npz"),
    ("a directory", "nested/dir/file"),
)
#: Bodies that are not the JSON object a route expects.
BODY_VALUES: tuple = (
    ("no body", ABSENT), ("an empty object", {}), ("a list", []), ("text", "abc"),
    ("null", None), ("a number", 1), ("the text NaN", "NaN"), ("not JSON", Raw(b"{")),
    ("bytes that are not UTF-8", Raw(b'{"a": "\xff\xfe"}')),
    ("50 000 nested lists", Raw(b"[" * 50_000)), ("trailing text", Raw(b"{} x")),
    ("a bare NaN", Raw(b"NaN")), ("an empty body", Raw(b"")),
)
REFUSE, REFUSE_WRITES, REFUSE_UNAUTHENTICATED = "always", "writes", "unauthenticated"
#: Header values, and which a rule refuses by definition (invariant 5):
#: a Host that is not a name of this machine (REST-025, REST-028), a
#: foreign Origin on a state-changing method (REST-021), and a forwarding
#: header without the token (REST-005).
HEADER_VALUES: dict[str, tuple] = {
    "Host": (
        ("empty", "", None), ("a foreign name", "attacker.example", REFUSE),
        ("a foreign name and port", "evil.example:8000", REFUSE),
        ("an unclosed bracket", "[::1", REFUSE), ("text after the bracket", "[::1]x", REFUSE),
        ("a space", "a b", REFUSE), ("a non-ASCII digit port", "localhost:²", REFUSE),
        ("two ports", "::1:80:80", REFUSE), ("userinfo", "user@127.0.0.1", REFUSE),
        ("a path", "127.0.0.1/x", REFUSE), ("very long", "h" * 5000, REFUSE),
        ("a port past 65535", "127.0.0.1:99999", None), ("localhost with a dot", "LOCALHOST.", None),
    ),
    "Origin": (
        ("a foreign origin", "http://evil.example", REFUSE_WRITES), ("null", "null", REFUSE_WRITES),
        ("userinfo", "http://evil.example@127.0.0.1", REFUSE_WRITES),
        ("an unclosed bracket", "http://[::1", REFUSE_WRITES),
        ("a non-ASCII digit port", "http://127.0.0.1:²", REFUSE_WRITES),
        ("no scheme", "127.0.0.1", REFUSE_WRITES),
        ("very long", "http://" + "o" * 5000, REFUSE_WRITES),
        ("empty", "", None), ("this server's own", "http://127.0.0.1", None),
    ),
    "X-Forwarded-For": tuple((label, value, REFUSE_UNAUTHENTICATED) for label, value in (
        ("a word", "x"), ("loopback", "127.0.0.1"), ("IPv6 loopback", "::1"),
        ("a chain", "203.0.113.5, 127.0.0.1"), ("empty", ""), ("a non-ASCII digit", "²"),
        ("very long", "1.2.3.4, " * 600))),
    "Forwarded": tuple((label, value, REFUSE_UNAUTHENTICATED) for label, value in (
        ("for loopback", "for=127.0.0.1"), ("for IPv6 loopback", 'for="[::1]"'),
        ("an obfuscated node", "for=_hidden;proto=https"), ("text", "garbage"),
        ("empty", ""), ("very long", "for=1.2.3.4, " * 400))),
    "Content-Length": tuple((label, value, None) for label, value in (
        ("zero", "0"), ("one", "1"), ("a negative", "-1"), ("text", "abc"), ("empty", ""),
        ("a fraction", "1.5"), ("a non-ASCII digit", "²"), ("30 digits", "9" * 30),
        ("one past the body limit", str(server_module.MAX_REQUEST_BODY_BYTES + 1)))),
    "Content-Type": tuple((label, value, None) for label, value in (
        ("text/plain", "text/plain"), ("empty", ""), ("another charset",
                                                     "application/json; charset=utf-16"),
        ("a form", "application/x-www-form-urlencoded"), ("text", "garbage"))),
}


@dataclasses.dataclass(frozen=True)
class Variant:
    """One change to a seed request: *where*, *what*, and how to make it."""

    where: str
    what: str
    apply: Callable[[Request], None]
    refuse: Optional[str] = None

    def must_refuse(self, op: Operation) -> bool:
        if self.refuse == REFUSE:
            return True
        if self.refuse == REFUSE_WRITES:
            return op.method in ("POST", "PUT", "PATCH", "DELETE")
        if self.refuse == REFUSE_UNAUTHENTICATED:
            return op.path not in UNAUTHENTICATED_PATHS
        return False


def _at(body: Any, pointer: tuple) -> Any:
    for key in pointer:
        body = body[key]
    return body


def _put(pointer: tuple, value: Any) -> Callable[[Request], None]:
    def apply(req: Request) -> None:
        if not pointer:
            req.body = copy.deepcopy(value) if not isinstance(value, Raw) and value is not ABSENT \
                else value
            if value is ABSENT:
                req.headers.pop("content-type", None)
            else:
                req.headers.setdefault("content-type", "application/json")
        else:
            _at(req.body, pointer[:-1])[pointer[-1]] = copy.deepcopy(value)
    return apply


def _drop(pointer: tuple) -> Callable[[Request], None]:
    def apply(req: Request) -> None:
        del _at(req.body, pointer[:-1])[pointer[-1]]
    return apply


def _schema_at(schema: Optional[dict], pointer: tuple) -> dict:
    """The declared schema of the body member at *pointer*, as far as the
    model declares one (a ``dict[str, Any]`` declares nothing of its
    members)."""
    for key in pointer:
        if not isinstance(schema, dict):
            return {}
        if isinstance(key, int):
            schema = schema.get("items")
        else:
            schema = (schema.get("properties") or {}).get(key)
    return schema if isinstance(schema, dict) else {}


def _kind(value: Any, schema: dict) -> str:
    """What a body member is: by the model's declared type where it
    declares one, else by the seed's value."""
    declared = schema.get("type")
    if declared in ("string", "object", "array", "boolean"):
        return declared
    if declared in ("number", "integer"):
        return "number"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "array" if isinstance(value, list) else "object"


def _members(body: Any, pointer: tuple = ()) -> Iterator[tuple]:
    """Every member of a JSON body, by pointer: an object's values, and an
    array's first element (its elements are one field, written many times)."""
    if isinstance(body, dict):
        for key, value in body.items():
            yield pointer + (key,)
            yield from _members(value, pointer + (key,))
    elif isinstance(body, list) and body:
        yield pointer + (0,)
        yield from _members(body[0], pointer + (0,))


def _pointer_text(pointer: tuple) -> str:
    return "body" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in pointer)


def body_variants(op: Operation, body: Any) -> Iterator[Variant]:
    """Every malformed value of every member of *body*, the member left
    out, a member added, and the body itself replaced."""
    if body is ABSENT:
        yield Variant("body", "a body where none is expected",
                      _put((), {"unexpected": 1.0}))
        return
    for label, value in BODY_VALUES:
        yield Variant("body", label, _put((), value))
    yield Variant("body", "an extra member", _put(("unexpected_member",), 1.0))
    for pointer in _members(body):
        where = _pointer_text(pointer)
        value = _at(body, pointer)
        kind = _kind(value, _schema_at(op.body, pointer))
        values = {"number": NUMBER_VALUES, "string": STRING_VALUES, "object": OBJECT_VALUES,
                  "boolean": NUMBER_VALUES}.get(kind)
        if kind == "array":
            values = (("empty", []), ("one element more", value + value[:1]),
                      ("one element fewer", value[:-1]), ("nested once more", [value]),
                      ("a scalar", 1.0), ("text", "abc"), ("null", None),
                      ("ragged", [value, 1.0]), ("an object", {"0": 1.0}))
        for label, bad in values:
            yield Variant(where, label, _put(pointer, bad))
        if not isinstance(pointer[-1], int):
            yield Variant(where, "left out", _drop(pointer))
        if isinstance(value, dict):
            yield Variant(where, "an extra member", _put(pointer + ("unexpected_member",), 1.0))


def _set_query(name: str, *values: str) -> Callable[[Request], None]:
    def apply(req: Request) -> None:
        req.query = [(k, v) for k, v in req.query if k != name] + [(name, v) for v in values]
    return apply


def _is_count(schema: dict) -> bool:
    options = schema.get("anyOf") or [schema]
    return any(o.get("type") in ("integer", "number") for o in options)


def query_variants(op: Operation, seed: Seed) -> Iterator[Variant]:
    for name, schema in op.query:
        values = QUERY_NUMBER_VALUES if _is_count(schema) else QUERY_STRING_VALUES
        for label, value in values:
            yield Variant(f"query {name}", label, _set_query(name, value))
        yield Variant(f"query {name}", "left out", _set_query(name))
        yield Variant(f"query {name}", "given twice",
                      _set_query(name, str(seed.query.get(name, "1")), "2"))
    yield Variant("query", "a parameter the route does not have",
                  _set_query("unexpected_parameter", "1"))


def path_variants(op: Operation) -> Iterator[Variant]:
    def put(name: str, value: str) -> Callable[[Request], None]:
        return lambda req: req.path.__setitem__(name, value)

    for name in op.path_params:
        for label, value in STRING_VALUES:
            if can_be_in_a_url(value):
                yield Variant(f"path {name}", label, put(name, value))


def header_variants() -> Iterator[Variant]:
    def put(name: str, value: str) -> Callable[[Request], None]:
        return lambda req: req.headers.__setitem__(name, value)

    for name, values in HEADER_VALUES.items():
        for label, value, refuse in values:
            yield Variant(f"header {name}", label, put(name, value), refuse)


def variants(op: Operation, seed: Seed, served: O.Served) -> list[Variant]:
    """Every variant of *seed*'s request: the battery."""
    req = request_of(op, seed, served)
    return [*path_variants(op), *query_variants(op, seed), *body_variants(op, req.body),
            *header_variants()]


# ---------------------------------------------------------------------------
# The route list
# ---------------------------------------------------------------------------

def test_every_documented_route_has_a_request_generator():
    """Fail closed: every route of a freshly built app is an operation of
    its OpenAPI document with a seed here, a route under an out-of-scope
    prefix, or one of FastAPI's own -- so a route added to the server
    arrives with a failing test, not untested; and no seed names a route
    the server no longer has."""
    app = SimulationServer({}).create_app()
    documented = documented_operations(app)
    for route in app.routes:
        path = route.path
        if not _in_scope(path):
            continue
        if isinstance(route, APIRoute):
            for method in route.methods:
                assert f"{method} {path}" in documented, (
                    f"{method} {path} is routed but not in the OpenAPI document "
                    "(include_in_schema=False?): it has no generator here")
        else:
            assert path in framework_routes(app), (
                f"{type(route).__name__} {path} is neither a documented operation, an "
                f"out-of-scope prefix {OUT_OF_SCOPE} nor one of FastAPI's own routes")
    in_scope = {key for key, op in documented.items() if _in_scope(op.path)}
    missing, stale = sorted(in_scope - set(SEEDS)), sorted(set(SEEDS) - in_scope)
    assert not missing, f"documented routes with no request generator (SEEDS): {missing}"
    assert not stale, f"SEEDS names routes the server does not document: {stale}"
    out = sorted(key for key, op in documented.items() if not _in_scope(op.path))
    assert out and all(key.split(" ", 1)[1].startswith(OUT_OF_SCOPE) for key in out)


def test_every_seed_names_every_parameter_its_route_declares():
    """A seed that left a declared parameter out would leave it without a
    battery: every path and query parameter of a route is in each of its
    seeds, and a route with a body model has a seed body with every
    required member."""
    for key, seeds in SEEDS.items():
        op = IN_SCOPE[key]
        for seed in seeds:
            assert set(seed.path) == set(op.path_params), (key, seed.label)
            assert set(seed.query) == {name for name, _ in op.query}, (key, seed.label)
            assert (seed.body is None) == (op.body is None), (key, seed.label)
            if op.body is not None and not callable(seed.body):
                assert set(op.body.get("required", ())) <= set(seed.body), (key, seed.label)


# ---------------------------------------------------------------------------
# The battery
# ---------------------------------------------------------------------------

def _run_battery(op: Operation, seeds: tuple, chosen: Callable[[Variant], bool]) -> None:
    problems, sent = [], 0
    for seed in seeds:
        served = serve(seed.graph)
        try:
            plain, resp = exchange(served, op, request_of(op, seed, served))
            assert not plain and resp.status_code == seed.status, (
                f"the seed itself: {resp.status_code} {resp.text[:300]} {plain}")
            pristine = seed.status >= 400 or op.method == "GET"
            for variant in variants(op, seed, served):
                if not chosen(variant):
                    continue
                if not pristine and op.key not in KEEPS_ITS_SEED:
                    settle(served)
                    served.close()
                    served = serve(seed.graph)
                req = request_of(op, seed, served)
                variant.apply(req)
                found, resp = exchange(served, op, req, must_refuse=variant.must_refuse(op))
                sent += 1
                problems += [f"[{seed.label or 'seed'}] {variant.where} = {variant.what}: {p}"
                             for p in found]
                pristine = resp.status_code >= 400 or op.method == "GET"
        finally:
            settle(served)
            served.close()
    assert sent > 0
    assert not problems, (f"{op.key}: {len(problems)} problem(s) in {sent} requests:\n"
                          + "\n".join(problems[:30]))


#: One case per seed (a route with three seeds sends some three hundred
#: requests, and serves a fresh graph after each one it accepts).
_SEEDED = [pytest.param(key, i, id=key + (f" ({seed.label})" if seed.label else ""))
           for key in sorted(SEEDS) for i, seed in enumerate(SEEDS[key])]


@pytest.mark.parametrize("key, index", _SEEDED)
def test_every_malformed_value_of_every_field_is_refused_whole_or_served(key, index):
    """The battery over a route's own fields: each path parameter, query
    parameter and body member replaced in turn by every malformed value of
    its kind, left out, or joined by one the route does not have."""
    _run_battery(IN_SCOPE[key], SEEDS[key][index:index + 1],
                 lambda v: not v.where.startswith("header"))


@contextlib.contextmanager
def jax_recorder_not_run() -> Iterator[None]:
    """JAX's own trace recorder, not run, for the cases that send one
    route many requests per push.

    Every served ``POST /sim/profile/jax/start`` started ``jax.profiler``'s
    recorder, and the oracle stops what a request started before it
    compares anything.  On the CI runners that pair cost about a second:
    the header battery's case for the route (some fifty requests) took
    26.5 s and 31.3 s on the two lanes, past the per-test budget, and the
    per-push fuzzer's 14.0 s (the header case is 1.7 s on a workstation,
    0.6 s without the recorder).
    What a header or a drawn field is answered does not depend on the
    recorder, so here the session, its directory and the route's own state
    are the real ones and only the two calls into JAX are not made.

    The recorder still runs per push where the route's own fields are sent
    (``test_every_malformed_value_of_every_field_...``) and in
    ``tests/api/test_jax_trace_stops_itself.py``, and in the slow lane for
    the fuzzer at the profile's depth."""
    import jax.profiler

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(jax.profiler, "start_trace", lambda *args, **kwargs: None)
        patch.setattr(jax.profiler, "stop_trace", lambda *args, **kwargs: None)
        yield


@pytest.mark.parametrize("key", sorted(IN_SCOPE) + sorted(FRAMEWORK))
def test_every_malformed_header_is_refused_whole_or_served(key):
    """The battery over the headers a proxy or the server interprets, on
    every route: no 5xx, a refusal changes nothing, and a forwarding
    header, a Host that is not this machine's or a foreign Origin on a
    write is never served without the token."""
    op = IN_SCOPE.get(key) or FRAMEWORK[key]
    with jax_recorder_not_run():
        _run_battery(op, SEEDS.get(key, (Seed(),))[:1], lambda v: v.where.startswith("header"))


# ---------------------------------------------------------------------------
# What the battery found on the tree it was written on (fixed since)
# ---------------------------------------------------------------------------

#: Every door by which a refusal echoed text its encoder refused, as the
#: battery met them: each integer query parameter given the text of a
#: non-finite token, a model's number field given one as a JSON string, a
#: body that is that string, a refused body echoed whole with one inside
#: it -- and the same echoes of a surrogate, which UTF-8 cannot encode.
#: ``(method, url, body, the refusal's status)``.
_ECHOED = [
    ("POST", "/sim/run?n_steps=NaN", None, 422),
    ("POST", "/sim/run?n_steps=Infinity", None, 422),
    ("POST", "/sim/run?n_steps=-Infinity", None, 422),
    ("PUT", "/sim/stride?steps_per_frame=NaN", None, 422),
    ("PUT", "/sim/stride?relay_stride=Infinity", None, 422),
    ("POST", "/sim/profile?n_steps=NaN", None, 422),
    ("POST", "/sim/profile?n_warmup=-Infinity", None, 422),
    ("POST", "/graph/nodes", {"type": "BallNode", "name": "new", "timestep": "NaN",
                              "params": {}}, 422),
    ("POST", "/graph/nodes", "NaN", 422),
    ("POST", "/graph/nodes", {"type": "NaN"}, 422),
    ("POST", "/graph/edges", {"source_node": "Infinity"}, 422),
    ("DELETE", "/graph/edges", "-Infinity", 422),
    ("PUT", "/graph/state/ball", {"state": "NaN"}, 422),
    ("PUT", "/graph/params/spring", {"params": "Infinity"}, 422),
    # A surrogate: in a 422's echo, and in the detail of a route's own 4xx.
    ("POST", "/graph/nodes", {"type": "BallNode", "name": "new", "timestep": "\ud800"}, 422),
    ("POST", "/graph/edges", {"source_node": "a\udfffb"}, 422),
    ("PUT", "/graph/state/ball", {"state": "\ud800"}, 422),
    ("POST", "/graph/nodes", {"type": "a\ud800b", "name": "new", "timestep": 0.01}, 400),
    ("POST", "/graph/nodes", {"type": "BallNode", "name": "a\ud800b", "timestep": 0.01}, 400),
    ("POST", "/graph/edges", {"source_node": "a\ud800b", "target_node": "spring",
                              "source_field": "position", "target_field": "x"}, 404),
    ("PUT", "/graph/params/ball", {"params": {"a\ud800b": 1.0}}, 400),
]


@pytest.mark.parametrize("method, url, body, status", _ECHOED, ids=[
    f"{m} {u}{'' if b is None else ' ' + json.dumps(b)[:40]}" for m, u, b, _ in _ECHOED])
def test_a_422_that_echoes_a_non_finite_token_as_text_is_not_a_500(method, url, body, status):
    """``POST /sim/run?n_steps=NaN`` -- what a browser sends for
    ``parseInt("")`` -- answered 500 Internal Server Error, where
    ``n_steps=abc`` answered 422: the handler encoded the echo with the
    encoder of *data*, which refuses that text.  Every door the battery
    met is a case, and so is the echo of a surrogate, which was a 500 by
    Starlette's own encoder (and, on ``POST /graph/nodes``, after the node
    had been added)."""
    served = serve()
    try:
        before = O.snapshot(served)
        kwargs = {} if body is None else {"content": json.dumps(body),
                                          "headers": {"content-type": "application/json"}}
        resp = served.client.request(method, url, **kwargs)
        assert not O.differences(before, O.snapshot(served))
        assert resp.status_code == status, (resp.status_code, resp.text[:200])
        O.assert_strict_json(resp, f"{method} {url}")
        resp.content.decode("utf-8")
    finally:
        served.close()


@pytest.mark.parametrize("token", TOKEN_TEXTS)
def test_an_edge_to_a_target_field_named_as_a_non_finite_token_is_refused_or_saves(token):
    """``POST /graph/edges {"source_node": "ball", "target_node": "spring",
    "source_field": "position", "target_field": "NaN"}`` answered 201, and
    ``GET /graph`` then answered 500 until the edge was removed: the config
    writes a non-finite float as that text, so its encoder refuses a string
    that spells it.  The edge is a 400 now, where the name is introduced;
    a lookalike is a field like any other."""
    served = serve()
    try:
        before = O.snapshot(served)
        edge = {"source_node": "ball", "target_node": "spring", "source_field": "position"}
        resp = served.client.post("/graph/edges", json={**edge, "target_field": token})
        assert resp.status_code == 400, resp.text
        assert repr(token) in resp.json()["detail"]
        assert not O.differences(before, O.snapshot(served))
        assert served.client.get("/graph").status_code == 200, "GET /graph after the refusal"
        resp = served.client.post("/graph/edges", json={**edge, "target_field": token.lower()})
        assert resp.status_code == 201, resp.text
        assert served.client.get("/graph").status_code == 200, "GET /graph after the edge"
        O.check_accepted_graph(served, f"an edge to {token.lower()!r}", step=False,
                               shapes_may_differ=True)
    finally:
        served.close()


def test_a_checkpoint_of_a_node_named_with_a_nul_loads_or_the_name_is_refused():
    """``POST /graph/nodes {"type": "BallNode", "name": "a\\u0000b",
    "timestep": 0.01}`` answered 201; ``POST /checkpoint/save`` then
    answered 200 and ``POST /checkpoint/load`` of that file 400 ("it is not
    an .npz archive of plain arrays saved by this API"): an archive
    member's name ends at the NUL.  The name is a 400 now, and a save of
    the graph loads."""
    served = serve()
    try:
        before = O.snapshot(served)
        resp = served.client.post("/graph/nodes", json={
            "type": "BallNode", "name": "a\x00b", "timestep": 0.01, "params": {}})
        assert resp.status_code == 400, resp.text
        assert "U+0000" in resp.json()["detail"]
        assert not O.differences(before, O.snapshot(served))
        saved = served.client.post("/checkpoint/save", params={"path": "nul.npz"})
        assert saved.status_code == 200, saved.text
        loaded = served.client.post("/checkpoint/load", params={"path": "nul.npz"})
        assert loaded.status_code == 200, loaded.text
    finally:
        served.close()


def test_what_a_request_started_is_stopped_before_the_graph_is_compared(monkeypatch):
    """``POST /sim/start`` answers and its runner goes on stepping the
    graph; a comparison made while it does is decided by the scheduler
    (three cases of this file failed on the CI runners and passed on a
    workstation).  By the time the accepted graph is looked at, the runner
    and the JAX trace a request started are stopped, and none is left when
    the exchange returns."""
    running_when_compared = []
    compare = O.check_accepted_graph

    def recorded(served, *args, **kwargs):
        running_when_compared.append(left_running(served))
        return compare(served, *args, **kwargs)

    monkeypatch.setattr(O, "check_accepted_graph", recorded)
    for key in ("POST /sim/start", "POST /sim/profile/jax/start"):
        op, served = IN_SCOPE[key], serve()
        try:
            problems, resp = exchange(served, op, request_of(op, SEEDS[key][0], served))
            assert resp.status_code == 200 and not problems, (key, resp.text, problems)
            assert not left_running(served), key
        finally:
            settle(served)
            served.close()
    assert running_when_compared == [[], []]


def test_a_refused_request_that_left_the_runner_running_is_a_problem():
    """The other half, shown able to fail: a route that starts the runner
    and then refuses is reported, whatever the scheduler does, and the
    runner is stopped all the same."""
    served = serve()
    try:
        @served.client.app.post("/probe/start-and-refuse", response_model=None)
        def _start_and_refuse():
            served.server._ensure_runner().start()  # noqa: SLF001
            raise server_module.HTTPException(status_code=409, detail="refused, after starting")

        op = IN_SCOPE["POST /sim/start"]
        probe = Request("POST", "/probe/start-and-refuse", {}, [], ABSENT, {})
        problems, resp = exchange(served, op, probe)
        assert resp.status_code == 409
        assert any("left the runner running" in problem for problem in problems), problems
        assert not left_running(served)
        # ... and the route itself, refused, starts nothing.
        started = request_of(op, SEEDS["POST /sim/start"][0], served)
        started.headers["Origin"] = "http://evil.example"
        problems, resp = exchange(served, op, started, must_refuse=True)
        assert resp.status_code == 403 and not problems, (resp.text, problems)
    finally:
        settle(served)
        served.close()


def test_the_runner_a_request_started_does_not_outlive_its_app():
    """No runner thread survives its server: the app's shutdown stops the
    runner ``POST /sim/start`` started, without the test's help."""
    served = serve()
    try:
        assert served.client.post("/sim/start").status_code == 200
        runner = served.server.runner
        assert runner is not None and runner.is_alive
    finally:
        served.close()
    assert not runner.is_alive and served.server.runner is None


# ---------------------------------------------------------------------------
# The fuzzer
# ---------------------------------------------------------------------------

#: A count no request should be made to run: small, or past every bound.
_COUNTS = st.one_of(st.integers(-16, 16), st.integers(2 ** 40, 2 ** 80),
                    st.integers(-2 ** 80, -2 ** 40))
_TEXT = st.one_of(st.text(max_size=30), st.sampled_from([v for _, v in STRING_VALUES
                                                         if can_be_in_a_url(v)]))
_SCALARS = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True), st.floats(width=32, allow_nan=False),
    _COUNTS, _TEXT, st.booleans(), st.none())
_JSON = st.recursive(_SCALARS, lambda inner: st.one_of(
    st.lists(inner, max_size=5), st.dictionaries(st.text(max_size=8), inner, max_size=4)),
    max_leaves=8)


@st.composite
def generated_requests(draw, op: Operation, seeds: tuple):
    """A seed with up to three of its fields replaced by a drawn value --
    of any JSON type, or one of the battery's -- and sometimes a header:
    ``(seed, [variant, ...])``."""
    seed = draw(st.sampled_from(seeds), label="seed")
    # The seed's own shape is all a variant needs; a body that is a
    # function of the served graph is resolved when the request is built.
    choices: list = []
    for name in op.path_params:
        choices.append(("path", name))
    for name, schema in op.query:
        choices.append(("query", name, _is_count(schema)))
    if op.body is not None and not callable(seed.body):
        choices += [("body", pointer) for pointer in _members(seed.body)] + [("body", ())]
    by_kind = {"number": NUMBER_VALUES, "boolean": NUMBER_VALUES, "string": STRING_VALUES,
               "object": OBJECT_VALUES}
    changes = []
    for _ in range(draw(st.integers(0, min(3, len(choices))), label="fields changed")):
        choice = draw(st.sampled_from(choices), label="field")
        if choice[0] == "path":
            value = draw(_TEXT, label=f"path {choice[1]}")
            changes.append(Variant(f"path {choice[1]}", repr(value),
                                   lambda req, n=choice[1], v=value: req.path.__setitem__(n, v)))
        elif choice[0] == "query":
            value = draw(st.one_of(_COUNTS.map(str), _TEXT) if choice[2] else _TEXT,
                         label=f"query {choice[1]}")
            changes.append(Variant(f"query {choice[1]}", repr(value),
                                   _set_query(choice[1], value)))
        else:
            # Half the time one of the battery's values for a member of
            # this kind (so they are met in combination), else any JSON.
            pointer = choice[1]
            listed = [v for _, v in by_kind.get(
                _kind(_at(seed.body, pointer), _schema_at(op.body, pointer)), ())
                if not isinstance(v, Raw) and v is not ABSENT] if pointer else []
            value = draw(st.one_of(st.sampled_from(listed), _JSON) if listed else _JSON,
                         label=_pointer_text(pointer))
            changes.append(Variant(_pointer_text(pointer), repr(value)[:80],
                                   _put(pointer, value)))
    if draw(st.integers(0, 3), label="a header") == 0:
        changes.append(draw(st.sampled_from(list(header_variants())), label="header"))
    return seed, changes


#: One served graph per ``(route, graph)``, kept while it is as it was
#: built: an example starts from a pristine graph, or Hypothesis could not
#: replay it.
_SERVED: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _close_served():
    yield
    for served in _SERVED.values():
        settle(served)
        served.close()
    _SERVED.clear()


def check_generated_request(op: Operation, seed: Seed, changes: list) -> None:
    key = (op.key, seed.graph)
    served = _SERVED.get(key)
    if served is None:
        served = _SERVED[key] = serve(seed.graph)
    req = request_of(op, seed, served)
    try:
        for change in changes:
            change.apply(req)
    except (KeyError, IndexError, TypeError):
        # A change to a member an earlier change removed: nothing to send.
        return
    note(req.describe())
    try:
        problems, resp = exchange(served, op, req,
                                  must_refuse=any(c.must_refuse(op) for c in changes))
    except NotSent:
        return
    except BaseException:
        del _SERVED[key]
        served.close()
        raise
    note(f"-> {resp.status_code} {resp.text[:200]}")
    if problems or not (resp.status_code >= 400 or op.method == "GET"):
        del _SERVED[key]
        settle(served)
        served.close()
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("key", sorted(IN_SCOPE))
# The floor, not the profile's depth: this is the per-push sibling of the
# slow property below, sized to the time budget (a fresh server after
# every request the route accepts).
@settings(max_examples=EXAMPLES_FLOOR, derandomize=True)
@given(data=st.data())
def test_a_generated_request_is_refused_whole_or_served(key, data):
    """A seed with drawn values in up to three of its fields: no 5xx, a
    refusal changes nothing and names no server path, a 2xx is strict
    JSON.  (JAX's recorder is not run: see :func:`jax_recorder_not_run`.
    The slow sibling below runs it.)"""
    op = IN_SCOPE[key]
    seed, changes = data.draw(generated_requests(op, SEEDS[key]))
    with jax_recorder_not_run():
        check_generated_request(op, seed, changes)


# Per push: tests/property/test_rest_requests_generated_from_the_schema.py::test_a_generated_request_is_refused_whole_or_served
@pytest.mark.slow  # a fresh server after every accepted request: 30 to 90 s a route at the profile's depth
@pytest.mark.parametrize("key", sorted(IN_SCOPE))
@given(data=st.data())
def test_a_generated_request_to_any_route_is_refused_whole_or_served(key, data):
    """The property above at the profile's depth, drawn anew on every run."""
    op = IN_SCOPE[key]
    seed, changes = data.draw(generated_requests(op, SEEDS[key]))
    check_generated_request(op, seed, changes)


# ---------------------------------------------------------------------------
# The headers the HTTP server itself reads, over a real server
# ---------------------------------------------------------------------------

def _raw_exchange(port: int, head: bytes, body: bytes = b"", *,
                  timeout: float = 10.0) -> Optional[int]:
    """Send one raw HTTP/1.1 request and read the status of the reply;
    ``None`` when the server sent none within *timeout* (a request that
    declares more body than it sends is waited for) or closed the
    connection without one (which it may, for a request that is not HTTP)."""
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.sendall(head + b"\r\n\r\n" + body)
        data = b""
        try:
            while b"\r\n" not in data:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        except (socket.timeout, ConnectionError):
            return None
    line = data.split(b"\r\n", 1)[0].split()
    return int(line[1]) if len(line) >= 2 and line[1].isdigit() else None


def test_a_real_server_answers_every_malformed_header_without_a_5xx(tmp_path):
    """The in-process client hands the application whatever it is given.
    Under uvicorn the HTTP parser reads ``Content-Length`` and ``Host``
    first, and its proxy-headers middleware (on by default, trusting
    loopback) rewrites the peer from ``X-Forwarded-For``: so the header
    battery is sent again as raw HTTP/1.1 to a uvicorn server on loopback,
    for a read, a write with a body and a write without one.  No reply is
    a 5xx, a request that is not served changes nothing, and a forwarding
    header, a foreign Host or a foreign Origin on a write is never served."""
    from tests.api import rest_claims_support as S

    chk = S.Check(fn=None, rows=(), bind="loopback", contexts=frozenset(), server_kw={},
                  patch={}, xfail={})
    body = json.dumps({"params": {"stiffness": 40.0}}).encode()
    targets = (("GET", "/graph/state", b""), ("POST", "/sim/step", b""),
               ("PUT", "/graph/params/spring", body))
    problems, sent = [], 0
    with S.loopback_server(chk, tmp_path) as (server, base):
        port = int(base.rsplit(":", 1)[1])
        served = O.Served(server, None, tmp_path, {}, None)   # for snapshot() alone
        for method, path, payload in targets:
            for variant in header_variants():
                name = variant.where.split(" ", 1)[1]
                value = next(v for label, v, _ in HEADER_VALUES[name] if label == variant.what)
                headers = {"Host": f"127.0.0.1:{port}", "Connection": "close"}
                if payload:
                    headers.update({"Content-Type": "application/json",
                                    "Content-Length": str(len(payload))})
                headers[name] = value
                head = f"{method} {path} HTTP/1.1".encode() + b"".join(
                    b"\r\n" + k.encode() + b": " + v.encode("latin-1") for k, v in headers.items())
                before = O.snapshot(served)
                # A request that declares more body than it sends is not
                # waited for long: a route that reads its body never gets it.
                declared = headers.get("Content-Length", "0")
                short = declared.isascii() and declared.isdigit() and int(declared) > len(payload)
                status = _raw_exchange(port, head, payload, timeout=1.0 if short else 10.0)
                sent += 1
                what = f"{method} {path} with {name}: {variant.what}"
                op = Operation(method, path, (), (), None, True)
                if status is not None and status >= 500:
                    problems.append(f"{what} -> {status}")
                if status is None or status >= 400 or method == "GET":
                    found = O.differences(before, O.snapshot(served))
                    if found:
                        problems.append(f"{what} -> {status} and the server changed: "
                                        + "; ".join(found)[:400])
                elif variant.must_refuse(op):
                    problems.append(f"{what} -> {status}: served, where the Host, Origin or "
                                    "forwarding rule refuses")
    assert sent == 3 * sum(len(v) for v in HEADER_VALUES.values())
    assert not problems, f"{len(problems)} problem(s) in {sent} requests:\n" + "\n".join(problems)
