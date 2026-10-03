"""The bearer-token, ``Origin`` and ``Host`` rules hold at the edges of
what the REST guide says about them.

The rows these pin are in ``docs/validation/rest_runpod_claims.yaml``
(``REST-0xx``, area ``auth`` and ``host-origin``).  The neighbouring
suites check the comfortable middle: an anonymous ``GET`` on a public
bind, a foreign ``Origin`` on a loopback bind, a rebound ``Host`` on a
handful of routes.  These go to the edges:

* every spelling of a loopback bind, and of a bind that is not one, and
  routable peers that are not plain IPv4 (an IPv4-mapped IPv6 address, a
  scoped link-local one);
* the exempt paths are exact: a trailing slash, ``HEAD`` or ``OPTIONS``
  is not exempt;
* the ``Origin`` rule still applies where the token is demanded and
  presented, over HTTP and over a WebSocket handshake;
* the ``Host`` rule covers every route, ``/healthz`` and ``/viz/*``
  included, and ignores an ``allowed_hosts`` entry's port;
* an unauthenticated caller is refused before its body is read, so an
  oversized body is a 401, not a 413;
* ``/healthz`` answers while another request holds the graph;
* the generated token's file is the owner's alone even when the file
  already existed, and even for a reader who had the old file open.

Nothing here can reach a cloud provider: ``HOME`` is an empty directory,
cloud credentials are unset and every launcher raises
(:func:`tests.property.differential.no_cloud_launch`), and
``/cloud/launch`` is left out of every enumeration.
"""

from __future__ import annotations

import os
import stat
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from tests._loopback_client import LoopbackTestClient as TestClient
from starlette.routing import Route

from maddening.api import server as server_module
from maddening.api.auth import TOKEN_FILE_ENV, UNAUTHENTICATED_PATHS, APIAuth
from maddening.api.server import SimulationServer, _rebinding_refusal
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from tests.property.differential import no_cloud_launch

TOKEN = "claims-test-token-not-a-credential"
REGISTRY = {"SpringDamperNode": SpringDamperNode}
FOREIGN = "https://evil.example"
LOOPBACK_PEER = ("127.0.0.1", 51234)
#: Routes a test here never sends a request to, whatever it enumerates.
NEVER_REACHED = {"/cloud/launch"}


@pytest.fixture(scope="module", autouse=True)
def _offline():
    with no_cloud_launch():
        yield


def _graph() -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="spring", timestep=0.01))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


def _server(bind_host: str, **kwargs) -> SimulationServer:
    return SimulationServer(REGISTRY, graph_manager=_graph(), bind_host=bind_host,
                            api_token=TOKEN, **kwargs)


def _bearer() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


def _routes(app) -> list[tuple[str, str, str]]:
    """``(method, concrete path, template)`` of every HTTP route but the
    ones never reached, with path parameters filled in."""
    out = []
    for route in app.routes:
        if not isinstance(route, Route) or route.path in NEVER_REACHED:
            continue
        path = route.path
        for name in getattr(route, "param_convertors", {}) or {}:
            path = path.replace("{" + name + "}", "spring")
        for method in sorted(route.methods or set()):
            if method not in {"HEAD", "OPTIONS"}:
                out.append((method, path, route.path))
    return out


def _structure(server: SimulationServer) -> tuple:
    gm = server.gm
    return (list(gm._nodes), list(gm._edges),
            {k: dict(v) for k, v in gm._state.items() if k != "_meta"},
            dict(gm._nodes["spring"].node.params))


# ---------------------------------------------------------------------------
# Which binds demand the token, and which peers the backstop challenges
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bind", ["127.0.0.1", "127.0.0.2", "localhost", "LOCALHOST",
                                  "::1", "[::1]", "::ffff:127.0.0.1"])
def test_every_loopback_spelling_of_the_bind_serves_without_a_token(bind):
    server = _server(bind)
    assert server.auth.enforced is False
    client = TestClient(server.create_app())
    assert client.get("/graph").status_code == 200
    assert client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize("bind", ["0.0.0.0", "::", "", " ", "10.0.0.4", "203.0.113.5",
                                  "example.org", "localhost.example.org"])
def test_every_other_bind_demands_the_token_and_hides_the_docs(bind):
    """An unknown or empty address fails closed (``is_loopback``)."""
    server = _server(bind)
    assert server.auth.enforced is True
    client = TestClient(server.create_app())
    assert client.get("/graph").status_code == 401
    assert client.get("/graph", headers=_bearer()).status_code == 200
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path, headers=_bearer()).status_code == 404, path


@pytest.mark.parametrize("peer", ["::ffff:203.0.113.5", "fe80::1%eth0", "10.0.0.4",
                                  "2001:db8::7"])
def test_a_routable_peer_of_any_spelling_is_challenged_on_a_loopback_bind(peer):
    client = TestClient(_server("127.0.0.1").create_app(), client=(peer, 40000))
    assert client.get("/graph").status_code == 401
    assert client.get("/graph", headers=_bearer()).status_code == 200


def test_a_routable_peer_is_challenged_on_a_websocket_handshake_too():
    client = TestClient(_server("127.0.0.1").create_app(), client=("203.0.113.5", 40000))
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/state"):
            pass
    with client.websocket_connect("/ws/state", headers=_bearer()):
        pass


# ---------------------------------------------------------------------------
# The exempt paths are exact
# ---------------------------------------------------------------------------

def test_only_the_exact_exempt_paths_answer_without_the_token():
    client = TestClient(_server("0.0.0.0").create_app())
    for path in sorted(UNAUTHENTICATED_PATHS):
        assert client.get(path).status_code == 200, path
        assert client.get(path + "/").status_code == 401, path + "/"
    for method in ("HEAD", "OPTIONS"):
        assert client.request(method, "/graph").status_code == 401, method


# ---------------------------------------------------------------------------
# The Origin rule where the token is demanded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bind", ["127.0.0.1", "0.0.0.0"])
def test_every_state_changing_route_refuses_a_foreign_origin(bind):
    """On either bind, and with the token presented where it is demanded:
    the Origin rule is the server's own, not a loopback special case."""
    server = _server(bind)
    client = TestClient(server.create_app(), raise_server_exceptions=False)
    before = _structure(server)
    checked = 0
    for method, path, template in _routes(client.app):
        if method not in {"POST", "PUT", "PATCH", "DELETE"}:
            continue
        resp = client.request(method, path, headers={**_bearer(), "Origin": FOREIGN,
                                                     "Content-Type": "text/plain"})
        assert resp.status_code == 403, (method, template, resp.status_code, resp.text)
        checked += 1
    assert checked >= 20, f"only {checked} state-changing routes found"
    assert _structure(server) == before


def test_a_state_stream_refuses_a_foreign_origin_that_holds_the_token():
    client = TestClient(_server("0.0.0.0").create_app())
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/state", headers={**_bearer(), "Origin": FOREIGN}):
            pass
    # The same handshake from no Origin at all (a script) is served.
    with client.websocket_connect("/ws/state", headers=_bearer()):
        pass


# ---------------------------------------------------------------------------
# The Host rule covers every route
# ---------------------------------------------------------------------------

def test_a_foreign_host_is_refused_on_every_route_the_exempt_ones_included():
    server = _server("127.0.0.1")
    client = TestClient(server.create_app(), base_url="http://attacker.example:8000",
                        client=LOOPBACK_PEER, raise_server_exceptions=False)
    before = _structure(server)
    templates = set()
    for method, path, template in _routes(client.app):
        resp = client.request(method, path)
        assert resp.status_code == 403, (method, template, resp.status_code, resp.text)
        templates.add(template)
    assert UNAUTHENTICATED_PATHS <= templates
    assert _structure(server) == before


def test_a_request_with_no_host_header_is_not_refused_for_its_host():
    auth = APIAuth(bind_host="127.0.0.1", token=TOKEN, environ={})
    assert _rebinding_refusal(auth, None, frozenset(), authenticated=False) is None
    assert _rebinding_refusal(auth, "attacker.example", frozenset(),
                              authenticated=False) is not None


def test_an_allowed_host_is_matched_whatever_port_either_side_names():
    server = _server("127.0.0.1", allowed_hosts=["sim.lab.example:9999"])
    client = TestClient(server.create_app(), base_url="http://sim.lab.example:1234",
                        client=LOOPBACK_PEER)
    assert client.get("/graph").status_code == 200
    assert client.get("/graph", headers={"Host": "sim.lab.example"}).status_code == 200


@pytest.mark.parametrize("entry", ["http://x", "a b", "[::1", "x:y", "host:80:90", "",
                                   "*.example.com", "evil.example/path", "user@host",
                                   "host?x=1", "host#frag", "\thost", "host ", "b\u00fccher.example",
                                   "-lead.example", "x" * 64 + ".example"])
def test_an_allowed_hosts_entry_that_is_not_a_host_name_is_refused(entry):
    """``allowed_hosts`` took anything ``Host``-shaped: ``"a b"``,
    ``"*.example.com"`` (no wildcard is implemented), ``"user@host"``,
    ``"evil.example/path"``, ``" host"`` -- none of which a browser's
    ``Host`` header can match, so the server was silently configured to
    serve a name it never would."""
    with pytest.raises(ValueError, match="is not a host name"):
        _server("127.0.0.1", allowed_hosts=[entry])


@pytest.mark.parametrize("entry", ["proxy.internal:8443", "Alias.Local.", "localhost",
                                   "10.0.0.7", "[::1]:8000", "[fe80::1]", "my_host.lan",
                                   "a" * 63 + ".example", "xn--bcher-kva.example"])
def test_an_allowed_hosts_entry_that_is_a_host_name_is_taken(entry):
    assert _server("127.0.0.1", allowed_hosts=[entry]) is not None


@pytest.mark.parametrize("token", ["abc ", " abc", "abc\n", "\tabc"])
def test_a_token_with_whitespace_around_it_is_refused_at_construction(token):
    """A client's ``Authorization: Bearer`` credential is read stripped, so
    a configured token with whitespace around it could never be presented:
    every client was refused, with nothing said at start-up.  It is a
    configuration error, as a blank one is -- from ``token=`` and from the
    environment alike."""
    with pytest.raises(ValueError, match="whitespace before or after it"):
        APIAuth(bind_host="0.0.0.0", token=token, environ={})
    with pytest.raises(ValueError, match="whitespace before or after it"):
        APIAuth(bind_host="0.0.0.0", environ={"MADDENING_API_TOKEN": token})
    # Whitespace inside a token is the operator's choice, and works.
    inner = APIAuth(bind_host="0.0.0.0", token="a b", environ={})
    client = TestClient(SimulationServer(REGISTRY, graph_manager=_graph(), bind_host="0.0.0.0",
                                         api_token="a b").create_app())
    assert inner.verify("a b")
    assert client.get("/graph", headers={"Authorization": "Bearer a b"}).status_code == 200


def test_the_token_file_is_read_from_the_environment_the_instance_was_given(tmp_path,
                                                                           monkeypatch):
    """``APIAuth(environ=...)`` read ``MADDENING_API_TOKEN_FILE`` from
    ``os.environ``: a path in *environ* was never written to, and an
    explicit empty *environ* still wrote the process's path."""
    given, process = tmp_path / "given", tmp_path / "process"
    monkeypatch.setenv(TOKEN_FILE_ENV, str(process))
    monkeypatch.delenv("MADDENING_API_TOKEN", raising=False)
    auth = APIAuth(bind_host="0.0.0.0", environ={TOKEN_FILE_ENV: str(given)})
    assert auth.announce(8000) is True
    assert given.read_text() == auth.token + "\n" and not process.exists()
    empty = APIAuth(bind_host="0.0.0.0", environ={})
    assert empty.announce(8000) is True
    assert not process.exists()
    # With no environ the process's environment is read, as documented.
    default = APIAuth(bind_host="0.0.0.0")
    default.announce(8000)
    assert process.read_text() == default.token + "\n"


# ---------------------------------------------------------------------------
# Order: who is refused before what
# ---------------------------------------------------------------------------

def test_an_anonymous_oversized_body_is_refused_for_its_credential_first(monkeypatch):
    monkeypatch.setattr(server_module, "MAX_REQUEST_BODY_BYTES", 1000)
    client = TestClient(_server("0.0.0.0").create_app(), raise_server_exceptions=False)
    body = b"x" * 5000
    headers = {"Content-Type": "application/json"}
    assert client.put("/graph/state/spring", content=body, headers=headers).status_code == 401
    assert client.put("/graph/state/spring", content=body,
                      headers={**headers, **_bearer()}).status_code == 413


def test_healthz_answers_while_another_request_holds_the_graph():
    server = _server("127.0.0.1")
    client = TestClient(server.create_app())
    assert server._graph_lock.acquire()
    try:
        t0 = time.monotonic()
        resp = client.get("/healthz")
        elapsed = time.monotonic() - t0
    finally:
        server._graph_lock.release()
    assert resp.status_code == 200 and resp.json()["status"] == "ok"
    assert elapsed < 2.0, f"/healthz waited {elapsed:.2f} s for the graph"


# ---------------------------------------------------------------------------
# The token file
# ---------------------------------------------------------------------------

def test_a_new_token_file_is_written_with_mode_0600(tmp_path, monkeypatch):
    path = tmp_path / "token"
    auth = APIAuth(bind_host="0.0.0.0", environ={TOKEN_FILE_ENV: str(path)})
    assert auth.announce(8000) is True
    assert path.read_text() == auth.token + "\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_token_file_that_already_existed_is_left_readable_by_its_owner_only(
        tmp_path, monkeypatch):
    """The README: "point MADDENING_API_TOKEN_FILE at a path on a mounted
    volume and the generated token is written there with mode 0600".  A
    path on a volume is often a file that is already there (a placeholder,
    the last run's token): ``os.open(..., O_CREAT, 0o600)`` applies the
    mode only to a file it creates, so the new token lands in a
    world-readable file."""
    path = tmp_path / "token"
    path.write_text("the last run's token\n")
    path.chmod(0o644)
    auth = APIAuth(bind_host="0.0.0.0", environ={TOKEN_FILE_ENV: str(path)})
    assert auth.announce(8000) is True
    assert path.read_text() == auth.token + "\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600, oct(stat.S_IMODE(path.stat().st_mode))


def test_a_reader_who_had_the_old_token_file_open_cannot_read_the_new_token(tmp_path, monkeypatch):
    """REST-016: the token goes into a new 0600 file moved over the path, not into the
    old one.  A ``chmod`` of the old file would leave a reader who opened it while it was
    world-readable able to read whatever is written into it next."""
    path = tmp_path / "token"
    path.write_text("the last run's token\n")
    path.chmod(0o644)
    with open(path, encoding="utf-8") as reader:          # opened while 0644
        auth = APIAuth(bind_host="0.0.0.0", environ={TOKEN_FILE_ENV: str(path)})
        assert auth.announce(8000) is True
        assert reader.read() == "the last run's token\n"
    assert path.read_text() == auth.token + "\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_token_file_path_that_is_a_link_writes_where_it_points(tmp_path, monkeypatch):
    """REST-016: a symbolic link at the path is followed -- a volume's path stays where
    the operator pointed it -- and the file it names becomes the owner's alone."""
    volume = tmp_path / "volume"
    volume.mkdir()
    real = volume / "token"
    real.write_text("old\n")
    real.chmod(0o664)
    link = tmp_path / "token"
    link.symlink_to(real)
    auth = APIAuth(bind_host="0.0.0.0", environ={TOKEN_FILE_ENV: str(link)})
    auth.announce(8000)
    assert link.is_symlink() and real.read_text() == auth.token + "\n"
    assert stat.S_IMODE(real.stat().st_mode) == 0o600
    assert not [p for p in volume.iterdir() if p.name != "token"], "a temporary file was left"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a read-only directory")
def test_a_token_file_that_cannot_be_written_is_left_alone_and_logged(tmp_path, monkeypatch,
                                                                      caplog):
    """REST-016: "The server does not fail if it cannot write that file; it logs the
    failure and carries on" -- and writes the token into nothing it could not make the
    owner's alone."""
    directory = tmp_path / "ro"
    directory.mkdir()
    path = directory / "token"
    path.write_text("old\n")
    path.chmod(0o644)
    directory.chmod(0o555)
    try:
        auth = APIAuth(bind_host="0.0.0.0", environ={TOKEN_FILE_ENV: str(path)})
        with caplog.at_level("WARNING", logger="maddening.api.auth"):
            assert auth.announce(8000) is True
        assert path.read_text() == "old\n"
        assert any("Could not write the generated API token" in r.getMessage()
                   for r in caplog.records)
    finally:
        directory.chmod(0o755)


#: Characters ``str.isdigit`` accepts that ``int`` refuses -- the latin-1
#: superscripts a client can send as obs-text -- and other non-ASCII digits.
NON_ASCII_DIGITS = ["²", "³", "¹", "1²", "٣", "１"]


@pytest.mark.parametrize("port", NON_ASCII_DIGITS)
def test_a_host_port_of_non_ascii_digits_is_not_a_port(port):
    from maddening.api.server import _host_and_port, _host_name
    for host in (f"localhost:{port}", f"[::1]:{port}", f"127.0.0.1:{port}"):
        assert _host_and_port(host) is None, host
        assert _host_name(host) is None, host
    assert _host_and_port("localhost:12") == ("localhost", 12)
    assert _host_name("[::1]:8000") == "::1"


def _asgi(app, scope_type: str, host: bytes, *, origin: bytes = b"http://evil.example"):
    """One request or handshake straight into the app, its Host and Origin
    as raw latin-1 bytes (an in-process client rewrites the Host from its
    URL); the messages the app sent."""
    import asyncio

    async def call():
        scope = {"type": scope_type, "asgi": {"version": "3.0"}, "http_version": "1.1",
                 "scheme": "http" if scope_type == "http" else "ws", "path": "/sim/step"
                 if scope_type == "http" else "/ws/state", "query_string": b"",
                 "root_path": "", "headers": [(b"host", host), (b"origin", origin)],
                 "client": LOOPBACK_PEER, "server": ("127.0.0.1", 8000), "subprotocols": []}
        if scope_type == "http":
            scope["method"] = "POST"
        scope["raw_path"] = scope["path"].encode()
        sent, first = [], [True]

        async def receive():
            if scope_type == "websocket":
                if first[0]:
                    first[0] = False
                    return {"type": "websocket.connect"}
                return {"type": "websocket.disconnect", "code": 1000}
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        await app(scope, receive, send)
        return sent
    return asyncio.run(call())


@pytest.mark.parametrize("port", ["²", "³", "¹"])
def test_a_host_port_of_a_latin1_superscript_is_refused_not_a_500(port):
    """``Host: localhost:\\u00b2`` raised ``ValueError`` in the origin check:
    a 500 on HTTP and on the WebSocket handshake.  It is not a name of this
    server, so the Host rule refuses it: 403, and a closed handshake."""
    app = _server("127.0.0.1").create_app()
    host = f"localhost:{port}".encode("latin-1")
    sent = _asgi(app, "http", host)
    assert next(m["status"] for m in sent if m["type"] == "http.response.start") == 403
    sent = _asgi(app, "websocket", host)
    assert sent and sent[0]["type"] == "websocket.close", sent
    # The control: the same request with an ASCII port reaches the origin check.
    sent = _asgi(app, "http", b"localhost:2")
    assert next(m["status"] for m in sent if m["type"] == "http.response.start") == 403


@pytest.mark.parametrize("token", ["пароль-0123456789",
                                   "pässwört-0123456789", "a\tb", "a\x7fb",
                                   "a b", "\U0001f511-key"])
def test_a_token_no_client_can_present_is_refused_at_construction(token):
    """A token outside printable ASCII was accepted, logged nothing, and
    then refused every client: a header's bytes are decoded as latin-1, so
    curl's UTF-8 bytes arrive as other characters, httpx and requests will
    not encode the header at all, and a latin-1 token matched only a client
    sending latin-1 bytes.  Refused like a blank token, from ``token=`` and
    from the environment, naming the character and not the token."""
    for kwargs in ({"token": token, "environ": {}},
                   {"environ": {"MADDENING_API_TOKEN": token}}):
        with pytest.raises(ValueError, match="not printable ASCII") as info:
            APIAuth(bind_host="0.0.0.0", **kwargs)
        assert "no client could ever present this token" in str(info.value)
        assert token not in str(info.value)


@pytest.mark.parametrize("token", ["a b", "!#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", "~" * 64])
def test_a_printable_ascii_token_is_taken_and_can_be_presented(token):
    server = SimulationServer(REGISTRY, graph_manager=_graph(), bind_host="0.0.0.0",
                              api_token=token)
    client = TestClient(server.create_app())
    assert client.get("/graph", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_a_token_file_failure_points_at_the_token_logged_above_it(tmp_path, caplog):
    """The warning for a token file that cannot be written says the token
    "is in the log line above"; the token used to be logged after it."""
    path = tmp_path / "missing-dir" / "token"
    auth = APIAuth(bind_host="0.0.0.0", environ={TOKEN_FILE_ENV: str(path)})
    with caplog.at_level("INFO", logger="maddening.api.auth"):
        assert auth.announce(8000) is True
    messages = [r.getMessage() for r in caplog.records]
    token_at = next(i for i, m in enumerate(messages) if auth.token in m)
    failure_at = next(i for i, m in enumerate(messages)
                      if "Could not write the generated API token" in m)
    assert token_at < failure_at, messages
    assert "log line above" in messages[failure_at]


def test_a_comma_joined_subprotocol_entry_still_carries_the_browser_token():
    """uvicorn 0.50.0 hands the app ``Sec-WebSocket-Protocol`` as one
    comma-joined entry, not the list ASGI specifies; the bearer carrier and
    ``maddening.v1`` were then not found, and the handshake was refused."""
    import asyncio

    from maddening.api.auth import bearer_from_subprotocols, websocket_credentials
    from maddening.api.server import _offered_subprotocols

    joined = ", ".join(websocket_credentials(TOKEN))
    assert bearer_from_subprotocols([joined]) == TOKEN
    assert bearer_from_subprotocols(websocket_credentials(TOKEN)) == TOKEN
    assert _offered_subprotocols({"subprotocols": [joined]})[-1] == "maddening.v1"
    app = _server("0.0.0.0").create_app()

    async def handshake(subprotocols):
        scope = {"type": "websocket", "asgi": {"version": "3.0"}, "scheme": "ws",
                 "path": "/ws/state", "raw_path": b"/ws/state", "query_string": b"",
                 "root_path": "", "headers": [(b"host", b"127.0.0.1")],
                 "client": LOOPBACK_PEER, "server": ("127.0.0.1", 8000),
                 "subprotocols": subprotocols}
        sent, inbox = [], [{"type": "websocket.connect"}]

        async def receive():
            if inbox:
                return inbox.pop(0)
            await asyncio.sleep(0.05)
            return {"type": "websocket.disconnect", "code": 1000}

        async def send(message):
            sent.append(message)

        await asyncio.wait_for(app(scope, receive, send), timeout=30)
        return sent

    accepted = asyncio.run(handshake([joined]))
    assert accepted[0]["type"] == "websocket.accept", accepted
    assert accepted[0].get("subprotocol") == "maddening.v1"
    refused_ = asyncio.run(handshake(["maddening.bearer.d3Jvbmc, maddening.v1"]))
    assert refused_[0]["type"] == "websocket.close", refused_
