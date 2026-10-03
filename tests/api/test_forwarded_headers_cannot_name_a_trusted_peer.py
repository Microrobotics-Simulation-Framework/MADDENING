"""A forwarding header cannot make a request look local, nor skip the Host check.

The peer the API's authentication sees is ``scope["client"]`` as the ASGI
server reports it.  uvicorn's defaults (``proxy_headers=True``,
``forwarded_allow_ips="127.0.0.1"``) rewrite it from ``X-Forwarded-For`` for
every request that arrives over loopback -- which is how a DNS-rebinding
page's requests arrive.  With ``X-Forwarded-For: x`` the peer was the
string ``"x"``: the Host check was asked only of a peer that was an IP
address, so it was skipped, and the peer backstop asked a token only of a
routable IP, so none was asked.  A rebound page (``X-Forwarded-For`` is not
a forbidden header for ``fetch()``) could step, read, add nodes and save
checkpoints.  Under ``forwarded_allow_ips="*"`` a routable peer's 401 became
a 200 the same way.

Now no decision keys on the peer's *name*: a peer that is not a loopback IP
literal is unknown and must present the token, a request that carries
``X-Forwarded-For`` or ``Forwarded`` must present it too, and on a loopback
bind the Host check is asked of every request without a valid token.  The
in-process client gets no allowance in the library; the tests that drive
the API as a local process build a loopback client
(``tests/_loopback_client.py``).  The library's own launch paths pass
``proxy_headers=False``.
"""

from __future__ import annotations

import ast
import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx", reason="the loopback tests drive a real server over HTTP")
uvicorn = pytest.importorskip("uvicorn", reason="the loopback tests run the app under uvicorn")

from fastapi.testclient import TestClient as StarletteTestClient  # noqa: E402
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware  # noqa: E402

import maddening  # noqa: E402
from maddening.api.auth import APIAuth  # noqa: E402
from maddening.api.server import SimulationServer  # noqa: E402
from maddening.core.graph_manager import GraphManager  # noqa: E402
from maddening.nodes import BallNode  # noqa: E402

TOKEN = "forwarded-header-test-token-0123456789"
BEARER = {"Authorization": f"Bearer {TOKEN}"}
#: The peers a forged header names: a non-address, the in-process client's
#: own name, the loopback literals and the loopback name.
FORGED = ["x", "testclient", "127.0.0.1", "::1", "localhost"]
#: The same, as RFC 7239 ``Forwarded`` values.
FORGED_FORWARDED = ["for=x", "for=127.0.0.1", 'for="[::1]"', "for=unknown"]

_SERVER = r'''
import sys, warnings
warnings.simplefilter("ignore")
import uvicorn
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode

gm = GraphManager()
gm.add_node(BallNode("ball", 0.01, initial_position=1.0))
gm.compile()
app = SimulationServer({"BallNode": BallNode}, graph_manager=gm, bind_host="127.0.0.1",
                       api_token=sys.argv[2], checkpoint_root=sys.argv[3]).create_app()
# uvicorn's defaults: proxy_headers=True, forwarded_allow_ips="127.0.0.1".
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
'''


@pytest.fixture(scope="module")
def loopback(tmp_path_factory):
    """A real uvicorn on 127.0.0.1 with uvicorn's default proxy settings,
    stopped by its own PID."""
    tmp = tmp_path_factory.mktemp("forwarded")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    script = tmp / "serve.py"
    script.write_text(_SERVER)
    log = open(tmp / "server.log", "w")
    proc = subprocess.Popen([sys.executable, str(script), str(port), TOKEN, str(tmp)],
                            stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                if httpx.get(base + "/healthz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert proc.poll() is None, (tmp / "server.log").read_text()
            time.sleep(0.1)
        else:
            raise AssertionError(f"the server did not start:\n{(tmp / 'server.log').read_text()}")
        yield port, tmp
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        log.close()


def _position(port: int) -> float:
    """The ball's position, read by a direct loopback request."""
    resp = httpx.get(f"http://127.0.0.1:{port}/graph/state", timeout=30)
    assert resp.status_code == 200, resp.text
    return resp.json()["ball"]["position"]


def _rebound(port: int, **extra: str) -> dict:
    """The headers a rebound page's ``fetch()`` sends: its own name as Host
    and Origin, plus *extra*."""
    host = f"attacker.example:{port}"
    return {"Host": host, "Origin": f"http://{host}", **extra}


@pytest.mark.parametrize("header, value", [("X-Forwarded-For", v) for v in FORGED]
                         + [("Forwarded", v) for v in FORGED_FORWARDED])
def test_a_rebound_page_with_a_forwarding_header_is_refused(loopback, header, value):
    port, tmp = loopback
    before = _position(port)
    base = f"http://127.0.0.1:{port}"
    headers = _rebound(port, **{header: value})
    for method, path in [("POST", "/sim/step"), ("GET", "/graph/state"), ("GET", "/graph"),
                         ("POST", "/checkpoint/save?path=rebound.npz")]:
        resp = httpx.request(method, base + path, headers=headers, timeout=30)
        assert resp.status_code in (401, 403), (method, path, resp.status_code, resp.text)
    resp = httpx.post(base + "/graph/nodes", headers=headers, timeout=30,
                      json={"type": "BallNode", "name": "planted", "timestep": 0.01,
                            "params": {}})
    assert resp.status_code in (401, 403), resp.text
    assert _position(port) == before
    assert not (tmp / "rebound.npz").exists()
    assert "planted" not in [n["name"] for n in httpx.get(base + "/graph", timeout=30)
                             .json()["nodes"]]


@pytest.mark.parametrize("value", FORGED)
def test_a_forwarding_header_needs_the_token_even_naming_this_machine(loopback, value):
    """A local tool that claims a proxy forwarded its request is not a direct
    loopback connection: 401, whatever peer the header names."""
    port, _tmp = loopback
    before = _position(port)
    resp = httpx.post(f"http://127.0.0.1:{port}/sim/step",
                      headers={"X-Forwarded-For": value}, timeout=30)
    assert resp.status_code == 401, resp.text
    assert "forwarding header" in resp.json()["detail"]
    assert _position(port) == before


def test_the_token_still_authenticates_a_forwarded_request(loopback):
    """The documented token path: with the bearer token a forwarded
    request is served, and so is one whose Host is not a loopback name."""
    port, _tmp = loopback
    base = f"http://127.0.0.1:{port}"
    before = _position(port)
    resp = httpx.post(base + "/sim/step", headers={"X-Forwarded-For": "x", **BEARER},
                      timeout=30)
    assert resp.status_code == 200, resp.text
    assert _position(port) != before
    resp = httpx.get(base + "/graph", headers={**_rebound(port), **BEARER}, timeout=30)
    assert resp.status_code == 200, resp.text


def test_the_loopback_flow_without_a_forwarding_header_is_unchanged(loopback):
    port, _tmp = loopback
    base = f"http://127.0.0.1:{port}"
    assert httpx.post(base + "/sim/step", timeout=30).status_code == 200
    resp = httpx.post(base + "/sim/step", headers=_rebound(port), timeout=30)
    assert resp.status_code == 403, resp.text
    assert "allowed_hosts" in resp.json()["detail"]


def _handshake(port: int, headers: dict) -> str:
    """The status line uvicorn answers a WebSocket handshake with."""
    lines = [f"GET /ws/state HTTP/1.1", "Upgrade: websocket", "Connection: Upgrade",
             "Sec-WebSocket-Version: 13", "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ=="]
    lines += [f"{k}: {v}" for k, v in headers.items()]
    with socket.create_connection(("127.0.0.1", port), timeout=30) as conn:
        conn.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        return conn.recv(4096).split(b"\r\n", 1)[0].decode("latin-1")


@pytest.mark.parametrize("value", FORGED)
def test_a_forwarded_websocket_handshake_is_refused(loopback, value):
    port, _tmp = loopback
    rebound = _handshake(port, _rebound(port, **{"X-Forwarded-For": value}))
    assert " 101 " not in rebound, rebound
    local = _handshake(port, {"Host": f"127.0.0.1:{port}", "X-Forwarded-For": value})
    assert " 101 " not in local, local


def test_a_websocket_handshake_with_the_token_or_no_forwarding_header_is_accepted(loopback):
    port, _tmp = loopback
    assert " 101 " in _handshake(port, {"Host": f"127.0.0.1:{port}"})
    assert " 101 " in _handshake(port, {"Host": f"127.0.0.1:{port}", "X-Forwarded-For": "x",
                                        **BEARER})


# ---------------------------------------------------------------------------
# The backstop under a proxy configuration that trusts every peer
# ---------------------------------------------------------------------------

def _graph() -> GraphManager:
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=1.0))
    gm.compile()
    return gm


def _asgi_status(app, *, client, headers) -> int:
    async def call():
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": "/sim/step",
                 "raw_path": b"/sim/step", "query_string": b"", "root_path": "",
                 "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
                 "client": client, "server": ("192.0.2.1", 8000)}
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        await app(scope, receive, send)
        return next(m["status"] for m in sent if m["type"] == "http.response.start")
    return asyncio.run(call())


@pytest.mark.parametrize("value", FORGED)
def test_a_routable_peer_cannot_forge_its_way_past_the_backstop(value):
    """bind_host not given (the server believes it is on loopback), uvicorn
    trusting every proxy: a routable peer naming any peer is still a 401,
    and the token still serves it."""
    server = SimulationServer({"BallNode": BallNode}, graph_manager=_graph(),
                              api_token=TOKEN)
    assert not server.auth.enforced
    app = ProxyHeadersMiddleware(server.create_app(), trusted_hosts="*")
    routable = ("203.0.113.9", 40000)
    for host in ("192.0.2.1:8000", "127.0.0.1"):
        headers = {"Host": host, "X-Forwarded-For": value}
        assert _asgi_status(app, client=routable, headers=headers) == 401, (host, value)
        assert _asgi_status(app, client=routable, headers={**headers, **BEARER}) == 200
    assert _asgi_status(app, client=routable, headers={"Host": "127.0.0.1"}) == 401


# ---------------------------------------------------------------------------
# A peer that is not a loopback address is unknown
# ---------------------------------------------------------------------------

def test_starlettes_own_in_process_client_must_present_the_token():
    """``TestClient``'s defaults -- peer ``"testclient"``, Host
    ``testserver`` -- are a name and a foreign host: the token is asked of
    them, and with it they are served."""
    server = SimulationServer({"BallNode": BallNode}, graph_manager=_graph(),
                              bind_host="127.0.0.1", api_token=TOKEN)
    anonymous = StarletteTestClient(server.create_app(), raise_server_exceptions=False)
    for method, path in [("GET", "/graph"), ("POST", "/sim/step"), ("GET", "/graph/state")]:
        resp = anonymous.request(method, path)
        assert resp.status_code == 401, (path, resp.text)
        assert "server.auth.token" in resp.json()["detail"]
    authed = StarletteTestClient(server.create_app(), headers=BEARER)
    assert authed.post("/sim/step").status_code == 200
    with authed.websocket_connect("/ws/state") as ws:
        assert ws.accepted_subprotocol is None


def test_a_connection_with_no_peer_address_must_present_the_token():
    """A Unix-socket connection reports no peer: unknown, so the token."""
    server = SimulationServer({"BallNode": BallNode}, graph_manager=_graph(),
                              bind_host="127.0.0.1", api_token=TOKEN)
    app = server.create_app()
    assert _asgi_status(app, client=None, headers={"Host": "127.0.0.1"}) == 401
    assert _asgi_status(app, client=None, headers={"Host": "127.0.0.1", **BEARER}) == 200


@pytest.mark.parametrize("peer, required", [
    ("127.0.0.1", False), ("127.8.9.10", False), ("::1", False), ("::ffff:127.0.0.1", False),
    ("x", True), ("testclient", True), ("localhost", True), ("", True), (None, True),
    ("203.0.113.5", True), ("10.0.0.1", True), ("::1x", True),
])
def test_only_a_loopback_ip_literal_is_served_without_a_token_on_a_loopback_bind(peer,
                                                                                 required):
    auth = APIAuth(bind_host="127.0.0.1", token=TOKEN, environ={})
    assert auth.required_for_peer(peer) is required


@pytest.mark.parametrize("headers", [{"x-forwarded-for": "127.0.0.1"},
                                     {"forwarded": "for=127.0.0.1"},
                                     {"x-forwarded-for": ""}])
def test_a_forwarding_header_makes_even_a_loopback_peer_present_the_token(headers):
    auth = APIAuth(bind_host="127.0.0.1", token=TOKEN, environ={})
    assert auth._required_for_request("127.0.0.1", headers)
    assert not auth._required_for_request("127.0.0.1", {"x-real-thing": "1"})


# ---------------------------------------------------------------------------
# The library's own launch paths
# ---------------------------------------------------------------------------

def _uvicorn_launches(tree: ast.AST):
    """Every ``uvicorn.run(...)`` / ``uvicorn.Config(...)`` call in *tree*."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("run", "Config")
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "uvicorn"):
            yield node


def test_every_launch_path_the_library_ships_passes_proxy_headers():
    """``uvicorn.run`` / ``uvicorn.Config`` in ``src/maddening`` -- the cloud
    entrypoint and the shipped example servers, including the scripts they
    embed as text -- each pass ``proxy_headers``: ``False``, or (the cloud
    entrypoint) the operator's opt-in through ``FORWARDED_ALLOW_IPS``."""
    src = Path(maddening.__file__).resolve().parent
    found = []
    for path in sorted(src.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "uvicorn" not in text:
            continue
        trees = [ast.parse(text)]
        # Scripts an example writes out and runs (a string constant).
        for node in ast.walk(trees[0]):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and "uvicorn." in node.value and "import uvicorn" in node.value):
                try:
                    trees.append(ast.parse(node.value))
                except SyntaxError:     # a docstring that shows a usage, not a script
                    continue
        for tree in trees:
            for call in _uvicorn_launches(tree):
                names = {kw.arg: kw.value for kw in call.keywords}
                found.append((path.relative_to(src), call.lineno))
                assert "proxy_headers" in names, (path, call.lineno)
                value = names["proxy_headers"]
                if isinstance(value, ast.Constant):
                    assert value.value is False, (path, call.lineno)
                else:
                    assert "FORWARDED_ALLOW_IPS" in ast.unparse(value), (path, call.lineno)
    assert len(found) >= 10, found
