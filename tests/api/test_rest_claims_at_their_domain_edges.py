"""REST claims at the edges of domains their own tests do not reach.

Cells of ``docs/validation/rest_runpod_claims.yaml``'s domain matrix that
need a scenario of their own rather than a row's check run in another
domain (``tests/api/test_rest_claims_in_every_domain.py``):

* ``large_payload`` for the Origin and Host rules (REST-021, REST-025): a
  request the browser rules refuse, carrying a body past
  ``MAX_REQUEST_BODY_BYTES`` -- declared and streamed -- is refused for its
  Origin or Host (403), not for its size, and changes nothing;
* ``hostile_input`` for a checkpoint load (REST-093): a checkpoint of this
  very graph whose parameters carry a value ``PUT /graph/params`` refuses
  (negative where the bound is 0, non-finite);
* ``loopback_bind`` / ``no_token`` for REST-109: a configured token with
  whitespace around it is refused at construction on a loopback bind too;
* ``loopback_bind`` / ``non_loopback_bind`` for REST-098, the statement
  that there is no TLS: on either bind the server speaks plain HTTP on its
  socket, the bearer header travels as written, and a TLS ClientHello gets
  no TLS answer.

Nothing here can reach a cloud provider (no request goes to ``/cloud/*``;
:func:`tests.property.differential.no_cloud_launch`).
"""

from __future__ import annotations

import json
import os
import socket
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api import server as server_module
from tests.api import rest_claims_support as S
from tests.property.differential import no_cloud_launch

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning:websockets.*"),
              pytest.mark.filterwarnings("ignore::DeprecationWarning:uvicorn.*")]

_ANY = S.Check(fn=None, rows=(), bind="any", contexts=frozenset(), server_kw={}, patch={},
               xfail={})


@pytest.fixture(scope="module", autouse=True)
def _offline():
    with no_cloud_launch():
        yield


def _served(bind: str, root):
    server = server_module.SimulationServer(
        S.REGISTRY, graph_manager=S.build_graph(), bind_host=bind, api_token=S.TOKEN,
        checkpoint_root=str(root))
    return server, server.create_app()


def _oversized(chunks: bool):
    body = json.dumps({"params": {"stiffness": 41.0}, "pad": "x" * 4000}).encode()
    if not chunks:
        return body

    def stream():
        for i in range(0, len(body), 128):
            yield body[i:i + 128]
    return stream()


@pytest.mark.parametrize("chunks", [False, True], ids=["declared", "streamed"])
@pytest.mark.parametrize("bind", ["127.0.0.1", "0.0.0.0"])
def test_a_foreign_origin_with_an_oversized_body_is_refused_for_its_origin(
        bind, chunks, tmp_path, monkeypatch):
    """REST-021 at and past the body cap: the Origin rule answers first."""
    monkeypatch.setattr(server_module, "MAX_REQUEST_BODY_BYTES", 1000)
    server, app = _served(bind, tmp_path)
    headers = {"Origin": S.FOREIGN_ORIGIN, "Content-Type": "text/plain",
               **(S.BEARER if server.auth.enforced else {})}
    before = float(np.asarray(server.gm.params["nodes"]["spring"]["stiffness"]))
    resp = TestClient(app, raise_server_exceptions=False).put(
        "/graph/params/spring", content=_oversized(chunks), headers=headers)
    assert resp.status_code == 403, resp.text
    assert "origin" in resp.json()["detail"].lower()
    assert float(np.asarray(server.gm.params["nodes"]["spring"]["stiffness"])) == before


@pytest.mark.parametrize("chunks", [False, True], ids=["declared", "streamed"])
def test_a_foreign_host_with_an_oversized_body_is_refused_for_its_host(
        chunks, tmp_path, monkeypatch):
    """REST-025 at and past the body cap: the Host rule answers first."""
    monkeypatch.setattr(server_module, "MAX_REQUEST_BODY_BYTES", 1000)
    server, app = _served("127.0.0.1", tmp_path)
    before = float(np.asarray(server.gm.params["nodes"]["spring"]["stiffness"]))
    resp = TestClient(app, raise_server_exceptions=False, client=S.LOOPBACK_PEER).put(
        "/graph/params/spring", content=_oversized(chunks),
        headers={"Host": "attacker.example", "Content-Type": "application/json"})
    assert resp.status_code == 403, resp.text
    assert "attacker.example" in resp.json()["detail"]
    assert float(np.asarray(server.gm.params["nodes"]["spring"]["stiffness"])) == before


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")],
                         ids=["below-bound", "nan", "inf"])
def test_a_checkpoint_carrying_a_value_a_params_write_refuses_is_not_loaded(value, tmp_path):
    """A checkpoint of this very graph -- same nodes, same shapes -- whose
    spring damping (bounds ``(0, None)``) holds a value the params route
    refuses, written into ``gm.params`` the way Python code may write any
    value: the load is a 400 and nothing is loaded."""
    donor = S.build_graph()
    donor.params["nodes"]["spring"]["damping"] = jnp.float32(value)
    donor.save_state(str(tmp_path / "hostile.npz"))
    server, app = _served("127.0.0.1", tmp_path)
    before = float(np.asarray(server.gm.params["nodes"]["spring"]["damping"]))
    resp = TestClient(app, raise_server_exceptions=False).post(
        "/checkpoint/load", params={"path": "hostile.npz"})
    assert resp.status_code == 400, (resp.status_code, resp.text)
    assert float(np.asarray(server.gm.params["nodes"]["spring"]["damping"])) == before


#: The first bytes of a TLS 1.2/1.3 ClientHello record: handshake (0x16),
#: version 3.1, a length, ClientHello (0x01).
_CLIENT_HELLO = bytes.fromhex("160301002e0100002a0303") + bytes(32) + bytes.fromhex(
    "000002002f0100")


@pytest.mark.parametrize("bind", ["127.0.0.1", "0.0.0.0"])
def test_the_server_speaks_plain_http_and_the_token_crosses_in_cleartext(bind, tmp_path):
    """REST-098: "There is still no TLS.  The token and every state snapshot
    cross the network in cleartext."  Over a real socket (uvicorn on
    loopback, the app told *bind*): a request written as plain bytes, with
    the bearer header as it would cross the network, is answered in plain
    HTTP with the state in the clear; a TLS ClientHello is answered with no
    TLS record (an HTTP 400, or a closed connection).  If TLS is ever added,
    this fails and the sentence must change with it."""
    chk = S.Check(fn=None, rows=(), bind="public" if bind == "0.0.0.0" else "loopback",
                  contexts=frozenset(), server_kw={}, patch={}, xfail={})
    with S.loopback_server(chk, tmp_path) as (server, base):
        port = int(base.rsplit(":", 1)[1])
        assert server.auth.enforced == (bind == "0.0.0.0")
        request = (f"GET /graph/state/ball HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                   f"Authorization: Bearer {S.TOKEN}\r\nConnection: close\r\n\r\n").encode()
        assert S.TOKEN.encode() in request          # the credential as it travels
        with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
            sock.sendall(request)
            reply = b""
            while chunk := sock.recv(65536):
                reply += chunk
        assert reply.startswith(b"HTTP/1.1 200"), reply[:200]
        assert b'"position"' in reply                 # the state, readable on the wire
        with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
            sock.sendall(_CLIENT_HELLO)
            try:
                answer = sock.recv(16)
            except (ConnectionResetError, socket.timeout):
                answer = b""
        assert not answer.startswith(b"\x16"), answer      # no TLS handshake record
        assert answer == b"" or answer.startswith(b"HTTP/1.1 4"), answer


@pytest.mark.parametrize("token", [" t", "t ", "t\n", "\tt"], ids=["lead", "trail", "newline", "tab"])
def test_a_token_with_whitespace_around_it_is_refused_on_a_loopback_bind_too(token):
    """REST-109 on a loopback bind, where no token is demanded of a loopback
    peer but the backstop demands it of a routable one: a configured token
    with whitespace around it is refused at construction there too, from
    ``token=`` and from the environment, and an inner space still works."""
    from maddening.api.auth import APIAuth

    for kwargs in ({"token": token, "environ": {}}, {"environ": {"MADDENING_API_TOKEN": token}}):
        with pytest.raises(ValueError, match="whitespace before or after it"):
            APIAuth(bind_host="127.0.0.1", **kwargs)
    with pytest.raises(ValueError, match="whitespace before or after it"):
        server_module.SimulationServer(S.REGISTRY, bind_host="127.0.0.1", api_token=token)
    assert not APIAuth(bind_host="127.0.0.1", token="a b", environ={}).enforced
