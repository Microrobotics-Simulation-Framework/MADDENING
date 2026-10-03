"""The REST inventory's claims as checks that run in any server domain.

``docs/validation/rest_runpod_claims.yaml`` gives each REST row a
``domains`` matrix: the server domains its conditions cover and a test that
exercises the claim in each (``testing_standards.md``, "The domain
matrix").  The rows' own tests run the claim on a loopback bind, one
request at a time, on a quiescent graph.  This module states each claim
once more, as a check keyed by its row id, written so that it holds
wherever the claim does:

* it reads and compares only what its own requests could change (a
  parameter, the node list, a file), never the whole state, so a runner or
  a run stepping the graph beside it cannot break it and neither can
  several copies of it at once;
* every value it writes is the same in every copy, and it puts back what
  it changed;
* a refusal is checked for its status, its words and that nothing in the
  request was written.

:func:`served` builds a fresh server and graph per check and puts it in a
domain; ``test_rest_claims_in_every_domain.py`` runs every check in the
in-process domains and ``test_rest_claims_under_concurrent_requests.py``
runs them again, several copies at once, over a real uvicorn server on
loopback.  Nothing here can reach a cloud provider: no check sends a
request to ``/cloud/*``, and both modules run under
:func:`tests.property.differential.no_cloud_launch`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import signal
import threading
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api import server as server_module
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.core.simulation.hybrid_node import HybridNode
from maddening.nodes import BallNode, HeatNode, SpringDamperNode
from maddening.nodes.rigid_body import RigidBodyNode
from tests.api.test_params_endpoint_constructed_values import _Legacy, _SelfBound
from tests.core.test_mapping_point_reference_writes import _rods

#: The token every server here is given; not a credential anywhere.
TOKEN = "rest-domains-token-not-a-credential"
#: How a request presents it: ``Authorization: Bearer <token>``.
BEARER = {"Authorization": f"Bearer {TOKEN}"}
#: An Origin that is not the server's own.
FOREIGN_ORIGIN = "http://evil.example"
#: A peer that is not this machine (TEST-NET-3, RFC 5737).
REMOTE_PEER = ("203.0.113.5", 44321)
#: A loopback TCP peer: an IP address, so the Host rule is asked of it.
LOOPBACK_PEER = ("127.0.0.1", 50123)

REGISTRY = {"BallNode": BallNode, "SpringDamperNode": SpringDamperNode,
            "HeatNode": HeatNode}

#: The in-process domains, by the name the test functions use.
IN_PROCESS = ("loopback", "public", "runner", "sim_run", "restored", "wrapper", "shutdown")
#: The matrix domain each in-process context fills.
FILLS = {
    "loopback": ("loopback_bind", "no_token"),
    "public": ("non_loopback_bind", "token_enforced"),
    "runner": ("runner_active",),
    "sim_run": ("sim_run_active",),
    "restored": ("checkpoint_restore",),
    "wrapper": ("wrapper_nodes",),
    "shutdown": ("shutdown",),
    "concurrent": ("concurrent",),
}


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------

def _zero_correction(state, boundary_inputs, dt):
    return {}


def _sharded(node):
    from maddening.cloud.multigpu.device_mesh import create_device_mesh
    from maddening.cloud.multigpu.sharded_node import ShardedStencilNode

    return ShardedStencilNode(node, create_device_mesh(shape=(1,)), {"devices": 0})


def build_graph(*, wrapped: bool = False) -> GraphManager:
    """Every node a check writes to, compiled:

    * ``a`` and ``b``: two uniform rods, ``a`` mapped onto ``b`` by an RBF
      mapping built from references to each rod's ``grid_x`` (REST-079);
    * ``c``: a third rod nothing maps (the Fourier limit, a grid, a cell
      count);
    * ``ball`` and ``spring``: live leaves, the spring's with
      ``ParamSpec`` bounds;
    * ``legacy``: a node that copied ``baked`` at construction;
    * ``selfbound``: a node no faithful copy can be made of;
    * ``body``: a rigid body whose ``constraints`` the step traces.

    *wrapped*: the spring is a ``HybridNode`` and the rods ``a`` and ``c``
    are ``ShardedStencilNode`` on a one-device mesh; ``legacy``,
    ``selfbound`` and ``body`` are left out.
    """
    wrap = _sharded if wrapped else None
    gm = _rods(wrap=wrap)
    c = HeatNode("c", 0.01, n_cells=8, length=1.0, thermal_diffusivity=0.005,
                 initial_temperature=0.5)
    gm.add_node(_sharded(c) if wrapped else c)
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=100.0, gravity=-1.0))
    spring = SpringDamperNode("spring", 0.01, stiffness=40.0, damping=0.5, initial_position=0.5)
    gm.add_node(HybridNode(spring, _zero_correction) if wrapped else spring)
    if not wrapped:
        gm.add_node(_Legacy(name="legacy", timestep=0.01))
        gm.add_node(_SelfBound(name="selfbound", timestep=0.01))
        gm.add_node(RigidBodyNode("body", 0.01, initial_velocity=(0.1, 0.0, 0.2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


# ---------------------------------------------------------------------------
# A check, and the registry of them
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Check:
    """One claim's check: *fn(ctx)* for each of *rows*."""

    fn: Callable
    rows: tuple[str, ...]
    #: "any", or the one bind the claim is about ("loopback" / "public")
    bind: str
    #: the domains it runs in
    contexts: frozenset
    #: extra ``SimulationServer`` arguments
    server_kw: dict
    #: ``maddening.api.server`` constants patched for the check
    patch: dict
    #: (row, domain) -> reason: a cell that is a strict xfail
    xfail: dict


CHECKS: dict[str, Check] = {}

ALL = frozenset(IN_PROCESS + ("concurrent",))


def check(*rows: str, bind: str = "any", skip=(), only=None, server_kw=None, patch=None,
          xfail=None):
    """Register *fn* as the check of *rows*, in every domain but *skip* (or
    only those in *only*); a check about one bind leaves out the other."""
    def register(fn):
        contexts = set(only) if only is not None else set(ALL)
        contexts -= set(skip)
        if bind == "loopback":
            contexts.discard("public")
        if bind == "public":
            contexts.discard("loopback")
        chk = Check(fn, rows, bind, frozenset(contexts), dict(server_kw or {}),
                    dict(patch or {}), dict(xfail or {}))
        for row in rows:
            assert row not in CHECKS, f"{row} has two checks"
            CHECKS[row] = chk
        return fn
    return register


def rows_for(domain: str) -> list:
    """``pytest.param`` per row whose check runs in *domain*, a strict xfail
    where the check registered one for that cell."""
    out = []
    for row in sorted(CHECKS):
        chk = CHECKS[row]
        if domain not in chk.contexts:
            continue
        reason = chk.xfail.get((row, domain)) or chk.xfail.get((row, "*"))
        marks = [pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)] \
            if reason else []
        out.append(pytest.param(row, marks=marks, id=row))
    return out


# ---------------------------------------------------------------------------
# The served graph, in a domain
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Ctx:
    """What a check sees: the server, a client of it, the checkpoint root."""

    domain: str
    server: SimulationServer
    app: object
    client: object
    root: Path
    #: copies of this check running at once, and which one this is
    copies: int = 1
    index: int = 0
    #: make a client: (headers, peer) -> client
    make_client: Optional[Callable] = None
    #: what the domain's background shares with the check
    extra: dict = dataclasses.field(default_factory=dict)

    @property
    def gm(self) -> GraphManager:
        return self.server.gm

    @property
    def enforced(self) -> bool:
        return self.server.auth.enforced

    def anonymous(self):
        """A client that presents no token."""
        return self.make_client(headers={}, peer=None)

    def with_headers(self, headers: dict, peer=None):
        base = dict(BEARER) if self.enforced else {}
        base.update(headers)
        return self.make_client(headers=base, peer=peer)

    def routable_peer(self, *, token: bool):
        """A client whose peer is a routable address."""
        return self.make_client(headers=dict(BEARER) if token else {}, peer=REMOTE_PEER)

    def ip_peer(self, headers=None):
        """A client whose peer is a loopback IP address, so the Host rule is
        asked (TestClient's own peer, ``testclient``, is not an IP)."""
        base = dict(BEARER) if self.enforced else {}
        base.update(headers or {})
        return self.make_client(headers=base, peer=LOOPBACK_PEER)

    def live(self, node: str, key: str):
        """The live leaf ``gm.params`` holds, as a host number or list (the
        node's own params for a node off the params contract)."""
        nodes = self.gm.params["nodes"]
        if node in nodes and key in nodes[node]:
            return np.asarray(nodes[node][key]).tolist()
        return self.node_param(node, key)

    def node_param(self, node: str, key: str):
        value = self.gm.get_node(node).params.get(key)
        return np.asarray(value).tolist() if value is not None else None

    @property
    def steps_beside(self) -> bool:
        """Whether something else steps the graph during the check."""
        return self.domain in ("runner", "sim_run") or self.copies > 1


def _in_process_client(app):
    def make(*, headers, peer):
        kw = {}
        if peer is not None:
            kw["client"] = peer
            # TestClient's own Host, ``testserver``, is no name of this
            # machine: an IP peer that sent it would be refused for its Host.
            headers = {"Host": "localhost", **headers}
        return TestClient(app, raise_server_exceptions=False, headers=headers, **kw)
    return make


def wait_for(predicate, timeout: float = 20.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def stop_runner(server: SimulationServer) -> None:
    with server._runner_lock:
        server._stop_runner()


def make_server(domain: str, chk: Check, root: Path, gm: Optional[GraphManager] = None
                ) -> SimulationServer:
    if domain == "public" or (domain != "loopback" and chk.bind == "public"):
        bind = "0.0.0.0"
    else:
        bind = "127.0.0.1"
    gm = gm if gm is not None else build_graph(wrapped=(domain == "wrapper"))
    return SimulationServer(REGISTRY, graph_manager=gm, bind_host=bind, api_token=TOKEN,
                            checkpoint_root=str(root), **chk.server_kw)


@contextlib.contextmanager
def patched(chk: Check):
    with pytest.MonkeyPatch.context() as mp:
        for name, value in chk.patch.items():
            mp.setattr(server_module, name, value)
        yield


@contextlib.contextmanager
def served(domain: str, chk: Check, root: Path):
    """A fresh server for *chk* in an in-process *domain*, with the
    domain's background running for the whole ``with`` body."""
    root.mkdir(parents=True, exist_ok=True)
    with patched(chk):
        server = make_server(domain, chk, root)
        app = server.create_app()
        make = _in_process_client(app)
        client = make(headers=dict(BEARER) if server.auth.enforced else {}, peer=None)
        ctx = Ctx(domain, server, app, client, root, make_client=make)
        with background(ctx):
            yield ctx


@contextlib.contextmanager
def background(ctx: Ctx):
    """What runs beside the check in its domain."""
    server, client = ctx.server, ctx.client
    if ctx.domain == "runner":
        assert client.post("/sim/start").status_code == 200
        start = server.relay.step_count
        assert wait_for(lambda: server.relay.step_count > start + 2), "the runner never stepped"
        try:
            yield
            # The check ran while the runner ran: the domain was reached.
            assert server.runner is not None and server.runner.is_alive, \
                "the runner stopped during the check"
        finally:
            stop_runner(server)
    elif ctx.domain == "sim_run":
        gm = server.gm
        real_run = gm.run
        done = threading.Event()

        def slow(n, *args, **kwargs):     # one 30 ms step a slice until the check is done
            if not done.is_set():
                time.sleep(0.03)
            return real_run(n, *args, **kwargs)

        gm.run = slow
        reply: dict = {}
        run = threading.Thread(target=lambda: reply.setdefault(
            "run", ctx.make_client(headers=dict(BEARER) if ctx.enforced else {}, peer=None)
            .post("/sim/run", params={"n_steps": server_module.MAX_RUN_STEPS})))
        run.start()
        ctx.extra.update(run_thread=run, run_reply=reply)
        try:
            assert wait_for(lambda: server.relay.step_count >= 2), "the /sim/run never stepped"
            yield
            assert run.is_alive() or ctx.extra.get("run_may_end"), \
                "the /sim/run ended during the check"
        finally:
            done.set()
            server.request_shutdown()
            run.join(60)
            server._shutdown.clear()
            gm.run = real_run
        assert reply["run"].status_code in (200, 503), reply["run"].text
    elif ctx.domain == "restored":
        assert client.post("/sim/run", params={"n_steps": 5}).status_code == 200
        assert client.post("/checkpoint/save", params={"path": "restored.npz"}).status_code == 200
        saved = {f: np.asarray(v) for f, v in server.gm.get_node_state("ball").items()}
        assert client.post("/sim/run", params={"n_steps": 3}).status_code == 200
        resp = client.post("/checkpoint/load", params={"path": "restored.npz"})
        assert resp.status_code == 200, resp.text
        for f, v in saved.items():
            assert np.array_equal(np.asarray(server.gm.get_node_state("ball")[f]), v)
        yield
    else:
        yield


def run_check(row: str, domain: str, root: Path) -> None:
    """Run *row*'s check in an in-process *domain*."""
    chk = CHECKS[row]
    assert domain in chk.contexts, (row, domain)
    with served(domain, chk, root) as ctx:
        if domain == "shutdown":
            mid_request_sigterm(chk, ctx)
        else:
            chk.fn(ctx)


def mid_request_sigterm(chk: Check, ctx: Ctx) -> None:
    """Run the check while SIGTERM arrives: the first time a request of the
    check takes the graph lock it waits there until the main thread has
    raised SIGTERM, which reaches the handler the server chains ahead of
    the one installed (here a recorder, as uvicorn's would be).  The check
    runs on a worker thread so the main thread is free to take the signal."""
    server = ctx.server
    lock = server._graph_lock
    real_acquire = lock.acquire
    entered, fired = threading.Event(), threading.Event()
    main = threading.main_thread()

    def acquire(*args, **kwargs):
        got = real_acquire(*args, **kwargs)
        if got and not fired.is_set() and threading.current_thread() is not main:
            entered.set()
            fired.wait(10)
        return got

    seen: list = []
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: seen.append(signum))
    restore = server._chain_shutdown_signals()
    lock.acquire = acquire
    error: list = []

    def body():
        try:
            chk.fn(ctx)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            error.append(exc)

    worker = threading.Thread(target=body)
    try:
        worker.start()
        while worker.is_alive() and not entered.wait(0.01):
            pass
        reached = entered.is_set()
        if reached:
            signal.raise_signal(signal.SIGTERM)
            shut = server._shutdown.is_set()
        fired.set()
        worker.join(120)
    finally:
        fired.set()
        del lock.acquire
        restore()
        signal.signal(signal.SIGTERM, previous)
        server._shutdown.clear()
    if error:
        raise error[0]
    assert reached, "no request of the check took the graph: SIGTERM never met one"
    assert seen == [signal.SIGTERM] and shut, "the signal did not reach the server's handler"


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------

STATE_CHANGING = [
    ("PUT", "/graph/params/spring", {"json": {"params": {"stiffness": 41.0}}}),
    ("PUT", "/graph/state/ball", {"json": {"state": {"position": 5.0, "velocity": 0.0}}}),
    ("POST", "/sim/step", {}),
    ("POST", "/graph/nodes", {"json": {"type": "BallNode", "name": "intruder",
                                       "timestep": 0.01, "params": {}}}),
    ("DELETE", "/graph/nodes/ball", {}),
    ("POST", "/checkpoint/save", {"params": {"path": "intruder.npz"}}),
    ("POST", "/sim/reset", {}),
]


def structure(ctx: Ctx) -> tuple:
    return (list(ctx.gm._nodes), [str(e) for e in ctx.gm._edges])


def files(root: Path) -> list:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def refused(resp, status: int, *words: str) -> None:
    assert resp.status_code == status, (resp.status_code, resp.text)
    if words:
        detail = resp.json()["detail"]
        text = detail if isinstance(detail, str) else json.dumps(detail)
        for w in words:
            assert w in text, (w, text)


def unchanged_param(ctx: Ctx, node: str, key: str, before) -> None:
    assert ctx.live(node, key) == before, (node, key, ctx.live(node, key), before)


# ===========================================================================
# Authentication (REST-001 to REST-020)
# ===========================================================================

@check("REST-001", bind="loopback", skip=("restored", "wrapper", "shutdown"))
def _loopback_serves_without_a_token(ctx):
    """A loopback bind demands nothing of a loopback or non-IP peer: no
    header, a header with any token, and a malformed one are all served."""
    assert not ctx.enforced
    for headers in ({}, {"Authorization": "Bearer anything"},
                    {"Authorization": "garbage \x7f"}, {"Authorization": "Basic Zm9vOmJhcg=="}):
        c = ctx.make_client(headers=headers, peer=None)
        assert c.get("/graph").status_code == 200
        assert c.get("/graph/state/ball").status_code == 200
        resp = c.put("/graph/params/spring", json={"params": {"stiffness": 40.0}})
        assert resp.status_code == 200, resp.text
        assert c.get("/viz/app").status_code == 200
    assert ctx.client.get("/openapi.json").status_code == 200


def _anonymous_is_refused_everywhere(ctx, *, peer=None):
    anon = ctx.make_client(headers={}, peer=peer)
    before = (ctx.live("spring", "stiffness"), structure(ctx))
    for method, url, kwargs in [("GET", "/graph", {}), ("GET", "/graph/state", {}),
                                ("GET", "/graph/params/spring", {})] + STATE_CHANGING:
        resp = anon.request(method, url, **kwargs)
        assert resp.status_code == 401, (method, url, resp.status_code, resp.text)
        assert resp.headers.get("www-authenticate", "").lower().startswith("bearer")
    assert (ctx.live("spring", "stiffness"), structure(ctx)) == before
    assert not (ctx.root / "intruder.npz").exists()
    return anon


@check("REST-002", bind="public", skip=("restored", "wrapper", "shutdown"))
def _a_non_loopback_bind_demands_the_token(ctx):
    """Every route but the exempt ones refuses an anonymous caller (401,
    ``WWW-Authenticate: Bearer``) and changes nothing; the token unlocks
    them."""
    assert ctx.enforced
    _anonymous_is_refused_everywhere(ctx)
    assert ctx.client.get("/graph").status_code == 200
    assert ctx.client.get("/graph/params/spring").status_code == 200


@check("REST-004", bind="public", skip=("restored", "wrapper", "shutdown"))
def _only_the_exact_exempt_paths_are_served_anonymously(ctx):
    anon = ctx.anonymous()
    for path in ("/healthz", "/viz/app", "/viz/graph", "/viz/render", "/viz/auth.js"):
        assert anon.get(path).status_code == 200, path
    for method, path in (("GET", "/healthz/"), ("GET", "/viz/app/"), ("HEAD", "/graph"),
                         ("OPTIONS", "/graph"), ("GET", "/graph/")):
        assert anon.request(method, path).status_code == 401, (method, path)


@check("REST-005", bind="loopback", skip=("restored", "wrapper", "shutdown"))
def _a_routable_peer_is_challenged_on_a_loopback_bind(ctx):
    assert not ctx.enforced
    _anonymous_is_refused_everywhere(ctx, peer=REMOTE_PEER)
    assert ctx.routable_peer(token=True).get("/graph").status_code == 200
    assert ctx.make_client(headers={}, peer=LOOPBACK_PEER).get("/graph").status_code == 200


@check("REST-006", bind="public", skip=("restored", "wrapper", "shutdown"))
def _the_401_says_why_and_where_the_token_is(ctx):
    for headers in ({}, {"Authorization": "Bearer"}, {"Authorization": "Token x"}):
        resp = ctx.make_client(headers=headers, peer=None).get("/graph")
        refused(resp, 401, "token")
    # The backstop names the misconfiguration (a routable peer of a
    # server bound, as far as it was told, to loopback) -- asked of a second
    # server built the same way but told it is bound to 127.0.0.1.
    loop = SimulationServer(REGISTRY, graph_manager=build_graph(), bind_host="127.0.0.1",
                            checkpoint_root=str(ctx.root))
    resp = TestClient(loop.create_app(), client=REMOTE_PEER,
                      raise_server_exceptions=False).get("/graph")
    refused(resp, 401)
    assert "bind_host" in resp.json()["detail"] or "MADDENING_HOST" in resp.json()["detail"]


@check("REST-011", bind="public", skip=("restored", "wrapper", "shutdown"))
def _only_the_bearer_header_authenticates(ctx):
    before = ctx.live("spring", "stiffness")
    for headers, params in (({}, {"token": TOKEN}),
                            ({"Authorization": "Bearer wrong-token"}, {}),
                            ({"Authorization": f"Basic {TOKEN}"}, {}),
                            ({"Authorization": f"Bearer{TOKEN}"}, {}),
                            ({"Authorization": f"{TOKEN}"}, {})):
        c = ctx.make_client(headers=headers, peer=None)
        assert c.get("/graph", params=params).status_code == 401, headers
        resp = c.put("/graph/params/spring", params=params,
                     json={"params": {"stiffness": 41.0}})
        assert resp.status_code == 401, headers
    unchanged_param(ctx, "spring", "stiffness", before)
    spaced = ctx.make_client(headers={"Authorization": f"Bearer   {TOKEN}"}, peer=None)
    assert spaced.get("/graph").status_code == 200


def _ws_refused(client, url="/ws/state", **kwargs) -> bool:
    from starlette.websockets import WebSocketDisconnect

    try:
        with client.websocket_connect(url, **kwargs) as ws:
            ws.receive_text()
    except WebSocketDisconnect as exc:
        return exc.code == 1008
    return False


@check("REST-003", "REST-012", bind="public",
       only=("public", "runner", "sim_run"))
def _a_websocket_handshake_needs_the_credential(ctx):
    from maddening.api.auth import websocket_credentials

    anon = ctx.anonymous()
    for url in ("/ws/state", "/ws/state/binary"):
        assert _ws_refused(anon, url), url
    assert _ws_refused(anon, subprotocols=["maddening.bearer.bm90LWJhc2U2NA!!", "maddening.v1"])
    assert _ws_refused(anon, subprotocols=websocket_credentials("wrong-token"))
    with anon.websocket_connect("/ws/state", subprotocols=websocket_credentials(TOKEN)) as ws:
        assert ws.accepted_subprotocol == "maddening.v1"
    with ctx.client.websocket_connect("/ws/state") as ws:
        assert ws.accepted_subprotocol is None


@check("REST-013", bind="public", skip=("restored", "wrapper", "shutdown", "sim_run"))
def _the_interactive_docs_are_not_served_where_the_token_is_enforced(ctx):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert ctx.client.get(path).status_code == 404, path


@check("REST-015", skip=("restored", "wrapper", "shutdown"))
def _healthz_is_never_authenticated_and_says_nothing(ctx):
    resp = ctx.anonymous().get("/healthz")
    assert resp.status_code == 200
    assert set(resp.json()) == {"status", "version"}
    for page in ("/viz/app", "/viz/graph", "/viz/render", "/viz/auth.js"):
        assert TOKEN not in ctx.anonymous().get(page).text, page


@check("REST-018", skip=("restored", "wrapper", "shutdown"))
def _healthz_answers_while_another_request_holds_the_graph(ctx):
    lock = ctx.server._graph_lock
    assert lock.acquire(timeout=10)
    try:
        t0 = time.monotonic()
        assert ctx.anonymous().get("/healthz").status_code == 200
        assert time.monotonic() - t0 < 2.0
    finally:
        lock.release()


@check("REST-020", bind="public", skip=("restored", "wrapper", "shutdown"),
       patch={"MAX_REQUEST_BODY_BYTES": 1000})
def _an_anonymous_oversized_body_is_refused_for_its_credential_first(ctx):
    before = ctx.live("spring", "stiffness")
    body = json.dumps({"params": {"stiffness": 41.0, "pad": "x" * 2000}})
    headers = {"Content-Type": "application/json"}
    resp = ctx.anonymous().put("/graph/params/spring", content=body, headers=headers)
    assert resp.status_code == 401, resp.text
    resp = ctx.client.put("/graph/params/spring", content=body, headers=headers)
    assert resp.status_code == 413, resp.text
    unchanged_param(ctx, "spring", "stiffness", before)


# ===========================================================================
# The Origin and Host rules (REST-021 to REST-028)
# ===========================================================================

@check("REST-021", skip=("restored", "wrapper", "shutdown"))
def _a_foreign_origin_cannot_change_state(ctx):
    """Every state-changing request from a foreign Origin, sent the way a
    browser sends one with no preflight (``text/plain``), is a 403 and
    changes nothing -- also where another rule (the 409 beside a stepper)
    would have refused it."""
    hostile = ctx.with_headers({"Origin": FOREIGN_ORIGIN, "Content-Type": "text/plain"})
    before = (ctx.live("spring", "stiffness"), structure(ctx), files(ctx.root))
    for method, url, kwargs in STATE_CHANGING:
        if "json" in kwargs:
            kwargs = {**{k: v for k, v in kwargs.items() if k != "json"},
                      "content": json.dumps(kwargs["json"])}
        resp = hostile.request(method, url, **kwargs)
        refused(resp, 403)
    assert (ctx.live("spring", "stiffness"), structure(ctx), files(ctx.root)) == before


@check("REST-022", skip=("restored", "wrapper", "shutdown"))
def _no_origin_and_a_cross_origin_read_are_served(ctx):
    resp = ctx.client.put("/graph/params/spring", json={"params": {"stiffness": 42.0}})
    assert resp.status_code == 200, resp.text
    assert ctx.with_headers({"Origin": FOREIGN_ORIGIN}).get("/graph/state").status_code == 200
    assert ctx.with_headers({"Origin": FOREIGN_ORIGIN}).get("/graph/params/spring").json()[
        "stiffness"] == pytest.approx(42.0)


@check("REST-023", skip=("restored", "wrapper", "shutdown"),
       server_kw={"allowed_origins": ["https://UI.example:8443/"]})
def _an_allowed_origin_is_matched_literally_and_the_rest_fail_closed(ctx):
    before = ctx.live("spring", "stiffness")
    for origin in ("https://ui.example:8443", "HTTPS://UI.EXAMPLE:8443/"):
        resp = ctx.with_headers({"Origin": origin}).put(
            "/graph/params/spring", json={"params": {"stiffness": before}})
        assert resp.status_code == 200, (origin, resp.text)
    for origin in ("null", "https://user@ui.example:8443", "https://ui.example:8444",
                   "http://ui.example:8443", "https://ui.example.evil:8443"):
        resp = ctx.with_headers({"Origin": origin}).put(
            "/graph/params/spring", json={"params": {"stiffness": 43.0}})
        refused(resp, 403)
    unchanged_param(ctx, "spring", "stiffness", before)


@check("REST-024", only=("runner", "sim_run"))
def _a_websocket_from_a_foreign_origin_is_refused(ctx):
    hostile = ctx.with_headers({"Origin": FOREIGN_ORIGIN})
    assert _ws_refused(hostile)
    with ctx.client.websocket_connect("/ws/state") as ws:
        assert ws.accepted_subprotocol is None


@check("REST-025", bind="loopback", skip=("restored", "wrapper", "shutdown"))
def _a_loopback_bind_answers_only_to_the_names_this_machine_is_reached_by(ctx):
    before = (ctx.live("spring", "stiffness"), structure(ctx))
    rebound = ctx.ip_peer({"Host": "attacker.example"})
    for method, url, kwargs in [("GET", "/healthz", {}), ("GET", "/viz/app", {}),
                                ("GET", "/graph/state", {})] + STATE_CHANGING:
        refused(rebound.request(method, url, **kwargs), 403)
    assert (ctx.live("spring", "stiffness"), structure(ctx)) == before
    for host in ("localhost", "localhost:8000", "127.0.0.1:9", "[::1]:8000", "LOCALHOST.",
                 "127.3.4.5"):
        assert ctx.ip_peer({"Host": host}).get("/graph").status_code == 200, host


@check("REST-026", bind="loopback", skip=("restored", "wrapper", "shutdown"),
       server_kw={"allowed_hosts": ["sim.lab:8000"]})
def _an_allowed_host_is_matched_whatever_port_either_side_names(ctx):
    for host in ("sim.lab", "sim.lab:8000", "sim.lab:9999", "SIM.LAB:1"):
        assert ctx.ip_peer({"Host": host}).get("/graph").status_code == 200, host
    refused(ctx.ip_peer({"Host": "sim.lab.evil"}).get("/graph"), 403)


@check("REST-027", bind="public", skip=("restored", "wrapper", "shutdown"))
def _the_host_rule_is_asked_only_where_nothing_else_authenticates(ctx):
    rebound = {"Host": "attacker.example"}
    assert ctx.ip_peer(rebound).get("/graph").status_code == 200
    refused(ctx.make_client(headers=rebound, peer=LOOPBACK_PEER).get("/graph"), 401)


@check("REST-028", skip=("restored", "wrapper", "shutdown"))
def _a_malformed_host_is_a_refusal_not_a_500(ctx):
    want = 401 if ctx.enforced else 403
    for host in ("[::1]x", "[::1", "a b", "::1:80:80", "[zz]"):
        c = ctx.make_client(headers={"Host": host}, peer=LOOPBACK_PEER)
        assert c.get("/graph").status_code == want, host
        assert c.put("/graph/params/spring",
                     json={"params": {"stiffness": 44.0}}).status_code == want, host


# ===========================================================================
# Request bounds and budgets (REST-029 to REST-039)
# ===========================================================================

@check("REST-029", skip=("wrapper",), patch={"MAX_REQUEST_BODY_BYTES": 1000})
def _a_body_over_the_limit_is_a_413_before_it_is_parsed(ctx):
    before = ctx.live("spring", "stiffness")
    headers = {"Content-Type": "application/json"}

    def body(size: int) -> bytes:
        head = b'{"params": {"stiffness": 45.0}, "pad": "'
        return head + b"x" * (size - len(head) - 2) + b'"}'

    over = ctx.client.put("/graph/params/spring", content=body(1001), headers=headers)
    refused(over, 413)

    def stream():
        data = body(1500)
        for i in range(0, len(data), 100):
            yield data[i:i + 100]

    streamed = ctx.client.put("/graph/params/spring", content=stream(), headers=headers)
    refused(streamed, 413)
    unchanged_param(ctx, "spring", "stiffness", before)
    exactly = ctx.client.put("/graph/params/spring", content=body(1000), headers=headers)
    assert exactly.status_code != 413, exactly.text      # read, and judged by the route


@check("REST-031", skip=("wrapper", "shutdown"), patch={"MAX_RUN_STEPS": 12})
def _a_run_of_one_step_more_than_the_bound_is_a_422(ctx):
    count = ctx.server.relay.step_count
    for n in (13, -1, 10**9):
        refused(ctx.client.post("/sim/run", params={"n_steps": n}), 422, "n_steps")
    if not ctx.steps_beside:
        assert ctx.server.relay.step_count == count
        resp = ctx.client.post("/sim/run", params={"n_steps": 12})
        assert resp.status_code == 200, resp.text
        assert ctx.server.relay.step_count == count + 12
        assert ctx.client.post("/sim/run", params={"n_steps": 0}).status_code == 200
        assert ctx.server.relay.step_count == count + 12


@check("REST-032", skip=("sim_run", "shutdown"))
def _the_integer_and_value_bounds_are_422s_that_write_nothing(ctx):
    from maddening.api.server import MAX_NODE_PARAM_INT

    before = (ctx.node_param("c", "n_cells"), structure(ctx))
    for value in (MAX_NODE_PARAM_INT + 1, -(MAX_NODE_PARAM_INT + 1), [MAX_NODE_PARAM_INT + 1]):
        refused(ctx.client.put("/graph/params/c", json={"params": {"n_cells": value}}), 422)
    if not ctx.steps_beside:
        resp = ctx.client.post("/graph/nodes", json={
            "type": "HeatNode", "name": f"big{ctx.index}", "timestep": 0.01,
            "params": {"n_cells": MAX_NODE_PARAM_INT + 1}})
        refused(resp, 422)
    assert (ctx.node_param("c", "n_cells"), structure(ctx)) == before


@check("REST-034", skip=("wrapper",))
def _a_put_that_would_build_too_large_a_node_is_refused_before_it_is_built(ctx):
    built = []
    real_init = HeatNode.__init__

    def counting(self, *args, **kwargs):
        built.append(kwargs.get("n_cells"))
        real_init(self, *args, **kwargs)

    before = ctx.node_param("c", "n_cells")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(HeatNode, "__init__", counting)
        resp = ctx.client.put("/graph/params/c", json={"params": {"n_cells": 9_000_000}})
    refused(resp, 400)
    assert 9_000_000 not in built
    assert ctx.node_param("c", "n_cells") == before


@check("REST-035", skip=("runner", "sim_run", "wrapper", "concurrent"))
def _the_graph_holds_exactly_its_state_budget(ctx):
    """At a budget of what the graph holds plus four elements, two
    two-element springs fill it exactly and a third is refused, with
    nothing added.  (Several copies at once race for one budget: the
    concurrent module's own check.)"""
    held = sum(server_module._state_elements(fields)
               for name, fields in ctx.gm._state.items() if name != "_meta")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(server_module, "MAX_GRAPH_STATE_ELEMENTS", held + 4)
        for name in ("s1", "s2"):
            resp = ctx.client.post("/graph/nodes", json={
                "type": "SpringDamperNode", "name": name, "timestep": 0.01, "params": {}})
            assert resp.status_code == 201, resp.text
        before = structure(ctx)
        resp = ctx.client.post("/graph/nodes", json={
            "type": "SpringDamperNode", "name": "s3", "timestep": 0.01, "params": {}})
        refused(resp, 400, "whole graph")
        assert structure(ctx) == before


@check("REST-037", skip=("runner", "sim_run"))
def _a_state_field_with_far_more_values_is_refused_before_conversion(ctx):
    before = np.asarray(ctx.gm.get_node_state("c")["temperature"]).copy()
    calls = []
    real = np.asarray

    def counting(value, *args, **kwargs):
        if isinstance(value, list) and len(value) > 1000:
            calls.append(len(value))
        return real(value, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(server_module.np, "asarray", counting)
        resp = ctx.client.put("/graph/state/c", json={"state": {"temperature": [0.5] * 100_000}})
    refused(resp, 400, "temperature")
    assert calls == []
    assert np.array_equal(np.asarray(ctx.gm.get_node_state("c")["temperature"]), before)


@check("REST-038", "REST-099", skip=("restored", "wrapper", "shutdown"))
def _the_stride_is_bounded_and_a_value_left_out_keeps_its_own(ctx):
    from maddening.api.server import MAX_RELAY_STRIDE, MAX_STEPS_PER_FRAME

    assert ctx.client.put("/sim/stride", params={"steps_per_frame": 3,
                                                 "relay_stride": 2}).status_code == 200
    for params in ({"steps_per_frame": 0}, {"relay_stride": -1},
                   {"steps_per_frame": MAX_STEPS_PER_FRAME + 1},
                   {"relay_stride": MAX_RELAY_STRIDE + 1}):
        refused(ctx.client.put("/sim/stride", params=params), 422)
    resp = ctx.client.put("/sim/stride", params={"steps_per_frame": MAX_STEPS_PER_FRAME})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["steps_per_frame"] == MAX_STEPS_PER_FRAME and body["relay_stride"] == 2
    assert ctx.server.relay.stride == 2
    if ctx.server.runner is not None:
        assert ctx.server.runner.steps_per_frame == MAX_STEPS_PER_FRAME
    assert ctx.client.put("/sim/stride", params={"steps_per_frame": 1,
                                                 "relay_stride": 1}).status_code == 200


@check("REST-039", skip=("runner", "sim_run", "wrapper", "concurrent"))
def _the_profile_step_count_is_clamped_not_refused(ctx):
    for n in (0, -5):
        resp = ctx.client.post("/sim/profile", params={"n_steps": n})
        assert resp.status_code == 200, resp.text
        assert "traceEvents" in resp.json()


# ===========================================================================
# The graph lock (REST-040 to REST-049)
# ===========================================================================

def _threads(jobs) -> None:
    errors: list = []

    def run(job):
        try:
            job()
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(job,)) for job in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors, errors
    assert not any(t.is_alive() for t in threads), "a request never returned"


@check("REST-040", "REST-050", skip=("runner", "sim_run", "wrapper"))
def _concurrent_steps_are_all_taken(ctx):
    """Every POST /sim/step is one step: n of them, from several clients at
    once, take n steps, the relay counts n, and each reply is one step on."""
    start = ctx.server.relay.step_count
    per, clients = 5, 3
    codes: list = []

    def job():
        c = ctx.make_client(headers=dict(BEARER) if ctx.enforced else {}, peer=None)
        for _ in range(per):
            codes.append(c.post("/sim/step").status_code)

    _threads([job] * clients)
    assert codes == [200] * (per * clients)
    if ctx.copies == 1:
        assert ctx.server.relay.step_count == start + per * clients


@check("REST-041", "REST-096", "REST-063", only=("public", "restored", "shutdown"))
def _every_write_is_a_409_beside_a_stepper(ctx):
    from tests.api.test_graph_lock_refusals_cover_every_route import WRITES

    assert ctx.client.post("/checkpoint/save", params={"path": "c.npz"}).status_code == 200
    assert ctx.client.post("/sim/start").status_code == 200
    try:
        assert wait_for(lambda: ctx.server.relay.step_count > 2)
        before = (structure(ctx), ctx.live("spring", "stiffness"))
        for label, method, url, kwargs in WRITES:
            resp = ctx.client.request(method, url, **kwargs)
            refused(resp, 409, "runner")
        refused(ctx.client.post("/sim/start"), 409)
        assert (structure(ctx), ctx.live("spring", "stiffness")) == before
    finally:
        stop_runner(ctx.server)


@check("REST-042", skip=("sim_run",),
       xfail={("REST-042", "wrapper"): "REST-042: a PUT /graph/params to a HybridNode is "
              "answered 200 and lost (the hybrid copies its physics node's params); "
              "pending fix/p4-18-rest"})
def _a_params_write_reaches_a_running_graph(ctx):
    started = ctx.server.runner is None
    if started:
        assert ctx.client.post("/sim/start").status_code == 200
    try:
        assert wait_for(lambda: ctx.server.runner is not None and ctx.server.relay.step_count > 1)
        resp = ctx.client.put("/graph/params/spring", json={"params": {"stiffness": 46.0}})
        assert resp.status_code == 200, resp.text
        assert ctx.server.runner.is_alive
        assert ctx.live("spring", "stiffness") == pytest.approx(46.0)
        # ... and the running node computes with it.
        assert float(np.asarray(ctx.gm.get_node("spring").params["stiffness"])) == 46.0
    finally:
        if started:
            stop_runner(ctx.server)


@check("REST-043", skip=("concurrent",), patch={"_GRAPH_LOCK_TIMEOUT": 0.2},
       xfail={("REST-043", "sim_run"): (
           "REST-043: a POST /sim/run whose next slice cannot have the graph in time answers "
           "the 503 'Nothing was changed' after it has taken steps; pending fix/p4-18-rest")})
def _a_request_that_cannot_have_the_graph_is_a_503_that_writes_nothing(ctx):
    """Every request that cannot have the graph in time answers the 503
    'Nothing was changed', with ``Retry-After``, and nothing was changed --
    beside an in-flight ``/sim/run`` the run is such a request too, and its
    own 503 must be as true as the others'."""
    lock = ctx.server._graph_lock
    if ctx.domain == "sim_run":
        ctx.extra["run_may_end"] = True
        steps_before_hold = ctx.server.relay.step_count
    assert lock.acquire(timeout=20)
    try:
        before = (ctx.live("spring", "stiffness"), structure(ctx))
        for method, url, kwargs in [
                ("GET", "/graph/state", {}),
                ("PUT", "/graph/params/spring", {"json": {"params": {"stiffness": 47.0}}})]:
            t0 = time.monotonic()
            resp = ctx.client.request(method, url, **kwargs)
            refused(resp, 503, "Nothing was changed")
            assert resp.headers.get("retry-after") == "1"
            assert time.monotonic() - t0 < 5.0
        assert (ctx.live("spring", "stiffness"), structure(ctx)) == before
        if ctx.domain == "sim_run":
            ctx.extra["run_thread"].join(30)
    finally:
        lock.release()
    if ctx.domain == "sim_run":
        run = ctx.extra["run_reply"]["run"]
        if run.status_code == 503 and steps_before_hold > 0:
            assert "Nothing was changed" not in run.text, run.text


@check("REST-044", "REST-105", only=("public",), patch={"_GRAPH_LOCK_TIMEOUT": 0.25})
def _a_runner_route_answers_within_one_timeout_of_its_arrival(ctx):
    from tests.api.test_runner_routes_answer_within_a_lock_timeout import (
        test_each_request_behind_a_long_holder_answers_within_about_one_timeout as scenario)

    del scenario    # the same scenario, on this server's bind: below
    lock = ctx.server._graph_lock
    assert lock.acquire(timeout=20)
    timings: dict = {}
    try:
        def call(label, method, url, **kw):
            def job():
                t0 = time.monotonic()
                resp = ctx.make_client(headers=dict(BEARER) if ctx.enforced else {},
                                       peer=None).request(method, url, **kw)
                timings[label] = (time.monotonic() - t0, resp.status_code)
            return job

        jobs = [call("start", "POST", "/sim/start"), call("reset", "POST", "/sim/reset"),
                call("stride", "PUT", "/sim/stride", params={"steps_per_frame": 2}),
                call("stop", "POST", "/sim/stop")]
        threads = []
        for job in jobs:
            t = threading.Thread(target=job)
            t.start()
            threads.append(t)
            time.sleep(0.02)
        for t in threads:
            t.join(30)
    finally:
        lock.release()
    for label, (elapsed, status) in timings.items():
        assert elapsed < 1.6 * 0.25 + 1.0, (label, elapsed, status)
    assert timings["stride"][1] == 200


@check("REST-045", only=("public",))
def _stop_is_answered_while_another_request_holds_the_graph(ctx):
    assert ctx.client.post("/sim/start").status_code == 200
    assert wait_for(lambda: ctx.server.relay.step_count > 1)
    lock = ctx.server._graph_lock
    assert lock.acquire(timeout=20)
    try:
        t0 = time.monotonic()
        resp = ctx.client.post("/sim/stop")
        assert time.monotonic() - t0 < 5.0
    finally:
        lock.release()
    assert resp.status_code in (200, 503), resp.text
    stop_runner(ctx.server)


@check("REST-046", only=("public",), patch={"_GRAPH_LOCK_TIMEOUT": 0.25})
def _a_reset_that_stopped_the_runner_says_so(ctx):
    assert ctx.client.post("/sim/start").status_code == 200
    assert wait_for(lambda: ctx.server.relay.step_count > 1)
    lock = ctx.server._graph_lock
    assert lock.acquire(timeout=20)
    try:
        resp = ctx.client.post("/sim/reset")
    finally:
        lock.release()
    assert resp.status_code == 503, resp.text
    detail = resp.json()["detail"]
    text = detail if isinstance(detail, str) else json.dumps(detail)
    assert "Nothing was changed" not in text and "stopped" in text
    assert ctx.server.runner is None or not ctx.server.runner.is_alive


@check("REST-047", only=("public", "sim_run"))
def _a_read_waits_for_at_most_the_step_in_flight(ctx):
    started = ctx.server.runner is None and ctx.domain != "sim_run"
    if started:
        assert ctx.client.post("/sim/start").status_code == 200
        assert wait_for(lambda: ctx.server.relay.step_count > 1)
    try:
        for _ in range(5):
            t0 = time.monotonic()
            assert ctx.client.get("/graph/state/ball").status_code == 200
            assert time.monotonic() - t0 < 2.0
    finally:
        if started:
            stop_runner(ctx.server)


@check("REST-049", only=("public",))
def _while_the_runner_is_still_stopping_writes_and_start_are_503s(ctx):
    from tests.api.test_runner_stop_and_stride import (
        test_while_the_runner_is_still_stopping_state_writes_are_503s as scenario)

    del scenario
    gm = ctx.gm
    real = gm._compiled_step
    release = threading.Event()
    entered = threading.Event()

    def blocked(*args):
        entered.set()
        release.wait(30)
        return real(*args)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(server_module, "_RUNNER_STOP_TIMEOUT", 0.2)
        assert ctx.client.post("/sim/start").status_code == 200
        assert wait_for(lambda: ctx.server.relay.step_count > 0)
        gm._compiled_step = blocked
        assert entered.wait(20)
        try:
            refused(ctx.client.post("/sim/stop"), 503)
            refused(ctx.client.put("/graph/state/ball", json={
                "state": {"position": 1.0, "velocity": 0.0}}), 503)
            refused(ctx.client.post("/sim/start"), 503)
        finally:
            release.set()
            gm._compiled_step = real
    assert wait_for(lambda: ctx.server.runner is None or not ctx.server.runner.is_alive)
