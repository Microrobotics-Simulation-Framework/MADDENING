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
import socket
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

    @contextlib.contextmanager
    def under_test(self):
        """Mark the request(s) the claim is about: in the shutdown domain
        SIGTERM meets the first of them to take the graph, not a request
        that only sets the scene.  (A server that has been signalled takes
        no new request, so a check stops after it there: see
        :attr:`signalled`.)"""
        self.extra["auto_arm"] = False
        self.extra["armed"] = True
        try:
            yield
        finally:
            self.extra["armed"] = False

    @property
    def signalled(self) -> bool:
        """Whether SIGTERM has been raised: uvicorn takes no new request after
        it, so a check sends none either."""
        return bool(self.extra.get("signalled"))

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

    ctx.extra["auto_arm"] = True

    def acquire(*args, **kwargs):
        got = real_acquire(*args, **kwargs)
        armed = ctx.extra.get("armed") or ctx.extra.get("auto_arm")
        if got and armed and not fired.is_set() and threading.current_thread() is not main:
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
            ctx.extra["signalled"] = True
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
# A real server on loopback
# ---------------------------------------------------------------------------

def _uvicorn():
    """uvicorn, with its own choice of WebSocket implementation imported
    once, quietly: an older uvicorn picks the websockets library's legacy
    one, whose import warns (an error in this suite, on the server's
    thread otherwise)."""
    import uvicorn

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import uvicorn.protocols.websockets.auto  # noqa: F401
    return uvicorn


@contextlib.contextmanager
def loopback_server(chk: Check, root: Path, gm=None):
    """``(SimulationServer, base URL)`` of the check's server served by
    uvicorn on 127.0.0.1, in a thread; shut down as an embedded server
    must be: ``request_shutdown()`` first, then ``should_exit``."""
    root.mkdir(parents=True, exist_ok=True)
    with patched(chk):
        server = make_server("concurrent", chk, root, gm=gm)
        app = server.create_app()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        config = _uvicorn().Config(app, host="127.0.0.1", port=port, log_level="warning",
                                lifespan="on", proxy_headers=True, ws="auto",
                                forwarded_allow_ips="127.0.0.1", timeout_keep_alive=5)
        userver = _uvicorn().Server(config)
        thread = threading.Thread(target=userver.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            assert wait_for(lambda: userver.started, 20), "uvicorn never started"
            yield server, f"http://127.0.0.1:{port}"
        finally:
            server.request_shutdown()
            userver.should_exit = True
            thread.join(30)
            sock.close()
            assert not thread.is_alive(), "uvicorn did not shut down"


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


_SPY_LOCK = threading.Lock()
_SPIES: dict = {}


@contextlib.contextmanager
def spy(owner, name: str):
    """Record every call of ``owner.name`` while the block runs, shared by
    every copy of a check running at once (one patch, reference-counted,
    so copies that finish in any order cannot leave the class patched).
    Yields the list of ``(args, kwargs)`` recorded since the block began."""
    key = (id(owner), name)
    with _SPY_LOCK:
        if key not in _SPIES:
            real = getattr(owner, name)
            calls: list = []

            def recording(*args, **kwargs):
                calls.append((args, kwargs))
                return real(*args, **kwargs)

            setattr(owner, name, recording)
            _SPIES[key] = [real, calls, 0]
        entry = _SPIES[key]
        entry[2] += 1
        start = len(entry[1])
    try:
        yield _SpyView(entry[1], start)
    finally:
        with _SPY_LOCK:
            entry[2] -= 1
            if entry[2] == 0:
                setattr(owner, name, entry[0])
                del _SPIES[key]


class _SpyView:
    def __init__(self, calls: list, start: int):
        self._calls, self._start = calls, start

    def __iter__(self):
        return iter(self._calls[self._start:])


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
                    {"Authorization": "garbage !! not a scheme"}, {"Authorization": "Basic Zm9vOmJhcg=="}):
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


@check("REST-013", bind="public", skip=("restored", "wrapper", "shutdown"))
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


@check("REST-018", skip=("restored", "wrapper", "shutdown", "concurrent"))
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

    def body(size: int, key: str = "stiffness") -> bytes:
        head = b'{"params": {"' + key.encode() + b'": 45.0}, "pad": "'
        return head + b"x" * (size - len(head) - 2) + b'"}'

    over = ctx.client.put("/graph/params/spring", content=body(1001), headers=headers)
    refused(over, 413)

    def stream():
        data = body(1500)
        for i in range(0, len(data), 100):
            yield data[i:i + 100]

    streamed = ctx.client.put("/graph/params/spring", content=stream(), headers=headers)
    refused(streamed, 413)
    # Hostile content changes nothing: a malformed body past the limit is
    # refused for its size before anything tries to parse it.
    malformed = b"{\"params\": [[[" + b"\xff" * 1200
    refused(ctx.client.put("/graph/params/spring", content=malformed, headers=headers), 413)
    unchanged_param(ctx, "spring", "stiffness", before)
    exactly = ctx.client.put("/graph/params/spring", content=body(1000, key="no_such_key"),
                             headers=headers)
    refused(exactly, 400, "no_such_key")      # read, and judged by the route


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


@check("REST-032", skip=("shutdown",))
def _the_integer_and_value_bounds_are_422s_that_write_nothing(ctx):
    from maddening.api.server import MAX_NODE_PARAM_INT

    before = (ctx.node_param("c", "n_cells"), structure(ctx))
    for value in (MAX_NODE_PARAM_INT + 1, -(MAX_NODE_PARAM_INT + 1), [MAX_NODE_PARAM_INT + 1]):
        refused(ctx.client.put("/graph/params/c", json={"params": {"n_cells": value}}), 422)
    # A boolean is not counted as an integer (nor taken for one): wrong-typed,
    # it is refused, and nothing is written either way.
    resp = ctx.client.put("/graph/params/c", json={"params": {"n_cells": True}})
    assert resp.status_code in (400, 422), resp.text
    resp = ctx.client.post("/graph/nodes", json={
        "type": "HeatNode", "name": f"big{ctx.index}", "timestep": 0.01,
        "params": {"n_cells": MAX_NODE_PARAM_INT + 1}})
    refused(resp, 422)       # the request model's, before any 409 a stepper would cause
    assert (ctx.node_param("c", "n_cells"), structure(ctx)) == before


@check("REST-034", skip=("wrapper",))
def _a_put_that_would_build_too_large_a_node_is_refused_before_it_is_built(ctx):
    before = ctx.node_param("c", "n_cells")
    with spy(HeatNode, "__init__") as calls:
        resp = ctx.client.put("/graph/params/c", json={"params": {"n_cells": 9_000_000}})
        built = [kw.get("n_cells") for _, kw in calls]
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
    with spy(server_module.np, "asarray") as calls:
        resp = ctx.client.put("/graph/state/c", json={"state": {"temperature": [0.5] * 100_000}})
        long = [len(a[0]) for a, _ in calls if a and isinstance(a[0], list) and len(a[0]) > 1000]
    refused(resp, 400, "temperature")
    assert long == []
    assert np.array_equal(np.asarray(ctx.gm.get_node_state("c")["temperature"]), before)


@check("REST-038", "REST-099", skip=("restored", "wrapper", "shutdown"))
def _the_stride_is_bounded_and_a_value_left_out_keeps_its_own(ctx):
    """Zero, a negative stride and one past each bound are 422s that keep
    the values in force; the largest is taken and echoed as applied; a value
    left out of the query keeps its own."""
    from maddening.api.server import MAX_RELAY_STRIDE, MAX_STEPS_PER_FRAME

    assert ctx.client.put("/sim/stride", params={"steps_per_frame": 3,
                                                 "relay_stride": 2}).status_code == 200
    if ctx.copies > 1:      # every copy writes the same values: a fixed point
        assert ctx.client.put("/sim/stride", params={"steps_per_frame": MAX_STEPS_PER_FRAME,
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
    if ctx.copies == 1:
        assert ctx.client.put("/sim/stride", params={"steps_per_frame": 1,
                                                     "relay_stride": 1}).status_code == 200


@check("REST-039", skip=("runner", "sim_run", "wrapper"))
def _the_profile_step_count_is_clamped_not_refused(ctx):
    """A zero or negative step count is clamped to one, not refused."""
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


@check("REST-040", "REST-050", skip=("runner", "sim_run", "wrapper", "concurrent"))
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


@check("REST-041", "REST-096", only=("public", "restored", "shutdown"))
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


@check("REST-042", skip=("sim_run", "concurrent"),
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


def _timed(ctx, method, url, **kw):
    """``(seconds, status)`` of one request from a client of its own."""
    t0 = time.monotonic()
    resp = ctx.make_client(headers=dict(BEARER) if ctx.enforced else {},
                           peer=None).request(method, url, **kw)
    return time.monotonic() - t0, resp.status_code


def _behind_a_long_holder(ctx, calls) -> dict:
    """Hold the graph and send each ``(label, method, url, kwargs)`` of
    *calls* 20 ms apart, each from a thread of its own; their timings."""
    lock = ctx.server._graph_lock
    assert lock.acquire(timeout=20)
    timings: dict = {}
    try:
        threads = []
        for label, method, url, kw in calls:
            t = threading.Thread(target=lambda label=label, method=method, url=url, kw=kw:
                                 timings.__setitem__(label, _timed(ctx, method, url, **kw)))
            t.start()
            threads.append(t)
            time.sleep(0.02)
        for t in threads:
            t.join(30)
    finally:
        lock.release()
    return timings


@check("REST-044", only=("public", "sim_run", "shutdown"), patch={"_GRAPH_LOCK_TIMEOUT": 0.25})
def _a_runner_route_answers_within_one_timeout_of_its_arrival(ctx):
    """Behind a long holder of the graph, a start, a reset and a stop each
    answer within about one (patched) lock timeout of their arrival.  Beside
    an in-flight /sim/run each is refused at once (409: a run is in
    progress, or no runner to stop).  When SIGTERM arrives while a start
    holds the graph, the start still answers within the timeout."""
    if ctx.domain == "sim_run":
        for method, url, want in (("POST", "/sim/start", 409), ("POST", "/sim/reset", 409),
                                  ("POST", "/sim/stop", 409)):
            elapsed, status = _timed(ctx, method, url)
            assert status == want and elapsed < 1.0, (url, status, elapsed)
        return
    if ctx.domain == "shutdown":
        try:
            t0 = time.monotonic()
            with ctx.under_test():
                resp = ctx.client.post("/sim/start")
            assert resp.status_code == 200 and time.monotonic() - t0 < 0.25 + 1.0, resp.text
        finally:
            stop_runner(ctx.server)
        return
    timings = _behind_a_long_holder(ctx, [("start", "POST", "/sim/start", {}),
                                          ("reset", "POST", "/sim/reset", {}),
                                          ("stop", "POST", "/sim/stop", {})])
    for label, (elapsed, status) in timings.items():
        assert elapsed < 1.6 * 0.25 + 1.0, (label, elapsed, status)
    stop_runner(ctx.server)


@check("REST-105", only=("public", "sim_run"), patch={"_GRAPH_LOCK_TIMEOUT": 0.25})
def _the_stride_is_answered_at_once_whatever_the_runner_routes_wait_for(ctx):
    """PUT /sim/stride answers at once and applies its value: behind a long
    holder of the graph with a start and a reset waiting there, and beside
    an in-flight /sim/run."""
    if ctx.domain == "sim_run":
        elapsed, status = _timed(ctx, "PUT", "/sim/stride", params={"steps_per_frame": 2})
    else:
        timings = _behind_a_long_holder(ctx, [
            ("start", "POST", "/sim/start", {}), ("reset", "POST", "/sim/reset", {}),
            ("stride", "PUT", "/sim/stride", {"params": {"steps_per_frame": 2}})])
        elapsed, status = timings["stride"]
        stop_runner(ctx.server)
    assert status == 200 and elapsed < 0.25 / 2 + 0.5, (status, elapsed)
    assert ctx.client.put("/sim/stride", params={}).json()["steps_per_frame"] == 2


@check("REST-045", only=("public", "sim_run"))
def _stop_is_answered_while_another_request_holds_the_graph(ctx):
    """With the runner running and the graph held by another request, POST
    /sim/stop is answered at once; beside an in-flight /sim/run (no runner)
    it is the 409 'not started', at once."""
    if ctx.domain == "sim_run":
        elapsed, status = _timed(ctx, "POST", "/sim/stop")
        assert status == 409 and elapsed < 1.0, (status, elapsed)
        return
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


@check("REST-046", only=("public", "shutdown"), patch={"_GRAPH_LOCK_TIMEOUT": 0.25})
def _a_reset_that_stopped_the_runner_says_so(ctx):
    """A reset stops the runner first and takes the graph after: when the
    graph cannot be had in time its 503 says the runner it stopped stays
    stopped; when SIGTERM arrives while the reset holds the graph, it
    resets, says the runner was running, and the runner is gone."""
    assert ctx.client.post("/sim/start").status_code == 200
    assert wait_for(lambda: ctx.server.relay.step_count > 1)
    if ctx.domain == "shutdown":
        with ctx.under_test():
            resp = ctx.client.post("/sim/reset")
        assert resp.status_code == 200 and resp.json()["was_running"] is True, resp.text
        assert ctx.server.runner is None or not ctx.server.runner.is_alive
        return
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


# ===========================================================================
# /sim/*: stepping, running, the runner (REST-050 to REST-064)
# ===========================================================================

def advance(ctx, n: int) -> None:
    """Set the scene with *n* single steps (a ``/sim/run`` is what a
    shutdown interrupts, REST-052)."""
    for _ in range(n):
        resp = ctx.client.post("/sim/step")
        assert resp.status_code == 200, resp.text


def _ball(ctx) -> dict:
    return {f: float(np.asarray(v)) for f, v in ctx.gm.get_node_state("ball").items()}


@check("REST-051", skip=("runner", "sim_run", "shutdown", "concurrent"))
def _a_run_returns_the_state_after_n_steps(ctx):
    """``POST /sim/run?n_steps=N`` takes N steps and returns the state after
    them: what N single steps of a twin graph give, bit for bit."""
    twin = build_graph(wrapped=(ctx.domain == "wrapper"))
    twin.set_node_state("ball", ctx.gm.get_node_state("ball"))
    start = ctx.server.relay.step_count
    resp = ctx.client.post("/sim/run", params={"n_steps": 7})
    assert resp.status_code == 200, resp.text
    for _ in range(7):
        twin.step()
    got = resp.json()["ball"]
    for field, value in twin.get_node_state("ball").items():
        assert float(np.asarray(got[field])) == float(np.asarray(value)), field
    if ctx.copies == 1:
        assert ctx.server.relay.step_count == start + 7


@check("REST-052", only=("public",))
def _a_run_stops_at_its_next_slice_when_the_server_shuts_down(ctx):
    gm = ctx.gm
    real = gm._compiled_step

    def slow(*args):
        time.sleep(0.001)
        return real(*args)

    gm._compiled_step = slow
    out: dict = {}
    t = threading.Thread(target=lambda: out.setdefault("r", ctx.client.post(
        "/sim/run", params={"n_steps": 100_000})))
    t.start()
    assert wait_for(lambda: ctx.server.relay.step_count > 20)
    ctx.server.request_shutdown()
    t.join(30)
    ctx.server._shutdown.clear()
    resp = out["r"]
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["status"] == "interrupted" and 0 < body["steps_run"] < 100_000
    assert ctx.server.relay.step_count == body["steps_run"]


@check("REST-053", only=("public",))
def _the_lifespan_shutdown_stops_the_runner_and_tells_a_run_to_stop(ctx):
    with TestClient(ctx.app, headers=dict(BEARER)) as client:
        assert not ctx.server._shutdown.is_set()
        assert client.post("/sim/start").status_code == 200
        assert wait_for(lambda: ctx.server.relay.step_count > 2)
    assert ctx.server._shutdown.is_set()
    assert ctx.server.runner is None
    ctx.server._shutdown.clear()


@check("REST-054", skip=("runner", "restored", "wrapper", "shutdown", "concurrent"))
def _a_second_start_is_a_409_and_a_dead_runner_is_replaced(ctx):
    if ctx.domain == "sim_run":
        refused(ctx.client.post("/sim/start"), 409, "/sim/run")
        return
    try:
        assert ctx.client.post("/sim/start").status_code == 200
        refused(ctx.client.post("/sim/start"), 409, "already started")
    finally:
        stop_runner(ctx.server)


@check("REST-055", skip=("runner", "restored", "wrapper", "shutdown", "concurrent"))
def _pause_and_resume_without_a_running_runner_are_409s(ctx):
    for route in ("/sim/pause", "/sim/resume"):
        refused(ctx.client.post(route), 409, "not")
    if ctx.domain == "sim_run":
        return
    try:
        assert ctx.client.post("/sim/start").status_code == 200
        assert ctx.client.post("/sim/pause").json()["status"] == "paused"
        assert ctx.client.post("/sim/resume").json()["status"] == "resumed"
    finally:
        stop_runner(ctx.server)


@check("REST-056", skip=("runner", "restored", "wrapper", "shutdown", "concurrent"))
def _stopped_means_no_runner_thread_steps_the_graph(ctx):
    if ctx.domain == "sim_run" or ctx.copies > 1:
        refused(ctx.client.post("/sim/stop"), 409, "not started")
        if ctx.domain == "sim_run":
            return
    if ctx.copies > 1:
        return
    assert ctx.client.post("/sim/start").status_code == 200
    assert wait_for(lambda: ctx.server.relay.step_count > 1)
    runner = ctx.server.runner
    resp = ctx.client.post("/sim/stop")
    assert resp.status_code == 200 and resp.json()["status"] == "stopped", resp.text
    assert not runner.is_alive
    refused(ctx.client.post("/sim/stop"), 409, "not started")


@check("REST-057", skip=("runner", "sim_run"))
def _a_reset_puts_every_node_back_and_restarts_the_clock(ctx):
    fresh = build_graph(wrapped=(ctx.domain == "wrapper"))
    initial = {f: float(np.asarray(v)) for f, v in fresh.get_node_state("ball").items()}
    if ctx.copies == 1:
        advance(ctx, 4)
    with ctx.under_test():
        resp = ctx.client.post("/sim/reset")
    assert resp.status_code == 200, resp.text
    assert resp.json()["was_running"] is False
    if ctx.copies == 1:
        assert {f: float(v) for f, v in resp.json()["state"]["ball"].items()
                if f in initial} == initial
        assert _ball(ctx) == initial
        assert ctx.server.relay.step_count == 0 and ctx.server.relay.elapsed == 0.0


@check("REST-058", skip=("runner", "sim_run", "concurrent"))
def _a_graph_that_cannot_step_is_a_400_naming_why_and_nothing_is_stepped(ctx):
    edge = {"source_node": "c", "target_node": "ball", "source_field": "temperature",
            "target_field": "table_position"}
    assert ctx.client.post("/graph/edges", json=edge).status_code == 201
    before, count = _ball(ctx), ctx.server.relay.step_count
    try:
        for method, url, kw in (("POST", "/sim/run", {"params": {"n_steps": 3}}),
                                ("POST", "/sim/step", {})):
            if ctx.signalled:
                break
            with ctx.under_test():
                resp = ctx.client.request(method, url, **kw)
            refused(resp, 400)
            assert _ball(ctx) == before and ctx.server.relay.step_count == count
    finally:
        assert ctx.client.request("DELETE", "/graph/edges", json=edge).status_code == 200


@check("REST-063", "REST-100", skip=("runner", "sim_run"))
def _a_profile_is_a_trace_and_leaves_the_live_simulation_where_it_was(ctx):
    if ctx.copies == 1:
        advance(ctx, 3)
    before, count = _ball(ctx), ctx.server.relay.step_count
    stiffness = ctx.live("spring", "stiffness")
    with ctx.under_test():
        resp = ctx.client.post("/sim/profile", params={"n_steps": 4})
    assert resp.status_code == 200, resp.text
    trace = resp.json()
    assert isinstance(trace["traceEvents"], list) and trace["traceEvents"]
    assert _ball(ctx) == before and ctx.server.relay.step_count == count
    assert ctx.live("spring", "stiffness") == stiffness


@check("REST-064", only=("public",))
def _one_jax_trace_at_a_time(ctx):
    resp = ctx.client.post("/sim/profile/jax/start")
    assert resp.status_code == 200, resp.text
    try:
        refused(ctx.client.post("/sim/profile/jax/start"), 409)
        assert ctx.client.post("/sim/step").status_code == 200
        assert ctx.client.get("/sim/profile/jax/status").json()["active"] is True
    finally:
        stop = ctx.client.post("/sim/profile/jax/stop")
    assert stop.status_code == 200, stop.text
    refused(ctx.client.post("/sim/profile/jax/stop"), 409)


# ===========================================================================
# Graph structure and state (REST-065 to REST-072)
# ===========================================================================

@check("REST-065", "REST-066", skip=("runner", "sim_run", "wrapper"))
def _an_add_node_that_is_refused_adds_nothing(ctx):
    before = structure(ctx)
    cases = [
        ({"type": "NoSuchNode", "name": "x", "timestep": 0.01, "params": {}}, 400),
        ({"type": "BallNode", "name": "ball", "timestep": 0.01, "params": {}}, 409),
        ({"type": "SpringDamperNode", "name": "bad", "timestep": 0.01,
          "params": {"stiffness": "stiff"}}, 400),
        ({"type": "BallNode", "name": "a/b", "timestep": 0.01, "params": {}}, 400),
        ({"type": "BallNode", "name": "", "timestep": 0.01, "params": {}}, 400),
        ({"type": "BallNode", "name": "NaN", "timestep": 0.01, "params": {}}, 400),
        ({"type": "HeatNode", "name": "fast", "timestep": 10.0,
          "params": {"n_cells": 8, "thermal_diffusivity": 1.0}}, 400),
    ]
    for body, status in cases:
        for _ in range(2):      # a repeat gets the same answer, never a 409
            refused(ctx.client.post("/graph/nodes", json=body), status)
    if ctx.copies == 1:
        assert structure(ctx) == before
    for body, _ in cases[2:]:
        assert body["name"] not in ctx.gm._nodes
    name = f"added{ctx.index}"
    resp = ctx.client.post("/graph/nodes", json={"type": "BallNode", "name": name,
                                                 "timestep": 0.01, "params": {}})
    assert resp.status_code == 201, resp.text
    assert name in [n["name"] for n in ctx.client.get("/graph").json()["nodes"]]


@check("REST-067", skip=("runner", "sim_run", "wrapper"))
def _removing_a_node_removes_every_edge_that_touches_it(ctx):
    name = f"hub{ctx.index}"
    assert ctx.client.post("/graph/nodes", json={"type": "BallNode", "name": name,
                                                 "timestep": 0.01, "params": {}}).status_code == 201
    for edge in ({"source_node": name, "target_node": "ball", "source_field": "position",
                  "target_field": "table_position"},
                 {"source_node": "ball", "target_node": name, "source_field": "position",
                  "target_field": "table_position"}):
        assert ctx.client.post("/graph/edges", json=edge).status_code == 201
    assert ctx.client.delete(f"/graph/nodes/{name}").status_code == 200
    assert all(name not in str(e) for e in ctx.gm._edges)
    refused(ctx.client.delete(f"/graph/nodes/{name}"), 404)
    assert ctx.client.post("/sim/step").status_code == 200


@check("REST-068", skip=("runner", "sim_run", "wrapper"))
def _an_edge_names_nodes_and_fields_that_exist(ctx):
    """An edge naming a missing node is a 404, a missing source field a 400,
    and a DELETE of an edge the graph does not have a 404; nothing changes."""
    before = structure(ctx)
    refused(ctx.client.post("/graph/edges", json={
        "source_node": "nope", "target_node": "ball", "source_field": "position",
        "target_field": "table_position"}), 404)
    refused(ctx.client.post("/graph/edges", json={
        "source_node": "ball", "target_node": "ball", "source_field": "no_field",
        "target_field": "table_position"}), 400)
    refused(ctx.client.request("DELETE", "/graph/edges", json={
        "source_node": "ball", "target_node": "spring", "source_field": "position",
        "target_field": "table_position"}), 404)
    assert structure(ctx) == before


@check("REST-069")
def _compile_returns_the_schedule_and_validate_the_issues(ctx):
    resp = ctx.client.post("/graph/compile")
    assert resp.status_code == 200, resp.text
    assert set(resp.json()["schedule"]) == set(ctx.gm._nodes)
    resp = ctx.client.post("/graph/validate")
    assert resp.status_code == 200, resp.text
    assert "issues" in resp.json()


@check("REST-070")
def _the_state_is_every_node_and_meta_and_an_unknown_node_is_a_404(ctx):
    resp = ctx.client.get("/graph/state")
    assert resp.status_code == 200, resp.text
    assert set(resp.json()) - {"_meta"} == set(ctx.gm._nodes)
    assert ("_meta" in resp.json()) == ("_meta" in ctx.gm._state)
    assert set(ctx.client.get("/graph/state/ball").json()) == {"position", "velocity"}
    refused(ctx.client.get("/graph/state/nope"), 404)


@check("REST-071", skip=("runner", "sim_run"))
def _a_state_write_the_field_cannot_hold_is_a_400_and_nothing_is_written(ctx):
    before = _ball(ctx)
    for state in ({"position": 1.0}, {"position": 1.0, "velocity": 0.0, "extra": 1.0},
                  {"position": [1.0, 2.0], "velocity": 0.0}, {"position": "NaN", "velocity": 0.0},
                  {"position": 1e39, "velocity": 0.0}, {"position": None, "velocity": 0.0}):
        resp = ctx.client.put("/graph/state/ball", json={"state": state})
        assert resp.status_code in (400, 422), (state, resp.status_code, resp.text)
        if ctx.copies == 1:
            assert _ball(ctx) == before, state
    big = float(np.finfo(np.float32).max)
    resp = ctx.client.put("/graph/state/ball", json={"state": {"position": big, "velocity": 0.0}})
    assert resp.status_code == 200, resp.text
    assert _ball(ctx)["position"] == big


@check("REST-072", skip=("wrapper",))
def _a_non_finite_reply_is_written_as_tokens(ctx):
    """A diverged state answers strict JSON, the non-finite values written as
    the quoted tokens."""
    big = float(np.finfo(np.float32).max)
    if ctx.domain in ("runner", "sim_run"):
        # The state cannot be written beside a stepper; a parameter an
        # in-process write left non-finite is, and every reply reading it --
        # and the state it then drives -- is tokenised.
        import jax.numpy as jnp

        ctx.gm.params["nodes"]["spring"]["damping"] = jnp.float32(np.nan)
        resp = ctx.client.get("/graph/params/spring")
        assert resp.status_code == 200, resp.text
        assert json.loads(resp.text, parse_constant=lambda c: pytest.fail(
            f"bare {c} in the reply"))["damping"] == "NaN"
        assert wait_for(lambda: '"NaN"' in ctx.client.get("/graph/state/spring").text)
        json.loads(ctx.client.get("/graph/state").text,
                   parse_constant=lambda c: pytest.fail(f"bare {c} in the state"))
        return
    resp = ctx.client.put("/graph/state/ball", json={"state": {"position": big,
                                                               "velocity": big}})
    assert resp.status_code == 200, resp.text
    assert ctx.client.put("/graph/params/ball", json={"params": {"gravity": -1e30}}).status_code \
        == 200
    for _ in range(4):
        resp = ctx.client.post("/sim/step")
        assert resp.status_code == 200, resp.text
    for url in ("/graph/state", "/graph/state/ball"):
        text = ctx.client.get(url).text
        json.loads(text, parse_constant=lambda c: pytest.fail(f"bare {c} in {url}"))
        assert '"Infinity"' in text or '"-Infinity"' in text or '"NaN"' in text, text


# ===========================================================================
# Parameters: PUT/GET /graph/params (REST-073 to REST-090, REST-106, REST-107)
# ===========================================================================

_HYBRID_LOST = ("a PUT /graph/params to a HybridNode is answered 200 and lost (the hybrid "
                "copies its physics node's params); pending fix/p4-18-rest")


def _put(ctx, node: str, params: dict):
    return ctx.client.put(f"/graph/params/{node}", json={"params": params})


@check("REST-073", xfail={("REST-073", "wrapper"): f"REST-073: {_HYBRID_LOST}"})
def _get_params_is_the_live_view(ctx):
    import jax.numpy as jnp

    if ctx.copies == 1:
        ctx.gm.params["nodes"]["spring"]["stiffness"] = jnp.asarray(48.0, jnp.float32)
        assert ctx.client.get("/graph/params/spring").json()["stiffness"] == pytest.approx(48.0)
    resp = _put(ctx, "spring", {"stiffness": 49.0})
    assert resp.status_code == 200, resp.text
    assert ctx.client.get("/graph/params/spring").json()["stiffness"] == pytest.approx(49.0)
    assert float(np.asarray(ctx.gm.get_node("spring").params["stiffness"])) == 49.0
    refused(ctx.client.get("/graph/params/nope"), 404)


@check("REST-074", xfail={("REST-074", "wrapper"): f"REST-074: {_HYBRID_LOST}"})
def _a_leaf_the_step_reads_takes_effect_on_the_next_step(ctx):
    """The spring's stiffness, written over REST, is what the next steps
    compute with, without a recompile: a twin graph built with the value
    steps the same, bit for bit, from the same state."""
    compiles = ctx.gm._compile_count if hasattr(ctx.gm, "_compile_count") else None
    resp = _put(ctx, "spring", {"stiffness": 80.0})
    assert resp.status_code == 200, resp.text
    assert not ctx.gm._dirty
    node = ctx.gm.get_node("spring")
    physics = getattr(node, "physics_node", node)
    assert float(np.asarray(physics.params["stiffness"])) == 80.0
    if not ctx.steps_beside and ctx.domain != "wrapper":
        twin = build_graph()
        twin.params["nodes"]["spring"]["stiffness"] = 80.0
        twin.set_node_state("spring", ctx.gm.get_node_state("spring"))
        assert ctx.client.post("/sim/step").status_code == 200
        twin.step()
        for f, v in twin.get_node_state("spring").items():
            assert float(np.asarray(ctx.gm.get_node_state("spring")[f])) == \
                float(np.asarray(v)), f
    if compiles is not None:
        assert ctx.gm._compile_count == compiles


@check("REST-075", only=("public", "restored"))
def _a_structural_value_recompiles_and_an_initial_condition_waits_for_the_reset(ctx):
    resp = _put(ctx, "legacy", {"k": 4.0})
    assert resp.status_code == 200, resp.text
    assert ctx.gm._dirty
    resp = _put(ctx, "spring", {"initial_position": 0.25})
    assert resp.status_code == 200, resp.text
    assert ctx.client.post("/sim/reset").status_code == 200
    assert float(np.asarray(ctx.gm.get_node_state("spring")["position"])) == 0.25


@check("REST-076", skip=("wrapper",))
def _a_value_the_running_node_cannot_use_is_a_400_and_nothing_is_written(ctx):
    before = (ctx.node_param("legacy", "baked"), ctx.node_param("legacy", "k"))
    resp = _put(ctx, "legacy", {"k": 4.0, "baked": 5.0})
    refused(resp, 400, "baked")
    assert (ctx.node_param("legacy", "baked"), ctx.node_param("legacy", "k")) == before


@check("REST-077")
def _a_value_the_constructor_refuses_with_the_live_values_is_a_400(ctx):
    before = ctx.live("c", "thermal_diffusivity")
    resp = _put(ctx, "c", {"thermal_diffusivity": 50.0})    # Fourier number 32: unstable
    refused(resp, 400, "thermal_diffusivity")
    assert "unstable" in resp.json()["detail"]
    unchanged_param(ctx, "c", "thermal_diffusivity", before)


@check("REST-078")
def _a_value_a_reload_would_build_differently_is_a_400(ctx):
    grid = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    before = ctx.node_param("c", "grid_points")
    resp = _put(ctx, "c", {"grid_points": grid})
    refused(resp, 400, "grid_points")
    assert ctx.node_param("c", "grid_points") == before
    assert ctx.client.get("/graph/params/c").json()["grid_points"] is None


@check("REST-079")
def _a_write_that_moves_a_mapping_s_points_is_a_400(ctx):
    before = ctx.live("a", "length")
    refused(_put(ctx, "a", {"length": 2.0}), 400, "length")
    unchanged_param(ctx, "a", "length", before)
    resp = _put(ctx, "a", {"thermal_diffusivity": 0.004})   # a calibration of the mapped rod
    assert resp.status_code == 200, resp.text


@check("REST-080")
def _a_non_finite_or_unrepresentable_value_is_a_400_before_anything_is_written(ctx):
    before = (ctx.live("spring", "stiffness"), ctx.live("spring", "damping"))
    for body in ('{"params": {"stiffness": NaN}}', '{"params": {"stiffness": Infinity}}',
                 '{"params": {"damping": -Infinity}}', '{"params": {"stiffness": 1e39}}',
                 '{"params": {"damping": 0.7, "stiffness": NaN}}',
                 '{"params": {"stiffness": 1' + "0" * 400 + '}}'):
        resp = ctx.client.put("/graph/params/spring", content=body,
                              headers={"Content-Type": "application/json"})
        assert resp.status_code in (400, 422), (body[:60], resp.status_code, resp.text)
    assert (ctx.live("spring", "stiffness"), ctx.live("spring", "damping")) == before


@check("REST-081")
def _the_whole_request_is_validated_before_anything_is_written(ctx):
    before = (ctx.live("spring", "stiffness"), ctx.live("spring", "damping"))
    resp = _put(ctx, "spring", {"stiffness": 51.0, "no_such_key": 1.0})
    refused(resp, 400, "no_such_key")
    assert "stiffness" in resp.json()["detail"]      # it lists the keys there are
    for bad in ("stiff", None, [1.0, 2.0], True, {"x": 1}):
        refused(_put(ctx, "spring", {"stiffness": 51.0, "damping": bad}), 400)
    assert (ctx.live("spring", "stiffness"), ctx.live("spring", "damping")) == before


@check("REST-082", only=("public", "restored"))
def _a_structural_value_is_stored_in_the_parameter_s_own_type(ctx):
    resp = _put(ctx, "c", {"n_cells": 8.0})
    assert resp.status_code == 200, resp.text
    assert type(ctx.gm.get_node("c").params["n_cells"]) is int
    refused(_put(ctx, "c", {"n_cells": 8.5}), 400)
    assert ctx.gm.get_node("c").params["n_cells"] == 8


@check("REST-083", skip=("wrapper",))
def _a_value_the_step_cannot_run_with_is_a_400_and_steps_go_on(ctx):
    for value in ({"w": 0.0}, {"y": {"finite": True}}, {"z": "high"}):
        resp = _put(ctx, "body", {"constraints": value})
        refused(resp, 400, "constraints")
        assert "step cannot run with it" in resp.json()["detail"]
    assert ctx.gm.get_node("body").params["constraints"] == {}
    if not ctx.steps_beside:
        assert ctx.client.post("/sim/step").status_code == 200


@check("REST-084")
def _a_value_that_changes_the_state_layout_is_refused_before_it_is_built(ctx):
    before = (ctx.node_param("c", "n_cells"), np.asarray(ctx.gm.get_node_state("c")[
        "temperature"]).shape)
    with spy(HeatNode, "__init__") as calls:
        resp = _put(ctx, "c", {"n_cells": 16})
        built = [kw.get("n_cells") for _, kw in calls]
    refused(resp, 400, "layout")
    assert 16 not in built
    assert (ctx.node_param("c", "n_cells"), np.asarray(ctx.gm.get_node_state("c")[
        "temperature"]).shape) == before


@check("REST-085", "REST-107")
def _a_value_outside_the_leaf_s_bounds_is_a_400_and_mutates_nothing(ctx):
    before = ctx.live("spring", "damping")
    for value in (-1.0, -1e-40, -0.0 - 1e-30):
        refused(_put(ctx, "spring", {"damping": value}), 400, "bound")
    unchanged_param(ctx, "spring", "damping", before)
    if ctx.domain != "wrapper" and ctx.copies == 1:
        resp = _put(ctx, "spring", {"damping": 1e-40})
        assert resp.status_code == 200, resp.text
        assert ctx.live("spring", "damping") == float(np.float32(1e-40))
        assert _put(ctx, "spring", {"damping": before}).status_code == 200


@check("REST-086", xfail={("REST-086", "wrapper"): f"REST-086: {_HYBRID_LOST}"})
def _the_reply_is_what_a_get_then_reads(ctx):
    resp = _put(ctx, "spring", {"stiffness": 52})
    assert resp.status_code == 200, resp.text
    got = ctx.client.get("/graph/params/spring").json()
    assert resp.json()["params"]["stiffness"] == got["stiffness"] == 52.0
    node = ctx.gm.get_node("spring")
    physics = getattr(node, "physics_node", node)
    assert float(np.asarray(physics.params["stiffness"])) == 52.0


@check("REST-088", skip=("wrapper",))
def _a_node_no_faithful_copy_can_be_made_of_is_refused(ctx):
    before = ctx.node_param("selfbound", "k")
    resp = _put(ctx, "selfbound", {"k": 4.0})
    refused(resp, 400, "no copy of _SelfBound")
    assert ctx.node_param("selfbound", "k") == before


@check("REST-106")
def _a_value_its_type_flushes_to_zero_is_refused(ctx):
    before = ctx.live("spring", "stiffness")
    for value in (1e-50, -1e-50, 1e-46):
        refused(_put(ctx, "spring", {"stiffness": value}), 400, "does not fit its type float32")
    unchanged_param(ctx, "spring", "stiffness", before)
    if not ctx.steps_beside:
        state = _ball(ctx)
        refused(ctx.client.put("/graph/state/ball", json={"state": {
            "position": 1e-50, "velocity": 0.0}}), 400, "does not fit its type float32")
        assert _ball(ctx) == state


# ===========================================================================
# Checkpoints (REST-091 to REST-095, REST-101, REST-102)
# ===========================================================================

def _manifest_ok(root: Path, name: str) -> dict:
    path = root / name
    manifest = json.loads((root / f"{name}.manifest.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest in json.dumps(manifest), (digest, manifest)
    return manifest


@check("REST-091")
def _a_save_writes_the_file_and_a_manifest_that_hashes_to_it(ctx):
    name = f"saved{ctx.index}.npz"
    resp = ctx.client.post("/checkpoint/save", params={"path": name})
    assert resp.status_code == 200, resp.text
    assert {"status", "path", "sim_time"} <= set(resp.json())
    _manifest_ok(ctx.root, name)


@check("REST-092")
def _a_checkpoint_path_is_a_file_inside_the_root(ctx):
    outside = ctx.root.parent / f"outside-{ctx.root.name}"
    outside.mkdir(exist_ok=True)
    before = files(outside)
    for path in ("", ".", "sub/..", "../x.npz", str(outside / "x.npz"),
                 f"../{outside.name}/x.npz", "a\x00b.npz"):
        for route in ("/checkpoint/save", "/checkpoint/load"):
            resp = ctx.client.post(route, params={"path": path})
            # Beside a stepper a load is refused before its path is read (REST-096).
            want = (400, 409) if route.endswith("load") and ctx.domain in (
                "runner", "sim_run") else (400,)
            assert resp.status_code in want, (route, path, resp.status_code, resp.text)
    assert files(outside) == before


@check("REST-093", skip=("runner", "sim_run"))
def _a_checkpoint_of_another_graph_is_a_400_and_nothing_is_loaded(ctx):
    other = GraphManager()
    other.add_node(BallNode("other", timestep=0.01))
    other.compile()
    other_name = f"other{ctx.index}.npz"
    other.save_state(str(ctx.root / other_name))
    before = (_ball(ctx), ctx.live("spring", "stiffness"))
    refused(ctx.client.post("/checkpoint/load", params={"path": other_name}), 400)
    refused(ctx.client.post("/checkpoint/load", params={"path": "missing.npz"}), 404)
    if ctx.copies == 1:
        assert (_ball(ctx), ctx.live("spring", "stiffness")) == before


@check("REST-094", skip=("runner", "sim_run"))
def _a_load_answers_the_checkpoint_s_clock(ctx):
    name = f"clock{ctx.index}.npz"
    if ctx.copies == 1:
        advance(ctx, 3)
    saved = ctx.client.post("/checkpoint/save", params={"path": name})
    assert saved.status_code == 200, saved.text
    sim_time = saved.json()["sim_time"]
    with ctx.under_test():
        resp = ctx.client.post("/checkpoint/load", params={"path": name})
    assert resp.status_code == 200, resp.text
    assert resp.json()["sim_time"] == sim_time and resp.json()["sim_time_from_checkpoint"]
    (ctx.root / f"{name}.manifest.json").unlink()
    resp = ctx.client.post("/checkpoint/load", params={"path": name})
    assert resp.status_code == 200, resp.text
    assert resp.json()["sim_time"] == 0.0 and not resp.json()["sim_time_from_checkpoint"]


@check("REST-095", skip=("runner", "sim_run", "wrapper"))
def _a_file_that_is_not_an_archive_is_a_400_naming_no_internals(ctx):
    for name, data in ((f"braces{ctx.index}.npz", b"{}"), (f"empty{ctx.index}.npz", b""),
                       (f"zip{ctx.index}.npz", b"PK\x03\x04garbage")):
        (ctx.root / name).write_bytes(data)
        resp = ctx.client.post("/checkpoint/load", params={"path": name})
        refused(resp, 400, f"could not load checkpoint '{name}'")
        detail = resp.json()["detail"]
        assert "pickle" not in detail and str(ctx.root) not in detail, detail


@check("REST-101", "REST-102")
def _a_save_refused_on_the_way_leaves_the_earlier_checkpoint(ctx):
    name = f"keep{ctx.index}.npz"
    assert ctx.client.post("/checkpoint/save", params={"path": name}).status_code == 200
    digest = hashlib.sha256((ctx.root / name).read_bytes()).hexdigest()
    manifest = ctx.root / f"{name}.manifest.json"
    manifest.unlink()
    manifest.mkdir()                     # the manifest's name taken by a directory
    try:
        resp = ctx.client.post("/checkpoint/save", params={"path": name})
        refused(resp, 400, "nothing was written")
        assert str(ctx.root) not in resp.json()["detail"]
        assert hashlib.sha256((ctx.root / name).read_bytes()).hexdigest() == digest
        assert not [p for p in ctx.root.rglob("*partial*")]
    finally:
        manifest.rmdir()


# ===========================================================================
# More rows: REST-033, REST-036, REST-087, REST-103, REST-104
# ===========================================================================

@check("REST-033", skip=("wrapper", "shutdown"),
       patch={"MAX_RUN_STEPS": 12, "MAX_NODE_STATE_ELEMENTS": 8})
def _each_bound_is_a_422_or_a_400_naming_the_size_at_its_edge(ctx):
    """The request-model bounds are 422s (``n_steps`` one past its bound, an
    integer one past its bound) whatever runs beside the request; a node
    whose state would pass the per-node cap (patched to eight elements) is a
    400 naming the size, and nothing is added."""
    from maddening.api.server import MAX_NODE_PARAM_INT

    refused(ctx.client.post("/sim/run", params={"n_steps": 13}), 422, "n_steps")
    refused(_put(ctx, "c", {"n_cells": MAX_NODE_PARAM_INT + 1}), 422)
    if ctx.domain in ("runner", "sim_run"):
        return          # a node cannot be added beside a stepper (REST-041's 409)
    before = structure(ctx)
    resp = ctx.client.post("/graph/nodes", json={
        "type": "HeatNode", "name": f"wide{ctx.index}", "timestep": 0.01,
        "params": {"n_cells": 9, "thermal_diffusivity": 0.001}})
    refused(resp, 400)
    assert "9" in resp.json()["detail"]
    if ctx.copies == 1:
        assert structure(ctx) == before
    assert f"wide{ctx.index}" not in ctx.gm._nodes


@check("REST-036", only=("loopback", "public", "restored", "shutdown"))
def _nodes_are_built_one_at_a_time(ctx):
    """Four POST /graph/nodes from four clients at once: every node is
    added, and no two constructors ever ran at the same time."""
    running, overlaps = [0], [0]
    guard = threading.Lock()
    real_init = SpringDamperNode.__init__

    def counting(self, *args, **kwargs):
        with guard:
            running[0] += 1
            overlaps[0] += running[0] > 1
        try:
            time.sleep(0.02)
            real_init(self, *args, **kwargs)
        finally:
            with guard:
                running[0] -= 1

    codes: list = []

    def add(i):
        def job():
            c = ctx.make_client(headers=dict(BEARER) if ctx.enforced else {}, peer=None)
            codes.append(c.post("/graph/nodes", json={
                "type": "SpringDamperNode", "name": f"n{i}", "timestep": 0.01,
                "params": {}}).status_code)
        return job

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SpringDamperNode, "__init__", counting)
        _threads([add(i) for i in range(4)])
    assert codes == [201] * 4
    assert overlaps[0] == 0, f"{overlaps[0]} constructors ran beside another"


@check("REST-087", bind="public", only=("wrapper",))
def _a_write_to_a_sharded_node_reaches_the_physics(ctx):
    """On a server told it is bound to 0.0.0.0 (every request with the
    token), a REST write to the rod a ``ShardedStencilNode`` wraps (``a``,
    whose temperatures vary along it) changes its next step exactly as the
    same write changes the unwrapped rod's, and unlike no write at all."""
    assert ctx.enforced
    start = ctx.gm.get_node_state("a")
    plain, unwritten = build_graph(), build_graph()
    for twin in (plain, unwritten):
        twin.set_node_state("a", start)
    resp = _put(ctx, "a", {"thermal_diffusivity": 0.02})
    assert resp.status_code == 200, resp.text
    plain.params["nodes"]["a"]["thermal_diffusivity"] = 0.02
    assert ctx.client.post("/sim/step").status_code == 200
    plain.step()
    unwritten.step()
    got = np.asarray(ctx.gm.get_node_state("a")["temperature"])
    np.testing.assert_allclose(got, np.asarray(plain.get_node_state("a")["temperature"]),
                               rtol=1e-6)
    assert not np.allclose(got, np.asarray(unwritten.get_node_state("a")["temperature"]),
                           rtol=1e-7, atol=0.0), "the write changed nothing the step reads"


def _stop_any_trace() -> None:
    from maddening.core.simulation import profiler

    if profiler.jax_trace_active():
        profiler.stop_jax_trace()


@check("REST-103", only=("public", "restored", "runner"), patch={"MAX_JAX_TRACE_STEPS": 4})
def _a_trace_stops_itself_at_its_step_budget(ctx):
    """A trace stops itself after its (patched) four steps: three leave it
    running, the fourth stops it, later ones are not counted -- the steps
    of POST /sim/step, or of the runner stepping the graph."""
    from maddening.core.simulation import profiler

    if ctx.domain == "runner":
        try:
            resp = ctx.client.post("/sim/profile/jax/start")
            assert resp.status_code == 200, resp.text
            assert wait_for(lambda: not profiler.jax_trace_active(), 20)
            status = ctx.client.get("/sim/profile/jax/status").json()
            assert status["active"] is False and status["steps"] == 4, status
            assert "step budget" in status["stopped_by"]
        finally:
            _stop_any_trace()
        return
    advance(ctx, 1)                       # compiled before tracing
    try:
        resp = ctx.client.post("/sim/profile/jax/start")
        assert resp.status_code == 200, resp.text
        advance(ctx, 3)
        assert ctx.client.get("/sim/profile/jax/status").json()["active"] is True
        advance(ctx, 3)
        status = ctx.client.get("/sim/profile/jax/status").json()
        assert status["active"] is False and status["steps"] == 4
        assert "step budget" in status["stopped_by"]
        refused(ctx.client.post("/sim/profile/jax/stop"), 409, "stopped itself after 4 steps")
    finally:
        _stop_any_trace()


@check("REST-104", only=("public", "restored", "runner", "sim_run"),
       patch={"MAX_JAX_TRACE_SECONDS": 0.1})
def _a_trace_stops_itself_at_its_time_budget(ctx):
    """A trace stops itself at its (patched) 0.1 s budget: an idle one by its
    own timer, and one the runner or a /sim/run keeps feeding steps."""
    from maddening.core.simulation import profiler

    if not ctx.steps_beside:
        advance(ctx, 1)
    try:
        resp = ctx.client.post("/sim/profile/jax/start")
        assert resp.status_code == 200, resp.text
        assert wait_for(lambda: not profiler.jax_trace_active(), 20), \
            "an idle trace did not stop at its time budget"
        status = ctx.client.get("/sim/profile/jax/status").json()
        assert status["active"] is False and "time budget" in status["stopped_by"]
    finally:
        _stop_any_trace()
