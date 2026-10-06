"""A failure injected into the middle of a REST write route's body.

"A request that fails leaves the graph exactly as it was" is a claim about
every point of a route's body, and a route reaches most of them only when
something raises where nothing is expected to.  :func:`injection_points`
wraps the steps the write routes call -- the graph's mutators, its compile,
step and store, the server's publish and reset, the relay's restore, the
reply encoder, each write to a live node's ``params`` -- and an
:class:`Injector` raises :class:`InjectedFailure` at the *k*-th point a
request reaches: just before a step, or just after it.

:func:`fail_at_every_point` sends one request again and again, failing at
point 0, 1, 2, ... and asking after each that nothing changed, until the
request reaches no armed point and runs to its end: that last reply is the
request's own.  So every request is tried with a failure at every point
between its first mutation and its last (and at the points before and
after them).

Only the served graph, its server and its relay are counted: a graph a
route builds to ask a question of (a reload, a dry run) is not the one the
claim is about.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable, Iterator, Optional

import maddening.api.server as server_module
import maddening.core.simulation.checkpoint as checkpoint_module
import maddening.surrogates.replace as replace_module
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.core.node import _ParamsDict
from maddening.viz.relay import StateRelay

from tests.property import rest_oracle as O
from tests.property.graph_fingerprint import assert_exactly_as_it_was, served_fingerprint


class InjectedFailure(Exception):
    """Raised at an armed point.  Not one of the errors a route reads as
    the graph's own (``_GRAPH_CONFIGURATION_ERRORS``), so a route that does
    not expect it answers 500."""


#: The methods wrapped, per class: each call is two points, before and after.
GRAPH_STEPS = ("add_node", "remove_node", "add_edge", "remove_edge", "add_external_input",
               "set_param_spec", "compile", "step", "run", "set_node_state", "reset_state",
               "load_state", "save_state", "_store_state")
SERVER_STEPS = ("_publish_state", "_reset_state", "_state_json", "_ensure_relay_attached")
RELAY_STEPS = ("restore", "reset")
#: Module-level functions the routes look up when they call them.
MODULE_STEPS = ((server_module, "_json_reply"),
                (checkpoint_module, "_restore_state_and_params"),
                (replace_module, "replace_node"))

_armed: Optional["Injector"] = None


class Injector:
    """Counts the points a request reaches on one served graph, and raises
    at the one it is armed for."""

    def __init__(self, served: O.Served) -> None:
        self.served = served
        self.at: Optional[int] = None
        self.count = 0
        self.fired: Optional[str] = None
        self._params: set[int] = set()

    def arm(self, at: Optional[int]) -> None:
        """Fail at point *at* of the next request (``None``: count only)."""
        global _armed
        self.at, self.count, self.fired = at, 0, None
        gm = self.served.gm
        self._params = {id(spec.node.params) for spec in gm._nodes.values()}  # noqa: SLF001
        _armed = self

    def disarm(self) -> Optional[str]:
        """Stop counting; the point that fired, or ``None``."""
        global _armed
        _armed = None
        return self.fired

    def owns(self, obj: Any) -> bool:
        served = self.served
        return (obj is served.gm or obj is served.server or obj is served.server.relay
                or id(obj) in self._params)

    def point(self, label: str) -> None:
        index = self.count
        self.count += 1
        if index == self.at:
            self.fired = f"point {index}, {label}"
            raise InjectedFailure(f"injected failure at {self.fired}")


def _wrap(name: str, original: Callable, *, owned: bool) -> Callable:
    @functools.wraps(original)
    def step(*args, **kwargs):
        injector = _armed
        if injector is None or (owned and not (args and injector.owns(args[0]))):
            return original(*args, **kwargs)
        injector.point(f"before {name}")
        result = original(*args, **kwargs)
        injector.point(f"after {name}")
        return result
    return step


@contextlib.contextmanager
def injection_points() -> Iterator[None]:
    """Wrap every step for the duration (a module-scoped fixture's): with
    no injector armed, each wrapper is one ``None`` check."""
    targets: list[tuple[Any, str, bool]] = (
        [(GraphManager, name, True) for name in GRAPH_STEPS]
        + [(SimulationServer, name, True) for name in SERVER_STEPS]
        + [(StateRelay, name, True) for name in RELAY_STEPS]
        + [(_ParamsDict, "__setitem__", True)]
        + [(module, name, False) for module, name in MODULE_STEPS])
    originals = [(owner, name, owner.__dict__[name]) for owner, name, _ in targets]
    try:
        for owner, name, owned in targets:
            label = name if not owned else f"{owner.__name__}.{name}"
            setattr(owner, name, _wrap(label, getattr(owner, name), owned=owned))
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def fail_at_every_point(served: O.Served, send: Callable[[], Any], what: str, *,
                        partial: Callable[[Any], int] = lambda resp: 0,
                        on_failure: Optional[Callable[[str, Any], None]] = None,
                        max_points: int = 400):
    """Send one request with a failure injected at each point it reaches in
    turn, then without one; return that last reply, the points tried, and
    the oracle's snapshot and the fingerprint taken just before the last
    request (for the caller to hold the request's own reply to).

    After every injected failure the reply is a refusal (a 500 carries the
    generic detail and nothing else of the server's), and the graph is
    exactly as it was: by the oracle's own snapshot, and object for object
    (:func:`~tests.property.graph_fingerprint.served_fingerprint`).

    *partial*: how many steps a reply says the request took before it
    failed (``POST /sim/run``, the one refusal documented to leave the
    graph changed): the streams' clock must then have advanced by exactly
    that many steps, the slice that failed put back.
    """
    injector = Injector(served)
    tried: list[str] = []
    for at in range(max_points):
        before = O.snapshot(served)
        fingerprint = served_fingerprint(served)
        injector.arm(at)
        try:
            resp = send()
        finally:
            fired = injector.disarm()
        if fired is None:
            return resp, tried, before, fingerprint
        tried.append(fired)
        where = f"{what}, with a failure injected at {fired},"
        if resp.status_code < 400:
            # A route may take a failure of an optional step in its stride
            # (a manifest's clock it cannot read): then this was the request.
            return resp, tried, before, fingerprint
        if resp.status_code >= 500:
            assert resp.status_code == 500, f"{where} -> {resp.status_code}"
            detail = resp.json()["detail"]
            assert "failed unexpectedly" in detail or "Could not restore" in detail, (where, detail)
            assert "InjectedFailure" not in resp.text and "injected" not in resp.text, (
                f"{where} named the failure in its reply: {resp.text[:300]}")
        if on_failure is not None:
            on_failure(fired, resp)
        steps = partial(resp)
        after = O.snapshot(served)
        if steps:
            assert after["clock"][0] == before["clock"][0] + steps, (
                f"{where} said it took {steps} step(s); the streams' clock went "
                f"{before['clock']} -> {after['clock']}")
            continue
        O.assert_nothing_changed(before, after, where)
        assert_exactly_as_it_was(fingerprint, served_fingerprint(served), where)
    raise AssertionError(f"{what} reached more than {max_points} injection points")
