"""What the two REST oracles hold every reply to.

``test_rest_write_sequences_leave_a_graph_that_reloads.py`` generates
*sequences* of the state-changing requests, and
``test_rest_requests_generated_from_the_schema.py`` generates *single
requests* from the server's own schema, valid and systematically malformed.
Neither predicts an answer.  Each asks the same questions of whatever the
server answered, and this module is where those questions are written once:

1. **no reply is a 5xx** (:func:`assert_no_server_error`);
2. **a refused request changes nothing** (:func:`snapshot`,
   :func:`assert_nothing_changed`): after a 4xx the graph's config
   (``to_dict``), every node's own ``params``, the live ``gm.params``
   leaves, the whole state (``_meta`` included), the files under the
   checkpoint root (names and SHA-256) and the streams' clock are what they
   were;
3. **an accepted graph reloads** (:func:`rebuilt_from_its_save`): its
   config carries the values its step reads,
   ``GraphManager.from_dict(gm.to_dict())`` succeeds, and a checkpoint saved
   now loads into that rebuilt graph and restores the state bit for bit;
4. **an accepted graph is the one its save reloads**
   (:func:`assert_runs_as_its_reload`): the live graph and the rebuilt one,
   stepped side by side, give bit-identical states -- or neither can step,
   which only an edge between fields of different shapes may bring about
   (:func:`check_accepted_graph`);
5. **a stable configuration stays stable**
   (:func:`stability_violations`): a stock node whose constructor enforces
   a stability limit is never left past it, by the values its step reads
   *and* by the values a save carries;

and of the reply itself, for the request fuzzer:

6. **a 4xx detail names no absolute server path**
   (:func:`server_paths_named`), and **a 2xx JSON reply is strict JSON**
   with no ``NaN`` or ``Infinity`` token (:func:`assert_strict_json`).

What the comparisons are
------------------------
Exact, as everywhere in the differential harness
(``tests/property/differential.py``): two leaves agree when they have the
same dtype, shape and bytes.  Nothing here takes a tolerance.

What observing costs
--------------------
:func:`snapshot` reads ``gm.to_dict()`` and ``gm.params``, and reading
``gm.params`` takes pending ``node.params`` writes into the graph
(``GraphManager._sync_node_param_writes``).  Every route does that with its
own first read of the graph, so a sequence observed here is the sequence
with a ``GET /graph`` between each two requests, which any client may send.

One thing a refusal may do
--------------------------
``POST /checkpoint/load`` compiles a graph that has been edited since its
last compile before it reads the file, so a load it then refuses leaves the
graph compiled.  A compile changes no value: it builds the params pytree
from the nodes' own values (keeping every live leaf that still fits),
drops the leaves of a node that was removed, and gives the scheduler its
``_meta`` slots.  :func:`assert_nothing_changed` allows exactly that -- the
dirty flag cleared, owners and ``_meta`` slots added or dropped -- and
still requires every value present on both sides to be bit-identical.

Nothing here can reach a cloud provider: no helper sends a request to
``/cloud/*``, and both oracles run under
:func:`tests.property.differential.no_cloud_launch`.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np

import maddening
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import (
    BallNode,
    HealthCheckNode,
    HeartPumpNode,
    HeatNode,
    LBMNode,
    LBMPipeNode,
    RigidBody2DNode,
    RigidBodyNode,
    SpringDamperNode,
    TableNode,
)
from maddening.nodes.adaptive import WaveletAdaptiveNode
from maddening.nodes.heat import MAX_FOURIER_NUMBER

from tests._loopback_client import LoopbackTestClient
from tests.property.differential import (
    _same_python,
    assert_trees_identical,
    deep_copy_jsonable,
    full_state,
    leaves_identical,
    params_tree,
    quiet,
    states,
)

#: The node classes a server of either oracle can build: the cheap stock
#: nodes (a compile in tens of milliseconds).
REGISTRY: dict[str, type] = {cls.__name__: cls for cls in (
    BallNode, HeatNode, SpringDamperNode, TableNode)}

#: What a graph raises when it is asked to compile or step and its
#: configuration does not allow it (an edge between fields of different
#: shapes, which ``POST /graph/edges`` accepts): the errors
#: ``POST /sim/step`` answers 400 for.
CANNOT_STEP = (RuntimeError, ValueError, TypeError, KeyError, AttributeError,
               IndexError, ArithmeticError, ExceptionGroup)


# ---------------------------------------------------------------------------
# A served graph
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Served:
    """A ``SimulationServer``, an in-process client of it that reaches it as
    a local process does (``tests/_loopback_client.py``), and its checkpoint
    root, in a temporary directory of its own."""

    server: SimulationServer
    client: LoopbackTestClient
    root: Path
    registry: dict
    tmp: tempfile.TemporaryDirectory
    #: whether the graph could step when :func:`check_accepted_graph` last
    #: looked (a graph is served able to)
    could_step: bool = True

    @property
    def gm(self) -> GraphManager:
        return self.server.gm

    def close(self) -> None:
        try:
            self.client.__exit__(None, None, None)
        finally:
            self.tmp.cleanup()


def serve(gm: Optional[GraphManager] = None, *, registry: Optional[dict] = None,
          **server_kw: Any) -> Served:
    """A server for *gm* (an empty graph when ``None``) on a loopback bind,
    its checkpoint root an empty directory."""
    registry = dict(REGISTRY if registry is None else registry)
    tmp = tempfile.TemporaryDirectory(prefix="maddening-rest-oracle-")
    root = Path(tmp.name).resolve() / "checkpoints"
    root.mkdir()
    server = SimulationServer(registry, graph_manager=gm, checkpoint_root=str(root),
                              **server_kw)
    # ``raise_server_exceptions=False``: an unhandled exception comes back
    # as the 500 a real client would get, which is what invariant 1 reads.
    client = LoopbackTestClient(server.create_app(), raise_server_exceptions=False)
    # Entered: the app's lifespan runs, as it does under a real server, and
    # every request shares one event loop (a client that is not entered
    # starts a thread and a loop per request, most of a request's cost here).
    client.__enter__()
    return Served(server, client, root, registry, tmp)


# ---------------------------------------------------------------------------
# 1. No reply is a 5xx
# ---------------------------------------------------------------------------

def describe(method: str, url: str, sent: Any = None) -> str:
    text = "" if sent is None else f" {sent!r}"
    if len(text) > 400:
        text = text[:400] + "..."
    return f"{method} {url}{text}"


def assert_no_server_error(resp, what: str) -> None:
    assert resp.status_code < 500, (
        f"{what} -> {resp.status_code}: {resp.text[:600]}")


# ---------------------------------------------------------------------------
# 2. A refused request changes nothing
# ---------------------------------------------------------------------------

def checkpoint_files(root: Path) -> dict[str, str]:
    """Every entry under the checkpoint root: ``{relative path: SHA-256}``
    for a file, ``"<dir>"`` for a directory."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_dir():
            out[rel] = "<dir>"
        else:
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def saved_config(gm: GraphManager) -> dict:
    """``gm.to_dict()`` through JSON text, as a saved file carries it.  The
    config's own warning (live mapping weights it does not carry, which a
    checkpoint does) is not this module's business."""
    with quiet():
        return json.loads(json.dumps(gm.to_dict(), allow_nan=True))


def snapshot(served: Served) -> dict:
    """Everything a request could change."""
    gm = served.gm
    try:
        config = json.dumps(saved_config(gm), sort_keys=True, allow_nan=True)
    except Exception as exc:  # noqa: BLE001 - compared as what it is
        config = f"<to_dict raised {type(exc).__name__}: {exc}>"
    relay = served.server.relay
    return {
        "config": config,
        "node_params": {name: deep_copy_jsonable(dict(spec.node.params))
                        for name, spec in gm._nodes.items()},  # noqa: SLF001
        "params": params_tree(gm),
        "state": full_state(gm),
        "dirty": bool(gm._dirty),  # noqa: SLF001
        "files": checkpoint_files(served.root),
        "clock": (int(relay.step_count), float(relay.elapsed)),
    }


def _tree_differences(before: dict, after: dict, *, what: str, compiled: bool,
                      droppable: Callable[[str], bool]) -> list[str]:
    """Differences between two ``{owner: {key: array}}`` trees.  With
    *compiled*, an owner or key on one side only is the compile's (see the
    module docstring) when *droppable* says so; a value on both sides is
    compared either way."""
    out = []
    for owner in sorted(set(before) | set(after)):
        if owner not in before or owner not in after:
            if not (compiled and droppable(owner)):
                side = "appeared" if owner not in before else "disappeared"
                out.append(f"{what}[{owner!r}] {side}")
            continue
        b, a = before[owner], after[owner]
        for key in sorted(set(b) | set(a)):
            if key not in b or key not in a:
                if not (compiled and droppable(owner)):
                    side = "appeared" if key not in b else "disappeared"
                    out.append(f"{what}[{owner!r}][{key!r}] {side}")
                continue
            problem = leaves_identical(b[key], a[key])
            if problem is not None:
                out.append(f"{what}[{owner!r}][{key!r}]: {problem}")
    return out


def differences(before: dict, after: dict) -> list[str]:
    """What differs between two :func:`snapshot` results, as one line each;
    empty when a refusal left everything as it was."""
    out = []
    if before["config"] != after["config"]:
        out.append(f"the config changed: {before['config'][:300]} -> {after['config'][:300]}")
    if before["files"] != after["files"]:
        gone = sorted(set(before["files"]) - set(after["files"]))
        new = sorted(set(after["files"]) - set(before["files"]))
        changed = sorted(k for k in set(before["files"]) & set(after["files"])
                         if before["files"][k] != after["files"][k])
        out.append(f"the checkpoint root changed: new {new}, gone {gone}, rewritten {changed}")
    if before["clock"] != after["clock"]:
        out.append(f"the streams' clock moved {before['clock']} -> {after['clock']}")
    for name in sorted(set(before["node_params"]) | set(after["node_params"])):
        b, a = before["node_params"].get(name), after["node_params"].get(name)
        if b is None or a is None or not _same_python(b, a):
            out.append(f"node {name!r} params changed: {b!r} -> {a!r}")
    # A refusal may have compiled an edited graph, and nothing else.
    compiled = before["dirty"] and not after["dirty"]
    if before["dirty"] != after["dirty"] and not compiled:
        out.append(f"the dirty flag moved {before['dirty']} -> {after['dirty']}")
    out += _tree_differences(before["params"], after["params"], what="gm.params",
                             compiled=compiled, droppable=lambda owner: True)
    out += _tree_differences(before["state"], after["state"], what="state",
                             compiled=compiled, droppable=lambda owner: owner == "_meta")
    return out


def assert_nothing_changed(before: dict, after: dict, what: str) -> None:
    found = differences(before, after)
    assert not found, f"{what} was refused and changed the server: " + "; ".join(found)


# ---------------------------------------------------------------------------
# 3. An accepted graph reloads
# ---------------------------------------------------------------------------

def save_differs_from_what_the_step_reads(gm: GraphManager, config: dict) -> list[str]:
    """Every constructor parameter the saved *config* carries with another
    value than the live ``gm.params`` leaf the step reads (compared in the
    leaf's dtype): a save must carry the graph that runs."""
    out = []
    saved = {node["name"]: node.get("params", {}) for node in config.get("nodes", [])}
    for name, leaves in (gm.params.get("nodes") or {}).items():
        if name not in gm._nodes:  # noqa: SLF001 - a removed node's leaves, until a compile
            continue
        for key, leaf in leaves.items():
            if key not in saved.get(name, {}):
                continue
            leaf = np.asarray(leaf)
            try:
                carried = np.asarray(saved[name][key], dtype=leaf.dtype)
            except (TypeError, ValueError):
                out.append(f"the save carries {name}.{key} = {saved[name][key]!r}, which is "
                           f"not a {leaf.dtype} as the live leaf is")
                continue
            if leaves_identical(leaf, carried) is not None:
                out.append(f"the save carries {name}.{key} = {saved[name][key]!r}, the step "
                           f"reads {leaf.tolist()!r}")
    return out


#: Compiled reloads kept for reuse, by the config they were rebuilt from
#: (:func:`rebuilt_from_its_save`, ``twins=``): the reload of a config is
#: rebuilt every time -- that *is* the question -- but stepping it needs an
#: XLA compile (tens of milliseconds for these graphs, most of an example),
#: and a graph whose config has not changed since the last request, or that
#: an earlier example reached, has a compiled twin already.  The twin's
#: state and parameters are loaded from a checkpoint of the served graph
#: before every use, so what it steps from is never its own history.
TWINS: "collections.OrderedDict[str, GraphManager]" = collections.OrderedDict()
_TWINS_KEPT = 64


def rebuilt_from_its_save(served: Served, what: str, *,
                          twins: Optional[collections.OrderedDict] = None
                          ) -> Optional[GraphManager]:
    """The graph a save of the served one reloads as: its config through
    JSON and ``GraphManager.from_dict``, compiled, with a checkpoint of the
    served graph loaded on top and the state it restores checked.

    Returns ``None`` when the rebuilt graph cannot compile (its
    configuration does not allow it); the caller decides whether the served
    graph may be in that condition.  With *twins* (:data:`TWINS`) the graph
    returned is the one already compiled for this config, when there is one.
    """
    gm = served.gm
    try:
        config = saved_config(gm)
    except Exception as exc:  # noqa: BLE001 - any failure to save is the finding
        raise AssertionError(
            f"after {what} the graph cannot be saved: to_dict() raised "
            f"{type(exc).__name__}: {exc}") from exc
    try:
        with quiet():
            rebuilt = GraphManager.from_dict(config, served.registry)
    except Exception as exc:  # noqa: BLE001 - any failure to reload is the finding
        raise AssertionError(
            f"after {what} the graph's save does not reload: from_dict raised "
            f"{type(exc).__name__}: {exc}\nconfig: {json.dumps(config)[:1500]}") from exc
    carried = save_differs_from_what_the_step_reads(gm, config)
    assert not carried, f"after {what} " + "; ".join(carried)
    key = json.dumps(config, sort_keys=True, allow_nan=True)
    if twins is not None and key in twins:
        rebuilt = twins.pop(key)        # put back below, once it has been used
    try:
        with quiet():
            # A twin kept from an earlier use is compiled already, and
            # compiling it again would throw its compiled step away.
            if rebuilt._dirty or rebuilt._compiled_step is None:  # noqa: SLF001
                rebuilt.compile()
    except CANNOT_STEP:
        return None
    with tempfile.TemporaryDirectory(prefix="maddening-rest-oracle-ckpt-") as tmp:
        try:
            ckpt = gm.save_state(Path(tmp) / "live.npz")
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"after {what} the graph cannot be checkpointed: save_state raised "
                f"{type(exc).__name__}: {exc}") from exc
        try:
            with quiet():
                rebuilt.load_state(ckpt)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"after {what} a checkpoint of the graph does not load into the graph "
                f"its config rebuilds: load_state raised {type(exc).__name__}: {exc}") from exc
    assert_trees_identical(states(gm), states(rebuilt),
                           what=f"after {what}: the state the reload restores")
    if twins is not None:
        twins[key] = rebuilt
        while len(twins) > _TWINS_KEPT:
            twins.popitem(last=False)
    return rebuilt


# ---------------------------------------------------------------------------
# 4. An accepted graph is the one its save reloads
# ---------------------------------------------------------------------------

#: Steps the live graph and its reload take side by side.
RELOAD_STEPS = 2


@contextlib.contextmanager
def unobserved(gm: GraphManager) -> Iterator[None]:
    """Step *gm* without its observers (a server's relay counts every step
    it sees into the streams' clock)."""
    observers = getattr(gm, "_observers", None)
    held = list(observers) if observers is not None else []
    if observers is not None:
        observers.clear()
    try:
        yield
    finally:
        if observers is not None:
            observers[:] = held


def assert_runs_as_its_reload(served: Served, rebuilt: GraphManager, what: str,
                              n_steps: int = RELOAD_STEPS) -> None:
    """The served graph and *rebuilt* (:func:`rebuilt_from_its_save`), each
    stepped *n_steps*, reach bit-identical states; the served graph is then
    put back exactly where it was.  Asked of a compiled graph only: stepping
    one that is waiting for a compile would compile it, which is not the
    oracle's to do."""
    gm = served.gm
    assert not gm._dirty and gm._compiled_step is not None  # noqa: SLF001
    assert_trees_identical(full_state(gm), full_state(rebuilt),
                           what=f"after {what}: restored state (with _meta)")
    with tempfile.TemporaryDirectory(prefix="maddening-rest-oracle-ckpt-") as tmp:
        ckpt = gm.save_state(Path(tmp) / "back.npz")
        held_state, held_params = full_state(gm), params_tree(gm)
        try:
            with unobserved(gm), quiet():
                gm.run(n_steps)
                live = states(gm)
            with quiet():
                rebuilt.run(n_steps)
            assert_trees_identical(live, states(rebuilt),
                                   what=f"after {what}: {n_steps} steps of the live graph "
                                        "against its reload")
        finally:
            with unobserved(gm), quiet():
                gm.load_state(ckpt)
    assert_trees_identical(held_state, full_state(gm),
                           what="a checkpoint of the live graph, loaded back: state")
    assert_trees_identical(held_params, params_tree(gm),
                           what="a checkpoint of the live graph, loaded back: params")


def check_accepted_graph(served: Served, what: str, *, shapes_may_differ: bool = False,
                         twins: Optional[collections.OrderedDict] = None,
                         step: bool = True) -> str:
    """Invariants 3, 4 and 5 of the served graph as it stands, after a
    request the server accepted (*what*).  Returns what was reached:
    ``"stepped"`` (the graph is compiled and was stepped beside its reload),
    ``"reloaded"`` (it is waiting for a compile, or *step* is off) or
    ``"cannot step"``.

    A graph that cannot compile is an accepted graph in one case only:
    ``POST /graph/edges`` takes an edge between fields of different shapes
    (*shapes_may_differ*), and the graph then answers 400 to every step
    until the edge is removed.  Any other accepted request that turns a
    graph that could step into one that cannot -- a node added with a
    timestep the scheduler cannot use, say -- fails here."""
    gm = served.gm
    assert_within_stability_limits(gm, what)
    rebuilt = rebuilt_from_its_save(served, what, twins=twins)
    compiled = not gm._dirty and gm._compiled_step is not None  # noqa: SLF001
    if rebuilt is None:
        assert not compiled, (
            f"after {what} the served graph is compiled, but the graph its save "
            "reloads cannot compile")
        assert not (served.could_step and gm._nodes and not shapes_may_differ), (  # noqa: SLF001
            f"after {what} the graph, which could step, cannot compile any more: an "
            "accepted request left a graph that answers 400 to every step")
        served.could_step = False
        return "cannot step"
    served.could_step = True
    if compiled and step:
        assert_runs_as_its_reload(served, rebuilt, what)
        return "stepped"
    return "reloaded"


def assert_the_reload_cannot_step_either(served: Served, what: str) -> None:
    """The served graph answered that it cannot step: the graph its save
    reloads must not step either."""
    try:
        with quiet():
            rebuilt = GraphManager.from_dict(saved_config(served.gm), served.registry)
            rebuilt.compile()
            rebuilt.step()
    except CANNOT_STEP:
        return
    raise AssertionError(
        f"{what} answered that the graph cannot step, but the graph its save "
        "reloads compiles and steps")


# ---------------------------------------------------------------------------
# 5. A stable configuration stays stable
# ---------------------------------------------------------------------------

def heat_fourier(dt: float, p: dict) -> tuple[float, float, str]:
    """``(Fourier number, its limit, the form it takes)`` of a ``HeatNode``
    with params *p* at timestep *dt*, restated from the node's documented
    hazard: ``dt*alpha/dx**2`` with ``dx = length/n_cells`` against the
    limit of its stencil order (``MAX_FOURIER_NUMBER``) on a uniform grid,
    ``dt*alpha/min(h_left*h_right)`` against the order-2 limit on a
    non-uniform one."""
    alpha, n = float(np.asarray(p["thermal_diffusivity"])), int(p["n_cells"])
    grid = p.get("grid_points")
    if grid is None:
        dx = float(np.asarray(p["length"])) / n
        return (dt * alpha / (dx * dx), MAX_FOURIER_NUMBER[int(p["stencil_order"])],
                "dt*alpha/dx**2")
    x = [float(v) for v in np.asarray(grid).ravel()]
    h = [b - a for a, b in zip(x, x[1:])]
    padded = [h[0], *h, h[-1]]
    spacing = min(a * b for a, b in zip(padded, padded[1:]))
    return dt * alpha / spacing, MAX_FOURIER_NUMBER[2], "dt*alpha/min(h_left*h_right)"


def _heat_limit(dt: float, p: dict) -> Optional[str]:
    """``HeatNode``: the explicit scheme's Fourier number against its limit."""
    fourier, limit, form = heat_fourier(dt, p)
    if not fourier <= limit:
        return f"Fourier number {form} = {fourier:.6g} is above its limit {limit:g}"
    return None


def _lbm_limit(dt: float, p: dict) -> Optional[str]:
    """``LBMNode``: BGK needs ``tau = 0.5 + viscosity/cs2 > 0.5``, that is a
    finite viscosity above zero (``cs2`` is 1/3 on both lattices)."""
    viscosity = float(np.asarray(p["viscosity"]))
    if not (np.isfinite(viscosity) and 0.5 + 3.0 * viscosity > 0.5):
        return f"tau = 0.5 + viscosity/cs2 is not above 0.5 (viscosity {viscosity!r})"
    return None


def _pipe_limit(dt: float, p: dict) -> Optional[str]:
    """``LBMPipeNode``: both relaxation times finite and above 0.5."""
    for key in ("tau", "tau_tracer"):
        if key in p and p[key] is not None:
            tau = float(np.asarray(p[key]))
            if not (np.isfinite(tau) and tau > 0.5):
                return f"{key} = {tau!r} is not above 0.5"
    return None


#: Every stock node class, and the stability limit its constructor enforces
#: (``None``: it enforces none).  ``test_every_stock_node_is_classified``
#: fails when ``maddening.nodes`` exports a class this does not list, so a
#: node that arrives with a limit cannot go unchecked.
STABILITY_LIMITS: dict[type, Optional[Callable[[float, dict], Optional[str]]]] = {
    HeatNode: _heat_limit,
    LBMNode: _lbm_limit,
    LBMPipeNode: _pipe_limit,
    BallNode: None,
    HealthCheckNode: None,
    HeartPumpNode: None,
    RigidBody2DNode: None,
    RigidBodyNode: None,
    SpringDamperNode: None,
    TableNode: None,
    WaveletAdaptiveNode: None,
}


def read_by_the_step(gm: GraphManager, name: str) -> dict:
    """The parameter values node *name*'s step reads: its own, with the
    live ``gm.params`` leaves over them."""
    spec = gm._nodes[name]  # noqa: SLF001
    node = getattr(spec.node, "physics_node", spec.node)
    live = gm.params.get("nodes", {}).get(name) or {}
    return {**dict(node.params), **{k: np.asarray(v) for k, v in live.items()}}


def limit_fraction(gm: GraphManager, name: str, changes: Optional[dict] = None
                   ) -> Optional[float]:
    """How far node *name* is towards its stability limit, as a fraction of
    it (1 is the limit), with *changes* written over the values its step
    reads; ``None`` for a node with no such number, or values it cannot be
    computed from.  For a generator that wants to write *towards* a limit:
    only ``HeatNode``'s is a number that several parameters move."""
    node = getattr(gm._nodes[name].node, "physics_node", gm._nodes[name].node)  # noqa: SLF001
    if type(node) is not HeatNode:
        return None
    try:
        fourier, limit, _ = heat_fourier(float(node.delta_t),
                                         {**read_by_the_step(gm, name), **(changes or {})})
    except Exception:  # noqa: BLE001 - a value the number cannot be computed from
        return None
    return fourier / limit


def stability_violations(gm: GraphManager) -> list[str]:
    """Every node of *gm* past the stability limit its constructor
    enforces: by the values its step reads (the live ``gm.params`` leaves
    over the node's own), and by the values a save carries
    (``effective_node_params``)."""
    out = []
    for name, spec in gm._nodes.items():  # noqa: SLF001
        node = getattr(spec.node, "physics_node", spec.node)
        limit = STABILITY_LIMITS.get(type(node))
        if limit is None:
            continue
        views = {
            "the values its step reads": read_by_the_step(gm, name),
            "the values a save carries": gm.effective_node_params(name),
        }
        for view, params in views.items():
            try:
                problem = limit(float(node.delta_t), params)
            except Exception as exc:  # noqa: BLE001 - a value the limit cannot read
                problem = f"its limit cannot be evaluated ({type(exc).__name__}: {exc})"
            if problem is not None:
                out.append(f"node {name!r} ({type(node).__name__}, dt={node.delta_t!r}), "
                           f"by {view}: {problem}")
    return out


def assert_within_stability_limits(gm: GraphManager, what: str) -> None:
    found = stability_violations(gm)
    assert not found, f"after {what} " + "; ".join(found)


# ---------------------------------------------------------------------------
# 6. The reply itself
# ---------------------------------------------------------------------------

def _forbid_constant(token: str):
    raise AssertionError(f"the reply holds the bare token {token}")


def assert_strict_json(resp, what: str) -> Any:
    """A 2xx JSON reply parses as strict JSON: no bare ``NaN``,
    ``Infinity`` or ``-Infinity``."""
    try:
        return json.loads(resp.text, parse_constant=_forbid_constant)
    except AssertionError as exc:
        raise AssertionError(f"{what} -> {resp.status_code}: {exc}: "
                             f"{resp.text[:300]}") from None
    except ValueError as exc:
        raise AssertionError(f"{what} -> {resp.status_code}: the reply is not JSON "
                             f"({exc}): {resp.text[:300]}") from None


def server_roots(served: Optional[Served] = None) -> list[str]:
    """Directories whose names only the server knows: its checkpoint root
    and temporary directory, the installed package, the interpreter's
    prefix and standard library, the working directory, the temporary
    directory and ``HOME``."""
    roots = {str(Path(maddening.__file__).resolve().parent), sys.prefix, sys.base_prefix,
             os.path.dirname(os.__file__), os.getcwd(), tempfile.gettempdir(),
             os.path.expanduser("~")}
    if served is not None:
        roots |= {str(served.root), str(Path(served.tmp.name)), str(Path(served.tmp.name).resolve())}
    return sorted((r for r in roots if r and r != "/"), key=len, reverse=True)


def server_paths_named(text: str, sent: str, served: Optional[Served] = None) -> list[str]:
    """The directories of this deployment (:func:`server_roots`) that
    *text* (a 4xx detail) names -- unless the request itself carried the
    name (*sent*: the request's URL, headers and body), in which case the
    server told the client nothing.  A path every machine has, named in
    prose (``/etc/hosts``), is not one of them."""
    return [root for root in server_roots(served) if root in text and root not in sent]


def detail_text(resp) -> str:
    """A reply's ``detail`` as text (the whole body when it has none)."""
    try:
        body = resp.json()
    except ValueError:
        return resp.text
    detail = body.get("detail", body) if isinstance(body, dict) else body
    return detail if isinstance(detail, str) else json.dumps(detail)
