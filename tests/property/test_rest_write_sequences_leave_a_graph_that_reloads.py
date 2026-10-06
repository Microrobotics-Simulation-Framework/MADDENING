"""Any sequence of REST writes leaves a graph that runs as its save reloads.

The REST tests are fixed scenarios, and two audit rounds in a row found a
defect that only a *sequence* reaches: ``PUT /graph/params``, save, PUT,
load, then PUT a value back to its original left a rod past its Fourier
limit with a 200 (MADD-ANO-178); and a checkpoint loaded into a graph whose
node had since been rebuilt restored a value the params route refuses
(MADD-ANO-163).  This module generates the sequences nobody wrote down.  A
:class:`~hypothesis.stateful.RuleBasedStateMachine` drives one server
through the state-changing routes in whatever order Hypothesis picks --

* add and remove a node, add and remove an edge;
* ``PUT /graph/params``, one key and several, with ordinary values, the
  values a client returns to (the node's own, the one it was built with, an
  earlier write) and values the route should refuse;
* ``PUT /graph/state``;
* ``POST /sim/step``, ``/sim/run`` with a small count, ``/sim/reset``,
  ``/graph/compile``;
* ``POST /checkpoint/save``, and ``/checkpoint/load`` of an earlier save;
* a direct ``gm.params`` write between two requests, as a fit makes one

-- and predicts no answer.  Whatever the server replies, the invariants of
``tests/property/rest_oracle.py`` are asked after every rule:

1. no reply is a 5xx, and a 2xx is strict JSON;
2. a refused request changes nothing: the config, every parameter, the
   whole state, the files under the checkpoint root and the streams' clock
   are bit for bit what they were;
3. an accepted graph reloads: ``GraphManager.from_dict(gm.to_dict())``
   succeeds, and a checkpoint saved now loads into the rebuilt graph;
4. an accepted graph is the one its save reloads: stepped side by side,
   the two give bit-identical states (asked whenever the served graph is
   compiled, and of every example's last graph; a graph whose
   configuration cannot step must be one whose reload cannot either);
5. a stable configuration stays stable: no ``HeatNode`` is left past its
   Fourier limit, by the values its step reads or by those a save carries;
6. a request that *fails unexpectedly* changes nothing either.  Every
   request is first sent with a failure injected into the route's body --
   at each point it reaches in turn, before and after every step that
   mutates the graph, publishes to the streams or writes the reply
   (``tests/property/injected_failures.py``) -- and after each the graph
   must be exactly as it was: by invariant 2's snapshot, and object for
   object (``tests/property/graph_fingerprint.py``).  A refusal of the
   request's own is held to the same object-for-object comparison;
7. on a server that demands the token, the same request *without* it is
   refused and reveals nothing.  Half the examples serve their graph as a
   network bind requires -- the server told its bind is ``0.0.0.0``, the
   client presenting the token on every request -- and are held to
   invariants 1 to 6 as the others are; and before each request the same
   request is sent in one of the ways of not presenting the token
   (``tests/property/without_the_token.py``): 401, nothing changed, and a
   body that is the one a server with an empty graph gives.

What makes the historical sequences reachable is the vocabulary.  A rod is
drawn *at* its limit, not safely inside it: its diffusivity, length,
stencil order and timestep come from short lists chosen so that about half
their combinations are past the limit (:data:`VOCABULARY`), and a write
returns to a value the graph has already held as often as it proposes a new
one.  Node names and checkpoint names come from pools of four and three, so
a node is rebuilt under its old name and an old save is loaded.

What it cannot see: a defect in which the served graph and its reload agree
on a wrong value, a sequence the rules cannot spell (the runner, the
streams, the surrogate routes and ``/cloud/*`` are out of scope and never
requested), and concurrency -- one request at a time.  No socket is
opened: the server that demands the token is *told* its bind, and its
client is in process.

Nothing here can reach a cloud provider: no rule sends a request to
``/cloud/*``, and the module runs under
:func:`tests.property.differential.no_cloud_launch`.
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
from typing import Any, Optional
from urllib.parse import quote

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import event as hypothesis_event
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

import maddening.nodes
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode, HeatNode, SpringDamperNode

from tests.conftest import EXAMPLES_FLOOR
from tests.property import rest_oracle as O
from tests.property.differential import (
    assert_trees_identical,
    full_state,
    no_cloud_launch,
    note,
    params_tree,
    quiet,
)
from tests.property.graph_fingerprint import assert_exactly_as_it_was, served_fingerprint
from tests.property.injected_failures import fail_at_every_point, injection_points
from tests.property.node_catalogue import f32
from tests.property.without_the_token import CREDENTIALS, assert_refused_without_the_token


@pytest.fixture(scope="module", autouse=True)
def _no_cloud():
    with no_cloud_launch(), injection_points():
        yield


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------

#: The timestep of the start graph, and the cell count of every rod (so a
#: checkpoint of one rod fits another).
DT = 0.01
N_CELLS = 8


def _alpha(fourier: float) -> float:
    """The diffusivity that puts a rod of :data:`N_CELLS` cells and length
    1 at Fourier number *fourier* at :data:`DT`; float32-exact, so a value
    written over REST is stored as the number that was sent."""
    return f32(fourier * (1.0 / N_CELLS) ** 2 / DT)


#: A rod's diffusivities, as Fourier numbers 0.05, 0.15, 0.24 and 0.45 at
#: length 1 and :data:`DT` (the order-2 limit is 1/2, the order-4 limit
#: 5/16).  A length of 0.75 multiplies each by 1.78 and one of 0.5 by 4, a
#: timestep of 0.02 by 2: 34 of the 48 combinations with the two stencil
#: orders are past their limit.
ALPHAS = tuple(_alpha(f) for f in (0.05, 0.15, 0.24, 0.45))
LENGTHS = (1.0, 0.75, 0.5)
TIMESTEPS = (DT, DT, 0.02)
#: Timesteps a new node should not be given.
AWKWARD_TIMESTEPS = (0, -DT, float("nan"), "fast")
_RAMP = tuple(f32(x) for x in np.linspace(0.0, 1.0, N_CELLS))

#: Per node class: the ordinary values of each constructor parameter.
VOCABULARY: dict[str, dict[str, tuple]] = {
    "HeatNode": {
        "thermal_diffusivity": ALPHAS,
        "length": LENGTHS,
        "stencil_order": (2, 4),
        "n_cells": (N_CELLS, N_CELLS, N_CELLS, 6),
        # Mostly the start graph's: the parameter is a leaf the step does
        # not read, so a checkpoint fits a rebuilt rod only when both rods
        # were given one value of it.
        "initial_temperature": (1.0, 1.0, 1.0, 1.0, 0.5, list(_RAMP)),
    },
    "SpringDamperNode": {
        "stiffness": (0.5, 20.0, 40.0, 100.0),
        "damping": (0.0, 0.5, 2.0),
        "mass": (0.5, 1.0, 2.0),
        "rest_length": (0.5, 1.0),
        "initial_position": (-0.5, 0.0, 0.5),
        "initial_velocity": (0.0, 0.5),
    },
    "BallNode": {
        "initial_position": (0.0, 1.0, 3.0),
        "initial_velocity": (-1.0, 0.0, 1.0),
        "elasticity": (0.0, 0.5, 1.0),
        "gravity": (f32(-9.81), -1.0, 0.0),
    },
    "TableNode": {"position": (-1.0, 0.0, 1.0)},
}
#: The parameters a node's stability limit reads: drawn three times as
#: often as its others.
LIMIT_KEYS = {"HeatNode": ("thermal_diffusivity", "length", "stencil_order")}
#: The constants a fit moves: ones the node's step reads whatever the graph
#: around it.  Not an initial condition, which no step reads (the graph
#: refuses to step with one moved in ``gm.params`` alone), and not a
#: ball's elasticity, which its step reads only through a connected
#: ``table_position``.
FITTED = {"HeatNode": ("thermal_diffusivity", "length"),
          "SpringDamperNode": ("stiffness", "damping", "mass", "rest_length"),
          "BallNode": ("gravity",)}
#: Parameters a new node is always given: a rod left at its default
#: temperature of zero, between rod ends at zero, never moves, and a graph
#: that never moves steps as any reload of it does.
ALWAYS_GIVEN = {"HeatNode": ("initial_temperature",)}
#: Values a route should refuse for most parameters (it is not told so:
#: the invariants are asked of whatever it answers).
AWKWARD: tuple = (float("nan"), float("inf"), -1.0, 0, 50.0, "a string", True, None,
                  [1.0, 2.0], 1e39, 1e-50, 2 ** 40, 2.5, 3, 16, {"x": 1.0})
#: Node names: four, so a node is rebuilt under a name a checkpoint knows.
NAMES = ("rod", "spring", "ball", "extra")
#: Names a new node should not be given: a delimiter of the checkpoint and
#: edge keys, and a name the state reserves.
AWKWARD_NAMES = ("bad/name", "_meta")
#: The class a name's node usually is (so a rebuilt node usually fits the
#: checkpoints of the one it replaces).
USUAL_TYPE = {"rod": "HeatNode", "spring": "SpringDamperNode", "ball": "BallNode",
              "extra": "HeatNode"}
#: Checkpoint names under the root (the last has no suffix; ``.npz`` is
#: appended), and two a save or load must refuse.  ``start.npz`` is saved
#: before the first rule, so every example has an earlier save to load.
START_SLOT = "start.npz"
SLOTS = ("a.npz", "b.npz", "nested/c")
REFUSED_SLOTS = ("../escaped.npz", "")
#: State values, float32-exact; the last is the largest float32, from which
#: a spring's next step overflows, so that replies carry non-finite values
#: (which a reply writes as quoted tokens, never as bare ``Infinity``).
LARGEST = float(np.finfo(np.float32).max)
STATE_VALUES = (-1.0, 0.0, 0.5, 2.0) * 4 + (LARGEST,)
#: What a state write does to one field of an otherwise valid body, beside
#: leaving a field out or adding one.
STATE_CHANGES = {"shape": [1.0, 2.0, 3.0], "nan": float("nan"), "string": "a string",
                 "null": None, "huge": 10 ** 400, "boolean": True}


def rod_graph() -> GraphManager:
    """The start graph: a rod at Fourier number 0.45 (its limit is 0.5), a
    spring and a ball the spring's position feeds, compiled."""
    gm = GraphManager()
    gm.add_node(HeatNode("rod", DT, n_cells=N_CELLS, length=1.0,
                         thermal_diffusivity=ALPHAS[-1], initial_temperature=1.0))
    gm.add_node(SpringDamperNode("spring", DT, stiffness=40.0, damping=0.5,
                                 initial_position=0.5))
    gm.add_node(BallNode("ball", DT, initial_position=3.0))
    gm.add_edge("spring", "ball", "position", "table_position")
    with quiet():
        gm.compile()
    return gm


#: The graphs an example starts from.  The rod graph is the one the
#: vocabulary is built around; the others are the kinds of graph the
#: routes have no request to build and users serve: a coupling group under
#: each solver, a mapped edge of each static kind, and nodes at three rates.
START_GRAPHS: dict[str, Any] = {
    "rod": rod_graph,
    "empty": GraphManager,
    **O.COUPLED_AND_MAPPED_GRAPHS,
}
assert O.ROD_CELLS == N_CELLS and O.ROD_DT == DT and set(O.ROD_ALPHAS) <= set(ALPHAS)
#: The starts beside the rod graph and the empty one.
COUPLED_AND_MAPPED = tuple(sorted(set(START_GRAPHS) - {"rod", "empty"}))


def start_graph(start: str) -> GraphManager:
    with quiet():
        gm = START_GRAPHS[start]()
        if gm._nodes and gm._dirty:  # noqa: SLF001
            gm.compile()
    return gm


def event(text: str) -> None:
    """``hypothesis.event`` inside a generated example, nothing outside one
    (a pinned sequence drives the same requests as an ordinary test)."""
    try:
        hypothesis_event(text)
    except InvalidArgument:
        pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "shape"):
        return np.asarray(value).tolist()
    return value


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(np.array_equal(np.asarray(a, dtype=np.float64),
                                   np.asarray(b, dtype=np.float64)))
    except (TypeError, ValueError):
        return a == b


def _partial_run(resp) -> int:
    """The steps a ``POST /sim/run`` says it took before its 4xx, 500 or
    503: the one refusal that is documented to leave the graph changed.
    Zero for any other reply."""
    try:
        body = resp.json()
    except ValueError:
        return 0
    return int(body.get("steps_run") or 0) if isinstance(body, dict) else 0


# ---------------------------------------------------------------------------
# The machine
# ---------------------------------------------------------------------------

class RestWriteSequences(RuleBasedStateMachine):
    """One server, driven through its write routes; see the module docstring.

    Every rule is a thin draw around a ``do_*`` method that takes plain
    arguments, so a sequence the machine found -- or one an audit did --
    can be replayed through the same requests and the same invariants
    (the pinned sequences at the end of this module)."""

    #: What a run exercised, over every example: ``(rule, "accepted" |
    #: "refused")`` and the invariants reached.  Read by the per-push test,
    #: which fails if the generator has gone vacuous.
    counts: collections.Counter = collections.Counter()

    def __init__(self) -> None:
        super().__init__()
        self.served: Optional[O.Served] = None
        self.saved: set[str] = set()
        #: (node, key) -> the JSON values written to it and accepted
        self.history: dict = collections.defaultdict(list)
        #: (node, key) -> the value the node was built with
        self.built_with: dict = {}
        self.last = "the start"
        #: the configuration the graph is served in (``rest_oracle.BINDS``)
        self.bind = "loopback"
        #: the kind of the last request sent ("add edge", "put params", ...)
        self.last_kind = "start"
        #: whether the last rule and its invariants ran to their end
        self.settled = False
        #: whether anything was accepted since invariants 3 to 5 were last
        #: asked: a refusal changed nothing (invariant 2 has just said so),
        #: so the graph is the one they were asked of already
        self.unchecked = True

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------

    def begin(self, start: str, bind: str = "loopback") -> None:
        assert bind in O.BINDS, bind
        gm = start_graph(start)
        self.bind = bind
        type(self).counts[f"started from {start}"] += 1
        self.served = O.serve(gm, token_enforced=bind == "token")
        for name, spec in gm._nodes.items():  # noqa: SLF001
            self._remember_construction(name, spec.node)
        self.last = f"the start ({start}, {bind})"
        if gm._nodes:  # noqa: SLF001
            assert self.do_save(START_SLOT).status_code == 200

    def _remember_construction(self, name: str, node) -> None:
        for key in list(self.history):
            if key[0] == name:
                del self.history[key]
        for key in [k for k in self.built_with if k[0] == name]:
            del self.built_with[key]
        for key, value in dict(node.params).items():
            self.built_with[(name, key)] = _jsonable(value)

    @property
    def gm(self) -> GraphManager:
        assert self.served is not None
        return self.served.gm

    def send(self, rule_name: str, method: str, url: str, *, body: Any = None,
             params: Optional[dict] = None):
        """One request, held to invariants 1 and 2."""
        assert self.served is not None
        assert not url.startswith(("/cloud", "/surrogate", "/ws")), url
        self.settled = False
        kwargs: dict = {}
        if body is not None:
            # NaN and Infinity spelled as Python's encoder (and a careless
            # client) spells them: the server's parser reads them.
            kwargs = {"content": json.dumps(body, allow_nan=True),
                      "headers": {"content-type": "application/json"}}
        if params is not None:
            kwargs["params"] = params
        what = O.describe(method, url, body if body is not None else params)
        served = self.served
        counts = type(self).counts
        if served.token_enforced:
            # Invariant 7, before the request itself: the same request
            # without the token (each way of not presenting it in turn).
            turn = counts["sent without the token"]
            counts["sent without the token"] += assert_refused_without_the_token(
                served, method, url, what,
                credentials=(CREDENTIALS[turn % len(CREDENTIALS)],), **kwargs)

        def moved_on(fired, failed):
            # A run that failed after some of its steps left the graph
            # after them: invariants 3 to 5 are asked of it again.
            note(f"{what} with a failure injected at {fired} -> {failed.status_code}")
            if _partial_run(failed):
                self.unchecked = True

        # Invariant 6, then the request itself.
        resp, injected, before, fingerprint = fail_at_every_point(
            served, lambda: served.client.request(method, url, **kwargs), what,
            partial=_partial_run, on_failure=moved_on)
        counts["failures injected"] += len(injected)
        counts[f"failures injected ({self.bind})"] += len(injected)
        counts[(rule_name, "failed at a point")] += bool(injected)
        note(f"{what} -> {resp.status_code} {resp.text[:300]}")
        O.assert_no_server_error(resp, what)
        if resp.status_code < 300:
            O.assert_strict_json(resp, what)
        outcome = "accepted" if resp.status_code < 400 else "refused"
        event(f"{rule_name}: {outcome}")
        counts[(rule_name, outcome)] += 1
        counts[f"{outcome} ({self.bind})"] += 1
        if resp.status_code >= 400 and not _partial_run(resp):
            O.assert_nothing_changed(before, O.snapshot(self.served), what)
            assert_exactly_as_it_was(fingerprint, served_fingerprint(self.served),
                                     f"{what}, refused {resp.status_code},")
        else:
            self.unchecked = True
            self.last_kind = rule_name
        self.last = what
        return resp

    def check_graph(self, what: str) -> None:
        """Invariants 3, 4 and 5 of the served graph as it stands."""
        assert self.served is not None
        reached = O.check_accepted_graph(
            self.served, what, twins=O.TWINS,
            # the start graph is what it is; an edge may join two shapes
            shapes_may_differ=self.last_kind in ("add edge", "start"))
        counts = type(self).counts
        if reached == "cannot step":
            counts["a graph that cannot compile"] += 1
        else:
            counts["reloaded"] += 1
            counts["stepped beside its reload"] += reached == "stepped"

    def _nodes(self) -> list[str]:
        return list(self.gm._nodes)  # noqa: SLF001

    def _withheld(self, name: str, params: dict) -> bool:
        """Whether a parameter write is one the machine does not send: a
        rod's ``length`` when a mapped edge was built from its grid.  The
        route refuses such a write, except after a fit or a load left
        another leaf away from the node's own value (pinned below as a
        strict xfail; drop this with it).  The refusal itself is in the
        tour of the mapped graphs."""
        mapped = {end for edge in self.gm._edges if edge.mapping is not None  # noqa: SLF001
                  for end in (edge.source_node, edge.target_node)}
        if name in mapped and "length" in params:
            type(self).counts["withheld"] += 1
            return True
        return False

    def _grouped(self) -> set[str]:
        """The nodes a coupling group names."""
        return {name for group in self.gm._coupling_groups for name in group.nodes}  # noqa: SLF001

    def _draw_node(self, data, *, ghost: bool = True) -> str:
        """A node of the graph, one with a stability limit three times as
        often; sometimes a name the graph does not have."""
        pool = []
        for name, spec in self.gm._nodes.items():  # noqa: SLF001
            limited = O.STABILITY_LIMITS.get(type(spec.node)) is not None
            pool += [name] * (6 if limited else 2)
        if ghost or not pool:
            pool.append("ghost")
        return data.draw(st.sampled_from(pool), label="node")

    def _keys(self, name: str) -> list[str]:
        node = self.gm._nodes[name].node  # noqa: SLF001
        live = self.gm.params.get("nodes", {}).get(name) or {}
        keys = sorted(set(node.params) | set(live))
        return keys + list(LIMIT_KEYS.get(type(node).__name__, ())) * 2

    def _draw_value(self, data, name: str, key: str) -> Any:
        """A value for ``name.key``: an ordinary one, one the graph has
        held before (the node's own, the live leaf, an earlier write, the
        one it was built with), or an awkward one."""
        node = self.gm._nodes[name].node  # noqa: SLF001
        ordinary = list(VOCABULARY.get(type(node).__name__, {}).get(key, ()))
        held = []
        if key in node.params:
            held += [_jsonable(node.params[key])] * 2
        live = self.gm.params.get("nodes", {}).get(name) or {}
        if key in live:
            held.append(np.asarray(live[key]).tolist())
        held += self.history[(name, key)][-3:]
        if (name, key) in self.built_with:
            held.append(self.built_with[(name, key)])
        kinds = ["ordinary"] * (4 if ordinary else 0) + ["held"] * (3 if held else 0) \
            + ["awkward"] * 2
        kind = data.draw(st.sampled_from(kinds), label=f"{key}: kind of value")
        pool = {"ordinary": ordinary, "held": held, "awkward": list(AWKWARD)}[kind]
        index = data.draw(st.integers(0, len(pool) - 1), label=f"{key}: which")
        return pool[index]

    # ------------------------------------------------------------------
    # start
    # ------------------------------------------------------------------

    @initialize(start=st.sampled_from(("rod",) * 10 + ("empty",) + COUPLED_AND_MAPPED),
                bind=st.sampled_from(O.BINDS))
    def start(self, start, bind):
        self.begin(start, bind)

    # ------------------------------------------------------------------
    # structure
    # ------------------------------------------------------------------

    def do_add_node(self, name: str, type_name: str, timestep: Any, params: dict):
        resp = self.send("add node", "POST", "/graph/nodes", body={
            "type": type_name, "name": name, "timestep": timestep, "params": params})
        if resp.status_code == 201:
            self._remember_construction(name, self.gm._nodes[name].node)  # noqa: SLF001
        return resp

    def do_remove_node(self, name: str):
        return self.send("remove node", "DELETE", f"/graph/nodes/{quote(name, safe='')}")

    @rule(data=st.data())
    def add_or_remove_a_node(self, data):
        present = self._nodes()
        # A member of a coupling group is not removed: the route takes the
        # request and leaves the group naming it, a graph whose save does
        # not reload (pinned below as a strict xfail; drop this filter with
        # it).
        removable = [n for n in present if n not in self._grouped()]
        if present and data.draw(st.integers(0, 2), label="remove") == 0:
            self.do_remove_node(data.draw(st.sampled_from(removable * 3 + ["ghost"]),
                                          label="name"))
            return
        free = [n for n in NAMES if n not in present]
        name = data.draw(st.sampled_from(free * 6 + list(NAMES) + list(AWKWARD_NAMES)),
                         label="name")
        usual = USUAL_TYPE.get(name, "BallNode")
        type_name = data.draw(st.sampled_from(
            [usual] * 6 + sorted(O.REGISTRY) + ["NoSuchNode"]), label="type")
        timestep = data.draw(st.sampled_from(TIMESTEPS * 6 + AWKWARD_TIMESTEPS),
                             label="timestep")
        vocabulary = VOCABULARY.get(type_name, {})
        params = {}
        for key in sorted(vocabulary):
            if key in ALWAYS_GIVEN.get(type_name, ()) \
                    or data.draw(st.booleans(), label=f"give {key}"):
                params[key] = data.draw(st.sampled_from(vocabulary[key]), label=key)
        if data.draw(st.integers(0, 7), label="awkward") == 0:
            key = data.draw(st.sampled_from(sorted(vocabulary) + ["no_such_param"]),
                            label="awkward key")
            params[key] = data.draw(st.sampled_from(AWKWARD), label="awkward value")
        self.do_add_node(name, type_name, timestep, params)

    def _edge_ends(self) -> tuple[list, list]:
        """``(sources, targets)``: every state field of every node, and
        every boundary input (a flux is left out: one in a cycle multiplies
        the state by the stiffness each step, which is the physics', not
        the API's)."""
        sources, targets = [], []
        for name, spec in self.gm._nodes.items():  # noqa: SLF001
            sources += [(name, field) for field in sorted(self.gm.get_node_state(name))]
            try:
                inputs = sorted(spec.node.boundary_input_spec() or {})
            except Exception:  # noqa: BLE001 - optional descriptor
                inputs = []
            targets += [(name, field) for field in inputs]
        return sources, targets

    def do_add_edge(self, src: str, src_field: str, tgt: str, tgt_field: str):
        return self.send("add edge", "POST", "/graph/edges", body={
            "source_node": src, "target_node": tgt,
            "source_field": src_field, "target_field": tgt_field})

    def do_remove_edge(self, src: str, src_field: str, tgt: str, tgt_field: str):
        return self.send("remove edge", "DELETE", "/graph/edges", body={
            "source_node": src, "target_node": tgt,
            "source_field": src_field, "target_field": tgt_field})

    def _draw_edge(self, data) -> tuple[str, str, str, str]:
        sources, targets = self._edge_ends()
        sources = sources * 4 + [("ghost", "position")] + (
            [(sources[0][0], "no_such_field")] if sources else [])
        targets = targets * 4 + [("ghost", "table_position")]
        src = data.draw(st.sampled_from(sources), label="source")
        tgt = data.draw(st.sampled_from(targets), label="target")
        return src[0], src[1], tgt[0], tgt[1]

    @precondition(lambda self: self.served is not None and self._nodes())
    @rule(data=st.data())
    def add_or_remove_an_edge(self, data):
        existing = sorted({(e.source_node, e.source_field, e.target_node, e.target_field)
                           for e in self.gm._edges})  # noqa: SLF001
        if existing and data.draw(st.integers(0, 1), label="remove") == 0:
            pool = existing * 3 + [self._draw_edge(data)]
            self.do_remove_edge(*data.draw(st.sampled_from(pool), label="edge"))
        else:
            self.do_add_edge(*self._draw_edge(data))

    # ------------------------------------------------------------------
    # parameters
    # ------------------------------------------------------------------

    def do_put(self, name: str, params: dict, rule_name: str = "put params"):
        resp = self.send(rule_name, "PUT", f"/graph/params/{quote(name, safe='')}",
                         body={"params": params})
        if resp.status_code == 200:
            for key, value in params.items():
                self.history[(name, key)].append(value)
        return resp

    @rule(data=st.data())
    def put_one_param(self, data):
        name = self._draw_node(data)
        if name == "ghost":
            self.do_put(name, {"stiffness": 1.0})
            return
        key = data.draw(st.sampled_from(self._keys(name) * 2 + ["no_such_param"]), label="key")
        value = 1.0 if key == "no_such_param" else self._draw_value(data, name, key)
        if not self._withheld(name, {key: value}):
            self.do_put(name, {key: value})

    @precondition(lambda self: self.served is not None and self._nodes())
    @rule(data=st.data())
    def put_several_params(self, data):
        name = self._draw_node(data, ghost=False)
        pool = self._keys(name)
        if not pool:
            return
        # A node with one parameter (a table) gets a one-key body.
        keys = data.draw(st.lists(st.sampled_from(pool), min_size=min(2, len(set(pool))),
                                  max_size=3, unique=True), label="keys")
        body = {key: self._draw_value(data, name, key) for key in keys}
        if not self._withheld(name, body):
            self.do_put(name, body, rule_name="put several params")

    @precondition(lambda self: self.served is not None and self._nodes())
    @rule(data=st.data())
    def put_params_back(self, data):
        """One or two parameters back to the node's own values -- what an
        undo does, or a form re-sent as it was loaded -- sometimes with one
        new value beside them."""
        name = self._draw_node(data, ghost=False)
        node = self.gm._nodes[name].node  # noqa: SLF001
        live = self.gm.params.get("nodes", {}).get(name) or {}
        keys = sorted(node.params)
        if not keys:
            return
        moved = [k for k in keys if k in live and not _same(live[k], node.params[k])]
        chosen = data.draw(st.lists(st.sampled_from(moved * 3 + keys), min_size=1, max_size=2,
                                    unique=True), label="keys")
        body = {key: _jsonable(node.params[key]) for key in chosen}
        others = [k for k in self._keys(name) if k not in body]
        if others and data.draw(st.integers(0, 2), label="and a new value") == 0:
            key = data.draw(st.sampled_from(others), label="new key")
            body[key] = self._draw_value(data, name, key)
        if not self._withheld(name, body):
            self.do_put(name, body, rule_name="put params back")

    def _limited(self) -> list[str]:
        return [name for name in self._nodes() if O.limit_fraction(self.gm, name) is not None]

    @precondition(lambda self: self.served is not None and self._limited())
    @rule(data=st.data())
    def put_towards_the_limit(self, data):
        """One parameter of a node with a stability limit, written to an
        ordinary value that takes the node *nearer* that limit than it is
        -- just inside it or past it; the route says which.  A generator
        that only wandered would seldom be at the edge, which is where a
        value returned to its original crosses it."""
        name = data.draw(st.sampled_from(self._limited()), label="node")
        node = self.gm._nodes[name].node  # noqa: SLF001
        now = O.limit_fraction(self.gm, name)
        current = O.read_by_the_step(self.gm, name)
        vocabulary = VOCABULARY[type(node).__name__]
        nearer = []
        for key in LIMIT_KEYS[type(node).__name__]:
            for value in vocabulary[key]:
                if _same(value, current.get(key)):
                    continue
                after = O.limit_fraction(self.gm, name, {key: value})
                if after is not None and after > now:
                    nearer.append((key, value))
        if not nearer:
            return
        key, value = data.draw(st.sampled_from(nearer), label="write")
        if not self._withheld(name, {key: value}):
            self.do_put(name, {key: value}, rule_name="put towards the limit")

    def do_fit(self, name: str, key: str, value: Any) -> bool:
        """A direct ``gm.params`` write, as a fit (or any Python code)
        makes one: the live leaf moves and the node's own value stays.  It
        is kept only when the graph's constructors still take the graph --
        a fit that leaves a graph its own save cannot reload is the
        caller's doing, not the API's -- and put back otherwise."""
        self.settled = False
        gm = self.gm
        tree = gm.params
        leaves = tree.get("nodes", {}).get(name)
        if leaves is None or key not in leaves:
            return False
        old = leaves[key]
        try:
            new = jnp.asarray(value, dtype=jnp.asarray(old).dtype)
        except (TypeError, ValueError):
            return False
        if new.shape != jnp.shape(old) or not bool(jnp.all(jnp.isfinite(new))):
            return False
        leaves[key] = new
        gm.params = tree
        try:
            with quiet():
                GraphManager.from_dict(O.saved_config(gm), self.served.registry)
            refused = bool(O.stability_violations(gm))
        except Exception:  # noqa: BLE001 - the constructors refuse the fitted graph
            refused = True
        if refused:
            tree = gm.params
            tree["nodes"][name][key] = old
            gm.params = tree
            type(self).counts[("fit", "put back")] += 1
            return False
        type(self).counts[("fit", "kept")] += 1
        event("fit: kept")
        self.unchecked = True
        self.last_kind = "fit"
        self.last = f"a gm.params write {name}.{key} = {value!r} (in process, as a fit)"
        note(self.last)
        return True

    @precondition(lambda self: self.served is not None
                  and self.gm.params.get("nodes"))
    @rule(data=st.data())
    def fit(self, data):
        live = self.gm.params["nodes"]
        names = [n for n in self._nodes() if live.get(n)]
        if not names:
            return
        pool = [n for n in names for _ in range(
            3 if O.STABILITY_LIMITS.get(type(self.gm._nodes[n].node)) else 1)]  # noqa: SLF001
        name = data.draw(st.sampled_from(pool), label="node")
        node = self.gm._nodes[name].node  # noqa: SLF001
        vocabulary = VOCABULARY.get(type(node).__name__, {})
        keys = sorted(k for k in live[name] if k in FITTED.get(type(node).__name__, ()))
        if not keys:
            return
        limit_keys = [k for k in LIMIT_KEYS.get(type(node).__name__, ()) if k in keys]
        key = data.draw(st.sampled_from(keys + limit_keys * 2), label="key")
        value = data.draw(st.sampled_from(vocabulary[key]), label="value")
        self.do_fit(name, key, value)

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def do_put_state(self, name: str, state: dict):
        return self.send("put state", "PUT", f"/graph/state/{quote(name, safe='')}",
                         body={"state": state})

    @rule(data=st.data())
    def put_state(self, data):
        name = self._draw_node(data)
        if name == "ghost":
            self.do_put_state(name, {"position": 0.0})
            return
        state = {}
        for field, leaf in sorted(self.gm.get_node_state(name).items()):
            value = data.draw(st.sampled_from(STATE_VALUES), label=field)
            state[field] = np.full(np.shape(leaf), value).tolist()
        fields = sorted(state)
        change = data.draw(st.sampled_from(
            ("none",) * 8 + ("missing", "extra") + tuple(sorted(STATE_CHANGES))), label="change")
        if change == "missing":
            del state[fields[-1]]
        elif change == "extra":
            state["no_such_field"] = 0.0
        elif change != "none":
            state[fields[0]] = STATE_CHANGES[change]
        self.do_put_state(name, state)

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------

    def _stepped(self, resp):
        """A step or a compile the graph's configuration does not allow is a
        400, and then the graph its save reloads must not step either."""
        if resp.status_code == 400:
            O.assert_the_reload_cannot_step_either(self.served, self.last)
        return resp

    def do_step(self):
        return self._stepped(self.send("step", "POST", "/sim/step"))

    @rule()
    def step(self):
        self.do_step()

    def do_run(self, n_steps: Any):
        resp = self.send("run", "POST", "/sim/run", params={"n_steps": n_steps})
        return self._stepped(resp) if isinstance(n_steps, int) and 0 <= n_steps <= 3 else resp

    @rule(n_steps=st.sampled_from((0, 1, 2, 3) * 4 + (-1, 100_001, "many")))
    def run(self, n_steps):
        self.do_run(n_steps)

    def do_reset(self):
        return self.send("reset", "POST", "/sim/reset")

    def do_compile(self):
        return self._stepped(self.send("compile", "POST", "/graph/compile"))

    @rule(reset=st.booleans())
    def reset_or_compile(self, reset):
        if reset:
            self.do_reset()
        else:
            self.do_compile()

    # ------------------------------------------------------------------
    # checkpoints
    # ------------------------------------------------------------------

    def do_save(self, slot: str):
        resp = self.send("save", "POST", "/checkpoint/save", params={"path": slot})
        if resp.status_code == 200:
            self.saved.add(slot)
        return resp

    @rule(slot=st.sampled_from(SLOTS * 4 + REFUSED_SLOTS))
    def save(self, slot):
        self.do_save(slot)

    def do_load(self, slot: str):
        return self.send("load", "POST", "/checkpoint/load", params={"path": slot})

    @rule(data=st.data())
    def load(self, data):
        slot = data.draw(st.sampled_from(sorted(self.saved) * 6 + list(SLOTS)
                                         + ["never-saved.npz"] + list(REFUSED_SLOTS)),
                         label="slot")
        self.do_load(slot)

    # ------------------------------------------------------------------
    # invariants 3, 4 and 5, after every rule
    # ------------------------------------------------------------------

    @invariant()
    def the_graph_reloads_runs_as_its_reload_and_is_within_its_limits(self):
        if self.served is None:
            return
        if self.unchecked:
            self.check_graph(self.last)
            self.unchecked = False
        self.settled = True

    def settle(self) -> None:
        """What every example ends with: one step through the API, so a
        graph left waiting for a compile is held to its reload as well."""
        resp = self.do_step()
        if resp.status_code == 200:
            self.check_graph(f"{self.last} (the example's last step)")

    def teardown(self) -> None:
        try:
            # Only after a rule and its invariants ran to their end: a
            # failure already raised must not be replaced by this one.
            if self.served is not None and self.settled:
                self.settle()
        finally:
            if self.served is not None:
                self.served.close()


# ---------------------------------------------------------------------------
# The generated sequences
# ---------------------------------------------------------------------------

#: Requests per example, per push and in the slow lane.  Absolute, as the
#: other state machines' are (``testing_standards.md``): it sets how long
#: one example is.  A request costs 10 to 80 ms here (the reload of a graph
#: whose config changed is compiled to be stepped), so ten keep an example
#: well under a second.  The slow lane's are long on purpose: a defect
#: that takes three particular requests in order is found in proportion to
#: the cube of the sequence's length, at a cost that grows with its length
#: (measured on the tree that had MADD-ANO-178, the PR that added this
#: module lists the examples it took at 50 and at 100).
STEPS_PER_PUSH = 10
STEPS_SLOW = 100


def test_short_sequences_of_rest_writes_leave_a_graph_that_runs_as_its_reload():
    """The machine at reduced depth, on every push: ten requests a sequence,
    drawn the same way on every run, the same rules and invariants -- and a
    check that the run was not vacuous: requests were both accepted and
    refused, a fit was kept, and the reload was stepped beside the served
    graph again and again.  (Which kinds of request a short derandomized
    run reaches moves with the Hypothesis version; that every kind is
    answered both ways is the fixed tour's to say, below.)

    ``max_examples`` is the house floor and is set here rather than left
    to the profile: this is the per-push sibling of the test below, sized
    to the time budget, and under the ``ci`` profile the depth comes from
    that test instead.
    """
    RestWriteSequences.counts = collections.Counter()
    run_state_machine_as_test(
        RestWriteSequences,
        settings=settings(stateful_step_count=STEPS_PER_PUSH, max_examples=EXAMPLES_FLOOR,
                          derandomize=True, database=None),
    )
    counts = RestWriteSequences.counts
    accepted = sum(n for key, n in counts.items() if key[1:] == ("accepted",))
    refused = sum(n for key, n in counts.items() if key[1:] == ("refused",))
    assert accepted >= 40 and refused >= 10, dict(counts)
    assert counts["stepped beside its reload"] >= 40, dict(counts)
    assert len({key[0] for key in counts if isinstance(key, tuple)}) >= 8, dict(counts)
    # Invariant 6 was asked: failures were injected, in most kinds of request.
    assert counts["failures injected"] >= 400, dict(counts)
    assert len({key[0] for key, n in counts.items()
                if isinstance(key, tuple) and key[1] == "failed at a point" and n}) >= 8, \
        dict(counts)
    # Both configurations were served, and in each requests were accepted,
    # refused and failed at a point; a token-holder's every request was
    # first sent without the token (invariant 7).
    for bind in O.BINDS:
        assert counts[f"accepted ({bind})"] >= 10 and counts[f"refused ({bind})"] >= 1, \
            dict(counts)
        assert counts[f"failures injected ({bind})"] >= 100, dict(counts)
    assert counts["sent without the token"] == \
        counts["accepted (token)"] + counts["refused (token)"], dict(counts)


# Per push: tests/property/test_rest_write_sequences_leave_a_graph_that_reloads.py::test_short_sequences_of_rest_writes_leave_a_graph_that_runs_as_its_reload
@pytest.mark.slow  # a state machine over the REST server: a hundred requests an example, about 3 s each
def test_any_sequence_of_rest_writes_leaves_a_graph_that_runs_as_its_reload():
    """Any sequence of the write routes, a hundred requests long: no 5xx, a
    refusal changes nothing, and what is accepted reloads, runs as its
    reload and is within its stability limits.  ``max_examples`` is the
    profile's."""
    run_state_machine_as_test(
        RestWriteSequences,
        settings=settings(stateful_step_count=STEPS_SLOW),
    )


# ---------------------------------------------------------------------------
# The oracle's own table
# ---------------------------------------------------------------------------

def test_every_stock_node_is_classified_by_its_stability_limit():
    """Invariant 5 reads ``rest_oracle.STABILITY_LIMITS``.  A stock node
    class it does not list would never be checked, so a class
    ``maddening.nodes`` exports without an entry -- a limit, or ``None``
    for a constructor that enforces none -- fails here."""
    exported = {getattr(maddening.nodes, name) for name in maddening.nodes.__all__}
    unlisted = sorted(cls.__name__ for cls in exported - set(O.STABILITY_LIMITS))
    assert not unlisted, (
        f"{unlisted} are exported by maddening.nodes and not classified in "
        "tests/property/rest_oracle.py::STABILITY_LIMITS")
    assert all(cls in O.STABILITY_LIMITS for cls in O.REGISTRY.values())


# ---------------------------------------------------------------------------
# Pinned sequences
# ---------------------------------------------------------------------------
# The two sequences audits found before there was a machine, and what the
# machine has found since, replayed through the machine's own requests and
# invariants.  Each also states the answer the fixed tree gives: on the tree
# before its fix the request is accepted and an invariant fails instead.

@contextlib.contextmanager
def replay(start: str = "rod", bind: str = "loopback"):
    """A fixed sequence through the machine: ``step(machine.do_x, ...)``
    sends the request under invariants 1 and 2 and then asks 3 to 5, as the
    machine does after every rule, and the sequence ends with the last step
    every generated example ends with."""
    machine = RestWriteSequences()
    machine.begin(start, bind)
    machine.check_graph(machine.last)

    def step(do, *args):
        result = do(*args)
        machine.check_graph(machine.last)
        return result

    try:
        yield machine, step
        machine.settle()
    finally:
        machine.served.close()


def test_every_kind_of_request_the_machine_sends_is_both_accepted_and_refused():
    """The vocabulary is not vacuous.  One fixed tour through the machine's
    own requests -- every route it drives, in a form the server takes and
    in one it refuses, each under invariants 1 to 5 -- and the count says
    every kind was answered both ways.  A machine whose every write was
    refused would hold all its invariants and find nothing."""
    RestWriteSequences.counts = collections.Counter()
    with replay() as (machine, step):
        def answered(status, do, *args):
            resp = step(do, *args)
            assert resp.status_code == status, (do.__name__, args, resp.status_code, resp.text)

        answered(200, machine.do_put, "spring", {"stiffness": 20.0})
        answered(400, machine.do_put, "spring", {"stiffness": -1.0})
        answered(404, machine.do_put, "ghost", {"stiffness": 1.0})
        answered(200, machine.do_put, "rod",
                 {"thermal_diffusivity": ALPHAS[0], "length": 0.75}, "put several params")
        answered(400, machine.do_put, "rod",
                 {"thermal_diffusivity": ALPHAS[-1], "length": 0.5}, "put several params")
        answered(200, machine.do_put_state, "ball", {"position": 2.0, "velocity": 0.5})
        answered(400, machine.do_put_state, "ball", {"position": 2.0})
        # A state at the edge of float32 overflows on the next steps: the
        # replies carry non-finite values, as quoted tokens.
        answered(200, machine.do_put_state, "spring", {"position": LARGEST, "velocity": 0.0})
        answered(200, machine.do_run, 3)
        assert not np.all(np.isfinite([float(np.asarray(v)) for v in
                                       machine.gm.get_node_state("spring").values()]))
        answered(200, machine.do_put_state, "spring", {"position": 0.5, "velocity": 0.0})
        answered(201, machine.do_add_node, "extra", "HeatNode", 0.02,
                 {"n_cells": N_CELLS, "thermal_diffusivity": ALPHAS[0],
                  "initial_temperature": 1.0})
        answered(409, machine.do_add_node, "extra", "BallNode", DT, {})
        answered(422, machine.do_add_node, "other", "BallNode", 0, {})
        answered(201, machine.do_add_edge, "ball", "position", "spring", "anchor_position")
        answered(404, machine.do_add_edge, "ghost", "position", "spring", "anchor_position")
        answered(200, machine.do_step)
        answered(200, machine.do_run, 2)
        answered(422, machine.do_run, -1)
        answered(200, machine.do_save, "a.npz")
        answered(400, machine.do_save, "../escaped.npz")
        answered(200, machine.do_remove_edge, "ball", "position", "spring", "anchor_position")
        answered(404, machine.do_remove_edge, "ball", "position", "spring", "anchor_position")
        # An edge between fields of different shapes is taken, and the
        # graph then cannot step: nor can its reload (invariant 4).
        answered(201, machine.do_add_edge, "rod", "temperature", "ball", "table_position")
        answered(400, machine.do_step)
        answered(400, machine.do_compile)
        answered(200, machine.do_remove_edge, "rod", "temperature", "ball", "table_position")
        answered(200, machine.do_compile)
        answered(200, machine.do_reset)
        answered(200, machine.do_load, "a.npz")
        answered(404, machine.do_load, "never-saved.npz")
        answered(200, machine.do_remove_node, "extra")
        answered(404, machine.do_remove_node, "extra")
        assert step(machine.do_fit, "spring", "stiffness", 40.0) is True
        # A fit that would leave a graph its constructor refuses is put back.
        assert step(machine.do_fit, "rod", "thermal_diffusivity", ALPHAS[-1]) is False
    counts = RestWriteSequences.counts
    for kind in ("put params", "put several params", "put state", "add node", "remove node",
                 "add edge", "remove edge", "step", "run", "compile", "save", "load"):
        for outcome in ("accepted", "refused"):
            assert counts[(kind, outcome)] > 0, (kind, outcome, dict(counts))
        # ... and with a failure injected into its body (invariant 6).
        assert counts[(kind, "failed at a point")] > 0, (kind, dict(counts))
    assert counts[("reset", "failed at a point")] and counts["failures injected"] >= 150, \
        dict(counts)
    assert counts[("reset", "accepted")] and counts[("fit", "kept")] \
        and counts[("fit", "put back")], dict(counts)
    assert counts["stepped beside its reload"] >= 15 and counts[
        "a graph that cannot compile"] >= 2, dict(counts)


# Per push: tests/property/test_rest_write_sequences_leave_a_graph_that_reloads.py::test_short_sequences_of_rest_writes_leave_a_graph_that_runs_as_its_reload
@pytest.mark.slow  # a second tour of every write route, each request sent at every injection point: 3 s
def test_a_failure_at_any_point_of_any_write_route_changes_nothing_for_a_token_holder():
    """Invariant 6 on a server that demands the token.  One accepted and
    one refused request of every kind the machine sends, from a
    token-holder, each first sent with a failure injected at every point
    its route body reaches: the 500 carries the generic detail and no word
    of the failure, and the graph is exactly as it was, object for object
    -- and each was first sent without the token (invariant 7).  The
    generated sequences reach this configuration in half their examples on
    every push (their test counts the failures injected there); which kinds
    of request they send there moves with the draw, and this tour does
    not."""
    RestWriteSequences.counts = collections.Counter()
    with replay(bind="token") as (machine, step):
        assert machine.served.token_enforced
        for status, do, *args in (
                (200, machine.do_put, "spring", {"stiffness": 20.0}),
                (400, machine.do_put, "rod", {"thermal_diffusivity": ALPHAS[-1], "length": 0.5}),
                (200, machine.do_put_state, "ball", {"position": 2.0, "velocity": 0.5}),
                (400, machine.do_put_state, "ball", {"position": 2.0}),
                (201, machine.do_add_node, "extra", "HeatNode", 0.02,
                 {"n_cells": N_CELLS, "thermal_diffusivity": ALPHAS[0],
                  "initial_temperature": 1.0}),
                (409, machine.do_add_node, "extra", "BallNode", DT, {}),
                (201, machine.do_add_edge, "ball", "position", "spring", "anchor_position"),
                (200, machine.do_step,),
                (200, machine.do_run, 2),
                (200, machine.do_save, "a.npz"),
                (200, machine.do_remove_edge, "ball", "position", "spring", "anchor_position"),
                (404, machine.do_remove_edge, "ball", "position", "spring", "anchor_position"),
                (200, machine.do_compile,),
                (200, machine.do_reset,),
                (200, machine.do_load, "a.npz"),
                (404, machine.do_load, "never-saved.npz"),
                (200, machine.do_remove_node, "extra"),
                (404, machine.do_remove_node, "extra")):
            resp = step(do, *args)
            assert resp.status_code == status, (do.__name__, args, resp.status_code, resp.text)
    counts = RestWriteSequences.counts
    for kind in ("put params", "put state", "add node", "remove node", "add edge",
                 "remove edge", "step", "run", "compile", "reset", "save", "load"):
        assert counts[(kind, "failed at a point")] > 0, (kind, dict(counts))
    assert counts["failures injected (token)"] == counts["failures injected"] >= 80, dict(counts)
    # The 18 requests above, the save every example starts with and the
    # step it ends with.
    assert counts["sent without the token"] == 20, dict(counts)


# ---------------------------------------------------------------------------
# Coupled, mapped and multi-rate graphs
# ---------------------------------------------------------------------------

#: A tour of the write routes over a graph of two rods (``rod`` and
#: ``extra``) or of a spring, a ball and a rod: requests each start takes
#: or refuses in its own way, and none is predicted.
_TOUR = (
    ("do_put", "rod", {"thermal_diffusivity": ALPHAS[0]}),
    ("do_put", "rod", {"thermal_diffusivity": -1.0}),
    ("do_put_state", "rod", {"temperature": [2.0] * N_CELLS}),
    ("do_step",),
    ("do_run", 3),
    ("do_save", "a.npz"),
    ("do_put", "rod", {"length": 0.75}),
    ("do_load", "a.npz"),
    # ... and, in the slow lane, the requests after which a graph is
    # compiled again (most of the cost of a coupled one):
    ("do_reset",),
    ("do_add_node", "spare", "SpringDamperNode", 2 * DT, {"stiffness": 20.0}),
    ("do_step",),
    ("do_remove_edge", "rod", "temperature", "extra", "heat_source"),
    ("do_step",),
    ("do_add_edge", "rod", "temperature", "extra", "heat_source"),
    ("do_remove_node", "spare"),
    ("do_compile",),
    ("do_load", START_SLOT),
)


#: The tour's requests per push: the writes that leave the compiled step,
#: over one start of each kind (the other solver and the other mapping
#: kind are toured whole in the slow lane, and started from by the machine).
_TOUR_PER_PUSH = 8
_TOURED_PER_PUSH = ("coupled (ift)", "mapped (sparse)", "multi-rate")


def _tour(start: str, requests: tuple) -> collections.Counter:
    RestWriteSequences.counts = collections.Counter()
    bind = O.BINDS[COUPLED_AND_MAPPED.index(start) % 2]
    with replay(start, bind) as (machine, step):
        gm = machine.gm
        assert bool(gm._coupling_groups) == start.startswith("coupled")  # noqa: SLF001
        assert any(e.mapping is not None for e in gm._edges) == start.startswith("mapped")  # noqa: SLF001
        assert (len({spec.node.delta_t for spec in gm._nodes.values()}) == 3) \
            == (start == "multi-rate")  # noqa: SLF001
        for name, *args in requests:
            step(getattr(machine, name), *args)
    counts = RestWriteSequences.counts
    counts["accepted"] = sum(n for key, n in counts.items() if key[1:] == ("accepted",))
    counts["refused"] = sum(n for key, n in counts.items() if key[1:] == ("refused",))
    return counts


@pytest.mark.parametrize("start", _TOURED_PER_PUSH)
def test_a_coupled_mapped_or_multi_rate_graph_runs_as_its_reload_after_a_write(start):
    """Invariants 1 to 7 over the graphs the routes cannot build and users
    serve: a coupling group under each solver, a mapped edge of each static
    kind, nodes at three rates.  A fixed tour over each -- parameters
    written and refused, a state written, a step, a run, a save and its
    load -- every request with a failure injected at each point, and after
    each accepted one the graph stepped beside the graph its save reloads,
    bit for bit.  The generated sequences start from these graphs too;
    which a short run reaches moves with the draw.  The starts alternate
    between the two configurations."""
    counts = _tour(start, _TOUR[:_TOUR_PER_PUSH])
    assert counts["accepted"] >= 6 and counts["refused"] >= 1, dict(counts)
    assert counts["stepped beside its reload"] >= 6, dict(counts)
    assert counts["failures injected"] >= 30, dict(counts)


# Per push: tests/property/test_rest_write_sequences_leave_a_graph_that_reloads.py::test_a_coupled_mapped_or_multi_rate_graph_runs_as_its_reload_after_a_write
@pytest.mark.slow  # a coupled graph is compiled again after every structural write: 3 to 12 s a start
@pytest.mark.parametrize("start", COUPLED_AND_MAPPED)
def test_a_coupled_mapped_or_multi_rate_graph_runs_as_its_reload_after_every_kind_of_write(start):
    """The whole tour: the requests above and then a reset, a node added
    and removed, an edge removed and added back (without its mapping, where
    it had one), a compile and a load of the save every example starts
    with."""
    counts = _tour(start, _TOUR)
    assert counts["accepted"] >= 12 and counts["refused"] >= 1, dict(counts)
    assert counts["stepped beside its reload"] >= 10, dict(counts)
    assert counts["failures injected"] >= 60, dict(counts)


@pytest.mark.xfail(strict=True, reason=(
    "DELETE /graph/nodes/{name} takes the removal of a coupling group's member and leaves "
    "the group naming it: the graph's save does not reload and every step is a 400 until a "
    "node of that name is added again"))
def test_a_graph_whose_coupling_group_lost_a_member_still_reloads():
    """Found by the tour above, under either solver.  ``DELETE /graph/nodes/extra`` on two rods
    in a coupling group answers 200; the group still names ``extra``, so
    ``GraphManager.from_dict`` refuses the graph's own save ("cannot be
    rebuilt: No node named 'extra'") and ``POST /sim/step`` and
    ``/graph/compile`` answer 400 ("coupling group references non-existent
    node") -- and no route removes a group.  Either answer would hold the
    invariant: refuse the removal, or drop the group with the node."""
    with replay("coupled (ift)") as (machine, step):
        resp = step(machine.do_remove_node, "extra")
        assert resp.status_code in (200, 400, 409), resp.text


@pytest.mark.xfail(strict=True, reason=(
    "PUT /graph/params takes a rod's new length, which moves the points a mapped edge was "
    "built from, when a load (or a fit) has left the rod's diffusivity away from the node's "
    "own value and the new length is unstable at the node's own: the graph's save does not "
    "reload"))
def test_a_length_written_after_a_load_is_held_to_the_points_a_mapping_was_built_from():
    """Found by the machine on the mapped graphs (of either kind: the dense
    one answers the same), over REST alone.  The
    rod's diffusivity is lowered, saved, raised again and the save loaded:
    the live leaf is the low value and the node's own the high one, as
    after a fit.  A length of 0.5 is then stable for the rod that runs
    (Fourier 0.2) and unstable at the node's own diffusivity (0.6).  The
    route refuses a length of 0.75 here, as it does without the load ("the
    interface mapping on edge ... was built from" the rod's grid); the
    length of 0.5 it answers 200, and the mapped edge keeps the operator
    built for the old grid while ``GraphManager.from_dict`` refuses the
    graph's own save (``PointReferenceError``)."""
    with replay("mapped (sparse)") as (machine, step):
        low, high = O.ROD_ALPHAS
        assert step(machine.do_put, "rod", {"thermal_diffusivity": low}).status_code == 200
        assert step(machine.do_save, "a.npz").status_code == 200
        assert step(machine.do_put, "rod", {"thermal_diffusivity": high}).status_code == 200
        assert step(machine.do_load, "a.npz").status_code == 200
        assert step(machine.do_put, "rod", {"length": 0.75}).status_code == 400
        resp = step(machine.do_put, "rod", {"length": 0.5})
        assert resp.status_code == 400, resp.text


def test_a_value_written_back_after_a_fit_is_held_to_the_stability_limit():
    """MADD-ANO-178 at its shortest.  A fit moves the rod's diffusivity down
    (the node's own value stays at Fourier 0.45); a length of 0.75 is then
    taken, at Fourier 0.09; the diffusivity written back to the node's own
    value is the rod at Fourier 0.8.  The route used to compare the written
    value with the node's own, find no change, ask its combined checks with
    the old live value and answer 200."""
    with replay() as (machine, step):
        assert step(machine.do_fit, "rod", "thermal_diffusivity", ALPHAS[0])
        assert step(machine.do_put, "rod", {"length": 0.75}).status_code == 200
        resp = step(machine.do_put, "rod", {"thermal_diffusivity": ALPHAS[-1]})
        assert resp.status_code == 400, resp.text


def test_a_value_written_back_after_a_load_is_held_to_the_stability_limit():
    """MADD-ANO-178 as the audit reached it, over REST alone: PUT, save,
    PUT, load, then PUT a value back to its original.  The load moves the
    live leaves and leaves the node's own values, as a fit does."""
    with replay() as (machine, step):
        assert step(machine.do_put, "rod", {"thermal_diffusivity": ALPHAS[0]}).status_code == 200
        assert step(machine.do_put, "rod", {"length": 0.75}).status_code == 200
        assert step(machine.do_save, "a.npz").status_code == 200
        assert step(machine.do_put, "rod", {"length": 1.0}).status_code == 200
        assert step(machine.do_put, "rod", {"thermal_diffusivity": ALPHAS[-1]}).status_code == 200
        assert step(machine.do_load, "a.npz").status_code == 200
        resp = step(machine.do_put, "rod", {"thermal_diffusivity": ALPHAS[-1]})
        assert resp.status_code == 400, resp.text


def test_a_checkpoint_loaded_into_a_rebuilt_node_is_held_to_the_stability_limit():
    """MADD-ANO-163.  The start graph is saved with its rod at Fourier 0.45;
    the rod is removed and built again under its name at twice the timestep
    with a low diffusivity; the save still fits the graph's names and
    shapes, and carries a diffusivity that is Fourier 0.9 for the new rod.
    ``POST /checkpoint/load`` used to check names, shapes and dtypes only,
    and answer 200."""
    with replay() as (machine, step):
        assert step(machine.do_save, "a.npz").status_code == 200
        assert step(machine.do_remove_node, "rod").status_code == 200
        assert step(machine.do_add_node, "rod", "HeatNode", 0.02, {
            "n_cells": N_CELLS, "thermal_diffusivity": ALPHAS[0],
            "initial_temperature": 1.0}).status_code == 201
        resp = step(machine.do_load, "a.npz")
        assert resp.status_code == 400, resp.text

