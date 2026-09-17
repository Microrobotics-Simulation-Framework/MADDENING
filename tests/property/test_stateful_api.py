"""An arbitrary sequence of REST calls keeps the server and a model in step.

Every existing test of :mod:`maddening.api.server` drives a sequence someone
wrote down by hand.  This module drives sequences nobody wrote down: a
:class:`~hypothesis.stateful.RuleBasedStateMachine` builds, mutates,
compiles, steps, checkpoints and resets a graph over HTTP in whatever order
Hypothesis picks, interleaving a fair share of calls that must be refused,
and after *every* call it checks the four things the surface promises:

* the server never answers 5xx -- an unhandled exception is a defect even
  when the request was nonsense;
* a valid call succeeds and leaves the server holding exactly what an
  in-process :class:`~maddening.core.graph_manager.GraphManager` driven with
  the same operations holds (structure, node states and parameters);
* an invalid call is a 4xx *with a message* and changes nothing -- the graph
  and the full state dict are byte-identical before and after;
* the graph the server reports is a faithful recipe: a fresh
  ``GraphManager`` built from ``GET /graph``, seeded with the server's own
  state, reproduces the server's next steps exactly.

The model is a real ``GraphManager`` rather than a hand-rolled emulator, so
what is under test is the *server layer*: its validation, its ordering
(validate-everything-then-write), its error mapping and its reporting.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)
from hypothesis import settings

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager

from tests.property.stateful_model import (
    BOUNDARY_INPUTS,
    DT,
    OUT_OF_BOUNDS,
    REGISTRY,
    STATE_FIELDS,
    VALID_PARAM_RANGE,
    canonical,
    constructor_params,
    float32_in,
    jsonify_state,
)

NAMES = ("n0", "n1", "n2")
"""The node-name pool -- three names cap the graph at three nodes.

Each extra node is another JAX trace on both the server and the model, and
three is already enough for a chain, a cycle, a self-loop and a
disconnected node.
"""

CHECKPOINT_SLOTS = ("first.npz", "nested/second.npz", "third")
"""Checkpoint names inside the configured root (the last one has no suffix,
which ``numpy.savez`` appends and ``load_state`` retries with)."""

ESCAPING_PATHS = ("../escaped.npz", "/tmp/escaped.npz", "a/../../escaped.npz")
"""Paths that resolve outside ``checkpoint_root`` and must be refused."""


class SimulationServerMachine(RuleBasedStateMachine):
    """Drive ``SimulationServer`` over HTTP against a ``GraphManager`` model."""

    def __init__(self) -> None:
        super().__init__()
        self._tmp = tempfile.TemporaryDirectory(prefix="maddening-stateful-api-")
        root = Path(self._tmp.name)
        self.server_root = root / "server"
        self.model_root = root / "model"
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.server = SimulationServer(node_registry=REGISTRY,
                                       checkpoint_root=str(self.server_root))
        # ``raise_server_exceptions=False`` is what makes "never 500"
        # checkable: an unhandled exception comes back as a response.
        self.client = TestClient(self.server.create_app(), raise_server_exceptions=False)
        self.model = GraphManager()
        self.types: dict[str, str] = {}
        self.edges: list[tuple[str, str, str, str]] = []
        self.written_params: dict[str, dict[str, float]] = {}
        self.saved: set[str] = set()

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------

    def _send(self, method: str, url: str, **kwargs):
        """One request, with the "never 5xx" promise checked on every reply."""
        resp = self.client.request(method, url, **kwargs)
        assert resp.status_code < 500, (
            f"{method} {url} -> {resp.status_code}: {resp.text[:500]}"
        )
        return resp

    def _snapshot(self) -> tuple:
        return (canonical(self._send("GET", "/graph").json()),
                canonical(self._send("GET", "/graph/state").json()))

    def _reject(self, resp, before: tuple) -> None:
        """A refusal: 4xx, a message, and nothing written."""
        assert 400 <= resp.status_code < 500, (resp.status_code, resp.text[:500])
        detail = resp.json().get("detail")
        assert detail, f"a refusal must say why: {resp.text[:500]}"
        assert self._snapshot() == before, "a refused call changed the server's state"

    def _boundary_targets(self) -> list[str]:
        return sorted(n for n, t in self.types.items() if BOUNDARY_INPUTS[t])

    # ------------------------------------------------------------------
    # structure: nodes
    # ------------------------------------------------------------------

    @rule(name=st.sampled_from(NAMES), type_name=st.sampled_from(sorted(REGISTRY)),
          data=st.data())
    def add_node(self, name, type_name, data):
        params = data.draw(constructor_params(type_name))
        before = self._snapshot()
        resp = self._send("POST", "/graph/nodes", json={
            "type": type_name, "name": name, "timestep": DT, "params": params,
        })
        if name in self.types:
            assert resp.status_code == 409, resp.text[:500]
            self._reject(resp, before)
            return
        assert resp.status_code == 201, resp.text[:500]
        self.model.add_node(REGISTRY[type_name](name=name, timestep=DT, **params))
        self.types[name] = type_name
        self.written_params.pop(name, None)

    @rule(name=st.sampled_from(NAMES))
    def remove_node(self, name):
        before = self._snapshot()
        resp = self._send("DELETE", f"/graph/nodes/{name}")
        if name not in self.types:
            assert resp.status_code == 404, resp.text[:500]
            self._reject(resp, before)
            return
        assert resp.status_code == 200, resp.text[:500]
        self.model.remove_node(name)
        del self.types[name]
        self.written_params.pop(name, None)
        self.edges = [e for e in self.edges if name not in (e[0], e[1])]

    @rule(kind=st.sampled_from(("unknown_type", "bad_name", "unknown_kwarg",
                                "non_numeric_constant", "nan_constant")),
          type_name=st.sampled_from(sorted(REGISTRY)))
    def rejected_node_creation(self, kind, type_name):
        """Every way of asking for a node the server must not build."""
        before = self._snapshot()
        body = {"type": type_name, "name": "rejected", "timestep": DT, "params": {}}
        if kind == "unknown_type":
            body["type"] = "NoSuchNode"
        elif kind == "bad_name":
            # '/', '#' and '->' delimit checkpoint keys and edge keys.
            body["name"] = "bad/name"
        elif kind == "unknown_kwarg":
            body["params"] = {"no_such_parameter": 1.0}
        elif kind == "non_numeric_constant":
            key = sorted(VALID_PARAM_RANGE[type_name])[0]
            body["params"] = {key: "not a number"}
        else:
            # NaN is not JSON the test client's encoder will produce, so the
            # body goes on the wire verbatim (the server's parser accepts it).
            key = sorted(VALID_PARAM_RANGE[type_name])[0]
            raw = json.dumps({**body, "params": {key: 0.0}}).replace(
                f'"{key}": 0.0', f'"{key}": NaN')
            resp = self._send("POST", "/graph/nodes", content=raw.encode(),
                              headers={"content-type": "application/json"})
            self._reject(resp, before)
            return
        resp = self._send("POST", "/graph/nodes", json=body)
        self._reject(resp, before)

    # ------------------------------------------------------------------
    # structure: edges
    # ------------------------------------------------------------------

    @precondition(lambda self: bool(self._boundary_targets()))
    @rule(data=st.data())
    def add_edge(self, data):
        src = data.draw(st.sampled_from(sorted(self.types)))
        tgt = data.draw(st.sampled_from(self._boundary_targets()))
        # State fields only, never a boundary *flux*: a flux is proportional
        # to the stiffness, so a two-node cycle through one multiplies the
        # state by k every step and diverges within a handful of steps.
        # That is a property of the physics, not of the REST surface; the
        # flux branch of the source-field check is pinned separately below.
        src_field = data.draw(st.sampled_from(sorted(STATE_FIELDS[self.types[src]])))
        tgt_field = data.draw(st.sampled_from(sorted(BOUNDARY_INPUTS[self.types[tgt]])))
        resp = self._send("POST", "/graph/edges", json={
            "source_node": src, "target_node": tgt,
            "source_field": src_field, "target_field": tgt_field,
        })
        assert resp.status_code == 201, resp.text[:500]
        self.model.add_edge(source=src, target=tgt,
                            source_field=src_field, target_field=tgt_field)
        self.edges.append((src, tgt, src_field, tgt_field))

    @precondition(lambda self: bool(self.edges))
    @rule(data=st.data())
    def remove_edge(self, data):
        edge = data.draw(st.sampled_from(sorted(set(self.edges))))
        src, tgt, src_field, tgt_field = edge
        resp = self._send("DELETE", "/graph/edges", json={
            "source_node": src, "target_node": tgt,
            "source_field": src_field, "target_field": tgt_field,
        })
        assert resp.status_code == 200, resp.text[:500]
        self.model.remove_edge(source=src, target=tgt,
                               source_field=src_field, target_field=tgt_field)
        # remove_edge drops every copy of a repeated edge, not just one.
        self.edges = [e for e in self.edges if e != edge]

    @precondition(lambda self: bool(self._boundary_targets()))
    @rule(kind=st.sampled_from(("unknown_source", "unknown_target", "unknown_field")),
          data=st.data())
    def rejected_edge(self, kind, data):
        tgt = data.draw(st.sampled_from(self._boundary_targets()))
        src = data.draw(st.sampled_from(sorted(self.types)))
        tgt_field = sorted(BOUNDARY_INPUTS[self.types[tgt]])[0]
        body = {"source_node": src, "target_node": tgt,
                "source_field": sorted(STATE_FIELDS[self.types[src]])[0],
                "target_field": tgt_field}
        if kind == "unknown_source":
            body["source_node"] = "ghost"
        elif kind == "unknown_target":
            body["target_node"] = "ghost"
        else:
            body["source_field"] = "no_such_field"
        before = self._snapshot()
        resp = self._send("POST", "/graph/edges", json=body)
        self._reject(resp, before)

    # ------------------------------------------------------------------
    # compile / step / reset
    # ------------------------------------------------------------------

    @rule()
    def compile_graph(self):
        resp = self._send("POST", "/graph/compile")
        if resp.status_code == 200:
            self.model.compile()
            assert resp.json()["schedule"] == self.model.schedule
            return
        assert resp.status_code == 400, resp.text[:500]
        with pytest.raises(RuntimeError):
            self.model.compile()

    @rule(n=st.integers(min_value=1, max_value=3))
    def step(self, n):
        for _ in range(n):
            resp = self._send("POST", "/sim/step")
            assert resp.status_code == 200, resp.text[:500]
            self.model.step()
            assert canonical(resp.json()) == jsonify_state(self.model._state)  # noqa: SLF001

    @rule(n=st.integers(min_value=0, max_value=4))
    def run(self, n):
        resp = self._send("POST", "/sim/run", params={"n_steps": n})
        assert resp.status_code == 200, resp.text[:500]
        self.model.run(n)
        assert canonical(resp.json()) == jsonify_state(self.model._state)  # noqa: SLF001

    @rule()
    def reset(self):
        resp = self._send("POST", "/sim/reset")
        assert resp.status_code == 200, resp.text[:500]
        self.model.reset_state()
        self.model._dirty = True                                    # noqa: SLF001

    @rule()
    def validate(self):
        resp = self._send("POST", "/graph/validate")
        assert resp.status_code == 200, resp.text[:500]
        assert resp.json()["issues"] == self.model.validate()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    @rule(name=st.sampled_from(NAMES))
    def read_node_state(self, name):
        resp = self._send("GET", f"/graph/state/{name}")
        if name not in self.types:
            assert resp.status_code == 404, resp.text[:500]
            return
        assert resp.status_code == 200, resp.text[:500]
        assert canonical(resp.json()) == jsonify_state(self.model.get_node_state(name))

    @precondition(lambda self: bool(self.types))
    @rule(data=st.data())
    def write_node_state(self, data):
        name = data.draw(st.sampled_from(sorted(self.types)))
        fields = STATE_FIELDS[self.types[name]]
        values = {f: data.draw(float32_in(-3.0, 3.0)) for f in fields}
        resp = self._send("PUT", f"/graph/state/{name}", json={"state": values})
        assert resp.status_code == 200, resp.text[:500]
        live = self.model.get_node_state(name)
        self.model.set_node_state(name, {
            f: jnp.asarray(v, dtype=jnp.asarray(live[f]).dtype) for f, v in values.items()
        })

    @precondition(lambda self: bool(self.types))
    @rule(kind=st.sampled_from(("unknown_node", "missing_field", "extra_field",
                                "wrong_shape", "non_numeric", "nan")),
          data=st.data())
    def rejected_state_write(self, kind, data):
        name = data.draw(st.sampled_from(sorted(self.types)))
        fields = STATE_FIELDS[self.types[name]]
        body = {f: 0.0 for f in fields}
        url = f"/graph/state/{name}"
        before = self._snapshot()
        if kind == "unknown_node":
            url = "/graph/state/ghost"
        elif kind == "missing_field":
            body = {f: 0.0 for f in fields[:-1]}
        elif kind == "extra_field":
            body["no_such_field"] = 0.0
        elif kind == "wrong_shape":
            body[fields[0]] = [1.0, 2.0, 3.0]
        elif kind == "non_numeric":
            body[fields[0]] = "not a number"
        else:
            raw = json.dumps({"state": body}).replace(
                f'"{fields[0]}": 0.0', f'"{fields[0]}": NaN')
            resp = self._send("PUT", url, content=raw.encode(),
                              headers={"content-type": "application/json"})
            self._reject(resp, before)
            assert "finite" in resp.json()["detail"]
            return
        resp = self._send("PUT", url, json={"state": body})
        self._reject(resp, before)

    # ------------------------------------------------------------------
    # parameters
    # ------------------------------------------------------------------

    @rule(name=st.sampled_from(NAMES))
    def read_params(self, name):
        resp = self._send("GET", f"/graph/params/{name}")
        if name not in self.types:
            assert resp.status_code == 404, resp.text[:500]
            return
        assert resp.status_code == 200, resp.text[:500]
        got = resp.json()
        for key, value in self.written_params.get(name, {}).items():
            assert got[key] == pytest.approx(value), (name, key, got)

    @precondition(lambda self: bool(self.types))
    @rule(data=st.data())
    def write_params(self, data):
        name = data.draw(st.sampled_from(sorted(self.types)))
        type_name = self.types[name]
        key = data.draw(st.sampled_from(sorted(VALID_PARAM_RANGE[type_name])))
        lo, hi = VALID_PARAM_RANGE[type_name][key]
        value = data.draw(float32_in(lo, hi))
        resp = self._send("PUT", f"/graph/params/{name}", json={"params": {key: value}})
        assert resp.status_code == 200, resp.text[:500]
        written = self._mirror_param_write(name, key, value)
        assert resp.json()["params"][key] == pytest.approx(written)
        self.written_params.setdefault(name, {})[key] = written

    def _mirror_param_write(self, name: str, key: str, value: float) -> float:
        """The same write against the model, by the server's own rules.

        A live leaf (``gm.params`` after a compile) is written in the leaf's
        dtype; before the first compile there is no live pytree and only the
        node's own ``params`` dict changes.  Either way the constructor
        entry keeps a Python float, never the raw JSON number.
        """
        node = self.model._nodes[name].node                         # noqa: SLF001
        live = self.model.params.get("nodes", {}).get(name) or {}
        probe_only = False
        if not live and getattr(node, "accepts_params", lambda: False)():
            live = dict(node.params_pytree())
            probe_only = True
        coerced = jnp.asarray(value, dtype=live[key].dtype)
        if not probe_only:
            live[key] = coerced
        node.params[key] = np.asarray(coerced).tolist()
        return float(np.asarray(coerced))

    @precondition(lambda self: bool(self.types))
    @rule(kind=st.sampled_from(("unknown_node", "unknown_key", "out_of_bounds",
                                "boolean", "wrong_shape", "nan")),
          data=st.data())
    def rejected_param_write(self, kind, data):
        name = data.draw(st.sampled_from(sorted(self.types)))
        type_name = self.types[name]
        key = sorted(VALID_PARAM_RANGE[type_name])[0]
        url = f"/graph/params/{name}"
        before = self._snapshot()
        if kind == "unknown_node":
            resp = self._send("PUT", "/graph/params/ghost",
                              json={"params": {key: 1.0}})
            assert resp.status_code == 404, resp.text[:500]
            self._reject(resp, before)
            return
        if kind == "unknown_key":
            body = {"no_such_param": 1.0}
        elif kind == "out_of_bounds":
            bad_key, bad_value = OUT_OF_BOUNDS[type_name]
            if bad_value != bad_value:          # NaN: must go as raw JSON
                raw = f'{{"params": {{"{bad_key}": NaN}}}}'
                resp = self._send("PUT", url, content=raw.encode(),
                                  headers={"content-type": "application/json"})
                self._reject(resp, before)
                return
            body = {bad_key: bad_value}
        elif kind == "boolean":
            body = {key: True}
        elif kind == "wrong_shape":
            body = {key: [1.0, 2.0]}
        else:
            raw = f'{{"params": {{"{key}": NaN}}}}'
            resp = self._send("PUT", url, content=raw.encode(),
                              headers={"content-type": "application/json"})
            self._reject(resp, before)
            assert "finite" in resp.json()["detail"]
            return
        resp = self._send("PUT", url, json={"params": body})
        self._reject(resp, before)

    # ------------------------------------------------------------------
    # checkpoints
    # ------------------------------------------------------------------

    @rule(slot=st.sampled_from(CHECKPOINT_SLOTS))
    def save_checkpoint(self, slot):
        resp = self._send("POST", "/checkpoint/save", params={"path": slot})
        assert resp.status_code == 200, resp.text[:500]
        assert Path(resp.json()["path"]).resolve().is_relative_to(
            self.server_root.resolve())
        mirror = self.model_root / slot
        mirror.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_state(str(mirror))
        self.saved.add(slot)

    @precondition(lambda self: bool(self.saved))
    @rule(data=st.data())
    def load_checkpoint(self, data):
        slot = data.draw(st.sampled_from(sorted(self.saved)))
        resp = self._send("POST", "/checkpoint/load", params={"path": slot})
        try:
            self.model.load_state(str(self.model_root / slot))
        except Exception:                                           # noqa: BLE001
            # A checkpoint written for a structure the graph no longer has
            # is a 4xx, never a 5xx and never a partial restore.
            assert 400 <= resp.status_code < 500, resp.text[:500]
            return
        assert resp.status_code == 200, resp.text[:500]
        assert canonical(resp.json()["state"]) == jsonify_state(
            self.model._state)                                      # noqa: SLF001

    @rule(op=st.sampled_from(("save", "load")), path=st.sampled_from(ESCAPING_PATHS))
    def rejected_checkpoint_path(self, op, path):
        before = self._snapshot()
        resp = self._send("POST", f"/checkpoint/{op}", params={"path": path})
        self._reject(resp, before)
        assert "checkpoint root" in resp.json()["detail"] or "must stay under" \
            in resp.json()["detail"]

    @rule()
    def load_missing_checkpoint(self):
        before = self._snapshot()
        resp = self._send("POST", "/checkpoint/load", params={"path": "never-saved.npz"})
        assert resp.status_code == 404, resp.text[:500]
        self._reject(resp, before)

    # ------------------------------------------------------------------
    # invariants
    # ------------------------------------------------------------------

    @invariant()
    def reported_state_matches_the_model(self):
        got = self._send("GET", "/graph/state")
        assert got.status_code == 200, got.text[:500]
        assert canonical(got.json()) == jsonify_state(self.model._state)  # noqa: SLF001

    @invariant()
    def reported_graph_matches_the_model(self):
        cfg = self._send("GET", "/graph").json()
        assert [n["name"] for n in cfg["nodes"]] == list(self.model._nodes)  # noqa: SLF001
        assert [n["type"] for n in cfg["nodes"]] == [
            type(s.node).__name__ for s in self.model._nodes.values()]  # noqa: SLF001
        assert [(e["source_node"], e["target_node"], e["source_field"],
                 e["target_field"]) for e in cfg["edges"]] == [
            (e.source_node, e.target_node, e.source_field, e.target_field)
            for e in self.model._edges]                             # noqa: SLF001
        assert canonical(cfg) == canonical({
            **self.model.to_dict(strict_mappings=False), "active_surrogates": []})

    # ------------------------------------------------------------------
    # teardown: the reported graph is a faithful recipe
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        try:
            self._rebuilt_graph_reproduces_the_trajectory()
        finally:
            self.client.close()
            self._tmp.cleanup()

    def _rebuilt_graph_reproduces_the_trajectory(self) -> None:
        cfg = self._send("GET", "/graph").json()
        fresh = GraphManager.from_dict(cfg, REGISTRY)
        live = self._send("GET", "/graph/state").json()
        for name in fresh.node_names:
            seed = fresh.get_node_state(name)
            fresh.set_node_state(name, {
                f: jnp.asarray(v, dtype=jnp.asarray(seed[f]).dtype)
                for f, v in live[name].items()
            })
        for _ in range(3):
            fresh.step()
            assert self._send("POST", "/sim/step").status_code == 200
        assert jsonify_state(fresh._state) == canonical(                 # noqa: SLF001
            self._send("GET", "/graph/state").json())


def test_arbitrary_rest_sequences_keep_the_server_and_the_model_in_step():
    """Any sequence of REST calls: no 5xx, no partial write, model agreement.

    ``stateful_step_count`` is 14 rather than Hypothesis's default 50: a
    rule that compiles or steps costs a JAX trace on both the server and the
    model (~60-90 ms for these graphs), so 50 steps would make one example
    several seconds and the ``ci`` profile's 200 examples unrunnable.  With
    three node names, fourteen steps is long enough to build a full graph,
    compile it, step it, checkpoint it, break it and rebuild it -- and every
    example additionally pays for the teardown's rebuild-and-replay.
    ``max_examples`` is left to the profile, per the house rule.
    """
    run_state_machine_as_test(
        SimulationServerMachine,
        settings=settings(stateful_step_count=14),
    )


# ---------------------------------------------------------------------------
# Pinned regression sequences
# ---------------------------------------------------------------------------
# Hypothesis prints a ``@reproduce_failure`` blob and the example database is
# on, but neither survives a clean checkout: what the machine found is pinned
# here as an ordinary test.

def _client(tmp_path, **kwargs):
    server = SimulationServer(node_registry=REGISTRY,
                              checkpoint_root=str(tmp_path / "ckpt"), **kwargs)
    return TestClient(server.create_app(), raise_server_exceptions=False)


def test_edge_from_a_boundary_flux_is_accepted_and_steps(tmp_path):
    """The source-field check accepts a flux, not only a state field.

    The state machine deliberately never draws a flux source (the gain makes
    the physics diverge), so the branch is pinned here instead.
    """
    client = _client(tmp_path)
    assert client.post("/graph/nodes", json={
        "type": "SpringDamperNode", "name": "s", "timestep": DT,
        "params": {"stiffness": 3.0}}).status_code == 201
    assert client.post("/graph/nodes", json={
        "type": "BallNode", "name": "b", "timestep": DT}).status_code == 201
    assert client.post("/graph/edges", json={
        "source_node": "s", "target_node": "b",
        "source_field": "spring_force", "target_field": "table_position",
    }).status_code == 201
    bad = client.post("/graph/edges", json={
        "source_node": "s", "target_node": "b",
        "source_field": "not_a_flux", "target_field": "table_position"})
    assert bad.status_code == 400 and "spring_force" in bad.json()["detail"]
    assert client.post("/sim/step").status_code == 200


def test_a_non_finite_constructor_constant_is_a_400_and_no_node(tmp_path):
    """Found by ``rejected_node_creation(kind='nan_constant')``.

    A NaN constant survived the constructor and the dry-run trace, the node
    went into the graph, and only then did the 201 body fail to serialise --
    so the caller saw a 500 and ``GET /graph`` answered 500 from then on.
    """
    client = _client(tmp_path)
    for body in (b'{"type":"BallNode","name":"n","timestep":0.01,'
                 b'"params":{"elasticity": NaN}}',
                 b'{"type":"BallNode","name":"n","timestep":0.01,'
                 b'"params":{"elasticity": Infinity}}',
                 b'{"type":"BallNode","name":"n","timestep":0.01,'
                 b'"params":{"gravity": [1.0, -Infinity]}}',
                 # 1e400 parses to inf; a bare integer literal that large is
                 # an unbounded Python int, and float() on it raises
                 # OverflowError -- which would be the 500 all over again
                 b'{"type":"BallNode","name":"n","timestep":0.01,'
                 b'"params":{"gravity": 1e400}}',
                 b'{"type":"BallNode","name":"n","timestep":0.01,'
                 b'"params":{"gravity": ' + b"9" * 400 + b'}}'):
        resp = client.post("/graph/nodes", content=body,
                           headers={"content-type": "application/json"})
        assert resp.status_code == 400, resp.text
        assert "finite" in resp.json()["detail"]
    assert client.get("/graph").json()["nodes"] == []
    assert client.get("/graph").status_code == 200


def test_a_param_write_before_the_first_compile_echoes_what_it_wrote(tmp_path):
    """Found by ``write_params`` on an uncompiled graph.

    With no live pytree yet the endpoint validates against a throwaway probe
    copy of the node's pytree; it used to echo *that* copy, so the reply
    reported the pre-write value while the very next GET reported the new
    one.
    """
    client = _client(tmp_path)
    assert client.post("/graph/nodes", json={
        "type": "BallNode", "name": "b", "timestep": DT}).status_code == 201
    echo = client.put("/graph/params/b", json={"params": {"elasticity": 0.25}})
    assert echo.status_code == 200
    assert echo.json()["params"]["elasticity"] == pytest.approx(0.25)
    assert client.get("/graph/params/b").json()["elasticity"] == pytest.approx(0.25)
    assert client.post("/graph/compile").status_code == 200
    echo = client.put("/graph/params/b", json={"params": {"elasticity": 0.75}})
    assert echo.json()["params"]["elasticity"] == pytest.approx(0.75)
    assert client.get("/graph/params/b").json()["elasticity"] == pytest.approx(0.75)


def test_params_written_before_the_first_compile_survive_a_reset(tmp_path):
    """A parameter write with no live pytree yet still reaches the node.

    Found by the machine's ``write_params`` -> ``reset`` -> ``step`` ordering:
    before the first ``compile`` the server takes the *probe* path and only
    ``node.params`` changes, and ``reset_state`` reads its seed from exactly
    there, so the reset state must show the written value.
    """
    client = _client(tmp_path)
    assert client.post("/graph/nodes", json={
        "type": "BallNode", "name": "b", "timestep": DT}).status_code == 201
    assert client.put("/graph/params/b", json={
        "params": {"initial_position": 2.5}}).status_code == 200
    reset = client.post("/sim/reset")
    assert reset.status_code == 200
    assert reset.json()["state"]["b"]["position"] == pytest.approx(2.5)
    assert client.get("/graph/params/b").json()["initial_position"] == pytest.approx(2.5)


def test_checkpoint_load_of_a_stale_structure_is_a_4xx_not_a_partial_restore(tmp_path):
    """Save, change the graph, load: a refusal that leaves the state alone."""
    client = _client(tmp_path)
    for name, type_name in (("a", "TableNode"), ("b", "BallNode")):
        assert client.post("/graph/nodes", json={
            "type": type_name, "name": name, "timestep": DT}).status_code == 201
    assert client.post("/graph/compile").status_code == 200
    assert client.post("/checkpoint/save", params={"path": "s.npz"}).status_code == 200
    assert client.delete("/graph/nodes/b").status_code == 200
    before = client.get("/graph/state").json()
    resp = client.post("/checkpoint/load", params={"path": "s.npz"})
    assert 400 <= resp.status_code < 500, resp.text
    assert client.get("/graph/state").json() == before


@pytest.mark.xfail(strict=True, reason=(
    "Pending an API decision: a diverged simulation makes every endpoint that "
    "reports state answer 500.  _jax_to_python emits Python floats and "
    "Starlette's JSONResponse serialises with allow_nan=False, so once the "
    "physics produces inf or NaN -- reachable with nothing but valid calls, "
    "here a stiffness the explicit integrator cannot hold at this timestep -- "
    "GET /graph/state, GET /graph/state/<node>, POST /sim/step and POST /sim/run "
    "all raise inside the response encoder, and only POST /sim/reset gets the "
    "server back.  Fixing it means deciding what a non-finite leaf looks like "
    "on the wire (null? a string? a 409 naming the diverged node?), which "
    "changes the public response schema and is not this branch's call."))
def test_state_endpoints_stay_below_500_when_the_simulation_diverges(tmp_path):
    client = _client(tmp_path)
    assert client.post("/graph/nodes", json={
        "type": "SpringDamperNode", "name": "s", "timestep": DT,
        "params": {"stiffness": 1.0e6, "mass": 0.5, "rest_length": 0.0,
                   "initial_position": 1.0},
    }).status_code == 201
    assert client.post("/graph/compile").status_code == 200
    assert client.post("/sim/run", params={"n_steps": 60}).status_code < 500
    assert client.get("/graph/state").status_code < 500
    assert client.get("/graph/state/s").status_code < 500
    assert client.post("/sim/step").status_code < 500
