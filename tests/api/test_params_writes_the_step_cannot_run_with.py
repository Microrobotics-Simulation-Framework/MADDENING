"""``PUT /graph/params`` refuses what the running node cannot step with, and
keeps a structural value in its parameter's type; ``POST /sim/step`` says
why a graph cannot step instead of answering 500.

* ``RigidBodyNode`` ``constraints`` of ``null``, ``{"w": 0}``,
  ``{"y": {"finite": true}}`` or ``{"z": "high"}`` each answered 200, and
  every ``POST /sim/step`` after it was a 500: the route's other checks ask
  whether a value is *used*, and counted a trace that raised as "cannot
  tell".  ``POST /graph/nodes`` refused three of the four by dry-running
  the node; the route now traces the node's hooks with the request's values.
* A JSON number does not say whether it is an integer: ``stencil_order:
  4.0`` was stored as a float, which ``params_pytree()`` exposed as a new
  trainable leaf of ``gm.params``.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api.server import (
    MAX_NODE_PARAM_INT,
    SimulationServer,
    _coerced_to_param_type,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes import HeatNode, RigidBodyNode


class LegacyCounter(SimulationNode):
    """A node on the legacy contract (``update`` takes no ``params``), so
    every key is structural: an integer and a float the step reads."""

    def __init__(self, name, timestep, count: int = 3, gain: float = 1.0):
        super().__init__(name, timestep, count=count, gain=gain)

    def initial_state(self):
        return {"x": jnp.zeros((), jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] + self.params["gain"] * self.params["count"] * dt}


REGISTRY = {"RigidBodyNode": RigidBodyNode, "HeatNode": HeatNode,
            "LegacyCounter": LegacyCounter}


def _graph(node, *, compile=True) -> GraphManager:
    gm = GraphManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_node(node)
        if compile:
            gm.compile()
            gm.step()
    return gm


def _client(gm) -> TestClient:
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm)
    return TestClient(server.create_app(), raise_server_exceptions=False)


def _body() -> GraphManager:
    return _graph(RigidBodyNode("body", 0.01, initial_velocity=(0.1, 0.0, 0.2)))


# ---------------------------------------------------------------------------
# A value the step cannot run with
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, {"w": 0.0}, {"y": {"finite": True}},
                                   {"z": "high"}],
                         ids=["null", "unknown-axis", "dict-value", "string-value"])
def test_a_constraint_the_step_cannot_run_with_is_refused_and_steps_go_on(value):
    gm = _body()
    client = _client(gm)
    resp = client.put("/graph/params/body", json={"params": {"constraints": value}})
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("constraints: ") and "step cannot run with it" in detail
    assert gm.get_node("body").params["constraints"] == {}
    assert gm._dirty is False
    assert [client.post("/sim/step").status_code for _ in range(2)] == [200, 200]


def test_a_constraint_the_step_runs_with_is_taken():
    """The control: a valid axis is written, recompiled and stepped with."""
    gm = _body()
    client = _client(gm)
    resp = client.put("/graph/params/body", json={"params": {"constraints": {"z": 0.5}}})
    assert resp.status_code == 200, resp.text
    step = client.post("/sim/step")
    assert step.status_code == 200, step.text
    assert step.json()["body"]["position"][2] == pytest.approx(0.5)


def test_one_bad_key_beside_a_good_one_writes_neither():
    gm = _body()
    resp = _client(gm).put("/graph/params/body", json={"params": {
        "constraints": {"w": 0.0}, "mass": 2.0}})
    assert resp.status_code == 400, resp.text
    node = gm.get_node("body")
    assert node.params["constraints"] == {} and node.params["mass"] == 1.0
    assert float(gm.params["nodes"]["body"]["mass"]) == 1.0


# ---------------------------------------------------------------------------
# A graph that cannot step: a 400 that says why, not a 500
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["/sim/step", "/sim/run?n_steps=3"])
@pytest.mark.parametrize("value, error", [({"w": 0.0}, "KeyError"),
                                          (None, "AttributeError"),
                                          ({"z": "high"}, "TypeError")])
def test_a_graph_configured_so_it_cannot_step_is_a_400_naming_why(route, value, error):
    """A value set in-process (no route asked) still cannot make the step
    routes answer 500: the step raises while it is traced, nothing is
    stepped, and the 400 names the error."""
    gm = _body()
    before = np.asarray(gm.get_node_state("body")["position"]).copy()
    gm.get_node("body").params["constraints"] = value
    gm._dirty = True
    resp = _client(gm).post(route)
    assert resp.status_code == 400, resp.text
    assert error in resp.json()["detail"] and "nothing was stepped" in resp.json()["detail"]
    np.testing.assert_array_equal(np.asarray(gm.get_node_state("body")["position"]), before)


def test_a_runtime_error_is_still_answered_with_its_own_message(monkeypatch):
    """``RuntimeError`` -- what ``compile()`` raises for a structure it
    refuses -- keeps the 400 and the bare message it always had."""
    gm = _body()

    def refuses(*args, **kwargs):
        raise RuntimeError("the graph has a cycle with no delay")

    monkeypatch.setattr(gm, "step", refuses)
    resp = _client(gm).post("/sim/step")
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "the graph has a cycle with no delay"


# ---------------------------------------------------------------------------
# A structural value keeps its parameter's type
# ---------------------------------------------------------------------------

def _rod() -> GraphManager:
    return _graph(HeatNode("rod", 0.05, n_cells=8, thermal_diffusivity=1e-3))


def test_an_integral_float_for_an_integer_parameter_is_stored_as_the_integer():
    gm = _rod()
    resp = _client(gm).put("/graph/params/rod", json={"params": {"stencil_order": 4.0}})
    assert resp.status_code == 200, resp.text
    stored = gm.get_node("rod").params["stencil_order"]
    assert stored == 4 and type(stored) is int
    assert resp.json()["params"]["stencil_order"] == 4
    gm.compile()
    assert "stencil_order" not in gm.params["nodes"]["rod"], "it became a fit leaf"


def test_the_same_integer_written_as_a_float_writes_nothing():
    gm = _rod()
    resp = _client(gm).put("/graph/params/rod", json={"params": {"stencil_order": 2.0}})
    assert resp.status_code == 200, resp.text
    assert type(gm.get_node("rod").params["stencil_order"]) is int
    gm.compile()
    assert "stencil_order" not in gm.params["nodes"]["rod"]


@pytest.mark.parametrize("value", [4.5, 2.000001])
def test_a_fractional_float_for_an_integer_parameter_is_refused(value):
    gm = _rod()
    resp = _client(gm).put("/graph/params/rod", json={"params": {"stencil_order": value}})
    assert resp.status_code == 400, resp.text
    assert "expected an integer" in resp.json()["detail"]
    assert gm.get_node("rod").params["stencil_order"] == 2


def test_an_integral_float_past_the_integer_bound_is_bounded_like_an_integer():
    """The request model bounds a JSON integer; the same number written as
    a float is an integer once coerced, and is bounded the same way."""
    gm = _rod()
    resp = _client(gm).put("/graph/params/rod",
                           json={"params": {"n_cells": float(3 * MAX_NODE_PARAM_INT)}})
    assert resp.status_code == 422, resp.text
    assert "integer magnitude must be at most" in resp.text
    assert gm.get_node("rod").params["n_cells"] == 8


def test_on_the_legacy_contract_an_integer_for_a_float_is_stored_as_a_float():
    gm = _graph(LegacyCounter("c", 0.1))
    client = _client(gm)
    resp = client.put("/graph/params/c", json={"params": {"gain": 2, "count": 5.0}})
    assert resp.status_code == 200, resp.text
    params = gm.get_node("c").params
    assert (params["gain"], type(params["gain"])) == (2.0, float)
    assert (params["count"], type(params["count"])) == (5, int)
    assert client.post("/sim/step").json()["c"]["x"] == pytest.approx(0.0 + 2.0 * 5 * 0.1
                                                                      + 1.0 * 3 * 0.1)


@pytest.mark.parametrize("old, new, want", [
    (2, 4.0, 4), (2, 4, 4), (2.5, 3, 3.0), (2.5, 3.5, 3.5),
    ([8, 8], [16.0, 8], [16, 8]), ((8, 8), [16, 8.0], [16, 8]),
    ([0.5, 1.0], [1, 2], [1.0, 2.0]),
    ({}, {"z": 1}, {"z": 1}), (None, 3.0, 3.0), ("a", 1.0, 1.0), (True, 1, 1),
    (np.int64(3), 7.0, 7),
])
def test_the_coercion_rule(old, new, want):
    got, problem = _coerced_to_param_type(old, new)
    assert problem is None and got == want and type(got) is type(want)
    if isinstance(want, list):
        assert [type(x) for x in got] == [type(x) for x in want]


@pytest.mark.parametrize("old, new", [(2, 2.5), (2, float("inf")), ([8, 8], [8.5, 8])])
def test_the_coercion_rule_refuses(old, new):
    got, problem = _coerced_to_param_type(old, new)
    assert got is None and problem.startswith("expected") and "integer" in problem
