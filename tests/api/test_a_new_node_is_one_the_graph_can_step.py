"""``POST /graph/nodes`` accepts a node only if the graph can step it with
its state's layout unchanged.

The route's dry run traced one ``update`` without the params pytree and
discarded what it returned.  So a list where the node's state is a scalar
(``BallNode`` ``initial_velocity: [0, 0]``) was a 201 -- the state changed
shape at the first step, and after a reset the graph refused the
checkpoint it had just saved -- and a constant the step reads from the
pytree (``HeartPumpNode`` ``venous_pressure: null``) was a 201 after which
every step was a 400 until the node was deleted.  ``PUT /graph/params``
refused both values.  The dry run now traces the update the way the graph
calls it and compares what it returns with ``initial_state()``.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.nodes import (
    BallNode,
    HeartPumpNode,
    RigidBody2DNode,
    RigidBodyNode,
    SpringDamperNode,
    TableNode,
)

DT = 0.01


class NeedsItsInput(SimulationNode):
    """Cannot step until ``drive`` is connected, and says so (KeyError)."""

    def initial_state(self):
        return {"x": jnp.zeros((), jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] + boundary_inputs["drive"]}

    def boundary_input_spec(self):
        return {"drive": BoundaryInputSpec(shape=(), description="what it adds")}


class ReadsThePytreeOnly(SimulationNode):
    """Takes its constant from the injected params and nowhere else: the
    graph always passes them, so the graph steps it."""

    def __init__(self, name, timestep, gain=2.0):
        super().__init__(name, timestep, gain=gain)

    def initial_state(self):
        return {"x": jnp.ones((), jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params):
        return {"x": state["x"] * params["gain"]}


class CountsInIntegers(SimulationNode):
    """A float state its update returns as an integer."""

    def initial_state(self):
        return {"x": jnp.zeros((), jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": jnp.asarray(state["x"], jnp.int32) + 1}


class DropsAField(SimulationNode):
    def initial_state(self):
        return {"x": jnp.zeros((), jnp.float32), "y": jnp.zeros((), jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"]}


REGISTRY = {cls.__name__: cls for cls in (
    BallNode, HeartPumpNode, RigidBody2DNode, RigidBodyNode, SpringDamperNode, TableNode,
    NeedsItsInput, CountsInIntegers, DropsAField, ReadsThePytreeOnly)}


@pytest.fixture()
def served(tmp_path):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.compile()
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm,
                              checkpoint_root=str(tmp_path))
    with TestClient(server.create_app(), raise_server_exceptions=False) as client:
        yield client, gm


def _add(client, type_name, params, name="new"):
    return client.post("/graph/nodes", json={"type": type_name, "name": name,
                                             "timestep": DT, "params": params})


def _assert_refused_whole(client, gm, resp, *needles):
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    for needle in needles:
        assert needle in detail, detail
    assert set(gm._nodes) == {"s"}
    assert [node["name"] for node in client.get("/graph").json()["nodes"]] == ["s"]
    assert client.post("/sim/step").status_code == 200


#: A list where the class's state is a scalar, or a list of lists where it
#: is a vector: the leaf it broadcasts at the first step.
_OTHER_RANK = [
    ("BallNode", "initial_velocity", [0.0, 0.0], "position"),
    ("BallNode", "initial_position", [[1.0]], "velocity"),
    ("SpringDamperNode", "initial_position", [0.0, 0.0], "velocity"),
    ("HeartPumpNode", "venous_pressure", [0.0, 0.0], "arterial_pressure"),
    ("RigidBody2DNode", "initial_omega", [0.0, 0.0], "angle"),
    ("RigidBodyNode", "gravity", [[0.0, 0.0, -9.81]], "position"),
]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("type_name, key, value, leaf", _OTHER_RANK,
                         ids=[f"{t}.{k}" for t, k, _, _ in _OTHER_RANK])
def test_a_value_of_a_rank_that_reshapes_the_state_at_the_first_step_is_a_400(
        served, type_name, key, value, leaf):
    client, gm = served
    resp = _add(client, type_name, {key: value})
    _assert_refused_whole(client, gm, resp, "node 'new' cannot run with these params",
                          f"params.{key}", f"'{leaf}' has shape")


@pytest.mark.parametrize("value", [None, {}])
def test_a_constant_the_step_cannot_read_from_the_params_pytree_is_a_400(served, value):
    client, gm = served
    resp = _add(client, "HeartPumpNode", {"venous_pressure": value})
    _assert_refused_whole(client, gm, resp, "node 'new' cannot run with these params",
                          "params.venous_pressure")
    # What the graph answers is what the route said: nothing was left that
    # a compile or a start would accept and a step refuse.
    assert client.post("/graph/compile").status_code == 200


def test_the_refusal_names_the_parameter_it_is_told_from_and_no_other(served):
    client, gm = served
    resp = _add(client, "BallNode", {"gravity": -3.0, "initial_velocity": [0.0, 0.0],
                                     "elasticity": 0.5})
    _assert_refused_whole(client, gm, resp, "(told from params.initial_velocity)")


def test_the_value_the_add_refuses_is_the_value_a_params_write_refuses(served):
    client, gm = served
    assert _add(client, "BallNode", {"initial_velocity": 0.0}, name="b").status_code == 201
    for value in ([0.0, 0.0], None):
        assert _add(client, "BallNode", {"initial_velocity": value}).status_code == 400
        assert client.put("/graph/params/b",
                          json={"params": {"initial_velocity": value}}).status_code == 400


def test_a_list_the_state_is_built_from_is_taken_and_its_checkpoint_reloads(served):
    """A constant that sizes the state consistently is not a reshaping one:
    the table's position as two numbers is a two-element state, before and
    after every step."""
    client, gm = served
    assert _add(client, "TableNode", {"position": [0.0, 0.0]}).status_code == 201
    assert _add(client, "BallNode", {"initial_velocity": 2.0}, name="b").status_code == 201
    stepped = client.post("/sim/step")
    assert stepped.status_code == 200 and stepped.json()["new"]["position"] == [0.0, 0.0]
    for url in ("/checkpoint/save?path=a.npz", "/sim/reset", "/checkpoint/load?path=a.npz"):
        resp = client.post(url)
        assert resp.status_code == 200, (url, resp.text)


def test_a_node_that_needs_an_input_it_has_not_got_yet_is_still_added(served):
    """It cannot step until its edge is added and the step says so: adding
    the node and then its edge is the order the routes allow."""
    client, gm = served
    assert _add(client, "NeedsItsInput", {}).status_code == 201
    assert client.post("/sim/step").status_code == 400
    edge = {"source_node": "s", "target_node": "new", "source_field": "position",
            "target_field": "drive"}
    assert client.post("/graph/edges", json=edge).status_code == 201
    assert client.post("/sim/step").status_code == 200


@pytest.mark.parametrize("type_name, needle", [
    ("CountsInIntegers", "'x' has dtype float32 before the update and int32 after it"),
    ("DropsAField", "'y' is missing after the update"),
])
def test_an_update_that_returns_another_kind_of_dtype_or_other_fields_is_a_400(
        served, type_name, needle):
    client, gm = served
    _assert_refused_whole(client, gm, _add(client, type_name, {}), needle)


def test_a_node_that_reads_its_constants_from_the_params_pytree_only_is_added(served):
    """The dry run calls ``update`` as the graph does, with the pytree: it
    used to call it without, and refused a node the graph steps."""
    client, gm = served
    resp = _add(client, "ReadsThePytreeOnly", {"gain": 3.0})
    assert resp.status_code == 201, resp.text
    assert client.post("/sim/step").json()["new"]["x"] == 3.0
    # ... and the same class with a gain the update cannot multiply by.
    resp = _add(client, "ReadsThePytreeOnly", {"gain": [1.0, 2.0]}, name="other")
    assert resp.status_code == 400 and "params.gain" in resp.json()["detail"]
