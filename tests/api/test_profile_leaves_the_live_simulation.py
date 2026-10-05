"""``POST /sim/profile`` leaves the live simulation where it was.

``profile_graph`` resets the graph to its initial state and steps it,
and the route handed it the live graph: after 640 steps (t = 10 s) a
profile left the state at 0.09375, and the streams -- the relay observed
the profiler's steps -- went on from t = 10 s, so every frame carried a
time its state was not at.  The route now restores the state and params
after the profile and keeps the relay out of it.

Checked on a single node and on a coupling group (whose profile also
measures the one-iteration variant, recompiling the step): the next step
after the profile is bit-identical to the next step of a twin that was
never profiled.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode
from maddening.nodes.spring import SpringDamperNode

DT = 1.0 / 64.0


def _counter() -> GraphManager:
    gm = GraphManager()
    gm.add_node(BallNode("c", timestep=DT, initial_position=0.0, initial_velocity=1.0,
                         gravity=0.0))
    return gm


def _coupled() -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, initial_position=0.3, damping=5.0))
    gm.add_node(SpringDamperNode("b", 0.01, initial_position=0.0, damping=5.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], predictor="linear")
    return gm


def _served(gm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({}, graph_manager=gm)
    return server, TestClient(server.create_app(), raise_server_exceptions=False)


def _arrays(state: dict) -> dict:
    return {n: {f: np.asarray(v) for f, v in d.items()} for n, d in state.items()}


def _assert_same(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for node in a:
        assert a[node].keys() == b[node].keys(), node
        for field in a[node]:
            np.testing.assert_array_equal(a[node][field], b[node][field], err_msg=f"{node}.{field}")


def test_a_profile_leaves_the_state_and_the_streams_clock_as_they_were():
    gm = _counter()
    server, client = _served(gm)
    assert client.post("/sim/run", params={"n_steps": 640}).status_code == 200
    before = client.get("/graph/state").json()
    seq, clock, _ = server.relay.latest_frame()
    resp = client.post("/sim/profile", params={"n_steps": 5, "n_warmup": 1})
    assert resp.status_code == 200, resp.text
    assert client.get("/graph/state").json() == before
    assert before["c"]["position"] == 10.0
    assert server.relay.latest_frame()[:2] == (seq, clock) == (seq, 10.0)
    assert server.relay.step_count == 640
    with client.websocket_connect("/ws/state") as ws:
        frame = ws.receive_json()
    assert frame["sim_time"] == frame["state"]["c"]["position"] == 10.0


def test_the_params_a_profile_runs_with_are_put_back():
    gm = _counter()
    server, client = _served(gm)
    assert client.put("/graph/params/c", json={"params": {"elasticity": 0.5}}).status_code == 200
    leaves = {k: np.asarray(v) for k, v in gm.params["nodes"]["c"].items()}
    assert client.post("/sim/profile", params={"n_steps": 3, "n_warmup": 1}).status_code == 200
    assert {k: np.asarray(v) for k, v in gm.params["nodes"]["c"].items()}.keys() == leaves.keys()
    for k, v in leaves.items():
        np.testing.assert_array_equal(np.asarray(gm.params["nodes"]["c"][k]), v)


def _profiled_then_stepped(make, steps_before: int) -> tuple[dict, dict]:
    """``(the profiled graph's state after one more step, the twin's)``."""
    gm = make()
    server, client = _served(gm)
    twin = make()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        twin.compile()
    for g in (gm, twin):
        g.run(steps_before)
    assert client.post("/sim/profile", params={"n_steps": 5, "n_warmup": 1}).status_code == 200
    _assert_same(_arrays(gm._state), _arrays(twin._state))
    gm.step()
    twin.step()
    return _arrays(gm._state), _arrays(twin._state)


def test_the_next_step_after_a_profile_is_the_unprofiled_graphs_next_step():
    _assert_same(*_profiled_then_stepped(_counter, 40))


def test_a_coupling_groups_state_and_warm_start_survive_a_profile():
    """The coupling profile recompiles a one-iteration variant of the step;
    ``_meta`` (the predictor's history) is state and must come back too."""
    profiled, twin = _profiled_then_stepped(_coupled, 25)
    assert any(k.startswith("coupling_") for k in profiled["_meta"])
    _assert_same(profiled, twin)
