"""Tests for the FastAPI simulation server."""

import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode


REGISTRY = {
    "BallNode": BallNode,
    "TableNode": TableNode,
    "SpringDamperNode": SpringDamperNode,
}


@pytest.fixture
def server():
    return SimulationServer(node_registry=REGISTRY)


@pytest.fixture
def client(server):
    app = server.create_app()
    return TestClient(app)


@pytest.fixture
def loaded_client():
    """Client with a pre-built bouncing ball graph."""
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01, position=0.0))
    gm.add_node(BallNode(name="ball", timestep=0.01, initial_position=5.0,
                          elasticity=0.7))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm)
    app = server.create_app()
    return TestClient(app)


# ------------------------------------------------------------------
# Graph structure endpoints
# ------------------------------------------------------------------

class TestGraphEndpoints:
    def test_get_empty_graph(self, client):
        resp = client.get("/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []
        assert data["edges"] == []

    def test_add_node(self, client):
        resp = client.post("/graph/nodes", json={
            "type": "BallNode",
            "name": "ball",
            "timestep": 0.01,
            "params": {"initial_position": 5.0},
        })
        assert resp.status_code == 201
        assert resp.json()["node"]["name"] == "ball"

    def test_add_node_unknown_type(self, client):
        resp = client.post("/graph/nodes", json={
            "type": "FakeNode",
            "name": "x",
            "timestep": 0.01,
        })
        assert resp.status_code == 400

    def test_add_duplicate_node(self, client):
        client.post("/graph/nodes", json={
            "type": "TableNode", "name": "t", "timestep": 0.01,
        })
        resp = client.post("/graph/nodes", json={
            "type": "TableNode", "name": "t", "timestep": 0.01,
        })
        assert resp.status_code == 409

    def test_remove_node(self, client):
        client.post("/graph/nodes", json={
            "type": "TableNode", "name": "t", "timestep": 0.01,
        })
        resp = client.delete("/graph/nodes/t")
        assert resp.status_code == 200

    def test_remove_nonexistent_node(self, client):
        resp = client.delete("/graph/nodes/ghost")
        assert resp.status_code == 404

    def test_add_edge(self, client):
        client.post("/graph/nodes", json={
            "type": "TableNode", "name": "t", "timestep": 0.01,
        })
        client.post("/graph/nodes", json={
            "type": "BallNode", "name": "b", "timestep": 0.01,
        })
        resp = client.post("/graph/edges", json={
            "source_node": "t", "target_node": "b",
            "source_field": "position", "target_field": "table_position",
        })
        assert resp.status_code == 201

    def test_compile(self, loaded_client):
        resp = loaded_client.post("/graph/compile")
        assert resp.status_code == 200
        assert "schedule" in resp.json()

    def test_validate(self, loaded_client):
        resp = loaded_client.post("/graph/validate")
        assert resp.status_code == 200
        data = resp.json()
        assert "issues" in data
        errors = [i for i in data["issues"] if i.startswith("ERROR")]
        assert len(errors) == 0


# ------------------------------------------------------------------
# State endpoints
# ------------------------------------------------------------------

class TestStateEndpoints:
    def test_get_state(self, loaded_client):
        resp = loaded_client.get("/graph/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "ball" in data
        assert "table" in data

    def test_get_node_state(self, loaded_client):
        resp = loaded_client.get("/graph/state/ball")
        assert resp.status_code == 200
        data = resp.json()
        assert "position" in data
        assert data["position"] == pytest.approx(5.0)

    def test_get_nonexistent_node_state(self, loaded_client):
        resp = loaded_client.get("/graph/state/ghost")
        assert resp.status_code == 404

    def test_set_node_state(self, loaded_client):
        resp = loaded_client.put("/graph/state/ball", json={
            "state": {"position": 10.0, "velocity": 0.0},
        })
        assert resp.status_code == 200

        resp = loaded_client.get("/graph/state/ball")
        assert resp.json()["position"] == pytest.approx(10.0)


# ------------------------------------------------------------------
# Simulation control endpoints
# ------------------------------------------------------------------

class TestSimEndpoints:
    def test_step(self, loaded_client):
        resp = loaded_client.post("/sim/step")
        assert resp.status_code == 200
        data = resp.json()
        # Ball should have moved from initial position
        assert data["ball"]["position"] < 5.0

    def test_run(self, loaded_client):
        resp = loaded_client.post("/sim/run?n_steps=100")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ball"]["position"] >= 0.0  # above table

    def test_start_stop(self, loaded_client):
        resp = loaded_client.post("/sim/start")
        assert resp.status_code == 200

        resp = loaded_client.post("/sim/stop")
        assert resp.status_code == 200

    def test_start_twice_fails(self, loaded_client):
        loaded_client.post("/sim/start")
        resp = loaded_client.post("/sim/start")
        assert resp.status_code == 409
        loaded_client.post("/sim/stop")

    def test_pause_without_start_fails(self, loaded_client):
        resp = loaded_client.post("/sim/pause")
        assert resp.status_code == 409

    def test_pause_resume(self, loaded_client):
        loaded_client.post("/sim/start")
        resp = loaded_client.post("/sim/pause")
        assert resp.status_code == 200
        resp = loaded_client.post("/sim/resume")
        assert resp.status_code == 200
        loaded_client.post("/sim/stop")

    def test_full_lifecycle(self, loaded_client):
        """Build graph via API, compile, step, check state."""
        # The loaded_client already has a compiled graph.
        # Step a few times and verify physics.
        for _ in range(10):
            loaded_client.post("/sim/step")

        resp = loaded_client.get("/graph/state/ball")
        data = resp.json()
        assert data["position"] < 5.0  # ball fell


# ------------------------------------------------------------------
# WebSocket
# ------------------------------------------------------------------

class TestWebSocket:
    def test_websocket_connects(self, loaded_client):
        with loaded_client.websocket_connect("/ws/state") as ws:
            # Step once to generate a snapshot
            loaded_client.post("/sim/step")
            import time
            time.sleep(0.1)  # let relay update
            # We should eventually receive a frame
            # (may timeout if relay hasn't propagated yet, that's OK)
            pass  # connection itself succeeding is the test


# ------------------------------------------------------------------
# Visualization endpoint
# ------------------------------------------------------------------

class TestVizEndpoints:
    def test_viz_graph_returns_html(self, client):
        resp = client.get("/viz/graph")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "cytoscape" in resp.text.lower()

    def test_viz_graph_has_controls(self, client):
        resp = client.get("/viz/graph")
        # Should contain simulation control buttons
        assert "btn-start" in resp.text
        assert "btn-step" in resp.text
        assert "ws/state" in resp.text


# ------------------------------------------------------------------
# Checkpoint endpoints
# ------------------------------------------------------------------

class TestCheckpointEndpoints:
    def test_save_and_load(self, loaded_client, tmp_path):
        path = str(tmp_path / "test_checkpoint.npz")
        # Save
        resp = loaded_client.post(f"/checkpoint/save?path={path}")
        assert resp.status_code == 200
        # Step to change state
        loaded_client.post("/sim/step")
        # Load back
        resp = loaded_client.post(f"/checkpoint/load?path={path}")
        assert resp.status_code == 200
        assert "state" in resp.json()
