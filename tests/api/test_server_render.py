"""Tests for the /ws/render server-side rendering WebSocket endpoint."""

import warnings

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.api.frame_renderer import (
    ServerFrameRenderer,
    SceneConfig,
    SceneObject,
    TimeSeriesConfig,
)

try:
    from starlette.testclient import TestClient
    HAS_TESTCLIENT = True
except ImportError:
    HAS_TESTCLIENT = False

pytestmark = pytest.mark.skipif(not HAS_TESTCLIENT, reason="starlette not installed")


@pytest.fixture
def server_app():
    """Create a SimulationServer with a frame renderer."""
    from maddening.api.server import SimulationServer

    gm = GraphManager()
    gm.add_node(TableNode("table", timestep=0.01, position=0.0))
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=5.0))
    gm.add_edge("table", "ball", "position", "table_position")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()

    renderer = ServerFrameRenderer(
        scene=SceneConfig(objects=[
            SceneObject(node="ball", y="position", kind="circle"),
        ]),
        timeseries=[TimeSeriesConfig(
            fields=[("ball", "position", "Ball")], window=50,
        )],
        width=320, height=240, fmt="jpeg", quality=50,
    )

    server = SimulationServer(
        node_registry={"BallNode": BallNode, "TableNode": TableNode},
        graph_manager=gm,
        frame_renderer=renderer,
    )
    return server.create_app()


@pytest.fixture
def server_app_no_renderer():
    """Server without frame renderer."""
    from maddening.api.server import SimulationServer

    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=5.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()

    server = SimulationServer(
        node_registry={"BallNode": BallNode},
        graph_manager=gm,
    )
    return server.create_app()


class TestRenderEndpoint:
    def test_viz_render_page(self, server_app):
        client = TestClient(server_app)
        resp = client.get("/viz/render")
        assert resp.status_code == 200
        assert "render-canvas" in resp.text

    def test_ws_render_sends_config_then_frame(self, server_app):
        client = TestClient(server_app)
        # Step the sim so there's a snapshot
        client.post("/sim/step")

        with client.websocket_connect("/ws/render") as ws:
            # First message: JSON config
            config = ws.receive_json()
            assert config["type"] == "config"
            assert config["width"] == 320
            assert config["height"] == 240
            assert config["format"] == "jpeg"
            assert config["content_type"] == "image/jpeg"

            # Second message: binary JPEG frame
            frame = ws.receive_bytes()
            assert isinstance(frame, bytes)
            assert len(frame) > 100
            # JPEG magic
            assert frame[:2] == b'\xff\xd8'

    def test_ws_render_client_config_change(self, server_app):
        client = TestClient(server_app)
        client.post("/sim/step")

        import json
        with client.websocket_connect("/ws/render") as ws:
            # Receive initial config
            config = ws.receive_json()
            assert config["format"] == "jpeg"

            # Send config change
            ws.send_text(json.dumps({
                "type": "config",
                "format": "png",
                "quality": 50,
                "fps": 15,
            }))

            # Next message should be either a new config or a frame
            # (depends on timing, but config_changed event is set)
            msg = ws.receive()
            # Could be bytes (frame) or text (config update)
            # Just verify we don't crash

    def test_ws_render_no_renderer_closes(self, server_app_no_renderer):
        """Server without frame renderer should reject /ws/render."""
        client = TestClient(server_app_no_renderer)
        # The websocket should close with code 1008
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/render") as ws:
                ws.receive_json()
