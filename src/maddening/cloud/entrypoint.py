"""Cloud container entrypoint.

Reads configuration from ``MADDENING_CLOUD_CONFIG`` environment variable
(JSON blob), creates a streaming session, starts the simulation, and
serves the FastAPI API.  Handles SIGTERM for graceful shutdown.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from typing import Optional

logger = logging.getLogger(__name__)


def main() -> None:
    """Cloud entrypoint: configure and run the simulation server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load configuration
    config_json = os.environ.get("MADDENING_CLOUD_CONFIG", "")
    if config_json:
        config = json.loads(config_json)
    else:
        config = _build_config_from_env()

    logger.info("Cloud entrypoint starting with config: %s", config)

    # Import simulation components
    from maddening.cloud.session import CloudConfig
    from maddening.cloud.streaming import StreamConfig, QualityPreset

    cloud_config = CloudConfig.from_dict(config) if config else CloudConfig()
    stream_config = cloud_config.stream_config

    # Create streaming session
    session: Optional[object] = None
    try:
        from maddening.cloud.selkies_session import SelkiesSession
        session = SelkiesSession()
        logger.info("Using SelkiesSession for streaming")
    except ImportError:
        logger.warning("GStreamer not available; streaming disabled")

    # Build simulation graph
    graph_usd = os.environ.get("MADDENING_GRAPH_USD", "")
    if graph_usd:
        logger.info("Loading graph from USD: %s", graph_usd)
        # USD graph loading would go here
        # from maddening.usd import load_graph
        # gm = load_graph(graph_usd)

    # Start FastAPI server
    from maddening.api.server import SimulationServer
    server = SimulationServer(node_registry={})

    # Graceful shutdown on SIGTERM
    shutdown_event = None

    def _handle_sigterm(signum, frame):
        logger.info("Received SIGTERM, shutting down...")
        if session is not None and hasattr(session, "stop"):
            session.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # Run uvicorn
    import uvicorn

    host = os.environ.get("MADDENING_HOST", "0.0.0.0")
    port = int(os.environ.get("MADDENING_PORT", "8000"))

    logger.info("Starting server on %s:%d", host, port)
    uvicorn.run(
        server.create_app(),
        host=host,
        port=port,
        log_level="info",
    )


def _build_config_from_env() -> dict:
    """Build a config dict from individual environment variables."""
    config: dict = {}

    preset = os.environ.get("MADDENING_STREAM_PRESET", "")
    if preset:
        from maddening.cloud.streaming import QualityPreset
        config["stream_config"] = {
            "width": 1280, "height": 720, "fps": 30,
            "bitrate_kbps": 4000, "codec": "h264",
            "pixel_format": "RGBA", "enable_audio": False,
            "ice_servers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        }
        try:
            p = QualityPreset(preset.lower())
            from maddening.cloud.streaming import StreamConfig
            sc = StreamConfig.from_preset(p)
            config["stream_config"]["width"] = sc.width
            config["stream_config"]["height"] = sc.height
            config["stream_config"]["fps"] = sc.fps
            config["stream_config"]["bitrate_kbps"] = sc.bitrate_kbps
        except ValueError:
            pass

    return config


if __name__ == "__main__":
    main()
