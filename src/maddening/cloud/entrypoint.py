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

    # v0.2 #8: resume from a remote checkpoint URL if requested.
    resume_url = os.environ.get("RESUME_FROM_URL", "")
    if resume_url:
        try:
            resume_from_url(server, resume_url)
            logger.info("Resumed simulation state from %s", resume_url)
        except Exception:
            # Non-fatal: log and continue with the in-memory state.
            logger.exception(
                "Failed to resume from %s; starting fresh", resume_url,
            )

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


def resume_from_url(server, url: str, *, skip_integrity_check: bool = False) -> dict:
    """Download a checkpoint from *url* and restore the server's graph.

    Used by the cloud entrypoint when the ``RESUME_FROM_URL`` env var
    is set — typically by the orchestrator that just relaunched after
    a spot preemption.

    Supported URL schemes: ``file://``, ``http(s)://``.

    Returns the checkpoint manifest dict for caller logging.
    """
    from maddening.core.simulation.checkpoint import download_and_load_state
    return download_and_load_state(
        server.gm, url, skip_integrity_check=skip_integrity_check,
    )


def make_preempt_snapshot_hook(
    server,
    *,
    snapshot_path: Optional[str] = None,
    extra_meta: Optional[dict] = None,
):
    """Build an ``on_preempted`` callback that auto-snapshots the
    server's GraphManager state to disk (v0.2 #8).

    Parameters
    ----------
    server : SimulationServer
        Source of the GraphManager to snapshot.
    snapshot_path : str, optional
        Destination (defaults to ``$MADDENING_SNAPSHOT_DIR`` or
        ``/tmp/maddening_preempt_snapshot.npz``).
    extra_meta : dict, optional
        Caller-supplied dict merged into the manifest's ``extra``
        block (commit hash, cluster id, sim_time, etc.).

    Returns
    -------
    callable
        ``(CloudSessionInfo) -> None`` suitable for
        ``CloudSession(on_preempted=...)``.
    """
    from maddening.core.simulation.checkpoint import save_state_with_manifest

    if snapshot_path is None:
        snapshot_path = os.environ.get(
            "MADDENING_SNAPSHOT_PATH",
        ) or os.path.join(
            os.environ.get("MADDENING_SNAPSHOT_DIR", "/tmp"),
            "maddening_preempt_snapshot.npz",
        )

    def _hook(info) -> None:
        try:
            extra = dict(extra_meta or {})
            extra["session_id"] = getattr(info, "session_id", None)
            extra["stage_at_snapshot"] = (
                info.stage.value if hasattr(info.stage, "value")
                else str(info.stage)
            )
            npz_path, manifest_path = save_state_with_manifest(
                server.gm, snapshot_path, extra=extra,
            )
            logger.info(
                "Preemption snapshot written: %s (manifest=%s)",
                npz_path, manifest_path,
            )
        except Exception:
            logger.exception("Failed to snapshot state on preemption")

    return _hook


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
