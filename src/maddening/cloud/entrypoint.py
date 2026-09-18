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
    resume_from_env(server)

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


#: Environment variables the entry point reads for the resume step.
RESUME_URL_ENV = "RESUME_FROM_URL"
RESUME_MANIFEST_URL_ENV = "RESUME_MANIFEST_URL"
RESUME_TIMEOUT_ENV = "MADDENING_RESUME_TIMEOUT"


def resume_from_env(server, environ: Optional[dict] = None) -> Optional[dict]:
    """Run the ``RESUME_FROM_URL`` step of the entry point (non-fatal).

    Reads ``RESUME_FROM_URL`` (checkpoint), ``RESUME_MANIFEST_URL``
    (optional: the manifest's own URL, needed with presigned storage
    because a presigned URL authorises one object only) and
    ``MADDENING_RESUME_TIMEOUT`` (seconds, default
    :data:`maddening.cloud.resume.DEFAULT_TIMEOUT`).  Any failure is
    logged and swallowed so a bad checkpoint never blocks a healthy
    server from starting.  URLs are logged with their query string
    redacted (presigned signatures are credentials).

    A failure is logged as ``RESUME FAILED``, never as a fresh start: the
    restore is atomic (see
    :func:`maddening.core.simulation.checkpoint.load_state`), so the
    graph is left exactly as this process built it, and the log has to
    let an operator tell that apart from a run that was never asked to
    resume.

    Parameters
    ----------
    server : SimulationServer
        Owner of the ``GraphManager`` to restore into.
    environ : dict, optional
        Environment mapping; defaults to :data:`os.environ`.

    Returns
    -------
    dict or None
        The manifest when the resume succeeded, else ``None`` (nothing
        requested, graph empty, or the resume failed).
    """
    env = os.environ if environ is None else environ
    url = env.get(RESUME_URL_ENV, "")
    if not url:
        return None
    shown = redact_url(url)
    node_names = getattr(server.gm, "node_names", None)
    if node_names is not None and len(node_names) == 0:
        # TODO(graph loading): main() has no graph source yet (MADDENING_GRAPH_USD
        # is a stub), so a checkpoint can never match.  Say so instead of
        # failing later with "Checkpoint node mismatch".
        logger.error(
            "%s=%s is set but the server's graph has no nodes; resume is "
            "impossible until a graph is loaded (MADDENING_GRAPH_USD loading is "
            "not implemented). Starting fresh.",
            RESUME_URL_ENV, shown,
        )
        return None
    manifest_url = env.get(RESUME_MANIFEST_URL_ENV) or None
    timeout_raw = env.get(RESUME_TIMEOUT_ENV, "")
    kwargs: dict = {}
    if timeout_raw:
        try:
            kwargs["timeout"] = float(timeout_raw)
        except ValueError:
            logger.warning(
                "Ignoring non-numeric %s=%r; using the default timeout",
                RESUME_TIMEOUT_ENV, timeout_raw,
            )
    try:
        manifest = resume_from_url(server, url, manifest_url=manifest_url, **kwargs)
    except Exception:
        # Non-fatal, but say what actually happened.  This used to log
        # "starting fresh", which is what a run with no RESUME_FROM_URL
        # at all does -- an operator reading the log could not tell a
        # clean start from a resume that did not happen.  The restore
        # itself is atomic (checkpoint.load_state rolls back), so the
        # graph here is the one this process built at start-up.
        logger.exception(
            "RESUME FAILED from %s (%s was set): the graph was NOT restored "
            "and still holds the state this process built at start-up, so "
            "this run does NOT continue the checkpointed one. A genuine "
            "fresh start logs nothing here.",
            shown, RESUME_URL_ENV,
        )
        return None
    extra = manifest.get("extra") or {}
    logger.info(
        "Resumed simulation state from %s (schema_version=%s size_bytes=%s "
        "sha256=%s session_id=%s stage_at_snapshot=%s)",
        shown,
        manifest.get("schema_version"),
        manifest.get("size_bytes"),
        (manifest.get("sha256") or "")[:12] or None,
        extra.get("session_id"),
        extra.get("stage_at_snapshot"),
    )
    return manifest


def redact_url(url: str) -> str:
    """*url* with its query string replaced by ``<redacted>`` for logging.

    Presigned URLs carry their signature in the query string, so the
    raw URL must never reach the container log.  The fragment is
    dropped too; scheme, host and path are kept so the log still says
    where the checkpoint came from.
    """
    import urllib.parse

    parts = urllib.parse.urlsplit(url)
    if not parts.query and not parts.fragment:
        return url
    return urllib.parse.urlunsplit(
        parts._replace(query="<redacted>" if parts.query else "", fragment=""),
    )


def resume_from_url(
    server,
    url: str,
    *,
    skip_integrity_check: bool = False,
    manifest_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> dict:
    """Download a checkpoint from *url* and restore the server's graph.

    Used by the cloud entrypoint when the ``RESUME_FROM_URL`` env var
    is set — typically by the orchestrator that just relaunched after
    a spot preemption.

    Supported URL schemes are the closed allow-list of
    :func:`maddening.cloud.resume.download_and_load_state` (``file://``,
    ``http(s)://`` and the listed ``fsspec`` cloud-storage schemes).
    *manifest_url* and *timeout* are forwarded unchanged (``None`` keeps
    the transport's defaults).

    Returns the checkpoint manifest dict for caller logging.
    """
    from maddening.cloud.resume import download_and_load_state
    kwargs: dict = {}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return download_and_load_state(
        server.gm, url,
        skip_integrity_check=skip_integrity_check,
        manifest_url=manifest_url,
        **kwargs,
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
