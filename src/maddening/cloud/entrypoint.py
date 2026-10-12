"""Cloud container entrypoint.

Reads configuration from the ``MADDENING_CLOUD_CONFIG`` environment
variable (a JSON blob), serves the FastAPI API, and streams the running
simulation to a WebRTC viewer.  Handles SIGTERM for graceful shutdown.

What the container streams
--------------------------
With GStreamer and PyGObject installed, a
:class:`~maddening.cloud.selkies_session.SelkiesSession` is started with
``CloudConfig.stream_config``.  Its signaling WebSocket listens on port
8443 (the port ``docker/Dockerfile.cloud`` exposes for it) of the API's
bind address, ``MADDENING_HOST``, and authenticates every viewer against
``MADDENING_STREAM_SECRET``: unset, the session holds a random secret
nobody knows and refuses every viewer.  The session id is logged at
start-up (the token is not); a viewer that holds the secret derives its
token as ``maddening.cloud.generate_session_token(session_id, secret)``
and presents it as ``Authorization: Bearer <token>`` or
``?token=<token>`` on ``ws://<host>:8443/signaling/<session_id>``.

The picture is the latest state the simulation published to the server's
relay (``POST /sim/start``, ``/sim/step``, ``/sim/run``), drawn by a
:class:`~maddening.api.frame_renderer.ServerFrameRenderer` as a time
series of the graph's scalar state fields (at most eight, by node and
field name; a graph with none shows an empty panel) and pushed through
:class:`~maddening.viz.backends.selkies_renderer.SelkiesRenderer` at the
stream's frame rate.  Without GStreamer the server starts and logs that
streaming is disabled.

The server's node registry holds the built-in node types
(``maddening.nodes``), so a client builds its graph through the REST
graph routes; ``MADDENING_GRAPH_USD`` is not loaded (see :func:`main`).
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import signal
import sys
import threading
import time
from typing import Any, Mapping, Optional

from maddening.viz.renderer import Renderer

logger = logging.getLogger(__name__)

#: Shared secret for WebRTC signaling tokens.  Unset means no client can
#: authenticate against the signaling server (see :class:`SelkiesSession`).
STREAM_SECRET_ENV = "MADDENING_STREAM_SECRET"

#: The WebRTC signaling port: the one ``docker/Dockerfile.cloud`` exposes
#: for it, and the one the documents name.
_SIGNALING_PORT = 8443

#: At most this many scalar state fields are drawn in the streamed picture.
_MAX_STREAMED_FIELDS = 8

#: The streamed figure's height in inches; its dpi is chosen so that the
#: figure is the stream's size in pixels, so the picture is laid out the
#: same at every resolution (matplotlib's layout works in inches).
_FIGURE_HEIGHT_IN = 4.8  # units: inches

#: How long stopping the stream waits for the thread that feeds it, in seconds.
_FEEDER_STOP_TIMEOUT_S = 10.0  # units: seconds

#: The state pushed while the relay holds no snapshot (an empty graph, a
#: graph not yet stepped, or just reset): one object, so the picture of
#: "nothing yet" is drawn once rather than at every frame.
_NO_STATE: dict[str, dict] = {}


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

    logger.info("Cloud entrypoint starting with config: %s", _redacted(config))

    # Import simulation components
    from maddening.cloud.session import CloudConfig

    cloud_config = CloudConfig.from_dict(config) if config else CloudConfig()
    stream_config = cloud_config.stream_config

    # Build simulation graph.  Not implemented: this entry point never
    # loads MADDENING_GRAPH_USD, so the server below starts with an empty
    # graph whatever the variable says.  The loader exists
    # (maddening.usd.load_graph_from_usd); wiring it in here is future
    # work.  The log line says so rather than announcing a load that
    # does not happen.  A client builds the graph through the REST graph
    # routes instead, from the built-in node types registered below.
    graph_usd = os.environ.get("MADDENING_GRAPH_USD", "")
    if graph_usd:
        logger.warning(
            "MADDENING_GRAPH_USD=%s is set, but this entry point does not load "
            "a graph from USD yet; the server starts with an empty graph.",
            graph_usd,
        )

    # Start FastAPI server.  The bind address is read here rather than
    # just before uvicorn.run() because the app has to know it: a
    # non-loopback bind is what turns the API's bearer token on, and the
    # app cannot see the socket uvicorn opens.
    host = os.environ.get("MADDENING_HOST", "0.0.0.0")
    port = int(os.environ.get("MADDENING_PORT", "8000"))

    from maddening.api.server import SimulationServer, warn_if_publicly_bound
    # The built-in node types: with no registry at all (as before 0.4.0)
    # every POST /graph/nodes was refused, so the container's API could
    # not build a graph and there was nothing to run or stream.
    server = SimulationServer(node_registry=_builtin_node_registry(), bind_host=host)

    # v0.2 #8: resume from a remote checkpoint URL if requested.
    resume_from_env(server)

    # Graceful shutdown on SIGTERM.  uvicorn installs its own handler while
    # it serves and, once it has shut down, raises the signal again for
    # this one; the SystemExit leaves uvicorn.run() and the finally below
    # stops the stream.  Before uvicorn starts, this handler alone runs.
    def _handle_sigterm(signum, frame):
        logger.info("Received SIGTERM, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    stream: Optional[_Stream] = None
    try:
        # Stream the simulation.  The signaling server authenticates every
        # client against this secret, so a deployment that wants a remote
        # viewer must set MADDENING_STREAM_SECRET and share it with that
        # viewer; without it the session generates a random secret nobody
        # holds and every signaling connection is rejected.  The socket
        # binds the API's own address.
        stream = _start_streaming(
            server, stream_config,
            secret=os.environ.get(STREAM_SECRET_ENV, ""),
            bind_host=host,
        )

        # Run uvicorn
        import uvicorn

        logger.info("Starting server on %s:%d", host, port)
        # The default is 0.0.0.0 because a container bound to 127.0.0.1 is
        # unreachable even with a published port.  That makes the exposure
        # the normal case here, so the API demands a bearer token: print it
        # (when it was generated) and then say what is still exposed.
        #
        # A container whose log nobody reads must set MADDENING_API_TOKEN,
        # or MADDENING_API_TOKEN_FILE to a path on a mounted volume.
        server.auth.announce(port)
        warn_if_publicly_bound(host, port)
        # uvicorn's proxy_headers (on by default, trusting 127.0.0.1) rewrites
        # the peer the API's authentication sees from X-Forwarded-For, so it
        # is off here unless the operator opts in by setting uvicorn's own
        # FORWARDED_ALLOW_IPS to the proxy that fronts this container.
        uvicorn.run(
            server.create_app(),
            host=host,
            port=port,
            log_level="info",
            proxy_headers=bool(os.environ.get("FORWARDED_ALLOW_IPS", "").strip()),
        )
    finally:
        # The server's own threads (the runner) stop in the app's lifespan
        # shutdown, inside uvicorn.run(); the stream's here.
        if stream is not None:
            stream.stop()


#: Keys whose values are credentials, never logged (``RTCIceServer``'s
#: TURN ``username`` and ``credential``, and the like), compared lower-case.
_SECRET_KEYS = frozenset({"credential", "username", "password", "passwd", "secret",
                          "token", "api_key", "apikey"})


def _redacted(value: Any) -> Any:
    """*value* with every credential replaced by ``"<redacted>"``, for the log.

    Walks dicts and lists: a dict key in ``_SECRET_KEYS`` (any case) has
    its value replaced, and a ``scheme://user:password@host`` URL string has
    its user and password replaced, so a TURN server in
    ``stream_config.ice_servers`` is logged without its credentials.
    """
    if isinstance(value, dict):
        return {key: "<redacted>" if str(key).lower() in _SECRET_KEYS else _redacted(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redacted(item) for item in value]
    if isinstance(value, str) and "@" in value:
        return re.sub(r"^([A-Za-z][A-Za-z0-9+.-]*:(?://)?)[^@/]*@", r"\1<redacted>@", value)
    return value


def _builtin_node_registry() -> dict[str, type]:
    """The node types the container's REST API can add: ``maddening.nodes``."""
    import maddening.nodes as nodes

    return {name: getattr(nodes, name) for name in nodes.__all__}


# ---------------------------------------------------------------------------
# Streaming: the server's frames, through SelkiesRenderer, into the session
# ---------------------------------------------------------------------------


def _start_streaming(
    server,
    stream_config,
    *,
    secret: str,
    bind_host: str,
    signaling_port: int = _SIGNALING_PORT,
) -> Optional["_Stream"]:
    """Start streaming *server*'s simulation, or say why it is not streamed.

    Builds the :class:`SelkiesSession` (``secret``, ``signaling_port``,
    ``bind_host``) and hands it to :func:`_stream_through`.  Returns
    ``None`` -- the server then runs without streaming -- when GStreamer
    is not installed, as before.
    """
    try:
        from maddening.cloud.selkies_session import SelkiesSession
        session = SelkiesSession(
            secret=secret, signaling_port=signaling_port, bind_host=bind_host,
        )
    except ImportError:
        logger.warning("GStreamer not available; streaming disabled")
        return None
    return _stream_through(server, session, stream_config)


def _stream_through(server, session, stream_config) -> Optional["_Stream"]:
    """Start *session* with *stream_config* and feed it *server*'s frames.

    The frames are drawn by :class:`_StateFrames` and pushed through a
    :class:`SelkiesRenderer` (whose ``setup`` starts the session) by a
    thread that reads the server's relay at ``stream_config.fps``.
    Returns the running :class:`_Stream`, or ``None`` after logging why
    when the frames cannot be drawn in the configured format or the
    session cannot start (for example, the signaling port is taken): the
    API is still served.
    """
    from maddening.cloud.streaming import StreamStartError
    from maddening.viz.backends.selkies_renderer import SelkiesRenderer
    from maddening.viz.renderer import GraphInfo

    if stream_config.pixel_format != "RGBA":
        logger.error(
            "Streaming disabled: the entry point draws RGBA frames, but "
            "stream_config.pixel_format is %r.", stream_config.pixel_format,
        )
        return None
    try:
        frames = _StateFrames(stream_config.width, stream_config.height)
    except ImportError as exc:
        logger.error("Streaming disabled: the frames cannot be drawn (%s).", exc)
        return None
    renderer = SelkiesRenderer(frames, session, config=stream_config)
    stream = _Stream(server, renderer, stream_config.fps)
    try:
        # The graph is built later, through the REST routes: the frames
        # read it from each snapshot, not from this.
        stream.start(GraphInfo(node_names=[], node_params={}, node_state_fields={},
                               edges=[], timestep=0.0))
    except StreamStartError as exc:
        logger.error("Streaming disabled: the WebRTC session did not start (%s).", exc)
        return None
    info = renderer.stream_info
    session_id = info.session_id if info is not None else "?"
    logger.info(
        "WebRTC streaming started: %dx%d at %d fps. A viewer connects to "
        "ws://<this host>:%s/signaling/%s and presents the token "
        "generate_session_token(%r, $%s) as 'Authorization: Bearer <token>' "
        "or '?token=<token>'.",
        stream_config.width, stream_config.height, stream_config.fps,
        _port_of(info.signaling_url) if info is not None else "?",
        session_id, session_id, STREAM_SECRET_ENV,
    )
    return stream


def _port_of(url: str) -> Optional[int]:
    import urllib.parse

    return urllib.parse.urlsplit(url).port


class _Stream:
    """A started stream: its :class:`SelkiesRenderer` and the thread feeding it.

    The thread (``maddening-stream-frames``) wakes ``fps`` times a second,
    reads the server relay's latest snapshot and hands it to the
    renderer's ``update``, which draws it when it is new and pushes the
    picture to the session either way: a viewer who connects while the
    simulation is paused still receives frames.  The session drops them
    while no viewer is connected.
    """

    def __init__(self, server, renderer, fps: int) -> None:
        self._server = server
        self._renderer = renderer
        self._period = 1.0 / max(1, int(fps))
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self, graph_info) -> None:
        """Start the session (``renderer.setup``), then the feeding thread."""
        self._renderer.setup(graph_info)
        self._thread = threading.Thread(
            target=self._feed, name="maddening-stream-frames", daemon=True,
        )
        self._thread.start()

    def _feed(self) -> None:
        reported = False
        due = time.monotonic()
        while True:
            due += self._period
            delay = due - time.monotonic()
            if delay < 0:
                # Behind (a slow frame): carry on from now, without a burst.
                due, delay = time.monotonic(), 0.0
            if self._stopping.wait(delay):
                return
            try:
                _seq, sim_time, snapshot = self._server.relay.latest_frame()
                self._renderer.update(sim_time, _NO_STATE if snapshot is None else snapshot)
            except Exception:  # noqa: BLE001 - reported once; the stream carries on
                if not reported:
                    logger.exception("A streamed frame could not be read, drawn or pushed")
                    reported = True

    def stop(self) -> None:
        """Stop the feeding thread, then the renderer and the session."""
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(_FEEDER_STOP_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning("The maddening-stream-frames thread did not stop within %s s",
                               _FEEDER_STOP_TIMEOUT_S)
        # SelkiesRenderer.teardown stops the session: its signaling server,
        # the viewer's pipeline and the GLib loop.
        self._renderer.teardown()


def _scalar_fields(state: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(node, field)`` of a snapshot's real scalar fields, sorted, at most :data:`_MAX_STREAMED_FIELDS`."""
    found: list[tuple[str, str]] = []
    for node in sorted(state):
        fields = state[node]
        if not isinstance(fields, Mapping):
            continue
        for name in sorted(fields):
            value = fields[name]
            dtype = getattr(value, "dtype", None)
            kind = getattr(dtype, "kind", None)
            if dtype is not None:
                scalar = getattr(value, "ndim", None) == 0 and kind in ("b", "i", "u", "f")
            else:
                scalar = isinstance(value, (int, float))
            if scalar:
                found.append((str(node), str(name)))
                if len(found) == _MAX_STREAMED_FIELDS:
                    return found
    return found


def _plotted_value(value: Any) -> float:
    """*value* as a float to plot; a non-finite value is a gap (``nan``)."""
    number = float(value)
    return number if math.isfinite(number) else math.nan


class _StateFrames(Renderer):
    """The streamed picture, drawn by the server's frame renderer.

    A :class:`~maddening.api.frame_renderer.ServerFrameRenderer` plots a
    time series of the snapshot's scalar fields (see
    :func:`_scalar_fields`); it is rebuilt when that set of fields changes
    (a node added, a graph rebuilt or emptied).  Its image is decoded to
    RGBA at the stream's size, as example 07 does, for
    :class:`SelkiesRenderer` to read with :meth:`read_framebuffer_cpu`.
    Only a snapshot not seen before is drawn: pushed again, the same
    snapshot object is the same picture, so a paused simulation costs no
    drawing and its time series gains no repeated points.
    """

    def __init__(self, width: int, height: int) -> None:
        # Imported here so that a missing matplotlib or Pillow is found
        # when the stream starts, not by the feeding thread.
        from maddening.api.frame_renderer import ServerFrameRenderer, TimeSeriesConfig
        from PIL import Image

        self._renderer_cls = ServerFrameRenderer
        self._series_cls = TimeSeriesConfig
        self._image = Image
        self.width = int(width)
        self.height = int(height)
        self._dpi = max(1, round(self.height / _FIGURE_HEIGHT_IN))
        self._fields: Optional[list[tuple[str, str]]] = None
        self._renderer: Any = None
        self._last_state: Any = None
        self._drawn = False
        self._pixels = b"\xff" * (self.width * self.height * 4)

    # -- the Renderer interface SelkiesRenderer calls ---------------------

    def setup(self, graph_info) -> None:
        """Nothing to set up: the graph is read from each snapshot."""

    def update(self, sim_time: float, state: Mapping[str, Any]) -> None:
        """Draw *state* unless it is the snapshot drawn last."""
        if self._drawn and state is self._last_state:
            return
        fields = _scalar_fields(state)
        if self._renderer is None or fields != self._fields:
            self._renderer = self._new_renderer(fields)
            self._fields = fields
        values: dict[str, dict[str, float]] = {}
        for node, name in fields:
            values.setdefault(node, {})[name] = _plotted_value(state[node][name])
        encoded = self._renderer.render(float(sim_time), values)
        image = self._image.open(io.BytesIO(encoded)).convert("RGBA")
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height))
        self._pixels = image.tobytes()
        self._last_state = state
        self._drawn = True

    def read_framebuffer_cpu(self) -> tuple[bytes, int, int, str]:
        """The last picture drawn: ``(pixels, width, height, "RGBA")``."""
        return self._pixels, self.width, self.height, "RGBA"

    def teardown(self) -> None:
        """Drop the figure."""
        self._renderer = None
        self._last_state = None
        self._drawn = False

    def _new_renderer(self, fields: list[tuple[str, str]]):
        series = [self._series_cls(
            fields=[(node, name, f"{node}.{name}") for node, name in fields],
            title="Simulation state",
        )] if fields else None
        return self._renderer_cls(
            timeseries=series, width=self.width, height=self.height, dpi=self._dpi,
            fmt="jpeg", quality=90,
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
