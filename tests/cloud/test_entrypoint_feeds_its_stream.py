"""The cloud entry point starts its streaming session and feeds it the simulation's frames.

These run on every lane, without GStreamer: the session is
``MockStreamSession`` (which records the frames pushed into it) or a
recording stand-in for ``SelkiesSession``; the server is a real
``SimulationServer`` whose graph is stepped in the test.  What GStreamer
does with the frames -- the answer, ICE, the decoded video -- is tested
in the streaming container by ``test_entrypoint_streams``.

Before the fix the entry point built a ``SelkiesSession`` and never
started it, never gave it a frame, and built its server with no node
types at all.
"""

from __future__ import annotations

import json
import logging
import threading
import time

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="the entry point serves the FastAPI app")
pytest.importorskip("matplotlib", reason="the entry point draws its frames with matplotlib")

from maddening.api.server import SimulationServer  # noqa: E402
from maddening.cloud import entrypoint, selkies_session  # noqa: E402
from maddening.cloud.mock_streaming import MockStreamSession  # noqa: E402
from maddening.cloud.streaming import StreamConfig  # noqa: E402
from maddening.nodes.spring import SpringDamperNode  # noqa: E402
from maddening.warnings import PerformanceWarning  # noqa: E402

WIDTH, HEIGHT, FPS = 160, 96, 20
CONFIG = StreamConfig(width=WIDTH, height=HEIGHT, fps=FPS, bitrate_kbps=500, ice_servers=[])
FEEDER = "maddening-stream-frames"
#: Long enough for several frames at ``FPS`` on a loaded runner.
WAIT_S = 10.0


def _server_with_a_spring() -> SimulationServer:
    server = SimulationServer(node_registry=entrypoint._builtin_node_registry(),
                              bind_host="127.0.0.1")
    server.gm.add_node(SpringDamperNode("spring", timestep=0.01, stiffness=50.0,
                                        initial_position=3.0))
    server.gm.compile()
    server.relay.attach(server.gm)
    return server


def _wait_for(condition, what: str) -> None:
    deadline = time.monotonic() + WAIT_S
    while not condition():
        assert time.monotonic() < deadline, f"{what} did not happen within {WAIT_S} s"
        time.sleep(0.02)


def _feeders() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == FEEDER and t.is_alive()]


def _stream(server, session, config=CONFIG):
    made: list = []
    try:
        with pytest.warns(PerformanceWarning, match="CPU path"):
            made.append(entrypoint._stream_through(server, session, config))
    except BaseException:
        # A stream left feeding outlives the test (its thread is a daemon).
        if made and made[0] is not None:
            made[0].stop()
        raise
    return made[0]


# ---------------------------------------------------------------------------
# The feed
# ---------------------------------------------------------------------------


def test_the_session_is_started_with_the_stream_config_and_fed_frames_of_its_size():
    server, session = _server_with_a_spring(), MockStreamSession(max_frames=500)
    stream = _stream(server, session)
    try:
        assert stream is not None
        assert session.is_alive() and session.config == CONFIG
        _wait_for(lambda: session.frame_count >= 3, "three frames")
        assert {len(frame) for frame in list(session._frames)} == {WIDTH * HEIGHT * 4}
        # A drawn figure (white background), not an empty buffer.
        frame = np.frombuffer(session.last_frame, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)
        assert frame[..., :3].mean() > 150 and (frame[..., 3] == 255).all()
    finally:
        stream.stop()
    assert not session.is_alive()
    assert not _feeders()


def test_the_frames_change_as_the_graph_steps_and_hold_while_it_does_not():
    server, session = _server_with_a_spring(), MockStreamSession(max_frames=500)
    stream = _stream(server, session)
    try:
        server.gm.step()
        _wait_for(lambda: server.relay.latest_frame()[2] is not None, "a snapshot")
        count = session.frame_count
        _wait_for(lambda: session.frame_count >= count + 3, "frames of the first snapshot")
        before = session.last_frame
        # Still: the same snapshot is the same picture, frame after frame.
        count = session.frame_count
        _wait_for(lambda: session.frame_count >= count + 3, "more frames")
        assert session.last_frame == before
        for _ in range(60):
            server.gm.step()
        seq = server.relay.latest_frame()[0]
        count = session.frame_count
        _wait_for(lambda: session.frame_count >= count + 3, "frames after stepping")
        assert server.relay.latest_frame()[0] == seq
        assert session.last_frame != before
    finally:
        stream.stop()


def test_frames_keep_coming_at_the_stream_rate_while_the_simulation_is_paused():
    """A viewer who connects during a pause still needs frames (a keyframe).

    The frame rate is the stream's: over two seconds, between a quarter and
    one and a half times ``FPS`` (an unchanged picture is pushed again, not
    redrawn, so a loaded runner does not slow it much).
    """
    server, session = _server_with_a_spring(), MockStreamSession(max_frames=500)
    stream = _stream(server, session)
    try:
        _wait_for(lambda: session.frame_count >= 1, "a first frame")
        count, start = session.frame_count, time.monotonic()
        time.sleep(2.0)
        rate = (session.frame_count - count) / (time.monotonic() - start)
        assert FPS / 4 <= rate <= FPS * 1.5, f"{rate:.1f} frames a second for {FPS}"
    finally:
        stream.stop()


# ---------------------------------------------------------------------------
# What the picture shows
# ---------------------------------------------------------------------------


def test_the_picture_plots_the_scalar_fields_of_the_state_and_is_redrawn_only_when_new():
    frames = entrypoint._StateFrames(WIDTH, HEIGHT)
    state = {"spring": {"position": np.float32(3.0), "velocity": np.float32(0.0),
                        "trace": np.zeros(4)}}
    assert entrypoint._scalar_fields(state) == [("spring", "position"), ("spring", "velocity")]

    frames.update(0.0, state)
    first, w, h, fmt = frames.read_framebuffer_cpu()
    assert (w, h, fmt, len(first)) == (WIDTH, HEIGHT, "RGBA", WIDTH * HEIGHT * 4)
    renderer = frames._renderer
    frames.update(0.0, state)  # the same object: nothing is drawn
    assert frames.read_framebuffer_cpu()[0] is first

    # A moved value draws a different picture with the same figure; a
    # non-finite one is a gap, not an error.
    for step, value in enumerate([2.0, 1.0, float("inf"), float("nan"), 0.5], start=1):
        frames.update(step * 0.1, {"spring": {"position": np.float32(value),
                                              "velocity": np.float32(-value),
                                              "trace": np.zeros(4)}})
    assert frames._renderer is renderer
    assert frames.read_framebuffer_cpu()[0] != first

    # Another set of fields (a node added): a new figure for it.
    frames.update(1.0, {**state, "ball": {"position": np.float32(1.0)}})
    assert frames._renderer is not renderer
    assert frames._fields == [("ball", "position"), ("spring", "position"),
                              ("spring", "velocity")]
    frames.teardown()


def test_at_most_eight_fields_are_plotted_and_only_real_scalars():
    state = {f"n{i}": {"x": np.float32(i), "flag": np.bool_(True), "z": np.complex64(1j),
                       "v": np.zeros(3)} for i in range(6)}
    fields = entrypoint._scalar_fields(state)
    assert len(fields) == entrypoint._MAX_STREAMED_FIELDS == 8
    assert all(name in ("x", "flag") for _node, name in fields)


# ---------------------------------------------------------------------------
# When streaming does not start
# ---------------------------------------------------------------------------


def test_without_gstreamer_the_server_runs_and_says_streaming_is_disabled(monkeypatch, caplog):
    monkeypatch.setattr(selkies_session, "_HAS_GST", False)
    with caplog.at_level(logging.WARNING, logger="maddening.cloud.entrypoint"):
        stream = entrypoint._start_streaming(_server_with_a_spring(), CONFIG,
                                             secret="s", bind_host="127.0.0.1")
    assert stream is None
    assert "GStreamer not available; streaming disabled" in caplog.text
    assert not _feeders()


def test_a_session_that_cannot_start_leaves_the_server_running_without_a_stream(caplog):
    with caplog.at_level(logging.ERROR, logger="maddening.cloud.entrypoint"):
        stream = _stream(_server_with_a_spring(), MockStreamSession(fail_on_start=True))
    assert stream is None
    assert "Streaming disabled: the WebRTC session did not start" in caplog.text
    assert not _feeders()


def test_a_pixel_format_the_frames_are_not_drawn_in_is_refused_at_start(caplog):
    session = MockStreamSession()
    with caplog.at_level(logging.ERROR, logger="maddening.cloud.entrypoint"):
        stream = entrypoint._stream_through(
            _server_with_a_spring(), session,
            StreamConfig(width=WIDTH, height=HEIGHT, pixel_format="BGRx", ice_servers=[]))
    assert stream is None and not session.is_alive()
    assert "pixel_format is 'BGRx'" in caplog.text


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class _RecordingSelkies(MockStreamSession):
    """``SelkiesSession`` as ``main`` constructs it, recording its arguments."""

    made: list["_RecordingSelkies"] = []

    def __init__(self, secret: str = "", signaling_port: int = 8443,
                 bind_host: str = "0.0.0.0") -> None:
        super().__init__(max_frames=500)
        self.arguments = {"secret": secret, "signaling_port": signaling_port,
                          "bind_host": bind_host}
        _RecordingSelkies.made.append(self)


def test_main_starts_the_session_from_the_environment_and_stops_it_at_shutdown(monkeypatch):
    import uvicorn

    _RecordingSelkies.made.clear()
    monkeypatch.setattr(selkies_session, "SelkiesSession", _RecordingSelkies)
    monkeypatch.setenv("MADDENING_CLOUD_CONFIG", json.dumps({"stream_config": {
        "width": WIDTH, "height": HEIGHT, "fps": FPS, "bitrate_kbps": 500, "ice_servers": []}}))
    monkeypatch.setenv("MADDENING_STREAM_SECRET", "the-shared-secret")
    monkeypatch.setenv("MADDENING_HOST", "10.1.2.3")
    monkeypatch.setenv("MADDENING_API_TOKEN", "an-api-token")
    for name in ("MADDENING_GRAPH_USD", "MADDENING_PORT", "RESUME_FROM_URL",
                 "FORWARDED_ALLOW_IPS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(entrypoint.signal, "signal", lambda *args: None)
    served: dict = {}

    def serve(app, **kwargs):
        # While uvicorn serves, the session is running and being fed.
        (session,) = _RecordingSelkies.made
        served.update(kwargs, alive=session.is_alive(), feeders=len(_feeders()))
        count = session.frame_count
        _wait_for(lambda: session.frame_count >= count + 3, "frames while serving")

    monkeypatch.setattr(uvicorn, "run", serve)
    with pytest.warns(PerformanceWarning, match="CPU path"):
        entrypoint.main()

    (session,) = _RecordingSelkies.made
    assert session.arguments == {"secret": "the-shared-secret", "signaling_port": 8443,
                                 "bind_host": "10.1.2.3"}
    assert session.config == CONFIG
    assert served["alive"] and served["feeders"] == 1
    assert (served["host"], served["port"]) == ("10.1.2.3", 8000)
    # Stopped after uvicorn returned: the session and its feeding thread.
    assert not session.is_alive()
    assert not _feeders()


def test_main_serves_the_built_in_node_types(monkeypatch):
    import maddening.nodes as nodes
    import uvicorn

    monkeypatch.setattr(selkies_session, "_HAS_GST", False)
    monkeypatch.delenv("MADDENING_CLOUD_CONFIG", raising=False)
    monkeypatch.delenv("MADDENING_STREAM_PRESET", raising=False)
    monkeypatch.setenv("MADDENING_HOST", "127.0.0.1")
    monkeypatch.setattr(entrypoint.signal, "signal", lambda *args: None)
    built: list = []
    real = entrypoint._builtin_node_registry

    def recording_registry():
        registry = real()
        built.append(registry)
        return registry

    monkeypatch.setattr(entrypoint, "_builtin_node_registry", recording_registry)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: None)
    entrypoint.main()
    assert built and set(built[0]) == set(nodes.__all__)
    assert all(built[0][name] is getattr(nodes, name) for name in nodes.__all__)


# ---------------------------------------------------------------------------
# Faults the feed survives, and what the log never holds
# ---------------------------------------------------------------------------


class _FlakyRelay:
    """A relay whose first two reads raise, then return a snapshot."""

    def __init__(self) -> None:
        self.reads = 0

    def latest_frame(self):
        self.reads += 1
        if self.reads <= 2:
            raise RuntimeError("relay read failed")
        return self.reads, 0.5, {"spring": {"x": 1.0}}


class _CountingRenderer:
    def __init__(self) -> None:
        self.updates = 0

    def setup(self, graph_info) -> None:
        pass

    def update(self, sim_time, state) -> None:
        self.updates += 1

    def teardown(self) -> None:
        pass


def test_a_relay_read_that_raises_is_logged_once_and_the_feed_carries_on(caplog):
    server = type("Server", (), {})()
    server.relay = _FlakyRelay()
    renderer = _CountingRenderer()
    stream = entrypoint._Stream(server, renderer, fps=50)
    with caplog.at_level(logging.ERROR, logger="maddening.cloud.entrypoint"):
        stream.start(None)
        deadline = time.monotonic() + 10.0
        while renderer.updates < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        stream.stop()
    assert server.relay.reads > 2 and renderer.updates >= 3, (server.relay.reads, renderer.updates)
    assert caplog.text.count("A streamed frame could not be read, drawn or pushed") == 1, caplog.text


def test_the_logged_config_holds_no_ice_server_credential(monkeypatch, caplog):
    import uvicorn

    turn = {"urls": ["turn:198.51.100.7:3478?transport=udp"],
            "username": "turn-user-not-for-logs", "credential": "turn-credential-not-for-logs"}
    config = {"stream_config": {"width": 320, "height": 240, "ice_servers": [
        {"urls": ["stun:198.51.100.8:3478"]}, turn,
        {"urls": "turn://url-user-not-for-logs:url-pass-not-for-logs@198.51.100.9:3478"}]}}
    monkeypatch.setattr(selkies_session, "_HAS_GST", False)
    monkeypatch.setenv("MADDENING_CLOUD_CONFIG", json.dumps(config))
    monkeypatch.setenv("MADDENING_HOST", "127.0.0.1")
    monkeypatch.setattr(entrypoint.signal, "signal", lambda *args: None)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: None)
    with caplog.at_level(logging.INFO, logger="maddening.cloud.entrypoint"):
        entrypoint.main()
    logged = [r.getMessage() for r in caplog.records if "starting with config" in r.getMessage()]
    assert len(logged) == 1, caplog.text
    assert "198.51.100.7" in logged[0] and "<redacted>" in logged[0], logged[0]
    assert "not-for-logs" not in caplog.text, logged[0]
