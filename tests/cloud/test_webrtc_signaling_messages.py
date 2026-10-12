"""The WebRTC signaling messages are routed and converted correctly.

The parts of ``maddening.cloud.selkies_session`` that need no GStreamer:
which messages of an authenticated viewer reach the negotiation and
which the input handler, that an unauthenticated one reaches neither,
which H.264 format of an offer the answer uses, how a viewer's ICE
message is read, and how ``StreamConfig.ice_servers`` becomes
``webrtcbin``'s STUN and TURN settings.  These run on every CI lane; the
end-to-end negotiation (a real peer decoding real frames) is
``tests/cloud/test_webrtc_loopback.py``, which needs GStreamer and runs
in a container.

The signaling tests drive a real ``websockets`` client against a real
server on a loopback port, with a recording negotiator in place of the
GStreamer one.
"""

import asyncio
import json
import logging
import threading

import pytest

websockets = pytest.importorskip(
    "websockets",
    reason="signaling needs the optional 'websockets' dependency "
           "(maddening[streaming] / maddening[api])",
)

from maddening.cloud._auth import generate_session_token  # noqa: E402
from maddening.cloud.selkies_session import (  # noqa: E402
    _h264_payload_type,
    _h264_profile_rank,
    _ice_server_settings,
    _make_signaling_handler,
    _remote_candidate,
)

SESSION_ID = "0123456789ab"
SECRET = "shared-secret"
VALID_TOKEN = generate_session_token(SESSION_ID, SECRET)

#: Long enough that a loopback round trip cannot lose a race, short enough
#: that a hung handler fails the test instead of the suite.
TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# Routing on the signaling socket
# ---------------------------------------------------------------------------


class RecordingNegotiator:
    """Records what the handler tells the negotiation, and can answer."""

    def __init__(self, reply_with: list | None = None) -> None:
        self.events: list[tuple] = []
        self.reply_with = reply_with or []

    def client_connected(self, client) -> None:
        self.events.append(("connected",))
        # Sent from another thread, as webrtcbin's callbacks are.
        replies = list(self.reply_with)
        threading.Thread(target=lambda: [client.send(m) for m in replies]).start()

    def client_message(self, client, message) -> None:
        self.events.append(("message", message))

    def client_disconnected(self, client) -> None:
        self.events.append(("disconnected",))


async def _session(messages, *, token=VALID_TOKEN, negotiator=None, expect_replies=0):
    """Send ``messages`` as one client; return (close code, replies, relayed, negotiator)."""
    relayed: list = []
    negotiator = negotiator if negotiator is not None else RecordingNegotiator()
    handler = _make_signaling_handler(SESSION_ID, SECRET, lambda: relayed.append, negotiator)
    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    url = f"ws://127.0.0.1:{port}/signaling/{SESSION_ID}"
    if token is not None:
        url += f"?token={token}"
    replies: list = []
    try:
        async with websockets.connect(url) as ws:
            try:
                for message in messages:
                    await ws.send(message if isinstance(message, (str, bytes))
                                  else json.dumps(message))
                for _ in range(expect_replies):
                    replies.append(json.loads(await asyncio.wait_for(ws.recv(), TIMEOUT_S)))
                # A message of the input kind last: once it is relayed, every
                # message before it has been handled.
                await ws.send(json.dumps({"type": "sentinel"}))
            except websockets.exceptions.ConnectionClosed:
                pass
            deadline = asyncio.get_running_loop().time() + TIMEOUT_S
            while ws.close_code is None and {"type": "sentinel"} not in relayed:
                if asyncio.get_running_loop().time() > deadline:
                    raise AssertionError("the handler neither relayed nor closed")
                await asyncio.sleep(0.01)
            close_code = ws.close_code
    finally:
        server.close()
        await server.wait_closed()
    if {"type": "sentinel"} in relayed:
        relayed.remove({"type": "sentinel"})
    return close_code, replies, relayed, negotiator


def _run(messages, **kwargs):
    return asyncio.run(asyncio.wait_for(_session(messages, **kwargs), TIMEOUT_S * 2))


OFFER = {"type": "sdp", "sdp": {"type": "offer", "sdp": "v=0\r\n"}}
CANDIDATE = {"type": "ice", "candidate": {"candidate": "candidate:1 1 UDP 1 192.0.2.1 5000 typ host",
                                          "sdpMLineIndex": 0}}


class TestRouting:
    def test_sdp_and_ice_go_to_the_negotiation_and_never_to_the_input_handler(self):
        close_code, _replies, relayed, negotiator = _run([OFFER, CANDIDATE])
        assert close_code is None
        assert relayed == []
        assert negotiator.events[:3] == [("connected",), ("message", OFFER),
                                         ("message", CANDIDATE)]

    def test_sdp_and_ice_reach_no_input_handler_even_without_a_negotiator(self):
        relayed: list = []
        handler = _make_signaling_handler(SESSION_ID, SECRET, lambda: relayed.append)

        async def go():
            server = await websockets.serve(handler, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            try:
                async with websockets.connect(
                        f"ws://127.0.0.1:{port}/s?token={VALID_TOKEN}") as ws:
                    for message in (OFFER, CANDIDATE, {"type": "sentinel"}):
                        await ws.send(json.dumps(message))
                    while not relayed:
                        await asyncio.sleep(0.01)
            finally:
                server.close()
                await server.wait_closed()

        asyncio.run(asyncio.wait_for(go(), TIMEOUT_S))
        assert relayed == [{"type": "sentinel"}]

    @pytest.mark.parametrize("event", [
        {"type": "mousemove", "x": 1, "y": 2},
        {"type": "offer", "sdp": "v=0"},       # not the protocol's "sdp": an input
        {"type": "keydown", "key": "Enter"},
        [1, 2, 3],                             # JSON that is not an object
    ])
    def test_every_other_message_goes_to_the_input_handler(self, event):
        close_code, _replies, relayed, negotiator = _run([event])
        assert close_code is None
        assert relayed == [event]
        assert ("message", event) not in negotiator.events

    def test_binary_frames_and_malformed_json_are_dropped(self):
        close_code, _replies, relayed, negotiator = _run([b"\x00\x01", "{not json"])
        assert close_code is None
        assert relayed == []
        assert [e for e in negotiator.events if e[0] == "message"] == []

    @pytest.mark.parametrize("token", [None, "f" * 64,
                                       generate_session_token("ba9876543210", SECRET),
                                       generate_session_token(SESSION_ID, "not-the-secret")])
    def test_an_unauthenticated_client_reaches_neither_the_negotiation_nor_the_handler(
            self, token):
        negotiator = RecordingNegotiator(reply_with=[{"type": "sdp", "sdp": {}}])
        close_code, replies, relayed, _ = _run([OFFER, CANDIDATE], token=token,
                                                 negotiator=negotiator)
        assert close_code == 1008
        assert replies == [] and relayed == []
        assert negotiator.events == []

    def test_the_negotiators_messages_reach_the_client_in_order(self):
        answer = {"type": "sdp", "sdp": {"type": "answer", "sdp": "v=0\r\n"}}
        ice = [{"type": "ice", "candidate": {"candidate": f"candidate:{i}", "sdpMLineIndex": 0}}
               for i in range(5)]
        negotiator = RecordingNegotiator(reply_with=[answer, *ice])
        _code, replies, _relayed, _ = _run([OFFER], negotiator=negotiator,
                                           expect_replies=1 + len(ice))
        assert replies == [answer, *ice]

    def test_the_negotiation_hears_when_the_client_leaves(self):
        # _run returns after the server has closed and its handlers ended.
        _code, _replies, _relayed, negotiator = _run([OFFER])
        assert negotiator.events[-1] == ("disconnected",)


# ---------------------------------------------------------------------------
# Which H.264 format the answer uses
# ---------------------------------------------------------------------------

#: The video section of an offer as Chrome writes it (abridged): several
#: H.264 formats under dynamic payload types, constrained baseline (42e01f)
#: neither first nor alone, and a packetization-mode=0 variant of it.
CHROME_LIKE = "\r\n".join([
    "v=0", "o=- 1 2 IN IP4 127.0.0.1", "s=-", "t=0 0",
    "m=audio 9 UDP/TLS/RTP/SAVPF 111", "a=mid:0", "a=rtpmap:111 opus/48000/2",
    "m=video 9 UDP/TLS/RTP/SAVPF 96 102 103 106 108 127",
    "a=mid:1",
    "a=rtpmap:96 VP8/90000",
    "a=rtpmap:102 H264/90000",
    "a=fmtp:102 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f",
    "a=rtpmap:103 H264/90000",
    "a=fmtp:103 level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=42e01f",
    "a=rtpmap:106 H264/90000",
    "a=fmtp:106 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f",
    "a=rtpmap:108 H264/90000",
    "a=fmtp:108 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=4d001f",
    "a=rtpmap:127 H264/90000",
    "a=fmtp:127 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=64001f",
    "",
])


class TestH264Format:
    def test_constrained_baseline_with_packetization_mode_1_is_chosen(self):
        assert _h264_payload_type(CHROME_LIKE) == (1, 106)

    def test_without_constrained_baseline_plain_baseline_comes_next(self):
        offer = CHROME_LIKE.replace("a=fmtp:106 level-asymmetry-allowed=1;packetization-mode=1;"
                                    "profile-level-id=42e01f",
                                    "a=fmtp:106 packetization-mode=0;profile-level-id=42e01f")
        assert _h264_payload_type(offer) == (1, 102)

    def test_any_profile_is_taken_when_no_baseline_one_is_offered(self):
        offer = "\r\n".join([
            "v=0", "m=video 9 UDP/TLS/RTP/SAVPF 127 108", "a=rtpmap:127 H264/90000",
            "a=fmtp:127 packetization-mode=1;profile-level-id=64001f",
            "a=rtpmap:108 h264/90000",
            "a=fmtp:108 packetization-mode=1;profile-level-id=4d001f", ""])
        assert _h264_payload_type(offer) == (0, 127)  # the offer's own order

    def test_no_format_with_packetization_mode_1_means_no_answer(self):
        offer = "\r\n".join([
            "v=0", "m=video 9 UDP/TLS/RTP/SAVPF 103", "a=rtpmap:103 H264/90000",
            "a=fmtp:103 packetization-mode=0;profile-level-id=42e01f", ""])
        assert _h264_payload_type(offer) is None

    def test_a_format_the_m_line_does_not_list_is_not_offered(self):
        offer = "\r\n".join([
            "v=0", "m=video 9 UDP/TLS/RTP/SAVPF 96", "a=rtpmap:96 VP8/90000",
            "a=rtpmap:106 H264/90000",
            "a=fmtp:106 packetization-mode=1;profile-level-id=42e01f", ""])
        assert _h264_payload_type(offer) is None

    def test_only_video_sections_are_read(self):
        offer = "\r\n".join([
            "v=0", "m=audio 9 UDP/TLS/RTP/SAVPF 106", "a=rtpmap:106 H264/90000",
            "a=fmtp:106 packetization-mode=1;profile-level-id=42e01f", ""])
        assert _h264_payload_type(offer) is None

    @pytest.mark.parametrize("profile, rank", [
        ("42e01f", 0), ("42c01f", 0), ("42401f", 0),   # constraint_set1: constrained
        ("42001f", 1), ("", 1),                        # baseline; absent = 420010
        ("4d001f", 2), ("64001f", 2), ("zz", 2),
    ])
    def test_profile_rank(self, profile, rank):
        assert _h264_profile_rank(profile) == rank


# ---------------------------------------------------------------------------
# A viewer's ICE message
# ---------------------------------------------------------------------------


class TestRemoteCandidate:
    MIDS = {"video0": 0, "1": 1}
    LINE = "candidate:1 1 UDP 1 192.0.2.1 5000 typ host"

    def test_the_m_line_index_is_used_when_given(self):
        assert _remote_candidate({"candidate": {"candidate": self.LINE, "sdpMLineIndex": 1,
                                                "sdpMid": "video0"}}, self.MIDS) == (1, self.LINE)

    def test_the_mid_is_looked_up_when_the_index_is_absent(self):
        assert _remote_candidate({"candidate": {"candidate": self.LINE, "sdpMid": "video0",
                                                "sdpMLineIndex": None}}, self.MIDS) == (0, self.LINE)

    @pytest.mark.parametrize("message", [
        {"candidate": {"candidate": "", "sdpMLineIndex": 0}},           # end of candidates
        {"candidate": {"candidate": "   ", "sdpMLineIndex": 0}},
        {"candidate": {"candidate": LINE, "sdpMLineIndex": True}},      # a bool is not an index
        {"candidate": {"candidate": LINE, "sdpMLineIndex": -1}},
        {"candidate": {"candidate": LINE, "sdpMid": "unknown"}},
        {"candidate": {"candidate": 5, "sdpMLineIndex": 0}},
        {"candidate": "candidate:1 1 UDP 1 192.0.2.1 5000 typ host"},   # not the browser's shape
        {},
    ])
    def test_a_message_without_a_usable_candidate_is_skipped(self, message):
        assert _remote_candidate(message, self.MIDS) is None


# ---------------------------------------------------------------------------
# StreamConfig.ice_servers -> webrtcbin
# ---------------------------------------------------------------------------


class TestIceServerSettings:
    def test_the_default_stream_config_names_one_stun_server(self):
        from maddening.cloud.streaming import StreamConfig

        assert _ice_server_settings(StreamConfig().ice_servers) == (
            "stun://stun.l.google.com:19302", [])

    def test_no_servers_means_none_are_contacted(self):
        assert _ice_server_settings([]) == (None, [])

    def test_stun_and_turn_in_the_browsers_shape(self):
        stun, turn = _ice_server_settings([
            {"urls": "stun:stun.example.org:3478"},
            {"urls": ["turn:turn.example.org:3478?transport=udp", "turns:turn.example.org:5349"],
             "username": "viewer", "credential": "s3cret"},
        ])
        assert stun == "stun://stun.example.org:3478"
        assert turn == ["turn://viewer:s3cret@turn.example.org:3478?transport=udp",
                        "turns://viewer:s3cret@turn.example.org:5349"]

    def test_turn_credentials_are_percent_encoded(self):
        # webrtcbin unescapes the user and password it finds in the URI, so
        # a credential holding ':', '@', '/' or a space survives the trip.
        _stun, turn = _ice_server_settings([
            {"urls": "turn:turn.example.org:3478", "username": "a@b", "credential": "p:w /x"}])
        assert turn == ["turn://a%40b:p%3Aw%20%2Fx@turn.example.org:3478"]

    def test_only_the_first_stun_server_is_used_and_the_rest_are_reported(self, caplog):
        with caplog.at_level(logging.WARNING, "maddening.cloud.selkies_session"):
            stun, _turn = _ice_server_settings([
                {"urls": ["stun:one.example.org:3478", "stun:two.example.org:3478"]}])
        assert stun == "stun://one.example.org:3478"
        assert "two.example.org" in caplog.text

    @pytest.mark.parametrize("entry", [
        {"urls": "turn:turn.example.org:3478"},                      # no credentials
        {"urls": "turn:turn.example.org:3478", "username": "u"},
        {"urls": "http://example.org/"},                             # not ICE
        {"urls": "stun:"},                                           # no host
        "stun:stun.example.org:3478",                                # not a dict
    ])
    def test_what_webrtcbin_cannot_use_is_skipped_with_a_warning(self, entry, caplog):
        with caplog.at_level(logging.WARNING, "maddening.cloud.selkies_session"):
            assert _ice_server_settings([entry]) == (None, [])
        assert "Ignoring" in caplog.text
