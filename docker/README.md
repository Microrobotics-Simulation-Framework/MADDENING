# Docker images

## `Dockerfile.cloud`

The cloud container: MADDENING's simulation server with GStreamer, on a
CUDA base image.  The build and run commands are in the file's header.

## `Dockerfile.streaming-test`

A CPU-only image for the WebRTC streaming tests,
`tests/cloud/test_webrtc_loopback.py`.  It holds Ubuntu 24.04's system
Python 3.12 with PyGObject, GStreamer with the plugins `webrtcbin` needs
(base, good, bad, ugly, `gstreamer1.0-nice`) plus `gstreamer1.0-libav`
for the test peer's H.264 decoder, JAX for CPU, and MADDENING installed
editable from the tree.  No GPU is used.

The tests start a `SelkiesSession` and connect a second GStreamer
`webrtcbin` to it, a peer that signals the way the browser client
`src/maddening/api/static/webrtc_client.html` does; it must receive the
answer, complete ICE and decode the frames that were pushed.

From the repository root:

```sh
docker build -f docker/Dockerfile.streaming-test -t maddening-streaming-test .

docker run --rm --network none --cap-add NET_ADMIN \
  --cpuset-cpus=0-3 --memory=8g \
  -v "$PWD":/src:ro maddening-streaming-test \
  python3 -m pytest tests/cloud/test_webrtc_loopback.py -q -p no:cacheprovider
```

The working tree is mounted read-only over the copy baked into the image,
so an edit needs no rebuild; rebuild after a dependency changes.

* `--network none` keeps the container off every network: no STUN or
  TURN server is reached and nothing leaves the container.
* `--cap-add NET_ADMIN` is needed because libnice (`webrtcbin`'s ICE
  agent) never gathers candidates on the loopback interface, which is
  all `--network none` leaves.  The image's entry point
  (`streaming-test-entrypoint.sh`) adds a veth pair whose two ends stay
  inside the container, with the never-routed address 192.0.2.1, for the
  session and the peer to meet on.  The capability covers only the
  container's own network namespace.
* `--cpuset-cpus` and `--memory` are optional limits.

Set `MADDENING_REQUIRE_STREAMING_TESTS=1` (`-e ...`) to make a missing
dependency fail the run instead of skipping it, as the `test-streaming`
job in `.github/workflows/ci.yml` does.  Outside the container the
module skips, since GStreamer and PyGObject are not pip-installable.

What these tests do not cover: a GPU encoder (the encoder is `x264enc`),
a real browser, and NAT traversal.
