# Docker images

## `Dockerfile.cloud`

The cloud container: MADDENING's simulation server with GStreamer, on a
CUDA base image.  The build and run commands are in the file's header.
Its entry point (`python3 -m maddening.cloud.entrypoint`) serves the API
on `MADDENING_PORT` (8000) and streams the simulation over WebRTC: the
signaling socket listens on 8443 of `MADDENING_HOST`, authenticates every
viewer against `MADDENING_STREAM_SECRET` (unset, it refuses everyone), and
the session id it logs at startup is what a viewer derives its token from
(`maddening.cloud.generate_session_token(session_id, secret)`).  A client
builds the graph through the REST graph routes, which take the built-in
node types.

## `Dockerfile.streaming-test`

A CPU-only image for the WebRTC streaming tests,
`tests/cloud/test_webrtc_loopback.py` and
`tests/cloud/test_entrypoint_streams.py`.  It holds Ubuntu 24.04's system
Python 3.12 with PyGObject, GStreamer with the plugins `webrtcbin` needs
(base, good, bad, ugly, `gstreamer1.0-nice`) plus `gstreamer1.0-libav`
for the test peer's H.264 decoder, JAX for CPU, the packages of the
`server` extra (what the cloud image installs for the entry point), and
MADDENING installed editable from the tree.  No GPU is used.

The tests start a `SelkiesSession` and connect a second GStreamer
`webrtcbin` to it, a peer that signals the way the browser client
`src/maddening/api/static/webrtc_client.html` does; it must receive the
answer, complete ICE and decode the frames that were pushed; one test
runs the session in a child process, fed from before the viewer connects.
`test_entrypoint_streams.py` runs the cloud entry point as the container
does (a child process, with the config JSON, the stream secret and the
bind host in its environment), builds and runs a graph through the API's
`/graph/` and `/sim/` routes on loopback, and connects the same peer to
port 8443 with a token from the secret: the frames must have the
configured size, change while the simulation runs and hold while it is
paused; a peer without a valid token gets nothing; SIGTERM ends the
process cleanly.

From the repository root:

```sh
docker build -f docker/Dockerfile.streaming-test -t maddening-streaming-test .

docker run --rm --network none --cap-drop ALL --cap-add NET_ADMIN \
  --cpuset-cpus=0-3 --memory=8g \
  -v "$PWD":/src:ro maddening-streaming-test \
  python3 -m pytest tests/cloud/test_webrtc_loopback.py \
    tests/cloud/test_entrypoint_streams.py -q -p no:cacheprovider
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
  session and the peer to meet on.
* The capability is not confined to the container's own network
  namespace.  The container runs in the host's initial user namespace
  (`/proc/self/uid_map` reads `0 0 4294967295`), so with NET_ADMIN it
  reaches the kernel paths that check `capable(CAP_NET_ADMIN)` rather
  than a capability over its own namespace: nf_tables, and the loading
  of `rtnl-link-*` and `netdev-*` kernel modules through the host's
  `/sbin/modprobe` (a link of a kind the host has not loaded yet).  So it
  is held as briefly as possible: `--cap-drop ALL` leaves NET_ADMIN the
  only capability, the entry point uses it for the veth pair alone (the
  veth driver is loaded wherever Docker runs), and then starts the
  command with every capability set cleared (effective, permitted,
  inheritable, ambient) and `no_new_privs` set, through `capsh`, so
  neither the tests nor anything they execute can hold NET_ADMIN again.
  The bounding set still lists it: removing it from there needs
  `CAP_SETPCAP`, which `--cap-drop ALL` takes away, and under
  `no_new_privs` the bounding set grants nothing.
  `test_webrtc_loopback.py` checks that the test process holds no
  capability and that a change to the test interface is refused.
* `--cpuset-cpus` and `--memory` are optional limits.

**Import order.**  In this image a process that imports numpy, then
loads GStreamer's library (PyGObject loads it at the first call into
`Gst`, such as `Gst.init` or `Gst.is_initialized`), then imports jax
aborts inside jaxlib's MLIR import (SIGSEGV or SIGABRT, three tries of
three); jax first, then GStreamer, runs, pipelines included.
`SelkiesSession` therefore imports jax, when it is installed and not
imported yet, before its first contact with GStreamer (in its
constructor and at the top of `start()`), and
`test_a_session_started_after_numpy_and_before_jax_does_not_abort_the_process`
runs that order; the cloud entry point and the test modules import jax
first anyway.  A program that calls into `Gst` itself must import jax
(or `maddening.core`) before it.

Set `MADDENING_REQUIRE_STREAMING_TESTS=1` (`-e ...`) to make a missing
dependency fail the run instead of skipping it, as the `test-streaming`
job in `.github/workflows/ci.yml` does.  Outside the container the
module skips, since GStreamer and PyGObject are not pip-installable.

The stream is encoded in software (`x264enc`) on every host; 0.4.0 has
no GPU encoder.  The same two test files passed, 39 of 39, in the cloud
image (with the test peer's packages added) on a GPU host on 2026-10-12,
and the entry point streamed with the simulation on CUDA.  What they do
not cover: a provider's network and NAT traversal
(both ends are in one container), and a real browser.
