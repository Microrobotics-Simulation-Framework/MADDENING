#!/bin/sh
# Entry point of the streaming test image (docker/Dockerfile.streaming-test).
#
# The tests run with --network none, which leaves the container nothing but
# its loopback interface, and libnice (webrtcbin's ICE agent) never gathers
# ICE candidates on a loopback interface: with nothing else, neither end of
# the WebRTC connection would have a single candidate.  So this gives the
# container one more interface before running the command: a veth pair
# whose two ends both stay inside the container (no link to the host or to
# any network), one of them holding 192.0.2.1, an address from TEST-NET-1
# (RFC 5737, never routed).  Packets between the server and the peer, both
# on 192.0.2.1, never leave the container.  veth, not a dummy interface:
# every host that runs Docker has the veth driver loaded already.
#
# Creating it needs CAP_NET_ADMIN over the container's own network
# namespace: run the container with --cap-add NET_ADMIN.
set -eu
if ! ip link show maddening0 >/dev/null 2>&1; then
  if ! ip link add maddening0 type veth peer name maddening1 2>/dev/null; then
    echo "streaming-test: cannot create the test interface; run the container" \
         "with --cap-add NET_ADMIN (see docker/README.md)" >&2
    exit 2
  fi
  ip addr add 192.0.2.1/24 dev maddening0
  ip link set maddening1 up
  ip link set maddening0 up
fi
exec "$@"
