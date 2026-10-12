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
# Creating it needs CAP_NET_ADMIN: run the container with
# --cap-drop ALL --cap-add NET_ADMIN.  The container runs in the host's
# initial user namespace, so that capability is not confined to the
# container's own network namespace (docker/README.md says what else it
# reaches), and the command must not keep it.  Once the interface exists,
# the command is started with every capability set cleared (effective,
# permitted, inheritable, ambient) and no_new_privs set, so neither it nor
# anything it executes can hold NET_ADMIN again: under no_new_privs an
# execve by root keeps no capability the caller did not already hold.
# (Removing NET_ADMIN from the bounding set as well would need
# CAP_SETPCAP, which --cap-drop ALL takes away; with no_new_privs the
# bounding set grants nothing.)  tests/cloud/test_webrtc_loopback.py checks
# that the test process holds no capability and cannot change a link.
set -eu
if ! ip link show maddening0 >/dev/null 2>&1; then
  if ! ip link add maddening0 type veth peer name maddening1 2>/dev/null; then
    echo "streaming-test: cannot create the test interface; run the container" \
         "with --cap-drop ALL --cap-add NET_ADMIN (see docker/README.md)" >&2
    exit 2
  fi
  ip addr add 192.0.2.1/24 dev maddening0
  ip link set maddening1 up
  ip link set maddening0 up
fi
# capsh (libcap2-bin): --caps="=" clears the effective, permitted and
# inheritable sets, --noamb the ambient one, --no-new-privs sets the flag;
# "--" then executes the shell given by --shell with the arguments after it.
exec capsh --caps="=" --noamb --no-new-privs --shell=/bin/sh -- \
  -c 'exec "$@"' streaming-test-entrypoint "$@"
