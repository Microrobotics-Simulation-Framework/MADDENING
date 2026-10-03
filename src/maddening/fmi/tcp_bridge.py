"""TCP bridge between the FMU C wrapper and :class:`FmuSidecar`.

The compiled FMU (``src/maddening/fmi/c/maddening_fmu.c``) is loaded
into the importer's process and speaks a tiny protocol that needs only
libc on the C side: every message is a 4-byte big-endian length prefix
followed by one frame.  This module is the Python end of that protocol:
it maps FMI value references onto the sidecar's inputs, outputs and
parameters using the :class:`ModelDescription` the FMU was built from,
and runs the master-step loop.

Requests (importer -> sidecar) and responses, JSON form::

    {"op": "hello"}                       -> {"ok": true, "token": ..., "model": ...,
                                              "master_dt": h, "protocol": 2, "binary": b}
    {"op": "set", "type": T, "vr": [..], "values": [..]}
                                          -> {"ok": true}
    {"op": "get", "type": T, "vr": [..]}  -> {"ok": true, "values": [..]}
    {"op": "initialize", "t": t0}         -> {"ok": true, "t": t0}
    {"op": "step", "t": t, "dt": h}       -> {"ok": true, "t": t + h}
    {"op": "get_state"}                   -> {"ok": true, "state": "<base64>"}
    {"op": "set_state", "state": ".."}    -> {"ok": true, "t": restored time}
    {"op": "reset"}                       -> {"ok": true}
    {"op": "terminate"}                   -> {"ok": true}   (Terminated until reset)
    any failure                           -> {"ok": false, "error": "..."}

**Instances.**  A connection that claims the bridge's instance slot (its
first complete frame) is a new FMU instance, and starts where FMI starts
one: the state the bridge was built over, every parameter at the
``start`` its model description advertises, every input at zero and the
time at zero.  ``reset`` (``fmi3Reset``) returns to the same point.
``initialize`` (``fmi3EnterInitializationMode``) sets the start time; it
is accepted until the instance's first step.  A ``step``'s ``t`` must be
the FMU's current time -- the previous ``t`` plus the previous ``h``, or
the start time -- to within a millionth of a master step
(``_COMM_POINT_TOLERANCE``); a point inside it is adopted, one outside it
is refused with nothing advanced.  The step size ``h`` is held to the same
absolute tolerance off a whole number of master steps, so a step the bridge
accepts never makes the next legal point fail.  And the point, and the
step's end ``t + h``, must stay within that tolerance of the time the FMU
has simulated -- the start time plus the master steps taken since -- plus a
few ulps of the time per step, the rounding an importer's running sum of
step sizes gathers (``_DRIFT_ULPS_PER_STEP``): an importer whose every step
size is a little long is refused once its errors add up past the tolerance,
where the reported time used to move ahead of the physics without bound.
Every tolerance is relative to the master step, at any master step, and no
rounding slack is ever more than a tenth of one (``_ROUNDING_SLACK_MAX``):
the reported time stays within a millionth plus a tenth of a master step of
the simulated time.  A time at which 16 ulps exceed that tenth -- 32 s and
beyond at a 1e-12 s master step, about 3.4e10 s at 1e-3 s -- is refused at
``initialize``, at a step that would reach it and at ``set_state``: the
float64 clock cannot place a communication point there (the uncapped slacks
used to admit whole master steps of drift).  ``t``,
``h`` and every entry of ``values`` must be JSON numbers: a string or a
boolean is refused, not parsed.

**Types.**  ``type`` (optional) is the FMI 3.0 type of the ``fmi3Get`` /
``fmi3Set`` function the request comes from (``"Float32"``, ``"Boolean"``,
...); the C wrapper always sends it, and a variable of another type is
refused with nothing read or written, as FMI 3.0 requires.  A Boolean
variable takes ``0`` or ``1``.  A ``set`` that names a value reference twice
is refused, and so is a numeric set of a ``<Clock>`` variable; a ``get`` may
name one twice, and every occurrence gets the value.

**Steps.**  ``master_dt`` is one step of the graph and must equal the step
the model description records (``ModelDescription.graph_timestep``); one
``step`` request runs at most ``max_steps_per_request`` graph steps, and
stops at the first one after :meth:`FmuTcpBridge.stop`.

``values`` are flat numbers in value-reference order; an array variable
contributes ``prod(shape)`` entries in row-major order.  A **non-finite**
value is written as the quoted token ``"NaN"``, ``"Infinity"`` or
``"-Infinity"`` rather than the bare token ``json.dumps`` would write,
which is not JSON (``MADD-ANO-006``); a ``get`` reply from a diverged
model is therefore a frame any conforming parser reads.  Both spellings
are accepted on the way in, and the C wrapper reads either.  A ``set``
carrying a non-finite value is refused by the sidecar whichever spelling
it arrives in.  Inputs are held
until the next ``step``; a communication step ``h`` must be a whole
multiple of ``master_dt`` (it runs ``h / master_dt`` graph steps; anything
else is refused, and the FMU advertises a fixed communication step).  A
``set`` is atomic: parameters are bounds-checked by the sidecar and inputs
are committed only when every value in the request was valid.  The bounds
are the ``min`` / ``max`` the model description advertises, whether or not
the sidecar was built with ``param_specs`` (a sidecar spec is enforced
beside them, and one that enforces another envelope -- or a description
whose graph's specs changed since it was built -- makes the bridge refuse
to start), and a declared external input
the description does not export (``held_inputs``) is held at zero on every
step, as ``GraphManager.step`` holds an input its caller omits.

**Binary frames (protocol 2).**  Bit 31 of the length prefix marks a
*binary* frame; the low 31 bits are the payload length.  A binary
payload is ``[u32 BE header_len][header JSON][raw bytes]``: the header
carries ``op`` and metadata, the raw part carries the data, so bulk
values and state blobs never pass through JSON text::

    set:        {"op":"set","vr":[..],"n":N,"dtype":"f64"}   raw = N little-endian float64
    get reply:  {"ok":true,"n":N,"dtype":"f64"}              raw = N little-endian float64
    set_state:  {"op":"set_state","n":L}                     raw = L bytes of npz
    get_state reply: {"ok":true,"n":L}                       raw = L bytes of npz

A client opts in with ``{"op": "hello", "protocol": 2, "binary": true}``;
only then does the bridge answer ``get`` / ``get_state`` with binary
frames and accept binary ``set`` / ``set_state`` requests.  Every other
op, every error reply and every JSON-only client is unchanged: a client
that sends ``{"op": "hello"}`` gets the protocol-1 behaviour (the hello
reply merely gains ``protocol`` and ``binary``; ``recv_message`` /
``send_message`` here speak both forms).  A client announcing a protocol
this bridge does not know is refused at hello.

**Frame limit.**  A frame is at most 64 MiB (``_MAX_MESSAGE``) in both
directions: a longer request drops the connection (its length is not to
be trusted), and a reply that would be longer (a huge state, a ``get``
whose JSON text passes the limit) is replaced by a JSON error reply, so the
connection stays in sync and the C wrapper, which refuses to read a
longer frame, never sees one from this bridge.  A ``get`` naming more
value references, or variables holding more values, than a binary reply
frame carries (``_MAX_GET_VALUES``, 8 388 480) is refused before anything
is read: one 64 MiB request naming a scalar 33.5 million times used to
take about 45 s and 6 GB before its reply was refused.

**Connection lifetime.**  A connection holds the bridge's single FMU
instance for as long as it lives, so no wait on it is unbounded: a peer
has ten seconds to begin its first frame, five minutes of silence
between frames once it has spoken (``FmuTcpBridge(idle_timeout=...)``;
``None`` lifts this one), and two minutes to finish a frame it
has announced the length of -- two minutes in total, whether the rest
of the frame dribbles in or stops arriving.  Overrunning any of them
ends the connection exactly as EOF does, and the instance slot is free
again.  The number of live connection threads is capped (16); further
connections are closed on accept.  ``stop()`` shuts every live
connection down, so a parked worker does not outlive the bridge, and
returns within about five seconds.  A worker still *inside a request*
then (a first step that is compiling a large graph, say) cannot be
interrupted: ``stop()`` logs a warning naming it, and the request
commits nothing -- no request that is still running when ``stop()``
begins, and none after it, changes the model's state, parameters,
inputs or time.  A bridge serves once: ``start()`` a second time, or
after ``stop()``, raises ``RuntimeError`` (build a new bridge to serve
again).

The importer is **untrusted**: nothing that arrives on the socket is ever
unpickled or evaluated.  The FMU-state blob is an ``npz`` archive of plain
arrays (``allow_pickle=False`` on load) carrying the schema token, the
time, the pending inputs, the node states and the params; on
``set_state`` the archive directory is checked first (only the expected
member names, each member's declared size capped by the live array it
replaces, and a cap on the total), so nothing is decompressed that the
model could not hold, and then every array is checked against the live
one (token, key set -- state fields, parameters and pending inputs --
shape, value) before anything is written.  Bind the bridge
to ``127.0.0.1`` unless the network is trusted.

ZMQ is not required.  A ZMQ transport with the same frame payloads can be
added later without touching the C wrapper's request format.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import socket
import struct
import threading
import time
import warnings
import zipfile
from typing import Any, Iterator, Optional, cast

import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.params import ParamSpec
from maddening.fmi.model_description import (
    _DTYPE_TO_FMI_TYPE,
    FMIVariable,
    ModelDescription,
    _advertised_bound,
    _envelope,
    _fmi_kind,
    _graph_changed_since,
    _parse_xs_value,
    _specs_changed_since,
)
from maddening.fmi.sidecar import (
    FmuSidecar,
    _checked_value,
    _declared_inputs_resolver,
    _key_set_error,
    _restored_leaf,
    _step_compile,
)
from maddening.serialization.json_codec import decode_non_finite
from maddening.serialization.json_codec import dumps as _json_dumps

logger = logging.getLogger(__name__)

_HEADER = struct.Struct(">I")
_MAX_MESSAGE = 64 * 1024 * 1024
"""Frame limit in bytes, both directions and both frame kinds (the C wrapper's FRAME_MAX)."""
_MAX_GET_VALUES = (_MAX_MESSAGE - 1024) // 8
"""The most values one ``get`` may ask for: what a binary reply frame holds
as raw float64 beside its header (8 388 480).  A larger ``get`` is refused
before anything is read (``FmuTcpBridge._get``); its reply could only
exceed the frame limit.  A JSON reply takes four to twenty-five bytes a
value, so one near this count may still exceed the limit, and is answered
with an error once its encoding passes it."""
_JSON_VALUES_CHUNK = 65536
"""Values encoded at a time for a ``get``'s JSON reply, which is abandoned
as soon as it passes the frame limit (``FmuTcpBridge._json_values_body``)."""
_BINARY_FLAG = 0x80000000
_LENGTH_MASK = 0x7FFFFFFF
_NPY_SLACK = 4096
"""Bytes an ``npz`` member may exceed its array by (the ``.npy`` header)."""
PROTOCOL_VERSION = 2
"""Highest sidecar protocol this bridge speaks (1 = JSON only, 2 = + binary frames)."""

_UNENCODABLE_PREFIX = "the bridge could not encode its reply: "
"""Prefix of the last-resort error reply (:meth:`FmuTcpBridge._unencodable_reply`).

A prefix, not a bare message, for two reasons: it can never itself equal
one of the three non-finite tokens -- so the reply that reports an
unencodable reply cannot be unencodable in turn -- and it says the
failure was on the bridge's side of the wire rather than in the
request."""

_ERROR_TEXT_MAX = 400
"""Characters of an exception's text carried in an error reply.

Long enough for the codec's path-naming message, short enough that the
reply cannot approach the frame limit however the exception was
formatted."""

# ---------------------------------------------------------------- timeouts
# A connection holds the bridge's single instance slot (``_busy``) for as
# long as it lives, so every wait on it is bounded.  The three budgets are
# separate because a legitimate importer's silences are of three different
# lengths, and one number generous enough for the longest would leave the
# instance slot parkable by a peer that says nothing at all.
_HANDSHAKE_TIMEOUT = 10.0
"""Seconds a freshly accepted connection has to start its first frame.

The C wrapper sends ``hello`` immediately after ``connect`` (see
``bridge_connect`` in ``c/maddening_fmu.c``), so ten seconds is already
three orders of magnitude more than a healthy importer needs, while a
port scan, a crashed importer or a dropped link is dropped promptly
instead of owning the instance for ever."""
_IDLE_TIMEOUT = 300.0
"""Default seconds an established connection may stay silent between frames
(``FmuTcpBridge(idle_timeout=...)``).

An importer is idle between ``doStep`` calls, and the master may be
waiting on a slow co-simulation partner or on a human at a debugger
prompt, so this one is deliberately generous: five minutes of silence
from a client that has already completed a handshake is a link that is
gone, not a slow one.  A master that legitimately pauses longer -- a
debugging session, a partner model that computes for an hour -- passes a
larger value, or ``None`` for no limit; the connection, and with it the
instance, used to be dropped after five minutes whatever the caller
needed."""
class _ModuleDefault:
    """The default of an argument that falls back to a module constant read
    when the bridge is built (so a test that shrinks the constant shrinks
    the default)."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


_IDLE_DEFAULT = _ModuleDefault("_IDLE_TIMEOUT")
_FRAME_TIMEOUT = 120.0
"""Seconds to finish a frame once its length prefix has arrived.

One deadline over the whole body, enforced inside each ``recv`` as well
as between them: every ``recv`` waits at most for what is left of it
(:func:`_recv_exact`).  So it bounds both the peer that dribbles -- one
byte every nine seconds would renew a plain socket timeout for ever --
and the peer that announces a frame, sends part of it and goes silent,
which would otherwise hold the instance for the whole idle budget.  Two
minutes still covers a full 64 MiB frame on a link of about 5 Mbit/s.
The four-byte length prefix itself is read under the handshake or idle
budget, whichever the connection is in."""
_MAX_CONNECTIONS = 16
"""Live connection threads allowed at once.

The bridge serves one FMU instance, so every connection beyond the first
is refused anyway; the cap exists so that refusing them costs a bounded
number of threads."""
_HANDOVER_GRACE = 2.0
"""Seconds a new connection waits for the instance slot before it is refused.

The slot is released by the *departing* connection's own worker thread,
in the ``finally`` that runs once its socket has reached EOF.  An
importer that frees one instance and immediately instantiates another --
which is exactly ``fmi3FreeInstance`` followed by
``fmi3InstantiateCoSimulation`` -- therefore races that thread, and is
refused a slot nobody holds whenever the scheduler reaches the new
worker first.  Nothing on the server side can close that gap: the new
connection can always arrive before the old worker is scheduled, so the
wait has to be on the acquiring side.

Measured with the two workers contending for one CPU, which is the
shape of a CI runner: 21-28% of immediate reconnects were refused, and
the gap between hang-up and a slot that could be claimed had a median of
0.18s and a maximum of 0.97s.  Those numbers are an upper bound taken on
a deliberately saturated box and should be read as one.  Two seconds is
about four times the measured p95, well under the five-second worker
join in :meth:`FmuTcpBridge.stop`, and still "refused, not blocked": a
peer that really is a second instance gets its error reply, two seconds
later, instead of hanging on a lock for ever."""
_HANDOVER_POLL = 0.05
"""Granularity of that wait, so ``stop()`` is not held up by the grace."""
_COMM_POINT_TOLERANCE = 1e-6
"""The bridge's one tolerance on time, as a fraction of the master timestep:
how far a ``step``'s communication point may sit from the FMU's own time,
and how far its step size may sit from a whole number of master steps
(each plus a few ulps of the times involved, for rounding).

The two used to differ -- the step size was allowed a millionth of a
master step *per master step it covered*, the communication point a
millionth of one -- so a step of ``3 * master_dt * (1 + 5e-7)`` was
accepted, the bridge's clock advanced ``3 * master_dt``, and the next
``doStep`` at ``t + h``, the point FMI requires, was refused.  With one
absolute tolerance, every step the bridge accepts leaves the next legal
communication point inside it: the ulp slack on the step (4 ulps) is
smaller than the communication point's (16), which covers the rounding of
both sums.

FMI 3.0 has each ``fmi3DoStep`` start where the previous one ended -- the
previous communication point plus the previous step size -- or, first, at
the start time ``fmi3EnterInitializationMode`` was given.  An importer
computes that point in its own floating point (FMPy as ``start + k * h``,
others as a running sum), so it agrees with the bridge's clock to rounding,
never exactly; a millionth of a master step is many orders of magnitude
above that rounding for any run shorter than about 10^9 master steps, and
far below any real discontinuity, which the physics could not honour anyway
(it advances whole master steps).  A point inside the tolerance is
*adopted* -- the step ends at ``t + n * master_dt`` on the importer's clock,
so the two clocks never drift apart -- and one outside it is refused with
nothing advanced.  Adoption alone let the importer's clock, and the time the
FMU reports, drift from the time it has simulated by up to the tolerance per
step, so both are also held to the simulated time
(``FmuTcpBridge._check_drift``, :data:`_DRIFT_ULPS_PER_STEP`)."""
_STEP_SIZE_ULPS = 4
"""Rounding slack, in ulps of the larger of ``h`` and ``n * master_dt``, on
a step size's distance from a whole number of master steps."""
_COMM_POINT_ULPS = 16
"""Rounding slack, in ulps of the largest time involved, on a communication
point's distance from the FMU's time.  Larger than :data:`_STEP_SIZE_ULPS`
so that the rounding of ``t + h`` and of ``t + n * master_dt`` cannot carry
an accepted step's error outside it."""
_DRIFT_ULPS_PER_STEP = 4
"""Rounding slack on the drift between the importer's clock and the
simulated time, in ulps of the times involved *per step since the reference
point*: an importer that keeps a running sum of its step sizes rounds once
per step (half an ulp of the time), and may hold the step size itself an
ulp or so off the master step.  A clock biased by more than that per step
gathers drift faster than the slack grows, and is refused once the drift
passes :data:`_COMM_POINT_TOLERANCE` of a master step plus the slack."""
_ROUNDING_SLACK_MAX = 0.1
"""The most any rounding slack may add to a time tolerance, as a fraction of
the master step.

Each slack above is in ulps of the times involved, which is the importer's
rounding at ordinary times; at a large ratio of time to master step an ulp
is itself a sizeable part of a step, and uncapped they admitted whole
master steps (16 ulps of 1000 s are 1.8 steps of 1e-12 s, and the drift
slack grew by 0.45 of a step per step).  So the reported time never sits
more than this (plus :data:`_COMM_POINT_TOLERANCE`) of a master step from
the time the FMU has simulated: less than half a step, so it always names
the master step the physics is at.  And a time so large that the
communication point's own slack (:data:`_COMM_POINT_ULPS` ulps) would pass
it is refused outright (``FmuTcpBridge._check_time_resolution``): the
clock cannot place a communication point there, whatever the importer
does."""
_MASTER_DT_RTOL = 1e-9
"""How far ``master_dt`` may sit from the graph step the model description
records, relatively: a different spelling of the same step (``0.01``
against a GCD that came out as ``0.009999999999999998``), never a different
step."""
MAX_STEPS_PER_REQUEST = 100_000
"""Default cap on the graph steps one ``step`` request may ask for.

A request names its work as ``h / master_dt``, and the bridge used to run
whatever that came to: ``dt = 1e9 * master_dt`` held the connection's
worker -- and the FMU's only instance -- for as long as a billion graph
steps take, with no way to stop it.  A hundred thousand steps is a
communication step far coarser than any coupling interval a master uses
(the default ``stepSize`` is one), and a few seconds of dispatch on a
small graph; it is per request, so a longer run is several ``doStep``
calls.  A bridge that must take larger steps raises it with
``FmuTcpBridge(max_steps_per_request=...)``.  A step's graph steps also
stop at the first one after :meth:`FmuTcpBridge.stop`, committing nothing.
The C wrapper's reply deadline (``MADDENING_FMU_TIMEOUT``, ten minutes by
default) must cover the longest step the bridge accepts."""
_STOP_JOIN_TIMEOUT = 5.0
"""Seconds :meth:`FmuTcpBridge.stop` waits, in total, for its threads.

One budget over the serve thread and every connection worker together,
so ``stop()`` is bounded however many connections were live.  A worker
parked on a read ends at once (``stop()`` shuts its socket down); one
that is *inside a request* -- the first step of a large graph spends
its first call compiling -- cannot be interrupted, and may outlive the
wait.  ``stop()`` then logs a warning naming it, and the request it is
serving is refused at its commit: nothing a worker computes after
``stop()`` is written into the model (see ``FmuTcpBridge._committing``)."""


def _json_object(body: bytes, what: str) -> Any:
    """``json.loads`` of ``body``; every failure is a ``ValueError``.

    The importer's bytes may be anything: not UTF-8, not JSON, or nested
    so deeply that the JSON scanner raises ``RecursionError``.  All of
    those must come out as the one exception the framing contract
    promises, so the connection loop answers with an error reply instead
    of dying with a traceback.
    """
    try:
        # ``json.loads`` accepts the bare ``NaN`` / ``Infinity`` tokens as
        # well as the quoted ones this module writes, so a peer of either
        # vintage is understood; ``decode_non_finite`` turns the quoted
        # form into the float the bare form already produced.
        return decode_non_finite(json.loads(body.decode("utf-8")))
    except RecursionError as exc:
        raise ValueError(f"{what} is nested too deeply") from exc
    except ValueError as exc:                  # JSONDecodeError, UnicodeDecodeError
        raise ValueError(f"{what} is not JSON: {exc}") from exc


def _recv_exact(conn: socket.socket, n: int,
                deadline: Optional[float] = None) -> Optional[bytes]:
    """``n`` bytes, ``None`` at EOF.

    ``deadline`` (a :func:`time.monotonic` value) bounds the whole read,
    not each ``recv``: a peer that dribbles one byte at a time renews the
    socket's own timeout indefinitely, and the connection it is dribbling
    on holds the bridge's only instance slot.  It is also enforced
    *during* each ``recv``, whose socket timeout is lowered to what is
    left of the deadline (and restored afterwards): checked only between
    calls, a peer that sends part of a frame and then goes silent would
    be released by the socket's own, longer timeout instead.  Overrunning
    it raises :exc:`socket.timeout`, which every caller already treats as
    a dead connection.
    """
    buf = bytearray()
    if deadline is None:
        while len(buf) < n:
            chunk = conn.recv(min(n - len(buf), 1 << 20))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)
    saved = conn.gettimeout()
    try:
        while len(buf) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise socket.timeout(
                    f"frame of {n} bytes was still incomplete after "
                    f"{len(buf)} bytes"
                )
            frame_bound = saved is None or remaining < saved
            conn.settimeout(remaining if frame_bound else saved)
            try:
                chunk = conn.recv(min(n - len(buf), 1 << 20))
            except socket.timeout:
                if frame_bound:
                    raise socket.timeout(
                        f"frame of {n} bytes was still incomplete after "
                        f"{len(buf)} bytes"
                    ) from None
                raise
            if not chunk:
                return None
            buf.extend(chunk)
    finally:
        try:
            conn.settimeout(saved)
        except OSError:
            pass                    # the socket was closed under us
    return bytes(buf)


def recv_raw(conn: socket.socket, *,
             frame_timeout: Optional[float] = None) -> Optional[tuple[bool, bytes]]:
    """One length-prefixed frame as ``(is_binary, payload)``.

    ``None`` at EOF; ``ValueError`` when the (31-bit) length exceeds the
    64 MiB limit, binary flag or not.  ``frame_timeout`` bounds the body
    once the length prefix has arrived (:exc:`socket.timeout` on
    overrun); the wait for the prefix itself is the socket's own timeout,
    which the caller sets according to what the connection is waiting
    for.
    """
    head = _recv_exact(conn, _HEADER.size)
    if head is None:
        return None
    (word,) = _HEADER.unpack(head)
    n = word & _LENGTH_MASK
    if n > _MAX_MESSAGE:
        raise ValueError(f"message of {n} bytes exceeds the {_MAX_MESSAGE}-byte limit")
    deadline = None if frame_timeout is None else time.monotonic() + frame_timeout
    body = _recv_exact(conn, n, deadline)
    if body is None:
        return None
    return bool(word & _BINARY_FLAG), body


def recv_frame(conn: socket.socket, *,
               frame_timeout: Optional[float] = None) -> Optional[bytes]:
    """One length-prefixed frame's payload (``None`` at EOF; ``ValueError``
    over the limit).  Use :func:`recv_raw` to learn whether it was binary."""
    got = recv_raw(conn, frame_timeout=frame_timeout)
    return None if got is None else got[1]


def encode_binary(header: dict, raw: bytes) -> bytes:
    """Payload of a binary frame: ``[u32 BE header_len][header JSON][raw]``."""
    hdr = _json_dumps(header, separators=(",", ":")).encode("utf-8")
    return _HEADER.pack(len(hdr)) + hdr + raw


def decode_binary(payload: bytes) -> tuple[dict, bytes]:
    """Split a binary payload into ``(header, raw)``; ``ValueError`` if malformed."""
    if len(payload) < _HEADER.size:
        raise ValueError("binary frame shorter than its header length field")
    (hlen,) = _HEADER.unpack_from(payload)
    if hlen > len(payload) - _HEADER.size:
        raise ValueError(f"binary header of {hlen} bytes exceeds the {len(payload)}-byte payload")
    header = _json_object(payload[_HEADER.size:_HEADER.size + hlen], "binary header")
    if not isinstance(header, dict):
        raise ValueError("binary header must be a JSON object")
    return header, payload[_HEADER.size + hlen:]


def recv_message(conn: socket.socket, *,
                 frame_timeout: Optional[float] = None) -> Optional[dict]:
    """One decoded message.  A JSON frame is its object; a binary frame is
    its header with the raw payload under ``"raw"`` (``bytes``), a dict
    :meth:`FmuTcpBridge.handle` accepts as is.  ``ValueError`` on a
    malformed frame of either kind."""
    got = recv_raw(conn, frame_timeout=frame_timeout)
    if got is None:
        return None
    is_binary, body = got
    if is_binary:
        header, raw = decode_binary(body)
        header["raw"] = raw
        return header
    return _json_object(body, "message")


def send_message(conn: socket.socket, payload: dict) -> None:
    """Send one JSON frame.

    A non-finite number in *payload* goes out as its quoted token, so the
    frame is valid JSON whatever the peer parses it with
    (``MADD-ANO-006``); :func:`_json_object` decodes it at the far end
    and the FMU's C wrapper reads it with ``strtod`` past the quote.
    """
    body = _json_dumps(payload, separators=(",", ":")).encode("utf-8")
    conn.sendall(_HEADER.pack(len(body)) + body)


def send_binary(conn: socket.socket, header: dict, raw: bytes) -> None:
    """Send one binary-flagged frame (``header`` JSON + ``raw`` bytes)."""
    body = encode_binary(header, raw)
    if len(body) > _LENGTH_MASK:
        raise ValueError("binary frame exceeds the 31-bit length field")
    conn.sendall(_HEADER.pack(_BINARY_FLAG | len(body)) + body)


def values_of(reply: dict) -> np.ndarray:
    """The ``values`` of a ``get`` reply as float64, whichever form it took.

    A JSON reply's non-finite entries may be quoted tokens (what this
    module writes since 0.4.0) or already floats (a bare token, which
    ``json.loads`` parses itself); :func:`decode_non_finite` makes both
    the same float before the array is built.
    """
    if "raw" in reply:
        return np.frombuffer(reply["raw"], dtype="<f8").astype(np.float64)
    return np.asarray(decode_non_finite(reply["values"]), dtype=np.float64)


def state_of(reply: dict) -> bytes:
    """The npz bytes of a ``get_state`` reply, whichever form it took."""
    if "raw" in reply:
        return reply["raw"]
    return base64.b64decode(reply["state"])


def _copy_tree(tree):
    """Deep copy of a nested dict of arrays without pickle."""
    if tree is None:
        return None
    if isinstance(tree, dict):
        return {k: _copy_tree(v) for k, v in tree.items()}
    return tree if hasattr(tree, "dtype") else np.asarray(tree)


def _value_reference(vr: Any) -> int:
    """``vr`` as a value reference, refused unless it is an integer.

    FMI value references are integers, and nothing on the wire may be
    coerced into one: ``int()`` truncates ``10.9`` and ``10.4`` to 10,
    parses ``"10"`` and turns ``true`` into 1, so each of those used to
    address a variable the importer never named -- a ``set`` wrote it, a
    ``get`` read it back, and the reply said ``ok``.  A Python ``int``
    (what JSON decodes an integer to) or a NumPy integer is accepted;
    ``bool`` is not, although Python counts it as an ``int``.

    Raises
    ------
    ValueError
        If ``vr`` is not an integer; the dispatcher answers it with the
        usual error reply and nothing is read or written.
    """
    if isinstance(vr, (bool, np.bool_)) or not isinstance(vr, (int, np.integer)):
        raise ValueError(f"value reference must be an integer, got {vr!r}")
    return int(vr)


def _real_number(x: Any, what: str) -> float:
    """``x`` as a finite float, refused unless it already is a number.

    The rule :func:`_value_reference` applies to a value reference, for the
    times a request carries: ``float()`` parses ``"0.01"`` and turns
    ``true`` into 1.0, so a ``step`` whose ``dt`` was the JSON boolean
    ``true`` advanced a hundred master steps of 0.01 s and answered ``ok``.
    A JSON integer or float (or a NumPy one) is accepted; a boolean, a
    string, ``null`` or a container is not.

    Raises
    ------
    ValueError
        If ``x`` is not a real number, or is not finite.
    """
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer,
                                                               np.floating)):
        raise ValueError(f"{what} must be a number, got {x!r}")
    try:
        v = float(x)
    except OverflowError as exc:                    # a JSON integer of 400 digits
        raise ValueError(f"{what} must be finite, got {x!r}") from exc
    if not np.isfinite(v):
        raise ValueError(f"{what} must be finite, got {x!r}")
    return v


def _flat_numbers(values: Any) -> np.ndarray:
    """A ``set`` request's ``values`` as float64, refused unless every entry
    already is a number.

    ``np.asarray(values, dtype=float64)`` parses the string ``"45"`` and
    turns ``true`` into 1.0, so both used to be stored as numbers with the
    reply ``ok``.  A flat list (or tuple) of JSON integers and floats is
    accepted -- a non-finite one too, as a float, for the finiteness check
    downstream to refuse by variable name -- and so is a float or integer
    array (the binary path's, or an in-process caller's).

    Raises
    ------
    ValueError
        If ``values`` is not a flat sequence of numbers.
    """
    if isinstance(values, np.ndarray):
        if values.dtype.kind not in "iuf":
            raise ValueError(f"values must be numbers, got an array of {values.dtype}")
        arr = values.astype(np.float64)
    elif isinstance(values, (list, tuple)):
        for i, x in enumerate(values):
            if isinstance(x, (bool, np.bool_)) or not isinstance(
                    x, (int, float, np.integer, np.floating)):
                raise ValueError(f"values must be a flat list of numbers; entry {i} is {x!r}")
        try:
            arr = np.asarray(values, dtype=np.float64)
        except OverflowError as exc:
            raise ValueError(f"values must be finite numbers: {exc}") from exc
    else:
        raise ValueError(f"values must be a flat list of numbers, got {type(values).__name__}")
    if arr.ndim != 1:
        raise ValueError("values must be a flat list of numbers")
    return arr


def _size(var: FMIVariable) -> int:
    return int(np.prod(var.shape)) if var.shape else 1


_CLOCK_TYPE = "Clock"
_FMI_TYPES = frozenset(_DTYPE_TO_FMI_TYPE.values()) | {_CLOCK_TYPE}
"""The FMI 3.0 variable types a request's ``type`` may name."""


def _fmi_type_of(var: FMIVariable) -> str:
    """The FMI 3.0 type of ``var`` as its ``modelDescription.xml`` declares it."""
    return _CLOCK_TYPE if var.is_clock else _DTYPE_TO_FMI_TYPE[var.dtype]


def _requested_type(fmi_type: Any) -> Optional[str]:
    """A request's ``type`` (the ``{VariableType}`` of the ``fmi3Get`` /
    ``fmi3Set`` call it carries), ``None`` when it names none."""
    if fmi_type is None:
        return None
    if not isinstance(fmi_type, str) or fmi_type not in _FMI_TYPES:
        raise ValueError(f"type must be one of {sorted(_FMI_TYPES)}, got {fmi_type!r}")
    return fmi_type


def _check_access_type(var: FMIVariable, fmi_type: Optional[str], access: str) -> None:
    """Refuse an ``fmi3{access}{fmi_type}`` call on a variable of another type.

    FMI 3.0 ("Getting and Setting Variable Values"): the variable's type in
    ``modelDescription.xml`` "determines the function
    ``fmi3Get/Set{VariableType}`` that must be used for accessing the
    respective variable values".  The C wrapper used to send every width
    as a float64 and cast the reply, so ``fmi3SetBoolean`` on a Float32
    parameter stored 1.0 and ``fmi3GetInt32`` on a Float32 output
    truncated 0.5 to 0, both answered ``fmi3OK``.  It now names the type
    of every call; a request that names none is not checked.
    """
    if fmi_type is None:
        return
    declared = _fmi_type_of(var)
    if fmi_type != declared:
        raise ValueError(
            f"variable {var.name!r} is {declared}, so fmi3{access}{fmi_type} cannot "
            f"address it: FMI 3.0 accesses a variable only through the fmi3Get / "
            f"fmi3Set function of its own type (fmi3{access}{declared}); nothing "
            "was read or written")


def _start_leaf(var: FMIVariable, live: Any) -> np.ndarray:
    """A parameter variable's advertised ``start`` in its live leaf's shape
    (``ValueError`` if the description cannot describe the leaf).

    Read in the lexical form of the variable's type (``true`` / ``false``,
    an integer literal, an ``xs:double``), into a dtype that holds it
    exactly: float64 for a float, int64 / uint64 for an integer (an int64
    past 2**53 is exact), bool for a Boolean.
    """
    live_arr = np.asarray(live)
    tokens = (var.start or "").split()
    if len(tokens) != live_arr.size:
        raise ValueError(
            f"the model description's start value for {var.name!r} has "
            f"{len(tokens)} entries, but the sidecar's leaf has shape "
            f"{live_arr.shape}; the description does not describe this sidecar")
    kind = _fmi_kind(var.dtype)
    wide = {"float": np.float64, "bool": np.bool_}.get(
        kind, np.uint64 if np.dtype(var.dtype).kind == "u" else np.int64)
    try:
        flat = np.asarray([_parse_xs_value(tok, var.dtype) for tok in tokens], dtype=wide)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"the model description's start value for {var.name!r} "
                         f"is not a value of its type {var.dtype}: {exc}") from exc
    return flat.reshape(live_arr.shape)


def _param_owner(var: FMIVariable) -> tuple[str, str]:
    """The ``(node, key)`` a parameter variable addresses
    (``params["nodes"][node][key]``).

    ``build_model_description`` records the pair on the variable
    (``node`` / ``field``), because ``<node>.params.<key>`` cannot be split
    back when the node's own name holds ``.params.``: splitting a variable
    of node ``rig.params.v2`` at the first one found no such node, so the
    bridge left out its advertised bounds and start value and every set
    was refused as unknown.  A hand-built variable without the pair is
    split at the last ``.params.``, as a parameter key holds no dot.
    """
    if var.node is not None and var.field:
        return var.node, var.field
    node, _, key = var.name.rpartition(".params.")
    return node, key


def checked_value(arr, dtype, *, what: str) -> np.ndarray:
    """``arr`` in ``dtype``, refused unless the model can hold it.

    The one value check on this module's write paths.  ``set`` and
    ``set_state`` both go through it, so an FMU-state archive cannot
    install a value a ``set`` of the same variable would refuse -- which
    it could until 0.4.0, because the two paths each had their own idea
    of what a valid value was and only one of them had any.
    :meth:`FmuSidecar.set_params <maddening.fmi.sidecar.FmuSidecar.set_params>`
    applies the same check (the implementation lives in the sidecar
    module, which this one imports), so the in-process door into the
    parameters is no wider than the wire.

    Parameters
    ----------
    arr : array-like
        The incoming value, in whatever dtype it arrived in (float64 off
        the wire, the archive's own dtype out of an ``npz``).
    dtype : numpy dtype
        The dtype of the live array it would replace.
    what : str
        How to name the value in an error, e.g. ``"variable 'm.params.k'"``.

    Returns
    -------
    numpy.ndarray
        ``arr`` cast to ``dtype``.

    Raises
    ------
    ValueError
        If the incoming value is not finite, or if ``dtype`` cannot hold
        it: a float32 field set to ``1e308`` would be stored (and read
        back) as ``inf``, and an integer would wrap silently.
    """
    return _checked_value(arr, dtype, what=what)


@stability(StabilityLevel.EVOLVING)
class FmuTcpBridge:
    """Serve one :class:`FmuSidecar` to the FMU C wrapper over TCP.

    Parameters
    ----------
    sidecar : FmuSidecar
    model_description : ModelDescription
        The description the FMU was built from; value references are
        resolved against its variables.
    master_dt : float
        The simulated time one sidecar ``step`` advances: one step of the
        graph, ``GraphManager.timestep``.  It must equal the step the
        model description records (:attr:`ModelDescription.graph_timestep
        <maddening.fmi.model_description.ModelDescription.graph_timestep>`,
        or ``default_step_size`` for a description built by hand), and the
        description's ``default_step_size`` must be a whole number of
        them.  A ``master_dt`` of half the graph's step used to be
        accepted: each ``doStep`` ran twice the graph steps it should
        have, and reported a time that was half the state's.
    host, port : str, int
        Bind address; ``port=0`` picks a free port (see :attr:`endpoint`).
    max_steps_per_request : int, default MAX_STEPS_PER_REQUEST
        The most graph steps one ``step`` request may ask for
        (``h / master_dt``); a larger request is refused with nothing
        advanced.  See :data:`MAX_STEPS_PER_REQUEST`.
    idle_timeout : float or None, default ``_IDLE_TIMEOUT`` (300)
        Seconds an established connection may stay silent between frames
        before the bridge drops it (and with it the FMU instance), and
        the bound on sending a reply.  ``None`` waits for ever.  See
        :data:`_IDLE_TIMEOUT`.  The handshake and frame budgets are not
        affected.

    Raises
    ------
    ValueError
        If ``master_dt`` is not a positive finite number, differs from the
        graph step the description records, or does not divide the
        description's ``default_step_size`` into a whole number of steps
        no larger than ``max_steps_per_request``; if the graph the
        description was built from, or the graph whose compiled step is
        the sidecar's ``step_fn``, has changed or been compiled again
        since; if a parameter's ``ParamSpec`` in the description's graph
        has changed since (``set_param_spec``) so that its ``min``, ``max``
        or ``unit`` is no longer the one advertised, or the sidecar's own
        spec for an exported parameter enforces another ``min`` / ``max``;
        if ``max_steps_per_request`` is not a positive integer; or if
        ``idle_timeout`` is neither ``None`` nor a positive finite number.
    """

    def __init__(
        self,
        sidecar: FmuSidecar,
        model_description: ModelDescription,
        *,
        master_dt: float,
        host: str = "127.0.0.1",
        port: int = 0,
        max_steps_per_request: int = MAX_STEPS_PER_REQUEST,
        idle_timeout: "Optional[float] | _ModuleDefault" = _IDLE_DEFAULT,
    ) -> None:
        self._sidecar = sidecar
        self._md = model_description
        if isinstance(idle_timeout, _ModuleDefault):
            idle_timeout = _IDLE_TIMEOUT
        if idle_timeout is not None:
            idle_timeout = _real_number(idle_timeout, "idle_timeout")
            if idle_timeout <= 0:
                raise ValueError(f"idle_timeout must be positive or None, got {idle_timeout!r}")
        self._idle_timeout: Optional[float] = idle_timeout
        if isinstance(max_steps_per_request, (bool, np.bool_)) or not isinstance(
                max_steps_per_request, (int, np.integer)) or max_steps_per_request < 1:
            raise ValueError(f"max_steps_per_request must be a positive integer, got "
                             f"{max_steps_per_request!r}")
        self._max_steps = int(max_steps_per_request)
        self._dt = _real_number(master_dt, "master_dt")
        if self._dt <= 0:
            raise ValueError(f"master_dt must be positive, got {master_dt!r}")
        self._check_master_dt(model_description)
        self._check_one_compile(model_description, sidecar)
        self._vars: dict[int, FMIVariable] = {
            v.value_reference: v for v in model_description.variables
        }
        # Every declared input starts at its advertised start value (0),
        # exactly as gm.step() fills an unset external input, so a node
        # whose update() has a non-zero fallback for a *missing* input
        # (HeatNode's T_left = T[0]) behaves like the graph.
        self._inputs: dict[str, dict[str, Any]] = self._zero_inputs()
        self._time = 0.0
        # The time the physics has reached, as a count of graph steps since a
        # reference point (the start time, a reset, a restored snapshot):
        # ``_t_ref + _n_ref * master_dt``.  ``_time`` adopts the importer's
        # point within the tolerance; this does not, so the drift between
        # the time the FMU reports and the time it has simulated is bounded
        # (``_drift_tolerance``).
        self._t_ref = 0.0
        self._n_ref = 0
        # Has the instance taken a step since it was instantiated or reset?
        # (FMI's "Instantiated" state is the one in which it has not, and
        # the only one in which ``initialize`` is accepted.)
        self._stepped = False
        # Has the instance been terminated (fmi3Terminate) since it was
        # instantiated or reset?  FMI's Terminated state allows reading
        # and the FMU-state functions, and leaves only by fmi3Reset.
        self._terminated = False
        self._initial_state = _copy_tree(sidecar.state)
        # The description is the FMU's contract: a graph parameter it does
        # not export as a tunable variable is one the step cannot read (see
        # ModelDescription.fixed_parameters), and no door into the sidecar's
        # parameter tree -- set, set_state, or the sidecar's own set_params
        # -- may install a new value for it.  Applied to the sidecar here,
        # however it was configured, so an FMU never reports a value it
        # does not compute with.
        exported = {v.name for v in model_description.variables
                    if v.causality == "parameter"}
        fixed = dict(getattr(model_description, "fixed_parameters", {}) or {})
        for owner, leaves in ((sidecar.params or {}).get("nodes") or {}).items():
            for key in leaves:
                name = f"{owner}.params.{key}"
                if name not in exported:
                    fixed.setdefault(
                        name, "it is not a tunable parameter variable of this "
                              "FMU's model description")
        sidecar._refuse_new_values_for(fixed)             # noqa: SLF001
        # The same contract for the bounds the description advertises.  The
        # sidecar enforces a ParamSpec it was given; one built without
        # ``param_specs`` (the default) enforced nothing, so the XML said
        # ``elasticity`` in [0, 1] and a ``set`` of 1.5 -- or an archive
        # installing -3.0 -- answered ok.  A sidecar spec that enforces
        # another envelope than the XML advertises is refused here, and the
        # advertised bounds are enforced alongside the sidecar's own specs.
        self._check_sidecar_envelope(model_description, sidecar)
        sidecar._adopt_advertised_bounds(                 # noqa: SLF001
            self._advertised_bounds(model_description, sidecar))
        # And for the inputs: the sidecar completes and checks each step's
        # inputs with the graph's own resolver when it was given one;
        # otherwise with one built from the description, which knows the
        # inputs it exports and those it holds at zero (``held_inputs``).
        # Without either, an input the description left out never reached
        # its node, which took its own "input missing" branch.
        sidecar._adopt_input_resolver(                    # noqa: SLF001
            _declared_inputs_resolver(self._declared_inputs(model_description),
                                      whose="this FMU's model description"))
        # What an instance starts from: FMI starts every instance at the
        # start values its modelDescription.xml advertises, which for a
        # parameter is the value it held when the description was built.
        # A sidecar configured with something else is brought into line,
        # loudly, so the FMU never computes with a value its XML denies.
        start = self._start_params(model_description, sidecar)
        if start is not None:
            sidecar._adopt_instantiation_params(start)    # noqa: SLF001
        self._initial_params = _copy_tree(sidecar.params)
        self._busy = threading.Lock()                     # one instance per bridge
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(_MAX_CONNECTIONS)
        # Set here, not at the top of ``_serve``: ``stop()`` may close the
        # socket before the serve thread's first line runs, and a
        # ``settimeout`` on a closed socket raises EBADF out of the thread.
        # ``accept`` on a closed socket raises an OSError the loop already
        # treats as "stopped".
        self._server.settimeout(0.2)
        self._host, self._port = self._server.getsockname()[:2]
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Orders start() against stop(): a start() that loses the race sees
        # the stop flag and refuses, one that wins has published its thread
        # before stop() reads it for the join.  It also orders every commit
        # into the model against stop() (``_committing``): stop() sets the
        # flag under it, so a commit either finishes before stop() goes on
        # or sees the flag and writes nothing.
        self._lifecycle_lock = threading.Lock()
        # Live accepted connections and the threads serving them.  ``stop()``
        # has to reach both: closing the listening socket says nothing to a
        # connection already accepted, and a worker parked on one of those
        # outlives the bridge that "stopped".
        self._live_lock = threading.Lock()
        self._live_conns: set[socket.socket] = set()
        self._live_workers: set[threading.Thread] = set()
        # The op each worker is serving right now, so a worker that outlives
        # stop() can be named with what it is doing.
        self._in_flight: dict[threading.Thread, Any] = {}
        self.requests_served = 0
        self.binary_frames_served = 0     # binary replies sent (get / get_state)
        self.binary_frames_received = 0   # binary requests accepted (set / set_state)
        self.connections_refused_over_cap = 0
        self.replies_unencodable = 0      # replies that could not be framed (answered as errors)

    # ----------------------------------------------------------------- server
    @property
    def endpoint(self) -> str:
        return f"{self._host}:{self._port}"

    def start(self) -> "FmuTcpBridge":
        """Start serving on :attr:`endpoint`; returns the bridge.

        A bridge serves once.  Its listening socket is bound in the
        constructor and closed by :meth:`stop`, so a ``start()`` after
        ``stop()`` would start a thread with nothing to accept on and
        return as if it served; and a second ``start()`` would run two
        accept loops on one socket, only the last of which ``stop()``
        joins.  Both are refused.

        Raises
        ------
        RuntimeError
            If the bridge was already started, or has been stopped (build
            a new :class:`FmuTcpBridge` to serve again).
        """
        with self._lifecycle_lock:
            if self._stop.is_set():
                raise RuntimeError(
                    f"FmuTcpBridge on {self.endpoint} has been stopped and its "
                    "listening socket closed; build a new FmuTcpBridge to serve "
                    "again"
                )
            if self._thread is not None:
                raise RuntimeError(
                    f"FmuTcpBridge on {self.endpoint} is already started; "
                    "start() it once"
                )
            self._thread = threading.Thread(target=self._serve,
                                            name="maddening-fmu-bridge", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        """Stop accepting and end every connection this bridge still holds.

        Closing the listening socket only stops new connections; a worker
        blocked reading an accepted one is untouched by it and used to
        survive ``stop()`` indefinitely, still holding the instance lock.
        Each live connection is therefore shut down here, which turns the
        worker's pending ``recv`` into an EOF, and the workers are joined.
        Stopping is final (see :meth:`start`) and idempotent: a second
        ``stop()``, or one before ``start()``, is quiet.

        The wait is bounded: ``stop()`` returns within about
        ``_STOP_JOIN_TIMEOUT`` (five seconds) however many connections
        were live.  A worker that is inside a request at that point -- a
        sidecar step can take longer than that, and the first one compiles
        the graph -- cannot be interrupted.  ``stop()`` then logs a warning
        naming every thread still alive and the op each worker is serving,
        and that request is refused where it would commit: once ``stop()``
        has begun, no request changes the model's state, parameters,
        inputs or time, whether it began before ``stop()`` or after.  The
        instance slot stays held until such a worker returns; a stopped
        bridge serves no one, so nothing waits on it.
        """
        with self._lifecycle_lock:
            self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass
        started = time.monotonic()
        deadline = started + _STOP_JOIN_TIMEOUT
        workers = self._shut_down_live_connections()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        # A connection the serve thread accepted just before the listening
        # socket closed may have registered after the first sweep.
        workers |= self._shut_down_live_connections()
        for worker in workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [t for t in (self._thread, *workers) if t is not None and t.is_alive()]
        if alive:
            with self._live_lock:
                doing = {t: self._in_flight.get(t) for t in alive}
            logger.warning(
                "FmuTcpBridge on %s: stop() returned after %.1f s with %d thread(s) "
                "still alive: %s.  A worker inside a request cannot be interrupted; "
                "its request will commit nothing, and it exits when the request "
                "returns.",
                self.endpoint, time.monotonic() - started, len(alive),
                "; ".join(self._describe_thread(t, doing[t]) for t in alive),
            )

    def _shut_down_live_connections(self) -> set[threading.Thread]:
        """Half-close every live connection; returns the workers serving them."""
        with self._live_lock:
            conns = list(self._live_conns)
            workers = set(self._live_workers)
        for conn in conns:
            # shutdown, not close: the worker owns the socket object and
            # closes it on its way out, and a half-close is what makes its
            # blocking recv return instead of waiting for a peer that is
            # never going to speak.
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        return workers

    def _describe_thread(self, thread: threading.Thread, op: Any) -> str:
        what = "the accept loop" if thread is self._thread else (
            f"serving op {op!r}" if op is not None else "between requests")
        return f"{thread.name} (ident {thread.ident}, {what})"

    @contextlib.contextmanager
    def _committing(self, op: str) -> Iterator[None]:
        """Hold the lifecycle lock around one commit into the model.

        Every write into the sidecar's state or parameters, the pending
        inputs or the time goes through here, *after* everything that can
        take long (a sidecar step, decoding an archive) has run on local
        copies.  ``stop()`` sets its flag under the same lock, so a commit
        either completes before ``stop()`` proceeds or finds the flag set
        and raises -- which the dispatcher answers as an error reply with
        nothing written.  Without it, a worker that outlived ``stop()``'s
        join replaced the model's state after ``stop()`` had returned.

        Raises
        ------
        RuntimeError
            If the bridge has been stopped.
        """
        with self._lifecycle_lock:
            if self._stop.is_set():
                raise RuntimeError(
                    f"FmuTcpBridge on {self.endpoint} has been stopped; the {op!r} "
                    "request was not committed"
                )
            yield

    def __enter__(self) -> "FmuTcpBridge":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            worker = threading.Thread(target=self._serve_conn, args=(conn,),
                                      name="maddening-fmu-conn", daemon=True)
            with self._live_lock:
                if len(self._live_conns) >= _MAX_CONNECTIONS:
                    over_cap = True
                else:
                    over_cap = False
                    self._live_conns.add(conn)
                    self._live_workers.add(worker)
            if over_cap:
                # Nothing is written back: a reply needs a send that a peer
                # which is not reading can stall, and stalling the accept
                # loop is exactly what the cap exists to prevent.
                self.connections_refused_over_cap += 1
                try:
                    conn.close()
                except OSError:
                    pass
                continue
            try:
                worker.start()
            except RuntimeError:
                # The process cannot make another thread.  Undo the
                # bookkeeping the worker's own ``finally`` would have done,
                # or the cap fills with connections nothing is serving.
                with self._live_lock:
                    self._live_conns.discard(conn)
                    self._live_workers.discard(worker)
                try:
                    conn.close()
                except OSError:
                    pass

    def _serve_conn(self, conn: socket.socket) -> None:
        try:
            self._serve_conn_inner(conn)
        finally:
            with self._live_lock:
                self._live_conns.discard(conn)
                self._live_workers.discard(threading.current_thread())
                self._in_flight.pop(threading.current_thread(), None)

    def _claim_instance(self) -> bool:
        """Claim the single instance slot, waiting out a hand-over.

        Returns ``True`` if this connection now holds the slot.  The wait
        is bounded by :data:`_HANDOVER_GRACE` and abandoned early if the
        bridge is stopping, so a refusal stays a refusal rather than
        becoming a hang -- the distinction the "refused, not blocked"
        contract is about.  See :data:`_HANDOVER_GRACE` for why waiting
        at all is necessary.
        """
        deadline = time.monotonic() + _HANDOVER_GRACE
        while True:
            if self._busy.acquire(timeout=_HANDOVER_POLL):
                return True
            if self._stop.is_set() or time.monotonic() >= deadline:
                return False

    def _serve_conn_inner(self, conn: socket.socket) -> None:
        with conn:
            # The read loop's condition below makes the same check before
            # its first read, so either one alone keeps a worker started
            # after stop() from serving (pinned by
            # test_a_worker_that_starts_after_stop_serves_nothing, which
            # fails only with both removed).
            if self._stop.is_set():
                return
            binary = False                # negotiated at hello, per connection
            held = False                  # does this connection hold the instance?
            try:
                while not self._stop.is_set():
                    # Every wait on this socket is finite, and the first one
                    # happens before the instance slot is claimed.  Both
                    # matter: a peer that connects and says nothing used to
                    # park the bridge's only FMU instance for ever (a port
                    # scan, a crashed importer or a dropped link was enough),
                    # and a peer that was refused the slot used to park a
                    # thread for ever.
                    conn.settimeout(self._idle_timeout if held else _HANDSHAKE_TIMEOUT)
                    try:
                        got = recv_raw(conn, frame_timeout=_FRAME_TIMEOUT)
                    except socket.timeout:
                        break        # a silent peer is a gone peer: same as EOF
                    except (OSError, ValueError):
                        break                              # socket / framing error
                    if got is None:
                        break
                    if not held:
                        # A bridge holds ONE sidecar state; a second instance
                        # would silently share it.  Refuse instead of
                        # blocking -- but only now that this peer has proved
                        # it has something to say, and only once a departing
                        # connection has had ``_HANDOVER_GRACE`` to release
                        # the slot it no longer holds.
                        if not self._claim_instance():
                            try:
                                send_message(conn, {"ok": False, "error":
                                                    "bridge already serves an FMU instance; "
                                                    "start one FmuTcpBridge per instance"})
                            except OSError:
                                pass
                            return
                        held = True
                        # A connection that claims the slot is a new FMU
                        # instance (the C wrapper opens one per
                        # fmi3InstantiateCoSimulation), and FMI starts every
                        # instance at the description's start values.  The
                        # slot used to carry the previous instance's state,
                        # parameters, inputs and time over, so FMPy's
                        # simulate_fmu called twice with the same arguments
                        # gave two different answers, each fmi3OK.  Here,
                        # in the bridge, so every client is covered.
                        try:
                            self._reset_instance("instantiate")
                        except RuntimeError:
                            return           # stopped meanwhile: serve nothing
                        # From here the generous budget applies to the reply
                        # send as well: a first frame that asks for a 64 MiB
                        # get should not be cut off by the handshake budget.
                        conn.settimeout(self._idle_timeout)
                    is_binary, body = got
                    # Bound before the parse: a malformed *first* frame
                    # leaves the name unset otherwise, and the reply-failure
                    # handler below reads it.
                    req: Any = None
                    negotiated: Optional[bool] = None
                    try:
                        if is_binary:
                            if not binary:
                                raise ValueError("binary frames need a hello with "
                                                 "\"protocol\": 2, \"binary\": true first")
                            req = self._binary_request(*decode_binary(body))
                            self.binary_frames_received += 1
                        else:
                            req = _json_object(body, "request")
                            if not isinstance(req, dict):
                                raise ValueError("request must be a JSON object")
                    except ValueError as exc:
                        # a corrupt request (an FMU-state blob with stray
                        # quotes, say) is an error reply, not a dead instance
                        reply = {"ok": False, "error": f"malformed request: {exc}"}
                    else:
                        with self._live_lock:
                            self._in_flight[threading.current_thread()] = req.get("op")
                        try:
                            reply = self._dispatch(req)
                        finally:
                            with self._live_lock:
                                self._in_flight.pop(threading.current_thread(), None)
                        if req.get("op") == "hello" and reply.get("ok"):
                            # Applied only once the reply is on the wire
                            # (below): a hello whose reply could not be sent
                            # must not leave this end speaking a protocol the
                            # peer never heard agreed to.
                            negotiated = bool(reply.get("binary"))
                    try:
                        if self._send_reply(conn, reply, binary):
                            self.binary_frames_served += 1
                    except OSError:
                        break          # the importer hung up mid-reply: nobody to tell
                    except Exception as exc:  # noqa: BLE001 - see below
                        # Building the frame failed, not the socket.  The
                        # protocol answers *every* request, including the
                        # ones the bridge itself cannot serve, so this is
                        # an error reply rather than an exception -- and
                        # this handler is the reason the protocol can keep
                        # that promise.  Without it the exception unwinds
                        # out of ``_serve_conn`` and ends the worker
                        # thread: the importer gets EOF instead of a
                        # reply, and the only record is a traceback from
                        # ``threading.excepthook``.  The reachable case is
                        # a ``model_name`` spelling a non-finite token
                        # (``MADD-ANO-010``), which used to kill the
                        # worker on the *first* frame of every connection
                        # and made the FMU unusable with nothing naming
                        # the model; ``build_model_description`` now
                        # refuses that name, but a ``ModelDescription``
                        # built by hand still reaches here, and the
                        # free-text surface of the protocol will grow.
                        # Logged, never swallowed: a genuine bug in reply
                        # construction must still be visible.
                        self.replies_unencodable += 1
                        logger.exception(
                            "FMU bridge could not encode the reply to op %r; "
                            "answering with an error reply",
                            req.get("op") if isinstance(req, dict) else None,
                        )
                        try:
                            send_message(conn, self._unencodable_reply(exc))
                        except OSError:
                            break
                    else:
                        if negotiated is not None:
                            binary = negotiated
            finally:
                if held:
                    self._busy.release()

    def _send_reply(self, conn: socket.socket, reply: dict, binary: bool) -> bool:
        """Send one reply; returns whether it went as a binary frame.

        A successful ``get`` / ``get_state`` on a binary connection is a
        binary frame, everything else JSON.  A reply of either kind that
        would exceed the frame limit is replaced by a JSON error reply:
        the bridge never puts a frame on the wire that the C wrapper
        would refuse to read, so the connection stays in sync.
        """
        if binary and reply.get("ok") and ("values" in reply or "state" in reply):
            if "values" in reply:
                arr = np.ascontiguousarray(reply["values"], dtype="<f8")
                header, raw = {"ok": True, "n": int(arr.size), "dtype": "f64"}, arr.tobytes()
            else:
                header, raw = {"ok": True, "n": len(reply["state"])}, reply["state"]
            body = encode_binary(header, raw)
            if len(body) <= _MAX_MESSAGE:
                conn.sendall(_HEADER.pack(_BINARY_FLAG | len(body)) + body)
                return True
        elif reply.get("ok") and "values" in reply and set(reply) == {"ok", "values"}:
            # A get's JSON reply is encoded a chunk at a time and abandoned
            # as soon as it passes the limit, so a reply too long to send
            # costs no more memory than a frame (the whole list, its copy
            # and its text used to be built first: ~0.5 GB at 8 M values).
            values_body = self._json_values_body(reply["values"])
            if values_body is not None:
                conn.sendall(_HEADER.pack(len(values_body)) + values_body)
                return False
            send_message(conn, {"ok": False, "error": f"reply of more than {_MAX_MESSAGE} "
                                                       f"bytes exceeds the {_MAX_MESSAGE}-byte "
                                                       "frame limit"})
            return False
        else:
            body = _json_dumps(self._jsonify(reply), separators=(",", ":")).encode("utf-8")
            if len(body) <= _MAX_MESSAGE:
                conn.sendall(_HEADER.pack(len(body)) + body)
                return False
        send_message(conn, {"ok": False, "error": f"reply of {len(body)} bytes exceeds the "
                                                   f"{_MAX_MESSAGE}-byte frame limit"})
        return False

    @staticmethod
    def _json_values_body(values: Any) -> Optional[bytes]:
        """``{"ok":true,"values":[...]}`` for a ``get`` on a JSON connection,
        byte for byte what encoding the whole reply writes, or ``None`` once
        it would pass the frame limit -- found while encoding, a chunk of
        :data:`_JSON_VALUES_CHUNK` values at a time."""
        arr = np.asarray(values, dtype=np.float64).ravel()
        head, tail = b'{"ok":true,"values":[', b"]}"
        size = len(head) + len(tail)
        pieces: list[bytes] = []
        for start in range(0, arr.size, _JSON_VALUES_CHUNK):
            text = _json_dumps(arr[start:start + _JSON_VALUES_CHUNK].tolist(),
                               separators=(",", ":")).encode("utf-8")[1:-1]
            size += len(text) + (1 if pieces else 0)
            if size > _MAX_MESSAGE:
                return None
            pieces.append(text)
        return head + b",".join(pieces) + tail

    @staticmethod
    def _unencodable_reply(exc: BaseException) -> dict:
        """A JSON error reply for a reply that could not be framed.

        Everything about it is chosen so that sending it cannot fail the
        way the reply it replaces did: the text is prefixed (so it can
        never itself equal a non-finite token, which is the collision
        that makes a reply unencodable in the first place), reduced to
        printable ASCII (the C wrapper pulls it out of the frame with
        ``strstr`` and hands it to the importer's log callback as a C
        string) and length-capped well below the frame limit.

        Parameters
        ----------
        exc : BaseException
            What :meth:`_send_reply` raised.

        Returns
        -------
        dict
            ``{"ok": False, "error": ...}`` -- the protocol's failure
            reply, which the C wrapper already logs and turns into
            ``fmi3Error``.
        """
        try:
            text = f"{type(exc).__name__}: {exc}"
        except Exception:  # noqa: BLE001 - an exception's own __str__ may raise
            text = type(exc).__name__
        text = text.encode("ascii", "backslashreplace").decode("ascii")
        text = "".join(c if " " <= c <= "~" else " " for c in text)
        if len(text) > _ERROR_TEXT_MAX:
            text = text[:_ERROR_TEXT_MAX - 3] + "..."
        return {"ok": False, "error": _UNENCODABLE_PREFIX + text}

    @staticmethod
    def _binary_request(header: dict, raw) -> dict:
        """A binary ``set`` / ``set_state`` request (its decoded header and
        raw part) as the plain request dict :meth:`_dispatch` takes
        (``values`` as an array, ``state`` as bytes).  ``ValueError`` on
        any inconsistency; nothing is trusted."""
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise ValueError("the raw part of a binary request must be bytes")
        op = header.get("op")
        n = header.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError("binary header needs a non-negative integer \"n\"")
        if op == "set":
            if header.get("dtype") != "f64":
                raise ValueError(f"unsupported binary dtype {header.get('dtype')!r} (only f64)")
            if len(raw) != 8 * n:
                raise ValueError(f"binary set announces {n} float64 but carries {len(raw)} bytes")
            values = np.frombuffer(raw, dtype="<f8").astype(np.float64)
            req: dict[str, Any] = {"op": "set", "vr": header.get("vr"), "values": values}
            if "type" in header:
                req["type"] = header["type"]
            return req
        if op == "set_state":
            if len(raw) != n:
                raise ValueError(f"binary set_state announces {n} bytes but carries {len(raw)}")
            return {"op": "set_state", "state": bytes(raw)}
        raise ValueError(f"op {op!r} has no binary request form")

    @staticmethod
    def _jsonify(reply: dict) -> dict:
        """The JSON (protocol-1) form of a reply: values as a list, state as base64."""
        if reply.get("ok"):
            if "values" in reply:
                reply = {**reply, "values": np.asarray(reply["values"], dtype=np.float64).tolist()}
            if "state" in reply:
                reply = {**reply, "state": base64.b64encode(reply["state"]).decode("ascii")}
        return reply

    # ---------------------------------------------------------------- handler
    def handle(self, req: dict) -> dict:
        """Serve one decoded request (also usable without a socket) in its
        JSON form: ``values`` come back as a list, ``state`` as base64.

        ``req`` is either the JSON form or the dict :func:`recv_message`
        returns for a binary frame (the header's keys plus ``"raw"``);
        the latter is validated exactly as on the socket path.
        """
        if "raw" in req:
            try:
                req = self._binary_request({k: v for k, v in req.items() if k != "raw"}, req["raw"])
            except ValueError as exc:
                return {"ok": False, "error": f"malformed request: {exc}"}
        return self._jsonify(self._dispatch(req))

    def _dispatch(self, req: dict) -> dict:
        """Serve one request; ``values`` is a float64 array and ``state``
        raw npz bytes, encoded by the caller for the wire in use."""
        self.requests_served += 1
        try:
            op = req.get("op")
            if op == "hello":
                proto = req.get("protocol", 1)
                if not isinstance(proto, int) or isinstance(proto, bool) or proto < 1:
                    raise ValueError(f"protocol must be a positive integer, got {proto!r}")
                if proto > PROTOCOL_VERSION:
                    raise ValueError(f"protocol {proto} is not supported by this bridge "
                                     f"(highest is {PROTOCOL_VERSION})")
                binary = proto >= 2 and req.get("binary") is True
                return {"ok": True, "token": self._md.instantiation_token,
                        "model": self._md.model_name, "master_dt": self._dt,
                        "protocol": PROTOCOL_VERSION, "binary": binary}
            if op == "set":
                self._set(req["vr"], req["values"], req.get("type"))
                return {"ok": True}
            if op == "get":
                return {"ok": True, "values": self._get(req["vr"], req.get("type"))}
            if op == "step":
                h = _real_number(req["dt"], "communication step")
                if h <= 0:
                    raise ValueError(f"communication step must be positive, got {h!r}")
                # The FMU advertises a fixed communication step: the physics
                # can only advance whole master steps, and reporting t + h
                # for a different advance would desync the importer's time
                # from the state.
                n = self._master_steps(h, "communication step")
                # The communication point is parsed *before* anything moves.
                # It used to be parsed after the loop, so a request carrying a
                # ``t`` that is not a number was answered "not ok" with the
                # physics already advanced and ``_time`` left behind it: the
                # importer's clock and the bridge's state desynchronise
                # permanently, and nothing on the wire says so.
                t0 = (_real_number(req["t"], "communication point") if "t" in req
                      else self._time)
                # At a time whose ulps are a sizeable part of a master step no
                # tolerance can tell rounding from a skipped step; refused as
                # such (the point, and the end the FMU would report), before
                # the tolerance checks below would misname it.
                self._check_time_resolution(t0, "communication point")
                self._check_time_resolution(t0 + h, "end of the step (t + h)")
                # And it must be where the FMU is (_COMM_POINT_TOLERANCE).  A
                # point that jumped -- 0.01 to 100, say -- used to be
                # accepted: the physics advanced one master step and ``time``
                # read 100.01, a state labelled with a time it never reached.
                tolerance = self._time_tolerance(t0, self._time, h, ulps=_COMM_POINT_ULPS)
                if abs(t0 - self._time) > tolerance:
                    raise ValueError(
                        f"communication point {t0!r} is not the FMU's current time "
                        f"{self._time!r} (tolerance {tolerance:.3g}): a step starts "
                        "where the previous one ended -- the previous communication "
                        "point plus the previous step size -- or, first, at the start "
                        "time given to initialize (fmi3EnterInitializationMode).  "
                        "Restore an FMU state to go back in time; nothing was advanced"
                    )
                # And the importer's clock must stay where the physics is.
                # Adopting each point within the tolerance of the previous one
                # let a biased importer -- every step size a little long --
                # move the reported time ahead of the simulated time by up to
                # the tolerance per step, without bound: 1.6 master steps
                # after 2000 steps at a 1e-12 s step.  Both the point and the
                # next legal one (t + h) are held to the simulated time, to
                # the tolerance plus the rounding a running sum can gather.
                self._check_drift(t0, h, n)
                self._refuse_if_terminated("step")
                # Every sub-step runs on a local state and the result is
                # committed once, at the end: a failed sub-step leaves no
                # partial advance behind (the importer is told nothing
                # happened), and a step that is still running when stop()
                # is called -- the first one compiles the graph and can
                # outlast stop()'s bounded join -- is refused at its commit
                # instead of replacing the state of a stopped bridge.
                state = self._sidecar._state                        # noqa: SLF001
                for done in range(n):
                    # A long step stops at the first graph step after
                    # stop(), instead of running to its end in a worker
                    # nobody is waiting for (the commit below would refuse
                    # it anyway).  One graph step -- the first, which
                    # compiles -- still cannot be interrupted.
                    if self._stop.is_set():
                        raise RuntimeError(
                            f"FmuTcpBridge on {self.endpoint} has been stopped; the "
                            f"'step' request was not committed (it stopped after "
                            f"{done} of its {n} graph steps)")
                    state = self._sidecar._advanced(state, self._inputs)  # noqa: SLF001
                with self._committing("step"):
                    self._sidecar._state = state                    # noqa: SLF001
                    # On the importer's clock: the point it sent, adopted
                    # within the tolerance, plus the master steps taken.
                    self._time = t0 + n * self._dt
                    self._n_ref += n
                    self._stepped = True
                    return {"ok": True, "t": self._time}
            if op == "initialize":
                # fmi3EnterInitializationMode(startTime): the instance's time
                # is the start time from here, before its first step.  The C
                # wrapper used to keep startTime to itself, so ``time`` read
                # 0.0 until the first step.
                t_start = _real_number(req.get("t"), "start time")
                self._check_time_resolution(t_start, "start time")
                if self._stepped:
                    raise ValueError(
                        "initialize after the instance has stepped: FMI enters "
                        "initialization mode once, between instantiation (or "
                        "reset) and the first step; reset first to start again"
                    )
                self._refuse_if_terminated("initialize")
                with self._committing("initialize"):
                    self._time = t_start
                    self._t_ref, self._n_ref = t_start, 0
                return {"ok": True, "t": t_start}
            if op == "get_state":
                return {"ok": True, "state": self._encode_state()}
            if op == "set_state":
                blob = req["state"]
                if not isinstance(blob, (bytes, bytearray)):
                    blob = base64.b64decode(blob)
                self._decode_state(bytes(blob))
                # The restored time, for the C wrapper's own clock: it
                # reports it as lastSuccessfulTime when a later doStep
                # fails, and used to report the time before the restore.
                return {"ok": True, "t": self._time}
            if op == "reset":
                # fmi3Reset: the state a fresh instance starts from, the
                # same commit a claim of the instance slot makes.
                self._reset_instance("reset")
                return {"ok": True}
            if op == "terminate":
                # fmi3Terminate: the instance is in FMI's Terminated state
                # until fmi3Reset.  A doStep after it used to advance the
                # model and answer ok.  Idempotent.
                with self._committing("terminate"):
                    self._terminated = True
                return {"ok": True}
            return {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as exc:  # noqa: BLE001 - reported to the importer
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _zero_inputs(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for var in self._md.variables:
            if var.causality == "input" and not var.is_clock:
                node, field = var.node_field()
                out.setdefault(node, {})[field] = jnp.zeros(var.shape or (), dtype=var.dtype)
        return out

    # ------------------------------------------------------------------- time
    def _slack(self, ulps: float, biggest: float) -> float:
        """``ulps`` ulps of ``biggest``, capped at :data:`_ROUNDING_SLACK_MAX`
        of a master step."""
        return min(ulps * float(np.spacing(biggest)), _ROUNDING_SLACK_MAX * self._dt)

    def _time_tolerance(self, *times: float, ulps: int) -> float:
        """The bridge's one tolerance on time (:data:`_COMM_POINT_TOLERANCE`
        of a master step), plus ``ulps`` ulps of the largest of ``times``,
        never more than :data:`_ROUNDING_SLACK_MAX` of a master step.

        Relative to the master step all the way down: the rounding slack is
        in ulps of the times involved, with no floor.  It used to be in ulps
        of ``max(|t|, 1.0)``, an absolute ~3.6e-15 s, so below a master step
        of about 1e-9 s the tolerance was that floor, not a millionth of a
        step: 8.9e-4 of a 1e-12 s step was accepted.  And with no cap, at a
        large ratio of time to master step it was more than a step: 16 ulps
        of 1000 s are 1.8 steps of 1e-12 s, and a ``doStep`` one whole
        master step ahead of the FMU's time was adopted.
        """
        biggest = max(abs(t) for t in times)
        return _COMM_POINT_TOLERANCE * self._dt + self._slack(ulps, biggest)

    def _drift_tolerance(self, t: float, steps: int) -> float:
        """How far the importer's clock ``t`` may sit from the simulated time
        ``_t_ref + steps * master_dt``: the communication point's own
        tolerance (:data:`_COMM_POINT_TOLERANCE` of a master step plus
        :data:`_COMM_POINT_ULPS` ulps), plus :data:`_DRIFT_ULPS_PER_STEP` ulps
        of the times involved for every graph step since the reference point
        -- the rounding an importer's running sum of step sizes gathers --
        with the ulps together never more than :data:`_ROUNDING_SLACK_MAX` of
        a master step.

        Uncapped, the slack outgrew a master step at a large ratio of time
        to step (by 0.45 of a 1e-12 s step per step at 1000 s), and an
        importer whose clock ran 40% fast was never refused.  Capped, an
        importer that keeps a running sum is accepted while the rounding it
        has gathered stays under the cap -- at most half an ulp of the time
        per step, so from a start of 0 for at least about forty million
        steps at any master step, from 1 s at a 1e-9 s step for about a
        million -- and refused, with nothing advanced, past it; one that
        computes ``start + k * h`` rounds once per point and is never
        refused."""
        simulated = self._t_ref + steps * self._dt
        biggest = max(abs(t), abs(simulated), abs(self._t_ref), steps * self._dt)
        ulps = _COMM_POINT_ULPS + _DRIFT_ULPS_PER_STEP * steps
        return _COMM_POINT_TOLERANCE * self._dt + self._slack(ulps, biggest)

    def _check_time_resolution(self, t: float, what: str) -> None:
        """Refuse a time at which the FMU's float64 clock cannot place a
        communication point to within :data:`_ROUNDING_SLACK_MAX` of a
        master step: one where :data:`_COMM_POINT_ULPS` ulps of ``t`` exceed
        it.

        At 1000 s one ulp is 0.11 of a 1e-12 s master step, and at 1.7e9 s
        (an epoch-like start) 0.24 of a 1e-6 s one: an importer's own
        rounding there is a sizeable part of a step, so no tolerance can
        tell an honest clock from one that has skipped a step.  The bridge
        used to widen its tolerance with the ulp and admit whole master
        steps of drift; it now refuses the configuration, loudly, at the
        start time (``initialize``), at the step that would reach such a
        time, and at a restored one.

        Raises
        ------
        ValueError
            Naming the time, the master step, the ratio and the largest
            time this master step resolves; nothing is written.
        """
        ulp = float(np.spacing(abs(t)))
        if _COMM_POINT_ULPS * ulp <= _ROUNDING_SLACK_MAX * self._dt:
            return
        # The largest power-of-two binade whose spacing still passes.
        limit = _ROUNDING_SLACK_MAX * self._dt / _COMM_POINT_ULPS
        largest = 2.0 ** (np.floor(np.log2(limit)) + 53)
        raise ValueError(
            f"the {what} {t!r} is too large for the master step {self._dt!r}: one "
            f"float64 ulp of it is {ulp / self._dt:.3g} of a master step, so the FMU's "
            f"clock cannot place a communication point to within "
            f"{_ROUNDING_SLACK_MAX:g} of a step there, and the time it reports could "
            f"leave its physics by whole steps.  This master step resolves times below "
            f"about {largest:.3g} s; start nearer 0 (fmi3EnterInitializationMode), or "
            "serve a graph with a larger timestep.  Nothing was written")

    def _check_drift(self, t0: float, h: float, n: int) -> None:
        """Refuse a step whose point ``t0``, or whose end on the importer's
        clock ``t0 + h``, has drifted from the simulated time by more than
        :meth:`_drift_tolerance`.

        The end is checked so that a step the bridge accepts never makes
        the next legal step fail: its point is ``t0 + h``, judged at the
        next step against the same simulated time with the same arguments.
        """
        for label, t, steps in (("communication point", t0, self._n_ref),
                                ("end of the step (t + h)", t0 + h, self._n_ref + n)):
            simulated = self._t_ref + steps * self._dt
            tolerance = self._drift_tolerance(t, steps)
            if abs(t - simulated) > tolerance:
                raise ValueError(
                    f"the {label} {t!r} is {(t - simulated) / self._dt:.3g} master steps "
                    f"from the time the FMU has simulated, {simulated!r} (tolerance "
                    f"{tolerance:.3g}): every step size this importer sent was within "
                    f"{_COMM_POINT_TOLERANCE:g} of a master step, but their errors add up, "
                    "and the time the FMU reports would leave the physics behind.  A "
                    "running sum of step sizes gathers up to half an ulp of the time per "
                    f"step, and past {_ROUNDING_SLACK_MAX:g} of a master step the bridge "
                    "refuses rather than drift.  Step at whole multiples of the master "
                    "step (the description's stepSize), computing each point as start + "
                    "k * h; nothing was advanced")

    def _master_steps(self, h: float, what: str) -> int:
        """The number of master steps ``h`` is, refused unless it is a whole
        number of them (to :meth:`_time_tolerance`), at least one, and no
        more than ``max_steps_per_request``."""
        ratio = h / self._dt
        if not ratio <= self._max_steps + 0.5:
            raise ValueError(
                f"{what} {h!r} is {ratio:.6g} master steps of {self._dt!r}, more than "
                f"the {self._max_steps} one request may take "
                "(FmuTcpBridge(max_steps_per_request=...)); split it into several "
                "steps")
        n = int(round(ratio))
        if n < 1 or abs(h - n * self._dt) > self._time_tolerance(
                h, n * self._dt, self._dt, ulps=_STEP_SIZE_ULPS):
            raise ValueError(
                f"{what} {h!r} is not a whole multiple of the master timestep "
                f"{self._dt!r} (to within {_COMM_POINT_TOLERANCE:g} of a master step)")
        return n

    def _check_master_dt(self, md: ModelDescription) -> None:
        """Refuse a ``master_dt`` the model description contradicts.

        One sidecar step is one graph step, and the description records
        what that is (``graph_timestep``; a hand-built description has
        only ``default_step_size``).  The bridge used to store whatever
        ``master_dt`` it was given, so a uniform 0.01 s graph served with
        ``master_dt=0.005`` ran two graph steps per 0.01 s ``doStep`` and
        reported t = 0.05 s for a state at 0.10 s.  And the step the
        description advertises must be one the bridge will take, or every
        ``doStep`` at it is refused.
        """
        graph_step = getattr(md, "graph_timestep", None)
        recorded = "graph_timestep"
        if graph_step is None:
            graph_step, recorded = md.default_step_size, "default_step_size"
        graph_step = _real_number(graph_step, f"the model description's {recorded}")
        if graph_step <= 0 or abs(self._dt - graph_step) > _MASTER_DT_RTOL * graph_step:
            raise ValueError(
                f"master_dt={self._dt!r} is not the step of the graph this model "
                f"description describes ({recorded} = {graph_step!r}).  One sidecar "
                "step is one graph step, so each doStep would run a different number "
                "of graph steps than the time it reports; pass "
                "master_dt=graph_manager.timestep")
        advertised = _real_number(md.default_step_size,
                                  "the model description's default_step_size")
        if advertised <= 0:
            raise ValueError(f"the model description's default_step_size must be "
                             f"positive, got {advertised!r}")
        try:
            self._master_steps(advertised, "the model description's default_step_size")
        except ValueError as exc:
            raise ValueError(
                f"{exc}; an importer steps at the advertised size "
                "(canHandleVariableCommunicationStepSize is false), and every doStep "
                "would be refused") from None

    @staticmethod
    def _check_one_compile(md: ModelDescription, sidecar: FmuSidecar) -> None:
        """Refuse a description or a sidecar step whose graph has changed, or
        been compiled again, since it was built.

        The FMU advertises the description and runs the sidecar's step, so
        each must still be the model its graph runs.  The description
        records its graph and compile (``build_model_description``), and a
        graph's compiled step carries its own (``GraphManager.compile``);
        either may be missing -- a hand-built description, a step that is
        not a graph's -- and is then not judged.  Both built from one graph
        and both current means both are its current compile.  An FMU whose
        graph took a structural ``node.params`` write between the
        description and the bridge used to run the old model while the
        graph ran the new one.
        """
        owner = _step_compile(getattr(getattr(sidecar, "_config", None), "step_fn", None))
        if owner is not None and owner[0] is not None:
            stale = _graph_changed_since(owner[0], owner[1])
            if stale is not None:
                raise ValueError(
                    f"the sidecar's step_fn is compile {owner[1]} of its graph, and "
                    f"{stale}.  The FMU would run a model the graph does not run.  Call "
                    "compile() on the graph and build the description, the sidecar and "
                    "the bridge again")
        ref = getattr(md, "_graph", None)
        graph = ref() if ref is not None else None
        if graph is None:
            return
        generation = getattr(md, "_graph_generation", None)
        stale = _graph_changed_since(graph, generation)
        if stale is not None:
            raise ValueError(
                f"the model description was built from compile {generation} of its "
                f"graph, and {stale}.  The FMU would advertise a model the graph does "
                "not run.  Call compile() on the graph and build the description, the "
                "sidecar and the bridge again")
        # A ParamSpec changed with set_param_spec dirties nothing (specs are
        # metadata to the step), so the generation cannot see it -- but the
        # description advertises the old envelope while a sidecar built from
        # gm.param_specs() since enforces the new one.
        changed = _specs_changed_since(md, graph)
        if changed is not None:
            raise ValueError(
                f"the model description was built from compile {generation} of its "
                f"graph, and {changed}.  The FMU would advertise bounds the graph no "
                "longer declares.  Build the description, the sidecar and the bridge "
                "again from the graph as it is now")

    # ------------------------------------------------- the description's contract
    @staticmethod
    def _declared_inputs(md: ModelDescription) -> dict[tuple[str, str], Any]:
        """Every external input the described graph reads, with its zero:
        the exported input variables and the description's ``held_inputs``."""
        declared: dict[tuple[str, str], Any] = {}
        for var in md.variables:
            if var.causality == "input" and not var.is_clock:
                declared[var.node_field()] = jnp.zeros(var.shape or (), dtype=var.dtype)
        for node, field, shape, dtype in (getattr(md, "held_inputs", {}) or {}).values():
            declared.setdefault((node, field), jnp.zeros(tuple(shape), dtype=dtype))
        return declared

    @staticmethod
    def _check_sidecar_envelope(md: ModelDescription, sidecar: FmuSidecar) -> None:
        """Refuse a sidecar whose own ``ParamSpec`` for an exported parameter
        enforces another envelope than the description advertises.

        The envelope a spec enforces is read as the description reads it
        (:func:`~maddening.fmi.model_description._advertised_bound`: an open
        ``log`` / ``logit`` bound as the outermost value inside it that
        ``ParamSpec.check`` accepts), and an infinite bound pointing outward
        is no bound.  The bridge used to keep a sidecar spec as it found it,
        on the assumption that the description had been built from it; a
        ``gm.set_param_spec`` between the two broke that, and the bridge
        accepted values its XML forbids, or refused values it declares
        settable.  A parameter the sidecar has no spec for takes the
        advertised bounds (:meth:`FmuSidecar._adopt_advertised_bounds`).

        Raises
        ------
        ValueError
            Naming the parameter and both envelopes.
        """
        leaves = (sidecar.params or {}).get("nodes") or {}
        own = (sidecar.param_specs or {}).get("nodes") or {}
        for var in md.variables:
            if var.causality != "parameter":
                continue
            node, key = _param_owner(var)
            if key not in leaves.get(node, {}):
                continue
            spec = (own.get(node) or {}).get(key)
            if spec is None:
                continue
            enforced = (_advertised_bound(spec, 0, var.dtype),
                        _advertised_bound(spec, 1, var.dtype))
            if _envelope(*enforced) != _envelope(var.min, var.max):
                raise ValueError(
                    f"the sidecar's ParamSpec for {var.name!r} enforces min="
                    f"{enforced[0]!r}, max={enforced[1]!r}, but the model description "
                    f"advertises min={var.min!r}, max={var.max!r}: the FMU would accept "
                    "values its modelDescription.xml forbids, or refuse values it "
                    "declares settable.  Build the sidecar with "
                    "param_specs=gm.param_specs() from the graph the description was "
                    "built from, without changing a spec (set_param_spec) in between -- "
                    "or build the description again")

    @staticmethod
    def _advertised_bounds(md: ModelDescription,
                           sidecar: FmuSidecar) -> dict[tuple[str, str], ParamSpec]:
        """A bounds-only ``ParamSpec`` per exported parameter that
        advertises a ``min`` or ``max`` (``ValueError`` on bounds no spec
        can hold, which no description this package builds carries)."""
        leaves = (sidecar.params or {}).get("nodes") or {}
        out: dict[tuple[str, str], ParamSpec] = {}
        for var in md.variables:
            if var.causality != "parameter" or (var.min is None and var.max is None):
                continue
            node, key = _param_owner(var)
            if key not in leaves.get(node, {}):
                continue
            try:
                out[(node, key)] = ParamSpec(
                    bounds=(var.min, var.max),
                    description=f"the min / max the model description advertises "
                                f"for {var.name!r}")
            except ValueError as exc:
                raise ValueError(
                    f"the model description advertises bounds ({var.min}, {var.max}) "
                    f"for {var.name!r} that no value can satisfy: {exc}") from exc
        return out

    @staticmethod
    def _start_params(md: ModelDescription, sidecar: FmuSidecar) -> Optional[dict]:
        """The sidecar's parameter tree with every exported parameter at the
        description's ``start``, or ``None`` when it already is.

        A leaf whose start, read back in its type (:func:`_start_leaf`),
        equals the sidecar's leaf (``NaN`` equal to ``NaN``) is kept as the
        sidecar holds it.  Any other difference is warned about, naming the
        parameter, and the description's value wins.
        """
        params = sidecar.params
        if params is None:
            return None
        nodes = params.get("nodes") or {}
        changed: dict[tuple[str, str], np.ndarray] = {}
        for var in md.variables:
            if var.causality != "parameter" or var.start is None:
                continue
            node, key = _param_owner(var)
            if key not in nodes.get(node, {}):
                continue
            live = nodes[node][key]
            start = _start_leaf(var, live)
            if np.array_equal(np.asarray(live), start, equal_nan=start.dtype.kind == "f"):
                continue
            changed[(node, key)] = start
        if not changed:
            return None
        warnings.warn(
            "FmuTcpBridge: the sidecar's parameters differ from the start values "
            "its model description advertises for "
            f"{sorted(f'{n}.params.{k}' for n, k in changed)}; every FMU instance "
            "starts from the description's values (FMI 3.0), so the bridge uses "
            "them.  Build the description and the sidecar from the same "
            "parameters to silence this.",
            UserWarning, stacklevel=3,
        )
        tree = {section: {owner: dict(leaves) for owner, leaves in owners.items()}
                for section, owners in params.items()}
        for (node, key), start in changed.items():
            live = np.asarray(tree["nodes"][node][key])
            tree["nodes"][node][key] = jnp.asarray(
                _checked_value(start, live.dtype,
                               what=f"start value of {node}.params.{key}")
                if np.all(np.isfinite(start)) else start.astype(live.dtype))
        return tree

    def _reset_instance(self, op: str) -> None:
        """Put the FMU where a freshly instantiated one starts.

        The state the bridge was built over, every parameter at the start
        value the model description advertises, every input at its zero
        start value, and the time at zero (``initialize`` sets the start
        time).  Applied when a new connection claims the instance slot --
        ``fmi3InstantiateCoSimulation`` -- and on ``reset`` --
        ``fmi3Reset``, which FMI 3.0 defines as returning the instance to
        that same state.  One commit, under the lifecycle lock.

        Raises
        ------
        RuntimeError
            If the bridge has been stopped; nothing is written.
        """
        state = cast("dict[str, dict[str, Any]]", _copy_tree(self._initial_state))
        params = _copy_tree(self._initial_params)
        inputs = self._zero_inputs()
        with self._committing(op):
            self._sidecar._state = state                          # noqa: SLF001
            if params is not None:
                self._sidecar._params = params                    # noqa: SLF001
            self._inputs, self._time, self._stepped = inputs, 0.0, False
            self._t_ref, self._n_ref = 0.0, 0
            self._terminated = False

    def _refuse_if_terminated(self, op: str) -> None:
        """Refuse ``op`` (``step``, ``set``, ``initialize``) in FMI 3.0's
        Terminated state, which allows only ``fmi3Get*``, the FMU-state
        functions and ``fmi3Reset``.

        Raises
        ------
        ValueError
            If the instance has been terminated; nothing is written.
        """
        if self._terminated:
            raise ValueError(
                f"the instance has been terminated (fmi3Terminate), so {op!r} is "
                "refused: FMI 3.0's Terminated state allows reading variables, "
                "the FMU-state functions and fmi3Reset, which starts the "
                "instance again; nothing was written")

    # -------------------------------------------------------- FMU state blob
    _META = "_meta"

    def _encode_state(self) -> bytes:
        """``npz`` of plain arrays: token, time, node states, ``_meta``,
        params (nodes + mappings) and the pending inputs.  No pickle."""
        arrays: dict[str, np.ndarray] = {
            "_token": np.array(self._md.instantiation_token),
            "_time": np.array(self._time, dtype=np.float64),
        }
        for node, fields in self._sidecar.state.items():
            for f, v in fields.items():
                arrays[f"s/{node}/{f}"] = np.asarray(v)
        params = self._sidecar.params or {}
        for section in ("nodes", "mappings"):
            for owner, leaves in params.get(section, {}).items():
                for k, v in leaves.items():
                    arrays[f"p/{section}/{owner}/{k}"] = np.asarray(v)
        for node, fields in self._inputs.items():
            for f, v in fields.items():
                arrays[f"i/{node}/{f}"] = np.asarray(v)
        buf = io.BytesIO()
        # numpy types `savez` as `savez(file, *args, allow_pickle=True,
        # **kwds)`, so a checker matches every `**` value against
        # `allow_pickle: bool` as well as against `**kwds`.  Every member
        # name built above is prefixed, so none can collide with it.
        np.savez(buf, **arrays)  # pyright: ignore[reportArgumentType]
        return buf.getvalue()

    def _member_caps(self) -> dict[str, int]:
        """Every member an FMU-state archive may carry (name without the
        ``.npy`` suffix) and the most bytes it may decompress to: the live
        array it replaces plus the ``.npy`` header."""
        caps = {"_token": 256, "_time": 8}
        for n, fields in self._sidecar.state.items():
            for f, v in fields.items():
                caps[f"s/{n}/{f}"] = int(np.asarray(v).nbytes)
        for section in ("nodes", "mappings"):
            for owner, leaves in (self._sidecar.params or {}).get(section, {}).items():
                for k, v in leaves.items():
                    caps[f"p/{section}/{owner}/{k}"] = int(np.asarray(v).nbytes)
        for var in self._md.variables:
            if var.causality == "input" and not var.is_clock:
                node, field = var.node_field()
                caps[f"i/{node}/{field}"] = int(np.zeros(var.shape or (), var.dtype).nbytes)
        return {k: v + _NPY_SLACK for k, v in caps.items()}

    def _check_archive_directory(self, blob: bytes) -> None:
        """Refuse the archive from its directory alone, before any member
        is decompressed: every member (whatever its name) must be one the
        model expects and declare no more than that member may hold, and
        the total declared size is capped too (a zip bomb is a small blob
        declaring gigabytes; deflate alone gives about 1000:1)."""
        caps = self._member_caps()
        try:
            zf = zipfile.ZipFile(io.BytesIO(blob))
        except Exception as exc:  # noqa: BLE001 - BadZipFile and friends
            raise ValueError(f"FMU state blob is not a valid archive: {exc}") from exc
        with zf:
            infos = zf.infolist()
            # A state field or parameter the model does not have is refused
            # with the key-set message ``FmuSidecar.set_fmu_state`` uses for
            # the same snapshot (both sets in full), still from the
            # directory alone.
            listed = {info.filename[:-4] for info in infos if info.filename.endswith(".npy")}
            for prefix, what in (("s/", "fields"), ("p/", "parameters"), ("i/", "inputs")):
                got = {k for k in listed if k.startswith(prefix)}
                expected = {k for k in caps if k.startswith(prefix)}
                refusal = _key_set_error(what, expected, got) if got - expected else None
                if refusal is not None:
                    raise refusal
            total = 0
            for info in infos:
                name = info.filename
                key = name[:-4] if name.endswith(".npy") else None
                if key is None or key not in caps:
                    kind = {"s": "state field", "p": "parameter", "i": "input"}.get(
                        name.split("/", 1)[0], "member")
                    raise ValueError(f"FMU state carries unknown {kind} {name!r}")
                if info.file_size > caps[key]:
                    raise ValueError(f"FMU state member {key!r} is {info.file_size} bytes, more "
                                     f"than the {caps[key]} the model can hold")
                total += info.file_size
            budget = sum(caps.values())
            if total > budget:
                raise ValueError(f"FMU state declares {total} bytes in total, more than the "
                                 f"{budget} the model can hold")

    def _decode_state(self, blob: bytes) -> None:
        """Validate against the live state before writing anything.

        An archive may only install values a ``set`` of the same variables
        would be allowed to install: every restored array goes through
        :func:`checked_value` (finite, and representable in the live
        array's dtype, unless it is the value the FMU was instantiated
        with: :meth:`FmuSidecar._initial_state_leaf
        <maddening.fmi.sidecar.FmuSidecar._initial_state_leaf>`) and the
        restored parameter tree through
        the sidecar's restore check (tunability, and the declared
        ``ParamSpec`` bounds of the values it would install:
        :meth:`FmuSidecar._check_restored_params
        <maddening.fmi.sidecar.FmuSidecar._check_restored_params>`).  A snapshot
        of a diverged model -- one holding ``inf`` or ``NaN`` -- therefore
        does not restore; the error names the field.
        """
        self._check_archive_directory(blob)
        try:
            data = np.load(io.BytesIO(blob), allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"FMU state blob is not a valid archive: {exc}") from exc
        with data:
            keys = set(data.files)
            if "_token" not in keys:
                raise ValueError("FMU state belongs to a different model (schema token mismatch)")
            token = data["_token"]
            if token.dtype.kind != "U" or token.shape != () or str(token) != self._md.instantiation_token:
                raise ValueError("FMU state belongs to a different model (schema token mismatch)")
            state = {n: dict(f) for n, f in self._sidecar.state.items()}
            expected = {f"s/{n}/{f}" for n, fields in state.items() for f in fields}
            refusal = _key_set_error("fields", expected,
                                     {k for k in keys if k.startswith("s/")})
            if refusal is not None:
                raise refusal
            # Rebuilt on the live skeleton: a node with no fields has no
            # member to name it, and used to vanish from the restored state.
            new_state: dict[str, dict[str, Any]] = {n: {} for n in state}
            for node, fields in state.items():
                for field, live in fields.items():
                    # The leaf check FmuSidecar.set_fmu_state applies too: one
                    # function, so the two restore paths cannot drift apart.
                    new_state[node][field] = jnp.asarray(_restored_leaf(
                        data[f"s/{node}/{field}"], live, what=f"FMU state {node}.{field}",
                        initial=self._sidecar._initial_state_leaf(node, field)))  # noqa: SLF001
            params = self._sidecar.params
            new_params = None
            if params is not None:
                expected_params = {f"p/{section}/{owner}/{k}"
                                   for section in ("nodes", "mappings")
                                   for owner, leaves in params.get(section, {}).items()
                                   for k in leaves}
                refusal = _key_set_error("parameters", expected_params,
                                         {k for k in keys if k.startswith("p/")})
                if refusal is not None:
                    raise refusal
                new_params = {"nodes": {}, "mappings": {}}
                for section in ("nodes", "mappings"):
                    for owner, leaves in params.get(section, {}).items():
                        new_params[section][owner] = {}
                        for k, v in leaves.items():
                            key = f"p/{section}/{owner}/{k}"
                            new_params[section][owner][k] = jnp.asarray(_restored_leaf(
                                data[key], v, what=f"FMU state param {owner}.params.{k}"))
            # The pending inputs are part of the state too, and held to the
            # same key-set rule: the bridge's own snapshot always carries
            # every one, so an archive missing one is partial.  It used to
            # restore with that input silently back at zero.
            inputs: dict[str, dict[str, Any]] = self._zero_inputs()
            refusal = _key_set_error(
                "inputs", {f"i/{n}/{f}" for n, fields in inputs.items() for f in fields},
                {k for k in keys if k.startswith("i/")})
            if refusal is not None:
                raise refusal
            for k in keys:
                if k.startswith("i/"):
                    _, node, field = k.split("/", 2)
                    var = next((v for v in self._md.variables if v.name == f"{node}.{field}"
                                and v.causality == "input"), None)
                    if var is None:
                        raise ValueError(f"FMU state carries unknown input {node}.{field}")
                    arr = data[k]
                    if tuple(arr.shape) != tuple(var.shape or ()):
                        raise ValueError(f"FMU state input {node}.{field}: bad shape {arr.shape}")
                    inputs.setdefault(node, {})[field] = jnp.asarray(
                        checked_value(arr, var.dtype,
                                      what=f"FMU state input {node}.{field}")
                    )
            # The time is part of the state: an archive without it used to
            # restore at t = 0 whatever the state was, and one storing it
            # as a string ("5") was parsed.  It is a real scalar or refused.
            if "_time" not in keys:
                raise ValueError("FMU state carries no time (member '_time')")
            stamp = data["_time"]
            if stamp.dtype.kind not in "iuf" or stamp.shape != ():
                raise ValueError(f"FMU state time must be a real scalar, got "
                                 f"{stamp.dtype} of shape {stamp.shape}")
            t = float(stamp)
            if not np.isfinite(t):
                raise ValueError("FMU state carries a non-finite time")
            # A time the clock cannot step from: every step after the restore
            # would be refused.  None of this FMU's own snapshots holds one
            # (no step reaches such a time, nor does initialize).
            self._check_time_resolution(t, "FMU state's time")
        if new_params is not None:
            # A parameter the step cannot read may not change through the
            # archive either (``set`` cannot address it at all), and the
            # bounds the model description advertises hold for the values
            # the archive would install.  Without the second an importer
            # could restore mass = -1.0 against a declared (0.1, 10.0) and
            # the bridge would answer ok.  Both through the sidecar's own
            # restore check, the one FmuSidecar.set_fmu_state applies, so
            # the two doors refuse the same snapshots with the same words --
            # and both restore the FMU's own snapshot when a parameter was
            # instantiated outside its bounds, as GraphManager.load_state
            # restores the graph's checkpoint.
            self._sidecar._check_restored_params(new_params)  # noqa: SLF001
        # every check passed: commit
        with self._committing("set_state"):
            self._sidecar._state = new_state                  # noqa: SLF001
            if new_params is not None:
                self._sidecar._params = new_params            # noqa: SLF001
            self._inputs, self._time = inputs, t
            self._t_ref, self._n_ref = t, 0

    # ----------------------------------------------------------- vr mapping
    def _set(self, vrs: list[int], values, fmi_type: Any = None) -> None:
        """Stage and commit one ``set``; nothing is written unless every
        value reference and value in it is valid.

        Refused: a value reference named twice (which value would win is
        not defined, and the first was dropped in silence), a ``<Clock>``
        variable (its ticks are implied by time; ``fmi3SetClock`` handles
        it and there is no number to store -- a numeric set used to be
        answered ok and dropped), and, when the request carries a
        ``type``, a variable of another FMI type (:func:`_check_access_type`).
        """
        if not isinstance(vrs, (list, tuple)):
            raise ValueError("vr must be a list of value references")
        fmi_type = _requested_type(fmi_type)
        values = _flat_numbers(values)
        pos = 0
        staged: list[tuple[FMIVariable, np.ndarray]] = []
        named: set[int] = set()
        for vr in vrs:
            ref = _value_reference(vr)
            var = self._vars.get(ref)
            if var is None:
                raise KeyError(f"unknown value reference {vr}")
            if ref in named:
                raise ValueError(
                    f"value reference {ref} ({var.name}) is named more than once in "
                    "one set; which value should win is not defined, so nothing was "
                    "written")
            named.add(ref)
            _check_access_type(var, fmi_type, "Set")
            if var.is_clock:
                raise ValueError(
                    f"variable {var.name!r} is a Clock: its ticks are implied by time "
                    "and it holds no value a set could store (fmi3SetClock); nothing "
                    "was written")
            n = _size(var)
            chunk = values[pos:pos + n]
            if chunk.size != n:
                raise ValueError(f"vr {vr} ({var.name}) expects {n} values, got {chunk.size}")
            pos += n
            staged.append((var, chunk.reshape(var.shape or ())))
        if pos != len(values):
            raise ValueError(f"{len(values) - pos} trailing values without a value reference")
        param_updates: dict[str, Any] = {}
        input_updates: list[tuple[str, str, np.ndarray]] = []
        for var, arr in staged:
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"variable {var.name!r}: value must be finite")
            if var.causality == "parameter":
                param_updates[var.name] = self._in_dtype(var, arr)
            elif var.causality == "input":
                node, field = var.node_field()
                input_updates.append((node, field, self._in_dtype(var, arr)))
            else:
                raise ValueError(f"variable {var.name!r} ({var.causality}) is read-only")
        self._refuse_if_terminated("set")
        # Atomic: parameters are validated (bounds) by the sidecar first;
        # inputs are only committed once nothing can fail any more.
        with self._committing("set"):
            if param_updates:
                self._sidecar.set_params(param_updates)
            for node, field, arr in input_updates:
                self._inputs.setdefault(node, {})[field] = arr

    @staticmethod
    def _in_dtype(var: FMIVariable, arr: np.ndarray) -> np.ndarray:
        """``arr`` in the variable's dtype, refused when the dtype cannot
        hold it.  The shared check :func:`checked_value`, named for the
        FMI variable; ``set_state`` applies the same one."""
        return checked_value(arr, var.dtype, what=f"variable {var.name!r}")

    def _get(self, vrs: list[int], fmi_type: Any = None) -> np.ndarray:
        """The values of ``vrs``, flat, as float64; a request carrying a
        ``type`` may only name variables of that FMI type
        (:func:`_check_access_type`).

        Bounded by what one reply frame can carry before anything is built:
        a request naming more value references than :data:`_MAX_GET_VALUES`,
        or whose variables hold more values than that in total, is refused
        -- its reply would exceed the frame limit, so it could only ever be
        an error.  A value reference may be named more than once (FMI 3.0
        does not forbid it, and the answer is not ambiguous, as a repeated
        ``set``'s is); each distinct variable is read once and copied into
        a preallocated reply.  One 64 MiB frame naming a scalar output 33.5
        million times used to build a small array per entry before the
        reply-size check could refuse anything, holding the instance for
        about 45 s and the process past 6 GB.
        """
        if not isinstance(vrs, (list, tuple)):
            raise ValueError("vr must be a list of value references")
        fmi_type = _requested_type(fmi_type)
        if len(vrs) > _MAX_GET_VALUES:
            raise ValueError(
                f"a get of {len(vrs)} value references cannot be answered: a reply "
                f"frame carries at most {_MAX_GET_VALUES} values ({_MAX_MESSAGE} "
                "bytes of float64); split it into several gets.  Nothing was read")
        # Every entry a plain JSON integer (the C wrapper's requests): the
        # references are the entries.  Anything else is checked entry by
        # entry, in order, so the first bad one is the one named.
        refs: Any = vrs if {type(v) for v in vrs} <= {int} else None
        named: dict[int, FMIVariable] = {}
        if refs is None:
            refs = []
            for vr in vrs:
                ref = _value_reference(vr)
                if ref not in named:
                    named[ref] = self._get_variable(ref, vr, fmi_type)
                refs.append(ref)
        else:
            for ref in dict.fromkeys(refs):          # each distinct one, in order
                named[ref] = self._get_variable(ref, ref, fmi_type)
        sizes = {ref: _size(var) for ref, var in named.items()}
        scalars = all(size == 1 for size in sizes.values())
        total = len(refs) if scalars else sum(map(sizes.__getitem__, refs))
        if total > _MAX_GET_VALUES:
            raise ValueError(
                f"a get whose variables hold {total} values cannot be answered: a reply "
                f"frame carries at most {_MAX_GET_VALUES} ({_MAX_MESSAGE} bytes of "
                "float64); split it into several gets.  Nothing was read")
        params = self._sidecar.get_params() if any(
            v.causality == "parameter" for v in named.values()) else {}
        values = {ref: self._value_of(var, params) for ref, var in named.items()}
        if scalars:
            one = {ref: float(v[0]) for ref, v in values.items()}
            return np.fromiter(map(one.__getitem__, refs), dtype=np.float64, count=total)
        out = np.empty(total, dtype=np.float64)
        pos = 0
        for ref in refs:
            chunk = values[ref]
            out[pos:pos + chunk.size] = chunk
            pos += chunk.size
        return out

    def _get_variable(self, ref: int, vr: Any, fmi_type: Optional[str]) -> FMIVariable:
        """The variable a ``get`` names by ``ref`` (``vr`` as the request
        spelled it, for the message), checked against the call's type."""
        var = self._vars.get(ref)
        if var is None:
            raise KeyError(f"unknown value reference {vr}")
        _check_access_type(var, fmi_type, "Get")
        return var

    def _value_of(self, var: FMIVariable, params: dict) -> np.ndarray:
        """``var``'s current value, flat, as float64 (``params``: the
        sidecar's :meth:`~FmuSidecar.get_params`, read once per ``get``)."""
        if var.causality == "independent":
            return np.asarray([self._time], dtype=np.float64)
        if var.causality == "parameter":
            return np.asarray(params[var.name], dtype=np.float64).ravel()
        if var.causality == "input":
            node, field = var.node_field()
            val = self._inputs.get(node, {}).get(field)
            if val is None:
                val = np.zeros(var.shape or (), dtype=np.float64)
            return np.asarray(val, dtype=np.float64).ravel()
        if var.is_clock:
            return np.zeros(1, dtype=np.float64)
        node, field = var.node_field()
        return np.asarray(self._sidecar.state[node][field], dtype=np.float64).ravel()


__all__ = ["FmuTcpBridge", "PROTOCOL_VERSION", "checked_value", "decode_binary",
           "encode_binary", "recv_frame", "recv_message", "recv_raw", "send_binary",
           "send_message", "state_of", "values_of"]
