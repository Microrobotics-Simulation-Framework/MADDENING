# Audit: authentication, transport and cloud exposure — round 2   (0219b82)

Worktree `/home/nick/MSF/msf/MADDENING-wt/audit-r2-security` (detached at
`origin/release/0.4.0` = `0219b82`).  Reproducers in `repro/`, mutation
scoreboard in `mutation_results.md`.  Baseline for the surface's own tests:
`tests/api/test_bearer_auth.py tests/api/test_bundled_ui_auth.py tests/security/
tests/cloud/test_signaling_auth.py tests/cloud/test_skypilot_ports.py
tests/cloud/test_cloud_session.py tests/fmi/test_bridge_security_and_stepping.py`
-> **188 passed, 0 skipped**.

## Summary

The cryptography is real.  A passive byte-tap on the wire — a transparent TCP
pump carrying a genuine ZMTP/CURVE handshake, with a control run that reads the
plaintext — shows the state stream is unreadable with CURVE on and readable with
it off (`repro/r01`).  Every shipped default binds loopback, read back out of
`/proc/net/tcp` rather than out of the source (`repro/r07`).  `is_loopback`,
`is_routable_peer` and `address_is_loopback` fail closed on 50 adversarial
inputs.  The HTTP middleware is default-deny, so a new HTTP route is protected
without anyone remembering.

What is wrong is mostly around the edges of that, and one of it bites on the
first launch: **`CloudSession.wait_ready()` cannot succeed against a 0.4.0
container**, because its health probes hit `/graph` with no bearer token.  Next:
the same secret is the cleartext HTTP credential *and* the CURVE key, so one
sniffed request undoes the encryption; the ZAP allowlist is untested on two of
the three CURVE servers; the WebSocket surface has neither default-deny nor an
enumerating gate; and the release notes' headline claim ("the last
unauthenticated listening sockets") is contradicted by the package's own
anomaly registry.

No auth bypass and no silent plaintext downgrade was found.

---

## Findings

### HIGH — SEC-R2-01: `CloudSession.wait_ready()` can never succeed against a 0.4.0 container

**What breaks:** `CloudSession.launch()` -> `wait_ready()` returns
`fully_ready=False` with `error_stage="container"` for every launch of the
shipped container image, after burning the full 120 s retry window.

**Why it happens:** `cloud/entrypoint.py:75` reads
`MADDENING_HOST` with the default `"0.0.0.0"` and passes it to
`SimulationServer(bind_host=...)`, which is what turns the bearer token on for
every route but `/healthz` and `/viz/*`.  `cloud/session.py:308` and `:313` then
probe `http://<vm>:8000/graph` and `/graph/state` through
`cloud/_health.py:probe_http`, which sends no `Authorization` header (`_health.py:40-55`).
Both get 401; `urllib` raises `HTTPError` (a `URLError`), which `probe_http`
converts to `HealthProbeError("container", ...)`; `wait_for` retries to the
deadline and propagates; `session.py:359` maps it to `CloudStage.ERROR`.
`_skypilot.launch_vm` passes no `MADDENING_API_TOKEN` into the container
(`_skypilot.py:90-95`), so `CloudSession` has no token to present even if
`probe_http` could carry one.  `/healthz` exists for exactly this and is not used.

**Evidence:** `repro/r04_cloudsession_health_probe_401.py`, run against a real
uvicorn socket on 127.0.0.1 with `bind_host="0.0.0.0"` (what the entry point passes):

```
  probe_http(/healthz      ) -> OK
  probe_http(/graph        ) -> HealthProbeError(stage='container') HTTP Error 401: Unauthorized
  probe_http(/graph/state  ) -> HealthProbeError(stage='container') HTTP Error 401: Unauthorized
  wait_for -> HealthProbeError(stage='container') after 6.0s  => CloudStage.ERROR
```

**What would make this a non-issue:** (a) if the container bound loopback — it
does not, and `entrypoint.py:100-104` explains why it must not; (b) if
`probe_http` sent a token — it has no parameter for one; (c) if a test covered
it — every `wait_ready` test in `tests/cloud/test_cloud_session.py` uses
`MockCloudSession`, which overrides `_launch_worker`, so the real probe path is
untested.  All three checked.

**Suggested fix:** point stages 2 and 3 at `/healthz` (exempt by design, and its
200 already means "process up, this version"), and if a liveness check on the
graph is wanted, give `probe_http` an optional `headers=` and plumb the token
that the operator put in `JobConfig.envs`.  Risk: `/healthz` says nothing about
the graph, so stage 3 ("simulation ready") would become weaker than it claims —
which argues for the token route for stage 3 specifically.

---

### MEDIUM — SEC-R2-02: one sniffed HTTP request yields both CURVE keypairs

**What breaks:** `MADDENING_API_TOKEN` is simultaneously the HTTP bearer
credential — sent in cleartext on every request, because there is no TLS — and
the seed for both CURVE keypairs.  In the one deployment where CURVE is on (a
non-loopback ZMQ bind) the API is also non-loopback, so a passive observer who
sees one API request can decrypt the "encrypted" state and command streams.

**Evidence:** `repro/r02_bearer_token_yields_curve_keys.py` (recording TCP pump
on loopback standing in for the network):

```
[1] sniffed from a cleartext HTTP request: MADDENING_API_TOKEN='s3cret-operator-token'
[2] derived CURVE server public key: /u3eEguNdjLngk375R*p]H[rl1XHqb1n!}YG$X:5
[3] frame read off the CURVE-'protected' PUB socket:
    b'{"t": 0.02, "state": {"robot": {"joint_angle": 1.25, "secret_position": 42.0}}}'
```

**Why it happens:** `transport_auth.py:_secret` derives from `self.token`, which
is `api.auth.TOKEN_ENV`; `api/auth.py` transmits that same value as a bearer
header with no transport security.  The two are joined deliberately
(`docs/release_notes/v0.4.0.md:1626`, "there is one secret to manage, not two").

**What would make this a non-issue:** if the docs stated the condition.  They do
not.  `transport_auth.py:52` claims CURVE "buys confidentiality and integrity
against anyone who does not hold the token"; the release-notes table
(`v0.4.0.md:1620-1624`) puts "No -- **no TLS**" and "Yes, by CURVE" in adjacent
rows of one table keyed on the same variable and never connects them;
`MADD-ANO-015`'s `residual_risk` says "no forward secrecy: a recorded session is
readable if the token later leaks" but not that the sibling surface leaks it on
every request by design.  Checked: the SSH-tunnel topology the docs recommend is
*not* affected (both ends loopback, CURVE off, ssh carries the secrecy) — the
exposure is confined to the published-port topology, which is the one the
transport_auth docstring is written for.

**Suggested fix:** documentation, in three places — the `transport_auth` module
docstring, the release-notes table (a footnote on the shared-secret row), and
`MADD-ANO-015`'s `residual_risk`: *if you publish the API port without TLS, the
CURVE streams are readable by anyone who can see one API request; put both
behind TLS or an SSH tunnel, or use a separate secret for the transports.*  A
code fix (a second env var for the transports) would break the "one secret"
property the release argues for, so this should be a decision, not a reflex.

---

### MEDIUM — SEC-R2-03: the ZAP allowlist is untested on two of the three CURVE servers

**What breaks:** deleting `start_authenticator` from `CommandPublisher`
(`viz/network.py:314`) or from `Coordinator` (`multigpu/coordinator.py:244`)
leaves **all 36** tests in `tests/security/test_zmq_transport_auth.py` green,
while opening exactly the hole the allowlist exists to close.  The release notes
call the allowlist "part of the fix, not a refinement of it"
(`v0.4.0.md:1929`); only `NetworkRelay` has a test that would notice it going away.

**Evidence:** mutations 9 and 10 in `mutation_results.md` (36 passed each), and
`repro/r05_zap_untested_on_two_of_three_servers.py`, which runs the attacker the
relay's own test models — holds the server's CURVE public key, brings its own
keypair, never holds the token:

```
--- ZAP allowlist INSTALLED (shipped)
    command frames the attacker read : 0
    coordinator registrations it made: {}
--- ZAP allowlist REMOVED  (surviving mutation)
    command frames the attacker read : 1
    coordinator registrations it made: {'flow': 'attacker.example:5555'}
```

The coordinator case is the one `MADD-ANO-015` singles out as worse than message
injection: a stored `address` is handed to *other* workers as the peer to
SUB-connect to.

**What would make this a non-issue:** if the shipped code were missing the
allowlist — it is not; all three servers install it (`grep secure_server` shows
3 call sites, each paired).  This is a coverage finding, not a live hole.  Also
checked: the attacker needs the server public key, which CURVE does not put on
the wire; the relay's own test already treats "assume it leaked" as in scope, so
the same assumption applies here.

**Suggested fix:** parametrise the existing `own_keypair` attacker over all three
servers, or factor a single `test_a_peer_with_its_own_keypair_is_refused`
fixture across `NetworkRelay`, `CommandPublisher` and `Coordinator`.  Risk: none;
it is three more sockets on loopback in a file that already opens them.

---

### MEDIUM — SEC-R2-04: the WebSocket surface has neither default-deny nor an enumerating gate

**What breaks:** `@app.middleware("http")` (`api/server.py:644`) does not run for
WebSocket connections, so each WS handler has to remember to call
`_authorise_ws`.  The gate that is supposed to catch a forgotten one,
`test_every_route_refuses_an_anonymous_caller`, enumerates only
`starlette.routing.Route` (`tests/api/test_bearer_auth.py:110-120`) and so sees
no WebSocket route at all — while the module docstring says it "enumerates the
app's routes rather than listing them, so a route added later is covered without
anybody remembering to add it here."  The three WS paths are covered by a
hardcoded `WS_PATHS = ["/ws/state", "/ws/state/binary"]` plus one test for
`/ws/render`.

**Evidence:** adding

```python
@app.websocket("/ws/audit_leak")
async def ws_audit_leak(websocket):
    await websocket.accept()
    ...
```

to `create_app()` and running `tests/api/test_bearer_auth.py`:

```
75 passed in 1.03s
```

and, on the same app with `bind_host="0.0.0.0"` (`auth.enforced is True`):

```
HTTP routes seen by the gate: 38
WebSocket routes NOT seen by the gate: ['/ws/audit_leak', '/ws/state', '/ws/state/binary', '/ws/render']
ANONYMOUS websocket accepted, payload: {'leaked': {}}
```

**What would make this a non-issue:** if a `WebSocketRoute`-aware test existed —
`grep -rn "WebSocketRoute" tests/` returns nothing.  If the HTTP middleware
covered WS — Starlette's `BaseHTTPMiddleware` does not.  Both checked.  The
three shipped WS routes *are* all authenticated today; this is about the next one.

**Suggested fix:** extend `_http_routes` (or add a sibling) that walks
`starlette.routing.WebSocketRoute`, connects anonymously to each on a
`0.0.0.0`-bind client and asserts the handshake is refused — the same enumerate,
don't-list shape the HTTP half already has.  Then the docstring's claim becomes
true.  Risk: none.

---

### MEDIUM — SEC-R2-05: the CURVE address asymmetry is silent for the viz transports, and `WorkerClient`'s explanation of it is unreachable

**What breaks:** two separate things, both in the asymmetry the brief names.

*(a)* A relay inside a container must bind `tcp://0.0.0.0:P` — CURVE on.  A
client on the host reaching the published port over `tcp://127.0.0.1:P` gets
CURVE off.  `WorkerClient` documents this case and raises.  `NetworkReceiver` and
`CommandReceiver` raise nothing, log nothing, expose no flag, and never time out:
`latest_snapshot()` returns `(0.0, None)` for ever, which is indistinguishable
from an idle simulation.  For `CommandReceiver` that is a dead actuation path.

*(b)* `WorkerClient.register_and_wait`'s `ConnectionError` branch
(`worker_client.py:202-210`) is the **only** place in the package that explains
the asymmetry to a user ("pass `secure=True` to WorkerClient in that case").  It
is unreachable: it requires `deliverable` to stay `False`, i.e. every
`send_multipart` to raise `zmq.Again` for the whole timeout.  A DEALER that
`connect()`s gets its pipe immediately, so the first 1000 sends (the default
`ZMQ_SNDHWM`) are accepted regardless of the handshake.  Measured: exactly 1000
sends accepted before the first `EAGAIN`.  With a 250 ms `RCVTIMEO` the loop
sends about 4 times a second, so reaching the branch needs a timeout of ~250 s.
Every realistic call gets `TimeoutError("No ACK from coordinator ...")`, which
says nothing about CURVE.

**Evidence:** `repro/r03_receiver_silent_on_curve_mismatch.py`:

```
NetworkRelay(0.0.0.0) <- NetworkReceiver(127.0.0.1):
  relay.secure = True   receiver secure = False
  after 6s of publishing, latest_snapshot() = (0.0, None)
  exceptions raised: none
  log records mentioning the mismatch: []

CommandPublisher(0.0.0.0) <- CommandReceiver(127.0.0.1):
  after 4s, latest_commands() = None  (control input silently dead)

Coordinator(0.0.0.0) <- WorkerClient(127.0.0.1)  [the documented one]:
  worker raised TimeoutError: No ACK from coordinator at 127.0.0.1:55463 within 3s...
```

and, five trials of the exact scenario the test models, all `TimeoutError`; plus

```
sends accepted before first EAGAIN: 1000  (elapsed 1.00s, EAGAIN=1)
```

**What would make this a non-issue:** if `test_a_worker_without_curve_does_not_hang_on_a_curve_coordinator`
pinned the message — it accepts `pytest.raises((ConnectionError, TimeoutError))`,
so it passes either way and cannot see that the diagnostic never runs.  If the
viz classes documented the case — `NetworkReceiver`/`CommandReceiver` docstrings
describe the address rule and never mention the asymmetry, which only
`WorkerClient`'s does.  Both checked.

**Suggested fix:** (a) give the SUB-side classes a `secure` mismatch signal — the
cheapest is a `ZMQ_EVENT_HANDSHAKE_FAILED_PROTOCOL` monitor socket, or, failing
that, a one-shot warning after N seconds with no frame naming the CURVE
asymmetry; (b) fold the CURVE explanation into the `TimeoutError` message too (or
decide on it by `self._secure` rather than by which branch was taken), and change
the test to assert on the *text*, not the type.  Risk for (a): a warning on a
genuinely idle publisher; gate it on `self._secure != <what the peer appears to
want>` being undecidable and word it as a hint.

---

### MEDIUM — SEC-R2-06: the loopback threat model omits the browser

**What breaks:** on a loopback bind the API is unauthenticated "exactly as it
always was", and the docs state the resulting threat model as "the caller is
anyone with a shell on this box" (`api/server.py:466-467`).  That omits every web
page the developer's browser loads.  There is no CORS middleware and no `Origin`
check, so a cross-origin *simple* request (`POST`, `Content-Type: text/plain`)
from any page reaches the handler.  The peer backstop cannot help: the peer *is*
127.0.0.1, so `is_routable_peer` is `False` by construction.  With DNS rebinding
the same is reachable from an arbitrary remote attacker.

**Evidence:** loopback-bound app, every request carrying
`Origin: https://evil.example` and `Content-Type: text/plain;charset=UTF-8`:

```
  POST  /sim/reset                           -> 200  CORS header present: False
  POST  /sim/start                           -> 200  CORS header present: False
  POST  /sim/stop                            -> 200  CORS header present: False
  POST  /graph/compile                       -> 200  CORS header present: False
  POST  /checkpoint/save?path=csrf.npz       -> 200  CORS header present: False
  POST  /sim/run?n_steps=1                   -> 200  CORS header present: False
  POST  /sim/profile/jax/start               -> 200  CORS header present: False
```

`checkpoints/csrf.npz` was really written.  (Removed afterwards; the worktree is
clean.)

**What would make this a non-issue:** the attacker cannot *read* the responses
(no `Access-Control-Allow-Origin`), so this is write-only: start/stop/reset a
simulation, write a file under `checkpoint_root`, or ask for a 100,000-step run.
`POST /cloud/launch` needs a JSON body, so it preflights and is blocked.  Checked
both.  That is why this is MEDIUM and not higher.

**Suggested fix:** the cheap, complete fix is to reject a request carrying an
`Origin` header that is not the server's own origin — a loopback-bound API has no
legitimate cross-origin caller, and the bundled pages are same-origin.  Risk: an
embedder serving the UI from a different port would break; make it a constructor
flag with the safe default.

---

### MEDIUM — SEC-R2-07: the release notes' headline security claim is false, and one shipped module contradicts it

**What breaks:** `docs/release_notes/v0.4.0.md` says, twice (lines 1612 and 1868),
that the ZeroMQ transports were "the last unauthenticated listening sockets in
MADDENING", and (line 1889) that "every socket now defaults to a loopback bind".
Neither holds:

* `maddening.fmi.tcp_bridge.FmuTcpBridge` binds and listens with its own
  unauthenticated protocol (`tcp_bridge.py:440-443`).  `MADD-ANO-015`'s own
  `residual_risk` says so; the release notes say the opposite.
* `maddening.examples.servers.vessel_flow_server` is an installed module,
  runnable as `python -m`, that builds its own FastAPI app with no credential
  and whose `--host` still defaults to `"0.0.0.0"`
  (`vessel_flow_server.py:409`).  It prints a warning and serves anyway.  Every
  other shipped example server moved to `host="127.0.0.1"` in this release.
* `SelkiesSession` hardcodes `websockets.serve(handler, "0.0.0.0", 8443)`
  (`selkies_session.py:390`) with no bind parameter, so its socket cannot be made
  loopback-only even deliberately.  (It *is* HMAC-authenticated and fails closed
  when `MADDENING_STREAM_SECRET` is unset, so this is exposure, not a bypass.)

**Evidence:** `grep -rn "\.bind(\|\.listen(\|serve(\|uvicorn.run" src/maddening/`
gives six listening surfaces, not three; `repro/r07_default_binds_over_real_sockets.py`
confirms the ZMQ and FMI defaults *are* loopback, read from `/proc/net/tcp`:

```
  NetworkRelay()               listening on 127.0.0.1:5555   (loopback only)
  CommandPublisher()           listening on 127.0.0.1:5556   (loopback only)
  Coordinator()                listening on 127.0.0.1:5580   (loopback only)
  FmuTcpBridge(defaults)       listening on 127.0.0.1:35463  (loopback only)
```

**What would make this a non-issue:** if "in the package" meant "among the
transports this section is about" — it does not; the sentence is the section's
opening claim and the note above it says "Three listening surfaces ... one rule".
This is the headline security claim of a release going to PyPI, and the
package's own regulatory registry contradicts it.

**Suggested fix:** reword to "the last unauthenticated *network* transports"; add
`FmuTcpBridge` as a fourth row of the table with "trusted clients only, loopback
by default, no credential"; give `SelkiesSession` a `bind_host="127.0.0.1"`
parameter; change `vessel_flow_server`'s `--host` default to `127.0.0.1`.  Risk:
the `vessel_flow_server` default change breaks anyone running the demo from
another machine — which is the point, and the warning it already prints says so.

---

### LOW — SEC-R2-08: `MADD-ANO-015`'s residual risk misstates the FMI bridge default

`residual_risk` says "maddening.fmi.tcp_bridge listens on TCP 5555 by default".
The constructor defaults are `host="127.0.0.1", port=0` (`tcp_bridge.py:422-423`);
`5555` appears only in a docstring example in `fmi/package.py:14-16`.  Measured
(`repro/r07`): `FmuTcpBridge default endpoint = 127.0.0.1:35463`.  The error is in
the conservative direction, and the rest of the scope statement — a different
component, not covered by the entry, documented trusted-clients-only
(`tcp_bridge.py:85-86`) — is correct.  **Fix:** "listens on a loopback TCP port
(ephemeral by default) with its own unauthenticated protocol".

### LOW — SEC-R2-09: two shipped cloud examples put the API token into a process argv

`examples/cloud/server/04_server_test.py:254` and `05_websocket_test.py:232` build
`f"MADDENING_API_TOKEN={shlex.quote(API_TOKEN)} {PYTHON} ..."` and hand it to
`CloudJob.ssh_run_background`, which wraps it in `nohup bash -c '<command>'`
(`launcher.py:347`) and passes the whole string to `ssh` as one argv element.
`shlex.quote` stops word splitting; it does nothing about visibility.  Measured
(`repro/r06`): the wrapping shell's `/proc/<pid>/cmdline` is mode `0444` and
contains the token.  The window is short (the remote `bash` execs the interpreter
away, after which the token is only in `/proc/<pid>/environ`, owner-readable), and
the local `ssh` process holds it for the invocation.  `MADDENING_API_TOKEN_FILE`
exists precisely for out-of-band delivery, and `ssh_run`/`ssh_run_background`
offer no way to set an environment variable other than in the command string —
that is the library-level gap.  **Not higher because:** examples, brief window,
and the examples' own comment says "a demo on a throwaway VM rather than a
deployment pattern".

### LOW — SEC-R2-10: no minimum token strength, and the transport token is always operator-chosen

`APIAuth` generates 256 bits when the variable is unset, but `TransportAuth`
cannot — both ends must hold the same value — so the CURVE key is *always* a
human-chosen string, and nothing checks its length or entropy.  Measured
(`repro/r08`): `"a"`, `"1"` and `"password"` are accepted by both classes; blank
is correctly refused by both.  The derivation is a single unsalted, uniterated
BLAKE2b, so an attacker who learns the CURVE server public key recovers the token
offline at ~20,000 candidates/s single-threaded in pure Python (a C
implementation is orders faster).  Domain separation itself is sound: the two
role literals differ at byte 0 and contain no NUL, so no (role, token) pair can
collide — 16 pairs, 16 distinct keypairs, no collisions.  **Fix:** refuse a token
shorter than, say, 16 characters with a message pointing at
`secrets.token_urlsafe(32)`; optionally iterate the derivation.

### LOW — SEC-R2-11: the key derivation has no pinned vector

Changing `_CURVE_PERSON` from `b"maddening-curve"` to `b"maddening-CURVE"` leaves
all 36 security tests green (mutation 12).  `TestKeyDerivation` checks
*self-consistency*, which is automatic when both ends run the same code; nothing
pins the derivation across versions.  A refactor of the role strings, the
separator or the hash would silently make 0.4.x and 0.5.x unable to talk to each
other, and the failure mode is SEC-R2-05's silent one.  **Fix:** one golden
vector — `TransportAuth(token="the-shared-token").server_keypair() == (b"...", b"...")`.

### LOW — SEC-R2-12: `probe_zmq` can never succeed against a secured relay

`cloud/_health.py:71-73` creates a plain `zmq.SUB` with no CURVE configuration.
Against a relay that published its port (and therefore has CURVE on) it receives
nothing — the same measurement as `repro/r01`/`r05`'s no-credential subscriber,
0 frames — so stage 5 always spends its 30 s and fails.  The failure is swallowed
(`session.py:344`), so the cost is latency, not correctness.  **Fix:** either give
`probe_zmq` the token, or replace it with a TCP connect probe, or drop the stage
the way 8080's was dropped.

### LOW — SEC-R2-13: a generated token is never announced when only the peer backstop is in play

`APIAuth.announce` returns early when `not self.enforced`, and `_write_token_file`
is only called from inside it.  So `uvicorn.run(app, host="0.0.0.0")` without
`bind_host` or `MADDENING_HOST` — the exact case rule 2 exists for — produces a
server that correctly refuses remote callers with a token that was never logged
and never written to `MADDENING_API_TOKEN_FILE`.  Fail-closed, and the 401 body
explains the fix ("pass bind_host ... so the token is logged at start-up"), so
this is recoverable rather than dangerous.  **Fix:** when the token was generated
and `announce` declines because the bind looks loopback, log one INFO line saying
a token was generated and how to make it visible.

---

## Unverified suspicions

* **`_skypilot.launch_vm` interpolates `config.container_image` into a shell
  command line unquoted** (`_skypilot.py:92-94`), while `ports` next to it is
  validated precisely because it is interpolated.  *What would make this a
  non-issue:* the image name comes from the operator's own YAML, so it is
  self-inflicted, and `JobConfig.from_yaml` has no untrusted source in the
  shipped flows.  **Checked** — I found no path where a third party supplies
  `container_image`.  Reported only because the neighbouring field was hardened
  and this one was not.
* **The WebRTC signaling server accepts `?token=` on the request target**
  (`selkies_session.py:106-110`), which is the exact carrier `api/auth.py`
  refuses by design ("no authenticated HTTP route accepts a token in the query
  string, so a credential never reaches an access log").  The header form is
  preferred and tried first, and the `websockets` library writes no access log,
  so nothing in *this* package logs it — but a reverse proxy would, and
  `static/webrtc_client.html:34` advertises the query form in its placeholder.
  *What would make this a non-issue:* a deployment with no proxy in front of
  8443.  **Not checked** beyond confirming `websockets` has no access log; I did
  not test a proxy.
* **Signaling tokens never expire and are not bound to a connection.**
  `generate_session_token` is `HMAC(secret, session_id)` with no nonce, no
  timestamp and no replay window, so anyone who ever sees the URL has permanent
  access for the life of the session.  *What would make this a non-issue:* the
  session id is regenerated per `start()`, so the blast radius is one streaming
  session.  **Checked** that much; I did not judge whether that is acceptable for
  the intended use.
* **`JobConfig._with_envs` passes `ports=self.ports`, sharing the list object**
  with the original config (`launcher.py:160`).  `CloudGroup._inject_rank0_env`
  rebinds rather than mutating, so today nothing aliases.  *What would make this
  a non-issue:* exactly that — **checked**; a future in-place `.append()` would
  silently open a port on every job built from the same spec.
* **`CloudGroup` publishes the coordinator port unconditionally**
  (`group.py:200-204`) while `Coordinator`'s default bind is now loopback, so the
  firewall hole leads to a socket that is not listening on it.  Harmless today,
  and the multi-VM example passes `bind_host="0.0.0.0"` deliberately, but the two
  halves now disagree about who decides exposure.  **Not reproduced** — it needs
  a real provider.

---

## What I checked and found sound

* **CURVE really encrypts, proved with a tap that can read the plaintext.**
  `repro/r01` puts a transparent TCP byte-pump between a real, authorised SUB and
  the relay, so a genuine ZMTP/CURVE handshake and every frame cross it:
  `secure=False` -> 253 bytes captured, payload readable; `secure=True` -> 513
  bytes captured, payload not readable; the SUB received 2 frames in both runs,
  so neither half is vacuous.  This is the non-vacuous version of the test the
  implementer warned about.
* **Every shipped default binds loopback**, read from `/proc/net/tcp` for this
  process's own listening sockets rather than from the source (`repro/r07`):
  `NetworkRelay` 127.0.0.1:5555, `CommandPublisher` 127.0.0.1:5556, `Coordinator`
  127.0.0.1:5580, `FmuTcpBridge` 127.0.0.1:ephemeral.  None on 0.0.0.0 or `::`.
* **Address classification fails closed** on 28 host strings and 22 ZMQ
  endpoints, including `127.1`, `0177.0.0.1`, `2130706433`, `tcp://lo:5555`,
  `tcp://evil.example#127.0.0.1:5555`, `tcp://` and `""` — every ambiguous case
  resolves to "requires a credential".  `::ffff:127.0.0.1` and `::ffff:7f00:1`
  are correctly loopback; `::ffff:10.0.0.4` and `fe80::1%eth0` correctly are not.
* **The HTTP middleware is default-deny.**  My attempt to seed "a new HTTP route
  with no auth" was not a mutation at all: the path-based middleware 401s it
  without the author doing anything.  That is the design working.
* **Gate mutation scoreboard: 10 caught, 3 missed** (`mutation_results.md`).
  Every gate that fired named the right thing — `required_for_peer` `or`->`and`
  listed all 33 exposed routes; the exempt-set widening was caught by the literal
  frozenset assertion, not silently absorbed; dropping the relay's ZAP handler
  produced "38 == 0" on the own-keypair attacker.  The three misses are
  SEC-R2-03, SEC-R2-04 and SEC-R2-11.
* **`_checkpoint_path`** resolves and rejects anything not under
  `checkpoint_root`, including absolute paths and symlink escapes
  (`server.py:992-1000`).
* **`UNAUTHENTICATED_PATHS` is pinned as a literal frozenset**, so widening it is
  a two-place, visible act; and `test_no_bundled_page_is_left_out_of_this_module`
  greps `server.py` for `_STATIC_DIR / "*.html"` so a fourth viz page cannot slip
  past the UI-auth contract tests.  `vessel_flow.html` and `webrtc_client.html`
  are correctly excluded (neither is served by `SimulationServer`).
* **Key-derivation domain separation is sound** — 16 (role, token) pairs, 16
  distinct keypairs, and the construction `role + b"\x00" + token` cannot collide
  because the two role literals differ at byte 0 and contain no NUL.
* **The `#token=` fragment does not reach the access log and `?token=` does**,
  exactly as `static/auth.js` documents.  Measured on a real uvicorn access log:
  `GET /viz/app HTTP/1.1` for the fragment form,
  `GET /viz/app?token=TOKEN123 HTTP/1.1` for the query form.  No *authenticated*
  route accepts a query-string token, so the only way to get one into the log is
  to use the form the docs tell you not to.
* **All three CURVE servers install the ZAP allowlist** (`grep secure_server`:
  three call sites, each immediately preceded by `start_authenticator`).  The
  finding above is about the tests, not the code.
* **The `secure=False`-on-a-routable-address downgrade is genuinely refused**, in
  the constructor rather than in a background thread, for both `NetworkRelay` and
  `Coordinator`; and `resolve_security` raises rather than warning.  No path was
  found where a routable bind ends up in cleartext.
* **`_port_flags` validates ports as integers in 1..65535** and the four
  skypilot gates catch reinstating the hardcoded `-p 5555:5555 -p 5556:5556`
  (mutation 14).
* **The signaling auth gate is real**: neutering `validate_session_token` fails 7
  tests (mutation 15), and an unset `MADDENING_STREAM_SECRET` generates a random
  one so every external client is rejected — fail-closed, with a warning.
