# Mutation test results — auth / transport / cloud-exposure gates

Harness: `repro/mutate_harness.py` (applies one seeded fault, runs a test
selection, reverts with `git checkout --`).  Worktree
`/home/nick/MSF/msf/MADDENING-wt/audit-r2-security` @ 0219b82.
Baseline over the whole selection: **188 passed, 0 skipped, 27 s**.

| # | Mutation | Target suite | Caught? | Test that named it |
|---|---|---|---|---|
| 1 | new `@app.get("/audit/leak")` HTTP route with no auth | test_bearer_auth | **n/a — not a hole** | the middleware is default-deny, so the added route 401s on its own; 75 passed correctly |
| 2 | new `@app.websocket("/ws/audit_leak")` with no `_authorise_ws` | test_bearer_auth | **NO** | 75 passed; anonymous WS accepted on a 0.0.0.0 bind (finding SEC-R2-04) |
| 3 | `required_for_peer`: `or` -> `and` | test_bearer_auth | yes | `test_every_route_refuses_an_anonymous_caller` (33 routes listed) |
| 4 | `is_routable_peer` -> `return False` | test_bearer_auth | yes | `test_a_routable_peer_is_challenged_even_on_a_believed_loopback_bind` |
| 5 | `UNAUTHENTICATED_PATHS` += `/graph` | test_bearer_auth | yes | 14 tests, first `test_the_exempt_paths_are_exactly_these` (the literal frozenset assertion) |
| 6 | `APIAuth.verify` -> `return True` | test_bearer_auth | yes | `test_every_route_refuses_an_anonymous_caller` |
| 7 | `_AllowOnly.callback` -> `return True` | test_zmq_transport_auth | yes | `test_a_subscriber_with_its_own_keypair_cannot_read_the_state_stream` (38 frames read) |
| 8 | drop `start_authenticator` from **NetworkRelay** | test_zmq_transport_auth | yes | same test (38 frames read) |
| 9 | drop `start_authenticator` from **CommandPublisher** | test_zmq_transport_auth | **NO** | 36 passed (finding SEC-R2-03) |
| 10 | drop `start_authenticator` from **Coordinator** | test_zmq_transport_auth | **NO** | 36 passed (finding SEC-R2-03) |
| 11 | `_secret`: drop the `role` prefix | test_zmq_transport_auth | yes | `test_the_server_and_client_keys_are_independent` |
| 12 | `_CURVE_PERSON` changed | test_zmq_transport_auth | **NO** | 36 passed — no pinned key vector (finding SEC-R2-11) |
| 13 | `resolve_security(addr, None)` -> `return False` | test_zmq_transport_auth | yes | 7 tests, incl. `test_a_wildcard_bind_turns_encryption_on_by_itself` |
| 14 | reinstate hardcoded `-p 5555:5555 -p 5556:5556` | test_skypilot_ports | yes | 4 tests, incl. `test_launch_vm_never_publishes_the_zmq_ports_unasked` |
| 15 | signaling handler stops validating the token | test_signaling_auth | yes | 7 tests, incl. `test_no_token_is_rejected` |

**Score: 10 caught, 3 missed, 1 non-mutation, 1 (the `person=` change) a
compatibility rather than a security gap.**  Every gate that fired named the
right thing; none failed with an unrelated message.
