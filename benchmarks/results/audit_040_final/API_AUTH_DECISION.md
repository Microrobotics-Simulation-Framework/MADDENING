# API authentication: the decision you have to make before 0.4.0 ships

Written from the `security/api-hardening` branch (2026-09-19).  That branch
fixed what was unambiguous — the signaling WebSocket now authenticates the
client, request sizes are bounded, a non-loopback bind warns at startup, and
the "bind localhost" guidance is now in the five documents a cloud user
actually opens.  **It deliberately did not change a default and did not add
API authentication.**  Those are product calls; this page is the input.

## The situation, in facts

* The HTTP API has no authentication and no TLS.  39 route decorators,
  ~60 operations, among them `POST /cloud/launch` and `POST /cloud/teardown`
  (`api/server.py:1264,1307`), which provision and destroy paid GPU instances
  with the credentials stored on that host, and `/docs` + `/openapi.json`,
  which hand an attacker the route list.
* Every shipped path publishes it: `cloud/entrypoint.py:81` defaults
  `MADDENING_HOST=0.0.0.0`; `docker/Dockerfile.cloud` sets it again and
  `EXPOSE`s 8000; `cloud/_skypilot.py:51` runs `-p 8000:8000`;
  `cloud/launcher.py:130` defaults `JobConfig.ports=[8000]`, consumed by
  `sky.Resources(ports=...)` at `:718`, which **opens the provider's
  firewall**; `get_runpod_endpoint()` (`:344`) returns the public NAT address
  and the shipped examples drive the API over it.
* So the documented posture ("a development tool, not a production
  deployment surface") and the shipped posture are opposites, and the
  shipped one is the default.

## Option A — make a non-loopback bind an explicit opt-in

`MADDENING_HOST` defaults to `127.0.0.1`; binding anything else requires
`MADDENING_ALLOW_PUBLIC_BIND=1` or the process refuses to start.

* **Breaks:** anyone who sets `MADDENING_HOST=0.0.0.0` today and nothing
  else.  Not the image — a container bound to loopback is unreachable even
  with `-p`, so `Dockerfile.cloud` must set the opt-in, and then **the cloud
  default is exposed again**.  It also does not reach the two example
  scripts that call `uvicorn.run(app, host="0.0.0.0")` directly, nor anyone
  embedding `create_app()`.
* **Costs:** two hours, plus a migration note.
* **Buys:** an operator on a workstation can no longer expose the API by
  accident.  In the configuration that matters — the cloud launch — it buys
  nothing, because the image opts in on the user's behalf.

## Option B — bearer token, on by default whenever the bind is non-loopback

A dependency on every route; token from `MADDENING_API_TOKEN`, else
generated at startup and logged once (the Jupyter pattern); loopback binds
stay open so local development is unchanged.

* **Breaks:** every network client.  The three bundled UIs
  (`static/app.html`, `graph.html`, `render.html`) fetch and open WebSockets
  with no credential; browsers cannot set headers on a WebSocket, so
  `/ws/*` needs `?token=` or a subprotocol, and the HTML must be handed the
  token (which means the page itself needs protecting, or it leaks the
  token to whoever can GET it).  `examples/cloud/server/04_server_test.py`
  and `05_websocket_test.py` drive the public endpoint and must pass the
  token.  Any user script that talks to a RunPod endpoint today breaks with
  a 401 — a loud, diagnosable break, not a silent one.
* **Costs:** one to one and a half days.  The dependency and the token are
  an afternoon; the three HTML pages, the WebSocket carrier, the examples,
  the tests and the docs are the rest.  Note `/docs` and `/openapi.json`
  must be behind it too, and a health probe needs an exempt route.
* **Buys:** the hole is closed in the default configuration, including the
  cloud one.  It is the only option here that does.

## Recommendation

**Option B, and treat A as not worth doing on its own.**  A protects the
configuration that is not the problem and leaves the one that is.  The cost
gap is smaller than it looks: B's expensive half is the bundled browser UIs,
and they are already the part of the surface with the least claim on
stability.

Alongside B, one line I would change regardless of which option you pick:
**drop `8000` from `JobConfig.ports`' default** (`cloud/launcher.py:130`).
That default is what opens the cloud provider's firewall.  With it gone the
API is reachable over the SSH tunnel the examples could document in one
line, and a user who wants public ingress asks for it in their job config.
It removes the public reachability without building anything, and it makes
B's failure mode a closed port rather than a 401 on the open internet.

If none of this fits the release window, the honest fallback is to say so in
the release notes in the same words the code now uses at startup — the
branch has already made the exposure impossible to miss at runtime and in
the docs — and to ship `[cloud]` with the port default change alone.  What
should not ship is 0.4.0 with the release notes saying "bind to localhost"
while every shipped path binds `0.0.0.0` and opens the firewall.
