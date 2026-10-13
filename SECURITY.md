# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in MADDENING, please report it responsibly:

**Email**: nick@microrobotica.org

Please include:
- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Any suggested fix (optional)

## Response Timeline

- **Acknowledgement**: within 3 business days
- **Initial assessment**: within 7 business days
- **Fix or mitigation**: best effort, prioritised by severity

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.4.x   | Yes       |
| 0.3.x   | Security fixes only |
| < 0.3   | No        |

## Dependency Monitoring

MADDENING monitors its core dependencies for known security vulnerabilities using two mechanisms:

1. **GitHub Dependabot alerts** — enabled on the repository; they flag a dependency with a published advisory (CVE / GHSA). Automated security-fix pull requests are **not** enabled and the repository has no `dependabot.yml`, so nothing opens a pull request by itself: an alert is acted on by hand.

2. **Manual changelog review** — JAX ecosystem libraries (JAX, jaxlib, Equinox, Optax) are reviewed at each MADDENING release for correctness-affecting changes (XLA compiler changes, numerical behaviour changes) that may not be classified as security vulnerabilities.

Security-relevant dependency updates are flagged in the `Security` section of `CHANGELOG.md`.

## Scope

MADDENING is primarily a computation library, but it is **not only** a
computation library: the `api`, `network` and `cloud` extras ship
listening sockets, and the `cloud` path provisions paid GPU instances.
Deployments that install those extras have a network attack surface and
are in scope.

The primary security concerns are:

- **Supply chain**: malicious or compromised dependencies
- **Numerical correctness**: silent computation errors (see `known_anomalies.yaml`)
- **Denial of service**: pathological inputs that cause excessive memory or compute usage
- **Network surfaces** (only with the `api`, `network` or `cloud` extras):
  the HTTP/WebSocket API on 8000, the WebRTC signaling socket on 8443,
  and the ZeroMQ transports on 5555 (state), 5556 (commands) and 5580
  (multi-job coordinator).  Separately, the FMU TCP bridge
  (`maddening.fmi.tcp_bridge`, base install) listens on an ephemeral
  loopback port while an FMU is served

### The rule for listening sockets, and its one exception

**A loopback bind is unauthenticated; any other bind demands a
credential.** Each surface has its own:

- **HTTP API: `MADDENING_API_TOKEN`**, as a bearer token. Unset, the
  server generates one at start-up and logs it once.
- **ZeroMQ transports: `MADDENING_TRANSPORT_TOKEN`**, the seed for their
  CURVE keypairs. `TransportAuth` reads it first and falls back to
  `MADDENING_API_TOKEN` only when it is unset. With neither, a socket on
  a reachable address refuses to open rather than falling back to
  cleartext.
- **WebRTC signaling: `MADDENING_STREAM_SECRET`**. Unset, the session
  uses a random secret nobody holds, and every client is rejected.

**Set `MADDENING_API_TOKEN` and `MADDENING_TRANSPORT_TOKEN` both, to
different values**, whenever the API port and a ZeroMQ port are reachable
from the same network. The API has no TLS, so its token crosses the
network in cleartext on every request; while that token is also the
CURVE seed, one observed request yields both CURVE keypairs and the
"encrypted" streams can be read (`MADD-ANO-015`). A `MADDENING_API_TOKEN`
or `MADDENING_TRANSPORT_TOKEN` that is set but blank is a configuration
error and refuses to start; an empty `MADDENING_STREAM_SECRET` is read as
unset.

**The FMU TCP bridge is the exception: it authenticates nobody, on any
bind.** Any process that can reach its port can read, write, step and
hold the model. It defaults to loopback, and it takes a non-loopback
`host` without a check or a warning. Keep it on loopback, on a
single-user host or in its own network namespace, for trusted clients
only; see `MADD-ANO-023`.

**There is no TLS on the HTTP API.** Its bearer token crosses the network
in cleartext, so it belongs behind an SSH tunnel or a TLS-terminating
proxy. The ZeroMQ transports are encrypted end to end by CURVE whenever
they are not on loopback.

Before v0.4.0 the ZeroMQ transports bound every interface with no
authentication or encryption at all; see `MADD-ANO-015` in
`docs/validation/known_anomalies.yaml`.  The HTTP API authenticated
nobody either (`MADD-ANO-051`), its checkpoint routes took any server path
(`MADD-ANO-052`), and the signaling socket's token check admitted every
client (`MADD-ANO-053`).

MADDENING assumes trusted inputs (Section 2 of `docs/regulatory/intended_use.md`). Input sanitization and validation is the responsibility of the downstream integration layer.
