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

1. **GitHub Dependabot** — enabled on the repository; automatically monitors PyPI dependencies for published CVEs and creates pull requests for security-relevant version bumps.

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
  (multi-job coordinator)

### The rule every listening socket follows

**A loopback bind is unauthenticated; any other bind demands a
credential.** One credential, `MADDENING_API_TOKEN`, covers the HTTP API
(as a bearer token) and the ZeroMQ transports (as the seed for their
CURVE keypairs). The signaling socket has its own,
`MADDENING_STREAM_SECRET`. No surface falls back to cleartext when its
credential is missing -- it refuses to start.

**There is no TLS on the HTTP API.** Its bearer token crosses the network
in cleartext, so it belongs behind an SSH tunnel or a TLS-terminating
proxy. The ZeroMQ transports are encrypted end to end by CURVE whenever
they are not on loopback.

Before v0.4.0 the ZeroMQ transports bound every interface with no
authentication or encryption at all; see `MADD-ANO-015` in
`docs/validation/known_anomalies.yaml`.

MADDENING assumes trusted inputs (Section 2 of `docs/regulatory/intended_use.md`). Input sanitization and validation is the responsibility of the downstream integration layer.
