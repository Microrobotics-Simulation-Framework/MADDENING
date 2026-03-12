# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in MADDENING, please report it responsibly:

**Email**: nicholas.ehsan.roy@gmail.com

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
| 0.1.x   | Yes       |

## Dependency Monitoring

MADDENING monitors its core dependencies for known security vulnerabilities using two mechanisms:

1. **GitHub Dependabot** — enabled on the repository; automatically monitors PyPI dependencies for published CVEs and creates pull requests for security-relevant version bumps.

2. **Manual changelog review** — JAX ecosystem libraries (JAX, jaxlib, Equinox, Optax) are reviewed at each MADDENING release for correctness-affecting changes (XLA compiler changes, numerical behaviour changes) that may not be classified as security vulnerabilities.

Security-relevant dependency updates are flagged in the `Security` section of `CHANGELOG.md`.

## Scope

MADDENING is a computation library, not a network service. The primary security concerns are:

- **Supply chain**: malicious or compromised dependencies
- **Numerical correctness**: silent computation errors (see `known_anomalies.yaml`)
- **Denial of service**: pathological inputs that cause excessive memory or compute usage

MADDENING assumes trusted inputs (Section 2 of `docs/regulatory/intended_use.md`). Input sanitization and validation is the responsibility of the downstream integration layer.
