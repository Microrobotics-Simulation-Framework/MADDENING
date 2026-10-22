# Downstream Integration — Dependency Chain and Responsibilities

## Four-Layer Dependency Chain

```
Layer 1: MADDENING          (open source, Apache-2.0, general-purpose framework)
    ↓
Layer 2: MIME                (open source, Apache-2.0 from 0.3.0, microrobotics physics engine)
    ↓
Layer 3: MICROROBOTICA       (open source, AGPL-3.0, research simulator)
    ↓
Layer 4: [Commercial Product] (regulated, CE-marked, built by commercial entity)
```

| Layer | Status | Regulatory Obligation |
|-------|--------|----------------------|
| MADDENING | Open-source research tool | None (provides {term}`SOUP` documentation voluntarily) |
| MIME | Open-source research tool | None (provides SOUP documentation voluntarily) |
| MICROROBOTICA | Open-source research tool | None (provides SOUP documentation voluntarily) |
| Commercial Product | Regulated medical device | Full {term}`EU MDR` manufacturer obligations |

## Per-Layer Responsibility Details

### Layer 1: MADDENING

**Role**: General-purpose {term}`JAX`-based multiphysics simulation framework.

**Provides**:

- Core simulation infrastructure: graph management, {term}`JIT compilation`, {term}`multi-rate <Multi-rate>` scheduling, adaptive timestepping
- Physics {term}`node <Node>` library: heat diffusion, rigid body dynamics, spring-damper, {term}`LBM` fluid
- Neural {term}`surrogate <Surrogate>` framework: training, validation, architecture library
- Compliance infrastructure: {codeterm}`NodeMeta`, {term}`anomaly registry <Anomaly registry>`, {term}`verification benchmark <Verification benchmark>` registry, `@stability` decorator, `HealthCheckNode`
- {term}`SOUP` documentation package (`docs/validation/soup_package.md`)

**Does NOT provide**:

- Clinical claims or intended clinical use
- Domain-specific physics for microrobotics ({term}`CSF` flow, Brownian motion, magnetic actuation)
- Risk management file ({term}`ISO 14971`) — only `hazard_hints` for downstream risk analysis
- QMS, clinical evaluation, or post-market surveillance

**SOUP classification**: When used in a regulated product, MADDENING is {term}`IEC 62304` SOUP. Its safety class is determined by the downstream manufacturer based on the context of use.

### Layer 2: MIME (MIcrorobotics Multiphysics Engine)

**Role**: Domain-specific physics engine for microrobot simulation, built on MADDENING.

**Provides**:

- {term}`CSF` flow simulation nodes (wrapping or extending LBMPipeNode)
- Magnetic field and actuation models
- Particle dynamics for microrobots (Brownian, drag, contact)
- Domain-specific `HealthCheckNode` configurations (e.g., CSF density bounds, velocity limits)
- Own SOUP documentation, anomaly registry, and verification benchmarks (using `MIME-` prefix)

**Inherits from MADDENING**:

- Compliance schema types (`NodeMeta`, `AnomalyRecord`, etc.) via `maddening.compliance`
- Anomaly registry validator CLI
- `@verification_benchmark` and `@stability` decorators
- `HealthCheckNode` base class

**SOUP classification**: When used in a regulated product, MIME is IEC 62304 SOUP. MADDENING is MIME's own SOUP dependency (SOUP-of-SOUP).

### Layer 3: MICROROBOTICA

**Role**: Open-source research simulator for microrobot-assisted drug delivery in CSF.

**Provides**:

- Pre-configured simulation scenarios for cerebrospinal fluid geometries
- Visualization and analysis tools for researchers
- Training data pipelines for reinforcement learning research
- Own SOUP documentation, anomaly registry, and verification benchmarks (using `MICROROBOTICA-` or `MBOT-` prefix)

**Inherits from MIME**: All MIME compliance infrastructure, plus MIME-specific physics and verification evidence.

**SOUP classification**: When used in a regulated product, MICROROBOTICA is IEC 62304 SOUP.

### Layer 4: Commercial Product

**Role**: CE-marked medical device software built by a downstream commercial entity.

**This is the ONLY layer subject to EU MDR.** The commercial entity is the EU MDR manufacturer (Article 2(30)).

## Commercial Responsibility Statement

The Layer 4 commercial entity is the EU MDR manufacturer (Article 2(30)) and is solely responsible for all regulatory obligations, including but not limited to:

- Establishing and maintaining an **{term}`ISO 13485` QMS** certified by a {term}`Notified Body`
- Performing the **conformity assessment** procedure appropriate to the device class (Class IIb: Annex IX or Annex X+XI; Class III: Annex IX or Annex X+XI with full technical file review)
- Engaging and satisfying the **Notified Body** for initial conformity assessment and annual surveillance audits
- Building and maintaining the **clinical evaluation report** (CER) with sufficient clinical evidence
- Operating **post-market surveillance** including PSUR (annual for Class III), vigilance reporting, and PMCF studies
- Bearing **manufacturer liability** for the device placed on the EU market
- Performing **SOUP assessment** of all open-source layers (MADDENING, MIME, MICROROBOTICA) per IEC 62304 Clause 5.3.3/5.3.4
- Maintaining **risk management file** (ISO 14971) incorporating SOUP failure modes

MADDENING provides SOUP documentation to support the commercial entity's regulatory obligations. This provision is voluntary and does not make MADDENING subject to EU MDR or IEC 62304.

## Classification-by-Use-Mode Analysis

The EU MDR classification of the Layer 4 commercial product depends on its intended use:

| Use Mode | Classification | Rationale |
|----------|---------------|-----------|
| Research / pre-clinical simulation | Not a medical device | No clinical decision-making; general-purpose tool ({term}`MDCG 2019-11` §3.2) |
| Treatment planning (pre-operative) | Class IIa | Software providing information used to make decisions with diagnosis or treatment purposes (Rule 11, Annex VIII); non-immediate clinical context |
| Intraoperative digital twin | Class IIb minimum, likely Class III | Real-time surgical decision support; erroneous output during live procedure in CSF spaces adjacent to striatum/substantia nigra could cause irreversible neurological harm (Rule 11(b)) |
| RL policy generation for microrobot control | Class III | Software directly informs an active control loop for a device operating in the CNS; failure could result in death or serious injury (Rule 11(b), Rule 22) |

The MADDENING documentation architecture is designed to support the most demanding classification (Class III / IEC 62304 Class C) so that less demanding use cases are automatically covered.

## Commercial Boundary Documentation

### What MADDENING provides to the commercial entity

1. **SOUP Package** (`docs/validation/soup_package.md`): Software identification, functional description, known anomalies, verification evidence, lifecycle activities, configuration management, anomaly management policy, dependency list
2. **Known Anomalies Registry** (`docs/validation/known_anomalies.yaml`): Machine-readable, versioned, validated by CI
3. **Verification Evidence**: Registered benchmarks with analytical solutions, convergence studies, and regression tests
4. **Hazard Hints**: Per-node qualitative risk information for integration into the manufacturer's ISO 14971 risk management file
5. **Algorithm Guides**: Per-node documentation of governing equations, discretization, assumptions, limitations, and validated regimes — with CI-validated bibliography citations (`docs/bibliography.bib`) and CI-validated implementation-to-code mappings
6. **HealthCheckNode**: Configurable execution-layer fault detection (NaN/Inf, bounds, moment checks) for use in downstream safety monitoring
7. **Compliance Schema Types**: Importable Python types (`NodeMeta`, `AnomalyRecord`, etc.) for downstream libraries to use and extend
8. **{term}`SBOM`**: CycloneDX 1.6 JSON SBOMs for the base install and for the `server`, `surrogates` and `usd` extras (`docs/validation/sbom/`).  Each records the resolved dependency tree, with every package's version, purl and declared licence, and the Python, platform and date it was resolved on.  They are regenerated from the release commit for each release and attached to the GitHub release.  Which installs are covered and why, and what an SBOM does not record, are in `docs/validation/soup_package.md` §6

### What MADDENING does NOT provide

- Clinical safety assessment
- Intended use for any specific clinical application
- Risk management file (ISO 14971)
- Clinical evaluation report (CER)
- Post-market surveillance plan
- QMS documentation (ISO 13485)
- Notified Body relationship
- End-user training materials for clinical use
- Cybersecurity risk assessment for the deployed product (MADDENING assumes trusted inputs; see `docs/regulatory/intended_use.md`)
- Authentication on every network surface, or anything beyond one shared secret per surface.  What 0.4.0 does provide, and where it stops (`SECURITY.md` has the detail):
  - **HTTP/WebSocket API** (`api` extra): a bearer token (`MADDENING_API_TOKEN`) is required on any bind other than loopback.  There is **no TLS**, so the token crosses the network in cleartext; put the API behind an SSH tunnel or a TLS-terminating proxy.  A loopback bind is unauthenticated.
  - **ZeroMQ transports** (`network` extra): off loopback, CURVE encryption and authentication are required, keyed from one shared token (`MADDENING_TRANSPORT_TOKEN`, else `MADDENING_API_TOKEN`).  Every holder of the token can derive both keypairs, so the transport does not tell token holders apart and has no per-peer keys.  There is no forward secrecy.  See the residual risk of MADD-ANO-015.  A loopback bind is unauthenticated.
  - **{term}`WebRTC` signaling socket** (`streaming` extra): clients are authenticated with HMAC tokens derived from `MADDENING_STREAM_SECRET`; without the secret no client can connect.
  - **{term}`FMU` TCP bridge** (`maddening.fmi.tcp_bridge`): **no authentication at all**.  Any process that can reach its port can read, write, step and hold the model, whatever interface it is bound to.  It binds loopback on an ephemeral port by default, and it is for trusted clients only (MADD-ANO-023, open).
  - No user accounts, per-user authorisation, access audit or rate limiting on any surface.

### Licence

From 0.4.2, MADDENING is licensed under Apache-2.0 (releases up to and including 0.4.1: LGPL). It is permissive, with no copyleft and no replaceability obligation: the commercial entity keeps the copyright and licence notices, ships the attributions in MADDENING's `NOTICE` file, and states any changes it made to MADDENING's files. The licence includes a patent grant from each contributor. Record the licence of every layer in the technical file, since they differ: MIME is Apache-2.0 from its 0.3.0, and MICROROBOTICA is AGPL-3.0-or-later, whose network-use clause applies to a product that serves it to users over a network.
