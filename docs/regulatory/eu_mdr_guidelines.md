# EU MDR Alignment Guide

## Scope

This document maps MADDENING's documentation and design to EU MDR (EU 2017/745) requirements that a downstream commercial manufacturer would need to address. MADDENING is not subject to EU MDR — this mapping is provided voluntarily.

## Annex I — General Safety and Performance Requirements (GSPRs)

| GSPR | Requirement | MADDENING Contribution | Manufacturer Responsibility |
|------|------------|----------------------|---------------------------|
| **1** | Devices shall achieve intended performance and not compromise safety | MADDENING provides verified computational tools; does not define intended performance | Define intended performance for their device; validate MADDENING outputs for their COU |
| **3** | Devices shall be safe and effective throughout intended lifetime | Version-pinned releases, SOUP package per version, known anomalies tracked | Pin MADDENING version, monitor for updates, maintain throughout product lifetime |
| **4** | Risk management per ISO 14971 | `NodeMeta.hazard_hints` provides technical hazard identification input; `known_anomalies.yaml` identifies candidate hazard contributors | Full ISO 14971 risk management file; clinical risk assessment |
| **17.1** | Software developed per state of the art, considering lifecycle, risk management, V&V | IEC 62304 lifecycle mapping (Section 11), registered verification benchmarks, anomaly management | Apply IEC 62304 to their own software; cite MADDENING evidence in SOUP assessment |
| **17.2** | Software safety classification | MADDENING provides Class C-ready SOUP documentation | Classify their software per IEC 62304 based on their risk assessment |
| **17.4** | Repeatability, reliability, performance | Functional purity (deterministic outputs), version pinning, regression tests | Validate in their operational environment; execution-layer monitoring |

## Annex II — Technical Documentation

| Annex II Section | Content Required | What MADDENING Provides |
|---|---|---|
| §1 — Device description | Component description including SOUP | `docs/regulatory/intended_use.md`, SOUP package Section 2 |
| §4 — Design and manufacturing | Software architecture, IEC 62304 lifecycle, SOUP list | `DESIGN.md`, `docs/regulatory/iec62304_mapping.md`, SOUP package |
| §5 — Benefit-risk analysis | ISO 14971 risk management file including SOUP failure mode analysis | `NodeMeta.hazard_hints`, `known_anomalies.yaml` |
| §6 — Product V&V | Computational V&V evidence | Verification benchmarks, algorithm guides, test suite |

## MDCG 2019-16 — Cybersecurity

MADDENING provides:
- **SBOM**: CycloneDX format, attached to each release (when implemented)
- **SECURITY.md**: vulnerability reporting process
- **Dependency monitoring**: GitHub Dependabot + manual JAX changelog review
- **Cybersecurity boundary**: documented in `docs/regulatory/intended_use.md`

The manufacturer must assess MADDENING's cybersecurity posture as part of their MDCG 2019-16 documentation and implement additional controls as needed for their deployment context.
