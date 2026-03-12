# MADDENING SOUP Package Document

**Version**: 0.1.0
**Date**: 2026-03-12

## 1. Software Identification

| Field | Value |
|-------|-------|
| Name | MADDENING |
| Full Name | Modular Automatic Differentiation and Data-Enhanced Neural-network INteracting Graph |
| Version | 0.1.0 |
| Release Date | 2025-03-01 |
| Licence | LGPL-3.0-or-later |
| Source Repository | https://github.com/Microrobotics-Simulation-Framework/MADDENING |
| Python Version | >=3.10 |
| Primary Dependencies | JAX >=0.4, jaxlib >=0.4, NumPy >=1.24 |
| Build System | hatchling |
| Install | `pip install maddening` |

## 2. Functional Description

### Core Capabilities

- **Graph-based multi-physics simulation**: Compose, couple, and run physics simulations as directed graphs of nodes
- **Functional state pattern**: Pure functions, no shared mutable state, explicit data flow
- **JIT compilation**: Full simulation step compiled to XLA via JAX
- **Automatic differentiation**: End-to-end differentiable simulation graphs
- **Neural surrogates**: Train and deploy neural network replacements for physics nodes
- **Adaptive timestepping**: Richardson extrapolation with PI controller
- **Parameter sweeps**: Batched simulation via `jax.vmap`
- **Coupling**: Gauss-Seidel iterative coupling via `jax.lax.fori_loop`

### Capabilities NOT Provided

- Clinical decision support
- Patient data processing
- Medical device functionality
- Real-time safety monitoring (HealthCheckNode provides infrastructure only; configuration and response logic are downstream responsibilities)
- Input validation or sanitization (assumes trusted inputs)

### Infrastructure Nodes

- **HealthCheckNode** (`maddening.nodes.health_check`): Base health monitor for execution-layer fault detection (NaN/Inf trapping, physical boundary checks). Downstream libraries instantiate and configure this node. Part of the MADDENING SOUP dependency, not a separate SOUP item.

## 3. Known Anomalies

See `known_anomalies.yaml` (same directory) for the full structured registry.

| ID | Title | Severity | Safety Relevance |
|----|-------|----------|-----------------|
| MADD-ANO-001 | LBM GPU segfault on CUDA 12.2 + jaxlib 0.5.1 | Major | Context-dependent |
| MADD-ANO-002 | HeatNode CFL stability not enforced at runtime | Major | Context-dependent |

## 4. Verification Evidence

*To be completed in Phase 2.*

## 5. IEC 62304 Lifecycle Activities

See `docs/regulatory/iec62304_mapping.md` for the full lifecycle mapping.

## 6. Configuration Management

- Version control: Git (GitHub)
- Release tags: semantic versioning (`vX.Y.Z`)
- SBOM: CycloneDX format (planned, Phase 3)
- CI: GitHub Actions

## 7. Anomaly Management Policy

See CONTRIBUTING.md for the three-phase anomaly lifecycle and three-tier release gate model.

## 8. Dependencies

*To be completed in Phase 2/3. See `pyproject.toml` for current dependency list.*
