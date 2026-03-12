# Framework Verification Summary

## Overview

MADDENING's verification evidence is maintained through automated testing and registered verification benchmarks.

## Test Suite

- **Total tests**: 500+ (and growing)
- **CI system**: GitHub Actions
- **Test runner**: pytest
- **Platforms tested**:
  - OS: Ubuntu Linux
  - Python: 3.12
  - JAX: 0.4+ (pinned in CI)
  - Backend: CPU (GPU tests disabled due to MADD-ANO-001)

## Test Organization

| Directory | Scope |
|-----------|-------|
| `tests/core/` | Core framework: GraphManager, scheduling, coupling, adaptive, checkpoint, sweep |
| `tests/nodes/` | Physics node correctness: HeatNode, LBMPipeNode, RigidBody2DNode, SpringDamperNode |
| `tests/surrogates/` | Neural surrogate training, architectures, dataset generation |
| `tests/api/` | FastAPI server, WebSocket, binary encoding, server-side rendering |
| `tests/viz/` | Visualization backends, ZMQ transport, serialization |
| `tests/compliance/` | Compliance infrastructure: metadata, anomaly validator, stability decorator |
| `tests/verification/` | Registered verification benchmarks (analytical comparisons, convergence studies) |

## Registered Verification Benchmarks

See `maddening.compliance.get_benchmark_registry()` for the machine-readable registry.

| Benchmark ID | Node | Type | Acceptance Criteria |
|---|---|---|---|
| MADD-VER-001 | HeatNode | Analytical | L2 error vs analytical solution < 1e-3 |

*Registry grows as benchmarks are added in Phase 2+.*
