# Uncertainty Quantification Roadmap

## Current Status

MADDENING's UQ support is in early development (Phase 4 per the documentation architecture).

## UQ Interface

The `UncertaintySpec` and `UncertainParameter` dataclasses in `maddening.core.uq` define the interface for nodes to declare their UQ capabilities.

### Node UQ Readiness

| Node | `uq_readiness` | Notes |
|------|----------------|-------|
| BallNode | NOT_READY | Simple enough for parameter sweep UQ |
| TableNode | NOT_READY | Static node, no uncertain parameters |
| SpringDamperNode | NOT_READY | Candidate for parameter sweep UQ (k, c) |
| HeatNode | NOT_READY | Candidate for parameter sweep UQ (alpha) |
| LBMPipeNode | NOT_READY | Complex; needs careful UQ design |
| RigidBody2DNode | NOT_READY | Candidate for parameter sweep UQ |

## V&V 40 Capability Mapping

| V&V 40 Requirement | MADDENING Capability | Status |
|---|---|---|
| Parameter sensitivity analysis | `GraphManager.run_sweep()` (vmap over parameters) | Available |
| Forward UQ propagation | Parameter sweeps via vmap | Available (basic) |
| Node-level UQ specification | `UncertaintySpec` interface | Interface defined, not populated |
| Ensemble analysis | Batched scan via `run_sweep()` | Available |
| Bayesian inference | Not yet supported | Planned |

## Future Work

- Populate `uncertainty_spec()` on physics nodes
- Add ensemble UQ utilities
- Integration with external UQ libraries (e.g., SALib for sensitivity analysis)
