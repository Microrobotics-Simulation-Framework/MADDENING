# Framework Verification Summary

## Overview

MADDENING's verification evidence is maintained through automated testing and registered {term}`verification benchmarks <Verification benchmark>`.

Nothing on this page is maintained by hand.  `scripts/generate_soup_tables.py`
builds every table below from `maddening.compliance.get_benchmark_registry()`,
the `tests/` tree, `pyproject.toml` and `.github/workflows/ci.yml`, and
`tests/compliance/test_soup_evidence.py` fails if a committed table no longer
matches its source.  The previous hand-maintained version of this page listed
one of four registered benchmarks, seven of twelve test packages, a Python
version CI had not used alone since 0.2, and "JAX: 0.4+" against a 0.10.2 pin.

## Test Suite

<!-- BEGIN GENERATED: test-suite -- scripts/generate_soup_tables.py; do not edit by hand -->
| Field | Value |
|---|---|
| Test runner | pytest |
| CI system | GitHub Actions |
| CI runners | `ubuntu-latest` |
| Python versions | 3.11, 3.12 (floor: >=3.11) |
| JAX | pinned to `0.10.2` in CI; `jax>=0.10,<0.13` supported |
| Backend | CPU (GPU tests are not run in CI — MADD-ANO-001) |
| Test packages | 12 — listed below |
<!-- END GENERATED: test-suite -->

## Test Organization

<!-- BEGIN GENERATED: test-organization -- scripts/generate_soup_tables.py; do not edit by hand -->
| Directory | Scope |
|---|---|
| `tests/adaptive/` | Adaptive timestepping: Richardson extrapolation, PI controller |
| `tests/api/` | FastAPI server, WebSocket, binary encoding, server-side rendering |
| `tests/cloud/` | Distributed execution: multi-GPU sharding, halo exchange, resume transport, checkpoint download |
| `tests/compliance/` | Compliance infrastructure: metadata, anomaly validator, stability decorator, benchmark registry, provenance |
| `tests/core/` | Core framework: GraphManager, scheduling, coupling, params, checkpoint, sweep, solver utilities |
| `tests/fmi/` | FMI/FMU export and import: model description, binary frames, parameter variables |
| `tests/nodes/` | Physics node correctness: HeatNode, LBMNode, LBMPipeNode, RigidBody2DNode, SpringDamperNode, AdaptiveNode |
| `tests/property/` | Hypothesis property tests over generated graphs, meshes and coupling configurations |
| `tests/surrogates/` | Neural {term}`surrogate <Surrogate>` training, architectures, dataset generation |
| `tests/usd/` | USD stage serialization and round-trips, geometry sources, interface mappings |
| `tests/verification/` | Registered verification benchmarks (analytical comparisons, convergence studies) |
| `tests/viz/` | Visualization backends, ZMQ transport, serialization |
<!-- END GENERATED: test-organization -->

## Registered Verification Benchmarks

`maddening.compliance.get_benchmark_registry()` is the registry of record; a
benchmark is registered by the `@verification_benchmark` decorator on the test
that enforces it, so the acceptance criteria below are the ones actually
asserted.

<!-- BEGIN GENERATED: verification-benchmarks -- scripts/generate_soup_tables.py; do not edit by hand -->
| Benchmark ID | Node | Type | Acceptance Criteria | Test |
|---|---|---|---|---|
| MADD-VER-001 | HeatNode | `analytical` | L2 relative error < 5% after 100 steps at CFL=0.25 (n=50) | `tests.verification.test_heat_analytical.test_heat_fourier_benchmark` |
| MADD-VER-002 | HeatNode | `convergence_study` | Interior spatial convergence rate between 1.5 and 2.5 (theoretical: 2.0). Global rate ~1.0 due to O(dx) boundary overwrite — documented in MADD-ANO-002. | `tests.verification.test_heat_analytical.test_heat_spatial_convergence` |
| MADD-VER-003 | LBMNode | `analytical` | After 500 steps: profile positive and peaked at the centre; a least-squares fit u = c0 + c1 r^2 concave with effective R^2 within [0.7, 1.5] of nominal; cross-section symmetric to rtol 1e-3; centreline velocity within +/-25% of u_max = F R^2 / (4 mu) | `tests.cloud.multigpu.test_lbm_poiseuille.test_poiseuille_sharded_profile_is_parabolic` |
| MADD-VER-004 | AdaptiveNode | `analytical` | Full-basis (K = n = 256) L2 relative error < 1e-4 on the grid and sensor error < 1e-6; over K in (4, 8, 16, 32) the K = 32 sensor error is the smallest of the four, is < 1e-6, and is at least two orders of magnitude below the K = 4 error. Not strictly decreasing in K: top-K on a non-nested basis means the K = 8 set is not a superset of the K = 4 set (9 of 48 swept configurations are non-monotone) | `tests.nodes.adaptive.test_verification.test_adaptive_solve_matches_greens_function_and_converges_in_k` |
| MADD-VER-005 | HeatNode | `manufactured_solution` | Observed spatial order over the finest pair of a 10/20/40/80/160 ladder within [-0.25, +1.0] of the declared 2.0 (measured: 1.982). Applies to the boundary convention the code implements, Dirichlet data at the first and last cell centre; see MADD-ANO-007. | `tests.verification.test_mms_order.test_heat_second_order_stencil_converges_at_its_declared_spatial_order` |
| MADD-VER-006 | HeatNode | `manufactured_solution` | Observed temporal order over the finest pair of a 250/500/1000/2000 step ladder within [-0.25, +1.0] of the declared 1.0 (measured: 0.998) | `tests.verification.test_mms_order.test_heat_converges_at_its_declared_temporal_order` |
| MADD-VER-007 | LBMNode | `manufactured_solution` | Observed spatial order over the finest pair of a 16/32/64 ladder within [-0.25, +1.0] of the declared 2.0 (measured: 1.998). Wall-free periodic domain; bounce-back walls are 1st order. | `tests.verification.test_mms_order.test_lbm_converges_at_its_declared_spatial_order` |
| MADD-VER-008 | RigidBodyNode | `manufactured_solution` | Observed temporal order over the finest pair of a 100/200/400/800 step ladder within [-0.25, +1.0] of the declared 1.0 (measured: 0.999) | `tests.verification.test_mms_order.test_rigid_body_converges_at_its_declared_temporal_order` |
| MADD-VER-009 | SpringDamperNode | `manufactured_solution` | Observed temporal order over the finest pair of a 100/200/400/800 step ladder within [-0.25, +1.0] of the declared 1.0 (measured: 1.029) | `tests.verification.test_mms_order_ode_nodes.test_spring_converges_at_its_declared_temporal_order` |
| MADD-VER-010 | BallNode | `manufactured_solution` | Observed temporal order over the finest pair of a 100/200/400/800 step ladder within [-0.25, +1.0] of the declared 1.0 (measured: 1.002). Smooth regime only: no table_position is supplied, so no order is claimed across a collision. | `tests.verification.test_mms_order_ode_nodes.test_ball_converges_at_its_declared_temporal_order` |
| MADD-VER-011 | RigidBody2DNode | `manufactured_solution` | Observed temporal order over the finest pair of a 100/200/400/800 step ladder within [-0.25, +1.0] of the declared 1.0 (measured: 1.000) | `tests.verification.test_mms_order_ode_nodes.test_rigid_body_2d_converges_at_its_declared_temporal_order` |
| MADD-VER-012 | HeartPumpNode | `manufactured_solution` | Observed temporal order over the finest pair of a 200/400/800/1600 step ladder within [-0.25, +1.0] of the declared 1.0 (measured: 1.000). The ladder stops at 1600 steps: the float32 downcast of backpressure (MADD-ANO-013) turns it over below about 6e-6 relative error, two orders of magnitude finer than the 1.9e-3 this ladder reaches. | `tests.verification.test_mms_order_ode_nodes.test_heart_pump_converges_at_its_declared_temporal_order` |
| MADD-VER-013 | LBMPipeNode | `convergence_study` | Monotone convergence (R = 0.846 in (0, 1)) with a determinate observed order (measured: 1.49) and a Grid Convergence Index on the finest solution below 25% at Fs = 3.0 (measured: 10.7%). Fs = 3.0 and not 1.25 because the node declares no order of accuracy, so the asymptotic range cannot be established from three levels; a fourth level shows it is not in it (MADD-VER-013 is a convergence statement, not an accuracy one). | `tests.verification.test_gci_order.test_the_pipe_flow_converges_under_refinement_where_mms_cannot_run` |

*13 benchmarks registered.*
<!-- END GENERATED: verification-benchmarks -->
