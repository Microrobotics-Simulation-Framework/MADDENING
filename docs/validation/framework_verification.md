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

```{warning}
**What this evidence was generated against is only partly recorded.**  CI
hard-pins `jax` and `jaxlib`; every other base dependency is installed from
its range, so the resolved version differs between runs and no run records
it.  Reading `jax>=0.10,<0.13` in the table below as "verified across that
range" would be wrong — one point in it has been exercised.  Closing this
properly needs a lock file or an SBOM captured per CI run (§6 of
`soup_package.md` records the CycloneDX SBOM as still planned); until then
the table says what is pinned and what is not, rather than letting the
declared range stand in for the tested one.
```

## Test Suite

<!-- BEGIN GENERATED: test-suite -- scripts/generate_soup_tables.py; do not edit by hand -->
| Field | Value |
|---|---|
| Test runner | pytest |
| CI system | GitHub Actions |
| CI runners | `ubuntu-latest` |
| Python versions | 3.12 (floor: >=3.12) |
| JAX | evidence generated at `0.10.2`, `0.11.2`, the 2 versions CI installs; `jax>=0.10,<0.13` is the *declared* range and no other point in it has been exercised |
| Other base dependencies | `lineax>=0.0.7`, `numpy>=1.24`, `pyyaml>=6.0` — installed from these ranges, not pinned, so the resolved version differs between runs and **is not recorded** |
| Backend | CPU (GPU tests are not run in CI — MADD-ANO-001) |
| Test packages | 13 — listed below |
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
| `tests/security/` | Transport authentication: ZMQ CURVE encryption and key derivation for the state, command and coordinator sockets |
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
| MADD-VER-001 | HeatNode | `analytical` | L2 relative error < 1e-4 after 100 steps at CFL=0.25 (n=50); measured 1.59e-5, a factor of 6.3 of margin. Was < 5% before 0.4.0, when the boundary-cell overwrite (MADD-ANO-007) put the error at 1.8% -- a threshold 2.8x the defect it was covering. | `tests.verification.test_heat_analytical.test_heat_fourier_benchmark` |
| MADD-VER-002 | HeatNode | `convergence_study` | Mean of the pairwise global L2 convergence rates over a 20/40/80 ladder at CFL=0.25 within [1.7, 2.3] of the theoretical 2.0 (measured: 1.900), and the error strictly decreasing. Before 0.4.0 this study measured ~1.0 -- the boundary-cell overwrite of MADD-ANO-007 -- and its band had been widened to [0.7, 2.5], which admitted that result; the band now rejects it by 0.7. | `tests.verification.test_heat_analytical.test_heat_spatial_convergence` |
| MADD-VER-003 | LBMNode | `analytical` | After 500 steps: profile positive and peaked at the centre; a least-squares fit u = c0 + c1 r^2 concave with effective R^2 within [0.7, 1.5] of nominal; cross-section symmetric to rtol 1e-3; centreline velocity in [0.75, 1.30] x u_max = F R^2 / (4 mu). The band is deliberately asymmetric: LBMNode's wall cells collide as well as reflect, so the hydrodynamic wall sits near the wall nodes rather than on the half-way plane (about 0.1 lattice units from the wall node in MADD-VER-016's straight channel). Here the fitted effective radius is 4.29 against the nominal 4.0 (R_eff^2 = 18.4 against 16), within 0.01 of the innermost wall-cell centre at r = 4.30, so the ratio is biased above 1 (measured 1.130). A symmetric +/-25% would leave 0.12 of headroom above the measured value on the side the discretisation pushes it | `tests.cloud.multigpu.test_lbm_poiseuille.test_poiseuille_sharded_profile_is_parabolic` |
| MADD-VER-004 | AdaptiveNode | `analytical` | Full-basis (K = n = 256) L2 relative error < 1e-4 on the grid and sensor error < 1e-6; over K in (4, 8, 16, 32) the K = 32 sensor error is the smallest of the four, is < 1e-6, and is at least two orders of magnitude below the K = 4 error. Not strictly decreasing in K: top-K on a non-nested basis means the K = 8 set is not a superset of the K = 4 set (9 of 48 swept configurations are non-monotone) | `tests.nodes.adaptive.test_verification.test_adaptive_solve_matches_greens_function_and_converges_in_k` |
| MADD-VER-005 | HeatNode | `manufactured_solution` | Observed spatial order over the finest pair of a 10/20/40/80/160 ladder within [-0.25, +1.0] of the declared 2.0 (measured: 2.000). Dirichlet data supplied at the rod ends x=0 and x=L, which is what boundary_input_spec documents and, since 0.4.0, what the node implements; before 0.4.0 the same ladder measured 1.001 (MADD-ANO-007). | `tests.verification.test_mms_order.test_heat_second_order_stencil_converges_at_its_declared_spatial_order` |
| MADD-VER-006 | HeatNode | `manufactured_solution` | Observed temporal order over the finest pair of a 250/500/1000/2000 step ladder within [-0.25, +1.0] of the declared 1.0 (measured: 1.000) | `tests.verification.test_mms_order.test_heat_converges_at_its_declared_temporal_order` |
| MADD-VER-007 | LBMNode | `manufactured_solution` | Observed spatial order over the finest pair of a 16/32/64 ladder within [-0.25, +1.0] of the declared 2.0 (measured: 1.998). Wall-free periodic domain; bounce-back walls are 1st order. | `tests.verification.test_mms_order.test_lbm_converges_at_its_declared_spatial_order` |
| MADD-VER-008 | RigidBodyNode | `manufactured_solution` | Observed temporal order over the finest pair of a 100/200/400/800 step ladder within [-0.25, +1.0] of the declared 1.0 (measured: 0.999) | `tests.verification.test_mms_order.test_rigid_body_converges_at_its_declared_temporal_order` |
| MADD-VER-009 | SpringDamperNode | `manufactured_solution` | Observed temporal order over the finest pair of a 100/200/400/800 step ladder within [-0.25, +1.0] of the declared 1.0 (measured: 1.029) | `tests.verification.test_mms_order_ode_nodes.test_spring_converges_at_its_declared_temporal_order` |
| MADD-VER-010 | BallNode | `manufactured_solution` | Observed temporal order over the finest pair of a 100/200/400/800 step ladder within [-0.25, +1.0] of the declared 1.0 (measured: 1.002). Smooth regime only: no table_position is supplied, so no order is claimed across a collision. | `tests.verification.test_mms_order_ode_nodes.test_ball_converges_at_its_declared_temporal_order` |
| MADD-VER-011 | RigidBody2DNode | `manufactured_solution` | Observed temporal order over the finest pair of a 100/200/400/800 step ladder within [-0.25, +1.0] of the declared 1.0 (measured: 1.000) | `tests.verification.test_mms_order_ode_nodes.test_rigid_body_2d_converges_at_its_declared_temporal_order` |
| MADD-VER-012 | HeartPumpNode | `manufactured_solution` | Observed temporal order over the finest pair of a 200/400/800/1600 step ladder within [-0.25, +1.0] of the declared 1.0 (measured: 1.000). The ladder stops at 1600 steps: the float32 downcast of backpressure (MADD-ANO-013) turns it over below about 6e-6 relative error, two orders of magnitude finer than the 1.9e-3 this ladder reaches. | `tests.verification.test_mms_order_ode_nodes.test_heart_pump_converges_at_its_declared_temporal_order` |
| MADD-VER-013 | LBMPipeNode | `convergence_study` | Monotone convergence (R = 0.846 in (0, 1)) with a determinate observed order (measured: 1.49) and a Grid Convergence Index on the finest solution below 25% at Fs = 3.0 (measured: 10.7%). Fs = 3.0 and not 1.25 because the node declares no order of accuracy, so the asymptotic range cannot be established from three levels; a fourth level shows it is not in it (MADD-VER-013 is a convergence statement, not an accuracy one). | `tests.verification.test_gci_order.test_the_pipe_flow_converges_under_refinement_where_mms_cannot_run` |
| MADD-VER-014 | WaveletAdaptiveNode | `manufactured_solution` | Observed spatial order over the finest pair of a 16/32/64/128/256 ladder within [-0.25, +1.0] of the declared 2.0 (measured: 2.000). The same manufactured study on the Dirichlet basis (23 .. 191 interior points) and in 2-D (8^2 .. 32^2) measures 2.00 as well; those ladders are asserted by the parametrised test beside this one. | `tests.verification.test_wavelet_mms_order.test_wavelet_node_converges_at_its_declared_spatial_order` |
| MADD-VER-015 | WaveletAdaptiveNode | `regression` | On the 128-point periodic basis at k = 8, for the Gaussian source at theta in {0.04, 0.30, 0.42, 0.50, 0.92}: the adaptive sensor reading is within 1e-2 relative of the full-basis reading (measured max 4.5e-4, at theta = 0.42, float64) and has the same sign at every position, including the two next to the periodic seam. At k = n_max the two coincide to 1e-10. No convergence rate in k is claimed. | `tests.verification.test_wavelet_mms_order.test_the_adaptive_sensor_reading_tracks_the_full_basis_one_across_the_source_range` |
| MADD-VER-016 | LBMNode | `analytical` | Centreline velocity over Hagen-Poiseuille (dp/L) H^2 / (8 rho nu) at H = 8 and 16 (L = 4H, tau = 1, float32) falls under refinement, and its extrapolation at the walls' declared first order, 2 r(16) - r(8), is within 1% of 1 (measured r = 1.2071, 1.1036; limit 1.0001). The finite-H excess is the first-order bounce-back wall (hydrodynamic wall about 0.09 lattice units from the wall node) and is the same under body-force driving. Before 0.4.0 the Zou-He closure imposed the wrong face density (MADD-ANO-020) and the ladder read 0.920, 0.658: away from 1, limit 0.396. | `tests.verification.test_lbm_pressure_poiseuille.test_pressure_driven_poiseuille_converges_to_hagen_poiseuille` |

*16 benchmarks registered.*
<!-- END GENERATED: verification-benchmarks -->
