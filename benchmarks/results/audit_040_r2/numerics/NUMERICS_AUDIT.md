# Audit R2: numerics — node schemes, integrators, rank cutoff, coupling error estimate

Commit audited: `0219b82` (`origin/release/0.4.0`, fetched 2026-09-20).
Worktree: `/home/nick/MSF/msf/MADDENING-wt/audit-r2-numerics` (detached, left clean — `git status --porcelain` empty).
Reproducers and raw output: `/home/nick/MSF/msf/MADDENING/benchmarks/results/audit_040_r2/numerics/` (`repro/*.py`, `repro/mutate.sh`, `repro/run_gate_probes.sh`, `*.log`, `mutation_log.txt`).
All studies under `jax_enable_x64` on CPU unless stated; one short GPU run (`.venv-cuda`, jax 0.11.2, CUDA 12.9) for the TF32 question, nothing installed into it.

## Summary

The numerics this release changed are, as numerics, in good shape. **All seven declared orders reproduce independently** (my own manufactured solutions, my own ladders and norms, not `maddening.testing.mms`): HeatNode 2.000 / 3.957–3.979, LBMNode 1.998, Spring 1.001, Ball 1.000, RigidBody 1.000, RigidBody2D 1.000, HeartPump 1.000. Every numeric claim in `integrators.py` reproduces to three figures, including the counter-intuitive rk4/euler = 1.53 inversion. `MAX_FOURIER_NUMBER` re-derives exactly as its docstring instructs. Thirteen mutations against the integrator, coupling and sysid gates were all caught.

Four things are wrong. `HeatNode.compute_boundary_fluxes` reports the flux half a cell inside the rod and **nothing anywhere asserts its value** (HIGH). Three nodes cast parameters to float32 in ways `MADD-ANO-013` does not cover, one putting a refinement-invariant 3.7e-8 relative error on `RigidBodyNode`'s angular velocity (MEDIUM). `scripts/check_heat_stability.py` has three holes, two of them the exact classes it was written to avoid, including a live count inflation (MEDIUM). And the rotational half of the semi-implicit claim on both rigid-body nodes is pinned by nothing (MEDIUM).

**The GPU/TF32 hypothesis in my brief is refuted**, with a measurement and a mechanism — see "What I checked and found sound".

---

## Findings

### HIGH — `HeatNode.compute_boundary_fluxes` reports the flux at `x = dx`, not at the rod end, and no test checks its value

**What breaks:** `boundary_flux_spec()` declares `left_heat_flux` as "Heat flux at left boundary". On a rod holding `T(x) = exp(x)` with `alpha = 1`, the true flux at the rod end `x = 0` is `-1.0`. At `n_cells = 10` the node returns `-1.10563`, a **10.6 % error**; at `n = 160`, `-1.00627` (0.63 %); at `n = 640`, `-1.00156`. These are not slow approximations of the rod-end flux — they are the flux at `x = dx` to 4–7 significant figures.

**Evidence:** `repro/r4_boundary_flux.py` → `r4_boundary_flux.log`

```
    n   reported_left     true q(0)    true q(dx)   err vs q(0)  err vs q(dx)
   10     -1.10563146   -1.00000000   -1.10517092     1.056e-01     4.167e-04
   40     -1.02534182   -1.00000000   -1.02531512     2.534e-02     2.604e-05
  160     -1.00627121   -1.00000000   -1.00626957     6.271e-03     1.628e-06
  640     -1.00156382   -1.00000000   -1.00156372     1.564e-03     1.017e-07

Convergence of the reported left flux to the TRUE ROD-END flux:
  n=  80-> 160  err 1.2585e-02 -> 6.2712e-03  order 1.005
What the correct rod-end reconstruction would give (-alpha*(T[0]-T_left)/(dx/2)):
  n=  80-> 160  err 3.1315e-03 -> 1.5641e-03  order 1.002
```

**Why it happens:** `src/maddening/nodes/heat.py:884-887`

```python
"left_heat_flux": -alpha * (T[1] - T[0]) / dx_left,
"right_heat_flux": -alpha * (T[-1] - T[-2]) / dx_right,
```

`T[1] - T[0]` spans two cell *centres*, so the centred difference sits at `x = dx`, a full cell inside the boundary the spec names. The method already receives `boundary_inputs` (carrying `left_temperature`, the rod-end datum) and never reads it. This is MADD-ANO-007 again — the datum half a cell out — on the flux side rather than the state side. Independently confirms and quantifies TODO FOLLOW-UP B.

**Why nothing detects it:** the only value tests are `tests/core/test_flux_coupling.py::TestComputeBoundaryFluxes::test_compute_fluxes_heat` (asserts `isfinite`), `test_compute_fluxes_heat_nonuniform` (asserts `> 0.0`) and `tests/core/test_spatial_accuracy.py:193-196` (keys exist, finite). Grepping `left_heat_flux`/`right_heat_flux` across `tests/` finds no assertion on the number; every other hit is edge wiring.

**What would make this a non-issue:** (a) a different intended convention — checked: the description says "at left boundary", the whole 0.4.0 boundary rework is about the datum living at `x = 0`/`x = L`, and the one-sided rod-end reconstruction needs only data the method already has; (b) cancellation in a rod-to-rod coupling — checked: on the steady linear profile `T = x` both ends report exactly `-1.0` (the scheme is exact there), so a linear-profile test cannot see it, and on any curved profile the two ends err with opposite signs; (c) harmlessly second-order — checked: it converges at order **1.005** against the node's 2.000 in the state, so a flux-coupled solve is capped at first order by the flux alone.

**Why HIGH and not CRITICAL:** the returned number is a correct, convergent quantity, just not the named one, and `TODO.md` already records it as FOLLOW-UP B. The escalation case is real though — silent, unchecked, on a coupling interface. If any shipped flux-coupling result is quoted anywhere, treat as CRITICAL.

**Suggested fix:** return `-alpha * (T[0] - T_left) / (dx/2)` when `left_temperature` is supplied (first order, 4x smaller constant), or a second-order one-sided form using the existing ghost; fall back to the present expression when no Dirichlet datum is given. Risk: changes every flux-coupling number by up to 10 %, so it needs a release note in the same breaking-changes table as MADD-ANO-007, and the conservation tests in `tests/core/test_coupling_helpers.py` will move.

---

### MEDIUM — three unregistered float32 parameter casts, one putting a refinement-invariant 3.7e-8 error on `RigidBodyNode`'s angular velocity

**What breaks:** `MADD-ANO-013` registers exactly one float32 downcast (`HeartPumpNode.backpressure`) and `TODO.md` FOLLOW-UP C names one more (`RigidBody2DNode.gravity`). There are at least six sites; three unregistered ones sit on *trainable parameters*, where the effect is a wrong number no refinement removes.

On problems whose exact solution the node's own scheme reproduces exactly (discretisation error identically zero, so everything left is the cast):

| node / cast | exact | returned | rel. err | changes with N? |
|---|---|---|---|---|
| `RigidBodyNode.inertia`, `I = 1/3` | `omega = 3.0` | 2.999999910593 | 2.98e-08 | no (N=10 and N=1000 identical) |
| `RigidBodyNode.inertia`, `I = 1.3` | 0.769230769231 | 0.769230797446 | 3.67e-08 | no |
| `RigidBodyNode.gravity`, `g = -9.81` | -9.81 | -9.810000658035 | 6.71e-08 | no |
| `RigidBody2DNode.gravity` (FOLLOW-UP C) | -9.81 | -9.810000658035 | 6.71e-08 | no |

`RigidBodyNode.inertia` at `1.0` and `1.0 + 1e-8` returns **bit-identical** angular velocity; central finite differences of `omega` w.r.t. `inertia` at `h = 1e-9` return exactly `0.0`.

A fourth instance makes one node disagree with itself: `BallNode.update()` uses raw `p["gravity"]` while `BallNode.derivatives()` casts it (`ball.py:152`), so under x64 the two paths differ about `g` by **4.196e-07 absolute**. Same pattern as MADD-ANO-011/012 (one node, two answers) by a different mechanism; not recorded.

A fifth: `HeatNode.initial_state()` returns `float32` under x64 (`heat.py:565`) and `grid_x` is pinned float32 (`heat.py:434,438`), so a non-uniform `grid_points` perturbation of 1e-8 relative is discarded bit-for-bit. The convergence studies escape this only because they bypass `initial_state()` and use a uniform grid.

**Evidence:** `repro/r7_float32_casts.py`, `repro/r7b_float32_value_error.py`

```
=== RigidBodyNode: omega(T) = torque/I * T, exact for symplectic Euler ===
  I=0.33333333333333331 N=   10  omega=2.999999910593036  exact=3.000000000000000  rel err=2.980e-08
  I=0.33333333333333331 N= 1000  omega=2.999999910592995  exact=3.000000000000000  rel err=2.980e-08
=== BallNode control: gravity NOT cast in update(), IS cast in derivatives() ===
  update()      dv over dt=1 : -9.810000000000000   rel err=0.000e+00
  derivatives() dv/dt        : -9.810000419616699   rel err=4.277e-08
  -> update() and derivatives() of the SAME node disagree about g by 4.196e-07 absolute
=== gradients survive the cast? d omega / d inertia ===
  AD grad = -0.899999976   exact = -0.900000000   central FD(h=1e-6) = -0.898540047
  central FD(h=1e-9) = 0.000000000
```

**Why it happens:** `rigid_body.py:240-241` (`update`), `:299-300` (`derivatives`), `rigid_body_2d.py:201`, `ball.py:152`, `heat.py:434,438,565` — all unconditional `jnp.asarray(..., dtype=jnp.float32)` rather than following the state's dtype.

**What would make this a non-issue:** (a) the order studies catching it — checked: the finest level of MADD-VER-008/011 reaches ~5.5e-4 relative error, four orders above the 3.7e-8 floor, so no ladder in the tree can see it; (b) destroyed AD gradients — checked: they survive (JAX's `convert_element_type` JVP passes the tangent through), so this is a precision floor, not a zero gradient; (c) a no-op at the framework default — true, and why MADD-ANO-013 is rated *minor*; it bites only callers who enable x64 precisely to get a converged or verifiable result.

**Suggested fix:** none is worth a behaviour change alone; **widen MADD-ANO-013 to name every site**, or open one anomaly for "nodes pin float32 in `initial_state()` and on parameters", which `TODO.md` already flags as a three-way public-contract decision under "AdaptiveNode dtype policy". Actually removing the casts risks the dtype-promotion breakage that entry describes.

---

### MEDIUM — `scripts/check_heat_stability.py`: a false negative, a live count inflation, and a scope hole

**(a) Positional `stencil_order` is invisible — the gate passes a construction `__init__` refuses.** `_POSITIONAL` (`check_heat_stability.py:59`) stops at `thermal_diffusivity` (index 4), so `stencil_order` given positionally (index 6) is never picked up and the default `2` is used, judging a 4th-order rod against `0.5` instead of `0.3125`.

```
$ python scripts/check_heat_stability.py <dir containing>
      HeatNode("d", 0.004, 10, 1.0, 1.0, 0.0, 4)      # Fourier = 0.4, order 4
OK: 1 HeatNode construction(s) verified within their stencil's stability limit
exit=0
$ python -c 'HeatNode("d", 0.004, 10, 1.0, 1.0, 0.0, 4)'
ValueError: timestep 0.004 is unstable ... Fourier number ... 0.4, above the 0.3125 limit
```

**(b) Count-vs-coverage conflation, with a live instance.** `seen.append(...)` at `:139` runs *before* the two `continue`s at `:142-146` (`min(dt, n_cells, length, alpha) <= 0`; `MAX_FOURIER_NUMBER.get(order) is None`). Constructions the gate declined to evaluate are counted in the headline. Instrumented on the real repository scope:

```
SKIPPED-BUT-COUNTED (unknown stencil_order=3): tests/core/test_spatial_accuracy.py:46
OK: 132 HeatNode construction(s) verified within their stencil's stability limit;
    103 further construction(s) have computed arguments and were NOT checked
```

**131 were verified, not 132.** This is the "16 verified of 17" bug reproduced in the file whose own docstring and PR history are about not doing that. The live instance is benign (a deliberate `pytest.raises` negative test), but the mechanism inflates by however many unevaluated constructions exist.

**(c) `heat.HeatNode(...)` is not recognised.** `getattr(node.func, "id", None) == "HeatNode"` (`:113`) matches only a bare `Name`. Alone in a scope the zero-scope guard fires (fail-closed, good); in a scope with any other construction it is silently missed:

```
--- attribute-spelled unstable rod (Fo=0.66, order 4) + one valid construction ---
OK: 1 HeatNode construction(s) verified within their stencil's stability limit
    exit=0
```

No attribute-spelled call exists in the tree today, so (c) is latent.

**Evidence:** `repro/gate_probes/*.py`, `repro/run_gate_probes.sh` → `check_heat_stability_gate_probes.log`. Exit codes verified directly: (a) 0, (b) 0, (c) 0 in mixed scope / 1 in single-file scope, positive control 1.

**What would make this a non-issue:** (a) positional `stencil_order` being impossible — it is not, it is the 7th positional parameter; (b) the conflated count never being quoted — the R2 brief records a wrong "verified" count from `check_impl_mapping` being quoted into three earlier audit reports as evidence of coverage, which is precisely the harm.

**Suggested fix:** move `seen.append` below both `continue`s and report declined constructions in a third bucket beside `unchecked`; extend `_POSITIONAL` to the full signature order (`name, timestep, n_cells, length, thermal_diffusivity, initial_temperature, stencil_order`); match `ast.Attribute` with `.attr == "HeatNode"` as well as `ast.Name`. Risk: (b) lowers the advertised count 132 → 131 and anything quoting 132 must be updated; (c) may surface constructions in modules nobody expected to be in scope.

---

### MEDIUM — the *rotational* half of the "semi-implicit in both DOFs" claim is pinned by nothing

**What breaks:** `RigidBody2DNode.meta.discretization_order.notes` says "Semi-implicit (symplectic) Euler in **both the translational and the rotational** degrees of freedom"; `RigidBodyNode`'s says "1st order globally in velocity **and angular velocity**". The translational half is pinned. The rotational half is not.

**Evidence:** `repro/mutate.sh` → `mutation_log.txt`

```
=== MUT-RB-ORIENT: rigid_body.py  omega_to_quat(ang_vel_new) -> omega_to_quat(ang_vel) ===
  tests/nodes/test_rigid_body.py + both MMS modules          -> 112 passed, 4 xfailed
  + test_builtin_nodes_verified.py, test_coupling_solver_equivalence.py -> 103 passed
=== MUT-RB2D-ANGLE: rigid_body_2d.py  angle + omega_new*dt -> angle + omega*dt ===
  tests/nodes/test_rigid_body_2d.py + MMS ode nodes          ->  50 passed, 4 xfailed
  + the two files above                                      ->  97 passed
```

Translational controls are caught immediately:

```
=== MUT-RB: pos_new = pos + vel_new*dt -> pos + vel*dt ===
  FAILED tests/nodes/test_rigid_body.py::TestFreeFall::test_free_fall_matches_analytical
  assert 5.099911689758301 == 5.090095000000001 ± 5.1e-04
=== MUT-RB2D: x_new/angle_new use old v/omega ===
  FAILED tests/nodes/test_rigid_body_2d.py::TestFreeFall::test_free_fall_position
```

**Why it happens:** the free-fall tests exercise translation under gravity only; nothing drives a torque and compares against the closed-form semi-implicit angle. The order ladders cannot see it — both schemes are 1st order, which `test_an_order_study_cannot_tell_forward_from_semi_implicit_euler` already asserts for the spring. `_forward_euler_agrees()` (`tests/verification/test_mms_order_ode_nodes.py:737`) is exactly the right probe and is applied to `BallNode`, `HeartPumpNode` and `SpringDamperNode` — not to either rigid body.

**What would make this a non-issue:** (a) the code already being wrong and me having it backwards — checked: the code *is* semi-implicit today, so this is an unguarded invariant, not a live defect; (b) another suite catching it — checked the four files that assert on `angle`/`orientation` values plus both MMS modules: all green under both mutations.

**Suggested fix:** extend `_forward_euler_agrees` to `RigidBodyNode` (`"orientation"`) and `RigidBody2DNode` (`"angle"`) with a non-zero torque, as the control already does for the spring. Risk: none — a new assertion on behaviour that already holds. This is the same hole PR 88 closed for the integrators with the stability polynomial; the node layer has the tool and has not applied it everywhere.

---

### MEDIUM — `DEFAULT_ORDER_EXCESS`'s justification does not reproduce, and no value of it could have caught the defect it is credited for

**What breaks:** `src/maddening/testing/mms.py:341-351` justifies `DEFAULT_ORDER_EXCESS = 1.0` with "the corrected fourth-order stencil measures **5.02** over one pair of a fourth-order ladder". Over four manufactured solutions (the release's own plus three I chose) on 10/20/40/80/160 with the correct cubic closure, the **largest pairwise order anywhere is 4.126**. `DEFAULT_ORDER_SHORTFALL = 0.25` cites "HeatNode 1.982 against 2" (now 2.000) and a coarsest pair "as much as 0.16 low (1.847)"; on my four heat profiles the coarsest pair is 2.007–2.023 (*above* 2), on the five ODE ladders 1.001–1.007, and on my LBM ladder 1.985. Confirms TODO FOLLOW-UP A and widens its evidence base: the true maximum is 4.13, not the 3.98 recorded there.

**The consequence is not just an unsupported constant.** With the pre-0.4.0 flat-ended profile, a linear (genuinely *2nd*-order) ghost closure measures **4.080** and passes `[3.75, 5.00]`. So `excess = 1.0` is part of why the disarmed-profile case got through — but tightening it cannot be the fix, because the legitimate `tanh_bump` study reaches **4.126 > 4.080**. **No single global excess separates the defect from a correct study.** The profile-curvature property test is the only real defence, and the band adds nothing to it.

**Evidence:** `repro/r8_order_band.py` → `r8_order_band.log`

```
--- CORRECT (cubic) ghost closure: largest pairwise order ---
  release  (sin + 0.5x + 1 + 0.4x^2)    orders=['3.760','3.831','3.913','3.957']  max=3.957
  flat-ends (sin + 0.5x + 1) [pre-0.4]  orders=['3.759','3.830','3.913','3.957']  max=3.957
  exp_sin                               orders=['3.999','3.926','3.946','3.971']  max=3.999
  tanh_bump                             orders=['4.126','4.025','4.005','4.001']  max=4.126
  >>> largest pairwise order anywhere: 4.1261   (mms.py justifies EXCESS=1.0 with 5.02)

--- MUTATION: linear (genuinely 2nd-order) ghost closure ---
  release                observed=2.000  band[3.75,5.0]=FAIL
  flat-ends [pre-0.4.0]  observed=4.080  band[3.75,5.0]=PASS   band[3.75,4.05]=FAIL
  exp_sin                observed=2.003  band[3.75,5.0]=FAIL
  tanh_bump              observed=2.001  band[3.75,5.0]=FAIL
--- MUTATION: quadratic ghost closure ---   all four observed ~2.99, band=FAIL
--- SHORTFALL justification: coarsest-pair order, 2nd-order stencil ---
  coarsest = 2.023 / 2.022 / 2.007 / 2.016   (docstring cites 1.847)
```

**What would make this a non-issue:** (a) 5.02 coming from a ladder I did not run — possible; I tried four profiles, two Fourier numbers and a 5→320 ladder and never exceeded 4.13, and the docstring names no fixture; (b) the 1.847 coming from the LBM or rigid-body ladders — the sentence attributes it to "the same ladders", and I could not reproduce it on any of the seven I ran.

**Suggested fix:** re-derive both constants against a recorded fixture and quote the fixture, or delete the specific figures and justify the band on principle. Do **not** tighten `excess` toward the measured maximum — it would false-fire on `tanh_bump` at 4.126 while still admitting the 4.080 defect. The honest statement is that `excess` catches an exactly representable manufactured solution (where the error is zero and the `isfinite` guard fires anyway) and little else; `TestTheSteadyProfileCanSeeABrokenScheme` is what does the work.

---

### LOW — `HeatNode.boundary_flux_spec` declares `W/m^2` for a quantity in `K·m/s`

`-alpha * dT/dx` with `alpha` in m²/s and `T` in K has units K·m/s. The conductive flux is `-k dT/dx = -rho*c_p*alpha*dT/dx`, and `rho*c_p` is not a parameter of this node, so `output_units="W/m^2"` (`heat.py:860,864`) is short by a factor of `rho*c_p` (~4.2e6 for water). Nothing in-tree converts using this field — `graph_manager.py:3397-3415` only *warns* on a mismatch — but `fmi/model_description.py:581,650` copies it into the exported `modelDescription.xml`, so an FMU consumer is told the wrong unit. Evidence: `repro/r11_heat_misc_claims.py`, CLAIM D.

### LOW — `_laplacian_4th_order_uniform`'s docstring names the wrong fallback cells

The docstring says "Boundary cells (i=0,1,n-2,n-1): fall back to 2nd-order"; the code is `use_4th = (idx >= 1) & (idx <= n - 2)`, i.e. only `i=0` and `i=n-1` (`heat.py:203-205`). The **code is right** — with two ghosts, cell 1 has a full 5-point stencil — and the docstring is stale by one cell each side. Evidence: `repro/r11_heat_misc_claims.py`, CLAIM B.

### LOW — the stability-polynomial gate alone does not pin the RK4 scheme

`tests/verification/test_integrator_order.py:302` says the polynomials are "a complete statement of their Butcher coefficients up to the compensations the polynomial cannot see (there are none for a 4-stage explicit method of this shape)". I constructed one: `c = (0, 1/2, 1/2, 2)` with `b = (1/4, 1/3, 1/3, 1/12)` satisfies `b·1 = 1`, `b·A1 = 1/2`, `b·A²1 = 1/6`, `b·A³1 = 1/24` — RK4's exact `R(z)` — but `b·c² = 1/2 ≠ 1/3`, so it is only 2nd order on a nonlinear problem. Under that mutation all four algebraic gates pass (22 passed) and `test_carrying_time_in_the_state_restores_each_methods_classical_order` catches it at 2.013. **The composition is sound**; only the parenthetical is false. Evidence: `mutation_log.txt`, M5 / M5b.

### LOW — the two-mode understatement test is one-sided

`tests/core/test_coupling_error_bound.py::test_a_hidden_slow_mode_is_the_recorded_size_and_is_not_flagged` asserts `understatement > 50.0`, which catches an improvement (its stated purpose) but is blind to a regression — an estimate understating by 10,000x instead of 122x would pass. Its sibling `test_the_estimate_is_invariant_to_the_relaxation_factor` is correctly two-sided (`0.9 <= ratio <= 1.15`).

---

## Unverified suspicions

**`derivatives()` and `implicit_residual()` have no `params`, so a calibrated parameter cannot reach `integrate_node` or `implicit_euler_step`.** This is TODO FOLLOW-UP B and it reproduces (`repro/r12_params_never_reach_derivatives.py`): with constructor stiffness 100 and a calibrated 400, `update(params=fitted)` gives `velocity = -4.0` while `integrate_node`, `euler_step(node.derivatives, ...)` and `implicit_euler_step` all give `-1.0`, and there is no argument by which 400 could be supplied. Listed as a suspicion, not a finding, because **no docstring on those two methods promises otherwise** and nothing inside `src/maddening` calls either entry point (`implicit_euler_step` has zero internal callers; `integrate_node` is public API only), so I cannot show a documented guarantee broken or a graph run producing a wrong number. *What would falsify it:* a graph path routing through `derivatives()` with injected params — I grepped every call site and found none.

**The `ast.Name`-only matching in `check_heat_stability.py` may repeat in the other four compliance scripts.** Finding (c) is confirmed for this gate; I did not check the others — another auditor's surface, flagged because the pattern is identical.

**`_group_thresholds` in `sysid.py:172` uses `thr = 1.0` for the `mixed` and `interface` convergence norms and `g.tolerance` otherwise.** I did not chase whether those two norms really are pre-normalised by the tolerance. *What would falsify it:* a group with `convergence_norm="mixed"` and `tolerance != 1.0` whose sysid mask disagrees with `coupling_diagnostics()['converged']`. Not built — the coupling auditor is better placed.

---

## What I checked and found sound

**All seven declared orders, measured independently.** Own manufactured solutions, own ladders, own norms; `maddening.testing.mms` used only via `declared_order` to read the claim. `repro/r1_heat_order_independent.py`, `repro/r6_declared_orders.py`, `repro/r10_lbm_order.py`.

| node | axis | declared | observed (finest pair) | ladder |
|---|---|---|---|---|
| HeatNode `stencil_order=2` | space | 2 | **2.0004** (release profile), 2.0001 (exp_sin) | 10..160 |
| HeatNode `stencil_order=4` | space | 4 | **3.9569** (release), 3.9714 (exp_sin), 3.9795 (5..320) | 10..160 |
| LBMNode D2Q9 | space | 2 | **1.9984** (two-mode Kolmogorov, my own closed-form force) | 16..128 |
| SpringDamperNode | time | 1 | **1.0009** | 100..1600 |
| BallNode | time | 1 | **1.0003** | 100..1600 |
| RigidBodyNode | time | 1 | **1.0003** (position+velocity+omega) | 100..1600 |
| RigidBody2DNode | time | 1 | **1.0001** | 100..1600 |
| HeartPumpNode | time | 1 | **1.0002** | 100..1600 |

**`MAX_FOURIER_NUMBER` re-derives exactly as its docstring instructs.** Building the operator column by column and bisecting on `max|1 + Fo*lambda| <= 1` (`repro/r2_fourier_bound.py`): order 2 gives **exactly 0.500000 for n = 5..320**, as claimed; order 4 gives **0.316920 at n=5** rising **monotonically** to **0.324851**, against the docstring's "0.3169 at n = 5, rising monotonically to 0.3249". `5/16 = 0.3125` sits below all of them. Every digit is right.

**Every numeric claim in `integrators.py` reproduces** (`repro/r3_integrator_orders.py`):

| | frozen `u` | constant `u` | stage-time `u` |
|---|---|---|---|
| euler | 1.022 (doc 1.02) | 1.003 | 1.022 |
| heun | 1.017 (doc 1.02) | 2.005 | 1.999 (doc 2.00) |
| rk4 | 1.016 (doc 1.02) | 4.006 | 4.002 (doc 4.00) |

rk4/euler error ratio under the frozen input: **1.534 / 1.540 / 1.543** at n = 80/160/320 (docstring "~1.5x less accurate"). The Butcher nodes a derivative-1 time field sees are exactly `(0,)`, `(0, 1)`, `(0, 1/2, 1/2, 1)`.

**Thirteen gate mutations, all caught, each naming the right thing** (`mutation_log.txt`).
*Integrators* (baseline 32 passed): dispatch `rk4 -> heun_step`; rk4 stage offset `0.5 -> 0.4`; rk4 output weight `2.0 -> 2.05`; heun weights `0.5/0.5 -> 0.45/0.55`; and the `R(z)`-preserving substitution.
*Coupling* (baseline 30 passed, 1 xfailed): `relaxation_step_scale` always 1.0 → `test_the_estimate_is_invariant_to_the_relaxation_factor`; `estimated_error` loses its residual floor → `test_a_growing_residual_is_rejected_rather_than_extrapolated`; `error_amplification` drops the two-step `sqrt` guard → `test_an_alternating_sequence_is_not_flattered_by_its_good_half`; drops the `rho < 1` rejection → same growing-residual test.
*sysid* (baseline 55 passed): cutoff loses `sqrt(m)` → `test_a_long_residual_does_not_make_a_missing_direction_identifiable`; `sqrt(m) -> m` and `_PRECISION_WARN_FACTOR 2.0 -> 1.01` → `test_a_rank_decided_at_the_float32_floor_warns_with_the_numbers`; `_precision_limited` drops the `eps_floor` condition → `test_a_raised_rank_rtol_is_a_modelling_choice_not_a_precision_limit`.
**Both edges tested:** widening `_PRECISION_WARN_FACTOR` to 8.0 is invisible to `tests/core/test_sysid_degenerate_inputs.py` (55 passed) and is caught only by `tests/verification/hypothesis/test_hypothesis_sysid.py::TestPrecisionLimitedRank::test_the_threshold_still_separates_the_two_populations` (`109/585 <= 0.02` fails) — exactly as PR 89 recorded. That gate lives under `tests/verification/hypothesis/`, so it *is* inside the `verify-hypothesis` `ci`-profile job; the R2 brief's profile-coverage gap does not bite here.

**The GPU / TF32 concern in my brief is refuted.** One GPU run (`.venv-cuda`, jax 0.11.2, `CudaDevice(id=0)`, `jax_default_matmul_precision=None`) on an exactly rank-deficient problem returned `rank=1/2`, `crb=[inf inf]`, no warning — under the GPU default, under `precision="highest"` and under explicit `precision="tensorfloat32"` (`r5b_fim_tf32_gpu.log`). A faithful CPU emulation (`repro/r5c_tf32_floor_emulation.py`: operands rounded to 10 or 7 explicit mantissa bits, fp32 accumulation, which is what tensor cores do) explains why — floor as a fraction of the cutoff `max(n, sqrt(m)) * eps_f32`:

```
  n      m     cutoff |       fp32       x |  TF32(10b)       x |   BF16(7b)       x
  2   4000  7.539e-06 |  6.205e-08    0.01 |  1.210e-07    0.02 |  4.936e-06    0.65
  5   2000  5.331e-06 |  3.117e-08    0.01 |  6.318e-08    0.01 |  4.481e-06    0.84
 12    800  3.372e-06 |  5.789e-08    0.02 |  5.858e-08    0.02 |  4.306e-06    1.28
 25    800  3.372e-06 |  5.159e-08    0.02 |  7.162e-08    0.02 |  3.731e-06    1.11
```

**Mechanism:** the Gram structure *squares* operand-rounding error. A null direction perturbed by relative `delta` contributes `delta²` to `lambda_min/lambda_max`, so TF32's 10-bit operands give `(4.9e-4)² ≈ 2.4e-7`, **0.02x** the cutoff — and tensor cores accumulate in fp32, which is the term the `sqrt(m) * eps` model already covers. The cutoff would only need revisiting for a true bf16 matmul (`x` reaches 1.1–1.3 for `n >= 12`), which nothing here selects. `fim` not inspecting the matmul precision is therefore not a live hole; it would become one if a bf16 default ever appeared.

**Also checked and sound:**
- The cubic ghost closure's algebraic identity: the 5-point and 3-point forms are bit-identical at cells 0 and n-1 and both equal `(16Tb - 25T0 + 10T1 - T2)/(5dx²)`, while differing in the interior — so "the 2nd-order fallback costs nothing" holds exactly (`r11`, CLAIM A).
- `HeatNode.compute_interface_correction` really is an identity against `update()` at both ends, to the last bit (`r11`, CLAIM C).
- `error_amplification` rejection semantics: non-decreasing residual, equal residuals, zero/negative/inf predecessor and NaN current all return `0.0`; a healthy contraction returns `1/(1-rho)`; a defaulted `prev2` gives exactly `sqrt` of the one-step rate (`r9`, CLAIMS 5–6).
- `estimated_error >= residual` over 20,000 random `(residual, amplification, step_scale)` draws: **0 violations** (`r9`, CLAIM 4).
- `relaxation_step_scale` returns `omega` for `"fixed"` and `1.0` for `"none"`, `"aitken"`, `"iqn-ils"`, `"iqn-imvj"` (`r9`, CLAIM 3).
- `assert_node_order_verified` raises `UndeclaredOrderError` on a node that declares no order rather than passing — "a gate must fail when it verified nothing", honoured in `mms.py`.
- The `bound_valid -> ratio_usable` rename carries a compat map (`graph_manager.py:1024-1050`); no stale `bound_valid` reads remain outside the deliberate compat tests.
- `stencil_order=4` is genuinely more accurate than the default at every level measured (5.5e-4 vs 1.6e-2 at n=10), so MADD-ANO-008's sharpest symptom is gone as well as its order.

**Two coupling docstring numbers I could not reproduce from the text alone, and do not treat as findings:** the "factor of fourteen" for the alternating sequence `0.5, 5, 0.25, 2.5` is the *rate* ratio `0.7071/0.05 = 14.1`, not the bound ratio (3.2x) — loose wording, mechanism correct and verified; and the two-mode "122x" needs the amplitudes in `_TWO_MODE_C = (1e-5, 1.0)`, which `tests/core/test_coupling_error_bound.py` pins with a concrete fixture, so the number is reproducible from the test even though the docstring does not state the setup.

## Process notes

`nodes/lbm_pipe.py`, `nodes/adaptive/`, `coupling/mapping_spec.py` and the IQN solvers were outside my brief. No full suite was run; the largest run was 136 tests (41 s). Every mutation was applied to a scratch copy of `src/` injected via `PYTHONPATH` (`repro/mutate.sh`) — neither the worktree nor the main checkout was edited at any point, and no `tests/` file was modified. Nothing was killed; all long runs used `timeout`, and the one background job I stopped was stopped by its own task ID. The shared venv was not written to; `.venv-cuda` was used read-only for a single invocation.

---

## Coordinator corrections folded in (2026-09-20)

- The `derivatives()` / `implicit_residual()` params gap is **TODO FOLLOW-UP B from PR 86**, not from my brief. My `repro/r12_params_never_reach_derivatives.py` is the first concrete demonstration of it. It stays classified as a suspicion, not a finding, for the reason given above.
- The count-vs-coverage conflation found here in `check_heat_stability.py` (132 reported / 131 verified) has an independently discovered sibling in `check_citations` (50 reported / 45 verified), found by the gates auditor this round.
