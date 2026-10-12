---
orphan: false
---

# Development record for v0.4.0

This page records how MADDENING 0.4.0 was built and checked. It is written for
reviewers, auditors and contributors. If you are upgrading from 0.3.1, read the
[release notes](../release_notes/v0.4.0.md) instead.

The release notes and the changelog were restructured before the release. They
now say what a user of 0.3.1 gets and what they have to change. Everything else
they held moved here: the account of each area with its audit rounds, the
defects found and fixed before the release, the claims inventories, and the test
and CI work.

**How the text was moved.** A section marked *moved* is text from
`docs/release_notes/v0.4.0.md` or from the `[Unreleased]` section of
`CHANGELOG.md` as they stood at commit `db22bb27`. A script copied it; nobody
retyped or edited it. Its numbers, statuses and counts are the ones its authors
measured when they wrote it, so some are out of date. Where it and the tree
disagree, the tree is right:

- the anomaly registry, `docs/validation/known_anomalies.yaml`, for the status
  of an entry;
- the generated [stability report](../developer_guide/stability_report.md) for
  stability labels;
- each claims inventory, `docs/validation/*_claims.yaml`, for its own rows.

**What is where.**

- [Defects found and fixed before the release](#defects-found-and-fixed-before-the-release):
  defects no tagged release had, by area, with their registry numbers.
- [Claims inventories and what they found](#claims-inventories-and-what-they-found).
- [Test and CI changes](#test-and-ci-changes), followed by the
  [verification evidence](#verification-evidence) and the
  [engineering notes](#engineering-notes).
- [The multi-GPU session runner](#the-multi-gpu-session-runner): development
  tooling in `benchmarks/multigpu/`, not part of the installed package.
- [Index of the anomaly registry](#index-of-the-anomaly-registry): every entry,
  reachable in 0.4.0 or not.
- [The account of each area](#the-account-of-each-area): the release notes'
  sections from "Calibratable graphs" to the REST audits under "Security", with
  every audit round, and the 0.4.0 stability freeze round.
- [The rest of the cycle's changelog](#the-rest-of-the-cycles-changelog).

## Defects found and fixed before the release

These defects were introduced and fixed during the 0.4.0 cycle. No tagged
release had them, so the release notes do not list them as fixes. Each is an
entry of the anomaly registry, with its reproducer and the test that pins the
fix; its `affected_versions` is `none`. The table was generated from the
registry at commit `db22bb27`.

104 entries, by area.

**Parameters and system identification**

| Entry | The defect, as the registry titles it |
|---|---|
| `MADD-ANO-018` | A calibrated params value reaches update() and cannot reach derivatives(), implicit_residual() or integrate_node() |
| `MADD-ANO-069` | hold_undetermined held every direction a fit's gradients had not spanned, so a short or fast-converging fit returned parameters above the loss it had reached |
| `MADD-ANO-093` | A node.params write made after compile was dropped by every later compile: compile() kept the stale gm.params leaf over the node's new value |
| `MADD-ANO-104` | A trainable parameter one step carried past the edge of its coordinate range stayed there for the rest of the fit, and fit_lm reported converged=True away from the optimum |
| `MADD-ANO-121` | fit_lm's Marquardt floor, scaled to the mean curvature, crushed the step of a parameter measured in small units, and fit_lm reported converged=True away from the truth |
| `MADD-ANO-122` | A pending node.params write reached the entry points that run the graph but not the code that reads gm.params: a jitted or differentiated sysid loss, the FMU export, save_state and GET /graph/params used the old value |
| `MADD-ANO-126` | fit and fit_multiple_shooting applied Adam's eps=1e-8 in the loss's own units: a loss near 1e-12 left its parameters where they started |
| `MADD-ANO-135` | The fitters' identifiability guard asked its questions in the optimiser's coordinates, so a parameter's units decided what it held: a damping the data determined, in units 1e-5 or 1e6, was held at its start and fit_lm's loss rose from 1e-11 to 13.9 or 0.1, silently |
| `MADD-ANO-136` | ParamSpec.check compared through jnp, which flushes a subnormal to zero on CPU: a float32 -1e-40 passed a (0, None) bound in check_params, the FMU's set and set_state and REST's PUT, and read back below the advertised min |
| `MADD-ANO-150` | Under jax_enable_x64, and with float32 constants in an x64 graph, the identifiability guard missed an exact degeneracy: full excited_rank, nothing held, the fitted scale 1-2% off its start |
| `MADD-ANO-151` | windowed_loss restarted every coupling warm start cold at each window: with a predictor or IQN-IMVJ the loss at the generating parameters was not zero, and a fit started there walked 6.4% away |
| `MADD-ANO-152` | fim's float32 rank verdict moved with noise_std once F = J^T J left the normal range: a determined direction dropped out, every crb inf, no warning |
| `MADD-ANO-153` | The fitters read wrongly typed arguments instead of refusing them: fit(tol=True) reported converged at an unfitted start, and a mask leaf "False" fitted its parameter |
| `MADD-ANO-174` | fit_lm reported converged=True at a wrong point, or at its unmoved start, for a float32 residual or parameter far from unit scale; fit returned its start |
| `MADD-ANO-175` | node.params writes that never reached gm.params: every write into a replaced mapping, and an in-place element write to a list or array |
| `MADD-ANO-176` | fim and fim_core reported rank 0 silently when F = J^T J flushed to exactly zero from a Jacobian that was not |
| `MADD-ANO-177` | A logit ParamSpec whose width overflows the leaf's dtype was accepted, and constrain mapped the midpoint to the upper edge and far coordinates to NaN |
| `MADD-ANO-202` | The fitters evaluated their objective at parameters other than the ones they returned: under a logit or offset log transform with bounds wide beside the value, best_loss was another point's loss, and a leaf left out of the fit was run perturbed |
| `MADD-ANO-203` | Assigning node.params a mapping reverted every other calibrated constant of the node to the node's own value |
| `MADD-ANO-216` | fit_lm reported converged=True beside a jump of a residual that is not differentiable in the trained parameters, at a loss far above the minimum and with no warning |
| `MADD-ANO-279` | A deep copy of a graph put a node's value back over a constant calibrated in gm.params: the copy ran and saved another value than the original |
| `MADD-ANO-289` | fim and fim_core read the rank from the eigenvalues of F = J^T J, whose rounding a constant or piecewise constant sensitivity aligns: two parameters that enter a residual only through their sum reported rank 2 of 2 and a finite Cramer-Rao bound, with no warning, in float32 and float64 (never released) |

**Coupling between nodes**

| Entry | The defect, as the registry titles it |
|---|---|
| `MADD-ANO-019` | A coupling iteration that diverged to a non-finite state was reported residual=0.0, converged=True |
| `MADD-ANO-070` | solver="ift" held a coupling group's non-floating fields at their first-pass values: a flag reading the iterate came back stale, and one read across an edge froze the map |
| `MADD-ANO-073` | Under acceleration="fixed" with relaxation below 1/(1+sqrt(rho)), a coupling group stopping on its first loop pass understated its distance about 2x and reported converged |
| `MADD-ANO-094` | Under Gauss-Seidel the coupling bound's float floor counted the worst node's evaluations, not the chain a pass composes: a stalled 32-relay ring read its spectral bound at 0.51x the true distance with spectral_usable=True |
| `MADD-ANO-115` | Below about 1e-33 the accelerators' steps and the report's spectral rate and gradient bound were formed from flushed differences: converged=True up to 7.4x the threshold from the fixed point, a gradient bound of 0.0 with gradient_bound_usable=True |
| `MADD-ANO-142` | gradient_relative_error_bound read up to 5.2x below the true gradient error with gradient_bound_usable=True: the resolvent applied to each secant was the Krylov-restricted Arnoldi factor, and Newton-Kantorovich used it too |
| `MADD-ANO-143` | For an array-valued constant the gradient bound covered one \|c\|-weighted random direction, not each entry: the gradient in a small entry read 23.6x above the bound, usable |
| `MADD-ANO-146` | spectral_error_bound read 0.94x the true distance in the norm it documents (the returned state's weights) on a group still growing toward its fixed point, spectral_usable=True |
| `MADD-ANO-154` | The IFT tangent and adjoint of a coupling group read exactly 0.0, reported successful, when the tangent or cotangent had a NaN or infinite entry |
| `MADD-ANO-162` | strict_convergence on a coupling group with a sharded member aborts the process |
| `MADD-ANO-187` | spectral_error_bound under convergence_norm='interface' with a transform on an internal edge read 0.0014-0.098x the true distance, spectral_usable=True |
| `MADD-ANO-188` | gradient_relative_error_bound read 0.65-0.986x the true gradient error, gradient_bound_usable=True, on a group stopped far from its fixed point |
| `MADD-ANO-195` | The interface convergence norm reads a mapped edge's source field, not what the edge delivers |
| `MADD-ANO-211` | The float floor of an interface edge that delivers a wider dtype than its source field was taken at the wider dtype's eps: a stalled float32 group read its spectral bound at 6.7e-7x the true distance, spectral_usable=True |
| `MADD-ANO-213` | Under the interface norm a field read by several internal edges was counted once per edge by the residual and once by the spectral analysis: on a star spectral_error_bound read 0.26 to 0.73 of the true distance, spectral_usable=True |
| `MADD-ANO-221` | The Arnoldi breakdown test was 1e-5 of the product in every dtype: rho_spectral read 0.379 for 0.5 and spectral_error_bound 0.61x the true distance, spectral_usable=True |
| `MADD-ANO-222` | A coupling loop that passes through a change of a field below that field's float rounding is not in rho_spectral: a float32 sub-cycled ring with a field 1e-6 of its driver reads 1e-12 for 1.25e-4, spectral_usable=True |
| `MADD-ANO-223` | The spectral radius of the compressed Jacobian was taken by repeated squaring, which is not backward stable for a non-normal matrix: a float32 ring of five read rho_spectral 0.0512 for 0.0500, and graded matrices of five 0.9555 for 0.9152, spectral_usable=True |
| `MADD-ANO-224` | spectral_usable tested only the Arnoldi residual, which bounds a Ritz value's error for a normal Jacobian alone: a non-normal group of twelve scalars read rho_spectral 0.941 for 0.735 and a float32 hub 0.574 for 0.092, flag set |
| `MADD-ANO-225` | Past eight interface scalars a non-normal group's rho_spectral can read settled outside the margin: twelve float32 scalars read 0.273 for 0.219, spectral_usable=True |
| `MADD-ANO-226` | gradient_relative_error_bound of an iterate stopped at its float floor can read below the error: 5.1e-6 for 3.0e-5 on a float32 Jacobi ring of mapped edges, gradient_bound_usable=True |
| `MADD-ANO-229` | A spectrum reported settled from sampled rounding could be outside its margin: float32 groups of rank six read rho_spectral 0.250 for 0.206 and 0.107 for 0.056, spectral_usable=True |
| `MADD-ANO-231` | coupling_diagnostics() measured the residual's float floor on the live state: after set_node_state the same step read another spectral_error_bound (0.0 with spectral_usable=True where every member was written to zero) |
| `MADD-ANO-234` | gradient_relative_error_bound read below the error, gradient_bound_usable=True, with a constant in it whose tangent is rounding: 1.09 for a relative error of 1.134 |
| `MADD-ANO-238` | A coupling group with a zero-size floating field fails to trace under solver='ift' with diagnostics=True: a bare ValueError from a reduction over no entries |
| `MADD-ANO-242` | spectral_error_bound of a group with a multilinear_grid edge read 0.13x the true distance with spectral_usable=True when a marker's fixed point lay across a lattice plane from the returned iterate |
| `MADD-ANO-252` | gradient_bound_usable=True on a gradient_relative_error_bound 15 to 70,000 times under the true error, and spectral_usable=True on a spectral_error_bound 17 to 1,294 times under the distance, for a group with a multilinear_grid edge whose fixed point lay just past a lattice plane that the Newton point stopped short of |
| `MADD-ANO-255` | The float floor of a group of mixed dtypes took each field at its own dtype's eps: a float64 field downstream of a float32 member diluted it and spectral_error_bound read 0.21x the true distance with the flags set |
| `MADD-ANO-256` | The dead band of an interface edge read at its source was asked of the source field alone: forces of 1e-8 handed on as 17 to 520 left the norm at atol=1e-6 and a Jacobi pair accepted after one pass 34% to 74% from its fixed point |
| `MADD-ANO-261` | Under "l2" and "mixed" the float floor of a coupling report did not count the rounding of the weights a geometry-dependent mapping forms: float32 positions 1000 to 2000 spacings into a grid kept both usable flags beside a bound at 0.03 to 0.19 of the distance, with no advisory |
| `MADD-ANO-276` | A coupling group whose boundary fluxes read each other in a loop converged, converged=True, to a state that does not solve the coupled equations |
| `MADD-ANO-284` | An integer or Boolean field carried by an edge inside a coupling group stopped the solve after one or two passes with converged=True, residual=0.0, far from the fixed point |

**Interface mappings**

| Entry | The defect, as the registry titles it |
|---|---|
| `MADD-ANO-196` | A hand-written Mapping class's weight table was kept by reference and unchecked: an edit of gm.params rewrote the mapping, and a weight could be silently left out of a restore |
| `MADD-ANO-210` | A '#' in an edge's field name gave two mapped edges on one field pair a single slot of weights |
| `MADD-ANO-215` | A length that moves the points a mapped edge was built from was taken after a fit or a checkpoint load had moved another leaf, or with that leaf in the same request: the mapping kept the old grid and the save did not reload |
| `MADD-ANO-237` | A geometry written as float32 after compile() bypassed the resolution refusal: on a grid float32 cannot resolve, step() ran and every sample was 0.0 |
| `MADD-ANO-243` | A marker exactly on the last lattice point of an axis failed the geometry self-check on an honest pass: every bound of the group withheld, with a reason blaming a member's derivative |
| `MADD-ANO-244` | Float32 fields read at float64 positions always failed the geometry self-check: every bound of the group withheld, with a reason blaming a member's derivative |
| `MADD-ANO-245` | A mapping of a registered kind short of a Mapping protocol member, or carrying another kind, was accepted by add_edge and written by to_dict(), and from_dict() then refused that config |
| `MADD-ANO-248` | A geometry written with another shape after compile() was broadcast by a mapping kind that does not check its own shape |

**Time stepping and multi-rate graphs**

| Entry | The defect, as the registry titles it |
|---|---|
| `MADD-ANO-127` | The adaptive error norm guarded its division with max(scale, 1e-300): in float64 an entry below about 1e-297 read its error up to 1e297 times too small |

**Graph construction and stepping**

| Entry | The defect, as the registry titles it |
|---|---|
| `MADD-ANO-072` | After jax.grad through run_scan, set_node_state wrote into the traced state and the next entry point discarded the write; reset_state raised on a predictor group |
| `MADD-ANO-218` | A value handed to step or a scan ran in the dtype it arrived in, not the external input's declared dtype: an x64 graph with a default-declared (float32) input ran 0.1 as a float64 where its own FMU ran float32(0.1) |
| `MADD-ANO-249` | The underflow-range warning was asked once per compile, of the first stepped state alone: a state written into the range later, or already below tiny, stepped unwarned |

**Multi-GPU and sharding**

| Entry | The defect, as the registry titles it |
|---|---|
| `MADD-ANO-066` | A ShardedStencilNode nested in another stacked a per-shard domain integral's initial value twice, so step() changed the state's shape |
| `MADD-ANO-081` | After a sharded static was rewritten in place, compile() traced the new step against the wrapper's cached copy of the old buffer |
| `MADD-ANO-267` | sharded_cg(differentiable=True) given an x0 started its tangent and cotangent solves from that x0: with max_iters=0 the derivative was x0 itself, and at \|b\| of 1e5 a unit-direction JVP was off 2.6e-2 (never released) |
| `MADD-ANO-268` | With differentiable=True an atol was applied to the tangent and cotangent solves of sharded_cg and sharded_gmres as well: a derivative was not linear in its direction, and zero for a direction under the floor (never released) |

**Saving and loading**

| Entry | The defect, as the registry titles it |
|---|---|
| `MADD-ANO-200` | load_state loaded an integer of the other signedness wrapped: -1 for a uint32 field as 4294967295, 4000000000 for an int32 field as -294967296 |
| `MADD-ANO-207` | A checkpoint value loaded into a bfloat16 leaf was cast with no check: a float32 1e-44 loaded as 0.0 and 3.4028235e38 as inf |
| `MADD-ANO-236` | A checkpoint of a sparse mapped edge loaded into the dense edge of the same key, and the reverse, returned normally and installed none of the saved weights |

**FMU export**

| Entry | The defect, as the registry titles it |
|---|---|
| `MADD-ANO-054` | The FMU TCP bridge unpickled the state blob an importer hands to fmi3SetFMUState: remote code execution for anyone who could reach its port |
| `MADD-ANO-080` | The FMU bridge split a variable's name at its first dot to find the node, so every input of a node whose name holds a '.' was filed under a node the graph does not have |
| `MADD-ANO-082` | The FMU bridge installed an importer's FMU-state archive without the value checks set applies: a parameter outside its declared bounds and non-finite state were accepted |
| `MADD-ANO-083` | A second FMU instance on the same bridge started from the previous instance's final state, parameters, inputs and time |
| `MADD-ANO-084` | An external input the FMU did not export, or that an in-process caller omitted, was not zero-filled: the node took its own "input missing" branch |
| `MADD-ANO-085` | The FMU advertised min / max for its parameters but enforced them only if the sidecar had been given param_specs: a set or an archive outside them was accepted |
| `MADD-ANO-100` | The FMU bridge took any master_dt: a doStep ran h / master_dt graph steps and reported t + h, so a 0.01 s graph served with master_dt=0.005 labelled a state at 0.10 s with the time 0.05 s |
| `MADD-ANO-101` | A number for a Boolean FMU variable was stored by its truthiness: set gate.open = 0.5, 2.0 or -3.0 read back as true, and an FMU-state archive holding 0.25 in a Boolean field restored as true |
| `MADD-ANO-102` | The FMU's C wrapper carried every width as a double and the bridge could not tell the calls apart: fmi3SetBoolean wrote 1.0 into a Float32 parameter and fmi3GetInt32 truncated a Float32 output, both fmi3OK |
| `MADD-ANO-103` | An FMU-state archive missing its pending-input members restored with those inputs silently at zero |
| `MADD-ANO-119` | An FMU-state archive installed new interface-mapping weights, which are not FMI variables: the FMU then computed a coupling its model description does not describe |
| `MADD-ANO-137` | The FMU's value check refused a value its type overflows but stored one it underflows: a float32 set to 1e-50 read back as 0.0, the value and its sign lost |
| `MADD-ANO-147` | An FMU built over a pending structural node.params write ran the old model, while every graph entry point ran the new one |
| `MADD-ANO-148` | A log parameter with no lower bound was advertised with no FMI min, and a bridge whose sidecar had no specs accepted and ran mass = -1 |
| `MADD-ANO-149` | The FMU bridge's time tolerance had an absolute floor, and a biased importer's reported time left the simulated time behind without bound |
| `MADD-ANO-167` | A ParamSpec changed between the FMU description and its sidecar was enforced in place of the advertised min / max: the bridge accepted values its XML forbids, or refused values it declares settable |
| `MADD-ANO-168` | At a large ratio of time to master step the FMU bridge's ulp slacks exceeded a master step: a doStep a whole step ahead was adopted, and a clock running 40% fast was never refused |
| `MADD-ANO-169` | One in-limit get frame drove the FMU bridge past 6 GB and held the instance for about 45 s before its reply was refused |
| `MADD-ANO-197` | Restoring an FMU state started the bridge's drift count again at the restored time, so a master that saved and restored between steps was never held to the time-drift bound |
| `MADD-ANO-198` | The FMU wire rounded an Int64 / UInt64 above 2**53: fmi3GetInt64 answered fmi3OK with a neighbouring integer, and a JSON set of 2**53 + 1 was stored as 2**53 |
| `MADD-ANO-199` | A set of negative zero through the FMU's C wrapper on a JSON connection was stored as positive zero |

**REST API and streams**

| Entry | The defect, as the registry titles it |
|---|---|
| `MADD-ANO-063` | A new value of a parameter from which a node derives the points of a mapped edge was taken by PUT /graph/params and gm.params, run with the old points' mapping weights, and saved as a config that does not load |
| `MADD-ANO-089` | A checkpoint path that resolved to the checkpoint root itself wrote, and loaded, <root>.npz in the root's parent |
| `MADD-ANO-131` | POST /surrogate/train's memory budget was checked on the graph at the request, and the job swept the graph as it was later: a node added in between bypassed it |
| `MADD-ANO-140` | A generated API token written to MADDENING_API_TOKEN_FILE kept the mode of a file already at that path: a placeholder of mode 0644 left the token readable by every local user |
| `MADD-ANO-141` | POST /checkpoint/save refused at its final move answered "nothing was written" having replaced the manifest of that name, and checkpoint refusals named the server's absolute paths |
| `MADD-ANO-163` | POST /checkpoint/load restored parameter values PUT /graph/params refuses -- past a node's constructor limit, outside a ParamSpec's bounds, non-finite, a boolean, a numeric string -- answered 200, and left a graph whose save did not reload and whose steps diverged |
| `MADD-ANO-165` | POST /sim/run that could not have the graph part-way through answered 503 'Nothing was changed; retry shortly' after it had stepped, so a client that retried stepped the graph twice; its final read waited with no timeout |
| `MADD-ANO-178` | A PUT /graph/params that wrote a leaf back to the node's own value skipped the combined checks: after a fit or a checkpoint load, a 200 left a rod past its Fourier limit and a save that did not reload |
| `MADD-ANO-179` | X-Forwarded-For let a web page past a loopback-bound API's DNS-rebinding defence and its peer backstop: uvicorn's default proxy headers made the peer any string a request named |

**Tooling and compliance**

| Entry | The defect, as the registry titles it |
|---|---|
| `MADD-ANO-204` | The library's warning probes silenced every thread of the process, and two that overlapped in two threads could leave every warning ignored for good |

### The changelog's entries for them (moved)

The `### Fixed` entries of the changelog in which every registry number names
one of these defects.

- **A coupling group's float floor is counted at its coarsest member's resolution** (MADD-ANO-255, never released): each field was counted at its own dtype's eps, so a float64 field of `n` values computed from one float32 value (or float32 values from a 16-bit one) diluted the floor and `spectral_error_bound` read 0.21x the distance at `n = 2e4` with its flags set. What an internal edge delivers counts too (a transform that narrows to float32 between float64 members: such a pair read its bound at 1e-7 of the distance with the flags set); a narrowing inside a node's own `update` cannot be seen and is not counted. Nothing changes in a group of one dtype whose edges keep it; otherwise the floor, and so `precision_limited`, can only rise. A position (the geometry field of a geometry-dependent mapping) keeps its own dtype's eps.
- **Under `convergence_norm="interface"` the dead band of an edge read at its source asks the field and what the edge delivers** (MADD-ANO-256, never released), and **`diagnostics=True` no longer asks for memory quadratic in the group** (a step raised `RESOURCE_EXHAUSTED` at 1e5 entries; the guide now gives the measured cost against the group's size).
- **`validate()` and `compile()` advise on a dead band (`atol > 0`) declared on any coupling group** (MADD-ANO-254, open): a change that has to cross a field inside the band is not seen until it reaches a field the norm keeps, and such a group can report `converged=True` after one pass thousands of tolerances from its fixed point (a Jacobi pair; a Gauss-Seidel ring of three swept against its data flow; Gauss-Seidel pairs whose loop passes its lagged read twice, and one member with two edges to itself). Leave `atol` at `0.0` in 0.4.0; a pair with one scalar field a member held where it was measured and is advised on all the same.
- **A coupling group that solves the positions of a `multilinear_grid` mapping has no usable flag in 0.4.0** (MADD-ANO-252, never released; EXPERIMENTAL tier). Where the group's pass reads a position from the iterate, or builds one and reads it in the same pass (a source-anchored geometry inside the group; the target-anchored geometry of a member that computes fluxes), `spectral_usable` and `gradient_bound_usable` are `False` on every step: `rho_spectral`, `spectral_error_bound` and `gradient_relative_error_bound` are reported as computed, uncertified, and `not_usable_reason` names the positions and says so. To keep the flags, fix every position during the pass: anchor the geometry at a target that `update` reads, or keep the node that holds the positions outside the group. Beside a lattice plane the flags had stood on a distance bound 17 to 1,874 times under the distance and on a gradient bound 15 to 70,000 times under the error; three rules that tried to certify the numbers there were each found short, so 0.4.0 certifies none (`docs/user_guide/geometry_dependent_mappings.md`, Limits).
- **Checks of state values hold after a state write** (MADD-ANO-248 and 249, never released): a geometry written with another shape after `compile()` (`set_node_state`) was broadcast by a mapping kind that does not check its own shape; the next `step()`/`run*` now raises `ValueError` naming the edge. `UnderflowRangeWarning` is due again at the first step after `set_node_state`, `load_state`, `PUT /graph/state` or `reset_state`, and reads the state that step starts from too (a state below `tiny` was flushed to zero unwarned); a graph written to before every step is asked at the first eight such steps of a compile and then once per 1024 stored states, since the check costs several steps of a small graph.
- **A checkpoint's mapping weights are installed or the load fails, never dropped** (MADD-ANO-236, never released): a checkpoint of a sparse mapped edge loaded into the dense edge of the same key (and the reverse) returned normally and installed nothing, because the two kinds name their weights `W` and `H`. `load_state` and `POST /checkpoint/load` now refuse it, and refuse saved mapping weights the graph has no leaf for.
- **A geometry's dtype rules hold after `compile()`** (MADD-ANO-237, never released): a `multilinear_grid` geometry compiled as float64 and then written as float32 by `set_node_state` stepped with every sample 0.0 on a grid float32 cannot resolve. The next `step()`/`run*` now raises `ValueError` naming the edge (`TypeError` for a dtype that is not float32 or float64); write the geometry back in float64.
- **`fit_lm` no longer reports `converged=True` beside a small jump of the residual read from a long rejected step** (MADD-ANO-216's residual case, never released): the one-sided test divided the jump by the linearisation's change over the whole step, so a late bounce 22 float spacings from the iterate read 341 and 456 against a threshold of `2**10` on the stock ball. A rejected step reading above `2**5` is now read again across the one float spacing where the residual departs most (a few residual evaluations, at the end of the run only); the run ends `converged=False` with the existing warning. Action: none.
- **`gradient_relative_error_bound` covers the constants one pass resolves** (MADD-ANO-234, never released): with a constant in it whose tangent at the returned iterate is rounding (the centre or curve of a nonlinearity evaluated on its centre) it read 1.09 for a relative error of 1.134, usable. Such a constant, and a gain whose whole value moves the pass by less than the residual's float floor, is no longer in the bound; the docstring says which constants are. Action: none.
- **`coupling_diagnostics()` describes the step that ran when the state is written afterwards** (MADD-ANO-231, never released): the float floor under `spectral_error_bound`, `precision_limited` and `spectral_usable` was measured on the live state, so `set_node_state` after a step moved that step's report (a bound of 0.0, flag set, with every member written to zero). The graph now keeps what the step left. A checkpoint saved after a member was written says so (an optional archive member), and the graph that loads it reports that group's bound as NaN and its `*_usable` flags False with a `not_usable_reason` until it steps; `_reports` is now a reserved node name. Action: none.
- **`coupling_diagnostics()`: three numbers that read wrong with their flag set, and one found beside them** (MADD-ANO-222, 225, 226, 229, never released): `gradient_relative_error_bound` at the float floor took the change along one stand-in direction (5.1e-6 for 3.0e-5); `rho_spectral` read settled past eight interface scalars (0.273 for 0.219), from sampled rounding on a float32 group far from normal (0.250 for 0.206), and lost a loop below a field's rounding on a sub-cycled group (1e-12 for 1.25e-4).
  The gradient bound's undirected distance now takes an operator norm; a Krylov space still growing at the cap is never settled (`spectral_usable=False` above seven interface scalars, eight where they are the whole state); a certificate over every perturbation of the measured size replaces the samples; a sub-cycled boundary value is differentiated as `(1 - alpha) a + alpha b` (values unchanged, gradients of sub-cycled groups move at rounding level).
  Action: none; expect `spectral_usable=False` on more float32 and 16-bit groups whose Jacobian is far from normal (usable fraction on the search's draws 0.979 to 0.976).
- **`rho_spectral` and `spectral_usable` under `diagnostics=True`** (MADD-ANO-221, 223, 224, never released): the spectral estimate took a Krylov direction below 1e-5 of a product for rounding in every dtype, took the compressed radius by repeated squaring, and called a spectrum settled on the Arnoldi residual alone; `rho_spectral` read 0.379 for 0.5 (float64), 0.0512 for 0.0500 (float32) and 0.941 for 0.735 (twelve non-normal scalars), and `spectral_error_bound` 0.61x the distance, each with `spectral_usable=True`.
  The breakdown test is now eight units of the products' own rounding, the radius is `eigvals`, and one more Jacobian-vector product (nine per group per step, was eight) checks the estimate against itself: where it moves the radius by more than 5% of `1 - rho_spectral` the flag is `False`.
  Action: none. The three the targeted search found beside these (MADD-ANO-222, 225 and 226: a loop below a float32 field's rounding, a radius read settled past eight interface scalars, the gradient bound at the float floor) are resolved by the `coupling_diagnostics()` entry above.
- **A params write is held to the points a mapped edge was built from whatever a fit or a checkpoint load did first** (`MADD-ANO-215`): `PUT /graph/params`, `POST /checkpoint/load` and a `gm.params` write refuse a `length` under a mapping when another leaf's live value differs from the one the node was built with, or is written in the same request; a structural value the live leaves allow is no longer refused for the built ones.
- **`convergence_norm="interface"` reads every internal edge as the step delivers it, and a static mapping onto more entries than its source holds at its source** (MADD-ANO-195, 211 and 213, never released): the norm, its float floor, the spectral bound's reading and the report applied an edge's transform and left its interface mapping out, so a mapped edge was judged on its source field -- a 9% change of the delivered value read 0.0071 of the tolerance, and `spectral_error_bound` 0.064-0.32x the distance in what the edge delivers, usable.
  One function now gives all four the step's value (the mapping with the step's weights, then the transform), or the source field itself where a static mapping delivers more entries than its source holds (the compact side; a tie is read as delivered): pooled into the norm, 30 marker forces scattered onto a grid left a converged group's forces 23 to 459 tolerances from their fixed point at 1e3 to 1e6 cells, and read at their source they are 3.3 to 3.6 at every size.  The step records the floor of a group with an edge read through its mapping (`reading_floor`), and a delivered value wider than its source field is resolved at the source's `eps` (a stalled float32 pair behind edges that cast to float64 read 6.7e-7x, usable).  A field that several internal edges read (a star's hub) was counted once per edge by the norm and once by the bound's analysis, which read 0.26-0.73x the true distance, usable; such a group's report is now analysed on the same reading.
  Action: none; a group without a mapped internal edge keeps its states, reports and compile counts bit for bit, except the floor of an interface edge that widens its dtype under x64 and the spectral keys of an interface group with a field read by several internal edges (with `diagnostics=True`; 8 more Jacobian-vector products per step).  MADD-ANO-212 (a Gauss-Seidel read of a difference within one field: bound 0.054x, usable) is open for 0.5.0.
- **sysid round-9 fixes: bounded transforms in the fitters** (MADD-ANO-104 corrected, MADD-ANO-202, 203, never released): `fit_lm` no longer ends on the bound of a `log`/`logit` parameter that an early step carried to the edge of its range (20 of 60 starts under x64 on the parameter guide's spring) -- a step is read on the transform's tangent where that moves the value less, and the range a fit steps in ends `sqrt(eps)` of the bounds inside each bound; the fitters make one evaluation of `theta -> params`, so `best_loss` is the loss of exactly the arrays returned and a leaf a fit does not move is never run through its transform (a damping left out by `mask=` at 2.0 under a wide `logit` was run at 2.0266, the stiffness beside it fitted to 30.06 for 30);
  a fit that ends on its transform's edge, an Adam fit started on one and a fitted value its transform cannot resolve to `sqrt(eps)` of itself are each named in a warning; `node.params = {**node.params, "k": v}` writes `k` and no longer reverts the node's other calibrated constants.
  Action: declare a parameter that belongs on its bound with `transform=None`; read the new warnings; a `fit_lm` run with `log`/`logit` parameters takes another path to the same answer.  SYS-144 to SYS-147 are new.
- **Warnings in a process with several threads** (MADD-ANO-204, never released): the library's sixteen warning probes (parameter writes, `PUT /graph/params`, the sharded wrappers, the profiler) used `warnings.catch_warnings()`, which silenced every thread while one probed and, for two that overlapped, could leave every warning ignored for good; a probe now silences its own thread only and saves no filters. Action: none; other code's `catch_warnings()` in threads is still unsafe (see "Warnings when several threads run graphs" in the parameters guide).
- **sysid round-8 fixes** (MADD-ANO-174 to 177, never released): `fit_lm`'s loss and solves are framed by powers of two, so a float32 residual or parameter from `1e-30` to `1e30` no longer reports `converged=True` at a wrong point or its start, and `fit` lifts a flushed gradient; `fim`/`fim_core` flag an `F` flushed to zero; in-place and replaced-mapping `node.params` writes reach `gm.params`;
  a `log`/`logit` spec the leaf's dtype cannot hold is refused; float32 leaves in an x64 graph no longer creep; `fit_lm`'s floor verdict no longer follows units or an on-bound gradient's rounding; counts read integers in every spelling.
  Action: none; an ordinary `fit_lm` run moves in its last bits (the equilibrated solve's pivots).  SYS-131 and SYS-132 are new.
- **FMU export, round-8 audit fixes** (MADD-ANO-167 to 169, never released): the bridge refuses to start when a `set_param_spec` since the description (or a sidecar spec) would enforce another `min` / `max` than the XML advertises, and holds every write to the advertised bounds too; a start, step end or restored time whose 16 ulps pass a tenth of the master step is refused, and no time slack exceeds a tenth of a step;
  a `get` past what one reply frame carries is refused before anything is read (one 64 MiB frame took 6 GB), a repeated reference is read once; an open `log` / `logit` bound is advertised as the outermost value `ParamSpec.check` accepts (a float32 `logit(-1, 1)` max was refused).
  Action: build the description, sidecar and bridge after the last `set_param_spec`; start an FMU at a time its master step resolves; compute communication points as `start + k * h`; re-package an FMU with a float32 `logit` leaf (its token may change).
- **sysid round-7 fixes** (MADD-ANO-150 to 153, never released): the identifiability guard holds an exact degeneracy under x64 and with float32 leaves in an x64 graph; `windowed_loss` replays coupling predictor and IQN-IMVJ warm starts across windows (zero loss at the truth); `fim` warns when its rank cutoff is below float32's normal range;
  bool and non-number hyper-parameters and non-bool mask leaves are refused; `ParamSpec.check` refuses a `log`/`logit` value without a finite coordinate; the nominal width guard tests the width, not its square; `fit_lm` reaches its float64 floor.
  Action: pass real numbers and bool mask leaves; a fit with a predictor or IQN-IMVJ group, or under x64, may return a different (correct) point.  SYS-071 is verified; SYS-127 to 130 are new.
- **FMU export, round-7 audit fixes** (MADD-ANO-147 to 149, never released): `build_model_description`, `FmuSidecar` and `FmuTcpBridge` refuse a graph changed since its `compile()` (an FMU built over a pending structural `node.params` write ran the old model); a `log` parameter without a lower bound advertises the smallest normal as `min`; the bridge's time tolerance is a millionth of the master step at any step, and the reported time stays within it of the simulated time;
  an input of a node the FMU does not export is held at zero, not exported; the C wrapper holds FMI 3.0's co-simulation state machine (no `fmi3DoStep` before initialization, no set once terminated, no `fmi3EnterStepMode`, configuration mode or `fmi3SetTime`).  FMU-012 and -017's wording is fixed.
  Action: `compile()` before exporting; initialize an FMU instance before reading or stepping it; re-package an FMU whose graph has a `log` parameter without a lower bound (its token changed).
- **The identifiability guard decides the same in any units** (MADD-ANO-135, never released): its tests, hold and tolerance measure an identity parameter relative to its size, as `fim(scale="relative")` does;
  **bounds checks compare exactly** (MADD-ANO-136): a float32 `-1e-40` no longer passes a `(0, None)` bound through XLA's subnormal flush; **a value its type flushes to 0 is refused** (MADD-ANO-137) like one it overflows.
  Action: none; a `fit_lm` with an identity parameter in units far from 1 may return a different (correct) point.  The sysid/FMU inventory's SYS-063, -088, -109, FMU-024 and -039 are verified (144 of 147 rows).
- **`fit_lm` no longer depends on the parameters' units, and a shrunken step cannot read as converged** (MADD-ANO-121, never released): the Marquardt floor is per column, and `converged` also needs the undamped Gauss-Newton step to be stationary;
  **every reader of `gm.params` sees a pending `node.params` write** (MADD-ANO-122, never released): one sync point, the `gm.params` getter, serves readers and runners alike, and the later write wins.  Action: none.
- **FMU export: schema-valid starts, a locale-proof wrapper, the FMI state machine** (MADD-ANO-119, never released): Boolean/integer `start`/`min`/`max` in their type's form and discrete; the C wrapper writes and reads numbers in the C locale; an archive cannot install mapping weights;
  the token covers starts and bounds; terminate refuses step/set/initialize until reset; `fmi3GetClock`/`SetClock` are `fmi3Error`; `FmuTcpBridge(idle_timeout=None)` waits for ever; a node named `x.params.y` has settable parameters.
  Action: rebuild FMUs (their tokens change) and re-package them with their bridge's description.
- **The FMU bridge and C wrapper refuse what they used to coerce or drop** (MADD-ANO-100 to 103, never released): a `master_dt` other than `md.graph_timestep`, a Boolean other than 0/1, an `fmi3Get`/`Set` of another type than the variable's, a repeated value reference, a numeric set of a Clock,
  an archive missing an input; step size and communication point share one tolerance; a step takes at most `max_steps_per_request` (100000) graph steps and stops at `stop()`; `SetFMUState` checks the whole frame; the wrapper waits at most `MADDENING_FMU_TIMEOUT` (600 s).
  Action: pass `master_dt=gm.timestep`; read and write each FMU variable with the function of its type (`getFloat32` for a Float32).
- **A `node.params` write after compile reaches the step at the next `compile()`** (MADD-ANO-093, never released: it was dropped); `fit_lm` reports `converged` at the float floor (its proposal is tested accepted or not; `step_tol` is relative, default 16 ulps); `windowed_loss` refuses a negative/NaN `continuity_weight` and a non-bool `mask_unconverged`;
  a clipped leaf on its bound has its full derivative into the range; a refused mask names the missing empty container.  Action: change a constant mid-run through `gm.params`; pass `step_tol` as a relative change.
- **FMU export, round-4 audit** (MADD-ANO-083 to 085, never released): a new instance starts at the description's start values (it inherited the last one's state, parameters and time); inputs `selected_inputs` leaves out are held at zero (`held_inputs`); `SidecarConfig.input_resolver` steps as `GraphManager.step`; advertised min/max hold without `param_specs`.
  `dt`/`t`/values must be numbers; `doStep` must start at the FMU's time, which `fmi3EnterInitializationMode` sets; the wrapper refuses a reply longer than `nValues` and an empty token, drops a half-sent frame and sets `TCP_NODELAY` (every call took >=40 ms).
  Action: pass `input_resolver=gm._resolve_external_inputs`; build the sidecar and the description from the same parameters; open one connection per instance.
- **`hold_undetermined` no longer returns a fit's parameters above the loss it reached** (MADD-ANO-069, never released): a direction is held only if the
  run's gradients missed it *and* the loss has no curvature there, and a hold that would raise the loss beyond rounding is refused (`FitResult.hold_declined`,
  `RuntimeWarning`).  `fit_lm` on a well-posed bowl went 0.0 -> 0.22.  Action: re-run guarded fits from earlier 0.4.0 builds, or compare to `best_loss`.
- **A coupling iteration that diverged to NaN/inf was reported `residual=0.0, converged=True`** by both
  solvers and all three norms (MADD-ANO-019 resolved); a non-finite field now fails the criterion (`residual=inf`,
  `converged=False`) and `strict_convergence=True` raises naming the non-finite state. No action needed.
- **A calibrated `params` now reaches `derivatives()`, `implicit_residual()`,
  `integrate_node(..., params=)` and `implicit_euler_step(..., params=)`** (MADD-ANO-018
  resolved); a non-empty `params` for an override without the keyword is a `ValueError`, not a silent drop
- **Audit fixes on the serialisation surfaces** (`MADD-ANO-010`): an FMU or node named
  `NaN`/`Infinity` is refused where you set the name, an unencodable FMI reply is an error
  reply rather than a dead worker thread, and `dumps_encoded()` writes `to_dict()` output
- **`gm.compile()` drops every node's materialised statics**, including one
  inside a wrapped node: `invalidate_static_cache` is now a `SimulationNode`
  method that forwards inwards, so a static rewritten in place is not baked in (MADD-ANO-081, never released)
- A `.` in a node name no longer misroutes that node's FMU inputs and outputs (MADD-ANO-080, never released)

## Claims inventories and what they found

A claims inventory lists every documented claim about one part of the library.
Each row gives the claim's conditions, the oracle that decides it and the test
that checks it. `tests/compliance/test_claims_inventories.py` holds every row to
its test. A row's domain cells say which precisions, schedules and binds the
claim was checked in. Four inventories were built in this cycle:

| Inventory | Row prefixes | Domains | What it covers |
|---|---|---|---|
| `docs/validation/coupling_claims.yaml` | `CPL` | `numeric` | Coupling groups: convergence, the error estimate, the bounds and their flags, acceleration, the IFT gradient. |
| `docs/validation/sysid_fmu_claims.yaml` | `SYS`, `FMU` | `numeric` | `maddening.sysid`, `ParamSpec`, `fim`, and the FMU export and bridge. |
| `docs/validation/rest_runpod_claims.yaml` | `REST`, `RPD` | `server` | The HTTP API's routes and refusals, and the multi-GPU session runner. |
| `docs/validation/mapping_claims.yaml` | `MAP` | `numeric` | Interface mappings: the kinds, sparse and geometry-dependent mappings, serialisation. |

What they found is written up in the account of each area:

- [The coupling claims inventory, and what it found](#the-coupling-claims-inventory-and-what-it-found);
- [The claims inventories' domain matrix, and what it found](#the-claims-inventories-domain-matrix-and-what-it-found);
- [The REST and run_pod inventory's server domains, and what they found](#the-rest-and-run_pod-inventorys-server-domains-and-what-they-found);
- [The coupling inventory's narrowed cells, lifted](#the-coupling-inventorys-narrowed-cells-lifted);
- [The coupling runtime, round-5 audit (against the claims inventory)](#the-coupling-runtime-round-5-audit-against-the-claims-inventory)
  and [round-6 audit](#the-coupling-runtime-round-6-audit-against-the-claims-inventory);
- [Units the guard could see, and values too small to check: the claims-inventory sysid fixes](#units-the-guard-could-see-and-values-too-small-to-check-the-claims-inventory-sysid-fixes);
- [The REST claims follow-up: what a route says it did, it did](#the-rest-claims-follow-up-what-a-route-says-it-did-it-did).

A registry entry that an inventory found names the rows that found it in its
description.

### The changelog's entries for the inventories (moved)

- **REST and run_pod claims inventory** (`docs/validation/rest_runpod_claims.yaml`, prefixes `REST`, `RPD`): every documented claim about the HTTP API and the multi-GPU session runner with its conditions, oracle and a test that can fail -- 143 rows, 132 verified, 5 failing (strict xfails), 5 ambiguous, 1 untested; the surrogate and streaming endpoints are out of scope (experimental).
  Read the failing and ambiguous rows before relying on those claims; a new REST or `run_pod.py` claim gets a row in the same change (`testing_standards.md`).
- **System identification and FMU claims inventory** (`docs/validation/sysid_fmu_claims.yaml`, `SYS-NNN` / `FMU-NNN`): every documented `sysid`, `ParamSpec` and FMU-export claim with its conditions, oracle and a test that can fail -- 147 rows, 139 verified, 4 failing (strict xfails), 4 ambiguous.
  `tests/compliance/test_coupling_claims.py` becomes `test_claims_inventories.py`, which checks every `docs/validation/*_claims.yaml` (each declares the id `prefixes` it owns).
- **Coupling claims inventory** (`docs/validation/coupling_claims.yaml`): every documented coupling claim with its conditions, oracle and a test that can fail -- 130 rows, 114 verified, 6 failing (strict xfails), 10 ambiguous -- checked by `tests/compliance/test_coupling_claims.py`.
  Read the failing and ambiguous rows before relying on those claims; a new coupling claim gets a row in the same change (`testing_standards.md`).
- **A domain matrix in the coupling and sysid+FMU claims inventories** (`domain_set: numeric`, a `domains:` mapping per row): for float32, float64, mixed dtypes, 16-bit, jit, gradients, vmap, multi-rate, sub-cycling, predictors and warm starts, `run_adaptive`, checkpoint restarts and sharding, each row names the test that exercises its claim there, `narrowed` (its conditions now exclude the domain) or `n/a`; `test_claims_inventories.py` checks it.
  It found four failing coupling domains (MADD-ANO-160 to 162, and a bfloat16 `rho_spectral` that is not float32-exact) and two minor sysid ones under x64 (SYS-024, SYS-071): eleven rows are `failing`, and many conditions are narrowed -- read a row's conditions before relying on it outside float32 (`testing_standards.md`, "The domain matrix").
- **A server-domain matrix in the REST and run_pod claims inventory** (`domain_set: server`): for loopback and non-loopback binds, the token demanded or not, simultaneous requests on a real loopback server, the runner, an in-flight `/sim/run`, oversized and hostile input, SIGTERM mid-request, wrapper nodes and a restored graph -- and run_pod's dry-run, relabelled and mixed-commit records -- each row names the test that exercises its claim there, `narrowed` or `n/a` (`testing_standards.md`, "The server set").
  Every row is `verified`: the four defects it reached (a `HybridNode` write lost, a `/sim/run` 503 saying nothing changed after steps, runner routes behind a saturated worker pool, a checkpoint load restoring values a write refuses) are the round-7 REST fixes' own, and its cells pass against them; sixteen rows' conditions are narrowed, and REST-098 (no TLS) is now pinned by a test -- read a row's conditions before relying on it outside the domains it names.
- **The coupling inventory's narrowed domain cells lifted** (`tests/core/coupling_domains.py`: one claim run over a memoryless pair in float64, mixed, bfloat16, float16, vmap, multi-rate, sub-cycled, predictor, restart, `run_adaptive` and sharded domains; the topology harness's invariances in eight): 188 cells in 24 rows, 350 still narrowed.
  It found MADD-ANO-170 to 173 (open, deferred to 0.5.0): the Aitken guard can cost two passes over its own exit (CPL-062 `failing`), `save_state` refuses a typed PRNG key, `reset_state()` inside a differentiated loss raises on a predictor group, and `run_adaptive` hangs on a NaN error norm.

## Test and CI changes

The cycle added most of the suite's checks that are not unit tests: property,
differential and metamorphic tests, the claims inventories, compliance gates
that fail, and a CI that runs in shards against a time budget.

### The changelog's "Verification" entries (moved)

- **A per-push witness must be a test that can fail** (`tests/compliance/test_slow_only_rule.py`): the slow-only rule read only a witness's `slow` and `skip` marks, so an `xfail` test, a `skipif(True)` test and one whose body skips were accepted. Two passing slow tests of the geometry search named a strict xfail pin, and a fault that made `spectral_error_bound` 1000 times too small stayed green on every push. The rule now takes each mark as pytest decides it and reads the body; an xfail pin is a witness only of a slow pin of the same open anomaly; the two tests name passing witnesses, one a new per-push cell (testing_standards, "What only the slow lane checks").
- **`tests/api/test_frame_renderer_3d.py` is recorded as never run by CI** (`scripts/check_anomalies.py`, `docs/validation/framework_verification.md`): it skips itself where `pyvista` is not installed and no job installs it (the `viz3d` extra), so every lane collected its 14 tests and ran none, and the record named `tests/viz` alone. The anomaly gate refuses it as evidence; the derivation that holds the record now reads each test file's import-time `importorskip` against the extras of the jobs that collect it, so the next such file fails the gate's tests until it is installed or recorded.
- **Two documents say what is true** (`docs/user_guide/inspection.md`, `pyproject.toml`): on a coupling group with a transform or a mapping on an internal edge, every `coupling_report()` call traces that edge's rule once for its shape and dtype, so a transform with a Python side effect shows it; the step is not compiled or traced again and the graph is untouched. The guide said nothing was traced; the read-only guard now has such a group and holds the trace as it is. And the test configuration no longer ignores a `UserWarning` "cycle detected" that the library has not emitted since 0.2.1.
- **The order "the mapping first, then the transform" is tested with transforms that do not commute with a mapping**: the test that pinned it scaled by two and passed with the two applications swapped, as did the property structures (negations and scalings only). It now uses an offset and a clip, `tests/core/test_an_edge_applies_its_mapping_before_its_transform.py` holds the order on every route that resolves an edge (the query, the step, the adaptive step, an edge to itself, additive edges, the edges at a coupling group including a sub-cycled member, a legacy closure), and the property structure `mapped-then-offset` holds it against the monolithic reference.
- **The capacity test reads a rung's record before the way its process ended** (`benchmarks/multigpu/run_capacity.py`, new in this release): a rung that recorded a failed check and was then killed, timed out or crashed is `check failed` and the ramp exits 1 (it was classified by the ending, and the ramp could exit 0 with a ceiling while `--summarise` exited 1); a rung whose complete record passed and whose process was then ended from outside is `passed`, with a note saying how. Every record a rung or a soak can leave is tested with ten endings through the ramp and `--summarise` both.
- **Tests for verdicts taken over a span of steps**: a `mask_unconverged` window whose early steps hit the cap and whose last step converged, the profiler's fractions on such a window, and the batched `strict_convergence` firing gate; the slow normal-contraction property asserts the bound where `spectral_usable` and holds the flag's share of its draws to a floor (it asserted the flag on every draw, which no claim promises). `docs/user_guide/inspection.md` states what `diagnostics=True` costs (a GPU measurement in `benchmarks/results/gpu_eigvals_probe/`; MADD-ANO-232 records a not-usable value that differs in kind between backends).
- **The step-program gate lowers each graph from emptied jax caches** (`scripts/capture_step_programs.py`, `tests/core/test_step_program_digests.py`): eleven gate graphs failed in a slow-lane process with no program changed. The lowered text also says how many identical copies of its own helpers (`isinf`, `frexp`, `_where`) jax wrote out, which depends on what its bounded trace caches still hold. `graph_digests` now calls `jax.clear_caches()` before it builds a graph; the capture of 396eb59a is unchanged and still holds for all 24 graphs on jax 0.10.2, 0.11.0 and 0.11.2.
- **REST write-sequence oracle: a run that fails part-way is held to a replay of the steps it counted** (`tests/property/injected_failures.py`): the rule "an empty graph's clock stays" was wrong for a graph emptied over the API, whose steps the streams go on counting, and failed the slow hundred-request machine; the server was consistent. A per-push example now runs an emptied and a never-filled graph with a later slice failing
- **MADD-VER-002's acceptance band no longer admits the defect it measured**:
  `[0.7, 2.5]` -> `[1.7, 2.3]` around the theoretical 2.0 (measured 1.900); it
  cited MADD-ANO-002 for a boundary defect. MADD-VER-001: 5% -> 1e-4
- **Four more nodes measured rather than skipped** (MADD-VER-009..012): `spring`,
  `ball`, `rigid_body_2d` and `heart_pump` declare a temporal order and meet it
  (1.029/1.002/1.000/1.000 against 1.0); 12 mutations confirm the ladders can fail
- **`LBMPipeNode` has a convergence verdict for the first time**
  (MADD-VER-013): the ladder converges, but a fourth level shows it is *not*
  in the asymptotic range, so no error band is quoted for it
- **Order of accuracy is measured, not asserted** (`maddening.testing.mms`):
  declare `NodeMeta(discretization_order=...)` and the harness refines a
  manufactured solution and fails the node on a shortfall (MADD-VER-005..008)
- **The committed stability report is compared with a fresh generation** in
  CI: it had rotted to 42 of 85 surfaces, hiding every deprecation.  Four
  surfaces that warn deprecated now carry the `DEPRECATED` tag
- **The coupled adjoint-identity property stops scaling by a cancelling
  inner product**: it divides by the norm of the terms contracted, not by
  the value they produce, which removes a latent float32 flake
- **Mapping-spec resolver, against generated input** (10 properties): a
  mutated spec is refused naming the edge or loads exactly the recipe on
  disk, and no generated asset path escapes the config directory
- **C-level tests for the FMU wrapper** (`tests/fmi/test_c_unit.py`,
  `tests/fmi/c/`): unit binary plain and under ASan/UBSan, a self-checking
  deterministic fuzz harness, valgrind, a libFuzzer campaign, FMPy against
  normal and hostile bridges, `validate_fmu`, and a `-std=c11 -pedantic`
  build.  CI installs valgrind and clang; each part self-skips if its tool is
  missing
- Full MADDENING test suite on 2026-10-02: **8634 tests collected** locally, 8169
  in CI's default lane (`-m "not slow"`, `--ignore=tests/viz`; 432 slow deselected).
  These counts are regenerated at the tag, with CI's pass/skip figures from the tag's
  run — see `docs/release_notes/v0.4.0.md`.  v0.2.1's own
  Verification block, which an edit during this cycle had moved here, is back
  under [0.2.1] as released
- Differentiable sharded solves and the C1 multi-physics IQN-IMVJ case match
  their dense and `fori` references in both differentiation modes

### The changelog's test-infrastructure entries from "Added" (moved)

- **Coupling interaction and topology harness** (`tests/property/test_differential_coupling_{interactions,topologies}.py`): a strength-3 covering array of the group's knobs, and drawn graph shapes -- several groups, outside nodes on a group's loop, flux, mapped and additive edges -- held to a monolithic float64 reference of the step.
  It found MADD-ANO-156 to 159 (open, since 0.1.0): a flux edge across a group's boundary or staggered in an ungrouped cycle fails to trace, a typed PRNG key fails under `ift` with Aitken or fixed relaxation, and a group joined off any cycle reads a node a step late, silently; workarounds in the release notes.
- **Domain oracles** (`tests/property/test_differential_{precision,replay,param_acceptance,fmu}.py`, `test_metamorphic_fmu_stability_filter.py`): float32 against float64 (a worker under x64),
  replay from recorded state, every door that writes or restores a parameter (checkpoint loads, `POST /graph/nodes`, wrapper nodes), the FMU four ways across `node.params` writes, and the
  stability filter; known disagreements are strict xfails naming their finding (`testing_standards.md`, "Domains the claims cover").
- **The slow-only rule is checked** (`tests/compliance/test_slow_only_rule.py`): a slow-marked framework test names a cheaper
  test the default lane runs (`# Per push: <node id>`) or is in the slow-only table; six framework properties that had neither
  now have one, and each slow hypothesis property a sibling on both JAX lanes.  A slow mark you add needs one (testing_standards)
- **Differential tests for coupling and numerics** (`tests/property/test_differential_*.py`): two paths that must agree, over
  generated graphs of synthetic nodes -- fori/ift, diagnostics on/off, the exact fixed point, multi-rate, sub-cycling, adaptive,
  `vmap`/`jit`, non-float leaves; each disagreement found is a strict xfail (`testing_standards.md`, "Differential tests")
- **Every Python code block in the docs runs in CI** (`tests/compliance/test_docs_snippets.py`), sandboxed, or carries a
  `<!-- snippet: no-run, reason: ... -->` marker whose imports and keywords are still checked; see the documentation
  standards.  The README quick start, which overrode the removed `requires_halo`, now runs
- **CI guard mutation test** (`tests/compliance/test_guard_mutations.py`, slow lane): seeded faults in the workflows, budget
  script, shard split, conftest, `pyproject.toml`, timing plugin, cache pruner and allowlist must each fail a guard test
  (83 of 87; 4 listed equivalents); anchors checked per push.  New guard: add its mutant (*Guard mutations*, testing_standards)
- **Docstring examples are executed in CI** (`scripts/check_doctests.py`):
  every `>>>` in `src/maddening` now runs, and the gate fails if the
  collection shrinks — an example that stops working is a failing build
- **Draw-rejection audit** (`scripts/audit_property_rejection.py`): measures what
  fraction of each property test's Hypothesis draws `assume`/`.filter` throws
  away, and fails CI over the gate — run it before narrowing a strategy
- **A performance regression gate on compile counts, not the clock**:
  `profile_graph` reports retraces, jaxpr primitives and lowered HLO ops;
  `python scripts/compile_counts.py --check` gates them and CI runs it
- **`maddening.sysid` contract properties** (`tests/property/test_sysid_contract.py`)
  over generated graphs; `windowed_loss` now rejects `sample_every <= 0`, a window
  wider than the data and observation leaves of unequal length instead of returning a meaningless loss
- **Property tests for the sharded surface** (`tests/cloud/multigpu/`):
  wrapper-contract, sharded-equals-unsharded, halo-exchange and round-trip
  invariants over generated meshes; two audit findings pinned as strict xfails
- Property-test coverage for round trips, the REST and FMU-bridge surfaces
  (stateful machines), the params pytree, `sysid`, retracing and binary frames

## Verification evidence

* **Suite size, 2026-10-02: 8634 tests collected**, of which 8169 are in
  CI's default lane (`-m "not slow"` with `--ignore=tests/viz`, as every CI
  lane runs it; 432 slow tests deselected).  This is a *collection* count
  from `pytest tests/ --collect-only`, taken on the release branch with the
  round-3 record fixes applied, in a local checkout (pytest 9.0.3, jaxlib
  0.11.0, CPU) with more optional extras installed than CI, which therefore
  collects fewer.  **These figures are regenerated at the tag**, with CI's
  passed, skipped and xfailed counts for each jax lane read from the tag's
  run; every merge before then moves them.  The figures this line carried
  until the round-3 audit were a week older: 6386 collected at `1ad3fa2`
  (2026-09-24), where CI (run 36072885507) passed 5986 on jax 0.10.2 and
  5985 on jax 0.11.2.  The number before that — "1680 passed, 3 skipped" — was written for v0.2.1
  (`6a597f2`, 2026-05-30) and copied verbatim into these notes by
  `c93bc89`; it described a suite two releases and ~9 000 lines of tests
  ago.  **Count-bearing lines should be generated at release time.**  The
  CHANGELOG convention that a branch never edits an existing line — which
  is what keeps parallel merges conflict-free — guarantees that any line
  carrying a count goes stale the moment the next branch lands, and
  nothing currently regenerates one.
* **C-level tests for the FMU wrapper** (`tests/fmi/test_c_unit.py`,
  `tests/fmi/c/`): a unit-test binary that includes the wrapper source —
  framing, JSON number parsing, endpoint discovery, and every FMI entry
  point against a fake sidecar on a socketpair and a loopback listener —
  built plain and with `-fsanitize=address,undefined`; a deterministic fuzz
  harness for the reply surface (3 seeds × 3000 iterations in the fast lane,
  60k in the slow lane, also exported as a libFuzzer target); the unit and
  fuzz binaries under valgrind memcheck; a short coverage-guided libFuzzer
  campaign with clang; FMPy driving an ASan build in a subprocess against a
  normal and a hostile bridge; FMPy's low-level FMI 3 API with two instances
  (params before initialisation, FMU state get/set/serialize, reset,
  terminate); `validate_fmu` on the packaged FMU; and a warning-free
  `-std=c11 -pedantic` build.  CI installs valgrind and clang so all of it
  runs there; each part self-skips where its tool is missing.
* **C1 acceptance**: a multi-physics IQN-IMVJ test (heat rod ⊗ spring,
  different operators per sub-domain) through the IFT while_loop with
  cross-timestep warm start, fori parity and a finite-difference gradient
  check.
* **Differentiable sharded solve parity**: gradient parity through the
  preconditioned solve against the dense reference (reverse and forward
  mode, RHS and operator coefficients) and on a 4-device CPU-virtual mesh.

## Engineering notes

This section is for contributors; nothing here changes the installed
package.

### Property testing

`tests/property/` is new.  `tests/property/strategies.py` draws *valid*
two-to-four-node graphs — real node classes, commensurate timesteps,
awkward-but-legal node names, edges with and without registered transforms,
additive edges, declared units, live ("calibrated") parameter overrides,
`ParamSpec` overrides including trainable interface-mapping weights,
external inputs, and interface mappings built by the RBF / nearest-neighbour
/ 1-D projection factories from either a node-field point reference or an
inlined point set.  `test_round_trips.py` asserts that a config round trip
preserves trajectory, parameter pytree (structure, dtype and bits) and
`param_specs()` including edge-key specs; that `to_dict` is idempotent
through `from_dict`; that a USD round trip owes the same; that a checkpoint
restores state and params exactly, continues a rollout identically and beats
the config for trained mapping weights; and that the two formats reload to
the same graph.  `test_adaptive_invariants.py` covers the `AdaptiveNode`
contract.  These are what found the USD naming and units defects above.

Two `RuleBasedStateMachine`s drive the externally driven surfaces where
every earlier test drove a hand-written sequence.  The REST machine works
over a FastAPI `TestClient` — add and remove nodes and edges, compile, step,
run, read and write state and parameters, save and load checkpoints, reset,
validate — interleaved with unknown nodes, wrong field sets, bad dtypes,
out-of-bounds and NaN values, and checkpoint paths outside the configured
root.  The bridge machine holds one real socket and mixes JSON and binary
frames on it.  Each checks, after *every* operation, against an in-process
model: a valid call must succeed and agree, a refused call must be a 4xx or
error reply that changes nothing and keeps the connection, and the REST
server must never answer 5xx.  The REST machine ends every example by
rebuilding a `GraphManager` from the server's own `GET /graph` and replaying
its trajectory.  Three of the *Fixed* entries in this release came directly
out of these machines.

Other property suites: `test_hypothesis_params.py` (baked ≡ traced step,
`step` ≡ `run_scan` ≡ `run_sweep` within ulps, float32 params gradient
against float64 finite differences, jvp/vjp adjoint identity through an
IFT-coupled group, no dirty/recompile/mutation on a modified pytree,
checkpoint round trip), `test_hypothesis_sysid.py`, a retrace property over
random graphs of built-in nodes, and the binary-frame
properties in `tests/fmi/` (bitwise float64 round trip including NaN payloads,
infinities, negative zero and subnormals; any byte string decodes
consistently or raises `ValueError`; any flagged frame after a binary hello
gets exactly one reply and the connection keeps serving).

Five oracles cover domains the claims inventories state but no test drew
from (`testing_standards.md`, "Domains the claims cover"): **float32
against float64**, the float64 side in a worker process under
`JAX_ENABLE_X64` (the fitters, `fim`, `windowed_loss`, the coupling report,
an IFT gradient, float32 leaves in an x64 graph; equal verdicts, answers
within tolerances derived from each problem); **replay from recorded
state** (`windowed_loss` exactly `0.0` at the generating parameters across
predictors, IQN warm starts, multi-rate records and sub-cycling, a fit
started at the truth, a checkpoint mid `run_adaptive`); **every door that
writes a parameter** (`check_params`, the sidecar, the bridge's `set` and
`set_state` with and without specs, `PUT /graph/params`) accepting the same
values -- and the restore doors (`load_state`, `POST /checkpoint/load`),
`POST /graph/nodes` and the wrapper nodes beside them; **the FMU four
ways**, the compiled C wrapper added, across
constant and structural `node.params` writes before an export; and **the
stability filter** as a metamorphic relation.  The disagreements they found
were strict xfails until their fixes landed -- the identifiability guard
and `fit_lm`'s floor under x64, `windowed_loss` with carried coupling
history, a structural write pending at an FMU export (now refused),
unfiltered FMU inputs and an open `log` bound without a `min` -- and these
still are: checkpoint loads restoring what a write
refuses, `POST /graph/nodes` applying no bounds, a lost write to a
`HybridNode` -- and four the audits had not reported: the REST route
storing a numeric string as the number; an open `log` / `logit` bound near
the float32 subnormal range advertised one float inside it, where
`ParamSpec.check` refuses; `POST /graph/nodes` taking a boolean for a float
constant; and the FMU doors and `check_params` taking a value the node's
constructor refuses.

### Hypothesis configuration and depth tiers

Hypothesis is configured once, in the root `tests/conftest.py`: two profiles,
`dev` (`max_examples=50`, the local default) and `ci` (`max_examples=200`,
what the `verify-hypothesis` job runs), both with `deadline=None`,
`print_blob=True` and `suppress_health_check=[HealthCheck.too_slow]`.
Select one with `MADDENING_HYPOTHESIS_PROFILE` (an unknown name is a hard
error); the active profile is printed in the pytest header.  This replaces
the single `jax` profile that applied to one directory, so property tests in
`tests/fmi/` and `tests/core/` get the same settings instead of hand-rolling
`deadline=None` per test.  Failing examples are now persisted to
`<repo>/.hypothesis/examples` and the CI job caches that directory, so a
falsifying example found once is replayed — which is how the rollout-parity
properties passed for months and then failed once.

The depth ratio was the real problem.  A per-test `@settings(max_examples=N)`
overrides the profile, and about a hundred call sites carried one, chosen ad
hoc between 12 and 1000 over the project's history.  **Measured on
`tests/verification/hypothesis/`: `dev` took 707 s and `ci` 721 s — 2 %
apart, for a profile that asks for four times the search.**  The root
conftest now resolves three named depth tiers from whatever profile is
loaded — `EXAMPLES_CHEAP` (4× the profile), `EXAMPLES_STANDARD` (the profile
itself) and `EXAMPLES_COSTLY` (two fifths of it, never below the house floor
of 20) — named for what one example *costs* rather than for a number, and 95
call sites now name a tier.  **The same suite now measures 462 s under `dev`
and 1365 s under `ci`**: the local default is a third faster, because the
tiers took depth away from tests whose examples cost seconds and gave it to
pure-function properties, and `ci` finally searches 3× deeper than `dev`
instead of 1.02×.  The eleven sites that keep an absolute cap are the ones
where the number encodes a real constraint — a search space Hypothesis
exhausts anyway, or an example measured in seconds — and each says so in a
comment.  No test's logic, strategies or assertions changed, and
`stateful_step_count` is deliberately not tiered: it sets how long one
example is, not how many there are.

The house rule is in {doc}`/developer_guide/testing_standards`: a property
test carries no `max_examples` and the profile owns the default; an explicit
value is an exception that needs a comment saying what makes a single
example expensive, and may not go below 20.

### Flaky tests fixed

Two property suites were flaky on a fixed absolute tolerance, and neither
was a library defect.  The `step` / `run_scan` / `run_sweep` agreement tests
compared a near-zero final value against `atol=1e-7`, but float32 round-off
accumulates with the magnitudes the arithmetic passes through, and a ball's
velocity crosses zero on every bounce: **a sweep of 1600 seed/step-count
combinations failed 7 times (all 15-step ball velocities, 2e-7 to 1e-6
absolute) — about a 6 % chance of a red run per CI job.**  The
interface-mapping adjoint test compared JAX's transpose of `apply` against
`apply_T` with a bare `rtol=1e-5` and failed on a 2e-7 absolute difference
where a component nearly cancels.  Both tolerances now scale with the
magnitudes involved, and the failing cases are pinned as explicit regression
tests.

### Six intermittent failures: one was the library's

Six tests failed now and then on CI and passed on a rerun.  Five were the
tests' own; the sixth was `MADD-ANO-191`.  Each was reproduced before it was
changed, on three cores carrying three busy loops beside the test unless
said otherwise (CPU, jaxlib 0.11.0), and each test was afterwards shown to
fail on faults seeded in a copy of the library (19 of 19).

- **Apps built at once** (`[REST-006]` of the concurrent domain; 11 of 50
  runs).  The check built a second server in each of its four copies, so four
  `create_app()` calls ran at once, FastAPI's `catch_warnings` blocks put
  back each other's filters, and a warning FastAPI silences was raised under
  the suite's error filter.  The `alias` that warning names is FastAPI's own,
  one per route parameter, not a `Field()` of the server's.
  `SimulationServer.create_app()` now builds one app at a time, and the check
  builds its second server once.
- **A refused save** (`[REST-101]` and `[REST-102]`, one check; 7 of 100
  runs).  The four copies saved into one directory, and a save writes its
  temporary files beside its target, so one copy's scan for leftovers saw
  another copy's save in flight.  The server leaves nothing behind: a save
  refused at its manifest is refused before anything is written.  The copies
  now save in a directory each.
- **A runner route behind a saturated worker pool** (`[stride]` and `[stop]`;
  three misses on CI, 0.70 to 0.76 s against 0.5 s, each on the jax 0.10.2
  lane).  The routes were never late: timed on the server, from the request's
  arrival to its answer, with every worker waiting, the stride takes under
  2 ms and the stop at most 55 ms.  The time was a full garbage collection of
  the test's own process, which stops every thread in it, the client and the
  server alike.  The test sent its 120 reads from 120 threads that each built
  an HTTP client first, an SSL context each, and timed the route from the
  client in the middle of that, which is where the collector's thresholds are
  crossed.  On jax 0.10.2, on one busy core, 2 of 60 stops and strides missed
  (0.54 s and 0.60 s), each with a collection of 0.46 s and of 0.42 s inside
  the timed request; the 11 requests a collection overlapped took 0.36 s at
  the median and the other 49 took 0.02 s.  Nothing in that was found to
  depend on the jax version: after the 716 tests CI's shard runs first, a
  full collection costs 0.19 to 0.27 s on 0.10.2 and on 0.11.2 alike, and
  on idle cores one overlapped the request in 6 of 80 tests on 0.10.2 and
  10 of 80 on 0.11.2.  The lane is taken to decide only when the
  collector's thresholds are crossed; that was not measured on CI.  The
  test now builds every client first, sharing one SSL context, asks the
  route once all 120 reads have arrived and none has been answered, reads
  the server's clock, and runs with the collector off (as `timeit` does),
  as does the `/healthz` test beside it.  Its lock timeout is 2 s, because
  on busy cores the reads take up to 0.54 s to arrive; the bounds are
  counted in timeouts and did not move.
- **`fim`'s sigma-scaling property** (19 of 54 790 admitted draws).  Dividing
  the residual by sigma rounds each float32 entry afresh, so the scaled
  matrix is `F / sigma**2` to about an ulp of its largest entry (9e-8 of it
  in the pinned case), and a condition number of 5.7e3 magnifies that into 0.11%
  between the two runs' weakest bound: past a fixed `rtol=1e-3`, with the
  float64 value between the two (float64 agrees with the scaling to 8e-13).
  `cond` and each bound are now held to four times `fim`'s own rounding
  estimate, `max(n, sqrt(m)) * eps * cond`: over 64 000 draws on jaxlib
  0.10.2, 0.11.0 and 0.11.2 the two runs differ by at most 1.6 times it.
  That is looser than 1e-3 only above a condition number of 500 to 1000, by
  the problem's size, and far tighter below: a bound off by 0.05% under a
  noise model passed the old test and fails this one.

After the changes the three checks of the concurrent domain passed 100 runs
each under the same load; the four runner routes and `/healthz` passed 50
runs each under it on jax 0.10.2, with no collection between the first read
and a runner route's answer in any, and 15 runs each beside a thread that
kept forcing full collections; and the property passed 30 runs of 200
examples across the three jaxlib versions.

### Other test infrastructure

`maddening.testing.strategies.node_states` samples bool and integer state
fields with their own dtype, so the `structure` check covers monitor-style
nodes (`HealthCheckNode`) instead of skipping.  `gm.trace_count` replaces
the jit cache size as the retrace probe, because the cache count reads 0 on
JAX 0.10, which broke CI.  `scripts/generate_stability_report.py` now
imports its tagged modules from a `STABILITY_MODULES` list that a compliance
test checks against a grep for `@stability(` over `src/maddening` — three
modules had dropped out of the generated report the release gate reads.  The
two Hypothesis integrator-order tests run under
`jax.experimental.enable_x64()` instead of skipping.  `fsspec` joins the
`ci` and `dev` extras so the cloud-URL checkpoint tests run in CI.

Every slow-marked test of a framework property now says, at its mark,
which cheaper test still checks the property on every push
(`# Per push: <node id>`), or is listed in the testing standards' table of
properties checked only in the slow lane.  `tests/compliance/test_slow_only_rule.py`
checks both from pytest's own collection, and that every named test runs in
the default lane.  Six framework properties had neither and now have a
per-push witness: `verify_node`'s `params_gradient_finite` failing,
`vmap(grad)` through an IFT-coupled step, an LM fit through an IFT-coupled
group, four properties of the gradient error bound, the coupling-sweep rows
the coupling guide quotes, and arbitrary REST call sequences.  The
slow-marked Hypothesis properties, which `verify-hypothesis` runs on jax
0.10.2 only, each have an unmarked sibling at reduced depth that both JAX
lanes run, among them the float32 rank-threshold calibration, whose counts
move with jaxlib.

### The compliance gates now fail

`scripts/check_anomalies.py`, `check_impl_mapping.py`, `check_citations.py`
and `check_transforms.py` are the IEC 62304 / MDCG evidence that this
project's documentation matches its code, and CI has run all four on every
push since v0.3.0.  A mutation audit planted 28 defects across them and
**22 passed**.  `check_transforms.py` was the worst: it scanned three
subpackages for string-literal `transform=` kwargs, all of which use
lambdas, and explicitly excluded `tests/`, where every string transform in
the repository actually lives — so it reported `OK: 0 string transform
reference(s) verified` and could not fail on anything.  `check_impl_mapping.py`
resolved names with `getattr`, which walks the MRO, so renaming
`HeatNode.update` left it green; it also split Markdown rows on a bare `|`,
which mangled any cell containing escaped LaTeX pipes and hid one
`AdaptiveNode` row from the gate entirely.  `check_citations.py` accepted a
`%`-commented bibliography entry — the exact case it exists to catch.

All four now fail on those defects, and a planted-defect test accompanies
each one so the coverage cannot quietly lapse again.  **A tree that was
green before may now report real documentation rot that was always there.**
That is the gates starting to work, not a regression: re-run them, and read
what they say rather than raising a minimum to clear it.

These changes reach downstream libraries that adopted MADDENING's compliance
tooling:

* **`resolution_status` is now required and enum-checked** by
  `maddening.compliance.validate_anomaly_registry()`, along with a rejection
  of unknown keys (a typo such as `workaroud:` used to drop the workaround
  silently), and resolution of every `affected_components` symbol and every
  `verification:` test path.  A downstream `known_anomalies.yaml` that
  omitted `resolution_status`, or that names a test that has since been
  renamed, will now fail where it passed.  Pass `resolve_references=False`
  for a schema-only check if the symbols are not importable in that context.
* **A symbol in an optional subpackage is reported as *unchecked*, not
  broken.**  `maddening.usd.serialization.save_graph_to_usd` is a legitimate
  `affected_components` entry, but a CI job installed with `[ci]` alone
  cannot import it — and the resolver used to discard that fact when a
  shorter prefix imported, failing the run with `'maddening' has no
  attribute 'usd'`.  It now says which extra to install and does not fail:
  asserting a reference is broken when the truth is "I could not look" is a
  false statement in a compliance artefact, and the alternative would make
  every anomaly touching `maddening.usd` or `maddening.viz` unrecordable.
  The skip is counted in the gate's own output line, and the gate still
  fails if *nothing* could be checked.  `validate_anomaly_registry()` takes
  an optional `notes=[]` list that collects these; the returned error list
  keeps its meaning.
* **`ResolutionStatus` gains `PARTIALLY_RESOLVED`.**  MADDENING's own
  registry has used `partially_resolved` for `MADD-ANO-005` since it was
  written, and the enum did not define it — so the registry this project
  ships could not be loaded into `AnomalyRecord`.  The addition is
  additive; a test now pins the enum and the validator's accepted set to be
  exactly equal in both directions.
* **A `verification:` node id is resolved the way pytest reads it.**
  `path::Class::method` used to pass whenever both names were defined
  *anywhere* in the file, so an entry putting a module-level test under a
  class it is not in — a node id pytest reports as "not found" — was
  accepted.  Each component is now looked up where pytest looks for it:
  module level first, then the body of the named class or a base class
  defined in the same file.  A downstream registry with a loose node id
  will now fail; add or correct the class component.

One rule applies to MADDENING's own registry only, in
`scripts/check_anomalies.py` rather than in `validate_anomaly_registry()`:
**every `affected_versions` is a PEP 440 range checked against the
registry's `maddening_version`.**  A reachable entry's range must admit
that version and a `resolved` entry's must not.  Until the release-record
audit the only rule was a `startswith(">=")` test in the SOUP generator, and
`MADD-ANO-016` shipped `partially_resolved` with `>=0.1.0, <0.4.0` — a
range saying 0.4.0 is unaffected beside a residual risk saying the 0.4.0
behaviour has never been observed.  The convention, including `none` for a
defect introduced and fixed within one development cycle (which
`MADD-ANO-018` and `-019` now carry), is in the registry's header comment,
and the SOUP generator runs the same function.

**`LBMNode`'s algorithm ID is now `MADD-NODE-011` (was `MADD-NODE-007`).** From 0.1.0 to 0.3.1 `LBMNode` and `RigidBodyNode` both carried `MADD-NODE-007`; `RigidBodyNode` keeps it, because a test shipped in every release pins it there, while `LBMNode`'s was first published in this release's algorithm guide. Code or records keyed on `LBMNode.meta.algorithm_id` must move to the new ID; `check_impl_mapping.py` now refuses a duplicate ID and a guide whose ID differs from its node's `NodeMeta`.

## The multi-GPU session runner

`benchmarks/multigpu/` holds the scripts that run a fixed checklist on a rented
multi-GPU machine and summarise what it measured. They are development tooling:
they live in the repository, not in the installed package, and carry no
stability label. Their guide is `benchmarks/multigpu/README.md`.

<!-- after the hardware session: name the session's date, the machine, the jaxlib it recorded and where its summary is filed; the measured results themselves go to the release notes' multi-GPU item and to docs/developer_guide/sharding_topology.md. -->

### Session tooling for the hardware run

`benchmarks/multigpu/run_pod.py` is the pod-side runner for the
human-supervised four-GPU session — nothing in the repository launches a pod.
`--goal exchange` times `all_to_all` against `ppermute` at 10⁵–10⁶ cells
(warmup plus repeats, min and median, bytes from `exchange_traffic()`, bit
identity); `--goal forward` runs a `ShardedUnstructuredNode` at 10⁶ cells on
a real mesh (`--mesh edges.npz`) or a synthetic one against the unsharded
node; `--goal gradient` reports sharded-versus-unsharded gradient parity
through a rollout and the preconditioned `sharded_cg`.  Each goal writes one
JSON under `--out`; `--summarise DIR` prints the ranking table and the
recommendation; `--dry-run` proves the whole script on CPU virtual devices.
`benchmarks/multigpu/README.md` is the session runbook.

The audit of this tooling mattered because its output would otherwise have
decided a default.  **The `exchange` goal timed a slab that lived on device
0, so every timed call also scattered the whole slab to the mesh and both
transports were charged the same constant — about two thirds of the measured
time on 4 virtual CPUs.**  All timed inputs for all three goals are now
placed once with the mesh `NamedSharding` outside the timed region, the
runner refuses to time anything else, and it records `input_presharded`.
`recommend()` accepted single-GPU rows (a no-op exchange with a
timer-noise ratio) and raised `TypeError` on a zero median; a row now
decides only from a real accelerator run at ≥ 10⁵ cells on ≥ 4 devices (≥ 2
with `--allow-fewer-devices`, recorded in the JSON) with a finite speedup,
and the reason line names what each row lacked.  Rollout gradient timings
compared a jitted unsharded grad against a retracing, host-re-partitioning
sharded one; both are now `jax.jit(jax.grad(...))` with statics placed once
and compile time reported separately (`compile_s`; JSON `schema_version` 2).

The `tests/cloud/multigpu` conftest now forces virtual host devices only
when no accelerator is about to be used, so on a GPU pod with
`JAX_PLATFORMS=cuda` the multi-device tests run on the GPUs instead of 16
virtual CPUs.  `MADDENING_VIRTUAL_DEVICES=N` overrides either way, and the
conftest is the only setter of `XLA_FLAGS` (enforced by a grep test).

The runner then grew to cover the whole sharding checklist, so that the
session is comparison against references that already exist rather than
judgement at a metered pod.  Five goals run first — `indivisible` (the
stencil and pointwise refusals, and an uneven unstructured split),
`halo` (every boundary mode and halo width on a 1-D and a 2×2 mesh, and
the unstructured exchange, forward and adjoint, bit for bit against
NumPy), `coupled`, `stencil` (a 2-D field with a sharded static, forward
and adjoint) and `hybrid` (`HybridNode(ShardedStencilNode(inner))` in a
graph) — and every goal, the three timing goals included, now records its
comparisons as pass/fail `checks` with the limit each was held to (JSON
`schema_version` 4, which also records the sense of each comparison).
The runner stops at the first goal whose checks fail and exits non-zero,
and `--summarise` reports each checklist item as closed only by a
real-GPU run on at least four devices with every check run —
`--allow-fewer-devices` is for the transport ranking and does not lower
that, because on two devices a halo taken from the wrong neighbour passes
every halo and stencil check.  A case the device count cannot express
(the 2-D pencil cases below four devices) is recorded as a check *not
run*, which keeps its items open, and `--summarise` re-derives every
verdict from the recorded value, limit and sense instead of trusting the
recorded flag.  `stencil` runs `"edge"` ends (the wrapper's default) and
Dirichlet ends held through a boundary input, besides periodic ones, so a
wrapper that wrapped the halos of an axis it does not shard whatever
`boundary` said can no longer pass the whole checklist.  **`coupled` is the one
item nothing covered on any backend: a sharded field member and a
replicated member in one coupling group, forward and adjoint**, under the
default solver and `"fori"`, against the same group with the node
unwrapped and against a float64 model of the implicitly coupled step.  Its
CPU-virtual test,
`tests/cloud/multigpu/test_coupling_group_with_sharded_and_replicated_members.py`,
measured the sharded group within 1 ulp of the unsharded one forward and
within 5 ulp in the gradient, identically on jaxlib 0.10.2, 0.11.0 and
0.11.2.  Under the default solver, above 50 coupled degrees of freedom, the
adjoint's failure check is a host callback pinned to device 0, and XLA
gathers the coupled state vector onto that device and back ("Involuntary
full rematerialization"); the goal counts it (`device0_pinned_ops`) and
the session measures its cost.

The confirmation audit then seeded faults into the *results* rather than
the runner, and **`--summarise` still read `CLOSED` on all six items for a
dry run relabelled as a 4-GPU run with any one of these**: a check's limit
loosened to 1.0 in the file, results tables that disagreed with the checks,
`stencil.json` cut to one check, stencil and hybrid files from schema 3 and
another commit without their edge and Dirichlet cases, `forward.json` from
another commit, `n_devices` 4 on an environment that saw one device, the
"is partitioned" checks deleted, or `schema_version` 99.  The verdict
re-derived each check's pass/fail but took the file's word for everything
else.  Every goal's checks are now a pure function of its recorded results
(`GOAL_CHECKS`), which the runner and `--summarise` share, and a file
decides nothing unless it is on the current schema, its `n_devices` is no
more than the devices its environment saw, its results hold exactly the
cases the runner runs for the file's own config, and its checks are exactly
those the runner derives from its results -- names, values, limits (so
`LIMITS`), senses and flags.  Such a file reads `INVALID`, the items it
decides stay open with the file and the reason in their status, and the
summary exits 3.  The files deciding one item must also record one git
commit.  The transport ranking counts an `exchange.json` only when it reads
`PASS`.  Each seed now reopens exactly the items it touches and names why;
the relabelled clean run still closes all six
(`tests/cloud/multigpu/test_run_pod_verdict_integrity.py`, on a committed
runner record).

A third audit seeded faults into the *library* instead, and **the same
relabelled summary closed items 1, 3, 4 and 6 with `ShardedStencilNode`
broken in each of four ways its unit tests catch**: a sharded static's
halos set to NaN, grid-shaped boundary inputs zeroed, domain integrals
halved, and every sharded axis after the first zero-filled at the global
edges.  The goals' node read its static only in the interior, took scalar
inputs only, declared no domain integral, and ran on a 1-D mesh, so none
of those paths reached an answer.  It now reads its mask one cell into the
halo (face-averaged conductances), takes a grid-shaped source and ambient,
and carries a domain integral in its state, named to sort before the grid
field; `stencil` runs it under all three ends and adds `LBMNode` on D2Q9
(whose streaming reads the halo corners), each on the 1-D and the 2×2
pencil mesh; `hybrid` and `coupled` run on the pencil mesh, the coupled
group coupled through the integral and a grid-shaped input (JSON
`schema_version` 5).  `tests/cloud/multigpu/test_run_pod_seeded_faults.py`
keeps the four faults as seeds of `sharded_node.py`, applies each to a
scratch copy of the library, and requires the three wrapper goals to fail
and the items they decide to read `FAILED`; the unseeded copy must pass.
The integral's name sorts first on purpose: the wrapper used to read the
block extent in `shard_info` off the first state field in sorted order, the
integral included (fixed in this release, see "Sharded wrappers"), and the
Dirichlet cases, which find the global edges from `shard_info`, fail on a
wrapper that does.  The dry run (`--cells 256 1024`) takes 92 s on
three cores, up from 55 s.  The runbook adds the procedure the files do
not yet enforce: a clean checkout at one commit, `maddening` imported from
it, every goal of an item on one pod, and `device_kinds` checked by eye.

A fourth audit found that **four devices -- the session's minimum -- did
not see a fault confined to the second sharded axis**.  The reason given
for the minimum was that on two devices a shard's left and right neighbour
are the same device; on four, the pencil mesh is 2 × 2, so *both* of its
axes have that property, and every grid was square.  Three seeds passed
every goal and, relabelled as a GPU run, closed all six items: each halo
along spatial axis 1 taken from the wrong neighbour (in `halo_exchange`),
`shard_info`'s block extent divided by the first mesh axis's size, and its
global extent read off spatial axis 0 for every axis (which passed on
eight devices too).  The `hybrid` and `coupled` goals ran on the pencil
only, the source had period 1/2 in x -- one block of the pencil -- and the
coupled group's grid-shaped input was a broadcast scalar.  The halo,
stencil, hybrid and coupled goals now run four meshes at their smallest
size: a 1-D mesh along each spatial axis (each axis split over all four
devices), a 1 × 4 two-axis mesh (mesh axes of different sizes), and the
pencil; grids are `ny × (ny + D)`; every grid-shaped input, the coupled
field's `ambient = u × profile` included, differs on every block of every
mesh; and a spatial axis no case splits over three or more devices is a
check not run.  The unsharded side of each case is run once and shared by
every mesh.  The three faults are seeds in
`test_run_pod_seeded_faults.py` (the halo seed fails `halo` too), and
`indivisible` adds the refusal along spatial axis 1 of the 1-D mesh (JSON
`schema_version` 6).  The dry run takes 77 s at `--cells 256` and 96 s at
`--cells 256 1024` on three shared cores, up from 44 s and 65 s.  The same audit found that `--summarise`
closed items on different commits without naming them -- the one-commit
rule is per item -- so each checklist line now names its commit, and a
directory whose files come from more than one commit prints `MIXED
COMMITS` and exits 4; and four of the runner's checks (a zero parameter
gradient, a negligible hybrid correction, a coupled step that did not
iterate, a divisible grid refused) had no test that a degenerate result
fails them, and now do.

### The changelog's entries for the runner (moved)

- Multi-GPU session tooling: `benchmarks/multigpu/run_pod.py` (`--goal
  exchange|forward|gradient`, `--summarise`, `--dry-run`) and
  `benchmarks/multigpu/README.md`.  Nothing here launches a pod
- Multi-GPU session runner covers the sharding checklist: `run_pod.py --goal checklist` (`indivisible`, `halo`, `coupled`,
  `stencil`, `hybrid`), each against its unsharded, NumPy or float64 reference; every goal records pass/fail `checks`
  (schema 4), stops the session on a failure; `--summarise` re-derives every verdict and closes an item only on >= 4 real GPUs
- **A capacity test for the multi-GPU session** (`benchmarks/multigpu/run_capacity.py`, not a gate): a sharded D3Q19 lattice larger than one card, compared with a small tile's own unsharded run, tiled, and ramped towards a target fill of every card with one process per rung; the runbook's new section "After the checklist: the stress tail" gives the commands, and `--dry-run --cells ...` proves it on a laptop.  It found MADD-ANO-262 (open): `ShardedStencilNode` builds the wrapped node's whole initial state on the default device.

## Index of the anomaly registry

Every entry of the registry is named below, reachable in 0.4.0 or not.
`tests/compliance/test_release_notes_known_anomalies.py` checks that: an entry
the paragraphs describe is named by its full number, and the closing
"**Resolved.**" paragraph names no entry a user can still reach. The release
notes' "Known anomalies" section lists only the entries a user can reach. The
text below is moved; the registry is the authority on each entry's status.


`MADD-ANO-160` — *`jax.grad` through `run_adaptive_scan` was 0.0 or NaN where
an attempt's step-doubling estimates agreed exactly, as for an accelerated
coupling group whose iterate starts at its fixed point* — shipped in every
tagged release and is **resolved in 0.4.0**.  Under `"aitken"`, `"iqn-ils"`
or `"iqn-imvj"`, two scalar relays `x <- 0.6 u + b` with a clock leaf and a
constant `b` gave a derivative of `x_a` in `b_a` of 0.0 (NaN on a later
tree, and at the tags) through `run_adaptive_scan`, where `run_scan` gives
the exact 1.5625.  The step-size controller's error norm, a square root, had
an infinite derivative wherever the two estimates agreed; it is now taken
through a double `where`.  Before 0.4.0, differentiate through `run_scan`.

`MADD-ANO-161` — *`jax.grad` through a 16-bit coupling group under
`solver="ift"` raises `NotImplementedError`* — shipped in 0.3.0 and 0.3.1 and
is **resolved in 0.4.0**: the IFT rule's linear solve runs in float32 on the
group's own operator and is cast back, to the 16-bit output's resolution.
Before 0.4.0, hold the group in float32 to differentiate it.

`MADD-ANO-162` — *`strict_convergence` on a coupling group with a sharded
member aborts the process* — was carried by no release (`strict_convergence`
is new in 0.4.0) and is **resolved**: on a step spanning several devices
the check raises on every device at once, so none is left waiting at an
all-reduce, and the documented error reaches Python.

`MADD-ANO-170` — *The Aitken two-pass guard can add two passes to Aitken's
own exit, where the guide says at most one* — is **open** and was carried by
no release.  The guard holds the pass to its error estimate and the
predecessor to its raw residual, and an Aitken estimate can rise back above
the threshold on the next pass with the raw residuals still falling: on a
three-mode pair Aitken without the guard stops at pass 9 and with it at pass
11.  The state returned is converged; what fails is the documented cost
(CPL-062 is `failing`).  Budget Aitken's iterations from the reported
`iterations`; deferred to 0.5.0.

`MADD-ANO-171` — *`save_state` cannot write a typed PRNG key leaf* — shipped
in every tagged release and is **open**: `np.asarray` of a key array raises
`TypeError`, so a graph holding a `jax.random.key(...)` leaf cannot be
checkpointed.  Hold the key's data (`jax.random.key_data`) or a legacy
`PRNGKey` instead; deferred to 0.5.0.  The convention for a node that draws
random numbers, and what holds through each door, is in the node authoring
guide ("A node that draws random numbers").

`MADD-ANO-172` — *`reset_state()` inside a differentiated loss raises
`UnexpectedTracerError` on a predictor group after an earlier transform* — is
**open** and was carried by no release (`reset_state` is new in 0.4.0).  A
loss that resets the graph and runs `run_scan` differentiates once and
raises on the second `jax.grad` when a group has a predictor.  Reset
outside the differentiated function, or call any entry point between two
transforms; deferred to 0.5.0.

`MADD-ANO-173` — *`run_adaptive` never returns once its step-doubling error
norm is NaN* — shipped in every tagged release and is **open**.  When a state
overflows, both attempts hold `inf`, the norm reads `inf / inf`, the next step
`step_decision` chooses is NaN, and no attempt is ever accepted again: the
call spins with no warning (a coupling group with an eigenvalue above one does
it once its iterate overflows).  Use `run_adaptive_scan`, whose `max_steps`
bounds it, or watch the state in a `callback`; deferred to 0.5.0.

`MADD-ANO-186` — *a bfloat16 or float16 coupling group's norms overflowed in
its own dtype* — shipped in every tagged release and is **resolved in
0.4.0**.  Above 65 504 active entries a float16 group's "mixed" or
"interface" residual read 0.0 (on this cycle's development builds the group
then stopped with `converged=True` 4% from its fixed point, where its
float32 twin runs to 0.9%); at the default `rtol` it read `inf` on a finite
state.  The norms, the floor and the residual are now float32.  On an older
release, keep a 16-bit group's coupled fields in float32.

`MADD-ANO-187` — *`spectral_error_bound` under the interface norm with an
edge transform read 0.0014-0.098x the true distance, usable* — was carried
by no release (the spectral bound is new in 0.4.0) and is **resolved**: the
bound is taken on the norm's transformed reading.

`MADD-ANO-188` — *`gradient_relative_error_bound` read 0.65-0.986x the true
gradient error, usable, far from the fixed point* — was carried by no
release (the gradient bound is new in 0.4.0) and is **resolved**: the
Kantorovich check takes `L` as an operator across the Newton step and the
bound carries the step's second-order miss.

`MADD-ANO-189` — *the adaptive steppers' error norm was computed in a 16-bit
state's own dtype* — shipped in every tagged release and is **resolved in
0.4.0**.  `run_adaptive` on a float16 state above 65 504 elements did not
return (its error norm read NaN), and `run_adaptive_scan` raised a
scan-carry `TypeError` on any bfloat16 or float16 state.  The norm is now
measured in float32.  On an older release, hold an adaptively stepped state
in float32.

`MADD-ANO-190` — *`run_adaptive_scan` raises a scan-carry `TypeError` on a
float32, bfloat16 or float16 state under `jax_enable_x64`* — shipped in
every tagged release and is **open**: the scan's clock carry is float64
there and its next timestep takes the error norm's dtype, so only an
all-float64 state is stepped.  Use `run_adaptive`, or hold the state in
float64; deferred to 0.5.0.

`MADD-ANO-191` — *Apps built at once in several threads left the process's
warnings filters changed: a silenced warning was raised, or every
`UserWarning` was ignored from then on* — shipped in every tagged release and
is **resolved in 0.4.0**.  FastAPI builds each route's fields inside
`warnings.catch_warnings()`, which is not thread-safe, so two
`SimulationServer.create_app()` calls at once put back each other's filters:
four servers built together raised a warning FastAPI silences in about a
quarter of 300 rounds and left `ignore::UserWarning` in the filters, for the
rest of the process, in 1 to 5 of them.  `create_app()` now builds one app at
a time.  An app built while another thread is inside a `catch_warnings` block
of its own can still trade filters with it.  Before 0.4.0, build the apps one
after another.

`MADD-ANO-192` — *Interface-mapping factories returned an operator for
coordinates their formula does not cover* — shipped in every tagged release
(the closure factories) and is **resolved in 0.4.0**.  Descending cell
boundaries gave `conservative_projection_1d` a matrix of zeros, a NaN point
was selected by nearest neighbour for every target, and unsorted, non-finite
or mismatched coordinates were mapped without a word; each is now a
`ValueError` naming the argument and index.  Before 0.4.0, check the
coordinates yourself.  See [Coordinates a factory cannot map are
refused](#coordinates-a-factory-cannot-map-are-refused).

`MADD-ANO-193` — *The conservation diagnostic and the surrogate dataset
generator rebuilt boundary inputs by a rule of their own, not the step's* —
shipped in every tagged release and is **resolved in 0.4.0**.
`check_conservation` and `DatasetGenerator` let an additive edge overwrite the
edges before it, dropped or mis-applied the rest (a flux-reading edge, a
transform across the time axis, 0.4.0's mappings), and a misspelt flux read
as `0.0`; they now resolve through the step's own edge rule.  Before 0.4.0,
do not use either on a node with additive incoming edges.

`MADD-ANO-194` — *`DatasetGenerator` pairs a forward edge's boundary input
with the source's state before the step* — shipped in every tagged release
and is **open**.  The dataset is rebuilt from the state history, so a sample's
input from a source that runs earlier in the schedule is the value of the
step before: `[11, 21]` where the step fed `[12, 22]`.  Inputs that do not
change, and back edges, are exact.  Shift `boundary_inputs` one sample later
yourself, or record the inputs in the node; deferred to 0.5.0.

`MADD-ANO-195` — *The interface convergence norm reads a mapped edge's source
field, not what the edge delivers* — was carried by no release (mapped edges
are new in 0.4.0) and is **resolved**.  `convergence_norm="interface"`, the
precision floor and the spectral bound's reading applied an edge's transform
but not its mapping, so a mapped edge inside a coupling group was measured on
its source field's scale: a 9% change of the delivered value read 0.0071
against a threshold of 1, and `spectral_error_bound` 0.064-0.32x the distance
in what the edge delivers, usable.  Every reading now goes through the step's
own edge rule, with the weights the step ran with.

`MADD-ANO-211` — *the float floor of an interface edge that delivers a wider
dtype than its source field was taken at the wider dtype's eps* — was
carried by no release (the floor is new in 0.4.0) and is **resolved**: under
`jax_enable_x64` a float32 pair stalled behind edges that cast to float64
read `spectral_error_bound` at 6.7e-7x the true distance, usable; each
delivered value is now resolved at the coarser of its own dtype's eps and
its source field's, and no entry of a group's floor below the eps of the
group's coarsest floating field.

`MADD-ANO-212` — *the floor's gain of a same-pass read is measured along the
source's own state, which a difference of two entries of one field cancels*
— was carried by no release (the floor and its gain measurement are new in
0.4.0) and is **open**: a Gauss-Seidel group reading `u[0] - u[1]` of a field
at 1000 through a mapping row `[1, -1]` stalls at `residual=0.0` and reports
`spectral_error_bound` at 0.054x the true distance with
`spectral_usable=True` (2.6e-4x at 1e5).  Run such a group under
`iteration_mode="jacobi"`, where the bound holds, or have the producer hold
the difference as a field of its own; deferred to 0.5.0's measured
`diagnostics="rounding"` level.

`MADD-ANO-213` — *under the interface norm a field read by several internal edges was counted once per edge by the residual and once by the spectral analysis* — was carried by no release and is resolved: on a star `spectral_error_bound` read 0.26 to 0.73 of the true distance with `spectral_usable=True`, and such a group's report is now taken on the reading (1.001 to 1.010).

`MADD-ANO-208` — *A node named with a NUL, another control character or a surrogate was taken, and its checkpoint, its FMU's model description or the server's own replies could not be read* — shipped in every tagged release and is **resolved in 0.4.0**: `add_node`, `add_edge`, `add_external_input` and the REST doors refuse such a name where it is introduced.

`MADD-ANO-209` — *A refusal that echoed a surrogate from the request answered 500* — shipped in every tagged release and is **resolved in 0.4.0**: every REST reply is written by one encoder that cannot fail on what it echoes.

`MADD-ANO-210` — *A `#` in an edge's field name gave two mapped edges on one field pair a single slot of weights* — carried by no release (mapped edges are new in 0.4.0) and is **resolved in 0.4.0**: `add_edge` refuses the name.

`MADD-ANO-020` — *`LBMNode`'s Zou-He pressure boundary did not impose the
pressure it was given* — shipped in every tagged release and is **resolved in
0.4.0**.  The closure for the face-normal velocity counted the populations
streamed out of the interior once where Zou and He count them twice, so the
face density came out as `rho_p + S_K`: a prescribed `p = 0.36` was imposed as
0.4156 after one step, on every face of both lattices, and reproduced
identically against extracted v0.1.0 and v0.3.1 trees.  A pressure-driven
channel carried 0.58 to 0.80 of the imposed pressure drop, its centreline
velocity was 0.48x what the imposed drop gives on the in-tree 64x16 case, and
`compute_boundary_fluxes()["outlet_pressure_avg"]` reported the wrong
pressure.  Nothing in the suite could see it: the only pressure-driven test
normalised both profiles, and every LBM benchmark was body-force driven.  The
closure is now derived per face (with the transverse-momentum correction of
Zou and He for D2Q9 and Hecht and Harting for D3Q19), documented in the new
`LBMNode` algorithm guide, and pinned by tests that read magnitudes: the face
density is the prescribed one to float32 rounding, and **MADD-VER-016** runs
a pressure-driven Poiseuille ladder whose first-order extrapolation is
Hagen-Poiseuille to 1e-4 (0.396 before the fix).  **Any pressure-driven
`LBMNode` result from an earlier release changes and should be re-run**;
body-force-driven results are unaffected.

`MADD-ANO-019` — *A coupling iteration that diverged to a non-finite state
was reported `residual=0.0, converged=True`* — was introduced and **resolved
within 0.4.0**.  The reference-scaled convergence norm that closed
MADD-ANO-005 drops a field whose magnitude sits inside the dead band, and a
`NaN` magnitude compared False against it: the field that had just diverged
contributed exactly zero, and a relaxed two-node relay that overflowed
float32 (`relaxation=1.5` on a `-0.95` cycle) reported `residual=0.0,
converged=True` on both solvers and all three norms, with
`strict_convergence=True` passing it through.  Under the L2 norm the same
verdict arrived one pass earlier on a *finite* state of `-1.2e38`, because
the scale's float32 reciprocal is subnormal there and the CPU backend's
broadcast divide flushes it to zero.  A field the norm cannot evaluate — a
non-finite reference, or a scale whose reciprocal is not a normal number of
the field's dtype — is now active and contributes `inf`, so the residual
is `inf`, `converged` is False, `ratio_usable` is False, and
`strict_convergence` raises with a message that names the non-finite state.
Finite fields are bit-identical through the guard; a field at zero still
leaves the dead band as before.  The one boundary this draws is that a
float32 field within a factor of four of overflow can never read as
converged.  No tagged release carried the defect: v0.3.1's unscaled norms
report `residual=nan` at the cap on the same reproducer.

`MADD-ANO-006` — *Non-finite numbers are written as non-standard JSON
tokens* — is **resolved in 0.4.0**.  All three surfaces now write the
quoted tokens; see "[Non-finite numbers are valid JSON
now](#non-finite-numbers-are-valid-json-now)" above.  The entry's claim
that a tagged form "would break the shipped C wrapper, which parses values
with `strtod`" was wrong about the mechanism: `strtod` parses the tokens
fine, and it was the quote character that stopped it.

`MADD-ANO-024` — *`PUT /graph/params` accepted a value a node consumes at
construction: reported and saved, never used by the running graph* —
shipped in every tagged release and is **resolved in 0.4.0**.  The route
wrote the node's params dict and answered 200, but a value the node had
consumed when it was built (a geometry mask, an assembled operator, an
`__init__` copy) was never read again: at v0.1.0, v0.2.1 and v0.3.1,
`pipe_radius=0.5` on an `LBMPipeNode` left the running velocity field
bit-identical to the old radius while `GET` and `to_dict()` reported 0.5,
so a reload ran a different pipe.  Such a write is now a 400 naming the key
and the reason, and `/checkpoint/load` refuses and undoes such a checkpoint;
to change the value, rebuild the node.  The same class was closed on the
FMI surface before it shipped.  See "[Persistence, REST and FMI follow the
params](#persistence-rest-and-fmi-follow-the-params)".

`MADD-ANO-025` — *A sharded `LBMNode` computed a different model from the
unsharded node* — shipped from v0.2.0 and is **resolved in 0.4.0**.
Sharding the axis of the pressure faces imposed the face pressure at every
seam (1.15x the unsharded centreline velocity at v0.2.0 and v0.3.1, 2.10x
once MADD-ANO-020's closure was corrected), and the wrapper's default
`boundary="edge"` halo fill disagreed with the node's periodic streaming
(-0.27% at the tags).  The first case is now refused at the first sharded
step, and the second at construction: `ShardedStencilNode` keeps its STABLE
`"edge"` default and refuses any fill other than the one the node declares
through `halo_boundary()`, so wrap an `LBMNode` with `boundary="periodic"`.
See "[A sharded `LBMNode` is the unsharded node, or it
refuses](#a-sharded-lbmnode-is-the-unsharded-node-or-it-refuses)".

`MADD-ANO-026` — *With `waveform_iterations > 1`, `coupling_diagnostics()`
reported only the last sweep's pass count* — shipped in every tagged release
and is **resolved in 0.4.0**.  An earlier sweep that exhausted
`max_iterations` read as `iterations=1` whenever the last sweep converged in
one pass (the same counts at v0.1.0, v0.2.0 and v0.3.1).  `iterations` is now
the largest sweep's count and `total_iterations` the sum; the state is
unchanged.  A cap check on `iterations` needs no change; read
`total_iterations` for the work done.

`MADD-ANO-027` — *Sub-cycled coupling relaxes no waveform* (open,
context-dependent, since 0.1.0).  `waveform_iterations > 1` re-solves the
same fixed point rather than relaxing a boundary waveform, and the
sub-step interpolation runs between iterates: after a converged first
sweep each later one adds at least one more pass, moving the state by
about one residual, `"quadratic"` interpolation is
`"linear"`, and `"linear"` differs from `"constant"` only by an
O(tolerance) in-pass change.
No state is wrong; the method the parameters name is not implemented.
The parameter is marked experimental; use `waveform_iterations=1`.  Real
waveform relaxation is planned for 0.5.0.  Both entries are described
under "[`waveform_iterations`: the report counts every sweep, and the
sweeps are
restarts](#waveform_iterations-the-report-counts-every-sweep-and-the-sweeps-are-restarts)".

`MADD-ANO-028` — *A sharded stencil node on a mesh axis of one device ran
with periodic global halos whatever `boundary` said* — shipped from v0.2.0 and
is **resolved in 0.4.0**.  `halo_exchange` skipped the `"edge"`/`"zero"` fill
on an axis of one device, whose exchange returns the shard's own opposite
edge, so a sharded model developed or tested on one device ran on a ring: a
`HeatNode` rod 0.40 from the unsharded one after 50 steps, where two devices
were bit-identical.  `MADD-ANO-029` — *the default `"edge"` fill of a halo two
or more cells wide copied the shard's first cells instead of repeating its
edge cell* — also shipped from v0.2.0 and is **resolved in 0.4.0**; the
4th-order `HeatNode` was 4.7e-3 from the unsharded rod.  No fill could close
that entirely, because that node's own cubic end closure is not a copy of any
cells; MADD-ANO-030's fix closes it.  See "[The halos at the edges of the
global grid](#the-halos-at-the-edges-of-the-global-grid)".

`MADD-ANO-030` — *A sharded `HeatNode` ignored its boundary temperature
inputs* — shipped from v0.2.0 and is **resolved in 0.4.0**: `update_padded`
never read `left_temperature` / `right_temperature`, so a sharded rod with
both ends held at 0 was 0.87 from the unsharded rod, and coupled sharded rods
did not couple.  The sharded rod now closes its ends exactly as `update` does,
and a `HeatNode` wrapped with `boundary="zero"` or `"periodic"` is refused;
pass the end temperatures instead.

`MADD-ANO-031` — *A `HeatNode` at `stencil_order=4` with no boundary input is
not insulated* (open, since `stencil_order=4` shipped in v0.1.0).  With no
`left_temperature` the datum is the end cell; at order 2 that makes the face
flux exactly zero, but the 4th-order cubic through it has a slope at the rod
end, and the mean of an `x**2` profile falls from 0.333313 to 0.332048 in 2000
steps where order 2 conserves it.  Give the end a temperature, or use
`stencil_order=2`; the fix is planned for 0.5.0.

`MADD-ANO-062` — *`HeatNode` accepted a non-positive length or timestep, a
negative diffusivity and `grid_points` out of order, and answered wrongly
without a word* — shipped in every tagged release and is **resolved in
0.4.0**.  A negative length left the Laplacian as it was, because it reads
`dx**2`, but flipped both rod-end fluxes.  Points out of order gave a
finite, wrong Laplacian.  A negative timestep or diffusivity ran the heat
equation backwards.  The constructor now raises `ValueError` for each,
and for a non-finite value.  See "[HeatNode: the boundary condition, the
4th-order stencil and the stability
bound](#heatnode-the-boundary-condition-the-4th-order-stencil-and-the-stability-bound)".

`MADD-ANO-010` — *A string spelling a non-finite token is refused by the
JSON serialisers* (open, context-dependent).  The other side of the change
above: `to_dict`, the USD JSON attributes and the FMI wire helpers raise on
a string leaf equal to `"NaN"`, `"Infinity"` or `"-Infinity"`, naming its
path, because the decoder would read it back as that float.  Spell such a
value differently; the registry entry has the reasoning and the workaround.

`MADD-ANO-003` — *AdaptiveNode frozen-set gradient omits a first-order term
at active-set switches* (open, context-dependent).  See "[What the gradient
actually is](#what-the-gradient-actually-is)" above for the measurement and
the registry entry in `docs/validation/known_anomalies.yaml` for the full
safety-relevance rationale.

**The rest of the reachable set.**  This section used to name three IDs,
and then a hand-counted total that was wrong by the time it was read, and
then claimed completeness while two reachable entries were missing from it.
Every entry in `known_anomalies.yaml` whose defect is reachable in 0.4.0 —
every entry whose status is not `resolved` or `duplicate`, so every `open`
one and every `partially_resolved` one, whose residual risk is live — is
listed in this section, above or below, and
`tests/compliance/test_release_notes_known_anomalies.py` fails if one is
not, reading the reachable set with the registry gate's own rule.  How many
that is, is the number `soup_package.md` §3 prints from the registry, and it
is not repeated here.  The registry is the record; these are the one-line
versions, and every one of them carries a workaround there.  Severities
follow the definition that 0.4.0 wrote down (`AnomalySeverity` in
`maddening.core.compliance.anomaly`): a silent wrong result is never
`minor`, which moved thirteen entries from `minor` to `major`, and
unauthorised access through a surface a shipped default exposes to other
hosts is `critical`, which moved one from `major` to `critical`.

`MADD-ANO-002` — *HeatNode CFL stability not enforced at runtime* (open,
context-dependent).  Narrowed, not closed.  The constructor now refuses a
timestep above its stencil's Fourier limit, naming the number.  It does so
on a uniform grid and on `grid_points`, where it applies
`dt*alpha/min(h_L*h_R) <= 1/2`; the check used to skip a rod built on
points.  What it
cannot see is a `dt` handed to `update()` or a `thermal_diffusivity` /
`length` moved by a calibration run — both are trainable, both still reach
`nan` in tens of steps with **no warning of any kind**, and neither can be
checked inside a traced step without a host callback.  Bound the parameter,
or check the field for finiteness after the step.

`MADD-ANO-005` — *A coupling group's converged flag was a residual test, not
a bound on the distance to the fixed point* (partially resolved in 0.4.0,
context-dependent).  The criterion is now applied to an error *estimate*,
but the estimate is not a bound: on a hidden slow mode it understated the
true distance by **122x** with `ratio_usable=True` and `converged=True`.
Where the residual sequence gives no usable ratio the criterion falls back
to the pre-0.4.0 residual test exactly, and reports that it did -- and float32
noise alone rejects the ratio of a group contracting at a rate near 1 once its
residual is within about `2 floor / (1 - rho)` of its float floor, so such a
group stops on the raw residual (up to 200 thresholds from its fixed point at
rate 0.995, with `precision_limited=False`); read `ratio_usable=False` beside
`converged=True` as the raw test.  Making that rejection not a pass would
add passes to most groups at the default tolerance, which sits at the float
floor of a small group, so it is left to 0.5.0.  Under
`solver="ift"` with `diagnostics=True` a bound is now *reported* beside
it, `spectral_error_bound` (8.05x over on the same case); it is not
applied to the criterion, `fori` has none, and a non-linear `F` makes it
asymptotic.  See "[The error estimate, and the flag that does not certify
it](#the-error-estimate-and-the-flag-that-does-not-certify-it)".

`MADD-ANO-011` — *BallNode's declared discretisation is forward Euler; the
implementation is semi-implicit Euler* (open, context-dependent).  Read the
scheme from the algorithm guide, not from `discretization`.

`MADD-ANO-012` — *HeartPumpNode samples its cardiac inflow at the end of the
step, disagreeing with its own `derivatives()`* (open, context-dependent).
`update()` and `integrate_node()` integrate different waveforms through the
same node.

`MADD-ANO-013` — *Several nodes pin a float64 quantity to float32* (open,
context-dependent).  Widened this release from HeartPumpNode's
`backpressure` to the parameter and `initial_state` casts in
`RigidBodyNode`, `RigidBody2DNode` and `HeatNode`.  Under `jax_enable_x64`
these put a floor of about 3e-8 to 7e-8 relative on anything downstream of
the cast — `inertia` at `1.0` and at `1.0 + 1e-8` returns a *bit-identical*
angular velocity — independent of the step count.  Gradients still flow, so
it is a precision floor and not a dead gradient, and at the framework's
float32 default every one of the casts is a no-op.  One instance,
`BallNode`'s gravity, **is** fixed here.

`MADD-ANO-014` — *Every explicit integrator converges at 1st order when a
time-varying boundary input is supplied once per step, including the one
named 4th-order* (partially resolved in 0.4.0, context-dependent).  The
degraded path is still the default and still silent; see "[The integrators'
order is conditional, and the condition was not
stated](#the-integrators-order-is-conditional-and-the-condition-was-not-stated)"
above.

`MADD-ANO-016` — *`cloud/_skypilot.py` was written against a SkyPilot API
older than the supported floor* (partially resolved in 0.4.0, major, not
safety-relevant).  Every `CloudSession` launch, status poll, teardown and
preemption check called SkyPilot's pre-client-server API, so none of them
could work against any SkyPilot the package allowed, and a teardown that
failed logged and returned as if it had released the VM.  0.4.0 ports the
four calls and binds each against the installed SkyPilot's signature in CI,
which proves they are well formed and nothing more: no real launch,
teardown or preemption has been observed, so the entry's range stays open.
Confirm a teardown with `sky status` whatever the version, or use
`CloudLauncher`, which has spoken the current API since 0.1.0.

`MADD-ANO-017` — *`jax_enable_x64` does not reach the scan and solver
paths* (open, minor, context-dependent).  A freshly compiled graph refuses
to scan under x64 because its params are float64 and its state seed is
float32 (`run_scan`, `run_scan_with_history`, `run_sweep` all raise; a
`step()` first promotes silently and then unblocks the scan).  The entry
now also names two surfaces below `GraphManager`, found by the release
audit and re-derived on this tree: `implicit_euler_step` refuses a float32
state under x64 whatever the params — with none, with the float64 pytree,
with the pytree cast to float32 — because its Newton solve is float64 and
the `fori_loop` carry changes type, and that has been so in **every
release since 0.1.0**, which is why the entry's range widened from
`>=0.4.0` to `>=0.1.0`; and `update(params=)` against
`integrate_node(..., params=)` on the same node and params differ in
*dtype*, not value (`update` promotes every field, the integrator only
the fields whose right-hand side reads a parameter).  Three workarounds
in the registry; leaving x64 off is the supported configuration.  Now
**partially resolved**: `SpringDamperNode` seeds its state in the dtype its
constants make (float64 under x64 for Python-float constants, float32 for
float32 ones), so a freshly compiled spring graph scans under x64 -- the
parameter guide's system-identification recipe and the sysid demo, which
run on it, raised there.  Every other node that seeds float32 is as
described.

`MADD-ANO-021` — *A node that branches on its own state inside `update()`
returns the gradient of the branch it took* (open, major,
context-dependent; present since 0.1.0, registered in this release).
`BallNode`'s gradient through a bounce omits the contact time.  The
derivative of the post-bounce height with respect to the drop height is
exactly 0 against an exact +0.5892, and at every step size tried; the
gravity derivative is 13 times too large.  The trajectory is right to one
step, and nothing warns.  Any contact, threshold or stick-slip node has
the same defect, and the node-authoring guide now says so.  Smooth the
switch, or do not differentiate through it.  Event-time derivatives are
planned for 0.5.0.

`MADD-ANO-022` — *A mapped edge whose point reference names a grid derived
from a trainable parameter keeps the constructor's geometry* (open, major,
context-dependent).  A `{"node", "field"}` reference to a uniform
`HeatNode`'s `grid_x` computes the weights from the constructed `length`,
and calibrating `length` -- a fit, or any `params=` pytree -- leaves them
there.  In the reproducer, a fit of `length` through the mapped edge stops
at 0.22 against a true 1.25, with a small loss, and nothing warns.  Writing
the result back into the running graph is refused since `MADD-ANO-063`'s
fix; the fit itself is not.  Freeze the parameter, or fit the geometry from
the node itself and rebuild the graph.

`MADD-ANO-063` — *A new value of a parameter from which a node derives the
points of a mapped edge was taken by `PUT /graph/params` and `gm.params`,
run with the old points' mapping weights, and saved as a config that does
not load* — introduced and **resolved within 0.4.0** (major,
context-dependent).  The live-write side of `MADD-ANO-022`: `length` on a
uniform `HeatNode` is a leaf its own step reads, so the route answered 200
and a `gm.params` write ran, the rod on the new length and the mapping on
the old grid's weights, and `from_dict()` refused the saved config's
mapping.  Every door that writes a value into the running graph now asks
whether the node rebuilt with it reads a referenced field differently, and
refuses it if so: the route with a 400 naming the edge, `gm.params` at the
next run, `to_dict()` and `save_state()`, a checkpoint at `POST
/checkpoint/load`, and an exported FMU by leaving the parameter fixed.
Rebuild the node and its mapped edge to change such a value.

`MADD-ANO-023` — *The FMU TCP bridge authenticates no caller* (open,
major, context-dependent).  Any process that can reach the bridge's port
can read the model, write its parameters, step it, and hold its one
instance slot.  Since `MADD-ANO-083`'s fix, what it writes before an
instance begins is reset when the instance claims the slot.  The `hello` token identifies the model; it is not a
credential.  The default bind is loopback, so on a single-user host or in
a container nothing else can reach it.  On a shared host, firewall the
port.  Never bind a non-loopback address on an untrusted network.

`MADD-ANO-035` — *`ShardedUnstructuredNode` reads a global-order state as
partition layout on a balanced partition not in global order* (open, major,
context-dependent; since 0.3.0).  When every shard is full and the cells are
not in global order, a state written in global cell order with
`set_node_state` has exactly the partition layout's shape, is stored without
a word, and each cell steps from another cell's value.  Renumber the cells
so that each device owns a contiguous ascending block, or convert the state
with `partition_value` before writing it; an explicit layout on the state
write is planned for 0.5.0.

`MADD-ANO-050` — *Two `HeatNode` rods coupled end to end are unstable above
Fourier number 3/8 (`stencil_order=2`) or 0.226 (`stencil_order=4`) once the
exchange converges, below the limit each rod's constructor accepts* (open,
major, context-dependent; new in 0.4.0).  The converged exchange makes each
rod's Dirichlet datum an implicit value, and the pair grows an interface
mode that flips sign every step.  The limit depends on the stencil.  At
`stencil_order=2` it is exactly 3/8, against 1/2 for one rod.  At
`stencil_order=4` it is 0.2261 (0.2261266 as `n_cells` grows, 0.2261215 at
5 cells), derived from the interface mode and confirmed on the node's own
operator, against 5/16.  A chain of three or more rods shorter than about
20 cells each goes unstable slightly lower, down to 0.3730 (order 2) and
0.2248 (order 4) for 5-cell rods.  It arrived with MADD-ANO-007's rod-end
fix.  `compile()` now emits a `UserWarning` naming both rods, their Fourier
numbers, the limit and this entry, when a coupling group holds two uniform
rods coupled through `extract_first` / `extract_last` and either rod is
past its pair limit.  It is a warning, not a refusal, and it judges only
that pattern, each pair against the pair figure.  A custom transform, a
wrapped or non-uniform rod, or a sub-cycled pair of different timesteps is
not judged, a short-rod chain between its own lower limit and the pair
figure is not warned about, and nothing else says anything before the
field is visibly wrong.  Keep `dt*alpha/dx**2` below 3/8 at order 2 and
below 0.226 at order 4 on rods coupled end to end through their
temperature inputs (lower on a short-rod chain), or exchange the data
without a coupling group, lagged by a step.

`MADD-ANO-032` — *After a parameter write and `compile()`, the sharded
wrappers kept computing with what they had built from the old value*
(partially resolved in 0.4.0, major, context-dependent).  Every `compile()`
now drops the wrappers' compiled steps, so the graph path takes the write.
A wrapper's `update()` called on its own, outside a graph, still keeps its
first trace after a write to a legacy node's constant, silently; call
`invalidate_static_cache()` on the wrapper after such a write.

`MADD-ANO-036` — *A config round trip dropped a node's sharding with no
word* (partially resolved in 0.4.0, minor, context-dependent).  `from_dict`
now warns and names the `replace_node` call that wraps the node again.  A
USD stage of a sharded graph still saves and cannot be loaded (a
`TypeError`); save such a graph with `to_dict()`.

`MADD-ANO-047` — *`PUT /graph/params` accepted a write that flipped a branch
the node fixed at construction* (partially resolved in 0.4.0, major,
context-dependent).  The route now refuses such a write.  A `gm.params`
write or a fit is not asked: `G` moved through zero on a multiphase
`LBMPipeNode` leaves the running pipe multiphase and its saved config
single-phase, 0.84 apart in density after ten steps.  Bound `G` to one sign.

`MADD-ANO-048` — *`PUT /graph/params` accepted values the node's own
constructor refuses, so a graph saved after the write could not be loaded*
(partially resolved in 0.4.0, major, context-dependent; since 0.1.0).  The
route now builds the node from the params a save would carry and refuses
the write with the constructor's message.  The path that keeps `047`
partially resolved is open here too: a `gm.params` write, and so a fit, is
not asked, and
`to_dict()` saves a value the constructor refuses without a word.  A
`HeatNode` whose `thermal_diffusivity` is moved past its Fourier limit, an
`LBMPipeNode` whose `rho_gas` is moved above its `rho_liquid`, or a
`HeatNode` written a negative `length` all save a graph that `from_dict()`
refuses.  Bound such a parameter's `ParamSpec` inside the constructor's
range, and check that a saved config reloads before relying on it.

`MADD-ANO-049` — *A non-finite value written through `PUT /graph/params`
broke every later `GET`* (since 0.1.0, minor, not safety-relevant) — is
**resolved in 0.4.0**.  The route refuses it, and a non-finite value set
in-process — including a calibration that diverged to `NaN` in `gm.params`
— is served by `GET /graph/params` as its quoted token rather than a 500
(see "[Non-finite numbers are valid JSON
now](#non-finite-numbers-are-valid-json-now)").

`MADD-ANO-068` — *XLA miscompiles a `ShardedStencilNode` step inside a
loop when a sharded static is replicated over a mesh axis and read in the
halo beside a window at the shard's offset* (partially resolved in 0.4.0,
major, context-dependent, arriving with 0.4.0's jax floor).  On jaxlib
0.10.2, 0.11.0 and 0.11.2 a node that reads a sharded `StaticArray` in its
halo and another array through a window at its `shard_info` offset, with the
static replicated over a mesh axis of two or more devices, is compiled
wrongly inside a loop: on a mesh axis the `axis_map` leaves unused every
cell came back 1e-3 to 1e-2 off, silently, and on a pencil whose static is
split along one axis only the step failed to compile.  The loop can be
`run_scan` or a coupling group's own iteration: such a node in a group under
`solver="fori"` was 3.4e-3 off on `step()`.  The miscompile reproduces
without MADDENING; jaxlib 0.5.3, the newest the 0.3.x pins allowed, does not
show it.  0.4.0 refuses it rather than compute it: `run_scan`,
`run_scan_with_history`, `run_sweep`, `run_adaptive_scan` and
`sysid.windowed_loss`, gradients through them included, raise a
`RuntimeError` naming the node, the static, the mesh axis and this entry,
and so does every entry point for such a node inside a coupling group.
Whether a node reads its static that way is read from its own
`update_padded`, so a static read only in its interior, one never read
(`HeatNode`'s `grid_x`), a window into the static itself (the multi-device
validation goals' field) and `LBMNode` are not refused.  Still live: a loop
written outside `GraphManager` around such a node is not asked.  Use a mesh
without the axis the static is copied along, or `step()` / `run()`.

`MADD-ANO-098` -- *Two `SpringDamperNode`s anchored on each other in a
converged coupling group gain momentum without bound when
`stiffness*dt > damping`, far inside the single node's stability limit*
(open, major, context-dependent; since 0.1.0).  Each node is explicit in its
own position and implicit in its partner's, so the spring forces do not
cancel.  The centre-of-mass velocity of an equal pair is multiplied by
`(m - c*dt)/(m - k*dt**2)` per step, which grows exactly when `k*dt > c`
wherever the coupling iteration converges (`k*dt**2 < m`) and `c*dt < m`.
The relative motion stays stable, so the spring settles while the pair
drifts off.  At `k = 1000`, `c = 2`, `m = 0.5`, `dt = 0.01` the drift grows
by 1.2 per step, while one node sits at 0.14 of its own bound
`k*dt**2 + 2*c*dt < 4*m`.  Heavier damping has a limit too: past
`c*dt + k*dt**2 = 2*m` the common velocity flips sign and grows.  The
node's `NodeMeta` and algorithm guide now state the bounds.  The numerics
are unchanged.  `compile()` now warns about such a pair, naming both nodes,
the growth factor and this entry, when the converged step grows (an
unequal pair by its spectral radius); it recognises only bare or
`identity` edges and never refuses the graph.  Keep
`stiffness*dt <= damping <= (2*mass - stiffness*dt**2)/dt` on such a pair.

`MADD-ANO-128` — *The adaptive steppers' default `atol=1e-6` and
`dt_min=1e-8` are absolute* — and `MADD-ANO-129` — *the deprecated
`calibrate` and `tune_coupling_params` default to absolute thresholds* —
are **open by design**: user-facing tolerances whose defaults match scipy's
and the old calibration loops', kept so that no run moves.  Pass `atol` in
the state's own units (a decay at `2**-30` then takes exactly the steps it
takes at scale one), and use `sysid.fit` for calibration.  See "[Every small
constant says its units](#every-small-constant-says-its-units)".

`MADD-ANO-156`, `MADD-ANO-157`, `MADD-ANO-158` and `MADD-ANO-159` were
found by the coupling topology harness, and every release carried them.
`156` and `157` are **open**, deferred to 0.5.0; `158` and `159` are
**resolved in 0.4.0**.  See "[The coupling runtime, configuration
interactions and graph
shapes](#the-coupling-runtime-configuration-interactions-and-graph-shapes)"
and "[The known coupling, sysid and FMU findings,
fixed](#the-known-coupling-sysid-and-fmu-findings-fixed)".

- **MADD-ANO-156** (open): a flux edge between a group member and an
  outside node, either way, fails to trace with `KeyError`.  Carry the
  quantity in a state field, or put both nodes in the group.
- **MADD-ANO-157** (open): a flux edge that an ungrouped cycle staggers
  fails the same way, depending on the build order.  Add the producer
  first.
- **MADD-ANO-158** (resolved): a typed PRNG key in a member failed under
  `"ift"` with `"aitken"` or `"fixed"`; the accelerator is now sized from
  the floating fields only.  Before 0.4.0, store `jax.random.key_data`, or
  use another iterator.
- **MADD-ANO-159** (resolved): a group whose members are joined only
  through an outside node, with no cycle, read that node a step late
  without a warning; `compile()` now names the staggered edge.  Group only
  nodes that form a cycle.

- **MADD-ANO-204** (resolved; carried by no release): the library's own warning probes silenced every thread of the process while one ran, and two that overlapped in two threads could leave every warning ignored for good; a probe now silences its own thread only.
- **MADD-ANO-205** (resolved; shipped from 0.1.0 to 0.3.1): a config from `to_dict()` shared its containers with the graph, so editing one to build a variant changed the graph it came from; before 0.4.0, edit `copy.deepcopy(gm.to_dict())`.
- **MADD-ANO-206** (**open**; shipped in every tagged release): a bfloat16 state or parameter leaf cannot be restored from a checkpoint (`load_state` raises); hold a state that must be checkpointed in float16 or float32; deferred to 0.5.0.
- **MADD-ANO-207** (resolved; carried by no release): a checkpoint value loaded into a bfloat16 leaf was cast with no check, a float32 `1e-44` as `0.0`; it is refused as for any other dtype.

- **MADD-ANO-217** (**open**; carried by no release before this one): `fit_lm` from a stiffness started 18 to 80 times too high can stop short, `converged=False`, on the end of a wide damping range; start it again from the returned parameters.

- **MADD-ANO-220** (resolved; shipped in every tagged release): `step`, `run` and `run_adaptive` stored a step that changed the shape of a state leaf (a list given for a scalar constant), after which the state's checkpoint did not reload; such a step now raises `ValueError` and stores nothing.

`MADD-ANO-230` — *A direction the Arnoldi breakdown test takes for rounding can
carry the dominant mode of a Jacobian far from normal: float32 fan-out hubs
with a field 1e-4 to 1e-2 of the rest read rho_spectral 0.157 for 0.349 and
0.503 for 0.397, settled* — **open**.  a direction the Arnoldi breakdown test
takes for rounding can carry the dominant mode of a float32 group with a field
below a hundredth of what it drives (0.157 for 0.349, settled); run such a
group in float64

`MADD-ANO-232` — *A `gradient_relative_error_bound` whose flag is `False` can
read NaN on a GPU backend where the CPU backend reads a number* — **open**.  Do
not compare not-usable values between backends.

`MADD-ANO-239` — *spectral_usable does not see a pass whose Jacobian one
float32 rounding of a same-pass value moves: a float32 Gauss-Seidel group whose
member multiplies two fields held in units of 1e6 apart read rho_spectral 0.324
for 0.0238, settled* — **open**.  A float32 Gauss-Seidel group whose member
multiplies two fields held in units 1e3 or more apart can read a settled
`rho_spectral` far from the radius at its float floor (0.324 for 0.0238; the
spectral keys are new in this release).  Run such a group in float64 or under
Jacobi.

`MADD-ANO-246` — *the geometry self-check can withhold the honest report of a
float32 Gauss-Seidel group for a step in which a position one member builds
and another reads lies within about 2e-5 of a spacing of a lattice plane* —
**open**, never released.  The reason says so; read the next step's report,
or run the group under `iteration_mode="jacobi"` or in float64.

`MADD-ANO-247` — *under the interface norm the float floor of a value an edge
delivers is taken at the delivered value's own magnitude, so a group whose
mapping differences two entries of one field can stall under its bound with the
flag set, under either schedule* — **open**, never released.  Read such a group
under the L2 or the mixed norm with `iteration_mode="jacobi"`, or have the
producer hold the difference as a field of its own.  The same holds for a
geometry-dependent gather that samples a field near its zero at float32
positions: hold the positions in float64.

`MADD-ANO-254` — *a change that has to cross a field inside the dead band is not
seen until it reaches a kept field* — **open**: a group with `atol > 0`, whatever its
member count and schedule, can report `converged=True` after one pass far from its
fixed point; leave `atol` at `0.0` (`compile()` warns for every group that sets it).

`MADD-ANO-257` — *the float floor of a coupling report does not count the rounding
of a mapped edge's row sum* — **open**, never released: a group at its float floor
behind a long row can report `spectral_error_bound` below its true distance.  Behind a
row of more than ten entries of any static mapping (a dense matrix by its width, a
sparse mapping in either layout) the report withdraws `spectral_usable` and
`gradient_bound_usable` with a reason; a geometry-dependent scatter is counted at
its number of points.
Use a wider dtype at the same tolerance.

`MADD-ANO-258` — *under the interface norm the float floor
counts no position for a `multilinear_grid` mapping read at its source and
anchored at its target* — **open**, never released: with no edge in the group
that delivers a value at those positions, float32 positions thousands of
spacings from the grid's first point are not flagged (2.6 to 16 tolerances off
beside a floor of 0.095).  Hold the positions in float64, or anchor the scatter
at its source.

`MADD-ANO-259` — *the float floor of a coupling report does not count what an
internal edge delivers from a boundary flux* — **open**, never released: a transform
that narrows the dtype on a flux edge can leave `spectral_usable=True` beside a bound
far below the distance.  Carry a value that is narrowed on its way as a state field.

`MADD-ANO-260` — *the mapped-row guard of a coupling report reads static mappings
only* — **open**, never released: a sum of many terms written inside an edge
`transform=` keeps `spectral_usable=True` at the float floor beside a bound below the
distance.  Declare such an operator as a static sparse mapping (`mapping=`).

`MADD-ANO-262` — *`ShardedStencilNode` builds the wrapped node's whole initial state
on the default device* — **open**, since 0.2.0: at construction and in
`initial_state()`, so a grid that fits the mesh but not one device ends in an
out-of-memory error before its first step.  Build the state one device's block at a
time (`benchmarks/multigpu/run_capacity.py` shows how).

`MADD-ANO-264` — *lineax's CG returns NaN in float32 at scale* — **partially
resolved**, since 0.3.0: without a preconditioner `sharded_cg`'s default backend is
lineax's CG, which answers NaN in every entry with `converged=False` above about 1e5
unknowns on a system of order-one scale, and for a smooth right-hand side from about a
thousand.  From 0.4.0 the default call solves such a system with the loop backend
instead and reports its true residual; where lineax answered, nothing changes.  Still
open: `backend="lineax"` asked for by name (NaN) and `ift_linear_solve(solver="cg")`
(raises): pass `backend="loop"` or `solver="gmres"`, or solve in float64.

`MADD-ANO-270` — *with `differentiable=True`, `converged` says nothing of the
derivative solves of `sharded_cg` / `sharded_gmres`* — **open**, new with the flag in
0.4.0: each derivative is one more solve from zero under the same `rtol` and
`max_iters`, and nothing reports whether it met its tolerance.  A solve that converges
quickly only because of its `x0` (or an `atol`, or a special right-hand side) reads
`converged=True` beside a derivative that is off (2.6e-3 measured at the default
tolerance and budget with a warm start).  Give `max_iters` what a solve from zero
needs.

`MADD-ANO-280` — *`sharded_gmres` on its default route counts `max_iters` and `.iters`
in restart cycles, not steps* — **open**, since 0.3.0: on 1000 unknowns a budget of 10
executed 1,583 to 4,490 operator products and reported `iters` of 32 to 89 (24
white-noise right-hand sides, lineax 0.0.7 and 0.1.1), where the loop backend executed
53.  The value and its flag are right.  For a bound on the work pass `backend="loop"`.

`MADD-ANO-282` — *with `differentiable=True` a solution just under the top of the
dtype's range is reported as not converged, or returned as NaN* — **open**, new with
the flag in 0.4.0: where the solution is within a factor of the operator's diagonal of
the dtype's largest number, the call returns the right value with `converged=False` and
an infinite `residual_norm`, and with an `x0` or an `atol` NaN, where the ordinary call
answers.  Rescale such a system, or solve it in float64.

`MADD-ANO-275` — *IQN acceleration under the Jacobi schedule can stall,
unconverged, on a coupling group the plain Jacobi loop solves* — **open**.
Since 0.1.0. `iqn-ils` and `iqn-imvj` under `iteration_mode="jacobi"` can
stall, reported unconverged (`converged=False` at the cap), on a group the
plain Jacobi loop solves; use Gauss-Seidel with IQN, or Jacobi with no
acceleration or Aitken.
`MADD-ANO-277` — *a constructor argument a node keeps outside `params` is dropped by
every save* — **partially resolved**, since 0.1.0: `to_dict()` and `save_graph_to_usd`
now warn, naming the node and the arguments; the argument is still not saved, and an
argument taken in by `**kwargs` or `*args` is not looked at.  Pass every constructor
argument to `super().__init__(...)`.

`MADD-ANO-285` — *under `acceleration="aitken"`, `"iqn-ils"` and `"iqn-imvj"` a
coupled run depends on the units one field is written in* — **open**, since 0.3.0:
write a group's quantities in comparable units, or use `"none"` or `"fixed"`.

**Resolved.**  `MADD-ANO-006`, `019`, `020`, `024`, `025`, `026`, `028`,
`029`, `030`, `049`, `062` and `063` above, `064`, `065`, `066` and `067`
in "[What the differential sharding harness
found](#what-the-differential-sharding-harness-found)", and `MADD-ANO-001`,
`004`, `007`, `008`,
`009`, `015`, `018`, `033`, `034`, `037`, `038`, `039`, `040`, `041`, `042`,
`043`, `044`, `045`, `046`, `051`, `052`, `053`, `054`, `055`, `056`,
`057`, `058`, `059`, `060`, `061`, `069`, `070`, `071`, `072`, `073`, `074`,
`075`, `076`, `077`, `078`, `079`, `080`, `081`, `082`, `083` to `085` in
"[What the round-4 audit found in the FMU
export](#what-the-round-4-audit-found-in-the-fmu-export)", and `086`, `087`,
`088`, `089`, `090`, `091`, `092`, `093` and `106` to `112`, and `113`,
`114`, `115`, `116`, `117` and `118` in "[The coupling runtime, round-4
audit](#the-coupling-runtime-round-4-audit)", `121` and `122` in "One
sync point and a per-column floor: the round-6 sysid fixes", `123`,
`124`, `125`, `126`, `127` and `130` in
"[Every small constant says its units](#every-small-constant-says-its-units)",
and `131`, `132`, `133` and `134` in "[The REST round-6
audit](#the-rest-round-6-audit-the-budget-at-the-sweep-a-profile-that-restores-routes-that-publish)",
and `135` to `137` in "Units the guard could see, and values too small to
check: the claims-inventory sysid fixes", `138` to `141` in "[The REST
claims follow-up](#the-rest-claims-follow-up-what-a-route-says-it-did-it-did)",
`142`, `143`, `144`, `145` and `146` in "[The coupling runtime, round-5
audit (against the claims
inventory)](#the-coupling-runtime-round-5-audit-against-the-claims-inventory)",
`147` to `149` in "[What the round-7 audit found in the FMU
export](#what-the-round-7-audit-found-in-the-fmu-export)", `150`, `151`,
`152` and `153` in "x64, replayed warm starts and arguments that are
refused: the round-7 sysid fixes", `154` and `155` in "[Krylov solves
of a non-finite right-hand
side](#krylov-solves-of-a-non-finite-right-hand-side)", `158` to `162`
in "[The known coupling, sysid and FMU findings,
fixed](#the-known-coupling-sysid-and-fmu-findings-fixed)", `163` to `166`
in "[The REST round-7
audit](#the-rest-round-7-audit-a-load-is-a-write-a-hybrid-shares-its-params-a-run-says-how-far-it-got)",
`167`, `168` and `169` in "[What the round-8 audit found in the FMU
export](#what-the-round-8-audit-found-in-the-fmu-export)", `174`, `175`,
`176` and `177` in "The whole float range, writes the graph missed, and
counts in any spelling: the round-8 sysid fixes", `178` to `185` in "[The REST round-8
audit](#the-rest-round-8-audit-a-write-back-is-a-write-a-forwarded-peer-is-unknown-a-load-reads-headers-first)",
`196` in "[Registering a mapping kind](#registering-a-mapping-kind)", and
`197`, `198`, `199`, `200` and `201` in "[What the round-9 audit found in
the FMU export](#what-the-round-9-audit-found-in-the-fmu-export)",
`202` and `203` in "[Bounded transforms in the fitters: the round-9 sysid
fixes](#bounded-transforms-in-the-fitters-the-round-9-sysid-fixes)", are
resolved in 0.4.0; they are in the
registry with their closure evidence, and most are described in the
sections above.  Eight of them are `critical`.  One is `083` (a second FMU
instance on one bridge started where the first one ended, which no release
carried), and the other seven are security
defects the release fixed: `015` (the ZeroMQ transports bound every
interface with no authentication or encryption), `051` (the HTTP API served
every route, `/cloud/launch` included, with no credential while the
container bound `0.0.0.0`), `052` (the checkpoint routes wrote and probed
any server path for that same anonymous caller), `053` (the signaling
server validated its own token, so it admitted every client), `054` (the
FMU bridge unpickled an importer's state blob, which no release carried)
and `076` (a web page in the user's browser could drive a loopback-bound
server: first through no `Origin` check, then, on the 0.4.0 tree, through
DNS rebinding, which the `Host` allowlist closes), and `179` (an
`X-Forwarded-For` header got a rebound page past that allowlist and the
peer backstop on the 0.4.0 tree, which no release carried); see
"[Security](#security)".  `056` to `058` are the sharding and LBM
refusals of the third sharding audit, `058` among them the default
`propeller_x` that a pipe of 10 planes or fewer now refuses (see the
breaking-changes table).  `059` to `061` are the coupling and stepper
fixes of "[The coupling runtime, second
audit](#the-coupling-runtime-second-audit)": every coupling acceleration
now acts on floating fields only (`059`), a group-internal flux edge that
the interface norm or a sub-cycled member's linear interpolation would read
is refused at compile (`060`), and `run_adaptive` advances its clock by the
step it accepts at `dt_min`, not by `dt_min` (`061`).  `062` is the `HeatNode`
constructor's refusals of a non-positive length or timestep, a negative
diffusivity and `grid_points` out of order.  `063` is the refusal of a
write that moves the points a mapped edge was built from.  `069` is the
identifiability guard that held directions the data determined and moved a
fit uphill (see "[The guard holds only what the curvature confirms, and
never raises the
loss](#the-guard-holds-only-what-the-curvature-confirms-and-never-raises-the-loss)").
`075` to `082` were registered after the round-3 confirmation audit found
fixed defects with no entry: opening a USD stage imported the module it
named (`075`, since 0.1.0), a surrogate swap dropped edges' `additive` flag
and units (`077`, since 0.1.0), FMI export of a real graph declared no inputs
and a 1e-3 step (`078`, since 0.3.0), a recompile re-phased a multi-rate
graph (`079`, since 0.1.0), and three majors of this cycle's own code: the
FMU bridge split a dotted node name at its first dot (`080`), `compile()`
traced against a stale sharded static (`081`), and the bridge's `set_state`
skipped the value checks `set` applies (`082`).
`086` to `091` are the REST fixes of "[The REST round-4 audit: sizes before
construction, an honest
stop](#the-rest-round-4-audit-sizes-before-construction-an-honest-stop)", and
`092` registers the unbounded `/sim/run` and `/surrogate/train` requests of
v0.1.0 to v0.3.1, which 0.4.0's request bounds closed.  `093` is a
`node.params` write after compile, which every later compile dropped in
favour of the stale `gm.params` leaf (see "`node.params` after compile,
and the round-4 sysid fixes").
`094` and `095` are the coupling fixes of "[The coupling runtime, round-3
confirmation audit](#the-coupling-runtime-round-3-confirmation-audit)": a
Gauss-Seidel pass's float floor counted its worst node rather than its chain
of same-pass reads, so the spectral and gradient bounds of a long stalled
chain read low with their flags set (`094`), and after `run_adaptive*` the
coupling report described only the second kept half step, hiding a first
half that ran out of passes (`095`, since 0.1.0).
`096` and `097` are the step a graph reports.  `gm.timestep` was the GCD of
the node timesteps, shorter than a step on a graph with a sub-cycling group,
so the runner's and the relays' clocks ran slow (`096`, since 0.1.0).  USD
`baseDt` and the FMU default step read an attribute nothing set (`097`,
since 0.1.0).  See "[`gm.timestep` is the step the graph
takes](#gmtimestep-is-the-step-the-graph-takes)".  `099` registers the
partial `external_inputs` that `step` did not zero-fill and the misspelt
names it dropped, from v0.1.0 to v0.3.1, which 0.4.0 fixed during its
development.
`100` to `103` are the FMU bridge fixes of "[What the round-5 audit found
in the FMU bridge](#what-the-round-5-audit-found-in-the-fmu-bridge)": a
`master_dt` other than the graph's step (`100`), a Boolean set by its
truthiness (`101`), an `fmi3Get` / `fmi3Set` of another type than the
variable's (`102`) and an archive missing an input (`103`).  `104` is the
fitters' parameter carried past a clip or clamp and called converged, and
`105` registers the `node.params` write that `gm.step` and a new
`run_scan` length ran differently, from v0.1.0 to v0.3.1 (see "Bounded
coordinates, residual units, record phase and counted writes: the round-5
sysid fixes").
`106` to `112` are the REST fixes of "[The REST round-5 audit: one graph
lock, bounded jobs and bodies, streams that follow the
graph](#the-rest-round-5-audit-one-graph-lock-bounded-jobs-and-bodies-streams-that-follow-the-graph)",
each since 0.1.0: concurrent requests that lost steps (`106`), a training
job that reset the live simulation (`107`), streams that served the state
from before a checkpoint load (`108`) and a replaced node's layout
(`109`), a dead runner reported running (`110`), a runner that burst
through a pause (`111`) and frames encoded on the event loop (`112`).
`119` is the FMU-state archive that installed new interface-mapping
weights (see "[What the round-6 audit found in the FMU
export](#what-the-round-6-audit-found-in-the-fmu-export)").
`120` is the schedule that ran a node reading a cycle
ahead of the cycle, and so a step late, when the node had been added
first, from v0.1.0 to v0.3.1 (see "[The coupling claims inventory, and
what it found](#the-coupling-claims-inventory-and-what-it-found)").
`121` is the round-5 Marquardt floor that made `fit_lm` depend on the
parameters' units, and `122` the readers of `gm.params` that missed a
pending `node.params` write (see "One sync point and a per-column floor:
the round-6 sysid fixes").
`135` is the identifiability guard that held a determined parameter
written in other units, `136` the bounds check that a subnormal passed and
`137` the value check that stored an underflow as zero (see "Units the guard
could see, and values too small to check: the claims-inventory sysid
fixes").  `138` is `load_state`'s unchecked cast of a wider checkpoint
value and `139` the stride a `PUT /sim/stride` reset, both from v0.1.0 to
v0.3.1; `140` is the token file that kept a looser mode and `141` the
checkpoint save that wrote a manifest while saying nothing was written (see
"The REST claims follow-up").
`147` is the FMU built over a pending structural `node.params` write, which
ran the old model, `148` the `log` parameter without a lower bound that
advertised no `min`, and `149` the bridge's time tolerance, which had an
absolute floor and let a biased importer's reported time leave the physics
behind (see "[What the round-7 audit found in the FMU
export](#what-the-round-7-audit-found-in-the-fmu-export)").
`154` is the IFT tangent and adjoint that read 0.0 for a non-finite
tangent or cotangent, and `155` the public Krylov solvers that returned
zeros for a non-finite right-hand side, `sharded_cg` since 0.3.0 (see
"Krylov solves of a non-finite right-hand side").
`163` is the checkpoint load that restored parameter values `PUT
/graph/params` refuses, `164` the `HybridNode` parameter write that was
answered and lost (from v0.1.0 to v0.3.1), `165` the `/sim/run` that said
nothing had changed after it had stepped, and `166` the `/sim/profile` 500
for a graph that cannot step (from v0.2.0 to v0.3.1; see "The REST round-7
audit").
`178` is the params write back to a node's own value that skipped the
combined checks, `179` the forwarding header that named a trusted peer,
`180` the state write that stored every field as float32 and parsed text,
`181` the zero, negative or non-finite timestep a node was added with,
`182` the node named `_meta`, `183` the `/sim/start` of an empty graph,
`184` the 422 that was a 500 for a NaN, and `185` the checkpoint load that
decompressed every member first (`180` to `185` from v0.1.0 to v0.3.1; see
"The REST round-8 audit").
`197` is the FMU-state restore that started the bridge's drift count again,
`198` the `Int64` / `UInt64` the FMU wire rounded, `199` the negative zero
the wrapper's JSON path stored as positive, `200` the integer of the other
signedness that `load_state` wrapped, and `201` the `selected_outputs` pair
that exported nothing, from v0.3.0 to v0.3.1 (see "[What the round-9 audit
found in the FMU export](#what-the-round-9-audit-found-in-the-fmu-export)").
`018`, `019`, `054`, `063`, `066`, `069`, `070`, `072`, `073`, `080`, `081`,
`082`, `083`, `084`, `085`, `089`, `093`, `094`, `100` to `103`, `104`, `119`, `121`, `122`, `126`, `127`, `135` to `137`, `140`, `141`, `147` to `149`, `150` to `153`, `154`, `163`, `165`, `174`, `175`, `197`, `198`, `199` and `200` were introduced and fixed
within this development cycle, so their `affected_versions` read `none`: no release
carried them.
`214` (resolved): remove_node left a coupling group naming the removed node:
every compile and step failed until a node of that name was added, and DELETE
/graph/nodes answered 200.
`215` (resolved): A length that moves the points a mapped edge was built from
was taken after a fit or a checkpoint load had moved another leaf, or with that
leaf in the same request: the mapping kept the old grid and the save did not
reload.
`216` (resolved; carried by no release): `fit_lm` reported `converged=True`
beside a jump of a residual that is not differentiable in the trained
parameters; it now ends `converged=False` and says so.
`218` (resolved): A value handed to step or a scan ran in the dtype it arrived
in, not the external input's declared dtype: an x64 graph with a default-
declared (float32) input ran 0.1 as a float64 where its own FMU ran
float32(0.1).
`219` (resolved): build_model_description(default_step_size=<NumPy scalar>)
wrote `stepSize="np.float64(0.05)"`: a modelDescription.xml no FMI importer
reads, with nothing said at export.
`221` (resolved; carried by no release): the Arnoldi breakdown test was 1e-5
of the product in every dtype, so `rho_spectral` read 0.379 for 0.5 and
`spectral_error_bound` 0.61x the distance with `spectral_usable=True`.
`223` (resolved; carried by no release): the compressed Jacobian's radius was
taken by repeated squaring, which in float32 read 0.0512 for 0.0500.
`224` (resolved; carried by no release): `spectral_usable` tested only the
Arnoldi residual, and a non-normal group read 0.941 for 0.735 with it set.
`222` is a loop below a float32 field's rounding missing from `rho_spectral` on a sub-cycled group (the interpolated boundary value's tangent is now exact; values unchanged);
`225` is a non-normal group past eight interface scalars reading settled outside the margin (a Krylov space still growing at the cap is never settled);
`226` is `gradient_relative_error_bound` reading below the error at the float floor (the undirected part of the distance now takes the operator norm);
`229` is a spectrum settled from sampled rounding outside its margin (float32, rank six: 0.250 for 0.206; a certificate over every perturbation of the measured size replaces the samples).
`227` (resolved; carried by 0.1.0 to 0.3.1): a multi-rate graph whose timesteps
are more than about 1e9 apart, or have no common step the float GCD finds, ran
its nodes on different clocks; now a `ValueError` at `compile()` and a 400 at
`POST /graph/nodes`.
`228` (resolved; carried by 0.1.0 to 0.3.1): REST read a boolean or numeric text
as a node's timestep and clamped `/sim/profile`'s counts silently; now a 422.
`231` (never released): a coupling report was measured on the live state and
moved when the state was written after the step.
`233` (resolved; carried by 0.3.0 and 0.3.1): a coupling group under
`acceleration="iqn-ils"` or `"iqn-imvj"` whose iterate left float range before
its cap never returned from `step()`; it now runs to the cap and reports
`converged=False`, `residual=inf`.
`234` (resolved, never released): `gradient_relative_error_bound` read 1.09 for
a relative error of 1.134 with a constant in it whose tangent is rounding; a
constant one pass does not resolve is no longer in the bound.
`235` (resolved): On a multi-rate graph a slow node's flux was computed from a
state the step discarded: between the node's firings its readers received the
flux of the next state, not of the one held.
`236` (resolved): A checkpoint of a sparse mapped edge loaded into the dense
edge of the same key, and the reverse, returned normally and installed none of
the saved weights.
`237` (resolved): A geometry written as float32 after compile() bypassed the
resolution refusal: on a grid float32 cannot resolve, step() ran and every
sample was 0.0.
`238` (resolved): A coupling group whose member holds a floating field with
no entries failed to trace under solver='ift' with diagnostics=True (and
under a quasi-Newton acceleration whose interface fields have none); it
stepped in 0.3.1.
`240` (resolved; 0.1.0 to 0.3.1): under `convergence_norm="interface"` a converged group returned a field no internal edge reads computed from readings that had not settled, a one-way pair under Jacobi its target at the pre-step source; every field the norm does not measure whole is now returned as one plain pass computes it at the accepted iterate.
`241` (resolved; 0.1.0 to 0.3.1): under `convergence_norm="interface"` the part of a field its edges deliver only through a mapping or a transform was a pass behind, or a blend of passes under a relaxation, with `converged=True`; the same return rule covers it.
`242` (resolved): spectral_error_bound of a group with a multilinear_grid edge
read 0.13x the true distance with spectral_usable=True when a marker's fixed
point lay across a lattice plane from the returned iterate.
`243` (resolved): A marker exactly on the last lattice point of an axis failed
the geometry self-check on an honest pass: every bound of the group withheld,
with a reason blaming a member's derivative.
`244` (resolved): Float32 fields read at float64 positions always failed the
geometry self-check: every bound of the group withheld, with a reason blaming a
member's derivative.
`245` (resolved): A mapping of a registered kind short of a Mapping protocol
member, or carrying another kind, was accepted by add_edge and written by
to_dict(), and from_dict() then refused that config.
`248` (resolved): A geometry written with another shape after compile() was
broadcast by a mapping kind that does not check its own shape.
`249` (resolved): The underflow-range warning was asked once per compile, of the
first stepped state alone: a state written into the range later, or already
below tiny, stepped unwarned.
`250` (resolved, since 0.1.0): A state field written with another shape after
compile() was broadcast into the node that reads it.
`251` (resolved, since 0.1.0): A node's delta_t written after add_node was
neither followed nor reported, and to_dict() wrote it.
`252` (never released): `spectral_usable` and `gradient_bound_usable` stood on bounds far under the distance and the error for a `multilinear_grid` group near a lattice plane; a group that solves the mapping's positions now has neither flag (its numbers are reported, uncertified), and the flags are available where every position is fixed during the pass.
`253` (0.1.0 to 0.3.1): `POST /graph/edges` and `POST /graph/nodes` dropped a key they do not apply (`additive`, `mapping`, a misspelt `params`) with a 201; now a 422 naming the key.
`255` is `MADD-ANO-255` (the float floor of a group of mixed dtypes took each
field at its own eps; never released)
`256` is `MADD-ANO-256` (the dead band of an interface edge read at its source
was asked of the source field alone; never released)
`261` is `MADD-ANO-261` (under `"l2"` and `"mixed"` the float floor did not
count the rounding of the weights a geometry-dependent mapping forms; never
released)
`263` is `MADD-ANO-263` (on the loop backend, which every preconditioned solve
uses, `sharded_cg` reported the residual its iteration updates and `sharded_gmres`
the preconditioned one: in float32 `converged=True` and a `residual_norm` under
`rtol` on solves whose `||b - A x||` was 10 to 100 times the tolerance; in 0.3.0
and 0.3.1.  Both fields now come from one product on the returned value, so such a
solve reads `converged=False` with its true residual: at the default `rtol=1e-6`
in float32, any system with a condition number over about 17.  Ask for an `rtol` a
few times above `eps * kappa`, or solve in float64)
`265` is `MADD-ANO-265` (0.1.0 to 0.3.1: an edge onto a field that is also a
declared external input never reached the node, since the input replaced it,
with zeros when it was not fed, and nothing said so; `compile()` now raises a
`ValueError` naming both and `validate()` lists it)
`266` (resolved): A derivative taken through sharded_cg or sharded_gmres called
without differentiable=True was silently wrong when the tangent or cotangent
was not of the right-hand side's size; such a call now refuses to be
differentiated.
`267` (resolved): sharded_cg(differentiable=True) given an x0 started its
tangent and cotangent solves from that x0: with max_iters=0 the derivative was
x0 itself, and at |b| of 1e5 a unit-direction JVP was off 2.6e-2 (never
released).
`268` (resolved): With differentiable=True an atol was applied to the tangent
and cotangent solves of sharded_cg and sharded_gmres as well: a derivative was
not linear in its direction, and zero for a direction under the floor (never
released).
`269` (resolved): sharded_cg's default call and backend='lineax' answered an
operator that returns NaN or infinity with zeros, converged=True and iters=0.
`273` (resolved): the FMU description wrote unit on integer and Boolean variables,
which FMI 3.0's schema refuses (in 0.3.0 and 0.3.1, on a hand-built variable).
`274` (resolved): a 422 sent a refused request body back whole, encoded on the
event loop (since 0.1.0).
`276` (resolved): A coupling group whose boundary fluxes read each other in a
loop converged, converged=True, to a state that does not solve the coupled
equations; such a group is refused at compile() (never released).
`278` (resolved, since 0.1.0): An assignment to node.params on a sharded
wrapper or a HybridNode replaced the wrapper's mapping alone: the step ran the
old value while node.params read the new one back.
`279` (resolved, never released): A deep copy of a graph put a node's value back
over a constant calibrated in gm.params after a node write: the copy ran, reported
and saved another value than the original.
`271` (resolved): sharded_cg and sharded_gmres did not read rtol, nor an atol
that is traced: a negative or NaN one read converged=True on the lineax routes
beside a value 72 % off, or zeros; in 0.3.0 and 0.3.1.  Given as a number it is
now refused with a ValueError, and given as a traced value answered with NaN and
converged=False.
`272` (resolved): a solution too large for the dtype came back as infinities
beside a finite residual_norm, with converged=True where the rescaled solve met
its tolerance (never released; in 0.3.0 and 0.3.1 the CG loop answered a
right-hand side whose norm overflows with zeros and converged=True).  It is now
answered with NaN and converged=False.
`281` (resolved): with atol=0.0, or with rtol=0, the lineax routes of sharded_cg
and sharded_gmres read converged=True after one step at a residual of 0.4 ||b||
(lineax's entrywise test is 0 / 0 at an entry that is exactly zero); in 0.3.0
and 0.3.1 for atol=0.0.  converged is there now
residual_norm <= max(atol, rtol * ||b||).
`283` (resolved): ift_linear_solve with solver="cg" or "gmres"
returned a value 0.36 to 1.0 off with no error where lineax's stopping floor
was zero (atol=0.0) or NaN, in 0.3.1; it now raises there.
`284` (resolved): an integer or Boolean field carried by an edge inside a coupling
group stopped the solve early with `converged=True` (never released).
`286` (resolved): 0.1.0 to 0.3.1: WebRTC streaming never negotiated, so a
viewer was sent no answer and no ICE and never received video; the session now
answers the viewer's offer and is tested end to end on CPU.
`287` (resolved): 0.1.0 to 0.3.1: the cloud container streamed nothing (its
entry point never started the WebRTC session) and its API could add no node;
the entry point now registers the built-in nodes and streams the simulation.
`288` (resolved): 0.1.0 to 0.3.1: the cloud entry point logged
MADDENING_CLOUD_CONFIG verbatim at INFO, so a TURN credential in
stream_config.ice_servers reached the log; credentials are now redacted.
`289` (resolved): fim and fim_core read the rank from the eigenvalues of
F = J^T J, whose rounding a constant or piecewise constant sensitivity aligns:
two parameters that enter a residual only through their sum could report rank
2 of 2 and a finite Cramer-Rao bound, with no warning (never released).

## The account of each area

The sections from here to "API stability" are the release notes' account of
each area of the release, moved. They hold the design of each feature as it was
built, what each audit round found in it and how each finding was fixed.

## Calibratable graphs

### The parameter pytree

`GraphManager.params` is `{"nodes": {name: node.params_pytree()},
"mappings": {}}`, snapshotted at compile time and passed on every `step`,
`run`, `run_scan`, `run_scan_with_history`, `run_sweep` and adaptive run
(each takes an optional `params=`).  Because the constants are traced,
`jax.grad` / `jax.jvp` / `jacfwd` reach them — including through coupling
groups, where `closure_convert` hoists them into the IFT rule — and a
changed value takes effect without a recompile.

A node opts in by declaring `update(..., *, params=None)` and reading its
constants from `params`.  Nodes still on the three-argument contract keep
working unchanged; `gm.nodes_without_params()` lists them and `compile()`
logs them.  Migrated in this release:

| Node | Injected constants |
|---|---|
| `SpringDamperNode`, `BallNode`, `HeatNode`, `RigidBodyNode` | first wave |
| `RigidBody2DNode` | mass, inertia, gravity |
| `HeartPumpNode` | the six Windkessel constants, `systole_fraction` on a logit spec |
| `TableNode`, `HealthCheckNode` | on the contract with no dynamics constant to inject (`checks`, surface height stay structural) |
| `LBMNode` | `viscosity`, so `tau` is a traced constant |
| `LBMPipeNode` | `tau`, `tau_tracer`, `propeller_strength`, `gravity`, and the Shan-Chen constants `G`, `rho_0`, `rho_wall`, `rho_liquid`, `rho_gas` (trainable only when the node was built multiphase, since `G != 0` selects the branch) |
| `SurrogateNode` | every floating leaf of the network weights, as flat `"weights<path>"` entries rebuilt inside `update` — so `jax.grad` of a trajectory loss reaches the surrogate weights and fine-tuned weights need no recompile |

`SimulationNode.params_pytree()` defaults to the float-valued entries of
`self.params`; structural values (`n_cells`, shapes) stay on the recompile
path.  Passing a pytree that names a node whose `update` takes no `params`,
an unknown node, or an unknown key is now a `ValueError` at trace time —
previously it was silently ignored, and a gradient with respect to it was
silently zero.

Flux edges take part too: `compute_boundary_fluxes` gained the same
keyword-only `params`, and the graph passes the node's entry on every flux
evaluation (Gauss-Seidel, Jacobi, uncoupled and IFT paths), so a calibrated
stiffness or diffusivity changes the force or heat flux an edge *delivers*.
The sharded wrappers take part when the wrapped node's `update_padded`
accepts `params`: `ShardedStencilNode` and `ShardedUnstructuredNode` report
`accepts_params()` / `params_pytree()` / `param_specs()` from the inner
node, and the node's entry is replicated to every shard.  Previously the
sharded path silently ignored `gm.params`.

The solver paths take part as well.  `SimulationNode.derivatives` and
`implicit_residual` take the same keyword-only `params`, every in-tree
override reads the merged dict, and `integrate_node(..., params=)` and
`implicit_euler_step(..., params=)` forward it — so a stiffness fitted
through `gm.params` drives an RK4 or backward-Euler step exactly as it
drives `update`.  It did not at first: these two methods had no `params`
at all, so the two paths of one node object disagreed by the whole
calibration (−1.0 against −4.0 on a spring calibrated from 100 to 400),
registered as MADD-ANO-018 and resolved within this release.  A non-empty
`params` for an override that has no `params` keyword is now a
`ValueError` naming the class and the method, never a silent fall back to
the constructor's constants; passing nothing calls the old two-argument
shape, so an override that predates the keyword keeps working.  The bare
steppers `euler_step` / `heun_step` / `rk4_step` remain node-agnostic and
take a callable — `functools.partial(node.derivatives, params=p)` is what
`integrate_node` builds — and `accepts_params(method="derivatives")` is
the probe the integrators use.

The interface correction is verified on the same footing, and it needed
to be.  `HeatNode.compute_interface_correction(params=)` took `params`
from the start of this release but built its Laplacian without the
injected `length`, so `dx` was the constructor's while `update`,
`derivatives`, `implicit_residual` and `compute_boundary_fluxes` used the
calibrated one.  Through the coupling system that put the two interface
cells of a pair of Dirichlet-coupled rods **11.39 K** from where rods
built with the calibrated length put them, after two steps, with every
interior cell agreeing and no error anywhere.  The release audit found
it; it is fixed, pinned at graph level for both the `ift` and `fori`
solvers, and — because the surface is 0.4.0-new and never shipped in a
tagged release — carries no anomaly entry.  The audit's per-path table,
every trainable leaf of `SpringDamperNode`, `BallNode`, `HeatNode`
(uniform 2nd- and 4th-order, non-uniform), `RigidBodyNode` and
`HeartPumpNode` against `update`, `derivatives`, `integrate_node`
(euler, rk4), `implicit_euler_step`, `compute_boundary_fluxes` and
`compute_interface_correction`, is 125 rows and now agrees with a node
rebuilt with the calibrated value on every one of them.

Two refinements to the refusal itself, from the same audit.  "Takes
`params`" now means an explicit keyword *or* a `**kwargs` that would
forward it — `ShardedStencilNode` already read `update_padded` that way,
`accepts_params` did not, and the two probes disagreed on the same
spelling — and the rule is one helper that `accepts_params`, the
integrators, the implicit solver and the verification battery all share.
And the refusal now holds one method deep: a subclass with an old-style
`derivatives(self, state, boundary_inputs)` under an inherited
`implicit_residual(..., params=)` used to end in Python's raw `TypeError`
from the inner call; every in-tree residual now reaches `derivatives`
through the shared binder, so it is the same `ValueError` naming the
class and the method on every route.  A residual decorated without
`functools.wraps` no longer crashes the solver with `AttributeError`
when params are given.

The graph's own coupling hooks now read that rule too.  The probes that
decide whether `compute_interface_correction` and `compute_boundary_fluxes`
are passed `params`, and `HybridNode`'s forwarding to its physics node,
still accepted only the explicit keyword, so a hook that forwarded
`**kwargs` to `super()` was called without `params` and ran on the
constructor's constants while `update` used the calibrated ones — the same
**11.39 K** at the interface cells of the rods above, and a spring
calibrated from 30 to 300 delivering −23.1 on its flux edge instead of
−285.0 — while the verification battery, which already read the shared
rule, called the node compliant.  All three now read the one helper; a
hook with neither spelling is still called with three arguments, as before.

`TestParameterRecovery` now recovers `k, c` through the real graph (a single
spring and a coupled group), with the float32 gradient matching a float64
finite difference to **2.4e-6 relative over 100 steps**.

### `ParamSpec` and the optimiser surface

`maddening.core.params.ParamSpec` carries `trainable`, `bounds` and
`transform` (`"log"` for strictly positive constants, `"logit"` for
intervals, `None` for clipping).  Nodes declare specs for their constants in
`SimulationNode.param_specs()` — `initial_*` entries default to
`trainable=False` — and `GraphManager.set_param_spec(node, key, spec)`
overrides per graph.  `gm.trainable_mask()`, `gm.unconstrain()` /
`gm.constrain(u)` and `gm.check_params()` are the pytree maps an optimiser
needs.  The transforms clamp to the representable float32 interior, so a
saturated step (`exp(89)`, `sigmoid(17)`) stays finite and invertible.

### System identification

`maddening.sysid` is new:

* `fit(gm, loss_fn, ...)` — Adam in the unconstrained coordinates under the
  trainable mask, returning physical params inside their bounds with frozen
  leaves bit-identical; raises on a non-finite gradient.
* `windowed_loss` — teacher-forced windowed trajectory loss, optionally
  masking windows where a coupling group exited unconverged.  Every
  observation leaf must have the same length, and leaves that disagree
  are refused by name: a window reads each leaf with a clamping
  `dynamic_slice`, so a short leaf would have been compared against its
  own last sample repeated (a loss of 0.32 at the true parameters in a
  two-spring case, with its minimum on the edge of the scan).
* `fim` — Fisher information `JᵀJ` from `jacfwd` sensitivities, with
  relative scaling, eigen-decomposition and Cramér–Rao bounds.  `mask=`
  restricts it to the leaves a mask selects, and `noise_std=` (scalar or
  per-leaf σ) weights the residual so `crb` comes out in the parameters'
  own units.  The identifiability check correctly isolates the (k, c, m)
  common-scale direction on a spring observed through position only.
* `fit_multiple_shooting` — `windowed_loss(..., window_states=,
  continuity_weight=)` restarts each window from a free state and ties
  consecutive windows with a continuity penalty; `init_window_states` seeds
  them from the observations, so noisy window starts no longer seed every
  window with measurement error.
* `fit_lm` — Levenberg–Marquardt over the same `jacfwd` sensitivities,
  recovering a spring's (k, c) from 2× perturbations in a handful of
  iterations where Adam needs hundreds.
* Progress events — all three fitters notify the graph's observers with a
  `"fit_progress"` event (`EVENT_FIT_PROGRESS`: method, iteration, loss,
  params) every `notify_every` iterations, so the REST relay and the live
  stage can show a calibration as it runs.

The user guide page is {doc}`/user_guide/parameters`.

#### Degenerate inputs are refused rather than reported

An independent audit of this surface
(`benchmarks/results/audit_040_final/params-io/`) found that the parts of
`sysid` that exist to tell you *not* to trust a number were the parts that
failed quietly.  All of them now raise, because each one used to produce a
plausible-looking answer and no other symptom:

* **The `+inf` fail-safe now holds.**  `FIMReport` documents `crb = +inf`
  "because it fails safe: `crb < threshold` is then False for an
  unidentifiable parameter".  With a NaN Fisher matrix the guard inverted:
  the resolved-subspace sum over an empty subspace is `0.0`, `support` is
  NaN, and `NaN > n * eps` is False, so the report came back
  `rank=0, cond=nan, crb=[0., 0.]` — the *most* trustworthy bound it can
  carry — for a matrix holding no information at all.  A bound is now
  reported finite only where it was positively established.
* **`noise_std` is validated** — finite and strictly positive *after* the
  cast to the residual's dtype, which is where `1e-320` becomes `0.0`.  A
  negative σ used to give exactly the answer its absolute value gives, so a
  sign error in a caller's noise model had no symptom at all.
* **`fim` raises `FloatingPointError` on a non-finite `F`**, the way `fit`
  already raises on a non-finite gradient.
* **A `mask` must have the same tree *structure* as `params`**, not merely
  the same leaf count.  The flags are read in flatten order, so a mask keyed
  by the caller's own symbol names was accepted and landed elsewhere: the
  audit marked `damping` and watched the fit move `stiffness` from 30.0 to
  76.3.  The error names the first path that differs.
* **`fim(scale="relative")`, the default, now says when it scaled a zero.**
  Relative scaling multiplies each Jacobian column by the parameter's value,
  so a parameter sitting at exactly `0.0` — `SpringDamperNode`'s
  `initial_velocity` default — drops out of `F` however well the data
  determine it.  `FIMReport.zero_scaled` names those parameters, so the
  `+inf` can be read as the statement about the scaling that it is; ask
  `scale=None` for the absolute question.
* **`fim(scale="nominal", specs=...)` fixes what `zero_scaled` reports.**  The
  column scale comes from the parameter's `ParamSpec` — the width `hi - lo` of
  a finite `bounds` — rather than from its value, so a parameter at `0.0`
  keeps a dimensionless column and `rank` / `crb` answer for the data.  A
  width, not a midpoint: the midpoint of `(-1, 1)` is the zero being removed.
  A spec with no finite width has no nominal scale; that column keeps the
  value scaling (`p`, or `p - lo` for a `"log"` spec, the transform's own
  gain) and `FIMReport.value_scaled` names it, so a nominal report is never
  quietly half relative — a value-scaled zero still appears in `zero_scaled`.
  Nothing falls back to `1.0`, and a width that is zero or `inf` once squared
  at the parameters' precision (`(0.0, 1e-300)`, `(0.0, 1e20)` in float32) is
  refused by name.  `specs` is static — reduced to per-column constants on the
  host, as `mask` is to indices — so `fim_core` traces as before and `fim`'s
  transfer count stays at four.  `"relative"` and `None` are untouched.
  `specs` must mirror `params`: a key that matches no parameter, a dict where
  a leaf needs a `ParamSpec` (`to_dict()` output), a `ParamSpec` above a dict
  level or a non-dict `specs` is a `ValueError` naming the key path, since a
  spec that reaches nothing would be `"relative"` under a `"nominal"` label.
  A spec keyed for a list/tuple-valued leaf now reaches it in `fim`,
  `trainable_mask`, `unconstrain` / `constrain` and `check_bounds` alike
  (`check_bounds` used to pass an out-of-range list leaf).
* **`fim` answers for the residual as it stands at the call.**  During
  0.4.0 development `fim` kept its compiled Jacobian cached on
  `residual_fn`, and tracing bakes in everything a residual reads from
  outside its argument.  The natural residual — a few leaves as arguments,
  the rest read from `gm.params` — therefore reported the first call's
  matrix after those leaves changed: once `damping` moved from 2 to 20,
  the (relative) Cramér–Rao bound on `mass` read 0.41 against a true 44.7,
  109x too tight, while `stiffness` read 0.362 against a true 0.238; and a
  bound method, which compares equal across attribute accesses, read
  rank 2 against a true 1.
  `fim` re-traces on every call by default (~230 ms for a 200-step rollout
  on four pinned CPU cores); `fim(..., reuse_trace=True)` keeps the
  cache (~1 ms warm) for a residual you assert depends on nothing that can
  change between calls.  `fim_core` under your own `jax.jit` has the same
  contract, now stated.
* **One walk decides which `ParamSpec` governs a leaf, for every reader.**
  `fim`'s specs check and the resolver behind `trainable_mask`,
  `constrain` and `check_bounds` were separate walks and disagreed on a
  namedtuple level: the check accepted a list of specs, the resolver read
  nothing, and `check_bounds` passed `k=50` against bounds `(-5, 5)`.  They
  share one step now.  A namedtuple or dataclass level is addressed by
  field name (a dict), as JAX addresses it; the tree maps refuse an entry
  on a leaf's path they cannot read instead of handing the leaf the
  default spec; a key that matches no leaf is still ignored there, because
  `gm.param_specs()` declares specs for constants `params_pytree()` leaves
  out.  `fim(scale="nominal")` refuses such a stray key only when it could
  have changed a column (a finite width, or a `"log"` offset), which is
  what lets `specs=gm.param_specs()` through for a uniform `HeatNode` graph
  and for a node built with an `int` constant — both used to be refused.
* **An integer or boolean params leaf is named, not a zero column.**  With
  `mask=None`, `fim` / `fim_core` leave it out and name it in
  `FIMReport.integer_excluded` (an integer `matrix_mapping` read as rank 0
  of 16); a `mask` selecting one is refused, and so is a fit whose
  trainable set holds one, which used to hand it back unchanged.
* **`ParamSpec` refuses a `NaN` bound** (every comparison with it is False,
  so `check_bounds` passed anything), an infinity pointing inward, and an
  infinite bound under a transform.  `(-inf, inf)` stays accepted and means
  exactly what `None` means, `fim`'s nominal scale included.  N-D leaves are
  labelled by element index (`['H'][0, 5]`, not the flat `['H'][5]`).
* **Hyper-parameters with no reading are rejected**: `lr <= 0` (silent
  gradient *ascent*), `eps <= 0`, `tol`/`step_tol` NaN or negative, `betas`
  outside `[0, 1)`, `lam0 <= 0`, `lam_up <= 1` (damping that *shrinks* on a
  rejected step), `lam_down > 1`, negative counts.
* **A fit that takes no step returns its input, bit for bit.**  `FitResult`
  promises that comparing input and output leaf by leaf says exactly which
  constants a calibration touched; `fit(n_iter=0)` and a `tol` that stopped
  before the first update still returned `30.000002` for a `log`-transformed
  `stiffness` of `30.0`, because the masked leaves round-tripped through
  `exp(log(p))`.  The test is now on the coordinates: a leaf whose
  unconstrained entries did not change comes back from the input untouched.
* **A `log`/`logit` `ParamSpec` refuses a non-floating leaf.**
  `to_constrained`'s identity branch preserves an integer leaf's dtype and
  the transform branches cannot, so `constrain(unconstrain(p))` silently
  returned a float where `core.params` promises the round trip.

#### Parameter precision follows `jax_enable_x64`

`SimulationNode.params_pytree()` pinned Python floats and lists of floats to
float32 outright — and `numpy.float64` is a subclass of `float`, so it took
that branch too.  Under `jax_enable_x64` a Python float, an `np.float64`
scalar and a list of floats therefore came back float32 (a 1.7e-8 relative
shift, no warning) while *the same value* written as a 0-d array, a 1-d array
or a `jnp` float64 array came back float64: one value, two dtypes, two
answers.  The `AdaptiveNode` audit hit the same code from the other side — a
node solving in float64 whose parameters, gradients and
`check_gradient_capture()` were float32, so the diagnostic evaluated at a
float32-rounded point the state had not been built at.

A value that carries a floating dtype of its own (an array, a numpy scalar)
now keeps it; a value that carries none (a Python float, a list) is placed at
JAX's canonical float precision, resolved the way `AdaptiveNode.__init__`
already resolves it (`jnp.zeros(()).dtype`).  All six spellings of one number
now agree, and nothing is narrowed below the precision the graph is working
in.  **Without `jax_enable_x64` nothing changes** — float32 is the canonical
dtype, so every spelling lands exactly where it did before.  With it on, node
parameters become float64, which is what asking for x64 meant.

This is deliberately not a warning.  Under the default policy float32 *is* the
working precision, and a `np.zeros(n)` default is float64, so warning on every
narrowing to the working precision would fire for most parameters of most
models.  Where a user does state a precision JAX cannot honour — an explicit
`dtype=jnp.float64` with x64 off — JAX itself already warns.

#### When a rank verdict is decided at the noise floor, `fim` says so

The other side of the same coin.  `FIMReport.rank` counts the eigenvalues
above `rank_rtol * max(eigvals)` — `n * eps` when this was written, and see
[the next section](#the-rank-cutoff-now-sees-the-residual-length) for the term
it has gained since — and in float32 that comparison can be between
two numbers that differ by less than the decomposition resolves.  The case
this came from reported `rank=1`, `crb=[inf, inf]` and a condition number, as
fact, off an eigenvalue ratio of **2.086e-07 against a 2.384e-07 cutoff** — a
12% margin, against a decomposition whose own error is the same size as the
cutoff.

`fim` now raises `PrecisionLimitWarning` when the eigenvalue ratio nearest the
cutoff falls within a factor of 2 of it, naming the ratio, the cutoff, the
factor, and the remedy: re-run under `jax_enable_x64`, which drops the cutoff
to `n * 2.22e-16` and settles the question.  (Already under x64 it says so
instead, because there is no wider precision to escalate to.)

The factor is measured, not chosen.  Over ~500k synthetic Fisher matrices of
known spectrum, put through `fim`'s own arithmetic — `J.T @ J` and `eigh` in
float32 — against a float64 reference applying the *same* rank rule, the two
precisions disagree only for true eigenvalue ratios in `[0.46x, 2.15x]` of the
cutoff and never above `2.15x`.  The threshold sits at that edge rather than
beyond it because margin is expensive: widening to 3x fires on 45% of ordinary
`2x..5x` verdicts float64 agrees with, and to 8x on two thirds of `5x..10x`
ones, for four to ten points of extra coverage.  A warning that fires
routinely gets suppressed, which is worse than silence.  The cost accepted
here was the ~14% of precision-limited verdicts that stayed quiet, nearly all
of them long-residual cases where the error in forming `J.T @ J` grows with the
number of residual rows and an `n * eps` cutoff cannot see it — **and that cost
is not accepted any more**; the next section removes it, and re-measures this
factor against the cutoff that does.

It does not fire on an exactly singular `F` (the null eigenvalue comes back at
or below zero — a real rank deficiency, and `zero_scaled` already explains the
common cause), nor on a close call against a `rank_rtol` the caller raised,
which is a modelling threshold and not a claim about arithmetic.

**x64 is not becoming the default.**  It is process-global and set before the
first JAX import, so it changes every library in the process, and fp64
measures **91x slower** than fp32 on this project's reference RTX A2000 (155
against 14,135 GFLOP/s — the fp32 figure is TF32 tensor cores, so treat it as
the ratio a user feels rather than a hardware constant).  Making the
precision-limited cases identify themselves costs nothing; making everything
float64 costs that.

#### The rank cutoff now sees the residual length

The section above flagged a defect it did not fix: `rank_rtol` defaulted to
`n * eps`, which estimates `eigh`'s error and nothing else, while forming
`F = J.T @ J` costs again and that cost grows with the number of residual
rows `m` — which `F` cannot show.  **`rank_rtol` now defaults to
`max(n, sqrt(m)) * eps`.**

How far below its own floor the old cutoff sat is measurable directly.  Build
an `F` that is *exactly* rank-deficient in float64, so the true answer is zero
and whatever float32 reports for that direction is its own rounding, and read
the floor off in units of `eps` (p99 over 400 draws per cell, worst over
spread / clustered / twin-null spectra):

| n \ m | 20 | 50 | 320 | 800 | 2000 | 4000 |
|---|---|---|---|---|---|---|
| **2** | 0.75 | 1.13 | 2.63 | 3.88 | 7.13 | 9.13 |
| **3** | 1.75 | 1.75 | 3.00 | 4.50 | 6.38 | 8.74 |
| **5** | 2.50 | 2.50 | 2.50 | 2.38 | 2.50 | 2.50 |
| **12** | 4.50 | 5.00 | 4.01 | 4.50 | 4.00 | 3.53 |
| **25** | — | 5.72 | 6.58 | 6.00 | 6.50 | 6.01 |

A factor of 200 in `m` moves the `n = 2` floor by 12x — `sqrt(200) = 14`, not
`200` — and the `m` term only overtakes the `n` term when `n` is small, which
is why the new cutoff is a `max` and not a sum.  At `n = 2, m = 4000` the old
cutoff was `2 eps` against a floor of `9 eps`: **below its own noise floor**,
where `rank` counts a direction that is not in the data and `crb` reports a
finite variance bound for parameters no estimator can determine.  Over 64
exactly-rank-deficient draws at that shape, 15% came back full rank with a
finite `crb`.  A false *identifiable* — the direction `FIMReport` fails safe
against everywhere else.

**The briefed fix was `max(n, m) * eps`, and that is wrong.**  It is the
worst-case summation bound taken literally: it assumes every rounding in an
`m`-term inner product aligns, which neither XLA's blocked accumulation nor a
real Jacobian does.  Measured against the table above it sits up to **1315x**
above the observed floor, and over the sweep it rejects **45%** of the
directions float32 and float64 *agree* are resolved, up to a true eigenvalue
ratio of `2.4e-04` — four decades of real resolution thrown away.  `sqrt(m)`
is both what the measurement shows and the textbook statistical model for an
accumulated rounding error over `m` terms.

**What moves, and for whom.**  Because the new form is a `max` it can only
widen, so no cutoff moves down and nothing that was called unresolved becomes
resolved.  Every problem with `m <= n**2` is unaffected *exactly*, bit for
bit — which is most fits, including the two- and three-parameter spring-damper
examples in the user guide.  Above that, `rank` can fall by one and the `crb`
entries for the parameters that direction spoils go `+inf`; `cond` is
unchanged, since it never used the cutoff.  To keep the old verdicts, pass
`rank_rtol=n * eps` explicitly — but read it as what it is, a statement that
you want the cutoff below the arithmetic's noise floor.

No released version is affected: `maddening.sysid` is new in 0.4.0, so the
`n * eps` cutoff never shipped.  This is a change against the 0.4.0 development
line, recorded under **Changed** because anyone tracking it will see rank
verdicts move.

**The warning factor was re-measured against the new cutoff, and stayed at
2.0.**  It had to be re-measured: the band is the arithmetic's error expressed
*in units of the cutoff*, so moving one moves the other.  Over 228,000
synthetic matrices in three independently seeded sweeps, the disagreement band
narrows from `[0.46x, 2.15x]` to `[0.46x, 1.47x]` and the population of
precision-limited verdicts falls from ~1.2% of draws to 0.35%.  What changed is
not the number but what it buys: at the old cutoff, factor 2.0 had a recall of
**0.912–0.916**, and the ~9% it missed were exactly the long-residual verdicts that
no factor could reach, because the cutoff was below the floor there and the
disagreements were nowhere near it.  At the new cutoff its recall is **1.000**,
for a false-fire rate of ~1% on ordinary `2x..5x` verdicts — *lower* than the
2.8% it cost before.  Full recall first arrives at 1.8, but 1.8 is fitted to
the single most extreme of 803 disagreements with no margin left; 2.0 keeps 13%
of margin and is still on the flat part of the cost curve, which turns sharply
at 2.5 (a quarter of ordinary `2x..5x` verdicts).

The `TestPrecisionLimitedRank` generators capped `m` at 32 for exactly the
reason this section removes.  The cap is gone, to 1024, and the calibration
gate's recall floor tightened from 0.75 to 0.90.

#### `fit` stops wandering inside the flat directions

`fit_lm` was believed to be unable to move along an exactly-null direction:
every gradient of a least-squares loss is `Jᵀr`, so a direction `v` with
`Jv = 0` has `g·v = 0` at every iterate, and there is nothing along `v` for
the solve to act on.  `fit` has no such protection.  Adam's step is not the
gradient but `−lr·D g` for a diagonal `D`, and `(D g)·v` is generally nonzero
where `g·v` is exactly zero, so Adam wanders inside the flat manifold.  The
loss does not notice — it is flat there — but the *parameters* do, and they
are physical quantities a caller reads, records as provenance and
extrapolates from.

Measured on the spring's `(k, c, m)` common-scale degeneracy, the one
`FIMReport.least_identifiable` already names, with the geometric mean of the
three as the coordinate along it.  On a 60-step trajectory loss with σ = 0.02
noise, starting from `(45, 3, 1)`:

| `lr` | 200 iters | 2,000 iters | 10,000 iters | loss at 10,000 |
| ---- | --------- | ----------- | ------------ | -------------- |
| 0.01 | +1.82% | +1.82% | +1.85% | 1.334495e-02 |
| 0.05 | +3.95% | +3.95% | +5.98% | 1.334495e-02 |
| 0.2  | +6.85% | +9.75% | **+46.06%** | 1.334494e-02 |

The loss is identical in its first six digits down every column; `mass` comes
back as 1.39 or as 1.90 depending only on the iteration budget.  Two runs of
the same fit on the same data, differing only in `n_iter`, return different
physical constants and both call themselves equally good.  That is the defect:
not a wrong optimum, an unreproducible one.

**`fit` now holds those directions at the values it was given.**  It
accumulates `G = Σₜ gₜ gₜᵀ` over the run — the directions the gradients ever
pointed along — and removes the net displacement's component along `G`'s
numerically-null eigenvectors.  Pooling over the run is what makes it robust:
near convergence a single gradient is mostly rounding noise, but the
*accumulated* energy along an exactly-null direction stays at the arithmetic's
floor.  Over 200 to 10,000 iterations and learning rates 0.01 to 0.2 the null
eigenvalue of `G` measures 1e-15 of the largest, against 8e-3 for the weakest
direction the data does resolve: twelve decades, and the cutoff
(`(max(n, √T)·eps)²`, `fim`'s own rank rule moved one level out) sits between
them with three decades of margin on the near side.

What a caller sees: the loss, the iterates, `losses`, `callback` and the
`fit_progress` events are all unchanged, and **a fit whose gradients spanned
everything gets its iterate back bit for bit** — a well-posed fit sees no
difference at all.  `FitResult` gains `excited_rank` (how many independent
directions the data determines -- first counted as the directions the run's
gradients spanned, which over-counted the undetermined ones; see "The guard
holds only what the curvature confirms" below) and `undetermined_drift` (how far the
raw iterate had wandered along the rest, in unconstrained coordinates).
`hold_undetermined=False` returns the selected iterate unguarded (until the
lowest-loss selection below, that was the previous behaviour exactly).

**`fit_lm` and `fit_multiple_shooting` run the same guard.**  Shipping a field
that is `None` for two of three fitters would have baked the asymmetry into
0.4.0, and neither is immune.  The gradient is orthogonal to the null space,
but no step rule here *is* the gradient: Marquardt damping solves
`(A + λ·diag(A))⁻¹g`, orthogonal to `null(A)` only where `diag(A)` is
isotropic there (`A = [[1, 2], [2, 4]]`, `g = (1, 2)` steps along `(2, 1)` for
every `λ`, against `null(A) = span(2, −1)`).

They are not the same defect, and the difference is worth stating.  `fit_lm`'s
step vanishes with the gradient, so its drift converges: on the spring's
`(k, c, m)` scale it settles at −1.78% (noiseless) or −0.66% (σ = 0.02) and
is then identical to the last bit for `n_iter` 10 through 200.  That is a
*consistency* fix — the value is reproducible, it is simply neither yours nor
one the data chose.  `fit_multiple_shooting` is Adam, and is `fit`'s defect:
−4.35% at `lr=0.01, n_iter=200`, −4.80% at 1,200, −1.97% at `lr=0.2, n_iter=200`,
−2.02% at 1,200 and −2.31% by 4,000 — a 3.0% spread in the returned constants
for a loss that agrees to four digits.  Every one of those losses is unchanged
bit for bit by the guard.  Only the parameters are held: the returned
`window_states` are that fit's own decision variables.

Two limits, both fail-open and both reported as a full `excited_rank` rather
than silently:

* **The cutoff is numerical, not statistical.**  A direction at 1e-3 of the
  top — where real weak identifiability sits — is kept, not held.  "Is this
  parameter determined *well enough*" is `FIMReport.crb`'s question against a
  declared tolerance, and answering it here would discard real information
  under the name of a rounding threshold.
* **The degeneracy has to be a fixed direction in the optimiser's
  coordinates.**  `SpringDamperNode` declares `transform="log"` on `stiffness`
  and `mass` but the identity transform on `damping`, so that a damping of
  exactly zero stays representable; the scale direction is then `(c, 1, 1)`
  and it *rotates* as `c` moves.  No direction is null for the whole run, `G`
  comes back full rank, and the guard holds nothing.  Declaring
  `transform="log"` on every parameter a scale degeneracy mixes is what makes
  it constant — and it is the coordinate system `fim(scale="relative")`
  already works in.

**A correction to the premise, while measuring it.**  `fit_lm` *does* move
along the exactly-null direction, by 0.5% of the scale on the same problem.
The gradient is orthogonal to `v`, but the Marquardt step is
`(A + λ·diag(A))⁻¹g`, and that is orthogonal to `null(A)` only when `diag(A)`
is isotropic on the relevant subspace.  What is different in kind, and why
`fit_lm` is not the defect `fit` is, is that its step vanishes with the
gradient: the drift converges and stops, so `n_iter` does not decide the
answer.

#### A fit returns the lowest-loss iterate it evaluated, not its last

`fit` returned Adam's last iterate, and Adam's step is about `lr` in size
whatever the gradient's: a fit started at, or reaching, a point whose
gradient is rounding noise walks away from it.  The parameters guide's own
ball-and-spring graph, started at its truth, began at loss `2.7e-8` and
returned parameters at `1.9e-4`; `fit_multiple_shooting`, which steps every
window state at `lr` too, went from `1e-13` to `2e-1`.  Nothing in the
result said so.

Both now return the lowest-loss iterate they evaluated, and `FitResult`
gains `best_iteration` (the number of updates that produced `params`, an
index into `losses`) and `best_loss`.  A tie goes to the later iterate, so
a run whose loss never rose returns its last iterate, bit for bit: across
the existing sysid tests, all 166 of the 205 fitter calls whose last
iterate was the lowest (104 `fit`, all 59 `fit_lm`, 3
`fit_multiple_shooting`) returned identical parameters, losses, `n_iter`,
`converged`, `excited_rank` and `undetermined_drift` to the previous
implementation.  The iterate the last update produced is one the loop never
evaluated (it evaluates before it steps), so when the budget runs out it is
evaluated once more, with the same compiled function; that evaluation is
not appended to `losses`, not reported to `callback` or the observers, and
its gradient is not fed to the `hold_undetermined` tracker.  A non-finite
loss there is warned about and not returned.  `fit_multiple_shooting`
takes the parameters and the window states from the same iterate.
`fit_lm` needed no change: it accepts only a step that lowers the loss, so
its last iterate was already its lowest, and it now fills the two fields.

The unguarded drift measurements above are of the *last* iterate.  The
selection does not remove that drift, because along an exactly flat
direction which iterate is lowest is decided by rounding: on the 120-step
σ = 0.02 record the unguarded lowest-loss iterate lands −7.50%, −7.12% and
−5.43% from the starting scale at `lr` 0.01, 0.05 and 0.2, for every budget
from 200 to 10,000 iterations, and `fit_multiple_shooting`'s spread is the
same 3.0%.  The guard is still what makes the answer the data's.

#### The guard holds only what the curvature confirms, and never raises the loss

As first shipped in this cycle, the guard held every direction the run's
gradients had not spanned.  Not spanned is necessary for a direction the data
cannot see -- every gradient of a least-squares loss lies in the row space of
`J` -- but not sufficient: a short or fast-converging run's gradients need
not span the directions the data *does* determine.  The guard then took real
progress back, silently, on the default path of all three fitters
(MADD-ANO-069, which no release carried).  `fit_lm` on a closed-form
four-parameter bowl, every direction determined, reached a loss of 0.0; the
guard reported `excited_rank` 1 of 4 and returned parameters at 0.22.  The
spring with stiffness 1% off went from 0.0 to 1.2e-4, and ten Adam steps on
the bowl were moved off the iterate the fit had selected, from 0.32342 to
0.32367.  Returning the lowest-loss iterate does not touch this: the guard
runs after the selection.

A direction is now held only if it passes two tests, each necessary and
neither sufficient alone: no gradient of the run pointed along it (the old
test), **and** the objective has no curvature along it at the selected
iterate.  For `fit_lm` that is `JᵀJ`, which it already forms, read with
`fim`'s own rank cutoff, so `fit_lm` holds a direction only if `fim` would
call it unresolved at the point returned; at most one extra `jacfwd`.  For
`fit` and `fit_multiple_shooting` it is the loss's Hessian, through
Hessian-vector products along the candidates (plus up to 8 of the
most-excited directions, for scale), at `sqrt(eps)` of the largest; on the
spring's degeneracy the null direction measures 1e-8 of the largest
eigenvalue and the weakest resolved one 6e-2.  Requiring both keeps a
degeneracy that rotates in the optimiser's coordinates unheld and reported
full rank, as before.  A fit whose gradients spanned everything pays none of
this.

Then **the loss gets the last word**.  The held point's loss, by the
fitter's own compiled function, may exceed the selected iterate's by at most
`2¹⁰·eps·|loss|` (1.2e-4 relative in float32) plus the first- and
second-order cost of perturbing every coordinate `u` of the held point by
four roundings, `eps·max(1, |u|)` each.  The relative term is the rounding of
evaluating the loss at an equivalent point: measured, the hold moved the
spring's loss by 0 to 5.3 `eps`, and `2¹⁰` admits a signal up to about a
thousand times its residual.  The absolute term is what rounding the held
point costs, which a fit at its precision floor needs: noiselessly, `fit_lm`
reached 1.2e-14 and the held point 2.5e-13.  Over every degenerate hold in
the spring fixtures that term had to admit at most 0.43 of one rounding per
coordinate, so four leave a margin of 37, and no more: a degeneracy that
rotates in the optimiser's coordinates is held along its tangent at a real
cost, which on the spring 1% off was 4.5e-11 on jaxlib 0.11.0 (held) and
2.0e-9 on 0.11.2 (declined), against a term of 4.4e-10.  If the check fails
nothing is held: `params` is the selected
iterate bit for bit, the new `FitResult.hold_declined` is `True`, and a
`RuntimeWarning` gives both losses.  It also refuses a hold that would cross
a bound, because it evaluates the same `constrain` the returned parameters
go through.

`excited_rank` now counts the directions that pass both tests, so it means
what its docstring always said: how many directions the data determines, as
far as the run and the returned point show it.  After the fix every case
above returns the fit's own loss, and the spring's scale is still held by
all three fitters: `fit` at `lr=0.2` lands within 1e-7 of the starting scale
for 200 and 10,000 iterations (unguarded: -5.43%), `fit_lm` within 1e-9
(unguarded: -0.66%), `fit_multiple_shooting` within 1e-7 (unguarded: -1.97%
to -4.35%), each with `hold_declined` False.  Every guarded result the
earlier guard left untouched is unchanged bit for bit, and so is every hold
it made correctly; over the sysid test files, the only results that differ
are ones it had moved uphill (see the PR for the per-call comparison).

### Persistence, REST and FMI follow the params

`gm.to_dict()` / `from_dict` and USD store each node's *effective* params
(`gm.effective_node_params`: constructor args with the live `gm.params`
written over them) and the graph's `ParamSpec` overrides, so a calibrated
graph reloads calibrated with the same trainable mask.  Checkpoints save and
restore `params`.

`build_model_description` exposes every `gm.params` leaf as an FMI
`parameter` / `tunable` variable `<node>.params.<key>` with the `ParamSpec`
description, units and bounds (`include_parameters=False` opts out).
`SidecarConfig(params=..., param_specs=gm.param_specs())` makes the sidecar
call the three-argument contract and serve `get_params` / `set_params`, as
wire requests too: a set value takes effect on the next step without
recompiling, and unknown names, wrong shapes or out-of-bounds values are
errors applied atomically.  So is a value that is not finite or that the
leaf's dtype cannot hold (`1e39` for a float32 leaf), with or without
`param_specs`: `set_params` applies the TCP bridge's own value check, so
the in-process door refuses what the bridge's `set` refuses.
`FmuSidecar.set_fmu_state` likewise applies the checks of the bridge's
`set_state`, through the same function: a snapshot holding a non-finite
value, a value a leaf's dtype cannot hold, or (with `param_specs`) a
parameter outside its bounds does not restore, and nothing is written.
The bounds apply to the values a restore would install: a parameter at the
value it holds now, or held when the FMU was instantiated, restores, so an
FMU exported with a parameter outside its declared bounds (bounds are
metadata to a graph) restores its own snapshots, as `load_state` restores
the graph's checkpoint; both doors refused them until this was found by the
state/IO differential harness.  State follows the same rule: a non-finite
value the FMU was instantiated with restores (a `diagnostics=True` coupling
group seeds its spectral `_meta` slots with `NaN`, and both doors refused a
snapshot holding the seeds), while a field that became non-finite still
does not.  FMU state snapshots carry parameters
(`serialize_fmu_state(params=)`,
`deserialize_fmu_state(return_params=True)`); legacy snapshots still load.

On the REST side, `PUT /graph/params/{node}` can address any leaf of the
node's live pytree — not only constructor params, so surrogate weights and
sharded wrappers whose inner node owns the params are reachable — updates
`gm.params` in place instead of forcing a recompile, and validates dtype
coercion, shape, finiteness and `ParamSpec` bounds for every key *before*
writing anything.

It also refuses, with a 400 naming the parameter and why, a value the
running node cannot use.  The route writes `node.params` as well as
`gm.params`, so the graph-level refusal below (which compares against the
node's own value) could not see it, and writing `node.params` rebuilds
nothing the node derived at construction: since 0.1.0 a write of
`WaveletAdaptiveNode.mass` or `LBMPipeNode.pipe_radius` / `propeller_x`
answered 200, was served by `GET`, left every step bit-identical to the
constructed value (even after `POST /graph/compile`), and was saved by
`to_dict()` / `save_state()` — reproduced at v0.1.0, v0.2.1 and v0.3.1;
at 0.4.0 before this fix the reloaded wavelet graph's objective was twice
the running one's.  The decision is the graph's
(`GraphManager._unused_node_write_reason`): a `static_data_deps` entry is
refused; a leaf the step or the node's hooks read is used next step; a
structural value is used when the hooks trace differently with it (the
recompile picks it up); otherwise a value `initial_state()` reads takes
effect at the next reset, and anything else is refused.  Ahead of that, the
route refuses a value that changes the fields, shapes or dtypes of the state
`initial_state()` builds, and a structural value the node's own constructor
refuses, since a saved graph is loaded through it.  The node's code
runs on a shallow copy reading the new value, never on the node, so a
runner thread stepping the graph is not disturbed; a wrapper that closes
over the node it wraps (both sharded wrappers) is answered for by that node,
which shares its params, and a write no copy can decide is refused.

The first version of this refusal left four routes open, found by the
audit of the frozen tree and closed before release.  A part-full
`LBMPipeNode` reads `pipe_radius` again in `initial_state()`, so the write
passed as "takes effect at the next reset" while the step kept the wall
mask of the old radius (bit-identical velocity; the reload 5.667e-03
away) — the node now declares its geometry in `static_data_deps`.
`HeatNode`'s `n_cells` passed the same way: unsharded the next step was a
500, sharded it returned a field for 16 cells at `dx = L/17` with the right
end never closed.  `stencil_order=3` passed as a structural value, and the
saved graph failed to load.  And every write to a node the probe could not
copy — each sharded wrapper, a node holding a method bound to itself — was
accepted unseen.  A sweep of every built-in node's constructor parameters
through the route now finds no accepted write that `to_dict()` does not
reproduce (MADD-ANO-024).  `POST /checkpoint/load`
likewise refuses, and undoes, a checkpoint carrying another value of a
parameter a node consumed at construction; it used to load, after which
every `/sim/step` was a 500.

One more such write was found by the state/IO differential harness: a
value the node's own step reads, but from which the node also derived the
points an interface mapping was built from -- a uniform `HeatNode`'s
`length`, from which its `grid_x` is built.  The route answered 200, the rod
stepped on the new length while the mapping kept the old grid's weights,
and the saved config did not load (MADD-ANO-063).  The graph now rebuilds
the node with and without the value and refuses it, on every door, when a
referenced field moves.

The FMU had the same hole with no graph in the way: the sidecar steps
`gm._compiled_step` directly, so `FmuSidecar.set_params({"p.params.pipe_radius":
0.5})` was accepted and reported by `get_params` while the velocity stayed
bit-identical to the old radius, and `build_model_description` exported every
`gm.params` leaf as a tunable parameter, initial conditions included.  It now
exports only the parameters the compiled step reads
(`GraphManager._param_leaves_the_step_cannot_read`, which unlike the
graph-level refusal does not spare a leaf only an unconnected input would
read: an FMU's graph is frozen) and lists the rest with the reason in
`ModelDescription.fixed_parameters`; `FmuTcpBridge` applies that list to its
sidecar, and `set_params`, `set_fmu_state` and the bridge's `set_state`
refuse a new value for a fixed parameter before anything is written, which
the C wrapper reports as `fmi3Error`.

### One params rule, and leaves the step cannot read

"Does this callable take `params`" had three answers in the tree:
explicit keyword or `**kwargs` (`accepts_params`, the graph's hook probes,
`ShardedStencilNode`); explicit keyword only (`ShardedUnstructuredNode`
and the duck-typed fallbacks of the graph and `ShardedPointwiseNode`); and
always `False` for a node object without `accepts_params` (`verify_node`,
the REST server).  So an inner `update_padded(**kwargs)` was calibratable
under `ShardedStencilNode` but silently absent from `gm.params` under
`ShardedUnstructuredNode`, and a duck-typed params node had its params
checks reported `SKIP`.  Every probe now reads one rule
(`maddening.core.node._method_accepts_params` /
`_signature_takes_params`), and a test runs every probe against one matrix
of spellings.

A leaf that is in `gm.params` but that the compiled step cannot read -- an
`initial_*` entry, a parameter baked into a static at construction
(`static_data_deps`: `HeatNode.grid_points` on a non-uniform grid,
`WaveletAdaptiveNode.mass`), geometry consumed in `__init__` -- was
accepted, ignored by the step and then written out by `to_dict()`, so the
saved graph reloaded as a different model (0.093 K after one heat step).
Changing one is now a `ValueError` naming the leaf, from every run method,
`check_params`, `to_dict` and `save_state`; see
[parameters](../user_guide/parameters.md).  "Cannot read" is declared
(`static_data_deps`) or found by a conservative walk of the traced step *and*
of the node's own hooks with every declared input supplied, so a leaf only
this graph leaves unexercised (a ball's `elasticity` without a table edge)
is carried, not refused.  Alongside: an int-spelled
constant a node declares trainable (`SpringDamperNode(stiffness=100)`) is
promoted into `gm.params` instead of silently dropped;
`ShardedPointwiseNode` / `ShardedStencilNode` forward the flux and
interface-correction hooks, and `ShardedUnstructuredNode` refuses interface
DOFs by name; `run_adaptive` / `run_adaptive_scan` resolve flux edges
instead of raising a bare `KeyError`.

### `node.params` after compile, and the round-4 sysid fixes

**A `node.params` write after compile reaches the step at the next
compile** (MADD-ANO-093).  A recompile kept every live `gm.params` leaf
over the node's value, because it could not tell a calibration from the
previous snapshot, so a write to `node.params` after the first compile
never reached the step: a `HeartPumpNode` set to 144 bpm and recompiled
went on at 72.  0.3.x, which had no `gm.params`, took such a write at the
recompile.  The rule now, per node constant: the value written since the
previous compile wins.  A calibration still survives a recompile; a node
write is taken at the next `compile()`, as a structural value always was;
both written since one compile, to different values, keeps `gm.params`'
value and `compile()` warns with both.  `gm.params` remains the way to
change a constant mid-run: the compiled step reads it on every call.

**`fit_lm` reports `converged` at the float floor.**  Only an accepted step
could set it, and a fit at its optimum proposes a step that rounds to
nothing, which cannot strictly lower a loss that is rounding noise: noiseless
spring data reached loss 8.7e-14 in four iterations and the run ended
"unconverged".  The iteration's proposal is now tested whether or not it is
accepted (a damped retry is not: it is short because it is damped).
`step_tol` is now a relative change of each physical parameter, `<=`, and
its default is `2**4` ulps of the parameter's own dtype, where the absolute
`1e-8` in the optimiser's coordinates it replaces was below one float32 ulp
of almost any coordinate.  `FitResult` documents `converged` per fitter:
for `fit` and `fit_multiple_shooting` it is only the `tol` test, so with
the default `tol=0.0` it is always `False`.

**Smaller fixes.**  `windowed_loss` refuses a negative or NaN
`continuity_weight` (its `> 0.0` guard read either as no penalty, where
`fit_multiple_shooting` refused -1.0) and a non-bool `mask_unconverged`
(`"no"` turned masking on).  A bounded identity leaf exactly on its bound
has the derivative into its range, 1, where `jnp.clip`'s tie rule gave 0.5
(a spring held at `damping = 0.0` had half its gradient there, and
`fit_lm`'s Jacobian column was halved); values and every other derivative
are unchanged bit for bit.  A hand-written mask refused for lacking only an
empty container (`"mappings": {}`) now says so, where the refusal printed
two identical leaf lists.  And two `windowed_loss` guards that survived
every sysid test as mutants are pinned: each window restarts on its own
multi-rate phase, and the mask reads every coupling group's verdict.

### Bounded coordinates, residual units, record phase and counted writes: the round-5 sysid fixes

**A parameter is kept where `constrain` is its transform** (MADD-ANO-104).
Past the clip of a `transform=None` leaf, or the clamp of a `log` /
`logit` leaf, the derivative is 0, and a coordinate one step carried there
stayed there: a spring's damping started at 4 against a truth of 0.05
landed on 0 and `fit_lm` said `converged=True`; a logit parameter thrown
to the sigmoid's edge did the same at 1.999998 of (0.5, 2).  (For a `log`
/ `logit` coordinate this corrected the verdict only -- the coordinate
still ended on its edge, unconverged and unannounced, until [the round-9
fix](#bounded-transforms-in-the-fitters-the-round-9-sysid-fixes).)  Every fitter
now projects the coordinates back after each update; `fit_lm` holds an
active bound out of its coupled step and never converges while a
parameter on its bound could lower the loss by moving into its range.  A
run that has lowered the loss and then rejects every candidate down to
one within `step_tol` converges at the rounding floor, which a weakly
determined parameter (whose floor proposal fits rounding noise) needs.
`tests/property/test_sysid_truth_recovery.py` holds this as an oracle:
from any truth and start in the box, `converged=True` means the truth.

**`fit_lm` no longer depends on the residual's units.**  The Marquardt
floor was an absolute `1e-12`; a residual of 1e-7 in its own units (or a
`noise_std` of 1e7) left the fit unconverged after 50 iterations.  It is
now `eps` times the mean of `diag(JᵀJ)`.

**`node.params` writes are counted** (MADD-ANO-105, and a refinement of
MADD-ANO-093).  A write is a write even of the value the node held, so
writing a constructor value back reverts a calibration; every entry point
-- the run methods, the fitters, `windowed_loss`, the profiler --
recompiles for a write the compiled step has not taken, so they all run
the same model (since 0.1.0 `gm.step` kept its trace while a new
`run_scan` length traced the write in); and `load_state` supersedes earlier
writes to the constants it restores.
`tests/property/test_differential_entry_points_after_a_node_write.py`
holds the agreement as an oracle.

**Smaller.**  `windowed_loss(start_step=)` (experimental, also on
`fit_multiple_shooting`) states the base step a multi-rate record began
at; a multi-rate graph without it warns that 0 is assumed (a record from
one step in was replayed on the wrong phase, 1.3e-2 at the truth).  A
`sample_every=2` test pins the `k * sample_every` window phase, which
survived every sysid test as a mutant.

### One sync point and a per-column floor: the round-6 sysid fixes

**`fit_lm` does not depend on the parameters' units** (MADD-ANO-121).
The round-5 fix's floor, `eps` times the *mean* of `diag(JᵀJ)`, crushed
the step of a parameter whose column was small beside another's: a
spring in SI units stopped after two iterations, `converged=True`, with
damping at 3x its truth, where the same spring in tonnes reached it.  The
floor is now `eps` times each column's own curvature.  As defence in
depth, neither convergence test fires unless the undamped,
column-equilibrated Gauss-Newton step from the iterate -- no damping, no
floor -- is also within `step_tol` (or, at the floor, lowers no loss).
The truth-recovery oracle now also requires the same answer, in about as
many iterations, with any parameter measured in units 1e4 times larger or
smaller.

**Every reader of `gm.params` sees a pending `node.params` write**
(MADD-ANO-122).  The round-5 fix took a write in at the entry points
that *run* the graph, but not where it is only *read*: a `jax.jit` or
`jax.grad` of a sysid loss handed `gm.params`, the FMU export,
`save_state` and `GET /graph/params` used the old value, and a
`gm.params` write of the old value after a node write was lost.
`gm.params` is now a property whose getter runs the one sync point the
entry points and `compile()` run too; every reader goes through it with
no change of its own.  The rule is now simply that the later write wins:
a `gm.params` write reads `gm.params` first, which orders it after any
node write before it (and the both-written warning is gone).  The
entry-point oracle covers the readers.

**Smaller.**  The floor rule's two guards -- the run has lowered the
loss, and no coordinate on a bound could descend inward -- are pinned
by a wrong-signed Jacobian and an on-bound case.

### Units the guard could see, and values too small to check: the claims-inventory sysid fixes

**The identifiability guard decides the same in any units** (MADD-ANO-135).
`hold_undetermined` compares coordinates with each other three times -- which
directions the run's gradients spanned, which the objective is flat in, and
how much loss rounding the held point may cost -- and did all three in the
optimiser's coordinates, where an identity-transform parameter is in its own
units.  `fit_lm` on a linear problem with the damping written in units of
`1e-5` reached a loss of `1.6e-11`, then held the damping direction, which the
data determines, at its start: loss `13.9`, `best_loss` still the pre-hold
value, no warning.  In units of `1e6` it returned `0.1` against `3.1e-12`;
in units of `1e-4` to `1e-2` and `1e4` it declined with a warning and
reported `excited_rank` 2 of 3.  All three questions are now asked with each
identity parameter measured relative to its own size -- `fim`'s
`scale="relative"`; a `log` or `logit` coordinate is relative already -- so
problems whose coordinates are all `log` are held to the same bits as before,
and the truth-recovery oracle's unit table now also requires the same
`excited_rank`, `hold_declined` and returned point from `1e-6` to `1e6`.
`fim(scale=None)` is raw sensitivities by design; its documentation now says
that its rank verdict therefore depends on the units.

**A bounds check compares the value it was given** (MADD-ANO-136).
`ParamSpec.check` compared through `jnp`, and XLA's CPU backend flushes a
subnormal operand to zero, so a float32 `-1e-40` passed the `(0, None)`
bound of a spring's damping -- in `check_params`, the FMU's `set` and
`set_state` and REST's `PUT` -- and read back below the advertised `min`.
The comparison is now made on the host, exactly, with the value and the
bound rounded to the dtype JAX compares them in; under a `log` or `logit`
transform a value whose distance from its bound the step's arithmetic
flushes to zero is refused as on it, because there the step's own `log`
of it is `-inf`.

**A value its type flushes to zero is refused** (MADD-ANO-137).  The FMU's
value check, and REST's twin of it, refused a float32 set to `1e308` (stored
as `inf`) but stored one set to `1e-50` as `0.0`.  It is now refused in the
same words; a value that rounds to a subnormal keeps its sign and magnitude
and is accepted.

The three were rows of the sysid and FMU claims inventory
(`docs/validation/sysid_fmu_claims.yaml`): SYS-063 and SYS-088 (the guard),
SYS-109 and FMU-039 (the bounds check) and FMU-024 (the underflow, which
the maintainer ruled refused) are now `verified`, which leaves 144 of the
inventory's 147 rows verified and three ambiguous.

### x64, replayed warm starts and arguments that are refused: the round-7 sysid fixes

**The identifiability guard holds an exact degeneracy under x64**
(MADD-ANO-150).  x64 is a supported mode -- `fim` recommends it for an
ill-conditioned problem -- and there the guard missed the spring's
`(k, c, m)` scale degeneracy in most runs.  Its tracker decomposed the
gradients' Gram matrix `G` with `eigh`, which resolves `G` only to about
`eps64` of its largest eigenvalue, while an exactly null direction carries
`eps64**2` of it: `5e-17` came back against a cutoff of `4e-31`,
`excited_rank` read 3 of 3 and Adam left the scale up to 1.8% off its start.
The tracker now keeps the triangular factor of the stacked gradients and
reads the spectrum as its singular values, which resolve to `eps` at either
precision; on the spring the null direction sits at most 0.30 of the cutoff
in float64 (0.28 in float32) with the next direction 14 decades above it.
The guard's precision is now the coarsest trainable leaf's: float32
constants in an x64 graph, judged at float64's `eps`, were never held.  The
held point agrees with earlier builds' to rounding, not to the bit, because
the eigenvectors now come from a different decomposition.

**`windowed_loss` replays coupling warm starts** (MADD-ANO-151).  A
coupling group's predictor history and IQN-IMVJ secant matrices are state
the observations cannot carry, and every window restarted them cold: with a
quadratic predictor at `max_iterations=1` the loss at the generating
parameters was `1.7e-2` over 10-sample windows and a fit started at the
truth walked 6.4% away.  Window `w` now starts from the warm starts window
`w - 1` ended with (window 0 from the cold ones `compile()` and
`reset_state()` leave), with their gradient stopped, so the loss is exactly
zero at the truth for a record taken from `compile()` or `reset_state()` and
gradients still do not compound across windows.  A window
`mask_unconverged` drops hands on cold warm starts; a record that began
after the graph had stepped (`start_step > 0`) warns that it cannot be
replayed exactly.

**`fim` says when its rank was decided below float32's normal range**
(MADD-ANO-152).  The rank does not move with `noise_std` while `F = JᵀJ` can
be formed; once its cutoff falls below `2 * m * tiny`, the products of the
smallest directions flush to zero and a determined direction dropped out
(rank 2 to 1, every `crb` `inf`) with no warning.  `fim` now raises a
`PrecisionLimitWarning` there and `FIMCore.precision_limited` is set.

**A wrongly typed argument is refused** (MADD-ANO-153).  `fit(tol=True)`
read `True` as `1.0`, stopped at its first evaluation and reported
`converged=True` at an unfitted start, and a mask leaf of `"False"` selected
its parameter.  A hyper-parameter must now be a real number, and every leaf
of a caller's mask a bool, for the three fitters, `fim` and `fim_core`.

And four smaller corrections.  `ParamSpec.check` computes a `log` or
`logit` value's coordinate as the transform does and refuses one that is
not finite (a float32 `nextafter(1, 0)` under `(-1, 1)` rounds to `t == 1`).
`fim(scale="nominal")`'s width guard tests the width -- zero, subnormal or
infinite at the parameters' precision -- and no longer its square, which
refused the bounds `(0, 1e-23)` of a parameter whose natural size is
`1e-23`.  Under x64 `fit_lm`'s floor rule ended one damped candidate short
of the default `step_tol` and reported a fit at loss `2e-30` unconverged;
the ladder now goes on to a candidate within the tolerance, and a parameter
whose truth is exactly 0 is documented as having no relative resolution
(such a run stops unconverged unless given a `tol`).
`fit_multiple_shooting` and `FitResult` now say the same about `best_loss`:
the selected pair's loss before the guard, which a held pair meets within
the guard's tolerance.

These were rows of the sysid and FMU claims inventory: SYS-006, -023, -029,
-061, -062, -064, -065, -067, -080, -081, -102, -103 and -104 hold under
their stated conditions (tightened where the audit's were right), SYS-071
is `verified`, and SYS-127 to SYS-130 record the new statements, which
leaves all 152 of the inventory's rows verified.

### The whole float range, writes the graph missed, and counts in any spelling: the round-8 sysid fixes

**`fit_lm` and `fit` across the float range** (MADD-ANO-174).  In float32,
XLA's CPU backend flushes a product below `tiny` to zero, and `fit_lm`
formed its loss `0.5 * sum(r * r)`, its normal matrix `JᵀJ` and its
gradient `Jᵀr` bare.  On a noiseless linear problem `r = s (A x - y)`,
whose solution does not depend on `s`, it returned a point 2% and 7% off at
`s = 1e-18` and `1e-19` and its unmoved start at `1e-20`, every time with
`converged=True` and `best_loss` 0.0; an identity parameter whose natural
scale is `1e-23` (the parameter guide's own example) or `1e23` came back
unmoved, "converged", at a loss of 5.35 against a minimum of 0; and `fit`
returned its start at `1e-19`, its gradient flushed to zero.  The loss is
now computed on the residual framed by a power of two and unframed in
float64; both solves frame every Jacobian column and the residual by powers
of two before forming anything and unframe the step exactly (an ordinary
fit's iterates move in their last bits, from the equilibrated solve's
pivots); `fit_lm` never reports `converged` on a solve with an
unrepresentable column or on a zero loss from a nonzero residual, and warns
instead; and `fit` and `fit_multiple_shooting` take a first gradient below
`tiny / eps` again with a power-of-two cotangent, which lifts the backward
pass out of the flush and leaves every other run bit for bit as it was.
Recovery is now pinned for residual scales `1e-30` to `1e37` and parameter
scales `1e-30` to `1e30` in float32 (`1e-157` to `1e150` and `1e-160` to
`1e160` in float64): any problem whose Jacobian holds normal numbers.
`fit`'s loss is yours, so it can only be evaluated where it is finite (a
`0.5 * ||s r||²` above about `1e38` is refused as non-finite), and its value
can still flush to 0.0, which a `RuntimeWarning` says once and which then
does not meet `tol`.  The same change settles three smaller findings: at
the float32 floor of a noisy fit `converged` flipped with an identity
parameter's units, because the damped candidates' verdicts were the loss's
rounding (the floor rule now takes the undamped Gauss-Newton step while it
still lowers the loss); with a truth exactly on a bound the on-bound
gradient's rounding decided the verdict (a pull whose own Newton step is
within `step_tol` now counts as zero); and a float32 constant in an x64
graph took five times the iterations, its float64 coordinate creeping
below one float32 ulp a step (it now stays on float32's grid, and a move
that rounds away is held while the others are solved again).

**`node.params` writes the graph missed** (MADD-ANO-175).  After
`node.params = {**node.params, "k": v}` -- a plain dict -- every later write
into it was lost to `gm.params`, every run, `compile()` and `to_dict()`;
and so was an element written in place into a list-valued parameter, though
the counting mapping's docstring said the graph compared values.
`SimulationNode.params` is now a property whose setter stores any mapping
in a fresh counting mapping (so a reference to the dict you assigned is not
`node.params`), and every sync compares each list, array and nested dict
value with a copy taken at the last one, bit for bit.

**`fim` flags a Fisher matrix that flushed to zero** (MADD-ANO-176).  At
`|J|` near `1e-20` in float32 every product of `F = JᵀJ` flushes, `F` is
exactly zero and the cutoff is 0: `fim` reported rank 0 with every `crb`
`inf` and no warning, and `fim_core` `precision_limited=False`, for a
residual of rank 2.  Both now flag it when `J` has a nonzero entry.

**A `log` or `logit` spec its leaf cannot hold is refused** (MADD-ANO-177).
`ParamSpec(bounds=(-3e38, 3e38), transform="logit")` has finite float32
bounds and a width that overflows float32: `constrain` mapped the midpoint
to the upper edge and far coordinates to NaN.  Every map and check of such
a leaf now refuses the spec by name; the same spec still maps a float64
leaf.

And three loud ones.  `n_iter`, `notify_every`, `sample_every` and
`start_step` read an integer in every spelling (a 0-d NumPy array and a JAX
integer were refused).  Under x64 the parameter guide's sysid recipe failed
on `SpringDamperNode` (MADD-ANO-017, now partially resolved: see "Known
anomalies").

These were rows of the sysid and FMU claims inventory: SYS-061, -080,
-082, -083, -084, -086, -088, -103, -104, -115 and -129 hold under their
stated conditions, now stated precisely (any residual and parameter whose
Jacobian holds normal numbers, every mutation of `node.params`, an `F`
flushed to zero), and SYS-131 and SYS-132 record the new statements.

### Bounded transforms in the fitters: the round-9 sysid fixes

Four audit rounds running found a serious defect where a bounded transform
(`log`, `logit`) meets the floating-point limits inside a fitter, each one
the neighbour of the case fixed before it.  This round's fixes are two
structural changes, not three point fixes, and two test batteries that
hold the class.

**`fit_lm` comes back from the edge of a `logit` range** (MADD-ANO-104,
which was registered resolved with only its verdict fixed).  The range a
`log` / `logit` coordinate could be stepped to ended just inside the clamp,
where `dp/du` is about `4 eps` of the bounds.  An early step that overshot
-- the damping carried along while the stiffness was still far off -- was
accepted there, and from there every Levenberg-Marquardt candidate, `1/eps`
long in the coordinate, landed on the *opposite* edge and was rejected, all
twelve damped retries with it.  The run ended on the bound with
`converged=False`, `n_iter` 3 to 5 and no warning.  Under x64, on the
parameter guide's spring from the guide's own starting stiffness, a `logit`
damping came back 2.0 for a truth of 1.9 from 20 of 60 starts well inside
its range, where the same fit with the bounds clipped recovered the truth
from all 60; in float32 the same once the stiffness started further off.
Three things changed:

- **A step is read on the transform's tangent where that moves the value
  less.**  The step is solved on the residual's linearisation, which for a
  transformed coordinate is the transform's tangent, and a step `v` can be
  read two ways, `p(u + v)` or `p(u) + p'(u) v`, that agree to first order.
  Where the transform is flat the tangent's is the shorter, and it is the
  step the same parameter would take clipped: a coordinate on the edge of
  its range comes back in one step.  Where a step heads for a bound the
  curve's is the shorter: it closes the distance by a factor of `e` at
  most, and never lands on the bound.  No tolerance is involved, and an
  identity coordinate is stepped bit for bit as before.
- **The range ends where the transform still resolves its bound**:
  `sqrt(eps)` of the bounds' own size inside each one (3.5e-4 of the range
  in float32, 1.5e-8 in float64) rather than a few `eps`.  No step of any
  fitter can land where the slope is below `sqrt(eps)` of the range.
- **A fit that ends on that edge says so**, in a `RuntimeWarning` naming
  the parameter, the edge and the remedy; `fit` and `fit_multiple_shooting`,
  whose step is about `lr` whatever the gradient, also name a parameter
  they are *started* with on it, since it takes them about `8 / lr` updates
  in float32 and `18 / lr` in float64 to leave.

The cost is stated rather than hidden.  A `log` / `logit` parameter cannot
be fitted nearer its bound than that margin; one that belongs on its bound
wants `transform=None`, which clips and can sit there.  And `fit_lm` takes
another path to the same answer for any `log` / `logit` parameter whose
step is not small: a `log` value asked to grow by three times itself now
grows fourfold instead of `e**3`-fold.  On the reproducer's x64 scan from
a stiffness of 5 the *control* recovers 18 of 18 starts where it recovered
11, because the stiffness beside the damping is a `log` coordinate too.  An
unguarded degenerate fit settles at another point of its flat direction.

**One evaluation of `theta -> params`** (MADD-ANO-202).  The fitters mapped
their coordinates to physical parameters twice: inside the jitted
objective, and again eagerly for `FitResult.params`.  A `logit` leaf is
rebuilt as `lo + (hi - lo) * sigmoid(u)` in its own dtype, so what the map
can return is spaced like the floats near the *bounds*, and XLA contracts
the multiply-add where eager JAX does not: the two evaluations landed on
different points of that grid.  At ordinary bounds that was an ulp, for one
value in 400.  In float32 under `logit` bounds `(-1e4, 1e4)`, `best_loss`
read 1.27e-8 for parameters whose loss is 4.5e-10.  And a leaf the fit did
not move was returned as given while the objective ran its round trip: a
damping left out by `mask=` at its true 2.0 under `logit` bounds
`(-1e6, 1e6)` was run at 2.0266, and the stiffness fitted beside it came
back 30.06 for a truth of 30 with `converged=True`.  This was the third
round to find two evaluations of this mapping disagreeing in their last
bits, so one of them is gone rather than reconciled: each fitter maps its
coordinates once, by one compiled function, runs the model on the arrays
that returns and hands those arrays back.  The model-side programs take
the physical tree as their argument and contain nothing of a transform, a
mask or a start.  A leaf the fit does not move is the object that went in,
in the result and in every evaluation.  `fit_lm` takes every loss from one
compiled residual, so `losses`, the acceptance test and `best_loss` are
values of one function.

**The transform's resolution is stated, and said** (SYS-104, SYS-146).
`gm.constrain(gm.unconstrain(p))` returns `p` to within 16 of the
transform's spacings -- `eps * max(|lo|, |hi|, hi - lo)` for `logit`,
`eps * |lo|` for `log` -- plus 16 `eps` of the value, not "to a few ulps".
A fitter warns (`PrecisionLimitWarning`, naming the leaf, the spacing and
the remedy) about a parameter it is asked to fit whose value that spacing
leaves fewer than half the working precision's digits of: a float32 3.0
under `logit` bounds 1e4 wide.

**`node.params = {**node.params, "k": v}` writes `k`** (MADD-ANO-203).  The
graph read a replaced mapping as a write of every key, so the one-key idiom
put the node's own value back over every other constant of that node a fit
had calibrated in `gm.params`: a stiffness calibrated to 45 went back to 30
when the damping was assigned this way, with no warning.  A replacement now
takes over the write counts of the mapping it replaces and counts a write
of each key whose value it changes, bit for bit, and of no other -- the
rule an in-place element write already followed.  The two spellings of a
write differ in one case, by design: handed the value the node already
holds, a key is not written by a replacement and is written by
`node.params[key] = value`, which is the way to put a node's own value back
over a calibration.

**The batteries.**  `tests/property/test_sysid_truth_recovery.py` runs the
guide's spring over transform (`log`, `log` from a non-zero bound, `logit`)
x precision x where the truth sits (5%, 50%, 95% of the range and a few
float spacings inside each edge) x where the start sits (likewise) x the
other parameter's start (0.1 to 30 times its truth) x the third leaf frozen
or left out by `mask=` x the three fitters, each cell beside the same fit
with the transform replaced by clipping.  Wherever that control recovers
the truth the transformed fit must, or must name the parameter in a
warning.  For `fit_lm` that holds in every cell.  For the Adam fitters it
is asserted where the stiffness starts within a factor of ten below, or
two either side, of its truth: from ten or thirty times too stiff the loss
has other basins, which Adam in a `log` coordinate (whose step multiplies
the value) walks into and Adam in the value's own coordinate does not --
a difference between two parametrisations, with no edge in it -- and those
cells are run and held to what nothing excuses, never ending on an edge
without the warning.  `tests/property/test_sysid_one_evaluation.py` stands
between each fitter and its model and reads the arrays every evaluation was
handed: one of them received exactly the arrays of `FitResult.params`, its
loss is `best_loss` bit for bit, and a leaf the fit does not move is the
object that went in, over drawn transforms, bounds up to a million times
the value, precisions, masks and guard settings.

These were rows SYS-054, -057, -071, -080, -086, -087, -104, -115 and -116
of the sysid and FMU claims inventory, restated; SYS-144 to SYS-147 record
the new statements.

### Node verification

The `verify_node` / `assert_node_verified` battery gained three checks:
`params_consistent` (an injected `params_pytree()` reproduces the
baked-constant step to float32 round-off), `params_gradient_finite`, and
`params_effective` (every trainable leaf has a non-zero gradient on at least
one sample — which catches a constant still being read from `self.params`).
`params_effective` also differentiates `derivatives()` and
`implicit_residual()` with respect to the injected leaves where a node
implements them, tells a constant the right-hand side does not consume from
one it reads from `self.params` while ignoring the injected value (by varying
the constructor value in place and restoring it), fails an override declared
without the keyword, and names the paths it checked in `detail`.  Every
perturbed element is also compared, per path, with a node rebuilt from
`to_dict()` at that value, so a constant split between the injected value
and a copy made in `__init__` (`0.5 * p["k"] + 0.5 * self._k0`: in a graph,
an injected `k = 4` gave `x(50) = 0.218` against `0.130` for a node built
with it) fails; the probe used to run only for an element that moved
nothing on the path.

The scalar it differentiates is a fixed random projection of the outputs,
not their plain sum.  The sum is blind to any conserved exchange: a
two-compartment node with `dA/dt = -k (A - B)`, `dB/dt = +k (A - B)` that
reads the injected `k` on every path has `d(sum)/dk == 0` exactly on
every sample, and the first form of the check FAILed it — and made
`assert_node_verified` raise — on a correct node.  The projection draws
one weight per output element from `numpy.random.default_rng([20260922,
crc32(field)])`, magnitude in `[0.5, 1.5]` with a random sign, so no law
with equal coefficients can cancel it and a verdict is reproducible from
the seed.  Every seeded broken node still fails.

**A consequence worth stating plainly for a downstream CI:** a subclass
whose `derivatives()` or `implicit_residual()` override predates the
`params` keyword now **FAILs** `params_effective`, so `assert_node_verified`
raises on it after this upgrade where it passed before.  Nothing else
about such a node changed — every call without `params` is byte-identical
to 0.3.1 — and the migration is one line per method: declare
`(..., *, params=None)` and read constants from `{**self.params, **params}`.

**Per path, per element, by value.**  The release audit planted five
nodes on which an injected leaf does not act and one correct node; the
first form of the check passed all five and failed the sixth.  It summed
`update` and the fluxes into one objective, so a flux reading the injected
value masked an `update` that did not; it never probed
`compute_interface_correction`; it counted a vector leaf live when *any*
element had a gradient; its constructor-value probe could not see a copy
made in `__init__`; `0.5 * p["k"] + 0.5 * self.params["k"]` has a non-zero
gradient and passed; and a correct node reading `x0` only through a
comparison (zero gradient) was reported as reading it from `self.params`.
`params_effective` now probes each path on its own (`update`,
`compute_boundary_fluxes`, `derivatives`, `implicit_residual`,
`compute_interface_correction`), perturbs each element of a vector leaf,
and adds a value test to the gradient screen: at a perturbed value inside
the leaf's bounds, the injected output must match the output with that
value constructed (swapped into `node.params` and restored); where nothing
moves on a path but the leaf acts on another, a node rebuilt from
`to_dict()` catches a copy made at construction.  All six cases are
closed.  Still open, and documented in the docstring: the
construction-copy probe needs a node that rebuilds faithfully from
`to_dict()` (one that does not is reported "not consumed" there, as
before); the value test runs on the first 20 samples, so a split read
confined to a region those samples miss can pass; a leaf larger than eight
elements is value-probed on eight evenly spaced ones.  `detail` now names
the flux and interface-correction paths as well, and what each path does
not consume.
They report a new `SKIP` status, which counts as passed, on nodes whose
`update` takes no `params`; `SimulationNode.accepts_params()` exposes that
probe.  `tests/verification/test_builtin_nodes_verified.py` runs the full
battery on every built-in node in CI.

The battery itself is also new in a broader sense:
`maddening.testing.verification` is a Hypothesis battery checking finite
outputs, preserved structure, determinism, jit/eager agreement and finite
gradients, plus opt-in `output_bounds`, `energy_fn` and custom
`invariants`, returning the shrunk counterexample on failure.

### Order of accuracy

Every check above compares the code to itself.  None of them can see a wrong
discretisation: a Laplacian with the wrong weight at the boundary is finite,
deterministic, jit-consistent and perfectly differentiable, and a six-auditor
whole-tree audit that produced ~28 findings structurally could not have found
one either.  `maddening.testing.mms` is the other half of the battery — it
compares the code to the mathematics.  A node declares the order its scheme is
designed for in `NodeMeta(discretization_order=DiscretizationOrder(spatial=...,
temporal=...))`, or per instance where the order depends on how the node was
constructed; the harness substitutes an analytic field into the equation the
node claims to solve, derives by automatic differentiation the source term
that makes that field exact, drives the node with it over a refinement ladder,
and fails the node when the *observed order of convergence* falls short of the
declaration.  A node that declares nothing is skipped explicitly rather than
passed silently.

Order rather than error, because error tolerances do not falsify anything: a
wrong stencil weight, a mishandled boundary or an off-by-one in a flux leaves
the absolute error looking perfectly acceptable on the one grid a threshold
test runs on, and shows up only as order 1 where order 2 was claimed.  Four
studies are registered (MADD-VER-005 to 008): `HeatNode` measures 2.000
against a declared 2 in space and 1.000 against a declared 1 in time,
`LBMNode` measures 1.998 against a declared 2 in space on a periodic
Kolmogorov flow driven by a manufactured body force, and `RigidBodyNode`
measures 0.999 against a declared 1 in time under a manufactured 6-DOF
force/torque trajectory.  The acceptance band is `[declared - 0.25,
declared + 1.0]` on the order over the finest pair of the ladder, and both
half-widths are set from recorded fixtures rather than from intuition.  The
finest pair lands close to theory everywhere measured — HeatNode 2.000
against 2 and 3.957 against 4, LBMNode 1.998, and the five ODE nodes 0.999 to
1.029 on their registered ladders (MADD-VER-008 to 012; `SpringDamperNode`
1.029, `BallNode` 1.002) and 1.000-1.001 on the longer 100-1600 step ladders
of `benchmarks/results/audit_040_r2/numerics/r6_declared_orders.log` — so the
largest honest shortfall is **0.043**, while the defects this
harness has found fall a full order or more short (MADD-ANO-007 measured
1.001 against a declared 2, MADD-ANO-008 0.954 against a declared 4).  The
upper half rejects a study that is not exercising the scheme at all; the
largest honest pairwise order recorded is **4.126** on a `tanh_bump` closure,
so `declared + 1.0` admits it.  An earlier draft of this paragraph justified
the lower half with "the coarsest pair of the same ladder sits up to 0.16
low", and `DEFAULT_ORDER_EXCESS` was defended with a measurement of 5.02.
Both figures were re-derived against the tracked fixtures for 0.4.0 and
neither reproduces — 5.02 would have failed the band it was quoted in
defence of — so both are withdrawn.  Every number the constants' docstrings
now quote comes from a fixture listed in
`maddening.testing.mms._ORDER_BAND_FIXTURES`, and
`TestTheOrderBandIsJustifiedByTheseNumbers` fails if one drifts.

It found two defects in `HeatNode` on the first run, plus a third that fell
out of the same work; all three are **fixed in this release** (see "HeatNode:
the boundary condition, the 4th-order stencil and the stability bound"
below).  `stencil_order=4` converged at **order 0.954**, not 4, and was 10x
*less* accurate at 80 cells than the default second-order stencil, because
the two ghost values of the 5-point stencil were populated one cell out of
position (MADD-ANO-008).  Separately, `boundary_input_spec` documents
`left_temperature` as the Dirichlet condition "at left end", while the code
applied it at the first *cell centre*, half a cell away: a caller who
followed the documentation got **order 1.001** instead of 2, and an error of
0.88% at 160 cells against 0.0065% under the other reading (MADD-ANO-007).
Both were recorded as strict xfails, so correcting the node turned them into
an XPASS that had to be dealt with — which is what made the fix and the tests
land as one change.  Both also sat comfortably inside the acceptance criteria
of MADD-VER-001 and MADD-VER-002, the pointwise-error and "rate between 0.7
and 2.5" benchmarks that already covered this node — which is the case for
measuring order instead of error, made by the code under test.  The third
finding: the CFL limit the node documented, `dt < dx^2/(2 alpha)`, is the
3-point stencil's; the 5-point stencil diverged at a Fourier number of 0.40,
inside it (MADD-ANO-009).

`maddening.testing.mms` also documents what it could **not** do.  `LBMNode`
declares no temporal order on purpose — the lattice fixes `dx = dt = 1` and
`update` ignores its `dt` argument, so there is no timestep to refine, and the
harness skips that axis explicitly.  `LBMPipeNode` cannot be verified this way
at all: its only forcing input is a scalar `propeller_force` confined to a
fixed actuator-disc mask, so there is nowhere to inject a spatially varying
manufactured source.  And the ladders run under `jax_enable_x64`: in float32
they measure a clean order to about 80 cells and then turn over as round-off
overtakes the discretisation error, which the harness reports as a
non-converging ladder rather than as a wrong order.

### The integrators' order is conditional, and the condition was not stated

The same question asked of `maddening.core.simulation.integrators` produced a
fourth finding, and this one is about a name rather than a stencil.
`rk4_step` is documented as "Classical 4th-order Runge-Kutta" and is what
`integrate_node` selects by default.  None of the four functions in that module
takes a time, so every Runge-Kutta stage is handed the same `boundary_inputs`
object the caller passed for the whole step — a zero-order hold on the input.
For an autonomous problem that is exact, and the classical orders are
delivered: measured 1.003, 2.005 and 4.006 for `euler`, `heun` and `rk4`.
Drive a genuinely time-varying input the obvious way, one value per step, and
the hold contributes an O(dt²) local error that no stage arithmetic can remove:
**all three methods converge at order 1** — measured 1.022, 1.017 and 1.016 on
a manufactured `dx/dt = -x + u(t)` over a 10/20/40/80/160 ladder in float64
(MADD-ANO-014).  Worse than a lost order, `rk4` is also 1.5x *less* accurate
than `euler` at the same `dt` in that regime, because its extra stages refine a
term that is no longer the dominant one.  Reaching for the higher-order method
is not the conservative choice a user would reasonably assume it to be.

Nothing in the arithmetic changed, because nothing in the arithmetic is wrong.
The full order is already reachable through the existing interface, by the
textbook autonomisation of a non-autonomous system: carry time as a state field
with derivative 1.  Each stepper builds its stage states as `state + a_ij*dt*k_j`,
so such a field arrives at stage *i* holding exactly the Butcher node
`t + c_i*dt`, and a forcing read from it is read at the right time — measured
1.022, 1.999 and **4.002** on the same problem and the same ladder.  That is
exact rather than an interpolation, and it is why no argument was added: two
cheaper-looking alternatives were measured and neither reaches 4, with linear
interpolation of the input between step endpoints capping `rk4` at 2.000 and
midpoint evaluation of the frozen input capping it at 2.000 as well.  An
optional stage-time or callable-input argument would have worked, and was
declined — four functions with no in-tree callers do not earn new public API to
reach a number that is already reachable.  What shipped instead is the honest
statement of the condition, in the docstrings and in a new algorithm guide
(`docs/algorithm_guide/solvers/explicit_integrators.md`), and measurement of
both regimes in `tests/verification/test_integrator_order.py` so neither can
drift.  MADD-ANO-014 is recorded as *partially resolved* rather than resolved,
because the degraded path is still the default and still silent: a stepper sees
an opaque dict and cannot tell a constant from a sample of a waveform, so there
is no warning to raise.

One methodological note carried over from that work, because it generalises.  A
refinement ladder answers "what order is this?" and never "which scheme is
this?".  Under the frozen input all three methods measure ~1.02 with errors
agreeing to three significant figures, so swapping one for another walks
straight through any convergence band — and mutation testing confirmed it:
rewiring `integrate_node`'s `"rk4"` entry to `heun_step` left every ladder
green.  Scheme identity needs an algebraic check, so one step of `x' = ax` is
pinned against each method's stability polynomial `R(z)` to 1e-13 relative,
which fixes every stage weight exactly.  With that added, 11 of 11 coefficient
and dispatch mutations are caught.

### HeatNode: the boundary condition, the 4th-order stencil and the stability bound

All three findings above are fixed, and the fix changes results for every
existing `HeatNode` caller.

**The Dirichlet data now lands where it is documented** (MADD-ANO-007).
`left_temperature` and `right_temperature` are the temperatures at the rod
ends `x = 0` and `x = L`, as `boundary_input_spec` has always said.  The grid
is cell centred, so those ends lie half a cell outside the first and last
cell centre, and the datum is imposed through a ghost cell
(`T_ghost = 2*T_b - T[0]`, the conservative finite-volume closure) rather
than by overwriting the end cells after the update.  Observed spatial order
moves from **1.001 to 2.000** over a 10/20/40/80/160 ladder, and the error at
160 cells from 0.88% to 0.0066%.  Neither the grid coordinates nor the
meaning of the input changed; what changed is the enforcement.

*What you will notice*: `T[0]` no longer equals `left_temperature` exactly —
it is a cell centre half a cell inside the rod, so it should never have.  If
you read a boundary value back out of the state array, read the value you
supplied instead.  And with no boundary data supplied the end cells now
evolve under a zero-flux closure, which makes the node exactly conservative;
they used to be frozen at their previous values, which was neither physical
nor conservative.

**The 4th-order stencil now is one** (MADD-ANO-008).  Both ghosts are built
by cubic extrapolation through the rod end and the three nearest cell
centres.  Observed order moves from **0.954 to 3.957**, and the error at 80
cells from 2.65e-3 to 2.03e-7: `stencil_order=4` is now 1260x *more* accurate
than the default, where it used to be 10x less.

**The stability bound is per-stencil, and enforced** (MADD-ANO-009).
`dt * alpha / dx^2 < 1/2` is the 3-point bound and is exactly right for
`stencil_order=2`.  For `stencil_order=4` the bare 5-point symbol allows 3/8
and the cubic boundary closure tightens the whole operator to 0.3169–0.3249
depending on `n_cells`; `maddening.nodes.heat.MAX_FOURIER_NUMBER` records
`5/16 = 0.3125`, conservative at every resolution.  **The constructor now
raises `ValueError` on a configuration above its stencil's limit.**
Documentation alone had already been tried here and had failed in exactly
this way — the published bound was wrong, so following it was not protection,
and the consequence was a silent divergence to NaN.  The check sees the
constructor's `timestep`, `thermal_diffusivity` and `length` only: a `dt`
handed to `update()`, or either constant moved by a calibration run, still
bypasses it, and MADD-ANO-002 stays open for that.

**A rod built on `grid_points` is checked too.**  Until the release's
final audit it was not: the same 20 cell centres given as `grid_points` at a
Fourier number of 5 built without a word and reached `inf` by step 32.  The
variable-spacing stencil is 2nd order whatever `stencil_order` says.  Its
check is `dt * alpha / min(h_L * h_R) <= 1/2`, where `h_L` and `h_R` are a
cell's spacings to its two neighbours and the end spacing repeats beyond
each end.  That is the Gershgorin bound of the operator.  It is exact on a
uniform grid, sufficient on any grid, and within 1.5x of the sharp limit on
the graded grids tested (1.007x on an abrupt 10x refinement).  It is less
strict than the smallest spacing squared, which would refuse stable rods.
The same round made the constructor refuse constants no rod can have
(MADD-ANO-062).  A non-positive `timestep` or `length`, a negative
`thermal_diffusivity`, a non-finite one of these, and `grid_points` that are
not finite and strictly increasing each used to run and answer wrongly.  A
negative length flipped both fluxes, points out of order gave a wrong
Laplacian, and a negative timestep or diffusivity ran the equation
backwards.  None of the shipped tests or examples was affected.

**Two rods coupled end to end have a lower limit, and `compile()` warns
about it** (MADD-ANO-050, open).  When each rod's end-cell temperature is
the other's Dirichlet datum and a coupling group converges that exchange,
each datum becomes an implicit value.  The pair then has an interface mode
that flips sign every step and is unstable above 3/8 at `stencil_order=2`
and above 0.226 at `stencil_order=4`, well inside the 1/2 and 5/16 each
constructor checks.  The order-4 figure was first quoted as 3/8 too.  It is
0.2261266 (0.2261215 at 5 cells), derived from the interface mode and
confirmed on the node's own operator; an order-4 pair at 0.25 is non-finite
by step 150.  `compile()` now emits a `UserWarning` naming both rods, their
Fourier numbers, the limit and MADD-ANO-050.  It fires when a coupling group
holds two uniform `HeatNode`s exchanging end cells through `extract_first`
/ `extract_last` at matching ends and either rod is past its pair limit.  It
is a warning, not a refusal.  It recognises only that pattern, so a custom
transform, a wrapped or non-uniform rod, or a sub-cycled pair of different
timesteps is not judged.  Nothing changes numerically.

**MADD-VER-002, the benchmark that concealed MADD-ANO-007, is corrected**.
It asserted `0.7 < rate < 2.5` while its acceptance criteria claimed 1.5–2.5
for the interior and its code measured the global rate; the executed band was
the widest of the three and wide enough to accept the ~1.0 the node was
producing.  It also cited MADD-ANO-002, which is about CFL enforcement.  The
band is now `[1.7, 2.3]` around the theoretical 2.0 — measured 1.900, stable
to [1.73, 2.10] across CFL and `t_final`, rejecting the old 1.0 by 0.7.
MADD-VER-001 had the same shape of problem, a 5% threshold sized around the
1.8% the defect produced; the error is now 1.59e-5 and the threshold is 1e-4.
Four more studies follow for the ODE nodes (MADD-VER-009 to 012), which take
the harness from 3 of the 11 node types to 7.  `SpringDamperNode` measures
**1.029**, `BallNode` **1.002**, `RigidBody2DNode` **1.000** and
`HeartPumpNode` **1.000**, each against a declared 1 in time; each declares
its order as a class attribute rather than the per-instance hook, because
none of the four has a constructor argument that selects a different
integrator.  Where the manufactured source goes in differs per node, and the
difference is itself a finding: `RigidBody2DNode` takes `force` and `torque`
directly, `SpringDamperNode`'s `anchor_position` and `HeartPumpNode`'s
`backpressure` enter the force and the outflow linearly so the source is
available in closed form, and `BallNode` **exposes no forcing input at all** —
its only boundary input is a collision surface, so its source has to go in
through the `gravity` parameter.  Twelve mutation cases (a mis-scaled
timestep, a timestep that drifts with the resolution, and a source frozen at
`t = 0`, across all four nodes) are each required to fail the ladder and name
the node; a thirteenth deliberately is not caught, and says so, because
swapping semi-implicit for forward Euler leaves the order inside the band.

That thirteenth case is the point of three more anomalies, none of which is an
order shortfall and none of which an order study could have found.
`BallNode` names forward Euler in its metadata and implements semi-implicit
Euler (MADD-ANO-011).  `HeartPumpNode` does the same, in a subtler way: it
advances the cardiac phase first and evaluates the inflow waveform at the
*end* of the step, while its own `derivatives()` — the right-hand side
`integrate_node` and `implicit_residual` use — evaluates it at the start, so
the node's two documented paths integrate waveforms one timestep apart
(MADD-ANO-012).  And `HeartPumpNode.update` casts `backpressure` to float32
unconditionally, pinning its coupling variable to single precision under any
caller precision: two backpressures differing by 1e-6 at 100 units give
bit-identical pressures, which floors a float64 ladder at about 6e-6 relative
error and silently demotes a weakly typed float64 pressure state until
`lax.scan` rejects the carry (MADD-ANO-013).  All three are pinned as strict
xfails.  The four nodes also gained the algorithm guides they never had.
### Grid convergence, where MMS cannot reach

The gap the section above ends on is closed rather than left open.  MMS needs
somewhere to inject a source, so it cannot reach `LBMPipeNode`, any node with
no natural forcing input, or most nodes a *user* writes.  The obvious fix — an
optional manufactured-source hook on `SimulationNode` — was proposed and
rejected: a generic source term is ill-defined for a lattice Boltzmann
collision operator (the forcing scheme changes the order of accuracy, so the
harness would measure the scheme) and for anything projecting onto a
constraint manifold, and an optional hook would not be implemented by the user
nodes it was meant to serve.  {doc}`/developer_guide/verification` records
the argument in full.

`maddening.testing.mms` instead gained a second mode over the same refinement
ladder.  A Richardson / Grid Convergence Index study compares three
successively refined solutions to **each other**: `solution_at(level)` returns
one scalar functional, and the harness returns the observed order of
convergence, the extrapolated limit and an error band on the finest solution.
It asks nothing of the node but the ability to refine, so it covers
`LBMPipeNode`, unforced nodes and user-written nodes alike.  The formulation
is the ASME V&V 20 procedure (Celik et al. 2008) with Roache's safety factor:
**1.25 where the asymptotic range is demonstrated, 3.0 everywhere else**,
which is the stricter reading of Roache (1994, 1998) — a node that declares no
order gets a wider, honest band rather than a narrow, unsupported one.

Three details decide whether such a harness is worth having.  **Non-constant
refinement ratios** make the observed-order equation implicit, and assuming a
constant ratio anyway is the classic way to get this wrong: on a 10/17/40
ladder over a second-order rule the textbook formula reads 0.98, a perfectly
plausible first-order answer that would be believed.  The harness solves the
implicit equation instead, by Celik et al.'s fixed-point iteration with a
bracketed bisection behind it for the ratio pairs where that iteration
diverges.  **Non-monotone convergence** is classified from the sign of the
solution differences and reported by name — oscillatory, divergent,
oscillatory-divergent — never averaged into an order.  And **a node that
silently fails to refine** returns the same number three times, which every
formula in a GCI divides by; identical solutions are refused as a stagnant
ladder, and that check runs *before* the node's declaration is read, so a node
that both declares nothing and ignores refinement cannot skip its way to a
pass.

`LBMPipeNode` now has a convergence verdict for the first time
(MADD-VER-013).  The dimensionless shape factor `u_max/u_mean` of the steady,
uniformly forced pipe flow converges monotonically over a 12/16/24
cross-section ladder at an observed order of 1.49, with a Grid Convergence
Index of 10.7% at `Fs = 3.0`.  The honest part is what a fourth level says:
the 12/16/24 and 16/24/32 triples disagree about the order by nearly two full
orders of accuracy (1.49 against 3.35), because the pipe wall is a circle
staircased onto a Cartesian lattice with bounce-back and the effective wall
position jumps about as the resolution changes.  The study is **not in the
asymptotic range**, and outside it the index is not conservative but wrong in
the confident direction — the 16/24/32 triple quotes 1.2% around a solution
3.6% from the Hagen-Poiseuille answer.  A study shown to be outside the range
therefore fails rather than reporting a band, and the ordinary pipe study
carries `Fs = 3.0` and says in its own output that the range is undetermined.
The node's `NodeMeta` now claims no more than this study shows.
`discretization` no longer says "2nd-order in space and time"; it claims no
order.  The `tau` regime is `0.8` to `0.8`, the one value the study runs,
where it used to declare `0.501` to `2.0`, a range nothing ran.  The
Reynolds-number and grid regimes are the study's Stokes-regime runs on 12 to
32 cells per side.  A new limitation says that no benchmark covers the
multiphase mode, the tracer, partial fill, gravity or a localised actuator
disc.  `LBMNode` claimed more than its benchmarks run in the same way, and
is narrowed the same way: `tau` from `0.501`–`2.0` to `0.8`–`1.0` (MADD-VER-003
and MADD-VER-007 run at 0.8, MADD-VER-016 at 1.0), and the Reynolds number
from `0`–`100` to `0`–`0.21`, measured on MADD-VER-016's channel and
MADD-VER-003's pipe, both in the Stokes regime.  Code that reads
`validated_regimes` to gate its inputs will see the narrower ranges.

What GCI cannot do is stated with it: it never sees the true answer, so it
cannot catch a scheme that converges cleanly to the *wrong* limit.  MMS can.
The node-authoring contract is one line — **a node with a natural forcing
input can be MMS-tested; everything else falls back to GCI** — and where both
apply, MMS is the stronger test and GCI is the uncertainty statement on top of
it.

## Interface mappings

`add_edge(..., mapping=)` takes a `Mapping` (`apply` / `apply_T` /
`params_pytree`), applied before the scalar `transform`.  Its weights are
snapshotted into `gm.params["mappings"]` and passed as a traced input on
every step, so `jax.grad` reaches them — through an IFT coupling group too —
and a replaced matrix needs no recompile.  `StaticLinearMapping` comes with
four factories: `rbf_mapping` (polynomial augmentation on by default, a
solve rather than an `inv`, kernel-relative ridge, and
`mode="consistent"|"conservative"` where conservative is the transpose of
the reverse consistent map and preserves totals exactly),
`nearest_neighbor_mapping`, `projection_1d_mapping` and `matrix_mapping`.
Matrices are assembled and solved in float64 on the host at construction.

Every kernel is gated by a patch test (constants and linear fields
reproduced across random non-conforming point sets, to float32 round-off and
1e-8 in float64) and a conservation test.  Mapping weights are
`trainable=False` by default; opt in with
`set_param_spec(edge.key, "H", ParamSpec())`.  As a side effect the closure
API `rbf_interpolation` now shares the same matrix construction, and its
multiquadric constant test went from `atol=0.1` to round-off.

Worked example: {doc}`/user_guide/moving_a_sampling_onto_an_edge` moves a node's own sampling of a cell-centred grid onto a `multilinear_grid` edge.

### Coordinates a factory cannot map are refused

The factories build their operator once, on the host, from coordinates, and
each computed *something* from coordinates its formula does not cover
(MADD-ANO-192).  The closure factories of `interface_mapping` have done so
since 0.1.0; the `*_mapping` factories inherited it.

| factory | input | what it returned |
|---------|-------|------------------|
| `projection_1d_mapping`, `conservative_projection_1d` | descending boundaries, on either grid | a matrix of zeros |
| | boundaries out of order, `[0, 2, 1, 3]` | rows summing to 4/3 |
| | a repeated target boundary; an infinite one | a zero row |
| `nearest_neighbor_mapping`, `nearest_neighbor_1d` / `_2d` | a NaN source coordinate | that source, for every target (`argmin` returns a NaN) |
| | a NaN target | source 0 |
| | `(n, 1)` targets against `(m, 2)` sources | a broadcast, not an error |
| `linear_interpolation_1d` | descending source coordinates | `x -> [3, 0]` where it is `[0.5, 2.5]` |
| `rbf_mapping`, `rbf_matrix`, `rbf_interpolation` | a NaN point | a matrix of NaN |
| | an infinite target, no polynomial | a row of zeros |
| `matrix_mapping` | a NaN or infinite entry | accepted |

Each is now a `ValueError` that names the argument and the first offending
index, for example `source_boundaries must be strictly increasing, but
source_boundaries[1] = 2.0 is not greater than source_boundaries[0] = 3.0`.
**Nothing is sorted or reversed for you**: the field keeps its cell order, so
a silently reversed boundary array would map the wrong cells.  A mapping
rebuilt by `GraphManager.from_dict` or `load_graph_from_usd` goes through the
same factory, so a config or a stage holding such a spec raises
`MappingRebuildError` naming the edge.

What is *not* refused, because it was not wrong:

- **grids that do not cover the same interval.**  A cell outside the other
  grid is treated as empty.  The projection then preserves the integral when
  the target grid covers the source grid, reproduces a constant on every
  target cell the source grid covers, and is a matrix of zeros for two grids
  that share no interval.  `projection_1d_mapping`'s docstring now states
  these conditions instead of an unconditional "integral-preserving"; nothing
  warns of two grids in different units or frames;
- **coincident points.**  Nearest neighbour takes the lowest index among
  equidistant sources; the RBF ridge shares the weight between coincident
  sources (with `ridge=0` the solve raises `LinAlgError`, a `ValueError`);
- **an empty set to evaluate at**, which gives an operator with no rows;
- **a repeated coordinate in `linear_interpolation_1d`**, whose zero-width
  interval is skipped, and targets outside the grid, which are clamped.

An accepted input gives the operator it gave before, bit for bit: 24
operators are pinned by digests taken on the tree before the checks, and
every RBF kernel against a transcription of the unchecked construction.
Coordinates handed to a closure factory as JAX tracers, and a traced matrix,
have no values to check and are taken as given.

### Every reader of an edge applies it as the step does

The step turns an edge into a boundary input by one rule: the mapping, then
the transform, additive edges summed, and an external input in place of
whatever edges delivered to its field.  Three readers outside the step had
rules of their own (MADD-ANO-193), and a node that stores the input it is fed
shows the difference:

| reader | edges into the input | it gave | the step feeds |
|--------|----------------------|---------|----------------|
| `check_conservation` | `x` mapped by a 2 x 3 matrix | `[1, 2, 3]` | `[5, 9]` |
| | `x` and, additive, `100 x` | `[1000, 2000]` | `[1010, 2020]` |
| | a flux output of another node | nothing (the edge was dropped) | the flux |
| `DatasetGenerator` | `x` mapped by a 2 x 3 matrix | samples of 3 | samples of 2 |
| | `x` through `v -> v[::-1][:2]` | a `(2, 3)` array (time was indexed) | 3 samples of 2 |
| | `x` and, additive, `100 x` | `100 x` | `101 x` |
| `resolve_boundary_inputs` | `x`, and an external input on the same field | `x` | the external input |

The additive and transform rows are as old as 0.1.0; the mapping rows came
with mapped edges.  All three now resolve through one private method,
`GraphManager._boundary_inputs_from`, which calls the step's own
`_apply_edge`; the dataset generator maps it over the history's sample axes,
so a mapping or a transform sees the field it sees in the step.

`check_conservation` changed in four more ways, each from a silent wrong
value to a value or an error:

- an edge that reads a **flux output** takes its source's flux at the same
  state (a Dirichlet-Neumann pair now compares the fluxes the Neumann side
  was actually given);
- two nodes that read **each other's** flux outputs (Robin-Robin) are
  refused with a `ValueError`: those inputs are what the coupling iteration
  converged to and cannot be rebuilt from a state;
- a flux name the node **does not report** is a `KeyError` (it read `0.0`,
  so two misspelt names were "conserved"), and so is a node missing from the
  state passed in (its edges were skipped);
- node constants and mapping weights are read from **`gm.params`**, as the
  step reads them.

It still takes every value from the one state it is given, where the step
mixes time levels; its docstring now says so.

Two things were found on the way.  One is **not** fixed:

- `DatasetGenerator` rebuilds inputs from the state history, and pairs a
  **forward** edge's input with the source's state *before* the step, where
  the step read it after (MADD-ANO-194, since 0.1.0; exact only for inputs
  that do not change, and for back edges).  Deferred to 0.5.0.

The other was fixed after it:

- `convergence_norm="interface"` and the precision floor read a mapped edge's
  source field through its transform, without the mapping (MADD-ANO-195,
  never released).  They now read every interface edge as the step delivers
  it: see [The interface norm reads a mapped edge as the step delivers
  it](#the-interface-norm-reads-a-mapped-edge-as-the-step-delivers-it).

The memory estimate of `POST /surrogate/train` follows the dataset: a mapped
input is counted at its `n_target`, not at its source field's size.

### Serialisation — `MappingSpec`

Mapped edges used to be refused by every writer.  Each factory now attaches
a `MappingSpec` — kind, hyper-parameters and *references* to its point sets,
never the weights — which `to_dict` / `from_dict`, the config helpers and
`save_graph_to_usd` / `load_graph_from_usd` (attribute
`maddening:mappingSpecJson`) carry.  A point reference is
`{"node": name, "field": key}`, `{"asset": "<file>.npy|.npz"}` (relative to
the config or stage directory, no absolute paths or `..`), or an inline list
of at most 64 points.  On load the mapping is rebuilt by the same factory,
weights bitwise equal, and registered in `params["mappings"]`; a checkpoint
loaded afterwards keeps its possibly-trained weights.  A mapping without a
complete spec is refused by the writers with a message naming the argument
to pass — `to_dict(strict_mappings=False)` is the display escape hatch, and
the REST `GET /graph` uses it.  The FMI exporter is unchanged: mapping
weights never reach the FMU.

The design decision worth recording is that the spec stores *references and
a content hash*, not point data.  A `sha256` of the array the factory was
given lets the writers refuse a stale reference — a removed node, a field
that changed — naming the edge, and lets the rebuild refuse a reference that
resolves to different points rather than silently building a different
operator.  Asset loading is correspondingly strict: the path must resolve
under `base_dir` (symlinks to files and to directories no longer escape it),
the `.npy` header or `.npz` directory entry is checked against
`MAX_ASSET_BYTES` (256 MiB) and the file's own size before anything is
allocated, and non-numeric dtypes are refused.  Every failure while
rebuilding one edge's mapping — including `BadZipFile`, `TypeError`,
`JSONDecodeError` from a corrupt attribute, `OSError` and `MemoryError` —
surfaces as a `MappingRebuildError` naming the edge, with the cause chained.

An inline point set with no `"dtype"` is read as `float64`, and the
coercion used to change some values without a word: an `np.clongdouble`
(or any complex) payload lost its imaginary part with only a NumPy
`ComplexWarning`, a Python int above 2**53 was rounded (9007199254740993
became ...992) and a `decimal.Decimal` went through `float()`.  Each is now
refused before the coercion, as `np.longdouble` already was, with a message
saying what to do instead: pass the real parts or the two parts as real
columns; add `"dtype": "int64"` to keep integer points exact; add
`"dtype": "float64"` or convert first to accept the rounding.  Complex
values are refused with or without a `"dtype"`, because the imaginary part
is lost rather than rounded.  Values `float64` holds exactly (2**53, powers
of two beyond it, `Decimal("0.5")`) and every other input are unaffected.

Guide: {doc}`/algorithm_guide/coupling/interface_mapping`, "Serialisation".

### Registering a mapping kind

*Experimental.*  The four kinds above were a closed set: three tables in
`mapping_spec` and a `getattr` on the module that defines the factories, so
a mapping from another library could not survive `to_dict` / `from_dict` or
a USD round trip.  They are now entries of one registry
(`maddening.core.coupling.mapping_registry`), which `MappingSpec` validation
and the rebuild read, and `register_mapping` adds a kind to it:

<!-- snippet: no-run, reason: fragment: the decorator on a factory whose body the guide gives in full -->
```python
from maddening.core.coupling.mapping import register_mapping

@register_mapping("inverse_distance",
                  arrays=("source_points", "target_points"),
                  hyperparameters={"power": float, "normalise": bool},
                  references={"source_points": "source_ref",
                              "target_points": "target_ref"})
def inverse_distance_mapping(source_points, target_points, *, power=2.0,
                             normalise=True, source_ref=None, target_ref=None):
    ...
```

`arrays` are the factory's array arguments (what a spec refers to by
reference), `hyperparameters` every other argument a spec may carry with
its type (`str`, `bool`, `int`, or `float` for a finite real), and
`references` the keyword that carries each array's reference
(`"<array>_ref"` when omitted).  The rebuild calls `factory(**arrays, **hyperparameters,
**references)`.  The declaration is checked against the factory's signature
at the decorator; a taken name -- a built-in kind's, or another factory's --
is a `ValueError`, the same factory registered again is a no-op, and nothing
removes a kind.

**A file can only name a kind.**  The lookup is a dictionary read.  Nothing
imports a module, follows a dotted path or discovers an entry point because
of what a config, a stage or a checkpoint contains; a kind exists only
because the running program imported the code that registered it, and a file
naming one this process has not registered is a `MappingRebuildError`
listing the kinds it has.  A registered kind's hyper-parameters are checked
against their declared types before its factory is called, and the limits
on point references (the asset size cap, the inline limits, the accepted
dtypes, the containment of asset paths, the content hash) hold for it as
for a built-in kind, because they sit in the reference resolver.  The
coordinate checks of "[Coordinates a factory cannot map are
refused](#coordinates-a-factory-cannot-map-are-refused)" do not: each
built-in factory makes them itself, so a registered factory is handed
whatever its references resolve to, and geometry its formula does not cover
is its own to refuse.  Whatever
the factory raises is reported as a `MappingRebuildError` naming the edge
and the kind.  Two failures of the built-in kinds join it: an `ImportError`
while rebuilding any kind (a factory that needs a package the environment
lacks) is now reported with its edge, and a real hyper-parameter given an
integer too large to be a float64 -- valid JSON, which puts no limit on an
integer's digits -- is refused as "must be finite", where `MappingSpec` used
to raise an `OverflowError` no loader wrapped.  What a registered factory
returns is held to the spec it was built from -- a
`Mapping` of that kind, carrying a `MappingSpec` that records the
references and the hyper-parameters it was given -- so a reloaded graph is
saved as the recipe that was read.

**What a mapping may put into `gm.params["mappings"]`.**  Checkpoints, `POST
/checkpoint/load`, system identification, the FMU's state archive and
`to_dict()`'s "live weights differ" warning all read a mapping's entry as a
flat table of arrays, and until now nothing checked that it was one.  For a
mapping of any class other than `StaticLinearMapping`, `add_edge` refuses a
`params_pytree()` that is not a plain `dict` (empty for a mapping without
weights) from Python identifiers to concrete, finite, floating-point JAX
arrays of any shape, the same on every call.  Every weight is then frozen by
default and opts in to a fit by name, is carried by a checkpoint under its
own name, and is fixed in an exported FMU, like `H`.  A registered factory
may also simply return a `StaticLinearMapping` (pass `kind=`).

The probes behind that rule found what a hand-written `Mapping` class could
already do to a 0.4.0 development build (**MADD-ANO-196**, carried by no
release: interface mappings are new in 0.4.0): the graph kept the returned
table by reference, so for a mapping that hands out the dict it keeps, an
in-place edit of `gm.params` rewrote the mapping's own weights --
`reset_params()` then restored the edit, and `to_dict()` compared the live
weights with themselves and never warned that a config would drop them; a
weight whose name held a `/` was written to a checkpoint and silently not
restored from it; a NumPy or Python-number weight drew the "live weights
differ" warning on a graph whose weights nobody had changed (after its first
step, or after loading its own checkpoint); a NaN weight made the FMU refuse
its own snapshot.  The snapshot now copies the table and `add_edge` refuses
the rest.

**The registry changed nothing that existed.**  The built-in kinds write and
rebuild exactly what they did: a capture of their config text, USD attribute,
weights and refusal messages taken on the tree before the registry
(`tests/core/builtin_mapping_pins.py`) is reproduced value for value, and
the mapping suites pass over the built-in kinds as they were.  (The capture
holds geometry the factories accept; the coordinates they now refuse are a
separate change, described above.)  The same
suites -- the hardening and serialisation tests, the USD round trips, the
FMU archive test, the checkpoint-load refusals, the writes that move a
referenced point set, and the property strategies every round-trip, replay,
write-door and sysid property draws from -- now run over registered kinds
too.  Not yet: a sharded graph and 16-bit weights of a registered kind are
not exercised (`mapping_claims.yaml`, MAP-007 and MAP-009).

Guide: {doc}`/algorithm_guide/coupling/interface_mapping`, "Registering your
own mapping kind".

## FMI 3.0: the gate closes

### C wrapper, TCP bridge and packaging

`src/maddening/fmi/c/maddening_fmu.c` implements the FMI 3.0 co-simulation
entry points — instantiate, initialise, `DoStep`, `Get`/`Set` for every
numeric type, FMU state get/set/serialize, reset, terminate; model exchange
and scheduled execution refuse — using nothing but libc.  Every call is
forwarded as a length-prefixed JSON message over TCP to
`maddening.fmi.tcp_bridge.FmuTcpBridge`, which maps value references onto
the sidecar's inputs, outputs and parameters and runs the master-step loop.
`maddening.fmi.package.build_fmu_binary` compiles the wrapper against the
vendored (BSD-2) FMI 3.0 headers, and `write_fmu` packages
`modelDescription.xml`, the binary and `resources/endpoint.txt` into a
`.fmu`.  `build_model_description(..., model_identifier=)` emits the
`<CoSimulation>` element.

Verified end to end: FMPy's `simulate_fmu` drives the compiled FMU through
the bridge and reproduces `gm.run_scan` with a set parameter and a driven
input.  ZMQ is not required — a ZMQ transport can carry the same payloads
later.

The choice of TCP/JSON over a shared-library-embedded interpreter is what
keeps the C wrapper dependency-free and the Python side a normal process;
the cost is a wire hop per call, which is what protocol 2 is about.

**A bridge serves one FMU instance, and a reconnect no longer races the one
that left.**  The instance slot is released by the departing connection's
own worker thread, once its socket has reached EOF, so a master that calls
`fmi3FreeInstance` and then `fmi3InstantiateCoSimulation` — which is what
every master does between runs — could be answered *"bridge already serves
an FMU instance"* with no second instance anywhere.  Measured with a direct
reconnect probe, two workers contending for one CPU: 28 % of 300 immediate
reconnects refused, 0 % after the fix; on an idle machine it was 0 in 500
either way, which is why it surfaced as an intermittent test rather than as
a report.  Nothing on the server side can close the gap — the new
connection can always arrive before the old worker is scheduled — so the
wait is on the acquiring side: a new connection now polls for the slot for
up to two seconds before it is refused.  "Refused, not blocked" is
unchanged; a genuine second instance still gets its error reply, two
seconds later.

### Protocol 2 — binary frames

Bit 31 of the four-byte length prefix marks a binary frame
(`[u32 BE header_len][header JSON][raw bytes]`).  After a
`{"op":"hello","protocol":2,"binary":true}` the bridge answers `get` /
`get_state` with raw little-endian float64 or raw `npz` bytes and accepts
binary `set` / `set_state`, so values no longer go through `%.17g` /
`strtod` and state blobs no longer through base64.

**Measured: a 10⁶-element `get` takes 2.3 s as JSON and 27 ms as binary over
loopback** — eight bytes per value instead of a decimal round trip.

The C wrapper negotiates it and falls back to JSON against a bridge whose
hello lacks `protocol`; a JSON-only client sees protocol-1 behaviour
unchanged, except that the hello reply gains `protocol` and `binary`.  An
unknown higher protocol is refused at hello.  All validation — value
references, bounds, read-only, state token/shape/size, malformed frames as
error replies — applies to both forms, and the C side checks the header
count against the raw length and the caller's array before any copy,
refusing flagged lengths over the 64 MiB limit before allocating.
`FmuTcpBridge.binary_frames_served` / `binary_frames_received` count the
traffic.  The user guide's "Wire protocol" section in
{doc}`/user_guide/fmu_export` is the reference.

### Multi-clock export

`build_model_description(multi_clock=True)` emits one FMI 3.0 `<Clock>`
(`intervalVariability="constant"`, `intervalDecimal=<dt>`) per distinct node
timestep among the exported nodes, and tags every exported output and
external input with its node's clock (`clocks=`, `variability="discrete"`;
clocked outputs are not initial unknowns), so an importer knows a node on a
coarser rate only changes on its ticks.  Clocks are `clock_<k>` in order of
increasing interval, and every interval is a whole number of graph steps.
With the default step size, one graph step, the fastest clock equals it when
the node timesteps divide each other; nodes at 0.02 and 0.03 run on a 0.01
step under clocks of 0.02 and 0.03, and a `default_step_size=` given
explicitly need not equal or divide any clock.  It is off by default — a
default export carries no clock — and validated with FMPy
(`validate=True, validate_model_structure=True`).

### What the audits found here

The protocol-2 audit turned up a fuzz harness that had been green for the
wrong reason.  The thread-free harness closed its fake server before the
client sent its request, so every exchange failed with `EPIPE` and the reply
parsers (`bridge_xfer`, `hdr_count`, `parse_binary_values`, `parse_values`,
the raw-state path) had had **no fuzz coverage at all** — the seeded
ASan/UBSan runs, the valgrind run and the libFuzzer campaign all passed
without ever reaching them.  The server end is now half-closed
(`shutdown(SHUT_WR)`), a third of the replies are well formed for the
operation under test so the success paths are reached, and the harness is
self-checking: the wrapper counts each parser path when built with
`MADDENING_FUZZ_COUNTERS` (off in the shipped FMU), the standalone run fails
when any path was never reached, and the test suite asserts the counts.
**Measured with gcov over seeds 1/7/12345 × 3000 iterations: coverage went
from 26 % to 57 % of the wrapper's lines.**

Framing, validation and lifetime fixes from the same and later rounds: a
reply over the 64 MiB limit (now enforced for JSON replies too, which could
previously announce 2 GiB and get it allocated) or cut short by the peer
drops the connection, so a later call returns `fmi3Error` instead of parsing
stale bytes — an unanswered `fmi3DoStep` used to return `fmi3OK`.  Values
are narrowed to the variable's dtype before the finiteness check, so a
float32 input set to `1e308` is no longer stored and read back as `inf`.  A
request nested too deeply is a malformed-request error reply rather than a
dropped connection with a thread traceback.  A `send()` to a closed peer
raised `SIGPIPE` and killed the importer's whole process; sends now use
`MSG_NOSIGNAL` (`SO_NOSIGPIPE` on macOS) and the call returns `fmi3Error`.
A communication step that is not a whole multiple of the master timestep is
refused rather than silently snapping the physics while reporting `t + h`,
and the FMU advertises a fixed communication step.  A `set` request is
atomic across parameters and inputs.  A second FMU instance on one bridge
gets a clear error instead of blocking.

The libFuzzer campaign itself had a memory problem worth recording: going
thread-free (preloading the fake reply into the socket instead of spawning a
thread per iteration) is what made the campaign reach 7.5 GB RSS and get
OOM-killed on the CI runner.  It is now built with UBSan only — the ASan
build reported about 8 GB RSS at `INITED` with 25 MB of live heap, a
host-accounting effect rather than a leak — and guarded by a per-allocation
`-malloc_limit_mb` instead of an RSS limit.  Memory safety of the same
harness stays covered by the ASan seeded runs and valgrind.

The confirmation audit after the freeze found four more, each with a
reproducer.  **`FmuTcpBridge.stop()` joined each thread for five seconds and
returned regardless**, so a worker still inside a sidecar step (the first
step of a large graph spends its first call compiling) went on to replace
the sidecar's state and the bridge's time after `stop()` had returned,
still holding the instance slot, with nothing logged.  Every commit into the
model (`step`, `set`, `set_state`, `reset`) now happens under the bridge's
lifecycle lock after the slow work has run on local copies, and is refused
once `stop()` has begun; a communication step's sub-steps run on a local
state and commit once.  `stop()` waits one five-second budget over all its
threads and logs a warning naming each thread still alive and the op it is
serving.  **`set` and `get` looked value references up with `int()`**, so
`10.9`, `10.4` and `"10"` wrote variable 10 and answered ok, and `true` read
variable 1 (`time`); anything but an integer is now the usual error reply.
**`FmuSidecar.set_fmu_state` checked only the leaves a snapshot carried**: a
snapshot missing a node restored and every later step raised `KeyError`, an
extra field was installed, a missing parameter failed the next step's
completeness check.  It now compares the node, field and parameter key sets
first, with the refusal the bridge's `set_state` gives for the same archive
(one shared function), and a snapshot that carries no parameters is refused
by a sidecar that has them, as the bridge refuses such an archive.  Fixing
it found a fifth: **the bridge rebuilt a restored state from the archive's
members, so a node with no state fields vanished from its own
`get_state`/`set_state` round trip** and every step after it failed with
`KeyError`; both restore paths now rebuild on the live model's skeleton.

### What the round-4 audit found in the FMU export

**A second FMU instance on one bridge started where the first one ended**
(`MADD-ANO-083`, critical, never released).  FMI starts every instance at
its `modelDescription.xml`'s start values, and a master frees the instance
and instantiates a new one between runs.  FMPy's `simulate_fmu` does it on
every call.  Claiming the bridge's instance slot reset nothing, so the second
instance inherited the first one's state, tuned parameters, pending inputs
and time, every call `fmi3OK`.  Two `simulate_fmu` calls with identical
arguments gave two answers (the second started at the first one's final
spring position), and a third with no start values computed with the first
one's stiffness of 45 against an advertised 30.  A connection that claims
the slot is now a new instance.  The bridge resets it to the state it was
built over, every parameter at the description's start value, every input
at zero and the time at zero, the same commit `fmi3Reset` makes.  A sidecar
built with other parameter values than its description advertises is
brought into line, with a `UserWarning`.  This also closes the path by which
MADD-ANO-023's stranger tampered silently: writes made before an instance
begins no longer reach it.

**An input the FMU did not export was not zero-filled** (`MADD-ANO-084`,
major, never released).  `GraphManager.step` zero-fills an omitted input,
but the bridge passed only the inputs its description exports, and
`FmuSidecar.step` called the compiled step with whatever it was given.  A
`BallNode` whose `table_position` was left out of the export took its "no
table" branch and fell through the floor: −2.178 after 0.8 s, where the
graph bounces at +0.4635.  `build_model_description` now lists the inputs
`selected_inputs` leaves out in `ModelDescription.held_inputs`, warns that
they will be held at zero, and refuses a name the graph does not declare.
The bridge holds those inputs at zero.  `SidecarConfig.input_resolver` (pass
`gm._resolve_external_inputs`) makes `FmuSidecar.step` complete and check
its inputs as the graph does, misspelt names included.  A sidecar behind a
bridge that was given no resolver gets one built from the description.  At
v0.3.0 and v0.3.1 the sidecar agreed with the graph, because the graph
did not complete a partial dict either.

**The advertised bounds held only if the sidecar had been given
`param_specs`** (`MADD-ANO-085`, major, never released), which defaults to
`None`.  The XML said `elasticity` was
in [0, 1], yet a `set` of 1.5, or an archive installing −3.0, answered ok.
The bridge now adds a bounds-only spec, from its description's `min` /
`max`, for every exported parameter its sidecar has no spec for, so `set`,
`set_state` and the sidecar's own `set_params` all hold the envelope the
XML advertises.

**Time.**  `fmi3EnterInitializationMode`'s start time stayed in the C
wrapper, so `time` read 0.0 until the first step.  It now reaches the
bridge (op `initialize`, accepted until the first step).  And `fmi3DoStep`'s
communication point was trusted: a jump from 0.01 to 100 advanced one master
step and labelled the state 100.01.  A point more than a millionth of a
master step from the FMU's time is now `fmi3Error` with nothing advanced,
and one inside that tolerance is adopted, so the importer's clock and the
FMU's never drift apart.

**Coercion.**  `set` stored the JSON string `"45"` and `true` as numbers,
`step` parsed `dt="0.01"` and `dt=true` (a hundred master steps of 0.01 s),
an archive without `_time` restored at t = 0, and a state member stored as
the string `"1.5"` restored as 1.5.  `FmuSidecar.set_params` took `"45"`
and `True`.  All are refused now.  `FmuSidecar.set_fmu_state` refuses a
snapshot carrying parameters into a model with no parameter tree, with the
words the bridge's `set_state` uses (it used to ignore them).

**The C wrapper.**  `fmi3Get*` answered `fmi3OK` with the surplus dropped
when a reply carried more values than `nValues`; it now refuses any count
but `nValues`.  An empty instantiation token skipped the token check, and
is now a mismatch.  A send that failed part-way returned `fmi3Error` but
kept the connection, with half a frame on the wire for the bridge to read
the next request into, and a signal (`EINTR`) counted as a failure.  The
send is now retried on `EINTR`, and any other failure drops the connection.
**Every FMI call took at least 40 ms on Linux**: a frame goes out as two
writes, and with Nagle's algorithm the second waited for the bridge's
delayed ACK.  With `TCP_NODELAY`, an 80-step FMPy run of 245 calls took
0.02 to 0.08 s instead of 10.1 s.  This was measured on 2026-10-02, on three
pinned cores of a shared machine (CPU, jaxlib 0.11.0).  The 40 ms is
Linux's delayed-ACK timer, not load.  And the sanitized FMPy test asserted
nothing about its hostile bridge (`got["hostile"] != "returned" or True`).
It now proves that each kind of hostile reply was served and ended in an
FMI error.

### What the round-5 audit found in the FMU bridge

**`master_dt` is the graph's step, and the bridge checks it**
(`MADD-ANO-100`, major, never released).  `FmuTcpBridge` stored any
`master_dt` it was given, and a `doStep` of `h` ran `h / master_dt` graph
steps and reported `t + h`.  A 0.01 s graph served with `master_dt=0.005`
therefore reported t = 0.05 s for a state at 0.10 s.
`build_model_description` now records the graph's step as
`ModelDescription.graph_timestep` (`gm.timestep`), whatever
`default_step_size` advertises.  The bridge refuses a `master_dt` other than
that step, and an advertised step that is not a whole number of graph steps.
With `gm.timestep` now the step a graph takes, a multi-rate or sub-cycled
graph served at `master_dt=gm.timestep` is the graph at the time it reports.

**A Boolean takes true / false or exactly 0 / 1** (`MADD-ANO-101`, major,
never released).  The value check shared by `set`, `set_state`,
`set_params` and `set_fmu_state` cast any number to a Boolean by its
truthiness: `set gate.open = 0.5` read back as true, and so did an archive
holding 0.25.  Anything but 0, 1 or a boolean is now refused.

**Each FMI call names its type** (`MADD-ANO-102`, major, never released).
The C wrapper carried every width as a double, so `fmi3SetBoolean` on a
Float32 parameter stored 1.0, and `fmi3GetInt32` on a Float32 output
returned 0 for 0.5, both `fmi3OK`.  Every get and set request now names
the FMI type of the function that made it, and the bridge refuses a
variable of another type, as FMI 3.0 requires.  **An importer that read a
Float32 variable with `fmi3GetFloat64` now gets `fmi3Error`**: read it with
`fmi3GetFloat32` (FMPy's `simulate_fmu` already does).  The wrapper also
refuses a reply value its C type cannot hold, so no conversion is undefined
behaviour.  The Int64 and UInt64 setters refuse a value a double cannot
carry exactly.

**An archive carries every pending input** (`MADD-ANO-103`, major, never
released).  An FMU-state archive missing an `i/` member used to restore with
that input at zero.  It is now refused, as a missing field or parameter
already was.

**Smaller refusals.**  A `set` that names one value reference twice is
refused; the first value used to be dropped.  A numeric set of a `<Clock>`
is refused; it used to be dropped with `ok`.  A step size and a
communication point now share one absolute tolerance, a millionth of a
master step.  The step size was allowed a millionth per master step it
covered, so a step of `3 * dt * (1 + 5e-7)` was accepted and the next
legal `doStep` refused.  `fmi3SetFMUState` checks the whole frame against
the 64 MiB limit, header included.  A blob of exactly the limit passed the
old check, and the bridge then dropped the connection.  One `step` request
takes at most `max_steps_per_request` graph steps (100000 by default), and a
long step stops at the first graph step after `stop()`, committing nothing.
The C wrapper no longer waits for ever: every receive and send times out
after `MADDENING_FMU_TIMEOUT` seconds of silence (600 by default, 0 for no
deadline).  Connecting and the hello get at most 30 s, so an endpoint that
accepts and never answers fails instantiation instead of blocking it.

### What the round-6 audit found in the FMU export

**Descriptions pass FMI 3.0's schema.**  Every input's `start` was written
as `"0.0"`, and every parameter's through `repr(float)`, whatever the
variable's type.  The schema refuses that for a Boolean or an integer, so
FMPy's `validate_fmu` reported it and `simulate_fmu`, which validates by
default, refused the FMU.  Starts and bounds are now written in the type's
own form: `true` / `false`, integer literals, an integer's bounds rounded
inward, and no `min` / `max` on a Boolean.  A Boolean or integer input or
output is `discrete`, as FMI 3.0 requires and FMPy's structure check then
asked.  No release carried this: v0.3.1 wrote no `start` at all.

**The wrapper ignores the importer's locale.**  `%.17g` and `strtod` follow
`LC_NUMERIC`.  An importer running under a locale whose decimal point is
`','` therefore sent `"dt":0,01`, and every `fmi3DoStep` and every
non-integer start time failed.  Each instance now formats and parses numbers
in its own C locale (`newlocale` / `uselocale` on POSIX, the `_l` functions
on Windows).  The C unit tests run again under a `','` locale, which
`localedef` builds into the test directory where none is installed.

**An archive cannot install mapping weights** (`MADD-ANO-119`, major, never
released).  Interface-mapping weights are not FMI variables, but an
FMU-state archive could replace them, and the FMU then computed a coupling
its description does not describe.  Every mapping leaf is now fixed on both
restore doors.

**The rest.**  The instantiation token now covers every variable's start,
bounds, variability, value reference and clocks.  An FMU packaged from a
graph with other parameter values used to instantiate against the bridge
while advertising other starts; now it does not.  After `fmi3Terminate` the
bridge refuses `fmi3DoStep`, `fmi3Set*` and `fmi3EnterInitializationMode`
until `fmi3Reset`, as FMI's Terminated state requires.  A step after
terminate used to advance the model.  `fmi3GetClock` and `fmi3SetClock` are
`fmi3Error`: FMI allows them only in Event Mode, which this FMU does not
have.  They used to answer `fmi3OK` for any value reference.  The wrapper's
clock, reported as `lastSuccessfulTime` when a step fails, now follows
`fmi3SetFMUState`, whose reply carries the restored time, and `fmi3Reset`,
but only once the reset succeeds.  `FmuTcpBridge(idle_timeout=...)` replaces
the fixed five-minute idle limit; `None` means no limit.  A node whose name
holds `.params.` has settable parameters, because a parameter variable now
carries its `(node, key)` and the sidecar matches names exactly.

### What the round-7 audit found in the FMU export

An audit of the sysid and FMU claims inventory
(`docs/validation/sysid_fmu_claims.yaml`) found these.  None reached a
release: the parameter export, the bridge's time checks and the C wrapper
are new in this cycle.

**An FMU runs the model its graph runs** (`MADD-ANO-147`, major, never
released).  A structural `node.params` write after `compile()`, such as
`HeatNode`'s `stencil_order` 2 → 4, is taken into the graph only as a
pending recompile, and every graph entry point runs the new model.
`build_model_description` and the guide's sidecar wiring
(`step_fn=gm._compiled_step`) never looked.  The FMU ran the old model if
the step had been traced before the write, 6.35 K off the graph after 50
steps, and the new one if it had not.  Now `build_model_description`
refuses a graph that has changed since its compile, or was never compiled.
It refuses rather than compiling, because an export is an artefact and a
compile inside it would hide which model it froze.  `compile()` tags its
jitted step with its graph and compile.  `FmuSidecar` refuses a graph's step
once that graph has changed or been compiled again, and `FmuTcpBridge`
refuses a description or a sidecar step whose graph has.  A leaf write is
exported as written, as before.

**A `log` parameter without a lower bound advertises a `min`**
(`MADD-ANO-148`, major, never released).  `ParamSpec(transform="log")` with
no lower bound is measured from 0, and the graph refuses `p <= 0`.  The
description advertised no `min`, so a bridge whose sidecar had no
`param_specs` accepted and ran `mass = -1`.  The `min` is now the smallest
positive normal of the leaf's type, which the bridge enforces either way.
The built-in `SpringDamperNode.mass` is such a leaf, so its FMU
description changes, and its instantiation token with it.

**The time the FMU reports stays with the physics** (`MADD-ANO-149`, major,
never released).  The bridge's rounding slack on time was in ulps of at
least 1.0 s, so below a master step of about 1e-9 s the tolerance was that
floor, not a millionth of a step: 8.9e-4 of a 1e-12 s step was accepted.  And
since each communication point within the tolerance is adopted, an importer
whose every step size was a little long moved the reported time ahead of
the simulated time without bound: 1.6 master steps after 2000 steps at
1e-12 s.  The slack is now in ulps of the times involved, and the point
and the step's end are held to the simulated time to the tolerance plus a
few ulps of the time per step, which an importer's running sum stays inside.

**The stability filter covers inputs.**  An external input of an
`EXPERIMENTAL` node was exported, even with `include_evolving=True`, while
the node's outputs and parameters were left out.  Such an input is now
held at zero and listed in `held_inputs`, with a warning, and naming it in
`selected_inputs` is refused.

**The C wrapper holds FMI 3.0's state machine.**  `fmi3DoStep` before
`fmi3EnterInitializationMode` advanced the model.  After `fmi3Terminate`, a
zero-length `fmi3Set*`, `fmi3ExitInitializationMode` and `fmi3EnterStepMode`
answered `fmi3OK`.  The wrapper now tracks Instantiated, Initialization Mode,
Step Mode and Terminated, and refuses a call its state does not allow, with
nothing sent.  `fmi3EnterStepMode` and `fmi3UpdateDiscreteStates` (Event
Mode only), `fmi3EvaluateDiscreteStates` (not declared), the
configuration-mode pair (no structural parameters) and `fmi3SetTime`
(model exchange; it moved the wrapper's clock) are always refused.  An
importer that read or stepped straight after instantiation must initialize
first, as FMI 3.0 requires and FMPy's `simulate_fmu` already does.

**Wording.**  The guide and these notes said the fastest clock *equals* the
default step; it does only with the default step size, on timesteps that
divide each other, and every clock is a whole number of graph steps.
`build_model_description` and the parameters guide said the FMU's
directional derivatives are the real `jax.jvp`.  The sidecar's Python API
computes them that way, but the FMU binary does not provide them: the
description does not declare `providesDirectionalDerivatives`, and the C
wrapper's `fmi3GetDirectionalDerivative` returns `fmi3Error`.

### What the round-8 audit found in the FMU export

A further audit of the sysid and FMU claims inventory found these.  None
reached a release: the bridge is new in this cycle.

**The bridge enforces exactly the envelope it advertises**
(`MADD-ANO-167`, major, never released).  `set_param_spec` changes a spec
without dirtying the graph, so a description built before the change
advertised the old `min` / `max` while a sidecar built after it from
`gm.param_specs()` enforced the new ones, and the bridge kept the sidecar's
spec: it accepted `damping = 7.0` against an advertised `[1, 5]`, or
refused `0.5` inside an advertised `[0, inf)`.  The bridge now refuses to
start when a parameter's spec in the description's graph no longer gives
the advertised `min`, `max` or `unit`, or when the sidecar's own spec
enforces another envelope, and holds every write and restore to the
advertised bounds as well as to the sidecar's spec.  Build the description,
the sidecar and the bridge after the last `set_param_spec`.

**Time the clock cannot resolve is refused** (`MADD-ANO-168`, major,
never released).  The bridge's rounding slacks were ulps of the time, with
no cap, so at 1000 s with a 1e-12 s master step (an ulp is 0.11 of a step)
a `doStep` a whole step ahead was adopted, and a clock running 40% fast
ran 364 steps ahead of the physics in 1000.  Every slack is now at most a
tenth of a master step, so the reported time stays within a millionth plus
a tenth of a step of the simulated time.  A time whose 16 ulps pass that
tenth -- 32 s and beyond at a 1e-12 s master step, about 3.4e10 s at 1e-3 s
-- is refused at `fmi3EnterInitializationMode`, at a `doStep` that would
reach it and at `fmi3SetFMUState`, with a message naming the largest time
the master step resolves.  The cap has a cost, stated in the guide: an
importer that keeps a running sum of its step sizes is refused once its
rounding passes a tenth of a step (about 1.2 million steps from 1 s at a
1e-9 s step); one that computes `start + k * h`, as FMPy does, is not.

**A `get` is bounded by one reply frame** (`MADD-ANO-169`, major, never
released).  One 64 MiB request naming a scalar output 33.5 million times
took about 45 s and over 6 GB before its reply was refused.  A `get` naming
more value references, or variables holding more values, than a binary
reply frame carries (8 388 480) is now refused before anything is read.  A
repeated value reference is read once and answered at every occurrence, and
a JSON reply too long to send is abandoned while it is encoded.

**An open bound is advertised as a value the sidecar accepts.**  A float32
`logit` leaf under `(-1, 1)` advertised `max = nextafter(1, 0)`, whose
`(p - lo) / (hi - lo)` rounds to 1, so a sidecar with `param_specs`, and FMPy
through the wrapper, refused the FMU's own advertised max.  An open `log` /
`logit` bound is now advertised as the outermost value inside it that
`ParamSpec.check` accepts in the leaf's type: the next float in where that
is accepted (unchanged for most specs), two in for `(-1, 1)`.  The same fix
covers a bound in float32's subnormal-spacing band, which was advertised a
distance the arithmetic flushes to zero and taken by a sidecar without
specs.  Descriptions with such a leaf change, and their tokens with them.

### What the round-9 audit found in the FMU export

A ninth audit of the sysid and FMU claims inventory found five defects in
the export, and the batteries written to close them found two more.  Only
the last one below reached a release.

**A restore resumes the drift count** (`MADD-ANO-197`, major, never
released).  The bridge holds the time it reports to the start time plus the
master steps taken since.  `fmi3SetFMUState` set that reference to the
restored time and the count to zero, and the restored time is the
importer's own clock with whatever it has gathered: every restore forgave
the drift so far.  With a save and restore after every step, as a rollback
master makes them, step sizes 0.45 millionths of a step too long were never
refused (the third step without restores), and an honest running sum at
30000 s with a 1e-9 s step drifted 1.78 master steps in 4000.  The start
time and the count are now part of the FMU state: `fmi3GetFMUState` saves
them and `fmi3SetFMUState` resumes them, so a restore changes no later
reply.  An archive without them, which only a development build wrote, is
refused by name.  `fmi3EnterInitializationMode` still names a new start,
after a restore too.  The bound is also stated more exactly: the reported
time stays within a millionth plus a tenth of a step of the simulated time
*to the rounding of the float64 clock* (three ulps of the time, under 0.02
of a step wherever the clock resolves one); the battery met a running sum
0.0013 of a step past the round figure.

**An Int64 or UInt64 is never rounded on the wire** (`MADD-ANO-198`,
major, never released).  The wire carries every value as a float64, which
is not every 64-bit integer: above 2^53 it holds only the multiples of its
spacing.  A `get` converted the model's int64 regardless, so `fmi3GetInt64`
answered `fmi3OK` with 9007199254740992 for a model holding
9007199254740993, and a JSON `set` of 2^53 + 1 was stored as 2^53.  A `get`
of a variable holding such a value is now refused, naming the variable and
the integer (`fmi3Error` from the getter), and so is a `set` of one to an
integer variable; a value a float64 is crosses exactly at any magnitude,
and the FMU-state archive carries every value in its own type.  The value
check behind `set`, `set_state` and `FmuSidecar.set_params` no longer
compares through float64, where 2^63 equals the largest `Int64`.  This is a
limit, and a loud one: a 64-bit integer that is not a float64 cannot be
read or set through the FMU's variables.

**Negative zero keeps its sign on the JSON path** (`MADD-ANO-199`, major,
never released).  The wrapper wrote it as `-0`, which a JSON reader takes
for the integer 0, so `fmi3SetFloat64(-0.0)` stored `+0.0` on a connection
without binary frames and `-0.0` on one with them.  The wrapper writes
`-0.0`, and the bridge reads the literal `-0` as negative zero from any
client.

**`load_state` refuses an integer of the other signedness**
(`MADD-ANO-200`, major, never released).  The check added for
`MADD-ANO-138` cast an integer to the leaf's type and back, which between a
signed and an unsigned type of one width always returns the value it
started with: `-1` for a `uint32` field loaded as 4294967295, and
4000000000 for an `int32` one as -294967296.  It and the FMU's value check
now share one exact question (`maddening.core._exact_integers`), and a
complex value with an imaginary part is refused for a real field.  The
tagged releases kept the checkpoint's own dtype and did not wrap.

**`fmi3GetFMUState` reuses the state it is handed.**  FMI 3.0 has a
non-NULL `*FMUState` point to a state "that is no longer needed and can be
overwritten".  The wrapper allocated a new object over it on every call, so
a master that keeps one variable and saves before every step leaked one
whole state per call in its own process: 2.8 kB a call for one spring, 82
kB for a 20000-cell rod.  The same object is now returned, its memory
reused; `fmi3FreeInstance` frees the states of its instance that are still
live, as the standard has it free everything the interface allocated; and
`fmi3DeserializeFMUState` refuses a vector no frame could carry before it
copies it.  Free a state before its instance, not after.  (Not in the
registry: no value was wrong, no release carried the wrapper, and freeing
before each save avoided it.)

**`selected_outputs` refuses a pair that exports nothing**
(`MADD-ANO-201`, minor, since 0.3.0).  A pair naming a misspelt field, a
node the graph does not have or a node the FMU does not export was skipped
in silence, so `selected_outputs=[("spring", "positon")]` gave an FMU with
no outputs.  It is a `ValueError` naming the node and the field, as an
unknown `selected_inputs` name already was; an entry that is not a
`(node, field)` pair of names is refused too.

**The instantiation token covers the unit.**  The bridge refuses to start
once a parameter's unit has been re-declared, so the unit is advertised as
the bounds are, but the token left it out: an FMU packaged when stiffness
was in N/m instantiated against a bridge serving the description that says
kN/m.  The unit is hashed now, so every token changes; the one attribute
left out is a variable's description text.  Re-package an FMU built by an
earlier development build.

## Coupling and solvers

**Coupling groups now default to the early-exit solver**
(`CouplingGroup.solver="ift"`).  The fixed-point iteration is a
`jax.lax.while_loop` that exits as soon as the group's convergence norm
meets its threshold, for every `acceleration`, `iteration_mode`,
`convergence_norm` and `diagnostics` setting.  The legacy unrolled
`fori_loop` ran `max_iterations` passes regardless of convergence — step
cost was linear in `max_iterations`, and **80–90 % of iterations were dead
at typical convergence**.  `solver="fori"` is deprecated and will be removed
in the next minor release.  IQN-IMVJ cross-timestep Jacobian reuse now runs
inside the while_loop with the same shift-and-insert column convention.

**The IFT derivative rule is a `jax.custom_jvp`** where it was a
`custom_vjp`.  JAX derives reverse mode by transposing the linear tangent
rule through lineax, so one definition serves `jax.jvp` / `jacfwd` — the FMI
`FORWARD` directional derivative, previously a `TypeError` through coupled
steps — as well as `jax.grad` / `jacrev` and `jax.hessian`.

`sharded_cg` / `sharded_gmres` gained `differentiable=True`, routing the
solve through `lax.custom_linear_solve` so that `jax.grad` and `jax.jvp`
through the result — and the IFT adjoint of a coupling group whose node
solves with them — are exact linear-solve adjoints using the same backend
*and the same preconditioner* in the adjoint solve (`iters` is then -1).
It is off by default to keep the `STABLE` result contract.
**Changed:** a call without it now refuses to be differentiated (`jax.jvp`, `jax.grad`, `jax.jacfwd` and `jax.jacrev` through it raise a `ValueError`): pass `differentiable=True` to a solve you differentiate, or `jax.lax.stop_gradient` what goes into one that is meant to be a constant (MADD-ANO-266).
With the flag, `converged` describes the value's solve only: each derivative is one more solve from zero under the same `rtol` and `max_iters`, so give `max_iters` what a solve from zero needs, not what a warm-started one does (MADD-ANO-270).
`jacobi_preconditioner(diag)` and `block_jacobi_preconditioner(blocks)` are
the first users of the `preconditioner=` hook, and `backend="lineax"` now
refuses a preconditioner instead of silently dropping it.

`SimulationNode.domain_integral_axes()` lets a domain integral be reduced
over a subset of mesh axes, or not at all (per-shard values stacked), on
both sharded wrappers — so a body-surface integral living on some shards no
longer needs a full-mesh `psum`.

Other coupling surfaces: `CouplingGroup.strict_convergence` raises
(jit-safe, via `equinox.error_if`) when a group exits at `max_iterations`
unconverged, since the IFT gradient is then invalid; it is off by default.
Iteration count and residual are now always written to `_meta` under the
default solver, not only with `diagnostics=True`, and
`coupling_diagnostics()` reports `"converged"` per group.  `lineax`, which
the IFT Krylov adjoint solves with, is now a base dependency rather than an
`[ift]` extra — see [Dependencies](../release_notes/v0.4.0.md#dependencies).  `iqn_ils_update` takes a
keyword-only `have_prev` flag.

**The reported `residual` has a float32 noise floor, and near the fixed
point the floor is the whole value.**  Every convergence norm divides
`F(x) - x` by a scale, so a converged group's residual is a *cancellation*:
one unit in the last place of the state is a full-size contribution to it.
The two solvers run their passes in different loop constructs — `"ift"` in a
`lax.while_loop` so it can exit early, `"fori"` in a `lax.fori_loop` — and
XLA is free to compile those to differently rounded arithmetic.  It does: on
a subcycled multi-rate group the same map applied to the same input differs
by one ulp on two of ten float32 components, which is the difference between
reporting `1.04e-05` and reporting exactly `0.0`.  Both solvers returned the
same state and both reported `converged=True`.

So `residual` agrees between solvers to about `eps_f32 / rtol` under the
mixed and interface norms and `eps_f32 * sqrt(n)` under L2 — the
measurement's own resolution.  Above that floor, `converged`, `iterations`
and the returned state agree between the solvers: over a 480-configuration
sweep of that group, with tolerances above the floor, the verdict never
disagreed and the returned states agreed to 7.3e-07 relative.  **At or below
the floor the verdict can disagree too.**  On the mixed-rate spring pair at
`tolerance=1e-8` (L2 norm, O(1) fields), 14 of 308 `ift`/`fori`
configurations disagreed on `converged` from the third step on (measured
while fixing the waveform report, MADD-ANO-026).  Two consequences for users:
do not assert equality of residuals, or of verdicts, across solvers, backends
or JAX versions below that floor; and note that an L2 group whose `tolerance`
is at or below `eps_f32 * sqrt(n)` has a criterion made of rounding.  Pinned by
`tests/core/test_coupling_solver_equivalence.py`.

### The error estimate, and the flag that does not certify it

`coupling_diagnostics()` applies the group's threshold to an *estimate*
of the distance to the fixed point, `ω · residual / (1 - rho)`, rather
than to the residual.  That estimate bounds the distance only if four
conditions hold, and the flag beside it checked exactly one of them
while being called `bound_valid`.  **It is now `ratio_usable`**, and
`gradient_error_bound` — which is numerically `error_estimate` and
inherits every way that number can understate — is now
`gradient_error_estimate`.  Both old names -- spellings of earlier 0.4.0
development builds, which no release carried -- still read through 0.4.x,
emit a `DeprecationWarning` naming the replacement, and are removed in
0.5.0.  They are not in `keys()`, so a recorded report carries only the
new names.  `error_estimate` keeps its name: it already says estimate.

`ratio_usable=True` means the contraction ratio was monotone, finite and
had a non-zero predecessor, so the criterion used the estimate instead of
falling back to the pre-0.4.0 raw residual test.  It does **not** mean
the estimate bounds the error.  Three mechanisms break that inequality
and none of them is visible to this flag:

* **`rho` reads the mode that dominates the step, not the one that owns
  the remaining error.**  On a linear two-mode contraction, modes
  `(0.999, 0.2)` at `tolerance=1e-4`, the reported estimate is
  `9.19e-05` against a true distance of `1.12e-02` — **122x** — with
  `ratio_usable=True` and `converged=True`.  This is characterised, not
  merely unfixed: over the passes that matter, the failing sequence and
  a genuine single-mode decay at 0.2 are *the same sequence*
  (consecutive ratios 0.2000 and 0.2024, stationary to three figures),
  so no function of the residual norms separates them.  Recovering the
  slow mode needs the spectrum.
* **Over-relaxation** understated by `1/ω` — 0.51x at ω = 1.95.  **Fixed
  in this release**: the series now sums the steps the iterate takes.
  Over an 88-point (gain, ω) grid `est/true` runs 0.992 to 4.53, exactly
  1.0 wherever the relaxed iteration does not overshoot.
* **Aitken and IQN** carry a step scale the series does not.  Aitken
  understates by **2.04x** where its per-pass factor saturates at its
  2.0 clip; IQN by **4.5x**, by the first mechanism — a superlinear
  sequence reads `rho → 0`.  Both are documented rather than corrected.

Read `error_estimate` as strictly better than the residual — it is never
smaller than it, and never looser than the criterion it replaced — and
not as a certificate.  The certificate is the spectrum, and it is
measured in this release.

### `spectral_error_bound`: a bound, from the spectrum of `dF/dx`

Under `solver="ift"` with `diagnostics=True`, `coupling_diagnostics()`
now also reports `rho_spectral`, `spectral_error_bound` and
`spectral_usable`.  Eight Arnoldi steps on the Jacobian-vector product
the IFT adjoint already builds, at the returned state and in the
group's own norm, give the spectral radius of `dF/dx` and the resolvent
norm of the Krylov-compressed Jacobian; the bound is the residual times
the larger of that norm and `1/(1 − rho_spectral)`.  For a linear `F`
the error of any iterate is `(A − I)⁻¹` of its residual whatever the
accelerator did, so this **is a bound** where `error_estimate` is an
estimate — on the two-mode case above it reads **8.05x over** the true
distance where `error_estimate` reads 122x under, 1.3x over under
`aitken` and `iqn-ils` (where the estimate reads 2x and 1000x under),
and never under on 80 random normal contractions of dimension 2–6 with
negative and near-1 eigenvalues.  On the heterogeneous benchmark fixture
(60 000-cell heat grid and four probes, Jacobi, Aitken) the radius alone
read 32x and 42x under on the two steps where Aitken dipped — the map
is far from normal there — and the resolvent term is what covers them
(2.9x and 6.5x over), at the price of being loose on that map elsewhere
(up to 95x, and 911x on the one precision-limited step, where the
60 000-entry float floor is the bound).  That looseness is why it is reported and not applied:
`converged`, the iteration counts and the recorded sweep rows are
unchanged, and the seven existing keys keep their names and values.

It costs eight Jacobian-vector products per group per step and is
charged only with `diagnostics=True`; `fori`, `diagnostics=False` and
`max_iterations=1` report NaN and `spectral_usable=False`, never a
number.  What it is still not: for a non-linear `F` it is asymptotic
(Ostrowski) — exact to float32 on a log map within tolerance of its
fixed point, an estimate far from one; it is in the group's norm at the
returned state, so the dead band's excluded fields are outside what it
bounds; its float floor is a model of the map's rounding; and a group
whose pass depends on the previous one through more than seven
independent scalars (eight where they are the whole state: what the loop
closes through under Gauss-Seidel, so a body exchanging one wrench with a
field of any size is within it) gets a radius from below and
`spectral_usable=False`.  Each entry says why a usable flag is `False`
as codes too (`reason_codes`, experimental: `interface_too_wide` is
expected, `spectral_self_check_failed` is a worry, `at_float_floor` says
to widen the dtype or loosen the tolerance), with a `not_usable_reason`
in words, and reports the floor as `residual_precision_floor`.  The figures in this section were
re-measured on the final tree, after the audit fixes below (jaxlib
0.11.0).

The figures above are measured on the two-mode fixture of
`tests/core/test_coupling_error_bound.py`, as re-derived on the final tree
in commit `23d9d1e`.  The decision is in
`benchmarks/results/audit_040_final/ERROR_BOUND_DECISION.md`, whose
2026-09-22 section records the first measurement (7.95x and 1.22x), taken
before the audit fixes below moved it; the standing caveat is
`MADD-ANO-005`; the field-by-field description is in
[Reading `coupling_diagnostics()`](../developer_guide/coupling_algorithm_guide.md#reading-coupling_diagnostics).

### `gradient_relative_error_bound`: the IFT gradient at an early exit

The IFT adjoint is solved at the iterate the forward *returned*, so a
forward that stops early hands back the derivative of the fixed point
linearised in the wrong place.  Under the same gate
(`solver="ift"`, `diagnostics=True`) `coupling_diagnostics()` now
reports `gradient_relative_error_bound` and `gradient_bound_usable`.
The tangent the adjoint returns differs from the fixed point's by
exactly the resolvent applied to the change, between the two points, of
the one-pass map's Jacobian-vector product along that tangent; the
bound is `spectral_error_bound` (the distance — never `error_estimate`)
times the resolvent factor that bound applies, times that change per
unit distance, measured as a second difference of the Jacobian-vector
product along the Newton correction, one probe per floating constant
the group reads, reported for the worst probe and relative to the
tangent.  It costs `11 + k + 5 n_p` Jacobian-vector products per group
per step (`k ≤ 8`, `n_p` the probes: every entry of a constant of at most
64 entries, one per larger constant; the round-5 fix below made it one
probe per entry and five products per probe) beside the spectral
bound's eight.  Measured (jaxlib 0.11.0, float32), with the fixed
point's gradient from tight `ift` and `fori` arms that agree:

| fixture | early exit | `bound / true` |
|---|---|---|
| concave map `a + g log(1 + u)`, caps 3–8 | 26% → 0.2% from `x*` | 1.10–2.10 (`d/da`), 8.5–10.5 (`d/dg`) |
| convex map `a + g u²`, caps 3–8 | 6.8% → 0.3% | 1.12–1.16 (`d/dg`), 3.06–3.30 (`d/da`) |
| affine map `a + g u`, `d/dg` | 81% → 35% | 1.35 at every cap (`d/da` exact) |
| spring pair `k = 6000`, `c = 60`, `dt = 0.01`, stiffness and mass, caps 2–6 | — | 1.19–1.70 |
| two-mode with a concave slow mode (`q = -0.014`), `converged=True` | 109x `error_estimate` | 25 (40x short with `error_estimate`'s distance) |

(Re-measured after the round-5 fix; the first edition of this table, on
the Arnoldi-restricted resolvent, read 1.21–2.37 / 9.4–11.5, 1.29–1.36 /
3.57–3.80, 1.81, 7–11 on a spring pair no longer in the tree, and 83 on
the two-mode case at `q = -0.02`, which the full-operator Kantorovich
check no longer certifies.)

**It is a statement about the gradient, not the solve.**  On a map that
is affine in its state with additive parameters the IFT gradient is the
fixed point's from *any* iterate, so the bound truthfully reads `0.0`
while the state is far off — `0.0` on the two-mode case sitting 1.1e-2
from its fixed point with `converged=True`, and 3.1e-7 — zero to
float32 — on the stiff spring pair under `iqn-ils` with the interface
norm, whose velocities are 1.7% off.  The returned `(value, gradient)` pair is then mutually
inconsistent — the gradient is `d(fixed point)/dθ` and the value is not
the fixed point — and nothing in this key says so.  Read
`spectral_error_bound` for the health of the solve, within its norm
(under the interface norm it covers the interface fields only, and on
that spring pair reads 9.0e-3).  The existing `gradient_error_estimate`
keeps its name and value.

What it rests on: every condition of `spectral_error_bound`; the change
in the linearisation being linear in the distance and along the Newton
correction (exact for an affine map, leading-order otherwise); and the
probes — a field-valued constant is probed along one random direction,
and the bound is relative to the tangent's norm, so a scalar loss whose
gradient nearly cancels across the state can carry a larger relative
error.  One probe per constant, not one combined probe, because a
combined direction can cancel: on the spring pair it read `0.0` while
the stiffness gradient was 0.8–4.8% off.  `fori`, `diagnostics=False`,
`max_iterations=1` report NaN and `gradient_bound_usable=False`; a
fresh or reset graph has no report at all.  The measurement is the
2026-09-23 section of `ERROR_BOUND_DECISION.md`.

### The bound keys, after their audit

A context-free audit of the two keys above, before release, found five
ways they read below what they bound with their usable flags `True`;
each is fixed and pinned (jaxlib 0.11.0, float32, fresh graph per
measurement):

* **A stalled float32 iterate.**  Once `(1 − rho)·|x − x*|` is below half
  an ulp a pass changes nothing and the residual is exactly `0.0`: both
  keys read `0.0` on a relay contracting at 0.99999 that sat 38 348 ulps
  (4.2e-3) from its fixed point with a 3e-3 gradient error.  A residual
  bounds the exact map's only after its evaluation error is added, so
  `spectral_error_bound` is now `(residual + floor) × factor`, the floor
  `4 eps` per entry the norm reads, in its units
  (`residual_precision_floor`; derivation on `PRECISION_FLOOR_ULPS`): 9.4e-2
  there.  The gradient bound's distance carries the same floor and takes
  its curvature along a floor-sized direction where the residual has
  none.  A new key, **`precision_limited`**, says when the residual is at
  that floor.  **`converged` is deliberately unchanged, and still reads
  `True` on such a state** — so does `strict_convergence` — because a
  floor in the criterion would make tight float32 tolerances
  unreachable; read the bound keys with `diagnostics=True` on a slow
  group.  `strict_convergence` consulting the spectral bound is planned
  for 0.5.0.
* **The residual outside the Krylov space.**  A repeated eigenvalue
  (`A = B ⊗ I₂`) broke the one-vector Krylov space down at dimension 2
  against a range of 4, "settled": the bound read 0.92x the distance.
  The space now continues from the residual at every breakdown: 1.05x.
* **A dead-banded field on the loop.**  Weighted zero in the spectrum it
  cut the loop: `rho_spectral` 0.000 for a gain of 0.9, the bound
  0.15–0.29x.  It keeps its own magnitude's weight now, with its
  residual share folded in: `rho_spectral` 0.900, the bound 2.8–5.5x.
* **A non-finite field the norm does not read** (interface norm, a NaN in
  an internal field): `converged=True`, `strict_convergence` silent,
  `spectral_usable=True`.  The verdict now reads every floating field of
  the returned state (`MADD-ANO-019`), and the spectral triple is NaN
  there.
* **Beyond the leading-order regime.**  At `F'(x*) = 0.99`, 0.65–4.5% from
  the fixed point, the gradient bound read 0.20–0.96x its error.  A
  Newton–Kantorovich check now scales it by `1/√(1 − 2h)` and makes it
  `inf`, unusable, at `h ≥ ½`: two of those points unusable, two hold at
  3.0–3.5x.

None of these moves a state, an iteration count or `converged`.  They
are new keys in this release, so none needed an anomaly entry of its
own.

A second, context-free confirmation audit of the merged tree found four
more ways the keys read low with their flags `True`, and two plumbing
defects beside them; each is fixed and pinned (jaxlib 0.11.0, float32,
fresh graph per measurement):

* **A change too small to be a number.**  A field below `tiny / eps`
  (about `1e-31` in float32) has a subnormal one-ulp change, and the
  norms took `|new − old|` *before* rescaling, so XLA's CPU backend
  flushed it to zero: at `x* = 1e-35` the group reported `residual=0.0,
  converged=True` after 43 passes, 1.5e-2 from its fixed point under the
  L2 norm (tolerance 1e-6), where the same group at `1e-29` takes
  126–130, and `spectral_error_bound` read 6e-4 of the true distance.  The
  pair is now rescaled by `1/tiny` before it is subtracted; the group
  scaled by `2**-20` reproduces the `1e-29` control's residual, verdict
  and pass count to the bit, under all three norms and both solvers.
  This one is in the criterion, so it is recorded on `MADD-ANO-019`.
* **A composite map.**  `PRECISION_FLOOR_ULPS = 4` was a per-pass
  constant, and a node that integrates by explicit Euler in `N`
  sub-steps is `5.8` units off the exact map at `N = 20` and `29.4` at
  `N = 100`: a stalled relay built on one read the bound 0.07–0.96x its
  true distance at `N` = 15–200 with `spectral_usable=True`, whether it
  looped inside `update` or the framework sub-cycled it.  The floor is
  now per evaluation, times the evaluations a pass rounds like — the
  framework's sub-cycling divider, counted automatically, times a new
  `SimulationNode.update_evaluations()` a node declares for its own
  loops.  **A node that does not declare is counted once, and its
  group reports `spectral_usable=False` (and `gradient_bound_usable=False`)
  wherever the residual is at the floor**, because there the unchecked
  count is the whole bound; a group whose nodes all declare keeps the
  flag.  If you read `spectral_usable` on a group converged to float32,
  declare `update_evaluations()` on its nodes (`1` for a single step).
* **A float16 field beside a float32 one.**  The gradient bound took its
  floor from the promoted float32 `eps` and probed the curvature across
  a step no float16 output could register: `0.0`, usable, against a true
  3.2% error.  Each field is now floored at its own dtype and the probe
  step sized by the coarsest (0.65 on that case).
* **The top of float32's range.**  Above `1/tiny` (`8.5e37`, which the
  mixed and interface norms still evaluate) the spectral weights
  `1/max|field|` were subnormal and flushed: `rho_spectral` read 0.0 for
  a 0.9 map and the bound 0.12x.  The weights are now scaled by a common
  power of two there; the group reproduces its `2**-100`-scaled control
  to the bit.
* **Plumbing.**  `compile()` seeded a group's `_meta` slots in the dtype
  of the first floating leaf of a frozenset, so a float16/float32 group's
  `run_scan` raised a scan-carry `TypeError` under some `PYTHONHASHSEED`
  values; the seed is now the promotion of the group's fields.  And
  `reset_state()` matched slots by suffix, so a group keyed
  `a+probe_spectral` had its residual reset to NaN where `compile()`
  seeds 0.0; it now matches exact slot names.

Again none of these moves a finite in-range state, an iteration count or
`converged`: the norms evaluate exactly the expressions they always did,
with the rescaled pair a separate value a `where` selects (a first,
algebraically equal form moved the fused node arithmetic by an ulp under
`solver="ift"` and the L2 norm, and was replaced).

### The IQN family, corrected

Four of these were long-standing and are worth naming:

* **IQN-ILS never activated without Jacobian reuse.**  The first-iteration
  test was `n_cols == 0` and it reset `n_cols` to 0, so the secant basis
  never grew and the method silently ran as Aitken.  With
  `jacobian_reuse > 0` it escaped only by admitting a bogus first column.
* **The safeguard vetoed valid steps on stiff problems.**  The quasi-Newton
  correction was rejected above 10× the residual, but a correct Newton step
  is ≈ residual / (1 − ρ) — 50× at ρ = 0.98.  The bound is now a blow-up
  guard.
* **The secant basis is built from differences of the raw operator
  outputs** (Degroote 2009) rather than of the inputs: **2 iterations
  instead of 5** on the ρ = 0.98 test contraction.
* **The gradient was NaN with `jacobian_reuse > 0`**, because
  `jnp.linalg.lstsq`'s SVD derivative is NaN on the repeated zero singular
  values of the masked secant matrix.  Replaced by `jnp.linalg.pinv`, which
  is rank-deficiency-safe through a `custom_jvp` and gives the same
  solution.

Also: the IFT path ignored `convergence_norm` (it hard-coded L2 against
`tolerance`) and, in IQN modes, iterated with non-interface fields frozen at
their first-pass values, which gave the residual a floor and ran it to the
cap.  The fori-path diagnostics recorded the residual one iteration stale.
The IFT linear solve reported a spurious GMRES "iterative breakdown" on long
runs, because lineax's tolerance is elementwise: exact-zero entries of a
cotangent had to reach `atol=1e-8` absolute while float32 round-off from the
large entries is around 1e-5.  The solve now goes through
`jax.lax.custom_linear_solve` with `rtol` no tighter than about 100 ulp of
the dtype.  This paragraph used to say `atol` was "scaled to the largest
rhs entry"; it was `1e-8 + rtol * max|b|`, and the absolute `1e-8` returned
an exactly zero derivative for any rhs below about `1e-8` (MADD-ANO-113,
since 0.3.0).  The solve now runs on the rhs rescaled by an exact power of
two with a purely relative tolerance; see "[The coupling runtime, round-4
audit](#the-coupling-runtime-round-4-audit)".

Gauss-Seidel coupling with a *flux* edge whose consumer is scheduled before
its producer raised `KeyError` on the first pass; fluxes are now seeded from
the previous iterate (two sweeps, since producers may depend on each other)
and overwritten as producers update.  IQN acceleration derived its interface
fields from edge source fields, so a flux edge raised `KeyError` at compile;
a flux source now maps to the producer's state fields.

### Non-float leaves through the adjoint

Three related defects: the float32 images that carry integer and boolean
leaves through the IFT closure were exact only below 2²⁴, so `uint32` /
`int32` values above that were corrupted and typed PRNG keys crashed at
trace time — images are now 16-bit limbs (`float_image` /
`from_float_image`), keys travel as their uint32 data, and the coupling
norms and predictor act on floating fields only, so a counter or flag is
computed by the pass instead of being extrapolated (at the returned floating
state, on both solvers: MADD-ANO-070).  Differentiating through
`lax.scan` over a coupled step failed when a node in the group had an
integer or boolean state leaf, because `closure_convert` hoisted the leaf as
an integer constant of the IFT `custom_jvp`, which JAX cannot linearise
under a scan; the solver now captures float32 images before
`closure_convert` and keeps such fields out of the fixed-point vector
altogether.  And `unflatten_coupled_state` returned every field as float32,
so an integer or boolean leaf came back as a float after each step —
semantic drift, plus a retrace.

### `waveform_iterations`: the report counts every sweep, and the sweeps are restarts

A sub-cycling group with `waveform_iterations=N` runs its fixed-point
solve `N` times per step, each sweep with its own `max_iterations`
budget.  `coupling_diagnostics()` kept only the last sweep's pass count,
and the first sweep is usually the one that works: on the sub-cycled
spring pair of `test_phases_5_7_8.py`, `max_iterations=2` reports
`iterations` 2 / 1 / 1 at `waveform_iterations` 1 / 2 / 3, the first
sweep stopped at the cap in every case, beside `converged=True`.  So the
documented cap check, `iterations >= max_iterations`, could not see the
cap (MADD-ANO-026, since 0.1.0).  `iterations` is now the **largest**
sweep's count, which makes that check exact again, and a new key,
**`total_iterations`**, is their sum -- the work done (3 / 4 / 5 there at
`max_iterations=10`, where `iterations` reads 3 throughout).  Every other
key stays the last sweep's, the sweep that produced the returned state;
`converged` is its verdict.  The returned state is bit-identical to
before in every configuration, and a group that runs one sweep carries,
reports and compiles to exactly what it did.  `tune_coupling_params`
now costs a configuration by `total_iterations`.

Measuring that showed what the sweeps are: **not waveform relaxation**
(MADD-ANO-027, open, since 0.1.0).  Each sweep re-solves the same fixed
point from where the previous one stopped, with a fresh accelerator, and
the sub-step interpolation runs between two estimates of the end-of-step
value.  So once the first sweep converges each later sweep only adds
at least one more pass, which moves the state by about one residual
(within the tolerance; identical only at exact stationarity),
`boundary_interpolation="quadratic"` is `"linear"` to the last
bit, and `"linear"` differs from `"constant"` only by the in-pass change
of a source scheduled before the sub-cycled node: equal to float32
round-off under Jacobi or with the sub-cycled node's source scheduled
after it (the modes compute the same input but compile to different
programs, so not to the bit; see "What was not changed" below), 2.71e-05
apart (relative) after 100 steps of a spring pair with the slow node
first at `tolerance=1e-4` -- O(tolerance), not an accuracy gain.  The
parameter is now
documented as experimental; use `waveform_iterations=1` and raise
`max_iterations` when the first sweep caps.  `strict_convergence=True`
under `solver="ift"` checks every sweep, which the docs now say.  See
"`waveform_iterations`: what the sweeps are" in the [coupling
guide](../developer_guide/coupling_algorithm_guide.md).

`profile_graph`'s coupling statistics, new in this release, now read the
report's own rules: a step is at the cap at `iterations >= max_iterations`
(it counted `max_iterations - 1`, a leftover from when the default solver
reported one pass fewer there), and `converged_fraction` applies the
relaxation factor under `acceleration="fixed"`.  On a spring pair that
never used more than three of its four passes, the at-cap fraction read
0.6 and is now 0.0; over-relaxed at 1.5, the converged fraction read
1.000 against the report's 0.250.  The report, the profiler and
`sysid`'s convergence mask now take the criterion from one helper,
`convergence_criterion`.

### The coupling runtime, audited on the frozen tree

A context-free audit of the frozen tree found three defects that shipped
and four that were new in this cycle.

**`run_adaptive` and `run_adaptive_scan` on a sub-cycled group**
(MADD-ANO-043, since 0.1.0).  A group with `subcycling=True` over nodes of
different timesteps is not multi-rate -- `compile()` schedules it at its
macro timestep -- so the adaptive steppers accept it, and they hand every
node the one adaptive `dt`.  The fast node was still sub-stepped
`round(macro / node_dt)` times per pass, each sub-step the whole `dt`:
a clock on a 10:1 group read 0.5 at `t_end = 0.05`, and 0.2 on a 4:1
group.  Each sub-step is now `dt * node_dt / macro`, so at an integer
ratio every member covers `dt`, and one adaptive step at `dt = macro` is
the compiled step to float32 rounding.  The multi-rate refusal said "All
nodes must share the same timestep", which a sub-cycled group contradicts;
it now names the rate dividers and the one way nodes of different
timesteps can share an adaptive run.

**A multi-rate group's `_meta` came from solves the step threw away**
(MADD-ANO-044, since 0.1.0).  A coupling group with a rate divider above
one solved on every base step and kept the result only on the steps it
fires on; its node states followed that rule and its `_meta` slots did
not.  Between firings `coupling_diagnostics()` reported the discarded
solves -- 10 iterations, unconverged, beside an applied solve that had
converged in 9 and then 1 -- and so did the profiler (at-cap fraction 0.9,
converged fraction 0.1, where every applied solve converged) and
`windowed_loss(mask_unconverged=True)`, which zeroed a window whose one
applied solve converged.  The linear predictor and the IQN-IMVJ warm start
learned from the discarded solves: with `predictor="linear"` the forward
state reached 8.7e12 at the second firing and 2.1e27 at the third, where
the fixed point is 1.33.  `strict_convergence=True`, new in this release,
raised about a discarded solve on the first step after a firing.  The
group's solve now runs under a `lax.cond` on the firing predicate: a base
step that does not fire neither solves nor touches the group's node
states or slots, so both are always the last applied solve's, and a group
at divider `d` no longer pays its fixed-point iteration `d` times per
solve it applies.  The strict check is gated on the same predicate,
because under `vmap` over states at different phases a batched `cond`
runs both branches.  The profiler samples a multi-rate group only on
the steps it fires on, so its `"n"` counts solves.  The node states are
bit-identical, on jaxlib 0.10.2, 0.11.0 and 0.11.2, wherever neither the
predictor nor IMVJ was learning from a discarded solve, and a group at
the base rate compiles to exactly what it did.

**`solver="fori"` with `iqn-imvj` warm-started from zeros**
(MADD-ANO-045, since 0.1.0; `"fori"` was the default until this release).
The legacy loop runs to `max_iterations` and freezes the state once
converged, and every pass on the frozen state shifted a zero secant column
into the matrices `jacobian_reuse` carries to the next step.  By the cap
the warm-start window was all zeros, so `jacobian_reuse` did nothing, and
from the second step on `"fori"` and `"ift"` took different passes (3
against 4 iterations on a stiff spring pair, 2.3e-04 apart in state).  The
columns now stop at the latching pass, as under `"ift"`; on an affine
three-DOF pair the two solvers now report the same iterations and return
the same state to the bit (jaxlib 0.11.0).

**`converged` is one verdict.**  The loops, `strict_convergence` and the
sysid mask compare `estimated_error(residual, amplification) <= threshold`
in-graph, in float32 for a float32 group; `coupling_diagnostics()` and the
profiler re-derived it on the host in float64, and within half a float32
ulp of the threshold they called unconverged a step the loop had stopped
on.  The host now makes the same comparison in the residual's dtype
(`reported_converged`), and `"error_estimate"` is the float32 estimate the
loop compared.

**Refusals and warnings.**  `windowed_loss(mask_unconverged=True)` refuses
a group with no convergence slot -- `solver="fori"` with
`diagnostics=False` -- which the mask used to skip without a word.
`profile_graph(measure_coupling=True)` no longer warns that the caller's
accelerations are "ignored under max_iterations=1", a cap only its own
one-iteration variant sets, and a group's inert-knob warnings skip
standard-library frames when they look for the caller's line (they were
attributed to `dataclasses.py`); nor does it repeat, from the variant's
and the restore's recompiles, the caller's own compile-time warnings.
`compile()` refuses a sub-cycled node whose timestep does not divide its
group's largest (MADD-ANO-046, since 0.1.0): it took
`round(macro / node_dt)` sub-steps of its own timestep and so covered
0.009 of every 0.01 at a timestep of 0.003, silently.  Under `jax.vmap`
`gradient_relative_error_bound` can differ from the unbatched value in
its leading digit (2.01e-06 against 1.84e-06 on one step, both above the
true 9.5e-07): its curvature factor is a difference of Jacobian-vector
products a few ulps apart, which the batched program rounds differently;
this is documented rather than changed.

**Documentation that said more than the code.**  `CouplingGroup`'s
`diagnostics` stored the iteration count and residual only under `"fori"`;
under `"ift"` its effect is the spectral and gradient-bound keys, at
`8 + 11 + k + 5 n_p` Jacobian-vector products per step.
`boundary_interpolation`'s modes coincide only when the sub-cycled node's
sources are scheduled after it, under Jacobi, or at exact stationarity
(2.7e-05 relative apart after 100 steps with the slow node first at
`tolerance=1e-4`); `"quadratic"` is `"linear"`.  The
`waveform_iterations` inert warning no longer calls the sweeps waveform
relaxation (MADD-ANO-027), and the report's `"amplification"` is
documented as the rate it is, `max(r_k / r_{k-1}, sqrt(r_k / r_{k-2}))`.

### The coupling runtime, second audit

A second audit of the coupling area found two defects that shipped, both
loud, and several that were new in this cycle; fixing them turned up a
third that shipped, in `run_adaptive`.  None of the fixes moves a
state it did not have to: a before/after dump of 274 configurations (five
fixtures, both solvers, every acceleration, both iteration modes, the L2
and mixed norms, the adaptive steppers with `strict_convergence` on and
off, and the sysid loss and gradient with the mask on and off), compared
leaf by leaf after every step, differs only in the integer leaves the
first fix below stops rounding and in a predictor seed that is never read
(jaxlib 0.11.0).

**Acceleration touches floating fields only** (MADD-ANO-059, since 0.1.0).
`acceleration="aitken"` and `"fixed"` relaxed every field of the group,
and the quasi-Newton accelerations every field their selection named.  A
group holding an integer, boolean or PRNG-key leaf raised a `TypeError` at
the first step in every release; in this cycle, once the flattening learned
to restore dtypes, `solver="fori"` returned such leaves rounded through
float32 instead (`0xdeadbeef` as `0xdeadbf00`, `2**24 + 1` as `2**24`) and
kept them, and `solver="ift"` with a quasi-Newton acceleration failed with
a broadcasting error.  Now every acceleration acts on floating fields, as
the IFT solver always iterated; a non-floating field named in
`accelerated_fields` is dropped, and a selection with no floating field is
refused at compile.

**A group-internal flux edge where the loop reads the state**
(MADD-ANO-060, since 0.1.0).  The interface norm, and a sub-cycled
member's linear (or quadratic) boundary interpolation, read an internal
edge's value from the producer's state, which does not hold a flux; both
died inside the step with `KeyError: 'spring_force'`.  `compile()` now
refuses each such edge by name and says what works: the `"mixed"` or
`"l2"` norm, `"constant"` interpolation.

**`strict_convergence` under the adaptive steppers checks the solves they
keep.**  `run_adaptive` and `run_adaptive_scan` estimate error by step
doubling and keep only an accepted attempt's two half steps.  The strict
check ran inside every solve, so an attempt whose discarded full step did
not converge raised -- though rejecting that attempt is how the controller
recovers, and on a spring pair at coupling gain 3 every solve the run kept
converged.  `run_adaptive` now checks the kept half steps on the host, and
the scan checks `accepted & ~done` in-graph; a kept solve that does not
converge still raises, with the same message.  (MADD-ANO-044's principle,
applied to the adaptive steppers.)

**`windowed_loss(mask_unconverged=True)` drops a masked window from the
gradient too.**  The mask multiplied the window's loss by 0.  A window
whose coupling diverged holds inf and NaN, `0 * inf` is NaN, and the
backward pass carried the zero cotangent through the window's non-finite
intermediates: the gradient came out NaN while the loss looked right,
because XLA's CPU backend happened to rewrite the product into a select.
The verdict is now taken in a gradient-free forward pass, the masked
window's inputs enter through a `where` whose derivative selects, its
outputs are replaced by their targets, and its loss is selected away; a
diverged window contributes exactly nothing, and the gradient is the sum
of the other windows' gradients.  Masking now costs one extra forward
simulation, and under multiple shooting a masked window's continuity
penalty goes with it.  A new test also pins the mask's criterion: on a
group whose residual is under the threshold and whose error estimate is
over it, a mask reading the raw residual used to pass every mask test.

**`run_adaptive` keeps its clock with the state at `dt_min`**
(MADD-ANO-061, since 0.1.0).  When an attempt's error was too large and
the shrunk step would fall to `dt_min`, the stepper accepted the attempt it
had just made -- whose half steps covered the attempted step, up to
`dt_min / min_factor` -- and advanced its clock by `dt_min`: a node
integrating its own clock read 0.025 at a reported `t = 0.02`.  The clock,
the histories and the callback now advance by the step the state covered.
`run_adaptive_scan` was not affected.

**The graph is put back after `jax.grad` whatever its first node.**  The
escaped-tracer recovery recognised a traced state by its first entry, and
a state that has been through a `lax.scan` has sorted keys: `_meta` came
first on a coupled graph whose node names sort after `"_"`, and a node
that reads no parameter comes back concrete beside traced ones.  Either
way the next `step()` after `jax.grad` of a `run_scan` raised JAX's
`UnexpectedTracerError`.  Every node is asked now, and
`coupling_diagnostics()` puts the graph back first like every other entry
point.

**`coupling_diagnostics()` judges the last step under the group that took
it.**  Replacing a group by another over the same nodes kept its slot
names, and the report re-derived the old step's verdict under the new
criterion (`converged=False` three passes into a fifty-pass budget).  It
now reads the group the compiled step was built from, and a recompile
restarts a replaced group's report slots, so it has no entry until it has
stepped.  The `converged` docstring now says that a non-finite state
records `residual=inf` by design, whatever the norm evaluates to.

**Documentation.**  `waveform_iterations`' documentation, the coupling
guide, MADD-ANO-027 and these notes said a converged first sweep leaves the
later sweeps nothing to change.  Every sweep starts with one pass, so each
later sweep applies at least one more and moves the state by about one
residual -- within the tolerance, and identical only at exact stationarity
(6.0e-08 relative on the guide's spring pair at `tolerance=1e-4`; 1.2e-04
on a linear pair contracting at 0.9025 at `tolerance=1e-3`).
`flux_coupling_demo` no longer calls its sweeps waveform relaxation or
asserts that `"quadratic"` interpolation is more accurate than
`"constant"`.  The compile-time predictor seed is written in the step's
node order rather than the per-process `frozenset` order; it is never read
for extrapolation, so no state moves.

### The coupling runtime, differential-harness findings

The coupling differential harness (`tests/property/test_differential_*.py`)
compares two paths that must agree over generated graphs, and pinned what
disagreed as strict xfails.  All thirteen are fixed; the generators no
longer steer around the configurations that used to raise.

**A group's integer and boolean fields describe the state returned**
(MADD-ANO-070, introduced and fixed in this cycle).  `solver="ift"` kept
only floating fields in its fixed-point vector and restored the rest from
the first pass, on the premise that such a field depends on the pre-step
state alone.  A flag that reads a boundary input does not: a pair whose
flag's input changed sign during the solve returned `sign=True` beside
`x = -1/15`, and a flag carried across a group-internal edge to a reader
that takes it from the iterate (Jacobi, or a reader scheduled first) stayed
shut inside the loop, so the group converged to the fixed point of the map
with the gate held shut (`x_b = -1/15` against the consistent `1/15`).  Both
solvers now return a group's non-floating fields as the pass computes them
at the returned floating state (`"fori"` used to take them from the raw pass
before an accelerated iterate), and `"ift"` evaluates an edge-carried one at
the iterate on every pass.  It costs one more evaluation of the pass per
step on a group with such fields, of which XLA keeps only what those fields
need; an all-floating group compiles to the program it always did.
MADD-ANO-059's text, which stated the premise, is corrected.

**A predictor keeps those fields in the iterate it starts from.**  It
replaced each node's state with its extrapolated floating fields, and the
mixed norm, which looked every field of the new iterate up in the old one,
raised `KeyError` naming the leaf on the first step.  The predicted fields
are merged over the state, and the mixed and interface norms check a field
is floating before reading the old iterate, as the L2 norm did.

**Jacobi seeds producers' fluxes like Gauss-Seidel** (MADD-ANO-071, since
0.1.0).  A flux producer reading another producer's flux raised `KeyError`
under `iteration_mode="jacobi"`; the Jacobi pass now takes the Gauss-Seidel
pass's two-sweep seed wherever a producer reads a flux, and keeps its
single sweep, and its program, everywhere else.

**Every entry point puts the graph back after `jax.grad`** (MADD-ANO-072,
introduced and fixed in this cycle).  `reset_state` -- the remedy the
recovery warning names -- raised `UnexpectedTracerError` on a predictor
group, and `set_node_state` wrote into the traced state, so the next step
put the graph back over the write and ran on from the state before the
transform.  `reset_state` (quietly), `set_node_state`, `get_node_state`,
`resolve_boundary_inputs`, `validate`, `add_node` and `remove_node` now
recover first.

**Under-relaxation reads the relaxed rate on the first loop pass**
(MADD-ANO-073, introduced and fixed in this cycle).  The first loop pass
took its ratio from the unrelaxed pre-loop pass, so under
`acceleration="fixed"` with `relaxation < 1/(1 + sqrt(rho))` a group that
stopped there understated its distance about 2x and reported converged
(0.72 thresholds estimated, 1.41 true).  On that pass an under-relaxed group
now divides the amplification by the relaxation, the relaxed iteration's
amplification; over-relaxation and every other acceleration are untouched.

**The adaptive error norm reads floating leaves only** (MADD-ANO-074,
since 0.1.0).  A `bool` raised `TypeError` in both adaptive steppers, a
`uint32` wrapped and the controller rejected almost every step (a scan
stopped at t = 0.296 of 2.0), and an `int32` counter moved the accepted
steps (60 became 50 on one node).  Integer, boolean and PRNG-key leaves now
add neither a term nor an element.

**What was not changed.**  The harness also measured that a group
contracting at a rate near 1 reads its ratio as `>= 1` from float32 noise,
and so stops on the documented raw-residual fallback far from its fixed
point; MADD-ANO-005's residual risk now records it (see "Known anomalies").
And `boundary_interpolation`'s "bit-identical" under Jacobi, in its
docstring, the coupling guide and MADD-ANO-027, holds to float32 round-off:
the modes compute the same input but compile to different programs (14 of
144 node-steps differed on sub-cycled Jacobi pairs, by up to two ulps).

### The coupling runtime, round-3 confirmation audit

A context-free audit of the merged tree, with reproducers, found two
defects that read a wrong verdict or bound with its flag set, a units
dependence, an unpinned threshold, and four smaller ones.  All are fixed.

**A Gauss-Seidel pass's float floor counts its chain of same-pass reads**
(MADD-ANO-094, introduced and fixed in this cycle).  The floor
`spectral_error_bound` adds is four ulps per entry per evaluation the pass
rounds like, and it took the worst node's count.  Under Gauss-Seidel a node
reads every member scheduled before it from the same pass, already rounded,
so the rounding at the end of a chain is the sum along it: on a ring of
scalar relays (loop gain 0.99, every node declaring `update_evaluations()`)
stalled at float32 the bound read 0.51x the true distance at 32 relays and
0.30x at 64 with `spectral_usable=True`, and the gradient bound 0.50x its
true error at 32.  The count is now the longest chain of same-pass reads
under Gauss-Seidel (Jacobi keeps the worst node's), and the same ring reads
16-23x over.  Only `spectral_error_bound`, `gradient_relative_error_bound`
and `precision_limited` move, on Gauss-Seidel groups of two or more chained
members; states, iteration counts and `converged` are bit-identical.

**After `run_adaptive*` the report covers both kept half steps**
(MADD-ANO-095, since 0.1.0).  It described only the second, so a first half
step that ran out of passes was hidden -- `converged=True` on a step
`strict_convergence=True` refused.  `iterations` is now the larger half's
count, `total_iterations` the sum where the group owns that slot
(`waveform_iterations > 1`; a one-sweep group reports `iterations`), and the
per-solve keys come together from the half whose verdict is the step's.

**Aitken and IQN do not depend on the one unit a group is written in.**
(One common factor on every field.  A change of the units of one field
alone is another run under these accelerations: MADD-ANO-285, open.)
Aitken guarded its denominator
with an absolute `sum(delta_r**2) > 1e-30`, so the same group at 1e-12 of
its units froze omega and hit its cap unconverged where scale 1 converged in
88 passes, and at 1e-16 Aitken was plain iteration; IQN's blow-up guard had
an absolute floor, its norms overflowed above about 1e19, and LAPACK
rescaled its least-squares solve at extreme scales.  Each now works on the
vectors rescaled by one exact power of two, which leaves every
configuration at ordinary scales bit-identical (125 dumped) and makes a
group at any power-of-two scale, the same on every field, reproduce the
unscaled run bit for bit -- which the new units oracle in the coupling
differential harness now checks for every acceleration, norm and solver.

**`strict_convergence`'s threshold is pinned** at twice the threshold, at
the raw residual inside it with the estimate outside, and exactly at it,
for `step()` and both adaptive steppers.  Loosening the predicate fourfold
or a thousandfold, or testing the raw residual, had survived every test.

**Smaller fixes.**  A bfloat16 or float16 group's pass count is carried as
int32, so the cap check holds above 256 (2048) passes, and `diagnostics=True`
runs its spectral and gradient analysis in float32 instead of raising
`NotImplementedError`.  Under `jax_enable_x64` a float32 group steps and
scans with every acceleration and both solvers: the accelerator carries
were seeded at canonical precision (MADD-ANO-017's item (3), which said the
fori path added no failure of its own, is corrected).  The profiler's
`coupling_per_iteration_ms` divides by the real step's `total_iterations`
less one pass per waveform sweep, which the capped variant still runs, and
its two recompiles no longer swallow a `node.params` write made before
profiling and not yet compiled (they restore the params snapshot the
caller's last compile committed; MADD-ANO-093's rule).  And
`converged=False` is documented as "at the cap, or not finite", which can
come before the cap.

### The coupling runtime, round-4 audit

A context-free audit of the merged tree, with reproducers, found one
shipped silent defect in the IFT derivative, two more ways the coupling
bounds read low with their flags set, an underflow range below which the
accelerators and the report's arithmetic flushed, and three loud or
smaller defects.  All are fixed.

**The IFT tangent and adjoint do not depend on the problem's units**
(MADD-ANO-113, since 0.3.0).  The matrix-free solve stopped on `|r| <= 1e-8
+ rtol * max|b|`, and with the zero initial guess the absolute `1e-8`
passed before a single step whenever the rhs was below about `1e-8`: the
gradient came back exactly `0.0`, reported successful, for a group in
small units and, in reverse mode, for any loss near its minimum.  At
v0.3.0 and v0.3.1 a least-squares gradient was right at bias `1e-6` and
exactly zero from `1e-8` down, where `solver="fori"` was right.  The solve
now runs on the rhs rescaled by an exact power of two, with a purely
relative tolerance, and answers a zero rhs with zeros.  A new differential
oracle compares `jacfwd` and `grad` with `linear_solver="dense"` and with
`solver="fori"` from `2**-100` to `2**60`, at 6 and 60 coupled DOF.

**The float floor weights every read by its measured gain** (folded into
MADD-ANO-094, introduced and fixed in this cycle).  The chain count of
round 3 assumed a read passes a relative rounding on at most unchanged.  A
squaring relay doubles it, so a stalled Gauss-Seidel ring of twelve squares
read its spectral bound at 0.20x the true distance and its gradient bound
at 0.018x the true error, flags set.  With `diagnostics=True` the step now
measures each group-internal read's relative gain at the returned state
(one JVP of the reading node's update per read, every step, in its own
`lax.cond` branch), scales each member's own count by the sum of its reads'
gains where that exceeds one (a member whose terms cancel amplifies its own
rounding too, which is what a Jacobi group's count rests on), weights each
same-pass read by its gain along the chain, and reports that count (the
`coupling_<key>_pass_evaluations` slot).  It is conservative: that ring
now reads 184x over.  It adds gain magnitudes, so where signs alternate it
is valid and far too loose to use: on a ten-link chain `c_j = 3 c_(j-1) -
2 c_(j-2) + c` it counts 3.6e5 evaluations and the bound reads 1.1e7x the
true distance with `spectral_usable=True`, and sub-cycled groups read
1.5e3x-9.6e4x over (that part predates the weighting).  A signed
composition is a bound only entry by entry, and would still read about
3e4x over there; MADD-ANO-094's residual risk has the analysis.  The report also stopped re-deriving the count from
the graph as it stands at report time: it reads the count the step
measured, or the one `compile()` snapshotted, so rebuilding a node after a
step no longer moves the bound of the step that ran (17x measured), and a
group whose member was removed since has no entry instead of raising
`KeyError`.  The fixed-point oracle gained adversarial generators: chains
whose links' relative gains exceed one (`u**2`, `u**4`, `u0 * u1`), and
signed affine chains whose terms cancel, under both iteration modes.

**Below about 1e-31 the accelerators and the report's arithmetic no longer
flush** (MADD-ANO-115, never released).  `fixed`, Aitken and IQN formed
their step from `x_raw - x_old`, which flushes to zero once a pass's change
is subnormal: the iterate stalled and the raw residual test reported
`converged=True` up to 7.4x the threshold from the fixed point.  The
accelerators now work in a power-of-two frame fixed for the solve, and the
relaxed step is formed entry by entry in its own frame, so a small field
beside a large one moves too.  The spectral rate's and the gradient bound's
residuals, curvature step and secant are formed in power-of-two frames, and
their JVP tangent, in state units, is lifted only out of the underflow
range: a gradient bound that read `0.0` with `gradient_bound_usable=True`
near `1e-34` now reads its control's value.  A group scaled by a power of
two reproduces its control's passes, verdict and residual to the bit down
to `2**-24` of the normal-range control (every acceleration), and its
spectral rate and bounds down to `2**-122`.  At the tags, the same groups
had already stopped on their first pass: the pre-0.4.0 L2 norm summed
squared absolute changes, which underflow below about `1e-19`, and read
`residual=0.0` 90% from the fixed point (MADD-ANO-114, since 0.1.0,
resolved by this release's norms).

**Smaller fixes.**  `solver="fori"` with Aitken or IQN on a bfloat16 or
float16 group no longer raises a carry `TypeError` (MADD-ANO-116, since
0.1.0); the two solvers take the same passes there.  `profile_graph`'s
per-node timing feeds each node its declared inputs instead of `{}`, which
raised `KeyError` for any node indexing one (MADD-ANO-117, since 0.1.0).
`run_adaptive` and `run_adaptive_scan` take their accept/reject decision and
next step from one function, `adaptive.step_decision`: `run_adaptive` used
to accept a failed attempt up to `dt_min / min_factor` whenever shrinking
it would reach `dt_min`, overshooting `t_end` on a decay held at `dt_min`
(MADD-ANO-118, since 0.1.0); it now retries at `dt_min` as the scan always
did.  The mutation gate's ring fixture now has a schedule that differs from
its sorted order, so a floor computed in sorted order fails it.

**What it costs.**  Only groups with `diagnostics=True` pay: the gain
measurement is one JVP of a member's update per group-internal read per
step, and the frames add elementwise exponent arithmetic to the report.
The compile-count gate's `coupled_diagnostics` workload grew from 7838 to
8567 jaxpr primitives and from 4835 to 5477 HLO ops (frames, lift and slot
+593 and +491; the gain measurement +136 and +151); every workload without
diagnostics compiles to the program it did.

### Every small constant says its units

Four of this release's coupling defects were one mistake: an absolute
constant inside a relative computation.  The IFT solve's `atol=1e-8` gave
gradients of exactly zero for small-unit states, IQN and `fit_lm` carried
`1e-12` floors, and the accelerators' steps flushed at small magnitudes.
`scripts/check_numeric_constants.py` now runs in CI's compliance job.  It
requires every small float literal (`|v| <= 1e-3`, or a decimal exponent
of `-4` or below) and every additive or floor use of `finfo(...).tiny` /
`.eps` in `src/maddening/core/`, `sysid.py` and `cloud/multigpu/` to carry an
inline `# units:` comment or a counted line in
`scripts/numeric_constants_allowlist.txt` saying what it is relative to.  It
fails on a stale or miscounted line and on an empty scope; 18 planted
mutants of the gate (and of the `safety_relevance` check below) are each
caught.

Classifying the 45 hits found four more that had shipped, now fixed:

* **The public Krylov solvers' default tolerance is relative**
  (MADD-ANO-123, since 0.3.0).  `sharded_cg` and `sharded_gmres` (STABLE)
  and `ift_linear_solve` defaulted to an absolute `atol=1e-8`.  On a
  12-unknown SPD system in float32, a right-hand side near `1e-9` came back
  with relative errors of 0.4-1.0 and `converged=True`; lineax GMRES raised
  "iterative breakdown" at some ordinary scales; and the gradient through
  `ift_linear_solve` was exactly zero for a small cotangent.  Every solve now
  runs on `b` rescaled by an exact power of two, and `atol` defaults to
  `None`, a purely relative test.  An explicit `atol` keeps its absolute
  meaning.  `ift_linear_solve`'s tangent and adjoint go through
  `jax.lax.custom_linear_solve`, each framed on its own right-hand side.
  The solution and the gradient are bit-identical at every power-of-two
  scale.  The loop GMRES also clamps `restart` to the system size: it
  returned NaN for any system smaller than `restart` (MADD-ANO-124, since
  0.3.0).
* **The multi-rate GCD is relative** (MADD-ANO-125, since 0.1.0).  Nodes
  at `1e-9` and `2e-9` got a base step of `2e-9`, and their clocks drifted
  apart.  The base step over 12,124 ordinary timestep sets is unchanged.
* **Adam's `eps` is relative to the run's first gradient** (MADD-ANO-126,
  never released).  A loss near `1e-12` left `fit`'s parameters where they
  started.  `fit` and `fit_multiple_shooting` now take the same steps, to
  the bit, for the loss scaled by any power of two.
* **The adaptive error norm has no absolute floor** (MADD-ANO-127, never
  released).  `max(scale, 1e-300)` read a float64 entry below about
  `1e-297` up to `1e297` times too small.

Three user-facing defaults stay absolute by design, and are now registered
so that their units are explicit: the adaptive steppers' `atol=1e-6` (state
units) and `dt_min=1e-8` (seconds), which are scipy's (MADD-ANO-128), and
the deprecated `calibrate` / `tune_coupling_params` thresholds
(MADD-ANO-129).

**One power-of-two helper.**  The coupling runtime's frames are now one
private module, `maddening.core._pow2_frame`: the Aitken and IQN
normalisers, the norm's underflow rescue, the IFT solve's right-hand side,
the accelerator frames and steps, and the report's framed differences and
JVP lift.  The solvers and the Adam fitters use the same module.
`tests/core/test_pow2_frame.py` refuses a `frexp` or `ldexp` anywhere else in
the numerical core.  The consolidation changes no number.  All 125
configurations of the coupling bit-identity dump are bit-identical before
and after on jaxlib 0.10.2, 0.11.0 and 0.11.2, and their step jaxprs (IFT
gradients included) are textually identical.  As a check that the dump can
tell, a frame that is not a power of two moved 88 of the 125 configurations.

**Coupled fields in the subnormal range warn.**  A coupled group whose
nonzero field magnitudes sit below `finfo.tiny / finfo.eps` of their dtype
(`9.9e-32` in float32, `0.0625` in float16) gets one `UnderflowRangeWarning`
per group on the first step after `compile()`.  The warning names the
field, its magnitude and the remedy: rescale the field's units.  Below that
magnitude a node's own arithmetic loses resolution (MADDENING's coupling
arithmetic is framed; a node's `update` is not), and below `tiny` the norm
reads the field as zero.  Exact zeros never warn, and the compiled step is
untouched.  See "Fields in the subnormal range" in the coupling guide.

**The counted floor's limits, with numbers.**  `PRECISION_FLOOR_ULPS`,
`spectral_usable`, the coupling guide and coupling claims CPL-092 and
CPL-102 now state the two assumptions behind the counted floor: nodes
declare `update_evaluations()` truthfully, and no node cancels internally.
They also give what breaking them costs on the 0.4.0 floor study: a
2000-sub-step node declared as one evaluation read 0.26-0.60x the true
distance with the flag set, and internal cancellation 0.02-0.65x.  In the
other direction, mixed-sign chains read up to 1e7x over and sub-cycled
groups 1e3-1e5x.  A measured `diagnostics="rounding"` level is planned for
0.5.0.

**The registry gate reads `safety_relevance` against the enum**
(MADD-ANO-130, since 0.1.0).  `scripts/check_anomalies.py` relied on the
validator's spelled-out copy of the enum.  A YAML list in the field crashed
both the validator and the gate with a traceback.

### The coupling claims inventory, and what it found

`docs/validation/coupling_claims.yaml` lists every documented claim about
coupling -- 130 of them, from the group's knobs to the profiler's
statistics and the sysid mask -- with the conditions it was stated for,
how its truth is known and a test that fails if it stops holding;
`tests/compliance/test_claims_inventories.py` holds every row to that.
Writing it found two defects and a docstring that understated the code,
fixed here, and eight claims whose conditions were not stated, now
stated.

**A node downstream of a coupling group reads the group's output in the
same step, whatever order it was added in** (MADD-ANO-120, since 0.1.0).
The schedule ordered every node Kahn's sort could not reach -- the
members of every cycle and everything downstream of one -- by the order
the nodes were added in.  So a node reading a group through an edge on
no cycle, added before the group's members, was scheduled ahead of them,
and its input was classified as a back edge and read from the previous
step: one step late, with no warning (0.409 against 0.821 after one step
on a two-relay group).  The same held between two cycles, and for a
staggered cycle with no group.  The unreached nodes are now ordered by
their strongly connected components -- each component whole, after the
components it reads, its members in the order they were added, which is
the order a Gauss-Seidel sweep follows -- so the only edges a schedule
reads from the previous step are a cycle's own.  A graph built with
every reader after its cycle (every component whole, after what it
reads) is scheduled exactly as before.

**The interface norm no longer depends on the order the edges were
added in.**  It summed its terms over the group's edges in insertion
order, and floating-point addition does not associate: the same group
with its `add_edge` calls reversed reported a residual that differed in
its last bits, enough to move a verdict near the threshold.  It now sums
in an order the group fixes -- by each edge's source's and target's place
in the group's sweep, the order the L2 and mixed norms sum the members
in -- so neither the `add_edge` order nor the nodes' names reach it.

**`solver="fori"` works under forward-mode AD.**  Its docstring said it
did not; it always has, by unrolling, like reverse mode.

Stated with their true conditions: the single-pass verdict
(`max_iterations=1`), `tolerance`'s relation to the pre-0.4.0 absolute
threshold, the dense adjoint's memory (`2 N**2` floats in reverse mode,
`3 N**2` in forward mode, the failure message included), the old and
new mixed scales, "one verdict" beside waveform sweeps, the cost of
Aitken's two-pass guard, Jacobi's independence from the build order
(states under `"none"` and `"fixed"`; round-off under the accelerators;
the gradient bound's probe directions follow the build order), and the
IFT derivative under different accelerations.  The coupling guide gains
a section on what a graph's build order can still change -- three or
more additive edges into one input sum in insertion order -- and a note
on Aitken stalling under Jacobi when one quantity enters the residual
twice.

Paths this changes: the schedule (and so the back edges, the block
order and `gm.schedule`) of every graph with a cycle in which a node or
another cycle downstream of it was added before it, on every entry point
-- `step`, the scans, the adaptive steppers, `sysid`, the FMU and REST
paths; and the interface residual of every group with two or more
internal edges not added in that order (its last bits, and with them a
pass or a verdict within an ulp of the threshold).  Nothing else moves:
a before/after dump of 190 configurations (the differential harness's
structures and five benchmark fixtures, every acceleration, both
schedules and solvers, all three norms) is bit-identical outside those
two (jaxlib 0.11.0).

### The claims inventories' domain matrix, and what it found

Audits kept breaking claims in a *domain* their conditions covered but no
test exercised: x64, a predictor inside a windowed fit, a write pending at
an FMU export.  The coupling and sysid+FMU inventories now carry a domain
matrix: each row says, for float32, float64 under x64, mixed dtypes, 16-bit
floats, jit, gradients, vmap, multi-rate graphs, sub-cycled groups,
predictors and warm starts, `run_adaptive`, checkpoint restarts and sharded
graphs, which test exercises the claim there, that its conditions now
exclude the domain ("Not claimed for ..."), or that the claim cannot apply
there.  `tests/compliance/test_claims_inventories.py` checks the matrix
(`testing_standards.md`, "The domain matrix").  A narrowed domain changes a
documented claim: read a row's conditions before relying on it outside
float32.

Two batteries state every coupling report, bound and gradient claim once
and run it in each domain, on a two-member affine group whose fixed point
and derivatives have closed forms
(`tests/core/test_coupling_claims_in_every_domain.py`, and with a member
sharded over four devices `tests/cloud/multigpu/test_coupling_claims_on_a_sharded_graph.py`);
smaller modules cover the float64 sysid, `ParamSpec`, checkpoint and FMU
cells.  The claims held in float64, with a float32 member under x64, under
`vmap`, on multi-rate and sub-cycled graphs, with predictors and IQN-IMVJ
warm starts, after a checkpoint restart and with a sharded member.  Four
domains did not, and all four are fixed in 0.4.0 (see "[The known
coupling, sysid and FMU findings, fixed](#the-known-coupling-sysid-and-fmu-findings-fixed)"):

- **16-bit gradients** (MADD-ANO-161): `jax.grad` or `jax.jvp` through a
  bfloat16 or float16 coupling group raised `NotImplementedError` -- the IFT
  rule's linear solve ran in the group's dtype, which LAPACK has no kernels
  for.  Since `solver="ift"` arrived (0.3.0), and the default path in this
  cycle.
- **`run_adaptive_scan` gradients** (MADD-ANO-160): under `"aitken"` or the
  IQN pair the derivative read exactly 0.0 (NaN on a later tree) once a
  member carried a step-dependent field (a clock) and the iterate started
  at its fixed point.  The releases returned NaN under `"aitken"` for the
  same graph.
- **`strict_convergence` with a sharded member** (MADD-ANO-162): the raise
  inside one device's thread deadlocked the step's all-reduce and XLA
  aborted the process after about a minute.
- **A bfloat16 group's `rho_spectral`** (CPL-087) was stored in bfloat16, so
  it read to 2**-8 rather than "to float32"; the spectral bound still held
  over a sweep of rates 0.90-0.99.  New in 0.4.0, minor, not registered.

Each was a strict xfail citing its row; CPL-030, CPL-033, CPL-087, CPL-093,
CPL-094, CPL-140, CPL-141, CPL-143 and CPL-146 were `failing` until it was
fixed, and are `verified` now.

In the sysid inventory the float64 and mixed-dtype cells
(`tests/core/test_sysid_claims_under_x64.py`: every leaf float64, and
float32 leaves in an x64 process) held but for two minor ones.  SYS-024:
with float32 leaves under x64 the `PrecisionLimitWarning` still says to
re-run under x64, which is already on (the remedy is float64 leaves).
SYS-071: a multiple-shooting run that returns its start returns the start's
bits while `best_loss` was evaluated after the `log` round trip, one float64
ulp away -- 5.2e-13 relative near the optimum.  Both are `failing`, neither
registered.  SYS-130 is narrowed to x64: in float32 the damping's bound
clips a zero-truth fit to 0.0 exactly and it converges.

Every narrowed cell was looked at again once the domain oracles (under
"Engineering notes") and the coupling covering array ("configuration
interactions and graph shapes", below) had landed.  42 of them, in 22
rows, now cite an oracle that runs the claim there, and those domains are
no longer excluded.  Among them: float64, sub-cycled and predictor cells
for the iteration-mode, sub-cycling, interpolation, predictor,
interface-norm and solver-parity claims; float64 for the spectral bound
(CPL-088 now reads "float16 to float64"); checkpoints of IMVJ warm starts
and of the adaptive report; multi-rate and sub-cycled records for
`observations_from_history` and multiple shooting; and predictor groups,
snapshot restores and refused steps for three FMU rows and for a
node.params write (SYS-115).  What stays narrowed is what no test runs.

### The REST and run_pod inventory's server domains, and what they found

The REST and run_pod claims inventory now carries a domain matrix of its
own (`domain_set: server`): for each row, whether its claim is exercised on
a loopback and on a non-loopback bind, where the token is demanded and
where it is not, under simultaneous requests on a real loopback server,
beside the realtime runner and beside an in-flight `/sim/run`, at and past
the size caps, under hostile input, when SIGTERM arrives mid-request, on
wrapper nodes and on a graph restored from a checkpoint -- and, for
`run_pod.py --summarise`, on a dry run's records as written, on records
relabelled as a real 4-GPU run and on a directory of mixed commits
(`testing_standards.md`, "The server set").

Most cells cite two batteries. `tests/api/rest_claims_support.py` states
each route row's claim once, as a check keyed by its row id;
`tests/api/test_rest_claims_in_every_domain.py` runs each check, on a fresh
server and graph, in every in-process domain (SIGTERM is raised through
the handler the server chains ahead of the installed one, while the
check's request holds the graph), and
`tests/api/test_rest_claims_under_concurrent_requests.py` runs four copies
of each at once against uvicorn on loopback and states the claims whose
point is concurrency -- N simultaneous steps are N steps, bit for bit; six
nodes added at once cannot take the graph past its budget; six starts
start one runner; saves of one name leave a manifest that hashes to the
file; `/healthz` answers while every worker waits for the graph.  The two
REST property oracles also run on a server that demands the token
(`tests/property/test_rest_properties_on_a_non_loopback_bind.py`).

The claims held in every domain.  On the way the matrix reached, by its
own route, four defects the round-7 REST audit had found and its fixes
(fix/p4-18-rest, under "The REST round-7 audit") remove; each cell was a
strict xfail until those fixes landed, and passes against them:

- **A write through a `HybridNode`** (REST-042, REST-073, REST-074,
  REST-086, `wrapper_nodes`) was answered 200 and lost.
- **A `/sim/run` that cannot have the graph between slices** (REST-043,
  `sim_run_active`) said "Nothing was changed" after it took steps.
- **The runner routes behind a saturated worker pool** (REST-044, REST-045,
  REST-046, REST-105, `concurrent`): with more reads waiting for the graph
  than uvicorn has worker threads, a stop or a stride waited about two lock
  timeouts for a worker, and a start or a reset answered one timeout late.
- **A checkpoint carrying a value `PUT /graph/params` refuses** (REST-093,
  `hostile_input`) was loaded with a 200.

REST-098 ("there is still no TLS") is no longer untested: on either bind
the server speaks plain HTTP on its socket and a TLS ClientHello gets no TLS
answer, so the day TLS is added the test fails and the sentence changes
with it.  Sixteen rows end their conditions with "Not claimed for ...",
mostly the profile, JAX trace, structural-write and model-based rows under
simultaneous requests, the runner and SIGTERM; REST-089 ("a REST write is
a reload") is among them, because its oracle -- the write's trajectory
against the reload's -- needs a graph nothing else steps.  Read a row's
conditions before relying on it there.

### The coupling inventory's narrowed cells, lifted

A narrowed cell meant *untested*, not broken, and 538 of the coupling
inventory's cells were narrowed.  `tests/core/coupling_domains.py` builds a
memoryless two-member group in each numeric domain -- float64 and a float32
member beside a float64 one under x64, bfloat16, float16, `jax.vmap` of the
step, a multi-rate graph, a sub-cycled group, a quadratic predictor (with
IQN-IMVJ's Jacobian reuse), a checkpoint restart, `run_adaptive`, and a
member sharded over four CPU virtual devices -- so a row's claim is written
once and run in every domain, its oracle the closed form in float64 from the
parameters as each dtype stores them.  The claims *about* `finfo` (the dead
band, the overflow and underflow rules of the norm, CPL-011, 043 and 044) are
held at each dtype's own `tiny`, `eps` and `maxexp`; the acceleration rows
(CPL-049, 050, 061, 073, 074, 076), the two-group independence (CPL-079), the
non-float leaves through the adjoint (CPL-145) and four configuration rows
(CPL-014, 015, 016, 142) run in the domains their conditions left out; and
the topology harness runs its renaming, build-order and identity-relay
invariances (CPL-180 to 183) and Jacobi's indifference to its members' order
(CPL-078) in eight more domains.  188 cells in 24 rows are lifted; 350 stay
narrowed, each still named in its row's "Not claimed for" clause.

Four findings, none a wrong state, gradient or verdict on a common path:

- **The Aitken guard's cost** (MADD-ANO-170, CPL-062 `failing`): the guide
  says the two-pass guard adds at most one pass to Aitken's own exit; it can
  add two, in float32 as in every domain, because it holds the predecessor to
  its raw residual and the pass to its estimate.  The returned state is
  converged; the cost sentence is wrong.
- **A typed PRNG key cannot be checkpointed** (MADD-ANO-171, since 0.1.0):
  `save_state` raises `TypeError` on one.  The invariances' restart cells run
  on the structures without their key leaves.
- **`reset_state()` inside a differentiated loss** on a predictor group raises
  `UnexpectedTracerError` on the second transform (MADD-ANO-172, new in
  0.4.0).
- **`run_adaptive` hangs on a NaN error norm** (MADD-ANO-173, since 0.1.0): a
  diverging group under `run_adaptive` never returns.

Some cells were tightened rather than lifted: `atol` is compared after
rounding to each field's dtype (CPL-011); a 16-bit group meets CPL-049 and
CPL-050 only at a rate and criterion whose residual ratio it resolves (a
stalled iterate rejects the ratio, CPL-054's stall); under `run_adaptive`
one controller chooses the step for the whole graph, so two groups keep to
themselves there only when both converge (CPL-079); and `run_adaptive`'s
report takes the estimate from the second half step and `iterations` from
the larger, so CPL-049's "past the first pass" cannot be read there.

### The coupling runtime, round-5 audit (against the claims inventory)

A context-free audit attacked the inventory row by row, with reproducers.
It found the gradient bound short of its claim in two ways, the spectral
bound in the wrong weights, two silent configuration defects that every
release shipped, and four claims to restate.  All are fixed or restated.

**The gradient bound applies the resolvent to each secant exactly**
(MADD-ANO-142, introduced and fixed in this cycle).  It took the factor
the spectral bound applies -- the Arnoldi resolvent norm, which is
`(I - J)^{-1}` *restricted to the Krylov space* of the start vector and
the residual -- and multiplied the secant's norm by it.  A secant need not
lie in that space: on a scalar Gauss-Seidel ring with a rank-one
Jacobian the restricted norm was 8.57 where the full one is 45.2, and
the bound read 5.2x below the true gradient error with
`gradient_bound_usable=True`.  The resolvent is now applied to each
probe's secant through the range basis (one JVP per probe), the
Newton-Kantorovich check takes the full-operator resolvent norm (exact
from the range basis; with the restricted factor it passed at `h` up to
1.75 on the same ring made nonlinear), each probe is brought to the
state's relative size by a power of two, and a probe whose tangent is not
finite makes the bound NaN instead of dropping out.  A new oracle compares
the bound with exact dense tangents on dense, rank-one and non-normal
groups of 6 and 12 coupled entries at early exits; twelve of its fifteen
cases failed before.  The measured ratios tightened (concave map
1.1–2.1x, affine 1.35x, a spring pair 1.2–1.7x) and the hidden-slow-mode
fixture moved to a curvature the full-norm check certifies (25x there).

**Each entry of an array constant is its own probe** (MADD-ANO-143,
introduced and fixed in this cycle).  A field-valued parameter was probed
along one |c|-weighted random direction, which its large entries
dominate: the gradient in the small entry of a two-entry gain was 23.6x
above the bound, usable.  Every entry of a constant of at most 64 entries
is now probed; a larger constant is probed as a whole and
`coupling_report()` names it.  The bound costs `11 + k + 5 n_p` JVPs
per group per step, `n_p` the probes.

**`spectral_error_bound` is in the returned state's weights**
(MADD-ANO-146, introduced and fixed in this cycle).  Its resolvent factor
is measured there, but the residual it multiplies divided each field by
the larger of its magnitude now and after one more pass, so on a group
still growing toward its fixed point the bound read 0.94x the true
distance in the norm it documents.  The factor now carries the ratio of
the two weightings of `F(x) - x`.

**A coupling group is one block in the schedule** (MADD-ANO-144, since
0.1.0).  Where a group is only part of a larger feedback loop, an
outside node added between two members was scheduled between them, ran
after the whole group all the same, and read the later member one step
late -- the same graph built in another order stepped differently.  The
members are now placed together, back edges are decided over the block
order, and `compile()` warns when a group is a strict subset of a
feedback loop, naming the outside edge it reads late (which follows the
build order).

**`CouplingGroup` refuses out-of-range numbers** (MADD-ANO-145, since
0.1.0).  `waveform_iterations=0` on a sub-cycling group ran no sweep: the
members never stepped and the report had no entry.  Counts must be
integers `>= 1` (`jacobian_reuse >= 0`), thresholds finite and `>= 0`,
`relaxation` finite and `> 0`.

**Restated.**  Under `vmap` the report slots can differ by ulps (the
amplification by up to 470) and a verdict within them of the threshold
can flip -- 19 passes against 20 measured -- so CPL-096 says so; making
every reduction and divide of the batched program round as the unbatched
one does is not in XLA's guarantees.  `windowed_loss`'s mask reads the
last waveform sweep's verdict.  A predictor leaves the fixed point where
it is and moves the returned iterate within the tolerance.

Paths this changes: every `diagnostics=True` report under `"ift"` (the
gradient bound's value and its cost, the spectral bound's factor where a
field grew across the last pass); the schedule and back edges of a graph
in which an outside node was added between a group's members; and
construction of a group with out-of-range knobs.  States are unchanged
elsewhere (see the PR's dump).

### The coupling runtime, configuration interactions and graph shapes

Coupling's defects have lived in *interactions* between settings -- the
deprecated `"fori"` loop with Aitken on a 16-bit group, `"fori"` with
IQN-IMVJ dropping `jacobian_reuse` -- and in graph *shapes* no generator
drew, such as a reader added before a group's members (MADD-ANO-120) or an
outside node between them (MADD-ANO-144).  A test-only harness now
attacks both.

**A covering array of the group's knobs.**
`tests/property/covering_array.py` is a deterministic IPOG generator with
constraints.  Over solver, schedule, acceleration, norm, linear solver,
predictor, sub-cycling, interpolation, waveform sweeps, diagnostics,
`strict_convergence`, dtype and budget, it gives 65 configurations in
which every valid pair and triple of values appears.  Each configuration
runs five oracles over four steps:

- a monolithic float64 reference (below);
- `fori` against `ift`;
- `diagnostics` on against off, bit for bit;
- `strict_convergence` against the report;
- every usable bound against the truth.

Every float32 configuration passes all five.  The float64 ones run in a
subprocess under `jax_enable_x64` in the slow lane.

**A topology generator** (`tests/property/coupled_topologies.py`) draws
whole graphs of linear relays:

- rings, stars and nested cycles as groups, with one or two groups;
- outside nodes in a group's strongly connected component, and two
  groups in one component;
- ungrouped cycles, and drivers and readers in any build order;
- three additive edges into one port;
- flux, mapped and transformed edges, and nodes of two sizes;
- integer, boolean and typed PRNG-key leaves, and three-argument nodes.

**The monolithic reference.**  Every node is affine, so a step is one
linear system.  Each edge reads its source's new value, except a back
edge, which reads the old one.  The documented rule picks the back edges,
restated in the test rather than called, and the system is solved in
extended precision.  The oracle is local and exact:

- each node outside a group sits within its own float rounding of its
  update, at the values that rule says it reads;
- each group's defect stays within what its *reported* residual allows,
  through the identity `x - Phi(x) = -(I - L)(F(x) - x)` of an affine
  pass, so the report must describe the state returned, converged or
  not;
- the whole state stays within the propagated allowance of the solve.

Renaming, build-order and identity-relay invariances run on the same
structures.  The harness found four defects, all carried since 0.1.0, each
pinned by a strict xfail.  MADD-ANO-158 and 159 are resolved in 0.4.0;
MADD-ANO-156 and 157 stay open, deferred to 0.5.0 (see "[The known
coupling, sysid and FMU findings, fixed](#the-known-coupling-sysid-and-fmu-findings-fixed)"):

- **A flux edge across a group's boundary fails to trace**
  (MADD-ANO-156).  A member's flux reaches only other members, and an
  outside node's flux only other outside nodes, so `drv.q -> g0` and
  `g1.q -> sink` raise `KeyError: 'q'`.
- **An ungrouped cycle staggered on a flux edge fails to trace**
  (MADD-ANO-157).  A back edge reads the old state, which holds no flux.
  The ring `a.q -> b`, `b -> a` runs when built `a, b` and raises when
  built `b, a`.
- **A typed PRNG key in a member fails under `"ift"` with Aitken or fixed
  relaxation** (MADD-ANO-158).  The accelerator is sized from every field,
  the key included.  MADD-ANO-059's fix reached every other path.
- **A group joined only through an outside node reads that node a step
  late, silently** (MADD-ANO-159).  The group `{a, b}` with `a -> c -> b`
  and no cycle runs as one block before `c`.  `compile()` warns only for
  strongly connected components, so it says nothing.

CPL-181 also overstates.  It says the `add_node` order of every node
outside a group does not reach the result, but in an ungrouped cycle it
picks which edge is read late.  The row is now ambiguous, with the
wording the tests support.

Paths this changes: none in the library (test code, records and
registry only).

### The coupling runtime, round-6 audit (against the claims inventory)

A context-free audit attacked the inventory again, with the covering
array and the topology generator in place.  It found the spectral bound
in the wrong norm under the interface norm with an edge transform, a
16-bit group's norms overflowing in their own dtype (shipped by every
release), the gradient bound short of its claim far from the fixed point,
and two docstrings to restate.  All are fixed or restated; IQN's stale
secant columns (the verdicts stay honest) are recorded for 0.5.0.

**Under `convergence_norm="interface"` the spectral bound is in the
norm's own reading** (MADD-ANO-187, introduced and fixed in this cycle).
That norm measures each internal edge's source value *after* the edge's
transform, each over its own magnitude, while the spectral analysis took
its weights from the raw source fields: the bound multiplied a residual
in one set of coordinates by a resolvent in another.  With an affine
offset on an edge (a unit conversion's 273.15; 1000 in the reproducer) it
read 0.0014x the true distance, and with the registered `"extract_last"`
0.078x, both with `spectral_usable=True`.  The report's spectral triple
is now taken on the reading itself: the reading `y = Φ(x)` iterates by
its own map, whose Jacobian `Φ′ G′` has `dF/dx`'s non-zero spectrum, and
it is applied through state tangents (an Arnoldi that carries each Krylov
vector's preimage), so no transform is ever inverted.  On the audit's
fixtures the bound now reads 1.001-1.1x the true distance at the caps
where the Jacobi residual is small, and more at the others.  The gradient
bound keeps the state's own analysis, in the raw fields' weights, and
stands only where that one settled.  Eight more Jacobian-vector products
per step, on such a group only.

**A 16-bit group's change is measured, summed and held in float32**
(MADD-ANO-186, every release).  The norms squared `|dx| / (rtol max|v|)`
and summed in the group's own dtype, from a weakly typed zero that took
float16, and divided by an int32 entry count that float16 holds as `inf`
above 65 504: a float16 group of two 40 000-entry fields read
`residual=0.0` after 8 passes and stopped 4.1% from its fixed point with
`converged=True` under "mixed" and "interface", where the float32 group
runs 15 passes to 0.94% (on the tagged releases, whose test is the raw
residual, the same group stops at 4.1% where the float32 group stops at
5.1%, with a residual that reads 0.0).  At the default `rtol=1e-6` a change of one
float16 ulp squares past 65 504, so the residual read `inf` on a finite
state and `strict_convergence` reported divergence ("no larger
max_iterations would help") where the cap was the cause; the precision
floor read `inf` for every `rtol` below 3.8e-6.  Each field is now widened
to float32 before its change is formed -- through an
`optimization_barrier`, because XLA's excess precision otherwise measured
a node's unrounded output, and the two solvers then stopped on different
passes -- the sums start from a float32 zero, and a 16-bit group's
residual, its loop carries and its `_meta` slots are float32.
`strict_convergence` decides "non-finite state" from the state (a field
that is NaN, inf or beyond the range its dtype can measure a change at),
so a finite state whose estimate overflowed -- at `rtol=1e-25` in float32
too -- is told to loosen the tolerance.  The L2 norm's overflow rule
(CPL-043) applies in float32 as well, so a float16 field is measured up to
65 504, where it read `inf` from `2**14` up and never converged under L2.
A float32 or float64 group's norms, residual, passes and state are bit-identical
(the coupling dump's 125 float32 configurations: state, report and gradients;
48 float64 ones, a linear and a bilinear pair under every norm and both
solvers: state and report).  The one report key that moves is the gradient
bound's, below.

**The adaptive steppers' error norm had the same defect** (MADD-ANO-189,
every release).  Sweeping the tree for the 16-bit norms' pattern found it
at the other norm the library takes: `_tree_error_norm`, the step-doubling
error that `run_adaptive`, `run_adaptive_scan` and `build_adaptive_step`
share, summed its squared ratios from a weakly typed zero and divided by an
int32 element count.  Above 65 504 elements of a float16 state it read 0.0
(80 000 elements eight ulps apart, closed form 0.775) or NaN: `run_adaptive`
on a 70 000-entry float16 field returned a NaN state after one accepted
step (on the tagged releases it did not return), where a 1 000-entry one
takes the float32 state's 13 steps.  Below that size a ratio above 256 read
`inf` on an ordinary rejected attempt.  And the 16-bit norm made the next
timestep 16-bit, which `run_adaptive_scan`'s carry refuses, so the scan
raised a scan-carry `TypeError` on any bfloat16 or float16 state.  Each
leaf is now widened before its difference is formed and the sum starts
from a float32 zero: both steppers take the float32 state's steps on the
70 000-entry field, and the scan runs a 16-bit state to its end time.  A
float32 or float64 state takes no new operation.  Under `jax_enable_x64`
the scan still refuses a 16-bit state, as it has always refused a float32
one there: its clock carry is float64 and takes the controller factor's
dtype (MADD-ANO-190, open; `run_adaptive` steps both).

**The gradient bound holds, or is withdrawn, far from the fixed point**
(MADD-ANO-188, introduced and fixed in this cycle).  On a pair bilinear in
its state stopped at `max_iterations=2` the Kantorovich check passed
(`h = 0.19`) and the bound read 0.986x the true gradient error, usable;
on a pair 57% from its fixed point `h` read 0.37 and the bound 0.65x.  The
check took the Jacobian's change along the Newton step alone, which there
is far below its change in the directions the step hardly moves (the
fixed point lay outside the radius the check certified), and the part of
`x* - x_k` the Newton step misses was not carried.  `h` is now also
Deuflhard's affine-covariant `||(I - J)^{-1} (J(x_k + delta) - J(x_k))||`,
an operator norm on the directions the Jacobian reads (`3k` more JVPs):
0.65 on that pair, no certified bound.  Each probe adds the Newton step's
second-order miss, and Kantorovich's radius is formed without the
cancellation that read it as 0 for `h` below 1.5e-8 in float32 (and as
1.2-4 times the Newton step up to 3e-7).  Over 240 random
bilinear pairs at caps 2-4 every usable bound (130 of them) now holds,
at 1.002x the true error and up; on the base, 180 of the same draws had
five usable bounds below it (0.65-0.985x).  The audit pair reads 1.21x at
cap 2, and the far pair fails the check.  What it costs: one point of the
rank-one ring behind CPL-095, certified at `h = 0.29` along `delta`, reads
0.64 affine-covariantly and is withdrawn, and so is a twelve-square
Gauss-Seidel chain stalled 21% from its fixed point (0.48 along `delta`,
0.57 affine-covariantly) (`beta` times the operator change,
tried first, withdrew all three of that ring's certified points).  All three
corrections vanish on an affine map: the coupling dump's 125
configurations keep their state, report and gradients bit-identical but
for one affine group's gradient bound, an ulp apart (the new operations
move how XLA fuses the rest).

**Restated.**  `rho_spectral` for a group with more than eight interface
scalars is an estimate from below only for a normal `dF/dx`: Ritz values
of a non-normal one lie in its field of values, which can pass the
spectral radius (1.17 on a Jacobi ring of nine relays at modulus 0.95,
flagged unusable).  `CouplingGroup.tolerance` is one threshold for the
whole group -- the root-sum-square over every entry of every field -- not
"tolerance * max|v| per field".

Paths this changes: every norm, the floor and the report of a bfloat16 or
float16 coupling group, under both solvers, `max_iterations=1`,
`run_adaptive*`, the sysid mask and the profiler (they read the float32
slots), and a checkpoint's slots (an old float16 slot is widened on load);
the report's spectral triple of an `"interface"` group with a transform on
an internal edge, under `diagnostics=True`; the gradient bound of every
`diagnostics=True` group whose Jacobian moves; `strict_convergence`'s
choice of message; and `run_adaptive`, `run_adaptive_scan` and
`build_adaptive_step` on a state with a bfloat16 or float16 leaf (the
error norm, and through it the accepted steps; the scan now runs such a
state).  Float32 and float64 states and reports are unchanged elsewhere.

### Krylov solves of a non-finite right-hand side

Every matrix-free solve in the library -- the IFT tangent and adjoint of a
coupling group, `ift_linear_solve`, and `sharded_cg` / `sharded_gmres` --
poses its Krylov iteration on the right-hand side framed by `max|b|`,
with a tolerance relative to it.  A NaN or infinite entry of `b` made that
tolerance NaN or `inf`, the zero initial guess passed lineax's test (and
the loop CG's) before a single step, and the solve returned zeros reported
successful.

**The IFT derivative of a coupled step read exactly 0.0** (MADD-ANO-154,
introduced and fixed in this cycle).  Under the default
`linear_solver="gmres"`, a tangent or cotangent with a NaN or `+-inf` entry
gave a zero derivative, where `"dense"` and `solver="fori"` read NaN: the
tangent in a constant whose derivative a `custom_jvp` makes NaN, and the
gradient of a least-squares loss with one NaN observation, at 2 and 60
coupled DOF under both iteration modes.  Neither the dense re-solve nor
the adjoint error fired, because the solve "succeeded".  v0.3.0 and v0.3.1
raised there instead: their absolute `1e-8` tolerance let GMRES run on the
NaN, and lineax refused its output.

**The public solvers returned zeros** (MADD-ANO-155, since 0.3.0).
`sharded_cg` returned zeros with `converged=True` for a NaN entry on its
default lineax backend and for `+-inf` on the loop backend, and
`ift_linear_solve(solver="cg")` returned zeros for a NaN entry with no
error; this cycle's relative default tolerance extended it to
`sharded_gmres` on lineax and to `ift_linear_solve`'s GMRES, and through
`differentiable=True` to the tangent and the cotangent.

Each solve now answers a right-hand side with a non-finite entry with NaN
in every entry -- what an honest Krylov iteration produces, its first
basis vector being `b / ||b||` -- and hands its backend zeros in its place,
so nothing iterates on NaN (an explicit `atol` ran lineax GMRES to its step
budget and raised) and no "failed" path, dense re-solve or adjoint error,
is taken.  A sharded result reads `converged=False` and a NaN
`residual_norm`.  The check is per right-hand side, so under `jacfwd`,
`jacrev` or `vmap` only the non-finite column is NaN.  The IFT scale
oracle now holds the non-finite case at every scale from `2**-100` to
`2**60` against dense and fori; coupling claim CPL-185 records it.

Paths this changes: the derivative of a coupled step under `solver="ift"`
(every `linear_solver`) whose tangent or cotangent is not finite; every
call of `ift_linear_solve`, `sharded_cg` and `sharded_gmres` (both
backends, primal and `differentiable=True`) whose right-hand side is not
finite.  A finite right-hand side takes exactly the path it took before:
364 derivatives and solves -- the IFT oracle's graphs at four scales, and
every public and private solver's primal, tangent and cotangent -- are
bit-identical to the tree before the fix on jaxlib 0.10.2, 0.11.0 and
0.11.2.  What is
unchanged: a non-finite *Jacobian* at a finite fixed point is a different
case -- below 50 coupled DOF it reads NaN through the dense re-solve, and
above it raises the documented adjoint error.  The gradient and spectral
bounds' power-of-two frames were checked for the same pattern and report
unusable there (`gradient_bound_usable` and `spectral_usable` False).

### The interface norm reads a mapped edge as the step delivers it

`convergence_norm="interface"` measures "the values the group's internal
edges carry".  The step turns an edge into a boundary input by one rule:
the edge's interface mapping, with the weights in `params["mappings"]`,
then its transform.  The norm, its float floor, the spectral analysis
behind `spectral_error_bound` and the report's own floor each read the
edge themselves, and each applied the transform and left the mapping out.
On a mapped edge inside a coupling group the criterion was therefore taken
on the source field, a value the consuming node never sees
(MADD-ANO-195, registered open with the interface-mapping fixes; mapped
edges are new in 0.4.0, so no release carried it).

Measured before the fix (CPU, jaxlib 0.11.0, float32).  With a mapping
that delivers the second entry of `[1000, x1]`, a change of `x1` from 1.0
to 1.1 read 0.0071 in units of `rtol = 1e-2` where the delivered value
moved by 9.09 of them, so the group was reported converged on that edge.
On the round-6 audit's `extract_last` pair with the selection written as
a `matrix_mapping`, `spectral_error_bound` read 0.064x, 0.318x and 0.074x
the true distance in the delivered values at `max_iterations` 2 to 4, with
`spectral_usable=True`: residual and bound agreed with each other and
described the wrong quantity.

**One definition, used everywhere: an interface edge is read as the step
delivers it.**  The edge rule is one function
(`maddening.core.edge._delivered`), which the step's boundary resolution
calls, and one generator built on it
(`acceleration._interface_readings`) gives every reader the same values:
the residual (`coupling_residual_interface`), the float floor
(`residual_precision_floor`), the reading the report's spectrum is taken
on, and the floor the report adds.  The two norm functions take the
step's mapping weights as a new `mappings=` argument (experimental).  On
the same pair the bound now reads 1.001x, 66.5x and 1.003x the distance
in the delivered values, the numbers the transform's spelling gives.

**The weights are the ones the step ran with.**  A mapping's weights can
be passed for one step (`gm.step(params=...)`, a fit, a sweep) and the
graph does not hold them afterwards, so the report cannot rebuild what a
mapped edge delivered from the returned state.  A group whose norm reads
a mapped edge therefore records its floor per evaluation in the step
(`_meta`'s `coupling_<key>_reading_floor`: NaN until a step writes it,
restarted with the group's other report slots), and
`coupling_diagnostics()` multiplies it by the pass's evaluation count.
Only such a group owns the slot.  The spectral analysis and the recorded
floor read the weights with their gradient stopped, as the analysis does
for the state: the report has no derivative.

**A delivered value is no finer than the field it came from**
(MADD-ANO-211, never released).  The floor counted each delivered value
at its own dtype's `eps`.  Under `jax_enable_x64` a float64 mapping
matrix applied to a float32 field -- which is what a NumPy matrix becomes
there -- or a transform that returns float64, delivers float64 numbers
that carry float32 rounding, and the floor was `2**-29` of what the
source resolves: a float32 pair stalled at `residual=0.0` behind edges
that cast to float64 reported `spectral_error_bound=4.4e-7` against a
true distance of 0.66, `spectral_usable=True`.  Each delivered value is
now resolved at the coarser of its own dtype's `eps` and its source
field's, and the same pair reads 238 (362x the distance, the float32
edges' figure).

**Found, not fixed** (MADD-ANO-212, open, never released).  The floor
weights a same-pass read by its relative gain along the source's own
state, which a difference of two entries of that field cancels.  A
Gauss-Seidel group reading `u[0] - u[1]` of a field at 1000 through a
mapping row `[1, -1]` stalls 0.3% from its fixed point at `residual=0.0`
and reports a bound of 0.054x the true distance, `spectral_usable=True`
(2.6e-4x at 1e5), under every norm, before and after this change.  Under
Jacobi the bound holds (37x).  Workaround: Jacobi, or have the producer
hold the difference as a field of its own.  CPL-092 and CPL-102 now state
the condition; the measured `diagnostics="rounding"` level planned for
0.5.0 closes it.

The topology reference (`tests/property/coupled_topologies.py`) had
restated the old rule -- "each internal edge's source" -- and agreed with
the defect.  It now models what each edge delivers, its per-push ring
carries a mapped edge and a mapped-then-transformed one, and the same
library before the fix fails it.

Paths this changes: under `convergence_norm="interface"`, a coupling
group with an interface mapping on an internal edge -- its residual, the
pass it stops on and so its state, `converged`, `error_estimate`,
`precision_limited`, the spectral triple and bound, its `_meta` (one more
slot), under both solvers, every acceleration, `max_iterations=1`,
`run_adaptive*`, the sysid convergence mask and the profiler (they read
the slots); and, under x64, the floor and bound of an interface group
with an edge that delivers a wider dtype than its source field.

### The known coupling, sysid and FMU findings, fixed

The domain matrix, the coupling topology harness and the parameter
acceptance oracle above left nine findings pinned by strict xfails.  Seven
are fixed here or, for the FMU's advertised bound, by the round-8 FMU fix,
each with a regression test that fails on the tree before the fix and
mutants that the tests catch; two loud ones are deferred.

- **`run_adaptive_scan`'s gradient where step doubling is exact**
  (MADD-ANO-160, since 0.1.0, resolved).  The cause was the step-size
  controller, not the coupling solve: the step-doubling error norm is a
  square root, whose derivative is infinite at zero, and the backward pass
  multiplied it by the zero the controller's `max(error_norm, 1e-10)`
  sends back.  Wherever an attempt's full step and two half steps agreed
  exactly -- an accelerated group at its fixed point, a memoryless or
  resting node, a step past `t_end` whose change rounds away -- the
  gradient read NaN, or exactly 0.0 depending on the surrounding program.
  The norm is now taken through a double `where`; the gradient matches
  `run_scan`'s, the closed form `1 / (1 - g_a g_b)` and a central
  difference of `run_adaptive`, with and without a coupling group.
- **16-bit IFT gradients** (MADD-ANO-161, since 0.3.0, resolved).  The
  tangent and adjoint solve of a bfloat16 or float16 group runs in float32
  on the group's own operator and is cast back.  The Krylov criterion is
  four units of the 16-bit dtype's roundoff: a float32 criterion cannot be
  met by an operator whose products are rounded to 16 bits (GMRES stalls
  and, above 50 coupled entries, raised the adjoint error), and at four
  units it converges within about twice the dense LU's error.  The
  derivative is the 16-bit map's, to the 16-bit output's resolution.
- **A 16-bit group's spectral report** (CPL-087).  The five spectral slots
  are kept in the analysis's float32, so `rho_spectral` and the bounds are
  no longer rounded to the group's dtype.  The radius is still computed
  from the map's own Jacobian-vector products, rounded to that dtype, so
  the docstring now says a 16-bit group's radius is exact to about one
  `eps` of its dtype times the Jacobian's norm (0.0036 measured on a
  radius-0.90 bfloat16 pair), not to float32.
- **`strict_convergence` on a step spanning several devices**
  (MADD-ANO-162, never released, resolved).  The check runs inside a
  `shard_map` replicated over the step's mesh, so every device raises at
  the same point and none waits at a collective; the callback's result is
  a zero token OR-ed into the bits of the checked state, an exact identity
  that keeps the check live without gathering a sharded field, with a
  derivative rule of its own so `jax.grad` through the check is the
  unchecked step's (a bit operation's derivative is zero).  The graph's
  entry points wait for such a call and drain the mesh before the error
  propagates.  The reproducer raises in about a second and the process
  exits cleanly.
- **A typed PRNG key under `"ift"` with Aitken or fixed relaxation**
  (MADD-ANO-158, since 0.1.0, resolved): the accelerator is sized from the
  group's floating fields only on that path, as on every other.
- **A group joined only through an outside node** (MADD-ANO-159, since
  0.1.0, resolved by saying so).  `compile()` warns for every back edge
  that joins two strongly connected components, naming the edge and the
  group whose block forced it.  The one-step lag is what such a group
  computes -- its members break the documented precondition -- so the
  warning names it and the remedies; it is not removed.
- **The precision warning's remedy under x64** (SYS-024): with float32
  leaves in an x64 process the `PrecisionLimitWarning` said to re-run under
  x64, which was already on.  It now names float64 leaves there, and so
  does the range-limited warning.
- **`best_loss` is the loss of exactly the parameters returned** (SYS-071,
  and SYS-050's `losses[0]`).  The fitters return a leaf no step moved as it
  went in, but their objectives evaluated its `constrain(unconstrain(p))`
  round trip, which for a `log` leaf lands an ulp away (30.0 in float32,
  30.03 in float64): `best_loss` was the loss of a point the fit did not
  return, off in the last bits.  `fit`, `fit_lm` and `fit_multiple_shooting`
  now evaluate exactly the values `params` returns -- an untouched leaf at
  its input, its derivative still the round trip's -- so without the
  `hold_undetermined` guard `best_loss`, `losses[best_iteration]` and the
  loss of the returned parameters are one number.
- **The FMU's advertised open bound** (the acceptance oracle's N2) is the
  round-8 FMU fix's (see "[What the round-8 audit found in the FMU
  export](#what-the-round-8-audit-found-in-the-fmu-export)"): an open bound is
  advertised as the outermost value `ParamSpec.check` accepts.  This batch
  adds the cases the oracle's broad draw found at a `logit` bound of zero:
  under `ParamSpec(bounds=(-1, 0), transform="logit")` the description
  advertised `max = -TINY`, whose coordinate rounds onto the bound, so every
  door refused it but a bridge over a sidecar with no specs took it.  The
  round-8 fix advertises `-2**-25` there; the oracle and the description
  tests now pin both ends of `(-1, 0)` and `(0, 1)`.

**Deferred to 0.5.0.**  MADD-ANO-156 (a flux edge across a group's
boundary fails to trace) and MADD-ANO-157 (a flux edge an ungrouped cycle
staggers fails to trace) stay open: both raise before anything is
returned.  Fixing them means carrying fluxes between the step and the
group's block in both directions and deciding what a staggered flux edge
reads (the producer's previous flux is not in the state), which is more
than this batch should take on.

**Code paths these fixes change**, beyond each reproducer, each with
neighbouring tests: every `run_adaptive_scan` gradient (the controller's
derivative is unchanged wherever the error is nonzero; tested on a moving
forcing against `run_adaptive`); every IFT tangent and adjoint of a 16-bit
group under both linear solvers and above the dense fallback's size;
every `strict_convergence` check on a graph with a sharded node or a field
placed over several devices, in `step`, `run`, `run_scan*`, `run_sweep`
and `run_adaptive_scan` (a converged strict step is bit-identical to the
unchecked one, keeps its partitioning, and has the unchecked step's
gradient); every objective of `fit`,
`fit_lm` and `fit_multiple_shooting` (every recorded loss of a run with a
leaf whose round trip is inexact moves in its last bits; a leaf whose round
trip is exact is computed exactly as before); and `compile()` of any graph
with a group whose members are joined through outside nodes off a cycle,
which now warns.

### Choosing one: measured, not extrapolated

Everything this project said about which coupling option suits which
problem used to be extrapolated from one measured graph and a heat pair.
[Choosing a coupling algorithm](../developer_guide/coupling_algorithm_guide.md)
replaces that with a sweep over eight synthetic graph shapes — chain,
star, ring, a contraction-factor sweep, two grid fixtures, a two-group
graph and a slow-drift case — each isolating one property that decides
the answer.  `benchmarks/coupling_fixtures.py` defines them,
`benchmarks/bench_coupling_sweep.py` runs the product of two iteration
modes, five accelerations and two convergence norms over them, and the
three recorded JSON files under `benchmarks/results/` are what the guide
cites.  The page opens with a table you can read by graph shape, says
where the measurement contradicted the prediction, and is explicit about
what a CPU-only run cannot tell you — nothing here measures Jacobi's
parallelism.

The files were re-recorded on the release tree on 2026-09-25, because
the first recording (2026-09-18) predated the error-bound criterion, the
relaxation correction and the `HeatNode` rod-end fix, and the guide was
still quoting it.  The rod-end fix (MADD-ANO-007) had doubled the three
heat fixtures' interface gain from `Fo` to `2·Fo` and pushed two of them
past a time-step limit of their own: two slabs exchanging end-cell
temperatures are unstable above `Fo = 3/8` once the exchange converges
(at the default `stencil_order=2` the fixtures use; 0.226 at
`stencil_order=4`, MADD-ANO-050), and `compile()` now warns about such a
pair past its limit.  `slow-drift`, recorded at 3.18 passes and 100% converged, took its
whole cap of 25 on every step and left float range within 90 steps.
Each heat fixture now runs at half its old Fourier number, so it
contracts at the rate it was designed at, and the builders refuse a slab
pair at or above 3/8.  Against the first recording the fast sweep takes
9.9% more iterations and 13 of its 420 rows no longer converge on every
step, ten of them under `acceleration="fixed"`.  Two recommendations
changed: the "fixed point that barely moves" row now starts from
`gauss-seidel` / `none` (ω = 0.8 is what brings *Jacobi* level with
it, not the best configuration outright), and the recorded
`fixed_point_agreement` is now dominated by the `iqn-*` /
interface-norm blind spot the guide already warned about — up to 0.93
of a field's scale on `stiff-pair-0.95`, where the first recording read
at most 4.8e-3.  `tests/core/test_coupling_fixture_invariants.py` now
re-measures one row of every recorded fixture against its file (five on
every push, the rest in the slow lane), so a file the code has moved
away from fails instead of being quoted.

`tests/core/test_coupling_fixture_invariants.py` asserts the properties
that outrank any timing: every configuration of a fixture reaches the
same fixed point, Gauss-Seidel on a ring is order-dependent and Jacobi is
not, a group past the contraction limit reports itself unconverged while
IQN converges it, IQN on a rank-deficient one-DOF interface agrees with
plain iteration, and two groups in one graph keep their own schedules.

## `AdaptiveNode`

`maddening.nodes.adaptive.AdaptiveNode` (`MADD-NODE-009`) is the
frozen-active-set adjoint pattern for adaptive solvers.  A subclass supplies
`compute_active_set` (any fixed-shape `jnp` selection rule) and
`solve_frozen` (the masked solve, through `ift_linear_solve`); the base
class wires them into a JAX-traceable `update` over a padded `(c, mask)`
state.  Adaptivity changes which mask entries are true, never an array
shape, so the step runs under `jit` / `lax.scan` unchanged.  The selection
is committed under `stop_gradient` and coefficients off the mask are zeroed.
Physical parameters live in the graph parameter pytree with the subclass's
`ParamSpec`s, so `fit` and `fim` reach them.  Both
`compute_active_set` and `solve_frozen` are `@abstractmethod`; `objective`
stays concrete because it is optional with the gate off.

Guides: {doc}`/algorithm_guide/nodes/adaptive_node` and
{doc}`/developer_guide/adaptive_node`, with benchmark `MADD-VER-004` (a
Green's-function reference for `-u'' + u = f`).

The first concrete subclass ships too: `WaveletAdaptiveNode`
(`MADD-NODE-010`, `EXPERIMENTAL`), an interpolating-wavelet solver for
`(-Laplacian + m) u = f` on the unit cube in 1-D, 2-D or 3-D, periodic or
Dirichlet, with Cohen–Dahmen–DeVore bulk chasing for the active set and a
gathered `k x k` frozen solve.  It declares `DiscretizationOrder(spatial=2)`
and measures 2.000 by MMS (`MADD-VER-014`); the adaptive reading is within
1e-2 of the full solve at the default budget (`MADD-VER-015`).  Its
developer guide answers the seven author-facing contract items the audit
listed, with the test that pins each, and records what porting a real
subclass onto this API found awkward — a tuple parameter is a trainable
leaf, the full-basis-gradient override needs private plumbing, and the
eagerly-called hooks make compile time the whole cold-start cost until the
subclass jits them.  See {doc}`/developer_guide/wavelet_adaptive_node`.

A merged-tree audit before release found that the CDD seed -- every
level-0 function, not the coarse block alone -- exceeded the default budget
on 16 default configurations and the gathered solve silently dropped the
excess (`dJ/dtheta` five times too small at `dim=2, n_levels=2`).  The
budget is now sized and validated against the real seed, an oversized
active set is refused eagerly and NaN-poisoned under `jit`, the operator
assembly refuses a non-symmetric stencil instead of symmetrising it away,
the node constructs inside a `jax.jit` trace, and
`selection_diagnostics()` reports whether the budget or the 30-iteration
bound ended the selection.

The conditioning guard (`CONDITION_LIMIT`) says what it bounds.  Its
`kappa * eps` models a direct solve, but `frozen_solver="cg"` stops at a
relative residual of `rtol` (1e-6 float32, 1e-10 float64), so on that path
the bound is `kappa * max(eps, rtol)` (float32 cg refuses a periodic mass
below about 2e-2).  That bounds a CG solve that converges, not whether it
does: at conditionings the bound accepts, lineax's CG stagnates or breaks
down (float32 at 256 points from `mass=0.3`), and 4x or 16x the step budget
does not help.  An eager solve that does not converge is now a `ValueError`
naming the conditioning and the fixes instead of equinox's "increase
`max_steps`"; under a trace the error still comes from lineax.  The default
`frozen_solver="gather"` is unaffected.

### What the gradient actually is

The node shipped claiming a "one-sided (Clarke) subgradient at the kinks".
That claim was wrong and the audit caught it.  The frozen-set objective has
*jump* discontinuities at active-set switches — the coefficients and sensor
weights of the two modes differ — so it is not locally Lipschitz there and
**no Clarke subgradient exists**.  What the node returns is exact *within*
an active-set region and ignores the set's dependence on the parameter;
across a boundary the objective jumps and gradient-based optimisation sees a
first-order error equal to the sum of the jumps crossed.

**Measured on the 1-D sine toy at n_max=256, k=16: integrating the returned
gradient over θ ∈ [0.40, 0.50] gives −1.5808e-3 against a true change of
−2.3598e-3 — a 33 % shortfall, exactly the sum of the 27 jumps crossed.**
It is negligible with a large basis budget: 4e-8 at k=64.

This is recorded as **`MADD-ANO-003`** (open, context-dependent) and
asserted by `tests/nodes/adaptive/test_active_set_switch.py`, which bisects
a switch and checks that the objective's jump does not vanish as the step
shrinks, that central finite differences across it diverge as `1/h` while
the returned gradient stays finite, that inside a region the gradient is the
derivative of the branch the forward pass selected (checked against a finite
difference *of that branch*, with a step verified to cross no switch — not
the exact-baseline pattern the design spike warns against), and that the
integral of the gradient plus the crossed jumps reconstructs the objective
change.

### The diagnostic, renamed and de-escalated

`blindness_ratio`, the name in earlier development builds of this release,
is now **`gradient_capture_ratio`** (the old name still resolves and warns), because the number measures active-set-budget
adequacy rather than a symmetry trap: at a fixed non-symmetric point it is a
function of `k` alone — **0.16 / 0.57 / 0.85 / 1.00 at k = 4/8/16/32, at
every `n_max`**.  `blindness_threshold` is likewise
`gradient_capture_threshold`, with a deprecated constructor alias.

The new cold-start check `AdaptiveNode.check_gradient_capture(params=None,
*, state=None, on_blind=None)` **warns** on a low ratio, naming the measured
value, the threshold, which of the two causes the evidence points at, and the
remedies in order.  Under the default `on_blind="warn"` it never raises — not
even when `frozen_gradient_vanishes_at` is true.  That check is *necessary*
for a Palais fixed point and not sufficient: it also fires at an ordinary
stationary point of the frozen objective, i.e. where a successful fit
converges, so refusing there would hard-error a user through the escape hatch
they had chosen.  `AdaptiveNodeBlindnessError` comes only from the opt-in
`on_blind="raise"` (and from `cold_start()` when one `symmetry_break` does not
lift the ratio).  `on_blind="ignore"` skips the check entirely.

`gradient_capture_ratio` additionally warns when the full-basis gradient with
respect to a **trainable** parameter is *bitwise* zero.  A direction the
objective is genuinely flat in gives a small gradient, not an exact `0.0`, so
this is a near-certain signature of a parameter baked into a constant in
`__init__` — the one failure where `jax.grad` and a finite difference agree on
the same wrong answer.  The message no
longer prescribes `cold_start()` / `symmetry_break()` where they make
matters worse (**measured 0.565 → 0.060 at a budget-limited point**), and
the guide's "validated" `K/n_max = 0.016` row is now reachable through the
default API.

The diagnostic is also cheap enough to leave on.  It costs two gradient
evaluations — 8–10× an unguarded `initial_state()` — and `initial_state()`
is called from `add_node`, `reset_state`, the profiler, the REST API, the
sharded paths, the FMI model description and the Hypothesis strategies.  It
is now memoised per instance and parameter point, skipped when the gate is
off, when `on_blind="ignore"` or when the node has no trainable leaf, and
can be disabled process-wide with `MADDENING_ADAPTIVE_DIAGNOSTICS=0` or
`maddening.nodes.adaptive.set_adaptive_diagnostics(False)`.  It also
re-selects the active set at the evaluated parameters instead of reusing
`state["mask"]`, which used to make a state selected at a healthy point
report a healthy ratio (1.005) at an exact trap.
`check_gradient_capture(gm.params["nodes"][name])` runs it at a graph's live
pytree — the point `sysid.fit` optimises, which the constructor-time gate
never saw.

The rest of the diagnostic surface: `frozen_gradient_vanishes_at`
(`is_trapped_at`, its development-build name, still resolves and warns: a Palais trap implies a vanishing
frozen gradient, but so does an ordinary stationary point, so a `False` rules
a trap out and a `True` does not establish one), `symmetry_break` (an
anisotropic step along the full-basis gradient, trainable leaves only),
`cold_start()` with one automatic escape, `mask_safe(mask, x, fill)` (the
inner half of the double-`where` idiom), and the module-level
`set_adaptive_diagnostics` / `adaptive_diagnostics_enabled` switches.  The
constants `blindness_break_delta = 0.05` and `D_threshold = 5` remain
documented class attributes.

`mask_safe` exists because of a trap the base class cannot repair: masking
the solve output protects the value, not the tangent.  A `solve_frozen` that
evaluates a singular expression on inactive entries returns a clean forward
pass and a `NaN` gradient.  The base class warns when it can see non-finite
pre-mask coefficients and offers `mask_safe`, which the authoring guide now
prescribes.

### Stability level and structural params

`AdaptiveNode` and `AdaptiveNodeBlindnessError` are **`EVOLVING`, not
`STABLE`**, and `maddening.core.solver_utils.ift_linear_solve` is back at
the `EXPERIMENTAL` it had before the merge.  Nothing has shipped, so
lowering the promise is free and raising it later would not be.  The 0.4.0
API freeze picks the final level, informed by two open questions now written
down in the authoring guide: whether `ift_linear_solve` should expose
`restart` / `max_steps` / `stagnation_iters` and stop leaking `lineax` /
`equinox` runtime error types, and whether the three hooks need
`boundary_inputs` / `dt` so an adaptive node can be an edge *sink*.

`n_max` is no longer stored in `AdaptiveNode.params` — it is structural —
while `blindness_gate`, `on_blind` and the diagnostic constants are, so a
node survives a config and USD round trip.  `params_pytree()` excludes them
so a fit never sees them as leaves.  A subclass whose basis size must round
trip declares its own integer parameter for it.  The constructor validates
`n_max` (a positive whole number; `2.7` and `"8"` are rejected, an integral
float from a JSON round trip is accepted), the diagnostic constants (finite
and non-negative) and `dtype` (floating, enforced on the coefficients when
given explicitly); the diagnostics reject a parameter key that is neither a
pytree leaf nor a constructor parameter instead of silently ignoring a typo.

Two round-trip defects were fixed along the way: `blindness_gate` was not
forwarded to `super().__init__`, so a node deliberately built with the gate
off reloaded with it on, and `n_max` *was* in `self.params` and was replayed
as a constructor keyword, so every subclass following the documented pattern
got `TypeError: got multiple values for keyword argument 'n_max'`.

## Multi-GPU and sharding

### Per-neighbour halo exchange

`exchange_unstructured(..., method="ppermute")` and
`ShardedUnstructuredNode(..., exchange="ppermute")` send one `lax.ppermute`
per *communicating cyclic shift*, each sized to that shift's largest
message, instead of one dense `(n_devices, n_ghost_max)` `all_to_all`
payload to every shard.  Results are bit-identical — verified over random
partitions, gradients, and a 10⁵-cell ring in the slow lane.

`exchange_traffic(layout)` reports cells moved per shard for both transports
plus the useful count, so the transport can be chosen from the partition
before any GPU time is spent: **a ring on 4 shards moves 2 cells with
ppermute against 8 with all_to_all**.  The default stays `all_to_all`
because the NCCL timings that would justify changing it need a real
multi-GPU host.

### Sharded boundary inputs

`ShardedStencilNode` replicated every boundary input, so a per-cell field
(an LBM `body_force` map, a `wall_mask_update`) reached each shard at its
global shape and broke `update_padded`, or silently mismatched — a sharded
LBM could only take a uniform force vector.  Grid-shaped inputs are now
sharded and halo-padded like state, on both the stencil and the unstructured
path; global-order inputs on the unstructured path are refused with a
pointer to `partition_value`.  When every shard is full the two orders have
the same length, and an interleaved 8-cell partition read the global-order
`[1..8]` as `[1, 5, 2, 6, 3, 7, 4, 8]`: on such a partition, unless it keeps
cells in global order, a per-cell input is refused in either spelling, and
the message says how to renumber the cells so that both readings agree.  The classification heuristic was also wrong:
it compared only the sharded axis's extent, so an `(n,)` input on an `n × n`
grid sharded on axis 0 was classified as grid-shaped; the input must now
match the full leading grid shape.  `HeatNode.update_padded` takes `params=`
like `update`, so a `ShardedStencilNode(HeatNode)` is calibratable and
differentiable like the unsharded node.

### `shard_axes`, var-keyword nodes and ragged grids

Three defects the independent cloud/sharding audit reproduced, each now
pinned by a property test:

* **`ShardedPointwiseNode` ignored `shard_axes`.**  The placement came from
  `P("devices")`, which names array axis 0 whatever the argument said;
  `shard_axes` was read only as a rank test.  `shard_axes=(1,)` on a
  `(3, 8)` state over 4 devices failed with an `IndivisibleError` blaming
  axis 0, an axis the caller had ruled out.  Any single axis is now
  honoured, and the leading axes are replicated.
* **An inner node declared with `**kwargs` never received `params`.**
  `ShardedStencilNode` probes `update_padded` once for three injected
  keywords and accepted a var-keyword signature for two of them.  The third
  fell back to the constructor constant: the physics was wrong by the
  difference between the two values and `d(loss)/d(rate)` came back exactly
  `0.0`, with no error anywhere.  No shipped node is written that way, so
  this was a trap for user-authored nodes.
* **A grid the mesh cannot split evenly is refused at construction.**  A
  pencil decomposition gives every device the same slab, so 17 cells over
  3 devices has no layout.  Both Cartesian wrappers now say so, naming the
  node, the cell count, the device count and the ways out; it used to
  surface as a `jax.errors.IndivisibleError` from inside `device_put` on
  the first step.  `ShardedUnstructuredNode` carries an explicit padded
  layout and takes any (device, cell) pair, which makes it a way out for a
  pointwise node but not for a stencil node: it now refuses one (see the
  MADD-ANO-037 entry below, and
  {doc}`/developer_guide/sharding_topology`).

`ShardedPointwiseNode.update` still applies no placement of its own, which
is deliberate: the operation is pointwise, so XLA's SPMD propagation keeps
a sharded input sharded and forcing a placement would add a resharding
collective to every step.  The consequence is worth knowing — fed an
unsharded array (a state that round-tripped through a host write) the step
runs correctly on a single device.  Start from `initial_state`, as
`GraphManager` does, or place the state yourself with the new `sharding`
property.

### A sharded `LBMNode` is the unsharded node, or it refuses

Two layouts made `ShardedStencilNode(LBMNode)` a different model from the
node it wraps, with no error; both date from v0.2.0 and were present in
v0.3.1.

* **A pressure face on a sharded axis.**  `update_padded` applies the Zou-He
  closure to the edge plane of its own slab, so a pressure-driven channel
  sharded along its inlet/outlet axis had every seam between shards forced
  to the face pressures: on a 16x10 D2Q9 channel on 2 devices the planes
  either side of the seam read density 0.990 and 1.010 and the centreline
  velocity was 2.10x the unsharded one.  The first sharded step that
  imposes a pressure on a face whose axis is split now raises, naming the
  axis to shard instead.  It fires at the first step rather than at
  construction because the faces act only when a pressure input arrives —
  a body-force-driven node may still be sharded along any axis.
* **The halo fill at the edges of the global grid.**  The wrapper's
  `boundary="edge"` default copied each edge cell into its own halo, while
  the unsharded node streams periodically: the same channel sharded across
  its walls ran 0.64% slow at the centreline after 200 steps.  A stencil
  node may now declare `halo_boundary()`, the fill under which its sharded
  step is its `update`; `LBMNode` declares `"periodic"`.  The wrapper
  refuses at construction a `boundary` that differs from a declared fill —
  **its own default `"edge"` included** — naming the node, both fills and
  `boundary="periodic"`.  The wrapper's STABLE signature and default are
  unchanged, and a node that declares nothing is wrapped exactly as before.
  With `boundary="periodic"` the same channel matches the unsharded node to
  2.6e-7.

`LBMNode` also refuses `inlet_face == outlet_face`, which silently dropped
the inlet pressure, and documents that the cells shared by perpendicular
faces carry the outlet pressure.  **Migration**: wrap an `LBMNode` with
`boundary="periodic"`; a wrapper built with the default or with `"edge"` /
`"zero"` is now refused at construction.

### The halos at the edges of the global grid

`halo_exchange` fills a halo between two shards from the neighbour and a halo
at an edge of the global grid from its `boundary` mode.  Two defects in that
second fill date from the primitive's first release, v0.2.0, and were present
in v0.3.1; both are fixed, and `halo_exchange` with `boundary="periodic"`
(any mesh), or with `"edge"`/`"zero"` and halos one cell wide on mesh axes of
two or more devices, is bit-identical to before, forward and adjoint.  So is a
`ShardedStencilNode` around any node but `HeatNode`, whose sharded step
changed for the reason below.

* **A mesh axis of one device** (MADD-ANO-028).  The boundary fill was applied
  only when the axis had more than one device.  With one, the exchange hands
  the shard its own opposite edge, so both global halos were periodic under
  `"edge"` and `"zero"` alike: a sharded stencil node run on one device — or
  on a 2-D mesh whose axis over the sharded cells has one device — ran on a
  ring, with no error.  A 64-cell `HeatNode` with a linear initial ramp ended
  0.40 away from the unsharded rod after 50 steps, its two ends drifting
  towards each other; on two and four devices the same run was bit-identical.
  A one-device axis now takes the boundary fill on both sides.
* **`"edge"` wider than one cell** (MADD-ANO-029).  The default mode is
  documented as replicating the edge cell; at width 2 it put the shard's first
  two cells, `r0, r1`, in front of `r0`.  It now repeats `r0` across the halo
  (`numpy.pad`'s `mode="edge"`), in the exchange and in the wrapper's local
  padding of a halo axis `axis_map` leaves unsharded.  The only built-in node
  this reaches is `HeatNode` at `stencil_order=4`, whose sharded rod was
  4.7e-3 from the unsharded one (7% at the end cell).

**A sharded `HeatNode` is the unsharded node** (MADD-ANO-030).  That 4.7e-3
did not go away with the fill alone: `update` closes each rod end with a
cubic through the rod end and three cells, which no fill can express, and
`update_padded` used the halos as they arrived.  It also never read
`left_temperature` / `right_temperature`, so a sharded rod with its ends held
at a temperature ignored them — 0.87 from the unsharded rod with both ends at
0, at either order and on any number of devices, and two sharded rods coupled
end to end through those inputs did not couple at all.  This shipped from
v0.2.0.  Now the block holding a rod end (the wrapper's `shard_info` says
which) rebuilds the ghosts beyond it with `update`'s own closure, from the
inputs, and the sharded rod is `update` for the same boundary inputs on 1, 2,
4 and 8 devices and on a size-1 pencil axis: exactly at `stencil_order=2`,
within 2.4e-7 at 4 (float32).

Since the fill at the rod ends no longer reaches the answer, `HeatNode`
declares `halo_boundary() == "edge"`, and `ShardedStencilNode` **refuses
`boundary="zero"` and `boundary="periodic"` for a `HeatNode`** rather than
ignore them.  `"zero"` was the documented way to give a sharded rod cold ends;
it imposed 0 at the ghost centres, half a cell outside the rod.  The refusal
says what to do instead: hold an end at 0 with `left_temperature=0.0` /
`right_temperature=0.0`, exactly as unsharded (in a graph,
`gm.add_external_input(<node>, "left_temperature")`, which defaults to 0).  The
wrapper also refuses a `boundary` that is not one of the three modes, as a
string, at construction: until now a typo failed at the first trace, and on a
halo axis `axis_map` leaves unsharded it was silently filled with zeros.

**Migration**: re-run a sharded result taken on one device or a size-1 mesh
axis with a non-periodic `boundary`, and every sharded `HeatNode` result with
end temperatures or `stencil_order=4`; replace `boundary="zero"` on a
`HeatNode` with the two end-temperature inputs.  A custom stencil node wider
than one cell that was tuned to the old `r0, r1` fill will change.

### What the sharding audit of the frozen tree found

* **A recompile did not reach the sharded step** (MADD-ANO-032, since
  v0.2.0).  Both wrappers keep a compiled `shard_map` per input signature, and
  a node on the three-argument contract has the constants it read from
  `self.params` baked into it.  `compile()` asks every node to drop its caches
  but reached only the per-device statics, so once the graph had stepped, a
  write and a recompile — what `PUT /graph/params` does for such a node — left
  the sharded step on the old value (a diffusion with `D` moved from 0.1 to
  2.0 kept stepping at 0.1, 7.4e-2 from the unsharded node).  The stencil
  wrapper also fixed its halo layout at construction, so a `HeatNode` written
  from `stencil_order=2` to 4 (halo 1 to 2) was padded with one cell and
  stripped of two, 3.9e-2 from either order.  Both wrappers now drop their
  compiled steps with the statics, and the halo layout is rebuilt on every
  `compile()`.
* **`dt` was cast to float32** (MADD-ANO-033, since v0.2.0).  Under
  `jax_enable_x64` a float64 node stepped with `float32(dt)` when sharded and
  drifted 1.66e-6 relative from the unsharded node over 1000 steps; `dt` is
  passed as given, and the two are bit-identical.
* **A state its node no longer builds is refused.**  The stencil path never
  sees the global grid, so a `HeatNode` whose `n_cells` was written under a
  running rod stepped the old cells with the new spacing; the wrapper compares
  the state with what `initial_state()` builds (abstractly, once per
  `compile()`) and names the field.  `HeatNode.update_padded` checks `n_cells`
  against the block it is given as well.
* **A wrapper around a wrapper** now takes the halo fill the inner one was
  built with: only the outer fill runs, so an outer default `"edge"` around an
  inner `"periodic"` LBM wrapper ran another model (`f` 0.020 off).
* **`ShardedPointwiseNode` refuses a node with nothing to shard** — no state
  field having the shard axis, such as a single spring's 0-d state — naming
  the fields; it used to replicate everything and say nothing.
* `ShardedUnstructuredNode` refuses a misspelt mesh axis in
  `domain_integral_axes` (it returned the stacked partials instead of the
  scalar), and it and `HybridNode` forward `update_evaluations()`, which they
  dropped, so a coupling group containing a wrapped sub-stepping node reads
  its float floor correctly.

* **Open: a global-order state on a balanced partition** (MADD-ANO-035,
  since v0.3.0).  When every shard of a `ShardedUnstructuredNode` is full
  and the cells are not in global order, a state written in global cell
  order with `set_node_state` has the partition-layout shape and is read as
  partition layout: each cell steps from another cell's value (an
  interleaved 8-cell, 2-device ring reads `[1..8]` as
  `[1, 5, 2, 6, 3, 7, 4, 8]`).  The per-cell boundary input in the same
  situation is refused; the state cannot be refused the same way without
  refusing the documented partition-layout spelling, which has the same
  shape, so it waits for an explicit layout on the state write in 0.5.0.
  Meanwhile convert a global-order state with `partition_value` before
  writing it, or renumber the cells so each device owns a contiguous
  ascending block (`np.argsort(partition_assignment, kind="stable")`).

**Migration**: re-run sharded results whose structural parameters were
written after the wrapper was built, and sharded float64 stencil runs under
x64; run a single scalar node unwrapped.

### What the second sharding audit found

* **`ShardedUnstructuredNode` stepped a Cartesian stencil node wrong**
  (MADD-ANO-037, since v0.3.0).  It hands `update_padded` the partition
  layout — `[own cells | ghost cells]` — and called the node's `halo_width()`
  "ignored", so a node that reads `[halo | interior | halo]` read the wrong
  cells: a `HeatNode` rod 43 K off on two devices (66 K at v0.3.x), every
  cell, and an `LBMNode` channel 100% off in velocity, with no error.  This
  cycle's stencil-wrapper divisibility refusal and the topology guide had
  recommended exactly that for an indivisible grid.  The wrapper now refuses a
  node with a non-empty `halo_width()`, the refusal and the guide send a
  stencil node to `ShardedStencilNode` (and only a pointwise node to the
  unstructured wrapper), and the guide describes the two `update_padded`
  layouts instead of calling them identical.
* **Cells past the layout's count vanished, and integrals counted padding**
  (MADD-ANO-039, since v0.3.0).  A 10-cell node on an 8-cell layout lost two
  cells (integral 32.4 for 49.5); it is now refused at construction, and
  `partition_value` refuses a value of the wrong length.  On an uneven
  partition `shard_info` said only `(offset, n_local_max)`, so a node summing
  its block summed the padding (36 for 35 on 7 cells over 2 devices); it now
  also carries `"n_local"`, the shard's own count, to mask it with.
  `"n_local"` is `shard_info`'s only string key (the others are `int` axes
  mapped to `(offset, extent)`), so a node that iterates
  `shard_info.items()` and unpacks each value as a tuple now fails under
  `ShardedUnstructuredNode`.  **Iterate over the integer keys only**
  (`for axis in (k for k in shard_info if isinstance(k, int))`), or index
  the axis you shard; the `update_padded` docstring and the sharding
  topology guide say the same.
* **A domain integral in the state was placed like a grid field**
  (MADD-ANO-040, since v0.2.1).  On the unstructured path an integral-emitting
  node failed on a graph's second step, or in `initial_state`; on the stencil
  path a vector integral (a 3-component drag on two devices) failed in
  `device_put`, and a per-shard integral fed back by a graph was refused as a
  grid changed by a parameter.  Both wrappers now place it as the step
  returns it — replicated once reduced, stacked along its unreduced mesh
  axes otherwise — and such a node runs in a `GraphManager` like the
  unsharded one.
* **A sharded static's global halos were edge-filled under
  `boundary="periodic"`** (MADD-ANO-038, since v0.2.1), on the reasoning that
  statics do not evolve.  The fill is about which cells lie beyond the edge,
  and on a periodic grid those are the opposite edge's: a periodic node
  reading a static in its halo ran 1.9e-2 off on 0.66.  Periodic now; `"edge"`
  and `"zero"` keep the edge fill.  No built-in node reads a sharded static in
  its halo, and a 130-configuration equivalence sweep of `HeatNode` and
  `LBMNode` is identical before and after.
* **Two ignored inputs are refused**: `ShardedStencilNode(axis_map={})`, which
  sharded nothing (MADD-ANO-042, since v0.2.0), and a `halo_exchange`
  `boundary` dict key naming no exchanged axis, which left that axis on
  `"edge"` (MADD-ANO-041, since v0.2.0).

**Migration**: shard a stencil node with `ShardedStencilNode`, sizing each
sharded axis to its device count; build an unstructured layout from the
node's own cells; mask a domain integral on an uneven partition with
`jnp.arange(n_local_max) < shard_info["n_local"]`; re-run periodic sharded
results whose node reads a static in its halo.

### What the third sharding audit found

* **`shard_info`'s block size was read off a domain integral**
  (MADD-ANO-056, since v0.2.1).  `ShardedStencilNode` took the local extent
  it hands `update_padded` from the first local state field with the axis,
  in sorted key order, integrals included.  Once the MADD-ANO-040 fix handed
  an integral to each shard whole, a `HeatNode` carrying its energy as a
  2-vector (or a per-shard value) named to sort before `temperature` was told
  its 4-cell blocks held 2 cells, and never closed its right rod end: 70.9 K
  off after 40 steps, with no error.  At the v0.2.1 to v0.3.1 tags a vector
  integral whose length the device count divides did the same to any node
  reading `shard_info`.  The extent is now fixed before tracing, from a grid
  field's global extent over the devices on its mesh axis; a node that reads
  `shard_info` whose grid fields disagree about an axis's extent is refused.
  An integral that `state_fields()` also lists (the default lists every
  `initial_state` key) is now summed under both wrappers; it failed with an
  exception.
* **A nested `ShardedStencilNode` dropped the node's parameters** (found and
  fixed in this cycle): the inner wrapper answered `accepts_params` for
  `update` only and its `update_padded` took no `params`, so `gm.params` had
  no entry for the node and a `run_scan(params=)` write of
  `thermal_diffusivity` moved nothing.  It now answers for `update_padded`
  and forwards `params`.
* **A mis-shaped `heat_source` or `body_force` was applied on every shard**
  (MADD-ANO-057, since v0.2.0).  A replicated input cannot be told from a
  local one, so 4 source values on a 16-cell rod split four ways were
  repeated along the rod, and a `(4, 8, 2)` force on a `(16, 8)` grid was
  zero-padded onto every slab; the unsharded node refuses both.  The wrapper
  now refuses a replicated input the node declares per cell unless it
  broadcasts to the declared shape, and the two nodes' `update_padded` take
  only the forms their `update` takes.
* **The LBM constructors accepted values that ran another model**
  (MADD-ANO-058, since v0.1.0): `LBMNode(viscosity=inf)` (no collision at
  all), a non-finite `tau` or `tau_tracer` on `LBMPipeNode`, and an
  `LBMPipeNode` propeller outside the grid or covering no cell, which ran
  with no propeller.  All are refused.  The default `propeller_x=10` is
  outside every pipe of 10 planes or fewer, so such a pipe now needs an
  explicit `propeller_x`.  A disc wider than the pipe, `propeller_radius`
  above the documented 0 to 1, pushed on wall cells (64 of 140 disc cells at
  1.5 on a 12x12 cross-section, the mean `u_x` 1.9% off); the round-3
  confirmation audit found it accepted, and it is refused too.
* Two sharding refusals that no test pinned (a pointwise wrapper's mesh
  without a `"devices"` axis; a halo that drops a sharded axis after
  construction) now have tests that fail without them.

**Migration**: re-run sharded results whose node carries a domain integral
in its state; pass `heat_source` as a scalar or one value per cell and
`body_force` as `(D,)` or the whole grid; give an `LBMPipeNode` of 10 planes
or fewer an explicit `propeller_x`.

### What the differential sharding harness found

The differential tests of the sharding wrappers
(`tests/cloud/multigpu/test_differential_sharding*.py`; "Differential
tests" in the testing standards) build each generated graph twice, sharded
and unsharded, and require the same answer on every graph surface, or a
loud refusal before any state.  They found five disagreements, all fixed.

* **A slab-length per-cell input ran on every shard** (MADD-ANO-064, since
  v0.3.0).  `ShardedUnstructuredNode` replicated any input that was neither
  in partition layout nor in global cell order, and `update_padded` cannot
  tell a replicated array from its own slab.  On a 4-cell ring over 4
  devices a `source` of 3 values gave every cell its first value, where the
  unsharded node refuses 3 values for 4 cells.  The wrapper now refuses an
  input the node declares per cell unless it is in partition layout or
  broadcasts to the declared shape, naming that global shape, as
  `ShardedStencilNode` has since MADD-ANO-057.
* **`gather_global` reshaped a listed domain integral as cells**
  (MADD-ANO-065, since v0.3.0).  It went by `state_fields()`, which by
  default lists every `initial_state` key: a scalar integral raised, and a
  per-shard one on a layout of one cell per shard came back reordered,
  silently.  It now recognises an integral first, as the step does.
* **A nested stencil wrapper stacked a per-shard integral twice**
  (MADD-ANO-066, never released): `(2, 2)` on two devices where a step
  returns `(2,)`, so `step()` changed the state's shape and `run_scan`
  refused the carry.  The outer wrapper now places the value the node
  itself builds.
* **A domain integral was summed over a mesh axis the `axis_map` leaves
  unused** (MADD-ANO-067, since v0.2.1).  Every device along such an axis
  holds the same block, so the releases counted each block once per device
  there: twice the total on a `(2, 2)` mesh sharded over one axis, silently.
  The development tree's `psum` was refused by JAX at the first step instead,
  and a per-shard integral was stacked over the axis with its partials
  repeated.  Such an axis is now neither summed nor stacked over, whatever
  `domain_integral_axes()` names; `ShardedUnstructuredNode` already treated
  every mesh axis but its own that way.  This was made to work rather than
  refused because the reduction is well defined: the sum over the axes that
  split the grid is the whole domain's.
* **Reverse mode through a scan over a partition with no ghost cells
  segfaulted jaxlib 0.11.2** (never released; 0.10.2 and 0.11.0 compiled it).
  `exchange_unstructured` joined the local block to a zero-size ghost tail,
  under both transports.  It now returns the block itself.

Widening the generated meshes to an unused axis surfaced one more
disagreement, which is not MADDENING's: XLA miscompiles a step inside a loop
when a sharded static is replicated over a mesh axis and the node reads it
in its halo beside a window at its `shard_info` offset (MADD-ANO-068,
partially resolved; see [Known anomalies](#known-anomalies)).
`GraphManager` refuses every entry point that would loop over such a step,
and the harness requires that refusal where it predicts it, on every
surface but `step` and `run`.

**Migration**: re-run a sharded domain integral computed on a mesh with an
axis the `axis_map` leaves unused, and drop the extra leading axis when
reading a per-shard integral from one.  Pass a per-cell input to
`ShardedUnstructuredNode` in partition layout (`partition_value`) or as a
value that broadcasts to the declared shape.

## Performance

| What | Before | After |
|---|---|---|
| Every run's compile count (MIME AR4, RTX A2000) | three compiles, 0.92 + 0.85 + 0.83 s | one |
| MIME AR4 "steady-state" step time (had absorbed two compiles) | 9.5–33 ms/step | 1.7 ms/step |
| `_default_external_inputs` allocation, GPU graph with external inputs | ~1.5 ms/step | shared, allocated once per compile |
| Coupling iterations at typical convergence | 80–90 % dead (`fori`, fixed cap) | early exit (`ift` while_loop) |
| FMU `get` of 10⁶ elements over loopback | 2.3 s (JSON) | 27 ms (binary) |
| One FMI call through the C wrapper, Linux loopback (80-step FMPy run, 245 calls) | ≥40 ms (Nagle + delayed ACK; 10.1 s) | 0.08–0.33 ms (`TCP_NODELAY`; 0.02–0.08 s) |
| IQN-ILS on the ρ = 0.98 contraction | 5 iterations | 2 iterations |

The compile-count fix is the one to understand.  Leaves seeded as
`jnp.array(0.0)` — nodes' initial states and MADDENING's own coupling
residual in `_meta` — are weak-typed; after one step they come back strongly
typed, so the jitted step retraced on the second step, and again on the
third for leaves that only change later.  `compile()` and `set_node_state()`
now normalise weak types (`_strong_typed`) and the `_meta` residual is
seeded as float32.  Results are bit-identical: only the trace signature
changed.

### Profiling and caching

The profiler (`maddening.core.simulation.profiler`) was rewritten.  Coupling
overhead is now *measured* — the graph is recompiled with every group capped
at one iteration and timed, and the difference is the cost of the extra
iterations, reported per iteration — instead of inferred from isolated node
timings.  Per-group iteration statistics over the run (mean / min / max
against `max_iterations`, fraction of steps at the cap, fraction converged)
replace a last-step count that assumed `max_iterations`.  New numbers:
`dispatch_floor_ms` (a jitted identity on the state pytree), median and p95
step time, and the device name.  With `trace=True` it records a short
`jax.profiler` trace and attributes device kernel time to the graph's
`jax.named_scope` labels (`node:<name>`, `coupling:residual`,
`coupling:accelerate`, `coupling:interface_override`, `edge:mapping`),
reporting the device-busy fraction and kernels per step that distinguish a
launch-bound step from a compute-bound one.  Recommendations use the new
numbers.

`benchmarks/bench_coupling.py` is the matching benchmark: coupling group
against staggered baseline, iterations used against cap, measured
per-iteration cost, optional trace attribution and the PERF-1 acceptance
check, for the MIME AR4 experiment graph (`--graph mime-ar4 --experiment
DIR`), a two-spring pair and a heat chain.

`maddening.core.simulation.compile_cache` (PERF-2) is a persistent
compilation cache: `enable(cache_dir)` points JAX's persistent cache at a
directory with thresholds that cache sub-second compiles, `compile()`
honours `MADDENING_COMPILATION_CACHE_DIR`, and `warm_cache(gm_factory,
n_steps=, scan_steps=)` compiles a graph's step and scan ahead of a run.  A
cross-process cache hit is tested — by the cache directory gaining no new
entries, rather than by a wall-clock ratio.  Developer guide:
{doc}`/developer_guide/profiling`.

### A performance regression gate that does not read the clock

MADDENING had no performance regression gate.  Compile time is what a user
feels first — the pitch is that the whole graph compiles to one jitted step —
so a silent 3× compile-time regression would have shipped unnoticed.  The
obvious gate is wall-clock, and this release learned what that costs:
`test_step_cost_does_not_scale_with_max_iterations` compared per-step cost at
two coupling iteration caps with a 3× bound, on quantities around 1e-4 s, and
failed CI at 3.27× on a pull request that changed 122 documentation files and
no code at all.  The idle-box ratio is 0.97–1.31.

So the profiler now reports integers that reproduce exactly on any machine
under any load.  `CompileCounts` — returned by `compile_counts(gm)` and
carried on every `ProfileReport` — gives the **retrace count** (Python traces
of the compiled step, one per XLA compilation), the **jaxpr primitive count**
for the step, and the **HLO op count** of the lowered StableHLO module.  The
HLO count is deliberately taken before XLA optimises: the post-fusion number
is the backend's decision and differs by platform, which would make a
committed baseline machine-dependent.  Wall-clock stays in the report, worth
trending, and gates nothing.

`scripts/compile_counts.py --check` is the gate, run from
`tests/core/test_compile_counts.py` and therefore on every CI run.  It
measures six workloads — a single node, a coupled pair, a multi-rate graph,
a heat chain with a `run_scan` program, a `ShardedStencilNode` over a
four-device mesh, and two coupling groups with `diagnostics=True` — against
`benchmarks/compile_counts_baseline.json`.
Retrace counts are compared **exactly**, since they are counted in Python and
do not depend on the JAX version; op counts are compared against a band,
±max(2 ops, 2%) on the baseline's own JAX version and ±max(10 ops, 25%) on
any other, because a gate that fails on every JAX upgrade is one people learn
to bypass.  The widths rest on a measurement: the baseline is recorded on
JAX 0.10.2 (the lower CI lane), and re-measured on JAX 0.11.0 every one of
its 21 counts — retraces, jaxpr primitives and HLO ops on all six workloads,
the heat chain's scan program included — came out identical.  That held when
the baseline was committed (bd5a1e9) and then stopped holding without the gate
saying so.  The MADD-ANO-033 fix (c02ee4b: the sharded stencil step takes `dt`
at the precision it is given instead of casting it to float32) added two jaxpr
primitives and two HLO ops to `sharded_heat` on both JAX versions (138 / 118
became 140 / 120; restoring the cast in a scratch copy restores 138 / 118).  A
drift of 2 is exactly the same-version band, so the gate stayed green while
the baseline went stale.  The baseline was regenerated under JAX 0.10.2 for
the release, and on 2026-09-25 (release/0.4.0 at db6ef8d) all 21 counts are
identical on 0.10.2 and 0.11.0 again.  A failure prints both
tables and names `python scripts/compile_counts.py` as the fix.

Contributors will meet this gate.  If it fires on a retrace count, that is an
extra XLA compile on every run and wants a cause, not a regenerated baseline;
if it fires on an op count and the change is intended, regenerate and say in
the commit message why the counts moved.  Current code retraces exactly once
on all six workloads and builds its scan program once.

Mutation-testing the gate found a flaw in the gate.  The first version
stepped each workload once before reading the counts, and one step cannot see
the retrace bugs this project has actually had — the weak-typed-seed bug in
`tests/core/test_step_retrace.py` traced again on step 2 and a third time on
step 3.  A node written to retrace on its second step scored a clean 1.
`compile_counts` now warms up for four steps by default.  Developer guide:
{doc}`/developer_guide/profiling`.

## Persistence and the graph API

`GraphManager.reset_state()` resets every node to `initial_state()` and
zeroes the `_meta` counters, coupling diagnostics and IQN warm-start caches
with the same weak-type normalisation `compile()` applies, so a reset never
retraces the jitted step.  The profiler, the REST reset and the example
servers use it instead of assigning `initial_state()` into `_state`.
`GraphManager.get_node(name)` is a new public accessor.

The USD stage now carries a node's own name (`maddening:nodeName`) and an
edge's declared units (`maddening:sourceUnits` / `maddening:targetUnits`).
Node names may legally contain characters a USD prim name may not (`-`, `.`,
spaces, parentheses) and may start with a digit, so the prim name is a
mangled, de-duplicated identifier and the node name is stored beside it.  A
stage written before this reads back as before.  This closed three
round-trip defects the new property tests found: `save_graph_to_usd` +
`load_graph_from_usd` renamed such a node and left every edge referring to
it dangling, so the reloaded graph would not compile; it collapsed two
distinct nodes whose names mangled alike (`"a-b"` and `"a.b"`) into one; it
raised from `pxr` on a name starting with a digit; and it dropped
`source_units` / `target_units` entirely.

### A saved graph rebuilds every constructor argument

A saved graph rebuilds each node as `cls(name=, timestep=, **params)`, so a
constructor argument a node keeps anywhere but `params` is dropped by every
reload while the reloaded graph still runs.  **`LBMNode`'s `wall_mask` was
such an argument from v0.1.0** (MADD-ANO-034): `from_dict` on a `to_dict()`
config and `load_graph_from_usd` on a saved stage both rebuilt the node with
no walls, and on a 12 x 8 D2Q9 channel driven by a body force the reloaded
flow reached 1.29 times the original's peak speed after 30 steps, and the
two velocity fields differed by more than that peak.
The mask is now recorded in `params` as nested lists of bool, the
JSON-faithful spelling `HeatNode` already uses for `grid_points`; the
constructor accepts any array-like, so `POST /graph/nodes` can build a
walled node too.  A node without a mask writes no key, so a wall-free
node's config is unchanged.  The mask is structural — not a leaf of the
parameter pytree — and `PUT /graph/params` refuses a new mask for a running
node, because the step reads the mask the constructor built.

A sweep of every built-in node's constructor arguments through the node,
JSON-config and USD reloads, compared field by field and by stepping the
reloaded graph, found one more loss: `AdaptiveNode(dtype=...)` was kept off
`params`, so a float32 `WaveletAdaptiveNode` reloaded as float64 under x64.
It is recorded by name now.  No release carried it.  Every other built-in
node reloads to the same trajectory, bit for bit, and
`tests/nodes/test_builtin_node_round_trip.py` holds each of them to that and
fails when a node is added without a case.

A sharded node is the one thing a config still does not rebuild, and it
now says so (MADD-ANO-036).  The wrappers' `to_dict` records the wrapped
node's type plus `sharded` and the wrapper's settings, and `from_dict` used
to ignore them and rebuild a single-device node without a word.  Restoring
the wrapper would need the device mesh (and, unstructured, the partition
layout), which a config does not carry and the loading machine need not
have; refusing would block the reload of a graph that computes the right
numbers.  So the node comes back unsharded with a `UserWarning` naming the
wrapper, the recorded settings and the call that wraps it again:
`maddening.surrogates.replace.replace_node(gm, name, ShardedStencilNode(gm.get_node(name), mesh, axis_map=..., boundary=...))`
before `compile()`.

**Migration**: re-run results from a reloaded graph that holds a walled
`LBMNode`.

### The run entry points are stateful, and now say so

`GraphManager.step`, `run`, `run_scan`, `run_scan_with_history`,
`run_adaptive` and `run_adaptive_scan` all end by writing their result back
into the graph: the graph is left at the state the run reached, and the next
call continues from there.  That has always been true and only `step` hinted
at it, so `run_scan` and the rest read as pure functions of their arguments.

Nothing about the behaviour changed in 0.4.0.  What changed is that all six
docstrings state it and that tests pin it, because getting it wrong is
silent.  A 0.4.0 measurement harness built its observations more than once
from a single `GraphManager` and so measured a different initial condition
each time; the figures were plausible and wrong (+3.256%, and -3.6% /
-10.0% / +87.6% / +36.4%), against -0.849% and -1.97% to -4.80% re-derived
with a fresh graph per measurement.

If you measure across repeated runs, construct the `GraphManager` inside the
measurement loop, or snapshot and restore explicitly with `save_state` /
`load_state`; `reset_state()` returns a graph to its nodes'
`initial_state()`.  `run_sweep` is the one run entry point that does *not*
advance the graph's state — it runs from the `initial_states` you pass and
writes nothing back — and its docstring now says so too, since an unmarked
non-mutator beside six unmarked mutators is just as easy to misread.

### Non-finite numbers are valid JSON now

Three serialisers handed a non-finite float to `json.dumps` at its default
`allow_nan=True`, which writes the bare tokens `NaN`, `Infinity` and
`-Infinity`: `GraphManager.to_dict`, the USD `maddening:paramsJson` family,
and the FMI JSON wire.  RFC 8259 has no literal for any of them.  Python
reads them back, so every round trip inside MADDENING was exact and nothing
in the suite noticed; every other reader — a C++ or Rust consumer of a saved
config, a `.usda` opened by another tool, a third-party sidecar client —
rejected the document outright.  It bit at the worst moment: the way a
non-finite value reaches a document at all is that a model diverged, and
serialising it is how you find out why.

**They are now written quoted** — `"NaN"`, `"Infinity"`, `"-Infinity"` — by
`maddening.serialization.json_codec`, which all three surfaces share.  The
quoted form rather than a tagged object such as `{"__float__": "NaN"}` for
three reasons: a pre-0.4.0 document and a 0.4.0 one differ by two characters
per value, so configs stay diffable and a human still sees what the number
was; every container keeps the shape it had, where a tagged object turns a
float inside an array of floats into a nested object; and the FMU's C
wrapper needs two lines rather than a JSON parser.  `json_codec.dumps` also
passes `allow_nan=False`, so a value its walk did not reach fails at the
call site instead of leaving an invalid document behind.

**Reading is lenient, writing is strict.**  A config, stage or peer written
before 0.4.0 carries the bare tokens, and still loads: `json.loads` parses
them into floats itself, and the decoder passes a float through.  Nothing
you have saved needs migrating, and nothing you save now needs a MADDENING
reader.

**Ambiguity is refused at write time.**  A blind decoder cannot tell
`float('nan')` from a string that happens to read `"NaN"`, and `to_dict` has
no schema for node params to decode against.  So the encoder raises on a
*string* leaf equal to one of the three tokens, naming its path — no
document MADDENING writes is ambiguous, which is what makes blind decoding
sound.  If you were storing one of those three strings as a parameter value,
spell it differently.

That refusal is a user-visible limitation on a wire format, so it is
registered as `MADD-ANO-010` rather than left in a docstring.  It applies to
a node name as much as to a parameter value, and to a header a client puts
on the FMI wire.  One reply `FmuTcpBridge` itself builds does carry free
text: the `hello` reply echoes the FMU's `model_name`.  So
`build_model_description` refuses a `model_name` spelled exactly as one of
the three tokens, and the bridge answers any reply it cannot encode with a
plain-ASCII error reply instead of dropping the connection.  A tagged object
such as `{"__nonfinite__": "Infinity"}` was considered and not used: a user
dict of that exact shape is ambiguous in precisely the same way, so the
choice is not between a refusal and no refusal but between a loud one at
write time and a silent mis-read at read time — and the tagged form would
additionally cost the FMU's C wrapper a nested-object parser where the
quoted token costs eight lines, in a file that has no JSON parser at all.
Both the refusal and the decode are exact string matches, so `"nan"`,
`"inf"` and `"+Infinity"` stay strings.

**The HTTP API's replies use the same encoding.**  `GET /graph` already
served `to_dict()`'s encoded tree, but every reply that carries state --
`GET /graph/state` (with `_meta`), `GET /graph/state/{node}`, `POST
/sim/step`, `POST /sim/run`, `POST /sim/reset`, `POST /checkpoint/load` --
handed Starlette bare floats, which its `JSONResponse` serialises with
`allow_nan=False`.  A diverged simulation made every one of them a 500, and
so did a graph with a `diagnostics=True` coupling group, whose spectral
`_meta` slots are seeded `NaN` at compile and at every reset until a solve
fills them: `POST /sim/reset` and `POST /checkpoint/load` failed in the
encoder *after* their change had been applied.  The state replies, `GET
/graph/params` and the `/ws/state` JSON frames (which wrote the bare tokens
a browser's `JSON.parse` rejects) now write the quoted tokens; a finite
reply is unchanged.  `PUT /graph/params` and `PUT /graph/state` also refuse
a value the leaf's dtype cannot hold (`1e39` into float32) *before* casting
it: the cast warned, a 500 wherever warnings are errors, and the refusal
after it called a finite value non-finite.

On the FMI side the descriptor half was already fixed: `modelDescription.xml`
emits `INF` / `-INF` / `NaN`, which are valid `xs:float`.  The wire half is
this change.  The shipped C wrapper reads both spellings — C99 `strtod`
parses `nan`, `inf` and `infinity` case-insensitively, and the quote was the
only thing it could not step over, so `parse_values` steps over it and
requires the closing quote so that `"1e5xyz"` cannot pass as `1e5`.  A
non-finite value is still never *sent* by the wrapper: `do_set` refuses one
before the request is built, as it always did.  Protocol 2's binary frames
carry raw float64 and never had the problem.

`compile()` also keeps calibrated params.  Every recompile used to overwrite
`gm.params` with the constructor snapshot, so adding an edge or external
input, replacing a node, a REST write on a legacy node, or the profiler's
one-iteration variant silently discarded a fit.  Live leaves whose node,
key, shape and dtype still exist are carried over (a dropped leaf warns),
and `gm.reset_params()` is the explicit way back.  A partial params pytree
through `gm.step` / `run` / `run_scan` is completed from the *live*
`gm.params` rather than falling back to the constructor constant; the raw
compiled step refuses an incomplete pytree with a message pointing at the
wrappers.

Two mapped edges on one field pair (two additive contributions) used to
share a single `params["mappings"]` slot, with the first silently using the
second's weights; each now gets its own slot via `EdgeSpec.ordinal`
(`"a.v->b.inp"`, `"a.v->b.inp#1"`).  `set_param_spec` overrides no longer
outlive `remove_node` / `remove_edge`, and an edge's `transform` (registered
name), `additive` flag and units survive the config round trip.

**A surrogate swap silently rewired its node's edges.**  `replace_node` saved
the edges touching the node, removed it and re-added them with five
positional arguments where `add_edge` takes nine, so every replacement reset
`additive` to `False`, both declared units to `None` and the interface
`mapping` to `None`.  Two additive contributions into one boundary input
measured 3.0 before a swap and 1.0 after it, with no error and no warning,
and a mapped edge lost its interface transfer whole.  The swap in the other
direction, `POST /surrogate/deactivate`, restored those nine attributes but
not the live mapping weights: `remove_node` drops
`gm.params["mappings"][edge.key]`, and the revert's `compile()` re-snapshots
that slot from the mapping object, so **weights a `sysid` fit had moved were
reverted to whatever the `MappingSpec` rebuilds** — the endpoint answered
`200` and the calibration was gone.  Both paths now carry every `EdgeSpec`
field, the live mapping weights and the edge's `ParamSpec` overrides across
the swap, and both check that the restored edge keys are the ones they saved
before putting weights back, since the key names the weights' slot.  The
re-add is built by `EdgeSpec.add_edge_kwargs()`, which derives the call from
the dataclass's own fields and refuses, naming the field, if `EdgeSpec`
grows one it does not know how to carry across.  **Results are affected if
you replaced a node with a surrogate — or reverted one — on a graph with
additive edges, declared units, interface mappings or fitted mapping
weights: the numbers changed at the swap, not at the fit.**

`ParamSpec` had its own edge cases: `constrain` with `transform="log"` and
`lo != 0` could land exactly *on* the open bound (`8.0 + exp(-15)` is `8.0`
in float32), so `check_params` rejected the fit's own output and
`unconstrain` returned `-inf` — the clamp is now relative to the bound, for
`logit` too.  `ParamSpec.check` / `check_params` reject `NaN` and `±inf`
(`NaN` used to pass every bounds comparison), `ParamSpec.from_dict` with a null `bounds` no longer crashes, and a bounded identity leaf keeps its dtype
through `constrain` (integer weights were being promoted to float32).
`ParamSpec.from_dict`, the reader of saved graphs and USD stages, no
longer coerces: it passed `trainable` through `bool()`, so `"false"`,
`"no"` and `"0"` read as trainable, and each bound through `float()`, so
`true` read as `1.0` and `"2"` as `2.0` past the refusals `ParamSpec`
itself applies to those values.  Both are refused; a saved graph carrying
one fails to load with the value named, and a USD stage warns and skips
that override, as it does for any override it cannot apply.

On the FMI side, a `log` or `logit` leaf's bound is strict but the FMI
attributes are inclusive, so the sidecar refused the very value the XML
advertised.  The model description now advertises the nearest representable
float32 inside the interval — the smallest normal for a zero bound, since
its subnormal neighbour is flushed to zero on XLA:CPU — so every advertised
bound is settable.  Inclusive bounds are unchanged.

Separately, FMU export of a real graph had no inputs and a wrong step size:
`build_model_description` looked for a `_external_input_specs` dict a
`GraphManager` never had, so external inputs were silently omitted, and read
a `_master_timestep` attribute that does not exist, so the default
experiment step was always 1e-3.  Inputs now come from the graph's
external-input list as `<node>.<field>` (description and unit from the
target's `boundary_input_spec`), and the step size is the graph's base
timestep.

### Inspecting a graph

A graph can now describe itself, through read-only methods marked
`experimental` (the user guide's *Inspecting a graph* has examples):
`gm.print_graph()` (or `format_graph()` for the string) lists nodes with
their types, rates, sub-cycling, state fields and sharding, the edges with
their kind, transform, mapping and flags, the coupling groups' settings,
external inputs and the execution order; `to_mermaid()` and `to_dot()`
give the same structure as diagram text with no new dependency; and
`state_summary()`, `params_table()`, `coupling_report()` and
`memory_estimate()` return tables of rows (each with a `print_*` form) --
per-field statistics, every parameter against its `ParamSpec` bounds,
`coupling_diagnostics()` with `ratio_usable=False`, `precision_limited`,
the iteration cap and a missing report flagged, and state bytes per node
and per device.  Output is deterministic plain text; `rich=True` is the
only way those methods use the optional `rich` renderer.

`print_graph_diagram()` draws the same structure as boxes and arrows in
the terminal: nodes as `name :: Type`, a frame per coupling group titled
`coupling group a+b`, `field→input` edge labels, and dotted flux edges and
external inputs.  It uses the optional `termaid` package, which the
`terminal` extra now installs, and with `rich` it colours a terminal in one
of termaid's themes (`theme="amber"`, `"phosphor"`, ...).  An unknown theme
is refused rather than drawn in the default, as termaid itself would, and
a file or pipe gets the same drawing as plain text.  `direction=` and
`use_ascii=` are termaid's own options.  It reads the graph through the
builder `to_mermaid()` uses, so the two cannot disagree on the structure;
the only differences are the two termaid's parser needs (a space before a
subgraph's title, `" :: "` for `<br/>`) and look-alikes for the few
characters it cannot carry in a label.  It is in the read-only guard's
matrix with the other methods, and is `experimental`.

"Read-only" is a tested property, not a description: a guard test
fingerprints every attribute of the graph around each call, on graphs
that are uncompiled, modified, coupled, multi-rate, sub-cycled, mapped,
sharded, stepped and left holding tracers by `jax.grad`, and records
JAX's compile events; eleven seeded faults (a state write, a flipped dirty
flag, a compile, a coupling report that ignores escaped tracers, ...) each
fail it.  A graph that is not ready is reported as it stands -- "not
compiled", "state holds tracers" -- never compiled or put back.

`maddening.show_versions()` and `python -m maddening info` print the
versions, JAX backend and devices, x64, allowlisted environment variables
(never a token or key) and installed extras; and `print()` of a
`FitResult` or `FIMReport` is now a short summary, with `repr()` unchanged.

### `gm.timestep` is the step the graph takes

`GraphManager.timestep` (and `base_timestep`) is now the simulated time one
`step()` advances, on every graph (MADD-ANO-096, since 0.1.0).  It was the
GCD of the nodes' own timesteps.  A `subcycling=True` coupling group,
though, solves once per macro step, its largest member timestep, and
`compile()` schedules every member there.  So two nodes at 0.01 and 0.02 in
one group read 0.01 while each step advanced 0.02.  `compile()` and
`timestep` now read one helper, so the step reported is the step the rate
dividers count in, and `validate()`'s multi-rate line reads the same
schedule.  Everything that clocks a run follows.  `RealtimeRunner` paced
and labelled such a run at half its real rate (50.5 s reported after
0.5 s at `time_scale=100`, against 101 s on the graph's own clocks), and so
did the `StateRelay` time on the REST server's WebSocket frames and the ZMQ
`NetworkRelay`'s `t`.  The runner and both relays now add `gm.timestep`
after every step, rather than multiplying a value read once.  A graph
edited between runs, such as a node of another timestep added over the
REST API, is then clocked by its new step.  `StateRelay.reset()` restarts
the clock, and `POST /sim/reset` calls it.

USD's `maddening:baseDt` and the FMU's default `stepSize` are now that
step too (MADD-ANO-097, since 0.1.0).  Both read a `_base_dt` attribute
that nothing set.  `USDWriter` wrote 0.01 for every graph, and
`save_graph_to_usd`, like the FMU default after MADD-ANO-078's fix, gave
the smallest node timestep.  That is wrong for timesteps that do not
divide each other (0.002 and 0.003 step by 0.001) and for a sub-cycled
graph.  `maddening:isMultirate` and the FMU's `multi_clock` intervals read
the same scheduled timesteps, so a sub-cycled member is on its group's
clock.  A uniform-rate graph's default step is unchanged.  `format_graph`
no longer needs its note that `gm.timestep` disagreed with the step.

## Cloud

**Resume-from-URL transport moved to `maddening.cloud.resume`.**
`download_and_load_state`, with its `file://` / `http(s)://` / fsspec fetch
helpers, now lives in the cloud package where deployment concerns such as
storage backends and credentials belong.  `maddening.core.simulation.checkpoint` keeps only the local save/load/manifest functions and imports
neither `maddening.cloud` nor `fsspec`.  The old import path still works as
a forwarding alias that emits a `DeprecationWarning`; **the alias is removed
in 1.0**.  Behaviour, signature and errors are unchanged.

The audit of that move found a real deployment bug: the manifest URL was
derived by appending `.manifest.json` to the whole URL, so a presigned
`https://…/snap.npz?X-Amz-Signature=…` fetched the `.npz` body as its own
manifest.  It is now appended to the URL *path*, keeping query string and
fragment — and because a presigned URL authorises exactly one object,
`download_and_load_state(..., manifest_url=)` and the entry point's
`RESUME_MANIFEST_URL` take the manifest's own URL.  HTTP(S) fetches have a
`timeout=` (default 60 s, `MADDENING_RESUME_TIMEOUT` in the entry point) and
raise `TimeoutError` instead of holding container start-up forever; the
timeout is forwarded to `s3fs` / `gcsfs` best-effort.  Downloads stream to
disk in 1 MiB chunks.  The default per-call temporary directory is removed
after the load, success or failure, while a caller-supplied `dest_dir` is
kept.  A `file://` URL is percent-decoded while a bare POSIX path is used
verbatim — `%` is an escape in a URL and an ordinary character in a
filename, and the source path and the downloaded copy's name now agree
about that.  An empty URL raises `ValueError("empty checkpoint URL")`, a
directory raises a clear `ValueError`, and Windows drive-letter paths are
documented as unsupported.  The entry point (`resume_from_env`, split out
of `main()`) logs URLs with the query string redacted, logs the manifest's
key fields on success, and says clearly when `RESUME_FROM_URL` is set but
the server's graph has no nodes.

**A failed resume is atomic, and is logged as a failure.**  `load_state`
applied every node state and only then validated the parameter leaves, so
resuming into a graph whose parameter *shapes* had changed — a redeploy
after a preemption, which is the case the module exists for — left the
checkpoint's states in a graph running fresh parameters.  Every name,
field, state shape and params leaf is now validated before anything is
written, and the write is wrapped in a rollback, so a restore that fails
leaves the graph exactly as it found it.  `resume_from_env` used to log
`"Failed to resume …; starting fresh"`, which is indistinguishable from a
run that was never asked to resume; it now logs `RESUME FAILED` and states
that the graph was not restored.  A log-scraping health check that matched
the old wording needs updating.

Container and example pins were stale and are corrected:
`docker/Dockerfile.cloud` pinned `jax[cuda12]>=0.4,<0.6`, and its Ubuntu
22.04 base carries Python 3.10, which neither MADDENING nor `jax>=0.10`
supports — the image moves to the CUDA 12.6 / Ubuntu 24.04 base with Python
3.12, and the SOUP table carries the `pyproject` range.  Cloud examples
04/06/07 installed under `python3.10` (uninstallable) and now use
`python3.12`, with PyGObject built by pip for that interpreter in the two
streaming examples.

## The shipped examples

Every example under `maddening.examples` now runs in CI, or is excluded
with a stated reason: the provisioning scripts in `examples/cloud/`, two
GUI-only viewers and one helper module.  `tests/test_examples_smoke.py`
classifies the directory and fails on an unclassified file.  Before this
release 8 of the 49 ran, all in the slow lane, and running the rest found
claims their own numbers contradicted.  The mutually-anchored spring demos
had no rest state -- both rest lengths `+L`, so the pair climbed for ever --
while printing "settled"; `acceleration_comparison` described a problem
where plain iteration struggles that it solved in three passes; and
`vessel_flow_server`'s parameter endpoints wrote `node.params` and marked
the graph dirty, and the recompile dropped the write (MADD-ANO-093, now
fixed); they write `gm.params` now, which takes effect on the next step
with no recompile.  The coupling demos now assert what they print.

The client/server examples run on one machine:
`python -m maddening.examples.servers.remote_viz_client --local` starts the
simulation server on an OS-chosen loopback port and stops it on exit, the
HTTP example servers take `--port 0`, and `cloud/server/04_server_test`,
`05_websocket_test` and `cloud/multijob/08_two_vm_test` take `--local`,
which runs their payload on loopback and never imports the cloud launcher.

Seven examples are new and one is extended, for features that had none,
and they hold to the same rule: each checks the claims it prints.
`advanced/profiling_demo` reads every part of a `profile_graph` report on
a coupled pair -- the measured coupling overhead and its cost per
iteration, the compile counts (identical on a freshly built graph), the
trace summary -- and writes its output to a temporary directory;
`profile_lbm_step` stays as the short Perfetto example.
`advanced/sysid_demo` takes a noisy spring through `fim`, `fit` and
`fit_lm`, with the scale degeneracy found, held and then removed by a
`ParamSpec` freeze; `advanced/checkpoint_resume_demo` resumes a coupled
run bitwise in a fresh graph, `_meta` warm starts included;
`advanced/sharding_demo` runs `ShardedStencilNode` on four emulated CPU
devices (emulation, not a speed test); `advanced/fmu_export_demo` builds,
packages and validates an FMU and drives the sidecar in-process;
`coupling/interface_mapping_demo` round-trips a mapped edge through
`to_dict` / `from_dict` and shows the MADD-ANO-063 refusal;
`servers/rest_params_demo` shows which `PUT /graph/params` writes are
honoured and which are refused, through the in-process test client; and
`coupling/convergence_diagnostics_demo` gains `print_coupling_report()`
and `strict_convergence`.  All of them run in CI on every push, at a
small size.

## Static typing: `py.typed` and a blocking pyright gate

**MADDENING now ships a `py.typed` marker ([PEP 561](https://peps.python.org/pep-0561/)),
so a downstream type checker reads MADDENING's annotations.**  Without the
marker a conforming checker must ignore every annotation in the package:
mypy, and pyright with `useLibraryCodeForTypes: false`, found none of four
deliberate mistakes in a throwaway consumer project, and find all four with
it ({doc}`/developer_guide/typing`, *PEP 561: shipping the marker*).
`tests/test_packaging_py_typed.py` builds a real wheel and asserts the marker
is in it.  Nothing to do but upgrade; a call site that was silently `Any`
before may now report a real type error.

What shipped:

* **Source annotations.**  Tier 1 (`__init__.py`, `core`, `nodes`, `fmi`,
  `cloud`, `sysid.py`, `serialization`, `testing`, `compliance`,
  `transport_auth.py`, `warnings.py`) was annotated to zero pyright errors,
  and every public signature in tier 2 (`api`, `surrogates`, `usd`, `viz`)
  was annotated.  The names behind each package's lazy `__getattr__` are
  re-exported under `if TYPE_CHECKING:`, so a checker can see them, and the
  renderer and trainer config dicts are PEP 589 `TypedDict`s.
* **A blocking `typecheck` CI job**, in two tiers defined by
  `typing_tiers.json`.  Tier 1 must stay at zero errors.  Tier 2, which sits
  on untyped or optional third-party libraries, has a committed per-package
  ceiling the count may not exceed: `api` 61, `surrogates` 55, `usd` 1,
  `viz` 45.  The job runs pyright once and gates both tiers off that stored
  run.
* **The tooling**: `pyrightconfig.json` (basic mode, `src/maddening` only,
  optional extras' imports downgraded to warnings), `pyright` pinned to
  `1.1.414` in the `ci` extra (a floor in `dev`), and
  `scripts/typing_baseline.py`, which summarises a run per rule, per file and
  per tier.

Measured with pyright 1.1.414 against `pip install -e ".[ci]"` and jax 0.10.2,
the environment `typing_tiers.json` records: **270 errors and 100 warnings
before the annotation work, 165 and 23 after it**.  Tier 1 went from 103 to
0 and tier 2 from 167 to 165; the later `TypedDict` pass took `surrogates`
from 58 to 55, so tier 2 now stands at 162, the sum of the ceilings above.
The earlier phase-1 baseline, taken before any annotation changed, is kept
in {doc}`/developer_guide/typing` for history.

The audit's finding here was that a broken pyright run looked like a result.
`scripts/typing_baseline.py` now exits **2** for an infrastructure failure,
distinct from 1 = errors found, when pyright is missing, exits with a code
other than 0/1 (3 = unparsable `pyrightconfig.json`, which used to be
flattened to "errors present"), produces no JSON, analyses zero files (a
missing `include` path gives a valid empty document and exit 0, which used
to pass `--fail-on-errors` silently), or fails to resolve the core imports
`jax` / `numpy` / `yaml` or more than `--max-missing-imports` (default 40).
That last one matters because **a wrong interpreter *lowers* the error
count — 395 to 230 in the audit — with nothing on stderr.**  pyright's
stderr is always forwarded, and a `--pythonpath` given after `--` is checked
to exist and to import numpy before pyright runs.  With `--tier`, a missing
tier file, an empty ceiling map, or a pyright release other than the one the
ceilings were measured with is also exit 2.  The CI job has three steps:
*Run pyright* fails the job on an infrastructure failure; *Tier 1 must be at
zero errors* and *Tier 2 may not grow past its committed ceiling* each fail
it on exit 1.

## Security

Three fixes from the first audit rounds, all on the FMU/REST attack
surface that an untrusted importer or client can reach; three more from the
params-io round of 2026-09-19 follow them; and the ZeroMQ transports, which
were the last unauthenticated listening sockets *that bound a wildcard
address*.  They are not the last unauthenticated listening sockets in the
package: `maddening.fmi.tcp_bridge` is one too, and 0.4.0 does not change it.
It is a much narrower exposure -- its measured constructor defaults are
`host="127.0.0.1"`, `port=0`, loopback on an ephemeral port -- and it is
documented as trusted-clients-only, but it is a listening socket with no
credential and it should not be read out of this sentence.

```{note}
**What "authenticated" covers in 0.4.0.**  Three listening surfaces, three
different mechanisms, one rule.  The rule is that **a loopback bind is
unauthenticated and any other bind demands a credential**; no surface
falls back to cleartext when its credential is missing.  (On the HTTP API,
"unauthenticated" is for a direct connection from a loopback address: a
forwarded request, or a peer that is not a loopback IP literal, presents
the token.)

| Surface | Port | Credential | Encrypted? |
|---|---|---|---|
| HTTP / WebSocket API | 8000 | `MADDENING_API_TOKEN` (bearer) | No -- **no TLS** |
| WebRTC signaling | 8443 | `MADDENING_STREAM_SECRET` (HMAC) | No |
| ZeroMQ state / commands / coordinator | 5555 / 5556 / 5580 | `MADDENING_TRANSPORT_TOKEN`, falling back to `MADDENING_API_TOKEN` (see below) | Yes, by CURVE |

**There are two secrets, and you should set both.**  An
earlier draft of these notes said "the API's token and the ZeroMQ keys come
from the *same* environment variable, deliberately: there is one secret to
manage, not two".  That is no longer how it works.  `TransportAuth` reads
`MADDENING_TRANSPORT_TOKEN` first and falls back to `MADDENING_API_TOKEN`
only when the former is unset -- a *blank* `MADDENING_TRANSPORT_TOKEN`
raises rather than falling through, because a variable that is set is the
operator's answer.  The reason to decouple them: the API has **no TLS**, so
the bearer token travels in cleartext in an `Authorization` header on every
request, and while the CURVE seed is that same string one sniffed HTTP
request yields both CURVE keypairs.  Setting `MADDENING_TRANSPORT_TOKEN`
costs one variable and removes that path entirely.  The single-variable
setup still works, and `TransportAuth.token_env` says which variable the
secret came from.  The cloud launch path always hands the container
`MADDENING_API_TOKEN` (the launching process's, or one `CloudSession`
generates) and `MADDENING_TRANSPORT_TOKEN` when one is set, so a published
ZeroMQ port there starts encrypted -- keyed off the HTTP credential unless
the transport token is set (`MADD-ANO-015`'s residual risk).  The
signaling socket keeps its own secret because it predates the rule and its
clients are browsers.
```

**The FMU bridge no longer unpickles importer bytes** (CRITICAL).
`FmuTcpBridge` `set_state` called `pickle.loads` on the base64 payload an
importer hands to `fmi3SetFMUState` — remote code execution for anyone able
to reach the bridge port.  The FMU-state blob is now an arrays-only `npz`
(`allow_pickle=False`) carrying the schema token, time, inputs, states and
params, and on `set_state` the token, key set and every shape are validated
before anything is written.  The pickle-based `FmuSidecar.handle` wire
protocol is for trusted in-process and Python clients only, and is now
documented as such.

**Zip bomb through an archive member without the `.npy` suffix.**  The
per-member size cap of the earlier arrays-only `npz` fix looked members up
by `name + ".npy"`, so a member named plainly (`_token`, say) escaped it and
was decompressed in full before the token check.  **Measured: 1 MiB on the
wire declaring 1 GiB drove the bridge process to +4.7 GB RSS; a 64 MiB frame
could declare some 60 GiB and OOM-kill it.**  Reachable over both the base64
(JSON) and raw (binary, protocol 2) `set_state`.  The bridge now checks the
archive *directory* before `np.load` touches anything: every member,
whatever its name, must be one the model expects (`_token`, `_time`, the
live state fields, parameters and declared inputs, all `.npy`) and must
declare no more than that array plus a header can hold, with the total
declared size capped too so duplicate names cannot multiply it.

**REST checkpoint endpoints are confined to a directory.**
`/checkpoint/save` and `/checkpoint/load` took an arbitrary server-side path
from an unauthenticated client — an arbitrary file write and a
file-existence oracle.  Paths are now relative to
`SimulationServer(checkpoint_root=)` (default `./checkpoints`) and must
resolve under it, and load errors no longer echo parser internals.

### The API authenticates when it is not bound to loopback

**A `127.0.0.1` bind is unauthenticated, exactly as before.**  Local
development is unchanged: no token, no header, nothing to configure.
**Any other bind requires `Authorization: Bearer <token>` on every route**
but `/healthz` and the static `/viz/*` pages.  That closes the hole in the
configuration that had it: every shipped container path binds `0.0.0.0`,
so the default cloud launch was serving `POST /cloud/launch` -- which
provisions paid GPU instances with the credentials on that host -- to
anyone who could route to the port.

The token is `MADDENING_API_TOKEN` if set, otherwise a
`secrets.token_urlsafe(32)` value generated at start-up and logged once.
**`MADDENING_API_TOKEN` set but blank raises**: it is what
`MADDENING_API_TOKEN=$UNSET_VARIABLE` produces, and reading it as
"authentication off" would rebuild the hole silently.  If nothing reads
your container's log, set the variable yourself or point
`MADDENING_API_TOKEN_FILE` at a path on a mounted volume, which the
generated token is written to with mode `0600`.  Tokens are compared with
`hmac.compare_digest`.

The server has to be told the bind address -- it cannot see the socket
uvicorn opens -- so pass `SimulationServer(bind_host=...)` or set
`MADDENING_HOST`.  Forgetting is not fatal: a request arriving from a
routable IP is challenged even when the server was told the bind is
loopback, so the failure mode is a 401 rather than an open API.  So is a
peer that is not a loopback IP literal, and any request carrying
`X-Forwarded-For` or `Forwarded`: only a direct connection from a loopback
address is served without the token (see "The REST round-8 audit").

Browsers cannot set a header on a WebSocket handshake, so `/ws/*` also
accepts the subprotocol pair
`["maddening.bearer.<base64url(token)>", "maddening.v1"]` and selects
`maddening.v1` in the response.  **No authenticated route accepts a token
in the query string**, so a credential never reaches an access log through
this API.  The bundled pages take the token from `/viz/app#token=<token>`
— a URL *fragment*, which a browser never sends, so even the page-load URL
reaches no access log.  (`?token=` is accepted too, for pasting through
tools that drop fragments; that form does reach the log.)  Either way the
page scrubs it from the address bar immediately, keeps it in
`sessionStorage`, and prompts for it on the first 401.

`/docs`, `/redoc` and `/openapi.json` are not served when the token is
enforced: Swagger UI fetches its own schema with no `Authorization`
header, so it cannot work behind a bearer token.  Reach them over an SSH
tunnel to a loopback-bound instance.

**`JobConfig.ports` no longer defaults to `[8000]`.**  That default was
what opened the API port in the cloud provider's firewall.  A launched
job's API is now reached over an SSH tunnel unless the job config asks for
the port.  (`launcher.py` also read `ports=job_config.ports or [8000]` on
the way to `sky.Resources`, which would have restored the port anyway;
both are changed.)

```{warning}
There is still **no TLS on the HTTP API**.  Its bearer token and every
snapshot it serves cross the network in cleartext, so a non-loopback bind
belongs on a private network, behind a TLS-terminating proxy, or -- best
-- not at all, with an SSH tunnel to a loopback bind instead.  `/healthz`
and the `/viz/*` pages answer without a credential by design.

This paragraph is about **port 8000 only**.  The ZeroMQ transports on
5555/5556/5580 are encrypted by CURVE whenever they are not on loopback
(see below), so a state snapshot leaving over ZMQ is not in cleartext.
```

**The WebRTC signaling server never authenticated anyone.**  Its handler
extracted no credential from the client: when the request carried no token
it re-validated the *server's own* token, generated at `start()` from the
same session id and secret, which is valid by construction -- so the
`close(1008, "Invalid token")` branch was unreachable and any client that
reached `0.0.0.0:8443` was relayed.  The handler now parses the token the
client presented (`Authorization: Bearer <token>` or `?token=`) and
validates *that* against the session, in constant time; absent, malformed,
duplicated, wrong, or issued for another session or secret are all closed
with 1008.  A client authenticates by computing
`generate_session_token(session_id, secret)`, so the container entry point
now reads the secret from `MADDENING_STREAM_SECRET`: without it the session
generates a random secret nobody holds and every connection is rejected,
which it says in the log.

**One unauthenticated request can no longer name unbounded work.**
`POST /sim/run?n_steps=` was unclamped while the neighbouring
`/sim/profile` capped at 1000; `POST /graph/nodes` let the caller choose an
array dimension (+433 MB of RSS from a single measured request); and
`TrainSurrogateRequest` had no bounds on any field.  The limits on a
request's own values now live on the request models -- `n_steps` ≤ 100000
(`0` stays a no-op), node integer parameters ≤ 10^7, at most 10^6 values in
one request's params, bounded training arguments -- so they appear in
`/openapi.json` and a violation is a 422 naming the field.  The caps on
what a request would build -- 2·10^7 state elements per node, 2 GiB of
build memory, 10^8 state elements in the whole graph -- are server
constants, and a node past one is a 400 naming the size.  All are far above what the bundled UI and the shipped
examples ask for.  The new 422 is for a constant that is a perfectly good
number and simply too large to be an array dimension, which used to be
accepted and allocated.

**`POST /graph/nodes` status codes did move, though, and a client that
branches on them needs to know how.**  Measured against a v0.3.1 tree,
five input classes go from **201 to 400**, because 0.4.0 refuses at the
endpoint what 0.3.x accepted and let fail later:

| Input | v0.3.1 | 0.4.0 |
|---|---|---|
| a name containing `/`, `#` or `->` | 201 | 400 "Node name … is invalid" |
| an empty name | 201 | 400, same message |
| a name spelling `NaN` / `Infinity` / `-Infinity` | 201 | 400 naming the token |
| a `HeatNode` timestep above its stencil's Fourier limit | 201 | 400 naming the Fourier number |
| a non-finite constructor constant | 201, then the response failed to serialise | 400 "value must be finite" naming the parameter |
| an array dimension over the new cap | 201 (and allocated) | 422 from the request model |

And one class goes from **409 to 400**: a *repeat* `POST` of any of the
five above.  On 0.3.1 the first request created the node, so the second
got 409 "already exists"; on 0.4.0 the first is refused, so the second
gets the same 400.  Nothing that already returned 400 or 404 changed.

Hardening from the audit rounds that is not itself a vulnerability but
belongs on the same surface: `PUT /graph/state` validates field set, dtype,
shape and finiteness before writing; a JSON boolean for a numeric param is a
400 (it used to drop the leaf at the next recompile); `POST /graph/nodes`
traces one update abstractly before adding the node and rejects a non-finite
constructor constant with a 400 naming the parameter, so a bad constant can
no longer wedge every later step or leave `GET /graph` answering 500 for the
life of the process; `POST /graph/edges` checks that the nodes and the
source field exist; and `add_node` refuses names containing `/`, `#` or
`->`.  On the core side, `load_state` refuses a state field of the wrong
shape and coerces dtype, a checkpoint without `_meta` keeps the freshly
compiled `_meta` instead of leaving a multirate graph to raise `KeyError`, a
wrong-shape params leaf is refused by `step(params=)` and by
`gm.params[...] =` (it used to broadcast the node's state permanently), and
an FMU-state member larger than the live leaf is refused before it is
decompressed.

**A web page is not an authorised caller** (CRITICAL, `MADD-ANO-076`, in
every release from v0.1.0; resolved).  A loopback bind trusts
whoever reaches the port, and that includes every page the user's browser
loads: a cross-origin `POST` with no body or a `text/plain` one needs no
preflight, so a page could reset, step, compile and checkpoint a server on
127.0.0.1 (at the tags, `/checkpoint/save?path=` wrote anywhere), and a
WebSocket, which the same-origin policy does not cover, could read
`/ws/state`.  A state-changing request or WebSocket handshake whose
`Origin` is not the server's own, or one named in
`SimulationServer(allowed_origins=)`, is now refused with 403; a request
with no `Origin` and a cross-origin `GET` are unaffected.  **DNS rebinding
is refused by the `Host`.**  A page whose own name the attacker re-points
at 127.0.0.1 sends a `Host` and an `Origin` that agree, so it passed the
`Origin` check.  On a loopback bind with no token it could drive every
route, `/cloud/launch` included, read the replies and open `/ws/state`.  A
loopback-bound server now answers only to the names this machine is
reached by: `localhost`, a `127.0.0.0/8` or `::1` literal (any port), and
the names in `SimulationServer(allowed_hosts=)`.  Any other `Host` is a 403
on every route and a refused WebSocket handshake.  The check applies where
no token is demanded and the peer is a TCP address; a non-loopback bind
demands the token, which such a page does not hold.  If you reach a
loopback-bound server through a proxy or an `/etc/hosts` alias, pass its
name in `allowed_hosts`.

### The params-io round: lifecycle, the second door, and the USD stage

**One silent TCP connection wedged the FMU bridge permanently, and every
connection leaked a thread** (HIGH).  `FmuTcpBridge` serves one FMU
instance, guarded by a lock; `_serve_conn` set `conn.settimeout(None)` and
then blocked reading *while holding it*.  A peer that connected and sent
nothing — a crashed importer, a dropped link, a port scan — parked the only
instance slot for ever, and every later client was answered *"bridge
already serves an FMU instance"*.  Nothing recovered it short of restarting
the sidecar: **`stop()` returned in 0.00 s with the worker still parked**,
and the same missing timeout on the refused path turned 300 idle
connections into 300 live threads.

Two things were wrong, and both are fixed.  The instance slot is now
claimed when a peer sends its first *complete frame*, not when it connects,
so silence claims nothing at all.  And every wait is finite, with a budget
named for what it is waiting on:

| Budget | Value | Why that value |
|---|---|---|
| Handshake | 10 s | The C wrapper sends `hello` immediately after `connect`, so this is already three orders of magnitude more than a healthy importer needs. |
| Idle, after the handshake | 300 s | An importer is idle between `doStep`s, and the master may be waiting on a slow partner or a human at a debugger prompt.  Deliberately generous. |
| One frame, once its length prefix has arrived | 120 s | Bounds a peer that dribbles — one byte every nine seconds renews a per-`recv` timeout for ever — and one that sends part of a frame and goes silent: each `recv` waits at most for what is left of the frame.  Still covers a full 64 MiB frame on a link of about 5 Mbit/s. |

A timeout is treated exactly as EOF.  Live connection threads are capped at
16, and `stop()` shuts every live connection down, so no worker outlives
the bridge that "stopped".  A bridge serves once: `start()` a second time,
or after `stop()`, raises `RuntimeError` (it used to start a second accept
loop, or a thread with nothing to accept on, and return as if it served).

**`set_state` bypassed every value check `set` applies** (HIGH).
`fmi3SetFMUState` takes a blob the importer may have built itself, so the
FMU-state archive is a second door into the state and parameter tree — and
it had no value checks.  `_decode_state` validated the schema token, the
member names and the shapes, then wrote through.  The dtype-fit check and
`ParamSpec.bounds` were both on the `set` path only, so the bridge that
refuses `set mass=-1.0` against declared bounds `(0.1, 10.0)` **accepted an
archive setting it, plus `position = inf` (a float64 `1e300` narrowed into
a float32 field) and `velocity = NaN`, answered `ok`, and one `step` later
the whole state was NaN.**  That defeats the documented guarantee that an
importer cannot silently tune a constant the graph declares invalid.

Both doors now share one function: finite, and representable in the live
array's dtype, over every restored state field, parameter and input, with
`check_bounds` over the restored parameter tree, all before the commit.

```{warning}
A snapshot of a *diverged* model — one whose state holds `inf` or `NaN` —
no longer restores.  The error names the field.  If you snapshot diverged
runs for debugging, keep them outside the FMU-state path.
```

**Opening an untrusted USD stage executed code** (HIGH, `MADD-ANO-075`,
in every release from v0.1.0 to v0.3.1).  A stage names the Python class of every node it carries, and
`load_graph_from_usd` resolved an unregistered one with
`importlib.import_module` — so merely *opening* a `.usda` ran the named
module's import-time code, before the `TypeError` that followed.  With
`sys.path[0]` being the running script's directory, a hostile stage shipped
next to a `.py` was plain code execution.  `GraphManager.from_dict` has
always required an explicit registry; USD did not, and nothing in the docs
called a stage untrusted.

`load_graph_from_usd` now takes `node_registry=` — keyed by qualified name
or by bare class name, so a registry written for `from_dict` works
unchanged — and instantiates only the built-ins, whatever
`register_node_class` registered, and whatever that argument passes in.
The import fallback is behind `allow_import=True`, off by default:

<!-- snippet: no-run, reason: fragment: stage is a USD stage and MyNode the reader's class -->
```python
gm = load_graph_from_usd(stage)                                  # built-ins only
gm = load_graph_from_usd(stage, node_registry={"MyNode": MyNode})  # plus yours
gm = load_graph_from_usd(stage, allow_import=True)               # a stage you trust
```

The refusal names the class and all three ways to allow it.  One latent bug
surfaced with it: `_ensure_builtins_registered` returned early whenever the
registry was non-empty, so a caller that registered its own class before
the first load kept the built-ins out of the registry for ever — the import
fallback had been covering that up.

Smaller items from the same round: `do_get` now has the `FRAME_MAX` check
on its request frame that `do_set` has (7M value references built a 77 MB
frame, which the bridge refuses to read, killing the instance instead of
answering an error); `FMUState`'s payload is an `npz` of plain arrays
instead of a pickle, and `FmuSidecar.handle` serves its pickled protocol
only when built with `SidecarConfig(allow_pickle_rpc=True)`; a non-finite
parameter exports as the `xs:float` literal `INF` / `NaN` rather than the
invalid `inf` / `nan`, and an infinite `ParamSpec` bound is left out
instead of written as `min="-inf"`; a USD stage raises on a param the JSON
encoder cannot represent rather than writing its `repr()` (the config path
already raised); and a stage now carries a zero-size param's shape
(`maddening:paramArrayShapesJson`) and an external input's dtype
(`maddening:dtype`), both of which it silently dropped.  A stage written
without either attribute loads exactly as it did before.

### The ZeroMQ transports: loopback by default, CURVE off-box

The last unauthenticated listening sockets in MADDENING that bound a
*wildcard* address were the ZeroMQ ones, and they had been open since
v0.1.0 (`MADD-ANO-015`).  (`maddening.fmi.tcp_bridge` is also
unauthenticated and is untouched by this release; it defaults to
`127.0.0.1` on an ephemeral port.  See the note at the top of this
section.)

`NetworkRelay` defaulted to `tcp://*:5555` and published the **full
simulation state dict**.  `CommandPublisher` defaulted to `tcp://*:5556`
and published `external_inputs`, which the paired `CommandReceiver` hands
straight to `GraphManager.step(external_inputs=...)` -- the interface's own
documented example is joint torques, so that socket is an actuation path.
Both receivers subscribe with `SUBSCRIBE ""`.  The multi-job `Coordinator`
bound `tcp://0.0.0.0:{port}` unconditionally.  None of them authenticated
anything.

**A signed envelope would not have been enough.**  A signature says who
sent a frame; it says nothing about who may read one, and the exposure on
the two PUB sockets is disclosure -- anyone who could reach the port
subscribed and read the state.  Only encryption closes that, so the fix is
ZMQ CURVE, whose crypto is the libsodium already inside the `pyzmq` wheel.
No new dependency.

**Two changes, in that order.**

*Nothing is published unless you ask.*  Every ZeroMQ socket now defaults to
a loopback bind, so the exposure is gone in the default configuration rather
than merely guarded.  Two qualifications, because "every socket" is not
true of the package as a whole.  `SelkiesSession` still defaults to
`bind_host="0.0.0.0"` -- deliberately, because it runs inside a container,
where a loopback bind is unreachable even with a published port -- but it
now *takes* `bind_host=`, so the socket can be narrowed where the
deployment allows it.  `examples/servers/vessel_flow_server`, which
defaulted `--host` to `0.0.0.0` through v0.3.1, now defaults it to
`127.0.0.1`, and a test enumerates the whole `examples/servers` directory
so the next one cannot ship bound to the world.  `cloud/_skypilot.py`'s `launch_vm` also stopped
hardcoding `-p 8000:8000 -p 8080:8080 -p 5555:5555 -p 5556:5556` and now
publishes exactly `JobConfig.ports`, which is empty by default -- that
function never consulted the field at all, which is why 0.4.0's earlier
change to its default did not reach the CloudSession path.  Port `8080`
is dropped outright: nothing in this package has ever served it (the
signaling server is on 8443, the API on 8000), and `cloud/session.py`
spent sixty seconds per launch probing a `/health` endpoint that could
only time out.

*If you do ask, it is encrypted.*  Binding any address another host can
reach turns on CURVE, and without a token the socket **refuses to open**
rather than falling back to cleartext.

**Key management is one environment variable per secret, and there are two
secrets.**  CURVE normally means a keypair per peer and a public-key
exchange; that ceremony is exactly the kind nobody performs, so it is not
asked for.  Both CURVE keypairs are derived deterministically from
`MADDENING_TRANSPORT_TOKEN` -- a Curve25519 secret key is 32 bytes of any
origin, so the scalar is a domain-separated BLAKE2b of the token and
`zmq.curve_public` supplies the public half.  Both ends compute both
keypairs, so neither has to learn anything from the other.  **There are no
key files.**  When `MADDENING_TRANSPORT_TOKEN` is unset the seed falls back
to `MADDENING_API_TOKEN`, the API's cleartext bearer credential; set both,
to different values, as the security table above explains.

<!-- snippet: no-run, reason: network: binds ZeroMQ sockets; the last two lines need a token and a remote host -->
```python
# Local development: unchanged, and needs nothing.
relay = NetworkRelay()                      # tcp://127.0.0.1:5555
receiver = NetworkReceiver()                # tcp://localhost:5555

# Remote, the recommended way: still no token, because both ends are
# loopback.  On the client:  ssh -L 5555:127.0.0.1:5555 user@host

# Remote, published port: same token on both machines, no key files.
#   export MADDENING_API_TOKEN=<secret>
relay = NetworkRelay("tcp://0.0.0.0:5555")  # CURVE on
receiver = NetworkReceiver("tcp://vm-ip:5555")
```

A ZAP allowlist is part of the fix, not a refinement of it.  Measured over
a real socket: CURVE *without* one completes a handshake with any client
keypair as long as the client knows the server's public key, and such a
client received every published frame.  The server now admits exactly the
one client key the token derives.

What this does not buy: every token holder can derive both keypairs, so an
authorised subscriber could also publish, and there is no forward secrecy
if the token later leaks.  That is the trust model of the shared bearer
token it reuses.  Where it is not enough, do not publish the port.

### The REST round-4 audit: sizes before construction, an honest stop

**A node too large to accept is refused before any of it is built**
(`MADD-ANO-086`).  The request bounds of 0.4.0 capped every integer a
request carries and the state a new node holds, but each size check ran on
the built node: `POST /graph/nodes` measured the state it had just
allocated, and `PUT /graph/params` checked a write by calling the node's
constructor with the new value.  A D3Q19 `LBMNode` of 140^3 cells took
1.1 GB before the state cap refused it.  A `WaveletAdaptiveNode` given
`n_levels=10000000` was never refused at all.  Its constructor built its
level labels as a Python list of `n_coarse * 2**n_levels` entries before
any check, and one such `PUT` grew the server to 57.7 GB and set off the
kernel's OOM killer.  `HeatNode`, `LBMNode`, `LBMPipeNode` and
`WaveletAdaptiveNode` now estimate, from their parameters alone, the state
they build and the memory their constructor takes.  The estimate is
private, `_allocation_estimate`, and for the state it is exact.
`POST /graph/nodes` and `PUT /graph/params` check it against
`MAX_NODE_STATE_ELEMENTS` and a new `MAX_NODE_BUILD_BYTES` (2 GiB) before
any constructor call.  `GraphManager.from_dict` and `load_graph_from_usd`
refuse (`ValueError`) a node that would not fit in the machine's memory.
The wavelet constructor refuses a basis it cannot build before it builds
any of it, and builds the labels with `np.repeat`.  Up to v0.3.1,
`POST /graph/nodes` had no bound at all, on an API that was
unauthenticated on the container's `0.0.0.0` bind (`MADD-ANO-051`).

**`POST /sim/stop` means stopped** (`MADD-ANO-088`).  The runner read its
stop flag only between frames, `stop()` waited two seconds, and the server
then dropped its handle and answered `"stopped"`, whatever the thread was
doing.  A frame longer than two seconds left an orphaned thread stepping
the graph.  That takes a large `steps_per_frame`, or one long step: the
recompile a parameter write asks for runs on the runner's thread.  With
`steps_per_frame=1e8`, `POST /sim/reset` answered position 1.0 and
`GET /graph/state` then served -196186.  The runner now checks its stop
and pause flags between every step.  `stop()` returns whether the thread
exited.  The server answers 503 and keeps the runner while the thread is
still in a step after ten seconds.  `/sim/step`, `/sim/run`,
`PUT /graph/state` and `/checkpoint/load` are a 409 while a runner thread
is alive, and a 503 while it is stopping.  `PUT /graph/params` still
reaches a running graph.  `PUT /sim/stride` is bounded at
`MAX_STEPS_PER_FRAME` and `MAX_RELAY_STRIDE` (100000, one `/sim/run`) and
refuses 0 with a 422.  A stride set before any runner exists is kept for
it; it used to be echoed and dropped (`MADD-ANO-090`).

**`PUT /graph/params` refuses what the step cannot run with**
(`MADD-ANO-087`).  `RigidBodyNode` `constraints` of `null`, `{"w": 0}`,
`{"y": {"finite": true}}` or `{"z": "high"}` each answered 200, after which
every `POST /sim/step` was a 500.  The route traces the node's hooks with
the request's values, on a copy, and refuses a value whose trace raises.
`POST /sim/step` and `/sim/run` answer a graph that cannot step with a 400
naming the error.  A structural value now keeps its parameter's numeric
type: `stencil_order: 4.0` is stored as `4` rather than as a float that
became a trainable leaf, and `4.5` is a 400.

**Checkpoint paths name a file inside the root** (`MADD-ANO-089`, never
released).  `path=""`, `"."` or `"sub/.."` resolved to the root itself,
which the confinement of `MADD-ANO-052` accepted.  `numpy.savez` then
wrote `<root>.npz` in the root's parent, and `/checkpoint/load` read it
back.  The check now runs on the file NumPy touches, and a NUL byte or an
unusable path is a 400 rather than a 500.  `DELETE /graph/edges` for an
edge the graph does not have is a 404 (`MADD-ANO-090`).

**A WebSocket stream ends with its client** (`MADD-ANO-091`, since
0.1.0).  `/ws/state`, `/ws/state/binary` and `/ws/render` noticed that a
client had left only when a send failed, and they send only on a new
snapshot.  So a client that left after `POST /sim/stop` kept its handler
alive, and SIGINT never shut the server down: measured at v0.1.0, v0.2.0
and v0.3.1 and on the development tree.  A client that left while the
simulation ran was logged as a "WebSocket error" with a traceback.  Each
stream now ends as soon as its client leaves, or the connection breaks,
and shutdown takes a fraction of a second.  An ordinary disconnect is
logged at INFO.

### The REST round-5 audit: one graph lock, bounded jobs and bodies, streams that follow the graph

**Concurrent requests use the graph one at a time** (`MADD-ANO-106`, since
0.1.0).  FastAPI runs the routes on a thread pool, and nothing serialised
their use of the `GraphManager`: two `POST /sim/step` in flight read the
same state and one result overwrote the other.  At every tag, 200
concurrent steps took about 60, every reply 200, and the streams' clock
counted all 200.  The same race let a state, parameter or checkpoint write
land inside a step.  One re-entrant lock now serialises every use of the
graph: each route that reads or writes it, the runner's every step, a
surrogate job's data sweep, and `POST /sim/run` slice by slice.  While a
run is in progress, writes are a 409 and reads are served between its
slices.  The lock is never held while waiting for the runner's thread, and
the runner waits for it in slices that watch its stop flag, so
`POST /sim/stop` is answered at once even while a long request holds the
graph.  A request waits at most 30 s for the lock, then answers 503.

**`POST /surrogate/train` leaves the live simulation alone, and is
bounded as a whole** (`MADD-ANO-107`, since 0.1.0; `MADD-ANO-092`
completed).  The job reset the live graph to its initial state, twice,
which its reply did not say; the data now come from `run_sweep` alone,
which writes nothing back, and a running runner keeps running.  Each of
the request bounds held one number, so one in-cap request took 2.2 GB (a
25 000-cell rod) or 5.9 GB and an out-of-memory error (16 layers of
8192).  The job's memory -- the sweep over the whole graph, its dataset
and training copies, the network and its optimiser -- is now estimated
before it starts and refused over `MAX_SURROGATE_TRAIN_BYTES` (2 GiB), and
`width**2 * depth` is at most 2^24.  One job runs at a time (409), the
last eight finished ones are kept, and a job stops at its next epoch when
the server shuts down; a job still in XLA at interpreter exit used to
abort the process.

**The streams follow the graph** (`MADD-ANO-108`, `109`, `112`, since
0.1.0).  After `POST /checkpoint/load` the streams served the state from
before the load, at the old step's time, until the next step.
`POST /checkpoint/save` now writes a manifest beside the checkpoint (its
SHA-256 and the streams' clock), and a load publishes the loaded state at
the clock it records, or at zero for a file without one
(`sim_time_from_checkpoint` says which).  `/ws/state/binary` kept the
layout of a node replaced over REST, so a client decoded phantom zeros, or
lost values, from a frame of the advertised length; the encoder is dropped
when the graph's layout changes, each stream sends a new schema before the
first frame of another layout, and `BinaryStateEncoder.encode` refuses a
field of another size.  `/ws/state` encoded every frame on the event loop,
once per client: frames are now encoded in a worker thread, once for every
client of the whole state, and at most 16 streams are open at once (1013).

**The runner says what it is doing, and keeps time after a pause**
(`MADD-ANO-110`, `111`, since 0.1.0).  A runner whose thread had died was
reported paused, resumed and running; the routes now read the thread,
`RealtimeRunner.error` says why it stopped, and `/sim/start` replaces it.
Adding or removing a node or an edge is a 409 while the runner runs, and
the `ExceptionGroup` that compiling a graph with an edge it cannot validate
raises is a 400 on every route that steps or compiles.  After a pause the
runner stepped flat out to catch up the time it had been paused; it paces
from the resume now, and moves its schedule rather than burst through a
stall longer than `max_catch_up` (0.25 s).

**What a request can make the server allocate is bounded as a whole**
(`MADD-ANO-086` completed).  A request body over `MAX_REQUEST_BODY_BYTES`
(32 MiB, the largest the element caps admit) is a 413 before it is parsed;
`PUT /graph/state` counts a field's values before converting them; the
whole graph's state is held to `MAX_GRAPH_STATE_ELEMENTS` (10^8); and nodes
are built one at a time.  **`POST /sim/run` stops when the server shuts
down**: it steps in slices, and SIGINT, SIGTERM or the lifespan shutdown
stop it within one, with a 503 naming the steps it took.  The signal is
chained ahead of uvicorn's handler, because uvicorn waits for in-flight
requests before it runs the lifespan shutdown.

**Smaller fixes, never released.**  `PUT /graph/params` asks the
constructor about a request's values together: HeatNode `stencil_order: 4`
with a smaller `thermal_diffusivity` in the same request was refused.  A
finite number a structural value cannot hold in float32 (`constraints:
{"z": 1e39}`) is a 400 on `PUT /graph/params` and `POST /graph/nodes`.  A
page served from `http://[::1]:8000` may change state on its own server
(the `Origin` check compared bracketed and unbracketed IPv6 literals).  A
malformed `Host` is a 403 (401 where a token is demanded), not a 500.  In
`run_pod.py`, `--keep-going` records a goal that raises as one failed
`goal raised` check and runs the rest, and `--summarise` exits 4 when no
file records a commit.

### The REST round-6 audit: the budget at the sweep, a profile that restores, routes that publish

**A surrogate job's memory is checked on the graph it sweeps**
(`MADD-ANO-131`, never released).  The estimate was taken in the request,
and the job's sweep took the graph lock again in its own thread; a node
added by a request queued in between was swept without being counted -- a
two-scalar graph's 0.5 MiB admitted a 2.15 GiB sweep.  The worker now
estimates again under the lock it sweeps with, and over the budget ends
the job `error`, naming it, before anything is swept.

**`POST /sim/profile` leaves the live simulation where it was**
(`MADD-ANO-132`, since 0.2.0).  The profiler resets the graph and steps it,
and the route left the run at the profiler's last step, while the streams
counted its steps.  The state and parameters are restored after the
profile and the relay is kept out of it; the next step is bit-identical to
an unprofiled graph's, a coupling group's warm start included.

**Every route that changes the state without a step publishes it**
(`MADD-ANO-133`, since 0.1.0).  After a reset the streams sent nothing,
after `PUT /graph/state` they served the old value, and after
`DELETE /graph/nodes` the removed node (the binary stream re-sending a
schema that put it back).  A reset and a surrogate swap now publish at
step 0, a state write and a node added or removed at the streams' clock,
on all three streams.  `StateRelay` adds a `run_adaptive` step's own `dt`
(`MADD-ANO-096`'s residual): twelve adaptive steps to t = 1 s of a 1/64 s
graph had been streamed as 0.1875 s.

**A JAX trace stops itself** (`MADD-ANO-134`, since 0.2.0).  The profiler
holds a trace's events in memory until it stops, and a trace left running
grew the server by kilobytes per step.  It now stops after 10 000 steps or
600 s, writing its files, and `GET /sim/profile/jax/status` says why.

**Smaller fixes, never released.**  A `/sim/run` whose step raised part-way
through said "nothing was stepped"; it reports `steps_run`.  A reset or a
surrogate swap that stopped the runner and then could not have the graph
said "Nothing was changed"; it says the runner stays stopped, with
`was_running`.  The runner routes answer within about one lock timeout of
their arrival (the k-th behind a long holder of the graph used to wait k),
and `PUT /sim/stride` at once.  A checkpoint save refused at its manifest
had already overwritten the file; it is written under a temporary name and
moved into place last.  Past the stream cap a real client got HTTP 403; the
handshake is accepted and closed with 1013.  `run_pod.py` records a commit
only when git names one (a repository with no commit recorded `"HEAD"`, and
`--summarise` took it as a commit), and `--summarise` reads an older
runner's file `INVALID` and exits 3 instead of stopping on its tables.

### The REST claims follow-up: what a route says it did, it did

The REST and RunPod claims inventory (`docs/validation/rest_runpod_claims.yaml`)
left five rows failing and five ambiguous; all ten are settled here.

**A checkpoint written in a wider dtype is refused where the graph's dtype
cannot hold a value** (`MADD-ANO-138`, since 0.1.0).  `load_state` cast every
leaf without a check: a float64 `1e39` loaded into a float32 field as `inf`,
and `-1e-50` as `-0.0`, silently -- a graph run under `jax_enable_x64` and
loaded without it was enough.  It now applies the FMU's rule
(`MADD-ANO-137`) to every state, params and `_meta` leaf before anything is
written: a finite value that would overflow, a non-zero one that would flush
to zero, an integer that would wrap is a `ValueError` naming the leaf; a
subnormal, and a value already `inf` or `NaN`, load.  A file that is not an
`.npz` of plain arrays is a `CheckpointFormatError` (a `ValueError`), and
`POST /checkpoint/load` answers it without NumPy's words about loading the
file unsafely.

**`PUT /sim/stride` keeps a value left out of the query** (`MADD-ANO-139`,
since 0.1.0).  Each value defaulted to 1, so changing the runner's steps per
frame reset the stream to every step.

**The generated token's file is the owner's alone** (`MADD-ANO-140`, never
released).  A file already at `MADDENING_API_TOKEN_FILE` -- a placeholder,
the last run's token -- kept its mode, and the new token landed in a
world-readable file.  The token now goes into a new `0600` file moved over
the path, so neither an old mode nor a reader who had the old file open can
reach it.

**A checkpoint save that says nothing was written wrote nothing**
(`MADD-ANO-141`, never released).  A save refused at its final move had
already replaced the manifest of that name.  Both files are now written
under temporary names and moved into place after both exist; a name that is
not a file is refused first; and no checkpoint 4xx detail names the
server's absolute paths.

**Smaller fixes, never released.**  A JAX trace's time budget has its own
timer, so an idle trace stops at 600 s too (it waited for the next step);
the stop route's 409 names no server path.  `run_pod.py --summarise` reads
a goal file it cannot parse as a JSON object (one `timeout` killed the
runner in the middle of writing) `INVALID` and exits 3, instead of ending
on a traceback with exit status 1, the code for "no goal JSON".  And four
documents now say what the code does: an over-size node is a 400 (only a
request model's own fields are 422s), a node no copy can be made of is
refused, a run_pod goal none of whose checks ran fails, and the seeded
wrapper faults are eight.

### The REST round-7 audit: a load is a write, a hybrid shares its params, a run says how far it got

A context-free audit of the REST and RunPod claims inventory
(`docs/validation/rest_runpod_claims.yaml`) on the frozen tree broke three
claims outright, and found smaller gaps in a dozen more.

**A checkpoint load asks what a parameter write asks** (`MADD-ANO-163`,
never released).  `POST /checkpoint/load` writes the archive's parameter
leaves into `gm.params`, and checked only that they fitted the graph's
shapes and dtypes.  A rod saved at `timestep=0.01` with
`thermal_diffusivity=0.4` loaded into a rod built at `timestep=0.05`
(Fourier number 2.0, which the constructor and `PUT /graph/params` both
refuse) answered 200; the graph then saved a config that did not reload,
and its temperatures reached 4e32 in 40 steps.  A damping below its
`ParamSpec` bound loaded the same way.  The load now asks of every leaf it
changes what `PUT` asks, by the same code (`_params_write_refusal`, after
the finiteness and bounds checks): the constructor with the graph's other
values, the state's layout, a value consumed at construction, a mapping's
points and the graph a save would reload -- against the graph as it was
before the load, which a refusal leaves untouched.  A property test writes
each of a dozen values with `PUT` on one server and loads a checkpoint
carrying it on another, and requires the same verdict.
The route asks finiteness and bounds per element, of a value that is
neither the leaf's now nor the node's own -- the rule of
`FmuSidecar.set_fmu_state` -- so a graph built outside its bounds (bounds
are metadata to a graph) reloads its own checkpoint, and a weight matrix
whose zeros a `log` spec refuses loads when a checkpoint moves only its
other entries.  `GraphManager.load_state` does not ask these checks, by
decision: Python code may hold a value outside a `ParamSpec`'s bounds on
purpose (a `gm.params` write is not checked), and a graph so written must
resume its own checkpoint into a freshly built graph -- asking them there
failed the checkpoint property that a generated graph resumes its run, and
would have failed a cloud resume.  It now refuses text and booleans for a
numeric leaf, which NumPy cast without a word (`"1.5"` to 1.5, `True` to
1.0) and no save writes.  The domain oracle
(`tests/property/test_differential_param_acceptance.py`) holds the route's
load to the write doors and records `load_state` as documented difference
D6; `check_params`, the FMU sidecar and the bridge still take a value only
the constructor refuses (D5, `MADD-ANO-047`'s residual).  `POST /graph/nodes` now applies the
`ParamSpec` bounds `PUT` applies, and refuses a boolean or text for a
numeric parameter, so a REST graph cannot hold such a value in the first
place; and `PUT` refuses a numeric string (it stored `"1.5"` as 1.5).

**A write to a `HybridNode` reaches its step** (`MADD-ANO-164`, since
0.1.0).  The hybrid passed its physics node's params to
`SimulationNode.__init__`, which keeps a dict of its own, while every step
reads the physics node's: `PUT /graph/params` answered 200, echoed the new
value and lost it, at every tag from v0.1.0 to v0.3.1 and on this tree
before and after a compile.  The hybrid now shares the dict, as the sharded
wrappers do, so a write through any surface -- REST, Python, a fit --
reaches the step, and the params route's checks (the constructor, the
saved graph) are asked of the physics node: a rod behind a hybrid that is
given a diffusivity past its Fourier limit is refused where it used to be
answered and lost.  No other wrapper in the tree copies its inner node's
params.

**A run that cannot have the graph says how many steps it took**
(`MADD-ANO-165`, never released).  `/sim/run` steps in slices, each taking
the graph lock; a slice that could not have it within
`_GRAPH_LOCK_TIMEOUT` answered "Nothing was changed; retry shortly" after
the slices before it had stepped the graph (240 steps in the reproducer),
so a client that followed the advice stepped it twice.  The run's final
read, and the compile of `n_steps=0`, waited with no timeout at all.
Every wait is now bounded, and one that runs out answers the
`{"status": "interrupted", "steps_run", "n_steps"}` 503 of the shutdown
path, the detail saying how many steps remain; only a run that took none
carries `Retry-After`.

**The runner routes do not queue behind requests waiting for the graph.**
They were synchronous routes on the worker pool every route shares, so
with every worker waiting for the graph a `/sim/stop` waited for a free
worker before its own deadline started: 5.6 s behind 240 queued reads at a
1 s timeout, where the guide promises about one timeout.  Start, stop,
pause and resume now run on a small pool of their own, reset on another,
each with its deadline counted from the request's arrival, and `PUT
/sim/stride`, which waits for nothing, answers on the event loop.

**An integral number for a float parameter is a float.**  The request
models bounded every JSON integer at 10^7, as a possible array dimension,
and `JSON.stringify(2e7)` is `20000000`: a stiffness of 2e7 N/m from a
browser -- the bundled `app.html` included -- was a 422 on both routes.
The bound now applies where the parameter is an integer, read from the
running node's value on `PUT` and from the constructor's default on
`POST` (a parameter with neither is bounded as written); the models bound
only the number of values.

**`/sim/profile` answers 400 for a graph that cannot step**
(`MADD-ANO-166`, since 0.2.0).  It caught only a `RuntimeError`, so the
`ExceptionGroup` an edge of mismatched shapes raises, or a node's own
`ValueError`, was a 500.

**Smaller fixes, never released.**  A stop or reset whose runner thread
does not stop in time says the runner was told to stop and stays stopped,
with `was_running` (it said "Nothing was changed").  A checkpoint name is
checked against the filesystem's name limit -- its manifest's name too --
before any directory is made, the temporary names no longer carry the
checkpoint's name (which put a valid 183-byte name over the limit), and a
save refused after its directories exist removes them.  An `allowed_hosts`
entry must be a host name (`"a b"`, `"*.example.com"`, `"user@host"` were
taken and could match no `Host`).  A token with whitespace around it
raises at construction: a presented token is read stripped, so it locked
out every client.  `APIAuth(environ=)` reads `MADDENING_API_TOKEN_FILE`
from that mapping.

**`run_pod.py`, before the hardware session.**  The default
`--synthetic grid` rounded `sqrt(n)`, so the "1e5-cell" exchange row
measured 316² = 99 856 cells, below the 1e5 cells a row needs to decide
the transport: the documented rule -- `all_to_all` faster at any point of
at least 1e5 cells keeps the default -- was decided on two of three
points, and that decision sets an API default.  A synthetic grid is now
the smallest square holding the requested count; each exchange row records
the size it was requested at (`requested_cells`, schema 7) and decides on
it, measured at no fewer cells; `--summarise` lists every row that does
not decide under **Rows that do not decide** and the recommendation says
"decided on N of M rows"; and the speedup is derived from the two medians
and must be finite (a NaN read as a tie, an infinite one decided).  A file
named for a goal that records another goal, or none, reads `INVALID`
(it was dropped, and the summary exited 0).  And the runner takes no
option abbreviations: `--dry` was accepted as `--dry-run`, but the CPU pin
reads the literal `--dry-run` before the options are parsed, so such a
"dry run" would have used the GPUs.

### The REST round-8 audit: a write back is a write, a forwarded peer is unknown, a load reads headers first

A context-free audit of the REST and RunPod claims inventory on the frozen
tree found two holes that needed no special configuration, and a dozen
smaller gaps.

**A params write back to a node's own value is asked the combined checks**
(`MADD-ANO-178`, never released).  `PUT /graph/params` counted a leaf as
changed when it differed from the node's own value, not from the live leaf
the graph runs with.  After a fit, or a `POST /checkpoint/load` (which
moves `gm.params` and leaves `node.params`), the two differ, and a leaf
written back to the node's own value was dropped from the write's checks
while still being written: the constructor asked with every key at once,
and the graph a save reloads, were asked with the leaf's old live value.
A HeatNode rod whose diffusivity a fit had moved took its own diffusivity
back together with a shorter length -- Fourier number 0.648, past the
explicit limit -- with a 200, ran to NaN, and saved a config its
constructor refused.  A leaf now changes when it differs from the live
leaf, and one that goes back to the node's own value is asked as a
checkpoint load asks it (`at_own_value`: skipped by the per-key checks,
kept in the combined ones).  Every door that writes a leaf is tested with
a fit or a load first: the multi-key and single-key `PUT`, and the load.

**A forwarding header cannot make a request look local**
(`MADD-ANO-179`, critical, never released).  The peer the API's
authentication saw is `scope["client"]` as the ASGI server reports it, and
uvicorn's defaults (`proxy_headers=True`, `forwarded_allow_ips="127.0.0.1"`)
rewrite it from `X-Forwarded-For` for every request arriving over loopback
-- which is how a DNS-rebinding page's requests arrive, and a page's
`fetch()` may set that header.  With `X-Forwarded-For: x` the peer was the
string `x`: the Host allowlist was asked only of an IP peer, and the
backstop asked a token only of a routable IP, so a rebound page could
step, read, add nodes and save checkpoints on a server started with
`uvicorn.run(app, host="127.0.0.1")`.  No decision keys on a peer's name
any more.  On a loopback bind only a direct connection from a loopback
address is served without the token: a peer that is not a loopback IP
literal -- a name such as Starlette's `testclient`, a Unix socket's empty
peer -- and any request carrying `X-Forwarded-For` or `Forwarded` must
present it, and the Host check is asked of every request without a valid
token, over HTTP and on the WebSocket handshake.  A valid token is served
under any `Host`.  The cloud entry point and the shipped example servers
pass `proxy_headers=False` (the entry point opts in when the operator sets
uvicorn's `FORWARDED_ALLOW_IPS`), and the docs say loopback must not be a
trusted proxy.  The in-process client gets no allowance in the library: a
test or a script that drives the API with `TestClient(app)` presents
`server.auth.token`, or builds a loopback client
(`TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))`);
the 401 says how.  Nothing in a documented deployment changes but this: a
reverse proxy in front of a loopback bind, whose name goes in
`allowed_hosts`, now has its forwarded requests asked the token -- a remote
client's always were, through the backstop, and a client on the same
machine now is too.

**Writes and adds refuse what cannot run.**  `PUT /graph/state` stored
every field as float32 whatever its dtype, and parsed text and booleans as
numbers (`MADD-ANO-180`, since 0.1.0); this tree coerced to the field's
dtype but truncated LBMNode's `uint8` wall-mask entry 0.5 to 0 with a 200
and answered 500 for 256.  Every element is now asked the rule of the
other write surfaces -- no text, booleans or null for a number, an integer
field only integral values in range -- and refused naming the field.  A
node whose timestep is zero, negative or non-finite was added and stepped
not at all or backwards (`MADD-ANO-181`, since 0.1.0); the request model
bounds it (422) and `GraphManager.add_node` refuses it for every caller.  A
node named `_meta` -- or `_params`, `_params_mappings`, a checkpoint's
prefixes -- broke every later step (`MADD-ANO-182`, since 0.1.0) and is
refused as the other reserved names are.  `POST /sim/start` on a graph
with no nodes answered "started" while its thread died (`MADD-ANO-183`,
since 0.1.0) and is a 409.  A 422 that echoed a NaN was a 500
(`MADD-ANO-184`, since 0.1.0) and is encoded as every reply is.

**A checkpoint load reads headers first** (`MADD-ANO-185`, since 0.1.0).
`load_state` decompressed every member of the archive before checking any
name: a 2 MB file holding an extra 2 GiB member filled 2 GiB before it was
refused.  Each member's name, shape and dtype now come from its `.npy`
header and are checked against the live graph; only a member the load
restores is read.

**Edges.**  A `Host` port of non-ASCII digits (`localhost:\u00b2`, a
latin-1 byte a client can send) was a 500 and is now refused as no name of
this server.  A token outside printable ASCII was accepted at start-up and
then refused every client; it is refused at construction, as a blank one
is.  The token-file warning pointed at a log line above it that came
below; the token is logged first.  uvicorn 0.50.0 hands the app
`Sec-WebSocket-Protocol` as one comma-joined entry; the browser's bearer
carrier is split out of it.  `run_pod.py` exits 2 when it refuses a run and
5 when a goal raises (both were 1, which the runbook reads as a failed
check), and the runbook says so; `--summarise` reads a goal file of any
shape (INVALID, exit 3, where a mistyped field ended it on a traceback),
counts a file with no `checks` (it exited 0), and a fuzz test changes one
field of a record at a time.  Checklist item 3's `sharded_cg` rows are
stated as what they test: the library's solver with and without a mesh, on
`run_pod.py`'s own sharded matvec -- placement and that matvec, not the
solver or its adjoint.  REST-102 now says that a 200 names absolute server
paths (`/checkpoint/save`'s `path`, the trace routes' `log_dir`) where no
4xx does.

## API stability

MADDENING reaches 1.0 by stabilising surfaces **incrementally across the
releases that lead up to it**, not by declaring the whole public API at once.
This release runs the first such round; the judgements, the evidence for each
and the surfaces deliberately left alone are in
[`docs/developer_guide/api_freeze_proposal.md`](../developer_guide/api_freeze_proposal.md).

**Nothing was promoted to `stable`.** The set stays at 17 surfaces, with 260
members as `scripts/check_stable_signatures.py` counts them (the twelve
`experimental` inspection methods are members of the `stable`
`GraphManager`, so the snapshot records them too). Everything this release added is days old: it has no usage history,
and lowering a level again is itself a breaking change, so a promotion made
now is expensive to take back and cheap to defer.

**A `stable` signature cannot change without saying so.**
`scripts/check_stable_signatures.py` records every `stable` signature —
parameters, kinds, defaults, annotations, return type, and for a class every
public method and property an instance *resolves*, inherited ones included —
in `docs/developer_guide/stable_api.json`, and CI compares the tree against
it. A rename, a reordering or a flipped default now fails there instead of
passing because the tests call the surface the new way. An intended change is
accepted with `--update` in the same commit, which is the only supported way
to move the snapshot.

**What was tagged.** This round took the report from 107 surfaces to 141:
the 34 below.  Code that landed after the round was tagged as it landed, so
the report generated for this release is longer again, and
[`stability_report.md`](../developer_guide/stability_report.md) — generated
from the tree — is where its count lives; it is not restated here, because a
count copied into prose is the one that goes stale.

* **Six types a `stable` signature already named** carried no level of their
  own — `SharedSolveResult` (returned by both frozen sharded solvers),
  `UncertaintySpec`, `BoundaryInputSpec`, `BoundaryFluxSpec`, `ShardingIssue`
  and `CouplingGroup`. A frozen method whose return type is free to change is
  half a promise. All six are now `evolving`, and a compliance test reads the
  annotations of every `stable` signature and fails on the next one, so the
  list does not have to be re-derived by hand each release.
* **The `[verify]` harnesses** — 28 surfaces across `maddening.testing.mms`,
  `.verification` and `.strategies` — are now `experimental`. They were absent
  from the report entirely: nothing imports `maddening.testing` on the way to
  anything else, so a documented module with a 30-name `__all__` in `mms.py`
  alone was missing from the contract.
  `experimental` is the honest level while the numeric policy (safety factors,
  asymptotic tolerances, the apparent-order bounds) is still calibrated
  against the built-in nodes only.

**Experimental in 0.4.0, to be hardened in 0.5.0: the surrogate-training
routes and the state streams.**  `POST /surrogate/train`,
`GET /surrogate/status/{job_id}`, `POST /surrogate/activate/{job_id}` and
`POST /surrogate/deactivate/{node_name}`, the WebSocket streams `/ws/state`,
`/ws/state/binary` and `/ws/render`, and `StateRelay`, the snapshot buffer
the streams read, may change in any minor release until 0.5.0 hardens them.
Three REST audits found defects in them this cycle; the fixes above are the
last 0.4.0 work on them, and a defect found in them from here on is
registered and deferred to 0.5.0.  None carried a level before, so this is
their first, not a demotion.  A route has no Python name for `@stability`
to tag, so `SimulationServer`'s `ROUTE_STABILITY` records it,
`maddening.core.compliance.stability.register_route_stability` lists it in
the stability report as `maddening.api.server:<route>`, and an HTTP route
carries `x-maddening-stability` in `/openapi.json`.  `BinaryStateEncoder`,
the frame format the binary stream uses, stays `evolving` as it was
released: lowering a released level is itself a breaking change.

**What is still untagged.** `SimulationServer` (but for the routes above),
`CloudSession`, `CloudConfig`, all of `maddening.viz` but `StateRelay`, and
the concrete surrogate architectures and training callbacks. That is a round of its own and the proposal says so rather than
guessing a level for forty surfaces at the end of a release.

## The rest of the cycle's changelog

The entries of the changelog's `[Unreleased]` section that are not moved above,
in their order and under their headings. The changelog itself now holds a short
version written for a user of 0.3.1.

### Added
- **Anomaly registries of other projects validate: exactly one project version field, and `--version-field NAME`**: `python -m maddening.compliance check-anomalies` (and `validate_anomaly_registry()`) accepts a registry whose top level carries exactly one key named `<project>_version` (the whole key matching `[a-z][a-z0-9_]*_version`, never `schema_version`; `maddening_version`, `mime_version`, `my_lib_version`) with a non-empty string value, and refuses none (which 0.3.1 accepted), two or more, or a non-string value, with a message saying what to change; `--version-field NAME` (`version_field=NAME`) requires that exact name, and `scripts/check_anomalies.py` requires `maddening_version`.
- **`coupling_diagnostics()`: a reason code beside every usable flag that is `False`** (experimental): each entry has `reason_codes`, a list of codes for `spectral_usable`, for `gradient_bound_usable` and for `precision_limited` (the constants of `maddening.core.coupling.reason_codes`), a `not_usable_reason` in words wherever a usable flag is `False` (a plain float32 pair at the default tolerance had none), and the float floor as `residual_precision_floor`.
  Action: branch on the codes, not the sentence: `interface_too_wide` is expected (one pass depends on the previous one through more than seven independent scalars: what the loop closes through under Gauss-Seidel), `spectral_self_check_failed` is a worry, `at_float_floor` says to use float64 or a looser tolerance. Code that took the presence of `not_usable_reason` to mean "numbers withheld" should read the codes instead. No flag and no number moved.
- **The report of a group under `convergence_norm="interface"` with a `multilinear_grid` edge (experimental) reads its float floor at every step**: `coupling_diagnostics()` gives it `precision_limited` by the rule of every group and a new key, `residual_precision_floor` (the floor of the state that step returned, in tolerances). That is the run-time reading of what `compile()`'s positions advisory asks once: float32 markers that drift far from the origin, a state write or a loaded checkpoint are read where they are. Read the floor's size against the tolerance; a floor of one or more says rounding alone is the tolerance asked. The floor and the advisory count a position where the mapping rounds it, at the larger of its distance from the coordinates' zero and of its distance from the grid's first point (its lattice coordinate): far into a long grid, float32 positions are flagged wherever the coordinates' origin is, and float64 positions are the remedy. The floor is pooled over every entry the norm reads (the advisory quotes one part's number) and is the same with `diagnostics=True` as without.
  The advisory now asks only what the floor counts (not a delivered value inside the dead band, a coordinate on an axis of one lattice point, or a point clamped to the hull) and says what its count assumes for a delivered value: a field that varies across a cell by about the size of the value delivered. A gather that samples a field near its zero is not warned of and can be several tolerances off at float32 positions (MADD-ANO-247: hold the positions in float64).
  The bounds of such a group under this norm (`rho_spectral`, `spectral_error_bound`, the gradient bound) are planned for 0.5.0; in 0.4.0 run it under `"mixed"` or `"l2"` with `diagnostics=True` to read them. Guide: "Geometry-dependent mappings".
- **Guide: moving a node's own sampling onto an edge** (`docs/user_guide/moving_a_sampling_onto_an_edge.md`): a worked example of porting a node that interpolates its cell-centred grid at another node's points to a `multilinear_grid` edge. The lattice's origin is the first cell centre (the mesh's corner is half a cell off, silently) and its shape the number of cells; `mode`, `layout`, the time level the points are read at, points held by a third node, a held body as a static sparse mapping. Every snippet runs, and `tests/core/test_moving_a_sampling_onto_an_edge.py` holds the two graphs to each other.
- **Guide: a node that draws random numbers** (`docs/developer_guide/node_authoring.md`): keep the key's raw data (`jax.random.key_data`, a `uint32` array) in the state, wrap it in `update` (`jax.random.wrap_key_data`), split, draw and return the new key's data, with the seed an `int` constructor parameter declared `ParamSpec(trainable=False)`. The section says what holds through each door -- scans and sweeps, checkpoints, configs and USD, coupling groups, multi-rate graphs, adaptive stepping, gradients, `verify_node`, `sysid`, FMU export, REST -- each pinned by `tests/core/test_a_node_that_draws_random_numbers.py`. A coupling-group member draws one sample per step, the same on every pass of the solve (CPL-194). A typed key leaf (`jax.random.key`) still has no checkpoint (MADD-ANO-171, deferred to 0.5.0).
- **Geometry-dependent interface mappings (experimental): `add_edge(..., mapping=m, geometry=(anchor, field))` and the kind `multilinear_grid`** (`maddening.core.coupling.grid_mapping`): a mapping that reads a moving geometry -- a state field of the edge's own source or target node, such as marker positions -- at the time level the edge's value has, in the plain step, coupling groups, sub-cycled members and their gradients; `multilinear_grid_mapping` gathers from and scatters to a uniform grid of one to three axes. Carried by `to_dict` / `from_dict`, USD, checkpoints and FMUs; `register_mapping(..., needs_geometry=True)` registers your own kind.
  Limits in 0.4.0: a coupling group that resolves such a mapping reports its solve and no bound (`not_usable_reason` says why), the adaptive steppers and sharded ends are refused for it, `convergence_norm="interface"` reads a `multilinear_grid` edge (a gather as delivered, a scatter at its source value and its positions in grid spacings; `compile()` warns where the dtype of the positions such a reading rests on, a scatter's own or the ones a gather's value is interpolated at, cannot resolve `rtol` at their distance from zero, e.g. float32 beyond 210 spacings at `rtol=1e-4`: hold them in float64, use coordinates local to the grid, or loosen `rtol`) and is refused for other kinds and sub-cycled groups, and the geometry must be held by the edge's source or target. Graphs without a geometry edge compile to the same programs as before. Guide: "Geometry-dependent mappings" (user guide and interface-mapping guide); claims MAP-026 to MAP-044.
- **Sparse interface mappings (experimental): `StaticSparseMapping` and three kinds, `sparse_nearest_neighbor`, `sparse_projection_1d` and `sparse_matrix`** (`maddening.core.coupling.sparse_mapping`): an integer index kept on the mapping and one weight per slot in `gm.params["mappings"]`, for interfaces whose dense matrix does not fit -- a million points a side build in seconds. The first two hold the dense kind's matrix bit for bit, nearest-neighbour ties included, and results agree with the dense mapping to the rounding of a row sum.
  The conservative nearest neighbour has two forms: a padded gather (the default; one result per compiled program, measured on CPU and GPU; refused past a byte cap with an error naming the other form) and `transpose="scatter"` (compact; measured not reproducible run to run on a GPU). A checkpoint records each sparse pattern's digest beside its weights, and `load_state` and `POST /checkpoint/load` refuse weights saved for another pattern.
  `scipy>=1.14` is now a declared base dependency (the k-d tree; jax already installed it, so no install changes). Interface mappings have their own claims inventory, `docs/validation/mapping_claims.yaml` (`MAP-NNN`; the registry's rows SYS-133 to SYS-143 moved there as MAP-001 to MAP-011). Guide: "Sparse mappings".
- **`register_mapping` (experimental): a mapping kind of your own, serialisable like the built-in ones.** The four interface-mapping kinds are now entries of one registry (`maddening.core.coupling.mapping_registry`); a factory decorated with `@register_mapping(kind, arrays=, hyperparameters=, references=)` survives `to_dict` / `from_dict` and USD, with its hyper-parameters and every reference limit checked before it runs and whatever it raises reported with its edge. An object the kind cannot be rebuilt as (short of a `Mapping` protocol member, or of another `kind`) is refused by `add_edge` and by both writers, as the loader refuses it (MADD-ANO-245, never released).
  A config, a stage or a checkpoint can only *name* a kind: nothing is imported because of what a file says, and a kind this process has not registered is a `MappingRebuildError` listing the ones it has. The built-in kinds write and rebuild exactly what they did.
  A mapping class other than `StaticLinearMapping` must return, from `params_pytree()`, a plain dict from identifiers to finite floating-point JAX arrays, the same on every call; `add_edge` now refuses anything else (MADD-ANO-196, never shipped: such a table was kept by reference and unchecked). Guide: "Registering your own mapping kind".
- **Terminal diagram of the graph** (experimental): `GraphManager.print_graph_diagram(theme=, direction=, use_ascii=, file=)` draws the structure `to_mermaid` exports as boxes and arrows (a frame per coupling group, dotted flux edges and external inputs), coloured by a termaid theme under `rich`.
  The `terminal` extra gains `termaid>=0.9,<1` (MIT, pure Python); `pip install "maddening[terminal]"` to use it. Read-only, like the rest of the inspection API.
- **Eight new or extended examples**, each asserting what it prints: `advanced.profiling_demo` (the full `ProfileReport`), `sysid_demo`,
  `checkpoint_resume_demo`, `sharding_demo` (4 emulated CPU devices), `fmu_export_demo`, `coupling.interface_mapping_demo`,
  `servers.rest_params_demo` (in-process), and `print_coupling_report()` / `strict_convergence` in `convergence_diagnostics_demo`
- **One-command local runs for the remote-simulation examples**: `python -m maddening.examples.servers.remote_viz_client --local` starts the
  simulation server on a free loopback port, streams to the viewer and stops the server on exit (Ctrl-C and errors included); the HTTP example
  servers take `--port 0`, and `cloud/server/04`, `05` and `cloud/multijob/08` gain a `--local` mode that needs no cloud account.
- **Read-only graph inspection** (experimental): `gm.print_graph()` / `format_graph()`, `to_mermaid()` / `to_dot()`, and tables
  `state_summary()`, `params_table()`, `coupling_report()` (caveats flagged), `memory_estimate()` with `print_*` forms; none writes or
  compiles anything.  `maddening.show_versions()` / `python -m maddening info` for bug reports; `print()` of `FitResult` / `FIMReport`
- **CycloneDX SBOMs** in `docs/validation/sbom/` for the base install and the `server`, `surrogates` and `usd` extras, from a
  clean install of the wheel (`scripts/generate_sbom.py`), checked offline against `pyproject.toml` and the SOUP package
  (`scripts/check_sbom.py`, in CI).  Regenerated from the release commit before each tag; see `soup_package.md` §6
- **`coupling_diagnostics()` gains `gradient_relative_error_bound`** (and `gradient_bound_usable`): the
  IFT gradient's relative error at an early exit, under `solver="ift"`, `diagnostics=True`.  About the
  gradient, not the solve -- it reads 0.0 on an affine group whose state is far off; read `spectral_error_bound` for that
- **`coupling_diagnostics()` gains a bound, `spectral_error_bound`** (with `rho_spectral`,
  `spectral_usable`): eight Arnoldi steps on `dF/dx` under `solver="ift"`, `diagnostics=True`; 8.05x
  *over* the true distance where `error_estimate` is 122x under.  Reported, not applied; existing keys unchanged
- **`WaveletAdaptiveNode`** (`maddening.nodes.adaptive`, `EXPERIMENTAL`): the first
  concrete `AdaptiveNode` — an interpolating-wavelet solver for `(-Δ + m) u = f`
  in 1/2/3-D with a CDD active set; declares and measures spatial order 2 by MMS
- **`fim(scale="nominal", specs=gm.param_specs())`: a column scale from the
  bounds width, not the value** — a parameter at `0.0` is judged on the data;
  columns with no finite width stay value-scaled and `FIMReport.value_scaled` names them
- **`sysid.fim_core` / `FIMCore`: the Fisher information with no host round
  trip** — jittable, zero device syncs, device-array verdicts for a control
  loop.  `fim` itself is fast only with `reuse_trace=True`, for a pure residual:
  ~1 ms warm against ~0.2 s per default call, which re-traces (see Fixed)
- **The 0.4.0 stability freeze round** (`docs/developer_guide/api_freeze_proposal.md`)
  plus a guard that a `stable` signature cannot change unannounced
  (`scripts/check_stable_signatures.py`): 34 surfaces tagged, none promoted; `stability_report.md` is the full list
- **`py.typed`: MADDENING's annotations now reach your type checker.** Under
  PEP 561 they were all ignored downstream; nothing to do but upgrade. The
  pyright job is blocking too — `core`/`nodes`/`fmi`/`cloud`/… at zero errors
- **GCI / Richardson mode in `maddening.testing.mms`** for nodes MMS cannot
  reach: `assert_node_gci_verified(node, solution_at=..., levels=...)` needs
  no source term and no reference — three refinements give an error band
- **`fim` says when `rank` was decided at the float32 noise floor**: a
  `PrecisionLimitWarning` naming the eigenvalue ratio, the cutoff and the
  `jax_enable_x64` re-run that settles it.  Quiet on well-conditioned problems
- **`ResolutionStatus.PARTIALLY_RESOLVED`** — MADDENING's own
  `known_anomalies.yaml` has used `partially_resolved` since MADD-ANO-005 was
  written; the enum could not represent the registry this project ships
- **`SimulationNode.static_data_deps()`** declares which parameters a
  `static_data` array was derived from; `compile()` now refuses a graph whose
  static derives from a *trainable* parameter — freeze it or stop deriving it
- **Measured guidance for choosing coupling options**: eight graph fixtures and
  a full option sweep behind `docs/developer_guide/coupling_algorithm_guide.md`;
  `profile_graph` gains `n_stat_steps` to pin the coupling-statistics window
- **Coupling groups are serialisable**: `to_dict` / `from_dict` carry a
  `coupling_groups` key with all 19 `CouplingGroup` fields, and the USD stage
  carries the same set, so a reloaded graph solves the way the saved one did
- **Graph parameter pytree**: the compiled step is `step_fn(state,
  external_inputs, params)`, so node constants are traced inputs that
  `jax.grad` reaches and that change without a recompile.  Opt a node in with
  `update(..., *, params=None)`; `gm.nodes_without_params()` lists those still
  baking constants
- Migrated to the params contract: `SpringDamperNode`, `BallNode`, `HeatNode`,
  `RigidBodyNode`, `RigidBody2DNode`, `HeartPumpNode`, `TableNode`,
  `HealthCheckNode`, `LBMNode`, `LBMPipeNode`, `SurrogateNode` (network
  weights), flux producers via `compute_boundary_fluxes(..., params=)`, and
  `ShardedStencilNode` / `ShardedUnstructuredNode` (which previously ignored
  `gm.params` entirely)
- **`ParamSpec`** (`maddening.core.params`): per-parameter `trainable` /
  `bounds` / `transform`, declared in `SimulationNode.param_specs()` or
  `gm.set_param_spec`, with `gm.trainable_mask()`, `gm.unconstrain()` /
  `gm.constrain()` and `gm.check_params()` as the optimiser-facing maps
- **`maddening.sysid`**: `fit`, `fit_lm`, `fit_multiple_shooting`,
  `windowed_loss` and `fim` (`mask=`, `noise_std=`, Cramér-Rao bounds); all
  fitters emit `EVENT_FIT_PROGRESS`.  Guide: `docs/user_guide/parameters.md`
- **Interface mappings on edges**: `add_edge(..., mapping=)` with
  `rbf_mapping`, `nearest_neighbor_mapping`, `projection_1d_mapping`,
  `matrix_mapping`; weights live in `gm.params["mappings"]`, are reachable by
  `jax.grad`, and are `trainable=False` until you opt in
- **Interface mappings are serialisable** (`MappingSpec`): config, USD and the
  REST view carry mapped edges instead of refusing them.  Pass `source_ref=` /
  `target_ref=` to the factories; an incomplete spec is refused with the
  argument named
- Params survive persistence and reach FMI: `to_dict`/`from_dict`, USD and
  checkpoints store effective params and `ParamSpec` overrides;
  `build_model_description` exposes each leaf as an FMI `parameter`/`tunable`;
  `SidecarConfig(params=, param_specs=)` serves `get_params` / `set_params`; it and `set_fmu_state`
  refuse a non-finite value, one the leaf's dtype cannot hold or one out of bounds, as the bridge does
- REST `PUT /graph/params/{node}` addresses any leaf of the live pytree,
  updates in place without a recompile, and validates dtype, shape, finiteness
  and `ParamSpec` bounds before writing
- **FMU C wrapper, TCP bridge and packaging**: a graph now builds a real FMI
  3.0 co-simulation `.fmu` (`build_fmu_binary`, `write_fmu`), driven end to
  end by FMPy.  A bridge starts once: `start()` again, or after `stop()`, raises `RuntimeError`
- **FMU sidecar protocol 2 — binary frames** for bulk
  `get`/`set`/`get_state`/`set_state`, negotiated at `hello`; JSON-only
  clients are unaffected.  See "Wire protocol" in
  `docs/user_guide/fmu_export.md`
- **Multi-clock FMU export**: `build_model_description(multi_clock=True)`
  emits one `<Clock>` per distinct node timestep and tags outputs and inputs
  with theirs.  Off by default
- **`AdaptiveNode`** (`maddening.nodes.adaptive`, `MADD-NODE-009`):
  frozen-active-set adjoint pattern for adaptive solvers, with
  `gradient_capture_ratio`, `check_gradient_capture`, `mask_safe` and
  `set_adaptive_diagnostics`.  Read `MADD-ANO-003` before optimising through
  one.  Guides: `algorithm_guide/nodes/adaptive_node.md`,
  `developer_guide/adaptive_node.md`; benchmark `MADD-VER-004`
- **Per-neighbour unstructured halo exchange**: `exchange_unstructured(...,
  method="ppermute")` and `ShardedUnstructuredNode(..., exchange="ppermute")`,
  bit-identical to the `all_to_all` default; `exchange_traffic(layout)`
  reports what each transport would move so you can choose before using a GPU
- `sharded_cg` / `sharded_gmres` take `differentiable=True` for exact
  linear-solve adjoints through the same backend and preconditioner, plus
  `jacobi_preconditioner` / `block_jacobi_preconditioner`
- `SimulationNode.domain_integral_axes()`: reduce a domain integral over a
  subset of mesh axes, or not at all, so a partial-surface integral needs no
  full-mesh `psum`
- Second `@stability` wave: the unstructured partition layout, halo exchange
  and partition/gather helpers, `SidecarConfig` and `FMUState` are `EVOLVING`
- **Persistent compilation cache**
  (`maddening.core.simulation.compile_cache`): `enable(cache_dir)`,
  `MADDENING_COMPILATION_CACHE_DIR`, and `warm_cache()` to compile step and
  scan ahead of a run
- Profiler rewrite (coupling overhead measured not inferred, per-group
  iteration statistics, `trace=True` kernel attribution;
  `docs/developer_guide/profiling.md`) and `benchmarks/bench_coupling.py`
- `GraphManager.reset_state()` (reset without retracing the jitted step) and
  `GraphManager.get_node()`
- The USD stage stores a node's own name (`maddening:nodeName`) and an edge's
  declared units, so node names USD cannot spell survive a round trip
- Coupling diagnostics: iteration count and residual always in `_meta`,
  `coupling_diagnostics()` reports `"converged"` per group, and
  `CouplingGroup.strict_convergence` raises on an unconverged exit (off by
  default — the IFT gradient is invalid there)
- The IFT Krylov adjoint needs `lineax`, which is a base dependency as of
  this release (it was an optional extra when this entry was first written)
- Node verification: `verify_node` gains `params_consistent` /
  `params_gradient_finite` / `params_effective` and a `SKIP` status
  (`SimulationNode.accepts_params()` exposes the probe);
  `maddening.testing.verification` is a Hypothesis battery over outputs,
  structure, determinism, jit/eager agreement and gradients;
  `strategies.node_states` samples bool and integer fields.  For a field
  that only means something under a constraint (a unit quaternion, positive
  populations, an index), `verify_node` takes `constrain_state=` /
  `constrain_boundary=` (a function applied to every draw, in every check)
  and `state_strategy=` / `boundary_strategy=` (your own Hypothesis strategy)
  Beyond the node (experimental): `maddening.testing.mapping.verify_mapping` / `assert_mapping_verified` check an interface mapping of any kind, registered or not, or a whole `EdgeSpec` (linearity, the claimed consistency / conservation, the adjoint identity, the position derivative, the save/load round trip),
  and `maddening.testing.coupled.verify_graph_order` / `verify_graph_gci` measure a coupled graph's order with a guard on the coupling's iteration error (`docs/developer_guide/verification.md`, "Verifying a coupled model in three levels").

### Changed
- **`run_pod.py`: the goals run three wrapper paths they never ran, and a fault on each now fails them** (session tooling, never released as a result; record schema 10, older files are refused): the unstructured goals' node reads a per-cell `source` and takes its `weight` from the injected `params` on every step of `forward`, `gradient` and `indivisible`, and `gradient` differentiates with respect to both; the `stencil` goal adds the field with a halo of two cells under periodic, edge and Dirichlet ends on every mesh (12 cases, 160 to 280 checks at one size; `gradient` 16 to 24).
  Until then `ShardedUnstructuredNode` with its per-cell inputs zeroed or its injected parameters doubled, and `ShardedStencilNode` zero-filling halos two or more cells wide at the global edges, passed all eight goals; ten seeded faults of those paths now fail every goal that runs them (`tests/cloud/multigpu/test_run_pod_seeded_faults.py`). The dry run takes about a third longer, and the runbook's time boxes for `stencil` and `forward` rose to 45 and 20 minutes (`benchmarks/multigpu/README.md`).
  Record schema 11 adds four more (the zero fill, `domain_integral_axes()` on both wrappers, the unstructured wrapper's `shard_info`), each with seeded faults that fail `stencil` or `forward`; the runbook has a table of every wrapper option with the goal and seed that prove it, or "not run"; and the default dry run's second size is 961 cells, which the unstructured goals run in the partitioner's scattered order.  Record schema 12 adds a stencil case whose static is split along the second sharded axis and one on a pencil whose mesh axes are crossed, and reads a reduced domain integral back from the state: three faults that passed every goal now fail `stencil`; the runbook launches the session on an image built from its commit with the JAX CI tests (`docker/Dockerfile.cloud` takes `JAX_SPEC`), not on `:latest`.
- **`add_external_input` refuses a second declaration of a field that differs from the first** (`ValueError` naming the field and both; `from_dict` and `load_graph_from_usd` refuse a file that holds two): a declaration's shape and dtype are what the step casts a supplied value to, what the FMU export types the input as and what `to_dict()` saves, and until now a second one was taken and the last won (in 0.3.x it decided the zeros of an omitted input). Declare each input once: a script that declared one twice with different shapes or dtypes keeps what it ran by keeping the last. The same declaration again is accepted and kept once.
- **A coupling group whose boundary fluxes read each other in a loop is refused at `compile()`** (`ValueError`; an `ERROR` line of `validate()`), and so is a chain of fluxes that runs against the group's sweep order for more than one step where the pass reads it: a flux is not part of what a group iterates, and such a group reported `converged=True` at a state that was not the coupled solution (never released: 0.3.1 raised a bare `KeyError`; MADD-ANO-276). **Action:** compute the quantity in the node's `update()`, return it as a state field, and name that field as the edge's `source_field` (the `flux_field_*` of `add_robin_coupling`); a group whose fluxes are computed from the state (or from inputs that state fields deliver) is not affected, whether or not the nodes declare their inputs. A flux hook that indexes an input (`boundary_inputs[name]`), or a field its state does not hold, is now an `ERROR` line of `validate()` naming the node and that key, where `validate()` raised a bare `KeyError`: read inputs with `.get`.
- **`copy.copy(gm)` of a `GraphManager` is refused** (`TypeError` naming `copy.deepcopy`): the shallow copy shared the graph's nodes, its parameter tree and its record of the `node.params` writes it had taken in, but not its compiled step, so after a write to a node whose `update` takes no `params` the graph that stepped second ran the old value while both reported the new one, with nothing said. Use `copy.deepcopy(gm)`, which is a graph of its own and runs and reports what the original does.
- **`to_dict()` warns of a constructor argument it does not save** (MADD-ANO-277, since 0.1.0): a node class that keeps a constructor argument outside `params` (`self.stiffness = stiffness`, as the quickstart taught) had it dropped by every save, and the reloaded graph ran the constructor's default with nothing said. `SimulationNode.to_dict()`, `GraphManager.to_dict()` and `save_graph_to_usd` now give a `UserWarning` naming the node, its class and the arguments, for a node constructed with such an argument at a value that is not exactly its default. The rule goes by the argument's name: a class that saves an argument another way overrides `to_dict()` to say so, which a config honours and a USD stage (it carries `params` only) does not, and an argument taken in by `**kwargs` or `*args` is not looked at. Pass every constructor argument to `super().__init__(name, timestep, key=value, ...)` and read it from `self.params`; the quickstart, the node authoring guide and the parameters guide now say so, and `GraphManager.to_dict`'s docstring says what a config holds (an array in `params` is not converted for JSON).
- **`sharded_cg` / `sharded_gmres` refuse a NaN or negative `rtol` given as a concrete number** (`ValueError`, as for `atol`; MADD-ANO-271, in 0.3.0 and 0.3.1, where a NaN `rtol` returned zeros with `converged=True` on the default backend): pass a number >= 0. A traced `rtol` or `atol` that is negative or NaN cannot raise and now returns NaN in every entry with `converged=False` (this replaces "are still passed on" in the entry below).
- **`compile()` refuses a coupling group whose integer, Boolean or key fields are computed from one another in a closed loop** (`ValueError`, and an error from `validate()`; MADD-ANO-284): two members that each compute a flag from the flag the other hands them, for one. Such fields have no value at a floating iterate, so a solve could read `converged=True` while they still change. Store one field of the loop in floating point, compute it outside the group, or use `max_iterations=1`. The loop is read from a trace of the group's pass, not from its edges: a step counter or a PRNG key beside edges or boundary fluxes is not on one, printed or not, and such a group steps as it did; the message calls a link the trace cannot see through (a `custom_jvp` function, a callback) assumed, and names it. A group that carries such fields along a chain that ends is solved: both solvers evaluate them at the iterate, so `solver="fori"` now takes the passes `"ift"` takes there.
- **`SelkiesSession` applies `StreamConfig.ice_servers` to its WebRTC endpoint, refuses a frame whose size or format differs from the stream's (`ValueError`), and `start()` raises `StreamStartError` when the signaling socket cannot listen**: the default `ice_servers` names a public STUN server, which the session now contacts when a viewer connects; pass `ice_servers=[]` to contact none. Frames pushed while no viewer is connected are dropped instead of kept (MADD-ANO-286).
- **The WebRTC signaling socket answers what it cannot use instead of ignoring it, and a viewer's malformed message no longer ends its stream**: an offer with no H.264 (packetization-mode=1) at a payload type of 96-127 (VP8, audio or data only, packetization-mode=0, H.264 only below 96) is sent `{"type": "error", ...}` and closed with code 4001, which `webrtc_client.html` shows; a message that is not JSON, nests deeper than the parser goes or has a `type` that is not a string is answered with an error instead of being dropped or closing the socket, and an input handler that raises is logged with its exception instead of closing the viewer's socket and tearing its pipeline down. `start()` on a started session raises `RuntimeError` and leaves it running.
- **`SelkiesSession.update_framebuffer_cpu` counts bytes, and the session asks no router for a port mapping**: a frame is a C-contiguous buffer of unsigned bytes holding exactly width x height x 4 bytes (a `(height, width, 4)` uint8 array included); a buffer of wider items (float32, uint32) is refused with `ValueError` even when its element count matches. The ICE agent's UPnP is off, so with `ice_servers=[]` the session sends nothing but connectivity checks to the viewer's candidates. the session imports jax before its first call into GStreamer (numpy, then GStreamer's library, then jax aborted the process in the test image); the streaming test container runs with `--cap-drop ALL --cap-add NET_ADMIN` and its tests with no capability (`docker/README.md`).
- **`fit_lm` no longer needs memory quadratic in the residual's length** (never released; `fit_lm` is new in 0.4.0): its identifiability guard took a full singular value decomposition of the Jacobian, whose `m x m` factor nobody read -- 6.4 GB at 20,000 residual rows, and a `MemoryError` at 100,000 after the fit had converged. It takes the reduced one; the guard's verdicts are unchanged.
- **`fim` and `fim_core` read `rank`, `crb`, `cond`, `eigvals` and `eigvecs` from the singular values of the Jacobian**, not from the eigenvalues of `F = J.T @ J` (MADD-ANO-289, never released: a steady state or a step aligns that product's rounding, and two parameters that enter a residual only through their sum could read rank 2 of 2 with a finite `crb`). The default `rank_rtol` keeps its value; `report.fim` is still the Gram product, so read identifiability from `rank` and `crb`, not from a decomposition of `report.fim`. Past about 16,300 rows in float32 `fim_core`'s default cutoff is the larger, and every direction between the two cutoffs is `precision_limited` there.
- **Differentiating through `sharded_cg` / `sharded_gmres` without `differentiable=True` is an error** (MADD-ANO-266, in 0.3.0 and 0.3.1): `jax.jvp`, `jax.grad`, `jax.jacfwd` and `jax.jacrev` through such a call now raise a `ValueError` that says what to do. Pass `differentiable=True` to a call you differentiate (`.iters` then reads -1); or, if the solve is meant to be a constant, put `jax.lax.stop_gradient` on what goes into the call. Until now the iteration itself was differentiated, which gave a silently wrong derivative when the tangent or cotangent was not of `b`'s size. A call that is not differentiated is unchanged.
- **`sharded_cg` / `sharded_gmres` refuse a negative or NaN `atol` given as a concrete number** (`ValueError`; one that is traced, and a NaN `rtol`, are still passed on: MADD-ANO-271), and with `differentiable=True` an `x0` and an `atol` apply to the primal solve only: the tangent and cotangent solves start from zero and are purely relative (MADD-ANO-267, MADD-ANO-268; `differentiable=` is new in this release).
- **`sharded_cg` / `sharded_gmres` on the loop backend (`backend="loop"`, and every preconditioned solve): `converged` can read `False` where it read `True`** (MADD-ANO-263): it is now `||b - A x|| <= max(atol, rtol * ||b||)` on the returned value, with no allowance for rounding. A float32 solve asked for a tolerance it does not reach (at the default `rtol=1e-6`: a condition number over about 17) answers `converged=False` and the true `residual_norm`, and one that stopped within the float32 floor of its tolerance can read `False` by a hair (measured: 1.06 times `rtol`). Code that branches on `converged` takes the other branch there: ask for an `rtol` a few times above `eps * kappa` (in float32, about `1e-7` times the condition number), or solve in float64, or read `residual_norm`. `run_pod.py --goal gradient` now reads the iteration count against its cap instead of this flag.
- **A graph in which an edge delivers to a field that is also a declared external input no longer compiles** (released behaviour, corrected: MADD-ANO-265; 0.1.0 to 0.3.1). The input replaced whatever the edge delivered on every step, with zeros when it was not fed, so the edge never reached the node and nothing said so (on 0.3.x the edge got through only when `external_inputs` was a dict that left the input out). `compile()` and every entry point that compiles now raise a `ValueError` naming each such pair, and `validate()` lists it. Keep one of the two: leave the `add_external_input` out where the graph is built (there is no `remove_external_input`), or call `remove_edge(...)` as the message spells it.
- **`run_pod.py --goal forward`: the cross-shard sum is checked exactly, on a field of ones** (session tooling, never released as a result): one more public step from a field of ones, which the node leaves ones bit for bit, must return `total` equal to the number of cells and a field still all ones, limit zero, sharded and unsharded; not run at 2**24 cells or more (`CHECK NOT RUN`, exit 0). The float `total` of the goal's cancelling field (`sum|x| / |sum x|` of 22 to 69,000) stays in the record and is held to no limit: orders of addition differ by up to 3.5e-4 of it, as much as a ghost row counted. Record schema 9. Action: re-run any dry run recorded at schema 8 (it reads `INVALID`).
- **Float32 matrix products on NVIDIA GPUs: documented, and fixed for the hardware session** (docs and session tooling; no code change): JAX computes float32 matrix products at reduced precision by default on these cards, and the lattice Boltzmann nodes (`LBMNode`, `LBMPipeNode`) and MLP surrogates use them. `docs/user_guide/installation.md` says so and gives the setting (`JAX_DEFAULT_MATMUL_PRECISION=highest`, or `jax.config.update("jax_default_matmul_precision", "highest")`); the multi-GPU runbook exports it before the goals, records it in `session.txt`, and measures the default once in its non-gating tail. On CPU the setting changes nothing (checked: the products are bit-identical). Action: on an NVIDIA GPU, set it for any float32 run that uses those nodes.
- **`run_pod.py --goal gradient`: the `sharded_cg` part solves a system float32 CG converges on, and fails unless it did** (session tooling, never released as a result): the 1-D Laplacian shifted on its diagonal (`CG_SHIFT = 0.03`, condition number 134) to `rtol` 1e-4 for a white-noise right-hand side, 55 to 58 iterations a solve at every size; each side's solve, adjoint solve and tangent solve must have stopped on the tolerance with a true residual within 2e-4 (computed on the host in float64), and the parity limit is 4e-4 (was 1e-3, read at 5.3e-4 to 9.2e-4 on solves of the unshifted operator that had not converged; now 2e-7 to 8e-7 on CPU). Record schema 8. Action: re-run any dry run recorded at schema 7 (it reads `INVALID`).
- **`coupling_diagnostics()` withdraws `spectral_usable` and `gradient_bound_usable` at the float floor behind a long row of a static mapping** (MADD-ANO-257, open; never released): the float floor does not count a mapped edge's row sum (a stalled float32 pair read its bound at 0.68 of the true distance behind a scatter-layout row of 100 entries, 0.002 behind 3e4, and 0.18 behind a dense matrix of three rows of 3000). Where an internal edge carries a static mapping of any kind (a dense matrix, counted at its width; a sparse mapping in either layout) with a row over 10 entries and the residual is not above the floor times the row's length, the flags are `False` with a `not_usable_reason`; every number is reported as computed. A geometry-dependent scatter (`multilinear_grid` from points to a grid, experimental) is counted at its number of points, whichever side its positions are anchored at: how many markers share a grid entry is state. In practice a float32 group whose mapping sums a few hundred entries a row loses the two flags at `rtol` 1e-4 or tighter; hold it in float64 at the same tolerance to keep them.
- **`compile()` emits a new `UserWarning` for every coupling group that sets `atol > 0`**, whatever its member count and schedule (the advisory of MADD-ANO-254; `validate()` returns the same line): a run that turns warnings into errors (`-W error`, `filterwarnings = ["error"]`) now fails on such a graph. Set `atol=0.0` (the default) there, or filter the message; no group is excepted.
- **`POST /graph/edges` and `POST /graph/nodes` refuse a key they do not apply** (released behaviour, corrected: MADD-ANO-253; 0.1.0 to 0.3.1). A body with `additive`, `transform`, `mapping`, `geometry`, a misspelt `params` or any other unknown key was answered 201 and the key dropped, so the server built another graph than the one asked for. It is now a 422 whose detail names the key, and nothing is added: send only `source_node`, `target_node`, `source_field`, `target_field` (edges) or `type`, `name`, `timestep`, `params` (nodes), and add an additive, transformed or mapped edge in-process with `GraphManager.add_edge`.
- **`validate()` reports an edge from a node to itself**: one `INFO:` line for each such edge, saying which of its own values the node reads through it (the previous step's outside a coupling group, so the term is explicit; its new value at convergence in a group, so the term is implicit; a boundary flux outside a group cannot be read, MADD-ANO-157). It returned nothing for such an edge. An `INFO:` line is not an error and `compile()` does not warn for it; code that tests `validate() == []` on a graph with such an edge should test for `ERROR` and `WARNING` lines instead.
- **`validate()`'s line for a loop that a coupling group iterates names that group's `iteration_mode`** (released behaviour, corrected; in every release to 0.3.1): it ended `(Gauss-Seidel)` for a `"jacobi"` group too, and now ends `(Jacobi)` there.
- **What a coupling group returns under `convergence_norm="interface"`** (released behaviour, corrected: MADD-ANO-240, MADD-ANO-241; 0.1.0 to 0.3.1). The norm measures what the internal edges deliver, and the iterate it accepted was returned as it was, so every field the norm does not measure whole could come from a pass before the readings it stopped on, with `converged=True`: a one-way pair under `iteration_mode="jacobi"` returned its target computed from the pre-step source at `iterations=1`, and the part of a field a mapping or a transform does not deliver was a pass behind (or, under `"fixed"` and `"aitken"`, a blend of passes). Now a field an internal edge delivers as it is, with no mapping and no transform, keeps the accepted iterate's value bit for bit, and **every other floating field of every member is returned as one plain pass computes it at the accepted iterate**, under either solver, schedule and acceleration, converged or at the cap. `iterations`, `residual`, `converged` and the bounds are unchanged and describe the accepted iterate; the state returned is within the reported residual of it on what the edges deliver. What to expect: one more evaluation of the pass per solve in a group with such a field (none where every field is delivered whole); trajectories stepped at an unconverged cap differ from 0.3.1's; an identity-mapped group equals its unmapped twin within the residual, no longer to the bit; a recomputed field is occasionally further from the fixed point than the iterate's value, by at most about a tolerance in a converged solve. Use `"mixed"` or `"l2"` where every field must meet the criterion itself.
- **`GraphManager`'s methods that take or return a `ParamSpec` are `EVOLVING`, as `ParamSpec` is**: `param_specs`, `set_param_spec`, `trainable_mask` and `param_spec_overrides` (all new in this release) now carry `@stability(StabilityLevel.EVOLVING)`. `gm.params` and the `params=` argument of the run methods stay `STABLE`.
- **Coupling diagnostics read a `multilinear_grid` geometry (EXPERIMENTAL).** A single-rate coupling group whose geometry-dependent mappings are all `multilinear_grid` now reports `rho_spectral`, `spectral_error_bound` and `gradient_relative_error_bound` under `convergence_norm="l2"` or `"mixed"`, which every such group withheld so far; with `diagnostics=True` the step checks its Jacobian-vector product along the positions against a finite difference and the report withholds the bounds (`not_usable_reason`) where they disagree. `spectral_usable` and `gradient_bound_usable` are set only where every position is fixed during the pass (a target-anchored geometry that `update` reads, or positions held by a node outside the group). The float floor of such a group counts the rounding of the weights the kernel forms from the positions in their dtype (one rounding of the lattice coordinate per evaluation, for every value field): float32 positions hundreds of spacings from the grid's first point put the report at its floor (`precision_limited=True`, both flags withdrawn unless every member declares `update_evaluations()`), and `compile()` says so; hold the positions in float64 to keep the flags (MADD-ANO-261, never released). A group that solves positions (its pass reads one from the iterate, or builds one and reads it in the same pass) has both flags `False` on every step: the numbers are the linearisation at the returned iterate and the stencil is another polynomial across a lattice plane, so they are reported as computed, uncertified, and `not_usable_reason` says so (MADD-ANO-242 and MADD-ANO-252, never released). Other mapping kinds, sub-cycled groups and the interface norm keep the withheld report, with a reason that says which. See `docs/user_guide/geometry_dependent_mappings.md`.
- **On a multi-rate graph a slow node's flux is the flux of the state the node holds** (released behaviour, corrected: MADD-ANO-235, in every release to 0.3.1): `compute_boundary_fluxes` of a node slower than the base step was called on the result of `update` before the step decided whether to keep it, so between the node's firings a reader of the flux received the flux of the *next* state (a counter holding 1, 1, 2, 2 was read as 1, 2, 2, 3). The hook is now called on every base step with the held state and that step's inputs; single-rate graphs and nodes at the base step are unchanged. If you compensated for the lead, remove the compensation.
- **A step that changes the layout of the state raises `ValueError`; it used to be stored** (released behaviour, corrected: `MADD-ANO-220`): `GraphManager.step`, `run`, `run_adaptive` and `FmuSidecar.step` refuse a stepped state whose leaf has another shape or kind of dtype than the state it replaces, or whose fields differ, naming the node and the leaf; nothing is stored. A node's `update` must return the layout `initial_state()` built: build a state that grew at its first step at its final shape.
- **`add_coupling_group` refuses a boolean option that is not a boolean, and `rtol=0` under the norms that divide by it**: `diagnostics`, `subcycling` and `strict_convergence` raise `TypeError` for anything but `True`/`False` (`diagnostics="off"` turned the diagnostics on); `rtol=0` under `convergence_norm="mixed"` or `"interface"` raises `ValueError` (the residual was `0/0`, the step ran to its cap and the report raised `ZeroDivisionError`). Action: pass a bool; use a positive `rtol`.
- **A multi-rate graph whose schedule would not keep a node's clock no longer compiles** (MADD-ANO-227, in every release to 0.3.1): timesteps more than about 1e9 apart (`1.0` and `1e-10` got rate dividers of 1 and 0 and ran on different clocks, silently), or with no common step the float GCD finds, are a `ValueError` at `compile()` naming the two nodes, an error from `validate()`, and a 400 from `POST /graph/nodes` and `DELETE /graph/nodes/{name}`. The dividers of every graph that is kept are unchanged. Give the nodes timesteps that are whole multiples of one step.
- **REST numbers are strict, and counts are held to their declared range** (MADD-ANO-228): `"timestep": true` (a node at 1.0 s), `"0.5"` and `" 0.25 "` are a 422, as are a boolean or text for any number of a request model; an integer query parameter is decimal digits only; `POST /sim/profile` answers 422 for `n_steps` outside 1..1000 or `n_warmup` outside 0..50 where it clamped them without saying so. Send JSON numbers, and counts inside the documented range.
- **`maddening.__version__` is the tree's own version, not the installed distribution's** (a source tree on `PYTHONPATH` beside another installed version called itself that version, in `GET /healthz` and every record), and **an `Authorization` header is read by RFC 9110's grammar**: the scheme in any case, one or more spaces, the token exactly; a tab or other whitespace beside the token no longer matches.
- **An external input runs in its declared dtype at every entry point** (`add_external_input(..., dtype=)`, default `float32` also under `jax_enable_x64`): `step`, `run`, the scans, sweeps, adaptive runs and the raw compiled step now cast a supplied value to it, as the FMU export, the zeros of an omitted input and the saved configuration already took it. An x64 graph with a default-declared input used to run `0.1` as a float64 where its FMU ran `float32(0.1)` (MADD-ANO-218); it now runs `float32(0.1)` and says so once per input with a `UserWarning` whenever the cast changes a supplied value. Declare `dtype=jnp.float64` to keep float64 inputs; nothing changes without x64, or for values the declared dtype holds exactly.
- **`build_model_description(default_step_size=)` takes a finite, positive Python or NumPy number that is a whole number of graph steps to within four ulps, and refuses anything else by name** (a bridge given a hand-built description likewise). A NumPy scalar used to be written as `stepSize="np.float64(0.05)"`, which importers reject (MADD-ANO-219, shipped in 0.3.x); a step such as `float(np.float32(0.05))` on a 0.01 s graph was accepted and refused at the fourteenth `fmi3DoStep`. Pass `n * gm.timestep`.
- **`ParamSpec`: `constrain` lands inside the bounds `check` enforces where a bound is a subnormal number** (it returned `-0.0` for `(None, -1.4e-45)`), a `logit` spec with a subnormal bound or an interval under four smallest normals of the leaf's dtype is refused by name, and `fit` / `fit_lm`'s curvature test answers "no curvature test" with its warning, not `LinAlgError`, when a Jacobian column or Hessian product overflows over its scale.
- **`fit_lm` ends `converged=False`, with a `RuntimeWarning`, where its run stops beside a jump of the residual; and it warns when `excited_rank` reads the full count on a fit whose `JᵀJ` does not**: on a residual that is not differentiable in the trained parameters (a node with a contact, an event or a valve: `BallNode`, `HeartPumpNode`) the floor rule reported `converged=True` at the edge of a smooth piece, at losses of 0.02 to 3.7 against 1e-13; it now tells a jump from the rounding floor by comparing each small rejected step with its mirror image. Smooth fits are unchanged. The condition is stated in the `fit_lm`, `fit`, `fim` and `FitResult.converged` docstrings and in the parameter guide; `excited_rank` is defined as the count minus the directions the guard found undetermined.
  Action: on a model with events, read `converged` and the warning, and compare `best_loss` across starts; declare `transform="log"` on an identity parameter that shares a scale (`HeartPumpNode.stroke_volume`).
- **CI: the per-push lane runs six shards per JAX version (it ran four), with six files pinned afresh** (`MADDENING_TEST_SHARD=i/6`; `PINS` and `PINS_FOR` in `tests/_sharding.py`): on three cold push runs the slowest of four jobs took 43 to 48 minutes; six are predicted at 28 to 30 (33 to 37 by hash alone). The files are dealt afresh, so the first run after the merge compiles cold and saves new caches; the cache keys and the durations artifacts are now named `-shard<i>of6`.
- **CI: the slow lane runs ten shards per JAX version (it ran eight), dealt from a weights table regenerated from the runs of 2026-10-10** (`MADDENING_TEST_SHARD=i/10:weighted`; `tests/slow_lane_weights.json`: 243 files and 635 minutes, against 211 and 477): 44 files of ten seconds or more were not in the old table (35 added since, 9 that grew) and went by the hash of their path, and one shard ran 122 to 125 minutes against a mean of 80. Predicted on the slowest shard: 66 to 67 minutes (81 to 83 at eight; 56 at twelve, not taken yet), on twenty jobs. The lane's artifacts are now named `-shard<i>of10`.
- **CI: the slow lane runs eight shards per JAX version, split by measured time** (`MADDENING_TEST_SHARD=i/8:weighted`, `tests/slow_lane_weights.json`, `scripts/slow_lane_weights.py`): by hash, one shard of four held nearly half
  of the lane and reached its 175-minute timeout. Every test still runs once; the per-push lane's split is unchanged.
- **`GraphManager.remove_node` (and `DELETE /graph/nodes/{name}`) takes the node out of its coupling group** (`MADD-ANO-214`): the group keeps its options over the remaining members and is removed when one would remain; it used to go on naming the node, and every compile and step failed until a node of that name was added. A `UserWarning` (the route's `coupling_groups` reply) names each group that changes: after re-adding a member under its name, call `add_coupling_group` again. A node whose points a mapping on an edge between two other nodes was built from is refused (`ValueError`, 400): remove that edge first. `replace_node` keeps the membership.
- **`maddening.core.graph_manager` holds `GraphManager`; its private module-level machinery moved to private modules**: the coupled block, the fixed-point loop, the IFT solve, the spectral and gradient bounds, the convergence reports, the adaptive scan, the parameter probes and the bookkeeping structs are now in `maddening.core.coupling._coupled_block`, `._fixed_point`, `._ift`, `._bounds`, `._reports`, `._group_layout` and `maddening.core._adaptive_scan`, `._param_probes`, `._graph_specs`, moved verbatim (the lowered step programs are unchanged on every tested jax). Every public name (`GraphManager`, `ExternalInputSpec`, `ShardingIssue`, the `EVENT_*` names, `GRADIENT_PROBE_ENTRY_LIMIT`) imports from `maddening.core.graph_manager` as before.
  Action: none for the public API. Code that imported or patched an underscore-prefixed name through `maddening.core.graph_manager` now gets an `ImportError` or `AttributeError`; those names were never part of the API.
- **The REST guide says who the server is for, and its transaction is now the routes' only undo**: `src/maddening/api/README.md` states the audience (a trusted client on loopback, or a token-holder), what the server defends against and what it does not (a token-holder's crafted archives, resource exhaustion, pathological names; TLS), and the promise that a refused or failed request leaves the graph as it was, with its carve-outs; the REST claims rows are narrowed to it.
  The experimental `POST /surrogate/deactivate`'s 500 is the generic one (it quoted the graph's error), and `POST /sim/profile` puts the whole graph back, so an edited graph is still waiting for its compile after a profile.
  Action: none; read a failed surrogate revert's reason in the server's log.
- **A REST write that fails, at any point, leaves the graph exactly as it was**: every route that can change the graph (nodes, edges, compile, `PUT /graph/state`, `PUT /graph/params`, checkpoint save and load, `/sim/step`, each slice of `/sim/run`, `/sim/start`, `/sim/reset`, `/sim/profile`, surrogate activate and deactivate) runs in one transaction that records the graph under the lock and puts it back when the route refuses or fails unexpectedly; each route used to validate first or undo for itself, and a failure after its first write left what was written.
  An unexpected failure is now a JSON 500 with a generic detail (the traceback is in the server's log), where it was a plain-text 500; a refused `POST /sim/step` or `/checkpoint/load` of an edited graph no longer leaves it compiled; a `POST /sim/run` that fails unexpectedly after some slices answers 500 with `steps_run`.
  Action: none; not covered, and unchanged: files under the checkpoint root, the runner's thread, surrogate jobs, frames a stream already sent, `/cloud`.
- **Names and text a graph could not save again are refused where they are introduced, and every REST reply is written by one encoder that cannot fail** (MADD-ANO-208, 209, 210): `GraphManager.add_node` (so `from_dict`, a USD stage and `POST /graph/nodes`) refuses a name holding U+0000 to U+001F (tab, line feed and carriage return excepted), a surrogate or U+FFFE/U+FFFF (a NUL-named node's checkpoint saved and did not load; an FMU's model description could not be read back); `add_edge` and `add_external_input` refuse a field named `NaN`, `Infinity` or `-Infinity` (`GET /graph` was a 500 while the edge stood), or holding `#` (two mapped edges on one field pair shared one slot of weights) or those characters; `POST /graph/nodes` and `PUT /graph/params` refuse a text value spelling a token or holding a surrogate, and `/checkpoint/save` and `/load` a name with those characters.
  A refusal that echoed the text `NaN`, `Infinity` or `-Infinity` (`POST /sim/run?n_steps=NaN`, what a browser sends for an empty count) or a surrogate was a 500 (never released for the text; the surrogate since 0.1.0): every route, 422 and `HTTPException` now replies through one encoder.
  Action: rename a node, field or checkpoint whose name holds such a character; an edge to a target field its node does not declare is still taken.
- **The FMU export refuses what it used to round, skip or forget** (MADD-ANO-197, 198, 200, 201): `build_model_description(selected_outputs=...)` raises `ValueError` for a pair naming no exportable state field (it returned a description without that output); a `get` of an `Int64` / `UInt64` variable holding a value a float64 cannot hold is an error (`fmi3Error`), not a neighbouring integer; an FMU-state archive carries the drift reference, and one without it is refused by name; `load_state` raises for an integer of the other signedness.
  Action: fix the misspelt pair; take FMU states again after upgrading a development build.  The time bound is restated with the float64 clock's own rounding (three ulps of the time beyond a millionth plus a tenth of a step).
- **`to_dict()` returns a copy, and a hook naming `params` where no keyword reaches it is refused when its node is added** (MADD-ANO-205, since 0.1.0): a config shared its containers with the graph, so `cfg = gm.to_dict(); cfg["nodes"][0]["params"]["k"] = 5.0` also changed `gm` (from its next recompile; on 0.4.0 builds a three-argument node's next step) and `yaml.safe_dump` refused a three-argument node's config; and a hook naming `params` positional-only or as `*params` is refused by `add_node` and by the wrappers (`ValueError` naming the node, the hook and the fix), where its first trace raised `TypeError`: a clearer and earlier error for a node that never ran. Action: edit configs freely; write through `node.params` to change a node; declare `*, params=None`.
- **Assigning `node.params` stores a counting mapping; `SpringDamperNode` seeds its state in its constants' dtype** (MADD-ANO-175, MADD-ANO-017): writes made through `node.params` after `node.params = {...}` now reach `gm.params` (they were lost), so a reference to the assigned dict is no longer `node.params`; under x64 a spring with Python-float constants seeds float64, so its graph scans (it raised; 0.3.x ran it in float32).
  Action: write through `node.params` after assigning it; give a spring float32 constants to keep a float32 state under x64.
- **Interface-mapping factories refuse coordinates they cannot map, and every reader of an edge applies the step's rule** (MADD-ANO-192, 193, since 0.1.0): `projection_1d_mapping` / `conservative_projection_1d` returned a zero or wrong matrix for boundaries that are not strictly increasing (descending: all zeros; `[0, 2, 1, 3]`: rows summing to 4/3), nearest neighbour selected a NaN point for every target, `linear_interpolation_1d` took unsorted coordinates, and the RBF factories and `matrix_mapping` took non-finite input.  Each is now a `ValueError` naming the argument and the first offending index (`MappingRebuildError` from a config or a USD stage); nothing is sorted for you, and accepted input gives the same operator bit for bit.
  `check_conservation` and `DatasetGenerator` rebuilt boundary inputs without the interface mapping, let an additive edge overwrite the edges before it, and (dataset) applied a transform across the time axis; both, and `resolve_boundary_inputs`, now resolve through the step's own edge rule.  The diagnostic also resolves an edge that reads a flux output, reads `gm.params`, and raises for a flux the node does not report (it read 0.0), a node missing from the state, or two nodes that read each other's flux; `POST /surrogate/train` sizes a mapped input as delivered.
  Action: pass boundaries (with the field) in increasing order and finite points; re-run a conservation check, or regenerate a surrogate dataset, that involved additive edges, mapped edges or a transform that indexes its field.  Still open: the dataset pairs a forward edge's input with the step before (MADD-ANO-194, since 0.1.0), and `convergence_norm="interface"` reads a mapped edge before its mapping (MADD-ANO-195, never released).
- **Two released behaviours corrected** (MADD-ANO-160, 159, since 0.1.0): `jax.grad` through `run_adaptive_scan` read NaN or exactly 0.0 wherever an attempt's step-doubling estimates agreed exactly (a coupled group at its fixed point, a memoryless or resting node) and now reads the derivative; `compile()` warns, naming the edge, when a coupling group whose members are joined only through outside nodes reads one of them a step late.
  Action: re-run a calibration or sensitivity study that differentiated `run_adaptive_scan`; a group the new warning names should take the joining nodes in, or be split.
- **`GraphManager.add_node` refuses a timestep that is not a finite number > 0 and the names `_meta`, `_params`, `_params_mappings`; a loopback-bound API asks the token of a non-loopback peer and of a forwarded request** (MADD-ANO-179, 181, 182): each was accepted, then ran silently wrong or broke later.
  `run_pod.py` exits 2 when it refuses a run and 5 when a goal raises (both were 1, "a check failed"); `--summarise`'s statuses are unchanged.
  Action: give every node a finite timestep > 0; in-process REST clients present `server.auth.token`; read a goal's exit status by the runbook's table.
- **A bfloat16 or float16 state is measured in float32 by the coupling norms and the adaptive error norm; coupling round-6 fixes** (MADD-ANO-186 to 189): a 16-bit group's norms, floor and residual are measured, summed and held in float32 -- above 65,504 active entries a float16 group's "mixed"/"interface" residual read 0.0 (every release; on this cycle's builds it then stopped 4% from its fixed point, `converged=True`), and at the default `rtol` `inf` on a finite state -- and `strict_convergence` names a non-finite state from the state, not the estimate; the adaptive steppers' step-doubling error norm had the same defect (every release: `run_adaptive` did not return on a float16 state above 65,504 elements, and `run_adaptive_scan` raised a scan-carry `TypeError` on any 16-bit state).
  Under `convergence_norm="interface"` with an edge transform `spectral_error_bound` is taken on the transformed reading (it read 0.0014-0.098x the true distance, usable); the gradient bound's Kantorovich check adds the affine-covariant constant as an operator and carries the Newton step's miss (0.65-0.986x, usable, far from the fixed point); `rho_spectral` and the L2 `tolerance` docstrings are restated.
  Action: none; a 16-bit group's `residual` and its `_meta` slots are now float32 and it may take a different (correct) number of passes, a 16-bit state steps adaptively as its float32 twin does, and the gradient bound costs `11 + 4k + 5 n_p` JVPs and can withdraw itself where it was usable at `h` near 1/2.
- **REST refusals agree with `PUT /graph/params`, and the runner routes stay off the shared workers**: `POST /checkpoint/load` and `POST /graph/nodes` refuse a parameter value `PUT` refuses (outside its `ParamSpec` bounds, a boolean or text for a number; on a load also non-finite or refused by the constructor), `GraphManager.load_state` refuses text and booleans for a number, and `PUT` refuses a numeric string (it stored `"1.5"` as 1.5); an integer is bounded at 10^7 only for an integer parameter (an integral JSON number for a float one is a float); a token with surrounding whitespace and an `allowed_hosts` entry that is not a host name raise at construction.
  Start, stop, pause, resume and reset run on pools of their own with deadlines from arrival (`PUT /sim/stride` on the event loop); `run_pod.py` takes no option abbreviations and writes schema 7 (`requested_cells`; a synthetic grid holds at least the requested cells).
  Action: keep parameters inside their bounds; strip the token; spell `run_pod.py` options out; re-run a session recorded at schema 6.
- **A Krylov solve answers a right-hand side with a NaN or infinite entry with NaN, not zeros** (MADD-ANO-154, 155): the IFT tangent and adjoint of a coupling group read exactly 0.0, reported successful, for a NaN or `+-inf` tangent or cotangent (never released; dense and fori read NaN),
  and `sharded_cg` (since 0.3.0, mostly with `converged=True`) and `ift_linear_solve`'s CG (since 0.3.1) returned zeros, as did every GMRES path on this cycle's relative tolerance.  Every Krylov path (`_ift_linear_solve`, `ift_linear_solve`, both `sharded_*` backends, tangents and cotangents included) now answers NaN in every entry; a sharded result reads `converged=False` and a NaN `residual_norm`.
  Action: none for a finite right-hand side (its path is unchanged); a solve or derivative that read 0.0 from a non-finite input now reads NaN -- check the input with `jnp.isfinite`.
- **The surrogate-training routes and the state streams are experimental in 0.4.0** (`/surrogate/train`, `/surrogate/status`, `/surrogate/activate`, `/surrogate/deactivate`; `/ws/state`, `/ws/state/binary`, `/ws/render`; `StateRelay`): they are to be hardened in 0.5.0 and may change in any minor release until then.
  They carried no level before; the routes are listed in the stability report (`register_route_stability`) and tagged `x-maddening-stability` in `/openapi.json`. Action: pin the version if you build on them.
- **No absolute constant hides in a relative computation** (MADD-ANO-123 to 130; gate `scripts/check_numeric_constants.py`): `sharded_cg`/`sharded_gmres`/`ift_linear_solve` default `atol=None` (relative; a right-hand side near 1e-9 came back 40-100% wrong with `converged=True`), and the multi-rate GCD, Adam's `eps` and the adaptive error norm no longer depend on units.
  Coupled fields below `finfo.tiny/eps` warn once (`UnderflowRangeWarning`); the power-of-two frames are one helper (bit-identical, 125 configs x 3 jaxlibs).  `run_adaptive`'s `atol`/`dt_min` stay absolute (MADD-ANO-128).
  Action: pass `atol` only as a noise floor in `b`'s units; re-run small-unit solves, `fit` runs and nanosecond multi-rate graphs; give adaptive steppers `atol` in your state's units.
- **`compile()` warns about two `SpringDamperNode`s anchored on each other in a coupling group whose converged step grows** (MADD-ANO-098, still open; a `UserWarning` naming both nodes and the growth factor `g`, never a refusal, judged from the live `gm.params`). Action: use a smaller timestep, or keep `k*dt <= c <= (2*m - k*dt**2)/dt`.
- **`LBMPipeNode` refuses a `propeller_radius` above 1** (MADD-ANO-058): the disc reached past the pipe wall and pushed on wall cells (64 of 140 disc cells at 1.5 on a 12x12 cross-section; mean `u_x` 1.9% off), with no error.
  Action: pass a radius in `(0, 1]` (1 is the whole cross-section); a config saved with a larger one no longer reloads.
- **`run_scan` and the other loop entry points refuse a `ShardedStencilNode` step XLA miscompiles inside a loop** (MADD-ANO-068, now `partially_resolved`): a node reading a sharded static in its halo beside a `shard_info`-offset window into another array, the static copied along a mesh axis of 2+ devices,
  raises `RuntimeError` from `run_scan`, `run_scan_with_history`, `run_sweep`, `run_adaptive_scan` and `sysid.windowed_loss` (gradients included), and from every entry point in a coupling group; `step()` / `run()` keep working otherwise.
  Action: use a mesh without the axis the static is copied along, or `step()` / `run()`; check a loop you write yourself around such a node against `step()`.
- **A domain integral is neither summed nor stacked over a mesh axis a `ShardedStencilNode`'s `axis_map` leaves unused** (MADD-ANO-067):
  releases summed it there, counting every block once per device along it (2x on a `(2, 2)` mesh), and a per-shard integral there now has
  one leading axis per mesh axis that splits the grid.  Action: re-run sharded totals from such a mesh; drop the extra axis when reading per-shard values
- **`maddening.sysid.fit` and `fit_multiple_shooting` return the lowest-loss iterate they evaluated, not the last**: Adam's ~`lr`-sized step could end a run above where it started (one fit went from loss `2.7e-8` to `1.9e-4`, unreported). `FitResult.best_iteration` and `best_loss` say which iterate was returned; a run whose loss never rose is bit-identical to before, and `fit_lm` already returned its lowest iterate.
  Action: nothing for a fit that converged; where you relied on the last iterate, read `best_iteration` (it equals `len(losses)` when the last update's result was the lowest).
- **`HeatNode` refuses more of what it used to run wrongly, and `compile()` warns about an unstable coupled pair**: a rod on `grid_points` is held to `dt*alpha/min(h_L*h_R) <= 1/2` (the Fourier check skipped it; MADD-ANO-002), and a non-positive `length` or `timestep`, a negative `thermal_diffusivity`, a non-finite one of these, or `grid_points` not strictly increasing raise `ValueError` (MADD-ANO-062).
  Two uniform rods coupled end to end through `extract_first`/`extract_last` in a coupling group, past their pair limit (3/8 at `stencil_order=2`, 0.226 at 4), get a `UserWarning` naming both rods and MADD-ANO-050; it is a warning, not a refusal.
  Action: give a refused rod a stable timestep or meaningful constants; for a warned pair use a smaller timestep or exchange the data without a coupling group, and check a pair the warning cannot recognise yourself.
- **`compile()` refuses a group-internal flux edge the coupling loop reads from the state** -- under `convergence_norm="interface"`, and into a sub-cycled member under `boundary_interpolation="linear"` / `"quadratic"` (a bare `KeyError` inside the step since 0.1.0, MADD-ANO-060) -- and an IQN `accelerated_fields` naming no floating field; a non-floating field it names is dropped.
  Action: use the `"mixed"` / `"l2"` norm or `"constant"` interpolation the message names; name a floating field.
- **`scripts/report_test_durations.py` warns when an allowlisted test passes the 20 s hard line** (the allowlist has no ceiling), and a `[cold-ci]` request whose head commit cannot be read now runs cold with a warning instead of silently warm.
  Action: none; if a kept test's warning appears, re-check its allowlist reason. `docs/developer_guide/testing_standards.md` now lists the framework properties that only the slow lane checks.
- **`scripts/report_test_durations.py` labels each shard's compilation cache from the hits its report records**: `warm` only when a restored cache served most lookups, else `restored but unused (cold)`, with every shard's hit rate; it warns when a `--cache-mode off` run records cache lookups in more than one file, and no longer offers a skipped or failed allowlisted test as removable.
  Action: none; compare times by a shard's label, not by whether it restored a cache. Pull requests that add or remove a slow mark, remove a test or edit the allowlist now run CI cold.
- **`compile()` refuses a sub-cycled node whose timestep does not divide its group's largest** (to 1e-9 relative), which covered `round(macro/node_dt) * node_dt` per macro step and drifted silently (MADD-ANO-046). Action: give it a dividing timestep; the error names the two nearest.
- **`windowed_loss(mask_unconverged=True)` refuses a coupling group with no convergence slot** (`solver="fori"` with `diagnostics=False`), which the mask silently never masked. Action: set `diagnostics=True` on the group, or use `solver="ift"`.
- **New sharding refusals of silently-wrong input**: `ShardedUnstructuredNode` refuses a node with a non-empty `halo_width()` and one whose per-cell state is not one row per layout cell, and `partition_value` a value of the wrong length (MADD-ANO-037, -039);
  `ShardedStencilNode` refuses an empty `axis_map` (MADD-ANO-042); `halo_exchange` refuses a `boundary` dict key naming no exchanged mesh axis (MADD-ANO-041).
  Action: shard a stencil node with `ShardedStencilNode` (only a pointwise node goes to the unstructured wrapper for an indivisible grid); build the layout from the node's cells; map a mesh axis; fix the misspelt axis name.
- **`GraphManager.from_dict` warns when it rebuilds a sharded node unsharded**: a config carries no device mesh, so the node comes back as the node it wraps; the `UserWarning` names the wrapper, the settings the config recorded and the `replace_node` call that wraps it again (MADD-ANO-036, silent since 0.2.0).
  Action: re-wrap after loading when the run must be sharded; filter the warning when an unsharded reload is what you want.
- **New refusals where a sharded wrapper or `PUT /graph/params` accepted silently-wrong input**: the route refuses a value that changes a node's state shape (`n_cells`), a structural value the node's constructor refuses, `LBMPipeNode`'s geometry and a write no probe copy can decide; `ShardedUnstructuredNode` refuses a per-cell input on a full partition not in global order, and a state not in partition layout;
  `ShardedStencilNode` refuses an outer `boundary` other than a wrapped `ShardedStencilNode`'s, and a state its node no longer builds; `ShardedPointwiseNode` refuses a node with no state field on the shard axis.
  Action: rebuild a node to change such a value; renumber cells with `np.argsort(partition_assignment, kind="stable")`; pass the inner wrapper's `boundary`; shard an axis the state has, or leave the node unwrapped.
- **`ShardedStencilNode` refuses `boundary="zero"` and `"periodic"` for a `HeatNode`**: the rod now builds its own end ghosts, so the fill would be ignored.
  Action: drop the argument and hold an end at 0 with `left_temperature=0.0` / `right_temperature=0.0` (in a graph, `gm.add_external_input(...)`), exactly as unsharded.
- **`ShardedStencilNode` refuses a per-axis `boundary` dict**: it was accepted but never worked in the wrapper (it zero-filled the halos of axes
  `axis_map` leaves unsharded). Action: pass one mode as a string; per-axis modes remain available on `halo_exchange` itself.
- **Coupling group keys are also refused when one plus `_total` spells another's** (a new `_meta` slot, `<key>_total_iterations`): groups keyed
  `a+b+c` and `a+b+c_total` now fail at `add_coupling_group`, with or without sub-cycling. Rename a node so the keys differ.
- **`profile_graph` reports `coupling_overhead_ms` signed, alongside a new
  `coupling_overhead_se_ms`**: it was clamped at zero, which biased it upward
  and printed `0.00 ms` for an overhead the run could not resolve
- **Python floor is now `>=3.12`** (0.3.1 shipped `>=3.10`): jax 0.11 requires
  3.12 and there is no 0.12, so a 3.11 floor made half of `jax>=0.10,<0.13`
  uninstallable. CI now runs 3.12 against both jax 0.10.2 and 0.11.2
- **Docs**: the SOUP identification table gives JAX's *verified* version as well
  as its permitted range; `DESIGN.md`, `DatasetGenerator.from_graph` and
  `SurrogateValidator.compare_graphs` say they advance the graph they are given
- **Renderer and trainer config dicts are PEP 589 `TypedDict`s**: a misspelled
  key or a wrong value type is now a type error, and a scene object whose `"x"`
  names a state field no longer kills `setup()` with a `ConversionError`
- **The six `step` / `run_*` entry points document that they advance the
  graph's own state**: a second call continues from the first's final state,
  so measure from a fresh `GraphManager`; `run_sweep`, which does not, says so
- **`fit_lm` and `fit_multiple_shooting` hold the undetermined directions too**,
  as `fit` already did, so all three fill `FitResult.excited_rank`; the spring's
  `(k, c, m)` scale drifted 0.43-4.8% before. `hold_undetermined=False` opts out
- **`fit` holds the directions the data cannot determine at the values it was
  given**, so a degenerate combination no longer lands wherever `n_iter` and
  `lr` leave it; `FitResult.excited_rank`, or `hold_undetermined=False`
- **`soup_package.md` §3 counts *reachable* defects, not `open` tickets**: it
  counted only `open` and left every `partially_resolved` entry out; one with a
  live residual risk now counts, by the version-range gate's own predicate
- **`fim`'s rank cutoff now sees the residual length**: `rank_rtol` defaults to
  `max(n, sqrt(m)) * eps`, not `n * eps`.  `rank` falls and `crb` goes `+inf`
  for some long-residual (`m > n**2`) fits; pass `rank_rtol=n * eps` to opt out
- **`coupling_diagnostics()` renames `bound_valid` to `ratio_usable` and
  `gradient_error_bound` to `gradient_error_estimate`** — the flag reports one
  of the four conditions the estimate rests on, not that it is a bound
- **`FitResult` is keyword-only**, the guard `FIMReport` got this release:
  no field has been inserted into it yet, and inserting one would silently
  swap `converged` and `n_iter` for any positional caller
- **Breaking:** `FMIVariable` is keyword-only (0.4.0 inserted `node` / `field`
  between `unit` and `shape`, so a positional call silently bound the wrong
  fields) and `load_graph_from_usd` gained `node_registry=` / `allow_import=`
- **`FIMReport` is keyword-only**: `rank` was inserted mid-dataclass this
  release, so positional construction silently reassigned every field after it;
  build it with keywords (every in-tree caller already did)
- **`AdaptiveNode` tightens its subclass contract and loosens its gate**:
  `compute_active_set` must return a non-empty **boolean** mask; `is_trapped_at`
  is now `frozen_gradient_vanishes_at`; `on_blind="warn"` no longer ever raises
- **`converged=True` means "within `tolerance` of the fixed point"**, not "the
  last step was small": the threshold is tested against `residual / (1 - rho)`
  and every norm is now relative, so expect more iterations and retune `atol`
- **Extended-precision point sets are refused, never silently narrowed**: a
  `MappingSpec` reference of `float128` / `np.longdouble` (unwritable as JSON,
  unstable to hash) raises; pass `np.asarray(points, dtype=np.float64)` instead
- **`solver="ift"` returns the iterate whose residual met the criterion**, as
  `fori` always has, so `converged=True` names the state you were handed and
  both solvers return it; every converged group's answer moves by one residual
- **Recorded `aitken` and `iqn-*` trajectories move**: Aitken's first pass of a
  timestep relaxes with the `omega` it was seeded with, not the clip floor 0.01
- **`coupling_diagnostics()["residual"]` describes the state the step returned**
  under either `solver`, agreeing between them only to its float32 noise floor
  (see Fixed); a group that arrives on its last pass now
  reports `converged=True` instead of raising under `strict_convergence`
- **The interactive path stops redoing host work**: sharded wrappers place
  their static arrays on device once, not per `update`, and `run_scan` and its
  siblings compile once per `compile()`, not per call (`gm.scan_trace_count`)
- **`lineax` is a base dependency**, not the `[ift]` extra: a coupling group
  at its default settings could not be differentiated on a base install.
  `pip install maddening` is enough; the now-empty `[ift]` extra still resolves
- **Version is now `0.4.0.dev0`** (was `0.3.1`) so a development build is
  distinguishable from the last release.  `maddening.__version__` prefers
  installed distribution metadata, so an editable install predating this
  keeps reporting the old version until reinstalled.
- **Coupling groups default to `solver="ift"`**, a `while_loop` that exits on
  convergence instead of always running `max_iterations`.  Set `solver="fori"`
  to keep the old behaviour
- **The IFT derivative rule is a `jax.custom_jvp`** (was `custom_vjp`), so
  `jax.jvp` / `jacfwd` and the FMI `FORWARD` directional derivative now work
  through coupled steps
- Node constants are no longer constant-folded, so `run_sweep` and an
  individual `run_scan` can differ by ~1 ulp where they were bit-identical.
  Loosen any bitwise assertion between the two
- **Resume-from-URL transport moved to `maddening.cloud.resume`**
  (`maddening.cloud.download_and_load_state`); the old path forwards and
  warns.  Behaviour, signature and errors are unchanged
- `AdaptiveNode` and `AdaptiveNodeBlindnessError` are `@stability(EVOLVING)`,
  not `STABLE`, and `ift_linear_solve` returns to `EXPERIMENTAL`; the 0.4.0
  freeze picks the final levels
- `AdaptiveNode.blindness_ratio` / `blindness_threshold` are
  `gradient_capture_ratio` / `gradient_capture_threshold`, and the cold-start
  check warns rather than raising unless `on_blind="raise"`.  Like `is_trapped_at`
  and the `bound_valid` / `gradient_error_bound` keys above, these are renames
  within the 0.4.0 cycle: no release carried the old names, which still warn
- The `AdaptiveNode` gradient is documented as exact within an active-set
  region and first-order wrong across a switch; the previous "Clarke
  subgradient" claim was false (`MADD-ANO-003`)
- `AdaptiveNode.n_max` is structural and no longer in `params`, while
  `blindness_gate`, `on_blind` and the diagnostic constants are; a subclass
  whose basis size must round trip declares its own integer parameter.
  `compute_active_set` and `solve_frozen` are `@abstractmethod`, and the
  constructor validates `n_max`, the diagnostic constants and `dtype`
- `iqn_ils_update` takes a keyword-only `have_prev` flag saying whether the
  previous-iterate arguments are real
- Hypothesis is configured once in the root `tests/conftest.py`: `dev`/`ci`
  profiles selected by `MADDENING_HYPOTHESIS_PROFILE`, three named depth
  tiers, a persisted example database, and a `max_examples` house rule in
  `docs/developer_guide/testing_standards.md`

### Deprecated
- `maddening.core.simulation.calibration.calibrate` and
  `tune_coupling_params` warn and are removed in 0.5.0; use
  `maddening.sysid.fit`, which has `ParamSpec` bounds and a trainable mask
- `CouplingGroup.solver="fori"` emits `DeprecationWarning`; removed in the
  next minor release
- `maddening.core.simulation.checkpoint.download_and_load_state` warns and is
  removed in 1.0; use `maddening.cloud.download_and_load_state`

### Removed
- The stelling formal-verification suite, CI job and `stelling` dependency.
  The `[verify]` extra now only pulls `hypothesis`.

### Fixed
- **`matrix_mapping` and `StaticLinearMapping` say that `mode` is a label**: `matrix_mapping(H, mode="conservative")` applies `H` as given (it neither transposes nor rescales a matrix you bring); the sparse forms said so, the dense ones did not. Docstrings and the mapping guide's table only.
- **An FMU's description carries `unit` on float variables only** (MADD-ANO-273, in 0.3.0 and 0.3.1 on a hand-built variable): `unit` was written on every variable element, and FMI 3.0's schema has it on `Float32` and `Float64` alone, so an `int32` input into a field whose node declares `expected_units`, or an integer or Boolean parameter whose `ParamSpec` names units, gave a description FMPy refuses (`Element 'Int32', attribute 'unit': The attribute 'unit' is not allowed`). The attribute and its `<UnitDefinitions>` entry are now written for the two float types; `FMIVariable.unit` keeps what the graph declares. Nothing to do.
- **A 422 no longer sends a refused request body back** (MADD-ANO-274, since 0.1.0): a validation error's `input` is echoed when it is at most 1024 characters of JSON and is otherwise left out, with `input_omitted` in its place saying so, and a reply lists at most 64 errors. A `PUT /graph/params` just under the 32 MiB limit was answered after 22 s with a 422 of 32 MiB; it is now under 1 kB after 13 s (refusing such a body still peaks at 2.4 to 2.6 GiB: the entry's residual). A client that read `input` from a 422 for a large value finds `input_omitted` there.
- **An FMU restores its own snapshot when a coupling group's report reads `inf`** (never released): `FmuSidecar.set_fmu_state` and the bridge's `set_state` refused it (`_meta.coupling_a+b_gradient_relative_error_bound: value must be finite`) for a `diagnostics=True` group stopped at its iteration cap, and on every step of a group under the experimental geometry diagnostics whose positions are constants of the pass. A group's report slots now restore with the non-finite readings they hold; a non-finite value in a node's field or in a group's warm starts is refused as before. Nothing to do.
- **A `node.params` assignment on a sharded or hybrid node is the write the step runs** (MADD-ANO-278, in 0.1.0 to 0.3.1): `node.params = {**node.params, "k": v}` on the node `gm.get_node(name)` returns for a sharded graph (or on a `HybridNode`) replaced the wrapper's mapping alone, so the step ran the old value while `node.params` and the saved config showed the new one, and every later keyed write through the wrapper was lost. A wrapper now has no mapping of its own: its `params` is the wrapped node's, whichever of the two you write or assign. Nothing to change in your code.
- **`sharded_cg` / `sharded_gmres` do not read `converged=True` on a stop that lineax's `0 / 0` caused** (MADD-ANO-281, in 0.3.0 and 0.3.1): with an explicit `atol=0.0` and a right-hand side that has zero entries, the default backend returned after one step with `converged=True` at a residual of 0.4 `||b||`. Where lineax's floor is zero (`atol=0.0`, `rtol=0`) `converged` is now `residual_norm <= max(atol, rtol * ||b||)`; the value is still lineax's early stop, so prefer `atol=None` (the default) or `backend="loop"`.
- **`sharded_cg` / `sharded_gmres` do not vouch for a value under a tolerance that is not one** (MADD-ANO-271, in 0.3.0 and 0.3.1): a negative or NaN `atol` or `rtol` that reached the default backend read `converged=True` beside a value 72 % off, or zeros. Traced, it now returns NaN with `converged=False`; concrete, it raises (`### Changed`).
- **`ift_linear_solve` does not return a value that is not the solution where its stopping floor is zero** (MADD-ANO-283, in 0.3.1): with `atol=0.0`, or a NaN `atol` or `rtol` (in 0.4.0's development builds also `rtol=0`, an underflowing or a negative tolerance), lineax's CG and GMRES stopped on a `0 / 0` and the call returned a value 0.36 to 1.0 off with no error, its derivatives the same. Such a solve is now judged by `||rhs - A x|| <= max(atol, rtol * ||rhs||)` and raises where that fails; prefer `atol=None`.
- **The cloud container streams its simulation** (MADD-ANO-287, in 0.1.0 to 0.3.1): `cloud/entrypoint.py` built a `SelkiesSession` and never started it, and served no node types, so the container streamed nothing and its API could not build a graph. It now registers the built-in nodes (`maddening.nodes`) and starts the session with `CloudConfig.stream_config` on port 8443 of `MADDENING_HOST`, authenticated against `MADDENING_STREAM_SECRET` as before, fed with the simulation's state drawn as a time series at the stream's frame rate; the session id is logged for the viewer's token. Build the graph over REST (`MADDENING_GRAPH_USD` is still not loaded).
- **WebRTC streaming connects: `SelkiesSession` answers the viewer's offer** (MADD-ANO-286, in 0.1.0 to 0.3.1): the signaling server authenticated a viewer and then sent it nothing, so no video ever reached a viewer. It now answers the offer with the H.264 payload type the viewer named, trickles ICE candidates both ways, keeps SDP and ICE away from the input handler, lets a newer viewer replace the current one, and `stop()` ends its threads. A GStreamer WebRTC peer decodes its frames on CPU in the new `test-streaming` CI job (`docker/Dockerfile.streaming-test`; `docker/README.md` has the local commands); a GPU encoder, a real browser and NAT traversal are not tested.
- **`sharded_cg` answers an operator that returns NaN or infinity with NaN, not with zeros** (MADD-ANO-269, in 0.3.0 and 0.3.1): the default call and `backend="lineax"` returned zeros with `converged=True` and `iters=0`. Every route of `sharded_cg` and `sharded_gmres` now returns NaN in every entry, a NaN `residual_norm` and `converged=False`, as for a non-finite right-hand side, and `converged` is never True beside a `residual_norm` that is not finite. Nothing to do.
- **`sharded_cg`'s default call answers in float32 at scale** (MADD-ANO-264, in 0.3.0 and 0.3.1): without a preconditioner the default backend is lineax's CG, which returned NaN in every entry with `converged=False` above about 1e5 unknowns on a system of order-one scale, and for a smooth right-hand side from about a thousand. The default call now solves such a system with the loop backend and returns that answer, with its true residual; where lineax answered, the result is bit for bit what it was. Nothing to do (on 0.3.x: pass `backend="loop"`). `backend="lineax"` asked for by name still returns the NaN, and `ift_linear_solve(solver="cg")` still raises there.
- **`sharded_cg` and `sharded_gmres` report the residual of the system on the loop backend** (MADD-ANO-263, in 0.3.0 and 0.3.1): `converged` and `residual_norm` came from the residual the CG loop updates by recurrence (on GMRES behind a preconditioner, from the preconditioned residual), so a float32 solve read `converged=True` with a norm under `rtol` while `||b - A x||` was 10 to 100 times the tolerance. Both now come from one extra product on the returned value. The solution and the iteration count are unchanged; see `### Changed` for what code that reads the flag does.
- **The shape `compile()` holds an edge's source to is asked wherever a step is traced** (MADD-ANO-250, every release since 0.1.0): after `compile()`, a field written with another shape (`set_node_state` with one entry for three) was broadcast into the node that reads it by `step()`, `run_scan()` and every other stepper, silently. They now raise the `ExceptionGroup` of `ShapeMismatchError` that `compile()` raises for that state, and a source field dropped from the state is a `KeyError` naming the edge; write fields with the shapes `initial_state()` gives them. The dtype rule is still asked by `compile()` and `validate()` only (`set_node_state`'s docstring says what a write is asked).
- **A node's `delta_t` written after `add_node` is an `ERROR` of `validate()`** (MADD-ANO-251, every release since 0.1.0): the graph kept stepping at the timestep it registered while `to_dict()` wrote the new value, so the saved config ran another step. `compile()` now refuses such a graph, and `to_dict()` writes the timestep the graph runs; construct the node with the timestep it should have.
- **A coupling group under `acceleration="iqn-ils"` or `"iqn-imvj"` returns from `step()` when its iterate leaves float range** (MADD-ANO-233; 0.3.0 and 0.3.1 under `solver="ift"` with `"iqn-imvj"`): the secant least-squares handed LAPACK's SVD a matrix holding an `inf`, and the step never returned. It now runs to `max_iterations` and reports `converged=False`, `residual=inf`, as the other accelerations do; steps of a finite run are unchanged, bit for bit. Action: none; on 0.3.x use `solver="fori"` or `"aitken"`.
- **`GraphManager.step`, `run` and `run_adaptive` no longer store a step that reshapes a state leaf** (`MADD-ANO-220`, in 0.1.0 to 0.3.1): a list given for a scalar constant (`BallNode(initial_velocity=[1.0, 2.0])`), or an external input of another shape at a later step, broadcast a scalar leaf, and the state's checkpoint then did not load after `reset_state()`. The step is refused by name (compared once per trace, host-side: no compiled program changes), as `run_scan` always did; the FMU sidecar and bridge refuse it too.
- **`POST /checkpoint/load` takes a checkpoint saved before an initial condition was written** (never released): after `PUT /graph/params {initial_*: v}` (or `TableNode.position`) every earlier checkpoint was a 400 "does not fit this graph", for six of the sixteen numeric parameters of the stock nodes. The load now installs each leaf it changes as `PUT` does (in the node too, so the next reset builds the state from it), answers as `PUT` of the same value on every parameter of every built-in node, and its 400 names the checkpoint's value and the route that fixes it.
- **`run_pod.py --summarise` reads a goal file with no cells or no checks INVALID (exit 3) for every goal** (never released): an emptied `forward.json`, `gradient.json` or `exchange.json` read "no checks" and the summary exited 0.
- **A constructor's refusal at `POST /graph/nodes` names the parameter** it can be told from, the value sent and the class's default (`n_cells: 8.0` was "'float' object cannot be interpreted as an integer").
- **An FMU parameter whose `ParamSpec` accepts no value advertises an empty `min` / `max`** (never released; the spec-level refusals are 0.4.0's own): a `logit` spec with a subnormal float32 bound, or a `log` / `logit` bound or width the leaf's dtype does not hold, is refused by `check_params`, the sidecar and REST for every value, while `build_model_description` advertised the neighbours of its bounds (`min = 1.18e-38`, `max` just under `1.42e-14` for `logit` on `(1.4e-45, 1.42e-14)`), so a bridge over a sidecar built without `param_specs` took every value between them. The variable now advertises `min = tiny > max = -tiny`, and the bridge refuses to serve the description, with or without specs. Action: none; narrow the bounds as the refusal says.
- **`POST /graph/nodes` accepts only a node the graph can step with its state as built** (the REST door of `MADD-ANO-220`; in process the step itself refuses such a node, see the entry above): one update is traced as the graph calls it (params pytree, with and without boundary inputs) and held to `initial_state()`'s fields, shapes and kinds of dtype; `venous_pressure: null` and a list for a scalar constant are a 400 naming the parameter. Also: lists and objects count against the 10^6 params bound; `POST /checkpoint/load` names why a graph cannot compile; `POST /surrogate/deactivate` (experimental) lists the edges it drops (`dropped_edges`).
- **`run_pod.py` refuses an option value no goal can use with exit 2, before the backend loads** (never released): `--cells 0`, `--warmup -1`, `--repeats 0`, `--steps 0`, a `--mesh` that cannot be read. They used to pass, or exit 5 as a crashed goal.
- **FMU export, round-9 audit fixes** (MADD-ANO-197 to 201; 201 since 0.3.0, the others never released): an FMU-state restore resumes the bridge's drift count instead of starting it again, so a master that saves and restores between steps is held to the time bound; an `Int64` / `UInt64` a float64 cannot hold is refused on `get` and `set` instead of rounded (`fmi3GetInt64` answered `fmi3OK` with a neighbour), and integer checks no longer compare through float64; negative zero keeps its sign on the wrapper's JSON path;
  `fmi3GetFMUState` reuses the state it is handed (it leaked one whole state per call) and `fmi3FreeInstance` frees its live states; `load_state` refuses an integer of the other signedness (`-1` for a `uint32` field loaded as 4294967295); `selected_outputs` refuses a pair that exports nothing; the instantiation token covers a variable's unit.
  Action: re-package an FMU built by an earlier development build (every token changes); free an FMU state before its instance; carry a 64-bit integer above 2^53 through the FMU-state functions.  FMU-059 to FMU-062 are new.
- **Checkpoints and bfloat16** (MADD-ANO-207, never released; MADD-ANO-206, open): a checkpoint value loaded into a bfloat16 leaf is held to the cast check every other dtype gets (a float32 `1e-44` loaded as `0.0`, `3.4028235e38` as `inf`). A bfloat16 leaf still cannot be restored from its own graph's checkpoint (`load_state` raises, every release); hold such a state in float16 or float32 until 0.5.0.
- **Apps built at once in several threads** (MADD-ANO-191, since 0.1.0): `SimulationServer.create_app()` builds one app at a time. FastAPI builds each route inside `warnings.catch_warnings()`, which is not thread-safe, so two builds at once could raise a warning FastAPI silences or leave `ignore::UserWarning` in the process's filters for good, dropping every MADDENING warning after it.
  Action: none on 0.4.0; before it, build the apps one after another.  An app built while another thread is inside a `catch_warnings` block of its own is not covered (the entry's residual risk).
- **The known coupling, sysid and FMU findings** (MADD-ANO-158 to 162, CPL-087, SYS-024, SYS-071): `jax.grad` / `jax.jvp` through a bfloat16 or float16 group under `solver="ift"` works (the adjoint solve runs in float32); a 16-bit group's spectral slots are float32; `strict_convergence` on a step spanning several devices raises instead of aborting the process; a typed PRNG key steps under `"ift"` with Aitken or fixed relaxation;
  the precision warning names float64 leaves when x64 is already on; `best_loss` and `losses` are the loss of exactly the parameters a fit returns (an untouched `log` leaf was evaluated at its round trip, an ulp away).
  Action: none; a 16-bit group's gradient is to its dtype's resolution.  MADD-ANO-156 and 157 (flux edges across a group's boundary or staggered by an ungrouped cycle) stay open for 0.5.0.
- **REST round-8 audit fixes** (MADD-ANO-178, 180 to 185): a params write back to a node's own value is asked the combined checks (178); `PUT /graph/state` refuses text, booleans and integers its field cannot hold (180, since 0.1.0); zero, negative or non-finite timesteps (181), a node named `_meta` (182), `/sim/start` of an empty graph (183) are refused; a 422 holding NaN is no 500 (184); a checkpoint load checks member headers before reading (185).
  Also: a non-ASCII `Host` port is a 403, not a 500; a token outside printable ASCII is refused at start-up; a comma-joined `Sec-WebSocket-Protocol` carries the token; `run_pod.py --summarise` reads any goal file's shape as INVALID and counts a file with no checks.
  Action: write state in its field's dtype; choose a printable ASCII token; re-run a summary that a mistyped file used to stop.
- **REST round-7 audit fixes** (MADD-ANO-163 to 166): `POST /checkpoint/load` no longer restores values `PUT /graph/params` refuses (163); a write to a `HybridNode`'s params reaches its step (164, lost since 0.1.0); a `/sim/run` that cannot have the graph says how many steps it took, and its final read is bounded (165); `/sim/profile` answers 400, not 500, for a graph that cannot step (166, since 0.2.0); the stop/reset 503 says the runner stays stopped; long checkpoint names save; `APIAuth(environ=)` reads the token file's path.
  `run_pod.py`: the "1e5-cell" exchange row decides (it measured 99 856 cells and never did), `--summarise` lists every row that does not decide and re-derives the speedup, a goal-named file of another goal reads INVALID, and `--dry` is refused (it used the GPUs).  Action: re-apply any `PUT` to a `HybridNode` made before 0.4.0.
- **Coupling round-5 audit fixes** (MADD-ANO-142 to 146; new, resolved): the gradient bound applies the exact resolvent to each secant (it read 5.2x below the true error, usable), probes each entry of an array constant, and certifies Kantorovich with the full resolvent norm; the spectral bound is in the returned state's weights;
  a group is one block in the schedule (an outside node between its members read it a step late, since 0.1.0) and compile() warns when a group is part of a larger loop; CouplingGroup refuses out-of-range counts and thresholds (waveform_iterations=0 froze a sub-cycling group, since 0.1.0).
  Action: re-read diagnostics=True bounds; expect a UserWarning for a group inside a larger feedback loop; fix any out-of-range CouplingGroup knob.
- **Checkpoints, the token file and the stride say what they did** (MADD-ANO-138 to 141): `load_state` refuses a value its dtype cannot hold (a float64 `1e39` loaded as `inf` since 0.1.0); `PUT /sim/stride` keeps an omitted value (it reset it to 1, since 0.1.0); the token file is a new `0600` file; a refused save writes nothing and no checkpoint 4xx names a server path;
  a JAX trace's time budget has its own timer; `run_pod.py --summarise` reads an unparseable goal file INVALID (exit 3).  Action: load a checkpoint into a graph of the precision that wrote it; send both stride values only if you relied on the reset.
- **REST round-6 audit fixes** (MADD-ANO-131 to 134; 096 completed): a surrogate job re-checks its memory budget on the graph it sweeps; `POST /sim/profile` restores the live state and keeps the streams out; a reset, a state write, a node edit and a surrogate swap publish to the streams (the relay adds `run_adaptive`'s `dt`); a JAX trace stops itself at 10 000 steps or 600 s;
  `/sim/run` reports `steps_run` when a step raises; runner routes answer within one lock timeout and say when they left the runner stopped; checkpoint saves are atomic; past the stream cap a client gets 1013, not 403; `run_pod.py` records only a real commit and `--summarise` survives older files.
  Action: none; read `GET /sim/profile/jax/status` if a long trace ends early.
- **A node reading a cycle runs after it in the same step, whatever order it was added in** (MADD-ANO-120, since 0.1.0: added before a coupling group's members, it read their previous-step output); the interface norm sums its edges in the group's sweep order, not insertion order; `solver="fori"` docs: forward mode works.
  Eight coupling claims now state their true conditions (`docs/validation/coupling_claims.yaml`).  Action: rerun a 0.3.x graph whose `gm.schedule` lists a node ahead of a cycle it reads; its results change.
- **Coupling round-4 audit fixes** (MADD-ANO-113 to 118, new, resolved; 094 extended): the IFT tangent/adjoint is scale-free (it was exactly 0 for an rhs below ~1e-8, since 0.3.0); the float floor weights every read by its measured gain (a squaring ring read its bounds at 0.02-0.2x the truth, flags set) and the report reads the step's count, not the live graph;
  accelerators and the report's residuals, secant and tangents no longer flush below ~1e-31 (converged=True at 7x the threshold; a usable gradient bound of 0.0); fori+Aitken/IQN on 16-bit groups traces; per-node profiling feeds declared inputs; both adaptive steppers share one acceptance rule (`adaptive.step_decision`).
  Action: recompute IFT gradients of small-unit groups or near-minimum losses; re-read `diagnostics=True` bounds; `run_adaptive` now retries a failed attempt at `dt_min` instead of accepting it.
- **The REST server uses its graph one request at a time, and bounds what a request can take** (MADD-ANO-106 to 112; 086 and 092 completed): concurrent `/sim/step`s no longer lose steps, `/surrogate/train` leaves the live simulation alone (one job, memory estimated first), streams follow a checkpoint load and a replaced node, `/sim/run` stops on shutdown (503),
  a dead runner is reported, structural edits beside the runner are 409, bodies over 32 MiB are 413, the whole graph holds at most 1e8 state elements; `run_pod.py --keep-going` records a goal that raises, and `--summarise` exits 4 when no file records a commit.
  Action: expect 409 for writes during a `/sim/run` or beside the runner, and read `/checkpoint/load`'s `sim_time`; a 503 from `/sim/run` names the steps it took.
- **Fitters keep each parameter where `constrain` is not a clip or a clamp** (MADD-ANO-104, never released: a coordinate past one stayed there and `fit_lm` said converged); `fit_lm`'s floor is relative to the residual; `windowed_loss(start_step=)` (experimental) says where a multi-rate record began;
  `node.params` writes are counted, so every entry point recompiles for one and `load_state` supersedes them (MADD-ANO-105, since 0.1.0: `gm.step` and a new `run_scan` length ran different models).  Action: after writing `node.params` on a 0.3.x graph, `compile()` before running it.
- **`gm.timestep` is the step a graph takes** (MADD-ANO-096/097, since 0.1.0): a sub-cycling group counts at its largest member timestep, so the runner's and relays' clocks, USD `baseDt` and the FMU default step (all now from it) no longer run slow; `SpringDamperNode` states its coupled-pair limit `c >= k*dt` (MADD-ANO-098, open);
  MADD-ANO-099 registers the partial `external_inputs` v0.1.0-v0.3.1 did not zero-fill.  Action: on a sub-cycled graph, recompute any step count taken as `duration / gm.timestep`; keep `damping >= stiffness*dt` on coupled spring pairs.
- **`run_pod.py`'s wrapper goals see a fault confined to one spatial axis on four devices** (schema 6): each axis is split over all four on a
  1-D mesh of its own, beside a 1 x 4 two-axis mesh and the 2 x 2 pencil, on non-square grids whose inputs differ block to block (three seeded
  faults closed all six items).  `--summarise` names each item's commit, and exits 4 with `MIXED COMMITS` across commits.  Action: re-run dry runs.
- **Coupling round-3 audit fixes** (MADD-ANO-094/095, new, resolved): a Gauss-Seidel pass's float floor counts its longest chain of same-pass reads (a stalled 32-relay ring's spectral and gradient bounds read 0.5x the truth, flags set); after `run_adaptive*` the report covers both kept half steps;
  Aitken and IQN repeat a run under one common power-of-two factor on every field, exactly (not under a change of one field's units: MADD-ANO-285); `strict_convergence`'s threshold is pinned; 16-bit groups count passes in int32 and run `diagnostics=True`; x64 float32 groups step with every accelerator; the profiler's per-pass cost divides by `total_iterations` less one per sweep.
  Action: none for states (bit-identical); re-read `spectral_error_bound`/`precision_limited` of Gauss-Seidel groups and post-adaptive reports.
- **REST, round-4 audit** (MADD-ANO-086 to 092): `POST /graph/nodes`, `PUT /graph/params`, `from_dict` and USD loading check the grid/basis nodes' size estimates before building (one `PUT n_levels=10000000` grew the server to 57.7 GB); `POST /sim/stop`
  never answers "stopped" while the runner's thread steps (503; state writes 409 while it runs); `PUT /graph/params` refuses values the step cannot trace and keeps a parameter's numeric type; the WebSocket
  streams end with their client (SIGINT hung); `PUT /sim/stride` is bounded (422); a missing edge's DELETE is a 404; root-equal checkpoint paths are a 400.  Action: send integers for integer params; stop the runner before writing state.
- **The shipped examples run, and print only what they measure** (49 examples; each now runs in CI or is excluded with a reason, see `examples/README.md`): two used the deprecated `RigidBody2DNode`, one raced its own server; the coupled-spring demos had no rest state yet reported "settled" (rest lengths now `+L`/`-L`);
  `vessel_flow_server`'s parameter endpoints reported success without changing the run (they write `gm.params` now); others printed claims their numbers contradicted, deleted the results they announced, or ignored `--gpu`.  Action: none, unless you copied an example -- re-copy it.
- **Coupling harness findings** (MADD-ANO-070 to 074, new, resolved): a group's integer/boolean fields are those the pass gives at the returned state, on both solvers (`ift` returned first-pass flags and froze edge-carried ones); predictor + mixed norm with such a field, Jacobi with a flux-reading producer, `reset_state`/`set_node_state` after `jax.grad`, under-relaxed `fixed` stopping on its first pass (~2x short) and the adaptive norm on integer leaves all fixed.
  `boundary_interpolation`'s "bit-identical under Jacobi" is round-off; a near-1 rate's noise-rejected ratio falls back to the raw test (MADD-ANO-005).  Action: re-run `run_adaptive*` results from graphs holding integer leaves.
- **Sharding, from the differential harness**: `ShardedUnstructuredNode` refuses a per-cell input neither in partition layout nor broadcastable
  (a slab-length one was read by every shard as its slab; MADD-ANO-064); `gather_global` passes an integral listed in `state_fields()` through
  (MADD-ANO-065); a nested stencil wrapper starts a per-shard integral stacked once (MADD-ANO-066); zero-ghost reverse scans no longer segfault jaxlib 0.11.2
- **State and IO, from the differential harness** (MADD-ANO-063, 049): a write that moves the points a mapped edge was built from (a uniform `HeatNode`'s `length` under a mapping on its `grid_x`) is refused -- `PUT /graph/params` 400, `gm.params` at the next run, `to_dict` and `save_state`, `POST /checkpoint/load` undone, an FMU parameter fixed -- where it ran on the old mapping weights and saved a config that did not load;
  state replies, `GET /graph/params` and `/ws/state` write a non-finite float as its `json_codec` token (a `diagnostics=True` group's NaN seeds made `POST /sim/reset`, `GET /graph/state` and `POST /checkpoint/load` a 500 after applying); `PUT /graph/params` and `/graph/state` refuse a float32-overflowing value before the cast (a 500 under `-W error`); the FMU bridge and sidecar restore their own snapshot when a parameter started outside its bounds or a state field started non-finite (a diagnostics group's NaN `_meta` seeds).
  Action: read `"NaN"` / `"Infinity"` / `"-Infinity"` in a state reply (`json_codec.loads` decodes them); to change such a geometry parameter, rebuild the node and its mapped edge.
- **`ParamSpec.from_dict` refuses a non-boolean `trainable` or a non-numeric bound** (`"false"` read as trainable, `true` as 1.0): a saved
  graph carrying one fails to load, a USD stage warns and skips it.  **`run_pod.py`'s stencil, hybrid and coupled goals now fail on a
  broken stencil wrapper** (four seeded faults; schema 5, pencil mesh, D2Q9).  Action: write JSON booleans; re-run schema-4 dry runs.
- **Coupling, second audit**: every acceleration acts on floating fields only (an integer, boolean or PRNG-key leaf raised a `TypeError` under `aitken` / `fixed` / IQN since 0.1.0, and `solver="fori"` rounded it through float32 during 0.4.0 development; MADD-ANO-059); `strict_convergence` under `run_adaptive*` raises only about a solve the stepper keeps; `windowed_loss(mask_unconverged=True)` drops a diverged window from the gradient too (it made the gradient NaN);
  `coupling_diagnostics()` judges the last step under the group that took it, not a replacement; after `jax.grad` of `run_scan` the graph is put back whatever its first node (`step()` raised `UnexpectedTracerError` on a coupled graph with lowercase node names); `run_adaptive` advances its clock by the step a `dt_min` accept keeps, not `dt_min` (MADD-ANO-061).
  Action: re-run `solver="fori"` results from accelerated groups holding integer state, masked sysid fits that saw a NaN gradient, and `run_adaptive` runs that warned "hit dt_min".
- **Sharded stencil wrapper and LBM nodes, round-3 audit** (MADD-ANO-056, 057, 058): `shard_info`'s block size comes from the grid, not a domain integral in the state (a sharded `HeatNode` carrying a vector energy integral left its right end open, 70.9 K off); a nested `ShardedStencilNode` keeps the node's parameters; a replicated `heat_source`/`body_force` the unsharded node refuses is refused sharded (it was applied on every shard);
  `LBMNode`/`LBMPipeNode` refuse a non-finite viscosity or `tau`, a `propeller_x` outside the grid and a disc covering no cell (each ran with no collision or no propeller).  Action: re-run sharded results whose node carries an integral in its state; give pipes of 10 planes or fewer an explicit `propeller_x` (the default is 10).
- **Compliance gates, second mutation audit**: `check_anomalies` requires `residual_risk` on a `partially_resolved` entry, and `_RETIRED_ANOMALY_IDS` is an `{id: reason}` dict that cannot retire an entry last committed as reachable (read from git; CI's compliance job now fetches full history); `check_impl_mapping` refuses a row traced to a class;
  `check_sbom` refuses an orphan component, a recorded Python outside `requires-python` and a missing licence; `check_citations` reads in-text `@Key`; `check_stable_signatures` calls a parameter a recorded `*args`/`**kwargs` used to catch breaking.
  Action: give a retirement its reason; begin a mapping row that means a class with ``Class `Name` ``; write a decorator in prose as code.
- **`PUT /graph/params` answers 200 only for a write a saved graph reproduces** (MADD-ANO-047, 048, 049): the constructor is asked with every changed key and the live values a save carries (a live leaf skipped it: a `HeatNode` past its Fourier limit, `rho_gas > rho_liquid`); a write the running node would honour unlike its rebuild is refused
  (`LBMPipeNode`'s `G` crossing 0; `HeatNode` now fixes its grid at construction, so `grid_points` on a uniform rod is refused); a non-finite value is a 400 before any write (it was stored, then a 500 on every GET); params carry POST's 422 bounds, and a
  state over the cap or of another layout is refused before anything that size is built.  Action: rebuild a node to change such a value, and check that configs saved after a REST write still load.
- **`profile_graph(measure_coupling=True)`** no longer repeats the caller's compile-time warnings (a disconnected node, an inert knob) from its variant and restore recompiles. Action: none.
- **Coupling runtime, audit of the frozen tree**: `run_adaptive*` sub-steps a sub-cycled node at `dt * node_dt / macro_dt` (it advanced `divider * dt`, MADD-ANO-043); on a multi-rate graph a group's diagnostics, predictor history and IQN-IMVJ warm start come only from the solves the step keeps, `strict_convergence` checks only those, and the group no longer solves on the base steps that discarded the result (MADD-ANO-044);
  `solver="fori"` + `iqn-imvj` carries the latching pass's secant columns, not zeros (MADD-ANO-045); `converged` is one verdict, in the residual's dtype, in the report, the profiler, sysid and strict; the profiler samples a multi-rate group on its firing steps only and stops re-warning about its one-iteration variant.
  Action: re-run `run_adaptive*` results with a sub-cycled group, and multi-rate results with a coupling group using `predictor` or `iqn-imvj`.
- **Sharded wrappers, round-2 audit**: `ShardedUnstructuredNode` no longer steps a Cartesian stencil node on the partition layout (a `HeatNode` rod ran 43 K off; MADD-ANO-037, since 0.3.0) nor drops a node's cells past the layout's count, and `shard_info["n_local"]` gives each shard its own cell count so an integral can leave the padding out (MADD-ANO-039, since 0.3.0);
  both wrappers place a domain integral carried in the state as the step returns it, so an integral-emitting node runs in a graph (MADD-ANO-040, since 0.2.1); `ShardedStencilNode` fills a sharded static's global halos periodically under `boundary="periodic"` (MADD-ANO-038, since 0.2.1).
  Action: re-run periodic sharded results whose node reads a static in its halo; mask a domain integral on an uneven partition with `jnp.arange(n_local_max) < shard_info["n_local"]`.
- **FMU bridge and the multi-GPU session verdict**: `FmuTcpBridge.stop()` returns within five seconds and logs any worker still inside a request, which then commits nothing; `set`/`get` refuse a non-integer value reference (`10.9`, `"10"` and `true` addressed variables 10 and 1); `FmuSidecar.set_fmu_state` refuses a snapshot whose nodes, fields or parameters differ from the model, as `set_state` does, and `set_state` keeps a node without state fields.
  `run_pod.py --summarise` closes a checklist item only from files on the current schema, from one commit, with `n_devices` within the devices they saw, holding every case the runner runs and exactly the checks it derives from their results under `LIMITS`; other files read `INVALID`.
  Action: send integer value references; restore snapshots that carry the model's parameters; re-run session goals whose files come from another commit or runner version.
- **`LBMNode(wall_mask=...)` keeps its walls through a save/reload**: the mask was not in `params`, so `from_dict` and a USD stage rebuilt the node with no walls and the reloaded graph ran an open domain, with no error (MADD-ANO-034, since 0.1.0).  The mask is recorded as nested lists of bool, and `POST /graph/nodes` takes it;
  `AdaptiveNode(dtype=...)` is recorded too (a reload came back at the canonical float).  Every built-in node's constructor arguments are now checked through a JSON and a USD reload.
  Action: re-run results from a reloaded graph that holds a walled `LBMNode`.
- **The coupling guide is re-measured on the release tree** (`benchmarks/results/coupling_sweep*_cpu.json`): its "fixed point that barely moves" row now
  starts from `gauss-seidel`/`none`, not Jacobi at ω = 0.8; the heat benchmark fixtures run at half their Fourier number (the rod-end fix doubled
  their gain; slab pairs are unstable above Fo = 3/8); `boundary_interpolation` modes can differ by O(tolerance), not "bit-identical"
- **Sharded wrappers, audit of the frozen tree**: a params write followed by `compile()` reaches the sharded step, for a legacy-contract node's constant and for a halo width that follows a parameter (`HeatNode.stencil_order`) (MADD-ANO-032, since 0.2.0); `ShardedStencilNode` keeps `dt` at the graph's precision under x64 (MADD-ANO-033, since 0.2.0); `HybridNode` and `ShardedUnstructuredNode` forward `update_evaluations()`;
  `ShardedUnstructuredNode` validates `domain_integral_axes` names.  The routes MADD-ANO-024's first fix left open (a part-full pipe's `pipe_radius`, `n_cells`, `stencil_order=3`, wrapped and unprobeable nodes) are closed.
  Action: re-run sharded results whose structural parameters were written after the wrapper was built, and sharded float64 stencil runs under x64.
- **CI test-time report** (`scripts/report_test_durations.py`): a cache read is no longer subtracted twice and is shown apart from compile; a share is capped at 100%; a test that started a process is listed as "work in a subprocess
  (not measured here)", not "slow even with a warm cache"; a report holding only collection errors exits 2 instead of passing.
  Action: none; a slow test's split in the lane summary now adds up, and a subprocess test is no longer sent for a code change a warm cache would make unneeded.
- **Compliance gates catch the defects a mutation audit slipped past them**: `check_doctests` pins every file's example count (`EXAMPLES_PER_FILE`) and it and `check_impl_mapping` sit at the counts; `check_anomalies` refuses a `pytest.skip()`/`xfail()` in a cited test, an aliased mark, `skipif(True)`,
  a broken first-party import and evidence under `tests/viz`, and counts imperative conditional skips; `check_citations` reads digit-led, line-wrapped and `@comment`-hidden keys; `check_heat_stability` judges `grid_points=None`;
  `generate_soup_tables --check` refuses a duplicated block. Action: a new docstring example goes into `EXAMPLES_PER_FILE` in `scripts/check_doctests.py`.
- **Sharded stencils at the edges of the global grid**: a size-1 mesh axis got periodic halos whatever `boundary` said (a one-device `HeatNode` ran as a ring, 0.40 off); `"edge"` wider than one cell put `r0, r1` before `r0`;
  a sharded `HeatNode` ignored `left_temperature`/`right_temperature` (0.87 off with both ends at 0) and now closes its rod ends exactly as unsharded; `ShardedStencilNode` refuses an unknown `boundary` at construction.
  Action: re-run sharded results taken on one device or a size-1 mesh axis, and every sharded `HeatNode` result with end temperatures or `stencil_order=4`.
- **`coupling_diagnostics()` counts every `waveform_iterations` sweep** (since 0.1.0, MADD-ANO-026): `iterations` is the largest sweep's, so `iterations >= max_iterations`
  is exact again (an earlier sweep at the cap read `iterations=1`); new `total_iterations` is the sum. State unchanged; a one-sweep group reports as before.
  Action: a cap check on `iterations` needs no change; read `total_iterations` for the work done.
- **Coupling bounds, confirmation audit:** a field below `tiny/eps` (~1e-31 in float32) no longer reads converged on a flushed change; the float floor counts the evaluations a pass rounds like (sub-cycling automatically,
  internal loops via the new `SimulationNode.update_evaluations()`; an undeclared node's group gets `spectral_usable=False` at the floor); float16-beside-float32 gradient floors, top-of-range spectral weights,
  `PYTHONHASHSEED`-dependent `_meta` seeds and `reset_state` of a key ending `_spectral` are fixed. Action: a node that sub-steps inside `update` should return its sub-step count from `update_evaluations()`.
- **`AdaptiveNode.update` refuses an injected `params` key it does not have**, for every subclass: `{"thetta": 0.9}` was merged,
  never read, and returned the constructor answer; fix the key the error names.  MADD-ANO-004 now also records `ShardedStencilNode`
  (0.2.0-0.3.1) and `ShardedUnstructuredNode` (0.3.0-0.3.1) ignoring a REST parameter write: on those releases set it on the inner node.
- **`PUT /graph/params` refuses a value the running node cannot use** (400, nothing written; since 0.1.0 it was saved and ignored, e.g. `LBMPipeNode.pipe_radius`): rebuild
  the node. `/checkpoint/load` refuses such a checkpoint; FMUs export only parameters the step reads and refuse the rest (pass `SidecarConfig(fixed_params=md.fixed_parameters)`).
  `params_effective` fails a constant split with an `__init__` copy; a `transform="log"` spec with no lower bound refuses values `<= 0`.
- **`LBMNode`'s algorithm ID is `MADD-NODE-011`** (it shared `MADD-NODE-007` with `RigidBodyNode`, which keeps it): update anything keyed on the old ID.
  The gates now refuse a duplicate node ID, a guide ID that differs from its `NodeMeta`, a `<FIX` that is not the entry's `resolution_version` or not a real release,
  a duplicated / skipped / uncollected `verification:` test, and a rod built through a local `HeatNode` subclass; the doctest and mapping floors sit at the current counts.
- **A sharded `LBMNode` is the unsharded node or refuses**: a pressure face on a sharded axis raises at the first step; `ShardedStencilNode` refuses a `boundary` other than a node's declared `halo_boundary()` (LBM: `"periodic"`), its default `"edge"` included; `inlet_face == outlet_face` is refused.
  `WaveletAdaptiveNode(frozen_solver="cg")` is bounded by `kappa * rtol` and a non-converging eager solve says why; inline points refuse complex values (any width), and without a `"dtype"` ints beyond 2**53 and inexact `Decimal`s.
  Action: wrap `LBMNode` with `boundary="periodic"` and shard an axis with no pressure face; use `frozen_solver="gather"`; add an explicit `"dtype"` (e.g. `"int64"`) to inline points.
- **Coupling bound keys no longer under-read**: `spectral_error_bound` adds the residual's float floor (new `precision_limited`); the gradient bound is `inf` where Newton-Kantorovich fails.
  `converged` still reads `True` on a stalled float32 iterate: read those keys with `diagnostics=True`. A NaN no edge reads, or an underflowing scale, no longer reads converged;
  a group with no step yet has no report; colliding group keys (node names containing `+`) are refused.
- **An inline point reference with no `"dtype"` refuses `np.longdouble` values** instead of rounding them to float64
  without a word; the error says no reference form keeps extended precision. Add `"dtype": "float64"` to accept the
  rounding, or convert first (`np.asarray(points, dtype=np.float64).tolist()`); float64 payloads are unaffected
- **Params contract audit:** one "takes `params`" rule for every probe (`update_padded(**kwargs)` calibratable under `ShardedUnstructuredNode`; duck-typed nodes verified);
  `params_effective` probes each path and vector element by value; a changed `gm.params` leaf the step cannot read (`initial_*`, `static_data_deps`) is a `ValueError`, never serialised; int-spelled
  declared constants kept; sharded wrappers forward flux/interface hooks; `run_adaptive*` resolve flux edges. Action: rebuild the node rather than edit such a leaf; fix nodes `params_effective` now fails.
- **`LBMNode`'s Zou-He pressure faces impose the pressure they are given** (MADD-ANO-020, every release): the face carried
  `p/cs2 + S_K` (+15%), a pressure-driven channel 0.58-0.80 of the imposed drop; pressure-driven results change, re-run them.
  `outlet_pressure_avg` reads the runtime wall mask; `LBMPipeNode` gains non-trainable `initial_rho_liquid`/`initial_rho_gas`.
- **`fim` no longer reuses a compiled trace across calls unless `reuse_trace=True`** (pure residuals only): the cache
  froze what the residual read, e.g. `gm.params` (a CRB of 0.41 for a true 44.7).  Specs resolve by one walk (namedtuple
  levels by field name; `check_bounds` refuses unreadable specs); integer leaves are named, not zeroed; NaN bounds refused
- **`WaveletAdaptiveNode` refuses a `mass` its dtype cannot carry** (float32 `mass=1e-6` read J 2.8x off, silently) and its
  active set no longer depends on rounding: eager, jit and graph agree, as do a float and its array spelling. Periodic
  source periodised; `n_levels=6.9` and unknown keys refused. Build in float64 or raise `mass` if refused.
- **Compliance gates no longer pass an empty or unreadable scope**: `check_transforms` / `check_stable_signatures`
  fail when nothing is verified; `check_doctests` floors executed examples and fails a `+SKIP`. Without the extras,
  run `check_transforms.py --allow-missing-optional`; drop a STABLE surface with `--update --accept-removal`.
- **Every anomaly's `affected_versions` is checked against the registry's own version** (PEP 440; convention in the
  registry header): ANO-016 is open-ended again, ANO-006 closes at 0.4.0, ANO-004 starts at 0.1.0, ANO-018/019 read `none`.
  A `verification:` `path::Class::method` must now name a method of that class; fix any loose node id in your own registry
- **`HeatNode.compute_interface_correction(params=)` reads the injected `length`** (a calibrated length
  left coupled interface cells 11 K off, silently); `params_effective` uses a seeded projection (a conserving
  node passes; old-signature subclasses FAIL under `assert_node_verified`); `**kwargs` overrides take `params`
- **`WaveletAdaptiveNode`: the default budget `k` now holds the whole CDD seed** (every
  level-0 function, not `n_coarse**dim`): 16 default configurations silently truncated the
  gathered solve (`dJ/dθ` 5x off at `dim=2, n_levels=2`); `k` below the seed is now refused
- **`fim(scale="nominal")` refuses a `specs` that does not mirror `params`** (misspelt key,
  `to_dict()` entry, wrong nesting, non-dict) by key path instead of silently reporting `"relative"`;
  a spec keyed for a list/tuple leaf now reaches `fim`, `trainable_mask`, `unconstrain` and `check_bounds`
- **Docs said `linear_solver="dense"` was the fallback when the GMRES adjoint
  struggles**: it needs `2*N^2*itemsize` and on a grid-coupled group is refused
  outright (523 GB at ~3.6e5 DOF). The option is unchanged; the advice is not
- **`fit_lm` no longer runs a whole rollout it throws away**: the residual was
  evaluated for a pytree structure `jax.eval_shape` can trace, once per call
  even with `noise_std=None`, and at `n_iter=0` it was the only call made
- **Three more gates that could not fail**: the pyright tiers now cover
  `maddening/__init__.py`, the SOUP drift classifier no longer calls a lost evidence
  row a safe regenerate, and `check_transforms` counts only confirmed references
- **`cloud/_skypilot.py` ported to SkyPilot's client-server API** (`MADD-ANO-016`):
  every call in it was written against the pre-0.7 API and could not work on the
  `>=0.11` floor, and a failed teardown or preemption check is now loud
- **`HeatNode` reports its boundary flux at the rod end**, not a cell inside:
  measured order 1.005 -> 1.999 (and 3.993 for `stencil_order=4`), and the units
  are `K*m/s`, not `W/m^2`. Flux-coupled results move by up to 10%
- **The compliance gates can now fail on the defects they exist to catch**: the
  anomaly and benchmark ID sets are pinned, four gates that reported success
  having verified nothing now fail closed, and two stopped inflating their counts
- **`rk4`/`heun` are 1st order with a time-varying boundary input** (`MADD-ANO-014`):
  every stage holds the one value you pass. The arithmetic is unchanged; carry time
  as a state field with derivative 1 to get 4.00 back — see the new solvers guide
- **`HeatNode` imposes its Dirichlet data at the rod ends, not half a cell in**,
  and its 4th-order ghosts sit where the stencil reads them: measured spatial
  order 1.001 -> 2.000 and 0.954 -> 3.957. `T[0]` no longer equals the datum
- **Non-finite numbers are written as valid JSON** (config, USD `paramsJson`, FMI
  wire): `NaN` / `Infinity` / `-Infinity` are quoted, and both spellings load
- **An FMU instance may reconnect at once**: the bridge no longer refuses the slot
- **A failed `compile()` leaves the graph exactly as it was**: schedule, rate
  dividers, external-input zeros and `params` commit after the last refusal.
  `auto_couple` keeps groups it cannot replace; `run_adaptive` checks after it
- **`coupling_diagnostics()["residual"]` has a float32 noise floor**, documented:
  a converged group's residual is a cancellation, so `solver="ift"` and `"fori"`
  can report `0.0` and `1e-05` for one state.  The state and verdict are exact
- **A compliance gate no longer calls an optional subpackage's symbol stale**:
  a `maddening.usd.*` reference in an environment without the `usd` extra is
  reported as *not checked*, with the extra named, instead of failing the run
- **A coupling group no longer reports `converged=True` with a small field far
  from its fixed point** — `atol` removes a field from the norm, so it defaults to
  `0.0` and is live under every norm — and `iterations` at the cap agrees by solver
- **A recompile no longer re-phases a multi-rate graph or restarts a coupling warm
  start** (MADD-ANO-079, since 0.1.0) — only graphs with a rate divider > 1 or a warm start were ever affected;
  `set_param_spec` and `external_inputs` are now checked as strictly as `params`
- **Swapping a surrogate in or out no longer resets an edge's `additive`, units,
  `mapping` or fitted mapping weights** (MADD-ANO-077, since 0.1.0): an additive input read 3.0 before a swap and
  1.0 after.  Re-check results crossing `replace_node` / `POST /surrogate/deactivate`
- **`jax.grad` no longer crashes on a stiff coupling group**: a failed GMRES
  adjoint re-solves directly at small DOF, or names `linear_solver="dense"`
- **`error_estimate` accounts for `relaxation`** — and is an estimate, not a bound
- **Degenerate sysid inputs are refused, not reported**: a non-finite Fisher matrix,
  a σ that is not positive, a mask keyed unlike `params`, and `lr`/`eps`/`lam_up`
  values that invert their meaning now raise; `params_pytree` keeps float64 under x64
- **The four pre-0.4.0 `scripts/check_*.py` compliance gates now fail on the defects they
  exist to catch** — zero-reference transform scan, MRO-resolved mappings, a
  `%`-commented bib entry, an unchecked `resolution_status`.  Re-run them
- **A failed graph mutation is now a no-op**: `add_node` builds the state before
  it registers the node, so an `initial_state()` that raises no longer wedges
  the graph with a ghost `step()` dies on; same for `reset_state`/`remove_node`
- **A failed resume now leaves the graph untouched** and logs `RESUME FAILED`, not
  "starting fresh"; sharded wrappers honour `shard_axes`, pass `params` to a
  `**kwargs` node (whose gradient was silently zero) and refuse an indivisible grid
- **Do not pair `convergence_norm="interface"` with the auto-detected
  `accelerated_fields`**: both are the edge fields, so an accelerator fixes
  exactly what the criterion measures — use a norm that sees the whole state
- **Every `CouplingGroup` knob its configuration ignores now warns** — `relaxation`,
  `jacobian_reuse`, `accelerated_fields`, `waveform_iterations`,
  `boundary_interpolation`, `linear_solver`, `strict_convergence` — at your call line
- **A `CouplingGroup` tolerance its norm never reads now warns** instead of
  turning silently: `tolerance` under `convergence_norm="mixed"`/`"interface"`,
  and `rtol` under `"l2"`.  Set the knob the message names instead
- **`fim` reports an unidentifiable parameter as `+inf`, not a tight bound**:
  `crb` was `diag(pinv(F))`, which is small in the null space; the new
  `FIMReport.rank` counts the directions the data resolves (`rank_rtol=`)
- **A fit returns the leaves it did not fit, bit for bit**: `fit`, `fit_lm` and
  `fit_multiple_shooting` copy every leaf outside the mask from the starting
  pytree, so comparing before and after says exactly what a calibration touched
- **A wrapper now reports the `static_data` of the node it wraps**, so the
  `static_data` drift check finally fires through `ShardedStencilNode`,
  `HybridNode` and friends instead of hashing to `0` forever
- **`fit`/`fit_lm`/`fit_multiple_shooting` refuse a `mask` that names a leaf its
  `ParamSpec` freezes**: it was optimised unclipped in physical coordinates and
  could leave its bounds — make the parameter trainable in the spec instead
- **`strict_convergence` no longer ignores a diverged group that overflowed**:
  a NaN residual raises like any other failure instead of passing silently,
  which is what `coupling_diagnostics()` already reported for it
- **Examples no longer save plots into the installed package** (they broke on
  a read-only install): output goes to the working directory, usage lines use
  `python -m maddening.examples...`, and a smoke test pins both
- **A coupling group no longer reports convergence it has not reached**:
  `acceleration="aitken"` needs the threshold met on two consecutive passes
  (a lone dip is not arrival), `max_iterations=1` reports its real residual
- Sharded pointwise nodes honour parameter writes again (`PUT /graph/params`)
- `POST /surrogate/deactivate` restores every edge field, or changes nothing (MADD-ANO-077)
- **FMU export of a real graph had no inputs and a wrong step size** (MADD-ANO-078, since 0.3.0): inputs
  now come from the graph's external-input list as `<node>.<field>`, and the
  step is the graph's base timestep
- **FMU bridge and C wrapper robustness**: a non-whole-multiple communication
  step, an unusable step time, an oversized or truncated reply, an over-nested
  request, a second instance and a vanished peer no longer desynchronise,
  wedge or kill the importer; `set` is atomic, values are narrowed before the
  finiteness check, an unset input reads as zero, and memory-safety defects
  (leaked and unterminated buffers, a missing reply parsed as data) are fixed
  along with the fuzz harness that had missed them
- **FMI `min`/`max` for open bounds** advertised a value the sidecar then
  refused; the nearest representable float32 inside the interval is advertised
  instead
- **`compile()` no longer discards a calibration**: live params leaves are
  carried across a recompile (`gm.reset_params()` to undo), a partial pytree
  is completed from the live values, and `load_state` compiles before
  restoring
- **Params pytree correctness**: flux edges honour `params`, `ParamSpec`
  clamps stay strictly inside open bounds and reject NaN/inf, leaf dtypes and
  Python scalar types are preserved, wrong-shape leaves are refused rather
  than broadcast, and `fim` / `fit_lm` accept a per-leaf `noise_std` pytree
- **Interface-mapping serialisation hardening**: edge-key `param_specs` apply
  after the edges exist, stale point references are refused by content hash,
  asset loading is bounded and confined to `base_dir`, rebuild failures are
  `MappingRebuildError`, and two mapped edges on one field pair get separate
  slots via `EdgeSpec.ordinal`
- **Resume-from-URL hardening**: the manifest URL is derived from the URL
  *path*, so pass `manifest_url=` / `RESUME_MANIFEST_URL` for a presigned
  object; fetches time out (`MADDENING_RESUME_TIMEOUT`), stream to disk, and
  clean up their temporary directory
- **`AdaptiveNode` survives a config / USD round trip** (`blindness_gate` was
  dropped, `n_max` replayed as a duplicate keyword); the gradient-capture
  diagnostic evaluates the parameters in use, is memoised, is disabled by
  `MADDENING_ADAPTIVE_DIAGNOSTICS=0`, and the `jnp.where` tangent trap is
  warned about with `mask_safe` as the remedy
- **Coupling with non-float state leaves**: integer, boolean and PRNG-key
  leaves survive the IFT closure exactly, keep their dtype through
  `unflatten_coupled_state`, and no longer break reverse-mode differentiation
  through `lax.scan`
- **IQN acceleration**, four defects: IQN-ILS silently ran as Aitken without
  Jacobian reuse; the safeguard vetoed valid steps on stiff problems; the
  secant basis is now built from operator-output differences (Degroote 2009);
  the gradient is no longer NaN with `jacobian_reuse > 0`
- **Coupling diagnostics and solver settings**: the IFT path honours
  `convergence_norm` and no longer freezes non-interface fields in IQN modes,
  fori diagnostics are no longer one iteration stale, the IFT linear solve no
  longer reports a spurious GMRES breakdown, and Gauss-Seidel no longer raises
  `KeyError` for a flux edge whose consumer precedes its producer or for a
  flux source under IQN
- **Every run compiled the step three times** because weak-typed leaves
  restrengthened after the first step; `compile()` and `set_node_state()`
  normalise weak types (results bit-identical), and
  `_default_external_inputs()` no longer reallocates zeros every step
- **Sharded boundary inputs**: grid-shaped inputs are sharded and halo-padded
  instead of replicated, the shape heuristic compares the full leading grid
  shape, and `HeatNode.update_padded` takes `params=`
- Node fixes: `LBMPipeNode` multiphase used `np.exp` on a tracer and had a NaN
  gradient in the EDM velocity clamp; `HeatNode` ignored an injected `length`
- **REST API**: a non-finite constructor constant is a 400 naming the
  parameter instead of a 500 that leaves `GET /graph` broken for the process's
  life; `PUT /graph/params/{node}` echoes what it wrote and validates first;
  `PUT /graph/state` and `POST /graph/edges` validate their inputs; `add_node`
  refuses names containing `/`, `#` or `->`
- **Multi-GPU session runner**: the `exchange` goal timed an unsharded input
  and charged both transports the same constant, and `recommend()` accepted
  rows no real GPU produced; inputs are pre-sharded outside the timed region
  and a recommendation needs a real accelerator run.  Stale `jax[cuda12]` /
  Python 3.10 pins in `docker/Dockerfile.cloud` and the cloud examples are
  corrected
- **`scripts/typing_baseline.py` can no longer make a broken pyright run look
  like a result**: infrastructure failures exit 2, a wrong interpreter or an
  empty analysis is detected, and `pyright` is pinned to `1.1.414` in `ci`
- `scripts/generate_stability_report.py` imports every `@stability`-tagged
  module from a list a compliance test checks against the source, so none can
  silently drop out of the release gate's report
- Property-test flakes removed: the rollout-parity and interface-mapping
  adjoint assertions scale their tolerance with the magnitudes compared and
  pin the failing cases.  No library code changed.
  `maddening.testing.strategies` no longer rejects float32 bounds that are not
  exactly representable

### Security
- **The cloud entry point no longer writes TURN credentials to its log** (MADD-ANO-288, in 0.1.0 to 0.3.1): it logged `MADDENING_CLOUD_CONFIG` verbatim at INFO, so a TURN `username` and `credential` in `stream_config.ice_servers` reached the container's log; credentials, and the user part of a `scheme://user:password@host` URL, are now logged as `<redacted>`. Action: rotate a TURN credential that an affected release's log holds.
- **`X-Forwarded-For` no longer gets a request past a loopback-bound API's `Host` check or its peer backstop** (CRITICAL, MADD-ANO-179, never released): uvicorn's default proxy headers let any request over loopback name its own peer, so a rebound page sending `X-Forwarded-For: x` drove every route;
  a peer that is not a loopback IP literal and any request carrying `X-Forwarded-For` / `Forwarded` now need the token, the `Host` check asks every request without one, and the library's launch paths pass `proxy_headers=False`.
  Action: never configure loopback as a trusted proxy; an in-process `TestClient` presents `server.auth.token` (or uses `base_url="http://127.0.0.1", client=("127.0.0.1", 50000)`); a reverse proxy to a loopback bind presents the token.
- **A loopback-bound API answers only to loopback host names** (CRITICAL, MADD-ANO-076, now resolved): a DNS-rebinding page's `Host` and
  `Origin` agree, so it passed the Origin check and could drive every route, `/cloud/launch` included; any other `Host` is now a 403.
  Action: pass `SimulationServer(allowed_hosts=)` to serve a loopback-bound server under a proxy's name or an `/etc/hosts` alias
- **Cloud launches reach ready again, and cross-origin browser requests are
  refused** (CRITICAL, MADD-ANO-076, since 0.1.0; partially resolved, a DNS-rebinding page still passes): set `MADDENING_TRANSPORT_TOKEN` so the ZeroMQ CURVE key is not the
  cleartext API bearer token; pass `allowed_origins=` to embed the UI elsewhere
- **The ZeroMQ transports bind loopback and encrypt any other bind** (CRITICAL, MADD-ANO-015):
  5555/5556/5580 published state, commands and worker rendezvous to anyone who
  could reach them; set `MADDENING_TRANSPORT_TOKEN` on both sides to stream off-box
- **The API requires a bearer token unless it is bound to loopback** (CRITICAL, MADD-ANO-051):
  set `MADDENING_API_TOKEN` or read the one logged at start-up; `JobConfig.ports`
  no longer defaults to `[8000]`, so a cloud launch stops opening the API port
- **FMI/USD hardening** (three HIGH): a silent TCP peer no longer wedges the FMU
  bridge, `set_state` is value-checked exactly as `set` is (MADD-ANO-082), and loading a USD
  stage no longer imports the class it names (MADD-ANO-075, since 0.1.0) — pass `node_registry=` to allow one
- **Cloud surface**: the signaling WebSocket validated its own token, not the
  client's (CRITICAL, MADD-ANO-053; set `MADDENING_STREAM_SECRET`), and the unauthenticated
  API now caps `n_steps`, node dimensions and training args, warning on 0.0.0.0
- **Mapping-spec assets are opened once** (`O_NOFOLLOW`, `fstat`): the size cap
  and the data now come from the descriptor that was checked, closing a
  time-of-check/time-of-use window for a writer in the config directory
- **FMU bridge no longer unpickles importer bytes** (CRITICAL, MADD-ANO-054): the FMU-state
  blob is an arrays-only `npz` validated before use — regenerate any stored
  blob.  `FmuSidecar.handle` stays pickle-based and trusted-clients-only (MADD-ANO-055)
- **FMU sidecar `set_state` zip bomb** via an archive member without a `.npy`
  suffix: the archive directory is checked before `np.load` runs, with
  per-member and total declared-size caps
- **REST checkpoint endpoints are confined to a directory** (MADD-ANO-052): paths are
  relative to `SimulationServer(checkpoint_root=)` (default `./checkpoints`).
  Since this release the API *does* authenticate on a non-loopback bind
  (bearer token, see the Security entry above); loopback is unchanged

### Known Anomalies
- **MADD-ANO-275 (new, open, minor, since 0.1.0)**: `acceleration="iqn-ils"` or `"iqn-imvj"` under `iteration_mode="jacobi"` can stall on a coupling group the plain Jacobi loop solves (measured: 50 passes of 50, 0.14 from the fixed point, where plain Jacobi takes 12). It is reported (`converged=False` at the cap; `strict_convergence=True` raises). Use Gauss-Seidel with IQN, or Jacobi with no acceleration or Aitken; the coupling guide's paragraph on Jacobi with IQN says what was measured.
- **MADD-ANO-277 (partially resolved in 0.4.0, since 0.1.0), MADD-ANO-278 (resolved in 0.4.0, since 0.1.0) and MADD-ANO-279 (new, never released, resolved)**: a constructor argument kept outside `params` is still not saved (`to_dict()` and `save_graph_to_usd` warn); an assignment to `node.params` on a wrapper was lost (`### Changed`, `### Fixed`); and a `copy.deepcopy` of a graph put a node's value back over a constant calibrated in `gm.params` after a node write.
- **MADD-ANO-280 (new, open, since 0.3.0)**: `sharded_gmres` on its default (lineax) backend counts `max_iters` and `.iters` in restart cycles, not steps: on 1000 unknowns `max_iters=10` executed 1,583 to 4,490 operator products and reported `iters` of 32 to 89 (24 white-noise right-hand sides, lineax 0.0.7 and 0.1.1), where `backend="loop"` executed 53. The value and its flag are right; for a bound on the work pass `backend="loop"`. Documented in 0.4.0, unit unchanged. **MADD-ANO-282 (new, open, never released; minor)**: with `differentiable=True` a solution within a factor of the operator's diagonal of the dtype's largest number reads `converged=False` beside an infinite `residual_norm` (NaN with an `x0` or an `atol`), where the ordinary call answers; rescale such a system. **MADD-ANO-271, 272, 281 and 283 are resolved in 0.4.0** (`### Fixed`, `### Changed`; 272's remaining form, infinities for a solution that overflows, was never released): this replaces their status in the entry below.
- **MADD-ANO-284 (new, resolved, never released) and MADD-ANO-285 (open, since 0.3.0)**: 273 (major): an integer or Boolean field carried by an edge inside a coupling group stopped the solve after one or two passes with `converged=True`, `residual=0.0`, far from the fixed point (`"ift"` under Gauss-Seidel with the reader swept first, `"ift"` with a relayed field, `"fori"` under Jacobi); every route now reaches the pass's fixed point in the same passes (rule and proof in `core/coupling/_carried_fields.py`). 274 (minor): under `"aitken"`, `"iqn-ils"` and `"iqn-imvj"` a run depends on the units ONE field is written in (8 passes converged against 25 unconverged with one field `2**20` smaller); `CouplingGroup.convergence_norm`, `_scaled_change` and the release notes said a verdict did not, and now say what holds: use comparable units, or `"none"` / `"fixed"`. Also documented, no change of behaviour: a `predictor` is live at `max_iterations=1` and moves the single pass's result at order one.
- **MADD-ANO-270 (new, open, never released), MADD-ANO-271 (open, since 0.3.0) and MADD-ANO-272 (partially resolved in 0.4.0, since 0.3.0)**: three things `sharded_cg` / `sharded_gmres` do not report or refuse, found by the audit of the solver change before it merged. 270 (major): with `differentiable=True`, `converged` is the value's; each derivative is one more solve from zero under the same `rtol` and `max_iters`, and a solve that converges quickly only because of its `x0` reads `converged=True` beside a derivative from a solve that stopped on its budget (JVP off 2.6e-3 at the default tolerance and budget, float64, condition number 4001): give `max_iters` what a solve from zero needs. 271 (major): a NaN `rtol`, and a negative or NaN `atol` that is traced, are not refused, and the lineax routes then read `converged=True` on a value 72 % off or on zeros: check a computed tolerance before passing it. 272 (major): a right-hand side whose norm overflows the dtype is now solved (in 0.3.0 and 0.3.1 the CG loop answered zeros with `converged=True`); a solution that itself overflows still comes back as infinities beside a finite `residual_norm` and, where the rescaled solve met its tolerance, `converged=True`: rescale, or test the value with `jnp.isfinite`.
- **MADD-ANO-263 (resolved in 0.4.0) and MADD-ANO-264 (resolved in 0.4.0 for the default call), both since 0.3.0**: two measured behaviours of `sharded_cg`, registered and then fixed in this release (`### Fixed`, `### Changed`). 263 (major): the loop backend (`backend="loop"`, or the default with a preconditioner) took `converged` and `residual_norm` from its recursively updated residual, so in float32 it answered `converged=True` under the default `rtol=1e-6` on solves whose true residual was 10 to 100 times the tolerance. 264 (minor, loud): the default route without a preconditioner is lineax's CG, which returns NaN with `converged=False` in float32 above about 1e5 unknowns (a smooth right-hand side from about a thousand); the default call now falls back to the loop there. 264 stays open for `backend="lineax"` asked for by name and for `ift_linear_solve(solver="cg")`: pass `backend="loop"`, `solver="gmres"`, or solve in float64.
- **MADD-ANO-259 and MADD-ANO-260 (new, open, never released)**: two idioms on an internal edge are outside what a coupling report's float floor counts, and `spectral_usable` can stay `True` beside a bound below the distance: a transform that narrows the dtype on an edge whose source is a boundary flux (259; carry the value as a state field), and a sum of many terms written inside an edge `transform=` (260; declare the operator as a static sparse mapping). The inspection guide states both beside the rule each is excepted from.
- **MADD-ANO-247 (new, open, never released)**: under `convergence_norm="interface"` the float floor of a value an edge delivers is taken at the delivered value's own magnitude. A group whose mapping differences two entries of one field (a gradient or jump operator as an interface mapping) can stall at `residual=0.0` with `spectral_error_bound` below the true distance and `spectral_usable=True`, under Jacobi as under Gauss-Seidel (0.25x at a field 100 times the difference). Read such a group under the L2 or the mixed norm with `iteration_mode="jacobi"`, or have the producer hold the difference as a field of its own; **MADD-ANO-212**'s workaround now says that Jacobi alone does not help under the interface norm.
- **MADD-ANO-239 (new, open, never released)**: `spectral_usable` does not see a pass whose Jacobian one float32 rounding of a same-pass value moves. A float32 Gauss-Seidel group whose member multiplies two other members' fields, held in units 1e3 or more apart, can read a settled `rho_spectral` far from the radius at its float floor (0.324 for 0.0238; `spectral_error_bound` held). Run such a group in float64 or under Jacobi, or keep the two fields in units within a decade of each other.
- **MADD-ANO-216 (new, never released, resolved)**: `fit_lm` reported `converged=True` beside a jump of a non-differentiable residual. **MADD-ANO-217 (new, open)**: `fit_lm` from a stiffness started 18 to 80 times too high can stop short, unconverged, on the end of a wide damping range; restart from the returned point. **MADD-ANO-021 (open)** gains its manifestation in system identification (what `fit_lm`, `fit` and `fim` say on a residual with jumps).
- **MADD-ANO-195 (never released) is now resolved**: the interface norm, its floor, its bound and the report read a mapped edge as the step delivers it (or at its source, where a static mapping delivers more entries than the source holds).  **MADD-ANO-211 (new, never released, resolved)**: the float floor of an interface edge that delivers a wider dtype than its source field was taken at the wider dtype's `eps` (see `### Fixed`).  **MADD-ANO-213 (new, never released, resolved)**: under the interface norm a field read by several internal edges was counted once per edge by the residual and once by the spectral analysis (see `### Fixed`).  **MADD-ANO-212 (new, never released, open)**: the floor's gain of a same-pass read is measured along the source's own state, which a difference of entries of one field cancels -- a Gauss-Seidel group stalled behind a mapping row `[1, -1]` reports `spectral_error_bound` at 0.054x to 2.6e-4x the true distance, `spectral_usable=True`; use Jacobi there, or have the producer hold the difference as its own field.
- **MADD-ANO-178 to 185 (new, resolved in this release)**: the REST round-8 audit's defects (see `### Security` and `### Fixed`): 178 (params write back) and 179 (forwarded peer, critical) never released; 180 to 185 carried from v0.1.0 to v0.3.1.
- **MADD-ANO-163 to 166 (new, resolved in this release)**: a checkpoint load restored parameter values `PUT /graph/params` refuses (163, never released); a `HybridNode` parameter write was lost (164, since 0.1.0); `/sim/run`'s 503 said nothing changed after it had stepped (165, never released); `/sim/profile` was a 500 for a graph that cannot step (166, since 0.2.0).  See `### Fixed`.
- **MADD-ANO-075 to 082 (new)**: defects this release fixes that had no entry, five of them carried by releases — a USD stage imported the module it named (075, since 0.1.0); the API checked no `Origin` (076, critical, since 0.1.0, `partially_resolved`: a DNS-rebinding page still drives a loopback bind, so run the API only while you use it); a surrogate swap dropped edges' `additive` and units (077, since 0.1.0); FMI export of a real graph had no inputs and a 1e-3 step (078, since 0.3.0); a recompile re-phased a multi-rate graph (079, since 0.1.0) — and three majors never released (080, 081, 082; see `### Fixed`, `### Security`).
  **MADD-ANO-048** is `partially_resolved`: a `gm.params` write or a fit still saves a value the constructor refuses, and the saved graph then does not load; bound such a parameter inside the constructor's range.  **MADD-ANO-058** now also covers `propeller_radius > 1`.
- **MADD-ANO-069 (new, never released)**: `hold_undetermined` held every direction a fit's gradients had not spanned, so a short or fast-converging fit came back above the loss it had reached (0.4.0 development builds only; see `### Fixed`)
- **MADD-ANO-064 to 067 (new, resolved in this release)**: the four sharding defects above (064, 065 and 067 since 0.2.1 or 0.3.0; 066 never released).
  **MADD-ANO-068 (new, `partially_resolved`)**: XLA (jaxlib 0.10.2 to 0.11.2) miscompiles a `ShardedStencilNode` step inside `run_scan` for a node reading a sharded
  static replicated over a mesh axis in its halo beside a window at its `shard_info` offset; the loop entry points now refuse such a node (see `### Changed`), and a loop you write yourself is not asked
- **MADD-ANO-063 (new, never released)**: a write moving the points a mapped edge was built from ran on the old mapping weights and saved a config that did not load (see `### Fixed`); **MADD-ANO-049** is now resolved (an in-process non-finite parameter is served as a token, not a 500), and **MADD-ANO-022** narrowed: a write is refused, a fit through the mapped edge still uses the constructor's geometry
- **Severities defined; fourteen relabelled; four entries partially resolved**: `AnomalySeverity` now defines each level, and a silent wrong result is never `minor`, so MADD-ANO-005, 009, 025, 027, 029, 031, 038, 041, 048, 055, 057, 058 and 061 move to `major`, and 052 (a default-exposed route) to `critical`.
  MADD-ANO-032, 036, 047 and 048 are `partially_resolved`, not `resolved`: their routes outside a graph or the REST route are still live (see each `residual_risk`); the release notes' Known anomalies section now names every reachable entry, and a test keeps it so.
  Sharded nodes: iterate `update_padded`'s `shard_info` over its `int` keys only; `"n_local"` (unstructured wrapper) is the one string key.
- **MADD-ANO-062 (new, resolved in this release)**: `HeatNode` accepted a non-positive `length` or `timestep`, a negative `thermal_diffusivity` and `grid_points` out of order, and answered wrongly without a word (since 0.1.0; see `### Changed`)
- **MADD-ANO-059, 060, 061 (new, resolved in this release)**: accelerating a coupling group holding an integer, boolean or PRNG-key leaf raised a `TypeError`; a group-internal flux edge read by the interface norm or a sub-cycled member's linear interpolation raised a bare `KeyError`; `run_adaptive` advanced its clock by `dt_min` on a `dt_min` accept whose state covered more (all since 0.1.0; see `### Fixed`, `### Changed`).
  **MADD-ANO-027** now says each extra waveform sweep applies at least one more pass, moving a converged state by about one residual
- **MADD-ANO-051, 052, 053 (new, resolved in this release; all since 0.1.0)**: the HTTP API served every route, `/cloud/launch` included, with no credential while the container bound `0.0.0.0`; the checkpoint routes took any server path; the signaling server admitted every client (see `### Security`).
  **MADD-ANO-054 (new, never released)**: the FMU bridge unpickled the importer's state blob, remote code execution; **MADD-ANO-055 (new, resolved)**: `deserialize_fmu_state` and `FmuSidecar.handle` unpickled their input (since 0.3.0).
  The registry now holds every defect a release carried and every critical or major one found in the cycle, shipped or not (CONTRIBUTING.md)
- **MADD-ANO-047, 048, 049 (new; 049 resolved in this release, 047 and 048 `partially_resolved`)**: `PUT /graph/params` accepted a write flipping a branch the node fixed at construction, values its constructor refuses, and a non-finite value (since 0.1.0; see `### Fixed`).
  **MADD-ANO-050 (new, open)**: two `HeatNode` rods coupled end to end by a converged exchange are unstable above Fo = 3/8 at `stencil_order=2` and 0.226 at 4, not the 1/2 or 5/16 each accepts; keep Fo below those on such pairs (`compile()` warns; see `### Changed`)
- **MADD-ANO-046 (new, resolved in this release)**: a sub-cycled node whose timestep did not divide the macro timestep drifted by a fixed fraction of every step (since 0.1.0; see `### Changed`)
- **MADD-ANO-043, 044, 045 (new, resolved in this release)**: `run_adaptive*` advanced a sub-cycled node `divider * dt` per step; a multi-rate group's diagnostics, predictor and IQN-IMVJ warm start came from discarded solves;
  `solver="fori"` + `iqn-imvj` carried zero secant columns, so `jacobian_reuse` did nothing (all since 0.1.0; see `### Fixed`)
- **MADD-ANO-037 to 042 (new, resolved in this release)**: `ShardedUnstructuredNode` stepped a Cartesian stencil node wrong, dropped cells past its layout's count and gave no way to leave padding out of an integral (all since 0.3.0); both sharded wrappers placed a domain integral in the state like a grid field (since 0.2.1);
  a sharded static was edge-filled under a periodic wrapper (since 0.2.1); `halo_exchange` ignored an unknown `boundary` key and `ShardedStencilNode` accepted an empty `axis_map` (since 0.2.0).  See `### Fixed` and `### Changed`
- **MADD-ANO-034 (new, resolved in this release), 036 (new, `partially_resolved`)**: a walled `LBMNode` reloaded with no walls (since 0.1.0; see `### Fixed`); a config round trip dropped a node's sharding with no word (since 0.2.0; now a warning, see `### Changed`).
  **MADD-ANO-035 (new, open)**: on a balanced partition not in global order, `ShardedUnstructuredNode` reads a global-order state written with `set_node_state` as partition layout, so each cell steps from another cell's value (since 0.3.0).
  Convert the state with `partition_value` first, or renumber the cells with `np.argsort(partition_assignment, kind="stable")`; an explicit layout on the state write is planned for 0.5.0
- **MADD-ANO-033 (new, resolved in this release), 032 (new, `partially_resolved`)**: `ShardedStencilNode` stepped with a float32 `dt` under x64; the sharded wrappers kept a compiled step across `compile()`, so a legacy node's params write after a step never reached the sharded physics (a wrapper's own `update()` still does until `invalidate_static_cache()`; both since 0.2.0; see `### Fixed`).
  **MADD-ANO-024** now records the routes its first fix left open, all closed, and **MADD-ANO-004**'s workaround says it held only before the first step
- **MADD-ANO-028, 029, 030 (new, resolved in this release)**: periodic global halos on a size-1 mesh axis; a wide `"edge"` fill that copied the shard's first cells;
  a sharded `HeatNode` that ignored its end temperatures (all since 0.2.0; see `### Fixed`).  **MADD-ANO-031 (new, open)**: at `stencil_order=4` an end with no
  boundary input is not insulated (order 2 is); give it a temperature or use `stencil_order=2`
- **MADD-ANO-026 (new, resolved)**: with `waveform_iterations > 1`, `iterations` was the last sweep's count, hiding an earlier sweep at the cap (since 0.1.0).
  **MADD-ANO-027 (new, open)**: `waveform_iterations > 1` restarts the same solve rather than relaxing a waveform, and sub-step interpolation runs between
  iterates, so a converged step is the same in every mode (since 0.1.0). The option is now marked experimental; use `waveform_iterations=1`
- **MADD-ANO-024, 025 (new, resolved in this release)**: `PUT /graph/params` accepted and saved a value a node consumes at
  construction (since 0.1.0); a sharded `LBMNode` imposed its pressure faces at every seam and, by default, filled its global
  halos unlike its periodic streaming (since 0.2.0). Both are refusals now (see `### Fixed`)
- **MADD-ANO-021, 022, 023 (open)**: a gradient through a state-triggered branch omits the event time (BallNode's bounce:
  exactly 0 in the drop height); a mapped edge on a grid derived from a trainable parameter keeps its constructor
  geometry when that parameter is calibrated; the FMU TCP bridge authenticates no caller. Workarounds in the registry
- **MADD-ANO-020 is resolved in this release**: `LBMNode`'s pressure BC imposed the wrong face density since 0.1.0 (see `### Fixed`)
- **MADD-ANO-019 is resolved in this release**: a diverged coupling state can no longer read as converged (see `### Fixed`)
- **MADD-ANO-017** now also names `implicit_euler_step` (a float32 state under x64 is refused, in every
  release) and the `update`/`integrate_node` dtype divergence; `affected_versions` widens to `>=0.1.0`.
  It also records the legacy `solver="fori"` Aitken carry, which fails under x64 exactly as every other
  configuration does and adds no failure of its own
- **MADD-ANO-018 is resolved in this release**: `params` reaches every solver path (see `### Fixed`)
- **MADD-ANO-018**: a parameter calibrated through `gm.params` or `fit` reached
  `update()` and could not reach `derivatives()`, `implicit_residual()` or
  `integrate_node()` -- they took no `params`, so they ran constructor values
- **MADD-ANO-017**: `jax_enable_x64` does not reach `GraphManager`'s scan paths --
  float64 params against a float32 state seed, so `run_scan*`/`run_sweep` raise
  `TypeError` (open, minor); node-level `update()` unaffected, workarounds in the registry
- **MADD-ANO-016**: `cloud/_skypilot.py` was written against a SkyPilot older than
  the supported floor, so every `CloudSession` launch, teardown and preemption check
  was broken -- partially resolved in 0.4.0; end-to-end behaviour still unverified
- **MADD-ANO-015**: the ZeroMQ transports bound every interface unauthenticated
  and in cleartext from 0.1.0, and `launch_vm` published them whatever the job
  config said -- resolved in 0.4.0 (critical, safety_relevant)
- MADD-ANO-011/012/013 (BallNode, HeartPumpNode, the rigid bodies; all open):
  both ODE nodes name forward Euler and implement something else, and a float64
  quantity is pinned to float32 in several places -- HeartPumpNode's
  `backpressure`, and the `inertia`, `gravity` and `initial_state` casts in
  `RigidBodyNode`, `RigidBody2DNode` and `HeatNode` -- read the scheme from the
  algorithm guide, not from `discretization`
- **MADD-ANO-010**: a *string* that spells `NaN` / `Infinity` / `-Infinity` is now
  refused by `to_dict`, the USD JSON attributes and the FMI wire, because it would
  read back as that float -- spell such a value differently (minor, context_dependent)
- **MADD-ANO-001 (LBM GPU segfault) is resolved**: it needed jaxlib 0.5.1, which
  0.1.0-0.3.1 permitted and 0.4.0's floor does not; re-verified on GPU at jaxlib
  0.11.2 / CUDA 12.9, `LBMPipeNode` GPU vs CPU agreeing to 2.4e-07
- **MADD-ANO-007/008/009 (HeatNode) are resolved in this release**, all found
  by the MMS harness and all fixed before release: Dirichlet data was applied at
  the first cell centre, not the documented rod end (order 1, not 2);
  `stencil_order=4` converged at order 1 and was less accurate than the default;
  and the documented Fourier bound of 1/2 was the 3-point stencil's, with the
  4th-order stencil diverging inside it.  ANO-008's recorded diagnosis was
  corrected on re-derivation: the ghosts sit at -dx/2 and -3dx/2, and the oracle
  restores 3.76/3.90/3.95, not 3.78/5.02/4.79.  ANO-007 also said
  `compute_boundary_fluxes` was "not touched"; it is fixed too, the reported
  left flux moving from 10.6% error at n=10 to 0.13% and from order 1.005 to
  2.005
- Every anomaly whose defect is still reachable now records an open-ended
  `affected_versions`; ANO-005 no longer claims 0.4.0 is clean, and ANO-002's
  workaround names `thermal_diffusivity`, not the `alpha=` `HeatNode` never had
- MADD-ANO-005: before 0.4.0 `converged=True` was a residual test, not a bound
  on the distance to the fixed point.  0.4.0 applies the threshold to an error
  *estimate* instead; read `coupling_diagnostics()['ratio_usable']` to see
  whether the contraction ratio was usable, and treat `False` as the old
  behaviour.  The estimate is not a bound and can understate by 122x on a
  hidden slow mode (major, partially_resolved in 0.4.0, context_dependent)
- MADD-ANO-003: AdaptiveNode frozen-set gradient omits a first-order term at
  active-set switches -- the frozen-set objective jumps where two candidates
  swap rank, so no Clarke subgradient exists there and the integral of the
  returned gradient misses the sum of the jumps crossed (measured: 33 % of the
  objective change over a 0.1-wide theta window at n_max=256, k=16; ~1e-8 at
  k=64) (open, context_dependent)
