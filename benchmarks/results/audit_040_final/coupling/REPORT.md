# Audit: coupling (`src/maddening/core/coupling/` + the coupling machinery in `graph_manager.py`)   (c51cd6ad075cb8c0d4a03f4ab49ff0f5dcca4885)

## Summary

Six findings, one of them a silent wrong answer in the **default**
configuration, and one a property test that fails on this commit.  0.4.0 gave every convergence norm an `atol` dead band,
including the L2 norm — but `CouplingGroup.atol`'s docstring and the
release's new inert-knob `UserWarning` both state that `atol` is *not
read* under `convergence_norm="l2"`.  It is (`graph_manager.py:1266`),
and a group whose float fields sit below the default `atol=1e-8` exits
after one pass reporting `residual=0.0, error_estimate=0.0,
converged=True` while 50% from its fixed point.  Separately, the new
error bound is not a bound (two independent mechanisms, 122x and
exactly `1/omega`), and `jax.grad` through the *default* `ift`/`gmres`
path raises a runtime error on a 4-DOF stiff group.  The coupling
suite is otherwise green: 662 passed, 3 skipped, 1 failed — the single
failure is finding 4, a merged-state collision between the inert-knob
warnings and the solver-equivalence property test.  `tests/verification/`
is 172 passed, 1 deselected.

Checked and sound: the IFT `custom_jvp` itself (reverse *and* forward
mode match the analytic fixed-point derivative to float32 across every
acceleration, iteration mode, norm and both linear solvers, on 4-DOF
and 80-DOF groups); the forward state returned by `fori` and `ift`
(480-cell sweep, zero disagreements); all five accelerations reaching
the same fixed point; non-float leaf handling.

## Findings

### CRITICAL — the `atol` dead band is live under `convergence_norm="l2"`, where the docs and the library's own warning say it is ignored; groups below it report `converged=True, residual=0.0` after one pass

**What breaks:** two nodes in a group, one field `tau_big` of order 1
that converges in one pass and one field `tau_small` of order 1e-9 with
a Gauss-Seidel contraction of 0.5.  Defaults throughout
(`convergence_norm="l2"`, `atol=1e-8`, `solver="ift"`),
`tolerance=1e-10`, `max_iterations=40`.  The group returns
`tau_small = 1e-9` where the fixed point is `2e-9` — 50% error — and
reports it as converged with a zero residual.

**Evidence:**

    PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu python repro_atol_deadband.py

    (1) every field below the default atol=1e-8, default l2 norm
     field scale        exact     returned   rel err iters  residual conv
           1e-06        2e-06        2e-06     0.00%    23  0.00e+00 True
           1e-08        2e-08        2e-08     0.00%    23  0.00e+00 True
           1e-09        2e-09        1e-09    50.00%     1  0.00e+00 True
           1e-12        2e-12        1e-12    50.00%     1  0.00e+00 True

    (2) one O(1) field and one O(1e-9) field in the same group
        tau_big   = 1  (exact 1.0,       rel err 0.00e+00)
        tau_small = 1e-09  (exact 2e-09, rel err 50.00%)
        diagnostics = {'iterations': 1, 'residual': 0.0, 'amplification': 1.0,
                       'error_estimate': 0.0, 'bound_valid': True,
                       'gradient_error_bound': 0.0, 'converged': True}

    (3) the warning the library emits for the knob that would fix it
        CouplingGroup.atol=1e-12 is ignored under convergence_norm='l2',
        which tests the global L2 norm of the state change against
        tolerance alone. ...

    (3b) and with atol lowered, the same group converges correctly:
        tau_small = 2e-09  (exact 2e-09)  iterations=23

**Why it happens:** `graph_manager.py:1266` passes `group.atol` into
`coupling_residual_l2`, whose 0.4.0 signature gained an `atol`
parameter (`acceleration.py:76-81`) forwarded to `_scaled_change`
(`acceleration.py:51-74`).  There `active = (ref > atol)` with
`ref = max|v|` over the *whole field*, and an inactive field
contributes exactly zero to the norm.  When every active field is
already at its fixed point the norm is 0.0 and the first `_met` test
passes.

The differential against `main` is the sharp part.  On `main`
`coupling_residual_l2` took no `atol` and returned the unscaled
`||dx||` (~7e-10 on the first pass here), so the same group is
*fixable*: tighten `tolerance` below that and it iterates.  On `HEAD`
the residual is exactly `0.0`, so no value of `tolerance` — not even
`0.0` — changes the verdict, and the one knob that does fix it
(`atol`) is the one the library tells the user is ignored.

Two pieces of the release then point the user away from the fix:
`CouplingGroup.atol`'s docstring ("Read **only** by the `"mixed"` and
`"interface"` norms; setting it away from its default under
`convergence_norm="l2"` is inert and warns"), and `_INERT_RULES`'
first rule (`group.py:525-529`, `live=lambda g: g.convergence_norm !=
"l2"`), whose source comment asserts "`coupling_residual_l2` does not
take them".  It takes `atol`.  (`rtol` genuinely is inert under L2 —
`coupling_residual_l2` hard-codes `rtol=1.0` at
`acceleration.py:118-120` — so the rule is right about one of its two
fields and wrong about the other.)

**What would make this a non-issue:** (a) if L2's dead band were
documented on the user-facing surface — it is documented only on
`coupling_residual_l2`'s own parameter, and contradicted on
`CouplingGroup.atol`; (b) if the user could simply set `atol` — under
this repo's own `filterwarnings = ["error"]` pytest setting, and for
any project that promotes `UserWarning`, `atol` cannot be set at all
under the default norm; (c) if fields below 1e-8 were unrealistic —
this is a microrobotics framework and `coupling_residual_mixed`'s own
docstring cites a 1.7e-05 N drag force as the motivating case; case (2)
above needs only *one* such field in an otherwise O(1) group;
(d) if `strict_convergence=True` caught it — it does not.  Measured
(`repro_strict_convergence.py`): the same group with
`strict_convergence=True` runs silently and returns `tau = 1e-9`
against an exact `2e-9`, reporting `{'iterations': 1, 'residual': 0.0,
'amplification': nan, 'error_estimate': 0.0, 'bound_valid': False,
'gradient_error_bound': inf, 'converged': True}` — `converged=True`
beside `bound_valid=False` and `gradient_error_bound=inf`, which is
self-contradictory and is the only hint the caller gets.  The guard
itself works: the same graph at `b=0.99, max_iterations=3` does raise.

**Suggested fix:** decide which is true and make the other match.
Either drop the `atol` argument from `coupling_residual_l2` (restoring
`main`'s behaviour and making the inert rule correct), or keep the dead
band, split the `("atol", "rtol")` rule so that only `rtol` is inert
under L2, and document the dead band on `CouplingGroup.atol`.  The
first is the safer choice for the release: it removes a silent-wrong-
answer path, and a group that wants a dead band can ask for the
`"mixed"` norm.  Risk of the first: any fixture relying on L2's dead
band to stop a legitimately-zero field from blocking convergence would
start iterating again — `test_a_field_that_is_legitimately_zero_does_not_block_the_group`
is the one to check (it should survive, since `_scaled_change`'s
`scale > 0` guard already handles an exactly-zero field).

### HIGH — `jax.grad` through the default `solver="ift"` / `linear_solver="gmres"` path raises a runtime error on a 4-DOF stiff coupling group

**What breaks:** a two-node group with a 4-element flat state and
contraction modes `(0.999, 0.2)`.  The forward step is fine.
`jax.grad` of a loss over `run_scan(1, ...)` aborts with lineax's
`_EquinoxRuntimeError: A form of iterative breakdown has occured in the
linear solve`.  `jax.jvp` on the same graph returns the correct number,
and `linear_solver="dense"` returns the correct number, so it is the
*adjoint* (transpose) GMRES solve that fails.

**Evidence:**

    PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu python repro_gmres_breakdown.py
    linear_solver=dense  jax.grad -> 1.260000  (exact 1.260000)
    linear_solver=gmres  jax.grad -> RAISED JaxRuntimeError
        equinox._errors._EquinoxRuntimeError: A form of iterative
        breakdown has occured in the linear solve. Try using a different
        solver for this problem or increase `restart` if using GMRES.

    # scan over the slow mode (repro_overrelaxation_and_iterations.py's sibling,
    #  scratch r11_gmres_scan.py):
     rho_slow    cond~   grad/gmres    jvp/gmres   grad/dense  env dense
          0.99       80        1.251        1.251        1.251      1.251
         0.995      160        1.252        1.252        1.252      1.252
         0.998      400    BREAKDOWN        1.255        1.255      1.255
         0.999      800    BREAKDOWN         1.26         1.26       1.26
        0.9995     1600         1.27         1.27         1.27       1.27

**Why it happens:** `_ift_linear_solve` (`graph_manager.py:635-748`)
drives `lx.GMRES` through `jax.lax.custom_linear_solve` with
`restart = min(n, 50)` and `atol = 1e-8 + rtol*max|b|`,
`rtol = max(1e-6, 100*eps)` = 1.19e-5 in float32.  Here `n = 4`, so
`restart = 4` is already the full space; the breakdown is lineax
declaring a happy-breakdown it cannot certify against that tolerance on
the transposed operator.  The non-monotonicity in `rho_slow` (0.9995
passes, 0.998 and 0.999 fail) says it is triggered by the particular
cotangent, not by a stiffness threshold — so it is not predictable from
the graph.  The docstring at `graph_manager.py:659-676` anticipates
exactly this failure and claims the `atol` scaling resolves it; on the
transpose path it does not.

**What would make this a non-issue:** (a) if it were flaky — it
reproduces identically on every run; (b) if the configuration were
exotic — `solver="ift"` and `linear_solver="gmres"` are both defaults,
the group has four degrees of freedom, and a mode with lambda near 1 is
the case the IFT machinery exists for; (c) if the message told the user
what to do — it names "increase `restart`", which is already at full
rank, and never mentions `linear_solver="dense"` or
`MADDENING_IFT_DENSE_SOLVE=1`, both of which do fix it.

**Suggested fix:** wrap the lineax solve so a reported failure falls
back to the dense LU (`_dense`) for small `n` instead of raising, or
pass `throw=False` and check `result` so MADDENING can raise its own
message naming `linear_solver="dense"`.  Risk: a silent fallback hides
a genuinely singular `I - dF/dx`; gate the fallback on `n` being small
enough for dense to be affordable and emit a warning.

### HIGH — the 0.4.0 "error bound" is not a bound; two independent mechanisms, measured at 122x and at exactly `1/omega`

`coupling_diagnostics()["error_estimate"]` is documented as "an
estimate of `||x - x*||` ... how far the returned state is from the
fixed point", `_fixed_point_while` calls `converged=True` "within
`threshold` of the fixed point", and `coupling_diagnostics`' own
docstring says "Since 0.4.0 it is also a *bound on the distance to the
fixed point*".  Both mechanisms below produce `bound_valid=True` and
`converged=True` at a state further from the fixed point than the
threshold.

**(a) Multi-mode contraction: `rho` reads the mode that dominates the
step, not the mode that dominates the remaining error.**

Two-node linear cycle with contraction modes `(0.999, 0.2)` and
forcing `(1e-5, 1.0)`; exact fixed point `(1e-2, 1.25)`.  Default L2
norm, `solver="ift"`, `max_iterations=60`.

    PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu python repro_error_bound_two_mode.py
    tol        iters  residual     amp    error_estimate bound_valid conv | TRUE dist   est/true
    1e-02      4      1.8108e-03   1.25   2.2628e-03     True        True | 1.1493e-02  0.1969
    1e-03      5      3.6219e-04   1.25   4.5274e-04     True        True | 1.1266e-02  0.0402
    1e-04      6      7.3294e-05   1.254  9.1889e-05     True        True | 1.1246e-02  0.0082

At `tolerance=1e-4` the reported estimate is 9.19e-05 and the true
distance, measured in the group's own norm against the analytic fixed
point, is 1.12e-02: the "bound" is **122x too small**, and the returned
state is 112x the tolerance it was tested against.  While the fast mode
dominates the step, both terms of
`rho = max(r_k/r_{k-1}, sqrt(r_k/r_{k-2}))` read 0.2, so the two-step
guard does not help; the remaining error is already all in the 0.999
mode.

**(b) Over-relaxation: the geometric series sums the *un-relaxed*
steps.** `fixed` relaxation advances by `omega*(F(x)-x)` but the
residual fed to the bound is `||F(x)-x||`, so the sum of remaining step
lengths is short by exactly `omega`:

    PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu python repro_overrelaxation_and_iterations.py
     omega iters    residual     err_est   TRUE dist  est/true conv
      0.50    79  2.2443e-03  4.4270e-02  2.2127e-02    2.0007 False
      1.00    79  3.0884e-05  3.0625e-04  3.0928e-04    0.9902 False
      1.50    57  1.4268e-05  9.6281e-05  1.4239e-04    0.6762 True
      1.80    46  1.6891e-05  9.7037e-05  1.6880e-04    0.5749 True
      1.95    42  1.7334e-05  8.8699e-05  1.7484e-04    0.5073 True

`est/true` tracks `1/omega` to three figures across the row, which
identifies the mechanism exactly.  At `omega=1.95, tolerance=1e-4` the
group reports `converged=True` at 1.75x its own tolerance.

**Why it happens:** `error_amplification`
(`acceleration.py:245-300`) estimates a *single* contraction rate from
the last two or three residuals and `_fixed_point_while` applies
`r_k/(1-rho)` (`graph_manager.py:486-498`, `_met`).  The derivation
needs (i) one rate governing the whole tail and (ii) the step lengths
that are summed to be the *actual* steps.  (a) breaks (i); (b) breaks
(ii).

**What would make this a non-issue:** (a) the caveat already in
`_fixed_point_while`'s docstring is about the triangle inequality
failing when the iterate's *scale* moves by orders of magnitude — here
the scale is constant at 1.25 across the whole tail and the map is
exactly linear; (b) `bound_valid` is documented as the escape hatch —
it reports `True` in every row above; (c) `MADD-ANO-005`'s recorded
`residual_risk` names "a mode that has not been excited by the
iterations taken" and "a strongly non-linear F" — in (a) both modes are
excited from the first pass and `F` is affine, and (b) is a plain
algebra slip, so neither case is covered by what is on record;
(d) the existing coverage does not reach it: the property test
(`tests/property/test_coupling_error_bound.py`) compares against a
reference at `tolerance/1000` with a 4x slack, and its generated graphs
do not produce a `0.999`/`0.2` spectrum.

`gradient_error_bound` is the same number, so it understates the
adjoint/finite-difference gap by the same factors;
`repro_gradient_error_bound.py` is the direct analogue of
`test_the_gradient_trust_bound_bounds_the_adjoint_finite_difference_gap`
on the two-mode fixture (it currently aborts on the GMRES breakdown
above, so run it with `MADDENING_IFT_DENSE_SOLVE=1`).

**Suggested fix:** for (b), multiply the amplification by the actual
step scale (`omega` for `fixed`; for Aitken/IQN the ratio
`||x_{k+1}-x_k|| / r_k` is available in the carry) — cheap and exact.
For (a) there is no cheap exact fix; the honest options are to take
`rho` as the max over a window of ratios rather than the last one
(conservative, costs iterations), or to weaken the documented claim
from "bound" to "estimate under a single-mode assumption" and say so in
`coupling_diagnostics`, `_fixed_point_while` and MADD-ANO-005.  Risk of
the windowed `rho`: more iterations on every group, which is the cost
the 0.4.0 work already paid once.

### MEDIUM — `tests/property/test_coupling_error_bound.py::test_the_bound_is_the_same_on_both_solvers` fails on this commit whenever Hypothesis draws `linear_solver="dense"`

**What breaks:** the test flips a drawn group's `solver` between
`"ift"` and `"fori"` with `dataclasses.replace`.  The strategy is
careful never to *draw* an inert knob (`tests/property/strategies.py:721-724`
gates `linear_solver` on `solver == "ift"`), but the flip to `"fori"`
makes a drawn `linear_solver="dense"` inert, which emits the 0.4.0
inert-knob `UserWarning`, which `pyproject.toml`'s
`filterwarnings = ["error"]` turns into a failure.  It is a merged-state
collision: the strategy branch, the inert-knob branch and this property
test are each correct on their own.

**Evidence:** it surfaced in my own run of the suite —

    1 failed, 491 passed, 4 deselected in 1263.46s (0:21:03)
    ...
    E           Explanation:
    E               These lines were always and only run by failing test cases:
    E                   src/maddening/core/coupling/group.py:404
    E               ... linear_solver='dense'),), ... solver='ift',

and reduces to three lines with no Hypothesis at all:

    PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu python repro_property_test_solver_flip.py
    drawn group  : ift dense -> built quietly
    replace(solver='ift') -> ok
    replace(solver='fori') -> UserWarning: CouplingGroup.linear_solver='dense'
      is ignored under solver='fori', which differentiates straight through
      the iterates and solves no tangent system. ...

**Why it happens:** `_warn_about_inert_settings` (`group.py:395-408`)
runs on every construction including a `dataclasses.replace`, and the
`linear_solver` rule is `live=lambda g: g.solver == "ift"`
(`group.py:589-598`).  `strict_convergence=True` would trip the same
way; the strategy pins it to `False` for an unrelated reason
(`strategies.py:725-729`), which is the only thing keeping the failure
rate down.

**What would make this a non-issue:** (a) if it were rare enough not to
matter — the drawn `linear_solver` is `sampled_from` two values under
`solver="ift"`, so roughly a quarter of drawn groups carry it, and CI
runs the `ci` profile at 200 examples; my single `dev`-profile run hit
it; (b) if PR 59 already fixed it — PR 59 touches tests, so it may; it
was open when I started and I audited the pre-merge state.  Worth
re-checking after it lands.

**Suggested fix:** in the test, reset `linear_solver` to its default
when flipping to `"fori"` (the knob is irrelevant to what the test
asserts), or flip with a `warnings.catch_warnings` suppression.  Risk:
none — the test is asserting solver-invariance of the *diagnostics*,
not of `linear_solver`.

### LOW — `coupling_diagnostics()["iterations"]` differs between `fori` and `ift` at the cap, contradicting the documented "nothing here moves"

**What breaks:** any group that exhausts `max_iterations`.  `fori`
reports `max_iterations`; `ift` reports `max_iterations - 1`.  A caller
who tests `diag["iterations"] >= group.max_iterations` to detect a
cap-out — the obvious idiom — never sees it fire under the default
solver.

**Evidence:**

    PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu python repro_overrelaxation_and_iterations.py
    ### C. `iterations` at the cap: fori reports cap, ift reports cap-1
        cap=6 none      fori=(6, False)  ift=(5, False)
        cap=6 aitken    fori=(6, False)  ift=(5, False)
        cap=6 fixed     fori=(6, False)  ift=(5, False)

A 480-configuration sweep (5 accelerations x 3 norms x 2 iteration
modes x 4 caps x int-leaf on/off) found `iterations` to be the *only*
field the two solvers disagree on — `tau`, `residual`, `converged` and
the integer leaf agreed everywhere.

**Why it happens:** `_fixed_point_while` returns `n_iters` = the number
of `while_loop` bodies, capped at `max_iter - 1`
(`graph_manager.py:569`, with the `i < max_iter - 1` guard at line 504);
the fori branches seed `icount = 1.0` for the pre-loop pass and add one
per non-converged body (`graph_manager.py:1637` and siblings), giving
`max_iters`.  Both are defensible conventions; they are not the same
one.  `coupling_diagnostics`' docstring says the two solvers "derive
both values the same way, so nothing here moves when a graph migrates
between them", and `test_both_solvers_report_the_same_bound` asserts
`a["iterations"] == b["iterations"]` — but only on a converged exit,
where they do agree.

**What would make this a non-issue:** if `iterations` were documented
as solver-dependent — it is documented as the opposite.  Note it is
*not* a forward-state defect: the states agree bit-for-bit.

**Suggested fix:** make `_fixed_point_while` return `n_iters + 1` (the
pre-loop pass is a pass) and extend
`test_both_solvers_report_the_same_bound` with a cap-exhausting case.
Risk: any fixture that pins the current `ift` count.

### LOW — two inert-knob predicates miss cases decided at build time

**What breaks:** knobs that turn silently, with no warning.

1. `subcycling=True` on a group whose nodes all share a timestep.
   `graph_manager.py:1069-1082` sets `use_subcycling = False` ("uniform
   timestep, no subcycling needed"), and `n_waveform` at line 1314 and
   the interpolation flags at 1083-1084 then read that build-time
   value.  The predicate is `live=lambda g: g.subcycling`
   (`group.py:568-588`), so `waveform_iterations=5` and
   `boundary_interpolation="quadratic"` are accepted in silence and do
   nothing.  This is the case the audit brief flags: liveness decided
   from the node timesteps, not from the group's declared fields.
2. `max_iterations=1` returns from `_run_coupling_inner` before the
   accelerator is built (`graph_manager.py:1329`), so `acceleration`,
   `relaxation`, `jacobian_reuse`, `accelerated_fields` and
   `linear_solver` are all inert.  This one *is* decidable from the
   declared fields.  Structurally the same branch means that
   `solver="ift"` at `max_iterations=1` never calls `_ift_solve` at
   all, so the step differentiates straight through the single pass
   rather than "via the implicit function theorem at the fixed point"
   as `CouplingGroup.solver`'s docstring says.  (`strict_convergence`
   *is* honoured on this path, `graph_manager.py:1356`.)

**Evidence:**

    PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu python repro_inert_knob_predicates.py
    === (a) subcycling=True but uniform timesteps: use_subcycling -> False
        warnings: NONE
        waveform_iterations=1 -> tau=4.685589790344238; =5 -> tau=4.685589790344238 (identical: True)
    === (b) max_iterations=1 short-circuits every acceleration knob
        warnings for relaxation=0.3 @ acceleration='fixed', cap=1: NONE
        relaxation=0.3 -> tau=1.0; 1.9 -> tau=1.0

**What would make this a non-issue:** if either configuration were
rejected earlier — `validate()` only complains about *mixed* timesteps
*without* subcycling (`graph_manager.py:3070-3082`), so
uniform + `subcycling=True` is legal and clean; `max_iterations=1` is
explicitly supported ("a legitimate 'one staggered pass, no iteration'
request", `graph_manager.py:1330`).

**Suggested fix:** for (2), add `g.max_iterations > 1` to the
`relaxation` / `jacobian_reuse` / `accelerated_fields` predicates.
(1) cannot be decided in `__post_init__` — the group does not know the
node timesteps — so it belongs in `add_coupling_group` or `validate()`,
which do.  Risk: a warning for a group whose timesteps later diverge;
`validate()` is the right place because it runs after the graph is
assembled.

## Unverified suspicions

* `fori` with `acceleration` in `("aitken", "fixed")` flattens *every*
  field of the group's nodes (`accel_fields` is `None`, so
  `flatten_coupled_state` takes `sorted(state[nn].keys())`), which
  promotes an integer leaf into the relaxed vector and casts it back on
  unflatten.  I could not make this corrupt anything, because
  `one_pass` recomputes every node from `_pre(nn)` (the pre-step
  state), so a counter holds the same value in every pass and
  relaxation between two equal numbers is a no-op.  A node whose
  integer leaf depends on a *boundary input* would break that; I did
  not build one.
* `aitken` on a strongly-coupled group (`b=0.9`) failed to converge in
  200 iterations at `tolerance=1e-9` where `acceleration="none"`
  converged in 138.  I believe this is the two-pass exit guard meeting
  float32 residual noise rather than a defect, but it is the opposite
  of what "acceleration" implies and I did not chase it to ground.
* `iqn-ils`/`iqn-imvj` at `max_iterations=6`, `b=0.99` stopped at
  different passes under the two solvers (`fori` 4, `ift` 5) with
  states 3e-6 apart — consistent with float32 rounding differences in
  the two flatten paths rather than a structural divergence, but I did
  not prove that.
* `mapping.py` / `mapping_spec.py` had a light pass only (they are
  serialisation plumbing and all six of the brief's questions are about
  the solver).  Their tests are green; I did not attack the asset-path
  hardening.

## What I checked and found sound

* **The IFT `custom_jvp` gives the right derivative.**  On a two-node
  affine cycle whose fixed-point derivative is known in closed form
  (2.0), `jax.grad` and `jax.jvp` both return 2.000000 for every
  combination of `{none, aitken, fixed, iqn-ils, iqn-imvj}` x
  `{gauss-seidel, jacobi}` x `{l2, mixed, interface}` x
  `{gmres, dense}`, matching a central difference to float32.  On an
  80-DOF group with a dense random coupling matrix the adjoint matches
  `sum((I-K)^{-1}1)` to 8.3e-10 relative under both linear solvers —
  so the GMRES `restart = min(n, 50)` guard is doing its job on the
  forward tangent path.
* **The forward state is solver-invariant.**  480 configurations
  (accelerations x norms x iteration modes x caps x integer-leaf),
  `fori` vs `ift`: `tau`, `residual`, `converged` and the integer leaf
  agreed in every cell.  The D2 regression (`ift` returning the
  successor of the measured iterate) is genuinely closed.
* **The state returned is the state measured.**  On a criterion exit
  `x_star = x_meas` and `final_res` is the loop's own measurement of
  it; at the cap one extra `step_pure` measures `x_star` itself
  (`graph_manager.py:545-568`).  The fori `_merge` keeps `s_cur` on the
  converging pass, matching.  `max_iterations=1` is the documented
  exception (the residual describes the pass's input state), and for a
  contraction that is conservative.
* **All five accelerations reach the same fixed point.**  `b=0.9`,
  `tolerance=1e-9`, cap 200: every acceleration x iteration mode lands
  within 2.2e-4 relative of the analytic `x* = 10`, with IQN arriving
  in 5-7 iterations against 138 for `none`.  No acceleration converges
  to a *different* point.
* **`coupling_diagnostics` field consistency.**  `error_estimate ==
  residual * amplification` reproduces the solver's own `_met`
  computation exactly; `bound_valid` is `amp >= 1.0` and matches
  `error_amplification`'s `0.0`-means-rejected convention;
  `amplification` is `nan` and `gradient_error_bound` `inf` exactly
  when rejected; the threshold used (`1.0` for mixed/interface,
  `tolerance` for l2) matches `conv_threshold_value`
  (`graph_manager.py:1281-1284`).  The fields are internally
  consistent — the problem is what the quantity means (finding 3), not
  how it is assembled.
* **The other seven inert-knob predicates** (`tolerance`, `relaxation`,
  `jacobian_reuse`, `accelerated_fields`, `boundary_interpolation`,
  `linear_solver`, `strict_convergence`) match their read sites for
  `max_iterations > 1`; `rtol` is genuinely dead under L2
  (`coupling_residual_l2` hard-codes `rtol=1.0`).
* **Non-float leaves** travel correctly: `float_image` /
  `from_float_image` round-trips, and an `int32` counter comes back as
  `int32` with the same value under both solvers in all 480 sweep
  cells.

## Test runs

    cd /home/nick/MSF/msf/MADDENING-wt/audit/coupling
    PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
      python -m pytest tests/core/test_coupling_error_bound.py \
        tests/core/test_coupling_solver_equivalence.py \
        tests/core/test_coupling_convergence_reporting.py \
        tests/property/test_coupling_group_inert_knobs.py -q -p no:cacheprovider -rs
    -> 171 passed, 3 skipped in 177.78s
       (skips: "cap 1 returns before the accelerator is built" — explained)

    PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
      python -m pytest tests/core/test_coupling.py \
        tests/core/test_coupling_acceleration.py \
        tests/core/test_coupling_convergence.py \
        tests/core/test_coupling_fixture_invariants.py \
        tests/core/test_coupling_group_serialisation.py \
        tests/core/test_coupling_group_validation.py \
        tests/core/test_coupling_helpers.py \
        tests/core/test_coupling_ift_*.py \
        tests/core/test_coupling_multiphysics_imvj.py \
        tests/core/test_coupling_non_float_leaves.py \
        tests/core/test_coupling_predictor.py \
        tests/core/test_coupling_subcycling.py \
        tests/core/test_coupling_while_default.py \
        tests/core/test_flux_coupling.py tests/core/test_mapping.py \
        tests/core/test_mapping_spec_hardening.py \
        tests/core/test_mapping_spec_serialisation.py \
        tests/property/test_coupling_error_bound.py -q -p no:cacheprovider -rs
    -> 1 failed, 491 passed, 4 deselected in 1263.46s
       (the failure is finding 4; full log in
        pytest_coupling_mapping_property.log)

    PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
      python -m pytest tests/verification/ -q -p no:cacheprovider -rs
    -> 172 passed, 1 deselected in 1186.19s

Totals across the three runs: 663 passed, 3 skipped, 1 failed.

Findings 1, 2, 3, 5 and 6 are gaps in coverage, not red tests.
Finding 4 *is* a red test.

## Files

* `repro_atol_deadband.py` — finding 1
* `repro_gmres_breakdown.py` — finding 2
* `repro_error_bound_two_mode.py` — finding 3(a)
* `repro_overrelaxation_and_iterations.py` — finding 3(b) and finding 5
* `repro_gradient_error_bound.py` — finding 3, gradient-bound analogue
* `repro_property_test_solver_flip.py` — finding 4
* `repro_inert_knob_predicates.py` — finding 6
* `repro_strict_convergence.py` — the `strict_convergence` check for finding 1
* `repro_gmres_stiffness_scan.py`, `repro_solver_sweep_480.py`,
  `repro_gradient_sweep.py`, `repro_gradient_80dof.py` — the
  "checked and found sound" evidence
* `harness.py` — shared two-node fixture
* `raw_output.txt` — pasted output of every reproducer
* `pytest_coupling_mapping_property.log`, `pytest_verification.log` — test runs
