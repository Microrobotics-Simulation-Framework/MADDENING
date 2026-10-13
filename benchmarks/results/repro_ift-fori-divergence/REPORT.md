# `fori` vs `ift`: minimal reproduction of the D2 divergence

Branch `repro/ift-fori-divergence`, worktree
`/home/nick/MSF/msf/MADDENING-wt/repro/ift-fori-divergence`, off
`origin/release/0.4.0` @ `c8cc774`.  jax 0.11.0, lineax 0.0.7, CPU.
Investigating `plans/MIME_vs_MADDENING_040_BASELINE.md` 4.2.

## Lead

**Yes, it reproduces outside MIME**, in a two-node MADDENING-only graph,
at the same 2.14%.

**The smallest ingredient set is: a coupling group whose convergence
criterion is met before the fixed point is reached.**  That is the
whole list.  It needs

- no int32 state leaf,
- no `accelerated_fields`,
- no `convergence_norm="interface"`,
- no acceleration at all (`"none"` reproduces it),
- no IQN, no Schwarz, no subcycling, no flux coupling, no float64.

Two nodes, two edges, an affine map, `acceleration="none"`,
`convergence_norm="l2"`: 2.14%.

**The standing hypothesis is wrong.**  The two paths do *not* build
different fixed-point vectors.  They build the same one, in the same
order, at the same dtype, and they are not converging to different
fixed points.  They are reporting *the same iteration one pass apart*:

    ift's returned state  ==  F( fori's returned state )

exactly, to the last bit, where `F` is the group's own Gauss-Seidel
pass.  Verified in `tests/core/test_coupling_solver_equivalence.py
::test_the_solver_gap_is_exactly_one_more_pass_of_the_group_map`.

## The mechanism

On an exit that **met the convergence criterion**:

- `fori` (`_run_coupled_block_impl`, the `fori_loop` branches,
  `src/maddening/core/graph_manager.py` ~1500-1690) computes
  `new_converged = converged | (residual <= conv_threshold)` and then
  `_merge(s_cur, s_new, new_converged)`, whose tree map is
  `jnp.where(new_converged, old, new)`.  On the pass where the
  criterion first holds it therefore **discards that pass's update and
  keeps `s_cur`** -- the iterate whose residual was measured.
- `ift` (`_fixed_point_while`, same file ~259-467) measures the
  residual of the iterate the body *starts* from and returns the
  iterate the body *produced*.  `cond` sees the sub-threshold residual
  only on the next test, by which time `x_star` is **one update
  further along**.

So the two returned states differ by exactly one residual.  Both
solvers then report that same residual and `converged=True` -- the
divergence is completely silent.  (This one-update lag is itself
already known: `coupling_diagnostics`' docstring states it, and
`tests/core/test_coupling_convergence_reporting.py` carries a strict
xfail for its effect on the *reported residual*.  What was not
appreciated is how large the lag is in *value* terms when the criterion
fires early.)

One residual is a percent-sized relative error whenever the criterion
is loose in relative terms, and **every convergence criterion in the
group is absolute at small magnitudes**:

| norm | threshold | residual it compares |
|---|---|---|
| `"l2"` | `tolerance` | `norm(dx)`, unscaled |
| `"mixed"` / `"interface"` | hard-coded `1.0` | `dx / (atol + rtol*abs(v))` -- i.e. `dx/atol` once `abs(v) << atol/rtol` |

With the default `atol=1e-8`, an interface quantity of magnitude
`1e-9`..`1e-5` -- MIME's D2 drag force is `1.7e-05` N -- satisfies the
criterion on its **first pass**, while the iteration is still `rho`
away from its fixed point.  `fori` then returns the first-pass state
and `ift` the second, and the gap is `rho`.

### Why the gradient goes with it

`ift`'s analytic adjoint is the *correct* IFT gradient -- of a fixed
point its forward never reached.  In the reproduction, `rho = 0.0214`:

| quantity | value | |
|---|---|---|
| `ift` analytic `d tau / d gain` | `1.0218679905` | `= 1/(1-rho)`, the true fixed point's sensitivity |
| `ift` central FD of its own forward | `1.0213990268` | `= (1+rho)`, one pass short |
| relative disagreement | **4.59e-04** | `= rho^2/(1+rho)` |
| `fori` analytic vs its own FD | **9.5e-07** | round-off (float32 params) |

`fori` differentiates straight through its iterates, so it is exactly
consistent with whatever it computed, converged or not.  `ift` is
consistent with a fixed point it did not return.  MIME's measured
`7.2e-05` is the same quantity at its own `rho`.

`strict_convergence=True` does **not** catch this: it re-tests the same
residual that already passed.  Verified in `float64_numbers.log`.

## Why 4.2's two controls showed nothing

Both controls run against D2 were no-ops, which is why the result read
as "two genuinely different fixed points":

1. **`max_iterations` 8 -> 40 cannot matter for a criterion exit.**  The
   exit happens on the criterion, before the cap.  Raising the cap
   changes nothing on either solver.  The invariance is real; it just
   does not mean what it looked like.
2. **`tolerance` 1e-4 -> 1e-14 is not read at all under
   `convergence_norm="interface"`.**  `conv_threshold_value` is
   hard-coded to `1.0` for the `"mixed"` and `"interface"` norms
   (`graph_manager.py` ~1146).  The live knobs there are `atol` and
   `rtol`.  Tightening `tolerance` on an interface-norm group is
   literally a no-op -- the single most useful thing to take from this
   investigation.

Tightening the criterion that *is* live does close the gap: under the
L2 norm, `tolerance=1e-6` with `max_iterations=200` puts both solvers
on the true fixed point and both gradients back at round-off
(`test_tightening_the_live_threshold_closes_the_gap`), and the `atol`
sweep in `confirm.py` C goes 5.0e-01 -> 4.9e-04 -> 6.0e-08 -> 7.0e-15 as
`atol` goes 1e-8 -> 1e-12 -> 1e-16 -> 0.

**On a cap exit the two solvers are bit-identical** -- they spend the
same number of passes by construction.  The divergence exists only on a
converged exit.

## Ruled out, with evidence

| suspected ingredient | verdict |
|---|---|
| int32 `i_step` leaf in a group node's state | **innocent.** Rows with and without it are bit-identical across the whole 48-cell sweep (`minimal.py`). The `ift` path does hold integer leaves out of its fixed-point vector, but every node integrates from the pre-step state (`_pre(nn)`), so an integer field's value cannot depend on the iterate, and holding it out costs nothing. |
| explicit `accelerated_fields` vs auto-detect | **innocent.** Identical results; the auto-detector finds the same fields. |
| `convergence_norm="interface"` | **not required.** Same 2.14% under `"l2"` and `"mixed"`. It is, however, why `tolerance` looked inert, and its `atol`-dominated scaling is why MIME's group exits early with nobody loosening anything. |
| `acceleration="iqn-ils"` | **not required.** Same gap with `"none"`, `"fixed"`, `"aitken"`. |
| different leaf set / order / dtype promotion between the two flattens | **disproved.** `ift == F(fori)` to the last bit; a different vector could not do that. |
| under-convergence as the *explanation* | half right. Both solvers are under-converged in the same way; what differs is only which of two adjacent iterates each returns. |

Negative result worth recording: `iteration_mode="jacobi"` shows a 0%
gap **on `tau`** in the sweep -- not because the defect is absent, but
because under Jacobi the extra pass moves `disp` and not `tau` on this
particular two-node cycle.  Do not read that row as a fix.

## What reproduces it

Pinned test: `tests/core/test_coupling_solver_equivalence.py`
(33 passed, 2 xfailed -- the two strict xfails are the defects).

    cd /home/nick/MSF/msf/MADDENING-wt/repro/ift-fori-divergence
    PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
      /home/nick/MSF/msf/.venv/bin/python -m pytest \
      tests/core/test_coupling_solver_equivalence.py -q -p no:cacheprovider -rs

Exploration scripts kept beside this report (run with the same
`PYTHONPATH`; they enable x64 themselves):

- `repro2.py` -- first parametric scan (scale / rho / norm / atol / accel).
- `confirm.py` -- A: the 2.14% and `ift == F(fori)`; B: invariance to cap
  and tolerance; C: the `atol` sweep; D: the two gradients.
- `minimal.py` -- the 48-cell ingredient sweep (int leaf x
  accelerated_fields x norm x acceleration x iteration mode).
- `float64_numbers.log` -- the float64 numbers quoted above, plus the
  `strict_convergence` and cap-exit checks.

## What a fix has to decide (not done here -- out of scope)

Someone has to pick which of the two adjacent states is the contract:

- **`fori`'s** (the measured iterate): `converged=True` then means what
  it says, but the IFT adjoint is taken one pass away from where the
  forward stopped, so `ift` would need its adjoint evaluated at the
  state it returns, or the extra pass kept and re-measured.
- **`ift`'s** (one update further): closer to the fixed point, but
  `converged` then names a state whose own residual was never measured
  -- exactly the strict xfail already open in
  `test_coupling_convergence_reporting.py`.

Either way the two solvers must return the same state before `"fori"`
is removed, or every graph that migrates gets a silent numerical
change.  A separate, cheaper mitigation worth considering on its own:
`tolerance` being silently ignored under the `"mixed"` and
`"interface"` norms should at least warn.

## Caveats

- One step, one group, scalar fields.  I did not test whether an
  array-valued interface field, subcycling, waveform relaxation or a
  predictor adds a *second*, independent divergence mechanism on top of
  this one.  This one is sufficient to produce every number in 4.2, but
  it is not proof that nothing else is also happening in D2.
- I did not re-run MIME's D2 case; the attribution is by signature
  (2.14% forward gap, tolerance- and cap-invariant, analytic-vs-FD
  mismatch on `ift` only, `interface` norm with a ~1e-5 interface
  quantity), not by direct measurement.
- Node params go through `params_pytree`, which promotes Python floats
  to float32; that is why the float64 log shows `1.0214000009` rather
  than `1.0214` exactly.  It affects no conclusion.
- No timings measured; other agents were running.
