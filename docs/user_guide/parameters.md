# Graph parameters, calibration and system identification

The compiled step of a graph is

```
step_fn(state, external_inputs, params)
```

`state` is what the integrators advance, `external_inputs` is what the
outside world supplies each step, and `params` is every time-invariant
constant a node reads that someone might want to differentiate or change
at runtime — stiffness, mass, diffusivity, gravity.  `GraphManager.params`
holds the compile-time snapshot:

```python
gm.compile()
gm.params
# {"nodes": {"spring": {"stiffness": Array(30.), "damping": Array(2.), ...},
#            "ball":   {"gravity": Array(-9.81), "elasticity": Array(0.7), ...}},
#  "mappings": {}}
```

Every run method takes an optional `params=`; `None` means the snapshot.
An explicit pytree is a *traced input*: a new value takes effect without
recompiling, and `jax.grad` / `jax.jvp` / `jax.jacfwd` reach it — through
coupling groups too, because the implicit-function-theorem rule carries
the parameter dependence.

```python
p = jax.tree.map(lambda x: x, gm.params)
p["nodes"]["spring"]["stiffness"] = jnp.asarray(45.0)
gm.run_scan(1000, params=p)          # no recompile

def loss(p):
    final = gm._compiled_step(gm._state, gm._default_external_inputs(), p)
    return final["spring"]["position"] ** 2

jax.grad(loss)(gm.params)["nodes"]["spring"]["stiffness"]
```

## Which nodes take part

A node opts in by declaring a keyword-only `params` on `update` and reading
its constants from it:

```python
def update(self, state, boundary_inputs, dt, *, params=None):
    p = self.params if params is None else {**self.params, **params}
    k = p["stiffness"]
    ...
```

`SimulationNode.params_pytree()` decides which entries of `self.params`
appear in the pytree: every float-valued one by default (Python floats,
float arrays, lists of numbers); ints, bools, strings and dicts are
structural and stay on the recompile path.  Nodes on the 3-argument
contract keep working, but their constants are baked into the trace and
are **not** in `gm.params`.  `gm.nodes_without_params()` lists them, and
passing an entry for such a node (or a misspelled key) is a `ValueError`,
not a silently ignored leaf.

The same holds for a leaf that *is* in `gm.params` but that the compiled
step cannot read: an `initial_*` entry (only `initial_state()` reads it, from
the node), a parameter a node bakes into a static when it is constructed
(`HeatNode`'s `grid_points` on a non-uniform grid, `WaveletAdaptiveNode`'s
`mass` -- declared by `static_data_deps`), or geometry a node consumed in
`__init__`.  (A leaf only *this* graph does not exercise -- a ball's
`elasticity` with no table edge -- is not one of them: the node reads it as
soon as the input arrives, and the value carried in `gm.params` is then the
one used, so it is kept and serialised.)  Changing one in `gm.params` is a `ValueError` naming the leaf
and why, raised by every run method, `check_params`, `to_dict` and
`save_state`; before 0.4.0 the edit was
ignored by the step and then written out by `to_dict`, so the saved graph
reloaded as a different model.  To change such a value, rebuild the node
with it; `gm.reset_params()` drops the edit.  Whether the step reads a leaf
is decided from the node's `static_data_deps` declaration and, failing
that, from one trace of the compiled step and one of the node's own hooks
with every declared boundary input supplied (both taken only when a leaf
differs from its node's value, once per compile); a value that also reaches
the node is not refused here, because the node's own value is the
reference -- `PUT /graph/params` writes both, so it takes the decision
itself (see "Writing parameters over REST" below) -- and a traced leaf
(inside a fit or an FIM) is never compared.  An explicit `params=` pytree is not
refused either: it is never serialised, and a leaf the step ignores may be
one your own code consumes (a residual that seeds the initial state from
`initial_velocity`); the live leaves a partial pytree is completed from are
checked as `gm.params`.

A value that carries a floating dtype of its own (an array, a numpy
scalar) keeps it; a value that carries none (a Python float, a list of
them) is placed at JAX's canonical float precision — float32, or float64
under `jax_enable_x64`.  Nothing is narrowed below that, so a graph run
under x64 has float64 constants as well as float64 arithmetic.

A node that exposes boundary fluxes takes `params` there too and reads
the same constants from it:

```python
def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
    p = self.params if params is None else {**self.params, **params}
    ...
```

The graph passes the node's entry on every flux evaluation, so a
calibrated stiffness changes the force a flux edge *delivers*, not only
the node's own integration.  The same rule applies to
`compute_interface_correction(..., *, params=None)` for nodes that
recompute coupled interface cells (`HeatNode`).  A flux producer whose `update` takes
`params` but whose `compute_boundary_fluxes` does not is exactly the
trap the 2026-09 audit found; `verify_node` now fails it.

`verify_node` checks the contract for you: `params_consistent` (injected
params reproduce the baked step *and* fluxes), `params_gradient_finite`, and
`params_effective` (every trainable leaf actually influences the outputs,
fluxes included, the way a constructed value does — the check that catches
a constant still read from `self.params`, or split between the injected
value and a copy made in `__init__`, when the node rebuilds from
`to_dict()`).  See [verification](../developer_guide/verification.md).

### Writing parameters over REST

`PUT /graph/params/{node}` writes a value to the node as well as to
`gm.params`, and writing `node.params` rebuilds nothing the node derived
from the value when it was constructed.  So before anything is written the
server asks the graph whether the running node would use the new value:

* a leaf the compiled step (or the node's own hooks, with every declared
  boundary input supplied) reads from the injected params takes effect on
  the next step, without a recompile;
* a structural value (an int, a bool, a string, or any constant of a node
  on the 3-argument contract) is accepted when the node's hooks trace
  differently with it; the graph is marked dirty and the recompile uses it;
* otherwise, a value `initial_state()` reads -- an `initial_*` condition --
  is accepted and takes effect at the next `POST /sim/reset`;
* anything else is a **400** naming the parameter and why: a
  `static_data_deps` entry (`WaveletAdaptiveNode.mass`), or a value nothing
  the running node computes reads (`LBMPipeNode.pipe_radius`, baked into
  the wall mask; `propeller_x`; `initial_rho_liquid`, which `initial_state`
  reads from a copy).  Nothing in the request is written.  To change such a
  value, rebuild the node: `DELETE /graph/nodes/{node}`, then
  `POST /graph/nodes` with the new value.

A 200 also promises that the graph a save would reload runs what the
running graph runs, so two more questions are asked, of the whole request
at once and with the params `to_dict()` would carry (every other key's
live value included, which a fit may have moved).  The node's constructor
must take them: a `HeatNode` `thermal_diffusivity` past its Fourier
limit, or a pipe's `rho_gas` above its `rho_liquid` (alone or written
together with a new `rho_liquid`), is a 400.  And where the rebuilt node
holds anything its constructor derives differently from the running one,
both are traced on the same state and params and both initial states are
built; a write they compute differently with is a 400 -- `LBMPipeNode`'s
`G` crossing zero, which picks the single- or multiphase branch at
construction.  A non-finite number anywhere in the request is a 400 before
anything else, integers are bounded as in `POST /graph/nodes` (a 422), and
a value that would change the state's layout, or take it past the API's
state cap, is refused before a node of that size is built wherever its
`initial_state()` can be evaluated abstractly.

Before 0.4.0 such a write answered 200, was served by `GET`, was ignored by
every step (even after `POST /graph/compile`) and was saved by `to_dict()`
and `save_state()`, so the reloaded graph ran a different model.  The check
runs the node's code on a shallow copy that reads the new value, never on
the node itself; when no faithful copy can be made (a node holding a method
bound to itself) or the code raises, nothing is refused.  It detects "no
path at all": a value a node consumes at construction *and* reads again
later passes, so a node that bakes a parameter should declare it in
`static_data_deps`, which refuses it on every surface.  Likewise
`POST /checkpoint/load` refuses, and undoes, a checkpoint whose parameters
include another value of one the node consumed at construction (a
checkpoint of a pipe built with another radius); before, the load
succeeded and every later `/sim/step` failed.

### Live values, recompiles and partial pytrees

`gm.params` is populated by `compile()` and **survives a recompile**: a
calibrated leaf whose node, key, shape and dtype still exist is carried
over when you add an edge or an external input, replace a node, or the
profiler recompiles behind your back.  Leaves that no longer fit are
dropped with a `RuntimeWarning`.  `gm.reset_params()` is the explicit
way back to the constructor snapshot.  A checkpoint loaded before the
first compile compiles the graph so its params are not lost.

The graph's *state* survives the same recompile, and so does the
internal bookkeeping that goes with it: a multi-rate graph keeps its
sub-step phase and a coupling group keeps its predictor history and IQN
warm start, so a mid-run edit changes no number.  Only a change that
moves a node's rate divider restarts the phase, because the sub-step it
counts then means something else.  `gm.reset_state()` is the explicit
way to zero all of it.

A *partial* pytree passed to `gm.step(params=...)` / `gm.run_scan` /
`gm.run` is completed from the **live** `gm.params` (a missing node or
key keeps its calibrated value, not its constructor constant).  The raw
compiled step (`gm._compiled_step`, what the FMI sidecar calls) refuses
an incomplete pytree instead of guessing.

## `ParamSpec`: what an optimiser may do

Each leaf carries a `ParamSpec` (`maddening.core.params`):

| field | meaning |
|-------|---------|
| `trainable` | may an optimiser move it (default `True`; `initial_*` entries default to `False`) |
| `bounds` | physical range, `(lo, hi)` with `None` for open |
| `transform` | `None` (clip to bounds), `"log"` (`p = lo + exp(u)`, strictly above `lo`, or above 0 when `lo` is `None` -- `check` refuses anything else), `"logit"` (`lo < p < hi`) |

Nodes declare specs for their own constants in `param_specs()`
(`SpringDamperNode`: stiffness and mass are `log`-positive, damping is
`>= 0`).  A graph overrides any of them:

```python
from maddening.core.params import ParamSpec
gm.set_param_spec("spring", "mass", ParamSpec(trainable=False))

gm.trainable_mask()        # params-shaped pytree of bools
u = gm.unconstrain()       # optimiser coordinates (log k, log m, ...)
gm.constrain(u)            # back to physical values, always inside bounds
gm.check_params(p)         # ValueError naming the first leaf out of range
```

## System identification: `maddening.sysid`

```python
from maddening.sysid import (fim, fim_core, fit, observations_from_history,
                             windowed_loss)

init = {n: gm.get_node_state(n) for n in gm.node_names}
_, hist = gm.run_scan_with_history(1000)
obs = observations_from_history(init, hist)          # T = 1001 samples

loss = lambda p: windowed_loss(
    gm, p, obs, obs_fn=lambda h: h["spring"]["position"], window=50)
```

`windowed_loss` is teacher-forced: every window restarts from the
measured state, so gradients cannot compound over a long stiff rollout
(`mask_unconverged=True` drops, from the loss and from its gradient,
windows in which a coupling group exited at `max_iterations`
unconverged, including a window that diverged).

Before fitting, ask what the data can identify:

```python
report = fim(lambda p: residual(p), gm.params, mask=gm.trainable_mask())
report.rank                  # < len(param_names) => directions the data misses
report.crb                   # +inf for every parameter those directions spoil
report.least_identifiable()  # ("['nodes']['spring']['mass']", 0.58)
report.eigvecs[:, 0]         # the weakest direction, in relative coordinates
```

For a spring observed through position only, `k`, `c` and `m` enter as
`k/m` and `c/m`: scaling all three together is invisible, and the FIM's
weakest eigenvector is `(1, 1, 1)/√3`.  `rank` is 2 of 3, and all
three bounds are `+inf` — each parameter lies partly along the invisible
direction, so none of them is separately determined.  Read `rank` rather
than `cond` for that verdict: `cond` is `eigvals[-1] / eigvals[0]` and in
float32 rescaling the residual (by `noise_std`, say) can round the
smallest eigenvalue to zero and turn a large `cond` into `inf`, whereas
`rank`'s threshold scales with the matrix.  Freeze one of them, then fit:

`scale="relative"` — the default, and the coordinates the eigenvectors
above are in — multiplies each Jacobian column by the parameter's value,
so a parameter sitting at exactly `0.0` (`SpringDamperNode`'s
`initial_velocity` default) has no column at all and reads as
unidentifiable however well the data determine it; `report.zero_scaled`
names those.  `scale="nominal"` removes the problem by taking the column
scale from the parameter's `ParamSpec` instead — the width `hi - lo` of a
finite `bounds`, a scale and not a location, so a symmetric range around
zero is no longer a zero:

```python
gm.set_param_spec("spring", "initial_velocity",
                  ParamSpec(trainable=False, bounds=(-1.0, 1.0)))
report = fim(residual, gm.params, scale="nominal", specs=gm.param_specs())
report.value_scaled          # columns whose spec had no finite width
```

A spec with no finite width — `(0.0, None)`, `(None, None)`, a `"log"`
constant — has no nominal scale, so that column keeps the value scaling
(`p`, or `p - lo` under `"log"`) and is named in `report.value_scaled`: a
nominal report says which columns were answered from the spec and which
from the value, and a value-scaled zero still appears in `zero_scaled`.
Nothing falls back to an absolute `1.0`, which would put units back into
`cond` unannounced.  The `fim` docstring tabulates the policy per spec.

`specs` must mirror `params`: a nested dict for every dict level — and
for every namedtuple or dataclass level, keyed by field name, since
that is how JAX addresses a field — a `ParamSpec` at each leaf, and for
a list/tuple level either a list of specs by position or one `ParamSpec`
covering every position.  A dict where a leaf needs a `ParamSpec`
(`ParamSpec.to_dict()` output), a `ParamSpec` above a dict or record
level, a list of specs for a namedtuple, or a `specs` that is not a dict
is a `ValueError` naming the key path, and not only in `fim`:
`trainable_mask`, `unconstrain`, `constrain` and `check_bounds` read
`specs` through the same walk and refuse the same entries, where they
used to hand the leaf the default (trainable, unbounded) spec.

A key that matches *no* parameter is where the two differ.  The tree
maps ignore it, because `gm.param_specs()` declares specs for constants
that are not leaves of `gm.params` — a uniform `HeatNode`'s
`grid_points=None`, any constant spelled as a Python `int` — and
`gm.check_params` hands them exactly that tree; a misspelt key is
therefore not caught there.  `fim(scale="nominal")` refuses one whenever
it could have changed the report: a stray spec carrying a finite width
or a `"log"` offset from a non-zero lower bound is a `ValueError`, while
a stray spec whose column record is the default's (unbounded,
one-sided, `(0, None)` under `"log"`) is accepted, because the report
is bit-identical with or without it.  That is what lets
`specs=gm.param_specs()` through for the graph it came from, and it is
also why slicing `params` to a sub-tree usually needs no matching slice
of the specs: the node's whole spec dict works unless an entry the
sub-tree does not reach carries a width.  A leaf *without* an entry
still gets the default spec and is named in `value_scaled`; `{}` remains
the explicit "no leaf has a declared width".

### Asking the same question inside a loop

`fim` is the reporting path: it reads the answer back to the host so it
can raise on a non-finite matrix, warn when the verdict rests on
rounding, and name the parameters `scale="relative"` found at zero.  For
a control loop, `fim_core` is the same computation with none of that —
it returns device arrays, reads nothing back, and traces:

```python
core_fn = jax.jit(functools.partial(fim_core, residual_fn))
core = core_fn(params)                       # no host sync at all
ok = core.finite & ~core.precision_limited & (core.crb[i] < tol)
```

**Tracing freezes what the residual reads.**  A residual usually takes a
few leaves as its argument and reads the rest of the model from outside
it — `gm.params` for the leaves it does not fit, an attribute of `self`,
a window of data.  Tracing reads those values once and compiles them in
as constants.  So `fim` re-traces on every call by default, and every
call answers for the residual as it stands then; a 200-step
spring-damper rollout measured ~230 ms per call that way on four pinned
CPU cores.  `fim(..., reuse_trace=True)` keeps the compiled Jacobian
across calls with an equal residual function (plus the same scale,
column set and nominal record) and costs ~1 ms warm — but it is only
correct for a *pure* residual, one that depends on its argument and
on nothing that can change between calls.  With it on, a residual that
reads `gm.params` reports the first call's matrix after those leaves
change, and a bound method (equal across attribute accesses) reports
the first call's matrix after its object changes.  Both are silent.

The same is true of `fim_core` under your own `jax.jit`: `core_fn`
holds what `residual_fn` read when it was first traced.  Pass anything
that changes through the argument, or build a new `core_fn` when it
changes.

`FIMCore` carries the same verdicts as `FIMReport` — `rank`, `cond`,
`crb`, `zero_scaled`, `value_scaled` — plus `finite`, which is what `fim` raises on, and
`precision_limited`, which is what it warns on.  Read them as a third
outcome rather than a refusal: a `precision_limited` core means *verdict
unavailable*, not *unidentifiable*.  `crb` keeps `fim`'s polarity, `+inf`
unless finiteness was positively established, so `crb < tol` is False for
an unidentifiable parameter and for a `NaN` matrix alike.

The two are not bit-identical and are not meant to be: `fim_core`
computes its verdicts at the matrix's own precision and lets the
compiler schedule `J.T @ J`, where `fim` widens to float64 on the host
and keeps the Gram product eager.  Measured over 40,000 synthetic Fisher
matrices spanning six decades of eigenvalue ratio, they reach a
different `(rank, precision_limited)` on 0.39% of them, never above five
times the rank cutoff, and 62% of the differences sit in the half-to-two
times band where `precision_limited` fires on 97% of cases anyway.

```python
gm.set_param_spec("spring", "mass", ParamSpec(trainable=False))
res = fit(gm, loss, n_iter=300, lr=0.1)
res.params["nodes"]["spring"]      # physical values, inside bounds
res.losses                         # per-iteration loss
```

`fit_lm` is the Gauss–Newton alternative: it reuses the `jacfwd`
sensitivities `fim` computes, so with a handful of parameters it
converges in a few iterations where Adam needs hundreds; give it a
residual function rather than a scalar loss, and `noise_std` (a scalar or
a per-leaf σ, also accepted by `fim`) to weight the residual so the
Cramér–Rao bound comes out in the parameters' own units.

For noisy data, `fit_multiple_shooting` replaces teacher forcing with
free per-window initial states and a continuity penalty
(`windowed_loss(..., window_states=, continuity_weight=)`), so the
optimum is one continuous trajectory rather than windows each seeded
with measurement error.  All three fitters emit a `"fit_progress"` event
to `gm.add_observer` callbacks every `notify_every` iterations.

`fit` runs Adam in the unconstrained coordinates under the trainable
mask, so positive constants stay positive.  Every leaf *outside* the
mask — frozen, or trainable but not selected — comes back bit-identical
to the value passed in, so comparing a fit's input and output leaf by
leaf says exactly which constants it touched.  Gradients through the
whole graph come from a single `jax.value_and_grad`; a non-finite
gradient raises rather than continuing.

`fit` also keeps out of the directions the data cannot determine.  Every
gradient of a least-squares loss is `Jᵀr`, so a direction `v` with `Jv = 0`
has `g·v = 0` at every iterate — but Adam's step is `−lr·D g` for a diagonal
`D`, and `(D g)·v` is not zero, so plain Adam wanders inside the flat
manifold.  The loss does not notice; the parameters do.  On the spring above,
unguarded, the fitted scale of `(k, c, m)` lands anywhere from −7.5% to +46%
of the value it started at depending only on `lr` and `n_iter`, with the loss
unchanged in its first six digits.

`fit` therefore accumulates the run's gradients and removes the net
displacement's component along the directions none of them pointed in,
leaving those at the values you supplied — the data has not contradicted
them.  The loss is flat there, so nothing is paid for it, and a fit whose
gradients spanned everything gets its iterate back bit for bit.

```python
res = fit(gm, loss, n_iter=300, lr=0.1)
res.excited_rank         # 2 of 3: the data left one direction undetermined
res.undetermined_drift   # how far the raw iterate had drifted along it
```

`excited_rank is None` means the question was not answered, not that the
answer was "full rank": fewer iterations than parameters, more than 512
trainable leaves, or `hold_undetermined=False`.  The cutoff is numerical —
a direction the data resolves *weakly* is kept, not held; for "well enough
to use", read `crb` against a tolerance you declare.  And the degeneracy has
to be a fixed direction in the unconstrained coordinates: `SpringDamperNode`
gives `damping` the identity transform so that zero damping stays
representable, which makes the scale direction `(c, 1, 1)` and rotates it as
`c` moves.  Declare `transform="log"` on every parameter a scale degeneracy
mixes — the coordinates `fim(scale="relative")` already assumes — and it
becomes constant:

```python
gm.set_param_spec("spring", "damping",
                  ParamSpec(bounds=(0.0, None), transform="log"))
```

**`fit_lm` and `fit_multiple_shooting` run the same guard**, on the same
`hold_undetermined` keyword, and fill the same two fields.  Neither is immune
for the reason it might look immune: the gradient is orthogonal to the null
space, but no step rule here *is* the gradient.  Levenberg–Marquardt solves
`(A + λ·diag(A))⁻¹g`, which is orthogonal to `null(A)` only where `diag(A)`
is isotropic there — take `A = [[1, 2], [2, 4]]` and `g = (1, 2)`, whose step
is `∝ (2, 1)` for every `λ` against a null space spanned by `(2, −1)`.

They differ in how badly.  `fit_lm`'s step vanishes with the gradient, so its
drift *converges*: on the spring above it settles 0.85% (noiseless) or 0.43%
(σ = 0.02) from the scale you supplied and stays there, bit for bit, from
iteration 10 to 200.  `fit_multiple_shooting` is Adam, so it does not settle:
2.0% to 4.8% across `lr` 0.01–0.2, a 3.0% spread that the schedule picks and
the data has no opinion about.  Only the parameters are held there — the
returned `window_states` are decision variables of that fit and come back as
the optimiser left them.

All three fitters take a `mask=` that *narrows* `gm.trainable_mask()` —
fit two of the three trainable constants, say.  It cannot widen it: a
mask naming a leaf whose spec says `trainable=False` is a `ValueError`,
because `unconstrain` / `constrain` apply a leaf's transform and bounds
only when its spec is trainable, so such a leaf would be stepped in
physical coordinates with nothing clipping it.  To fit a frozen
parameter, make it trainable in its `ParamSpec` — that is what turns its
bounds and transform on.  (`fim`'s `mask=` is unrestricted: it
linearises, it never steps a parameter.)

## Persistence and FMI

* **Checkpoints** (`save_state` / `load_state`) store `gm.params`
  alongside the state, so a calibrated graph resumes calibrated.
* **Graph serialisation** (`gm.to_dict()`, `maddening.serialization.config`,
  `save_graph_to_usd`) writes each node's *effective* params — the
  constructor arguments with the live `gm.params` values written over
  them — plus any `set_param_spec` overrides (`param_specs` in the dict,
  `maddening:paramSpecOverridesJson` on the USD node prim).  Reloading
  gives a node constructed with the calibrated constants and the same
  trainable mask.  A config and a stage are both **untrusted input**:
  each names the Python class of every node, so each is read against an
  explicit registry — `from_dict(config, node_registry)` and
  `load_graph_from_usd(stage, node_registry=...)`, which also accepts
  whatever `register_node_class` registered and the built-ins.  Before
  0.4.0 the USD reader imported the module a stage named, which ran that
  module; pass `allow_import=True` to get that back, and only for a
  stage you trust as much as a script.
* **FMI**: `build_model_description` exposes every leaf of `gm.params`
  that the compiled step reads (see below for the rest)
  as a `causality="parameter"`, `variability="tunable"` variable named
  `<node>.params.<key>` (its own namespace, mirroring the pytree path, so
  it never collides with a `<node>.<field>` output), with
  `ParamSpec.description` / `units` as its metadata and
  `ParamSpec.bounds` as the XML `min` / `max` attributes
  (`include_parameters=False` to opt out).  A sidecar built with
  `SidecarConfig(params=gm.params, param_specs=gm.param_specs(),
  step_fn=gm._compiled_step)` serves them through `get_params` /
  `set_params` (wire kinds `get_params` / `set_params`); a set value
  takes effect on the next step without recompiling, a value outside
  the declared bounds is rejected before anything is written (the same
  rule as `PUT /graph/params`), and `GetFMUState` / `SetFMUState`
  snapshots carry the parameters.  Directional derivatives with respect to a
  parameter are the same `jax.jvp` the graph uses everywhere.  A leaf the
  step cannot read -- an `initial_*` condition, a value a node consumed at
  construction or declares in `static_data_deps`, one only an unconnected
  input would read -- is not exported (an FMU's graph is frozen, so it is a
  knob that does nothing) and is listed with the reason in
  `md.fixed_parameters`; pass that as `SidecarConfig(fixed_params=...)`
  (`FmuTcpBridge` applies it to its sidecar either way), and `set_params`,
  `set_fmu_state` and the bridge's `set_state` refuse a new value for one of
  them before anything is written.  The sidecar holds only the compiled
  step, so a standalone sidecar built without `fixed_params` cannot tell.

## Mapping weights

Edges with an interface `mapping` keep their weights under
`gm.params["mappings"]["<src>.<field>-><tgt>.<field>"]`, with the same
traced-input semantics as node constants.  A second mapped edge on the
same field pair (two additive contributions) gets its own slot,
`"...#1"`, `"...#2"`, so the two never share weights.  They are `trainable=False`
by default (an interface operator is geometry, not a physical constant);
a learned edge opts in with `gm.set_param_spec(edge.key, "H", ParamSpec())`.
See the [interface mapping guide](../algorithm_guide/coupling/interface_mapping.md).
A config (`to_dict`, USD) stores the mapping's *recipe* (`MappingSpec`:
kind, hyper-parameters, point references) and rebuilds the weights on
load; a checkpoint stores the weights themselves (`_params_mappings/`),
and when both are loaded the checkpoint's — possibly trained — weights win.

### Calibrating a parameter that a mapped edge's grid derives from

**The weights do not follow a calibrated geometry parameter, and nothing
tells you (MADD-ANO-022).** A mapping built from a node's coordinates
(the point reference `{"node": "rod", "field": "grid_x"}`, or the same
array passed by hand) computes its weights once, from the grid as it was
constructed. On the default, uniform `HeatNode`, `grid_x` is derived from
`length`, and `length` is trainable. Calibrating `length` through
`gm.params` therefore moves the rod and leaves the mapped edge
interpolating from the old grid. `compile()` accepts the graph, no
warning is raised, and the reference's recorded hash still matches,
because the static array itself never changed.

What you would see, measured on an 8-cell rod calibrated from `length`
1.0 to 1.25 and mapped onto a 16-cell rod:

- the target rod moves by about `4e-4`, where the same graph constructed
  at 1.25 moves it by about `1.2e-2`;
- the gradient of the target with respect to `length` has the wrong sign;
- `fit_lm` on the target's data stops at `length ≈ 0.22` with a small
  loss (about `1e-5`), so the fit looks successful. The truth is 1.25.
  The same fit on the source rod's own data recovers 1.25.

Until a fix lands (being scoped for 0.5.0), use one of these:

- **Freeze the parameter** when a mapped edge references a grid it
  derives: `gm.set_param_spec("rod", "length", ParamSpec(trainable=False))`.
  The default mask then leaves it alone, and `fit` refuses a mask that
  tries to widen back onto it.
- **Give the mapping explicit coordinates**, as an
  `{"asset": "points.npy"}` or `{"inline": [...]}` reference. The config
  then states that the mapping's geometry is fixed, and does not imply
  that it follows the node.
- **If the geometry has to be calibrated**, fit it from observations of
  the node itself rather than through the mapped edge. Then rebuild the
  graph at the fitted value, so the mapping is rebuilt from the new grid.

The same applies to any node whose static data is derived from a
trainable parameter while its step reads the parameter directly.

## What is not a parameter

* Initial conditions (`initial_*`): they are state, not dynamics, and are
  `trainable=False` by default.
* Structural constants (`n_cells`, stencil order, shapes, grid geometry):
  changing one changes the trace; they stay Python-side and require a
  recompile.
* `static_data` (meshes, precomputed operators): constants baked into the
  compiled step by design.
* Anything in `state`: the adaptive error norm, history logging and
  coupling residuals would see it.
