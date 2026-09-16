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

`verify_node` checks the contract for you: `params_consistent` (injected
params reproduce the baked step), `params_gradient_finite`, and
`params_effective` (every trainable leaf actually influences the output —
the check that catches a constant still read from `self.params`).  See
[verification](../developer_guide/verification.md).

## `ParamSpec`: what an optimiser may do

Each leaf carries a `ParamSpec` (`maddening.core.params`):

| field | meaning |
|-------|---------|
| `trainable` | may an optimiser move it (default `True`; `initial_*` entries default to `False`) |
| `bounds` | physical range, `(lo, hi)` with `None` for open |
| `transform` | `None` (clip to bounds), `"log"` (`p = lo + exp(u)`, strictly positive), `"logit"` (`lo < p < hi`) |

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
from maddening.sysid import fim, fit, observations_from_history, windowed_loss

init = {n: gm.get_node_state(n) for n in gm.node_names}
_, hist = gm.run_scan_with_history(1000)
obs = observations_from_history(init, hist)          # T = 1001 samples

loss = lambda p: windowed_loss(
    gm, p, obs, obs_fn=lambda h: h["spring"]["position"], window=50)
```

`windowed_loss` is teacher-forced: every window restarts from the
measured state, so gradients cannot compound over a long stiff rollout
(`mask_unconverged=True` zeroes windows in which a coupling group exited
at `max_iterations`).

Before fitting, ask what the data can identify:

```python
report = fim(lambda p: residual(p), gm.params, mask=gm.trainable_mask())
report.cond                  # inf => an exactly unidentifiable direction
report.least_identifiable()  # ("['nodes']['spring']['mass']", 0.58)
report.eigvecs[:, 0]         # the direction itself, in relative coordinates
```

For a spring observed through position only, `k`, `c` and `m` enter as
`k/m` and `c/m`: scaling all three together is invisible, and the FIM's
weakest eigenvector is `(1, 1, 1)/√3`.  Freeze one of them, then fit:

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
mask, so positive constants stay positive and frozen leaves are returned
bit-identical.  Gradients through the whole graph come from a single
`jax.value_and_grad`; a non-finite gradient raises rather than
continuing.

## Persistence and FMI

* **Checkpoints** (`save_state` / `load_state`) store `gm.params`
  alongside the state, so a calibrated graph resumes calibrated.
* **Graph serialisation** (`gm.to_dict()`, `maddening.serialization.config`,
  `save_graph_to_usd`) writes each node's *effective* params — the
  constructor arguments with the live `gm.params` values written over
  them — plus any `set_param_spec` overrides (`param_specs` in the dict,
  `maddening:paramSpecOverridesJson` on the USD node prim).  Reloading
  gives a node constructed with the calibrated constants and the same
  trainable mask.
* **FMI**: `build_model_description` exposes every leaf of `gm.params`
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
  parameter are the same `jax.jvp` the graph uses everywhere.

## Mapping weights

Edges with an interface `mapping` keep their weights under
`gm.params["mappings"]["<src>.<field>-><tgt>.<field>"]`, with the same
traced-input semantics as node constants.  They are `trainable=False`
by default (an interface operator is geometry, not a physical constant);
a learned edge opts in with `gm.set_param_spec(edge.key, "H", ParamSpec())`.
See the [interface mapping guide](../algorithm_guide/coupling/interface_mapping.md).
Checkpoints store node params only; mapping weights are rebuilt from the
graph definition.

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
