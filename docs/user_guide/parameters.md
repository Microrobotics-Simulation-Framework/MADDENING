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
import jax
import jax.numpy as jnp
from maddening import GraphManager
from maddening.nodes import BallNode, SpringDamperNode

gm = GraphManager()
gm.add_node(SpringDamperNode("spring", 0.01, stiffness=30.0, damping=2.0,
                             initial_position=0.5))
gm.add_node(BallNode("ball", 0.01, elasticity=0.7))
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

<!-- snippet: continues -->
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

<!-- snippet: continues -->
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

<!-- snippet: continues -->
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

Two more rules close what that leaves.  A value the node's step cannot run
with is a 400 naming the error: the node's hooks are traced, on a copy,
with the request's values, and a trace that raises where the current
values' trace does not is refused (`RigidBodyNode` `constraints: {"w": 0}`
used to answer 200 and make every later step a 500).  And a structural
value is stored in its parameter's own numeric type: a float with no
fractional part written for an integer parameter is stored as that integer
(`stencil_order: 4.0` is `4`), any other float for one is a 400, and an
integer written for a float parameter is stored as a float.  Before 0.4.0
the raw value was stored, and `stencil_order: 4.0` became a trainable leaf
of `gm.params`.

The size of what a write, or a new node, would build is checked before
anything is built.  `HeatNode`, `LBMNode`, `LBMPipeNode` and
`WaveletAdaptiveNode` estimate, from their parameters alone, the state they
build and the memory their constructor takes; `POST /graph/nodes` and
`PUT /graph/params` refuse a node over `MAX_NODE_STATE_ELEMENTS` state
elements or `MAX_NODE_BUILD_BYTES` of build memory before calling its
constructor, and `GraphManager.from_dict` refuses (`ValueError`) one that
would not fit in the machine's memory.  A node class without an estimate
is checked on the state it builds, as before.

`python -m maddening.examples.servers.rest_params_demo` walks through
these answers in-process (FastAPI's `TestClient`, no port, presenting the
server's token as an in-process client must): a write
honoured on the next step without a recompile, a bound and a
constructor refusal, and an initial condition applied at reset.

Before 0.4.0 such a write answered 200, was served by `GET`, was ignored by
every step (even after `POST /graph/compile`) and was saved by `to_dict()`
and `save_state()`, so the reloaded graph ran a different model.  The check
runs the node's code on a shallow copy that reads the new value, never on
the node itself.  When no faithful copy can be made (a node holding a
method bound to itself) the write is refused, with a 400 saying so.  When
the copy's code raises with the new value where it does not with the
current one, the write is refused as one the step cannot run with; when it
raises with both, or needs a concrete value, that question decides
nothing.  It detects "no
path at all": a value a node consumes at construction *and* reads again
later passes, so a node that bakes a parameter should declare it in
`static_data_deps`, which refuses it on every surface.  Likewise
`POST /checkpoint/load` refuses, and undoes, a checkpoint whose parameters
include another value of one the node consumed at construction (a
checkpoint of a pipe built with another radius); before, the load
succeeded and every later `/sim/step` failed.  More generally the load asks
of every parameter it changes what `PUT /graph/params` asks, by the same
code: a non-finite value, one outside its `ParamSpec` bounds, one the
node's constructor refuses with the graph's other values (a rod past its
Fourier limit at this graph's timestep), one that moves a mapped edge's
points -- each is a 400 and nothing is loaded.  Finiteness and the bounds
are asked, per element, of a value that is neither the leaf's now nor the
node's own, so a graph built outside its bounds reloads its own
checkpoint.  `GraphManager.load_state` in Python does not ask these: like
a `gm.params` write, it may hold a value outside a spec's bounds on
purpose, and a graph whose parameters Python moved outside them must
resume its own checkpoint into a freshly built graph.  It refuses only
text and booleans for a numeric leaf, which no save writes.  `POST
/graph/nodes` applies the same `ParamSpec` bounds to the values it is
given, and refuses a boolean or text for a numeric parameter.

### Live values, recompiles and partial pytrees

`gm.params` is populated by `compile()` and **survives a recompile**: a
calibrated leaf whose node, key, shape and dtype still exist is carried
over when you add an edge or an external input, replace a node, or the
profiler recompiles behind your back.  Leaves that no longer fit are
dropped with a `RuntimeWarning`.  `gm.reset_params()` is the explicit
way back to the constructor snapshot.  A checkpoint loaded before the
first compile compiles the graph so its params are not lost.

**The later of a `node.params` write and a `gm.params` write wins.**
Writing `gm.params` takes effect on the next step with no recompile, and
is the way to change a constant mid-run.  A `node.params` write reaches
the graph the next time anything reads `gm.params` -- your code,
`save_state`, `to_dict`, the FMU export, `GET /graph/params`, a
`jax.jit` of a sysid loss handed `gm.params` -- or runs it (`step`,
`run`, every `run_scan*`, the fitters, `windowed_loss`), or compiles it,
whichever comes first: one sync point takes it in.  A constant
`gm.params` carries is copied into it (no recompile); anything else --
a structural value, any constant of a node on the three-argument
contract -- marks the graph dirty, and the next run recompiles, so every
entry point runs the same model.  An export cannot recompile the step it
hands out, so the FMU export refuses a graph left dirty this way, asking
for `compile()` first; until 0.4.0's fix it ran the old model.
`node.params` counts its writes, so
writing a constructor value back is a write (it reverts a calibration).
Assigning it -- `node.params = {**node.params, "k": v}` -- stores the
items in a fresh counting mapping, so writes made through `node.params`
afterwards count as well (a reference to the dict you assigned is not
`node.params`: write through `node.params`); and an element written in
place into a list, a NumPy array or a nested dict
(`node.params["rates"][0] = 5.0`) is found by comparing each such value
with a copy taken at the last sync, which costs a copy and a comparison
of those values per sync.  Until 0.4.0's fix both were lost to
`gm.params`, every run, `compile()` and `to_dict()`.
A `gm.params` write reads `gm.params` first, which takes in any pending
node write, so the two are always ordered and the later wins; the one
exception is a `gm.params` leaf written through a reference held across
a node write, with no read in between, which the node write replaces.
`gm.params = tree` is later than every node write before it, and
`load_state()` restores `gm.params` later than them too.  (Until 0.4.0 a
recompile kept every live leaf, so a `node.params` write after the first
compile was dropped without a word -- a `HeartPumpNode` set to 144 bpm
and recompiled went on at 72 -- and then for a time it reached the
entry points that ran the graph but not the code that read
`gm.params`.)

<!-- snippet: continues -->
```python
gm.get_node("ball").params["elasticity"] = 0.6    # on the node, after compile
assert float(gm.params["nodes"]["ball"]["elasticity"]) == float(jnp.float32(0.6))
gm.params["nodes"]["ball"]["elasticity"] = jnp.float32(0.7)   # later: it wins
gm.compile()
assert float(gm.params["nodes"]["ball"]["elasticity"]) == float(jnp.float32(0.7))
```

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

### Warnings when several threads run graphs

Much of what this page promises is said as a Python warning: a
compile-time advisory, a leaf dropped at a recompile, a rank decided at
the precision floor.  Python keeps one list of warning filters for the
whole process, so what one thread does to that list, every thread lives
with.

To check a write, MADDENING runs the node's own code a second time -- a
trace of the step to see which leaves it reads, the constructor with the
new value, `initial_state()` for the state's layout -- and silences what
that code warns, because the real run says it.  The same is done when a
sharded wrapper checks its inner node's state and when the profiler
compiles its one-iteration variant.  That silence covers **the thread
that is probing and no other**, and nothing is saved or put back: graphs,
servers and profilers running in other threads keep every warning of
their own, and the process's filters afterwards are the ones before.
(During 0.4.0's development the probes used `warnings.catch_warnings()`,
which does neither; no release carried it.  MADD-ANO-204.)

**What MADDENING cannot do** is make `warnings.catch_warnings()` safe for
*other* code that uses it while your threads run.  That block saves the
process's filter list on the way in and puts it back on the way out.  Two
of them that overlap in two threads can put back each other's list, and a
filter one of them set inside -- often `ignore` -- then stays for good,
with no sign.  (Python 3.14 changes this only when started with
`-X context_aware_warnings`, the default of its free-threaded build;
MADDENING is tested on neither.)  Where such a block overlaps one of
MADDENING's probes the damage to MADDENING is bounded: at worst that
probe is not silenced for the rest of its run, so what the node's code
warns there is shown -- or, where warnings are errors, raised, which the
probe reads as "cannot tell".  Nothing of MADDENING's that ignores a
warning is left in the filters.  Two foreign blocks can still trade
filters with each other.

Libraries on MADDENING's own paths that use the block, found by recording
every `catch_warnings` entered (JAX 0.11.0, NumPy 2.4.6, lineax 0.0.7,
FastAPI 0.136.1, pydantic 2.13.4):

| path | who opens a block | what it sets inside |
|---|---|---|
| `compile()`, `step()`, `run()`, `run_scan()`, a `gm.params` write, `save_state()` / `load_state()` | nobody (JAX and NumPy use the block only in their test utilities and at import) | -- |
| tracing a gradient through a coupling group (the implicit solve's backward pass: `jax.grad` of a run, a `sysid` fit) | lineax, twice per trace | `ignore`, for every warning |
| `SimulationServer.create_app()` | FastAPI, 48 times for a server with the built-in routes | `ignore` for `UserWarning`, and for one pydantic category |
| the first `GET /openapi.json` of each app | FastAPI, 44 times | `ignore` for one pydantic category |

`create_app()` builds one app at a time since 0.4.0 (MADD-ANO-191).  The
lineax blocks are a few microseconds long: four threads each tracing such
a gradient at once changed the filters in 0 rounds of 30, and left an
`ignore` for every warning in 1 round of 10 once every block was held
open a millisecond longer.  So it is rare, and it is not excluded.

What to do in a process with more than one thread:

- **Set your filters once, at start-up**, before any thread starts:
  `warnings.simplefilter(...)`, `-W`, or `PYTHONWARNINGS`.  A filter set
  that way is never saved or put back.
- **Do not open `warnings.catch_warnings()` -- or `pytest.warns`, which
  is one -- in a thread while other threads run graphs.**  Record
  warnings in the main thread, with the others idle.
- **Take a gradient through a coupling group in one thread at a time**,
  unless it is compiled with `jax.jit` and was traced before the threads
  started: only the trace opens lineax's blocks.
- To see whether a process has lost its warnings, look for
  `('ignore', None, Warning, None, 0)` at the front of `warnings.filters`.

## `ParamSpec`: what an optimiser may do

Each leaf carries a `ParamSpec` (`maddening.core.params`):

| field | meaning |
|-------|---------|
| `trainable` | may an optimiser move it (default `True`; `initial_*` entries default to `False`) |
| `bounds` | physical range, `(lo, hi)` with `None` for open |
| `transform` | `None` (clip to bounds; a leaf exactly on a bound has derivative 1, the one-sided one into its range, where `jnp.clip` alone gives 0.5), `"log"` (`p = lo + exp(u)`, strictly above `lo`, or above 0 when `lo` is `None` -- `check` refuses anything else), `"logit"` (`lo < p < hi`) |

Nodes declare specs for their own constants in `param_specs()`
(`SpringDamperNode`: stiffness and mass are `log`-positive, damping is
`>= 0`).  A graph overrides any of them:

<!-- snippet: continues -->
```python
from maddening.core.params import ParamSpec
gm.set_param_spec("spring", "mass", ParamSpec(trainable=False))

gm.trainable_mask()        # params-shaped pytree of bools
u = gm.unconstrain()       # optimiser coordinates (log k, log m, ...)
gm.constrain(u)            # back to physical values, always inside bounds
gm.check_params(p)         # ValueError naming the first leaf out of range
```

## System identification: `maddening.sysid`

<!-- snippet: continues -->
```python
from maddening.sysid import (fim, fim_core, fit, observations_from_history,
                             windowed_loss)

gm.reset_state()          # the spring from its initial position, not at rest
init = {n: gm.get_node_state(n) for n in gm.node_names}
_, hist = gm.run_scan_with_history(1000)
obs = observations_from_history(init, hist)          # T = 1001 samples

loss = lambda p: windowed_loss(
    gm, p, obs, obs_fn=lambda h: h["spring"]["position"], window=50)
```

`python -m maddening.examples.advanced.sysid_demo` runs this section end
to end on a spring recorded with noise: the FIM's rank-2-of-3 verdict
for `(k, c, m)`, `fit` holding the undetermined scale (`excited_rank`,
`hold_declined`), `fit_lm` recovering `k` and `c` after a `ParamSpec`
freeze and under a `mask=`, `params_table()` before and after, and the
Cramér–Rao bounds the fit lands within.  Its residual is a `run_sweep`
rollout, which does not advance the graph.

Record data that can tell the parameters apart.  `run_scan` leaves the
graph at its final state, and the 1000 steps run above left the spring
at rest, where its position moves by about `3e-5` and says almost
nothing about `k` or `c`.  `gm.reset_state()` starts it again from
`initial_position=0.5`, so the record holds the whole transient.

`windowed_loss` is teacher-forced: every window restarts from the
measured state, so gradients cannot compound over a long stiff rollout
(`mask_unconverged=True` drops, from the loss and from its gradient,
windows in which a coupling group exited at `max_iterations`
unconverged, including a window that diverged).  On a multi-rate graph
each window also restarts on the sub-step its first sample was recorded
at, which needs to know where the record began: pass `start_step=` (0
for a record taken from `compile()` or `reset_state()`, `n` after
`gm.run(n)`; experimental).  A multi-rate graph without it warns that 0
is assumed; a record that began elsewhere would be replayed on the wrong
phase in every window.  A coupling group's predictor history and IQN-IMVJ
warm starts are not in the observations either, so they are replayed:
each window starts from the ones the previous window ended with (gradient
stopped), the first from the cold ones `compile()` and `reset_state()`
leave, and the loss is exactly zero at the parameters that generated a
record taken from either.  A record that began after the graph had
stepped (`start_step > 0`) cannot be replayed exactly, and says so.

Before fitting, ask what the data can identify:

<!-- snippet: continues, no-run, reason: fragment: residual stands for the reader's residual function -->
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
`rank`'s threshold scales with the matrix (until the matrix itself leaves
float32's normal range -- or flushes to exactly zero from a Jacobian that
is not, at `|J|` below about `1e-19` -- which `fim` reports with a
`PrecisionLimitWarning`, and `fim_core` with `precision_limited`).
Freeze one of them, then fit:

`scale="relative"` — the default, and the coordinates the eigenvectors
above are in — multiplies each Jacobian column by the parameter's value,
so a parameter sitting at exactly `0.0` (`SpringDamperNode`'s
`initial_velocity` default) has no column at all and reads as
unidentifiable however well the data determine it; `report.zero_scaled`
names those.  `scale="nominal"` removes the problem by taking the column
scale from the parameter's `ParamSpec` instead — the width `hi - lo` of a
finite `bounds`, a scale and not a location, so a symmetric range around
zero is no longer a zero:

<!-- snippet: continues, no-run, reason: fragment: residual stands for the reader's residual function -->
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

<!-- snippet: continues, no-run, reason: fragment: residual_fn, params, i and tol are the reader's -->
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

<!-- snippet: continues -->
```python
gm.set_param_spec("spring", "mass", ParamSpec(trainable=False))
start = jax.tree.map(lambda x: x, gm.params)
start["nodes"]["spring"]["stiffness"] = jnp.asarray(45.0)    # the data's: 30
start["nodes"]["spring"]["damping"] = jnp.asarray(3.0)       # the data's: 2
res = fit(gm, loss, params=start, n_iter=300, lr=0.1)
res.params["nodes"]["spring"]      # stiffness ~30, damping ~2: physical, inside bounds
res.losses                         # per-iteration loss: 0.58 at the start, below 1e-10 by the end
res.best_iteration, res.best_loss  # which iterate `params` is, and its loss
```

`fit` returns the lowest-loss iterate it evaluated, which is not always
its last.  Adam's step is about `lr` in size however small the gradient,
so near a minimum it can step past it and end the run higher than it
was, and a fit started right next to the minimum can walk away from it.
`best_iteration` counts the updates that produced `params` (0 is the
start) and indexes `losses`:
`res.losses[res.best_iteration] == res.best_loss`, except when it equals
`len(res.losses)`.  Then `params` is the iterate the last update
produced, which the loop never evaluated, and `fit` evaluated it once
more to compare it with the rest.  A run whose loss never rose returns
its last iterate.  `fit_multiple_shooting` chooses the same way, taking
the parameters and the window states from the same iterate, and `fit_lm`
needs no choice: it accepts only a step that lowers the loss, so its last
iterate is its lowest.

`fit_lm` is the Gauss–Newton alternative: it reuses the `jacfwd`
sensitivities `fim` computes, so with a handful of parameters it
converges in a few iterations where Adam needs hundreds; give it a
residual function rather than a scalar loss, and `noise_std` (a scalar or
a per-leaf σ, also accepted by `fim`) to weight the residual so the
Cramér–Rao bound comes out in the parameters' own units.  Its
`converged` is `True` when the loss reached `tol`, or when the step an
iteration proposed moved every trainable parameter by no more than
`step_tol` of its own magnitude -- whether or not that step lowered the
loss, since at the float floor a fit's proposal rounds to nothing and
cannot.  The default `step_tol` is `2**4` ulps of each parameter's dtype,
so a float32 fit that reaches its optimum reports it; and once the fit
has lowered the loss, an iteration that rejects every candidate down to
one within `step_tol` has reached the rounding floor and converges too.
Neither fires while a parameter on its bound could lower the loss by
moving into its range.  (`fit` and `fit_multiple_shooting` have only the
`tol` test, so with the default `tol=0.0` their `converged` is always
`False`; read `best_loss`.)  The solve's floor is `eps` times each
column's own curvature, so the step depends neither on the residual's
units nor on any parameter's (a floor shared across columns once crushed
the step of a parameter measured in small units, which then read as
converged), and with the identifiability guard below making its tests in
coordinates no change of units moves, neither does the answer `fit_lm`
returns with its defaults.  That holds across the whole float range, for
any residual and any parameter whose residual entries (its own rounding at
the optimum included) and Jacobian entries are normal numbers of the
working precision -- in float32 a residual of `1e-30` or `1e30`, a
parameter whose natural scale is `1e-23` (a 10 nm particle's volume in
cubic metres) or `1e23` -- because the loss is computed on the residual
framed by a power of two and the solve on Jacobian columns framed the
same way, exactly; until 0.4.0's fix `r * r`, `JᵀJ` and `Jᵀr` flushed or
overflowed there and `fit_lm` reported `converged=True` at a wrong point or
at its start.  It never reports `converged` on a solve whose columns it
could not represent, or on a loss of `0.0` from a residual that is not
zero; it warns instead.  And neither test fires unless the undamped Gauss-Newton step
from the iterate -- least squares on the equilibrated Jacobian, with no
damping and no floor -- would also move every parameter by no more than
`step_tol`, or, at the floor, would not lower the loss: a shrunken step
cannot read as stationary.  When that step would still lower the loss
there it is taken as the next iterate, so the verdict at the floor does
not depend on which units the damped candidates' rounding happened to
favour.  An on-bound gradient whose own Newton step is within `step_tol`
is the residual's rounding and counts as zero, so a truth exactly on a
bound converges on either bound.  A float32 constant in an x64 graph is
optimised on float32's grid, as the model sees it, so it converges in as
few iterations as a float64 one.

Every fitter keeps each optimiser coordinate where `constrain` is its
transform and not a clamp: a `transform=None` leaf inside its bounds,
and a `log` / `logit` leaf short of where `exp` floors or overflows and
the sigmoid meets the edge of its representable range.  Past those the
derivative is 0, and a coordinate one step carried there used to stay
there: a spring's damping started at 4 against a truth of 0.05 landed on
0 and `fit_lm` called it converged.

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
gradient raises rather than continuing.  A loss written in units so small
that its gradient's products flush (`0.5 * ||s * r||²` with `s = 1e-19`)
has its gradients taken with a power-of-two cotangent that lifts them out
of the flush, exactly, so `fit` and `fit_multiple_shooting` move as they
would at scale one; the loss's own value is yours and can still read
`0.0` there, which they say once with a `RuntimeWarning` and which then
does not meet `tol`.

`fit` also keeps out of the directions the data cannot determine.  Every
gradient of a least-squares loss is `Jᵀr`, so a direction `v` with `Jv = 0`
has `g·v = 0` at every iterate — but Adam's step is `−lr·D g` for a diagonal
`D`, and `(D g)·v` is not zero, so plain Adam wanders inside the flat
manifold.  The loss does not notice; the parameters do.  On a spring like
the one above, observed with σ = 0.02 noise and fitted unguarded, the scale of
`(k, c, m)` lands from −7.5% to −5.4% of the value it started at depending
only on `lr`, with the loss unchanged in its first six digits, and Adam's
iterate keeps drifting: +46% after 10,000 iterations at `lr=0.2` on a shorter
record.  Returning the lowest-loss iterate does not help here, because along
a flat direction which iterate is lowest is decided by rounding, not by the
data.

`fit` therefore holds such directions at the values you supplied — the
data has not contradicted them.  A direction is held only if it passes two
tests.  First, none of the run's gradients pointed along it: `fit`
accumulates them, and every gradient of a least-squares loss lies in the
span the data can see.  That is necessary but not sufficient, because a
short or fast-converging run's gradients need not span everything the data
does determine.  So, second, the loss must have no curvature along it at the
iterate `fit` returns, measured with Hessian-vector products there (`fit_lm`
reads `JᵀJ` instead, with `fim`'s rank rule).  The net displacement along
the directions that pass both is removed.  Both tests, the hold and the loss
check below compare parameters with each other, so they are made with each
identity-transform parameter measured relative to its own size (a `log` or
`logit` coordinate is relative already) — the coordinates
`fim(scale="relative")` uses — and the guard decides the same in any units.
Made in the optimiser's own coordinates, a damping written in units of
`1e-5` looked undetermined beside the others and was held at its start,
raising `fit_lm`'s loss from `1.6e-11` to `13.9`.

Then the loss gets the last word.  The held point's loss is evaluated, and
if it is above the selected iterate's by more than rounding — `2¹⁰·eps`
relative (1.2e-4 in float32) plus what storing the held point in the working
precision can cost — nothing is held, `res.hold_declined` is `True`, and a
`RuntimeWarning` gives both losses.  So the guard never trades loss for
reproducibility, and a fit with no direction passing both tests gets its
iterate back bit for bit.  Earlier 0.4.0 development builds applied the
first test alone: `fit_lm` on a well-posed four-parameter bowl reached a
loss of 0.0 and came back at 0.22.

The degeneracy has to be a fixed direction in the unconstrained
coordinates, though.  `SpringDamperNode` gives `damping` the identity
transform so that zero damping stays representable, which makes the scale
direction `(c, 1, 1)` and rotates it as `c` moves; then no direction is
null for the whole run, and the guard holds nothing.  Declare
`transform="log"` on every parameter a scale degeneracy mixes — the
coordinates `fim(scale="relative")` already assumes — and it becomes
constant.  Then fit all three of `(k, c, m)`, and nothing else:

<!-- snippet: continues -->
```python
gm.set_param_spec("spring", "damping",
                  ParamSpec(bounds=(0.0, None), transform="log"))
gm.set_param_spec("spring", "mass",                      # trainable again
                  ParamSpec(bounds=(0.0, None), transform="log"))
mask = jax.tree.map(lambda _: False, gm.params)          # see mask= below
for key in ("stiffness", "damping", "mass"):
    mask["nodes"]["spring"][key] = True
res = fit(gm, loss, params=start, mask=mask, n_iter=300, lr=0.1)
res.excited_rank         # 2 of 3: the data left one direction undetermined
res.undetermined_drift   # how far the raw iterate had drifted along it (~1e-2)
res.hold_declined        # False: holding it cost no loss, so it was held
```

Without the `damping` line, the same fit reports an `excited_rank` of 3 and
holds nothing.

`excited_rank is None` means the question was not answered, not that the
answer was "full rank".  That happens when:

- `hold_undetermined=False`;
- there are more than 512 trainable coordinates (array elements, not
  leaves);
- the run took fewer iterations than there are trainable coordinates, so a
  direction no gradient has pointed along yet cannot be told from one that
  none ever will; or
- every gradient was zero, a degenerate spectrum with nothing to measure a
  rank against: a fit started at an exact optimum, or a loss that reads none
  of the trainable parameters.

The cutoff is numerical — a direction the data resolves *weakly* is kept,
not held; for "well enough to use", read `crb` against a tolerance you
declare.

**`fit_lm` and `fit_multiple_shooting` run the same guard**, on the same
`hold_undetermined` keyword, and fill the same three fields.  Neither is immune
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
returned `window_states` are decision variables of that fit and come back
from the selected iterate as the optimiser left them.

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
  alongside the state, so a calibrated graph resumes calibrated
  (`python -m maddening.examples.advanced.checkpoint_resume_demo`).
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
  snapshots carry the parameters.  The sidecar's Python API computes
  directional derivatives with respect to a parameter with `jax.jvp`
  (`FmuSidecar.get_directional_derivative`); the FMU binary does not
  provide them: the description does not declare
  `providesDirectionalDerivatives`, and the C wrapper's
  `fmi3GetDirectionalDerivative` returns `fmi3Error`.  A leaf the
  step cannot read -- an `initial_*` condition, a value a node consumed at
  construction or declares in `static_data_deps`, one only an unconnected
  input would read -- is not exported (an FMU's graph is frozen, so it is a
  knob that does nothing) and is listed with the reason in
  `md.fixed_parameters`; pass that as `SidecarConfig(fixed_params=...)`
  (`FmuTcpBridge` applies it to its sidecar either way), and `set_params`,
  `set_fmu_state` and the bridge's `set_state` refuse a new value for one of
  them before anything is written.  The sidecar holds only the compiled
  step, so a standalone sidecar built without `fixed_params` cannot tell.
  The description, the sidecar and the bridge each refuse a graph that has
  changed since its `compile()`: a structural `node.params` write is taken
  into the graph only as a pending recompile, which the compiled step the
  FMU runs would never see.  Compile first.

## Mapping weights

Edges with an interface `mapping` keep their weights under
`gm.params["mappings"]["<src>.<field>-><tgt>.<field>"]`, with the same
traced-input semantics as node constants.  A second mapped edge on the
same field pair (two additive contributions) gets its own slot,
`"...#1"`, `"...#2"`, so the two never share weights.  They are `trainable=False`
by default (an interface operator is geometry, not a physical constant);
a learned edge opts in with `gm.set_param_spec(edge.key, "H", ParamSpec())`.
`H` is the one weight of the built-in dense mappings; a mapping of a
[registered kind](../algorithm_guide/coupling/interface_mapping.md#registering-your-own-mapping-kind)
names its own weights, each a floating-point array under an identifier,
and what it may put here is checked when its edge is added.
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
`length`, and `length` is trainable. Calibrating `length` -- a fit, or any
`params=` pytree you pass -- therefore moves the rod and leaves the mapped
edge interpolating from the old grid. `compile()` accepts the graph, no
warning is raised, and the reference's recorded hash still matches,
because the static array itself never changed.

*Writing* the new value into the running graph is refused (MADD-ANO-063):
a `gm.params` write is a `ValueError` at the next run, `check_params`,
`to_dict()` and `save_state()`; `PUT /graph/params` is a 400 naming the
mapped edge and the field; `POST /checkpoint/load` refuses and undoes a
checkpoint carrying one; and an exported FMU leaves the parameter out of its
tunable set. The graph asks whether the node rebuilt with the value reads a
referenced field differently, because the node would use the value while
the mapping kept the old points, and the saved config would not load. Until
0.4.0's fix such a write was taken, the graph ran on the old weights, and
`to_dict()` wrote a config whose mapping `from_dict()` refused to rebuild.
So the result of a fit through the mapped edge cannot be written back; the
fit itself is not refused.

`python -m maddening.examples.coupling.interface_mapping_demo` shows the
refusal at the next run and at `to_dict()`, the state left untouched,
and the supported alternative: rebuilding the node and the mapped edge
from the new points.

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
