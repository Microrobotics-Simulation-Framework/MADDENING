# Audit: core graph engine — `graph_manager.py` (minus the coupling solver), `node.py`, `edge.py`, `solver_utils.py`, `simulation/{checkpoint,calibration,profiler}.py`   (c51cd6a)

## Summary

Two HIGHs.  (1) `compile()` preserves node state and `params` across a rebuild
but **wipes `_meta`**, so any mid-run structural edit resets a multi-rate
graph's sub-step phase: declaring one unused, zero-valued external input
halfway through an 8-step run moves the final velocity by 33 %.  The same wipe
costs a coupled graph its predictor and IQN warm starts.  (2) The release's
new "a static derived from a trainable parameter is refused" rule reads the
**node's own** `param_specs()` and ignores the graph-level `set_param_spec()`
overrides that every other part of the params API honours — unfreezing a
parameter past the rule yields a gradient of 1.0 where the truth is 2.0, and
the documented escape hatch does not clear the refusal.  Five MEDIUMs and
four LOWs follow.

The parts I expected to be fragile are not.  The *new* caching is in good
shape: the scan cache is correctly keyed on `_compile_generation` and I could
not construct a stale hit; `step` ≡ `run` ≡ `run_scan` ≡
`run_scan_with_history` bit-for-bit on multi-rate and IFT-coupled graphs
including `_meta` (they disagree in exactly one situation, a partial-`_meta`
resume, which is one of the MEDIUMs); a checkpoint of a coupled graph with
calibrated params and
external inputs round-trips bit-identically, and the restored graph steps and
*differentiates* identically to one that was never saved.  A cache-by-cache
inventory is at the end.

One out-of-surface CRITICAL-class defect found incidentally in
`surrogates/replace/_core.py` is flagged before the inventory for whoever
owns it.

## Findings

### HIGH — any recompile of a multi-rate graph resets the sub-step phase, so a mid-run edit silently changes the trajectory

**What breaks:** `compile()` preserves node state and `params` across a
rebuild — that is the whole reason `load_state` compiles first — but it
throws `_meta` away.  On a multi-rate graph it replaces the dict with a fresh
`step_count = 0`; on a uniform-rate graph it pops the key outright.  Since
`step_count` is what decides which sub-steps a node with a rate divider > 1
fires on, every structural edit re-phases the schedule mid-run.

A `TableNode` at dt=0.01 driving a `BallNode` at dt=0.03 (divider 3), run for
8 steps.  Declaring one unused, zero-valued external input after step 4 —
which changes no physics at all — moves the ball's final velocity by 33 %:

```
$ PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python repro/e26_multirate_phase.py
step_count before the edit: 4
step_count after 4 more   : 4 (expected 8)
uninterrupted ball: {'position': Array(0.947026, dtype=float32), 'velocity': Array(-0.88290006, dtype=float32)}
interrupted   ball: {'position': Array(0.91171, dtype=float32), 'velocity': Array(-1.1772001, dtype=float32)}
agree: False
```

The uninterrupted run fires the ball at counts 0, 3, 6 (three updates in
eight steps); the interrupted one fires at 0, 3 and then, after the reset, at
0, 3 again — four updates.  The error grows with the run.

The same script runs the control that isolates the cause: repeat the
interrupted run, but write `step_count` back after the recompile.  The
divergence disappears completely, so nothing else about the edit is
responsible.

```
control (counter restored): {'position': Array(0.947026, dtype=float32), 'velocity': Array(-0.88290006, dtype=float32)}
control == uninterrupted  : True
```

The same wipe costs a coupled graph its warm starts.  Node state survives a
recompile; the predictor history and the IQN-IMVJ V/W matrices do not:

```
$ PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python repro/e27_meta_wipe.py
pred_count before recompile: 3
pred_count after recompile : 0
node state preserved       : True
```

**Why it happens:** `graph_manager.py:3194-3196` (multi-rate branch)
assigns `self._state[_META_KEY] = {"step_count": jnp.array(0, ...)}` — a
replacement, not a merge — and `graph_manager.py:3201` (uniform-rate branch)
does `self._state.pop(_META_KEY, None)`.  The diagnostics block that follows
(`3216`) then does `meta = self._state.get(_META_KEY, {})`, so it rebuilds
its own keys from zero.  `load_state` is unaffected because it restores
`_meta` *after* the compile (`checkpoint.py:230`).

**What would make this a non-issue:** if a mid-run recompile were not a
supported operation, or if it were documented as a restart.  Neither holds.
`docs/user_guide/parameters.md:87` sells a recompile as transparent in
exactly the scenario that breaks here — *"`gm.params` … **survives a
recompile**: a calibrated leaf … is carried over when you add an edge or an
external input, replace a node, or the profiler recompiles behind your
back"* — and says nothing about the state counter that does not survive it.
And it is: `add_node`, `add_edge`, `add_external_input`, `remove_node`,
`remove_edge`, `add_coupling_group`, `remove_coupling_group` and
`enable_multigpu` all set `_dirty` on a live graph, and
`maddening.api.server` sets `gm._dirty = True` from its node-parameter write
endpoint (`api/server.py:608`) — the interactive-slider path, which has no
reason to restart anything.  I also checked that the phase reset is not compensated
anywhere: `_apply_multirate` reads `step_count` straight out of `_meta`, and
nothing else records how far the graph has run.

Two existing tests show the project already treats this counter as state that
must survive.  `tests/core/test_step_retrace.py:111` pins `reset_state()` as
the *explicit* way to zero it — so `compile()` doing the same thing silently
is the anomaly, not the intent.  And
`tests/core/test_params_persistence_edge_cases.py:101`
(`test_load_state_before_compile_equals_compile_then_load`) asserts
`step_count == 7` after a resume, i.e. the checkpoint path was already
hardened against exactly this loss; the plain-recompile path was not.
`tests/core/test_multirate.py::TestMultirateRecompile` covers only the
uniform↔multi-rate structure transitions, never the counter's continuity
across a recompile of an already-multi-rate graph.

I have rated this HIGH rather than CRITICAL because there is a reading in
which a structural edit is allowed to restart the schedule.  If a mid-run
edit counts as a supported operation — and the REST server makes it one —
this is a silent wrong result and belongs at CRITICAL.

**Suggested fix:** merge rather than replace in both branches — keep an
existing `_meta`, add `step_count` if it is missing, and drop only the keys
whose owning group no longer exists.  Risk: a stale key from a removed group
would otherwise persist in the scan carry forever, so the drop half has to be
explicit; and a graph whose rate dividers changed arguably *should* restart
its phase, which argues for resetting `step_count` only when
`self._rate_dividers` actually changed.

---

### HIGH — the static-data/trainable-parameter refusal ignores `set_param_spec()`, so the gradient it exists to protect is still silently wrong

**What breaks:** a node whose `static_data["table"]` is built in `__init__`
from `self.params["alpha"]`, declared via `static_data_deps()`, and frozen
with `param_specs()`.  `compile()` accepts it (correct).  The user then does
the most natural next thing — `gm.set_param_spec("n", "alpha", ParamSpec())`
to fit `alpha` — and the refusal never re-runs.  `jax.grad` reports
`d y / d alpha = 1.0`; the true value is `2.0` (the term through the baked
static is missing).  An explicit `gm.compile()` afterwards **also** accepts
the graph, so there is no recovery path short of rebuilding the node.

The same asymmetry breaks the documented remedy.  A node that declares the
dependency and leaves `alpha` trainable is refused, as intended — but the
first of the two ways out the error message names ("declare `alpha` as
`ParamSpec(trainable=False)`") does not work when it is done with
`gm.set_param_spec(...)`, even though `gm.param_specs()` and
`gm.trainable_mask()` both then report `trainable=False`.

**Evidence:**

```
$ PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python repro/e3_spec_bypass.py
compile with alpha frozen: OK
trainable_mask after unfreeze: True
graph dirty? False
d y / d alpha reported : 1.0
d y / d alpha truth    : 2.0000338554382324
recompile: accepted (!)

$ PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python repro/e4_spec_freeze.py
compile REFUSES (expected): node 'n' declares static_data['table'] as derived from parameter 'alph ...
graph param_specs says trainable: False
trainable_mask says             : False
compile after graph-level freeze STILL REFUSES: node 'n' declares static_data['table'] as derived from parameter 'alph ...
```

**Why it happens:** two independent causes that compound.

* `graph_manager.py:3361-3384` calls
  `static_data_dep_violations(spec.node)`.  That walker
  (`node.py:137-215`) resolves each declaration against `obj.param_specs()`
  — the *node's* method.  `GraphManager.param_specs()`
  (`graph_manager.py:2431-2456`) is the one that merges
  `self._param_spec_overrides`, and it is what `trainable_mask`,
  `unconstrain`, `check_params` and `maddening.sysid` all use.  The rule and
  the optimiser therefore read two different notions of "trainable".
* `set_param_spec` (`graph_manager.py:2457`, docstring: *"Does not dirty the
  graph: specs are optimiser-side metadata"*) leaves `_dirty` False, so even
  a corrected check would not re-run on the next `step()`.

This is exactly the shape of defect the brief says to look for: the rule is
new in this release (`027e3ca`, "declare and enforce static-data parameter
dependencies"), the override mechanism (`set_param_spec` /
`_param_spec_overrides`) already existed on `origin/main`, and neither branch
had a reason to think about the other.  Neither
`tests/property/test_static_data_deps.py` nor `tests/core/test_static_data.py`
mentions `set_param_spec`, so the property suite is true over a space that
excludes the hole.

**What would make this a non-issue:** (a) if `set_param_spec` could not reach
a parameter that a `static_data_deps()` entry names — it can, there is no
such check; (b) if the overrides were invisible to the optimiser too, making
the two views consistently node-only — they are not, `trainable_mask()`
returns the merged value and `sysid.fit` masks on it; (c) if the gradient
happened to be right anyway — it is not, 1.0 vs 2.0 above.  I also checked
that the node in the reproducer is genuinely differentiable through the
traced half (`p["alpha"]` reaches `update`), so the reported 1.0 is the
traced term alone, exactly the failure mode D10 describes.

**Suggested fix:** have `static_data_dep_violations` take the effective specs
rather than discover them, i.e. `compile()` passes
`self.param_specs()["nodes"].get(name, {})` (and the wrapped nodes' own specs
for inner declarations) into the walker; and make `set_param_spec` set
`self._dirty = True` when the key it touches appears in the node's
`static_data_deps()`.  Risk: the walker currently resolves each wrapped node
against its own specs, which the graph cannot always supply for an inner node
it does not know by name — the outer-name case must keep working, so the
override lookup has to be additive rather than a replacement.  Dirtying on
`set_param_spec` also costs a recompile on a call that is documented not to,
so scoping it to dependency-named keys matters.

---

### MEDIUM — `run_sweep` crashes on multi-rate graphs and on coupled graphs with diagnostics

**What breaks:** `run_sweep` never supplies `_meta`, which the compiled step
requires for any graph that has one.  A multi-rate graph raises
`KeyError: '_meta'`; a coupled graph with `diagnostics=True` raises a
`lax.scan` carry-structure mismatch.  Nothing in the docstring restricts
`run_sweep` to uniform-rate uncoupled graphs (`run_adaptive` and
`run_adaptive_scan` *do* raise a clear `RuntimeError` on multi-rate, so the
silence here reads as support).

**Evidence:**

```
$ PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python repro/e11_sweep.py
plain: OK -> {'ball': {'position': (3,), 'velocity': (3,)}}
multirate: True meta: ['step_count']
multirate: FAILED KeyError: '_meta'
coupled meta: ['coupling_a+b_amplification', 'coupling_a+b_iterations', 'coupling_a+b_residual']
coupled: FAILED TypeError: scan body function carry input and carry output must have the same pytree structure, but they differ:
The input carry state is a <class 'dict'> with 2 children but the correspondi
```

**Why it happens:** `graph_manager.py:4331` passes the user's
`initial_states` straight into `jax.vmap(simulate)` as the scan carry.  Every
other entry point carries `self._state`, which `compile()` seeded with
`_META_KEY` (`graph_manager.py:3282`).  `run_sweep` has no equivalent.

**What would make this a non-issue:** a documented restriction (there is
none), or a `_meta`-free compiled step for these graphs (there is not —
`_meta` carries `step_count` for multi-rate and the diagnostics the error
bound reads).  Pre-existing: `origin/main`'s `run_sweep` has no `_meta`
handling either, so this is not a merge regression.  `tests/core/test_sweep.py`
builds every graph at a single `timestep=0.01` and never adds a coupling
group, so the suite is silent on both cases.

**Suggested fix:** in `run_sweep`, broadcast `self._state[_META_KEY]` to the
batch size (`jax.tree.map(lambda x: jnp.broadcast_to(x, (n, *x.shape))`) and
merge it into `init_states` before the vmap, stripping it from the result as
`_user_state` already does.  Risk: the batch size has to be inferred from the
user's leaves, and a `_meta` entry whose value is genuinely per-batch (the
IQN V/W warm start) would then be shared-then-diverged rather than
per-simulation, which is correct but changes warm-start behaviour.

---

### MEDIUM — `load_state` replaces `_meta` wholesale, so a resume into a graph with more `_meta` keys makes `step()` and `run_scan()` disagree — one works, the other crashes

**What breaks:** run a coupled graph, checkpoint it, then turn the predictor
on (`predictor="linear"`) and resume.  `load_state` overwrites `_meta` with
the checkpoint's key set, dropping the `coupling_*_pred_*` entries the
recompiled graph seeded.  `gm.step()` then works — it rebuilds `_meta` each
call — but `gm.run_scan()` and `gm.run_scan_with_history()` die inside
`lax.scan`, because the carry gains a key the loaded state does not have.
This is the only place I found where the execution paths disagree.

**Evidence:**

```
$ PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python repro/e7_meta2.py
plain meta: ['coupling_a+b_amplification', 'coupling_a+b_iterations', 'coupling_a+b_residual']
pred meta : ['coupling_a+b_amplification', 'coupling_a+b_iterations', 'coupling_a+b_pred_0', 'coupling_a+b_pred_1', 'coupling_a+b_pred_count', 'coupling_a+b_residual']
after load: ['coupling_a+b_amplification', 'coupling_a+b_iterations', 'coupling_a+b_residual']
step after resume: OK
run_scan(3) after resume: TypeError: scan body function carry input and carry output must have the same pytree structure, but they differ:
run_scan_with_history(3) after resume: TypeError: scan body function carry input and carry output must have the same pytree structure, but they differ:
```

with the full message naming the missing key:

```
The input carry component carry['_meta'] is a <class 'dict'> with 5 children but the
corresponding component of the carry output is a <class 'dict'> with 6 children, ...
with the symmetric difference of key sets: {'coupling_a+b_pred_1'}.
```

**Why it happens:** `checkpoint.py:230-232` assigns
`raw_state[_META_KEY] = {field: jnp.array(arr) for ...}` — a replacement.
The comment immediately below handles the *absent*-`_meta` case ("keeps the
freshly compiled `_meta` of this graph") but not the *partial* one.  Nothing
validates the `_meta` key set against the graph the way node fields are
validated twenty lines above.

**What would make this a non-issue:** the checkpoint schema version.
`CHECKPOINT_SCHEMA_VERSION` is still 1, and the file's own bump policy
(`checkpoint.py:36-61`) calls "a new MANDATORY key is added" bump-worthy — so
in principle a stale key set should be rejected.  It is not: the manifest
verifies and the load proceeds.  I checked the adjacent, more worrying case —
a 0.3-era checkpoint that predates the `coupling_*_amplification` key this
release added — and that one *does* survive, `step` and `run_scan` both
(`repro/e29_old_ckpt.py`), so the schema-version omission has not bitten yet.
The predictor case is the reachable one.

**Suggested fix:** merge instead of replacing — start from the compiled
graph's `_meta` and write the checkpoint's fields over the keys it has,
warning about any key the checkpoint carries that the graph does not.  Risk:
a genuinely stale key from a removed coupling group would then be kept rather
than dropped, so the merge has to be keyed on the current graph's key set,
not the union.

---

### MEDIUM — the documented `jax.grad(... gm.run_scan ...)` idiom leaves tracers in `gm._state` and bricks the graph

**What breaks:** `docs/user_guide/quickstart.md` §"Differentiable Everything"
shows a loss that calls `gm.set_node_state(...)` then `gm.run_scan(...)`, and
differentiates it.  The gradient is correct, but `run_scan` assigns its
traced result back into `self._state` (`graph_manager.py:4194`), so when
`jax.grad` returns, the caller's `GraphManager` holds escaped tracers.  Every
subsequent use fails with an `UnexpectedTracerError` (or
`TracerArrayConversionError` from `save_state`) that points at JAX, not at
the framework.

**Evidence** — the doc snippet run verbatim:

```
$ PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python repro/e23_quickstart.py
d(final_pos)/d(init_vel) = 0.99999934
gm._state['ball']['position'] is now a LinearizeTracer
  gm.step(): UnexpectedTracerError: Encountered an unexpected tracer. A function transformed by JAX had a side effect, allowing for a reference to
  gm.run_scan(1): UnexpectedTracerError: Encountered an unexpected tracer. A function transformed by JAX had a side effect, allowing for a reference to
  gm.save_state('/tmp/x.npz'): TracerArrayConversionError: The numpy.ndarray conversion method __array__() was called on traced array with shape float32[]
  after gm.reset_state(): OK
```

**Why it happens:** `step`, `run`, `run_scan` and `run_scan_with_history` all
write `self._state = <traced result>`.  This is the stateful API working as
designed; it is only a problem because the documented differentiation recipe
goes through it.  `maddening.sysid` avoids it by calling
`gm._build_step_fn()` and threading state itself.

**What would make this a non-issue:** if the recovery were obvious.  It is
partly — `gm.reset_state()` clears it — but nothing says so, and on a graph
whose `_meta` also caught tracers the reset is the only thing that helps.  I
verified the coupled case recovers from both `reset_state()` and `compile()`
(`repro/e24_grad_coupled.py`).

**Suggested fix:** cheapest is documentation — change the quickstart to
differentiate through `gm._build_step_fn()` (as `sysid` and
`docs/user_guide/parameters.md` already do) or to end the loss with a state
restore.  A code fix would be to skip the `self._state = ...` write when the
result contains tracers (`isinstance(leaf, jax.core.Tracer)`), which risks
silently making `run_scan` non-stateful in a jitted caller that *wants* the
carry.

---

### MEDIUM — `external_inputs` is neither completed from the declarations nor validated, while `params` on the same call is validated exhaustively

**What breaks:** `step(external_inputs=None)` zero-fills every declared input
(documented).  A *partial* dict does not: the omitted inputs simply never
reach the node.  A typo in the node name or the field name is silently
ignored, and an undeclared extra field is silently accepted.  The contrast
with `params` is stark — `_validate_params` (`graph_manager.py:2341`) rejects
an unknown node, an unknown key, a missing key and a missing node, each with
a paragraph of explanation.

**Evidence:**

```
$ PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python repro/e15_partial_ext.py
external_inputs=None      -> 0.0
only f given ({'s':{'f':1}}) -> -98.0     # 'g' fell back to the node's own default
empty dict {}             -> -198.0       # both did

$ PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python repro/e25_ext_typo.py
correct       -> 7.0
field typo    -> 0.0
node typo     -> 0.0
undeclared     -> 7.0
```

A node that indexes `boundary_inputs["g"]` instead of `.get` gets a
`KeyError` at trace time rather than a wrong number; either way the
declaration did not hold.

**Why it happens:** `_default_external_inputs` (`graph_manager.py:3876`) is
only consulted when the argument is `None` (`graph_manager.py:4030`,
`4065`, `4170`, `4235`, `4309`, `4518`, `4660`).  The zero leaves it would supply are already
materialised per compile in `self._default_ext_leaves`, so completing a
partial dict would cost nothing.

**What would make this a non-issue:** if the docs said "supply all or none".
They say the opposite by omission: *"If ``None``, zeros are used for all
declared external inputs"* is the only statement, repeated at
`graph_manager.py:4017`, `4156`, `4216`.

**Suggested fix:** complete a supplied dict from `_default_ext_leaves` and
raise on an unknown `(node, field)` pair, mirroring `_validate_params`.
Risk: a caller currently relying on "omitted means the node's own fallback"
would change behaviour, and the unknown-key rejection could break a caller
that passes one dict to several graphs.

---

### MEDIUM — `to_dict()` drops an external input's dtype, so a config round trip turns an integer input into float32

**What breaks:** `add_external_input(..., dtype=jnp.int32)` survives in the
live graph but is written to the config as shape only.  `from_dict` rebuilds
it at the `add_external_input` default, `jnp.float32`.  A node that uses the
input as an index (`x[mode]`) then fails on the reloaded graph; one that does
arithmetic gets a different trace.

**Evidence:**

```
$ PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python repro/e10_extdtype.py
original ext spec dtype: <class 'jax.numpy.int32'>
original default leaf  : int32
serialised external_inputs: [{'target_node': 'ball', 'target_field': 'mode', 'shape': [3]}]
reloaded ext spec dtype: <class 'jax.numpy.float32'>
reloaded default leaf  : float32
```

**Why it happens:** the `external_inputs` block of `to_dict`
(`graph_manager.py:4840-4847`) writes `target_node`, `target_field` and
`shape`; `ExternalInputSpec.dtype` has no slot.  `from_dict`
(`graph_manager.py:4926`) passes only `shape`.

**What would make this a non-issue:** if the round-trip property suite
covered it — `tests/property/test_round_trips.py` generates graphs from
`tests/property/strategies.py`, whose external inputs are all default-dtype,
so the property is true over the generated space and silent outside it.

**Suggested fix:** write `"dtype": jnp.dtype(ei.dtype).name` and read it back
with a `jnp.dtype(...)` lookup, defaulting to float32 when the key is absent
so old configs still load.  Risk: none I can see beyond the usual
forward-compat of a new optional key.

---

### LOW — `auto_couple()` can remove every coupling group without dirtying the graph

**What breaks:** `auto_couple` clears `self._coupling_groups` and then marks
the graph dirty only as a side effect of `add_coupling_group`.  When it finds
no cycles it creates no groups, so the graph is left describing itself as
uncoupled while the compiled step is still the coupled one.  Reachable
because `add_coupling_group` happily accepts an acyclic node set, and
`find_strongly_connected_components` (`core/schedule.py:150`) only returns
SCCs of size > 1.

**Evidence:**

```
$ PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python repro/e21_autocouple3.py
auto_couple -> [] | _dirty = False | gm._coupling_groups = []
repr says: GraphManager(3 nodes, 2 edges, multi-rate, compiled)
...
coupling_diagnostics() now reports: {}
_meta still carries: ['coupling_a+b_amplification', 'coupling_a+b_iterations', 'coupling_a+b_residual', 'step_count']
```

The compiled step goes on writing coupling diagnostics into `_meta` that
`coupling_diagnostics()` no longer reports, and a `to_dict()` taken here
writes no `coupling_groups`, so the config and the running program disagree.

**What would make this a non-issue — and mostly does:** I could not turn it
into a wrong number.  The only reachable trigger is a group over an acyclic
node set, and for a DAG the Gauss-Seidel block converges in one pass, so the
coupled and uncoupled steps agree bit-for-bit.  I tried `max_iterations=1`
with `relaxation`/`acceleration="fixed"`, and `subcycling=True`
(`repro/e19`, `e20`, `e21`): all three gave identical numbers.  That is why
this is LOW and not HIGH — the invariant ("`_dirty` is True whenever the
compiled step no longer matches the graph") is genuinely broken, but nothing
downstream currently reads the difference.

**Suggested fix:** `self._dirty = True` immediately after
`self._coupling_groups.clear()` in `auto_couple`
(`graph_manager.py:2902`).  One line, no risk.

---

### LOW — `step()`, `run_scan()` and `get_node_state()` hand the caller live internal state

**What breaks:** for a graph with no `_meta` (the common uncoupled,
uniform-rate case) `_user_state` returns `full_state` *itself*, so
`gm.step()` returns `gm._state`.  When `_meta` is present the outer dict is
copied but the per-node dicts are still shared.  `get_node_state(name)`
returns `self._state[name]` with no docstring and no copy.  A caller that
clamps a value in the dict it was handed silently rewrites the simulation.

**Evidence:**

```
$ PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python repro/e1.py
returned dict is internal dict? True inner alias? True
after caller mutates returned state, internal position = 999.0
next step position = 998.99805
get_node_state is live ref? True
internal after mutating get_node_state: -42.0
```

**Why it happens:** `graph_manager.py:3895-3899` and `4699-4705`.

**What would make this a non-issue:** a documented "do not mutate".  The only
statement I found is in `docs/developer_guide/node_authoring.md` and is
addressed to node authors, not to callers of `step()`.  Pre-existing —
`origin/main` has the same `_user_state`.

**Suggested fix:** shallow-copy the per-node dicts in `_user_state` and in
`get_node_state` (the arrays stay shared; only the dicts are new).  Risk:
one extra dict allocation per node per step on the hot path, which is why it
probably was not done; a docstring saying "the returned dicts alias internal
state" is the zero-cost alternative.

---

### LOW — a wrapper's `static_data` and `static_data_deps` do not line up, contrary to the contract

**What breaks:** `static_data_deps()` documents its forwarded result as
*"keyed identically to the forwarded `static_data` so the two line up"*.
They only line up when both dicts collide on the same keys.  A wrapper over
two nodes that both publish `table`, where only the second declares a
dependency, reports the dependency under the *first* node's key.

**Evidence:**

```
$ PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python repro/e22_wrapper_keys.py
static_data keys     : ['inner_b.table', 'table']
static_data_deps keys: ['table']
```

**Why it happens:** `_merge_from_wrapped` (`node.py:74-132`) qualifies a key
with the holding attribute only *on collision*, and the collision pattern is
a property of the dict being merged, not of the walk.

**What would make this a non-issue:** whether anything consumes the pairing.
Today nothing does: `static_data_dep_violations` walks every node separately
and resolves each declaration against that node's own pytree, so the
mis-keyed wrapper entry cannot hide or invent a violation (I checked this on
the same graph).  The docstring is wrong and the deferred D10 step-4 rebuild
hook, which would key a rebuild off exactly this pairing, would be wrong with
it.

**Suggested fix:** qualify unconditionally when the wrapper holds more than
one `SimulationNode`, so both dicts are keyed by attribute in the same way;
or drop the "they line up" claim until the rebuild hook needs it.  Risk:
unconditional qualification changes the keys a single-wrapped node reports
(`HybridNode(ShardedStencilNode(inner))` would go from `table` to
`inner.table`), which changes `static_data_hash()` and forces one recompile
on upgrade.

---

### LOW — `run_adaptive` and `run_adaptive_scan` take no `params=`, contradicting "every run method takes an optional `params=`"

**What breaks:** `docs/user_guide/parameters.md:23` states *"Every run method
takes an optional `params=`; `None` means the snapshot."*  Five of the seven
do; the two adaptive ones do not, and both read `self.params` directly
(`graph_manager.py:4534`, `4688`) rather than going through
`_params_or_default`.  Differentiating `run_adaptive_scan` — advertised as
"fully JIT-compiled and differentiable" — therefore means writing the tracer
into `gm.params`, which works but leaves the object holding a tracer, the
same trap as the MEDIUM above.

**Evidence:**

```
$ PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python repro/e28_adaptive_params.py
step                     params=
run                      params=
run_scan                 params=
run_scan_with_history    params=
run_sweep                params=
run_adaptive             NO params=
run_adaptive_scan        NO params=
run_adaptive_scan(params=...) -> TypeError GraphManager.run_adaptive_scan() got an unexpected keyword argument 'params'
grad via gm.params mutation: 0.001458819955587387
gm.params left holding a LinearizeTracer
```

**What would make this a non-issue:** if the adaptive entry points were not
"run methods".  They are listed as such in
`docs/release_notes/v0.4.0.md:55` alongside the five that do take `params=`.

**Suggested fix:** add `*, params: Optional[dict] = None` to both and route
it through `_params_or_default`, as every other entry point does; the
adaptive scan already passes `self.params` as a traced argument, so the
plumbing exists.  Risk: none beyond the new keyword; the scan cache key does
not need to change because params is an argument, not a closure.

---

## Out of my surface — flagged for whoever owns `maddening/surrogates/`

### CRITICAL-class — `replace_node` re-adds the saved edges positionally and drops `additive`, the units and the interface `mapping`

`surrogates/replace/_core.py:78-83` calls
`gm.add_edge(e.source_node, e.target_node, e.source_field, e.target_field, e.transform)`.
`add_edge`'s remaining parameters (`additive`, `source_units`,
`target_units`, `mapping`) fall back to their defaults, so two additive
contributions to one boundary field become one overwriting edge, and a mapped
edge loses its interface mapping entirely.

```
$ PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python repro/e12_replace.py
before: additive flags = [True, True]
before: anchor seen by s = {'anchor_position': Array(3., dtype=float32)}
after : additive flags = [True, False]
after : anchor seen by s = {'anchor_position': Array(1., dtype=float32)}
```

Silent wrong numbers after a supported operation.  (`replace_node` also only
checks `surrogate_node.name`, never the type, which is what let the
reproducer stand a plain `TableNode` in for a `SurrogateNode`.)

## Unverified suspicions

* **`sysid.fit`, `profile_graph` and `viz/runner` gate on `gm._dirty` alone**
  (`sysid.py:173`, `simulation/profiler.py:418`, `viz/runner.py:62`) and
  never call `_check_static_data_dirty()`, which is the only thing that
  notices a `static_data` shape change after a compile.  For the profiler and
  the viz runner the next `gm.step()` self-heals, so I could not build a
  failing case.  `sysid.fit` is the one that worries me: it calls
  `gm._build_step_fn()` directly and never steps, so it traces against the
  *current* statics while `gm._state` and the shapes in `gm.params` are the
  pre-change ones.  I did not construct a node whose statics and state shapes
  move together.
* **`_EMPTY_EXTERNAL_INPUTS`** (`graph_manager.py:812`) is a module-level
  dict returned by `_default_external_inputs()` for graphs with no external
  inputs, shared process-wide and mutable, three lines above a comment
  promising "fresh outer dicts each call (callers may edit them)".  No caller
  mutates it today.
* **`remove_edge`** silently no-ops on an edge that does not exist and
  removes *every* edge matching the four names, including the ordinal-keyed
  siblings that `add_edge` deliberately keeps distinct.
* **`set_node_state`** accepts any pytree; a wrong field set surfaces several
  calls later as a bare `KeyError: 'y'` from inside the jitted step
  (`repro/e25_ext_typo.py`).

## What I checked and found sound

### Cache inventory (the brief's headline question)

| cache | what invalidates it | what should but does not |
|---|---|---|
| `_compiled_step` (`jax.jit`) | `_dirty`, set by `add_node`/`add_edge`/`add_external_input`/`remove_node`/`remove_edge`/`add_coupling_group`/`remove_coupling_group`/`enable_multigpu`; plus `_check_static_data_dirty()` at every entry point | `auto_couple()` when it creates no groups (LOW above); `set_param_spec()` on a key a `static_data_deps()` entry names (HIGH above); a same-shape rewrite of a static's contents (documented, and an explicit `compile()` does pick it up) |
| `_state[_meta]` (step counter, coupling warm starts) | rebuilt from scratch by every `compile()` — the opposite problem: it is invalidated when it should be **kept** (HIGH above) | n/a |
| `_scan_cache` | `_compile_generation` in the key **and** `compile()` clears the dict | nothing found — I looked for something `build()` closes over that the dirty flag does not cover; `return_history` and the adaptive controller constants are both in the key, state/ext/params are all jit arguments |
| `_default_ext_leaves` | rebuilt wholesale by `compile()`, plus a shape/dtype guard on every read (`graph_manager.py:3883-3885`, `_default_external_inputs`) | nothing found |
| `_static_data_hashes` | snapshotted by `compile()` | contents, by design and by docstring |
| `_params_dtypes` / `_params_shapes` | snapshotted by `compile()`; live leaves carried over by `_merge_live_params` | nothing found |
| per-node materialised statics (sharded wrappers) | `invalidate_static_cache()` on every node at every `compile()`, forwarded through wrappers | a node holding inner nodes in a list or dict rather than a plain attribute (documented) |


* **Checkpoint round trip, non-trivial graph.**  Two IFT-coupled
  `SpringDamperNode`s, a coupling group with diagnostics, an external input,
  and one parameter moved off its constructor value: state, `_meta` and
  `params` all restore **bit-identically** into a freshly built graph, and
  ten further steps on the restored graph are bit-identical to ten further
  steps on the graph that was never saved, with identical coupling
  diagnostics (`repro/e5_ckpt.py`).  `_meta` dtypes and shapes survive the
  round trip exactly (`repro/e6_meta.py`).
* **Gradients through a checkpointed segment.**  `d(a.position)/d(a.stiffness)`
  over ten steps after a restore is bit-identical to the same gradient on the
  never-saved graph, and agrees with a central finite difference to 0.2 % at
  float32 (`repro/e13_grad_ckpt.py`).
* **A checkpoint missing a `_meta` key this release added still resumes.**
  0.4.0 introduced `_meta["coupling_*_amplification"]` without bumping
  `CHECKPOINT_SCHEMA_VERSION`; I rebuilt a 0.3-style archive without that key,
  wrote a matching manifest, and both `step()` and `run_scan()` recovered
  (`repro/e29_old_ckpt.py`).  The version omission is real but currently
  harmless.
* **`step` ≡ `run` ≡ `run_scan` ≡ `run_scan_with_history`**, bit-for-bit, on
  a multi-rate graph (9 steps) and an IFT-coupled graph (7 steps), including
  the `_meta` contents; `history[-1] == final_state` in both
  (`repro/e17_agree.py`).  Also agreed after a live `gm.params` edit
  (`repro/e2.py`).
* **Scan-cache invalidation.**  `_cached_scan` prepends `_compile_generation`
  and `compile()` both bumps it and clears the dict, so a graph mutated
  between two `run_scan(n)` calls with the *same* `n` returns exactly what a
  graph built that way from the start returns (`repro/e18_mutate_between.py`).
  I looked for something the built program closes over that is not covered by
  the dirty flag — `return_history` and the adaptive controller constants are
  both in the key, and state/ext/params are all arguments — and found
  nothing.
* **The `static_data` contents contract behaves as documented.**  A
  same-shape rewrite of a static's buffer is invisible to
  `static_data_hash()` and to the dirty check, and an explicit `compile()`
  *does* pick it up, on both the `step` and the `run_scan` paths
  (`repro/e14_static_contents.py`).  The `invalidate_static_cache` /
  `_build_step_fn` ordering in `compile()` is what makes the second half
  true.
* **Live-parameter survival across a recompile.**  A calibrated leaf survives
  a structural change (`add_node` + `add_edge` + recompile) with no warning
  and no loss; `remove_node` discards the node's params entry deliberately
  and, as intended, without a spurious drop warning
  (`repro/e16_params_recompile.py`).
* **No retrace from weak-typed params.**  A Python float assigned into
  `gm.params` leaves `trace_count` at 1 over three `step()`s and
  `scan_trace_count` at 1 over three `run_scan` / `run_adaptive_scan` calls
  (`repro/e9_retrace.py`), so `_coerce_params_leaves` is doing its job and
  `run_adaptive_scan`'s bypass of `_params_or_default` costs nothing.
* **`_default_ext_leaves`** is rebuilt on every `compile()` and guarded by a
  shape/dtype check on read, so it cannot serve a leaf of the wrong shape.
* **`reset_state()` covers every `_meta` key the compiler seeds.**  I walked
  the seeding block (`graph_manager.py:3216-3281`) against the reset's
  key patterns (`4711-4735`): `step_count`, `sub_step`, `*_iterations`,
  `*_residual`, `*_amplification`, `*_V`, `*_W`, `*_pred_*` and
  `*_pred_count` are all matched, so a reset leaves no stale warm start.
* **`remove_node` / `remove_edge` clean up `_param_spec_overrides` and
  `params["mappings"]`**, so a to_dict/from_dict after a removal does not
  trip over an orphaned override.
* **`edge.py`, `solver_utils.py`** changed only in docstrings in this release
  (verified against `origin/main`); `EdgeSpec` is a frozen dataclass and the
  `edges` property returns a fresh list.
* **`simulation/calibration.py`** is deprecation-only in this release, and
  `simulation/profiler.py`'s changes (the `amp_key` in `_meta_group_keys`,
  the `n_stat_steps` window pin) are self-contained; `_one_iteration_variant`
  restores groups, state, params and the compiled step correctly.

## Test runs

```
cd /home/nick/MSF/msf/MADDENING-wt/audit/graphcore
PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest \
  tests/core/test_graph_manager.py tests/core/test_step_retrace.py \
  tests/core/test_scan_program_cache.py tests/core/test_static_data.py \
  tests/core/test_checkpoint.py tests/core/test_checkpoint_and_params_shape_guards.py \
  tests/core/test_graph_params_api.py tests/core/test_params_spec.py \
  tests/core/test_params_contract_completeness.py \
  tests/core/test_params_persistence_edge_cases.py \
  tests/core/test_param_spec_edge_cases.py tests/core/test_sweep.py \
  tests/core/test_adaptive.py tests/core/test_node.py tests/core/test_edge.py \
  tests/property/test_round_trips.py tests/property/test_static_data_deps.py \
  -q -p no:cacheprovider -rs
```

`317 passed in 996.21s` — no failures, no skips.  So none of the findings
above is caught by the suites that cover this surface; each reproducer is in
`repro/` next to this report and each was run against the worktree at
`c51cd6a`.

Second batch, same invocation with
`tests/core/{test_profiler,test_calibration,test_calibrate,test_compile_cache,test_static_array,test_replace_static_sharding,test_multirate,test_schedule}.py`:
`138 passed in 1193.16s` — again no failures and no skips.  455 tests over the
two batches, all green.

The four compliance scripts (`check_anomalies.py`, `check_impl_mapping.py`,
`check_citations.py`, `check_transforms.py`) all exit 0 on this tree.

Nothing in the repository was modified.  Reproducers live in `repro/` under
this report directory, not in `tests/`.
