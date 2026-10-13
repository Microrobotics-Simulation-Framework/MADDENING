# Audit of the fix: `compile()` invalidates a node's materialised statics

Independent, context-free audit of `git diff origin/release/0.4.0..HEAD` on
`fix/perf-interactive-audit` (14 lines in `GraphManager.compile()` plus two
regression tests).  The auditor did not write the fix, and its author was
itself an auditor, so the question asked here is the narrow one: does the fix
put back everything the optimisation took away, and does it take anything else
away in the process.

## The direct answer

**The fix is correct but was not complete, and the hole was the same class of
defect it was written to remove.**

`getattr(spec.node, "invalidate_static_cache", None)` asks one object — the
node the graph holds — for a hook by name.  A wrapper holds the node it wraps
as an attribute, so for `HybridNode(ShardedStencilNode(inner))` the probe finds
nothing on the `HybridNode` while the `ShardedStencilNode` one level in keeps
the pre-rewrite device buffer.  `compile()` then traces the rebuilt step
against that buffer, and the simulation is silently wrong with no recovery.
That is a public, in-tree composition and it is **proven, reproduced, and
fixed** on this branch (MAJOR-1 below).

With that fixed, what can still go stale:

* **A static rewritten in place with no `compile()` and no retrace.**  Proven.
  This is the accepted trade of the PR the fix sits on, it is contract-backed
  (`SimulationNode.static_data`: values "should be stable across calls"), and
  the recovery is a `compile()`.  It was under-documented — the guide named
  only `wrapper.invalidate_static_cache()`, which on its own does **not** help
  inside a graph, because a plain `step()` loop never re-enters the Python that
  would re-materialise.  Fixed in `docs/developer_guide/sharded_static_data.md`
  (MINOR-1).
* **The wrappers' *other* cache, `_sharded_cache` / `_local_update_fn`.**
  Neither `invalidate_static_cache()` nor `compile()` clears it.  It holds
  compiled `shard_map` functions, and the statics reach them as *arguments*, so
  it cannot serve a stale static **value**; what it can serve stale is the
  inner node's Python behaviour read at build time.  Pre-existing, confirmed
  by the prior audit (§X9), out of scope for this fix (MINOR-2).
* **`GraphManager` never notices a sharded node's static data at all.**  None
  of the wrappers proxies `static_data`, so `spec.node.static_data_hash()` is
  `0` forever and `_check_static_data_dirty()` — the automatic recompile that
  covers a plain node — can never fire for exactly the nodes that have the
  cache.  Pre-existing, but it is why `compile()` is the *only* recovery point
  and therefore why the fix had to be there (MINOR-3).

Ordering is right; per-frame cost is unchanged (counted, not timed); the loop
reaches every node the graph holds, and there is no node container in
`GraphManager` other than `self._nodes`.

---

## CRITICAL

None.

---

## MAJOR

### MAJOR-1 — the hook is probed by name on one object, so a cache one level in is missed

**Confidence: proven.**  Reproduced, regression-tested, fixed in this branch
(commit `1fce120`).

**What breaks.**  `compile()` does

```python
for spec in self._nodes.values():
    invalidate = getattr(spec.node, "invalidate_static_cache", None)
```

`spec.node` is the outermost object.  A grep for node-holding node classes in
`src/maddening/` finds exactly four, and only two of them implement the hook:

| class | holds | implements the hook |
|---|---|---|
| `ShardedStencilNode` | `_inner` | yes |
| `ShardedUnstructuredNode` | `_inner` | yes |
| `ShardedPointwiseNode` | `_inner` | no (has no static cache) |
| `HybridNode` | `physics_node` | no |

`HybridNode` (`maddening.core.simulation.hybrid_node`) is shipped and tested,
and the CHANGELOG bills it as "a drop-in `SimulationNode` replacement".  It
delegates `update`, `halo_width`, `initial_state`, `state_fields` and the
params contract to the node it augments — including to a sharded wrapper.  Put
one over a `ShardedStencilNode` and the fix does nothing:

```
spec.node type: HybridNode
has hook? False
mask-zero honoured after compile()? False      # the field still moves
cache still populated after compile()? True
```

That is bit-for-bit the failure mode MAJOR-1 of the previous audit described,
reachable through a supported API after the fix for it landed.

**Why it is structural, not a typo.**  The hook was duck-typed: nothing in
`SimulationNode` declared it, nothing enforced it, and the two implementers did
not forward it inward.  A `getattr` probe cannot distinguish "this node has no
cache" from "this node has a cache it forgot to expose" from "this node wraps
one that does".  A future wrapper would opt out silently by doing nothing.

**The fix (implemented).**  Make it a contract method rather than a name:

* `SimulationNode.invalidate_static_cache()` — documented, `@stability(STABLE)`,
  a no-op for the node itself whose default **forwards to every
  `SimulationNode` held as an instance attribute**, with a re-entrancy guard so
  a cycle terminates.  A node built as a frozen dataclass still carries the
  guard (`object.__setattr__`), and a node that refuses the attribute entirely
  cannot hold a mutable back-reference, so forwarding stays finite.
* `ShardedStencilNode` / `ShardedUnstructuredNode` drop their cache and then
  call `super()`, so the chain continues past them.
* `HybridNode` and `ShardedPointwiseNode` need no change — they inherit the
  forwarding — and neither does any future wrapper.
* `compile()` keeps its `getattr`/`callable` probe: it now always resolves, and
  it still covers the duck-typed node objects the tests hand the graph.

**Regression tests** (`tests/cloud/multigpu/test_sharded_static_cache.py`, all
on a single device so a 1-CPU runner does not skip them):

* `test_compile_reaches_a_static_cache_nested_inside_another_node`
* `test_invalidate_static_cache_forwards_to_the_node_it_wraps`
* `test_invalidate_static_cache_terminates_on_a_cycle`

`3 failed, 9 passed` against the pre-fix tree (extracted with `git archive` so
the comparison is code, not memory); `12 passed` after.

---

## MINOR

### MINOR-1 — the documented escape hatch does not work inside a graph

**Confidence: proven** (the mechanism), **judgement** (the severity).

`docs/developer_guide/sharded_static_data.md` said: a static rewritten in place
is what the check cannot see, "call `wrapper.invalidate_static_cache()` if you
do that".  Inside a `GraphManager` that advice is not sufficient on its own.
Instrumenting `_materialise_sharded_statics` shows it is entered **once**, at
the first trace, and not again:

```
after compile: n_traces = 0   materialise calls = 0
after step 1 : n_traces = 1   materialise = 1
after step 6 : n_traces = 1   materialise = 1
```

The materialised arrays are concrete (`jax.ensure_compile_time_eval`), so they
are baked into the outer `jax.jit` as constants.  Clearing a cache that nothing
will re-read changes nothing; you need a new trace, and the framework's way of
asking for one is `gm.compile()`.

Answering the brief's question directly — *if a user rewrites a static in place
and does not call `compile()`, what happens?*  Nothing: the old buffer is used
for every subsequent step, silently.  Proven:

```
graph: NEW array object, no compile()     honoured? True
graph: NEW array object, compile()        honoured? True
graph: in-place rewrite, no compile()     honoured? False
graph: in-place rewrite, compile()        honoured? True
```

This is acceptable — it is out of contract, the recovery exists, and an
identity-changing rebuild (the in-contract way to change a static) is picked up
— but it was not documented.  **Fixed**: the guide now says `compile()` is the
recovery inside a graph, that it reaches wrapped nodes too, and that a rewrite
with no recompile is not picked up by a later `step()` because the static-data
check hashes shape and dtype, never contents.

One consequence worth stating plainly, because it is a real behaviour change
the PR made and the fix does not undo: **before the PR every trace re-read the
buffer; now only a `compile()` does.**  A retrace that is not a compile — a
state whose sharding or dtype changes, `jax.grad`, a scan program built by
`run_scan`, the profiler's rebuild — re-enters `_materialise_sharded_statics`
but hits the identity-keyed cache and reuses the stale buffer.  The framework
used to be accidentally robust here; it now requires the documented call.

### MINOR-2 — `_sharded_cache` and `_local_update_fn` are not reached by anything

**Confidence: proven** (they are not cleared), **believed** (that it cannot
corrupt a static's value).

`invalidate_static_cache()` clears `_static_device_cache` only.  The wrappers
also hold `_sharded_cache` (compiled `shard_map` functions, keyed on shapes,
dtypes, `static_data_hash()` and a params signature — all content-blind) and
`_local_update_fn` (built once in `__init__`, closing over `inner`,
`state_set`, `integrals`, `integral_reduction`, `static_exchange`, `axis_map`,
`boundary`).  Neither is cleared by `invalidate_static_cache()` or by
`compile()`.

It cannot serve a stale static *value*: `_materialise_sharded_statics()`'s
output is passed to the compiled function as an **argument**
(`fn(state, boundary_inputs, dt, static_materialised, params)`), so a stale
compiled function with a fresh array still computes the fresh answer.  What it
can serve stale is the inner node's Python behaviour read at build time — the
`mode="fast"` case the prior audit confirmed as pre-existing on
`origin/release/0.4.0`.  Out of scope for this fix; recorded so the same worry
is not re-opened against it.

### MINOR-3 — a sharded node's static data is invisible to the graph's dirty check

**Confidence: proven** by reading.  Pre-existing, not caused by this diff.

None of `ShardedStencilNode`, `ShardedUnstructuredNode` or
`ShardedPointwiseNode` overrides `static_data`, so they inherit the base's
`{}` and `static_data_hash()` returns `0` for the life of the node.
`GraphManager._static_data_hashes` therefore records `0`, and
`_check_static_data_dirty()` — the automatic "a static's shape or dtype drifted,
recompile" safety net that covers an unwrapped node — can never fire for a
sharded one.

This is the reason `compile()` had to be the invalidation point, and it is
worth stating as a finding rather than leaving implicit: the only nodes with a
static cache are exactly the nodes the automatic recompile cannot see.  Fixing
it properly means proxying `static_data` from the wrappers, which changes what
`_get_sharded_fn` keys on and what `replace_node` warns about — not a change to
make inside an audit.

### MINOR-4 — the invalidation is placed after `_build_step_fn()`, which is the fragile order

**Confidence: proven** that it is correct today, **believed** that the other
order is strictly better.  Not implemented — there is no failing test to write,
and a placement change to code under audit deserves its own review.

The loop sits between `self._compiled_step = jax.jit(_counted_step)` and the
`_static_data_hashes` snapshot.  Two constraints apply:

1. **It must precede the hash snapshot.**  It does.  This matters for a node
   whose invalidation rebuilds its statics: the snapshot has to describe the
   post-invalidation node, or the next `_check_static_data_dirty()` reads drift
   that is not there.
2. **It must precede anything in `compile()` that could materialise.**  It does
   *today*, but only because `_build_step_fn()` and `jax.jit` are both lazy —
   measured: `materialise calls == 0` at the end of `compile()`, so nothing
   before the loop populates a cache and nothing after it repopulates one.  If
   `_build_step_fn` ever traced eagerly, the current order would trace against
   the stale cache and *then* clear it: the rebuilt step would still be wrong,
   and the loop would look like it had done its job.

Moving the four lines to just above `step_fn = self._build_step_fn()` satisfies
both constraints and is correct by construction rather than by a property of
the current `_build_step_fn`.  Recommended, one line moved.

### MINOR-5 — the cost comment understates the unstructured path

`compile()`'s comment said "the cost is one `device_put` per sharded static per
compile".  True for `ShardedStencilNode`.  `ShardedUnstructuredNode`'s
re-materialisation is a `device_get`, a NumPy gather through the layout and a
`device_put` per static — the round trip the PR was written to remove.  Still
per compile, not per frame, so the conclusion holds; the comment now reads "one
re-materialisation ... paid lazily on the next trace".

---

## Verified safe

Chased and found sound.  Recorded so they are not re-investigated.

* **The loop reaches every node the graph holds.**  `self._nodes` is the only
  node container in `GraphManager`.  Edges (`EdgeSpec`), back edges, coupling
  groups (`CouplingGroup.nodes`), external inputs and interface mappings all
  address nodes **by name** and are validated against `self._nodes` at
  registration (`add_edge`, `add_external_input`, `add_coupling_group`,
  `accelerated_fields`); the multi-rate and substep paths index
  `nodes[nn].node`, the same object.  There is no in-process sub-graph node —
  `SubgraphSpec` is the distributed launcher's unit, a separate process with
  its own `GraphManager`.  `_NodeSpec.update_fn` is a bound method of
  `spec.node`, not a second object.  `replace_node` goes through
  `remove_node`/`add_node`, so the replacement is registered normally and its
  cache starts empty.
* **No other node class caches a materialised static.**  Of the caches in
  `src/maddening/`: `AdaptiveNode._capture_cache` / `_trapped_cache` memoise a
  gradient-capture ratio and a trapped flag keyed on a *content* fingerprint of
  the params — diagnostics, not arrays, and content-keyed so not stale-able the
  same way.  `halo_unstructured`'s `_shift_tables_cache` and
  `_ghost_source_table_cache` hang off the `UnstructuredPartitionLayout` (a
  frozen dataclass, fixed for the wrapper's life) and are derived from the
  layout's own fields, not from `static_data`.  `usd/writer` and the viz
  backends cache stage prims and iso-surfaces, nothing that reaches `update`.
* **Per-frame cost is unchanged — the point of the PR survives.**  Counted, not
  timed (the machine is shared; no wall-clock number is reported here).  With
  the fix in place, on a one-device stencil graph:

  ```
  steady-state device_put over 10 steps        0
  device_put during compile() itself           0
  device_put on first step after compile()     1
  device_put over next 10 steps                0
  ```

  The invalidation costs one `None` assignment plus one no-op forwarding call
  per node per `compile()`; the re-materialisation is deferred to the next
  trace, which is where it belongs.
* **A rewrite between `compile()` and the next `step()` is picked up.**  The
  brief's most obvious candidate is not a hole: `compile()` only *clears*, and
  the re-materialisation is lazy, so the buffer is read at the first trace
  after the compile.  Proven — `rewrite AFTER compile(), before step` honours
  the new mask.
* **The `id()`-recycling window is not widened by the fix.**  Dropping the
  cache drops the pin the key relied on, but the next comparison is against a
  freshly computed key with `cached is None`, so a recycled address can never
  be compared against a key that outlived its object.
* **An in-contract change still needs no `compile()` on the eager path.**  A
  standalone `wrapper.update(...)` picks up a new array object immediately
  (`eager: NEW array object picked up? True`), so the identity key still earns
  its keep where the Python actually re-runs.
* **Nothing between the new loop and the end of `compile()` repopulates a
  cache.**  Measured (`materialise calls == 0` after `compile()`), and by
  reading: `static_data_hash()` does not touch the device cache, and the
  remaining statements are `_dirty = False`, the generation bump, the scan-cache
  clear and `_notify`.

## What was run

From `/home/nick/MSF/msf/MADDENING-wt/perf/interactive-paths`, always with
`PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`:

```
pytest tests/cloud/multigpu/test_sharded_static_cache.py -q -rs
  pre-fix  (src from `git archive HEAD`):  3 failed, 9 passed
  post-fix:                                12 passed
```

The four compliance scripts (`check_anomalies`, `check_impl_mapping`,
`check_citations`, `check_transforms`) are clean; `check_citations` emits its
two pre-existing "not cited by any algorithm guide" warnings and exits 0.

```
pytest tests/cloud/multigpu tests/core/test_node.py tests/core/test_hybrid_node.py \
       tests/core/test_static_array.py tests/core/test_static_data.py \
       tests/core/test_scan_program_cache.py tests/core/test_replace_static_sharding.py \
       tests/core/test_compile_cache.py tests/compliance -q -rs
  -> 486 passed, 10 deselected, 3 xfailed in 384s
```

The 10 deselected are the `slow` markers; the 3 xfails are the two audit
findings the property-test branch pinned, unrelated to this change.

Differential evidence came from running the same probe script against the
pre-fix tree and the fixed tree, so the before/after is code rather than
recollection.  The full suite was left to CI, per the working rule.  No timing
number is reported: three other agents were on the machine.
