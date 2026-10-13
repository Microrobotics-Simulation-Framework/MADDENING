# Audit: `perf/interactive-paths` — the two interactive-path caches

Independent, diff-focused audit of `git diff origin/release/0.4.0...HEAD`
on `perf/interactive-paths` (tip `ad38b96`).  The auditor did not write
the code under review.

## The central question

**Can either cache serve a stale result?  Yes — one of them can, and the
route is reachable through a public, documented API.**

* **The scan-program cache (`GraphManager._scan_cache`) cannot serve an
  unrecoverable stale result.**  It is keyed on `_compile_generation`,
  and `compile()` both bumps that counter and clears the dict, so the
  framework's explicit "rebuild everything" operation does invalidate
  it.  It *does* narrow what `run_scan` notices by itself (§MINOR-1),
  but `compile()` always fixes it.

* **The sharded-statics cache (`_static_device_cache` on
  `ShardedStencilNode` / `ShardedUnstructuredNode`) can, and
  `gm.compile()` does not clear it.**  A static array whose buffer is
  rewritten in place keeps the same `id()`, the same shape and the same
  dtype, so the key cannot see the change — and because the cache lives
  on the node rather than on the graph, a full `GraphManager.compile()`
  followed by a `step()` re-traces the graph and bakes the **old**
  buffer into the new program.  Before this diff, `compile()` picked the
  rewrite up.  Proven, reproduced on a single device, fixed on
  `fix/perf-interactive-audit`.

The author's admission that they "erred toward wrongly hitting rather
than missing" for an in-place rewrite is accurate as far as it goes, but
it understates the blast radius: the escape hatch they added
(`wrapper.invalidate_static_cache()`) is on the wrapper, and nothing
told a user that `gm.compile()` had stopped being sufficient.

---

## CRITICAL

None found.

---

## MAJOR

### MAJOR-1 — `gm.compile()` no longer invalidates a node's materialised statics

**What breaks.**  `ShardedStencilNode._static_device_cache` (and the
`ShardedUnstructuredNode` twin) survives `GraphManager.compile()`.  The
cache key is `(key, id(value), shape, dtype, shard_axis)`, which by
design cannot see a static array rewritten in place.  That is defensible
for a steady-state `update()`; it is not defensible for `compile()`,
which is the framework's documented way of saying "throw everything
away and rebuild".  `compile()` clears `_compiled_step`, the scan cache
and the static-data hashes, but it never reaches the node-level cache,
so the recompiled step is traced against the stale device buffer and the
simulation is silently wrong from then on.

Before this diff the same sequence was correct, because
`_materialise_sharded_statics` ran a fresh `device_put` on every trace.
This is therefore a regression, not a pre-existing hole.

**Smallest reproduction** (single device; no 4-device mesh needed):

```python
inner = Diff1D("d")                       # mask is a StaticArray(shard, axis 0)
w = ShardedStencilNode(inner, create_device_mesh(shape=(1,)),
                       axis_map={"devices": 0}, boundary="edge")
gm = GraphManager(); gm.add_node(w); gm.compile(); gm.step()
start = np.asarray(gm.get_node_state("d")["f"]).copy()

inner._mask[:] = 0.0        # rewrite the static's buffer in place
gm.compile()                # the explicit "rebuild everything" hammer
gm.set_node_state("d", {"f": jnp.asarray(start)})
gm.step()
# mask == 0 means the field must not move.  It moves.
assert np.allclose(np.asarray(gm.get_node_state("d")["f"]), start)
```

`origin/release/0.4.0`: passes.  `perf/interactive-paths`: fails.
Full script: `probes/pD.py` in the audit scratch (reproduced verbatim as
the regression test below).

**Recommended fix (implemented).**  Make `compile()` reach the hook the
author already provided.  Four lines at the end of
`GraphManager.compile()`:

```python
for spec in self._nodes.values():
    invalidate = getattr(spec.node, "invalidate_static_cache", None)
    if callable(invalidate):
        invalidate()
```

This keeps every per-frame saving the diff is for — `compile()` is rare,
and the cost is one `device_put` per sharded static per compile — while
restoring the pre-diff guarantee that a recompile is a clean slate.  It
is duck-typed, so it covers both wrappers and any future node with the
same hook.

Regression test:
`tests/cloud/multigpu/test_sharded_static_cache.py::test_compile_rematerialises_a_static_rewritten_in_place`
(and a device-independent companion asserting `compile()` calls the hook).
Both fail on `perf/interactive-paths` and pass on
`fix/perf-interactive-audit`.

**Confidence: proven.**  Reproduced against both branches, on one device
and on four.

---

## MINOR

### MINOR-1 — `run_scan` no longer notices an undirtied change to a node's Python state

`run_scan` used to call `_build_step_fn()` and re-trace on *every* call,
so any change to a node's Python-level state between two calls was
picked up for free — including one the framework's dirty-tracking cannot
see, because `static_data_hash()` hashes `(key, shape, dtype,
replication, shard_axis)` and never the contents.  With the program
cached, `run_scan` inherits exactly the staleness `step()` has always
had.

Reproduction: a node that rebuilds its static array (the documented
`static_data_provider` reconstruction — same shape, same dtype, new
object, new contents) between two `run_scan` calls.

```
BASE    scan before k-change: 1.0   scan after: 5.0   step loop: 1.0
BRANCH  scan before k-change: 1.0   scan after: 1.0   step loop: 1.0
```

I am reporting this as minor rather than major on three grounds, and I
want the reasoning on the record so it is not re-litigated:

1. `SimulationNode.static_data`'s contract already says the dict "should
   be **stable across calls** for a given node instance", and the
   `nodes_without_params()` error text already says non-params constants
   "are baked into the trace".
2. `step()`, `run()` and the whole REST layer (`/sim/step` is the only
   entry point `api/server.py` drives) have always behaved this way, so
   the change makes `run_scan` *consistent* rather than newly wrong.
3. `gm.compile()` clears the scan cache and bumps `_compile_generation`,
   so the recovery path exists and works.

Recommended (docs only, not implemented per the brief): say in
`run_scan`'s docstring and in `docs/developer_guide/profiling.md` that
the scan program is now built once per `compile()` and that a change the
dirty-check cannot see needs an explicit `gm.compile()` — the same
sentence `step()` deserves.

**Confidence: proven** (the behaviour change), **judgement** (the
severity).

### MINOR-2 — the scan program is built lazily, so it and `_compiled_step` can be built from different snapshots

`_cached_scan`'s docstring claims "a cached scan is exactly as fresh as
`_compiled_step`".  That is not quite true: `_compiled_step`'s `step_fn`
is built inside `compile()`, while each scan program calls
`self._build_step_fn()` again at the first `run_scan` after that
compile — arbitrarily later.  Anything the builder reads at build time
(node Python state, `self.params` as the `params=None` snapshot) can
differ between the two, so `gm.step()` and `gm.run_scan()` on the same
graph can return different trajectories.  The claim would become true if
`compile()` stashed its `step_fn` and the scan builders reused it.

I did **not** implement that: `_build_step_fn` closes over a mutable
`flux_state` dict that it writes to during tracing (`graph_manager.py`,
in `_resolve_and_update_node`), and sharing one `step_fn` between
`_compiled_step` and every scan program would share that dict across
traces too.  I convinced myself it is harmless today — a flux producer
is always scheduled before its consumer, so the entry is overwritten
before it is read — but it is not a change to make blind inside an
audit.  Worth a separate look.

**Confidence: proven** (the lazy build), **believed** (the `flux_state`
reasoning).

### MINOR-3 — WITHDRAWN: the new sharded tests *do* run in CI

I first reported that every test in the new file is
`@pytest.mark.skipif(not _HAS_4_DEVICES)` and that no workflow sets
`--xla_force_host_platform_device_count`, so the file would be skipped
on a 1-CPU runner.  That is wrong, and I am leaving it in the record
rather than deleting it.  `tests/cloud/multigpu/conftest.py` forces
16 virtual host devices (rule 4 of its policy) before JAX is imported
whenever `JAX_PLATFORMS=cpu`, which the root `tests/conftest.py` sets by
default.  `_HAS_4_DEVICES` is therefore true under pytest on a plain CPU
runner, and the file runs.  Verified: `9 passed` with no XLA flag set on
the command line.  No action needed.

### MINOR-4 — `test_param_write_through_the_rest_layer_reaches_a_derived_static` does not test what it says

The fixture node's `update` takes no `params` keyword, so
`accepts_params` is `False` and `params_pytree()` is `{}`.  The REST
handler therefore takes its `else` branch, writes `node.params[key]` and
sets `gm._dirty = True`; the following `gm.step()` recompiles, and the
new mask arrives through the recompile rather than through the cache's
invalidation logic.  The test would pass with the identity check
removed.  Not a defect in the shipped code — just a test that is not
pinning the invariant its name claims.

### MINOR-5 — in-place mutation on the eager `update()` path

Independently of MAJOR-1, a direct `wrapper.update(...)` loop now ignores
an in-place rewrite where it used to honour it (base: honoured; branch:
ignored; `invalidate_static_cache()` restores it).  With MAJOR-1 fixed
this is a documented, contract-backed tradeoff
(`StaticArray.value`: "held by reference; do not mutate after
wrapping"), and the docs change in `sharded_static_data.md` says so.
Recording it so the tradeoff is explicit rather than implicit.

---

## Verified safe

Things I chased and found sound.  Recording them so the same worry is
not re-opened.

* **`id()` reuse after garbage collection cannot cause a false hit.**
  Both caches store the classified `StaticArray` dict alongside the key
  (`self._static_device_cache = (key, out, sharded)`), and
  `coerce_static_data_value` returns the *same* `StaticArray` object for
  a `StaticArray` input (`static_data.py`), so every `id()` in a live
  key names an object the cache itself keeps alive.  A recycled address
  can therefore never collide with a key that is still in the cache.
  The old entry is only dropped after the new key has been computed, so
  there is no window either.  This is the only structural defence the
  scheme has and it holds.

* **A parameter written through the now-shared `self.params = node.params`
  dictionary is not masked by a cached static.**  Two independent
  routes checked: (a) a float leaf in `params_pytree()` — the REST
  handler writes both `gm.params` and `node.params`, and `gm.params` is
  an *argument* of the cached scan, so the write lands without a
  rebuild; (b) a key outside `params_pytree()` — the handler's `else`
  branch sets `gm._dirty`, which forces a `compile()`, which clears the
  scan cache.  Where a derived static is involved, MAJOR-1's fix is what
  makes route (b) actually correct end to end.

* **Shape/dtype-preserving edits that replace the array object are
  caught.**  A node handing back a different array object misses the key
  and re-materialises; verified by the author's tests and independently.

* **The scan cache is bounded and has a working invalidation path.**
  `_SCAN_CACHE_MAX = 64` with FIFO eviction, plus a full clear on every
  `compile()`.  No unbounded growth across a long interactive session.
  `_static_device_cache` is a single slot per node, replaced on miss —
  also no growth.  (The pre-existing `_sharded_cache` on the wrappers
  *is* unbounded, but this diff only adds `clear()` calls to it; it does
  not widen that.)

* **The same node object in two graphs is safe.**  `_static_device_cache`
  depends only on the node's own mesh/layout, not on the graph, so two
  `GraphManager`s wrapping one node share one correct entry.
  `_scan_cache` is per-`GraphManager`.

* **The non-interactive path is numerically unchanged.**  Moving
  `ext`/`params`/`state` from closed-over constants to jitted arguments
  is inert; `test_run_scan_still_matches_the_step_loop` pins it against
  a `step()` loop.  `sysid.py` is the only in-tree consumer of
  `run_scan_with_history`, and it passes `params=` explicitly, so the
  values flow as jitted arguments and it gets the cache for free with no
  semantic change (read-verified; I did not run the sysid suite — CI
  does).  One caveat I could not remove by reading: the scan program is
  built at the first call rather than at `compile()`, so a fit whose
  first `run_scan` happens *inside* `jax.grad` builds the program from
  the traced avals; JAX's own cache retraces for the eager avals
  afterwards, which is correct but costs one extra compile.

## Confirmed / refuted from the prior audit (`audit_fixset-2026-09-18` §X9)

* **CONFIRMED: `gm.compile()` does not retrace `ShardedStencilNode._sharded_cache`.**
  A non-array param change (`node.params["mode"] = "fast"`) plus
  `gm._dirty = True` plus an explicit `gm.compile()` produced a
  bit-identical trajectory to the pre-change one, and a different one
  from a freshly constructed `mode="fast"` wrapper.  Reproduced
  identically on `origin/release/0.4.0`, so it is **pre-existing** and
  **not** caused by this diff.

* **On whether the new statics cache widens that window:** it does not
  widen `_sharded_cache` itself — the diff only *adds* `clear()` calls
  to it.  But it did create a second instance of the same pattern, and
  that one (MAJOR-1) was worse, because until this diff the statics
  flowed as fresh runtime arguments into the stale `shard_map` and so
  `compile()` still fixed the values even when it could not fix the
  trace.  That safety net is what the diff removed and what the fix
  restores.

* **On whether `_compile_generation`'s reasoning is sound or merely
  sound-sounding:** sound, but narrower than the docstring implies.  The
  chain "every mutation dirties, every entry point compiles a dirty
  graph, `compile()` clears the cache" holds: `_dirty` is set to `False`
  in exactly one place (`compile()`), which is also the only place
  `_compiled_step` and `_compile_generation` change.  So the scan cache
  really is exactly as fresh as `_compiled_step`.  What is *not* true is
  the unstated corollary that `_compiled_step` is itself always fresh —
  `_sharded_cache` (pre-existing) and `_static_device_cache` (MAJOR-1)
  are both counterexamples.  The scan cache faithfully inherits their
  staleness rather than adding its own.

* **`SimulationNode.to_dict()` returning `self.params` by reference**
  (`core/node.py`): confirmed by reading — it is another route by which
  a node's parameter dict changes with no cache key changing.  It is
  pre-existing and orthogonal to both caches here (params reach the
  cached scan as arguments), so it is not a finding against this diff.

## What was run

With the MAJOR-1 fix applied, from
`/home/nick/MSF/msf/MADDENING-wt/perf/interactive-paths`:

```
PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests/cloud/multigpu tests/core/test_scan_program_cache.py \
                   tests/api/test_params_endpoint_live_view.py -q -rs
  -> 262 passed, 10 deselected (the `slow` markers) in 243s

scripts/check_anomalies.py  check_impl_mapping.py
scripts/check_citations.py  check_transforms.py           -> all clean
```

Before/after on the regression tests (the two new ones in
`test_sharded_static_cache.py`): `2 failed, 7 passed` without the
`compile()` change, `9 passed` with it.

Differential evidence for MAJOR-1 and MINOR-1 came from running the same
probe script against `origin/release/0.4.0` (extracted with `git archive`
into a scratch tree) and against the branch, so the comparison is code,
not memory.  The full suite was left to CI.

Timings were deliberately not measured: the machine is shared.
