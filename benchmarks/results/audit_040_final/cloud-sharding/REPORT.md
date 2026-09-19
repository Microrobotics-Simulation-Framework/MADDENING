# Audit: cloud + sharding   (c51cd6ad075cb8c0d4a03f4ab49ff0f5dcca4885)

Surface: `src/maddening/cloud/` (`resume.py`, `entrypoint.py`, `__init__.py`,
`multigpu/`) and `src/maddening/api/server.py`.
Worktree: `/home/nick/MSF/msf/MADDENING-wt/audit/cloud-sharding`, detached at
`origin/release/0.4.0`.  All work on CPU (`JAX_PLATFORMS=cpu`); the GPU was not
touched and nothing ran remotely.

## Summary

The numerical core of the sharding work is sound: **sharded equals unsharded to
the bit** (max abs diff `0.0e+00`) for the stencil and unstructured wrappers at
1/2/3/4 devices, and the **sharded adjoint matches the unsharded adjoint** to
float32 roundoff.  Resume integrity is also sound - truncation and single-bit
corruption are both caught by the SHA-256 manifest, concurrent resumes do not
collide, and a resumed trajectory is bit-identical to an uninterrupted one.

Six real defects.  The two that matter most: a **failed resume leaves the graph
half-restored while the entry point logs "starting fresh"**, and the **HTTP API
has no authentication at all while every shipped cloud path binds it to
`0.0.0.0` and opens the cloud firewall** - the opposite of what the release
notes tell users to do.  Below that: `ShardedPointwiseNode` silently ignores its
`shard_axes` argument, a `**kwargs` inner node silently loses injected `params`
(zero gradient), two unauthenticated endpoints let a caller choose unbounded CPU
and memory, and stencil sharding has an undocumented, untested divisibility
requirement.

Reproducers and their captured output are in this directory (`repro_*.py` /
`repro_*.out`).  Nothing was fixed.

---

## Findings

### CRITICAL - the HTTP API has no authentication, and every shipped cloud path exposes it to the internet

**What breaks:** `POST /cloud/launch`, `POST /cloud/teardown`, `PUT
/graph/state/{node}`, `DELETE /graph/nodes/{name}`, `POST /checkpoint/save` and
~55 other routes are reachable with no credential of any kind. `/cloud/launch`
provisions paid cloud GPUs using the host's stored provider credentials;
`/cloud/teardown` destroys them.

**Evidence:** `repro_server_security.out`, section 1 - no credentials sent:

```
   GET    /graph                   -> 200
   GET    /graph/state             -> 200
   POST   /sim/step                -> 200
   POST   /graph/compile           -> 200
   DELETE /graph/nodes/relax       -> 200
```

`grep -n "Depends\|HTTPBearer\|api_key\|middleware" src/maddening/api/server.py`
returns nothing.  `/docs`, `/redoc` and `/openapi.json` all return 200, so the
full route list is self-serving.

**Why it happens:** the absence of auth is *known* and documented -
`src/maddening/api/server.py:242-245` ("Bind the server to localhost or put it
behind auth"), `docs/release_notes/v0.4.0.md:890-893`, `CHANGELOG.md:348-352`,
and `docs/regulatory/intended_use.md:35-36` ("the FastAPI server is a
development tool, not a production deployment surface").

The finding is that **every shipped deployment path does the opposite of that
advice**:

- `src/maddening/cloud/entrypoint.py:81` - `host = os.environ.get("MADDENING_HOST", "0.0.0.0")`
- `docker/Dockerfile.cloud:88-93` - `EXPOSE 8000 ...` + `ENV MADDENING_HOST=0.0.0.0`
- `src/maddening/cloud/_skypilot.py:51-53` - `docker run ... -p 8000:8000 ...`
- `src/maddening/cloud/launcher.py:130` - `ports: list[int] = field(default_factory=lambda: [8000])  # Ports to expose via RunPod NAT`, consumed at `:718-729` by `sky.Resources(..., ports=job_config.ports or [8000])`, which opens the cloud firewall / security group
- `src/maddening/cloud/launcher.py:344-376` - `get_runpod_endpoint()` returns the **public** NAT mapping, and the shipped examples (`src/maddening/examples/cloud/server/04_server_test.py:234-242`) then drive the API over it

So the default `maddening[cloud]` launch publishes an unauthenticated,
TLS-less, fully graph-mutating API on a public IP.  No `ssl_keyfile`,
`ssl_certfile` or reverse-proxy config exists anywhere in the repo.  The
"bind to localhost" guidance appears only in the changelog, the release notes,
a regulatory boundary statement and a source comment - never in `README.md`,
`docs/user_guide/`, `src/maddening/api/README.md`, `docker/Dockerfile.cloud`
or `src/maddening/examples/cloud/README.md`, which are where a user
following the cloud quickstart actually looks.

**What would make this a non-issue:** if the launcher put the port behind a
private network, an SSH tunnel, or a token. It does not - `sky.Resources(ports=)`
is an ingress rule and `get_runpod_endpoint()` exists precisely to hand back
the public address.  It would also be a non-issue if the docs a cloud user
reads said "do not do this"; I checked the five entry-point docs listed above
and none does.

**Suggested fix:** default `MADDENING_HOST` to `127.0.0.1` and make `0.0.0.0`
an explicit opt-in; drop `8000` from `JobConfig.ports`' default so the API is
reached through an SSH tunnel unless the user asks otherwise; add a shared-secret
header check (a dozen lines, `Depends`) enabled whenever the bind address is not
loopback.  Risk: this breaks the shipped `examples/cloud/server/*` scripts and
anything that currently talks to a RunPod public endpoint, so it needs a
release note and a migration line - but shipping 0.4.0 as-is means the
documented posture and the actual posture disagree.

---

### HIGH - a failed resume leaves the graph half-restored, and the entry point logs "starting fresh"

**What breaks:** resume a checkpoint into a graph whose parameter *shapes* have
changed (a redeploy with a code change - exactly the resume-after-preemption
case).  `load_state` applies every node's saved state, *then* validates
parameters and raises.  The node states stay applied.
`entrypoint.resume_from_env` catches the exception, logs `"Failed to resume
from ...; starting fresh"` and returns `None`, so the operator believes the run
started clean.  It did not: it is running checkpoint state with fresh
parameters.

**Evidence:** `python repro_resume_partial.py` -> `repro_resume_partial.out`:

```
saved  state x = [ 7.  8.  9. 10.]
saved  gainvec = [2. 2. 2.]

target graph BEFORE resume: x = [0. 0. 0. 0.], gainvec = [1. 1. 1. 1.]
resume FAILED as expected: ValueError: Checkpoint params nodes['relax']['gainvec'] has shape (3,), graph has (4,)
target graph AFTER  resume: x = [ 7.  8.  9. 10.], gainvec = [1. 1. 1. 1.]

*** STATE WAS MUTATED BY THE FAILED RESUME ***
    x went [0. 0. 0. 0.] -> [ 7.  8.  9. 10.] despite 'starting fresh'
```

**Why it happens:** `src/maddening/core/simulation/checkpoint.py:224-225` applies
all staged node states:

```python
for node_name, new_state in staged_states.items():
    graph_manager.set_node_state(node_name, new_state)
```

`_restore("nodes", param_keys)` runs afterwards at `:258` and raises at `:251-255`
on a shape mismatch.  `_restore` also mutates `current[owner][pname]` in place as
it iterates, so parameters themselves can be left half-applied across nodes.
`src/maddening/cloud/entrypoint.py:152-157` then swallows it:

```python
except Exception:
    # Non-fatal: log and continue with the in-memory state.
    logger.exception("Failed to resume from %s; starting fresh", shown)
    return None
```

The comment says "continue with the in-memory state" - but the in-memory state
is no longer the fresh one.

**What would make this a non-issue:** if the node-state apply were staged past
the parameter restore, or if the entry point rebuilt the graph on failure.
Neither happens - I checked that the *node-name*, *field-name* and *state-shape*
validations at `:188-221` all run before any mutation (those failures **are**
atomic, which is why this only bites through the params path), and that
`resume_from_env` has no rollback.

**Suggested fix:** stage the parameter restore the same way node states are
staged - validate every leaf of both sections first, then apply both.
Alternatively have `resume_from_env` snapshot `gm._state` + `gm.params` before
calling and restore them in the `except`. The staging fix is cheap and local;
its only risk is that a caller currently relying on the partial write (nothing
in the repo does) would change behaviour.

---

### MEDIUM - `ShardedPointwiseNode` silently ignores `shard_axes`

**What breaks:** `ShardedPointwiseNode(node, mesh, shard_axes=(1,))` shards axis
0 regardless.  On a `(3, 8)` state with 4 devices - axis 1 divides, axis 0 does
not - the request fails with an `IndivisibleError` that blames axis 0, an axis
the caller explicitly did not select.

**Evidence:** `repro_pointwise_axis.out`, section B:

```
== B. shard_axes=(1,) on a (3, 8) state, 4 devices ==
   axis 0 = 3 (NOT divisible by 4); axis 1 = 8 (divisible by 4)
  shard_axes=(0,): RAISED IndivisibleError: One of device_put args was given the sharding NamedSharding(...
  shard_axes=(1,): RAISED IndivisibleError: One of device_put args was given the sharding NamedSharding(...
```

Identical failure for both, from `sharded_node.py:121`.

**Why it happens:** `src/maddening/cloud/multigpu/sharded_node.py:104` hardcodes

```python
self._sharding = NamedSharding(mesh, P("devices"))
```

`P("devices")` always names axis 0.  `self._shard_axes` is consulted in exactly
one place, `:120`, as a *rank* test (`if arr.ndim > self._shard_axes[0]`), never
to choose the axis.  Every test in the repo passes `shard_axes=(0,)`
(`test_sharded_node.py:79-86`, `test_sharded_params.py:128,223`,
`property_support.py:426`), so the bug is invisible to the suite.

**What would make this a non-issue:** if `shard_axes` were documented as
axis-0-only. It is not - `sharded_node.py:68-72` documents it as "Which axes of
the state arrays to shard", and only *multi*-axis is rejected
(`NotImplementedError` at `:91`), which implies single non-zero axes work.

Note separately that `ShardedPointwiseNode.update` (`:126-137`) contains no
`shard_map` and no `device_put` - it delegates straight to the inner node.
Sharding survives only if the caller feeds it arrays that `initial_state`
already placed. Fed an unsharded array, `update` returns
`SingleDeviceSharding(device=CpuDevice(id=0))` (`repro_pointwise_axis.out`,
section C). That is defensible for a pointwise op under XLA's SPMD propagation,
but it means the wrapper silently degrades to one device if anything in the
graph round-trips state through an unsharded path.

**Suggested fix:** build the spec from `shard_axes`, or reject `shard_axes !=
(0,)` with `NotImplementedError`. The second is a one-liner and honest; the
first is the documented behaviour. Risk: near zero - no caller in the repo uses
a non-zero axis.

---

### MEDIUM - an inner node with `**kwargs` silently loses injected `params`; its gradient is zero

**What breaks:** a node whose `update_padded` is declared `(self, state_padded,
boundary_inputs, dt, **kwargs)` receives `static_padded` and `shard_info` (the
wrapper honours var-keyword for those) but **not** `params`.  The step silently
uses the constructor constant, and `d(loss)/d(param)` comes back exactly `0.0`
with no warning.

**Evidence:** `repro_wrapping_params.out`, sections 2 and 5:

```
   _inner_accepts_static_padded = True
   _inner_accepts_shard_info    = True
   _inner_accepts_params        = False   <-- var-keyword NOT recognised (sharded_node.py:284)
   injected rate=0.9 honoured? err vs reference = 2.344e-03 *** PARAMS SILENTLY DROPPED ***
   with params vs without params: max diff = 0.000e+00 (params had NO effect)
...
   unsharded reference           d/d rate = -0.08005735
   sharded, explicit params kwarg d/d rate = -0.08005735
   sharded, **kwargs inner        d/d rate = 0.00000000   <-- SILENTLY ZERO
```

**Why it happens:** `src/maddening/cloud/multigpu/sharded_node.py:277-284`:

```python
self._inner_accepts_static_padded = ("static_padded" in params or has_var_kw)
self._inner_accepts_shard_info    = ("shard_info" in params or has_var_kw)
self._inner_accepts_params        = "params" in params      # <-- no `or has_var_kw`
```

Two lines out of three consider var-keyword; the third does not.  `_local_update`
then skips the kwarg at `:489-490` (`if accepts_params and local_params:`), so
the inner falls back to `self.params` and the injected leaf never enters the
trace - hence the exact zero.

**What would make this a non-issue:** if no node could be written that way.  A
`**kwargs` `update_padded` is a signature the wrapper visibly supports for the
other two kwargs, so a user would reasonably expect it to work.  Reachability
is the mitigating factor and the reason this is MEDIUM not HIGH: I grepped every
`update_padded` in `src/` (`heat.py:402`, `lbm.py:748`, `node.py:678`,
`sharded_node.py:319`) and **none uses var-keyword**, and
`docs/developer_guide/node_authoring.md:396` and `sharded_static_data.md:124`
both show the explicit `*, static_padded=None, shard_info=None` form.  So no
shipped node is affected today; this is a trap for user-authored nodes, and when
it fires it produces wrong physics and a zero gradient with no error.

**Suggested fix:** add `or has_var_kw` to line 284. One-line, and it makes the
three probes consistent. Risk: a node that takes `**kwargs` but genuinely cannot
accept `params` would start receiving it - but such a node is already broken for
`static_padded`.

---

### MEDIUM - two unauthenticated endpoints let the caller choose unbounded CPU and memory

**What breaks:** `POST /sim/run?n_steps=N` has no upper bound on `N`; a single
request occupies a worker thread indefinitely with no API-level way to cancel it.
`POST /graph/nodes` accepts an integer constructor parameter that becomes an
array dimension, so one request allocates as much memory as the caller names.

**Evidence:** `repro_server_dos.out`:

```
== A. /sim/run: no upper bound on n_steps ==
   n_steps=100    -> 200 in   0.157s  (1566.9 us/step)
   n_steps=1000   -> 200 in   0.100s  ( 100.0 us/step)
   n_steps=10000  -> 200 in   0.867s  (  86.7 us/step)

== B. add_node: caller chooses the state array size ==
   RSS before:    233.8 MB
   n=1000000     -> 201   RSS now    240.5 MB (+    6.7 MB)   expected array 4 MB
   n=10000000    -> 201   RSS now    283.7 MB (+   49.9 MB)   expected array 40 MB
   n=100000000   -> 201   RSS now    666.9 MB (+  433.1 MB)   expected array 400 MB
```

(Timings come from a machine shared with five other agents - the finding is the
linear scaling in an unclamped caller-chosen value, not the absolute numbers.
Flag for re-measurement on a quiet box if anyone wants the constants.)

**Why it happens:** `src/maddening/api/server.py:657-663` passes `n_steps`
straight to `self.gm.run(n_steps)`.  The neighbouring `/sim/profile` *does*
clamp - `:995`, `n_steps = max(1, min(1000, int(n_steps)))` - which shows the
authors considered the problem for one endpoint and not the other.  For
`add_node`, `_non_finite_param` (`:134`) rejects a non-finite float but nothing
bounds an integer, and `_dry_run_node` (`:168`) traces abstractly so it never
observes the size.  `TrainSurrogateRequest` (`:122-127`) likewise has no
`Field(le=...)` on `n_data_steps`, `n_epochs`, `hidden_sizes` or `batch_size`.

**What would make this a non-issue:** a request size limit or a reverse-proxy
timeout in front. There is none in the repo, and per the CRITICAL above the
server is reachable from the internet in the default cloud config.

**Suggested fix:** clamp `n_steps` in `/sim/run` the way `/sim/profile` already
does, add `Field(ge=1, le=...)` bounds to `TrainSurrogateRequest`, and cap total
state elements in `add_node`. Risk: a legitimate long run now needs repeated
calls or a runner (`/sim/start`), so the cap should be generous and configurable.

---

### MEDIUM - stencil sharding silently requires the cell count to divide the device count; the failure is an opaque JAX error and the suite never tests it

**What breaks:** `ShardedStencilNode` on 16 cells over 3 devices, or 18 cells
over 4, raises `jax.errors.IndivisibleError` from deep inside `device_put`.
Nothing in `ShardedStencilNode.__init__` validates divisibility, and nothing in
the docs mentions the constraint.

**Evidence:** `repro_divisibility.out` (all values exact where it works):

```
devices=2 n_cells=16 (divides=True)              max|sharded-unsharded| = 0.000e+00  OK
devices=2 n_cells=17 (divides=False)             RAISED IndivisibleError: One of device_put args was given the sharding NamedSharding(mesh=Mesh('x': 2, ...
devices=3 n_cells=16 (divides=False)             RAISED IndivisibleError: ...
devices=4 n_cells=18 (divides=False)             RAISED IndivisibleError: ...
devices=4 n_cells=20 (divides=True)              max|sharded-unsharded| = 0.000e+00  OK
```

**Why it happens:** `sharded_node.py:306-311` (`initial_state`) and `:705`
(`_materialise_sharded_statics`) both `device_put` with a `NamedSharding` whose
spec shards a dimension JAX cannot split evenly.  `__init__` validates axis
names (`:218`) and halo coverage (`:227`) but never divisibility.

`grep -rni "divisib\|must divide" docs/ src/maddening/cloud/ CHANGELOG.md`
returns nothing, and `tests/cloud/multigpu/property_support.py:66` pins
`DEVICE_COUNTS = tuple(n for n in (1, 2, 4) if n <= _AVAILABLE_DEVICES)` - 3 is
excluded by construction, so no property ever generates a ragged stencil shard.

**What would make this a non-issue:** if the unstructured path had the same
limitation it would be a framework-wide given rather than a wrapper bug. It does
not - `ShardedUnstructuredNode` handles **every** combination correctly, because
it carries an explicit padded layout: `repro_unstructured_odd.out` shows
`max err = 0.000e+00` for all twelve (device, cell) pairs including 3 devices /
17 cells. So the constraint is specific to the stencil wrapper.

**Suggested fix:** validate in `ShardedStencilNode.__init__` - the mesh extent
per axis is known there, and the grid extent is available from
`node.initial_state()` - and raise a message naming the axis, the cell count and
the device count. Real padding support is a much larger change. Risk of the
validation alone: none; it converts an opaque late error into an early clear one.

---

### LOW - `resume._local_path` does not percent-decode a bare path, contrary to its docstring

**Evidence:**

```
   /tmp/a%20b.npz             -> /tmp/a%20b.npz
   file:///tmp/a%20b.npz      -> /tmp/a b.npz
```

**Why it happens:** `src/maddening/cloud/resume.py:226-227` returns `Path(url)`
unchanged for `scheme == ""`, while the `file://` branch at `:233` calls
`urllib.parse.unquote`.  The docstring at `:104-105` says a bare POSIX path "is
treated the same" and "the path is percent-decoded".  Meanwhile the *local
destination* filename at `:172` **is** unquoted for both, so the two disagree
within one function.  Consequence is limited to a confusing `FileNotFoundError`
on a local path containing a literal `%`.

**Suggested fix:** pick one. Not decoding a bare path is the safer behaviour (a
POSIX filename may legitimately contain `%`); fix the docstring rather than the
code.

---

## Unverified suspicions

- **The only authentication code in the package never rejects anything.**
  `src/maddening/cloud/selkies_session.py:233-238` guards the WebRTC signaling
  WebSocket. The outer test looks at the client's query string, but the inner
  `validate_session_token(self._session_id, token, self._secret)` re-validates
  the **server's own** token, generated at `:95` from the same `session_id` and
  `secret` - so it is unconditionally `True` and `await ws.close(1008, "Invalid
  token")` is unreachable. I confirmed the predicate is always true
  (`validate_session_token(sid, generate_session_token(sid, secret), secret)` ->
  `True`) but did **not** stand up a live signaling server to drive an
  unauthorised client through it, so I am listing it here rather than as a
  finding. The socket binds `0.0.0.0:8443` (`:252`) and the entry point
  constructs `SelkiesSession()` with no secret (`entrypoint.py:47`), so the
  secret is a fresh `uuid4()` nobody holds. `selkies_session.py` is in my
  package but outside the files I was assigned - worth handing to whoever owns it.

- `graph_manager.compile()` is called *inside* `load_state`
  (`checkpoint.py:180-182`) when the graph is dirty. A compile that raises there
  would leave the resume even further from atomic than the HIGH above describes.
  I did not construct a graph that fails to compile at that point.

- `resume.download_and_load_state` streams an HTTP body to disk with no size
  limit (`resume.py:252-256`). The URL comes from the operator's own
  `RESUME_FROM_URL`, so this is only a hazard if that env var is
  attacker-influenced. I did not find a path where it is.

## What I checked and found sound

- **Sharded == unsharded.** Exact (`0.0e+00`) for `ShardedStencilNode` at 1/2/3/4
  devices across every divisible cell count tried (12/16/17/18/20), including
  halo exchange at periodic boundaries, a `replication="shard"` `StaticArray`,
  a grid-shaped boundary input and a replicated scalar. Exact for
  `ShardedUnstructuredNode` at **all** twelve (device, cell) pairs including the
  ragged ones. Exact for `ShardedPointwiseNode` on the default mesh.
- **Gradients through sharding.** `d(loss)/d(rate)` and `d(loss)/d(f0)` through a
  sharded step match the unsharded adjoint at 1/2/4 devices: `err_rate` <= 7.5e-09,
  `err_f0` = 2.4e-07 (float32 roundoff). **This tests the sharding logic on CPU
  virtual devices, not the real multi-GPU collective path** - the project records
  the latter as never verified on hardware, and this audit does not change that.
- **Nesting.** `HybridNode(ShardedStencilNode(inner))` steps correctly
  (err `0.0e+00`) and proxies the params contract (`accepts_params=True`,
  `params_pytree` keys `['rate']`).
- **Params pytree reaching a wrapped node.** A sharded node inside a
  `GraphManager` gets its entry in `gm.params["nodes"]`, and writing that entry
  changes the next step (max diff 8.8e-03) - the write is not stranded on the
  wrapper. Injected `params=` reaches `update_padded` identically at 1/2/4
  devices for explicitly-declared signatures.
- **Resume integrity.** Truncated checkpoint -> `CheckpointIntegrityError` (SHA
  mismatch). Single-bit flip at the same length -> `CheckpointIntegrityError`.
  `.npz` written but manifest missing (crash mid-save) -> `FileNotFoundError`
  naming the manifest. All three are caught *before* any state is touched.
- **Resume determinism.** 4 steps -> checkpoint -> resume -> 6 steps is
  bit-identical to 10 uninterrupted steps (max diff `0.0e+00`).
- **Concurrent resumes.** 8 threads through `download_and_load_state` against one
  `file://` URL: 8 succeeded, 0 failed, all results identical. The per-call
  `tempfile.mkdtemp` at `resume.py:166` does its job.
- **Resume across a changed device count** is a non-question by construction:
  `save_state` calls `np.asarray(value)` per field (`checkpoint.py:96`), which
  gathers to a single global host array, so checkpoints carry no device-count
  information.
- **`/checkpoint/{save,load}` path containment holds.** `..` traversal, an
  absolute path, a deep `../../../..` chain and a **symlink** pointing out of the
  root are all rejected with 400; `..%2F` and `....//` are treated as literal
  filenames and stay inside. Nothing was written outside the root
  (`glob /tmp/audit_escape*` -> `[]`). `_checkpoint_path` (`server.py:614-622`)
  resolves before comparing, which is what makes the symlink case safe.
- **No unsafe deserialisation.** No `pickle` anywhere on this surface;
  `np.load(..., allow_pickle=False)` at `checkpoint.py:149`; `_python_to_jax`
  (`server.py:78-86`) only converts numbers and lists. `PUT
  /graph/state/{node}` validates field set, dtype, shape and finiteness and
  stages before applying - a rejected write changes nothing.
- **Presigned-URL redaction.** `entrypoint.redact_url` and `resume._redact` both
  strip the query string and fragment before logging, so signatures do not reach
  the container log.
- **Test suite baseline**, this worktree, this commit:
  ```
  PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    python -m pytest tests/cloud tests/api -q -p no:cacheprovider -rs
  666 passed, 3 skipped, 11 deselected, 3 xfailed in 554.76s
  ```
  The three skips are honest and explained (two need the optional `fsspec`
  extra; one needs `schema_version >= 3` to exist). The property sharded tests
  under `tests/cloud/multigpu/test_property_sharded*` are included in that run
  and pass.
