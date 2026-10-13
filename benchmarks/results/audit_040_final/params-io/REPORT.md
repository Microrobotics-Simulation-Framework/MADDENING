# Audit: params-io — sysid, core/params, serialization, usd, fmi   (c51cd6ad075cb8c0d4a03f4ab49ff0f5dcca4885)

## Summary

Four HIGH findings, all on the trust boundaries rather than the maths.  The FMU
TCP bridge has no read timeout and no connection cap, so a single TCP connect
that sends nothing parks its only instance slot for ever and leaks a thread per
connection (H1).  `set_state` walks past every value check `set` applies, so an
importer can write a parameter outside its declared `ParamSpec` bounds and put
`inf`/`NaN` into the node state (H2).  `load_graph_from_usd` calls
`importlib.import_module` on a string taken from the stage, so opening an
untrusted `.usda` runs that module's import-time code (H3, pre-existing on
`main`).  `fim`'s documented `crb = +inf` fail-safe inverts to `crb = 0.0` — the
most trustworthy value it can report — whenever the Fisher matrix is NaN (H4).

The C wrapper is the strongest part of the surface: 80k fresh ASan+UBSan fuzz
iterations and a targeted probe of every request-building buffer found nothing.
`ParamSpec` round-trips exactly through config and USD (unicode, embedded NUL,
subnormal and infinite bounds, logit); the trainable mask is honoured
leaf-for-leaf; `fim`'s naming and index alignment are correct with array and
zero-size leaves.  317 tests over `tests/fmi tests/usd
tests/property/test_round_trips.py tests/property/test_sysid_contract.py` pass.

Reproducers are in `repro/`; run each with
`PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu <venv>/bin/python repro/<name>.py`.

## Findings

### HIGH — one TCP connect that sends nothing wedges the FMU bridge permanently, and every connection leaks a thread

**What breaks:** `FmuTcpBridge` serves one FMU instance at a time, guarded by
`self._busy`.  `_serve_conn` sets `conn.settimeout(None)` and then blocks in
`_recv_exact` while holding that lock.  A peer that connects and sends nothing —
a crashed importer, a dropped link, a port scanner, an attacker — holds the lock
for ever.  Every later connection is answered *"bridge already serves an FMU
instance; start one FmuTcpBridge per instance"*, and nothing short of restarting
the sidecar process recovers it.  Separately, a *refused* connection also blocks
for ever in `recv_message`, so the thread count grows without bound.

**Evidence:**

    $ python repro/r5c_bare_connect.py
    real importer's hello -> (False, b'{"ok":false,"error":"bridge already serves an
    FMU instance; start one FmuTcpBridge per instance"}')

    $ python repro/r5_bridge_wedge.py
    before the attack:
      honest client: (False, b'{"ok":true,"token":"a607723c-...","protocol":2,...}')
    attacker: 4-byte prefix announcing 100 bytes, body never sent, socket kept open
      bridge._busy held by the stalled connection: True
    for the next 10 s every honest client is refused:
      honest client t+ 0.0s: (False, b'{"ok":false,"error":"bridge already serves ...
      honest client t+ 9.0s: (False, b'{"ok":false,"error":"bridge already serves ...
    still wedged: True | serving thread: ['Thread-2 (_serve_conn)']

    $ python repro/r19_thread_leak.py
    300 idle connections -> live threads: 300 (bridge threads before: 2)

`repro/r5b.py` prints the wedged thread's stack:
`_serve_conn -> recv_raw -> _recv_exact -> conn.recv`.

**Why it happens:** `src/maddening/fmi/tcp_bridge.py:326` (`conn.settimeout(None)`),
:330 (`self._busy.acquire(blocking=False)`), :122-129 (`_recv_exact` loops on a
blocking socket), :315 (a daemon thread per connection, no cap).  `stop()` closes
only the listening socket; the parked `_serve_conn` threads survive it.

**What would make this a non-issue:** (a) if the bridge were only ever reached by
a well-behaved local process — but the module docstring says *"The importer is
**untrusted**"* and qualifies only the network ("Bind the bridge to 127.0.0.1
unless the network is trusted"), and a *crashed* importer that never sends FIN
reaches the same state with no attacker at all; (b) if `stop()` cleared it —
checked, it returned in 0.00 s with the thread still parked; (c) if the accept
loop re-checked the lock — it does not.

**Suggested fix:** give the accepted socket a finite timeout and treat
`socket.timeout` in `_serve_conn` the way EOF is already treated; move the
`settimeout` before the `_busy` probe so a refused connection cannot park
either; cap the number of live connection threads.  Risk: a legitimate importer
idle between `doStep`s longer than the timeout would be dropped, so the timeout
must be generous, or a keepalive used instead of a hard idle timeout.

---

### HIGH — `set_state` bypasses every value check `set` applies: out-of-bounds parameters and `inf`/`NaN` state

**What breaks:** `FmuTcpBridge._set` refuses a value a variable's dtype cannot
hold, and `FmuSidecar.set_params` refuses a value outside the leaf's declared
`ParamSpec.bounds`.  `_decode_state` checks the token, the key set and the
shapes, then writes with `jnp.asarray(arr, dtype=live.dtype)` — no bounds check,
no dtype-fit check.  An importer sends an otherwise valid FMU-state archive and
gets `mass = -1.0` (declared bounds `(0.1, 10.0)`), `position = inf` (a float64
`1e300` narrowed into a float32 field) and `velocity = NaN`.  The next `step`
turns the whole state into NaN and the bridge answers `ok`.

**Evidence:**

    $ python repro/r17_setstate_bypass.py
    === what the documented door refuses ===
      set mass = -1.0  -> {'ok': False, 'error': 'ValueError: spring.params.mass=-1.0 below bound 0.1'}
      set mass = 1e308 -> {'ok': False, 'error': "ValueError: variable 'spring.params.mass': value does not fit its type float32"}
    === the same values through set_state ===
      set_state -> {'ok': True}
      sidecar params['nodes']['spring']['mass'] = -1.0
      sidecar state['spring']['position']       = inf  dtype float32
      sidecar state['spring']['velocity']       = nan
      read back through get: {'ok': True, 'values': [-1.0, inf]}
      one step -> {'ok': True, 't': 0.01}
      state after the step: {'position': nan, 'velocity': nan}
      the model description advertises mass min/max = 0.10000000894069672 9.999999046325684

**Why it happens:** `src/maddening/fmi/tcp_bridge.py:634` (state field:
`jnp.asarray(arr, dtype=live.dtype)` after only a shape check) and :651 (the same
for `p/<section>/<owner>/<key>`).  `_in_dtype` (`tcp_bridge.py:717-732`) and
`FmuSidecar.set_params`'s `spec.check` (`sidecar.py:196-198`) are on the `set`
path only.  Only `_time` gets a finiteness check (:667).

**What would make this a non-issue:** if the blob could only come from this
bridge's own `get_state` — but `fmi3SetFMUState` takes a blob the importer may
have produced with `fmi3DeserializeFMUState` from bytes of its own, and the whole
framing of the module is that the importer is untrusted.  I checked whether the
archive-directory guard caught it: it does not — the member names and sizes are
all legitimate.

**Suggested fix:** run `_in_dtype`-equivalent finiteness/fit checks over every
restored array and `check_bounds(new_params, param_specs)` over the restored
parameter tree, before the commit at `tcp_bridge.py:670`.  Risk: a state
legitimately captured while a field held `inf` (a diverged run snapshotted for
debugging) would no longer restore; that should be a `ValueError` naming the
field, not a silent clamp.

---

### HIGH — `load_graph_from_usd` imports any module a (hostile) stage names

**What breaks:** `GraphManager.from_dict` requires an explicit `node_registry`,
so a config can only name node classes the caller allowed.  The USD reader falls
back to `importlib.import_module` on whatever string sits in
`maddening:nodeType`.  Opening an untrusted `.usda` therefore executes that
module's import-time code before any validation.  Since `sys.path[0]` is the
running script's directory, a hostile `.usda` shipped next to a `.py` is
straightforward code execution.

**Evidence:**

    $ python repro/r15_usd_import.py   # repro/evilmod/hostile_payload.py writes a marker on import
    marker before: False
    load raised: TypeError Anything() takes no arguments
    marker after : True -> imported
    'hostile_payload' in sys.modules: True

**Why it happens:** `src/maddening/usd/serialization.py:78-99`
(`_resolve_node_class`: `mod = importlib.import_module(module_path);
cls = getattr(mod, class_name)`), reached from `load_graph_from_usd` at :400 with
the stage's own string.

**What would make this a non-issue:** (a) if the attacker had to supply an
importable module — true, but `sys.path[0]`, `PYTHONPATH` and any stdlib module
with import side effects all qualify, and the failure is silent (the `TypeError`
arrives *after* the payload ran); (b) if it were new in 0.4.0 — it is not,
`git show origin/main:src/maddening/usd/serialization.py` has the same line, so
this is a pre-existing hole the release inherits; (c) if the docs warned —
`grep -rni untrusted docs/` finds the claim only for FMI and for mapping assets,
never for USD.  It is not in `docs/validation/known_anomalies.yaml`.

**Suggested fix:** make `load_graph_from_usd` take a registry the way `from_dict`
does (defaulting to `_NODE_CLASS_REGISTRY` plus the built-ins) and keep the
`importlib` fallback behind an explicit `allow_import=True`.  Risk: a stage
written with a third-party node class stops loading without an extra argument —
a source-compatible break for exactly the callers who should be opting in.

---

### HIGH — `fim`'s documented `crb = +inf` fail-safe inverts to `crb = 0.0` on a NaN Fisher matrix

**What breaks:** `FIMReport`'s docstring: *"`crb` … is `+inf` for every parameter
with support in the null space … `+inf` rather than `NaN` because it fails safe:
`crb < threshold` is then False for an unidentifiable parameter instead of
quietly propagating a `NaN`."*  When every eigenvalue is NaN (`noise_std=0`,
`noise_std=nan`, a `noise_std` that underflows the residual's dtype, or a
residual containing a NaN), the report comes back `rank=0`, `cond=nan` — and
`crb=[0.0, 0.0]`.  A caller's `crb < tol` test says *yes, identified* for a
matrix that contains no information at all.

**Evidence:**

    $ python repro/r7_fim_crb.py
    healthy:          rank=2 cond=1.0 crb=[1. 1.]
    noise_std = 0.0:  rank=0 cond=nan eigvals=[nan nan] crb=[0. 0.]
       crb < 1e-6  -> True  (a caller's 'is it identified?' test passes)
    noise_std = -1.0: rank=2 crb=[1. 1.]   (a negative sigma is accepted silently)
    NaN in the residual: rank=0 cond=nan crb=[0. 0.]

`repro/r6_fim.py` shows the same for `noise_std` in `{0.0, nan, 1e-30}`.

**Why it happens:** `src/maddening/sysid.py:588-591`.  `resolved = ev >
max(ev[-1], 0) * rank_rtol` is all-False for NaN, so the resolved-subspace sum is
an empty sum — `0.0`, not `inf`.  The rescue is `crb = np.where(support > n * eps,
np.inf, crb)`, but `support` is itself NaN and `NaN > x` is False, so the `+inf`
branch never fires.  Nothing upstream rejects `noise_std <= 0`
(`_inverse_noise_std`, :502-517, only divides) or a non-finite Jacobian
(`fit`/`fit_lm` raise `FloatingPointError` on one; `fim` does not).

**What would make this a non-issue:** if degenerate input were rejected first —
`rank_rtol` *is* validated (`sysid.py:571-576`) but `noise_std` is not, and a
negative sigma is accepted and silently gives the same answer as its absolute
value.  `rank=0` is a signal, but `cond=nan` and `crb=0` both read as "fine"
under the obvious tests, and `FIMReport` carries no validity flag.

**Suggested fix:** validate `noise_std` (finite, strictly positive, and still
strictly positive after the cast to the residual dtype) and raise on a non-finite
`F` in `fim` the way `fit` raises on a non-finite gradient; independently, make
`_rank_and_crb` set `crb = inf` wherever `support` is not finite, so the
fail-safe holds whatever produced the NaN.  Risk: none I can see — a NaN Fisher
matrix has no legitimate use.

---

### MEDIUM — the sysid `mask` is checked for leaf *count* only, so a mismatched mask silently fits a different parameter

**What breaks:** `_masked_indices`, `_resolve_mask` and `_physical_params` all zip
`jax.tree.leaves(mask)` against the params leaves and raise *"mask must have the
same tree structure as params"* only when the lengths differ.  A mask whose dict
keys are different — copied from another graph, or keyed by the user's own symbol
names — is accepted and its flags land on whatever parameter occupies that
position in flatten order.

**Evidence:**

    $ python repro/r14_mask_structure.py
    params leaves, flatten order: ['damping', 'initial_position', 'initial_velocity',
                                   'mass', 'rest_length', 'stiffness']
    caller's mask (by their own symbol names, zeta == damping):
        {'zeta': True, 'alpha': False, 'beta': False, 'gamma': False,
         'delta': False, 'epsilon': False}
    sorted label order: ['alpha','beta','delta','epsilon','gamma','zeta']
    -> the single True sits at index 5 which in params is stiffness

    correct mask       accepted, moved {'damping': (2.0, 0.9995478987693787)}
    mislabelled mask   accepted, moved {'stiffness': (30.0, 76.29719543457031)}

**Why it happens:** `src/maddening/sysid.py:351`, :409, :474 — three copies of
`if len(flags) != len(entries): raise ValueError("mask must have the same tree
structure as params")`.  `jax.tree_util.tree_structure` is never compared.

**What would make this a non-issue:** if the only supported way to build a mask
were `gm.trainable_mask()` plus `jax.tree.map` — but `fit`'s own docstring invites
a hand-built narrowing mask, and `_resolve_mask`'s `trainable=False` guard only
catches the subset of misalignments that happen to land on a frozen leaf.

**Suggested fix:** compare `jax.tree_util.tree_structure(mask)` against
`tree_structure(params)` in one shared helper and name the first differing path.
Risk: a caller passing a structurally loose mask that happened to work now gets
an error — the point, but a behaviour break for them.

---

### MEDIUM — the params pytree silently narrows float64 to float32, and only for some spellings of the same value

**What breaks:** with `JAX_ENABLE_X64=1`, a node parameter given as a Python
float, a `numpy.float64` scalar, or a list of floats is silently cast to float32;
the same value given as a 0-d or 1-d `numpy.float64` array, or a `jnp` float64
array, keeps float64.  Two spellings of one number give two dtypes and two
answers, with no warning.  This contradicts the position the release took
elsewhere — `docs/algorithm_guide/coupling/interface_mapping.md` refuses to narrow
a point set because *"Downcasting quietly would hide a precision loss you did not
ask for — a well-known source of numerical bugs that are very hard to trace back
to their cause"*.

**Evidence:**

    $ JAX_ENABLE_X64=1 python repro/r13_dtype_narrowing.py
    jax_enable_x64: True
    isinstance(np.float64(x), float) -> True
      python float          -> dtype=float32 value=0.12345679104328156 rel_err=1.65e-08 warnings=[]
      np.float64 scalar     -> dtype=float32 value=0.12345679104328156 rel_err=1.65e-08 warnings=[]
      np.float64 0-d array  -> dtype=float64 value=0.12345678901234568 rel_err=0.00e+00 warnings=[]
      np.float64 1-d array  -> dtype=float64 value=0.12345678901234568 rel_err=0.00e+00 warnings=[]
      list of python floats -> dtype=float32 value=0.12345679104328156 rel_err=1.65e-08 warnings=[]
      jnp float64 array     -> dtype=float64 value=0.12345678901234568 rel_err=0.00e+00 warnings=[]

`repro/r12_dtype_x64.py` shows the mixed tree surviving config `to_dict` /
`from_dict` bit-for-bit — the round trip is faithful, it is the *entry* that
narrows.

**Why it happens:** `src/maddening/core/node.py:341-342`
(`if isinstance(value, float): out[key] = jnp.asarray(value, dtype=jnp.float32)`)
and :353-354 (`if isinstance(value, (list, tuple)): arr = arr.astype(jnp.float32)`).
`numpy.float64` is a subclass of `float`, so it takes the first branch; an ndarray
does not.  This is `core/node.py` rather than my assigned files, but it is the
source of every pytree `sysid`, the FMI parameter surface and both serialisers
round-trip, so it is the answer to the dtype-honesty question on this surface.

**What would make this a non-issue:** if x64 were unsupported — but the release
notes say two Hypothesis suites now run under `jax.experimental.enable_x64()`
instead of skipping, and `core/params.py`'s transforms are dtype-faithful under
x64 (`repro/r12`: float64 in, float64 out, `log` round trip exact).  The library
is x64-aware everywhere except the entry point.

**Suggested fix:** use `jnp.asarray(value)` and let JAX's x64 policy decide, or
keep float32 as the default and warn once when x64 is on and a float64 input is
being narrowed.  Risk: graphs silently relying on float32 params change dtype
under x64 and retrace; float32/float64 mixes inside one params tree start
promoting through `ravel_pytree` in the fitters.

---

### MEDIUM — `fim(scale="relative")`, the default, calls a parameter whose value is 0.0 unidentifiable

**What breaks:** relative scaling multiplies each Jacobian column by the
parameter's current value, so a parameter sitting at exactly `0.0` gets a zero
column and comes back with `crb = +inf` and a reduced `rank`, however well the
data determine it.  `initial_velocity=0.0` is the default on `SpringDamperNode`,
so this fires on the most obvious first call a user makes.  `FIMReport` explains
`+inf` as a statement about the data (*"the data cannot separate it from the
combinations that null space mixes it with"*), which is not what happened.

**Evidence:**

    $ python repro/r8_fim_relative_zero.py
    ### J = I (exactly identifiable)
      scale='relative'  rank=5/6  cond=inf
          crb={'damping': 0.25, 'initial_position': 4.0, 'initial_velocity': inf,
               'mass': 0.444, 'rest_length': 1.0, 'stiffness': 0.0011}
      scale=None        rank=6/6  cond=1
          crb={'damping': 1.0, ..., 'initial_velocity': 1.0, ...}

**Why it happens:** `src/maddening/sysid.py:673-674`, `J = J * theta0[None, :]`.

**What would make this a non-issue:** it is arithmetically what "relative
sensitivity" means, and `scale=None` gives the right answer — but nothing in the
report or the docstring distinguishes "the data cannot see this" from "you asked
about a relative change to zero", and it is the default.

**Suggested fix:** record on the report (or warn) when `scale="relative"` and any
selected `theta0` entry is exactly zero, naming the parameters.  Risk: noise on
graphs that legitimately have zero-valued constants — a report field is probably
better than a warning.

---

### MEDIUM — USD silently turns an unserialisable node param into its `repr`; the config path raises

**What breaks:** `save_graph_to_usd` writes node params with
`json.dumps(..., default=str)`.  A param the JSON encoder cannot handle — the
`static_data_provider` object `core/node.py`'s own docstring tells users to
*"store … in `self.params` so it survives a checkpoint/restore round-trip"* — is
written as its `repr` and reloads as a `str`.  `GraphManager.to_dict` +
`json.dumps` raises a `TypeError` for the same graph.

**Evidence:**

    $ python repro/r20_misc.py
    === A. USD silently stringifies an unserialisable node param (default=str) ===
      stored: {"provider": "<Provider /data/mesh.vtu>", "k": 3}
      reloaded provider: '<Provider /data/mesh.vtu>' str
      config to_dict for the same graph:
        json.dumps(to_dict()) raises TypeError: Object of type Provider is not JSON serializable

**Why it happens:** `src/maddening/usd/serialization.py:205-207`
(`json.dumps(_params_to_serializable(node_params), default=str)`).  Pre-existing
on `main`; reported because it is the one place the two serialisers disagree
about what is representable.

**What would make this a non-issue:** if such params did not exist — but the
provider pattern is documented, and a reload that produces a `str` where the node
expects an object fails much later and elsewhere.

**Suggested fix:** drop `default=str` and let the `TypeError` name the key, as the
config path already does.  Risk: stages that currently save (with a mangled
param) start failing at save time — the earlier and more actionable failure.

---

### LOW — zero-size array params lose their shape through USD

`np.zeros((0, 3))` serialises through `.tolist()` as `[]` and reloads as shape
`(0,)`.  `repro/r2_usd_params.py`: `empty_2d in: (0, 3)  out: (0,)`.
`src/maddening/usd/serialization.py:600-607` (`_params_to_serializable`).  Narrow,
but it is silent shape loss on a round trip the property suite advertises as
faithful.

### LOW — non-finite numbers produce non-standard JSON in three places

`json.dumps` writes bare `NaN` / `Infinity` / `-Infinity`, which Python reads back
but no strict JSON reader will:

* `GraphManager.to_dict()` for `ParamSpec(bounds=(-inf, inf))` —
  `repro/r16_spec_roundtrip_gaps.py`: `strict JSON: FAILS -> non-standard JSON
  token '-Infinity'`;
* the USD `maddening:paramsJson` attribute for a non-finite param —
  `repro/r2_usd_params.py`: `stored JSON: {"gain": NaN, "cap": Infinity, ...}`;
* the FMI wire, for a `get` reply carrying a non-finite value —
  `repro/r20_misc.py`: `{"ok":true,"values":[1.0,Infinity,NaN]}`.

The Python-to-Python round trips are all exact; the exposure is any other reader
of a saved config, a `.usda`, or the documented JSON protocol.

### LOW — a non-finite parameter is exported as `start="inf"` / `start="nan"` in modelDescription.xml

`repro/r11_fmi_desc.py`:

    s.params.stiffness  start='inf'   min=1.1754943508222875e-38 max=None
    s.params.damping    start='nan'   min=0.0                    max=None
    XML: <Float32 name="s.params.stiffness" ... start="inf" ...

`src/maddening/fmi/model_description.py:705`
(`" ".join(repr(float(x)) for x in flat)`).  `inf`/`nan` are not `xs:float`
literals (`INF`, `NaN` are), so the FMU is malformed — and the bridge then refuses
to `set` the value it advertises as the start.  `build_model_description` never
calls `check_params`.

### LOW — a fit that takes no step still reports its masked leaves as changed

`FitResult`: *"comparing a fit's input and output leaf by leaf says exactly which
constants the calibration touched."*  `repro/r10_noop_fit_drift.py`:

    1. fit(n_iter=0) -- no gradient is ever evaluated:
       losses: []  n_iter: 0  converged: False
        stiffness  30.0 -> 30.000001907348633  bitwise_same=False
    2. fit(tol=1e30) -- stops on the first loss, before any update:  same

`_physical_params` (`sysid.py:445-487`) copies leaves *outside* the mask
bit-for-bit but lets masked leaves through `constrain(unconstrain(p))`, which for
a `log` leaf is `exp(log(p))` — one ulp off for some values (30.0 drifts, 1.5 does
not).  It does not accumulate (fixed point after one pass).

### LOW — `fit` / `fit_lm` / `fim` accept nonsense hyper-parameters

`repro/r9_fit_degenerate.py` and `repro/r6_fim.py`: `lr=0.0` and `lr=-0.1`
(gradient *ascent*) run to completion; `eps=-1e-8` runs; `tol=nan` runs;
`notify_every=-1` runs; `fit_lm` accepts `lam0=0.0`, `lam0=-1.0`, `lam_up=0.5`
(damping that shrinks on rejection) and `step_tol=-1.0`; `fim` accepts
`noise_std=-1.0`.  Only `rank_rtol` is validated.  `eps=0.0` does fail, but as
`FloatingPointError: non-finite loss or gradient`, blaming the loss for a 0/0 in
the Adam step.

### LOW — `do_get` can build a request frame over the 64 MiB limit; `do_set` cannot

`repro/c_probe.c` (ASan+UBSan):

    do_set JSON  nvr=200000 nvalues=200000 -> status 3, frame 7150030 bytes (cap 16777216)
    do_get  nvr=7000000 -> status 3, request frame 77000019 bytes, FRAME_MAX=67108864, over=YES

`do_set` (`maddening_fmu.c:358-361`, :381-385, :398-402) refuses such a frame;
`do_get` (:406-419) has no equivalent check, so the bridge's `recv_raw` raises and
drops the connection, killing the instance instead of returning an error.  The
wrapper's own docstring claims the limit only for `set`.  No memory error — the
buffer sizing is correct.

### LOW — an integer leaf with a `log`/`logit` transform changes dtype through `unconstrain`/`constrain`

`repro/r1_paramspec_roundtrip.py`:

    dtype=int32  transform=log    bounds=(0.0, None)  -> u.dtype=float32 p'.dtype=float32 p'=3.0
    dtype=int32  transform=logit  bounds=(0.0, 100.0) -> u.dtype=float32 p'.dtype=float32 p'=2.999999761581421

`core/params.py`'s module docstring promises `constrain(unconstrain(p)) == p` on
the whole tree, and `to_constrained`'s identity branch goes out of its way to
preserve the dtype (*"a leaf's dtype is part of the pytree contract, so keep it"*,
:175-181) — the transform branches do not.  Only reachable for an integer leaf
someone made trainable with a transform, which nothing in-tree produces.

### LOW — `FmuSidecar.handle` and `deserialize_fmu_state` still unpickle

`sidecar.py:246` (`pickle.loads(request)`) and `fmu_state.py:132`
(`pickle.loads(fmu_state.payload)`).  Not reachable from the TCP bridge — `grep`
over `src/` shows nothing calls `handle` or `set_fmu_state`, and `FmuTcpBridge`
uses its own npz path with `allow_pickle=False`.  Recorded because these are
public, `@stability(EVOLVING)`-tagged entry points whose docstrings describe them
as a wire protocol ("Request (host -> sidecar)"), so the next transport that
reaches for them re-opens the historical RCE.

## Unverified suspicions

* `ravel_pytree` in the fitters promotes a mixed-dtype params tree to one common
  dtype and casts back through `unravel`, so a bfloat16 or float16 leaf alongside
  float32 would round-trip through float32 every iteration.  I could not build a
  graph that produces such a tree (`params_pytree` emits float32 or the array's
  own dtype), so there is no reproducer.
* `bridge_xfer` decides a JSON reply succeeded with
  `strstr(in->resp, "\"ok\":true")` over the whole body.  I tried to smuggle that
  substring through an error message and could not: `json.dumps` escapes the
  quotes, so the literal bytes never appear.  Recorded only because the test is
  position-free and one unescaped path would defeat it.
* `_decode_state` splits an input key with `k.split("/", 2)`, so a node name
  containing `/` would route to the wrong `(node, field)` pair.  `add_node`
  forbids `/` in a node name, so I could not reach it.

## What I checked and found sound

* **The C wrapper.**  80,000 fresh ASan+UBSan fuzz iterations over four seeds the
  suite does not use (424242, 991, 20260919, 7777) — clean, every parser path
  reached.  Plus a targeted probe (`repro/c_probe.c`) of the request side the
  fuzzer under-covers: the widest `%.17g` any double can produce is 24 chars
  against the 32-byte-per-value budget; `do_set` with 200k extreme doubles
  (`-DBL_MAX`, `5e-324`, `-1.2345678901234567e-308`) is in-bounds;
  `fmi3SetFMUState`'s 1 MiB JSON path is in-bounds; `read_endpoint` handles
  `[::1]:p`, a missing host, a missing/zero/overflowing port and a 500-char host
  without overflowing its 256-byte buffer.  No ASan or UBSan diagnostic anywhere.
* **Bridge frame validation.**  `repro/r4_bridge_attack.py` drives 30 hostile
  requests over a real socket: a non-list `vr`, a ragged `values`, a string value,
  `vr = 2**70`, a float `vr`, `dt = nan`, `t = inf`, a length prefix over the limit
  with and without the binary flag, a binary frame before negotiation, `n`
  negative / boolean / 2**40, a wrong `dtype`, `header_len` past the payload, a
  payload shorter than the header field, a non-object header.  Every one is an
  error reply or a deliberate drop; nothing crashes and nothing corrupts the
  instance.
* **FMU-state archive guard.**  A truncated archive, an empty blob, a non-zip, a
  member named `../../../../tmp/pwned.npy`, and a 50 MiB member in a 51 KiB
  deflate bomb are all refused from the directory alone, before decompression.
* **`ParamSpec` serialisation.**  `repro/r16_spec_roundtrip_gaps.py` round-trips
  the cases the property strategy never draws — `logit` with a finite upper bound,
  `bounds=(-inf, inf)`, `bounds=(5e-324, None)`, `bounds=(-1e308, 1e308)`, a
  description containing a NUL byte and a tab — through `to_dict`/`from_dict`,
  through an in-memory stage, and through a `.usda` file on disk.  All four
  identical in all three.
* **The trainable mask, end to end.**  `repro/r3_sysid_mask.py`: with a narrowing
  mask and with the graph's own declarations, every unselected leaf comes back
  bit-identical (`tobytes()` comparison), the selected one moves, and dtypes are
  preserved.  A mask that widens the trainable set is refused with the leaf named;
  a mask that selects nothing is refused.
* **`fim` naming and index alignment.**  `repro/r18_fim_names.py`: with a
  3-element, a zero-size, a scalar and a 2-element leaf, `param_names`, the masked
  sub-matrix and `crb` all line up with the hand-computed answer
  (`1, 1/4, 1/9, 1/16, 1/25, 1/36`).  Masking only the zero-size leaf correctly
  raises *"mask selects no parameters"*.  `least_identifiable` names the right
  parameter on a deliberately rank-deficient Jacobian.
* **Transforms under x64.**  `repro/r12_dtype_x64.py`: `log` round-trips a float64
  value exactly, `logit` to 1.1e-16, both keeping float64.
* **Config-to-pytree fidelity.**  A mixed float32/float64 params tree survives
  `to_dict` -> `json.dumps` -> `json.loads` -> `from_dict` bit-for-bit in every
  leaf.
* **Existing tests.**  `tests/fmi/ tests/usd/ tests/property/test_round_trips.py
  tests/property/test_sysid_contract.py` — **317 passed, 0 failed, 0 skipped** in
  996 s (`tests.log`).  Command:
  `PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  python -m pytest <paths> -q -p no:cacheprovider -rs`.
