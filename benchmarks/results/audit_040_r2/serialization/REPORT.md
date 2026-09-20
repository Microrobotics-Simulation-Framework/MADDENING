# Audit: serialization / graph-engine mutation semantics / FMI bridge   (0219b82c9a5a22b209c1dd370cc7d406e4ec1079, round 2)

Worktree: `/home/nick/MSF/msf/MADDENING-wt/audit-r2-serialization` (detached at
`origin/release/0.4.0`).  Reproducers in `repro/`, raw mutation logs in `logs/`.
Nothing outside this directory and that worktree was touched; the worktree is
clean (`git status --porcelain` empty) after every mutation.

## Summary

Five findings.  The most important is **HIGH**: `MADD-ANO-010`'s reachability
claim — *"no request or reply `FmuTcpBridge` itself builds can reach it, because
the protocol has no free-text field"* — is **false**.  The `hello` reply carries
`model`, the caller's `model_name`, which is unvalidated free text; an FMU named
exactly `NaN`, `Infinity` or `-Infinity` makes `_send_reply` raise `ValueError`
out of `_serve_conn_inner`, killing the connection worker on the *first* frame.
The same sentence appears in `known_anomalies.yaml`, `docs/release_notes/v0.4.0.md`
and `src/maddening/serialization/README.md`.

The rest: `json_codec.dumps(gm.to_dict())` — composing two public 0.4.0 APIs —
raises, because `encode_non_finite` is not idempotent and `to_dict` returns
already-encoded data; `save_graph_to_usd` accepts a node name the anomaly record
says it refuses; and `compile()`'s `accelerated_fields` gate cannot fire under
`iqn-imvj`, the one acceleration in which the field is most used.

**Sound:** the codec round trip is total on everything it accepts (nested
containers, keys, `-0.0`, subnormals, max double, 17-digit floats, non-ASCII
keys, numpy float64, arrays via `tolist`).  `compile()` atomicity held against
**ten** distinct failure modes.  **13 of 14 mutations** against the non-finite
gates were caught, including all four against the C wrapper.

---

## Findings

### HIGH — the FMI `hello` reply carries free text, so `MADD-ANO-010` *is* reachable from a reply the bridge builds, and it kills the connection worker

**What breaks:** `build_model_description(gm, model_name="Infinity")` →
`FmuTcpBridge` → a client sends `{"op":"hello","protocol":2,"binary":true}` →
the worker thread dies with an unhandled `ValueError` and the client gets EOF
instead of a reply.  Every connection to that FMU dies the same way; the FMU is
unusable.  `model_name` is a free-text argument with no validation anywhere in
`model_description.py`.

**Evidence:** `repro/fmi_hello_model_name_kills_worker.py`

```
model_name = 'Plant'
  handle() -> {'ok': True, 'token': 'd5bd...', 'model': 'Plant'}
  reply after 0.000s: (False, b'{"ok":true,"token":"d5bd...","model":"Plant","master_dt":0.01,"protocol":2,"binary":true}')
======================================================================
model_name = 'Infinity'
  handle() -> {'ok': True, 'token': 'd2f7...', 'model': 'Infinity'}
  reply after 0.001s: None
  worker thread died: ValueError $.model: the string 'Infinity' cannot be written to JSON, because it is how a non-finite float is encoded and would read back as that float.  Store it as something else (a different spelling, or a tagged value of your own).
======================================================================
model_name = 'NaN'          -> same
model_name = '-Infinity'    -> same
```

**Why it happens:**
`tcp_bridge.py:740-742` (`_dispatch`, the `hello` branch) builds
`{"ok": True, "token": ..., "model": self._md.model_name, ...}`.
`tcp_bridge.py:647,667` (`_send_reply`) passes that through
`_json_dumps` = `json_codec.dumps`, whose `encode_non_finite` walk refuses a
*string* leaf equal to a token (`json_codec.py:136`).
`tcp_bridge.py:641-642` (`_serve_conn_inner`) wraps the send in
`except OSError` only, so the `ValueError` leaves the loop, leaves
`_serve_conn` (`tcp_bridge.py:549`, which has a bare `finally` and no `except`)
and ends the thread.

**Correction to the reported consequence.** The orchestrator's brief says this
would "leave the importer waiting out a 300 s `_IDLE_TIMEOUT`".  Measured, it
does not: `_serve_conn_inner` holds the socket in `with conn:`, so the socket is
*closed* as the exception unwinds and the client's `recv` returns EOF in ~1 ms
(`reply after 0.001s: None` above).  The instance slot is also released
correctly by the `finally` at `tcp_bridge.py:643-645`.  What is actually lost is
(a) the error reply the protocol promises for every failure, (b) the connection,
and (c) a traceback on the bridge's stderr from `threading.excepthook`.  The C
wrapper sends the hello from inside `fmi3InstantiateCoSimulation`
(`maddening_fmu.c:579`), so the EOF becomes a failed `bridge_call` and the
function returns `NULL`: the importer gets "cannot instantiate", with nothing
to say that the model's *name* is what broke it.

**What would make this a non-issue:** (i) if `model_name` were validated or
derived — it is not; `build_model_description` passes it straight through
(`model_description.py:784`) and the only other constrained field,
`instantiation_token`, is a UUID-shaped hash that cannot spell a token.  (ii) if
`handle()` failed the same way, making it a plainly-user-visible error — it does
not; `handle()` returns the reply fine, so only the socket path breaks.  (iii) if
some other reply field were the real free text — I checked every reply
`_dispatch` builds: all error strings are `f"{prefix}: {exc}"` or use `!r`, so
none can equal a bare token; `state` is base64 (could in principle spell
`Infinity` — 8 base64 chars from 6 bytes — but an npz blob is never 6 bytes);
`values` are floats the codec encodes correctly.  `model` is the only one.

**Suggested fix:** two parts, and they are independent.  Make `_serve_conn_inner`
catch `Exception` (not just `OSError`) around `_send_reply` and answer with a
plain-ASCII error reply, so *no* unencodable reply can ever cost the connection —
this is the part that matters, because the free-text surface will grow.  Then
either reject a `model_name` in `NON_FINITE_TOKENS` in
`build_model_description` (fail at build time, where the path is actionable) or
have `_dispatch` spell the hello's `model` defensively.  Risk: catching
`Exception` around a send can mask a genuine bug in reply construction, so the
handler should log it rather than swallow it.  **Also correct the three
documents** — the reachability sentence is load-bearing in `MADD-ANO-010`'s
`safety_relevance_rationale` ("a fail-stop at write time that names the path"),
and here it is not a fail-stop at the write site but a dead worker thread.

---

### MEDIUM — `encode_non_finite` is not idempotent, and `to_dict()` returns already-encoded data, so `json_codec.dumps(gm.to_dict())` raises

**What breaks:** any graph carrying a non-finite float — the commonest source
being `ParamSpec(bounds=(-inf, inf))`, the documented way to say "unbounded" —
cannot be written with the serialization package's own dumper:

```python
gm.set_param_spec("a", "stiffness", ParamSpec(bounds=(-math.inf, math.inf)))
json_codec.dumps(gm.to_dict())
# ValueError: $.param_specs.a.stiffness.bounds[0]: the string '-Infinity'
#   cannot be written to JSON, because it is how a non-finite float is encoded
#   and would read back as that float.
```

`to_dict` has already turned `-inf` into `"-Infinity"`; `dumps` runs
`encode_non_finite` a second time, now sees a *data* string spelling a token,
and applies the `MADD-ANO-010` refusal to the encoder's own output.

**Evidence:** `repro/to_dict_not_idempotent.py`

```
[idem] json_codec.dumps(gm.to_dict()) -> ValueError: $.param_specs.a.stiffness.bounds[0]: the string '-Infinity' cannot be written to JSON, ...
[doc] strict-parseable: True
[rt ] to_dict -> json -> from_dict -> to_dict is a fixed point: True
```

**Why it happens:** `graph_manager.py:5724` applies `encode_non_finite` to the
assembled tree, and `json_codec.py:226` (`dumps`) applies it again.
`decode_non_finite` *is* idempotent (a float passes through), so the read side
composes fine; only the write side does not.

**What would make this a non-issue:** if nothing in the tree ever composed the
two.  Checked: every shipped caller and every test uses plain `json.dumps` on
`to_dict()` output (`tests/core/test_non_finite_json_tokens.py:161,181,253`,
`tests/core/test_mapping_spec_serialisation.py:154,208,298`,
`src/maddening/core/coupling/mapping_spec.py:879`), and Starlette's
`JSONResponse` (`api/server.py:714`) also uses plain `json.dumps`, so nothing
live is broken today.  It is a trap, not an outage: `dumps`'s own docstring
advertises it as taking "a JSON-shaped tree", and `to_dict`'s advertises its
result as JSON-valid, so the composition is the natural thing to write.

**Suggested fix:** the cheap one is to document that `to_dict()` returns an
*already-encoded* document and must be written with `json.dumps`, not
`json_codec.dumps`.  The better one is to stop encoding inside `to_dict` and
move the encoding to the write boundary — but that changes what `to_dict`
returns (floats, not tokens) and `from_dict` would keep working either way,
so it is a behaviour change to weigh rather than a drop-in.  Either way, add a
test for the composition: nothing pins it today.

---

### MEDIUM — the three surfaces do not agree: `save_graph_to_usd` accepts a node name that `to_dict` refuses, contradicting `MADD-ANO-010`

**What breaks:** a graph with a node named `NaN` (or `Infinity`, `-Infinity`)
saves to a `.usda` stage and reloads from it correctly, but cannot be written as
a config.  `MADD-ANO-010` says the opposite: *"The refusal is on every surface
that shares the codec: `GraphManager.to_dict`, the USD JSON attributes written
by `save_graph_to_usd`, and the FMI JSON wire helpers"*, and lists
`maddening.usd.serialization.save_graph_to_usd` under `affected_components`.
The same claim is in the release notes and in `serialization/README.md` ("The
rule applies to a node *name* as well as a parameter value").

**Evidence:** `repro/three_surface_agreement.py`

```
### node named 'NaN'
  to_dict          : ValueError: $.nodes[0].name: the string 'NaN' cannot be written to JSON, ...
  save_graph_to_usd: OK  (reloaded nodes ['NaN']; to_dict on the RELOADED graph: ValueError)
      stage text contains '"NaN"'
  encode_binary    : ValueError: $.tag: the string 'NaN' cannot be written to JSON, ...

### string param 'Infinity'
  to_dict          : ValueError: $.nodes[0].params.label: ...
  save_graph_to_usd: ValueError: $.label: ...
```

**Why it happens:** the node name is written to a *typed USD String* attribute,
not to JSON — `usd/serialization.py:328-330`
(`prim.CreateAttribute("maddening:nodeName", Sdf.ValueTypeNames.String).Set(node_name)`)
— and read back the same way, so the codec never sees it.  Only the JSON-valued
attributes (`paramsJson`, `paramSpecOverridesJson`, `mappingSpecJson`,
`paramArrayShapesJson`, the coupling group's `accelerated_fields`) go through
`_json_dumps`.  `GraphManager.to_dict`, by contrast, puts the name in the JSON
tree (`nodes[i]["name"]`), so it is refused there.

**What would make this a non-issue:** if the USD path refused the name
elsewhere.  It does not — the coupling group's node list is a `Vt.StringArray`
and the edge endpoints are String attributes, all raw.  A mapped edge whose
`MappingSpec` carries a node-field *point reference* to that node **would** be
refused (the node name is a JSON value there), so the USD refusal is
configuration-dependent rather than absent, which is worse to document than
either extreme.

**Suggested fix:** decide which behaviour is intended and make the three agree.
Refusing a token-spelled node name uniformly (a check in `add_node`, beside the
existing `/ # ->` check at `graph_manager.py:2920`) puts the failure at the
point of entry, which is exactly what `MADD-ANO-010`'s own workaround text
recommends ("validate names ... where they are accepted, not where they are
saved").  Risk: it is a new refusal on a public constructor path, so it needs a
CHANGELOG line and a migration note.  Whatever is chosen, the anomaly record,
the release notes and the README must be corrected — as written, all three
describe a refusal that a USD-only user will never see.

---

### LOW — `compile()`'s `accelerated_fields` validation cannot fire under `iqn-imvj`, the acceleration that actually consumes the field

**What breaks:** a typo in `accelerated_fields` reports a bare, unattributed
`KeyError` under `acceleration="iqn-imvj"` and the intended, actionable
`ValueError` under every other mode — including `acceleration="none"`, where the
field is *ignored altogether* and `CouplingGroup` already warns that it is inert.
The gate fires where the mistake is harmless and is shadowed where it matters.

**Evidence:** `repro/accelerated_fields_message.py`

```
none       compile -> ValueError: accelerated_fields['a'] names ['not_a_field']: not a state field of 'a' (state fields: ['position', 'velocity'])
iqn-ils    compile -> ValueError: accelerated_fields['a'] names ['not_a_field']: not a state field of 'a' (state fields: ['position', 'velocity'])
iqn-imvj   compile -> KeyError: 'not_a_field'
aitken     compile -> ValueError: accelerated_fields['a'] names ['not_a_field']: not a state field of 'a' (state fields: ['position', 'velocity'])
```

**Why it happens:** ordering inside `compile()`.  The `_meta` seeding block for
`iqn-imvj` reads the user's field list and calls `flatten_coupled_state(...,
fields=af)` at `graph_manager.py:3673-3679`, which raises `KeyError` on an
unknown field.  The dedicated validation that produces the good message is at
`graph_manager.py:3738-3757`, *after* it.

**What would make this a non-issue:** if `iqn-imvj` were the only mode where
`accelerated_fields` is inert — it is the opposite; `CouplingGroup` warns that
the field is ignored unless acceleration is `iqn-ils` or `iqn-imvj`.  Atomicity
is not affected: the `KeyError` still leaves the graph untouched (verified,
`repro/compile_atomicity_imvj.py`).

**Suggested fix:** move the `accelerated_fields` validation block above the
`_meta` seeding.  Risk: essentially none — the block reads only `self._nodes`,
`self._state` and the group, all of which are already available at that point.

---

### LOW — gate gap: the tuple branch of `encode_non_finite` is unexercised

**What breaks:** nothing today; recording it because round two asked for gate
coverage as a result in its own right.  Mutation **M1** — make
`encode_non_finite` stop recursing into tuples (`isinstance(obj, (list, tuple))`
→ `isinstance(obj, list)`) — leaves **all 57** gate tests green
(`logs/mutation_round1.log`).  The docstring promises "Walks dicts, lists and
tuples".

**Why it does not bite:** no production `to_dict` tree contains a tuple.  Probe
in `repro/to_dict_not_idempotent.py`:

```
[tuple] tuples in a to_dict tree (pre-encode): none
```

and the failure mode is loud anyway — an unencoded non-finite inside a tuple
hits `json.dumps(allow_nan=False)` and raises.  A one-line addition to
`tests/core/test_non_finite_json_tokens.py` closes it.

---

## Unverified suspicions

1. **`strtod` and the Python end disagree on `nan(chars)` and C99 hex floats.**
   `repro/strtod_probe.c` vs the Python decoder: `strtod` accepts `nan(0x1)`,
   `NAN(quiet)` and `0x1p3`; `decode_non_finite` leaves them as strings and
   `values_of`/`_set` then raise `ValueError` from numpy.  *What would falsify
   it as a defect:* the only writer on that wire being the Python bridge, which
   emits exactly three spellings.  **Checked — that is the case**, and
   `do_set` refuses non-finite before building a request, so the shipped stack
   never exercises the divergence.  It only matters for a third-party bridge.
   Not reported as a finding.
2. **The README's "read back as the strings they are" claim is surface-specific.**
   `"nan"`, `"INF"`, `"-inf"`, `"+Infinity"`, `" Infinity"` and `"NaN "` survive
   as strings through `to_dict`/`from_dict` and the USD attributes, but on the
   FMI `values` path `np.asarray(..., dtype=np.float64)` coerces every one of
   them to a float (`repro/`, printed in the strtod comparison).  This is not
   corruption — `values` is declared numeric — but the README sentence reads as
   a global invariant.  *Would falsify:* `values_of` rejecting strings; it does
   not.
3. **`do_set` refuses non-finite even on a binary (protocol 2) connection**
   (`maddening_fmu.c:394`, before the `if (in->binary)` split), although raw
   little-endian float64 carries NaN losslessly and the bridge's own `_set`
   refuses it anyway.  So the two ends agree and nothing is inconsistent — but
   the restriction is enforced twice in a place where the stated reason
   (`%.17g` cannot spell a non-finite in JSON) does not apply.  Not a defect;
   noted in case the binary path is later meant to carry a diverged input.
4. **`compile()` calls `_recover_from_escaped_tracers()` (which rewrites
   `self._state`) and `enable_from_env()` (a process-global XLA cache switch)
   before anything that can fail.** Both survive a failed compile.  Neither is a
   correctness problem — the first restores a *known-good* prior state and the
   second is idempotent — but neither is covered by the "bit-identical" wording
   of the atomicity comment at `graph_manager.py:3563-3567`.  Not reproduced as
   a failure; recorded so the wording is not read as stronger than it is.
5. **`check_anomalies.py`'s zero-scope guard is conditional on `notes`**
   (`scripts/check_anomalies.py:98`): if every `affected_components` entry
   resolves, `notes` is empty and `_components_checked` is never called, so a
   registry with *no* `affected_components` at all would pass without the
   "verified nothing" guard firing.  I did not build this reproducer — it needs
   an edited registry, and the anomaly registry belongs to the gates auditor's
   surface.  Flagging the shape, not claiming the instance.

---

## What I checked and found sound

**Codec round trip (`repro/codec_round_trip.py`).**  Encode→decode and
encode→`json`→decode are the identity on: the three non-finite floats; `-0.0`
(sign preserved); the smallest subnormal `5e-324`; the largest double; a
17-significant-digit float; nested dict/list/tuple mixtures; non-ASCII keys
(`ключ`, `日本`); `numpy.float64` non-finites; `bool` (correctly *not* treated as
a float); strings that merely resemble tokens (`"nan"`, `"INF"`, `"nan(0x1)"`).
The documented non-members behave as documented and fail loudly, never silently:
`numpy.float32` and 0-d `ndarray` raise `TypeError` from `json.dumps`; a
non-finite *dict key* raises `ValueError: Out of range float values are not JSON
compliant` (the `allow_nan=False` backstop).  A finite float dict key still
stringifies, which is plain `json` semantics and unchanged by this release.

**Config surface float fidelity (`repro/config_float_fidelity.py`).**
`to_dict → json.dumps → json_codec.loads → from_dict → to_dict` is a byte-exact
fixed point for a graph carrying `nan`, an array param holding
`[1.0, nan, inf, -0.0]`, an `inf` coupling tolerance and `(-inf, inf)` ParamSpec
bounds; and `-0.0`, `5e-324`, `-1e-309`, `1.797...e308` and a 17-digit float all
survive with sign and every bit.  The written document is accepted by a strict
reader (`json.loads(..., parse_constant=<raises>)`).

**`compile()` atomicity — ten failure modes, all atomic**
(`repro/compile_atomicity.py`, `repro/compile_atomicity_imvj.py`).  Snapshotting
`_schedule`, `_is_multirate`, `_rate_dividers`, `_committed_rate_dividers`,
`params`, `_params_dtypes`, `_params_shapes`, the whole `_state` tree (values,
shapes, dtypes) including every `_meta` key, the identity of `_compiled_step`,
`_static_data_hashes`, `_dirty`, `_compile_generation`, `_n_traces`, the scan
cache and the default external-input leaves, *after* the pre-compile mutation
and *before* `compile()`:

| failure | result |
|---|---|
| `accelerated_fields` names a non-state field | atomic |
| `static_data_hash()` raises (last statement before the commit point) | atomic |
| `add_node` changes the rate dividers, then the compile fails | atomic |
| a second coupling group with a bogus accelerated field | atomic |
| a multi-rate rebuild that fails edge validation (`ExceptionGroup`) | atomic |
| `reset_state()` then a failing compile | atomic |
| `iqn-imvj` bogus field → `KeyError` inside the `_meta` build | atomic |
| removing a node a coupling group still names | atomic |
| `static_data_hash()` raises on an `iqn-imvj` graph | atomic |
| `update_fn` replaced with a raiser | *does not raise* — `_build_step_fn` + `jax.jit` are lazy, so no node code runs during `compile()` |

In particular the warm starts, the predictor history and `step_count` survive
every failed rebuild, which is what the `_StepPlan` comment claims and what the
`previous_dividers` / `_committed_rate_dividers` split exists for.  `add_node`
and `add_edge` are individually atomic too (`add_node` builds `initial_state()`
before touching either dict; `add_edge` resolves the transform and builds the
mapping before appending).

**Gate mutation testing — 13 of 14 caught.**  Full logs in
`logs/mutation_round1.log` and `logs/mutation_round2.log`; scripts in
`repro/mutate_gates_round*.sh`.  Gate set:
`tests/core/test_non_finite_json_tokens.py`, `tests/usd/test_usd_params.py`,
`tests/fmi/test_non_finite_json_tokens.py`,
`tests/fmi/test_binary_frames_properties.py`, `tests/fmi/test_c_unit.py`
(57 tests, ~48 s baseline).

| # | mutation | caught |
|---|---|---|
| M1 | `encode`: stop recursing into tuples | **NO** (see LOW finding) |
| M2 | `encode`: drop the ambiguous-string refusal | yes (5 failed) |
| M3 | `decode`: do not decode inside lists | yes (4) |
| M4 | `dumps`: `allow_nan=True`, bare tokens back | yes (9) |
| M5 | sign flip: `"Infinity"` decodes to `-inf` | yes (11) |
| M6 | sign flip: `-inf` encodes as `"Infinity"` | yes (7) |
| M7 | `NaN` encodes as the infinity token | yes (11) |
| M8 | `decode`: drop the token table (identity) | yes (6) |
| M10 | `encode`: identity shortcut always copies | yes (1 — the no-copy test) |
| M11 | `encode`: identity shortcut returns the unencoded original | yes (13 + 2 errors) |
| M12 | C `parse_values`: do not step over the opening quote | yes (4, incl. asan/valgrind) |
| M13 | C `parse_values`: drop the closing-quote check | yes (3) |
| M14 | C `do_set`: let non-finite through the JSON path | yes (12) |

(M9 was dropped: its anchor is ambiguous between `encode` and `decode`; M10/M11
cover the same branch from both directions.)

Notable: the sign-flip mutations M5/M6/M7 are the silent-corruption class, and
all three are caught by *several* independent tests each, including the
end-to-end test that drives the compiled C binary.  The C mutations are caught
by the C unit tests under plain, asan+ubsan and valgrind builds.  These gates
can fail, they name the right thing, and their fixtures can express the defect.

**Things I looked at and found no defect in:** `decode_binary` /
`encode_binary` framing and bounds checks; `recv_raw`'s 31-bit length mask and
64 MiB cap; `hdr_count`'s rejection of `"1e3"`, `"5abc"`, `"0x10"` as counts;
`parse_values`' trailing-junk rejection inside quotes; `checked_value` and
`_set` agreeing that a non-finite input is refused on both the `set` and the
`set_state` door; JSON string escaping blocking `strstr`-needle injection into
the C wrapper through the hello reply's `model` field (a `"` inside a string is
always written `\"`, so `"ok":true`, `"values":[`, `"protocol":` and
`"binary":true` cannot be forged from free text); the `_recover_from_escaped_tracers`
/ `_strong_typed` / `_merge_live_params` helpers, none of which mutate `self`
outside the commit point (`_snapshot_params`, `_merge_live_params`,
`_build_step_fn`, `validate` and `_build_dt_step_fn` were all statically checked
for `self.<attr> = ...`, `self.<attr>[...] = ...` and in-place container calls,
and all are clean).

## Severity note on `MADD-ANO-010`

The brief asked whether I agree with the anomaly's `context_dependent` rating
rather than restating it.  **I agree with the rating and not with the
rationale.**  The rating is right: a write-time fail-stop on a fault path is a
genuine availability risk that depends entirely on whether the product's names
and string parameters come from the field, and the entry's advice — validate at
data entry, not at save — is the correct remedy.

But the rationale's load-bearing sentence, *"a fail-stop at write time that
names the path to the offending value, so no wrong number reaches a document
and nothing is lost silently"*, does not survive the HIGH finding above.  On the
FMI wire the refusal is not a fail-stop *at* the write site in any sense a
caller can act on: it unwinds out of a daemon thread, the caller is a C wrapper
on the far end of a closed socket, and the path it names (`$.model`) reaches
only the bridge process's stderr.  The entry should either be widened to say
that, or the bridge should be fixed so that the sentence becomes true again — I
would do the latter, since the reachability claim in the same paragraph is what
justified not hardening `_serve_conn_inner` in the first place.
