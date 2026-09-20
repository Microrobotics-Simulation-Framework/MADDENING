# Stability freeze round — 0.4.0

MADDENING reaches 1.0 by **stabilising surfaces incrementally across the
releases that lead up to it**, not by declaring the whole public API at once.
This is the 0.4.0 round: it judges what this release added, applies the
changes it can defend, and writes down the evidence for every call — including
the calls that were *not* to promote anything.

It supersedes the round written on 2026-09-17 against a tree 470 commits
behind this one (branch `chore/stability-freeze`, commit `603fef3`). What that
round found is re-measured here rather than carried over; where it is not
re-measured, this document says so instead of repeating it.

Run last in the release, deliberately: every other 2.x phase had to land first
so that the surface had stopped moving underneath the judgement.

## What a level costs

Promoting a surface to `stable` says its signature will not change
incompatibly before the next major version. The mechanics are in
[the deprecation policy](deprecation_policy.md); the price is:

- a breaking change needs two minor releases of `DeprecationWarning` and can
  only land in a major release;
- **lowering** the level again is itself a breaking change, so a promotion is
  hard to take back — which is why a level is cheap to set *before* the
  release that first ships it and expensive afterwards;
- a `stable` *class* promises every public method and property an instance
  answers to, inherited ones included. `stable_api.json` records 239 members
  behind today's 17 tagged surfaces, and
  `scripts/check_stable_signatures.py` fails on any change to any of them —
  **distinguishing a break from a compatible widening**. A new parameter with
  a default, added after the existing ones, is classified `COMPATIBLE` and
  says so ("the snapshot being out of date, NOT a break of the contract"); a
  rename, a reordering, a changed default, a changed annotation or a removal
  is `BREAKING` and asks for a major version bump. Both still fail, because
  the snapshot has to move either way; they fail with different messages, and
  it is the message that carries the verdict.

One clarification this round adopts explicitly, because the level docstrings
are ambiguous about it: **between 0.4.0 and 1.0.0 a `stable` tag is announced
intent, not yet the contract.** The snapshot guard makes every change to such
a surface visible and deliberate in the meantime; at 1.0.0 the snapshot
*becomes* the contract. Without that reading the
[PEP 589 typing phase](typing.md), which replaces bare `dict` annotations with
`TypedDict`s on exactly these surfaces, would be a breaking change on the day
it lands.

## The criteria

A surface is promoted to `stable` only when **all** of these hold:

1. it is covered by tests that exercise its *contract* rather than its
   implementation;
2. it did not change during the 0.4.0 cycle, or changed only additively;
3. its signature names no type we do not control, and no MADDENING type that
   is itself unfrozen;
4. something outside its own test file already depends on it, so the promise
   is being made in practice rather than in principle;
5. there is a mechanism that fails if it changes incompatibly.

`experimental` is the default for anything whose *numbers* are still being
calibrated — thresholds, safety factors, tolerances — because a changed
default is a silent break, and the policy rates that worse than a failure.

## What this round changed

| | before | after |
|---|---|---|
| surfaces in the report | 107 | 141 |
| `stable` | 17 | **17** |
| `evolving` | 74 | 80 |
| `experimental` | 10 | 38 |
| `deprecated` | 6 | 6 |

**Nothing was promoted to `stable`.** That is the round's main judgement and
§"Not promoted" gives the reason for each candidate.

Two things were tagged that carried no level at all, and a level of `none` is
not a weaker promise than `experimental` — it is the *absence* of a statement,
in a report whose header claims to enumerate the public API.

### Six types a `stable` signature names (→ `evolving`)

`sharded_cg` and `sharded_gmres` are `stable` and return `SharedSolveResult`.
`SimulationNode` is `stable`; `boundary_input_spec()` returns
`BoundaryInputSpec`, the flux hook returns `BoundaryFluxSpec`, and
`uncertainty_spec()` returns `UncertaintySpec`. `GraphManager` is `stable`;
`validate_sharding()` returns `list[ShardingIssue]` and `add_coupling_group`
takes a `CouplingGroup`. **None of the six carried a tag of its own**, so the
frozen half of each promise was the method and the free half was the object it
hands you.

*Evidence for `evolving`, not `stable`:* all six are frozen dataclasses whose
shape is settled, and a frozen dataclass gaining a field is compatible for a
reader — which is what `evolving` means. Against `stable`: three of the six
were touched during this cycle, none has usage history outside the package,
and criterion 4 is not met for any of them. Against leaving them untagged:
criterion 3 is a rule the `stable` set must eventually satisfy, and an
untagged type cannot satisfy or fail it.

This set was found by hand on 2026-09-17. Three days and 470 commits later it
was unchanged **and one longer** — `DiscretizationOrder` joined it when
`HeatNode.discretization_order` was added. That is the argument for
mechanising it, which
`tests/compliance/test_stable_signatures.py::TestAStableSignatureNamesNoUntaggedType`
now does: it reads the annotations of every `stable` signature, resolves each
name (including a forward reference nested in a subscript, and a class only a
`if TYPE_CHECKING:` block imports), and fails on an untagged MADDENING type or
on a name it cannot account for at all.

`DiscretizationOrder` is the one recorded exemption and cannot be fixed the
same way: it lives in `core/compliance/metadata.py`, which defines
`StabilityLevel`, and `core/compliance/stability.py` imports that, so
`metadata` cannot import the decorator without a cycle.

### The `[verify]` harnesses (→ `experimental`, 28 surfaces)

`maddening.testing` is new in 0.4.0, has a 31-name `__all__` in `mms.py`
alone, is documented in `node_authoring.md`, `testing_standards.md`,
`verification.md` and `adaptive_node.md`, and is installed by its own extra.
It carried **zero** `@stability` tags, and nothing imports it on the way to
anything else, so it was absent from the report entirely — a documented public
module the stability contract did not mention.

*Evidence for `experimental`:* the MMS and GCI entry points are days old (9
commits on `mms.py` this cycle) and their public surface is largely *numeric
policy* — `DEFAULT_SAFETY_FACTOR`, `CAUTIOUS_SAFETY_FACTOR`,
`DEFAULT_ASYMPTOTIC_ORDER_TOLERANCE`, `DEFAULT_STAGNATION_RTOL`,
`MIN_APPARENT_ORDER`. Those are the thresholds a convergence verdict turns on,
they have been calibrated against the built-in nodes and nothing else, and
changing one silently changes whether a user's node passes. `experimental`
("may break in any minor release, opt-in only") is exactly the promise that
fits an opt-in extra whose constants are still being tuned.

*What is not covered:* `@stability` decorates a class or a function, so the
ten module constants in `mms.__all__` are outside the registry. The
deprecation policy already says module constants are not covered by the
signature guard; this is the first place it matters in practice, and a reader
should take the level on the surrounding functions as speaking for them.

## Not promoted, and what would change that

| Surface | Level | Why not `stable` | What would change it |
|---|---|---|---|
| `api.auth.APIAuth`, `is_loopback`, `is_routable_peer` | `evolving` | 41 tests in `tests/api/test_bearer_auth.py`, so criterion 1 is met — but the module is three days old, `api/__init__.py` exports nothing so it is deep-import only, and the bearer-token model itself has an open residual risk (single shared secret, no per-peer identity). | A release cycle in which the token model does not change, and a decision on whether `maddening.api` should export anything. |
| `transport_auth.*` (5) | `evolving` | Same age. 40 tests, one dedicated file, and CURVE key derivation from a shared token is the same trust model with the same open residual risk (MADD-ANO-015). Criterion 4 fails: nothing outside the package uses it yet. | Downstream use in MIME/MICROROBOTICA, and a decision on per-peer keys, which would change the signatures. |
| `api.server.origin_is_same_site` | `evolving` | **Zero tests name it.** It is exercised only through `SimulationServer` in `tests/api/test_cross_origin_requests.py`. Criterion 1 is not met: the integration test pins the behaviour of the server, not the contract of the function. | A direct test of the function's own contract — which is worth writing regardless of the level. |
| `api.server.warn_if_publicly_bound` | `evolving` | 18 tests, but the warning text and category are the observable behaviour and neither is pinned by the signature. | Nothing this release; it is a diagnostic, and `evolving` is the right resting place. |
| `serialization.json_codec.dumps_encoded` and the other four | `evolving` | The *wire format* (quoted `"NaN"`/`"Infinity"` tokens) is settled and 35 tests cover the round trip. The **reserved literals are a trap**: adding one changes every `st.text()` draw in the package, which has already caught three property tests this release. Freezing the function does not freeze the token set. | Either the token set becomes part of the frozen contract explicitly, or it is documented as outside it. |
| `core.params.*` (5) | `evolving` | Best-documented addition in the set (7 doc pages) and `ParamSpec` is named in 38 test files. Fails criterion 2 outright: 6 commits this cycle, and the params pytree is the direct input to the PEP 589 `TypedDict` phase, which is *designed* to change these annotations. | The `TypedDict` conversion landing, then a cycle of quiet. |
| `core.coupling.mapping_spec.build_mapping` | `evolving` | **Zero tests name it.** Exercised only through `MappingSpec.build()`. The free function is public API with no direct coverage at all. | A direct test, or making it private and keeping `MappingSpec.build` as the entry point. The second is probably the right answer. |
| `core.simulation.profiler.count_hlo_ops` | `evolving` | **Zero tests name it**; reached only from inside `compile_counts()`. Same shape as `build_mapping`. | As above. |
| `core.simulation.profiler.compile_counts`, `CompileCounts` | `evolving` | 25 dedicated tests, and the compile-count gate is the one piece of this group with a real contract. But `profiler.py` took 7 commits this cycle and `profile_graph`'s signature grew four parameters. | One quiet cycle. This is the strongest promotion candidate in the release for 0.5.0. |
| `cloud.resume.download_and_load_state` | `evolving` | 48 tests across two files, a dedicated user guide, a real `__all__` export, and a deprecation shim at the old location — the best-evidenced new surface by some margin. It still fails criterion 2 (it *moved* this cycle) and criterion 4. | One cycle at its new home with the shim still in place. Promote in 0.5.0 if nothing moves. |
| `sysid.*` (9) | `evolving` | 27 commits on `sysid.py` this cycle — the highest churn in the release — and **no dedicated user-guide page** for the single largest new surface. | Documentation first, then stability. Promoting an undocumented nine-function API would be promising a shape nobody can read. |
| `nodes.adaptive.base.AdaptiveNode` | `evolving` | The 2026-09-17 round argued for demoting it to `experimental` "for free today and not after 0.4.0 ships". I am **not** taking that recommendation: since then it gained a documented dtype policy, an algorithm guide, a verification benchmark and 12 test files, and `docs/release_notes/v0.4.0.md` states the `evolving` choice deliberately. The premise the demotion rested on has changed. | Nothing — `evolving` is the right level and the release notes already say why. |
| `cloud.multigpu.halo_unstructured.*` (6) | `evolving` | Five of the six existed before 0.4.0 untagged and were tagged this cycle; `exchange_traffic` is new. Tagging them *was* this release's step. | A cycle at `evolving` with the sharding topology doc stable. |
| `fmi.tcp_bridge.FmuTcpBridge` | `evolving` | Publicly exported and well tested, but it is an **unauthenticated listening socket** with its own protocol, documented as trusted-clients-only and explicitly out of scope of MADD-ANO-015's fix. Freezing its constructor freezes the shape of that exposure. | A decision on whether it gets the same auth treatment as the ZMQ transports. That decision will change its signature. |

## Three `stable` surfaces whose *behaviour* moved

The round-3 audit diffed all 17 `stable` surfaces against `main`. Every
**signature** change this release is additive, which is the right answer and is
now enforced (see below). It also found three **behavioural** changes, which
the signature guard cannot see at all:

- `HeatNode(timestep=0.01, thermal_diffusivity=1.0)` constructed on `main` and
  raises on 0.4.0. Reproduced: `ValueError: timestep 0.01 is unstable for this
  rod: the Fourier number dt*alpha/dx^2 is 1, above the 0.5 limit of the
  order-2 stencil`.
- `HeatNode`'s default `initial_state()["temperature"][0]` moved from 100.0 to
  0.02 — measured 0.0 on the shipped defaults today.
- The `HeatNode` boundary-flux units changed.

**Judgement: `stable` remains the right level for `HeatNode`, and these are not
violations to fix.** Three reasons, in order of weight:

1. `StabilityLevel.STABLE`'s own docstring and the report's gloss both say the
   contract is locked **at v1.0.0**, not now. Between 0.4.0 and 1.0.0 the tag
   is announced intent — the reading this document adopts explicitly above.
2. All three are disclosed in the release notes, so the change is deliberate
   and a reader upgrading is told.
3. The first is a *refusal replacing silent wrongness*: the configuration it
   now rejects diverged to NaN. Turning a wrong answer into an error is the
   direction the deprecation policy prefers, and `scripts/check_heat_stability.py`
   already gates it (138 constructions verified, 120 not statically evaluable
   and explicitly **not** checked).

**What this round does record is the gap they expose.** The signature guard
reports `OK` on all three, because none of them touches a signature. The
deprecation policy already lists behaviour, exception types, the contents of
dict-shaped arguments, instance attributes and module constants as outside the
guard — this is the first release where that exclusion cost something
concrete, and the audit found the changes by diffing two trees by hand rather
than by anything committed.

So: before 1.0.0 the `stable` set needs a **behaviour** pin as well as a
signature pin, or the tag promises a reader more than anything checks. That is
not built here — a behaviour snapshot needs a decision about what to record
(constructor acceptance? `initial_state()`? one step of `update()`?) and that
decision is a round of its own. It is the single largest thing this freeze
leaves open, and it is named here so the 0.5.0 round starts from it rather
than rediscovering it.

## Still public, still untagged

The report covers the modules this release worked on. It does not cover:

- **`maddening.api.server.SimulationServer`** — the REST API's central class,
  in every example, with no tag. Two functions beside it in the same file now
  have one, which makes the omission more visible rather than less.
- **`maddening.CloudSession` / `maddening.CloudConfig`** — importable from the
  top-level package, with new fields added this release (PR 94:
  `api_token`, `transport_token`, `container_env()`, `CloudConfig.ports`,
  `CloudConfig.api_port`), and untagged. So is the rest of `cloud.session`
  and all of `cloud.launcher`.
- **The whole of `maddening.viz` and `maddening.viz.backends`** — zero
  `@stability` occurrences in the package, including `NetworkRelay`,
  `NetworkReceiver`, `CommandPublisher` and `CommandReceiver`, which are the
  components MADD-ANO-015 is about.
- **`maddening.surrogates`' concrete architectures and training callbacks** —
  the ABCs `SurrogateArchitecture` and `SurrogateTrainer` are tagged
  `experimental`; none of `MLPDirect`, `FNODerivative`, `DeepONetDirect`,
  `EarlyStopping`, `ModelCheckpoint` or the five physics losses is.
- **`core.simulation.checkpoint`'s `save_state`/`load_state` family**, and
  `core.transforms`, `core.schedule`.

This is not an oversight in one place; it is a systematic gap outside the
modules 0.4.0 touched. **It is deliberately not closed in this round** — three
of those groups (`viz`, `cloud.session`, `surrogates`) would need a level per
surface and the evidence to justify it, which is a round of its own, and
guessing a level for forty surfaces at the end of a release is how a wrong
promise gets made at scale. The 0.5.0 round should take `viz` and
`cloud.session` together, since PR 94's new parameters land on both.

The one exception this round *did* close is the set a `stable` signature
already named, because those were not merely untagged — they were making half
a promise that something else had already committed to.

## The 2026-09-17 round's open questions, re-checked

Two are answered by work that landed since:

- **`GraphManager.to_dict` has no coupling-group field.** It has one now:
  `to_dict` emits `coupling_groups` carrying every field of every group.
- **`ift_linear_solve`: expose the solver knobs.** Done —
  `solver`, `preconditioner`, `rtol` and `atol` are keyword-only parameters
  today. The second half of that question, whether it should own its error
  type, is not re-checked here.

One has changed shape:

- **"The REST API has no `@stability` tag anywhere."** It has two now
  (`origin_is_same_site`, `warn_if_publicly_bound`), both added by PR 94.
  `SimulationServer` itself still has none, so the underlying question — what
  the API promises about its response schema, and what it returns when a
  simulation diverges — stands.

Three are **carried over unverified**: the `AdaptiveNode` dtype policy under
`jax_enable_x64`, the silent USD mapping-weight path, and whether the adaptive
hooks should take `boundary_inputs` and `dt`. They are recorded at
`603fef3:docs/developer_guide/api_freeze_proposal.md`. They are not restated
here, because restating a judgement taken against a tree 470 commits ago as if
it were current is the specific mistake this round was told to avoid.

## What this does not decide

- The 1.0 `stable` set. This is one increment towards it.
- Anything about `viz`, `cloud.session`, `cloud.launcher` or the concrete
  surrogate architectures.
- Whether `build_mapping` and `count_hlo_ops` should be public at all.
- Module constants, instance attributes, exception types, behaviour, or the
  contents of dict-shaped arguments — none of which the signature guard
  covers. See [the deprecation policy](deprecation_policy.md), and the section
  above on the three `stable` surfaces whose behaviour moved this release.
- What a behaviour pin for the `stable` set should record.  That is the
  largest thing left open, and 0.5.0 should start from it.
