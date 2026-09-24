# Testing Standards

Test requirements at three levels, all enforced by CI.

## 1. Unit Tests (mandatory for all code)

Location: `tests/<module>/test_<name>.py`

Requirements:
- Test each public method
- Test normal operation and edge cases
- For nodes: verify {term}`JAX-traceability <JAX-traceable>` (`jax.jit`, `jax.grad`, `jax.vmap`)
- Test boundary input handling (missing inputs, default values)
- Test parameter edge cases

```python
def test_your_node_basic():
    node = YourNode("test", timestep=0.01)
    state = node.initial_state()
    new_state = node.update(state, {}, 0.01)
    assert "field" in new_state

def test_your_node_jit():
    node = YourNode("test", timestep=0.01)
    state = node.initial_state()
    jitted = jax.jit(node.update)
    new_state = jitted(state, {}, 0.01)
    # Must not raise
```

## 2. Integration Tests (mandatory for nodes)

Test the node within a `GraphManager`:

```python
def test_your_node_in_graph():
    gm = GraphManager()
    node = YourNode("test", timestep=0.01)
    gm.add_node(node)
    state = gm.run(n_steps=10)
    assert "test" in state
```

Test edge connections if the node consumes or produces coupled data.

## 3. Verification Benchmarks (mandatory for physics nodes)

Location: `tests/verification/test_<name>_<benchmark>.py`

Compare against analytical solutions or published reference data. Register with the `@verification_benchmark` decorator:

```python
from maddening.core.validation import verification_benchmark

@verification_benchmark(
    benchmark_id="MADD-VER-XXX",
    description="Analytical solution comparison for ...",
    node_class="YourNode",
    reference="AuthorYear",
)
def test_your_node_analytical():
    """Compare YourNode output to analytical solution."""
    node = YourNode("bench", timestep=0.001, ...)
    gm = GraphManager()
    gm.add_node(node)
    state = gm.run(n_steps=1000)

    # Compute analytical solution
    analytical = ...

    # Compare
    error = jnp.abs(state["bench"]["field"] - analytical)
    assert jnp.max(error) < tolerance, f"Max error {jnp.max(error)} exceeds {tolerance}"
```

{term}`Verification benchmarks <Verification benchmark>` must:
- Use a registered benchmark ID (`MADD-VER-XXX`)
- Cite the analytical solution or reference data source
- State the tolerance and justify it
- Be referenced in the node's algorithm guide (Verification Evidence section)

## Running Tests

```bash
# Activate venv
source ../venvs/.maddening/bin/activate

# Full test suite (exclude viz — requires display)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v --tb=short --ignore=tests/viz

# Just compliance + verification
python -m pytest tests/compliance/ tests/verification/ -v --tb=short

# Just your new node's tests
python -m pytest tests/nodes/test_your_node.py tests/verification/test_your_node_*.py -v

# Compliance CI scripts
python scripts/check_anomalies.py
python scripts/check_impl_mapping.py
python scripts/check_citations.py
```

## Environment Variables

| Variable | Purpose | Typical Value |
|----------|---------|---------------|
| `JAX_PLATFORMS` | Force CPU backend (avoids GPU issues) | `cpu` |
| `XLA_FLAGS` | Disable GPU autotune (avoids equinox segfaults) | `--xla_gpu_autotune_level=0` |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD` | Prevent plugin conflicts | `1` |
| `MADDENING_HYPOTHESIS_PROFILE` | Hypothesis depth/settings profile (`dev` or `ci`) | `dev` |

## 4. Property-Based Testing (recommended for physics nodes)

Location: `tests/verification/hypothesis/` for the numerics suite; property
tests that belong with a package (`tests/fmi/`, `tests/core/`) stay there.
Configuration is global either way — see *Hypothesis profiles* below.

Beyond analytical benchmarks, MADDENING provides a Hypothesis-based
property-testing layer. See the full [Verification Guide](verification.md).

**Property-based testing** (hypothesis) — test universal invariants:

```python
from maddening.testing.strategies import node_states, bounded_dt
from hypothesis import given

@given(state=node_states(my_node, bounds={...}), dt=bounded_dt())
def test_my_node_finite(state, dt):
    out = my_node.update(state, {}, dt)
    for val in out.values():
        assert jnp.all(jnp.isfinite(val))
```

Note the absence of `@settings`: `deadline`, `print_blob`, the example
database and `max_examples` all come from the active profile.

### Hypothesis profiles

Profiles are registered and loaded in the **root** `tests/conftest.py`, so
every property test in the tree gets them, whether or not the Hypothesis
pytest plugin is loaded (the suite runs with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, which disables the plugin and its
`--hypothesis-profile` flag). Do not register profiles in a sub-directory
`conftest.py`, and do not repeat `deadline=None` per test.

| Profile | `max_examples` | Used by | Notes |
|---------|----------------|---------|-------|
| `dev`   | 50  | default for every local run | fast enough to keep the full suite in its ~50-minute budget |
| `ci`    | 200 | the `verify-hypothesis` GitHub Actions job | 4x the search; the depth tiers below scale with it |

A per-test `@settings(max_examples=...)` **overrides the profile**, so a test
that sets its own cap runs at that cap under both profiles and `ci` buys it
nothing. That is the whole reason for the house rule and the depth tiers
below.

Both set `deadline=None` (a JAX compile blows any wall-clock deadline),
`print_blob=True` (a failure prints a `@reproduce_failure` blob you can
paste into the test) and `suppress_health_check=[HealthCheck.too_slow]`.

Select a profile with the `MADDENING_HYPOTHESIS_PROFILE` environment
variable; an unknown name is a hard error rather than a silent fallback:

```bash
# local default -- same as setting nothing
MADDENING_HYPOTHESIS_PROFILE=dev pytest tests/verification/hypothesis/

# reproduce what CI runs
MADDENING_HYPOTHESIS_PROFILE=ci pytest tests/verification/hypothesis/
```

Every run prints the active profile in the pytest header, e.g.
`hypothesis profile: ci (max_examples=200, database=.../.hypothesis/examples)`.

### The example database

Failing examples are written to `<repo>/.hypothesis/examples` (git-ignored,
absolute path fixed in `tests/conftest.py` so it does not depend on the
working directory). Hypothesis replays them first on the next run, so a
falsifying example found once stays found. The `verify-hypothesis` CI job
caches that directory with `actions/cache`, keyed per run with a
`restore-keys` prefix, so the database survives across runs; a cache miss
is not an error, the job just starts from an empty database.

To clear a stale entry locally, delete `.hypothesis/`.

### Depth tiers

A hand-picked `max_examples` overrides the profile, so a test that carries
one runs at that number under `dev` and under `ci` alike. With about a
hundred such call sites across the tree, the deeper profile bought almost
nothing: `tests/verification/hypothesis/` measured **707 s under `dev`
against 721 s under `ci`** — 2% apart, for a profile that asks for four
times the search. With the tiers in place the same suite measures **462 s
under `dev` and 1365 s under `ci`** — `dev` is a third faster than it was,
and `ci` is 3x `dev` because it is finally doing more work rather than the
same work.

Depth a test really does need to state is therefore expressed as a
**tier**: a multiple or fraction of the active profile's `max_examples`,
resolved once in the root `tests/conftest.py` and imported by name.

```python
from tests.conftest import EXAMPLES_CHEAP, EXAMPLES_COSTLY, EXAMPLES_STANDARD

@given(x=st.floats(-1e4, 1e4, allow_nan=False))
@settings(max_examples=EXAMPLES_CHEAP)
def test_clamp_is_idempotent(x):
    ...
```

| Tier | Resolves to | `dev` | `ci` | One example is |
|------|-------------|-------|------|----------------|
| `EXAMPLES_CHEAP` | 4x the profile | 200 | 800 | a pure function on scalars or small arrays, or a codec/socket round trip: no JAX trace, no graph build, no fresh compile — a few milliseconds at most |
| `EXAMPLES_STANDARD` | the profile itself | 50 | 200 | an eager node update on a fixed shape, a call into an already-compiled step, a constrain/unconstrain round trip |
| `EXAMPLES_COSTLY` | 2/5 of the profile, floored at 20 | 20 | 80 | a fresh JAX trace and compile, a dense solve or a `vjp` per draw, an optimiser loop, a multi-device `shard_map`, a full rollout, a graph built and compiled per draw |

The tiers are named for **cost per example**, because that is the only
thing the test author is in a position to judge; how wide to search at
that cost is the profile's business. Pick one by reading what a single
example actually does, not by matching the number that used to be there.
Every run prints the resolved tiers in the pytest header beside the
profile name.

`stateful_step_count` is deliberately **not** tiered: it sets how long one
example is, not how many there are, so scaling it with the profile would
multiply `ci`'s work by the square and turn one example into seconds. The
state machines in `tests/property/` keep an absolute step count — with the
reason in the docstring — and leave `max_examples` to the profile.

### `max_examples`: the house rule

**A property test carries no `max_examples`. The profile owns it.** That is
what makes "run the suite deeper" a one-line change instead of a hundred.

An explicit `max_examples` is an exception that must be justified *in a
comment on the line above it*, and is only justified when a single example
is genuinely expensive — a fresh JAX trace/compile per draw, an optimiser
loop, a multi-device `shard_map`, a full rollout. Then:

* the floor is **20**. Below that Hypothesis barely gets past reuse and
  generation and the test is decoration: it can pass for months and fail
  once, which is exactly the failure mode this configuration exists to
  remove;
* reach for a **tier** before a number. A tier says what the example costs
  and lets `ci` search deeper than `dev`; a number freezes both;
* a bare number is still right where it encodes a real constraint, and the
  comment must say which:
  * *the search space is exhausted anyway* — a rate multiplier drawn from
    2..5, a pair of timesteps sampled from four values. Hypothesis stops
    when it has seen every case, so a profile-relative cap would promise
    depth that does not exist;
  * *one example is measured in seconds* — a 30-iteration
    Levenberg–Marquardt fit over a full rollout. `EXAMPLES_COSTLY` under
    `ci` would turn that test into minutes on its own;
* an explicit value *above* the profile default is fine and needs no
  special pleading — it only ever deepens the test.

`verify_node` / `assert_node_verified` are a separate knob: they build
their own `settings` internally (with `database=None`), so the profile does
not reach them. Their `max_examples` argument defaults to 200; a call site
that lowers it is subject to the same floor and the same
comment-your-reason rule.

### `assume` is a budget, and nothing tells you when you have spent it

**Generate the valid shape instead of rejecting the invalid one.** Every
`assume()` that fails, and every strategy `.filter()` that rejects, throws
away a draw the test already paid to build — and says nothing about it. A
test rejecting 5% of its draws and one rejecting 85% print the same green
tick and the same `max_examples`.

The one thing that ever says otherwise is `HealthCheck.filter_too_much`, and
it is a sampling test on the first few dozen draws, so its edge is very
soft. Measured against the constants in Hypothesis (50 rejected draws before
10 accepted ones — unchanged across 6.165-6.168, and checked behaviourally on
whatever version is installed), the chance that *one run* of a test trips it
is:

| filter rate | 40% | 55% | 64% | 70% | 80% | 85% | 90% |
|---|---|---|---|---|---|---|---|
| per-run risk | 2e-12 | 1e-6 | 4e-4 | 7e-3 | 23% | 61% | 93% |

Which is how a test sits green for months and then goes red for whoever next
narrows an unrelated strategy.
`test_accelerating_every_field_lands_on_the_same_answer_as_plain_iteration`
did exactly that: its first gate rejected 55% of draws on a clean tree, an
inert-knob change moved it to 64%, and together with five more `assume`
calls it failed CI with "9 inputs generated successfully, 50 filtered out".

What a high rejection rate does **not** do on this Hypothesis version is
silently shallow the search. The engine keeps drawing until it has
`max_examples` *valid* examples and only gives up below ~1% valid; measured
against a synthetic gate, a test at 98% rejection still ran its full 200
examples, using 9068 draws to do it. So the cost is wall-clock and
health-check fragility, not a number of examples that lies.

**Measure it, do not guess.** `scripts/audit_property_rejection.py` prints
the rate for every test in a run:

```bash
MADDENING_HYPOTHESIS_PROFILE=ci PYTHONPATH=src JAX_PLATFORMS=cpu \
    python scripts/audit_property_rejection.py \
        tests/property tests/verification/hypothesis
```

`--check` turns it into a gate: the run fails if any test exceeds
`MAX_REJECTION`. That is what the `verify-hypothesis` CI job runs, wrapped
around the suite it was already running, so the measurement costs nothing.

**The pattern.** When a gate rejects anything worth mentioning, the fix is
almost always a parameter on the strategy, defaulted so no other caller
changes:

```python
# before: a third of every draw built a graph and threw it away
recipe = draw(graph_recipes(require_coupling_group=True))
assume(not any(g.subcycling for g in recipe.coupling_groups))

# after: 0.0% rejected
recipe = draw(graph_recipes(require_coupling_group=True,
                            allow_subcycling=False))
assert not any(g.subcycling for g in recipe.coupling_groups), (
    "allow_subcycling=False must not produce a subcycled group")
```

Note the second half. **Promote the `assume` to an `assert`.** A generator
that stops holding up its end then fails loudly instead of quietly going
back to discarding half the search.

Where the condition is an outcome rather than a shape — did the solve
converge, did the fit find a gradient — it cannot be generated. Say so, and
**put the measured residual in the test's docstring** so the next reader
knows what that test's `max_examples` actually buys.

**Never reach for `suppress_health_check=[HealthCheck.filter_too_much]`.**
It makes the red go away and deletes the only signal anyone gets that the
gate is getting worse — the test goes on spending most of its wall-clock on
draws it throws away, and the next person to narrow that strategy has
nothing to notice. Measure it and fix the generator, or measure it and
record the residual; suppressing is neither.

**A rejection rate is a property of the checkout path, not just the test.**
Since the 6.16x line Hypothesis harvests the literal constants out of every
*local* module and injects them into draws. Whether a module counts as local
is decided by `is_local_module_file`, which excludes any path containing a
`test` or `tests` component — so a git worktree under
`MADDENING-wt/test/<branch>/` has the whole of `src/` classified as test
files and injection silently **off**, while CI at
`/home/runner/work/MADDENING/MADDENING` has it **on**. Measured on one
commit: 0 local constants in such a worktree against 497 without the
component, and the four `TestFIM` rates came out 3–10 points *lower* with
injection on. The audit prints which side it ran on with every run; if you
are comparing two measurements, check that line first.

**Overruns are a different problem.** The audit reports them in their own
column. An `overrun` is Hypothesis running out of entropy for a large draw,
not the test rejecting an input; it answers to `HealthCheck.data_too_large`
(20 overruns before 10 valid, a much tighter window) and it is fixed by
drawing smaller structures, not by removing a gate. Every
`hypothesis.extra.numpy.arrays` property in this tree overruns a few percent
of its draws with no `assume` written anywhere in it.

**Node battery** — finite outputs, structure, determinism, jit/eager
agreement, finite gradients, in one call:

```python
from maddening.testing.verification import assert_node_verified

def test_my_node():
    assert_node_verified(my_node, bounds={"field": (lo, hi)})
```

Install with `pip install maddening[verify]`.

## Test time budget

Every push runs the default lane (everything not marked
`@pytest.mark.slow`) twice, once per JAX lane, and it is what everyone waits
on. Each test's wall-clock on the GitHub runner (setup + call + teardown)
falls in one of three bands:

| Time on CI | What it means |
|---|---|
| up to 1 s | Fine. |
| 1–5 s | The watch list. Optimise it when you are in the file; usually the cost is JIT compilation (see below). |
| over 5 s | Mark it `@pytest.mark.slow`, unless it can be made much faster, or it guards something important enough to pay for on every push. In that case, add it to `tests/duration_allowlist.txt` as `<node id> # kept: <why>`. |

Slow tests are not lost: `slow-tests.yml` runs the whole suite, slow tests
included, on Monday, Wednesday and Friday, and on demand from the Actions
tab. A property under `tests/verification/hypothesis/` that is marked slow
also still runs on every push, at the `ci` profile's depth, in the
`verify-hypothesis` job, which selects `-m "slow or not slow"`. A slow test
that needs a tool the default lane installs (valgrind, for
`tests/fmi/test_c_unit.py`) needs `slow-tests.yml` to install it too, or it
skips there and runs nowhere.

CI enforces the budget in the `Test time budget` step of each test lane
(`scripts/report_test_durations.py`, reading pytest's JUnit XML):

- The run page's summary shows the band counts, the 30 slowest tests and
  the 15 slowest files. Every test over 1 s is also listed at the end of the
  log (`--durations=0 --durations-min=1.0`). The XML is uploaded as the
  `test-durations-*` artifact and kept for 90 days.
- An unlisted test over **5 s** gets a warning annotation on the run.
- An unlisted test over **20 s** fails the job.

The hard line is four times the policy line because of runner noise. On
eight green lane-runs of the same tree, one test's time varied by 1.7x
between runs at the median and 3x at the 90th percentile. Some of that is
the runner. The rest is that the first test to compile something pays for
every later test that reuses it, so moving or marking one test moves
another's time. Judge a test on more than one run.

### Sharded lanes

Each JAX lane runs as four jobs on four runners, each running its share of
the suite one test at a time (`MADDENING_TEST_SHARD=i/4`,
`tests/_sharding.py`). The split is by test file, and it is stable. A file's
shard is a hash of its path, or an explicit pin in `PINS` for the heaviest
files, never a function of the test list. So:

- adding or removing tests, or whole files, moves no other file;
- a file's tests stay together, so its fixtures build once;
- shard *i* of a pull request holds the same files as shard *i* of the base
  branch, which is what lets a shard reuse that shard's compilation cache.

`slow-tests.yml` is split the same way, four runners per lane, which takes
the whole-suite run from about three hours to under one.

Every job still collects the whole suite, so every `conftest.py` runs as
it would in a single process, and deselects the other shards' files. The
per-shard `Test time budget` step gates. The `Test durations` job writes
one summary per lane from all four shards' reports.

To rebalance, edit `PINS`, which moves only the files you pin, and take the
per-file totals from the lane summary. Changing the job count re-deals
every file: change `shard:` and `MADDENING_TEST_SHARD` in `ci.yml`, and
`PINS_FOR`. `tests/compliance/test_ci_sharding.py` checks that they agree.

`pending triage` entries in the allowlist are tests that were already over
5 s when the budget arrived (2026-09-24). Each one is to be marked slow,
made faster, or kept with a reason, and the list only shrinks. The summary
lists entries that may now be removable.

### The compilation cache: warm and cold runs

The test lanes use JAX's persistent XLA compilation cache, and whether a
run reads it decides what its times mean:

| Run | Cache | Times mean |
|---|---|---|
| Pull request | **warm**: restores the base branch's cache, never saves one | Fast. Anything the PR adds or changes still compiles from scratch, because its programs are not in the base cache, so a new slow test is still caught on the PR that adds it. |
| Push to `main` / `release/**` (after a merge) | **cold**: starts empty, saves the result for the next PRs | Accurate: every compile is paid in full. |
| `slow-tests.yml` (Mon/Wed/Fri) | **off**, one process per shard | The authoritative timing of the whole suite. Triage and allowlist edits are based on these runs. |
| Pull request with `[cold-ci]` in its head commit message | **cold**, not saved | For before/after numbers while optimising tests. |

Pull requests never save, so a second push cannot read the first push's
cache: a new test that takes 25 s cold would otherwise pass at 8 s. Each
shard has its own cache, which works because shard *i* holds the same files
on every branch (see *Sharded lanes*). The cache key also includes the
runner's CPU model, because XLA compiles for the host's instruction set. A
shard that finds no cache for its model runs cold. Every lane summary
states which kind of run it was: `warm`, `cold`, or `mixed` when the shards
differed.
A warm run never lists allowlist entries as removable, because a test that
is only fast when its compile is cached is still slow.

### Why a test is slow

Each lane records every test's JAX tracing, lowering, XLA compile and
cache-read time (`tests/_jax_timing.py`, switched on by
`MADDENING_TEST_JAX_TIMING=1`). The summary splits the slowest tests into:

- **compiling**: XLA backend compilation. This is the only part a cache
  removes.
- **tracing/lowering**: building the program in Python. No cache removes
  it.
- **running**: executing, Python overhead, I/O, and anything a subprocess
  does. An un-jitted `for` loop over `node.update` lands here.

**Slow even with a warm cache** lists every test still over 5 s once its
compile time is subtracted. Every run shows this list, cold or warm. A cache
cannot fix these tests; they need a code change or `@pytest.mark.slow`.

The usual fixes:

- **Running**: move the loop into JAX. Use `jax.lax.fori_loop` or
  `jax.lax.scan` over a jitted update instead of a Python `for` loop that
  dispatches every step op by op. Measured 2026-09-24:
  `test_heart_pump.py::test_steady_state_pressure` (16,660 eager steps) and
  `test_rigid_body.py::test_quaternion_stays_normalized` (10,000) spent
  over 99% of their time this way.
- **Tracing/lowering**: build the graph or function once, in a
  module-scoped fixture, and reuse it rather than rebuilding it per test
  or per example.
- **Compiling**: in a property test, keep array shapes and static arguments
  fixed across examples. A new shape or a new Python-level constant is a
  new program, so draw values rather than shapes and pass them as traced
  arguments. Use the `EXAMPLES_COSTLY` tier (see *Depth tiers* above) for
  properties that still compile per example.

The other usual cost is eager dispatch. A Python `for` loop over
`node.update` runs every operation one at a time, and thousands of steps
of it take minutes. Run the same updates through `jax.lax.fori_loop` or
`jax.lax.scan`: the heart-pump steady-state test went from 170 s on CI to
under a second that way, with a bit-identical result.

## Test Organization

```
tests/
├── core/           # GraphManager, scheduling, coupling, adaptive, checkpoint
├── nodes/          # Per-node unit tests
├── surrogates/     # Surrogate framework tests
├── api/            # Server, WebSocket, binary encoder tests
├── viz/            # Visualization tests (skipped in CI — require display)
├── compliance/     # Metadata, anomaly registry, stability tests
└── verification/   # Verification suite
    ├── hypothesis/     # Property-based tests (hypothesis)
    │   └── nodes/      # Per-node property tests
    ├── test_gradient_health.py    # Pre-existing gradient checks
    └── test_heat_analytical.py    # Pre-existing analytical benchmark
```

## pytest Configuration

Test warnings are escalated to errors by default (`filterwarnings = ["error"]` in `pyproject.toml`). Known-safe warnings are explicitly ignored. If your code produces a new warning, either fix the cause or add a documented filter to `pyproject.toml` with a comment explaining why.
