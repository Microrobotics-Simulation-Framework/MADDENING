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
from maddening.compliance import BenchmarkType, verification_benchmark

@verification_benchmark(
    benchmark_id="MADD-VER-XXX",
    description="Analytical solution comparison for ...",
    node_type="YourNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria="Max error below the stated tolerance",
    references=("AuthorYear",),
)
def test_your_node_analytical():
    """Compare YourNode output to analytical solution."""
    node = YourNode("bench", timestep=0.001)  # plus your node's own parameters
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
python scripts/check_numeric_constants.py
```

## Environment Variables

| Variable | Purpose | Typical Value |
|----------|---------|---------------|
| `JAX_PLATFORMS` | Force CPU backend (avoids GPU issues) | `cpu` |
| `XLA_FLAGS` | Disable GPU autotune (avoids equinox segfaults) | `--xla_gpu_autotune_level=0` |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD` | Prevent plugin conflicts | `1` |
| `MADDENING_HYPOTHESIS_PROFILE` | Hypothesis depth/settings profile (`dev` or `ci`) | `dev` |
| `MADDENING_TEST_SHARD` | Keep only shard `i` of `N` of the test files (`tests/_sharding.py`); CI sets it per job. Unset: every file | `2/4` |
| `MADDENING_TEST_JAX_TIMING` | Record each test's JAX trace / lower / compile / cache-read time and processes started into the JUnit XML (`tests/_jax_timing.py`), for the time budget. CI sets it | `1` |
| `MADDENING_REQUIRE_PACKAGING_TESTS` | Fail, rather than skip, the wheel test when the build backend is missing. CI's test lanes set it | `1` |
| `JAX_COMPILATION_CACHE_DIR` | JAX's persistent compilation cache directory; the test lanes point it at the restored or empty cache (see *The compilation cache* below). Unset locally: no cache | `$RUNNER_TEMP/jax-cache` |

## 4. Property-Based Testing (recommended for physics nodes)

Location: `tests/verification/hypothesis/` for the numerics suite; property
tests that belong with a package (`tests/fmi/`, `tests/core/`) stay there.
Configuration is global either way — see *Hypothesis profiles* below.

Beyond analytical benchmarks, MADDENING provides a Hypothesis-based
property-testing layer. See the full [Verification Guide](verification.md).

**Property-based testing** (hypothesis) — test universal invariants:

<!-- snippet: no-run, reason: pseudo-code: my_node and bounds={...} are placeholders -->
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

<!-- snippet: no-run, reason: external: imports the repository's tests/conftest.py, which only the test suite can -->
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

<!-- snippet: no-run, reason: fragment: lines inside a composite strategy -->
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

Slow tests still run, but less often, and not on every branch.
`slow-tests.yml` runs the whole suite, slow tests included, on a schedule
(Monday, Wednesday and Friday) that covers `main` only, because GitHub runs
a scheduled workflow on the default branch and nowhere else. A release
branch (`release/**`) gets a slow-lane run only when someone dispatches the
workflow on it by hand, from the Actions tab or with
`gh workflow run slow-tests.yml --ref <branch>`. So a slow test you add on
a release branch has not run on that branch until someone does. A property
under `tests/verification/hypothesis/` that is marked slow also still runs
on every push, at the `ci` profile's depth, in the `verify-hypothesis` job,
which selects `-m "slow or not slow"`, but that job installs jax 0.10.2
only. So on the other JAX lane the property runs only in the slow lane.
Give it an unmarked sibling at reduced depth (a few fixed draws, one
rollout length, a smaller population) that the default lane runs on both
JAX lanes, and name it at the mark as `# Per push: <node id>`. Every slow
property in that directory has one. A slow test that needs a tool the
default lane installs (valgrind, for `tests/fmi/test_c_unit.py`) needs
`slow-tests.yml` to install it too, or it skips there and runs nowhere.
Before you slow-mark a test that checks a framework property, read *What
only the slow lane checks* below.

Two more ways a test can fail to run where you expect it to:

- **A test that reads documentation.** A pull request that changes only
  documentation (`docs/`, Markdown, `plans/`, `.claude/`, `LICENSE`,
  `CITATION.cff`) runs the `compliance` job alone, and that job runs
  `tests/compliance/` and nothing else. A test that reads one of those
  paths therefore belongs in `tests/compliance/`;
  `tests/compliance/test_ci_workflows.py` fails if a test elsewhere names
  one. A push to `main` or `release/**` always runs every lane.
- **A test that needs `usd-core`.** Only the `test-usd` job installs the
  `usd` extra, so a test outside `tests/usd/` that imports `pxr` or
  `maddening.usd` skips in the sharded lanes. List its file in `test-usd`'s
  pytest command in `ci.yml`; the same compliance test checks that you did.
  Do not slow-mark it: `test-usd` runs no slow test and the slow lane has
  no `usd-core`, so it would run nowhere (the compliance test checks this
  too).

CI enforces the budget in the `Test time budget` step of each test lane
(`scripts/report_test_durations.py`, reading pytest's JUnit XML):

- The run page's summary shows the band counts, the 30 slowest tests and
  the 15 slowest files. Every test over 1 s is also listed at the end of the
  log (`--durations=0 --durations-min=1.0`). The XML is uploaded as the
  `test-durations-*` artifact and kept for 90 days.
- An unlisted test over **5 s** gets a warning annotation on the run.
- An unlisted test over **20 s** fails the job. An allowlisted one passes
  (the allowlist has no ceiling), but gets a warning annotation, so a kept
  test that regresses is still seen.
- GitHub shows at most ten annotations per step, and the script emits no
  more than that: the summary table is the complete list. The `Test
  durations` job emits the 5 s warnings again for the whole lane, so each
  one appears once on its shard and once on the lane summary.

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
every file, and the count is written in several places that must move
together:

- in `ci.yml`: `shard:` and `MADDENING_TEST_SHARD` in the `test` job, the
  `of4` in the compilation cache's key and in the durations artifact's
  name, and the `-ne 4` and "of 4" in the `Test durations` job's
  "Summarise the lane" step;
- in `slow-tests.yml`: `shard:`, `MADDENING_TEST_SHARD`, the artifact
  name's `of4` and the "of 4" in the step titles;
- `PINS_FOR` in `tests/_sharding.py`, and the pins themselves.

The compliance tests pin most of these at four, so a change to the count
fails them until each place is updated: `test_ci_sharding.py` (`shard:`,
`MADDENING_TEST_SHARD` and `PINS_FOR`, in both workflows),
`test_ci_workflows.py` and `test_report_test_durations.py` (the cache key,
and the lane summary run on four shards' reports and on three). The
artifact names and the step titles are not checked.

Every allowlist entry is `<node id> # kept: <why it must run on every
push>`. The tests that were already over 5 s when the budget arrived
(2026-09-24) have all been marked slow, made faster, or kept that way. The
summary lists entries that may now be removable: ones no longer in the lane,
or that passed under 5 s (a skipped or failed test says nothing about its
cost, so it is never listed).

### The compilation cache: warm and cold runs

The test lanes use JAX's persistent XLA compilation cache, and whether a
run reads it decides what its times mean:

| Run | Cache | Times mean |
|---|---|---|
| Pull request | **warm**: restores the base branch's cache, never saves one | Fast. A program the PR adds or changes is not in the base cache, so it still compiles from scratch and a new slow test is caught on the PR that adds it. A cost that *moves* is not: see below. |
| Pull request whose diff adds or removes a slow mark, removes a test function, or edits `tests/duration_allowlist.txt` or `tests/_sharding.py` | **cold**, not saved | Accurate for the tests the change moved a compile onto (see below). |
| Push to `main` / `release/**` (after a merge) | **cold**: starts empty, saves the result for the next PRs | Accurate: each program compiles in full the first time the run needs it. A later test that needs the same program reads it back from the cache the run is writing (17-25% of lookups on CI), so it can look fast because an earlier test paid. |
| `slow-tests.yml`: scheduled Mon/Wed/Fri on `main` only; on a release branch only when dispatched by hand | **off**, one process per shard | A cold, uncontended timing of the whole suite, for the commit it ran on. Triage and allowlist edits are based on these runs, so check that commit: on a release branch the last run is the last dispatch, which may be well behind the tip. |
| Pull request with `[cold-ci]` in its head commit message | **cold**, not saved | For before/after numbers while optimising tests. The job reads the message from its checkout; if it cannot, it runs cold and says so in a warning. |

Pull requests never save, so a second push cannot read the first push's
cache: a new test that takes 25 s cold would otherwise pass at 8 s.

**Why some pull requests run cold.** The first test to compile a shared
program pays for every later test that reuses it. When a pull request
slow-marks, deletes or moves that test, the next one inherits the compile.
On a warm run it reads the program from the base branch's cache and looks
fast; only the cold run after the merge sees what it now costs, and fails
it (commit `04cad05` found one such case by hand). So the `changes` job asks for
a cold run when the diff adds or removes `mark.slow`, removes a
`def test...` line (a test deleted, renamed or moved), or edits the
allowlist or the shard assignment. What it does not see, so a pull request
that does one of these runs warm and the cold run after the merge is the
first to see what the change moved:

- a new test placed before an existing one that shares its program, which
  reads the program from the base cache on the pull request and pays for
  it after the merge;
- slow marks applied by a helper or a hook rather than written as
  `mark.slow`;
- a parametrize case removed (a value deleted from the list, with no
  `def test` line removed);
- a test that stops running without being deleted: an added `skip`,
  `skipif`, `xfail(run=False)` or `importorskip`.

Closing the last two is planned for 0.5.0. Until then, put `[cold-ci]` in
the head commit of a pull request that does either to a test sharing a
compiled program with the tests after it.

Each shard has its own cache, which works because shard *i* holds the same
files on every branch (see *Sharded lanes*). XLA:CPU compiles for the
host's instruction set, and what keeps an executable off a host it was not
built for is JAX's own cache key, which includes the CPU's features. The
workflow's key also includes the runner's CPU model, but only to raise the
hit rate: runners reporting the same model can still differ in features,
and a shard on one of those restores a cache and then misses almost every
lookup. So the summary labels each shard from the hits and misses its
report records: **warm** only when it restored a cache *and* more than half
its lookups hit, otherwise **restored but unused (cold)**. It prints every
shard's label and hit rate, and the lane as `warm`, `cold`,
`restored but unused (cold)`, or `mixed` when the shards differ or a shard
recorded no mode. A run in which any shard restored a cache never lists
allowlist entries as removable, because a test that is only fast when its
compile is cached is still slow.

`slow-tests.yml` runs with no cache. If its report records cache lookups in
more than one file, the summary warns: a test switched a persistent cache
on in-process and left it on, and the times after it are partly warm. A
test that turns a cache on (`maddening.core.simulation.compile_cache.enable`
or `warm_cache`) must restore the JAX settings and reset JAX's cache
object afterwards, as `tests/core/test_compile_cache.py` does.

A cache is saved only after a push run whose test step passed, and only
once `scripts/prune_jax_cache.py` has deleted every entry that does not
decompress whole. JAX writes an entry in place and never rewrites one, and
a test's subprocess (which inherits the cache directory) can be killed
mid-write; a truncated entry, once saved, would make every pull request
that restores it raise (JAX warns, and `filterwarnings = ["error"]` turns
that into a test error).

A shard whose report holds nothing but collection errors (a test file that
fails to import stops pytest before any test runs) fails the budget step
with exit 2: no test ran, so there is nothing to pass. The lane summary
leaves such a report out and lists no allowlist entry as removable.

### Why a test is slow

Each lane records every test's JAX tracing, lowering, XLA compile and
cache-read time, and how many processes it started (`tests/_jax_timing.py`,
switched on by `MADDENING_TEST_JAX_TIMING=1`). The summary splits the
slowest tests into:

- **compiling**: XLA backend compilation. This is the only part a cache
  removes. JAX's compile timer wraps the whole compile-or-read-the-cache
  call, so on a cache hit what it times *is* the cache read; the read is
  subtracted from it and shown on its own.
- **cache read**: reading compiled programs back from the persistent
  cache. A warm run pays it; a cold run does not.
- **tracing/lowering**: building the program in Python. No cache removes
  it.
- **running**: executing, Python overhead, I/O, and anything a subprocess
  does. An un-jitted `for` loop over `node.update` lands here.

JAX's timers can overlap: on jax 0.10.2 one test's tracing alone summed to
more than its wall time. A share is therefore capped at 100%, and the
diagnosis says when the parts add up to more than the whole.

**Slow even with a warm cache** lists every test still over 5 s once its
compile time (not its cache reads) is subtracted. Every run shows this
list, cold or warm. A cache cannot fix these tests; they need a code change
or `@pytest.mark.slow`.

**Work in a subprocess (not measured here)** lists the tests that would
otherwise be on that list but started a process. For a child that runs
JAX, the child's tracing and compiling never reach the pytest process, and
the child inherits the cache directory, so a warm cache may still speed it
up. Measured: a test whose JAX work runs in a child recorded no compile at
all, and a warm run took it to about half its cold time. Compare a warm
run before treating one as slow. The list does not know what a child
runs, though: a test whose child is a C compiler, valgrind or a fuzzer is
listed here too, although no cache can help it. Telling the two apart is
planned for 0.5.0; until then, read the test before concluding a warm
cache would help. Processes are counted from Python's audit events (`subprocess.Popen`,
`os.system`, `os.fork`, `os.posix_spawn`), which miss `multiprocessing`
children started with the `spawn` or `forkserver` method. A report written
before the count existed cannot tell, so there a test with no JAX activity
at all in the pytest process is listed here.

The usual fixes:

- **Running**: move the loop into JAX. Use `jax.lax.fori_loop` or
  `jax.lax.scan` over a jitted update instead of a Python `for` loop that
  dispatches every step op by op. Measured 2026-09-24:
  `test_heart_pump.py::test_steady_state_pressure` (16,660 eager steps) and
  `test_rigid_body.py::test_quaternion_stays_normalized` (10,000) spent
  over 99% of their time this way. Both now step in a `fori_loop` and take
  under a second, with the same results (the heart pump's mean pressure
  is bit-identical).
- **Tracing/lowering**: build the graph or function once, in a
  module-scoped fixture, and reuse it rather than rebuilding it per test
  or per example.
- **Compiling**: in a property test, keep array shapes and static arguments
  fixed across examples. A new shape or a new Python-level constant is a
  new program, so draw values rather than shapes and pass them as traced
  arguments. Use the `EXAMPLES_COSTLY` tier (see *Depth tiers* above) for
  properties that still compile per example.

### What only the slow lane checks

A slow mark takes a property off every push. On a release branch it then
runs only when someone dispatches `slow-tests.yml`, so a regression can sit
unseen until then. So write the reason at the mark: what the test costs on
CI and what makes it cost that. For a test of a framework property (the
coupling solvers and their reports, gradients, the params pytree,
sharding, checkpoints, the CI gates), say one of two things there as well:

- **where the property is still checked on every push.** Usually this is a
  cheaper witness: a smaller graph, one step instead of eight, one check of
  a battery, a closed-form reference instead of a converged one. Name it in
  a comment among the comments and decorators directly above the test's
  `def` (or above its class, or the module's `pytestmark`, if the mark is
  written there), as `# Per push: <node id>`: the full id,
  `tests/<file>.py::[<Class>::]<test>[<params>]`, one or more, with any
  prose after them on the same or the following comment lines. A test
  named without its parameters stands for its parametrisations that are
  not slow-marked. For example, the adjoint of a coupling group with a
  sharded member is slow-marked in
  `tests/cloud/multigpu/test_coupling_group_with_sharded_and_replicated_members.py`,
  and its comment names
  `tests/cloud/multigpu/test_coupling_group_adjoint_through_a_sharded_member.py::test_reverse_mode_ad_through_a_coupling_group_with_a_sharded_member`,
  which differentiates one step of the same group on every push;
- **or that it is slow-only on purpose**, in which case the property
  belongs in the table below with the reason.

`tests/compliance/test_slow_only_rule.py` checks this rule on every push
for every slow-marked test under `tests/core`, `tests/property`,
`tests/verification`, `tests/cloud/multigpu`, `tests/compliance`,
`tests/api` and `tests/fmi`, using pytest's own collection (so a mark
applied through a parameter list or a class counts). It fails when a
slow test has neither, when a named witness is not a test the default
lane runs (it is slow-marked itself, misspelt, or named without its
class), and when a table row or an exemption names no slow test. Files
under those paths that check a node rather than the framework (the LBM
and wavelet convergence studies, the `run_pod` session harness, two
timing benchmarks) are listed in its `NOT_FRAMEWORK`, each with the
reason. Under `tests/verification/hypothesis/` the table is no way out:
those properties need a per-push sibling (see *Test time budget*).

These framework properties are checked only in the slow lane, on purpose:

| Property | Slow-marked tests | Why nothing checks it on every push |
|---|---|---|
| Second-order differentiation (a Hessian) through the IFT coupling step | `tests/core/test_coupling_ift_forward_mode.py::test_hessian_through_ift_step` | A second-order transform of the IFT step, compiled for both solvers: 5-10 s on CI, nearly all compile. First-order forward and reverse mode through the same step run on every push. |
| The coupling fixture registry at full width: every registered fixture builds and steps; 19 of the 24 recorded baseline rows; 22 of the 24 cells of the multi-rate solver-equivalence class | `tests/core/test_coupling_fixture_invariants.py::test_every_registered_fixture_builds_and_steps` and `::test_every_recorded_fixture_still_measures_its_baseline_row`; `tests/core/test_coupling_solver_equivalence.py::test_the_multirate_solvers_agree_across_the_configuration_class` | Each fixture, row or cell is a graph of its own to build and compile: 4-10 s each on CI, and 34-50 s for the registry. The fixtures the per-push tests use are built on every push anyway. Five baseline rows, at least one from each recorded file, and the two cells in the counterexample's own configuration stay on every push. |
| The FMU C wrapper under valgrind's memcheck, and the libFuzzer campaign | `tests/fmi/test_c_unit.py::test_c_unit_tests_under_valgrind`, `::test_libfuzzer_short_campaign` and `::test_fuzz_long_run` | The tools' own run time: 22-28 s each on CI for the first two. On every push the wrapper's unit tests and a short fuzz run pass built plain and under ASan/UBSan, and the fuzz binary runs under valgrind. |
| The compile-count gate end to end: the baseline regenerated by the one command the gate prints, and the gate pinning its own device count under inherited `XLA_FLAGS` | `tests/core/test_compile_counts.py::test_regenerating_the_baseline_is_the_documented_one_command` and `::test_the_gate_measures_its_own_device_count_not_the_ambient_one` | Each runs the gate twice in a subprocess: 7-11 s on CI. The gate itself (`scripts/compile_counts.py --check` against the committed baseline) runs on every push, and so do its refusals, on synthetic documents. |
| The CI guards fail when what they guard is broken: a seeded fault in a workflow, the budget script, the shard split, the conftest, `pyproject.toml`, the timing plugin, the cache pruner or the allowlist is caught | `tests/compliance/test_guard_mutations.py::test_every_seeded_ci_fault_fails_a_guard` | Each mutant re-runs its guard test files in a fresh copy of the tree, so the whole table takes minutes. Per push, `test_every_mutant_anchor_matches_the_tree_exactly_once` checks that every mutant still applies to the tree; see *Guard mutations* below. |
| The deprecated `calibrate()` and `tune_coupling_params()` | `tests/core/test_calibrate.py::TestCalibratePhysics`, `tests/core/test_calibration.py::TestTuneCouplingParams` and `tests/property/test_sysid_contract.py::TestTuneCouplingParams` | Hundreds of eager gradient steps (30-165 s on CI) and a compiled graph per grid point (up to 15 s each), over APIs deprecated for removal in 0.5.0 in favour of `maddening.sysid.fit`, which is tested on every push. Their deprecation warnings are checked on every push. |
| The coupling benchmark script runs end to end and writes its JSON | `tests/core/test_cross_feature_edge_cases.py::test_bench_coupling_script_runs_on_springs` | A subprocess that imports the library and times a coupled and an uncoupled spring graph: over 5 s on CI. `benchmarks/bench_coupling.py` is a measurement tool, not library code; the coupling step it times is tested on every push. |

To add a row, say what the property is, which tests check it, and why no
cheaper test can. If a cheaper test can, write it instead.

### Guard mutations

A guard that still passes when the workflow, script or config it reads is
broken protects nothing. `tests/compliance/test_guard_mutations.py` checks
the CI guards mechanically. Each entry in its `MUTANTS` table is one seeded
fault: an id (a group letter and the next free number), the file, an
`anchor` that occurs exactly once in it, the `replacement`, the guard tests
to run, and what the fault would let through in CI. The slow-marked
`test_every_seeded_ci_fault_fails_a_guard` applies each fault to a
`git archive HEAD` copy in a temporary directory, never the working tree,
and requires at least one guard test to fail. Per push,
`test_every_mutant_anchor_matches_the_tree_exactly_once` fails if an anchor
is missing or ambiguous, so a refactor cannot quietly turn a mutant into a
no-op. When you add or change a guard, add one entry for the fault it
exists to catch, commit, and run that case with
`pytest "tests/compliance/test_guard_mutations.py::test_every_seeded_ci_fault_fails_a_guard[<id>]" -m "slow or not slow"`
(it reads `HEAD`, not your working tree). If you touch an anchor, update it
to the new spelling and keep the same fault. Two kinds of entry are
exceptions, and each needs its reason. A mutant that changes nothing CI
relies on gets `equivalent="<why>"` and is asserted to survive. A live gap
gets `gap="<what is unguarded>"` and runs as a strict xfail until a guard
catches it.

## The claims inventories

Each `docs/validation/*_claims.yaml` lists the documented claims of one
audit area that a user could rely on:

| file | prefixes | covers |
|------|----------|--------|
| `coupling_claims.yaml` | `CPL` | coupling groups, their solvers, schedules and accelerations; sub-cycling, multi-rate groups and the adaptive steppers on coupled graphs; `coupling_diagnostics()`, `coupling_report()` and `strict_convergence`; the precision floor and every bound and `*_usable` flag; IFT gradients; the profiler's coupling statistics; `windowed_loss`'s convergence mask |
| `sysid_fmu_claims.yaml` | `SYS`, `FMU` | `maddening.sysid` (`windowed_loss`, `fim`, `fim_core`, the three fitters and their results, truth recovery and units); `ParamSpec` bounds and transforms; `node.params` and `gm.params` as fitting and export see them; the FMI 3.0 export: model description, TCP bridge, sidecar, FMU state, C wrapper, conformance, refusals, timeouts, tokens and the terminated state |
| `rest_runpod_claims.yaml` | `REST`, `RPD` | the HTTP API: authentication, the `Host` and `Origin` rules, request bounds and budgets, the graph lock and its 409/503, `/graph/*`, `/sim/*`, `/checkpoint/*`, shutdown and the runner behind `/sim/start` (not the experimental surrogate and streaming endpoints); `benchmarks/multigpu/run_pod.py`, its runbook, records, verdicts and exit codes |

Each row records:

- the claim's wording and where it is stated;
- the domain the documentation states, with any part it leaves unstated
  called out;
- the oracle: how truth is known (an exact float64 fixed point, a dense
  solve, finite differences, bit-identity, a closed form, the graph the
  FMU was exported from, or what an earlier release wrote);
- the pytest node ids that fail if the claim stops holding;
- a status: `verified`, `failing`, `untested` or `ambiguous`.

An audit attacks the lists rather than hunting open-ended, and a fix PR
re-classifies the rows it touches.

To add a claim, add a row in the same change as the sentence, in the file
whose area it belongs to. Give it the next free id under that file's
prefix and cite a test that can fail at the edge of the claim's
conditions, not in their comfortable middle: units on every axis, a value
on and just past a bound, every transform, float32 and float64, multi-rate
and sub-cycled graphs, each surface that serves the same model. If the
tree does not meet the claim, the test is
`@pytest.mark.xfail(strict=True, raises=..., reason="<ID>: <one line>; pending fix")`
and the row is `failing`, with a `finding` that gives the reproducer and a
file:line guess. If the documentation leaves the domain unstated, or two
documents disagree, the row is `ambiguous`: it carries the
`proposed_wording` its tests support and leaves the docs to the fix PR.

A new inventory needs only its file: `schema_version: 1`, a `prefixes`
list of the id prefixes it owns (two to six capital letters, owned by no
other file) and its `claims`.
`tests/compliance/test_claims_inventories.py` finds every
`docs/validation/*_claims.yaml` and checks five rules, mutation-tested by
self-tests in the same module:

- every file is well-formed, and no two files own a prefix;
- every row is well-formed, with an id under one of its file's prefixes;
- every cited test is collected and runs on every push, or is slow-marked
  with a `# Per push:` witness;
- every `failing` row cites a strict xfail that names it;
- no `verified` row cites an xfail, and every xfail that names a row is
  strict, cited by that row, and names a row some inventory holds.

`tests/property/test_coupling_invariances.py` holds three metamorphic rows:

- renaming every node changes nothing;
- the build order changes nothing (the members' relative order excepted:
  Gauss-Seidel follows it on purpose);
- an identity relay on an internal edge moves the fixed point only by
  rounding.

## Numeric constants carry their units

`scripts/check_numeric_constants.py` (CI's compliance job) scans
`src/maddening/core/`, `src/maddening/sysid.py` and
`src/maddening/cloud/multigpu/` for small float literals (`|v| <= 1e-3`, or a
decimal exponent of `-4` or below) and for `finfo(...).tiny` / `.eps` used
additively or as a floor (`x + eps`, `max(x, tiny)`).  Each must carry an
inline `# units: <what it is relative to>` comment or a counted line in
`scripts/numeric_constants_allowlist.txt`.  A stale or miscounted line fails
the gate, as does a scope that no longer exists.

Four 0.4.0 defects were one mistake: an absolute constant inside a
relative computation.  The IFT solve's `atol=1e-8` returned gradients of
exactly zero for small-unit states, IQN and `fit_lm` had `1e-12` floors, and
the accelerators' steps flushed at small magnitudes.  The gate found four more
that had shipped: the public Krylov solvers' `atol=1e-8`, the multi-rate
GCD's `1e-9`, Adam's `eps`, and the adaptive error norm's `1e-300`.  So when you
add a small constant to the numerical core:

- **Write the justification as a ratio.**  "Dimensionless, a fraction of
  `max|b|`" or "dtype range: the smallest normal number" is one; "small" or
  "avoids division by zero" is not.  If you cannot say what the number is
  relative to, it is absolute in some quantity's units, and the fix is to
  make it relative, not to allowlist it.
- **Pin the fix with a scale test.**  The test that catches this class runs
  the same computation at two power-of-two scales and asserts bit-identity
  (`tests/core/test_linear_solvers_in_any_units.py`,
  `test_sysid_adam_in_any_units.py`, `test_multirate_in_any_units.py`).
  A decimal scale changes the inputs' rounding; a power of two does not.
- **Compare coordinates only where no units move them.**  A cutoff relative
  to the largest eigenvalue, a norm over several parameters or a projection
  is a constant in disguise when its coordinates carry units: the fitters'
  identifiability guard held a determined parameter written in units of
  `1e-5` until it made its tests relative to each parameter's size
  (`tests/core/test_sysid_hold_in_any_units.py` runs it over decades).
- **Compare a value against a bound on the host.**  XLA's CPU backend
  flushes subnormal operands, so a `jnp` comparison passed float32 `-1e-40`
  as `>= 0` (`tests/core/test_bounds_checks_compare_exactly.py`).
- **Build a power-of-two frame with `maddening.core._pow2_frame`.**  It is
  the one place the coupling runtime, the solvers and the fitters build
  their frames.  `tests/core/test_pow2_frame.py` refuses a `frexp` or
  `ldexp` anywhere else in the scanned scope, and a frame that is not an
  exact power of two breaks the coupling bit-identity claims.

## Differential tests

A differential test compares two paths through the library that must agree,
over generated inputs, so a disagreement is found by CPU rather than by an
audit. Each harness writes tests only: a disagreement it finds is pinned as
`@pytest.mark.xfail(strict=True, reason="differential: ...; pending fix")`,
so CI stays green and the fix must flip it. Each subsection below is owned
by one harness.

### State and I/O

Files: `tests/property/test_differential_{rest_params,param_writes,checkpoint,serialisation,fmu}.py`,
`tests/usd/test_usd_differential_round_trip.py`,
`tests/cloud/multigpu/test_property_sharded_state_io_differential.py`.
Shared scaffolding: `tests/property/node_catalogue.py` (every built-in
node, with valid constructions and writes in named categories) and
`tests/property/differential.py` (exact NaN-safe comparison, and a guard
that points `HOME` at an empty directory and makes every cloud launcher
raise). Every comparison is bit for bit: dtype, shape and bytes. No oracle
takes a tolerance, because both sides run the same compiled computation.

| Oracle | Paths compared | Covers | Cannot see |
|---|---|---|---|
| REST write is a reload | `PUT /graph/params/{node}` then run, against `to_dict` -> JSON -> `from_dict` with a checkpoint loaded, then run; and after `POST /sim/reset` against `reset_state()` on the reload. A 4xx must leave every node's params, every `gm.params` leaf, the whole state and the dirty flag unchanged; `GET` must equal what the step reads and what a save carries | Every cheap built-in node per push, with valid, boundary, out-of-bounds, non-finite, oversized, wrong-type, unknown, initial-condition, structural and cross-parameter writes; a coupled member with a `ParamSpec` override; the LBM, pipe and wavelet nodes and generated multi-node graphs in the slow lane; a sharded rod (re-wrapped reload) | A defect the constructor and the live step share; a write the catalogue does not propose |
| `gm.params` / fit write is a reload | A write into `gm.params` (leaf by leaf, or the whole tree as `gm.params = fit(...).params`), run, against the saved config plus a checkpoint, run; and after `reset_state()` on both. A refused write must be refused by the next run and by `to_dict()` alike | In-bounds writes into every cheap node; the output of `maddening.sysid.fit`; calibration-like writes into generated graphs (slow); a sharded rod | Out-of-bounds and non-finite values (a graph does not check `gm.params` writes against `ParamSpec`; the REST route and the FMU do) |
| Checkpoint resumes the run | `save_state` at step *k*, `load_state` into a fresh graph or one that ran ahead, then *N* steps, against the run that never stopped; the restored state is compared before stepping, `_meta` included; `reset_state()` must equal a freshly compiled graph | Every cheap node with calibrated params; predictor history, IQN-IMVJ warm starts, sub-cycling and multi-rate phase; the REST `/checkpoint/save` and `/load` routes crossed with the in-process calls; generated graphs, the costly nodes and a diagnostics group in the slow lane; a sharded rod | State outside `gm._state` and `gm.params`; a defect `save_state` and `load_state` share (one key scheme) |
| Serialisation is the original | `to_dict` -> JSON -> `from_dict`, and a USD stage, against the original: config, structure, initial state, params, `param_specs()` and trajectory | Every built-in node with generated arguments; generated graphs over seven node kinds (`strategies.ALL_NODE_KINDS`), with coupling groups in the slow lane; a sharded rod reloads unsharded with a warning and, re-wrapped, bit for bit | A field both `to_dict` and `from_dict` ignore that changes nothing they compare |
| Three FMU paths are one model | The TCP bridge over a socket, `FmuSidecar` driven in process, and the `GraphManager` itself, over generated `set` / `get` / `step` / `get_state` / `set_state` / `reset` sequences: every value and the full state agree after every operation; a parameter refused by the bridge's `set` is refused by `set_params` (same reason) and by `check_params`; an edited snapshot is refused by `set_state` and `set_fmu_state` with the same message and changes nothing | A uniform graph, a multi-rate graph with an array input, a coupled graph with a predictor per push; generated graphs in the slow lane | A defect in the compiled step (all three share it); input values, which only the bridge checks |

Per push, each property stays under about 5 s on a CI runner by drawing
values on a fixed graph shape (`EXAMPLES_COSTLY`), reusing compiled graphs
where the oracle checks the reuse itself (a reset graph must equal a fresh
one), or by a short fixed table where every case compiles. Every slow mark
names its per-push sibling. Every property is `derandomize=True`, so CI
draws the same examples on every run. Each oracle was mutation-tested on a
scratch copy of `src/` (the PR that added them lists the mutants and the
test that caught each).

### Coupling and numerics

`tests/property/test_differential_*.py`, over graphs of synthetic nodes
from `tests/property/coupled_graphs.py`. Every node is `x <- alpha x_pre +
sum G_j f(u_j) + b + beta dt` (or explicit Euler on the same right-hand
side). The gains are drawn and rescaled to a drawn spectral radius (up to
0.999, non-normal or rank one). Nodes can have `tanh` inputs, `int32`,
`uint32` and `bool` leaves, an unread float field, a clock, and flux
edges. A graph has a cycle, chords, a driver and a sink. A group of
linear nodes has its fixed point as a float64 solve. The per-push cases
fix the structure and configuration and draw the values, which reach the
compiled step as parameters, so each case compiles once. Each slow-marked
broad case draws the structure and configuration too, and names its
per-push sibling.

Two programs that evaluate the same arithmetic with different rounding
(the two solvers' loops, a fused against an unrolled update, batched
against unbatched kernels, eager against jitted) are held to *round-off
per pass*: the forward error of each node's sum, `2 T eps sum |term|`,
per coupling pass (`rounding_bound` in the kit). That is at least eight
ulps of `|x|`, and more where large terms cancel, which a non-normal gain
does.

| Oracle | Paths | Tolerance | Cannot see |
|---|---|---|---|
| fori == ift | `solver="fori"` against `"ift"`: every acceleration, norm, mode, predictor, cap, flux edges, `tanh` | `iterations` and `converged` are equal when the threshold clears the norm's float32 floor (`residual_noise_floor`) by 4x, unless an estimate lies within its own rounding of the threshold (`criterion_is_resolved`). With equal passes, states agree to round-off per pass | The one-pass map, the norms and the criterion arithmetic, which both solvers share |
| diagnostics on == off | the same group with `diagnostics` False and True, both solvers | Bitwise: states, `iterations`, `converged`, the predictor and IMVJ warm starts, and `jax.grad` through `run_scan` | A fault that moves both settings the same way |
| converged => near the exact fixed point | the returned state against the float64 solve, measured in a NumPy restatement of the group's norm | Single coupling mode, iterate started on it, `none` or `fixed`: distance <= threshold + `floor * amp` + `2 omega floor amp**2`. That is the estimate's own float32 resolution, derived from its formula; at a rate of 0.99 it admits that float32 cannot certify the distance. `precision_limited` and `ratio_usable=False` groups are outside the claim, as documented. A usable `spectral_error_bound` must be >= the distance, with no slack, on general non-normal spectra under every acceleration | Non-linear groups (no closed form); the definition of "fixed point of the evaluated map" |
| multi-rate == hand-unrolled | the gated multi-rate step against a Python loop that fires each block or leaves it alone, with the group as a uniform-rate sub-graph | Bitwise: a non-firing block's state and `_meta` are unchanged. States agree to round-off per pass; `iterations` are equal | A fault inside the coupled solve that both paths call |
| sub-cycled == uniform-rate | `subcycling=True` with constant interpolation, against a hand-written node that sub-steps `d` times in a `lax.scan` at the macro timestep | Equal passes; round-off per pass. Clocks are bitwise equal to the reference's, and exact against the graph's time when `d` is a power of two. `quadratic` == `linear` bitwise (MADD-ANO-027). `linear` == `constant` to round-off under Jacobi or with the sub-cycled node first | The shared sub-step update |
| adaptive at a pinned dt == run_scan | `run_adaptive`/`run_adaptive_scan` with `dt_min = dt_max`, against `run_scan` at half the timestep; replaying `dt_history` through the dt step | Round-off per pass. The replay is bitwise. Every node's clock equals the stepper's time (MADD-ANO-061), within the rounding of its float32 sum | The dt-parameterised step itself |
| vmap == per-member, jit == eager | `run_sweep` against `run_scan` per member; `jax.vmap` of the step against the step; `vmap(grad)` against `grad`; `jax.disable_jit()` against `jit` | States to round-off per pass; `iterations` equal; the residual to the norm's float32 floor; `rho_spectral` to its Arnoldi residual + `sqrt(8 eps)` (a nearly defective non-normal Jacobian moves its eigenvalue by the square root of a perturbation); `gradient_relative_error_bound` within 25% (its ~8% is documented); gradients within 1e-4 relative | Every fault in the step, which is the same batched and unbatched |
| int/uint32/bool leaves survive | each non-float leaf after `k` updates, against its closed form, in every configuration above and under sub-cycling, waveform sweeps, multi-rate, adaptive stepping and `vmap` | Bitwise, dtype included | A leaf that reads a coupled input (the solver oracle covers that) |
| units-invariance | the same equivariant group (no `beta * dt`, no leaves, linear) with every bias and initial state times `2**k`, `k` in -53..66, against the unscaled run (`test_differential_coupling_units.py`), every acceleration, norm, mode and solver | Bitwise: passes, verdict, residual, amplification, state / `2**k`. Powers of two only: a decimal scale changes the inputs' rounding, which Aitken is sensitive to | A constant that happens to be a power of two |
| long Gauss-Seidel chains at the floor | a stalled ring of 8 (per push, both modes) or 24-32 relays (slow) against its float64 fixed point; drawn rank-one gains **and** the worst-case uniform ring (every link `rho**(1/m)`, no cancellation), which is what fails a floor that undercounts the chain | A usable `spectral_error_bound` >= the distance, no slack | Chains shorter than the floor's headroom (per push) |
| adversarial chains | Gauss-Seidel rings closed by an affine head: links with relative gain above one and no cancellation (`u**2`, `u**4`, `u0 * u1`), and signed affine links whose terms cancel (`|a u|` up to 150 `|x|`), kept as separate cases, under Gauss-Seidel (under Jacobi their noise keeps the residual above the floor); the exact fixed point by float64 Newton on the head's loop map, started at the designed root | A usable `spectral_error_bound` >= the distance; a usable `gradient_relative_error_bound` >= the true relative error of the gradient in the head's bias (slow) | Cancellation inside a node's own update, which nothing outside it can see |
| IFT derivatives at every scale | `jax.jacfwd` and `jax.grad` of a step under `linear_solver="gmres"` against `"dense"` (an LU solve, no tolerance) and against `solver="fori"` (unrolled), the group at `2**k`, `k` in -100..60, 6 and 60 coupled DOF, both modes, the draw rescaled so its sweep contracts (`test_differential_ift_derivatives.py`), jitted once through `gm._compiled_step`; `jacfwd` in a gain multiplier so its rhs scales with the state | Relative 1e-3 | A defect all three solves share (the one-pass map's JVP) |

Each oracle was mutation-tested against a scratch copy of `src/` with at
least one seeded fault, and each fault was caught. The PR that added the
harness lists them. Random draws have twice let a floor defect through
(rank-one gains are below one per link almost surely): an oracle whose
claim rests on a model's premise needs a generator that attacks the
premise, not only one that samples around it.

### Configuration interactions and graph shapes

Files: `tests/property/test_differential_coupling_interactions.py` and
`test_differential_coupling_topologies.py`, with the scaffolding in
`covering_array.py` and `coupled_topologies.py`.  The coupling-and-numerics
harness above draws one group in a cycle, and its tables hold one knob at
a time.  This harness draws *combinations* of knobs and *shapes* of graph.

**The covering array** (`covering_array.py`) is a deterministic IPOG
greedy with constraints.  The coupling knob space is `solver`,
`iteration_mode`, `acceleration`, `convergence_norm`, `linear_solver`,
`predictor`, `subcycling`, `boundary_interpolation`, `waveform_iterations`,
`diagnostics`, `strict_convergence`, the dtype and a budget.  Over it the
generator gives 65 rows, in which every valid pair and triple of values
appears.  `test_the_array_covers_every_valid_triple` regenerates the array
and checks its coverage, and dropping any one row fails the check.  The
constraints are the inert-knob rules: under `"fori"` `linear_solver` and
`strict_convergence` do nothing, and without sub-cycling neither do the
two waveform knobs.  `"bicgstab"` is not an option (CPL-039).  Per push
the harness runs a one-way slice of the array, the float32 rows that hold
every value of every knob.  The slow lane runs every row, and the float64
rows in subprocesses under `jax_enable_x64`.

**The topology generator** (`coupled_topologies.py`) describes whole
graphs of linear relays:

- several groups: rings, stars and nested cycles;
- outside nodes inside a group's component, two groups in one component,
  and ungrouped cycles;
- drivers and readers in any build order;
- additive edges (three into one port), flux edges, mapped edges between
  sizes and transformed edges;
- `int32`, `uint32`, `bool` and typed PRNG-key leaves, and three-argument
  nodes.

Per push the harness runs four named structures, each in its own build
order and in an *interleaved* one: readers added before the cycle they
read, and outside nodes added between a group's members.  The slow lane
draws the structure.

**The monolithic reference** is `LinearModel`.  One step of linear relays
is one linear system: each edge reads its source's new value, except a
back edge, which reads the old one.  The back edges come from the
documented rule, restated rather than called: an edge between two
components points forward; inside a component the nodes run in their
build order, each group as one block at its first member's place.  The
system is solved in 80-bit extended precision, refined from a float64
solve, so a float64 graph's rounding is about 2000 reference ulps.  The
oracle is local:

- a node outside every group is within its float rounding of its update
  at the values it read;
- a group's defect `x - Phi(x)` is within what its *reported* residual
  allows, `-(I - L)(F(x) - x) + epsilon` for an affine pass, and the
  reported residual is `||F(x) - x||` of the returned state to its
  rounding;
- a converged group is within its threshold's tolerance of its exact
  fixed point;
- the whole state is within `|(I - M)^{-1}|` of those allowances of the
  exact solve.

The same structures carry renaming, build-order and identity-relay
invariances.  The build order keeps the relative orders the documentation
says reach the result: a group's members, every node of a component that
is not exactly one group, and three or more additive edges into one port.

| Oracle | Paths | Tolerance | Cannot see |
|---|---|---|---|
| monolithic reference | the step against the extended-precision solve of the whole graph, per step from the library's own pre-step state, for every array row and every structure | A node's float rounding, `T eps sum|term|`. `T` counts the rounding chain: the constants, and per port the product, the additive sum and a mapping's inner product. `T eps` is twice `T u`, which covers `1 / (1 - T u)` and any reassociation or FMA. A group adds `||D S (I - L) S^+ diag(rho_up)||` times its reported residual, and its float evaluation `(N + 4) eps` | Non-linear nodes. Multi-rate and adaptive stepping. A fault the restated schedule shares with the documentation |
| fori == ift | the array row against its solver twin, in lock step: before each step the twin takes the row's whole state, node states and the shared warm-start slots, so each step compares one solve from one input | The coupling-and-numerics oracle's parity rule, at every step; with equal passes, round-off of that step's passes | As above |
| diagnostics on == off | the row against its diagnostics twin | Bitwise: states, `_meta` slots, passes, verdicts | A fault that moves both settings the same way |
| strict == report | the row with `strict_convergence` against without | It raises on exactly the steps reported unconverged. With `waveform_iterations > 1` it also raises where an earlier sweep hit the cap (CPL-052). Where it does not raise, bitwise | A raise that happens to fall on a step that is also unconverged |
| usable bounds | `spectral_error_bound` against the distance in the returned state's weights; `gradient_relative_error_bound` against central differences of the exact fixed point in every member's gains and biases (slow) | None: the bound must be at least the truth | Constants outside the group |

The harness found MADD-ANO-154 to 157, which are strict xfails naming
CPL-003, CPL-185 and CPL-186.  Each oracle was mutation-tested against a
scratch copy of `src/` with a seeded fault, and the PR that added the
harness lists them:

- back edges decided over the node order rather than the block order;
- readers downstream of a cycle scheduled in build order;
- an understated residual;
- the ift loop returning the iterate after the measured one;
- Aitken dropping its relaxation carry (caught in the slow lane, by the
  rows whose Aitken runs past three passes);
- IQN-IMVJ ignoring `jacobian_reuse` under `"ift"`;
- the fori loop not freezing the converged iterate;
- diagnostics nudging the state;
- `strict_convergence` that never raises;
- a spectral bound reported at a tenth of its value, and a gradient bound
  at a hundredth (caught in the slow lane);
- a group swept in its members' name order (caught by renaming).

### System identification and the params machinery

Two oracles; each was mutation-tested against a scratch copy of `src/`
(the PR that added them lists the faults).

| Oracle | Paths compared | Covers | Cannot see |
|---|---|---|---|
| Truth recovery (`tests/property/test_sysid_truth_recovery.py`) | A fit's `converged=True` against the truth it was generated from: noiseless data, a truth drawn anywhere inside the bounds (near them too), a start anywhere in the box. The fit must recover the truth to 1e-3 of the coordinate's range, or say `converged=False`; and the same fit with one parameter measured in a unit up to 1e6 times larger or smaller must give the same answer -- the same returned point, `excited_rank` and `hold_declined` -- in about as many iterations | `fit_lm` per push on a closed-form problem whose only constrained stationary point is the truth (one strictly monotone, non-saturating block of residuals per coordinate, nonlinear so that a Gauss-Newton step overshoots), over every transform: a clipped `transform=None` leaf, `log`, `logit`. In the slow lane, more draws, `fit` (Adam with a `tol`), and all three fitters on the spring graph | A problem with a second stationary point (a plateau where the model saturates is one, and a fit may stop there); the size of a fit's error when it says `converged=False` |
| Entry points and readers agree after a `node.params` write (`tests/property/test_differential_entry_points_after_a_node_write.py`) | `gm.step`, `gm.run`, `gm.run_scan` at a length traced before the write and at a new one, `gm.run_scan_with_history` and `sysid.windowed_loss`, each from one fixed state, after a write with no compile and again after `compile()`: all must run the model the write leaves. Readers likewise, with nothing run first: `gm.params`, a `jax.jit` and a `jax.grad` of a sysid loss handed `gm.params`, `to_dict` reloaded, a checkpoint, `GET /graph/params` (the FMU export: `tests/core/test_node_params_writes_are_observed.py`) | A constant of a three-argument node, a structural `int` and a `gm.params` leaf of a params-taking node; two writes in a row | An entry point that keeps its own copy of the compiled step (an FMU sidecar built before the write runs the step it was given) |

### Sharding wrappers

**Oracle.** For a node and a composition of sharded wrappers around it,
the graph built with the sharded composition and the same graph built
with the node unwrapped give the same answer on every graph surface:
`step`, `run`, `run_scan`, `run_scan(params=)`, a `gm.params` write (a
`node.params` write for a node on the three-argument contract) followed
by `compile()`, a `set_node_state` write followed by `run_scan`, and the
gradient of a loss of `run_scan`'s final state. Otherwise the sharded
composition must be refused loudly before it produces a state: at
construction, at `compile()`, or at the first step. (`compile()` does not
trace the step, so every check a wrapper makes in `update` fires at the
first step.)

**Where.** `tests/cloud/multigpu/differential_sharding_support.py` holds
the generators, the two paths and the comparisons.
`test_differential_sharding.py` covers the synthetic nodes and the
`property_support` examples; `test_differential_sharding_builtins.py`
covers the built-in nodes, the refusals and the known cases.

**What is generated.** Three synthetic node families, one per wrapper.
In each, the unsharded `update` and the per-shard `update_padded` call one
shared kernel, so the only thing that can make the paths differ is what
the wrapper delivers:

- *Stencil* (`ShardedStencilNode`):
  - grid and halo: a 1-D or 2-D grid, halo width 1 or 2 per axis, and a
    global fill of `edge`, `zero` or `periodic`, declared through
    `halo_boundary()` or not;
  - contract: the legacy three-argument contract or the params contract;
  - domain integral in the state: none, scalar, 2-vector or per-shard;
    named to sort before or after the grid field; listed in
    `state_fields()` or not;
  - `shard_info`: read or not;
  - statics: a sharded `StaticArray` read in the interior or in the halo,
    and a replicated one read through the offsets;
  - boundary inputs: per-cell, scalar or mis-shaped; a gain; face inputs
    applied on the global edges;
  - precision: float32 or float64.
- *Pointwise* (`ShardedPointwiseNode`):
  - a 1-D or 2-D state, sharded along either axis;
  - the contract;
  - per-cell, scalar or mis-shaped inputs;
  - a whole-domain total in the state.
- *Unstructured* (`ShardedUnstructuredNode`):
  - partitions: uneven (an empty shard possible), balanced in global
    order, and balanced out of global order;
  - the exchange: `all_to_all` or `ppermute`;
  - the contract and the integral kinds above, the integral masked with
    `shard_info["n_local"]`;
  - a partitioned weight read on owned cells only, or on ghosts too;
  - per-cell, scalar or mis-shaped inputs.

Meshes: 1, 2 or 4 devices; the pencils (2, 2), (1, 4) and (4, 1),
including their size-1 axes; slabs of a 2-D grid; and a (2, 2) mesh whose
`axis_map` uses one axis, the node replicated along the other. Wrappings: the
wrapper alone; nested one level (stencil and pointwise);
`HybridNode(wrapper)`; and the pointwise wrapper around a `HybridNode`.

Built-in nodes: `HeatNode` at orders 2 and 4, with rod-end temperatures
and sources. A non-uniform rod must be refused sharded, and runs
unsharded. `LBMNode` D2Q9 and D3Q19 channels, with walls and an obstacle,
on slabs and pencils, with body force and pressure faces. A pressure face
is refused on a split axis. `ShardedStencilNode(LBMPipeNode)` is refused.

The non-dividing-grid refusal's advice is followed both ways: the next
multiple, and each device count that divides. The sharded graph then
answers as the unsharded one.

**Tolerances.**

- *Grid fields*: within `16 eps max(1, |field|)` per step, growing by at
  most 1.5 per later step (`grid_atol`). The two paths run one kernel on
  the same data, so they would agree bit for bit but for one thing:
  XLA:CPU contracts `a*b + c` into a fused multiply-add inside a fusion,
  and a `shard_map` is a fusion boundary the unsharded program does not
  have. An FMA skips one rounding. Measured on jaxlib 0.11.0, every
  float32 `a*b + c` of 10^5 matched the correctly rounded FMA, and 23%
  differed from the split computation. The largest difference observed
  over the generated matrix was 1/16 of the bound, one ulp of the field's
  scale.
- *Domain integrals*: the summation-reordering bound
  `(n - 1) eps sum|t|`, plus the integrated grid field's own bound.
- *Gradients*: the larger of two bounds (`gradient_bound`).
  - *A derived base*, from a forward-mode pass of the unsharded path: the
    absolute sum of the gradient's terms before they cancel, the tangent,
    and the departure from the initial state. A near-equilibrium state,
    whose gradient is a small difference of large terms, gets the looser
    base it needs.
  - *The gradient's measured sensitivity to reordering*: four times the
    distance between the unsharded path's reverse-mode and forward-mode
    gradients, in the run's own precision. The two compute one derivative
    with every operation reordered, while sharding reorders only the
    shard-boundary terms.

  Each half was added after an oracle error, not a wrapper defect:

  - The first version used a fixed allowance relative to `|g|`, and an LBM
    channel one step from rest exceeded it.
  - The base alone then failed in the slow lane, on an unstructured ring
    with an empty shard, relaxed to nearly equal values. There
    `d x / d rate` is a small difference of nearly equal numbers. The two
    paths agree to 5e-13 in float64, while in float32 each is 1e-4 off
    the float64 gradient, the sharded path the closer. Sharded and
    unsharded differed by 0.5 times the forward/reverse distance in
    float32 and 0.8 times in float64.

  That case is a per-push witness. The largest well-conditioned gradient
  difference observed was 0.05 of the base.
- *Precision*: every bound is in units of the dtype's own `eps`, so a
  float64 run holds the sharded path to float64. A `dt` rounded to
  float32 under x64 (MADD-ANO-033) is a failure.

**Budget.** The per-push tests draw one surface per example at the
`EXAMPLES_COSTLY` depth, with `derandomize=True`. Gradients run on a fixed
set per push. On a 6-core slice with CI's compilation-cache settings, the
slowest per-push test takes 3.2 s warm and 9.4 s cold (the generated D2Q9
test, allowlisted: one LBM compile per path per example). The rest take
at most 2.7 s warm and 5.2 s cold. Each `@pytest.mark.slow` test is the broad
version of a named per-push test: every surface on every example, at
`EXAMPLES_STANDARD` depth, gradients over the whole matrix.

**Harness mutants.** Each fault was seeded into a scratch copy of `src/`;
the per-push tests catch each one.

| Seeded fault | Per-push tests that catch it |
|---|---|
| `shard_info`'s extent read from the first state field in sorted order, a domain integral included (MADD-ANO-056) | generated 1-D stencil |
| A sharded static's halo always edge-filled (the pre-0.4.0 rule) | generated 1-D and 2-D stencil; gradients |
| A nested `ShardedStencilNode` not forwarding `params` | generated 1-D and 2-D stencil; generated heat rod; built-in gradients |
| A per-cell input replicated instead of sharded: stencil wrapper | ten tests, every family the stencil wrapper serves |
| A per-cell input replicated instead of sharded: unstructured wrapper | generated unstructured; gradients; the known balanced-partition refusal |
| An integral summing the padding rows (`shard_info["n_local"]` set to `n_local_max`) | generated unstructured; gradients |
| The gradient stopped at the stencil wrapper (`stop_gradient` on `params`) | gradients (synthetic and built-in) |
| A recompile that keeps the stale compiled step (MADD-ANO-032) | generated 1-D and 2-D stencil |
| A mis-shaped per-cell input not refused (MADD-ANO-057) | generated 1-D and 2-D stencil; generated D2Q9 |
| The `shard_info` offset not scaled by the extent | generated 1-D and 2-D stencil; generated heat rod; gradients; the non-dividing-grid advice |
| `dt` cast to float32 under x64 (MADD-ANO-033) | generated 1-D and 2-D stencil; gradients |
| `ShardedPointwiseNode` not forwarding `params` | generated pointwise; gradients; the examples |
| The halo of an unsharded axis edge-filled whatever the fill | generated 2-D stencil; generated D2Q9; D3Q19; gradients |
| `ShardedUnstructuredNode` not forwarding `params` | generated unstructured; gradients; the examples |
| A mis-shaped per-cell input not refused by `ShardedUnstructuredNode`, or a slab-length one let through (MADD-ANO-064) | generated unstructured (the first); the slab-length case |
| `gather_global` reading `state_fields()` before the integrals (MADD-ANO-065) | generated unstructured, through its `gather_global` cross-check; the two listed-integral cases |
| A nested stencil wrapper placing its inner wrapper's already-stacked integral (MADD-ANO-066) | the nested per-shard case |
| A domain integral summed over a mesh axis the `axis_map` leaves unused (MADD-ANO-067) | generated 1-D and 2-D stencil; the unused-axis integral cases |
| A per-shard integral stacked over such an axis | generated 1-D stencil; the unused-axis integral cases |

Two more faults are caught only beside the harness: an unused axis that
`domain_integral_axes()` names, summed over
(`test_unused_mesh_axis_integrals.py`), and the zero-size ghost tail of
`exchange_unstructured` put back (`test_zero_ghost_exchange.py`, on every
jaxlib; on 0.11.2 the subprocess crash test catches it too).  All eight
faults of that round were caught (jaxlib 0.11.0).

The first run of this table caught 13 of the 14 faults, three of them only
through the fixed gradient set, and missed the stale compiled step. Both
causes were in the harness:

- Hypothesis biases generation towards the first element of a
  `sampled_from`, so half of a 20-example per-push run was single-device
  on a one-cell grid. The strategies now list their richest value first.
- The write surface wrote before the graph had ever traced, so there was
  no stale trace to keep. It now runs, writes, recompiles and runs again.

**What it cannot see.**

- *Code both paths share*: `GraphManager`, its params validation, and
  the node's own kernel. A kernel that computes the wrong physics
  computes it on both paths.
- *Specification errors*: where the harness node follows the wrapper's
  documented contract (the static-halo fill rule, the `shard_info`
  layout), a harness that agrees with a wrong document agrees with a
  wrong wrapper.
- *Excluded cases*: the cases excluded while a fix is pending
  (`PENDING_*` in the support module). None is left. The last,
  `PENDING_XLA_REPLICATED_STATIC_IN_SCAN` (an XLA miscompile inside a
  loop, MADD-ANO-068), went when `GraphManager` began refusing such a
  loop: the configurations are drawn again, and on every surface but
  `step` and `run` the oracle requires the refusal where
  `loop_refusal_expected` predicts it (naming the static and every mesh
  axis it is copied along) instead of comparing the two paths. So the
  harness checks that the refusal fires there and nowhere else, not
  what XLA would have computed.
- *Hardware*: real multi-GPU hardware and NCCL. The harness runs on
  virtual CPU devices only.
- *Performance*.

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
