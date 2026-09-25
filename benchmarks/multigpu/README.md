# Multi-GPU hardware session — runbook

Human-driven, one sitting, one pod.  **Nothing in this directory launches,
stops or tears down anything**: `run_pod.py` only runs *on* the machine
that has the GPUs and writes JSON.  Launching the pod, copying results
back and tearing it down are done by the maintainer, by hand, at the
steps marked below.

Every sharding claim of the release was verified on CPU virtual devices
only.  The session closes that gap on real multi-GPU collectives, and it
is **pure comparison**: every goal compares what it runs against a
reference computed on the same pod (the unsharded node, a NumPy or float64
model, or the refusal the library promises) and records a pass/fail check
per comparison, with the limit it was held to.  Nothing is decided at the
pod except whether to stop.

## What the session settles

### The sharding checklist

| # | claim | goal(s) | compared against | limit |
|---|---|---|---|---|
| 1 | sharded ≡ unsharded, stencil and unstructured wrappers | `stencil` (periodic, edge and Dirichlet ends), `forward` | the unsharded node, same pod | rel 1e-5 |
| 2 | halo exchange at the shard and global boundaries | `halo` | NumPy, every slot, forward and adjoint | **0** (bit for bit) |
| 3 | sharded adjoint ≡ unsharded adjoint | `stencil`, `gradient`, `coupled` | the unsharded adjoint | rel 1e-5 (rollouts), 1e-4 (coupled, IFT), 1e-3 (`sharded_cg`) |
| 4 | nested `HybridNode(ShardedStencilNode(inner))` | `hybrid` | `HybridNode(inner)` in the same graph | rel 1e-5 |
| 5 | an indivisible grid is refused | `indivisible` | the documented `ValueError`, naming both numbers | exact |
| 6 | one sharded and one replicated member in one coupling group, with its adjoint | `coupled` | the same group with the node unwrapped, **and** a float64 model of the coupled step | rel 1e-5 forward; gradient 1e-4 (IFT) / 1e-5 (`"fori"`), 1e-4 against the model |

The limits live in `LIMITS` in `run_pod.py`, each with its reason; a dry
run is held to the same ones.  Item 6 has no expected value anywhere else,
which is why its goal also carries an independent float64 model: a fault
that moved the sharded and the unsharded group alike (a solver problem on
the GPU backend, say) passes a sharded-versus-unsharded comparison and
fails against the model.

### The transport question

| goal | question | decides |
|---|---|---|
| `exchange` | NCCL time of `exchange_unstructured(method="all_to_all")` vs `"ppermute"` at 1e5, 3e5, 1e6 cells on 4 GPUs | whether `ppermute` becomes the default `exchange=` of `ShardedUnstructuredNode` (an API default: change it before the stability freeze or not at all).  Only a real-GPU run on **≥ 4 devices** decides (2–3 with `--allow-fewer-devices`, recorded in the JSON; 1 device exchanges nothing and is refused) |
| `forward` | a 1e6-cell `ShardedUnstructuredNode` forward run on a real unstructured mesh, both transports | the v0.4.0 "real-mesh size" commitment (`docs/developer_guide/sharding_topology.md`) |
| `gradient` | `jax.grad` through a sharded rollout (both transports) and through the Jacobi-preconditioned `sharded_cg` at 1e5–1e6 DOF | the real-GPU half of the gradient-parity-at-scale gate (the CPU-virtual half is `tests/cloud/multigpu/test_iterative_solver.py::TestGradientParityAtScale`) |

## Checklist → coverage map

What already covers each item on CPU virtual devices, and which goal
carries it to the GPUs.

| # | CPU virtual devices (`tests/cloud/multigpu/`) | on the pod |
|---|---|---|
| 1 | `test_property_sharded_equals_unsharded.py` (all three wrappers, 1–4 devices, node and graph level), `test_sharded_stencil_node.py`, `test_sharded_unstructured.py`, `test_exchange_ppermute.py` | `stencil` (a 2-D field with a sharded `StaticArray`, 1e5–1e6 cells; periodic ends at every size, and at the smallest also `"edge"` ends -- the wrapper's default fill, on the sharded and the unsharded axis -- and Dirichlet ends held through a boundary input), `forward` (unstructured, both transports) |
| 2 | `test_halo.py` (slab and pencil fills, halo 1 and 2, the gradient by finite differences), `test_property_exchange_transports.py`, `test_exchange_ppermute.py` | `halo`: every mode × width on a 1-D and a 2×2 mesh, and the unstructured exchange under both transports, forward and adjoint, bit for bit |
| 3 | `test_property_sharded_equals_unsharded.py::test_a_gradient_through_the_sharded_path_matches_the_unsharded_one`, `test_property_injected_params_gradient.py`, `test_sharded_gradient.py` (finite differences), `test_iterative_solver.py` | `stencil` (d/d initial field and d/d a parameter), `gradient`, `coupled` |
| 4 | `test_sharded_static_cache.py`: static-cache invalidation, hashing and drift through `HybridNode(ShardedStencilNode(...))` with an empty correction — no forward or adjoint parity with a non-zero correction | `hybrid`: `run_scan` and `jax.grad` of the graph with a non-local correction (a shift across shards) |
| 5 | `test_property_shard_construction.py` (the stencil and pointwise refusals; the unstructured wrapper taking a prime cell count) | `indivisible`: the same refusals on the real mesh, a 2×2 pencil refusal naming axis 1 (recorded as *not run* on a device count with no pencil mesh), and an uneven unstructured split matching the unsharded node |
| 6 | `test_coupling_group_with_sharded_and_replicated_members.py`: forward on every push, adjoint in the slow lane, default solver and `"fori"`, against the unwrapped group and a float64 model | `coupled` |

Before item 6's test, the nearest coverage was
`test_sharded_wrapper_coupling_hooks.py`, which couples a
`ShardedPointwiseNode` to a relay under `solver="ift"` — forward only,
three steps — around a spring whose state is 0-d and so lives whole on one
device: nothing in the group was partitioned.  `test_coupled_sharded.py`
couples two sharded nodes by staggered edges (no coupling group, a 10%
tolerance) and `test_graph_multigpu.py` places whole nodes on devices.

## 0. Before spending anything (laptop, free)

```sh
cd MADDENING
JAX_PLATFORMS=cpu python benchmarks/multigpu/run_pod.py --goal all --dry-run --cells 256 --out /tmp/mg-dry
python benchmarks/multigpu/run_pod.py --summarise /tmp/mg-dry     # needs no JAX
pytest tests/cloud/multigpu -m "slow or not slow" -q             # includes the runner dry-run test
```

The dry run takes about a minute on three cores.  It must print
`checks n/n passed` for all eight goals, no `CHECK NOT RUN` line, and exit
0.  The summary must
exit 0, show every checklist item as `open: passed on CPU / dry run only`
(a dry run never closes an item), list nothing under "Records that cannot
decide", and show the transport recommendation as `undecided`.  Anything
else: fix it here, not on the pod.

What the timings measure: every timed callable gets inputs that were
placed on the mesh once, with the `NamedSharding` the compiled
`shard_map` expects, *outside* the timed region (`input_presharded` in
the JSON; the runner refuses to time anything else).  An uncommitted
device-0 array would otherwise be scattered to the mesh on every call
and charged to both transports equally, compressing the very ratio the
session measures.  Compile time is reported apart (`compile_s`, pure
compile, no execution) on the sharded and the unsharded side alike.

If a real unstructured mesh (a helix-in-vessel mesh, say) is to be used
for `forward`, export it now as an `.npz` with `edges` (`(n_edges, 2)`
int, cell–cell adjacency, global ids) and optionally `partition`
(`(n_cells,)` int in `[0, 4)`, e.g. from PyMetis).  Without `partition`
the runner partitions with PyMetis if installed on the pod, else
reverse-Cuthill-McKee blocks (SciPy), else contiguous cell ids.  Keep the
file under a few hundred MB; it is copied to the pod with the source tree.

## 1. Launch (the maintainer, by hand)

Nothing here or in the runner launches a pod; the maintainer does, with
the existing launcher, once the dry run above is clean.  Two equivalent
manual routes:

* **`maddening.cloud.launcher.CloudLauncher` + `JobConfig`** from a
  Python REPL: `provider="runpod"`, `gpu_type` an A100 name from
  `sky show-gpus --cloud runpod` (`A100-80GB-SXM` or `A100-80GB`),
  `gpu_count=4`, `use_spot=False` (a preempted benchmark is a wasted
  hour), `workdir=<MADDENING checkout>` so the tree is synced, and a
  `CostPolicy(max_cost_per_hour=12.0, max_total_budget=40.0,
  autostop_minutes=20, auto_teardown=False)` — autostop is the safety net
  for a pod left idle; teardown stays with the maintainer.  `run` can be
  `"sleep 7200"`; the runner itself is driven over SSH with
  `CloudJob.ssh_run`.  Credentials come from
  `~/.maddening/cloud_credentials.yaml` (`CloudLauncher._load_credentials`),
  the RunPod key lands in `~/.runpod/config.toml` only for the duration
  of the launch (`_credential_context`).
* **SkyPilot's CLI** (`sky launch` with `--gpus A100-80GB:4
  --cloud runpod --no-use-spot --workdir .`), typed by the maintainer.

`src/maddening/examples/cloud/multigpu/09_real_gpu_benchmark.py` shows
the launch → `ssh_run` → copy-back → `teardown` shape end to end (it is
the 2-GPU Jacobi-coupling crossover benchmark; do not run it in this
session, it is a different measurement).

## 2. Run (on the pod)

### 2a. Sanity, and what to record before the first goal

```sh
cd ~/sky_workdir
nvidia-smi -L                                    # four GPUs, else stop (section 3)
pip install -q "jax[cuda12]>=0.10,<0.13" && pip install -q -e .
python -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__, jax.devices())"
mkdir -p results/multigpu
{ date -u; git rev-parse HEAD; nvidia-smi; pip freeze | grep -iE "^(jax|jaxlib|jax-cuda|lineax|equinox|numpy|scipy)"; } \
    > results/multigpu/session.txt 2>&1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
set -o pipefail
```

Optional: `pip install pymetis` for a real graph partition of the mesh.

### 2b. The goals, in this order, each under its time box

Run one command at a time and read its exit status (`echo $?`) before the
next: **0** = no check failed, go on; **1** = a check failed (the log
names it on a `CHECK FAILED` line); **124** = the time box ran out;
anything else = a crash.  Anything but 0 is the stop condition in
section 3.  A `CHECK NOT RUN` line is a case this device count cannot
express (the 2-D pencil cases need an even count of at least 4); it is
not a failure and does not stop the session, but it keeps the item open.
On four GPUs there are none.

```sh
R="python benchmarks/multigpu/run_pod.py --out results/multigpu"
L=results/multigpu
timeout 10m $R --goal indivisible 2>&1 | tee $L/indivisible.log      # checklist 5
timeout 10m $R --goal halo        2>&1 | tee $L/halo.log             # checklist 2
timeout 20m $R --goal coupled     2>&1 | tee $L/coupled.log          # checklist 6 (and 3)
timeout 15m $R --goal stencil     2>&1 | tee $L/stencil.log          # checklist 1 and 3
timeout 15m $R --goal hybrid      2>&1 | tee $L/hybrid.log           # checklist 4
timeout 10m $R --goal exchange    2>&1 | tee $L/exchange.log         # the transport ranking
timeout 15m $R --goal forward     2>&1 | tee $L/forward.log   # add --mesh /path/to/mesh.npz for a real mesh
timeout 20m $R --goal gradient    2>&1 | tee $L/gradient.log         # checklist 1 and 3, unstructured
python benchmarks/multigpu/run_pod.py --summarise results/multigpu | tee $L/summary.txt
```

The checklist goals come first because they are cheap and decisive; the
coupled goal is third because item 6 is the one no other measurement
covers.  `--goal checklist` runs the first five in one process and
`--goal all` all eight; both stop after the first goal whose checks fail
(`--keep-going` overrides; do not use it on the pod).  Defaults on GPUs:
cells `1e5 3e5 1e6` (square 2-D fields for the stencil goals, rows a
multiple of the device count), 5 warmup + 20 timed repeats, 20 steps per
timed block, 5 differentiated steps, `--n-devices 4`.  The checklist goals
refuse one device.  Re-run `exchange` with `--fields 5 --out
results/multigpu-f5` if time allows (the payload of a 5-field state); the
summary ranks each directory separately.

`pytest tests/cloud/multigpu` on the pod runs on 16 *virtual CPU*
devices unless `JAX_PLATFORMS=cuda` is exported (the root conftest
defaults to `cpu`); with it and four GPUs the suite uses them and skips
the tests that need 8/16 devices.  It is not part of the session.

### What to record

* **The JSON files**, one per goal.  Each records the `jax` and `jaxlib`
  versions, the platform, `device_kinds` and `n_devices_visible`, the
  mesh size (`n_devices`), `dry_run`, the `nvidia-smi` output, the git
  commit, the full CLI config, every check with its value and limit, and
  `passed`.  **jaxlib's float32 results are not comparable across
  versions** (0.11.2's CPU reductions are measurably tighter than
  0.11.0's), so a number is only ever compared with one taken on the same
  `jaxlib`.
* **The logs** (`*.log`): one line per measurement, the `CHECK FAILED`
  lines, and XLA's own warnings.  Under the default solver the coupled
  goal logs `[SPMD] Involuntary full rematerialization ... f32[N+1]`: the
  adjoint's failure check (a host callback, present above 50 coupled
  degrees of freedom) is pinned to device 0, and XLA gathers the coupled
  state vector onto that device and scatters it back.  It is expected,
  it is counted in `device0_pinned_ops`, and its cost is inside the
  sharded `value_and_grad` time — record it, do not act on it at the pod.
* **`session.txt`** and **`summary.txt`**, and by hand: the pod type and
  region, the price per hour, the launch and teardown times, and the final
  cost (`job.cost_so_far()` or the provider console).

### Expected durations on 4×A100 (estimates, not measurements)

Estimated from the sizes and from the CPU dry run, where compile time
dominates; a GPU compile of the coupled group's adjoint is assumed to take
10–30 s.  The time box is what `timeout` enforces.

| goal | what runs | estimate | time box |
|---|---|---|---|
| `indivisible` | 3 refusals; a 1e5-cell uneven unstructured run, 20 public `update()` calls | ~1 min | 10 min |
| `halo` | 3 programs at 1e6 cells (1-D mesh, 2×2 mesh, unstructured) and the NumPy reference | 1–2 min | 10 min |
| `coupled` | per size, 4 programs (2 solvers × sharded/unsharded) and the float64 model on the host | 5–10 min | 20 min |
| `stencil` | 4 programs (rollout and gradient × 2 paths) per case: periodic ends at each size, edge and Dirichlet ends at 1e5 cells (5 cases) | 4–7 min | 15 min |
| `hybrid` | per size, 2 graphs × (`run_scan` and the gradient) | 3–6 min | 15 min |
| `exchange` | per size, 2 transports | ~5 min | 10 min |
| `forward` | per size, 2 transports, public and compiled step | ~10 min | 15 min |
| `gradient` | per size, 3 rollout gradients and `sharded_cg` (3000-iteration cap) | ~15 min | 20 min |

About 45–55 minutes of goals, 1–1.5 h with setup and copy-back.  Budget
one and a half pod-hours; the whole session is capped at **2 hours**.

## 3. Stop condition

**Stop running goals, copy back what exists, and tear the pod down**
(section 4) as soon as any of these happens — do not debug on a metered
pod:

* a goal exits non-zero: `1` (a check failed), `124` (its time box ran
  out), or anything else (a crash, an out-of-memory);
* `nvidia-smi -L` lists fewer than four GPUs, or `jax.devices()` does
  not list four CUDA devices;
* the session passes **2 hours** since launch, or the provider's cost
  reaches the `CostPolicy` budget.

A failed goal's JSON is still written (it records which check failed and
by how much); copy it back with the rest.  The failure is reproduced and
fixed off the pod, and a later session runs the failed goal, the goals
after it, and every earlier goal that decides a checklist item together
with one of those: an item is decided only by files from one git commit
(section 5), so a fix commit re-runs every goal of the items it touches.

## 4. Copy back, then tear down (the maintainer, by hand)

From the laptop, with the `CloudJob` still in hand:

```python
job.ssh_run("cd ~/sky_workdir && tar czf /tmp/multigpu-results.tgz results/multigpu")
# then scp -P <job.ssh_port> root@<job.vm_ip>:/tmp/multigpu-results.tgz .
job.teardown()
```

or with the CLI: `rsync -avz -e "ssh -p <port>" root@<ip>:~/sky_workdir/results/multigpu/ benchmarks/results/multigpu/`
followed by `sky down <cluster>`.  Confirm on the RunPod console that
the pod is gone.  Cost check: `job.cost_so_far()`.

Store the JSON under `benchmarks/results/multigpu/` (tracked, small)
and commit it with the summary output pasted into the commit body.

## 5. Read the result

`python benchmarks/multigpu/run_pod.py --summarise benchmarks/results/multigpu`
exits 0 when no recorded check failed, 3 when any failed or a file cannot
decide (below), and 1 when the directory holds no goal JSON.  It does not
take a check's recorded `passed` on trust: pass/fail is re-derived from
the check's `value`, `limit` and `sense`, and a record that disagrees --
a value of 0.5 against a limit of 0.0 recorded as passed, say -- is
listed as failed (`recorded passed=True, but the value fails its limit`),
as is a file whose top-level `passed` its checks do not bear out.

Nor does it take a file's word for *what* was checked.  A file is
evidence only if it is what `run_pod.py`, as it stands, would have
written; otherwise its goal reads `INVALID` and it closes nothing.  A
file must:

* be on the current `schema_version` (4);
* record an `n_devices` no larger than the devices its `environment`
  lists (`n_devices_visible`, which must count `devices`), and the same
  `n_devices` in its `config` and in every result entry;
* hold every case the runner runs for the file's own `config` (`cells`,
  `n_devices`, `synthetic`, `mesh`) and no other: every boundary mode ×
  width × mesh of `halo`, the `"edge"` and Dirichlet ends of `stencil`
  at its smallest size, both solvers of `coupled`, both transports, ...;
* carry exactly the checks the runner derives from its results -- the
  same names, values, limits, senses and flags.  So every limit is the
  one in `LIMITS` (a limit loosened, or tightened, in the file is
  refused), a results table that disagrees with a check is refused, and
  a check deleted from the file or added to it is refused.  The order of
  the checks does not matter.

And the files that decide one checklist item must all record the same
`git_commit` (and must record one).  A directory holding goals from two
commits keeps the items they share open; re-run those goals on one
commit.  If the runner itself changed between the session and the
summary (a new check, a new case), the session's files no longer match
it and read `INVALID`: summarise with the commit the session ran, which
every file records.  The transport ranking uses the rows of an
`exchange.json` only when that file reads `PASS`.

It prints:

* **Runs**: one line per JSON file — platform, devices, device kind,
  `jax / jaxlib`, dry run, checks passed, verdict (`PASS`, `FAIL`,
  `INVALID`, `INCOMPLETE`).
* **Checklist**: per item, `CLOSED` (every goal that decides it passed,
  with every check run and every file valid, on real GPUs, not a dry run,
  on ≥ 4 devices, from one commit — `--allow-fewer-devices` does not
  lower that: it is for the transport ranking, and on 2 devices a halo
  taken from the wrong neighbour passes every check), `FAILED` (a
  deciding goal failed a check), `open: <file> cannot decide it (<why>)`
  (a deciding file is `INVALID`), `open` (a deciding goal was not run,
  recorded no checks, or is `INCOMPLETE`: a check was recorded as not
  run), `open: passed on CPU / dry run only`, `open: passed on fewer than
  4 devices`, or `open: its files come from N commits` / `open: no git
  commit recorded in <file>`.  The session succeeded when all six read
  `CLOSED`.
* **Records that cannot decide**: every reason a file is `INVALID`, and
  every item whose files come from more than one commit.
* **Failed checks**, each with its value and limit, and **Checks not
  run**, each with why.
* The per-goal tables (indivisible, halo, coupled, stencil, hybrid), then
  the transport ranking and the forward and gradient tables.

For the transport: **Recommendation `ppermute`**: median ppermute time
beats all_to_all by ≥ 1.05× at every real-GPU point ≥ 1e5 cells → change
the default `exchange=` of `ShardedUnstructuredNode` and
`exchange_unstructured` to `"ppermute"` (one-line change each, plus the
docs paragraph in `sharding_topology.md` and a CHANGELOG entry under
Changed).  **`all_to_all`**: it is faster at some hardware-sized point →
keep the default, record the numbers in the docs paragraph.  **`tie`**:
keep `all_to_all` (one collective) unless the byte savings (`bytes_total`
columns) matter for the target mesh; write that down.  **`undecided`**: no
row decides.  A row decides only when it comes from a real accelerator
run (not `--dry-run`), has ≥ 1e5 cells, ran on ≥ 4 devices (≥ 2 if that
run recorded `allow_fewer_devices`) and has a finite speedup (a 0 ms
median is below timer resolution).  The `decides` column of the table and
the reason line say what each row lacked.

The timings are secondary to the checks but worth reading:
`wrapper_step` (public `update()`, which re-uploads the partitioned
static arrays every call) against `device_step` (the compiled step) in
`forward`; the gradient timings in `gradient` and `stencil`, which are
like-for-like (`jax.jit(jax.grad(...))` on both sides, statics placed
once, compile time in `compile_s`), so "the sharded grad is N× the
unsharded one" is a statement about the compiled step, not about Python;
and in `coupled`, the sharded against the unsharded `value_and_grad`
time, which is where the device-0 gather above shows its cost at scale.

## Schema of the JSON (schema_version 4)

Common: `goal`, `dry_run`, `allow_fewer_devices`, `n_devices` (the mesh
size), `environment` (`hostname`, `timestamp_utc`, `python`, `jax`,
`jaxlib`, `platform`, `devices`, `device_kinds`, `n_devices_visible`,
`nvidia_smi` (a list, or the string `"skipped (dry run)"`), `xla_flags`,
`jax_platforms`, `git_commit`), `config` (the CLI namespace), `wall_s`,
`results`, `checks` (a list of `{name, value, limit, sense, passed[,
detail]}`: a numeric check has `sense: "<="` and passes when `value <=
limit` and is finite, a yes/no check has `sense: "=="` and `limit: true`;
a case the device count cannot express is `{name, value: null, limit:
null, sense: null, passed: false, not_run: true, detail}`, detail saying
what it needs) and `passed` (every check ran and passed, and there was at
least one).  Timings are `{warmup, repeats, ms: [...], min_ms,
median_ms, mean_ms}`; parity blocks are `{max_abs, max_rel,
reference_scale, finite}`; a bare `parity_*` number is the largest
componentwise relative difference; every `compile_s` is an ahead-of-time
compile without execution.

* `indivisible.json` results (one entry): `stencil`, `pointwise` and,
  on an even count ≥ 4 devices, `pencil` = `{shape, raised, message}`
  (otherwise a check not run) (`stencil` also
  `divisible_shape`, `divisible_raised`: one row fewer is accepted);
  `unstructured` = `{cells, partition, cells_per_device, steps,
  parity_x}`.
* `halo.json` results (one entry): `stencil_cases` = `[{mesh,
  mesh_shape, shape, halo, boundary, forward_max_abs, adjoint_max_abs}]`,
  `unstructured` = `{cells, partition, n_local_max, n_ghost_max,
  methods.<m>.{forward_max_abs, adjoint_max_abs}}`.
* `coupled.json` results: `cells`, `shape`, `steps`, `coupled_dof`,
  `max_iterations`, `tolerance`, `parameters`, `model` (`loss`, `grad`
  and `wall_s` of the float64 model), `solvers.<ift|fori>` =
  `{sharded, unsharded}` (`compile_s`, `value_and_grad` timing with
  `ms_per_step`, `loss`, `grad`, `last_step_iterations` (`null` under
  `"fori"`), `partitioned`, `device0_pinned_ops`), `parity_f`,
  `parity_u`, `parity_loss`, `parity_grad`, `model.<side>.{f, u, loss,
  grad}`.
* `stencil.json` results, one per case: `cells`, `shape`, `boundary`
  (`periodic`, `edge` or `dirichlet`), `steps`, `grad_steps`,
  `input_partitioned`, `forward.{sharded,unsharded}` (`compile_s`,
  `rollout` timing with `ms_per_step`), `forward.parity_f`,
  `gradient.{sharded,unsharded}` (`compile_s`, `grad` timing, `loss`,
  `grad_diffusivity`), `gradient.parity_loss`,
  `gradient.parity_grad_initial_field`, `gradient.parity_grad_diffusivity`.
* `hybrid.json` results: `cells`, `shape`, `steps`, `grad_steps`,
  `correction_shift_rows`, `{sharded,unsharded}` (`run_scan_first_call_s`,
  `partitioned`, `compile_s`, `value_and_grad` timing, `loss`, `grad`),
  `correction_rel`, `parity_f`, `parity_loss`, `parity_grad`.
* `exchange.json` results: `cells`, `mesh`, `partition`, `n_devices`,
  `fields_per_cell`, `n_local_max`, `n_ghost_max`, `layout_build_s`,
  `traffic_cells_per_shard` (`exchange_traffic()` output),
  `input_sharding`, `input_presharded`,
  `methods.{all_to_all,ppermute}` = timing + `compile_s`,
  `bytes_per_shard`, `bytes_total`, `messages`, `bandwidth_GBps`;
  `bit_identical`, `ppermute_speedup_median` (`null` when the ppermute
  median is 0), `ppermute_speedup_min`.
* `forward.json` results: `cells`, `mesh`, `partition`, `n_devices`,
  `steps`, `n_local_max`, `n_ghost_max`, `traffic_cells_per_shard`,
  `methods.<m>` = `compile_s`, `wrapper_first_call_s` (host
  partitioning of the statics by the public `update()`),
  `input_presharded`, `wrapper_step`, `device_step` (timings with
  `ms_per_step`), `parity_x`, `parity_total`.
* `gradient.json` results: `cells`, `mesh`, `partition`, `n_devices`,
  `grad_steps`, `rollout.{unsharded,all_to_all,ppermute}` (`grad`
  timing and `compile_s`; `input_presharded` and `parity` for the
  sharded ones), `sharded_cg` (`dof`, `input_presharded`,
  `grad_sharded`, `grad_unsharded`, `compile_s.{sharded,unsharded}`,
  `grad_parity`, `jvp_parity`).

Schema 3 files carry no `sense` (the summary reads it from the limit's
type, which is how schema 3 wrote checks), never record a check as not
run, and ran the stencil goal with periodic ends only.  Schema 2 files
(the first three goals, before the checklist goals) carry
no `checks`, `passed` or top-level `n_devices`; the summary shows them as
`no checks` and they close no checklist item.  Schema 1 files lack the
`compile_s`/`input_presharded` keys and carried gradient timings that were
not like-for-like; nothing under `benchmarks/results/` was written with
either.
