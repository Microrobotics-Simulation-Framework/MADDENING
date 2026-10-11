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
| 1 | sharded ≡ unsharded, stencil and unstructured wrappers | `stencil` (a field under periodic, edge and Dirichlet ends and a D2Q9 lattice, on every mesh below), `forward` | the unsharded node, same pod; `forward` also the number of cells | rel 1e-5 on every field; `forward`'s count of its cells **0** (exact) |
| 2 | halo exchange at the shard and global boundaries | `halo` | NumPy, every slot, forward and adjoint | **0** (bit for bit) |
| 3 | sharded adjoint ≡ unsharded adjoint | `stencil`, `gradient`, `coupled` | the unsharded adjoint | rel 1e-5 (rollouts), 1e-4 (coupled, IFT), 4e-4 (`sharded_cg`, each of whose solves must have converged: true residual within 2e-4) |
| 4 | nested `HybridNode(ShardedStencilNode(inner))` | `hybrid` (every mesh) | `HybridNode(inner)` in the same graph | rel 1e-5 |
| 5 | an indivisible grid is refused | `indivisible` | the documented `ValueError`, naming both numbers | exact |
| 6 | one sharded and one replicated member in one coupling group, with its adjoint | `coupled` (every mesh) | the same group with the node unwrapped, **and** a float64 model of the coupled step | rel 1e-5 forward; gradient 1e-4 (IFT) / 1e-5 (`"fori"`), 1e-4 against the model |

The limits live in `LIMITS` in `run_pod.py`, each with its reason; a dry
run is held to the same ones.  The `forward` goal checks its cross-shard
sum exactly: one more public step from a field of ones, which the node
leaves a field of ones bit for bit, must return `total` equal to the
number of cells and a field still all ones, with a limit of zero, sharded
under both transports and unsharded.  A float32 sum of ones is exact in
any order of addition under 2**24 = 16,777,216 cells, so a shard left out
or counted twice reads off by its 25,122 to 250,000 cells at the gate's
sizes, ghost rows counted by their 1,532 to 4,830, and pad rows counted by
half a cell each (1.5 cells at 100,489; 300,304 and 1,000,000 cells divide
by four, have no pad row, and nothing can be miscounted there).  The float
`total` of the goal's own field is recorded (`parity_total`) and must be
finite, and is held to no limit: that field's sum cancels (`sum|x| / |sum
x|` of 22 to 69,000), so eight float32 orders of addition of the same
field differ by up to 3.5e-4 of the total while every ghost row counted
moves it by 3.6e-4 at 1,000,000 cells, and no limit on that number tells
the two apart.  (Until schema 8 it was held to 1e-5: on CPU it read
6.95e-6 at 300,304 cells after the dry run's 3 steps and 0.0 at the gate's
sizes after 20, with the field itself bit-identical.)  The `sharded_cg` rows of item 3 solve
`(2 + s) x[i] - x[i-1] - x[i+1] = b[i]` with `s = CG_SHIFT = 0.03` to
`CG_RTOL = 1e-4` in float32, for a white-noise `b`: a system both sides
converge on in 55 to 58 iterations a solve at every size (condition number
134), where the two derivatives then agree to float32 round-off (2e-7 to
8e-7 on CPU from 256 to 1e7 unknowns, against the limit of 4e-4).  Each side's three
solves -- the solve, the adjoint solve behind the gradient and the tangent
solve behind the jvp -- must have converged: the loop stopped on its
tolerance and not on its cap, and the true residual `|rhs - A x| / |rhs|`,
computed on the host in float64, is within 2e-4.  "Stopped on its
tolerance" is read from the loop's iteration count, which must be under
the cap (the loop has those two exits); the `converged` flag in the record
is information, not a check: it is `sharded_cg`'s float32 residual against
`rtol` with no allowance, which lands within about 12 % of `CG_RTOL` here.
Until schema 7 these
rows solved the unshifted operator, on which no float32 solve converges
(true residual 6e2 to 2e4 at the session's sizes after all 3000
iterations), and passed on parity alone, within 8 % of their limit at
1e6.  Item 6 has no expected value anywhere else,
which is why its goal also carries an independent float64 model: a fault
that moved the sharded and the unsharded group alike (a solver problem on
the GPU backend, say) passes a sharded-versus-unsharded comparison and
fails against the model.

### The meshes, and why four devices need more than the pencil

Every goal that runs the stencil wrapper or the halo exchange runs it on
four meshes (`STENCIL_MESHES`), named for what they shard; over four
devices, as devices along spatial axis 0 × axis 1:

| mesh | shape | what it adds |
|---|---|---|
| `1d` | 4 × 1 | spatial axis 0 split over all four devices; axis 1 whole on each, filled by the wrapper as unsharded |
| `1d-axis1` | 1 × 4 | the same 1-D mesh sharding spatial axis 1 (`axis_map={MESH_AXIS: 1}`): axis 1 over all four, axis 0 the unsharded one |
| `2d-flat` | 1 × 4 | a 2-D mesh whose two axes have different sizes: the wrapper's one-device exchange on axis 0, all four on axis 1 |
| `2d` | 2 × 2 | the pencil: two exchanged axes, so a node reading a halo corner reads one that crossed both |

On a mesh axis of two devices a shard's left and right neighbour are the
same device, so a halo taken from the wrong neighbour passes every
comparison.  On four devices the pencil is 2 × 2 and **both** of its axes
are like that; until schema 6 the wrapper goals ran on `1d` and `2d` only,
on square grids, and three faults confined to spatial axis 1 passed every
goal and closed all six items once relabelled as a GPU run (below).  So
each goal records a spatial axis that none of its cases splits over three
or more devices as a check *not run* (it is never the case on four devices
with the meshes above), every grid is non-square (`nx = ny + 4` on four
devices, 16 × 20 in the dry run), and every grid-shaped input -- the
source, the lattice's body force, the coupled field's `ambient` -- differs
on every block of every mesh.  The smallest size runs every mesh; the
larger sizes run the pencil only.

### What a broken wrapper does to the goals

A sharded-versus-unsharded comparison proves the wrapper only on the paths
its node carries into the answer.  Until 0.4.0 the stencil goals' node
read its static only in the interior, took scalar inputs only, declared no
domain integral and ran on a 1-D mesh, and `ShardedStencilNode` broken in
four ways -- a static's halos set to NaN, grid-shaped inputs zeroed,
domain integrals halved, every sharded axis after the first zero-filled at
the global edges -- passed every goal and closed items 1, 3, 4 and 6 once
relabelled as a GPU run.  The goals' `Field2D` now reads its mask one cell
into the halo (face-averaged conductances, the mask sharded along a
spatial axis the wrapper shards and sliced along the other by
`shard_info`), takes a grid-shaped `source` (read through a 5-point
smoothing) and a grid-shaped `ambient`, and carries a domain integral,
`averages`, in its state; the goals run on every mesh above, and the
`stencil` goal adds `LBMNode` on D2Q9, whose streaming reads the halo
corners.  In the `coupled` goal the field reaches the far field through
`averages` and the far field reaches the field as `ambient = u ×
profile`, a non-uniform grid (until schema 6, `u` broadcast to every cell,
under which a wrapper handing each shard another shard's block of an input
passed), so both paths are inside the group's fixed point and its adjoint.

Three more faults passed every goal on four devices until schema 6: each
halo along spatial axis 1 taken from the wrong neighbour (in
`halo_exchange`), `shard_info`'s block extent divided by the first mesh
axis's size, and `shard_info`'s global extent read off spatial axis 0 for
every axis (which passed on eight devices too, the grids being square).
They now fail on `1d-axis1` and `2d-flat`, on `2d-flat`, and on every mesh
that shards axis 1, respectively.

`tests/cloud/multigpu/test_run_pod_seeded_faults.py` holds eight faults
as seeds of `sharded_node.py` and `halo.py` -- the four and the three
above, and domain integrals summed over the first mesh axis only: slow-marked, it
applies each to a scratch copy of the library, runs `--goal checklist
--dry-run --keep-going`, and requires `stencil`, `hybrid` and `coupled`
to read `FAIL` (and `halo` too for the `halo_exchange` seed; the other
checklist goals `PASS`), every item those goals decide `FAILED` once
relabelled as four GPUs, the failed `stencil` checks to be on the meshes
the fault can reach, and the unseeded copy to pass everything.  Its
per-push tests fail when a seed no longer matches the library, and check
one step of the goals' own node against each seeded wrapper in-process.

### The transport question

| goal | question | decides |
|---|---|---|
| `exchange` | NCCL time of `exchange_unstructured(method="all_to_all")` vs `"ppermute"` at 1e5, 3e5, 1e6 cells on 4 GPUs | whether `ppermute` becomes the default `exchange=` of `ShardedUnstructuredNode` (an API default: change it before the stability freeze or not at all).  Only a real-GPU run on **≥ 4 devices** decides (2–3 with `--allow-fewer-devices`, recorded in the JSON; 1 device exchanges nothing and is refused) |
| `forward` | a 1e6-cell `ShardedUnstructuredNode` forward run on a real unstructured mesh, both transports | the v0.4.0 "real-mesh size" commitment (`docs/developer_guide/sharding_topology.md`) |
| `gradient` | `jax.grad` through a sharded rollout (both transports) and through the Jacobi-preconditioned `sharded_cg` at 1e5–1e6 DOF, on solves that must have converged | the real-GPU half of the gradient-parity-at-scale gate (the CPU-virtual half is `tests/cloud/multigpu/test_iterative_solver.py::TestGradientParityAtScale`) |

## Checklist → coverage map

What already covers each item on CPU virtual devices, and which goal
carries it to the GPUs.

| # | CPU virtual devices (`tests/cloud/multigpu/`) | on the pod |
|---|---|---|
| 1 | `test_property_sharded_equals_unsharded.py` (all three wrappers, 1–4 devices, node and graph level), `test_sharded_stencil_node.py`, `test_sharded_unstructured.py`, `test_exchange_ppermute.py`, `test_stencil_static_halo_and_axis_map.py`, `test_sharded_boundary_inputs.py`, `test_stencil_domain_integral_state.py`, `test_lbm_sharded.py` | `stencil` (a 2-D field reading a sharded `StaticArray` in its halo, with a grid-shaped source and a domain integral, 1e5–1e6 cells: periodic ends at every size on the pencil mesh, and at the smallest size `"edge"` ends -- the wrapper's default fill -- and Dirichlet ends held through a boundary input, plus a D2Q9 lattice with a grid-shaped body force, each on all four meshes), `forward` (unstructured, both transports) |
| 2 | `test_halo.py` (slab and pencil fills, halo 1 and 2, the gradient by finite differences), `test_property_exchange_transports.py`, `test_exchange_ppermute.py` | `halo`: every mode × width on all four meshes (a 1-D mesh along each spatial axis, the 1 × 4 two-axis mesh, the 2 × 2 pencil), and the unstructured exchange under both transports, forward and adjoint, bit for bit |
| 3 | `test_property_sharded_equals_unsharded.py::test_a_gradient_through_the_sharded_path_matches_the_unsharded_one`, `test_property_injected_params_gradient.py`, `test_sharded_gradient.py` (finite differences), `test_iterative_solver.py` | `stencil` (d/d initial field and d/d a parameter), `gradient`, `coupled` |
| 4 | `test_sharded_static_cache.py`: static-cache invalidation, hashing and drift through `HybridNode(ShardedStencilNode(...))` with an empty correction — no forward or adjoint parity with a non-zero correction | `hybrid`: `run_scan` and `jax.grad` of the graph on all four meshes at the smallest size (the pencil at the others) with a non-local correction (a shift across shards along both axes), a grid-shaped source held through an external input, and the field's domain integral compared |
| 5 | `test_property_shard_construction.py` (the stencil and pointwise refusals; the unstructured wrapper taking a prime cell count) | `indivisible`: the same refusals on the real mesh, along spatial axis 0 and along axis 1 of the 1-D mesh, a 2×2 pencil refusal naming axis 1 (recorded as *not run* on a device count with no pencil mesh), and an uneven unstructured split matching the unsharded node |
| 6 | `test_coupling_group_with_sharded_and_replicated_members.py`: forward on every push, adjoint in the slow lane, default solver and `"fori"`, against the unwrapped group and a float64 model | `coupled`, on all four meshes at the smallest size (the pencil at the others), coupled through the field's domain integral and a non-uniform grid-shaped input |

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

The dry run takes about 80 s on three shared cores (about 100 s with
`--cells 256 1024`; 44 s and 65 s before schema 6 added the meshes), most
of it XLA compiling the `stencil` cases and the coupled group's adjoint on
each of the four meshes.  It must print `checks n/n passed` for all eight goals, no
`CHECK NOT RUN` line, and exit 0.  Spell every option out: the runner
takes no abbreviations (`--dry` is refused with exit 2), because the CPU
pin of a dry run reads the literal `--dry-run` before the options are
parsed, and an abbreviated one used to run on the GPUs.  The summary must
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
  Python REPL: `provider="runpod"`, `gpu_type="RTX4090"` (the name the
  launcher's catalogue lists for RunPod, `src/maddening/cloud/providers.py`;
  confirm it with `sky show-gpus --cloud runpod`),
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
* **SkyPilot's CLI** (`sky launch` with `--gpus RTX4090:4
  --cloud runpod --no-use-spot --workdir .`), typed by the maintainer.

**The session is on 4 × RTX 4090, 24 GB each.**  What that changes:

* The cards talk over PCIe; an RTX 4090 has no NVLink.  Every transport
  timing of the session (`exchange`, and the exchange inside `forward`
  and `gradient`) is a PCIe timing, and the ranking `exchange` decides is
  a ranking on that interconnect.
* The duration estimates of section 2 were made for A100s and have not
  been made again; the time boxes are unchanged.
* No goal needs more than one card: each compares a sharded run with the
  unsharded node on one device of the same pod, at 1e5 to 1e6 cells.
  Whether a run *larger* than one card works is not a checklist question;
  the stress tail after section 4 asks it, and gates nothing.

`src/maddening/examples/cloud/multigpu/09_real_gpu_benchmark.py` shows
the launch → `ssh_run` → copy-back → `teardown` shape end to end (it is
the 2-GPU Jacobi-coupling crossover benchmark; do not run it in this
session, it is a different measurement).

## 2. Run (on the pod)

### 2a. Sanity, and what to record before the first goal

Each file records the commit `git rev-parse HEAD` names -- when git exits 0
with a full SHA; otherwise none, which keeps every item open (a tree
synced without `.git` and then `git init`-ed used to record `"HEAD"`, which
the summary took for a commit) -- and nothing more:
not whether the tree was dirty, nor where `maddening` was imported from.
Until the runner records those, the session makes them true by procedure:

* **Run from a clean checkout at one commit.**  `git status --porcelain`
  prints nothing, before the first goal and after the last.
* **Import `maddening` from that checkout.**  Print `maddening.__file__`
  (below); it must be under the synced tree, not an older install.
* **One pod per item.**  Every goal that decides a checklist item runs on
  the same pod, in one session: the summary checks that an item's files
  share a commit, not that they share a machine.
* **Check `device_kinds` by eye** in the summary's run table: every line
  of the session names the GPU the pod was rented with.  The same table
  names each file's commit, every checklist line names the commit its
  item closed on, and a directory mixing commits prints `MIXED COMMITS`
  and exits 4 (section 5).

```sh
cd ~/sky_workdir
nvidia-smi -L                                    # four GPUs, else stop (section 3)
pip install -q "jax[cuda12]>=0.10,<0.13" && pip install -q -e .
echo "/results/" >> .git/info/exclude            # the session's output is not the tree
test -z "$(git status --porcelain)" || { git status --short; echo "dirty tree: stop"; }
python -c "import jax, jaxlib, maddening; print(jax.__version__, jaxlib.__version__, jax.devices()); print(maddening.__file__)"
mkdir -p results/multigpu
{ date -u; git rev-parse HEAD; git status --porcelain; python -c "import maddening; print(maddening.__file__)"; \
  nvidia-smi; pip freeze | grep -iE "^(jax|jaxlib|jax-cuda|lineax|equinox|numpy|scipy)"; } \
    > results/multigpu/session.txt 2>&1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest       # float32 matrix products at float32 (below)
env | grep -E "^(JAX|XLA)_" >> results/multigpu/session.txt
set -o pipefail
```

**Float32 matrix products.**  On an NVIDIA card JAX computes float32
matrix products at reduced precision unless told otherwise
(`jax_default_matmul_precision`; this project has seen the default move a
float32 trajectory by 15 % on an RTX A2000, and `highest` remove the
difference).  `LBMNode` computes its momentum and its `e . u` with matrix
products, so the D2Q9 cases of `stencil` and the D3Q19 lattice of the
capacity test are the goals it reaches.  The session asks whether a
sharded run equals the unsharded one, not what that default does, so every
goal runs with `JAX_DEFAULT_MATMUL_PRECISION=highest`, and `session.txt`
records the variable.  What the default does to the lattice is measured
once, in the stress tail, and gates nothing.  On CPU the setting changes
nothing, so no dry run could have shown either number.

A dirty tree is the stop condition of section 3, before anything is
spent on goals.  Run `git status --porcelain` again after the last goal:
it must still print nothing.

Optional: `pip install pymetis` for a real graph partition of the mesh.

### 2b. The goals, in this order, each under its time box

Run one command at a time and read its exit status (`echo $?`) before the
next: **0** = no check failed, go on; **1** = a check failed (the log
names it on a `CHECK FAILED` line); **2** = the runner refused the run
(an option it does not take, more devices than are visible, a device
count or a `--mesh` it cannot use; the reason is the last line);
**5** = a goal raised (its traceback, then a `CRASHED` line);
**124** = the time box ran out; anything else = the process was killed
(137 for an out-of-memory kill).  Anything but 0 is the stop condition in
section 3.  A `CHECK NOT RUN` line is a case this device count cannot
express (the 2-D pencil cases need an even count of at least 4; a halo
from the wrong neighbour shows only on a mesh axis of three or more
devices, which the 1-D meshes give each spatial axis); it is not a
failure and does not stop the session, but it keeps the item open.  On
four GPUs there are none.

```sh
R="python benchmarks/multigpu/run_pod.py --out results/multigpu"
L=results/multigpu
timeout 10m $R --goal indivisible 2>&1 | tee $L/indivisible.log      # checklist 5
timeout 10m $R --goal halo        2>&1 | tee $L/halo.log             # checklist 2
timeout 25m $R --goal coupled     2>&1 | tee $L/coupled.log          # checklist 6 (and 3)
timeout 30m $R --goal stencil     2>&1 | tee $L/stencil.log          # checklist 1 and 3
timeout 20m $R --goal hybrid      2>&1 | tee $L/hybrid.log           # checklist 4
timeout 10m $R --goal exchange    2>&1 | tee $L/exchange.log         # the transport ranking
timeout 15m $R --goal forward     2>&1 | tee $L/forward.log   # add --mesh /path/to/mesh.npz for a real mesh
timeout 20m $R --goal gradient    2>&1 | tee $L/gradient.log         # checklist 1 and 3, unstructured
python benchmarks/multigpu/run_pod.py --summarise results/multigpu | tee $L/summary.txt
```

The checklist goals come first because they are cheap and decisive; the
coupled goal is third because item 6 is the one no other measurement
covers.  `--goal checklist` runs the first five in one process and
`--goal all` all eight; both stop after the first goal whose checks fail,
and an exception in a goal ends the run (`--keep-going` overrides both:
a goal that raises is then recorded as one failed `goal raised` check,
with the exception's type, message and traceback under `raised`, and the
next goal runs; do not use it on the pod).  Defaults on GPUs:
cells `1e5 3e5 1e6` (2-D fields of `ny × (ny + 4)` for the stencil goals,
never square, both a multiple of the device count; for the unstructured
goals' default `--synthetic grid`, the smallest square holding at least
that many cells -- 317² = 100 489, 548² = 300 304 and 1000² -- so no row
has fewer cells than it was asked for: the grid used to round, and the
"1e5" exchange row measured 316² = 99 856 cells and could never decide), 5 warmup + 20 timed
repeats, 20 steps per
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
  goal logs `[SPMD] Involuntary full rematerialization ... f32[N+3]` (the
  field, its two `averages`, the far field): the adjoint's failure check
  (a host callback, present above 50 coupled degrees of freedom) is
  pinned to device 0, and XLA gathers the coupled state vector onto that
  device and scatters it back.  It is expected,
  it is counted in `device0_pinned_ops`, and its cost is inside the
  sharded `value_and_grad` time — record it, do not act on it at the pod.
* **`session.txt`** and **`summary.txt`**, and by hand: the pod type and
  region, the price per hour, the launch and teardown times, and the final
  cost (`job.cost_so_far()` or the provider console).

### Expected durations (estimated for 4×A100, not measurements; the session is on 4×RTX 4090)

Estimated from the sizes and from the CPU dry run, where compile time
dominates; a GPU compile of the coupled group's adjoint is assumed to take
10–30 s.  The time box is what `timeout` enforces.

| goal | what runs | estimate | time box |
|---|---|---|---|
| `indivisible` | 4 refusals; a 1e5-cell uneven unstructured run, 20 public `update()` calls | ~1 min | 10 min |
| `halo` | 5 programs at 1e6 cells (the four meshes, unstructured) and the NumPy reference | 1–2 min | 10 min |
| `coupled` | 2 solvers × (the unsharded group once, and the sharded group on each of the four meshes) at 1e5 cells, and on the pencil at 3e5 and 1e6 -- 18 programs -- and the float64 model on the host | 7–15 min | 25 min |
| `stencil` | rollout and gradient per side: at 1e5 cells the field under each of the three ends and the D2Q9 lattice on each of the four meshes (16 cases, the unsharded side run once per node and ends: 40 programs), and periodic ends on the pencil mesh at 3e5 and 1e6 (18 cases, 48 programs) | 10–17 min | 30 min |
| `hybrid` | `run_scan` and the gradient of the unsharded graph once per size, and of the sharded one on each of the four meshes at 1e5 cells and on the pencil at the others (9 graphs) | 4–8 min | 20 min |
| `exchange` | per size, 2 transports | ~5 min | 10 min |
| `forward` | per size, 2 transports, public and compiled step | ~10 min | 15 min |
| `gradient` | per size, 3 rollout gradients and `sharded_cg` (55 to 58 iterations a solve; the 3000-iteration cap is never reached) | ~8 min | 20 min |

About 55–75 minutes of goals, 1.4–1.9 h with setup and copy-back.
Budget one and three-quarter pod-hours; the whole session is capped at
**2 hours**.  The extra meshes (schema 6) cost about ten minutes of that
estimate; if the cap binds, the larger sizes of `stencil`, `hybrid` and
`coupled` run on the pencil only and are the place to cut (`--cells 100000
300000`), not the meshes.

## 3. Stop condition

**Stop running goals, copy back what exists, and tear the pod down**
(section 4) as soon as any of these happens — do not debug on a metered
pod:

* a goal exits non-zero: `1` (a check failed), `2` (the runner refused
  the run), `5` (a goal raised), `124` (its time box ran out), or anything
  else (the process was killed: an out-of-memory);
* `nvidia-smi -L` lists fewer than four GPUs, or `jax.devices()` does
  not list four CUDA devices;
* `git status --porcelain` prints anything, or `maddening.__file__` is not
  under the synced checkout (section 2a);
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

## After the checklist: the stress tail (not a gate)

Run only when the summary of section 2b has exited 0, the checklist's
results are copied back (section 4, the copy without the teardown), and
the 2-hour cap leaves 30 minutes.  **Nothing here changes the checklist's
verdict**: every command below writes to a directory of its own,
`run_pod.py --summarise results/multigpu` reads none of them, and a
failure here is a finding to reproduce off the pod, not a reason to
reopen an item.  The tail has **30 minutes** from its first command;
when they are up, stop, whatever is left.

| step | expected | time box |
|---|---|---|
| (a) `forward`, `exchange` at 1e7 cells | 1 min each | 2 min each |
| (a) `gradient` at 1e7 cells | 2–3 min | 4 min |
| (a) `forward`, `exchange` at 3e7 cells | 2–3 min each | 4 min each |
| (b) the pencil's ramp, five rungs | about 1 min a rung | 120 s a rung |
| (b) the soak | 3 min, and its last block | 7 min |
| (b) the 1-D mesh, two rungs | about 1 min a rung | 120 s a rung |
| (c) `stencil` at 1e5 cells with the default matmul precision | 1–2 min | 10 min |
| (a) `forward` at 1e8 cells, last and only if time is left | 7–9 min | 9 min |

The expected times are estimates: (a) from the CPU dry run below, (b)
from nothing but the sizes (no accelerator has run it).  The time boxes
add up to more than 30 minutes; the 30-minute rule is what holds.

Before the session, on the laptop (about 15 s on four cores):

```sh
JAX_PLATFORMS=cpu python benchmarks/multigpu/run_capacity.py --dry-run \
    --cells 8000 30000 100000 --soak-minutes 0.02 --out /tmp/capacity-dry
python benchmarks/multigpu/run_capacity.py --summarise /tmp/capacity-dry    # needs no JAX
```

It must print three `passed` rungs with the fill as `n/m` (CPU keeps no
device memory statistics: the fill is *not measured* there, a check not
run, never a pass), `SOAK: ... passed`, and exit 0 both times.

### (a) The runner at larger sizes

The unstructured goals at ten, thirty and a hundred times the gate's
largest size, with the runner's existing options only.  One size and one
goal per command, each size in a directory of its own (a goal's file is
written when the goal ends: a run killed at 1e8 must not take the 3e7
result with it); stop at the first non-zero exit.

```sh
T=results/stress; mkdir -p $T
big() { timeout "$1" python benchmarks/multigpu/run_pod.py --goal "$2" --cells "$3" \
            --partition contiguous --warmup 1 --repeats 3 \
            --out $T/cells-$3 2>&1 | tee $T/$2-$3.log; }
big 2m forward  10000000;  echo $?
big 2m exchange 10000000;  echo $?
big 4m gradient 10000000;  echo $?
big 4m forward  30000000;  echo $?
big 4m exchange 30000000;  echo $?
# then section (b); and only if the 30 minutes are not up after it:
big 9m forward  100000000; echo $?
```

At 3e7 and 1e8 cells the `forward` goal prints one `CHECK NOT RUN` a size
and `INCOMPLETE`, and still exits 0: its exact count of the cells is a
float32 sum of ones, exact only under 2**24 = 16,777,216 cells, and is
not run at or over that.  The field is compared entry by entry at every
size, and the 1e7 run counts its 10,004,569 cells.

**What bounds it is the host's RAM, not the cards.**  The unstructured
path builds its mesh, its partition and its layout on the host and
indexes with int32.  Measured in the CPU dry run (four cores, jaxlib
0.11.0, the dry run's three steps), `--goal forward --dry-run --cells
10000000` -- 10,004,569 cells, the smallest square that holds them --
peaked at 3.41 GiB of host memory (3,577,584 kB) in 43 s with the default
partitioner (reverse Cuthill-McKee there: PyMetis was not installed) and
at 3.35 GiB (3,511,632 kB) in 41 s with `--partition contiguous`: about
**360 bytes of host memory per cell** all told, either way, and about 4 s
per million cells.  At that rate 3e7 cells need about 10 GiB and 2
minutes and 1e8 about 34 GiB and 7 minutes, before the timed steps; and a
pod should not be asked for more cells than half its RAM holds: 90
million on 64 GiB, 180 million on 128 GiB.  Check `free -g` first, and
leave the 1e8 run out on a pod with less than 70 GiB.  The commands pass
`--partition contiguous` because PyMetis at these sizes was not measured,
and `--warmup 1 --repeats 3` because twenty timed repeats of twenty steps
are for the ranking at the gate's sizes, not for this.

**`gradient` is in the tail at 1e7 cells, and no further.**  At 10,004,569
cells in the CPU dry run (the command above with `--dry-run
--cg-max-iters 3000`: a dry run alone caps `sharded_cg` at 300; sixteen
cores, jaxlib 0.11.0) all sixteen of its checks pass in 154 s at a peak of 4.30 GiB of host memory (4,504,536 kB),
about 450 bytes a cell: the rollout gradients agree to 1.3e-7 against
1e-5, and each side's three `sharded_cg` solves stop after 56 iterations
with true residuals of 8.8e-5 to 8.9e-5 (limit 2e-4) and derivatives that
agree to 7.5e-7 and 4.4e-7 against 4e-4, as at the gate's sizes (the
shifted system of the checklist's item 3 has the same condition number at
every size).  It was not run at 3e7 cells, about 13 GiB of host memory at
that rate, and is not asked for there.

### (b) The capacity ramp: a sharded run larger than one card

`run_capacity.py` asks what no goal can: every goal compares with the
unsharded node on one device, so none can be larger than one card.  It
steps a periodic D3Q19 lattice (`LBMNode` in `ShardedStencilNode`) that
is the same small tile repeated, and compares it with the tile's own
unsharded run, tiled: by translation symmetry the two are equal at every
step, and nothing unsharded is ever needed at full size.  The field is
built and compared one device's block at a time, on the devices; the
whole of it never exists on one card or on the host.  Its header gives
the tiling rules that make a wrong halo show (an odd number of tiles
along every split axis, sharing no factor with the devices there).

```sh
C="python benchmarks/multigpu/run_capacity.py --device-memory-gb 24 --rung-timeout-s 120"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95       # with PREALLOCATE=false of section 2a
timeout 18m $C --mesh 2x2 --ramp 0.25 0.5 0.75 0.85 0.9 --soak-minutes 3 \
    --out results/capacity-2x2 2>&1 | tee results/capacity-2x2.log;  echo $?
timeout 5m  $C --mesh 4x1 --ramp 0.25 0.85 \
    --out results/capacity-4x1 2>&1 | tee results/capacity-4x1.log;  echo $?
python benchmarks/multigpu/run_capacity.py --summarise results/capacity-2x2 | tee results/capacity-2x2.txt
python benchmarks/multigpu/run_capacity.py --summarise results/capacity-4x1 | tee results/capacity-4x1.txt
```

Each rung is a process of its own under `--rung-timeout-s`, so an
out-of-memory loses nothing and leaves no allocator state behind (and a
ramp that `timeout` ends takes its rung with it).  The
first rung's size is a guess (97 bytes of state per cell times 6: about
44 million cells at 0.25 of four 24 GiB cards); every later rung is
sized from the peak bytes per cell the rung before it measured.  The
fill is the largest `peak_bytes_in_use` of `jax.devices()[i].memory_stats()`
over the 24 GiB given, over the four cards.  The soak then repeats
build, ten steps and comparison at 0.75 of the ceiling's cells for three
minutes and reports how the memory moved from block to block.

A rung ends as exactly one of:

| outcome | meaning | the ramp |
|---|---|---|
| `passed` | every field within 1e-5 of the tiled reference (the runner's forward limit), the mass that of the tile times the number of tiles, every value finite | goes on |
| `check failed` | a number is wrong: the one outcome that is a defect | stops, exit **1** |
| `out of memory` | the allocator refused (the record keeps its message), or the process was killed (137) | stops: this is the ceiling's reason |
| `timed out` | the rung's time box ran out | stops |
| `crashed` | anything else; the record keeps the end of its stderr | stops |
| `refused by the host guard` | the rung never started: the host would have had to hold more than half its available memory | stops |

and the script exits **0** when every rung that reached its checks
passed them (an out-of-memory or a time-out above a passing rung is the
result), **1** when a check failed, **2** when it refused its options,
**6** when the first rung did not pass (no ceiling), **5** if it raised
itself; `--summarise` re-derives every verdict from the recorded values
and exits **3** for a record that disagrees with itself.  The last lines
name the ceiling: the largest size that passed, its fill, and what
stopped the next rung.

**The allocator, and what it does to the ceiling.**  With
`XLA_PYTHON_CLIENT_PREALLOCATE=false` (section 2a; the script sets it
where it is unset) the pool takes memory from the card as the run asks
for it and never gives it back or joins two pieces of it, so a block can
be refused while the sum of what is free would hold it: the ceiling the
ramp finds is the ceiling of a growing pool, which is what a run has by
default here, and may be below what one preallocated pool would take.
The pool is also understood to have an upper limit of its own,
`XLA_PYTHON_CLIENT_MEM_FRACTION` of the card's free memory, 0.75 where
unset -- below the ramp's last two targets -- which is why the commands
export 0.95 (the script sets that too where it is unset).  JAX documents
the fraction for a preallocated pool; that it also bounds a growing one
is read from the allocator, and neither effect was measured before the
session.  Each rung records the three variables as
found and as used and every card's `bytes_limit`, and the summary prints
the allocator's own limit beside the ceiling: **an out-of-memory at a
fill just under `bytes_limit` over 24 GiB is the allocator's limit, not
the card's.**

Read in the first rung's record before trusting the rest (nothing below
has run on an accelerator before): `environment.platform` is `gpu` and
`device_kinds` names the card; `memory.readings` hold numbers on all four
devices; `memory.bytes_limit` is about 0.95 of what the card had free;
`memory.fill` against the 0.25 asked for says how good the factor 6 was; and
`results.fields` gives each field's `max_rel` and whether it was exactly
0 (in the CPU dry run it is, at the default tile).  Device 0 also holds
a bool per cell that `LBMNode`'s constructor builds there, so it is the
fullest by about a byte per cell.  The lattice is `LBMNode` with two methods the wrapper calls
at construction answered by the script, because `ShardedStencilNode`
otherwise builds the whole grid on the default device before placing a
block (MADD-ANO-262).

### (c) The default matmul precision, once

Every goal ran with `JAX_DEFAULT_MATMUL_PRECISION=highest` (section 2a).
This is the same `stencil` goal with the variable unset, at the smallest
size, into a directory of its own:

```sh
env -u JAX_DEFAULT_MATMUL_PRECISION timeout 10m python benchmarks/multigpu/run_pod.py \
    --goal stencil --cells 100000 --out results/stress/stencil-default-precision 2>&1 \
    | tee results/stress/stencil-default-precision.log;  echo $?
```

Any exit status is a result here.  Read the `lbm` rows against the gate's
own `stencil.json`: whether sharded still equals unsharded at the default
precision (the two sides round the same products), and how far the
lattice's velocities are from the ones computed at `highest`.

**Copy back** `results/stress/`, `results/capacity-2x2/`,
`results/capacity-4x1/` and the `*.log` and `*.txt` beside them, with
the rest (section 4), and store them under
`benchmarks/results/multigpu-stress/`, apart from the checklist's files.

## 5. Read the result

`python benchmarks/multigpu/run_pod.py --summarise benchmarks/results/multigpu`
exits 0 when no recorded check failed and every file records one commit,
3 when any check failed or a file cannot decide (below), 4 when nothing
failed but the files come from more than one commit, or any file records
none -- every file recording none included, which used to exit 0; a recorded
commit that is not a full SHA counts as none -- (below), and 1 when the
directory holds no goal JSON.  A file this runner cannot read in full (an
older runner's) reads `INVALID` and is left out of the tables, which say
so; it used to stop the summary on a traceback.  So does a file named for
a goal (`halo_rerun.json` beside `halo.json`) that records another goal,
or none: it used to be dropped without a word.  A goal that raised under
`--keep-going` reads `FAIL`, its one `goal raised` check naming the
exception; its record is valid only with no results and exactly that
check.  It does not
take a check's recorded `passed` on trust: pass/fail is re-derived from
the check's `value`, `limit` and `sense`, and a record that disagrees --
a value of 0.5 against a limit of 0.0 recorded as passed, say -- is
listed as failed (`recorded passed=True, but the value fails its limit`),
as is a file whose top-level `passed` its checks do not bear out.

Nor does it take a file's word for *what* was checked.  A file is
evidence only if it is what `run_pod.py`, as it stands, would have
written; otherwise its goal reads `INVALID` and it closes nothing.  A
file must:

* be on the current `schema_version` (9);
* record an `n_devices` no larger than the devices its `environment`
  lists (`n_devices_visible`, which must count `devices`), and the same
  `n_devices` in its `config` and in every result entry;
* hold every case the runner runs for the file's own `config` (`cells`,
  `n_devices`, `synthetic`, `mesh`) and no other: every boundary mode ×
  width × mesh of `halo`, every node × ends × mesh case of `stencil`,
  the mesh of `hybrid` and `coupled`, both solvers of `coupled`, both
  transports, ...;
* carry exactly the checks the runner derives from its results -- the
  same names, values, limits, senses and flags.  So every limit is the
  one in `LIMITS` (a limit loosened, or tightened, in the file is
  refused), a results table that disagrees with a check is refused, and
  a check deleted from the file or added to it is refused.  The order of
  the checks does not matter.

And the files that decide one checklist item must all record the same
`git_commit` (and must record one).  A directory holding goals from two
commits keeps the items they share open; re-run those goals on one
commit.  That rule is per item, so items decided by different goals can
each close on a different commit -- the directory a session leaves when it
re-runs some goals on a fix commit and keeps the earlier passing files.
So every checklist line names the commit its item's files record (each
commit with its files, when they disagree), and a directory whose files
come from more than one commit -- or where any records none, all of them
included -- prints `WARNING: MIXED COMMITS`, naming each commit with the items it decides and
its files, repeats the warning as the last line, and exits 4 (3 if a check
failed as well).  Read such a checklist as several sessions, not one.  If the runner itself changed between the session and the
summary (a new check, a new case), the session's files no longer match
it and read `INVALID`: summarise with the commit the session ran, which
every file records.  The transport ranking uses the rows of an
`exchange.json` only when that file reads `PASS`.

It prints:

* **Runs**: one line per JSON file — platform, devices, device kind,
  `jax / jaxlib`, dry run, checks passed, commit (12 characters), verdict
  (`PASS`, `FAIL`, `INVALID`, `INCOMPLETE`).
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
  commit recorded in <file>`, then the commit its files record.  The
  session succeeded when all six read `CLOSED` and the summary exits 0.
* **Records that cannot decide**: every reason a file is `INVALID`, and
  every item whose files come from more than one commit.
* **`WARNING: MIXED COMMITS`**, when the directory's files come from more
  than one commit: each commit, the items it decides and its files.
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
run (not `--dry-run`) whose file reads `PASS`, was requested at ≥ 1e5
cells (its `requested_cells`; the `asked` column) and measured at no fewer
(`cells`), ran on ≥ 4 devices (≥ 2 if that run recorded
`allow_fewer_devices`) and has a finite speedup, which the summary
derives from the two medians rather than reading the file's
`ppermute_speedup_median` (a 0 ms median is below timer resolution).  The
`decides` column of the table says which rows decide; every row that does
not is listed under **Rows that do not decide** with what it lacked
(`WARNING:` for a real-GPU row), and when some rows decide and others do
not, the recommendation line says "decided on N of M row(s)".

The timings are secondary to the checks but worth reading:
`wrapper_step` (public `update()`, which re-uploads the partitioned
static arrays every call) against `device_step` (the compiled step) in
`forward`; the gradient timings in `gradient` and `stencil`, which are
like-for-like (`jax.jit(jax.grad(...))` on both sides, statics placed
once, compile time in `compile_s`), so "the sharded grad is N× the
unsharded one" is a statement about the compiled step, not about Python;
and in `coupled`, the sharded against the unsharded `value_and_grad`
time, which is where the device-0 gather above shows its cost at scale.

## Schema of the JSON (schema_version 9)

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

* `indivisible.json` results (one entry): `stencil`, `stencil_axis1`
  (the 1-D mesh sharding spatial axis 1, one column too many),
  `pointwise` and, on an even count ≥ 4 devices, `pencil` = `{shape,
  raised, message}` (otherwise a check not run) (`stencil` and
  `stencil_axis1` also `divisible_shape`, `divisible_raised`,
  `divisible_message`: one row, or column, fewer is accepted);
  `unstructured` = `{cells, partition, cells_per_device, steps,
  parity_x}`.
* `halo.json` results (one entry): `stencil_cases` = `[{mesh,
  mesh_shape, shape, halo, boundary, forward_max_abs, adjoint_max_abs}]`
  (`mesh` one of `1d`, `1d-axis1`, `2d-flat`, `2d`; `mesh_shape` the
  devices along spatial axes 0 and 1),
  `unstructured` = `{cells, partition, n_local_max, n_ghost_max,
  methods.<m>.{forward_max_abs, adjoint_max_abs}}`.
* `coupled.json` results, one per mesh and size: `mesh` (`1d`,
  `1d-axis1`, `2d-flat` or `2d`), `mesh_shape`, `cells`,
  `shape`, `steps`, `coupled_dof`, `max_iterations`, `tolerance`,
  `parameters`, `model` (`loss`, `grad` and `wall_s` of the float64
  model), `solvers.<ift|fori>` = `{sharded, unsharded}` (`compile_s`,
  `value_and_grad` timing with `ms_per_step`, `loss`, `grad`,
  `last_step_iterations` (`null` under `"fori"`), `partitioned`,
  `device0_pinned_ops`), `parity_f`, `parity_averages`, `parity_u`,
  `parity_loss`, `parity_grad`, `model.<side>.{f, averages, u, loss,
  grad}`.
* `stencil.json` results, one per case: `node` (`field` or `lbm`), `mesh`
  (`1d`, `1d-axis1`, `2d-flat` or `2d`), `mesh_shape`, `cells`, `shape`,
  `boundary`
  (`periodic`, `edge` or `dirichlet`), `steps`, `grad_steps`,
  `parameter` (`diffusivity` or `viscosity`), `input_partitioned`,
  `forward.{sharded,unsharded}` (`compile_s`, `rollout` timing with
  `ms_per_step`), `forward.parity.<field>` (`f` and `averages`, or `f` and
  `velocity`), `gradient.{sharded,unsharded}` (`compile_s`, `grad`
  timing, `loss`, `grad_parameter`), `gradient.parity_loss`,
  `gradient.parity_grad_initial_field`, `gradient.parity_grad_parameter`.
* `hybrid.json` results, one per mesh and size: `mesh`, `mesh_shape`,
  `cells`, `shape`, `steps`, `grad_steps`, `correction_shift` (rows,
  columns), `{sharded,unsharded}`
  (`run_scan_first_call_s`, `partitioned`, `compile_s`, `value_and_grad`
  timing, `loss`, `grad`), `correction_rel`, `parity_f`,
  `parity_averages`, `parity_loss`, `parity_grad`.
* `exchange.json` results: `cells` (measured), `requested_cells` (the
  `--cells` entry it was measured for), `mesh`, `partition`, `n_devices`,
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
  `ms_per_step`), `parity_x`, `parity_total` (information: no check
  reads its `max_rel`), `ones` = `total`, `max_abs_from_one`,
  `entries_not_one` (the step from a field of ones); and
  `ones_unsharded`, the same three for the unsharded node.  At 2**24
  cells or more neither `ones` is recorded and the count is one check not
  run.
* `gradient.json` results: `cells`, `mesh`, `partition`, `n_devices`,
  `grad_steps`, `rollout.{unsharded,all_to_all,ppermute}` (`grad`
  timing and `compile_s`; `input_presharded` and `parity` for the
  sharded ones), `sharded_cg` (`dof`, the system solved: `shift`, `rtol`,
  `max_iters`; `input_presharded`, `grad_sharded`, `grad_unsharded`,
  `compile_s.{sharded,unsharded}`, `grad_parity`, `jvp_parity`, and
  `solve.{sharded,unsharded}` = `converged` and `iterations` (the loop's
  own count, which the goal holds under `max_iters`, and the same call's
  flag, recorded as information), `true_residual.{solve,adjoint,tangent}`).

Schema 8 files held the `forward` goal's float `total` to 1e-5 of
itself and did not count the cells: they record no step from a field of
ones.  Schema 7 files ran the `gradient` goal's `sharded_cg` part on the
unshifted operator with a smooth right-hand side and `rtol=1e-6`, recorded
neither the system nor its solves, and checked parity only: neither side
had converged.  Schema 5 files ran the wrapper goals on the 1-D (axis 0) and the pencil
mesh only (the graph goals on the pencil only), on square grids, with a
source of period 1/2 in x and the coupled goal's `ambient` uniform; on
four devices no case split spatial axis 1 over more than two devices.
The unsharded side of every case is now run once per grid and shared by
every mesh's case.  Schema 4 files ran the stencil goals on a 1-D mesh only, with a node that
read its static in the interior only and took no grid-shaped input or
domain integral.  Schema 3 files carry no `sense` (the summary reads it
from the limit's type, which is how schema 3 wrote checks), never record a check as not
run, and ran the stencil goal with periodic ends only.  Schema 2 files
(the first three goals, before the checklist goals) carry
no `checks`, `passed` or top-level `n_devices`; the summary shows them as
`no checks` and they close no checklist item.  Schema 1 files lack the
`compile_s`/`input_presharded` keys and carried gradient timings that were
not like-for-like; nothing under `benchmarks/results/` was written with
either.
