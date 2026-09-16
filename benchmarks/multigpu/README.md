# Multi-GPU hardware session — runbook

Human-driven, one sitting, one pod.  Nothing in this directory launches
anything: `run_pod.py` only runs *on* the machine that has the GPUs and
writes JSON.  Provisioning, copying results back and tearing down are
the steps below, done by hand with the existing helpers.

## What the session settles

| goal | question | decides |
|---|---|---|
| `exchange` | NCCL time of `exchange_unstructured(method="all_to_all")` vs `"ppermute"` at 1e5, 3e5, 1e6 cells on 4 GPUs | whether `ppermute` becomes the default `exchange=` of `ShardedUnstructuredNode` (an API default: change it before the stability freeze or not at all) |
| `forward` | a 1e6-cell `ShardedUnstructuredNode` forward run on a real unstructured mesh, both transports, checked against the unsharded node | the v0.4.0 "real-mesh size" commitment (`docs/developer_guide/sharding_topology.md`) |
| `gradient` | `jax.grad` through a sharded rollout (both transports) and through the Jacobi-preconditioned `sharded_cg` at 1e5–1e6 DOF vs the unsharded references | the real-GPU half of the gradient-parity-at-scale gate (the CPU-virtual half is `tests/cloud/multigpu/test_iterative_solver.py::TestGradientParityAtScale`) |

Correctness and byte counts are already settled on CPU virtual devices
(`tests/cloud/multigpu/test_exchange_ppermute.py`); the session buys the
*timings* and the real-mesh run.  Budget: one pod, 4×A100 (80 GB or
40 GB both fine), about 1.5 h including setup; stop the pod as soon as
the JSON is copied back.

## 0. Before spending anything (laptop, free)

```sh
cd MADDENING
JAX_PLATFORMS=cpu python benchmarks/multigpu/run_pod.py --goal all --dry-run --out /tmp/mg-dry
python benchmarks/multigpu/run_pod.py --summarise /tmp/mg-dry     # prints "undecided"
pytest tests/cloud/multigpu -m "slow or not slow" -q             # includes the runner dry-run test
```

If MIME's helix-in-vessel mesh is to be used for `forward`, export it
now as an `.npz` with `edges` (`(n_edges, 2)` int, cell–cell adjacency,
global ids) and optionally `partition` (`(n_cells,)` int in `[0, 4)`,
e.g. from PyMetis).  Without `partition` the runner partitions with
PyMetis if installed on the pod, else reverse-Cuthill-McKee blocks
(SciPy), else contiguous cell ids.  Keep the file under a few hundred
MB; it is copied to the pod with the source tree.

## 1. Launch (by hand, with consent)

Use the existing launcher, not a new script.  Two equivalent routes:

* **`maddening.cloud.launcher.CloudLauncher` + `JobConfig`** from a
  Python REPL: `provider="runpod"`, `gpu_type` an A100 name from
  `sky show-gpus --cloud runpod` (`A100-80GB-SXM` or `A100-80GB`),
  `gpu_count=4`, `use_spot=False` (a preempted benchmark is a wasted
  hour), `workdir=<MADDENING checkout>` so the tree is synced, and a
  `CostPolicy(max_cost_per_hour=12.0, max_total_budget=40.0,
  autostop_minutes=20, auto_teardown=False)`.  `run` can be
  `"sleep 7200"`; the benchmark itself is driven over SSH with
  `CloudJob.ssh_run`.  Credentials come from
  `~/.maddening/cloud_credentials.yaml` (`CloudLauncher._load_credentials`),
  the RunPod key lands in `~/.runpod/config.toml` only for the duration
  of the launch (`_credential_context`).
* **SkyPilot directly** (`sky launch` with `--gpus A100-80GB:4
  --cloud runpod --no-use-spot --workdir .`), if you prefer the CLI.
  `maddening.cloud._skypilot.launch_vm` / `teardown_vm` are the thin
  wrappers `CloudSession` uses; they work on a `CloudConfig` but the
  runbook does not need them.

`src/maddening/examples/cloud/multigpu/09_real_gpu_benchmark.py` shows
the launch → `ssh_run` → copy-back → `teardown` shape end to end (it is
the 2-GPU Jacobi-coupling crossover benchmark; do not run it in this
session, it is a different measurement).

Sanity-check before benchmarking, on the pod:

```sh
nvidia-smi -L                                  # four A100s
pip install -q "jax[cuda12]>=0.10,<0.13" && pip install -q -e ~/sky_workdir
python -c "import jax; print(jax.devices())"   # [CudaDevice(id=0..3)]
```

Optional: `pip install pymetis` for a real graph partition of the mesh.

## 2. Run (on the pod, ~30–45 min)

```sh
cd ~/sky_workdir
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python benchmarks/multigpu/run_pod.py --goal exchange --out results/multigpu        # ~5 min
python benchmarks/multigpu/run_pod.py --goal forward  --out results/multigpu \
       --mesh /path/to/helix_mesh.npz                                                # ~10 min
python benchmarks/multigpu/run_pod.py --goal gradient --out results/multigpu        # ~15 min
python benchmarks/multigpu/run_pod.py --summarise results/multigpu
```

`--goal all` does the three in one go (synthetic grid mesh for
`forward`; add `--mesh` for the real one).  Defaults on GPUs: cells
`1e5 3e5 1e6`, 5 warmup + 20 timed repeats per point, 20 steps per timed
block, `--n-devices 4`.  `--fields N` makes the exchange payload `N`
float32 per cell (use the state width of the node you care about, e.g.
5 for a compressible FVM state) — bytes moved scale with it, message
counts do not.  Re-run `exchange` with `--fields 5 --out
results/multigpu-f5` if time allows; the summary ranks each directory
separately.

Every run prints one line per measurement, so a hang is visible; the
JSON is written per goal at the end of that goal.  If a goal fails,
re-run only that goal.  Everything is deterministic (fixed seeds).

## 3. Copy back, then stop the pod

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

## 4. Read the result

`python benchmarks/multigpu/run_pod.py --summarise benchmarks/results/multigpu`

* **Recommendation `ppermute`**: median ppermute time beats all_to_all by
  ≥ 1.05× at every real-GPU point ≥ 1e5 cells → change the default
  `exchange=` of `ShardedUnstructuredNode` and `exchange_unstructured`
  to `"ppermute"` (one-line change each, plus the docs paragraph in
  `sharding_topology.md` and a CHANGELOG entry under Changed).
* **`all_to_all`**: it is faster at some hardware-sized point → keep the
  default, record the numbers in the docs paragraph.
* **`tie`**: keep `all_to_all` (one collective) unless the byte savings
  (`bytes_total` columns) matter for the target mesh; write that down.
* **`undecided`**: the directory holds no real-GPU measurement at
  ≥ 1e5 cells — nothing to decide on.

For `forward`, `parity_x.max_rel` should sit at float32 round-off
(≤ 1e-5) and both transports should report finite results at 1e6 cells;
`wrapper_step` (public `update()`, which re-uploads the partitioned
static arrays every call) vs `device_step` (the compiled step) shows how
much of the per-step cost is host-side.  For `gradient`, the rollout
parity should be ≤ 1e-5 relative and the `sharded_cg` grad/jvp parity
≤ 1e-3, matching the CPU-virtual gate.

## Schema of the JSON (schema_version 1)

Common: `goal`, `dry_run`, `environment` (`hostname`, `timestamp_utc`,
`python`, `jax`, `jaxlib`, `platform`, `devices`, `device_kinds`,
`n_devices_visible`, `nvidia_smi`, `xla_flags`, `jax_platforms`,
`git_commit`), `config` (the CLI namespace), `wall_s`, `results` (one
entry per cell count).  Timings are `{warmup, repeats, ms: [...],
min_ms, median_ms, mean_ms}`; parity blocks are `{max_abs, max_rel,
reference_scale, finite}`.

* `exchange.json` results: `cells`, `mesh`, `partition`, `n_devices`,
  `fields_per_cell`, `n_local_max`, `n_ghost_max`, `layout_build_s`,
  `traffic_cells_per_shard` (`exchange_traffic()` output),
  `methods.{all_to_all,ppermute}` = timing + `compile_s`,
  `bytes_per_shard`, `bytes_total`, `messages`, `bandwidth_GBps`;
  `bit_identical`, `ppermute_speedup_median`, `ppermute_speedup_min`.
* `forward.json` results: `cells`, `mesh`, `partition`, `n_devices`,
  `steps`, `n_local_max`, `n_ghost_max`, `traffic_cells_per_shard`,
  `methods.<m>` = `compile_s`, `wrapper_step`, `device_step` (timings
  with `ms_per_step`), `parity_x`, `parity_total`.
* `gradient.json` results: `cells`, `mesh`, `partition`, `n_devices`,
  `grad_steps`, `rollout.{unsharded,all_to_all,ppermute}` (`grad`
  timing, `parity` for the sharded ones), `sharded_cg` (`dof`,
  `grad_sharded`, `grad_unsharded`, `grad_parity`, `jvp_parity`).
