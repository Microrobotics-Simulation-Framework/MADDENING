# 4 × RTX PRO 6000 Blackwell, 2026-10-12: the multi-GPU hardware session

**Verdict: all six items of the sharding checklist CLOSED, on commit
`c0c6add8`, on four real GPUs** (`multigpu/summary.txt`).  Every goal of
section 2b exited 0.  The procedure is the runbook,
`benchmarks/multigpu/README.md`; this file is the record of one session.
Every number below names the file it comes from.

## What it ran on

* **Pod:** RunPod, US, 4 × NVIDIA RTX PRO 6000 Blackwell Server Edition,
  97,887 MiB each, driver 595.91.07 (`multigpu/session.txt`, the
  `nvidia-smi` block).  PCIe only, no NVLink: GPU 0-1 and GPU 2-3 each sit
  behind one PCIe switch (`PIX`), and the two pairs are `NODE` (read with
  `nvidia-smi topo -m` on the pod; not in a file here).  Host: 1,511 GB of
  RAM (`free -g` on the pod).
* **Launch:** `maddening.cloud.launcher.CloudLauncher`, 4 cards on demand
  (`use_spot=False`), `auto_teardown=True`, the image built from the
  session's commit (`maddening-cloud:c0c6add8...`).  Up at 15:31Z, torn down
  at 17:16:38Z, about 1 h 45 min at $9.96/h, about $17.4.  These were
  recorded by hand from the provider: the launcher's `cost_so_far()` read
  0.0 for the job.
* **Checkout:** a fresh clone of the public repository on the pod,
  `git checkout --detach c0c6add8`, clean before the first goal, and
  `maddening` imported from it (`multigpu/session.txt`: the commit, an
  empty `git status --porcelain`, `maddening.__file__`).
* **Software:** Python 3.12.3; jax, jaxlib, jax-cuda12-plugin and
  jax-cuda12-pjrt 0.11.2; lineax 0.1.1; equinox 0.13.8; numpy 2.5.4; scipy
  1.18.1.  `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
  `JAX_DEFAULT_MATMUL_PRECISION=highest` (`multigpu/session.txt`, last
  lines; `scripts/session_env.sh`).

## Layout

The pod's `~/sky_workdir/results/` is this directory's root, so the
runbook's paths read the same here.  The two additions wrote beside the
checkout and keep their names.

| here | what | from |
|---|---|---|
| `multigpu/` | the eight goals of section 2b: `<goal>.json`, `<goal>.log`, `session.txt`, `summary.txt`, `progress.txt` (start, end and exit status of each goal) | `results/multigpu/` |
| `multigpu-mesh/` | `forward` on the real mesh (1,001,664 tetrahedral cells), and its own summary | `results/multigpu-mesh/` |
| `multigpu-f5/`, `multigpu-f5.log` | `exchange --fields 5` | `results/multigpu-f5/` |
| `stress/` | the tail's part (a), part (c), and their `progress.txt` | `results/stress/` |
| `capacity-2x2/`, `capacity-4x1/` and their `.log`, `.txt` | the capacity ramps (part (b)) and `--summarise` of each | `results/capacity-*` |
| `precision_fix_results/` | the LBM precision patch and `stencil` run with it | `~/precision_fix_results/`, `~/lbm_matmul_precision.patch` |
| `gve_results/` | graph against eager, and `profile_graph` | `~/gve_results/`, `~/graph_vs_eager.py` |
| `scripts/` | the shell scripts that ran each part, as run (two comment lines reworded to name no person), `session_env.sh`, the additions' progress | `~/` |

Not committed: the profiler traces under `gve_results/profile/*/plugins/`
(1.3 GB; the reports beside them are), the mesh file, and the logs of the
WebRTC streaming check that ran between section 2b and the tail (it passed;
it is not part of this benchmark).

## In order

| UTC | step | script | result |
|---|---|---|---|
| 15:33:43-15:50:43 | section 2b, the runbook's order and time boxes, stopping at the first non-zero exit | `scripts/run_goals.sh` | every goal exit 0 (`multigpu/progress.txt`) |
| 15:50:43 | `--summarise` of both directories | same | exit 0 both; items 1-6 CLOSED (`multigpu/summary.txt`) |
| 16:01:28-16:26:59 | the stress tail | `scripts/run_tail.sh` | two steps timed out; see below (`stress/progress.txt`) |
| 16:27:00-17:07:42 | the timed-out steps again, longer boxes, unbuffered | `scripts/run_tail2.sh` | all passed (`stress/progress.txt`) |
| 17:08:55-17:12:26 | the precision patch; graph against eager and `profile_graph` | `scripts/run_extra.sh` | exit 0 both (`scripts/extra_progress.txt`) |

## Section 2b: the checklist

| goal | checks | wall | box | file |
|---|---|---|---|---|
| `indivisible` | 13/13 | 18 s | 10 min | `multigpu/indivisible.json` |
| `halo` | 52/52 | 39 s | 10 min | `multigpu/halo.json` |
| `coupled` | 210/210 | 4 min 51 s | 25 min | `multigpu/coupled.json` |
| `stencil` | 376/376 | 2 min 40 s | 45 min | `multigpu/stencil.json` |
| `hybrid` | 54/54 | 1 min 3 s | 20 min | `multigpu/hybrid.json` |
| `exchange` | 6/6 | 33 s | 10 min | `multigpu/exchange.json` |
| `forward` | 42/42 | 1 min 41 s | 20 min | `multigpu/forward.json` |
| `gradient` | 72/72 | 2 min 12 s | 20 min | `multigpu/gradient.json` |
| `forward --mesh` | 14/14 | 3 min 1 s | 25 min | `multigpu-mesh/forward.json` |

Walls are the process's, from `multigpu/progress.txt`: 17 minutes of goals
against the runbook's estimate of 70-109.

What the summary shows (`multigpu/summary.txt` unless named):

* **Halo:** every one of the 24 stencil cases and both unstructured
  transports bit for bit against NumPy, forward and adjoint.
* **Coupled:** the gradient of the sharded group within 5.8e-7 (max-rel) of
  the unsharded one on every mesh and size, against limits of 1e-4 (IFT)
  and 1e-5 (`"fori"`); within 2.3e-6 of the float64 model.
* **Stencil:** fields and integrals to 1.9e-6 at worst; the D2Q9 lattice's
  gradients 2.1e-6 (initial `f`) and 1.6e-6 to 1.8e-6 (viscosity), at
  `highest`.
* **Gradient:** each `sharded_cg` solve stopped on its tolerance after 56
  iterations of 3,000, true residual at most 8.97e-5 (limit 2e-4); the
  derivative within 6.8e-7 (`multigpu/gradient.log`).
* **Exchange:** `ppermute` faster at 1e5 and 3e5 cells (×1.06, ×1.04),
  `all_to_all` at 1e6 (×0.83), results identical; recommendation: keep
  `all_to_all` the default.  With five fields (`multigpu-f5.log`)
  `all_to_all` is as fast or faster at every size (×0.99, ×0.84, ×0.85).
* **The real mesh:** 1,001,664 cells, 14/14, field within 8.34e-7
  (`multigpu-mesh/forward.log`).
* `forward`'s sharded field differs from one card's by 0 at 1e5 cells,
  4.77e-7 at 3e5, 7.15e-7 at 1e6 and 8.34e-7 on the mesh (max |dx|,
  `multigpu/forward.log`, `multigpu-mesh/forward.log`): float32, and other
  GPU programs on the two sides.  On CPU the same comparison is bit for bit.
  Every limit holds.

## The stress tail (gates nothing)

`scripts/run_tail.sh` departs from the runbook of the session's commit in
four ways, each in its header: `--device-memory-gb 96` (the runbook said
24, for a 24 GB card), `--rung-timeout-s 360` with a 40-minute box for the
2×2 ramp, no 30-minute limit on the tail (the 2-hour cap it came from was
lifted for this session), and `exchange --fields 5` first.

### (a) The runner at larger sizes (`--partition contiguous --warmup 1 --repeats 3`)

| goal, cells | box | exit | wall | result | file |
|---|---|---|---|---|---|
| `forward` 1e7 (10,004,569) | 2 min | 0 | 96 s | 14/14; device 3.39 / 4.27 ms a step | `stress/cells-10000000/forward.json` |
| `exchange` 1e7 | 2 min | 0 | 100 s | 2/2; `ppermute` ×2.02 faster | `stress/cells-10000000/exchange.json` |
| `gradient` 1e7 | 4 min | **124** | - | timed out; empty log | `stress/gradient-10000000.log` |
| `forward` 3e7 | 4 min | **124** | - | timed out; empty log | `stress/forward-30000000.log` |
| `exchange` 3e7 (30,008,484) | 4 min | 0 | 132 s | 2/2; `ppermute` ×1.13 | `stress/cells-30000000/exchange.json` |
| `forward` 1e8 | 9 min | 143 | - | stopped at once by hand for the rerun below | `stress/forward-100000000.log` |
| `gradient` 1e7, rerun | 15 min | 0 | 418 s | 24/24; rollout gradient 7.43 ms sharded, 7.91 unsharded | `stress/gradient-10000000-rerun.log` |
| `forward` 3e7, rerun | 15 min | 0 | 256 s | 6/6, 1 not run | `stress/forward-30000000-rerun.log` |
| `forward` 1e8, rerun | 30 min | 0 | 1,750 s | 6/6, 1 not run; device 4.05 / 4.53 ms a step; field within 8.34e-7 | `stress/forward-100000000-rerun.log` |

(Device times are `all_to_all` / `ppermute`; walls are each record's
`wall_s`.)  "1 not run" is the exact count of the cells, a float32 sum of
ones, which is exact only under 2**24 cells.

**Why two steps timed out, and left nothing.**  The 4-minute boxes were
scaled from a CPU dry run and missed the host's setup at these sizes.  The
output went through a pipe, where Python buffers it, and the buffer died
with the process: the logs are empty.  The reruns ran under `python3 -u`.

**The wrapper column at these sizes is an artifact.**  The public
`update()` took 251 to 285 ms a step at 1e7 cells, 709 to 750 at 3e7 and
2.4 to 2.5 s at 1e8, against a compiled step of 3 to 5 ms.  The runner's
own test node built a new static array on every access, so the wrapper's
static cache, keyed on the arrays' identity, missed and re-partitioned on
every call (graph against eager, below, measures it).  The runner is fixed
in the same change as this record.  The same artifact is in section 2b's
wrapper column (8.7 to 25.9 ms a step at 1e5 to 1e6 cells, 127 ms on the
real mesh); no check reads that column.

### (b) The capacity ramp: one D3Q19 lattice larger than one card

| mesh | rung | cells | outcome | fill of 96 GB | peak bytes a cell | file |
|---|---|---|---|---|---|---|
| 2×2 | 1 | 176,463,360 | passed, max_rel 0 | 0.184 | 430.1 | `capacity-2x2/capacity_rung_01.json` |
| 2×2 | 2 | 471,905,280 | **passed**, max_rel 1.2e-6 | 0.486 | 424.7 | `capacity-2x2/capacity_rung_02.json` |
| 2×2 | 3 | 712,673,280 | out of memory | - | - | `capacity-2x2/capacity_rung_03.json` |
| 2×2 | soak | 350,784,000 | passed, 35 blocks | 0.362 | 425.7 | `capacity-2x2/capacity_soak.json` |
| 4×1 | 1 | 174,182,400 | passed, max_rel 1.2e-6 | 0.183 | 433.3 | `capacity-4x1/capacity_rung_01.json` |
| 4×1 | 2 | 802,414,080 | out of memory | - | - | `capacity-4x1/capacity_rung_02.json` |

* **Ceiling: 471,905,280 cells** on the 2×2 pencil (`capacity-2x2.txt`),
  about twice what one card holds.  Each rung took 21 to 44 s; the whole
  2×2 ramp with its soak 5 minutes, the 4×1 ramp 1 minute
  (`stress/progress.txt`).
* **The step peaks at about 4.3 times the state.**  Rung 2, per device:
  11.44 GB after the build, 49.55 GB peak after the first compiled step
  (`capacity-2x2/capacity_rung_02.json`, `memory.readings`, device 1); 425
  to 433 bytes a cell at peak against 97 of state.  The first rung's guess
  of 6 times the state gave a fill of 0.18 for the 0.25 asked.
* **Both out-of-memory rungs fell on one allocation below the predicted
  peak:** 37.46 GiB in `jit__local_update` on device 0 at rung 3, where 4.3
  times the 17.36 GB built per device is 75 GB, under the allocator's limit
  of 96.88 GB (`capacity-2x2/capacity_rung_03.stderr.log`); 42.23 GiB at
  the 4×1's rung 2.  Most likely fragmentation of the growing pool; not
  reproduced.
* **The soak:** device bytes in use grew by 0 after the first block, host
  RSS by 2.4 MiB (`capacity-2x2.txt`).

### (c) The default matmul precision, once

`stencil` at 1e5 cells with `JAX_DEFAULT_MATMUL_PRECISION` unset: **348/356**
(`stress/stencil-default-precision.log`, exit 1, which is a result here).
All eight failures are the D2Q9 lattice's gradients, on all four meshes:
d loss / d initial `f` 1.637e-3 (limit 1e-5) and d loss / d viscosity
4.74e-5 to 5.02e-5.  Every forward and halo check passed.  Measured at
`highest` the same gradients are 2.1e-6 and 1.6e-6 to 1.8e-6 (section 2b).

## The additions (gate nothing)

* **The LBM precision patch** (`precision_fix_results/lbm_matmul_precision.patch`:
  seven products in `lbm.py` and `lbm_pipe.py` at `Precision.HIGHEST`),
  applied to a copy of the checkout, imported from the copy
  (`scripts/extra_progress.txt`), `stencil` at 1e5 cells at the default
  precision: **356/356**.  The lattice's gradients 2.0e-6 to 2.1e-6 (initial
  `f`) and 8.8e-7 to 1.4e-6 (viscosity)
  (`precision_fix_results/stencil-default-precision-fixed.log`).  The record
  names commit `c0c6add8` but ran on a patched copy, so it decides nothing.
  The library change is a pull request of its own (#327).
* **Graph against eager at 1e7 cells** (`gve_results/gve.json`,
  `gve_results/gve.log`), the runner's node on four GPUs:

  | variant | ms a step | re-partitions |
  |---|---|---|
  | A: public `update()`, statics built on every access (the runner as it was) | 242.5 | 40 in 20 steps |
  | B: public `update()`, statics held | 4.45 | 0 |
  | C: `GraphManager.step()` | 7.23 | 2 at compile, then 0 |
  | C: `GraphManager.run_scan(20)` | 0.81 | 0 |

  A and B end on the same field bit for bit; the graph within 2.4e-7.  A
  compiled graph never pays the cache miss.  `gve_results/graph_vs_eager.py`
  is the script as run; against a tree with the runner fixed, A and B are
  the same node.
* **`profile_graph` at 1e7 cells, four cards sharded against one card
  unsharded** (`gve_results/profile/4_cards_sharded.txt`,
  `gve_results/profile/1_card_unsharded.txt`): median 3.82 against 1.25 ms
  a step; dispatch floor 2.41 against 0.90 ms; device busy 1.10 ms (29 %)
  against 0.28 ms (22 %); 43 kernels a step against 2, the largest NCCL's
  send/receive at 0.71 ms; compile 3.3 against 1.7 s.  For a node this
  cheap the sharded step is bound by dispatch and the halo exchange.

## Reproduce the verdict

```sh
python benchmarks/multigpu/run_pod.py --summarise \
    benchmarks/results/multigpu/2026-10-12_rtxpro6000/multigpu        # needs no JAX
```

It exits 0 and prints the six items CLOSED on `c0c6add8a355`, identical to
`multigpu/summary.txt`.  `run_capacity.py --summarise` on each
`capacity-*` directory reprints its `.txt`.
