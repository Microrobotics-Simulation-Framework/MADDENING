# Cloud Module Roadmap

Status of the cloud module (`maddening/cloud/`) and planned next steps.
Last updated: 2026-03-19.

---

## Completed

### Core Cloud Module (PR cc6fab3)

All code in `src/maddening/cloud/`, 88 tests passing.

- **StreamingSession ABC** + `StreamConfig`, `StreamInfo`, `QualityPreset`, `GPUFramebuffer`
- **MockStreamSession** — zero-dep mock for testing
- **HMAC-SHA256 auth** — `generate_session_token()` / `validate_session_token()`
- **SelkiesSession** — GStreamer/WebRTC implementation (requires PyGObject)
- **CloudSession** — server-side state machine with typed health probes, preemption detection
- **SelkiesRenderer(Renderer)** — wraps inner renderer + StreamingSession, GPU/CPU path detection
- **Multi-GPU Jacobi** — `create_device_mesh()`, `assign_nodes_to_devices()`, `build_sharded_jacobi_pass()`, `GraphManager.enable_multigpu()`
- **Cloud container** — `docker/Dockerfile.cloud`, `entrypoint.py`
- **Server endpoints** — `POST /cloud/launch`, `GET /cloud/status`, `POST /cloud/teardown`

### CloudLauncher (PR 8d86392 + 65f190b)

User-facing launch path, 46 tests passing.

- **CloudProvider ABC** + `RunPodProvider` + `LambdaLabsProvider` (stub)
- **CloudLauncher** — loads credentials from `~/.maddening/cloud_credentials.yaml`, calls `sky.*` directly
- **CloudJob** — handle with `phase`, `vm_ip`, `ports`, `from_cluster_name()` reconnect
- **JobConfig** + `CostPolicy` — YAML-based, reserved env var validation
- **Credential context manager** — write/delete with pre-existing file preservation
- **Cost guards** — hourly rate check (hard reject) + budget check (best-effort, lagged)
- **Examples** — `src/maddening/examples/cloud/01_validate.py`, `02_runpod_launch.py`

### Package Restructure + Import Guards (PR 0a3980b)

Consistent install experience, 146 cloud + guard tests passing.

- **Pyproject.toml restructured** — hardware extras (`cuda12`, `tpu`), per-provider cloud (`runpod`, `lambda`, `aws`, `gcp`), combo (`cloud`, `cloud-all`), task bundles (`server`, `client`)
- **Import guards on all optional deps** — every missing dep raises `ImportError` with exact `pip install maddening[extra]` command
- **`__getattr__` lazy imports improved** — `viz/`, `viz/backends/`, `surrogates/`, `cloud/` all catch ImportError with install hints
- **Cloud examples consolidated** — moved from `examples/cloud/` to `src/maddening/examples/cloud/`
- **User documentation** — `docs/user_guide/installation.md`, `docs/user_guide/quickstart.md`
- **Import guard tests** — `tests/test_import_guards.py` (subprocess-based) + TOML consistency checks
- **Removed obsolete extras** — `cloud-deploy` (conflated streaming + cloud)

### Launch Improvements + End-to-End Validation

- **Spot fallback** — `CostPolicy(spot_fallback=True)` auto-retries on-demand when spot is unavailable, subject to the same cost guards
- **`retry_until_up`** — handles transient SSH/provisioning failures automatically
- **Hourly cost tracking** — `CloudJob` stores resolved hourly cost from catalog; `status()` returns it correctly
- **Error truncation** — spot unavailability errors are truncated to a one-line summary with actionable advice; other errors preserved in full
- **Remaining pyvista/usd import guards** — all bare `import pyvista` calls now go through `_import_pyvista()` helper with `maddening[viz3d]` message
- **End-to-end validated** — A40 spot launch on RunPod: provision → status → teardown, all confirmed clean via both SkyPilot and RunPod API

---

## Next Steps Checklist

### Immediate: End-to-End Cloud Launch

- [x] Run real launch on RunPod (A40 on-demand, A40 spot) — VM provisioned, status checked, teardown confirmed
- [x] Verify `CloudJob.status()` returns correct `cluster_status: UP`, `vm_ip`, `hourly_cost`
- [x] Verify spot fallback: RTXA4000 spot sold out → auto-retry on-demand works
- [x] Verify truncated error messages for spot unavailability
- [x] Verify `retry_until_up` handles transient SSH failures
- [x] Verify credential cleanup: `~/.runpod/config.toml` deleted after teardown, RunPod API confirms zero pods
- [x] Test `CloudJob.from_cluster_name()` reconnect — status() and teardown() work on reconnected handle
- [x] Fix: `retry_until_up` now only set for on-demand (spot + retry = infinite loop on no capacity)
- **Note:** SkyPilot setup overhead is ~12 min on RunPod (Ray install, SSH config). The pod starts in seconds but SkyPilot's runtime setup is slow. This is a known SkyPilot issue, not a MADDENING bug.

### Short-term: Cloud Simulation Server (Option A — setup script, no Docker)

Using SkyPilot `setup:` to `pip install` on a base CUDA image. This lets us
discover system deps, JAX CUDA quirks, port requirements, and entrypoint bugs
before baking them into a Docker image.

**Issues discovered so far (2026-03-19):**

1. **Python version mismatch on `runpod/base`**: `python3` is 3.10 but `pip` targets 3.12.
   Must use `python3.12` explicitly or create a venv.
2. **JAX CUDA plugin conflict**: Base image has `jax_cuda12_plugin 0.9.2` pre-installed,
   incompatible with `jax 0.5.3`. Need `pip install "jax[cuda12]==0.5.3" "jaxlib==0.5.3"`
   with explicit version pins, or uninstall pre-installed plugin first.
3. **SkyPilot Ray workers can't see GPU**: `CUDA_ERROR_NO_DEVICE` in Ray worker processes.
   GPU visible from SSH but not from Ray's sandboxed environment.
4. **`sky.get()` raises spurious `AssertionError`**: Workaround via `stream_and_get()` +
   fallback status polling implemented in launcher.py.
5. **SkyPilot overhead**: ~2 min US (cached image), ~12+ min EU (cold cache), ~35+ min
   CZ (very slow Docker pull). Prefer US region.
6. **RunPod port mapping**: Ports use NAT. Must query RunPod API for `port2endpoint`
   mapping (private_port -> public ip:port). Cannot use `vm_ip:8000` directly.
7. **`.skyignore` required**: Without it, SkyPilot rsyncs `.venv/` (1.4 GB). With it, 2.5 MB.

**Recommended approach for next attempt:**
Instead of running through SkyPilot's Ray job scheduler, run the server directly
via SSH after `sky.launch()` brings the VM UP. This bypasses Ray's GPU isolation.
Flow: `sky.launch()` (no setup/run) → SSH in → `pip install` in system python3.12
→ fix JAX versions → start server directly.

- [x] Added `.skyignore` to exclude `.venv/`, `.git/`, `__pycache__/`, etc.
- [x] Added `setup` and `run` fields to `JobConfig` + `workdir` support
- [x] Fixed `sky.get()` AssertionError via `stream_and_get` + fallback polling
- [x] Discovered RunPod port mapping requirement (NAT, port2endpoint)
- [x] Discovered Python 3.10/3.12 mismatch on runpod/base image
- [x] Discovered JAX CUDA plugin version conflict
- [x] Discovered Ray worker GPU isolation issue
- [x] Fix Python version: use system pip (targets python3.12) via SSH directly
- [x] Fix JAX: system pip installs compatible jax[cuda12] — CudaDevice(id=0) confirmed
- [x] Bypass Ray: SSH-based approach works — `CloudJob.ssh_run()` method added
- [x] Verify FastAPI server accessible via RunPod port mapping — all endpoints work
- [x] Verify REST API: `/graph`, `/graph/state`, `/sim/step`, `/sim/run` all tested end-to-end
- [x] Verify WebSocket state streaming — JSON (`/ws/state`) and binary (`/ws/state/binary`) both work over network from cloud GPU

### Short-term: SelkiesSession Integration Testing

- [x] Test `SelkiesSession` with real GStreamer on cloud GPU — all 5 tests pass
  - GStreamer 1.20.3 imports, pipeline builds with x264enc (needs `gstreamer1.0-plugins-ugly`)
  - Session start/stop lifecycle works, signaling URL generated
  - CPU framebuffer push works
  - SelkiesRenderer wraps inner renderer + pushes frames correctly
  - **Must use python3.10** (system python) for `gi` bindings — python3.12 can't load system `_gi.so`
  - System packages needed: `gstreamer1.0-plugins-{base,good,bad,ugly}`, `gstreamer1.0-nice`,
    `gir1.2-{gst-plugins-bad-1.0,gstreamer-1.0}`, `python3-gi`, `python3-gi-cairo`
- [ ] Verify WebRTC streaming from cloud GPU to local browser (requires browser-side WebRTC client)
- [x] Test `SelkiesRenderer` wrapping a real renderer — works with DummyRenderer on cloud GPU

### Medium-term: Multi-GPU

- [ ] Wire `enable_multigpu()` into `_build_step_fn()` so sharded pass replaces `one_pass_jacobi`
- [ ] Replace sequential per-device loop with actual `jax.experimental.shard_map`
- [ ] Benchmark on real multi-GPU hardware (2+ GPUs)
- [ ] Run multi-GPU tests on RunPod with multi-GPU instance (e.g. `A100-80GB:2`)

### Medium-term: Multi-Job Architecture

- [ ] Implement ZMQ coordinator process (ROUTER socket, registration, topology broadcast)
- [ ] Implement `CloudGroup` with `provision_all()` / `start()` / `teardown_all()`
- [ ] Implement rendezvous barrier (block until all N workers registered)
- [ ] Implement heartbeat monitoring + `TEARDOWN_ALL` failure mode
- [ ] Implement `ISOLATE` failure mode (`PEER_DEAD` notification)
- [ ] Test with 2-VM setup on RunPod (two `RTXA4000` instances)

### After Multi-Job: Finalize Docker Image

Build `docker/Dockerfile.cloud` only after multi-job is working via Option A.
By then we'll have discovered all system deps, CUDA requirements, port
mappings, and coordinator process needs. One image serves both single-job
and multi-job (coordinator is a lightweight Python process, not a separate
container).

- [ ] Bake all discovered system deps into Dockerfile
- [ ] Build and push to Docker Hub / GHCR
- [ ] Verify launch with `image_id: docker:...` matches Option A behavior
- [ ] Remove setup script workarounds
- [ ] Test `CloudSweep` as degenerate case (N independent jobs, no ZMQ)

### Medium-term: Lambda Labs Validation

- [ ] Test `LambdaLabsProvider` end-to-end (currently stub only)
- [ ] Validate credential write/delete lifecycle with real Lambda API key
- [ ] Test `sky check lambda` after `CloudLauncher` writes credentials
- [ ] Run a real job on Lambda Labs

### Future: Additional Providers

- [ ] AWS provider (credential handling for `~/.aws/credentials`)
- [ ] GCP provider (credential handling for service account JSON)
- [ ] Test cross-provider multi-job (e.g. rank-0 on RunPod, workers on Lambda)

---

## Architecture: Two Paths to Cloud

```
CloudLauncher  — User-facing, script/CLI path. Loads credentials from
                 ~/.maddening/cloud_credentials.yaml. Calls sky.* directly.
                 Future basis for CloudSweep and CloudGroup.

CloudSession   — Server-side orchestration path. Credentials assumed
                 pre-configured on the machine. Uses _skypilot.py wrapper.
                 Future basis for cloud API endpoints in MICROBOTICA.
```

Both use `providers.py` for credential file management.

---

## Future: Multi-Job Architecture (Design Only)

### Coordinator on Rank-0

For distributed MADDENING graphs (each subgraph on a separate cloud VM), we use a coordinator process on the rank-0 VM to solve the rendezvous problem.

**Why rank-0, not a separate VM:**
- Extra VM adds cost, provisioning complexity, another failure point
- The coordinator is a lightweight Python process, not a simulation workload
- Process-level isolation on rank-0: coordinator survives if the simulation process crashes
- If rank-0's VM dies, workers detect via heartbeat timeout and self-terminate

**Sequencing:**
1. Provision rank-0 VM
2. Rank-0's SkyPilot `setup` script starts coordinator process (`nohup ... &`), which binds a ZMQ ROUTER socket on a known port
3. Extract rank-0's IP from SkyPilot
4. Provision all other VMs in parallel, injecting `COORDINATOR_ADDR=<rank0_ip>:<port>` via `task.envs`
5. Each worker's `setup` sends a registration message to the coordinator
6. Coordinator blocks until all N workers have registered (or `rendezvous_timeout` expires)
7. Coordinator broadcasts topology to all workers
8. All workers (including rank-0) start their MADDENING subgraphs

**Verified:** SkyPilot background processes started in `setup` survive into `run` phase. The setup script runs via SSH as a blocking bash command; processes backgrounded with `nohup &` persist.

### CloudGroup Interface

```python
class GroupFailureMode(Enum):
    TEARDOWN_ALL = "teardown_all"   # any failure tears down everything
    ISOLATE = "isolate"             # mark failed job dead, others continue

@dataclass(frozen=True)
class GroupConfig:
    failure_mode: GroupFailureMode = GroupFailureMode.TEARDOWN_ALL
    rendezvous_timeout: float = 300.0
    heartbeat_interval: float = 10.0
    heartbeat_timeout: float = 30.0
    coordinator_port: int = 5580

@dataclass(frozen=True)
class SubgraphSpec:
    subgraph_id: str
    job_config: JobConfig
    zmq_ports: dict[str, int]       # {service_name: port}

class CloudGroup:
    def __init__(self, specs: list[SubgraphSpec], group_config: GroupConfig, credentials_path=None): ...

    @classmethod
    def from_cluster_names(cls, names: dict[str, str], group_config=None) -> "CloudGroup": ...

    def provision_all(self) -> dict[str, CloudJob]:
        """Provision all VMs. Rank-0 first, then others in parallel."""
        ...

    def start(self) -> None:
        """Trigger topology broadcast. Workers unblock and start graphs."""
        ...

    def status(self) -> dict[str, dict]: ...
    def cost_so_far(self) -> float: ...
    def teardown_all(self) -> None: ...
    def teardown_one(self, subgraph_id: str) -> None: ...  # ISOLATE mode only
```

**CloudSweep is a degenerate case:** a CloudGroup where `zmq_ports` is empty for all specs. No inter-job communication, no topology, trivial rendezvous.

### ZeroMQ Topology Descriptor

Broadcast from coordinator to each worker after rendezvous:

```json
{
  "subgraph_id": "bem_nearfield",
  "peers": [
    {
      "peer_id": "lbm_farfield",
      "address": "tcp://10.0.0.43:5555",
      "role": "connect",
      "socket_type": "SUB",
      "edge_name": "pressure_bc"
    },
    {
      "peer_id": "lbm_farfield",
      "address": "tcp://10.0.0.42:5556",
      "role": "bind",
      "socket_type": "PUB",
      "edge_name": "velocity_bc"
    }
  ]
}
```

**Bind/connect rule:** upstream node (data producer) binds PUB, downstream node connects SUB. Bidirectional edges get two entries.

### Failure Handling

**TEARDOWN_ALL mode:**
- During rendezvous: worker fails to register within timeout → coordinator sends `SHUTDOWN` to all registered workers → `CloudGroup.provision_all()` raises `LaunchError`
- During execution: missed heartbeats → coordinator sends `SHUTDOWN` to all → user's `CloudGroup.status()` shows failure → `teardown_all()` cleans up

**ISOLATE mode:**
- During rendezvous: same as TEARDOWN_ALL (can't run partial graph without topology)
- During execution: coordinator sends `PEER_DEAD {peer_id}` to survivors → surviving workers mark ZMQ sockets dead, continue with stale/zero boundary data
- **Not correctness-preserving** — only for auxiliary nodes (monitoring, visualization)

### Reserved Environment Variables

| Env var | Set by | Injected when |
|---------|--------|---------------|
| `MADDENING_CLOUD_CONFIG` | CloudLauncher | At provision (task.envs) |
| `COORDINATOR_ADDR` | CloudGroup | At provision (task.envs) |
| `SUBGRAPH_ID` | CloudGroup | At provision (task.envs) |
| `MADDENING_TOPOLOGY` | Coordinator process | At runtime (ZMQ message, not env var) |

### CloudJob Fields for Multi-Job

These exist now on CloudJob but are unused in single-job mode:
- `phase` — `PROVISIONING | WAITING | EXECUTING | DONE | FAILED` (WAITING = at rendezvous barrier)
- `vm_ip` — needed by CloudGroup to build topology
- `ports` — `dict[str, int]`, populated from SubgraphSpec.zmq_ports

---

## Future: Multi-GPU Enhancements

### Current state
- `build_sharded_jacobi_pass()` partitions nodes by device and runs sequentially per device
- Does NOT yet use `jax.experimental.shard_map` — logical partitioning only
- `enable_multigpu()` added to GraphManager but not wired into `_build_step_fn()`

### TODO
- Wire `enable_multigpu()` into `_build_step_fn()` so sharded pass replaces `one_pass_jacobi` when enabled
- Replace sequential per-device loop with actual `shard_map` for true cross-device parallelism
- Benchmark on real multi-GPU hardware (2+ GPUs)
- Test with `XLA_FLAGS=--xla_force_host_platform_device_count=N` (currently tests skip on CI with 1 device)

---

## Verified SkyPilot Internals

These findings constrain the design. Recorded here to avoid re-verification.

### Credential lifetime
- Credentials are read DURING provisioning (inside `sky.get()`), not at request dispatch time
- `sky/server/requests/executor.py:500-511`: `func()` runs inside `override_request_env_and_config()` context
- `sky/clouds/runpod.py:362`: filesystem check `os.path.exists(credential_file)` runs in worker process
- **Consequence:** credential context manager must wrap `sky.launch()` + `sky.get()`

### SkyPilot setup vs run process lifetime
- Background processes started in `setup` survive into `run` phase
- `cloud_vm_ray_backend.py:3708-3720`: setup runs via `runner.run(setup_cmd)` (blocking SSH)
- Standard Unix behavior: `nohup ... &` processes persist after script exits
- **Consequence:** coordinator can be started in `setup`, worker in `run`

### RunPod credential paths
- SkyPilot hardcodes `~/.runpod/config.toml` (no env var override for file check)
- `sky/adaptors/runpod.py:26` falls back to `RUNPOD_API_KEY` env var for API calls
- Both file AND env var must be set

### Lambda Labs credential paths
- Hardcoded `~/.lambda_cloud/lambda_keys`
- Format: `api_key = YOUR_KEY` (plain text key=value)

### SkyPilot async SDK (0.11+)
- `sky.launch()` returns `RequestId`, use `sky.get(request_id)` to retrieve result
- `sky.status()` same pattern
- `sky.stream_and_get(request_id)` streams logs + returns result

### Cost reporting
- `sky.cost_report()` is NOT real-time billing — estimated from local cache
- `hourly_rate * uptime` from locally tracked usage intervals
- Does not reflect spend from other machines or clusters managed outside SkyPilot

### RunPod GPU names
- RunPod uses `RTXA4000`, `A40`, `RTX4090`, `L4`, `A100-80GB` etc.
- NOT `A4000` — SkyPilot catalog names don't always match marketing names
- `sky show-gpus --cloud runpod --all` to list
- RunPod regions use country codes: `NL`, `SE`, `US`, `CZ`, `NO`, `IS`, `CA`, `RO` (not `EU`)
