# Cloud Module Roadmap

Status of the cloud module (`maddening/cloud/`) and planned next steps.

---

## Completed

### Phase 1-7: Core Cloud Module (PR cc6fab3)

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
- **Package extras** — `streaming`, `cloud`, `cloud-deploy` (being restructured, see below)

### CloudLauncher (PR 8d86392 + 65f190b)

User-facing launch path, 46 tests passing.

- **CloudProvider ABC** + `RunPodProvider` + `LambdaLabsProvider` (stub)
- **CloudLauncher** — loads credentials from `~/.maddening/cloud_credentials.yaml`, calls `sky.*` directly
- **CloudJob** — handle with `phase`, `vm_ip`, `ports`, `from_cluster_name()` reconnect
- **JobConfig** + `CostPolicy` — YAML-based, reserved env var validation
- **Credential context manager** — write/delete with pre-existing file preservation
- **Cost guards** — hourly rate check (hard reject) + budget check (best-effort, lagged)
- **Examples** — `01_validate.py`, `02_runpod_launch.py`

---

## In Progress

### Pyproject.toml Restructure + Import Guards

Replace the current extras with task-oriented bundles and add consistent dependency error messages across the codebase. See the plan below.

**New extras:**
- `cuda12`, `tpu` — hardware acceleration
- `runpod`, `lambda`, `aws`, `gcp` — per-provider cloud (includes SkyPilot + provider SDK)
- `cloud` — all MADDENING-supported providers (RunPod + Lambda)
- `cloud-all` — every SkyPilot provider
- `server` — FastAPI + ZMQ + rich + matplotlib (Docker image target)
- `client` — ZMQ + rich (thin viewer)
- Keep fine-grained extras (`viz`, `terminal`, `network`, `api`, `surrogates`, `viz3d`, `gpu-viz`, `usd`, `streaming`, `sbom`)

**Import guard work:**
- Every optional dep import boundary gets a try/except with `pip install maddening[extra]` message
- `__getattr__` lazy imports catch ImportError and re-raise with install hint
- New `tests/test_import_guards.py` (subprocess-based) + `tests/test_packaging.py` (TOML validation)

**Extras being removed:** `cloud` (bare SkyPilot, useless), `cloud-deploy` (conflated streaming + cloud)

---

## Next: End-to-End Cloud Testing

### Prerequisites
- User has `~/.maddening/cloud_credentials.yaml` with RunPod API key
- `pip install maddening[runpod]`
- `sky check runpod` shows enabled

### Test sequence
1. `python src/maddening/examples/cloud/01_validate.py` — parse config, resolve GPU, check cost ✅ (done)
2. `python src/maddening/examples/cloud/02_runpod_launch.py` — real launch on RunPod, stream logs, status, teardown
3. Launch with `maddening[server,cuda12]` Docker image on RunPod
4. Connect to running simulation via WebSocket from local machine

---

## Future: Multi-Job Architecture

### Architecture Decision: Coordinator on Rank-0

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

### CloudJob Fields Added for Multi-Job

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
