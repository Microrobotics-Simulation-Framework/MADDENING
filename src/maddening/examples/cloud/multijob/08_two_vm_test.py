#!/usr/bin/env python3
"""Test multi-job architecture with 2 VMs on RunPod.

Provisions 2 VMs:
  - VM 0 (rank-0): runs coordinator + "flow" subgraph
  - VM 1 (worker): runs "structure" subgraph

Tests the rendezvous flow:
  1. Launch rank-0, start coordinator via SSH
  2. Launch worker, register with coordinator via SSH
  3. Verify both workers registered and each received its topology
  4. Tear down both

It checks registration and topology only; it does not exchange data
over the PUB/SUB sockets the topology describes.  Exits non-zero if a
check fails.

``--local`` runs the same rendezvous on this machine instead: the
coordinator and both workers in one process, on loopback, with no cloud
account.  It never imports the cloud launcher.

Usage:
    # No cloud account needed:
    python -m maddening.examples.cloud.multijob.08_two_vm_test --local

    # Provisions two billable RunPod VMs (needs ~/.maddening/cloud_credentials.yaml):
    python -m maddening.examples.cloud.multijob.08_two_vm_test
    python -m maddening.examples.cloud.multijob.08_two_vm_test --gpu RTX4090
    python -m maddening.examples.cloud.multijob.08_two_vm_test --keep
"""

import argparse
import json
import os
import secrets
import shlex
import socket
import sys
import threading
import time


# Install script — same deps for both VMs
INSTALL_CMD = (
    "python3 -m pip install -q --root-user-action=ignore"
    ' "pyzmq>=25.0" "pyyaml>=6.0" "numpy>=1.24"'
    " && [ -d ~/sky_workdir/src ]"
    " && python3 -m pip install -q --root-user-action=ignore -e ~/sky_workdir"
    " ; echo INSTALL_DONE"
)

# Coordinator script (runs on rank-0)
COORDINATOR_SCRIPT_TEMPLATE = """
import json, os, sys, time, traceback
sys.path.insert(0, os.path.expanduser("~/sky_workdir/src"))
print(f"Python: {{sys.executable}}", flush=True)
print(f"sys.path: {{sys.path[:3]}}", flush=True)

try:
    from maddening.cloud.multigpu.coordinator import Coordinator
    print("Coordinator imported OK", flush=True)
except Exception as e:
    print(f"IMPORT FAILED: {{e}}", flush=True)
    traceback.print_exc()
    sys.exit(1)

expected = {expected_json}
edges = {edges_json}
port = {port}

print(f"Starting coordinator on :{{port}}, expecting {{expected}}", flush=True)
try:
    import zmq
    print(f"ZMQ version: {{zmq.zmq_version()}}", flush=True)
except Exception as e:
    print(f"ZMQ IMPORT FAILED: {{e}}", flush=True)
    sys.exit(1)

# bind_host="0.0.0.0" is required here: the structure worker reaches this
# coordinator across the internet. That turns on ZMQ CURVE, which needs
# MADDENING_API_TOKEN set to the same value in both jobs' envs -- the
# coordinator refuses to start without it rather than listening in the
# clear. See maddening.transport_auth.
coord = Coordinator(
    expected_workers=expected,
    edges=edges,
    port=port,
    heartbeat_timeout=60.0,
    bind_host="0.0.0.0",
)
coord.start()
print("Coordinator thread started, waiting for workers...", flush=True)

if coord.wait_for_all(timeout=300):
    print("All workers registered!", flush=True)
    topo = coord.build_topology()
    for wid, t in topo.items():
        print(f"  {{wid}}: {{len(t.peers)}} peers", flush=True)
    with open("/tmp/coordinator_ready.json", "w") as f:
        json.dump({{"status": "ready", "workers": list(coord.registered_workers.keys())}}, f)
    print("Coordinator ready. Sleeping...", flush=True)
    while True:
        time.sleep(1)
else:
    print("TIMEOUT: not all workers registered", flush=True)
    with open("/tmp/coordinator_ready.json", "w") as f:
        json.dump({{"status": "timeout"}}, f)
    sys.exit(1)
"""

WORKER_SCRIPT_TEMPLATE = """
import json, os, sys, time, socket, traceback
sys.path.insert(0, os.path.expanduser("~/sky_workdir/src"))
print(f"Python: {{sys.executable}}", flush=True)

try:
    from maddening.cloud.multigpu.worker_client import WorkerClient
    print("WorkerClient imported OK", flush=True)
except Exception as e:
    print(f"IMPORT FAILED: {{e}}", flush=True)
    traceback.print_exc()
    sys.exit(1)

coordinator_addr = "{coordinator_addr}"
subgraph_id = "{subgraph_id}"
my_ip = socket.gethostbyname(socket.gethostname())

print(f"Worker {{subgraph_id}} connecting to coordinator at {{coordinator_addr}}", flush=True)
print(f"My IP: {{my_ip}}", flush=True)

# secure=True is explicit because rank 0's own worker reaches the
# coordinator over 127.0.0.1, and a loopback address would otherwise turn
# CURVE off while the coordinator (bound 0.0.0.0) has it on. The
# coordinator's posture is set by its bind address, which this side
# cannot see.
client = WorkerClient(
    coordinator_addr=coordinator_addr,
    subgraph_id=subgraph_id,
    address=f"{{my_ip}}:5555",
    zmq_ports={{"state": 5555}},
    secure=True,
)

try:
    topology = client.register_and_wait(timeout=120)
    print(f"Topology received: {{len(topology)}} peers", flush=True)
    for peer in topology:
        print(f"  {{peer.peer_id}}: {{peer.role}} {{peer.socket_type}} @ {{peer.address}}", flush=True)

    with open("/tmp/worker_ready.json", "w") as f:
        json.dump({{
            "status": "ready",
            "subgraph_id": subgraph_id,
            "peers": len(topology),
        }}, f)

    client.start_heartbeat(interval=5.0)
    print("Worker ready. Heartbeating...", flush=True)
    while True:
        time.sleep(1)

except Exception as e:
    print(f"Worker {{subgraph_id}} FAILED: {{e}}", flush=True)
    traceback.print_exc()
    with open("/tmp/worker_ready.json", "w") as f:
        json.dump({{"status": "error", "error": str(e)}}, f)
    sys.exit(1)
"""


# The inter-job edges, shared by the cloud and the --local runs.
EDGES = [{
    "source": "flow", "target": "structure",
    "source_field": "pressure", "target_field": "load",
}, {
    "source": "structure", "target": "flow",
    "source_field": "displacement", "target_field": "wall_bc",
}]
WORKERS = ["flow", "structure"]


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_local(timeout: float = 60.0) -> int:
    """The rendezvous on loopback: coordinator and two workers, one process.

    The same Coordinator and WorkerClient the VMs run, bound to
    127.0.0.1, where neither needs a token (CURVE turns on only for a
    non-loopback address).
    """
    from maddening.cloud.multigpu.coordinator import Coordinator
    from maddening.cloud.multigpu.worker_client import WorkerClient

    port = _free_loopback_port()
    print(f"Local run: coordinator on tcp://127.0.0.1:{port}, "
          f"workers {WORKERS}")
    coord = Coordinator(
        expected_workers=WORKERS, edges=EDGES, port=port,
        heartbeat_timeout=60.0, bind_host="127.0.0.1",
    )
    coord.start()

    clients: dict[str, WorkerClient] = {}
    topologies: dict[str, list] = {}
    errors: dict[str, str] = {}

    def _worker(subgraph_id: str) -> None:
        data_port = _free_loopback_port()
        client = WorkerClient(
            coordinator_addr=f"127.0.0.1:{port}",
            subgraph_id=subgraph_id,
            address=f"127.0.0.1:{data_port}",
            zmq_ports={"state": data_port},
        )
        clients[subgraph_id] = client
        try:
            topologies[subgraph_id] = client.register_and_wait(timeout=timeout)
        except Exception as exc:  # reported below, not swallowed
            errors[subgraph_id] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=_worker, args=(w,), daemon=True)
               for w in WORKERS]
    all_ok = True
    try:
        for t in threads:
            t.start()
        registered = coord.wait_for_all(timeout=timeout)
        for t in threads:
            t.join(timeout=timeout)

        print()
        print("Verification")
        print("-" * 40)
        workers = set(coord.registered_workers)
        if registered and workers == set(WORKERS):
            print(f"  PASS: both workers registered with the coordinator")
        else:
            print(f"  FAIL: expected {set(WORKERS)}, registered {workers}")
            all_ok = False
        for w in WORKERS:
            if w in topologies:
                peers = topologies[w]
                print(f"  PASS: worker {w} received topology ({len(peers)} peers)")
                for peer in peers:
                    print(f"        {peer.peer_id}: {peer.role} "
                          f"{peer.socket_type} @ {peer.address}")
                # One edge out and one edge in per worker.
                if len(peers) != 2:
                    print(f"  FAIL: worker {w} expected 2 peers")
                    all_ok = False
            else:
                print(f"  FAIL: worker {w}: {errors.get(w, 'no topology')}")
                all_ok = False
    finally:
        for client in clients.values():
            client.stop()
        coord.shutdown()

    print()
    print("  ALL MULTI-JOB TESTS PASSED!" if all_ok
          else "  SOME TESTS FAILED -- see above")
    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser(description="2-VM multi-job test")
    parser.add_argument("--gpu", default="RTX4090", help="GPU type")
    parser.add_argument("--keep", action="store_true", help="Don't teardown")
    parser.add_argument(
        "--local", action="store_true",
        help="Run the rendezvous on this machine (loopback, no cloud)",
    )
    args = parser.parse_args()

    if args.local:
        sys.exit(run_local())

    # Imported here so that --local never loads the cloud launcher.
    from maddening.cloud.launcher import (
        CloudJob,
        CloudLauncher,
        CostPolicy,
        JobConfig,
        LaunchError,
    )

    # One shared secret for both VMs. The coordinator's ROUTER and both
    # workers derive their ZMQ CURVE keypairs from it, so there are no key
    # files to ship; it is the same variable the HTTP API uses. Generated
    # per run here -- in production, set it yourself and keep it.
    api_token = os.environ.get("MADDENING_API_TOKEN") or secrets.token_urlsafe(32)

    project_root = os.path.dirname(os.path.abspath(__file__))
    while project_root != "/" and not os.path.exists(
        os.path.join(project_root, "pyproject.toml")
    ):
        project_root = os.path.dirname(project_root)

    base_config = dict(
        provider="runpod",
        gpu_type=args.gpu,
        use_spot=False,
        region="US",
        cost=CostPolicy(
            max_cost_per_hour=2.0,
            max_total_budget=10.0,
            autostop_minutes=10,
            auto_teardown=False,
            spot_fallback=True,
        ),
        run="echo 'VM ready'; sleep 7200",
        workdir=project_root,
        ports=[8000, 5580, 5555, 5556],  # API, coordinator, ZMQ data
        envs={"MADDENING_API_TOKEN": api_token},
    )

    launcher = CloudLauncher()
    jobs: dict[str, CloudJob] = {}
    all_ok = False

    try:
        # --- Phase 1: Launch rank-0 ---
        print("=" * 60)
        print("Phase 1: Launching rank-0 (coordinator + flow)")
        print("=" * 60)
        config0 = JobConfig(**base_config)
        jobs["flow"] = launcher.launch(config0)
        rank0_ip = jobs["flow"].vm_ip
        rank0_ssh = jobs["flow"].ssh_port
        print(f"  Rank-0: {rank0_ip}:{rank0_ssh}")

        # --- Phase 2: Launch worker ---
        print()
        print("=" * 60)
        print("Phase 2: Launching worker (structure)")
        print("=" * 60)
        config1 = JobConfig(**base_config)
        jobs["structure"] = launcher.launch(config1)
        worker_ip = jobs["structure"].vm_ip
        worker_ssh = jobs["structure"].ssh_port
        print(f"  Worker: {worker_ip}:{worker_ssh}")

        # --- Phase 3: Install deps on both ---
        print()
        print("=" * 60)
        print("Phase 3: Installing deps on both VMs")
        print("=" * 60)
        for name, job in jobs.items():
            result = job.ssh_run(INSTALL_CMD, timeout=300, capture=True)
            last_line = (result.stdout or "").strip().split("\n")[-1]
            print(f"  {name}: {last_line}")

        # --- Phase 4: Start coordinator on rank-0 ---
        print()
        print("=" * 60)
        print("Phase 4: Starting coordinator on rank-0")
        print("=" * 60)

        # Generate coordinator script with embedded values (no env vars needed)
        coord_script = COORDINATOR_SCRIPT_TEMPLATE.format(
            expected_json=json.dumps(WORKERS),
            edges_json=json.dumps(EDGES),
            port=5580,
        )
        jobs["flow"].ssh_run(
            f"cat > /tmp/coordinator.py << 'PYEOF'\n{coord_script}\nPYEOF",
            check=True,
        )

        # Start coordinator with PID health check
        jobs["flow"].ssh_run(
            "python3 /tmp/coordinator.py > /tmp/coordinator.log 2>&1 &"
            " COORD_PID=$!; sleep 2;"
            " if ! kill -0 $COORD_PID 2>/dev/null; then"
            "   echo 'COORDINATOR CRASHED:'; cat /tmp/coordinator.log; exit 1;"
            " fi;"
            " echo COORD_PID=$COORD_PID",
            check=False,
        )
        # Verify it's alive
        time.sleep(3)
        result = jobs["flow"].ssh_run(
            "cat /tmp/coordinator.log 2>/dev/null | head -10",
            capture=True, check=False,
        )
        print(f"  Coordinator log:\n    " +
              "\n    ".join((result.stdout or "").strip().split("\n")[:5]))

        # --- Phase 5: Start workers ---
        print()
        print("=" * 60)
        print("Phase 5: Registering workers with coordinator")
        print("=" * 60)

        # Discover the RunPod-mapped public port for 5580
        coord_endpoint = jobs["flow"].get_runpod_endpoint(5580)
        if coord_endpoint:
            # Extract host:port from http://host:port
            coordinator_addr = coord_endpoint.replace("http://", "")
        else:
            coordinator_addr = f"{rank0_ip}:5580"
            print("  WARNING: Could not find RunPod port mapping for 5580")
        print(f"  Coordinator address: {coordinator_addr}")

        # Generate and start flow worker on rank-0 (localhost:5580)
        flow_script = WORKER_SCRIPT_TEMPLATE.format(
            coordinator_addr="127.0.0.1:5580",
            subgraph_id="flow",
        )
        jobs["flow"].ssh_run(
            f"cat > /tmp/worker.py << 'PYEOF'\n{flow_script}\nPYEOF",
            check=True,
        )
        jobs["flow"].ssh_run(
            "python3 /tmp/worker.py > /tmp/worker.log 2>&1 &"
            " W_PID=$!; sleep 2;"
            " if ! kill -0 $W_PID 2>/dev/null; then"
            "   echo 'FLOW WORKER CRASHED:'; cat /tmp/worker.log;"
            " fi;"
            " echo WORKER_PID=$W_PID",
            check=False,
        )
        print("  flow worker started on rank-0 (localhost:5580)")

        # Generate and start structure worker on VM 1 (public address)
        struct_script = WORKER_SCRIPT_TEMPLATE.format(
            coordinator_addr=coordinator_addr,
            subgraph_id="structure",
        )
        jobs["structure"].ssh_run(
            f"cat > /tmp/worker.py << 'PYEOF'\n{struct_script}\nPYEOF",
            check=True,
        )
        jobs["structure"].ssh_run(
            "python3 /tmp/worker.py > /tmp/worker.log 2>&1 &"
            " W_PID=$!; sleep 2;"
            " if ! kill -0 $W_PID 2>/dev/null; then"
            "   echo 'STRUCTURE WORKER CRASHED:'; cat /tmp/worker.log;"
            " fi;"
            " echo WORKER_PID=$W_PID",
            check=False,
        )
        print(f"  structure worker started on VM 1 ({coordinator_addr})")

        # --- Phase 6: Wait for rendezvous ---
        print()
        print("=" * 60)
        print("Phase 6: Waiting for rendezvous")
        print("=" * 60)

        # Poll coordinator status
        for attempt in range(30):
            time.sleep(3)
            try:
                result = jobs["flow"].ssh_run(
                    "cat /tmp/coordinator_ready.json 2>/dev/null",
                    capture=True, check=False,
                )
                if result.stdout and '"ready"' in result.stdout:
                    coord_status = json.loads(result.stdout.strip())
                    print(f"  Coordinator: {coord_status}")
                    break
            except Exception:
                pass
            if attempt % 5 == 0:
                print(f"  Waiting... (attempt {attempt})")
        else:
            print("  TIMEOUT waiting for coordinator")
            # Check coordinator logs
            result = jobs["flow"].ssh_run(
                "cat /tmp/coordinator.log 2>/dev/null | tail -20",
                capture=True, check=False,
            )
            print(f"  Coordinator log:\n{result.stdout}")
            # Check if coordinator process is alive
            result = jobs["flow"].ssh_run(
                "ps aux | grep coordinator | grep -v grep",
                capture=True, check=False,
            )
            print(f"  Coordinator processes: {result.stdout.strip() or 'NONE'}")
            # Check worker logs
            for name, job in jobs.items():
                result = job.ssh_run(
                    "cat /tmp/worker.log 2>/dev/null | tail -10",
                    capture=True, check=False,
                )
                print(f"  Worker {name} log:\n{result.stdout}")

        # Check worker status
        for name, job in jobs.items():
            try:
                result = job.ssh_run(
                    "cat /tmp/worker_ready.json 2>/dev/null",
                    capture=True, check=False,
                )
                if result.stdout:
                    worker_status = json.loads(result.stdout.strip())
                    print(f"  Worker {name}: {worker_status}")
            except Exception:
                print(f"  Worker {name}: status unknown")

        # --- Phase 7: Verify ---
        print()
        print("=" * 60)
        print("Phase 7: Verification")
        print("=" * 60)

        all_ok = True

        # Check coordinator registered both workers
        result = jobs["flow"].ssh_run(
            "cat /tmp/coordinator_ready.json 2>/dev/null",
            capture=True, check=False,
        )
        if result.stdout and '"ready"' in result.stdout:
            data = json.loads(result.stdout.strip())
            workers = data.get("workers", [])
            if set(workers) == {"flow", "structure"}:
                print("  PASS: Both workers registered with coordinator")
            else:
                print(f"  FAIL: Expected {{flow, structure}}, got {workers}")
                all_ok = False
        else:
            print("  FAIL: Coordinator not ready")
            all_ok = False

        # Check each worker got topology
        for name, job in jobs.items():
            result = job.ssh_run(
                "cat /tmp/worker_ready.json 2>/dev/null",
                capture=True, check=False,
            )
            if result.stdout and '"ready"' in result.stdout:
                data = json.loads(result.stdout.strip())
                peers = data.get("peers", 0)
                print(f"  PASS: Worker {name} received topology ({peers} peers)")
            else:
                print(f"  FAIL: Worker {name} not ready")
                all_ok = False

        if all_ok:
            print()
            print("  ALL MULTI-JOB TESTS PASSED!")
        else:
            print()
            print("  SOME TESTS FAILED — check logs above")

    finally:
        if args.keep:
            print()
            for name, job in jobs.items():
                print(f"  {name}: ssh -p {job.ssh_port} root@{job.vm_ip}")
        else:
            print()
            print("Tearing down all VMs...")
            for name, job in jobs.items():
                try:
                    job.teardown()
                    print(f"  {name}: torn down")
                except Exception as e:
                    print(f"  {name}: teardown failed ({e})")
            print("Done.")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
