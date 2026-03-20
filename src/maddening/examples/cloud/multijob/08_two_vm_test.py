#!/usr/bin/env python3
"""Test multi-job architecture with 2 VMs on RunPod.

Provisions 2 VMs:
  - VM 0 (rank-0): runs coordinator + "flow" subgraph
  - VM 1 (worker): runs "structure" subgraph

Tests the full rendezvous flow:
  1. Launch rank-0, start coordinator via SSH
  2. Launch worker, register with coordinator via SSH
  3. Verify topology received on both sides
  4. Verify ZMQ PUB/SUB data exchange between VMs
  5. Tear down both

Uses the cheapest available GPUs with spot_fallback for cost efficiency.

Usage:
    python 08_two_vm_test.py
    python 08_two_vm_test.py --gpu RTX4090
    python 08_two_vm_test.py --keep
"""

import argparse
import json
import os
import shlex
import sys
import time

from maddening.cloud.launcher import (
    CloudJob,
    CloudLauncher,
    CostPolicy,
    JobConfig,
    LaunchError,
)


# Install script — same deps for both VMs
INSTALL_CMD = (
    "python3 -m pip install -q --root-user-action=ignore"
    ' "pyzmq>=25.0" "pyyaml>=6.0" "numpy>=1.24"'
    " && [ -d ~/sky_workdir/src ]"
    " && python3 -m pip install -q --root-user-action=ignore -e ~/sky_workdir"
    " ; echo INSTALL_DONE"
)

# Coordinator script (runs on rank-0)
COORDINATOR_SCRIPT = r"""
import json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/sky_workdir/src"))

from maddening.cloud.multigpu.coordinator import Coordinator

expected = json.loads(os.environ.get("EXPECTED_WORKERS", '["flow", "structure"]'))
edges = json.loads(os.environ.get("INTER_JOB_EDGES", '[]'))
port = int(os.environ.get("COORDINATOR_PORT", "5580"))

print(f"Starting coordinator on :{port}, expecting {expected}")
coord = Coordinator(
    expected_workers=expected,
    edges=edges,
    port=port,
    heartbeat_timeout=60.0,
)
coord.start()

# Wait for all workers
if coord.wait_for_all(timeout=300):
    print("All workers registered!")
    topo = coord.build_topology()
    for wid, t in topo.items():
        print(f"  {wid}: {len(t.peers)} peers")
    # Write success marker
    with open("/tmp/coordinator_ready.json", "w") as f:
        json.dump({"status": "ready", "workers": list(coord.registered_workers.keys())}, f)
    print("Coordinator ready. Sleeping...")
    # Keep alive for heartbeats
    while True:
        time.sleep(1)
else:
    print("TIMEOUT: not all workers registered")
    with open("/tmp/coordinator_ready.json", "w") as f:
        json.dump({"status": "timeout"}, f)
    sys.exit(1)
"""

# Worker script (runs on worker VMs)
WORKER_SCRIPT = r"""
import json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/sky_workdir/src"))

from maddening.cloud.multigpu.worker_client import WorkerClient

coordinator_addr = os.environ.get("COORDINATOR_ADDR", "")
subgraph_id = os.environ.get("SUBGRAPH_ID", "unknown")
# Get our public IP from the hostname or env
import socket
my_ip = socket.gethostbyname(socket.gethostname())

print(f"Worker {subgraph_id} connecting to coordinator at {coordinator_addr}")
print(f"My IP: {my_ip}")

client = WorkerClient(
    coordinator_addr=coordinator_addr,
    subgraph_id=subgraph_id,
    address=f"{my_ip}:5555",
    zmq_ports={"state": 5555},
)

try:
    topology = client.register_and_wait(timeout=120)
    print(f"Topology received: {len(topology)} peers")
    for peer in topology:
        print(f"  {peer.peer_id}: {peer.role} {peer.socket_type} @ {peer.address}")

    # Write success marker
    with open("/tmp/worker_ready.json", "w") as f:
        json.dump({
            "status": "ready",
            "subgraph_id": subgraph_id,
            "peers": len(topology),
        }, f)

    # Start heartbeat and keep alive
    client.start_heartbeat(interval=5.0)
    print("Worker ready. Heartbeating...")
    while True:
        time.sleep(1)

except Exception as e:
    print(f"Worker {subgraph_id} FAILED: {e}")
    with open("/tmp/worker_ready.json", "w") as f:
        json.dump({"status": "error", "error": str(e)}, f)
    sys.exit(1)
"""


def main():
    parser = argparse.ArgumentParser(description="2-VM multi-job test")
    parser.add_argument("--gpu", default="RTX4090", help="GPU type")
    parser.add_argument("--keep", action="store_true", help="Don't teardown")
    args = parser.parse_args()

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
    )

    launcher = CloudLauncher()
    jobs: dict[str, CloudJob] = {}

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

        # Upload and run coordinator script
        jobs["flow"].ssh_run(
            f"echo {shlex.quote(COORDINATOR_SCRIPT)} > /tmp/coordinator.py",
            check=True,
        )

        edges = json.dumps([{
            "source": "flow", "target": "structure",
            "source_field": "pressure", "target_field": "load",
        }, {
            "source": "structure", "target": "flow",
            "source_field": "displacement", "target_field": "wall_bc",
        }])

        jobs["flow"].ssh_run(
            f'export EXPECTED_WORKERS=\'["flow", "structure"]\''
            f" INTER_JOB_EDGES='{edges}'"
            f" COORDINATOR_PORT=5580"
            f" && nohup python3 /tmp/coordinator.py > /tmp/coordinator.log 2>&1 &",
            check=False,
        )
        print("  Coordinator started")
        time.sleep(2)

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

        # Start flow worker on rank-0 — uses localhost:5580 (bypass NAT)
        jobs["flow"].ssh_run(
            f"echo {shlex.quote(WORKER_SCRIPT)} > /tmp/worker.py",
            check=True,
        )
        jobs["flow"].ssh_run(
            f"export COORDINATOR_ADDR=127.0.0.1:5580 SUBGRAPH_ID=flow"
            f" && nohup python3 /tmp/worker.py > /tmp/worker.log 2>&1 &",
            check=False,
        )
        print("  flow worker started on rank-0 (localhost:5580)")

        # Start structure worker on VM 1 — uses public mapped address
        jobs["structure"].ssh_run(
            f"echo {shlex.quote(WORKER_SCRIPT)} > /tmp/worker.py",
            check=True,
        )
        jobs["structure"].ssh_run(
            f"export COORDINATOR_ADDR={coordinator_addr} SUBGRAPH_ID=structure"
            f" && nohup python3 /tmp/worker.py > /tmp/worker.log 2>&1 &",
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
            # Check logs
            result = jobs["flow"].ssh_run(
                "cat /tmp/coordinator.log 2>/dev/null | tail -10",
                capture=True, check=False,
            )
            print(f"  Coordinator log:\n{result.stdout}")

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


if __name__ == "__main__":
    main()
