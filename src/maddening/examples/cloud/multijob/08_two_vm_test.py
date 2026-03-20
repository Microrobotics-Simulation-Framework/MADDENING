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

coord = Coordinator(
    expected_workers=expected,
    edges=edges,
    port=port,
    heartbeat_timeout=60.0,
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

client = WorkerClient(
    coordinator_addr=coordinator_addr,
    subgraph_id=subgraph_id,
    address=f"{{my_ip}}:5555",
    zmq_ports={{"state": 5555}},
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

        # Generate coordinator script with embedded values (no env vars needed)
        edges_list = [{
            "source": "flow", "target": "structure",
            "source_field": "pressure", "target_field": "load",
        }, {
            "source": "structure", "target": "flow",
            "source_field": "displacement", "target_field": "wall_bc",
        }]

        coord_script = COORDINATOR_SCRIPT_TEMPLATE.format(
            expected_json=json.dumps(["flow", "structure"]),
            edges_json=json.dumps(edges_list),
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


if __name__ == "__main__":
    main()
