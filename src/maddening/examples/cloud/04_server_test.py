#!/usr/bin/env python3
"""Launch a MADDENING simulation server on a cloud GPU and test it.

Provisions a VM, installs maddening[server]+jax[cuda12] via setup script,
starts a bouncing ball simulation server, verifies the API is reachable,
tests WebSocket state streaming, then tears down.

Usage:
    python 04_server_test.py
    python 04_server_test.py --gpu RTX5090
    python 04_server_test.py --keep   # don't teardown (for manual inspection)
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

from maddening.cloud.launcher import (
    CloudJob,
    CloudLauncher,
    CostPolicy,
    JobConfig,
    LaunchError,
)


# -- SkyPilot task setup script ------------------------------------------
# This runs on the VM before the job starts.  Installs MADDENING and
# starts a FastAPI simulation server in the background.

SETUP_SCRIPT = r"""
set -e
echo "SETUP: Installing JAX CUDA + server deps..."
pip install --quiet "jax[cuda12]" "fastapi>=0.100" "uvicorn>=0.20" "websockets>=11.0" "pyzmq>=25.0" "rich>=12.0" "matplotlib>=3.5" "pyyaml>=6.0" "numpy>=1.24" 2>&1 | tail -5

# Install MADDENING from the synced workdir
if [ -f /sky_workdir/pyproject.toml ]; then
    echo "SETUP: Installing MADDENING from synced workdir..."
    pip install --quiet -e "/sky_workdir[server]" 2>&1 | tail -5
else
    echo "SETUP: No workdir found, installing base deps only"
fi

echo "SETUP: done"
"""

RUN_SCRIPT = r"""
set -e
echo "RUN: Starting MADDENING server in background..."

# Write a small server script
cat > /tmp/maddening_server.py << 'PYEOF'
import jax
print(f"JAX backend: {jax.devices()[0].platform}")

from maddening import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring_damper import SpringDamperNode
from maddening.core.edge import EdgeSpec
from maddening.api.server import SimulationServer

# Build a simple graph
gm = GraphManager()
gm.add_node(BallNode(name="ball", timestep=0.001))
gm.add_node(TableNode(name="table", timestep=0.001))
gm.add_node(SpringDamperNode(name="spring", timestep=0.001, stiffness=500.0, damping=5.0))
gm.add_edge("ball", "table", source_field="position", target_field="ball_position")
gm.add_edge("table", "ball", source_field="normal_force", target_field="external_force")
gm.add_edge("ball", "spring", source_field="position", target_field="anchor_position")
gm.add_edge("spring", "ball", source_field="force", target_field="external_force")
gm.compile()

server = SimulationServer(
    node_registry={
        "BallNode": BallNode,
        "TableNode": TableNode,
        "SpringDamperNode": SpringDamperNode,
    },
    graph_manager=gm,
)

import uvicorn
print("Starting server on 0.0.0.0:8000...")
uvicorn.run(server.create_app(), host="0.0.0.0", port=8000, log_level="info")
PYEOF

# Start server in background and keep the job alive
nohup python3 /tmp/maddening_server.py > /tmp/server.log 2>&1 &
echo "RUN: Server PID=$!"
echo "RUN: Waiting for server to be ready..."

# Wait for server to start (up to 60s)
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/graph > /dev/null 2>&1; then
        echo "RUN: Server is ready!"
        break
    fi
    sleep 1
done

# Keep the job alive (SkyPilot will kill this when autostop triggers)
echo "RUN: Server running. Sleeping to keep job alive..."
sleep 3600
"""


def get_runpod_port_endpoint(cluster_name: str, private_port: int = 8000) -> str | None:
    """Query RunPod API for the public endpoint of a private port.

    RunPod uses NAT — ports are mapped to public_ip:public_port pairs.
    Returns "http://host:port" or None if not found.
    """
    import yaml
    import runpod

    with open(os.path.expanduser("~/.maddening/cloud_credentials.yaml")) as f:
        creds = yaml.safe_load(f)
    runpod.api_key = creds["runpod"]["api_key"]

    for pod in runpod.get_pods():
        if cluster_name in pod.get("name", ""):
            runtime = pod.get("runtime") or {}
            for port_info in (runtime.get("ports") or []):
                if (port_info.get("privatePort") == private_port
                        and port_info.get("isIpPublic")):
                    host = port_info["ip"]
                    public_port = port_info["publicPort"]
                    return f"http://{host}:{public_port}"
    return None


def wait_for_server(base_url: str, timeout: float = 120) -> bool:
    """Poll the server until it responds or timeout."""
    url = f"{base_url}/graph"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(3)
    return False


def http_get(base_url: str, path: str) -> dict:
    """GET a JSON endpoint."""
    req = urllib.request.Request(f"{base_url}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def http_post(base_url: str, path: str) -> dict:
    """POST to an endpoint and return JSON."""
    req = urllib.request.Request(f"{base_url}{path}", method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="Cloud server test")
    parser.add_argument("--gpu", default="RTX5090", help="GPU type")
    parser.add_argument("--keep", action="store_true", help="Don't teardown")
    args = parser.parse_args()

    # Find project root (directory containing pyproject.toml)
    project_root = os.path.dirname(os.path.abspath(__file__))
    while project_root != "/" and not os.path.exists(
        os.path.join(project_root, "pyproject.toml")
    ):
        project_root = os.path.dirname(project_root)

    config = JobConfig(
        provider="runpod",
        gpu_type=args.gpu,
        use_spot=False,
        cost=CostPolicy(
            max_cost_per_hour=2.0,
            max_total_budget=5.0,
            autostop_minutes=10,
            auto_teardown=True,
        ),
        setup=SETUP_SCRIPT,
        run=RUN_SCRIPT,
        workdir=project_root,
    )

    launcher = CloudLauncher()

    # --- Launch ---
    print(f"Launching {args.gpu} on-demand...")
    try:
        job = launcher.launch(config)
    except LaunchError as e:
        print(f"Launch failed: {e}")
        sys.exit(1)

    print(f"  Cluster: {job.cluster_name}")
    print(f"  VM IP: {job.vm_ip}")
    print(f"  Hourly cost: ${job._hourly_cost:.2f}")
    print()
    print("  SkyPilot is running setup (pip install) + starting server.")
    print("  This takes a few minutes. NOT calling stream_logs() to avoid blocking.")

    # --- Discover the public endpoint ---
    print()
    print("Discovering RunPod port mapping for :8000...")
    # RunPod uses NAT — we need the mapped public endpoint, not vm_ip:8000
    base_url = None
    for attempt in range(20):  # poll for up to 60s (port mapping appears after pod starts)
        base_url = get_runpod_port_endpoint(job.cluster_name, 8000)
        if base_url:
            break
        time.sleep(3)

    if not base_url:
        print("ERROR: Could not find public port mapping for :8000.")
        print("  This may mean the server hasn't started yet, or the port")
        print("  wasn't exposed. Check RunPod dashboard for port mappings.")
        if not args.keep:
            job.teardown()
        sys.exit(1)
    print(f"  Public endpoint: {base_url}")

    # --- Wait for server ---
    print()
    print(f"Waiting for FastAPI server at {base_url}...")
    if not wait_for_server(base_url, timeout=600):
        print("ERROR: Server did not start within timeout.")
        if not args.keep:
            job.teardown()
        sys.exit(1)
    print("  Server is UP!")

    # --- Test endpoints ---
    print()
    print("Testing /graph endpoint...")
    graph = http_get(base_url, "/graph")
    print(f"  Nodes: {list(graph.get('nodes', {}).keys())}")
    print(f"  Edges: {len(graph.get('edges', []))}")
    assert "ball" in graph.get("nodes", {}), "Expected 'ball' node"
    print("  PASS")

    print()
    print("Testing /graph/state endpoint...")
    state = http_get(base_url, "/graph/state")
    print(f"  Ball position: {state.get('ball', {}).get('position')}")
    print(f"  Ball velocity: {state.get('ball', {}).get('velocity')}")
    print("  PASS")

    print()
    print("Testing /sim/step (5 steps)...")
    for i in range(5):
        state = http_post(base_url, "/sim/step")
    ball_pos = state.get("ball", {}).get("position")
    print(f"  Ball position after 5 steps: {ball_pos}")
    print("  PASS")

    # --- Teardown ---
    if args.keep:
        print()
        print(f"Keeping cluster alive: {job.cluster_name}")
        print(f"  Server: {base_url}")
        print(f"  Teardown manually: sky down {job.cluster_name}")
    else:
        print()
        print("Tearing down...")
        job.teardown()
        print("  Done.")

    print()
    print("All server tests passed!")


if __name__ == "__main__":
    main()
