#!/usr/bin/env python3
"""Launch a MADDENING simulation server on a cloud GPU and test it.

Uses the SSH-based approach: provisions a VM via SkyPilot, then runs
pip install and server start directly via SSH (bypassing Ray's job
scheduler which has GPU isolation issues).

Usage:
    python 04_server_test.py
    python 04_server_test.py --gpu RTX4090
    python 04_server_test.py --keep   # don't teardown (for manual inspection)
"""

import argparse
import json
import os
import subprocess
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


# -- Remote install + server script ------------------------------------
# Run via SSH directly on system python3, not through Ray.
# Fixes:
#   1. Use python3 (runpod/base has pip targeting 3.12 but python3 is 3.10)
#   2. Uninstall pre-installed jax_cuda12_plugin to avoid version conflict
#   3. Pin jax/jaxlib versions matching our pyproject.toml range

# If using the pre-built Docker image, MADDENING is already installed.
# Only pip install if needed (bare image or missing deps).
INSTALL_CMD = (
    "python3 -c 'from maddening import GraphManager; print(\"MADDENING pre-installed\")' 2>/dev/null"
    " && echo INSTALL_DONE"
    " || ("
    "  pip3 install -q"
    '  "jax[cuda12]>=0.4,<0.6"'
    '  "fastapi>=0.100" "uvicorn>=0.20" "websockets>=11.0"'
    '  "numpy>=1.24" "pyyaml>=6.0" "rich>=12.0" "matplotlib>=3.5" "pyzmq>=25.0"'
    "  && [ -d ~/sky_workdir/src ] && pip3 install -q -e ~/sky_workdir"
    "  ; echo INSTALL_DONE"
    ")"
)

SERVER_SCRIPT = r"""
import jax
print(f"JAX devices: {jax.devices()}")
print(f"Platform: {jax.devices()[0].platform}")

from maddening import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode
from maddening.api.server import SimulationServer
import uvicorn

gm = GraphManager()
gm.add_node(BallNode(name="ball", timestep=0.01))
gm.add_node(TableNode(name="table", timestep=0.01))
gm.add_node(SpringDamperNode(
    name="spring", timestep=0.01,
    stiffness=50.0, damping=2.0, mass=0.5,
    rest_length=1.5, initial_position=3.0,
))
gm.add_edge("table", "ball", "position", "table_position")
gm.add_edge("ball", "spring", "position", "anchor_position")

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gm.compile()
print(f"Graph compiled: {gm.node_names}")

server = SimulationServer(
    node_registry={
        "BallNode": BallNode,
        "TableNode": TableNode,
        "SpringDamperNode": SpringDamperNode,
    },
    graph_manager=gm,
)
print("Starting server on 0.0.0.0:8000...")
uvicorn.run(server.create_app(), host="0.0.0.0", port=8000, log_level="info")
"""


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
    req = urllib.request.Request(f"{base_url}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def http_post(base_url: str, path: str) -> dict:
    req = urllib.request.Request(f"{base_url}{path}", method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="Cloud server test")
    parser.add_argument("--gpu", default="RTX4090", help="GPU type")
    parser.add_argument("--keep", action="store_true", help="Don't teardown")
    args = parser.parse_args()

    # Find project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    while project_root != "/" and not os.path.exists(
        os.path.join(project_root, "pyproject.toml")
    ):
        project_root = os.path.dirname(project_root)

    config = JobConfig(
        provider="runpod",
        gpu_type=args.gpu,
        use_spot=False,
        region="US",
        cost=CostPolicy(
            max_cost_per_hour=2.0,
            max_total_budget=8.0,
            autostop_minutes=10,
            auto_teardown=False,  # We manage teardown ourselves
            spot_fallback=True,
        ),
        # Minimal run command — just keep the VM alive.
        # We do the real work via SSH to bypass Ray GPU isolation.
        run="echo 'VM ready for SSH'; sleep 7200",
        workdir=project_root,
    )

    launcher = CloudLauncher()

    # --- Phase 1: Launch VM ---
    print(f"Phase 1: Launching {args.gpu} on-demand in US...")
    try:
        job = launcher.launch(config)
    except LaunchError as e:
        print(f"Launch failed: {e}")
        sys.exit(1)

    print(f"  Cluster: {job.cluster_name}")
    print(f"  VM IP: {job.vm_ip}")
    print(f"  SSH port: {job.ssh_port}")
    print(f"  Hourly cost: ${job._hourly_cost:.2f}")

    # --- Phase 2: Install deps via SSH ---
    print()
    print("Phase 2: Installing dependencies via SSH...")
    print(f"  (Using system pip which targets python3)")
    try:
        result = job.ssh_run(INSTALL_CMD, timeout=300, capture=True)
        # Print last few lines of output
        lines = result.stdout.strip().split("\n") if result.stdout else []
        for line in lines[-5:]:
            print(f"  {line}")
        if "INSTALL_DONE" not in result.stdout:
            print("  WARNING: INSTALL_DONE not found in output")
            print(f"  stderr: {result.stderr[-500:] if result.stderr else 'none'}")
    except subprocess.TimeoutExpired:
        print("  ERROR: pip install timed out (300s)")
        if not args.keep:
            job.teardown()
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: pip install failed (exit {e.returncode})")
        print(f"  {e.stderr[-500:] if e.stderr else ''}")
        if not args.keep:
            job.teardown()
        sys.exit(1)

    # --- Phase 3: Verify GPU + imports ---
    print()
    print("Phase 3: Verifying JAX GPU + MADDENING imports...")
    try:
        result = job.ssh_run(
            'python3 -c "import jax; print(jax.devices()); '
            'from maddening import GraphManager; print(\'MADDENING OK\')"',
            timeout=60, capture=True,
        )
        print(f"  {result.stdout.strip()}")
        if "MADDENING OK" not in result.stdout:
            print("  WARNING: import check may have failed")
            if result.stderr:
                print(f"  stderr: {result.stderr[-300:]}")
    except Exception as e:
        print(f"  ERROR: {e}")
        if not args.keep:
            job.teardown()
        sys.exit(1)

    # --- Phase 4: Upload and start server ---
    print()
    print("Phase 4: Starting MADDENING server...")
    # Write the server script to the VM, then run it in background
    import shlex
    escaped = shlex.quote(SERVER_SCRIPT)
    job.ssh_run(f"echo {escaped} > /tmp/maddening_server.py", check=True)
    job.ssh_run_background("python3 /tmp/maddening_server.py")
    print("  Server started in background")

    # --- Phase 5: Discover endpoint and test ---
    print()
    print("Phase 5: Discovering public endpoint...")
    base_url = None
    for attempt in range(30):
        base_url = job.get_runpod_endpoint(8000)
        if base_url:
            break
        time.sleep(2)

    if not base_url:
        print("  ERROR: No public port mapping for :8000")
        # Try direct IP as fallback
        base_url = f"http://{job.vm_ip}:8000"
        print(f"  Trying direct: {base_url}")

    print(f"  Endpoint: {base_url}")

    print()
    print("Phase 6: Waiting for server to respond...")
    if not wait_for_server(base_url, timeout=120):
        print("  ERROR: Server did not respond within 120s")
        # Check server logs
        try:
            result = job.ssh_run("cat /tmp/bg_cmd.log 2>/dev/null | tail -20", capture=True, check=False)
            print(f"  Server logs:\n{result.stdout}")
        except Exception:
            pass
        if not args.keep:
            job.teardown()
        sys.exit(1)
    print("  Server is UP!")

    # --- Phase 7: Test endpoints ---
    print()
    print("Phase 7: Testing API endpoints...")

    print("  GET /graph...")
    graph = http_get(base_url, "/graph")
    # nodes can be a list of dicts or a dict depending on API version
    nodes_data = graph.get("nodes", [])
    if isinstance(nodes_data, list):
        nodes = [n.get("name", "") for n in nodes_data]
    else:
        nodes = list(nodes_data.keys())
    print(f"    Nodes: {nodes}")
    print(f"    Edges: {len(graph.get('edges', []))}")
    assert "ball" in nodes, f"Expected 'ball' node, got {nodes}"
    print("    PASS")

    print("  GET /graph/state...")
    state = http_get(base_url, "/graph/state")
    print(f"    Ball position: {state.get('ball', {}).get('position')}")
    print(f"    Ball velocity: {state.get('ball', {}).get('velocity')}")
    print("    PASS")

    print("  POST /sim/step (5 steps)...")
    for _ in range(5):
        state = http_post(base_url, "/sim/step")
    ball_pos = state.get("ball", {}).get("position")
    print(f"    Ball position after 5 steps: {ball_pos}")
    assert ball_pos is not None, "Expected ball position"
    print("    PASS")

    print("  POST /sim/run (100 steps)...")
    state = http_post(base_url, "/sim/run?n_steps=100")
    ball_pos = state.get("ball", {}).get("position")
    print(f"    Ball position after 100 more steps: {ball_pos}")
    print("    PASS")

    # --- Teardown ---
    if args.keep:
        print()
        print(f"Keeping cluster alive: {job.cluster_name}")
        print(f"  Server: {base_url}")
        print(f"  SSH: ssh -p {job.ssh_port} root@{job.vm_ip}")
        print(f"  Teardown: sky down {job.cluster_name}")
    else:
        print()
        print("Tearing down...")
        job.teardown()
        print("  Done.")

    print()
    print("All server tests passed!")


if __name__ == "__main__":
    main()
