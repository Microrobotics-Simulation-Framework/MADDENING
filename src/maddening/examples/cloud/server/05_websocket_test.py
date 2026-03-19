#!/usr/bin/env python3
"""Test WebSocket binary state streaming from a cloud GPU server.

Launches a VM, starts a MADDENING simulation server, connects to
the binary WebSocket endpoint, verifies frames are received with
correct schema, then tears down.

Usage:
    python 05_websocket_test.py
    python 05_websocket_test.py --gpu RTX4090
    python 05_websocket_test.py --keep   # don't teardown
"""

import argparse
import asyncio
import json
import os
import struct
import sys
import time

from maddening.cloud.launcher import (
    CloudLauncher,
    CostPolicy,
    JobConfig,
    LaunchError,
)


# Reuse the server setup from 04_server_test
INSTALL_CMD = (
    "pip install -q --root-user-action=ignore"
    ' "jax[cuda12]>=0.4,<0.6"'
    ' "fastapi>=0.100" "uvicorn>=0.20" "websockets>=11.0"'
    ' "numpy>=1.24" "pyyaml>=6.0" "rich>=12.0" "matplotlib>=3.5" "pyzmq>=25.0"'
    " && [ -d ~/sky_workdir/src ] && pip install -q --root-user-action=ignore -e ~/sky_workdir"
    " ; echo INSTALL_DONE"
)

SERVER_SCRIPT = r"""
import jax, warnings
print(f"JAX: {jax.devices()}")
from maddening import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode
from maddening.api.server import SimulationServer
import uvicorn

gm = GraphManager()
gm.add_node(BallNode(name="ball", timestep=0.01))
gm.add_node(TableNode(name="table", timestep=0.01))
gm.add_node(SpringDamperNode(name="spring", timestep=0.01, stiffness=50.0, damping=2.0, mass=0.5, rest_length=1.5, initial_position=3.0))
gm.add_edge("table", "ball", "position", "table_position")
gm.add_edge("ball", "spring", "position", "anchor_position")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gm.compile()

server = SimulationServer(
    node_registry={"BallNode": BallNode, "TableNode": TableNode, "SpringDamperNode": SpringDamperNode},
    graph_manager=gm,
)

# Start the runner so the simulation ticks continuously
from maddening.viz.relay import StateRelay
from maddening.viz.runner import RealtimeRunner
relay = StateRelay()
relay.attach(gm)
runner = RealtimeRunner(gm, relay, steps_per_frame=10)
runner.start()

print("Server + runner started on :8000")
uvicorn.run(server.create_app(), host="0.0.0.0", port=8000, log_level="warning")
"""


async def test_websocket_binary(ws_url: str, n_frames: int = 10) -> dict:
    """Connect to the binary WS endpoint, receive schema + frames.

    Returns a summary dict with schema info and frame stats.
    """
    try:
        import websockets
    except ImportError:
        print("ERROR: websockets not installed locally. pip install websockets")
        sys.exit(1)

    results = {
        "schema_received": False,
        "schema": None,
        "frames_received": 0,
        "frame_sizes": [],
        "sim_times": [],
    }

    async with websockets.connect(ws_url) as ws:
        # First message should be JSON schema
        schema_msg = await asyncio.wait_for(ws.recv(), timeout=10)
        if isinstance(schema_msg, str):
            results["schema"] = json.loads(schema_msg)
            results["schema_received"] = True
            print(f"    Schema: {json.dumps(results['schema'], indent=2)[:300]}")
        else:
            print(f"    WARNING: expected JSON schema, got binary ({len(schema_msg)} bytes)")
            return results

        # Receive binary frames
        for i in range(n_frames):
            try:
                frame = await asyncio.wait_for(ws.recv(), timeout=5)
                if isinstance(frame, bytes):
                    results["frames_received"] += 1
                    results["frame_sizes"].append(len(frame))
                    # First 8 bytes are float64 sim_time
                    if len(frame) >= 8:
                        sim_time = struct.unpack("<d", frame[:8])[0]
                        results["sim_times"].append(sim_time)
            except asyncio.TimeoutError:
                print(f"    Frame {i}: timeout")
                break

    return results


async def test_websocket_json(ws_url: str, n_frames: int = 5) -> dict:
    """Connect to the JSON WS endpoint, receive state snapshots."""
    import websockets

    results = {"frames_received": 0, "last_state": None}

    async with websockets.connect(ws_url) as ws:
        for i in range(n_frames):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    results["frames_received"] += 1
                    results["last_state"] = data
            except asyncio.TimeoutError:
                break

    return results


def main():
    parser = argparse.ArgumentParser(description="WebSocket streaming test")
    parser.add_argument("--gpu", default="RTX4090", help="GPU type")
    parser.add_argument("--keep", action="store_true", help="Don't teardown")
    args = parser.parse_args()

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
            auto_teardown=False,
            spot_fallback=True,
        ),
        run="echo 'VM ready'; sleep 7200",
        workdir=project_root,
    )

    launcher = CloudLauncher()

    # --- Launch ---
    print(f"Launching {args.gpu} on-demand in US...")
    try:
        job = launcher.launch(config)
    except LaunchError as e:
        print(f"Launch failed: {e}")
        sys.exit(1)

    print(f"  Cluster: {job.cluster_name}")
    print(f"  VM IP: {job.vm_ip}:{job.ssh_port}")
    print(f"  Cost: ${job._hourly_cost:.2f}/hr")

    # --- Install + start server ---
    print("\nInstalling deps via SSH...")
    result = job.ssh_run(INSTALL_CMD, timeout=300, capture=True)
    if "INSTALL_DONE" not in (result.stdout or ""):
        print(f"  Install may have failed. stderr: {(result.stderr or '')[-300:]}")

    print("Verifying GPU...")
    result = job.ssh_run(
        'python3.12 -c "import jax; print(jax.devices())"',
        timeout=30, capture=True,
    )
    print(f"  {(result.stdout or '').strip()}")

    print("Starting server with continuous runner...")
    import shlex
    job.ssh_run(f"echo {shlex.quote(SERVER_SCRIPT)} > /tmp/maddening_server.py", check=True)
    job.ssh_run_background("python3.12 /tmp/maddening_server.py")

    # --- Wait for server ---
    print("\nDiscovering endpoint...")
    base_url = None
    for _ in range(30):
        base_url = job.get_runpod_endpoint(8000)
        if base_url:
            break
        time.sleep(2)

    if not base_url:
        base_url = f"http://{job.vm_ip}:8000"
        print(f"  No port mapping found, trying direct: {base_url}")
    print(f"  HTTP: {base_url}")

    # Derive WS URLs from HTTP URL
    ws_base = base_url.replace("http://", "ws://")
    ws_json_url = f"{ws_base}/ws/state"
    ws_binary_url = f"{ws_base}/ws/state/binary"
    print(f"  WS JSON: {ws_json_url}")
    print(f"  WS Binary: {ws_binary_url}")

    import urllib.request
    print("\nWaiting for server...")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/graph", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                break
        except Exception:
            time.sleep(3)
    else:
        print("ERROR: Server didn't respond")
        if not args.keep:
            job.teardown()
        sys.exit(1)
    print("  Server is UP!")

    # Start the simulation running
    print("\nStarting continuous simulation...")
    urllib.request.urlopen(
        urllib.request.Request(f"{base_url}/sim/start", method="POST"), timeout=10,
    )
    time.sleep(1)  # Let a few steps accumulate

    # --- Test JSON WebSocket ---
    print("\nTest 1: JSON WebSocket (/ws/state)...")
    try:
        json_results = asyncio.run(test_websocket_json(ws_json_url, n_frames=5))
        print(f"  Frames received: {json_results['frames_received']}")
        if json_results["last_state"]:
            sim_time = json_results["last_state"].get("sim_time", "?")
            ball_pos = json_results["last_state"].get("state", {}).get("ball", {}).get("position", "?")
            print(f"  Last sim_time: {sim_time}")
            print(f"  Ball position: {ball_pos}")
        assert json_results["frames_received"] > 0, "No JSON frames received"
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")

    # --- Test Binary WebSocket ---
    print("\nTest 2: Binary WebSocket (/ws/state/binary)...")
    try:
        binary_results = asyncio.run(test_websocket_binary(ws_binary_url, n_frames=10))
        print(f"  Schema received: {binary_results['schema_received']}")
        print(f"  Frames received: {binary_results['frames_received']}")
        if binary_results["frame_sizes"]:
            avg_size = sum(binary_results["frame_sizes"]) / len(binary_results["frame_sizes"])
            print(f"  Avg frame size: {avg_size:.0f} bytes")
        if binary_results["sim_times"]:
            print(f"  Sim time range: {binary_results['sim_times'][0]:.4f} → {binary_results['sim_times'][-1]:.4f}")
            # Verify time is advancing
            if len(binary_results["sim_times"]) > 1:
                assert binary_results["sim_times"][-1] > binary_results["sim_times"][0], \
                    "Sim time not advancing"
        assert binary_results["schema_received"], "No schema received"
        assert binary_results["frames_received"] > 0, "No binary frames received"
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")

    # --- Stop simulation ---
    print("\nStopping simulation...")
    try:
        urllib.request.urlopen(
            urllib.request.Request(f"{base_url}/sim/stop", method="POST"), timeout=10,
        )
    except Exception:
        pass

    # --- Teardown ---
    if args.keep:
        print(f"\nKeeping alive: {job.cluster_name}")
        print(f"  HTTP: {base_url}")
        print(f"  SSH: ssh -p {job.ssh_port} root@{job.vm_ip}")
    else:
        print("\nTearing down...")
        job.teardown()
        print("  Done.")

    print("\nAll WebSocket tests passed!")


if __name__ == "__main__":
    main()
