#!/usr/bin/env python3
"""Test WebSocket state streaming (JSON and binary) from a cloud GPU server.

Launches a VM, starts a MADDENING simulation server, connects to
the JSON and binary WebSocket endpoints, verifies frames are received
with the right schema, then tears down.  Exits non-zero if a check fails.

``--local`` runs the same server script on this machine instead -- on a
free loopback port, no cloud account, no GPU -- and runs the same
WebSocket checks against it.  It never imports the cloud launcher.

Usage:
    # No cloud account needed:
    python -m maddening.examples.cloud.server.05_websocket_test --local

    # Provisions a billable RunPod VM (needs ~/.maddening/cloud_credentials.yaml):
    python -m maddening.examples.cloud.server.05_websocket_test
    python -m maddening.examples.cloud.server.05_websocket_test --gpu RTX4090
    python -m maddening.examples.cloud.server.05_websocket_test --keep   # don't teardown
"""

import argparse
import asyncio
import json
import os
import re
import secrets
import shlex
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.request


# Reuse the server setup from 04_server_test
INSTALL_CMD = (
    "pip install -q --root-user-action=ignore"
    ' "jax[cuda12]>=0.10,<0.13"'
    ' "fastapi>=0.100" "uvicorn>=0.20" "websockets>=11.0"'
    ' "numpy>=1.24" "pyyaml>=6.0" "rich>=12.0" "matplotlib>=3.5" "pyzmq>=25.0"'
    " && [ -d ~/sky_workdir/src ] && pip install -q --root-user-action=ignore -e ~/sky_workdir"
    " ; echo INSTALL_DONE"
)


# The VM's API binds 0.0.0.0, which is not loopback, so it requires a
# bearer token on every route.  This script chooses the token, passes it
# to the remote process in MADDENING_API_TOKEN, and presents it on every
# request -- there is no TLS, so treat the endpoint as a demo on a
# throwaway VM rather than a deployment pattern.
API_TOKEN = secrets.token_urlsafe(32)

# Host and port come from the environment so that --local can run this
# exact script on a free loopback port; on the VM the defaults apply.
SERVER_SCRIPT = r"""
import os, socket
HOST = os.environ.get("MADDENING_DEMO_HOST", "0.0.0.0")
PORT = int(os.environ.get("MADDENING_DEMO_PORT", "8000"))

# --local: stop if the process that started us goes away.
_parent = os.environ.get("MADDENING_DEMO_PARENT_PID")
if _parent:
    import signal, threading, time
    def _watch_parent():
        while os.getppid() == int(_parent):
            time.sleep(1.0)
        os.kill(os.getpid(), signal.SIGINT)
    threading.Thread(target=_watch_parent, daemon=True).start()

import jax, warnings
print(f"JAX: {jax.devices()}")
from maddening import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode
from maddening.api.server import SimulationServer
import uvicorn

gm = GraphManager()
gm.add_node(BallNode(name="ball", timestep=0.01, initial_position=5.0))
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
    # The bind address has to be handed to the server: a non-loopback one
    # turns on the bearer token, and the app cannot see the socket.
    bind_host=HOST,
)

# The client starts the server's own runner with POST /sim/start.  (A
# second RealtimeRunner here would step the same graph from two threads.)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((HOST, PORT))
print(f"Serving on {HOST}:{sock.getsockname()[1]}", flush=True)
try:
    uvicorn.Server(uvicorn.Config(server.create_app(), log_level="warning")).run(
        sockets=[sock])
except KeyboardInterrupt:
    pass
"""


def _ws_auth() -> dict:
    """Handshake kwargs carrying the bearer token.

    A non-browser client can set the header, which is the carrier to
    prefer: unlike a query parameter it does not reach an access log.
    Browsers, which cannot set one, use the ``maddening.bearer.*``
    subprotocol instead -- see ``maddening.api.auth``.
    """
    return {"additional_headers": {"Authorization": f"Bearer {API_TOKEN}"}}


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

    async with websockets.connect(ws_url, **_ws_auth()) as ws:
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

    async with websockets.connect(ws_url, **_ws_auth()) as ws:
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


def _post(base_url: str, path: str) -> None:
    urllib.request.urlopen(
        urllib.request.Request(
            f"{base_url}{path}", method="POST",
            headers={"Authorization": f"Bearer {API_TOKEN}"},
        ), timeout=10,
    )


def wait_for_server(base_url: str, timeout: float = 120) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(
                f"{base_url}/graph", method="GET",
                headers={"Authorization": f"Bearer {API_TOKEN}"},
            )
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            time.sleep(1)
    return False


def run_ws_checks(base_url: str) -> bool:
    """Start the simulation, check both WebSocket endpoints, stop it.

    Shared by the cloud and the --local runs.  Returns True if every
    check passed.
    """
    ws_base = base_url.replace("http://", "ws://")
    ws_json_url = f"{ws_base}/ws/state"
    ws_binary_url = f"{ws_base}/ws/state/binary"
    print(f"  WS JSON: {ws_json_url}")
    print(f"  WS Binary: {ws_binary_url}")

    print("\nStarting continuous simulation...")
    _post(base_url, "/sim/start")
    time.sleep(1)  # Let a few steps accumulate
    failures = []

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
        failures.append("json")

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
        failures.append("binary")

    # --- Stop simulation ---
    # The server's stream handlers notice a departed client only when
    # they next send a frame, so keep the simulation running a moment
    # longer: a handler still waiting when frames stop never exits, and
    # then holds up the server's shutdown.
    time.sleep(0.5)
    print("\nStopping simulation...")
    try:
        _post(base_url, "/sim/stop")
    except Exception:
        pass
    if failures:
        print(f"\nWebSocket tests FAILED: {', '.join(failures)}")
    return not failures


def _stop_process(proc: subprocess.Popen, grace: float = 10.0) -> int:
    """Stop *proc* and reap it: SIGINT, then SIGTERM, then SIGKILL."""
    if proc.poll() is None:
        steps = [proc.terminate, proc.kill]
        if os.name == "posix":
            steps.insert(0, lambda: proc.send_signal(signal.SIGINT))
        for step in steps:
            step()
            try:
                proc.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
    return proc.wait()


def start_local_server(script: str, startup_timeout: float = 180.0):
    """Run *script* in a subprocess on a free loopback port.

    Returns ``(process, base_url)``.  The script binds port 0 and prints
    the port the OS gave it, so nothing can take the port in between.
    """
    env = dict(os.environ)
    env.update({
        "MADDENING_DEMO_HOST": "127.0.0.1",
        "MADDENING_DEMO_PORT": "0",
        "MADDENING_DEMO_PARENT_PID": str(os.getpid()),
        "MADDENING_API_TOKEN": API_TOKEN,
    })
    proc = subprocess.Popen(
        [sys.executable, "-c", script], env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        # Its own session: a Ctrl-C here reaches only this script, which
        # then stops the server in its ``finally``.
        start_new_session=(os.name == "posix"),
    )
    output: list[str] = []
    port: list[int] = []
    done = threading.Event()

    def _drain():
        # Read for the server's whole life, so a full pipe never blocks it.
        for line in proc.stdout:
            output.append(line)
            m = re.search(r"Serving on 127\.0\.0\.1:(\d+)", line)
            if m and not port:
                port.append(int(m.group(1)))
                done.set()
        done.set()

    threading.Thread(target=_drain, daemon=True).start()
    done.wait(startup_timeout)
    if not port:
        code = _stop_process(proc)
        raise RuntimeError(
            f"local server did not start (exit code {code}); last output:\n"
            + "".join(output[-20:])
        )
    return proc, f"http://127.0.0.1:{port[0]}"


def run_local() -> int:
    """Run SERVER_SCRIPT on this machine and check its WebSockets."""
    print("Local run: starting the server script on a free loopback port...")
    proc, base_url = start_local_server(SERVER_SCRIPT)
    print(f"  Server pid {proc.pid} at {base_url}")
    try:
        if not wait_for_server(base_url):
            print("ERROR: Server didn't respond")
            return 1
        print("  Server is UP!")
        ok = run_ws_checks(base_url)
    finally:
        code = _stop_process(proc)
        print(f"  Local server (pid {proc.pid}) stopped, exit code {code}")
    if not ok:
        return 1
    print("\nAll WebSocket tests passed!")
    return 0


def main():
    parser = argparse.ArgumentParser(description="WebSocket streaming test")
    parser.add_argument("--gpu", default="RTX4090", help="GPU type")
    parser.add_argument("--keep", action="store_true", help="Don't teardown")
    parser.add_argument(
        "--local", action="store_true",
        help="Run the server script on this machine (loopback, no cloud)",
    )
    args = parser.parse_args()

    if args.local:
        sys.exit(run_local())

    # Imported here so that --local never loads the cloud launcher.
    from maddening.cloud.launcher import (
        CloudLauncher,
        CostPolicy,
        JobConfig,
        LaunchError,
    )

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
        # JobConfig.ports is empty by default so a launch does not open
        # the API in the provider's firewall.  This demo needs the public
        # NAT mapping, so it asks for it explicitly.
        ports=[8000],
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
    job.ssh_run(f"echo {shlex.quote(SERVER_SCRIPT)} > /tmp/maddening_server.py", check=True)
    # Over ssh's stdin, not in the command string: see 04_server_test.py.
    job.ssh_run_background(
        "python3.12 /tmp/maddening_server.py",
        env={"MADDENING_API_TOKEN": API_TOKEN},
    )

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

    print("\nWaiting for server...")
    if not wait_for_server(base_url):
        print("ERROR: Server didn't respond")
        if not args.keep:
            job.teardown()
        sys.exit(1)
    print("  Server is UP!")

    ok = run_ws_checks(base_url)

    # --- Teardown ---
    if args.keep:
        print(f"\nKeeping alive: {job.cluster_name}")
        print(f"  HTTP: {base_url}")
        print(f"  SSH: ssh -p {job.ssh_port} root@{job.vm_ip}")
    else:
        print("\nTearing down...")
        job.teardown()
        print("  Done.")

    if not ok:
        sys.exit(1)
    print("\nAll WebSocket tests passed!")


if __name__ == "__main__":
    main()
