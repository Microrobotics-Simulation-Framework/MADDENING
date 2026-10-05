#!/usr/bin/env python3
"""Launch a MADDENING simulation server on a cloud GPU and test it.

Uses the SSH-based approach: provisions a VM via SkyPilot, then runs
pip install and server start directly via SSH (bypassing Ray's job
scheduler which has GPU isolation issues).

``--local`` runs the same server script on this machine instead -- on a
free loopback port, no cloud account, no GPU -- and runs the same
endpoint checks against it.  It never imports the cloud launcher.

Usage:
    # No cloud account needed:
    python -m maddening.examples.cloud.server.04_server_test --local

    # Provisions a billable RunPod VM (needs ~/.maddening/cloud_credentials.yaml):
    python -m maddening.examples.cloud.server.04_server_test
    python -m maddening.examples.cloud.server.04_server_test --gpu RTX4090
    python -m maddening.examples.cloud.server.04_server_test --keep   # don't teardown
"""

import argparse
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error


# -- Remote install + server script ------------------------------------
# Run via SSH directly, not through Ray.
#
# Interpreter: on ``runpod/base`` the system ``python3`` is 3.10 but
# ``pip`` targets 3.12.  MADDENING requires Python >= 3.12 and the JAX
# range below (``pyproject`` ``cuda12`` extra, ``jax>=0.10,<0.13``)
# reaches 0.11, which requires >= 3.12 too, so *neither* installs under
# 3.10 or 3.11 -- every remote command here uses ``PYTHON`` (3.12)
# explicitly.  This example
# needs no GStreamer/gi bindings, so the interpreter choice is free.
PYTHON = "python3.12"

# If using the pre-built Docker image, MADDENING is already installed.
# Only pip install if needed (bare image or missing deps).
INSTALL_CMD = (
    f"{PYTHON} -c 'from maddening import GraphManager; print(\"MADDENING pre-installed\")' 2>/dev/null"
    " && echo INSTALL_DONE"
    " || ("
    f"  {PYTHON} -m pip install -q --root-user-action=ignore"
    '  "jax[cuda12]>=0.10,<0.13"'
    '  "fastapi>=0.100" "uvicorn>=0.20" "websockets>=11.0"'
    '  "numpy>=1.24" "pyyaml>=6.0" "rich>=12.0" "matplotlib>=3.5" "pyzmq>=25.0"'
    f"  && [ -d ~/sky_workdir/src ] && {PYTHON} -m pip install -q --root-user-action=ignore -e ~/sky_workdir"
    "  ; echo INSTALL_DONE"
    ")"
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
gm.add_node(BallNode(name="ball", timestep=0.01, initial_position=5.0))
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
    # The bind address has to be handed to the server: a non-loopback one
    # turns on the bearer token, and the app cannot see the socket.
    bind_host=HOST,
)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((HOST, PORT))
print(f"Serving on {HOST}:{sock.getsockname()[1]}", flush=True)
try:
    uvicorn.Server(uvicorn.Config(server.create_app(), log_level="info",
                                  proxy_headers=False)).run(
        sockets=[sock])
except KeyboardInterrupt:
    pass
"""


def _request(base_url: str, path: str, method: str) -> urllib.request.Request:
    """A request carrying the bearer token the VM's API demands."""
    return urllib.request.Request(
        f"{base_url}{path}", method=method,
        headers={"Authorization": f"Bearer {API_TOKEN}"},
    )


def wait_for_server(base_url: str, timeout: float = 120) -> bool:
    """Poll the server until it responds or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                _request(base_url, "/graph", "GET"), timeout=5,
            ) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                # A live server refusing the token is not "not up yet".
                print("  ERROR: server rejected the bearer token (401)")
                return False
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(3)
    return False


def http_get(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(_request(base_url, path, "GET"), timeout=10) as resp:
        return json.loads(resp.read())


def http_post(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(_request(base_url, path, "POST"), timeout=10) as resp:
        return json.loads(resp.read())


def check_endpoints(base_url: str) -> None:
    """The endpoint checks, shared by the cloud and the --local runs."""
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
    start_pos = state.get("ball", {}).get("position")
    print(f"    Ball position: {start_pos}")
    print(f"    Ball velocity: {state.get('ball', {}).get('velocity')}")
    assert start_pos is not None, "Expected ball position"
    print("    PASS")

    print("  POST /sim/step (5 steps)...")
    for _ in range(5):
        state = http_post(base_url, "/sim/step")
    ball_pos = state.get("ball", {}).get("position")
    print(f"    Ball position after 5 steps: {ball_pos}")
    # Released from 5 m, the ball has fallen a little.
    assert ball_pos is not None and ball_pos < start_pos, (
        f"Expected the ball to fall from {start_pos}, got {ball_pos}"
    )
    print("    PASS")

    print("  POST /sim/run (100 steps)...")
    state = http_post(base_url, "/sim/run?n_steps=100")
    after_run = state.get("ball", {}).get("position")
    print(f"    Ball position after 100 more steps: {after_run}")
    assert after_run is not None and after_run != ball_pos, (
        f"Expected /sim/run to advance the ball from {ball_pos}"
    )
    print("    PASS")


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
    """Run SERVER_SCRIPT on this machine and test it.  No cloud involved."""
    print("Local run: starting the server script on a free loopback port...")
    proc, base_url = start_local_server(SERVER_SCRIPT)
    print(f"  Server pid {proc.pid} at {base_url}")
    try:
        print("Waiting for server to respond...")
        if not wait_for_server(base_url, timeout=120):
            print("  ERROR: Server did not respond within 120s")
            return 1
        print("  Server is UP!")
        print()
        print("Testing API endpoints...")
        check_endpoints(base_url)
    finally:
        code = _stop_process(proc)
        print(f"  Local server (pid {proc.pid}) stopped, exit code {code}")
    print()
    print("All server tests passed!")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Cloud server test")
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
        # JobConfig.ports is empty by default so a launch does not open
        # the API in the provider's firewall.  This demo needs the public
        # NAT mapping, so it asks for it explicitly.
        ports=[8000],
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
    print(f"  (Installing into {PYTHON}; JAX and MADDENING need Python >= 3.12)")
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
            f'{PYTHON} -c "import jax; print(jax.devices()); '
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
    escaped = shlex.quote(SERVER_SCRIPT)
    job.ssh_run(f"echo {escaped} > /tmp/maddening_server.py", check=True)
    # The token goes over ssh's stdin, not in the command string: a
    # remote `VAR=value cmd` puts the secret in /proc/<pid>/cmdline,
    # which is mode 0444 for as long as the shell lives.  shlex.quote
    # stops word splitting and does nothing about who can read it.
    job.ssh_run_background(
        f"{PYTHON} /tmp/maddening_server.py",
        env={"MADDENING_API_TOKEN": API_TOKEN},
    )
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
    check_endpoints(base_url)

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
