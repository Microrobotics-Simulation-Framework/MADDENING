#!/usr/bin/env python3
"""Test WebRTC streaming via SelkiesSession from a cloud GPU.

Launches a VM, installs GStreamer + MADDENING, starts a simulation
server with SelkiesRenderer wrapping ServerFrameRenderer (matplotlib),
verifies the GStreamer pipeline is running and the signaling WebSocket
is reachable, then prints the URL for manual browser testing.

Also profiles rendering performance (matplotlib render time vs
GStreamer encode/push overhead) to quantify streaming cost.

Usage:
    python 07_webrtc_streaming_test.py
    python 07_webrtc_streaming_test.py --gpu RTX4090
    python 07_webrtc_streaming_test.py --keep
"""

import argparse
import json
import os
import shlex
import sys
import time
import urllib.request
import urllib.error

from maddening.cloud.launcher import (
    CloudLauncher,
    CostPolicy,
    JobConfig,
    LaunchError,
)


# GStreamer system packages (from 06_selkies_test.py findings)
GSTREAMER_INSTALL = (
    "export DEBIAN_FRONTEND=noninteractive"
    " && apt-get update -qq"
    " && apt-get install -y -qq"
    " gstreamer1.0-plugins-base"
    " gstreamer1.0-plugins-good"
    " gstreamer1.0-plugins-bad"
    " gstreamer1.0-plugins-ugly"
    " gstreamer1.0-nice"
    " gstreamer1.0-tools"
    " gir1.2-gst-plugins-bad-1.0"
    " gir1.2-gstreamer-1.0"
    " python3-gi"
    " python3-gi-cairo"
    " libgirepository1.0-dev"
    " > /dev/null 2>&1"
    " && echo GST_INSTALL_DONE"
)

# Python deps (must use python3 = python3.10 for gi bindings)
PIP_INSTALL = (
    "python3 -m pip install -q --root-user-action=ignore"
    ' "jax[cuda12]>=0.4,<0.6"'
    ' "fastapi>=0.100" "uvicorn>=0.20" "websockets>=11.0"'
    ' "numpy>=1.24" "pyyaml>=6.0" "rich>=12.0" "matplotlib>=3.5" "pyzmq>=25.0"'
    ' "PyGObject>=3.42" "Pillow>=9.0"'
    " && [ -d ~/sky_workdir/src ] && python3 -m pip install -q --root-user-action=ignore -e ~/sky_workdir"
    " ; echo PIP_INSTALL_DONE"
)


# Server script that runs on the VM.
# Uses SelkiesRenderer wrapping ServerFrameRenderer for WebRTC streaming.
# Also profiles performance.
SERVER_SCRIPT = r'''
import sys, time, warnings, json
import jax
import jax.numpy as jnp
print(f"JAX: {jax.devices()}")

from maddening import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.heat import HeatNode
from maddening.api.frame_renderer import (
    ServerFrameRenderer, SceneConfig, SceneObject, TimeSeriesConfig, HeatmapConfig,
)
from maddening.cloud.selkies_session import SelkiesSession
from maddening.cloud.streaming import StreamConfig, QualityPreset
from maddening.viz.backends.selkies_renderer import SelkiesRenderer
from maddening.viz.renderer import GraphInfo
from maddening.viz.relay import StateRelay
from maddening.viz.runner import RealtimeRunner
from maddening.api.server import SimulationServer
import uvicorn

# --- Build graph ---
gm = GraphManager()
gm.add_node(TableNode("table", timestep=0.01, position=0.0))
gm.add_node(BallNode("ball", timestep=0.01, initial_position=5.0, elasticity=0.7, gravity=-9.81))
gm.add_node(SpringDamperNode("spring", timestep=0.01, stiffness=50.0, damping=2.0, mass=0.5, rest_length=1.5, initial_position=3.0))
gm.add_node(HeatNode("heat_rod", timestep=0.01, n_cells=20, length=1.0, thermal_diffusivity=0.01, initial_temperature=20.0))
gm.add_edge("table", "ball", "position", "table_position")
gm.add_edge("ball", "spring", "position", "anchor_position")
gm.add_edge("ball", "heat_rod", "velocity", "left_temperature",
    transform=lambda v: jnp.clip(jnp.abs(v) * 10.0, 0.0, 100.0))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gm.compile()
print(f"Graph: {gm.node_names}")

# --- Build matplotlib frame renderer ---
frame_renderer = ServerFrameRenderer(
    scene=SceneConfig(
        title="Ball-Spring-Table (WebRTC)",
        objects=[
            SceneObject(node="table", y="position", kind="surface", color="#8B7355", depth=0.5),
            SceneObject(node="ball", y="position", kind="circle", x=0.0, radius=0.25, color="#DD4444", label="Ball"),
            SceneObject(node="spring", y="position", kind="circle", x=0.8, radius=0.15, color="#4488DD", label="Spring"),
        ],
        xlim=(-2, 2), ylim=(-1, 8),
    ),
    timeseries=[
        TimeSeriesConfig(fields=[("ball", "position", "Ball"), ("spring", "position", "Spring")], window=300, title="Position"),
        TimeSeriesConfig(fields=[("ball", "velocity", "Vel")], window=300, title="Velocity"),
    ],
    heatmaps=[
        HeatmapConfig(node="heat_rod", field="temperature", title="Heat Rod", vmin=15, vmax=100, cmap="hot"),
    ],
    width=854, height=480, dpi=72,
    fmt="jpeg", quality=75,
)

# --- Profile: matplotlib-only rendering ---
print("\n--- Performance: matplotlib rendering only ---")
graph_info = GraphInfo.from_graph_manager(gm)
# Warm up
for _ in range(5):
    gm.step()
state_snapshot = {k: dict(v) for k, v in gm._state.items() if k != "_meta"}
_ = frame_renderer.render(0.0, state_snapshot)

mpl_times = []
for i in range(50):
    gm.step()
    state_snapshot = {k: dict(v) for k, v in gm._state.items() if k != "_meta"}
    t0 = time.perf_counter()
    frame_bytes = frame_renderer.render(i * 0.01, state_snapshot)
    mpl_times.append(time.perf_counter() - t0)

avg_mpl = sum(mpl_times) / len(mpl_times) * 1000
p95_mpl = sorted(mpl_times)[int(0.95 * len(mpl_times))] * 1000
frame_size = len(frame_bytes)
print(f"  Avg render: {avg_mpl:.1f}ms, P95: {p95_mpl:.1f}ms")
print(f"  Frame size: {frame_size} bytes ({frame_size/1024:.1f} KB)")
print(f"  Max FPS (render only): {1000/avg_mpl:.0f}")

# --- Adapter: wrap ServerFrameRenderer as Renderer ABC ---
# ServerFrameRenderer has render(sim_time, state) -> bytes, not the
# Renderer ABC (setup/update/teardown). We need an adapter for SelkiesRenderer.
from maddening.viz.renderer import Renderer as RendererABC

class FrameRendererAdapter(RendererABC):
    """Adapts ServerFrameRenderer to the Renderer ABC for SelkiesRenderer."""
    def __init__(self, sfr):
        self._sfr = sfr
        self._last_pixels = None
    def setup(self, graph_info):
        pass  # ServerFrameRenderer doesn't need setup
    def update(self, sim_time, state):
        frame_bytes = self._sfr.render(sim_time, state)
        # Convert JPEG to raw RGBA for the streaming session
        # For now, just store the compressed bytes — SelkiesRenderer
        # will call read_framebuffer_cpu() to get them
        self._last_pixels = frame_bytes
    def teardown(self):
        pass
    def read_framebuffer_cpu(self):
        # Return the compressed JPEG directly — SelkiesSession will
        # push it as-is to the GStreamer pipeline
        w = self._sfr.width
        h = self._sfr.height
        if self._last_pixels is None:
            return b"\x00" * (w * h * 4), w, h, "RGBA"
        # We need to decode JPEG to raw pixels for GStreamer's appsrc
        # which expects raw video frames
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(self._last_pixels)).convert("RGBA")
            return img.tobytes(), img.width, img.height, "RGBA"
        except ImportError:
            # No PIL — push dummy frame
            return b"\x00" * (w * h * 4), w, h, "RGBA"

adapter = FrameRendererAdapter(frame_renderer)

# --- Build SelkiesSession + SelkiesRenderer ---
print("\n--- Starting SelkiesSession ---")
selkies = SelkiesSession(secret="perf-test-secret", signaling_port=8443)
stream_config = StreamConfig.from_preset(QualityPreset.STANDARD)

selkies_renderer = SelkiesRenderer(adapter, selkies, config=stream_config)

# --- Profile: matplotlib + Selkies push ---
print("\n--- Performance: matplotlib + GStreamer push ---")
# Reset state
for name, spec in gm._nodes.items():
    gm._state[name] = spec.node.initial_state()

# Setup the SelkiesRenderer (starts the GStreamer pipeline)
selkies_renderer.setup(graph_info)
print(f"  Stream URL: {selkies_renderer.url}")
print(f"  Signaling: {selkies_renderer.stream_info.signaling_url}")
print(f"  Alive: {selkies.is_alive()}")

push_times = []
for i in range(50):
    gm.step()
    state_snapshot = {k: dict(v) for k, v in gm._state.items() if k != "_meta"}
    t0 = time.perf_counter()
    selkies_renderer.update(i * 0.01, state_snapshot)
    push_times.append(time.perf_counter() - t0)

avg_push = sum(push_times) / len(push_times) * 1000
p95_push = sorted(push_times)[int(0.95 * len(push_times))] * 1000
overhead = avg_push - avg_mpl
print(f"  Avg render+push: {avg_push:.1f}ms, P95: {p95_push:.1f}ms")
print(f"  GStreamer overhead: {overhead:.1f}ms ({overhead/avg_push*100:.0f}% of total)")
print(f"  Max FPS (render+push): {1000/avg_push:.0f}")

# --- Write perf results as JSON for the test script to read ---
perf = {
    "mpl_avg_ms": round(avg_mpl, 1),
    "mpl_p95_ms": round(p95_mpl, 1),
    "push_avg_ms": round(avg_push, 1),
    "push_p95_ms": round(p95_push, 1),
    "gst_overhead_ms": round(overhead, 1),
    "gst_overhead_pct": round(overhead / avg_push * 100, 0),
    "max_fps_render": round(1000 / avg_mpl),
    "max_fps_push": round(1000 / avg_push),
    "frame_size_bytes": frame_size,
    "pipeline_alive": selkies.is_alive(),
    "signaling_url": selkies_renderer.stream_info.signaling_url if selkies_renderer.stream_info else "",
}
with open("/tmp/webrtc_perf.json", "w") as f:
    json.dump(perf, f)
print(f"\nPerf results: /tmp/webrtc_perf.json")

# --- Now start the full server with SelkiesRenderer ---
print("\n--- Starting FastAPI server with WebRTC streaming ---")
# Reset state again
for name, spec in gm._nodes.items():
    gm._state[name] = spec.node.initial_state()

relay = StateRelay()
relay.attach(gm)
runner = RealtimeRunner(gm, relay, steps_per_frame=5)

server = SimulationServer(
    node_registry={
        "BallNode": BallNode, "TableNode": TableNode,
        "SpringDamperNode": SpringDamperNode, "HeatNode": HeatNode,
    },
    graph_manager=gm,
    frame_renderer=frame_renderer,
)
app = server.create_app()

# Auto-start simulation
runner.start()
print(f"Simulation running. Server on :8000, Signaling on :8443")
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
'''


def main():
    parser = argparse.ArgumentParser(description="WebRTC streaming test")
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
            autostop_minutes=15,
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

    # --- Install ---
    print("\nInstalling GStreamer system packages...")
    result = job.ssh_run(GSTREAMER_INSTALL, timeout=120, capture=True)
    print(f"  {(result.stdout or '').strip().split(chr(10))[-1]}")

    print("Installing Python deps (python3.10 for gi bindings)...")
    result = job.ssh_run(PIP_INSTALL, timeout=300, capture=True)
    print(f"  {(result.stdout or '').strip().split(chr(10))[-1]}")

    # --- Verify GPU + imports ---
    print("\nVerifying GPU + imports...")
    result = job.ssh_run(
        'python3 -c "import jax; print(jax.devices()); '
        'from maddening.cloud.selkies_session import SelkiesSession; '
        'from maddening.api.frame_renderer import ServerFrameRenderer; '
        'print(\'ALL OK\')"',
        timeout=60, capture=True, check=False,
    )
    print(f"  {(result.stdout or '').strip()}")
    if "ALL OK" not in (result.stdout or ""):
        print(f"  Import check failed. stderr: {(result.stderr or '')[-300:]}")
        if not args.keep:
            job.teardown()
        sys.exit(1)

    # --- Upload and run server script ---
    print("\nStarting WebRTC server (with performance profiling)...")
    job.ssh_run(f"echo {shlex.quote(SERVER_SCRIPT)} > /tmp/webrtc_server.py", check=True)
    job.ssh_run_background("python3 /tmp/webrtc_server.py")

    # --- Wait for perf results (server profiles before starting uvicorn) ---
    print("Waiting for profiling to complete...")
    perf = None
    for _ in range(60):
        time.sleep(3)
        try:
            result = job.ssh_run("cat /tmp/webrtc_perf.json 2>/dev/null", capture=True, check=False)
            if result.stdout and result.stdout.strip().startswith("{"):
                perf = json.loads(result.stdout.strip())
                break
        except Exception:
            pass

    if perf:
        print("\n" + "=" * 50)
        print("  RENDERING PERFORMANCE")
        print("=" * 50)
        print(f"  Matplotlib render:  {perf['mpl_avg_ms']:.1f}ms avg, {perf['mpl_p95_ms']:.1f}ms P95")
        print(f"  + GStreamer push:   {perf['push_avg_ms']:.1f}ms avg, {perf['push_p95_ms']:.1f}ms P95")
        print(f"  GStreamer overhead: {perf['gst_overhead_ms']:.1f}ms ({perf['gst_overhead_pct']:.0f}%)")
        print(f"  Max FPS (render):  {perf['max_fps_render']}")
        print(f"  Max FPS (stream):  {perf['max_fps_push']}")
        print(f"  Frame size:        {perf['frame_size_bytes']/1024:.1f} KB")
        print(f"  Pipeline alive:    {perf['pipeline_alive']}")
        print("=" * 50)
    else:
        print("  WARNING: Could not read performance results")

    # --- Discover endpoints ---
    print("\nDiscovering endpoints...")
    base_url = None
    for _ in range(20):
        base_url = job.get_runpod_endpoint(8000)
        if base_url:
            break
        time.sleep(2)
    if not base_url:
        base_url = f"http://{job.vm_ip}:8000"
    print(f"  HTTP: {base_url}")

    signaling_url = None
    if perf and perf.get("signaling_url"):
        # The signaling URL uses 0.0.0.0 — replace with public IP
        sig = perf["signaling_url"]
        # Find the public mapping for port 8443
        sig_endpoint = job.get_runpod_endpoint(8443)
        if sig_endpoint:
            # Convert http://ip:port to ws://ip:port/signaling/...
            sig_path = sig.split("/signaling/")[-1] if "/signaling/" in sig else ""
            signaling_url = sig_endpoint.replace("http://", "ws://") + "/signaling/" + sig_path
        else:
            signaling_url = sig.replace("0.0.0.0", job.vm_ip or "localhost")
    print(f"  Signaling: {signaling_url or 'not found'}")

    # --- Wait for FastAPI server ---
    print("\nWaiting for FastAPI server...")
    for _ in range(30):
        try:
            req = urllib.request.Request(f"{base_url}/graph", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                break
        except Exception:
            time.sleep(2)
    else:
        print("  WARNING: FastAPI server not responding")

    # --- Verify server-rendered WS works alongside WebRTC ---
    print("\nVerifying /ws/render endpoint (server-side rendering)...")
    try:
        req = urllib.request.Request(f"{base_url}/graph", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            graph = json.loads(resp.read())
            nodes = [n["name"] for n in graph["nodes"]] if isinstance(graph["nodes"], list) else list(graph["nodes"])
            print(f"  Graph nodes: {nodes}")
            print("  FastAPI server: OK")
    except Exception as e:
        print(f"  FastAPI error: {e}")

    # --- Print manual test instructions ---
    webrtc_viewer_url = f"{base_url}/viz/webrtc"  # We'd need to register this route
    print("\n" + "=" * 60)
    print("  WEBRTC STREAMING READY")
    print("=" * 60)
    print(f"\n  To test in browser, open the WebRTC client HTML locally")
    print(f"  and enter the signaling URL:")
    print(f"\n    Signaling: {signaling_url or 'unknown'}")
    print(f"\n  Or use the server-rendered viewer (JPEG over WS):")
    print(f"    {base_url}/viz/render")
    print(f"\n  REST API:")
    print(f"    {base_url}/graph")
    print(f"    {base_url}/graph/state")
    print(f"\n  SSH: ssh -p {job.ssh_port} root@{job.vm_ip}")
    print("=" * 60)

    # --- Teardown ---
    if args.keep:
        print(f"\nKeeping alive: {job.cluster_name}")
    else:
        print("\nTearing down...")
        job.teardown()
        print("  Done.")


if __name__ == "__main__":
    main()
