#!/usr/bin/env python3
"""Test SelkiesSession GStreamer/WebRTC on a cloud GPU.

Provisions a VM, installs GStreamer system packages + PyGObject,
verifies that SelkiesSession can be constructed and a GStreamer
pipeline can be built. Does NOT test full WebRTC streaming (that
requires a browser client), but validates the GStreamer layer.

Usage:
    python 06_selkies_test.py
    python 06_selkies_test.py --gpu RTX4090
    python 06_selkies_test.py --keep
"""

import argparse
import os
import shlex
import sys
import time

from maddening.cloud.launcher import (
    CloudLauncher,
    CostPolicy,
    JobConfig,
    LaunchError,
)


# System packages needed for GStreamer + PyGObject
GSTREAMER_INSTALL = (
    "export DEBIAN_FRONTEND=noninteractive"
    " && apt-get update -qq"
    " && apt-get install -y -qq"
    " gstreamer1.0-plugins-base"
    " gstreamer1.0-plugins-good"
    " gstreamer1.0-plugins-bad"
    " gstreamer1.0-nice"
    " gstreamer1.0-plugins-ugly"
    " gstreamer1.0-tools"
    " gir1.2-gst-plugins-bad-1.0"
    " gir1.2-gstreamer-1.0"
    " python3-gi"
    " python3-gi-cairo"
    " libgirepository1.0-dev"
    " > /dev/null 2>&1"
    " && echo GST_INSTALL_DONE"
)

# Python deps (PyGObject for python3.12, plus websockets for signaling)
PYGOBJECT_INSTALL = (
    "pip install -q --root-user-action=ignore"
    " PyGObject>=3.42 websockets>=11.0"
    " && echo PYGOBJECT_INSTALL_DONE"
)

# The test script to run on the VM
SELKIES_TEST_SCRIPT = r"""
import sys
print(f"Python: {sys.version}")

# Test 1: GStreamer import
print("\n--- Test 1: GStreamer import ---")
try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)
    print(f"GStreamer version: {Gst.version_string()}")
    print("PASS: GStreamer imported")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# Test 2: SelkiesSession construction
print("\n--- Test 2: SelkiesSession construction ---")
try:
    from maddening.cloud.selkies_session import SelkiesSession
    session = SelkiesSession(secret="test-secret", signaling_port=18443)
    print("PASS: SelkiesSession constructed")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# Test 3: StreamConfig + start/stop
print("\n--- Test 3: Session start/stop ---")
try:
    from maddening.cloud.streaming import StreamConfig, QualityPreset
    config = StreamConfig.from_preset(QualityPreset.PREVIEW)
    print(f"Config: {config.width}x{config.height} @ {config.fps}fps")
    info = session.start(config)
    print(f"Session ID: {info.session_id}")
    print(f"Signaling URL: {info.signaling_url}")
    print(f"Alive: {session.is_alive()}")
    assert session.is_alive(), "Session should be alive after start"
    session.stop()
    assert not session.is_alive(), "Session should not be alive after stop"
    print("PASS: start/stop lifecycle works")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Test 4: CPU framebuffer push
print("\n--- Test 4: CPU framebuffer push ---")
try:
    info = session.start(StreamConfig.from_preset(QualityPreset.PREVIEW))
    # Push a dummy frame (black 854x480 RGBA)
    pixels = b"\x00" * (854 * 480 * 4)
    session.update_framebuffer_cpu(pixels, 854, 480, "RGBA")
    print("PASS: CPU framebuffer pushed without error")
    session.stop()
except Exception as e:
    print(f"FAIL: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Test 5: SelkiesRenderer wrapping a mock inner renderer
print("\n--- Test 5: SelkiesRenderer integration ---")
try:
    from maddening.viz.backends.selkies_renderer import SelkiesRenderer
    from maddening.viz.renderer import GraphInfo, Renderer

    class DummyRenderer(Renderer):
        def setup(self, graph_info):
            pass
        def update(self, sim_time, state):
            pass
        def teardown(self):
            pass
        def read_framebuffer_cpu(self):
            return b"\xFF" * (64 * 32 * 4), 64, 32, "RGBA"

    inner = DummyRenderer()
    session2 = SelkiesSession(secret="test2", signaling_port=18444)
    renderer = SelkiesRenderer(inner, session2, config=StreamConfig.from_preset(QualityPreset.PREVIEW))

    graph_info = GraphInfo(
        node_names=["test"],
        node_params={"test": {}},
        node_state_fields={"test": ["x"]},
        edges=[],
        timestep=0.01,
    )
    renderer.setup(graph_info)
    print(f"Stream URL: {renderer.url}")
    assert renderer.stream_info is not None, "stream_info should be set"

    renderer.update(0.0, {"test": {"x": 1.0}})
    print("PASS: SelkiesRenderer setup + update works")

    renderer.teardown()
    print("PASS: SelkiesRenderer teardown works")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n=== ALL SELKIES TESTS PASSED ===")
"""


def main():
    parser = argparse.ArgumentParser(description="SelkiesSession test")
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

    # --- Install base Python deps ---
    # Use python3.10 (system default) for SelkiesSession because it needs
    # the system python3-gi package which only has C extensions for 3.10.
    # python3.12 has pip but not gi bindings.
    print("\nInstalling Python deps via SSH (targeting python3.10)...")
    result = job.ssh_run(
        "python3 -m pip install -q --root-user-action=ignore"
        ' "jax[cuda12]>=0.4,<0.6" "fastapi>=0.100" "uvicorn>=0.20"'
        ' "websockets>=11.0" "numpy>=1.24" "pyyaml>=6.0"'
        " && [ -d ~/sky_workdir/src ] && python3 -m pip install -q --root-user-action=ignore -e ~/sky_workdir"
        " ; echo BASE_INSTALL_DONE",
        timeout=300, capture=True,
    )
    print(f"  {(result.stdout or '').strip().split(chr(10))[-1]}")

    # --- Install GStreamer system packages ---
    print("\nInstalling GStreamer system packages...")
    result = job.ssh_run(GSTREAMER_INSTALL, timeout=120, capture=True)
    last_line = (result.stdout or "").strip().split("\n")[-1]
    print(f"  {last_line}")
    if "GST_INSTALL_DONE" not in (result.stdout or ""):
        print(f"  WARNING: GStreamer install may have failed")
        print(f"  stderr: {(result.stderr or '')[-300:]}")

    # --- Install PyGObject for python3.10 (system python) ---
    print("\nInstalling PyGObject...")
    # PyGObject via pip for 3.10 — the system python3-gi may already work
    result = job.ssh_run(
        "python3 -m pip install -q --root-user-action=ignore PyGObject>=3.42 websockets>=11.0"
        " ; echo PYGOBJECT_INSTALL_DONE",
        timeout=120, capture=True,
    )
    last_line = (result.stdout or "").strip().split("\n")[-1]
    print(f"  {last_line}")

    # --- Run Selkies test script ---
    print("\nRunning SelkiesSession tests on VM...")
    job.ssh_run(
        f"echo {shlex.quote(SELKIES_TEST_SCRIPT)} > /tmp/selkies_test.py",
        check=True,
    )
    try:
        result = job.ssh_run(
            "python3 /tmp/selkies_test.py",
            timeout=60, capture=True, check=False,
        )
        print(result.stdout or "")
        if result.returncode != 0:
            print(f"stderr: {(result.stderr or '')[-500:]}")
            print(f"\nSelkies tests FAILED (exit code {result.returncode})")
        else:
            print("Selkies tests completed successfully!")
    except Exception as e:
        print(f"ERROR running tests: {e}")

    # --- Teardown ---
    if args.keep:
        print(f"\nKeeping alive: {job.cluster_name}")
        print(f"  SSH: ssh -p {job.ssh_port} root@{job.vm_ip}")
    else:
        print("\nTearing down...")
        job.teardown()
        print("  Done.")


if __name__ == "__main__":
    main()
