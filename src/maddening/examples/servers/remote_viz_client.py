#!/usr/bin/env python
"""
Connect to a running simulation over ZeroMQ and visualize it.

The simulation side is ``remote_sim_server``.  It can run on this
machine or on a remote / HPC node.

Usage
-----
    # One command, everything local: start the server as a subprocess on
    # a free loopback port, connect, render, and stop the server on exit
    # (Ctrl-C and errors included):
    python -m maddening.examples.servers.remote_viz_client --local

    # The same, headless: print 20 frames as plain text, then exit
    # (non-zero if they do not all arrive within --timeout seconds):
    python -m maddening.examples.servers.remote_viz_client --local --mode print --frames 20

    # Connect to a server you started yourself (default tcp://localhost:5555):
    python -m maddening.examples.servers.remote_viz_client

    # With a matplotlib scene instead of the terminal table:
    python -m maddening.examples.servers.remote_viz_client --mode scene

    # Server on a remote / HPC node, through an SSH tunnel (run this
    # first on your local machine; the client then connects to localhost):
    #   ssh -L 5555:localhost:5555 user@hpc-node

    # Or connect to a non-loopback address directly.  That turns on ZMQ
    # CURVE encryption, so both sides need the same token
    # (MADDENING_TRANSPORT_TOKEN, falling back to MADDENING_API_TOKEN):
    MADDENING_TRANSPORT_TOKEN=<secret> \\
      python -m maddening.examples.servers.remote_viz_client --connect tcp://hpc-node:5555

Modes: ``terminal`` (a live table, needs ``rich``; works over SSH),
``print`` (one plain line per frame, for logs and CI) and ``scene``
(matplotlib windows).  ``--frames N`` stops after N frames in the first
two; without it they run until Ctrl-C.
"""

import argparse
import contextlib
import os
import signal
import socket
import subprocess
import sys
import threading
import time

from maddening.viz.network import NetworkReceiver
from maddening.viz.renderer import GraphInfo

FIELDS = {"ball": ["position", "velocity"], "table": ["position"]}

# GraphInfo stub -- the remote side does not send its graph description,
# so the renderers are set up from what this example knows it serves.
GRAPH_INFO = GraphInfo(
    node_names=["table", "ball"],
    node_params={"table": {"position": 0.0},
                 "ball": {"initial_position": 5.0, "elasticity": 0.7}},
    node_state_fields={"table": ["position"], "ball": ["position", "velocity"]},
    edges=[],
    timestep=0.01,
)


# ---------------------------------------------------------------------------
# Local mode: run the server as a subprocess on a free loopback port
# ---------------------------------------------------------------------------

SERVER_MODULE = "maddening.examples.servers.remote_sim_server"
READY_LINE = "Simulation publishing on"


def _free_loopback_port() -> int:
    """A TCP port nothing is listening on right now (the OS picks it)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stop_process(proc: subprocess.Popen, grace: float = 5.0) -> int:
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


@contextlib.contextmanager
def local_server(time_scale: float = 1.0, lifetime: float = 600.0,
                 startup_timeout: float = 120.0, attempts: int = 3):
    """Start ``remote_sim_server`` on a free loopback port; stop it on exit.

    Yields ``(address, process)``.  The port is chosen by the OS, so
    several local runs (or tests) can share a machine.  Between choosing
    the port and the server binding it another process could take it;
    the server then exits at once and a fresh port is tried.  The server
    is also told to stop by itself after *lifetime* seconds, so it cannot
    outlive this process by much even if this one is killed outright.
    """
    for attempt in range(1, attempts + 1):
        address = f"tcp://127.0.0.1:{_free_loopback_port()}"
        cmd = [sys.executable, "-m", SERVER_MODULE, "--bind", address,
               "--time-scale", str(time_scale), "--duration", str(lifetime)]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            # Its own session: a Ctrl-C in this terminal reaches only the
            # client, which then stops the server in the ``finally`` below.
            start_new_session=(os.name == "posix"),
        )
        ready = threading.Event()
        output: list[str] = []

        def _drain(stream=proc.stdout, ready=ready, output=output):
            # Read the server's output for its whole life, so a full pipe
            # can never block it, and keep it for error messages.
            for line in stream:
                output.append(line)
                if READY_LINE in line:
                    ready.set()

        threading.Thread(target=_drain, daemon=True).start()
        try:
            deadline = time.monotonic() + startup_timeout
            while not ready.wait(timeout=0.1):
                if proc.poll() is not None or time.monotonic() > deadline:
                    break
            if ready.is_set():
                print(f"Started local server (pid {proc.pid}) on {address}",
                      flush=True)
                yield address, proc
                return
        finally:
            code = _stop_process(proc)
            if ready.is_set():
                print(f"Local server (pid {proc.pid}) stopped, exit code {code}",
                      flush=True)
        if attempt == attempts:
            tail = "".join(output[-20:])
            raise RuntimeError(
                f"remote_sim_server did not start (exit code {proc.returncode}); "
                f"last output:\n{tail}"
            )


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _new_frames(receiver, max_frames, timeout):
    """Yield ``(sim_time, state)`` for each new frame the receiver gets."""
    last_time = None
    count = 0
    deadline = None if timeout is None else time.monotonic() + timeout
    while max_frames is None or count < max_frames:
        if deadline is not None and time.monotonic() > deadline:
            return
        sim_time, state = receiver.latest_snapshot()
        if state is not None and sim_time != last_time:
            last_time = sim_time
            count += 1
            yield sim_time, state
        else:
            time.sleep(0.005)


def run_print(receiver, max_frames=None, timeout=None) -> int:
    """Plain text, one line per frame.  Returns the number of frames."""
    print("Waiting for data from simulation server...", flush=True)
    n = 0
    for sim_time, state in _new_frames(receiver, max_frames, timeout):
        n += 1
        ball = state.get("ball", {})
        print(f"frame {n:4d}  t={sim_time:8.3f}s  "
              f"ball.position={ball.get('position', float('nan')):8.4f}  "
              f"ball.velocity={ball.get('velocity', float('nan')):8.4f}",
              flush=True)
    return n


def run_terminal(receiver, max_frames=None, timeout=None) -> int:
    """Live terminal table (needs ``rich``).  Returns the number of frames."""
    from maddening.viz.backends.terminal_renderer import TerminalRenderer

    renderer = TerminalRenderer(receiver, config={
        "title": "MADDENING — Remote Monitor",
        "fields": FIELDS,
        "precision": 4,
    })
    renderer.setup(GRAPH_INFO)

    print("Waiting for data from simulation server...", flush=True)
    if max_frames is None and timeout is None:
        renderer.run_event_loop(interval_ms=50)   # until Ctrl-C
        return 0
    renderer.start_background(interval_ms=50)
    try:
        return sum(1 for _ in _new_frames(receiver, max_frames, timeout))
    finally:
        renderer.teardown()
        print(flush=True)   # end the live table's last line


def run_scene(receiver):
    """Matplotlib scene + time-series visualization."""
    from maddening.viz.backends.matplotlib_renderer import (
        MatplotlibSceneRenderer,
        MatplotlibTimeSeriesRenderer,
        run_matplotlib,
    )

    # Scene renderer
    scene = MatplotlibSceneRenderer(receiver, scene_config={
        "title": "Bouncing Ball — Remote",
        "xlim": (-1, 1),
        "ylim": (-0.8, 6.0),
        "objects": [
            {
                "type": "surface",
                "node": "table",
                "y": "position",
                "depth": 0.5,
                "color": "#8B7355",
            },
            {
                "type": "circle",
                "node": "ball",
                "y": "position",
                "x": 0.0,
                "radius": 0.2,
                "color": "#DD4444",
                "edgecolor": "#991111",
            },
        ],
    })

    # Time-series renderer
    timeseries = MatplotlibTimeSeriesRenderer(receiver, plot_config={
        "title": "Bouncing Ball — Remote Time Series",
        "fields": {"ball": ["position", "velocity"]},
    })

    scene.setup(GRAPH_INFO)
    timeseries.setup(GRAPH_INFO)

    print("Waiting for data from simulation server...")
    print("Close plot windows to stop.")
    run_matplotlib(scene, timeseries, interval_ms=33)


def visualize(address, mode, frames, timeout) -> int:
    """Connect to *address* and render; returns a process exit code."""
    receiver = NetworkReceiver(address)
    receiver.start()
    try:
        if mode == "scene":
            run_scene(receiver)
            return 0
        runner = run_print if mode == "print" else run_terminal
        n = runner(receiver, frames, timeout if frames else None)
        if frames is not None:
            print(f"Received {n} of {frames} frames from {address}", flush=True)
            if n < frames:
                if receiver.handshake_error:
                    print(receiver.handshake_error, flush=True)
                return 1
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        receiver.stop()
        print("\nDisconnected.", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MADDENING remote visualization client")
    parser.add_argument("--connect", default="tcp://localhost:5555",
                        help="ZMQ connect address (default: tcp://localhost:5555)")
    parser.add_argument("--local", action="store_true",
                        help="Start remote_sim_server as a subprocess on a free "
                             "loopback port and connect to it (ignores --connect)")
    parser.add_argument("--mode", choices=["terminal", "print", "scene"],
                        default="terminal",
                        help="Visualization mode (default: terminal)")
    parser.add_argument("--frames", type=int, default=None,
                        help="Stop after this many frames (terminal/print modes)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="With --frames: give up after this many seconds "
                             "(default: 60)")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="With --local: the server's speed multiplier")
    args = parser.parse_args(argv)

    if args.frames is not None and args.mode == "scene":
        parser.error("--frames applies to the terminal and print modes")

    if not args.local:
        return visualize(args.connect, args.mode, args.frames, args.timeout)

    # The server stops itself well after the client would have given up,
    # so even a client killed outright does not leave it running for long.
    lifetime = (args.timeout + 120.0) if args.frames else 24 * 3600.0
    with local_server(time_scale=args.time_scale, lifetime=lifetime) as (address, _):
        return visualize(address, args.mode, args.frames, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
