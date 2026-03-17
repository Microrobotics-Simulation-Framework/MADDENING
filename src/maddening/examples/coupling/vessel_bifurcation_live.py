"""
Real-time vessel bifurcation -- live 3D visualization with USD recording.

Demonstrates the full MADDENING real-time pipeline:

1. **Build** a Y-shaped vessel phantom from USD geometry
2. **Run** three coupled HeatNodes on a daemon thread (RealtimeRunner)
3. **Visualize** live in a PyVista window (PyVistaLiveRenderer)
4. **Record** simulation state to USD (USDWriter) in parallel
5. **Inject** a heat pulse via external inputs (demonstrating commands)

The physics runs at dt=0.0001 on a background thread.  The renderer
polls the StateRelay at ~30 fps and updates the 3D tube coloring.
Meanwhile, every 100th physics step is written to a USD file for
later replay.

After 5 seconds of sim time, a heat source pulse is injected into
the parent tube (demonstrating external input / command injection).
The pulse propagates through the bifurcation to both daughters.

Architecture::

    RealtimeRunner (daemon thread)
        | physics at dt=0.0001
        |
    StateRelay (thread-safe, stride=10)
        | polled by:
        +-> PyVistaLiveRenderer (main thread, timer callback ~30fps)
        +-> USDWriter (observer, every 100th step -> .usda)

Keyboard controls:
    Space       pause / resume
    Up/Down     speed up / slow down (time scale)
    Mouse       rotate, zoom, pan
    Close window to stop

Usage::

    JAX_PLATFORMS=cpu python -m maddening.examples.coupling.vessel_bifurcation_live
"""

import os
import sys
import tempfile
import threading
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

# USD schemas must be imported before any Usd.Stage operations
import maddening.usd
from maddening.usd.geometry import create_vessel_phantom, load_grid_from_usd
from maddening.usd.writer import USDWriter

from maddening.core.graph_manager import GraphManager
from maddening.core.transforms import register_transform
from maddening.nodes.heat import HeatNode

from maddening.viz.relay import StateRelay
from maddening.viz.runner import RealtimeRunner
from maddening.viz.renderer import GraphInfo


# --- Register transforms for USD serialization ---

@register_transform("vessel_extract_right", "Right boundary temperature")
def vessel_extract_right(T):
    return T[-1]


@register_transform("vessel_extract_left", "Left boundary temperature")
def vessel_extract_left(T):
    return T[0]


def main():
    print("=" * 60)
    print("Real-Time Vessel Bifurcation with Live 3D + USD Recording")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Create vessel phantom
    # ------------------------------------------------------------------
    print("\n1. Creating vessel phantom...")
    tmpdir = tempfile.mkdtemp()
    vessel_path = os.path.join(tmpdir, "vessel.usda")
    results_path = os.path.join(tmpdir, "live_results.usda")

    vessel_stage = create_vessel_phantom(
        vessel_path,
        parent_length=1.0,
        daughter_length=0.8,
        parent_n_points=20,
        daughter_n_points=16,
        bifurcation_angle=30.0,
    )
    print(f"   Vessel: {vessel_path}")

    # ------------------------------------------------------------------
    # 2. Read geometry and build simulation graph
    # ------------------------------------------------------------------
    print("\n2. Building simulation graph...")
    parent_x = load_grid_from_usd(vessel_stage, "/Vessel/parent", axis=0)
    left_x = load_grid_from_usd(vessel_stage, "/Vessel/daughter_left", axis=0)
    right_x = load_grid_from_usd(vessel_stage, "/Vessel/daughter_right", axis=0)

    dt = 0.0001
    n_parent = len(parent_x)
    n_left = len(left_x)
    n_right = len(right_x)

    gm = GraphManager()
    gm.add_node(HeatNode(
        "parent", dt, n_cells=n_parent,
        length=float(parent_x[-1] - parent_x[0]),
        thermal_diffusivity=0.01,
        initial_temperature=100.0,
        grid_points=list(parent_x - parent_x[0]),
    ))
    gm.add_node(HeatNode(
        "daughter_left", dt, n_cells=n_left,
        length=float(left_x[-1] - left_x[0]),
        thermal_diffusivity=0.01,
        initial_temperature=20.0,
        grid_points=list(left_x - left_x[0]),
    ))
    gm.add_node(HeatNode(
        "daughter_right", dt, n_cells=n_right,
        length=float(right_x[-1] - right_x[0]),
        thermal_diffusivity=0.01,
        initial_temperature=20.0,
        grid_points=list(right_x - right_x[0]),
    ))

    gm.add_edge("parent", "daughter_left",
                 "temperature", "left_temperature",
                 transform=vessel_extract_right)
    gm.add_edge("parent", "daughter_right",
                 "temperature", "left_temperature",
                 transform=vessel_extract_right)
    gm.add_edge("daughter_left", "parent",
                 "temperature", "right_temperature",
                 transform=vessel_extract_left)

    # External input: heat source on the parent tube (for pulse injection)
    gm.add_external_input("parent", "heat_source", shape=(n_parent,))

    gm.add_coupling_group(
        ["parent", "daughter_left", "daughter_right"],
        max_iterations=20, tolerance=1e-8,
    )
    gm.compile()
    print(f"   3 nodes, dt={dt}, external input: parent.heat_source")

    # ------------------------------------------------------------------
    # 3. Real-time infrastructure
    # ------------------------------------------------------------------
    print("\n3. Setting up real-time pipeline...")

    relay = StateRelay(stride=10)
    relay.attach(gm)

    # USD writer as observer
    from pxr import Usd
    results_stage = Usd.Stage.CreateNew(results_path)
    usd_writer = USDWriter(results_stage, gm)
    write_stride = 100
    step_counter = [0]

    def usd_observer(event, state):
        if event == "step":
            step_counter[0] += 1
            if step_counter[0] % write_stride == 0:
                usd_writer.write_frame(state, step_counter[0] * dt)

    gm.add_observer(usd_observer)

    # Heat pulse injection: a background thread that fires a pulse
    # after a few seconds of sim time, demonstrating external inputs.
    pulse_active = [False]
    ext_inputs_lock = threading.Lock()
    ext_inputs = [gm._default_external_inputs()]

    def pulse_injector():
        """Inject a heat pulse into the parent tube's midsection."""
        time.sleep(3.0)  # wait 3 wall-seconds before first pulse
        n = n_parent
        source = np.zeros(n, dtype=np.float32)
        # Gaussian pulse in the middle of the parent
        mid = n // 2
        for i in range(n):
            source[i] = 5000.0 * np.exp(-((i - mid) / 3.0) ** 2)

        print("\n  >> HEAT PULSE injected into parent midsection!")
        pulse_active[0] = True
        with ext_inputs_lock:
            ext_inputs[0] = {
                "parent": {"heat_source": jnp.array(source)},
            }

        time.sleep(5.0)  # pulse lasts 5 wall-seconds

        print("  >> Heat pulse OFF")
        pulse_active[0] = False
        with ext_inputs_lock:
            ext_inputs[0] = gm._default_external_inputs()

    pulse_thread = threading.Thread(target=pulse_injector, daemon=True)

    # Custom command receiver that reads from ext_inputs
    class PulseCommandReceiver:
        def latest_commands(self):
            with ext_inputs_lock:
                return ext_inputs[0]

    runner = RealtimeRunner(
        gm, relay,
        time_scale=10.0,
        steps_per_frame=200,
        command_receiver=PulseCommandReceiver(),
    )

    # ------------------------------------------------------------------
    # 4. Build live 3D renderer
    # ------------------------------------------------------------------
    print("\n4. Setting up live 3D renderer...")

    from maddening.viz.backends.pyvista_live import PyVistaLiveRenderer

    def _get_centerline(stage, path):
        prim = stage.GetPrimAtPath(path)
        pts = prim.GetAttribute("points").Get()
        return np.array([[p[0], p[1], p[2]] for p in pts])

    parent_pts = _get_centerline(vessel_stage, "/Vessel/parent")
    left_pts = _get_centerline(vessel_stage, "/Vessel/daughter_left")
    right_pts = _get_centerline(vessel_stage, "/Vessel/daughter_right")

    renderer = PyVistaLiveRenderer(
        window_size=(1400, 800),
        title="MADDENING -- Vessel Bifurcation (Live + USD)",
    )
    renderer.add_curve_tube(
        "parent", "temperature", parent_pts,
        radius=0.05, cmap="coolwarm", clim=(20.0, 120.0),
        label="Temperature (C)",
    )
    renderer.add_curve_tube(
        "daughter_left", "temperature", left_pts,
        radius=0.035, cmap="coolwarm", clim=(20.0, 120.0),
    )
    renderer.add_curve_tube(
        "daughter_right", "temperature", right_pts,
        radius=0.035, cmap="coolwarm", clim=(20.0, 120.0),
    )

    graph_info = GraphInfo.from_graph_manager(gm)
    renderer.setup(graph_info)

    # ------------------------------------------------------------------
    # 5. Go!
    # ------------------------------------------------------------------
    print("\n5. Starting real-time simulation...")
    print(f"   Physics: dt={dt}, {runner.steps_per_frame} steps/frame")
    print(f"   Time scale: {runner.time_scale}x")
    print(f"   USD: every {write_stride} steps -> {results_path}")
    print(f"   Heat pulse: injected after ~3s wall time")
    print()

    runner.start()
    pulse_thread.start()

    try:
        # This blocks until the user closes the PyVista window
        renderer.run_live(relay, runner=runner, target_fps=30)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        runner.stop()
        results_stage.Save()

        sim_time, state = relay.latest_snapshot()
        n_usd_frames = step_counter[0] // write_stride
        print(f"\n   Stopped at t={runner.sim_time:.4f}s")
        print(f"   {step_counter[0]} physics steps, "
              f"{n_usd_frames} USD frames saved")
        if state:
            T_p = np.asarray(state["parent"]["temperature"])
            T_l = np.asarray(state["daughter_left"]["temperature"])
            print(f"   Parent: T=[{T_p.min():.1f}, {T_p.max():.1f}] C")
            print(f"   Left:   T=[{T_l.min():.1f}, {T_l.max():.1f}] C")

    try:
        os.unlink(vessel_path)
        os.unlink(results_path)
        os.rmdir(tmpdir)
    except OSError:
        pass

    print("\nDone.")


if __name__ == "__main__":
    if "--gpu" in sys.argv:
        os.environ["JAX_PLATFORMS"] = ""
        print("GPU mode: JAX auto-detecting backend")
    main()
