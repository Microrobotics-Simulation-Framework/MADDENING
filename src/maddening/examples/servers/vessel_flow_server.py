"""
Vessel Flow Server -- FastAPI demo for coupled HeartPump + LBM simulation.

A lightweight standalone server that runs the HeartPump/LBM vessel
flow simulation with real-time parameter control and live vital
signs streamed via WebSocket.

Usage::

    JAX_PLATFORMS=cpu python -m maddening.examples.servers.vessel_flow_server

Then open http://localhost:8000 in your browser.

Design note -- FSI and wall_mask_update
---------------------------------------
The ``wall_mask_update`` boundary input allows runtime mask modification
(clot injection, moving rigid bodies).  Mask updates are full boolean
arrays of the same shape as the grid that REPLACE the mask entirely.
This is JIT-friendly because the replacement uses ``jnp.where``
(traceable) and the array shape is static.

For a full FSI coupling (rigid body moving through fluid), the mask
would need to be recomputed at Python level each step from the body's
position, which precludes ``lax.scan`` for the coupled loop.  The
simple clot injection used here does not have that limitation since the
clot mask is set once per REST call and remains constant between calls.
"""

import os
# Don't override JAX_PLATFORMS if already set (e.g. JAX_PLATFORMS=gpu)
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import asyncio
import logging
import time

import jax.numpy as jnp
import numpy as np

from maddening.examples.coupling.vessel_flow_helpers import build_vessel_flow_graph
from maddening.viz.relay import StateRelay
from maddening.viz.runner import RealtimeRunner

logger = logging.getLogger(__name__)


def _build_centerlines(grid_shape, vessel_params):
    """Compute 3D centerlines for the vessel renderer.

    Returns a list of (name, centerline_array, radius) tuples.
    """
    from maddening.examples.coupling.vessel_flow_helpers import _DEFAULT_VESSEL_PARAMS
    vp = vessel_params or _DEFAULT_VESSEL_PARAMS

    nx, ny, nz = grid_shape
    y_mid = (ny - 1) / 2.0
    z_mid = (nz - 1) / 2.0

    pr = vp["parent_radius"]
    dr = vp["daughter_radius"]
    pl = vp["parent_length"]
    dl = vp["daughter_length"]
    angle_rad = np.deg2rad(vp["bifurcation_angle"])

    # Parent: along x-axis from 0 to slightly past the bifurcation
    # (extend past bif to ensure watertight junction)
    overlap = pr * 0.5  # extend into bifurcation zone
    n_pts = max(20, int(pl + overlap))
    parent_pts = np.column_stack([
        np.linspace(0, pl + overlap, n_pts),
        np.full(n_pts, y_mid),
        np.full(n_pts, z_mid),
    ])

    # Daughter centerlines start INSIDE the parent (overlap for watertight join)
    left_dir = np.array([np.cos(angle_rad), -np.sin(angle_rad), 0.0])
    right_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
    bif = np.array([pl, y_mid, z_mid])

    # Start daughters slightly before the bifurcation point
    start_offset = -pr * 0.3  # start inside the parent
    n_d = max(16, int(dl))
    t = np.linspace(start_offset, dl, n_d)
    left_pts = bif[None, :] + t[:, None] * left_dir[None, :]
    right_pts = bif[None, :] + t[:, None] * right_dir[None, :]

    return [
        ("parent", parent_pts, pr * 0.8),
        ("daughter_left", left_pts, dr * 0.8),
        ("daughter_right", right_pts, dr * 0.8),
    ]


def create_app(grid_shape=(64, 32, 32), vessel_params=None):
    """Build the FastAPI application and simulation.

    Returns ``(app, gm, relay, runner)`` so callers can inspect
    internals or change the grid shape.

    Architecture note (HPC / remote deployment)
    --------------------------------------------
    The server binds to ``0.0.0.0`` so it accepts connections from
    any network interface.  For HPC deployments where physics runs
    on a remote GPU node:

    - ``/ws/render`` streams server-rendered JPEG frames (no GPU
      needed on the client).  Works over any network.
    - ``/ws/state`` streams raw scalar data for client-side rendering
      or data logging.
    - Both endpoints coexist, enabling hybrid rendering strategies.

    The same binary runs identically for localhost (development) and
    remote (HPC) — only the client's URL changes.
    """
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import HTMLResponse, Response
    from pathlib import Path

    t0 = time.perf_counter()
    gm, vessel_mask = build_vessel_flow_graph(
        grid_shape=grid_shape, vessel_params=vessel_params,
    )
    build_time = time.perf_counter() - t0

    relay = StateRelay(stride=10)  # capture every 10th step for WS
    relay.attach(gm)
    # LBM dt=1.0 in lattice units. At 16ms/step on CPU (64x32x32),
    # we can do ~60 steps/second. Batch 10 steps per frame for
    # smooth throughput. time_scale >> 1 to run as fast as possible
    # (no wall-clock pacing — physics runs flat out).
    runner = RealtimeRunner(gm, relay, steps_per_frame=10,
                            time_scale=1e6)

    # Build centerline geometry for client-side 3D rendering.
    # The server sends geometry once via /sim/geometry, then streams
    # scalar data per frame via /ws/state.  The browser renders with
    # Three.js — no PyVista/VTK needed on the server, works over any
    # network, and the client gets interactive 3D orbit/zoom/pan.
    centerlines = _build_centerlines(grid_shape, vessel_params)

    # Stash references for clot injection
    _base_mask = vessel_mask              # original geometry
    _clot_active = [False]                # mutable flag
    _clot_mask = [vessel_mask]            # current mask (may include clot)
    _clot_pos = [None]                    # (x, y, z, radius) for 3D rendering

    app = FastAPI(title="MADDENING Vessel Flow")

    # --- Static files ---
    _STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "api" / "static"

    @app.get("/", response_class=HTMLResponse)
    async def root():
        html_path = _STATIC_DIR / "vessel_flow.html"
        return HTMLResponse(content=html_path.read_text(), status_code=200)

    # --- Simulation control ---

    @app.post("/sim/start")
    async def sim_start():
        runner.start()
        return {"status": "started"}

    @app.post("/sim/stop")
    async def sim_stop():
        runner.stop()
        return {"status": "stopped"}

    @app.post("/sim/reset")
    async def sim_reset():
        runner.stop()
        for name, spec in gm._nodes.items():
            gm._state[name] = spec.node.initial_state()
        gm._dirty = True
        _clot_active[0] = False
        _clot_pos[0] = None
        _clot_mask[0] = _base_mask
        runner.reset_time()
        relay._step_count = 0
        relay._sim_time = 0.0
        relay._snapshot = None
        return {"status": "reset"}

    # --- Parameter tuning ---

    @app.put("/sim/heart_rate")
    async def set_heart_rate(bpm: float):
        """Adjust heart rate in real time."""
        gm._nodes["heart"].node.params["heart_rate"] = bpm
        gm._dirty = True
        return {"heart_rate": bpm}

    @app.put("/sim/stroke_volume")
    async def set_stroke_volume(sv: float):
        """Adjust stroke volume (LBM units)."""
        gm._nodes["heart"].node.params["stroke_volume"] = sv
        gm._dirty = True
        return {"stroke_volume": sv}

    @app.put("/sim/resistance")
    async def set_resistance(r: float):
        """Adjust vascular resistance."""
        gm._nodes["heart"].node.params["resistance"] = r
        gm._dirty = True
        return {"resistance": r}

    @app.put("/sim/compliance")
    async def set_compliance(c: float):
        """Adjust arterial compliance."""
        gm._nodes["heart"].node.params["compliance"] = c
        gm._dirty = True
        return {"compliance": c}

    # --- Clot injection ---

    @app.post("/sim/inject_clot")
    async def inject_clot(x: int, y: int, z: int, radius: int = 3):
        """Mark a spherical region as wall (simulating stenosis/thrombus).

        The wall_mask_update is injected as an external input to the LBM
        node on every subsequent step.  This is JIT-compatible because
        the mask is a full-shape boolean array that replaces the previous
        mask entirely via jnp.where.
        """
        nx, ny_g, nz_g = grid_shape
        xx, yy, zz = np.mgrid[:nx, :ny_g, :nz_g]
        clot_region = ((xx - x)**2 + (yy - y)**2 + (zz - z)**2) <= radius**2
        new_mask = np.array(_base_mask, copy=True) | clot_region
        _clot_mask[0] = jnp.asarray(new_mask)
        _clot_active[0] = True
        _clot_pos[0] = [x, y, z, radius]

        # Register external input if not already registered
        _ensure_mask_external()
        return {
            "status": "clot_injected",
            "center": [x, y, z],
            "radius": radius,
        }

    @app.post("/sim/clear_clot")
    async def clear_clot():
        """Restore original vessel geometry."""
        _clot_mask[0] = _base_mask
        _clot_active[0] = False
        _clot_pos[0] = None
        return {"status": "clot_cleared"}

    _mask_ext_registered = [False]

    def _ensure_mask_external():
        if not _mask_ext_registered[0]:
            gm.add_external_input("vessel", "wall_mask_update",
                                  shape=grid_shape, dtype=jnp.bool_)
            gm._dirty = True
            _mask_ext_registered[0] = True

    # Override runner step to inject clot mask
    _orig_step = gm.step

    def _step_with_clot(external_inputs=None):
        if _clot_active[0]:
            _ensure_mask_external()
            if external_inputs is None:
                external_inputs = {}
            if "vessel" not in external_inputs:
                external_inputs["vessel"] = {}
            external_inputs["vessel"]["wall_mask_update"] = _clot_mask[0]
        return _orig_step(external_inputs)

    gm.step = _step_with_clot

    # --- Geometry (for client-side 3D rendering) ---

    @app.get("/sim/geometry")
    async def get_geometry():
        """Return vessel centerlines + radii for Three.js rendering.

        Called once by the client on startup to build the 3D scene.
        The client then receives per-frame scalar data via /ws/state
        to color the tubes by velocity magnitude.
        """
        tubes = []
        for name, pts, radius in centerlines:
            tubes.append({
                "name": name,
                "points": pts.tolist(),
                "radius": float(radius),
            })
        return {"tubes": tubes, "grid_shape": list(grid_shape)}

    # --- Vitals ---

    @app.get("/sim/vitals")
    async def get_vitals():
        """Current pressure, flow rate, heart phase."""
        _, state = relay.latest_snapshot()
        if state and "heart" in state:
            return {
                "arterial_pressure": float(state["heart"]["arterial_pressure"]),
                "flow_rate": float(state["heart"]["flow_rate"]),
                "phase": float(state["heart"]["phase"]),
            }
        return {}

    # --- WebSocket ---

    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket):
        """Stream simulation vitals as JSON at ~30 fps."""
        await ws.accept()
        last_sim_time = -1.0
        try:
            while True:
                sim_time, state = relay.latest_snapshot()
                if state is not None and sim_time != last_sim_time:
                    last_sim_time = sim_time
                    data = {
                        "sim_time": sim_time,
                        "step_count": int(sim_time),
                    }
                    if "heart" in state:
                        hs = state["heart"]
                        data["arterial_pressure"] = float(hs["arterial_pressure"])
                        data["flow_rate"] = float(hs["flow_rate"])
                        data["phase"] = float(hs["phase"])
                    # Extract field profiles along vessel centerline
                    # for client-side 3D coloring
                    if "vessel" in state:
                        vel = np.asarray(state["vessel"]["velocity"])
                        prs = np.asarray(state["vessel"]["pressure"])
                        if vel.ndim == 4:  # (nx,ny,nz,3)
                            mag = np.sqrt(np.sum(vel ** 2, axis=-1))
                            ny_g, nz_g = mag.shape[1], mag.shape[2]
                            cy, cz = ny_g // 2, nz_g // 2
                            data["velocity_profile"] = mag[:, cy, cz].tolist()
                        if prs.ndim == 3:
                            ny_g, nz_g = prs.shape[1], prs.shape[2]
                            cy, cz = ny_g // 2, nz_g // 2
                            data["pressure_profile"] = prs[:, cy, cz].tolist()
                    # Send clot status + position for 3D rendering
                    data["clot_active"] = _clot_active[0]
                    if _clot_pos[0] is not None:
                        data["clot_pos"] = _clot_pos[0]
                    await ws.send_json(data)
                await asyncio.sleep(0.033)  # ~30 fps
        except Exception:
            pass

    # --- Print startup info ---

    def _print_banner():
        nx, ny_g, nz_g = grid_shape
        n_fluid = int(jnp.sum(~vessel_mask))
        hp = gm._nodes["heart"].node.params
        print()
        print("=" * 60)
        print("  MADDENING Vessel Flow Server")
        print("=" * 60)
        print(f"\n  Grid: {nx} x {ny_g} x {nz_g}  "
              f"({n_fluid} fluid cells, "
              f"{n_fluid / np.prod(grid_shape) * 100:.0f}% fill)")
        print(f"  Build time: {build_time:.2f}s")
        print(f"\n  Heart pump parameters:")
        print(f"    heart_rate     = {hp['heart_rate']} bpm")
        print(f"    stroke_volume  = {hp['stroke_volume']}")
        print(f"    resistance     = {hp['resistance']}")
        print(f"    compliance     = {hp['compliance']}")
        print(f"\n  Endpoints:")
        print(f"    GET  /                    Browser UI")
        print(f"    POST /sim/start           Start simulation")
        print(f"    POST /sim/stop            Stop simulation")
        print(f"    POST /sim/reset           Reset to initial state")
        print(f"    PUT  /sim/heart_rate?bpm= Set heart rate")
        print(f"    PUT  /sim/resistance?r=   Set resistance")
        print(f"    PUT  /sim/compliance?c=   Set compliance")
        print(f"    POST /sim/inject_clot     Inject clot (x,y,z,radius)")
        print(f"    POST /sim/clear_clot      Remove clot")
        print(f"    GET  /sim/vitals          Current vital signs")
        print(f"    GET  /sim/geometry         Vessel centerlines (for 3D)")
        print(f"    WS   /ws/state            Live vitals + velocity data")
        print(f"\n  Open http://localhost:8000 in your browser")
        print(f"  3D rendering: client-side (Three.js, no server GPU needed)")
        print(f"  (Works from any machine — server binds to 0.0.0.0)")
        print()

    _print_banner()
    return app, gm, relay, runner


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="MADDENING Vessel Flow Server")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (CUDA) backend instead of CPU")
    parser.add_argument("--grid", type=int, nargs=3, default=[64, 32, 32],
                        metavar=("NX", "NY", "NZ"),
                        help="LBM grid resolution (default: 64 32 32)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--profile", action="store_true",
                        help="Run profiler before starting server")
    args = parser.parse_args()

    if args.gpu:
        # Override to auto-detect (CUDA or ROCm)
        os.environ["JAX_PLATFORMS"] = ""
        print(f"  GPU mode: JAX will auto-detect available backend")
    else:
        print(f"  CPU mode (use --gpu for GPU)")

    app, gm, relay, runner = create_app(grid_shape=tuple(args.grid))

    if args.profile:
        from maddening.core.profiler import profile_graph
        print("\nProfiling...")
        report = profile_graph(gm, n_steps=50)
        print(report)
        print()

    # Auto-start simulation
    runner.start()

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
