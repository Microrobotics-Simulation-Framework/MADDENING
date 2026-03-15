"""
USD Stage Viewer -- visualize simulation results from a USD stage.

Provides :func:`viewer_from_usd` which reads a USD stage (written by
:class:`~maddening.usd.writer.USDWriter`) and returns a configured
:class:`~maddening.viz.history_viewer.HistoryViewer3D` ready for
interactive playback.

Also provides :func:`render_usd_frame` for quick single-frame
screenshots without the full interactive viewer.

This is a bridge between the USD data format and the general-purpose
``HistoryViewer3D``.  All rendering is done by PyVista via the
history viewer — this module only handles data extraction from USD.

Usage::

    from maddening.viz.usd_viewer import viewer_from_usd

    viewer = viewer_from_usd("results.usda")
    viewer.show()

Or with curve tubes from vessel geometry::

    from maddening.viz.usd_viewer import viewer_from_usd_with_geometry

    viewer = viewer_from_usd_with_geometry(
        results_path="results.usda",
        geometry_path="vessel.usda",
        tube_configs=[
            {"prim": "/Vessel/parent", "node": "parent",
             "field": "temperature", "radius": 0.05},
            {"prim": "/Vessel/daughter_left", "node": "daughter_left",
             "field": "temperature", "radius": 0.035},
        ],
    )
    viewer.show()

Requires: pyvista, pxr (usd-core), scipy
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _read_history_from_usd(stage, node_names=None):
    """Extract a history dict from a USD stage.

    Reads all time-sampled attributes from simulation node prims
    and assembles them into the ``{node: {field: array(n_frames, ...)}}``
    format expected by ``HistoryViewer3D``.

    Parameters
    ----------
    stage : Usd.Stage
        USD stage with time-sampled simulation data.
    node_names : list of str or None
        Node names to extract.  If None, discovers all
        MaddeningNode prims.

    Returns
    -------
    history : dict
        History dict compatible with HistoryViewer3D.
    dt : float
        Time step between frames (inferred from time codes).
    """
    from pxr import Usd

    # Discover time codes
    all_times = set()
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            samples = attr.GetTimeSamples()
            if samples:
                all_times.update(samples)

    if not all_times:
        raise ValueError("No time-sampled attributes found in stage")

    times = sorted(all_times)
    n_frames = len(times)
    dt = times[1] - times[0] if n_frames > 1 else 1.0

    # Discover node prims
    if node_names is None:
        node_names = []
        for prim in stage.Traverse():
            if prim.GetTypeName() == "MaddeningNode":
                node_names.append(prim.GetName())

    # Read time-sampled data
    history = {}
    sim_root = None
    for prim in stage.Traverse():
        if prim.GetTypeName() == "MaddeningSimulationGraph":
            sim_root = prim.GetPath()
            break

    for node_name in node_names:
        # Try to find the prim
        candidates = [
            f"{sim_root}/nodes/{node_name}" if sim_root else None,
            f"/Simulation/nodes/{node_name}",
            f"/Simulation/{node_name}",
        ]
        prim = None
        for path in candidates:
            if path:
                p = stage.GetPrimAtPath(path)
                if p.IsValid():
                    prim = p
                    break

        if prim is None:
            continue

        node_history = {}
        for attr in prim.GetAttributes():
            name = attr.GetName()
            # Skip metadata attributes
            if name.startswith("maddening:"):
                continue
            samples = attr.GetTimeSamples()
            if not samples:
                continue

            # Read all time samples
            values = []
            for t in times:
                val = attr.Get(t)
                if val is None:
                    val = attr.Get()  # fallback to default
                if val is None:
                    break
                if hasattr(val, '__len__'):
                    values.append(np.array(list(val), dtype=np.float32))
                else:
                    values.append(np.float32(val))
            else:
                node_history[name] = np.stack(values)

        if node_history:
            history[node_name] = node_history

    return history, dt


def viewer_from_usd(
    filepath: str,
    node_names: list[str] | None = None,
    **viewer_kwargs,
):
    """Create a HistoryViewer3D from a USD results file.

    Parameters
    ----------
    filepath : str
        Path to a USD file with time-sampled simulation data.
    node_names : list of str or None
        Which nodes to load.  None = all.
    **viewer_kwargs
        Passed to HistoryViewer3D (window_size, background, etc.).

    Returns
    -------
    HistoryViewer3D
        Configured viewer ready for ``viewer.show()``.
    """
    from pxr import Usd
    from maddening.viz.history_viewer import HistoryViewer3D

    stage = Usd.Stage.Open(filepath)
    history, dt = _read_history_from_usd(stage, node_names)
    return HistoryViewer3D(history, dt=dt, **viewer_kwargs)


def viewer_from_usd_with_geometry(
    results_path: str,
    geometry_path: str,
    tube_configs: list[dict],
    node_names: list[str] | None = None,
    **viewer_kwargs,
):
    """Create a HistoryViewer3D with curve tubes from USD geometry.

    Combines simulation results (time-sampled scalars) with vessel
    geometry (centerline curves) to produce tube visualizations
    colored by field values.

    Parameters
    ----------
    results_path : str
        Path to USD results file.
    geometry_path : str
        Path to USD geometry file (e.g., vessel phantom).
    tube_configs : list of dict
        Each dict specifies a tube:

        - ``"prim"``: geometry prim path (e.g., "/Vessel/parent")
        - ``"node"``: simulation node name
        - ``"field"``: scalar field for coloring (e.g., "temperature")
        - ``"radius"`` (optional): tube radius
        - ``"cmap"`` (optional): colormap
        - ``"clim"`` (optional): color limits
        - ``"label"`` (optional): color bar label

    node_names : list of str or None
        Which nodes to load from results.
    **viewer_kwargs
        Passed to HistoryViewer3D.

    Returns
    -------
    HistoryViewer3D
    """
    from pxr import Usd
    from maddening.viz.history_viewer import HistoryViewer3D

    # Load results
    results_stage = Usd.Stage.Open(results_path)
    history, dt = _read_history_from_usd(results_stage, node_names)

    # Load geometry
    geo_stage = Usd.Stage.Open(geometry_path)

    viewer = HistoryViewer3D(history, dt=dt, **viewer_kwargs)

    # Add tubes from geometry
    for cfg in tube_configs:
        prim = geo_stage.GetPrimAtPath(cfg["prim"])
        if not prim.IsValid():
            print(f"Warning: prim {cfg['prim']} not found in geometry")
            continue

        pts_attr = prim.GetAttribute("points")
        if not pts_attr.IsValid():
            # Try child prims
            for child in prim.GetChildren():
                pts_attr = child.GetAttribute("points")
                if pts_attr.IsValid():
                    break

        if not pts_attr.IsValid() or pts_attr.Get() is None:
            print(f"Warning: no points on {cfg['prim']}")
            continue

        pts = pts_attr.Get()
        centerline = np.array([[p[0], p[1], p[2]] for p in pts],
                              dtype=np.float64)

        viewer.add_curve_tube(
            node=cfg["node"],
            field=cfg["field"],
            centerline=centerline,
            radius=cfg.get("radius", 0.02),
            cmap=cfg.get("cmap", "coolwarm"),
            clim=cfg.get("clim"),
            label=cfg.get("label", f"{cfg['node']}.{cfg['field']}"),
        )

    return viewer


def render_usd_frame(
    results_path: str,
    geometry_path: str,
    tube_configs: list[dict],
    time: float | None = None,
    output_path: str = "frame.png",
    window_size: tuple[int, int] = (1280, 720),
    camera_position: str = "xy",
    zoom: float = 1.5,
    node_names: list[str] | None = None,
):
    """Render a single frame from USD data to an image file.

    Parameters
    ----------
    results_path : str
        Path to USD results file.
    geometry_path : str
        Path to USD geometry file.
    tube_configs : list of dict
        Tube configurations (see :func:`viewer_from_usd_with_geometry`).
    time : float or None
        Time code to render.  None = last frame.
    output_path : str
        Output image path.
    window_size : tuple
        Image size.
    camera_position : str
        Camera preset ("xy", "xz", "yz", "iso").
    zoom : float
        Camera zoom factor.
    node_names : list of str or None
        Which nodes to load.
    """
    import pyvista as pv
    from pxr import Usd
    from scipy.spatial import cKDTree

    results_stage = Usd.Stage.Open(results_path)
    history, dt = _read_history_from_usd(results_stage, node_names)
    geo_stage = Usd.Stage.Open(geometry_path)

    # Find the frame index for the requested time
    n_frames = next(
        arr.shape[0]
        for node_data in history.values()
        for arr in node_data.values()
    )
    if time is None:
        frame_idx = n_frames - 1
        time = frame_idx * dt
    else:
        frame_idx = min(int(round(time / dt)), n_frames - 1)

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")

    # Determine global color limits across all tubes
    all_vals = []
    for cfg in tube_configs:
        node = cfg["node"]
        field = cfg["field"]
        if node in history and field in history[node]:
            all_vals.append(history[node][field][frame_idx])
    if all_vals:
        flat = np.concatenate([np.asarray(v).ravel() for v in all_vals])
        global_clim = (float(flat.min()), float(flat.max()))
    else:
        global_clim = (0, 1)

    first_tube = True
    for cfg in tube_configs:
        prim = geo_stage.GetPrimAtPath(cfg["prim"])
        if not prim.IsValid():
            continue
        pts_attr = prim.GetAttribute("points")
        if not pts_attr.IsValid() or pts_attr.Get() is None:
            continue
        pts = pts_attr.Get()
        centerline = np.array([[p[0], p[1], p[2]] for p in pts])
        n_center = len(centerline)

        # Build tube
        cells = np.zeros(n_center + 1, dtype=np.int64)
        cells[0] = n_center
        cells[1:] = np.arange(n_center)
        line = pv.PolyData(centerline, lines=cells)
        tube = line.tube(
            radius=cfg.get("radius", 0.02),
            n_sides=cfg.get("n_sides", 12),
        )

        # Map scalars
        node = cfg["node"]
        field = cfg["field"]
        if node in history and field in history[node]:
            scalars = np.asarray(history[node][field][frame_idx],
                                 dtype=np.float32)
            if len(scalars) != n_center:
                x_s = np.linspace(0, 1, len(scalars))
                x_c = np.linspace(0, 1, n_center)
                scalars = np.interp(x_c, x_s, scalars).astype(np.float32)
            tree = cKDTree(centerline)
            _, idx = tree.query(tube.points)
            tube["Temperature"] = scalars[idx]

            clim = cfg.get("clim", global_clim)
            plotter.add_mesh(
                tube, scalars="Temperature",
                cmap=cfg.get("cmap", "coolwarm"),
                clim=clim,
                opacity=cfg.get("opacity", 1.0),
                show_scalar_bar=first_tube,
                scalar_bar_args={
                    "title": cfg.get("label", "Temperature (C)"),
                    "color": "black",
                },
            )
            first_tube = False
        else:
            plotter.add_mesh(tube, color="gray", opacity=0.5)

    plotter.add_text(
        f"t = {time:.4f}s", position="upper_right",
        font_size=10, color="black",
    )

    plotter.camera_position = camera_position
    plotter.camera.zoom(zoom)
    plotter.screenshot(output_path)
    plotter.close()
    return output_path
