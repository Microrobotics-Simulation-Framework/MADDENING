"""
Helper to build a coupled HeartPump + LBM vessel flow graph.

Provides ``build_vessel_flow_graph()`` which wires a
:class:`~maddening.nodes.heart_pump.HeartPumpNode` to an
:class:`~maddening.nodes.lbm.LBMNode` with bidirectional pressure
coupling and returns a compiled, ready-to-run graph.

Coupling strategy
-----------------
Both edges use **state fields** with transforms rather than flux
coupling.  This avoids the issue where the coupling iteration's
Gauss-Seidel pass would need flux data from a node that hasn't been
updated yet in the current iteration.

* **Heart -> LBM**: ``heart.arterial_pressure`` -> ``vessel.inlet_pressure``
  (direct scalar state field).
* **LBM -> Heart**: ``vessel.pressure`` -> ``heart.backpressure``
  (full pressure grid, with a transform that computes the mean fluid
  pressure at the outlet face).

Usage::

    from maddening.examples.coupling.vessel_flow_helpers import build_vessel_flow_graph
    gm, vessel_mask = build_vessel_flow_graph(grid_shape=(64, 32, 32))
    gm.run(100)
"""

from __future__ import annotations

import warnings
from typing import Optional

import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.nodes.heart_pump import HeartPumpNode
from maddening.nodes.lbm import LBMNode, _FACE_MAP
from maddening.nodes.lbm_geometry import voxelize_vessel


# Default vessel geometry parameters (in lattice units)
_DEFAULT_VESSEL_PARAMS = {
    "parent_radius": 6.0,
    "daughter_radius": 4.0,
    "parent_length": 30.0,
    "daughter_length": 25.0,
    "bifurcation_angle": 30.0,
}


def _make_outlet_pressure_transform(wall_mask, outlet_face, ndim):
    """Build a JAX-traceable transform that extracts mean outlet pressure.

    Parameters
    ----------
    wall_mask : jnp.ndarray
        Boolean wall mask (True = wall).
    outlet_face : str
        Face name, e.g. ``"x_max"``.
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    callable
        ``transform(pressure_grid) -> scalar`` that computes the mean
        fluid-cell pressure at the outlet face.
    """
    outlet_axis, outlet_side = _FACE_MAP[outlet_face]
    face_slices = [slice(None)] * ndim
    if outlet_side == "min":
        face_slices[outlet_axis] = 0
    else:
        face_slices[outlet_axis] = -1
    face_sl = tuple(face_slices)
    wall_face = wall_mask[face_sl]

    def outlet_pressure_avg(pressure):
        p_face = pressure[face_sl]
        fluid_count = jnp.sum(~wall_face)
        p_sum = jnp.sum(jnp.where(wall_face, 0.0, p_face))
        return p_sum / jnp.maximum(fluid_count, 1.0)

    return outlet_pressure_avg


def build_vessel_flow_graph(
    grid_shape: tuple = (64, 32, 32),
    vessel_params: Optional[dict] = None,
    heart_rate: float = 72.0,
    stroke_volume: float = 0.002,
    resistance: float = 10.0,
    compliance: float = 1.0,
    viscosity: float = 0.1,
    dt: float = 0.001,
    max_coupling_iters: int = 10,
    coupling_tolerance: float = 1e-6,
) -> tuple[GraphManager, jnp.ndarray]:
    """Build a coupled HeartPump + LBM graph for vessel flow.

    The coupling is bidirectional:

    * HeartPump ``arterial_pressure`` --> LBM ``inlet_pressure``
      (direct state edge).
    * LBM ``pressure`` --> HeartPump ``backpressure``
      (state edge with outlet-face-average transform).

    Parameters
    ----------
    grid_shape : tuple of int
        LBM grid dimensions, e.g. ``(64, 32, 32)``.
    vessel_params : dict or None
        Vessel geometry parameters for ``voxelize_vessel``.
        If None, uses sensible defaults scaled to *grid_shape*.
    heart_rate : float
        Heart rate in BPM.
    stroke_volume : float
        Stroke volume in LBM units.
    resistance : float
        Peripheral vascular resistance.
    compliance : float
        Arterial compliance.
    viscosity : float
        LBM kinematic viscosity.
    dt : float
        LBM timestep (both nodes use the same timestep).
    max_coupling_iters : int
        Maximum iterations for the coupling group.
    coupling_tolerance : float
        Convergence tolerance for the coupling group.

    Returns
    -------
    gm : GraphManager
        Compiled graph, ready for ``gm.step()`` or ``gm.run(n)``.
    vessel_mask : jnp.ndarray
        Boolean wall mask (True = wall), for reference or clot injection.
    """
    # 1. Vessel geometry
    if vessel_params is None:
        # Scale default params to grid_shape
        nx, ny, nz = grid_shape
        vessel_params = {
            "parent_radius": ny * 0.2,
            "daughter_radius": ny * 0.13,
            "parent_length": nx * 0.5,
            "daughter_length": nx * 0.4,
            "bifurcation_angle": 30.0,
        }

    vessel_mask = voxelize_vessel(grid_shape, vessel_params)

    # 2. Create nodes
    lbm_node = LBMNode(
        name="vessel",
        timestep=dt,
        grid_shape=grid_shape,
        viscosity=viscosity,
        wall_mask=np.asarray(vessel_mask),
        inlet_face="x_min",
        outlet_face="x_max",
    )

    heart_node = HeartPumpNode(
        name="heart",
        timestep=dt,
        resistance=resistance,
        compliance=compliance,
        heart_rate=heart_rate,
        stroke_volume=stroke_volume,
        venous_pressure=0.0,
        systole_fraction=0.35,
        initial_pressure=1.0 / 3.0,  # match LBM equilibrium pressure
    )

    # 3. Build graph
    gm = GraphManager()
    gm.add_node(heart_node)
    gm.add_node(lbm_node)

    # HeartPump -> LBM: arterial pressure drives LBM inlet directly
    gm.add_edge(
        "heart", "vessel",
        "arterial_pressure", "inlet_pressure",
    )

    # LBM -> HeartPump: outlet face average pressure -> backpressure
    # We use a transform that extracts the mean pressure on the outlet
    # face (masking wall cells), equivalent to what
    # LBMNode.compute_boundary_fluxes does for 'outlet_pressure_avg'.
    outlet_transform = _make_outlet_pressure_transform(
        vessel_mask, "x_max", ndim=len(grid_shape),
    )
    gm.add_edge(
        "vessel", "heart",
        "pressure", "backpressure",
        transform=outlet_transform,
    )

    # 4. No coupling group needed.
    #
    # The HeartPump <-> LBM coupling is weakly coupled: the Windkessel
    # responds on a cardiac-cycle timescale (~1s) while the LBM advances
    # at lattice dt (~1 unit).  One-step-lagged staggered feedback (via
    # back-edges) is physically appropriate and avoids the 10x overhead
    # of iterative coupling.  The cycle warning from compile() is
    # expected and safe to ignore.

    # 5. Compile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()

    return gm, vessel_mask
