"""
USD geometry reader -- load grid/mesh data from USD prims.

Provides :func:`load_grid_from_usd` to read point coordinates
from a USD prim (e.g., a UsdGeomBasisCurves or a prim with
``points`` attribute) and convert them to a numpy array.

Also provides :func:`create_vessel_phantom` to programmatically
generate a Y-shaped bifurcating vessel as a USD stage.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pxr import Sdf, Usd, UsdGeom, Vt, Gf


def load_grid_from_usd(
    stage: Usd.Stage,
    prim_path: str,
    axis: int = 0,
) -> np.ndarray:
    """Read grid point coordinates from a USD prim.

    The prim should have a ``points`` attribute (e.g.,
    ``UsdGeomBasisCurves`` or ``UsdGeomPoints``).  The coordinates
    along the specified *axis* are extracted and returned sorted.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage containing the geometry.
    prim_path : str
        SdfPath to the prim.
    axis : int
        Which coordinate axis to extract (0=x, 1=y, 2=z).

    Returns
    -------
    numpy.ndarray, shape ``(N,)``
        Sorted coordinates along the specified axis.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"No prim at {prim_path}")

    points_attr = prim.GetAttribute("points")
    if not points_attr.IsValid():
        raise ValueError(f"Prim at {prim_path} has no 'points' attribute")

    points = points_attr.Get()
    if points is None:
        raise ValueError(f"'points' attribute at {prim_path} has no value")

    # Convert VtVec3fArray to numpy
    coords = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float64)

    # Extract the specified axis and sort
    axis_vals = coords[:, axis]
    axis_vals.sort()

    # Remove duplicates (within tolerance)
    if len(axis_vals) > 1:
        diffs = np.diff(axis_vals)
        mask = np.concatenate([[True], diffs > 1e-10])
        axis_vals = axis_vals[mask]

    return axis_vals


def create_vessel_phantom(
    filepath: str,
    parent_length: float = 1.0,
    daughter_length: float = 0.8,
    parent_radius: float = 0.05,
    daughter_radius: float = 0.035,
    parent_n_points: int = 20,
    daughter_n_points: int = 16,
    bifurcation_angle: float = 30.0,
) -> Usd.Stage:
    """Create a Y-shaped bifurcating vessel as a USD file.

    The vessel has three segments:
    - Parent tube: along the x-axis from x=0 to x=parent_length
    - Left daughter: branches at the bifurcation point with -angle
    - Right daughter: branches at the bifurcation point with +angle

    Each segment is stored as a ``UsdGeomBasisCurves`` prim with
    ``points`` representing the centreline and ``widths`` for the
    cross-section radius.

    Parameters
    ----------
    filepath : str
        Output file path (.usda or .usdc).
    parent_length : float
        Length of the parent tube.
    daughter_length : float
        Length of each daughter tube.
    parent_radius : float
        Radius of the parent tube.
    daughter_radius : float
        Radius of daughter tubes.
    parent_n_points : int
        Number of points along the parent centreline.
    daughter_n_points : int
        Number of points along each daughter centreline.
    bifurcation_angle : float
        Half-angle of the bifurcation in degrees.

    Returns
    -------
    Usd.Stage
        The created stage (already saved to disk).
    """
    stage = Usd.Stage.CreateNew(filepath)
    stage.SetDefaultPrim(stage.DefinePrim("/Vessel"))

    angle_rad = np.radians(bifurcation_angle)

    # --- Parent tube ---
    parent_x = np.linspace(0.0, parent_length, parent_n_points)
    parent_points = [Gf.Vec3f(float(x), 0.0, 0.0) for x in parent_x]
    parent_widths = [parent_radius * 2.0] * parent_n_points

    parent_prim = UsdGeom.BasisCurves.Define(stage, "/Vessel/parent")
    parent_prim.GetPointsAttr().Set(Vt.Vec3fArray(parent_points))
    parent_prim.GetWidthsAttr().Set(Vt.FloatArray(parent_widths))
    parent_prim.GetCurveVertexCountsAttr().Set(Vt.IntArray([parent_n_points]))
    parent_prim.GetTypeAttr().Set("linear")

    # Bifurcation point
    bif_x = parent_length
    bif_y = 0.0

    # --- Left daughter (branches downward: negative y) ---
    left_t = np.linspace(0.0, daughter_length, daughter_n_points)
    left_dx = left_t * np.cos(-angle_rad)
    left_dy = left_t * np.sin(-angle_rad)
    left_points = [
        Gf.Vec3f(float(bif_x + dx), float(bif_y + dy), 0.0)
        for dx, dy in zip(left_dx, left_dy)
    ]
    left_widths = [daughter_radius * 2.0] * daughter_n_points

    left_prim = UsdGeom.BasisCurves.Define(stage, "/Vessel/daughter_left")
    left_prim.GetPointsAttr().Set(Vt.Vec3fArray(left_points))
    left_prim.GetWidthsAttr().Set(Vt.FloatArray(left_widths))
    left_prim.GetCurveVertexCountsAttr().Set(Vt.IntArray([daughter_n_points]))
    left_prim.GetTypeAttr().Set("linear")

    # --- Right daughter (branches upward: positive y) ---
    right_dx = left_t * np.cos(angle_rad)
    right_dy = left_t * np.sin(angle_rad)
    right_points = [
        Gf.Vec3f(float(bif_x + dx), float(bif_y + dy), 0.0)
        for dx, dy in zip(right_dx, right_dy)
    ]
    right_widths = [daughter_radius * 2.0] * daughter_n_points

    right_prim = UsdGeom.BasisCurves.Define(stage, "/Vessel/daughter_right")
    right_prim.GetPointsAttr().Set(Vt.Vec3fArray(right_points))
    right_prim.GetWidthsAttr().Set(Vt.FloatArray(right_widths))
    right_prim.GetCurveVertexCountsAttr().Set(
        Vt.IntArray([daughter_n_points])
    )
    right_prim.GetTypeAttr().Set("linear")

    # Add metadata
    vessel_prim = stage.GetPrimAtPath("/Vessel")
    vessel_prim.SetMetadata(
        "comment",
        "Y-shaped bifurcating vessel phantom for coupled heat transfer"
    )

    stage.Save()
    return stage
