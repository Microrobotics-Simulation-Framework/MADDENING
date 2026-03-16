"""
Analytical voxelizers for LBM wall-mask generation.

Provides geometry primitives that produce boolean wall masks on a regular
grid, suitable for use with :class:`~maddening.nodes.lbm.LBMNode`.

All functions return JAX arrays (``jnp.ndarray``) so the masks are ready
for JIT-compatible use.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def voxelize_vessel(grid_shape: tuple, vessel_params: dict) -> jnp.ndarray:
    """Generate a boolean wall mask for a Y-bifurcation vessel.

    The geometry consists of:
    1. A parent tube (cylinder along x-axis from x=0 to x=parent_length,
       centred at (y_mid, z_mid)).
    2. A left daughter tube branching at -bifurcation_angle from the
       bifurcation point.
    3. A right daughter tube branching at +bifurcation_angle from the
       bifurcation point.

    For each grid cell, the distance to the nearest centreline segment is
    computed.  If the distance is less than the radius for that segment,
    the cell is marked as fluid (mask=False).

    Parameters
    ----------
    grid_shape : tuple of int
        ``(nx, ny, nz)`` grid dimensions.
    vessel_params : dict
        Keys:
        - ``parent_radius`` : float -- parent tube radius in grid units.
        - ``daughter_radius`` : float -- daughter tube radius in grid units.
        - ``parent_length`` : float -- parent tube length in grid units.
        - ``daughter_length`` : float -- daughter tube length in grid units.
        - ``bifurcation_angle`` : float -- half-angle of the Y in degrees.
          Each daughter branches at +/- this angle from the parent axis.

    Returns
    -------
    wall_mask : jnp.ndarray, shape grid_shape, dtype bool
        True = wall, False = fluid.
    """
    nx, ny, nz = grid_shape
    pr = vessel_params["parent_radius"]
    dr = vessel_params["daughter_radius"]
    pl = vessel_params["parent_length"]
    dl = vessel_params["daughter_length"]
    angle_deg = vessel_params["bifurcation_angle"]
    angle_rad = np.deg2rad(angle_deg)

    # Centre of the yz cross-section
    y_mid = (ny - 1) / 2.0
    z_mid = (nz - 1) / 2.0

    # Bifurcation point: end of the parent tube
    bif_x = pl
    bif_y = y_mid
    bif_z = z_mid

    # Daughter centreline directions (branching in the x-y plane)
    # Left daughter: angle = -angle_rad (towards -y)
    # Right daughter: angle = +angle_rad (towards +y)
    left_dir = np.array([np.cos(angle_rad), -np.sin(angle_rad), 0.0])
    right_dir = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])

    # Build coordinate grids (use numpy for speed, convert to JAX at the end)
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    z = np.arange(nz, dtype=np.float64)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    # === Distance to parent centreline segment ===
    # Parent centreline: from (0, y_mid, z_mid) to (pl, y_mid, z_mid)
    # Project onto segment: t = clamp((P - A) . d / |d|^2, 0, 1)
    # Distance = |P - A - t * d|
    # For the parent along x-axis, simplify:
    # The centreline is y=y_mid, z=z_mid, x in [0, pl].
    t_parent = np.clip(xx / max(pl, 1e-10), 0.0, 1.0)
    closest_x_p = t_parent * pl
    dist_parent = np.sqrt(
        (xx - closest_x_p) ** 2
        + (yy - y_mid) ** 2
        + (zz - z_mid) ** 2
    )
    fluid_parent = dist_parent < pr

    # === Distance to left daughter centreline segment ===
    # Segment from (bif_x, bif_y, bif_z) to (bif + dl * left_dir)
    left_end = np.array([bif_x, bif_y, bif_z]) + dl * left_dir
    left_seg = dl * left_dir  # direction vector (not unit -- length = dl)
    # Vector from bif to each point
    dx_l = xx - bif_x
    dy_l = yy - bif_y
    dz_l = zz - bif_z
    seg_len_sq = np.dot(left_seg, left_seg)
    t_left = np.clip(
        (dx_l * left_seg[0] + dy_l * left_seg[1] + dz_l * left_seg[2])
        / max(seg_len_sq, 1e-10),
        0.0, 1.0,
    )
    closest_x_l = bif_x + t_left * left_seg[0]
    closest_y_l = bif_y + t_left * left_seg[1]
    closest_z_l = bif_z + t_left * left_seg[2]
    dist_left = np.sqrt(
        (xx - closest_x_l) ** 2
        + (yy - closest_y_l) ** 2
        + (zz - closest_z_l) ** 2
    )
    fluid_left = dist_left < dr

    # === Distance to right daughter centreline segment ===
    right_end = np.array([bif_x, bif_y, bif_z]) + dl * right_dir
    right_seg = dl * right_dir
    seg_len_sq_r = np.dot(right_seg, right_seg)
    t_right = np.clip(
        (dx_l * right_seg[0] + dy_l * right_seg[1] + dz_l * right_seg[2])
        / max(seg_len_sq_r, 1e-10),
        0.0, 1.0,
    )
    closest_x_r = bif_x + t_right * right_seg[0]
    closest_y_r = bif_y + t_right * right_seg[1]
    closest_z_r = bif_z + t_right * right_seg[2]
    dist_right = np.sqrt(
        (xx - closest_x_r) ** 2
        + (yy - closest_y_r) ** 2
        + (zz - closest_z_r) ** 2
    )
    fluid_right = dist_right < dr

    # Combine: fluid if inside any segment
    fluid = fluid_parent | fluid_left | fluid_right

    # Wall = not fluid
    wall_mask = ~fluid
    return jnp.asarray(wall_mask, dtype=jnp.bool_)
