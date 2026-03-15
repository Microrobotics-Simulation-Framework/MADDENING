"""
Vessel bifurcation example -- three HeatNodes coupled at a Y-junction.

Demonstrates:
- USD vessel phantom creation (Y-shaped pipe)
- Reading grid coordinates from USD geometry
- Three coupled HeatNodes at the bifurcation point
- Non-uniform grid support via geometry_source

The parent tube feeds hot fluid into two daughter branches.  The
bifurcation point couples all three nodes: the parent's right
temperature flows into both daughters' left temperatures, and
the daughters' reflected heat fluxes feed back to the parent.
"""

import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

# Import USD module first (before any Usd.Stage operations)
import maddening.usd
from maddening.usd.geometry import create_vessel_phantom, load_grid_from_usd

from maddening.core.graph_manager import GraphManager
from maddening.core.transforms import register_transform
from maddening.nodes.heat import HeatNode


@register_transform("extract_right_temp", "Extract rightmost temperature")
def extract_right_temp(T):
    return T[-1]


@register_transform("extract_left_temp", "Extract leftmost temperature")
def extract_left_temp(T):
    return T[0]


def main():
    # --- Create vessel phantom ---
    tmpdir = tempfile.mkdtemp()
    vessel_path = os.path.join(tmpdir, "vessel.usda")
    stage = create_vessel_phantom(
        vessel_path,
        parent_length=1.0,
        daughter_length=0.8,
        parent_n_points=20,
        daughter_n_points=16,
    )

    # --- Read grid coordinates from USD ---
    parent_x = load_grid_from_usd(stage, "/Vessel/parent", axis=0)
    left_x = load_grid_from_usd(stage, "/Vessel/daughter_left", axis=0)
    right_x = load_grid_from_usd(stage, "/Vessel/daughter_right", axis=0)

    print(f"Parent grid:  {len(parent_x)} points, x=[{parent_x[0]:.3f}, {parent_x[-1]:.3f}]")
    print(f"Left daughter: {len(left_x)} points, x=[{left_x[0]:.3f}, {left_x[-1]:.3f}]")
    print(f"Right daughter: {len(right_x)} points, x=[{right_x[0]:.3f}, {right_x[-1]:.3f}]")

    # --- Create simulation nodes ---
    dt = 0.0001  # small timestep for stability with fine grid

    parent_node = HeatNode(
        name="parent",
        timestep=dt,
        n_cells=len(parent_x),
        length=float(parent_x[-1] - parent_x[0]),
        thermal_diffusivity=0.01,
        initial_temperature=100.0,  # hot fluid in parent
        grid_points=list(parent_x - parent_x[0]),  # shift to start at 0
    )

    left_node = HeatNode(
        name="daughter_left",
        timestep=dt,
        n_cells=len(left_x),
        length=float(left_x[-1] - left_x[0]),
        thermal_diffusivity=0.01,
        initial_temperature=20.0,  # cool daughter branches
        grid_points=list(left_x - left_x[0]),
    )

    right_node = HeatNode(
        name="daughter_right",
        timestep=dt,
        n_cells=len(right_x),
        length=float(right_x[-1] - right_x[0]),
        thermal_diffusivity=0.01,
        initial_temperature=20.0,
        grid_points=list(right_x - right_x[0]),
    )

    # --- Build graph ---
    gm = GraphManager()
    gm.add_node(parent_node)
    gm.add_node(left_node)
    gm.add_node(right_node)

    # Parent -> daughters at bifurcation
    gm.add_edge("parent", "daughter_left",
                 "temperature", "left_temperature",
                 transform=extract_right_temp)
    gm.add_edge("parent", "daughter_right",
                 "temperature", "left_temperature",
                 transform=extract_right_temp)

    # Daughters -> parent (feedback via left temperature)
    gm.add_edge("daughter_left", "parent",
                 "temperature", "right_temperature",
                 transform=extract_left_temp)

    # Couple the bifurcation
    gm.add_coupling_group(
        ["parent", "daughter_left", "daughter_right"],
        max_iterations=20,
        tolerance=1e-8,
        diagnostics=True,
    )

    gm.compile()

    # --- Run simulation ---
    n_steps = 100
    for step in range(n_steps):
        state = gm.step()

    # --- Report ---
    T_parent = state["parent"]["temperature"]
    T_left = state["daughter_left"]["temperature"]
    T_right = state["daughter_right"]["temperature"]

    print(f"\nAfter {n_steps} steps (t = {n_steps * dt:.4f}s):")
    print(f"  Parent:   T_min={float(T_parent.min()):.2f}, T_max={float(T_parent.max()):.2f}")
    print(f"  Left:     T_min={float(T_left.min()):.2f}, T_max={float(T_left.max()):.2f}")
    print(f"  Right:    T_min={float(T_right.min()):.2f}, T_max={float(T_right.max()):.2f}")
    print(f"  Bifurcation: parent_right={float(T_parent[-1]):.2f}, "
          f"left_in={float(T_left[0]):.2f}, right_in={float(T_right[0]):.2f}")

    # Clean up
    os.unlink(vessel_path)
    os.rmdir(tmpdir)


if __name__ == "__main__":
    main()
