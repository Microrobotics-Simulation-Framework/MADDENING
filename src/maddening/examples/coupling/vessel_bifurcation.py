"""
Vessel bifurcation example -- three HeatNodes coupled at a Y-junction.

Demonstrates the full USD workflow end-to-end:

1. **Create** a Y-shaped vessel phantom as a USD file
2. **Read** grid coordinates from USD geometry prims
3. **Build** three coupled HeatNodes at the bifurcation point
4. **Simulate** heat transfer from hot parent into cool daughters
5. **Write** results back to USD as time-sampled attributes
6. **Visualize** the vessel colored by temperature (optional, if PyVista available)

The parent tube carries hot fluid (100C) that flows into two daughter
branches (initially 20C).  At the bifurcation, the parent's rightmost
temperature is the Dirichlet BC for both daughters, and one daughter's
leftmost temperature feeds back as the parent's right BC.

Usage::

    JAX_PLATFORMS=cpu python -m maddening.examples.coupling.vessel_bifurcation

With visualization (requires pyvista)::

    JAX_PLATFORMS=cpu python -m maddening.examples.coupling.vessel_bifurcation --viz
"""

import os
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

# Import USD module first (before any Usd.Stage operations)
import maddening.usd
from maddening.usd.geometry import create_vessel_phantom, load_grid_from_usd
from maddening.usd.writer import USDWriter

from maddening.core.graph_manager import GraphManager
from maddening.core.transforms import register_transform
from maddening.nodes.heat import HeatNode


# --- Register transforms (required for USD serialization) ---

@register_transform("extract_right_temp", "Extract rightmost temperature")
def extract_right_temp(T):
    return T[-1]


@register_transform("extract_left_temp", "Extract leftmost temperature")
def extract_left_temp(T):
    return T[0]


def main(visualize: bool = False):
    print("=" * 60)
    print("Vessel Bifurcation: Y-junction heat coupling via OpenUSD")
    print("=" * 60)

    # --- Step 1: Create vessel phantom as USD file ---
    print("\n1. Creating Y-shaped vessel phantom...")
    tmpdir = tempfile.mkdtemp()
    vessel_path = os.path.join(tmpdir, "vessel.usda")
    results_path = os.path.join(tmpdir, "results.usda")

    stage = create_vessel_phantom(
        vessel_path,
        parent_length=1.0,
        daughter_length=0.8,
        parent_n_points=20,
        daughter_n_points=16,
        bifurcation_angle=30.0,
    )
    print(f"   Vessel phantom saved to: {vessel_path}")

    # --- Step 2: Read grid coordinates from USD geometry ---
    print("\n2. Reading grid coordinates from USD prims...")
    parent_x = load_grid_from_usd(stage, "/Vessel/parent", axis=0)
    left_x = load_grid_from_usd(stage, "/Vessel/daughter_left", axis=0)
    right_x = load_grid_from_usd(stage, "/Vessel/daughter_right", axis=0)

    print(f"   Parent:  {len(parent_x):3d} points, "
          f"x = [{parent_x[0]:.3f}, {parent_x[-1]:.3f}]")
    print(f"   Left:    {len(left_x):3d} points, "
          f"x = [{left_x[0]:.3f}, {left_x[-1]:.3f}]")
    print(f"   Right:   {len(right_x):3d} points, "
          f"x = [{right_x[0]:.3f}, {right_x[-1]:.3f}]")

    # --- Step 3: Create simulation nodes from USD geometry ---
    print("\n3. Building simulation graph from vessel geometry...")
    dt = 0.0001  # small timestep for stability with fine grid

    # Each node gets its grid_points from the USD centerline
    parent_node = HeatNode(
        name="parent",
        timestep=dt,
        n_cells=len(parent_x),
        length=float(parent_x[-1] - parent_x[0]),
        thermal_diffusivity=0.01,
        initial_temperature=100.0,  # hot fluid in parent
        grid_points=list(parent_x - parent_x[0]),
    )

    left_node = HeatNode(
        name="daughter_left",
        timestep=dt,
        n_cells=len(left_x),
        length=float(left_x[-1] - left_x[0]),
        thermal_diffusivity=0.01,
        initial_temperature=20.0,  # cool daughter
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

    gm = GraphManager()
    gm.add_node(parent_node)
    gm.add_node(left_node)
    gm.add_node(right_node)

    # Parent -> both daughters at the bifurcation (Dirichlet BC)
    gm.add_edge("parent", "daughter_left",
                 "temperature", "left_temperature",
                 transform=extract_right_temp)
    gm.add_edge("parent", "daughter_right",
                 "temperature", "left_temperature",
                 transform=extract_right_temp)

    # One daughter feeds back to parent (right BC)
    gm.add_edge("daughter_left", "parent",
                 "temperature", "right_temperature",
                 transform=extract_left_temp)

    # Coupling group for the bifurcation
    gm.add_coupling_group(
        ["parent", "daughter_left", "daughter_right"],
        max_iterations=20,
        tolerance=1e-8,
        diagnostics=True,
    )

    gm.compile()
    print(f"   Graph compiled: 3 nodes, {len(gm._edges)} edges, "
          f"1 coupling group")

    # --- Step 4: Run simulation and write results to USD ---
    print("\n4. Running simulation and writing results to USD...")
    from pxr import Usd, Sdf
    results_stage = Usd.Stage.CreateNew(results_path)
    writer = USDWriter(results_stage, gm)

    n_steps = 200
    write_every = 10  # write every 10th step to keep file small

    for step in range(n_steps):
        state = gm.step()
        if step % write_every == 0:
            sim_time = step * dt
            writer.write_frame(state, sim_time)
            if step % 50 == 0:
                T_p = state["parent"]["temperature"]
                print(f"   Step {step:4d}: parent T_mean={float(T_p.mean()):.2f}, "
                      f"T_right={float(T_p[-1]):.2f}")

    # Final state
    writer.write_frame(state, n_steps * dt)
    results_stage.Save()
    print(f"   Simulation complete. Results saved to: {results_path}")

    # --- Step 5: Report final state ---
    print("\n5. Final state:")
    T_parent = state["parent"]["temperature"]
    T_left = state["daughter_left"]["temperature"]
    T_right = state["daughter_right"]["temperature"]

    total_energy = (float(jnp.sum(T_parent)) + float(jnp.sum(T_left))
                    + float(jnp.sum(T_right)))
    initial_energy = (len(parent_x) * 100.0 + len(left_x) * 20.0
                      + len(right_x) * 20.0)

    print(f"   Parent:   T = [{float(T_parent.min()):.1f}, "
          f"{float(T_parent.max()):.1f}] C")
    print(f"   Left:     T = [{float(T_left.min()):.1f}, "
          f"{float(T_left.max()):.1f}] C")
    print(f"   Right:    T = [{float(T_right.min()):.1f}, "
          f"{float(T_right.max()):.1f}] C")
    print(f"   Bifurcation: parent_right={float(T_parent[-1]):.1f} C, "
          f"left_in={float(T_left[0]):.1f} C, "
          f"right_in={float(T_right[0]):.1f} C")
    print(f"   Total energy: {total_energy:.1f} "
          f"(initial: {initial_energy:.1f}, "
          f"delta: {total_energy - initial_energy:+.1f})")

    diag = gm.coupling_diagnostics()
    group_key = list(diag.keys())[0]
    print(f"   Coupling: {diag[group_key]['iterations']} iterations, "
          f"residual = {diag[group_key]['residual']:.2e}")

    # --- Step 6: Visualize (optional) ---
    if visualize:
        print("\n6. Rendering visualization...")
        try:
            from maddening.viz.usd_viewer import render_usd_frame

            # Use the general-purpose USD viewer to render the final frame.
            # This reads simulation results from USD and vessel geometry
            # from the phantom file, producing a 3D rendering colored by
            # temperature.
            tube_configs = [
                {"prim": "/Vessel/parent", "node": "parent",
                 "field": "temperature", "radius": 0.05,
                 "label": "Temperature (C)"},
                {"prim": "/Vessel/daughter_left", "node": "daughter_left",
                 "field": "temperature", "radius": 0.035},
                {"prim": "/Vessel/daughter_right", "node": "daughter_right",
                 "field": "temperature", "radius": 0.035},
            ]

            # output = render_usd_frame(
            #     results_path=results_path,
            #     geometry_path=vessel_path,
            #     tube_configs=tube_configs,
            #     output_path="vessel_bifurcation.png",
            #     camera_position="xy",
            #     zoom=1.5,
            # )
            # print(f"   Screenshot saved to: {output}")

            #For interactive replay, use viewer_from_usd_with_geometry:
            
            from maddening.viz.usd_viewer import viewer_from_usd_with_geometry
            viewer = viewer_from_usd_with_geometry(
                results_path, vessel_path, tube_configs)
            viewer.show()  # interactive playback with time slider

        except ImportError as e:
            print(f"   Skipping visualization: {e}")
            print("   Install pyvista and scipy for visualization support.")
    else:
        print("\n   (Run with --viz for 3D visualization)")

    # Clean up
    os.unlink(vessel_path)
    os.unlink(results_path)
    os.rmdir(tmpdir)

    print("\nDone.")


if __name__ == "__main__":
    viz = "--viz" in sys.argv
    main(visualize=viz)
