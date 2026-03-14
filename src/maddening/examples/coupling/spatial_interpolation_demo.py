#!/usr/bin/env python
"""
Spatial interpolation maps demo.

Demonstrates the ``interface_mapping`` module which provides
ready-made transform factories for coupling nodes with different
spatial discretizations.

Setup: two 1D heat rods coupled at their shared boundary.
Rod A has a coarse grid (10 cells), rod B has a fine grid (50 cells).
Spatial interpolation maps the temperature fields between grids.

Also shows standalone usage of each interpolation method:
nearest-neighbor, linear, RBF, and conservative projection.

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python -m maddening.examples.coupling.spatial_interpolation_demo
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.core.interface_mapping import (
    conservative_projection_1d,
    linear_interpolation_1d,
    nearest_neighbor_1d,
    rbf_interpolation,
)
from maddening.nodes.heat import HeatNode


def demo_standalone_interpolation():
    """Demonstrate each interpolation method on a test signal."""
    print("=" * 60)
    print("Standalone Interpolation Methods")
    print("=" * 60)

    # Source: coarse grid, sine wave
    source_x = jnp.linspace(0, 1, 8)
    target_x = jnp.linspace(0, 1, 30)
    values = jnp.sin(source_x * jnp.pi)

    # Nearest-neighbor
    nn = nearest_neighbor_1d(source_x, target_x)
    result_nn = nn(values)

    # Linear
    lin = linear_interpolation_1d(source_x, target_x)
    result_lin = lin(values)

    # RBF (Gaussian)
    source_pts = source_x.reshape(-1, 1)
    target_pts = target_x.reshape(-1, 1)
    rbf = rbf_interpolation(source_pts, target_pts, epsilon=3.0,
                            kernel="gaussian")
    result_rbf = rbf(values)

    # Conservative projection
    source_bounds = jnp.linspace(0, 1, 9)   # 8 cells
    target_bounds = jnp.linspace(0, 1, 31)  # 30 cells
    cons = conservative_projection_1d(source_bounds, target_bounds)
    result_cons = cons(values)

    # Reference: exact sine on fine grid
    exact = jnp.sin(target_x * jnp.pi)

    # Compute errors
    methods = [
        ("Nearest-neighbor", result_nn),
        ("Linear", result_lin),
        ("RBF (Gaussian)", result_rbf),
        ("Conservative", result_cons),
    ]

    print(f"\n  Source: sin(pi*x) on {len(source_x)} points")
    print(f"  Target: {len(target_x)} points")
    print()
    print(f"  {'Method':<20} {'Max error':>10} {'RMS error':>10}")
    print(f"  {'-'*42}")

    for name, result in methods:
        max_err = float(jnp.max(jnp.abs(result - exact)))
        rms_err = float(jnp.sqrt(jnp.mean((result - exact) ** 2)))
        print(f"  {name:<20} {max_err:10.6f} {rms_err:10.6f}")
    print()


def demo_coupled_heat_rods():
    """Two heat rods at different resolutions, coupled at boundary."""
    print("=" * 60)
    print("Coupled Heat Rods (Different Resolutions)")
    print("=" * 60)

    n_coarse = 10
    n_fine = 40
    dt = 0.0005
    alpha = 0.01
    n_steps = 5000

    print(f"\n  Rod A: {n_coarse} cells (coarse), T_init = 100 C")
    print(f"  Rod B: {n_fine} cells (fine),   T_init = 0 C")
    print(f"  Coupling: A's right boundary <-> B's left boundary")
    print(f"  dt = {dt}, alpha = {alpha}, steps = {n_steps}")
    print()

    gm = GraphManager()

    rod_a = HeatNode(
        name="coarse", timestep=dt, n_cells=n_coarse,
        thermal_diffusivity=alpha, length=1.0,
        initial_temperature=100.0,
    )
    rod_b = HeatNode(
        name="fine", timestep=dt, n_cells=n_fine,
        thermal_diffusivity=alpha, length=1.0,
        initial_temperature=0.0,
    )
    gm.add_node(rod_a)
    gm.add_node(rod_b)

    # Coarse rightmost cell -> fine left BC (scalar)
    gm.add_edge("coarse", "fine", "temperature", "left_temperature",
                transform=lambda T: T[-1])
    # Fine leftmost cell -> coarse right BC (scalar)
    gm.add_edge("fine", "coarse", "temperature", "right_temperature",
                transform=lambda T: T[0])

    gm.add_coupling_group(
        ["coarse", "fine"],
        max_iterations=10, tolerance=1e-8,
        diagnostics=True,
    )
    gm.compile()

    state = gm.run_scan(n_steps)

    T_coarse = np.array(state["coarse"]["temperature"])
    T_fine = np.array(state["fine"]["temperature"])

    print(f"  Coarse rod: T_min={T_coarse.min():.2f}, "
          f"T_max={T_coarse.max():.2f}, T_mean={T_coarse.mean():.2f}")
    print(f"  Fine rod:   T_min={T_fine.min():.2f}, "
          f"T_max={T_fine.max():.2f}, T_mean={T_fine.mean():.2f}")
    print()

    # Check continuity at interface
    interface_gap = abs(T_coarse[-1] - T_fine[0])
    print(f"  Interface continuity gap: {interface_gap:.4f} C")
    assert interface_gap < 5.0, f"Interface gap too large: {interface_gap}"
    print("  Check: interface temperature is approximately continuous.")

    # Temperature should be monotonically decreasing across both rods
    combined = np.concatenate([T_coarse, T_fine])
    print(f"  Combined profile: {len(combined)} cells, "
          f"range [{combined.min():.1f}, {combined.max():.1f}] C")
    print()


def demo_full_field_interpolation():
    """Demonstrate using interpolation maps for full-field coupling."""
    print("=" * 60)
    print("Full-Field Interpolation via EdgeSpec Transform")
    print("=" * 60)

    n_coarse = 8
    n_fine = 32

    # Create interpolation maps
    coarse_x = jnp.linspace(0, 1, n_coarse)
    fine_x = jnp.linspace(0, 1, n_fine)
    coarse_to_fine = linear_interpolation_1d(coarse_x, fine_x)
    fine_to_coarse = linear_interpolation_1d(fine_x, coarse_x)

    # Test the transform
    coarse_field = jnp.sin(coarse_x * jnp.pi) * 100.0
    interpolated = coarse_to_fine(coarse_field)
    back_projected = fine_to_coarse(interpolated)

    # Round-trip error
    roundtrip_err = float(jnp.max(jnp.abs(back_projected - coarse_field)))
    print(f"\n  Coarse grid: {n_coarse} points")
    print(f"  Fine grid:   {n_fine} points")
    print(f"  Test field:  100*sin(pi*x)")
    print(f"  Round-trip error (coarse -> fine -> coarse): {roundtrip_err:.6f}")
    print()

    # Conservative projection preserves integral
    coarse_bounds = jnp.linspace(0, 1, n_coarse + 1)
    fine_bounds = jnp.linspace(0, 1, n_fine + 1)
    cons_c2f = conservative_projection_1d(coarse_bounds, fine_bounds)

    cell_values = jnp.sin(jnp.linspace(0.0625, 0.9375, n_coarse) * jnp.pi) * 100.0
    projected = cons_c2f(cell_values)

    coarse_widths = coarse_bounds[1:] - coarse_bounds[:-1]
    fine_widths = fine_bounds[1:] - fine_bounds[:-1]
    integral_coarse = float(jnp.sum(cell_values * coarse_widths))
    integral_fine = float(jnp.sum(projected * fine_widths))
    integral_err = abs(integral_coarse - integral_fine)

    print(f"  Conservative projection integral:")
    print(f"    Coarse integral: {integral_coarse:.6f}")
    print(f"    Fine integral:   {integral_fine:.6f}")
    print(f"    Error:           {integral_err:.2e}")
    assert integral_err < 1e-5, f"Conservation violation: {integral_err}"
    print("  Check: integral is conserved.")
    print()


def main() -> None:
    demo_standalone_interpolation()
    demo_coupled_heat_rods()
    demo_full_field_interpolation()
    print("All demos complete.")


if __name__ == "__main__":
    main()
