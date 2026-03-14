#!/usr/bin/env python
"""
Spatial interpolation maps demo.

**The problem**: when coupling nodes with different spatial
discretizations (e.g., a coarse structural mesh to a fine fluid mesh),
you need to map field values between grids.  The ``interface_mapping``
module provides ready-made transform factories that plug directly into
``EdgeSpec(transform=...)``.

**Available methods:**

- **Nearest-neighbor**: fast, simple, discontinuous. Use for quick
  prototyping or when accuracy isn't critical.

- **Linear**: smooth, O(dx^2) accurate. Best general-purpose choice
  for 1D grids.

- **RBF** (radial basis functions): smooth, works in any dimension,
  handles scattered points. Best for non-uniform grids or 2D/3D.
  Kernels: gaussian, multiquadric, inverse_multiquadric, thin_plate_spline.

- **Conservative projection**: preserves the integral of the field.
  Essential for flux quantities (heat flux, mass flow) where the
  total must be conserved across grids.

All factories pre-compute interpolation weights at graph construction
time.  The returned transform is a pure JAX function -- JIT-compiled,
differentiable, and compatible with ``jax.grad`` through the graph.

Usage
-----
    python -m maddening.examples.coupling.spatial_interpolation_demo
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
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


def demo_interpolation_accuracy():
    """Compare interpolation methods on a known test function."""
    print("=" * 65)
    print("Part 1: Interpolation Accuracy Comparison")
    print("=" * 65)
    print()
    print("  Interpolating sin(pi*x) from 8 source points to 30 target points.")
    print("  Lower error = more accurate interpolation.")
    print()

    source_x = jnp.linspace(0, 1, 8)
    target_x = jnp.linspace(0, 1, 30)
    values = jnp.sin(source_x * jnp.pi)
    exact = jnp.sin(target_x * jnp.pi)

    # Build all transforms
    nn_fn = nearest_neighbor_1d(source_x, target_x)
    lin_fn = linear_interpolation_1d(source_x, target_x)
    source_pts = source_x.reshape(-1, 1)
    target_pts = target_x.reshape(-1, 1)
    rbf_fn = rbf_interpolation(source_pts, target_pts, epsilon=3.0,
                                kernel="gaussian")

    methods = [
        ("Nearest-neighbor", nn_fn(values)),
        ("Linear", lin_fn(values)),
        ("RBF (Gaussian)", rbf_fn(values)),
    ]

    print(f"  {'Method':<20} {'Max error':>10} {'RMS error':>10}  Notes")
    print(f"  {'-'*65}")
    for name, result in methods:
        max_err = float(jnp.max(jnp.abs(result - exact)))
        rms_err = float(jnp.sqrt(jnp.mean((result - exact) ** 2)))
        if name == "Nearest-neighbor":
            note = "fast, discontinuous jumps"
        elif name == "Linear":
            note = "smooth, good general-purpose"
        else:
            note = "smoothest, handles scattered points"
        print(f"  {name:<20} {max_err:10.6f} {rms_err:10.6f}  {note}")

    print()
    print("  RBF is most accurate but involves a matrix solve at setup time.")
    print("  Linear is the best balance of speed and accuracy for 1D grids.")
    print()


def demo_conservative_projection():
    """Show that conservative projection preserves integrals."""
    print("=" * 65)
    print("Part 2: Conservative Projection (Integral Preservation)")
    print("=" * 65)
    print()
    print("  For flux-like quantities (heat flux, mass flow), the TOTAL across")
    print("  all cells must be preserved when mapping between grids.  Linear")
    print("  interpolation does NOT guarantee this; conservative projection does.")
    print()

    n_coarse, n_fine = 8, 32
    coarse_bounds = jnp.linspace(0, 1, n_coarse + 1)
    fine_bounds = jnp.linspace(0, 1, n_fine + 1)
    coarse_centers = 0.5 * (coarse_bounds[:-1] + coarse_bounds[1:])
    fine_centers = 0.5 * (fine_bounds[:-1] + fine_bounds[1:])

    # Test field: sin(pi*x) cell averages
    cell_values = jnp.sin(coarse_centers * jnp.pi) * 100.0

    # Conservative
    cons_fn = conservative_projection_1d(coarse_bounds, fine_bounds)
    cons_result = cons_fn(cell_values)

    # Linear (for comparison)
    lin_fn = linear_interpolation_1d(coarse_centers, fine_centers)
    lin_result = lin_fn(cell_values)

    # Compute integrals
    coarse_widths = coarse_bounds[1:] - coarse_bounds[:-1]
    fine_widths = fine_bounds[1:] - fine_bounds[:-1]
    integral_source = float(jnp.sum(cell_values * coarse_widths))
    integral_cons = float(jnp.sum(cons_result * fine_widths))
    integral_lin = float(jnp.sum(lin_result * fine_widths))

    print(f"  Source integral ({n_coarse} cells):       {integral_source:.6f}")
    print(f"  Conservative ({n_fine} cells):      {integral_cons:.6f}  "
          f"error = {abs(integral_cons - integral_source):.2e}")
    print(f"  Linear interp ({n_fine} cells):     {integral_lin:.6f}  "
          f"error = {abs(integral_lin - integral_source):.2e}")
    print()
    print("  Conservative projection preserves the integral exactly.")
    print("  Linear interpolation introduces a small integration error.")
    print()


def demo_coupled_heat_rods():
    """Two heat rods at different resolutions, coupled at boundary."""
    print("=" * 65)
    print("Part 3: Coupled Heat Rods (Different Resolutions)")
    print("=" * 65)
    print()
    print("  Rod A (coarse, 10 cells) starts at 100 C.")
    print("  Rod B (fine, 40 cells) starts at 0 C.")
    print("  They exchange temperature at their shared boundary.")
    print("  Over time, heat flows from A to B through the interface.")
    print()

    n_coarse, n_fine = 10, 40
    dt = 0.0005
    alpha = 0.01
    n_steps = 10000

    gm = GraphManager()
    gm.add_node(HeatNode(
        name="coarse", timestep=dt, n_cells=n_coarse,
        thermal_diffusivity=alpha, length=1.0,
        initial_temperature=100.0,
    ))
    gm.add_node(HeatNode(
        name="fine", timestep=dt, n_cells=n_fine,
        thermal_diffusivity=alpha, length=1.0,
        initial_temperature=0.0,
    ))

    # Scalar boundary coupling (rightmost cell <-> leftmost cell)
    gm.add_edge("coarse", "fine", "temperature", "left_temperature",
                transform=lambda T: T[-1])
    gm.add_edge("fine", "coarse", "temperature", "right_temperature",
                transform=lambda T: T[0])

    # External BCs: coarse left = 100 C (hot wall), fine right = 0 C (cold wall)
    gm.add_external_input("coarse", "left_temperature")
    gm.add_external_input("fine", "right_temperature")

    gm.add_coupling_group(["coarse", "fine"],
                           max_iterations=10, tolerance=1e-8)
    gm.compile()

    ext = {
        "coarse": {"left_temperature": jnp.array(100.0)},
        "fine": {"right_temperature": jnp.array(0.0)},
    }
    state = gm.run_scan(n_steps, external_inputs=ext)

    T_coarse = np.array(state["coarse"]["temperature"])
    T_fine = np.array(state["fine"]["temperature"])

    print(f"  After {n_steps} steps (t = {n_steps * dt:.1f}s):")
    print(f"  Coarse rod: T = [{T_coarse[0]:.1f}, ..., {T_coarse[-1]:.1f}] C")
    print(f"  Fine rod:   T = [{T_fine[0]:.1f}, ..., {T_fine[-1]:.1f}] C")
    print()

    interface_gap = abs(T_coarse[-1] - T_fine[0])
    print(f"  Interface gap: |T_coarse[-1] - T_fine[0]| = {interface_gap:.4f} C")
    if interface_gap < 1.0:
        print("  -> Good: temperature is approximately continuous at interface.")
    else:
        print(f"  -> Gap is significant ({interface_gap:.1f} C).")
        print("     This is expected -- the coupling transmits boundary values,")
        print("     not fluxes. For better continuity, couple heat flux too.")
    print()


def demo_gradient_through_interpolation():
    """Show that gradients flow through interpolation transforms."""
    print("=" * 65)
    print("Part 4: Differentiability")
    print("=" * 65)
    print()
    print("  All interpolation maps are JAX-traceable.  Gradients flow")
    print("  through them, enabling end-to-end differentiation of coupled")
    print("  multi-resolution simulations.")
    print()

    source_x = jnp.linspace(0, 1, 5)
    target_x = jnp.linspace(0, 1, 20)
    interp = linear_interpolation_1d(source_x, target_x)

    def loss_fn(values):
        interpolated = interp(values)
        return jnp.sum(interpolated ** 2)

    grad = jax.grad(loss_fn)(jnp.ones(5))
    print(f"  d/d(source_values) sum(interp(values)^2) = {np.array(grad)}")
    print(f"  All finite: {bool(jnp.all(jnp.isfinite(grad)))}")
    print(f"  Non-zero:   {bool(jnp.any(grad != 0.0))}")
    print()
    print("  This means you can optimize source-grid parameters using")
    print("  gradients computed through the interpolation + simulation.")
    print()


def main() -> None:
    demo_interpolation_accuracy()
    demo_conservative_projection()
    demo_coupled_heat_rods()
    demo_gradient_through_interpolation()
    print("All demos complete.")


if __name__ == "__main__":
    main()
