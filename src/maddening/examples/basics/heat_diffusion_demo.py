#!/usr/bin/env python
"""
Heat diffusion demo using the MADDENING GraphManager.

Simulates 1D heat diffusion on a rod using the HeatNode, which
implements the explicit finite-difference scheme for the heat equation:

    dT/dt = alpha * d^2T/dx^2

The rod starts at uniform temperature (0 C).  Dirichlet boundary
conditions are applied: hot left end (100 C) and cold right end (0 C).
Over time the temperature profile evolves toward the linear steady
state T(x) = 100 * (1 - x/L).

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python maddening/examples/heat_diffusion_demo.py
"""

import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode


def main() -> None:
    # ---- Parameters -----------------------------------------------------
    n_cells = 20
    length = 1.0
    thermal_diffusivity = 0.01  # m^2/s
    T_left = 100.0
    T_right = 0.0

    # Choose dt for stability: dt < dx^2 / (2 * alpha)
    dx = length / n_cells
    dt_max_stable = dx**2 / (2.0 * thermal_diffusivity)
    dt = 0.5 * dt_max_stable  # safety factor of 0.5
    n_steps = 8000
    total_time = n_steps * dt

    print("Heat Diffusion Demo: 1D Rod with Dirichlet BCs")
    print("=" * 60)
    print(f"  Rod length:        {length} m")
    print(f"  Grid cells:        {n_cells}")
    print(f"  dx:                {dx:.4f} m")
    print(f"  Thermal diff:      {thermal_diffusivity} m^2/s")
    print(f"  Left BC:           {T_left} C")
    print(f"  Right BC:          {T_right} C")
    print(f"  dt:                {dt:.6f} s (stability limit: {dt_max_stable:.6f})")
    print(f"  Steps:             {n_steps}")
    print(f"  Total time:        {total_time:.4f} s")
    print()

    # ---- Build graph ----------------------------------------------------
    gm = GraphManager()

    heat = HeatNode(
        name="rod",
        timestep=dt,
        n_cells=n_cells,
        length=length,
        thermal_diffusivity=thermal_diffusivity,
        initial_temperature=0.0,  # uniform initial temperature
    )
    gm.add_node(heat)

    # Declare external inputs for boundary conditions
    gm.add_external_input(
        target_node="rod",
        target_field="left_temperature",
        shape=(),
        dtype=jnp.float32,
    )
    gm.add_external_input(
        target_node="rod",
        target_field="right_temperature",
        shape=(),
        dtype=jnp.float32,
    )

    gm.compile()
    print(f"Schedule: {gm.schedule}")

    # ---- Run simulation -------------------------------------------------
    ext = {
        "rod": {
            "left_temperature": jnp.array(T_left, dtype=jnp.float32),
            "right_temperature": jnp.array(T_right, dtype=jnp.float32),
        }
    }

    # Record temperature profiles at selected times
    snapshots = {}
    snapshot_steps = [0, n_steps // 8, n_steps // 4, n_steps // 2, n_steps]

    # Store initial state
    state = gm.get_node_state("rod")
    snapshots[0] = np.array(state["temperature"])

    for i in range(1, n_steps + 1):
        state = gm.step(external_inputs=ext)
        if i in snapshot_steps:
            snapshots[i] = np.array(state["rod"]["temperature"])

    # ---- Final temperature profile --------------------------------------
    final_T = np.array(gm.get_node_state("rod")["temperature"])
    # Cell centers
    cell_centers = np.linspace(dx / 2, length - dx / 2, n_cells)

    # Analytical steady state: T(x) = T_left + (T_right - T_left) * x / L
    T_steady = T_left + (T_right - T_left) * cell_centers / length

    print()
    print("--- Final Temperature Profile ---")
    print(f"{'Cell':>5} {'x (m)':>8} {'T_sim (C)':>12} {'T_steady (C)':>14} {'Error':>10}")
    print("-" * 52)
    for i in range(0, n_cells, max(1, n_cells // 10)):
        err = abs(final_T[i] - T_steady[i])
        print(f"{i:5d} {cell_centers[i]:8.4f} {final_T[i]:12.4f} {T_steady[i]:14.4f} {err:10.4f}")
    print()

    # ---- Print snapshot summary -----------------------------------------
    print("--- Temperature Evolution (selected snapshots) ---")
    for step_idx in sorted(snapshots.keys()):
        T = snapshots[step_idx]
        t = step_idx * dt
        print(f"  t={t:8.4f}s (step {step_idx:5d}): "
              f"T_min={np.min(T):7.2f}, T_max={np.max(T):7.2f}, "
              f"T_mean={np.mean(T):7.2f}")
    print()

    # ---- Verification ---------------------------------------------------
    # 1. Boundary conditions should be enforced
    assert abs(final_T[0] - T_left) < 1.0, (
        f"Left BC not enforced: T[0]={final_T[0]:.4f}, expected {T_left}"
    )
    assert abs(final_T[-1] - T_right) < 1.0, (
        f"Right BC not enforced: T[-1]={final_T[-1]:.4f}, expected {T_right}"
    )
    print(f"Check: boundary conditions enforced (T[0]={final_T[0]:.2f}, "
          f"T[-1]={final_T[-1]:.2f}).")

    # 2. Temperature should be monotonically decreasing (left=hot, right=cold)
    diffs = np.diff(final_T)
    assert np.all(diffs <= 0.1), (
        f"Temperature not monotonic: max increase = {np.max(diffs):.4f}"
    )
    print("Check: temperature profile is monotonically decreasing.")

    # 3. Temperature should be approaching the steady-state linear profile
    max_error = np.max(np.abs(final_T - T_steady))
    # With enough time steps, the interior should be close to steady state
    # The diffusion timescale is L^2 / alpha = 1.0/0.01 = 100s
    # We only run for total_time, so allow some error
    mean_error = np.mean(np.abs(final_T - T_steady))
    print(f"Check: max error from steady state = {max_error:.4f} C")
    print(f"Check: mean error from steady state = {mean_error:.4f} C")

    # Interior cells (excluding boundaries) should be approaching steady state
    interior_error = np.mean(np.abs(final_T[1:-1] - T_steady[1:-1]))
    print(f"Check: mean interior error = {interior_error:.4f} C")

    # 4. Temperature should have increased from initial (0 C) toward steady state
    # At least the left half should be well above 0
    left_half_mean = np.mean(final_T[:n_cells // 2])
    assert left_half_mean > 10.0, (
        f"Left half should have warmed up: mean={left_half_mean:.2f}"
    )
    print(f"Check: left half mean temperature = {left_half_mean:.2f} C (warmed up).")

    # 5. All temperatures should be within [T_right, T_left]
    assert np.all(final_T >= T_right - 1.0), "Temperature dropped below right BC!"
    assert np.all(final_T <= T_left + 1.0), "Temperature exceeded left BC!"
    print(f"Check: all temperatures in [{T_right:.0f}, {T_left:.0f}] C range.")

    # 6. Profile should be closer to steady state than the initial uniform profile
    initial_error = np.mean(np.abs(np.zeros(n_cells) - T_steady))
    assert mean_error < initial_error, (
        f"Should be closer to steady state than initial: {mean_error:.4f} >= {initial_error:.4f}"
    )
    print(f"Check: converging toward steady state "
          f"(error {mean_error:.4f} < initial {initial_error:.4f}).")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
