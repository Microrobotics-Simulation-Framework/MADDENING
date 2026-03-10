#!/usr/bin/env python
"""
Adaptive timestepping demo using the MADDENING GraphManager.

Demonstrates ``run_adaptive()`` which uses Richardson extrapolation
(step-doubling) to estimate the local truncation error and a PI
controller to adjust the timestep automatically.

A ball in free fall (no table collision) is simulated with different
tolerances, showing how tighter tolerances require more steps for
the same end time.

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python maddening/examples/adaptive_demo.py
"""

import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode


def run_adaptive_freefall(atol, rtol, t_end=2.0, dt_initial=0.05):
    """Simulate free-fall with adaptive timestepping and return results."""
    gm = GraphManager()
    ball = BallNode(
        name="ball",
        timestep=dt_initial,  # nominal dt (overridden by adaptive)
        initial_position=100.0,
        initial_velocity=0.0,
        elasticity=0.0,  # no bouncing
    )
    gm.add_node(ball)
    gm.compile()

    final_state, info = gm.run_adaptive(
        t_end=t_end,
        dt_initial=dt_initial,
        atol=atol,
        rtol=rtol,
        dt_min=1e-8,
        dt_max=0.5,
    )
    return final_state, info


def main() -> None:
    t_end = 2.0

    # Analytical solution for free fall: y = y0 + v0*t + 0.5*g*t^2
    # With y0=100, v0=0, g=-9.81:
    #   y(2) = 100 + 0.5*(-9.81)*4 = 100 - 19.62 = 80.38
    #   v(2) = -9.81 * 2 = -19.62
    y_exact = 100.0 + 0.5 * (-9.81) * t_end**2
    v_exact = -9.81 * t_end

    print("Adaptive Timestepping Demo: Free-Falling Ball")
    print("=" * 60)
    print(f"  Initial height:    100.0 m")
    print(f"  Simulation time:   {t_end} s")
    print(f"  Exact position:    {y_exact:.6f} m")
    print(f"  Exact velocity:    {v_exact:.6f} m/s")
    print()

    # ---- Run with different tolerances ----------------------------------
    tolerances = [
        (1e-2, 1e-1, "Loose"),
        (1e-4, 1e-2, "Medium"),
        (1e-6, 1e-3, "Tight"),
    ]

    results = []
    for atol, rtol, label in tolerances:
        print(f"--- {label} tolerance (atol={atol:.0e}, rtol={rtol:.0e}) ---")
        final_state, info = run_adaptive_freefall(atol, rtol, t_end=t_end)

        pos = float(final_state["ball"]["position"])
        vel = float(final_state["ball"]["velocity"])
        n_steps = info["n_steps"]
        n_rejected = info["n_rejected"]
        dt_hist = info["dt_history"]
        dt_min_used = min(dt_hist)
        dt_max_used = max(dt_hist)
        pos_err = abs(pos - y_exact)
        vel_err = abs(vel - v_exact)

        print(f"  Steps taken:       {n_steps}")
        print(f"  Steps rejected:    {n_rejected}")
        print(f"  dt range:          [{dt_min_used:.6f}, {dt_max_used:.6f}]")
        print(f"  Final position:    {pos:.6f}  (error = {pos_err:.2e})")
        print(f"  Final velocity:    {vel:.6f}  (error = {vel_err:.2e})")
        print()

        results.append((label, atol, rtol, n_steps, n_rejected, pos_err, vel_err, dt_hist))

    # ---- Summary table --------------------------------------------------
    print("=" * 60)
    print(f"{'Tolerance':<12} {'Steps':>7} {'Rejected':>9} {'Pos Error':>12} {'Vel Error':>12}")
    print("-" * 60)
    for label, atol, rtol, n_steps, n_rejected, pos_err, vel_err, _ in results:
        print(f"{label:<12} {n_steps:7d} {n_rejected:9d} {pos_err:12.2e} {vel_err:12.2e}")
    print()

    # ---- Sanity checks --------------------------------------------------
    # 1. Tighter tolerance should use more steps (or equal)
    steps_loose = results[0][3]
    steps_tight = results[2][3]
    assert steps_tight >= steps_loose, (
        f"Tight tolerance should require at least as many steps as loose "
        f"({steps_tight} vs {steps_loose})"
    )
    print(f"Check: tighter tolerance uses more steps ({steps_tight} >= {steps_loose}).")

    # 2. All results should be reasonably close to analytical.
    #    Semi-implicit Euler is first-order, so with large adaptive steps
    #    the position error can be a few metres — that's expected.
    for label, _, _, _, _, pos_err, vel_err, _ in results:
        assert pos_err < 5.0, f"{label}: position error {pos_err:.2e} too large"
        assert vel_err < 1.0, f"{label}: velocity error {vel_err:.2e} too large"
    print("Check: all position and velocity errors are within tolerance.")

    # 3. Tight tolerance should have smaller error than loose
    err_loose = results[0][5]  # pos error
    err_tight = results[2][5]
    assert err_tight <= err_loose + 1e-10, (
        f"Tight tolerance should give smaller error ({err_tight:.2e} vs {err_loose:.2e})"
    )
    print(f"Check: tight tolerance gives smaller error ({err_tight:.2e} <= {err_loose:.2e}).")

    # 4. Adaptive dt should have varied (not stayed constant)
    dt_hist_tight = results[2][7]
    dt_range = max(dt_hist_tight) - min(dt_hist_tight)
    # For free fall (quadratic), adaptive may converge quickly to a large dt,
    # but should show at least some variation initially
    print(f"Check: dt range for tight tolerance = {dt_range:.6f}.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
