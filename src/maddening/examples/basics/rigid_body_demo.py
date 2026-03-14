#!/usr/bin/env python
"""
RigidBody2DNode demo using the MADDENING GraphManager.

Simulates a 2D rigid body (projectile) launched at an angle with an
applied torque, producing simultaneous translational (parabolic
trajectory) and rotational motion.

The results are verified against analytical solutions for:
- Position:  x(t) = x0 + vx0*t,  y(t) = y0 + vy0*t + 0.5*g*t^2
  (semi-implicit Euler, so small numerical error expected)
- Angle:  theta(t) = theta0 + omega0*t + 0.5*alpha*t^2
  where alpha = torque / inertia

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python maddening/examples/rigid_body_demo.py
"""

import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import math

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.rigid_body_2d import RigidBody2DNode


def main() -> None:
    # ---- Parameters -----------------------------------------------------
    dt = 0.001
    n_steps = 2000
    t_end = n_steps * dt  # 2.0 s

    mass = 2.0
    inertia = 0.5
    torque_value = 3.0  # constant applied torque (N*m)

    launch_speed = 20.0
    launch_angle = math.radians(45.0)
    vx0 = launch_speed * math.cos(launch_angle)
    vy0 = launch_speed * math.sin(launch_angle)

    gx, gy = 0.0, -9.81

    print("RigidBody2D Demo: Projectile with Rotation")
    print("=" * 60)
    print(f"  Mass:           {mass} kg")
    print(f"  Inertia:        {inertia} kg*m^2")
    print(f"  Launch speed:   {launch_speed} m/s at {math.degrees(launch_angle):.0f} deg")
    print(f"  vx0, vy0:       ({vx0:.4f}, {vy0:.4f}) m/s")
    print(f"  Applied torque: {torque_value} N*m (constant)")
    print(f"  Gravity:        ({gx}, {gy}) m/s^2")
    print(f"  Timestep:       {dt} s, Steps: {n_steps}, Total: {t_end} s")
    print()

    # ---- Build graph ----------------------------------------------------
    gm = GraphManager()

    body = RigidBody2DNode(
        name="body",
        timestep=dt,
        mass=mass,
        inertia=inertia,
        gravity=(gx, gy),
        initial_x=0.0,
        initial_y=0.0,
        initial_vx=vx0,
        initial_vy=vy0,
        initial_angle=0.0,
        initial_omega=0.0,
    )
    gm.add_node(body)

    # Declare external torque input
    gm.add_external_input(
        target_node="body",
        target_field="torque",
        shape=(),
        dtype=jnp.float32,
    )

    gm.compile()
    print(f"Schedule: {gm.schedule}")

    # ---- Run simulation -------------------------------------------------
    xs = []
    ys = []
    angles = []

    ext = {"body": {"torque": jnp.array(torque_value, dtype=jnp.float32)}}

    for i in range(n_steps):
        state = gm.step(external_inputs=ext)
        xs.append(float(state["body"]["x"][0]))
        ys.append(float(state["body"]["x"][1]))
        angles.append(float(state["body"]["angle"]))

    # ---- Analytical solution --------------------------------------------
    # Semi-implicit Euler for constant acceleration converges to the
    # exact solution for linear-in-time quantities.  For position with
    # constant acceleration the scheme is:
    #   v_new = v + a*dt
    #   x_new = x + v_new*dt
    # This is equivalent to x(t) = x0 + v0*t + a*t*(t+dt)/2 for the
    # accumulated result.  But for many steps the error is O(dt), so we
    # just compare at the final time with a tolerance.

    # Exact analytical:
    x_exact = vx0 * t_end
    y_exact = vy0 * t_end + 0.5 * gy * t_end**2
    vx_exact = vx0
    vy_exact = vy0 + gy * t_end

    alpha = torque_value / inertia  # angular acceleration
    angle_exact = 0.5 * alpha * t_end**2
    omega_exact = alpha * t_end

    final_state = gm.get_node_state("body")
    x_sim = float(final_state["x"][0])
    y_sim = float(final_state["x"][1])
    vx_sim = float(final_state["v"][0])
    vy_sim = float(final_state["v"][1])
    angle_sim = float(final_state["angle"])
    omega_sim = float(final_state["omega"])

    print()
    print("--- Final State Comparison ---")
    print(f"{'Quantity':<16} {'Simulated':>14} {'Analytical':>14} {'Error':>14}")
    print("-" * 60)
    print(f"{'x':.<16} {x_sim:14.6f} {x_exact:14.6f} {abs(x_sim - x_exact):14.2e}")
    print(f"{'y':.<16} {y_sim:14.6f} {y_exact:14.6f} {abs(y_sim - y_exact):14.2e}")
    print(f"{'vx':.<16} {vx_sim:14.6f} {vx_exact:14.6f} {abs(vx_sim - vx_exact):14.2e}")
    print(f"{'vy':.<16} {vy_sim:14.6f} {vy_exact:14.6f} {abs(vy_sim - vy_exact):14.2e}")
    print(f"{'angle (rad)':.<16} {angle_sim:14.6f} {angle_exact:14.6f} {abs(angle_sim - angle_exact):14.2e}")
    print(f"{'omega':.<16} {omega_sim:14.6f} {omega_exact:14.6f} {abs(omega_sim - omega_exact):14.2e}")
    print()

    # ---- Trajectory summary ---------------------------------------------
    max_height = max(ys)
    range_x = xs[-1]
    total_rotation_deg = math.degrees(angle_sim)

    print(f"  Max height:     {max_height:.4f} m")
    print(f"  Horizontal range: {range_x:.4f} m")
    print(f"  Total rotation: {total_rotation_deg:.2f} degrees ({angle_sim:.4f} rad)")
    print()

    # ---- Verification ---------------------------------------------------
    # Position tolerance: semi-implicit Euler has O(dt) error.
    # With dt=0.001 and t=2.0, expect errors on the order of dt*t*a ~ 0.02
    pos_tol = 0.1  # generous tolerance

    assert abs(x_sim - x_exact) < pos_tol, (
        f"X position error too large: {abs(x_sim - x_exact):.4e}"
    )
    print(f"Check: x position error {abs(x_sim - x_exact):.4e} < {pos_tol}")

    assert abs(y_sim - y_exact) < pos_tol, (
        f"Y position error too large: {abs(y_sim - y_exact):.4e}"
    )
    print(f"Check: y position error {abs(y_sim - y_exact):.4e} < {pos_tol}")

    # Velocity should be very close (Euler velocity is exact for constant accel)
    vel_tol = 0.01
    assert abs(vx_sim - vx_exact) < vel_tol, (
        f"vx error too large: {abs(vx_sim - vx_exact):.4e}"
    )
    assert abs(vy_sim - vy_exact) < vel_tol, (
        f"vy error too large: {abs(vy_sim - vy_exact):.4e}"
    )
    print(f"Check: velocity errors within {vel_tol}")

    # Angle
    angle_tol = 0.1
    assert abs(angle_sim - angle_exact) < angle_tol, (
        f"Angle error too large: {abs(angle_sim - angle_exact):.4e}"
    )
    print(f"Check: angle error {abs(angle_sim - angle_exact):.4e} < {angle_tol}")

    # Angular velocity should be very close
    omega_tol = 0.01
    assert abs(omega_sim - omega_exact) < omega_tol, (
        f"omega error too large: {abs(omega_sim - omega_exact):.4e}"
    )
    print(f"Check: omega error {abs(omega_sim - omega_exact):.4e} < {omega_tol}")

    # Projectile should have gone forward
    assert x_sim > 10.0, f"Projectile didn't travel far enough: x={x_sim}"
    print(f"Check: projectile traveled {x_sim:.2f} m horizontally.")

    # Body should have rotated significantly with applied torque
    assert angle_sim > 1.0, f"Body didn't rotate enough: angle={angle_sim}"
    print(f"Check: body rotated {total_rotation_deg:.1f} degrees.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
