#!/usr/bin/env python
"""
Planar rigid-body demo using the MADDENING GraphManager.

Simulates a rigid body (projectile) launched at an angle with an
applied torque, producing simultaneous translational (parabolic
trajectory) and rotational motion.

The body is a :class:`~maddening.nodes.rigid_body.RigidBodyNode` (6-DOF)
held in the x-y plane by DOF constraints -- ``z`` locked at 0 and no
rotation about ``x`` or ``y`` -- which is how 0.4.0 expresses what the
deprecated ``RigidBody2DNode`` used to.  Its orientation is a
quaternion; the in-plane rotation angle is read back from it.

The results are verified against analytical solutions for:
- Position:  x(t) = x0 + vx0*t,  y(t) = y0 + vy0*t + 0.5*g*t^2
  (semi-implicit Euler, so small numerical error expected)
- Angle:  theta(t) = theta0 + omega0*t + 0.5*alpha*t^2
  where alpha = torque / inertia

Usage
-----
    python -m maddening.examples.basics.rigid_body_demo
"""

import math

import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.nodes.rigid_body import RigidBodyNode


def planar_angle(orientation) -> np.ndarray:
    """Unwrapped rotation angle about z from a history of quaternions.

    ``orientation`` is ``(..., 4)`` in ``(w, x, y, z)`` order.  With the
    ``rx``/``ry`` constraints the rotation is purely about z, so the
    quaternion is ``(cos(theta/2), 0, 0, sin(theta/2))``.
    """
    q = np.asarray(orientation, dtype=np.float64)
    return np.unwrap(2.0 * np.arctan2(q[..., 3], q[..., 0]))


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

    print("Planar Rigid-Body Demo: Projectile with Rotation")
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

    body = RigidBodyNode(
        name="body",
        timestep=dt,
        mass=mass,
        # Only the z entry matters in the plane; the other two are inert
        # because rotation about x and y is locked.
        inertia=(inertia, inertia, inertia),
        gravity=(gx, gy, 0.0),
        constraints={"z": 0.0, "rx": 0.0, "ry": 0.0},
        initial_position=(0.0, 0.0, 0.0),
        initial_velocity=(vx0, vy0, 0.0),
    )
    gm.add_node(body)

    # Declare external torque input (a 3-vector; only z acts in the plane)
    gm.add_external_input(
        target_node="body",
        target_field="torque",
        shape=(3,),
        dtype=jnp.float32,
    )

    gm.compile()
    print(f"Schedule: {gm.schedule}")

    # ---- Run simulation -------------------------------------------------
    # The torque is constant, so the whole run fits in one lax.scan.
    ext = {"body": {"torque": jnp.array([0.0, 0.0, torque_value],
                                        dtype=jnp.float32)}}
    final_state, history = gm.run_scan_with_history(n_steps, external_inputs=ext)

    positions = np.asarray(history["body"]["position"])
    xs = positions[:, 0]
    ys = positions[:, 1]
    angles = planar_angle(history["body"]["orientation"])

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

    body_final = final_state["body"]
    x_sim = float(body_final["position"][0])
    y_sim = float(body_final["position"][1])
    vx_sim = float(body_final["velocity"][0])
    vy_sim = float(body_final["velocity"][1])
    angle_sim = float(angles[-1])
    omega_sim = float(body_final["angular_velocity"][2])
    z_sim = float(body_final["position"][2])

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
    max_height = float(np.max(ys))
    range_x = float(xs[-1])
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

    # The constraints hold the body in the plane
    assert z_sim == 0.0, f"z constraint not held: z={z_sim}"
    print("Check: body stayed in the x-y plane (z = 0).")

    # Projectile should have gone forward
    assert x_sim > 10.0, f"Projectile didn't travel far enough: x={x_sim}"
    print(f"Check: projectile traveled {x_sim:.2f} m horizontally.")

    # Body should have rotated significantly with applied torque
    assert angle_sim > 1.0, f"Body didn't rotate enough: angle={angle_sim}"
    print(f"Check: body rotated {total_rotation_deg:.1f} degrees.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
