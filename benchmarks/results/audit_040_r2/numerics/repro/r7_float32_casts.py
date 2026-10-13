"""float32 hard-casts on *parameters*, under jax_enable_x64.

MADD-ANO-013 registers one instance (HeartPumpNode.backpressure, a
boundary input).  TODO FOLLOW-UP C names a second (RigidBody2DNode's
gravity).  This enumerates what is actually in the tree and measures the
consequence for each: two parameter values that differ produce
bit-identical state.
"""
import os, warnings
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from maddening.nodes.rigid_body import RigidBodyNode
from maddening.nodes.rigid_body_2d import RigidBody2DNode
from maddening.nodes.ball import BallNode
from maddening.nodes.heart_pump import HeartPumpNode
from maddening.nodes.heat import HeatNode

warnings.simplefilter("ignore", DeprecationWarning)
DT = 0.01


def resolves(label, run, base, delta):
    a = run(base)
    b = run(base + delta)
    same = (a == b)
    print(f"  {label:58s} delta={delta:8.1e} rel={delta/base:8.1e}  "
          f"bit-identical={same}")
    return same


print("Under jax_enable_x64, does a parameter change below one float32 ulp "
      "move the state?\n")

# --- RigidBodyNode.inertia (cast at rigid_body.py:240 and :299) ---
def rb_inertia(I):
    n = RigidBodyNode("r", timestep=DT, mass=1.0, inertia=(I, I, I))
    st = {"position": jnp.zeros(3, dtype=jnp.float64),
          "orientation": jnp.asarray([1.0, 0., 0., 0.], dtype=jnp.float64),
          "velocity": jnp.zeros(3, dtype=jnp.float64),
          "angular_velocity": jnp.zeros(3, dtype=jnp.float64)}
    bi = {"torque": jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)}
    for _ in range(5):
        st = n.update(st, bi, DT)
    return float(st["angular_velocity"][0])

resolves("RigidBodyNode.inertia  (update)", rb_inertia, 1.0, 1e-8)
resolves("RigidBodyNode.inertia  (update)", rb_inertia, 1.0, 1e-6)

def rb_gravity(g):
    n = RigidBodyNode("r", timestep=DT, mass=1.0, gravity=(0.0, 0.0, g))
    st = {"position": jnp.zeros(3, dtype=jnp.float64),
          "orientation": jnp.asarray([1.0, 0., 0., 0.], dtype=jnp.float64),
          "velocity": jnp.zeros(3, dtype=jnp.float64),
          "angular_velocity": jnp.zeros(3, dtype=jnp.float64)}
    for _ in range(5):
        st = n.update(st, {}, DT)
    return float(st["velocity"][2])

resolves("RigidBodyNode.gravity  (update)", rb_gravity, -9.81, 1e-7)
resolves("RigidBodyNode.gravity  (update)", rb_gravity, -9.81, 1e-5)

def rb2_gravity(g):
    n = RigidBody2DNode("r2", timestep=DT, mass=1.0, gravity=(0.0, g))
    st = {"x": jnp.zeros(2, dtype=jnp.float64),
          "angle": jnp.asarray(0.0, dtype=jnp.float64),
          "v": jnp.zeros(2, dtype=jnp.float64),
          "omega": jnp.asarray(0.0, dtype=jnp.float64)}
    for _ in range(5):
        st = n.update(st, {}, DT)
    return float(st["v"][1])

resolves("RigidBody2DNode.gravity (update)  [TODO FOLLOW-UP C]", rb2_gravity, -9.81, 1e-7)

def ball_g(g):
    n = BallNode("b", timestep=DT, gravity=g)
    st = {"position": jnp.asarray(0.0, dtype=jnp.float64),
          "velocity": jnp.asarray(0.0, dtype=jnp.float64)}
    for _ in range(5):
        st = n.update(st, {}, DT)
    return float(st["velocity"])

resolves("BallNode.gravity (update -- NOT cast, control)", ball_g, -9.81, 1e-7)

def ball_g_deriv(g):
    n = BallNode("b", timestep=DT, gravity=g)
    st = {"position": jnp.asarray(0.0, dtype=jnp.float64),
          "velocity": jnp.asarray(0.0, dtype=jnp.float64)}
    return float(n.derivatives(st, {})["velocity"])

resolves("BallNode.gravity (derivatives -- cast at ball.py:152)",
         ball_g_deriv, -9.81, 1e-7)

def heart_bp(p):
    n = HeartPumpNode("h", timestep=DT)
    st = {"arterial_pressure": jnp.asarray(80.0, dtype=jnp.float64),
          "phase": jnp.asarray(0.0, dtype=jnp.float64),
          "flow_rate": jnp.asarray(0.0, dtype=jnp.float64)}
    return float(n.update(st, {"backpressure": jnp.asarray(p)}, DT)["arterial_pressure"])

resolves("HeartPumpNode.backpressure (MADD-ANO-013, registered)", heart_bp, 100.0, 1e-6)

# --- HeatNode.initial_state / grid_x ---
print("\nHeatNode dtype pinning under x64 (heat.py:565, :434, :438):")
h = HeatNode("h", timestep=1e-4, n_cells=10, length=1.0, thermal_diffusivity=0.01,
             initial_temperature=1.0)
print(f"  initial_state()['temperature'].dtype = {h.initial_state()['temperature'].dtype}"
      f"   (x64 enabled: {jax.config.jax_enable_x64})")
print(f"  static_data['grid_x'].value.dtype     = {h.static_data['grid_x'].value.dtype}")
hn = HeatNode("hn", timestep=1e-4, n_cells=10, length=1.0, thermal_diffusivity=0.01,
              grid_points=np.linspace(0.05, 0.95, 10))
print(f"  non-uniform grid_x dtype              = {hn._grid_x.dtype}")

def heat_nonuniform_run(shift):
    gp = np.linspace(0.05, 0.95, 10) + shift
    n = HeatNode("hn", timestep=1e-4, n_cells=10, length=1.0,
                 thermal_diffusivity=0.01, grid_points=gp)
    T = jnp.asarray(np.sin(np.linspace(0.05, 0.95, 10)), dtype=jnp.float64)
    out = n.update({"temperature": T},
                   {"left_temperature": jnp.asarray(0.0),
                    "right_temperature": jnp.asarray(1.0)}, 1e-4)
    return float(jnp.sum(out["temperature"]))

print()
resolves("HeatNode non-uniform grid_points (grid_x float32)",
         heat_nonuniform_run, 1.0, 1e-8)

# --- what the cast does to a convergence ladder on the rotational DOF ---
print("\nRigidBodyNode: float64 ladder on the ANGULAR dof, exact solution "
      "omega(t) = torque/I * t, so the discretisation error is ZERO and\n"
      "everything left is the float32 inertia cast:")
I = 1.0/3.0
for N in (100, 1000, 10000):
    dtN = 1.0/N
    n = RigidBodyNode("r", timestep=dtN, mass=1.0, inertia=(I, I, I))
    st = {"position": jnp.zeros(3, dtype=jnp.float64),
          "orientation": jnp.asarray([1., 0., 0., 0.], dtype=jnp.float64),
          "velocity": jnp.zeros(3, dtype=jnp.float64),
          "angular_velocity": jnp.zeros(3, dtype=jnp.float64)}
    bi = {"torque": jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)}
    for _ in range(N):
        st = n.update(st, bi, dtN)
    got = float(st["angular_velocity"][0])
    exact = 1.0/I
    print(f"   N={N:6d}  omega={got:.12f}  exact={exact:.12f}  "
          f"rel err={abs(got-exact)/exact:.3e}")
