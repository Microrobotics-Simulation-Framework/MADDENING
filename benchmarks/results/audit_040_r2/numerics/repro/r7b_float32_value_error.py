"""The float32 parameter casts as a WRONG NUMBER, not just a lost ulp.

Each case below has an exact closed-form answer that the node's scheme
reproduces exactly (zero discretisation error), so the whole of the
residual error is the cast.  Refinement does not remove it.
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
warnings.simplefilter("ignore", DeprecationWarning)

f64 = jnp.float64

def rb_state(**kw):
    return {"position": jnp.zeros(3, dtype=f64),
            "orientation": jnp.asarray([1., 0., 0., 0.], dtype=f64),
            "velocity": jnp.zeros(3, dtype=f64),
            "angular_velocity": jnp.zeros(3, dtype=f64)}

print("=== RigidBodyNode: omega(T) = torque/I * T, exact for symplectic Euler ===")
for I in (1.0/3.0, 0.7, 1.3):
    for N in (10, 1000):
        dt = 1.0/N
        n = RigidBodyNode("r", timestep=dt, mass=1.0, inertia=(I, I, I))
        st = rb_state()
        bi = {"torque": jnp.asarray([1.0, 0., 0.], dtype=f64)}
        for _ in range(N):
            st = n.update(st, bi, dt)
        got, exact = float(st["angular_velocity"][0]), 1.0/I
        print(f"  I={I:<20.17g} N={N:5d}  omega={got:.15f}  exact={exact:.15f}  "
              f"rel err={abs(got-exact)/exact:.3e}")

print("\n=== RigidBodyNode: v(T) = g*T under free fall, exact ===")
for g in (-9.81, -1.62, -0.1):
    for N in (10, 1000):
        dt = 1.0/N
        n = RigidBodyNode("r", timestep=dt, mass=2.0, gravity=(0., 0., g))
        st = rb_state()
        for _ in range(N):
            st = n.update(st, {}, dt)
        got, exact = float(st["velocity"][2]), g
        print(f"  g={g:<10g} N={N:5d}  v={got:.15f}  exact={exact:.15f}  "
              f"rel err={abs(got-exact)/abs(exact):.3e}")

print("\n=== RigidBody2DNode: v_y(T) = g*T, exact  (TODO FOLLOW-UP C) ===")
for N in (10, 1000):
    dt = 1.0/N
    n = RigidBody2DNode("r2", timestep=dt, mass=2.0, gravity=(0., -9.81))
    st = {"x": jnp.zeros(2, dtype=f64), "angle": jnp.asarray(0., dtype=f64),
          "v": jnp.zeros(2, dtype=f64), "omega": jnp.asarray(0., dtype=f64)}
    for _ in range(N):
        st = n.update(st, {}, dt)
    print(f"  N={N:5d}  v_y={float(st['v'][1]):.15f}  exact=-9.810000000000000  "
          f"rel err={abs(float(st['v'][1])+9.81)/9.81:.3e}")

print("\n=== BallNode control: gravity NOT cast in update(), IS cast in derivatives() ===")
n = BallNode("b", timestep=0.1, gravity=-9.81)
st = {"position": jnp.asarray(0., dtype=f64), "velocity": jnp.asarray(0., dtype=f64)}
u = n.update(st, {}, 1.0)
d = n.derivatives(st, {})
print(f"  update()      dv over dt=1 : {float(u['velocity']):.15f}   "
      f"rel err={abs(float(u['velocity'])+9.81)/9.81:.3e}")
print(f"  derivatives() dv/dt        : {float(d['velocity']):.15f}   "
      f"rel err={abs(float(d['velocity'])+9.81)/9.81:.3e}")
print("  -> update() and derivatives() of the SAME node disagree about g "
      f"by {abs(float(u['velocity'])-float(d['velocity'])):.3e} absolute")

print("\n=== gradients survive the cast? d omega / d inertia ===")
def om(I):
    n = RigidBodyNode("r", timestep=0.1, mass=1.0, inertia=(1.0, 1.0, 1.0))
    st = rb_state()
    bi = {"torque": jnp.asarray([1.0, 0., 0.], dtype=f64)}
    out = n.update(st, bi, 0.1, params={"inertia": jnp.asarray([I, I, I])})
    return out["angular_velocity"][0]
I0 = jnp.asarray(1.0/3.0, dtype=f64)
g_ad = float(jax.grad(om)(I0))
h = 1e-6
g_fd = float((om(I0 + h) - om(I0 - h)) / (2*h))
print(f"  AD grad = {g_ad:.9f}   exact = {-0.1/ (1/3)**2:.9f}   "
      f"central FD(h=1e-6) = {g_fd:.9f}")
h = 1e-9
g_fd9 = float((om(I0 + h) - om(I0 - h)) / (2*h))
print(f"  central FD(h=1e-9) = {g_fd9:.9f}   <- FD below the float32 ulp of "
      f"inertia collapses")
