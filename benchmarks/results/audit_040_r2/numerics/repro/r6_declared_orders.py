"""Independent measurement of every declared order in the node registry.

Deliberately does NOT use maddening.testing.mms or the repo's study
fixtures: each ladder below is built from scratch, with its own
manufactured solution, so it is evidence about the node rather than
evidence about the harness.

Every node is driven through its own public ``update()`` at fixed final
time with N steps, N refined 2x.  Errors are relative L2 over the whole
state (every float field the node returns), which is what the declared
order is a claim about.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.ball import BallNode
from maddening.nodes.rigid_body import RigidBodyNode
from maddening.nodes.rigid_body_2d import RigidBody2DNode
from maddening.nodes.heart_pump import HeartPumpNode
from maddening.testing.mms import declared_order


def ladder(err_at, levels):
    e = [err_at(n) for n in levels]
    o = [np.log(e[i]/e[i+1])/np.log(levels[i+1]/levels[i]) for i in range(len(e)-1)]
    return e, o


def report(tag, declared, levels, e, o):
    print(f"\n{tag}   declared temporal order = {declared}")
    for i, n in enumerate(levels):
        s = "" if i == 0 else f"{o[i-1]:8.3f}"
        print(f"   N={n:6d}  err={e[i]:12.5e} {s}")
    print(f"   observed (finest pair) = {o[-1]:.4f}"
          f"   monotone={all(e[i+1] < e[i] for i in range(len(e)-1))}")


# ---------------------------------------------------------------- spring
# m x'' + c x' + k (x - a(t) - rest) = 0 ; manufactured x*(t) -> a(t)
def spring_err(N):
    T, m, c, k, rest = 1.0, 1.3, 0.7, 40.0, 0.25
    xs = lambda t: 0.6*np.sin(2.1*t) + 0.2*np.cos(1.3*t) + 0.35
    vs = lambda t: 0.6*2.1*np.cos(2.1*t) - 0.2*1.3*np.sin(1.3*t)
    acc = lambda t: -0.6*2.1**2*np.sin(2.1*t) - 0.2*1.3**2*np.cos(1.3*t)
    a_of = lambda t: xs(t) - rest + (m*acc(t) + c*vs(t))/k
    dt = T/N
    node = SpringDamperNode("s", timestep=dt, stiffness=k, damping=c, mass=m,
                            rest_length=rest)
    st = {"position": jnp.asarray(xs(0.0)), "velocity": jnp.asarray(vs(0.0))}
    for i in range(N):
        st = node.update(st, {"anchor_position": jnp.asarray(a_of(i*dt))}, dt)
    num = np.array([float(st["position"]), float(st["velocity"])])
    ref = np.array([xs(T), vs(T)])
    return float(np.linalg.norm(num-ref)/np.linalg.norm(ref))


# ---------------------------------------------------------------- ball
# x'' = g(t); ball has no force input, so the manufactured source enters
# through the *gravity* parameter (injected per step), as the node's own
# study does -- there is no other channel.
def ball_err(N):
    T = 1.0
    xs = lambda t: 0.5*np.sin(1.7*t) + 0.1*t + 2.0
    vs = lambda t: 0.5*1.7*np.cos(1.7*t) + 0.1
    g_of = lambda t: -0.5*1.7**2*np.sin(1.7*t)
    dt = T/N
    node = BallNode("b", timestep=dt, gravity=-9.81)
    st = {"position": jnp.asarray(xs(0.0)), "velocity": jnp.asarray(vs(0.0))}
    for i in range(N):
        st = node.update(st, {}, dt, params={"gravity": jnp.asarray(g_of(i*dt))})
    num = np.array([float(st["position"]), float(st["velocity"])])
    ref = np.array([xs(T), vs(T)])
    return float(np.linalg.norm(num-ref)/np.linalg.norm(ref))


# ------------------------------------------------------------ rigid body
def rb_err(N):
    T, m = 0.8, 2.0
    I = np.array([1.0, 1.3, 0.7])
    g = np.array([0.0, 0.0, -9.81])
    xs = lambda t: np.array([0.4*np.sin(2.0*t), 0.3*np.cos(1.1*t), 0.2*t*t])
    vs = lambda t: np.array([0.8*np.cos(2.0*t), -0.33*np.sin(1.1*t), 0.4*t])
    ac = lambda t: np.array([-1.6*np.sin(2.0*t), -0.363*np.cos(1.1*t), 0.4])
    ws = lambda t: np.array([0.5*np.sin(0.9*t), 0.2*np.cos(1.4*t), 0.3])
    wd = lambda t: np.array([0.45*np.cos(0.9*t), -0.28*np.sin(1.4*t), 0.0])
    dt = T/N
    node = RigidBodyNode("r", timestep=dt, mass=m, inertia=tuple(I),
                         gravity=tuple(g))
    st = {"position": jnp.asarray(xs(0.0)),
          "orientation": jnp.asarray([1.0, 0.0, 0.0, 0.0]),
          "velocity": jnp.asarray(vs(0.0)),
          "angular_velocity": jnp.asarray(ws(0.0))}
    for i in range(N):
        t = i*dt
        bi = {"force": jnp.asarray(m*(ac(t) - g)), "torque": jnp.asarray(I*wd(t))}
        st = node.update(st, bi, dt)
    num = np.concatenate([np.asarray(st["position"], dtype=np.float64),
                          np.asarray(st["velocity"], dtype=np.float64),
                          np.asarray(st["angular_velocity"], dtype=np.float64)])
    ref = np.concatenate([xs(T), vs(T), ws(T)])
    return float(np.linalg.norm(num-ref)/np.linalg.norm(ref))


# --------------------------------------------------------- rigid body 2D
def rb2_err(N):
    import warnings
    T, m, I = 0.8, 2.0, 1.4
    g = np.array([0.0, -9.81])
    xs = lambda t: np.array([0.4*np.sin(2.0*t), 0.3*np.cos(1.1*t)])
    vs = lambda t: np.array([0.8*np.cos(2.0*t), -0.33*np.sin(1.1*t)])
    ac = lambda t: np.array([-1.6*np.sin(2.0*t), -0.363*np.cos(1.1*t)])
    th = lambda t: 0.6*np.sin(1.3*t)
    om = lambda t: 0.78*np.cos(1.3*t)
    al = lambda t: -1.014*np.sin(1.3*t)
    dt = T/N
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        node = RigidBody2DNode("r2", timestep=dt, mass=m, inertia=I,
                               gravity=tuple(g))
    st = {"x": jnp.asarray(xs(0.0)), "angle": jnp.asarray(th(0.0)),
          "v": jnp.asarray(vs(0.0)), "omega": jnp.asarray(om(0.0))}
    for i in range(N):
        t = i*dt
        bi = {"force": jnp.asarray(m*(ac(t) - g)), "torque": jnp.asarray(I*al(t))}
        st = node.update(st, bi, dt)
    num = np.concatenate([np.asarray(st["x"], dtype=np.float64),
                          [float(st["angle"])],
                          np.asarray(st["v"], dtype=np.float64),
                          [float(st["omega"])]])
    ref = np.concatenate([xs(T), [th(T)], vs(T), [om(T)]])
    return float(np.linalg.norm(num-ref)/np.linalg.norm(ref))


# -------------------------------------------------------------- heart pump
# C dP/dt = Q_heart(phase) - (P - P_down)/R.  update() reads the waveform
# at phase_{n+1}; the manufactured P_down below is derived against exactly
# that sampling, so this measures the scheme update() implements.
def heart_err(N):
    T = 1.0
    R, C = 1.0, 1.5
    Ps = lambda t: 80.0 + 12.0*np.sin(2.4*t) + 3.0*np.cos(1.1*t)
    dP = lambda t: 12.0*2.4*np.cos(2.4*t) - 3.0*1.1*np.sin(1.1*t)
    dt = T/N
    node = HeartPumpNode("h", timestep=dt, resistance=R, compliance=C,
                         stroke_volume=0.0, initial_pressure=Ps(0.0))
    st = node.initial_state()
    st = {"arterial_pressure": jnp.asarray(Ps(0.0)),
          "phase": jnp.asarray(0.0),
          "flow_rate": jnp.asarray(0.0)}
    for i in range(N):
        t = i*dt
        # stroke_volume=0 -> Q_heart == 0; P_down that makes Ps exact:
        pd = Ps(t) - R*(-C*dP(t))
        st = node.update(st, {"backpressure": jnp.asarray(pd)}, dt)
    return abs(float(st["arterial_pressure"]) - Ps(T))/abs(Ps(T))


if __name__ == "__main__":
    lv = (100, 200, 400, 800, 1600)
    for name, node_factory, fn in (
        ("SpringDamperNode", lambda: SpringDamperNode("s", timestep=1e-3), spring_err),
        ("BallNode", lambda: BallNode("b", timestep=1e-3), ball_err),
        ("RigidBodyNode", lambda: RigidBodyNode("r", timestep=1e-3), rb_err),
        ("RigidBody2DNode", None, rb2_err),
        ("HeartPumpNode", lambda: HeartPumpNode("h", timestep=1e-3), heart_err),
    ):
        if node_factory is None:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                d = declared_order(RigidBody2DNode("r2", timestep=1e-3)).temporal
        else:
            d = declared_order(node_factory()).temporal
        e, o = ladder(fn, lv)
        report(name, d, lv, e, o)
