"""Independent test of every numerical claim in integrators.py's docstrings.

Claims under test (module docstring table + MADD-ANO-014):
  frozen u:      euler 1.02   heun 1.02   rk4 1.02
  stage-time u:  euler 1.02   heun 2.00   rk4 4.00
  and: under the frozen input rk4 is ~1.5x LESS accurate than euler.
Also tests the module docstring's stage-time recipe (time as a state
field with derivative 1) and the Butcher nodes it asserts.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from maddening.core.simulation.integrators import euler_step, heun_step, rk4_step

T_END = 0.75
# manufactured: x*(t) = sin(2 pi t) exact for dx/dt = -x + u(t)
xstar = lambda t: jnp.sin(2*jnp.pi*t)
u_of = lambda t: 2*jnp.pi*jnp.cos(2*jnp.pi*t) + jnp.sin(2*jnp.pi*t)

STEPPERS = {"euler": euler_step, "heun": heun_step, "rk4": rk4_step}


def run_frozen(step, n):
    dt = T_END/n
    def f(s, b):
        return {"x": -s["x"] + b["u"]}
    s = {"x": jnp.asarray(0.0)}
    for i in range(n):
        s = step(f, s, {"u": u_of(jnp.asarray(i*dt))}, dt)
    return abs(float(s["x"]) - float(xstar(jnp.asarray(T_END))))


def run_constant(step, n):
    """u held genuinely constant -> autonomous problem; full order expected."""
    dt = T_END/n
    uc = jnp.asarray(1.3)
    def f(s, b):
        return {"x": -s["x"] + b["u"]}
    s = {"x": jnp.asarray(0.0)}
    for _ in range(n):
        s = step(f, s, {"u": uc}, dt)
    # exact: x(t) = 1.3*(1-exp(-t))
    return abs(float(s["x"]) - 1.3*(1-np.exp(-T_END)))


def run_stage_time(step, n):
    dt = T_END/n
    def f(s, _b):
        return {"x": -s["x"] + u_of(s["t"]), "t": jnp.asarray(1.0)}
    s = {"x": jnp.asarray(0.0), "t": jnp.asarray(0.0)}
    for _ in range(n):
        s = step(f, s, {}, dt)
    return abs(float(s["x"]) - float(xstar(jnp.asarray(T_END))))


def orders(fn, step, levels):
    e = [fn(step, n) for n in levels]
    o = [np.log(e[i]/e[i+1])/np.log(levels[i+1]/levels[i]) for i in range(len(e)-1)]
    return e, o


levels = (10, 20, 40, 80, 160)
print("problem: dx/dt = -x + u(t), u sinusoidal, T=0.75, float64\n")
for label, fn in (("frozen (u read once per step)", run_frozen),
                  ("constant u (autonomous)", run_constant),
                  ("stage-time u (docstring recipe)", run_stage_time)):
    print(f"== {label} ==")
    for name, step in STEPPERS.items():
        e, o = orders(fn, step, levels)
        print(f"  {name:6s} order(finest pair) = {o[-1]:6.3f}   orders={[f'{x:.3f}' for x in o]}")
    print()

# the accuracy-inversion claim: rk4 ~1.5x less accurate than euler, frozen
print("== frozen-input accuracy, euler vs rk4 ==")
for n in (80, 160, 320):
    dt = T_END/n
    ee = run_frozen(euler_step, n)
    er = run_frozen(rk4_step, n)
    print(f"  n={n:4d} dt={dt:.5f}  euler={ee:.4e}  rk4={er:.4e}  rk4/euler={er/ee:.3f}")

# Butcher-node claim: a state field with derivative 1 arrives holding t + c_i*dt
print("\n== Butcher nodes seen by the stages (docstring claim) ==")
for name, step, expect in (("euler", euler_step, (0.0,)),
                           ("heun", heun_step, (0.0, 1.0)),
                           ("rk4", rk4_step, (0.0, 0.5, 0.5, 1.0))):
    seen = []
    def f(s, _b):
        seen.append(float(s["t"]))
        return {"t": jnp.asarray(1.0)}
    step(f, {"t": jnp.asarray(0.0)}, {}, 1.0)
    print(f"  {name:6s} seen={tuple(seen)}  expected={expect}  match={tuple(seen)==expect}")
