"""A calibrated parameter reaches update() -- and, since the fix, every
derivatives()-based path too (MADD-ANO-018, resolved in 0.4.0).

At 0.4.0.dev0, SimulationNode.derivatives(state, boundary_inputs) and
implicit_residual(state_new, state_old, boundary_inputs, dt) took no
``params``, so integrate_node() and implicit_euler_step() -- both public,
both documented -- integrated the CONSTRUCTOR's constants whatever the
graph injected, while update() honoured them: the two paths of one node
object disagreed by the whole calibration (-1.0 against -4.0 below).

The fix threads ``*, params=None`` through both methods and both entry
points with the same ``{**self.params, **params}`` rule as update().
This script now records both facts and exits non-zero if either stops
holding:

  * without params, every path still integrates the constructor's
    constants (backward compatibility);
  * with params, every path agrees with update(params=...) -- and the
    implicit solve agrees with a node *built* with the calibrated value.
"""
import functools
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.heat import HeatNode
from maddening.core.simulation.integrators import integrate_node, euler_step
from maddening.core.simulation.implicit import implicit_euler_step

failures = []


def check(label, got, want, rel=1e-9):
    ok = abs(got - want) <= rel * max(1.0, abs(want))
    print(f"  {label:<44s} -> {got:.10f}   {'ok' if ok else 'MISMATCH (want %.10f)' % want}")
    if not ok:
        failures.append(label)


dt = 0.01
node = SpringDamperNode("s", timestep=dt, stiffness=100.0, damping=1.0, mass=1.0,
                        rest_length=1.0)
st = {"position": jnp.asarray(2.0), "velocity": jnp.asarray(0.0)}
fitted = {"stiffness": jnp.asarray(400.0)}   # what a calibration produced
v = lambda s: float(s["velocity"])

print("SpringDamperNode, constructor stiffness=100, calibrated stiffness=400")
print(" no params: the constructor's constants (unchanged behaviour)")
check("update()", v(node.update(st, {}, dt)), -1.0)
check("euler_step(node.derivatives)", v(euler_step(node.derivatives, st, {}, dt)), -1.0)
check("integrate_node(method=euler)", v(integrate_node(node, st, {}, dt, method="euler")), -1.0)
implicit_100 = v(implicit_euler_step(node.implicit_residual, st, {}, dt)[0])
check("implicit_euler_step", implicit_100, -0.9803921103845444, rel=1e-6)

print(" params=fitted: the calibrated constants, on every path")
check("update(params=fitted)", v(node.update(st, {}, dt, params=fitted)), -4.0)
check("euler_step(partial(derivatives, params=))",
      v(euler_step(functools.partial(node.derivatives, params=fitted), st, {}, dt)), -4.0)
check("integrate_node(method=euler, params=)",
      v(integrate_node(node, st, {}, dt, method="euler", params=fitted)), -4.0)
ref = SpringDamperNode("r", timestep=dt, stiffness=400.0, damping=1.0, mass=1.0,
                       rest_length=1.0)
implicit_ref = v(implicit_euler_step(ref.implicit_residual, st, {}, dt)[0])
check("implicit_euler_step(params=) == node built with k=400",
      v(implicit_euler_step(node.implicit_residual, st, {}, dt, params=fitted)[0]),
      implicit_ref, rel=1e-9)
if abs(implicit_ref - implicit_100) < 1.0:
    failures.append("implicit reference does not move with the calibration")

print("\nHeatNode, constructor alpha=0.01, calibrated alpha=0.5")
h = HeatNode("h", timestep=1e-4, n_cells=8, length=1.0, thermal_diffusivity=0.01)
T = {"temperature": jnp.asarray([0., 1., 4., 9., 9., 4., 1., 0.])}
bi = {"left_temperature": jnp.asarray(0.0), "right_temperature": jnp.asarray(0.0)}
fp = {"thermal_diffusivity": jnp.asarray(0.5)}
via_update = float(h.update(T, bi, 1e-4, params=fp)["temperature"][3])
check("update(params=fitted)[3]", via_update, 8.984)
check("integrate_node(euler, params=fitted)[3]",
      float(integrate_node(h, T, bi, 1e-4, method="euler", params=fp)["temperature"][3]),
      via_update)
check("update(no params)[3]", float(h.update(T, bi, 1e-4)["temperature"][3]), 8.99968)
check("integrate_node(euler, no params)[3]",
      float(integrate_node(h, T, bi, 1e-4, method="euler")["temperature"][3]), 8.99968)

print("\nThe refusal that replaced the silence")


class Legacy(SpringDamperNode):
    def derivatives(self, state, boundary_inputs):      # predates the keyword
        return super().derivatives(state, boundary_inputs)


legacy = Legacy("l", timestep=dt, stiffness=100.0, damping=1.0, mass=1.0, rest_length=1.0)
check("legacy override, no params", v(integrate_node(legacy, st, {}, dt, method="euler")), -1.0)
try:
    integrate_node(legacy, st, {}, dt, method="euler", params=fitted)
    print("  legacy override, params=fitted                -> NO REFUSAL")
    failures.append("legacy override accepted params silently")
except ValueError as e:
    print(f"  legacy override, params=fitted                -> ValueError: {str(e)[:70]}...")

if failures:
    print("\nFAIL:", failures)
    sys.exit(1)
print("\nOK: every path honours params, none drops it, none changed without it.")
