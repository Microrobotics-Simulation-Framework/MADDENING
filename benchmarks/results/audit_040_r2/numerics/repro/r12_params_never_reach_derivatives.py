"""A calibrated parameter reaches update() and cannot reach derivatives().

SimulationNode.derivatives(state, boundary_inputs) and
implicit_residual(state_new, state_old, boundary_inputs, dt) take no
``params``, so integrate_node() and implicit_euler_step() -- both public,
both documented -- integrate the CONSTRUCTOR's constants whatever the
graph injected.  update() honours the injected params, so the two paths
of the same node disagree by the whole calibration.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.heat import HeatNode
from maddening.core.simulation.integrators import integrate_node, euler_step
from maddening.core.simulation.implicit import implicit_euler_step

dt = 0.01
node = SpringDamperNode("s", timestep=dt, stiffness=100.0, damping=1.0, mass=1.0,
                        rest_length=1.0)
st = {"position": jnp.asarray(2.0), "velocity": jnp.asarray(0.0)}
fitted = {"stiffness": jnp.asarray(400.0)}   # what a calibration produced

print("SpringDamperNode, constructor stiffness=100, calibrated stiffness=400")
print(f"  update(params=fitted)       velocity -> "
      f"{float(node.update(st, {}, dt, params=fitted)['velocity']):.10f}")
print(f"  update(no params)           velocity -> "
      f"{float(node.update(st, {}, dt)['velocity']):.10f}")
print(f"  euler_step(node.derivatives) velocity -> "
      f"{float(euler_step(node.derivatives, st, {}, dt)['velocity']):.10f}")
print(f"  integrate_node(method=euler) velocity -> "
      f"{float(integrate_node(node, st, {}, dt, method='euler')['velocity']):.10f}")
new, res = implicit_euler_step(node.implicit_residual, st, {}, dt)
print(f"  implicit_euler_step          velocity -> {float(new['velocity']):.10f}")
print("  -> integrate_node / implicit_euler_step have NO parameter by which the")
print("     calibrated stiffness could be supplied; they use 100, not 400.")

print("\nHeatNode, constructor alpha=0.01, calibrated alpha=0.5")
h = HeatNode("h", timestep=1e-4, n_cells=8, length=1.0, thermal_diffusivity=0.01)
T = {"temperature": jnp.asarray([0., 1., 4., 9., 9., 4., 1., 0.])}
bi = {"left_temperature": jnp.asarray(0.0), "right_temperature": jnp.asarray(0.0)}
fp = {"thermal_diffusivity": jnp.asarray(0.5)}
print(f"  update(params=fitted)[3]  -> "
      f"{float(h.update(T, bi, 1e-4, params=fp)['temperature'][3]):.10f}")
print(f"  integrate_node(euler)[3]  -> "
      f"{float(integrate_node(h, T, bi, 1e-4, method='euler')['temperature'][3]):.10f}")
print(f"  update(no params)[3]      -> "
      f"{float(h.update(T, bi, 1e-4)['temperature'][3]):.10f}")
