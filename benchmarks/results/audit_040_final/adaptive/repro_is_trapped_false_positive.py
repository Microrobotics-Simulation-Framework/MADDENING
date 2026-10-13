"""Repro: is_trapped_at() reports a Palais symmetry trap at any stationary
point of the frozen objective -- i.e. exactly where an optimiser converges.

The guide recommends running it between optimiser steps above D_threshold
parameters (algorithm guide, 'Validated Physical Regimes'); it will fire on
success.
"""
import sys, warnings
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING-wt/audit/adaptive/tests")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
from nodes.adaptive._toys import PoissonSineTopKNode
from maddening.nodes.adaptive import AdaptiveNodeBlindnessError

node = PoissonSineTopKNode(n=256, k=64, theta=0.30, sigma=0.04,
                           sensor_x=1.0/3.0, blindness_gate=False)
st = node.initial_state()

def gfroz(t):
    def J(x):
        out = node.update(st, {}, 1.0, params={"theta": x})
        return node.objective(out, {**node.params, "theta": x})
    return float(jax.grad(J)(jnp.asarray(t)))

# the sensor reading peaks when the source sits on the sensor: bracket it
lo, hi = 0.30, 0.37
print(f"dJ/dtheta at {lo} = {gfroz(lo): .6e}   at {hi} = {gfroz(hi): .6e}")
t_star = brentq(gfroz, lo, hi, xtol=1e-12)
print(f"stationary point of the FROZEN objective: theta* = {t_star:.12f}")
print(f"   dJ_frozen/dtheta(theta*) = {gfroz(t_star): .3e}")
print(f"   the problem has no symmetry fixing theta* = {t_star:.6f} "
      f"(the only reflection fixed point of this toy is theta = 0.5)")

p = {"theta": jnp.asarray(t_star)}
print(f"   is_trapped_at(theta*)       = {node.is_trapped_at(st, p)}")
print(f"   gradient_capture_ratio      = {node.gradient_capture_ratio(st, p):.6f}")

print("\nwhat check_gradient_capture does there (on_blind='warn', the default):")
n2 = PoissonSineTopKNode(n=256, k=64, theta=t_star, sigma=0.04, sensor_x=1.0/3.0)
try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = n2.check_gradient_capture()
    print(f"   returned ratio {r}; warnings: {[str(x.message)[:90] for x in w]}")
except AdaptiveNodeBlindnessError as e:
    print("   AdaptiveNodeBlindnessError:", str(e)[:420])
