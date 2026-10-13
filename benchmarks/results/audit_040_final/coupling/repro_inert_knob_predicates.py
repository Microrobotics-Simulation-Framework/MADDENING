"""F4: liveness predicates vs the actual read sites."""
import sys, warnings
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/benchmarks/results/audit_040_final/coupling")
from harness import graph, Flow, Struct
import numpy as np
from maddening.core.graph_manager import GraphManager


def warns(**kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gm = graph("ift", **kw)
        return gm, [str(x.message).split(".")[0] for x in w
                    if issubclass(x.category, UserWarning)
                    and not issubclass(x.category, DeprecationWarning)]


print("=== (a) subcycling=True but uniform timesteps: use_subcycling -> False")
print("    read site: graph_manager.py:1069-1083, 1314")
gm, w = warns(subcycling=True, waveform_iterations=5,
              boundary_interpolation="quadratic", max_iterations=6)
print("    warnings:", w or "NONE")
# Prove inertness: change waveform_iterations, see if anything moves
res = {}
for n in (1, 5):
    g = graph("ift", subcycling=True, waveform_iterations=n,
              max_iterations=6, tolerance=1e-3, b=0.9)
    g.step()
    res[n] = float(g.get_node_state("flow")["tau"])
print(f"    waveform_iterations=1 -> tau={res[1]!r}; =5 -> tau={res[5]!r} "
      f"(identical: {res[1]==res[5]})")

print()
print("=== (b) max_iterations=1 short-circuits every acceleration knob")
print("    read site: graph_manager.py:1326 (`if max_iters <= 1: return`)")
gm, w = warns(max_iterations=1, acceleration="fixed", relaxation=0.3)
print("    warnings for relaxation=0.3 @ acceleration='fixed', cap=1:", w or "NONE")
a = graph("ift", max_iterations=1, acceleration="fixed", relaxation=0.3, b=0.9)
b = graph("ift", max_iterations=1, acceleration="fixed", relaxation=1.9, b=0.9)
a.step(); b.step()
print(f"    relaxation=0.3 -> tau={float(a.get_node_state('flow')['tau'])!r}; "
      f"1.9 -> tau={float(b.get_node_state('flow')['tau'])!r}")

print()
print("=== (c) atol IS read under convergence_norm='l2' (see r1_atol_l2.py)")
print("    read site: graph_manager.py:1266  coupling_residual_l2(..., group.atol)")

print()
print("=== (d) rtol is genuinely inert under l2 (hard-coded 1.0 at")
print("    acceleration.py:120) -- the rule pairs it with atol anyway")
