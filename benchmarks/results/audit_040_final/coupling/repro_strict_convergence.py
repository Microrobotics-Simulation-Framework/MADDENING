import sys, warnings
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/benchmarks/results/audit_040_final/coupling")
warnings.simplefilter("ignore")
from harness import graph

print("strict_convergence fires on a genuine cap-out:")
gm = graph("ift", b=0.99, max_iterations=3, tolerance=1e-12,
           strict_convergence=True)
try:
    gm.step(); print("   NO ERROR (unexpected)")
except Exception as e:
    print("   raised", type(e).__name__, "-> guard works")

print("ratio_usable at max_iterations=1 (docstring says False):")
gm = graph("ift", b=0.5, max_iterations=1, tolerance=1e-12)
gm.step(); print("  ", gm.coupling_diagnostics()["flow+struct"])

print("strict_convergence does NOT fire on the atol dead-band case:")
gm = graph("ift", b=0.5, scale=1e-9, max_iterations=40, tolerance=1e-12,
           strict_convergence=True)
gm.step()
print("   tau =", float(gm.get_node_state("flow")["tau"]),
      "(exact 2e-09)  diag =", gm.coupling_diagnostics()["flow+struct"])
