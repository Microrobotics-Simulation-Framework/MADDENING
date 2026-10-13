import sys, warnings
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/benchmarks/results/audit_040_final/coupling")
warnings.simplefilter("ignore")
from harness import graph
import numpy as np

print("### A. `fixed` over-relaxation: the bound sums un-relaxed steps")
print("    x_{k+1} = x_k + w*(F(x_k)-x_k); the reported residual is")
print("    ||F(x)-x||, but the actual step is w times longer, so the")
print("    geometric series is short by a factor w.  b=0.9 => x*=10.")
B = 0.9
EXACT = 1.0 / (1.0 - B)
print(f"    {'omega':>6} {'iters':>5} {'residual':>11} {'err_est':>11} "
      f"{'TRUE dist':>11} {'est/true':>9} conv")
for w in (0.5, 1.0, 1.5, 1.8, 1.95):
    gm = graph("ift", b=B, max_iterations=80, tolerance=1e-4,
               acceleration="fixed", relaxation=w)
    gm.step()
    d = gm.coupling_diagnostics()["flow+struct"]
    tau = float(gm.get_node_state("flow")["tau"])
    disp = float(gm.get_node_state("struct")["disp"])
    # the group's own scaled-l2 distance to the exact fixed point
    def sl2(a, b_):
        ref = max(abs(a), abs(b_))
        return 0.0 if ref <= 1e-8 else ((a - b_) / ref) ** 2
    true_d = (sl2(tau, EXACT) + sl2(disp, B * EXACT)) ** 0.5
    print(f"    {w:6.2f} {d['iterations']:5d} {d['residual']:11.4e} "
          f"{d['error_estimate']:11.4e} {true_d:11.4e} "
          f"{d['error_estimate']/true_d if true_d else float('nan'):9.4f} "
          f"{d['converged']}")

print()
print("### B. every acceleration reaches the same fixed point (tight tol)")
for acc in ("none", "aitken", "fixed", "iqn-ils", "iqn-imvj"):
    kw = {}
    if acc == "fixed": kw["relaxation"] = 0.8
    if acc == "iqn-imvj": kw["jacobian_reuse"] = 2
    for mode in ("gauss-seidel", "jacobi"):
        gm = graph("ift", b=B, max_iterations=200, tolerance=1e-9,
                   acceleration=acc, iteration_mode=mode, **kw)
        gm.step()
        d = gm.coupling_diagnostics()["flow+struct"]
        tau = float(gm.get_node_state("flow")["tau"])
        print(f"    {acc:<9} {mode:<13} tau={tau!r:<20} "
              f"rel_err={abs(tau-EXACT)/EXACT:9.2e} iters={d['iterations']:<4} "
              f"conv={d['converged']}")

print()
print("### C. `iterations` at the cap: fori reports cap, ift reports cap-1")
for acc in ("none", "aitken", "fixed", "iqn-ils", "iqn-imvj"):
    kw = {}
    if acc == "fixed": kw["relaxation"] = 0.8
    if acc == "iqn-imvj": kw["jacobian_reuse"] = 1
    out = {}
    for s in ("fori", "ift"):
        gm = graph(s, b=0.99, max_iterations=6, tolerance=1e-12,
                   acceleration=acc, **kw)
        gm.step()
        d = gm.coupling_diagnostics()["flow+struct"]
        out[s] = (d["iterations"], d["converged"])
    print(f"    cap=6 {acc:<9} fori={out['fori']}  ift={out['ift']}")
