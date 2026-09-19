"""F3: fori vs ift across the configuration space (forward state + diagnostics)."""
import sys, warnings, itertools
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/benchmarks/results/audit_040_final/coupling")
warnings.simplefilter("ignore")
from harness import graph
import numpy as np

ACCELS = ["none", "aitken", "fixed", "iqn-ils", "iqn-imvj"]
NORMS  = ["l2", "mixed", "interface"]
MODES  = ["gauss-seidel", "jacobi"]
CAPS   = [2, 3, 5, 10]

rows = []
for acc, norm, mode, cap, intl in itertools.product(
        ACCELS, NORMS, MODES, CAPS, [False, True]):
    kw = dict(acceleration=acc, convergence_norm=norm, iteration_mode=mode,
              max_iterations=cap, int_leaf=intl, b=0.5)
    if acc == "fixed":
        kw["relaxation"] = 0.7
    if acc == "iqn-imvj":
        kw["jacobian_reuse"] = 1
    if norm == "l2":
        kw["tolerance"] = 1e-6
    else:
        kw["atol"], kw["rtol"] = 1e-10, 1e-6
    out = {}
    for solver in ("fori", "ift"):
        try:
            gm = graph(solver, **kw)
            gm.step()
            st = gm.get_node_state("flow")
            d = gm.coupling_diagnostics()["flow+struct"]
            out[solver] = (float(st["tau"]),
                           int(st["i_step"]) if intl else None,
                           d["iterations"], float(d["residual"]),
                           bool(d["converged"]))
        except Exception as e:
            out[solver] = ("ERR", type(e).__name__, str(e)[:60], None, None)
    if out["fori"] != out["ift"]:
        rows.append((acc, norm, mode, cap, intl, out["fori"], out["ift"]))

print(f"{len(rows)} disagreeing configurations out of "
      f"{len(ACCELS)*len(NORMS)*len(MODES)*len(CAPS)*2}")
seen = set()
for acc, norm, mode, cap, intl, f, i in rows:
    # classify
    tags = []
    if isinstance(f[0], float) and isinstance(i[0], float):
        if abs(f[0]-i[0]) > 1e-6*max(1.0, abs(f[0])): tags.append("TAU")
        if f[1] != i[1]: tags.append("INT_LEAF")
        if f[2] != i[2]: tags.append("ITERS")
        if abs(f[3]-i[3]) > 1e-9: tags.append("RESID")
        if f[4] != i[4]: tags.append("CONVERGED")
    else:
        tags.append("EXCEPTION")
    key = tuple(tags)
    print(f"  {acc:<9} {norm:<10} {mode:<13} cap={cap:<3} int={int(intl)} "
          f"{'+'.join(tags):<22} fori={f} ift={i}")
