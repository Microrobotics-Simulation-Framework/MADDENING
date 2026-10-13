"""MADD-ANO-005 names coupling_diagnostics keys that 0.4.0 deprecated.

The registry entry's description, workaround and residual_risk all tell the
reader to read ``bound_valid`` / ``gradient_error_bound``.  0.4.0 renamed both
(CHANGELOG "Changed" and "Deprecated").
"""
import os, warnings, importlib.util
os.environ.setdefault("JAX_PLATFORMS", "cpu")
spec = importlib.util.spec_from_file_location(
    "cbe", "tests/core/test_coupling_error_bound.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

gm = m._contracting_graph(tolerance=1e-4)
gm.compile(); gm.step()
d = gm.coupling_diagnostics()["a+b"]
print("keys actually reported:", sorted(d.keys()))
for old in ("bound_valid", "gradient_error_bound"):
    print(f"  {old!r} in keys(): {old in d.keys()}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v = d[old]
        print(f"  d[{old!r}] -> {v}   warnings: "
              f"{[f'{x.category.__name__}: {str(x.message)[:90]}' for x in w]}")
