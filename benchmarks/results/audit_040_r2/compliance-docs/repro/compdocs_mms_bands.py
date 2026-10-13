"""Do the numbers that justify DEFAULT_ORDER_SHORTFALL / _EXCESS reproduce?

src/maddening/testing/mms.py justifies the live acceptance band published in
docs/validation/framework_verification.md ("within [-0.25, +1.0] of the
declared order") with three measured figures:
  * "HeatNode 1.982 against 2" over the finest pair;
  * "the coarsest pair ... sits as much as 0.16 low (1.847)";
  * "the corrected fourth-order stencil measures 5.02 over one pair".
"""
import os, math, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "tests")
import importlib.util
spec = importlib.util.spec_from_file_location("tmms", "tests/verification/test_mms_order.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def ladder(label, fn, levels):
    errs = [fn(n) for n in levels]
    print(f"{label}: levels={levels}")
    for i in range(len(levels) - 1):
        p = math.log(errs[i] / errs[i+1]) / math.log(levels[i+1] / levels[i])
        print(f"    pair {levels[i]:>4}->{levels[i+1]:<4} err {errs[i]:.4e} -> {errs[i+1]:.4e}   observed order = {p:.3f}")
    return errs

ladder("HeatNode stencil_order=2, rod-end BC, Fo=0.4 (MADD-VER-005 ladder)",
       lambda n: m._heat_steady_error(n), (10, 20, 40, 80, 160))
ladder("HeatNode stencil_order=4, Fo=0.3 (the 4th-order ladder)",
       lambda n: m._heat_steady_error(n, stencil_order=4, fourier=0.3), (10, 20, 40, 80, 160))
