"""Reproducer: docs/algorithm_guide/nodes/adaptive_node.md's "Validated
Physical Regimes" row states the gradient-capture warning fires below an
*active fraction* K/n_max ~= 0.1.  The ratio is a function of K alone.

Run from the worktree root:
  PYTHONPATH=$PWD/src:$PWD JAX_PLATFORMS=cpu python <this file>
"""
import warnings
from tests.nodes.adaptive._toys import PoissonSineTopKNode

print(f"{'n_max':>6} {'k':>4} {'K/n_max':>8} {'ratio':>7} {'warns?':>7}  alg-guide predicts")
for n, k in [(256, 4), (256, 8), (256, 16), (256, 32), (64, 4), (64, 8),
             (64, 16), (32, 4), (32, 8), (32, 16)]:
    node = PoissonSineTopKNode(n=n, k=k, blindness_gate=False)
    st = node.initial_state()
    r = float(node.gradient_capture_ratio(st))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PoissonSineTopKNode(n=n, k=k).initial_state()
        warned = any("gradient-capture ratio" in str(x.message) for x in w)
    frac = k / n
    predicted = "warn (frac<0.1)" if frac < 0.1 else "no warn (frac>=0.1)"
    flag = "  <-- CONTRADICTS" if (frac < 0.1) != warned else ""
    print(f"{n:6d} {k:4d} {frac:8.4f} {r:7.3f} {str(warned):>7}  {predicted}{flag}")
