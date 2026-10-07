"""Where does the diagnostics-on step time go on GPU: the eigenvalue solve
(a host round trip) or the rest of the diagnostics program?  Same pair,
once with eigvals and once with the squaring fallback forced."""
import os, sys, time
os.environ["JAX_PLATFORMS"] = "cuda,cpu"
import jax, numpy as np
from maddening.core.coupling import acceleration as acc
if sys.argv[1] == "squaring":
    acc._EIGVALS_BACKENDS = ("cpu",)
from maddening import GraphManager
from maddening.nodes import SpringDamperNode
gm = GraphManager()
gm.add_node(SpringDamperNode("left", 0.01, stiffness=30.0, damping=2.0, initial_position=1.5))
gm.add_node(SpringDamperNode("right", 0.01, stiffness=30.0, damping=2.0))
gm.add_edge("left", "right", "position", "anchor_position")
gm.add_edge("right", "left", "position", "anchor_position")
gm.add_coupling_group(["left", "right"], max_iterations=8, tolerance=1e-6, solver="ift", diagnostics=True)
gm.compile(); gm.step()
t = time.perf_counter()
for _ in range(100): gm.step()
step = (time.perf_counter() - t) / 100
t = time.perf_counter(); gm.run_scan(2000); scan = (time.perf_counter() - t)
t = time.perf_counter(); gm.run_scan(2000); scan2 = (time.perf_counter() - t) / 2000
print(sys.argv[1], jax.default_backend(), f"step {step*1e3:.3f} ms; run_scan second call {scan2*1e3:.4f} ms per step")
