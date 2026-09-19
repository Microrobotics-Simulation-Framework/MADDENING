import jax, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode

def build():
    gm = GraphManager(); gm.add_node(BallNode("ball", 0.01, initial_position=1.0)); gm.compile(); return gm

# B: python float written into gm.params, repeated step()
gm = build()
gm.params["nodes"]["ball"]["gravity"] = -1.0          # plain python float
for _ in range(3): gm.step()
print("step: trace_count after 3 steps with python-float param:", gm.trace_count)

# E: same via run_adaptive_scan (which bypasses _params_or_default)
gm2 = build()
gm2.params["nodes"]["ball"]["gravity"] = -1.0
for i in range(3):
    gm2.run_adaptive_scan(t_end=0.05, max_steps=8)
    print(f"  run_adaptive_scan call {i}: scan_trace_count={gm2.scan_trace_count}")

# and with a proper array
gm3 = build()
gm3.params["nodes"]["ball"]["gravity"] = jnp.array(-1.0, jnp.float32)
for i in range(3):
    gm3.run_adaptive_scan(t_end=0.05, max_steps=8)
print("  array-typed param: scan_trace_count =", gm3.scan_trace_count)

# repeated run_scan
gm4 = build()
gm4.params["nodes"]["ball"]["gravity"] = -1.0
for i in range(3): gm4.run_scan(5)
print("run_scan with python-float param: scan_trace_count =", gm4.scan_trace_count)
