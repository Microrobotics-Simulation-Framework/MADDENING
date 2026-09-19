"""Resume a checkpoint into a graph whose _meta has MORE keys than the
checkpoint's: load_state replaces _meta wholesale, so the extra keys vanish."""
import numpy as np, jax.numpy as jnp, tempfile, os
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
TMP = tempfile.mkdtemp()

def build(**gkw):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.001, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.001, initial_position=2.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], tolerance=1e-8, max_iterations=20,
                          diagnostics=True, **gkw)
    gm.compile()
    return gm

plain = build()
plain.run(5)
print("plain meta:", sorted(plain._state["_meta"]))
plain.save_state(os.path.join(TMP, "p.npz"))

# User now turns the predictor on and resumes from that checkpoint.
pred = build(predictor="linear")
print("pred meta :", sorted(pred._state["_meta"]))
pred.load_state(os.path.join(TMP, "p.npz"))
print("after load:", sorted(pred._state["_meta"]))
try:
    pred.step()
    print("step after resume: OK")
except Exception as e:
    print("step after resume FAILED:", type(e).__name__, str(e)[:200])

for label, call in (("run_scan(3)", lambda: pred.run_scan(3)),
                    ("run_scan_with_history(3)", lambda: pred.run_scan_with_history(3))):
    try:
        call(); print(f"{label} after resume: OK; meta now {sorted(pred._state['_meta'])}")
    except Exception as e:
        print(f"{label} after resume: {type(e).__name__}: {str(e).splitlines()[0][:130]}")
