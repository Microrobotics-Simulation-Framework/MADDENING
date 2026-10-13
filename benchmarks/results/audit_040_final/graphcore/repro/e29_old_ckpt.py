"""0.4.0 added _meta['coupling_*_amplification'] but did not bump
CHECKPOINT_SCHEMA_VERSION.  load_state replaces _meta wholesale, so a
checkpoint written by 0.3.x (which has no such key) resumes into a graph whose
step writes one -- and the scan carry no longer matches."""
import numpy as np, jax.numpy as jnp, tempfile, os
from maddening.core.graph_manager import GraphManager
from maddening.core.simulation.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION, save_state_with_manifest, load_state_with_manifest,
)
from maddening.nodes.spring import SpringDamperNode
TMP = tempfile.mkdtemp()

def build():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.001, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.001, initial_position=2.0))
    gm.add_edge("a","b","position","anchor_position")
    gm.add_edge("b","a","position","anchor_position")
    gm.add_coupling_group(["a","b"], tolerance=1e-10, max_iterations=20,
                          diagnostics=True)
    gm.compile(); return gm

print("CHECKPOINT_SCHEMA_VERSION =", CHECKPOINT_SCHEMA_VERSION)
gm = build(); gm.run(5)
p = os.path.join(TMP, "new.npz")
save_state_with_manifest(gm, p)

# Rewrite the archive the way maddening 0.3.x would have written it: same
# schema version, same manifest, no amplification key.
d = np.load(p)
old = {k: d[k] for k in d.files if not k.endswith("_amplification")}
print("0.3-style keys:", sorted(k for k in old if k.startswith("_meta")))
p_old = os.path.join(TMP, "old.npz")
np.savez(p_old, **old)
from maddening.core.simulation.checkpoint import write_manifest
write_manifest(p_old)

g2 = build()
m = load_state_with_manifest(g2, p_old)         # manifest says v1 -> accepted
print("manifest accepted, schema_version =", m["schema_version"])
print("_meta after load:", sorted(g2._state["_meta"]))
for label, call in (("step()", lambda: g2.step()),
                    ("run_scan(3)", lambda: g2.run_scan(3)),
                    ("coupling_diagnostics()", lambda: g2.coupling_diagnostics())):
    try:
        r = call(); print(f"  {label}: OK", (r if label.endswith("()") and "diag" in label else ""))
    except Exception as e:
        print(f"  {label}: {type(e).__name__}: {str(e).splitlines()[0][:130]}")
