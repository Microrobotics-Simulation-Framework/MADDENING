"""load_state replaces _meta wholesale. Save from graph A, load into graph B
whose _meta has MORE keys (a coupling group was added)."""
import numpy as np, jax, jax.numpy as jnp, tempfile, os, traceback
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
TMP = tempfile.mkdtemp()

def build(coupled):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.001, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.001, initial_position=2.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    if coupled:
        gm.add_coupling_group(["a", "b"], tolerance=1e-8, max_iterations=20,
                              acceleration="iqn-ils", diagnostics=True)
    gm.compile()
    return gm

# A: uncoupled (back-edge), B: coupled with IQN history in _meta
a = build(False); a.run(5)
print("A meta keys:", sorted(a._state.get("_meta", {})))
a.save_state(os.path.join(TMP, "a.npz"))

b = build(True); b.run(5)
print("B meta keys:", sorted(b._state.get("_meta", {})))
try:
    b.load_state(os.path.join(TMP, "a.npz"))
    print("load OK; B meta keys now:", sorted(b._state.get("_meta", {})))
    b.step()
    print("step after load OK")
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:300])

# reverse: save coupled, load into uncoupled
b2 = build(True); b2.run(5); b2.save_state(os.path.join(TMP,"b.npz"))
a2 = build(False)
try:
    a2.load_state(os.path.join(TMP,"b.npz"))
    a2.step()
    print("reverse direction OK; meta keys:", sorted(a2._state.get("_meta", {})))
except Exception as e:
    print("reverse FAILED:", type(e).__name__, str(e)[:200])

# dtype preservation of _meta
c = build(True); c.run(3)
before = {k: (np.asarray(v).dtype, np.asarray(v).shape) for k,v in c._state["_meta"].items()}
c.save_state(os.path.join(TMP,"c.npz")); c.load_state(os.path.join(TMP,"c.npz"))
after = {k: (np.asarray(v).dtype, np.asarray(v).shape) for k,v in c._state["_meta"].items()}
diffs = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
print("meta dtype/shape drift on round trip:", diffs or "none")
