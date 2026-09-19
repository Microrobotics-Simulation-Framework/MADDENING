"""run_sweep on graphs that need _meta (multi-rate / coupled-with-diagnostics)."""
import jax, jax.numpy as jnp, traceback
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode

def try_(tag, fn):
    try:
        out = fn()
        print(f"{tag}: OK ->", jax.tree.map(lambda x: getattr(x,'shape',x), out))
    except Exception as e:
        print(f"{tag}: FAILED {type(e).__name__}: {str(e)[:180]}")

# 1. plain graph
gm = GraphManager()
gm.add_node(BallNode("ball", 0.01, initial_position=1.0))
gm.compile()
init = {"ball": {"position": jnp.array([1.0,2.0,3.0]), "velocity": jnp.zeros(3)}}
try_("plain", lambda: gm.run_sweep(3, init))

# 2. multirate graph (needs _meta['step_count'])
gm2 = GraphManager()
gm2.add_node(TableNode("table", 0.01, position=0.0))
gm2.add_node(BallNode("ball", 0.02, initial_position=1.0))
gm2.add_edge("table","ball","position","table_position")
gm2.compile()
print("multirate:", gm2.is_multirate, "meta:", sorted(gm2._state.get("_meta", {})))
init2 = {"table": {"position": jnp.zeros(3)},
         "ball": {"position": jnp.array([1.0,2.0,3.0]), "velocity": jnp.zeros(3)}}
try_("multirate", lambda: gm2.run_sweep(3, init2))

# 3. coupled with diagnostics (meta has coupling_* keys)
gm3 = GraphManager()
gm3.add_node(SpringDamperNode("a", 0.001, initial_position=0.0))
gm3.add_node(SpringDamperNode("b", 0.001, initial_position=2.0))
gm3.add_edge("a","b","position","anchor_position")
gm3.add_edge("b","a","position","anchor_position")
gm3.add_coupling_group(["a","b"], tolerance=1e-8, max_iterations=10, diagnostics=True)
gm3.compile()
print("coupled meta:", sorted(gm3._state["_meta"]))
init3 = {n: {f: jnp.stack([v, v]) for f, v in gm3._state[n].items()} for n in ("a","b")}
try_("coupled", lambda: gm3.run_sweep(3, init3))
