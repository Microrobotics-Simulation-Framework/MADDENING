"""OUT OF SURFACE (surrogates/replace/_core.py): replace_node re-adds the saved
edges positionally and drops additive / units / mapping."""
import jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode
from maddening.surrogates.replace import replace_node
from maddening.surrogates.node import SurrogateNode

gm = GraphManager()
gm.add_node(TableNode("t1", 0.01, position=1.0))
gm.add_node(TableNode("t2", 0.01, position=2.0))
gm.add_node(SpringDamperNode("s", 0.01, initial_position=0.0))
gm.add_edge("t1", "s", "position", "anchor_position", additive=True)
gm.add_edge("t2", "s", "position", "anchor_position", additive=True)
gm.compile()
print("before: additive flags =", [e.additive for e in gm._edges])
print("before: anchor seen by s =", gm.resolve_boundary_inputs("s"))

# swap t1 for a surrogate with the same interface
class Fake(SurrogateNode):
    pass
import inspect
print("SurrogateNode ctor:", str(inspect.signature(SurrogateNode.__init__))[:200])

# replace_node only checks the name, so a plain node stands in for the surrogate
replace_node(gm, "t1", TableNode("t1", 0.01, position=1.0))
gm.compile()
print("after : additive flags =", [e.additive for e in gm._edges])
print("after : anchor seen by s =", gm.resolve_boundary_inputs("s"))
