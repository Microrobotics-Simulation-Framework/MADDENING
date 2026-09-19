"""Live (calibrated) params across a recompile, and the drop warning."""
import warnings, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode

gm = GraphManager()
gm.add_node(BallNode("ball", 0.01, initial_position=1.0))
gm.compile()
gm.params["nodes"]["ball"]["gravity"] = jnp.array(-3.0, jnp.float32)

# a legitimate structural change that dirties the graph
gm.add_node(TableNode("table", 0.01, position=0.0))
gm.add_edge("table", "ball", "position", "table_position")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    gm.compile()
    print("warnings:", [str(x.message)[:70] for x in w])
print("gravity survived recompile:", gm.params["nodes"]["ball"]["gravity"])

# remove the node the leaf belongs to
gm.params["nodes"]["table"]["position"] = jnp.array(9.0, jnp.float32)
gm.remove_node("table")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    gm.compile()
    print("warnings on remove:", [str(x.message)[:120] for x in w])
print("params now:", gm.params)
