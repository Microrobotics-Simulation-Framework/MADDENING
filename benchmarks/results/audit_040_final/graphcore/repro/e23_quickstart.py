"""The documented 'Differentiable Everything' snippet from
docs/user_guide/quickstart.md, verbatim, then one ordinary step afterwards."""
import jax, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode

gm = GraphManager()
gm.add_node(TableNode("table", 0.01, position=0.0))
gm.add_node(BallNode("ball", 0.01, initial_position=5.0))
gm.add_edge("table", "ball", "position", "table_position")
gm.compile()

def loss(initial_velocity):
    gm.set_node_state("ball", {"position": jnp.array(5.0),
                               "velocity": initial_velocity})
    state = gm.run_scan(n_steps=100)
    return state["ball"]["position"]

grad_fn = jax.grad(loss)
print("d(final_pos)/d(init_vel) =", grad_fn(jnp.array(0.0)))

# the graph the user still holds
print("gm._state['ball']['position'] is now a", type(gm._state['ball']['position']).__name__)
for label, call in (("gm.step()", lambda: gm.step()),
                    ("gm.run_scan(1)", lambda: gm.run_scan(1)),
                    ("gm.save_state('/tmp/x.npz')", lambda: gm.save_state("/tmp/x.npz"))):
    try:
        call(); print(f"  {label}: OK")
    except Exception as e:
        print(f"  {label}: {type(e).__name__}: {str(e).splitlines()[0][:110]}")
# and the documented recovery
try:
    gm.reset_state(); gm.step(); print("  after gm.reset_state(): OK")
except Exception as e:
    print("  after gm.reset_state():", type(e).__name__, str(e).splitlines()[0][:110])
