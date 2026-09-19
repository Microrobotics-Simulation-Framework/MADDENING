import jax, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
gm = GraphManager()
gm.add_node(SpringDamperNode("a", 0.001, initial_position=0.0))
gm.add_node(SpringDamperNode("b", 0.001, initial_position=2.0))
gm.add_edge("a","b","position","anchor_position")
gm.add_edge("b","a","position","anchor_position")
gm.add_coupling_group(["a","b"], tolerance=1e-10, max_iterations=20, diagnostics=True)
gm.compile()
def loss(p): return gm.run_scan(3, params=p)["a"]["position"]
print("grad:", float(jax.grad(loss)(jax.tree.map(lambda x: x, gm.params))["nodes"]["a"]["stiffness"]))
print("_meta leaf type:", type(gm._state["_meta"]["coupling_a+b_residual"]).__name__)
for label, call in (("reset_state()+step()", lambda: (gm.reset_state(), gm.step())),
                    ("compile()+step()", lambda: (gm.compile(), gm.step()))):
    try:
        call(); print(f"  {label}: OK")
    except Exception as e:
        print(f"  {label}: {type(e).__name__}: {str(e).splitlines()[0][:110]}")
