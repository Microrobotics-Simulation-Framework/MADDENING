"""Which error does a bogus accelerated_fields entry produce, per acceleration?"""
import os, dataclasses, warnings
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

for accel in ("none", "iqn-ils", "iqn-imvj", "aitken"):
    g = GraphManager()
    g.add_node(SpringDamperNode(name="a", timestep=0.01, stiffness=30.0, damping=2.0))
    g.add_node(SpringDamperNode(name="b", timestep=0.01, stiffness=10.0, damping=1.0))
    g.add_edge("a", "b", "position", "anchor_position")
    g.add_edge("b", "a", "position", "anchor_position")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            g.add_coupling_group(["a", "b"], max_iterations=5, acceleration=accel,
                                 accelerated_fields={"a": ["not_a_field"]})
        except Exception as e:
            print(f"{accel:10} add_coupling_group -> {type(e).__name__}: {str(e)[:90]}")
            continue
        try:
            g.compile()
            print(f"{accel:10} compile() did NOT raise")
        except Exception as e:
            print(f"{accel:10} compile -> {type(e).__name__}: {str(e)[:120]}")
