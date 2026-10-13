"""to_dict() drops an external input's dtype; from_dict() rebuilds it float32."""
import jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode

gm = GraphManager()
gm.add_node(BallNode("ball", 0.01, initial_position=1.0))
gm.add_external_input("ball", "mode", shape=(3,), dtype=jnp.int32)
gm.compile()
print("original ext spec dtype:", gm._external_inputs[0].dtype)
print("original default leaf  :", gm._default_external_inputs()["ball"]["mode"].dtype)

cfg = gm.to_dict()
print("serialised external_inputs:", cfg["external_inputs"])

gm2 = GraphManager.from_dict(cfg, {"BallNode": BallNode})
gm2.compile()
print("reloaded ext spec dtype:", gm2._external_inputs[0].dtype)
print("reloaded default leaf  :", gm2._default_external_inputs()["ball"]["mode"].dtype)
