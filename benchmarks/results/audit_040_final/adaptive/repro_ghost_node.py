"""How far the 'ghost node' left by a failed add_node propagates."""
import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING-wt/audit/adaptive/tests")
import jax
jax.config.update("jax_enable_x64", True)
from maddening.core.graph_manager import GraphManager
from maddening.nodes.adaptive import AdaptiveNodeBlindnessError
from nodes.adaptive._toys import PoissonSineTopKNode

gm = GraphManager()
try:
    gm.add_node(PoissonSineTopKNode("trap", n=64, k=8, theta=0.5))
except AdaptiveNodeBlindnessError:
    pass
gm.add_node(PoissonSineTopKNode("good", n=64, k=32, theta=0.42))
gm.compile()
print("node_names           :", list(gm.node_names))
print("state keys           :", list(gm._state))
print("params['nodes'] keys :", list(gm.params.get("nodes", {})))
out = gm.step()
print("step() returned keys :", list(out) if isinstance(out, dict) else type(out))
for meth, args in (("get_state", ("trap",)), ("get_node", ("trap",)),
                   ("node_state", ("trap",)), ("to_dict", ()), ("save_checkpoint", None)):
    fn = getattr(gm, meth, None)
    if fn is None or args is None:
        continue
    try:
        r = fn(*args)
        print(f"{meth}{args}: ok -> {type(r).__name__}")
    except Exception as e:
        print(f"{meth}{args}: {type(e).__name__}: {str(e)[:140]}")
import tempfile, os
try:
    p = os.path.join(tempfile.mkdtemp(), "ck")
    gm.save_checkpoint(p)
    print("save_checkpoint: ok")
    gm2 = GraphManager()
    gm2.add_node(PoissonSineTopKNode("good", n=64, k=32, theta=0.42))
    gm2.compile(); gm2.load_checkpoint(p)
    print("load_checkpoint into a clean graph: ok")
except Exception as e:
    print("checkpoint:", type(e).__name__, str(e)[:200])
