"""Do the three surfaces agree about what they will and will not serialise?"""
import os, sys, json, tempfile, traceback
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.core.node import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from pxr import Usd
from maddening.usd.serialization import save_graph_to_usd, load_graph_from_usd
from maddening.fmi.tcp_bridge import encode_binary, send_message

def graph(node_name="s1", **kw):
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name=node_name, timestep=0.01, stiffness=30.0,
                                 damping=2.0, initial_position=0.5, **kw))
    gm.compile()
    return gm

def probe(label, gm):
    print(f"\n### {label}")
    # surface 1 -- to_dict
    try:
        d = gm.to_dict()
        json.dumps(d)          # must be strict-JSON dumpable
        print("  to_dict          : OK")
    except Exception as e:
        print(f"  to_dict          : {type(e).__name__}: {str(e)[:110]}")
    # surface 2 -- USD
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "g.usda")
        try:
            stage = Usd.Stage.CreateNew(p)
            save_graph_to_usd(gm, stage)
            stage.Save()
            gm2 = load_graph_from_usd(Usd.Stage.Open(p))
            names = sorted(gm2._nodes)
            d2 = None
            try:
                d2 = gm2.to_dict()
            except Exception as e2:
                d2 = f"to_dict on the RELOADED graph: {type(e2).__name__}"
            print(f"  save_graph_to_usd: OK  (reloaded nodes {names}; {d2 if isinstance(d2,str) else 'reloaded to_dict OK'})")
            txt = open(p).read()
            for tok in ('"NaN"', '"Infinity"', 'NaN', 'Infinity'):
                if tok in txt:
                    print(f"      stage text contains {tok!r}")
                    break
        except Exception as e:
            print(f"  save_graph_to_usd: {type(e).__name__}: {str(e)[:130]}")
    # surface 3 -- FMI wire helpers
    try:
        encode_binary({"op": "set", "tag": list(gm._nodes)[0]}, b"")
        print("  encode_binary    : OK")
    except Exception as e:
        print(f"  encode_binary    : {type(e).__name__}: {str(e)[:110]}")

# A: an ordinary graph
probe("ordinary node name 's1'", graph())
# B: a node named exactly like a token  (MADD-ANO-010 says all three refuse)
probe("node named 'NaN'", graph("NaN"))
probe("node named 'Infinity'", graph("Infinity"))
# C: a non-finite scalar param
g = graph(); g._nodes["s1"].node.params["stiffness"] = float("inf")
probe("non-finite scalar param", g)
# D: a string param equal to a token
g = graph(); g._nodes["s1"].node.params["label"] = "Infinity"
probe("string param 'Infinity'", g)
