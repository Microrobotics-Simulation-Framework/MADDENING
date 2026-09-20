"""Is GraphManager.to_dict()'s 'the result is JSON-valid' promise true?"""
import os, json, math, tempfile
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.core.node import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.heat import HeatNode
from maddening.serialization.json_codec import dumps as jdumps, loads as jloads

def strict(text):
    def refuse(t): raise ValueError(f"bare token {t!r}")
    return json.loads(text, parse_constant=refuse)

def show(label, fn):
    try:
        print(f"[OK  ] {label}: {fn()}")
    except Exception as e:
        print(f"[FAIL] {label}: {type(e).__name__}: {str(e)[:170]}")

# --- 1. array param with a NaN in it
gm = GraphManager()
gm.add_node(HeatNode(name="h", timestep=0.01, n_cells=4))
gm.compile()
gm._nodes["h"].node.params["initial_temperature"] = np.array([1.0, np.nan, 3.0, 4.0])
show("to_dict with an ndarray param holding NaN",
     lambda: type(gm.to_dict()["nodes"][0]["params"]["initial_temperature"]).__name__)
show("  ... json.dumps of that to_dict", lambda: jdumps(gm.to_dict())[:80])

# --- 2. jnp array param
gm2 = GraphManager()
gm2.add_node(HeatNode(name="h", timestep=0.01, n_cells=4))
gm2.compile()
gm2._nodes["h"].node.params["initial_temperature"] = jnp.array([1.0, jnp.nan, 3.0, 4.0])
show("to_dict with a jnp array param holding NaN",
     lambda: repr(gm2.to_dict()["nodes"][0]["params"]["initial_temperature"])[:60])
show("  ... json.dumps of that to_dict", lambda: jdumps(gm2.to_dict())[:80])

# --- 3. full round trip with non-finite scalars + inf ParamSpec bounds + a group
def build():
    g = GraphManager()
    g.add_node(SpringDamperNode(name="a", timestep=0.01, stiffness=30.0, damping=2.0))
    g.add_node(SpringDamperNode(name="b", timestep=0.01, stiffness=10.0, damping=1.0))
    g.add_edge("a", "b", "position", "anchor_position")
    g.add_edge("b", "a", "position", "anchor_position")
    g.add_coupling_group(["a", "b"], max_iterations=4, tolerance=float("inf"))
    g.set_param_spec("a", "stiffness", ParamSpec(bounds=(-math.inf, math.inf)))
    g.compile()
    return g
g3 = build()
g3._nodes["a"].node.params["damping"] = float("nan")
d = g3.to_dict()
txt = json.dumps(d, sort_keys=True)   # plain json: to_dict is ALREADY encoded
print()
try:
    jdumps(d)
    print("[idem] json_codec.dumps(gm.to_dict()) -> OK")
except Exception as e:
    print(f"[idem] json_codec.dumps(gm.to_dict()) -> {type(e).__name__}: {str(e)[:150]}")
print("[doc] strict-parseable:", bool(strict(txt)))
reg = {"SpringDamperNode": SpringDamperNode}
g4 = GraphManager.from_dict(jloads(txt), reg)
txt2 = json.dumps(g4.to_dict(), sort_keys=True)
print("[rt ] to_dict -> json -> from_dict -> to_dict is a fixed point:", txt == txt2)
if txt != txt2:
    import difflib
    for line in list(difflib.unified_diff(txt.split(","), txt2.split(","), lineterm=""))[:40]:
        print("   ", line)

# --- 4. does the 'tuple' branch of encode_non_finite occur in any production tree?
def has_tuple(o, path="$"):
    if isinstance(o, tuple): return [path]
    if isinstance(o, dict):
        return [p for k, v in o.items() for p in has_tuple(v, f"{path}.{k}")]
    if isinstance(o, list):
        return [p for i, v in enumerate(o) for p in has_tuple(v, f"{path}[{i}]")]
    return []
print("\n[tuple] tuples in a to_dict tree (pre-encode):",
      has_tuple(dict(nodes=[s.node.to_dict() for s in g3._nodes.values()],
                     edges=[e.to_dict() for e in g3._edges],
                     groups=[g.to_dict() for g in g3._coupling_groups],
                     specs={n: {k: s.to_dict() for k, s in o.items()}
                            for n, o in g3.param_spec_overrides().items()})) or "none")
