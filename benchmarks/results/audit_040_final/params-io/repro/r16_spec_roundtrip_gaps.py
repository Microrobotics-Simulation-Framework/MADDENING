"""ParamSpec round trip through config and USD for the cases the property
strategy never draws: logit, a finite upper bound, unicode text, -inf/inf."""
import json, math
import numpy as np, jax.numpy as jnp
from pxr import Usd
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.usd.serialization import save_graph_to_usd, load_graph_from_usd

NUL = chr(0)
SPECS = {
    "stiffness": ParamSpec(bounds=(1.0, 1e6), transform="logit",
                           description="unicode ok " + NUL + " tab\there", units="N.m-1"),
    "damping":   ParamSpec(trainable=False, bounds=(-math.inf, math.inf)),
    "mass":      ParamSpec(bounds=(5e-324, None), transform="log"),
    "rest_length": ParamSpec(bounds=(-1e308, 1e308)),
}

def build():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.5))
    gm.compile()
    for k, s in SPECS.items():
        gm.set_param_spec("s", k, s)
    return gm

gm = build()

def compare(tag, g):
    print(tag)
    for k, s in SPECS.items():
        got = g.param_specs()["nodes"]["s"][k]
        ok = got == s
        print(f"  {k:12s} identical={ok}")
        if not ok:
            print(f"      in : {s!r}")
            print(f"      out: {got!r}")

print("=== config to_dict / from_dict ===")
cfg = gm.to_dict()
txt = json.dumps(cfg)
print("  param_specs JSON:", json.dumps(cfg["param_specs"])[:300])
try:
    json.loads(txt, parse_constant=lambda c: (_ for _ in ()).throw(
        ValueError("non-standard JSON token " + repr(c))))
    print("  strict JSON: OK")
except ValueError as e:
    print("  strict JSON: FAILS ->", e)
gm2 = GraphManager.from_dict(json.loads(txt), {"SpringDamperNode": SpringDamperNode})
gm2.compile()
compare("  round trip:", gm2)

print()
print("=== USD in-memory ===")
stage = Usd.Stage.CreateInMemory()
save_graph_to_usd(gm, stage)
gm3 = load_graph_from_usd(stage); gm3.compile()
compare("  round trip:", gm3)

print()
print("=== USD via a .usda file on disk ===")
import tempfile, pathlib
d = pathlib.Path(tempfile.mkdtemp())
st2 = Usd.Stage.CreateNew(str(d / "g.usda"))
save_graph_to_usd(gm, st2); st2.Save()
gm4 = load_graph_from_usd(Usd.Stage.Open(str(d / "g.usda"))); gm4.compile()
compare("  round trip:", gm4)
