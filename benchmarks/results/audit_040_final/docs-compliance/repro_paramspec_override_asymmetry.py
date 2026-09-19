"""Reproducer: a ParamSpec override naming a key the node does not have is a
hard ValueError through GraphManager.from_dict and a warn-and-drop through
load_graph_from_usd.  Same data, same contract, two verdicts; neither is
documented in docs/ or CHANGELOG.md.

  cd <worktree>
  PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu python <this file>
"""
import json, warnings
from pxr import Usd, Sdf
from maddening import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.core.params import ParamSpec
from maddening.usd.serialization import save_graph_to_usd, load_graph_from_usd

REG = {"SpringDamperNode": SpringDamperNode}
BAD = {"no_such_parameter": ParamSpec(trainable=False).to_dict()}


def fresh():
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="s", timestep=0.01))
    return gm


print("--- path 1: GraphManager.from_dict ---")
cfg = fresh().to_dict()
cfg["param_specs"] = {"s": BAD}
try:
    GraphManager.from_dict(cfg, REG)
    print("    loaded; override silently ignored")
except Exception as e:
    print(f"    {type(e).__name__}: {str(e)[:150]}")

print("--- path 2: load_graph_from_usd, same override ---")
stage = Usd.Stage.CreateInMemory()
save_graph_to_usd(fresh(), stage)
prim = stage.GetPrimAtPath("/Simulation/nodes/s")
prim.CreateAttribute("maddening:paramSpecOverridesJson",
                     Sdf.ValueTypeNames.String).Set(json.dumps(BAD))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        load_graph_from_usd(stage)
        msgs = [f"{type(x.message).__name__}: {str(x.message)[:110]}" for x in w]
        print("    loaded OK.  warnings:", msgs or "none")
    except Exception as e:
        print(f"    {type(e).__name__}: {str(e)[:150]}")
