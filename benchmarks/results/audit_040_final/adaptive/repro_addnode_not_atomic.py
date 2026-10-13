"""Repro: GraphManager.add_node is not exception-safe, and AdaptiveNode is the
node that makes that reachable on a documented, supported configuration.

graph_manager.py:2584-2585
    self._nodes[node.name] = spec              # registered first
    self._state[node.name] = node.initial_state()   # <- AdaptiveNode raises here

The documented recovery from AdaptiveNodeBlindnessError ("use cold_start() and
seed gm.params ... or perturb the parameters yourself", developer guide
'Failure modes') needs the node re-added; the name is now permanently taken.
"""
import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING-wt/audit/adaptive/tests")
import jax
jax.config.update("jax_enable_x64", True)
from maddening.core.graph_manager import GraphManager
from maddening.nodes.adaptive import AdaptiveNodeBlindnessError
from nodes.adaptive._toys import PoissonSineTopKNode

gm = GraphManager()
try:
    gm.add_node(PoissonSineTopKNode("trap", n=64, k=8, theta=0.5))   # the known trap
except AdaptiveNodeBlindnessError as e:
    print("add_node raised AdaptiveNodeBlindnessError (documented, recoverable)\n")

print("gm.node_names              :", list(gm.node_names))
print("gm._nodes (private)      :", list(gm._nodes))
print("gm._state (private)      :", list(gm._state))
print()

print("recovery attempt 1 -- perturb theta and re-add under the same name:")
try:
    gm.add_node(PoissonSineTopKNode("trap", n=64, k=8, theta=0.47))
    print("   ok")
except Exception as e:
    print("  ", type(e).__name__, e)

print("\nrecovery attempt 2 -- remove_node first:")
try:
    gm.remove_node("trap")
    print("   remove_node ok")
except Exception as e:
    print("  ", type(e).__name__, e)

print("\nwhat the half-added node does to the graph:")
for call, fn in (("compile()", gm.compile),
                 ("state_dict()", getattr(gm, "state_dict", None)),
                 ("reset_state()", getattr(gm, "reset_state", None))):
    if fn is None:
        continue
    try:
        fn()
        print(f"   {call}: ok")
    except Exception as e:
        print(f"   {call}: {type(e).__name__}: {str(e)[:170]}")

print("\nrunning the corrupted graph:")
for call in ("step", "run_scan"):
    fn = getattr(gm, call, None)
    if fn is None:
        print(f"   {call}: n/a"); continue
    try:
        fn(5) if call == "run_scan" else fn()
        print(f"   {call}: ok (!)")
    except Exception as e:
        print(f"   {call}: {type(e).__name__}: {str(e)[:200]}")
print("\nand a *clean* second node added afterwards:")
try:
    gm.add_node(PoissonSineTopKNode("good", n=64, k=32, theta=0.42))
    gm.compile()
    gm.step()
    print("   the graph still runs the good node: ok")
except Exception as e:
    print(f"   {type(e).__name__}: {str(e)[:220]}")
