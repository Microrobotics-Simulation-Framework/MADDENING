"""MADD-ANO-015 affected_versions says ">=0.1.0"; its own description says
"From v0.1.0 to v0.3.1" and resolution_version is 0.4.0.  What does 0.4.0 do?"""
import os, inspect
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from maddening.viz import network as nw
from maddening.cloud.multigpu import coordinator as co
for cls in (nw.NetworkRelay, nw.NetworkReceiver, nw.CommandPublisher, nw.CommandReceiver):
    sig = inspect.signature(cls.__init__)
    print(f"{cls.__name__:<18} address default = {sig.parameters['address'].default!r}")
sig = inspect.signature(co.Coordinator.__init__)
print("Coordinator params:", {k: v.default for k, v in sig.parameters.items() if k != "self"})
