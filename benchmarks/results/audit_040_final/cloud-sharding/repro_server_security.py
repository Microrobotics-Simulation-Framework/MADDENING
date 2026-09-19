"""api/server.py security probes: auth, filesystem reach, resource limits."""
import os, sys, tempfile, threading, time
import numpy as np
import jax.numpy as jnp
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode


class Relax(SimulationNode):
    def __init__(self, name="relax", n=4, rate=0.5, timestep=0.1):
        super().__init__(name=name, timestep=timestep, rate=float(rate), n=int(n))
    def halo_width(self): return {}
    def state_fields(self): return ["x"]
    def initial_state(self):
        return {"x": jnp.arange(int(self.params["n"]), dtype=jnp.float32)}
    def boundary_input_spec(self): return {}
    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": state["x"] + p["rate"] * dt * (1.0 - state["x"])}


root = tempfile.mkdtemp(prefix="audit_ckpt_")
gm = GraphManager(); gm.add_node(Relax()); gm.compile()
srv = SimulationServer(node_registry={"Relax": Relax}, graph_manager=gm,
                       checkpoint_root=root)
c = TestClient(srv.create_app())

print("== 1. Authentication on state-changing endpoints (no credentials sent) ==")
for method, path in [("GET", "/graph"), ("GET", "/graph/state"),
                     ("POST", "/sim/step"), ("POST", "/graph/compile"),
                     ("DELETE", "/graph/nodes/relax")]:
    r = c.request(method, path)
    print(f"   {method:6s} {path:24s} -> {r.status_code}")

print("\n== 2. /checkpoint/{save,load} filesystem containment ==")
outside = os.path.join(tempfile.gettempdir(), "audit_escape")
for p in ["ok.npz", "../audit_escape.npz", "../../../../tmp/audit_escape.npz",
          "/tmp/audit_escape.npz", "sub/dir/ok.npz",
          "..%2Faudit_escape.npz", "....//audit_escape.npz"]:
    r = c.post("/checkpoint/save", params={"path": p})
    print(f"   save path={p!r:38s} -> {r.status_code} {str(r.json())[:70]}")

print("\n   symlink escape attempt:")
link = os.path.join(root, "link")
try:
    os.symlink(tempfile.gettempdir(), link)
    r = c.post("/checkpoint/save", params={"path": "link/audit_escape.npz"})
    print(f"   save path='link/audit_escape.npz' -> {r.status_code} {str(r.json())[:70]}")
except OSError as e:
    print(f"   (symlink not creatable: {e})")

print("\n   /checkpoint/load as a file-existence oracle:")
for p in ["/etc/passwd", "../../../../etc/passwd", "nope.npz"]:
    r = c.post("/checkpoint/load", params={"path": p})
    print(f"   load path={p!r:30s} -> {r.status_code} {str(r.json())[:70]}")

print("\n== 3. Unbounded resource consumption ==")
r = c.post("/sim/run", params={"n_steps": -5})
print(f"   /sim/run?n_steps=-5            -> {r.status_code} {str(r.json())[:60]}")
r = c.post("/graph/nodes", json={"type": "Relax", "name": "big", "timestep": 0.1,
                                 "params": {"n": 10**9}})
print(f"   add node with n=1e9            -> {r.status_code} {str(r.json())[:90]}")
r = c.post("/surrogate/train", json={"node_name": "relax", "n_data_steps": 10**9,
                                     "n_epochs": 10**9, "hidden_sizes": [1]*100,
                                     "batch_size": 1})
print(f"   /surrogate/train huge budget   -> {r.status_code} {str(r.json())[:90]}")

print("\n   n_steps upper bound check (is there one?):")
import inspect, re
src = inspect.getsource(SimulationServer.create_app)
for name in ["n_steps", "n_data_steps", "n_epochs", "hidden_sizes"]:
    has_bound = bool(re.search(rf"{name}\s*[<>]|Field\([^)]*{name}|le=|ge=", src))
    print(f"     {name:14s} any explicit bound in create_app: {has_bound}")

print("\n== 4. Docs / OpenAPI exposure ==")
for p in ["/docs", "/openapi.json", "/redoc"]:
    r = c.get(p)
    print(f"   {p:16s} -> {r.status_code}")

print("\n== 5. What was written outside the checkpoint root? ==")
import glob
leaked = glob.glob(os.path.join(tempfile.gettempdir(), "audit_escape*"))
print(f"   files matching /tmp/audit_escape*: {leaked}")
for f in leaked:
    os.remove(f)
