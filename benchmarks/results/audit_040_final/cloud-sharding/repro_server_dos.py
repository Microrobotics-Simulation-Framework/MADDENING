"""Unbounded CPU / memory on unauthenticated endpoints (safe sizes only)."""
import os, time
import jax.numpy as jnp
from fastapi.testclient import TestClient
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode

def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return -1.0

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

gm = GraphManager(); gm.add_node(Relax()); gm.compile()
c = TestClient(SimulationServer(node_registry={"Relax": Relax},
                                graph_manager=gm).create_app())

print("== A. /sim/run: no upper bound on n_steps ==")
print("   /sim/profile clamps n_steps to [1,1000] (server.py:995).")
print("   /sim/run (server.py:657-663) passes n_steps straight to gm.run().")
for n in (100, 1000, 10000):
    t = time.perf_counter()
    r = c.post("/sim/run", params={"n_steps": n})
    dt = time.perf_counter() - t
    print(f"   n_steps={n:<6d} -> {r.status_code} in {dt:7.3f}s  "
          f"({dt/n*1e6:6.1f} us/step)")
print("   Timings are from a machine shared with other agents; the finding is")
print("   the linear scaling in an unclamped, caller-chosen n_steps, not the")
print("   absolute numbers.  A single request with n_steps=10**9 occupies a")
print("   worker thread for ~days with no way to cancel it over the API.")

print("\n== B. add_node: caller chooses the state array size ==")
base = rss_mb()
print(f"   RSS before: {base:8.1f} MB")
for n in (10**6, 10**7, 10**8):
    r = c.post("/graph/nodes", json={"type": "Relax", "name": f"big{n}",
                                     "timestep": 0.1, "params": {"n": n}})
    print(f"   n={n:<11d} -> {r.status_code}   RSS now {rss_mb():8.1f} MB "
          f"(+{rss_mb()-base:7.1f} MB)   expected array {n*4/1e6:.0f} MB")
print("   _non_finite_param() rejects a non-finite float but nothing bounds an")
print("   integer that becomes an array dimension; _dry_run_node traces")
print("   abstractly so it never notices the size.")
