#!/usr/bin/env python3
"""Write parameters over the REST API, in-process: what is honoured, what is refused.

Serves a two-node graph -- a spring-damper and a heat rod -- through
:class:`~maddening.api.server.SimulationServer` and drives it with
FastAPI's in-process ``TestClient``: no port is opened and no other
process is started.  Then:

1. ``GET /graph/params/spring`` reads the values in force.
2. ``PUT /graph/params/spring`` with a new stiffness is **honoured**: the
   reply echoes it, and the next ``POST /sim/step`` is exactly the step a
   graph *built* with that stiffness takes from the same state -- without
   recompiling (the step's trace count does not move).
3. A damping below its ``ParamSpec`` bound is **refused**: a 400 that says
   why, and nothing is written.
4. A heat-rod diffusivity past the explicit stencil's stability limit is
   **refused** too: the running graph could step with it, but a graph
   saved with it could not be loaded, because ``HeatNode``'s constructor
   refuses it.  The 400 carries the constructor's reason.
5. An initial condition is accepted and takes effect at the next
   ``POST /sim/reset``.

The rules behind each answer are in ``docs/user_guide/parameters.md``,
"Writing parameters over REST".  To run the same API as a real server on
loopback, see ``api_server.py``.  This example never calls the
``/cloud/*`` routes.

Usage
-----
    python -m maddening.examples.servers.rest_params_demo
    python -m maddening.examples.servers.rest_params_demo --steps 5
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode

DT = 0.01
REGISTRY = {"SpringDamperNode": SpringDamperNode, "HeatNode": HeatNode}


def build(stiffness: float = 30.0) -> GraphManager:
    """A spring and a 16-cell rod, each fed by external inputs (zero by
    default: the spring's anchor and the rod's end temperatures)."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", DT, stiffness=stiffness, damping=2.0,
                                 initial_position=0.5))
    gm.add_node(HeatNode("rod", DT, n_cells=16, length=1.0, thermal_diffusivity=0.01,
                         initial_temperature=np.linspace(0.0, 1.0, 16).tolist()))
    gm.add_external_input("spring", "anchor_position")
    gm.add_external_input("rod", "left_temperature")
    gm.add_external_input("rod", "right_temperature")
    gm.compile()
    return gm


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def show(response) -> dict:
    print(f"  -> {response.status_code} {response.json()}")
    return response.json()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=50,
                        help="Steps taken before the first write (default 50)")
    args = parser.parse_args(argv)

    gm = build()
    with tempfile.TemporaryDirectory(prefix="maddening_rest_demo_") as tmp:
        server = SimulationServer(node_registry=REGISTRY, graph_manager=gm,
                                  checkpoint_root=tmp)
        client = TestClient(server.create_app())
        run(client, gm, args.steps)
    print()
    print("All checks passed.")
    return 0


def run(client: TestClient, gm: GraphManager, steps: int) -> None:
    section("1. Read the parameters in force")
    print("GET /graph/params/spring")
    before = show(client.get("/graph/params/spring"))
    assert before["stiffness"] == 30.0
    r = client.post(f"/sim/run?n_steps={steps}")
    assert r.status_code == 200
    print(f"POST /sim/run?n_steps={steps}  -> {r.status_code}")

    section("2. A valid write is honoured on the next step, without a recompile")
    state = {name: gm.get_node_state(name) for name in gm.node_names}
    traces = gm.trace_count
    print('PUT /graph/params/spring {"params": {"stiffness": 45.0}}')
    reply = show(client.put("/graph/params/spring", json={"params": {"stiffness": 45.0}}))
    assert reply["params"]["stiffness"] == 45.0
    print("POST /sim/step")
    stepped = client.post("/sim/step").json()["spring"]
    print(f"  -> spring {stepped}")

    reference = build(stiffness=45.0)          # a graph built with the new value
    for name, fields in state.items():
        reference.set_node_state(name, fields)
    reference.step()
    expected = {f: float(v) for f, v in reference.get_node_state("spring").items()}
    print(f"  a graph built with stiffness=45, stepped from the same state: {expected}")
    assert stepped == expected, (stepped, expected)
    print(f"  identical; the step's trace count is {gm.trace_count} (was {traces}): "
          f"no recompile")
    assert gm.trace_count == traces

    section("3. A value outside its ParamSpec bounds is refused")
    print('PUT /graph/params/spring {"params": {"damping": -1.0}}')
    refused = show(client.put("/graph/params/spring", json={"params": {"damping": -1.0}}))
    assert "below bound" in refused["detail"]
    assert client.get("/graph/params/spring").json()["damping"] == 2.0
    print("  damping is still 2.0: nothing was written")

    section("4. A value the node's constructor refuses is refused here too")
    print('PUT /graph/params/rod {"params": {"thermal_diffusivity": 1.0}}')
    r = client.put("/graph/params/rod", json={"params": {"thermal_diffusivity": 1.0}})
    detail = r.json()["detail"]
    print(f"  -> {r.status_code} {detail[:300]}...")
    assert r.status_code == 400 and "unstable" in detail and "Nothing was written" in detail
    alpha = client.get("/graph/params/rod").json()["thermal_diffusivity"]
    print(f"  thermal_diffusivity is still {alpha:.6g}: nothing was written")
    assert abs(alpha - 0.01) < 1e-8          # float32 0.01, read back as a float

    section("5. An initial condition takes effect at the next reset")
    print('PUT /graph/params/spring {"params": {"initial_position": 2.0}}')
    show(client.put("/graph/params/spring", json={"params": {"initial_position": 2.0}}))
    reset = client.post("/sim/reset").json()["state"]["spring"]
    print(f"POST /sim/reset  -> spring {reset}")
    assert reset == {"position": 2.0, "velocity": 0.0}
    final = client.get("/graph/params/spring").json()
    print(f"GET /graph/params/spring  -> {final}")
    assert final["stiffness"] == 45.0 and final["damping"] == 2.0


if __name__ == "__main__":
    sys.exit(main())
