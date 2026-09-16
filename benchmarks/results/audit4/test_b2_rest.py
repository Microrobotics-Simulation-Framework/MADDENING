import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax.numpy as jnp, pytest
from fastapi.testclient import TestClient
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode
REGISTRY = {"BallNode": BallNode, "TableNode": TableNode, "SpringDamperNode": SpringDamperNode}

@pytest.fixture
def loaded():
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01, position=0.0))
    gm.add_node(BallNode(name="ball", timestep=0.01, initial_position=5.0, elasticity=0.7))
    gm.add_node(SpringDamperNode(name="s", timestep=0.01, stiffness=30.0, damping=2.0, initial_position=0.5))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm)
    return TestClient(server.create_app(), raise_server_exceptions=False), gm

def test_put_state_wrong_shape(loaded):
    client, gm = loaded
    before = {k: np.asarray(v).copy() for k, v in gm.get_node_state("s").items()}
    r = client.put("/graph/state/s", json={"state": {"position": [1.0, 2.0, 3.0], "velocity": 0.0}})
    print("wrong shape ->", r.status_code, r.text[:200])
    r2 = client.post("/sim/step")
    print("step ->", r2.status_code, r2.text[:300])
    assert r.status_code == 400
    for k, v in before.items():
        assert np.asarray(gm.get_node_state("s")[k]).shape == v.shape

def test_put_state_missing_field(loaded):
    client, gm = loaded
    r = client.put("/graph/state/s", json={"state": {"position": 1.0}})
    print("missing field ->", r.status_code, r.text[:200])
    r2 = client.post("/sim/step")
    print("step ->", r2.status_code, r2.text[:300])
    r3 = client.post("/sim/step")
    print("step again ->", r3.status_code, r3.text[:300])
    assert r.status_code == 400

def test_put_state_unknown_field(loaded):
    client, gm = loaded
    r = client.put("/graph/state/s", json={"state": {"position": 1.0, "velocity": 0.0, "bogus": 3.0}})
    print("unknown field ->", r.status_code, r.text[:200])
    r2 = client.post("/sim/step")
    print("step ->", r2.status_code, r2.text[:300])
    assert r.status_code == 400

def test_put_state_string(loaded):
    client, gm = loaded
    r = client.put("/graph/state/s", json={"state": {"position": "x", "velocity": 0.0}})
    print("string ->", r.status_code, r.text[:200])
    r2 = client.post("/sim/step")
    print("step ->", r2.status_code, r2.text[:300])
    assert r.status_code == 400

def test_put_state_nan_then_run_scan(loaded):
    client, gm = loaded
    r = client.put("/graph/state/s", json={"state": {"position": float("nan"), "velocity": 0.0}})
    print("nan ->", r.status_code, r.text[:200])
    assert r.status_code == 400

def test_put_state_then_run_scan_dtype(loaded):
    client, gm = loaded
    r = client.put("/graph/state/s", json={"state": {"position": 1, "velocity": 0}})
    assert r.status_code == 200
    print({k: v.dtype for k, v in gm.get_node_state("s").items()})
    gm.run_scan(2)

def test_checkpoint_save_arbitrary_path(loaded, tmp_path):
    client, gm = loaded
    target = tmp_path / "anywhere" / "evil"
    target.parent.mkdir()
    r = client.post("/checkpoint/save", params={"path": str(target)})
    print("save ->", r.status_code, r.text[:200], list(target.parent.iterdir()))
    assert not (target.parent / "evil.npz").exists(), "unauthenticated REST wrote an arbitrary server path"

def test_checkpoint_load_arbitrary_path(loaded, tmp_path):
    client, gm = loaded
    r = client.post("/checkpoint/load", params={"path": "/etc/passwd"})
    print("load ->", r.status_code, r.text[:300])

def test_sim_run_negative_or_huge(loaded):
    client, gm = loaded
    r = client.post("/sim/run", params={"n_steps": -5})
    print("run -5 ->", r.status_code, r.text[:100])
    r = client.post("/sim/run", params={"n_steps": "abc"})
    print("run abc ->", r.status_code, r.text[:100])

def test_add_node_bad_params(loaded):
    client, gm = loaded
    r = client.post("/graph/nodes", json={"type": "SpringDamperNode", "name": "s2", "timestep": 0.01, "params": {"stiffness": "hot"}})
    print("add node bad param ->", r.status_code, r.text[:200])
    print("nodes now:", list(gm._nodes))
    r2 = client.post("/sim/step"); print("step ->", r2.status_code, r2.text[:300])
    assert r.status_code in (400, 422)
    assert "s2" not in gm._nodes or r2.status_code == 200

def test_add_edge_bad(loaded):
    client, gm = loaded
    r = client.post("/graph/edges", json={"source_node": "nope", "target_node": "ball", "source_field": "x", "target_field": "y"})
    print("add edge bad ->", r.status_code, r.text[:200])
    r2 = client.post("/sim/step"); print("step ->", r2.status_code, r2.text[:200])
    assert r.status_code in (400, 404, 422)
    assert r2.status_code == 200
