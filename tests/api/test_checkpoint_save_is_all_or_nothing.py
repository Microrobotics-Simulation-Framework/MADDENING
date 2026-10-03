"""``POST /checkpoint/save`` writes the checkpoint whole, or not at all.

It wrote the ``.npz`` and then its manifest, so a save refused at the
manifest answered 400 "could not save checkpoint" having already written
the ``.npz`` -- over any earlier file of that name, which then loaded with
200 at the wrong time.  Reproduced through REST alone: a save to
``x.npz.manifest.json/y.npz`` makes the directory ``x.npz.manifest.json``,
and every later save of ``x.npz`` failed on its manifest.  The checkpoint
is now written under a temporary name, the manifest next, and the file
moved into place last.
"""

from __future__ import annotations

import hashlib
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode


def _client(root):
    gm = GraphManager()
    gm.add_node(BallNode("c", timestep=1.0 / 64.0, initial_velocity=1.0, gravity=0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({}, graph_manager=gm, checkpoint_root=str(root))
    return TestClient(server.create_app(), raise_server_exceptions=False)


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_save_refused_at_its_manifest_leaves_the_earlier_checkpoint(tmp_path):
    client = _client(tmp_path)
    assert client.post("/sim/run", params={"n_steps": 64}).status_code == 200
    assert client.post("/checkpoint/save", params={"path": "x.npz"}).status_code == 200
    before = _digest(tmp_path / "x.npz")
    # A directory where x.npz's manifest goes, made through the API.
    (tmp_path / "x.npz.manifest.json").unlink()
    assert client.post("/checkpoint/save",
                       params={"path": "x.npz.manifest.json/y.npz"}).status_code == 200
    assert client.post("/sim/run", params={"n_steps": 64}).status_code == 200
    resp = client.post("/checkpoint/save", params={"path": "x.npz"})
    assert resp.status_code == 400, resp.text
    assert "nothing was written" in resp.json()["detail"]
    assert _digest(tmp_path / "x.npz") == before
    leftovers = [p.name for p in tmp_path.iterdir() if "partial" in p.name]
    assert leftovers == []
    load = client.post("/checkpoint/load", params={"path": "x.npz"})
    assert load.status_code == 200 and load.json()["state"]["c"]["position"] == 1.0


def test_a_save_refused_at_its_manifest_writes_no_new_checkpoint(tmp_path):
    client = _client(tmp_path)
    assert client.post("/checkpoint/save",
                       params={"path": "z.npz.manifest.json/y.npz"}).status_code == 200
    resp = client.post("/checkpoint/save", params={"path": "z.npz"})
    assert resp.status_code == 400, resp.text
    assert not (tmp_path / "z.npz").exists()


def test_a_save_still_writes_the_checkpoint_and_its_manifest(tmp_path):
    from maddening.core.simulation.checkpoint import verify_manifest

    client = _client(tmp_path)
    assert client.post("/sim/run", params={"n_steps": 3}).status_code == 200
    resp = client.post("/checkpoint/save", params={"path": "ok.npz"})
    assert resp.status_code == 200, resp.text
    verify_manifest(tmp_path / "ok.npz")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ok.npz", "ok.npz.manifest.json"]
