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


def _listing(root) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def _name_max(root) -> int:
    try:
        return os.pathconf(root, "PC_NAME_MAX")
    except (OSError, ValueError):
        return 255


def test_every_name_the_filesystem_holds_saves_with_its_manifest(tmp_path):
    """The longest checkpoint name whose manifest name still fits saves and
    loads.  The temporary names carried the checkpoint's name and some 50
    bytes more, so a valid 183-byte name was refused."""
    client = _client(tmp_path)
    limit = _name_max(tmp_path)
    longest = "x" * (limit - len(".npz.manifest.json")) + ".npz"
    for name in ("x" * 150 + ".npz", "x" * 200 + ".npz", longest):
        resp = client.post("/checkpoint/save", params={"path": f"d/sub/{name}"})
        assert resp.status_code == 200, (len(name), resp.text)
        assert (tmp_path / "d" / "sub" / name).is_file()
        assert (tmp_path / "d" / "sub" / (name + ".manifest.json")).is_file()
        assert client.post("/checkpoint/load",
                           params={"path": f"d/sub/{name}"}).status_code == 200
    assert not [p for p in _listing(tmp_path) if "partial" in p]


def test_a_name_too_long_is_refused_before_any_directory_is_made(tmp_path):
    """A name the filesystem cannot hold -- the checkpoint's, a directory's
    or its manifest's -- is a 400 that writes nothing: its directories used
    to be made first and left behind by the 400 that said "nothing was
    written"."""
    client = _client(tmp_path)
    limit = _name_max(tmp_path)
    before = _listing(tmp_path)
    for path in ("new/dirs/" + "x" * (limit - len(".npz.manifest.json") + 1) + ".npz",
                 "new/dirs/" + "x" * (limit - 3) + ".npz",
                 "new/" + "d" * (limit + 1) + "/c.npz"):
        resp = client.post("/checkpoint/save", params={"path": path})
        assert resp.status_code == 400, resp.text
        assert "nothing was written" in resp.json()["detail"]
        assert "this filesystem takes at most" in resp.json()["detail"]
        assert _listing(tmp_path) == before, path


def test_a_save_refused_after_its_directories_were_made_removes_them(tmp_path, monkeypatch):
    """A refusal after the directories exist (the graph's own save fails)
    removes the ones this save made, and only those."""
    client = _client(tmp_path)
    (tmp_path / "kept").mkdir()
    before = _listing(tmp_path)

    def refuse(self, path):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(GraphManager, "save_state", refuse)
    resp = client.post("/checkpoint/save", params={"path": "kept/a/b/c.npz"})
    assert resp.status_code == 400, resp.text
    assert "nothing was written: No space left on device" in resp.json()["detail"]
    assert _listing(tmp_path) == before
