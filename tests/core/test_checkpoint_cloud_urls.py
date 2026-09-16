"""C3 (v0.4.0 plan): cloud-storage URL schemes for ``download_and_load_state``
via fsspec — exercised end-to-end on fsspec's in-memory filesystem, plus
the actionable error when the backend is missing."""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.simulation import checkpoint as ck
from maddening.nodes.spring import SpringDamperNode


def _gm():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, initial_position=0.0))
    gm.compile()
    return gm


def test_unknown_scheme_still_rejected():
    with pytest.raises(ValueError, match="Unsupported URL scheme 'ftp'"):
        ck.download_and_load_state(_gm(), "ftp://host/x.npz")


def test_missing_fsspec_gives_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "fsspec", None)
    with pytest.raises(ImportError, match=r"pip install fsspec s3fs"):
        ck.download_and_load_state(_gm(), "s3://bucket/run/ckpt.npz")


def test_memory_filesystem_round_trip(tmp_path):
    fsspec = pytest.importorskip("fsspec")
    gm = _gm()
    for _ in range(7):
        gm.step()
    local, _mpath = ck.save_state_with_manifest(gm, tmp_path / "ckpt.npz")
    # "Upload" both files to the in-memory filesystem.
    fs = fsspec.filesystem("memory")
    for src, name in ((local, "ckpt.npz"), (ck.Path(str(local) + ".manifest.json"),
                                            "ckpt.npz.manifest.json")):
        with fs.open(f"/run/{name}", "wb") as f:
            f.write(ck.Path(src).read_bytes())
    fresh = _gm()
    manifest = ck.download_and_load_state(fresh, "memory:///run/ckpt.npz")
    assert manifest
    assert float(fresh.get_node_state("s")["position"]) == float(gm.get_node_state("s")["position"])
