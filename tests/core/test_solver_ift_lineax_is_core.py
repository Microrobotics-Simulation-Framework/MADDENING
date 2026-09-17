"""Without lineax the IFT Krylov adjoint must fail with an actionable
message pointing at the ``[ift]`` extra (v0.4.0 plan: release-followup)."""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager, _import_lineax
from maddening.nodes.spring import SpringDamperNode


def _coupled(linear_solver):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.01, initial_position=3.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=10, tolerance=1e-8,
                          linear_solver=linear_solver)
    gm.compile()
    return gm


def _grad_k(gm):
    ext = gm._default_external_inputs()

    def loss(k):
        p = jax.tree.map(lambda x: x, gm.params)
        p["nodes"]["a"]["stiffness"] = k
        return gm._compiled_step(gm._state, ext, p)["b"]["position"]

    return float(jax.grad(loss)(jnp.asarray(100.0, jnp.float32)))


def test_import_helper_message_points_at_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "lineax", None)
    with pytest.raises(ImportError, match=r"pip install maddening\[ift\]") as ei:
        _import_lineax()
    assert "linear_solver='dense'" in str(ei.value)


def test_gmres_adjoint_without_lineax_raises_friendly_error(monkeypatch):
    gm = _coupled("gmres")
    monkeypatch.setitem(sys.modules, "lineax", None)
    with pytest.raises(ImportError, match=r"maddening\[ift\]"):
        _grad_k(gm)


def test_dense_adjoint_needs_no_lineax(monkeypatch):
    monkeypatch.setitem(sys.modules, "lineax", None)
    gm = _coupled("dense")
    g = _grad_k(gm)
    assert g != 0.0 and abs(g) < 1e3


def test_ift_extra_declared():
    import tomllib
    from pathlib import Path

    py = tomllib.loads(Path(__file__).resolve().parents[2].joinpath("pyproject.toml").read_text())
    ift = py["project"]["optional-dependencies"]["ift"]
    assert any(d.startswith("lineax") for d in ift), ift
