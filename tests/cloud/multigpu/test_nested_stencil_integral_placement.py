"""A domain integral starts in the shape and placement a step returns, under any wrapping.

A ``ShardedStencilNode`` nested in another took a per-shard integral's
initial value from the inner wrapper's ``initial_state`` -- already
stacked along its unreduced mesh axes -- and stacked it again: ``(2, 2)``
on two devices where a step returns ``(2,)``, so ``step()`` silently
changed the state's shape and ``run_scan`` refused the carry.  The outer
wrapper now places the value the node itself builds.

The neighbours: every integral kind, two and three levels of nesting, a
rod and a pencil, a partial-axis integral, ``HybridNode`` around a nested
wrapper, and the same invariant in the other two wrappers.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode, ShardedStencilNode
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from maddening.core.graph_manager import GraphManager
from maddening.core.simulation.hybrid_node import HybridNode
from tests.cloud.multigpu import differential_sharding_support as D

pytestmark = pytest.mark.skipif(
    len(jax.devices()) < 4,
    reason="needs 4 devices (the directory conftest forces 16 virtual CPU devices)")

_MESHES = {
    "rod-on-2": dict(mesh_shape=(2,), axis_names=("devices",), axis_map=(("devices", 0),),
                     shape=(4,), halo=(1,)),
    "pencil-2x2": dict(mesh_shape=(2, 2), axis_names=("px", "py"),
                       axis_map=(("px", 0), ("py", 1)), shape=(2, 4), halo=(1, 1)),
}


def _cfg(mesh_name, integral):
    return D.StencilConfig(
        fill="edge", declares=False, contract="params", integral=integral,
        integral_name="a_total", integral_listed=True, reads_shard_info=True, kappa=None,
        kappa_axis=0, table=None, source="none", misshapen_shape=(), gain=False, faces=False,
        dtype="float32", wrapping="single", steps=2, seed=5, surface="run_scan",
        **_MESHES[mesh_name])


def _wrap(node, cfg, depth):
    mesh = create_device_mesh(shape=cfg.mesh_shape, axis_names=cfg.axis_names)
    for _ in range(depth):
        node = ShardedStencilNode(node, mesh, dict(cfg.axis_map), boundary=cfg.fill)
    return node


def _shapes(state):
    return {k: tuple(np.shape(v)) for k, v in state.items()}


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("integral", ["scalar", "vector", "per_shard"])
@pytest.mark.parametrize("mesh_name", sorted(_MESHES))
def test_the_initial_state_has_the_shapes_a_step_returns(mesh_name, integral, depth):
    cfg = _cfg(mesh_name, integral)
    node = _wrap(D.make_stencil_node(cfg), cfg, depth)
    start = node.initial_state()
    after = node.update(start, {}, node.delta_t)
    assert _shapes(start) == _shapes(after)
    if integral == "per_shard":
        # One leading axis per mesh axis, whatever the depth.
        assert start["a_total"].shape == tuple(cfg.mesh_shape)


@pytest.mark.parametrize("integral", ["scalar", "vector", "per_shard"])
@pytest.mark.parametrize("mesh_name", sorted(_MESHES))
def test_three_nested_wrappers_scan_as_one(mesh_name, integral):
    """Only the outermost wrapper runs: the result is the single wrapper's, bit for bit."""
    cfg = _cfg(mesh_name, integral)
    finals = []
    for depth in (1, 3):
        gm = GraphManager()
        gm.add_node(_wrap(D.make_stencil_node(cfg), cfg, depth))
        gm.compile()
        finals.append({k: np.asarray(v) for k, v in gm.run_scan(cfg.steps)[D.NODE_NAME].items()})
    assert finals[0].keys() == finals[1].keys()
    for k in finals[0]:
        np.testing.assert_array_equal(finals[1][k], finals[0][k], err_msg=k)


class _PerRowIntegral:
    """Mixin: the integral reduced over ``px`` only, stacked along ``py``."""

    def domain_integral_axes(self):
        return {"a_total": ("px",)}


def test_a_partial_axis_integral_under_a_nested_wrapper_starts_stacked_once():
    cfg = _cfg("pencil-2x2", "scalar")
    base = D.stencil_node_class(cfg.contract, cfg.reads_shard_info, cfg.declares)
    cls = type("PerRow", (_PerRowIntegral, base), {})
    node = _wrap(cls(cfg), cfg, 2)
    start = node.initial_state()
    after = node.update(start, {}, node.delta_t)
    assert start["a_total"].shape == after["a_total"].shape == (2,)
    gm = GraphManager()
    gm.add_node(_wrap(cls(cfg), cfg, 2))
    gm.compile()
    assert gm.run_scan(2)[D.NODE_NAME]["a_total"].shape == (2,)


def test_a_hybrid_node_around_a_nested_wrapper_answers_as_the_unsharded_hybrid():
    cfg = replace(_cfg("pencil-2x2", "per_shard"), wrapping="nested")
    case = D.build_case(cfg)
    make = case.make
    case.make = lambda sharded: (HybridNode(make(True), D._correction) if sharded
                                 else HybridNode(make(False), D._correction))
    for surface in ("run_scan", "step", "set_state"):
        D.assert_paths_agree(case, D.run_surface(case, surface, True),
                             D.run_surface(case, surface, False), surface)


def test_the_unstructured_wrapper_starts_a_per_shard_integral_in_the_shape_a_step_returns():
    cfg = D.UnstructuredConfig(
        n_devices=4, n_cells=6, assignment=(0, 1, 2, 3, 0, 1), partition="uneven", chords=(),
        contract="params", integral="per_shard", integral_name="a_total", integral_listed=True,
        weight=None, source="none", misshapen_len=0, gain=False, dtype="float32",
        wrapping="single", steps=1, seed=1, surface="run_scan")
    layout = D.unstructured_layout(cfg)
    node = ShardedUnstructuredNode(D.unstructured_node_class("params")(cfg, layout),
                                   create_device_mesh(shape=(4,)), layout)
    start = node.initial_state()
    assert _shapes(start) == _shapes(node.update(start, {}, 0.1))
    assert start["a_total"].shape == (4,)


def test_nested_pointwise_wrappers_keep_a_whole_domain_total_in_shape():
    cfg = D.PointwiseConfig(n_devices=4, shape=(8,), shard_axis=0, contract="params",
                            total=True, source="none", misshapen_shape=(), gain=False,
                            dtype="float32", wrapping="nested", steps=1, seed=1,
                            surface="run_scan")
    mesh = create_device_mesh(shape=(4,))
    node = ShardedPointwiseNode(ShardedPointwiseNode(D.pointwise_node_class("params")(cfg), mesh),
                                mesh)
    start = node.initial_state()
    assert _shapes(start) == _shapes(node.update(start, {}, 0.05))
