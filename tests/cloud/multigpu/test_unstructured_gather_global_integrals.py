"""``ShardedUnstructuredNode.gather_global`` passes every domain integral through.

It picked the fields to gather by ``state_fields()`` membership, and the
default ``state_fields()`` lists every ``initial_state`` key, an integral
included: a reduced integral was reshaped as a per-cell field and numpy
refused, and per-shard values on a layout of one cell per shard were
gathered as cells -- reordered into global cell order, silently.  It now
recognises an integral first, as the step classifies its outputs.

The neighbours here: every integral kind, listed or not, on the initial
state and after a step, on a layout with padding rows and on one with a
single cell per shard out of global order.  (The generated differential
tests also check ``gather_global`` on every sharded unstructured run.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from tests.cloud.multigpu import differential_sharding_support as D

pytestmark = pytest.mark.skipif(
    len(jax.devices()) < 4,
    reason="needs 4 devices (the directory conftest forces 16 virtual CPU devices)")

_LAYOUTS = {
    # Device d owns cell 3 - d: one cell per shard, out of global order.
    "one-cell-per-shard-reversed": dict(n_devices=4, n_cells=4, assignment=(3, 2, 1, 0),
                                        partition="balanced_nonglobal"),
    # Device 0 owns 2 cells, device 1 owns 3: a padding row on device 0.
    "uneven-with-padding": dict(n_devices=2, n_cells=5, assignment=(0, 0, 1, 1, 1),
                                partition="uneven"),
}


def _pair(layout_name, integral, listed):
    cfg = D.UnstructuredConfig(
        chords=(), contract="params", integral=integral, integral_name="a_total",
        integral_listed=listed, weight=None, source="none", misshapen_len=0, gain=False,
        dtype="float32", wrapping="single", steps=1, seed=8, surface="run_scan",
        **_LAYOUTS[layout_name])
    layout = D.unstructured_layout(cfg)
    cls = D.unstructured_node_class("params")
    wrapped = ShardedUnstructuredNode(cls(cfg, layout), create_device_mesh(shape=(cfg.n_devices,)),
                                      layout)
    return cfg, wrapped, cls(cfg, layout), layout


@pytest.mark.parametrize("listed", [True, False], ids=["listed", "unlisted"])
@pytest.mark.parametrize("integral", ["scalar", "vector", "per_shard"])
@pytest.mark.parametrize("layout_name", sorted(_LAYOUTS))
def test_gather_global_of_the_initial_state_passes_the_integral_through(
        layout_name, integral, listed):
    cfg, wrapped, inner, _ = _pair(layout_name, integral, listed)
    assert ("a_total" in wrapped.state_fields()) == listed
    placed = wrapped.initial_state()
    got = wrapped.gather_global(placed)
    want = inner.initial_state()
    np.testing.assert_array_equal(got["x"], np.asarray(want["x"]))
    # The integral exactly as placed: replicated once reduced, stacked
    # along the mesh axis (one row per device) when not.
    np.testing.assert_array_equal(got["a_total"], np.asarray(jax.device_get(placed["a_total"])))
    lead = (cfg.n_devices,) if integral == "per_shard" else ()
    assert got["a_total"].shape == lead + tuple(np.shape(want["a_total"]))


@pytest.mark.parametrize("listed", [True, False], ids=["listed", "unlisted"])
@pytest.mark.parametrize("integral", ["scalar", "vector", "per_shard"])
@pytest.mark.parametrize("layout_name", sorted(_LAYOUTS))
def test_gather_global_after_a_step_gathers_cells_and_passes_the_integral_through(
        layout_name, integral, listed):
    cfg, wrapped, inner, layout = _pair(layout_name, integral, listed)
    out = wrapped.update(wrapped.initial_state(), {}, 0.1)
    want = inner.update(inner.initial_state(), {}, 0.1)
    got = wrapped.gather_global(out)
    np.testing.assert_allclose(got["x"], np.asarray(want["x"]), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(got["a_total"], np.asarray(jax.device_get(out["a_total"])))
    total = got["a_total"].sum(axis=0) if integral == "per_shard" else got["a_total"]
    np.testing.assert_allclose(total, np.asarray(want["a_total"]), rtol=1e-5)
    if integral == "per_shard":
        # Shard d's own total, in device order: never reordered by cell.
        x = np.asarray(want["x"])
        per_device = [np.sum(x[np.asarray(ids)] + 1.0) for ids in layout.local_global_ids]
        np.testing.assert_allclose(got["a_total"], per_device, rtol=1e-5)
