"""``ShardedUnstructuredNode`` refuses a per-cell input the unsharded node refuses.

The differential harness found that a ``source`` of 3 values on a 4-cell
ring split over 4 devices -- the length of a shard's slab, 1 owned and 2
ghost rows -- ran sharded, every shard reading it as its own slab, where
the unsharded node refuses 3 values for 4 cells.  ``ShardedStencilNode``
refuses the same thing (MADD-ANO-057); the unstructured wrapper had no
such check.  These are the neighbouring cases: other lengths, both
exchange transports, a per-cell input with trailing components, the
forms that must still pass, and an input the node does not declare.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import (
    build_unstructured_partition,
    partition_value,
)
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.simulation.hybrid_node import HybridNode
from tests.cloud.multigpu import differential_sharding_support as D

pytestmark = pytest.mark.skipif(
    len(jax.devices()) < 4,
    reason="needs 4 devices (the directory conftest forces 16 virtual CPU devices)")

#: 7 cells in a ring over 2 devices: device 0 owns cells 0-3, device 1
#: cells 4-6.  n_local_max = 4, so the layout has 8 rows; each shard has
#: 2 ghosts, so its slab is 6 rows.  None of the lengths below is 7 (the
#: global count) or 8 (the layout's rows), and none broadcasts to 7.
_RING7 = D.UnstructuredConfig(
    n_devices=2, n_cells=7, assignment=(0, 0, 0, 0, 1, 1, 1), partition="uneven",
    chords=(), contract="params", integral=None, integral_name="a_total",
    integral_listed=False, weight=None, source="none", misshapen_len=0, gain=False,
    dtype="float32", wrapping="single", steps=1, seed=2, surface="run_scan")


def _wrapper(cfg, exchange="all_to_all", hybrid=False):
    layout = D.unstructured_layout(cfg)
    node = D.unstructured_node_class(cfg.contract)(cfg, layout)
    wrapped = ShardedUnstructuredNode(node, create_device_mesh(shape=(cfg.n_devices,)),
                                      layout, exchange=exchange)
    return (HybridNode(wrapped, D._correction) if hybrid else wrapped), node, layout


def test_the_ring_has_the_layout_the_lengths_below_assume():
    layout = D.unstructured_layout(_RING7)
    assert (layout.n_local_max, layout.n_ghost_max) == (4, 2)


@pytest.mark.parametrize("length", [6, 4, 5, 14], ids=["slab", "owned-block", "five", "twice"])
@pytest.mark.parametrize("exchange", ["all_to_all", "ppermute"])
@pytest.mark.parametrize("hybrid", [False, True], ids=["wrapper", "hybrid-around"])
def test_a_per_cell_input_of_another_length_is_refused_naming_the_global_shape(
        length, exchange, hybrid):
    wrapped, inner, _ = _wrapper(_RING7, exchange, hybrid)
    bad = jnp.ones((length,), jnp.float32)
    with pytest.raises(ValueError,
                       match=rf"boundary input 'source' has shape \({length},\).*"
                             r"global shape \(7,\).*partition layout \(\(8,\)"):
        wrapped.update(wrapped.initial_state(), {"source": bad}, 0.1)
    # ... which the unsharded node refuses too: it is not a value it takes.
    with pytest.raises((ValueError, TypeError)):
        inner.update(inner.initial_state(), {"source": bad}, 0.1)


@pytest.mark.parametrize("exchange", ["all_to_all", "ppermute"])
def test_the_forms_the_unsharded_node_takes_still_run_sharded(exchange):
    """A scalar, a ``(1,)`` array, and the per-cell values in partition layout."""
    wrapped, inner, layout = _wrapper(_RING7, exchange)
    rows = D.layout_rows_of_global(layout)
    per_cell = np.linspace(-1.0, 1.0, 7).astype(np.float32)
    in_layout = partition_value(value=per_cell, layout=layout).reshape(8)
    for sharded_value, unsharded_value in [
            (np.float32(0.5), np.float32(0.5)),
            (np.full((1,), 0.5, np.float32), np.full((1,), 0.5, np.float32)),
            (in_layout, per_cell)]:
        got = wrapped.update(wrapped.initial_state(), {"source": jnp.asarray(sharded_value)}, 0.1)
        want = inner.update(inner.initial_state(), {"source": jnp.asarray(unsharded_value)}, 0.1)
        np.testing.assert_allclose(np.asarray(got["x"])[rows], np.asarray(want["x"]),
                                   rtol=0, atol=1e-6)


class _Force(SimulationNode):
    """Cells driven by a per-cell 2-vector input ``force``, declared ``(n, 2)``."""

    def __init__(self, n: int) -> None:
        super().__init__(name="cells", timestep=0.1)
        self.n = n

    def halo_width(self):
        return {}

    def initial_state(self):
        return {"x": jnp.arange(self.n, dtype=jnp.float32)}

    def boundary_input_spec(self):
        return {"force": BoundaryInputSpec(shape=(self.n, 2), description="per-cell 2-vector")}

    def _step(self, x, bi, dt):
        f = jnp.broadcast_to(jnp.asarray(bi.get("force", 0.0), x.dtype), x.shape + (2,))
        return {"x": x + dt * (f[:, 0] - 2.0 * f[:, 1])}

    def update(self, state, boundary_inputs, dt):
        return self._step(state["x"], boundary_inputs, dt)

    def update_padded(self, state_padded, boundary_inputs, dt, *, static_padded=None,
                      shard_info=None):
        return self._step(state_padded["x"], boundary_inputs, dt)


def _force_pair():
    layout = build_unstructured_partition(
        partition_assignment=np.asarray(_RING7.assignment, np.int32),
        edges=_RING7.edges, n_devices=2)
    wrapped = ShardedUnstructuredNode(_Force(7), create_device_mesh(shape=(2,)), layout)
    return wrapped, _Force(7), layout


@pytest.mark.parametrize("shape", [(6, 2), (4, 2), (3,)], ids=["slab", "owned-block", "wrong-trailing"])
def test_a_per_cell_vector_input_of_another_shape_is_refused(shape):
    wrapped, inner, _ = _force_pair()
    bad = jnp.ones(shape, jnp.float32)
    with pytest.raises(ValueError, match=r"global shape \(7, 2\).*partition layout \(\(8, 2\)"):
        wrapped.update(wrapped.initial_state(), {"force": bad}, 0.1)
    with pytest.raises((ValueError, TypeError)):
        inner.update(inner.initial_state(), {"force": bad}, 0.1)


def test_a_uniform_or_per_cell_vector_input_runs_sharded_as_unsharded():
    wrapped, inner, layout = _force_pair()
    rows = D.layout_rows_of_global(layout)
    per_cell = np.stack([np.linspace(0, 1, 7), np.linspace(1, -1, 7)], axis=1).astype(np.float32)
    for sharded_value, unsharded_value in [
            (np.asarray([0.25, -0.5], np.float32),) * 2,
            (np.asarray([[0.25, -0.5]], np.float32),) * 2,
            (partition_value(value=per_cell, layout=layout).reshape(8, 2), per_cell)]:
        got = wrapped.update(wrapped.initial_state(), {"force": jnp.asarray(sharded_value)}, 0.1)
        want = inner.update(inner.initial_state(), {"force": jnp.asarray(unsharded_value)}, 0.1)
        np.testing.assert_allclose(np.asarray(got["x"])[rows], np.asarray(want["x"]),
                                   rtol=0, atol=1e-6)


def test_an_input_the_node_does_not_declare_is_left_to_the_node():
    """The wrapper judges only what ``boundary_input_spec()`` declares per cell.

    An undeclared input is the node's business, as in
    ``ShardedStencilNode``: here the node ignores it, and the step runs.
    """
    wrapped, _, _ = _force_pair()
    out = wrapped.update(wrapped.initial_state(), {"unused": jnp.ones((6,), jnp.float32)}, 0.1)
    assert out["x"].shape == (8,)
