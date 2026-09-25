"""``shard_info``'s local extent is the grid's, whatever else the state carries.

``ShardedStencilNode`` hands ``update_padded`` ``shard_info = {axis:
(offset, local_extent)}``.  The extent used to be read inside the traced
step off the first local state field long enough to have the axis, in
sorted key order, and a domain integral carried in the state was not
skipped.  A ``HeatNode`` that carries an energy integral named to sort
before ``temperature`` -- a vector, or a per-shard value -- was told its
4-cell blocks held 2 cells (or 1), no block ever reached the rod's right
end, and that end was never closed: 70.9 K off after 40 steps, finite, no
warning.  The extent is now fixed before tracing, from a grid field's
global extent over the devices on its mesh axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.node import SimulationNode
from maddening.nodes.heat import HeatNode

_HAS_4 = len(jax.devices()) >= 4
pytestmark = pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")

N_CELLS = 16
N_DEVICES = 4
DT = 0.2 * (1.0 / N_CELLS) ** 2 / 0.01
T0 = (300.0 + 40.0 * np.sin(np.linspace(0.0, 3.0, N_CELLS))).tolist()
ENDS = {"left_temperature": jnp.float32(200.0), "right_temperature": jnp.float32(400.0)}
STEPS = 30


def _rod_class(shape, per_shard, lists_integral):
    """A HeatNode that also carries ``a_total``, its thermal energy.

    The integral's name sorts before ``temperature``, which is what used to
    make it the field the extent was read from.  ``lists_integral`` keeps
    the default ``state_fields()`` (every ``initial_state`` key, the
    integral included); otherwise only the grid field is listed.
    """

    class RodWithTotal(HeatNode):
        seen: list

        if not lists_integral:
            def state_fields(self):
                return ["temperature"]

        def domain_integral_fields(self):
            return {"a_total"}

        def domain_integral_axes(self):
            return {"a_total": ()} if per_shard else {}

        def initial_state(self):
            state = super().initial_state()
            state["a_total"] = jnp.zeros(shape, jnp.float32)
            return state

        def _total(self, temperature):
            total = jnp.sum(temperature) * (self.params["length"] / self.params["n_cells"])
            return jnp.broadcast_to(total, shape) if shape else total

        def update(self, state, boundary_inputs, dt, *, params=None):
            out = super().update(state, boundary_inputs, dt, params=params)
            out["a_total"] = self._total(out["temperature"])
            return out

        def update_padded(self, state_padded, boundary_inputs, dt, *,
                          static_padded=None, shard_info=None, params=None):
            self.seen.append(shard_info[0][1])
            out = super().update_padded(state_padded, boundary_inputs, dt,
                                        static_padded=static_padded,
                                        shard_info=shard_info, params=params)
            h = self.halo_width()[0]
            out["a_total"] = self._total(out["temperature"][h:-h])
            return out

    return RodWithTotal


@pytest.mark.parametrize("shape, per_shard, lists_integral", [
    ((2,), False, False),
    ((), True, False),
    ((), False, False),
    ((2,), False, True),
    ((), True, True),
    ((), False, True),
], ids=["vector", "per-shard", "scalar", "vector-listed", "per-shard-listed",
        "scalar-listed"])
def test_a_rod_carrying_an_integral_is_the_unsharded_rod(shape, per_shard, lists_integral):
    cls = _rod_class(shape, per_shard, lists_integral)
    ref = cls("rod", DT, n_cells=N_CELLS, initial_temperature=T0)
    ref.seen = []
    want = ref.initial_state()
    for _ in range(STEPS):
        want = ref.update(want, ENDS, DT)

    node = cls("rod", DT, n_cells=N_CELLS, initial_temperature=T0)
    node.seen = []
    wrapped = ShardedStencilNode(node, create_device_mesh(shape=(N_DEVICES,)),
                                 {"devices": 0}, boundary="edge")
    got = wrapped.initial_state()
    for _ in range(STEPS):
        got = wrapped.update(got, ENDS, DT)

    # What update_padded was told: 4-cell blocks, not the integral's length.
    assert node.seen and set(node.seen) == {N_CELLS // N_DEVICES}
    T_got, T_want = np.asarray(got["temperature"]), np.asarray(want["temperature"])
    assert np.all(np.isfinite(T_got))
    # The right rod end is closed on the block that holds it.
    np.testing.assert_allclose(T_got, T_want, rtol=0, atol=1e-3)
    total = np.asarray(got["a_total"])
    if per_shard:
        assert total.shape == (N_DEVICES,)
        np.testing.assert_allclose(total.sum(), np.asarray(want["a_total"]), rtol=1e-5)
    else:
        assert total.shape == shape
        np.testing.assert_allclose(total, np.asarray(want["a_total"]), rtol=1e-5)


class _TwoGrids(SimulationNode):
    """Two fields sharded along axis 0 with different extents (16 and 8)."""

    def __init__(self):
        super().__init__(name="two", timestep=0.1)

    def halo_width(self):
        return {0: 1}

    def initial_state(self):
        return {"a": jnp.zeros(16, jnp.float32), "b": jnp.zeros(8, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {k: v + dt for k, v in state.items()}


class _TwoGridsReadingShardInfo(_TwoGrids):
    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None, shard_info=None):
        return {k: v + dt for k, v in state_padded.items()}


class _TwoGridsIgnoringShardInfo(_TwoGrids):
    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None):
        return {k: v + dt for k, v in state_padded.items()}


def test_grid_fields_of_different_extents_are_refused_for_a_node_reading_shard_info():
    """One ``(offset, extent)`` pair cannot describe blocks of 4 and of 2 cells."""
    wrapped = ShardedStencilNode(_TwoGridsReadingShardInfo(),
                                 create_device_mesh(shape=(N_DEVICES,)), {"devices": 0})
    with pytest.raises(ValueError, match="disagree about the extent") as exc:
        wrapped.update(wrapped.initial_state(), {}, 0.1)
    msg = str(exc.value)
    assert "state field 'a' has 16" in msg and "state field 'b' has 8" in msg


def test_grid_fields_of_different_extents_step_for_a_node_that_ignores_shard_info():
    """Nothing to describe: the refusal is for nodes that read the pair."""
    wrapped = ShardedStencilNode(_TwoGridsIgnoringShardInfo(),
                                 create_device_mesh(shape=(N_DEVICES,)), {"devices": 0})
    out = wrapped.update(wrapped.initial_state(), {}, 0.1)
    np.testing.assert_allclose(np.asarray(out["a"]), 0.1, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(out["b"]), 0.1, rtol=1e-6)
