"""What ``ShardedUnstructuredNode`` hands a node, and what it refuses.

The wrapper steps a node in *partition layout* -- ``[owned cells | ghost
cells]`` per shard, each owned block padded to the largest one -- and
three things about that layout were silent before 0.4.0:

* a Cartesian stencil node (non-empty ``halo_width()``) was accepted and
  read the partition layout as ``[halo | interior | halo]``: every cell
  wrong, no error.  The stencil wrapper's divisibility refusal pointed
  such a node at this wrapper;
* a node whose cell count differed from the layout's lost the extra cells;
* ``shard_info`` gave every shard the padded block length and nothing
  else, so a node summing its block summed the padding into its integral.

And one thing was loud: a domain integral carried in the state (a graph's
second step) was partitioned like a per-cell field and the step failed.
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
from maddening.cloud.multigpu.sharded_node import (
    ShardedPointwiseNode,
    ShardedStencilNode,
)
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode

_HAS_4 = len(jax.devices()) >= 4
pytestmark = pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")


def _chain_layout(n, n_devices):
    """Contiguous blocks of a chain: ``n=7`` on 2 devices owns (4, 3) cells."""
    pa = (np.arange(n) * n_devices // n).astype(np.int32)
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int32)
    return build_unstructured_partition(partition_assignment=pa, edges=edges,
                                        n_devices=n_devices)


class _Source(SimulationNode):
    """``x <- x + dt`` per cell and ``total = sum(x)``: pointwise, so written
    for the partition layout (no halo), with a domain integral.

    ``update_padded`` sums only its own cells, masking the padded rows of
    its block with the count ``shard_info["n_local"]`` -- the contract the
    wrapper documents.  ``mask=False`` sums the whole block, as a node
    could only do before the count existed.
    """

    def __init__(self, n, *, mask=True, with_total=False, stacked=False):
        super().__init__(name="src", timestep=0.1)
        self.n, self._mask, self._with_total, self._stacked = n, mask, with_total, stacked

    def initial_state(self):
        state = {"x": jnp.arange(1.0, self.n + 1.0, dtype=jnp.float32)}
        if self._with_total:
            state["total"] = jnp.float32(0.0)
        return state

    def state_fields(self):
        return ["x"]

    def domain_integral_fields(self):
        return {"total"}

    def domain_integral_axes(self):
        return {"total": ()} if self._stacked else {}

    def update(self, state, boundary_inputs, dt):
        x = state["x"] + dt
        return {"x": x, "total": jnp.sum(x)}

    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None, shard_info=None):
        _, n_local_max = shard_info[0]
        x = state_padded["x"][:n_local_max] + dt
        if self._mask:
            x_owned = jnp.where(jnp.arange(n_local_max) < shard_info["n_local"], x, 0.0)
        else:
            x_owned = x
        return {"x": x, "total": jnp.sum(x_owned)}


# ---------------------------------------------------------------------------
# H1: a Cartesian stencil node is refused
# ---------------------------------------------------------------------------


def _heat(n):
    from maddening.nodes.heat import HeatNode

    x = (np.arange(n) + 0.5) / n
    t0 = (300.0 + 40.0 * np.sin(2.3 * np.pi * x)).astype(np.float32)
    return HeatNode("rod", timestep=1.0, n_cells=n, length=1.0,
                    thermal_diffusivity=0.2 / n ** 2, initial_temperature=t0.tolist())


def _lbm():
    from maddening.nodes.lbm import LBMNode

    return LBMNode("lbm", timestep=1.0, grid_shape=(16, 8), viscosity=0.1,
                   lattice="D2Q9")


@pytest.mark.parametrize("make, n_cells, n_devices, halo", [
    (lambda: _heat(16), 16, 2, "{0: 1}"),
    (lambda: _heat(16), 16, 4, "{0: 1}"),
    (lambda: _heat(17), 17, 2, "{0: 1}"),
    (_lbm, 16, 2, "{0: 1, 1: 1}"),
], ids=["heat16-2dev", "heat16-4dev", "heat17-2dev", "lbm16x8-2dev"])
def test_a_cartesian_stencil_node_is_refused_by_the_unstructured_wrapper(
        make, n_cells, n_devices, halo):
    """Accepted before 0.4.0, a HeatNode rod stepped 43 K off on two devices
    (every cell, cell 0 never updated), an LBMNode 100% off in velocity, and
    HeatNode(17) failed with a false "n_cells was changed"."""
    node = make()
    with pytest.raises(ValueError) as exc:
        ShardedUnstructuredNode(node, create_device_mesh(shape=(n_devices,)),
                                _chain_layout(n_cells, n_devices))
    msg = str(exc.value)
    assert f"declares halo_width() == {halo}" in msg
    assert "[halo | interior | halo]" in msg and "partition layout" in msg
    assert "ShardedStencilNode" in msg
    assert "n_cells was changed" not in msg


class _PointwiseWithHalo(SimulationNode):
    """Declares a Cartesian halo, steps pointwise: reads either layout."""

    def __init__(self, opt_in):
        super().__init__(name="scale", timestep=0.1, k=0.5)
        if opt_in is not None:
            self._reads_partition_layout = opt_in

    def initial_state(self):
        return {"x": jnp.arange(1.0, 8.0, dtype=jnp.float32)}

    def halo_width(self):
        return {0: 1}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] * (1.0 - dt * self.params["k"])}

    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None, shard_info=None):
        return {"x": state_padded["x"] * (1.0 - dt * self.params["k"])}


@pytest.mark.parametrize("opt_in, accepted", [
    (True, True), (lambda: True, True),
    (None, False), (False, False), (lambda: False, False),
], ids=["attr-true", "method-true", "absent", "attr-false", "method-false"])
def test_only_an_explicit_opt_in_lets_a_node_with_a_halo_through(opt_in, accepted):
    """The private ``_reads_partition_layout`` opt-in, as a value or a
    method; anything but a true answer is refused.  An accepted node steps
    as it does unwrapped (7 cells over 2 devices: padded, ghosts unread)."""
    node = _PointwiseWithHalo(opt_in)
    mesh = create_device_mesh(shape=(2,))
    layout = _chain_layout(7, 2)
    if not accepted:
        with pytest.raises(ValueError, match=r"declares halo_width\(\) == \{0: 1\}"):
            ShardedUnstructuredNode(node, mesh, layout)
        return
    sharded = ShardedUnstructuredNode(node, mesh, layout)
    got = sharded.gather_global(sharded.update(sharded.initial_state(), {}, 0.1))["x"]
    want = node.update(node.initial_state(), {}, 0.1)["x"]
    np.testing.assert_array_equal(got, np.asarray(want))


def test_the_stencil_divisibility_refusal_does_not_point_a_stencil_node_at_the_unstructured_wrapper():
    """Its advice, followed, was the defect above: the message now says the
    unstructured wrapper is not a way out for a stencil node."""
    with pytest.raises(ValueError) as exc:
        ShardedStencilNode(_heat(17), create_device_mesh(shape=(2,)), {"devices": 0})
    msg = str(exc.value)
    assert "ShardedUnstructuredNode is not a way out for a stencil node" in msg
    assert "accepts any (device, cell) pair" not in msg
    assert "the next one up is 18" in msg and "([1])" in msg


def test_the_pointwise_divisibility_refusal_gives_advice_that_works_when_followed():
    """``ShardedPointwiseNode`` still points at the unstructured wrapper,
    which is right for a pointwise node.  Followed on 7 cells over 2
    devices, with no edges, it steps the node exactly as unwrapped."""
    node = _Source(7)
    mesh = create_device_mesh(shape=(2,))
    with pytest.raises(ValueError) as exc:
        ShardedPointwiseNode(node, mesh)
    assert "use ShardedUnstructuredNode" in str(exc.value)
    assert "edges may be empty" in str(exc.value)
    pa = (np.arange(7) * 2 // 7).astype(np.int32)
    layout = build_unstructured_partition(
        partition_assignment=pa, edges=np.zeros((0, 2), np.int32), n_devices=2)
    sharded = ShardedUnstructuredNode(node, mesh, layout)
    out = sharded.update(sharded.initial_state(), {}, 0.1)
    ref = node.update(node.initial_state(), {}, 0.1)
    np.testing.assert_array_equal(sharded.gather_global(out)["x"], np.asarray(ref["x"]))
    assert float(out["total"]) == pytest.approx(float(ref["total"]), rel=1e-6)


# ---------------------------------------------------------------------------
# M5: the node's cell count is the layout's, and shard_info says which rows
# of a block are padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_node", [10, 6], ids=["more-cells", "fewer-cells"])
def test_a_node_whose_cell_count_is_not_the_layouts_is_refused_at_construction(n_node):
    """A 10-cell node on an 8-cell layout lost cells 8 and 9 (its integral
    32.4 where the node's is 49.5); a 6-cell one failed later with a bare
    IndexError from inside ``initial_state``."""
    with pytest.raises(ValueError) as exc:
        ShardedUnstructuredNode(_Source(n_node), create_device_mesh(shape=(2,)),
                                _chain_layout(8, 2))
    msg = str(exc.value)
    assert f"state field 'x' has shape ({n_node},), {n_node} rows" in msg
    assert "the layout partitions 8 cells" in msg


class _ZeroDField(_Source):
    """A 0-d state field that is not declared a domain integral."""

    def initial_state(self):
        return {**super().initial_state(), "clock": jnp.float32(0.0)}


def test_a_zero_d_field_that_is_not_a_domain_integral_is_refused_by_name():
    with pytest.raises(ValueError, match=r"state field 'clock' has shape \(\), no cell axis"):
        ShardedUnstructuredNode(_ZeroDField(8), create_device_mesh(shape=(2,)),
                                _chain_layout(8, 2))


@pytest.mark.parametrize("value", [np.arange(10.0), np.arange(6.0), np.float32(1.0),
                                   np.zeros((10, 3))],
                         ids=["10-rows", "6-rows", "0-d", "10x3"])
def test_partition_value_refuses_a_value_that_is_not_one_row_per_layout_cell(value):
    """The backstop under the construction check, for every other caller."""
    with pytest.raises(ValueError, match="the layout partitions 8 cells"):
        partition_value(value=value, layout=_chain_layout(8, 2))


def test_partition_value_still_pads_a_value_with_one_row_per_cell():
    out = partition_value(value=np.arange(7.0), layout=_chain_layout(7, 2), pad_value=-1.0)
    assert out.tolist() == [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, -1.0]]


def test_shard_info_carries_each_shards_own_cell_count():
    """7 cells on 2 devices own (4, 3); ``n_local`` is that count per shard,
    read back as a per-shard (unreduced) integral."""

    class Count(_Source):
        def domain_integral_axes(self):
            return {"total": ()}

        def update_padded(self, state_padded, boundary_inputs, dt, *,
                          static_padded=None, shard_info=None):
            _, n_local_max = shard_info[0]
            return {"x": state_padded["x"][:n_local_max],
                    "total": shard_info["n_local"].astype(jnp.float32)}

    sharded = ShardedUnstructuredNode(Count(7), create_device_mesh(shape=(2,)),
                                      _chain_layout(7, 2))
    out = sharded.update(sharded.initial_state(), {}, 0.1)
    assert np.asarray(out["total"]).tolist() == [4.0, 3.0]


def test_a_node_that_masks_with_the_count_integrates_only_its_cells():
    """The integral on an uneven partition equals the unsharded one after
    ten steps (35.0), where summing the padded block gives 36.0: the
    padded slot of the short shard grows by ``dt`` a step like a cell."""
    mesh = create_device_mesh(shape=(2,))
    layout = _chain_layout(7, 2)
    ref_node = _Source(7)
    ref = ref_node.initial_state()
    results = {}
    for mask in (True, False):
        sharded = ShardedUnstructuredNode(_Source(7, mask=mask), mesh, layout)
        st = sharded.initial_state()
        for _ in range(10):
            st = sharded.update({"x": st["x"]}, {}, 0.1)
        results[mask] = float(st["total"])
    for _ in range(10):
        ref = ref_node.update({"x": ref["x"]}, {}, 0.1)
    assert results[True] == pytest.approx(float(ref["total"]), rel=1e-6)
    assert results[True] == pytest.approx(35.0, rel=1e-6)
    # What the count is for: without it the padding is integrated.
    assert results[False] == pytest.approx(36.0, rel=1e-6)


# ---------------------------------------------------------------------------
# L6: a domain integral carried in the state is replicated, not partitioned
# ---------------------------------------------------------------------------


def test_a_domain_integral_fed_back_as_state_steps_again():
    """Step 1's output, fed back, used to fail: the integral key was given
    ``P(mesh_axis)``, "too long" for a scalar."""
    sharded = ShardedUnstructuredNode(_Source(8), create_device_mesh(shape=(2,)),
                                      _chain_layout(8, 2))
    st = sharded.update(sharded.initial_state(), {}, 0.1)
    st = sharded.update(st, {}, 0.1)
    assert float(st["total"]) == pytest.approx(36.0 + 8 * 0.2, rel=1e-6)


def _graph(node):
    gm = GraphManager()
    gm.add_node(node)
    gm.compile()
    return gm


@pytest.mark.parametrize("stacked", [False, True], ids=["reduced", "per-shard"])
def test_a_node_declaring_its_integral_runs_in_a_graph_like_the_unsharded_node(stacked):
    """With the integral in ``initial_state`` a graph's scan carry keeps its
    structure; wrapped, ``initial_state`` used to partition the scalar and
    raise ``IndexError``.  A per-shard integral's initial value is stacked
    along the mesh axis, as the step returns it."""
    mesh = create_device_mesh(shape=(2,))
    layout = _chain_layout(8, 2)
    ref = _graph(_Source(8, with_total=True))
    ref.run_scan(3)
    gm = _graph(ShardedUnstructuredNode(_Source(8, with_total=True, stacked=stacked),
                                        mesh, layout))
    gm.run_scan(3)
    total = np.asarray(gm.get_node_state("src")["total"])
    want = float(np.asarray(ref.get_node_state("src")["total"]))
    if stacked:
        assert total.shape == (2,)
        # blocks [1..4] and [5..8], each cell + 0.3
        np.testing.assert_allclose(total, [10.0 + 1.2, 26.0 + 1.2], rtol=1e-6)
        assert float(total.sum()) == pytest.approx(want, rel=1e-6)
    else:
        assert total.shape == ()
        assert float(total) == pytest.approx(want, rel=1e-6)
    gm.step()
    assert np.all(np.isfinite(np.asarray(gm.get_node_state("src")["total"])))


def test_a_node_without_an_initial_integral_steps_in_a_graph():
    """``gm.step()`` feeds step 1's output (now carrying the integral) back
    in: three steps, the integral of the last."""
    gm = _graph(ShardedUnstructuredNode(_Source(8), create_device_mesh(shape=(2,)),
                                        _chain_layout(8, 2)))
    for _ in range(3):
        gm.step()
    assert float(np.asarray(gm.get_node_state("src")["total"])) == pytest.approx(
        36.0 + 8 * 0.3, rel=1e-6)
