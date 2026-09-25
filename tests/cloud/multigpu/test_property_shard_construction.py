"""What a sharded wrapper promises before the first step runs.

Two construction-time invariants, both from the cloud/sharding audit of
2026-09-19 (report and reproducers under
``benchmarks/results/audit_040_final/cloud-sharding/``):

* **the wrapper shards the axis it was given.**
  ``ShardedPointwiseNode`` built its ``NamedSharding`` from
  ``P("devices")``, which names array axis 0 whatever ``shard_axes``
  said, and consulted ``shard_axes`` only as a rank test.  Every test in
  the suite passed ``shard_axes=(0,)``, so the argument looked honoured.
  ``shard_axes=(1,)`` on a ``(3, 8)`` state over 4 devices failed with an
  ``IndivisibleError`` blaming axis 0 -- an axis the caller had ruled out.

* **construction either succeeds or explains itself.**  A pencil
  decomposition splits each sharded axis evenly, so
  ``ShardedStencilNode`` cannot take 17 cells over 3 devices.  It used to
  discover that inside ``device_put``, on the first ``initial_state`` or
  the first step, as a ``jax.errors.IndivisibleError`` naming neither the
  node, the cell count nor the device count.  The limitation is specific
  to the stencil path: ``ShardedUnstructuredNode`` carries an explicit
  padded layout and takes any pair, which the last property here pins so
  the two paths cannot quietly converge.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import (
    ShardedPointwiseNode,
    ShardedStencilNode,
)
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.cloud.multigpu.property_support import (
    DEVICE_COUNTS,
    StencilDiffusion1D,
    build_unstructured,
    device_counts,
)
from tests.conftest import EXAMPLES_COSTLY, EXAMPLES_STANDARD


class _Pointwise2D(SimulationNode):
    """Pointwise decay on a 2-D state, so a shard axis can be chosen.

    ``x <- x + rate * dt * (source - x)`` elementwise.  Two axes of
    different extents is the whole point: a wrapper that ignores
    ``shard_axes`` shards the wrong one, and with different extents that
    is visible in the partition spec rather than only in the timings.
    """

    def __init__(self, name: str = "p2d", rows: int = 4, cols: int = 8,
                 rate: float = 0.5, timestep: float = 0.1) -> None:
        super().__init__(name=name, timestep=timestep, rate=float(rate),
                         rows=int(rows), cols=int(cols))

    def halo_width(self) -> dict[int, int]:
        return {}

    def state_fields(self) -> list[str]:
        return ["x"]

    def initial_state(self) -> dict:
        rows, cols = int(self.params["rows"]), int(self.params["cols"])
        return {"x": jnp.asarray(
            np.arange(rows * cols, dtype=np.float32).reshape(rows, cols))}

    def boundary_input_spec(self) -> dict:
        return {"source": BoundaryInputSpec(
            shape=(int(self.params["rows"]), int(self.params["cols"])),
            description="per-cell target", expected_units="1")}

    def update(self, state, boundary_inputs, dt, *, params=None) -> dict:
        p = self.params if params is None else {**self.params, **params}
        source = boundary_inputs.get("source", jnp.zeros_like(state["x"]))
        return {"x": state["x"] + p["rate"] * dt * (source - state["x"])}


def _spec_of(array) -> tuple:
    """The array's partition spec, one entry per axis.

    ``PartitionSpec`` drops trailing ``None``\\ s, so ``P("devices")`` and
    ``P("devices", None)`` are the same object; padding to the array's
    rank makes "which axis is sharded" a direct comparison.
    """
    spec = tuple(array.sharding.spec)
    return spec + (None,) * (jnp.ndim(array) - len(spec))


# ---------------------------------------------------------------------------
# The wrapper shards the axis it was given
# ---------------------------------------------------------------------------


@given(n_devices=device_counts(), shard_axis=st.integers(min_value=0, max_value=1),
       cells_per_shard=st.integers(min_value=1, max_value=3))
@settings(max_examples=EXAMPLES_COSTLY)
def test_a_pointwise_wrapper_shards_the_axis_it_was_given(
        n_devices, shard_axis, cells_per_shard):
    """The mesh axis lands at ``shard_axes[0]``; every other axis replicates.

    Stated on the placement rather than on a timing so it holds at one
    device too, where every layout computes the same answer.
    """
    sharded_extent = n_devices * cells_per_shard
    other_extent = sharded_extent + 1          # deliberately not divisible
    rows, cols = ((sharded_extent, other_extent) if shard_axis == 0
                  else (other_extent, sharded_extent))
    mesh = create_device_mesh(shape=(n_devices,))
    node = _Pointwise2D(rows=rows, cols=cols)
    wrapper = ShardedPointwiseNode(node, mesh, shard_axes=(shard_axis,))

    state = wrapper.initial_state()
    expected = tuple("devices" if axis == shard_axis else None
                     for axis in range(2))
    assert _spec_of(state["x"]) == expected
    # ...and the state itself is the node's, wherever its shards live.
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(state["x"])),
        np.asarray(node.initial_state()["x"]))


@given(n_devices=device_counts(), shard_axis=st.integers(min_value=0, max_value=1),
       remainder=st.integers(min_value=1, max_value=3))
@settings(max_examples=EXAMPLES_COSTLY)
def test_a_pointwise_wrapper_refuses_an_axis_the_mesh_cannot_split(
        n_devices, shard_axis, remainder):
    """An indivisible axis is refused at construction, naming both numbers.

    And it names the axis the *caller* chose: the old wrapper always
    reported axis 0.
    """
    assume(n_devices > 1 and remainder < n_devices)
    bad_extent = n_devices + remainder
    good_extent = 2 * n_devices
    rows, cols = ((bad_extent, good_extent) if shard_axis == 0
                  else (good_extent, bad_extent))
    mesh = create_device_mesh(shape=(n_devices,))

    with pytest.raises(ValueError) as excinfo:
        ShardedPointwiseNode(_Pointwise2D(rows=rows, cols=cols), mesh,
                             shard_axes=(shard_axis,))
    message = str(excinfo.value)
    assert f"axis {shard_axis}" in message
    assert f"{bad_extent} cells" in message
    assert f"{n_devices} devices" in message


def test_a_pointwise_wrapper_refuses_a_mesh_without_a_devices_axis():
    """A mesh whose axis is named something else is refused by name."""
    mesh = create_device_mesh(shape=(1,), axis_names=("x",))
    with pytest.raises(ValueError, match="devices"):
        ShardedPointwiseNode(_Pointwise2D(rows=2, cols=2), mesh)


# ---------------------------------------------------------------------------
# Construction either succeeds or explains itself
# ---------------------------------------------------------------------------


@given(n_devices=device_counts(), n_cells=st.integers(min_value=2, max_value=24))
@settings(max_examples=EXAMPLES_STANDARD)
def test_stencil_construction_either_succeeds_or_says_why_it_cannot(
        n_devices, n_cells):
    """Constructible exactly when the cell count divides by the devices.

    The failure has to be the validation error -- naming the node, the
    cell count and the device count -- and not an ``IndivisibleError``
    from inside ``device_put`` on some later call.
    """
    mesh = create_device_mesh(shape=(n_devices,))
    node = StencilDiffusion1D(name="diff", n_cells=n_cells)

    if n_cells % n_devices == 0:
        wrapper = ShardedStencilNode(node, mesh, axis_map={"devices": 0},
                                     boundary="periodic")
        assert _spec_of(wrapper.initial_state()["f"]) == ("devices",)
        return

    with pytest.raises(ValueError) as excinfo:
        ShardedStencilNode(node, mesh, axis_map={"devices": 0},
                           boundary="periodic")
    message = str(excinfo.value)
    # Our validation, not JAX's ``IndivisibleError`` from device_put
    # (which is a ValueError too, and says none of the following).
    assert type(excinfo.value) is ValueError
    assert "diff" in message
    assert f"{n_cells} cells" in message
    assert f"{n_devices} devices" in message
    # The ways out are named, so the message is actionable -- and the
    # unstructured wrapper is named as *not* one for a stencil node: it
    # used to be recommended here, and stepped a stencil node wrong.
    assert f"the next one up is {(n_cells // n_devices + 1) * n_devices}" in message
    assert "ShardedUnstructuredNode is not a way out for a stencil node" in message


@pytest.mark.parametrize("n_devices", DEVICE_COUNTS)
def test_the_unstructured_wrapper_takes_a_cell_count_the_stencil_one_refuses(
        n_devices):
    """The divisibility rule is the stencil path's, not the framework's.

    The unstructured wrapper carries an explicit padded layout, so the
    ragged case that stops ``ShardedStencilNode`` at construction runs
    there and agrees with the unsharded node.
    """
    n_cells = 17                                   # prime: divides by 1 only
    assume_divisible = n_cells % n_devices == 0
    case = build_unstructured(n_devices=n_devices, n_cells=n_cells)
    sharded = case.run(steps=2, sharded=True)
    reference = case.run(steps=2, sharded=False)
    for field, value in reference.items():
        np.testing.assert_allclose(sharded[field], value, rtol=1e-6, atol=1e-6,
                                   err_msg=f"{field} at {n_devices} devices")

    if not assume_divisible and n_devices > 1:
        mesh = create_device_mesh(shape=(n_devices,))
        with pytest.raises(ValueError, match=f"{n_cells} cells"):
            ShardedStencilNode(StencilDiffusion1D(name="diff", n_cells=n_cells),
                               mesh, axis_map={"devices": 0})
