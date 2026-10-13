"""``shard_info`` has one string key, ``"n_local"``, and only one wrapper passes it.

``SimulationNode.update_padded`` receives ``shard_info`` as ``{int axis:
(offset, extent)}``.  0.4.0 added ``"n_local"`` -- a traced scalar, this
shard's own cell count -- from ``ShardedUnstructuredNode`` only, so a node
that iterates ``shard_info.items()`` and unpacks each value as a tuple fails
under that wrapper.  The docstring's idiom iterates over the integer keys only.

One probe, written with that idiom, runs under all three sharded wrappers.
The key sets it records pin what each wrapper passes: a new non-``int`` key,
or ``"n_local"`` turning up under another wrapper, fails here instead of in a
user's node.  The last test pins the caveat itself: the naive loop does fail
under the unstructured wrapper, so the documentation is not warning about
nothing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
from maddening.cloud.multigpu.sharded_node import (
    ShardedPointwiseNode,
    ShardedStencilNode,
)
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from maddening.core.node import SimulationNode

N_CELLS = 16
N_DEVICES = 4

pytestmark = pytest.mark.skipif(
    len(jax.devices()) < N_DEVICES,
    reason=f"needs {N_DEVICES} JAX devices (tests/cloud/multigpu/conftest.py "
           "forces 16 virtual CPU devices unless an accelerator is selected)",
)


def integer_axes(shard_info):
    """The documented idiom: the ``int`` keys of *shard_info*, i.e. its axes."""
    return (k for k in shard_info if isinstance(k, int))


class _KeyProbe(SimulationNode):
    """A 1-D node that records the ``shard_info`` keys each trace receives.

    ``halo=True`` makes it a stencil node (``ShardedStencilNode`` requires a
    halo); ``halo=False`` a pointwise one (the other two wrappers refuse a
    halo).  Its step leaves ``x`` unchanged: what is under test is the
    argument, not the physics.
    """

    def __init__(self, name: str, *, halo: bool, naive_loop: bool = False):
        super().__init__(name=name, timestep=1.0)
        self._halo = halo
        self._naive_loop = naive_loop
        #: One key set per ``update_padded`` trace; empty if never called.
        self.seen: list[frozenset] = []
        #: The ``(offset, extent)`` pairs the idiom unpacked, per trace.
        self.unpacked: list[dict] = []

    def halo_width(self) -> dict[int, int]:
        return {0: 1} if self._halo else {}

    def state_fields(self) -> list[str]:
        return ["x"]

    def initial_state(self) -> dict:
        return {"x": jnp.arange(N_CELLS, dtype=jnp.float32) + 1.0}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"]}

    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None, shard_info=None):
        if shard_info is not None:
            self.seen.append(frozenset(shard_info))
            if self._naive_loop:
                pairs = {axis: (offset, extent)
                         for axis, (offset, extent) in shard_info.items()}
            else:
                pairs = {}
                for axis in integer_axes(shard_info):
                    offset, extent = shard_info[axis]
                    pairs[axis] = (offset, extent)
            self.unpacked.append(pairs)
        return {"x": state_padded["x"]}


def _mesh():
    return create_device_mesh(shape=(N_DEVICES,))


def _stencil(probe):
    return ShardedStencilNode(probe, _mesh(), axis_map={"devices": 0},
                              boundary="edge")


def _pointwise(probe):
    return ShardedPointwiseNode(probe, _mesh())


def _unstructured(probe):
    # A contiguous partition in global order: four full shards of four cells.
    pa = np.repeat(np.arange(N_DEVICES), N_CELLS // N_DEVICES).astype(np.int32)
    edges = np.array([[i, (i + 1) % N_CELLS] for i in range(N_CELLS)],
                     dtype=np.int32)
    layout = build_unstructured_partition(
        partition_assignment=pa, edges=edges, n_devices=N_DEVICES)
    return ShardedUnstructuredNode(probe, _mesh(), layout)


WRAPPERS = {
    "stencil": (_stencil, True),
    "pointwise": (_pointwise, False),
    "unstructured": (_unstructured, False),
}


def _step(wrapper_name, *, naive_loop=False):
    build, halo = WRAPPERS[wrapper_name]
    probe = _KeyProbe("probe", halo=halo, naive_loop=naive_loop)
    sharded = build(probe)
    out = sharded.update(sharded.initial_state(), {}, 1.0)
    jax.block_until_ready(out)
    return probe


@pytest.fixture(scope="module")
def probes():
    """Each wrapper stepped once with the idiomatic probe."""
    return {name: _step(name) for name in WRAPPERS}


def _non_int_keys(probe):
    return {k for keys in probe.seen for k in keys if not isinstance(k, int)}


def test_n_local_is_the_only_non_int_key_any_wrapper_passes(probes):
    non_int = set().union(*(_non_int_keys(p) for p in probes.values()))
    assert non_int == {"n_local"}, non_int


def test_n_local_arrives_only_under_the_unstructured_wrapper(probes):
    assert probes["unstructured"].seen, "update_padded was never traced"
    assert all("n_local" in keys for keys in probes["unstructured"].seen)
    for name in ("stencil", "pointwise"):
        assert _non_int_keys(probes[name]) == set(), name


def test_the_stencil_wrapper_passes_only_its_sharded_axis(probes):
    seen = probes["stencil"].seen
    assert seen and all(keys == frozenset({0}) for keys in seen), seen


def test_the_pointwise_wrapper_never_calls_update_padded(probes):
    # It steps the inner node's update(), so no shard_info exists to carry
    # "n_local" -- recorded rather than assumed, since the claim above rests
    # on it.
    assert probes["pointwise"].seen == []


def test_the_idiom_unpacks_an_offset_and_extent_for_every_axis(probes):
    for name in ("stencil", "unstructured"):
        for pairs in probes[name].unpacked:
            assert set(pairs) == {0}, (name, pairs)
            offset, extent = pairs[0]
            assert extent == N_CELLS // N_DEVICES, (name, extent)


def test_the_naive_items_loop_fails_under_the_unstructured_wrapper():
    """The caveat is real: unpacking every value as a tuple meets n_local."""
    with pytest.raises((TypeError, ValueError)):
        _step("unstructured", naive_loop=True)
    # ... and not under the stencil wrapper, which passes int keys only.
    _step("stencil", naive_loop=True)
