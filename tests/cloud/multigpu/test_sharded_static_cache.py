"""Sharded statics are materialised once, not once per ``update``.

The sharded wrappers used to re-partition and re-copy every sharded
``StaticArray`` on every public ``update`` call: one ``device_put`` per
array per frame for :class:`ShardedStencilNode`, and a full
device->host->partition->host->device round trip for
:class:`ShardedUnstructuredNode`.  None of that depends on the state, so
on the interactive path (a slider, an HTTP request, a Python loop) it
was pure per-frame host overhead.

These tests assert *counted* quantities -- how many transfers happen --
rather than durations, and pin the invalidation paths: the cache must
never serve a stale static.  What invalidates it:

* the inner node handing back a different array object (a derived
  static rebuilt after a parameter write, a ``static_data_provider``
  reconstruction after a restore, a different node under
  ``replace_node``);
* a change to the set of sharded keys or their ``shard_axis``;
* an explicit :meth:`invalidate_static_cache` call, which is the escape
  hatch for a node that rewrites a static's buffer in place.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import (
    build_unstructured_partition,
)
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.cloud.multigpu.sharded_unstructured import (
    ShardedUnstructuredNode,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.static_data import StaticArray

_HAS_4_DEVICES = len(jax.devices()) >= 4
_SKIP_4 = "Requires >=4 JAX devices"

N_CELLS = 16
N_DEVICES = 4


# ---------------------------------------------------------------------------
# Counting helpers -- deterministic, unlike wall-clock on a shared machine.
# ---------------------------------------------------------------------------


class _Transfers:
    """Counts of the host<->device transfers a block of code performs."""

    def __init__(self) -> None:
        self.put = 0
        self.get = 0


@contextlib.contextmanager
def count_transfers(monkeypatch):
    """Count ``jax.device_put`` / ``jax.device_get`` calls in the block."""
    counts = _Transfers()
    real_put, real_get = jax.device_put, jax.device_get

    def put(*args, **kwargs):
        counts.put += 1
        return real_put(*args, **kwargs)

    def get(*args, **kwargs):
        counts.get += 1
        return real_get(*args, **kwargs)

    monkeypatch.setattr(jax, "device_put", put)
    monkeypatch.setattr(jax, "device_get", get)
    try:
        yield counts
    finally:
        monkeypatch.setattr(jax, "device_put", real_put)
        monkeypatch.setattr(jax, "device_get", real_get)


# ---------------------------------------------------------------------------
# Fixture node: a 1-D diffusion whose mask is a *derived* sharded static.
# ---------------------------------------------------------------------------


class ScaledMaskDiffusion1D(SimulationNode):
    """1-D diffusion modulated by a per-cell sharded mask.

    The mask is derived from ``self.params["mask_scale"]`` and cached on
    the instance, so the ``StaticArray``'s ``value`` changes identity
    exactly when the scale does.  That is the realistic shape of a
    static that depends on a parameter (a remeshed geometry, a wall mask
    rebuilt for a new porosity), and it is what makes the identity-keyed
    cache correct: a parameter write is not cached past.
    """

    def __init__(self, name: str, n: int, alpha: float, mask_scale: float = 1.0):
        super().__init__(name=name, timestep=0.01, mask_scale=float(mask_scale))
        self._n = int(n)
        self._alpha = float(alpha)
        self._base = np.linspace(0.5, 1.5, int(n)).astype(np.float32)
        self._built_scale: float | None = None
        self._mask = jnp.asarray(self._base)
        self.static_rebuilds = 0

    # -- the derived static -------------------------------------------

    def _current_mask(self):
        scale = float(self.params["mask_scale"])
        if scale != self._built_scale:
            self._mask = jnp.asarray(self._base * scale)
            self._built_scale = scale
            self.static_rebuilds += 1
        return self._mask

    @property
    def static_data(self) -> dict:
        return {
            "mask": StaticArray(
                value=self._current_mask(), replication="shard", shard_axis=0,
            ),
        }

    # -- node contract -------------------------------------------------

    def halo_width(self) -> dict[int, int]:
        return {0: 1}

    def state_fields(self) -> list[str]:
        return ["f"]

    def initial_state(self) -> dict:
        rng = np.random.default_rng(7)
        return {"f": jnp.asarray(rng.standard_normal(self._n).astype(np.float32))}

    def update(self, state, boundary_inputs, dt):
        f = state["f"]
        f_pad = jnp.pad(f, 1, mode="edge")
        lap = f_pad[2:] - 2 * f_pad[1:-1] + f_pad[:-2]
        return {"f": f + self._alpha * self._current_mask() * lap * dt}

    def update_padded(
        self, state_padded, boundary_inputs, dt, *,
        static_padded=None, shard_info=None,
    ):
        f_pad = state_padded["f"]
        m_pad = static_padded["mask"]
        f = f_pad[1:-1]
        lap = f_pad[2:] - 2 * f_pad[1:-1] + f_pad[:-2]
        f_new = f + self._alpha * m_pad[1:-1] * lap * dt
        return {"f": jnp.pad(f_new, 1, mode="edge")}


class DecayWithPartitionedMass(SimulationNode):
    """Cell-local decay with a ``replication="partition"`` static.

    The unstructured wrapper has to partition ``mass`` through the
    layout (a device->host copy, a NumPy gather, a host->device copy)
    before it can call ``update_padded``; this node exists to count
    how often that happens.
    """

    def __init__(self, name: str, partition_assignment: np.ndarray):
        super().__init__(name=name, timestep=1.0)
        self._pa = np.asarray(partition_assignment)
        self._n = int(self._pa.shape[0])
        self._mass = np.arange(self._n, dtype=np.float32) + 1.0

    @property
    def static_data(self) -> dict:
        return {
            "mass": StaticArray(
                self._mass, replication="partition",
                partition_assignment=self._pa,
            ),
        }

    def state_fields(self) -> list[str]:
        return ["x"]

    def initial_state(self) -> dict:
        return {"x": jnp.asarray(
            np.arange(self._n, dtype=np.float32) + 1.0
        )}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] * 0.5}

    def update_padded(
        self, state_padded, boundary_inputs, dt, *,
        static_padded=None, shard_info=None,
    ):
        # ``mass`` rides along so the partitioned-static path is exercised.
        return {"x": state_padded["x"] * 0.5 + 0.0 * static_padded["mass"]}


def _stencil_wrapper(mask_scale: float = 1.0):
    inner = ScaledMaskDiffusion1D("diff", n=N_CELLS, alpha=0.1, mask_scale=mask_scale)
    mesh = create_device_mesh(shape=(N_DEVICES,))
    return inner, ShardedStencilNode(
        inner, mesh, axis_map={"devices": 0}, boundary="edge",
    )


def _drive(wrapper, state, n_steps):
    for _ in range(n_steps):
        state = {"f": wrapper.update(state, {}, 0.01)["f"]}
    return state


# ---------------------------------------------------------------------------
# The counted invariant.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_stencil_sharded_static_is_device_put_once_not_once_per_update(monkeypatch):
    """One ``device_put`` per sharded static, not one per frame."""
    _, wrapper = _stencil_wrapper()
    state = wrapper.initial_state()
    state = _drive(wrapper, state, 1)          # warm the compile cache

    with count_transfers(monkeypatch) as counts:
        _drive(wrapper, state, 20)
    assert counts.put == 0, (
        "the sharded mask was re-copied to device during steady-state "
        f"updates ({counts.put} device_put calls over 20 frames)"
    )


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_unstructured_partitioned_static_is_transferred_once(monkeypatch):
    """The partition round trip happens once, not once per frame."""
    pa = (np.arange(N_CELLS) % N_DEVICES).astype(np.int32)
    edges = np.array(
        [[i, (i + 1) % N_CELLS] for i in range(N_CELLS)], dtype=np.int32,
    )
    layout = build_unstructured_partition(
        partition_assignment=pa, edges=edges, n_devices=N_DEVICES,
    )
    mesh = create_device_mesh(shape=(N_DEVICES,))
    wrapper = ShardedUnstructuredNode(
        DecayWithPartitionedMass("toy", pa), mesh, layout,
    )
    state = wrapper.initial_state()
    state = {"x": wrapper.update(state, {}, 1.0)["x"]}

    with count_transfers(monkeypatch) as counts:
        for _ in range(20):
            state = {"x": wrapper.update(state, {}, 1.0)["x"]}
    assert (counts.put, counts.get) == (0, 0), (
        "the partitioned static was re-partitioned during steady-state "
        f"updates (device_put={counts.put}, device_get={counts.get})"
    )


# ---------------------------------------------------------------------------
# Invalidation.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_a_rebuilt_static_array_is_not_served_from_the_cache(monkeypatch):
    """A new array object from the node is picked up on the next update."""
    inner, wrapper = _stencil_wrapper()
    state = wrapper.initial_state()
    first = wrapper.update(state, {}, 0.01)["f"]

    inner.params["mask_scale"] = 3.0          # rebuilds the derived mask
    with count_transfers(monkeypatch) as counts:
        second = wrapper.update(state, {}, 0.01)["f"]
    assert counts.put >= 1, "the rebuilt mask was never copied to device"
    assert inner.static_rebuilds == 2
    assert not np.allclose(np.asarray(first), np.asarray(second)), (
        "the update still used the pre-write mask"
    )


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_invalidate_static_cache_forces_rematerialisation(monkeypatch):
    """The explicit hook re-copies a static whose buffer changed in place."""
    _, wrapper = _stencil_wrapper()
    state = wrapper.initial_state()
    _drive(wrapper, state, 1)

    with count_transfers(monkeypatch) as counts:
        wrapper.update(state, {}, 0.01)
        assert counts.put == 0
        wrapper.invalidate_static_cache()
        wrapper.update(state, {}, 0.01)
    assert counts.put == 1


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_param_write_through_the_rest_layer_reaches_a_derived_static():
    """``PUT /graph/params`` on a sharded node is not cached past."""
    pytest.importorskip(
        "fastapi", reason="the REST layer is the optional [api] extra",
    )
    from fastapi.testclient import TestClient

    from maddening.api.server import SimulationServer

    inner, wrapper = _stencil_wrapper()
    gm = GraphManager()
    gm.add_node(wrapper)
    gm.compile()
    gm.step()
    before = np.asarray(gm.get_node_state("diff")["f"]).copy()

    client = TestClient(
        SimulationServer(node_registry={}, graph_manager=gm).create_app(),
        raise_server_exceptions=False,
    )
    resp = client.put("/graph/params/diff", json={"params": {"mask_scale": 5.0}})
    assert resp.status_code == 200, resp.text
    # The wrapper shares the inner node's params dict, so the write lands
    # where the derived static reads it.
    assert float(inner.params["mask_scale"]) == 5.0

    gm.set_node_state("diff", {"f": jnp.asarray(before)})
    gm.step()
    after = np.asarray(gm.get_node_state("diff")["f"])

    gm2 = GraphManager()
    _, reference_wrapper = _stencil_wrapper(mask_scale=5.0)
    gm2.add_node(reference_wrapper)
    gm2.compile()
    gm2.set_node_state("diff", {"f": jnp.asarray(before)})
    gm2.step()
    expected = np.asarray(gm2.get_node_state("diff")["f"])
    np.testing.assert_allclose(after, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_checkpoint_restore_is_not_served_from_the_static_cache(tmp_path):
    """A restore rewinds the trajectory exactly, cache or no cache."""
    _, wrapper = _stencil_wrapper()
    gm = GraphManager()
    gm.add_node(wrapper)
    gm.compile()
    for _ in range(3):
        gm.step()
    gm.save_state(tmp_path / "snap.npz")
    expected = np.asarray(gm.get_node_state("diff")["f"]).copy()

    for _ in range(5):
        gm.step()
    assert not np.allclose(np.asarray(gm.get_node_state("diff")["f"]), expected)

    gm.load_state(tmp_path / "snap.npz")
    np.testing.assert_allclose(
        np.asarray(gm.get_node_state("diff")["f"]), expected, rtol=0, atol=0,
    )
    gm.step()
    assert np.all(np.isfinite(np.asarray(gm.get_node_state("diff")["f"])))


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_replace_node_brings_its_own_static(monkeypatch):
    """A replacement wrapper's static is used, not the old one's."""
    from maddening.surrogates.replace import replace_node

    _, wrapper = _stencil_wrapper(mask_scale=1.0)
    gm = GraphManager()
    gm.add_node(wrapper)
    gm.compile()
    gm.step()
    start = np.asarray(gm.get_node_state("diff")["f"]).copy()
    gm.step()
    old_next = np.asarray(gm.get_node_state("diff")["f"]).copy()

    _, replacement = _stencil_wrapper(mask_scale=4.0)
    replace_node(gm, "diff", replacement)
    gm.compile()
    gm.set_node_state("diff", {"f": jnp.asarray(start)})
    gm.step()
    new_next = np.asarray(gm.get_node_state("diff")["f"])
    assert not np.allclose(old_next, new_next), (
        "the replacement node's mask never reached the update"
    )


# ---------------------------------------------------------------------------
# compile() is the framework's "rebuild everything"; it has to reach the
# one change the identity key cannot see.  These two run on a single
# device so they are not skipped on a 1-CPU CI runner.
# ---------------------------------------------------------------------------


class InPlaceMaskDiffusion1D(SimulationNode):
    """1-D diffusion whose sharded mask is a NumPy buffer.

    Unlike :class:`ScaledMaskDiffusion1D` the buffer is writable, so a
    test can rewrite it *in place* -- the one change
    ``ShardedStencilNode``'s identity-keyed static cache cannot see.
    """

    def __init__(self, name: str, n: int):
        super().__init__(name=name, timestep=0.01)
        self._n = int(n)
        self._mask = np.ones(int(n), dtype=np.float32)

    @property
    def static_data(self) -> dict:
        return {
            "mask": StaticArray(
                value=self._mask, replication="shard", shard_axis=0,
            ),
        }

    def halo_width(self) -> dict[int, int]:
        return {0: 1}

    def state_fields(self) -> list[str]:
        return ["f"]

    def initial_state(self) -> dict:
        return {"f": jnp.asarray(
            np.linspace(0.0, 1.0, self._n).astype(np.float32)
        )}

    def update(self, state, boundary_inputs, dt):
        f = state["f"]
        f_pad = jnp.pad(f, 1, mode="edge")
        lap = f_pad[2:] - 2 * f_pad[1:-1] + f_pad[:-2]
        return {"f": f + 0.1 * jnp.asarray(self._mask) * lap * dt}

    def update_padded(
        self, state_padded, boundary_inputs, dt, *,
        static_padded=None, shard_info=None,
    ):
        f_pad = state_padded["f"]
        m_pad = static_padded["mask"]
        lap = f_pad[2:] - 2 * f_pad[1:-1] + f_pad[:-2]
        f_new = f_pad[1:-1] + 0.1 * m_pad[1:-1] * lap * dt
        return {"f": jnp.pad(f_new, 1, mode="edge")}


def test_compile_rematerialises_a_static_rewritten_in_place():
    """``gm.compile()`` must not bake in the previous static buffer.

    The per-device placement is cached on the *node*, keyed on the array's
    identity, so a buffer rewritten in place is invisible to it.  That is
    the accepted trade-off for a steady-state ``update``, but ``compile()``
    is the framework's explicit "throw everything away and rebuild": it
    rebuilds the step, clears the scan cache and re-snapshots the
    static-data hashes, so it has to drop the materialised statics too.
    Otherwise the freshly traced step closes over the old buffer and every
    later step is silently wrong.
    """
    inner = InPlaceMaskDiffusion1D("d", n=N_CELLS)
    mesh = create_device_mesh(shape=(1,))
    wrapper = ShardedStencilNode(
        inner, mesh, axis_map={"devices": 0}, boundary="edge",
    )
    gm = GraphManager()
    gm.add_node(wrapper)
    gm.compile()
    gm.step()
    start = np.asarray(gm.get_node_state("d")["f"]).copy()

    # A mask of zeros freezes the field: the step becomes the identity.
    inner._mask[:] = 0.0
    gm.compile()
    gm.set_node_state("d", {"f": jnp.asarray(start)})
    gm.step()

    np.testing.assert_allclose(
        np.asarray(gm.get_node_state("d")["f"]), start, rtol=0, atol=0,
        err_msg="compile() traced the step against the pre-rewrite mask",
    )


def test_compile_invalidates_every_node_static_cache():
    """The hook is called for each node that offers it.

    Pins the contract itself (rather than one wrapper's behaviour), so a
    future node that caches a materialisation gets the same treatment.
    """
    class _Recorder(SimulationNode):
        def __init__(self) -> None:
            super().__init__(name="rec", timestep=0.1)
            self.invalidations = 0

        def state_fields(self) -> list[str]:
            return ["x"]

        def initial_state(self) -> dict:
            return {"x": jnp.array(0.0, dtype=jnp.float32)}

        def update(self, state, boundary_inputs, dt):
            return {"x": state["x"] + dt}

        def invalidate_static_cache(self) -> None:
            self.invalidations += 1

    node = _Recorder()
    gm = GraphManager()
    gm.add_node(node)
    gm.compile()
    assert node.invalidations == 1
    gm.compile()
    assert node.invalidations == 2
