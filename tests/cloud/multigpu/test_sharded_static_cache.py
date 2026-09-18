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
from maddening.core.simulation.hybrid_node import HybridNode
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


# ---------------------------------------------------------------------------
# ... including a cache that is not on the object the graph holds.  The
# graph sees only the outermost node, so the hook has to be a contract
# method that forwards inwards rather than a name probed on one object.
# Single device, so these are not skipped on a 1-CPU CI runner.
# ---------------------------------------------------------------------------


def _in_place_mask_wrapper():
    """A one-device ``ShardedStencilNode`` over a rewritable mask."""
    inner = InPlaceMaskDiffusion1D("d", n=N_CELLS)
    mesh = create_device_mesh(shape=(1,))
    return inner, ShardedStencilNode(
        inner, mesh, axis_map={"devices": 0}, boundary="edge",
    )


def _step_with_zeroed_mask(gm, inner, wrapper):
    """Run one step, zero the mask in place, recompile, run one more.

    Returns ``(state_before, state_after)`` from the same starting
    field, so a mask of zeros must leave them identical.
    """
    gm.compile()
    gm.step()
    start = np.asarray(gm.get_node_state("d")["f"]).copy()
    inner._mask[:] = 0.0
    gm.compile()
    # Read the cache here: the step below re-materialises it, so after
    # the step "populated" says nothing about whether compile() cleared it.
    assert wrapper._static_device_cache is None, (
        "compile() left the wrapper's materialised statics in place"
    )
    gm.set_node_state("d", {"f": jnp.asarray(start)})
    gm.step()
    return start, np.asarray(gm.get_node_state("d")["f"])


def test_compile_reaches_a_static_cache_nested_inside_another_node():
    """A wrapped sharded node's cache is dropped by ``compile()`` too.

    ``HybridNode`` holds the node it augments as an attribute, so the
    graph's node -- the only object ``compile()`` sees -- is the
    ``HybridNode``.  Probing *it* for a materialised-statics cache finds
    nothing, while the ``ShardedStencilNode`` one level in still holds
    the pre-rewrite device buffer.  Invalidation has to follow the
    wrapping, or a composed graph keeps the silently-wrong results that
    invalidating at all was meant to remove.
    """
    inner, wrapper = _in_place_mask_wrapper()
    hybrid = HybridNode(wrapper, lambda state, boundary_inputs, dt: {})
    gm = GraphManager()
    gm.add_node(hybrid)

    start, after = _step_with_zeroed_mask(gm, inner, wrapper)

    np.testing.assert_allclose(
        after, start, rtol=0, atol=0,
        err_msg="compile() traced the step against the pre-rewrite mask "
                "of a sharded node nested inside a HybridNode",
    )


def test_invalidate_static_cache_forwards_to_the_node_it_wraps():
    """The contract, not one wrapper's behaviour.

    Every node answers ``invalidate_static_cache`` (the base class does),
    and the default forwards to the nodes held as attributes, so a
    wrapper that adds no cache of its own still passes the call inwards.
    """
    _, wrapper = _in_place_mask_wrapper()
    wrapper.update(wrapper.initial_state(), {}, 0.01)
    assert wrapper._static_device_cache is not None

    hybrid = HybridNode(wrapper, lambda state, boundary_inputs, dt: {})
    hybrid.invalidate_static_cache()
    assert wrapper._static_device_cache is None


def test_invalidate_static_cache_terminates_on_a_cycle():
    """Two nodes holding each other must not recurse forever."""
    class _Holder(SimulationNode):
        def __init__(self, name: str) -> None:
            super().__init__(name=name, timestep=0.1)
            self.other: SimulationNode | None = None
            self.invalidations = 0

        def state_fields(self) -> list[str]:
            return ["x"]

        def initial_state(self) -> dict:
            return {"x": jnp.array(0.0, dtype=jnp.float32)}

        def update(self, state, boundary_inputs, dt):
            return {"x": state["x"] + dt}

        def invalidate_static_cache(self) -> None:
            self.invalidations += 1
            super().invalidate_static_cache()

    a, b = _Holder("a"), _Holder("b")
    a.other, b.other = b, a
    a.invalidate_static_cache()          # must return, not recurse
    assert a.invalidations >= 1 and b.invalidations >= 1
    # The guard sits in the base method, so a cycle re-enters an
    # override at most once more before the forwarding stops.
    assert a.invalidations <= 2 and b.invalidations <= 2


# ---------------------------------------------------------------------------
# The wrapper proxies its inner node's static_data.
#
# ``SimulationNode.static_data`` defaults to the merged static_data of the
# nodes held as attributes, so a sharded wrapper -- which declares none of
# its own -- reports what it wraps.  Before that every wrapper hashed to
# ``0`` forever and ``GraphManager._check_static_data_dirty`` could never
# fire for the nodes that carry the cache above: the drift check did
# nothing for exactly the nodes it was most needed on.
#
# What a wrapper reports is the inner node's *declaration*, not the
# per-device materialisation.  See ``SimulationNode.static_data``: the
# materialised shard is a bare ``jax.Array`` that has lost ``replication``
# and ``shard_axis`` (and would raise ``MigrationError`` when hashed),
# producing it costs a ``device_put`` on a property the drift check reads
# every step, and a per-device view is a function of the mesh rather than
# of the statics.
# ---------------------------------------------------------------------------


class _BiasedMaskDiffusion1D(InPlaceMaskDiffusion1D):
    """Sharded mask plus a replicated static that can appear later.

    Models a node that acquires a static after the graph is compiled (a
    reconfiguration, a provider that fills in on restore) -- a change of
    the *key set*, which is what the hash is contracted to catch.
    """

    def __init__(self, name: str, n: int):
        super().__init__(name=name, n=n)
        self.bias = None

    @property
    def static_data(self) -> dict:
        sd = dict(super().static_data)
        if self.bias is not None:
            sd["bias"] = StaticArray(value=self.bias)
        return sd


def _biased_wrapper(n_devices: int = 1):
    inner = _BiasedMaskDiffusion1D("d", n=N_CELLS)
    mesh = create_device_mesh(shape=(n_devices,))
    return inner, ShardedStencilNode(
        inner, mesh, axis_map={"devices": 0}, boundary="edge",
    )


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_sharded_wrapper_reports_the_declaration_not_the_per_shard_view():
    """Full shape and sharding policy, over four devices.

    A per-shard view would report ``(N_CELLS // N_DEVICES,)`` and a bare
    array, so an unchanged node would hash differently on a different
    mesh -- every compile would look like drift.
    """
    inner, wrapper = _stencil_wrapper()
    declared = wrapper.static_data["mask"]
    assert isinstance(declared, StaticArray)
    assert declared.shape == (N_CELLS,)          # not N_CELLS // N_DEVICES
    assert declared.replication == "shard"
    assert declared.shard_axis == 0
    assert wrapper.static_data_hash() == inner.static_data_hash() != 0


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_materialising_the_shards_does_not_move_the_wrappers_hash():
    """Reading the per-device cache must not change what is reported."""
    _, wrapper = _stencil_wrapper()
    before = wrapper.static_data_hash()
    _drive(wrapper, wrapper.initial_state(), 2)
    assert wrapper._static_device_cache is not None
    assert wrapper.static_data_hash() == before


def test_a_hybrid_over_a_sharded_node_reports_the_statics_two_levels_in():
    """``HybridNode(ShardedStencilNode(node))`` -- the composed shape.

    The graph holds only the ``HybridNode``; a proxy covering one level
    would leave this at ``0``.
    """
    inner, wrapper = _in_place_mask_wrapper()
    hybrid = HybridNode(wrapper, lambda state, boundary_inputs, dt: {})
    assert set(hybrid.static_data) == {"mask"}
    assert hybrid.static_data["mask"].replication == "shard"
    assert hybrid.static_data_hash() == inner.static_data_hash() != 0


def test_drift_check_fires_for_a_changed_static_behind_two_wrappers():
    """The whole point: a changed inner static reaches the graph.

    The node gains a replicated static after ``compile()``; the next
    ``step()`` has to retrace rather than run the stale executable.
    """
    inner, wrapper = _biased_wrapper()
    hybrid = HybridNode(wrapper, lambda state, boundary_inputs, dt: {})
    gm = GraphManager()
    gm.add_node(hybrid)
    gm.compile()
    assert gm._static_data_hashes["d"] != 0, (
        "the wrapper hashed to 0, so the drift check cannot fire"
    )
    gm.step()
    # ``_n_traces`` resets on every compile, so the compiled step's own
    # identity is what says a rebuild happened.
    step_before = gm._compiled_step

    inner.bias = np.zeros(N_CELLS, dtype=np.float32)

    assert gm._check_static_data_dirty() is True
    assert gm._dirty is True
    gm.step()
    assert gm._compiled_step is not step_before
    assert gm._static_data_hashes["d"] == hybrid.static_data_hash()


def test_drift_check_stays_quiet_behind_two_wrappers_when_nothing_changed():
    """A check that always fires is as useless as one that never does.

    Includes a parameter-driven rebuild of the static array: a *new
    array object* of the same shape and dtype must not read as drift,
    because the hash is over shape/dtype/replication/shard_axis by
    contract and recompiling on it would undo the cached
    materialisation's whole purpose.
    """
    inner, wrapper = _biased_wrapper()
    hybrid = HybridNode(wrapper, lambda state, boundary_inputs, dt: {})
    gm = GraphManager()
    gm.add_node(hybrid)
    gm.compile()
    gm.step()
    step_before = gm._compiled_step

    for _ in range(3):
        gm.step()
        assert gm._check_static_data_dirty() is False
        assert gm._dirty is False

    inner._mask = np.full(N_CELLS, 0.5, dtype=np.float32)   # same shape
    assert gm._check_static_data_dirty() is False
    gm.step()
    assert gm._compiled_step is step_before
