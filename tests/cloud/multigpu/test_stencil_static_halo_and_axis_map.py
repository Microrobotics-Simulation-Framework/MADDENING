"""How ``ShardedStencilNode`` fills a sharded static's halos, and two
refusals of input it used to ignore.

* A ``StaticArray(replication="shard")`` was halo-exchanged with
  ``boundary="edge"`` whatever the wrapper's mode, on the reasoning that
  statics do not evolve.  The halo fill is about space -- which cells lie
  beyond the edge -- and on a periodic grid those are the opposite edge's
  cells, static or not: a periodic node reading a static in its halo ran
  about 3% off the unsharded node.  It is now periodic under a periodic
  wrapper; under ``"edge"`` and ``"zero"`` it keeps the edge fill.
* ``axis_map={}`` shards nothing; it was accepted for an ``LBMNode``.
* ``halo_exchange(boundary=<dict>)`` ignored a key naming no exchanged
  mesh axis, so a misspelt axis silently took ``"edge"``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import shard_map
from jax.sharding import PartitionSpec as P

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo import halo_exchange
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.node import SimulationNode
from maddening.core.static_data import StaticArray

_HAS_4 = len(jax.devices()) >= 4
pytestmark = pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")

N = 16


class _PeriodicVarDiffusion(SimulationNode):
    """1-D periodic diffusion whose coefficient ``k`` is a sharded static,
    read at the faces: ``k_face = (k_i + k_j) / 2``.  Unsharded it wraps
    ``k`` with ``jnp.roll``; sharded it reads ``static_padded["k"]``, so
    the global-edge faces use whatever the wrapper put in the static's
    halo."""

    def __init__(self):
        super().__init__(name="ring", timestep=0.1)
        x = (np.arange(N) + 0.5) / N
        self._k = (1.0 + 0.5 * np.sin(2 * np.pi * x) + 0.8 * x).astype(np.float32)
        self._static = {"k": StaticArray(value=jnp.asarray(self._k),
                                         replication="shard", shard_axis=0)}

    @property
    def static_data(self):
        return self._static

    def halo_width(self):
        return {0: 1}

    def halo_boundary(self):
        return "periodic"

    def initial_state(self):
        x = (np.arange(N) + 0.5) / N
        return {"T": jnp.asarray(np.cos(2 * np.pi * x) + 0.3 * np.sin(6 * np.pi * x),
                                 jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        t = state["T"]
        k = jnp.asarray(self._k)
        kr, kl = 0.5 * (k + jnp.roll(k, -1)), 0.5 * (k + jnp.roll(k, 1))
        return {"T": t + dt * (kr * (jnp.roll(t, -1) - t) - kl * (t - jnp.roll(t, 1)))}

    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None, shard_info=None):
        tp, kp = state_padded["T"], static_padded["k"]
        t = tp[1:-1]
        kr, kl = 0.5 * (kp[1:-1] + kp[2:]), 0.5 * (kp[1:-1] + kp[:-2])
        return {"T": tp.at[1:-1].set(t + dt * (kr * (tp[2:] - t) - kl * (t - tp[:-2])))}


@pytest.mark.parametrize("n_devices", [1, 2, 4])
def test_a_periodic_node_reading_a_sharded_static_in_its_halo_matches_unsharded(n_devices):
    """20 steps; before 0.4.0 every cell was 1.9e-2 off (on 0.66) on 1, 2
    and 4 devices alike.  Now float32 rounding (the two paths add the face
    terms in a different order on a split grid)."""
    node = _PeriodicVarDiffusion()
    ref_step = jax.jit(node.update)
    ref = node.initial_state()
    for _ in range(20):
        ref = ref_step(ref, {}, 0.1)
    sharded = ShardedStencilNode(node, create_device_mesh(shape=(n_devices,)),
                                 {"devices": 0}, boundary="periodic")
    st = node.initial_state()
    for _ in range(20):
        st = sharded.update(st, {}, 0.1)
    np.testing.assert_allclose(np.asarray(st["T"]), np.asarray(ref["T"]),
                               rtol=0, atol=1e-6)


class _StaticHaloProbe(SimulationNode):
    """Writes the static's halo neighbours of every cell into the state:
    ``left = k[i-1]``, ``right = k[i+1]`` as ``static_padded`` has them.
    Integer ``k`` keeps every value exact."""

    def __init__(self, boundary):
        super().__init__(name="probe", timestep=1.0)
        self._boundary = boundary
        self._static = {"k": StaticArray(value=jnp.arange(1.0, N + 1.0, dtype=jnp.float32),
                                         replication="shard", shard_axis=0)}

    @property
    def static_data(self):
        return self._static

    def halo_width(self):
        return {0: 1}

    def halo_boundary(self):
        return self._boundary

    def state_fields(self):
        return ["left", "right"]

    def initial_state(self):
        return {"left": jnp.zeros(N, jnp.float32), "right": jnp.zeros(N, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return dict(state)

    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None, shard_info=None):
        kp = static_padded["k"]
        return {"left": state_padded["left"].at[1:-1].set(kp[:-2]),
                "right": state_padded["right"].at[1:-1].set(kp[2:])}


_K = np.arange(1.0, N + 1.0)
_EXPECTED_HALO = {
    # the cells beyond a periodic edge are the opposite edge's
    "periodic": (np.roll(_K, 1), np.roll(_K, -1)),
    # no cell beyond the edge: the edge cell repeated (the fill statics
    # always had, kept under "edge" and "zero")
    "edge": (np.concatenate([[1.0], _K[:-1]]), np.concatenate([_K[1:], [16.0]])),
    "zero": (np.concatenate([[1.0], _K[:-1]]), np.concatenate([_K[1:], [16.0]])),
}


@pytest.mark.parametrize("n_devices", [1, 4])
@pytest.mark.parametrize("boundary", ["periodic", "edge", "zero"])
def test_a_sharded_statics_global_edge_halo_follows_the_documented_fill(boundary, n_devices):
    """Between shards the static's halo is the neighbour's cell under every
    mode; at the two global edges it is the periodic wrap under
    ``"periodic"`` and the edge cell under ``"edge"`` and ``"zero"``."""
    node = _StaticHaloProbe(boundary)
    sharded = ShardedStencilNode(node, create_device_mesh(shape=(n_devices,)),
                                 {"devices": 0}, boundary=boundary)
    out = sharded.update(node.initial_state(), {}, 1.0)
    want_left, want_right = _EXPECTED_HALO[boundary]
    np.testing.assert_array_equal(np.asarray(out["left"]), want_left)
    np.testing.assert_array_equal(np.asarray(out["right"]), want_right)


# ---------------------------------------------------------------------------
# An empty axis_map is refused
# ---------------------------------------------------------------------------


def _heat():
    from maddening.nodes.heat import HeatNode

    return HeatNode("rod", timestep=1.0, n_cells=16, length=1.0,
                    thermal_diffusivity=1e-3)


def _lbm():
    from maddening.nodes.lbm import LBMNode

    return LBMNode("lbm", timestep=1.0, grid_shape=(16, 8), viscosity=0.1,
                   lattice="D2Q9")


@pytest.mark.parametrize("make, boundary", [(_heat, "edge"), (_lbm, "periodic")],
                         ids=["heat", "lbm"])
def test_an_empty_axis_map_is_refused_for_what_it_is(make, boundary):
    """An LBMNode was accepted and every device ran the whole grid; a
    HeatNode was refused with a message about its ``grid_x`` static."""
    with pytest.raises(ValueError) as exc:
        ShardedStencilNode(make(), create_device_mesh(shape=(4,)), {}, boundary=boundary)
    msg = str(exc.value)
    assert "empty axis_map" in msg and "no spatial axis would be sharded" in msg
    assert "grid_x" not in msg


# ---------------------------------------------------------------------------
# halo_exchange: a boundary dict names only exchanged axes
# ---------------------------------------------------------------------------


def _exchange(boundary, mesh):
    fn = shard_map(
        lambda v: halo_exchange(v, mesh=mesh, mesh_axis="devices", spatial_axis=0,
                                halo=1, boundary=boundary),
        mesh=mesh, in_specs=P("devices"), out_specs=P("devices"),
    )
    return np.asarray(jax.jit(fn)(jnp.arange(8.0))).reshape(4, 4)


@pytest.mark.parametrize("boundary", [{"device": "periodic"},
                                      {"devices": "periodic", "spatial_z": "zero"}],
                         ids=["misspelt-axis", "unknown-beside-a-known-axis"])
def test_a_boundary_dict_key_naming_no_exchanged_axis_is_refused(boundary):
    """``{"device": "periodic"}`` on a ``"devices"`` exchange returned the
    edge fill with no error."""
    mesh = create_device_mesh(shape=(4,))
    with pytest.raises(ValueError, match=r"boundary names mesh axes \[.*\] that this call "
                                         r"does not exchange; it exchanges \['devices'\]"):
        _exchange(boundary, mesh)


def test_a_boundary_dict_key_naming_a_mesh_axis_this_call_does_not_exchange_is_refused():
    """On a 2x2 mesh, exchanging along ``spatial_y`` only: a mode for
    ``spatial_z`` -- a real mesh axis -- would be ignored as silently as a
    misspelt one."""
    mesh = create_device_mesh(shape=(2, 2))
    fn = shard_map(
        lambda v: halo_exchange(v, mesh=mesh, axes=[("spatial_y", 0, 1)],
                                boundary={"spatial_y": "edge", "spatial_z": "periodic"}),
        mesh=mesh, in_specs=P("spatial_y", "spatial_z"),
        out_specs=P("spatial_y", "spatial_z"),
    )
    with pytest.raises(ValueError, match=r"\['spatial_z'\] that this call does not exchange"):
        jax.jit(fn)(jnp.zeros((4, 4)))


def test_a_boundary_dict_of_exchanged_axes_fills_as_asked_and_defaults_to_edge():
    """8 cells on 4 devices, halo 1: each row is one shard's padded block."""
    mesh = create_device_mesh(shape=(4,))
    interior = [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]]
    assert _exchange({"devices": "periodic"}, mesh).tolist() == (
        [[7.0, 0.0, 1.0, 2.0]] + interior + [[5.0, 6.0, 7.0, 0.0]])
    assert _exchange({}, mesh).tolist() == (
        [[0.0, 0.0, 1.0, 2.0]] + interior + [[5.0, 6.0, 7.0, 7.0]])
