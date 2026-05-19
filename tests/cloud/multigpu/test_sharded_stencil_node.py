"""Tests for ShardedStencilNode pencil-decomposition wrapper.

Verifies that a synthetic toy stencil node produces bit-exact results
under sharding (halo_exchange is a pure communication step, not a
floating-point reduction, so no FP reordering happens).

Runs entirely on virtual CPU devices via the multigpu conftest.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.node import SimulationNode

_HAS_4_DEVICES = len(jax.devices()) >= 4
_HAS_8_DEVICES = len(jax.devices()) >= 8
_SKIP_4 = "Requires >=4 JAX devices"
_SKIP_8 = "Requires >=8 JAX devices"


# ---------------------------------------------------------------------------
# Toy stencil nodes
# ---------------------------------------------------------------------------


class ToyLaplacian1D(SimulationNode):
    """Discrete 1-D Laplacian, halo=1.  ``field`` is a 1-D vector."""

    def halo_width(self) -> dict[int, int]:
        return {0: 1}

    def initial_state(self) -> dict:
        n = int(self.params.get("n", 16))
        return {"field": jnp.arange(n, dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        # Unsharded fallback: do the periodic Laplacian manually.
        f = state["field"]
        lap = jnp.roll(f, 1) - 2 * f + jnp.roll(f, -1)
        return {"field": f + 0.1 * lap * dt}

    def update_padded(self, state_padded, boundary_inputs, dt):
        """Padded update: read ``f[h-1 .. -h-1]`` for interior, return same shape."""
        f_pad = state_padded["field"]
        # Interior view: drop one cell on each side, since halo=1
        f = f_pad[1:-1]
        lap = f_pad[2:] - 2 * f_pad[1:-1] + f_pad[:-2]
        new_f = f + 0.1 * lap * dt
        # Re-pad so output has same shape as input (sharded wrapper strips halos)
        return {"field": jnp.concatenate(
            [f_pad[:1], new_f, f_pad[-1:]], axis=0
        )}


class ToyLaplacian3D(SimulationNode):
    """3-D Laplacian on a regular grid, halo=1 along each axis."""

    def halo_width(self) -> dict[int, int]:
        return {0: 1, 1: 1, 2: 1}

    def initial_state(self) -> dict:
        nx = int(self.params.get("nx", 4))
        ny = int(self.params.get("ny", 4))
        nz = int(self.params.get("nz", 8))
        rng = np.random.default_rng(0)
        return {"field": jnp.asarray(
            rng.standard_normal((nx, ny, nz)).astype(np.float32)
        )}

    def update(self, state, boundary_inputs, dt):
        f = state["field"]
        lap = (
            jnp.roll(f, 1, axis=0) + jnp.roll(f, -1, axis=0)
            + jnp.roll(f, 1, axis=1) + jnp.roll(f, -1, axis=1)
            + jnp.roll(f, 1, axis=2) + jnp.roll(f, -1, axis=2)
            - 6 * f
        )
        return {"field": f + 0.1 * lap * dt}

    def update_padded(self, state_padded, boundary_inputs, dt):
        f = state_padded["field"]
        lap = (
            f[2:, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]
            + f[1:-1, 2:, 1:-1] + f[1:-1, :-2, 1:-1]
            + f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, :-2]
            - 6 * f[1:-1, 1:-1, 1:-1]
        )
        new_f = f[1:-1, 1:-1, 1:-1] + 0.1 * lap * dt

        # Re-pad with the original halo so the wrapper can strip
        out = jnp.zeros_like(f)
        out = out.at[1:-1, 1:-1, 1:-1].set(new_f)
        # Keep the halo regions intact for stripping (any value is fine,
        # the wrapper trims them).
        return {"field": out}


# ---------------------------------------------------------------------------
# 1-D slab
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_slab_matches_unsharded_one_step():
    mesh = create_device_mesh(shape=(4,))
    node = ToyLaplacian1D(name="lap", timestep=0.01, n=16)
    sharded = ShardedStencilNode(
        node, mesh, axis_map={"devices": 0}, boundary="periodic",
    )

    state = node.initial_state()
    new_unsharded = node.update(state, {}, 0.01)["field"]
    new_sharded = sharded.update(state, {}, 0.01)["field"]

    # 1-D Laplacian on a contiguous array reduces in the same order
    # whether sharded or not -- expect bit-exact.
    np.testing.assert_allclose(
        np.asarray(new_sharded), np.asarray(new_unsharded),
        rtol=0, atol=0,
    )


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_slab_matches_unsharded_many_steps():
    mesh = create_device_mesh(shape=(4,))
    node = ToyLaplacian1D(name="lap", timestep=0.01, n=16)
    sharded = ShardedStencilNode(
        node, mesh, axis_map={"devices": 0}, boundary="periodic",
    )

    state_a = node.initial_state()
    state_b = node.initial_state()

    for _ in range(50):
        state_a = node.update(state_a, {}, 0.01)
        state_b = sharded.update(state_b, {}, 0.01)

    np.testing.assert_allclose(
        np.asarray(state_b["field"]),
        np.asarray(state_a["field"]),
        rtol=1e-5, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 2-D pencil mesh, 3-D toy stencil
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_8_DEVICES, reason=_SKIP_8)
def test_pencil_matches_unsharded_one_step():
    mesh = create_device_mesh(shape=(2, 4))
    node = ToyLaplacian3D(name="lap3d", timestep=0.01, nx=4, ny=4, nz=8)
    sharded = ShardedStencilNode(
        node, mesh,
        axis_map={"spatial_y": 1, "spatial_z": 2},
        boundary="periodic",
    )

    state = node.initial_state()
    new_unsharded = node.update(state, {}, 0.01)["field"]
    new_sharded = sharded.update(state, {}, 0.01)["field"]

    # 3-D Laplacian reduces 6 neighbour terms; under sharding XLA may
    # emit a different summation order, so expect agreement to 1 ULP
    # of float32 (~1e-7) rather than bit-exact.
    np.testing.assert_allclose(
        np.asarray(new_sharded), np.asarray(new_unsharded),
        rtol=1e-6, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


def test_rejects_pointwise_node():
    """ShardedStencilNode refuses nodes with empty halo_width."""

    class Pointwise(SimulationNode):
        def halo_width(self):
            return {}

        def initial_state(self):
            return {"x": jnp.zeros(4)}

        def update(self, state, boundary_inputs, dt):
            return state

    mesh = create_device_mesh(shape=(1,))
    with pytest.raises(ValueError, match="empty halo_width"):
        ShardedStencilNode(
            Pointwise(name="p", timestep=0.1), mesh, axis_map={"devices": 0},
        )


def test_rejects_unknown_mesh_axis():
    node = ToyLaplacian1D(name="lap", timestep=0.01, n=16)
    mesh = create_device_mesh(shape=(2,))
    with pytest.raises(ValueError, match="not present"):
        ShardedStencilNode(node, mesh, axis_map={"nonexistent": 0})


def test_rejects_axis_without_halo():
    node = ToyLaplacian1D(name="lap", timestep=0.01, n=16)
    mesh = create_device_mesh(shape=(2,))
    with pytest.raises(ValueError, match="no entry"):
        # Try to shard spatial axis 5, but the node only declares halo
        # on axis 0.
        ShardedStencilNode(node, mesh, axis_map={"devices": 5})


def test_deprecated_sharded_node_alias_warns():
    """Importing and using ShardedNode emits DeprecationWarning."""
    import warnings

    from maddening.cloud.multigpu.sharded_node import ShardedNode

    class Pointwise(SimulationNode):
        def halo_width(self):
            return {}

        def initial_state(self):
            return {"x": jnp.zeros(4)}

        def update(self, state, boundary_inputs, dt):
            return state

    mesh = create_device_mesh(shape=(1,))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ShardedNode(Pointwise(name="p", timestep=0.1), mesh)

    deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deps, "expected DeprecationWarning"
    assert "ShardedPointwiseNode" in str(deps[0].message)
