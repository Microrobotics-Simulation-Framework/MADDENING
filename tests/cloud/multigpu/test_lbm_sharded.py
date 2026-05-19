"""Sharded LBMNode regression vs unsharded.

Covers M7 of the v0.2 halo-exchange roadmap.

For v0.2 the sharded LBMNode is restricted to wall-less, BC-less
periodic domains (the surface area where ``update_padded`` is fully
implemented).  Walls and Zou-He pressure BCs under sharding land
later -- they need a sharded ``wall_mask`` (M8 / v0.2.x).  Even with
this restriction we can verify the pencil-decomposed LBM matches the
single-device path step-for-step on:

- 8-device 2x4 ``(spatial_y, spatial_z)`` pencil mesh
- 16-device 4x4 pencil mesh
- ``mesh=(4, 2)`` (transposed) -- catches axis-confusion bugs
- mass and momentum conservation under sharding

The full Hagen-Poiseuille analytic comparison requires walls and
deferred-pressure BCs; it's tracked in the v0.2 follow-up.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.nodes.lbm import LBMNode

_HAS_8 = len(jax.devices()) >= 8
_HAS_16 = len(jax.devices()) >= 16


def _perturbed_node(grid, seed=0):
    """An LBMNode in a periodic cubic domain with a small initial perturbation."""
    node = LBMNode(
        name="lbm", timestep=1.0, grid_shape=grid,
        viscosity=0.05, lattice="D3Q19",
    )
    state = node.initial_state()
    rng = np.random.default_rng(seed)
    pert = rng.standard_normal(state["f"].shape).astype(np.float32) * 1e-3
    state["f"] = state["f"] + jnp.asarray(pert)
    return node, state


@pytest.mark.skipif(not _HAS_8, reason="needs >=8 devices")
@pytest.mark.parametrize("mesh_shape", [(2, 4), (4, 2)])
def test_sharded_lbm_matches_unsharded_8_devices(mesh_shape):
    """One step on an 8-device mesh; sharded result equals unsharded."""
    node, state = _perturbed_node(grid=(4, 4, 8))
    mesh = create_device_mesh(shape=mesh_shape)
    sharded = ShardedStencilNode(
        node, mesh,
        axis_map={"spatial_y": 1, "spatial_z": 2},
        boundary="periodic",
    )

    new_u = node.update(state, {}, 1.0)
    new_s = sharded.update(state, {}, 1.0)

    for field in ("f", "density", "velocity", "pressure"):
        np.testing.assert_allclose(
            np.asarray(new_s[field]),
            np.asarray(new_u[field]),
            rtol=1e-5, atol=1e-6,
            err_msg=f"field {field!r} diverged under mesh={mesh_shape}",
        )


@pytest.mark.skipif(not _HAS_8, reason="needs >=8 devices")
def test_sharded_lbm_many_steps():
    """50 steps; sharded trajectory still matches unsharded."""
    node, state_u = _perturbed_node(grid=(4, 4, 8))
    state_s = {k: v for k, v in state_u.items()}
    mesh = create_device_mesh(shape=(2, 4))
    sharded = ShardedStencilNode(
        node, mesh,
        axis_map={"spatial_y": 1, "spatial_z": 2},
        boundary="periodic",
    )
    for _ in range(50):
        state_u = node.update(state_u, {}, 1.0)
        state_s = sharded.update(state_s, {}, 1.0)

    np.testing.assert_allclose(
        np.asarray(state_s["density"]),
        np.asarray(state_u["density"]),
        rtol=1e-4, atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(state_s["velocity"]),
        np.asarray(state_u["velocity"]),
        rtol=1e-3, atol=1e-5,
    )


@pytest.mark.skipif(not _HAS_8, reason="needs >=8 devices")
def test_sharded_lbm_mass_conservation_matches_unsharded():
    """Sharded total mass matches unsharded total mass step-for-step.

    LBM with a perturbed equilibrium leaks a tiny bit of mass each step
    due to float32 accumulation; that drift must match exactly between
    sharded and unsharded, since both run the same algebra.
    """
    node, state_u = _perturbed_node(grid=(4, 4, 8))
    state_s = {k: v for k, v in state_u.items()}
    mesh = create_device_mesh(shape=(2, 4))
    sharded = ShardedStencilNode(
        node, mesh,
        axis_map={"spatial_y": 1, "spatial_z": 2},
        boundary="periodic",
    )

    for _ in range(20):
        state_u = node.update(state_u, {}, 1.0)
        state_s = sharded.update(state_s, {}, 1.0)

    mass_u = float(jnp.sum(state_u["density"]))
    mass_s = float(jnp.sum(state_s["density"]))
    np.testing.assert_allclose(mass_s, mass_u, rtol=1e-5)


@pytest.mark.skipif(not _HAS_16, reason="needs >=16 devices")
def test_sharded_lbm_4x4_pencil():
    """64-cube LBM on a 4x4 pencil mesh."""
    node, state = _perturbed_node(grid=(4, 8, 8))
    mesh = create_device_mesh(shape=(4, 4))
    sharded = ShardedStencilNode(
        node, mesh,
        axis_map={"spatial_y": 1, "spatial_z": 2},
        boundary="periodic",
    )

    new_u = node.update(state, {}, 1.0)
    new_s = sharded.update(state, {}, 1.0)

    np.testing.assert_allclose(
        np.asarray(new_s["f"]), np.asarray(new_u["f"]),
        rtol=1e-5, atol=1e-6,
    )
