"""Regression test for the halo-aware LBM streaming kernel.

Covers M6 of the v0.2 halo-exchange roadmap: verifies that
:func:`_stream_padded` on a periodic-padded distribution gives the
exact same result as the legacy ``jnp.roll`` :func:`_stream`, so the
single-device path is unaffected by the refactor.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.nodes.lbm import (
    LBMNode,
    _stream,
    _stream_padded,
    d2q9,
    d3q19,
)


def _periodic_pad(f: np.ndarray, halo: int, ndim: int) -> np.ndarray:
    """Pad ``f`` periodically by ``halo`` cells on every spatial axis."""
    out = f
    for d in range(ndim):
        out = np.concatenate(
            [np.take(out, range(-halo, 0), axis=d), out,
             np.take(out, range(halo), axis=d)],
            axis=d,
        )
    return out


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_stream_padded_matches_legacy_d3q19(seed):
    rng = np.random.default_rng(seed)
    grid = (6, 5, 7)
    Q = 19
    f = rng.standard_normal(grid + (Q,)).astype(np.float32)
    e = d3q19().e

    legacy = np.asarray(_stream(jnp.asarray(f), e, ndim=3))
    f_pad = _periodic_pad(f, halo=1, ndim=3)
    new = np.asarray(_stream_padded(jnp.asarray(f_pad), e, ndim=3, halo=1))

    np.testing.assert_array_equal(new, legacy)


@pytest.mark.parametrize("seed", [0, 1])
def test_stream_padded_matches_legacy_d2q9(seed):
    rng = np.random.default_rng(seed)
    grid = (5, 6)
    Q = 9
    f = rng.standard_normal(grid + (Q,)).astype(np.float32)
    e = d2q9().e

    legacy = np.asarray(_stream(jnp.asarray(f), e, ndim=2))
    f_pad = _periodic_pad(f, halo=1, ndim=2)
    new = np.asarray(_stream_padded(jnp.asarray(f_pad), e, ndim=2, halo=1))

    np.testing.assert_array_equal(new, legacy)


def test_lbm_update_padded_no_walls_matches_update():
    """LBMNode.update_padded with a periodic pad must match update for a
    cubic, wall-less, BC-less domain.

    Single-step bit-exact regression: the streaming kernel is the only
    new code path; collision & macroscopic are unchanged.
    """
    grid = (4, 4, 4)
    node = LBMNode(
        name="lbm", timestep=1.0, grid_shape=grid,
        viscosity=0.1, lattice="D3Q19",
    )
    state = node.initial_state()
    # Perturb to avoid the trivial equilibrium fixed point.
    rng = np.random.default_rng(0)
    perturb = rng.standard_normal(state["f"].shape).astype(np.float32) * 0.01
    state = {**state, "f": state["f"] + jnp.asarray(perturb)}

    new_unsharded = node.update(state, {}, 1.0)

    # Build a halo-padded version of the input state, mimicking what
    # ShardedStencilNode would feed when the entire domain is on one
    # device (periodic pad on every spatial axis).
    padded = {}
    for field, arr in state.items():
        arr_np = np.asarray(arr)
        spatial_ndim = 3
        if arr.ndim == spatial_ndim:
            padded[field] = jnp.asarray(_periodic_pad(arr_np, 1, spatial_ndim))
        elif arr.ndim == spatial_ndim + 1:
            # (nx,ny,nz, Q or D)
            padded[field] = jnp.asarray(
                _periodic_pad(arr_np, 1, spatial_ndim)
            )
        else:
            padded[field] = arr

    new_padded = node.update_padded(padded, {}, 1.0)

    # Strip halos from the padded output (interior region only).
    sl = (slice(1, -1),) * 3
    np.testing.assert_allclose(
        np.asarray(new_padded["f"])[sl],
        np.asarray(new_unsharded["f"]),
        rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(new_padded["density"])[sl],
        np.asarray(new_unsharded["density"]),
        rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(new_padded["velocity"])[sl],
        np.asarray(new_unsharded["velocity"]),
        rtol=1e-5, atol=1e-6,
    )


def test_lbm_update_padded_with_walls_matches_update():
    """LBMNode.update_padded with walls on the unsharded (single-device)
    path must match update() bit-close for the cells away from the
    domain boundary (where periodic-pad vs walls choice differs)."""
    grid = (4, 4, 4)
    wall_mask = np.zeros(grid, dtype=bool)
    wall_mask[:, 0, :] = True
    wall_mask[:, -1, :] = True
    node = LBMNode(
        name="lbm", timestep=1.0, grid_shape=grid,
        viscosity=0.1, lattice="D3Q19", wall_mask=wall_mask,
    )
    state = node.initial_state()
    rng = np.random.default_rng(0)
    state["f"] = state["f"] + jnp.asarray(
        rng.standard_normal(state["f"].shape).astype(np.float32) * 0.01
    )

    new_unsharded = node.update(state, {}, 1.0)
    padded = {}
    for fname, arr in state.items():
        arr_np = np.asarray(arr)
        padded[fname] = jnp.asarray(_periodic_pad(arr_np, 1, 3))

    new_padded = node.update_padded(padded, {}, 1.0)
    sl = (slice(1, -1),) * 3
    np.testing.assert_allclose(
        np.asarray(new_padded["f"])[sl],
        np.asarray(new_unsharded["f"]),
        rtol=1e-5, atol=1e-6,
    )


def test_lbm_update_padded_with_pressure_bcs_matches_update():
    """Zou-He pressure BCs work in update_padded (single-device, no walls)."""
    grid = (8, 4, 4)
    node = LBMNode(
        name="lbm", timestep=1.0, grid_shape=grid,
        viscosity=0.1, lattice="D3Q19",
    )
    state = node.initial_state()

    bi = {
        "inlet_pressure": jnp.float32(1.0 / 3.0 + 1e-3),
        "outlet_pressure": jnp.float32(1.0 / 3.0),
    }

    new_unsharded = node.update(state, bi, 1.0)

    padded = {f: jnp.asarray(_periodic_pad(np.asarray(arr), 1, 3))
              for f, arr in state.items()}
    new_padded = node.update_padded(padded, bi, 1.0)

    sl = (slice(1, -1),) * 3
    np.testing.assert_allclose(
        np.asarray(new_padded["f"])[sl],
        np.asarray(new_unsharded["f"]),
        rtol=1e-4, atol=1e-5,
    )
