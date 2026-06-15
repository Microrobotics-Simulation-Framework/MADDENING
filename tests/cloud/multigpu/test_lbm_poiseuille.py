"""MADD-VER-003: LBM Hagen-Poiseuille analytical verification.

Validates the sharded LBMNode against the steady-state laminar
Poiseuille profile in a circular pipe driven by a uniform body force.

Analytical solution (for body-force-driven flow in a pipe of radius R
with dynamic viscosity mu, periodic in the streamwise axis)::

    u_max = F * R**2 / (4 * mu)
    u(r)  = u_max * (1 - (r/R)**2)

We use a body force rather than pressure BCs because the periodic
domain develops a fully-formed parabolic profile from the start --
viscous time scale R^2 / nu is what matters, no inlet developing
region.

Runs on 8 virtual CPU devices (the multigpu conftest provides them
via ``XLA_FLAGS=--xla_force_host_platform_device_count=16``).  The
sharded path uses a 2x4 pencil mesh over ``(spatial_y, spatial_z)``
with the streamwise axis replicated.

We also verify the equivalent run on the *unsharded* path, so the
test catches both bugs in the new ``update_padded`` and bugs in any
future refactor of plain ``update``.
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


def _cylindrical_wall_mask(nx: int, ny: int, nz: int, radius: float):
    """True where the cell centre lies *outside* a cylinder of given radius."""
    yy = np.arange(ny) - (ny - 1) / 2.0
    zz = np.arange(nz) - (nz - 1) / 2.0
    Y, Z = np.meshgrid(yy, zz, indexing="ij")
    r2 = Y * Y + Z * Z
    mask_2d = r2 > radius * radius
    mask = np.broadcast_to(mask_2d[None, :, :], (nx, ny, nz))
    return mask.copy()


def _analytic_u(r: np.ndarray, R: float, F: float, mu: float) -> np.ndarray:
    u_max = F * R * R / (4.0 * mu)
    profile = u_max * (1.0 - (r / R) ** 2)
    return np.where(r <= R, profile, 0.0)


def _run_poiseuille(node, sharded, n_steps: int, F: float):
    """Run ``sharded`` (which may be the node itself) and return final state."""
    state = node.initial_state()
    # Uniform body-force vector (D,); both unsharded update() and
    # update_padded() broadcast it across the local grid.
    bi = {"body_force": jnp.asarray([F, 0.0, 0.0], dtype=jnp.float32)}
    for _ in range(n_steps):
        state = sharded.update(state, bi, 1.0)
    return state


@pytest.mark.skipif(not _HAS_8, reason="needs >=8 virtual devices")
def test_poiseuille_sharded_profile_is_parabolic():
    """Sharded LBM (2x4 pencil) develops a parabolic Poiseuille-like profile.

    We check the *shape* of the profile (parabolic, symmetric) rather
    than its absolute amplitude.  The Guo body-force scheme in our LBM
    has a known coefficient mismatch with the textbook Hagen-Poiseuille
    formula (the macroscopic body force seen by the fluid is scaled by
    ``(1 - 1/(2*tau))`` instead of unity at finite tau).  This is an
    LBM implementation property, not a sharding bug -- the unsharded
    path reproduces the same profile (verified separately below).

    Steady-state reached after ``~5 * R^2 / nu`` time steps.
    """
    nx, ny, nz = 4, 12, 12
    R = 4.0
    nu = 0.1
    F = 1e-4

    wall_mask = _cylindrical_wall_mask(nx, ny, nz, R)
    node = LBMNode(
        name="poiseuille", timestep=1.0, grid_shape=(nx, ny, nz),
        viscosity=nu, lattice="D3Q19", wall_mask=wall_mask,
    )

    mesh = create_device_mesh(shape=(2, 4))
    sharded = ShardedStencilNode(
        node, mesh,
        axis_map={"spatial_y": 1, "spatial_z": 2},
        boundary="periodic",
    )

    state = _run_poiseuille(node, sharded, n_steps=500, F=F)

    u_x = np.asarray(state["velocity"][..., 0])
    u_x_cross = u_x.mean(axis=0)  # average over streamwise axis

    yy = np.arange(ny) - (ny - 1) / 2.0
    zz = np.arange(nz) - (nz - 1) / 2.0
    Y, Z = np.meshgrid(yy, zz, indexing="ij")
    r = np.sqrt(Y * Y + Z * Z)

    fluid_2d = r <= R - 0.5
    u_fluid = u_x_cross[fluid_2d]
    r_fluid = r[fluid_2d]

    # Profile must be positive everywhere and peak at the centre.
    assert u_fluid.min() > 0
    cy, cz = (ny - 1) // 2, (nz - 1) // 2
    # Even-sized cross-section: four cells at indices (cy, cz),
    # (cy+1, cz), (cy, cz+1), (cy+1, cz+1) are equally central
    # and physics-tied for the peak.  Which one wins .max() in
    # float32 is reduction-order-dependent (varies across
    # jaxlib builds / Python versions), so we assert the centre
    # cell equals the peak within float roundoff rather than
    # bit-exactly -- the physics check ("centre is the peak")
    # is preserved.
    np.testing.assert_allclose(u_x_cross[cy, cz], u_fluid.max(), rtol=1e-5)

    # Fit u = a (R_eff^2 - r^2) on fluid cells; expect r^2-coefficient
    # negative and close to -a*1 with a > 0.
    # Use least squares: u = c0 + c1 * r^2.
    A = np.stack([np.ones_like(r_fluid), r_fluid ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(A, u_fluid, rcond=None)
    c0, c1 = float(coef[0]), float(coef[1])
    assert c1 < 0, f"profile not concave: c1={c1}"
    # R_eff from u(R_eff) = 0: R_eff^2 = -c0 / c1
    r2_eff = -c0 / c1
    assert 0.7 * R * R < r2_eff < 1.5 * R * R, (
        f"effective R^2={r2_eff:.2f} too far from nominal R^2={R*R:.2f}"
    )

    # Symmetric: u(cy, cz) == u(cy+1, cz) (cells equidistant from centre)
    u_cross = u_x_cross
    np.testing.assert_allclose(
        u_cross[cy, cz], u_cross[cy + 1, cz], rtol=1e-3,
    )
    np.testing.assert_allclose(
        u_cross[cy, cz], u_cross[cy, cz + 1], rtol=1e-3,
    )

    # Amplitude: with the Guo body-force half-correction the centerline
    # velocity matches the textbook u_max = F R^2 / (4 mu) to within the
    # discretization offset (effective R > nominal R from mid-link
    # bounce-back placement).  Pre-fix the ratio was ~0.42; post-fix it
    # lands at ~1.13.  Tolerance: within +/-25% of nominal u_max.
    mu = 0.1  # nu * rho with rho = 1
    u_max_analytic = F * R * R / (4.0 * mu)
    u_center = u_x_cross[cy, cz]
    ratio = u_center / u_max_analytic
    assert 0.75 < ratio < 1.30, (
        f"centerline u={u_center:.4e} vs u_max={u_max_analytic:.4e} "
        f"(ratio {ratio:.2f}) outside [0.75, 1.30] -- Guo correction broken?"
    )


@pytest.mark.skipif(not _HAS_8, reason="needs >=8 virtual devices")
def test_poiseuille_sharded_matches_unsharded_step_by_step():
    """Sharded vs unsharded Poiseuille run must agree at every step."""
    nx, ny, nz = 4, 8, 8
    R = 3.0
    wall_mask = _cylindrical_wall_mask(nx, ny, nz, R)
    node = LBMNode(
        name="lbm_pois", timestep=1.0, grid_shape=(nx, ny, nz),
        viscosity=0.1, lattice="D3Q19", wall_mask=wall_mask,
    )
    mesh = create_device_mesh(shape=(2, 4))
    sharded = ShardedStencilNode(
        node, mesh,
        axis_map={"spatial_y": 1, "spatial_z": 2},
        boundary="periodic",
    )

    state_u = node.initial_state()
    state_s = {k: v for k, v in state_u.items()}
    bi = {"body_force": jnp.asarray([1e-4, 0.0, 0.0], dtype=jnp.float32)}
    for _ in range(100):
        state_u = node.update(state_u, bi, 1.0)
        state_s = sharded.update(state_s, bi, 1.0)

    u_u = np.asarray(state_u["velocity"][..., 0])
    u_s = np.asarray(state_s["velocity"][..., 0])
    np.testing.assert_allclose(u_s, u_u, rtol=1e-3, atol=1e-5)
