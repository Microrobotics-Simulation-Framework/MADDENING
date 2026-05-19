"""Sharded HeatNode vs Fourier-series analytic solution.

Covers M5 of the v0.2 halo-exchange roadmap: HeatNode now exposes
``update_padded``; wrap it with :class:`ShardedStencilNode` and verify
that the sharded result matches the analytical solution to the same
order as the unsharded reference, on multiple shard counts including
the thin-shard regime (8 cells per shard on a 16-device mesh).

For ``T(x,0) = sin(pi x/L)`` with Dirichlet ``T(0,t)=T(L,t)=0`` the
sharded path uses ``boundary="zero"`` for halo exchange -- the global
ghost cells are filled with zero, which is exactly the Dirichlet BC the
stencil expects.  This is *more* accurate than the unsharded path,
which overwrites the boundary cells (introducing O(dx) error -- see
MADD-ANO-002 in the analytical test).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.nodes.heat import HeatNode

_HAS_4 = len(jax.devices()) >= 4
_HAS_8 = len(jax.devices()) >= 8
_HAS_16 = len(jax.devices()) >= 16


def _heat_analytical(x: np.ndarray, t: float, L: float, alpha: float) -> np.ndarray:
    return np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi / L) ** 2 * t)


def _build(n_cells, L=1.0, alpha=0.01, stencil_order=2):
    dx = L / n_cells
    CFL = 0.25
    dt = CFL * dx * dx / alpha
    x = np.linspace(dx / 2, L - dx / 2, n_cells)
    T0 = np.sin(np.pi * x / L)
    node = HeatNode(
        "heat", timestep=dt, n_cells=n_cells, length=L,
        thermal_diffusivity=alpha,
        initial_temperature=T0.astype(np.float32),
        stencil_order=stencil_order,
    )
    return node, dx, dt, x


# ---------------------------------------------------------------------------
# 2nd-order sharded vs analytic
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_4, reason="needs >=4 virtual devices")
@pytest.mark.parametrize("n_devices,n_cells", [(2, 64), (4, 64), (8, 64)])
def test_sharded_matches_fourier_analytic_2nd_order(n_devices, n_cells):
    if len(jax.devices()) < n_devices:
        pytest.skip(f"needs >={n_devices} devices")

    node, dx, dt, x = _build(n_cells)
    mesh = create_device_mesh(shape=(n_devices,))
    sharded = ShardedStencilNode(
        node, mesh, axis_map={"devices": 0}, boundary="zero",
    )

    state = node.initial_state()
    n_steps = 100
    for _ in range(n_steps):
        state = sharded.update(state, {}, dt)
    t_final = n_steps * dt

    T_num = np.asarray(state["temperature"])
    T_exact = _heat_analytical(x, t_final, 1.0, 0.01)

    l2 = np.sqrt(np.sum((T_num - T_exact) ** 2) / np.sum(T_exact ** 2))
    assert l2 < 0.05, f"sharded L2 error {l2:.4f} exceeds 5% (n={n_devices})"


# ---------------------------------------------------------------------------
# Thin-shard regime: 16 devices, 8 cells per shard
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_16, reason="needs >=16 virtual devices")
def test_sharded_thin_shards_16_devices():
    """8 cells per shard exercises the boundary case where halo dominates."""
    node, dx, dt, x = _build(n_cells=128)
    mesh = create_device_mesh(shape=(16,))
    sharded = ShardedStencilNode(
        node, mesh, axis_map={"devices": 0}, boundary="zero",
    )
    state = node.initial_state()
    for _ in range(100):
        state = sharded.update(state, {}, dt)
    t_final = 100 * dt
    T_num = np.asarray(state["temperature"])
    T_exact = _heat_analytical(x, t_final, 1.0, 0.01)
    l2 = np.sqrt(np.sum((T_num - T_exact) ** 2) / np.sum(T_exact ** 2))
    assert l2 < 0.05, f"thin-shard L2 error {l2:.4f}"


# ---------------------------------------------------------------------------
# 4th-order sharded (halo=2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_4, reason="needs >=4 virtual devices")
def test_sharded_4th_order():
    node, dx, dt, x = _build(n_cells=64, stencil_order=4)
    assert node.halo_width() == {0: 2}
    mesh = create_device_mesh(shape=(4,))
    sharded = ShardedStencilNode(
        node, mesh, axis_map={"devices": 0}, boundary="zero",
    )
    state = node.initial_state()
    for _ in range(100):
        state = sharded.update(state, {}, dt)
    t_final = 100 * dt
    T_num = np.asarray(state["temperature"])
    T_exact = _heat_analytical(x, t_final, 1.0, 0.01)
    l2 = np.sqrt(np.sum((T_num - T_exact) ** 2) / np.sum(T_exact ** 2))
    # 4th-order interior + halo=2 Dirichlet handling should be at least as
    # accurate as 2nd-order; we keep the same 5% threshold.
    assert l2 < 0.05, f"4th-order sharded L2 error {l2:.4f}"


# ---------------------------------------------------------------------------
# Bit-exact-ish vs unsharded (boundary handling differs, so just close)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_4, reason="needs >=4 virtual devices")
def test_sharded_close_to_unsharded():
    """Sharded "zero" boundary ≈ unsharded Dirichlet BCs.

    They are not bit-exact because the unsharded path overwrites
    boundary cells (the MADD-ANO-002 boundary overwrite), whereas the
    sharded path lets the boundary cell evolve naturally with a zero
    ghost.  Both converge to the analytic solution; the difference is
    bounded by the boundary-overwrite error, ~1.5% at n=64.
    """
    node, dx, dt, x = _build(n_cells=64)
    mesh = create_device_mesh(shape=(4,))
    sharded = ShardedStencilNode(
        node, mesh, axis_map={"devices": 0}, boundary="zero",
    )

    state_u = node.initial_state()
    state_s = node.initial_state()
    for _ in range(100):
        state_u = node.update(
            state_u,
            {"left_temperature": jnp.float32(0.0),
             "right_temperature": jnp.float32(0.0)},
            dt,
        )
        state_s = sharded.update(state_s, {}, dt)

    T_u = np.asarray(state_u["temperature"])
    T_s = np.asarray(state_s["temperature"])
    diff = np.sqrt(np.sum((T_s - T_u) ** 2) / np.sum(T_u ** 2))
    assert diff < 0.03, f"sharded/unsharded diff {diff:.4f} unexpectedly large"


# Gradient through ppermute / sharded HeatNode is covered by the
# halo_exchange primitive gradient test (M3) and the full multi-device
# audit in M8 (Heat<->LBM coupled rollout vs FD reference).
