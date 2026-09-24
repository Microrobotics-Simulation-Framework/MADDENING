"""Sharded HeatNode vs Fourier-series analytic solution.

Covers M5 of the v0.2 halo-exchange roadmap: HeatNode exposes
``update_padded``; wrap it with :class:`ShardedStencilNode` and verify
that the sharded result matches the analytical solution to the same
order as the unsharded reference, on multiple shard counts including
the thin-shard regime (8 cells per shard on a 16-device mesh).

For ``T(x,0) = sin(pi x/L)`` with Dirichlet ``T(0,t)=T(L,t)=0`` both
paths take the end temperatures as boundary inputs,
``left_temperature=0`` / ``right_temperature=0``.  Until 0.4.0 the
sharded path could not: ``update_padded`` never read those inputs
(MADD-ANO-030), and these tests wrapped the rod with the ``"zero"``
halo fill to put 0 in the ghost cells instead -- the datum at the ghost
centres, half a cell outside the rod, not at the rod end where
``update`` imposes it -- and could only bound the sharded run against
the unsharded one.  Now ``update_padded`` closes the rod ends exactly
as ``update`` does, the wrapper refuses ``"zero"`` for a HeatNode, and
the comparisons below are parity to float32 rounding.
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


#: Both rod ends held at 0 -- the same inputs for the sharded and the
#: unsharded node.
_COLD_ENDS = {"left_temperature": jnp.float32(0.0),
              "right_temperature": jnp.float32(0.0)}

#: float32 rounding on O(1) temperatures (measured <= 1.2e-7).
_ATOL = 1e-6


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
    sharded = ShardedStencilNode(node, mesh, axis_map={"devices": 0})

    state = node.initial_state()
    n_steps = 100
    for _ in range(n_steps):
        state = sharded.update(state, _COLD_ENDS, dt)
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
    sharded = ShardedStencilNode(node, mesh, axis_map={"devices": 0})
    state = node.initial_state()
    for _ in range(100):
        state = sharded.update(state, _COLD_ENDS, dt)
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
    sharded = ShardedStencilNode(node, mesh, axis_map={"devices": 0})
    state = node.initial_state()
    for _ in range(100):
        state = sharded.update(state, _COLD_ENDS, dt)
    t_final = 100 * dt
    T_num = np.asarray(state["temperature"])

    # The sharded 4th-order rod is the unsharded one: the cubic end
    # closure is applied on the shards holding the rod ends.
    state_u = node.initial_state()
    for _ in range(100):
        state_u = node.update(state_u, _COLD_ENDS, dt)
    np.testing.assert_allclose(T_num, np.asarray(state_u["temperature"]),
                               rtol=0, atol=_ATOL)

    T_exact = _heat_analytical(x, t_final, 1.0, 0.01)
    l2 = np.sqrt(np.sum((T_num - T_exact) ** 2) / np.sum(T_exact ** 2))
    assert l2 < 0.05, f"4th-order sharded L2 error {l2:.4f}"


# ---------------------------------------------------------------------------
# Sharded equals unsharded, same Dirichlet inputs
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_4, reason="needs >=4 virtual devices")
def test_sharded_close_to_unsharded():
    """Sharded Dirichlet ends are the unsharded Dirichlet ends.

    Both runs get ``left_temperature=0`` / ``right_temperature=0``.  Until
    0.4.0 the sharded one ignored them and this test could only bound the
    difference against the ``"zero"`` fill (under 3% at n=64); now the two
    are equal to float32 rounding.
    """
    node, dx, dt, x = _build(n_cells=64)
    mesh = create_device_mesh(shape=(4,))
    sharded = ShardedStencilNode(node, mesh, axis_map={"devices": 0})

    state_u = node.initial_state()
    state_s = node.initial_state()
    for _ in range(100):
        state_u = node.update(state_u, _COLD_ENDS, dt)
        state_s = sharded.update(state_s, _COLD_ENDS, dt)

    np.testing.assert_allclose(np.asarray(state_s["temperature"]),
                               np.asarray(state_u["temperature"]),
                               rtol=0, atol=_ATOL)


# Gradient through ppermute / sharded HeatNode is covered by the
# halo_exchange primitive gradient test (M3) and the full multi-device
# audit in M8 (Heat<->LBM coupled rollout vs FD reference).
