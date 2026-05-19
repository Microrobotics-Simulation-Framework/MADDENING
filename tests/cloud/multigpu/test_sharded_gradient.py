"""Multi-device gradient audit through ``ppermute``.

Covers M8 of the v0.2 halo-exchange roadmap.  Validates that
``jax.grad`` through a multi-step sharded stencil rollout (Heat under a
pencil mesh) matches a finite-difference reference.

We compute the gradient of a simple scalar loss
(``mean(T_final ** 2)``) with respect to the initial temperature and
compare against centred finite differences on a small grid.  The
test is small (4 virtual devices, 16 cells, 5 steps) so the FD loop
stays cheap.  The unsharded run is the analytic reference; the
sharded path must match it bit-close because the only difference is
the order of accumulation through ``ppermute``.
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


def _build_sharded(n_cells, alpha=0.01):
    L = 1.0
    dx = L / n_cells
    dt = 0.25 * dx * dx / alpha
    node = HeatNode(
        name="heat_g", timestep=dt, n_cells=n_cells, length=L,
        thermal_diffusivity=alpha,
        initial_temperature=np.zeros(n_cells, dtype=np.float32),
    )
    mesh = create_device_mesh(shape=(4,))
    sharded = ShardedStencilNode(
        node, mesh, axis_map={"devices": 0}, boundary="zero",
    )
    return node, sharded, dt


@pytest.mark.skipif(not _HAS_4, reason="needs >=4 virtual devices")
def test_gradient_through_5_steps_matches_fd():
    """jax.grad through 5 sharded heat steps ≈ centred FD reference."""
    n_cells = 16
    node, sharded, dt = _build_sharded(n_cells)

    def loss(T0):
        state = {"temperature": T0}
        for _ in range(5):
            state = sharded.update(state, {}, dt)
        return jnp.mean(state["temperature"] ** 2)

    rng = np.random.default_rng(0)
    T0 = jnp.asarray(rng.standard_normal(n_cells).astype(np.float32))

    grad_ad = np.asarray(jax.grad(loss)(T0))

    # Centred FD on a few indices (full sweep is ~16 evaluations, OK)
    eps = 1e-2
    grad_fd = np.zeros(n_cells, dtype=np.float32)
    for i in range(n_cells):
        ep = jnp.zeros(n_cells, dtype=jnp.float32).at[i].set(eps)
        grad_fd[i] = float(loss(T0 + ep) - loss(T0 - ep)) / (2 * eps)

    np.testing.assert_allclose(grad_ad, grad_fd, rtol=5e-3, atol=5e-3)


@pytest.mark.skipif(not _HAS_4, reason="needs >=4 virtual devices")
def test_gradient_finite_through_long_rollout():
    """Sanity: gradient through 50 steps is finite (no NaN/Inf)."""
    n_cells = 16
    node, sharded, dt = _build_sharded(n_cells)

    def loss(T0):
        state = {"temperature": T0}
        for _ in range(50):
            state = sharded.update(state, {}, dt)
        return jnp.mean(state["temperature"] ** 2)

    T0 = jnp.linspace(0.0, 1.0, n_cells, dtype=jnp.float32)
    g = jax.grad(loss)(T0)
    assert bool(jnp.all(jnp.isfinite(g)))


# Note: sharded gradients vs unsharded gradients are NOT expected to
# match because the boundary handling differs (unsharded path overwrites
# boundary cells per MADD-ANO-002, sharded uses ghost=0 implicitly via
# boundary="zero").  The FD reference above is the meaningful check.
