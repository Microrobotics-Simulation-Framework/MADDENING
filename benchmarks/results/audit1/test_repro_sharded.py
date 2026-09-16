"""Audit reproducer: grid-shaped boundary-input heuristic on square grids."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.node import BoundaryInputSpec, SimulationNode

pytestmark = pytest.mark.skipif(len(jax.devices()) < 4, reason="needs 4 devices")


class Plate(SimulationNode):
    """2-D field on an (n, n) grid with a per-column profile input of shape (n,)."""
    def __init__(self, name, timestep, n=8):
        super().__init__(name, timestep, n=n)

    def halo_width(self):
        return {0: 1}

    def initial_state(self):
        n = self.params["n"]
        return {"T": jnp.asarray(np.arange(n * n, dtype=np.float32).reshape(n, n))}

    def boundary_input_spec(self):
        n = self.params["n"]
        return {"profile": BoundaryInputSpec(shape=(n,), description="per-column source (axis 1)")}

    def _step(self, T, profile, dt):
        lap = jnp.roll(T, 1, 0) + jnp.roll(T, -1, 0) - 2 * T
        return T + dt * (0.1 * lap + profile[None, :])

    def update(self, state, bi, dt):
        n = self.params["n"]
        profile = bi.get("profile", jnp.zeros(n, jnp.float32))
        return {"T": self._step(state["T"], profile, dt)}

    def update_padded(self, state_padded, bi, dt, *, static_padded=None, shard_info=None):
        n = self.params["n"]
        T_pad = state_padded["T"]
        profile = bi.get("profile", jnp.zeros(n, jnp.float32))
        T_new = self._step(T_pad, profile, dt)
        return {"T": jnp.concatenate([T_pad[:1], T_new[1:-1], T_pad[-1:]], axis=0)}


def test_square_grid_profile_input_is_misclassified_as_grid_shaped():
    mesh = create_device_mesh(shape=(4,))
    un = Plate("p", 1.0, n=8)
    sh = ShardedStencilNode(Plate("p", 1.0, n=8), mesh, axis_map={"devices": 0}, boundary="periodic")
    st = un.initial_state()
    bi = {"profile": jnp.asarray(np.linspace(0, 1, 8), jnp.float32)}
    assert sh._grid_shaped_boundary_inputs(st, bi) == frozenset(), \
        "a (n,) profile along the *unsharded* axis was classified as grid-shaped"
    a = sh.update(st, bi, 1.0)["T"]
    b = un.update(st, bi, 1.0)["T"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)


def test_rectangular_grid_profile_input_is_fine():
    mesh = create_device_mesh(shape=(4,))

    class Rect(Plate):
        def initial_state(self):
            return {"T": jnp.zeros((8, 5), jnp.float32)}

        def boundary_input_spec(self):
            return {"profile": BoundaryInputSpec(shape=(5,), description="")}

        def update(self, state, bi, dt):
            return {"T": self._step(state["T"], bi.get("profile", jnp.zeros(5)), dt)}

        def update_padded(self, sp, bi, dt, *, static_padded=None, shard_info=None):
            T_pad = sp["T"]
            T_new = self._step(T_pad, bi.get("profile", jnp.zeros(5)), dt)
            return {"T": jnp.concatenate([T_pad[:1], T_new[1:-1], T_pad[-1:]], axis=0)}

    un = Rect("p", 1.0)
    sh = ShardedStencilNode(Rect("p", 1.0), mesh, axis_map={"devices": 0}, boundary="periodic")
    st = un.initial_state()
    bi = {"profile": jnp.asarray(np.linspace(0, 1, 5), jnp.float32)}
    a = sh.update(st, bi, 1.0)["T"]
    b = un.update(st, bi, 1.0)["T"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)
