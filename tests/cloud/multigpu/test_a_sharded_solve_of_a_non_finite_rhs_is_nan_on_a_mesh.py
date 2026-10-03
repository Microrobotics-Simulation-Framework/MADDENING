"""On a 4-device mesh, a right-hand side with a NaN or infinite entry in any shard gives NaN, not converged.

The single-device cases are in
``tests/core/test_krylov_solves_propagate_a_non_finite_rhs.py`` (MADD-ANO-151).
Here ``b`` is laid out across the mesh with the bad entry in a shard other
than the first, so the finiteness check is a reduction across devices, and
the matvec exchanges ghost cells with ``ppermute``: the answer must be NaN
in every entry of every shard, ``converged=False`` and a NaN residual norm,
on both backends of both solvers, and the cotangent through
``differentiable=True`` must be NaN too.  The conftest in this directory
exposes the host devices.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax, shard_map
from jax.sharding import PartitionSpec as P

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.iterative_solver import sharded_cg, sharded_gmres

N_PER_SHARD = 8
N = 4 * N_PER_SHARD
BAD_ENTRY = 2 * N_PER_SHARD + 3          # in the third shard


def _laplacian(mesh):
    """The 1-D Dirichlet Laplacian, sharded, its ghosts exchanged by ``ppermute``."""
    def shard_matvec(x):
        n_dev = mesh.devices.shape[0]
        left_ghost = lax.ppermute(x[-1], "devices", [(i, (i + 1) % n_dev) for i in range(n_dev)])
        right_ghost = lax.ppermute(x[0], "devices", [(i, (i - 1) % n_dev) for i in range(n_dev)])
        idx = lax.axis_index("devices")
        left_ghost = jnp.where(idx == 0, 0.0, left_ghost)
        right_ghost = jnp.where(idx == n_dev - 1, 0.0, right_ghost)
        left = jnp.concatenate([jnp.asarray([left_ghost], x.dtype), x[:-1]])
        right = jnp.concatenate([x[1:], jnp.asarray([right_ghost], x.dtype)])
        return 2 * x - left - right

    def matvec(x):
        return shard_map(shard_matvec, mesh=mesh, in_specs=(P("devices"),),
                         out_specs=P("devices"))(x)

    return matvec


@pytest.fixture(scope="module")
def mesh():
    return create_device_mesh(shape=(4,))


@pytest.mark.parametrize("backend", ["lineax", "loop"])
@pytest.mark.parametrize("fn", [sharded_cg, sharded_gmres], ids=["cg", "gmres"])
def test_a_non_finite_entry_in_one_shard_gives_nan_in_every_shard(mesh, fn, backend):
    mv = _laplacian(mesh)
    kw = dict(mesh=mesh, in_specs=P("devices"), backend=backend, max_iters=400, rtol=1e-4)
    if fn is sharded_gmres:
        kw["restart"] = N

    @jax.jit
    def run(b):
        r = fn(mv, b, **kw)
        return r.value, r.converged, r.residual_norm

    b0 = jnp.arange(N, dtype=jnp.float32) + 1.0
    x, converged, _ = run(b0)
    assert bool(converged) and np.all(np.isfinite(np.asarray(x)))
    for bad in (float("nan"), float("inf"), float("-inf")):
        value, converged, residual_norm = run(b0.at[BAD_ENTRY].set(bad))
        assert np.all(np.isnan(np.asarray(jax.device_get(value)))), (bad, value)
        assert not bool(converged), bad
        assert np.isnan(float(residual_norm)), bad


@pytest.mark.parametrize("fn", [sharded_cg, sharded_gmres], ids=["cg", "gmres"])
def test_a_non_finite_cotangent_on_a_mesh_gives_a_nan_gradient(mesh, fn):
    mv = _laplacian(mesh)
    kw = dict(mesh=mesh, in_specs=P("devices"), backend="lineax", differentiable=True)
    if fn is sharded_gmres:
        kw["restart"] = N

    @jax.jit
    def cotangent(b, ct):
        return jax.vjp(lambda bb: fn(mv, bb, **kw).value, b)[1](ct)[0]

    b0 = jnp.arange(N, dtype=jnp.float32) + 1.0
    ct = jnp.ones(N, jnp.float32)
    assert np.all(np.isfinite(np.asarray(cotangent(b0, ct))))
    for bad in (float("nan"), float("inf"), float("-inf")):
        g = np.asarray(jax.device_get(cotangent(b0, ct.at[BAD_ENTRY].set(bad))))
        assert np.all(np.isnan(g)), (bad, g)
