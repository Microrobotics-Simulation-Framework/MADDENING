"""Tests for the sharded sparse iterative solver substrate (v0.3.0 §A5).

The conftest in this directory forces XLA to expose 16 virtual CPU
devices, so the 4-device mesh tests run locally without real GPUs.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maddening.cloud.multigpu.iterative_solver import (
    SharedSolveResult,
    sharded_cg,
    sharded_gmres,
)
from maddening.cloud.multigpu.device_mesh import create_device_mesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_4_device_mesh() -> Mesh:
    return create_device_mesh(shape=(4,))


def _laplacian_1d_dense(n: int, dtype=jnp.float32) -> jnp.ndarray:
    """1-D second-difference operator with Dirichlet BCs."""
    main = 2 * jnp.eye(n, dtype=dtype)
    off = jnp.eye(n, k=1, dtype=dtype) + jnp.eye(n, k=-1, dtype=dtype)
    return main - off


def _laplacian_1d_matvec_unsharded(n: int, dtype=jnp.float32):
    """Closed-form Laplacian matvec — no sharding."""
    def matvec(x):
        # x[i+1] - 2 x[i] + x[i-1] with x[-1] = x[n] = 0
        left = jnp.concatenate([jnp.zeros((1,), dtype=x.dtype), x[:-1]])
        right = jnp.concatenate([x[1:], jnp.zeros((1,), dtype=x.dtype)])
        return 2 * x - left - right
    return matvec


def _laplacian_1d_matvec_sharded(mesh: Mesh, n_per_shard: int, dtype=jnp.float32):
    """Sharded Laplacian matvec using shard_map + neighbour ppermute.

    Each shard holds ``n_per_shard`` consecutive entries.  Ghost cells
    are obtained via ``lax.ppermute`` so cross-shard contributions are
    correct.
    """
    def shard_matvec(x_shard):
        # x_shard shape: (n_per_shard,)
        # Get left neighbour's rightmost entry and right neighbour's leftmost.
        n_devices = mesh.devices.shape[0]
        left_perm = [(i, (i + 1) % n_devices) for i in range(n_devices)]
        right_perm = [(i, (i - 1) % n_devices) for i in range(n_devices)]
        # Left neighbour's rightmost value (sent rightward).
        left_ghost = lax.ppermute(x_shard[-1], "devices", left_perm)
        # Right neighbour's leftmost value (sent leftward).
        right_ghost = lax.ppermute(x_shard[0], "devices", right_perm)

        device_idx = lax.axis_index("devices")
        # Zero out the wrap-around contribution at the global boundaries.
        left_ghost = jnp.where(device_idx == 0, 0.0, left_ghost)
        right_ghost = jnp.where(device_idx == n_devices - 1, 0.0, right_ghost)

        left = jnp.concatenate([jnp.asarray([left_ghost], dtype=x_shard.dtype),
                                x_shard[:-1]])
        right = jnp.concatenate([x_shard[1:],
                                 jnp.asarray([right_ghost], dtype=x_shard.dtype)])
        return 2 * x_shard - left - right

    def matvec(x):
        return shard_map(
            shard_matvec,
            mesh=mesh,
            in_specs=(P("devices"),),
            out_specs=P("devices"),
            check_rep=False,
        )(x)

    return matvec


# ---------------------------------------------------------------------------
# Smoke tests — single-device (no mesh).  Validate the API + correctness
# of both the lineax-backed path and the loop fallback before going
# sharded.
# ---------------------------------------------------------------------------


class TestSingleDeviceCorrectness:
    """CG / GMRES converge to the dense reference on a small Laplacian.

    Uses the loop backend for the strict-tolerance checks since float32
    + lineax's strict mode can hit max_steps before the loop backend
    plateaus.  The lineax path gets its own (looser) smoke test below.
    """

    def test_cg_solves_laplacian_loop(self):
        n = 32
        A = _laplacian_1d_dense(n)
        b = jnp.arange(n, dtype=jnp.float32) + 1.0
        x_ref = jnp.linalg.solve(A, b)

        matvec = _laplacian_1d_matvec_unsharded(n)
        result = sharded_cg(matvec, b, max_iters=500, backend="loop")

        assert isinstance(result, SharedSolveResult)
        assert bool(result.converged), (
            f"CG did not converge (residual={float(result.residual_norm):.2e})"
        )
        assert jnp.allclose(result.value, x_ref, atol=1e-3, rtol=1e-3), (
            f"CG solution mismatch "
            f"(max diff={float(jnp.max(jnp.abs(result.value - x_ref))):.2e})"
        )

    def test_gmres_solves_laplacian_loop(self):
        n = 32
        A = _laplacian_1d_dense(n)
        b = jnp.arange(n, dtype=jnp.float32) + 1.0
        x_ref = jnp.linalg.solve(A, b)

        matvec = _laplacian_1d_matvec_unsharded(n)
        # rtol=1e-3 chosen to accommodate float32 — the GMRES loop
        # achieves residual ~7e-4 on this problem (sqrt(eps)-class
        # plateau on a non-symmetric LSQ Hessenberg solve).
        result = sharded_gmres(
            matvec, b, restart=n, max_iters=4 * n, rtol=1e-3, atol=1e-4,
            backend="loop",
        )

        assert bool(result.converged), (
            f"GMRES did not converge "
            f"(residual={float(result.residual_norm):.2e})"
        )
        assert jnp.allclose(result.value, x_ref, atol=1e-3, rtol=1e-3), (
            f"GMRES solution mismatch "
            f"(max diff={float(jnp.max(jnp.abs(result.value - x_ref))):.2e})"
        )

    def test_cg_zero_rhs_returns_zero(self):
        n = 16
        b = jnp.zeros(n, dtype=jnp.float32)
        matvec = _laplacian_1d_matvec_unsharded(n)
        result = sharded_cg(matvec, b, backend="loop")
        assert jnp.allclose(result.value, jnp.zeros(n), atol=1e-7)

    def test_cg_auto_backend_returns_reasonable_solution(self):
        """Auto backend (lineax path with float32 limits) gives a usable
        answer — the absolute tolerance is looser to accommodate
        lineax's stricter convergence check on float32 problems.
        """
        n = 32
        A = _laplacian_1d_dense(n)
        b = jnp.arange(n, dtype=jnp.float32) + 1.0
        x_ref = jnp.linalg.solve(A, b)

        matvec = _laplacian_1d_matvec_unsharded(n)
        # Loose tolerances so lineax does converge on float32.
        result = sharded_cg(
            matvec, b, rtol=1e-3, atol=1e-4, max_iters=500, backend="auto",
        )
        # Relative error should be small even if .converged is False.
        rel_err = jnp.linalg.norm(result.value - x_ref) / jnp.linalg.norm(x_ref)
        assert float(rel_err) < 1e-2, (
            f"auto backend gave bad solution (rel_err={float(rel_err):.2e})"
        )


# ---------------------------------------------------------------------------
# Sharded correctness — the load-bearing acceptance test for v0.3.0 A5.
# ---------------------------------------------------------------------------


class TestShardedCorrectness:
    """Bit-compat between sharded run and unsharded reference."""

    def test_cg_sharded_matches_unsharded(self):
        mesh = _make_4_device_mesh()
        n_per_shard = 16
        n = n_per_shard * 4

        b = jnp.arange(n, dtype=jnp.float32) + 1.0

        # Unsharded reference.
        matvec_ref = _laplacian_1d_matvec_unsharded(n)
        x_ref = sharded_cg(matvec_ref, b, max_iters=500, backend="loop").value

        # Sharded — both backends.
        matvec_sh = _laplacian_1d_matvec_sharded(mesh, n_per_shard)
        result = sharded_cg(
            matvec_sh, b, mesh=mesh, in_specs=P("devices"),
            max_iters=500, backend="loop",
        )
        x_sh = jax.device_get(result.value)
        x_ref_np = jax.device_get(x_ref)

        # ATOL 1e-5 per A5 acceptance criterion.
        assert jnp.allclose(jnp.asarray(x_sh), jnp.asarray(x_ref_np),
                            atol=1e-5, rtol=1e-4), (
            f"sharded vs unsharded mismatch "
            f"(max diff={float(jnp.max(jnp.abs(jnp.asarray(x_sh) - jnp.asarray(x_ref_np)))):.2e})"
        )

    def test_gmres_sharded_matches_unsharded(self):
        mesh = _make_4_device_mesh()
        n_per_shard = 16
        n = n_per_shard * 4

        b = jnp.arange(n, dtype=jnp.float32) + 1.0

        matvec_ref = _laplacian_1d_matvec_unsharded(n)
        x_ref = sharded_gmres(
            matvec_ref, b, restart=n, max_iters=2 * n, backend="loop",
        ).value

        matvec_sh = _laplacian_1d_matvec_sharded(mesh, n_per_shard)
        result = sharded_gmres(
            matvec_sh, b, mesh=mesh, in_specs=P("devices"),
            restart=n, max_iters=2 * n, backend="loop",
        )
        x_sh = jax.device_get(result.value)
        x_ref_np = jax.device_get(x_ref)

        assert jnp.allclose(jnp.asarray(x_sh), jnp.asarray(x_ref_np),
                            atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Differentiability — confirms the FMI 3 directional-derivative path
# keeps working on sharded operators (A5 acceptance criterion).
# ---------------------------------------------------------------------------


class TestDifferentiability:

    def test_jvp_through_sharded_cg_matches_unsharded(self):
        """jax.jvp through sharded_cg matches jax.jvp through the unsharded
        reference on the same problem.  This is the FMI 3 directional-
        derivative export path's regression guard.
        """
        mesh = _make_4_device_mesh()
        n_per_shard = 8
        n = n_per_shard * 4

        matvec_ref = _laplacian_1d_matvec_unsharded(n)
        matvec_sh = _laplacian_1d_matvec_sharded(mesh, n_per_shard)

        def solve_unsharded(b):
            return sharded_cg(matvec_ref, b, max_iters=500, backend="loop").value

        def solve_sharded(b):
            return sharded_cg(
                matvec_sh, b, mesh=mesh, in_specs=P("devices"),
                max_iters=500, backend="loop",
            ).value

        b = jnp.arange(n, dtype=jnp.float32) + 1.0
        v = jnp.ones(n, dtype=jnp.float32)

        x_ref, dx_ref = jax.jvp(solve_unsharded, (b,), (v,))
        x_sh, dx_sh = jax.jvp(solve_sharded, (b,), (v,))

        # Primal solutions agree (already covered above, but sanity).
        assert jnp.allclose(jnp.asarray(jax.device_get(x_sh)),
                            jnp.asarray(jax.device_get(x_ref)),
                            atol=1e-5, rtol=1e-4)
        # Directional derivatives agree.
        assert jnp.allclose(jnp.asarray(jax.device_get(dx_sh)),
                            jnp.asarray(jax.device_get(dx_ref)),
                            atol=1e-4, rtol=1e-3), (
            f"jvp mismatch "
            f"(max diff={float(jnp.max(jnp.abs(jnp.asarray(jax.device_get(dx_sh)) - jnp.asarray(jax.device_get(dx_ref))))):.2e})"
        )


# ---------------------------------------------------------------------------
# Construction-time validation — A5 plan calls for clear errors when
# the matvec / mesh / in_specs disagree.
# ---------------------------------------------------------------------------


class TestValidation:

    def test_non_callable_matvec_rejected(self):
        with pytest.raises(TypeError, match="matvec must be callable"):
            sharded_cg("not a function", jnp.ones(4))

    def test_non_array_b_rejected(self):
        with pytest.raises(TypeError, match="b must be array-like"):
            sharded_cg(lambda x: x, "not an array")

    def test_mesh_without_specs_rejected(self):
        mesh = _make_4_device_mesh()
        with pytest.raises(ValueError, match="in_specs must also be provided"):
            sharded_cg(lambda x: x, jnp.ones(4), mesh=mesh)

    def test_unknown_mesh_axis_rejected(self):
        mesh = _make_4_device_mesh()  # axis_names=("devices",)
        with pytest.raises(ValueError, match="not in mesh.axis_names"):
            sharded_cg(
                lambda x: x, jnp.ones(4),
                mesh=mesh, in_specs=P("nonexistent_axis"),
            )

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            sharded_cg(lambda x: x, jnp.ones(4), backend="wat")

    def test_lineax_backend_requested_without_lineax(self, monkeypatch):
        """When backend='lineax' but lineax import fails, raise a clear error."""
        import sys
        # Make sure the import really fails.
        original = sys.modules.pop("lineax", None)
        monkeypatch.setitem(sys.modules, "lineax", None)
        try:
            with pytest.raises(RuntimeError, match="lineax not installed"):
                sharded_cg(lambda x: x, jnp.ones(4), backend="lineax")
        finally:
            if original is not None:
                sys.modules["lineax"] = original


# ---------------------------------------------------------------------------
# Stability tagging — these functions are the v0.4.0 commitment surface
# per §A5 + §A6.
# ---------------------------------------------------------------------------


class TestStabilityTagging:
    def test_sharded_cg_tagged_stable(self):
        from maddening.core.compliance.metadata import StabilityLevel
        assert sharded_cg._stability_level == StabilityLevel.STABLE

    def test_sharded_gmres_tagged_stable(self):
        from maddening.core.compliance.metadata import StabilityLevel
        assert sharded_gmres._stability_level == StabilityLevel.STABLE
