"""Tests for the sharded sparse iterative solver substrate (added in v0.3.0).

The conftest in this directory forces XLA to expose 16 virtual CPU
devices, so the 4-device mesh tests run locally without real GPUs.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

import jax
import jax.numpy as jnp
from jax import lax
from jax import shard_map
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

    def test_auto_backend_uses_lineax_not_the_loop_fallback(self, monkeypatch):
        """``auto`` with no preconditioner really routes through lineax.

        While lineax was an optional extra, ``auto`` silently fell back
        to the loop backend when the import failed — a solve could
        quietly change backend depending on how MADDENING was
        installed.  lineax is a base dependency now, so the only
        remaining reason ``auto`` takes the loop is a preconditioner.
        """
        from maddening.cloud.multigpu import iterative_solver as solver_mod

        calls = []
        real = solver_mod._lineax_solve

        def _spy(kind, *args, **kwargs):
            calls.append(kind)
            return real(kind, *args, **kwargs)

        monkeypatch.setattr(solver_mod, "_lineax_solve", _spy)
        n = 16
        matvec = _laplacian_1d_matvec_unsharded(n)
        b = jnp.ones(n, dtype=jnp.float32)
        sharded_cg(matvec, b, rtol=1e-3, atol=1e-4, max_iters=200,
                   backend="auto")
        assert calls == ["cg"], calls

        calls.clear()
        # With a preconditioner, auto must still choose the loop.
        sharded_cg(matvec, b, rtol=1e-3, atol=1e-4, max_iters=200,
                   backend="auto", preconditioner=lambda r: r)
        assert calls == [], calls


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


# ---------------------------------------------------------------------------
# Stability tagging — these functions are the sharded solver's public
# surface.
# ---------------------------------------------------------------------------


class TestStabilityTagging:
    def test_sharded_cg_tagged_stable(self):
        from maddening.core.compliance.metadata import StabilityLevel
        assert sharded_cg._stability_level == StabilityLevel.STABLE

    def test_sharded_gmres_tagged_stable(self):
        from maddening.core.compliance.metadata import StabilityLevel
        assert sharded_gmres._stability_level == StabilityLevel.STABLE


# ---------------------------------------------------------------------------
# v0.4.0: preconditioners + gradient parity through the (preconditioned)
# solve.  The IFT / custom_vjp adjoint of a node that solves with
# sharded_cg must stay exact with the preconditioner applied in the
# adjoint solve too — checked here against the dense reference.
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from maddening.cloud.multigpu.iterative_solver import (  # noqa: E402
    block_jacobi_preconditioner,
    jacobi_preconditioner,
)


def _scaled_laplacian(n, scale):
    """SPD ``D A D`` with a wide diagonal spread: CG needs the preconditioner."""
    A = _laplacian_1d_dense(n)
    D = jnp.diag(scale)
    return D @ A @ D


def _blocks_of(M, bs):
    n = M.shape[0]
    return jnp.stack([M[i:i + bs, i:i + bs] for i in range(0, n, bs)])


class TestPreconditioned:

    def test_block_jacobi_matches_dense_and_cuts_iterations(self):
        n, bs = 32, 4
        scale = jnp.asarray(np.geomspace(1.0, 100.0, n), jnp.float32)
        M = _scaled_laplacian(n, scale)
        b = jnp.arange(n, dtype=jnp.float32) + 1.0
        x_ref = jnp.linalg.solve(M, b)
        mv = lambda x: M @ x  # noqa: E731
        plain = sharded_cg(mv, b, max_iters=2000, rtol=1e-6, atol=1e-8, backend="loop")
        pc = sharded_cg(mv, b, max_iters=2000, rtol=1e-6, atol=1e-8, backend="loop",
                        preconditioner=block_jacobi_preconditioner(_blocks_of(M, bs)))
        for r in (plain, pc):
            assert jnp.allclose(r.value, x_ref, rtol=1e-3, atol=1e-3)
        assert int(pc.iters) < int(plain.iters), (int(pc.iters), int(plain.iters))
        jac = sharded_cg(mv, b, max_iters=2000, rtol=1e-6, atol=1e-8, backend="loop",
                         preconditioner=jacobi_preconditioner(jnp.diag(M)))
        assert jnp.allclose(jac.value, x_ref, rtol=1e-3, atol=1e-3)
        assert int(jac.iters) < int(plain.iters)

    def test_lineax_backend_rejects_preconditioner(self):
        # No importorskip: lineax is a base dependency as of v0.4.0.
        with pytest.raises(ValueError, match="cannot apply a preconditioner"):
            sharded_cg(lambda x: x, jnp.ones(4), backend="lineax",
                       preconditioner=lambda r: r)
        with pytest.raises(ValueError, match="cannot apply a preconditioner"):
            sharded_gmres(lambda x: x, jnp.ones(4), backend="lineax",
                          preconditioner=lambda r: r)

    def test_block_jacobi_shape_validation(self):
        with pytest.raises(ValueError, match="n_blocks, bs, bs"):
            block_jacobi_preconditioner(jnp.ones((3, 2)))
        M = block_jacobi_preconditioner(jnp.eye(2)[None].repeat(3, 0))
        with pytest.raises(ValueError, match="built for n=6"):
            M(jnp.ones(5))

    @pytest.mark.parametrize("solver", ["cg", "gmres"])
    def test_grad_and_jvp_through_preconditioned_solve_match_dense(self, solver):
        """Reverse and forward mode through the differentiable, preconditioned
        solve equal the dense linear-solve derivatives — w.r.t. the RHS and
        w.r.t. the operator coefficients the matvec closes over."""
        n, bs = 16, 4
        scale = jnp.asarray(np.geomspace(1.0, 10.0, n), jnp.float32)
        A0 = _scaled_laplacian(n, scale)
        b0 = jnp.arange(n, dtype=jnp.float32) + 1.0
        blocks = _blocks_of(A0, bs)

        def solve_it(coef, b):
            A = A0 + coef * jnp.eye(n, dtype=jnp.float32)
            mv = lambda x: A @ x  # noqa: E731
            kw = dict(max_iters=500, rtol=1e-7, atol=1e-9, backend="loop",
                      preconditioner=block_jacobi_preconditioner(blocks),
                      differentiable=True)
            r = sharded_cg(mv, b, **kw) if solver == "cg" else sharded_gmres(mv, b, restart=16, **kw)
            return r.value

        def dense(coef, b):
            return jnp.linalg.solve(A0 + coef * jnp.eye(n, dtype=jnp.float32), b)

        loss = lambda coef, b: jnp.sum(solve_it(coef, b) ** 2)  # noqa: E731
        loss_d = lambda coef, b: jnp.sum(dense(coef, b) ** 2)  # noqa: E731
        c0 = jnp.asarray(0.5, jnp.float32)
        g_c, g_b = jax.grad(loss, argnums=(0, 1))(c0, b0)
        d_c, d_b = jax.grad(loss_d, argnums=(0, 1))(c0, b0)
        assert jnp.allclose(g_c, d_c, rtol=2e-3, atol=1e-3), (g_c, d_c)
        assert jnp.allclose(g_b, d_b, rtol=2e-3, atol=1e-3)
        # forward mode
        v = jnp.ones_like(b0)
        _, t = jax.jvp(lambda b: solve_it(c0, b), (b0,), (v,))
        _, t_d = jax.jvp(lambda b: dense(c0, b), (b0,), (v,))
        assert jnp.allclose(t, t_d, rtol=2e-3, atol=1e-3)
        # diagnostics on the differentiable path
        # float32 CG plateaus around 1e-6 relative; ask for a reachable
        # tolerance when checking the post-hoc diagnostics.
        r = sharded_cg(lambda x: A0 @ x, b0, max_iters=500, rtol=1e-4, backend="loop",
                       differentiable=True)
        assert bool(r.converged) and int(r.iters) == -1 and float(r.residual_norm) < 1e-2

    def test_grad_through_sharded_preconditioned_cg_on_4_device_mesh(self):
        """Sharded matvec + preconditioner + reverse mode: parity with the
        unsharded reference (the plan's 4-device CPU-virtual gate)."""
        mesh = _make_4_device_mesh()
        n_per_shard = 8
        n = n_per_shard * 4
        matvec_ref = _laplacian_1d_matvec_unsharded(n)
        matvec_sh = _laplacian_1d_matvec_sharded(mesh, n_per_shard)
        diag = jnp.full((n,), 2.0, jnp.float32)
        pc = jacobi_preconditioner(diag)

        def loss_sh(b):
            x = sharded_cg(matvec_sh, b, mesh=mesh, in_specs=P("devices"),
                           max_iters=500, backend="loop", preconditioner=pc,
                           differentiable=True).value
            return jnp.sum(x ** 2)

        def loss_ref(b):
            x = sharded_cg(matvec_ref, b, max_iters=500, backend="loop",
                           preconditioner=pc, differentiable=True).value
            return jnp.sum(x ** 2)

        b = jnp.arange(n, dtype=jnp.float32) + 1.0
        g_sh = jax.device_get(jax.grad(loss_sh)(b))
        g_ref = jax.device_get(jax.grad(loss_ref)(b))
        assert jnp.allclose(jnp.asarray(g_sh), jnp.asarray(g_ref), rtol=1e-3, atol=1e-3)


@pytest.mark.slow
class TestGradientParityAtScale:
    """The v0.4.0 plan's gradient-parity gate at real-mesh size, the half
    that runs on CPU-virtual devices: 10^5 DOF over 4 shards, reverse and
    forward mode through the differentiable, Jacobi-preconditioned CG,
    against the unsharded reference."""

    def test_grad_and_jvp_through_sharded_cg_at_1e5_dof(self):
        mesh = _make_4_device_mesh()
        n_per_shard = 25_000
        n = n_per_shard * 4
        matvec_ref = _laplacian_1d_matvec_unsharded(n)
        matvec_sh = _laplacian_1d_matvec_sharded(mesh, n_per_shard)
        pc = jacobi_preconditioner(jnp.full((n,), 2.0, jnp.float32))
        kw = dict(max_iters=3000, rtol=1e-6, atol=1e-8, backend="loop",
                  preconditioner=pc, differentiable=True)
        # a smooth right-hand side keeps the CG iteration count moderate
        b = jnp.sin(jnp.linspace(0.0, 6.0, n, dtype=jnp.float32)) + 0.1

        def loss_sh(b):
            return jnp.sum(sharded_cg(matvec_sh, b, mesh=mesh, in_specs=P("devices"), **kw).value ** 2)

        def loss_ref(b):
            return jnp.sum(sharded_cg(matvec_ref, b, **kw).value ** 2)

        g_sh = np.asarray(jax.device_get(jax.grad(loss_sh)(b)))
        g_ref = np.asarray(jax.device_get(jax.grad(loss_ref)(b)))
        scale = np.max(np.abs(g_ref))
        assert scale > 0
        np.testing.assert_allclose(g_sh / scale, g_ref / scale, rtol=1e-3, atol=1e-3)
        v = jnp.ones_like(b)
        _, t_sh = jax.jvp(lambda bb: sharded_cg(matvec_sh, bb, mesh=mesh, in_specs=P("devices"),
                                                **kw).value, (b,), (v,))
        _, t_ref = jax.jvp(lambda bb: sharded_cg(matvec_ref, bb, **kw).value, (b,), (v,))
        t_sh, t_ref = np.asarray(jax.device_get(t_sh)), np.asarray(jax.device_get(t_ref))
        s2 = np.max(np.abs(t_ref))
        np.testing.assert_allclose(t_sh / s2, t_ref / s2, rtol=1e-3, atol=1e-3)
