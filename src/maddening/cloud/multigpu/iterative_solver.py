"""Sharded sparse iterative solvers — v0.3.0 §A5.

Builds on the IFT branch's matrix-free lineax integration
(see :func:`maddening.core.graph_manager._ift_solve_bwd`).
This module extends the same pattern to **distributed** matvec
operators: the user supplies a callable ``matvec(x) → A · x`` whose
internal implementation uses ``shard_map`` to run across multiple
devices, and we wrap it in a small public API for CG / GMRES.

Why this module exists (v0.3.0 plan §A5):

* FVM PISO pressure correction in MIME v0.5.0 (against MADDENING v0.4.0
  — the hard downstream gate) needs scalable Krylov solves on
  distributed operators.
* The FMI 3.0 substrate (§A1) commits us to exposing exact directional
  derivatives via ``fmi3GetDirectionalDerivative`` → ``jax.jvp`` /
  ``jax.vjp``.  Differentiating through a sharded solve requires the
  solver participate in autodiff cleanly.

Design:

* The solver accepts a **global** matvec — i.e. one that takes the
  full-shape vector ``x`` as a sharded JAX array (typically
  ``NamedSharding``-ed) and returns ``A · x`` as the same shape.  The
  matvec is responsible for its own ``shard_map``.  The solver runs
  the Krylov iteration at the outer (un-shard_mapped) level, where
  JAX's ``vdot`` / ``linalg.norm`` correctly account for the
  partitioning of the inputs.
* When ``lineax`` is available we route through
  ``lineax.FunctionLinearOperator`` + ``lineax.GMRES`` / ``lineax.CG``
  to reuse the IFT branch's battle-tested adjoint path.
* When ``lineax`` cannot consume a sharded matvec cleanly (the
  contingency anticipated in §A5's risk callouts), the hand-rolled
  ``lax.fori_loop`` paths in ``_cg_loop`` and ``_gmres_loop`` provide
  a fallback that depends only on stock JAX.

Differentiability: forward-mode JVP composes naturally with stock
JAX operators in both the lineax and the fori_loop fallback; the
``custom_vjp`` adjoint pattern from the IFT branch extends here too,
but the v0.3.0 substrate exposes the primal solver only — adjoint
plumbing through the user-supplied matvec is the caller's responsibility
(``jax.linear_transpose`` on the matvec gives the transpose action).

Preconditioning: Jacobi-only for v0.3.0 (per §A5 scope).  The interface
accepts an optional ``preconditioner`` callable for forward compat
with AMG / block-Jacobi in v0.4.0+.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability


# ---------------------------------------------------------------------------
# Result struct
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedSolveResult:
    """Return type for :func:`sharded_cg` and :func:`sharded_gmres`.

    Attributes
    ----------
    value : jax.Array
        The approximate solution ``x`` to ``A x = b``.
    converged : jax.Array
        Boolean scalar — True iff the solver converged within tolerance.
    iters : jax.Array
        Integer scalar — number of iterations used.
    residual_norm : jax.Array
        Final ``||b - A x||_2``.
    """
    value: jax.Array
    converged: jax.Array
    iters: jax.Array
    residual_norm: jax.Array


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_solver_inputs(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    mesh: Optional[Mesh],
    in_specs: Optional[PartitionSpec],
    name: str,
) -> None:
    if not callable(matvec):
        raise TypeError(f"{name}: matvec must be callable, got {type(matvec)!r}")
    if not hasattr(b, "shape"):
        raise TypeError(f"{name}: b must be array-like, got {type(b)!r}")
    if mesh is not None and not isinstance(mesh, Mesh):
        raise TypeError(f"{name}: mesh must be jax.sharding.Mesh, got {type(mesh)!r}")
    if mesh is not None and in_specs is None:
        raise ValueError(
            f"{name}: when mesh is provided, in_specs must also be provided "
            "(the PartitionSpec describing how b is partitioned across mesh)."
        )
    if mesh is not None and in_specs is not None:
        if not isinstance(in_specs, PartitionSpec):
            raise TypeError(
                f"{name}: in_specs must be jax.sharding.PartitionSpec, "
                f"got {type(in_specs)!r}"
            )
        # Catch the most common foot-gun: spec naming an axis the mesh
        # doesn't have.  The shard_map call site would error later but
        # the message is much less actionable.
        mesh_axes = set(mesh.axis_names)
        for ax in in_specs:
            if ax is None:
                continue
            if isinstance(ax, str) and ax not in mesh_axes:
                raise ValueError(
                    f"{name}: in_specs references mesh axis {ax!r} which is "
                    f"not in mesh.axis_names={sorted(mesh_axes)}."
                )
            if isinstance(ax, tuple):
                for sub in ax:
                    if sub not in mesh_axes:
                        raise ValueError(
                            f"{name}: in_specs references mesh axis "
                            f"{sub!r} which is not in "
                            f"mesh.axis_names={sorted(mesh_axes)}."
                        )


def _materialise_b(
    b: jax.Array,
    *,
    mesh: Optional[Mesh],
    in_specs: Optional[PartitionSpec],
) -> jax.Array:
    """Place b on the right sharding so the solver inherits the layout."""
    if mesh is None:
        return jnp.asarray(b)
    sharding = NamedSharding(mesh, in_specs)
    return jax.device_put(jnp.asarray(b), sharding)


# ---------------------------------------------------------------------------
# Hand-rolled fallbacks (no lineax dependency)
# ---------------------------------------------------------------------------


def _cg_loop(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    x0: jax.Array,
    rtol: float,
    atol: float,
    max_iters: int,
    preconditioner: Optional[Callable[[jax.Array], jax.Array]],
) -> SharedSolveResult:
    """Preconditioned conjugate-gradient on a global sharded matvec.

    Inner products use ``jnp.vdot``, which is correctly all-reduced
    when its inputs are sharded across a mesh axis (the partial sums
    on each shard get summed automatically by XLA).
    """
    M = preconditioner if preconditioner is not None else (lambda r: r)
    b_norm = jnp.linalg.norm(b)
    # Guard against a zero RHS — solution is trivially x0.
    tol2 = jnp.maximum(atol, rtol * b_norm) ** 2

    r0 = b - matvec(x0)
    z0 = M(r0)
    rho0 = jnp.vdot(r0, z0).real

    def cond(state):
        x, r, z, p, rho, iters = state
        not_converged = jnp.vdot(r, r).real > tol2
        not_exhausted = iters < max_iters
        return jnp.logical_and(not_converged, not_exhausted)

    def body(state):
        x, r, z, p, rho, iters = state
        Ap = matvec(p)
        alpha = rho / jnp.maximum(jnp.vdot(p, Ap).real, jnp.finfo(b.dtype).tiny)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = M(r_new)
        rho_new = jnp.vdot(r_new, z_new).real
        beta = rho_new / jnp.maximum(rho, jnp.finfo(b.dtype).tiny)
        p_new = z_new + beta * p
        return (x_new, r_new, z_new, p_new, rho_new, iters + 1)

    init = (x0, r0, z0, z0, rho0, jnp.int32(0))
    x, r, _, _, _, iters = lax.while_loop(cond, body, init)

    res_norm = jnp.linalg.norm(r)
    converged = res_norm <= jnp.sqrt(tol2)
    return SharedSolveResult(
        value=x, converged=converged, iters=iters, residual_norm=res_norm,
    )


def _gmres_inner_cycle(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    x0: jax.Array,
    restart: int,
) -> tuple[jax.Array, jax.Array]:
    """One restart cycle of GMRES.  Returns (x_new, residual_norm)."""
    r = b - matvec(x0)
    beta = jnp.linalg.norm(r)
    eps = jnp.finfo(b.dtype).tiny
    q0 = r / jnp.maximum(beta, eps)

    n = b.shape[0]
    # Arnoldi basis: (restart + 1) vectors each of length n.
    Q = jnp.zeros((restart + 1, n), dtype=b.dtype).at[0].set(q0)
    # Hessenberg: (restart + 1) x restart.
    H = jnp.zeros((restart + 1, restart), dtype=b.dtype)

    def arnoldi_step(j, carry):
        Q, H = carry
        v = matvec(Q[j])
        # Modified Gram-Schmidt against existing basis vectors.
        def mgs_step(i, vh):
            v_, H_ = vh
            h_ij = jnp.vdot(Q[i], v_)
            v_ = v_ - h_ij * Q[i]
            H_ = H_.at[i, j].set(h_ij)
            return (v_, H_)
        v, H = lax.fori_loop(0, j + 1, mgs_step, (v, H))
        h_next = jnp.linalg.norm(v)
        H = H.at[j + 1, j].set(h_next)
        Q = Q.at[j + 1].set(v / jnp.maximum(h_next, eps))
        return (Q, H)

    Q, H = lax.fori_loop(0, restart, arnoldi_step, (Q, H))

    # Solve the (restart+1) x restart least-squares Hessenberg system
    # minimising ||beta e1 - H y||_2; the residual norm of the LSQ
    # solution is the GMRES residual norm.
    e1 = jnp.zeros(restart + 1, dtype=b.dtype).at[0].set(beta)
    y, *_ = jnp.linalg.lstsq(H, e1, rcond=None)
    x_new = x0 + Q[:restart].T @ y

    r_new = b - matvec(x_new)
    return x_new, jnp.linalg.norm(r_new)


def _gmres_loop(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    x0: jax.Array,
    rtol: float,
    atol: float,
    restart: int,
    max_iters: int,
    preconditioner: Optional[Callable[[jax.Array], jax.Array]],
) -> SharedSolveResult:
    """Restarted GMRES fallback.

    Limitations of the fallback (versus the lineax path):
    * Modified Gram-Schmidt only — no Householder reorthogonalisation.
    * Preconditioning is applied as a left preconditioner on the
      matvec (``matvec' = M @ A``); right preconditioning isn't
      exposed in v0.3.0.

    These are acceptable for the v0.3.0 substrate.  The lineax path is
    preferred when lineax can consume the sharded matvec; this
    fallback is for the contingency where it cannot.
    """
    if preconditioner is not None:
        original_matvec = matvec
        M = preconditioner

        def matvec_pc(x):
            return M(original_matvec(x))

        # Also precondition b on the left.
        b_eff = M(b)
    else:
        matvec_pc = matvec
        b_eff = b

    b_norm = jnp.linalg.norm(b_eff)
    tol = jnp.maximum(atol, rtol * b_norm)
    restart = max(1, int(restart))
    n_cycles = max(1, (max_iters + restart - 1) // restart)

    def cycle_body(i, carry):
        x, res, converged, cycles_used = carry
        # Skip the actual work once converged, but keep the loop shape
        # static for JIT.
        def do_cycle(_):
            x_new, res_new = _gmres_inner_cycle(matvec_pc, b_eff, x, restart)
            return (x_new, res_new, res_new <= tol, cycles_used + 1)

        def skip(_):
            return (x, res, converged, cycles_used)

        return lax.cond(converged, skip, do_cycle, operand=None)

    init = (x0, jnp.linalg.norm(b_eff - matvec_pc(x0)),
            jnp.array(False), jnp.int32(0))
    x, res, converged, cycles_used = lax.fori_loop(
        0, n_cycles, cycle_body, init,
    )
    return SharedSolveResult(
        value=x, converged=converged,
        iters=cycles_used * restart, residual_norm=res,
    )


# ---------------------------------------------------------------------------
# Lineax-backed path
# ---------------------------------------------------------------------------


def _try_lineax_solve(
    solver_kind: str,
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    rtol: float,
    atol: float,
    restart: int,
    max_iters: int,
) -> Optional[SharedSolveResult]:
    """Attempt the lineax-backed path.  Returns None if lineax is unavailable.

    Raises any error other than ``ImportError`` — those propagate (a
    user expecting lineax to be installed should see the real failure).
    """
    try:
        import lineax as lx  # noqa: PLC0415
    except ImportError:
        return None

    tags = ()
    if solver_kind == "cg":
        # lineax CG requires the user to assert positive-semidefiniteness.
        # Callers who reach sharded_cg are committing to that (CG assumes
        # SPD); pass the tag so lineax's runtime check passes.
        tags = (lx.positive_semidefinite_tag, lx.symmetric_tag)
    op = lx.FunctionLinearOperator(
        matvec, jax.eval_shape(lambda: b), tags=tags,
    )
    if solver_kind == "cg":
        # CG assumes SPD; we expose the convergence info but trust the user.
        # max_steps gives the iteration budget.
        solver = lx.CG(rtol=rtol, atol=atol, max_steps=max_iters)
    elif solver_kind == "gmres":
        # See the restart-gotcha comment in graph_manager._ift_solve_bwd.
        n = b.shape[0]
        restart_clamped = min(int(n), int(restart))
        solver = lx.GMRES(
            rtol=rtol, atol=atol, restart=restart_clamped,
            max_steps=max(4 * restart_clamped, max_iters),
        )
    else:
        raise ValueError(f"Unknown solver_kind {solver_kind!r}")
    # ``throw=False``: hand max-steps and other non-fatal failures back
    # as ``result.result`` instead of raising.  Users inspect
    # ``SharedSolveResult.converged`` to decide what to do — the same
    # contract the loop fallback follows.
    result = lx.linear_solve(op, b, solver=solver, throw=False)
    res_norm = jnp.linalg.norm(b - matvec(result.value))
    # lineax's RESULTS object doesn't always populate iters; treat as -1.
    stats = result.stats or {}
    n_iters = jnp.int32(stats.get("num_steps", -1))
    converged = result.result == lx.RESULTS.successful
    return SharedSolveResult(
        value=result.value,
        converged=jnp.asarray(converged),
        iters=n_iters,
        residual_norm=res_norm,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@stability(StabilityLevel.STABLE)
def sharded_cg(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    mesh: Optional[Mesh] = None,
    in_specs: Optional[PartitionSpec] = None,
    x0: Optional[jax.Array] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_iters: int = 200,
    preconditioner: Optional[Callable[[jax.Array], jax.Array]] = None,
    backend: str = "auto",
) -> SharedSolveResult:
    """Preconditioned conjugate gradient on a sharded matvec.

    Parameters
    ----------
    matvec : callable
        ``matvec(x) -> A @ x``.  The matvec is responsible for any
        ``shard_map`` it requires; the solver itself runs at the outer
        (global) level.  Must be self-adjoint and positive-definite —
        for general operators use :func:`sharded_gmres` instead.
    b : jax.Array
        Right-hand side.  1-D.  Sharded according to ``in_specs`` when
        ``mesh`` is provided.
    mesh, in_specs : optional
        When both provided, ``b`` and ``x`` are placed on the mesh with
        the given PartitionSpec.  When omitted, the solver runs on
        whatever device(s) JAX chooses for ``b`` — single-device or a
        pre-sharded input.
    x0 : jax.Array, optional
        Initial guess.  Defaults to zeros_like(b).
    rtol, atol : float
        Stopping criterion: ``||r|| <= max(atol, rtol * ||b||)``.
    max_iters : int
        Iteration budget.
    preconditioner : callable, optional
        ``M(r) -> M @ r``.  Applied as a left preconditioner.  v0.3.0
        ships only the identity (None) — Jacobi or other preconditioners
        are the caller's responsibility.
    backend : {"auto", "lineax", "loop"}
        Solver backend.  ``"auto"`` (default) tries lineax then falls
        back to the hand-rolled loop.  ``"loop"`` forces the
        loop fallback (useful if lineax misbehaves with shard_map).

    Returns
    -------
    SharedSolveResult
        ``.value`` (solution), ``.converged``, ``.iters``, ``.residual_norm``.

    Stability
    ---------
    This signature is tagged ``@stability(STABLE)`` — see
    ``docs/developer_guide/stability_report.md``.  The v0.4.0 commitment
    (sharded FVM in MIME) reads against this surface.
    """
    _validate_solver_inputs(
        matvec, b, mesh=mesh, in_specs=in_specs, name="sharded_cg",
    )
    b = _materialise_b(b, mesh=mesh, in_specs=in_specs)
    if x0 is None:
        x0 = jnp.zeros_like(b)
    else:
        x0 = _materialise_b(x0, mesh=mesh, in_specs=in_specs)

    if backend == "lineax":
        result = _try_lineax_solve(
            "cg", matvec, b, rtol=rtol, atol=atol,
            restart=0, max_iters=max_iters,
        )
        if result is None:
            raise RuntimeError("backend='lineax' requested but lineax not installed")
        return result
    if backend == "loop":
        return _cg_loop(
            matvec, b, x0=x0, rtol=rtol, atol=atol,
            max_iters=max_iters, preconditioner=preconditioner,
        )
    if backend != "auto":
        raise ValueError(f"Unknown backend {backend!r}; expected auto/lineax/loop")

    # auto: try lineax first if no preconditioner (lineax doesn't take ours);
    # otherwise loop.
    if preconditioner is None:
        result = _try_lineax_solve(
            "cg", matvec, b, rtol=rtol, atol=atol,
            restart=0, max_iters=max_iters,
        )
        if result is not None:
            return result
    return _cg_loop(
        matvec, b, x0=x0, rtol=rtol, atol=atol,
        max_iters=max_iters, preconditioner=preconditioner,
    )


@stability(StabilityLevel.STABLE)
def sharded_gmres(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    mesh: Optional[Mesh] = None,
    in_specs: Optional[PartitionSpec] = None,
    x0: Optional[jax.Array] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    restart: int = 50,
    max_iters: int = 200,
    preconditioner: Optional[Callable[[jax.Array], jax.Array]] = None,
    backend: str = "auto",
) -> SharedSolveResult:
    """Restarted GMRES on a sharded matvec.

    See :func:`sharded_cg` for the matvec / mesh / in_specs contract.

    Parameters
    ----------
    restart : int
        Krylov subspace dimension before restart.  Capped to ``b.shape[0]``
        inside the solver.  Lineax's default is 20, which silently
        produces low-rank approximations on problems larger than 20
        unknowns — we default to 50 here.  See the regression test in
        ``tests/core/test_coupling_ift_lineax.py`` for the historical
        gotcha.

    Stability
    ---------
    Tagged ``@stability(STABLE)``.  v0.4.0 / M3 commitment: MIME's
    FVM PISO pressure correction reads against this signature.
    """
    _validate_solver_inputs(
        matvec, b, mesh=mesh, in_specs=in_specs, name="sharded_gmres",
    )
    b = _materialise_b(b, mesh=mesh, in_specs=in_specs)
    if x0 is None:
        x0 = jnp.zeros_like(b)
    else:
        x0 = _materialise_b(x0, mesh=mesh, in_specs=in_specs)

    if backend == "lineax":
        result = _try_lineax_solve(
            "gmres", matvec, b, rtol=rtol, atol=atol,
            restart=restart, max_iters=max_iters,
        )
        if result is None:
            raise RuntimeError("backend='lineax' requested but lineax not installed")
        return result
    if backend == "loop":
        return _gmres_loop(
            matvec, b, x0=x0, rtol=rtol, atol=atol,
            restart=restart, max_iters=max_iters,
            preconditioner=preconditioner,
        )
    if backend != "auto":
        raise ValueError(f"Unknown backend {backend!r}; expected auto/lineax/loop")

    if preconditioner is None:
        result = _try_lineax_solve(
            "gmres", matvec, b, rtol=rtol, atol=atol,
            restart=restart, max_iters=max_iters,
        )
        if result is not None:
            return result
    return _gmres_loop(
        matvec, b, x0=x0, rtol=rtol, atol=atol,
        restart=restart, max_iters=max_iters,
        preconditioner=preconditioner,
    )


__all__ = [
    "SharedSolveResult",
    "sharded_cg",
    "sharded_gmres",
]
