"""The implicit-function-theorem solve of a coupled block and its adjoint.

Moved verbatim out of ``maddening.core.graph_manager``.  Private.
"""

from __future__ import annotations

import os
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

# ``lineax`` is a base dependency (v0.4.0) but is still imported lazily
# inside ``_ift_linear_solve``: it pulls in equinox + jaxtyping, an order
# of magnitude more import time than ``import maddening`` itself costs.
# Only users who opt into ``solver='ift'`` pay it.  The import needs no
# guard — a missing lineax is now an installation fault, not a
# user-recoverable "install the extra" condition.

from maddening.core._pow2_frame import pow2_frame
from maddening.core.coupling._bounds import _F_dispatch, _analysis_dtype
from maddening.core.coupling._fixed_point import _fixed_point_while


def _ift_solve_impl(
    step_pure, x0, consts, accel_init, first_res, threshold, max_iter,
    acceleration, relaxation, n_reuse, sub_idx, linear_solver,
):
    """Returns ``(x_star, aux)`` with ``x_star = F(x_star, *consts)``.

    ``aux = (n_iters, final_res, final_amp, (V, W))`` is forward-only
    bookkeeping
    from :func:`_fixed_point_while` (diagnostics and IQN secant matrices
    for cross-timestep reuse); it carries a zero derivative.

    ``x_star`` differentiates via the implicit function theorem:
        ``dx*/d(consts) = (I - dF/dx)^{-1} dF/d(consts)``
    evaluated at the fixed point.  ``x0``, ``accel_init`` and
    ``first_res`` receive a zero derivative (the fixed point is
    invariant under the initial guess, the acceleration state and the
    stopping bookkeeping in the converged limit).

    The rule is installed as a ``jax.custom_jvp`` (see
    ``_ift_solve_jvp``) rather than a ``custom_vjp``: JAX obtains
    reverse mode by transposing the (linear) tangent rule, and lineax's
    ``linear_solve`` is transposable, so one definition serves
    ``jax.jvp`` / ``jacfwd`` (the FMI ``FORWARD`` directional
    derivative), ``jax.grad`` / ``vjp`` / ``jacrev``, and higher
    order.  A ``custom_vjp`` cannot be forward-differentiated at all.

    The derivative is valid only at a converged fixed point, and it is
    taken at the state this function *returns*: on a criterion exit
    that is the iterate whose residual met ``threshold`` (see
    :func:`_fixed_point_while`), which is also the state the fori path
    returns.  The rule is unchanged by that choice — the IFT tangent
    is the derivative of ``F``'s fixed point, linearised at whatever
    ``x_star`` is handed to it — but the linearisation point moves by
    one pass, so on a non-linear ``F`` the reported gradient moves by
    ``O(residual)`` too.

    ``final_res`` bounds how far ``x_star`` is from a true fixed point
    (surfaced through ``GraphManager.coupling_diagnostics`` and, with
    ``CouplingGroup.strict_convergence``, a runtime error); the
    derivative is off by roughly ``residual * cond(I - dF/dx)``
    whenever it is non-zero, whether the loop stopped on its criterion
    or at ``max_iter``.  With ``diagnostics=True`` that error is
    bounded, per constant and relative to the tangent, by
    ``coupling_diagnostics()['gradient_relative_error_bound']`` (see
    :func:`_gradient_error_bound_at`).  A finite difference of this function's own
    output is therefore *not* the quantity the adjoint computes: it is
    the derivative of a truncated iterate, and the two agree only as
    ``final_res`` goes to zero.  Tighten the criterion that is live
    for the group's norm (``tolerance`` for ``"l2"``; ``atol`` /
    ``rtol`` for ``"mixed"`` and ``"interface"``, whose threshold is
    hard-coded to 1.0) if the gradient has to match the forward.

    ``acceleration`` / ``relaxation`` / ``n_reuse`` are static and
    control only the forward iterator.  The derivative *rule* is the same
    for all of them -- it is ``F``'s, linearised at the iterate the
    forward returns, whatever path reached it -- so the gradients agree
    to the extent the returned iterates do: exactly on a map affine in
    its state with additive constants, and to about the solve's
    tolerance times the map's curvature otherwise (each acceleration
    stops on its own iterate).  ``linear_solver`` selects the
    tangent/adjoint solver — see ``_ift_linear_solve``.
    """
    x_star, n_iters, final_res, final_amp, vw = _fixed_point_while(
        step_pure, x0, consts, accel_init, first_res, threshold, max_iter,
        acceleration, relaxation, n_reuse, sub_idx,
    )
    return x_star, (n_iters, final_res, final_amp, vw)



#: Largest flat coupling-group size at which a Krylov adjoint solve
#: that reports failure is silently re-solved with a dense LU.  Tied to
#: the ``restart = min(N, 50)`` clamp below: at or under this size the
#: Krylov space GMRES builds is already the whole space, so a direct
#: solve costs no more matvecs and is backward stable, while ``N**2``
#: floats of scratch is negligible.  Above it, the matrix-free path is
#: load-bearing and the failure is raised instead.
_DENSE_ADJOINT_FALLBACK_MAX_DOF = 50


def _dense_peak(n: int) -> str:
    """Peak working set of the dense adjoint path at ``n`` coupled DOF.

    ``_dense`` materialises the ``n x n`` Jacobian and the identity
    basis ``jacfwd`` builds it from, so the peak of the transposed
    solve ``jax.grad`` runs is ``2 * n**2 * itemsize``.  Measured
    against XLA's compiled-module memory analysis on the exact
    ``_dense`` body below, which reproduces the figure to within a few
    tens of kilobytes at every size from 64 to 3.6e5 DOF.  The tangent
    solve ``jax.jvp`` runs keeps ``I - J`` beside the basis and the
    Jacobian-vector products, ``3 * n**2 * itemsize`` for a dense
    coupling Jacobian; the message says so.  float32;
    ``jax_enable_x64`` doubles it.

    This exists so the message a user gets when the Krylov adjoint
    fails names the price of the alternative *at their own N*, rather
    than calling it "O(N^2)" and leaving them to find out on a grid.
    """
    total = 2 * n * n * 4
    for unit, scale in (("TiB", 2 ** 40), ("GiB", 2 ** 30), ("MiB", 2 ** 20)):
        if total >= scale:
            return f"~{total / scale:.1f} {unit}"
    return f"~{total} bytes"


#: Raised (through ``equinox.error_if``, at runtime inside jit) when a
#: Krylov adjoint solve fails on a group too large to re-solve densely.
#: It replaces lineax's own message, whose "increase ``restart``"
#: remedy does not address the mechanism — see ``_ift_linear_solve``.
#:
#: The remedies used to be listed with ``linear_solver='dense'`` first
#: and "O(N^2) memory" as the whole of its cost.  Every group that
#: reaches this message is already past
#: ``_DENSE_ADJOINT_FALLBACK_MAX_DOF``, and on a grid-coupled group the
#: dense path does not run at all — so the first remedy offered was the
#: one that could not work.  It now leads with the remedy that scales
#: and prices the dense one at the caller's own N.
_ADJOINT_SOLVE_FAILED_MSG = (
    "MADDENING: the coupling adjoint solve did not converge "
    "(linear_solver={solver!r}, {n} coupled DOF).  This is usually an "
    "ill-conditioned (I - dF/dx): cond(A) ~ 1/(1 - rho) in the group's "
    "slowest contraction rate, and float32 cannot resolve the solver's "
    "tolerance once eps*cond(A) exceeds it.  The remedy that scales is "
    "to make the group less stiff: stronger relaxation, a smaller "
    "timestep, or splitting the cycle.  You can also re-solve exactly "
    "by passing linear_solver='dense' to add_coupling_group() (or "
    "MADDENING_IFT_DENSE_SOLVE=1 to force it globally for triage), but "
    "price it first: that path materialises the full Jacobian, so at "
    "{n} coupled DOF it needs {dense_peak} of device memory in float32 "
    "(for jax.grad; half as much again for jax.jvp) and grows as N^2.  On a grid-coupled group it is not an escape "
    "hatch -- it does not run at all.  Raising GMRES's restart will "
    "NOT help: it is already min(N, 50)."
)


def _ift_linear_solve(matvec, rhs, linear_solver):
    """Solve ``A v = rhs`` for the matrix-free operator ``v -> matvec(v)``.

    ``A`` is ``I - dF/dx`` at the fixed point.  Wrapped in
    ``jax.lax.custom_linear_solve`` so JAX treats the result as linear
    in ``rhs``: forward mode re-solves with the tangent rhs and reverse
    mode calls ``transpose_solve`` with ``A^T`` — that is what lets JAX
    derive the reverse-mode rule (an adjoint solve) from the
    forward-mode rule automatically — while the *inside* of the solve
    is free to depend on the rhs non-linearly.  We use that freedom for
    the tolerance: lineax's criterion is elementwise (residual entry
    ``i`` under ``atol + rtol * |rhs_i|``), and cotangent / tangent
    vectors routinely carry exact zeros (a loss touching only some
    fields), whose entries would otherwise have to reach ``atol``
    absolute while float32 round-off from the large entries is
    ``~eps * max|rhs|``.  Once the Krylov space is exhausted (small
    systems use ``restart = n``) lineax then reports an "iterative
    breakdown" for a solve that is as exact as the dtype allows.  So
    ``atol`` is scaled to the largest rhs entry — the usual "relative
    to ||b||" Krylov criterion — and ``rtol`` is no tighter than ~100
    ulp of the dtype (1e-6 in float64, 1.2e-5 in float32).  **Nothing in
    the criterion is absolute**: the solve runs on ``b`` rescaled by an
    exact power of two and scales its answer back, and a zero ``b`` is
    answered with exact zeros.  Until 0.4.0's round-4 fix ``atol`` also
    carried an absolute ``1e-8``, below which the zero initial guess
    passed lineax's test before a single step: the tangent or adjoint of
    a group in small units, or of a loss near its minimum, came back
    exactly zero and "successful" (MADD-ANO-113).  A right-hand side
    with a NaN or infinite entry is answered with NaN in every entry,
    under every backend: the Krylov backends never see it (they are
    handed zeros), because its ``max|b|`` made the tolerance NaN or
    ``inf`` and lineax passed its test at the zero initial guess, so a
    non-finite tangent or cotangent came back as an exactly zero
    derivative, reported successful, where ``"dense"`` and
    ``solver="fori"`` read NaN (MADD-ANO-154).  NaN everywhere is also
    what an honest Krylov iteration produces -- its first basis vector
    is ``b / ||b||`` -- and ``"dense"`` agrees wherever its LU
    propagates the entry.  Memory is
    O(N) for the matrix-free backends; no Jacobian is ever
    materialised except under ``"dense"``.

    Backends, dispatched by ``linear_solver`` plus the
    ``MADDENING_IFT_DENSE_SOLVE`` env var (env var wins for triage):

    * ``"gmres"`` (default) — lineax GMRES.  Safe non-symmetric solver.
    * ``"dense"`` — materialise ``A`` with ``jacfwd`` and LU-solve.
      O(N^3) compute, and a peak working set of ``2 * N**2 *
      itemsize`` in reverse mode: the Jacobian plus the identity basis
      ``jacfwd`` builds it from (``3 * N**2 * itemsize`` in forward
      mode, where ``I - J`` is live beside them).  In float32 that is 0.48 GiB at N = 8,000, 2 GiB
      at N = 16,384, 32 GiB at N = 65,536 and, at N ≈ 3.6e5, a single
      523 GB allocation XLA refuses outright (``Out of memory
      allocating 523186046552 bytes``).  Triage fallback for a small
      group, promoted to a first-class config option; **not** a
      fallback for a grid-coupled one, where the numbers above are
      ordinary sizes.  See ``_dense_peak``.
    * ``"bicgstab"`` — lineax BiCGStab.  *Disabled at the CouplingGroup
      field level* in lineax 0.0.7: BiCGStab returns NaN when driving a
      ``FunctionLinearOperator`` — confirmed on a well-conditioned
      ``0.5*I`` test, so this is a lineax-side issue, not a property of
      MADDENING's coupling Jacobian.  The dispatch arm is left in place
      so a future lineax fix can re-enable it by widening the
      ``linear_solver`` Literal on CouplingGroup.

    **Why a failed Krylov solve re-solves directly at small N.**  A
    stiff coupling group makes ``A = I - dF/dx`` ill-conditioned:
    ``cond(A) ~ 1 / (1 - rho)`` in the group's slowest contraction
    rate, so ``rho = 0.999`` is already ``cond ~ 2e3``.  The relative
    accuracy *any* solver can reach on such an operator in float32 is
    ``~eps * cond(A)`` — ``2.4e-4`` at ``cond = 2e3`` — which is
    looser than the ``rtol`` asked for above (100 ulp, ``1.2e-5``).
    GMRES therefore exhausts its Krylov space without passing lineax's
    convergence test; the next restart cycle re-orthogonalises against
    a space that is already complete, Arnoldi returns a zero vector,
    and lineax reports ``RESULTS.breakdown``.  Lineax forgives a
    breakdown only when the solve *also* passes its tolerance test
    (``breakdown & not_converged``), which this one cannot, so the
    error escapes.  Measured on a 4-DOF two-node cycle with contraction
    modes ``(0.999, 0.2)``: GMRES stops three restart cycles in holding
    a solution whose relative error is ``1.0e-5`` — as accurate as
    float32 allows — and raises anyway.

    Two consequences.  This is *not* a Krylov breakdown in the textbook
    sense (a lucky zero in Arnoldi that a longer subspace would avoid),
    so lineax's "increase ``restart``" advice cannot help: ``restart``
    is already ``min(N, 50)``, i.e. the whole space at small ``N``.
    And it is a round-off lottery — whether the float32 iterate happens
    to land inside an unreachable tolerance depends on the cotangent —
    so the failure is non-monotone in stiffness (``rho = 0.998`` and
    ``0.999`` fail, ``0.9995`` passes) and a user cannot predict it.

    So the Krylov backends run with ``throw=False`` and this function
    acts on ``result`` itself:

    * ``N <= _DENSE_ADJOINT_FALLBACK_MAX_DOF``: re-solve densely under
      a ``lax.cond``.  At that size the dense LU is *cheaper* than the
      restart cycles GMRES already burned (``N`` matvecs against
      ``3 * N`` in the measured case), needs ``N**2`` floats of
      scratch, and is backward stable — so the fallback is a better
      answer, not a degraded one.  Only the failing branch runs; a
      successful GMRES solve is returned untouched, which is why
      ``"gmres"`` still means GMRES.
    * ``N`` above that: raise a MADDENING error naming the remedies
      that do work.  A dense fallback is not offered there because
      ``N**2`` is the compile-time memory the matrix-free path exists
      to avoid, and ``lax.cond`` reserves a branch's scratch whether or
      not the branch runs.

    A non-converged adjoint therefore stays loud, but the message names
    a remedy instead of one that cannot help.
    """
    force_dense = os.environ.get("MADDENING_IFT_DENSE_SOLVE") == "1"
    effective_solver = "dense" if force_dense else linear_solver
    if effective_solver not in ("gmres", "bicgstab", "dense"):
        raise ValueError(
            f"_ift_linear_solve: unsupported linear_solver="
            f"{linear_solver!r}; expected one of "
            f"'gmres', 'bicgstab', 'dense'."
        )
    n = rhs.shape[0]
    # A bfloat16 or float16 group's solve runs in float32 and its answer is
    # cast back (MADD-ANO-161): LAPACK has no 16-bit kernels, so the dense
    # path's LU and lineax's QR raised, and every jax.grad / jax.jvp through
    # a 16-bit group under solver="ift" failed.  The operator is still the
    # group's own -- ``I - dF/dx`` applied in its dtype, each argument
    # rounded to it -- so the dense path factors the 16-bit operator's exact
    # matrix and the Krylov path iterates on it; only the arithmetic of the
    # solve is widened, as the diagnostics' is (``_analysis_dtype``).
    work = _analysis_dtype(rhs.dtype)
    if work == rhs.dtype:
        rtol = max(1e-6, 100.0 * float(jnp.finfo(rhs.dtype).eps))
    else:
        # A 16-bit operator resolves its products to its own unit roundoff,
        # so a float32 criterion is unreachable (GMRES stalls at a residual
        # of a few 16-bit units and reports failure).  Four units of
        # roundoff of the dtype the answer is returned in: measured on
        # contractions of 20 to 400 entries, radius 0.6 to 0.95, GMRES then
        # converges, and its answer is within about twice the dense LU's
        # error -- both at the 16-bit output's own resolution.
        rtol = max(100.0 * float(jnp.finfo(work).eps),
                   2.0 * float(jnp.finfo(rhs.dtype).eps))

    def _dense(mv, b):
        A = jax.jacfwd(mv)(jnp.zeros_like(b))
        return jnp.linalg.solve(A, b)

    def _krylov(mv, b):
        # Lazy import — lineax is a base dependency (v0.4.0) but its
        # equinox/jaxtyping transitive deps cost an order of magnitude
        # more import time than ``import maddening`` does, so keep it
        # out of module load time.  Only callers who opt into
        # ``solver='ift'`` pay this import cost.
        import lineax as lx  # noqa: PLC0415  (lazy by design)

        # The solve is posed on ``b`` rescaled by one exact power of two
        # (largest entry in ``[0.5, 1)``), and its answer scaled back by
        # the reciprocal: ``A`` is linear, so that is the same solution,
        # and a power of two leaves every product of a vector in range as
        # it was.  The tolerance is then relative to the rescaled rhs and
        # nothing else.  It used to carry an absolute ``1e-8``: with the
        # zero initial guess lineax counted a solve "converged" before its
        # first step whenever ``max|b| <= ~1e-8`` and returned 0, so the
        # IFT tangent (forward mode) and adjoint (reverse mode) were
        # *exactly zero*, reported successful, wherever the tangent or the
        # cotangent was small -- a group in small units, or a loss close
        # to its minimum -- and above the dense fallback's size a
        # moderately small ``b`` raised the "ill-conditioned" error
        # instead.  A zero rhs is answered with exact zeros.
        #
        # A rhs with a NaN or infinite entry is answered with NaN in every
        # entry, and lineax is handed zeros in its place.  Posed on the rhs
        # itself, ``max|b|`` made ``atol`` NaN or ``inf``, the zero initial
        # guess passed lineax's test at once and the solve returned zeros
        # reported successful: a NaN tangent or cotangent came back as an
        # exactly zero derivative (MADD-ANO-154).  The zeros keep the
        # Krylov loop from iterating on NaN, and a non-finite rhs is
        # neither "failed" (no dense re-solve, no adjoint error above
        # the fallback's size) nor "zero".
        scale = pow2_frame(b)
        finite = jnp.all(jnp.isfinite(b))
        b_hat = jnp.where(finite, b * scale, jnp.zeros_like(b))
        # lineax declares `atol: float`, but it only ever compares against
        # it, and under `jit` this is a traced scalar that must stay one.
        atol = cast(float, rtol * jnp.max(jnp.abs(b_hat)))
        op = lx.FunctionLinearOperator(mv, jax.eval_shape(lambda: b))
        if effective_solver == "bicgstab":
            # BiCGStab has no ``restart`` parameter (it operates on a
            # fixed three-vector recurrence rather than building a
            # Krylov subspace).  ``max_steps`` only needs to bound the
            # outer iteration count.
            solver = lx.BiCGStab(
                rtol=rtol, atol=atol, max_steps=max(4 * n, 200),
            )
        else:
            # (I - dF/dx) is in general non-symmetric; GMRES is the
            # safe default.
            #
            # *** GMRES restart gotcha ***
            #
            # ``restart`` directly bounds the dim of the Krylov subspace
            # GMRES builds.  Lineax's default is 20.  For coupling
            # groups whose flat state is larger than 20 floats (any
            # chain of >=10 two-DOF nodes — common!), the
            # default-20 GMRES silently converges to a *low-rank
            # approximation* of the solve.  It looks fine
            # (converged=True, residual small in the projected
            # subspace) but the returned vector lies in a 20-D
            # subspace of an N-D problem, so the resulting derivative
            # is structurally wrong — *not* a near-correct answer
            # with extra noise, but a different gradient.
            #
            # We set restart = min(N, 50) so small problems stay cheap
            # while N>=50 problems still see a meaningful subspace,
            # and bump ``max_steps`` to give the algorithm headroom
            # for several restart cycles.  Do not regress this without
            # bumping the restart cap in lockstep.  The regression guard
            # is tests/core/test_coupling_ift_lineax.py::
            # test_gmres_call_uses_explicit_restart_at_least_minN50,
            # which checks that every lx.GMRES construction in the IFT
            # backward passes an explicit restart >= min(N, 50) and
            # max_steps >= 4 * restart -- the arguments, not a gradient.
            restart = min(n, 50)
            solver = lx.GMRES(
                rtol=rtol,
                atol=atol,
                restart=restart,
                max_steps=max(4 * restart, 100),
            )
        # ``throw=False`` so the failure is *this* module's to handle:
        # lineax's own message recommends raising ``restart``, which is
        # already the full space at small N and is not the mechanism
        # (see the docstring).
        sol = lx.linear_solve(op, b_hat, solver=solver, throw=False)
        is_zero = jnp.logical_not(jnp.any(b != 0))
        failed = jnp.logical_and(
            jnp.logical_and(jnp.logical_not(sol.result == lx.RESULTS.successful),
                            jnp.logical_not(is_zero)),
            finite)
        value = jnp.where(is_zero, jnp.zeros_like(b), sol.value / scale)
        value = jnp.where(finite, value, jnp.full_like(b, jnp.nan))
        if n <= _DENSE_ADJOINT_FALLBACK_MAX_DOF:
            return jax.lax.cond(
                failed, lambda bb: _dense(mv, bb), lambda _bb: value, b,
            )
        import equinox as eqx  # noqa: PLC0415  (lineax transitive dep)

        return eqx.error_if(
            value, failed,
            _ADJOINT_SOLVE_FAILED_MSG.format(
                solver=effective_solver, n=n, dense_peak=_dense_peak(n),
            ),
        )

    solve = _dense if effective_solver == "dense" else _krylov
    if work != rhs.dtype:
        narrow_solve = solve

        def solve(mv, b):
            def mv_work(v):
                return mv(v.astype(b.dtype)).astype(work)

            return narrow_solve(mv_work, b.astype(work)).astype(b.dtype)

    # ``transpose_solve`` receives ``vecmat = v -> A^T v`` and must
    # solve ``A^T x = b``; the same routine serves both.
    return jax.lax.custom_linear_solve(
        matvec, rhs, solve, transpose_solve=solve,
    )


def _ift_solve_jvp(
    step_pure, threshold, max_iter, acceleration, relaxation, n_reuse,
    sub_idx, linear_solver, primals, tangents,
):
    # Tangent rule of the implicit function theorem at ``x*``:
    #     (I - dF/dx) x_dot = dF/d(consts) . consts_dot
    # Linear in ``consts_dot`` (a jvp of F composed with a lineax solve),
    # so JAX can transpose it for reverse mode.  ``x0_dot``,
    # ``accel_dot`` and ``first_res_dot`` are ignored: the converged
    # fixed point does not depend on the initial guess, the
    # accelerator's seed state or the stopping bookkeeping, and
    # ``aux`` is forward-only bookkeeping with a zero tangent.
    x0, consts, accel_init, first_res = primals
    _x0_dot, consts_dot, _accel_dot, _first_res_dot = tangents
    x_star, aux = _ift_solve(
        step_pure, x0, consts, accel_init, first_res, threshold, max_iter,
        acceleration, relaxation, n_reuse, sub_idx, linear_solver,
    )
    _, rhs = jax.jvp(
        lambda cc: _F_dispatch(step_pure, x_star, cc), (consts,), (consts_dot,)
    )

    def _matvec(v):
        _, Jv = jax.jvp(
            lambda xx: _F_dispatch(step_pure, xx, consts), (x_star,), (v,)
        )
        return v - Jv

    x_dot = _ift_linear_solve(_matvec, rhs, linear_solver)
    aux_dot = jax.tree.map(_zero_tangent, aux)
    return (x_star, aux), (x_dot, aux_dot)


def _zero_tangent(x):
    """A zero tangent for ``x``: of its own dtype, or ``float0`` for an integer."""
    x = jnp.asarray(x)
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return jnp.zeros_like(x)
    return np.zeros(x.shape, dtype=jax.dtypes.float0)


# nondiff_argnums: 0=step_pure (callable), 5=threshold (static float),
#                  6=max_iter (static int), 7=acceleration (static str),
#                  8=relaxation (static float), 9=n_reuse (static int),
#                  10=sub_idx (static tuple | None), 11=linear_solver
#                  (static str).  4=first_res is a *traced* scalar (the
#                  residual of the pass before the loop), so it is a
#                  primal with an ignored tangent, not a static.
_ift_solve = jax.custom_jvp(
    _ift_solve_impl, nondiff_argnums=(0, 5, 6, 7, 8, 9, 10, 11)
)
_ift_solve.defjvp(_ift_solve_jvp)
