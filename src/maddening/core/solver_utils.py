"""Public solver utilities.

This module exposes the :func:`ift_linear_solve` primitive — a thin,
``@stability(EXPERIMENTAL)`` wrapper over :func:`lineax.linear_solve` that
any node solving a linear system in ``update()`` can use to obtain a
clean differentiable path.  The derivative is JAX's implicit one
(``jax.lax.custom_linear_solve``): the tangent and adjoint are solves of
their own, each posed on its right-hand side rescaled by an exact power of
two with a relative tolerance, so the answer does not depend on the units
of the solution or the loss.  No ``custom_vjp`` is installed.

.. note::
   **Experimental pilot (v0.3.1).**  ``ift_linear_solve`` is shipped early —
   ahead of its roadmapped 0.4/M3 home — as an ``@stability(EXPERIMENTAL)``
   pilot for downstream projects building on MADDENING.  The signature and
   behaviour are validated but not yet frozen.  It was briefly promoted to
   ``@stability(STABLE)`` alongside the ``AdaptiveNode`` framework and put
   back: the 0.4.0 API freeze decides the final level, informed by the open
   questions in ``docs/developer_guide/adaptive_node.md`` (whether the
   wrapper should expose ``restart`` / ``max_steps`` / ``stagnation_iters``
   and stop leaking ``lineax`` / ``equinox`` runtime error types — the
   remedy lineax prints for a stagnating solve is not reachable through
   this signature).  ``lineax`` is a base dependency as of v0.4.0, so
   ``pip install maddening`` is enough — no extra is required.

Background
----------

The wrapper exists because the existing in-tree pattern for
matrix-free linear solves (``graph_manager._ift_linear_solve``) is
module-private — it builds a ``lineax.FunctionLinearOperator`` from a
callable, calls ``lineax.GMRES`` with a carefully-chosen restart, and
returns the solution.  Any node author writing an adaptive PDE solver
needs the same idiom.  Exposing it as a public primitive avoids each
node author re-deriving the GMRES restart clamp from the
``_ift_linear_solve`` regression test.

The restart clamp is critical.  Lineax's default GMRES restart is 20.
For a coupling group whose flat state is larger than 20 floats (any
chain of ≥ 10 two-DOF nodes), the default-20 GMRES silently converges
to a low-rank approximation of the adjoint solve.  The returned ``u``
lies in a 20-D subspace of an N-D problem, so the resulting gradient
is structurally wrong — *not* a near-correct answer with extra noise,
but a different gradient.  The coupling layer's guard is
``tests/core/test_coupling_ift_lineax.py::
test_gmres_call_uses_explicit_restart_at_least_minN50``: it spies on
``lineax.GMRES`` while the IFT backward of a 60-float group is traced
and asserts every construction passes an explicit
``restart >= min(N, 50)`` and ``max_steps >= 4 * restart``.  It checks
the arguments, not a gradient.  This module applies the same
``restart = min(N, 50)`` clamp, pinned the same way by
``tests/adaptive/test_ift_linear_solve.py::test_gmres_restart_clamp``.

Evidence
--------

In ``tests/adaptive/test_ift_linear_solve.py``: the
``test_autodiff_correctness_*`` tests hold ``jax.grad`` through the
dense, GMRES and CG backends to a central difference within 1e-5
relative; ``tests/core/test_linear_solvers_in_any_units.py`` holds the
solution and the gradient to the same bits at every power-of-two scale; and
``test_bcoo_operator_compatibility`` shows a
``jax.experimental.sparse.BCOO`` matrix composes with the
``FunctionLinearOperator`` path (solution within 1e-6 of a dense
solve, gradient within 1e-4 of a central difference).
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import jax
import jax.numpy as jnp

from maddening.core._pow2_frame import pow2_frame
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability


_ALLOWED_SOLVERS = ("gmres", "cg", "dense")


@stability(StabilityLevel.EXPERIMENTAL)
def ift_linear_solve(
    operator_fn: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    solver: Literal["gmres", "cg", "dense"] = "gmres",
    preconditioner: Optional[Callable[[jax.Array], jax.Array]] = None,
    rtol: float = 1e-6,
    atol: Optional[float] = None,
) -> jax.Array:
    """Solve ``A x = b`` where ``A`` is given by a matrix-free callable.

    Parameters
    ----------
    operator_fn : callable
        ``v -> A @ v``.  Must be JAX-traceable.  Accepts and returns a
        rank-1 array of the same shape as ``rhs``.
    rhs : jax.Array
        Right-hand side vector ``b``.  Shape ``(N,)``.
    solver : {"gmres", "cg", "dense"}, default "gmres"
        Backend.  ``"gmres"`` is the safe default for general (possibly
        non-symmetric) ``A``.  ``"cg"`` asserts that ``A`` is
        symmetric positive semidefinite (passed to lineax as
        ``symmetric_tag`` + ``positive_semidefinite_tag``); the user
        is responsible for the assertion.  ``"dense"`` materialises
        ``A`` columnwise on ``jnp.eye(N)`` and falls back to
        ``jnp.linalg.solve``.  Appropriate for a **small** problem or
        for triage, and "small" is a number: both the basis and the
        matrix are live at once, so the peak working set is
        ``2 * N**2 * itemsize`` — 0.48 GiB at N = 8,000, 2 GiB at
        N = 16,384, 32 GiB at N = 65,536 in float32, and twice that
        under ``jax_enable_x64``.  There is no degradation at the top
        of that curve: at N ≈ 3.6e5 the matrix alone is a single
        523 GB allocation and XLA refuses it.  See
        ``ift_linear_solve``'s Raises section.
    preconditioner : callable or None, default None
        ``v -> M^{-1} @ v``.  If provided, applied during the linear
        solve.  The preconditioner's gradient is blocked via
        ``jax.lax.stop_gradient`` on its output: at convergence the
        solution and its sensitivity are independent of ``M``, so
        gradient flow through ``M`` is wasted compute and a potential
        source of noise.  Pinned by
        ``test_preconditioner_gradient_blocked`` in
        ``tests/adaptive/test_ift_linear_solve.py``.
    rtol : float
        Relative tolerance of the iterative solvers.  Ignored when
        ``solver="dense"``.
    atol : float or None
        Absolute floor of the iterative solvers' (entrywise) stopping test,
        in ``rhs``'s units: give a number only to assert a noise floor in
        ``rhs``'s own units.  ``None`` (the default) is ``rtol * max|rhs|``,
        so the test is relative and means the same in any units.  The solve
        itself runs on ``rhs`` rescaled by the power of two that brings
        ``max|rhs|`` into ``[0.5, 1)``, which is exact.  Before 0.4.0 the
        default was an absolute ``1e-8``: a right-hand side below about
        ``1e-2`` (``1e-8 / rtol``) was solved only to that floor, one below
        about ``1e-8`` passed the test with the zero initial guess and came
        back as zeros with no error, and at some scales an entry of
        ``rhs`` near zero made GMRES raise "iterative breakdown".  Ignored
        when ``solver="dense"``.

    Returns
    -------
    jax.Array
        Solution ``x`` of shape ``(N,)``.

    Notes
    -----
    No ``custom_vjp`` is installed.  The Krylov solve goes through
    ``jax.lax.custom_linear_solve``, so ``jax.grad`` and ``jax.jvp``
    through a call to this function return the standard sensitivity
    ``∂J/∂θ = -(A^{-T} ∂J/∂x)^T (∂A/∂θ x - ∂b/∂θ)``, with the tangent and
    adjoint solves posed -- rescaled and toleranced -- on their own
    right-hand sides (GMRES's adjoint is GMRES on ``A^T``; CG's is CG).
    Until 0.4.0 the adjoint went through lineax's own autodiff, which
    reused the forward solve's absolute ``atol``: a cotangent below it (a
    loss near its minimum, a solution in small units) gave a gradient of
    exactly zero with no error.

    A right-hand side, tangent or cotangent with a NaN or infinite entry
    gives NaN in every entry of the Krylov backends' answer, as an honest
    Krylov iteration does (its first basis vector is ``b / ||b||``);
    ``"dense"`` is non-finite wherever its LU propagates the entry.  Until
    0.4.0 the Krylov backends returned zeros with no error there
    (MADD-ANO-155).

    For ``solver="gmres"`` the internal restart is clamped to
    ``min(N, 50)`` to guard against the silent-low-rank-adjoint bug
    documented in ``graph_manager._ift_linear_solve``.

    Raises
    ------
    ValueError
        If ``solver`` is not one of ``"gmres"``, ``"cg"``, ``"dense"``,
        or if ``rhs`` is not rank-1.
    equinox._errors._EquinoxRuntimeError or jax.errors.JaxRuntimeError
        Propagated from ``lineax`` when an iterative solve does not
        converge or stagnates (the former eagerly, the latter under
        ``jit`` / ``scan``).  These are third-party types and the remedy
        lineax names (``stagnation_iters``, ``restart``) is not reachable
        through this signature; both are open questions for the 0.4.0 API
        freeze (see ``docs/developer_guide/adaptive_node.md``).  Until
        then, ``solver="dense"`` is the escape hatch **for a small
        system only** — it needs ``2 * N**2 * itemsize`` of device
        memory, so it is an escape hatch at N = 10^3 (8 MiB), a
        deliberate choice at N = 10^4 (0.75 GiB), and nothing at all on
        a grid-coupled problem, where N runs to 10^5-10^6 and the
        allocation is refused before the solve starts.  There the
        remedy is to condition the problem better, not to change
        backend.

    Examples
    --------
    Solve a symmetric positive-definite system via CG:

    >>> import jax.numpy as jnp
    >>> A = jnp.eye(8) * 2.0
    >>> b = jnp.ones(8)
    >>> x = ift_linear_solve(lambda v: A @ v, b, solver="cg")
    >>> bool(jnp.allclose(x, 0.5 * jnp.ones(8)))
    True
    """
    if solver not in _ALLOWED_SOLVERS:
        raise ValueError(
            f"ift_linear_solve: unsupported solver={solver!r}; "
            f"expected one of {_ALLOWED_SOLVERS!r}."
        )
    if rhs.ndim != 1:
        raise ValueError(
            f"ift_linear_solve: rhs must be rank-1; got shape {rhs.shape}."
        )

    n = int(rhs.shape[0])

    if solver == "dense":
        return _dense_solve(operator_fn, rhs, n)

    # Lazy for import time only (lineax is a base dependency as of
    # v0.4.0): it drags in equinox + jaxtyping, which cost an order of
    # magnitude more than ``import maddening`` itself.  Keep that off
    # the import path until a caller actually opts into a Krylov solve.
    import lineax as lx  # noqa: PLC0415

    # Build the preconditioner as a lineax PSD-tagged FunctionLinearOperator.
    # Lineax's CG and GMRES both accept the preconditioner via the
    # solver-options dict (see lineax._solver.misc.preconditioner_and_y0).
    # The preconditioner must be tagged positive_semidefinite for CG.
    # ``stop_gradient`` on the preconditioner output enforces the round-7
    # finding that M is gradient-irrelevant at convergence (and the solve
    # below is never differentiated, so nothing reaches M through it).
    options: dict = {}
    if preconditioner is not None:
        def _precond_blocked(v: jax.Array) -> jax.Array:
            return jax.lax.stop_gradient(preconditioner(v))
        options["preconditioner"] = lx.FunctionLinearOperator(
            _precond_blocked, jax.eval_shape(lambda: rhs),
            tags=(lx.positive_semidefinite_tag, lx.symmetric_tag),
        )
        import equinox as eqx  # noqa: PLC0415 — lineax transitive dep
        arrays, statics = eqx.partition(options, eqx.is_array)
        arrays = jax.tree.map(jax.lax.stop_gradient, arrays)
        options = eqx.combine(arrays, statics)

    def _framed_solve(mv, b):
        """One Krylov solve of ``mv(x) = b``, posed on ``b`` rescaled by an
        exact power of two (largest entry in ``[0.5, 1)``) and scaled back.

        The operator is linear, so it is the same solution and every Krylov
        product is the unscaled one times a power of two; an explicit
        ``atol`` -- an absolute floor in ``b``'s units -- is rescaled with
        it, and ``atol=None`` is relative to the largest entry of the
        rescaled ``b``: lineax's test is entrywise (``|r_i| <= atol + rtol
        |b_i|``), so an entry of ``b`` at zero needs a floor, and an
        absolute one below float rounding of the large entries made GMRES
        raise "iterative breakdown" on a solve as exact as the dtype allows.

        A ``b`` with a NaN or infinite entry is answered with NaN in every
        entry, and lineax is handed zeros in its place: posed on ``b``
        itself, ``max|b|`` made the relative ``atol`` NaN or ``inf`` and
        the zero initial guess passed lineax's test at once, so the
        solution, the tangent and the gradient came back as zeros with no
        error (CG did so under an explicit ``atol`` as well), and an
        explicit ``atol`` under GMRES raised lineax's "non-finite output"
        error instead (MADD-ANO-155).
        """
        p = pow2_frame(b)
        finite = jnp.all(jnp.isfinite(b))
        b_hat = jnp.where(finite, b * p, jnp.zeros_like(b))
        atol_hat: Any = rtol * jnp.max(jnp.abs(b_hat)) if atol is None else atol * p
        shape = jax.eval_shape(lambda: b)
        if solver == "cg":
            op = lx.FunctionLinearOperator(
                mv, shape, tags=(lx.positive_semidefinite_tag, lx.symmetric_tag),
            )
            # CG iteration budget: 4 * N covers any reasonable conditioning
            # at the floating-point regime MADDENING uses.
            solver_obj = lx.CG(rtol=rtol, atol=atol_hat, max_steps=max(4 * n, 200))
        else:  # gmres
            op = lx.FunctionLinearOperator(mv, shape)
            # *** Restart clamp ***
            #
            # Lineax's default GMRES restart is 20.  For N > 20 this
            # silently converges to a low-rank approximation; the resulting
            # gradient is structurally wrong.  Clamp to min(N, 50) and
            # bump max_steps for headroom.  See module docstring and
            # graph_manager._ift_linear_solve for the long-form
            # rationale and the coupling-layer regression guard at
            # tests/core/test_coupling_ift_lineax.py.
            restart = min(n, 50)
            solver_obj = lx.GMRES(
                rtol=rtol, atol=atol_hat, restart=restart,
                max_steps=max(4 * restart, 100),
            )
        x = lx.linear_solve(op, b_hat, solver=solver_obj, options=options).value / p
        return jnp.where(finite, x, jnp.full_like(x, jnp.nan))

    # ``custom_linear_solve`` rather than lineax's own autodiff: the tangent
    # and adjoint solves are then calls of ``_framed_solve`` on *their own*
    # right-hand sides, each rescaled and toleranced relative to itself.
    # Under lineax's autodiff the adjoint solve reused the forward solver's
    # absolute ``atol``, so a small cotangent -- a loss near its minimum, a
    # solution in small units -- came back as exact zeros, the defect
    # ``graph_manager._ift_linear_solve`` had (MADD-ANO-113).  The derivative
    # is the standard implicit one; GMRES's adjoint runs GMRES on the
    # transposed operator with the same preconditioner (a left
    # preconditioner changes the iteration, not the solution).
    if solver == "cg":
        return jax.lax.custom_linear_solve(operator_fn, rhs, _framed_solve, symmetric=True)
    return jax.lax.custom_linear_solve(operator_fn, rhs, _framed_solve,
                                       transpose_solve=_framed_solve)


def _dense_solve(
    operator_fn: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    n: int,
) -> jax.Array:
    """Materialise A column-wise and fall back to dense direct solve."""
    eye = jnp.eye(n, dtype=rhs.dtype)
    # vmap so JAX materialises one matvec per basis vector.
    A = jax.vmap(operator_fn, in_axes=1, out_axes=1)(eye)
    return jnp.linalg.solve(A, rhs)
