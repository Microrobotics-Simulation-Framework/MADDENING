"""Public solver utilities.

This module exposes the :func:`ift_linear_solve` primitive — a thin,
``@stability(EXPERIMENTAL)`` wrapper over :func:`lineax.linear_solve` that
any node solving a linear system in ``update()`` can use to obtain a
clean differentiable path.  Lineax's native autodiff propagates the
linear-solve adjoint correctly; this wrapper does **not** install a
MADDENING-level ``custom_vjp``.

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
relative, which is why no ``custom_vjp`` is installed; and
``test_bcoo_operator_compatibility`` shows a
``jax.experimental.sparse.BCOO`` matrix composes with the
``FunctionLinearOperator`` path (solution within 1e-6 of a dense
solve, gradient within 1e-4 of a central difference).
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp

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
    atol: float = 1e-8,
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
    rtol, atol : float
        Relative and absolute tolerances for the iterative solvers.
        Ignored when ``solver="dense"``.

    Returns
    -------
    jax.Array
        Solution ``x`` of shape ``(N,)``.

    Notes
    -----
    No ``custom_vjp`` is installed at the MADDENING level.  Lineax's
    native autodiff propagates the linear-solve adjoint correctly:
    ``jax.grad`` through a call to this function returns the standard
    sensitivity ``∂J/∂θ = -(A^{-T} ∂J/∂x)^T (∂A/∂θ x - ∂b/∂θ)``.

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
    # finding that M is gradient-irrelevant at convergence.
    options: dict = {}
    if preconditioner is not None:
        def _precond_blocked(v: jax.Array) -> jax.Array:
            return jax.lax.stop_gradient(preconditioner(v))
        options["preconditioner"] = lx.FunctionLinearOperator(
            _precond_blocked, jax.eval_shape(lambda: rhs),
            tags=(lx.positive_semidefinite_tag, lx.symmetric_tag),
        )

    if solver == "cg":
        op = lx.FunctionLinearOperator(
            operator_fn, jax.eval_shape(lambda: rhs),
            tags=(lx.positive_semidefinite_tag, lx.symmetric_tag),
        )
        # CG iteration budget: 4 * N covers any reasonable conditioning
        # at the floating-point regime MADDENING uses.
        solver_obj = lx.CG(rtol=rtol, atol=atol, max_steps=max(4 * n, 200))
    else:  # gmres
        op = lx.FunctionLinearOperator(
            operator_fn, jax.eval_shape(lambda: rhs),
        )
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
            rtol=rtol, atol=atol, restart=restart,
            max_steps=max(4 * restart, 100),
        )

    # Lineax refuses to autodiff through ``options``, so any tangent
    # captured by the preconditioner closure must be stopped before
    # the dict is passed in.  Use eqx.partition to split arrays from
    # the embedded Jaxpr / static structure, apply ``stop_gradient``
    # to the array half, and recombine.  (Round-7 finding: M is
    # gradient-irrelevant at convergence, so this is correct.)
    if preconditioner is not None:
        import equinox as eqx  # noqa: PLC0415 — lineax transitive dep
        arrays, statics = eqx.partition(options, eqx.is_array)
        arrays = jax.tree.map(jax.lax.stop_gradient, arrays)
        options = eqx.combine(arrays, statics)
    result = lx.linear_solve(op, rhs, solver=solver_obj, options=options)
    return result.value


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
