"""Sharded sparse iterative solvers (added in v0.3.0).

Builds on the IFT branch's matrix-free lineax integration
(see :func:`maddening.core.coupling._ift._ift_linear_solve`).
This module extends the same pattern to **distributed** matvec
operators: the user supplies a callable ``matvec(x) → A · x`` whose
internal implementation uses ``shard_map`` to run across multiple
devices, and we wrap it in a small public API for CG / GMRES.

Why this module exists:

* FVM PISO pressure correction in MIME v0.5.0 (against MADDENING v0.4.0
  — the hard downstream gate) needs scalable Krylov solves on
  distributed operators.
* The FMI 3.0 substrate (:mod:`maddening.fmi`) commits us to exposing exact directional
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
* By default we route through ``lineax.FunctionLinearOperator`` +
  ``lineax.GMRES`` / ``lineax.CG`` to reuse the IFT branch's
  battle-tested adjoint path.  ``lineax`` is a base dependency as of
  v0.4.0, so this path is always available.
* When ``lineax`` cannot consume a sharded matvec cleanly, or when a
  preconditioner is supplied (lineax's CG / GMRES take none), the
  hand-rolled ``lax.fori_loop`` paths in ``_cg_loop`` and
  ``_gmres_loop`` provide a fallback that depends only on stock JAX.
  Select it explicitly with ``backend="loop"``.
* Every route reports the residual of the system it was given:
  ``residual_norm`` is ``||b - A x||`` of the returned value from one
  product, and on the loop backends ``converged`` is that norm against
  ``max(atol, rtol * ||b||)``.  Until 0.4.0 the CG loop reported the
  residual it updates by recurrence, which float32 let read under a
  tolerance the true residual was far above (MADD-ANO-263).
* lineax's CG returns NaN in float32 above about 1e5 unknowns
  (MADD-ANO-264).  The default route of :func:`sharded_cg` then solves
  the system on the loop backend instead; ``backend="lineax"`` by name
  still returns the NaN.  Under a trace the choice is a ``jnp.where`` on
  a loop whose iteration cap is zero where lineax answered, so the
  default route can be traced wherever the loop can (inside a
  ``shard_map`` body too) at two operator products per call.  See
  ``backend`` and ``Returns`` of :func:`sharded_cg`.
* An operator that returns a NaN or an infinity is answered as a
  non-finite right-hand side is, on every route: NaN in every entry of
  ``value``, a NaN ``residual_norm``, ``converged=False``
  (MADD-ANO-269; ``_framed``).  On no route is ``converged`` True beside
  a ``residual_norm`` that is not finite.
* The same answer for a solution too large for the dtype (the answer is
  tested after it is scaled back, MADD-ANO-272) and for a tolerance that
  is negative or NaN and arrives as a traced value (MADD-ANO-271); given
  as a concrete number such a tolerance is refused.  A result is every
  entry of ``value`` finite beside a finite ``residual_norm``, or NaN in
  both (one exception under ``differentiable=True`` at the top of the
  dtype's range: MADD-ANO-275, ``Returns`` of :func:`sharded_cg`).
* lineax's verdict is ``converged`` only where the floor of its entrywise
  test is positive.  With a floor of zero (``rtol=0``, an explicit
  ``atol=0.0``) its quotient is ``0 / 0`` at every entry that is exactly
  zero and it reports success; ``converged`` is then the loop backends'
  statement on the value lineax returned (``_lineax_solve``,
  MADD-ANO-274).

Differentiability: pass ``differentiable=True``.  The solve then goes
through ``lax.custom_linear_solve``: forward and reverse mode are linear
solves with the same backend and preconditioner, started from zero and
purely relative to their own right-hand side, and the derivative with
respect to what the matvec closes over flows through
(``_differentiable_solve``).  A call WITHOUT it refuses to be
differentiated: ``jax.jvp``, ``jax.grad``, ``jax.jacfwd`` and
``jax.jacrev`` through it raise a ``ValueError`` that says so
(``_refusing_a_derivative``).  Differentiating the iteration is not
differentiating the solve, and until 0.4.0 it gave a silently wrong
derivative whenever the tangent or cotangent was not of the right-hand
side's size (MADD-ANO-266).

Preconditioning: Jacobi only.  The interface
accepts an optional ``preconditioner`` callable for forward compat
with AMG / block-Jacobi in v0.4.0+.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
# ``jax.core`` is not re-exported by ``jax/__init__.py``; imported by name
# so ``jax.core.Tracer`` resolves for a type checker (as in graph_manager).
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from maddening.core._pow2_frame import pow2_frame
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability


# ---------------------------------------------------------------------------
# Result struct
# ---------------------------------------------------------------------------


@stability(StabilityLevel.EVOLVING)
@dataclass(frozen=True)
class SharedSolveResult:
    """Return type for :func:`sharded_cg` and :func:`sharded_gmres`.

    Attributes
    ----------
    value : jax.Array
        The approximate solution ``x`` to ``A x = b``.  NaN in every
        entry where there is no answer (see ``residual_norm``), and
        every entry finite otherwise.
    converged : jax.Array
        Boolean scalar.  On the loop backends, on :func:`sharded_cg`'s
        default route where it falls back to the loop, and under
        ``differentiable=True``: ``residual_norm <= max(atol, rtol * ||b||)``
        for the returned ``value``, in the working dtype, with no
        allowance for rounding.  On the lineax backend: lineax's own
        verdict (``Returns`` of :func:`sharded_cg`), except where the
        floor of lineax's entrywise test is zero (``rtol=0``, an
        explicit ``atol=0.0``, a tolerance that underflows): there it
        is the statement above too, which for a tolerance of zero is
        ``residual_norm == 0``.  False where ``value`` is NaN, and never
        True beside a ``residual_norm`` that is not finite.  It is a
        statement about ``value``: under ``differentiable=True`` it
        does not cover the tangent and cotangent solves (MADD-ANO-270).
        Until 0.4.0 the CG loop answered from the
        residual it updates by recurrence, and the GMRES loop behind a
        preconditioner from the preconditioned residual: ``True`` on
        float32 solves whose ``||b - A x||`` was far above the tolerance
        (MADD-ANO-263).
    iters : jax.Array
        Integer scalar — number of iterations used (-1 under
        ``differentiable=True``).  The unit depends on the route.
        :func:`sharded_cg`: steps, one operator product each.
        :func:`sharded_gmres` on the loop backend: steps, in whole
        restart cycles (a multiple of ``restart``).
        :func:`sharded_gmres` on the lineax backend, which is its
        default: restart CYCLES, of up to ``restart`` products each
        (lineax's ``num_steps``; MADD-ANO-273, see ``max_iters``
        there).
    residual_norm : jax.Array
        ``||b - A x||_2`` of the returned ``value``: one product in the
        working dtype, with the operator and the right-hand side the
        caller gave, on every route.  (The product is taken on the
        right-hand side rescaled into ``[0.5, 1)`` and the norm scaled
        back.)  NaN, with NaN in every entry of
        ``value`` and ``converged=False``, where there is no answer: a
        ``b`` with a NaN or infinite entry (MADD-ANO-155); an
        operator that returned a NaN or an infinity, or a value so far
        off that the norm overflows (MADD-ANO-269); ``backend="lineax"``
        CG in float32 at scale (MADD-ANO-264); a solution too large
        for the dtype, or a residual whose norm is (both are tested as
        they are returned, after the scaling back; MADD-ANO-272); an
        ``rtol`` or ``atol`` that is negative or NaN and arrived as a
        traced value (MADD-ANO-271).  Finite otherwise, with one
        exception under ``differentiable=True`` (MADD-ANO-275,
        ``Returns`` of :func:`sharded_cg`).
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
    rtol: Any,
    atol: Any = None,
) -> None:
    # ``atol`` is an absolute floor in ``b``'s units.  A negative one was
    # taken as it came: the loop backends' ``max(atol, rtol * ||b||)`` hid
    # it, and lineax's entrywise ``|r_i| <= atol + rtol * |b_i|`` did not
    # (its verdict then read True at 2e5 times the tolerance asked for).
    # A traced value cannot be read here: ``_tolerances_are_numbers``
    # tests it in the program, and ``_framed`` answers it.
    if (atol is not None and not isinstance(atol, jax.core.Tracer)
            and not bool(np.all(np.asarray(atol) >= 0))):
        raise ValueError(
            f"{name}: atol must be None (a purely relative test) or a number "
            f">= 0 (an absolute floor in b's units), got {atol!r}."
        )
    # ``rtol`` was not read at all until 0.4.0: a NaN stopped lineax on its
    # zero start with a verdict of success (zeros, ``converged=True``), and
    # a negative one was refused by lineax on its own routes only.  ``not
    # (rtol >= 0)`` is True for a NaN.
    if (not isinstance(rtol, jax.core.Tracer)
            and not bool(np.all(np.asarray(rtol) >= 0))):
        raise ValueError(
            f"{name}: rtol must be a number >= 0 (a fraction of ||b||), "
            f"got {rtol!r}."
        )
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


def _tolerances_are_numbers(rtol: Any, atol: Any) -> Any:
    """Whether ``rtol`` and ``atol`` are numbers >= 0, for the tolerances
    :func:`_validate_solver_inputs` could not read: ``True`` (the Python
    value) where both came as concrete values, which were refused or are
    valid, and otherwise a traced boolean, False for a traced tolerance
    that is negative or NaN (``not (t >= 0)``).

    A traced tolerance is one that is an argument of a jitted or vmapped
    function.  It cannot raise, so :func:`_framed` answers it as it
    answers a right-hand side that is not finite: NaN in every entry,
    a NaN ``residual_norm``, ``converged=False``.  Until 0.4.0's fix it was
    passed on, and the lineax routes read ``converged=True`` on a value
    72 % off (a traced ``atol=-1``) or on zeros (a traced NaN)
    (MADD-ANO-271).  Infinity is a number >= 0 and passes.
    """
    valid: Any = True
    for tolerance in (rtol, atol):
        if isinstance(tolerance, jax.core.Tracer):
            valid = jnp.logical_and(valid, jnp.all(tolerance >= 0))
    return valid


def _materialise_b(
    b: jax.Array,
    *,
    mesh: Optional[Mesh],
    in_specs: Optional[PartitionSpec],
) -> jax.Array:
    """Place b on the right sharding so the solver inherits the layout."""
    if mesh is None:
        return jnp.asarray(b)
    # `_validate_sharding_args` refuses a mesh without in_specs, and every
    # caller runs it first.
    assert in_specs is not None
    sharding = NamedSharding(mesh, in_specs)
    return jax.device_put(jnp.asarray(b), sharding)




def _framed(solve, b, x0, atol, tolerances_are_numbers: Any = True):
    """Run ``solve(b_hat, x0_hat, atol_hat)`` on ``b`` rescaled by an exact
    power of two and scale its result back.

    ``p`` brings ``max|b|`` into ``[0.5, 1)`` (:func:`~maddening.core._pow2_frame.pow2_frame`);
    the operator is linear, so the solution of ``A x = b p`` is ``x p``
    exactly, and every Krylov product on the rescaled vectors is the
    unscaled one times a power of two -- bit for bit where the unscaled
    arithmetic stayed in the normal range, and in range where it did not
    (the residual's squared norm of a ``b`` near ``1e-20`` flushed to zero
    in float32).  An ``atol`` the caller gave is an absolute floor in
    ``b``'s units and is rescaled with ``b``; ``None`` stays ``None``, and
    each backend reads it as a purely relative test (the loop backends'
    ``||r|| <= rtol * ||b||``; for lineax's entrywise test the floor
    ``rtol * max|b|``, the usual "relative to ``||b||``" Krylov criterion,
    which also lets an entry of ``b`` at or near zero converge on rounding
    rather than on an absolute number below it).  Until 0.4.0 the default
    was an absolute ``1e-8``: a right-hand side below about ``1e-2``
    (``1e-8 / rtol``) was solved only to that absolute floor, a relative
    accuracy of ``1e-8 / ||b||`` instead of ``rtol``, and one below about
    ``1e-8`` passed the test with the zero initial guess before a single
    step -- measured on a 12-unknown SPD system in float32, relative errors
    of 5e-3 at ``||b|| ~ 1e-6`` and 0.4-1.0 at ``1e-9``, every backend
    reporting ``converged=True``: the defect ``coupling._ift._ift_linear_solve``
    had (MADD-ANO-113).  ``p`` is ``stop_gradient``-ed: it is piecewise
    constant in ``b``.

    A ``b`` with a NaN or infinite entry is answered with NaN in every
    entry of ``value``, a NaN ``residual_norm`` and ``converged=False``,
    and the backend is handed zeros (and a zero start) in its place, so it
    does not iterate on NaN.  Posed on ``b`` itself, ``max|b|`` made
    lineax's relative ``atol`` NaN or ``inf`` and the zero initial guess
    passed its test at once, and the loop CG's tolerance went ``inf`` with
    ``||b||``: CG (both backends) and lineax GMRES returned zeros, mostly
    reported ``converged=True`` (MADD-ANO-155).

    The same answer where the backend's ``residual_norm`` is not finite:
    an operator that returned a NaN or an infinity (or a value so far off
    that the norm of its residual overflows).  This is the one exit of
    every route of both solvers, so the rule holds on all of them, and
    ``converged`` is never True beside a norm that is not finite.  Until
    0.4.0 lineax's CG answered such an operator with its zero start,
    ``converged=True`` and ``iters=0`` (the start is finite, so lineax
    reports success at once), on ``backend="lineax"`` and on the default
    call; the CG loop returned zeros with ``converged=False``
    (MADD-ANO-269).

    **The answer is tested after it is scaled back.**  The two fields are
    tested as they are returned, ``value / p`` and ``residual_norm / p``,
    not as the backend left them in the frame: a solution too large for
    the dtype is finite in the frame and overflows on being scaled back,
    and until 0.4.0's fix the result then held infinities beside a finite
    ``residual_norm`` and the frame's ``converged`` (MADD-ANO-272).  The
    same for a norm that is finite in the frame and overflows on being
    scaled back (a ``b`` whose own norm overflows, left unsolved): it was
    returned as ``inf``, beside ``converged=True`` where the tolerance was
    infinite too.  So a result is one of two things on every route: every
    entry of ``value`` finite beside a finite ``residual_norm``, or NaN in
    both with ``converged=False``.

    **A traced tolerance that is not a number >= 0** gets that second
    answer too: ``tolerances_are_numbers`` is the verdict of
    :func:`_tolerances_are_numbers` on the ``rtol`` and ``atol`` of the
    CALL (not of this solve: the tangent and cotangent solves of
    ``differentiable=True`` are purely relative, and are refused with the
    call they belong to).  ``True``, the Python value, where both were
    concrete: nothing is added to the program then.
    """
    p = jax.lax.stop_gradient(pow2_frame(b))
    finite = jnp.all(jnp.isfinite(b))
    res = solve(jnp.where(finite, b * p, jnp.zeros_like(b)),
                jnp.where(finite, x0 * p, jnp.zeros_like(x0)),
                None if atol is None else atol * p)
    value = res.value / p
    residual_norm = res.residual_norm / p
    answered = jnp.logical_and(
        finite,
        jnp.logical_and(jnp.all(jnp.isfinite(value)), jnp.isfinite(residual_norm)))
    if tolerances_are_numbers is not True:
        answered = jnp.logical_and(answered, tolerances_are_numbers)
    return SharedSolveResult(
        value=jnp.where(answered, value, jnp.full_like(b, jnp.nan)),
        converged=jnp.logical_and(res.converged, answered), iters=res.iters,
        residual_norm=jnp.where(answered, residual_norm, jnp.nan),
    )


# ---------------------------------------------------------------------------
# Hand-rolled fallbacks (no lineax dependency)
# ---------------------------------------------------------------------------


def _residual_of_the_system(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    value: jax.Array,
    *,
    rtol: float,
    atol: Any,  # a float, or a traced scalar in the solve's frame
) -> tuple[jax.Array, jax.Array]:
    """``(residual_norm, converged)`` of *value* as a solution of ``A x = b``.

    ``residual_norm`` is ``||b - A value||_2`` from one product in the
    working dtype, with the operator and the right-hand side the caller
    gave (not a preconditioned pair).  ``converged`` is that norm against
    ``max(atol, rtol * ||b||)`` (``rtol * ||b||`` when ``atol`` is
    ``None``), the tolerance the CG loop stops on.  No allowance for
    rounding, and a NaN norm reads ``False``.

    A statement about the system, whichever iteration produced *value*:
    the loop backends and the default CG route's fallback all report
    through this function, and ``differentiable=True`` makes the same
    statement (``_differentiable_solve``).  Until 0.4.0 the CG loop
    reported the residual it updates by recurrence instead, which in
    float32 read under the tolerance on solves that had not met it
    (MADD-ANO-263).
    """
    res_norm = jnp.linalg.norm(b - matvec(value))
    b_norm = jnp.linalg.norm(b)
    tol = rtol * b_norm if atol is None else jnp.maximum(atol, rtol * b_norm)
    return res_norm, res_norm <= tol


def _cg_iterations(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    x0: jax.Array,
    rtol: float,
    atol: Any,  # a float, or a traced scalar in the solve's frame
    max_iters: Any,  # an int, or a traced count (the default route's cap)
    preconditioner: Optional[Callable[[jax.Array], jax.Array]],
) -> tuple[jax.Array, jax.Array]:
    """The preconditioned conjugate-gradient iteration on a global sharded
    matvec: ``(x, iters)``.

    Inner products use ``jnp.vdot``, which is correctly all-reduced
    when its inputs are sharded across a mesh axis (the partial sums
    on each shard get summed automatically by XLA).

    The residual the loop STOPS on is the recursively updated ``r``
    (``r <- r - alpha * A p``), never ``b - A x``: the two drift apart by
    about ``eps * kappa``, so in float32 the loop can stop on its test
    while the residual of the system is still above the tolerance.  That
    is why nothing is reported from ``r``: see :func:`_cg_loop`.
    """
    M = preconditioner if preconditioner is not None else (lambda r: r)
    b_norm = jnp.linalg.norm(b)
    # Guard against a zero RHS — solution is trivially x0.
    tol2 = (rtol * b_norm if atol is None else jnp.maximum(atol, rtol * b_norm)) ** 2

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
    x, _, _, _, _, iters = lax.while_loop(cond, body, init)
    return x, iters


def _cg_loop(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    x0: jax.Array,
    rtol: float,
    atol: Any,  # a float, or a traced scalar in the solve's frame
    max_iters: Any,  # an int, or a traced count (the default route's cap)
    preconditioner: Optional[Callable[[jax.Array], jax.Array]],
) -> SharedSolveResult:
    """The CG loop backend: :func:`_cg_iterations`, and the report of
    :func:`_residual_of_the_system` on the value it returns.

    ``value`` and ``iters`` are the iteration's, which stops where it
    always did (on its recursively updated residual, or on ``max_iters``).
    ``residual_norm`` and ``converged`` come from one extra product on
    that value, so a float32 solve the iteration stopped early on reads
    ``converged=False`` with the residual of the system (until 0.4.0:
    ``True``, with the recursive residual; MADD-ANO-263, pinned by
    ``tests/cloud/multigpu/test_what_sharded_cg_reports_on_each_route.py``).
    """
    x, iters = _cg_iterations(
        matvec, b, x0=x0, rtol=rtol, atol=atol, max_iters=max_iters,
        preconditioner=preconditioner,
    )
    res_norm, converged = _residual_of_the_system(
        matvec, b, x, rtol=rtol, atol=atol,
    )
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
    atol: Any,  # a float, or a traced scalar in the solve's frame
    restart: int,
    max_iters: int,
    preconditioner: Optional[Callable[[jax.Array], jax.Array]],
    a_zero_rhs_needs_no_cycle: bool = False,
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

    With a preconditioner the cycles test ``||M (b - A x)||`` against
    ``max(atol, rtol * ||M b||)``.  What is reported is not that: it is
    :func:`_residual_of_the_system` of the returned value, as on every
    other route (until 0.4.0 the preconditioned pair was reported,
    MADD-ANO-263).  When the cycles stop is not changed.
    """
    if preconditioner is not None:
        original_matvec = matvec
        M = preconditioner

        # Bound below rather than defined under the name: a `def` and an
        # assignment in the other branch are two declarations of one name.
        def _preconditioned_matvec(x):
            return M(original_matvec(x))

        matvec_pc = _preconditioned_matvec

        # Also precondition b on the left.
        b_eff = M(b)
    else:
        matvec_pc = matvec
        b_eff = b

    b_norm = jnp.linalg.norm(b_eff)
    tol = rtol * b_norm if atol is None else jnp.maximum(atol, rtol * b_norm)
    # A Krylov space of an ``n``-vector has at most ``n`` dimensions: past
    # that the Arnoldi step normalises a zero (or rounding) vector, and the
    # cycle's least squares returned NaN with ``converged=False`` whenever
    # ``restart`` exceeded ``n`` -- the default 50 against any system of
    # fewer than 50 unknowns.  The lineax path clamps the same way.
    restart = max(1, min(int(restart), int(b.shape[0])))
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

    # The first cycle always runs (an ordinary call counts it even where
    # ``x0`` already meets the tolerance), except on request for a
    # right-hand side that is exactly zero: ``differentiable=True`` solves
    # one from a zero start on every call that carries an ``x0`` or an
    # ``atol`` (``_differentiable_solve``), and a cycle on zeros is
    # ``restart`` products that return zeros.
    b_norm_is_zero = b_norm == 0
    init = (x0, jnp.linalg.norm(b_eff - matvec_pc(x0)),
            b_norm_is_zero if a_zero_rhs_needs_no_cycle else jnp.array(False),
            jnp.int32(0))
    x, res, converged, cycles_used = lax.fori_loop(
        0, n_cycles, cycle_body, init,
    )
    if preconditioner is not None:
        # The cycles test ``||M (b - A x)||`` against ``||M b||``: neither is
        # in ``b``'s units.  What is REPORTED is the residual of the system
        # the caller gave, as on every other route (one more product).
        res, converged = _residual_of_the_system(
            matvec, b, x, rtol=rtol, atol=atol,
        )
    return SharedSolveResult(
        value=x, converged=converged,
        iters=cycles_used * restart, residual_norm=res,
    )


# ---------------------------------------------------------------------------
# Lineax-backed path
# ---------------------------------------------------------------------------


def _lineax_solve(
    solver_kind: str,
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    rtol: float,
    atol: Any,  # a float, or a traced scalar in the solve's frame
    restart: int,
    max_iters: int,
) -> SharedSolveResult:
    """Run the lineax-backed path.

    ``lineax`` is a base dependency as of v0.4.0, so this no longer
    has an "unavailable" outcome; the import stays lazy purely to keep
    equinox + jaxtyping off ``import maddening``'s critical path.
    Errors propagate — the caller sees the real failure.

    **Where lineax's verdict is not used.**  lineax tests the quotients
    ``|r_i| / (floor + rtol |b_i|)`` and ``|dx_i| / (floor + rtol |x_i|)``
    against one, entry by entry, and takes their maximum.  With a floor of
    zero the quotient is ``0 / 0 = NaN`` at every entry where the residual
    and the right-hand side (or the update and the iterate) are both
    exactly zero; the maximum is then NaN, ``NaN > 1`` is False, and lineax
    stops and reports success, whatever the other entries hold.  Two calls
    get there: ``rtol=0`` (the floor of a call without ``atol`` is
    ``rtol * max|b|``), measured as ``converged=True`` at a residual of
    5e-8 to 5e-5 of ``||b||`` on GMRES and of 0.4 on CG for a ``b`` that is
    one unit vector; and an explicit ``atol=0.0`` beside any ``rtol``, for
    a ``b`` with entries that are exactly zero (a local source behind a
    stencil: every entry the Krylov space has not reached yet), measured
    as ``converged=True`` after ONE step at a residual of 0.4 ``||b||``.
    A floor that underflows does the same (XLA's CPU backend flushes a
    subnormal to zero): ``rtol=1e-44`` in float32.  MADD-ANO-274; the
    ``atol=0.0`` form was in 0.3.0 and 0.3.1.

    So lineax's verdict is ``converged`` only where the floor it was
    handed is at least the smallest normal number of the dtype: every
    quotient then has a positive denominator.  Under that, ``converged``
    is the statement of the loop backends on the value lineax returned,
    ``residual_norm <= max(atol, rtol * ||b||)``
    (:func:`_residual_of_the_system`, from the product the report takes
    anyway), which for a tolerance of exactly zero is
    ``residual_norm == 0``.  ``value`` and ``iters`` are lineax's on
    both sides: when it stops is not changed.  (An ``rtol`` of infinity
    on a zero ``b`` makes the floor NaN, which is not "under": lineax's
    verdict, as before.)
    """
    import lineax as lx  # noqa: PLC0415  (lazy: import-time cost only)

    atol_given = atol
    if atol is None:
        # Relative to the largest entry of ``b``: lineax's test is entrywise
        # (``|r_i| <= atol + rtol |b_i|``), so an entry of ``b`` at zero needs
        # a floor, and one below float rounding of the large entries made it
        # raise "iterative breakdown" on a solve as exact as the dtype allows.
        # A tolerance carries no derivative (lineax refuses a tangent on it).
        atol = jax.lax.stop_gradient(rtol * jnp.max(jnp.abs(b)))
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
        # See the restart-gotcha comment in coupling._ift._ift_linear_solve.
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
    res_norm, within_the_norm = _residual_of_the_system(
        matvec, b, result.value, rtol=rtol, atol=atol_given,
    )
    # lineax's RESULTS object doesn't always populate iters; treat as -1.
    stats = result.stats or {}
    n_iters = jnp.int32(stats.get("num_steps", -1))
    verdict = jnp.asarray(result.result == lx.RESULTS.successful)
    the_floor_is_zero = atol < jnp.finfo(b.dtype).tiny
    return SharedSolveResult(
        value=result.value,
        converged=jnp.where(the_floor_is_zero, within_the_norm, verdict),
        iters=n_iters,
        residual_norm=res_norm,
    )


# ---------------------------------------------------------------------------
# The default CG route: lineax, and the loop where lineax breaks down
# ---------------------------------------------------------------------------


def _lineax_cg_or_the_loop_where_it_breaks_down(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    *,
    x0: jax.Array,
    rtol: float,
    atol: Any,  # a float, or a traced scalar in the solve's frame
    max_iters: int,
) -> SharedSolveResult:
    """``sharded_cg``'s default route: lineax's CG, and where the value it
    returns is not all finite, the same system on the loop backend.

    lineax's CG sets its step to NaN unless
    ``|p.A p| > 200 * eps * n * |r.r|`` (MADD-ANO-264): in float32 no
    operator of order-one scale passes above about 1e5 unknowns, and every
    entry of its answer is then NaN.  The right-hand side is finite here
    (``_framed`` hands a non-finite ``b`` on as zeros), so a non-finite
    value is lineax's breakdown or an operator neither route can solve.

    Where lineax's value is finite its result is returned as it is, all
    four fields: the lineax solve is the same program as
    ``backend="lineax"``, and what follows only selects.

    Otherwise the loop solves the system from ``x0`` and the result is
    :func:`_cg_loop`'s: its ``value``, its ``iters``, and
    ``residual_norm`` / ``converged`` from
    :func:`_residual_of_the_system` on that value (one extra product, NOT
    the loop's recursively updated residual, which reads under the
    tolerance on solves that have not met it, MADD-ANO-263).

    **How the choice is made.**  Called eagerly the predicate ("is
    lineax's value all finite?") is a number and the branch is taken in
    Python: where lineax answered, nothing else runs.  Under a trace
    (``jit``, ``vmap``, a ``shard_map`` body) the predicate is traced, and
    the loop is run with an iteration cap of ``max_iters`` where lineax
    broke down and of ZERO where it answered; each field is then selected
    with ``jnp.where`` on the predicate.  A loop capped at zero does no
    iteration, but its set-up and its report are still executed: a
    default call that does not fall back costs TWO operator products more
    than ``backend="lineax"`` (``b - A x0`` before the loop, and the
    product of the report), counted by
    ``tests/cloud/multigpu/test_sharded_cg_default_route_in_every_context.py``.

    Only a ``lax.while_loop`` and ``jnp.where`` are used, which is what
    ``backend="loop"`` uses, so the default route can be traced wherever
    the loop backend can: inside a ``shard_map`` body too (one solve per
    device).  A ``lax.cond`` between the two solvers cannot be traced
    there (lineax's result and the loop's differ in the mesh axes they
    are typed as varying over), and neither can a
    ``lax.custom_linear_solve`` around the loop; the first form of this
    fallback used both and raised on every default call in such a body.
    Nothing here needs to be differentiable: a call without
    ``differentiable=True`` refuses to be differentiated
    (:func:`_refusing_a_derivative`), and with it the whole route sits
    inside ``lax.custom_linear_solve`` and is never differentiated.
    """
    by_lineax = _lineax_solve(
        "cg", matvec, b, rtol=rtol, atol=atol, restart=0, max_iters=max_iters,
    )
    broke_down = jnp.logical_not(jnp.all(jnp.isfinite(by_lineax.value)))
    if not isinstance(broke_down, jax.core.Tracer):
        if not bool(broke_down):
            return by_lineax
        return _cg_loop(
            matvec, b, x0=x0, rtol=rtol, atol=atol, max_iters=max_iters,
            preconditioner=None,
        )
    by_loop = _cg_loop(
        matvec, b, x0=x0, rtol=rtol, atol=atol,
        max_iters=jnp.where(broke_down, max_iters, 0), preconditioner=None,
    )
    return SharedSolveResult(
        value=jnp.where(broke_down, by_loop.value, by_lineax.value),
        converged=jnp.where(broke_down, by_loop.converged, by_lineax.converged),
        iters=jnp.where(broke_down, by_loop.iters, by_lineax.iters),
        residual_norm=jnp.where(
            broke_down, by_loop.residual_norm, by_lineax.residual_norm),
    )


# ---------------------------------------------------------------------------
# Differentiable wrapper (v0.4.0: gradient parity through the solve)
# ---------------------------------------------------------------------------


def _differentiable_solve(
    matvec, b, *, solve_value, transpose_solve_value, symmetric, rtol,
    atol: Optional[float],
    primal_value: Optional[Callable[[], jax.Array]] = None,
) -> SharedSolveResult:
    """The solve of ``differentiable=True``: its derivatives are linear
    solves through ``lax.custom_linear_solve``.

    Forward mode (``jax.jvp``) and reverse mode (``jax.grad``) use
    ``solve_value`` / ``transpose_solve_value`` on tangents and
    cotangents -- the same backend, *with the same preconditioner* -- and
    the derivative with respect to anything ``matvec`` closes over
    (operator coefficients) flows through.  The Krylov loop itself is
    never differentiated, which is what makes reverse mode possible at
    all (``lax.while_loop`` has no transpose).

    ``solve_value(mv, rhs)`` is the DERIVATIVE solve: it starts from zero
    and its test is purely relative to its own right-hand side, so that a
    derivative is linear in its direction.  ``lax.custom_linear_solve``
    traces its ``solve`` once and uses that one program for the primal,
    the tangent and (for a symmetric operator) the cotangent solve, so
    nothing that belongs to the primal alone may be inside it:

    * ``primal_value is None`` (the caller gave no ``x0`` and no ``atol``):
      the primal solve is itself a zero-start, purely relative solve, and
      the one program serves all three::

          x = custom_linear_solve(A, b, solve_value)

    * otherwise ``primal_value()`` is the ordinary solve, from the
      caller's ``x0`` and with the caller's ``atol``, on inputs whose
      gradient is stopped (a constant of the differentiation, ``x_p``).
      The derivative is attached by a solve of a right-hand side that is
      zero in value and carries the tangent of the residual::

          r = b - A x_p                       # x_p constant
          x = x_p + custom_linear_solve(A, r - stop_gradient(r), solve_value)

      The correction is exactly zero (a zero right-hand side from a zero
      start), so the value is the ordinary solve's.  As a function of
      ``b`` and of the operator's coefficients ``x`` is
      ``A^-1 (b - c)`` with the constant ``c = stop_gradient(r)``, the
      residual the primal solve left: its derivatives of every order are
      those of the exact solve of the system the value satisfies.  Until
      0.4.0 the one shared program held the caller's ``x0`` and ``atol``:
      the tangent and cotangent solves started from the primal's ``x0``
      (MADD-ANO-267) and stopped on the primal's absolute floor
      (MADD-ANO-268).

    The iteration count cannot escape ``custom_linear_solve`` (its
    ``solve`` returns the vector only), so ``iters`` is reported as -1;
    ``residual_norm`` / ``converged`` come from one extra matvec.
    """
    kw: dict[str, Any] = ({"symmetric": True} if symmetric
                          else {"transpose_solve": transpose_solve_value})
    if primal_value is None:
        x = lax.custom_linear_solve(matvec, b, solve_value, **kw)
    else:
        x_primal = primal_value()
        r = b - matvec(x_primal)
        x = x_primal + lax.custom_linear_solve(
            matvec, r - lax.stop_gradient(r), solve_value, **kw)
    res = b - matvec(x)
    # The test is taken on ``b`` and the residual rescaled by one power of
    # two (see ``_framed``), so their norms do not flush at small units.
    p = jax.lax.stop_gradient(pow2_frame(b))
    res_norm_hat = jnp.linalg.norm(res * p)
    tol_hat = rtol * jnp.linalg.norm(b * p)
    if atol is not None:
        tol_hat = jnp.maximum(atol * p, tol_hat)
    residual_norm = res_norm_hat / p
    # ``x`` is all finite, or NaN in every entry where a solve inside had
    # no answer (``_framed``: a right-hand side that is not finite, an
    # operator that returned a NaN, a solution or a residual norm that
    # overflows, a traced tolerance that is not a number >= 0).  Where
    # there is no answer there is no derivative either.  The tangent and
    # cotangent solves have right-hand sides of their own, which can be
    # solved where the primal was not: left alone they returned a finite
    # derivative beside a NaN value.  A FACTOR of NaN carries to both modes
    # (a ``jnp.where`` on the value would give a derivative of zero); the
    # factor is exactly one where there is an answer, so the value and its
    # derivatives keep their bits.
    answered = jnp.all(jnp.isfinite(x))
    real = jnp.finfo(x.dtype).dtype
    no_answer_is_nan = jnp.where(
        answered, jnp.ones((), real), jnp.full((), jnp.nan, real))
    # The flag is not True beside a norm that is not finite.  ``res`` above
    # is a product on the bare vectors, as it always was (its bits are
    # those of 0.4.0's earlier builds): for a solution within a factor of
    # the operator's diagonal of the dtype's largest number that product
    # overflows in an intermediate although ``x`` and ``A x`` are
    # representable, and the report is then ``converged=False`` beside an
    # infinite ``residual_norm`` where the ordinary call answers
    # (MADD-ANO-275, open; with an ``x0`` or an ``atol`` the residual ``r``
    # above overflows the same way and the value is NaN).
    return SharedSolveResult(
        value=x * no_answer_is_nan,
        converged=jnp.logical_and(
            res_norm_hat <= tol_hat,
            jnp.logical_and(answered, jnp.isfinite(residual_norm))),
        iters=jnp.int32(-1),
        residual_norm=residual_norm,
    )


def _constant_under_differentiation(fun, like: jax.Array):
    """*fun* (an operator or a preconditioner applied to vectors like
    *like*; ``None`` stays ``None``) as a constant of any differentiation:
    the same function of its argument, with the gradient stopped on every
    value it closes over that could carry a tangent.

    ``jax.closure_convert`` finds those values (it traces *fun* once and
    hoists the closed-over tracers that may be perturbed).  Stopping the
    gradient of what *fun* RETURNS would not do: the closed-over values
    would still reach ``lax.custom_linear_solve`` and ``lax.while_loop`` as
    operands with a tangent.

    A function that closes over no such value is returned as it is, not
    wrapped.  That is not only a saving: inside a ``shard_map`` body
    lineax can trace an operator only if it is a plain function that
    closes over nothing (it converts any other callable on an abstract
    vector that carries no mesh axis), and a wrapper would make every
    operator a closure.
    """
    if fun is None:
        return None
    converted, closed_over = jax.closure_convert(fun, like)
    if not closed_over:
        return fun
    constants = [lax.stop_gradient(value) for value in closed_over]
    return lambda v: converted(v, *constants)


# ---------------------------------------------------------------------------
# A call without ``differentiable=True`` refuses to be differentiated
# ---------------------------------------------------------------------------

_NOT_DIFFERENTIABLE = (
    "{name} was differentiated without differentiable=True: jax.jvp, "
    "jax.grad, jax.jacfwd, jax.jacrev or a transformation built on them "
    "reached the result of this call.  Differentiating the iteration itself "
    "does not give the derivative of the solve (it was silently wrong "
    "whenever the tangent or cotangent was not of the right-hand side's "
    "size), so the call refuses.  Pass differentiable=True: the derivative "
    "is then that of the linear solve, taken with the same backend and "
    "preconditioner, and .iters reads -1.  If the solve is meant to be a "
    "constant of the differentiation, stop the gradient on what goes INTO "
    "the call (jax.lax.stop_gradient on b, on x0, and inside matvec and the "
    "preconditioner on every differentiated value they close over); a "
    "stop_gradient on the result comes too late and raises this error."
)


def _refusal(name: str):
    """The identity on ``(value, residual_norm)`` whose JVP rule raises."""

    @jax.custom_jvp
    def as_solved(value, residual_norm):
        return value, residual_norm

    @as_solved.defjvp
    def _refuse(primals, tangents):
        del primals, tangents
        raise ValueError(_NOT_DIFFERENTIABLE.format(name=name))

    return as_solved


_REFUSALS = {name: _refusal(name) for name in ("sharded_cg", "sharded_gmres")}


def _refusing_a_derivative(result: SharedSolveResult, name: str) -> SharedSolveResult:
    """*result* as it is, made to raise if a derivative is taken through it.

    The two floating-point fields pass through an identity whose JVP rule
    raises (``jax.custom_jvp``; reverse mode linearises through the same
    rule, and ``jacfwd`` / ``jacrev`` are built on the two).  JAX calls the
    rule only when one of the two fields carries a tangent, that is, when
    it depends on something being differentiated: through ``b``, ``x0``,
    or a value the operator closes over.  A call that is not
    differentiated is not touched (the identity is inlined when the
    program is lowered: the same bits, the same compiled program), and
    neither is a call whose inputs do not depend on what is differentiated.

    Why refuse.  Differentiating the iteration is not differentiating the
    solve.  lineax's own tangent and transpose solves reuse the primal's
    tolerance, which is absolute in the primal right-hand side's frame: a
    tangent or cotangent of another size was solved to the wrong accuracy
    or not at all (measured in float32 on 256 unknowns, unit direction:
    off 6.4e-4 at ``|b|`` about 1e3 and 0.65 at 1e10; ``sharded_gmres``
    returned zeros at 1e10).  A ``lax.while_loop`` has no transpose, so
    reverse mode through the loop backends raised already.  The
    derivative of a solve is ``differentiable=True``
    (:func:`_differentiable_solve`).  MADD-ANO-266.
    """
    value, residual_norm = _REFUSALS[name](result.value, result.residual_norm)
    return SharedSolveResult(
        value=value, converged=result.converged, iters=result.iters,
        residual_norm=residual_norm,
    )


# ---------------------------------------------------------------------------
# Preconditioners (v0.4.0: first users of the ``preconditioner=`` hook)
# ---------------------------------------------------------------------------


@stability(StabilityLevel.EVOLVING)
def jacobi_preconditioner(diag: jax.Array) -> Callable[[jax.Array], jax.Array]:
    """``M(r) = r / diag`` — the diagonal (point-Jacobi) preconditioner.

    ``diag`` is the global diagonal of the operator, laid out like the
    solution vector (sharded the same way when the solver runs on a
    mesh).  Zero entries are replaced by 1 so a padded / masked row
    cannot produce ``inf``.
    """
    diag = jnp.asarray(diag)
    safe = jnp.where(diag == 0, jnp.ones_like(diag), diag)

    def apply(r):
        return r / safe

    return apply


@stability(StabilityLevel.EVOLVING)
def block_jacobi_preconditioner(
    blocks: jax.Array,
) -> Callable[[jax.Array], jax.Array]:
    """``M(r) = blockdiag(blocks)⁻¹ r`` — block-Jacobi with dense blocks.

    ``blocks`` has shape ``(n_blocks, bs, bs)``: the diagonal blocks of
    the operator in the solution vector's ordering (block ``k`` covers
    entries ``k*bs : (k+1)*bs``).  Each application is one batched
    ``jnp.linalg.solve`` — no factorisation is cached, which keeps the
    preconditioner a pure function the adjoint can differentiate
    through (the IFT / ``custom_vjp`` adjoint applies it in the adjoint
    solve as well, see ``TestPreconditioned`` in
    ``tests/cloud/multigpu/test_iterative_solver.py``).  For the
    partitioned FVM case the natural blocks are per-shard (or
    per-cell-group) diagonal blocks the mesh-prep step extracts
    host-side.
    """
    blocks = jnp.asarray(blocks)
    if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
        raise ValueError(f"blocks must have shape (n_blocks, bs, bs), got {blocks.shape}")
    n_blocks, bs, _ = blocks.shape

    def apply(r):
        if r.shape[0] != n_blocks * bs:
            raise ValueError(
                f"block_jacobi_preconditioner built for n={n_blocks * bs}, "
                f"got a vector of length {r.shape[0]}"
            )
        rb = r.reshape(n_blocks, bs, *r.shape[1:])
        if rb.ndim == 2:
            out = jnp.linalg.solve(blocks, rb[..., None])[..., 0]
        else:
            out = jnp.linalg.solve(blocks, rb)
        return out.reshape(r.shape)

    return apply


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
    atol: Optional[float] = None,
    max_iters: int = 200,
    preconditioner: Optional[Callable[[jax.Array], jax.Array]] = None,
    backend: str = "auto",
    differentiable: bool = False,
) -> SharedSolveResult:
    """Preconditioned conjugate gradient on a sharded matvec.

    ``differentiable=True`` routes the solve through
    ``lax.custom_linear_solve`` so ``jax.grad`` / ``jax.jvp`` through the
    result (and the IFT adjoint of a coupling group whose node calls
    this) are exact linear-solve adjoints using the same backend and
    preconditioner; ``iters`` is then -1.  Off by default so the STABLE
    result contract (iteration count) is unchanged; a node whose
    ``update`` calls this inside a differentiated step must set it.

    From 0.4.0 that "must" is enforced: a call without
    ``differentiable=True`` refuses to be differentiated (``Raises``).
    Until then ``jax.jvp`` / ``jax.grad`` through such a call
    differentiated the iteration itself, which is not the derivative of
    the solve: it was silently wrong whenever the tangent or cotangent
    was not of ``b``'s size (measured in float32 on 256 unknowns with a
    unit direction: off 6.4e-4 at ``|b|`` about 1e3 and 0.65 at 1e10;
    MADD-ANO-266).  With the flag the same derivatives were measured
    within 1.2e-6 of a float64 solve at every ``|b|`` from 1e-10 to
    1e10 (two directions).

    The tangent and cotangent solves of ``differentiable=True`` start
    from zero and are purely relative to their own right-hand side,
    whatever ``x0`` and ``atol`` the call was given: those two belong to
    the primal solve (``x0``, ``atol``).  The preconditioner and ``x0``
    carry no derivative.  The value is the ordinary call's, bit for bit;
    with an ``x0`` or an ``atol`` the call executes three to six
    operator products more than the ordinary one, and without them one
    more (three more on the default route of a call that is not
    jitted).

    **``converged`` does not cover the derivative solves
    (MADD-ANO-270, open).**  Each derivative is one more solve of the
    same system, from zero, under the same ``rtol`` and ``max_iters``,
    and nothing reports whether it met its tolerance: ``converged`` and
    ``residual_norm`` are the value's.  A primal that needs fewer steps
    than a solve from zero (an ``x0`` near the solution, an ``atol``
    above ``||b||``, a right-hand side that is one eigenvector) reads
    ``converged=True`` beside a derivative from a solve that stopped on
    its budget.  Measured in float64 on 4096 unknowns at a condition
    number of 4001, with the default ``rtol`` and ``max_iters=200`` and
    ``x0`` the solution of the previous right-hand side:
    ``converged=True``, the JVP off 2.6e-3 and the gradient 2.4e-3 on
    the loop backend.  With ``differentiable=True`` give ``max_iters``
    what a solve from zero needs.

    Parameters
    ----------
    matvec : callable
        ``matvec(x) -> A @ x``.  The matvec is responsible for any
        ``shard_map`` it requires; the solver itself runs at the outer
        (global) level.  Must be self-adjoint and positive-definite —
        for general operators use :func:`sharded_gmres` instead.  One
        that returns a NaN or an infinity gets NaN (``Returns``).  The
        solver can also be called inside a ``shard_map`` body, one solve
        per device; the lineax route (the default) then takes only a
        plain function that closes over nothing, and ``backend="loop"``
        any callable (lineax converts a closure once, on an abstract
        vector that carries no mesh axis).  With ``differentiable=True``
        a ``jacobi_preconditioner`` built in the body from a diagonal
        that is the same constant on every device raises there (JAX's
        ``pvary`` error); one built from a per-device diagonal answers.
        :func:`sharded_gmres` cannot be called inside a ``shard_map``
        body at all.
    b : jax.Array
        Right-hand side.  1-D.  Sharded according to ``in_specs`` when
        ``mesh`` is provided.
    mesh, in_specs : optional
        When both provided, ``b`` and ``x`` are placed on the mesh with
        the given PartitionSpec.  When omitted, the solver runs on
        whatever device(s) JAX chooses for ``b`` — single-device or a
        pre-sharded input.
    x0 : jax.Array, optional
        Initial guess.  Defaults to zeros_like(b).  Used by the loop
        backend (lineax's CG is not given it).  Under
        ``differentiable=True`` it starts the primal solve only: the
        tangent and cotangent solves start from zero.  Until 0.4.0 they
        started from this ``x0`` rescaled to their own right-hand side
        (with ``max_iters=0`` the derivative was ``x0`` itself; at
        ``|b|`` about 1e5 with ``x0`` the solution a unit-direction JVP
        was off 2.6e-2; MADD-ANO-267).
    rtol : float
        Relative tolerance: stop when ``||r|| <= max(atol, rtol * ||b||)``.
        On the loop backend ``r`` is the residual the iteration updates
        (``r <- r - alpha * A p``), which drifts from ``b - A x`` by about
        ``eps * kappa`` (``kappa`` the condition number), so the loop can
        stop on its test before the residual of the system has met the
        tolerance.  The result then says so: ``converged=False`` and the
        residual it has (``Returns``).  In float32 the default ``1e-6``
        is met up to a ``kappa`` of about 17; ask for an ``rtol`` a few
        times above ``eps * kappa``, or solve in float64.  A number
        >= 0.  A NaN or negative ``rtol`` given as a concrete number is
        refused with a ``ValueError``; one that is traced (an argument
        of a jitted or vmapped function) cannot raise and is answered
        with NaN (``Returns``).  Until 0.4.0 ``rtol`` was not read: a
        NaN gave zeros with ``converged=True`` on the lineax routes
        (MADD-ANO-271).  ``rtol=0`` without a positive ``atol`` asks
        for a residual of exactly zero, and ``converged`` is then
        ``residual_norm == 0`` on every route; infinity passes
        everything.
    atol : float or None
        Absolute floor of that test, in ``b``'s units: give a number only to
        assert a noise floor in ``b``'s own units.  ``None`` (the default)
        makes the test purely relative, so it means the same in any units:
        ``||r|| <= rtol * ||b||`` on the loop backend, and on lineax's
        entrywise test the floor ``rtol * max|b|``.  The solve itself runs
        on ``b`` rescaled by the power of two that brings ``max|b|`` into
        ``[0.5, 1)``, which is exact.  Before 0.4.0 the default was an
        absolute ``1e-8``: a right-hand side below about ``1e-2``
        (``1e-8 / rtol``) was solved only to that floor, and one below about
        ``1e-8`` passed the test with the zero initial guess before its
        first step and came back wrong with ``converged=True``.  On the loop
        backend a ``b`` with ``||b|| >= 1e-8 / rtol`` stops exactly where it
        did.  A negative (or NaN) ``atol`` given as a concrete number
        is refused with a ``ValueError``; one that is traced (an
        argument of a jitted or vmapped function) cannot be read when
        the call is traced and is answered with NaN (``Returns``).
        Until 0.4.0 it was passed on, and the lineax routes read
        ``converged=True`` on a value 72 % off (MADD-ANO-271).
        ``atol=0.0`` is the relative test, as ``None`` is, with one
        difference on the lineax routes: lineax is handed the floor
        zero instead of ``rtol * max|b|``, stops where it stops with
        that (``Returns``), and ``converged`` is
        ``residual_norm <= rtol * ||b||``.  Prefer ``None``.
        Under ``differentiable=True`` an ``atol`` bounds
        the primal solve only: the tangent and cotangent solves are
        purely relative (``||r|| <= rtol * ||rhs||`` of their own
        right-hand side), so that a derivative is linear in its
        direction.  Until 0.4.0 the floor was applied to them too, in
        their own frame: at ``atol=1e-3`` a direction of norm 1e-5 gave
        a derivative of zero on the loop backend and of half the size on
        lineax (MADD-ANO-268).
    max_iters : int
        Iteration budget.
    preconditioner : callable, optional
        ``M(r) -> M @ r``.  Applied as a left preconditioner.  v0.3.0
        ships only the identity (None) — Jacobi or other preconditioners
        are the caller's responsibility.
    backend : {"auto", "lineax", "loop"}
        Solver backend.  ``"auto"`` (default) uses lineax's CG, and the
        hand-rolled loop in two cases: when ``preconditioner`` is set
        (lineax's CG takes none), and where lineax's CG breaks down
        (below).  ``"loop"`` forces the loop.  ``"lineax"`` forces lineax,
        with no fallback.

        **Where lineax's CG breaks down (MADD-ANO-264).**  lineax's CG
        sets its step to NaN when ``|p.A p| <= 200 * eps * n * |r.r|``
        (``n`` unknowns): in float32 a quotient ``p.A p / r.r`` under
        ``2.4e-5 * n``, which is 2.4 at 1e5 unknowns and 24 at 1e6.  The
        quotient lies between the operator's extreme eigenvalues, so an
        operator whose eigenvalues are of order one cannot pass it there,
        and a smooth right-hand side fails far below (measured: at 4096
        unknowns).  Every entry of lineax's answer is then NaN.

        From 0.4.0 the default route answers there.  ``"auto"`` without a
        preconditioner runs lineax's CG as before; if the value it returns
        is not all finite, the same system is solved by the loop, from
        ``x0``, and the result is the loop's: its ``.value``, its
        ``.iters``, and ``.converged`` / ``.residual_norm`` as on the loop
        backend (``Returns``).  Where lineax's value is finite the result
        is lineax's, bit for bit what it was.  0.3.0 and 0.3.1 returned
        the NaN: there, pass ``backend="loop"``.  ``backend="lineax"``,
        asked for by name, still returns NaN in every entry of ``.value``,
        a NaN ``.residual_norm`` and ``converged=False`` after one or two
        steps.  (lineax's CG is not given ``x0``: where it answers, the
        initial guess is not used.)

        What the fallback costs.  Under a trace (``jit``, ``vmap``, a
        ``shard_map`` body) both solvers are in the program.  The loop
        is given an iteration cap of ``max_iters`` where lineax's value
        is not finite and of zero where it is, and each field of the
        result is then selected: a default call that does not fall back
        executes two operator products more than ``backend="lineax"``
        (the loop's ``b - A x0`` and the product of its report) and no
        iteration.  Under ``vmap`` the loop iterates as long as any
        member of the batch that fell back needs.  Called eagerly, only
        lineax runs where lineax answers.  To differentiate through a
        solve, whether it falls back or not, set ``differentiable=True``.

    Returns
    -------
    SharedSolveResult
        ``.value`` (solution), ``.converged``, ``.iters``, ``.residual_norm``.
        ``.residual_norm`` is ``||b - A x||`` of the returned value on
        every route: one extra product in the working dtype, with the
        operator and the right-hand side given.  What ``.converged``
        reports depends on the route:

        * loop backend (``backend="loop"``; ``"auto"`` with a
          preconditioner; ``"auto"`` where lineax's CG broke down):
          ``.converged`` is ``.residual_norm <= max(atol, rtol * ||b||)``,
          with no allowance for rounding.  The iteration itself stops on
          the residual it updates by recurrence, which in float32 drifts
          from ``b - A x`` by about ``eps * kappa``.  Where it stopped
          before the system's residual met the tolerance, the result
          reads ``converged=False`` with the value the loop reached and
          the residual that value has.  Measured in float32 on a shifted
          1-D Laplacian of 4096 unknowns with one white-noise ``b``
          (``numpy.random.default_rng(0)``), the smallest ``rtol`` of
          1e-3, 3e-4, 1e-4, ... that reads ``converged=True``::

              condition number     2     9     17    41    134   401   4001
              smallest rtol met    3e-7  1e-6  1e-6  3e-6  1e-5  1e-4  1e-3

          A second right-hand side (seed 1) gives the same row; a third
          (seed 2) reads 3e-5 at 401 and 3e-4 at 4001, one rung lower:
          the last two columns are within a factor of about three of the
          rung, not on it.  So the default ``rtol=1e-6`` is met up to a
          condition number of about 17; at 401 the loop stops at a
          residual of 1.1e-5 to 1.2e-5 ``||b||`` and at 4001 of 1.0e-4
          to 1.2e-4 (the three right-hand sides).  A solve that stopped on its
          tolerance within the float32 floor of it can read ``False`` by
          a hair (measured: 1.06 times ``rtol``).  Choose an ``rtol`` a
          few times above ``eps * kappa``, or solve in float64, or read
          ``.residual_norm``.  Until 0.4.0 this route reported the
          recursive residual instead: ``converged=True`` and a norm under
          ``rtol * ||b||`` on those same solves (MADD-ANO-263).
          ``.value`` and ``.iters`` are what they were: when the loop
          stops has not changed.
        * lineax backend (``backend="lineax"``; ``"auto"`` without a
          preconditioner where lineax's value is finite): ``.converged``
          is lineax's own verdict, not a test of ``.residual_norm``: its
          entrywise test ``|r_i| <= atol + rtol * |b_i|`` on a residual
          it recomputes every tenth step, and the same test on its last
          step.  By name, in float32, this route returns NaN above about
          1e5 unknowns (MADD-ANO-264, see ``backend``).  One exception:
          where the floor lineax is handed is zero (``rtol=0``, whose
          floor without an ``atol`` is ``rtol * max|b|``; an explicit
          ``atol=0.0``; a tolerance that underflows, in float32
          ``rtol=1e-44``), lineax's test is a quotient that is
          ``0 / 0`` at every entry where the residual and ``b`` are
          both exactly zero, and it stops there and reports success.
          ``.converged`` is then not its verdict but the loop
          backend's statement on the value lineax returned,
          ``.residual_norm <= max(atol, rtol * ||b||)``, which for a
          tolerance of zero is ``.residual_norm == 0``.  Until 0.4.0's
          fix: ``converged=True`` after one step at a residual of 0.4
          ``||b||``, for a ``b`` that is one unit vector, with
          ``atol=0.0`` or with ``rtol=0`` (MADD-ANO-274; ``atol=0.0``
          in 0.3.0 and 0.3.1).  ``.value`` and ``.iters``
          are lineax's as they were: such a call still stops early,
          and now says so.
        * ``differentiable=True``, on either backend: ``.converged`` is
          the loop backend's statement,
          ``||b - A x|| <= max(atol, rtol * ||b||)`` on the returned
          value; ``.iters`` is -1.  It says nothing of the tangent and
          cotangent solves (MADD-ANO-270, above).

        The loop backend and ``differentiable=True`` so give one answer
        for one system.  lineax's verdict is its own and can differ
        (measured: ``False`` after all 3000 steps at a residual of
        7.2e-6 ``||b||``, where ``rtol`` was 1e-6).

        A ``b`` with a NaN or infinite entry gives NaN in every entry of
        ``.value``, a NaN ``.residual_norm`` and ``converged=False`` (and
        under ``differentiable=True`` a non-finite tangent or cotangent a
        NaN derivative).  Until 0.4.0 it gave zeros, on lineax and for
        ``+-inf`` on the loop backend with ``converged=True``
        (MADD-ANO-155).

        An operator that returns a NaN or an infinity gets the same
        answer on every route, and so does any solve whose
        ``.residual_norm`` is not finite: ``converged`` is never True
        beside such a norm.  Until 0.4.0 the default call and
        ``backend="lineax"`` answered a non-finite operator with zeros,
        ``converged=True`` and ``iters=0`` (MADD-ANO-269).

        At the bottom of the dtype's range the value cannot hold the
        solution: entries of it under the smallest normal number
        (float32: 1.2e-38) are flushed to zero when the result is scaled
        back, ``.residual_norm`` reads 0.0, and ``.converged`` is the
        verdict of the scaled solve.  Measured in float32 on 64
        unknowns, by ``max|b|``, on two systems: a relative error of
        2e-2 and 9e-2 at ten times that number, 1e-3 and 4e-3 at a
        hundred times, none (5e-7) at 1e4 times.  Rescale such a system.

        At the top of the range the answer is tested after it is
        scaled back.  A solution too large for the dtype gets NaN in
        every entry, a NaN ``.residual_norm`` and ``converged=False``
        on every route, and under ``differentiable=True`` a derivative
        of NaN (until 0.4.0's fix: infinities beside a finite
        ``.residual_norm``, and ``converged=True`` where the scaled
        solve met its tolerance; MADD-ANO-272).  So does a solve whose
        residual has a norm too large for the dtype: a right-hand side
        whose own norm overflows, left far from solved.  A right-hand
        side whose norm overflows is solved where its solution fits
        (until 0.4.0 the loop answered it with zeros and
        ``converged=True``).  Rescale a system whose solution does not
        fit, or solve it in float64.

        A traced ``rtol`` or ``atol`` that is negative or NaN gets the
        same answer on every route (MADD-ANO-271), and under
        ``differentiable=True`` a derivative of NaN.  A result is so
        one of two things: every entry of ``.value`` finite beside a
        finite ``.residual_norm``, or NaN in both with
        ``converged=False``.

        One exception, under ``differentiable=True`` only (MADD-ANO-275,
        open): the report of that route, and the residual it attaches
        the derivative to, are products on the vectors as they are, not
        in the frame of ``b``.  For a solution within a factor of the
        operator's diagonal of the dtype's largest number such a
        product overflows in an intermediate although the solution
        fits.  The call then returns the right value with
        ``converged=False`` and an infinite ``.residual_norm``, and
        with an ``x0`` or an ``atol`` NaN in every entry, where the
        ordinary call answers (measured in float32 at a solution of 0.5
        and 0.7 of the largest number behind a diagonal of 2.05).
        Rescale such a system, or solve it in float64.

    Raises
    ------
    ValueError
        When the result of a call without ``differentiable=True`` is
        differentiated (``jax.jvp``, ``jax.grad``, ``jax.jacfwd``,
        ``jax.jacrev``, or a transformation built on them), with respect
        to ``b`` or a value ``matvec`` closes over, and with respect to
        ``x0`` on the routes that read it (the loop backends, and the
        default route under a trace; elsewhere ``x0`` is not read and
        its derivative is zero).  Pass
        ``differentiable=True``; or, if the solve is meant to be a
        constant, stop the gradient on what goes INTO the call
        (``jax.lax.stop_gradient`` on ``b``, on ``x0``, and inside
        ``matvec`` on the values it closes over): a ``stop_gradient`` on
        the result comes too late.  A call that is not differentiated,
        or whose inputs do not depend on what is differentiated, is not
        affected: same bits, same compiled program.  An input that is
        handed an explicit tangent of zeros (``jax.jvp``) counts as
        differentiated and is refused.  ``rtol``, ``atol`` and
        ``max_iters`` are not differentiable arguments: a derivative
        with respect to one of them is zero or an error of JAX or of
        lineax, by route.
    ValueError
        For a negative or NaN ``rtol`` or ``atol`` given as a concrete
        number (a Python or NumPy number, or a JAX array that is not
        traced), an unknown ``backend``, a
        ``preconditioner`` with ``backend="lineax"``, or a ``mesh``
        without ``in_specs``.

    Stability
    ---------
    This signature is tagged ``@stability(STABLE)`` — see
    ``docs/developer_guide/stability_report.md``.  The v0.4.0 commitment
    (sharded FVM in MIME) reads against this surface.
    """
    _validate_solver_inputs(
        matvec, b, mesh=mesh, in_specs=in_specs, name="sharded_cg",
        rtol=rtol, atol=atol,
    )
    tolerances_are_numbers = _tolerances_are_numbers(rtol, atol)
    b = _materialise_b(b, mesh=mesh, in_specs=in_specs)
    x0_given = x0 is not None
    if x0 is None:
        x0 = jnp.zeros_like(b)
    else:
        x0 = _materialise_b(x0, mesh=mesh, in_specs=in_specs)

    def _dispatch(mv, rhs, x_start, atol_hat, M):
        if backend == "lineax":
            if preconditioner is not None:
                raise ValueError(
                    "backend='lineax' cannot apply a preconditioner (lineax's CG "
                    "takes none); use backend='loop' or 'auto' (auto routes a "
                    "preconditioned solve to the loop backend)."
                )
            return _lineax_solve(
                "cg", mv, rhs, rtol=rtol, atol=atol_hat,
                restart=0, max_iters=max_iters,
            )
        if backend == "loop":
            return _cg_loop(
                mv, rhs, x0=x_start, rtol=rtol, atol=atol_hat,
                max_iters=max_iters, preconditioner=M,
            )
        if backend != "auto":
            raise ValueError(f"Unknown backend {backend!r}; expected auto/lineax/loop")

        # auto: lineax unless a preconditioner is set (lineax's CG takes
        # none), in which case the loop backend applies it.  Without one,
        # the loop also answers where lineax's value is not finite.
        if M is None:
            return _lineax_cg_or_the_loop_where_it_breaks_down(
                mv, rhs, x0=x_start, rtol=rtol, atol=atol_hat,
                max_iters=max_iters,
            )
        return _cg_loop(
            mv, rhs, x0=x_start, rtol=rtol, atol=atol_hat,
            max_iters=max_iters, preconditioner=M,
        )

    def _run(mv, rhs, x_start, atol_given, M):
        # One solve, in a power-of-two frame of its right-hand side (``_framed``).
        return _framed(
            lambda b_hat, x_hat, atol_hat: _dispatch(mv, b_hat, x_hat, atol_hat, M),
            rhs, x_start, atol_given, tolerances_are_numbers)

    if not differentiable:
        return _refusing_a_derivative(
            _run(matvec, b, x0, atol, preconditioner), "sharded_cg")

    # The solution does not depend on what preconditions the iteration: the
    # preconditioner carries no derivative.  (Left as it came, a
    # preconditioner that closes over a differentiated value made
    # ``lax.custom_linear_solve`` raise where nothing else was
    # differentiated.)
    M_constant = _constant_under_differentiation(preconditioner, b)

    def _derivative_solve(mv, rhs):
        # The tangent and cotangent solves: from zero, and purely relative
        # to their own right-hand side (``_differentiable_solve``).
        return _run(mv, rhs, jnp.zeros_like(rhs), None, M_constant).value

    primal_value: Optional[Callable[[], jax.Array]] = None
    if x0_given or atol is not None:
        def _the_ordinary_solve():
            # From ``x0`` and with ``atol``, as a constant of the
            # differentiation.
            return _run(
                _constant_under_differentiation(matvec, b), lax.stop_gradient(b),
                lax.stop_gradient(x0), atol, M_constant,
            ).value

        primal_value = _the_ordinary_solve

    return _differentiable_solve(
        matvec, b, solve_value=_derivative_solve,
        transpose_solve_value=None, symmetric=True,
        rtol=rtol, atol=atol, primal_value=primal_value,
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
    atol: Optional[float] = None,
    restart: int = 50,
    max_iters: int = 200,
    preconditioner: Optional[Callable[[jax.Array], jax.Array]] = None,
    backend: str = "auto",
    differentiable: bool = False,
) -> SharedSolveResult:
    """Restarted GMRES on a sharded matvec.

    ``differentiable=True`` as for :func:`sharded_cg`; the transpose
    solve is GMRES on the transposed operator (``jax.linear_transpose``),
    with the same preconditioner.  As there, a call without it refuses to
    be differentiated (a ``ValueError``, MADD-ANO-266: until 0.4.0 the
    default route returned a derivative of zeros in float32 at ``|b|``
    of 1e10 and of 1e-10), and with it the tangent and cotangent solves
    start from zero and are purely relative, whatever ``x0`` and
    ``atol`` were given; ``converged`` does not cover them
    (MADD-ANO-270).

    Unlike :func:`sharded_cg`, this solver cannot be called inside a
    ``shard_map`` body: every route raises there (lineax a
    ``ValueError`` on the structures of vector and operator, the loop a
    ``TypeError`` on the carry of its ``while_loop``), as before 0.4.0.

    See :func:`sharded_cg` for the matvec / mesh / in_specs contract, and
    for the answer to a ``b`` with a NaN or infinite entry, to an
    operator that returns one, to a solution too large for the dtype and
    to a traced tolerance that is negative or NaN: NaN, with
    ``converged=False`` (MADD-ANO-155, MADD-ANO-269, MADD-ANO-272,
    MADD-ANO-271).

    Parameters
    ----------
    x0 : jax.Array, optional
        Initial guess, as for :func:`sharded_cg`.  Read by the loop
        backend only: the default route is lineax's GMRES, which is not
        given it, so that where the default route answers the initial
        guess is not used.
    rtol, atol : float, float or None
        As for :func:`sharded_cg`: ``atol=None`` (the default) is
        relative; a negative or NaN ``rtol`` or ``atol`` given as a
        concrete number is refused with a ``ValueError``, and one that
        is traced is answered with NaN (MADD-ANO-271).  With a floor of
        zero (``rtol=0``, an explicit ``atol=0.0``) ``converged`` on
        the default route is not lineax's verdict but
        ``residual_norm <= max(atol, rtol * ||b||)`` (``Returns``).
    max_iters : int
        Iteration budget.  Its unit depends on the backend.

        On the loop backend it is a number of steps, of one operator
        product each, counted in whole restart cycles, rounded up, and
        at least one cycle runs: a budget of 0 or 1 is ``restart``
        steps.  The call executes ``restart + 2`` products per cycle
        and one more (measured: 53 for one cycle of 50, 209 for four).

        **On the lineax backend, which is the default, it is not a
        bound on the work (MADD-ANO-273, open).**  lineax is handed
        ``max_steps = max(4 * restart, max_iters)``, and lineax counts
        one step per restart CYCLE of up to ``restart`` operator
        products.  So the budget is in cycles, ``restart`` times what
        the argument names, and never under ``4 * restart`` cycles;
        ``.iters`` counts the same cycles.  Measured in float32 on 1000
        unknowns at a condition number of 4001 (white noise, the
        default ``restart=50`` and ``rtol``) with ``max_iters=10``:
        3,164 operator products and ``iters=63`` on lineax 0.0.7, and
        1,685 products and ``iters=34`` on lineax 0.1.1 (lineax stopped
        before its 200 cycles, at a different one on each version); the
        loop backend executed 53 products and reported ``iters=50``.
        A caller who needs a bound on the work passes
        ``backend="loop"``.  The unit of this route is left as it is in
        0.4.0: a budget in steps would change what every default call
        with a binding budget returns.
    restart : int
        Krylov subspace dimension before restart.  Capped to ``b.shape[0]``
        inside the solver, on both backends (the loop backend returned NaN
        above it until 0.4.0).  Lineax's default is 20, which silently
        produces low-rank approximations on problems larger than 20
        unknowns — we default to 50 here.  See the regression test in
        ``tests/core/test_coupling_ift_lineax.py`` for the historical
        gotcha.  On the lineax backend ``restart`` is also the number
        of operator products behind each unit of ``max_iters`` and of
        ``.iters`` (``max_iters``).

    Returns
    -------
    SharedSolveResult
        What ``.converged`` and ``.residual_norm`` report depends on the
        route, as it does for :func:`sharded_cg`:

        * loop backend (``backend="loop"``, and ``"auto"`` with a
          preconditioner): ``.residual_norm`` is ``||b - A x||`` of the
          returned value (one product, in the working dtype) and
          ``.converged`` is that against ``max(atol, rtol * ||b||)``.
          Without a preconditioner this is the residual the last restart
          cycle computed, and the test the cycles stop on.  With a
          ``preconditioner`` the cycles test the left-preconditioned
          residual, ``||M (b - A x)||`` against
          ``max(atol, rtol * ||M b||)``, which is not in ``b``'s units;
          the report is still of the system given, from one more
          product with the caller's operator.  A solve whose
          preconditioned residual passed its test can therefore read
          ``converged=False`` (measured in float32 on a diagonally
          scaled Laplacian of 32 unknowns behind block-Jacobi at
          ``rtol=1e-6``: ``||b - A x||`` 12 to 18 times the tolerance, by
          CPU).  Until
          0.4.0 both fields were the preconditioned system's
          (MADD-ANO-263): a number not in ``b``'s units, 0.40 times
          ``||b - A x||`` behind a Jacobi preconditioner of diagonal 2.5.
          At the float32 floor ``.residual_norm`` and the same residual
          evaluated in float64 differ by rounding of the order of the
          residual (measured on one CPU: 8.1e-6 against 6.1e-6, with
          ``converged=False``; on the system above the two were 4 % apart
          on one CPU and 19 % on another).  ``.iters`` counts whole
          cycles, a multiple of ``restart``; ``.value`` and ``.iters``
          are what they were before 0.4.0.
        * lineax backend (the default without a preconditioner):
          ``.residual_norm`` is ``||b - A x||`` of the returned value;
          ``.converged`` is lineax's own verdict, except where the
          floor of its entrywise test is zero (``rtol=0``, an explicit
          ``atol=0.0``, a tolerance that underflows): there lineax
          stops on a quotient of ``0 / 0`` and reports success, and
          ``.converged`` is ``.residual_norm <= max(atol, rtol * ||b||)``
          instead, which for a tolerance of zero is
          ``.residual_norm == 0`` (until 0.4.0's fix: ``True`` at a
          residual of 5e-8 to 5e-5 of ``||b||`` with ``rtol=0``, and of
          0.17 for a ``b`` that is one unit vector).  ``.iters`` counts
          restart CYCLES (lineax's ``num_steps``), not steps: about
          ``.iters * restart`` operator products were executed
          (``max_iters``; MADD-ANO-273).  The NaN of
          MADD-ANO-264 is lineax CG's alone: this route was measured to
          converge in float32 up to 1e6 unknowns on the systems where
          :func:`sharded_cg`'s default route returns NaN.
        * ``differentiable=True``: one extra product on the returned
          value, unpreconditioned, as for :func:`sharded_cg`.

    Stability
    ---------
    Tagged ``@stability(STABLE)``.  v0.4.0 / M3 commitment: MIME's
    FVM PISO pressure correction reads against this signature.
    """
    _validate_solver_inputs(
        matvec, b, mesh=mesh, in_specs=in_specs, name="sharded_gmres",
        rtol=rtol, atol=atol,
    )
    tolerances_are_numbers = _tolerances_are_numbers(rtol, atol)
    b = _materialise_b(b, mesh=mesh, in_specs=in_specs)
    x0_given = x0 is not None
    if x0 is None:
        x0 = jnp.zeros_like(b)
    else:
        x0 = _materialise_b(x0, mesh=mesh, in_specs=in_specs)

    def _dispatch(mv, rhs, x_start, atol_hat, M, of_a_derivative):
        if backend == "lineax":
            if preconditioner is not None:
                raise ValueError(
                    "backend='lineax' cannot apply a preconditioner (lineax's GMRES "
                    "takes none); use backend='loop' or 'auto'."
                )
            return _lineax_solve(
                "gmres", mv, rhs, rtol=rtol, atol=atol_hat,
                restart=restart, max_iters=max_iters,
            )
        if backend not in ("loop", "auto"):
            raise ValueError(f"Unknown backend {backend!r}; expected auto/lineax/loop")

        # auto: lineax unless a preconditioner is set (lineax's GMRES
        # takes none), in which case the loop backend applies it.
        if backend == "auto" and M is None:
            return _lineax_solve(
                "gmres", mv, rhs, rtol=rtol, atol=atol_hat,
                restart=restart, max_iters=max_iters,
            )
        return _gmres_loop(
            mv, rhs, x0=x_start, rtol=rtol, atol=atol_hat,
            restart=restart, max_iters=max_iters,
            preconditioner=M, a_zero_rhs_needs_no_cycle=of_a_derivative,
        )

    def _run(mv, rhs, x_start, atol_given, M, of_a_derivative=False):
        # As in :func:`sharded_cg`: in a power-of-two frame of the rhs.
        return _framed(
            lambda b_hat, x_hat, atol_hat: _dispatch(
                mv, b_hat, x_hat, atol_hat, M, of_a_derivative),
            rhs, x_start, atol_given, tolerances_are_numbers)

    if not differentiable:
        return _refusing_a_derivative(
            _run(matvec, b, x0, atol, preconditioner), "sharded_gmres")

    M_constant = _constant_under_differentiation(preconditioner, b)

    def _derivative_solve(mv, rhs):
        # As in :func:`sharded_cg`: from zero, purely relative.  ``mv`` is
        # the operator for the tangent solve and its transpose
        # (``jax.linear_transpose``) for the cotangent solve.
        return _run(mv, rhs, jnp.zeros_like(rhs), None, M_constant,
                    of_a_derivative=x0_given or atol is not None).value

    primal_value: Optional[Callable[[], jax.Array]] = None
    if x0_given or atol is not None:
        def _the_ordinary_solve():
            return _run(
                _constant_under_differentiation(matvec, b), lax.stop_gradient(b),
                lax.stop_gradient(x0), atol, M_constant,
            ).value

        primal_value = _the_ordinary_solve

    return _differentiable_solve(
        matvec, b, solve_value=_derivative_solve,
        transpose_solve_value=_derivative_solve, symmetric=False,
        rtol=rtol, atol=atol, primal_value=primal_value,
    )


__all__ = [
    "SharedSolveResult",
    "block_jacobi_preconditioner",
    "jacobi_preconditioner",
    "sharded_cg",
    "sharded_gmres",
]
