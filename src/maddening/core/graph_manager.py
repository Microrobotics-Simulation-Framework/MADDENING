"""
GraphManager -- central orchestrator for the MADDENING simulation graph.

Owns all node state, builds the execution schedule, and JIT-compiles the
full graph step into a single XLA computation via ``jax.jit``.

Supports multi-rate timesteps: each node declares its own ``delta_t``,
and the graph manager derives a *base timestep* (GCD of all node
timesteps).  The compiled step advances at the base rate; each node
updates only when its own sub-step counter fires.  For JAX traceability
the update is always computed but conditionally applied via
``jnp.where``.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

# ``lineax`` is imported lazily inside ``_ift_solve_bwd`` — it pulls in
# equinox + optax transitively, which we do NOT want to make a hard
# module-load-time dependency.  Only users who opt into ``solver='ift'``
# trigger the lineax import path.

from maddening.core.coupling import CouplingGroup
from maddening.core.edge import EdgeSpec
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.node import SimulationNode
from maddening.core.compliance.stability import stability
from maddening.core.schedule import (
    detect_cycles,
    find_strongly_connected_components,
    identify_back_edges,
    topological_sort,
)


# ------------------------------------------------------------------
# Internal bookkeeping structs
# ------------------------------------------------------------------

@dataclass
class _NodeSpec:
    """Everything the graph manager needs to know about a node."""
    node: SimulationNode          # the descriptor object
    update_fn: Callable           # node.update  (pure function)
    timestep: float


@dataclass(frozen=True)
class ShardingIssue:
    """A single issue found by :meth:`GraphManager.validate_sharding`.

    ``severity`` is ``"error"`` (raise-worthy) or ``"warning"``
    (advisory).  ``code`` is a short slug callers can switch on.
    """
    severity: str   # "error" | "warning"
    code: str       # short stable slug, e.g. "sharded_node_mesh_axes_mismatch"
    message: str    # human-readable explanation
    affected_nodes: list[str]


@dataclass(frozen=True)
class ExternalInputSpec:
    """Declares an external input that flows into a node's boundary_inputs.

    External inputs come from outside the graph (controllers, sensors,
    user commands) rather than from other nodes via edges.
    """
    target_node: str
    target_field: str
    shape: tuple
    dtype: Any = jnp.float32


# ------------------------------------------------------------------
# Implicit-function-theorem fixed-point solver
# ------------------------------------------------------------------
#
# The functions below implement the "deep equilibrium" / IFT
# differentiation pattern for coupling-group fixed points.  They are
# defined at module scope so neither the forward nor the backward path
# closes over any JAX tracer — this is the key constraint that lets
# ``jax.grad(jax.jit(gm.step))`` flow correctly through the custom_vjp
# rule (see optimistix's ``_implicit_impl`` / its ``_is_global_function``
# assertion for the same pattern, and JAX issue #2912 for the
# DynamicJaxprTracer-as-constant failure mode when this rule is
# violated).
#
# ``_F_dispatch`` is *the* one-iteration function; it is invoked from
# a top-level signature ``(x, consts)`` where ``consts`` is a pytree
# of tracers extracted by ``jax.closure_convert`` at the call site.


def _F_dispatch(F_pure, x, consts):
    # Trampoline: forwards to a closure-converted pure function.
    # Kept top-level so the custom_vjp residual sees ``F_pure`` as a
    # plain Python global, not a captured closure.
    return F_pure(x, *consts)


def _ift_fixed_point_fwd_impl(F_pure, x0, consts, tol, max_iter, acceleration):
    """While-loop fixed-point iteration of ``x = F_pure(x, *consts)``.

    Returns ``(x_star, n_iters)``.  No autodiff machinery here — that
    is layered on by ``_ift_solve``'s custom_vjp.

    ``acceleration`` is a static Python string selecting the forward
    iterator wrapper.  Supported values:

    - ``"none"``    : bare Gauss-Seidel, ``x_{k+1} = F(x_k)``.
    - ``"aitken"``  : Aitken delta-squared relaxation around ``F``.
    - ``"iqn-imvj"``: Interface quasi-Newton with inverse multi-vector
      Jacobian.  Builds ``V`` (input differences) and ``W`` (residual
      differences) matrices within the while_loop carry, solves a
      rank-deficient least-squares each iteration to get a coefficient
      vector ``c``, and applies ``dx_qn = V c - r_cur``.  Per-step only
      — V/W reset to zeros at the start of each timestep; cross-timestep
      warm-start is a deferred follow-up.

    The backward (IFT adjoint) is **acceleration-agnostic**: it
    differentiates the bare contraction ``F`` at the fixed point
    ``x*``, since acceleration is just a forward-pass technique for
    *getting to* ``x*`` faster — at the fixed point ``x* = F(x*)``
    regardless of what wrapper was used to reach it.  This is why the
    bwd rule below does not need an ``acceleration`` argument.
    """
    if acceleration == "none":

        def cond(carry):
            x, x_prev, i = carry
            not_converged = jnp.linalg.norm(x - x_prev) > tol
            not_maxed = i < max_iter
            first = i == jnp.int32(0)
            return jnp.logical_or(first, jnp.logical_and(not_converged, not_maxed))

        def body(carry):
            x, _x_prev, i = carry
            x_new = _F_dispatch(F_pure, x, consts)
            return (x_new, x, i + jnp.int32(1))

        x_star, _, n_iters = jax.lax.while_loop(
            cond,
            body,
            (x0, x0 + jnp.array(1.0, dtype=x0.dtype), jnp.int32(0)),
        )
        return x_star, n_iters

    if acceleration == "aitken":
        # Lazy import to keep this module's load cheap.
        from maddening.core.coupling.acceleration import (  # noqa: PLC0415
            aitken_relaxation,
        )

        n_dof = x0.shape[0]
        dtype = x0.dtype

        # Carry: (x_cur, x_prev, i, omega, prev_r).  ``x_prev`` lets
        # the cond_fun check ||F(x_cur) - x_cur|| via the raw residual
        # bookkeeping each iter rather than re-evaluating F just to
        # decide on termination.  We use the same "first iter always
        # runs" guard as the no-accel path so a single F-evaluation
        # always happens (matches fori-Aitken semantics).
        def cond(carry):
            x, x_prev, i, _omega, _prev_r = carry
            not_converged = jnp.linalg.norm(x - x_prev) > tol
            not_maxed = i < max_iter
            first = i == jnp.int32(0)
            return jnp.logical_or(first, jnp.logical_and(not_converged, not_maxed))

        def body(carry):
            x_cur, _x_prev, i, omega, prev_r = carry
            # One raw Gauss-Seidel pass.
            x_raw = _F_dispatch(F_pure, x_cur, consts)
            # Aitken-relaxed update: x_rel = x_cur + new_omega * (x_raw - x_cur)
            x_rel, new_omega, cur_r = aitken_relaxation(
                x_cur, x_raw, prev_r, omega,
            )
            return (x_rel, x_cur, i + jnp.int32(1), new_omega, cur_r)

        init_omega = jnp.array(1.0, dtype=dtype)
        init_prev_r = jnp.zeros(n_dof, dtype=dtype)
        # Seed x_prev as x0 + 1 so the first cond evaluation lets the
        # loop body run at least once (matches the no-accel path).
        init_carry = (
            x0,
            x0 + jnp.array(1.0, dtype=dtype),
            jnp.int32(0),
            init_omega,
            init_prev_r,
        )
        x_star, _, n_iters, _, _ = jax.lax.while_loop(cond, body, init_carry)
        return x_star, n_iters

    if acceleration == "iqn-imvj":
        # Interface quasi-Newton with inverse multi-vector Jacobian.
        # Carry: (x_cur, x_prev, i, V, W, prev_r).
        #
        # The math (per iteration i >= 1):
        #   r_cur  = F(x_cur) - x_cur                 # current residual
        #   V[:, i-1] = x_cur - x_prev                # input diff column
        #   W[:, i-1] = r_cur - prev_r                # residual diff column
        #   solve  W c ≈ r_cur                        # secant LS:
        #                                              c = (W^T W)^{-1} W^T r_cur
        #   dx_qn = -V c                              # Δx_QN ≈ -J_R^{-1} r_cur
        #                                              where J_R^{-1} ≈ V c / r_cur
        #   x_new = x_cur + dx_qn = x_cur - V c
        # At i == 0 there are no columns yet: do a bare Gauss-Seidel step.
        #
        # V/W are preallocated to ``(n_dof, max_cols)`` zeros at iter 0;
        # while_loop requires fixed shapes, so columns past ``i-1`` stay
        # zero.  jnp.linalg.lstsq is robust to the resulting rank deficit.
        #
        # **Per-step only.**  V/W reset to zeros each timestep — no
        # cross-timestep warm-start (deferred follow-up; the per-step
        # variant captures within-step convergence benefit and the
        # warm-start machinery is mutable-state coupled to CouplingGroup
        # which is JAX-trace-incompatible without special care).
        n_dof = x0.shape[0]
        dtype = x0.dtype
        max_cols = max(max_iter - 1, 1)

        def cond(carry):
            # Convergence uses the *residual* ``prev_r = F(x_prev) - x_prev``
            # rather than the step size ``||x - x_prev||``.  For Gauss-Seidel
            # / Aitken the two are equivalent (the step is the residual), but
            # for IMVJ the quasi-Newton step can be tiny even when the residual
            # is still large (e.g. ill-conditioned early-iter LS solve), which
            # would otherwise produce a spurious "converged" exit.  We force
            # at least one iter via the ``first`` flag, so the iter-0 zero
            # ``prev_r`` does not abort the loop.
            _x, _x_prev, i, _V, _W, prev_r = carry
            not_converged = jnp.linalg.norm(prev_r) > tol
            not_maxed = i < max_iter
            first = i == jnp.int32(0)
            return jnp.logical_or(first, jnp.logical_and(not_converged, not_maxed))

        def _gs_step(args):
            # First iter (i==0): pure Gauss-Seidel.  Just return F(x_cur).
            x_cur, _x_prev, _i, V, W, _prev_r, x_new, r_cur = args
            return x_new, V, W, r_cur

        def _imvj_step(args):
            # i >= 1: write column (i-1) into V/W, lstsq, apply dx_qn.
            x_cur, x_prev, i, V, W, prev_r, _x_new, r_cur = args
            delta_x = x_cur - x_prev
            delta_r = r_cur - prev_r
            # Masked write: column index = i-1.  ``.at[:, i-1].set`` is a
            # dynamic-index slice update, fine inside while_loop.
            col_idx = i - jnp.int32(1)
            V_new = V.at[:, col_idx].set(delta_x)
            W_new = W.at[:, col_idx].set(delta_r)
            # Zero out columns >= i (active count = i).  Cleaner than
            # trusting lstsq to handle the leftover zeros — explicit mask
            # makes the rank deficit visible in W_masked.
            col_mask = jnp.arange(max_cols) < i
            V_masked = V_new * col_mask[None, :]
            W_masked = W_new * col_mask[None, :]
            # Solve  W^T W c = W^T r_cur  via lstsq.  rcond keeps the
            # solve well-defined when W still has only a handful of
            # populated columns and the rest are zero (rank-deficient).
            c, _resid, _rank, _sv = jnp.linalg.lstsq(
                W_masked, r_cur, rcond=1e-10,
            )
            # Standard IMVJ update (Degroote 2008):  x_{k+1} = x_k - V c
            # where c solves the secant LS  W c ≈ r_cur.  This implies
            # J_R^{-1} r_cur ≈ V c, so Δx_QN = -V c is the QN step.
            dx_qn = -(V_masked @ c)
            x_qn = x_cur + dx_qn
            # Guard against blow-up: if the QN step produced NaN/inf,
            # fall back to a Gauss-Seidel step.  Cheap insurance against
            # ill-conditioned early-iter lstsq edge cases.
            qn_finite = jnp.all(jnp.isfinite(x_qn))
            x_out = jnp.where(qn_finite, x_qn, x_cur + r_cur)
            return x_out, V_new, W_new, r_cur

        def body(carry):
            x_cur, _x_prev, i, V, W, prev_r = carry
            x_new = _F_dispatch(F_pure, x_cur, consts)
            r_cur = x_new - x_cur
            args = (x_cur, _x_prev, i, V, W, prev_r, x_new, r_cur)
            x_out, V_out, W_out, r_out = jax.lax.cond(
                i == jnp.int32(0), _gs_step, _imvj_step, args,
            )
            return (x_out, x_cur, i + jnp.int32(1), V_out, W_out, r_out)

        init_V = jnp.zeros((n_dof, max_cols), dtype=dtype)
        init_W = jnp.zeros((n_dof, max_cols), dtype=dtype)
        init_prev_r = jnp.zeros(n_dof, dtype=dtype)
        init_carry = (
            x0,
            x0 + jnp.array(1.0, dtype=dtype),
            jnp.int32(0),
            init_V,
            init_W,
            init_prev_r,
        )
        x_star, _, n_iters, _, _, _ = jax.lax.while_loop(
            cond, body, init_carry,
        )
        return x_star, n_iters

    raise ValueError(
        f"_ift_fixed_point_fwd_impl: unsupported acceleration={acceleration!r}; "
        "supported values are 'none', 'aitken', and 'iqn-imvj'."
    )


def _ift_solve_impl(
    F_pure, x0, consts, tol, max_iter, acceleration, linear_solver
):
    """Returns ``x_star`` such that ``x_star = F_pure(x_star, *consts)``.

    Differentiates via the implicit function theorem:
        ``dx*/d(consts) = (I - dF/dx)^{-1} dF/d(consts)``
    evaluated at the fixed point.  ``x0`` itself receives a zero
    cotangent (the fixed point is invariant under the initial guess
    in the converged limit).

    ``acceleration`` is a static Python string — see
    ``_ift_fixed_point_fwd_impl`` for supported values.  It controls
    only the forward iterator; the backward is identical for all
    values because the IFT adjoint depends on ``F`` at ``x*``, not on
    the path taken to reach ``x*``.

    ``linear_solver`` is a static Python string — ``"gmres"`` (default),
    ``"bicgstab"``, or ``"dense"`` — selecting the backward adjoint
    solver.  See ``_ift_solve_bwd`` for the dispatch details.
    """
    x_star, _ = _ift_fixed_point_fwd_impl(
        F_pure, x0, consts, tol, max_iter, acceleration
    )
    return x_star


def _ift_solve_fwd(
    F_pure, x0, consts, tol, max_iter, acceleration, linear_solver
):
    # fwd has the SAME signature as the original function under
    # jax.custom_vjp's nondiff_argnums convention.  Only bwd is
    # rearranged (nondiff first, then residual, then output cotangent).
    x_star, _ = _ift_fixed_point_fwd_impl(
        F_pure, x0, consts, tol, max_iter, acceleration
    )
    return x_star, (x_star, consts)


def _ift_solve_bwd(
    F_pure, tol, max_iter, acceleration, linear_solver, residual, g
):
    del tol, max_iter, acceleration  # backward is acceleration-agnostic
    x_star, consts = residual
    # Linear system to solve:  (I - dF/dx)^T u = g.
    #
    # The matrix-vector product v -> (I - dF/dx)^T v is exactly
    # ``v - J^T v``, where ``J^T v`` is the vjp of the one-iteration
    # function ``F`` at ``x_star`` applied to ``v``.  This is matrix
    # free — no jacobian is ever materialized — so memory is O(N) and
    # compute per matvec is one F-vjp.  We hand the matvec to lineax
    # as a ``FunctionLinearOperator`` and let its GMRES (or BiCGStab)
    # iterate.
    #
    # Backends, dispatched by ``linear_solver`` plus the
    # ``MADDENING_IFT_DENSE_SOLVE`` env var (env var wins for triage):
    #
    # * ``"gmres"`` (default) — lineax GMRES.  Safe non-symmetric solver.
    # * ``"dense"`` — historical ``jacrev + jnp.linalg.solve``.  O(N^2)
    #   memory, O(N^3) compute.  Kept as a swap-in triage fallback and
    #   promoted to a first-class config option.
    #
    # * ``"bicgstab"`` — lineax BiCGStab.  *Disabled at the
    #   CouplingGroup field level* in lineax 0.0.7: BiCGStab returns
    #   NaN when driving a ``FunctionLinearOperator`` (the matrix-free
    #   shape this backward uses) — confirmed on a well-conditioned
    #   ``0.5*I`` test, so this is a lineax-side issue, not a property
    #   of MADDENING's coupling Jacobian.  The dispatch arm is left in
    #   place so a future lineax fix can re-enable it by widening the
    #   ``linear_solver`` Literal on CouplingGroup; users who want to
    #   try it can still construct CouplingGroup with
    #   ``linear_solver="bicgstab"`` (the runtime accepts the string).
    _, vjp_fn = jax.vjp(lambda xx: _F_dispatch(F_pure, xx, consts), x_star)

    force_dense = os.environ.get("MADDENING_IFT_DENSE_SOLVE") == "1"
    effective_solver = "dense" if force_dense else linear_solver

    if effective_solver == "dense":
        J = jax.jacrev(lambda xx: _F_dispatch(F_pure, xx, consts))(x_star)
        n = x_star.shape[0]
        A = jnp.eye(n, dtype=x_star.dtype) - J
        u = jnp.linalg.solve(A.T, g)
    else:
        # Lazy import — keeps lineax (and its equinox/optax transitive
        # deps) out of module load time.  Only callers who opt into
        # ``solver='ift'`` pay this import cost.
        import lineax as lx  # noqa: PLC0415  (lazy by design)

        def _matvec(v):
            (Jt_v,) = vjp_fn(v)
            return v - Jt_v

        op = lx.FunctionLinearOperator(
            _matvec, jax.eval_shape(lambda: g)
        )
        n = x_star.shape[0]

        if effective_solver == "bicgstab":
            # BiCGStab has no ``restart`` parameter (it operates on a
            # fixed three-vector recurrence rather than building a
            # Krylov subspace).  ``max_steps`` only needs to bound the
            # outer iteration count.
            max_steps = max(4 * n, 200)
            solver = lx.BiCGStab(
                rtol=1e-6, atol=1e-8, max_steps=max_steps,
            )
        elif effective_solver == "gmres":
            # (I - dF/dx)^T is in general non-symmetric; GMRES is the
            # safe default.  rtol/atol are matched to the float32
            # regime the surrounding code uses.
            #
            # *** GMRES restart gotcha ***
            #
            # ``restart`` directly bounds the dim of the Krylov subspace
            # GMRES builds.  Lineax's default is 20.  For coupling
            # groups whose flat state is larger than 20 floats (any
            # chain of >=10 two-DOF nodes — common!), the
            # default-20 GMRES silently converges to a *low-rank
            # approximation* of the adjoint solve.  It looks fine
            # (converged=True, residual small in the projected
            # subspace) but the returned ``u`` lies in a 20-D
            # subspace of an N-D problem, so the resulting gradient
            # is structurally wrong — *not* a near-correct answer
            # with extra noise, but a different gradient.
            #
            # We set restart = min(N, 50) so small problems stay cheap
            # while N>=50 problems still see a meaningful subspace,
            # and bump ``max_steps`` to give the algorithm headroom
            # for several restart cycles.  Do not regress this without
            # bumping the restart cap in lockstep — see
            # tests/core/test_coupling_ift_lineax.py::
            # test_gmres_restart_too_small_silently_corrupts_gradient
            # for the regression guard.
            restart = min(n, 50)
            max_steps = max(4 * restart, 100)
            solver = lx.GMRES(
                rtol=1e-6,
                atol=1e-8,
                restart=restart,
                max_steps=max_steps,
            )
        else:
            raise ValueError(
                f"_ift_solve_bwd: unsupported linear_solver="
                f"{linear_solver!r}; expected one of "
                f"'gmres', 'bicgstab', 'dense'."
            )

        result = lx.linear_solve(op, g, solver=solver)
        u = result.value
    # consts cotangent: dF/d(consts)^T @ u, via vjp wrt consts only.
    _, vjp_fn_c = jax.vjp(lambda cc: _F_dispatch(F_pure, x_star, cc), consts)
    (consts_bar,) = vjp_fn_c(u)
    # x0 receives zero cotangent.
    x0_bar = jnp.zeros_like(x_star)
    return (x0_bar, consts_bar)


# nondiff_argnums: 0=F_pure (callable), 3=tol (static float),
#                  4=max_iter (static int), 5=acceleration (static str),
#                  6=linear_solver (static str).
_ift_solve = jax.custom_vjp(
    _ift_solve_impl, nondiff_argnums=(0, 3, 4, 5, 6)
)
_ift_solve.defvjp(_ift_solve_fwd, _ift_solve_bwd)


# ------------------------------------------------------------------
# Observer event names
# ------------------------------------------------------------------
EVENT_NODE_ADDED = "node_added"
EVENT_NODE_REMOVED = "node_removed"
EVENT_EDGE_ADDED = "edge_added"
EVENT_EDGE_REMOVED = "edge_removed"
EVENT_COMPILED = "compiled"
EVENT_STEP = "step"


_EMPTY_EXTERNAL_INPUTS: dict[str, dict] = {}

# Key for internal multi-rate metadata in the full state dict.
_META_KEY = "_meta"


# ------------------------------------------------------------------
# Floating-point-tolerant GCD
# ------------------------------------------------------------------

def _float_gcd(a: float, b: float, tol: float = 1e-9) -> float:
    """GCD of two positive floats using Euclidean algorithm with tolerance."""
    if a < b:
        a, b = b, a
    while b > tol:
        a, b = b, a % b
    return a


def _multi_gcd(values: Sequence[float], tol: float = 1e-9) -> float:
    """GCD of multiple positive floats."""
    result = values[0]
    for v in values[1:]:
        result = _float_gcd(result, v, tol)
    return result


def _apply_interface_overrides(node_state, pre_state, boundary_inputs, dt,
                               node_obj, coupled_bi_names=None):
    """Correct interface DOFs after update to undo internal BC enforcement.

    Nodes like HeatNode enforce Dirichlet BCs by overwriting boundary
    cells after the FD update.  When those BCs come from coupling,
    the overwrite destroys the physically meaningful stencil-computed
    value.  This function asks the node to recompute those values via
    ``compute_interface_correction``.

    Only boundary inputs that come from coupling edges are corrected.
    External inputs and non-coupling edges are left as-is (standard
    Dirichlet enforcement is correct for those).

    Parameters
    ----------
    node_state : dict
        The node's state dict after ``update()`` was called.
    pre_state : dict
        The node's state dict **before** ``update()`` was called.
    boundary_inputs : dict
        The boundary inputs that were passed to ``update()``.
    dt : float
        The timestep used for the update.
    node_obj : SimulationNode
        The node descriptor.
    coupled_bi_names : set or None
        Boundary input names that come from coupling edges.
        Only these are eligible for interface correction.
        If None, all boundary inputs are eligible (backward compat).

    Returns
    -------
    dict
        The (possibly modified) node state.
    """
    iface = node_obj.interface_dof_indices()
    if not iface:
        return node_state
    # Filter boundary inputs to only coupled ones
    if coupled_bi_names is not None:
        filtered_bi = {k: v for k, v in boundary_inputs.items()
                       if k in coupled_bi_names}
    else:
        filtered_bi = boundary_inputs
    if not filtered_bi:
        return node_state
    corrections = node_obj.compute_interface_correction(
        pre_state, filtered_bi, dt
    )
    if not corrections:
        return node_state
    result = {**node_state}
    for field, idx_val_list in corrections.items():
        arr = result[field]
        for idx, val in idx_val_list:
            arr = arr.at[idx].set(val)
        result[field] = arr
    return result


def _run_coupled_block_impl(
    group, group_schedule, new_state, full_state, external_inputs,
    runtime_dt, *, nodes, edges_by_target, ext_by_target,
    back_edge_set, has_external, all_edges,
    multigpu_device_map=None,
):
    """Execute a coupling group with iterative fixed-point iteration.

    Supports Gauss-Seidel and Jacobi iteration modes, multiple
    convergence norms (L2, mixed, interface), acceleration methods
    (Aitken, fixed relaxation, IQN-ILS, IQN-IMVJ), additive edges,
    flux-based coupling, subcycling with linear/quadratic/constant
    interpolation, and waveform relaxation.

    This is the shared implementation used by both ``_build_step_fn``
    and ``_build_dt_step_fn``.
    """
    from maddening.core.coupling.acceleration import (
        aitken_relaxation,
        coupling_residual_interface,
        coupling_residual_l2,
        coupling_residual_mixed,
        fixed_relaxation,
        flatten_coupled_state,
        iqn_ils_update,
        unflatten_coupled_state,
    )

    max_iters = group.max_iterations
    group_node_names = list(group_schedule)
    group_node_set = set(group_node_names)
    use_mixed_norm = group.convergence_norm == "mixed"
    use_interface_norm = group.convergence_norm == "interface"
    use_acceleration = group.acceleration != "none"
    use_jacobi = group.iteration_mode == "jacobi"

    # Edges internal to this group are forced forward
    group_internal = set()
    group_internal_list = []
    for edge in all_edges:
        if edge.source_node in group.nodes and edge.target_node in group.nodes:
            group_internal.add(edge)
            group_internal_list.append(edge)

    # Precompute which boundary inputs come from coupling (intra-group) edges
    # per target node -- only these get interface correction
    coupled_bi_names_by_node: dict[str, set] = {}
    for edge in group_internal_list:
        coupled_bi_names_by_node.setdefault(
            edge.target_node, set()
        ).add(edge.target_field)

    # Auto-detect interface fields for IQN acceleration
    if group.acceleration in ("iqn-ils", "iqn-imvj"):
        if group.accelerated_fields is not None:
            accel_fields = group.accelerated_fields
        else:
            interface_fields: dict[str, set] = {}
            for edge in group_internal_list:
                interface_fields.setdefault(
                    edge.source_node, set()
                ).add(edge.source_field)
            if interface_fields:
                accel_fields = {
                    nn: tuple(sorted(fields))
                    for nn, fields in interface_fields.items()
                }
            else:
                accel_fields = None
    else:
        accel_fields = None

    # Detect which nodes produce flux fields
    from maddening.core.node import SimulationNode as _SimBase
    flux_producing_nodes = set()
    for nn in group_node_names:
        node_obj = nodes[nn].node
        if type(node_obj).compute_boundary_fluxes is not _SimBase.compute_boundary_fluxes:
            flux_producing_nodes.add(nn)
    # Also check nodes outside the group that feed edges into the group
    for edge in all_edges:
        src_nn = edge.source_node
        if edge.target_node in group_node_set and src_nn not in group_node_set:
            if src_nn in nodes:
                node_obj = nodes[src_nn].node
                if type(node_obj).compute_boundary_fluxes is not _SimBase.compute_boundary_fluxes:
                    flux_producing_nodes.add(src_nn)

    # Check if any edge references a flux field (not in state)
    has_flux_edges = False
    for edge in all_edges:
        if edge.target_node in group_node_set or edge.source_node in group_node_set:
            src_nn = edge.source_node
            if src_nn in nodes:
                src_fields = set(new_state.get(src_nn, {}).keys())
                if edge.source_field not in src_fields and src_nn in flux_producing_nodes:
                    has_flux_edges = True
                    break

    # Save the initial state for each node at the beginning of
    # the timestep -- this is what we always integrate FROM.
    initial_node_states = {nn: new_state[nn] for nn in group_node_names}

    def _get_dt(nn):
        spec = nodes[nn]
        return runtime_dt if runtime_dt is not None else spec.timestep

    def _resolve_value(edge, src_state, flux_s):
        """Get value from state or flux dict."""
        src_nn = edge.source_node
        src_dict = src_state.get(src_nn, {})
        if edge.source_field in src_dict:
            return src_dict[edge.source_field]
        if flux_s and src_nn in flux_s and edge.source_field in flux_s[src_nn]:
            return flux_s[src_nn][edge.source_field]
        # Fall back (will KeyError if truly missing)
        return src_state[src_nn][edge.source_field]

    def _resolve_boundary(nn, s, flux_s=None):
        """Resolve boundary inputs for node nn from state s."""
        boundary_inputs: dict[str, Any] = {}
        for edge in edges_by_target[nn]:
            if edge in back_edge_set and edge not in group_internal:
                src_state = full_state
            else:
                src_state = s
            value = _resolve_value(edge, src_state, flux_s)
            if edge.transform is not None:
                value = edge.transform(value)
            if edge.additive and edge.target_field in boundary_inputs:
                boundary_inputs[edge.target_field] = (
                    boundary_inputs[edge.target_field] + value
                )
            else:
                boundary_inputs[edge.target_field] = value

        if nn in has_external:
            node_ext = external_inputs.get(nn, {})
            for ei in ext_by_target[nn]:
                if ei.target_field in node_ext:
                    boundary_inputs[ei.target_field] = node_ext[ei.target_field]
        return boundary_inputs

    # Compute subcycling rate dividers if needed.
    use_subcycling = group.subcycling
    if use_subcycling:
        group_timesteps_list = sorted(
            {nodes[nn].timestep for nn in group_node_names}
        )
        if len(group_timesteps_list) > 1:
            group_macro_dt = max(group_timesteps_list)
            group_dividers = {
                nn: max(round(group_macro_dt / nodes[nn].timestep), 1)
                for nn in group_node_names
            }
        else:
            group_dividers = {nn: 1 for nn in group_node_names}
            use_subcycling = False  # uniform timestep, no subcycling needed
        use_linear_interp = group.boundary_interpolation == "linear"
        use_quadratic_interp = group.boundary_interpolation == "quadratic"

    def _resolve_boundary_interpolated(nn, s_prev, s_cur, alpha,
                                        flux_s=None, s_prev_prev=None):
        """Resolve boundary inputs with time interpolation.

        alpha=0 means start (s_prev values), alpha=1 means end (s_cur).
        Only interpolates edges that are internal to the coupling group.
        """
        boundary_inputs: dict[str, Any] = {}
        for edge in edges_by_target[nn]:
            if edge in back_edge_set and edge not in group_internal:
                src_state = full_state
                value = _resolve_value(edge, src_state, flux_s)
            elif edge in group_internal:
                if use_quadratic_interp and s_prev_prev is not None:
                    # Quadratic Lagrange through 3 points:
                    # (0, v_pp), (0.5, v_prev), (1, v_cur)
                    v_pp = s_prev_prev[edge.source_node][edge.source_field]
                    v_prev = s_prev[edge.source_node][edge.source_field]
                    v_cur = s_cur[edge.source_node][edge.source_field]
                    alpha_sq = alpha * alpha
                    value = (
                        (1.0 - 3.0 * alpha + 2.0 * alpha_sq) * v_pp
                        + (4.0 * alpha - 4.0 * alpha_sq) * v_prev
                        + (-alpha + 2.0 * alpha_sq) * v_cur
                    )
                else:
                    # Linear interpolation
                    v_prev = s_prev[edge.source_node][edge.source_field]
                    v_cur = s_cur[edge.source_node][edge.source_field]
                    value = jax.tree.map(
                        lambda a, b: a + alpha * (b - a), v_prev, v_cur
                    )
            else:
                value = _resolve_value(edge, s_cur, flux_s)
            if edge.transform is not None:
                value = edge.transform(value)
            if edge.additive and edge.target_field in boundary_inputs:
                boundary_inputs[edge.target_field] = (
                    boundary_inputs[edge.target_field] + value
                )
            else:
                boundary_inputs[edge.target_field] = value

        if nn in has_external:
            node_ext = external_inputs.get(nn, {})
            for ei in ext_by_target[nn]:
                if ei.target_field in node_ext:
                    boundary_inputs[ei.target_field] = node_ext[ei.target_field]
        return boundary_inputs

    def _run_substeps(nn, n_substeps, sub_dt, s_prev, s_cur,
                       flux_s=None, s_prev_prev=None):
        """Run n_substeps sub-steps for a fast node using lax.scan."""
        init_sub_state = initial_node_states[nn]

        def substep_body(sub_state, sub_idx):
            alpha = (sub_idx + 1.0) / n_substeps
            if use_subcycling and (use_linear_interp or use_quadratic_interp):
                bi = _resolve_boundary_interpolated(
                    nn, s_prev, s_cur, alpha,
                    flux_s=flux_s, s_prev_prev=s_prev_prev,
                )
            else:
                # constant: use end-of-step values
                bi = _resolve_boundary(nn, s_cur, flux_s)
            new_sub = nodes[nn].update_fn(sub_state, bi, sub_dt)
            new_sub = _apply_interface_overrides(
                new_sub, sub_state, bi, sub_dt, nodes[nn].node,
                coupled_bi_names=coupled_bi_names_by_node.get(nn),
            )
            return new_sub, None

        final_sub, _ = jax.lax.scan(
            substep_body, init_sub_state, jnp.arange(n_substeps)
        )
        return final_sub

    def one_pass_gs(latest_results):
        """Gauss-Seidel: sequential updates, each sees latest results."""
        s = {k: v for k, v in latest_results.items()}
        flux_s: dict[str, dict] = {}
        for nn in group_node_names:
            if use_subcycling and group_dividers[nn] > 1:
                n_sub = group_dividers[nn]
                s[nn] = _run_substeps(
                    nn, n_sub, _get_dt(nn),
                    latest_results, s, flux_s=flux_s,
                )
                # Interface overrides already applied per sub-step
            else:
                bi = _resolve_boundary(nn, s, flux_s)
                pre = initial_node_states[nn]
                s[nn] = nodes[nn].update_fn(pre, bi, _get_dt(nn))
                s[nn] = _apply_interface_overrides(
                    s[nn], pre, bi, _get_dt(nn), nodes[nn].node,
                    coupled_bi_names=coupled_bi_names_by_node.get(nn),
                )
            # Compute fluxes for this node
            if nn in flux_producing_nodes:
                bi_for_flux = _resolve_boundary(nn, s, flux_s)
                flux_s[nn] = nodes[nn].node.compute_boundary_fluxes(
                    s[nn], bi_for_flux, _get_dt(nn)
                )
        return s

    def one_pass_jacobi(latest_results):
        """Jacobi: all nodes read from frozen previous-iteration state."""
        # Pre-compute fluxes from previous iteration state
        flux_s: dict[str, dict] = {}
        if has_flux_edges:
            for nn in group_node_names:
                if nn in flux_producing_nodes:
                    bi = _resolve_boundary(nn, latest_results)
                    flux_s[nn] = nodes[nn].node.compute_boundary_fluxes(
                        latest_results[nn], bi, _get_dt(nn)
                    )

        results = {}
        for nn in group_node_names:
            if use_subcycling and group_dividers[nn] > 1:
                n_sub = group_dividers[nn]
                results[nn] = _run_substeps(
                    nn, n_sub, _get_dt(nn),
                    latest_results, latest_results, flux_s=flux_s,
                )
                # Interface overrides already applied per sub-step
            else:
                # Optionally place computation on assigned device
                bi = _resolve_boundary(nn, latest_results, flux_s)
                pre = initial_node_states[nn]
                if multigpu_device_map is not None and nn in multigpu_device_map:
                    dev_idx = multigpu_device_map[nn]
                    devices = jax.devices()
                    if dev_idx < len(devices):
                        device = devices[dev_idx]
                        pre = jax.device_put(pre, device)
                        bi = jax.tree.map(
                            lambda x: jax.device_put(x, device), bi,
                        )
                results[nn] = nodes[nn].update_fn(pre, bi, _get_dt(nn))
                results[nn] = _apply_interface_overrides(
                    results[nn], pre, bi, _get_dt(nn), nodes[nn].node,
                    coupled_bi_names=coupled_bi_names_by_node.get(nn),
                )
        s = {k: v for k, v in latest_results.items()}
        for nn in group_node_names:
            s[nn] = results[nn]
        return s

    one_pass = one_pass_jacobi if use_jacobi else one_pass_gs

    def _compute_residual(s_new, s_old):
        if use_interface_norm:
            return coupling_residual_interface(
                s_new, s_old, group_internal_list,
                group.atol, group.rtol,
            )
        if use_mixed_norm:
            return coupling_residual_mixed(
                s_new, s_old, group_node_names,
                group.atol, group.rtol,
            )
        return coupling_residual_l2(s_new, s_old, group_node_names)

    # Convergence threshold depends on norm type
    conv_threshold = (
        jnp.array(1.0) if (use_mixed_norm or use_interface_norm)
        else jnp.array(group.tolerance)
    )

    # Helper: flatten/unflatten with optional auto-detected fields
    def _flatten(s):
        return flatten_coupled_state(s, group_node_names, fields=accel_fields)

    def _unflatten(flat, template):
        return unflatten_coupled_state(
            flat, template, group_node_names, fields=accel_fields
        )

    def _build_accel_state(s_raw, s_partial):
        """Merge accelerated interface fields with raw non-interface fields."""
        if accel_fields is None:
            return s_partial
        result = {}
        for nn in group_node_names:
            result[nn] = {}
            af = accel_fields.get(nn, ())
            for fld in s_raw[nn]:
                if fld in af and nn in s_partial:
                    result[nn][fld] = s_partial[nn][fld]
                else:
                    result[nn][fld] = s_raw[nn][fld]
        return result

    # ------------------------------------------------------------------
    # Waveform relaxation wrapper
    # ------------------------------------------------------------------
    n_waveform = group.waveform_iterations if use_subcycling else 1

    def _run_coupling_inner(new_state_inner):
        """Run the core coupling iteration (may be called multiple times
        for waveform relaxation).

        ``initial_node_states`` (the beginning-of-timestep state that
        nodes integrate FROM) is never changed by waveform re-runs.
        Only ``new_state_inner`` (used for boundary resolution) is
        updated between waveform passes.
        """

        # Run first iteration
        state_after_first = one_pass(new_state_inner)

        if max_iters <= 1:
            r = {k: v for k, v in new_state_inner.items()}
            for nn in group_node_names:
                r[nn] = state_after_first[nn]
            return r, None, None

        # Determine n_dof for acceleration
        if use_acceleration:
            n_dof_flat = _flatten(state_after_first)
            n_dof = n_dof_flat.shape[0]

        track_diag = group.diagnostics
        first_r = _compute_residual(state_after_first, new_state_inner)

        # Helper: build the merge step
        def _merge(s_cur, s_result, new_converged):
            s_merged = {}
            for k_s in s_cur:
                if k_s in group_node_set:
                    s_merged[k_s] = jax.tree.map(
                        lambda n, o: jnp.where(new_converged, o, n),
                        s_result[k_s], s_cur[k_s],
                    )
                else:
                    s_merged[k_s] = s_cur[k_s]
            return s_merged

        def _run_ift_forward(template_state, acceleration):
            """Run the IFT forward (while_loop) and return the full state.

            ``template_state`` is the post-first-pass full state dict
            (``state_after_first``); the IFT solver operates on its
            flattened coupling subset while preserving the embedding
            into the full state for ``one_pass``.  ``acceleration`` is
            ``"none"`` or ``"aitken"`` and selects the while_loop body
            wrapper; the IFT backward is unchanged across acceleration
            modes (it is intrinsic to ``F`` at ``x*``).
            """
            # Operate on the flattened group-state vector ``x``.  We
            # need a top-level ``F_pure(x, *consts)`` so the
            # custom_vjp rule does not close over any tracer
            # (see JAX issue #2912 / optimistix's _is_global_function
            # assertion).  jax.closure_convert hoists any tracers
            # ``one_pass`` captures into an explicit ``consts``
            # pytree we then pass through ``_ift_solve``.
            #
            # **Embedded coupling groups** (group members read state
            # from non-group nodes): ``_unflatten`` only emits the
            # coupling subset, but ``one_pass`` needs the full state
            # dict so its boundary-resolution can look up the
            # outside nodes.  We embed the unflattened group state
            # into ``template_state`` (which is ``state_after_first``
            # and already carries every node), then pass the full
            # dict to ``one_pass``.  The outside-node entries in
            # ``template_state`` are tracers from the surrounding
            # jit trace — ``jax.closure_convert`` will hoist them
            # into ``consts`` automatically, so the IFT adjoint
            # propagates gradients back through them.
            # When ``accel_fields`` is non-None (IQN modes), ``_flatten``
            # / ``_unflatten`` operate on the *interface-field subset*
            # rather than the full per-node state.  We need to splice
            # those interface fields back into the full node dict before
            # calling ``one_pass`` (which expects every node to carry
            # all of its state), and again on the way out so ``_merge``
            # sees a result with the same pytree shape as
            # ``template_state``.
            def _splice_group(s_full, group_part):
                """Merge accelerated interface fields from ``group_part``
                into the corresponding nodes of ``s_full``; non-accel
                fields stay as they were in ``s_full``.
                """
                if accel_fields is None:
                    # Whole-node replacement (e.g. acceleration='none').
                    s_out = {k: v for k, v in s_full.items()}
                    for nn in group_node_names:
                        s_out[nn] = group_part[nn]
                    return s_out
                s_out = {k: v for k, v in s_full.items()}
                for nn in group_node_names:
                    merged = {fld: val for fld, val in s_full[nn].items()}
                    af = accel_fields.get(nn, ())
                    for fld in af:
                        if fld in group_part.get(nn, {}):
                            merged[fld] = group_part[nn][fld]
                    s_out[nn] = merged
                return s_out

            def _F_one_pass_flat(x_flat):
                group_part = _unflatten(x_flat, template_state)
                s = _splice_group(template_state, group_part)
                s_new = one_pass(s)
                # ``_flatten`` only emits the accelerated-field subset
                # (when accel_fields is set), so the round-trip is
                # well-defined regardless of the splicing above.
                return _flatten(s_new)

            x0_flat = _flatten(template_state)
            F_pure, consts_list = jax.closure_convert(
                _F_one_pass_flat, x0_flat
            )
            consts = tuple(consts_list)

            x_star_flat = _ift_solve(
                F_pure, x0_flat, consts,
                float(group.tolerance),
                int(max_iters),
                acceleration,
                str(getattr(group, "linear_solver", "gmres")),
            )
            # Reconstruct the full per-node state at the fixed point.
            # We need one more ``one_pass`` evaluation here for the
            # non-accelerated fields (e.g. velocity), because the IFT
            # solver only carries the accelerated-field subset through
            # its while_loop carry.  At ``x*`` the position is fixed, so
            # this final pass produces velocity consistent with the
            # converged position.
            group_part_final = _unflatten(x_star_flat, template_state)
            spliced = _splice_group(template_state, group_part_final)
            if accel_fields is not None:
                final_full = one_pass(spliced)
            else:
                final_full = spliced
            return _merge(template_state, final_full, jnp.array(False))

        # ---- Build fori_loop body based on acceleration + diagnostics ----

        if group.acceleration == "aitken":
            if track_diag:
                def body_fn(i, carry):
                    s_cur, converged, icount, fres, omega, prev_r = carry
                    s_raw = one_pass(s_cur)
                    residual = _compute_residual(s_raw, s_cur)
                    new_converged = converged | (residual <= conv_threshold)
                    x_old = _flatten(s_cur)
                    x_raw = _flatten(s_raw)
                    x_rel, new_omega, cur_r = aitken_relaxation(
                        x_old, x_raw, prev_r, omega
                    )
                    s_partial = _unflatten(x_rel, s_cur)
                    s_accel = _build_accel_state(s_raw, s_partial)
                    s_merged = _merge(s_cur, s_accel, new_converged)
                    new_count = icount + jnp.where(new_converged, 0.0, 1.0)
                    new_res = jnp.where(new_converged, fres, residual)
                    return s_merged, new_converged, new_count, new_res, new_omega, cur_r

                init_carry = (
                    state_after_first, jnp.array(False),
                    jnp.array(1.0), first_r,
                    jnp.array(1.0), jnp.zeros(n_dof),
                )
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]
                iter_count, final_res = final_carry[2], final_carry[3]
            else:
                use_ift_aitken = (
                    getattr(group, "solver", "fori") == "ift"
                    and not use_jacobi
                )
                if use_ift_aitken:
                    # ----- IFT + Aitken path.  See _run_ift_forward for
                    # the shared plumbing.  The forward while_loop wraps
                    # each ``F(x)`` call in Aitken delta-squared
                    # relaxation; the backward IFT adjoint uses the bare
                    # ``F`` (acceleration-agnostic at the fixed point).
                    final_state = _run_ift_forward(
                        state_after_first, "aitken",
                    )
                else:
                    def body_fn(i, carry):
                        s_cur, converged, omega, prev_r = carry
                        s_raw = one_pass(s_cur)
                        residual = _compute_residual(s_raw, s_cur)
                        new_converged = converged | (residual <= conv_threshold)
                        x_old = _flatten(s_cur)
                        x_raw = _flatten(s_raw)
                        x_rel, new_omega, cur_r = aitken_relaxation(
                            x_old, x_raw, prev_r, omega
                        )
                        s_partial = _unflatten(x_rel, s_cur)
                        s_accel = _build_accel_state(s_raw, s_partial)
                        s_merged = _merge(s_cur, s_accel, new_converged)
                        return s_merged, new_converged, new_omega, cur_r

                    init_carry = (
                        state_after_first, jnp.array(False),
                        jnp.array(1.0), jnp.zeros(n_dof),
                    )
                    final_carry = jax.lax.fori_loop(
                        1, max_iters, body_fn, init_carry
                    )
                    final_state = final_carry[0]

        elif group.acceleration in ("iqn-ils", "iqn-imvj"):
            # ----- IFT + IQN-IMVJ short-circuit -----
            #
            # When the user opts into ``solver='ift'`` with
            # ``acceleration='iqn-imvj'``, route into ``_run_ift_forward``
            # so the QN updates happen inside the IFT while_loop and the
            # backward goes through the acceleration-agnostic custom_vjp
            # rule (matrix-free lineax GMRES at the fixed point).
            #
            # **Per-step only** for this prototype.  The fori-loop branch
            # below still owns cross-timestep warm-start (V/W persisted
            # in ``_META_KEY``); the IFT path re-zeros V/W each step.
            #
            # Why not cross-timestep warm-start here yet?  Three coupled
            # changes are required and each has a sharp edge:
            #
            # 1. ``_ift_solve``'s custom_vjp must return ``(x_star, V, W)``
            #    rather than just ``x_star``, with zero cotangents on V/W
            #    in the backward (V/W are forward-only state).  Mechanical
            #    but touches the autodiff signature.
            # 2. ``_ift_fixed_point_fwd_impl`` for ``iqn-imvj`` must accept
            #    ``init_V``, ``init_W``, ``init_ncols`` as dynamic args
            #    and return the final V/W from the while_loop carry.
            # 3. The column-write logic in ``_imvj_step`` currently writes
            #    at column ``i - 1`` (zero-based, starting from the first
            #    body iter).  Warm-start means iteration 0 must write at
            #    column ``init_ncols`` and (when ``init_ncols + max_iter``
            #    exceeds ``max_cols``) cycle older columns out — the same
            #    shift-or-modulo convention the fori path's
            #    ``iqn_ils_update`` already implements.  This is a real
            #    change to the IMVJ math, not just plumbing.
            #
            # The blocking item is (3): the per-step IMVJ math here uses
            # a "write at column i-1" convention that is structurally
            # incompatible with warm-start.  Adopting the fori path's
            # shift-and-insert convention (insert at column 0, shift the
            # rest right) inside a while_loop body is doable but needs
            # its own correctness test against the fori-loop IMVJ output
            # before being trusted — and the IFT-IMVJ adjoint at the
            # fixed point is mathematically unaffected, so the gradient
            # parity test alone doesn't catch a column-write bug.
            #
            # Deferred to 0.4.x along with the IQN-ILS IFT extension.
            # The 0.3.x polish does not regress the per-step variant.
            use_ift_iqn_imvj = (
                group.acceleration == "iqn-imvj"
                and getattr(group, "solver", "fori") == "ift"
                and not use_jacobi
                and not track_diag
            )
            if use_ift_iqn_imvj:
                final_state = _run_ift_forward(
                    state_after_first, "iqn-imvj",
                )
                # No cross-timestep V/W persistence for the IFT variant.
                # The vw_data block below checks for the IFT short-circuit
                # and skips the META write in that case.
                final_V = None
                final_W = None
            else:
                max_cols = max(max_iters - 1, 1)

                # IQN-IMVJ: warm-start V/W from previous timestep
                if group.acceleration == "iqn-imvj":
                    group_key = "+".join(sorted(group.nodes))
                    meta = new_state_inner.get(_META_KEY, {})
                    stored_V = meta.get(
                        f"coupling_{group_key}_V",
                        jnp.zeros((n_dof, max_cols)),
                    )
                    stored_W = meta.get(
                        f"coupling_{group_key}_W",
                        jnp.zeros((n_dof, max_cols)),
                    )
                    n_reuse = min(group.jacobian_reuse, max_cols)
                    # Keep first n_reuse columns from previous timestep
                    reuse_mask = jnp.arange(max_cols) < n_reuse
                    init_V = stored_V * reuse_mask[None, :]
                    init_W = stored_W * reuse_mask[None, :]
                    init_ncols = jnp.int32(n_reuse)
                else:
                    init_V = jnp.zeros((n_dof, max_cols))
                    init_W = jnp.zeros((n_dof, max_cols))
                    init_ncols = jnp.int32(0)

                init_flat = _flatten(state_after_first)

                if track_diag:
                    def body_fn(i, carry):
                        (s_cur, converged, icount, fres,
                         V, W, nc, prev_r, prev_s, omega, prev_ra) = carry
                        s_raw = one_pass(s_cur)
                        residual = _compute_residual(s_raw, s_cur)
                        new_converged = converged | (residual <= conv_threshold)
                        x_old = _flatten(s_cur)
                        x_raw = _flatten(s_raw)
                        (x_new, nV, nW, nnc,
                         cur_r, cur_s, n_omega, cur_ra) = iqn_ils_update(
                            x_raw, x_old, prev_r, prev_s,
                            V, W, nc, omega, prev_ra,
                        )
                        s_partial = _unflatten(x_new, s_cur)
                        s_accel = _build_accel_state(s_raw, s_partial)
                        s_merged = _merge(s_cur, s_accel, new_converged)
                        new_count = icount + jnp.where(new_converged, 0.0, 1.0)
                        new_res = jnp.where(new_converged, fres, residual)
                        return (s_merged, new_converged, new_count, new_res,
                                nV, nW, nnc, cur_r, cur_s, n_omega, cur_ra)

                    init_carry = (
                        state_after_first, jnp.array(False),
                        jnp.array(1.0), first_r,
                        init_V, init_W, init_ncols,
                        jnp.zeros(n_dof), init_flat,
                        jnp.array(1.0), jnp.zeros(n_dof),
                    )
                    final_carry = jax.lax.fori_loop(
                        1, max_iters, body_fn, init_carry
                    )
                    final_state = final_carry[0]
                    iter_count, final_res = final_carry[2], final_carry[3]
                    final_V, final_W = final_carry[4], final_carry[5]
                else:
                    def body_fn(i, carry):
                        (s_cur, converged,
                         V, W, nc, prev_r, prev_s, omega, prev_ra) = carry
                        s_raw = one_pass(s_cur)
                        residual = _compute_residual(s_raw, s_cur)
                        new_converged = converged | (residual <= conv_threshold)
                        x_old = _flatten(s_cur)
                        x_raw = _flatten(s_raw)
                        (x_new, nV, nW, nnc,
                         cur_r, cur_s, n_omega, cur_ra) = iqn_ils_update(
                            x_raw, x_old, prev_r, prev_s,
                            V, W, nc, omega, prev_ra,
                        )
                        s_partial = _unflatten(x_new, s_cur)
                        s_accel = _build_accel_state(s_raw, s_partial)
                        s_merged = _merge(s_cur, s_accel, new_converged)
                        return (s_merged, new_converged,
                                nV, nW, nnc, cur_r, cur_s, n_omega, cur_ra)

                    init_carry = (
                        state_after_first, jnp.array(False),
                        init_V, init_W, init_ncols,
                        jnp.zeros(n_dof), init_flat,
                        jnp.array(1.0), jnp.zeros(n_dof),
                    )
                    final_carry = jax.lax.fori_loop(
                        1, max_iters, body_fn, init_carry
                    )
                    final_state = final_carry[0]
                    final_V, final_W = final_carry[2], final_carry[3]

        elif group.acceleration == "fixed":
            omega_val = group.relaxation

            if track_diag:
                def body_fn(i, carry):
                    s_cur, converged, icount, fres = carry
                    s_raw = one_pass(s_cur)
                    residual = _compute_residual(s_raw, s_cur)
                    new_converged = converged | (residual <= conv_threshold)
                    x_old = _flatten(s_cur)
                    x_raw = _flatten(s_raw)
                    x_rel = fixed_relaxation(x_old, x_raw, omega_val)
                    s_partial = _unflatten(x_rel, s_cur)
                    s_accel = _build_accel_state(s_raw, s_partial)
                    s_merged = _merge(s_cur, s_accel, new_converged)
                    new_count = icount + jnp.where(new_converged, 0.0, 1.0)
                    new_res = jnp.where(new_converged, fres, residual)
                    return s_merged, new_converged, new_count, new_res

                init_carry = (
                    state_after_first, jnp.array(False),
                    jnp.array(1.0), first_r,
                )
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]
                iter_count, final_res = final_carry[2], final_carry[3]
            else:
                def body_fn(i, carry):
                    s_cur, converged = carry
                    s_raw = one_pass(s_cur)
                    residual = _compute_residual(s_raw, s_cur)
                    new_converged = converged | (residual <= conv_threshold)
                    x_old = _flatten(s_cur)
                    x_raw = _flatten(s_raw)
                    x_rel = fixed_relaxation(x_old, x_raw, omega_val)
                    s_partial = _unflatten(x_rel, s_cur)
                    s_accel = _build_accel_state(s_raw, s_partial)
                    s_merged = _merge(s_cur, s_accel, new_converged)
                    return s_merged, new_converged

                init_carry = (state_after_first, jnp.array(False))
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]

        else:
            # No acceleration ("none")
            use_ift = (
                getattr(group, "solver", "fori") == "ift"
                and not use_jacobi
                and not track_diag
            )

            if use_ift:
                # ----- IFT path: while_loop forward + custom_vjp backward.
                # See ``_run_ift_forward`` below for the shared
                # closure_convert + ``_ift_solve`` plumbing; this
                # acceleration='none' branch is just the most common
                # entry point.
                final_state = _run_ift_forward(
                    state_after_first, "none",
                )
            elif track_diag:
                def body_fn(i, carry):
                    s_cur, converged, icount, fres = carry
                    s_new = one_pass(s_cur)
                    residual = _compute_residual(s_new, s_cur)
                    new_converged = converged | (residual <= conv_threshold)
                    s_merged = _merge(s_cur, s_new, new_converged)
                    new_count = icount + jnp.where(new_converged, 0.0, 1.0)
                    new_res = jnp.where(new_converged, fres, residual)
                    return s_merged, new_converged, new_count, new_res

                init_carry = (
                    state_after_first, jnp.array(False),
                    jnp.array(1.0), first_r,
                )
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]
                iter_count, final_res = final_carry[2], final_carry[3]
            else:
                def body_fn(i, carry):
                    s_cur, converged = carry
                    s_new = one_pass(s_cur)
                    residual = _compute_residual(s_new, s_cur)
                    new_converged = converged | (residual <= conv_threshold)
                    s_merged = _merge(s_cur, s_new, new_converged)
                    return s_merged, new_converged

                init_carry = (state_after_first, jnp.array(False))
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]

        # Merge coupled nodes back into the full state
        r = {k: v for k, v in new_state_inner.items()}
        for nn in group_node_names:
            r[nn] = final_state[nn]

        # Write diagnostics to _meta if requested
        diag_data = None
        if track_diag:
            diag_data = (iter_count, final_res)

        # Store V/W for IQN-IMVJ.  ``final_V`` / ``final_W`` are None
        # when the IFT-IMVJ short-circuit fired (per-step variant — no
        # cross-timestep warm-start, see comments at the IMVJ branch).
        vw_data = None
        if group.acceleration in ("iqn-ils", "iqn-imvj"):
            if final_V is not None and final_W is not None:
                vw_data = (final_V, final_W)

        return r, diag_data, vw_data

    # ------------------------------------------------------------------
    # Predictor: extrapolate initial guess from previous converged states
    # ------------------------------------------------------------------
    use_predictor = group.predictor != "none"
    group_key = "+".join(sorted(group.nodes))

    if use_predictor:
        meta = new_state.get(_META_KEY, {})
        pred_count = meta.get(
            f"coupling_{group_key}_pred_count", jnp.array(0, jnp.int32)
        )
        # Read stored converged flattened states
        n_hist = 3 if group.predictor == "quadratic" else 2
        pred_hist = []
        for pi in range(n_hist):
            pk = f"coupling_{group_key}_pred_{pi}"
            if pk in meta:
                pred_hist.append(meta[pk])

        if len(pred_hist) >= 2:
            # Apply extrapolation.  pred_0 is most recent, pred_1 is
            # one step before, pred_2 (if exists) is two steps before.
            x_n = pred_hist[0]    # most recent converged state
            x_nm1 = pred_hist[1]  # one before

            if group.predictor == "quadratic" and len(pred_hist) >= 3:
                x_nm2 = pred_hist[2]
                # Quadratic: x_pred = 3*x_n - 3*x_{n-1} + x_{n-2}
                has_enough = pred_count >= 3
                x_pred_q = 3.0 * x_n - 3.0 * x_nm1 + x_nm2
                # Linear fallback: x_pred = 2*x_n - x_{n-1}
                x_pred_l = 2.0 * x_n - x_nm1
                x_pred = jnp.where(has_enough, x_pred_q, x_pred_l)
            else:
                # Linear: x_pred = 2*x_n - x_{n-1}
                x_pred = 2.0 * x_n - x_nm1

            # Only apply if we have at least 2 stored states
            has_history = pred_count >= 2
            x_cur = flatten_coupled_state(new_state, group_node_names)
            x_use = jnp.where(has_history, x_pred, x_cur)

            # Unflatten and update new_state with predicted values
            predicted = unflatten_coupled_state(
                x_use, new_state, group_node_names
            )
            new_state = {k: v for k, v in new_state.items()}
            for nn in group_node_names:
                if nn in predicted:
                    new_state[nn] = predicted[nn]

    # ------------------------------------------------------------------
    # Run coupling (with waveform relaxation wrapper)
    # ------------------------------------------------------------------
    current_state = new_state
    diag_data = None
    vw_data = None

    for _wf in range(n_waveform):
        current_state, diag_data, vw_data = _run_coupling_inner(current_state)

    result = current_state

    # ------------------------------------------------------------------
    # Store predictor history in _meta
    # ------------------------------------------------------------------
    if use_predictor:
        converged_flat = flatten_coupled_state(result, group_node_names)
        result.setdefault(_META_KEY, {})
        meta_update = dict(result.get(_META_KEY, {}))

        n_hist = 3 if group.predictor == "quadratic" else 2
        # Shift history: pred_2 = old pred_1, pred_1 = old pred_0,
        # pred_0 = current converged
        for pi in range(n_hist - 1, 0, -1):
            prev_key = f"coupling_{group_key}_pred_{pi - 1}"
            cur_key = f"coupling_{group_key}_pred_{pi}"
            if prev_key in meta_update:
                meta_update[cur_key] = meta_update[prev_key]
        meta_update[f"coupling_{group_key}_pred_0"] = converged_flat

        # Increment counter (capped at n_hist)
        old_count = meta_update.get(
            f"coupling_{group_key}_pred_count", jnp.array(0, jnp.int32)
        )
        meta_update[f"coupling_{group_key}_pred_count"] = jnp.minimum(
            old_count + 1, n_hist
        )
        result[_META_KEY] = meta_update

    # Write diagnostics to _meta if requested
    if group.diagnostics and diag_data is not None:
        iter_count, final_res = diag_data
        result.setdefault(_META_KEY, {})
        result[_META_KEY] = {
            **result.get(_META_KEY, {}),
            f"coupling_{group_key}_iterations": jnp.array(
                iter_count, dtype=jnp.int32
            ),
            f"coupling_{group_key}_residual": final_res,
        }

    # Store V/W matrices for IQN-IMVJ Jacobian reuse
    if group.acceleration == "iqn-imvj" and vw_data is not None:
        final_V, final_W = vw_data
        result.setdefault(_META_KEY, {})
        result[_META_KEY] = {
            **result.get(_META_KEY, {}),
            f"coupling_{group_key}_V": final_V,
            f"coupling_{group_key}_W": final_W,
        }

    return result


@stability(StabilityLevel.STABLE)
class GraphManager:
    """Build, validate, compile and run a simulation graph.

    Supports multi-rate scheduling: nodes may have different timesteps.
    The graph steps at the *base timestep* (GCD of all node timesteps).
    Each node updates only on the sub-steps that are multiples of its
    own rate divider.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, _NodeSpec] = {}
        self._edges: list[EdgeSpec] = []
        self._state: dict[str, dict] = {}
        self._schedule: list[str] = []
        self._compiled_step: Optional[Callable] = None
        self._dirty: bool = True
        self._observers: list[Callable] = []
        self._back_edges: list[EdgeSpec] = []
        self._external_inputs: list[ExternalInputSpec] = []
        self._is_multirate: bool = False
        # v0.2 #3: snapshot of per-node static_data hashes captured at
        # ``compile()`` time.  Used by ``_check_static_data_dirty`` so a
        # node whose ``static_data`` changes after compile (typical
        # case: ``replace_node`` brings a different mesh) forces a
        # recompile on the next ``step()``.
        self._static_data_hashes: dict[str, int] = {}
        self._rate_dividers: dict[str, int] = {}
        self._coupling_groups: list[CouplingGroup] = []
        # Multi-GPU state (set by enable_multigpu)
        self._multigpu_mesh = None
        self._multigpu_device_map: Optional[dict[str, int]] = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, node: SimulationNode) -> None:
        """Register a node and initialise its state."""
        if node.name in self._nodes:
            raise ValueError(f"Node '{node.name}' already exists in the graph.")

        spec = _NodeSpec(
            node=node,
            update_fn=node.update,
            timestep=node.delta_t,
        )
        self._nodes[node.name] = spec
        self._state[node.name] = node.initial_state()
        self._dirty = True
        self._notify(EVENT_NODE_ADDED, node.name)

    def add_edge(
        self,
        source: str,
        target: str,
        source_field: str,
        target_field: str,
        transform: Optional[Callable] = None,
        additive: bool = False,
        source_units: Optional[str] = None,
        target_units: Optional[str] = None,
    ) -> None:
        """Add a data-dependency edge between two nodes.

        The *transform* parameter accepts either a callable or a
        string name registered via ``@register_transform``.  String
        names are resolved immediately; a ``KeyError`` is raised if
        the name is not in the registry.

        Parameters
        ----------
        source_units : str or None
            Physical units of the source field (e.g. ``"lattice"``).
            Informational -- used for documentation and validation.
        target_units : str or None
            Physical units after transform (e.g. ``"N"``).
            Checked against the target node's ``expected_units``.
        """
        if isinstance(transform, str):
            from maddening.core.transforms import resolve_transform
            transform = resolve_transform(transform)
        edge = EdgeSpec(source, target, source_field, target_field,
                        transform, additive, source_units, target_units)
        self._edges.append(edge)
        self._dirty = True
        self._notify(EVENT_EDGE_ADDED, edge)

    def add_external_input(
        self,
        target_node: str,
        target_field: str,
        shape: tuple = (),
        dtype: Any = jnp.float32,
    ) -> None:
        """Declare an external input that will be injected each step.

        External inputs appear in the target node's ``boundary_inputs``
        dict alongside edge-delivered values.  They are supplied via the
        ``external_inputs`` argument to :meth:`step` or :meth:`run`.

        Parameters
        ----------
        target_node : str
            Name of the node that receives this input.
        target_field : str
            Key in the node's ``boundary_inputs`` dict.
        shape : tuple
            Array shape (default ``()`` for scalar).
        dtype
            JAX dtype (default ``jnp.float32``).
        """
        spec = ExternalInputSpec(target_node, target_field, shape, dtype)
        self._external_inputs.append(spec)
        self._dirty = True

    def remove_node(self, name: str) -> None:
        """Remove a node and all edges / external inputs that reference it."""
        if name not in self._nodes:
            raise KeyError(f"No node named '{name}'.")
        del self._nodes[name]
        del self._state[name]
        self._edges = [
            e for e in self._edges
            if e.source_node != name and e.target_node != name
        ]
        self._external_inputs = [
            e for e in self._external_inputs if e.target_node != name
        ]
        self._dirty = True
        self._notify(EVENT_NODE_REMOVED, name)

    def remove_edge(
        self,
        source: str,
        target: str,
        source_field: str,
        target_field: str,
    ) -> None:
        """Remove a specific edge."""
        edge = EdgeSpec(source, target, source_field, target_field)
        self._edges = [
            e for e in self._edges
            if not (
                e.source_node == edge.source_node
                and e.target_node == edge.target_node
                and e.source_field == edge.source_field
                and e.target_field == edge.target_field
            )
        ]
        self._dirty = True
        self._notify(EVENT_EDGE_REMOVED, edge)

    # ------------------------------------------------------------------
    # Coupling groups
    # ------------------------------------------------------------------

    def add_coupling_group(
        self,
        nodes: Sequence[str],
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        **kwargs,
    ) -> CouplingGroup:
        """Register an iteratively-coupled group of nodes.

        Within each timestep, the nodes in the group are executed
        repeatedly until convergence or *max_iterations*.
        All edges between nodes in the group use current-iteration
        values rather than staggered (previous-timestep) values.

        Parameters
        ----------
        nodes : sequence of str
            Node names forming the coupling group.  Must all exist in
            the graph and should form (part of) a cycle.
        max_iterations : int
            Maximum iterations per timestep.
        tolerance : float
            Convergence threshold (L2 norm of state change).
        **kwargs
            Additional keyword arguments forwarded to
            :class:`~maddening.core.coupling.CouplingGroup`
            (e.g. ``convergence_norm``, ``acceleration``,
            ``iteration_mode``, ``diagnostics``).

        Returns
        -------
        CouplingGroup
            The created coupling group descriptor.
        """
        for name in nodes:
            if name not in self._nodes:
                raise KeyError(f"No node named '{name}'.")
        # Check for overlap with existing coupling groups
        new_set = frozenset(nodes)
        for existing in self._coupling_groups:
            overlap = new_set & existing.nodes
            if overlap:
                raise ValueError(
                    f"Nodes {overlap} already belong to a coupling group."
                )
        group = CouplingGroup(
            nodes=new_set,
            max_iterations=max_iterations,
            tolerance=tolerance,
            **kwargs,
        )
        self._coupling_groups.append(group)
        self._dirty = True
        return group

    def remove_coupling_group(self, nodes: Sequence[str]) -> None:
        """Remove a coupling group by its node set."""
        target = frozenset(nodes)
        self._coupling_groups = [
            g for g in self._coupling_groups if g.nodes != target
        ]
        self._dirty = True

    def auto_couple(
        self,
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        **kwargs,
    ) -> list[CouplingGroup]:
        """Automatically create coupling groups from graph cycles.

        Uses Tarjan's algorithm to find strongly connected components
        and creates a coupling group for each SCC with more than one
        node.  Existing coupling groups are cleared first.

        Parameters
        ----------
        max_iterations : int
            Maximum iterations per timestep.
        tolerance : float
            Convergence threshold (L2 norm of state change).
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`add_coupling_group`.

        Returns
        -------
        list of CouplingGroup
            The created coupling groups.
        """
        self._coupling_groups.clear()
        sccs = find_strongly_connected_components(
            list(self._nodes.keys()), self._edges
        )
        groups = []
        for scc in sccs:
            g = self.add_coupling_group(
                scc, max_iterations, tolerance, **kwargs
            )
            groups.append(g)
        return groups

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Check graph integrity.  Returns a list of warning/error strings."""
        issues: list[str] = []
        node_names = set(self._nodes.keys())

        # Edge endpoint checks
        for e in self._edges:
            if e.source_node not in node_names:
                issues.append(f"ERROR: edge references non-existent source node '{e.source_node}'")
            if e.target_node not in node_names:
                issues.append(f"ERROR: edge references non-existent target node '{e.target_node}'")

            # Field existence (check state fields and flux fields)
            if e.source_node in self._state:
                if e.source_field not in self._state[e.source_node]:
                    # Also check flux fields from compute_boundary_fluxes
                    is_flux_field = False
                    if e.source_node in self._nodes:
                        node_obj = self._nodes[e.source_node].node
                        from maddening.core.node import SimulationNode as _SimBase
                        if type(node_obj).compute_boundary_fluxes is not _SimBase.compute_boundary_fluxes:
                            flux_keys = node_obj.compute_boundary_fluxes(
                                self._state[e.source_node], {}, 0.0
                            ).keys()
                            if e.source_field in flux_keys:
                                is_flux_field = True
                    if not is_flux_field:
                        issues.append(
                            f"ERROR: source field '{e.source_field}' not in state of node '{e.source_node}'. "
                            f"Available: {list(self._state[e.source_node].keys())}"
                        )

        # Edge validation: shape, dtype, units against BoundaryInputSpec
        for e in self._edges:
            if e.target_node not in self._nodes:
                continue
            bi_spec = self._nodes[e.target_node].node.boundary_input_spec()
            if e.target_field not in bi_spec:
                continue
            spec = bi_spec[e.target_field]

            # Shape check: compare when both source and spec shapes are
            # concrete.  ``spec.shape == ()`` means the input is a scalar
            # — non-scalar sources still get flagged.
            source_state = self._state.get(e.source_node, {})
            src_val = source_state.get(e.source_field)
            if src_val is not None:
                src_shape = tuple(int(d) for d in getattr(src_val, "shape", ()))
                spec_shape = tuple(spec.shape)
                # Skip when spec leaves any dimension symbolic (negative
                # convention) or when a transform may reshape on the fly.
                if (e.transform is None
                        and all(d >= 0 for d in spec_shape)
                        and src_shape != spec_shape):
                    issues.append(
                        f"WARNING[shape]: edge "
                        f"{e.source_node}.{e.source_field} -> "
                        f"{e.target_node}.{e.target_field}: "
                        f"source shape {src_shape} disagrees with "
                        f"target BoundaryInputSpec shape {spec_shape} "
                        f"and no transform is set"
                    )

            # Dtype check: only when both source and spec dtypes are set.
            if src_val is not None and spec.dtype is not None:
                src_dtype = getattr(src_val, "dtype", None)
                if src_dtype is not None and e.transform is None:
                    if str(src_dtype) != str(jnp.dtype(spec.dtype)):
                        issues.append(
                            f"WARNING[dtype]: edge "
                            f"{e.source_node}.{e.source_field} -> "
                            f"{e.target_node}.{e.target_field}: "
                            f"source dtype {src_dtype} disagrees with "
                            f"target BoundaryInputSpec dtype "
                            f"{jnp.dtype(spec.dtype)} "
                            f"and no transform is set"
                        )

            # Unit checks (existing behaviour, retained).
            if (e.target_units is not None
                    and spec.expected_units is not None
                    and e.target_units != spec.expected_units):
                issues.append(
                    f"WARNING[units]: unit mismatch on edge "
                    f"{e.source_node}.{e.source_field} -> "
                    f"{e.target_node}.{e.target_field}: "
                    f"edge declares target_units='{e.target_units}' "
                    f"but node expects '{spec.expected_units}'"
                )
            if (e.source_units is not None
                    and spec.expected_units is not None
                    and e.source_units != spec.expected_units
                    and e.transform is None):
                issues.append(
                    f"WARNING[units]: edge "
                    f"{e.source_node}.{e.source_field} -> "
                    f"{e.target_node}.{e.target_field} has "
                    f"source_units='{e.source_units}' but target "
                    f"expects '{spec.expected_units}' and no "
                    f"transform is set"
                )

        # External input endpoint checks
        for ei in self._external_inputs:
            if ei.target_node not in node_names:
                issues.append(
                    f"ERROR: external input references non-existent node '{ei.target_node}'"
                )

        # Disconnected-node warning.  Only meaningful when the graph
        # has multiple nodes -- a single-node graph is trivially
        # "disconnected" but the warning is just noise (the quickstart
        # shape).  v0.2.1 gates this behind ``len(node_names) > 1``.
        if len(node_names) > 1:
            connected = set()
            for e in self._edges:
                connected.add(e.source_node)
                connected.add(e.target_node)
            for ei in self._external_inputs:
                connected.add(ei.target_node)
            for n in node_names:
                if n not in connected:
                    issues.append(
                        f"WARNING: node '{n}' is disconnected "
                        "(no edges or external inputs)"
                    )

        # Multi-rate timestep informational message
        timesteps = {spec.timestep for spec in self._nodes.values()}
        if len(timesteps) > 1:
            base_dt = _multi_gcd(sorted(timesteps))
            dividers = {
                name: round(spec.timestep / base_dt)
                for name, spec in self._nodes.items()
            }
            issues.append(
                f"INFO: multi-rate scheduling enabled. "
                f"Base timestep: {base_dt}, rate dividers: {dividers}"
            )

        # Coupling group validation
        coupled_nodes: set[str] = set()
        for group in self._coupling_groups:
            for n in group.nodes:
                if n not in node_names:
                    issues.append(
                        f"ERROR: coupling group references non-existent node '{n}'"
                    )
            # Check uniform timestep within coupling group
            # (relaxed when subcycling is enabled)
            group_timesteps = {
                self._nodes[n].timestep
                for n in group.nodes
                if n in self._nodes
            }
            if len(group_timesteps) > 1 and not group.subcycling:
                issues.append(
                    f"ERROR: coupling group {set(group.nodes)} has mixed "
                    f"timesteps {group_timesteps}. All nodes in a coupling "
                    f"group must share the same timestep.  Set "
                    f"subcycling=True to enable mixed-timestep coupling."
                )
            coupled_nodes |= group.nodes

        # Cycle detection (only on edges with valid endpoints)
        valid_edges = [
            e for e in self._edges
            if e.source_node in node_names and e.target_node in node_names
        ]
        cycles = detect_cycles(list(self._nodes.keys()), valid_edges)
        for cyc in cycles:
            # Check if cycle is covered by a coupling group
            cyc_set = set(cyc)
            covered = any(cyc_set <= g.nodes for g in self._coupling_groups)
            if covered:
                issues.append(
                    f"INFO: cycle {' -> '.join(cyc)} handled by iterative "
                    f"coupling (Gauss-Seidel)."
                )
            else:
                # Uncovered cycles are handled by staggering (back-edges
                # read previous-timestep values) -- not an error and not
                # something the user can usually act on at compile time.
                # v0.2.1 demotes this from a UserWarning to a
                # ``logging.info`` record so it stops bubbling up through
                # downstream ``filterwarnings=["error"]`` configs.  The
                # prefix flip from ``WARNING:`` to ``INFO:`` also takes
                # the message out of compile()'s warning-emission loop.
                msg = (
                    f"cycle detected: {' -> '.join(cyc)}. "
                    "Back-edges will use previous-timestep values "
                    "(staggering)."
                )
                logger.info(msg)
                issues.append(f"INFO: {msg}")

        return issues

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(self) -> None:
        """Topologically sort the graph and JIT-compile the step function."""
        issues = self.validate()
        errors = [i for i in issues if i.startswith("ERROR")]
        if errors:
            raise RuntimeError(
                "Cannot compile graph with errors:\n" + "\n".join(errors)
            )
        # Aggregate every problem so the user sees them all in a single
        # pass.  Since v0.2.1, shape and dtype mismatches are hard
        # errors (pre-announced in v0.2.0 release notes; semver
        # carve-out documented in docs/developer_guide/
        # edge_validation_migration.md).  They are collected here and
        # raised together as an ExceptionGroup.  Unit mismatches stay
        # as warnings; plain advisory "WARNING:" issues stay as
        # UserWarning.
        from maddening.warnings import (
            DtypeMismatchError,
            EdgeValidationError,  # noqa: F401 — exported for callers
            ExceptionGroup,
            ShapeMismatchError,
            UnitMismatchWarning,
        )
        validation_errors: list[EdgeValidationError] = []
        for issue in issues:
            if not issue.startswith("WARNING"):
                continue
            if issue.startswith("WARNING[shape]"):
                validation_errors.append(ShapeMismatchError(issue))
            elif issue.startswith("WARNING[dtype]"):
                validation_errors.append(DtypeMismatchError(issue))
            elif issue.startswith("WARNING[units]"):
                warnings.warn(issue, UnitMismatchWarning, stacklevel=2)
            else:
                warnings.warn(issue, stacklevel=2)
        if validation_errors:
            raise ExceptionGroup(
                "edge validation failed", validation_errors
            )

        node_names = list(self._nodes.keys())
        self._schedule = topological_sort(node_names, self._edges)
        self._back_edges = identify_back_edges(self._schedule, self._edges)

        # Compute multi-rate info.
        # For nodes in subcycling coupling groups, use the group's
        # macro timestep (max of member timesteps) for rate divider
        # computation, since the coupling block handles sub-stepping.
        effective_timesteps = {
            name: spec.timestep for name, spec in self._nodes.items()
        }
        for g in self._coupling_groups:
            if g.subcycling:
                group_max_dt = max(
                    self._nodes[n].timestep for n in g.nodes
                    if n in self._nodes
                )
                for n in g.nodes:
                    if n in effective_timesteps:
                        effective_timesteps[n] = group_max_dt

        timesteps = sorted(set(effective_timesteps.values()))
        if len(timesteps) > 1:
            self._is_multirate = True
            base_dt = _multi_gcd(timesteps)
            self._rate_dividers = {
                name: round(effective_timesteps[name] / base_dt)
                for name in self._nodes
            }
            # Initialise the step counter in the state
            self._state[_META_KEY] = {
                "step_count": jnp.array(0, dtype=jnp.int32),
            }
        else:
            self._is_multirate = False
            self._rate_dividers = {name: 1 for name in self._nodes}
            # Remove meta if it existed from a previous compile
            self._state.pop(_META_KEY, None)

        # Ensure _meta exists with correct structure when coupling
        # diagnostics are enabled.  Pre-populate diagnostic keys so
        # the pytree structure is stable across lax.scan iterations.
        has_diagnostics = any(g.diagnostics for g in self._coupling_groups)
        has_imvj = any(
            g.acceleration == "iqn-imvj" for g in self._coupling_groups
        )
        has_predictor = any(
            g.predictor != "none" for g in self._coupling_groups
        )
        if has_diagnostics or has_imvj or has_predictor:
            meta = self._state.get(_META_KEY, {})
            for g in self._coupling_groups:
                key = "+".join(sorted(g.nodes))
                if g.diagnostics:
                    meta[f"coupling_{key}_iterations"] = jnp.array(
                        0, dtype=jnp.int32
                    )
                    meta[f"coupling_{key}_residual"] = jnp.array(0.0)
                if g.acceleration == "iqn-imvj":
                    # Pre-populate V/W matrices for IQN-IMVJ
                    from maddening.core.coupling.acceleration import (
                        flatten_coupled_state,
                    )
                    group_names = sorted(g.nodes)
                    # Determine accel_fields
                    if g.accelerated_fields is not None:
                        af = g.accelerated_fields
                    else:
                        ifields: dict[str, set] = {}
                        for edge in self._edges:
                            if (edge.source_node in g.nodes
                                    and edge.target_node in g.nodes):
                                ifields.setdefault(
                                    edge.source_node, set()
                                ).add(edge.source_field)
                        af = {
                            nn: tuple(sorted(fs))
                            for nn, fs in ifields.items()
                        } if ifields else None
                    n_dof = flatten_coupled_state(
                        self._state, list(g.nodes), fields=af
                    ).shape[0]
                    max_cols = max(g.max_iterations - 1, 1)
                    meta[f"coupling_{key}_V"] = jnp.zeros(
                        (n_dof, max_cols)
                    )
                    meta[f"coupling_{key}_W"] = jnp.zeros(
                        (n_dof, max_cols)
                    )
                if g.predictor != "none":
                    # Pre-populate predictor history with flattened
                    # node states.  Use flatten_coupled_state with
                    # all fields (no acceleration field filtering).
                    from maddening.core.coupling.acceleration import (
                        flatten_coupled_state as _fcs_pred,
                    )
                    group_names_pred = list(g.nodes)
                    flat0 = _fcs_pred(self._state, group_names_pred)
                    n_pred = 3 if g.predictor == "quadratic" else 2
                    for pi in range(n_pred):
                        meta[f"coupling_{key}_pred_{pi}"] = flat0
                    # Counter for how many converged states have
                    # been stored (0 at start, up to n_pred).
                    meta[f"coupling_{key}_pred_count"] = jnp.array(
                        0, dtype=jnp.int32
                    )
            self._state[_META_KEY] = meta

        step_fn = self._build_step_fn()
        self._compiled_step = jax.jit(step_fn)

        # Snapshot static_data hashes so we can detect drift.
        self._static_data_hashes = {
            name: spec.node.static_data_hash()
            for name, spec in self._nodes.items()
        }

        self._dirty = False
        self._notify(EVENT_COMPILED, self._schedule)

    def _check_static_data_dirty(self) -> bool:
        """Return True (and set ``self._dirty=True``) if any node's
        :attr:`static_data` has changed shape/dtype since the last
        ``compile()``.

        Called from each public entry-point (``step``, ``run``, etc.)
        before the standard dirty-check so a stale JIT cache is caught
        without requiring the caller to mark the graph dirty manually.
        """
        for name, spec in self._nodes.items():
            if spec.node.static_data_hash() != self._static_data_hashes.get(name, 0):
                self._dirty = True
                return True
        return False

    # ------------------------------------------------------------------
    # Sharding validation (v0.2 #3 follow-up)
    # ------------------------------------------------------------------

    def validate_sharding(self) -> list["ShardingIssue"]:
        """Structural checks for the sharding spec across the graph.

        Today: sharding spec consistency only.

        Returns a list of :class:`ShardingIssue` instances (each typed
        with a ``severity`` and a ``code``); the empty list means
        healthy.  Callers decide severity — turn into a raising call
        by filtering for ``severity == "error"`` and calling ``raise``.

        Scope (deliberately tight to avoid a god-method):

        * A sharded node exists, but the graph has no device mesh
          configured.
        * A sharded node's mesh disagrees with the graph's device mesh.

        NOT in scope here:

        * Edge-validation (compile-time; see
          :meth:`validate`).
        * Multi-rate divisibility.
        * Cycle detection.

        If you find yourself wanting to extend this method past
        sharding-spec consistency, consider a sibling
        ``validate_<topic>()`` instead.
        """
        issues: list[ShardingIssue] = []
        # Identify sharded nodes by either explicit class or by carrying
        # a `_mesh` attribute (the Sharded*Node convention).
        sharded_nodes = [
            (name, spec.node) for name, spec in self._nodes.items()
            if hasattr(spec.node, "_mesh") and getattr(spec.node, "_mesh", None) is not None
        ]

        if not sharded_nodes:
            return issues  # nothing to validate

        if self._multigpu_mesh is None:
            issues.append(ShardingIssue(
                severity="warning",
                code="sharded_node_without_graph_mesh",
                message=(
                    f"{len(sharded_nodes)} sharded node(s) present but the "
                    f"GraphManager has no device mesh configured.  "
                    f"Call gm.enable_multigpu(...) or remove the sharding "
                    f"from the affected nodes."
                ),
                affected_nodes=[name for name, _ in sharded_nodes],
            ))
            return issues

        # All sharded nodes must agree with the graph's mesh.
        graph_axis_names = tuple(self._multigpu_mesh.axis_names)
        for name, node in sharded_nodes:
            node_mesh = getattr(node, "_mesh", None)
            node_axes = tuple(node_mesh.axis_names)
            if node_axes != graph_axis_names:
                issues.append(ShardingIssue(
                    severity="error",
                    code="sharded_node_mesh_axes_mismatch",
                    message=(
                        f"Sharded node {name!r} uses mesh axes {node_axes!r} "
                        f"but the graph's mesh is {graph_axis_names!r}.  "
                        f"Re-create the node with the graph's mesh, or "
                        f"call enable_multigpu with matching axes."
                    ),
                    affected_nodes=[name],
                ))
        return issues

    # ------------------------------------------------------------------
    # Multi-GPU
    # ------------------------------------------------------------------

    def enable_multigpu(
        self,
        n_devices: Optional[int] = None,
        partition_strategy: str = "auto",
        *,
        mesh_shape: Optional[tuple[int, ...]] = None,
        mesh_axes: Optional[tuple[str, ...]] = None,
    ) -> None:
        """Enable multi-GPU coupling and (in v0.2) stencil sharding.

        Requires at least one coupling group with ``iteration_mode="jacobi"``.
        Uses ``jax.experimental.shard_map`` to distribute node updates
        across a device mesh.

        Parameters
        ----------
        n_devices : int, optional
            Number of devices to use.  Defaults to ``prod(mesh_shape)`` when
            ``mesh_shape`` is provided, otherwise all available devices.
        partition_strategy : str
            ``"auto"`` (default) assigns coupled nodes to the same device.
        mesh_shape : tuple[int, ...], optional
            Mesh shape for N-D (pencil) decomposition, e.g. ``(2, 4)`` for
            an 8-device 2-D pencil mesh.  When omitted the mesh is 1-D
            (slab decomposition, v0.1 behaviour).
        mesh_axes : tuple[str, ...], optional
            Axis names for the mesh.  Defaults: ``("devices",)`` for 1-D
            and ``("spatial_y", "spatial_z")`` for 2-D.  Length must match
            ``len(mesh_shape)``.
        """
        from maddening.cloud.multigpu.device_mesh import create_device_mesh
        from maddening.cloud.multigpu.partition import assign_nodes_to_devices

        jacobi_groups = [
            g for g in self._coupling_groups
            if g.iteration_mode == "jacobi"
        ]
        if not jacobi_groups:
            raise ValueError(
                "enable_multigpu requires at least one coupling group "
                "with iteration_mode='jacobi'"
            )

        self._multigpu_mesh = create_device_mesh(
            n_devices, shape=mesh_shape, axis_names=mesh_axes
        )
        n = len(self._multigpu_mesh.devices.reshape(-1))

        coupling_sets = [set(g.nodes) for g in self._coupling_groups]
        edges_dicts = [e.to_dict() for e in self._edges]
        self._multigpu_device_map = assign_nodes_to_devices(
            node_names=list(self._nodes.keys()),
            edges=edges_dicts,
            coupling_groups=coupling_sets,
            n_devices=n,
        )
        self._dirty = True

    def _build_step_fn(self) -> Callable:
        """Create a pure function ``(full_state, ext_inputs) -> full_state``.

        When multi-rate is active, the step function increments an
        internal step counter and conditionally applies each node's
        update based on whether ``step_count % rate_divider == 0``.
        The update is always *computed* (to keep the function
        JAX-traceable with static structure), but the result is applied
        only when the node should fire.

        When coupling groups are defined, nodes within each group are
        wrapped in a ``jax.lax.while_loop`` that iterates
        (Gauss-Seidel) until convergence or max_iterations.
        """
        schedule = list(self._schedule)
        nodes = dict(self._nodes)
        back_edge_set = set(self._back_edges)
        is_multirate = self._is_multirate
        rate_dividers = dict(self._rate_dividers)
        coupling_groups = list(self._coupling_groups)

        # Map node -> coupling group
        node_to_group: dict[str, CouplingGroup] = {}
        for group in coupling_groups:
            for name in group.nodes:
                node_to_group[name] = group

        # Build block schedule: list of (type, data) where type is
        # "node" (single node) or "coupled" (CouplingGroup, node_list)
        blocks: list[tuple] = []
        handled_groups: set[int] = set()
        for node_name in schedule:
            if node_name in node_to_group:
                group = node_to_group[node_name]
                gid = id(group)
                if gid not in handled_groups:
                    handled_groups.add(gid)
                    # Collect nodes in this group in schedule order
                    group_schedule = [n for n in schedule if n in group.nodes]
                    blocks.append(("coupled", group, group_schedule))
            else:
                blocks.append(("node", node_name))

        # Identify edges within each coupling group (these become
        # forward edges during iteration, not back-edges)
        coupled_internal_edges: set[EdgeSpec] = set()
        for group in coupling_groups:
            for edge in self._edges:
                if edge.source_node in group.nodes and edge.target_node in group.nodes:
                    coupled_internal_edges.add(edge)

        # Pre-index edges by target node -- O(E) setup, O(degree) per node
        edges_by_target: dict[str, list[EdgeSpec]] = defaultdict(list)
        for edge in self._edges:
            edges_by_target[edge.target_node].append(edge)

        # Pre-index external inputs by target node
        ext_by_target: dict[str, list[ExternalInputSpec]] = defaultdict(list)
        for ei in self._external_inputs:
            ext_by_target[ei.target_node].append(ei)

        # Capture which nodes have external inputs (for the fast path)
        has_external = set(ext_by_target.keys())

        has_coupling = bool(coupling_groups)

        # Track flux outputs for flux-based edges in non-coupled path
        flux_state: dict[str, dict] = {}

        def _resolve_and_update_node(
            node_name, new_state, full_state, external_inputs,
            force_forward_edges=None,
        ):
            """Resolve boundary inputs and update a single node.

            Parameters
            ----------
            force_forward_edges : set or None
                If provided, edges in this set are treated as forward
                (use new_state) even if they are in back_edge_set.
            """
            boundary_inputs: dict[str, Any] = {}

            for edge in edges_by_target[node_name]:
                # Determine source state: back-edges read from full_state
                # (previous timestep), forward edges from new_state.
                if edge in back_edge_set and (
                    force_forward_edges is None
                    or edge not in force_forward_edges
                ):
                    src_state = full_state
                else:
                    src_state = new_state
                # Check state first, then flux outputs
                src_nn = edge.source_node
                src_dict = src_state.get(src_nn, {})
                if edge.source_field in src_dict:
                    value = src_dict[edge.source_field]
                elif src_nn in flux_state and edge.source_field in flux_state[src_nn]:
                    value = flux_state[src_nn][edge.source_field]
                else:
                    value = src_state[src_nn][edge.source_field]
                if edge.transform is not None:
                    value = edge.transform(value)
                if edge.additive and edge.target_field in boundary_inputs:
                    boundary_inputs[edge.target_field] = (
                        boundary_inputs[edge.target_field] + value
                    )
                else:
                    boundary_inputs[edge.target_field] = value

            if node_name in has_external:
                node_ext = external_inputs.get(node_name, {})
                for ei in ext_by_target[node_name]:
                    if ei.target_field in node_ext:
                        boundary_inputs[ei.target_field] = node_ext[ei.target_field]

            spec = nodes[node_name]
            new_node_state = spec.update_fn(
                new_state[node_name], boundary_inputs, spec.timestep
            )

            # Compute fluxes for this node if it produces them
            from maddening.core.node import SimulationNode as _SimBase
            if type(spec.node).compute_boundary_fluxes is not _SimBase.compute_boundary_fluxes:
                fluxes = spec.node.compute_boundary_fluxes(
                    new_node_state, boundary_inputs, spec.timestep
                )
                if fluxes:
                    flux_state[node_name] = fluxes

            return new_node_state

        def _run_coupled_block(group, group_schedule, new_state,
                               full_state, external_inputs,
                               runtime_dt=None):
            """Execute a coupling group with Gauss-Seidel iteration.

            In Gauss-Seidel coupling, each iteration re-solves the SAME
            timestep with updated boundary conditions from the latest
            iteration.  Crucially, each node integrates from the
            *initial* state (beginning of timestep), NOT from the
            previous iteration's output.  Only the boundary conditions
            change between iterations.

            Parameters
            ----------
            group : CouplingGroup
                Configuration for this coupling group.
            group_schedule : list of str
                Node names in execution order within the group.
            new_state : dict
                Current accumulated state for this timestep.
            full_state : dict
                State from the previous timestep (for back-edges).
            external_inputs : dict
                External inputs dict.
            runtime_dt : JAX scalar or None
                If provided, overrides each node's compiled timestep
                (used by adaptive timestepping).
            """
            return _run_coupled_block_impl(
                group, group_schedule, new_state, full_state,
                external_inputs, runtime_dt,
                nodes=nodes, edges_by_target=edges_by_target,
                ext_by_target=ext_by_target, back_edge_set=back_edge_set,
                has_external=has_external, all_edges=self._edges,
                multigpu_device_map=self._multigpu_device_map,
            )

        if not is_multirate and not has_coupling:
            # ---- Uniform-rate, no coupling: fast path ----
            def graph_step(full_state, external_inputs):
                new_state = {k: v for k, v in full_state.items()}

                for node_name in schedule:
                    new_state[node_name] = _resolve_and_update_node(
                        node_name, new_state, full_state, external_inputs
                    )
                return new_state

            return graph_step

        if has_coupling and not is_multirate:
            # ---- Coupling groups, uniform rate ----
            def graph_step_coupled(full_state, external_inputs):
                new_state = {k: v for k, v in full_state.items()}

                for block in blocks:
                    if block[0] == "node":
                        node_name = block[1]
                        new_state[node_name] = _resolve_and_update_node(
                            node_name, new_state, full_state, external_inputs
                        )
                    else:
                        _, group, group_schedule = block
                        new_state = _run_coupled_block(
                            group, group_schedule, new_state,
                            full_state, external_inputs,
                        )

                return new_state

            return graph_step_coupled

        # ---- Multi-rate path (with or without coupling) ----
        def graph_step_multirate(full_state, external_inputs):
            step_count = full_state[_META_KEY]["step_count"]
            new_state = {k: v for k, v in full_state.items()}

            def _apply_multirate(node_name, updated, current_state):
                rd = rate_dividers[node_name]
                if rd == 1:
                    return updated
                should_run = (step_count % rd) == 0
                return jax.tree.map(
                    lambda new_val, old_val: jnp.where(should_run, new_val, old_val),
                    updated,
                    current_state[node_name],
                )

            if has_coupling:
                for block in blocks:
                    if block[0] == "node":
                        node_name = block[1]
                        updated = _resolve_and_update_node(
                            node_name, new_state, full_state, external_inputs
                        )
                        new_state[node_name] = _apply_multirate(
                            node_name, updated, new_state
                        )
                    else:
                        _, group, group_schedule = block
                        coupled_result = _run_coupled_block(
                            group, group_schedule, new_state,
                            full_state, external_inputs,
                        )
                        for nn in group_schedule:
                            new_state[nn] = _apply_multirate(
                                nn, coupled_result[nn], new_state
                            )
                        # Propagate diagnostic keys from coupled result
                        if _META_KEY in coupled_result:
                            new_state[_META_KEY] = {
                                **new_state.get(_META_KEY, {}),
                                **coupled_result[_META_KEY],
                            }
            else:
                for node_name in schedule:
                    updated = _resolve_and_update_node(
                        node_name, new_state, full_state, external_inputs
                    )
                    new_state[node_name] = _apply_multirate(
                        node_name, updated, new_state
                    )

            # Increment step counter (preserve diagnostic keys)
            new_state[_META_KEY] = {
                **new_state.get(_META_KEY, {}),
                "step_count": step_count + 1,
            }
            return new_state

        return graph_step_multirate

    def _default_external_inputs(self) -> dict[str, dict]:
        """Build a zero-valued external_inputs dict matching declared specs."""
        if not self._external_inputs:
            return _EMPTY_EXTERNAL_INPUTS
        ext: dict[str, dict] = {}
        for ei in self._external_inputs:
            ext.setdefault(ei.target_node, {})[ei.target_field] = jnp.zeros(
                ei.shape, dtype=ei.dtype
            )
        return ext

    # ------------------------------------------------------------------
    # Internal helpers for _meta stripping
    # ------------------------------------------------------------------

    def _user_state(self, full_state: dict) -> dict:
        """Return state dict without the internal ``_meta`` key."""
        if _META_KEY not in full_state:
            return full_state
        return {k: v for k, v in full_state.items() if k != _META_KEY}

    def coupling_diagnostics(self) -> dict[str, dict]:
        """Return coupling convergence info from the last step.

        Returns
        -------
        dict
            Keyed by coupling group identifier (sorted node names
            joined by ``"+"``), each containing:

            - ``"iterations"`` : int — coupling iterations used
            - ``"residual"`` : float — final residual norm

            Empty dict if no coupling groups have ``diagnostics=True``
            or no step has been taken yet.
        """
        meta = self._state.get(_META_KEY, {})
        result: dict[str, dict] = {}
        for group in self._coupling_groups:
            if not group.diagnostics:
                continue
            key = "+".join(sorted(group.nodes))
            iter_key = f"coupling_{key}_iterations"
            res_key = f"coupling_{key}_residual"
            if iter_key in meta:
                result[key] = {
                    "iterations": int(meta[iter_key]),
                    "residual": float(meta[res_key]),
                }
        return result

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def step(self, external_inputs: Optional[dict[str, dict]] = None) -> dict[str, dict]:
        """Advance the simulation by one base timestep.

        Parameters
        ----------
        external_inputs : dict, optional
            Values injected from outside the graph, structured as
            ``{node_name: {field_name: value, ...}, ...}``.
            If ``None``, zeros are used for all declared external inputs.

        Returns the full state dict after the step (excluding internal
        metadata).
        """
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()

        if external_inputs is None:
            external_inputs = self._default_external_inputs()

        self._state = self._compiled_step(self._state, external_inputs)
        user_state = self._user_state(self._state)
        self._notify(EVENT_STEP, user_state)
        return user_state

    def run(
        self,
        n_steps: int,
        callback: Optional[Callable] = None,
        external_inputs: Optional[dict[str, dict]] = None,
    ) -> None:
        """Run *n_steps* simulation steps (at the base timestep rate).

        Parameters
        ----------
        n_steps : int
            Number of base-rate steps to execute.
        callback : callable, optional
            Called after every step with ``(step_index, state_dict)``.
            The state dict excludes internal metadata.
        external_inputs : dict, optional
            Static external inputs applied every step.  For dynamic
            inputs that change each step, use :meth:`step` in a loop
            or use a ``CommandReceiver`` with ``RealtimeRunner``.
        """
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()

        if external_inputs is None:
            external_inputs = self._default_external_inputs()

        for i in range(n_steps):
            self._state = self._compiled_step(self._state, external_inputs)
            user_state = self._user_state(self._state)
            self._notify(EVENT_STEP, user_state)
            if callback is not None:
                callback(i, user_state)

    def run_scan(
        self,
        n_steps: int,
        external_inputs: Optional[dict[str, dict]] = None,
    ) -> dict[str, dict]:
        """Run *n_steps* using ``jax.lax.scan`` for maximum performance.

        Unlike :meth:`run`, this method pushes the entire loop into XLA
        via ``jax.lax.scan``, eliminating Python-loop and JAX-dispatch
        overhead.  The full computation is JIT-compiled into a single
        XLA program.

        Trade-offs compared to :meth:`run` / :meth:`step`:

        * No per-step callback or observer notifications.
        * External inputs are **static** -- the same values are applied
          at every timestep.  For dynamic per-step inputs, use
          :meth:`step` in a loop or ``RealtimeRunner``.

        Parameters
        ----------
        n_steps : int
            Number of base-rate simulation steps to execute.
        external_inputs : dict, optional
            Static external inputs applied identically every step.
            If ``None``, zeros are used for all declared external inputs.

        Returns
        -------
        dict[str, dict]
            The final state of the graph after *n_steps* (excluding
            internal metadata).
        """
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()

        if external_inputs is None:
            external_inputs = self._default_external_inputs()

        # Build the raw (unjitted) step function -- lax.scan will JIT
        # the entire scan body, so an inner jit would be redundant.
        step_fn = self._build_step_fn()

        # Close over the static external inputs so the scan body has
        # the correct signature: (carry, x) -> (carry, None)
        ext = external_inputs

        def scan_body(state, _unused):
            new_state = step_fn(state, ext)
            return new_state, None

        final_state, _ = jax.lax.scan(scan_body, self._state, None, length=n_steps)
        self._state = final_state
        return self._user_state(self._state)

    def run_scan_with_history(
        self,
        n_steps: int,
        external_inputs: Optional[dict[str, dict]] = None,
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        """Run *n_steps* via ``jax.lax.scan``, returning all intermediate states.

        Like :meth:`run_scan` but also collects the state at every
        timestep into stacked arrays, which is useful for plotting and
        post-hoc analysis without Python-loop overhead.

        Parameters
        ----------
        n_steps : int
            Number of base-rate simulation steps to execute.
        external_inputs : dict, optional
            Static external inputs applied identically every step.
            If ``None``, zeros are used for all declared external inputs.

        Returns
        -------
        (final_state, history) : tuple
            *final_state* is the state dict after the last step
            (same as :meth:`run_scan` would return), excluding internal
            metadata.
            *history* has the same nested-dict structure as state (also
            excluding internal metadata), but each leaf is a JAX array
            with an extra leading axis of size *n_steps*.
            ``history["ball"]["position"]`` is a 1-D array of shape
            ``(n_steps,)`` (or ``(n_steps, *field_shape)`` for
            non-scalar fields) holding the value **after** each step.
        """
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()

        if external_inputs is None:
            external_inputs = self._default_external_inputs()

        step_fn = self._build_step_fn()
        ext = external_inputs

        def scan_body(state, _unused):
            new_state = step_fn(state, ext)
            return new_state, new_state  # carry, output (stacked by scan)

        final_state, history = jax.lax.scan(scan_body, self._state, None, length=n_steps)
        self._state = final_state
        return self._user_state(final_state), self._user_state(history)

    # ------------------------------------------------------------------
    # Parameter sweeps via vmap
    # ------------------------------------------------------------------

    def run_sweep(
        self,
        n_steps: int,
        initial_states: dict[str, dict],
        external_inputs: Optional[dict[str, dict]] = None,
        return_history: bool = False,
    ):
        """Run a batch of simulations over different initial conditions.

        Uses ``jax.vmap`` over ``jax.lax.scan`` to execute all
        variations in parallel (vectorised on GPU/TPU).

        Parameters
        ----------
        n_steps : int
            Number of steps per simulation.
        initial_states : dict[str, dict]
            Batched initial states.  Each leaf array must have a
            leading batch dimension of the same size.  For example::

                {"ball": {"position": jnp.array([1.0, 2.0, 3.0]),
                          "velocity": jnp.zeros(3)}}

            runs 3 simulations with initial positions 1, 2, 3.
        external_inputs : dict, optional
            Static external inputs (not batched — same for all runs).
        return_history : bool
            If True, return ``(final_states, histories)`` where
            histories has shape ``(batch, n_steps, ...)``.
            If False (default), return only ``final_states``.

        Returns
        -------
        final_states : dict[str, dict]
            Batched final states (leading batch dimension on each leaf).
        histories : dict[str, dict], optional
            Only if ``return_history=True``.  Batched histories with
            shape ``(batch, n_steps, ...)`` on each leaf.
        """
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()

        if external_inputs is None:
            external_inputs = self._default_external_inputs()

        step_fn = self._build_step_fn()
        ext = external_inputs

        # Build the scan function
        if return_history:
            def simulate(init_state):
                def scan_body(state, _unused):
                    new_state = step_fn(state, ext)
                    return new_state, new_state
                final, hist = jax.lax.scan(scan_body, init_state, None, length=n_steps)
                return self._user_state(final), self._user_state(hist)

            vmapped = jax.vmap(simulate)
            finals, histories = vmapped(initial_states)
            return finals, histories
        else:
            def simulate(init_state):
                def scan_body(state, _unused):
                    new_state = step_fn(state, ext)
                    return new_state, None
                final, _ = jax.lax.scan(scan_body, init_state, None, length=n_steps)
                return self._user_state(final)

            vmapped = jax.vmap(simulate)
            return vmapped(initial_states)

    # ------------------------------------------------------------------
    # Adaptive timestepping
    # ------------------------------------------------------------------

    def _build_dt_step_fn(self) -> Callable:
        """Build a step function parameterised by ``dt``.

        Returns a function ``(state, external_inputs, dt) -> new_state``
        where *dt* is a JAX scalar that overrides each node's compiled
        timestep.  Used by :meth:`run_adaptive`.
        """
        schedule = list(self._schedule)
        nodes_dict = dict(self._nodes)
        back_edge_set = set(self._back_edges)
        coupling_groups = list(self._coupling_groups)

        node_to_group: dict[str, CouplingGroup] = {}
        for group in coupling_groups:
            for name in group.nodes:
                node_to_group[name] = group

        blocks: list[tuple] = []
        handled_groups: set[int] = set()
        for node_name in schedule:
            if node_name in node_to_group:
                group = node_to_group[node_name]
                gid = id(group)
                if gid not in handled_groups:
                    handled_groups.add(gid)
                    group_schedule = [n for n in schedule if n in group.nodes]
                    blocks.append(("coupled", group, group_schedule))
            else:
                blocks.append(("node", node_name))

        edges_by_target: dict[str, list[EdgeSpec]] = defaultdict(list)
        for edge in self._edges:
            edges_by_target[edge.target_node].append(edge)

        ext_by_target: dict[str, list[ExternalInputSpec]] = defaultdict(list)
        for ei in self._external_inputs:
            ext_by_target[ei.target_node].append(ei)

        has_external = set(ext_by_target.keys())
        has_coupling = bool(coupling_groups)

        coupled_internal_edges: set[EdgeSpec] = set()
        for group in coupling_groups:
            for edge in self._edges:
                if edge.source_node in group.nodes and edge.target_node in group.nodes:
                    coupled_internal_edges.add(edge)

        def _resolve_and_update(node_name, new_state, full_state, ext, dt,
                                force_forward_edges=None):
            boundary_inputs: dict[str, Any] = {}
            for edge in edges_by_target[node_name]:
                if edge in back_edge_set and (
                    force_forward_edges is None
                    or edge not in force_forward_edges
                ):
                    src_state = full_state
                else:
                    src_state = new_state
                value = src_state[edge.source_node][edge.source_field]
                if edge.transform is not None:
                    value = edge.transform(value)
                if edge.additive and edge.target_field in boundary_inputs:
                    boundary_inputs[edge.target_field] = (
                        boundary_inputs[edge.target_field] + value
                    )
                else:
                    boundary_inputs[edge.target_field] = value

            if node_name in has_external:
                node_ext = ext.get(node_name, {})
                for ei in ext_by_target[node_name]:
                    if ei.target_field in node_ext:
                        boundary_inputs[ei.target_field] = node_ext[ei.target_field]

            spec = nodes_dict[node_name]
            return spec.update_fn(new_state[node_name], boundary_inputs, dt)

        def dt_step_fn(state, external_inputs, dt):
            new_state = {k: v for k, v in state.items()}

            if has_coupling:
                for block in blocks:
                    if block[0] == "node":
                        nn = block[1]
                        new_state[nn] = _resolve_and_update(
                            nn, new_state, state, external_inputs, dt
                        )
                    else:
                        _, group, group_schedule = block
                        new_state = _run_coupled_block_impl(
                            group, group_schedule, new_state, state,
                            external_inputs, runtime_dt=dt,
                            nodes=nodes_dict,
                            edges_by_target=edges_by_target,
                            ext_by_target=ext_by_target,
                            back_edge_set=back_edge_set,
                            has_external=has_external,
                            all_edges=self._edges,
                            multigpu_device_map=self._multigpu_device_map,
                        )
            else:
                for nn in schedule:
                    new_state[nn] = _resolve_and_update(
                        nn, new_state, state, external_inputs, dt
                    )

            return new_state

        return dt_step_fn

    def run_adaptive(
        self,
        t_end: float,
        dt_initial: float = 0.01,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        dt_min: float = 1e-8,
        dt_max: float = 0.1,
        external_inputs: Optional[dict[str, dict]] = None,
        callback: Optional[Callable] = None,
    ) -> tuple[dict[str, dict], dict]:
        """Run with adaptive timestepping until *t_end*.

        Uses Richardson extrapolation (step-doubling) for error
        estimation and a PI controller for step-size adjustment.
        Incompatible with multi-rate graphs.

        Parameters
        ----------
        t_end : float
            Target simulation end time.
        dt_initial : float
            Initial timestep guess.
        atol, rtol : float
            Absolute and relative error tolerances.
        dt_min, dt_max : float
            Timestep bounds.
        external_inputs : dict, optional
            Static external inputs applied every step.
        callback : callable, optional
            Called after every *accepted* step with
            ``(sim_time, dt_used, state_dict)``.

        Returns
        -------
        (final_state, info) : tuple
            *final_state* is the state dict (excluding metadata).
            *info* is a dict with ``n_steps``, ``n_rejected``,
            ``dt_history`` (list of used timesteps), and
            ``t_history`` (list of simulation times).
        """
        if self._is_multirate:
            raise RuntimeError(
                "Adaptive timestepping is incompatible with multi-rate "
                "graphs.  All nodes must share the same timestep."
            )
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()

        if external_inputs is None:
            external_inputs = self._default_external_inputs()

        from maddening.core.simulation.adaptive import AdaptiveConfig, _tree_error_norm

        config = AdaptiveConfig(
            dt_initial=dt_initial,
            atol=atol,
            rtol=rtol,
            dt_min=dt_min,
            dt_max=dt_max,
        )

        dt_step_fn = self._build_dt_step_fn()
        # JIT-compile the dt-parameterised step
        dt_step_jit = jax.jit(dt_step_fn)

        t = 0.0
        dt = dt_initial
        n_steps = 0
        n_rejected = 0
        dt_history = []
        t_history = []
        state = self._state

        while t < t_end:
            # Clamp dt so we don't overshoot t_end
            dt = min(dt, t_end - t)
            dt = max(dt, dt_min)

            dt_jax = jnp.array(dt)

            # Full step
            state_full = dt_step_jit(state, external_inputs, dt_jax)
            # Two half-steps
            half_dt = dt_jax / 2.0
            state_half = dt_step_jit(state, external_inputs, half_dt)
            state_half = dt_step_jit(state_half, external_inputs, half_dt)

            # Error estimate
            user_full = self._user_state(state_full)
            user_half = self._user_state(state_half)
            error_norm = float(_tree_error_norm(
                user_half, user_full, config.atol, config.rtol
            ))

            if error_norm <= 1.0:
                # Accept step -- use the more accurate (half-step) result
                state = state_half
                t += dt
                n_steps += 1
                dt_history.append(dt)
                t_history.append(t)

                if callback is not None:
                    callback(t, dt, self._user_state(state))
                self._notify(EVENT_STEP, self._user_state(state))

                # Grow dt
                if error_norm > 0:
                    factor = config.safety * (1.0 / error_norm) ** (1.0 / (config.order + 1))
                else:
                    factor = config.max_factor
                factor = min(max(factor, config.min_factor), config.max_factor)
                dt = min(dt * factor, dt_max)
            else:
                # Reject step -- shrink dt and retry
                n_rejected += 1
                factor = config.safety * (1.0 / error_norm) ** (1.0 / (config.order + 1))
                factor = min(max(factor, config.min_factor), config.max_factor)
                dt = max(dt * factor, dt_min)

                if dt <= dt_min:
                    # Cannot shrink further; accept with warning
                    warnings.warn(
                        f"Adaptive stepper hit dt_min={dt_min} at t={t:.6g} "
                        f"(error={error_norm:.3e}). Accepting step.",
                        stacklevel=2,
                    )
                    state = state_half
                    t += dt_min
                    n_steps += 1
                    dt_history.append(dt_min)
                    t_history.append(t)
                    if callback is not None:
                        callback(t, dt_min, self._user_state(state))
                    self._notify(EVENT_STEP, self._user_state(state))

        self._state = state
        info = {
            "n_steps": n_steps,
            "n_rejected": n_rejected,
            "dt_history": dt_history,
            "t_history": t_history,
        }
        return self._user_state(self._state), info

    def run_adaptive_scan(
        self,
        t_end: float,
        max_steps: int = 10000,
        dt_initial: float = 0.01,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        dt_min: float = 1e-8,
        dt_max: float = 0.1,
        external_inputs: Optional[dict[str, dict]] = None,
    ) -> tuple[dict[str, dict], dict[str, dict], dict]:
        """Adaptive timestepping via ``jax.lax.scan`` (differentiable).

        Like :meth:`run_adaptive` but fully JIT-compiled and
        differentiable.  Uses a fixed *max_steps* allocation; steps
        past ``t_end`` are no-ops.

        Parameters
        ----------
        t_end : float
            Target end time.
        max_steps : int
            Maximum number of steps (scan length).  Steps after reaching
            ``t_end`` produce no-op outputs.
        dt_initial, atol, rtol, dt_min, dt_max : float
            Same as :meth:`run_adaptive`.
        external_inputs : dict, optional
            Static external inputs.

        Returns
        -------
        (final_state, history, info) : tuple
            *final_state*: state after last accepted step.
            *history*: stacked state at each step (shape ``(max_steps, ...)``).
            *info*: dict with ``n_steps`` (actual steps taken, as JAX array).
        """
        if self._is_multirate:
            raise RuntimeError(
                "Adaptive timestepping is incompatible with multi-rate graphs."
            )
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()

        if external_inputs is None:
            external_inputs = self._default_external_inputs()

        from maddening.core.simulation.adaptive import AdaptiveConfig, _tree_error_norm

        config = AdaptiveConfig(
            dt_initial=dt_initial, atol=atol, rtol=rtol,
            dt_min=dt_min, dt_max=dt_max,
        )

        dt_step_fn = self._build_dt_step_fn()
        ext = external_inputs

        t_end_jax = jnp.array(t_end)
        dt_min_jax = jnp.array(dt_min)
        dt_max_jax = jnp.array(dt_max)

        def scan_body(carry, _unused):
            state, t, dt, n_accepted = carry

            # Clamp dt to not overshoot
            dt = jnp.minimum(dt, t_end_jax - t)
            dt = jnp.maximum(dt, dt_min_jax)

            # Check if we've already reached t_end
            done = t >= t_end_jax

            # Full step + two half-steps
            state_full = dt_step_fn(state, ext, dt)
            half_dt = dt / 2.0
            state_half = dt_step_fn(state, ext, half_dt)
            state_half = dt_step_fn(state_half, ext, half_dt)

            # Error estimate
            user_full = {k: v for k, v in state_full.items() if k != _META_KEY}
            user_half = {k: v for k, v in state_half.items() if k != _META_KEY}
            error_norm = _tree_error_norm(user_half, user_full, config.atol, config.rtol)

            accepted = (error_norm <= 1.0) | (dt <= dt_min_jax)

            # PI controller
            safe_error = jnp.maximum(error_norm, 1e-10)
            factor = config.safety * jnp.power(1.0 / safe_error, 1.0 / (config.order + 1))
            factor = jnp.clip(factor, config.min_factor, config.max_factor)
            dt_next = jnp.clip(dt * factor, dt_min_jax, dt_max_jax)

            # If done, keep state unchanged; if accepted, use half-step result
            new_state = jax.tree.map(
                lambda s, h: jnp.where(done, s, jnp.where(accepted, h, s)),
                state, state_half,
            )
            new_t = jnp.where(done, t, jnp.where(accepted, t + dt, t))
            new_dt = jnp.where(done, dt, dt_next)
            new_n = jnp.where(done, n_accepted, jnp.where(accepted, n_accepted + 1, n_accepted))

            # Output the state for history (will be no-op state if not accepted)
            output_state = self._user_state(new_state)

            return (new_state, new_t, new_dt, new_n), output_state

        init_carry = (
            self._state,
            jnp.array(0.0),
            jnp.array(dt_initial),
            jnp.array(0, dtype=jnp.int32),
        )

        (final_state, final_t, final_dt, n_accepted), history = jax.lax.scan(
            scan_body, init_carry, None, length=max_steps
        )

        self._state = final_state
        info = {"n_steps": n_accepted, "final_t": final_t, "final_dt": final_dt}
        return self._user_state(final_state), history, info

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_node_state(self, name: str) -> dict:
        if name not in self._nodes:
            raise KeyError(f"No node named '{name}'.")
        if name not in self._state:
            raise KeyError(f"No node named '{name}'.")
        return self._state[name]

    def set_node_state(self, name: str, state: dict) -> None:
        if name not in self._nodes:
            raise KeyError(f"No node named '{name}'.")
        self._state[name] = state

    # ------------------------------------------------------------------
    # Observer pattern
    # ------------------------------------------------------------------

    def add_observer(self, callback: Callable) -> None:
        """Register a callback.  Called as ``callback(event, data)``."""
        self._observers.append(callback)

    def _notify(self, event: str, data: Any = None) -> None:
        for cb in self._observers:
            cb(event, data)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the graph structure (not runtime state)."""
        return {
            "nodes": [spec.node.to_dict() for spec in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
            "external_inputs": [
                {
                    "target_node": ei.target_node,
                    "target_field": ei.target_field,
                    "shape": list(ei.shape),
                }
                for ei in self._external_inputs
            ],
        }

    @classmethod
    def from_dict(
        cls,
        config: dict,
        node_registry: dict[str, type],
    ) -> "GraphManager":
        """Reconstruct a GraphManager from a serialised config.

        *node_registry* maps node type names (e.g. ``"BallNode"``) to
        the corresponding class.
        """
        gm = cls()
        for nd in config["nodes"]:
            node_cls = node_registry[nd["type"]]
            node = node_cls(name=nd["name"], timestep=nd["timestep"], **nd.get("params", {}))
            gm.add_node(node)
        for ed in config["edges"]:
            gm.add_edge(
                source=ed["source_node"],
                target=ed["target_node"],
                source_field=ed["source_field"],
                target_field=ed["target_field"],
            )
        for ei in config.get("external_inputs", []):
            gm.add_external_input(
                target_node=ei["target_node"],
                target_field=ei["target_field"],
                shape=tuple(ei.get("shape", ())),
            )
        return gm

    # ------------------------------------------------------------------
    # Checkpoint / restore
    # ------------------------------------------------------------------

    def save_state(self, path) -> "Path":
        """Save all node states to an ``.npz`` file.

        See :func:`maddening.core.checkpoint.save_state` for details.
        """
        from maddening.core.simulation.checkpoint import save_state
        return save_state(self, path)

    def load_state(self, path) -> None:
        """Load node states from an ``.npz`` file.

        See :func:`maddening.core.checkpoint.load_state` for details.
        """
        from maddening.core.simulation.checkpoint import load_state
        load_state(self, path)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def timestep(self) -> float:
        """Return the base timestep (GCD of all node timesteps).

        For uniform-rate graphs this is the common timestep.  For
        multi-rate graphs this is the smallest step at which the
        compiled function advances.
        """
        timesteps = sorted({spec.timestep for spec in self._nodes.values()})
        if not timesteps:
            raise RuntimeError("No nodes registered.")
        if len(timesteps) == 1:
            return timesteps[0]
        return _multi_gcd(timesteps)

    @property
    def is_multirate(self) -> bool:
        """Whether the graph has nodes with different timesteps."""
        return self._is_multirate

    @property
    def rate_dividers(self) -> dict[str, int]:
        """Per-node rate divider (node_dt / base_dt, rounded).

        Only meaningful after :meth:`compile`.
        """
        return dict(self._rate_dividers)

    @property
    def base_timestep(self) -> float:
        """Alias for :attr:`timestep`."""
        return self.timestep

    @property
    def node_names(self) -> list[str]:
        return list(self._nodes.keys())

    @property
    def schedule(self) -> list[str]:
        return list(self._schedule)

    def __repr__(self) -> str:
        n = len(self._nodes)
        e = len(self._edges)
        ei = len(self._external_inputs)
        compiled = "compiled" if not self._dirty else "dirty"
        parts = [f"{n} nodes", f"{e} edges"]
        if ei:
            parts.append(f"{ei} external inputs")
        if self._is_multirate:
            parts.append("multi-rate")
        return f"GraphManager({', '.join(parts)}, {compiled})"
