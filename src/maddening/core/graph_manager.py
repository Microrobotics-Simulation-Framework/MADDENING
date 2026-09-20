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
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# ``lineax`` is a base dependency (v0.4.0) but is still imported lazily
# inside ``_ift_linear_solve``: it pulls in equinox + jaxtyping, an order
# of magnitude more import time than ``import maddening`` itself costs.
# Only users who opt into ``solver='ift'`` pay it.  The import needs no
# guard — a missing lineax is now an installation fault, not a
# user-recoverable "install the extra" condition.

from maddening.core.coupling import CouplingGroup, coupling_group_kwargs
from maddening.core.coupling.acceleration import (
    float_fields_of,
    relaxation_step_scale,
    state_float_image,
    state_from_float_image,
)
from maddening.core.edge import EdgeSpec
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.node import SimulationNode
from maddening.core.params import (
    ParamSpec,
    check_bounds as _check_bounds,
    constrain as _constrain,
    trainable_mask as _trainable_mask,
    unconstrain as _unconstrain,
)
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
    # True when ``update`` declares a ``params`` keyword: the graph then
    # passes the node's entry of the graph parameter pytree on every
    # call (traced, differentiable).  Nodes that don't opt in keep the
    # 3-argument contract and read constants from ``self.params``.
    accepts_params: bool = False
    # Same for ``compute_boundary_fluxes``: a flux producer that reads
    # its constants from ``params`` gets the node's pytree entry on
    # every flux evaluation (a calibrated stiffness changes the force a
    # flux edge delivers).
    flux_accepts_params: bool = False


@dataclass(frozen=True)
class _StepPlan:
    """The derived values ``_build_step_fn`` closes over.

    ``compile()`` recomputes all of them and can still raise afterwards
    (the ``accelerated_fields`` validation, the static-data refusal,
    ``_build_step_fn`` itself), so they travel as a plan and are written
    onto the graph only at the commit point at the end of a successful
    compile.  A failed ``compile()`` leaves the graph exactly as it was:
    anything reading ``_schedule``, ``_is_multirate``, ``_rate_dividers``
    or :attr:`~GraphManager.params` as a description of the step that is
    actually running would otherwise read a step that was never built.
    """
    schedule: list[str]
    back_edges: list[EdgeSpec]
    is_multirate: bool
    rate_dividers: dict[str, int]
    params: dict


def _correction_accepts_params(node: SimulationNode) -> bool:
    fn = getattr(node, "compute_interface_correction", None)
    if fn is None:
        return False
    try:
        return "params" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _flux_accepts_params(node: SimulationNode) -> bool:
    fn = getattr(node, "compute_boundary_fluxes", None)
    if fn is None:
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return "params" in sig.parameters


def _node_fluxes(spec: _NodeSpec, state, boundary_inputs, dt, node_params):
    """``spec.node.compute_boundary_fluxes`` with the params contract."""
    if spec.flux_accepts_params and node_params is not None:
        return spec.node.compute_boundary_fluxes(
            state, boundary_inputs, dt, params=node_params,
        )
    return spec.node.compute_boundary_fluxes(state, boundary_inputs, dt)


def _update_accepts_params(node: SimulationNode) -> bool:
    probe = getattr(node, "accepts_params", None)
    if callable(probe):
        return bool(probe())
    # Duck-typed node objects that don't subclass SimulationNode.
    try:
        sig = inspect.signature(node.update)
    except (TypeError, ValueError):
        return False
    return "params" in sig.parameters


def _node_update(spec: _NodeSpec, state, boundary_inputs, dt, node_params):
    # named_scope is trace-time metadata only (no runtime cost); the
    # profiler's trace attribution keys device kernels on it.
    with jax.named_scope(f"node:{spec.node.name}"):
        if spec.accepts_params and node_params is not None:
            return spec.update_fn(state, boundary_inputs, dt, params=node_params)
        return spec.update_fn(state, boundary_inputs, dt)


class _ResolvedParams(NamedTuple):
    """The graph parameter pytree split for the step builders: per-node
    pytrees (``nodes[name]``) and per-edge mapping weights
    (``mappings[edge.key]``).  Both are traced when an explicit ``params``
    is passed, baked constants otherwise."""
    nodes: dict
    mappings: dict


def _interface_state_fields(edges, group_nodes, state) -> Optional[dict]:
    """Per-node state fields to accelerate for a coupling group.

    The fields the group's internal edges *read* from each producer.  An
    edge whose ``source_field`` is not a state field (a boundary flux
    from ``compute_boundary_fluxes``) is a function of the producer's
    state, so the producer's whole state stands in for it.  ``None``
    when no internal edge exists (accelerate everything).
    """
    ifields: dict[str, set] = {}
    for edge in edges:
        if edge.source_node in group_nodes and edge.target_node in group_nodes:
            fields = state.get(edge.source_node, {})
            if edge.source_field in fields:
                ifields.setdefault(edge.source_node, set()).add(edge.source_field)
            else:
                ifields.setdefault(edge.source_node, set()).update(fields.keys())
    return {nn: tuple(sorted(fs)) for nn, fs in ifields.items()} if ifields else None


def _strong_typed(tree):
    """Strip JAX weak typing from every array leaf.

    ``jnp.array(0.0)`` is *weak-typed*; the same leaf after one step is
    strongly typed (it is the result of arithmetic with typed arrays),
    and a jitted step keyed on ``(shape, dtype, weak_type)`` retraces —
    once per leaf whose weak type flips, typically on the second and
    third steps of every run (measured: three compiles of the same step
    on the MIME AR4 graph, ~2.6 s).  Normalising the seed state and the
    values callers hand in keeps the trace signature constant.
    """
    def _fix(x):
        if getattr(x, "weak_type", False):
            return x.astype(x.dtype)
        return x
    return jax.tree.map(_fix, tree)


def _apply_edge(edge: EdgeSpec, value, params):
    """Mapping (interface transfer) first, then the scalar transform."""
    if edge.mapping is not None:
        weights = None
        if params is not None:
            weights = params.mappings.get(edge.key)
        with jax.named_scope("edge:mapping"):
            value = edge.mapping.apply(value, weights)
    if edge.transform is not None:
        value = edge.transform(value)
    return value


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

    def to_dict(self) -> dict:
        """Serialise for :meth:`GraphManager.to_dict`.

        ``dtype`` is written by name (``"int32"``, ``"float32"``).  It
        used to be left out, so an ``int32`` input reloaded as
        ``float32`` — a node using it as an index then failed on the
        reloaded graph, and one doing arithmetic with it got a different
        trace.  Every field of this dataclass has a slot here, and
        ``tests/core/test_external_input_serialisation.py`` asserts that from
        ``dataclasses.fields`` so the next field added cannot be dropped
        silently.
        """
        return {
            "target_node": self.target_node,
            "target_field": self.target_field,
            "shape": list(self.shape),
            "dtype": jnp.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "ExternalInputSpec":
        """Rebuild from :meth:`to_dict`.

        ``dtype`` is optional: a config written before it was recorded
        reloads at ``float32``, which is what such a graph got then.
        """
        return cls(
            target_node=config["target_node"],
            target_field=config["target_field"],
            shape=tuple(config.get("shape", ())),
            dtype=(
                jnp.dtype(config["dtype"]) if config.get("dtype") is not None
                else jnp.float32
            ),
        )


# ------------------------------------------------------------------
# Implicit-function-theorem fixed-point solver
# ------------------------------------------------------------------
#
# The functions below implement the "deep equilibrium" / IFT
# differentiation pattern for coupling-group fixed points.  They are
# defined at module scope so neither the forward nor the backward path
# closes over any JAX tracer — this is the key constraint that lets
# ``jax.grad(jax.jit(gm.step))`` flow correctly through the custom_jvp
# rule (see optimistix's ``_implicit_impl`` / its ``_is_global_function``
# assertion for the same pattern, and JAX issue #2912 for the
# DynamicJaxprTracer-as-constant failure mode when this rule is
# violated).
#
# ``_F_dispatch`` is *the* one-iteration function; it is invoked from
# a top-level signature ``(x, consts)`` where ``consts`` is a pytree
# of tracers extracted by ``jax.closure_convert`` at the call site.


def _bound_helpers():
    # Lazy: ``maddening.core.coupling.acceleration`` imports jax at
    # module scope and ``graph_manager`` is imported eagerly.
    from maddening.core.coupling.acceleration import (  # noqa: PLC0415
        error_amplification,
        estimated_error,
        relaxation_step_scale,
    )
    return error_amplification, estimated_error, relaxation_step_scale


def _F_dispatch(step_pure, x, consts):
    # Trampoline: forwards to a closure-converted pure function and
    # keeps only the next iterate.  Kept top-level so the custom_jvp
    # rule sees ``step_pure`` as a plain Python global, not a captured
    # closure.
    return step_pure(x, *consts)[0]


#: Accelerations that may not *stop iterating* on a single pass at or
#: below the threshold: the criterion has to hold on two consecutive
#: passes.  See :func:`_fixed_point_while` for the argument, and for
#: why IQN is not here.  Both coupling solvers read this list and both
#: seed the streak with the residual of the pass before their loop, so
#: ``"ift"`` and ``"fori"`` stop at the same pass at every cap.
_TWO_PASS_EXIT = ("aitken",)


def _fixed_point_while(
    step_pure, x0, consts, accel_init, first_res, threshold, max_iter,
    acceleration, relaxation, n_reuse, sub_idx,
):
    """Early-exit fixed-point iteration ``x = F(x)`` with acceleration.

    ``step_pure(x, *consts) -> (F(x), residual)`` is the closure-converted
    one-pass function; ``residual`` is the group's configured convergence
    measure of ``F(x)`` against ``x`` (L2 / mixed / interface norm).

    **The criterion is an error estimate, not a residual test.**  What
    is compared against the static ``threshold`` is not ``r_k`` but an
    estimate of the distance to the fixed point,
    ``omega * r_k / (1 - rho)``, with ``rho = r_k / r_{k-1}`` taken from
    the two residuals the loop already has (see
    :func:`~maddening.core.coupling.acceleration.error_amplification`)
    and ``omega`` the ratio of the step the iterate takes to the
    residual that is measured (see
    :func:`~maddening.core.coupling.acceleration.relaxation_step_scale`).
    ``converged=True`` therefore means "estimated within ``threshold``
    of the fixed point" rather than "the last step was smaller than
    ``threshold``" — the gap MADD-ANO-005 recorded.  The estimate is
    never smaller than ``r_k``, so this criterion is never looser than
    the raw one it replaces: a group that stops here would have
    stopped under the old rule too, possibly later.

    **Why "estimate" and not "bound".**  ``omega * r_k / (1 - rho)`` is
    the sum of a geometric series of remaining step lengths.  That sum
    bounds the distance only if **four** conditions hold.  Three of them
    can fail undetected; only the fourth is checked:

    1. *The measure is not a metric.*  Summing step lengths bounds the
       distance only under the triangle inequality, and the 0.4.0
       measures divide each field's change by that field's own
       magnitude — a scale that depends on the pair being compared.
       The inequality fails when the iterate detours through a state
       orders of magnitude larger than its neighbours (pinned by
       ``test_the_triangle_inequality_does_not_hold``).  That is the
       price of units-invariance and it was paid deliberately.  The
       estimate is rigorous where the iterate's scale is stable across
       the tail, which is the regime a converging iteration is in.
    2. *``rho`` reads the mode that dominates the step, not the mode
       that dominates the remaining error.*  On a linear two-mode
       contraction the residual sequence is a clean geometric decay at
       the *fast* rate for as long as the fast mode's amplitude
       dominates, even though the distance still to travel is already
       owned by the slow one.  Measured: modes ``(0.999, 0.2)`` at
       ``tolerance=1e-4`` report ``9.19e-05`` against a true distance
       of ``1.12e-02``, a 122x understatement, with ``ratio_usable``
       and ``converged`` both true.  The two-step ``sqrt`` guard in
       ``error_amplification`` reads the same fast rate and does not
       help, and no test on the residual sequence can: the sequence is
       indistinguishable from a single-mode decay at 0.2 until the slow
       mode emerges.  The same mechanism, inverted, is why IQN
       understates — a superlinear sequence reads ``rho -> 0``.
    3. *A dynamic step scale.*  ``omega`` above is exact for
       ``acceleration="fixed"``, where relaxation is a constant, and
       for ``"none"``, where it is 1.  It is *not* corrected for
       Aitken's clipped per-pass factor (a measured 2.04x
       understatement when it saturates at 2.0) or for the IQN
       quasi-Newton step, which is not a multiple of ``F(x) - x``.
    4. *A non-monotone ratio* — the one that is caught.  ``rho >= 1``,
       a zero predecessor or a non-finite residual reject the estimate
       and ``ratio_usable`` records it.

    ``ratio_usable`` therefore reports exactly condition 4 and nothing
    else: a usable *ratio*, not a valid *bound*.  That is why it is no
    longer called ``bound_valid`` — that name asserted all four while
    checking one.  The old key still reads through 0.4.x and warns; see
    ``benchmarks/results/audit_040_final/ERROR_BOUND_DECISION.md``.

    On a non-monotone sequence the ratio is meaningless, so it is
    rejected (``rho >= 1``, a zero predecessor, a non-finite residual)
    and the raw residual test stands in, with
    ``coupling_diagnostics()['ratio_usable']`` recording that it did.
    When ``rho`` approaches 1 the bound diverges, which is the honest
    answer — a group that is barely contracting *is* far from its fixed
    point — and it costs iterations that the old criterion did not
    charge.  The loop always runs at
    least one body iteration and at most ``max_iter - 1``, so the number
    of state updates at the cap (first pass + body iterations) equals
    ``max_iter`` — the same budget as the legacy fori path.  An exit
    that did *not* meet the criterion evaluates ``F`` once more to
    measure what it is returning (see ``final_res`` below); that
    evaluation updates nothing, and the fori path pays it too.

    **On a criterion exit the state returned is the iterate that
    passed**, not the one it went on to produce.  Every pass measures
    the iterate it starts from, so the pass that satisfies the
    criterion has already computed a successor by the time the loop
    stops; that successor is discarded.  ``converged=True`` therefore
    means "the residual of the state you were handed is at or below
    ``threshold``" rather than "some earlier iterate passed", and the
    fori path — whose ``_merge`` keeps ``s_cur`` on the converging
    pass — returns the identical state.  The discarded successor is
    nearer the fixed point, but nothing ever measured it, and a
    non-monotone (non-normal) group can put it well outside the
    tolerance the flag just claimed.  Keeping it and re-measuring was
    the alternative; it costs one evaluation of ``F`` per converged
    group per step and still leaves the two solvers returning
    different states.  See ``plans/MADDENING_040_DECISIONS.md`` D1/D2.

    ``first_res`` is the residual of the pass that produced ``x0`` —
    the one ``_run_coupling_inner`` ran before this loop.  It seeds the
    two-consecutive-passes streak, which is what lets the guard below
    be *provable* at ``max_iter == 2``, where the loop itself only gets
    to measure one residual.  The fori path seeds its streak from the
    same quantity, so both solvers stop at the same pass.

    ``sub_idx`` (static tuple of ints, or None) restricts the
    acceleration to a subset of ``x`` — the IQN interface fields — while
    the iterate, the residual and the fixed point stay the full vector.
    Non-accelerated entries take the raw ``F(x)`` value each iteration,
    matching the fori path's ``_build_accel_state``.

    **Aitken must meet the threshold twice** (``_TWO_PASS_EXIT``).
    Under a constant iterator — ``none``, or ``fixed`` at any
    relaxation — the iterate advances by one fixed linear operator and
    the residual sequence is asymptotically monotone, so one value at
    or below ``threshold`` is evidence the iteration has arrived.
    Aitken re-derives a scalar relaxation factor from each pair of
    residuals and clips it to ``[0.01, 2.0]``.  When its
    single-dominant-mode assumption fails — a degenerate or partly
    divergent Jacobi spectrum — the factor saturates alternately at
    both bounds and the residual sequence stops being monotone: it
    dips two decades below its own trend for a single pass, while the
    iterate has barely moved, and springs back on the next.  Stopping
    on such a dip returns a state far from the fixed point, and since
    the dip undershoots any plausible threshold, tightening
    ``tolerance`` does not move the exit either.  Aitken therefore
    *stops* only when two consecutive passes are at or below
    ``threshold``: a genuine arrival pays one extra pass and a
    transient dip is rejected.  The guard is on the exit only — what
    is *reported* is a measurement of the state being handed back (see
    ``final_res`` below), which is the one number a dip cannot
    flatter.

    IQN is deliberately *not* on that list.  Its step comes from a
    least-squares solve over an accumulating secant basis, not from a
    clipped scalar, and it converges superlinearly — a large one-pass
    drop is the method working, not a dip.  Across the 18 coupling
    sweep fixtures every ``iqn-ils`` / ``iqn-imvj`` row converges in
    2-4 iterations at a converged fraction of 1.0, with no measured
    instance of the Aitken pathology, so charging it a mandatory
    second pass would cost 30-50% of its iteration budget against no
    evidence.  Its Aitken fallback (no secant columns yet) is covered
    by the fix to ``aitken_relaxation``'s zero-seed.  If an IQN
    residual sequence is ever measured dipping, add the name to
    ``_TWO_PASS_EXIT``.

    Returns ``(x_star, n_iters, final_res, final_amp, (V, W))``:
    ``n_iters`` is the number of coupling passes that produced
    ``x_star`` (as a float, for the diagnostics carry) -- the pre-loop
    pass plus the bodies whose update it kept, which is the count the
    fori path reports and is ``max_iter`` exactly at the cap, not the
    bare body count -- ``final_amp`` the amplification
    ``1/(1 - rho)`` of the pair of residuals that ends on ``x_star``
    (``0.0`` where the estimate was rejected), and ``(V, W)`` the IQN
    secant matrices (an empty tuple for other accelerations).

    ``final_res`` is the number ``coupling_diagnostics``' ``converged``
    flag and ``strict_convergence`` are derived from, so it has to be
    a statement about ``x_star`` and not about some iterate before it.
    Every pass measures the residual of the iterate it starts from, so
    the loop's own last measurement describes the iterate the last
    body started from.  When the loop leaves on its criterion that
    iterate *is* ``x_star`` (see above), so the loop's own measurement
    is already a statement about what is being returned and nothing
    further is evaluated.  When it leaves because ``max_iter`` ran
    out, ``x_star`` is the successor instead (an Aitken step routinely
    arrives on the pass that had no successor), so one extra
    evaluation of ``F`` measures ``x_star`` itself and *that* is what
    is reported.  The extra pass is charged only on the exit that was
    about to report failure, and it is also the second opinion the
    two-pass guard wanted: a residual that dipped for one pass springs
    back here, on the very state the caller is being handed.  Both
    coupling solvers report this same quantity by the same rule, and
    both now return the same state as well.

    No autodiff machinery here; the IFT rule is layered on by
    ``_ift_solve``.

    Acceleration wrappers (static ``acceleration``):

    - ``"none"``   : ``x_{k+1} = F(x_k)``.
    - ``"fixed"``  : constant relaxation ``x + relaxation * (F(x) - x)``.
    - ``"aitken"`` : Aitken delta-squared relaxation.
    - ``"iqn-ils"`` / ``"iqn-imvj"``: interface quasi-Newton via
      ``iqn_ils_update`` (shift-and-insert secant columns, Aitken
      fallback).  ``accel_init = (V, W)`` seeds the secant matrices —
      zeros for ILS, the previous timestep's columns masked to the first
      ``n_reuse`` for IMVJ — so cross-timestep Jacobian reuse runs inside
      the while_loop with the same column convention as the fori path.
    """
    from maddening.core.coupling.acceleration import (  # noqa: PLC0415
        aitken_relaxation,
        fixed_relaxation,
        iqn_ils_update,
    )

    idx = None if sub_idx is None else jnp.asarray(sub_idx, dtype=jnp.int32)
    x0_acc = x0 if idx is None else x0[idx]
    n_dof = x0_acc.shape[0]
    dtype = x0.dtype
    zeros = jnp.zeros(n_dof, dtype=dtype)
    one = jnp.array(1.0, dtype=dtype)
    is_iqn = acceleration in ("iqn-ils", "iqn-imvj")

    if acceleration in ("none", "fixed"):
        acc0 = ()
    elif acceleration == "aitken":
        acc0 = (one, zeros)  # omega, prev_residual
    elif is_iqn:
        V0, W0 = accel_init
        # V, W, n_cols, prev_residual, prev_raw, omega, prev_r_aitken
        acc0 = (V0, W0, jnp.int32(n_reuse), zeros, x0_acc, one, zeros)
    else:
        raise ValueError(
            f"_fixed_point_while: unsupported acceleration="
            f"{acceleration!r}; expected one of 'none', 'fixed', "
            "'aitken', 'iqn-ils', 'iqn-imvj'."
        )

    def accelerate(x, x_raw, acc, i):
        with jax.named_scope("coupling:accelerate"):
            return _accelerate(x, x_raw, acc, i)

    def _accelerate(x, x_raw, acc, i):
        if acceleration == "none":
            return x_raw, acc
        if acceleration == "fixed":
            return fixed_relaxation(x, x_raw, relaxation), acc
        if acceleration == "aitken":
            omega, prev_r = acc
            x_new, omega, cur_r = aitken_relaxation(x, x_raw, prev_r, omega)
            return x_new, (omega, cur_r)
        V, W, n_cols, prev_r, prev_s, omega, prev_ra = acc
        x_new, V, W, n_cols, cur_r, cur_s, omega, cur_ra = iqn_ils_update(
            x_raw, x, prev_r, prev_s, V, W, n_cols, omega, prev_ra,
            have_prev=i > 0,
        )
        return x_new, (V, W, n_cols, cur_r, cur_s, omega, cur_ra)

    # See the docstring for why this list holds Aitken and not IQN.
    # An empty ``prev`` slot means the carry -- and so the emitted HLO
    # -- is unchanged for every acceleration that is not on it.
    two_pass_exit = acceleration in _TWO_PASS_EXIT

    amplification, error_of, step_scale_of = _bound_helpers()
    # Static: ``acceleration`` and ``relaxation`` are both nondiff
    # arguments of the custom_jvp, so this is a Python float and costs
    # nothing in the loop.
    step_scale = step_scale_of(acceleration, relaxation)

    def _met(res, res_prev, res_prev2):
        """The stopping criterion: the *estimated distance to the fixed
        point* is at or below ``threshold``, not merely the last step."""
        est = error_of(res, amplification(res, res_prev, res_prev2),
                       step_scale)
        met = est <= threshold
        if two_pass_exit:
            # The streak the Aitken guard wants, with the current pass
            # held to the estimate and its predecessor to the raw
            # residual it was able to measure.  ``est >= res`` always,
            # so this is strictly stronger than the pair of raw tests
            # it replaces.
            met = jnp.logical_and(met, res_prev <= threshold)
        return met

    def cond(carry):
        _x, _x_meas, res, res_prev, res_prev2, i, _acc = carry
        first = i == jnp.int32(0)
        above = jnp.logical_not(_met(res, res_prev, res_prev2))
        keep_going = jnp.logical_and(above, i < max_iter - 1)
        return jnp.logical_or(first, keep_going)

    def body(carry):
        x, _x_meas, res_prev, res_prev2, _res_prev3, i, acc = carry
        x_raw, res = step_pure(x, *consts)
        if idx is None:
            x_new, acc = accelerate(x, x_raw, acc, i)
        else:
            x_new_sub, acc = accelerate(x[idx], x_raw[idx], acc, i)
            x_new = x_raw.at[idx].set(x_new_sub)
        # ``x`` is the iterate ``res`` is a measurement of; carrying it
        # is what lets a criterion exit hand back the state its own
        # criterion passed on.  See the docstring.  ``res_prev`` is now
        # carried for every acceleration, not only the two-pass ones:
        # the error bound needs the ratio of consecutive residuals.
        return x_new, x, res, res_prev, res_prev2, i + jnp.int32(1), acc

    # The seed is the residual of the pass that produced ``x0``, not
    # ``inf``: the streak the guard tests then has a first member even
    # when the loop only runs one body (``max_iter == 2``), so the
    # guard is provable at every cap instead of being switched off at
    # the smallest one.  ``cond`` forces the first body regardless, so
    # this value only ever reaches the criterion as the ``r_{k-1}``
    # of the first ratio.
    seed = jnp.asarray(first_res, dtype)
    init = (x0, x0, seed, seed, seed, jnp.int32(0), acc0)
    (x_next, x_meas, final_res, res_prev, res_prev2, n_iters,
     acc) = jax.lax.while_loop(cond, body, init)
    # Did the loop leave on its criterion, or because it ran out of
    # passes?
    #
    # On the criterion, ``final_res`` is a true statement
    # (``<= threshold``) about ``x_meas``, the iterate the last body
    # started from -- so ``x_meas`` is what is returned, and the flag
    # derived from ``final_res`` describes it exactly.  ``x_next``, one
    # update further along, is discarded: it is nearer the fixed point
    # but nothing measured it, and the fori path has always made the
    # same choice (``_merge`` keeps ``s_cur`` on the pass that
    # converged).  The two solvers therefore return the same state.
    #
    # At the cap the last measurement is a statement about an iterate
    # the caller never sees, and the Aitken step that produced
    # ``x_next`` is exactly the one that most often crossed the
    # threshold -- so ``x_next`` is returned and one evaluation of
    # ``F`` measures it.  See the docstring.
    criterion_met = _met(final_res, res_prev, res_prev2)
    x_star = jnp.where(criterion_met, x_meas, x_next)
    loop_res = final_res

    def _measure_at_cap(_x):
        r = step_pure(_x, *consts)[1]
        return r, amplification(r, loop_res, res_prev)

    # ``final_amp`` has to describe the pair that ends on the state
    # being returned.  On the criterion that pair is
    # ``(res_prev, final_res)``; at the cap the extra evaluation makes
    # ``(final_res, res(x_star))`` the consecutive pair instead.
    final_res, final_amp = jax.lax.cond(
        criterion_met,
        lambda _x: (loop_res, amplification(loop_res, res_prev, res_prev2)),
        _measure_at_cap,
        x_star,
    )
    vw = (acc[0], acc[1]) if is_iqn else ()
    # ``iterations`` counts the coupling passes that produced the state
    # being returned, which is what the fori path has always reported
    # and what ``coupling_diagnostics`` promises does not move when a
    # graph migrates.  ``n_iters`` counts loop bodies, and the two
    # differ by exactly one exit: body ``k`` measures ``x_{k-1}`` (the
    # product of ``k`` passes, counting the pre-loop one) and produces
    # ``x_k``.  A criterion exit returns ``x_meas = x_{k-1}``, so
    # ``n_iters`` is already the pass count; the cap returns ``x_next =
    # x_k``, one pass further along.  Reporting ``n_iters`` at the cap
    # published ``max_iterations - 1`` there, so the documented
    # ``iterations >= max_iterations`` cap check never fired under the
    # default solver.
    n_passes = jnp.where(criterion_met, n_iters, n_iters + jnp.int32(1))
    return x_star, n_passes.astype(dtype), final_res, final_amp, vw


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
    or at ``max_iter``.  A finite difference of this function's own
    output is therefore *not* the quantity the adjoint computes: it is
    the derivative of a truncated iterate, and the two agree only as
    ``final_res`` goes to zero.  Tighten the criterion that is live
    for the group's norm (``tolerance`` for ``"l2"``; ``atol`` /
    ``rtol`` for ``"mixed"`` and ``"interface"``, whose threshold is
    hard-coded to 1.0) if the gradient has to match the forward.

    ``acceleration`` / ``relaxation`` / ``n_reuse`` are static and
    control only the forward iterator; the derivative is identical for
    all of them because the IFT rule depends on ``F`` at ``x*``, not on
    the path taken to reach ``x*``.  ``linear_solver`` selects the
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

#: Raised (through ``equinox.error_if``, at runtime inside jit) when a
#: Krylov adjoint solve fails on a group too large to re-solve densely.
#: It replaces lineax's own message, whose "increase ``restart``"
#: remedy does not address the mechanism — see ``_ift_linear_solve``.
_ADJOINT_SOLVE_FAILED_MSG = (
    "MADDENING: the coupling adjoint solve did not converge "
    "(linear_solver={solver!r}, {n} coupled DOF).  This is usually an "
    "ill-conditioned (I - dF/dx): cond(A) ~ 1/(1 - rho) in the group's "
    "slowest contraction rate, and float32 cannot resolve the solver's "
    "tolerance once eps*cond(A) exceeds it.  Remedies, in order: pass "
    "linear_solver='dense' to add_coupling_group() (exact, but O(N^2) "
    "memory); set MADDENING_IFT_DENSE_SOLVE=1 to force that globally "
    "for triage; or make the group less stiff (stronger relaxation, a "
    "smaller timestep, or splitting the cycle).  Raising GMRES's "
    "restart will NOT help: it is already min(N, 50)."
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
    ulp of the dtype (1e-6 in float64, 1.2e-5 in float32).  Memory is
    O(N) for the matrix-free backends; no Jacobian is ever
    materialised except under ``"dense"``.

    Backends, dispatched by ``linear_solver`` plus the
    ``MADDENING_IFT_DENSE_SOLVE`` env var (env var wins for triage):

    * ``"gmres"`` (default) — lineax GMRES.  Safe non-symmetric solver.
    * ``"dense"`` — materialise ``A`` with ``jacfwd`` and LU-solve.
      O(N^2) memory, O(N^3) compute.  Triage fallback, promoted to a
      first-class config option.
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
    rtol = max(1e-6, 100.0 * float(jnp.finfo(rhs.dtype).eps))

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

        atol = 1e-8 + rtol * jnp.max(jnp.abs(b))
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
            # bumping the restart cap in lockstep — see
            # tests/core/test_coupling_ift_lineax.py::
            # test_gmres_restart_too_small_silently_corrupts_gradient
            # for the regression guard.
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
        sol = lx.linear_solve(op, b, solver=solver, throw=False)
        failed = jnp.logical_not(sol.result == lx.RESULTS.successful)
        if n <= _DENSE_ADJOINT_FALLBACK_MAX_DOF:
            return jax.lax.cond(
                failed, lambda bb: _dense(mv, bb), lambda _bb: sol.value, b,
            )
        import equinox as eqx  # noqa: PLC0415  (lineax transitive dep)

        return eqx.error_if(
            sol.value, failed,
            _ADJOINT_SOLVE_FAILED_MSG.format(solver=effective_solver, n=n),
        )

    solve = _dense if effective_solver == "dense" else _krylov
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
    aux_dot = jax.tree.map(jnp.zeros_like, aux)
    return (x_star, aux), (x_dot, aux_dot)


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


# ------------------------------------------------------------------
# Observer event names
# ------------------------------------------------------------------
EVENT_NODE_ADDED = "node_added"
EVENT_NODE_REMOVED = "node_removed"
EVENT_EDGE_ADDED = "edge_added"
EVENT_EDGE_REMOVED = "edge_removed"
EVENT_COMPILED = "compiled"
EVENT_STEP = "step"
# Emitted by maddening.sysid.fit / fit_lm / fit_multiple_shooting.
EVENT_FIT_PROGRESS = "fit_progress"


_EMPTY_EXTERNAL_INPUTS: dict[str, dict] = {}

# Key for internal multi-rate metadata in the full state dict.
_META_KEY = "_meta"


# ------------------------------------------------------------------
# Deprecated ``coupling_diagnostics()`` field names
# ------------------------------------------------------------------

#: 0.4.0 renames, old name -> new name.  Both old names asserted a
#: *bound* the code does not establish; see
#: :class:`_CouplingDiagnostics` and
#: ``benchmarks/results/audit_040_final/ERROR_BOUND_DECISION.md``.
#: Reading through the old name still works through 0.4.x and warns;
#: the old names are removed in 0.5.0.
_DIAGNOSTICS_RENAMES = {
    "bound_valid": "ratio_usable",
    "gradient_error_bound": "gradient_error_estimate",
}

#: Why each name moved, quoted into the warning so a caller does not
#: have to find the memo to learn what it had been reading.
_DIAGNOSTICS_RENAME_REASON = {
    "bound_valid": (
        "the flag checks one of the four conditions the estimate rests "
        "on -- that the contraction ratio was monotone and finite -- "
        "and not that the estimate bounds the error; the worst measured "
        "understatement with it True is 122x"
    ),
    "gradient_error_bound": (
        "it is numerically 'error_estimate' and inherits every way that "
        "number can understate, so it is an estimate and not a bound"
    ),
}


class _CouplingDiagnostics(dict):
    """A per-group diagnostics mapping that still answers the 0.3.x names.

    ``coupling_diagnostics()`` renamed two fields in 0.4.0, because each
    called itself a *bound*:

    * ``bound_valid`` -> ``ratio_usable``
    * ``gradient_error_bound`` -> ``gradient_error_estimate``

    Reading an old name returns the same value and emits a
    :class:`DeprecationWarning`.  The old names are removed in 0.5.0.

    They are deliberately **not** in :meth:`keys`, iteration or
    :func:`len`, so ``dict(diag)``, a JSON dump and anything else that
    enumerates the report carry only the new names: a recorded artefact
    should not preserve a name the next release deletes.

    Only reads are aliased.  Writing, popping or ``setdefault``-ing an
    old name is not forwarded -- this mapping is a report, and a caller
    mutating it is not a compatibility case anyone had.
    """

    __slots__ = ()

    def _resolve(self, key):
        """Map a deprecated key to its replacement, warning; else pass through."""
        new = _DIAGNOSTICS_RENAMES.get(key)
        if new is None or not dict.__contains__(self, new):
            return key
        warnings.warn(
            f"coupling_diagnostics()[{key!r}] is deprecated: "
            f"{_DIAGNOSTICS_RENAME_REASON[key]}.  Use {new!r}, which "
            f"carries the same value.  The old name is removed in 0.5.0.",
            DeprecationWarning, stacklevel=3,
        )
        return new

    def __getitem__(self, key):
        return dict.__getitem__(self, self._resolve(key))

    def get(self, key, default=None):
        return dict.get(self, self._resolve(key), default)

    def __contains__(self, key):
        return dict.__contains__(self, self._resolve(key))


def _holds_tracer(state: dict) -> bool:
    """Whether *state* came out of a JAX transform rather than a run.

    One node is enough: the state of a graph stepped under a transform
    is the transform's output, so a node's fields are tracers together
    or not at all, and so are the nodes.  Scanning the first node's
    fields rather than all of them keeps this off the per-step cost of a
    large graph, while still catching a hand-written partial state whose
    traced field is not the first one (the shape
    ``set_node_state`` is given by a differentiable initial condition).
    """
    for value in state.values():
        if isinstance(value, dict):
            return any(isinstance(leaf, jax.core.Tracer) for leaf in value.values())
        return isinstance(value, jax.core.Tracer)
    return False


def _outside_jax_trace() -> bool:
    """Whether no JAX transform is currently active.

    Best effort, on a private JAX helper: when it is not there the
    answer is "cannot tell", which every caller reads as "do not
    intervene".  Being wrong in that direction costs the old behaviour
    (JAX's own ``UnexpectedTracerError`` later), never a wrong number.
    """
    try:
        from jax._src import core as _jax_core
        return bool(_jax_core.trace_state_clean())
    except Exception:            # pragma: no cover - JAX internals moved
        return False


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
                               node_obj, coupled_bi_names=None, node_params=None):
    """Correct interface DOFs after update to undo internal BC enforcement.

    A node may enforce Dirichlet BCs by overwriting its boundary cells
    after the update.  When those BCs come from coupling, the overwrite
    destroys the physically meaningful stencil-computed value.  This
    function asks the node to recompute those values via
    ``compute_interface_correction``.

    HeatNode was the motivating case and no longer needs it: since
    0.4.0 it imposes the datum through its ghost cells and never
    overwrites a cell (MADD-ANO-007), so its correction returns what
    ``update`` already produced and applying it is an identity.  The
    hook stays because the contract is about nodes in general, and a
    node that does overwrite is still entitled to it.

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
    with jax.named_scope("coupling:interface_override"):
        if node_params is not None and _correction_accepts_params(node_obj):
            corrections = node_obj.compute_interface_correction(
                pre_state, filtered_bi, dt, params=node_params,
            )
        else:
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
    multigpu_device_map=None, node_params=None,
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
        error_amplification,
        estimated_error,
        fixed_relaxation,
        flatten_coupled_state,
        iqn_ils_update,
        relaxation_step_scale,
        unflatten_coupled_state,
    )

    max_iters = group.max_iterations
    # How much longer the iterate's step is than the residual that is
    # measured -- ``relaxation`` under ``acceleration="fixed"``, 1.0
    # otherwise.  Static, and identical on both solver paths so
    # ``solver`` stays invisible in ``coupling_diagnostics()``.
    step_scale = relaxation_step_scale(group.acceleration, group.relaxation)
    group_node_names = list(group_schedule)
    _node_params = node_params.nodes if node_params is not None else {}

    def _np(nn):
        return _node_params.get(nn)
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
            accel_fields = _interface_state_fields(
                group_internal_list, group.nodes, new_state,
            )
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
    # Float32 *images* of the pre-step states: ``one_pass`` closes over
    # them, and an integer / boolean / PRNG-key leaf hoisted by
    # ``closure_convert`` into the IFT custom_jvp's constants cannot be
    # linearised under a ``lax.scan`` (see ``_run_ift_forward``).  The
    # images are bit-exact for every supported dtype (wide integers
    # travel as 16-bit limbs, keys as their uint32 data); ``_pre(nn)``
    # restores the leaves at the point of use.
    _init_imgs, _init_metas = {}, {}
    for nn in group_node_names:
        _init_imgs[nn], _init_metas[nn] = state_float_image(new_state[nn])
    initial_node_states = _init_imgs

    def _pre(nn):
        return state_from_float_image(initial_node_states[nn], _init_metas[nn])

    def _get_dt(nn):
        spec = nodes[nn]
        return runtime_dt if runtime_dt is not None else spec.timestep

    _MISSING = object()

    def _resolve_value(edge, src_state, flux_s, strict=True):
        """Get value from state or flux dict."""
        src_nn = edge.source_node
        src_dict = src_state.get(src_nn, {})
        if edge.source_field in src_dict:
            return src_dict[edge.source_field]
        if flux_s and src_nn in flux_s and edge.source_field in flux_s[src_nn]:
            return flux_s[src_nn][edge.source_field]
        if not strict:
            return _MISSING
        # Fall back (will KeyError if truly missing)
        return src_state[src_nn][edge.source_field]

    def _resolve_boundary(nn, s, flux_s=None, strict=True):
        """Resolve boundary inputs for node nn from state s.

        ``strict=False`` omits inputs whose flux is not available yet
        (used to seed fluxes from the previous iterate before a pass).
        """
        boundary_inputs: dict[str, Any] = {}
        for edge in edges_by_target[nn]:
            if edge in back_edge_set and edge not in group_internal:
                src_state = full_state
            else:
                src_state = s
            value = _resolve_value(edge, src_state, flux_s, strict=strict)
            if value is _MISSING:
                continue
            value = _apply_edge(edge, value, node_params)
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
            value = _apply_edge(edge, value, node_params)
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
        init_sub_state = _pre(nn)

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
            new_sub = _node_update(nodes[nn], sub_state, bi, sub_dt, _np(nn))
            new_sub = _apply_interface_overrides(
                new_sub, sub_state, bi, sub_dt, nodes[nn].node,
                coupled_bi_names=coupled_bi_names_by_node.get(nn),
                    node_params=_np(nn),
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
        if has_flux_edges:
            # A flux consumer scheduled *before* its producer reads the
            # producer's flux from the previous iterate (the fixed-point
            # semantics); once the producer updates below, its entry is
            # overwritten for the nodes that follow it.  Without this a
            # back-edge on a flux field raised KeyError in the first pass.
            # Two sweeps: producers may need each other's fluxes, so the
            # first sweep tolerates missing ones, the second has them all.
            for strict in (False, True):
                for nn in group_node_names:
                    if nn in flux_producing_nodes:
                        bi0 = _resolve_boundary(nn, latest_results, flux_s, strict=strict)
                        flux_s[nn] = _node_fluxes(
                            nodes[nn], latest_results[nn], bi0, _get_dt(nn), _np(nn),
                        )
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
                pre = _pre(nn)
                s[nn] = _node_update(nodes[nn], pre, bi, _get_dt(nn), _np(nn))
                s[nn] = _apply_interface_overrides(
                    s[nn], pre, bi, _get_dt(nn), nodes[nn].node,
                    coupled_bi_names=coupled_bi_names_by_node.get(nn),
                    node_params=_np(nn),
                )
            # Compute fluxes for this node
            if nn in flux_producing_nodes:
                bi_for_flux = _resolve_boundary(nn, s, flux_s)
                flux_s[nn] = _node_fluxes(
                    nodes[nn], s[nn], bi_for_flux, _get_dt(nn), _np(nn),
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
                    flux_s[nn] = _node_fluxes(
                        nodes[nn], latest_results[nn], bi, _get_dt(nn), _np(nn),
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
                pre = _pre(nn)
                if multigpu_device_map is not None and nn in multigpu_device_map:
                    dev_idx = multigpu_device_map[nn]
                    devices = jax.devices()
                    if dev_idx < len(devices):
                        device = devices[dev_idx]
                        pre = jax.device_put(pre, device)
                        bi = jax.tree.map(
                            lambda x: jax.device_put(x, device), bi,
                        )
                results[nn] = _node_update(nodes[nn], pre, bi, _get_dt(nn), _np(nn))
                results[nn] = _apply_interface_overrides(
                    results[nn], pre, bi, _get_dt(nn), nodes[nn].node,
                    coupled_bi_names=coupled_bi_names_by_node.get(nn),
                    node_params=_np(nn),
                )
        s = {k: v for k, v in latest_results.items()}
        for nn in group_node_names:
            s[nn] = results[nn]
        return s

    one_pass = one_pass_jacobi if use_jacobi else one_pass_gs

    def _compute_residual(s_new, s_old):
        with jax.named_scope("coupling:residual"):
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
            return coupling_residual_l2(
                s_new, s_old, group_node_names, group.atol,
            )

    def _estimate(residual, prev_residual, prev2_residual):
        """``(estimated distance to the fixed point, amplification)``.

        The criterion both solvers stop on.  See
        :func:`_fixed_point_while` for the argument and for what
        happens when the ratio is rejected.
        """
        amp = error_amplification(residual, prev_residual, prev2_residual)
        return estimated_error(residual, amp, step_scale), amp

    # Convergence threshold depends on norm type
    conv_threshold_value = (
        1.0 if (use_mixed_norm or use_interface_norm)
        else float(group.tolerance)
    )
    conv_threshold = jnp.array(conv_threshold_value)

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
            # ``max_iterations=1`` is a legitimate "one staggered pass,
            # no iteration" request, so it reports like any other cap
            # rather than being refused.  Returning ``diag_data=None``
            # here used to leave the ``_meta`` entries at the values
            # ``compile()`` seeded them with (iterations 0, residual
            # 0.0), which ``coupling_diagnostics()`` then reads as
            # ``converged=True`` whatever the state, and which
            # ``strict_convergence`` could never contradict because the
            # check lives in ``_run_ift_forward``.  ``single_r`` is the
            # residual of the state the single pass started from: how
            # far that pass moved.  Every larger cap reports the
            # residual of the state it *returns*, measuring it with one
            # extra evaluation of ``F`` when its criterion was not met
            # (see ``_fixed_point_while``); a cap of one is the
            # exception, because it is the one setting that is a
            # request about cost -- one staggered pass has to cost one
            # pass, and the profiler's one-iteration variant depends on
            # it.  So this cap reports its single measurement.
            #
            # ``solver="ift"`` never reaches ``_ift_solve`` here, so the
            # gradient is straight through the one pass rather than the
            # implicit-function derivative of a fixed point.  That is
            # the only derivative available -- one pass defines no fixed
            # point to differentiate -- and it is what ``"fori"`` gives
            # too, which is why the solvers still agree.  Documented on
            # ``CouplingGroup.max_iterations``; ``strict_convergence``
            # is checked below so the caller still hears about it.
            single_r = _compute_residual(state_after_first, new_state_inner)
            # One pass means one residual and no ratio, so there is no
            # error bound to be had: the amplification is reported
            # rejected (``0.0``) and the criterion falls back to the
            # raw residual test.  A caller who wants the bound has to
            # allow the group a second pass.
            single_amp = jnp.zeros_like(single_r)
            sub = {nn: state_after_first[nn] for nn in group_node_names}
            if group.strict_convergence and group.solver == "ift":
                import equinox as eqx  # noqa: PLC0415

                sub = eqx.error_if(
                    # ``not (r <= t)``, not ``r > t``: a NaN residual
                    # answers False to *both* comparisons, so the
                    # second form lets the one state the IFT gradient
                    # is certainly invalid at through silently.  See
                    # the ift branch below.
                    sub,
                    jnp.logical_not(
                        estimated_error(single_r, single_amp, step_scale)
                        <= conv_threshold_value
                    ),
                    f"coupling group {sorted(group.nodes)} exited at "
                    f"max_iterations={max_iters} without converging; "
                    "the IFT gradient is invalid here. Raise "
                    "max_iterations, loosen the tolerance, or set "
                    "strict_convergence=False to only report this via "
                    "coupling_diagnostics().",
                )
            r = {k: v for k, v in new_state_inner.items()}
            for nn in group_node_names:
                r[nn] = sub[nn]
            # One coupling pass ran, so report one: a residual was
            # measured, and ``iterations=0`` beside a non-zero residual
            # would contradict itself.  ``solver="fori"`` keeps its own
            # ``diagnostics=True`` gate.
            if group.solver == "ift" or group.diagnostics:
                return r, (jnp.array(1.0), single_r, single_amp), None
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

        def _iqn_warm_start():
            """Seed the IQN secant matrices.  Returns ``(V, W, n_reuse)``.

            Zeros for IQN-ILS; for IQN-IMVJ the previous timestep's
            columns from ``_meta``, masked to the first
            ``jacobian_reuse`` so stale columns beyond the reuse window
            do not enter the secant solve.
            """
            max_cols = max(max_iters - 1, 1)
            if group.acceleration != "iqn-imvj":
                return (jnp.zeros((n_dof, max_cols)),
                        jnp.zeros((n_dof, max_cols)), 0)
            group_key = "+".join(sorted(group.nodes))
            meta = new_state_inner.get(_META_KEY, {})
            stored_V = meta.get(
                f"coupling_{group_key}_V", jnp.zeros((n_dof, max_cols)),
            )
            stored_W = meta.get(
                f"coupling_{group_key}_W", jnp.zeros((n_dof, max_cols)),
            )
            n_reuse = min(group.jacobian_reuse, max_cols)
            reuse_mask = jnp.arange(max_cols) < n_reuse
            return (stored_V * reuse_mask[None, :],
                    stored_W * reuse_mask[None, :], n_reuse)

        def _run_ift_forward(template_state):
            """Run the early-exit while_loop solver; return ``(state, diag, vw)``.

            ``template_state`` is the post-first-pass full state dict
            (``state_after_first``).  The solver iterates on the
            flattened *full* state of the group's nodes (every field,
            like the fori path), embedded back into ``template_state``
            for ``one_pass`` so boundary resolution can see nodes
            outside the group.  IQN acceleration acts on the
            interface-field subset through a static index map into
            that vector.  ``diag`` is ``(n_iters, final_res)`` and
            ``vw`` the IQN ``(V, W)`` matrices (``None`` for other
            accelerations).  The IFT derivative is intrinsic to ``F``
            at ``x*`` and unchanged across acceleration modes.
            """
            # We need a top-level ``step_pure(x, *consts)`` so the
            # custom_jvp rule does not close over any tracer (see JAX
            # issue #2912 / optimistix's _is_global_function
            # assertion).  jax.closure_convert hoists any tracers
            # ``one_pass`` captures — including the outside-node
            # entries of ``template_state`` — into an explicit
            # ``consts`` pytree, so the IFT rule propagates
            # derivatives through them.
            # Only floating fields live in the fixed-point vector.  An
            # integer / boolean field (a counter, a flag) is recomputed
            # from the pre-step state on every pass, so its first-pass
            # value is already the converged one; keeping it out avoids
            # float<->int casts in the loop and float0 tangents in the
            # IFT rule (which leaked tracers under reverse mode through
            # a scan).
            float_fields = {
                nn: tuple(
                    f for f in sorted(template_state[nn])
                    if jnp.issubdtype(template_state[nn][f].dtype, jnp.floating)
                )
                for nn in group_node_names
            }

            # ``jax.closure_convert`` hoists every tracer ``_step_flat``
            # touches into the custom_jvp's constants.  An *integer or
            # boolean* constant there breaks JAX's linearisation of the
            # rule under a ``lax.scan`` (UnexpectedTracerError in reverse
            # mode, a missing constant handler in forward mode; reproduced
            # on JAX 0.10 / 0.11 with a bare custom_jvp + closure_convert).
            # So the closure only ever sees bit-exact float32 *images*
            # of such leaves (``float_image``: 16-bit limbs for wide
            # integers, uint32 data for PRNG keys), restored inside.
            leaf_metas: dict = {}
            template_img: dict = {}
            for k, d in template_state.items():
                if isinstance(d, dict):
                    template_img[k], leaf_metas[k] = state_float_image(d)
                else:
                    template_img[k] = d

            def _flatten_full(s):
                return flatten_coupled_state(s, group_node_names, fields=float_fields)

            def _embed(x_full):
                part = unflatten_coupled_state(
                    x_full, template_img, group_node_names, fields=float_fields,
                )
                s = {}
                for k, d in template_img.items():
                    if isinstance(d, dict):
                        s[k] = state_from_float_image(d, leaf_metas[k])
                    else:
                        s[k] = d
                for nn in group_node_names:
                    s[nn] = {**s[nn], **part[nn]}
                return s

            def _step_flat(x_full):
                s = _embed(x_full)
                s_new = one_pass(s)
                # The residual is the group's configured norm on the
                # full per-node state, exactly as the fori path
                # computes it.
                return _flatten_full(s_new), _compute_residual(s_new, s)

            x0_full = _flatten_full(template_state)
            step_pure, consts_list = jax.closure_convert(
                _step_flat, x0_full
            )
            consts = tuple(consts_list)

            if accel_fields is not None:
                # Positions of the accelerated (interface) fields in
                # the full flat vector: flatten an index-valued state
                # of the same structure, restricted to those fields.
                idx_state = unflatten_coupled_state(
                    np.arange(int(x0_full.shape[0]), dtype=np.int32),
                    template_img, group_node_names, fields=float_fields,
                )
                # Pure numpy: the same node/field order as
                # ``flatten_coupled_state(..., fields=accel_fields)``,
                # but without going through jnp (which would trace).
                parts = [
                    np.ravel(np.asarray(idx_state[nn][fld]))
                    for nn in group_node_names if nn in accel_fields
                    for fld in sorted(accel_fields[nn])
                    if fld in float_fields[nn]
                ]
                sub_idx = tuple(int(i) for i in np.concatenate(parts))
            else:
                sub_idx = None

            if group.acceleration in ("iqn-ils", "iqn-imvj"):
                init_V, init_W, n_reuse = _iqn_warm_start()
                accel_init = (init_V, init_W)
            else:
                accel_init, n_reuse = (), 0

            x_star_full, (n_iters, final_res, final_amp, vw) = _ift_solve(
                step_pure, x0_full, consts, accel_init,
                jnp.asarray(first_r, x0_full.dtype),
                conv_threshold_value,
                int(max_iters),
                group.acceleration,
                float(group.relaxation),
                int(n_reuse),
                sub_idx,
                str(group.linear_solver),
            )
            if group.strict_convergence:
                # Lazy for import time only: equinox is a transitive
                # dependency of lineax, which is a base dependency.
                import equinox as eqx  # noqa: PLC0415

                x_star_full = eqx.error_if(
                    # ``not (r <= t)`` rather than ``r > t``: the two
                    # differ exactly on NaN, which answers False to
                    # both, and a NaN residual is the one case where
                    # the IFT gradient is certainly invalid.  It is
                    # also reachable *because* of the measurement
                    # above: a solve that overflowed reports ``inf``
                    # from the pass before the cap, but ``inf - inf``
                    # -- NaN -- when the state it returns is measured.
                    # ``coupling_diagnostics()`` already reads NaN as
                    # ``converged=False``; the guard has to agree.
                    x_star_full,
                    jnp.logical_not(
                        estimated_error(final_res, final_amp, step_scale)
                        <= conv_threshold_value
                    ),
                    f"coupling group {sorted(group.nodes)} exited at "
                    f"max_iterations={max_iters} without converging; "
                    "the IFT gradient is invalid here. Raise "
                    "max_iterations, loosen the tolerance, or set "
                    "strict_convergence=False to only report this via "
                    "coupling_diagnostics().",
                )
            final = _merge(template_state, _embed(x_star_full), jnp.array(False))
            return final, (n_iters, final_res, final_amp), (vw if vw else None)

        if group.solver == "ift":
            (final_state, (iter_count, final_res, final_amp),
             vw) = _run_ift_forward(state_after_first)
            r = {k: v for k, v in new_state_inner.items()}
            for nn in group_node_names:
                r[nn] = final_state[nn]
            # Always reported (not only with ``diagnostics=True``): the
            # scalars are already in the carry, and the converged flag
            # is what tells a training loop that the IFT gradient
            # through this step is trustworthy.
            diag_data = (iter_count, final_res, final_amp)
            return r, diag_data, vw

        # ---- Legacy unrolled fori_loop path (``solver="fori"``,
        # deprecated).  Runs ``max_iterations`` passes regardless of
        # convergence, freezing the state once converged, and
        # differentiates straight through the iterates. ----

        if group.acceleration == "aitken":
            # Aitken is in ``_TWO_PASS_EXIT``, so it latches
            # ``converged`` -- and so freezes the state -- only after
            # two consecutive passes at or below the threshold; see
            # ``_fixed_point_while`` for why one is not evidence.  Here
            # the loop runs ``max_iterations`` passes whatever happens,
            # so the second pass costs nothing.  ``first_r`` is the
            # residual of the pass before the loop, which is the right
            # seed for the streak.
            first_below = first_r <= conv_threshold

            if track_diag:
                def body_fn(i, carry):
                    (s_cur, converged, prev_below, prev_res, prev_res2,
                     icount, fres, famp, omega, prev_r) = carry
                    s_raw = one_pass(s_cur)
                    residual = _compute_residual(s_raw, s_cur)
                    below = residual <= conv_threshold
                    est, amp = _estimate(residual, prev_res, prev_res2)
                    new_converged = converged | (
                        (est <= conv_threshold) & prev_below
                    )
                    x_old = _flatten(s_cur)
                    x_raw = _flatten(s_raw)
                    x_rel, new_omega, cur_r = aitken_relaxation(
                        x_old, x_raw, prev_r, omega
                    )
                    s_partial = _unflatten(x_rel, s_cur)
                    s_accel = _build_accel_state(s_raw, s_partial)
                    s_merged = _merge(s_cur, s_accel, new_converged)
                    new_count = icount + jnp.where(new_converged, 0.0, 1.0)
                    new_res = jnp.where(converged, fres, residual)
                    new_amp = jnp.where(converged, famp, amp)
                    return (s_merged, new_converged, below, residual,
                            prev_res, new_count, new_res, new_amp,
                            new_omega, cur_r)

                init_carry = (
                    state_after_first, jnp.array(False), first_below,
                    first_r, first_r,
                    jnp.array(1.0), first_r, jnp.zeros_like(first_r),
                    jnp.array(1.0), jnp.zeros(n_dof),
                )
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]
                iter_count, final_res = final_carry[5], final_carry[6]
                final_amp, prev_loop_res = final_carry[7], final_carry[4]
            else:
                def body_fn(i, carry):
                    (s_cur, converged, prev_below, prev_res, prev_res2,
                     omega, prev_r) = carry
                    s_raw = one_pass(s_cur)
                    residual = _compute_residual(s_raw, s_cur)
                    below = residual <= conv_threshold
                    est, _amp = _estimate(residual, prev_res, prev_res2)
                    new_converged = converged | (
                        (est <= conv_threshold) & prev_below
                    )
                    x_old = _flatten(s_cur)
                    x_raw = _flatten(s_raw)
                    x_rel, new_omega, cur_r = aitken_relaxation(
                        x_old, x_raw, prev_r, omega
                    )
                    s_partial = _unflatten(x_rel, s_cur)
                    s_accel = _build_accel_state(s_raw, s_partial)
                    s_merged = _merge(s_cur, s_accel, new_converged)
                    return (s_merged, new_converged, below, residual,
                            prev_res, new_omega, cur_r)

                init_carry = (
                    state_after_first, jnp.array(False), first_below,
                    first_r, first_r,
                    jnp.array(1.0), jnp.zeros(n_dof),
                )
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]

        elif group.acceleration in ("iqn-ils", "iqn-imvj"):
            init_V, init_W, n_reuse = _iqn_warm_start()
            init_ncols = jnp.int32(n_reuse)
            init_flat = _flatten(state_after_first)

            if track_diag:
                def body_fn(i, carry):
                    (s_cur, converged, prev_res, prev_res2, icount, fres,
                     famp, V, W, nc, prev_r, prev_s, omega, prev_ra) = carry
                    s_raw = one_pass(s_cur)
                    residual = _compute_residual(s_raw, s_cur)
                    est, amp = _estimate(residual, prev_res, prev_res2)
                    new_converged = converged | (est <= conv_threshold)
                    x_old = _flatten(s_cur)
                    x_raw = _flatten(s_raw)
                    (x_new, nV, nW, nnc,
                     cur_r, cur_s, n_omega, cur_ra) = iqn_ils_update(
                        x_raw, x_old, prev_r, prev_s,
                        V, W, nc, omega, prev_ra,
                        have_prev=i > 1,
                    )
                    s_partial = _unflatten(x_new, s_cur)
                    s_accel = _build_accel_state(s_raw, s_partial)
                    s_merged = _merge(s_cur, s_accel, new_converged)
                    new_count = icount + jnp.where(new_converged, 0.0, 1.0)
                    new_res = jnp.where(converged, fres, residual)
                    new_amp = jnp.where(converged, famp, amp)
                    return (s_merged, new_converged, residual, prev_res,
                            new_count, new_res, new_amp,
                            nV, nW, nnc, cur_r, cur_s, n_omega, cur_ra)

                init_carry = (
                    state_after_first, jnp.array(False), first_r, first_r,
                    jnp.array(1.0), first_r, jnp.zeros_like(first_r),
                    init_V, init_W, init_ncols,
                    jnp.zeros(n_dof), init_flat,
                    jnp.array(1.0), jnp.zeros(n_dof),
                )
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]
                iter_count, final_res = final_carry[4], final_carry[5]
                final_amp, prev_loop_res = final_carry[6], final_carry[3]
                final_V, final_W = final_carry[7], final_carry[8]
            else:
                def body_fn(i, carry):
                    (s_cur, converged, prev_res, prev_res2,
                     V, W, nc, prev_r, prev_s, omega, prev_ra) = carry
                    s_raw = one_pass(s_cur)
                    residual = _compute_residual(s_raw, s_cur)
                    est, _amp = _estimate(residual, prev_res, prev_res2)
                    new_converged = converged | (est <= conv_threshold)
                    x_old = _flatten(s_cur)
                    x_raw = _flatten(s_raw)
                    (x_new, nV, nW, nnc,
                     cur_r, cur_s, n_omega, cur_ra) = iqn_ils_update(
                        x_raw, x_old, prev_r, prev_s,
                        V, W, nc, omega, prev_ra,
                        have_prev=i > 1,
                    )
                    s_partial = _unflatten(x_new, s_cur)
                    s_accel = _build_accel_state(s_raw, s_partial)
                    s_merged = _merge(s_cur, s_accel, new_converged)
                    return (s_merged, new_converged, residual, prev_res,
                            nV, nW, nnc, cur_r, cur_s, n_omega, cur_ra)

                init_carry = (
                    state_after_first, jnp.array(False), first_r, first_r,
                    init_V, init_W, init_ncols,
                    jnp.zeros(n_dof), init_flat,
                    jnp.array(1.0), jnp.zeros(n_dof),
                )
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]
                final_V, final_W = final_carry[4], final_carry[5]

        elif group.acceleration == "fixed":
            omega_val = group.relaxation

            if track_diag:
                def body_fn(i, carry):
                    (s_cur, converged, prev_res, prev_res2, icount, fres,
                     famp) = carry
                    s_raw = one_pass(s_cur)
                    residual = _compute_residual(s_raw, s_cur)
                    est, amp = _estimate(residual, prev_res, prev_res2)
                    new_converged = converged | (est <= conv_threshold)
                    x_old = _flatten(s_cur)
                    x_raw = _flatten(s_raw)
                    x_rel = fixed_relaxation(x_old, x_raw, omega_val)
                    s_partial = _unflatten(x_rel, s_cur)
                    s_accel = _build_accel_state(s_raw, s_partial)
                    s_merged = _merge(s_cur, s_accel, new_converged)
                    new_count = icount + jnp.where(new_converged, 0.0, 1.0)
                    new_res = jnp.where(converged, fres, residual)
                    new_amp = jnp.where(converged, famp, amp)
                    return (s_merged, new_converged, residual, prev_res,
                            new_count, new_res, new_amp)

                init_carry = (
                    state_after_first, jnp.array(False), first_r, first_r,
                    jnp.array(1.0), first_r, jnp.zeros_like(first_r),
                )
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]
                iter_count, final_res = final_carry[4], final_carry[5]
                final_amp, prev_loop_res = final_carry[6], final_carry[3]
            else:
                def body_fn(i, carry):
                    s_cur, converged, prev_res, prev_res2 = carry
                    s_raw = one_pass(s_cur)
                    residual = _compute_residual(s_raw, s_cur)
                    est, _amp = _estimate(residual, prev_res, prev_res2)
                    new_converged = converged | (est <= conv_threshold)
                    x_old = _flatten(s_cur)
                    x_raw = _flatten(s_raw)
                    x_rel = fixed_relaxation(x_old, x_raw, omega_val)
                    s_partial = _unflatten(x_rel, s_cur)
                    s_accel = _build_accel_state(s_raw, s_partial)
                    s_merged = _merge(s_cur, s_accel, new_converged)
                    return s_merged, new_converged, residual, prev_res

                init_carry = (state_after_first, jnp.array(False),
                              first_r, first_r)
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]

        else:
            # No acceleration ("none")
            if track_diag:
                def body_fn(i, carry):
                    (s_cur, converged, prev_res, prev_res2, icount, fres,
                     famp) = carry
                    s_new = one_pass(s_cur)
                    residual = _compute_residual(s_new, s_cur)
                    est, amp = _estimate(residual, prev_res, prev_res2)
                    new_converged = converged | (est <= conv_threshold)
                    s_merged = _merge(s_cur, s_new, new_converged)
                    new_count = icount + jnp.where(new_converged, 0.0, 1.0)
                    new_res = jnp.where(converged, fres, residual)
                    new_amp = jnp.where(converged, famp, amp)
                    return (s_merged, new_converged, residual, prev_res,
                            new_count, new_res, new_amp)

                init_carry = (
                    state_after_first, jnp.array(False), first_r, first_r,
                    jnp.array(1.0), first_r, jnp.zeros_like(first_r),
                )
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]
                iter_count, final_res = final_carry[4], final_carry[5]
                final_amp, prev_loop_res = final_carry[6], final_carry[3]
            else:
                def body_fn(i, carry):
                    s_cur, converged, prev_res, prev_res2 = carry
                    s_new = one_pass(s_cur)
                    residual = _compute_residual(s_new, s_cur)
                    est, _amp = _estimate(residual, prev_res, prev_res2)
                    new_converged = converged | (est <= conv_threshold)
                    s_merged = _merge(s_cur, s_new, new_converged)
                    return s_merged, new_converged, residual, prev_res

                init_carry = (state_after_first, jnp.array(False),
                              first_r, first_r)
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]

        # Report the residual of the state being handed back, by the
        # same rule as the ift path (see ``_fixed_point_while``): once
        # ``converged`` latches, ``_merge`` freezes the state on the
        # iterate that was measured, so the in-loop number already
        # describes what is returned; at the cap it does not, so that
        # exit -- and only that one -- pays one more evaluation of
        # ``F``.  ``converged`` is index 1 of every branch's carry.
        # Keeping the rule identical on both solvers is what makes
        # ``solver`` invisible in ``coupling_diagnostics()``.
        if track_diag:
            loop_res = final_res

            def _measure_at_cap(_s):
                r = _compute_residual(one_pass(_s), _s)
                return r, error_amplification(r, loop_res, prev_loop_res)

            final_res, final_amp = jax.lax.cond(
                final_carry[1],
                lambda _s: (loop_res, final_amp),
                _measure_at_cap,
                final_state,
            )

        # Merge coupled nodes back into the full state
        r = {k: v for k, v in new_state_inner.items()}
        for nn in group_node_names:
            r[nn] = final_state[nn]

        # Write diagnostics to _meta if requested
        diag_data = None
        if track_diag:
            diag_data = (iter_count, final_res, final_amp)

        vw_data = None
        if group.acceleration in ("iqn-ils", "iqn-imvj"):
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

            # Only apply if we have at least 2 stored states.  Only the
            # floating fields are extrapolated; a counter or flag keeps
            # its first-pass value.
            has_history = pred_count >= 2
            pred_fields = float_fields_of(new_state, group_node_names)
            x_cur = flatten_coupled_state(new_state, group_node_names, fields=pred_fields)
            x_use = jnp.where(has_history, x_pred, x_cur)

            # Unflatten and update new_state with predicted values
            predicted = unflatten_coupled_state(
                x_use, new_state, group_node_names, fields=pred_fields,
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
        converged_flat = flatten_coupled_state(
            result, group_node_names, fields=float_fields_of(result, group_node_names),
        )
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

    # Write diagnostics to _meta.  Always under solver="ift" *when the
    # incoming state already carries ``_meta`` (compile() pre-populates
    # it, keeping the pytree structure stable across scan); a state
    # built by hand without ``_meta`` keeps its structure.  The legacy
    # fori path only reports with diagnostics=True.
    if diag_data is not None and (group.diagnostics or _META_KEY in full_state):
        iter_count, final_res, final_amp = diag_data
        result.setdefault(_META_KEY, {})
        result[_META_KEY] = {
            **result.get(_META_KEY, {}),
            f"coupling_{group_key}_iterations": jnp.array(
                iter_count, dtype=jnp.int32
            ),
            f"coupling_{group_key}_residual": final_res,
            f"coupling_{group_key}_amplification": jnp.asarray(
                final_amp, dtype=jnp.asarray(final_res).dtype
            ),
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


# How many built scan programs one graph keeps between compiles.
_SCAN_CACHE_MAX = 64


def _build_adaptive_scan(
    dt_step_fn: Callable,
    user_state: Callable[[dict], dict],
    max_steps: int,
    safety: float,
    order: int,
    min_factor: float,
    max_factor: float,
    on_trace: Callable[[], None],
) -> Callable:
    """Build the jitted adaptive-timestepping scan for ``run_adaptive_scan``.

    Kept at module level (and built through
    :meth:`GraphManager._cached_scan`) so the program is compiled once
    per graph compile rather than once per call.  ``t_end``, the
    tolerances and the timestep bounds arrive as the traced ``knobs``
    tuple, so changing any of them reuses the compilation; the PI
    controller's constants are closed over and therefore belong in the
    cache key.

    Parameters
    ----------
    dt_step_fn : callable
        ``(state, external_inputs, dt, params) -> state``.
    user_state : callable
        Strips the internal ``_meta`` key from a state dict.
    max_steps : int
        Scan length.
    safety, order, min_factor, max_factor
        PI step-size controller constants.
    on_trace : callable
        Called once per Python trace of the program, for
        :attr:`GraphManager.scan_trace_count`.

    Returns
    -------
    Callable
        ``(state, ext, params, knobs) -> ((state, t, dt, n), history)``.
    """
    from maddening.core.simulation.adaptive import _tree_error_norm

    def adaptive_scan(init_state, ext, params, knobs):
        on_trace()
        t_end, dt_initial, atol, rtol, dt_min, dt_max = knobs

        def scan_body(carry, _unused):
            state, t, dt, n_accepted = carry

            # Clamp dt to not overshoot
            dt = jnp.minimum(dt, t_end - t)
            dt = jnp.maximum(dt, dt_min)

            # Check if we've already reached t_end
            done = t >= t_end

            # Full step + two half-steps
            state_full = dt_step_fn(state, ext, dt, params)
            half_dt = dt / 2.0
            state_half = dt_step_fn(state, ext, half_dt, params)
            state_half = dt_step_fn(state_half, ext, half_dt, params)

            # Error estimate
            user_full = {k: v for k, v in state_full.items() if k != _META_KEY}
            user_half = {k: v for k, v in state_half.items() if k != _META_KEY}
            error_norm = _tree_error_norm(user_half, user_full, atol, rtol)

            accepted = (error_norm <= 1.0) | (dt <= dt_min)

            # PI controller
            safe_error = jnp.maximum(error_norm, 1e-10)
            factor = safety * jnp.power(1.0 / safe_error, 1.0 / (order + 1))
            factor = jnp.clip(factor, min_factor, max_factor)
            dt_next = jnp.clip(dt * factor, dt_min, dt_max)

            # If done, keep state unchanged; if accepted, use half-step result
            new_state = jax.tree.map(
                lambda s, h: jnp.where(done, s, jnp.where(accepted, h, s)),
                state, state_half,
            )
            new_t = jnp.where(done, t, jnp.where(accepted, t + dt, t))
            new_dt = jnp.where(done, dt, dt_next)
            new_n = jnp.where(
                done, n_accepted, jnp.where(accepted, n_accepted + 1, n_accepted),
            )

            # Output the state for history (no-op state if not accepted)
            output_state = user_state(new_state)

            return (new_state, new_t, new_dt, new_n), output_state

        init_carry = (
            init_state,
            jnp.array(0.0),
            dt_initial,
            jnp.array(0, dtype=jnp.int32),
        )
        return jax.lax.scan(scan_body, init_carry, None, length=max_steps)

    return jax.jit(adaptive_scan)


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
        # Bumped by every ``compile()``.  The scan cache keys on it, so a
        # cached scan is exactly as fresh as ``_compiled_step``: every
        # graph mutation marks the graph dirty, every public entry point
        # recompiles a dirty graph, and the recompile invalidates the
        # cache.
        self._compile_generation: int = 0
        # ``jax.lax.scan`` programs built by ``run_scan`` and its
        # siblings, keyed by ``_cached_scan``.
        self._scan_cache: dict[tuple, Callable] = {}
        self._n_scan_traces: int = 0
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
        # Differentiable graph parameters — the third pytree of the
        # compiled step, next to state and external inputs.  Refreshed
        # from the nodes on every compile; edit in place (or pass
        # ``params=`` to step/run_scan) to change constants without a
        # recompile, and differentiate with respect to it for
        # calibration / system identification.
        self.params: dict = {"nodes": {}, "mappings": {}}
        # Graph-level ParamSpec overrides: {node: {key: ParamSpec}}.
        self._param_spec_overrides: dict[str, dict[str, ParamSpec]] = {}
        # Escaped-tracer bookkeeping; see ``_recover_from_escaped_tracers``.
        self._state_traced = False
        self._state_before_trace: Optional[dict] = None
        # The rate dividers of the step that is actually compiled, as
        # opposed to ``_rate_dividers``, which ``compile`` overwrites on
        # its way through and leaves behind if it raises.  This is what
        # the sub-step phase in ``_meta`` is indexed by.
        self._committed_rate_dividers: dict[str, int] = {}

    def _snapshot_params(self) -> dict:
        return {
            "nodes": {
                name: spec.node.params_pytree()
                for name, spec in self._nodes.items()
                if spec.accepts_params
            },
            "mappings": {
                edge.key: edge.mapping.params_pytree()
                for edge in self._edges
                if edge.mapping is not None
            },
        }

    @staticmethod
    def _merge_live_params(fresh: dict, live: Optional[dict]) -> dict:
        """``fresh`` (constructor snapshot) with every leaf of ``live`` that
        still fits written over it; leaves that no longer fit warn."""
        if not live:
            return fresh
        dropped = []
        for section in ("nodes", "mappings"):
            fresh_sec = fresh.setdefault(section, {})
            for owner, leaves in (live.get(section) or {}).items():
                if owner not in fresh_sec:
                    if leaves:
                        dropped.append(f"{section}[{owner!r}]")
                    continue
                for key, value in leaves.items():
                    base = fresh_sec[owner].get(key)
                    if base is None:
                        dropped.append(f"{section}[{owner!r}][{key!r}]")
                        continue
                    base_dtype = jnp.asarray(base).dtype
                    if jnp.shape(value) != jnp.shape(base):
                        dropped.append(f"{section}[{owner!r}][{key!r}] (shape changed)")
                        continue
                    # A Python float assigned into gm.params is float64
                    # under x64 and weak-typed otherwise: coerce to the
                    # leaf's own dtype (strongly typed) so it is kept and
                    # the jitted step is not retraced.
                    fresh_sec[owner][key] = jnp.asarray(value, dtype=base_dtype)
        if dropped:
            warnings.warn(
                "compile() dropped live gm.params leaves that no longer fit "
                f"the graph: {dropped}", RuntimeWarning, stacklevel=3,
            )
        return fresh

    def _check_param_shapes(self, section: str, owner: str, leaves: dict) -> None:
        """A leaf of the wrong shape would broadcast the node's *state* to
        that shape for good; shapes are static, so this is free."""
        expected = (getattr(self, "_params_shapes", None) or {}).get(section, {}).get(owner)
        if not expected:
            return
        for key, v in leaves.items():
            want = expected.get(key)
            if want is not None and tuple(jnp.shape(v)) != want:
                raise ValueError(
                    f"params[{section!r}][{owner!r}][{key!r}] has shape "
                    f"{tuple(jnp.shape(v))}, expected {want}"
                )

    def _coerce_params_leaves(self, tree: dict) -> None:
        """In place: non-array / weak-typed leaves -> strongly typed arrays
        of the dtype the leaf had at compile time (float32 fallback)."""
        dtypes = getattr(self, "_params_dtypes", {}) or {}
        for section in ("nodes", "mappings"):
            for owner, leaves in (tree.get(section) or {}).items():
                if not isinstance(leaves, dict):
                    continue
                for key, v in leaves.items():
                    dt = dtypes.get(section, {}).get(owner, {}).get(key)
                    if not hasattr(v, "dtype"):
                        leaves[key] = jnp.asarray(v, dtype=dt or jnp.float32)
                    elif getattr(v, "weak_type", False):
                        leaves[key] = v.astype(v.dtype)
                self._check_param_shapes(section, owner, leaves)

    @property
    def trace_count(self) -> int:
        """How many times the compiled step has been traced since the
        last ``compile()`` (0 before the first step; more than 1 after
        steady state means the step is being retraced).

        This counts ``step`` / ``run`` only.  ``run_scan`` and its
        siblings build their own program around the step and are counted
        by :attr:`scan_trace_count`; a loop that kept recompiling a scan
        used to leave ``trace_count`` at 1 and so look healthy.
        """
        return int(getattr(self, "_n_traces", 0))

    @property
    def scan_trace_count(self) -> int:
        """How many scan programs have been traced since the last
        ``compile()``.

        ``run_scan``, ``run_scan_with_history``, ``run_sweep`` and
        ``run_adaptive_scan`` each build a ``jax.lax.scan`` around the
        step function and cache it (see :meth:`_cached_scan`).  This
        counts the Python traces of those programs -- one per XLA
        compile.  Calling the same entry point again with the same step
        count and the same argument shapes must not increase it.
        """
        return int(getattr(self, "_n_scan_traces", 0))

    def _count_scan_trace(self) -> None:
        """Record one Python trace of a cached scan program."""
        self._n_scan_traces += 1

    def reset_params(self) -> None:
        """Discard live/calibrated values: ``gm.params`` becomes the
        constructor snapshot again (no recompile needed)."""
        self.params = self._snapshot_params()

    def _params_or_default(self, params):
        """``gm.params`` when ``params`` is None; otherwise ``params``
        completed from ``gm.params``: a node or key the caller left out
        keeps its *live* value (not the constructor constant), so a
        partial pytree means "override these" and nothing else."""
        if params is None:
            # A Python scalar assigned into gm.params (``gm.params[...] =
            # 32.0``) would reach the jitted step weak-typed and retrace
            # it; coerce such leaves in place to the leaf dtype recorded
            # at compile time.
            self._coerce_params_leaves(self.params)
            return self.params
        if not isinstance(params, dict):
            return params                  # let _validate_params complain
        out = {}
        for section in ("nodes", "mappings"):
            live_sec = self.params.get(section, {})
            given = params.get(section, {}) or {}
            merged = {owner: dict(leaves) for owner, leaves in live_sec.items()}
            for owner, leaves in given.items():
                if owner in merged and isinstance(leaves, dict):
                    # keep the live leaf's dtype for a Python scalar the
                    # caller hands in (weak types retrace the step)
                    fixed = {
                        k: (jnp.asarray(v, dtype=jnp.asarray(merged[owner][k]).dtype)
                            if k in merged[owner] and not hasattr(v, "dtype") else v)
                        for k, v in leaves.items()
                    }
                    merged[owner] = {**merged[owner], **fixed}
                else:
                    merged[owner] = leaves      # unknown owner: validation reports it
            out[section] = merged
        for k, v in params.items():
            if k not in ("nodes", "mappings"):
                out[k] = v
        return _strong_typed(out)

    def _validate_params(self, params: dict) -> None:
        """Reject a ``params`` pytree that names something the step cannot
        use.  Runs Python-side at trace time (dict keys are static), so
        it costs nothing per step.

        A node that does not take ``params`` is *absent* from
        ``gm.params`` — passing an entry for it would be silently
        ignored, and a gradient with respect to it silently zero — so an
        explicit entry is an error, as is an unknown parameter name.
        """
        nodes = params.get("nodes", {}) if isinstance(params, dict) else None
        if nodes is None:
            raise TypeError(
                "params must be a dict with a 'nodes' entry (see GraphManager.params)"
            )
        for node_name, node_params in nodes.items():
            spec = self._nodes.get(node_name)
            if spec is None:
                raise ValueError(
                    f"params['nodes'] names unknown node {node_name!r}; "
                    f"graph nodes: {sorted(self._nodes)}"
                )
            if not spec.accepts_params:
                raise ValueError(
                    f"params['nodes'][{node_name!r}] given, but "
                    f"{type(spec.node).__name__}.update() takes no 'params' "
                    "keyword: its constants are baked into the trace, so this "
                    "entry would be ignored and any gradient with respect to it "
                    "would be zero.  Migrate the node (declare "
                    "update(self, state, boundary_inputs, dt, *, params=None) "
                    "and read constants from params) or drop the entry."
                )
            known = set(spec.node.params_pytree())
            unknown = set(node_params) - known
            if unknown:
                raise ValueError(
                    f"params['nodes'][{node_name!r}] has unknown key(s) "
                    f"{sorted(unknown)}; {type(spec.node).__name__}.params_pytree() "
                    f"exposes {sorted(known)}"
                )
            missing = known - set(node_params)
            if missing:
                raise ValueError(
                    f"params['nodes'][{node_name!r}] is missing key(s) "
                    f"{sorted(missing)}.  The compiled step needs a complete "
                    "pytree (a missing leaf would silently fall back to the "
                    "constructor constant); pass a partial tree through "
                    "gm.step / gm.run_scan(params=...), which completes it "
                    "from the live gm.params."
                )
            self._check_param_shapes("nodes", node_name, node_params)
        absent = [
            n for n, sp in self._nodes.items() if sp.accepts_params and n not in nodes
        ]
        if absent:
            raise ValueError(
                f"params['nodes'] is missing node(s) {sorted(absent)}.  The "
                "compiled step needs a complete pytree (a missing node would "
                "silently use its constructor constants); pass a partial tree "
                "through gm.step / gm.run_scan(params=...), which completes it "
                "from the live gm.params."
            )
        mapped = {e.key: e for e in self._edges if e.mapping is not None}
        given_maps = params.get("mappings", {}) or {}
        absent_maps = sorted(set(mapped) - set(given_maps))
        if absent_maps:
            raise ValueError(
                f"params['mappings'] is missing edge(s) {absent_maps}; pass a "
                "partial tree through gm.step / gm.run_scan(params=...)."
            )
        for key, weights in given_maps.items():
            edge = mapped.get(key)
            if edge is None:
                raise ValueError(
                    f"params['mappings'] names unknown edge {key!r}; mapped "
                    f"edges: {sorted(mapped)}"
                )
            known = set(edge.mapping.params_pytree())
            unknown = set(weights) - known
            if unknown:
                raise ValueError(
                    f"params['mappings'][{key!r}] has unknown key(s) "
                    f"{sorted(unknown)}; the mapping exposes {sorted(known)}"
                )
            self._check_param_shapes("mappings", key, weights)

    # ------------------------------------------------------------------
    # ParamSpec: trainable mask, bounds, reparametrisation
    # ------------------------------------------------------------------

    def param_specs(self) -> dict:
        """``{"nodes": {name: {key: ParamSpec}}, "mappings": {}}`` mirroring
        :attr:`params`: each node's :meth:`SimulationNode.param_specs`
        with the graph's :meth:`set_param_spec` overrides applied.
        Leaves without an entry use the default (trainable, unbounded)."""
        out: dict = {"nodes": {}, "mappings": {}}
        for name, spec in self._nodes.items():
            if not spec.accepts_params:
                continue
            merged = dict(spec.node.param_specs())
            merged.update(self._param_spec_overrides.get(name, {}))
            out["nodes"][name] = merged
        # Interface-mapping weights are geometry-derived operators, not
        # physical constants: frozen unless a learned edge opts in with
        # ``set_param_spec(edge.key, "H", ParamSpec())``.
        for edge in self._edges:
            if edge.mapping is None:
                continue
            merged = {
                k: ParamSpec(trainable=False, description="interface mapping weights")
                for k in edge.mapping.params_pytree()
            }
            merged.update(self._param_spec_overrides.get(edge.key, {}))
            out["mappings"][edge.key] = merged
        return out

    def set_param_spec(self, node: str, key: str, spec: ParamSpec) -> None:
        """Override one parameter's :class:`ParamSpec` for this graph
        (e.g. freeze a node's ``mass`` when the data cannot identify it,
        or make a mapped edge's weights trainable by passing the edge
        key — ``"<src>.<field>-><tgt>.<field>"`` — as ``node``).

        Specs are optimiser-side metadata, so this does not dirty the
        graph — *unless* ``key`` is one a
        :meth:`~maddening.core.node.SimulationNode.static_data_deps`
        entry names.  Then it decides whether ``compile()`` refuses the
        graph (a static baked from a trainable parameter loses the
        gradient through it), so the verdict has to be taken again."""
        if not isinstance(spec, ParamSpec):
            raise TypeError(f"spec must be a ParamSpec, got {type(spec).__name__}")
        mapped = {e.key: e for e in self._edges if e.mapping is not None}
        if node in mapped:
            known = mapped[node].mapping.params_pytree()
            if key not in known:
                raise KeyError(
                    f"mapping on edge {node!r} has no weight {key!r}; it exposes "
                    f"{sorted(known)}"
                )
            self._param_spec_overrides.setdefault(node, {})[key] = spec
            return
        if node not in self._nodes:
            raise KeyError(f"unknown node {node!r}")
        if not self._nodes[node].accepts_params:
            raise ValueError(
                f"node {node!r} takes no params; nothing to specify"
            )
        if key not in self._nodes[node].node.params_pytree():
            raise KeyError(
                f"node {node!r} has no parameter {key!r}; "
                f"params_pytree() exposes "
                f"{sorted(self._nodes[node].node.params_pytree())}"
            )
        # Asked *before* the override is committed: ``static_data_deps``
        # is a node-supplied method and may raise, and an override stored
        # without the dirty flag that goes with it is the half-applied
        # mutation the atomicity work is about.  It forwards from wrapped
        # nodes, so the outer declaration is enough to know whether this
        # key is load-bearing for the compile-time refusal.
        declared = self._nodes[node].node.static_data_deps() or {}
        dirties = any(key in names for names in declared.values())
        self._param_spec_overrides.setdefault(node, {})[key] = spec
        if dirties:
            self._dirty = True

    def trainable_mask(self, params: Optional[dict] = None) -> dict:
        """``params``-shaped pytree of Python bools (``True`` = an
        optimiser may move the leaf)."""
        return _trainable_mask(self._params_or_default(params), self.param_specs())

    def unconstrain(self, params: Optional[dict] = None) -> dict:
        """Map trainable leaves to unconstrained optimiser coordinates
        (``log`` for positive constants, ``logit`` for intervals); other
        leaves pass through.  Inverse of :meth:`constrain`."""
        return _unconstrain(self._params_or_default(params), self.param_specs())

    def constrain(self, u: dict) -> dict:
        """Map optimiser coordinates back to a physical ``params`` pytree
        (also clips bounded identity leaves)."""
        return _constrain(u, self.param_specs())

    def check_params(self, params: Optional[dict] = None) -> None:
        """Raise ``ValueError`` if any leaf is outside its declared bounds
        or the pytree names a node/key the step cannot use."""
        params = self._params_or_default(params)
        self._validate_params(params)
        _check_bounds(params, self.param_specs())

    def nodes_without_params(self) -> list[str]:
        """Names of nodes whose ``update`` takes no ``params`` keyword —
        their constants are not differentiable through the graph."""
        return [n for n, s in self._nodes.items() if not s.accepts_params]

    def effective_node_params(self, name: str, params: Optional[dict] = None) -> dict:
        """The node's constructor ``params`` with the live values of
        :attr:`params` (or ``params``) written over them, as plain Python
        scalars/lists.  This is what serialisation stores, so a calibrated
        graph reloads with the calibrated constants."""
        spec = self._nodes[name]
        out = dict(spec.node.params)
        live = self._params_or_default(params).get("nodes", {}).get(name, {})
        snapshot = spec.node.params_pytree()
        for key, value in live.items():
            # Only constructor params can be written back; a derived leaf
            # (a surrogate's ``weights['scale']``) is not a constructor
            # argument and would break reconstruction.  Checkpoints carry
            # those.
            if key not in spec.node.params:
                continue
            # Only overlay a leaf that actually changed: the pytree holds
            # float32 promotions of the constructor floats (0.05 ->
            # 0.05000000074505806), and an uncalibrated constant should
            # serialise exactly as it was given.
            base = snapshot.get(key)
            if base is not None and np.array_equal(np.asarray(base), np.asarray(value)):
                continue
            out[key] = np.asarray(value).tolist()
        return out

    def param_spec_overrides(self) -> dict[str, dict[str, ParamSpec]]:
        """Graph-level overrides set with :meth:`set_param_spec`."""
        return {n: dict(o) for n, o in self._param_spec_overrides.items() if o}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def get_node(self, name: str) -> SimulationNode:
        """The :class:`~maddening.core.node.SimulationNode` registered as ``name``.

        The read-only counterpart of :meth:`add_node`, for callers that
        need the node object itself — its ``static_data``, ``params`` or
        ``boundary_input_spec`` — rather than the graph's view of it; a
        point reference ``{"node": ..., "field": ...}`` of an interface
        mapping resolves through it.  An unknown name is a ``KeyError``.
        """
        if name not in self._nodes:
            raise KeyError(f"unknown node {name!r}; the graph has {sorted(self._nodes)}")
        return self._nodes[name].node

    def add_node(self, node: SimulationNode) -> None:
        """Register a node and initialise its state."""
        if node.name in self._nodes:
            raise ValueError(f"Node '{node.name}' already exists in the graph.")
        bad = [t for t in ("/", "#", "->") if t in node.name]
        if not node.name or bad:
            # These tokens delimit checkpoint keys, mapping slots and edge
            # keys; a node name containing them corrupts those namespaces.
            raise ValueError(
                f"Node name {node.name!r} is invalid: must be non-empty and must "
                f"not contain {bad or ['/', '#', '->']}"
            )
        from maddening.serialization.json_codec import (  # noqa: PLC0415
            NON_FINITE_TOKENS,
        )
        if node.name in NON_FINITE_TOKENS:
            # MADD-ANO-010: the JSON surfaces refuse a string that spells a
            # non-finite token, and a node name is a JSON *value* in
            # ``to_dict`` (``nodes[i]["name"]``) and in any mapping point
            # reference.  It reached the stage untouched, though, because
            # ``save_graph_to_usd`` writes it to a typed USD String
            # attribute that never sees the codec -- so a ``.usda`` could
            # round-trip to a graph that could not be written as a config,
            # and the same graph was refused or accepted depending on which
            # surface it met.  Refused here instead, at the point of entry,
            # which is what the anomaly's own workaround recommends
            # ("validate names ... where they are accepted, not where they
            # are saved") and what makes the three surfaces agree.
            raise ValueError(
                f"Node name {node.name!r} is invalid: it spells a non-finite "
                f"JSON token, which the serialisers reserve (MADD-ANO-010), so "
                f"a graph holding it could not be written as a config or "
                f"referenced from an interface mapping.  A different spelling "
                f"({node.name.lower()!r}, say) is fine."
            )

        spec = _NodeSpec(
            node=node,
            update_fn=node.update,
            timestep=node.delta_t,
            accepts_params=_update_accepts_params(node),
            flux_accepts_params=_flux_accepts_params(node),
        )
        # Atomic on purpose: build the state *before* committing to either
        # dict.  ``initial_state()`` is a documented, recoverable failure
        # point -- an ``AdaptiveNode`` raises ``AdaptiveNodeBlindnessError``
        # at a Palais trap and the developer guide's recovery is to perturb
        # the parameters and re-add under the same name.  Registering the
        # spec first left ``_nodes[name]`` populated and ``_state[name]``
        # missing: the name was taken for good (``add_node`` raised
        # "already exists", ``remove_node`` raised ``KeyError``),
        # ``compile()`` accepted the graph, ``params["nodes"]`` carried a
        # node that can never run, and ``step()`` died much later with a
        # bare ``KeyError`` inside the compiled step.  A failed ``add_node``
        # must leave the graph exactly as it was.
        state = node.initial_state()
        self._nodes[node.name] = spec
        self._state[node.name] = state
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
        mapping: Optional[Any] = None,
    ) -> None:
        """Add a data-dependency edge between two nodes.

        ``mapping`` (a :class:`maddening.core.coupling.mapping.Mapping`)
        transfers the source field onto the target interface before
        ``transform`` is applied; its weights are snapshotted into
        ``params["mappings"][edge.key]`` at compile time and passed as a
        traced input on every step.  Its ``n_source`` must equal the
        source field's size; ``n_target`` must match the target's
        declared ``boundary_input_spec`` shape when that is an array.
        A :class:`~maddening.core.coupling.mapping_spec.MappingSpec` (or
        its dict form) is rebuilt first with :meth:`point_resolver`.

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
        if mapping is not None:
            from maddening.core.coupling.mapping_spec import MappingSpec  # noqa: PLC0415
            if isinstance(mapping, (MappingSpec, dict)):
                # A spec (or its dict form, as written by to_dict): rebuild
                # the mapping from this graph's node fields; asset paths
                # are relative to the working directory.
                if isinstance(mapping, dict):
                    mapping = MappingSpec.from_dict(mapping)
                mapping = mapping.build(self.point_resolver())
            self._check_mapping_shapes(source, source_field, target, target_field, mapping)
        ordinal = 0
        if mapping is not None:
            # Mapping weights live in params["mappings"][edge.key]; a
            # second mapped edge on the same field pair (two additive
            # contributions, say) gets its own slot via the ordinal
            # instead of silently sharing -- and using -- the other's
            # weights.
            base = f"{source}.{source_field}->{target}.{target_field}"
            ordinal = sum(
                1 for e in self._edges
                if e.mapping is not None and e.key.split("#")[0] == base
            )
        edge = EdgeSpec(source, target, source_field, target_field,
                        transform, additive, source_units, target_units,
                        mapping=mapping, ordinal=ordinal)
        self._edges.append(edge)
        self._dirty = True
        self._notify(EVENT_EDGE_ADDED, edge)

    def point_resolver(self, base_dir=None) -> Callable[[dict], Any]:
        """``resolve_points`` for :meth:`MappingSpec.build`: node-field
        references are looked up in this graph's nodes, asset paths are
        relative to ``base_dir`` (the working directory when ``None``)."""
        from maddening.core.coupling.mapping_spec import make_point_resolver  # noqa: PLC0415
        return make_point_resolver(self, base_dir)

    def _check_mapping_shapes(self, source, source_field, target, target_field, mapping):
        for attr in ("apply", "params_pytree", "n_source", "n_target"):
            if not hasattr(mapping, attr):
                raise TypeError(
                    f"mapping must implement the Mapping protocol (missing {attr!r})"
                )
        src_spec = self._nodes.get(source)
        if src_spec is not None:
            src_state = src_spec.node.initial_state()
            if source_field in src_state:
                n = int(np.prod(np.shape(src_state[source_field])[:1] or (1,)))
                if mapping.n_source != n:
                    raise ValueError(
                        f"mapping n_source={mapping.n_source} does not match "
                        f"{source}.{source_field} (size {n} along axis 0)"
                    )
        tgt_spec = self._nodes.get(target)
        if tgt_spec is not None:
            bspec = tgt_spec.node.boundary_input_spec().get(target_field)
            shape = tuple(getattr(bspec, "shape", ()) or ()) if bspec is not None else ()
            if shape and mapping.n_target != int(shape[0]):
                raise ValueError(
                    f"mapping n_target={mapping.n_target} does not match "
                    f"{target}.{target_field} declared shape {shape}"
                )

    @property
    def edges(self) -> list[EdgeSpec]:
        return list(self._edges)

    def resolve_boundary_inputs(self, node_name: str, params: Optional[dict] = None) -> dict:
        """Boundary inputs ``node_name`` would receive from the *current*
        state: every incoming edge (mapping, transform, additive) plus the
        zero defaults of its external inputs.  A debugging / inspection
        helper; the compiled step resolves edges itself."""
        if node_name not in self._nodes:
            raise KeyError(f"unknown node {node_name!r}")
        p = self._params_or_default(params)
        resolved = _ResolvedParams(p.get("nodes", {}), p.get("mappings", {}))
        out: dict[str, Any] = {}
        for edge in self._edges:
            if edge.target_node != node_name:
                continue
            value = self._state[edge.source_node][edge.source_field]
            value = _apply_edge(edge, value, resolved)
            if edge.additive and edge.target_field in out:
                out[edge.target_field] = out[edge.target_field] + value
            else:
                out[edge.target_field] = value
        for ei in self._external_inputs:
            if ei.target_node == node_name and ei.target_field not in out:
                out[ei.target_field] = jnp.zeros(ei.shape, dtype=ei.dtype)
        return out

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
        # ``pop`` rather than ``del``: a graph whose state entry is missing
        # must still be removable, so the removal cannot itself fail
        # half-way and leave ``_nodes`` and ``_state`` disagreeing.
        self._state.pop(name, None)
        self._edges = [
            e for e in self._edges
            if e.source_node != name and e.target_node != name
        ]
        self._external_inputs = [
            e for e in self._external_inputs if e.target_node != name
        ]
        # ParamSpec overrides for the node and for mapped edges that
        # touched it would otherwise survive and break to_dict/from_dict;
        # its live params entry is discarded on purpose (an intentional
        # removal / replacement must not warn at the next compile).
        self._param_spec_overrides.pop(name, None)
        for key in list(self._param_spec_overrides):
            if key.startswith(f"{name}.") or f"->{name}." in key:
                self._param_spec_overrides.pop(key, None)
        self.params.get("nodes", {}).pop(name, None)
        for key in list(self.params.get("mappings", {})):
            if key.startswith(f"{name}.") or f"->{name}." in key:
                self.params["mappings"].pop(key, None)
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
        for key in list(self._param_spec_overrides):
            if key.split("#")[0] == edge.key:
                self._param_spec_overrides.pop(key, None)
        for key in list(self.params.get("mappings", {})):
            if key.split("#")[0] == edge.key:
                self.params["mappings"].pop(key, None)
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
            Convergence threshold (L2 norm of state change).  Read
            **only** under ``convergence_norm="l2"``; ``"mixed"`` and
            ``"interface"`` carry their tolerances in ``atol`` /
            ``rtol`` and test against a fixed threshold of ``1.0``.
            Setting a knob the chosen norm ignores warns
            (``UserWarning``) rather than turning silently.
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
        new_set = frozenset(nodes)
        # Check for overlap with existing coupling groups
        for existing in self._coupling_groups:
            overlap = new_set & existing.nodes
            if overlap:
                raise ValueError(
                    f"Nodes {overlap} already belong to a coupling group."
                )
        group = self._make_coupling_group(
            nodes, max_iterations, tolerance, **kwargs
        )
        self._coupling_groups.append(group)
        self._dirty = True
        return group

    def _make_coupling_group(
        self,
        nodes: Sequence[str],
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        **kwargs,
    ) -> CouplingGroup:
        """Validate and construct a group without registering it.

        Everything that can refuse a group -- an unknown node name,
        ``CouplingGroup``'s own validation of the knobs -- happens here,
        so a caller replacing several groups at once can build them all
        before it touches the graph.
        """
        for name in nodes:
            if name not in self._nodes:
                raise KeyError(f"No node named '{name}'.")
        return CouplingGroup(
            nodes=frozenset(nodes),
            max_iterations=max_iterations,
            tolerance=tolerance,
            **kwargs,
        )

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
        # Every group built before any of them is registered.  The old
        # order cleared the groups first and built them one at a time, so
        # a ``kwargs`` the ``CouplingGroup`` constructor refuses -- a
        # misspelled knob -- destroyed the groups the graph already had
        # and put nothing back.  Clearing is itself a change to the
        # compiled step, so the dirty flag is part of the same commit:
        # ``add_coupling_group`` used to be the only thing that set it,
        # and an ``auto_couple`` that found no cycles left the graph
        # describing itself as uncoupled while still running the coupled
        # step it was compiled with.
        sccs = find_strongly_connected_components(
            list(self._nodes.keys()), self._edges
        )
        groups = [
            self._make_coupling_group(scc, max_iterations, tolerance, **kwargs)
            for scc in sccs
        ]
        self._coupling_groups[:] = groups
        self._dirty = True
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
                if e.mapping is not None and src_shape:
                    # The mapping changes axis 0 to its n_target; the
                    # rest of the shape (vector components) passes through.
                    src_shape = (int(e.mapping.n_target),) + src_shape[1:]
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
                if src_dtype is not None and e.transform is None and e.mapping is None:
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
        # Preserving the state across the rebuild is only safe if the
        # state is usable; a graph still holding a transform's tracers
        # goes back to the state it had before it first.
        self._recover_from_escaped_tracers()
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

        # Everything from here to the commit point at the end of the
        # method is computed into locals.  The ``accelerated_fields``
        # validation, the static-data refusal and ``_build_step_fn`` can
        # all still raise, and a compile that fails must leave the graph
        # bit-identical to what it was -- not describing a step that was
        # never built.  See ``_StepPlan``.
        node_names = list(self._nodes.keys())
        schedule = topological_sort(node_names, self._edges)
        back_edges = identify_back_edges(schedule, self._edges)

        # Explicit accelerated_fields must name state fields of the group's
        # nodes (a boundary flux is not a state field; use the default,
        # which maps a flux edge to the producer's state fields).
        #
        # Before *everything* that reads the field, and in particular
        # before the ``iqn-imvj`` ``_meta`` seeding below, which calls
        # ``flatten_coupled_state(..., fields=...)`` with the user's list
        # and dies on an unknown field with a bare ``KeyError: 'typo'``.
        # That shadowed this message under the one acceleration in which
        # ``accelerated_fields`` is most used, while it fired cleanly
        # under ``acceleration="none"``, where ``CouplingGroup`` already
        # warns that the field is ignored altogether.  The block reads
        # only ``self._coupling_groups``, ``self._nodes`` and
        # ``self._state``, all of which are final here.
        for g in self._coupling_groups:
            if g.accelerated_fields is None:
                continue
            for nn, fields in g.accelerated_fields.items():
                if nn not in self._nodes or nn not in g.nodes:
                    raise ValueError(
                        f"accelerated_fields names node {nn!r}, not in coupling "
                        f"group {sorted(g.nodes)}"
                    )
                have = set(self._state.get(nn, {}).keys())
                bad = [f for f in fields if f not in have]
                if bad:
                    raise ValueError(
                        f"accelerated_fields[{nn!r}] names {bad}: not a state field "
                        f"of {nn!r} (state fields: {sorted(have)})"
                    )

        # ``_meta`` is *state*, not derived data: ``step_count`` decides
        # which sub-steps a node with a rate divider > 1 fires on, and the
        # ``coupling_*`` entries are the predictor history and the IQN
        # warm start.  A recompile preserves node state and ``params``; it
        # has to preserve these for the same reason.  A mid-run structural
        # edit (adding an edge or an external input, the profiler or the
        # REST server recompiling behind your back) used to replace this
        # dict, silently re-phasing a multi-rate schedule and restarting
        # every warm start.  Snapshotted here, *before* the rate dividers
        # are recomputed, and re-applied below over the key set the new
        # graph expects.  ``reset_state()`` remains the explicit way to
        # zero the counters.
        previous_meta = dict(self._state.get(_META_KEY, {}))
        # From the last *successful* compile, not from ``_rate_dividers``:
        # a compile that raises after recomputing them (the static-data
        # refusal, a failing ``_build_step_fn``) leaves them describing a
        # step that was never built, and comparing against those would
        # restart the phase on the repair.  (The ``accelerated_fields``
        # typo used to be one of those; it is now refused above, before
        # the dividers are touched at all.)
        previous_dividers = dict(self._committed_rate_dividers)

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
            is_multirate = True
            base_dt = _multi_gcd(timesteps)
            rate_dividers = {
                name: round(effective_timesteps[name] / base_dt)
                for name in self._nodes
            }
        else:
            is_multirate = False
            rate_dividers = {name: 1 for name in self._nodes}

        # Build ``_meta`` fresh over the key set *this* graph needs, so a
        # key whose owning coupling group is gone cannot linger in the
        # scan carry, then carry the previous values back over it below.
        meta: dict = {}
        if is_multirate:
            meta["step_count"] = jnp.array(0, dtype=jnp.int32)

        # Ensure _meta exists with correct structure when coupling
        # diagnostics are enabled.  Pre-populate diagnostic keys so
        # the pytree structure is stable across lax.scan iterations.
        has_diagnostics = any(
            g.diagnostics or g.solver == "ift" for g in self._coupling_groups
        )
        has_imvj = any(
            g.acceleration == "iqn-imvj" for g in self._coupling_groups
        )
        has_predictor = any(
            g.predictor != "none" for g in self._coupling_groups
        )
        if has_diagnostics or has_imvj or has_predictor:
            for g in self._coupling_groups:
                key = "+".join(sorted(g.nodes))
                if g.diagnostics or g.solver == "ift":
                    meta[f"coupling_{key}_iterations"] = jnp.array(
                        0, dtype=jnp.int32
                    )
                    # Seed in the dtype the residual is computed in (the
                    # group's floating state), so a float64 graph under
                    # x64 keeps a stable scan carry / trace signature.
                    res_dtype = jnp.float32
                    for nn_ in g.nodes:
                        for leaf in self._state.get(nn_, {}).values():
                            if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating):
                                res_dtype = jnp.asarray(leaf).dtype
                                break
                        else:
                            continue
                        break
                    meta[f"coupling_{key}_residual"] = jnp.array(0.0, dtype=res_dtype)
                    # The amplification 1/(1-rho) the error bound is
                    # built from; 0.0 reads as "no usable estimate".
                    meta[f"coupling_{key}_amplification"] = jnp.array(
                        0.0, dtype=res_dtype
                    )
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
                        af = _interface_state_fields(self._edges, g.nodes, self._state)
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
                    flat0 = _fcs_pred(
                        self._state, group_names_pred,
                        fields=float_fields_of(self._state, group_names_pred),
                    )
                    n_pred = 3 if g.predictor == "quadratic" else 2
                    for pi in range(n_pred):
                        meta[f"coupling_{key}_pred_{pi}"] = flat0
                    # Counter for how many converged states have
                    # been stored (0 at start, up to n_pred).
                    meta[f"coupling_{key}_pred_count"] = jnp.array(
                        0, dtype=jnp.int32
                    )

        # Carry the live ``_meta`` back over the seeds, key by key.  Only
        # keys the new graph expects are kept (a group that was removed
        # takes its diagnostics and warm start with it), and only when the
        # live value still fits the seed's shape and dtype -- a group
        # whose interface DOF count changed gets a fresh, correctly shaped
        # warm start rather than a crash inside ``lax.scan``.
        # ``step_count`` restarts only when a divider moved, because the
        # sub-step it indexes is then not the sub-step it indexed before.
        # Judged over the nodes that survived the edit only: adding or
        # removing a node must not re-phase the ones already running,
        # which is the whole point of preserving the counter.
        phase_still_means_the_same = all(
            previous_dividers[name] == divider
            for name, divider in rate_dividers.items()
            if name in previous_dividers
        )
        for key_, seed in meta.items():
            if key_ == "step_count" and not phase_still_means_the_same:
                continue
            live = previous_meta.get(key_)
            if live is None:
                continue
            live = jnp.asarray(live)
            if live.shape == jnp.shape(seed) and live.dtype == jnp.asarray(seed).dtype:
                meta[key_] = live
        # Computed here, committed at the very end of ``compile`` -- the
        # validation below, the static-data refusal and ``_build_step_fn``
        # can all still raise, and a compile that fails must leave the
        # sub-step phase and the warm starts exactly as it found them.

        # ``subcycling=True`` on a group whose nodes all share a
        # timestep is demoted to ``use_subcycling = False`` in
        # ``_run_coupled_block_impl``, which leaves
        # ``waveform_iterations`` and ``boundary_interpolation`` dead
        # while ``CouplingGroup``'s own ``subcycling`` predicate says
        # they are live.  The group cannot see that -- it does not know
        # its members' timesteps -- so the rule is enforced here, where
        # the nodes are known and the first step has not run yet.
        from maddening.core.coupling.group import (
            _FIELD_DEFAULTS,
            inert_uniform_timestep_message,
        )
        _SUBCYCLED_ONLY = ("waveform_iterations", "boundary_interpolation")
        for g in self._coupling_groups:
            if not g.subcycling:
                continue        # the CouplingGroup rule already covers it
            timesteps = {self._nodes[nn].timestep for nn in g.nodes
                         if nn in self._nodes}
            if len(timesteps) > 1:
                continue        # genuinely subcycled: the knobs are live
            named = tuple(
                name for name in _SUBCYCLED_ONLY
                if getattr(g, name) != _FIELD_DEFAULTS[name]
            )
            if named:
                warnings.warn(
                    inert_uniform_timestep_message(g, named),
                    UserWarning,
                    stacklevel=2,
                )

        # Persistent XLA cache, if the user asked for one via the env var
        # (see maddening.core.simulation.compile_cache).
        from maddening.core.simulation.compile_cache import enable_from_env
        enable_from_env()

        # A weak-typed leaf in the seed state would retrace the jitted
        # step once it comes back strongly typed after the first step.
        state = _strong_typed(self._state)
        # Zero external inputs are allocated once per compile, not per
        # step (``jnp.zeros`` per input per call cost ~1.5 ms/step on GPU).
        default_ext_leaves = {
            (ei.target_node, ei.target_field): jnp.zeros(ei.shape, dtype=ei.dtype)
            for ei in self._external_inputs
        }

        # Snapshot the differentiable parameters before building the
        # step so the closure default (``params=None``) is this snapshot.
        # Live values survive a recompile: a calibrated leaf whose
        # node/key/shape/dtype still exist is carried over (adding an
        # edge or an external input must not discard a fit); anything
        # that no longer fits is dropped with a warning.  ``reset_params``
        # restores the constructor values on purpose.
        params = self._merge_live_params(self._snapshot_params(), self.params)
        params_dtypes = {
            section: {
                owner: {k: jnp.asarray(v).dtype for k, v in leaves.items()}
                for owner, leaves in params.get(section, {}).items()
            }
            for section in ("nodes", "mappings")
        }
        params_shapes = {
            section: {
                owner: {k: tuple(jnp.shape(v)) for k, v in leaves.items()}
                for owner, leaves in params.get(section, {}).items()
            }
            for section in ("nodes", "mappings")
        }
        baked = self.nodes_without_params()
        if baked:
            logger.info(
                "nodes without a params keyword (constants baked, not "
                "differentiable through the graph): %s", baked,
            )

        # D10 step 3: a static derived from a *trainable* parameter is
        # refused outright.  The static is baked into the HLO as a
        # constant while the parameter is traced, so the gradient would
        # be missing the term through the static -- silently, and in the
        # direction an optimiser is pushing.  No rebuild hook can fix
        # that, so the graph does not compile.
        #
        # Declared, not inferred: ``compile`` cannot see which values a
        # traced closure reads, so ``static_data_deps`` is the node's own
        # statement of provenance.  The walk reaches wrapped nodes, each
        # resolved against its own specs, so a wrapper cannot hide one.
        # Placed in the same region as the invalidation below: before
        # ``_build_step_fn`` and before the static-data hash snapshot.
        # Resolved against the *merged* specs -- the node's own with this
        # graph's ``set_param_spec`` overrides applied -- because that is
        # the view ``trainable_mask``, ``unconstrain``, ``check_params``
        # and ``maddening.sysid`` optimise against.  Reading the node
        # alone made the rule disagree with the optimiser both ways: a
        # graph-level unfreeze walked past the refusal into a silently
        # wrong gradient, and a graph-level freeze -- the first remedy the
        # message below names -- did not clear it.
        from maddening.core.node import static_data_dep_violations
        for name, spec in self._nodes.items():
            for owner, static_key, param_key in static_data_dep_violations(
                spec.node, self._param_spec_overrides.get(name)
            ):
                where = (
                    f"node {name!r}" if owner == name
                    else f"node {name!r} (declared by the wrapped node {owner!r})"
                )
                raise ValueError(
                    f"{where} declares static_data[{static_key!r}] as derived "
                    f"from parameter {param_key!r}, which is trainable.  "
                    f"static_data is baked into the compiled HLO as a "
                    f"constant, and you cannot differentiate through a "
                    f"constant: the gradient with respect to {param_key!r} "
                    f"would silently omit the term through "
                    f"{static_key!r}, so a fit would move {param_key!r} "
                    f"while {static_key!r} stayed at its __init__ value.  "
                    f"Either declare {param_key!r} as "
                    f"ParamSpec(trainable=False), or stop deriving "
                    f"{static_key!r} from it and compute the quantity "
                    f"inside update() from the traced parameter instead."
                )

        # A node may keep its own materialised copy of its static arrays
        # (the sharded wrappers cache the per-device placement, keyed on
        # the arrays' identity).  Such a key cannot see a static whose
        # buffer was rewritten in place, and the cache lives on the node,
        # so without this the step just rebuilt would be traced against
        # the previous buffer.  ``compile()`` is the framework's explicit
        # "rebuild everything", so it has to reach those caches too; it
        # runs rarely, and the cost is one re-materialisation per sharded
        # static per compile, paid lazily on the next trace.
        #
        # ``invalidate_static_cache`` is a ``SimulationNode`` contract
        # method whose default forwards to any node this one wraps, so a
        # cache nested inside a wrapper (a sharded node inside a
        # HybridNode) is reached too.  The getattr probe stays for the
        # duck-typed node objects the graph also accepts.
        #
        # Ordered before the build rather than after it.  Both work today
        # only because ``_build_step_fn`` and ``jax.jit`` are lazy and
        # materialise nothing; clearing first is correct whether or not
        # that stays true, and it still precedes the static-data hash
        # snapshot below, which is the other ordering constraint.
        for spec in self._nodes.values():
            invalidate = getattr(spec.node, "invalidate_static_cache", None)
            if callable(invalidate):
                invalidate()

        # Built against the plan, not against the graph: the build is the
        # last thing that can raise, and it must be able to fail without
        # having moved the graph off the step it is running.
        plan = _StepPlan(
            schedule=schedule,
            back_edges=back_edges,
            is_multirate=is_multirate,
            rate_dividers=rate_dividers,
            params=params,
        )
        step_fn = self._build_step_fn(plan)

        def _counted_step(full_state, external_inputs, params=None):
            self._n_traces += 1
            return step_fn(full_state, external_inputs, params)

        compiled_step = jax.jit(_counted_step)

        # Snapshot static_data hashes so we can detect drift.
        # ``static_data_hash`` is a node-supplied method, so this is the
        # last thing in the method that can raise -- it stays above the
        # commit point.
        static_data_hashes = {
            name: spec.node.static_data_hash()
            for name, spec in self._nodes.items()
        }

        # ------------------------------------------------------------------
        # Commit point.  Nothing below raises, so everything computed above
        # is written onto the graph here, together: the plan the step was
        # built from, the state and parameter snapshots it closes over, the
        # ``_meta`` built above (see there) and the dividers the next
        # compile will judge its phase against.  Up to here, a raise leaves
        # the graph running exactly the step it was running before.
        # ------------------------------------------------------------------
        self._schedule = plan.schedule
        self._back_edges = plan.back_edges
        self._is_multirate = plan.is_multirate
        self._rate_dividers = plan.rate_dividers
        self.params = plan.params
        self._params_dtypes = params_dtypes
        self._params_shapes = params_shapes
        self._state = state
        self._default_ext_leaves = default_ext_leaves
        if meta:
            self._state[_META_KEY] = meta
        else:
            self._state.pop(_META_KEY, None)
        self._committed_rate_dividers = dict(plan.rate_dividers)
        # Count Python-level traces of the step: a robust, JAX-version-
        # independent retrace probe (the jit object's C++ cache count is
        # not comparable across versions).  ``trace_count`` is 0 right
        # after compile() and 1 after the first step of a well-behaved
        # graph; a growing count means something in the call signature
        # (weak types, dtypes, params structure) keeps changing.
        self._n_traces = 0
        self._compiled_step = compiled_step
        self._static_data_hashes = static_data_hashes

        self._dirty = False
        # A rebuilt step invalidates every scan built against the old
        # one.  Bumping the generation as well as clearing means a scan
        # a caller still holds can never be re-entered into the cache.
        self._compile_generation += 1
        self._scan_cache.clear()
        self._n_scan_traces = 0
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
        Uses ``jax.shard_map`` to distribute node updates
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

        # Mesh and device map into locals, committed together below.
        # ``assign_nodes_to_devices`` and ``EdgeSpec.to_dict`` can raise,
        # and a mesh committed on its own leaves ``validate_sharding``
        # judging the nodes against a mesh the graph is not using while
        # the step still runs on the previous device map -- with
        # ``_dirty`` never set, so nothing recompiles to reconcile them.
        mesh = create_device_mesh(
            n_devices, shape=mesh_shape, axis_names=mesh_axes
        )
        n = len(mesh.devices.reshape(-1))

        coupling_sets = [set(g.nodes) for g in self._coupling_groups]
        edges_dicts = [e.to_dict() for e in self._edges]
        device_map = assign_nodes_to_devices(
            node_names=list(self._nodes.keys()),
            edges=edges_dicts,
            coupling_groups=coupling_sets,
            n_devices=n,
        )
        self._multigpu_mesh = mesh
        self._multigpu_device_map = device_map
        self._dirty = True

    def _committed_plan(self) -> "_StepPlan":
        """The plan the graph is currently running, as last committed by
        ``compile()``.  Rebuilding the step from this reproduces the step
        that is live, which is what a caller that asks for a step function
        outside ``compile()`` means."""
        return _StepPlan(
            schedule=list(self._schedule),
            back_edges=list(self._back_edges),
            is_multirate=self._is_multirate,
            rate_dividers=dict(self._rate_dividers),
            params=self.params,
        )

    def _build_step_fn(self, plan: Optional["_StepPlan"] = None) -> Callable:
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

        Parameters
        ----------
        plan : _StepPlan, optional
            The schedule, multi-rate info and parameter snapshot to build
            against.  ``compile()`` passes the plan it has computed but
            not yet committed, so the build -- which can raise -- happens
            before the graph is touched.  ``None`` builds against what is
            committed on the graph, which is what a caller rebuilding the
            step of an already-compiled graph wants.
        """
        if plan is None:
            plan = self._committed_plan()
        schedule = list(plan.schedule)
        nodes = dict(self._nodes)
        back_edge_set = set(plan.back_edges)
        is_multirate = plan.is_multirate
        rate_dividers = dict(plan.rate_dividers)
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

        # ``params=None`` on the step means "the compile-time snapshot":
        # baked in as constants, exactly the pre-params behaviour.  An
        # explicit ``params`` is a traced input, so ``jax.grad`` reaches
        # it and a new value needs no recompile.
        params_snapshot = plan.params

        def _resolve_params(params):
            if params is None:
                params = params_snapshot
            else:
                self._validate_params(params)
            return _ResolvedParams(params.get("nodes", {}), params.get("mappings", {}))

        def _resolve_and_update_node(
            node_name, new_state, full_state, external_inputs, node_params,
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
                value = _apply_edge(edge, value, node_params)
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
            new_node_state = _node_update(
                spec, new_state[node_name], boundary_inputs, spec.timestep,
                node_params.nodes.get(node_name),
            )

            # Compute fluxes for this node if it produces them
            from maddening.core.node import SimulationNode as _SimBase
            if type(spec.node).compute_boundary_fluxes is not _SimBase.compute_boundary_fluxes:
                fluxes = _node_fluxes(
                    spec, new_node_state, boundary_inputs, spec.timestep,
                    node_params.nodes.get(node_name),
                )
                if fluxes:
                    flux_state[node_name] = fluxes

            return new_node_state

        def _run_coupled_block(group, group_schedule, new_state,
                               full_state, external_inputs, node_params,
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
                node_params=node_params,
            )

        if not is_multirate and not has_coupling:
            # ---- Uniform-rate, no coupling: fast path ----
            def graph_step(full_state, external_inputs, params=None):
                node_params = _resolve_params(params)
                new_state = {k: v for k, v in full_state.items()}

                for node_name in schedule:
                    new_state[node_name] = _resolve_and_update_node(
                        node_name, new_state, full_state, external_inputs, node_params
                    )
                return new_state

            return graph_step

        if has_coupling and not is_multirate:
            # ---- Coupling groups, uniform rate ----
            def graph_step_coupled(full_state, external_inputs, params=None):
                node_params = _resolve_params(params)
                new_state = {k: v for k, v in full_state.items()}

                for block in blocks:
                    if block[0] == "node":
                        node_name = block[1]
                        new_state[node_name] = _resolve_and_update_node(
                            node_name, new_state, full_state, external_inputs, node_params
                        )
                    else:
                        _, group, group_schedule = block
                        new_state = _run_coupled_block(
                            group, group_schedule, new_state,
                            full_state, external_inputs, node_params,
                        )

                return new_state

            return graph_step_coupled

        # ---- Multi-rate path (with or without coupling) ----
        def graph_step_multirate(full_state, external_inputs, params=None):
            node_params = _resolve_params(params)
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
                            node_name, new_state, full_state, external_inputs, node_params
                        )
                        new_state[node_name] = _apply_multirate(
                            node_name, updated, new_state
                        )
                    else:
                        _, group, group_schedule = block
                        coupled_result = _run_coupled_block(
                            group, group_schedule, new_state,
                            full_state, external_inputs, node_params,
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
                        node_name, new_state, full_state, external_inputs, node_params
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
        cache = getattr(self, "_default_ext_leaves", None) or {}
        ext: dict[str, dict] = {}
        for ei in self._external_inputs:
            leaf = cache.get((ei.target_node, ei.target_field))
            if leaf is None or leaf.shape != tuple(ei.shape) or leaf.dtype != jnp.dtype(ei.dtype):
                leaf = jnp.zeros(ei.shape, dtype=ei.dtype)
            # Fresh outer dicts each call (callers may edit them); the
            # zero arrays themselves are immutable and shared.
            ext.setdefault(ei.target_node, {})[ei.target_field] = leaf
        return ext

    def _resolve_external_inputs(
        self, external_inputs: Optional[dict[str, dict]],
    ) -> dict[str, dict]:
        """Complete and validate a caller's ``external_inputs``.

        ``None`` means "zeros for every declared input", which was
        already documented.  A *partial* dict now means the same for the
        inputs it omits, rather than leaving them out of
        ``boundary_inputs`` altogether and letting the node fall back to
        its own default -- a 98 N difference in the case that found this,
        with nothing said about it anywhere.

        An unknown ``(node, field)`` pair is an error naming the declared
        ones.  A typo'd node or field name used to be accepted in
        silence, which is exactly the failure mode ``_validate_params``
        spends a paragraph per case avoiding for the ``params`` argument
        of the very same call.
        """
        if external_inputs is None:
            return self._default_external_inputs()
        declared = {
            (ei.target_node, ei.target_field) for ei in self._external_inputs
        }
        unknown = sorted(
            f"{node}.{field}"
            for node, fields in external_inputs.items()
            for field in fields
            if (node, field) not in declared
        )
        if unknown:
            known = sorted(f"{n}.{f}" for n, f in declared)
            raise ValueError(
                f"external_inputs names {unknown}, which this graph does not "
                f"declare; declared external inputs: {known or ['(none)']}.  "
                f"An undeclared name never reaches the node, so accepting it "
                f"would mean the value silently did nothing.  Declare it with "
                f"add_external_input(), or fix the name."
            )
        if not declared:
            return external_inputs
        # Complete from the per-compile zero cache.  Fresh outer dicts,
        # like ``_default_external_inputs``: callers may edit them.
        out: dict[str, dict] = {}
        for node, fields in self._default_external_inputs().items():
            out[node] = {**fields, **external_inputs.get(node, {})}
        return out

    # ------------------------------------------------------------------
    # Escaped tracers
    # ------------------------------------------------------------------

    def _store_state(self, new_state: dict) -> None:
        """Write *new_state* back, remembering whether it is traced.

        ``step``, ``run``, ``run_scan`` and their siblings are stateful:
        they assign their result into ``self._state``.  Under a JAX
        transform that result is a pytree of tracers, so a loss that
        calls one -- the recipe ``docs/user_guide/quickstart.md`` shows
        -- left the graph holding tracers once the transform returned,
        and every later ``step`` / ``run_scan`` / ``save_state`` failed
        with an error pointing at JAX rather than at the framework.

        The write still happens, because a Python loop of ``gm.step()``
        *inside* a trace depends on it.  What is added is the state to
        come back to: see :meth:`_recover_from_escaped_tracers`.
        """
        if _holds_tracer(new_state):
            if not self._state_traced:
                self._state_before_trace = self._state
            self._state_traced = True
        else:
            self._state_traced = False
            self._state_before_trace = None
        self._state = new_state

    def _recover_from_escaped_tracers(self) -> None:
        """Put the graph back to the last untraced state, if it needs it.

        Called from every entry point.  It costs one attribute test when
        there is nothing to do, which is always except right after a
        transform that stepped the graph.  Inside a transform it does
        nothing, so a traced multi-step loop still works.
        """
        if not self._state_traced or not _outside_jax_trace():
            return
        restored = self._state_before_trace
        if restored is None:            # pragma: no cover - defensive
            self._state_traced = False
            return
        # State first, flags after, so a graph that somehow failed to be
        # put back is still marked as holding tracers and tries again.
        self._state = restored
        self._state_traced = False
        self._state_before_trace = None
        warnings.warn(
            "the graph held JAX tracers left behind by a transform and has "
            "been put back to the state it had before it.  step() / run() / "
            "run_scan() assign their result into the graph, so a loss that "
            "calls one leaves tracers in it when jax.grad returns; the "
            "gradient itself is unaffected.  Set the state you want "
            "explicitly (set_node_state / reset_state / load_state) after "
            "differentiating if the recovered state is not the one you meant.",
            RuntimeWarning,
            stacklevel=3,
        )

    # ------------------------------------------------------------------
    # Internal helpers for _meta stripping
    # ------------------------------------------------------------------

    def _user_state(self, full_state: dict) -> dict:
        """The caller's view of *full_state*: no internal ``_meta`` key,
        and a fresh dict per node.

        It used to hand back ``self._state`` itself for a graph with no
        ``_meta`` (the common uncoupled, uniform-rate case), and the
        per-node dicts even when it did copy the outer one, so a caller
        clamping a value in the dict it was given rewrote the running
        simulation.  The arrays are immutable and stay shared; only the
        dicts are new, which is one small allocation per node per step.
        """
        return {
            name: dict(fields) if type(fields) is dict else fields
            for name, fields in full_state.items()
            if name != _META_KEY
        }

    def coupling_diagnostics(self) -> dict[str, dict]:
        """Return coupling convergence info from the last step.

        Returns
        -------
        dict
            Keyed by coupling group identifier (sorted node names
            joined by ``"+"``), each containing:

            - ``"iterations"`` : int — coupling passes used to produce
              the state this step returned, counting the first
              staggered pass.  Equal to ``max_iterations`` exactly when
              the group exhausted its budget, whichever solver ran, so
              ``iterations >= max_iterations`` is a usable cap check.
            - ``"residual"`` : float — ``||F(x) - x||`` in the group's
              convergence norm for the state ``x`` this step returned.
              At ``max_iterations=1`` it is the distance the single
              pass moved, which is the same thing measured one pass
              earlier.  **It carries a floating-point noise floor**,
              and near the fixed point that floor is the whole value:
              see the note on ``solver`` below.
            - ``"amplification"`` : float — the estimated
              ``1 / (1 - rho)`` of the group's slowest mode, from the
              ratio of the last two residuals.  ``nan`` when the
              estimate was rejected (see ``"ratio_usable"``).
            - ``"error_estimate"`` : float — ``residual * omega *
              amplification``, an estimate of ``||x - x*||`` in the
              same norm: how far the returned state is from the fixed
              point, rather than how far the last pass moved.
              ``omega`` is the relaxation factor under
              ``acceleration="fixed"`` and 1 otherwise — the series is
              over the steps the iterate takes, and over-relaxation
              makes those longer than the residual that is measured.
              Falls back to ``residual`` when the estimate was
              rejected.  **It is an estimate, not a bound**: it can
              understate, and by large factors — see ``"ratio_usable"``
              and
              ``benchmarks/results/audit_040_final/ERROR_BOUND_DECISION.md``.
            - ``"ratio_usable"`` : bool — **whether the contraction
              ratio was usable this step, and nothing more.**  The
              estimate above rests on four conditions; this flag checks
              exactly one of them, the fourth.

              *What ``True`` means*: ``rho < 1``, the predecessor
              residual was non-zero, and every residual in the ratio
              was finite — so ``1/(1 - rho)`` is a number worth
              extrapolating from, and the criterion used the estimate.

              *What ``True`` does not mean*: it does **not** certify
              that the estimate bounds the distance to the fixed point.
              Three further conditions are unchecked — the measure
              obeying the triangle inequality, ``rho`` being at least
              the *asymptotic* rate rather than the rate of the mode
              that happens to dominate the step, and the step scale
              being the one actually applied (uncorrected under
              ``"aitken"`` and ``"iqn-*"``).  Each is measured broken
              in 0.4.0; the worst is a **122x** understatement with
              ``ratio_usable=True`` and ``converged=True``.  See
              :func:`_fixed_point_while` for all four.

              ``False`` on a non-monotone (non-normal) sequence, on a
              zero or non-finite predecessor, and at
              ``max_iterations=1``, where there is no pair of residuals
              to take a ratio of.  The criterion then falls back to the
              raw residual test, which is what ``converged`` reports.

              *Renamed in 0.4.0* from ``"bound_valid"``, which asserted
              all four conditions while checking one.  The old key is
              still readable through 0.4.x, warns, and is removed in
              0.5.0.
            - ``"gradient_error_estimate"`` : float — how far the IFT
              adjoint may be from a finite difference of this group's
              own forward, ``residual * cond(I - dF/dx)`` estimated
              along the observed slowest mode (numerically the same
              number as ``"error_estimate"``: both are
              ``(I - dF/dx)^-1`` applied to a residual).  ``inf`` when
              ``ratio_usable`` is ``False`` — no contraction was
              observed, so nothing constrains the disagreement.  Being
              the same number, it inherits every way
              ``"error_estimate"`` can understate, which is why it is
              an estimate and not a bound.  *Renamed in 0.4.0* from
              ``"gradient_error_bound"``, on the same alias terms as
              ``"ratio_usable"``.
            - ``"converged"`` : bool — the *error estimate* met the
              group's threshold (``tolerance`` for the L2 norm, ``1.0``
              for the mixed / interface norms).  ``False`` means the
              group hit ``max_iterations`` *and* the state it returned
              is still outside the threshold; under ``solver="ift"``
              the gradient through that step is then unreliable.

            ``converged=True`` is a statement about the state this step
            returned: both solvers stop on the iterate whose residual
            met the criterion rather than on the update it went on to
            produce, so recomputing ``||F(x) - x||`` on the state you
            were handed reproduces ``"residual"``.  Since 0.4.0 it is
            also an *estimate of the distance to the fixed point* and
            not only of the last step: the threshold is applied to
            ``omega * residual / (1 - rho)`` with ``rho`` measured from
            the residual sequence, which is what MADD-ANO-005 recorded
            as missing.  Where the ratio is unusable the flag degrades
            to the old residual test and says so through
            ``"ratio_usable"``.  It is strictly stronger than the
            pre-0.4.0 flag in every case and still not a guarantee —
            do not treat ``converged=True`` as certifying a distance.

            ``"ift"`` (the default) and the legacy ``"fori"`` run the
            same passes, stop on the same pass and derive every value
            here by the same rule, so migrating a graph between them
            does not move the answer or the verdict.  The returned
            state agrees to float32 round-off -- bit-identical on most
            graphs, and 7.3e-07 relative in the worst of 480
            configurations of a subcycled multi-rate group, none of
            which disagreed on ``converged``.

            **The reported ``"residual"`` is the one number that moves
            further than that, and only by float32 round-off.**  Every norm
            here divides ``F(x) - x`` by a scale, so it is a
            *cancellation*: near the fixed point the numerator is the
            difference of two nearly equal float32 states, and one unit
            in the last place of either is a full-size contribution to
            it.  The two solvers run their passes in different loop
            constructs -- ``"ift"`` in a ``lax.while_loop`` so it can
            exit early, ``"fori"`` in a ``lax.fori_loop`` -- which XLA
            is free to compile to differently rounded arithmetic, and
            it does: one ulp on a couple of components of the map's
            output is enough to move a residual of 1e-05 to ``0.0``.
            So ``"residual"`` agrees between the solvers to about
            ``eps_float32 / rtol`` under the mixed norm and
            ``eps_float32 * sqrt(n)`` under the L2 norm, which is the
            measurement's own resolution and not a bound on anything
            physical.  A residual at that floor means "converged to
            float32", and comparing two of them -- across solvers,
            across JAX versions or across backends -- compares rounding.
            ``converged``, ``iterations`` and the returned state carry
            no such caveat.  Pinned by
            ``tests/core/test_coupling_solver_equivalence.py``.

            Reported for every group under ``solver="ift"``; ``"fori"``
            groups only with ``diagnostics=True``.  Empty dict if no
            step has been taken yet.
        """
        meta = self._state.get(_META_KEY, {})
        result: dict[str, dict] = {}
        for group in self._coupling_groups:
            key = "+".join(sorted(group.nodes))
            iter_key = f"coupling_{key}_iterations"
            res_key = f"coupling_{key}_residual"
            amp_key = f"coupling_{key}_amplification"
            if iter_key in meta:
                residual = float(meta[res_key])
                amp = float(meta.get(amp_key, 0.0))
                # A valid amplification is ``1/(1 - rho)`` with
                # ``rho`` in ``[0, 1)``, so it is always >= 1; the
                # solvers write 0.0 for "rejected".
                valid = amp >= 1.0
                # The geometric series is over the steps the iterate
                # takes, which are ``relaxation`` times the residual
                # that is measured under ``acceleration="fixed"``.
                # Both solvers apply the same factor to the same
                # criterion, so this reproduces their ``converged``.
                scale = relaxation_step_scale(
                    group.acceleration, group.relaxation,
                )
                error_estimate = (
                    residual * max(scale * amp, 1.0) if valid else residual
                )
                threshold = (
                    1.0 if group.convergence_norm in ("mixed", "interface")
                    else group.tolerance
                )
                result[key] = _CouplingDiagnostics({
                    "iterations": int(meta[iter_key]),
                    "residual": residual,
                    "amplification": amp if valid else float("nan"),
                    "error_estimate": error_estimate,
                    "ratio_usable": valid,
                    "gradient_error_estimate": (
                        error_estimate if valid else float("inf")
                    ),
                    "converged": error_estimate <= threshold,
                })
        return result

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def step(
        self,
        external_inputs: Optional[dict[str, dict]] = None,
        *,
        params: Optional[dict] = None,
    ) -> dict[str, dict]:
        """Advance the simulation by one base timestep.

        Parameters
        ----------
        external_inputs : dict, optional
            Values injected from outside the graph, structured as
            ``{node_name: {field_name: value, ...}, ...}``.
            Zeros are used for every declared input this does not
            supply, ``None`` included; an undeclared ``node.field``
            is a ``ValueError`` naming the declared ones.
        params : dict, optional
            Graph parameter pytree (see :attr:`params`).  ``None`` uses
            :attr:`params`.  Passing a modified pytree changes node
            constants for this step without recompiling.

        Returns the full state dict after the step (excluding internal
        metadata).
        """
        self._recover_from_escaped_tracers()
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()

        external_inputs = self._resolve_external_inputs(external_inputs)
        params = self._params_or_default(params)

        self._store_state(
            self._compiled_step(self._state, external_inputs, params)
        )
        user_state = self._user_state(self._state)
        self._notify(EVENT_STEP, user_state)
        return user_state

    def run(
        self,
        n_steps: int,
        callback: Optional[Callable] = None,
        external_inputs: Optional[dict[str, dict]] = None,
        *,
        params: Optional[dict] = None,
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
            Completed and validated as in :meth:`step`.
        """
        self._recover_from_escaped_tracers()
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()

        external_inputs = self._resolve_external_inputs(external_inputs)
        params = self._params_or_default(params)

        for i in range(n_steps):
            self._store_state(
                self._compiled_step(self._state, external_inputs, params)
            )
            user_state = self._user_state(self._state)
            self._notify(EVENT_STEP, user_state)
            if callback is not None:
                callback(i, user_state)

    def _cached_scan(self, key: tuple, build: Callable[[], Callable]) -> Callable:
        """Return the jitted scan program for *key*, building it once.

        Building a scan means tracing the whole in-XLA loop and
        compiling it: hundreds of milliseconds upward.  ``run_scan`` and
        its siblings used to do that on every call, so a caller driving
        a simulation from a Python loop, a slider or an HTTP handler
        paid a compile per call.

        Parameters
        ----------
        key : tuple
            Everything that determines the built program *beyond* the
            graph structure: which entry point, the scan length, and any
            Python value the body closes over (the adaptive controller's
            constants, for instance).  ``_compile_generation`` is
            prepended, which covers the structure -- every graph
            mutation marks the graph dirty, every entry point recompiles
            a dirty graph before it gets here, and ``compile()`` clears
            this cache.
        build : callable
            Zero-argument builder returning the jitted program.  Called
            only on a miss.

        Returns
        -------
        Callable
            The cached (or freshly built) jitted program.

        Notes
        -----
        Shapes and dtypes of the runtime values are deliberately *not*
        in the key.  State, external inputs and params are arguments of
        the jitted program rather than constants closed over by it, so
        JAX's own cache retraces when an aval changes and reuses the
        compilation when it does not.  Baking them into the key instead
        would mean hashing arrays -- and closing over them, as the old
        code did, is what made the rebuild mandatory in the first place.
        """
        full_key = (self._compile_generation,) + tuple(key)
        fn = self._scan_cache.get(full_key)
        if fn is None:
            fn = build()
            # Bounded, because an HTTP handler taking ``n_steps`` from the
            # request would otherwise grow this without limit.  Insertion
            # order eviction: a workload cycling over more than
            # ``_SCAN_CACHE_MAX`` distinct step counts is pathological,
            # and pays what it used to pay on every call.
            if len(self._scan_cache) >= _SCAN_CACHE_MAX:
                self._scan_cache.pop(next(iter(self._scan_cache)))
            self._scan_cache[full_key] = fn
        return fn

    def run_scan(
        self,
        n_steps: int,
        external_inputs: Optional[dict[str, dict]] = None,
        *,
        params: Optional[dict] = None,
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
            Zeros are used for every declared input this does not
            supply, ``None`` included; an undeclared ``node.field``
            is a ``ValueError`` naming the declared ones.
        params : dict, optional
            Graph parameter pytree; ``None`` uses :attr:`params`.

        Returns
        -------
        dict[str, dict]
            The final state of the graph after *n_steps* (excluding
            internal metadata).
        """
        self._recover_from_escaped_tracers()
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()

        external_inputs = self._resolve_external_inputs(external_inputs)
        params = self._params_or_default(params)

        # External inputs and params are *arguments* of the jitted scan,
        # not constants closed over by it: that is what lets the built
        # program be cached across calls whose values differ.
        def build():
            step_fn = self._build_step_fn()

            def scan(state, ext, params):
                self._count_scan_trace()

                def scan_body(carry, _unused):
                    return step_fn(carry, ext, params), None

                final, _ = jax.lax.scan(
                    scan_body, state, None, length=int(n_steps),
                )
                return final

            return jax.jit(scan)

        fn = self._cached_scan(("run_scan", int(n_steps)), build)
        self._store_state(fn(self._state, external_inputs, params))
        return self._user_state(self._state)

    def run_scan_with_history(
        self,
        n_steps: int,
        external_inputs: Optional[dict[str, dict]] = None,
        *,
        params: Optional[dict] = None,
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
            Zeros are used for every declared input this does not
            supply, ``None`` included; an undeclared ``node.field``
            is a ``ValueError`` naming the declared ones.

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
        self._recover_from_escaped_tracers()
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()

        external_inputs = self._resolve_external_inputs(external_inputs)
        params = self._params_or_default(params)

        def build():
            step_fn = self._build_step_fn()

            def scan(state, ext, params):
                self._count_scan_trace()

                def scan_body(carry, _unused):
                    new_state = step_fn(carry, ext, params)
                    return new_state, new_state  # carry, stacked output

                return jax.lax.scan(
                    scan_body, state, None, length=int(n_steps),
                )

            return jax.jit(scan)

        fn = self._cached_scan(("run_scan_with_history", int(n_steps)), build)
        final_state, history = fn(self._state, external_inputs, params)
        self._store_state(final_state)
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
        *,
        params: Optional[dict] = None,
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
            Completed and validated as in :meth:`step`.
        return_history : bool
            If True, return ``(final_states, histories)`` where
            histories has shape ``(batch, n_steps, ...)``.
            If False (default), return only ``final_states``.

        Notes
        -----
        Multi-rate and coupled graphs are supported.  Their internal
        ``_meta`` (the sub-step counter, the coupling diagnostics and the
        predictor / IQN warm starts) is not part of ``initial_states``:
        every simulation in the batch starts from the graph's current
        ``_meta`` and evolves its own copy from there, and none of it
        appears in the returned states.  Pass an explicit ``_meta`` entry
        in ``initial_states`` — batched like any other leaf — to start
        each simulation from a different phase.

        Returns
        -------
        final_states : dict[str, dict]
            Batched final states (leading batch dimension on each leaf).
        histories : dict[str, dict], optional
            Only if ``return_history=True``.  Batched histories with
            shape ``(batch, n_steps, ...)`` on each leaf.
        """
        self._recover_from_escaped_tracers()
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()

        external_inputs = self._resolve_external_inputs(external_inputs)
        params = self._params_or_default(params)

        # Every other entry point carries ``self._state``, which
        # ``compile()`` seeded with ``_meta``; the batched carry is the
        # caller's ``initial_states``, which has none.  Without this a
        # multi-rate graph raised ``KeyError: '_meta'`` and a coupled one
        # with diagnostics a scan carry mismatch, though nothing
        # documented either as unsupported.  Passed as an argument rather
        # than closed over so a cached program cannot serve a stale
        # counter, and unbatched inside ``vmap`` so each simulation forks
        # its own copy of the warm start.
        meta = self._state.get(_META_KEY)

        def build():
            step_fn = self._build_step_fn()

            def sweep(init_states, ext, params, meta0):
                self._count_scan_trace()

                def simulate(init_state):
                    carry = dict(init_state)
                    if meta0 is not None:
                        carry.setdefault(_META_KEY, meta0)

                    def scan_body(state, _unused):
                        new_state = step_fn(state, ext, params)
                        return new_state, (new_state if return_history else None)

                    final, hist = jax.lax.scan(
                        scan_body, carry, None, length=int(n_steps),
                    )
                    if return_history:
                        return self._user_state(final), self._user_state(hist)
                    return self._user_state(final)

                return jax.vmap(simulate)(init_states)

            return jax.jit(sweep)

        fn = self._cached_scan(
            ("run_sweep", int(n_steps), bool(return_history)), build,
        )
        return fn(initial_states, external_inputs, params, meta)

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

        params_snapshot = self.params

        def _resolve_and_update(node_name, new_state, full_state, ext, dt,
                                node_params, force_forward_edges=None):
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
                value = _apply_edge(edge, value, node_params)
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
            return _node_update(
                spec, new_state[node_name], boundary_inputs, dt,
                node_params.nodes.get(node_name),
            )

        def dt_step_fn(state, external_inputs, dt, params=None):
            if params is None:
                params = params_snapshot
            else:
                self._validate_params(params)
            node_params = _ResolvedParams(
                params.get("nodes", {}), params.get("mappings", {}),
            )
            new_state = {k: v for k, v in state.items()}

            if has_coupling:
                for block in blocks:
                    if block[0] == "node":
                        nn = block[1]
                        new_state[nn] = _resolve_and_update(
                            nn, new_state, state, external_inputs, dt,
                            node_params,
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
                            node_params=node_params,
                        )
            else:
                for nn in schedule:
                    new_state[nn] = _resolve_and_update(
                        nn, new_state, state, external_inputs, dt,
                        node_params,
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
        *,
        params: Optional[dict] = None,
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
            Static external inputs applied every step.  Completed and
            validated as in :meth:`step`.
        callback : callable, optional
            Called after every *accepted* step with
            ``(sim_time, dt_used, state_dict)``.
        params : dict, optional
            Graph parameter pytree (see :attr:`params`).  ``None`` uses
            :attr:`params`.

        Returns
        -------
        (final_state, info) : tuple
            *final_state* is the state dict (excluding metadata).
            *info* is a dict with ``n_steps``, ``n_rejected``,
            ``dt_history`` (list of used timesteps), and
            ``t_history`` (list of simulation times).
        """
        self._recover_from_escaped_tracers()
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()
        # Judged after the recompile, not before it.  ``_is_multirate`` is
        # derived by ``compile()``, so on a graph that has been edited --
        # or never compiled -- it describes the step last built rather
        # than the one about to run.  Asked first, the refusal let a graph
        # that had just become multi-rate through, and refused one that
        # had just stopped being multi-rate.
        if self._is_multirate:
            raise RuntimeError(
                "Adaptive timestepping is incompatible with multi-rate "
                "graphs.  All nodes must share the same timestep."
            )

        external_inputs = self._resolve_external_inputs(external_inputs)

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
        params = self._params_or_default(params)

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
            state_full = dt_step_jit(state, external_inputs, dt_jax, params)
            # Two half-steps
            half_dt = dt_jax / 2.0
            state_half = dt_step_jit(state, external_inputs, half_dt, params)
            state_half = dt_step_jit(state_half, external_inputs, half_dt, params)

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

        self._store_state(state)
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
        *,
        params: Optional[dict] = None,
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
            Static external inputs.  Completed and validated as in
            :meth:`step`.
        params : dict, optional
            Graph parameter pytree (see :attr:`params`).  ``None`` uses
            :attr:`params`.  It is a traced argument of the scan, so
            ``jax.grad`` reaches it without writing a tracer into
            :attr:`params`.

        Returns
        -------
        (final_state, history, info) : tuple
            *final_state*: state after last accepted step.
            *history*: stacked state at each step (shape ``(max_steps, ...)``).
            *info*: dict with ``n_steps`` (actual steps taken, as JAX array).
        """
        self._recover_from_escaped_tracers()
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()
        # After the recompile, for the reason given in ``run_adaptive``.
        if self._is_multirate:
            raise RuntimeError(
                "Adaptive timestepping is incompatible with multi-rate graphs."
            )

        external_inputs = self._resolve_external_inputs(external_inputs)

        from maddening.core.simulation.adaptive import AdaptiveConfig

        config = AdaptiveConfig(
            dt_initial=dt_initial, atol=atol, rtol=rtol,
            dt_min=dt_min, dt_max=dt_max,
        )

        # The tolerances and the time bounds ride in as arguments
        # (``knobs``) so a caller sweeping them reuses the compilation;
        # only the controller constants, which this signature does not
        # expose, are baked in and therefore part of the cache key.
        fn = self._cached_scan(
            ("run_adaptive_scan", int(max_steps), config.safety,
             config.order, config.min_factor, config.max_factor),
            lambda: _build_adaptive_scan(
                self._build_dt_step_fn(), self._user_state, int(max_steps),
                config.safety, config.order, config.min_factor,
                config.max_factor, self._count_scan_trace,
            ),
        )
        knobs = (
            jnp.array(t_end), jnp.array(dt_initial), jnp.array(atol),
            jnp.array(rtol), jnp.array(dt_min), jnp.array(dt_max),
        )
        (final_state, final_t, final_dt, n_accepted), history = fn(
            self._state, external_inputs, self._params_or_default(params), knobs,
        )

        self._store_state(final_state)
        info = {"n_steps": n_accepted, "final_t": final_t, "final_dt": final_dt}
        return self._user_state(final_state), history, info

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_node_state(self, name: str) -> dict:
        """The node's current state fields.

        A fresh dict, not the internal one: the arrays inside are
        immutable and shared, but writing a new value into the dict you
        are handed does not reach the simulation.  Use
        :meth:`set_node_state` for that.
        """
        if name not in self._nodes:
            raise KeyError(f"No node named '{name}'.")
        if name not in self._state:
            raise KeyError(f"No node named '{name}'.")
        return dict(self._state[name])

    def set_node_state(self, name: str, state: dict) -> None:
        """Overwrite one node's state fields.

        A traced value is accepted -- writing the argument of a loss in
        is how a differentiable initial condition is expressed -- and
        noted, so the graph can be put back afterwards rather than
        keeping the tracer (see ``_recover_from_escaped_tracers``).
        """
        if name not in self._nodes:
            raise KeyError(f"No node named '{name}'.")
        state = _strong_typed(state)
        if _holds_tracer({name: state}) and not self._state_traced:
            # Snapshot before the write, per node, so the recovery has
            # something untraced to go back to.  Only on the tracer path,
            # so an ordinary call allocates nothing extra.
            self._state_before_trace = {
                k: (dict(v) if type(v) is dict else v)
                for k, v in self._state.items()
            }
            self._state_traced = True
        self._state[name] = state

    def reset_state(self) -> None:
        """Reset every node to its ``initial_state()`` and the internal
        counters in ``_meta`` to zero, keeping the compiled step valid.

        Prefer this to assigning ``initial_state()`` into ``_state``
        directly: the seed values are normalised the way ``compile``
        normalises them (weak types stripped), so the jitted step does not
        retrace after a reset, and ``_meta``'s structure is preserved.
        """
        # Every ``initial_state()`` first, then one commit: an
        # ``initial_state`` that raises (an ``AdaptiveNode`` at a Palais
        # trap) must not leave half the graph reset and half of it carrying
        # the state from before the call.  ``_meta`` is computed in the
        # same pass and committed with them: it is state too, so a reset
        # that leaves the node states fresh and the sub-step phase stale
        # would re-phase a multi-rate graph exactly the way a recompile
        # used to.
        fresh = {
            name: _strong_typed(spec.node.initial_state())
            for name, spec in self._nodes.items()
        }
        live_meta = self._state.get(_META_KEY)
        fresh_meta = None
        if live_meta is not None:
            fresh_meta = dict(live_meta)
            for key, value in live_meta.items():
                if key in ("step_count", "sub_step") or key.endswith("_iterations") \
                        or key.endswith("_pred_count"):
                    fresh_meta[key] = jnp.zeros_like(value)
                elif key.endswith("_residual") or key.endswith(
                    "_amplification"
                ):
                    fresh_meta[key] = jnp.zeros_like(value)
                # IQN V/W and predictor histories are warm-start caches:
                # zeroing them restarts cleanly too.
                elif key.endswith("_V") or key.endswith("_W") or "_pred_" in key:
                    fresh_meta[key] = jnp.zeros_like(value)

        self._state.update(fresh)
        if fresh_meta is not None:
            # Refilled, not replaced: the compiled step and any caller
            # holding the dict keep the object they were given.
            live_meta.clear()
            live_meta.update(fresh_meta)
        # Cleared last, with the commit: a reset that raised would
        # otherwise have taken the graph's way back from an escaped
        # tracer with it while leaving the tracer in place.
        self._state_traced = False
        self._state_before_trace = None

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

    def _warn_about_unsaved_mapping_weights(self) -> None:
        """``UserWarning`` for every mapped edge whose live weights are no
        longer the ones its recipe rebuilds.

        A config carries the ``MappingSpec``, not the weights, so weights
        moved by ``sysid`` or edited in ``params["mappings"]`` are dropped
        by a config-only round trip (checkpoints carry them, and win).
        The graph is the only place that knows both, so it says so here.
        """
        live = (self.params or {}).get("mappings") or {}
        for e in self._edges:
            if e.mapping is None:
                continue
            weights = live.get(e.key)
            if not weights:
                continue
            recipe = e.mapping.params_pytree()
            drifted = sorted(
                k for k, v in weights.items()
                if k in recipe and not (
                    np.shape(v) == np.shape(recipe[k])
                    and np.array_equal(np.asarray(v), np.asarray(recipe[k]))
                )
            )
            if drifted:
                warnings.warn(
                    f"edge {e.key}: live mapping weights {drifted} differ from what the "
                    f"MappingSpec rebuilds; a config carries the recipe only, so these "
                    f"values are not in it — save a checkpoint (gm.save_state(...)) and "
                    f"load it after the config to keep them",
                    UserWarning, stacklevel=3,
                )

    def to_dict(self, *, strict_mappings: bool = True) -> dict:
        """Serialise the graph structure (not runtime state).

        Node ``params`` are the *effective* values — the constructor
        arguments with the live :attr:`params` written over them (see
        :meth:`effective_node_params`) — and ``param_specs`` carries the
        graph's :meth:`set_param_spec` overrides.  An edge's interface
        mapping is written as its
        :class:`~maddening.core.coupling.mapping_spec.MappingSpec` (kind,
        hyper-parameters, point references — never the weights, which
        checkpoints carry).  With ``strict_mappings`` (the default) a
        mapping that cannot be rebuilt from a config — no spec, or a
        point set neither referenced nor small enough to inline — is a
        ``ValueError`` naming the ``source_ref=`` / ``target_ref=`` /
        ``asset=`` argument to pass — as is a node reference that no
        longer resolves to the points it was built from (a removed node,
        a renamed field).  ``strict_mappings=False`` writes whatever the
        mapping describes, for display, and checks nothing.

        Only the *recipe* is written, so live weights that were trained
        or hand-edited away from it would be lost: a ``UserWarning``
        naming the edge says so, pointing at :meth:`save_state`.

        ``coupling_groups`` carries *every* field of every group (see
        :meth:`~maddening.core.coupling.group.CouplingGroup.to_dict`),
        and is absent when the graph has none.  Partial would be worse
        than nothing: a group that came back missing its acceleration or
        its iteration cap would still be a group, and would quietly
        solve the same graph a different way.

        The result is JSON-*valid*, not merely JSON-shaped: a non-finite
        float anywhere in it -- a diverged param, an infinite ParamSpec
        bound -- is the quoted token ``"NaN"`` / ``"Infinity"`` /
        ``"-Infinity"``, not the bare token ``json.dumps`` writes by
        default and no conforming reader accepts (``MADD-ANO-006``).
        :meth:`from_dict` turns those back into floats, and reads the
        bare tokens an older config carries as well.  The encoding is
        applied once, to the assembled tree, so each part's own
        ``to_dict`` still returns plain floats for the callers that want
        numbers (the USD writer sets typed stage attributes from
        :meth:`CouplingGroup.to_dict`).  See
        :mod:`maddening.serialization.json_codec`.

        Because the encoding is applied here rather than at the write
        boundary -- so that plain ``json.dumps`` of this result is valid,
        which is what every caller in the tree does -- the result is
        *already encoded*.  Write it with ``json.dumps`` or
        :func:`~maddening.serialization.json_codec.dumps_encoded`, and
        **not** with :func:`~maddening.serialization.json_codec.dumps`:
        that one encodes what it is given, the encoding is not
        idempotent, and the second walk refuses the tokens the first one
        wrote (``$.param_specs.<node>.<key>.bounds[0]: the string
        '-Infinity' cannot be written to JSON``, from any graph with an
        unbounded :class:`~maddening.core.params.ParamSpec`).  Read it
        back with :func:`~maddening.serialization.json_codec.loads`,
        which *is* composable.
        """
        from maddening.serialization.json_codec import (  # noqa: PLC0415
            encode_non_finite,
        )
        if strict_mappings:
            from maddening.core.coupling.mapping_spec import (  # noqa: PLC0415
                check_mapping_serialisable,
            )
            resolve = self.point_resolver()
            for e in self._edges:
                if e.mapping is not None:
                    check_mapping_serialisable(e.mapping, edge_key=e.key,
                                               resolve_points=resolve)
            self._warn_about_unsaved_mapping_weights()
        nodes = []
        for name, spec in self._nodes.items():
            d = spec.node.to_dict()
            if spec.accepts_params:
                d["params"] = self.effective_node_params(name)
            nodes.append(d)
        overrides = {
            n: {k: s.to_dict() for k, s in o.items()}
            for n, o in self.param_spec_overrides().items()
        }
        return encode_non_finite({
            "nodes": nodes,
            **({"param_specs": overrides} if overrides else {}),
            "edges": [e.to_dict() for e in self._edges],
            "external_inputs": [ei.to_dict() for ei in self._external_inputs],
            # Every field of every group, or the key is absent: a config
            # that carried only some of a group's solver settings would
            # reload as a graph that *runs* differently -- a fixed point
            # iterated to convergence becoming a single staggered pass --
            # without anything saying so.  Absent, like ``param_specs``,
            # when there is nothing to say, so an uncoupled graph writes
            # exactly the config it wrote before this key existed.
            **({"coupling_groups": [g.to_dict() for g in self._coupling_groups]}
               if self._coupling_groups else {}),
        })

    @classmethod
    def from_dict(
        cls,
        config: dict,
        node_registry: dict[str, type],
        *,
        base_dir=None,
    ) -> "GraphManager":
        """Reconstruct a GraphManager from a serialised config.

        *node_registry* maps node type names (e.g. ``"BallNode"``) to
        the corresponding class.  Edge mappings are rebuilt from their
        ``MappingSpec`` (node-field references resolve against the
        nodes just created; ``{"asset": ...}`` paths are relative to
        ``base_dir``, the directory the config was read from — the
        working directory when ``None``) and registered in
        ``params["mappings"]`` exactly as ``add_edge(mapping=)`` does.
        A checkpoint loaded afterwards overwrites the rebuilt weights.

        ``param_specs`` overrides are applied *after* the edges, because
        an override may name a mapped edge's key (``set_param_spec(
        edge.key, "H", ParamSpec())`` — trainable mapping weights) and
        those slots only exist once the edge does; node overrides do not
        depend on the edges, so the order is safe for them too.

        ``coupling_groups`` are rebuilt with :meth:`add_coupling_group`,
        so a stored group is checked exactly like a hand-written one; a
        group that cannot be rebuilt — an unknown node, a node already
        in another group, a misspelled enum — raises ``ValueError``
        naming the group and what is wrong with it.  A config without
        the key (one written before it existed) loads unchanged.

        Non-finite numbers are decoded first, so both spellings load:
        the quoted ``"NaN"`` / ``"Infinity"`` / ``"-Infinity"`` that
        :meth:`to_dict` writes since 0.4.0, and the bare tokens an older
        config carries, which ``json.loads`` has already turned into
        floats by the time the dict arrives here (``MADD-ANO-006``).
        """
        from maddening.serialization.json_codec import (  # noqa: PLC0415
            decode_non_finite,
        )
        config = decode_non_finite(config)
        gm = cls()
        for nd in config["nodes"]:
            node_cls = node_registry[nd["type"]]
            node = node_cls(name=nd["name"], timestep=nd["timestep"], **nd.get("params", {}))
            gm.add_node(node)
        resolve = gm.point_resolver(base_dir)
        for ed in config["edges"]:
            mapping = None
            if ed.get("mapping") is not None:
                mapping = gm._rebuild_mapping(ed, resolve)
            gm.add_edge(
                source=ed["source_node"],
                target=ed["target_node"],
                source_field=ed["source_field"],
                target_field=ed["target_field"],
                transform=ed.get("transform"),          # registered name
                additive=bool(ed.get("additive", False)),
                source_units=ed.get("source_units"),
                target_units=ed.get("target_units"),
                mapping=mapping,
            )
        for owner, overrides in config.get("param_specs", {}).items():
            for key, spec_dict in overrides.items():
                try:
                    gm.set_param_spec(owner, key, ParamSpec.from_dict(spec_dict))
                except KeyError as exc:
                    # set_param_spec reports an unknown owner / key as a
                    # KeyError; the loader speaks ValueError like the rest
                    # of from_dict, and names what it was applying.
                    raise ValueError(
                        f"param_specs[{owner!r}][{key!r}] cannot be applied to this "
                        f"config ({owner!r} is neither one of its nodes nor one of its "
                        f"mapped edge keys): {exc}"
                    ) from exc
        for ei in config.get("external_inputs", []):
            spec = ExternalInputSpec.from_dict(ei)
            gm.add_external_input(
                target_node=spec.target_node,
                target_field=spec.target_field,
                shape=spec.shape,
                dtype=spec.dtype,
            )
        for i, cg in enumerate(config.get("coupling_groups", [])):
            # Straight back through ``add_coupling_group``, so a loaded
            # group is checked by the same code as a hand-written one:
            # the node names against this graph, the node set against the
            # groups already registered, and every enum by
            # ``CouplingGroup.__post_init__``.  What those checks do not
            # know is *which* group of a multi-group config they are
            # talking about, which is the only thing that makes a
            # hand-edited file actionable -- so name it here.
            try:
                nodes, kwargs = coupling_group_kwargs(cg)
                gm.add_coupling_group(nodes, **kwargs)
            except (KeyError, TypeError, ValueError) as exc:
                named = ""
                if isinstance(cg, dict) and isinstance(cg.get("nodes"), (list, tuple)):
                    named = f" (nodes {sorted(cg['nodes'])})"
                # ``str(KeyError)`` is the *repr* of its message; unwrap it,
                # and say what a bare missing key means.
                detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
                if isinstance(exc, KeyError) and detail == "nodes":
                    detail = "it has no 'nodes' key"
                raise ValueError(
                    f"coupling_groups[{i}]{named} cannot be rebuilt: {detail}"
                ) from exc
        return gm

    @staticmethod
    def _rebuild_mapping(edge_dict: dict, resolve_points) -> Any:
        """The mapping of a serialised edge, rebuilt from its spec dict and
        checked against the ``shape`` recorded with it.

        *Every* way a spec can fail — a malformed dict, a reference that
        does not resolve or no longer matches, an unreadable / oversized
        asset, a hyper-parameter of the wrong type, a singular solve —
        comes back as a
        :class:`~maddening.core.coupling.mapping_spec.MappingRebuildError`
        (a ``ValueError``) naming this edge, with the original exception
        chained: a config is untrusted input and the edge it broke on is
        the only thing that makes the failure actionable.
        """
        import zipfile  # noqa: PLC0415
        from maddening.core.coupling.mapping_spec import (  # noqa: PLC0415
            MappingRebuildError,
            MappingSpec,
        )
        # MappingRebuildError prefixes "edge "; this is just the key.
        where = (f"{edge_dict['source_node']}.{edge_dict['source_field']} -> "
                 f"{edge_dict['target_node']}.{edge_dict['target_field']}")
        d = edge_dict["mapping"]
        kind = d.get("kind") if isinstance(d, dict) else None
        try:
            mapping = MappingSpec.from_dict(d).build(resolve_points)
            shape = d.get("shape")
            if shape is not None:
                if (isinstance(shape, (str, bytes)) or not isinstance(shape, (list, tuple))
                        or len(shape) != 2
                        or any(isinstance(s, bool) or not isinstance(s, int) for s in shape)):
                    raise ValueError(
                        f"'shape' must be a two-element list of ints "
                        f"[n_target, n_source], got {shape!r}"
                    )
                if [mapping.n_target, mapping.n_source] != list(shape):
                    raise ValueError(
                        f"rebuilt mapping has shape "
                        f"{[mapping.n_target, mapping.n_source]} but the config recorded "
                        f"{list(shape)}; the referenced point sets changed"
                    )
        except (ValueError, TypeError, KeyError, OSError, MemoryError,
                zipfile.BadZipFile) as exc:
            # json.JSONDecodeError is a ValueError and numpy's
            # UFuncTypeError a TypeError, so both land here too.
            raise MappingRebuildError(where, kind, exc) from exc
        return mapping

    # ------------------------------------------------------------------
    # Checkpoint / restore
    # ------------------------------------------------------------------

    def save_state(self, path) -> "Path":
        """Save all node states to an ``.npz`` file.

        See :func:`maddening.core.checkpoint.save_state` for details.
        """
        from maddening.core.simulation.checkpoint import save_state
        self._recover_from_escaped_tracers()
        return save_state(self, path)

    def load_state(self, path) -> None:
        """Load node states from an ``.npz`` file.

        See :func:`maddening.core.checkpoint.load_state` for details.
        """
        from maddening.core.simulation.checkpoint import load_state
        self._recover_from_escaped_tracers()
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
