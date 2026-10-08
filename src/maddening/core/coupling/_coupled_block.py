"""The coupled-block runner (``_run_coupled_block_impl``).

Moved verbatim out of ``maddening.core.graph_manager``.  Private.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core._pow2_frame import pow2_frame
from maddening.core.coupling.acceleration import (
    _field_reference,
    _interface_readings,
    _reading_eps,
    float_fields_of,
    residual_precision_floor,
    spectral_rate_settled,
    state_float_image,
    state_from_float_image,
)
from maddening.core._graph_specs import (
    _META_KEY,
    _apply_edge,
    _edge_geom,
    _traceable_geometry,
    _correction_accepts_params,
    _node_fluxes,
    _node_update,
)
from maddening.core.coupling._interface_plan import interface_plan
from maddening.core.coupling._group_layout import (
    _group_accel_fields,
    _group_dividers,
    _group_evaluations,
    _geometry_diagnostics_refusal,
    _group_pass_structure,
    _group_residual_dtype,
    _group_state_finite,
    _group_waveform_sweeps,
    _non_finite_reads_as_diverged,
    _reading_is_the_fields,
    _reads_mapping_weights,
    _state_measurable,
)
from maddening.core.coupling._bounds import (
    _analysis_dtype,
    _geometry_product_gap,
    _gradient_error_bound_at,
    _interface_spectral_rate_at,
    _residual_resolution,
    _spectral_rate_at,
    _whole_probe_names,
)
from maddening.core.coupling._ift import _ift_solve
from maddening.core.coupling._reports import (
    _strict_convergence_messages,
    _strict_error_if,
)


@jax.custom_jvp
def _interpolate(a, b, alpha):
    return a + alpha * (b - a)


@_interpolate.defjvp
def _interpolate_jvp(primals, tangents):
    a, b, alpha = primals
    a_dot, b_dot, alpha_dot = tangents
    return _interpolate(a, b, alpha), (
        (1.0 - alpha) * a_dot + alpha * b_dot + alpha_dot * (b - a))


def _interpolated(a, b, alpha):
    """``a + alpha * (b - a)``, differentiated as ``(1 - alpha) a + alpha b``.

    The value is the expression a sub-cycled member's boundary has always
    been interpolated with, to the bit.  Its tangent by the chain rule is
    ``a_dot + alpha * (b_dot - a_dot)``, which rounds ``b_dot`` at one
    ``eps`` of ``a_dot``: at the last sub-step (``alpha = 1``) the exact
    tangent is ``b_dot`` alone, and under Gauss-Seidel ``a`` is the
    iterate and ``b`` this pass's value of the same field, so a loop whose
    gain passes through a change of that field below its rounding left the
    Jacobian-vector product altogether -- a float32 ring with a field 1e-6
    of what it drives read ``rho_spectral = 1e-12`` for a radius of
    1.25e-4, ``spectral_usable=True`` (MADD-ANO-222).  The rule below
    weights the two tangents separately, which is exact at both ends and
    rounds each at its own size between them.  A leaf that is not
    floating is interpolated as before.
    """
    if not (jnp.issubdtype(jnp.result_type(a), jnp.inexact)
            and jnp.issubdtype(jnp.result_type(b), jnp.inexact)):
        return a + alpha * (b - a)
    return _interpolate(a, b, alpha)


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
    multigpu_device_map=None, node_params=None, fires=None,
    strict_sink=None, probe_sink=None, strict_mesh=None,
):
    """Execute a coupling group with iterative fixed-point iteration.

    Supports Gauss-Seidel and Jacobi iteration modes, multiple
    convergence norms (L2, mixed, interface), acceleration methods
    (Aitken, fixed relaxation, IQN-ILS, IQN-IMVJ), additive edges,
    flux-based coupling, subcycling with linear/quadratic/constant
    interpolation, and repeated waveform sweeps (restarts of the same
    solve, not waveform relaxation: MADD-ANO-027).

    This is the shared implementation used by both ``_build_step_fn``
    and ``_build_dt_step_fn``.

    ``runtime_dt`` (``run_adaptive*``) is the step every node advances
    by.  A sub-cycled node still takes ``round(macro_dt / node_dt)``
    sub-steps per pass, so each is scaled to
    ``runtime_dt * node_dt / macro_dt``: at ``runtime_dt == macro_dt``
    that is the node's own timestep, the sub-step the compiled step
    takes.  Handing every sub-step the whole ``runtime_dt`` advanced the
    fast node by ``divider * runtime_dt`` per step.

    ``fires`` is ``None``, or -- on a multi-rate graph, for a group
    whose rate divider is above one -- the traced boolean saying whether
    this base step applies the group's solve at all.
    ``graph_step_multirate`` runs this block under a ``lax.cond`` on it,
    so a step that does not fire does not solve; but a batched ``cond``
    (under ``vmap``) runs both branches, and ``strict_convergence`` must
    not raise about a solve the step discards, so its predicates are
    gated on ``fires`` here as well.

    ``strict_sink`` is ``None``, or a list: then ``strict_convergence``
    raises nothing here and appends each checked solve's ``(non-finite,
    unconverged)`` predicates to it instead, for a caller that knows only
    later whether the step keeps the solve -- ``run_adaptive*``, whose
    error-estimate full step and rejected attempts are discarded.

    ``strict_mesh`` is the device mesh of a step that spans more than one
    device (:func:`_multi_device_mesh`), on every device of which
    ``strict_convergence`` raises (:func:`_strict_error_if`); ``None`` on
    one device.
    """
    from maddening.core.coupling.acceleration import (
        aitken_relaxation,
        coupling_residual_interface,
        coupling_residual_l2,
        coupling_residual_mixed,
        error_amplification,
        estimated_error,
        first_pass_relaxed_amplification,
        fixed_relaxation,
        flatten_coupled_state,
        iqn_ils_update,
        relaxation_step_scale,
        relaxes_first_pass,
        unflatten_coupled_state,
    )

    max_iters = group.max_iterations
    # How much longer the iterate's step is than the residual that is
    # measured -- ``relaxation`` under ``acceleration="fixed"``, 1.0
    # otherwise.  Static, and identical on both solver paths so
    # ``solver`` stays invisible in ``coupling_diagnostics()``.
    step_scale = relaxation_step_scale(group.acceleration, group.relaxation)
    # Static: whether the first loop pass's amplification is relaxed
    # (``first_pass_relaxed_amplification``), identically on both solver
    # paths.  Only a loop for which it is true reads its pass index.
    relax_first = relaxes_first_pass(group.acceleration, group.relaxation)
    group_node_names = list(group_schedule)
    _node_params = node_params.nodes if node_params is not None else {}

    def _np(nn):
        return _node_params.get(nn)
    group_node_set = set(group_node_names)
    use_mixed_norm = group.convergence_norm == "mixed"
    use_interface_norm = group.convergence_norm == "interface"
    use_acceleration = group.acceleration != "none"
    use_jacobi = group.iteration_mode == "jacobi"

    # The group's edges, described once (``_interface_plan``): everything
    # below that asks which edges are internal, what one delivers or which
    # fields are read asks this.  Edges internal to the group are forced
    # forward; the interface norm sums over them in an order the group
    # fixes (``plan.internal``), not the order of the ``add_edge`` calls.
    plan = interface_plan(group.nodes, all_edges, group_node_names, new_state, nodes)
    # The mapping weights this step runs with (``params["mappings"]``,
    # baked or traced as the step has them): what ``_resolve_boundary``
    # hands ``_apply_edge``, and so what every reading of an interface
    # edge below is taken with.
    step_mappings = node_params.mappings if node_params is not None else None
    # Whether the interface norm's reading depends on those weights
    # (static).  Such a group's float floor is recorded by the step.
    reads_mapping_weights = _reads_mapping_weights(group, plan)
    # The same weights for what *reports on* the returned state rather
    # than taking part in the map -- the spectral analysis's reading and
    # the recorded floor: read as the step has them, and differentiated
    # with respect to nothing (the analysis stops the gradient of the
    # state and of the pass's constants in the same way).  Only the
    # interface norm reads an edge, so only it builds them.
    report_mappings = (
        None if step_mappings is None or not use_interface_norm else {
            key: jax.tree.map(jax.lax.stop_gradient, step_mappings[key])
            for key in plan.mapped_keys() if key in step_mappings})

    # Precompute which boundary inputs come from coupling (intra-group) edges
    # per target node -- only these get interface correction
    coupled_bi_names_by_node = plan.coupled_inputs()

    # The geometry fields the diagnostics' self-check moves (experimental;
    # ``_geometry_product_gap``): those of the geometry-dependent mappings
    # this group's pass resolves, where the report reads a geometry at all
    # (``_geometry_diagnostics_refusal``).  Static, and empty for every
    # group without such an edge and for every group without diagnostics:
    # nothing below is traced for them.
    geometry_checked = (
        plan.geometry_holders()
        if group.diagnostics and group.solver == "ift"
        and _geometry_diagnostics_refusal(group, nodes, plan) is None
        else [])

    # The fields the acceleration flattens: IQN's interface (or
    # ``accelerated_fields``) set, and for ``"aitken"`` / ``"fixed"`` on
    # the fori path the whole group -- floating fields only, either way
    # (``_group_accel_fields``).  ``None`` flattens every field, which is
    # what an all-floating group still gets.  The IFT path relaxes
    # ``"aitken"`` / ``"fixed"`` on its own floating vector and reads
    # this only for IQN's index map, so it stays ``None`` there.
    if group.acceleration in ("iqn-ils", "iqn-imvj") or group.solver == "fori":
        accel_fields = _group_accel_fields(group, plan, new_state)
    else:
        accel_fields = None

    # The members that produce flux fields, and whether any edge with an
    # end in the group carries a flux (a source that is not in the state,
    # of a node that defines ``compute_boundary_fluxes``).
    flux_producing_nodes = plan.flux_members
    has_flux_edges = plan.resolves_a_flux()

    # How many sweeps the Jacobi pass seeds producers' fluxes in: two
    # when a producer in the group reads a flux (it needs another
    # producer's flux before it can compute its own; see
    # ``one_pass_jacobi``), one otherwise.  Static, so a group that
    # never needed the second sweep compiles to the program it always
    # did.  ``one_pass_gs`` always takes two.
    jacobi_flux_sweeps = (
        (False, True)
        if has_flux_edges and plan.flux_producer_reads_a_flux()
        else (True,)
    )

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

    # Compute subcycling rate dividers if needed (``None``: the group
    # does not sub-cycle, including ``subcycling=True`` over one timestep).
    group_dividers = _group_dividers(group, nodes) or {}
    use_subcycling = bool(group_dividers)
    # The group's macro timestep: the time one coupling pass covers.
    macro_dt = (max(nodes[nn].timestep for nn in group_dividers)
                if use_subcycling else None)

    def _get_dt(nn):
        spec = nodes[nn]
        if runtime_dt is None:
            return spec.timestep
        if use_subcycling and spec.timestep != macro_dt:
            # ``run_adaptive*``: the step covers ``runtime_dt``.  A member
            # of a sub-cycling group keeps its ratio to the macro step, so
            # its ``group_dividers[nn]`` sub-steps cover ``runtime_dt`` as
            # well (exactly, at an integer ratio), and at ``runtime_dt ==
            # macro_dt`` each is the node's own timestep, the sub-step the
            # compiled step takes.  See the docstring.
            return runtime_dt * (spec.timestep / macro_dt)
        return runtime_dt

    def _gate_on_firing(predicate):
        """``predicate``, but only on a base step that keeps this solve."""
        if fires is None:
            return predicate
        return jnp.logical_and(fires, predicate)

    _strict_nonfinite_msg, _strict_unconverged_msg = _strict_convergence_messages(group)

    def _strict_check(value, est, state_ok):
        """``strict_convergence`` on a solve whose error estimate is ``est``.

        ``state_ok`` is the verdict on the returned state itself
        (:func:`_state_measurable`), which decides which message an
        unconverged exit raises (``_strict_convergence_messages``): the
        estimate cannot, since a norm can overflow on a finite state.
        Raises in-graph about a solve the step keeps (``_gate_on_firing``),
        or, with ``strict_sink``, hands the two predicates to the caller.
        """
        # ``not (r <= t)``, not ``r > t``: see below.
        unmet = jnp.logical_not(est <= conv_threshold_value)
        nonfinite = jnp.logical_and(jnp.logical_not(state_ok), unmet)
        if strict_sink is not None:
            strict_sink.append((
                nonfinite,
                jnp.logical_and(state_ok, unmet),
            ))
            return value
        # Two checks with exclusive predicates, so the message names the
        # cause (``_strict_convergence_messages``).  Both are gated on the
        # step keeping this solve: a multi-rate base step computes and
        # discards it on every phase the group does not fire on, and used
        # to raise about a solve nothing applied.  On a step spanning
        # several devices both raise on every device (``strict_mesh``).
        value = _strict_error_if(value, _gate_on_firing(nonfinite),
                                 _strict_nonfinite_msg, strict_mesh)
        return _strict_error_if(
            # ``not (r <= t)`` rather than ``r > t``: the two differ
            # exactly on NaN, which answers False to both, and a NaN
            # residual is the one case where the IFT gradient is certainly
            # invalid.  A non-finite state is named by the check above; this
            # predicate stays in the closed form and is made exclusive of
            # it so exactly one message fires.  ``coupling_diagnostics()``
            # already reads a non-finite residual as ``converged=False``;
            # the guard has to agree.
            value,
            _gate_on_firing(jnp.logical_and(state_ok, unmet)),
            _strict_unconverged_msg,
            strict_mesh,
        )

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

    def _resolve_boundary(nn, s, flux_s=None, strict=True, consumer=None):
        """Resolve boundary inputs for node nn from state s.

        ``strict=False`` omits inputs whose flux is not available yet
        (used to seed fluxes from the previous iterate before a pass).

        *consumer* is the ``state`` argument of the hook these inputs are
        for (or a zero-argument callable returning it): what an edge with
        a target-anchored geometry reads its geometry from.  A
        source-anchored geometry is read from the dict the value is read
        from.  Both are read here, inside the pass, from the pass's own
        tracers: that is what makes the group's gradient see them.
        """
        boundary_inputs: dict[str, Any] = {}
        for edge in edges_by_target[nn]:
            if edge in back_edge_set and not plan.is_internal(edge):
                src_state = full_state
            else:
                src_state = s
            value = _resolve_value(edge, src_state, flux_s, strict=strict)
            if value is _MISSING:
                continue
            value = _apply_edge(edge, value, node_params,
                                _edge_geom(edge, src_state, consumer))
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

    use_linear_interp = group.boundary_interpolation == "linear"
    use_quadratic_interp = group.boundary_interpolation == "quadratic"
    # How many evaluations one pass rounds like, for the float floor the
    # diagnostics compare against (see ``_group_evaluations``).
    pass_evaluations, _declared = _group_evaluations(
        group, nodes, group_node_names, plan.declared_edges())
    gs_order, gs_own, gs_same_pass, _ = _group_pass_structure(
        group, nodes, group_node_names, plan.declared_edges())
    gs_reads = plan.member_reads()

    def _read_gain(s_star, src, dst):
        """Measured relative gain of ``dst``'s update in its read of ``src``.

        Measured for every group-internal read (from the same pass, the
        previous iterate, or ``dst`` itself): at the fixed point both read
        the same value.  One JVP of ``dst``'s update, at ``s_star``, along ``src``'s own
        state (a relative perturbation of every floating field of
        ``src`` by one), the response measured per floating field of
        ``dst`` relative to that field's magnitude, worst field taken:
        how many times a relative rounding of ``src`` grows in ``dst``.
        A squaring relay reads 2, ``3u - 2v`` at ``u ~ v`` reads about 3,
        an affine relay of gain ``g`` reads ``|g| * |u| / |x|``.  A read
        through a sub-cycled member counts as one (its update is several
        sub-steps; its own count already carries them).  The perturbation
        moves ``src``'s fluxes with its state, so a flux edge is measured
        through the flux.
        """
        if group_dividers.get(src, 1) > 1 or group_dividers.get(dst, 1) > 1:
            return jnp.float32(1.0)
        dst_pre = _pre(dst)
        src_state = s_star[src]
        floats = [f for f, v in src_state.items()
                  if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)]

        def dst_of(src_floats):
            s = {**s_star, src: {**src_state, **src_floats}}
            flux_s: dict[str, dict] = {}
            if has_flux_edges:
                # Two sweeps, as ``one_pass_gs`` seeds them: a producer may
                # read another producer's flux.
                for strict in (False, True):
                    for nn in group_node_names:
                        if nn in flux_producing_nodes:
                            flux_s[nn] = _node_fluxes(
                                nodes[nn], s[nn],
                                _resolve_boundary(nn, s, flux_s, strict=strict,
                                                  consumer=s[nn]),
                                _get_dt(nn), _np(nn))
            out = _node_update(nodes[dst], dst_pre,
                               _resolve_boundary(dst, s, flux_s, consumer=dst_pre),
                               _get_dt(dst), _np(dst))
            return {f: v for f, v in out.items()
                    if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)}

        primal = {f: src_state[f] for f in floats}
        out, d_out = jax.jvp(dst_of, (primal,), (primal,))
        gain = jnp.float32(0.0)
        for f, v in out.items():
            ref = jnp.max(jnp.abs(v)).astype(jnp.float32)
            change = jnp.max(jnp.abs(d_out[f])).astype(jnp.float32)
            gain = jnp.maximum(gain, jnp.where(ref > 0, change / jnp.where(ref > 0, ref, 1.0), 0.0))
        return gain

    def _resolve_boundary_interpolated(nn, s_prev, s_cur, alpha,
                                        flux_s=None, s_prev_prev=None, consumer=None):
        """Resolve boundary inputs with time interpolation.

        alpha=0 means start (s_prev values), alpha=1 means end (s_cur).
        Only interpolates edges that are internal to the coupling group.
        """
        boundary_inputs: dict[str, Any] = {}
        for edge in edges_by_target[nn]:
            geom = None
            if edge in back_edge_set and not plan.is_internal(edge):
                src_state = full_state
                value = _resolve_value(edge, src_state, flux_s)
                geom = _edge_geom(edge, src_state, consumer)
            elif plan.is_internal(edge):
                if edge.geometry is None:
                    pass
                elif edge.geometry[0] != "source":
                    geom = _edge_geom(edge, s_cur, consumer)
                else:
                    # A source-anchored geometry is interpolated as the
                    # value below is, between the same two (or three)
                    # snapshots with the same weight, componentwise, and
                    # the mapping is applied to the interpolated pair.
                    g_field = edge.geometry[1]
                    # The dtype rules ``_edge_geom`` asks of every other
                    # read, asked of both ends of this one.
                    g_prev = _traceable_geometry(
                        edge, g_field, s_prev[edge.source_node][g_field])
                    g_cur = _traceable_geometry(
                        edge, g_field, s_cur[edge.source_node][g_field])
                    if use_quadratic_interp and s_prev_prev is not None:
                        g_pp = s_prev_prev[edge.source_node][g_field]
                        geom = (
                            (1.0 - 3.0 * alpha + 2.0 * alpha * alpha) * g_pp
                            + (4.0 * alpha - 4.0 * alpha * alpha) * g_prev
                            + (-alpha + 2.0 * alpha * alpha) * g_cur
                        )
                    else:
                        geom = _interpolated(g_prev, g_cur, alpha)
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
                        lambda a, b: _interpolated(a, b, alpha), v_prev, v_cur
                    )
            else:
                value = _resolve_value(edge, s_cur, flux_s)
                geom = _edge_geom(edge, s_cur, consumer)
            value = _apply_edge(edge, value, node_params, geom)
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
                    flux_s=flux_s, s_prev_prev=s_prev_prev, consumer=sub_state,
                )
            else:
                # constant: use end-of-step values
                bi = _resolve_boundary(nn, s_cur, flux_s, consumer=sub_state)
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
                        bi0 = _resolve_boundary(nn, latest_results, flux_s, strict=strict,
                                                consumer=latest_results[nn])
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
                # The member integrates from its pre-step state, which is
                # therefore what a target-anchored geometry is read from;
                # lazily, so ``_pre`` is traced where it always was.
                bi = _resolve_boundary(nn, s, flux_s, consumer=lambda nn=nn: _pre(nn))
                pre = _pre(nn)
                s[nn] = _node_update(nodes[nn], pre, bi, _get_dt(nn), _np(nn))
                s[nn] = _apply_interface_overrides(
                    s[nn], pre, bi, _get_dt(nn), nodes[nn].node,
                    coupled_bi_names=coupled_bi_names_by_node.get(nn),
                    node_params=_np(nn),
                )
            # Compute fluxes for this node
            if nn in flux_producing_nodes:
                bi_for_flux = _resolve_boundary(nn, s, flux_s, consumer=s[nn])
                flux_s[nn] = _node_fluxes(
                    nodes[nn], s[nn], bi_for_flux, _get_dt(nn), _np(nn),
                )
        return s

    def one_pass_jacobi(latest_results):
        """Jacobi: all nodes read from frozen previous-iteration state."""
        # Pre-compute fluxes from previous iteration state.  A producer
        # whose own inputs include another producer's flux needs that
        # flux before it can compute its own, so such a group seeds in
        # two sweeps exactly as ``one_pass_gs`` does (the first tolerates
        # a missing flux, the second has them all); a single strict
        # sweep raised ``KeyError`` naming the flux at trace.  Every
        # other group keeps the single sweep, and its program.
        flux_s: dict[str, dict] = {}
        if has_flux_edges:
            for strict in jacobi_flux_sweeps:
                for nn in group_node_names:
                    if nn in flux_producing_nodes:
                        bi = _resolve_boundary(nn, latest_results, flux_s, strict=strict,
                                               consumer=latest_results[nn])
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
                bi = _resolve_boundary(nn, latest_results, flux_s,
                                       consumer=lambda nn=nn: _pre(nn))
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
        # In the group's residual dtype (``_group_residual_dtype``: the
        # promotion of its floating fields, at least float32), which every
        # loop carry and report slot holds it in.  The norms already return
        # it wherever the fields they read carry the group's widest dtype;
        # the cast is for the interface norm reading only narrower fields.
        res_dtype = _group_residual_dtype(s_new, group_node_names)
        with jax.named_scope("coupling:residual"):
            if use_interface_norm:
                # Each internal edge as the step delivers it: through its
                # mapping, with this step's weights, then its transform.
                return coupling_residual_interface(
                    s_new, s_old, plan,
                    group.atol, group.rtol, mappings=step_mappings,
                ).astype(res_dtype)
            if use_mixed_norm:
                return coupling_residual_mixed(
                    s_new, s_old, group_node_names,
                    group.atol, group.rtol,
                ).astype(res_dtype)
            return coupling_residual_l2(
                s_new, s_old, group_node_names, group.atol,
            ).astype(res_dtype)

    def _estimate(residual, prev_residual, prev2_residual, first=False):
        """``(estimated distance to the fixed point, amplification)``.

        The criterion both solvers stop on.  See
        :func:`_fixed_point_while` for the argument and for what
        happens when the ratio is rejected.  ``first`` says this is the
        loop's first pass, whose ratio spans the unrelaxed pre-loop pass
        (``first_pass_relaxed_amplification``).
        """
        amp = first_pass_relaxed_amplification(
            error_amplification(residual, prev_residual, prev2_residual),
            group.acceleration, group.relaxation, first)
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
    # Waveform sweeps (``waveform_iterations``; restarts, see below)
    # ------------------------------------------------------------------
    n_waveform = _group_waveform_sweeps(group, nodes)

    def _run_coupling_inner(new_state_inner):
        """Run the core coupling iteration once: one waveform sweep.

        ``initial_node_states`` (the beginning-of-timestep state that
        nodes integrate FROM) is never changed by a sweep.
        ``new_state_inner`` is only the iteration's starting guess (and
        carries the nodes outside the group): ``one_pass`` reads its
        input iterate and ``initial_node_states`` and nothing else a
        sweep changes, so every sweep iterates the same map and a later
        sweep resumes the fixed-point solve where the previous one
        stopped, with its accelerator started afresh.  That is a restart,
        not waveform relaxation, and the sub-step interpolation runs
        between the incoming iterate and the in-pass state, never from
        the beginning-of-step value (MADD-ANO-027).
        """

        # Run first iteration
        state_after_first = one_pass(new_state_inner)

        # The group's non-floating fields (counters, flags, keys), and the
        # rule both solvers return them by.  The fixed-point iteration
        # acts on floating fields only, so what a solve returns is a
        # floating state; the non-floating fields beside it are the ones
        # the pass computes *at* that state -- as at a fixed point, where
        # the pass that produced the state and the pass it produces are
        # the same.  ``solver="ift"`` used to restore them from the first
        # pass, on the premise that such a field depends on the pre-step
        # state alone; a flag that reads a boundary input (a contact flag
        # reading a gap) does not, and the returned flag described an
        # input the solve had long left.  ``solver="fori"`` took them from
        # the raw pass that preceded the returned iterate, which under an
        # acceleration is not the pass of the returned floating state
        # either.  Static: an all-floating group never builds the pass.
        nonfloat_fields = {
            nn: tuple(f for f in sorted(state_after_first[nn])
                      if not jnp.issubdtype(
                          jnp.asarray(state_after_first[nn][f]).dtype, jnp.floating))
            for nn in group_node_names
        }

        def _with_nonfloat_fields_at(s_full):
            """*s_full* with the group's non-floating fields recomputed at it.

            One more evaluation of the pass, of which only those fields
            are kept, so XLA keeps only the arithmetic that feeds them (a
            counter's increment; a flag's boundary resolution).  The
            floating inputs go in through ``stop_gradient``: nothing kept
            from this pass has a derivative, and the floating state's --
            the IFT rule's, or the unrolled loop's -- is untouched.
            """
            if not any(nonfloat_fields.values()):
                return s_full
            frozen = jax.tree.map(
                lambda v: (jax.lax.stop_gradient(v)
                           if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating) else v),
                s_full,
            )
            s_leaves = one_pass(frozen)
            out = dict(s_full)
            for nn in group_node_names:
                if nonfloat_fields[nn]:
                    out[nn] = {**s_full[nn],
                               **{f: s_leaves[nn][f] for f in nonfloat_fields[nn]}}
            return out

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
            single_r, single_amp = _non_finite_reads_as_diverged(
                _group_state_finite(state_after_first, group_node_names),
                single_r, single_amp,
            )
            sub = {nn: state_after_first[nn] for nn in group_node_names}
            if group.strict_convergence and group.solver == "ift":
                # Which message, decided from the state; see
                # ``_strict_check`` and the ift branch below.
                single_est = estimated_error(single_r, single_amp, step_scale)
                sub = _strict_check(sub, single_est, _state_measurable(
                    state_after_first, group_node_names, _compute_residual))
            r = {k: v for k, v in new_state_inner.items()}
            for nn in group_node_names:
                r[nn] = sub[nn]
            # One coupling pass ran, so report one: a residual was
            # measured, and ``iterations=0`` beside a non-zero residual
            # would contradict itself.  ``solver="fori"`` keeps its own
            # ``diagnostics=True`` gate.
            # No fixed point was solved for, so there is no Jacobian
            # to take a spectrum of: the spectral triple and the
            # gradient's bound are NaN, which ``coupling_diagnostics``
            # reports as ``spectral_usable=False`` and
            # ``gradient_bound_usable=False``.
            if group.solver == "ift" or group.diagnostics:
                nan = jnp.full((), jnp.nan, jnp.asarray(single_r).dtype)
                return r, (jnp.array(1.0), single_r, single_amp, nan, nan, nan, nan, nan, nan), None
            return r, None, None

        # Determine n_dof for acceleration
        if use_acceleration:
            # From the floating fields when ``accel_fields`` is ``None`` (the
            # IFT path under ``"aitken"`` / ``"fixed"``, which relaxes its own
            # floating vector and reads this only for the carries' dtype):
            # flattening every field concatenated a typed PRNG key with the
            # floats and raised ``ValueError: dtype=key<fry> is not a valid
            # dtype for JAX type promotion`` at the first step (MADD-ANO-158).
            # An all-floating group flattens exactly as before.
            n_dof_flat = (_flatten(state_after_first) if accel_fields is not None
                          else flatten_coupled_state(
                              state_after_first, group_node_names,
                              fields=float_fields_of(state_after_first, group_node_names)))
            n_dof = n_dof_flat.shape[0]
            # The accelerator carries (Aitken's omega and previous
            # residual, IQN's secant matrices) are seeded in the interface
            # vector's dtype, widened to at least float32 as the coupling
            # diagnostics are (``_analysis_dtype``).  They were seeded at
            # canonical precision, which is float64 under
            # ``jax_enable_x64``: a float32 group then raised a fori_loop
            # carry ``TypeError`` under ``solver="fori"`` with
            # ``"aitken"``, ``"iqn-ils"`` or ``"iqn-imvj"``, and under
            # ``"ift"`` narrowed a float64 IQN step back into the float32
            # iterate through a scatter JAX warns will become an error.
            # Unchanged wherever x64 is off, and for a float64 group.
            acc_dtype = _analysis_dtype(n_dof_flat.dtype)
            # The accelerators' frame, fixed for the solve: the same
            # power-of-two rescaling the ift path applies
            # (``_fixed_point_while``), so a group in small units forms its
            # relaxed step from normal numbers and the two solvers agree.
            accel_frame = pow2_frame(n_dof_flat)

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
                return (jnp.zeros((n_dof, max_cols), acc_dtype),
                        jnp.zeros((n_dof, max_cols), acc_dtype), 0)
            group_key = "+".join(sorted(group.nodes))
            meta = new_state_inner.get(_META_KEY, {})
            # ``compile()`` seeds the slots in the same dtype; a state whose
            # dtype moved since (a node promoting its state on the first
            # step) is read in the one the step now iterates in.
            stored_V = jnp.asarray(meta.get(
                f"coupling_{group_key}_V", jnp.zeros((n_dof, max_cols), acc_dtype),
            ), acc_dtype)
            stored_W = jnp.asarray(meta.get(
                f"coupling_{group_key}_W", jnp.zeros((n_dof, max_cols), acc_dtype),
            ), acc_dtype)
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
            that vector.  ``diag`` is ``(n_iters, final_res, final_amp,
            rho_spectral, arnoldi_residual, amplification,
            gradient_bound)`` -- the last four NaN
            unless ``diagnostics=True`` -- and ``vw`` the IQN ``(V, W)``
            matrices (``None`` for other accelerations).  The IFT derivative is intrinsic to ``F``
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
            # Only floating fields live in the fixed-point vector: keeping
            # an integer / boolean field (a counter, a flag) out avoids
            # float<->int casts in the loop and float0 tangents in the
            # IFT rule (which leaked tracers under reverse mode through
            # a scan).  Inside the loop such a field keeps the value the
            # first pass gave it.  That is its returned value only when
            # the node computes it from the pre-step state alone (a step
            # counter); a flag that reads a boundary input (a contact
            # flag reading a gap) depends on the iterate, and the
            # first-pass value described an input the solve had long
            # left.  So the returned state's non-floating fields are
            # recomputed below, from the returned floating fields, by one
            # more evaluation of the pass (``_with_nonfloat_fields_at``).
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

            # Non-floating fields a group-internal edge carries (a flag
            # one member computes and another reads).  ``_embed`` holds
            # every non-floating field at its first-pass value, and a
            # reader that takes such a field from the iterate -- under
            # Jacobi, or scheduled before its producer -- then iterated a
            # map with the field frozen: on a pair whose flag flips during
            # the solve, ``"ift"`` converged to the fixed point of the
            # frozen map with ``converged=True``, where ``"fori"``, which
            # carries the whole state, found the consistent one.  So such
            # fields are evaluated at the iterate itself: one more pass,
            # of which only those fields are kept (XLA keeps only what
            # feeds them).  At a fixed point that is the value the pass
            # gives there, so the fixed point is the consistent one.
            # Static: without such an edge the map is the one it always
            # was.
            live_nonfloat = sorted(
                (nn, fld) for nn, fld in plan.source_fields()
                if fld in template_state.get(nn, {})
                and fld not in float_fields.get(nn, ())
            )

            def _embed_live(x_full):
                s = _embed(x_full)
                if not live_nonfloat:
                    return s
                s_nf = one_pass(s)
                for nn, fld in live_nonfloat:
                    s[nn] = {**s[nn], fld: s_nf[nn][fld]}
                return s

            def _step_flat(x_full):
                s = _embed_live(x_full)
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
            if probe_sink is not None and group.diagnostics:
                # Which constants the gradient bound probes as a whole, by
                # name where a constant is a parameter, a state field or an
                # external input the closure read directly (``_probe_plan``).
                probe_sink["+".join(sorted(group.nodes))] = _whole_probe_names(
                    consts, _node_params, full_state, new_state, external_inputs)

            def _measured_pass_evaluations(x_full):
                """The pass's evaluation count with every read weighted by its measured gain.

                Each member's own count ``d_n e_n`` is scaled by
                ``max(1, sum_m g_nm)`` over the members it reads -- the
                amplification its own rounding suffers where its terms
                cancel (``3u - 2v`` at ``u ~ v`` reads 5) -- and under
                Gauss-Seidel the count is ``max_n depth(n)``, ``depth(n) =
                own'(n) + sum_m g_nm depth(m)`` over the members ``n``
                reads from the same pass; under Jacobi, ``max_n own'(n)``.
                ``g_nm`` is the measured relative gain of the read
                (``_read_gain``).  Never below the structural count.  In
                its own ``lax.cond`` branch so its products cannot share
                subexpressions with the forward.
                """
                def measure(xx):
                    s_star = _embed_live(xx)
                    gains = {(mm, nn): _read_gain(s_star, mm, nn)
                             for nn in gs_order for mm in sorted(gs_reads[nn])}
                    own_w = {}
                    for nn in gs_order:
                        total = functools.reduce(
                            jnp.add, [gains[(mm, nn)] for mm in sorted(gs_reads[nn])],
                            jnp.float32(0.0))
                        own_w[nn] = jnp.float32(gs_own[nn]) * jnp.maximum(
                            jnp.float32(1.0), total)
                    if use_jacobi:
                        weighted = functools.reduce(jnp.maximum, own_w.values())
                    else:
                        depth = {}
                        for nn in gs_order:
                            acc = own_w[nn]
                            for mm in sorted(gs_same_pass[nn]):
                                acc = acc + gains[(mm, nn)] * depth[mm]
                            depth[nn] = acc
                        weighted = functools.reduce(jnp.maximum, depth.values())
                    return jnp.maximum(jnp.float32(pass_evaluations), weighted)

                if not any(gs_reads.values()):
                    return jnp.asarray(pass_evaluations, jnp.float32)
                return jax.lax.cond(
                    jnp.all(jnp.isfinite(x_full)), measure,
                    lambda _xx: jnp.asarray(pass_evaluations, jnp.float32), x_full)

            def _read_fields(s_star):
                """``(node, field, value)`` for every field the norm reads."""
                read = plan.source_fields()
                for nn in group_node_names:
                    for fld in float_fields[nn]:
                        if use_interface_norm and (nn, fld) not in read:
                            continue
                        yield nn, fld, jnp.asarray(s_star[nn][fld])

            def _weight_scale(x_full):
                """A common power-of-two factor for the weights: 1.0 in range.

                A weight is ``1 / max|field|``, and above ``1 / tiny``
                (about ``8.5e37`` in float32 -- which the mixed and
                interface norms still evaluate, their scale being
                ``rtol * max|field|``) that reciprocal is subnormal and
                XLA's CPU backend flushes it to zero.  A zero weight cuts
                the field out of the spectrum -- the dead-band failure at
                the other end of the range -- and ``rho_spectral`` read
                0.0 for a map contracting at 0.9, with the bound 0.12x the
                true distance and ``spectral_usable=True``.  Every weight
                is therefore multiplied by 16 when any read field is up
                there: ``max|field| < 4 / tiny`` in every IEEE format, so
                ``16 / max|field|`` is at least four times ``tiny``.  A
                common factor changes nothing the weights feed -- the
                Arnoldi runs on a similarity, and the floors the helpers
                compare against are scaled by the same factor -- and in
                range it is exactly 1.0, so nothing moves by a bit.
                """
                top = jnp.array(False)
                for _nn, _fld, val in _read_fields(_embed(x_full)):
                    ref = _field_reference(val, val)
                    top = jnp.logical_or(top, ref * jnp.finfo(val.dtype).tiny > 1.0)
                return jnp.where(top, 16.0, 1.0).astype(x_full.dtype)

            def _field_magnitudes(x_full):
                """At each entry of the flat state, its field's ``max|v|``."""
                s_star = _embed(x_full)
                m = {nn: {fld: jnp.broadcast_to(
                    jnp.max(jnp.abs(jnp.asarray(s_star[nn][fld]))),
                    jnp.shape(s_star[nn][fld])).astype(jnp.asarray(s_star[nn][fld]).dtype)
                    for fld in float_fields[nn]}
                    for nn in group_node_names}
                return _flatten_full({**s_star, **m})

            def _norm_weights(x_full, zero_field_weight=None, scale: Any = 1.0):
                """Per-entry factors of the group's norm at ``x_full``.

                Mirrors ``_scaled_change``: a field the norm reads is
                divided by its own magnitude (``max|field|``) and is
                read only above the dead band; under the interface norm
                only edge-source fields are read.  A constant factor
                (``rtol``, the mixed norm's ``1/count``) is left out --
                the resolvent norm the weights feed is invariant to it.
                With ``zero_field_weight`` given, the weights are the
                *spectrum's* instead (see ``_spectral_rate_at``): a read
                field inside the dead band keeps its own magnitude's
                weight rather than zero -- the dead band takes a field
                out of the residual, not out of the coupling loop -- and
                a read field whose magnitude is exactly zero (or below
                the dtype's normal range) gets ``zero_field_weight``.
                Every weight is multiplied by ``scale``
                (:func:`_weight_scale`) before it is rounded, so a
                reciprocal that would be subnormal never is.
                """
                s_star = _embed(x_full)
                w = {nn: {fld: jnp.zeros_like(jnp.asarray(s_star[nn][fld]))
                          for fld in float_fields[nn]}
                     for nn in group_node_names}
                for nn, fld, val in _read_fields(s_star):
                    ref = _field_reference(val, val)
                    k = jnp.asarray(scale, val.dtype)
                    if zero_field_weight is None:
                        active = jnp.logical_and(ref > group.atol, ref > 0)
                        inv = jnp.where(active, k / jnp.where(active, ref, 1.0), 0.0)
                    else:
                        scaled = ref >= jnp.finfo(val.dtype).tiny
                        inv = jnp.where(scaled, k / jnp.where(scaled, ref, 1.0),
                                        zero_field_weight * k)
                    w[nn][fld] = jnp.broadcast_to(inv, val.shape).astype(val.dtype)
                return _flatten_full({**s_star, **w})

            # Under ``convergence_norm="interface"`` the norm reads what
            # each internal edge *delivers* -- its source value through the
            # mapping, then the transform -- once per edge.  With a mapping
            # or a transform on an internal edge, or a field that more than
            # one internal edge reads, that is not the read fields weighted
            # once each, and the report's spectral analysis is taken on the
            # reading (``_interface_spectral_rate_at``); the raw source
            # fields' weights above measured a different norm.  Static
            # (``_reading_is_the_fields``): every other group keeps the
            # analysis it had.
            transformed_reading = use_interface_norm and not _reading_is_the_fields(
                plan, float_fields)
            def _reading_parts(s_star):
                """The interface norm's reading at ``s_star``, as ``(source dtype,
                value)`` per edge: what each internal edge delivers, in the order
                and by the rules ``coupling_residual_interface`` sums them
                (``_interface_readings``, which both iterate)."""
                return [(source_dtype, jnp.asarray(v))
                        for _e, source_dtype, v in _interface_readings(
                            plan, s_star, mappings=report_mappings)]

            def _reading_values(s_star):
                """The delivered values alone."""
                return [v for _source_dtype, v in _reading_parts(s_star)]

            def _reading(x_full):
                """``Phi(x)``: the reading as one flat vector."""
                vals = _reading_values(_embed(x_full))
                work = _analysis_dtype(jnp.result_type(*[v.dtype for v in vals]))
                return jnp.concatenate([jnp.ravel(v).astype(work) for v in vals])

            def _reading_reference(x_full):
                """At each entry of the reading, its edge's ``max|Phi_e(x)|``."""
                vals = _reading_values(_embed(x_full))
                work = _analysis_dtype(jnp.result_type(*[v.dtype for v in vals]))
                return jnp.concatenate([
                    jnp.broadcast_to(_field_reference(v, v), (v.size,)).astype(work)
                    for v in vals])

            def _reading_weights(x_full, zero_field_weight=None):
                """``(weights, scale)`` of the reading at ``x_full``, entry by entry.

                ``_norm_weights`` and ``_weight_scale`` on the reading: each
                edge's value over its own magnitude, read only above the dead
                band (or, with ``zero_field_weight``, the spectrum's weights,
                a dead-banded edge keeping its own magnitude's), times a
                common power of two that keeps every reciprocal normal.
                """
                vals = _reading_values(_embed(x_full))
                work = _analysis_dtype(jnp.result_type(*[v.dtype for v in vals]))
                top = jnp.array(False)
                for v in vals:
                    top = jnp.logical_or(
                        top, _field_reference(v, v) * jnp.finfo(v.dtype).tiny > 1.0)
                parts = []
                for v in vals:
                    ref = _field_reference(v, v)
                    k = jnp.where(top, 16.0, 1.0).astype(v.dtype)
                    if zero_field_weight is None:
                        active = jnp.logical_and(ref > group.atol, ref > 0)
                        inv = jnp.where(active, k / jnp.where(active, ref, 1.0), 0.0)
                    else:
                        scaled = ref >= jnp.finfo(v.dtype).tiny
                        inv = jnp.where(scaled, k / jnp.where(scaled, ref, 1.0),
                                        zero_field_weight * k)
                    parts.append(jnp.broadcast_to(inv, (v.size,)).astype(v.dtype).astype(work))
                return jnp.concatenate(parts), jnp.where(top, 16.0, 1.0).astype(work)

            def _reading_resolution(x_full, scale, evaluations):
                """The reading's float resolution per entry, each edge's value at
                its own eps -- its dtype's, or its source field's where that is
                coarser (``_reading_eps``, as ``residual_precision_floor`` takes
                it) -- in the reading's weights' units (``_residual_resolution``),
                for a pass that rounds like ``evaluations``."""
                parts = _reading_parts(_embed(x_full))
                work = _analysis_dtype(jnp.result_type(*[v.dtype for _src, v in parts]))
                eps = jnp.concatenate([
                    jnp.full((v.size,), _reading_eps(source_dtype, v), work)
                    for source_dtype, v in parts])
                return (scale * evaluations.astype(work)) * _residual_resolution(eps)

            def _geometry_gap_at(x_sg, check_weights, resolution):
                """The self-check of the pass's product along the geometry
                (``_geometry_product_gap``), at the returned state.

                Two directions at most.  One moves, in the iterate, every
                geometry field a member holds: what a source-anchored
                edge inside the group reads, and what a flux hook's
                target-anchored edge reads.  The other moves the same
                fields where they are constants of the pass: a member's
                pre-step state (a target-anchored edge of ``update``) and
                the state of a node outside the group (a source-anchored
                edge into it).  Each step is the mapping's own
                (``_probe_step``).  A geometry the pass must read from a
                constant that the closure conversion did not hoist (a
                step traced on concrete values) cannot be moved: the gap
                is NaN, which the report reads as a failed check.
                """
                n = int(x0_full.shape[0])
                positions = unflatten_coupled_state(
                    np.arange(n, dtype=np.int32), template_img, group_node_names,
                    fields=float_fields)
                field_ids = np.zeros((n,), np.int32)
                n_fields = 0
                for nn in group_node_names:
                    for fld in float_fields[nn]:
                        field_ids[np.asarray(positions[nn][fld], np.int64).ravel()] = n_fields
                        n_fields += 1
                dx = jnp.zeros_like(x_sg)
                moved_in_iterate = False
                dconsts: list = [None] * len(consts)
                located = True
                for holder, fld, mapping in geometry_checked:
                    if holder in group_node_set:
                        if fld in float_fields[holder]:
                            at = np.asarray(positions[holder][fld], np.int64)
                            where = jnp.asarray(np.ravel(at), jnp.int32)
                            step = mapping._probe_step(x_sg[where].reshape(at.shape))
                            dx = dx.at[where].set(jnp.ravel(step).astype(dx.dtype))
                            moved_in_iterate = True
                        held = [initial_node_states[holder].get(fld)]
                        # A member's pre-step positions are a constant
                        # only a target-anchored edge needs.
                        needed = plan.reads_own_geometry(holder, fld)
                    else:
                        held = [tree.get(holder, {}).get(fld)
                                for tree in (full_state, new_state, template_state)]
                        needed = True
                    ids = {id(v) for v in held if v is not None}
                    found = [i for i, c in enumerate(consts) if id(c) in ids]
                    for i in found:
                        dconsts[i] = mapping._probe_step(consts[i])
                    located = located and (bool(found) or not needed)
                directions = []
                if moved_in_iterate:
                    directions.append((dx, [None] * len(consts)))
                if any(t is not None for t in dconsts):
                    directions.append((jnp.zeros_like(x_sg), dconsts))
                gap = _geometry_product_gap(
                    step_pure, x_sg, consts, directions, check_weights, resolution,
                    field_ids, n_fields)
                return gap if located else jnp.full_like(gap, jnp.nan)

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
                first_r,
                conv_threshold_value,
                int(max_iters),
                group.acceleration,
                float(group.relaxation),
                int(n_reuse),
                sub_idx,
                str(group.linear_solver),
            )
            # The verdict on a non-finite state is taken over *every*
            # floating field of the group, not only the ones the norm
            # reads (see ``_non_finite_reads_as_diverged``).
            final_res, final_amp = _non_finite_reads_as_diverged(
                jnp.all(jnp.isfinite(x_star_full)), final_res, final_amp,
            )
            # The spectral bound's ingredients, at the state being
            # returned (``x_star_full`` before the strict guard, whose
            # value it is).  Only with ``diagnostics=True``: it costs
            # ``SPECTRAL_KRYLOV_STEPS + 1`` Jacobian-vector products
            # per group per step, which the always-on ift report is
            # not charged for.  NaN is what ``coupling_diagnostics``
            # reads as "not computed" (``spectral_usable=False``).
            # The bound on the IFT gradient's error is built on that
            # triple, so it has the same gate and the same NaN.
            grad_bound = jnp.full((), jnp.nan, x0_full.dtype)
            pass_evals = jnp.full((), jnp.nan, x0_full.dtype)
            geometry_gap = jnp.full((), jnp.nan, x0_full.dtype)
            if group.diagnostics:
                # How many evaluations' rounding the pass carries, with
                # each same-pass read weighted by its measured relative
                # gain at the returned state (``_gain_weighted_evaluations``),
                # never below the structural count.
                pass_evals = _measured_pass_evaluations(
                    jax.lax.stop_gradient(x_star_full)).astype(x0_full.dtype)
                weight_scale = _weight_scale(jax.lax.stop_gradient(x_star_full))
                weights = _norm_weights(jax.lax.stop_gradient(x_star_full),
                                        scale=weight_scale)
                # A dead-banded field keeps its own magnitude's weight in
                # the spectrum; one that is exactly zero has no magnitude,
                # and gets the caller's atol (the declared unit of "zero")
                # or, with no dead band declared, 1 -- any positive weight
                # keeps it on the loop.
                spec_weights = _norm_weights(
                    jax.lax.stop_gradient(x_star_full),
                    zero_field_weight=(1.0 / float(group.atol)) if group.atol > 0 else 1.0,
                    scale=weight_scale,
                )
                # The residual's float resolution per entry, each field at
                # its own dtype's eps (``_residual_resolution``), in the
                # weights' units (so times their common scale), for a pass
                # that rounds like ``pass_evals`` single ones.
                resolution = (weight_scale * pass_evals) * _residual_resolution(_flatten_full({
                    nn: {fld: jnp.full(
                        jnp.shape(template_state[nn][fld]),
                        jnp.finfo(template_state[nn][fld].dtype).eps,
                        template_state[nn][fld].dtype)
                        for fld in float_fields[nn]}
                    for nn in group_node_names
                }))
                # The rounding of the map's Jacobian-vector products: ``eps``
                # of the coarsest field the pass evaluates in (static).
                map_eps = max(
                    float(jnp.finfo(template_state[nn][fld].dtype).eps)
                    for nn in group_node_names for fld in float_fields[nn]
                ) if any(float_fields[nn] for nn in group_node_names) else None
                if transformed_reading:
                    # The gradient bound's triple, in the state's weights,
                    # which its own norms are taken in.
                    rho_spec, spec_resid, spec_amp = _spectral_rate_at(
                        step_pure, x_star_full, consts, weights, spec_weights,
                        resolution=resolution, map_eps=map_eps,
                    )
                else:
                    rho_spec, spec_resid, spec_amp, pair_ratio = _spectral_rate_at(
                        step_pure, x_star_full, consts, weights, spec_weights,
                        resolution=resolution, field_reference=_field_magnitudes,
                        map_eps=map_eps,
                    )
                # Its distance is the spectral bound (never below the Newton
                # step); the resolvent is applied exactly to each probe's
                # secant, a second difference of the adjoint's own matvec.
                # ``11 + 4 k + 5 n_p + 2 k n_p`` more JVPs, ``n_p`` the probes (see
                # ``_gradient_error_bound_at``).
                grad_bound = _gradient_error_bound_at(
                    step_pure, x_star_full, consts, weights,
                    rho_spec, spec_resid, spec_amp, resolution=resolution,
                )
                if transformed_reading:
                    # The report reads ``gradient_bound_usable`` off the
                    # reading's spectrum below; the gradient bound rests on
                    # the state's, so it stands only where that one settled.
                    grad_bound = jnp.where(
                        spectral_rate_settled(rho_spec, spec_resid), grad_bound,
                        jnp.full_like(grad_bound, jnp.nan))
                    # The report's triple, on the interface norm's own
                    # reading: each internal edge's delivered value over its
                    # own magnitude -- the coordinates ``residual`` and the
                    # floor are measured in (``_interface_spectral_rate_at``).
                    x_sg = jax.lax.stop_gradient(x_star_full)
                    read_w, read_scale = _reading_weights(x_sg)
                    read_spec_w, _ = _reading_weights(
                        x_sg,
                        zero_field_weight=(1.0 / float(group.atol)) if group.atol > 0 else 1.0)
                    rho_spec, spec_resid, spec_amp, pair_ratio = _interface_spectral_rate_at(
                        step_pure, x_star_full, consts, spec_weights, _reading,
                        read_w, read_spec_w,
                        _reading_resolution(x_sg, read_scale, pass_evals),
                        _reading_reference, map_eps=map_eps,
                    )
                if geometry_checked:
                    geometry_gap = _geometry_gap_at(
                        jax.lax.stop_gradient(x_star_full), spec_weights, resolution)
                # The factor the report's bound applies, in the returned
                # state's weights: the gradient bound above takes its own
                # residual in those weights already, so only the stored
                # factor carries the pair-to-returned ratio.
                spec_amp = spec_amp * pair_ratio.astype(spec_amp.dtype)
            else:
                rho_spec = jnp.full((), jnp.nan, x0_full.dtype)
                spec_resid = spec_amp = rho_spec
            if group.strict_convergence:
                # A non-finite *state* is the diverged iteration: since
                # 0.4.0 every norm reports ``inf`` for a field it cannot
                # evaluate -- a NaN or inf entry, or a magnitude beyond
                # what its dtype can measure a change at -- instead of
                # dropping it from the dead band (MADD-ANO-019), and no
                # larger cap would help: ``_strict_check`` names it first.
                # Decided from the state (``_state_measurable``), not the
                # estimate, which a norm can also overflow on a finite one.
                final_est = estimated_error(final_res, final_amp, step_scale)
                x_star_full = _strict_check(x_star_full, final_est, _state_measurable(
                    _embed(x_star_full), group_node_names, _compute_residual))
            # ``_embed`` restores the non-floating fields from the first
            # pass; they are recomputed at the returned floating state
            # (``_with_nonfloat_fields_at``, the rule both solvers share).
            final = _merge(template_state,
                           _with_nonfloat_fields_at(_embed_live(x_star_full)),
                           jnp.array(False))
            return (final, (n_iters, final_res, final_amp, rho_spec, spec_resid,
                            spec_amp, grad_bound, pass_evals, geometry_gap),
                    (vw if vw else None))

        if group.solver == "ift":
            (final_state, (iter_count, final_res, final_amp, rho_spec,
                           spec_resid, spec_amp, grad_bound, pass_evals, geometry_gap),
             vw) = _run_ift_forward(state_after_first)
            r = {k: v for k, v in new_state_inner.items()}
            for nn in group_node_names:
                r[nn] = final_state[nn]
            # Always reported (not only with ``diagnostics=True``): the
            # scalars are already in the carry, and the converged flag
            # is what tells a training loop that the IFT gradient
            # through this step is trustworthy.  The spectral pair is
            # NaN unless ``diagnostics=True`` (see ``_run_ift_forward``).
            diag_data = (iter_count, final_res, final_amp, rho_spec,
                         spec_resid, spec_amp, grad_bound, pass_evals, geometry_gap)
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
                def body_fn(i: Any, carry: tuple) -> tuple:
                    (s_cur, converged, prev_below, prev_res, prev_res2,
                     icount, fres, fprev, omega, prev_r) = carry
                    s_raw = one_pass(s_cur)
                    residual = _compute_residual(s_raw, s_cur)
                    below = residual <= conv_threshold
                    est, _amp = _estimate(residual, prev_res, prev_res2)
                    new_converged = converged | (
                        (est <= conv_threshold) & prev_below
                    )
                    x_old = _flatten(s_cur) * accel_frame
                    x_raw = _flatten(s_raw) * accel_frame
                    x_rel, new_omega, cur_r = aitken_relaxation(
                        x_old, x_raw, prev_r, omega
                    )
                    # Out of the frame, and the carries in their own dtypes:
                    # a 16-bit group's residual is 16-bit while its carry is
                    # float32 (``acc_dtype``), which was a carry TypeError.
                    x_rel = x_rel / accel_frame
                    new_omega = new_omega.astype(omega.dtype)
                    cur_r = cur_r.astype(prev_r.dtype)
                    s_partial = _unflatten(x_rel, s_cur)
                    s_accel = _build_accel_state(s_raw, s_partial)
                    s_merged = _merge(s_cur, s_accel, new_converged)
                    new_count = icount + jnp.where(new_converged, 0.0, 1.0)
                    new_res = jnp.where(converged, fres, residual)
                    new_prev = (jnp.where(converged, fprev[0], prev_res),
                                jnp.where(converged, fprev[1], prev_res2))
                    return (s_merged, new_converged, below, residual,
                            prev_res, new_count, new_res, new_prev,
                            new_omega, cur_r)

                init_carry = (
                    state_after_first, jnp.array(False), first_below,
                    first_r, first_r,
                    jnp.array(1.0), first_r, (first_r, first_r),
                    jnp.array(1.0, acc_dtype), jnp.zeros(n_dof, acc_dtype),
                )
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]
                iter_count, final_res = final_carry[5], final_carry[6]
                frozen_prev, prev_loop_res = final_carry[7], final_carry[4]
            else:
                def body_fn(i: Any, carry: tuple) -> tuple:
                    (s_cur, converged, prev_below, prev_res, prev_res2,
                     omega, prev_r) = carry
                    s_raw = one_pass(s_cur)
                    residual = _compute_residual(s_raw, s_cur)
                    below = residual <= conv_threshold
                    est, _amp = _estimate(residual, prev_res, prev_res2)
                    new_converged = converged | (
                        (est <= conv_threshold) & prev_below
                    )
                    x_old = _flatten(s_cur) * accel_frame
                    x_raw = _flatten(s_raw) * accel_frame
                    x_rel, new_omega, cur_r = aitken_relaxation(
                        x_old, x_raw, prev_r, omega
                    )
                    # Out of the frame, and the carries in their own dtypes:
                    # a 16-bit group's residual is 16-bit while its carry is
                    # float32 (``acc_dtype``), which was a carry TypeError.
                    x_rel = x_rel / accel_frame
                    new_omega = new_omega.astype(omega.dtype)
                    cur_r = cur_r.astype(prev_r.dtype)
                    s_partial = _unflatten(x_rel, s_cur)
                    s_accel = _build_accel_state(s_raw, s_partial)
                    s_merged = _merge(s_cur, s_accel, new_converged)
                    return (s_merged, new_converged, below, residual,
                            prev_res, new_omega, cur_r)

                init_carry = (
                    state_after_first, jnp.array(False), first_below,
                    first_r, first_r,
                    jnp.array(1.0, acc_dtype), jnp.zeros(n_dof, acc_dtype),
                )
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]

        elif group.acceleration in ("iqn-ils", "iqn-imvj"):
            init_V, init_W, n_reuse = _iqn_warm_start()
            # Into the frame (see ``accel_frame``).
            init_V, init_W = init_V * accel_frame, init_W * accel_frame
            init_ncols = jnp.int32(n_reuse)
            init_flat = _flatten(state_after_first) * accel_frame

            # The secant columns IQN-IMVJ carries to the next step are
            # the ones the latching pass left, as under ``"ift"``, whose
            # loop stops there.  This loop runs on, and every pass on a
            # frozen state measures the same residual and the same raw
            # output, so it used to shift a zero column in each time: by
            # the cap the warm-start window was all zeros,
            # ``jacobian_reuse`` did nothing, and the solvers disagreed
            # on ``iterations`` from the second step on.  IQN-ILS keeps
            # nothing across steps, so its program is left as it was.
            freeze_columns = group.acceleration == "iqn-imvj"

            def _secant_live(i, converged):
                if freeze_columns:
                    return jnp.logical_and(i > 1, jnp.logical_not(converged))
                return i > 1

            if track_diag:
                def body_fn(i: Any, carry: tuple) -> tuple:
                    (s_cur, converged, prev_res, prev_res2, icount, fres,
                     fprev, V, W, nc, prev_r, prev_s, omega, prev_ra) = carry
                    s_raw = one_pass(s_cur)
                    residual = _compute_residual(s_raw, s_cur)
                    est, _amp = _estimate(residual, prev_res, prev_res2)
                    new_converged = converged | (est <= conv_threshold)
                    x_old = _flatten(s_cur) * accel_frame
                    x_raw = _flatten(s_raw) * accel_frame
                    (x_new, nV, nW, nnc,
                     cur_r, cur_s, n_omega, cur_ra) = iqn_ils_update(
                        x_raw, x_old, prev_r, prev_s,
                        V, W, nc, omega, prev_ra,
                        have_prev=_secant_live(i, converged),
                    )
                    # Out of the frame; the carries keep their dtypes.
                    x_new = x_new / accel_frame
                    nV, nW = nV.astype(V.dtype), nW.astype(W.dtype)
                    cur_r, cur_s = cur_r.astype(prev_r.dtype), cur_s.astype(prev_s.dtype)
                    n_omega, cur_ra = n_omega.astype(omega.dtype), cur_ra.astype(prev_ra.dtype)
                    s_partial = _unflatten(x_new, s_cur)
                    s_accel = _build_accel_state(s_raw, s_partial)
                    s_merged = _merge(s_cur, s_accel, new_converged)
                    new_count = icount + jnp.where(new_converged, 0.0, 1.0)
                    new_res = jnp.where(converged, fres, residual)
                    new_prev = (jnp.where(converged, fprev[0], prev_res),
                                jnp.where(converged, fprev[1], prev_res2))
                    return (s_merged, new_converged, residual, prev_res,
                            new_count, new_res, new_prev,
                            nV, nW, nnc, cur_r, cur_s, n_omega, cur_ra)

                init_carry = (
                    state_after_first, jnp.array(False), first_r, first_r,
                    jnp.array(1.0), first_r, (first_r, first_r),
                    init_V, init_W, init_ncols,
                    jnp.zeros(n_dof, acc_dtype), init_flat,
                    jnp.array(1.0, acc_dtype), jnp.zeros(n_dof, acc_dtype),
                )
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]
                iter_count, final_res = final_carry[4], final_carry[5]
                frozen_prev, prev_loop_res = final_carry[6], final_carry[3]
                final_V, final_W = final_carry[7] / accel_frame, final_carry[8] / accel_frame
            else:
                def body_fn(i: Any, carry: tuple) -> tuple:
                    (s_cur, converged, prev_res, prev_res2,
                     V, W, nc, prev_r, prev_s, omega, prev_ra) = carry
                    s_raw = one_pass(s_cur)
                    residual = _compute_residual(s_raw, s_cur)
                    est, _amp = _estimate(residual, prev_res, prev_res2)
                    new_converged = converged | (est <= conv_threshold)
                    x_old = _flatten(s_cur) * accel_frame
                    x_raw = _flatten(s_raw) * accel_frame
                    (x_new, nV, nW, nnc,
                     cur_r, cur_s, n_omega, cur_ra) = iqn_ils_update(
                        x_raw, x_old, prev_r, prev_s,
                        V, W, nc, omega, prev_ra,
                        have_prev=_secant_live(i, converged),
                    )
                    # Out of the frame; the carries keep their dtypes.
                    x_new = x_new / accel_frame
                    nV, nW = nV.astype(V.dtype), nW.astype(W.dtype)
                    cur_r, cur_s = cur_r.astype(prev_r.dtype), cur_s.astype(prev_s.dtype)
                    n_omega, cur_ra = n_omega.astype(omega.dtype), cur_ra.astype(prev_ra.dtype)
                    s_partial = _unflatten(x_new, s_cur)
                    s_accel = _build_accel_state(s_raw, s_partial)
                    s_merged = _merge(s_cur, s_accel, new_converged)
                    return (s_merged, new_converged, residual, prev_res,
                            nV, nW, nnc, cur_r, cur_s, n_omega, cur_ra)

                init_carry = (
                    state_after_first, jnp.array(False), first_r, first_r,
                    init_V, init_W, init_ncols,
                    jnp.zeros(n_dof, acc_dtype), init_flat,
                    jnp.array(1.0, acc_dtype), jnp.zeros(n_dof, acc_dtype),
                )
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]
                final_V, final_W = final_carry[4] / accel_frame, final_carry[5] / accel_frame

        elif group.acceleration == "fixed":
            omega_val = group.relaxation

            if track_diag:
                def body_fn(i: Any, carry: tuple) -> tuple:
                    (s_cur, converged, prev_res, prev_res2, icount, fres,
                     fprev) = carry
                    s_raw = one_pass(s_cur)
                    residual = _compute_residual(s_raw, s_cur)
                    est, _amp = _estimate(residual, prev_res, prev_res2,
                                          relax_first and i == 1)
                    new_converged = converged | (est <= conv_threshold)
                    x_old = _flatten(s_cur) * accel_frame
                    x_raw = _flatten(s_raw) * accel_frame
                    x_rel = fixed_relaxation(x_old, x_raw, omega_val) / accel_frame
                    s_partial = _unflatten(x_rel, s_cur)
                    s_accel = _build_accel_state(s_raw, s_partial)
                    s_merged = _merge(s_cur, s_accel, new_converged)
                    new_count = icount + jnp.where(new_converged, 0.0, 1.0)
                    new_res = jnp.where(converged, fres, residual)
                    new_prev = (jnp.where(converged, fprev[0], prev_res),
                                jnp.where(converged, fprev[1], prev_res2))
                    return (s_merged, new_converged, residual, prev_res,
                            new_count, new_res, new_prev)

                init_carry = (
                    state_after_first, jnp.array(False), first_r, first_r,
                    jnp.array(1.0), first_r, (first_r, first_r),
                )
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]
                iter_count, final_res = final_carry[4], final_carry[5]
                frozen_prev, prev_loop_res = final_carry[6], final_carry[3]
            else:
                def body_fn(i: Any, carry: tuple) -> tuple:
                    s_cur, converged, prev_res, prev_res2 = carry
                    s_raw = one_pass(s_cur)
                    residual = _compute_residual(s_raw, s_cur)
                    est, _amp = _estimate(residual, prev_res, prev_res2,
                                          relax_first and i == 1)
                    new_converged = converged | (est <= conv_threshold)
                    x_old = _flatten(s_cur) * accel_frame
                    x_raw = _flatten(s_raw) * accel_frame
                    x_rel = fixed_relaxation(x_old, x_raw, omega_val) / accel_frame
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
                def body_fn(i: Any, carry: tuple) -> tuple:
                    (s_cur, converged, prev_res, prev_res2, icount, fres,
                     fprev) = carry
                    s_new = one_pass(s_cur)
                    residual = _compute_residual(s_new, s_cur)
                    est, _amp = _estimate(residual, prev_res, prev_res2)
                    new_converged = converged | (est <= conv_threshold)
                    s_merged = _merge(s_cur, s_new, new_converged)
                    new_count = icount + jnp.where(new_converged, 0.0, 1.0)
                    new_res = jnp.where(converged, fres, residual)
                    new_prev = (jnp.where(converged, fprev[0], prev_res),
                                jnp.where(converged, fprev[1], prev_res2))
                    return (s_merged, new_converged, residual, prev_res,
                            new_count, new_res, new_prev)

                init_carry = (
                    state_after_first, jnp.array(False), first_r, first_r,
                    jnp.array(1.0), first_r, (first_r, first_r),
                )
                final_carry = jax.lax.fori_loop(
                    1, max_iters, body_fn, init_carry
                )
                final_state = final_carry[0]
                iter_count, final_res = final_carry[4], final_carry[5]
                frozen_prev, prev_loop_res = final_carry[6], final_carry[3]
            else:
                def body_fn(i: Any, carry: tuple) -> tuple:
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

            def _at_latch(_s):
                # The amplification the criterion used on the latching
                # pass, from the residual triple the loop froze there.
                # Computed here, after the loop and in a branch of its own,
                # and never inside the loop body: an amplification carried
                # through the loop was a second computation of the
                # criterion's own arithmetic, XLA rewrote the criterion
                # around it, ``est`` moved by an ulp near the float floor,
                # the latch fell on a different pass, and
                # ``diagnostics=True`` returned a state one ulp away from
                # ``diagnostics=False`` (chain-5, 4 of 36 configurations on
                # jaxlib 0.11.0; an ``optimization_barrier`` around it held
                # on 0.11.0 and not on 0.11.2).  The loop now carries only
                # *selects* of residuals it already computes, which leave
                # the forward bit-identical on 0.10.2, 0.11.0 and 0.11.2.
                # A latch on the first loop pass froze that pass's pair,
                # whose predecessor is the unrelaxed pre-loop pass
                # (``first_pass_relaxed_amplification``): ``iter_count``
                # stays 1 exactly then.
                return loop_res, first_pass_relaxed_amplification(
                    error_amplification(loop_res, frozen_prev[0], frozen_prev[1]),
                    group.acceleration, group.relaxation,
                    relax_first and iter_count == 1.0)

            final_res, final_amp = jax.lax.cond(
                final_carry[1],
                _at_latch,
                _measure_at_cap,
                final_state,
            )
            final_res, final_amp = _non_finite_reads_as_diverged(
                _group_state_finite(final_state, group_node_names),
                final_res, final_amp,
            )

        # The non-floating fields of the returned state, by the rule the
        # ift path applies (``_with_nonfloat_fields_at``): the raw pass
        # they came from preceded the returned iterate, which under an
        # acceleration is not that pass's output.
        final_state = _with_nonfloat_fields_at(final_state)

        # Merge coupled nodes back into the full state
        r = {k: v for k, v in new_state_inner.items()}
        for nn in group_node_names:
            r[nn] = final_state[nn]

        # Write diagnostics to _meta if requested.  The fori path has
        # no linearisation of ``F`` to take a spectrum of, so the
        # spectral pair is NaN here by construction, never a number.
        diag_data = None
        if track_diag:
            nan = jnp.full((), jnp.nan, jnp.asarray(final_res).dtype)
            diag_data = (iter_count, final_res, final_amp, nan, nan, nan, nan, nan, nan)

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
            # floating fields are extrapolated; a counter or flag is
            # recomputed by the first pass like any other field.
            has_history = pred_count >= 2
            pred_fields = float_fields_of(new_state, group_node_names)
            x_cur = flatten_coupled_state(new_state, group_node_names, fields=pred_fields)
            x_use = jnp.where(has_history, x_pred, x_cur)

            # Unflatten and update new_state with predicted values.
            # ``predicted`` holds the floating fields only, so they are
            # merged over the node's state rather than replacing it: a
            # replacement dropped the group's integer and boolean leaves
            # from the iterate the solve starts from, and the mixed norm,
            # which looked every field of the new iterate up in the old
            # one, raised ``KeyError`` naming the leaf on the first
            # residual.
            predicted = unflatten_coupled_state(
                x_use, new_state, group_node_names, fields=pred_fields,
            )
            new_state = {k: v for k, v in new_state.items()}
            for nn in group_node_names:
                if nn in predicted:
                    new_state[nn] = {**new_state[nn], **predicted[nn]}

    # ------------------------------------------------------------------
    # Run coupling (``waveform_iterations`` sweeps of it)
    # ------------------------------------------------------------------
    current_state = new_state
    diag_data = None
    vw_data = None
    # Every sweep's pass count, when the step runs more than one sweep.
    # Each sweep iterates the same one-pass map from where the previous
    # one stopped (``one_pass`` reads its argument and the pre-step
    # state, nothing a sweep changes), so the last sweep's residual,
    # amplification and spectral keys describe the returned state and
    # are what the report keeps.  Its pass count is not the step's,
    # though: an earlier sweep can exhaust ``max_iterations`` and leave
    # the last one a single pass from the fixed point, which read
    # ``iterations=1`` at a cap.  The counts are only read off the
    # sweeps' outputs and reduced after the loop -- nothing here feeds a
    # sweep -- and with one sweep this list stays empty and the step is
    # the program it always was.
    sweep_counts = []

    for _wf in range(n_waveform):
        current_state, diag_data, vw_data = _run_coupling_inner(current_state)
        if n_waveform > 1 and diag_data is not None:
            sweep_counts.append(jnp.array(diag_data[0], dtype=jnp.int32))

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
        (iter_count, final_res, final_amp, rho_spec, spec_resid, spec_amp,
         grad_bound, pass_evals, geometry_gap) = diag_data
        # Written in the dtype ``compile()`` seeded the slot with, so the
        # scan carry keeps its type whatever the residual was computed
        # in (the seed is the promotion of the group's floating fields,
        # which is what the residual is computed in, so this is a no-op
        # wherever the two already agreed).  A hand-built state with no
        # seed keeps the residual's own dtype.
        seeded = full_state.get(_META_KEY, {}).get(f"coupling_{group_key}_residual")
        res_dtype = (jnp.asarray(seeded).dtype if seeded is not None
                     else jnp.asarray(final_res).dtype)
        iterations = jnp.array(iter_count, dtype=jnp.int32)
        sweep_meta = {}
        if sweep_counts:
            # ``iterations`` is the largest sweep's count, so
            # ``iterations >= max_iterations`` holds exactly when some
            # sweep exhausted its budget -- the documented cap check --
            # and ``total_iterations`` is every pass the step ran.
            iterations = total = sweep_counts[0]
            for count in sweep_counts[1:]:
                iterations = jnp.maximum(iterations, count)
                total = total + count
            sweep_meta[f"coupling_{group_key}_total_iterations"] = total
        floor_meta = {}
        if reads_mapping_weights:
            # The residual's float floor per evaluation, at the state this
            # step returns and with the mapping weights it ran with.  The
            # report takes every other group's floor from the returned
            # state alone (``coupling_diagnostics``); a mapped edge's
            # delivered value also depends on ``params["mappings"]``, which
            # a caller may override for one step and which the graph no
            # longer holds afterwards -- so the step measures it, once,
            # after the solve, by the function the report calls.  Only its
            # dead-band and finiteness tests read the values, so it carries
            # no derivative.  It is the floor of the state the step leaves,
            # as every other group's is: ``run_adaptive*`` keeps the last
            # half step's (``_fold_kept_half_step_reports`` does not fold it).
            floor_meta[f"coupling_{group_key}_reading_floor"] = jnp.asarray(
                residual_precision_floor(
                    {nn: result[nn] for nn in group_node_names}, group_node_names,
                    "interface", group.atol, group.rtol, plan,
                    evaluations=1.0, mappings=report_mappings,
                ), dtype=res_dtype)
        result.setdefault(_META_KEY, {})
        result[_META_KEY] = {
            **result.get(_META_KEY, {}),
            f"coupling_{group_key}_iterations": iterations,
            **sweep_meta,
            f"coupling_{group_key}_residual": jnp.asarray(final_res, dtype=res_dtype),
            f"coupling_{group_key}_amplification": jnp.asarray(
                final_amp, dtype=res_dtype
            ),
            **floor_meta,
        }
        # The spectral triple exists only where it can be computed
        # (``solver="ift"``) and was asked for (``diagnostics=True``);
        # ``compile()`` seeds exactly the same keys under the same
        # condition so the scan carry keeps its structure.
        if group.diagnostics and group.solver == "ift":
            # The analysis outputs are kept in the dtype they were computed
            # in, at least float32 (``_analysis_dtype``): stored in a 16-bit
            # group's own dtype, ``rho_spectral`` read to bfloat16's 2**-8
            # (0.361328125 for a radius of 0.3618774) where it is documented
            # exact to float32, and the bound the host derives from it
            # inherited the rounding (CPL-087).  ``compile()`` seeds them in
            # the same dtype; a hand-built state with no seed takes it too.
            seeded_spec = full_state.get(_META_KEY, {}).get(
                f"coupling_{group_key}_rho_spectral")
            spec_dtype = (jnp.asarray(seeded_spec).dtype if seeded_spec is not None
                          else _analysis_dtype(res_dtype))
            result[_META_KEY] = {
                **result[_META_KEY],
                f"coupling_{group_key}_rho_spectral": jnp.asarray(
                    rho_spec, dtype=spec_dtype
                ),
                f"coupling_{group_key}_spectral_residual": jnp.asarray(
                    spec_resid, dtype=spec_dtype
                ),
                f"coupling_{group_key}_spectral_amplification": jnp.asarray(
                    spec_amp, dtype=spec_dtype
                ),
                f"coupling_{group_key}_gradient_relative_error_bound": jnp.asarray(
                    grad_bound, dtype=spec_dtype
                ),
                f"coupling_{group_key}_pass_evaluations": jnp.asarray(
                    pass_evals, dtype=spec_dtype
                ),
            }
            if geometry_checked:
                # The self-check of the pass's product along the geometry
                # (experimental): only where the report reads one, so the
                # carry of every other group is the one it always was.
                result[_META_KEY][f"coupling_{group_key}_geometry_gap"] = jnp.asarray(
                    geometry_gap, dtype=spec_dtype)

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
