"""Structure of a coupling group: its interface fields, its pass and
sub-cycle schedule, and the refusals raised when a group is declared.

Moved verbatim out of ``maddening.core.graph_manager``.  Private.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import jax.numpy as jnp
import numpy as np

from maddening.core.coupling import _interface_plan
from maddening.core.coupling.acceleration import _has_entries, float_fields_of
from maddening.core.coupling.group import CouplingGroup


def _floating_accel_fields(fields, state, group_nodes) -> Optional[dict]:
    """``fields`` restricted to the floating fields an accelerator may touch.

    An accelerator flattens the fields it acts on into one floating
    vector and writes the relaxed vector back.  An integer, unsigned,
    boolean or PRNG-key leaf that took that round trip came back rounded
    through float32 -- ``0xdeadbeef`` as ``0xdeadbf00``, ``2**24 + 1`` as
    ``2**24`` -- and the rounded value was the one the step kept.  Such a
    leaf is computed by every pass and has no place in a relaxation, which
    only means anything on a continuous quantity; ``solver="ift"`` has
    always iterated on the floating fields only.  The leaf is *not*
    independent of the iterate in general -- a flag a node computes from
    a boundary input (a contact flag reading a gap) changes with it -- so
    both solvers recompute it from the returned floating state after the
    solve (``_with_nonfloat_fields_at`` in ``_run_coupled_block_impl``).

    ``fields`` is ``None`` for "every field of every node in
    ``group_nodes``", or a ``{node: (field, ...)}`` mapping.  Returned
    unchanged -- ``None`` included -- when it names floating fields only,
    so an all-floating group flattens exactly as it always did.  A
    selection with no floating field left falls back to every floating
    field of the group (an explicit ``accelerated_fields`` naming none
    is refused by ``compile()`` before this is reached).
    """
    floats = float_fields_of(state, sorted(group_nodes))
    if fields is None:
        if all(len(floats[nn]) == len(state[nn]) for nn in floats):
            return None
        return {nn: fs for nn, fs in floats.items() if fs}
    kept = {
        nn: tuple(f for f in sorted(fs) if f in floats.get(nn, ()))
        for nn, fs in fields.items()
    }
    if all(len(kept[nn]) == len(tuple(fields[nn])) for nn in kept):
        return fields
    kept = {nn: fs for nn, fs in kept.items() if fs}
    return kept if kept else {nn: fs for nn, fs in floats.items() if fs}


def _group_accel_fields(group, plan, state) -> Optional[dict]:
    """The fields ``group``'s acceleration flattens; ``None`` for all of them.

    The quasi-Newton accelerations read ``accelerated_fields`` (or the
    interface fields the group's internal edges read:
    ``InterfacePlan.iqn_fields`` of the group's *plan*); ``"aitken"`` and
    ``"fixed"`` relax the whole group.  Either way only floating fields
    enter (:func:`_floating_accel_fields`).  Shared by the step builder
    and by ``compile()``'s IQN-IMVJ warm-start seeding, which must agree
    on the vector's length.
    """
    if group.acceleration in ("iqn-ils", "iqn-imvj"):
        chosen = (group.accelerated_fields if group.accelerated_fields is not None
                  else plan.iqn_fields())
    elif group.acceleration in ("aitken", "fixed"):
        chosen = None
    else:
        return None
    return _floating_accel_fields(chosen, state, group.nodes)


def _reads_mapping_weights(group, plan) -> bool:
    """Does *group*'s norm read a value that depends on interface-mapping weights?

    True under ``convergence_norm="interface"`` when an internal edge
    whose source field is floating is read through its mapping: that
    norm reads such an edge as it is delivered
    (:func:`~maddening.core.coupling.acceleration._interface_readings`),
    its source field through weights that live in ``params["mappings"]``
    and may be overridden per step.  An edge the norm reads at its
    source (a mapping onto more entries than its field holds) is the
    stored field, whatever the weights, and does not count.  The
    float floor of such a group's residual therefore cannot be taken
    from the returned state alone, and the step records it
    (``coupling_<key>_reading_floor``).  Static, and shared by
    ``compile()``'s seeding, the step's write and ``reset_state()``, so
    the three agree on which groups own the slot; every other group's
    ``_meta`` and compiled step are what they were.  *plan* is the
    group's description (``InterfacePlan.norm_reads_mapping_weights``).
    """
    return group.convergence_norm == "interface" and plan.norm_reads_mapping_weights()


def _reading_is_the_fields(interface_edges, float_fields) -> bool:
    """Is the interface norm's reading the fields it reads, each of them once?

    ``coupling_residual_interface`` sums over a group's internal *edges*:
    what the norm reads on each one, over its own magnitude.  The state's
    weights (``_norm_weights`` in the step) give each read *field* its own
    magnitude's weight once.  The two are one norm exactly when the
    reading of every internal edge with a floating source field is that
    field as it is -- an edge with no mapping and no transform, or one
    the norm reads at its source (a mapping onto more entries than the
    field holds) -- and no field is read by more than one of them.  Then
    the report's spectral analysis is taken in the
    state's weights (``_spectral_rate_at``); otherwise on the reading
    (``_interface_spectral_rate_at``):

    * a mapping or a transform the reading goes through changes what an
      entry is and which magnitude it is divided by;
    * a field that ``k`` internal edges read is counted ``k`` times by
      the norm and once by the state's weights.  On a star whose hub's
      field every leaf reads, the residual of the edges times the
      resolvent of the fields read 0.26 to 0.73 of the true distance
      with ``spectral_usable=True`` (2 to 16 leaves, Jacobi, float64;
      MADD-ANO-213).

    Static: *interface_edges* are the group's internal edges (its plan,
    or a bare sequence of edges) and *float_fields* the floating fields a
    norm can read, by node -- those with entries: an edge whose source
    field has none delivers nothing and, like one that carries a counter,
    is not read (``acceleration._has_entries``) -- so a group keeps one
    analysis for the life of its compiled step.  Whether an edge's
    reading is its field as it is follows from the side the norm reads
    it on (``InterfaceEdge.reads_source_as_is``).
    """
    read = set()
    for record in _interface_plan.interface_records(interface_edges):
        node, field = record.source
        if field not in float_fields.get(node, ()):
            continue
        if not record.reads_source_as_is:
            return False
        if record.source in read:
            return False
        read.add(record.source)
    return True


def _fields_the_interface_norm_misses(group, interface_edges, schedule, state) -> dict:
    """``{member: (field, ...)}``: the floating fields *group*'s solve returns recomputed at the iterate it accepts.

    ``convergence_norm="interface"`` measures, on each pass, how far what
    the internal edges deliver moved between an iterate and its successor,
    and nothing else.  The members of the iterate it accepts were computed
    from the readings of the iterate *before* it, which the exit does not
    compare with anything.  So the norm answers only for a field it
    **measures whole**: the source field of an internal edge that delivers
    it as it is, with no mapping and no transform
    (:func:`maddening.core.edge._delivered` applies nothing else) -- an
    edge whose reading *is* its source field, which the group's plan
    answers (``InterfaceEdge.reads_source_as_is``).  Every
    other floating field could be returned from a pass before the readings
    the verdict was taken on, with ``converged=True``:

    * a field **no internal edge reads** (a one-way pair under Jacobi
      returned its target computed from the pre-step source at
      ``iterations=1`` and a residual of exactly zero: MADD-ANO-240);
    * a field internal edges read **only through a mapping or a
      transform**, which may deliver less than the field: the part of it
      they do not deliver is measured by nothing (MADD-ANO-241).

    A static mapping onto more entries than its source holds is read on
    its compact side, at the source (``_interface_plan._norm_side``): the
    reading is the source field itself, before the mapping and the
    transform, so that field is measured whole and kept, like one a plain
    edge reads.

    The return rule (``_with_nonfloat_fields_at`` in the step): a field
    measured whole keeps the accepted iterate's value, bit for bit; every
    field named here takes the value **one plain pass of the group's own
    schedule computes at the accepted iterate**, from the readings the
    verdict was taken on.  One rule for every member, schedule,
    acceleration, solver and verdict: under a relaxation an unmeasured
    field is a blend of every pass so far, and a member a sweep feeds
    forward lags too once its inputs came through a mapping, so no
    narrower set was found that is right everywhere.

    ``{}`` for every other norm, and for an interface group whose every
    floating field is measured whole: such a group keeps its compiled
    step and is not charged the pass.  Static.  **The one place the set
    is defined**, and a derivation over the group's description:
    *interface_edges* is its plan (or a bare sequence of edges), and
    which edge reads its source field as it is is the plan's answer, on
    the side the norm reads that edge.  So a field the norm does measure
    whole is never recomputed, whichever side that is.

    Which fields are floating is decided on *state* alone (the step
    hands the first pass's, the state its non-floating fields are named
    on), as the norm's reading decides it on the state it is handed
    (``acceleration._interface_readings``) and not on a record's
    ``source_kind``, which is of the state the plan was built from.  The
    two agree wherever a pass keeps each field's dtype kind.
    """
    if group.convergence_norm != "interface":
        return {}
    floats = float_fields_of(state, list(schedule))
    whole = {record.source
             for record in _interface_plan.interface_records(interface_edges)
             if record.reads_source_as_is}
    missed = {nn: tuple(f for f in floats[nn] if (nn, f) not in whole)
              for nn in schedule}
    return {nn: fields for nn, fields in missed.items() if fields}


#: Every ``_meta`` slot a coupling group can own, as the suffix after
#: ``coupling_<group key>_``.  Read by :func:`_refuse_colliding_group_keys`.
_GROUP_META_SUFFIXES = (
    "iterations", "total_iterations", "residual", "amplification", "rho_spectral",
    "spectral_residual", "spectral_amplification",
    "gradient_relative_error_bound", "pass_evaluations", "reading_floor",
    "geometry_gap", "geometry_plane_limit", "V", "W", "pred_count", "pred_0", "pred_1",
    "pred_2",
)


def _refuse_colliding_group_keys(groups) -> None:
    """Raise if two coupling groups would share a report key or a ``_meta`` slot.

    A group is keyed by its sorted node names joined with ``"+"`` -- the
    key ``coupling_diagnostics()`` reports it under, and the prefix of
    every ``_meta`` slot it owns (diagnostics, IQN-IMVJ warm starts,
    predictor history).  Node names may themselves contain ``"+"`` (and
    ``"_"``), so two different groups can produce the same key --
    ``{"a+b", "c"}`` and ``{"a", "b+c"}`` are both ``a+b+c`` -- or one
    group's key plus a suffix can spell another's slot (a group keyed
    ``a+b_spectral`` owns ``coupling_a+b_spectral_residual``, which is
    ``a+b``'s ``spectral_residual``).  Either way one report stood for
    two groups and their carries overwrote each other; IQN-IMVJ warm
    starts of different sizes failed with a broadcasting ``ValueError``
    deep inside the step.

    Refused here, at registration, rather than keyed differently: every
    existing report key stays what it was, and the only graphs affected
    are those that could not have been reported correctly anyway.
    """
    seen: dict[str, frozenset] = {}
    slots: dict[str, frozenset] = {}
    for group in groups:
        key = "+".join(sorted(group.nodes))
        other = seen.get(key)
        if other is not None and other != group.nodes:
            raise ValueError(
                f"Coupling groups {sorted(other)} and {sorted(group.nodes)} "
                f"would share the diagnostics key {key!r}: a group is keyed "
                "by its sorted node names joined with '+', and node names "
                "may contain '+'.  Rename a node so the two keys differ."
            )
        seen[key] = group.nodes
        for suffix in _GROUP_META_SUFFIXES:
            slot = f"coupling_{key}_{suffix}"
            owner = slots.get(slot)
            if owner is not None and owner != group.nodes:
                raise ValueError(
                    f"Coupling groups {sorted(owner)} and {sorted(group.nodes)} "
                    f"would share the internal state slot {slot!r}: a group's "
                    "slots are named from its sorted node names joined with "
                    "'+', and one group's name plus a suffix spells the "
                    "other's.  Rename a node so the two keys differ."
                )
            slots[slot] = group.nodes


#: Relative tolerance on "``macro_dt / node_dt`` is a whole number" for a
#: sub-cycled group.  Timesteps are decimal floats, so an exact ratio such
#: as ``0.01 / 0.001`` computes as ``10.000000000000002``: a few float64
#: ulps (~1e-15 relative) of representation noise, which this must admit.
#: A genuine mismatch is a decimal digit off (``0.01 / 0.003``, 11% off)
#: and must not be admitted.  At 1e-9 the rounded divider drifts the
#: member's clock by at most one sub-step in a billion macro steps, far
#: below anything a simulation can resolve, and it is the tolerance the
#: multi-rate scheduler already uses for its base timestep
#: (:func:`_float_gcd`).
_SUBCYCLING_RATIO_RTOL = 1e-9


def _subcycling_ratio_errors(group, nodes) -> list[str]:
    """``ERROR:`` issues for sub-cycled members whose timestep does not divide.

    A sub-cycled node takes ``round(macro_dt / node_dt)`` sub-steps of its
    own timestep per coupling pass (:func:`_group_dividers`), so unless the
    ratio is a whole number it covers ``divider * node_dt`` per macro step,
    not ``macro_dt``: its clock drifts from the rest of the graph by a
    fixed fraction of every step, silently.  ``0.003`` in a group whose
    macro timestep is ``0.01`` took three sub-steps and covered ``0.009``.
    Refused, naming the two nearest timesteps that do divide.
    """
    names = sorted(n for n in group.nodes if n in nodes)
    if not names:
        return []
    macro = max(nodes[n].timestep for n in names)
    errors = []
    for n in names:
        node_dt = nodes[n].timestep
        ratio = macro / node_dt
        whole = max(round(ratio), 1)
        if abs(ratio - whole) <= _SUBCYCLING_RATIO_RTOL * ratio:
            continue
        below, above = max(math.floor(ratio), 1), math.ceil(ratio)
        nearest = sorted({macro / above, macro / below}, reverse=True)
        errors.append(
            f"ERROR: coupling group {names}: node {n!r} has timestep "
            f"{node_dt:.6g}, which does not divide the group's macro timestep "
            f"{macro:.6g} (ratio {ratio:.6g}).  A sub-cycled node takes a whole "
            f"number of sub-steps per coupling pass, so it would take "
            f"{whole} of {node_dt:.6g} and cover {whole * node_dt:.6g} per "
            f"macro step instead of {macro:.6g}.  Give it a timestep that "
            f"divides {macro:.6g} -- the nearest are "
            + " and ".join(f"{t:.6g} ({macro:.6g}/{round(macro / t)})"
                           for t in nearest)
            + " -- or change the macro timestep."
        )
    return errors


#: The mapping kind whose moving geometry the coupling diagnostics read.
_DIAGNOSED_GEOMETRY_KIND = "multilinear_grid"
#: The convergence norms a group with a geometry edge is diagnosed under:
#: the two that measure the members' state, the geometry included.
_DIAGNOSED_GEOMETRY_NORMS = ("l2", "mixed")

#: Why a group whose pass resolves a geometry-dependent mapping reports no
#: bound (``coupling_diagnostics()[key]["not_usable_reason"]``): the stem
#: every such reason starts with, and what each case adds.
_GEOMETRY_DIAGNOSTICS_REASON = (
    "the group resolves geometry-dependent mapping(s) on edge(s) {keys}; in 0.4.0 the "
    "coupling diagnostics do not read a moving geometry {why}, so no bound or estimate "
    "is reported for this group. iterations, residual and converged are the solve's own."
)
_GEOMETRY_KIND_WHY = (
    "of a mapping kind other than 'multilinear_grid' (edge(s) {others} carry kind(s) "
    "{kinds})"
)
_GEOMETRY_NORM_WHY = (
    "under convergence_norm={norm!r} (they do under 'l2' and 'mixed', which measure the "
    "members' state, the geometry included)"
)
_GEOMETRY_SUBCYCLED_WHY = (
    "in a sub-cycled group (members {members} take several sub-steps per pass, and the "
    "geometry of each sub-step is not followed)"
)
#: The check's gap over its tolerance.  One number does not say which of
#: two things it measured, so the reason names both: a product that is
#: not the derivative, and a finite difference that is not one either.
#: Three honest passes read a gap (``GEOMETRY_GAP_TOLERANCE`` has the
#: measurements): a position the pass builds and then reads in the same
#: sweep within the check's step of a plane of the lattice (the step
#: moves it through the mapped input, across the plane); a field whose
#: update passes through a much larger one in the same sweep (a gather
#: of a field that changes sign across a lattice cell, read under
#: Gauss-Seidel, rounds at the larger field's resolution, and the
#: allowance of ``_bounds._GEOMETRY_GAP_RESOLUTIONS`` is in the smaller
#: one's); a field in a coarser dtype than the step was sized for.
_GEOMETRY_SELF_CHECK_WHY = (
    "where the pass's Jacobian-vector product along the geometry disagrees with a finite "
    "difference of the pass along the same direction (relative gap {gap:.3g}, allowed "
    "{allowed:.3g}: either a member or a mapping whose derivative is not that of its value, "
    "or a finite difference that could not be formed at this state -- a position the pass "
    "builds and then reads in the same sweep lies within the check's step of a plane of "
    "the lattice, or a mapped sample cancels digits, as a gather of a field that changes "
    "sign across a lattice cell does, or a field in a coarser dtype does not resolve the "
    "step; the check does not tell these apart)"
)
#: The same check where it produced no number: nothing was compared, so
#: nothing is said about any member's derivative.
_GEOMETRY_SELF_CHECK_UNEVALUATED_WHY = (
    "where the pass's Jacobian-vector product along the geometry could not be compared "
    "with a finite difference of the pass (relative gap {gap:.3g}, allowed {allowed:.3g}: "
    "the pass or its product is not a number at the returned state, or the pass reads a "
    "geometry from a constant the step could not move)"
)
#: Why ``spectral_usable`` is False for a group whose step passed its
#: self-check (``_bounds._geometry_plane_limit``, with the gradient
#: bound's Newton-Kantorovich check).  The numbers stay: they are the
#: linearisation's own, in the lattice cells of the returned iterate.
_GEOMETRY_PLANE_REASON = (
    "the group resolves geometry-dependent mapping(s) on edge(s) {keys}; a position its "
    "pass reads from the iterate is within {reach:g} times spectral_error_bound of a "
    "lattice plane of the mapping's grid or of a face of its hull (spectral_error_bound is "
    "{bound:.3g}; no plane is within its reach up to {limit:.3g} at this state), and the "
    "step did not certify its linearisation across the Newton step to the fixed point "
    "(gradient_relative_error_bound is not finite). Across a lattice plane the mapping is "
    "another polynomial of the positions, and rho_spectral and spectral_error_bound are "
    "the linearisation at the returned iterate, which describes the pass only in the "
    "lattice cells its positions are in there: the fixed point may be in another cell, "
    "where the pass contracts at another rate. spectral_usable and gradient_bound_usable "
    "are therefore False; the numbers are reported as computed. The bound shrinks with the "
    "residual: a tighter tolerance usually brings the iterate into the fixed point's cell."
)


def _geometry_diagnostics_refusal(group, nodes, plan) -> Optional[str]:
    """Why *group*'s report withholds its bounds on account of a geometry
    edge, whatever the step measures; ``None`` for a group without one and
    for a group the diagnostics read the geometry of.

    Experimental.  The diagnostics read a moving geometry for the
    ``multilinear_grid`` kind, under ``convergence_norm="l2"`` or
    ``"mixed"``, in a group that does not sub-cycle.  The step then checks
    its own Jacobian-vector product along the geometry
    (``_bounds._geometry_product_gap``), and the report withholds the
    bounds where that check fails
    (:func:`_geometry_self_check_reason`).  *plan* is the group's
    description: the edges are ``InterfacePlan.resolved_geometry_edges``.
    """
    geometry = plan.resolved_geometry_edges()
    if not geometry:
        return None
    keys = [r.key for r in geometry]
    others = [r for r in geometry if r.mapping_kind != _DIAGNOSED_GEOMETRY_KIND]
    if others:
        why = _GEOMETRY_KIND_WHY.format(
            others=[r.key for r in others],
            kinds=sorted({str(r.mapping_kind) for r in others}))
    elif group.convergence_norm not in _DIAGNOSED_GEOMETRY_NORMS:
        why = _GEOMETRY_NORM_WHY.format(norm=group.convergence_norm)
    elif _group_dividers(group, nodes):
        dividers = _group_dividers(group, nodes) or {}
        why = _GEOMETRY_SUBCYCLED_WHY.format(
            members=sorted(n for n, d in dividers.items() if d > 1))
    else:
        return None
    return _GEOMETRY_DIAGNOSTICS_REASON.format(keys=keys, why=why)


def _geometry_self_check_reason(keys, gap: float, allowed: float) -> str:
    """The reason of a report whose step failed its geometry self-check:
    a gap over *allowed* (the product is not the derivative, or the
    finite difference is rounding: the two read alike), or a gap that is
    not a number (the two could not be compared)."""
    why = _GEOMETRY_SELF_CHECK_WHY if gap == gap else _GEOMETRY_SELF_CHECK_UNEVALUATED_WHY
    return _GEOMETRY_DIAGNOSTICS_REASON.format(
        keys=list(keys), why=why.format(gap=gap, allowed=allowed))


def _geometry_plane_reason(keys, bound: float, limit: float, reach: float) -> str:
    """The reason of a report whose bound reaches a lattice plane
    (``spectral_error_bound`` over the step's ``geometry_plane_limit``)."""
    return _GEOMETRY_PLANE_REASON.format(
        keys=list(keys), bound=bound, limit=limit, reach=reach)


_WRITTEN_BEFORE_SAVE_REASON = (
    "this report was loaded from a checkpoint saved after the group's state had been "
    "written (set_node_state) since its last step; the state that step returned, which "
    "the float floor is measured on, is not in the checkpoint, so spectral_error_bound, "
    "precision_limited and the *_usable flags are not reported until the group steps. "
    "iterations, residual, converged and the estimates are the step's own."
)


def _geometry_edge_coupling_errors(group, plan) -> list[str]:
    """``ERROR:`` issues for a group setting a geometry-dependent mapping
    cannot serve (experimental; empty for every other group).

    ``convergence_norm="interface"`` measures the change, between
    iterates, of what each internal edge delivers from its source field.
    For an edge whose mapping reads a geometry that reading needs the
    geometry of each iterate too, which the norm does not read in 0.4.0:
    refused, naming the norms that measure the state instead.  *plan* is
    the group's description (``InterfacePlan.geometry_edges``).
    """
    if group.convergence_norm != "interface":
        return []
    names = sorted(group.nodes)
    return [
        f"ERROR: coupling group {names} uses convergence_norm='interface', which "
        f"measures the values the group's internal edges carry, but edge {r.key!r} "
        f"carries its value through a geometry-dependent mapping (geometry "
        f"{r.anchor[0]}.{r.anchor[1]}), and the norm does not read a moving "
        f"geometry in 0.4.0.  Use convergence_norm='mixed' or 'l2', which measure "
        f"the members' state, the geometry included."
        for r in plan.geometry_edges()
    ]


def _flux_edge_coupling_errors(group, nodes, plan, state) -> list[str]:
    """``ERROR:`` issues for group-internal flux edges the group cannot read.

    A flux edge carries a value ``compute_boundary_fluxes`` returns, which
    is not a field of the producer's state.  Two parts of the coupling
    loop read an internal edge's value *from the state* and so cannot
    serve one, and both used to fail inside the trace with a bare
    ``KeyError`` naming the flux:

    * ``convergence_norm="interface"`` measures the change, between
      iterates, of what each internal edge delivers from its source
      field;
    * a sub-cycled member under ``boundary_interpolation="linear"`` (or
      ``"quadratic"``, which is linear) interpolates each internal input
      between the pass's incoming iterate and the in-pass state.

    Refused here, naming the setting that does work.  (Validation has
    already checked that a source field missing from the state is one of
    the producer's fluxes.)  *plan* is the group's description, built
    from *state* (``InterfacePlan.flux_edges``).
    """
    names = sorted(group.nodes)
    if not all(n in nodes and n in state for n in names):
        return []
    flux_edges = plan.flux_edges()
    errors = []
    for r in flux_edges:
        if group.convergence_norm == "interface":
            errors.append(
                f"ERROR: coupling group {names} uses convergence_norm="
                f"'interface', which measures the values the group's internal "
                f"edges carry, but edge {r.key!r} carries the boundary flux "
                f"{r.source[1]!r}, which {r.source[0]!r} computes in "
                "compute_boundary_fluxes and does not hold in its state, so "
                "the norm cannot read it.  Use convergence_norm='mixed' or "
                "'l2', which measure the state the flux is computed from."
            )
    dividers = _group_dividers(group, nodes) or {}
    if group.boundary_interpolation != "constant":
        for r in flux_edges:
            if dividers.get(r.target[0], 1) > 1:
                errors.append(
                    f"ERROR: coupling group {names}: node {r.target[0]!r} is "
                    f"sub-cycled ({dividers[r.target[0]]} sub-steps per pass) "
                    f"and reads the boundary flux {r.source[1]!r} of "
                    f"{r.source[0]!r} through edge {r.key!r}.  "
                    f"boundary_interpolation={group.boundary_interpolation!r} "
                    "interpolates an input between the pass's incoming "
                    "iterate and the in-pass state, and a flux is computed "
                    "in compute_boundary_fluxes rather than held in either.  "
                    "Set boundary_interpolation='constant', which reads the "
                    "in-pass flux at every sub-step."
                )
    return errors


def _group_dividers(group, nodes):
    """Evaluations of each node per coupling pass under ``subcycling=True``, or ``None``.

    ``None`` when the group does not sub-cycle: ``subcycling=False``, or
    every node shares one timestep.  Otherwise each node is advanced
    ``round(macro_dt / node_dt)`` times per pass, ``macro_dt`` the
    largest timestep in the group.
    """
    if not group.subcycling:
        return None
    names = sorted(group.nodes)
    steps = sorted({nodes[nn].timestep for nn in names})
    if len(steps) <= 1:
        return None
    macro = max(steps)
    return {nn: max(round(macro / nodes[nn].timestep), 1) for nn in names}


def _group_waveform_sweeps(group, nodes) -> int:
    """How many waveform sweeps one step of *group* runs.

    ``waveform_iterations`` for a group that sub-cycles, and ``1`` for
    every other group whatever ``waveform_iterations`` says (see
    :func:`_group_dividers`).  Each sweep is a whole fixed-point solve of
    up to ``max_iterations`` passes, so a step that runs more than one
    reports the largest sweep's count as ``"iterations"`` and their sum
    as ``"total_iterations"`` -- the sum carried in a ``_meta`` slot that
    only a group running more than one sweep owns.
    """
    return int(group.waveform_iterations) if _group_dividers(group, nodes) else 1


def _block_schedule(schedule, groups):
    """*schedule* with each coupling group's members moved together, at its first member's place.

    The step runs a coupling group as one block where its first member
    is scheduled (``_build_step_fn``), so a node scheduled *between* two
    members -- possible where the group is only part of a larger feedback
    loop, whose strongly connected component ``topological_sort`` keeps in
    ``add_node`` order -- in fact ran after the whole group.  Back edges
    were decided over the node order all the same: that node's read of a
    later member was staggered to the previous step although the member
    had already run this step, and the same graph built with the outside
    nodes added in another order stepped differently (the round-5 audit's
    case: one outside node added between the members, ``A, D, B, E``,
    read ``B`` one step late).  Deciding them over the block order, which
    is what this reorder makes the schedule, reads a source whose block
    already ran fresh.  A graph whose groups are already contiguous --
    every graph built without such an interleaving -- keeps its schedule.
    """
    member_of = {nn: id(g) for g in groups for nn in g.nodes}
    out: list[str] = []
    placed: set[int] = set()
    for nn in schedule:
        gid = member_of.get(nn)
        if gid is None:
            out.append(nn)
        elif gid not in placed:
            placed.add(gid)
            out.extend(m for m in schedule if member_of.get(m) == gid)
    return out


def _loop_through_outside_nodes(schedule, edges, groups, back_edges):
    """``UserWarning`` texts for each coupling group that is a strict subset of a feedback loop.

    Where a group's members and some outside nodes form one strongly
    connected component, the loop through the outside nodes is closed by
    a back edge, read from the previous step, and *which* edge that is
    depends on the order the nodes were added: another order staggers
    another edge and steps differently (CPL-181).  The step is right for
    the order it was given, but the result depends on a choice the user
    did not know they were making, so ``compile()`` names it.
    """
    from maddening.core.schedule import find_strongly_connected_components  # noqa: PLC0415

    names = list(schedule)
    texts = []
    for scc in find_strongly_connected_components(names, edges):
        component = set(scc)
        for g in groups:
            if not (set(g.nodes) <= component) or set(g.nodes) == component:
                continue
            outside = sorted(component - set(g.nodes))
            staggered = sorted(
                f"{e.source_node}.{e.source_field} -> {e.target_node}.{e.target_field}"
                for e in back_edges
                if _interface_plan.is_internal(e, component)
                and not _interface_plan.is_internal(e, g.nodes))
            texts.append(
                f"coupling group {sorted(g.nodes)} is part of a larger feedback loop "
                f"through {outside}, which the group does not iterate: that loop is "
                f"closed by reading {staggered} from the previous step, and which edge "
                "is read late depends on the order the nodes were added (another order "
                "staggers another edge and steps differently).  Add those nodes to the "
                "group to iterate the whole loop, or accept the one-step lag as part of "
                "the model.")
    return texts


def _staggered_across_components(schedule, edges, groups, back_edges):
    """``UserWarning`` texts for each back edge that joins two strongly connected components.

    An edge between two components always points forward (CPL-025): no
    cycle runs through it, so an uncoupled step reads it this step.  A
    coupling group runs as one block at its first member's place
    (``_block_schedule``), so a group whose members are joined only through
    an outside node -- ``a -> c -> b`` and ``a -> b``, no edge back, so no
    cycle at all -- runs ``a`` and ``b`` together before ``c``, and
    ``identify_back_edges`` staggers ``c -> b``: ``b`` reads ``c`` from the
    previous step.  ``CouplingGroup`` documents that its members "form
    (part of) a cycle"; nothing checked it, and the lag was silent
    (MADD-ANO-159).  ``compile()`` names each such edge and the group whose
    block forced it.
    """
    from maddening.core.schedule import find_strongly_connected_components  # noqa: PLC0415

    # ``find_strongly_connected_components`` returns the cycles only; every
    # other node is a component of its own (a self-loop stays inside it).
    component = {name: ("node", name) for name in schedule}
    for i, scc in enumerate(find_strongly_connected_components(list(schedule), edges)):
        for name in scc:
            component[name] = ("cycle", i)
    texts = []
    for e in sorted(back_edges, key=lambda e: (e.source_node, e.source_field,
                                               e.target_node, e.target_field)):
        if component.get(e.source_node) == component.get(e.target_node):
            continue
        blocks = sorted(sorted(g.nodes) for g in groups
                        if e.source_node in g.nodes or e.target_node in g.nodes)
        texts.append(
            f"'{e.source_node}.{e.source_field} -> {e.target_node}.{e.target_field}' "
            "is read from the previous step although no cycle runs through it: "
            f"coupling group {blocks[0] if len(blocks) == 1 else blocks} runs as one "
            "block, and its members are joined through nodes outside it rather than "
            "on a cycle, so the block runs before the edge's source.  A coupling "
            "group's members must form (part of) a cycle: add the nodes that join "
            "them to the group, or split the group, or accept the one-step lag as "
            "part of the model.")
    return texts


def _declared_evaluations(node):
    """``node.update_evaluations()``, validated: a finite number ``>= 1``, or ``None``."""
    own = getattr(node, "update_evaluations", None)
    value = own() if callable(own) else None
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) \
            or not math.isfinite(float(value)) or float(value) < 1.0:
        raise ValueError(
            f"node {getattr(node, 'name', node)!r}: update_evaluations() returned "
            f"{value!r}; it must be None (not declared) or a finite number >= 1 -- "
            "the number of evaluations' worth of rounding one update carries."
        )
    return float(value)


def _group_evaluations(group, nodes, schedule, edges):
    """``(evaluations, declared)``: how many evaluations one coupling pass rounds like.

    The float floor of the coupling bound
    (:func:`~maddening.core.coupling.acceleration.residual_precision_floor`)
    is ``PRECISION_FLOOR_ULPS`` units of ``eps * max|field|`` *per
    evaluation* of the one-pass map, a constant calibrated on a pass
    that evaluates each node once.  A pass evaluates node ``n`` ``d_n``
    times (its sub-cycling divider, else once), and each evaluation
    rounds like ``e_n`` (:meth:`SimulationNode.update_evaluations`,
    undeclared counting as one), so node ``n``'s own update rounds like
    ``d_n * e_n`` single evaluations.  ``declared`` is whether every
    node declared its ``e_n``.  Measured, float32: a node sub-cycled 100
    times per pass (or looping 100 explicit Euler sub-steps inside
    ``update``) is ~29 units off the exact map, against the 4 the floor
    allowed before it was counted -- and the bound read 0.16x the true
    distance with ``spectral_usable=True``.

    **How the nodes' counts combine depends on the iteration mode.**
    Under ``"jacobi"`` every node reads the previous iterate, which is
    stored and exact, so a node's output carries its own rounding only
    and the pass rounds like its worst node: ``max_n d_n e_n``.  Under
    ``"gauss-seidel"`` a node reads the output of every group member
    scheduled before it *from the same pass*, already rounded, so its
    output carries that rounding too: the pass is a composition, and
    the error at the end of a chain of same-pass reads is the sum of
    the roundings along it.  ``evaluations`` is then the longest such
    chain, ``depth(n) = d_n e_n + max(depth(m))`` over the members ``m``
    scheduled before ``n`` that ``n`` reads (an edge from a member
    scheduled at or after ``n`` reads the previous iterate and starts no
    chain).  Measured on a Gauss-Seidel ring of ``N`` scalar relays of
    loop gain 0.99, every node declaring one evaluation: the exact
    residual at the float32 stall is 0.6, 1.2 and 2.3x the per-pass
    floor at ``N`` = 16, 32 and 64, and the bound, which took the worst
    node's count, read 0.51x (``N = 32``) and 0.30x (``N = 64``) the
    true distance with ``spectral_usable=True`` (MADD-ANO-094); counted
    along the chain the floor is ``N`` times larger and the bound reads
    23x and 16x over at ``N`` = 16 and 32 (the gradient bound 36x and
    17x over its true error; jaxlib 0.11.0, CPU).  The
    longest chain and not the sum over the members: a rounding reaches
    a node only along a path of reads, and where several paths meet the
    node's output is a combination of its inputs whose relative gains
    -- in the norm's units, each field divided by its own magnitude --
    sum to at most one *if* no read amplifies.  That is the structural
    count, all this function sees.  A read can amplify (a squaring
    relay doubles a relative rounding; a node whose terms cancel
    amplifies its own as well), so with ``diagnostics=True`` the step
    weights every read by its relative gain measured at the returned
    state and reports the larger count (``_run_coupled_block_impl``,
    ``coupling_<key>_pass_evaluations``; see
    :data:`~maddening.core.coupling.acceleration.PRECISION_FLOOR_ULPS`).
    For a chain or a ring the longest chain and the sum coincide.

    ``schedule`` is the order the step sweeps the group's members in
    (the compiled schedule; members not in it are swept last, each
    starting a chain of its own) and ``edges`` every edge of the graph;
    only those between two members count.
    """
    order, own, same_pass, declared = _group_pass_structure(group, nodes, schedule, edges)
    if group.iteration_mode == "jacobi":
        return max([1.0, *own.values()]), declared
    depth: dict[str, float] = {}
    for nn in order:
        depth[nn] = own[nn] + max((depth[m] for m in same_pass[nn]), default=0.0)
    return max([1.0, *depth.values()]), declared


def _group_pass_structure(group, nodes, schedule, edges):
    """``(order, own, same_pass, declared)``: what a pass's rounding is counted on.

    ``order`` is the sweep order (``schedule`` restricted to the group,
    any member missing from it last), ``own[n]`` node ``n``'s own count
    ``d_n e_n`` (sub-cycling divider times declared evaluations, one
    where undeclared), ``same_pass[n]`` the members scheduled before
    ``n`` that ``n`` reads (an empty set for every node under Jacobi is
    the caller's business: this lists the reads), and ``declared``
    whether every member declared its count.  See
    :func:`_group_evaluations`.
    """
    dividers = _group_dividers(group, nodes) or {}
    declared = True
    own: dict[str, float] = {}
    for nn in sorted(group.nodes):
        count = _declared_evaluations(nodes[nn].node)
        if count is None:
            declared = False
            count = 1.0
        own[nn] = float(dividers.get(nn, 1)) * count
    order = [nn for nn in schedule if nn in group.nodes]
    order += sorted(nn for nn in group.nodes if nn not in order)
    position = {nn: i for i, nn in enumerate(order)}
    reads = _interface_plan.member_reads(edges, position)
    same_pass: dict[str, set] = {
        nn: {src for src in reads[nn] if position[src] < position[nn]} for nn in order}
    return order, own, same_pass, declared


def _group_residual_dtype(state, node_names):
    """The dtype a group's residual and its ``_meta`` slots are held in.

    The promotion of every floating field of the group's nodes -- the
    dtype of the fixed-point vector both solvers iterate on and of the
    norm computed from it -- taken over the nodes in sorted order.
    ``jnp.result_type`` is order-independent, but the loop it replaced
    was not: it took the dtype of the *first* floating leaf it met
    iterating ``group.nodes``, a frozenset whose order follows the
    per-process string hash, so a group with a float16 field beside a
    float32 one was seeded float16 or float32 depending on
    ``PYTHONHASHSEED``, and ``run_scan`` raised a scan-carry dtype
    ``TypeError`` in some processes and not in others.  ``float32`` for
    a group with no floating field.

    **At least float32.**  Every norm measures and accumulates in at
    least float32 (``acceleration._widened``), so a bfloat16 or float16
    group's residual is a float32 number, and it is held in one: kept in
    the group's 16-bit dtype, a residual the norm computed finitely
    overflowed again on the way into the loop's carry or the report slot
    (float16 holds nothing above 65 504, and a mixed-norm ratio at
    ``rtol=1e-6`` is about 977 per ulp), and the verdict compared a
    float16 rounding of it.  A float32 or float64 group is unchanged.
    """
    dtypes = [
        jnp.asarray(leaf).dtype
        for nn in sorted(node_names)
        for leaf in state.get(nn, {}).values()
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating)
    ]
    if not dtypes:
        return jnp.dtype(jnp.float32)
    return jnp.promote_types(jnp.result_type(*dtypes), jnp.float32)


def _state_measurable(state, node_names, norm):
    """Whether *state* is one a coupling group's norm can measure a change at.

    Every floating field of every member finite (:func:`_group_state_finite`),
    and every field the norm reads within the range its dtype can measure
    a change at: the group's norm of *state* against itself (``norm(s_new,
    s_old)``, the group's ``_compute_residual``) is ``0.0`` there and ``inf``
    on a field it cannot evaluate (``acceleration._scaled_change``).  A
    verdict on the *state*, not on the residual: ``strict_convergence``
    reads it to tell a diverged iteration from one whose estimate was
    non-finite on a perfectly finite state.
    """
    return jnp.logical_and(_group_state_finite(state, node_names),
                           jnp.isfinite(norm(state, state)))


def _group_state_finite(state, node_names):
    """Whether every floating field of the group's nodes is finite."""
    ok = jnp.array(True)
    for nn in node_names:
        for v in state[nn].values():
            v = jnp.asarray(v)
            if jnp.issubdtype(v.dtype, jnp.floating) and _has_entries(v):
                ok = jnp.logical_and(ok, jnp.all(jnp.isfinite(v)))
    return ok


def _non_finite_reads_as_diverged(state_finite, residual, amplification):
    """``(residual, amplification)``, reported as diverged on a non-finite state.

    Every norm reports ``inf`` for a field it cannot evaluate
    (MADD-ANO-019), but a norm only sees the fields it reads: under
    ``convergence_norm="interface"`` a ``NaN`` in a field no edge reads
    -- from the initial state, a parameter or an external input -- left
    the residual finite, and the group was reported ``converged=True``,
    ``strict_convergence`` did not raise and ``spectral_usable`` was
    ``True`` on a state that was not finite.  The verdict is about the
    state the step returns, so it is taken over every floating field of
    it: a non-finite one makes the residual ``inf`` and rejects the
    amplification, which is exactly what the norms report for a field
    they read.  A finite state takes the selected branch of a ``where``
    and is reported bit-identically.
    """
    residual = jnp.asarray(residual)
    amplification = jnp.asarray(amplification)
    return (
        jnp.where(state_finite, residual, jnp.full_like(residual, jnp.inf)),
        jnp.where(state_finite, amplification, jnp.zeros_like(amplification)),
    )


def _group_without_member(group: CouplingGroup, name: str) -> Optional[CouplingGroup]:
    """*group* as ``GraphManager.remove_node`` leaves it when node *name*
    is removed: the same object when it does not name the node; ``None``
    (the group is removed) when fewer than two members would remain;
    otherwise a group of the remaining members with every option kept --
    ``accelerated_fields`` without the node's entry, and ``None`` (the
    interface fields) when no other entry selects a field, which
    ``CouplingGroup`` refuses to be told with an empty mapping.  The
    options were validated, and warned about, when the group was made."""
    if name not in group.nodes:
        return group
    nodes = group.nodes - {name}
    if len(nodes) < 2:
        return None
    accelerated = group.accelerated_fields
    if accelerated is not None:
        accelerated = {k: v for k, v in accelerated.items() if k != name}
        if not any(accelerated.values()):
            accelerated = None
    # A copy with the two fields set, not ``dataclasses.replace``: that
    # runs ``__post_init__`` again, which warns again about every option
    # the group was already warned about when it was made.  Nothing here
    # needs validating a second time: the members are a subset, and the
    # selection is a subset that still selects a field, or ``None``.
    smaller = copy.copy(group)
    object.__setattr__(smaller, "nodes", frozenset(nodes))
    object.__setattr__(smaller, "accelerated_fields", accelerated)
    return smaller
