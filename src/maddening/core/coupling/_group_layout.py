"""Structure of a coupling group: its interface fields, its pass and
sub-cycle schedule, and the refusals raised when a group is declared.

Moved verbatim out of ``maddening.core.graph_manager``.  Private.
"""

from __future__ import annotations

import math
from typing import Optional

import jax.numpy as jnp
import numpy as np

from maddening.core.coupling.acceleration import float_fields_of


def _interface_edge_order(edges, member_order) -> list:
    """A coupling group's internal *edges* in the order its interface norm sums them.

    By the source's place in *member_order* (the group's sweep), then the
    source field, the target's place, the target field and the ordinal:
    the order the L2 and mixed norms sum the members in, so the interface
    norm, like them, depends neither on the order of the ``add_edge``
    calls nor on the nodes' names.  Edges with the same endpoints (an
    additive pair with different transforms) keep their relative order.
    """
    place = {nn: i for i, nn in enumerate(member_order)}
    last = len(place)
    return sorted(edges, key=lambda e: (place.get(e.source_node, last), e.source_field,
                                        place.get(e.target_node, last), e.target_field,
                                        e.ordinal))


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


def _group_accel_fields(group, edges, state) -> Optional[dict]:
    """The fields ``group``'s acceleration flattens; ``None`` for all of them.

    The quasi-Newton accelerations read ``accelerated_fields`` (or the
    interface fields the group's internal edges read); ``"aitken"`` and
    ``"fixed"`` relax the whole group.  Either way only floating fields
    enter (:func:`_floating_accel_fields`).  Shared by the step builder
    and by ``compile()``'s IQN-IMVJ warm-start seeding, which must agree
    on the vector's length.
    """
    if group.acceleration in ("iqn-ils", "iqn-imvj"):
        chosen = (group.accelerated_fields if group.accelerated_fields is not None
                  else _interface_state_fields(edges, group.nodes, state))
    elif group.acceleration in ("aitken", "fixed"):
        chosen = None
    else:
        return None
    return _floating_accel_fields(chosen, state, group.nodes)


def _reads_mapping_weights(group, edges, state) -> bool:
    """Does *group*'s norm read a value that depends on interface-mapping weights?

    True under ``convergence_norm="interface"`` when an internal edge
    whose source field is floating carries a mapping: that norm reads
    what each internal edge delivers
    (:func:`~maddening.core.coupling.acceleration._interface_readings`),
    and a mapped edge delivers its source field through weights that
    live in ``params["mappings"]`` and may be overridden per step.  The
    float floor of such a group's residual therefore cannot be taken
    from the returned state alone, and the step records it
    (``coupling_<key>_reading_floor``).  Static, and shared by
    ``compile()``'s seeding, the step's write and ``reset_state()``, so
    the three agree on which groups own the slot; every other group's
    ``_meta`` and compiled step are what they were.
    """
    if group.convergence_norm != "interface":
        return False
    for e in edges:
        if (e.mapping is not None
                and e.source_node in group.nodes and e.target_node in group.nodes):
            value = state.get(e.source_node, {}).get(e.source_field)
            if value is not None and jnp.issubdtype(jnp.asarray(value).dtype, jnp.floating):
                return True
    return False


def _reading_is_the_fields(interface_edges, float_fields) -> bool:
    """Is the interface norm's reading the fields it reads, each of them once?

    ``coupling_residual_interface`` sums over a group's internal *edges*:
    what each one delivers, over its own magnitude.  The state's weights
    (``_norm_weights`` in the step) give each read *field* its own
    magnitude's weight once.  The two are one norm exactly when every
    internal edge with a floating source field delivers that field as it
    is -- no mapping, no transform -- and no field is read by more than
    one of them.  Then the report's spectral analysis is taken in the
    state's weights (``_spectral_rate_at``); otherwise on the reading
    (``_interface_spectral_rate_at``):

    * a mapping or a transform changes what an entry is and which
      magnitude it is divided by;
    * a field that ``k`` internal edges read is counted ``k`` times by
      the norm and once by the state's weights.  On a star whose hub's
      field every leaf reads, the residual of the edges times the
      resolvent of the fields read 0.26 to 0.73 of the true distance
      with ``spectral_usable=True`` (2 to 16 leaves, Jacobi, float64;
      MADD-ANO-213).

    Static: *interface_edges* are the group's internal edges and
    *float_fields* its floating fields by node, so a group keeps one
    analysis for the life of its compiled step.
    """
    read = set()
    for e in interface_edges:
        if e.source_field not in float_fields.get(e.source_node, ()):
            continue
        if e.transform is not None or e.mapping is not None:
            return False
        if (e.source_node, e.source_field) in read:
            return False
        read.add((e.source_node, e.source_field))
    return True


#: Every ``_meta`` slot a coupling group can own, as the suffix after
#: ``coupling_<group key>_``.  Read by :func:`_refuse_colliding_group_keys`.
_GROUP_META_SUFFIXES = (
    "iterations", "total_iterations", "residual", "amplification", "rho_spectral",
    "spectral_residual", "spectral_amplification",
    "gradient_relative_error_bound", "pass_evaluations", "reading_floor",
    "V", "W", "pred_count", "pred_0", "pred_1", "pred_2",
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


def _flux_edge_coupling_errors(group, nodes, edges, state) -> list[str]:
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
    the producer's fluxes.)
    """
    names = sorted(group.nodes)
    if not all(n in nodes and n in state for n in names):
        return []
    flux_edges = [
        e for e in edges
        if e.source_node in group.nodes and e.target_node in group.nodes
        and e.source_field not in state[e.source_node]
    ]
    errors = []
    for e in flux_edges:
        if group.convergence_norm == "interface":
            errors.append(
                f"ERROR: coupling group {names} uses convergence_norm="
                f"'interface', which measures the values the group's internal "
                f"edges carry, but edge {e.key!r} carries the boundary flux "
                f"{e.source_field!r}, which {e.source_node!r} computes in "
                "compute_boundary_fluxes and does not hold in its state, so "
                "the norm cannot read it.  Use convergence_norm='mixed' or "
                "'l2', which measure the state the flux is computed from."
            )
    dividers = _group_dividers(group, nodes) or {}
    if group.boundary_interpolation != "constant":
        for e in flux_edges:
            if dividers.get(e.target_node, 1) > 1:
                errors.append(
                    f"ERROR: coupling group {names}: node {e.target_node!r} is "
                    f"sub-cycled ({dividers[e.target_node]} sub-steps per pass) "
                    f"and reads the boundary flux {e.source_field!r} of "
                    f"{e.source_node!r} through edge {e.key!r}.  "
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
                if e.source_node in component and e.target_node in component
                and not (e.source_node in g.nodes and e.target_node in g.nodes))
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
    same_pass: dict[str, set] = {nn: set() for nn in order}
    for edge in edges:
        src, dst = edge.source_node, edge.target_node
        if src in position and dst in position and position[src] < position[dst]:
            same_pass[dst].add(src)
    return order, own, same_pass, declared


def _group_reads(group, edges):
    """``{member: members it reads through a group-internal edge}``, itself included."""
    reads: dict[str, set] = {nn: set() for nn in group.nodes}
    for edge in edges:
        if edge.source_node in group.nodes and edge.target_node in group.nodes:
            reads[edge.target_node].add(edge.source_node)
    return reads


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
            if jnp.issubdtype(v.dtype, jnp.floating) and v.size > 0:
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
