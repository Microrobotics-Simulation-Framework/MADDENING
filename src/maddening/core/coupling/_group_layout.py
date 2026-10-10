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

from maddening.core.coupling import _interface_plan, reason_codes
from maddening.core.coupling.acceleration import (
    PRECISION_FLOOR_ULPS,
    _has_entries,
    _positions_floors,
    float_fields_of,
)
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
    """Does *group*'s norm read a value the state a solve returns does not determine?

    True under ``convergence_norm="interface"`` when an internal edge
    whose source field is floating is read through its mapping: that
    norm reads such an edge as it is delivered
    (:func:`~maddening.core.coupling.acceleration._interface_readings`),
    its source field through weights that live in ``params["mappings"]``
    and may be overridden per step.  An edge the norm reads at its
    source (a mapping onto more entries than its field holds) is the
    stored field, whatever the weights, and does not count.  True also
    where an edge is read as delivered through a geometry-dependent
    mapping anchored at its **target**: the reading is taken at the
    target's pre-step geometry, which the returned state does not hold
    (such a mapping has no weights; one anchored at its source is read
    at the returned state's own geometry and does not count).  The
    float floor of such a group's residual therefore cannot be taken
    from the returned state alone, and the step records it
    (``coupling_<key>_reading_floor``).  Static, and shared by
    ``compile()``'s seeding, the step's write and ``reset_state()``, so
    the three agree on which groups own the slot; every other group's
    ``_meta`` and compiled step are what they were.  *plan* is the
    group's description (``InterfacePlan.norm_reads_beyond_the_state``).

    **With a dead band** (``atol > 0``) an edge read at its source counts
    too: the band is asked of the source field and of what the edge
    delivers, so which entries the floor counts depends on the weights
    the step ran with, or on its target's pre-step geometry
    (``InterfacePlan.band_reads_beyond_the_state``).  At the default
    ``atol == 0`` no group gains the slot.
    """
    return group.convergence_norm == "interface" and (
        plan.norm_reads_beyond_the_state()
        or (group.atol > 0 and plan.band_reads_beyond_the_state()))


#: The sentence every advisory about a dead band carries
#: (:func:`_dead_band_advisories`); its tests find the advisory by it.
_DEAD_BAND_ADVISORY = "a change that has to cross a field at or below atol is not seen"

#: How the advisory opens for a group under Jacobi.  The test configuration
#: filters ``compile()``'s copy of the advisory by THIS text, so only for
#: such a group (the domain batteries and the searches run them on
#: purpose); the advisory of a group of three or more members under
#: Gauss-Seidel is filtered nowhere.
_DEAD_BAND_UNDER_JACOBI = "declares a dead band under iteration_mode='jacobi'"

#: The fewest members at which a dead band is advised on under
#: Gauss-Seidel.  A pair held in both sweep orders in every case measured;
#: a ring of three did not (MADD-ANO-254).
_DEAD_BAND_MEMBERS = 3


def _dead_band_advisories(group) -> list:
    """The ``WARNING:`` line of a group whose dead band can hide a change.

    One line for a group with ``atol > 0`` that has **three or more
    members, or** ``iteration_mode="jacobi"``; none for any group at the
    default ``atol == 0``, and none for a pair under Gauss-Seidel.

    A field at or below ``atol`` leaves the residual, and the loop
    accepts on the first residual it measures (between its first two
    passes).  A change that has to cross such a field on its way to a
    field the norm keeps is therefore not seen until it arrives there:

    * **under Jacobi** every member reads the previous iterate, so with
      ``p`` dropped the residual of a pair ``p <- f(q)``, ``q <- g(p)``
      is ``|g(p_k) - q_k|``: nothing in that pass tests ``p = f(q)``.  A
      pair whose three forces of 1e-8 fall inside ``atol = 1e-6`` and
      are amplified by the member that reads them accepted after one
      pass on every other step, 4.5e5 to 7.4e5 tolerances off, under
      all three norms; where the dropped member carries state it
      accepted after two to four passes, 5e5 to 6.3e5 off, on every
      step after the first;
    * **on three or more members under Gauss-Seidel** a member swept
      before the one it reads still reads the previous pass.  A ring of
      three swept against its data flow (members added A, B, C; edges
      A -> C -> B -> A; two fields of 1e-8 inside ``atol = 1e-6`` in
      series between the change and the one kept field) accepted after
      one pass with the kept field 4.6e3 to 1.8e4 tolerances off, under
      all three norms, with no acceleration, with Aitken and with
      IQN-ILS: the change needs three passes to reach the kept field.

    (CPU, jaxlib 0.11.0.)  A PAIR under Gauss-Seidel held on the same
    data in both sweep orders in every case measured (0.5 to 4.9
    tolerances), and so does ``atol = 0`` on every group: there only a
    field that is exactly zero leaves the residual, and every non-zero
    field, however small, is measured against its own magnitude.

    Not refused, and not sharpened: the same ring swept along its data
    flow held, and so does any group none of whose kept fields depends
    on one that can fall inside the band, but which groups those are is
    a criterion nobody has proved, so the condition is the member count
    and the schedule and nothing read from the graph.  Static.  Read by
    ``GraphManager._coupling_group_advisories`` (``validate()``, and
    ``compile()`` as a ``UserWarning``).
    """
    members = len(group.nodes)
    jacobi = group.iteration_mode == "jacobi"
    if not (group.atol > 0 and (jacobi or members >= _DEAD_BAND_MEMBERS)):
        return []
    measured = []
    if jacobi:
        opening = f"{_DEAD_BAND_UNDER_JACOBI} (atol={group.atol!r}, {members} members)"
        measured.append("a pair under Jacobi accepted after one pass 4.5e5 to 7.4e5 "
                        "tolerances off")
    else:
        opening = (f"declares a dead band on {members} members (atol={group.atol!r}, "
                   f"iteration_mode={group.iteration_mode!r})")
    if members >= _DEAD_BAND_MEMBERS:
        measured.append("a ring of three members under Gauss-Seidel, swept against its "
                        "data flow, accepted after one pass 4.6e3 to 1.8e4 tolerances off")
    return [
        f"WARNING: coupling group {sorted(group.nodes)} {opening}: a field at or below atol "
        f"leaves the residual, and {_DEAD_BAND_ADVISORY} until it reaches a field the norm "
        f"keeps, so the group can report converged=True after one pass while a kept field "
        f"is thousands of tolerances from its fixed point (MADD-ANO-254; measured under all "
        f"three norms: {'; '.join(measured)}).  A pair under Gauss-Seidel is not affected in "
        f"any case measured (it held in both sweep orders).  Set atol=0.0 (the default) on this group: "
        f"then only a field that is exactly zero leaves the residual, and every non-zero "
        f"field, however small, is measured against its own magnitude."]


def _floor_needs_the_step(group, interface_edges) -> bool:
    """Can the float floor of *group*'s residual be measured only by the
    step that solved it, whatever weights the graph holds?

    True under ``convergence_norm="interface"`` where an internal edge
    is read as delivered at its target's pre-step geometry
    (``InterfaceEdge.reads_pre_step_geometry``): outside the step that
    state is gone.  The report's fallback floor
    (``coupling_diagnostics``, for a state whose ``reading_floor`` slot
    was never written) then has nothing to measure on and says so,
    where a group that reads mapping weights falls back to the graph's
    own.  *interface_edges* is the group's plan, or the bare edges the
    report keeps (``InterfacePlan.norm_edges``).  With a dead band
    (``atol > 0``) an edge read at its source through a mapping anchored
    at its target counts as well: the band is asked of what it delivers
    at that geometry.
    """
    return group.convergence_norm == "interface" and any(
        record.reads_pre_step_geometry
        or (group.atol > 0 and record.band_reads_pre_step_geometry)
        for record in _interface_plan.interface_records(interface_edges))


#: Why a report built on the float floor is withheld where the floor
#: could only have been measured by the step (:func:`_floor_needs_the_step`).
_FLOOR_NEEDS_THE_STEP_REASON = (
    "the float floor of this group's residual reads what an internal edge delivers at "
    "its target's pre-step geometry, which only the step that solved the group holds, "
    "and this state carries no floor recorded by a step (it was not produced by one of "
    "this graph's steps, or the recorded value is not finite); spectral_error_bound, "
    "precision_limited and the *_usable flags are not reported until the group steps."
)


#: The longest row of a static mapping whose own rounding the float floor
#: is taken to cover (MADD-ANO-257, open).  One limit for every static
#: kind: a dense matrix, a static sparse mapping in the gather layout or
#: in the scatter layout, a registered kind's own static class.
#:
#: **Measured, not proved.**  The floor counts ``PRECISION_FLOOR_ULPS``
#: units of ``eps`` per evaluation of a pass.  A mapped edge delivers
#: sums over its rows, and a float sum of ``k`` terms of one sign rounds
#: by up to ``(k - 1) / 2`` units of ``eps`` of the sum; nothing in the
#: floor counts that.  Where the terms are nearly equal (a uniform
#: field, or any field beside a much larger common value) and the sum
#: is taken in order, the rounding is systematic, not a random walk: it
#: grows like ``k``, and the pass then has a fixed point of its own that
#: far from the exact one.
#:
#: The measurement: a pair of relays that declare one evaluation each
#: (``m`` values, each fed by the sum of its own ``k`` values, each of
#: which reads its one back), loop gains 0.9 and 0.99, both schedules,
#: stalled at the float floor, CPU, jax 0.10.2, 0.11.0 and 0.11.2.  The
#: smallest ``spectral_error_bound`` over the true distance among the
#: reports that set ``spectral_usable``, by row length.
#:
#: **The scatter layout** (``StaticSparseMapping(layout="scatter")``,
#: which ``sparse_nearest_neighbor_mapping(mode="conservative",
#: transpose="scatter")`` builds) adds a target's row up one entry after
#: another: a scatter-add, the in-order sum on the CPU.  One row; six
#: random fields, four uniform ones and a ramp, 44 runs per cell;
#: float32; the same digits on the three versions:
#:
#: ===============  ====  ====  ====  ====  ====  =====  =====  ======  ======
#: row              3     10    30    100   300   1000   3000   1e4     3e4
#: ===============  ====  ====  ====  ====  ====  =====  =====  ======  ======
#: ``"interface"``  6.2   3.9   2.07  0.68  0.24  0.069  0.023  0.0068  0.0022
#: ``"mixed"``      6.3   4.9   2.09  0.66  0.22  0.069  0.023  0.0067  0.0022
#: ===============  ====  ====  ====  ====  ====  =====  =====  ======  ======
#:
#: At the float64 floor (``rtol=1e-16`` under x64) the same pair read
#: 8.0, 5.0, 2.08, 0.63, 0.22, 0.068, 0.023, 0.0065 and 0.0022 under
#: ``"interface"`` and 9.1, 5.0, 2.09, 0.63, 0.22, 0.069, 0.023, 0.0065
#: and 0.0022 under ``"mixed"``, on the three versions alike.
#:
#: **The gather layout and the dense kinds** are reduced by XLA in an
#: order of its choosing, which depends on the jax version, the dtype
#: and the operator's shape.  Short rows (one, three and sixteen rows of
#: ``k`` entries in float32, one and three at the float64 floor; two
#: random fields, two uniform ones and a ramp; ``"interface"``; the same
#: digits on the three versions):
#:
#: ===================  ====  ====  ====  ====
#: row                  3     10    30    100
#: ===================  ====  ====  ====  ====
#: gather, float32      6.5   4.9   3.4   2.8
#: dense, float32       6.5   4.9   2.24  2.8
#: gather, float64      8.9   5.0   3.05  2.6
#: dense, float64       8.9   5.0   2.08  2.6
#: ===================  ====  ====  ====  ====
#:
#: Long rows, flags set throughout:
#:
#: * one row behind a uniform field, both forms alike: 1.8x the
#:   distance at 300 entries, 1.2x at 1000, 1.09x at 3000, and 3.7x or
#:   more at 1e4 and 3e4 (5x and more behind the other fields);
#: * **the gather layout on jax 0.10.2 in float32: its rows of 1e4 and
#:   3e4 entries are summed in order and read exactly as the scatter
#:   layout's do, 0.0068 and 0.0022 of the distance** (not in float64,
#:   and not on 0.11.0 or 0.11.2);
#: * **the dense kind with three rows of 3000 entries: 0.18 of the
#:   distance in float32 on the three versions, 0.09 at the float64
#:   floor** (0.84 with three rows of 300 there); the gather layout of
#:   the same operator held by 4x and more.
#:
#: So the limit is the largest power of ten at which every one of those
#: runs held by a factor of two on the three jax versions, **in every
#: kind**: the scatter layout sets it (30 entries hold by 2.07, 100 do
#: not hold).  The reduced forms held by two up to 100 entries, but
#: which rows XLA sums in order is its own to choose, per version, dtype
#: and shape, so they are counted at the scatter layout's limit and not
#: at one of their own.  Other fields, another backend or another jax
#: may move it.
#:
#: **Fields whose terms cancel are another matter** (MADD-ANO-247, open:
#: a delivered value is no finer than the terms it was computed from).
#: Behind a field that changes sign within a row, the row's rounding is
#: larger by the cancellation ``sum |t| / |sum t|`` whatever the row's
#: length and whichever way it is summed.  Measured on the same pairs
#: with ``b = 0`` and fields alternating in sign: with a cancellation of
#: 3 or less every kind held by two as above (3.7x at 10 entries; the
#: scatter layout 2.14x at 30); with a cancellation of 25, at 10 entries
#: in every kind, 1.26x in float32 and **0.85x at the float64 floor,
#: with the flag set**, and 0.80x and 0.72x behind a dense matrix of
#: three rows of ten (which is thirty wide, and so over the limit).  The
#: limit is not taken on those runs: no row length answers for a field's
#: signs, and a row within the limit behind such a field keeps its flag.
#:
#: **What a row is** for each kind is in :func:`_longest_row`: counted
#: whatever the weights are, a dense matrix by its width.
#:
#: **What it does not cover.**  A geometry-dependent mapping's rows
#: (``multilinear_grid`` from points to a grid: a grid node adds up as
#: many entries as there are points within one spacing of it, by the
#: same scatter-add) are known only in the step and are not counted:
#: with 8, 300 and 3000 markers in one cell behind a uniform field the
#: bound read 13.8x, 3.8x and 1.3x the distance under ``"mixed"`` (it
#: held, by less than two at 3000).  That group solves the markers'
#: positions, and in 0.4.0 such a group has no usable flag whatever its
#: rows (:func:`_geometry_flags`); one whose positions are constants of
#: the pass keeps its flags behind the same uncounted rows, which was
#: not measured.  MADD-ANO-257 records all of it.
MAPPED_ROW_FLOOR_LIMIT = 10  # units: entries of one row of a static mapping

#: Why ``spectral_usable`` and ``gradient_bound_usable`` are withdrawn
#: from a report whose residual does not stand clear of the float floor
#: a long row of a static mapping would give it
#: (:func:`_mapped_row_reason`).  The numbers stay as computed.  The way
#: out it names is the one that was measured to hold on every jax
#: version, for every kind.  It names no other kind or layout: a row of
#: the same length is counted alike in each (see the constant above).
_MAPPED_ROW_REASON = (
    "the group's internal edge {key} carries {what} whose longest row adds up {row} "
    "entries (the limit is {limit}), and the residual ({residual:.3g}) is not above the "
    "float floor ({floor:.3g}) times the row's length. The sum of a row rounds by more "
    "than the fixed number of ulps per evaluation the float floor counts once the row "
    "is longer than the limit, so spectral_error_bound (and the gradient bound built on "
    "it) can read below the true distance here (MADD-ANO-257). spectral_usable and "
    "gradient_bound_usable are therefore False; every number is reported as computed. "
    "The way out: a wider dtype at the same tolerance, so that the residual stands clear "
    "of the floor, for every floating field of the group's members (one no loop passes "
    "through included) and for what its internal edges deliver: the floor is counted at "
    "the coarsest of them.{others}"
)


def _longest_row(mapping) -> tuple:
    """``(what, entries)`` of *mapping*, a static one: how it is applied,
    in the words the report's reason uses, and the most entries it adds
    up into one delivered value.

    Counted on what is frozen, on the host, **whatever the weights are**:
    they are a parameter a step may be handed, zeros included.

    * A static sparse mapping in the scatter layout: the most valid
      slots that name one target (the longest row of the operator, as
      opposed to the longest row of its storage, which is a source's).
    * In the gather layout: the most valid slots of one row (a padded
      slot reads a zero whatever its weight is).
    * A dense matrix (``StaticLinearMapping``, which every dense kind
      builds): **the matrix's width, not its non-zeros.**  ``H @ field``
      adds up one product per source entry, and which of them are zero
      is for the weights to say, not the mapping: ``H`` is the mapping's
      one parameter, and a step handed another matrix of the same shape
      sums as many non-zero terms as that one holds.  So a selection
      matrix (one non-zero a row, whose sum is exact) is counted at its
      width too; that is part of the guard's price.
    * A static mapping of another class (a registered kind's own): the
      entries of its source side, the most one delivered value can add
      up.  Its ``apply`` is its author's; nothing was measured on it.
    """
    kind = getattr(mapping, "kind", None)
    named = "" if kind is None else f" ({kind})"
    form = _interface_plan._mapping_form(mapping)
    if form == _interface_plan.STATIC_DENSE:
        return (f"a dense matrix mapping{named}, counted at the matrix's width,",
                int(mapping.n_source))
    if form != _interface_plan.STATIC_SPARSE:
        source_lead, _target_lead = _interface_plan._mapping_leads(mapping)
        return (f"a static mapping of class {type(mapping).__name__}{named}, counted at "
                f"the entries of its source side,",
                _interface_plan._entries(source_lead))
    what = f"a static sparse mapping{named} in the {mapping.layout} layout"
    index = np.asarray(mapping.indices)
    counts = None if mapping.counts is None else np.asarray(mapping.counts)
    if mapping.layout == "gather":
        if counts is None:
            return what, int(index.shape[1]) if index.shape[0] else 0
        return what, int(counts.max()) if counts.size else 0
    if counts is not None:
        index = index[np.arange(index.shape[1])[None, :] < counts[:, None]]
    if index.size == 0:
        return what, 0
    return what, int(np.bincount(index.ravel().astype(np.int64)).max())


def _mapped_rows(interface_edges) -> tuple:
    """``((edge key, what, longest row), ...)`` over a group's internal
    edges that carry a static mapping, of every kind: a dense matrix, a
    static sparse mapping in either layout, a registered kind's own
    static class (:func:`_longest_row` has what a row is for each).

    Read by ``compile()`` for the report's guard on the float floor
    (:func:`_mapped_row_reason`).  *interface_edges* is the group's plan
    or its bare internal edges.  Every norm: the row's rounding is in
    what the edge delivers to its target, whichever fields or readings
    the residual is taken on.  Not a geometry-dependent mapping: its
    rows are not known before the step (a ``multilinear_grid`` scatter's
    are the markers in a cell's support), and it is not counted here
    (see :data:`MAPPED_ROW_FLOOR_LIMIT`).
    """
    static = (_interface_plan.STATIC_DENSE, _interface_plan.STATIC_SPARSE,
              _interface_plan.STATIC_OTHER)
    return tuple((record.key, *_longest_row(record.mapping))
                 for record in _interface_plan.interface_records(interface_edges)
                 if record.mapping_form in static)


def _mapped_row_reason(rows, residual: float, floor: float) -> Optional[str]:
    """Why a report withdraws the flags that rest on its float floor on
    account of a long row of a static mapping; ``None`` where it does not.

    *rows* is :func:`_mapped_rows` of the group the step was built from,
    *residual* and *floor* the report's own.  Withdrawn where both hold:

    * a row is longer than :data:`MAPPED_ROW_FLOOR_LIMIT` entries; and
    * the residual is at or below ``floor * row``: the report is at the
      float floor that row could give it.  An in-order sum of ``row``
      terms of one sign rounds by at most ``(row - 1) / 2`` ``eps``,
      which is ``(row - 1) / 8`` of the ``PRECISION_FLOOR_ULPS`` one
      evaluation is counted at, so a flag is kept only where that worst
      case is under an eighth of the residual.  Any other order of the
      sum rounds by no more.

    ``precision_limited`` (the residual at or below the floor as
    counted) is the nearer part of the second condition and is not
    enough: with a tolerance above the counted floor the scatter pair
    accepts with its residual up to 1000 floors and its bound at 0.007
    of the distance behind 3e4 entries (float32; 35 of the 83 such
    reports measured below their distance on each jax version were not
    ``precision_limited``, 56 of 112 at the float64 tolerances).
    Measured with the rule (float32, tolerances from 1e-6 to 1e-1, rows
    of 100, 1000 and 3e4): every flag that is kept reads at least 0.9925
    of the distance on a pair (two evaluations a pass), which is where
    the bound's own estimate reads in a residual-dominated report under
    ``"mixed"`` whatever the mapping.  That is not the rule's floor: on a
    group of ONE member with an edge to itself (one evaluation a pass,
    so the threshold is half the pair's) a kept flag read 0.982 of the
    distance by construction (a report just above the threshold, a
    uniform field, gain 0.999, 1000 entries a row, ``"interface"``; 14
    of 176 kept flags under 0.9925).  At the threshold the bound's
    allowance for rounding is ``1 / row`` of the residual while the
    row's rounding is up to ``1 / (8 evaluations)`` of it.

    It only withdraws, and reads nothing but the report's own two
    numbers: a group with no static mapping on an internal edge, a row
    within the limit, and a residual that stands clear all keep their
    flags, and no number of the report moves.
    """
    long_rows = [entry for entry in rows if entry[2] > MAPPED_ROW_FLOOR_LIMIT]
    if not long_rows or not floor > 0.0:
        return None
    key, what, row = max(long_rows, key=lambda entry: entry[2])
    if not residual <= floor * row:
        return None
    others = [k for k, _what, _row in long_rows if k != key]
    return _MAPPED_ROW_REASON.format(
        key=key, what=what, row=row, limit=MAPPED_ROW_FLOOR_LIMIT, residual=residual,
        floor=floor,
        others=(f" Other edges with a row over the limit: {others}." if others else ""))


def _joined_reasons(earlier: Optional[str], reason: str) -> str:
    """*reason*, after the reason another rule already gave the report.

    A rule may leave a flag of its own ``False`` with its reason while
    ``spectral_usable`` stands (a group with a geometry-dependent
    mapping whose positions are constants of the pass and whose gradient
    bound is not finite).  A later rule that withdraws
    ``spectral_usable`` there keeps that reason and adds its own.
    """
    return reason if not earlier else f"{earlier} Also: {reason}"


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
    edge whose reading *is* its source field -- and every other field a
    part of a reading holds entry for entry, which the group's plan
    answers (``InterfaceEdge.measured_whole``).  Every
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
    edge reads.  A geometry-dependent mapping read at its source
    (experimental) is read at its inputs: its source field, and the
    positions a source anchor takes from the iterate, each over a
    constant length -- both are measured whole and kept.  A geometry its
    target holds is the pre-step state, not a reading, and that field is
    recomputed like any other.

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
    whole = {field
             for record in _interface_plan.interface_records(interface_edges)
             for field in record.measured_whole}
    missed = {nn: tuple(f for f in floats[nn] if (nn, f) not in whole)
              for nn in schedule}
    return {nn: fields for nn, fields in missed.items() if fields}


#: Every ``_meta`` slot a coupling group can own, as the suffix after
#: ``coupling_<group key>_``.  Read by :func:`_refuse_colliding_group_keys`.
_GROUP_META_SUFFIXES = (
    "iterations", "total_iterations", "residual", "amplification", "rho_spectral",
    "spectral_residual", "spectral_amplification",
    "gradient_relative_error_bound", "pass_evaluations", "reading_floor",
    "geometry_gap", "geometry_plane_limit", "geometry_plane_margin", "V", "W", "pred_count",
    "pred_0", "pred_1", "pred_2",
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
    "under convergence_norm={norm!r} (the solve's own criterion reads it there, but the "
    "analysis of that reading behind the bounds is not in 0.4.0; a diagnostic run of the "
    "group under 'mixed' or 'l2', which measure the members' state, the geometry included, "
    "reports them)"
)
#: What the report of a group withheld on account of its norm still says
#: of the residual's float floor (:func:`_floor_reading_of_a_norm_withheld_report`):
#: appended to the reason, so that it names every number the entry reports.
_GEOMETRY_NORM_FLOOR_REPORTED = (
    " The residual's float floor is reported: residual_precision_floor is the float "
    "resolution of the residual at the state this step returned, in the residual's units "
    "(tolerances), by the rule of every group's report, and precision_limited says whether "
    "the residual is at or below it. Where it is, the residual is rounding and not motion: "
    "converged=True does not say the readings have settled to the tolerance, and a group at "
    "max_iterations may be held there by rounding alone. Positions a reading rests on enter "
    "that floor by their rounding, at the state of every step: eps of their dtype times the "
    "larger of a position's distance from the coordinates' zero and of its lattice "
    "coordinate (its distance from the grid's first point, which the mapping forms in the "
    "positions' dtype), in grid spacings. This is the run-time reading of what compile() "
    "warns of once, on the state it sees, for each part by itself; the floor here is pooled, "
    "as the residual is, over every entry the norm reads, and it takes the pass's structural "
    "evaluation count whether diagnostics are on or not."
)
#: The same where the step ran and its state is not finite: there is no
#: resolution to report, and neither "not until the group steps" nor "is
#: reported" would be true of it.
_GEOMETRY_NORM_FLOOR_NOT_FINITE = (
    " The residual's float floor could not be measured, because the state this step "
    "returned is not finite where the norm reads it (residual is {residual}): "
    "residual_precision_floor is NaN and precision_limited is False, which says nothing of "
    "rounding here. converged=False is the step's own verdict on that state."
)
#: The same where the floor itself could not be reported (a checkpoint
#: saved after a state write; a floor only the step could have measured).
_GEOMETRY_NORM_FLOOR_WITHHELD = (
    " The residual's float floor (residual_precision_floor, precision_limited) is not "
    "reported either: {why}"
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


def _geometry_refusal_code(group, nodes, plan) -> Optional[str]:
    """The reason code of :func:`_geometry_diagnostics_refusal` for the
    same group: the cause its sentence names, by the same precedence (a
    mapping kind the diagnostics do not read, else the interface norm,
    else sub-cycling); ``None`` where it gives no reason.  Where two of
    the three hold the report names the first, in words and in code
    alike."""
    geometry = plan.resolved_geometry_edges()
    if not geometry:
        return None
    if any(r.mapping_kind != _DIAGNOSED_GEOMETRY_KIND for r in geometry):
        return reason_codes.GEOMETRY_KIND_NOT_DIAGNOSED
    if group.convergence_norm not in _DIAGNOSED_GEOMETRY_NORMS:
        return reason_codes.GEOMETRY_NORM_NOT_DIAGNOSED
    if _group_dividers(group, nodes):
        return reason_codes.GEOMETRY_SUBCYCLED
    return None


def _withheld_on_account_of_its_norm(group, reason: Optional[str]) -> bool:
    """Is *reason* (a group's ``_geometry_diagnostics_refusal``, or
    ``None``) the one of a group whose bounds are withheld because its
    norm is ``"interface"`` -- the group whose report keeps the float
    floor?  The one test of that, for the two readers below."""
    return reason is not None and _GEOMETRY_NORM_WHY.format(
        norm=group.convergence_norm) in reason


def _reports_the_structural_count(group, reason: Optional[str]) -> bool:
    """Does *group*'s report count its float floor with the pass's
    **structural** evaluation count even where the step measured one
    (``diagnostics=True``)?  ``True`` for a group withheld on account of
    the interface norm (*reason*: its ``_geometry_diagnostics_refusal``).

    Experimental.  The measured count
    (``_coupled_block._measured_pass_evaluations``) weights every read
    by its relative gain, and the gain of a read of a member that holds
    **positions** is taken against the positions' own magnitude: it
    grows with their distance from the coordinates' zero (2.0, 15.5,
    147, 468 and 3269 at 0, 100, 1e3, 5e3 and 2e4 spacings on one pair).
    Under the interface norm the floor already counts a position in
    spacings (``eps`` times the magnitude it is rounded at), so the
    distance was counted twice, and for float64 positions at the float32
    values' ``eps``: float64 positions beside float32 values read a
    floor of 0.078 without diagnostics and 0.60, 5.7, 18 and 127 with
    them (up to 1189 where the grid node holds the positions), and
    ``precision_limited=True`` on groups 0.1 to 0.6 tolerances from
    their fixed point, on a state that is the same to the bit either
    way.  No other number of such a group reads the measured count in
    0.4.0 (its bounds are not reported), so its floor takes the count
    ``compile()``'s advisory takes, and **the same floor with
    diagnostics on or off**.

    What is given up: the measured count also sees a member whose terms
    cancel (``3u - 2v`` at ``u ~ v`` reads 5), which the structural
    count does not; that is true of this group's report without
    diagnostics as well, where the count was always the structural one.
    Read by ``GraphManager.coupling_diagnostics`` in one place.
    """
    return _withheld_on_account_of_its_norm(group, reason)


def _floor_reading_of_a_norm_withheld_report(group, reason: str, report, floor: float) -> dict:
    """What the report of a group whose bounds are withheld **on account
    of its norm** keeps of the residual's float floor; ``{}`` for a
    report withheld for any other reason.

    Experimental.  A group under ``convergence_norm="interface"`` that
    resolves a ``multilinear_grid`` mapping solves, and its bounds are
    not reported in 0.4.0 (:func:`_geometry_diagnostics_refusal`).  The
    float floor of its residual is not one of them: it is
    :func:`~maddening.core.coupling.acceleration.residual_precision_floor`
    of the state the step returned, the function and the rule of every
    other group's report, and under this norm it is the one place the
    rounding of stored positions is counted at run time (``eps`` times
    the larger of a position's distance from the coordinates' zero and
    of its lattice coordinate, in spacings: ``_interface_plan._rounded_at``;
    a reading that rests on the position carries it).  ``compile()`` asks that question
    once, of the state it is called with
    (:func:`_unresolved_position_warnings`); markers that drift, a state
    write and a loaded checkpoint are all later than that, and a float32
    pair whose markers had drifted 957 to 6659 spacings read
    ``converged=True``, ``residual=0.0`` and ``precision_limited=False``
    with its positions 3 to 20 tolerances from the fixed point while the
    step's own recorded floor was 117 to 557 tolerances.  So the entry
    keeps:

    * ``"precision_limited"``: as computed for any group -- the floor is
      positive and the (finite) residual is at or below it;
    * ``"residual_precision_floor"``: that floor, in the residual's
      units, NaN where it could not be measured (the key is present
      either way, and ``"precision_limited"`` is then ``False``);
    * ``"not_usable_reason"``: *reason*, and what the entry reports of
      the floor (or why it does not).

    **Three cases, told apart.**  The floor was measured: it is
    reported.  The step ran and returned a state that is **not finite**
    where the norm reads it (a non-finite position or value: the
    residual is ``inf`` and the floor is not a number): the reason says
    that, whichever way the floor would have been taken -- before, such
    an entry said the flags "are not reported until the group steps"
    where the floor is the step's to record (the group had stepped), and
    that the floor "is reported" beside a NaN where it is measured on
    the returned state.  Otherwise the floor was not measured (a
    checkpoint saved after a state write, or a floor only the step could
    have recorded and the state carries none): the reason gives the
    entry's own account of that.

    **The floor is pooled** over every entry the norm reads, as the
    residual is; ``compile()``'s advisory quotes what the positions put
    into one part by itself.  With entries beside that part that are
    finer the two differ by up to the root of the part's share of the
    entries (95.7 quoted and 0.61 reported, for four markers a thousand
    spacings out pooled with two plain edges of 1e5 entries); the reason
    says so.  And it is counted with the pass's **structural**
    evaluation count (:func:`_reports_the_structural_count`).

    *report* is the group's entry as every group's is built, before the
    bounds are withheld: its ``precision_limited`` and ``residual``, and
    a ``not_usable_reason`` where the floor itself was not reported (a
    checkpoint saved after a state write, or a floor only the step could
    have measured).  *floor* is the floor that entry was built with.
    Read by ``GraphManager.coupling_diagnostics`` in one place.
    """
    if not _withheld_on_account_of_its_norm(group, reason):
        return {}
    unmeasured = report.get("not_usable_reason")
    residual = float(report.get("residual", float("nan")))
    if not math.isfinite(float(floor)) and not math.isfinite(residual):
        return {
            "precision_limited": False,
            "residual_precision_floor": float("nan"),
            "not_usable_reason": reason + _GEOMETRY_NORM_FLOOR_NOT_FINITE.format(
                residual=residual),
        }
    if unmeasured is not None:
        return {
            "precision_limited": False,
            "residual_precision_floor": float("nan"),
            "not_usable_reason": reason + _GEOMETRY_NORM_FLOOR_WITHHELD.format(why=unmeasured),
        }
    return {
        "precision_limited": bool(report["precision_limited"]),
        "residual_precision_floor": float(floor),
        "not_usable_reason": reason + _GEOMETRY_NORM_FLOOR_REPORTED,
    }


def _geometry_self_check_reason(keys, gap: float, allowed: float) -> str:
    """The reason of a report whose step failed its geometry self-check:
    a gap over *allowed* (the product is not the derivative, or the
    finite difference is rounding: the two read alike), or a gap that is
    not a number (the two could not be compared)."""
    why = _GEOMETRY_SELF_CHECK_WHY if gap == gap else _GEOMETRY_SELF_CHECK_UNEVALUATED_WHY
    return _GEOMETRY_DIAGNOSTICS_REASON.format(
        keys=list(keys), why=why.format(gap=gap, allowed=allowed))


def _geometry_self_check_code(gap: float) -> str:
    """The reason code of :func:`_geometry_self_check_reason`: a gap over
    its tolerance, or one that is not a number (nothing was compared)."""
    return (reason_codes.GEOMETRY_SELF_CHECK_FAILED if gap == gap
            else reason_codes.GEOMETRY_SELF_CHECK_NOT_EVALUATED)


#: The one reason of a group that solves the positions of a
#: geometry-dependent mapping (:func:`_geometry_flags`): neither flag is
#: set for it in 0.4.0, on any step.
_GEOMETRY_SOLVED_REASON = (
    "the group solves position(s) {solved} read by geometry-dependent mapping(s) on "
    "edge(s) {keys} (its pass reads them from the iterate, or builds them and reads them "
    "in the same pass), and 0.4.0 does not certify a bound for such a group: "
    "spectral_usable and gradient_bound_usable are False on every step, because a lattice "
    "plane of the mapping's grid within reach of the solve makes the pass another "
    "polynomial, with another fixed point or none, and three independent audits each found "
    "a flag set beside a wrong number there (MADD-ANO-252). The numbers are reported as "
    "computed, uncertified. The flags are available where every position is fixed during "
    "the pass (a target-anchored geometry read by update, or positions held by a node "
    "outside the group); no convergence_norm restores them for this group in 0.4.0."
)
#: The causes of a ``False`` flag of a group whose pass resolves a
#: geometry-dependent mapping at positions that are constants of the
#: pass (:func:`_geometry_flags`): a smooth group's, each one clause of
#: the report's ``not_usable_reason``.  None names a lattice plane: no
#: plane can come between the iterate and the fixed point of such a pass.
_CAUSE_BOUND_NOT_FINITE = (
    "spectral_error_bound is {bound} (the linearised pass does not contract at the "
    "returned iterate, or the estimate could not be evaluated)"
)
_CAUSE_NOT_SETTLED = (
    "the spectral estimate did not settle: its Arnoldi residual, {residual:.3g}, is over "
    "{fraction:g} of 1 - rho_spectral ({allowed:.3g}), which is what a pass with more "
    "independent interface scalars than the estimate's {steps} Krylov steps gives; no "
    "tolerance changes that"
)
_CAUSE_FLOOR = (
    "the residual is at its float floor (precision_limited) and not every member declares "
    "how many evaluations its update makes (update_evaluations), so the floor the bound "
    "rests on is not checked; the residual is rounding there, and a tighter tolerance does "
    "not lower it"
)
#: The step's own record of which positions its pass read is missing:
#: the state the report reads was not written by this build's step.
_CAUSE_NOT_RECORDED = (
    "the step did not record that every position of the group's geometry-dependent "
    "mapping(s) on edge(s) {keys} was fixed during its pass (geometry_plane_limit and "
    "geometry_plane_margin, which this build's step writes as inf for such a group, read "
    "{limit:g} and {margin:g}: the state was not written by this build's step), so nothing "
    "says which positions that step read; the flags return when the group steps"
)
_CAUSE_GRADIENT_NAN = (
    "gradient_relative_error_bound was not computed (NaN): the Jacobian's range was not "
    "captured by the bound's {steps} directions (more independent interface scalars than "
    "that), or the fixed point responds to no constant the bound probes"
)
_CAUSE_GRADIENT_INF = (
    "gradient_relative_error_bound is inf: its Newton-Kantorovich check did not pass (the "
    "Jacobian changes too much across the Newton step), or nothing contracts"
)


#: The clauses of a report whose step computed no estimate at all
#: (``rho_spectral`` is NaN), one per cause (:func:`_flag_causes`).
_CAUSE_SOLVER_NOT_IFT = (
    "the group's solver is {solver!r}, which has no linearisation of the pass: the spectral "
    "estimate and the gradient bound exist only under solver='ift' with diagnostics=True"
)
_CAUSE_DIAGNOSTICS_OFF = (
    "the group was built with diagnostics=False, so its steps do not compute the spectral "
    "estimate (rho_spectral is NaN) or the gradient bound; build the group with "
    "diagnostics=True for a run that reports them"
)
_CAUSE_SINGLE_PASS = (
    "max_iterations=1 runs one staggered pass and solves no fixed point, so there is no "
    "spectrum of a solve to estimate; allow the group a second pass"
)
_CAUSE_STATE_NOT_FINITE = (
    "the state this step returned is not finite (residual is {residual:g}), and nothing is "
    "computed at such a state: the iteration diverged"
)
_CAUSE_ESTIMATE_NOT_RECORDED = (
    "the state holds no spectral estimate for this group (rho_spectral is absent or NaN) "
    "although the group asks for one: it was not written by a step of this graph as "
    "compiled (a state set by hand, or a checkpoint of a graph built without diagnostics); "
    "the estimates return when the group steps"
)
#: An estimate that did not settle in a group whose pass ``compile()``
#: counted within the estimate's Krylov steps (:func:`_pass_width`): the
#: steps span such a pass, so what is unsettled is the estimate's check
#: of itself, and :data:`_CAUSE_NOT_SETTLED` (a pass wider than the
#: steps) would say the wrong thing of it.
_CAUSE_SELF_CHECK = (
    "the spectral estimate did not settle: its Arnoldi residual, {residual:.3g}, is over "
    "{fraction:g} of 1 - rho_spectral ({allowed:.3g}), although one pass depends on the "
    "previous one through at most {width} scalar(s) in a state of {entries} entries, which "
    "the estimate's {steps} Krylov steps span: the estimate's check of itself did not pass "
    "(one more product moved the radius by more than that margin, or rounding could have, "
    "or a direction was discarded as rounding, or the residual was not absorbed into the "
    "space), so rho_spectral and the bound are not to be trusted here; a wider dtype is "
    "the usual way out"
)
#: What an unsettled estimate adds where its margin alone makes the bound
#: ``inf``: said after the cause, so that such a report is not told its
#: pass does not contract.
_CAUSE_BOUND_INF_BY_MARGIN = (
    "spectral_error_bound is inf because twice that residual added to rho_spectral "
    "({rho:.3g}) is not below 1: the pass may well contract, and the estimate cannot say"
)


def _pass_width(group, nodes, plan, state) -> tuple:
    """``(width, entries)``: how many scalars one pass of *group* depends
    on the previous one through, at most, and how many entries the
    group's floating state holds.  Structural, read by ``compile()`` on
    the host; the report tells an estimate that did not settle in a pass
    **wider than its Krylov steps** from one that did not although the
    steps span the pass (:func:`_unsettled_cause`).

    **What is counted.**  A pass ``F`` maps the group's state to the
    next iterate, and each member's update starts from its pre-step
    state, a constant of the pass.  So ``F`` reads the previous iterate
    only through the internal edges it takes *from the previous pass*:
    every internal edge under Jacobi and in a group that sub-cycles (an
    interpolated read takes both ends); under Gauss-Seidel an edge whose
    source is swept at or after its target (*plan*'s order: the order
    the members were added in, a self-edge included).  The rank of
    ``dF/dx`` is then at most each of:

    * the entries of the distinct source fields those edges read (and of
      the geometry field of a source-anchored mapping on one of them);
    * the floating entries of the members that read one: every other
      member reads only this pass's values, which are functions of those
      members' new states;
    * the entries of the whole state.

    *width* is the smallest of the three.  **An upper bound, not the
    rank**: a member that uses only a combination of what it reads (a
    mean, a selection inside ``update``), and a mapping that delivers
    fewer entries than its source holds, have a lower rank than is
    counted here.  A group with a boundary-flux edge between members is
    counted at its whole state (a flux is computed from its producer's
    state and inputs, and neither count above holds for it).

    The estimate's Krylov space holds its start vector and the range of
    the pass's Jacobian, so ``SPECTRAL_KRYLOV_STEPS`` steps span a pass
    of width ``SPECTRAL_KRYLOV_STEPS - 1`` or less, and any pass of a
    state of ``SPECTRAL_KRYLOV_STEPS`` entries or fewer (the space is
    then the whole state).  Measured on linear pairs in float64
    (Gauss-Seidel, a member of 6, 7 and 8 scalars beside one of 300
    entries: settled, settled, not; Jacobi with 4 + 4, 3 + 5 and 5 + 5
    entries: settled, settled, not; ``"fixed"`` with and without
    relaxation and ``"aitken"`` alike; jax 0.11.0).
    """
    def entries(name) -> int:
        return sum(int(np.prod(np.shape(value), dtype=np.int64))
                   for value in (state.get(name) or {}).values()
                   if _is_float_leaf(value))

    members = sorted(group.nodes)
    whole = sum(entries(name) for name in members)
    position = {name: i for i, name in enumerate(plan.order)}
    last = len(position)
    every = group.iteration_mode == "jacobi" or bool(_group_dividers(group, nodes))
    readers: set = set()
    fields: set = set()
    counted = not plan.flux_members
    for record in plan.internal:
        source, target = record.source[0], record.target[0]
        previous = every or position.get(source, last) >= position.get(target, last)
        if not previous and record.read_from_state:
            continue
        # (An edge whose source is not a state field is taken from the
        # previous pass whatever the order: counted at the whole state.)
        readers.add(target)
        if not record.read_from_state:
            counted = False
            continue
        fields.add(record.source)
        if record.anchor is not None and record.anchor[0] == "source":
            fields.add((source, record.anchor[1]))
    if not counted:
        return whole, whole

    def size(name, field) -> int:
        value = (state.get(name) or {}).get(field)
        if value is None or not _is_float_leaf(value):
            return 0
        return int(np.prod(np.shape(value), dtype=np.int64))

    through_fields = sum(size(name, field) for name, field in fields)
    through_readers = sum(entries(name) for name in readers)
    return min(through_fields, through_readers, whole), whole


def _is_float_leaf(value) -> bool:
    dtype = getattr(value, "dtype", None)
    return dtype is not None and bool(jnp.issubdtype(dtype, jnp.floating))


def _unsettled_cause(width, *, rho: float, arnoldi_residual: float, fraction: float,
                     steps: int) -> tuple:
    """``(code, clause)`` of a spectral estimate that did not settle.

    *width* is :func:`_pass_width` of the group the step was built from.
    Where the pass is wider than the estimate's steps span, the limit
    explains the unsettled estimate, which is expected of such a group
    (``interface_too_wide``); where the steps span the pass, the space
    closed and what is unsettled is the estimate's check of itself
    (``spectral_self_check_failed``).  The step stores one number for
    the two (the larger of what the space missed and of what its check
    measured), so nothing but the count tells them apart: in a pass
    counted as too wide, rounding may have had its part as well, and the
    report does not say.
    """
    scalars, entries = width
    allowed = fraction * (1.0 - rho)
    if entries <= steps or scalars <= steps - 1:
        return (reason_codes.SPECTRAL_SELF_CHECK_FAILED, _CAUSE_SELF_CHECK.format(
            residual=arnoldi_residual, fraction=fraction, allowed=allowed, width=scalars,
            entries=entries, steps=steps))
    return (reason_codes.INTERFACE_TOO_WIDE, _CAUSE_NOT_SETTLED.format(
        residual=arnoldi_residual, fraction=fraction, allowed=allowed, steps=steps))


#: A width that reads as wider than any number of steps: what a caller
#: that holds no count passes (:func:`_geometry_flags`).
_UNCOUNTED_WIDTH = (math.inf, math.inf)


def _flag_causes(group, *, bound: float, rho: float, arnoldi_residual: float,
                 amplification: float, settled: bool, residual: float,
                 precision_limited: bool, declared: bool, gradient_bound: float, width,
                 fraction: float, steps: int, floor_measured: bool = True) -> tuple:
    """``(spectral, own)``: every cause of a ``False`` ``spectral_usable``
    and the gradient flag's own, each a ``(code, clause)`` pair, from the
    numbers a report holds.  The one place a cause is given its code and
    its words, for a plain group and for one whose geometry is a constant
    of its pass alike.

    ``spectral_usable`` is ``isfinite(bound) and settled and (declared or
    not precision_limited)``, and *spectral* is empty exactly where that
    holds (the branches below are that expression's, taken apart):

    * no estimate (*rho* is NaN): why the step computed none, from
      *group* -- its solver, its diagnostics switch, a budget of one
      pass, a state that is not finite, or none of them (the state was
      not written by this graph's step);
    * *bound* is NaN beside an estimate: it could not be evaluated;
    * ``rho >= 1`` or a resolvent that is not finite: the linearised
      pass does not contract;
    * the estimate did not settle: :func:`_unsettled_cause`, and where
      its margin makes the bound ``inf`` the clause says that too;
    * the residual at its float floor with an evaluation count that is
      not declared.

    *own* (only where an estimate was computed) tells a gradient bound
    that was not computed (NaN) from one that did not certify (``inf``).
    ``gradient_bound_usable`` is ``spectral_usable and
    isfinite(gradient_bound)``.

    With *floor_measured* ``False`` (a report whose float floor another
    rule withholds) the two causes that read the floor are left to that
    rule: a bound that is NaN, and the residual at its floor.
    """
    spectral: list = []
    own: list = []
    if rho != rho:
        if group is not None and group.solver != "ift":
            spectral.append((reason_codes.SOLVER_NOT_IFT,
                             _CAUSE_SOLVER_NOT_IFT.format(solver=group.solver)))
        elif group is not None and not group.diagnostics:
            spectral.append((reason_codes.DIAGNOSTICS_OFF, _CAUSE_DIAGNOSTICS_OFF))
        elif not math.isfinite(residual):
            spectral.append((reason_codes.STATE_NOT_FINITE,
                             _CAUSE_STATE_NOT_FINITE.format(residual=residual)))
        elif group is not None and group.max_iterations <= 1:
            spectral.append((reason_codes.SINGLE_PASS, _CAUSE_SINGLE_PASS))
        else:
            spectral.append((reason_codes.ESTIMATE_NOT_RECORDED,
                             _CAUSE_ESTIMATE_NOT_RECORDED))
    else:
        if bound != bound:
            if floor_measured:
                spectral.append((reason_codes.BOUND_NOT_EVALUATED,
                                 _CAUSE_BOUND_NOT_FINITE.format(bound=f"{bound:g}")))
        elif not rho < 1.0 or not math.isfinite(amplification):
            spectral.append((reason_codes.NOT_CONTRACTING,
                             _CAUSE_BOUND_NOT_FINITE.format(bound=f"{bound:g}")))
        elif not settled:
            code, clause = _unsettled_cause(
                width, rho=rho, arnoldi_residual=arnoldi_residual, fraction=fraction,
                steps=steps)
            spectral.append((code, clause))
            if not math.isfinite(bound):
                spectral.append((code, _CAUSE_BOUND_INF_BY_MARGIN.format(rho=rho)))
        elif not math.isfinite(bound) and floor_measured:
            spectral.append((reason_codes.BOUND_NOT_EVALUATED,
                             _CAUSE_BOUND_NOT_FINITE.format(bound=f"{bound:g}")))
        if gradient_bound != gradient_bound:
            own.append((reason_codes.GRADIENT_BOUND_NOT_COMPUTED,
                        _CAUSE_GRADIENT_NAN.format(steps=steps)))
        elif not math.isfinite(gradient_bound):
            own.append((reason_codes.GRADIENT_BOUND_NOT_CERTIFIED, _CAUSE_GRADIENT_INF))
    if precision_limited and not declared and floor_measured:
        spectral.append((reason_codes.AT_FLOAT_FLOOR, _CAUSE_FLOOR))
    return spectral, own


def _causes_sentence(spectral, own, spectral_stands: bool = True) -> Optional[str]:
    """The words of :func:`_flag_causes`' two lists (each clause once),
    in the frame every report of a ``False`` flag whose numbers stand
    uses; ``None`` where both are empty.  *spectral_stands* is ``False``
    where another rule already holds ``spectral_usable`` down and only
    the gradient flag's own causes are left to say."""
    def clauses(causes) -> str:
        return "; ".join(dict.fromkeys(clause for _code, clause in causes))

    if spectral:
        told = "spectral_usable is False, and gradient_bound_usable with it: " + clauses(spectral)
        if own:
            told += ". gradient_bound_usable has a cause of its own as well: " + clauses(own)
    elif own and spectral_stands:
        told = "gradient_bound_usable is False (spectral_usable stands): " + clauses(own)
    elif own:
        told = "gradient_bound_usable has a cause of its own as well: " + clauses(own)
    else:
        return None
    return told + ". The numbers are reported as computed."


def _codes_in_order(*codes) -> list:
    """*codes* (iterables of reason codes), each once, in the order of
    ``reason_codes.ALL``: what a report lists for a flag."""
    seen = {code for group_codes in codes for code in group_codes}
    return [code for code in reason_codes.ALL if code in seen]


def _geometry_flag_causes(keys, *, solved, bound: float, gradient_bound: float, rho: float,
                          arnoldi_residual: float, settled: bool, precision_limited: bool,
                          declared: bool, limit, margin, fraction: float, steps: int,
                          group=None, amplification: float = 1.0, residual: float = 0.0,
                          width=_UNCOUNTED_WIDTH, floor_measured: bool = True) -> tuple:
    """``(spectral_usable, gradient_bound_usable, reason, spectral, own)``:
    :func:`_geometry_flags` with the causes behind its reason, each a
    ``(code, clause)`` pair (*spectral*: of ``spectral_usable``; *own*:
    the gradient flag's own).

    *reason* is that function's: ``None`` where the step computed no
    estimate (*rho* is NaN), though the causes are listed then too (why
    there is none, from *group*); the report puts them into words.
    *group*, *amplification*, *residual* and *width* are what
    :func:`_flag_causes` reads beyond the flags' own inputs; their
    defaults are a caller's that holds none of them.
    """
    def number(value) -> float:
        try:
            return float("nan") if value is None else float(value)
        except (TypeError, ValueError):
            return float("nan")

    if solved:
        reason = _GEOMETRY_SOLVED_REASON.format(solved=list(solved), keys=list(keys))
        spectral = [(reason_codes.GEOMETRY_POSITIONS_SOLVED, reason)]
        if rho != rho:
            spectral += _flag_causes(
                group, bound=bound, rho=rho, arnoldi_residual=arnoldi_residual,
                amplification=amplification, settled=settled, residual=residual,
                precision_limited=False, declared=declared, gradient_bound=gradient_bound,
                width=width, fraction=fraction, steps=steps,
                floor_measured=floor_measured)[0]
        return False, False, (None if rho != rho else reason), spectral, []
    margin, limit = number(margin), number(limit)
    recorded = solved is not None and margin == math.inf and limit == math.inf
    spectral, own = _flag_causes(
        group, bound=bound, rho=rho, arnoldi_residual=arnoldi_residual,
        amplification=amplification, settled=settled, residual=residual,
        precision_limited=precision_limited, declared=declared,
        gradient_bound=gradient_bound, width=width, fraction=fraction, steps=steps,
        floor_measured=floor_measured)
    if not recorded:
        spectral.append((reason_codes.GEOMETRY_RECORD_MISSING, _CAUSE_NOT_RECORDED.format(
            keys=list(keys), limit=limit, margin=margin)))
    spectral_usable = not spectral
    gradient_bound_usable = spectral_usable and not own
    reason = None if rho != rho else _causes_sentence(spectral, own)
    return spectral_usable, gradient_bound_usable, reason, spectral, own


def _geometry_flags(keys, *, solved, bound: float, gradient_bound: float, rho: float,
                    arnoldi_residual: float, settled: bool, precision_limited: bool,
                    declared: bool, limit, margin, fraction: float,
                    steps: int) -> tuple[bool, bool, Optional[str]]:
    """``(spectral_usable, gradient_bound_usable, reason)`` of a group whose
    pass resolves a geometry-dependent mapping the diagnostics read, from
    what ``compile()`` committed and its step stored (experimental).  The
    one place the flags of such a group are decided.

    **A group that solves positions has no flag in 0.4.0.**  *solved*
    names the position fields the pass reads from the iterate, or builds
    and reads in the same pass (``InterfacePlan.geometry_iterate_reads``,
    the set the step takes ``geometry_plane_limit`` and
    ``geometry_plane_margin`` over).  Where there is one, both flags are
    ``False`` on every step, whatever the margin, the limit, the bounds or
    the Newton-Kantorovich check read, and the reason is
    :data:`_GEOMETRY_SOLVED_REASON`.  The numbers stay as computed.

    Why no number sets a flag there: a multilinear stencil is one
    polynomial inside a lattice cell and another in the next, and the
    bounds are the linearisation at the returned iterate, which describes
    the pass only in the cells its positions are in *there*.  Three rules
    in turn tried to certify that the fixed point is in those cells (a
    screen on the bound, MADD-ANO-242; the Newton-Kantorovich check where
    a plane is within reach of the bound; a margin of the
    Newton-Kantorovich ball to the nearest plane), and an independent
    audit of each found ``spectral_usable`` set beside a bound far under
    the distance (MADD-ANO-252).  The last: the margin's argument takes
    the cell's polynomial to satisfy Newton-Kantorovich (``h <= 1/2``),
    and a cell whose polynomial has no fixed point at all (a saddle-node
    with a gap of 1e-5 to 1e-8), where the step's own check had failed
    and the nearest plane was beyond the limit's reach, kept the flag on
    a bound 34 to 1,874 times under the distance to the pass's only
    fixed point, one cell on.  A sharper rule is not attempted in 0.4.0.

    **A group whose positions are constants of the pass** (*solved* is
    empty: a target-anchored geometry read by ``update``, positions held
    by a node outside the group) has a smooth group's flags: *bound*
    finite, the estimate *settled*, the floor counted where the residual
    is at it; and for the gradient's a finite *gradient_bound*.  No plane
    can come between the iterate and the fixed point of such a pass.  The
    step's record must agree: it writes *limit* and *margin* as ``inf``
    for such a group, and anything else there -- absent, not a number, a
    finite number, or no *solved* record at all (``None``) -- sets no
    flag (the state was not written by this build's step).

    *reason* is ``None`` where both flags stand, and where the step
    computed no spectral estimate (*rho* is NaN: the numbers say so
    themselves).  Otherwise it is the one reason of a group that solves
    positions, or every cause of each ``False`` flag of one that does
    not: a gradient bound that was not computed (NaN) is told from one
    that did not certify (``inf``).  No reason of a group whose positions
    are constants names a lattice plane.
    """
    spectral_usable, gradient_bound_usable, reason, _spectral, _own = _geometry_flag_causes(
        keys, solved=solved, bound=bound, gradient_bound=gradient_bound, rho=rho,
        arnoldi_residual=arnoldi_residual, settled=settled,
        precision_limited=precision_limited, declared=declared, limit=limit, margin=margin,
        fraction=fraction, steps=steps)
    return spectral_usable, gradient_bound_usable, reason


_WRITTEN_BEFORE_SAVE_REASON = (
    "this report was loaded from a checkpoint saved after the group's state had been "
    "written (set_node_state) since its last step; the state that step returned, which "
    "the float floor is measured on, is not in the checkpoint, so spectral_error_bound, "
    "precision_limited and the *_usable flags are not reported until the group steps. "
    "iterations, residual, converged and the estimates are the step's own."
)


#: The mapping kind whose moving geometry ``convergence_norm="interface"``
#: reads: the one that declares the length scale a position is measured in.
_INTERFACE_NORM_GEOMETRY_KIND = "multilinear_grid"


def _geometry_edge_coupling_errors(group, nodes, plan) -> list[str]:
    """``ERROR:`` issues for a group setting a geometry-dependent mapping
    cannot serve (experimental; empty for every other group).

    ``convergence_norm="interface"`` measures the change, between
    iterates, of what the norm reads on each internal edge.  For an edge
    whose mapping reads a geometry that reading needs the geometry too
    (``InterfaceEdge.parts``), and the norm reads it for the
    ``multilinear_grid`` kind in a group that does not sub-cycle.  Two
    settings are refused, each naming the norms that measure the state
    instead:

    * a geometry-dependent mapping of **another kind** on an internal
      edge: a position is measured in units of the kind's own length
      scale, and only ``multilinear_grid`` declares one in 0.4.0;
    * a **sub-cycled** group with such an edge: the reading is one value
      per pass (the end-of-pass iterate, at the pre-step target
      geometry), which is not what a member that takes several sub-steps
      per pass was handed.

    *plan* is the group's description (``InterfacePlan.geometry_edges``).
    """
    if group.convergence_norm != "interface":
        return []
    names = sorted(group.nodes)
    stem = (
        "ERROR: coupling group {names} uses convergence_norm='interface', which "
        "measures the values the group's internal edges carry, but edge {key!r} "
        "carries its value through a geometry-dependent mapping (geometry "
        "{side}.{field}), {why}.  Use convergence_norm='mixed' or 'l2', which "
        "measure the members' state, the geometry included."
    )
    dividers = _group_dividers(group, nodes) or {}
    sub_cycled = sorted(nn for nn, d in dividers.items() if d > 1)
    errors = []
    for r in plan.geometry_edges():
        if r.mapping_kind != _INTERFACE_NORM_GEOMETRY_KIND:
            why = (f"of kind {str(r.mapping_kind)!r}, and the norm reads a moving geometry "
                   f"only for the {_INTERFACE_NORM_GEOMETRY_KIND!r} kind in 0.4.0 (a "
                   f"position is measured in units of the kind's own length scale)")
        elif sub_cycled:
            why = (f"and the group sub-cycles (members {sub_cycled} take several sub-steps "
                   f"per pass): the norm does not read a moving geometry in a sub-cycled "
                   f"group in 0.4.0")
        else:
            continue
        errors.append(stem.format(names=names, key=r.key, side=r.anchor[0],
                                  field=r.anchor[1], why=why))
    return errors


#: What the rounding of stored positions puts into the float floor of one
#: part of an interface reading, taken by itself, at which ``compile()``
#: warns: the threshold the interface criterion compares the residual
#: with.  At or above it the rounding the floor counts for those
#: positions is, entry for entry, the tolerance asked of the part or more.
_POSITIONS_FLOOR_WARNED = 1.0  # units: tolerances (the residual's units under the interface norm)


def _unresolved_position_warnings(group, plan, state, evaluations) -> list[str]:
    """``UserWarning`` texts for positions an interface reading rests on
    whose rounding the float floor counts at the group's tolerance or
    above (experimental; empty for every other group).

    Under ``convergence_norm="interface"`` the stored positions of a
    geometry-dependent mapping enter the reading of an edge in one of two
    ways (``InterfaceEdge.parts``; which one is decided by the entry
    counts the mapping declares, never by its mode's name):

    * **as a part of their own**, in units of the mapping kind's length
      scale, where the mapping delivers more entries than it reads and
      is anchored at its source (a scatter of fewer points than the grid
      has; a gather onto more points than the grid has, whose positions
      the grid node holds): the criterion asks that they change by less
      than ``rtol`` lengths;
    * **through the value the edge delivers**, where the mapping does
      not deliver more entries than it reads (a gather onto no more
      points than the grid has, a scatter of at least as many points as
      the grid has, a tie; either anchor): the value is computed at
      those positions, and the criterion asks that it change by less
      than ``rtol`` of its own magnitude.

    **Where a position is rounded.**  A position ``u`` lengths from the
    coordinates' zero is stored to ``eps * |u|`` lengths.  The kernel
    then forms its own coordinate from it, in the positions' dtype --
    for ``multilinear_grid`` the lattice coordinate ``(x - origin) /
    spacing``, the distance from the grid's first point -- and its
    weights are resolved to ``eps`` times *that* coordinate's magnitude.
    The count takes the larger of the two per coordinate, ``r`` lengths
    (``_interface_plan._rounded_at``, the one place they are compared;
    the largest over the lattices that read the positions).  Counted
    from the coordinates' zero alone, markers near zero on a grid whose
    first point is far were not warned of and read a floor under one
    with the state 2 to 34 tolerances from the float64 fixed point
    (:func:`~maddening.core.coupling._interface_plan._rounded_at` has
    the numbers).  **No choice of the coordinates' origin changes a
    lattice coordinate**: positions further than ``rtol / (count *
    eps)`` spacings from the grid's first point are warned of wherever
    the zero is put, and the message then offers no change of
    coordinates (float64 positions or a looser tolerance are what is
    left).

    ``eps * r`` lengths is the rounding of a positions part entry for
    entry.  It also moves a kernel weight by as much, and what that does
    to a delivered value depends on the field: the value moves by that
    rounding times the field's variation across one length over the
    value's own magnitude.  **The count takes that ratio to be one** (a
    field that varies across one cell by about the size of the value
    delivered).  It is not a bound in either direction, and it is
    stated as the assumption it is:

    * a field that varies **less** across a cell is moved by less, and
      the warning is early: on two float32 pairs of gathers the count
      reached the tolerance at 132 and 222 spacings while the pairs took
      float64's passes, their readings within 0.03 and 0.05 of a
      tolerance of float64's, and stayed within 1.4 tolerances at 48 and
      24 times the threshold (5032 spacings);
    * a value delivered **far smaller** than the field's variation
      across a cell (a field sampled near its zero: markers on the zero
      contour of a level set, a velocity at a stagnation point) is moved
      by more than the count says, and is **not** warned of at a smaller
      count: at a ratio of 250 and 2500, 100 spacings from zero (count
      0.96, no warning), a float32 pair reported ``converged=True`` 7.3
      and 23 tolerances from its fixed point, and at 10 spacings (count
      0.097) 2.8 tolerances at a ratio of 2500, where the same pairs
      with float64 positions are within 0.5 (MADD-ANO-247, open; all on
      jaxlib 0.11.0, CPU, ``rtol=1e-4``, Gauss-Seidel).

    The float floor of the residual
    (:func:`~maddening.core.coupling.acceleration.residual_precision_floor`)
    counts ``PRECISION_FLOOR_ULPS`` of those roundings per evaluation of
    the pass for every entry of either part.  **Warned: a part for which
    that count reaches the tolerance**,

        ``PRECISION_FLOOR_ULPS * evaluations * eps * max r >= rtol``,

    which is where what the positions put into the floor of the part by
    itself reaches the criterion's threshold
    (``acceleration._positions_floors``, the floor's own arithmetic:
    the whole floor of a positions part, and of a delivered value
    wherever the positions' rounding is coarser than the value's own).
    So a group that is not warned has a floor the positions leave below
    its threshold (pooled with the other entries the norm reads they
    contribute at most the largest part's), and one that is warned has
    entries whose counted rounding is the tolerance asked of them or
    more, and a floor of at least that times the root of their share of
    the entries the norm reads: the loop can run to its cap on rounding
    alone.

    **One part's number, beside a pooled floor.**  The number the
    message quotes is what the positions put into the floor of *this
    part by itself*.  The report's ``residual_precision_floor`` is
    pooled, one RMS over every entry the group's norm reads, so with
    finer entries beside the part it is smaller, by up to the root of
    the part's share (four markers a thousand spacings out beside two
    plain edges of 1e5 entries: 95.7 quoted here, 0.61 reported).  The
    message says which to compare with what: this number with one (are
    these positions resolved to the tolerance asked of them?), the
    report's floor with its ``residual`` and with one (is the group's
    criterion?).

    **The floor's own decisions.**  The parts asked are the parts the
    floor counts, by the floor's own functions
    (``acceleration._positions_floors``): a delivered value the dead
    band drops (``atol`` above its magnitude) puts nothing into the
    floor and is not warned of, and neither is a delivered value for a
    coordinate its mapping does not read -- one on an axis of one
    lattice point, or of a point clamped to the hull from further out
    than its rounding can cross
    (``MultilinearGridMapping.geometry_coordinates_read``).  A positions
    part is asked of every coordinate, because the criterion reads every
    coordinate of it (in the spacing declared for its axis, an axis of
    one lattice point included).

    A mapping read at its source and anchored at its **target** is not
    asked: its reading is the source field alone (the positions are the
    pre-step state, a constant of the solve), and the floor counts no
    position for it.  (A gap, MADD-ANO-258, open: the kernel still forms
    its weights from those positions in their dtype, and with no edge in
    the group that delivers a value at them, float32 positions thousands
    of spacings from the grid's first point are flagged by nothing.)  A
    delivered value at a target anchor is asked at
    the target's positions in *state*: what a step started from it reads.

    A warning, never a refusal, and it changes no number: where the
    positions settle to the bit the group converges as before.
    Measured on two float32 pairs with a scatter at ``rtol=1e-4``
    against the same pairs in float64 (jaxlib 0.11.0, CPU): the same
    passes up to 2 and 6 times the threshold, more passes from 3 and 10
    times, and one of the two at its cap at 24 times.  Below the
    threshold the positions still enter the floor: the two pairs stop at
    residuals of 0.25 and 0.48 tolerances, which is at or under their
    floors at 0.44 and 0.87 of the threshold (and above them at 0.15 and
    0.30), so a residual can be at its floor without this warning; what
    the warning marks is where the positions' rounding by itself reaches
    the tolerance.

    **Asked once, of the state ``compile()`` sees.**  Positions that
    move afterwards -- by the markers' own update, a ``set_node_state``,
    a loaded checkpoint -- are not asked again by this function.  The
    run-time reading is the report's: ``coupling_diagnostics()`` gives
    the floor of the state every step returned
    (``residual_precision_floor``) and ``precision_limited``
    (:func:`_floor_reading_of_a_norm_withheld_report`).  The two count
    a position alike, at the ``eps`` of the dtype it is stored in times
    the magnitude it is rounded at
    (``acceleration._positions_resolution`` of
    ``PartReading.positions``; the group's coarsest floating dtype is
    for the floor's value entries): float64 positions beside float32
    fields are silent here and put only float64's ``eps * r`` into the
    report's floor.
    *evaluations* is the group's structural count
    (:func:`_group_evaluations`, on the compiled schedule), which is
    also the count the report's floor of such a group takes, with
    diagnostics on or off (:func:`_reports_the_structural_count`).
    """
    if group.convergence_norm != "interface":
        return []
    names = sorted(group.nodes)
    rtol = float(group.rtol)
    count = PRECISION_FLOOR_ULPS * float(evaluations)
    out = []
    lattices = _interface_plan._position_lattices(plan.internal)
    for reading, holder, held, resolution, floor in _positions_floors(
            plan, state, rtol, evaluations, atol=float(group.atol)):
        if not floor >= _POSITIONS_FLOOR_WARNED:
            continue            # resolved (or not a number: the criterion's own failure)
        edge, dtype = reading[0], np.dtype(held.dtype)
        positions = np.abs(np.asarray(held, np.float64))
        if not np.all(np.isfinite(positions)):
            continue            # a non-finite position fails the criterion by itself
        columns = positions.reshape(positions.shape[0], -1)
        axis = int(np.argmax(np.max(columns, axis=0)))
        reach = float(np.max(columns))
        spacing = _interface_plan._kernel_lengths(edge.mapping)[axis]
        eps = float(np.finfo(dtype).eps)
        node, field = holder
        # The two magnitudes the count took the larger of, on that axis,
        # from the functions it took them from: which one it is decides
        # what the message says of the coordinates.
        among = lattices.get(holder) or (edge.mapping,)
        geometry = state[node][field]

        def on_axis(magnitude) -> float:
            values = np.asarray(magnitude(among, geometry), np.float64)
            return float(np.max(values.reshape(values.shape[0], -1)[:, axis]))

        from_zero = on_axis(_interface_plan._stored_magnitude)
        on_lattice = on_axis(_interface_plan._kernel_magnitude)
        allowed = rtol / (count * eps)      # units: spacings (the count is one there)
        by_lattice = on_lattice > from_zero
        key = getattr(edge, "key", None)
        entries = int(np.asarray(reading[2]).size)
        if by_lattice:
            where = (
                f"They reach {reach:.6g} spacings from the grid's first point (axis {axis}, "
                f"spacing {spacing:g}; {from_zero:.6g} from zero), where the mapping forms a "
                f"{dtype} lattice coordinate to {resolution:.3g} spacings")
        else:
            where = (
                f"They reach {reach:.6g} spacings from zero (axis {axis}, spacing "
                f"{spacing:g}; {on_lattice:.6g} from the grid's first point), where a {dtype} "
                f"position is stored to {resolution:.3g} spacings")
        if reading.part.unit == _interface_plan.KERNEL_LENGTH:
            # The positions are a part of the reading.
            subject = (
                f"the {dtype} positions {node}.{field} read on edge {key!r} cannot be "
                f"resolved to this tolerance. The norm measures them in grid spacings and "
                f"asks that they change by less than rtol of one.")
            stored = where
            consequence = (
                "Rounding alone can keep these positions from meeting the criterion (the "
                "group then runs to max_iterations), and where it is met it says little "
                "of them.")
        else:
            # The reading is a value computed at the positions.
            subject = (
                f"the {dtype} positions {node}.{field} that the value on edge {key!r} is "
                f"delivered at cannot be resolved to this tolerance of a grid spacing. The "
                f"norm reads what the edge delivers, a value its mapping computes at those "
                f"positions, and asks that it change by less than rtol of its own magnitude.")
            stored = (
                f"{where} and a "
                f"weight of the mapping moves by as much. The count takes the delivered "
                f"value to move by that fraction of its own magnitude, which assumes a "
                f"field that varies across one cell by about the size of the value "
                f"delivered: a field that varies less is moved by less (the warning is then "
                f"early), and a value delivered far smaller than the field's variation "
                f"across a cell (a field sampled near its zero) is moved by more, which "
                f"this count does not see (MADD-ANO-247)")
            consequence = (
                "Rounding alone can keep the delivered value from meeting the criterion "
                "(the group then runs to max_iterations), and where it is met it says "
                "little of the value's last digits.")
        if len(among) > 1:
            stored += (f" (the largest over the {len(among)} lattices that read these "
                       f"positions, each in its own spacing)")
        remedies = [f"loosen rtol above {count * resolution:.3g}"]
        if on_lattice < allowed:
            remedies.insert(0, (
                f"use coordinates local to the grid (for a {dtype} position this count is "
                f"under the tolerance within {allowed:.3g} spacings of zero and of the "
                f"grid's first point)"))
            coordinates = ""
        else:
            coordinates = (
                f" No choice of the coordinates' origin brings this count under the "
                f"tolerance: a lattice coordinate does not depend on it, and these positions "
                f"are {on_lattice:.6g} spacings from the grid's first point where a {dtype} "
                f"one allows {allowed:.3g}.")
        if dtype != np.dtype(np.float64):
            remedies.insert(0, (
                f"hold {node}.{field} in float64 (under jax_enable_x64; the mapping computes "
                f"its weights in the geometry's dtype and casts them to the field's, so the "
                f"other fields can stay as they are: the report's floor counts a position "
                f"at its own dtype too)"))
        out.append(
            f"coupling group {names} (convergence_norm='interface', rtol={rtol:g}): "
            f"{subject} {stored}; "
            f"a position is rounded at the larger of its distance from the coordinates' "
            f"zero and of its lattice coordinate. "
            f"The residual's float floor counts {count:g} of those roundings per pass "
            f"(PRECISION_FLOOR_ULPS times {float(evaluations):g} evaluation(s)), which is "
            f"{floor:.3g} times the tolerance for this part by itself. {consequence} "
            f"This is asked once, of the state compile() sees, and of this part alone: the "
            f"report's residual_precision_floor is the reading at every step, and it is "
            f"pooled, one RMS over every entry the group's norm reads (this part's "
            f"{entries} among them), so it is smaller than this number where the entries "
            f"beside it are finer. Compare this number with one (are these positions "
            f"resolved to the tolerance asked of them), and the report's floor with its "
            f"residual (precision_limited) and with one (is the group's criterion)."
            f"{coordinates} "
            f"Remedies: " + "; or ".join(remedies) + "."
        )
    return out


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
