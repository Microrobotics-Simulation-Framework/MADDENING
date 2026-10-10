"""Reason codes of ``GraphManager.coupling_diagnostics()`` (EXPERIMENTAL).

A report's ``"spectral_usable"`` and ``"gradient_bound_usable"`` are
``False`` for causes that mean different things to a caller, and a test
cannot branch on the sentence in ``"not_usable_reason"``.  So every
group's entry carries ``"reason_codes"``: a dict with one list of the
constants below for each of ``"spectral_usable"``,
``"gradient_bound_usable"`` and ``"precision_limited"``::

    codes = gm.coupling_diagnostics()["a+b"]["reason_codes"]
    if reason_codes.INTERFACE_TOO_WIDE in codes["spectral_usable"]:
        ...   # expected for this group: nothing is wrong

The rules the report keeps:

* a usable flag that is ``False`` has at least one code, and the entry a
  ``"not_usable_reason"`` that names each of them in words; a flag that
  is ``True`` has none;
* every cause that holds is listed, in the order of :data:`ALL`: a group
  can be too wide for the estimate *and* at its float floor;
* ``"gradient_bound_usable"`` rests on ``"spectral_usable"``, so its list
  holds the spectral flag's codes and then its own
  (:data:`GRADIENT_BOUND_NOT_COMPUTED`, :data:`GRADIENT_BOUND_NOT_CERTIFIED`);
* ``"precision_limited"`` has a code exactly where the float floor was
  **not measured** (``"residual_precision_floor"`` is then NaN and the
  flag reads ``False`` without saying anything of rounding); where the
  floor was measured its list is empty, whatever the flag reads.

The codes are plain lower-case strings, so a report goes through JSON, a
checkpoint and the REST server as it is.  They are computed on the host
from what the step stored and what ``compile()`` recorded; the compiled
step is the same program with or without them.

**Three kinds of cause** (:data:`EXPECTED`, :data:`CONFIGURATION`,
:data:`WORRY`): what the group is and 0.4.0 does not report, what the
caller chose, and what deserves a look.  The split is a reading aid; the
guide (``docs/developer_guide/coupling_algorithm_guide.md``, "The reason
codes of a report") says what to do about each code.

Experimental: the codes, their names and the two keys may change in a
later release.
"""

from __future__ import annotations

#: The group's solver is not ``"ift"``.  The legacy ``"fori"`` solver has
#: no linearisation of the pass, so neither the spectral estimate nor the
#: gradient bound exists for it, with or without ``diagnostics=True``.
SOLVER_NOT_IFT = "solver_not_ift"

#: The group was built with ``diagnostics=False`` (the default): its
#: steps do not compute the spectral estimate or the gradient bound.
#: Build the group with ``diagnostics=True`` for a run that reports them.
DIAGNOSTICS_OFF = "diagnostics_off"

#: ``max_iterations=1``: one staggered pass solves no fixed point, so
#: there is no Jacobian of a solve to take a spectrum of.
SINGLE_PASS = "single_pass"

#: The state the step returned is not finite (the residual is ``inf``):
#: the iteration diverged, and nothing is computed at such a state.
STATE_NOT_FINITE = "state_not_finite"

#: The state holds no spectral estimate for a group that asks for one
#: (``solver="ift"``, ``diagnostics=True``, a finite residual): it was
#: not written by a step of this graph as compiled -- a state set by
#: hand, or a checkpoint of a graph built without diagnostics.  The
#: estimates return when the group steps.
ESTIMATE_NOT_RECORDED = "estimate_not_recorded"

#: The step recorded an estimate, and ``spectral_error_bound`` is still
#: not a number: the residual, or its float floor, is not finite at the
#: state the report describes.
BOUND_NOT_EVALUATED = "bound_not_evaluated"

#: The linearised pass does not contract at the returned state:
#: ``rho_spectral >= 1``, or the compressed ``I - H`` is singular.
#: Nothing is bounded; the fixed-point iteration itself may still have
#: converged under relaxation or acceleration.
NOT_CONTRACTING = "not_contracting"

#: The spectral estimate did not settle, and ``compile()`` counted more
#: scalars in one pass's dependence on the previous one than the
#: estimate's eight Krylov steps can span (more than seven, in a state of
#: more than eight entries).  **Expected for such a group: nothing is
#: wrong**, and no tolerance or dtype changes it.  The count is
#: structural (what the internal edges carry and what the members that
#: read the previous pass hold: an upper bound on the rank), so this code
#: says the limit explains the unsettled estimate, not that rounding had
#: no part in it.
INTERFACE_TOO_WIDE = "interface_too_wide"

#: The spectral estimate did not settle although the pass depends on the
#: previous one through few enough scalars for the eight Krylov steps to
#: span: the estimate's check of itself did not pass.  One more product
#: moved the radius by more than the flag's margin, or rounding could
#: have, or the breakdown test discarded a direction, or the residual was
#: not absorbed into the space.  **A worry**: ``rho_spectral`` and the
#: bound are not to be trusted here.  A wider dtype is the usual way out.
SPECTRAL_SELF_CHECK_FAILED = "spectral_self_check_failed"

#: The residual is at its float floor (``precision_limited``) and not
#: every member declares how many evaluations its update makes
#: (``update_evaluations``), so the floor the bound rests on is not
#: checked.  A wider dtype, or a looser tolerance, lifts the residual off
#: the floor; declaring the counts keeps the flag at the floor.
AT_FLOAT_FLOOR = "at_float_floor"

#: ``gradient_relative_error_bound`` is NaN beside a computed spectral
#: estimate: the Jacobian's range was not captured by the bound's eight
#: directions, or the fixed point responds to no constant the bound
#: probes (a group whose members declare no parameter and read no
#: pre-step state has nothing to take a gradient in).
GRADIENT_BOUND_NOT_COMPUTED = "gradient_bound_not_computed"

#: ``gradient_relative_error_bound`` is ``inf``: its Newton-Kantorovich
#: check did not pass (the Jacobian changes too much across the Newton
#: step), or nothing contracts.
GRADIENT_BOUND_NOT_CERTIFIED = "gradient_bound_not_certified"

#: The residual does not stand clear of the float floor that a long row
#: of a static mapping on an internal edge would give it (a row over
#: ``MAPPED_ROW_FLOOR_LIMIT`` entries, of any static kind): the floor
#: does not count the rounding of such a row's sum (MADD-ANO-257).  The
#: numbers are reported as computed; a wider dtype at the same tolerance
#: is the way out.
LONG_MAPPED_ROW = "long_mapped_row"

#: The report was loaded from a checkpoint saved after the group's state
#: had been written since its last step: the state that step returned,
#: which the float floor is measured on, is not in the checkpoint.
#: Everything built on the floor is withheld until the group steps.
WRITTEN_BEFORE_SAVE = "written_before_save"

#: The float floor of this group's residual reads what an internal edge
#: delivers at its target's pre-step geometry, which only the step holds,
#: and the state carries no floor recorded by a step.  Everything built
#: on the floor is withheld until the group steps.
FLOOR_NEEDS_THE_STEP = "floor_needs_the_step"

#: The group resolves a geometry-dependent mapping of a kind other than
#: ``multilinear_grid``, whose moving geometry the diagnostics do not
#: read in 0.4.0: no bound or estimate is reported.
GEOMETRY_KIND_NOT_DIAGNOSED = "geometry_kind_not_diagnosed"

#: The group resolves a geometry-dependent mapping under
#: ``convergence_norm="interface"``, whose bounds are not in 0.4.0.  A
#: diagnostic run under ``"mixed"`` or ``"l2"`` reports them.  The float
#: floor of such a group is still reported.
GEOMETRY_NORM_NOT_DIAGNOSED = "geometry_norm_not_diagnosed"

#: The group resolves a geometry-dependent mapping and sub-cycles: the
#: geometry of each sub-step is not followed, and no bound or estimate
#: is reported.
GEOMETRY_SUBCYCLED = "geometry_subcycled"

#: The step's check of its own Jacobian-vector product along the
#: geometry read a gap over its tolerance: a member or a mapping whose
#: derivative is not that of its value, or a finite difference that
#: could not be formed at this state.  **A worry** where it persists.
GEOMETRY_SELF_CHECK_FAILED = "geometry_self_check_failed"

#: The same check produced no number (the pass or its product is not a
#: number at the returned state, or the pass reads a geometry from a
#: constant the step could not move): nothing was compared.
GEOMETRY_SELF_CHECK_NOT_EVALUATED = "geometry_self_check_not_evaluated"

#: The group solves the positions a geometry-dependent mapping reads
#: (its pass reads them from the iterate, or builds and reads them in the
#: same pass).  0.4.0 certifies no bound for such a group, on any step
#: (MADD-ANO-252); the numbers are reported, uncertified.
GEOMETRY_POSITIONS_SOLVED = "geometry_positions_solved"

#: The step did not record that every position of the group's
#: geometry-dependent mappings was fixed during its pass: the state was
#: not written by this build's step.  The flags return when the group
#: steps.
GEOMETRY_RECORD_MISSING = "geometry_record_missing"

#: Every code, in the order a report lists them.
ALL = (
    SOLVER_NOT_IFT, DIAGNOSTICS_OFF, SINGLE_PASS, STATE_NOT_FINITE, ESTIMATE_NOT_RECORDED,
    WRITTEN_BEFORE_SAVE, FLOOR_NEEDS_THE_STEP,
    GEOMETRY_KIND_NOT_DIAGNOSED, GEOMETRY_NORM_NOT_DIAGNOSED, GEOMETRY_SUBCYCLED,
    GEOMETRY_SELF_CHECK_FAILED, GEOMETRY_SELF_CHECK_NOT_EVALUATED,
    GEOMETRY_POSITIONS_SOLVED, GEOMETRY_RECORD_MISSING,
    BOUND_NOT_EVALUATED, NOT_CONTRACTING, INTERFACE_TOO_WIDE, SPECTRAL_SELF_CHECK_FAILED,
    AT_FLOAT_FLOOR, LONG_MAPPED_ROW,
    GRADIENT_BOUND_NOT_COMPUTED, GRADIENT_BOUND_NOT_CERTIFIED,
)

#: What the group is, and 0.4.0 does not report for it: nothing is wrong,
#: and nothing short of another model changes it.
EXPECTED = frozenset({
    INTERFACE_TOO_WIDE, GRADIENT_BOUND_NOT_COMPUTED, GEOMETRY_KIND_NOT_DIAGNOSED,
    GEOMETRY_NORM_NOT_DIAGNOSED, GEOMETRY_SUBCYCLED, GEOMETRY_POSITIONS_SOLVED,
})

#: What the caller chose or can change at once: the solver, the
#: diagnostics switch, the budget, the dtype or the tolerance, a
#: declaration, a step to take.
CONFIGURATION = frozenset({
    SOLVER_NOT_IFT, DIAGNOSTICS_OFF, SINGLE_PASS, ESTIMATE_NOT_RECORDED, AT_FLOAT_FLOOR,
    LONG_MAPPED_ROW, WRITTEN_BEFORE_SAVE, FLOOR_NEEDS_THE_STEP, GEOMETRY_RECORD_MISSING,
})

#: What deserves a look: the solve, the model or the arithmetic is not
#: what the estimate takes it to be.
WORRY = frozenset({
    STATE_NOT_FINITE, BOUND_NOT_EVALUATED, NOT_CONTRACTING, SPECTRAL_SELF_CHECK_FAILED,
    GRADIENT_BOUND_NOT_CERTIFIED, GEOMETRY_SELF_CHECK_FAILED,
    GEOMETRY_SELF_CHECK_NOT_EVALUATED,
})

#: The flags a report's ``"reason_codes"`` has a list for, in its order.
FLAGS = ("spectral_usable", "gradient_bound_usable", "precision_limited")

__all__ = [
    "ALL", "CONFIGURATION", "EXPECTED", "FLAGS", "WORRY",
    "AT_FLOAT_FLOOR", "BOUND_NOT_EVALUATED", "DIAGNOSTICS_OFF", "ESTIMATE_NOT_RECORDED",
    "FLOOR_NEEDS_THE_STEP", "GEOMETRY_KIND_NOT_DIAGNOSED", "GEOMETRY_NORM_NOT_DIAGNOSED",
    "GEOMETRY_POSITIONS_SOLVED", "GEOMETRY_RECORD_MISSING", "GEOMETRY_SELF_CHECK_FAILED",
    "GEOMETRY_SELF_CHECK_NOT_EVALUATED", "GEOMETRY_SUBCYCLED",
    "GRADIENT_BOUND_NOT_CERTIFIED", "GRADIENT_BOUND_NOT_COMPUTED", "INTERFACE_TOO_WIDE",
    "LONG_MAPPED_ROW", "NOT_CONTRACTING", "SINGLE_PASS", "SOLVER_NOT_IFT",
    "SPECTRAL_SELF_CHECK_FAILED", "STATE_NOT_FINITE", "WRITTEN_BEFORE_SAVE",
]
