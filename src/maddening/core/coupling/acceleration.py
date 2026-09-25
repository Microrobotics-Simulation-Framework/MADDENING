"""
Coupling acceleration methods and convergence utilities.

Provides residual norms, state flattening/unflattening, and
acceleration strategies (Aitken, fixed relaxation, IQN-ILS, IQN-IMVJ)
for iterative coupling.

All functions are JAX-traceable pure functions suitable for use
inside ``jax.lax.fori_loop``.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np


# ------------------------------------------------------------------
# Convergence norms
# ------------------------------------------------------------------

def _is_float_leaf(v) -> bool:
    return jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)


def float_fields_of(state: dict[str, dict], node_names) -> dict[str, tuple[str, ...]]:
    """``{node: (float fields...)}`` for the given nodes -- the fields a
    coupling norm, a predictor or a fixed-point vector may contain.  A
    counter, a flag or a PRNG key is recomputed from the pre-step state
    on every pass and has no place in a floating-point norm."""
    return {
        nn: tuple(f for f in sorted(state[nn]) if _is_float_leaf(state[nn][f]))
        for nn in node_names
    }


def _field_reference(new_val, old_val):
    """The field's own magnitude, used as the scale of its criterion.

    ``max |v|`` over the whole field rather than element by element:
    the question a convergence criterion answers is "has *this
    quantity* stopped moving", and a quantity is one field, not one
    array entry.  Taking the maximum over both iterates makes the
    reference monotone in the iterate rather than oscillating with it.
    """
    return jnp.maximum(jnp.max(jnp.abs(new_val)), jnp.max(jnp.abs(old_val)))


def _scaled_change(new_val, old_val, atol: float, rtol: float):
    """``(|dx| / (rtol * ref), active)`` for one field.

    ``active`` is the dead band: a field whose own magnitude does not
    exceed ``atol`` is *at zero within the tolerance the caller
    declared*, and contributes nothing.  That is the only place an
    absolute number enters, and it is the only place one can: "how
    small is indistinguishable from zero" is the one question that
    genuinely has units.  Everywhere above the dead band the criterion
    is a ratio, so it says the same thing whether a force is quoted in
    newtons or micronewtons.

    **The dead band is an assertion the caller makes, so its default
    asserts nothing (``atol=0.0``).**  Leaving the norm is not the same
    as being held to a looser threshold: an excluded field contributes
    exactly zero, so a group every one of whose moving fields is
    excluded reports ``residual=0.0, converged=True`` after one pass
    however far it is from its fixed point, and no ``tolerance`` can
    contradict it.  Only the caller knows which of their quantities is
    noise -- a float32 holding 1e-9 carries the same seven significant
    digits as one holding 1.0, so nothing here can tell "small because
    it is nothing" from "small because it is measured in metres".
    0.4.0 changed ``atol`` from a floor under the scale, where 1e-8
    merely *loosened* a small field's criterion, into this exclusion,
    where 1e-8 *removes* it; the default had to move with the meaning.
    Set it to the field's noise floor when you have one.

    Dividing is safe by construction — the denominator is only ever
    used where ``ref > atol``, and elsewhere the ``where`` selects a
    zero contribution — so a field that is legitimately at zero neither
    divides by something tiny nor blocks convergence forever.  At the
    default that ``scale > 0`` term is the whole of the guard, and it
    is the only exclusion that needs no units: a field with no scale
    has no ratio to contribute.

    **A field the criterion cannot be evaluated on fails it; it does
    not leave it.**  Two conditions make a field unevaluable, and both
    return ``(inf, True)``: every entry reads ``inf``, the field counts
    as active, and every norm built on this helper is ``inf`` -- so
    ``converged`` is False, ``residual`` is reported non-finite,
    ``error_amplification`` rejects the ratio and ``strict_convergence``
    raises.  The dead band above is untouched for finite fields: a
    field that is genuinely at zero still drops out exactly as before.

    1. *A non-finite reference.*  One ``NaN`` or ``inf`` entry in either
       iterate makes ``ref`` non-finite.  Before 0.4.0 that fed straight
       into the norm and the residual read ``NaN``, which no criterion
       accepts.  The dead band changed that: ``NaN > atol`` is False, so
       a NaN field was *inactive* and contributed exactly zero, and a
       relaxed iteration that diverged past float32 range reported
       ``residual=0.0, converged=True`` on the state it had just
       destroyed (MADD-ANO-019).
    2. *A reference too large for its own scale.*  ``diff / scale`` is a
       broadcast divide, which XLA's CPU backend lowers to a multiply by
       the reciprocal of ``scale``; above ``1 / finfo.tiny`` (about
       ``8.5e37`` in float32, only reachable under the L2 norm's
       ``rtol=1.0``) that reciprocal is subnormal and is flushed to
       zero, so a field moving by twice its own size read as not moving
       at all, and the same iteration reported ``converged=True`` one
       pass *before* it overflowed, with a finite state of ``-1.2e38``.
       Rather than depend on how a backend lowers a divide, a scale
       whose reciprocal is not a normal number of the field's dtype is
       out of range.  The boundary this draws: a float32 field within a
       factor of four of overflow can never read as converged.  No
       simulation state lives there on purpose, and one that does has
       already lost every digit of the step it is being asked about.

    **The other end of the range: a scale too small to be a number.**
    Under the ``"mixed"`` and ``"interface"`` norms the scale is
    ``rtol * ref``, and for a field whose own magnitude is a perfectly
    normal float32 but below ``finfo.tiny / rtol`` (about ``1.2e-32``
    at ``rtol=1e-6``) that product is subnormal.  The CPU backend
    flushes it to zero, ``scale > 0`` was False, and the field was
    *inactive* -- the dead band by another route, one the caller never
    declared: a field at ``1e-33`` was reported ``iterations=1,
    residual=0.0, converged=True`` while 90% from its fixed point.
    Failing closed there, as the overflow end does, would be wrong in
    the other direction: the field holds its full 24 bits and its
    change is perfectly measurable, it is only the product that is not.
    So a field above the dead band whose scale is not a normal number
    is measured on the rescaled pair ``|new * 2**k - old * 2**k| / (rtol
    * (ref * 2**k))`` with ``2**k = 1 / finfo.tiny``, which is the same
    quotient -- a power of two scales exactly -- computed where every
    operand is normal.

    **The change is rescaled before it is taken, not after.**  The
    *difference* of two normal numbers is subnormal whenever it is below
    ``finfo.tiny``, which for a field of magnitude ``ref`` is every
    change of less than ``tiny / ref`` of itself -- one ulp as soon as
    ``ref < tiny / eps`` (about ``9.9e-32`` in float32), whatever the
    norm's ``rtol``.  The CPU backend flushes that difference to zero,
    so rescaling ``|new - old|`` afterwards rescaled a zero: a field at
    ``1e-35`` read ``residual=0.0, converged=True`` 43 passes into an
    iteration that takes 126-130 at ``1e-29``, under all three norms and
    both solvers, and the spectral bound built on that residual read
    6e-4 of the true distance with ``spectral_usable=True``.  So wherever
    a one-ulp change of the field is below the normal range the pair is
    multiplied by ``2**k`` first and the difference is taken between two
    normal numbers; the norm then says the same thing about a group at
    ``1e-35`` as about the same group scaled by any power of two into
    the normal range -- to the bit, pinned by
    ``test_the_verdict_does_not_change_below_the_change_underflow``.

    A field whose magnitude is *itself* subnormal is below what the
    dtype resolves: a flush-to-zero backend (XLA's CPU backend is one)
    reads it as zero and it leaves the norm as a field at zero does;
    elsewhere it is rescaled like the rest.  Every other field is
    measured by the expressions the norms always evaluated, selected by a
    ``where``, so its value -- and the compiled loop it sits in -- is
    bit-identical to what it was before either guard existed.
    """
    ref = _field_reference(new_val, old_val)
    scale = rtol * ref
    dtype = jnp.asarray(scale).dtype
    info = jnp.finfo(dtype)
    tiny = info.tiny
    # ``scale * tiny`` is exact (``tiny`` is a power of two) unless it
    # underflows, and an underflow means the scale is small, which is
    # the evaluable side.  Non-finite ``ref`` fails ``isfinite``; an
    # ``inf`` scale also fails the product test, so the two agree.
    evaluable = jnp.logical_and(
        jnp.isfinite(ref),
        scale * tiny <= 1.0,
    )
    # The underflow end of the *scale*: above the caller's dead band (so
    # ``ref > 0``) but a scale below the smallest normal number (or
    # flushed to zero).  There numerator and denominator are both
    # multiplied by ``k = 1 / tiny``, a power of two, so the quotient is
    # the same one; everywhere else ``k`` is exactly ``1.0`` and
    # multiplying by it is exact.  NaN ``ref`` and ``scale`` compare
    # False, so this is finite-only.
    #
    # These are the expressions the norms have always evaluated, kept
    # verbatim -- the L2 norm's two-op form included -- because the
    # residual shares a compiled loop body with the node updates, and
    # XLA fuses the two: rewritten in an equivalent form (the pair
    # rescaled before subtracting, ``k`` exactly 1.0 in range) the
    # arithmetic of the *node updates* moved by an ulp under
    # ``solver="ift"`` and the L2 norm, and one group stopped on a
    # different pass.  So the rescaled pair below is an alternative the
    # ``where`` selects, not a change to these.
    diff = jnp.abs(new_val - old_val)
    above = ref > atol
    if isinstance(rtol, (int, float)) and float(rtol) >= 1.0:
        # ``scale >= ref`` here (the L2 norm passes ``rtol=1.0``), so a
        # field with a normal magnitude has a normal scale.
        active = jnp.logical_and(above, scale > 0)
        safe = jnp.where(active, scale, jnp.ones_like(scale))
        scaled = jnp.where(active, diff / safe, jnp.zeros_like(diff))
    else:
        underflow = jnp.logical_and(above, scale < tiny)
        k = jnp.where(underflow, 1.0 / tiny, 1.0).astype(dtype)
        active = jnp.logical_or(jnp.logical_and(above, scale > 0), underflow)
        safe = jnp.where(active, rtol * (ref * k), jnp.ones_like(scale))
        scaled = jnp.where(active, (diff * k) / safe, jnp.zeros_like(diff))
    # The underflow end of the *change*: a field so small that a change
    # of one ulp of it is subnormal (``ref < tiny / eps``), which the
    # subtraction above flushes to zero, whatever ``rtol``.  There the
    # pair is multiplied by ``K = 1 / tiny`` *before* it is subtracted,
    # so the difference is taken between two normal numbers, and the
    # scale by the same ``K``: the same quotient, to the bit, as the
    # same group scaled by a power of two into the normal range.  This
    # covers the scale's underflow end too wherever ``rtol >= eps``.
    # ``ref * K`` stays below ``1 / eps`` on the fields it is applied to.
    small = jnp.logical_and(above, ref < float(tiny) / float(info.eps))
    big_k = float(1.0 / float(tiny))
    active = jnp.logical_or(active, small)
    pair = jnp.abs(new_val * big_k - old_val * big_k)
    small_safe = jnp.where(small, rtol * (ref * big_k), jnp.ones_like(scale))
    scaled = jnp.where(small, pair / small_safe, scaled)
    # ``where`` with the finite computation in the *selected* branch:
    # an evaluable field's value, and its gradient, are exactly what
    # they were before this guard existed.
    scaled = jnp.where(evaluable, scaled, jnp.full_like(diff, jnp.inf))
    active = jnp.logical_or(active, jnp.logical_not(evaluable))
    return scaled, active


def coupling_residual_l2(
    s_new: dict[str, dict],
    s_old: dict[str, dict],
    node_names: list[str],
    atol: float = 0.0,
) -> jnp.ndarray:
    """L2 norm of the *relative* state change between iterations.

    Each field's change is divided by the field's own magnitude before
    the norm is taken, so the number is dimensionless and a group whose
    fields happen to be quoted in small units is held to the same
    standard as one quoted in large ones.  For fields of order one this
    is the unscaled ``||dx||`` it replaces.

    Parameters
    ----------
    s_new : dict
        New iteration state.
    s_old : dict
        Previous iteration state.
    node_names : list of str
        Node names to include in the norm.
    atol : float
        Dead band: a field whose magnitude does not exceed ``atol`` is
        treated as being at zero and contributes nothing, so the group
        stops being held to any criterion on it.  The default asserts
        no noise floor and excludes only a field with no scale at all;
        see :func:`_scaled_change` for why the caller owns this number.

    Returns
    -------
    jnp.ndarray
        Scalar norm, compared against ``CouplingGroup.tolerance``, which
        is therefore a *relative* tolerance.
    """
    total = jnp.array(0.0)
    for nn in node_names:
        for field_name in s_new[nn]:
            new_val = s_new[nn][field_name]
            if not _is_float_leaf(new_val):
                continue        # counters / flags / keys: not part of the norm
            old_val = s_old[nn][field_name]
            if jnp.asarray(new_val).size == 0:
                continue
            # ``rtol=1.0``: the L2 norm carries its threshold in
            # ``tolerance``, so the scale here is the bare magnitude.
            scaled, _active = _scaled_change(new_val, old_val, atol, 1.0)
            total = total + jnp.sum(scaled ** 2)
    return jnp.sqrt(total)


def coupling_residual_mixed(
    s_new: dict[str, dict],
    s_old: dict[str, dict],
    node_names: list[str],
    atol: float,
    rtol: float,
) -> jnp.ndarray:
    """Scale-aware RMS convergence norm over every float field.

    Uses the formula::

        err_i = |new_i - old_i| / (rtol * ref_field)

    where ``ref_field = max |v|`` over the field, and a field whose
    ``ref_field`` does not exceed ``atol`` is treated as being at zero
    and is left out of the norm entirely.  Converged when the result is
    <= 1.0.

    This replaces the elementwise ``atol + rtol * |v_i|`` scale, which
    was ``atol`` alone — an absolute criterion — for every field
    smaller than ``atol / rtol``.  A 1.7e-05 N force against the default
    ``atol=1e-8`` was being asked to move by less than 6e-04 of itself,
    not by less than ``rtol``, and satisfied that on its first pass
    while still percent-sized from its fixed point.  Above the dead band
    the new scale is a pure ratio, so ``rtol`` means the same thing in
    every field's units; for fields well above ``atol / rtol`` the two
    formulas agree to within ``1 + atol/(rtol*|v|)``.

    Parameters
    ----------
    s_new : dict
        New iteration state.
    s_old : dict
        Previous iteration state.
    node_names : list of str
        Node names to include in the norm.
    atol : float
        Dead band, in the field's own units: below this a field counts
        as zero and leaves the norm, so the group is no longer held to
        any criterion on it.  ``CouplingGroup``'s default is ``0.0`` --
        see :func:`_scaled_change` for why the caller owns this number.
    rtol : float
        Relative change demanded of every field above the dead band.

    Returns
    -------
    jnp.ndarray
        Scalar RMS error norm.  Converged when <= 1.0.
    """
    sum_sq = jnp.array(0.0)
    count = jnp.array(0, dtype=jnp.int32)
    for nn in node_names:
        for field_name in s_new[nn]:
            new_val = s_new[nn][field_name]
            old_val = s_old[nn][field_name]
            if not _is_float_leaf(new_val):
                continue        # counters / flags / keys: not part of the norm
            if jnp.asarray(new_val).size == 0:
                continue
            scaled, active = _scaled_change(new_val, old_val, atol, rtol)
            sum_sq = sum_sq + jnp.sum(scaled ** 2)
            count = count + jnp.where(active, scaled.size, 0)
    return jnp.sqrt(sum_sq / jnp.maximum(count, 1))


def coupling_residual_interface(
    s_new: dict[str, dict],
    s_old: dict[str, dict],
    interface_edges: list,
    atol: float = 0.0,
    rtol: float = 1e-6,
) -> jnp.ndarray:
    """Interface consistency, on the scale of each interface quantity.

    Computes the difference in interface values (edge source fields)
    between two successive iterations.  Only the fields that appear
    on intra-group edges are compared.  The scaling is the one
    :func:`coupling_residual_mixed` documents: relative to the
    quantity's own magnitude, with ``atol`` as a dead band rather than
    as a floor under the scale.

    Parameters
    ----------
    s_new : dict
        New iteration state.
    s_old : dict
        Previous iteration state.
    interface_edges : list of EdgeSpec
        Edges internal to the coupling group.
    atol : float
        Dead band, in the interface quantity's own units: below this a
        quantity leaves the norm and stops being held to any criterion.
        ``CouplingGroup``'s default is ``0.0``; see
        :func:`_scaled_change` for why the caller owns this number.
    rtol : float
        Relative change demanded of every interface quantity above the
        dead band.

    Returns
    -------
    jnp.ndarray
        Scalar RMS error norm.  Converged when <= 1.0.
    """
    sum_sq = jnp.array(0.0)
    count = jnp.array(0, dtype=jnp.int32)
    for edge in interface_edges:
        new_val = s_new[edge.source_node][edge.source_field]
        old_val = s_old[edge.source_node][edge.source_field]
        if not _is_float_leaf(new_val):
            continue            # an integer interface field cannot carry a norm
        if edge.transform is not None:
            new_val = edge.transform(new_val)
            old_val = edge.transform(old_val)
        if jnp.asarray(new_val).size == 0:
            continue
        scaled, active = _scaled_change(new_val, old_val, atol, rtol)
        sum_sq = sum_sq + jnp.sum(scaled ** 2)
        count = count + jnp.where(active, scaled.size, 0)
    return jnp.sqrt(sum_sq / jnp.maximum(count, 1))


# ------------------------------------------------------------------
# Distance to the fixed point, estimated from the residual sequence
# ------------------------------------------------------------------

def error_amplification(residual, prev_residual, prev2_residual=None):
    """Estimate ``1 / (1 - rho)`` from the last two or three residuals.

    For a linear contraction with rate ``rho``, the distance from the
    current iterate to the fixed point is bounded by
    ``||x_k - x*|| <= r_k / (1 - rho)`` (sum the remaining steps of a
    geometric series), and ``rho`` is free: it is ``r_k / r_{k-1}``.

    That one-step ratio is the estimate the brief calls unreliable, and
    the case that breaks it is *alternation*, not growth.  A non-normal
    group whose residuals run ``0.5, 5, 0.25, 2.5`` is converging at
    ``rho = 0.71`` per pass, but every second one-step ratio reads
    ``0.05`` and flatters the bound by a factor of fourteen.  So the
    rate taken is the worst of the one-step ratio and the two-step
    ``sqrt(r_k / r_{k-2})``, which is the same number on a monotone
    geometric sequence and is immune to alternation:

        rho = max(r_k / r_{k-1}, sqrt(r_k / r_{k-2}))

    Written as ``r_{k-1} / (r_{k-1} - r_k)`` where it can be, so the
    cancellation happens between two measured numbers rather than
    against 1.

    ``prev2_residual`` defaults to ``prev_residual``, which is what the
    first pass of a loop has; the two-step term is then
    ``sqrt`` of the one-step one, i.e. slightly conservative, which is
    the right way to be wrong about a rate nothing has confirmed yet.

    Returns ``0.0`` — an impossible amplification, since a valid one is
    always ``>= 1`` — when the estimate must be rejected: a
    non-decreasing residual (``rho >= 1``, so there is no contraction
    to extrapolate), a zero or non-finite predecessor, or a non-finite
    current residual.  Callers fall back to the raw residual test and
    report that they did; see
    ``GraphManager.coupling_diagnostics``' ``ratio_usable``.  Rejecting
    is deliberate: a trusted bad estimate is worse than an honest
    fallback, and the fallback is exactly the criterion that shipped
    before 0.4.0.

    **What a non-rejected rate does not promise.**  This rate describes
    the mode that dominates the *step*, which is not always the mode
    that dominates the remaining error.  On a two-mode contraction the
    residual sequence is a clean geometric decay at the fast rate until
    the fast mode's amplitude falls below the slow one's, and over that
    stretch it is *indistinguishable* from a single-mode decay — the
    consecutive ratios are stationary, so the ``sqrt`` term above
    agrees with the one-step term and a longer window would agree with
    both.  Measured on modes ``(0.999, 0.2)``: ``rho`` reads 0.2 while
    the distance still to travel is 122x the estimate that rate
    produces.  Nothing computable from the residual norms alone
    separates that from a genuine 0.2 contraction; it needs the
    spectrum.  So a rate this function accepts is an estimate, and
    ``ratio_usable`` (renamed from ``bound_valid`` during 0.4.0's
    development, for exactly this reason; no release carried the old
    name) reports a usable *ratio*, not a valid *bound*.  The
    full list of what the estimate rests on is in
    ``graph_manager._fixed_point_while``; the decision it feeds is in
    ``benchmarks/results/audit_040_final/ERROR_BOUND_DECISION.md``.

    **The spectrum is measured, beside this, under ``solver="ift"``
    with ``diagnostics=True``.**  :func:`arnoldi_spectral_radius` takes
    eight Jacobian-vector products of the one-pass map at the returned
    iterate and :func:`spectral_error_bound` turns them into
    ``coupling_diagnostics()['spectral_error_bound']``, which on the
    same two-mode case reads 8x *over* the true distance where this
    rate's estimate reads 122x under.  This function is unchanged
    because the criterion both solvers stop on is built from it and
    438 recorded step verdicts depend on that; the spectral keys are
    reported next to it, not in place of it.
    """
    if prev2_residual is None:
        prev2_residual = prev_residual
    finite = jnp.logical_and(
        jnp.isfinite(residual),
        jnp.logical_and(jnp.isfinite(prev_residual),
                        jnp.isfinite(prev2_residual)),
    )
    positive = jnp.logical_and(prev_residual > 0, prev2_residual > 0)
    usable = jnp.logical_and(finite, positive)
    safe1 = jnp.where(usable, prev_residual, jnp.ones_like(prev_residual))
    safe2 = jnp.where(usable, prev2_residual, jnp.ones_like(prev2_residual))
    rho = jnp.maximum(residual / safe1, jnp.sqrt(residual / safe2))
    ok = jnp.logical_and(usable, rho < 1)
    den = jnp.where(ok, 1.0 - rho, jnp.ones_like(rho))
    return jnp.where(ok, 1.0 / den, jnp.zeros_like(residual))


def relaxation_step_scale(acceleration: str, relaxation: float) -> float:
    """How much longer the iterate's step is than the measured residual.

    The residual every acceleration reports is ``||F(x) - x||``, but
    what the iterate actually moves is ``||x_next - x||``, and the two
    are only the same under ``acceleration="none"``.  Constant
    relaxation moves ``omega`` times as far
    (``x + omega * (F(x) - x)``), so a geometric series of *residuals*
    is short of the distance the iterate still has to travel by exactly
    ``omega``.  Over-relaxation therefore made ``estimated_error``
    understate: measured ``est/true`` tracked ``1/omega`` to three
    figures (0.68 at ``omega=1.5``, 0.51 at ``omega=1.95``) on an
    affine two-node group.  Returning ``omega`` here is what puts the
    series back on the step the iteration takes.

    ``1.0`` for every other acceleration, and that is *not* the same
    statement for each of them:

    * ``"none"`` — exact, the step is the residual.
    * ``"aitken"`` — **an underestimate**, and a known one.  Aitken's
      relaxation factor is re-derived each pass and clipped to
      ``[0.01, 2.0]``; on the same affine group it saturates at 2.0 and
      the estimate understates by 2.04x.  It is not corrected here
      because the factor is dynamic: it would have to be carried
      through both solvers' loop state and the ``_meta`` diagnostics
      payload, and the value that matters is the one the *next* step
      will use, which nothing has measured.  A static ``2.0`` would be
      a bound but would tighten the criterion for every Aitken group.
    * ``"iqn-ils"`` / ``"iqn-imvj"`` — the quasi-Newton step is not a
      scalar multiple of ``F(x) - x`` at all, so no scale exists.  The
      estimate understates there too (4.5x measured), but by the
      *rate* mechanism rather than this one: a superlinear residual
      sequence reads ``rho -> 0``, so the amplification collapses to 1
      while the true remaining error is still ``1/(1 - rho_spectral)``
      of the residual.

    See ``benchmarks/results/audit_040_final/ERROR_BOUND_DECISION.md``.
    """
    return float(relaxation) if acceleration == "fixed" else 1.0


def estimated_error(residual, amplification, step_scale=1.0):
    """``residual * step_scale * amplification``, floored at ``residual``.

    The quantity a convergence criterion should be testing: an estimate
    of ``||x - x*||`` in the group's own norm, rather than of how far
    the last pass moved.

    ``step_scale`` is :func:`relaxation_step_scale` -- the ratio of the
    step the iterate takes to the residual that is measured.  The
    geometric series being summed is over *steps*, so leaving it out
    understated the distance by ``omega`` under over-relaxation.

    Still never smaller than ``residual``, so a group that meets this
    criterion also meets the raw residual test it replaces.  The floor
    binds only under *under*-relaxation of a strongly oscillatory mode
    (``step_scale * amplification < 1`` needs ``rho < 1 - omega``,
    reachable only for a negative eigenvalue), where it keeps the
    compatibility guarantee at the cost of being conservative -- the
    safe direction.
    """
    scaled = jnp.asarray(step_scale) * amplification
    return residual * jnp.maximum(scaled, jnp.ones_like(scaled))


def convergence_criterion(group) -> tuple[float, float]:
    """``(threshold, step_scale)``: what a coupling group's ``converged`` compares.

    A group has converged when
    ``estimated_error(residual, amplification, step_scale) <= threshold``.
    ``threshold`` is ``tolerance`` under the L2 norm and ``1.0`` under
    the mixed and interface norms, whose tolerances are inside the
    norm.  ``step_scale`` is :func:`relaxation_step_scale`.

    ``GraphManager.coupling_diagnostics()``, the profiler's
    ``converged_fraction`` and ``sysid.windowed_loss(mask_unconverged=True)``
    re-derive ``converged`` from a step's ``_meta`` slots, and all three
    read the criterion from here.  The profiler used to test
    ``residual * amplification`` without the step scale, so under
    ``acceleration="fixed"`` it overstated convergence when over-relaxed
    and understated it when under-relaxed.
    """
    threshold = (1.0 if group.convergence_norm in ("mixed", "interface")
                 else float(group.tolerance))
    return threshold, relaxation_step_scale(group.acceleration, group.relaxation)


def reported_error_estimate(residual: float, amplification: float,
                            step_scale: float) -> float:
    """:func:`estimated_error` on a reported ``(residual, amplification)`` pair.

    Host-side arithmetic in Python floats, for readers of a step's
    ``_meta`` slots.  There ``amplification`` is ``1 / (1 - rho)`` when
    the contraction ratio was usable and ``0.0`` when it was rejected, so
    anything below one falls back to the raw residual.  This is the
    number ``coupling_diagnostics()`` reports as ``"error_estimate"``,
    and ``<= threshold`` (see :func:`convergence_criterion`) is its
    ``"converged"``.
    """
    if amplification >= 1.0:
        return residual * max(step_scale * amplification, 1.0)
    return residual


# ------------------------------------------------------------------
# The spectral bound (``solver="ift"``, ``diagnostics=True``)
# ------------------------------------------------------------------

#: Arnoldi steps per group per timestep, i.e. the number of
#: Jacobian-vector products the spectral bound costs.  Each step is one
#: ``jax.jvp`` of the group's one-pass map -- roughly the price of one
#: coupling pass.  A coupling Jacobian's rank is at most the number of
#: boundary scalars that cross the group's edges, and a Krylov space of
#: that dimension *is* the Jacobian's range, so eight steps give the
#: exact non-zero spectrum of any group with up to eight independent
#: interface scalars and report, through the Arnoldi residual, when a
#: group has more (see :func:`arnoldi_spectral_radius`).
SPECTRAL_KRYLOV_STEPS = 8

#: How many times the Arnoldi residual ``h_{k+1,k}`` is added to the
#: Ritz spectral radius before the spectral-radius form of the bound is
#: formed.  That residual is the norm of the part of ``A q_k`` the
#: Krylov space does not contain, in the same units as the eigenvalues;
#: for a normal ``A`` every Ritz value lies within it of a true
#: eigenvalue (Bauer-Fike with constant one).  It is zero, up to float32
#: rounding, when the Krylov space is invariant, so the margin costs a
#: resolved spectrum nothing.
SPECTRAL_MARGIN = 2.0

#: The convergence test behind ``spectral_usable``: the Arnoldi
#: residual as a fraction of the gap ``1 - rho`` the bound divides by.
#: A shift of the dominant eigenvalue by the residual would move the
#: bound by this fraction of itself, so it is the quantity that decides
#: whether the number is settled to a few percent.
SPECTRAL_SETTLED_FRACTION = 0.05

#: Arnoldi breakdown threshold, relative to ``||A q_j||``.  Below it
#: the new direction is float32 noise left over from orthogonalising a
#: vector that lay in the Krylov space, and normalising noise into a
#: basis vector would let a *non-normal* Jacobian's field of values --
#: which can exceed its spectral radius -- leak into the Ritz values.
#: The basis stops growing instead and the remaining Hessenberg columns
#: stay zero, which is the exact answer.
_ARNOLDI_BREAKDOWN_RTOL = 1e-5

#: Squarings in the Gelfand estimate of a small matrix's spectral
#: radius, ``||H^(2^J)||_F ** (1 / 2^J)``.  The estimate is never below
#: the true radius and its excess after ``J`` squarings is bounded by
#: ``2**-J`` times the log of the eventual squaring ratio, i.e. below
#: 1e-6 relative for anything float32 can distinguish.
_GELFAND_SQUARINGS = 24


def _spectral_radius_small(H, n_squarings: int = _GELFAND_SQUARINGS):
    """``rho(H)`` of a small dense matrix by repeated squaring, from above.

    Gelfand: ``rho(H) = lim ||H^m||^(1/m)``, and ``||H^m||_F >= rho^m``
    for every ``m``, so the truncated estimate is an upper bound that
    converges to the radius.  Computed in log space over normalised
    powers, ``log rho = log ||H|| + sum_j 2^-(j+1) log ||N_j^2||_F``,
    so neither ``0.999**(2**24)`` nor ``0.2**(2**24)`` is ever formed.
    Only matrix products and norms: it lowers on every backend, where a
    non-symmetric ``eigvals`` does not.  A nilpotent ``H`` (some
    ``N_j^2 == 0``) reports exactly ``0.0``.
    """
    H = jnp.asarray(H)
    c0 = jnp.linalg.norm(H)
    alive0 = c0 > 0
    N0 = jnp.where(alive0, H / jnp.where(alive0, c0, 1.0), H)
    log_rho0 = jnp.where(alive0, jnp.log(jnp.where(alive0, c0, 1.0)), 0.0)

    def body(j, carry):
        N, log_rho, alive = carry
        M = N @ N
        s = jnp.linalg.norm(M)
        alive = jnp.logical_and(alive, s > 0)
        safe = jnp.where(alive, s, 1.0)
        N = jnp.where(alive, M / safe, M)
        log_rho = log_rho + jnp.log(safe) / (2.0 ** (j + 1))
        return N, log_rho, alive

    _N, log_rho, alive = jax.lax.fori_loop(
        0, int(n_squarings), body, (N0, log_rho0, alive0),
    )
    return jnp.where(alive, jnp.exp(log_rho), jnp.zeros_like(log_rho))


def arnoldi_spectral_radius(matvec, v0, n_steps: int = SPECTRAL_KRYLOV_STEPS,
                            v_extra=None):
    """``(rho, residual, amplification)`` after ``n_steps`` of Arnoldi on ``dF/dx``.

    ``matvec(v)`` applies the coupling Jacobian ``dF/dx`` (at the point
    the caller chose, in the coordinates the caller's norm is taken in)
    to ``v``; ``v0`` is the start vector.  Modified Gram-Schmidt Arnoldi
    builds an orthonormal basis ``Q`` of the Krylov space and the
    Hessenberg matrix ``H = Q^T A Q``; ``rho`` is the spectral radius
    of ``H`` -- the largest Ritz value in modulus, taken by
    :func:`_spectral_radius_small` so the call lowers on every backend
    -- ``residual`` is ``h_{k+1,k}``, the norm of the part of ``A q_k``
    outside the space, and ``amplification`` is
    ``||(I - H)^{-1}||_2 = 1 / sigma_min(I - H)``, the norm of the
    compressed map's resolvent at 1.

    **Why the resolvent and not only the radius.**  For a normal ``A``
    the two agree, ``||(I - A)^{-1}|| = 1 / min|1 - lambda| <=
    1/(1 - rho)``.  A coupling Jacobian need not be normal, and the
    Jacobi map of a group in which one side responds strongly to the
    other and the other weakly back is measured *far* from it: its
    ``+/-lambda`` eigenvector pair is nearly parallel, an error of the
    shape "grid consistent with probes, both off" has a residual
    ``(1 - lambda**2)`` times its probe part while its size is the
    grid's response to that part, and ``residual / (1 - rho)`` read
    30-40x below the true distance on the heterogeneous benchmark
    fixture with ``rho`` exactly right.  When the Krylov space is
    invariant (``residual == 0``) it satisfies ``A Q = Q H``, so for
    any vector ``r`` in it ``(I - A)^{-1} r = Q (I - H)^{-1} Q^T r`` and
    ``||(I - A)^{-1} r|| <= amplification * ||r||`` holds *whatever*
    the eigenvectors do -- **for a vector in the space**, which an
    invariant space does not make every vector.

    **Why the residual is put in the space (``v_extra``).**  A Krylov
    space grown from one start vector is that vector's cyclic subspace,
    and it can break down -- ``residual == 0``, invariant, settled --
    while holding only part of the Jacobian's range: where an
    eigenvalue is repeated the minimal polynomial's degree is below the
    rank, and ``A = B (x) I_2`` broke down at dimension 2 with a range
    of dimension 4.  A zero residual was read as certifying that the
    residual being bounded lay in the space; it did not, and the bound
    read 0.92x the true distance with ``spectral_usable=True``.  Nor is
    the residual of a relaxed or accelerated iterate in the range at
    all (only ``x_k = F(x_{k-1})`` makes ``F(x_k) - x_k`` a Jacobian
    image).  So the vector the caller will apply the resolvent to is
    passed as ``v_extra``: whenever the space breaks down, the part of
    ``v_extra`` it does not yet contain becomes the next basis vector
    and the iteration continues from it (``H`` keeps a zero on its
    subdiagonal there, so its spectrum is the union of the blocks').  A
    space that then breaks down again is invariant *and* contains
    ``v_extra`` by construction, which is the certificate.  What is
    still outside the final space -- ``v_extra`` minus its projection,
    relative to its norm -- is folded into ``residual`` (the larger of
    the two is returned), so a ``v_extra`` the space never absorbed
    reads as an unsettled space rather than as a certified one.

    **When the answer is exact, and how it says so.**  A coupling
    Jacobian has rank at most the number of boundary scalars crossing
    the group's edges; once the Krylov space contains that range the
    next Arnoldi vector is zero (breakdown, see
    ``_ARNOLDI_BREAKDOWN_RTOL``), the remaining columns of ``H`` stay
    zero, the non-zero Ritz values are the non-zero eigenvalues of
    ``A`` exactly, and ``residual`` is ``0.0``.  A group with more
    independent interface scalars than ``n_steps`` gets Ritz values
    that lie, for a normal ``A``, inside the convex hull of the
    spectrum -- an estimate of ``rho`` *from below* -- and a
    ``residual`` that is not small, which is what
    :func:`spectral_rate_settled` reports and
    :func:`spectral_error_bound` adds a margin for.  Eight steps
    resolve any spectrum whatever its clustering where the rank allows
    it, which is where a power iteration of the same cost does not: on
    random symmetric contractions of dimension 6 with eigenvalues
    0.949 and 0.983 the power iteration read 0.949 after eight
    products, while Arnoldi's space of dimension 6 is the whole range.

    ``n_steps`` is static (a Python int) and is the number of
    Jacobian-vector products the call costs, with or without
    ``v_extra``; the SVD behind ``amplification`` is of a ``k x k``
    matrix and costs nothing beside them.  A zero ``v0``, or a Jacobian
    that annihilates the start (``A v0 = 0``), gives ``rho = 0.0``,
    ``residual = 0.0`` and ``amplification = 1.0`` when there is no
    ``v_extra`` to continue from: nothing is amplified, so nothing is
    extrapolated.  A zero ``v_extra`` (a stalled iterate's residual)
    adds nothing.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> A = jnp.diag(jnp.array([0.999, -0.2, 0.0]))
    >>> rho, res, amp = arnoldi_spectral_radius(lambda v: A @ v, jnp.ones(3), n_steps=8)
    >>> bool(abs(rho - 0.999) < 1e-5), bool(res < 1e-5), bool(abs(amp - 1000.0) < 1.0)
    (True, True, True)
    >>> B = jnp.kron(jnp.array([[0.0, 2.0], [0.02, 0.0]]), jnp.eye(2))   # every eigenvalue twice
    >>> r = jnp.array([1.0, 0.0, 0.0, 1.0])
    >>> _, res, amp = arnoldi_spectral_radius(lambda v: B @ v, jnp.array([1.0, 1.0, 1.0, 1.0]), v_extra=r)
    >>> exact = float(jnp.linalg.norm(jnp.linalg.solve(jnp.eye(4) - B, r)))
    >>> bool(res < 1e-5), bool(float(amp) * float(jnp.linalg.norm(r)) >= exact)
    (True, True)
    """
    if n_steps < 1:
        raise ValueError(
            f"arnoldi_spectral_radius: n_steps={n_steps} < 1; at least "
            "one Jacobian-vector product is needed."
        )
    v0 = jnp.asarray(v0)
    k = int(n_steps)
    n = v0.shape[0]
    dtype = v0.dtype

    def _unit(v):
        nrm = jnp.linalg.norm(v)
        ok = nrm > 0
        return jnp.where(ok, v / jnp.where(ok, nrm, 1.0), v)

    extra = None if v_extra is None else jnp.asarray(v_extra, dtype)

    def _outside(Q, v):
        """``v`` minus its projection on the rows of ``Q`` (twice, for float32)."""
        for _sweep in range(2):
            for i in range(Q.shape[0]):
                v = v - jnp.dot(Q[i], v) * Q[i]
        return v

    Q0 = jnp.zeros((k + 1, n), dtype).at[0].set(_unit(v0))
    H0 = jnp.zeros((k + 1, k), dtype)

    def body(j, carry):
        Q, H = carry
        w = matvec(Q[j])
        scale = jnp.linalg.norm(w)
        # Modified Gram-Schmidt against every basis vector; rows not yet
        # filled (and rows after a breakdown) are zero and contribute
        # nothing, which keeps the loop static in shape.
        for i in range(k + 1):
            h = jnp.dot(Q[i], w)
            w = w - h * Q[i]
            H = H.at[i, j].set(h)
        h_next = jnp.linalg.norm(w)
        grown = h_next > _ARNOLDI_BREAKDOWN_RTOL * scale
        H = H.at[j + 1, j].set(jnp.where(grown, h_next, 0.0))
        q_next = jnp.where(grown, w / jnp.where(grown, h_next, 1.0), 0.0)
        if extra is not None:
            # A breakdown with ``v_extra`` not yet in the space: continue
            # from the part of it the space lacks.  ``H[j + 1, j]`` stays
            # zero -- the new direction is not ``A q_j``'s.
            e = _outside(Q, extra)
            e_norm = jnp.linalg.norm(e)
            fresh = e_norm > _ARNOLDI_BREAKDOWN_RTOL * jnp.linalg.norm(extra)
            restart = jnp.logical_and(jnp.logical_not(grown), fresh)
            q_next = jnp.where(restart, e / jnp.where(restart, e_norm, 1.0), q_next)
        Q = Q.at[j + 1].set(q_next)
        return Q, H

    Q, H = jax.lax.fori_loop(0, k, body, (Q0, H0))
    Hk = H[:k, :k]
    rho = _spectral_radius_small(Hk)
    sigma = jnp.linalg.svd(jnp.eye(k, dtype=dtype) - Hk, compute_uv=False)
    sigma_min = sigma[-1]
    invertible = sigma_min > 0
    amplification = jnp.where(
        invertible, 1.0 / jnp.where(invertible, sigma_min, 1.0), jnp.inf,
    )
    residual = H[k, k - 1]
    if extra is not None:
        # The certificate, tested rather than assumed: the fraction of
        # ``v_extra`` outside the final space.  Zero (to float32) when the
        # space absorbed it; otherwise it reads as unresolved spectrum.
        ex_norm = jnp.linalg.norm(extra)
        missed = jnp.linalg.norm(_outside(Q[:k], extra)) / jnp.where(ex_norm > 0, ex_norm, 1.0)
        missed = jnp.where(missed > _ARNOLDI_BREAKDOWN_RTOL, missed, 0.0)
        residual = jnp.maximum(residual, missed)
    return rho, residual, amplification


#: Units of ``eps * max|field|`` per entry that a computed residual can
#: differ from the exact ``F(x) - x`` by -- the constant in
#: :func:`residual_precision_floor`.
#:
#: **Derivation.**  The residual a group reports is ``N(F~(x) - x)``,
#: with ``F~`` the one-pass map as float arithmetic evaluates it and
#: ``N`` the group's norm, which divides each field by its own
#: magnitude.  What a distance bound needs is ``N(F(x) - x)`` for the
#: *exact* map, and the two differ by at most ``N(F~(x) - F(x))``, the
#: map's evaluation error (triangle inequality).  Each entry of
#: ``F~(x)`` is at least rounded once at the output, which is half a
#: unit in the last place of a number no larger than the field's
#: magnitude, ``<= eps * max|field| / 2`` -- so a residual of exactly
#: ``0.0`` does not say ``F(x) == x``; it says the last pass moved no
#: entry by more than about one ulp, and a slow group can stall there
#: arbitrarily far from its fixed point (``(1 - rho) * |x - x*|`` below
#: half an ulp).  Measured, in units of ``eps * max|field|`` per entry
#: and in the norm the group reports (jaxlib 0.11.0, CPU, float32):
#:
#: * the evaluation error of a dense linear update ``A @ u + c``,
#:   ``n <= 6``, over 3 000 random normal contractions near their fixed
#:   points: at most **0.72** (median 0.18) -- this *includes* the
#:   output rounding;
#: * the disagreement between two compilations of the same pass (the
#:   ``"ift"`` and ``"fori"`` solvers' reported residuals over the
#:   480-cell sweep behind ``tests/core/test_coupling_solver_equivalence.py``):
#:   at most **0.82**.
#:
#: Their sum is 1.54; four units is 2.6x that, room for an evaluation
#: with a few more roundings per entry than one dense product (a flux
#: computed from two iterates).
#:
#: **It is per evaluation, not per pass.**  A composite map is several
#: evaluations, and its error grows with their number: explicit Euler
#: ``x <- x + h (T - x)`` in ``N`` sub-steps, measured the same way,
#: reaches 1.3 units at ``N = 4``, 2.6 at 10, 5.8 at 20, 15.6 at 50 and
#: 29.4 at 100 (worst of 2 000 draws each, jaxlib 0.11.0) -- every
#: sub-step whose increment is below half an ulp of its field is rounded
#: away.  While the floor was a flat four
#: units, a stalled relay built on such a node read a bound 0.07-0.96x
#: its true distance at ``N`` = 15-200 with ``spectral_usable=True``,
#: whether the node looped inside ``update`` or the framework
#: sub-cycled it.  So the floor is this constant times the number of
#: evaluations one coupling pass rounds like: the largest sub-cycling
#: divider times :meth:`SimulationNode.update_evaluations` in the group
#: (``maddening.core.graph_manager._group_evaluations``, a module-level
#: function; a pass evaluates no node more
#: often than that, so it rounds like at most that many single passes),
#: which is ``4N`` -- 16, 40, 80, 200 and 400 units against those
#: figures, the same 2.6x-or-more headroom the single evaluation has
#: and 14x at ``N = 100``, the price of one constant.  A node that loops
#: inside ``update`` without declaring it is counted as one evaluation,
#: and ``coupling_diagnostics`` withholds ``spectral_usable`` wherever
#: the residual is at the floor, where that count carries the bound.
#:
#: It is a model of the map's rounding, not a proof of it: a node whose
#: update cancels catastrophically -- a small output computed as the
#: difference of two large intermediates -- can exceed any fixed number
#: of ulps of its *output's* magnitude, and nothing outside the node can
#: see that.
PRECISION_FLOOR_ULPS = 4.0


def residual_precision_floor(state, node_names, convergence_norm="l2",
                             atol: float = 0.0, rtol: float = 1.0,
                             interface_edges=(), evaluations: float = 1.0):
    """The float resolution of a residual the group's norm reports at *state*.

    ``PRECISION_FLOOR_ULPS`` units of ``eps * max|field|`` in every
    entry the norm reads, measured in that norm: the size a residual can
    have that is float rounding and not motion (see
    :data:`PRECISION_FLOOR_ULPS` for the derivation of the constant).
    Because every 0.4.0 norm divides a field by its own magnitude, one
    unit of ``eps * max|field|`` per entry is ``eps`` per entry in the
    norm's coordinates, so:

    * ``"l2"``: ``C * sqrt(sum over the active fields of size * eps**2)``,
      i.e. ``C * eps * sqrt(n)`` over the ``n`` float entries it reads;
    * ``"mixed"`` and ``"interface"``: the RMS of ``eps / rtol`` over the
      active entries, i.e. ``C * eps / rtol`` for a single dtype;

    with ``C = PRECISION_FLOOR_ULPS`` and "active" exactly the rule the
    norm applies (:func:`_scaled_change`): a field inside the dead band
    contributes nothing to the residual and nothing here.  ``0.0`` when
    the norm reads no active field -- it then measures nothing, and the
    residual is ``0.0`` too.

    Parameters
    ----------
    state : dict
        ``{node: {field: array}}`` -- the state the residual describes.
    node_names : iterable of str
        The group's nodes (read under ``"l2"`` and ``"mixed"``).
    convergence_norm : {"l2", "mixed", "interface"}
        The group's norm.
    atol, rtol : float
        The group's dead band and relative tolerance (``rtol`` is not
        read by the L2 norm, whose threshold is ``tolerance``).
    interface_edges : iterable of EdgeSpec
        The group's internal edges (read under ``"interface"``).
    evaluations : float
        How many evaluations of the map one coupling pass rounds like:
        the floor is ``PRECISION_FLOOR_ULPS`` units *per evaluation*.
        ``GraphManager.coupling_diagnostics`` passes the largest
        ``sub-cycling divider * SimulationNode.update_evaluations()`` in
        the group; ``1.0`` is a pass that evaluates each node once.

    Returns
    -------
    jnp.ndarray
        Scalar, in the units ``residual`` is reported in.  Traceable, so
        it can be taken inside a jitted step as well as on the host.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> s = {"n": {"x": jnp.ones(4, jnp.float32), "k": jnp.int32(3)}}
    >>> eps = float(jnp.finfo(jnp.float32).eps)
    >>> round(float(residual_precision_floor(s, ["n"], "l2")) / eps, 4)   # 4 * sqrt(4)
    8.0
    >>> round(float(residual_precision_floor(s, ["n"], "mixed", rtol=1e-3)) / eps)  # 4 / rtol
    4000
    >>> float(residual_precision_floor(s, ["n"], "l2", atol=2.0))   # dead-banded: nothing read
    0.0
    >>> round(float(residual_precision_floor(s, ["n"], "l2", evaluations=100.0)) / eps)
    800
    """
    norm = str(convergence_norm)
    use_rtol = 1.0 if norm == "l2" else float(rtol)
    values = []
    if norm == "interface":
        for edge in interface_edges:
            v = state[edge.source_node][edge.source_field]
            if not _is_float_leaf(v):
                continue
            if edge.transform is not None:
                v = edge.transform(v)
            values.append(jnp.asarray(v))
    else:
        for nn in node_names:
            for field_name in state[nn]:
                v = state[nn][field_name]
                if _is_float_leaf(v):
                    values.append(jnp.asarray(v))
    values = [v for v in values if v.size > 0]
    if not values:
        return jnp.zeros((), jnp.float32)
    dtype = jnp.result_type(*[v.dtype for v in values])
    sum_sq = jnp.zeros((), dtype)
    count = jnp.zeros((), dtype)
    for v in values:
        _scaled, active = _scaled_change(v, v, atol, use_rtol)
        eps = float(jnp.finfo(v.dtype).eps) / use_rtol
        n = jnp.where(active, float(v.size), 0.0).astype(dtype)
        sum_sq = sum_sq + n * (eps * eps)
        count = count + n
    if norm != "l2":
        sum_sq = sum_sq / jnp.maximum(count, 1.0)
    return (PRECISION_FLOOR_ULPS * float(evaluations)) * jnp.sqrt(sum_sq)


def spectral_error_bound(residual, rho, arnoldi_residual, amplification=1.0,
                         margin: float = SPECTRAL_MARGIN, floor: Any = 0.0):
    """The distance to the fixed point, from the spectrum of ``dF/dx``.

    ``(residual + floor) * max(amplification, 1 / (1 - rho_safe))``,
    with ``rho_safe = rho + margin * arnoldi_residual`` and ``floor`` the
    residual's own float resolution (:func:`residual_precision_floor`;
    ``0.0`` by default, which is the exact-arithmetic statement).

    For a *linear* map ``F(x) = A x + b`` the error of any iterate is
    exactly ``x - x* = (A - I)^{-1} (F(x) - x)``, whatever iteration
    produced ``x``: no step sequence, no relaxation factor and no
    accelerator enters.  Its size is therefore at most
    ``||(I - A)^{-1}|| * residual``.  Two things stand in for that
    operator norm, and the larger is used:

    * ``amplification``, the resolvent norm ``||(I - H)^{-1}||_2`` of
      the Krylov-compressed Jacobian from
      :func:`arnoldi_spectral_radius`.  When the Krylov space is
      invariant this *is* the operator norm on that space, whatever the
      eigenvectors do, and the residual of a coupling iterate lies in
      it.  It is the term that holds on a non-normal map.
    * ``1 / (1 - rho_safe)``, the spectral-radius form: exact for a
      normal ``A`` with a positive dominant eigenvalue, conservative
      otherwise, and the only one of the two that can be pushed up when
      the Krylov space is *not* invariant -- the Arnoldi residual is
      zero when the space captured the Jacobian's range, so a resolved
      spectrum pays no margin, and where it is not zero the Ritz radius
      is an estimate from below and is inflated by the size of what the
      space missed.

    Neither is what :func:`error_amplification` could state: it read
    ``rho`` off the residual sequence, which reports the mode
    dominating the *step*; these come from ``dF/dx`` itself, which sees
    every mode whatever its current amplitude.  The result is ``inf``
    when ``rho_safe >= 1`` or the compressed ``I - H`` is singular --
    the raw iteration would not contract, so nothing is bounded -- and
    NaN when ``rho`` is NaN, which is how a solver that did not compute
    one reports it.  Never smaller than ``residual + floor`` where it
    is finite.

    **Why the floor is added and not compared.**  The operator bound is
    ``||x - x*|| <= amp * N(F(x) - x)`` for the *exact* map, and the
    residual that was measured is ``N(F~(x) - x)`` for the map float
    arithmetic evaluated.  The two differ by the evaluation error, so
    the measured residual bounds the exact one only after the floor is
    added: ``N(F(x) - x) <= residual + floor``.  Without it a float32
    iterate that has *stalled* -- ``F~(x) == x`` bitwise, because
    ``(1 - rho) * |x - x*|`` is below half an ulp -- reported a residual
    of ``0.0`` and a bound of ``0.0`` while thousands of ulps from its
    fixed point (38 348 ulps, a true distance of 4.2e-3, on a relay
    contracting at 0.99999).  With it the bound on that state is the
    floor times the amplification, which is what float32 can resolve
    about a group that slow; see :data:`PRECISION_FLOOR_ULPS`.  The
    floor term is amplified by the same factor as the residual, which
    is exact for a normal Jacobian whose non-zero spectrum the Krylov
    space resolved and an estimate otherwise: the rounding error's
    direction is not measured.

    **When it is a bound and when it is an estimate.**  It is a bound
    on ``||x - x*||`` in the group's norm under three conditions, each
    stated because each can fail: ``F`` is linear, or the iterate is
    close enough that ``dF/dx`` does not change between ``x`` and
    ``x*`` -- Ostrowski's theorem makes the statement *asymptotic* for
    a differentiable non-linear ``F``, and an estimate elsewhere; the
    Krylov space is invariant and the residual lies in it, which is the
    generic case once Arnoldi has broken down and is what a zero
    ``arnoldi_residual`` certifies (a group with more independent
    interface scalars than :data:`SPECTRAL_KRYLOV_STEPS` does not get
    there, :func:`spectral_rate_settled` says so, and only the
    spectral-radius form with its margin then stands); and the group's
    norm is close enough to a norm on the tail -- the weights it
    divides each field by are taken at the returned iterate and the
    dead band's excluded fields are outside it, the same conditions
    ``error_amplification`` documents.

    Examples
    --------
    >>> round(float(spectral_error_bound(1e-4, 0.999, 0.0, 1000.0)), 4)
    0.1
    >>> round(float(spectral_error_bound(1e-4, 0.5, 0.1)), 6)   # radius form: 0.5 + 2*0.1
    0.000333
    >>> round(float(spectral_error_bound(1e-4, 0.5, 0.0, 40.0)), 6)  # resolvent form wins
    0.004
    >>> float(spectral_error_bound(1e-4, 1.0, 0.0, 1.0))
    inf
    >>> round(float(spectral_error_bound(0.0, 0.999, 0.0, 1000.0, floor=1e-6)), 6)  # stalled
    0.001
    """
    residual = jnp.asarray(residual)
    rho = jnp.asarray(rho)
    arnoldi_residual = jnp.asarray(arnoldi_residual)
    amplification = jnp.asarray(amplification)
    floor = jnp.asarray(floor)
    dtype = jnp.result_type(residual, rho, arnoldi_residual, amplification, floor)
    residual = residual.astype(dtype)
    rho = rho.astype(dtype)
    arnoldi_residual = arnoldi_residual.astype(dtype)
    amplification = amplification.astype(dtype)
    floor = floor.astype(dtype)
    rho_safe = rho + margin * arnoldi_residual
    computed = jnp.logical_and(jnp.isfinite(rho_safe), jnp.isfinite(residual))
    contracting = jnp.logical_and(rho_safe < 1, jnp.isfinite(amplification))
    ok = jnp.logical_and(computed, contracting)
    den = jnp.where(ok, 1.0 - rho_safe, jnp.ones_like(rho_safe))
    amp = jnp.maximum(jnp.where(ok, amplification, 1.0), 1.0 / den)
    # The exact map's residual is at most the measured one plus the
    # evaluation error; see the docstring.
    measured = residual + floor
    bound = jnp.maximum(measured * amp, measured)
    nan = jnp.full_like(bound, jnp.nan)
    inf = jnp.full_like(bound, jnp.inf)
    return jnp.where(computed, jnp.where(contracting, bound, inf), nan)


def spectral_rate_settled(rho, arnoldi_residual, fraction: float = SPECTRAL_SETTLED_FRACTION):
    """Whether the Arnoldi residual was small against ``1 - rho``.

    ``arnoldi_residual <= fraction * (1 - rho)``.  The bound divides by
    ``1 - rho``, so a shift of the dominant eigenvalue by the residual
    moves the bound by that fraction of itself; this asks that the
    unresolved part of the spectrum was worth at most ``fraction`` of
    the bound.  False for a NaN ``rho`` (nothing was computed) and for
    ``rho >= 1`` (nothing is bounded).

    A zero residual means the Krylov space was invariant and the Ritz
    values are eigenvalues.  A small non-zero one means the space
    nearly was; it does not say which eigenvalue was approximated, so
    read this beside :func:`spectral_error_bound`'s conditions.

    Examples
    --------
    >>> bool(spectral_rate_settled(0.999, 0.0)), bool(spectral_rate_settled(0.9, 0.05))
    (True, False)
    """
    rho = jnp.asarray(rho)
    arnoldi_residual = jnp.asarray(arnoldi_residual, dtype=rho.dtype)
    gap = 1.0 - rho
    finite = jnp.logical_and(jnp.isfinite(rho), jnp.isfinite(arnoldi_residual))
    return jnp.logical_and(
        jnp.logical_and(finite, gap > 0), arnoldi_residual <= fraction * gap,
    )


# ------------------------------------------------------------------
# The IFT gradient's error at an early exit (``solver="ift"``,
# ``diagnostics=True``)
# ------------------------------------------------------------------

#: Relative Frobenius residual ``||J U - U M|| / ||J U||`` above which
#: :func:`jacobian_range_basis` reports that the basis did not capture
#: the Jacobian's range.  Rounding on a captured range is ~1e-6
#: relative in float32; a range larger than the basis leaves an O(1)
#: residual.  The gap is wide, and 1e-4 leaves room for a badly scaled
#: field without admitting a missed direction.
RANGE_CAPTURED_RTOL = 1e-4


def jacobian_range_basis(matvec, n, n_vectors: int = SPECTRAL_KRYLOV_STEPS,
                         dtype=jnp.float32, key=None, rtol: float = RANGE_CAPTURED_RTOL):
    """``(U, M, captured)``: an orthonormal basis of ``range(J)`` and ``J`` on it.

    ``matvec(v)`` applies the coupling Jacobian ``J = dF/dx`` (the same
    product :func:`arnoldi_spectral_radius` takes) to a vector of
    length ``n``.  ``n_vectors`` fixed-seed normal draws are pushed
    through it and the images orthonormalised (``U``, ``n`` by
    ``min(n, n_vectors)``); ``M = U^T J U`` is the compression of ``J``
    onto that span, and ``captured`` says whether the span is
    invariant, ``||J U - U M||_F <= rtol ||J U||_F``.

    A coupling Jacobian's rank is at most the number of boundary
    scalars crossing the group's edges, so for a group with at most
    ``n_vectors`` of them the images of ``n_vectors`` generic vectors
    span the whole range, ``J = U U^T J`` exactly, and ``captured`` is
    ``True``.  Unlike a Krylov space grown from one start vector, which
    finds only that vector's cyclic subspace (a rank-two ``J`` with a
    repeated eigenvalue gives it a one-dimensional space), a basis
    built from several starts spans the range whatever the spectrum's
    multiplicities -- which is what :func:`resolvent_apply` needs, and
    what :func:`arnoldi_spectral_radius` does not (a radius is read
    correctly off any invariant subspace).

    Costs ``n_vectors + min(n, n_vectors)`` Jacobian-vector products
    and one thin QR of an ``n x n_vectors`` matrix.  A zero Jacobian
    gives a basis of arbitrary directions, ``M = 0`` and
    ``captured = True``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> A = jnp.zeros((5, 5)).at[0, 1].set(0.5).at[1, 0].set(0.5)   # rank 2
    >>> U, M, ok = jacobian_range_basis(lambda v: A @ v, 5, n_vectors=3)
    >>> U.shape, M.shape, bool(ok)
    ((5, 3), (3, 3), True)
    >>> sorted(round(abs(float(x)), 4) for x in jnp.linalg.eigvalsh(0.5 * (M + M.T)))
    [0.0, 0.5, 0.5]
    """
    if n_vectors < 1:
        raise ValueError(
            f"jacobian_range_basis: n_vectors={n_vectors} < 1; at least one "
            "Jacobian-vector product is needed."
        )
    key = jax.random.PRNGKey(2) if key is None else key
    v0 = jax.random.normal(key, (int(n_vectors), int(n)), dtype)
    images = jax.vmap(matvec)(v0)                      # rows: J v
    U, _ = jnp.linalg.qr(images.T, mode="reduced")     # (n, k)
    JU = jax.vmap(matvec)(U.T).T                        # (n, k)
    M = U.T @ JU
    resid = jnp.linalg.norm(JU - U @ M)
    scale = jnp.linalg.norm(JU)
    captured = resid <= rtol * scale
    captured = jnp.logical_or(captured, scale == 0)
    return U, M, captured


def resolvent_apply(U, M, w, Jw):
    """``(I - J)^{-1} w`` from the range basis of :func:`jacobian_range_basis`.

    With ``range(J)`` inside ``span(U)`` and ``M = U^T J U``, the
    solution of ``(I - J) t = w`` is ``t = w + U z`` with
    ``(I - M) z = U^T J w`` -- the Woodbury identity for a low-rank
    ``J`` -- so a solve costs one Jacobian-vector product (``Jw``, which
    the caller supplies) and one ``k x k`` linear solve, whatever the
    state's dimension.  ``w`` and ``Jw`` are single vectors; ``vmap``
    over a batch of right-hand sides.  A singular ``I - M``
    (``rho(J) = 1``) gives non-finite entries, never a plausible
    vector.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> A = jnp.diag(jnp.array([0.5, -0.25, 0.0]))
    >>> U, M, _ = jacobian_range_basis(lambda v: A @ v, 3, n_vectors=3)
    >>> w = jnp.array([1.0, 1.0, 1.0])
    >>> [round(float(x), 4) for x in resolvent_apply(U, M, w, A @ w)]   # 1/(1 - lambda)
    [2.0, 0.8, 1.0]
    """
    k = M.shape[0]
    z = jnp.linalg.solve(jnp.eye(k, dtype=M.dtype) - M, U.T @ Jw)
    return w + U @ z


def ift_gradient_error_bound(amplification, distance, secant, step, tangent):
    """``amplification * distance * secant / (step * tangent)``: the IFT tangent's relative error.

    The arithmetic of ``coupling_diagnostics()['gradient_relative_error_bound']``,
    elementwise, with the conventions below.  The IFT tangent rule
    solves ``(I - J(x)) t = F_c(x) c_dot`` at the iterate the forward
    *returned*, ``x_k``, where the derivative of the fixed point is the
    same solve at ``x*``.  Subtracting the two equations gives, exactly,

        ``t_k - t* = (I - J(x*))^{-1} [G(x_k) - G(x*)]``,
        ``G(x) = J(x) t_k + F_c(x) c_dot``

    -- the resolvent applied to the change, between the two points, of
    the one-pass map's Jacobian-vector product along the tangent the
    rule returned.  Each factor here bounds one piece of that:

    * ``amplification`` bounds the resolvent: the factor
      :func:`spectral_error_bound` applies to a residual (the larger of
      ``||(I - H)^{-1}||_2`` and ``1 / (1 - rho_safe)``);
    * ``distance`` bounds ``||x_k - x*||``: :func:`spectral_error_bound`
      itself -- not the residual and not ``error_estimate``, which
      understate the distance by up to 122x;
    * ``secant / step`` is the change in ``G`` per unit distance: the
      norm of ``G(x_k + delta) - G(x_k)`` over the norm of the step
      ``delta`` it was measured across;
    * ``tangent`` is ``||t_k||``, so the result is relative.

    All norms are the group's, and ``distance``, ``secant``, ``step``
    and ``tangent`` must be taken in the *same* one: the result is not
    invariant to rescaling one of them alone.

    Conventions, so a number is never reported where none was computed:
    NaN where ``amplification`` or ``distance`` is NaN (nothing was
    computed), where ``tangent`` is zero (the fixed point does not
    respond to the probe, so a relative error is undefined), where the
    secant is not finite, or where ``step`` is zero with ``distance``
    not -- a curvature cannot be read off a zero step.  ``0.0`` where
    ``distance`` is ``0.0`` and finite everything else.  A distance
    taken from :func:`spectral_error_bound` with its ``floor`` is never
    ``0.0`` at a finite iterate the group's norm reads anything of --
    the floor is what float32 can resolve, and a residual of ``0.0``
    means only that the last pass moved less than it -- so the branch
    is reached only where the norm reads nothing and so measures
    nothing.  (Before 0.4.0 shipped it was reached on every stalled
    float32 iterate, and read ``0.0`` against true gradient errors of
    3% at ``F'(x*) = 0.999``.)  ``inf`` where
    ``amplification`` or ``distance`` is ``inf`` (nothing contracts, so
    nothing is bounded) and the rest is finite and non-zero.

    Examples
    --------
    >>> round(float(ift_gradient_error_bound(2.0, 0.01, 3e-3, 0.01, 1.0)), 6)
    0.006
    >>> float(ift_gradient_error_bound(2.0, 0.0, 0.0, 0.0, 1.0))   # a norm that reads nothing
    0.0
    >>> import math
    >>> math.isnan(float(ift_gradient_error_bound(2.0, 0.01, 0.0, 0.01, 0.0)))  # no response
    True
    >>> math.isinf(float(ift_gradient_error_bound(math.inf, math.inf, 3e-3, 0.01, 1.0)))
    True
    """
    amp = jnp.asarray(amplification)
    dist = jnp.asarray(distance)
    sec = jnp.asarray(secant)
    stp = jnp.asarray(step)
    tan = jnp.asarray(tangent)
    dtype = jnp.result_type(amp, dist, sec, stp, tan)
    amp, dist, sec, stp, tan = (a.astype(dtype) for a in (amp, dist, sec, stp, tan))
    responds = tan > 0
    measured = jnp.logical_and(jnp.isfinite(sec), jnp.isfinite(stp))
    computed = jnp.logical_and(~jnp.isnan(amp), ~jnp.isnan(dist))
    at_fixed_point = jnp.logical_and(dist == 0, jnp.isfinite(amp))
    curvable = stp > 0
    safe_den = jnp.where(jnp.logical_and(curvable, responds), stp * tan, 1.0)
    rel = sec / safe_den
    value = jnp.where(at_fixed_point, jnp.zeros_like(rel), amp * dist * rel)
    ok = computed & responds & measured & (at_fixed_point | curvable)
    return jnp.where(ok, value, jnp.full_like(value, jnp.nan))


# ------------------------------------------------------------------
# State flattening / unflattening
# ------------------------------------------------------------------

def flatten_coupled_state(
    state: dict[str, dict],
    node_names: list[str],
    fields: Optional[dict[str, tuple[str, ...]]] = None,
) -> jnp.ndarray:
    """Flatten coupled nodes' state fields into a single 1D vector.

    Fields are iterated in sorted order for determinism.

    Parameters
    ----------
    state : dict
        Nested state dict ``{node_name: {field: array}}``.
    node_names : list of str
        Which nodes to include.
    fields : dict or None
        If provided, only include the specified fields per node.
        ``{node_name: (field1, field2, ...)}``
        If None, include all fields.

    Returns
    -------
    jnp.ndarray
        1D vector of all field values concatenated.
    """
    parts = []
    for nn in node_names:
        if fields is not None:
            if nn not in fields:
                continue  # Skip nodes not in the fields dict
            field_list = sorted(fields[nn])
        else:
            field_list = sorted(state[nn].keys())
        for field in field_list:
            parts.append(jnp.ravel(state[nn][field]))
    return jnp.concatenate(parts)


# ------------------------------------------------------------------
# Exact float32 images of non-float leaves
# ------------------------------------------------------------------
#
# The IFT coupling solver closes over the pre-step state through
# ``jax.closure_convert``; an integer / boolean / PRNG-key constant in
# that closure breaks JAX's linearisation of the custom_jvp rule under a
# ``lax.scan``.  So such leaves travel as float32 *images* and are
# restored to their own dtype at the point of use.  A single float32
# holds 24 bits exactly, so the image is exact only if we split wider
# integers into 16-bit limbs (one leading axis of limbs, most
# significant first) and unpack typed PRNG keys into their uint32 data.

_IMAGE_SMALL = ("bool", "int8", "uint8", "int16", "uint16")


def float_image(v) -> tuple[jnp.ndarray, tuple]:
    """``(image, meta)``: a float32 array carrying ``v`` exactly.

    ``meta`` is what :func:`from_float_image` needs to rebuild ``v``:
    ``("float", dtype)`` (image is ``v`` itself), ``("small", dtype)``
    (one float32 per element), ``("limbs", dtype, n_limbs)`` (16-bit
    limbs on a new leading axis) or ``("key", impl, n_limbs)`` for a
    typed PRNG key.
    """
    v = jnp.asarray(v)
    dt = v.dtype
    if jnp.issubdtype(dt, jnp.floating):
        return v, ("float", dt)
    # `jax.dtypes.issubdtype` is documented public API but is not in the
    # submodule's `__all__`, so pyright reads it as a private import.
    if jax.dtypes.issubdtype(  # pyright: ignore[reportPrivateImportUsage]
            dt, jax.dtypes.prng_key):
        data = jax.random.key_data(v)               # uint32, shape (*v.shape, 2)
        img, (_, _, n) = float_image(data)
        return img, ("key", jax.random.key_impl(v), n)
    if str(dt) in _IMAGE_SMALL:
        return v.astype(jnp.float32), ("small", dt)
    if jnp.issubdtype(dt, jnp.integer):
        nbits = jnp.iinfo(dt).bits
        n = nbits // 16
        u = v.view(jnp.dtype(f"uint{nbits}"))          # bit pattern, no sign issues
        limbs = [((u >> (16 * (n - 1 - i))) & 0xFFFF).astype(jnp.float32) for i in range(n)]
        return jnp.stack(limbs, axis=0), ("limbs", dt, n)
    raise TypeError(
        f"cannot carry a leaf of dtype {dt} through the coupling solver; "
        "supported: floating, bool, integer, typed PRNG keys"
    )


def from_float_image(img, meta):
    """Inverse of :func:`float_image` (bit-exact)."""
    kind = meta[0]
    if kind == "float":
        return img if img.dtype == meta[1] else img.astype(meta[1])
    if kind == "small":
        return img.astype(meta[1])
    if kind == "limbs":
        _, dt, n = meta
        nbits = 16 * n
        udt = jnp.dtype(f"uint{nbits}")
        acc = jnp.zeros(img.shape[1:], udt)
        for i in range(n):
            acc = acc | (img[i].astype(udt) << (16 * (n - 1 - i)))
        return acc.view(dt)
    if kind == "key":
        _, impl, n = meta
        data = from_float_image(img, ("limbs", jnp.dtype("uint32"), n))
        return jax.random.wrap_key_data(data, impl=impl)
    raise ValueError(f"unknown image kind {kind!r}")


def state_float_image(state: dict) -> tuple[dict, dict]:
    """Per-field :func:`float_image` of a node state dict -> ``(images, metas)``."""
    imgs, metas = {}, {}
    for f, v in state.items():
        imgs[f], metas[f] = float_image(v)
    return imgs, metas


def state_from_float_image(imgs: dict, metas: dict) -> dict:
    return {f: from_float_image(v, metas[f]) for f, v in imgs.items()}


def unflatten_coupled_state(
    flat: jnp.ndarray | np.ndarray,
    template: dict[str, dict],
    node_names: list[str],
    fields: Optional[dict[str, tuple[str, ...]]] = None,
) -> dict[str, dict]:
    """Unflatten a 1D vector back into the nested state dict structure.

    Parameters
    ----------
    flat : jnp.ndarray
        1D vector produced by :func:`flatten_coupled_state`.
    template : dict
        State dict with the correct shapes (used as a template).
    node_names : list of str
        Which nodes were included in the flat vector.
    fields : dict or None
        If provided, only these fields per node are in the flat vector.

    Returns
    -------
    dict
        Nested state dict with restored shapes.
    """
    result: dict[str, dict[str, Any]] = {}
    offset = 0
    for nn in node_names:
        if fields is not None:
            if nn not in fields:
                continue  # Skip nodes not in the fields dict
            field_list = sorted(fields[nn])
        else:
            field_list = sorted(template[nn].keys())
        result[nn] = {}
        for field in field_list:
            tmpl = template[nn][field]
            shape = tmpl.shape
            size = 1
            for s in shape:
                size *= s
            part = flat[offset:offset + size].reshape(shape)
            # The flat vector is floating; restore the field's own dtype
            # so an integer / boolean leaf (a step counter, a flag) does
            # not come back as float32 after a coupled step — which is
            # both a semantic drift and a retrace of the jitted step.
            dtype = getattr(tmpl, "dtype", None)
            if dtype is not None and part.dtype != dtype:
                part = part.astype(dtype)
            result[nn][field] = part
            offset += size
    return result


# ------------------------------------------------------------------
# Acceleration methods
# ------------------------------------------------------------------

def aitken_relaxation(
    x_old_flat: jnp.ndarray,
    x_raw_flat: jnp.ndarray,
    prev_residual_flat: jnp.ndarray,
    omega: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Aitken delta-squared accelerated update.

    Aitken's method computes an optimal relaxation factor from
    two successive residuals::

        omega_{k+1} = -omega_k * (r_k . (r_{k+1} - r_k))
                      / ||r_{k+1} - r_k||^2

    Parameters
    ----------
    x_old_flat : jnp.ndarray
        Previous iteration state (flattened).
    x_raw_flat : jnp.ndarray
        Raw fixed-point result (flattened).
    prev_residual_flat : jnp.ndarray
        Residual from the previous iteration.  An all-zero vector is
        the caller's "no previous residual yet" sentinel (the coupling
        loops seed it that way on the first pass of every timestep);
        the formula needs two successive residuals, so that pass keeps
        ``omega`` unchanged instead of deriving one from the sentinel.
    omega : jnp.ndarray
        Current relaxation factor.

    Returns
    -------
    x_relaxed : jnp.ndarray
        Relaxed state vector.
    new_omega : jnp.ndarray
        Updated relaxation factor.
    residual : jnp.ndarray
        Current residual (for next iteration).
    """
    residual = x_raw_flat - x_old_flat
    delta_r = residual - prev_residual_flat
    denom = jnp.sum(delta_r ** 2)
    # Guard against zero/non-finite denominator.  When denom overflows to
    # inf in float32 (delta_r entries > ~1.84e19), the division produces
    # nan.  The isfinite check catches this and falls back to input omega.
    denom_ok = (denom > 1e-30) & jnp.isfinite(denom)
    # First pass of a timestep: ``prev_residual_flat`` is the zero
    # sentinel, so the numerator is identically 0 and the clip floor
    # (0.01) would silently override the caller's seeded omega -- the
    # loops seed omega=1.0 and then threw away 99% of the first
    # correction.  Treat the sentinel like the degenerate denominator.
    have_prev = jnp.any(prev_residual_flat != 0)
    usable = denom_ok & have_prev
    safe_denom = jnp.where(usable, denom, jnp.array(1.0))
    new_omega = -omega * jnp.sum(prev_residual_flat * delta_r) / safe_denom
    new_omega = jnp.clip(new_omega, 0.01, 2.0)
    # Fall back to current omega when the denominator is degenerate or
    # overflowed, or when there is no previous residual to extrapolate from.
    new_omega = jnp.where(usable, new_omega, omega)

    x_relaxed = x_old_flat + new_omega * residual
    return x_relaxed, new_omega, residual


def iqn_ils_update(
    x_raw_flat: jnp.ndarray,
    x_old_flat: jnp.ndarray,
    prev_residual: jnp.ndarray,
    prev_state: jnp.ndarray,
    V_mat: jnp.ndarray,
    W_mat: jnp.ndarray,
    n_cols: jnp.ndarray,
    omega: jnp.ndarray,
    prev_r_aitken: jnp.ndarray,
    *,
    have_prev,
) -> tuple:
    """IQN-ILS quasi-Newton update with Aitken fallback.

    Builds a low-rank approximation of the inverse Jacobian from
    residual and state differences across iterations.  Falls back
    to Aitken relaxation when no secant columns are available yet or
    the quasi-Newton step is invalid (NaN, or a blow-up).

    ``have_prev`` (bool scalar, keyword-only) says whether
    ``prev_residual`` / ``prev_state`` hold a real previous iterate.
    A new secant column is appended only when it is True; on the first
    iteration of a step it must be False, otherwise the seeded zeros
    enter the secant basis as a bogus column.  It is independent of
    ``n_cols`` so that warm-started columns (``jacobian_reuse``) can
    accelerate from the very first iteration.

    Parameters
    ----------
    x_raw_flat : jnp.ndarray
        Raw fixed-point result (flattened), shape ``(n_dof,)``.
    x_old_flat : jnp.ndarray
        Previous iteration state (flattened), shape ``(n_dof,)``.
    prev_residual : jnp.ndarray
        Residual from the previous iteration, shape ``(n_dof,)``.
    prev_state : jnp.ndarray
        Raw fixed-point result ``x_raw`` from the previous iteration,
        shape ``(n_dof,)`` (the sixth return value of the previous
        call).
    V_mat : jnp.ndarray
        Pre-allocated residual difference matrix, shape
        ``(n_dof, max_cols)``.
    W_mat : jnp.ndarray
        Pre-allocated state difference matrix, shape
        ``(n_dof, max_cols)``.
    n_cols : jnp.ndarray
        Number of active columns (int32 scalar).
    omega : jnp.ndarray
        Aitken relaxation factor (for fallback).
    prev_r_aitken : jnp.ndarray
        Previous Aitken residual (for fallback), shape ``(n_dof,)``.

    Returns
    -------
    x_new : jnp.ndarray
        Updated state vector.
    V_mat : jnp.ndarray
        Updated V matrix.
    W_mat : jnp.ndarray
        Updated W matrix.
    n_cols : jnp.ndarray
        Updated active column count.
    residual : jnp.ndarray
        Current residual.
    x_raw_flat : jnp.ndarray
        Current raw fixed-point result (becomes ``prev_state`` next
        iteration).
    new_omega : jnp.ndarray
        Updated Aitken omega.
    cur_r_aitken : jnp.ndarray
        Current Aitken residual.
    """
    residual = x_raw_flat - x_old_flat
    add_col = jnp.asarray(have_prev)

    # Secant columns (Degroote 2009): V holds residual differences, W
    # holds differences of the *raw operator outputs* x~.  The update
    # x_raw + W c with V c ~= -r then approximates the output at zero
    # residual.  Building W from input differences instead turns the
    # step into a hybrid that converges markedly slower on stiff
    # contractions (5 vs 2 iterations on the rho=0.98 test scene).
    delta_r = residual - prev_residual
    delta_x = x_raw_flat - prev_state

    # Shift existing columns right, add new column at position 0
    max_cols = V_mat.shape[1]
    new_V = jnp.where(
        add_col,
        jnp.roll(V_mat, shift=1, axis=1).at[:, 0].set(delta_r),
        V_mat,
    )
    new_W = jnp.where(
        add_col,
        jnp.roll(W_mat, shift=1, axis=1).at[:, 0].set(delta_x),
        W_mat,
    )
    new_n_cols = jnp.where(
        add_col, jnp.minimum(n_cols + 1, max_cols), n_cols,
    )

    # Mask inactive columns to zero
    col_mask = jnp.arange(max_cols) < new_n_cols
    V_masked = new_V * col_mask[None, :]
    W_masked = new_W * col_mask[None, :]

    # Solve min_c ||V c + r||_2 via the SVD pseudo-inverse.
    # This avoids the normal equations (V^T V) which square the condition
    # number of V — problematic in float32 near convergence when V columns
    # become nearly collinear.  ``pinv`` rather than ``lstsq``: the masked
    # matrix routinely carries several exactly-zero columns (inactive or
    # warm-started-but-empty), i.e. repeated zero singular values, and
    # lstsq's SVD derivative divides by ``s_i^2 - s_j^2`` there, so the
    # unrolled (fori) gradient came out NaN.  pinv's custom_jvp is
    # well-defined for rank-deficient input; the forward value is the
    # same minimum-norm solution with the same relative cutoff.
    c = jnp.linalg.pinv(V_masked, rtol=1e-6) @ (-residual)

    # QN correction
    correction = W_masked @ c + residual
    x_qn = x_old_flat + correction

    # Aitken fallback
    x_aitken, new_omega, cur_r_aitken = aitken_relaxation(
        x_old_flat, x_raw_flat, prev_r_aitken, omega
    )

    # Validate QN result: finite, and not a blow-up.  The bound is
    # deliberately loose: a correct quasi-Newton step is roughly
    # ``residual / (1 - rho)`` for a contraction of spectral radius
    # ``rho``, i.e. 50x the residual at rho = 0.98.  A tight cap (this
    # used to be 10x) silently vetoes IQN on exactly the stiff problems
    # it exists for and degrades it to Aitken.
    correction_norm = jnp.sqrt(jnp.sum(correction ** 2))
    residual_norm = jnp.sqrt(jnp.sum(residual ** 2))
    is_valid = (
        jnp.all(jnp.isfinite(x_qn))
        & (correction_norm < 1e6 * jnp.maximum(residual_norm, 1e-12))
        & (new_n_cols > 0)
    )
    x_new = jnp.where(is_valid, x_qn, x_aitken)

    return (
        x_new, new_V, new_W, new_n_cols,
        residual, x_raw_flat,
        new_omega, cur_r_aitken,
    )


def fixed_relaxation(
    x_old_flat: jnp.ndarray,
    x_raw_flat: jnp.ndarray,
    omega: float,
) -> jnp.ndarray:
    """Apply fixed (constant) under-relaxation.

    Parameters
    ----------
    x_old_flat : jnp.ndarray
        Previous iteration state (flattened).
    x_raw_flat : jnp.ndarray
        Raw fixed-point result (flattened).
    omega : float
        Relaxation factor.  ``omega=1.0`` is no relaxation,
        ``0 < omega < 1`` is under-relaxation,
        ``1 < omega < 2`` is over-relaxation.

    Returns
    -------
    jnp.ndarray
        Relaxed state vector.
    """
    return x_old_flat + omega * (x_raw_flat - x_old_flat)
