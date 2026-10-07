"""The fixed-point iteration of a coupled block (``_fixed_point_while``).

Moved verbatim out of ``maddening.core.graph_manager``.  Private.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from maddening.core._pow2_frame import pow2_frame


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


#: Accelerations that may not *stop iterating* on a single pass at or
#: below the threshold: the criterion has to hold on two consecutive
#: passes.  See :func:`_fixed_point_while` for the argument, and for
#: why IQN is not here.  Both coupling solvers read this list and both
#: seed the streak with the residual of the pass before their loop, so
#: ``"ift"`` and ``"fori"`` stop at the same pass at every cap.
_TWO_PASS_EXIT = ("aitken",)


def _fixed_point_while(
    step_pure, x0, consts, accel_init, first_res, threshold, max_iter,
    acceleration, relaxation, n_reuse, sub_idx, lagged_reading=False,
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
       **Under ``solver="ift"`` with ``diagnostics=True`` this one is
       measured rather than guessed**: eight Arnoldi steps on ``dF/dx``
       at the returned iterate, in the group's own norm coordinates,
       give ``rho_spectral`` and the resolvent norm of the compressed
       Jacobian, and ``coupling_diagnostics()['spectral_error_bound']``
       is the residual times the larger of that norm and
       ``1 / (1 - rho_spectral)`` (with a margin for an unresolved
       Krylov space).  For a linear ``F`` the error of *any* iterate is
       ``(A - I)^{-1}`` of its residual, so that is a bound whatever
       the step sequence, the relaxation or the accelerator did; for a
       non-linear ``F`` it is asymptotic (see
       :func:`~maddening.core.coupling.acceleration.spectral_error_bound`
       for the conditions).  On the same two-mode case it reports 8x
       *over* the true distance where ``error_estimate`` reports 122x
       under, and 1.3x over under ``aitken`` and ``iqn-ils``.  The
       criterion this loop stops on is unchanged; the spectral number
       is reported beside it, not applied.
    3. *A dynamic step scale.*  ``omega`` above is exact for
       ``acceleration="fixed"``, where relaxation is a constant, and
       for ``"none"``, where it is 1.  It is *not* corrected for
       Aitken's clipped per-pass factor (a measured 2.04x
       understatement when it saturates at 2.0) or for the IQN
       quasi-Newton step, which is not a multiple of ``F(x) - x``.
    4. *A non-monotone ratio* — the one that is caught.  ``rho >= 1``,
       a zero predecessor or a non-finite residual reject the estimate
       and ``ratio_usable`` records it.  A rejection is not proof of a
       non-monotone sequence: the ratio carries about
       ``2 floor / r_k`` of float rounding, so a contraction at a rate
       within that of 1 reads ``rho >= 1`` from noise, and the raw
       residual test decides (measured up to 200 thresholds from the
       fixed point at rate 0.995; MADD-ANO-005).

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
    different states.  ``tests/core/test_coupling_solver_equivalence.py``
    pins that both solvers return the state that was measured.

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

    **A lagged reading must settle twice too** (``lagged_reading``,
    static).  Under ``convergence_norm="interface"`` the residual reads
    what the group's internal edges deliver, and ``x_k``'s members were
    computed from the readings of ``x_{k-1}``: where a member takes an
    input from the previous iterate and has a field no edge delivers
    whole (``_state_lags_the_reading``), ``r_k <= threshold`` says the
    readings of ``x_k`` and of its successor agree and nothing about the
    readings that field was computed from.  A one-way pair under Jacobi
    stopped on its first pass with a residual of exactly zero -- the
    source does not depend on the iterate -- and returned a target
    computed from the pre-step source (MADD-ANO-235).  Such a group
    stops only when the previous residual is at or below ``threshold``
    as well, which is the statement that the readings ``x_k`` was
    computed from are within the threshold of the ones judged: the same
    streak as Aitken's, for a different reason.  At the cap the state
    returned is the successor, computed from the readings of the last
    iterate the loop measured, so the residual reported there is the
    larger of the extra evaluation's and the loop's last whenever the
    loop's last was above ``threshold`` -- the change of the readings
    over the pass that computed the state -- and the verdict every
    reader derives from it cannot be ``True`` on a state computed from
    readings that had not settled.  Every other group's criterion,
    report and program are unchanged.

    Returns ``(x_star, n_iters, final_res, final_amp, (V, W))``:
    ``n_iters`` is the number of coupling passes that produced
    ``x_star`` (an int32) -- the pre-loop
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
    # The accelerator's scalars and residual carries in at least float32,
    # as the fori path seeds them (``acc_dtype``), so a 16-bit group takes
    # the same passes under either solver; a float32 or wider group is
    # unchanged.
    acc_dt = jnp.promote_types(dtype, jnp.float32)
    zeros = jnp.zeros(n_dof, dtype=acc_dt)
    one = jnp.array(1.0, dtype=acc_dt)
    is_iqn = acceleration in ("iqn-ils", "iqn-imvj")
    # The accelerators work in a frame: the accelerated vector times one
    # exact power of two, fixed for the solve (its largest starting entry
    # in ``[0.5, 1)``), so their carries -- Aitken's previous residual,
    # IQN's secant columns and previous raw output -- are in one unit from
    # pass to pass.  A power of two scales every product exactly, so a
    # group at ordinary magnitudes steps to the bit as before; a group in
    # small units no longer forms its step from a difference below the
    # normal range (``x_raw - x_old`` flushed to zero below about 1e-38,
    # the relaxed iterate stopped moving while the norm still measured the
    # residual, the stalled ratio read 1 and was rejected, and the raw
    # residual test reported ``converged=True`` up to 7x the threshold
    # away; MADD-ANO-115).  Not formed without an accelerator, so the
    # plain loop's program is the one it was.
    frame = pow2_frame(x0_acc) if acceleration != "none" else None

    if acceleration in ("none", "fixed"):
        # Annotated: the three branches below build tuples of different
        # arity, and the empty one would otherwise fix the declared type
        # at `tuple[()]` for the `acc[0]`/`acc[1]` read after the loop.
        acc0: tuple = ()
    elif acceleration == "aitken":
        acc0 = (one, zeros)  # omega, prev_residual
    elif is_iqn:
        V0, W0 = accel_init
        # V, W, n_cols, prev_residual, prev_raw, omega, prev_r_aitken --
        # all in the frame (the secant columns arrive in state units).
        acc0 = (V0 * frame, W0 * frame, jnp.int32(n_reuse), zeros,
                x0_acc * frame, one, zeros)
    else:
        raise ValueError(
            f"_fixed_point_while: unsupported acceleration="
            f"{acceleration!r}; expected one of 'none', 'fixed', "
            "'aitken', 'iqn-ils', 'iqn-imvj'."
        )

    def accelerate(x, x_raw, acc, i):
        with jax.named_scope("coupling:accelerate"):
            x_new, new_acc = _accelerate(x, x_raw, acc, i)
            # The loop carry keeps its types: the step in the iterate's
            # dtype, each accelerator carry in its seed's.
            return (x_new.astype(x.dtype),
                    jax.tree.map(lambda new, old: jnp.asarray(new).astype(jnp.asarray(old).dtype),
                                 new_acc, acc))

    def _accelerate(x, x_raw, acc, i):
        if acceleration == "none":
            return x_raw, acc
        assert frame is not None  # formed for every acceleration but "none"
        # In the frame (see above); the step is scaled back on the way out.
        x, x_raw = x * frame, x_raw * frame
        if acceleration == "fixed":
            return fixed_relaxation(x, x_raw, relaxation) / frame, acc
        if acceleration == "aitken":
            omega, prev_r = acc
            x_new, omega, cur_r = aitken_relaxation(x, x_raw, prev_r, omega)
            return x_new / frame, (omega, cur_r)
        V, W, n_cols, prev_r, prev_s, omega, prev_ra = acc
        x_new, V, W, n_cols, cur_r, cur_s, omega, cur_ra = iqn_ils_update(
            x_raw, x, prev_r, prev_s, V, W, n_cols, omega, prev_ra,
            have_prev=i > 0,
        )
        return x_new / frame, (V, W, n_cols, cur_r, cur_s, omega, cur_ra)

    # See the docstring for why this list holds Aitken and not IQN.
    # An empty ``prev`` slot means the carry -- and so the emitted HLO
    # -- is unchanged for every acceleration that is not on it.
    # ``lagged_reading`` asks for the same streak; see the docstring.
    two_pass_exit = acceleration in _TWO_PASS_EXIT or lagged_reading

    from maddening.core.coupling.acceleration import (  # noqa: PLC0415
        first_pass_relaxed_amplification,
        relaxes_first_pass,
    )

    raw_amplification, error_of, step_scale_of = _bound_helpers()
    # Static: only an under-relaxed ``"fixed"`` loop reads which pass it
    # is on, so every other loop's program is the one it always was.
    relax_first = relaxes_first_pass(acceleration, relaxation)
    # Static: ``acceleration`` and ``relaxation`` are both nondiff
    # arguments of the custom_jvp, so this is a Python float and costs
    # nothing in the loop.
    step_scale = step_scale_of(acceleration, relaxation)

    def amplification(res, res_prev, res_prev2, first=False):
        """:func:`error_amplification`, relaxed on the loop's first pass.

        ``first`` says the ratio is the first loop pass's, whose
        predecessor is the unrelaxed pre-loop pass
        (:func:`~maddening.core.coupling.acceleration.first_pass_relaxed_amplification`;
        a no-op except under an under-relaxed ``"fixed"``).
        """
        return first_pass_relaxed_amplification(
            raw_amplification(res, res_prev, res_prev2),
            acceleration, relaxation, first)

    def _met(res, res_prev, res_prev2, first):
        """The stopping criterion: the *estimated distance to the fixed
        point* is at or below ``threshold``, not merely the last step."""
        est = error_of(res, amplification(res, res_prev, res_prev2, first),
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
        # ``i`` bodies have run; ``res`` is the first loop pass's at 1.
        above = jnp.logical_not(_met(res, res_prev, res_prev2,
                                     relax_first and i == jnp.int32(1)))
        keep_going = jnp.logical_and(above, i < max_iter - 1)
        return jnp.logical_or(first, keep_going)

    def body(carry):
        x, _x_meas, res_prev, res_prev2, _res_prev3, i, acc = carry
        x_raw, res = step_pure(x, *consts)
        # The residual carry is at least float32 (the seed below); a step
        # that hands back a narrower residual -- a 16-bit iterate's own
        # dtype -- is widened, exactly.  The graph's own step already
        # returns it at least float32 (``_group_residual_dtype``), so this
        # is the identity there.
        res = jnp.asarray(res).astype(acc_dt)
        if idx is None:
            x_new, acc = accelerate(x, x_raw, acc, i)
        else:
            x_new_sub, acc = accelerate(x[idx], x_raw[idx], acc, i)
            x_new = x_raw.at[idx].set(x_new_sub.astype(x_raw.dtype))
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
    # of the first ratio.  In the residual's dtype, at least float32
    # (``_group_residual_dtype``), which ``step_pure`` returns it in:
    # the iterate's own for a float32 or wider group.
    seed = jnp.asarray(first_res, acc_dt)
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
    on_first_pass = relax_first and n_iters == jnp.int32(1)
    criterion_met = _met(final_res, res_prev, res_prev2, on_first_pass)
    x_star = jnp.where(criterion_met, x_meas, x_next)
    loop_res = final_res

    def _measure_at_cap(_x):
        r = jnp.asarray(step_pure(_x, *consts)[1]).astype(acc_dt)   # as the body's
        amp = amplification(r, loop_res, res_prev)
        if lagged_reading:
            # ``_x`` was computed from the readings ``loop_res`` saw move;
            # see the docstring.
            r = jnp.where(loop_res > threshold, jnp.maximum(r, loop_res), r)
        return r, amp

    # ``final_amp`` has to describe the pair that ends on the state
    # being returned.  On the criterion that pair is
    # ``(res_prev, final_res)``; at the cap the extra evaluation makes
    # ``(final_res, res(x_star))`` the consecutive pair instead.
    # The criterion's pair is the first loop pass's exactly when the loop
    # stopped after one body; the cap's extra evaluation follows a
    # relaxed step, so its pair never is.
    final_res, final_amp = jax.lax.cond(
        criterion_met,
        lambda _x: (loop_res, amplification(loop_res, res_prev, res_prev2,
                                            on_first_pass)),
        _measure_at_cap,
        x_star,
    )
    # The secant matrices leave the frame for the warm start in ``_meta``.
    vw = (acc[0] / frame, acc[1] / frame) if is_iqn else ()
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
    # Returned as the int32 it is counted in.  It used to be cast to the
    # group's floating dtype for the diagnostics carry, which rounds every
    # count above 256 in bfloat16 (2048 in float16) to that dtype's grid:
    # a 16-bit group that ran its whole budget of 257 passes reported 256,
    # and the documented cap check ``iterations >= max_iterations`` was
    # false at the cap.  The custom_jvp gives it a ``float0`` tangent.
    return x_star, n_passes, final_res, final_amp, vw
