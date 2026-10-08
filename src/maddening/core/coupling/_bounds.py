"""Spectral-rate estimates and the gradient error bound of a coupled block.

Moved verbatim out of ``maddening.core.graph_manager``.  Private, apart
from ``GRADIENT_PROBE_ENTRY_LIMIT``, which stays importable from
``maddening.core.graph_manager``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core._pow2_frame import pow2_frame
from maddening.core._graph_specs import _META_KEY


def _F_dispatch(step_pure, x, consts):
    # Trampoline: forwards to a closure-converted pure function and
    # keeps only the next iterate.  Kept top-level so the custom_jvp
    # rule sees ``step_pure`` as a plain Python global, not a captured
    # closure.
    return step_pure(x, *consts)[0]


#: The largest relative gap between the pass's Jacobian-vector product
#: along a geometry and a finite difference of the pass along the same
#: direction (:func:`_geometry_product_gap`) at which
#: ``coupling_diagnostics()`` still reports the bounds of a group with a
#: geometry edge.  Experimental.  A product that misses a term which
#: moves a field by ``G`` reads ``G / (G + c res)`` on that field (``res``
#: the field's float resolution, ``c``
#: :data:`_GEOMETRY_GAP_RESOLUTIONS`): 1 where the term is strong, and
#: under this tolerance where it moves the field by fewer than about 11
#: resolutions.  One that sees half of a strong term reads 0.5.  An
#: honest pass reads the finite difference's own error, a few
#: ``sqrt(eps)`` of the coarsest dtype the pass evaluates in.
#:
#: Measured on jaxlib 0.10.2, 0.11.0 and 0.11.2 (MAP-045):
#:
#: * honest, 350 drawn examples of six cells of the geometry search: a
#:   median of 6e-4 and at most 5.7e-3 in float32, at most 4.3e-6 in
#:   float64; the search's hunts away from a lattice plane (2171
#:   examples, and a thousand plane draws further than 2e-5 of a spacing
#:   from their plane): at most 9.5e-3;
#: * every geometry read under ``stop_gradient``, the same six cells: 0.305
#:   to 0.98 in float32 and 1.0 in float64, all 353 withheld;
#: * the source-anchored reads alone under ``stop_gradient``, on a deposit
#:   that moves the grid's field by nine resolutions (and the spectral
#:   radius from 0.025 to 0.003): 0.22, under this tolerance.
#:
#: It is not lower, to catch that, because honest float32 passes of two
#: kinds read as much and more, and their reports would be withheld:
#:
#: * a pass that reads a position it built in the same sweep
#:   (Gauss-Seidel, a source-anchored geometry whose holder is swept
#:   first) with that position within about 2e-5 of a spacing of a
#:   lattice plane.  The step moves it across the plane through its
#:   holder's mapped input.  Of 2829 plane draws of two such cells on
#:   jaxlib 0.11.0 (1489 within 2e-5), 47 read over 0.05, 13 over 0.1, 2
#:   over 0.2 and one 0.59; one draw of the plane hunt reads 0.26 and
#:   0.29 on 0.10.2 and 0.11.2 (MADD-ANO-246);
#: * a Gauss-Seidel pass behind a gather of a field that changes sign
#:   across a cell: 0.025 to 0.087 where the lattice values are up to a
#:   thousand times the sample, 0.14 to 0.75 beyond (MADD-ANO-212, whose
#:   bound is wrong there, so that withholding is no loss).
#:
#: At 0.05 the self-check would withhold 7 to 9 of the 4,450 honest
#: examples of the search's hunts (all of the first kind) where it now
#: withholds none or one.  The reason of a withheld report names both
#: readings of a gap.
GEOMETRY_GAP_TOLERANCE = 0.25  # units: relative gap
#: How many of the residual's float resolutions the finite difference of
#: :func:`_geometry_product_gap` is allowed as rounding, per field: a
#: field the geometry moves by less than this is not compared.
_GEOMETRY_GAP_RESOLUTIONS = 32.0  # units: resolutions


def _geometry_product_gap(step_pure, x_star, consts, directions, weights, resolution,
                          field_ids, n_fields):
    """How far the pass's Jacobian-vector product along a geometry is from
    a finite difference of the pass along the same direction (experimental).

    The spectral estimate and the gradient bound of a group are built from
    Jacobian-vector products of its pass ``F``.  Where an edge's mapping
    reads a moving geometry, ``F`` depends on the iterate and on the
    pre-step state *through* that geometry, and a product that did not
    see it would be the product of another map: a spectral radius read
    low with every flag set.  This is the check that it does, on the
    object the estimates differentiate (``step_pure``).

    *directions* is a sequence of ``(dx, dconsts)``: a step of the flat
    iterate and one per constant (``None`` where a constant is not
    moved), each moving only geometry the pass reads -- the positions
    held in the iterate, and those held in a constant of the pass (a
    member's pre-step state, a node outside the group).  For each, with
    ``t`` the step as the floats realise it (``(x + dx) - x``),

        ``gap_f = ||D (F(x + t) - F(x) - J t)||_f
                  / (max(||D (F(x + t) - F(x))||_f, ||D J t||_f) + c ||res||_f)``

    per field ``f`` of the state (*field_ids* maps an entry to its field;
    a term in one field is not diluted by another's), ``D`` the *weights*,
    ``res`` the residual's float *resolution* and ``c``
    :data:`_GEOMETRY_GAP_RESOLUTIONS`.  Returns the largest ``gap_f`` over
    the fields and the directions, in the analysis dtype: 0 where the
    geometry moves nothing, NaN where it cannot be evaluated.  One
    Jacobian-vector product and one evaluation of the pass per direction.
    """
    x = jax.lax.stop_gradient(x_star)
    cc = tuple(jax.lax.stop_gradient(c) for c in consts)
    work = _analysis_dtype(x.dtype)
    d = jax.lax.stop_gradient(jnp.asarray(weights, work))
    res = jax.lax.stop_gradient(jnp.asarray(resolution, work))
    ids = np.asarray(field_ids, np.int32)

    def by_field(v):
        return jnp.sqrt(jax.ops.segment_sum(v * v, ids, num_segments=n_fields))

    noise = _GEOMETRY_GAP_RESOLUTIONS * by_field(res)
    worst = jnp.zeros((), work)
    for dx, dconsts in directions:
        x1 = x + jax.lax.stop_gradient(dx)
        c1 = tuple(c if t is None else c + jax.lax.stop_gradient(t)
                   for c, t in zip(cc, dconsts))
        tangents = tuple(jnp.zeros_like(c) if t is None else moved - c
                         for c, t, moved in zip(cc, dconsts, c1))
        base, product = jax.jvp(lambda x_, *c_: _F_dispatch(step_pure, x_, c_),
                                (x, *cc), (x1 - x, *tangents))
        difference = (_F_dispatch(step_pure, x1, c1) - base).astype(work)
        product = product.astype(work)
        miss = by_field(d * (difference - product))
        size = jnp.maximum(by_field(d * difference), by_field(d * product)) + noise
        # 0 / 0 (a field the geometry does not move, with no rounding
        # allowed) is no gap; a size or a miss that is not a number is one
        # that could not be evaluated, and ``max`` must not drop it.
        gap = miss / jnp.where(size > 0, size, 1.0)
        gap = jnp.where(jnp.all(jnp.isfinite(size)) & jnp.all(jnp.isfinite(gap)),
                        jnp.max(gap), jnp.nan)
        worst = jnp.where(jnp.isnan(gap) | jnp.isnan(worst), jnp.nan, jnp.maximum(worst, gap))
    return worst


#: How many times ``spectral_error_bound`` every position the pass reads
#: from the iterate must be from the nearest lattice plane of its mapping
#: for the bound's flag to stand (:func:`_geometry_plane_limit`).
#: Experimental.  Two is the radius of the Newton-Kantorovich ball in
#: units of the Newton step at the largest nonlinearity the theorem
#: admits (``t* <= 2 eta`` at ``h = 1/2``), and the bound is at least the
#: Newton step.
GEOMETRY_PLANE_REACH = 2.0  # units: spectral_error_bound
#: How many float resolutions of a position a lattice plane must be away
#: from it for the plane rule to take the position as off the plane
#: (:func:`_reader_plane_distance`).  Experimental.  Nearer, the position
#: counts as *on* the plane: its distance is zero.  The kernel decides
#: the lattice cell from a rounded quotient and a pass builds a position
#: with a rounding of its own, so which cell's polynomial the floats
#: evaluated is not decided by the position's value there: an iterate 4.7
#: resolutions before a plane with its fixed point 0.7 past it kept both
#: flags on a gradient bound 25x under the reference's error
#: (MADD-ANO-251).  The resolution is the mapping's
#: (``_plane_resolution``): ``eps`` of the position's dtype times the
#: largest coordinate magnitude of the lattice's axis, or the position's
#: own where that is larger.
GEOMETRY_PLANE_ULPS = 8.0  # units: float resolutions of the position


def _reader_plane_distance(mapping, here, work):
    """How far every coordinate of the positions *here* is from the nearest
    lattice plane of *mapping*, flat, in the analysis dtype *work*, with a
    coordinate within :data:`GEOMETRY_PLANE_ULPS` float resolutions of a
    plane taken as on it (zero).  NaN for a coordinate that is not finite.
    """
    distance = jnp.ravel(mapping._plane_distance(here)).astype(work)
    window = GEOMETRY_PLANE_ULPS * jnp.ravel(mapping._plane_resolution(here)).astype(work)
    return jnp.where(distance <= window, jnp.zeros((), work), distance)


def _geometry_plane_limit(step_pure, x_star, consts, readers, weights, unit):
    """The largest ``spectral_error_bound`` at which no position the pass
    reads from the iterate can be across a lattice plane at the fixed
    point (experimental).

    A geometry-dependent mapping is piecewise smooth in the positions it
    reads: a multilinear stencil is one polynomial inside a lattice cell
    and another in the next, so the pass's Jacobian jumps where a
    position crosses a lattice plane.  The spectral estimate and the
    bounds built on it are the linearisation at the returned iterate
    ``x_k``, which describes the pass only in the cells the positions are
    in *there*.  With the fixed point across a plane the bound read
    0.13x the true distance with its flag set (MADD-ANO-242).

    The bound ``B`` states ``unit * ||D (x_k - x*)||_2 <= B`` (``D`` the
    *weights*, ``unit`` the norm's constant: 1 under ``"l2"``, ``1 /
    (rtol sqrt(count))`` under ``"mixed"``), so entry ``j`` of the fixed
    point is within ``B / (unit D_j)`` of the iterate's.  For every entry
    ``j`` of a position field the pass reads from the iterate or from the
    state the same pass has built (*readers*), with ``d_j`` the distance
    from ``x_k[j]`` to the nearest plane of the mapping that reads it
    (``mapping._plane_distance``):

        ``limit = unit * min_j (D_j d_j) / GEOMETRY_PLANE_REACH``,

    with ``d_j`` taken as zero where the pass itself moves the position
    further than that (``|F(x_k)_j - x_k[j]| > d_j``: the position a
    Gauss-Seidel sweep reads after its holder's update is then in another
    cell than the iterate's).  Where ``B <= limit``:

    * a fixed point across a lattice plane is further than
      ``GEOMETRY_PLANE_REACH * B`` from ``x_k``, so it is not the one
      ``B`` is about;
    * the pass's polynomial piece at ``x_k`` has, by Newton-Kantorovich
      (``h <= 1/2``), a fixed point within twice the Newton step, hence
      within ``GEOMETRY_PLANE_REACH * B``; its positions are then in the
      cells of ``x_k``'s, where the piece *is* the pass, so it is a fixed
      point of the pass and the linearisation describes the way to it as
      it would on a smooth map.

    Where ``B > limit`` a plane is within reach, and the report keeps the
    flag only if the step certified the linearisation across it: the
    gradient bound's Newton-Kantorovich check
    (:func:`_gradient_error_bound_at`) takes the pass's Jacobian at
    ``x_k`` and at the Newton point ``x_k + delta`` -- across whatever
    plane the predicted fixed point is beyond -- and is finite only where
    the resolvent applied to the difference is under one half.  A screen
    alone would withdraw the flag of nearly every step of a group with
    many points (one of them is always near a plane); the check alone
    would withdraw it wherever a smooth nonlinearity fails Kantorovich
    far from any plane, which the bound's documented conditions already
    cover.

    A position that is a constant of the pass (a target-anchored edge of
    ``update``, a node outside the group) does not move between the
    iterate and the fixed point and is not in *readers*.

    *readers* is a sequence of ``(where, shape, dtype, mapping)``: the
    flat indices of a position field, its shape and dtype, and a mapping
    that reads it.  Returns a scalar in the analysis dtype: ``inf`` with
    no reader, NaN where a position is not finite.  One evaluation of the
    pass, in a branch of its own (as :func:`_spectral_rate_at`).
    """
    x_sg = jax.lax.stop_gradient(x_star)
    consts_sg = tuple(jax.lax.stop_gradient(c) for c in consts)
    work = _analysis_dtype(x_sg.dtype)
    d = jax.lax.stop_gradient(jnp.asarray(weights, work))
    scale = jax.lax.stop_gradient(jnp.asarray(unit, work))
    if not readers:
        return jnp.full((), jnp.inf, work)

    def limit(operands):
        xx, cc, dd, uu = operands
        passed = _F_dispatch(step_pure, xx, cc)
        moved = jnp.abs(passed.astype(work) - xx.astype(work))
        nearest = jnp.full((), jnp.inf, work)
        for where, shape, dtype, mapping in readers:
            at = jnp.asarray(where, jnp.int32)
            here = xx[at].reshape(shape).astype(dtype)
            distance = _reader_plane_distance(mapping, here, work)
            distance = jnp.where(moved[at] > distance, jnp.zeros((), work), distance)
            # The position the pass builds (the one a Gauss-Seidel sweep
            # reads after its holder's update) on a plane too.
            built = _reader_plane_distance(mapping, passed[at].reshape(shape).astype(dtype), work)
            distance = jnp.where(built > 0, distance, built)
            nearest = jnp.minimum(nearest, jnp.min(distance * dd[at]))
        return uu * nearest / GEOMETRY_PLANE_REACH

    return jax.lax.cond(
        jnp.all(jnp.isfinite(x_sg)), limit, lambda _operands: jnp.full((), jnp.nan, work),
        (x_sg, consts_sg, d, scale))


def _spectral_rate_at(step_pure, x_star, consts, weights, spectral_weights=None,
                      resolution=None, field_reference=None, map_eps=None):
    """``(rho, arnoldi_residual, amplification)`` of ``dF/dx`` at ``x_star``.

    With ``field_reference`` -- a function from a flat state to the flat
    vector holding, at each entry, its field's ``max|v|`` -- a fourth
    value is returned: ``||D r|| / ||D_pair r||``, ``r = F(x*) - x*``, the
    factor by which the residual in the returned state's weights ``D``
    (each field over its own magnitude at ``x*``, the weights the
    resolvent is measured in) exceeds the same residual in the weights the
    loop measured it in (each field over ``max(max|F(x*)|, max|x*|)``,
    the pair the pass compared).  It is ``1`` unless a field grew across
    the last pass, and ``coupling_diagnostics()`` multiplies the bound's
    resolvent factor by it so that ``spectral_error_bound`` bounds the
    distance in the returned state's weights, the norm it documents
    (MADD-ANO-146).

    ``weights`` is the flat vector of per-entry factors the group's
    convergence norm multiplies a state change by at ``x_star`` --
    ``1 / max|field|`` on a field the norm reads, ``0`` on one it does
    not (dead band, or off the interface under the interface norm).
    ``spectral_weights`` is the same vector with the *dead-banded* fields
    the norm would otherwise read given a positive weight instead of
    zero -- ``1 / max|field|``, their own magnitude's, as if there were
    no dead band (``1 / atol`` for a field that is exactly zero); it
    defaults to ``weights``.  The Arnoldi iteration runs on
    ``D' J D'^{-1}`` for those weights, so the resolvent norm it reports
    is in a norm that agrees with the group's on every field the group
    reads, and the spectral radius is the map's own wherever the weights
    are positive.

    **Why the dead band does not get weight zero here.**  A field inside
    the dead band leaves the *residual*; it does not leave the *loop*.
    Weighted zero, a dead-banded field on the coupling loop -- a small
    displacement feeding a large stiffness -- zeroed its row and column
    of ``D J D^+`` and cut the loop out of the spectrum: ``rho_spectral``
    read ``0.000`` for a map whose spectral radius is ``0.9``, and the
    bound read 0.15-0.29x the true distance of a field the norm *keeps*
    with ``spectral_usable=True``.  A positive weight keeps the loop.
    The bound is still on the group's own norm: ``||e||_{D'} >=
    ||e_kept||_D`` because the weights agree on the kept fields, and
    ``||e||_{D'} <= resolvent * ||D' r||``, whose dead-banded part the
    group's residual does not contain -- so that share is measured here
    (one evaluation of ``F`` at ``x_star``) and folded in: where it is
    non-zero the returned ``amplification`` is the factor the bound
    applies, ``max(resolvent, 1 / (1 - rho_safe)) * sqrt(1 + share**2)``,
    with ``share`` the dead-banded part of ``D' r`` over the kept part
    (or over the residual's float floor, where the kept part is below
    it).  Where it is zero -- no dead-banded field, the usual case --
    ``amplification`` is the resolvent norm exactly as before.

    **Why the residual is passed into the Krylov space.**  The resolvent
    norm bounds ``(I - J)^{-1} r`` only for an ``r`` in the invariant
    space it was computed on, and a space grown from one start vector
    need not contain the residual (a repeated eigenvalue breaks it down
    early; a relaxed iterate's residual is not a Jacobian image at all).
    So ``D' r`` is handed to
    :func:`~maddening.core.coupling.acceleration.arnoldi_spectral_radius`
    as ``v_extra``: the space continues from it at every breakdown and
    the part it never absorbed is reported as unresolved.  The start
    vector is still a fixed-seed normal draw, which has a component in
    every mode, so ``rho`` is the spectral radius of the whole map and
    not only of the residual's cyclic subspace.

    The matvec is the same ``jax.jvp`` of the one-pass map the IFT
    tangent rule builds (see ``_ift_solve_jvp``); it is applied
    ``SPECTRAL_KRYLOV_STEPS + 1`` times (the last is the product the
    estimate checks itself against), which with the one evaluation of
    ``F`` is the whole cost.  The inputs are ``stop_gradient``-ed so the
    estimate is forward-only bookkeeping like the rest of the
    diagnostics: under ``jax.grad`` nothing here is linearised, and the
    adjoint of the step is unchanged by its presence.

    ``resolution`` is the residual's float resolution per entry in the
    weighted coordinates (:func:`_residual_resolution`); it defaults to
    ``PRECISION_FLOOR_ULPS * eps`` of the flat vector's dtype, which is
    the right number only for a group whose fields all share that dtype.
    ``map_eps`` is the ``eps`` of the coarsest floating dtype among the
    group's fields (a Python float): the rounding of the products the
    Arnoldi iteration is built from, which sets its breakdown test and
    the backward error its result is good to
    (:data:`~maddening.core.coupling.acceleration._ARNOLDI_NOISE_ULPS`).
    It defaults to the flat vector's own ``eps``.
    """
    from maddening.core.coupling.acceleration import (  # noqa: PLC0415
        SPECTRAL_MARGIN,
        arnoldi_spectral_radius,
    )

    x_sg = jax.lax.stop_gradient(x_star)
    consts_sg = tuple(jax.lax.stop_gradient(c) for c in consts)
    # The analysis runs in at least float32 (``_analysis_dtype``); the map
    # itself is evaluated in the group's own dtype, its tangents and
    # outputs cast at the boundary -- no-ops for a float32 or wider group.
    work = _analysis_dtype(x_sg.dtype)
    d = jax.lax.stop_gradient(jnp.asarray(weights, work))
    dw = d if spectral_weights is None else jax.lax.stop_gradient(
        jnp.asarray(spectral_weights, work))
    res = jax.lax.stop_gradient(_default_resolution(x_sg).astype(work) if resolution is None
                                else jnp.asarray(resolution, work))

    def spectrum(operands):
        xx, cc, dd, ww, rr = operands
        live = ww > 0
        w_inv = jnp.where(live, 1.0 / jnp.where(live, ww, 1.0), 0.0)
        # The tangent lifted out of the underflow range where the group is
        # that small (``pow2_frame(..., mode="lift")``; 1 elsewhere) and the residual's
        # difference taken entry by entry in a power-of-two frame
        # (``_framed_difference``): in state units a group of magnitude
        # ~1e-34 handed the JVP tangents below the normal range, and its
        # residual ``F(x) - x`` flushed to zero.
        lift = pow2_frame(w_inv, mode="lift")

        def matvec(v):
            _, Jv = jax.jvp(
                lambda x_: _F_dispatch(step_pure, x_, cc), (xx,),
                ((v * (w_inv * lift)).astype(xx.dtype),)
            )
            return (ww / lift) * Jv.astype(work)

        r_w = _framed_difference(_F_dispatch(step_pure, xx, cc).astype(work),
                                 xx.astype(work), ww)
        v0 = jax.random.normal(jax.random.PRNGKey(0), xx.shape, work)
        rho, resid, amp = arnoldi_spectral_radius(
            matvec, v0, v_extra=r_w, noise_eps=_map_eps(xx.dtype, map_eps))
        # The share of ``D' r`` the group's residual does not see: the
        # dead-banded fields (weight 0 in ``dd``, positive in ``ww``).
        kept = dd > 0
        r_kept = jnp.linalg.norm(jnp.where(kept, r_w, 0.0))
        r_unread = jnp.linalg.norm(jnp.where(kept, 0.0, r_w))
        floor = _floor_of(rr, kept)
        denom = jnp.maximum(r_kept, floor)
        unread = r_unread > 0
        share = jnp.where(unread, r_unread / jnp.where(denom > 0, denom, 1.0), 0.0)
        share = jnp.where(jnp.logical_and(unread, denom <= 0), jnp.inf, share)
        rho_safe = rho + SPECTRAL_MARGIN * resid
        contracting = rho_safe < 1
        radius = jnp.where(contracting, 1.0 / jnp.where(contracting, 1.0 - rho_safe, 1.0),
                           jnp.inf)
        folded = jnp.maximum(amp, radius) * jnp.sqrt(1.0 + share * share)
        amp_out = jnp.where(unread, folded, amp)
        if field_reference is None:
            return rho, resid, amp_out
        # The pair the loop's residual divided each field by: its magnitude
        # at ``x*`` or at ``F(x*)``, whichever is larger.  The ratio of the
        # two norms of the same ``r`` is a ratio of the norm's own units, so
        # the constants it leaves out (``rtol``, a mixed norm's ``1/count``)
        # cancel.
        fx = _F_dispatch(step_pure, xx, cc)
        ref_x = field_reference(xx).astype(work)
        ref_pair = jnp.maximum(field_reference(fx).astype(work), ref_x)
        shrink = jnp.where(ref_pair > 0, ref_x / jnp.where(ref_pair > 0, ref_pair, 1.0), 1.0)
        r_pair = jnp.linalg.norm(jnp.where(kept, r_w * shrink, 0.0))
        ratio = jnp.where(r_pair > 0, r_kept / jnp.where(r_pair > 0, r_pair, 1.0), 1.0)
        return rho, resid, amp_out, jnp.maximum(ratio, 1.0)

    nan = jnp.full((), jnp.nan, work)
    # A Jacobian at a state that has left float range describes nothing
    # (a spectral radius of 0.0 read there as "contracts instantly"), so
    # a non-finite state reports the triple as NaN -- "not computed" --
    # like the gradient bound beside it.  In a branch of its own for the
    # same reason that one is: compiled as a separate computation, the
    # Jacobian-vector products cannot share subexpressions with the
    # forward and so cannot move the state it diagnoses.
    nans = (nan, nan, nan) if field_reference is None else (nan, nan, nan, nan)
    return jax.lax.cond(
        jnp.all(jnp.isfinite(x_sg)), spectrum, lambda _operands: nans,
        (x_sg, consts_sg, d, dw, res),
    )


def _interface_spectral_rate_at(step_pure, x_star, consts, x_weights, reading,
                                weights, spectral_weights, resolution, reference,
                                map_eps=None):
    """``(rho, arnoldi_residual, amplification, ratio)`` in the interface norm's own coordinates.

    :func:`_spectral_rate_at` with ``field_reference``, for a group under
    ``convergence_norm="interface"`` with a mapping or a transform on an
    internal edge, or a field that more than one internal edge reads
    (:func:`_reading_is_the_fields`).  That norm measures what each
    internal edge *delivers*
    -- its source value through the edge's interface mapping and then its
    transform (``coupling_residual_interface``) -- each over its own
    magnitude; the spectral analysis took its weights from the raw source
    fields, and the bound multiplied a residual in one set of coordinates
    by a resolvent norm in the other.  An offset (a unit conversion's
    273.15) shrinks the reading's relative change, a selection
    (``"extract_last"``, or a mapping that delivers one entry) changes
    which magnitude an entry is divided by, and the bound read
    0.0014-0.098x the true distance with ``spectral_usable=True``.

    So here every quantity is taken on the reading ``y = Phi(x)`` itself
    (``reading``: the flat vector of the delivered edge values, in the
    order the norm sums them).  The pass reads the iterate only through
    those edges, ``F = G o Phi``, so the reading iterates by its own map
    ``Phi o G``, whose Jacobian ``A = Phi' G'`` has ``dF/dx``'s non-zero
    spectrum, and for an affine reading of an affine map ``Phi(x) -
    Phi(x*) = (I - A)^{-1} (Phi(x) - Phi(F(x)))`` exactly: the resolvent of
    ``A`` in the reading's weights carries the residual the loop measured
    to the distance in the norm the loop measured it in.  ``A`` is applied
    through preimages (:func:`~maddening.core.coupling.acceleration._arnoldi_through`):
    the Krylov vectors are readings of state tangents, so no mapping or
    transform is ever inverted.

    ``x_weights`` are the state's spectral weights, which only condition
    the preimages; ``weights``, ``spectral_weights`` and ``resolution``
    are the reading's, entry by entry, with the meanings
    :func:`_spectral_rate_at` gives them for the state's (dead band,
    spectrum, float floor); ``reference(x)`` is, at each entry of the
    reading, its edge's ``max|Phi_e(x)|``, which gives the pair-to-returned
    ratio (MADD-ANO-146).  The share of a dead-banded edge's residual is
    folded into the factor as there.  ``map_eps`` as there.  NaN on a
    non-finite state.
    """
    from maddening.core.coupling.acceleration import (  # noqa: PLC0415
        SPECTRAL_MARGIN,
        _arnoldi_through,
    )

    x_sg = jax.lax.stop_gradient(x_star)
    consts_sg = tuple(jax.lax.stop_gradient(c) for c in consts)
    work = _analysis_dtype(x_sg.dtype)
    wx = jax.lax.stop_gradient(jnp.asarray(x_weights, work))
    dd0 = jax.lax.stop_gradient(jnp.asarray(weights, work))
    ww0 = jax.lax.stop_gradient(jnp.asarray(spectral_weights, work))
    rr0 = jax.lax.stop_gradient(jnp.asarray(resolution, work))

    def spectrum(operands):
        xx, cc, wxx, dd, ww, rr = operands
        live = wxx > 0
        w_inv = jnp.where(live, 1.0 / jnp.where(live, wxx, 1.0), 0.0)
        lift = pow2_frame(w_inv, mode="lift")

        def matvec(v):
            _, Jv = jax.jvp(
                lambda x_: _F_dispatch(step_pure, x_, cc), (xx,),
                ((v * (w_inv * lift)).astype(xx.dtype),)
            )
            return (wxx / lift) * Jv.astype(work)

        def measure(v):
            # Only the fields the reading reads have a weight, so a
            # preimage's other entries carry nothing into it.
            _, Lv = jax.jvp(reading, (xx,), ((v * (w_inv * lift)).astype(xx.dtype),))
            return (ww / lift) * Lv.astype(work)

        fx = _F_dispatch(step_pure, xx, cc)
        r_y = _framed_difference(reading(fx).astype(work), reading(xx).astype(work), ww)
        r_x = _framed_difference(fx.astype(work), xx.astype(work), wxx)
        u0 = jax.random.normal(jax.random.PRNGKey(0), xx.shape, work) * live.astype(work)
        rho, resid, amp = _arnoldi_through(
            matvec, measure, u0, extra=(r_x, r_y), noise_eps=_map_eps(xx.dtype, map_eps))
        kept = dd > 0
        r_kept = jnp.linalg.norm(jnp.where(kept, r_y, 0.0))
        r_unread = jnp.linalg.norm(jnp.where(kept, 0.0, r_y))
        floor = _floor_of(rr, kept)
        denom = jnp.maximum(r_kept, floor)
        unread = r_unread > 0
        share = jnp.where(unread, r_unread / jnp.where(denom > 0, denom, 1.0), 0.0)
        share = jnp.where(jnp.logical_and(unread, denom <= 0), jnp.inf, share)
        rho_safe = rho + SPECTRAL_MARGIN * resid
        contracting = rho_safe < 1
        radius = jnp.where(contracting, 1.0 / jnp.where(contracting, 1.0 - rho_safe, 1.0),
                           jnp.inf)
        folded = jnp.maximum(amp, radius) * jnp.sqrt(1.0 + share * share)
        amp_out = jnp.where(unread, folded, amp)
        ref_x = reference(xx).astype(work)
        ref_pair = jnp.maximum(reference(fx).astype(work), ref_x)
        shrink = jnp.where(ref_pair > 0, ref_x / jnp.where(ref_pair > 0, ref_pair, 1.0), 1.0)
        r_pair = jnp.linalg.norm(jnp.where(kept, r_y * shrink, 0.0))
        ratio = jnp.where(r_pair > 0, r_kept / jnp.where(r_pair > 0, r_pair, 1.0), 1.0)
        return rho, resid, amp_out, jnp.maximum(ratio, 1.0)

    nan = jnp.full((), jnp.nan, work)
    return jax.lax.cond(
        jnp.all(jnp.isfinite(x_sg)), spectrum, lambda _operands: (nan, nan, nan, nan),
        (x_sg, consts_sg, wx, dd0, ww0, rr0),
    )


def _map_eps(dtype, map_eps=None) -> float:
    """The products' rounding: *map_eps*, or ``eps`` of the state's own *dtype*."""
    return float(jnp.finfo(dtype).eps) if map_eps is None else float(map_eps)


def _analysis_dtype(dtype):
    """The dtype the coupling diagnostics' linear algebra runs in: at least float32.

    The spectral bound and the gradient bound orthogonalise Krylov vectors
    and call LAPACK (an SVD, a QR, a solve), which has no bfloat16 or
    float16 kernels: ``diagnostics=True`` on a 16-bit group raised
    ``NotImplementedError`` from inside the step.  The map is still
    evaluated in the group's own dtype -- its rounding is what the bound
    is about -- and only the analysis is widened; a float32 or float64
    group's dtype is returned unchanged, so its program is the one it was.
    """
    return jnp.promote_types(dtype, jnp.float32)


def _framed_difference(a, b, weight):
    """``weight * (a - b)``, the difference taken entry by entry in a power-of-two frame.

    Each pair is rescaled by its own power of two (``max(|a_i|, |b_i|)``
    into ``[0.5, 1)``), differenced, and the weight carries the inverse
    scale.  A power of two scales exactly, so between ordinary numbers
    this is ``weight * (a - b)`` to the bit; below about 1e-31 in float32
    a change of an ulp is subnormal and the bare difference flushed to
    zero.
    """
    k = pow2_frame(a, b, mode="entrywise")
    return (a * k - b * k) * (weight / k)


def _framed_shift(x, step, scale):
    """``x + step * scale``, formed entry by entry in a power-of-two frame of ``x``.

    Between ordinary numbers this is the bare sum to the bit; for a state
    near 1e-34 the increment ``step * scale`` is below the normal range
    and the bare product flushed to zero before it was added.
    """
    k = pow2_frame(x, mode="entrywise")
    return (x * k + step * (scale * k)) / k


def _default_resolution(x):
    """``PRECISION_FLOOR_ULPS * eps`` per entry, in ``x``'s own dtype."""
    from maddening.core.coupling.acceleration import (  # noqa: PLC0415
        PRECISION_FLOOR_ULPS,
    )
    return jnp.full(x.shape, PRECISION_FLOOR_ULPS * jnp.finfo(x.dtype).eps, x.dtype)


def _floor_of(resolution, mask):
    """The L2 norm of ``resolution`` over the entries ``mask`` selects.

    ``sqrt(sum(mask * resolution**2))``: for a uniform resolution
    ``C * eps`` (``C`` and ``eps`` powers of two) this is exactly
    ``C * eps * sqrt(sum(mask))``, the form it replaces, so a group
    whose fields share one dtype gets the same bits as before.
    """
    return jnp.sqrt(jnp.sum(mask.astype(resolution.dtype) * (resolution * resolution)))


def _residual_resolution(fields_eps):
    """The residual's float resolution per entry of the weighted flat vector.

    ``fields_eps`` is the flat vector of each entry's *own* field's
    ``finfo(dtype).eps`` -- not the flat vector's, which is the promoted
    dtype of every field in the group.  In coordinates where each field
    is divided by its magnitude one unit of ``eps * max|field|`` is
    ``eps`` of that field's dtype, so this is ``PRECISION_FLOOR_ULPS``
    of those, the quantity
    :func:`~maddening.core.coupling.acceleration.residual_precision_floor`
    reports in the group's own norm.  A float16 field beside a float32
    one rounds 8192 times more coarsely than the promoted float32 eps
    says; taken from the promoted dtype, the gradient bound's floor and
    its floor-sized probe step were below float16 resolution, and the
    bound read ``0.0`` with ``gradient_bound_usable=True`` against a true
    error of 3.2%.
    """
    from maddening.core.coupling.acceleration import (  # noqa: PLC0415
        PRECISION_FLOOR_ULPS,
    )
    return PRECISION_FLOOR_ULPS * fields_eps


def _probe_direction(c, key):
    """One constant's probe: its own magnitude, entry by entry, with random signs.

    A zero entry is probed at the constant's largest magnitude (or 1.0
    for an all-zero constant), so a parameter that happens to sit at
    zero is still probed -- the relative error of ``dx*/dc`` is defined
    there even though ``c`` is not a scale.  ``key=None`` returns the
    magnitudes alone: the size an entry probe moves each entry by.
    """
    mag = jnp.abs(c)
    top = jnp.max(mag)
    mag = jnp.where(mag > 0, mag, jnp.where(top > 0, top, jnp.ones_like(top)))
    if key is None:
        return mag
    return mag * jax.random.rademacher(key, c.shape, c.dtype)


#: A floating constant of at most this many entries is probed entry by
#: entry by the gradient bound (each entry is "one scalar constant" of
#: ``gradient_relative_error_bound``'s reading); a larger one is probed as
#: a whole, along one fixed-seed direction weighted by its magnitudes, and
#: ``coupling_report()`` names it.  The bound costs five Jacobian-vector
#: products per probe.
GRADIENT_PROBE_ENTRY_LIMIT = 64


def _whole_probe_names(consts, node_params, full_state, new_state, external_inputs):
    """``((name, entries), ...)`` for each floating constant above :data:`GRADIENT_PROBE_ENTRY_LIMIT`.

    A constant of the closure-converted pass is named by identity: a node
    parameter (``node.param``), a state field at the step's start
    (``node.field (step start)``), a value this step already computed
    upstream (``node.field (this step)``) or an external input
    (``external node.field``); anything else -- a value the step derived
    before the group -- by its shape.
    """
    names: dict[int, str] = {}

    def walk(tree, prefix, suffix=""):
        if isinstance(tree, dict):
            for k, v in tree.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k), suffix)
        else:
            names.setdefault(id(tree), prefix + suffix)

    walk(node_params or {}, "")
    walk({f"external {k}": v for k, v in (external_inputs or {}).items()}, "")
    walk({k: v for k, v in (full_state or {}).items() if k != _META_KEY}, "", " (step start)")
    walk({k: v for k, v in (new_state or {}).items() if k != _META_KEY}, "", " (this step)")
    out = []
    for c in consts:
        if not jnp.issubdtype(jnp.asarray(c).dtype, jnp.floating):
            continue
        size = int(np.prod(jnp.shape(c), dtype=np.int64))
        if size > GRADIENT_PROBE_ENTRY_LIMIT:
            out.append((names.get(id(c), f"a derived constant of shape {tuple(jnp.shape(c))}"),
                        size))
    return tuple(out)


def _probe_plan(consts, probed):
    """``[(constant index, entry index or -1)]``: one row per gradient-bound probe.

    Every entry of a floating constant of at most
    :data:`GRADIENT_PROBE_ENTRY_LIMIT` entries is its own probe; a larger
    constant is one probe (``-1``) along :func:`_probe_direction`.
    """
    rows: list[tuple[int, int]] = []
    for i in probed:
        size = int(np.prod(jnp.shape(consts[i]), dtype=np.int64))
        if size <= GRADIENT_PROBE_ENTRY_LIMIT:
            rows.extend((i, j) for j in range(size))
        else:
            rows.append((i, -1))
    return rows


def _gradient_error_bound_at(step_pure, x_star, consts, weights, rho,
                             arnoldi_residual, amplification, resolution=None,
                             plane_readers=None):
    """A bound on the relative error of the IFT tangent at the returned iterate.

    The IFT rule (:func:`_ift_solve_jvp`) solves
    ``(I - J(x)) t = F_c(x) c_dot`` at the iterate the forward
    *returned*, ``x_k``; the derivative of the fixed point is the same
    solve at ``x*``.  Exactly,
    ``t_k - t* = (I - J(x*))^{-1} [G(x_k) - G(x*)]`` with
    ``G(x) = J(x) t_k + F_c(x) c_dot`` the one-pass map's
    Jacobian-vector product along the tangent the rule returned (see
    :func:`~maddening.core.coupling.acceleration.ift_gradient_error_bound`).
    ``G`` does not move between the two points when ``F`` is affine in
    the state with coefficients the state does not change -- the reason
    a linear group's gradient is exact wherever its forward stops -- and
    moves by ``O(||x_k - x*|| * d^2 F)`` otherwise.  This measures that
    movement and bounds the rest:

    1. **A resolvent that can be applied.**
       :func:`~maddening.core.coupling.acceleration.jacobian_range_basis`
       spans ``range(J)`` with eight random images (``8 + k`` JVPs,
       ``k = min(n, 8)``; the rank of a coupling Jacobian is at most
       the number of boundary scalars crossing the group's edges) and
       :func:`~maddening.core.coupling.acceleration.resolvent_apply`
       then solves ``(I - J) t = w`` for one JVP each.  A group whose
       range the basis did not capture reports NaN.
    2. **One probe per scalar entry** of every floating constant the
       closure-converted map captures -- every parameter leaf, the
       pre-step states, the states of nodes outside the group it reads
       -- each entry moved by its own magnitude, the others held; a
       constant of more than :data:`GRADIENT_PROBE_ENTRY_LIMIT` entries
       is probed as a whole, along one fixed-seed direction weighted by
       its magnitudes (:func:`_probe_direction`), and ``coupling_report()``
       names it.  One JVP of ``F`` in the constants per probe gives
       ``w_i = F_c(x_k) c_dot_i`` (and, as its primal, ``F(x_k)``),
       brought to the state's relative size by an exact power of two.
       Per entry, because a relative error is a different number for
       each and a combined direction can cancel or be dominated: one
       probe over every constant read ``0.0`` at every cap on the stiff
       spring pair while its stiffness gradient was 0.8-4.8% off, and
       one |c|-weighted probe of a two-entry gain read 23.6x under the
       error of its small entry (MADD-ANO-143).
    3. **The tangents and the direction to the fixed point.**
       ``t_i = (I - J(x_k))^{-1} w_i`` (one JVP each) is what the
       adjoint returns for probe ``i``; ``delta = (I - J(x_k))^{-1} r``
       with ``r = F(x_k) - x_k`` (one JVP) is the Newton correction --
       exactly ``x* - x_k`` for an affine ``F``, second-order close
       otherwise.  It supplies the *direction* only.
    4. **The curvature, as a directional second difference of the
       adjoint's own matvec**: ``G_i`` evaluated at ``x_k`` and at
       ``x_k + delta`` by the same Jacobian-vector product (a JVP pair
       per probe, so the difference is exactly zero on a map whose
       JVP does not depend on the point, rather than rounding noise
       amplified by the resolvent).
    5. **The distance, and the resolvent applied to each secant**:
       ``distance`` is the larger of
       :func:`~maddening.core.coupling.acceleration.spectral_error_bound`
       of ``||r||`` (the Arnoldi triple :func:`_spectral_rate_at`
       already computed at ``x_k``) and ``||delta||``.  Per probe the
       bound is ``distance * ||(I - J(x_k))^{-1} (G_i(x_k + delta) -
       G_i(x_k))|| / (||delta|| * ||t_i||)`` -- the resolvent applied
       exactly to the secant (one JVP), not its norm times the
       Arnoldi factor, which is the resolvent restricted to the Krylov
       space and read 0.19x the true error on a ring whose secant lies
       outside it (MADD-ANO-142) -- and the reported value is the
       largest over the probes the fixed point responds to
       (``||t_i|| > 0``, or non-finite).
    6. **Newton-Kantorovich** with the full-operator resolvent norm
       (:func:`_full_resolvent_norm`, exact from the range basis), ``h``
       the larger of ``beta`` times the Jacobian's change along ``delta``
       and the affine-covariant constant
       ``||(I - J(x_k))^{-1} (J(x_k + delta) - J(x_k))||`` as an operator on
       the Jacobian's row space (``3 k`` JVPs); and the Newton step's
       second-order miss, ``t* - eta``, carried into each probe's bound.

    ``11 + 4 k + 5 n_p + 2 k n_p`` Jacobian-vector products (plus one linearisation and ``k`` reverse-mode products where the state has more than ``k`` entries) in all (at most ``43 + 21 n_p`` forward), ``n_p``
    the number of probes, which is
    why it is gated behind ``diagnostics=True``; the per-probe products
    are ``vmap``-ed, so the primal is evaluated once.  Every input is
    ``stop_gradient``-ed: forward-only bookkeeping, and the adjoint of
    the step is unchanged by its presence.  The linear algebra runs in
    coordinates scaled by the norm weights where those are non-zero (a
    similarity, so every solution is the same vector) so fields of
    unlike magnitude do not swamp the orthogonalisation; every norm is
    the group's, over the fields its norm reads.  ``resolution`` is the
    residual's float resolution per entry, as :func:`_spectral_rate_at`
    takes it.

    With *plane_readers* (experimental: a group whose pass reads a moving
    geometry; the ``readers`` of :func:`_geometry_plane_limit`, at least
    one) the return is the pair ``(bound, margin)``, ``margin`` the
    lattice-plane margin of the Newton-Kantorovich ball around the iterate
    (:func:`_kantorovich_ball_plane_margin`): one more evaluation of the
    pass.  Without it nothing of that is traced and the return is the
    bound alone.
    """
    x_sg = jax.lax.stop_gradient(x_star)
    consts_sg = tuple(jax.lax.stop_gradient(jnp.asarray(c)) for c in consts)
    # In at least float32, as ``_spectral_rate_at`` (``_analysis_dtype``).
    dtype = _analysis_dtype(x_sg.dtype)
    d = jax.lax.stop_gradient(jnp.asarray(weights, dtype))
    res = jax.lax.stop_gradient(_default_resolution(x_sg).astype(dtype) if resolution is None
                                else jnp.asarray(resolution, dtype))
    nan = jnp.full((), jnp.nan, dtype)

    probed = [
        i for i, c in enumerate(consts_sg)
        if jnp.issubdtype(c.dtype, jnp.floating) and c.size > 0
    ]
    readers = tuple(plane_readers) if plane_readers else ()
    nothing = (nan, nan) if readers else nan
    if not probed or x_sg.size == 0:
        # Nothing the fixed point can respond to, or a fixed-point vector
        # with no entries (every floating field of the group has none):
        # no gradient, no error.
        return nothing

    def bound(operands):
        return _gradient_error_bound_body(step_pure, probed, *operands, readers=readers)

    # In a branch of its own, so XLA compiles it as a separate
    # computation.  Inlined beside the forward, the Jacobian-vector
    # products in the constants share constants and subexpressions with
    # the forward's first pass, the algebraic simplifier then rewrites
    # that pass differently (its rewrites depend on how many users an
    # instruction has), and the returned *state* moved by one ulp on
    # one of twelve configurations measured (chain-5, Jacobi, measured
    # 2.4e-7 relative on the first step it showed) -- a diagnostic
    # must not move the answer it diagnoses.  A non-finite state has no
    # gradient bound to compute, which is what makes the predicate a
    # runtime one.
    return jax.lax.cond(
        jnp.all(jnp.isfinite(x_sg)), bound, lambda _operands: nothing,
        (x_sg, consts_sg, d, rho, arnoldi_residual, amplification, res),
    )


def _scaled_transpose(step_pure, x_sg, consts_sg, s, s_inv, lift, dtype):
    """``J(x_k)^T`` in the gradient bound's scaled coordinates, or ``None``.

    The transpose of the bound's ``matvec`` (``z -> (s / lift) J ((s_inv
    lift) z)``): ``u -> (s_inv lift) J^T ((s / lift) u)``, one
    reverse-mode product of the one-pass map per call.  ``None`` where
    the map cannot be differentiated in reverse mode -- a node that loops
    with ``lax.while_loop`` inside ``update`` -- which forward-mode
    products, all the rest of the bound uses, still go through.
    """
    try:
        _, vjp_fn = jax.vjp(lambda xx: _F_dispatch(step_pure, xx, consts_sg), x_sg)

        def matvec_t(u):
            (out,) = vjp_fn(((s / lift) * u).astype(x_sg.dtype))
            return (s_inv * lift) * out.astype(dtype)

        jax.eval_shape(matvec_t, jnp.zeros(x_sg.shape, dtype))
    except (ValueError, TypeError, NotImplementedError):
        return None
    return matvec_t


def _full_resolvent_norm(U, M, make_transpose):
    """``||(I - J)^{-1}||_2`` over the whole space, from the range basis; ``None`` without ``J^T``.

    With ``range(J)`` inside ``span(U)`` (orthonormal, ``n x k``) and
    ``M = U^T J U``, the resolvent is ``I + U (I - M)^{-1} B`` with
    ``B = U^T J``: the identity plus a rank-``k`` term whose columns lie in
    ``span(U)`` and whose rows lie in ``span(B^T)``.  It maps ``W =
    span(U, B^T)`` (at most ``2k`` dimensions) into itself and is the
    identity on ``W``'s orthogonal complement, so its norm is the larger
    of ``1`` and the norm of its compression to ``W`` -- exact, from a
    ``2k x 2k`` SVD.  ``B`` costs one linearisation of the map and ``k``
    reverse-mode products (``make_transpose()``, called only here).  Where
    ``U`` is square (``n <= k``) the resolvent is ``U (I - M)^{-1} U^T``
    and needs no product.  ``None`` where the map has no transpose and
    ``U`` is not square.
    """
    return _full_resolvent_norm_and_rows(U, M, make_transpose)[0]


def _full_resolvent_norm_and_rows(U, M, make_transpose):
    """``(norm, R)``: :func:`_full_resolvent_norm` and a basis of ``J``'s row space.

    ``R`` is an orthonormal basis (``n x min(n, k)``) of ``span(B^T)``,
    which ``B`` already computed -- ``J``'s row space, since ``J^T`` maps
    ``range(J)`` onto it -- or the whole space where ``U`` is square, so
    the Kantorovich check can take the Jacobian's change as an operator on
    the directions ``J`` reads at no further reverse-mode cost.  ``(None,
    None)`` without a transpose.
    """
    from maddening.core.coupling.acceleration import _spectral_norm  # noqa: PLC0415

    n, k = U.shape
    eye_k = jnp.eye(k, dtype=M.dtype)
    if n <= k:
        return _spectral_norm(jnp.linalg.inv(eye_k - M)), jnp.eye(n, dtype=M.dtype)
    matvec_t = make_transpose()
    if matvec_t is None:
        return None, None
    B = jax.vmap(matvec_t)(U.T)                     # row i: u_i^T J
    P, _ = jnp.linalg.qr(jnp.concatenate([U, B.T], axis=1))
    T = jnp.eye(P.shape[1], dtype=M.dtype) + (P.T @ U) @ jnp.linalg.solve(eye_k - M, B @ P)
    R, _ = jnp.linalg.qr(B.T)
    return jnp.maximum(_spectral_norm(T), jnp.ones((), M.dtype)), R


def _kantorovich_root_and_miss(step, h):
    """``(sqrt(1 - 2h), t* - eta)``: Kantorovich's root and the Newton step's miss.

    ``t* = eta (1 - sqrt(1 - 2h)) / h`` bounds the distance from the
    Newton step's start to the fixed point, so ``t* - eta`` is the most the
    step of length ``eta`` (*step*) can miss it by.  It is formed as ``eta
    2h / (1 + root)^2`` (``1 - root = 2h / (1 + root)``), which does not
    cancel: in float32 the quotient form read ``t* = 0`` for ``h`` below
    about 1.5e-8 (``1 - 2h`` rounds to 1), 2-4 ``eta`` up to 3e-8 and
    1.19 ``eta`` up to about 3e-7, where ``t*`` is ``eta`` to within ``h``.
    The root is clipped at 0 (``h >= 1/2``, where the check fails) and the
    miss is 0 at ``h = 0``.
    """
    root = jnp.sqrt(jnp.maximum(1.0 - 2.0 * h, 0.0))
    return root, jnp.where(h > 0, step * (2.0 * h) / ((1.0 + root) * (1.0 + root)), 0.0)


def _kantorovich_ball_plane_margin(readers, x_k, f_k, f_newton, weights, moved, radius):
    """How many times the radius of the Newton-Kantorovich ball around the
    returned iterate the nearest lattice plane of a position the pass reads
    is away (experimental): over one, the iterate, the Newton point and the
    fixed point are in one polynomial piece of the pass.

    **Why the gradient's flag needs it.**  The gradient bound is the
    smooth theory's: it takes the pass's Jacobian at the returned iterate
    ``x_k`` and at the Newton point ``x_N = x_k + delta`` and bounds the
    Jacobian at the fixed point ``x*`` from the two.  A
    geometry-dependent mapping is another polynomial of the positions
    across a lattice plane, and two Jacobians say nothing of a plane
    neither point is beyond: with ``x*`` just past a plane ``x_N`` stops
    short of, both are one cell's, the check reads no change (``h`` near
    0), and the gradient of the fixed point is the other cell's.  The
    bound was 15 to 70,000 times under the reference's error with its
    flag set (MADD-ANO-251).  So the flag does not stand on what the
    check reads; it stands where no plane is in the ball at all.

    **The argument**, with ``D`` the *weights*, ``eta = ||D delta||_2``,
    *moved* ``= |D delta|`` entry by entry and *radius* ``= eta + floor``
    (:func:`_gradient_error_bound_body` forms them), under these
    assumptions:

    1. *The kernel.*  Between two positions with no lattice plane and no
       face of the hull between them the mapping is one polynomial of the
       positions, and ``mapping._plane_distance`` is the distance to the
       nearest (``multilinear_grid``: MAP-049).
    2. *The members.*  Away from those planes the pass is smooth: no
       member's ``update`` has a kink of its own inside the ball (the
       condition every nonlinear group's bound has, CPL-088).
    3. *The smooth theory.*  For the polynomial piece ``p`` the pass is
       at ``x_k``, taken as a map of the whole space, Newton-Kantorovich
       holds with a constant ``h = beta L eta <= 1/2`` (the step measures
       ``h`` on the secant from ``x_k`` to ``x_N`` and the bound is
       ``inf`` where it reads 1/2 or more).  Then ``p`` has a fixed point
       ``x_p`` within ``t* <= 2 eta`` of ``x_k`` and within ``t* - eta <=
       eta`` of the Newton point, ``||D (x_p - x_N)||_2 <= eta``,
       whatever ``h`` is under one half: the radius does not use the
       measured value (:data:`GEOMETRY_PLANE_REACH`, two, is ``t* /
       eta`` at its largest: one step to the Newton point, one more
       around it).
    4. *The floats.*  The measured residual is the exact pass's to within
       its float resolution, which the resolvent carries into the Newton
       step and into ``eta``, once each (``floor`` is twice that); and a
       position within :data:`GEOMETRY_PLANE_ULPS` float resolutions of a
       plane is on it (:func:`_reader_plane_distance`).
    5. *The norm.*  ``||D v||_2 <= R`` bounds entry ``j`` by ``R / D_j``
       where ``D_j > 0``, and bounds nothing where ``D_j = 0`` (an entry
       the norm does not read counts as on a plane).

    For every entry ``j`` of a position field the pass reads from the
    iterate or from the state the same pass has built (*readers*, those of
    :func:`_geometry_plane_limit`: a constant of the pass is not one), let
    ``d_j`` be the distance from ``x_k[j]`` to the nearest plane, zero
    where the position the pass *builds* at either point it is evaluated
    at -- ``F(x_k)[j]`` (*f_k*) and ``F(x_N)[j]`` (*f_newton*), what a
    Gauss-Seidel sweep reads after its holder's update -- is not strictly
    inside that cell (it is further than ``d_j`` from ``x_k[j]``, or on a
    plane):

        ``margin = min_j (D_j d_j) / (moved_j + radius)``.

    Where ``margin > 1``: entry ``j`` of the Newton point is ``|delta_j|``
    from ``x_k[j]``, and by 3, 4 and 5 entry ``j`` of ``x_p`` is within
    ``radius / D_j`` of that, so every position of ``x_N`` and of ``x_p``
    is in the cell ``x_k``'s is in;
    the positions the pass builds at ``x_k`` and at ``x_N`` are in it by
    the two evaluations, and at ``x_p`` they are ``x_p``'s own; so by 1
    and 2 the pass *is* ``p`` at the three points, ``x_p`` is a fixed
    point of the pass, both Jacobians the bound takes and the one it
    bounds are ``p``'s, and the bound is the smooth one.  Where it is not,
    a plane is in the ball and ``gradient_bound_usable`` is withdrawn
    whatever ``h`` reads.  No point is sampled to speak for a plane it is
    not at: the ball is covered by a distance, and the two evaluations
    answer only for themselves.

    Returns a scalar in the weights' dtype: ``inf`` with no reader, NaN
    where a position, a move or the radius is not a number.
    """
    work = weights.dtype
    zero = jnp.zeros((), work)
    nearest = jnp.full((), jnp.inf, work)
    for where, shape, dtype, mapping in readers:
        at = jnp.asarray(where, jnp.int32)
        start = x_k[at].astype(work)
        distance = _reader_plane_distance(mapping, x_k[at].reshape(shape).astype(dtype), work)
        unknown = jnp.isnan(distance)
        for built in (f_k[at], f_newton[at]):
            on_a_plane = _reader_plane_distance(mapping, built.reshape(shape).astype(dtype), work)
            inside = (jnp.abs(built.astype(work) - start) < distance) & (on_a_plane > 0)
            unknown = unknown | jnp.isnan(on_a_plane)
            distance = jnp.where(inside, distance, zero)
        distance = jnp.where(unknown, jnp.full((), jnp.nan, work), distance)
        reach = moved[at].astype(work) + radius
        # ``0 / 0`` (on a plane, with nothing to reach it by) is no margin.
        ratio = (distance * weights[at]) / jnp.where(reach > 0, reach, 1.0)
        ratio = jnp.where(reach > 0, ratio, jnp.where(distance * weights[at] > 0, jnp.inf, 0.0))
        ratio = jnp.where(jnp.isnan(distance) | jnp.isnan(reach), jnp.nan, ratio)
        nearest = jnp.where(jnp.any(jnp.isnan(ratio)) | jnp.isnan(nearest), jnp.nan,
                            jnp.minimum(nearest, jnp.min(ratio)))
    return nearest


def _gradient_error_bound_body(step_pure, probed, x_sg, consts_sg, d, rho,
                               arnoldi_residual, amplification, res, readers=()):
    """The arithmetic of :func:`_gradient_error_bound_at`, on stopped inputs."""
    from maddening.core.coupling.acceleration import (  # noqa: PLC0415
        _spectral_norm,
        ift_gradient_error_bound,
        jacobian_range_basis,
        resolvent_apply,
        spectral_error_bound,
    )

    # The analysis dtype (``_analysis_dtype``): ``d`` arrives in it, and the
    # map's tangents and outputs are cast to and from the group's own dtype
    # at the boundary -- no-ops for a float32 or wider group.
    dtype = d.dtype
    x_dtype = x_sg.dtype
    live = (d > 0).astype(dtype)
    s = jnp.where(d > 0, d, jnp.ones_like(d))
    s_inv = 1.0 / s
    nan = jnp.full((), jnp.nan, dtype)
    keys = jax.random.split(jax.random.PRNGKey(1), len(consts_sg))
    plan = _probe_plan(consts_sg, probed)
    n_rows = len(plan)
    # Per row the constant it moves and the entry (``-1``: the whole
    # constant along one direction); one more row past the probes moves
    # nothing (the Kantorovich row, below).
    row_const = jnp.asarray([i for i, _ in plan] + [-1], jnp.int32)
    row_entry = jnp.asarray([j for _, j in plan] + [0], jnp.int32)
    whole = {i for i, j in plan if j < 0}
    direction = {i: _probe_direction(consts_sg[i], keys[i]) for i in whole}
    entry_scale = {i: _probe_direction(consts_sg[i], None) for i in set(probed) - whole}

    def tangent_for(row):
        """The constants' tangent for probe ``row``: one entry (or one constant), the rest held."""
        out = []
        for i, c in enumerate(consts_sg):
            mine = row_const[row] == i
            if i in direction:
                out.append(jnp.where(mine, direction[i], jnp.zeros_like(c)))
            elif i in entry_scale:
                one_hot = (jnp.arange(c.size) == row_entry[row]).reshape(c.shape).astype(c.dtype)
                out.append(jnp.where(mine, one_hot * entry_scale[i], jnp.zeros_like(c)))
            elif jnp.issubdtype(c.dtype, jnp.floating):
                out.append(jnp.zeros_like(c))
            else:
                out.append(np.zeros(c.shape, dtype=jax.dtypes.float0))
        return tuple(out)

    # The state tangent lifted out of the underflow range where the group
    # is that small, as ``_spectral_rate_at`` takes it (``pow2_frame``'s lift).
    lift = pow2_frame(s_inv, mode="lift")

    def matvec(z):
        """``J(x_k)`` in the scaled coordinates."""
        _, Jv = jax.jvp(
            lambda xx: _F_dispatch(step_pure, xx, consts_sg), (x_sg,),
            ((z * (s_inv * lift)).astype(x_dtype),)
        )
        return (s / lift) * Jv.astype(dtype)

    U, M, captured = jacobian_range_basis(matvec, x_sg.shape[0], dtype=dtype)

    def rhs_for(row):
        return jax.jvp(
            lambda cc: _F_dispatch(step_pure, x_sg, cc), (consts_sg,), (tangent_for(row),)
        )

    rows = jnp.arange(n_rows)
    f_k, w = jax.vmap(rhs_for)(rows)          # the primal is unbatched inside
    w = w.astype(dtype)
    # Entry by entry in a power-of-two frame: a raw ``F(x) - x`` of a
    # group near 1e-34 flushed to zero, and the bound read 0.0, usable.
    r_s = _framed_difference(f_k[0].astype(dtype), x_sg.astype(dtype), s)

    def norm(v):
        return jnp.linalg.norm(live * v, axis=-1)

    # The residual's float resolution in the norm used below: ``C`` units
    # of ``eps * max|field|`` per live entry, which the scaling ``s``
    # makes ``C * eps`` each, ``eps`` the entry's *own* field's (``res``;
    # see ``residual_precision_floor``, which is the same quantity in the
    # group's own norm).  It enters twice.  The
    # distance carries it, so a stalled float32 iterate -- ``F(x) == x``
    # bitwise, residual ``0.0`` -- is not reported at the fixed point.
    # And where the residual is not above it, the residual carries no
    # direction to take the curvature along (it is rounding, or exactly
    # zero), so a fixed-seed floor-sized vector stands in: its resolvent
    # image is dominated by the slowest mode, which is where an error the
    # rounding left behind is amplified to.
    #
    # The *step* the curvature is taken across is sized by the coarsest
    # field's resolution, not entry by entry: the secant is a difference
    # of ``G``, and ``G`` is rounded in each *output* field's dtype, so a
    # step that moves a float32 input by a float32-sized amount changes a
    # float16 output that reads it by less than the float16 output can
    # hold.  Sized entry by entry the secant read exactly ``0.0`` on a
    # float16 field beside a float32 one and the bound ``0.0``, usable,
    # against a true error of 3.2%.  In a group whose fields share one
    # dtype ``coarse`` is that dtype's and nothing changes.
    floor = _floor_of(res, live)
    coarse = jnp.max(jnp.where(live > 0, res, jnp.zeros_like(res)))
    floor_dir = coarse * live * jax.random.rademacher(
        jax.random.PRNGKey(3), x_sg.shape, dtype,
    )
    resolved = norm(r_s) > _floor_of(jnp.full_like(res, coarse), live)
    r_dir = jnp.where(resolved, r_s, floor_dir)
    delta_s = resolvent_apply(U, M, r_dir, matvec(r_dir))
    # Each probe's response brought to the state's own relative size by an
    # exact power of two (``pow2_frame``), and its constant's tangent with
    # it: every quantity below is linear in the probe, and the bound is a
    # ratio of two of them, so the scale cancels exactly.  A probe is sized
    # by its constant's magnitude, which says nothing about the state's:
    # an all-zero bias probed at 1.0 beside a state near 1e-30 handed the
    # map a tangent 1e30 times the state, and a node's ``1/u`` derivative
    # overflowed (a non-finite tangent, which read NaN, unusable).
    probe_frame = jax.vmap(lambda v: pow2_frame(v, mode="common"))(s * w)
    t_s = jax.vmap(lambda ws: resolvent_apply(U, M, ws, matvec(ws)))(
        (s * w) * probe_frame[:, None])
    frame_ext = jnp.concatenate([probe_frame, jnp.ones((1,), dtype)])

    def linearisation(xx, row, ts):
        """``G_row(xx)``: the map's JVP at ``xx`` along ``(t_row, c_dot_row)``."""
        c_dot = tuple(
            c if c.dtype == jax.dtypes.float0 else (c * frame_ext[row]).astype(c.dtype)
            for c in tangent_for(row))
        _, out = jax.jvp(
            lambda x_, c_: _F_dispatch(step_pure, x_, c_),
            (xx, consts_sg), ((ts * s_inv).astype(x_dtype), c_dot),
        )
        return out

    # Both points through one batched evaluation, so ``G`` at ``x_k``
    # and at ``x_k + delta`` are the same computation on two inputs,
    # and bit-identical wherever the map's JVP does not depend on the
    # point.  The barrier keeps the two materialised before they are
    # subtracted, and the weights are applied *after* the difference:
    # with ``s * G1 - s * G0`` XLA contracts one product into a fused
    # multiply-add and the difference comes out as that product's
    # rounding error -- measured 2.9e-10 on an affine map, which the
    # resolvent then amplified into a bound of 2.5e-4 where the true
    # error is exactly zero.
    # The shifted point in a per-entry power-of-two frame: in state units
    # the step ``delta * s_inv`` of a group near 1e-34 is below the normal
    # range and was flushed, so ``G`` was evaluated twice at ``x_k``.
    points = jnp.stack([x_sg, _framed_shift(x_sg.astype(dtype), delta_s, s_inv).astype(x_dtype)])
    # One extra row beside the probes: the tangent ``delta`` itself with
    # no constant moved, so its ``G`` is ``J(x) delta`` and the secant of
    # that row is how much the *Jacobian* changes across the step (see
    # the leading-order check below).  ``tangent_for`` of an index past
    # the probes moves no constant.
    rows_ext = jnp.arange(n_rows + 1)
    ts_ext = jnp.concatenate([t_s, delta_s[None]], axis=0)
    G = jax.vmap(
        lambda xx: jax.vmap(lambda row, ts: linearisation(xx, row, ts))(rows_ext, ts_ext)
    )(points)
    G = jax.lax.optimization_barrier(G).astype(dtype)
    # Their difference is a change of a value near the state's own size:
    # taken entry by entry in a power-of-two frame, so it is not flushed.
    secant_ext = _framed_difference(G[1], G[0], s)
    secant_s, jac_secant_s = secant_ext[:-1], secant_ext[-1]

    amp = spectral_error_bound(jnp.ones((), dtype), rho, arnoldi_residual, amplification)
    # Never below the Newton step itself: ``delta = (I - J)^{-1} r`` is
    # ``x* - x_k`` exactly on an affine map, so the spectral bound, which
    # bounds the same vector through the Krylov compression, can only
    # fall short of it by rounding or where that compression missed part
    # of the range.  (NaN and ``inf`` pass through ``maximum``.)
    distance = jnp.maximum(
        spectral_error_bound(norm(r_s), rho, arnoldi_residual, amplification, floor=floor),
        norm(delta_s))

    # **The resolvent applied to each secant, not its norm.**  The error
    # is ``(I - J)^{-1}`` applied to the change of ``G``, and the factor
    # the spectral bound applies (``amp``, the Arnoldi resolvent norm) is
    # ``(I - J)^{-1}`` *restricted to the Krylov space* ``span(v0, r)`` --
    # exact for the residual, which lies in it, and not for a secant,
    # which need not: on an affine scalar Gauss-Seidel ring stopped at
    # three passes it was 8.57 where the full resolvent norm is 45.2, and
    # the bound read 0.19x the true gradient error with
    # ``gradient_bound_usable=True`` (MADD-ANO-142).  The range basis
    # applies the resolvent exactly for one more JVP per probe
    # (``resolvent_apply``), so the per-probe bound is
    # ``distance * ||(I - J)^{-1} secant_i|| / (||delta|| * ||t_i||)``.
    # On a map that is affine in the state the secant *is* the change of
    # ``G`` between ``x_k`` and ``x*`` (``delta = x* - x_k`` exactly), so
    # the bound is then the true error times ``distance / ||delta|| >= 1``.
    def resolve(v):
        return resolvent_apply(U, M, v, matvec(v))

    resolved_secant = jax.vmap(resolve)(secant_s)
    # ``amp`` is kept for the conventions alone: NaN where nothing was
    # computed and ``inf`` where nothing contracts read as before.
    conventions = ift_gradient_error_bound(
        amp, distance, norm(secant_s), norm(delta_s), norm(t_s),
    )
    along_step = ift_gradient_error_bound(jnp.ones((), dtype), distance, norm(resolved_secant),
                                          norm(delta_s), norm(t_s))
    step = norm(delta_s)
    beta, rows = _full_resolvent_norm_and_rows(
        U, M, lambda: _scaled_transpose(step_pure, x_sg, consts_sg, s, s_inv, lift, dtype))
    # **The part of the distance that has no direction.**  The secant
    # above is the change of ``G`` along ``delta``, which is the direction
    # to the fixed point only as far as the residual it was solved from is
    # the exact map's.  The measured residual is that plus the pass's own
    # rounding, of norm up to the floor and of no known direction, so
    # ``x* - x_k = delta + R e`` with ``||e|| <= floor``; and at an iterate
    # whose residual is not above the floor ``delta`` is a stand-in and the
    # whole distance has no direction.  A probe's ``G`` can change many
    # times faster along another direction than along that one: a gain
    # that multiplies a delivered value two source entries nearly cancel
    # in has a small tangent and a change of ``G`` per unit distance set
    # by the entries, and the fixed-seed stand-in read 5.1e-6 for a true
    # error of 3.0e-5 on a float32 ring of mapped edges stalled at its
    # floor (eight sign patterns of the stand-in: 1.7e-6 to 2.4e-5; the
    # operator norm times the distance: 3.9e-5).  So that part of the
    # distance is multiplied by the *operator norm* of ``z -> (I -
    # J)^{-1} dG_i/dx z`` over the directions the Jacobian reads (its row
    # space, as the Kantorovich constant below; the whole space where the
    # range basis is square): one forward-over-forward product and one
    # Jacobian-vector product per probe and direction, the derivative
    # exact (no step to size, and exactly zero where ``G`` does not depend
    # on the point, so an affine group with additive parameters still
    # reads ``0.0``).
    if rows is None:
        # No transpose, so no basis of the row space: the whole space
        # where that is at most a probed constant's size, else nothing
        # bounds the undirected part (``inf`` below unless ``G`` is
        # constant along the step).
        rows_g = (jnp.eye(x_sg.shape[0], dtype=dtype)
                  if x_sg.shape[0] <= GRADIENT_PROBE_ENTRY_LIMIT else None)
    else:
        rows_g = rows
    if rows_g is None:
        any_dir = jnp.where(norm(secant_s) > 0, jnp.inf, jnp.zeros_like(norm(secant_s)))
    else:
        # The direction is handed to the map at order one and the scale
        # returned afterwards (a power of two, so nothing is rounded): with
        # both tangents in state units their product left float32's range
        # for a group near 2**100, and the map's second derivative dropped
        # out of the norm.  Near 2**-100 it still does (the other tangent,
        # the adjoint's own, is in state units, and framing it too loses
        # the term at 2**100 instead): there a nonlinear map's undirected
        # term is its first-order part alone, 3.5% of the bound on the
        # pair of ``tests/core/test_coupling_bounds_in_any_units.py``.
        unit = pow2_frame(s_inv * lift, mode="common")

        def g_derivative(row, ts, z):
            """``(I - J)^{-1} dG_row/dx z`` in the scaled coordinates."""
            _, out = jax.jvp(lambda xx: linearisation(xx, row, ts), (x_sg,),
                             ((z * (s_inv * lift * unit)).astype(x_dtype),))
            return resolve(((s / lift) / unit) * out.astype(dtype))

        dG = jax.vmap(
            lambda row, ts: jax.vmap(lambda z: g_derivative(row, ts, z))(rows_g.T))(
                jnp.arange(n_rows), t_s)
        any_dir = _spectral_norm(live[None, None, :] * dG)
    # The undirected distance: the floor through the resolvent (the larger
    # of the Krylov factor the distance itself uses and the full norm),
    # or the whole distance at an unresolved iterate.
    floor_reach = spectral_error_bound(jnp.zeros((), dtype), rho, arnoldi_residual,
                                       amplification, floor=floor)
    if beta is not None:
        floor_reach = jnp.maximum(floor_reach, beta * floor)
    undirected = jnp.where(resolved, jnp.minimum(floor_reach, distance), distance)
    tangent_size = norm(t_s)
    any_dir_term = jnp.where(
        any_dir > 0,
        undirected * any_dir / jnp.where(tangent_size > 0, tangent_size, 1.0),
        jnp.zeros_like(any_dir))
    per_probe = jnp.where(
        jnp.isfinite(conventions),
        jnp.where(resolved, along_step + any_dir_term, jnp.maximum(along_step, any_dir_term)),
        conventions,
    )
    # The worst probe the fixed point responds to.  A responding probe
    # whose bound is NaN (a non-finite secant) poisons the maximum
    # rather than dropping out of it; NaN where nothing responds, or
    # where the basis did not capture the range.
    # A probe whose tangent could not be computed (non-finite) counts as
    # responding, so its NaN bound poisons the maximum; it used to drop
    # out silently where the tangent was NaN (``NaN > 0`` is False).
    #
    # **And only a probe the pass resolves.**  A relative error divides by
    # the tangent's size, and a tangent that is rounding has none: where
    # moving the constant by the probe's whole size (its own magnitude)
    # moves one pass by no more than the pass's float resolution -- the
    # floor the distance carries, in this norm -- the right-hand side can
    # be what the iterate's last rounding, or a cancellation inside the
    # node's own derivative, left of it, and so is the tangent solved from
    # it.  That is the same statement as a zero tangent (the fixed point
    # does not respond to the probe) made at the resolution it can be made
    # at, and such a probe is not in the worst either.  It used to be: for
    # the centre and the curve of a nonlinearity evaluated on its centre
    # (a constant the fixed point does not respond to; right-hand sides of
    # 2e-33 to 5e-18 beside ones of order one, float64) the bound read
    # 1.09 for a relative error of 1.134 on a converged hub, usable.  The
    # absolute error for such a constant is of the size of its tangent
    # (1.7e-34 there, beside gradients of 3.5e3).  The rule is the
    # magnitude's, so a weak constant whose tangent is exact is left out
    # with the cancelled one: nothing measured here tells them apart (the
    # tangent's sensitivity to a floor-sized change of the state read 1.6%
    # for that curve).  A right-hand side that is not finite still counts
    # (its NaN bound poisons the maximum).
    rhs_size = norm(s * w)
    resolved_probe = jnp.logical_or(rhs_size > floor, jnp.logical_not(jnp.isfinite(rhs_size)))
    responds = jnp.logical_and(
        jnp.logical_or(norm(t_s) > 0, jnp.logical_not(jnp.isfinite(norm(t_s)))),
        resolved_probe)
    worst = jnp.max(jnp.where(responds, per_probe, -jnp.inf))
    worst = jnp.where(jnp.any(responds), worst, nan)

    # **Where the leading-order term is not the whole story.**  The bound
    # above takes the resolvent and the distance at ``x_k``; the exact
    # error needs the resolvent at ``x*``, and a distance that holds for
    # a map whose Jacobian moves.  Newton-Kantorovich supplies both from
    # what is already measured.  With ``beta = ||(I - J(x_k))^{-1}||`` --
    # the *full-operator* norm (``_full_resolvent_norm``), not the
    # Krylov-restricted ``amp``, which can be several times smaller
    # (MADD-ANO-142) -- ``eta = ||delta||`` (the Newton correction) and
    # ``L`` the Jacobian's Lipschitz constant -- estimated across the step,
    # the larger of the extra row's ``||(J(x_k + delta) - J(x_k)) delta|| /
    # ||delta||**2`` and the operator norm of ``J(x_k + delta) - J(x_k)``
    # over ``||delta||`` (below) -- the check is ``h = beta L eta < 1/2``.  (The
    # affine-covariant form, the exact resolvent applied to that change
    # of the Jacobian along ``delta`` alone, was measured too and is not
    # enough: on the stiff ``x <- a + g u**2`` pair at three passes it
    # read ``h = 0.48`` where ``beta L eta = 0.67``, and the bound came out
    # 0.999x the true error.)  Where it holds a fixed
    # point exists within ``t* = eta (1 - sqrt(1 - 2h)) / h`` of ``x_k``
    # and ``||(I - J(x*))^{-1}|| <= beta / sqrt(1 - 2h)``, so the bound is
    # multiplied by ``1 / sqrt(1 - 2h)`` and its distance is the larger
    # of the spectral bound and ``t*``.  Where it fails, nothing measured
    # at ``x_k`` bounds the resolvent at the fixed point, and the bound
    # is ``inf`` -- ``gradient_bound_usable`` False.  ``h`` is exactly
    # zero on a map whose Jacobian does not depend on the point (the
    # extra row's secant is then bit-identical zero), so an affine
    # group's bound is untouched.  Measured on ``x <- a + g u**2`` at
    # ``F'(x*) = 0.99`` with the forward 0.65-4.5% short of its fixed
    # point, the uncorrected bound read 0.20-0.96x the true relative
    # error with the flag True; ``h`` there is 0.48-0.58.
    jac_change = norm(jac_secant_s)
    if beta is None or rows is None:
        # No transpose to take the full norm with (a node the map cannot be
        # reverse-differentiated through, in a group larger than the range
        # basis): a map whose Jacobian does not move (``jac_change == 0``,
        # bit for bit on an affine map) needs no ``beta``; any other has no
        # certified bound.
        numerator = jnp.where(jac_change > 0, jnp.inf, jnp.zeros_like(jac_change))
    else:
        # **The affine-covariant constant, as an operator.**  ``h`` was
        # ``beta * ||(J(x_k + delta) - J(x_k)) delta|| / ||delta||``: the
        # Jacobian's change along ``delta`` alone, which can be much smaller
        # than its change in the directions ``delta`` hardly moves.  Then
        # ``h`` passed where the fixed point lay outside the radius it
        # certified: on a bilinear pair stopped two passes in, ``h = 0.37``
        # with the fixed point 0.568 away against ``t* = 0.552``, and the
        # bound read 0.81x the true gradient error, usable.  So the check
        # also takes Deuflhard's affine-covariant Kantorovich constant,
        # ``omega = ||(I - J(x_k))^{-1} (J(x_k + delta) - J(x_k))|| /
        # ||delta||`` -- an operator norm, on the directions the Jacobian
        # reads (its row space, which the full resolvent norm's reverse-mode
        # products already span, ``k`` of them), with the resolvent applied
        # exactly through the range basis -- and ``h`` is the larger of the
        # two: 0.65 on that pair, no certified bound.  ``beta`` times the
        # Jacobian's operator change would also have caught it, but reads
        # 3.5-16x the directional value on a ring whose Jacobian depends on
        # one field the Newton step barely moves, withdrawing bounds that
        # hold; the resolvent in front measures the change where it lands.
        # ``3 k`` more JVPs, both points through one batched evaluation and
        # differenced after a barrier, so an affine map's change is exactly
        # zero, as the extra row's is.
        def jac_at(xx, z):
            _, Jz = jax.jvp(lambda x_: _F_dispatch(step_pure, x_, consts_sg), (xx,),
                            ((z * (s_inv * lift)).astype(x_dtype),))
            return Jz

        JR = jax.vmap(lambda xx: jax.vmap(lambda z: jac_at(xx, z))(rows.T))(points)
        JR = jax.lax.optimization_barrier(JR).astype(dtype)
        change = live * _framed_difference(JR[1], JR[0], s / lift)
        op_change = _spectral_norm(live * jax.vmap(resolve)(change))
        numerator = jnp.maximum(beta * jac_change, op_change * step)
    h = jnp.where(
        step > 0, numerator / jnp.where(step > 0, step, 1.0), 0.0,
    )
    certified = h < 0.5
    # ``t* - eta``, the most the Newton step can miss the fixed point by,
    # in the form that does not cancel (see the helper).
    root, miss = _kantorovich_root_and_miss(step, h)
    t_star = step + miss
    stretch = jnp.where(distance > 0,
                        jnp.maximum(distance, t_star) / jnp.where(distance > 0, distance, 1.0),
                        1.0)
    safe_root = jnp.where(certified, jnp.where(root > 0, root, 1.0), 1.0)
    factor = stretch / safe_root
    # **The second-order term: the Newton step's miss.**  The secant is
    # ``G`` across ``delta``, and ``x* - x_k = delta + e`` with ``||e|| <=
    # t* - eta`` (Kantorovich), so the error's ``G(x*) - G(x_k)`` differs
    # from the secant by ``G(x_k + delta + e) - G(x_k + delta)``, of size at
    # most ``Lip(G) ||e||``.  Stretching the secant to ``t*`` covers only
    # the part of ``e`` along ``delta``; the rest can point where ``G``
    # changes faster.  So each probe adds that term, with ``Lip(G)`` read
    # off its own secant (``||G(x_k + delta) - G(x_k)|| / eta``, along
    # ``delta`` as ``L`` is) and the full resolvent norm at ``x*``
    # (``beta / root``) applied to it: ``beta ||secant_i|| (t* - eta) /
    # (eta ||t_i|| root)``, the largest over the responding probes added
    # to the largest leading-order term.  Exactly zero where the Jacobian
    # does not move (``h = 0``, ``miss = 0``), so an affine group's bound
    # is the one it was, to its last bit or so (the extra operations can
    # move how XLA fuses the rest: one ulp on one of 125 configurations
    # measured).  With ``h`` along ``delta`` alone and without it, the
    # bound read 0.986x the true gradient error, usable, on a bilinear
    # pair stopped two passes in (``h = 0.19``): the leading-order term
    # took the change of ``G`` along the Newton direction for its change
    # along ``x* - x_k``.  With the affine-covariant ``h`` (0.22 there)
    # the leading-order term alone reads 1.07x, and on 294 usable drawn
    # bilinear pairs it never fell below the true error; this term adds up
    # to 42% of it.  It stays because it is the part of Kantorovich's
    # argument the stretch to ``t*`` does not cover, not because a
    # measured case needed it.
    if beta is None:
        # ``h`` is 0 here or the bound is not certified: no miss to carry.
        extra = jnp.zeros_like(worst)
    else:
        tangent_norm = norm(t_s)
        took = jnp.logical_and(responds, miss > 0)
        den = jnp.where(jnp.logical_and(took, tangent_norm > 0), step * tangent_norm, 1.0)
        extra_i = jnp.where(took, beta * norm(secant_s) * miss / den, 0.0)
        extra = jnp.max(extra_i)
    worst = jnp.where(certified, worst * factor + extra / safe_root,
                      jnp.full_like(worst, jnp.inf))
    if not readers:
        return jnp.where(captured, worst, nan)
    # **Which polynomial piece the fixed point is in** (a pass that reads
    # a moving geometry; ``_kantorovich_ball_plane_margin`` has the
    # argument).  The check above compares the Jacobian at ``x_k`` and at
    # the Newton point and cannot see a lattice plane beyond both, so the
    # gradient's flag stands only where no plane is in the Kantorovich
    # ball around ``x_k`` (``t* <= 2 eta`` at any ``h <= 1/2``: not the
    # measured ``h``), entry by entry: the Newton step's own move of the
    # entry, and one more step ``eta`` around the Newton point
    # (``GEOMETRY_PLANE_REACH`` steps in all where the entry carries the
    # whole step).  Plus the part of the distance that has no direction,
    # the residual's rounding through the resolvent: once in the Newton
    # point and once in ``eta``.  At an iterate whose residual is not
    # above its floor the step is a stand-in and the whole distance has
    # no direction.  One more evaluation of the pass, at the Newton point.
    f_newton = _F_dispatch(step_pure, points[1], consts_sg)
    radius = (GEOMETRY_PLANE_REACH - 1.0) * jnp.where(resolved, step, distance) + (
        GEOMETRY_PLANE_REACH * undirected)
    margin = _kantorovich_ball_plane_margin(
        readers, x_sg, f_k[0], f_newton, d, jnp.abs(live * delta_s), radius)
    return jnp.where(captured, worst, nan), jnp.where(captured, margin, nan)
