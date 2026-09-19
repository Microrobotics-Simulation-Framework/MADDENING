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

    Dividing is safe by construction — the denominator is only ever
    used where ``ref > atol``, and elsewhere the ``where`` selects a
    zero contribution — so a field that is legitimately at zero neither
    divides by something tiny nor blocks convergence forever.
    """
    ref = _field_reference(new_val, old_val)
    scale = rtol * ref
    active = jnp.logical_and(ref > atol, scale > 0)
    safe = jnp.where(active, scale, jnp.ones_like(scale))
    diff = jnp.abs(new_val - old_val)
    return jnp.where(active, diff / safe, jnp.zeros_like(diff)), active


def coupling_residual_l2(
    s_new: dict[str, dict],
    s_old: dict[str, dict],
    node_names: list[str],
    atol: float = 1e-8,
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
        treated as being at zero and contributes nothing.  See
        :func:`_scaled_change`.

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
        as zero.
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
    atol: float = 1e-8,
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
        Dead band, in the interface quantity's own units.
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
    ``GraphManager.coupling_diagnostics``' ``bound_valid``.  Rejecting
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
    ``bound_valid`` reports a usable *ratio*, not a valid *bound*.  The
    full list of what the estimate rests on is in
    ``graph_manager._fixed_point_while``; the decision it feeds is in
    ``benchmarks/results/audit_040_final/ERROR_BOUND_DECISION.md``.
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


def float_image(v):
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
    if jax.dtypes.issubdtype(dt, jax.dtypes.prng_key):
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
    flat: jnp.ndarray,
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
