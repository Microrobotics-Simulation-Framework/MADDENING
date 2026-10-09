"""Convergence reports of a coupled block: the diagnostics record, the
strict-convergence error and the folding of per-solve reports.

Moved verbatim out of ``maddening.core.graph_manager``.  Private.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core._graph_specs import _META_KEY
from maddening.core.coupling.acceleration import _has_entries

#: The ``_meta`` slots ``coupling_diagnostics()`` reads a group's report
#: from, as ``coupling_{group key}_{suffix}``.  A step writes them; they
#: describe that step and the group it ran under, unlike the warm starts
#: (``_V`` / ``_W``, ``_pred_*``), which are state.
_REPORT_SLOT_SUFFIXES = (
    "iterations", "total_iterations", "residual", "amplification",
    "rho_spectral", "spectral_residual", "spectral_amplification",
    "gradient_relative_error_bound", "pass_evaluations", "reading_floor",
    "geometry_gap", "geometry_plane_limit",
)


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
    """A per-group diagnostics mapping that still answers two development-era names.

    ``coupling_diagnostics()`` renamed two fields during 0.4.0's
    development, because each called itself a *bound* (no release
    carried the old names; 0.3.x reported neither field):

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


# In an unbroken run of write-then-step (a state write before every
# step), the positions at which the underflow-range check is made: every
# one of the first few, then each power of two, then one in every so many.
_UNDERFLOW_RUN_ALWAYS = 8
_UNDERFLOW_RUN_EVERY = 128


def _underflow_check_due(run: int) -> bool:
    """Is the underflow-range check made at the *run*-th step (from 1) of
    an unbroken run of steps that each follow a state write?

    The check reads every field of every coupled group twice on the host,
    which costs several times a small graph's step (measured: a
    three-entry pair's loop of ``set_node_state`` and ``step`` went from
    45 to 170 microseconds, one of 100,000 entries from 0.6 to 1.1 ms).
    One write, or a few, is asked at the next step, every time.  A loop
    that writes before every step is asked at its first
    ``_UNDERFLOW_RUN_ALWAYS`` steps, at each power of two after that and
    at every ``_UNDERFLOW_RUN_EVERY``-th from there on: a number of checks
    that grows with the logarithm of the loop's length and then by one in
    ``_UNDERFLOW_RUN_EVERY`` steps, instead of one per step.
    """
    return (run <= _UNDERFLOW_RUN_ALWAYS or run % _UNDERFLOW_RUN_EVERY == 0
            or (run < _UNDERFLOW_RUN_EVERY and run & (run - 1) == 0))


def _underflow_range_fields(groups, state) -> dict[str, list]:
    """``{group key: [(node, field, magnitude, dtype), ...]}``, smallest first.

    The floating fields of each coupled group whose magnitude ``max|field|``
    is nonzero, finite and below ``finfo(dtype).tiny / finfo(dtype).eps``:
    the range where a change of one ulp of the field is subnormal.  An
    exactly zero field (or an empty one) is never listed -- zero is not a
    small unit -- nor is a non-finite one, which the coupling verdict
    already reports.  Read on the host (``numpy``), so it compiles nothing.
    """
    out: dict[str, list] = {}
    for group in groups:
        key = "+".join(sorted(group.nodes))
        hits = []
        for nn in sorted(group.nodes):
            for fld, value in sorted((state.get(nn) or {}).items()):
                # The dtype first: a typed PRNG key (or any other extended
                # dtype) cannot be converted to a numpy array at all.
                # ``jnp``'s predicates and ``finfo``, because numpy's do not
                # know bfloat16; neither traces anything.
                dtype = getattr(value, "dtype", None)
                if dtype is None or not jnp.issubdtype(dtype, jnp.floating):
                    continue
                arr = np.asarray(jax.device_get(value))
                if not _has_entries(arr):
                    continue        # no entries: no magnitude (``_has_entries``)
                mag = float(np.max(np.abs(arr.astype(np.float64))))
                info = jnp.finfo(arr.dtype)
                if 0.0 < mag < float(info.tiny) / float(info.eps):
                    hits.append((nn, fld, mag, arr.dtype))
        if hits:
            out[key] = sorted(hits, key=lambda h: h[2])
    return out


def _strict_convergence_messages(group) -> tuple[str, str]:
    """``(non-finite, unconverged)``: what ``strict_convergence`` raises for *group*.

    Two checks with exclusive predicates, so the message names the cause.
    The first fires on a non-finite *state*, decided from the state itself
    (:func:`_state_measurable`): a field that is NaN or inf, or whose
    magnitude is beyond the range its dtype can measure a change at --
    since 0.4.0 the norm reports ``inf`` for such a field rather than
    dropping it (MADD-ANO-019), and it is the one case where no amount of
    iteration would help.  The second fires on every other unconverged
    exit, a non-finite *estimate* on a finite state included: it was once
    taken for the first, and a float16 group whose norm overflowed at the
    default ``rtol`` on a finite state was told it had diverged and that no
    larger cap would help, when the cap was the whole story.  Shared by
    the in-graph checks and by ``run_adaptive*``, which raise them only
    about a solve the step keeps.
    """
    head = (f"coupling group {sorted(group.nodes)} exited at "
            f"max_iterations={group.max_iterations} without converging")
    return (
        head + ": its state is non-finite (a field is NaN, inf, or beyond "
        "the range its dtype can measure a change at), so the coupling "
        "residual is non-finite and the IFT gradient is invalid here. The "
        "iteration diverged and no larger max_iterations would help; check "
        "the relaxation and the node updates, or set strict_convergence=False "
        "to only report this via coupling_diagnostics().",
        head + "; the IFT gradient is invalid here. Raise max_iterations, "
        "loosen the tolerance, or set strict_convergence=False to only "
        "report this via coupling_diagnostics().",
    )


def _strict_error_if(value, pred, msg, mesh=None):
    """``equinox.error_if(value, pred, msg)``, raised on every device of *mesh* at once.

    ``error_if`` raises from a host callback, and in a program partitioned
    over several devices a callback runs once, on the first device: the
    others went on to the step's next all-reduce and waited there for a
    device that had stopped, until XLA reported the stuck rendezvous and
    aborted the process (SIGABRT after about a minute), so the error never
    reached Python (MADD-ANO-162).  With *mesh* the check runs inside a
    ``shard_map`` replicated over every device of it: *pred* is one value on
    all of them (a verdict on the group's reduced residual), so every device
    raises at the same program point and none is left waiting.

    The callback returns a zero token rather than *value* itself: handed
    *value*, the replicated ``shard_map`` would gather a sharded field onto
    every device and hand it back replicated.  The token is OR-ed into the
    bits of each floating leaf -- an exact identity no compiler can fold
    away, which keeps the check live (a callback whose result is unused is
    removed) without rounding, flushing or moving a sign.  Without *mesh*
    (one device) this is ``error_if`` itself, unchanged.
    """
    import equinox as eqx  # noqa: PLC0415  (lineax transitive dep)

    if mesh is None:
        return eqx.error_if(value, pred, msg)
    from jax import shard_map  # noqa: PLC0415
    from jax.sharding import PartitionSpec  # noqa: PLC0415

    token = shard_map(
        lambda p: eqx.error_if(jnp.zeros((), jnp.uint8), p, msg),
        mesh=mesh, in_specs=PartitionSpec(), out_specs=PartitionSpec(),
        check_vma=False,
    )(jnp.asarray(pred))

    def tie(v):
        v = jnp.asarray(v)
        if not jnp.issubdtype(v.dtype, jnp.floating):
            return v
        return _bit_tie(v, token)

    return jax.tree.map(tie, value)


@jax.custom_jvp
def _bit_tie(v, token):
    """``v`` with the zero ``token`` OR-ed into its bits: ``v`` itself, bit for bit,
    with a data dependency on ``token`` no compiler can fold away.

    A bit operation has no derivative (``bitcast_convert_type`` has none to
    give), so the rule below makes the tie an identity to differentiation
    as well: without it a strict step on several devices would have broken
    ``jax.grad`` through the very solve the check guards.
    """
    bits = jnp.dtype(f"uint{jnp.finfo(v.dtype).bits}")
    return jax.lax.bitcast_convert_type(
        jax.lax.bitcast_convert_type(v, bits) | token.astype(bits), v.dtype)


@_bit_tie.defjvp
def _bit_tie_jvp(primals, tangents):
    v, token = primals
    v_dot, _token_dot = tangents
    return _bit_tie(v, token), v_dot


def _raise_if_a_kept_solve_failed(messages: dict, verdicts: Sequence[dict]) -> None:
    """``strict_convergence`` for ``run_adaptive``: raise about a kept solve.

    ``verdicts`` are the ``collect_strict`` verdicts of the steps the
    stepper keeps (the two half steps of an accepted attempt), and
    ``messages`` the builder's ``strict_messages``.  Raises
    ``RuntimeError`` -- what the in-graph check raises through
    ``jax.jit`` is a subclass -- with the message the in-graph check
    would have given, non-finite named first.
    """
    for key, (nonfinite_msg, unconverged_msg) in messages.items():
        found = [v[key] for v in verdicts if key in v]
        if any(bool(nf) for nf, _ in found):
            raise RuntimeError(nonfinite_msg)
        if any(bool(uc) for _, uc in found):
            raise RuntimeError(unconverged_msg)


#: The ``_meta`` report slots that describe one solve, beside the pass
#: counts: the per-solve keys :func:`_fold_kept_half_step_reports` takes
#: from one half step as a set, so ``coupling_diagnostics()`` derives
#: every value it reports from one solve.
_PER_SOLVE_REPORT_SUFFIXES = (
    "residual", "amplification", "rho_spectral", "spectral_residual",
    "spectral_amplification", "gradient_relative_error_bound", "pass_evaluations",
    "geometry_gap", "geometry_plane_limit",
)


def _fold_kept_half_step_reports(groups, first_state, second_state):
    """``second_state`` with each group's report covering both kept half steps.

    An accepted ``run_adaptive*`` attempt keeps two solves per coupling
    group, its two half steps, and ``strict_convergence`` checks both
    (see :func:`_raise_if_a_kept_solve_failed`).  The ``_meta`` report
    slots used to be whatever the second half step wrote, so
    ``coupling_diagnostics()`` said ``converged=True`` about an accepted
    step whose first half step had exited at ``max_iterations``
    unconverged -- the step ``strict_convergence=True`` refuses -- and
    the report, the strict check and the docs, which promise them one
    verdict, gave two.  Folded here, per group that has a report:

    * ``iterations`` is the larger half's count, so the cap check
      ``iterations >= max_iterations`` reads "a kept solve exhausted its
      budget", as it does across waveform sweeps;
    * ``total_iterations`` is the sum, where the group owns that slot
      (``waveform_iterations > 1``; a one-sweep group has no slot to
      hold it, and reports ``iterations``);
    * the per-solve slots (residual, amplification, the spectral triple,
      the gradient bound) are taken together from one half -- the first
      when it alone did not converge, else the second, which produced
      the returned state -- so ``converged`` is ``False`` exactly when a
      kept solve did not converge, and every key is still a statement
      about one solve.

    Each half's verdict is the criterion the solve, ``strict_convergence``
    and :func:`~maddening.core.coupling.acceleration.reported_converged`
    apply: ``estimated_error(residual, amplification, step_scale) <=
    threshold`` in the residual's dtype.  A non-finite residual is
    unconverged.  Pure ``jax.numpy`` on scalars, traced into
    ``run_adaptive_scan``'s program and jitted once per call of
    ``run_adaptive``; the states are otherwise untouched.
    """
    from maddening.core.coupling.acceleration import (  # noqa: PLC0415
        convergence_criterion,
        estimated_error,
    )

    if _META_KEY not in second_state or _META_KEY not in first_state:
        # No report slots to fold (a graph without coupling groups, or a
        # hand-built state): the state keeps its structure exactly.
        return second_state
    first_meta = first_state[_META_KEY]
    meta = dict(second_state[_META_KEY])
    for group in groups:
        key = "+".join(sorted(group.nodes))
        iter_key = f"coupling_{key}_iterations"
        if iter_key not in meta or iter_key not in first_meta:
            continue
        threshold, scale = convergence_criterion(group)

        def converged(m, key=key, threshold=threshold, scale=scale):
            res = jnp.asarray(m[f"coupling_{key}_residual"])
            amp = jnp.asarray(m[f"coupling_{key}_amplification"])
            return estimated_error(res, amp, scale) <= threshold

        use_first = jnp.logical_and(jnp.logical_not(converged(first_meta)),
                                    converged(meta))
        meta[iter_key] = jnp.maximum(first_meta[iter_key], meta[iter_key])
        total_key = f"coupling_{key}_total_iterations"
        if total_key in meta and total_key in first_meta:
            meta[total_key] = first_meta[total_key] + meta[total_key]
        for suffix in _PER_SOLVE_REPORT_SUFFIXES:
            slot = f"coupling_{key}_{suffix}"
            if slot in meta and slot in first_meta:
                meta[slot] = jnp.where(use_first, first_meta[slot], meta[slot])
    return {**second_state, _META_KEY: meta}
