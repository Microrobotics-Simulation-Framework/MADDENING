"""System identification helpers built on the graph parameter pytree.

Three pieces, the first two pure JAX so they compose with ``jax.jit`` /
``jax.grad``:

* :func:`windowed_loss` — a teacher-forced, windowed trajectory loss.
  Long rollouts of stiff or chaotic dynamics give exploding gradients; the
  standard remedy is to reset the simulation to the measured state every
  ``window`` samples and sum the per-window losses.  Each window is an
  independent ``lax.scan``, so memory is O(window) and gradients cannot
  compound across windows.
* :func:`fim` — the Fisher information matrix ``J^T J`` of a residual
  function, from ``jax.jacfwd`` sensitivities.  Its eigen-decomposition
  says which parameter *combinations* the data cannot distinguish; the
  eigenvector of a near-zero eigenvalue names them.
* :func:`fit` — Adam on a loss over the params pytree, in the
  unconstrained coordinates and under the trainable mask the graph's
  :class:`~maddening.core.params.ParamSpec` declarations define.

All take the ``params`` pytree exactly as ``GraphManager.params`` holds
it, so the same object flows into ``gm.run_scan(params=...)``, an
optimiser, and these diagnostics.

Usage::

    obs = observations_from_history(init_state, history)   # T samples
    loss = jax.jit(lambda p: windowed_loss(
        gm, p, obs, obs_fn=lambda h: h["spring"]["position"], window=20))
    g = jax.grad(loss)(gm.params)

    report = fim(lambda p: residuals(p), gm.params)
    report.eigvecs[:, 0]      # least identifiable direction
"""

from __future__ import annotations

import functools
import math
import numbers
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from maddening.core.coupling.acceleration import (
    estimated_error,
    relaxation_step_scale,
)
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
# ``_resolve_specs`` is the one walk that decides which ParamSpec governs
# each params leaf -- the walk ``trainable_mask`` / ``constrain`` /
# ``check_bounds`` use -- and ``_validate_specs_mirror`` is the same walk
# refusing, in addition, an entry that reaches no leaf.  Duplicating
# either here would be a second definition of "which spec governs this
# leaf", which is how a namedtuple level once resolved two ways.
from maddening.core.params import (
    DEFAULT_SPEC,
    _resolve_specs,
    _validate_specs_mirror,
)
from maddening.warnings import PrecisionLimitWarning

_META_KEY = "_meta"

#: Factor either side of the rank cutoff within which a float32 rank
#: verdict is treated as not reproducible.  Measured on this module's
#: own arithmetic -- ``F = J.T @ J`` and ``eigh`` in float32 -- against a
#: float64 reference applying the *same* rank rule, over 228,000
#: synthetic Fisher matrices of known spectrum (n = 2..25 parameters,
#: m = 20..2000 residual rows, spread / clustered / twin-null spectra),
#: in three independently seeded sweeps.  The tables below are the last
#: of them (57,000 matrices, seed 771003); the other two agree with it
#: to the digits shown.
#:
#: **Re-measured when the cutoff gained its ``sqrt(m)`` term, because the
#: two are not independent.**  The band is the width of the arithmetic's
#: error *expressed in units of the cutoff*, so moving the cutoff moves
#: the band and the factor fitted to it.  The same sweep, scored against
#: both cutoffs -- disagreement rate binned by the matrix's *true*
#: eigenvalue ratio:
#:
#: =====================  ==============  ==================
#: true ratio / cutoff    at ``n * eps``  at ``max(n,√m)*eps``
#: =====================  ==============  ==================
#: 0.10 -- 0.22           0.0077          0.0000
#: 0.22 -- 0.32           0.0117          0.0000
#: 0.32 -- 0.46           0.0176          0.0006
#: 0.46 -- 0.68           0.0432          0.0114
#: 0.68 -- 1.0            0.1148          0.0467
#: 1.0  -- 1.47           0.1075          0.0422
#: 1.47 -- 2.15           0.0043          0.0000
#: 2.15 -- 3.16           0.0010          0.0000
#: 3.16 and above         0.0000          0.0000
#: =====================  ==============  ==================
#:
#: The population itself shrank, from 1.21% of draws to 0.35%: two
#: thirds of the verdicts that used to rest on rounding do not any more,
#: because the cutoff no longer sits *below* the arithmetic's own floor
#: at long residuals.  What is left is gathered tightly on the cutoff
#: instead of smeared over decades below it.
#:
#: **The factor stays at 2.0, and it now buys something different.**
#: What the band actually tests is the *float32* deciding ratio, and at
#: a disagreement that lies in ``[0.566x, 1.562x]`` of the cutoff over
#: all three sweeps -- asymmetric, with the lower edge binding, so
#: symmetric coverage needs at least 1.77.  Recall over the
#: disagreements, and fire rate on verdicts the two precisions *agree*
#: about, by true ratio:
#:
#: =========  ========  ==========  =========  ======
#: factor     recall    2x .. 5x    5x .. 10x  10x+
#: =========  ========  ==========  =========  ======
#: 1.5        0.9949    0.0000      0.000      0.000
#: 1.8        1.0000    0.0002      0.000      0.000
#: 2.0        1.0000    0.0110      0.000      0.000
#: 2.5        1.0000    0.2522      0.000      0.000
#: 3.0        1.0000    0.4578      0.000      0.000
#: 8.0        1.0000    1.0000      0.692      0.000
#: =========  ========  ==========  =========  ======
#:
#: Against the old cutoff the same sweep put 2.0's recall at **0.912**
#: (0.916 in the earlier pair); against this one it is 1.000.  The ~9%
#: that were unreachable were the
#: long-residual verdicts -- not because no factor was wide enough, but
#: because the cutoff was below the floor there, so the disagreements
#: were not near the cutoff at all.  Making the cutoff see ``m`` is what
#: brought them into reach; no change to this constant could have.
#:
#: 2.0 rather than the 1.8 that first attains full recall, because 1.8
#: is fitted to the single most extreme of 803 disagreements (1/1.8 =
#: 0.556 against an observed edge of 0.566, i.e. no margin at all on an
#: extreme order statistic), while 2.0 keeps 13% of margin for a false
#: fire rate of ~1% on ordinary ``2x..5x`` verdicts -- less than the
#: 2.8% it cost under the old cutoff.  Past 2.0 the curve turns: 2.5
#: fires on a quarter of ordinary ``2x..5x`` reports and 8 on two thirds
#: of ``5x..10x`` ones, which this project's own spring-damper
#: identification tests produce routinely, for no recall at all.  A
#: warning that fires routinely gets suppressed, which is worse than
#: silence.
#:
#: Full recall here is a statement about 803 measured disagreements, not
#: a proof.  The residual risk is sampling error on that population,
#: which is a far better place to be than the previous structural blind
#: spot; ``TestPrecisionLimitedRank`` holds the separation from both
#: sides so that a later edit cannot quietly give it up.
_PRECISION_WARN_FACTOR = 2.0


@stability(StabilityLevel.EVOLVING)
def observations_from_history(
    initial_state: dict[str, dict], history: dict[str, dict],
) -> dict[str, dict]:
    """Prepend the initial state to a ``run_scan_with_history`` history.

    :func:`windowed_loss` wants sample ``0`` to be the state the
    trajectory starts from; ``run_scan_with_history`` records states
    *after* each step.  Returns a pytree with leading axis ``T = 1 + n``.
    """
    return jax.tree.map(
        lambda s0, h: jnp.concatenate([jnp.asarray(s0)[None], h], axis=0),
        initial_state, history,
    )


def _leading_len(tree) -> int:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        raise ValueError("observations pytree has no leaves")
    return int(leaves[0].shape[0])


def _group_thresholds(gm) -> list[tuple[str, str, float, float]]:
    """``(residual key, amplification key, threshold, step scale)`` per group.

    The pair of keys is what the convergence criterion is built from:
    the flag tests ``omega * residual / (1 - rho)`` -- an estimate of
    the distance to the fixed point -- and not the residual alone, so a
    mask derived here agrees with
    ``GraphManager.coupling_diagnostics()['converged']``.  ``omega`` is
    the step scale (see
    :func:`~maddening.core.coupling.acceleration.relaxation_step_scale`)
    and has to be carried too, or an over-relaxed group would be masked
    on a different criterion from the one it converged under.
    """
    out = []
    for g in gm._coupling_groups:  # noqa: SLF001
        key = "+".join(sorted(g.nodes))
        thr = 1.0 if g.convergence_norm in ("mixed", "interface") else float(g.tolerance)
        out.append((f"coupling_{key}_residual",
                    f"coupling_{key}_amplification", thr,
                    relaxation_step_scale(g.acceleration, g.relaxation)))
    return out


@stability(StabilityLevel.EVOLVING)
def windowed_loss(
    gm,
    params: dict,
    observations: dict[str, dict],
    *,
    obs_fn: Callable[[dict], Any],
    window: int,
    sample_every: int = 1,
    external_inputs: Optional[dict] = None,
    mask_unconverged: bool = False,
    window_states: Optional[dict] = None,
    continuity_weight: float = 0.0,
) -> jnp.ndarray:
    """Windowed squared-error loss of a graph against data.

    Teacher-forced by default (every window restarts from the measured
    state); with ``window_states`` it is **multiple shooting**: window
    ``w`` restarts from the free state ``window_states[w]`` and a
    continuity penalty ties each window's end to the next window's start,
    so the fitted trajectory is a single continuous solution at the
    optimum instead of ``n_windows`` teacher-forced pieces.

    Parameters
    ----------
    gm : GraphManager
        Compiled graph.  Only its step function and coupling-group config
        are used; ``gm._state`` is not modified.
    params : dict
        Graph parameter pytree (``gm.params`` layout).  Differentiate the
        returned loss with respect to it.
    observations : pytree
        Ground-truth **full user state** with a leading time axis of
        length ``T``: sample ``k`` is the state after ``k * sample_every``
        base steps, sample ``0`` the initial state — the layout
        :func:`observations_from_history` produces.  The full state is
        needed because each window *resets* the simulation to it; the
        measured subset the loss compares is selected by ``obs_fn``.
    obs_fn : callable
        Maps a state pytree with a leading time axis to the measured
        quantities (a pytree of arrays, same leading axis).
    window : int
        Samples per window.  ``T - 1`` must be a multiple of it, and it
        must lie in ``[1, T - 1]`` (``T == 1`` has nothing to fit).  Every
        window starts from the ground-truth sample at its start and
        integrates ``window * sample_every`` steps; the loss compares the
        ``window`` simulated samples against the next ``window``
        observations.
    sample_every : int
        Base steps between consecutive observation samples; ``>= 1``.
    external_inputs : dict, optional
        Static external inputs, completed and validated as in
        ``gm.step``: zeros for every declared input this does not
        supply, and a ``ValueError`` for an undeclared ``node.field``.
    mask_unconverged : bool
        Multiply a window's loss by 0 when any coupling group exited at
        ``max_iterations`` unconverged during it (the IFT gradient is
        unreliable there).  Uses the always-on residual in ``_meta``.
    window_states : pytree, optional
        Multiple shooting: free initial **user states** per window, a
        pytree with leading axis ``n_windows = (T - 1) // window``
        (:func:`init_window_states` seeds it from the observations).
        Differentiate the loss with respect to these too and optimise
        them jointly with ``params`` (:func:`fit_multiple_shooting`).
    continuity_weight : float
        Weight of the continuity penalty ``Σ_w ||end_w - window_states[w+1]||²``
        (sum over every user-state leaf) under multiple shooting; ignored
        when ``window_states`` is ``None``.

    Returns
    -------
    jnp.ndarray
        Scalar: the sum over windows and samples of the squared error of
        ``obs_fn`` outputs (plus the continuity penalty under multiple
        shooting).
    """
    if gm._dirty or gm._compiled_step is None:  # noqa: SLF001
        gm.compile()
    step_fn = gm._build_step_fn()  # noqa: SLF001
    # Completed and validated exactly as ``gm.step(external_inputs=)``
    # does: a fit whose forcing was silently dropped by a typo would
    # move the parameters to make up for the missing input.
    ext = gm._resolve_external_inputs(external_inputs)  # noqa: SLF001

    observations = {k: v for k, v in observations.items() if k != _META_KEY}
    if sample_every <= 0:
        # ``lax.scan(length=0)`` would advance the simulation by nothing
        # and compare each window's *initial* state against the next
        # ``window`` observations: a plausible-looking number that is not
        # a loss.
        raise ValueError(f"sample_every={sample_every} must be >= 1")
    T = _leading_len(observations)
    if window <= 0 or (T - 1) % window != 0 or window > T - 1:
        # ``window > T - 1`` is only reachable at ``T == 1``, where every
        # window "divides" ``T - 1 == 0``.  The scan below is still traced
        # once for zero windows, so it used to die inside
        # ``dynamic_slice_in_dim`` instead of saying what was wrong.
        raise ValueError(
            f"window={window} must divide T-1={T - 1} and lie in "
            f"[1, T-1] (T={T} samples)"
        )
    n_windows = (T - 1) // window

    meta0 = gm._state.get(_META_KEY)  # noqa: SLF001
    thresholds = _group_thresholds(gm) if mask_unconverged else []

    def _state_from_obs(obs_k, k):
        s = {nn: dict(fields) for nn, fields in obs_k.items()}
        if meta0 is not None:
            m = jax.tree.map(jnp.zeros_like, meta0)
            if "step_count" in m:
                m["step_count"] = jnp.asarray(
                    k * sample_every, dtype=meta0["step_count"].dtype,
                )
            s[_META_KEY] = m
        return s

    def _converged(state):
        ok = jnp.array(True)
        meta = state.get(_META_KEY, {})
        for key, amp_key, thr, scale in thresholds:
            if key in meta:
                amp = meta.get(amp_key, jnp.zeros_like(meta[key]))
                ok = ok & (estimated_error(meta[key], amp, scale) <= thr)
        return ok

    def _advance_one_sample(carry, _):
        def inner(c, _):
            s, ok = c
            s = step_fn(s, ext, params)
            return (s, ok & _converged(s)), None

        (state, ok), _ = jax.lax.scan(inner, carry, None, length=sample_every)
        user = {k: v for k, v in state.items() if k != _META_KEY}
        return (state, ok), user

    if window_states is not None:
        n_ws = _leading_len(window_states)
        if n_ws != n_windows:
            raise ValueError(
                f"window_states has leading axis {n_ws}, expected n_windows={n_windows}"
            )
        window_states = {k: v for k, v in window_states.items() if k != _META_KEY}

    def _window(w, _):
        start = w * window
        if window_states is None:
            obs_start = jax.tree.map(
                lambda x: jax.lax.dynamic_index_in_dim(x, start, keepdims=False),
                observations,
            )
        else:
            obs_start = jax.tree.map(
                lambda x: jax.lax.dynamic_index_in_dim(x, w, keepdims=False),
                window_states,
            )
        state0 = _state_from_obs(obs_start, start)
        (final, ok), sim = jax.lax.scan(
            _advance_one_sample, (state0, jnp.array(True)), None, length=window,
        )
        truth = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, start + 1, window),
            observations,
        )
        sq = jax.tree.map(
            lambda a, b: jnp.sum((a - b) ** 2), obs_fn(sim), obs_fn(truth),
        )
        # `sum` is typed as returning `int` for its empty-sequence start
        # value; every leaf here is an Array, so the result is one.
        loss_w: Any = sum(jax.tree.leaves(sq))
        if mask_unconverged:
            loss_w = loss_w * ok.astype(loss_w.dtype)
        if window_states is not None and continuity_weight > 0.0:
            # Tie this window's end to the next window's free start
            # (no penalty after the last window).
            nxt = jax.tree.map(
                lambda x: jax.lax.dynamic_index_in_dim(
                    x, jnp.minimum(w + 1, n_windows - 1), keepdims=False),
                window_states,
            )
            end_user = {k: v for k, v in final.items() if k != _META_KEY}
            gap = jax.tree.map(lambda a, b: jnp.sum((a - b) ** 2), end_user, nxt)
            pen = sum(jax.tree.leaves(gap))
            loss_w = loss_w + continuity_weight * pen * (w < n_windows - 1)
        return w + 1, loss_w

    _, losses = jax.lax.scan(_window, jnp.int32(0), None, length=n_windows)
    return jnp.sum(losses)


@stability(StabilityLevel.EVOLVING)
def init_window_states(observations: dict, window: int) -> dict:
    """Seed multiple-shooting ``window_states`` from the observations: the
    measured user state at every window start (leading axis ``n_windows``)."""
    observations = {k: v for k, v in observations.items() if k != _META_KEY}
    T = _leading_len(observations)
    if window <= 0 or (T - 1) % window != 0 or window > T - 1:
        raise ValueError(
            f"window={window} must divide T-1={T - 1} and lie in [1, T-1]"
        )
    n_windows = (T - 1) // window
    return jax.tree.map(lambda x: x[: n_windows * window : window], observations)


@stability(StabilityLevel.EVOLVING)
@dataclass(frozen=True, kw_only=True)
class FIMReport:
    """Fisher information of a residual with respect to the parameters.

    Keyword-only by construction.  ``rank`` was inserted between
    ``eigvecs`` and ``cond`` during 0.4.0, which moved every field after
    it: positional construction then assigned ``cond`` to ``rank`` and so
    on, with no ``TypeError`` and no warning, and a wrong ``rank`` is a
    wrong identifiability verdict -- the same failure as the ``crb``
    fail-safe inversion, arriving by a different route.  ``kw_only``
    makes that class of break impossible, here and for the next field.

    ``eigvals`` ascend; ``eigvecs[:, i]`` is the direction for
    ``eigvals[i]`` in the (possibly relatively scaled) parameter space
    ordered as ``param_names``.

    ``rank`` counts the directions the data actually resolves: the
    eigenvalues strictly above ``rank_rtol * eigvals[-1]``, where
    ``rank_rtol`` defaults to ``max(n, sqrt(m)) * eps`` for ``n``
    parameters and ``m`` residual rows at the matrix's own precision.
    The ``n`` term is the relative form ``numpy.linalg.matrix_rank``
    uses and bounds ``eigh``; the ``sqrt(m)`` term is the error in
    forming ``F = J.T @ J`` over ``m`` rows, which ``F`` itself cannot
    show and which the cutoff ignored before 0.4.0 --
    :func:`~maddening.sysid._resolve_rank_rtol` has the measurement.
    ``rank < len(param_names)`` says
    the data leaves that many independent parameter combinations
    undetermined.  Unlike ``cond`` the verdict does not move when the
    residual is rescaled (by ``noise_std``, say), because the threshold
    scales with the matrix; a float32 ``cond`` can flip between a finite
    number and ``inf`` under exactly that rescaling.

    The verdict is a comparison of two numbers, and at float32 it can be
    a comparison of two numbers that differ by less than the
    decomposition can resolve.  :func:`fim` says so when that happens --
    a :class:`~maddening.warnings.PrecisionLimitWarning` naming the
    measured ratio, the cutoff and the ``jax_enable_x64`` re-run that
    settles it -- rather than reporting the coin-flip as a fact.  No
    warning means the deciding ratio was more than
    ``_PRECISION_WARN_FACTOR`` away from the cutoff, not that the
    problem is well conditioned.

    ``crb`` is the Cramér–Rao lower bound on each parameter's variance
    (unit noise variance; relative variance under ``scale="relative"``).
    It is ``+inf`` for every parameter with support in the null space --
    no unbiased estimator of such a parameter has finite variance, since
    the data cannot separate it from the combinations that null space
    mixes it with -- and the diagonal of the inverse over the resolved
    subspace for the rest.  ``+inf`` rather than ``NaN`` because it
    fails safe: ``crb < threshold`` is then False for an unidentifiable
    parameter instead of quietly propagating a ``NaN``.  The bound is
    reported finite only where it was positively established, so a
    decomposition that resolves nothing reads ``+inf`` as well, never
    ``0.0`` -- the direction the fail-safe exists to cover.

    ``zero_scaled`` names the parameters ``scale="relative"`` found
    sitting at exactly ``0.0``.  Relative scaling multiplies each column
    of ``J`` by the parameter's own value, which asks a question with no
    content at zero: the column vanishes however well the data determine
    the parameter, so it joins the null space, ``rank`` drops by one and
    its ``crb`` is ``+inf``.  That ``+inf`` is a fact about the scaling,
    not about the data -- ``fim(..., scale=None)`` asks the absolute
    question and answers it -- and this field is how the report says
    which of the two happened.  Empty under ``scale=None``.

    ``value_scaled`` names the parameters ``scale="nominal"`` could not
    give a nominal scale.  Nominal scaling multiplies each column by the
    width ``hi - lo`` of the parameter's declared ``bounds`` instead of
    by its value, so a parameter at ``0.0`` keeps a coherent
    dimensionless column; a spec with no finite width has no such scale,
    and that column is scaled by the value (``p``, or ``p - lo`` for a
    ``transform="log"`` spec with a lower bound) exactly as
    ``"relative"`` would.  This field says which columns those were, so
    a report under ``"nominal"`` is never quietly half relative; a
    value-scaled column that is also at zero appears in ``zero_scaled``
    too.  Empty under every other scale.  The policy is tabulated in
    :func:`fim`.

    ``integer_excluded`` names the leaves of integer or boolean dtype
    that are **not** in the matrix -- one entry per leaf, by its key
    path, since none of their entries is a column.  There is no
    derivative with respect to an integer; left in, such a leaf became a
    zero column and read as unidentifiable whatever the data said (an
    integer ``matrix_mapping`` reported rank 0 of 16).  Only ``mask=None``
    leaves one out; a ``mask`` that selects one is refused.  Store a
    leaf as floating-point to analyse it.
    """
    fim: jnp.ndarray
    eigvals: jnp.ndarray
    eigvecs: jnp.ndarray
    rank: int
    cond: float
    crb: jnp.ndarray
    param_names: tuple[str, ...]
    zero_scaled: tuple[str, ...] = ()
    value_scaled: tuple[str, ...] = ()
    integer_excluded: tuple[str, ...] = ()

    def least_identifiable(self) -> tuple[str, float]:
        """Name and weight of the largest component of the weakest direction."""
        v = np.abs(np.asarray(self.eigvecs[:, 0]))
        i = int(np.argmax(v))
        return self.param_names[i], float(v[i])


def _leaf_size(leaf) -> int:
    """Number of entries in a params leaf, **without reading it**.

    ``int(np.asarray(leaf).size)`` -- what this used to be, in three
    places -- pulls the whole leaf across the device boundary to read a
    number that is part of its shape and therefore already known on the
    host.  On a jax array ``np.asarray`` goes through the C buffer
    protocol, so it does not show up as an ``__array__`` call, and
    before Python 3.12 gave that protocol the PEP 688 ``__buffer__``
    dunder it does not show up as *any* attribute access; it is a
    device-to-host transfer all the same, and it blocks on whatever
    computation produced the leaf.  ``np.shape`` reads ``.shape`` and
    transfers nothing, and falls back to ``np.asarray`` only for a leaf
    that has no shape of its own (a Python float), where there is
    nothing on a device to wait for.
    """
    return int(math.prod(np.shape(leaf)))


def _is_differentiable(leaf) -> bool:
    """Whether a params leaf has a derivative to take: a floating (or
    complex) dtype.  Read from the dtype, which the host already has, so
    it transfers nothing and works on a tracer."""
    return bool(jnp.issubdtype(jnp.result_type(leaf), jnp.inexact))


def _param_names(params, *, differentiable_only: bool = False) -> tuple[str, ...]:
    """One name per column, in ``ravel_pytree`` order.

    A leaf of one entry is named by its key path; a 1-D leaf's entries
    as ``path[i]``; an N-D leaf's as ``path[i, j, ...]``, the NumPy index
    of the element in the leaf.  Row-major flattening is what orders the
    columns, but a flat position is not an index into the leaf: labelled
    ``['H'][5]``, the unobserved element ``H[0, 5]`` of a 6x6 matrix
    read, as the NumPy index it looks like, as the whole of row 5.
    """
    names: list[str] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        if differentiable_only and not _is_differentiable(leaf):
            continue
        base = jax.tree_util.keystr(path)
        shape = tuple(np.shape(leaf))
        n = _leaf_size(leaf)
        if n == 1:
            names.append(base)
        elif len(shape) <= 1:
            names.extend(f"{base}[{i}]" for i in range(n))
        else:
            names.extend(f"{base}[{', '.join(str(int(k)) for k in ix)}]"
                         for ix in np.ndindex(*shape))
    return tuple(names)


def _leaf_paths(tree) -> list[str]:
    return [jax.tree_util.keystr(p)
            for p, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]


def _mask_flags(params: dict, mask: dict) -> list:
    """Leaves of ``mask``, refusing a mask not shaped like ``params``.

    Every reader of a mask zips its leaves against the params leaves in
    flatten order, so a mask with the right leaf *count* and the wrong
    keys is not a mismatch anything notices: the flags simply land on
    whichever parameter occupies that position.  The audit passed a mask
    keyed by the caller's own symbol names, marked ``damping``, and
    watched the fit move ``stiffness`` from 30.0 to 76.3 -- the wrong
    science, with no symptom at all.  ``tree_structure`` compares the
    keys the leaf count cannot, and naming the first path that differs
    is what makes the error fixable.
    """
    params_def = jax.tree_util.tree_structure(params)
    mask_def = jax.tree_util.tree_structure(mask)
    if mask_def == params_def:
        return jax.tree.leaves(mask)
    p_paths, m_paths = _leaf_paths(params), _leaf_paths(mask)
    where = ""
    for i in range(max(len(p_paths), len(m_paths))):
        if p_paths[i:i + 1] != m_paths[i:i + 1]:
            where = (
                f" They first differ at leaf {i}: params has "
                f"{p_paths[i] if i < len(p_paths) else '<no such leaf>'}, "
                f"mask has {m_paths[i] if i < len(m_paths) else '<no such leaf>'}."
            )
            break
    raise ValueError(
        "mask must have the same tree structure as params, key for key. "
        "The flags are read in flatten order, so a mask with the right "
        "number of leaves and different keys is not refused by a count: it "
        "quietly fits whichever parameter sits at that position."
        f"{where} params has {len(p_paths)} leaves {p_paths[:6]}"
        f"{' ...' if len(p_paths) > 6 else ''}; mask has {len(m_paths)} "
        f"leaves {m_paths[:6]}{' ...' if len(m_paths) > 6 else ''}. Build the "
        "mask from the params tree itself -- jax.tree.map over params, or "
        "gm.trainable_mask(params) with the leaves you do not want dropped "
        "to False."
    )


def _masked_indices(params: dict, mask: Optional[dict]) -> Optional[np.ndarray]:
    """Flat indices (in ``ravel_pytree`` order) of the leaves ``mask``
    marks True; ``None`` when there is no mask."""
    if mask is None:
        return None
    leaves = jax.tree.leaves(params)
    flags = _mask_flags(params, mask)
    idx, offset = [], 0
    for leaf, flag in zip(leaves, flags):
        n = _leaf_size(leaf)
        if bool(flag):
            idx.extend(range(offset, offset + n))
        offset += n
    if not idx:
        raise ValueError("mask selects no parameters")
    return np.asarray(idx, dtype=np.intp)


def _fim_indices(params: dict, mask: Optional[dict]):
    """``(idx, integer_excluded)`` for :func:`fim` / :func:`fim_core`.

    ``idx`` indexes the flat vector of the **differentiable** leaves only
    (:func:`_is_differentiable`), in flatten order -- the vector
    :func:`_fim_jacobian` differentiates -- and is ``None`` when every one
    of them is a column.  An integer or boolean leaf has no derivative:
    ``ravel_pytree`` used to promote it into the float vector and cast it
    back on the way out, JAX's derivative through that cast is
    identically zero, and the leaf came back as a zero column -- "the
    data cannot determine this", however well they do, with nothing in
    the report to say otherwise.  An integer ``matrix_mapping`` read as
    rank 0 of 16.

    So an integer leaf is never a column.  With ``mask=None`` -- "every
    leaf", implicitly -- it is left out and named in
    ``integer_excluded``; a ``mask`` that selects one *explicitly* asks
    for a derivative that does not exist and is refused, naming it.
    """
    entries = jax.tree_util.tree_flatten_with_path(params)[0]
    keep = [_is_differentiable(leaf) for _, leaf in entries]
    n_diff = sum(_leaf_size(leaf) for (_, leaf), k in zip(entries, keep) if k)
    if mask is None:
        excluded = tuple(jax.tree_util.keystr(path)
                         for (path, _), k in zip(entries, keep) if not k)
        if n_diff == 0:
            raise ValueError(
                "params has no floating-point leaf, so there is nothing to "
                "differentiate: integer and boolean leaves have no derivative "
                f"(the leaves are {list(excluded)[:6]}"
                f"{' ...' if len(excluded) > 6 else ''}). Store the "
                "parameters as floating-point arrays.")
        return None, excluded
    flags = _mask_flags(params, mask)
    asked = [jax.tree_util.keystr(path)
             for (path, leaf), k, flag in zip(entries, keep, flags)
             if bool(flag) and not k]
    if asked:
        dtypes = sorted({str(jnp.result_type(leaf))
                         for (_, leaf), k, flag in zip(entries, keep, flags)
                         if bool(flag) and not k})
        raise ValueError(
            f"mask selects {len(asked)} leaf/leaves of integer or boolean "
            f"dtype ({', '.join(dtypes)}): {asked[:6]}"
            f"{' ...' if len(asked) > 6 else ''}. A derivative with respect "
            "to an integer does not exist -- JAX's is identically zero -- so "
            "its column of J would be zero and the leaf would read as "
            "unidentifiable whatever the data say. Store it as a "
            "floating-point array if it is a parameter to analyse, or leave "
            "it out of mask.")
    idx, offset = [], 0
    for (_, leaf), k, flag in zip(entries, keep, flags):
        if not k:
            continue
        n = _leaf_size(leaf)
        if bool(flag):
            idx.extend(range(offset, offset + n))
        offset += n
    if not idx:
        raise ValueError("mask selects no parameters")
    return np.asarray(idx, dtype=np.intp), ()


def _refuse_integer_trainable(params: dict, mask: dict) -> None:
    """Refuse a fit whose trainable set holds an integer or boolean leaf.

    The optimisers move a float vector; an integer leaf rides along
    promoted to float, its gradient is identically zero, and the fit
    returned it bit-for-bit unchanged -- with a finite loss, an
    ``excited_rank`` that quietly counted it as undetermined, and no
    word about why.  ``params_pytree()`` never produces such a leaf for
    a node constant, so the one way to get here is to have asked: a
    ``set_param_spec(..., ParamSpec())`` on an integer mapping weight,
    a hand-built ``params``, or a ``mask``.
    """
    entries = jax.tree_util.tree_flatten_with_path(params)[0]
    bad = [(path, leaf) for (path, leaf), flag in zip(entries, _mask_flags(params, mask))
           if bool(flag) and not _is_differentiable(leaf)]
    if not bad:
        return
    listed = "\n".join(
        f"  - {_leaf_location(path)}  (params{jax.tree_util.keystr(path)}, "
        f"dtype {jnp.result_type(leaf)})" for path, leaf in bad)
    raise ValueError(
        f"the trainable set holds {len(bad)} leaf/leaves of integer or "
        f"boolean dtype:\n{listed}\nAn optimiser cannot move an integer: "
        "its gradient is identically zero, so the fit would hand it back "
        "unchanged while reporting a loss as if it had been fitted. Store it "
        "as a floating-point array (e.g. matrix_mapping(H.astype(float))) if "
        "it is a parameter to fit, or keep it out of the trainable set -- "
        "ParamSpec(trainable=False), the default for mapping weights, or a "
        "mask that leaves it out.")


def _leaf_location(path) -> str:
    """Human description of a ``params`` leaf path: ``"node 's', parameter
    'damping'"`` for a node constant, ``"mapping '<edge>', weight 'H'"``
    for an interface-mapping weight."""
    keys = [k for k in (getattr(k, "key", None) for k in path) if k is not None]
    if len(keys) == 3 and keys[0] == "nodes":
        return f"node {keys[1]!r}, parameter {keys[2]!r}"
    if len(keys) == 3 and keys[0] == "mappings":
        return f"mapping {keys[1]!r}, weight {keys[2]!r}"
    return f"leaf {jax.tree_util.keystr(path)}"


def _resolve_mask(gm, params: dict, mask: Optional[dict]) -> dict:
    """The trainable mask the fitters optimise under.

    ``None`` means the graph's own declarations (``gm.trainable_mask``).
    An explicit ``mask`` may only *narrow* that set: the
    :class:`~maddening.core.params.ParamSpec`, not the mask, is what
    ``constrain`` / ``unconstrain`` consult, and they pass a
    ``trainable=False`` leaf through untransformed and unclipped.  A
    mask that marked such a leaf would therefore have the optimiser step
    it in physical coordinates, straight through its declared bounds, so
    it is refused here — once, before any coordinates are built — rather
    than in each fitter.

    Raises
    ------
    ValueError
        If ``mask``'s tree structure differs from ``params``' (the flags
        are read in flatten order, so different keys fit a different
        parameter), or if it marks a leaf whose ``ParamSpec`` declares
        ``trainable=False`` -- naming every such leaf and the spec change
        that would make it fittable -- or if the resolved trainable set,
        the graph's own included, holds a leaf of integer or boolean
        dtype, which no optimiser can move (its gradient is identically
        zero, and the fit used to hand it back unchanged in silence).
    """
    if mask is None:
        resolved = gm.trainable_mask(params)
        _refuse_integer_trainable(params, resolved)
        return resolved
    entries = jax.tree_util.tree_flatten_with_path(params)[0]
    flags = _mask_flags(params, mask)
    per_leaf = _resolve_specs(params, gm.param_specs())
    frozen = []
    for (path, _), flag, spec in zip(entries, flags, per_leaf):
        if bool(flag) and not spec.trainable:
            frozen.append((path, spec))
    if frozen:
        listed = "\n".join(
            f"  - {_leaf_location(path)}  "
            f"(params{jax.tree_util.keystr(path)}, bounds={spec.bounds}, "
            f"transform={spec.transform!r})"
            for path, spec in frozen
        )
        first_path, first_spec = frozen[0]
        keys = [k for k in (getattr(k, "key", None) for k in first_path)
                if k is not None]
        owner, key = (keys[1], keys[2]) if len(keys) == 3 else ("<node>", "<key>")
        raise ValueError(
            f"mask marks {len(frozen)} parameter(s) whose ParamSpec declares "
            f"trainable=False:\n{listed}\n"
            "The ParamSpec, not the mask, decides what an optimiser may move: "
            "constrain/unconstrain apply a leaf's transform and bounds only "
            "when its spec is trainable, so fitting a frozen leaf through the "
            "mask would step it in physical coordinates with no transform and "
            "no clipping and could leave its declared bounds. To fit it, make "
            "it trainable in the spec -- "
            f"gm.set_param_spec({owner!r}, {key!r}, ParamSpec(trainable=True, "
            f"bounds={first_spec.bounds}, transform={first_spec.transform!r})) "
            "-- which is what activates those bounds and that transform. A "
            "mask may only narrow the trainable set, never widen it; drop the "
            "leaf from the mask to leave it frozen."
        )
    _refuse_integer_trainable(params, mask)
    return mask


def _physical_params(gm, start: dict, flat_u, unravel, idx):
    """``theta -> physical params``, leaving the untouched leaves alone.

    The optimisers carry only ``theta`` -- the ``ravel_pytree`` entries
    the resolved ``mask`` selects -- so mapping back means
    :meth:`GraphManager.constrain` over the *whole* tree, and for a
    ``log`` or ``logit`` leaf that round trip is ``exp(log(p))`` in
    float32, exact only to about one ulp.  A leaf no step touched would
    therefore come back perturbed (a ``HeatNode``'s
    ``thermal_diffusivity`` moved by ~1e-7 relative while only
    ``length`` was masked), and a bit comparison of a calibration's
    input and output could not tell "not fitted" from "fitted and
    barely moved".

    So the test is on the coordinates rather than on the mask: a leaf
    whose unconstrained entries are bit-for-bit what they were comes
    back from ``start``, bit for bit.  That is the exact complement of
    the entries ``idx`` selects *and* the leaves a selected-but-unmoved
    step left alone -- ``fit(n_iter=0)`` and a ``tol`` that stops before
    the first update both used to return ``30.000002`` for a
    ``stiffness`` of ``30.0``, which is precisely the case
    :class:`FitResult` promises will not happen.

    Returned as a closure because every fitter needs it twice -- for the
    ``callback`` / observer pytree as well as the final one -- and the
    two must agree.
    """
    leaves_start, treedef = jax.tree.flatten(start)
    edges = np.cumsum([0] + [int(np.asarray(leaf).size) for leaf in leaves_start])
    base = np.asarray(flat_u)

    def to_params(theta):
        flat_new = flat_u.at[idx].set(theta)
        full = gm.constrain(unravel(flat_new))
        # ``!=`` rather than a tolerance: the claim is bitwise identity,
        # and a leaf an optimiser moved by one ulp *was* fitted.
        moved = np.asarray(flat_new) != base
        return jax.tree.unflatten(treedef, [
            fitted if bool(moved[lo:hi].any()) else untouched
            for untouched, fitted, lo, hi
            in zip(leaves_start, jax.tree.leaves(full), edges[:-1], edges[1:])
        ])

    return to_params


def _inverse_noise_std(noise_std, residual):
    """``1 / sigma`` flattened like ``ravel_pytree(residual)``, or ``None``.

    A real number or a 0-d array is one sigma for every residual entry;
    anything else must be a pytree matching ``residual`` (per-leaf sigma,
    scalar or broadcastable to the leaf).  ``jnp.ndim`` cannot tell the
    two apart -- it reports ``0`` for a dict -- so the scalar branch is
    keyed on the type.

    The cast and the reciprocal are computed in **numpy**, not ``jnp``,
    and only the finished array is handed to a device.  Every ``jnp``
    operation performed while a ``jax.jit`` trace is open is staged into
    that trace, concrete inputs or not, so a ``jnp`` cast here would
    make ``sig`` a tracer and :func:`_check_noise_std` -- which has to
    read it, because its job is to refuse it -- would raise
    ``TracerArrayConversionError`` instead.  That is what made
    ``fim_core(..., noise_std=...)`` untraceable at first.  numpy and
    XLA both round a float32 division correctly, so the value is
    unchanged.

    ``residual`` is used for its *structure* only -- the pytree shape,
    each leaf's shape and dtype, and the dtype ``ravel_pytree`` would
    promote them to -- so it may be (and from :func:`fim` is) a tree of
    :class:`jax.ShapeDtypeStruct` from :func:`jax.eval_shape` rather
    than a computed residual.  :func:`fim` used to evaluate
    ``residual_fn(params)`` in full to get here and then throw the
    values away, a whole extra rollout per call that bought nothing at
    all when ``noise_std`` was ``None`` -- the case that returns on the
    line below.  ``jax.eval_shape`` gets the same structure by tracing,
    with no compute and no compilation.
    """
    if noise_std is None:
        return None
    flat_r = jax.eval_shape(lambda t: ravel_pytree(t)[0], residual)
    dtype = np.dtype(flat_r.dtype)
    is_scalar = isinstance(noise_std, numbers.Real) or (
        isinstance(noise_std, (np.ndarray, jax.Array)) and noise_std.ndim == 0
    )
    if is_scalar:
        sig = np.asarray(noise_std, dtype=dtype)
        _check_noise_std(sig, noise_std)
        return jnp.asarray(np.ones((), dtype) / sig)
    sig = jax.tree.map(
        lambda leaf, sd: np.broadcast_to(
            np.asarray(sd, dtype=np.dtype(leaf.dtype)), np.shape(leaf)),
        residual, noise_std,
    )
    # ``ravel_pytree``'s own two steps -- cast every leaf to the promoted
    # dtype, then concatenate in flatten order -- done in numpy, because
    # ``dtype`` above *is* that promotion, read off the abstract ravel.
    leaves = [np.asarray(x, dtype=dtype).ravel() for x in jax.tree.leaves(sig)]
    flat_sig = (np.concatenate(leaves) if leaves
                else np.zeros(0, dtype=dtype))
    _check_noise_std(flat_sig, noise_std)
    return jnp.asarray(np.ones((), dtype) / flat_sig)


def _check_noise_std(sig, original) -> None:
    """Refuse a sigma that is not strictly positive *at the residual's
    own precision*.

    ``F = Jᵀ Σ⁻¹ J`` is a Fisher matrix only for a positive sigma, and
    the three ways it stops being one are all silent.  ``sigma = 0``, or
    a sigma that underflows the residual dtype (``1e-320`` is ``0.0`` in
    float32), divides by zero and fills ``F`` with ``inf``/``NaN``;
    ``sigma = NaN`` does the same; and a *negative* sigma gives exactly
    the answer its absolute value gives, so a sign error in a caller's
    noise model would never surface.  The check runs after the cast to
    the residual dtype because that is where the underflow happens.
    """
    arr = np.asarray(sig)
    ok = np.isfinite(arr) & (arr > 0.0)
    if bool(np.all(ok)):
        return
    flat_ok = np.atleast_1d(ok).reshape(-1)
    first = np.atleast_1d(arr).reshape(-1)[int(np.argmin(flat_ok))]
    raise ValueError(
        f"noise_std must be finite and strictly positive at the residual's "
        f"own precision; got {original!r}, which is {first} as {arr.dtype}. "
        "F = J^T Sigma^-1 J is a Fisher matrix only for a positive sigma: a "
        "zero, underflowed or NaN sigma fills F with inf/NaN (and the report "
        "then cannot tell you that it did), while a negative sigma gives "
        "exactly the answer its absolute value gives. Pass noise_std=None "
        "for unweighted sensitivities."
    )


def _resolve_rank_rtol(dtype, n: int, rank_rtol: Optional[float], *,
                       n_residual: Optional[int]) -> float:
    """The relative eigenvalue cutoff ``rank`` is decided against.

    One definition, used by the rank itself and by the check that asks
    whether that rank was decided at the noise floor: a warning derived
    from a *different* cutoff from the one in force would be describing
    a verdict nobody took.

    Parameters
    ----------
    dtype
        Precision of the decomposition; supplies ``eps``.
    n : int
        Number of parameters -- the order of ``F``.
    rank_rtol : float, optional
        A cutoff the caller stated, which is returned as given (after
        validation).  ``None`` asks for the default derived below.
    n_residual : int, optional
        Number of residual rows ``m``, i.e. ``J.shape[0]``.  Keyword-only
        and **required**, so that a call site cannot forget it and get a
        cutoff that silently ignores the residual length; pass ``None``
        where there is genuinely no ``J`` (a bare eigendecomposition),
        which falls back to the ``n``-only form.

    Notes
    -----
    The default is ``max(n, sqrt(m)) * eps``.  It is an estimate of the
    error in the eigenvalues of ``F`` as this module computes them,
    relative to ``max(eigvals)``, and it has two terms because the
    computation has two stages.

    ``eigh`` returns each eigenvalue of a symmetric matrix with an
    absolute error of order ``p(n) * eps * ||F||``; ``n * eps`` is the
    conventional stand-in for ``p(n)``, and the relative form
    ``numpy.linalg.matrix_rank`` uses.

    Forming ``F = J.T @ J`` costs again, and that cost cannot be seen
    from ``F``: each entry is an inner product over ``m`` rows, whose
    rounding error grows with ``m``.  The worst case over summation
    orders is ``m * eps * sum_k |J_ki J_kj|``, i.e. ``m * eps`` relative
    to ``max(eigvals)`` -- but that bound requires every rounding to
    align, and neither XLA's blocked accumulation nor a real Jacobian
    does that.  Measured on this module's own arithmetic, with an
    *exactly* rank-deficient ``F`` so that float64 says the answer is
    zero and anything float32 reports is the floor, the floor grows as
    ``sqrt(m)`` and not as ``m`` (p99 over 400 draws per cell, worst
    over spread / clustered / twin-null spectra, in units of ``eps``):

    ====  ======  ======  ======  ======  ======  ======
    n\\m   20      50      320     800     2000    4000
    ====  ======  ======  ======  ======  ======  ======
    2     0.75    1.13    2.63    3.88    7.13    9.13
    3     1.75    1.75    3.00    4.50    6.38    8.74
    5     2.50    2.50    2.50    2.38    2.50    2.50
    12    4.50    5.00    4.01    4.50    4.00    3.53
    25    --      5.72    6.58    6.00    6.50    6.01
    ====  ======  ======  ======  ======  ======  ======

    A factor of 200 in ``m`` moves the ``n = 2`` floor by 12x, which is
    ``sqrt(200) = 14`` and not ``200``; and the ``m`` term only overtakes
    the ``n`` term for small ``n``, which is why ``max`` rather than a
    sum.  ``sqrt(m)`` is also the textbook statistical model of an
    accumulated rounding error over ``m`` terms, so this is the expected
    law rather than a curve fitted to the table.

    The ``n``-only form is *below* that floor wherever ``sqrt(m) > n``:
    at ``n = 2, m = 4000`` the cutoff was 2 eps against a floor of 11
    eps, so ``fim`` reported directions that were not in the data as
    resolved, with a finite ``crb``.  Because the new form is a ``max``
    it can only widen: no problem's cutoff moves down, and every problem
    with ``m <= n**2`` is unaffected exactly.

    ``max(n, m) * eps`` -- the worst-case bound taken literally -- was
    considered and rejected by the same measurement.  It is 1315x the
    observed floor at its loosest, and over the sweep it rejects 45% of
    the directions that float32 and float64 *agree* are resolved, up to
    a true eigenvalue ratio of 2.4e-04, four decades above the floor.  A
    rank cutoff that discards four decades of real resolution is not a
    more careful answer, it is a different and wronger one.
    """
    if rank_rtol is None:
        floor = float(n)
        if n_residual is not None:
            floor = max(floor, math.sqrt(float(n_residual)))
        return floor * float(np.finfo(dtype).eps)
    rank_rtol = float(rank_rtol)
    if not np.isfinite(rank_rtol) or rank_rtol < 0.0:
        raise ValueError(
            "rank_rtol must be a finite non-negative number, got "
            f"{rank_rtol!r}")
    return rank_rtol


def _precision_limited(eigvals, rank_rtol: float, eps_floor: float,
                       factor: float = _PRECISION_WARN_FACTOR):
    """``(deciding ratio, cutoff)`` when ``rank`` rests on rounding, else ``None``.

    Two conditions, and the second is the one that keeps this honest.
    The ratio has to be within ``factor`` of the cutoff **and** at the
    precision floor ``eps_floor`` (``max(n, sqrt(m)) * eps``, the
    intrinsic resolution of this module's arithmetic -- the *default*
    cutoff, whatever cutoff is actually in force).  Under the default
    ``rank_rtol`` the two
    coincide and the second is implied.  They come apart the moment a
    caller *raises* ``rank_rtol``, which is a modelling decision -- "I
    call anything below 1e-3 unidentifiable in practice" -- and not a
    statement about arithmetic: an eigenvalue ratio of 2e-4 against a
    1e-3 cutoff is a close call, but it is a close call between two
    numbers float32 knows to three more decimal places, and warning
    about precision there would be simply wrong.  Lowering
    ``rank_rtol`` below the floor goes the other way and still warns,
    correctly: a cutoff under the noise floor makes every verdict noise.

    The deciding ratio is the eigenvalue ratio ``lambda_i / lambda_max``
    lying closest to ``rank_rtol`` in log distance -- the one an error
    of the wrong size would carry across the cutoff and so change
    ``rank`` by one.  It is not always ``eigvals[0]``: a spectrum with
    two near-null directions has a second ratio just as close, and a
    matrix whose smallest eigenvalue is far *below* the cutoff can still
    have a different one sitting on it.

    A ratio that is zero or negative is deliberately **not** reported.
    A non-positive eigenvalue of a PSD matrix is unambiguously rounding,
    but it is also what an exactly rank-deficient Fisher matrix produces
    -- ``scale="relative"`` with a parameter at ``0.0`` is the common
    case, and :attr:`FIMReport.zero_scaled` already explains it.
    Warning there fires on 100% of exact zero-column reports and ~65% of
    exactly-dependent ones (measured), for a verdict float64 agrees
    with; that is the routine firing that gets a warning suppressed.
    The cost is the 1% of precision-limited verdicts whose smallest
    eigenvalue came back non-positive, which stay silent.
    """
    ev = np.asarray(eigvals, dtype=np.float64)
    hi = float(ev[-1]) if ev.size else 0.0
    if not np.isfinite(hi) or hi <= 0.0 or rank_rtol <= 0.0:
        # No cutoff (``rank_rtol=0`` resolves everything by request) and
        # no usable scale (a non-positive or non-finite largest
        # eigenvalue) both mean there is no threshold to sit near.
        return None
    ratios = ev / hi
    pos = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    if pos.size == 0:
        return None
    deciding = float(pos[np.argmin(np.abs(np.log(pos / rank_rtol)))])
    if (rank_rtol / factor <= deciding <= rank_rtol * factor
            and deciding <= factor * eps_floor):
        return deciding, float(rank_rtol)
    return None


def _rank_and_crb(eigvals, eigvecs, rank_rtol: Optional[float], *,
                  n_residual: Optional[int] = None):
    """``(rank, crb)`` from the eigendecomposition of a Fisher matrix.

    Parameters
    ----------
    eigvals, eigvecs : array
        Ascending eigenvalues and their orthonormal columns, as
        ``jnp.linalg.eigh`` returns them for the symmetric PSD ``F``.
    rank_rtol : float, optional
        Eigenvalues at or below ``rank_rtol * eigvals[-1]`` count as
        zero.  ``None`` uses ``max(n, sqrt(m)) * eps`` at the matrix's
        own precision -- see :func:`_resolve_rank_rtol`.
    n_residual : int, optional
        Number of residual rows ``m`` that ``F = J.T @ J`` was summed
        over, when it is known.  ``None`` -- the default, for a caller
        holding only a decomposition -- drops the ``sqrt(m)`` term and
        so understates the cutoff for a long residual.  :func:`fim`
        always passes it.

    Notes
    -----
    Two thresholds are involved and they measure different things.

    The first is on the eigenvalues and decides which *directions* the
    data resolves.  It has to be relative to the largest eigenvalue:
    ``eigh`` returns each eigenvalue with an absolute error of order
    ``eps * ||F||``, so anything below that is noise whatever units the
    residual carries -- it can even come back negative, as the spring's
    ``(stiffness, damping, mass)`` scale direction does.  ``n * eps`` is
    the form ``numpy.linalg.matrix_rank`` uses (``max(shape) * eps``,
    and ``F`` is square); ``sqrt(m) * eps`` is the part of the floor
    ``F`` itself cannot show, contributed by summing ``m`` residual rows
    into each entry, and :func:`_resolve_rank_rtol` explains why it is
    ``sqrt(m)`` and not ``m``.  The cutoff sits at ``eps`` rather than
    ``sqrt(eps)`` because ``F = JᵀJ`` has already squared the
    conditioning of ``J``: a direction below ``sqrt(eps)`` in ``J`` is
    below ``eps`` here, and forming ``F`` is what lost it.  Being
    relative is also what keeps ``rank`` steady where ``cond`` is not --
    dividing the residual by a large ``noise_std`` can round the
    smallest eigenvalue of a float32 matrix to exactly zero and send
    ``cond`` to ``inf``, but it moves the eigenvalue and the threshold
    together.

    The second is on the null-space projector ``P = V₀ V₀ᵀ`` and decides
    which *parameters* those unresolved directions spoil.  The bound on
    parameter ``i`` is infinite as soon as ``eᵢ`` has any component
    outside the range of ``F``, so the test is ``diag(P)ᵢ > 0`` rather
    than "is ``eᵢ`` the weakest eigenvector": a parameter spread over
    several near-null directions has a small component in each and would
    survive a per-eigenvector test while being wholly unidentifiable.
    The projector is also the better conditioned object -- whenever two
    independent combinations are invisible the null eigenvalues are
    degenerate, which leaves the individual null eigenvectors arbitrary
    within the subspace but not their projector.  ``diag(P)`` is a sum
    of squared direction cosines, hence dimensionless and in ``[0, 1]``;
    a parameter genuinely orthogonal to the null space measures 0 there,
    so ``n * eps`` clears the floor by orders of magnitude and still
    catches a component of relative length ``sqrt(n * eps)`` (5e-4 of
    the direction, in float32).
    """
    # ``jnp.result_type`` rather than ``jnp.asarray(...).dtype``: the
    # latter ships the array to a device to read a dtype the host
    # already knows, and ``fim`` now hands this function host arrays.
    dtype = jnp.result_type(eigvals)
    ev = np.asarray(eigvals, dtype=np.float64)
    vecs = np.asarray(eigvecs, dtype=np.float64)
    n = int(ev.size)
    eps = float(np.finfo(dtype).eps)
    rank_rtol = _resolve_rank_rtol(dtype, n, rank_rtol, n_residual=n_residual)
    resolved = ev > max(float(ev[-1]), 0.0) * rank_rtol
    rank = int(resolved.sum())
    # The inverse over the resolved subspace, diag(V Λ⁻¹ Vᵀ) with the
    # null directions dropped.  Built from the eigendecomposition
    # already in hand rather than from ``pinv`` so that ``crb`` and
    # ``rank`` agree on what is singular by construction: ``pinv``
    # applies a cutoff of its own, which need not be this one.
    crb = ((vecs[:, resolved] ** 2) / ev[resolved]).sum(axis=1)
    support = (vecs[:, ~resolved] ** 2).sum(axis=1)
    # Fail safe by *establishing* finiteness rather than by excluding it.
    # ``np.where(support > n * eps, inf, crb)`` reads the right way round
    # and is the wrong way round: when ``F`` is NaN every eigenvalue is
    # NaN, ``resolved`` is all-False, ``crb`` is an empty sum -- ``0.0``,
    # the most trustworthy value this report can carry -- and ``support``
    # is NaN, so ``NaN > n * eps`` is False and the rescue never fires.
    # A caller's ``crb < tol`` then answers "identified" for a matrix
    # holding no information at all. So a bound stays finite only where
    # both the support test and the bound itself came out finite.
    determined = np.isfinite(support) & (support <= n * eps) & np.isfinite(crb)
    crb = np.where(determined, crb, np.inf)
    return rank, jnp.asarray(crb, dtype=dtype)


# ---------------------------------------------------------------------------
# The jitted core: everything the report needs, as device arrays
# ---------------------------------------------------------------------------


@stability(StabilityLevel.EXPERIMENTAL)
@dataclass(frozen=True, kw_only=True)
class FIMCore:
    """:func:`fim_core`'s output: the same quantities as :class:`FIMReport`,
    as **device arrays**, from a function that can be ``jax.jit``-ed.

    Registered as a pytree, so it can be returned from a jitted function
    and carried through :func:`jax.lax.scan`.  ``param_names``,
    ``n_residual`` and ``rank_rtol`` are static metadata; every other
    field is a traced array.

    Keyword-only for the reason :class:`FIMReport` is: a field inserted
    in the middle would silently re-map every positional argument after
    it, and a mis-assigned ``rank`` or ``crb`` is a wrong
    identifiability verdict that raises nothing.

    The fields answer the same questions :class:`FIMReport` answers and
    by the same rules -- see it for what ``rank``, ``crb``,
    ``zero_scaled`` and the precision band *mean*; only the types and
    two deliberate differences are described here.

    **Everything is a device array, including the verdicts.**  ``rank``
    is a 0-d ``int32``, ``cond`` a 0-d float, ``finite`` and
    ``precision_limited`` 0-d bools, ``zero_scaled`` a boolean mask over
    ``param_names`` rather than a tuple of names.  Nothing here has been
    read back to the host, which is the whole point: a control loop can
    branch on ``crb`` with :func:`jax.lax.cond` or fold it into a
    :func:`jax.lax.scan` carry without ever stalling on a transfer.
    The host-side spellings -- the ``FloatingPointError`` on a
    non-finite ``F``, the ``PrecisionLimitWarning``, the names in
    ``zero_scaled`` -- all live in :func:`fim`, which is the reporting
    path and is unchanged.

    **The arithmetic runs at the matrix's own precision.**
    :func:`fim`'s ``rank``/``crb``/precision-band verdicts widen the
    eigendecomposition to float64 on the host first; these are computed
    in float32 (or whatever ``F`` is) on the device, because widening
    would mean a transfer and JAX has no float64 without the global
    ``jax_enable_x64``.  The rule applied is identical and the two agree
    on ``rank`` and on which ``crb`` entries are ``+inf`` over the
    tested spread; the finite ``crb`` values differ in the last bits,
    and a verdict close enough to the cutoff for the two to disagree is
    exactly the one ``precision_limited`` is there to flag.

    ``crb`` keeps :func:`fim`'s fail-closed polarity: ``+inf`` unless
    finiteness was *positively* established, so ``crb < tol`` reads
    False for an unidentifiable parameter and for a ``NaN`` matrix
    rather than propagating a ``NaN`` into a gate.

    Attributes
    ----------
    fim : jnp.ndarray
        ``F = Jᵀ Σ⁻¹ J``, shape ``(n, n)``.
    eigvals, eigvecs : jnp.ndarray
        Ascending eigenvalues and their orthonormal columns.
    rank : jnp.ndarray
        0-d ``int32``: directions resolved above ``rank_rtol``.
    cond : jnp.ndarray
        0-d float: ``eigvals[-1] / eigvals[0]``, ``+inf`` when the
        smallest eigenvalue is not positive -- including when it is
        ``NaN``, where :attr:`FIMReport.cond` reports ``NaN``.  ``inf``
        is the fail-closed reading and this is the gating path; the
        reporting path keeps the number it has always published.
    crb : jnp.ndarray
        Cramér–Rao bound per parameter, ``(n,)``.
    finite : jnp.ndarray
        0-d bool: ``all(isfinite(F))``.  False is what makes
        :func:`fim` raise; a loop should gate on it rather than trust
        the rest of the record.
    zero_scaled : jnp.ndarray
        0-d-per-parameter bool mask, ``(n,)``: the parameters
        ``scale="relative"`` found at exactly ``0.0``.  All False under
        ``scale=None``.
    precision_limited : jnp.ndarray
        0-d bool: the rank verdict rests on a difference this precision
        cannot resolve -- the condition :func:`fim` turns into a
        :class:`~maddening.warnings.PrecisionLimitWarning`.  A live gate
        should read it as a third outcome, "verdict unavailable",
        rather than as a refusal.
    deciding_ratio : jnp.ndarray
        0-d float: the eigenvalue ratio nearest the cutoff in log
        distance -- the number :func:`fim`'s warning quotes -- or
        ``0.0`` when ``precision_limited`` is False.  Zero and not
        ``NaN``: a reported ratio is always strictly positive, so zero
        is unambiguous, and nothing in the core produces a ``NaN`` that
        would stop a ``jax_debug_nans`` run on a value it discarded.
    param_names : tuple of str
        Static.  Names in the order the matrix is indexed.
    n_residual : int
        Static.  Rows of ``J``, i.e. the flattened residual length.
    rank_rtol : float
        Static.  The cutoff actually in force, already resolved through
        :func:`_resolve_rank_rtol`.
    value_scaled : tuple of str
        Static.  :attr:`FIMReport.value_scaled`: the columns
        ``scale="nominal"`` scaled by the value because their spec has
        no finite width.  Decided from ``specs`` on the host at trace
        time, so it is metadata and not a mask -- nothing about it is
        read back.  Empty under every other scale.
    integer_excluded : tuple of str
        Static.  :attr:`FIMReport.integer_excluded`: the integer and
        boolean leaves left out of the matrix, decided from dtypes on the
        host at trace time.
    """
    fim: jnp.ndarray
    eigvals: jnp.ndarray
    eigvecs: jnp.ndarray
    rank: jnp.ndarray
    cond: jnp.ndarray
    crb: jnp.ndarray
    finite: jnp.ndarray
    zero_scaled: jnp.ndarray
    precision_limited: jnp.ndarray
    deciding_ratio: jnp.ndarray
    param_names: tuple[str, ...]
    n_residual: int
    rank_rtol: float
    value_scaled: tuple[str, ...] = ()
    integer_excluded: tuple[str, ...] = ()


jax.tree_util.register_dataclass(
    FIMCore,
    data_fields=["fim", "eigvals", "eigvecs", "rank", "cond", "crb",
                 "finite", "zero_scaled", "precision_limited",
                 "deciding_ratio"],
    meta_fields=["param_names", "n_residual", "rank_rtol", "value_scaled",
                 "integer_excluded"],
)


def _device_rank_crb(eigvals, eigvecs, rank_rtol: float, n: int):
    """``(rank, crb)`` on the device, by :func:`_rank_and_crb`'s rule.

    The same two thresholds, the same fail-closed construction of
    ``crb``; see :func:`_rank_and_crb` for why each is what it is.  Two
    mechanical differences, both forced by staying on the device:

    * **Precision.**  :func:`_rank_and_crb` widens to float64 before it
      divides and sums.  There is no float64 here without the global
      ``jax_enable_x64``, so this runs at ``eigvals``'s own precision.
    * **No boolean indexing.**  ``vecs[:, resolved]`` needs a shape
      known at trace time and ``resolved`` is traced, so the sums are
      written as masked sums over all ``n`` columns.

    The unresolved columns are divided by a substituted ``1.0`` rather
    than by their own eigenvalue, which is the standard JAX
    double-``where``.  It does not change a single returned value --
    ``jnp.where`` *selects*, it does not multiply, so an ``inf`` in the
    rejected branch is discarded rather than turned into ``0 * inf`` --
    and it is not defensive padding either: an unresolved eigenvalue is
    routinely exactly ``0.0`` or negative (the spring's ``(k, c, m)``
    scale direction, any ``zero_scaled`` parameter), so without it the
    division really does produce ``inf``/``NaN`` on every rank-deficient
    problem.  That trips ``jax_debug_nans`` for a user who has turned it
    on to find a real ``NaN``, and it is the term ``jax.grad`` of this
    would carry.  ``TestCoreFailsClosed`` runs the whole core under
    ``jax.debug_nans`` for exactly that reason.
    """
    eps = float(np.finfo(eigvals.dtype).eps)
    resolved = eigvals > jnp.maximum(eigvals[-1], 0.0) * rank_rtol
    rank = jnp.sum(resolved.astype(jnp.int32))
    sq = eigvecs ** 2
    safe = jnp.where(resolved, eigvals, jnp.ones_like(eigvals))
    crb = jnp.sum(jnp.where(resolved[None, :], sq / safe, 0.0), axis=1)
    support = jnp.sum(jnp.where(resolved[None, :], 0.0, sq), axis=1)
    determined = (jnp.isfinite(support) & (support <= n * eps)
                  & jnp.isfinite(crb))
    return rank, jnp.where(determined, crb, jnp.inf)


def _device_precision_limited(eigvals, rank_rtol: float, eps_floor: float,
                              factor: float = _PRECISION_WARN_FACTOR):
    """``(limited, deciding ratio)`` on the device, by
    :func:`_precision_limited`'s rule.

    Both of its conditions and both of its refusals survive: the ratio
    must be within ``factor`` of the cutoff *and* at the precision
    floor, a non-positive or non-finite largest eigenvalue means there
    is no scale to compare against, and a non-positive ratio is not
    reported (an exactly rank-deficient ``F`` is a real deficiency, not
    a precision-limited verdict).  ``rank_rtol <= 0`` is a static
    Python value, so that branch is taken at trace time and no
    comparison is emitted at all.

    ``argmin`` over the log distance replaces the host version's
    ``argmin`` over a filtered array: the non-positive ratios are
    pushed to ``+inf`` distance rather than removed, which selects the
    same entry.

    **Nothing here produces a ``NaN``, deliberately.**  Every divisor
    and every ``log`` argument is masked to a safe value first, and the
    "no verdict" answer is the ratio ``0.0`` rather than ``NaN`` -- a
    reported ratio is a ratio of a positive eigenvalue to the largest,
    so zero is unambiguous.  A ``NaN`` computed and then discarded stops
    a user who has switched on ``jax_debug_nans`` to find their own, on
    a value this function never used; and a control loop that logs
    ``deciding_ratio`` every tick should not be logging ``NaN`` as its
    normal output.
    """
    hi = eigvals[-1]
    zero = jnp.zeros((), dtype=eigvals.dtype)
    if rank_rtol <= 0.0:
        return jnp.zeros((), dtype=bool), zero
    usable = jnp.isfinite(hi) & (hi > 0.0)
    ratios = eigvals / jnp.where(usable, hi, jnp.ones_like(hi))
    pos = jnp.isfinite(ratios) & (ratios > 0.0)
    distance = jnp.where(
        pos, jnp.abs(jnp.log(jnp.where(pos, ratios, jnp.ones_like(ratios)))
                     - math.log(rank_rtol)),
        jnp.inf)
    deciding = ratios[jnp.argmin(distance)]
    limited = (usable & jnp.any(pos)
               & (rank_rtol / factor <= deciding)
               & (deciding <= rank_rtol * factor)
               & (deciding <= factor * eps_floor))
    return limited, jnp.where(limited, deciding, zero)


_SCALES = ("relative", "nominal", None)


def _nominal_entry(spec) -> tuple[Optional[float], float]:
    """``(width, offset)`` -- the column record ``scale="nominal"`` gives a
    leaf governed by ``spec``: the width ``hi - lo`` of two finite bounds,
    else no width and the offset the value is measured from (a ``"log"``
    spec's lower bound, the transform's own gain ``dp/du = p - lo``; ``0``
    otherwise).  The policy table in :func:`fim`, as code, in one place.
    """
    # An infinite bound is accepted by ``ParamSpec`` only where it means
    # "unbounded" and is read that way here: a width needs two *finite*
    # bounds, as the table in :func:`fim` says.  (A ``"log"`` spec cannot
    # carry an infinite lower bound at all.)
    lo, hi = spec.bounds
    lo = None if lo is None or math.isinf(lo) else float(lo)
    hi = None if hi is None or math.isinf(hi) else float(hi)
    if lo is not None and hi is not None:
        return hi - lo, 0.0
    if spec.transform == "log":
        return None, 0.0 if lo is None else lo
    return None, 0.0


def _changes_no_column(spec) -> bool:
    """Whether ``spec`` gives a column exactly the record the default spec
    gives it -- so that a ``specs`` entry holding it cannot have changed a
    nominal report, whichever leaf it was meant for.

    This is what lets a key that matches no parameter through
    :func:`_resolve_nominal` when, and only when, it is harmless:
    ``gm.param_specs()`` declares a spec for every constant a node knows,
    including those ``params_pytree()`` leaves out (a uniform
    ``HeatNode``'s ``grid_points=None``, a constant given as a Python
    ``int``), and refusing those refused the documented
    ``specs=gm.param_specs()`` for every such graph.  A misspelt key that
    carries a width or a ``"log"`` offset is still refused, because
    there the report would differ.
    """
    return _nominal_entry(spec) == _nominal_entry(DEFAULT_SPEC)


def _resolve_nominal(params, specs, idx, scale):
    """The static per-column record ``scale="nominal"`` multiplies by,
    or ``None`` under any other scale.

    One entry per column of the (masked) Jacobian, ``(name, width,
    offset)``: ``width`` is ``hi - lo`` for a spec with two finite
    bounds and ``None`` where there is no finite width, in which case
    the column is scaled by ``theta - offset`` -- ``offset`` being the
    lower bound of a ``transform="log"`` spec (the transform's own gain
    ``dp/du = p - lo``) and ``0.0`` otherwise, i.e. plain relative
    scaling.  The policy :func:`fim` tabulates lives in
    :func:`_nominal_entry` and :func:`_nominal_column_vector` and
    nowhere else.

    Everything here is host Python over ``specs`` and the *structure*
    of ``params``; no leaf value is read, so it costs no transfer and
    the result is a hashable tuple -- which is what lets it key
    :func:`_fim_jacobian_compiled` and be closed over by a trace, just
    as ``mask`` is reduced to ``idx``.

    A ``specs`` given under another scale is refused rather than
    ignored: a spec that changes nothing should not be accepted as if
    it had.  ``specs=None`` under ``"nominal"`` is refused too -- an
    empty dict is the honest way to say "no spec has a width", and it
    produces a report that names every column in ``value_scaled``.
    The same principle refuses a ``specs`` that does not mirror
    ``params`` (:func:`~maddening.core.params._validate_specs_mirror`):
    a mis-nested tree, a ``ParamSpec.to_dict()`` entry, a misspelt key
    or a non-dict ``specs`` would otherwise yield a report bit-identical
    to ``scale="relative"`` and indistinguishable from the honest
    ``specs={}``.  One exemption, by :func:`_changes_no_column`: a key
    that matches no parameter is accepted when its spec would give any
    column the default record, so ``gm.param_specs()`` -- which declares
    specs for constants ``params_pytree()`` leaves out -- is accepted
    for the graph it came from while a misspelt key carrying a width is
    not.
    """
    if scale != "nominal":
        if specs is not None:
            raise ValueError(
                f"specs is read only under scale='nominal'; got specs with "
                f"scale={scale!r}. Drop it, or ask scale='nominal' if the "
                f"bounds-derived scaling is what you want.")
        return None
    if specs is None:
        raise ValueError(
            "scale='nominal' derives each column's scale from a ParamSpec "
            "and needs specs= -- gm.param_specs() for a graph's parameter "
            "tree, or a dict of ParamSpec mirroring params. Pass {} to say "
            "explicitly that no leaf has a declared width; every column is "
            "then value-scaled and FIMReport.value_scaled names them all.")
    leaf_specs = _validate_specs_mirror(
        params, specs, unmatched_ok=_changes_no_column)
    # Columns exist for the differentiable leaves only (:func:`_fim_indices`),
    # so the record is built over exactly those, in the same order.
    names = _param_names(params, differentiable_only=True)
    kept = [(spec, leaf) for spec, leaf in zip(leaf_specs, jax.tree.leaves(params))
            if _is_differentiable(leaf)]
    cols = [_nominal_entry(spec) for spec, leaf in kept
            for _ in range(_leaf_size(leaf))]
    if idx is not None:
        cols = [cols[int(i)] for i in idx]
        names = tuple(names[int(i)] for i in idx)
    return tuple((nm, w, off) for nm, (w, off) in zip(names, cols))


def _nominal_column_vector(nominal, theta0):
    """The per-column multiplier for ``scale="nominal"``, as an array
    shaped like ``theta0`` -- constants where the spec has a width,
    ``theta0 - offset`` where it has not.

    The width guard is here because this is where the dtype is known.
    ``ParamSpec`` already refuses ``lo >= hi``, so a width is positive
    in float64 -- but it is applied at ``theta0``'s precision and
    squared on the way into ``F = J.T @ J``, and a width of ``1e-300``
    or a pair of bounds like ``(0.0, 1e39)`` is a zero or an ``inf``
    column by the time it gets there.  A zero column is the defect this
    scale exists to remove, so a width that does not survive both is
    refused by name rather than let through.  Trace time, host side:
    ``nominal`` is static, so no value of ``theta0`` is consulted.
    """
    dtype = theta0.dtype
    widths = np.ones(len(nominal), dtype=dtype)
    offsets = np.zeros(len(nominal), dtype=dtype)
    has_width = np.zeros(len(nominal), dtype=bool)
    for j, (name, width, offset) in enumerate(nominal):
        if width is None:
            offsets[j] = offset
            continue
        # The overflow and underflow are the thing being tested for, so
        # numpy's warnings about them are the guard working, not a
        # finding: silence them here and read the result.
        with np.errstate(all="ignore"):
            w = np.asarray(width, dtype=dtype)
            sq = w * w
        if not (np.isfinite(w) and w > 0 and np.isfinite(sq) and sq > 0):
            raise ValueError(
                f"scale='nominal': the width of {name}'s bounds is {width!r}, "
                f"which is not a positive finite number at {np.dtype(dtype)} "
                f"once squared into F = J^T J ({w!r} -> {sq!r}). A column "
                f"scaled by it would be zero or non-finite and the parameter "
                f"would read as unidentifiable whatever the data say -- the "
                f"failure this scale exists to remove -- so it is refused "
                f"rather than reported. Widen or narrow the bounds to a "
                f"width this precision can carry, or re-run under x64.")
        widths[j] = w
        has_width[j] = True
    if bool(has_width.all()):
        return jnp.asarray(widths)
    value_scaled = theta0 - jnp.asarray(offsets)
    if not bool(has_width.any()):
        return value_scaled
    return jnp.where(jnp.asarray(has_width), jnp.asarray(widths), value_scaled)


def _fim_jacobian(residual_fn, params, *, scale, idx, inv_sigma,
                  nominal=None):
    """``(J, zero-scaled mask)`` -- the expensive half, and the only half.

    Composition, row then column: ``_r`` divides residual row ``i`` by
    ``sigma_i`` *before* differentiation, so ``J[i, j] = (1 / sigma_i)
    * d r_i / d theta_j``; the scale multiplies column ``j`` afterwards,
    ``J[i, j] * s_j``, with ``s_j = theta_j`` under ``"relative"``, the
    :func:`_resolve_nominal` record under ``"nominal"`` and ``1`` under
    ``None``.  ``idx`` has already selected the columns, so every
    ``s_j`` is the scale of the parameter that column belongs to.

    ``jacfwd`` over an N-step rollout is what made :func:`fim`
    unusable in a loop: in eager mode every operation of the
    ``lax.scan`` is dispatched separately, measured at 61-69 ms for a
    200-step spring-damper rollout against 14-64 **microseconds** for
    the same Jacobian under ``jax.jit``.  Everything downstream of
    ``J`` is ``O(n*m)`` and ``O(n**3)`` on a handful of parameters and
    costs well under a millisecond however it is scheduled.

    Split out so that :func:`fim` can jit *this* -- under
    ``reuse_trace=True``, the only case where a compiled Jacobian is kept
    between calls and so the only case where compiling it pays -- and
    leave the rest eager.  That is not squeamishness: ``J`` is bit-for-bit the same
    jitted or not, but ``F = J.T @ J`` is not -- ``jax.jit`` folds the
    transpose into the dot's dimension numbers instead of materialising
    it, which accumulates in a different order and moves ``F`` by up to
    a float32 ulp.  That is inside the ``max(n, sqrt(m)) * eps`` floor
    this module declares its answers are good to, and it still moves
    published verdicts, because ``rank`` and ``cond`` are decided by
    comparisons *at* that floor: measured over this module's own test
    problems, the spring's ``(k, c, m)`` scale direction -- whose
    smallest eigenvalue is a rounding artefact either side of zero --
    moved ``cond`` from ``inf`` to ``3.45e+07``, and a Fisher matrix
    built with its smallest eigenvalue ratio *on* the cutoff moved
    ``rank`` from 3 to 2.  :func:`fim` is the reporting path and its
    numbers do not move; :func:`fim_core` jits the lot and says so.
    """
    leaves, treedef = jax.tree.flatten(params)
    keep = [_is_differentiable(leaf) for leaf in leaves]
    if all(keep):
        flat, unravel = ravel_pytree(params)
    else:
        # Integer and boolean leaves are not columns (:func:`_fim_indices`)
        # and do not go into the vector at all: promoted to float and cast
        # back they would have a zero derivative, and an integer above
        # 2**24 would not even survive the round trip in float32.  They
        # reach ``residual_fn`` as the objects they are.
        flat, unravel_kept = ravel_pytree(
            [leaf for leaf, k in zip(leaves, keep) if k])

        def _unravel_mixed(vec):
            it = iter(unravel_kept(vec))
            return jax.tree.unflatten(
                treedef, [next(it) if k else leaf
                          for leaf, k in zip(leaves, keep)])

        unravel = _unravel_mixed

    def _r(theta):
        full = theta if idx is None else flat.at[idx].set(theta)
        r = ravel_pytree(residual_fn(unravel(full)))[0]
        return r if inv_sigma is None else r * inv_sigma

    theta0 = flat if idx is None else flat[idx]
    J = jax.jacfwd(_r)(theta0)
    if scale == "relative":
        # A column scaled by a parameter sitting at exactly zero is zero,
        # so the parameter drops out of F however well the data determine
        # it.  Report that as what it is rather than as a verdict on the
        # data: SpringDamperNode's initial_velocity defaults to 0.0, so
        # this is the first thing a user meets on the default scale.
        zero_scaled = theta0 == 0.0
        J = J * theta0[None, :]
    elif scale == "nominal":
        col = _nominal_column_vector(nominal, theta0)
        # Tested on the multiplier itself and not on ``has_width``: a
        # width column cannot be zero past the guard, but if it ever
        # were, this is the field that has to say so.
        zero_scaled = col == 0.0
        J = J * col[None, :]
    else:
        zero_scaled = jnp.zeros(theta0.shape, dtype=bool)
    return J, zero_scaled


def _fim_spectrum(J):
    """``(F, all-finite, eigvals, eigvecs)`` from the scaled Jacobian.

    One definition, run two ways: eagerly by :func:`fim`, where it is
    the same three operations in the same order that function has
    always performed, and inside the trace by :func:`fim_core`.
    ``eigh`` is evaluated before the finiteness of ``F`` is known
    because a traced computation cannot branch on it; on a non-finite
    ``F`` it returns ``NaN`` eigenvalues and eigenvectors rather than
    raising, which is what :func:`fim` then refuses to build a report
    from.
    """
    F = J.T @ J
    eigvals, eigvecs = jnp.linalg.eigh(F)
    return F, jnp.all(jnp.isfinite(F)), eigvals, eigvecs


def _value_scaled_names(nominal) -> tuple[str, ...]:
    """The columns of a :func:`_resolve_nominal` record with no width."""
    if nominal is None:
        return ()
    return tuple(nm for nm, width, _ in nominal if width is None)


def _fim_core_traced(residual_fn, params, *, scale, idx, inv_sigma,
                     rank_rtol, nominal=None,
                     integer_excluded: tuple[str, ...] = ()) -> FIMCore:
    """The whole of :func:`fim`'s computation, with nothing read back."""
    J, zero_scaled = _fim_jacobian(residual_fn, params, scale=scale, idx=idx,
                                   inv_sigma=inv_sigma, nominal=nominal)
    F, finite, eigvals, eigvecs = _fim_spectrum(J)
    lo, hi = eigvals[0], eigvals[-1]
    # ``lo > 0`` rather than ``not (lo <= 0)``, so a ``NaN`` smallest
    # eigenvalue reads ``inf`` -- singular -- and not ``NaN``, and the
    # division never sees a zero.  :func:`fim` computes ``cond`` on the
    # host from float64 copies and keeps the ``NaN`` there, because that
    # is the number it has always published.
    positive = lo > 0.0
    cond = jnp.where(positive, hi / jnp.where(positive, lo, jnp.ones_like(lo)),
                     jnp.inf)
    names = _param_names(params, differentiable_only=True)
    if idx is not None:
        names = tuple(names[i] for i in idx)
    # The residual length ``J`` was summed over.  ``_r`` ravels its output,
    # so this is the flattened residual row count whatever pytree shape
    # ``residual_fn`` returns, and it is the only place ``m`` is visible:
    # ``F`` and its decomposition have already discarded it.
    n_residual = int(J.shape[0])
    n_params = int(eigvals.shape[0])
    rtol = _resolve_rank_rtol(eigvals.dtype, n_params, rank_rtol,
                              n_residual=n_residual)
    # The precision floor the band is anchored to is the *default*
    # cutoff, which is what "the resolution of this arithmetic" means.
    # It has to move with the cutoff: anchoring to the n-only form
    # would re-impose the blind spot the sqrt(m) term removes, on the
    # warning if no longer on the rank.
    floor = _resolve_rank_rtol(eigvals.dtype, n_params, None,
                               n_residual=n_residual)
    rank, crb = _device_rank_crb(eigvals, eigvecs, rtol, n_params)
    limited, ratio = _device_precision_limited(eigvals, rtol, floor)
    return FIMCore(
        fim=F, eigvals=eigvals, eigvecs=eigvecs, rank=rank, cond=cond,
        crb=crb, finite=finite, zero_scaled=zero_scaled,
        precision_limited=limited, deciding_ratio=ratio,
        param_names=names, n_residual=n_residual, rank_rtol=rtol,
        value_scaled=_value_scaled_names(nominal),
        integer_excluded=integer_excluded,
    )


#: Distinct ``(residual_fn, scale, mask, nominal)`` signatures whose traced
#: Jacobian :func:`fim` keeps compiled under ``reuse_trace=True``.
#: Bounded because each entry holds a strong reference to the caller's
#: ``residual_fn`` and to everything it closes over -- a graph, a window
#: of observations -- and an unbounded cache keyed on user callables is a
#: leak.  ``jax.jit`` caches on the same principle.  Nothing is stored
#: here by a default call.
_FIM_JACOBIAN_CACHE_SIZE = 32


@functools.lru_cache(maxsize=_FIM_JACOBIAN_CACHE_SIZE)
def _fim_jacobian_compiled(residual_fn, scale, idx_key, nominal=None):
    """The jitted :func:`_fim_jacobian` for one static signature --
    consulted only by ``fim(..., reuse_trace=True)``.

    ``nominal`` is the :func:`_resolve_nominal` record -- a hashable
    tuple of Python floats and names, or ``None`` -- and is part of the
    key because two ``specs`` with different widths compile to
    different constants in the same shape.

    **What the key cannot see is why this is opt-in.**  Tracing reads
    every value ``residual_fn`` takes from anywhere but its argument --
    the ``params`` of a :class:`GraphManager` it closes over, an
    attribute of the object it is a bound method of, a module global --
    and bakes it into the program as a constant.  The key is
    ``residual_fn`` itself (by equality: a bound method compares equal
    across attribute accesses), so a later call after that state changed
    hits this entry and gets the *first* call's Fisher matrix, rank and
    bounds, with no warning.  Measured before this was made opt-in: a
    spring's stiffness bound reported as 0.41 against a true 44.7 after
    its damping moved from 2 to 20 in ``gm.params``, and rank 2 against a
    true rank 1 after a bound method's excitation changed.  No key can
    be derived from the callable that changes when the state it reads
    does, so the default re-traces and only a caller who asserts the
    residual is pure (``reuse_trace=True``) gets this entry.

    What reuse buys, for that caller: tracing and compiling the rollout
    costs about what eager ``jacfwd`` costs (the scan is compiled either
    way), so the win is entirely in *not paying it again* -- a 200-step
    spring-damper rollout measured ~230 ms per default call against ~1 ms
    warm here, on four pinned cores.  A fresh closure per call misses the
    cache and gains nothing.

    ``inv_sigma`` is an *argument* of the returned function rather than
    part of the key, so a noise model does not have to be hashable to
    be cached and changing sigma between calls does not recompile: only
    its shape and dtype are baked in, by ``jax.jit`` itself.
    """
    idx = None if idx_key is None else np.asarray(idx_key)

    @jax.jit
    def run(params, inv_sigma):
        return _fim_jacobian(residual_fn, params, scale=scale, idx=idx,
                             inv_sigma=inv_sigma, nominal=nominal)

    return run


def _resolved_noise(residual_fn, params, noise_std, *, reuse_trace=False):
    """``1 / sigma``, or ``None``, without evaluating the residual.

    ``jax.eval_shape`` traces ``residual_fn`` for its output structure
    and dtypes and runs none of it.  ``fim`` used to call
    ``residual_fn(params)`` in full here -- a whole extra rollout per
    call -- and then hand the result to :func:`_inverse_noise_std`,
    which returns ``None`` on the first line when ``noise_std`` is
    ``None`` and never looks at it.  That is the common case and the
    rollout was pure waste; now nothing is traced at all unless a noise
    model was given.

    ``jax.eval_shape`` is itself cached by JAX on the function it is
    handed, which is the same trap :func:`_fim_jacobian_compiled` is
    opt-in for: a residual whose output *structure* depends on state it
    reads (a window length held on ``self``) would be given the first
    call's.  So unless the caller asserted purity with ``reuse_trace``,
    the residual goes in wrapped in a closure made for this call, which
    no cache entry can match.
    """
    if noise_std is None:
        return None
    shape_fn = residual_fn if reuse_trace else (lambda p: residual_fn(p))
    return _inverse_noise_std(noise_std, jax.eval_shape(shape_fn, params))


@stability(StabilityLevel.EXPERIMENTAL)
def fim_core(
    residual_fn: Callable[[dict], Any],
    params: dict,
    *,
    scale: Optional[str] = "relative",
    mask: Optional[dict] = None,
    noise_std: Optional[Any] = None,
    rank_rtol: Optional[float] = None,
    specs: Optional[dict] = None,
) -> FIMCore:
    """:func:`fim`'s computation with no host round trip -- jit this.

    Same arguments and same mathematics as :func:`fim`; see it for what
    every argument means.  What differs is that nothing is read back to
    the host: the result is a :class:`FIMCore` of device arrays, there
    is no ``FloatingPointError``, no ``PrecisionLimitWarning`` and no
    tuple of ``zero_scaled`` names, and the whole thing traces.  Use it
    where :func:`fim` is too slow to call -- inside a control loop, or
    under ``jax.lax.scan`` -- and :func:`fim` where a human reads the
    answer.

    ``residual_fn``, ``scale``, ``mask``, ``rank_rtol`` and ``specs``
    are static: close over them rather than passing them as traced
    arguments (``specs`` is reduced to a tuple of per-column constants
    on the host before anything is traced, exactly as ``mask`` is
    reduced to indices, so ``scale="nominal"`` adds no traced operand
    and its ``value_scaled`` verdict is metadata, not a mask)::

        core_fn = jax.jit(functools.partial(fim_core, residual_fn))
        core = core_fn(params)              # device arrays, no sync
        ok = core.finite & (core.crb[0] < tol) & ~core.precision_limited

    **Static means frozen at trace time, including what ``residual_fn``
    reads.**  ``fim_core`` keeps no cache of its own -- called eagerly it
    re-traces every time -- but under your ``jax.jit`` the compiled
    ``core_fn`` holds every value ``residual_fn`` took from anywhere
    other than its argument (the ``params`` of a graph it closes over, an
    attribute of ``self``, a global) as the constant it was when
    ``core_fn`` was first traced.  That is ``jax.jit``'s contract for any
    function, stated here because the natural residual closes over a
    ``GraphManager`` whose ``params`` a calibration loop changes: pass
    the changing values in through ``params``, or build a new
    ``core_fn`` when they change.  :func:`fim` re-traces on every call by
    default for exactly this reason (see its ``reuse_trace``).

    ``params`` and ``noise_std`` may be traced, with one restriction:
    ``noise_std`` is validated on the host (a sigma that is zero,
    negative, non-finite or underflows the residual's dtype is refused,
    and a ``jax.jit`` trace cannot refuse anything), so inside ``jit``
    it has to be a value the trace already holds -- a constant closed
    over, not an argument of the jitted function.  Pass the reciprocal
    yourself if you need sigma to vary per call.

    Returns
    -------
    FIMCore
        ``fim``, ``eigvals``, ``eigvecs``, and the verdicts ``rank``,
        ``cond``, ``crb``, ``finite``, ``zero_scaled``,
        ``precision_limited`` and ``deciding_ratio``, all as device
        arrays.  ``crb`` keeps :func:`fim`'s fail-closed ``+inf``
        polarity, so ``crb < tol`` is False for an unidentifiable
        parameter and for a ``NaN`` matrix alike.

    Raises
    ------
    ValueError
        For a ``scale``, ``mask``, ``rank_rtol`` or ``noise_std`` this
        function cannot answer for -- at trace time, since that is when
        they are read.

    Notes
    -----
    A non-finite ``F`` is reported in ``finite`` rather than raised.
    That is the one deliberate behaviour difference from :func:`fim`
    and it is forced: a traced computation has no value to test.  It is
    also what a loop wants, which is to notice and hold last-known-good
    rather than to unwind.  **A caller that ignores ``finite`` gets a
    report built on ``NaN``**, where ``rank`` is 0 and every ``crb`` is
    ``+inf`` -- fail-closed, but silent.

    See Also
    --------
    fim : the same computation with the host-side verdict layer on top.
    """
    if scale not in _SCALES:
        raise ValueError(
            f"scale must be 'relative', 'nominal' or None, got {scale!r}")
    idx, integer_excluded = _fim_indices(params, mask)
    return _fim_core_traced(
        residual_fn, params, scale=scale, idx=idx,
        inv_sigma=_resolved_noise(residual_fn, params, noise_std),
        rank_rtol=rank_rtol,
        nominal=_resolve_nominal(params, specs, idx, scale),
        integer_excluded=integer_excluded,
    )


@stability(StabilityLevel.EVOLVING)
def fim(
    residual_fn: Callable[[dict], Any],
    params: dict,
    *,
    scale: Optional[str] = "relative",
    mask: Optional[dict] = None,
    noise_std: Optional[Any] = None,
    rank_rtol: Optional[float] = None,
    specs: Optional[dict] = None,
    reuse_trace: bool = False,
) -> FIMReport:
    """Fisher information matrix ``J^T J`` of ``residual_fn`` at ``params``.

    ``J = d residual / d params`` is computed with ``jax.jacfwd`` (one
    forward pass per parameter — cheap for the handful of parameters a
    physical model has, and it works through coupled steps because the
    IFT rule is a ``custom_jvp``).  This is the Gauss–Newton Hessian of
    ``0.5 * ||residual||^2``: it drops the residual-weighted second-order
    term, so unlike ``jax.hessian`` of the loss it is positive
    semi-definite and meaningful away from the optimum.

    Parameters
    ----------
    residual_fn : callable
        ``params -> residual`` (array or pytree of arrays), e.g. the
        simulated-minus-measured trajectory.
    params : dict
        Parameter pytree at which to linearise.
    scale : {"relative", "nominal", None}
        ``"relative"`` (default) multiplies each column of ``J`` by the
        parameter's value, i.e. sensitivities to *relative* changes, so
        parameters in different units are comparable and the condition
        number is not dominated by units.  ``None`` uses raw
        sensitivities.  A parameter whose value is exactly ``0.0`` has
        no relative scale: its column vanishes and it reads as
        unidentifiable however well the data determine it.  Those
        parameters are named in :attr:`FIMReport.zero_scaled`; ask
        ``scale=None`` for the absolute question about them, or
        ``scale="nominal"`` for the dimensionless one.

        ``"nominal"`` multiplies each column by a scale read from the
        parameter's :class:`~maddening.core.params.ParamSpec` (via
        ``specs``) instead of from its value: the **width** ``hi - lo``
        of a finite ``bounds``.  A width and not a midpoint, because a
        scale is what a column needs and the midpoint of ``(-1, 1)`` is
        the ``0.0`` this mode exists to escape; ``ParamSpec`` enforces
        ``lo < hi``, so a width is never zero, and one that would round
        to zero or overflow at the parameters' precision is refused by
        name.  Columns are still dimensionless, so ``cond`` still
        compares like with like, and the answer no longer depends on
        where in its range the parameter happens to sit.  A spec with
        no finite width has no nominal scale; that column keeps the
        value scaling ``"relative"`` would give it, and the parameter is
        named in :attr:`FIMReport.value_scaled` so the report says which
        of its columns were answered from the spec and which from the
        value.  The policy, per spec:

        ==================================  =============  ============  ================  =================
        ``bounds``                          ``transform``  column scale  ``value_scaled``  ``zero_scaled``
        ==================================  =============  ============  ================  =================
        ``(lo, hi)``, both finite           any            ``hi - lo``   no                never
        ``(lo, None)``, ``lo`` finite       ``"log"``      ``p - lo``    yes               at ``p == lo``
        ``(None, None)``                    ``"log"``      ``p``         yes               at ``p == 0``
        one or both ``None``                ``None``       ``p``         yes               at ``p == 0``
        ==================================  =============  ============  ================  =================

        The ``"log"`` rows use the transform's own gain ``dp/du =
        p - lo``, which is the scale an optimiser moving that parameter
        sees; it is still a value and can still be zero (at the
        boundary of the transform's domain), so those columns are
        value-scaled and named like the rest.  Nothing here falls back
        to ``1.0``: an absolute column among dimensionless ones would
        put units back into ``cond`` without saying so.  Order of
        composition: ``noise_std`` divides row ``i`` before
        differentiation, the scale multiplies column ``j`` after it,
        ``J[i, j] = (1 / sigma_i) * (d r_i / d theta_j) * s_j``, and
        ``mask`` selects the columns first, so each ``s_j`` is the scale
        of the parameter its column belongs to.
    mask : pytree of bool, optional
        Same structure as ``params``; only leaves marked ``True`` are
        treated as parameters (``GraphManager.trainable_mask()``).  The
        report's ``param_names`` / matrix are restricted accordingly.
        ``None`` means every *differentiable* leaf: an integer or boolean
        leaf has no derivative, so it is left out and named in
        :attr:`FIMReport.integer_excluded`, and a mask that selects one
        is a ``ValueError``.
        Unlike the fitters' ``mask`` this one is free to name a leaf the
        specs freeze: ``fim`` only linearises, it never steps a
        parameter, and the sensitivity of a frozen constant is a
        legitimate thing to ask for.
    noise_std : float or pytree, optional
        Measurement noise model.  A scalar σ (same for every residual
        entry) or a pytree matching ``residual_fn``'s output (per-leaf σ,
        scalar or broadcastable).  Each residual row is divided by its σ,
        so ``F = Jᵀ Σ⁻¹ J`` and ``crb`` is the Cramér–Rao bound in the
        parameters' own units (relative variance under
        ``scale="relative"``) rather than "per unit noise variance".
        Every σ must be finite and strictly positive at the residual's
        own precision, which is where a σ that underflows to ``0.0``
        and a negative σ are caught.
    specs : dict, optional
        Nested dict of :class:`~maddening.core.params.ParamSpec`
        mirroring ``params`` -- ``gm.param_specs()`` for a graph's tree,
        or ``{"k": ParamSpec(bounds=(0.0, 100.0))}`` for a flat one.
        Required under ``scale="nominal"`` and refused under any other
        scale, so a spec that is not being read is never silently
        carried.  A leaf without an entry gets the default (unbounded)
        spec and is therefore value-scaled and named; ``{}`` is the
        explicit way to say no leaf has a width.  The tree must mirror
        ``params``, though: a dict where a leaf needs a ``ParamSpec``
        (``to_dict()`` output), a ``ParamSpec`` above a dict or record
        level, a list of specs for a namedtuple or dataclass (address
        its fields by name, with a dict), a non-dict ``specs``, or a key
        that matches no parameter *and could have changed a column* (its
        spec has a finite width, or a ``"log"`` offset from a non-zero
        lower bound) is a ``ValueError`` naming the key path, because a
        spec that reaches nothing would otherwise produce ``"relative"``
        in disguise.  A stray key whose spec gives the default column
        record is accepted -- the report is identical either way -- so
        ``gm.param_specs()``, which declares specs for constants
        ``params_pytree()`` leaves out, is accepted for its own graph.  A
        list/tuple of leaves takes a list/tuple of specs by position or
        one ``ParamSpec`` covering every position.  Static: the column
        scales are derived from it on the host once and baked into the
        traced Jacobian as constants, as ``mask`` is reduced to indices.
    rank_rtol : float, optional
        Relative tolerance deciding which directions the data resolves:
        an eigenvalue at or below ``rank_rtol * max(eigvals)`` counts as
        zero, and the parameters with support in the directions it
        rejects get an infinite ``crb``.  The default is
        ``max(n, sqrt(m)) * eps`` for ``n`` parameters and ``m`` residual
        rows at the matrix's precision: the point below which this
        function is reporting its own rounding error, counting both
        stages that produce it -- ``eigh``'s ``n * eps`` (the relative
        form ``numpy.linalg.matrix_rank`` uses) and the ``sqrt(m) * eps``
        of forming ``F = J.T @ J`` over ``m`` rows.  Raise it to declare
        a merely ill-conditioned direction unidentifiable too.
        ``0.0`` resolves every positive eigenvalue and suppresses the
        precision warning with it: there is no threshold left to sit
        near.

        **Changed in 0.4.0**: the ``sqrt(m)`` term is new.  A cutoff of
        ``n * eps`` alone sits below the float32 noise floor whenever
        ``sqrt(m) > n``, and there ``rank`` could count a direction the
        data does not contain and ``crb`` report a finite bound for it.
        Verdicts move only for ``m > n**2``, and only ever toward
        "unresolved"; pass ``rank_rtol=n * eps`` explicitly to keep the
        old cutoff.
    reuse_trace : bool
        Reuse the Jacobian compiled by an earlier call with an equal
        ``residual_fn`` (same ``scale``, ``mask`` and ``specs`` record)
        instead of tracing it afresh.  **Off by default, and only correct
        for a pure residual.**  Tracing reads every value ``residual_fn``
        takes from anywhere but its argument -- the ``params`` of a
        ``GraphManager`` it closes over, an attribute of the object it is
        a bound method of, a module global -- and bakes it into the
        compiled program; a reused program keeps those values.  With
        ``reuse_trace=True``, a residual that reads the graph's
        ``params`` for the leaves it does not take as arguments reports
        the *first* call's matrix after those leaves change, silently,
        and a bound method (which compares equal across attribute
        accesses) does the same after its object changes.  So pass it
        only when the output depends on ``params`` and on nothing that
        can change between calls.

        What it buys is the trace and compile: a 200-step spring-damper
        rollout measured ~230 ms per default call against ~1 ms warm
        with reuse (four pinned CPU cores).  A fresh closure per call
        misses the cache and gains nothing; an unhashable
        ``residual_fn`` cannot key it and is traced afresh.  For a loop
        that needs no host-side report, :func:`fim_core` under your own
        ``jax.jit`` is the faster path and states the same contract.

    Returns
    -------
    FIMReport
        Notably ``rank``, the number of resolved directions, and
        ``crb``, which is ``+inf`` for a parameter the unresolved ones
        leave undetermined.  See :class:`FIMReport`.

    Warns
    -----
    ~maddening.warnings.PrecisionLimitWarning
        If the eigenvalue ratio deciding ``rank`` lands within
        ``_PRECISION_WARN_FACTOR`` of the cutoff, i.e. the verdict rests
        on a difference the matrix's own precision cannot resolve.  The
        message names the ratio, the cutoff and the remedy: re-run under
        ``jax_enable_x64``.  x64 is not the default and is not going to
        be -- it is process-global, set before the first JAX import, and
        fp64 measures 91x slower than fp32 on the reference RTX A2000
        (155 vs 14,135 GFLOP/s, the fp32 figure being TF32 tensor
        cores).  The design goal is that the cases which need it say so.

        Measured, not assumed: the factor covers the band where float32
        and float64 verdicts were observed to diverge over 228,000
        synthetic Fisher matrices of known rank, re-measured against the
        ``max(n, sqrt(m)) * eps`` cutoff this release introduced; see
        ``_PRECISION_WARN_FACTOR`` for the distribution and for the
        false-firing cost of every wider choice.  It is quiet on
        ordinary problems -- no fire at all above five times the cutoff
        -- and it does **not** fire on an exactly singular ``F``, whose
        null eigenvalue comes back at or below zero: that is a real rank
        deficiency, not a precision-limited verdict, and
        :attr:`FIMReport.zero_scaled` already names the common cause.
        Nor does it fire on a close call against a ``rank_rtol`` the
        caller raised, which is a modelling threshold and not a
        statement about arithmetic.

        Over that population it caught every precision-limited verdict,
        which it did not before 0.4.0: against the old ``n * eps``
        cutoff the same measurement put its recall at 0.912-0.916, and the
        misses were the long-residual cases where the cutoff sat below
        the arithmetic's own noise floor.  Those are now inside the
        cutoff rather than outside the band.  "Every" is still 605
        measured disagreements and not a proof; the one class it
        deliberately does not cover is an eigenvalue that came back zero
        or negative.

    Raises
    ------
    ValueError
        If ``scale``, ``rank_rtol``, ``noise_std`` or ``specs`` is not a
        value this function can answer for (in particular a σ that is
        zero, negative, non-finite, or underflows the residual's dtype;
        ``specs`` given without ``scale="nominal"`` or withheld with it,
        or a ``specs`` tree that does not mirror ``params``; a bounds
        width that is not positive and finite once squared at the
        parameters' precision; a ``mask`` selecting an integer or
        boolean leaf, or a ``params`` with no floating-point leaf).
    FloatingPointError
        If ``F`` comes out non-finite -- a diverged rollout, an
        overflowing Jacobian, a residual holding a ``NaN``.  There is no
        rank, condition number or bound to read from a NaN
        decomposition, so this raises rather than reporting one.
    """
    if scale not in _SCALES:
        raise ValueError(
            f"scale must be 'relative', 'nominal' or None, got {scale!r}")
    _check_flag("reuse_trace", reuse_trace)
    idx, integer_excluded = _fim_indices(params, mask)
    nominal = _resolve_nominal(params, specs, idx, scale)
    inv_sigma = _resolved_noise(residual_fn, params, noise_std,
                                reuse_trace=reuse_trace)
    run = None
    if reuse_trace:
        idx_key = None if idx is None else tuple(int(i) for i in idx)
        try:
            run = _fim_jacobian_compiled(residual_fn, scale, idx_key, nominal)
        except TypeError:
            # An unhashable ``residual_fn`` (a callable object that
            # defines ``__eq__`` without ``__hash__``) cannot key the
            # cache.  That is a reason to skip the cache, not to refuse
            # the call.
            run = None
    if run is None:
        # The default, and deliberately not jitted-and-cached: this
        # traces ``residual_fn`` now, so everything it reads from outside
        # its argument is read as it stands at this call.  ``J`` is
        # bit-for-bit the jitted one (see :func:`_fim_jacobian`), so the
        # report does not depend on which branch built it.
        J, zero_mask = _fim_jacobian(residual_fn, params, scale=scale,
                                     idx=idx, inv_sigma=inv_sigma,
                                     nominal=nominal)
    else:
        J, zero_mask = run(params, inv_sigma)
    # Eager, and deliberately: see :func:`_fim_jacobian` for why the
    # Gram product stays off the compiler's hands in the reporting path.
    F, finite, eigvals, eigvecs = _fim_spectrum(J)

    # One blocking read, four buffers, and everything below is host
    # numpy.  ``_rank_and_crb`` and ``_precision_limited`` re-widen to
    # float64 and would each have pulled ``eigvals`` across on their
    # own; handing them arrays that are already on the host means the
    # count does not grow with the number of host-side verdicts.  It is
    # asserted in the tests, because a stray ``float(...)`` reintroducing
    # a sync is invisible in every other way.
    eigvals_h, eigvecs_h, finite_h, zero_h = jax.device_get(
        (eigvals, eigvecs, finite, zero_mask))
    if not bool(finite_h):
        raise FloatingPointError(
            "non-finite Fisher matrix: F = J^T J holds inf or NaN at these "
            "params. eigh of a non-finite matrix returns NaN eigenvalues and "
            "NaN eigenvectors, from which no rank, condition number or "
            "Cramer-Rao bound can be read, so this raises here -- as fit() "
            "does on a non-finite gradient -- rather than returning a report "
            "built on NaN. Check that residual_fn(params) is finite (a "
            "diverged rollout), that its Jacobian does not overflow, and that "
            "noise_std is not so small that r / sigma does."
        )
    names = _param_names(params, differentiable_only=True)
    if idx is not None:
        names = tuple(names[i] for i in idx)
    zero_scaled: tuple[str, ...] = tuple(
        nm for nm, at_zero in zip(names, zero_h) if bool(at_zero))
    # float() of two float32s, divided in float64 -- as before.  Doing it
    # on the device instead would round the quotient to float32 and move
    # a published number for no gain; ``eigvals`` is on the host already.
    lo, hi = float(eigvals_h[0]), float(eigvals_h[-1])
    cond = float("inf") if lo <= 0.0 else hi / lo
    n_residual = int(J.shape[0])
    rank, crb = _rank_and_crb(eigvals_h, eigvecs_h, rank_rtol,
                              n_residual=n_residual)
    n_params = int(eigvals_h.shape[0])
    limited = _precision_limited(
        eigvals_h,
        _resolve_rank_rtol(eigvals_h.dtype, n_params, rank_rtol,
                           n_residual=n_residual),
        # The precision floor the band is anchored to is the *default*
        # cutoff, which is what "the resolution of this arithmetic" means.
        # It has to move with the cutoff: anchoring to the n-only form
        # would re-impose the blind spot the sqrt(m) term removes, on the
        # warning if no longer on the rank.
        _resolve_rank_rtol(eigvals_h.dtype, n_params, None,
                           n_residual=n_residual),
    )
    if limited is not None:
        ratio, cutoff = limited
        dtype = eigvals_h.dtype
        # Under x64 the remedy has already been taken, and repeating it
        # would send a user round a loop they have finished.  There is
        # no third precision to escalate to, so say what is left: the
        # comparison is at the floor of the best precision available.
        remedy = (
            "Settle it by re-running under x64 -- "
            "jax.config.update('jax_enable_x64', True) before the first "
            "array is made, or JAX_ENABLE_X64=1 -- which moves the "
            "cutoff to max(n, sqrt(m)) * 2.22e-16 and computes the "
            "ratio to match."
            if dtype == np.float32 else
            "This is already the widest precision JAX offers, so no "
            "re-run settles it: the two numbers being compared are "
            "genuinely indistinguishable here. Decide the direction by "
            "hand -- raise rank_rtol to call it unidentifiable, or "
            "rescale the residual so the comparison is not this close."
        )
        warnings.warn(
            f"rank={rank} of {len(names)} was decided at the "
            f"{dtype} noise floor: the eigenvalue ratio nearest the "
            f"cutoff is {ratio:.4g} against a cutoff of {cutoff:.4g}, a "
            f"factor of {max(ratio / cutoff, cutoff / ratio):.3g} -- "
            f"within the {_PRECISION_WARN_FACTOR:g}x band where the "
            f"float32 and float64 verdicts were measured to disagree. "
            f"eigh returns each eigenvalue with an absolute error of "
            f"order n * eps * max(eigvals), and forming F = J^T J over "
            f"{n_residual} residual rows in {dtype} costs about "
            f"sqrt(m) * eps * max(eigvals) again -- the cutoff is the "
            f"larger of the two -- so a difference this "
            f"size is rounding and not information: rank, crb and cond "
            f"all follow this one comparison and are provisional "
            f"together. {remedy} This is advisory: nothing about the "
            f"report is "
            f"wrong, only undetermined. Silence it with "
            f"warnings.simplefilter('ignore', PrecisionLimitWarning), "
            f"or raise rank_rtol to declare the direction "
            f"unidentifiable on purpose.",
            PrecisionLimitWarning,
            stacklevel=2,
        )
    return FIMReport(
        fim=F, eigvals=eigvals, eigvecs=eigvecs, rank=rank, cond=cond,
        crb=crb, param_names=names, zero_scaled=zero_scaled,
        value_scaled=_value_scaled_names(nominal),
        integer_excluded=integer_excluded,
    )


# ---------------------------------------------------------------------------
# Fitting under ParamSpec (mask + reparametrisation)
# ---------------------------------------------------------------------------


EVENT_FIT_PROGRESS = "fit_progress"


def _check_hyper(name: str, value, *, gt=None, ge=None, lt=None, le=None,
                 why: str = "") -> float:
    """Reject a hyper-parameter with no reading the algorithm can act on.

    These are refused rather than clamped because each one produces a
    plausible :class:`FitResult` and no other symptom: ``lr <= 0`` runs
    Adam as gradient *ascent* (or as a no-op) and still reports losses
    and an ``n_iter``; ``eps <= 0`` divides by zero in the Adam
    denominator, and the resulting NaN is reported as "non-finite loss
    or gradient", blaming the model; ``tol = NaN`` compares False
    against every loss, so the early stop never fires and the run reads
    as merely unconverged; ``lam_up <= 1`` shrinks Levenberg-Marquardt's
    damping on a *rejected* step, which is the opposite of what
    rejecting a step means.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number, got {value!r}.{why}") from None
    if not np.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {value!r}.{why}")
    if gt is not None and not v > gt:
        raise ValueError(f"{name} must be greater than {gt}, got {v!r}.{why}")
    if ge is not None and not v >= ge:
        raise ValueError(f"{name} must be at least {ge}, got {v!r}.{why}")
    if lt is not None and not v < lt:
        raise ValueError(f"{name} must be less than {lt}, got {v!r}.{why}")
    if le is not None and not v <= le:
        raise ValueError(f"{name} must be at most {le}, got {v!r}.{why}")
    return v


def _check_count(name: str, value, *, minimum: int = 0) -> int:
    """Reject a non-integer or too-small iteration/interval count."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _check_adam_hyper(n_iter, lr, tol, betas, eps, notify_every) -> None:
    """The hyper-parameters :func:`fit` and :func:`fit_multiple_shooting`
    share."""
    _check_count("n_iter", n_iter)
    _check_hyper("lr", lr, gt=0.0,
                 why=" A non-positive learning rate descends nothing: at 0 the "
                     "step vanishes, below it the step climbs the loss.")
    _check_hyper("tol", tol, ge=0.0,
                 why=" tol is compared with <=, so NaN never stops the loop and "
                     "a negative tol never can either; 0.0 disables the early stop.")
    try:
        b1, b2 = betas
    except (TypeError, ValueError):
        raise ValueError(f"betas must be a pair (beta1, beta2), got {betas!r}") from None
    for i, b in enumerate((b1, b2)):
        _check_hyper(f"betas[{i}]", b, ge=0.0, lt=1.0,
                     why=" Adam's bias correction divides by 1 - beta**i, which "
                         "is exactly 0 at beta = 1.")
    _check_hyper("eps", eps, gt=0.0,
                 why=" eps is the floor of Adam's denominator; at 0 the first "
                     "step is 0/0 and the NaN is reported against the loss.")
    _check_count("notify_every", notify_every)


#: Largest trainable-parameter count for which :func:`fit` accumulates the
#: gradient Gram matrix that :class:`_ExcitationTracker` uses.
#:
#: The tracker holds one ``n x n`` float64 matrix and costs one ``n x n``
#: outer product per iteration plus one ``eigh`` at the end -- ``n**2``
#: fused multiply-adds per iteration and ``O(n**3)`` once, against the full
#: rollout and reverse pass the gradient itself costs, and it only runs at
#: all once there are ``n`` gradients to pool.  At the cap that is 2.1 MB,
#: 0.26 MFLOP per iteration and ~0.13 GFLOP for the decomposition.  Wall
#: clock is not quoted: measured on this box the same 512-wide ``eigh``
#: ranged over 52-892 ms across five back-to-back runs, which is contention
#: from the other work sharing it and not a property of the code.  A caller
#: who does not want the cost at all passes ``hold_undetermined=False``.
#:
#: Above the cap the tracker is not built and
#: :attr:`FitResult.excited_rank` is ``None`` -- "not measured", never
#: "full rank", because a silent full-rank verdict would read as "no
#: undetermined direction was found" when nothing looked.
_EXCITATION_MAX_PARAMS = 512


class _ExcitationTracker:
    """Accumulated gradient second moment ``G = sum_t g_t g_t^T`` of a fit.

    Every gradient of a least-squares loss is ``J^T r``, so it lies in the
    row space of ``J``: a direction ``v`` with ``J v = 0`` has ``g . v = 0``
    at *every* iterate, and the whole run's gradients stay inside the
    subspace the data can see.  ``G``'s near-null eigenvectors therefore
    name the directions no gradient ever pointed along -- the ones the
    optimiser had no information about and must not have moved in.

    Pooling over the run rather than testing one gradient at a time is what
    makes the verdict robust: near convergence a single gradient is mostly
    rounding noise, but its *accumulated* energy along an exactly-null
    direction stays at the arithmetic's floor while every direction the data
    resolves keeps the energy it collected during the descent.  Measured on
    the spring's ``(k, c, m)`` common-scale degeneracy, the null eigenvalue
    sits at ``1e-15`` of the largest over 200 to 10,000 iterations and
    learning rates 0.01 to 0.2, against ``8e-3`` for the weakest direction
    the data does resolve -- twelve decades of separation.

    ``G`` is accumulated in float64 whatever the gradients' own precision:
    the quantity being resolved is twelve decades below the top eigenvalue,
    which float32 accumulation cannot represent at all.
    """

    def __init__(self, n: int, dtype) -> None:
        self.n = int(n)
        self.eps = float(np.finfo(dtype).eps)
        self.count = 0
        self._gram = np.zeros((self.n, self.n), dtype=np.float64)

    def observe(self, g) -> None:
        """Fold one gradient into the accumulated second moment."""
        gv = np.asarray(g, dtype=np.float64).reshape(-1)
        self._gram += np.outer(gv, gv)
        self.count += 1

    def projector(self) -> tuple[Optional[int], Optional[np.ndarray]]:
        """``(excited_rank, P)`` for the excited subspace, or ``(None, None)``.

        ``P`` is the orthogonal projector onto the span of the directions
        the gradients excited, and is ``None`` when the rank is full (there
        is nothing to remove, and returning the iterate untouched keeps it
        bit for bit).  ``(None, None)`` means the question was not answered:
        fewer gradients than parameters, so a direction can be unobserved
        merely for want of iterations, or a degenerate spectrum.

        The cutoff is ``(max(n, sqrt(T)) * eps)**2`` of the largest
        eigenvalue, for ``n`` parameters and ``T`` gradients -- the same rule
        and the same reasoning as :func:`_resolve_rank_rtol`'s default, moved
        one level out: ``max(n, sqrt(T)) * eps`` is the relative floor of a
        gradient *component* under an ``eigh`` of an ``n x n`` matrix summed
        over ``T`` terms, and ``G``'s eigenvalues are those components
        squared.  It is a numerical floor, not a statistical one: it removes
        only directions the arithmetic says carry no gradient at all, and
        leaves every merely weakly-identified direction in place.  The
        statistical question -- "is this parameter determined *well enough*"
        -- is :attr:`FIMReport.crb`'s, and answering it here would silently
        discard real, if weak, information.
        """
        if self.count < self.n:
            return None, None
        evals, evecs = np.linalg.eigh(self._gram)
        top = float(evals[-1])
        if not np.isfinite(top) or top <= 0.0:
            return None, None
        cutoff = (max(self.n, math.sqrt(self.count)) * self.eps) ** 2 * top
        keep = np.asarray(evals > cutoff)
        rank = int(np.count_nonzero(keep))
        if rank == self.n:
            return rank, None
        basis = evecs[:, keep]
        return rank, basis @ basis.T


def _check_flag(name: str, value) -> None:
    """Refuse a non-bool switch.

    A truthy non-bool (``"no"``, ``0.0``, an array) would silently pick a
    branch the caller did not mean, and for both switches that use this
    the branch decides whether the answer can be trusted:
    ``hold_undetermined`` whether the returned parameters are
    reproducible, ``reuse_trace`` whether ``fim`` may answer from an
    earlier call's trace.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}")


def _check_hold_undetermined(value) -> None:
    """Refuse a non-bool ``hold_undetermined``.

    Called at the top of each fitter, with the rest of the
    hyper-parameter checks, so an argument error is raised before any
    model evaluation -- and again inside :func:`_make_excitation_tracker`,
    so no caller of that can skip it.  See :func:`_check_flag`.
    """
    _check_flag("hold_undetermined", value)


def _make_excitation_tracker(hold_undetermined, theta0) -> Optional[_ExcitationTracker]:
    """The tracker for a fit's trainable block, or ``None`` if it cannot run.

    Shared by :func:`fit`, :func:`fit_lm` and :func:`fit_multiple_shooting`
    so that all three answer :attr:`FitResult.excited_rank` by the same
    rule.  ``None`` means "not measured", which is what
    :attr:`FitResult.excited_rank` reports as ``None``: the caller switched
    the guard off, there are more trainable coordinates than
    :data:`_EXCITATION_MAX_PARAMS`, or the block is empty or not
    floating-point.
    """
    _check_hold_undetermined(hold_undetermined)
    if (hold_undetermined
            and 0 < theta0.size <= _EXCITATION_MAX_PARAMS
            and jnp.issubdtype(theta0.dtype, jnp.floating)):
        return _ExcitationTracker(int(theta0.size), theta0.dtype)
    return None


def _hold_undetermined_directions(tracker, theta, theta0):
    """``(theta, excited_rank, undetermined_drift)`` with the undetermined
    component of ``theta - theta0`` removed.

    The one implementation of the hold, shared by all three fitters: an
    optimiser-specific copy would let the three drift apart in exactly the
    quantity they exist to make reproducible.  Every step rule this module
    has moves along ``null(J)`` for its own reason -- Adam through its
    diagonal preconditioner, Levenberg-Marquardt through ``lam * diag(A)``,
    which is orthogonal to ``null(A)`` only where ``diag(A)`` is isotropic
    there -- and none of them is told anything about those directions by
    the data.

    ``theta`` comes back untouched, and therefore bit for bit, whenever the
    rank is full or the question could not be answered.
    """
    if tracker is None:
        return theta, None, None
    excited_rank, projector = tracker.projector()
    if projector is None:
        return theta, excited_rank, (None if excited_rank is None else 0.0)
    moved = (np.asarray(theta, dtype=np.float64)
             - np.asarray(theta0, dtype=np.float64))
    kept = projector @ moved
    drift = float(np.linalg.norm(moved - kept))
    # ``theta0 + kept``, not ``kept`` alone: the guard holds the
    # undetermined directions at the values they *started* at, which
    # is the one thing about them the data has not contradicted.
    return theta0 + jnp.asarray(kept, dtype=theta0.dtype), excited_rank, drift


def _progress_notifier(gm, method: str, n_iter: int, notify_every: int):
    """Observer notification every ``notify_every`` iterations (or None)."""
    if notify_every <= 0 or not getattr(gm, "_observers", None):
        return None

    def _emit(i, loss, params):
        if i % notify_every == 0 or i == n_iter:
            gm._notify(EVENT_FIT_PROGRESS, {  # noqa: SLF001
                "method": method, "iteration": int(i), "n_iter": int(n_iter),
                "loss": float(loss), "params": params,
            })
    return _emit


@stability(StabilityLevel.EVOLVING)
@dataclass(frozen=True, kw_only=True)
class FitResult:
    """Outcome of :func:`fit`, :func:`fit_lm` and
    :func:`fit_multiple_shooting`.

    Keyword-only by construction, for the reason :class:`FIMReport`
    became so during 0.4.0: inserting a field anywhere but the end
    reassigns every positional argument after it, with no ``TypeError``
    and no warning.  Here the two adjacent ``bool``/``int`` fields make
    it worse than a shift -- ``converged`` and ``n_iter`` each accept
    the other's value silently, so a run that stopped at iteration 12
    would read as converged.  No field has been inserted yet and no
    caller built one positionally; ``kw_only`` is what keeps that true
    for the next field.

    ``params`` is a physical pytree (already mapped back through
    ``GraphManager.constrain``); ``losses[i]`` is the loss *before*
    update ``i``; ``converged`` is whether ``losses[-1] <= tol``.

    Every leaf no step moved is the value that went in, bit for bit --
    not merely close -- so comparing a fit's input and output leaf by
    leaf says exactly which constants the calibration touched.  That
    covers the leaves outside the resolved mask and the masked ones a
    run that stopped before its first update never stepped: neither is
    round-tripped through ``constrain(unconstrain(p))``, which for a
    ``log`` leaf is ``exp(log(p))`` and lands one ulp away.

    ``excited_rank`` and ``undetermined_drift`` report the identifiability
    guard (``hold_undetermined``), which all three fitters run, and are
    ``None`` from one that could not answer the question.

    ``excited_rank``
        How many independent directions the run's gradients spanned, out
        of the trainable coordinate count.  Less than that count means the
        data left the rest undetermined and ``params`` holds the value they
        started at.  ``None`` is **not** "full rank": it is "not measured"
        -- ``hold_undetermined=False``, more trainable parameters than
        :data:`_EXCITATION_MAX_PARAMS`, or fewer iterations than parameters,
        where an unobserved direction cannot be told from an unobservable
        one.
    ``undetermined_drift``
        How far the raw iterate had wandered along those undetermined
        directions before the guard removed it, as a Euclidean norm in the
        **unconstrained** coordinates (``log`` for a positive parameter, so
        a drift of 0.04 there is a 4% drift in the parameter itself).
        ``0.0`` when the rank was full and nothing was removed; ``None``
        when ``excited_rank`` is.  It is a diagnostic, not a residual
        error: the value it reports has already been taken out of
        ``params``.  A number far above the fit's own step scale says the
        loss surface has a flat direction worth naming with :func:`fim`.

        For :func:`fit_multiple_shooting` it covers the **parameter** block
        only.  The window starts are that fit's own decision variables and
        are returned as the optimiser left them, in the second element of
        its result rather than in ``params``.
    """
    params: dict
    losses: np.ndarray
    converged: bool
    n_iter: int
    excited_rank: Optional[int] = None
    undetermined_drift: Optional[float] = None


@stability(StabilityLevel.EVOLVING)
def fit(
    gm,
    loss_fn: Callable[[dict], Any],
    *,
    params: Optional[dict] = None,
    mask: Optional[dict] = None,
    n_iter: int = 200,
    lr: float = 0.05,
    tol: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    callback: Optional[Callable[[int, float, dict], None]] = None,
    notify_every: int = 1,
    hold_undetermined: bool = True,
) -> FitResult:
    """Adam on ``loss_fn(params)`` respecting the graph's :class:`ParamSpec`.

    The optimiser works in the unconstrained coordinates of
    ``GraphManager.unconstrain`` (log for positive constants, logit for
    intervals), so a positive parameter cannot cross zero and a bounded
    one cannot leave its interval, and it only moves the leaves the
    ``mask`` marks trainable (default ``gm.trainable_mask()``: the
    nodes' declarations plus ``gm.set_param_spec`` overrides; a ``mask``
    may narrow that set but not widen it).  Freezing a leaf with
    ``gm.set_param_spec(node, key, ParamSpec(trainable=False))`` after
    :func:`fim` has named it keeps a fit out of a parameter the data
    cannot see at all.

    A degeneracy is rarely one parameter, though — the spring's data
    determines ``k/m`` and ``c/m`` but not the scale of ``(k, c, m)``,
    and no single leaf is the culprit.  ``hold_undetermined`` (default
    ``True``) holds those *directions* instead: see below.

    Parameters
    ----------
    gm : GraphManager
        Supplies the specs; ``gm.params`` is the default start.
    loss_fn : callable
        ``params (physical pytree) -> scalar``; typically a closure over
        :func:`windowed_loss`.
    params : dict, optional
        Starting pytree (``gm.params`` layout).
    mask : pytree of bool, optional
        Narrows ``gm.trainable_mask()``.  Same tree structure as
        ``params``, key for key -- the flags are read in flatten order,
        so a mask with the right leaf count and different keys would fit
        a different parameter, and is refused.  It may only *narrow* the
        trainable set: a mask that marks a leaf whose :class:`ParamSpec` declares
        ``trainable=False`` is a ``ValueError``, because ``constrain`` /
        ``unconstrain`` transform and clip a leaf only when its spec
        says trainable — make the parameter trainable in the spec
        instead, which is what activates its bounds and transform.
        A trainable set (the default one included) holding an integer or
        boolean leaf is refused the same way: its gradient is identically
        zero, so the fit could only hand it back unchanged.
    n_iter, lr, tol, betas, eps
        Adam hyper-parameters; ``tol > 0`` stops early once the loss is
        at or below it.
    callback : callable, optional
        ``callback(i, loss, params)`` after each evaluation.
    notify_every : int
        Every ``notify_every`` iterations the graph's observers receive a
        ``"fit_progress"`` event (``GraphManager.add_observer``) with
        ``{"method", "iteration", "n_iter", "loss", "params"}`` — the REST
        relay / live stage can show a calibration as it runs.  ``0``
        disables it.
    hold_undetermined : bool
        Keep the fitted parameters out of the directions the data does not
        determine, holding them at the values they started at.  **New in
        0.4.0, and on by default**; ``False`` restores the pre-0.4.0
        iterate exactly.

        Every gradient of a least-squares loss is ``Jᵀr``, so a direction
        ``v`` with ``Jv = 0`` has ``g·v = 0`` at every iterate: the data
        never says anything about it, and the loss is flat along it.  Adam
        moves along it anyway — the diagonal preconditioner makes
        ``Δ·v = gᵀDv`` nonzero even where ``g·v`` is zero — so the returned
        value of a degenerate combination is set by the iteration count and
        the learning rate rather than by the data.  Measured on the spring's
        ``(k, c, m)`` scale degeneracy at ``lr=0.2``: the geometric mean of
        the three drifts 3.4% by iteration 200 and **46% by iteration
        10,000**, with the loss unchanged in its first six digits, and
        ``mass`` lands anywhere from 1.33 to 1.90 depending only on the
        budget.  :func:`fit_lm` and :func:`fit_multiple_shooting` drift too
        — ``λ·diag(A)`` damping is no more orthogonal to the null space
        than Adam's preconditioner is — and since 0.4.0 all three run this
        same guard.  What differs is that LM's step vanishes with the
        gradient, so its drift *converges*; see :func:`fit_lm`.

        With the guard on, :func:`fit` accumulates ``Σ_t g_t g_tᵀ`` over the
        run and removes the net displacement's component along that matrix's
        numerically-null eigenvectors.  The **loss is unaffected** (it is
        flat in exactly those directions), the iterates, ``losses``,
        ``callback`` and observer events are unchanged, and a fit whose
        gradients spanned everything gets its iterate back bit for bit — so
        a well-posed fit sees no difference at all.
        :attr:`FitResult.excited_rank` and
        :attr:`FitResult.undetermined_drift` say what the guard found.

        Two limits, both fail-open.  The cutoff is *numerical*, so a merely
        weakly-identified direction is kept, not held — ask :func:`fim` for
        ``crb`` to decide whether a direction is determined *well enough*.
        And the degeneracy must be one the whole run saw: a null direction
        that rotates in the unconstrained coordinates as the fit moves
        (mixed ``log`` and identity transforms on the parameters it mixes,
        say) leaves no null direction in the accumulated matrix, and the
        guard correctly reports full ``excited_rank`` and does nothing.
        Declaring ``transform="log"`` on every parameter of a scale
        degeneracy is what makes it constant, and it is what
        ``fim(scale="relative")`` already assumes.
    """
    _check_adam_hyper(n_iter, lr, tol, betas, eps, notify_every)
    _check_hold_undetermined(hold_undetermined)
    start = gm._params_or_default(params)  # noqa: SLF001
    gm.check_params(start)
    mask = _resolve_mask(gm, start, mask)
    b1, b2 = betas
    progress = _progress_notifier(gm, "adam", n_iter, notify_every)

    u0 = gm.unconstrain(start)
    flat_u, unravel = ravel_pytree(u0)
    idx = _masked_indices(start, mask)
    if idx is None:
        idx = np.arange(flat_u.size)
    to_params = _physical_params(gm, start, flat_u, unravel, idx)
    theta0 = flat_u[idx]

    def objective(theta):
        u = unravel(flat_u.at[idx].set(theta))
        return loss_fn(gm.constrain(u))

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    @jax.jit
    def adam_step(theta, m, v, g, i):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** i)
        v_hat = v / (1 - b2 ** i)
        return theta - lr * m_hat / (jnp.sqrt(v_hat) + eps), m, v

    theta = theta0
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    tracker = _make_excitation_tracker(hold_undetermined, theta0)
    losses: list[float] = []
    converged = False
    i = 0
    for i in range(1, n_iter + 1):
        loss, g = value_and_grad(theta)
        loss_f = float(loss)
        losses.append(loss_f)
        if not np.isfinite(loss_f) or not bool(jnp.all(jnp.isfinite(g))):
            raise FloatingPointError(
                f"non-finite loss or gradient at iteration {i} (loss={loss_f})"
            )
        if tracker is not None:
            # Before the ``tol`` break, not after the step: this gradient is
            # information about the loss surface whether or not it moved
            # anything, and a run that stops on ``tol`` has still seen it.
            tracker.observe(g)
        if callback is not None or progress is not None:
            current = to_params(theta)
            if callback is not None:
                callback(i, loss_f, current)
            if progress is not None:
                progress(i, loss_f, current)
        if tol > 0.0 and loss_f <= tol:
            converged = True
            break
        theta, m, v = adam_step(theta, m, v, g, jnp.asarray(i, theta.dtype))

    theta, excited_rank, undetermined_drift = _hold_undetermined_directions(
        tracker, theta, theta0)

    final = to_params(theta)
    return FitResult(
        params=final, losses=np.asarray(losses), converged=converged, n_iter=i,
        excited_rank=excited_rank, undetermined_drift=undetermined_drift,
    )


@stability(StabilityLevel.EVOLVING)
def fit_lm(
    gm,
    residual_fn: Callable[[dict], Any],
    *,
    params: Optional[dict] = None,
    mask: Optional[dict] = None,
    n_iter: int = 50,
    lam0: float = 1e-2,
    lam_up: float = 10.0,
    lam_down: float = 0.1,
    tol: float = 0.0,
    step_tol: float = 1e-8,
    noise_std: Optional[Any] = None,
    callback: Optional[Callable[[int, float, dict], None]] = None,
    notify_every: int = 1,
    hold_undetermined: bool = True,
) -> FitResult:
    """Levenberg–Marquardt on ``0.5 * ||residual_fn(params)||²`` under the
    graph's :class:`ParamSpec` (unconstrained coordinates, trainable mask).

    Uses the same ``jacfwd`` sensitivities as :func:`fim` — one forward
    pass per trainable parameter — so for the handful of physical
    constants a model has it converges in a few iterations where Adam
    needs hundreds, and at the solution ``JᵀJ`` *is* the Fisher matrix.
    The damping ``λ`` multiplies ``diag(JᵀJ)`` (Marquardt scaling): a
    step that lowers the loss is accepted and ``λ`` shrinks by
    ``lam_down``; a step that raises it is rejected and ``λ`` grows by
    ``lam_up``.  ``noise_std`` weights the residual as in :func:`fim`.

    Returns a :class:`FitResult` whose ``losses[i]`` is the (weighted)
    ``0.5 ||r||²`` at the start of iteration ``i``; ``converged`` when
    the loss reached ``tol`` or the step norm in unconstrained
    coordinates fell below ``step_tol``.

    Parameters
    ----------
    hold_undetermined : bool
        Keep the fitted parameters out of the directions the data does not
        determine, exactly as :func:`fit` does and by the same shared
        machinery.  **New in 0.4.0, and on by default**; ``False`` restores
        the pre-0.4.0 iterate exactly.

        The gradient ``g = Jᵀr`` is orthogonal to ``null(J)``, but the step
        is not the gradient: the Marquardt solve is
        ``(A + λ·diag(A))⁻¹ g``, and that is orthogonal to ``null(A)`` only
        where ``diag(A)`` is isotropic on the relevant subspace.  Take
        ``A = [[1, 2], [2, 4]]`` and ``g = (1, 2)``: the step is
        ``∝ (2, 1)`` for every ``λ`` while ``null(A)`` is spanned by
        ``(2, −1)``.  So LM drifts along an exactly-null direction too.

        It drifts **differently from Adam**, and the difference is worth
        knowing.  LM's step vanishes with the gradient, so the drift
        converges and stops.  Measured on the spring's ``(k, c, m)``
        common-scale degeneracy, the geometric mean of the three lands
        **−0.849%** (noiseless data) or **+0.429%** (σ = 0.02) from the
        value it was given, and then does not move again: the answer is
        identical to the last bit for ``n_iter`` 10 through 200.  On the
        same data :func:`fit` lands −7.1% at ``lr=0.05`` and −5.4% at
        ``lr=0.2`` — a spread the *schedule* chooses, not the data.

        So this is a **consistency** fix and not the reproducibility defect
        :func:`fit` had: the value LM returns for an undetermined
        combination does not depend on the budget, it is simply neither the
        caller's value nor one the data chose.  The reason to fix it anyway
        is that :attr:`FitResult.excited_rank` and
        :attr:`FitResult.undetermined_drift` would otherwise be ``None``
        here for no better reason than that nobody had done it.

        Everything :func:`fit` promises holds here.  The loss, the
        iterates, ``losses``, ``callback`` and the ``fit_progress`` events
        are unchanged; a fit whose gradients spanned every direction gets
        its iterate back bit for bit; the cutoff is numerical rather than
        statistical, so a merely weakly-identified direction is kept;
        and the degeneracy has to be a fixed direction in the optimiser's
        coordinates.  :attr:`FitResult.excited_rank` and
        :attr:`FitResult.undetermined_drift` say what the guard found.
    """
    _check_count("n_iter", n_iter)
    _check_hyper("lam0", lam0, gt=0.0,
                 why=" lambda multiplies diag(J^T J): at 0 or below the solve is "
                     "the undamped Gauss-Newton one and there is nothing for a "
                     "rejected step to raise.")
    _check_hyper("lam_up", lam_up, gt=1.0,
                 why=" lam_up is applied when a step is *rejected*, so a factor "
                     "at or below 1 damps less after a failure and the retry "
                     "loop cannot recover.")
    _check_hyper("lam_down", lam_down, gt=0.0, le=1.0,
                 why=" lam_down is applied when a step is *accepted*, so a "
                     "factor above 1 damps more after a success.")
    _check_hyper("tol", tol, ge=0.0,
                 why=" tol is compared with <=, so NaN never stops the loop and "
                     "a negative tol never can either; 0.0 disables the early stop.")
    _check_hyper("step_tol", step_tol, ge=0.0,
                 why=" step_tol is compared with <, so a negative one can never "
                     "be met and 'converged' could only ever mean the loss "
                     "reached tol.")
    _check_count("notify_every", notify_every)
    _check_hold_undetermined(hold_undetermined)
    start = gm._params_or_default(params)  # noqa: SLF001
    gm.check_params(start)
    mask = _resolve_mask(gm, start, mask)
    u0 = gm.unconstrain(start)
    flat_u, unravel = ravel_pytree(u0)
    idx = _masked_indices(start, mask)
    if idx is None:
        idx = np.arange(flat_u.size)
    to_params = _physical_params(gm, start, flat_u, unravel, idx)
    theta0 = flat_u[idx]
    theta = theta0
    progress = _progress_notifier(gm, "lm", n_iter, notify_every)

    # ``_resolved_noise``, not ``_inverse_noise_std(noise_std,
    # residual_fn(...))``: the residual is wanted for its *structure*
    # only, and evaluating it cost a whole extra rollout per call --
    # bought nothing at all in the ``noise_std is None`` case, which
    # returns before looking at it.  Confirmed by counting entries into
    # ``residual_fn``: 4 per call, of which this was 1, and 1 per call at
    # ``n_iter=0`` where nothing else ran at all.
    inv_sigma = _resolved_noise(residual_fn, gm.constrain(u0), noise_std)

    def _residual(th):
        p = gm.constrain(unravel(flat_u.at[idx].set(th)))
        r = ravel_pytree(residual_fn(p))[0]
        return r if inv_sigma is None else r * inv_sigma

    residual_and_jac = jax.jit(lambda th: (_residual(th), jax.jacfwd(_residual)(th)))
    residual_only = jax.jit(_residual)

    @jax.jit
    def _lm_step(th, r, J, lam):
        A = J.T @ J
        g = J.T @ r
        A_damped = A + lam * jnp.diag(jnp.diag(A)) + 1e-12 * jnp.eye(A.shape[0], dtype=A.dtype)
        delta = jnp.linalg.solve(A_damped, g)
        return th - delta

    lam = float(lam0)
    tracker = _make_excitation_tracker(hold_undetermined, theta0)
    losses: list[float] = []
    converged = False
    i = 0
    for i in range(1, n_iter + 1):
        r, J = residual_and_jac(theta)
        loss = 0.5 * float(jnp.sum(r * r))
        if not np.isfinite(loss) or not bool(jnp.all(jnp.isfinite(J))):
            raise FloatingPointError(f"non-finite residual or Jacobian at iteration {i}")
        losses.append(loss)
        if tracker is not None:
            # ``J.T @ r`` is the same ``g`` ``_lm_step`` forms, recomputed
            # here rather than returned from it: the tracker must not
            # change the step, and ``_lm_step``'s own ``g`` lives inside a
            # ``jax.jit`` whose fusion decides ``A``'s last bits (PR 101).
            # Contracted on the device so only the ``n``-vector is read
            # back, not the whole ``m x n`` Jacobian.  Folded in before
            # the ``tol`` break, as in ``fit``: this gradient is
            # information about the loss surface whether or not a step
            # was taken on it.
            tracker.observe(J.T @ r)
        if callback is not None or progress is not None:
            current = to_params(theta)
            if callback is not None:
                callback(i, loss, current)
            if progress is not None:
                progress(i, loss, current)
        if tol > 0.0 and loss <= tol:
            converged = True
            break
        # Try a step; shrink lambda on success, grow it (and retry) on failure.
        accepted = False
        step_norm = float("inf")   # only meaningful once a step is accepted
        for _ in range(12):
            cand = _lm_step(theta, r, J, jnp.asarray(lam, theta.dtype))
            r_new = residual_only(cand)
            loss_new = 0.5 * float(jnp.sum(r_new * r_new))
            if np.isfinite(loss_new) and loss_new < loss:
                step_norm = float(jnp.linalg.norm(cand - theta))
                theta = cand
                lam = max(lam * lam_down, 1e-12)
                accepted = True
                break
            lam = min(lam * lam_up, 1e12)
        if not accepted or step_norm < step_tol:
            converged = accepted and step_norm < step_tol
            break

    theta, excited_rank, undetermined_drift = _hold_undetermined_directions(
        tracker, theta, theta0)

    final = to_params(theta)
    return FitResult(
        params=final, losses=np.asarray(losses), converged=converged, n_iter=i,
        excited_rank=excited_rank, undetermined_drift=undetermined_drift,
    )


@stability(StabilityLevel.EVOLVING)
def fit_multiple_shooting(
    gm,
    observations: dict,
    *,
    obs_fn: Callable[[dict], Any],
    window: int,
    params: Optional[dict] = None,
    mask: Optional[dict] = None,
    window_states: Optional[dict] = None,
    continuity_weight: float = 1.0,
    sample_every: int = 1,
    external_inputs: Optional[dict] = None,
    n_iter: int = 200,
    lr: float = 0.05,
    lr_states: Optional[float] = None,
    tol: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    callback: Optional[Callable[[int, float, dict], None]] = None,
    notify_every: int = 1,
    hold_undetermined: bool = True,
) -> tuple[FitResult, dict]:
    """Multiple-shooting fit: Adam jointly over the trainable params (in
    unconstrained coordinates) and the free per-window initial states.

    Compared with :func:`fit` on the teacher-forced :func:`windowed_loss`,
    the window starts are decision variables and a continuity penalty
    (``continuity_weight``) joins consecutive windows, so the optimum is a
    single continuous trajectory and noisy observations at window starts
    do not seed every window with measurement error.

    Returns ``(FitResult, window_states)``.

    Parameters
    ----------
    hold_undetermined : bool
        Keep the fitted **parameters** out of the directions the data does
        not determine, exactly as :func:`fit` does and by the same shared
        machinery.  **New in 0.4.0, and on by default**; ``False`` restores
        the pre-0.4.0 iterate exactly.

        This is the same Adam step rule :func:`fit` uses, so it is
        :func:`fit`'s defect and not merely the consistency issue
        :func:`fit_lm` had: the drift has not settled, and the *schedule*
        picks the answer.  Measured on the spring's ``(k, c, m)``
        common-scale degeneracy with σ = 0.02 observations, the geometric
        mean of the three lands **−4.35%** at ``lr=0.01, n_iter=200``,
        **−4.80%** at 1,200, **−1.97%** at ``lr=0.2, n_iter=200`` and
        **−2.02%** at 1,200 — a 3.0% spread in the returned constants for a
        loss that agrees to four digits.  Raising the budget to 4,000 moves
        ``lr=0.2`` on again, to −2.31%, so it is not converging to a value
        either.  Two runs on the same data return different physical
        constants and neither is preferred by the objective.

        Only ``theta`` is guarded.  The returned ``window_states`` are
        nuisance variables of the fit rather than constants a caller
        records as provenance, and a caller warm-starting from them needs
        the values the optimiser actually reached; the gradient Gram is
        accumulated over the parameter block alone, which is where
        :attr:`FitResult.excited_rank` counts its directions.  That block's
        gradient is still ``J_θᵀ r`` at every iterate, so a ``v`` with
        ``J_θ v = 0`` has ``g·v = 0``, which is the whole premise.
    """
    _check_adam_hyper(n_iter, lr, tol, betas, eps, notify_every)
    if lr_states is not None:
        _check_hyper("lr_states", lr_states, gt=0.0,
                     why=" The window starts are decision variables like the "
                         "parameters; a non-positive rate moves them the wrong way.")
    _check_hyper("continuity_weight", continuity_weight, ge=0.0,
                 why=" A negative weight pays the fit to tear the trajectory "
                     "apart at the window joins.")
    _check_count("sample_every", sample_every, minimum=1)
    _check_hold_undetermined(hold_undetermined)
    start = gm._params_or_default(params)  # noqa: SLF001
    gm.check_params(start)
    mask = _resolve_mask(gm, start, mask)
    ws0 = init_window_states(observations, window) if window_states is None else window_states
    b1, b2 = betas
    lr_s = lr if lr_states is None else lr_states
    progress = _progress_notifier(gm, "multiple_shooting", n_iter, notify_every)

    u0 = gm.unconstrain(start)
    flat_u, unravel = ravel_pytree(u0)
    idx = _masked_indices(start, mask)
    if idx is None:
        idx = np.arange(flat_u.size)
    to_params = _physical_params(gm, start, flat_u, unravel, idx)
    theta0 = flat_u[idx]
    ws_flat0, unravel_ws = ravel_pytree(ws0)

    def objective(theta, ws_flat):
        p = gm.constrain(unravel(flat_u.at[idx].set(theta)))
        return windowed_loss(
            gm, p, observations, obs_fn=obs_fn, window=window,
            sample_every=sample_every, external_inputs=external_inputs,
            window_states=unravel_ws(ws_flat), continuity_weight=continuity_weight,
        )

    value_and_grad = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))

    @jax.jit
    def adam(x, m, v, g, i, rate):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        return x - rate * (m / (1 - b1 ** i)) / (jnp.sqrt(v / (1 - b2 ** i)) + eps), m, v

    theta, ws = theta0, ws_flat0
    m_t = jnp.zeros_like(theta); v_t = jnp.zeros_like(theta)
    m_s = jnp.zeros_like(ws); v_s = jnp.zeros_like(ws)
    tracker = _make_excitation_tracker(hold_undetermined, theta0)
    losses: list[float] = []
    converged = False
    i = 0
    for i in range(1, n_iter + 1):
        loss, (g_t, g_s) = value_and_grad(theta, ws)
        loss_f = float(loss)
        losses.append(loss_f)
        if not np.isfinite(loss_f) or not bool(jnp.all(jnp.isfinite(g_t))) \
                or not bool(jnp.all(jnp.isfinite(g_s))):
            raise FloatingPointError(f"non-finite loss or gradient at iteration {i}")
        if tracker is not None:
            # The parameter block's gradient only: the window states are
            # decision variables of this fit and are returned as the
            # optimiser left them.  Before the ``tol`` break, as in ``fit``.
            tracker.observe(g_t)
        if callback is not None or progress is not None:
            current = to_params(theta)
            if callback is not None:
                callback(i, loss_f, current)
            if progress is not None:
                progress(i, loss_f, current)
        if tol > 0.0 and loss_f <= tol:
            converged = True
            break
        it = jnp.asarray(i, theta.dtype)
        theta, m_t, v_t = adam(theta, m_t, v_t, g_t, it, lr)
        ws, m_s, v_s = adam(ws, m_s, v_s, g_s, it, lr_s)

    theta, excited_rank, undetermined_drift = _hold_undetermined_directions(
        tracker, theta, theta0)

    final = to_params(theta)
    return (FitResult(params=final, losses=np.asarray(losses), converged=converged,
                      n_iter=i, excited_rank=excited_rank,
                      undetermined_drift=undetermined_drift),
            unravel_ws(ws))
