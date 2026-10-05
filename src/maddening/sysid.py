"""System identification helpers built on the graph parameter pytree.

Three pieces, the first two pure JAX so they compose with ``jax.jit`` /
``jax.grad``:

* :func:`windowed_loss` — a teacher-forced, windowed trajectory loss.
  Long rollouts of stiff or chaotic dynamics give exploding gradients; the
  standard remedy is to reset the simulation to the measured state every
  ``window`` samples and sum the per-window losses.  Each window is its
  own ``lax.scan``, so memory is O(window) and gradients cannot compound
  across windows: the coupling warm starts a window inherits from the one
  before it (see :func:`windowed_loss`) enter with their gradient stopped.
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
import jax.core
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from maddening.core.coupling.acceleration import (
    convergence_criterion,
    estimated_error,
)
from maddening.core._pow2_frame import pow2_exponent, pow2_frame, pow2_rescale
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

#: Why :func:`windowed_loss` and :func:`fit_multiple_shooting` refuse a
#: ``continuity_weight`` that is negative or not a finite number -- one
#: message for both, so the loss and its fitter cannot disagree on it.
_CONTINUITY_WEIGHT_WHY = (
    " A negative weight pays the fit to tear the trajectory apart at the "
    "window joins, and a NaN one compares False against everything, so it "
    "used to read as no penalty at all.")

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


def _x64_enabled() -> bool:
    """Whether ``jax_enable_x64`` is on: a float32 decomposition then comes
    from float32 leaves, and its precision remedy is float64 leaves rather
    than the x64 re-run (SYS-024)."""
    return bool(jax.config.read("jax_enable_x64"))


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


def _leaf_name(path) -> str:
    """``node.field`` spelling of a pytree path, for error messages."""
    parts = []
    for key in path:
        for attr in ("key", "idx", "name"):
            if hasattr(key, attr):
                parts.append(str(getattr(key, attr)))
                break
        else:
            parts.append(str(key))
    return ".".join(parts) or "<root>"


#: Leaves named per length in the ragged-axis refusal before it abbreviates.
_RAGGED_NAMES_SHOWN = 4


def _leading_len(tree, what: str = "observations") -> int:
    """The leading (time or window) axis every leaf of *tree* shares.

    Every leaf is checked, not only the first.  The callers index each
    leaf with ``dynamic_index_in_dim`` / ``dynamic_slice_in_dim``, which
    *clamp* an out-of-range start instead of failing, so a leaf shorter
    than the rest would be compared against its own last sample repeated
    -- a finite, biased loss, non-zero at the true parameters -- and a
    leaf longer than the first would have its tail ignored.  Both are
    refused here, naming the leaves and their lengths.

    Raises
    ------
    ValueError
        If *tree* has no leaves, if a leaf has no leading axis, or if the
        leaves' leading axes disagree.
    """
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    if not flat:
        raise ValueError(f"{what} pytree has no leaves")
    by_length: dict[int, list[str]] = {}
    for path, leaf in flat:
        shape = np.shape(leaf)
        if len(shape) == 0:
            raise ValueError(
                f"{what} leaf {_leaf_name(path)!r} is a scalar; every leaf "
                "needs a leading axis"
            )
        by_length.setdefault(int(shape[0]), []).append(_leaf_name(path))
    if len(by_length) > 1:
        groups = []
        for length, names in sorted(by_length.items()):
            shown = ", ".join(repr(n) for n in names[:_RAGGED_NAMES_SHOWN])
            more = len(names) - _RAGGED_NAMES_SHOWN
            groups.append(
                f"{length}: {shown}" + (f" (+{more} more)" if more > 0 else "")
            )
        raise ValueError(
            f"{what} leaves disagree on the leading axis ({'; '.join(groups)}); "
            "every leaf must have the same length, or a window would read a "
            "clamped sample instead of data"
        )
    (length,) = by_length
    return length


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
    on a different criterion from the one it converged under.  Both come
    from :func:`~maddening.core.coupling.acceleration.convergence_criterion`,
    which the report and the profiler read as well.
    """
    out = []
    for g in gm._coupling_groups:  # noqa: SLF001
        key = "+".join(sorted(g.nodes))
        thr, scale = convergence_criterion(g)
        out.append((f"coupling_{key}_residual",
                    f"coupling_{key}_amplification", thr, scale))
    return out


def _refuse_unmaskable_groups(gm, meta0, thresholds) -> None:
    """Raise if ``mask_unconverged=True`` has no verdict to read for a group.

    The mask reads each group's residual slot; a group without one --
    ``solver="fori"`` with ``diagnostics=False``, which records nothing
    -- was skipped by an ``if key in meta`` and so never masked, with
    no sign that the mask was inert for it.
    """
    meta0 = meta0 or {}
    for group, (res_key, _amp_key, _thr, _scale) in zip(
            gm._coupling_groups, thresholds):  # noqa: SLF001
        if res_key in meta0:
            continue
        why = ("it runs solver='fori' with diagnostics=False, which records "
               "no convergence verdict"
               if group.solver == "fori" and not group.diagnostics else
               "its state carries no residual slot to read")
        raise ValueError(
            f"mask_unconverged=True cannot mask coupling group "
            f"{sorted(group.nodes)}: {why}, so the mask would be silently "
            "inert for it.  Set diagnostics=True on the group, or use "
            "solver='ift' (whose verdict is always recorded), or pass "
            "mask_unconverged=False."
        )


def _warm_start_slots(gm, meta0) -> tuple[str, ...]:
    """The ``_meta`` slots a step reads as *state*, other than ``step_count``.

    A coupling group's predictor history (``coupling_<key>_pred_*``) and its
    IQN-IMVJ secant matrices (``coupling_<key>_V`` / ``_W``): the step
    starts its fixed-point iteration from them, so with a finite
    ``max_iterations`` they change the state it returns.  The diagnostics
    slots are outputs of a step and never read by the next one, and
    ``step_count`` is set from ``start_step``.  Named exactly, from the
    groups that own them, in sorted order (the carry's structure must not
    depend on dict order); only slots ``meta0`` actually holds.
    """
    if not meta0:
        return ()
    slots = []
    for group in gm._coupling_groups:  # noqa: SLF001
        key = "+".join(sorted(group.nodes))
        if group.acceleration == "iqn-imvj":
            slots += [f"coupling_{key}_V", f"coupling_{key}_W"]
        if group.predictor != "none":
            slots += [f"coupling_{key}_pred_count"]
            slots += [f"coupling_{key}_pred_{pi}" for pi in range(3)]
    return tuple(sorted(s for s in set(slots) if s in meta0))


def _sync_compiled(gm) -> None:
    """Compile ``gm`` if a run method would: dirty, never compiled, a
    changed static, or a ``node.params`` write the compiled step has not
    taken.  The fitters and :func:`windowed_loss` call it before they read
    ``gm.params``, so they start from the values ``gm.step`` would run."""
    gm._check_static_data_dirty()  # noqa: SLF001
    if gm._dirty or gm._compiled_step is None:  # noqa: SLF001
        gm.compile()


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
    start_step: Optional[int] = None,
) -> jnp.ndarray:
    """Windowed squared-error loss of a graph against data.

    Teacher-forced by default (every window restarts from the measured
    state); with ``window_states`` it is **multiple shooting**: window
    ``w`` restarts from the free state ``window_states[w]`` and a
    continuity penalty ties each window's end to the next window's start,
    so the fitted trajectory is a single continuous solution at the
    optimum instead of ``n_windows`` teacher-forced pieces.

    **Coupling warm starts are replayed, not reset.**  A coupling group
    with a ``predictor`` or ``acceleration="iqn-imvj"`` carries state the
    user state does not hold -- its predictor history and its IQN-IMVJ
    secant matrices -- and with a finite ``max_iterations`` it changes the
    state a step returns.  The observations cannot carry it, so it is
    replayed: window ``w`` starts from the warm starts window ``w - 1``
    ended with, and window ``0`` from the cold ones ``compile()`` and
    ``reset_state()`` leave.  At the parameters that generated a record
    taken from ``compile()`` or ``reset_state()`` each window therefore
    reproduces it exactly and the loss is exactly zero, whatever the
    predictor, the acceleration or ``max_iterations``.  The inherited warm
    starts enter a window with their gradient stopped, so the gradient of
    each window's loss is that of its own scan and gradients still cannot
    compound across windows; a window ``mask_unconverged`` drops hands the
    next one cold warm starts rather than its own, which may have
    diverged.  (0.4.0 development builds zeroed them at every window start:
    with ``max_iterations=1`` and a quadratic predictor the loss at the
    generating parameters was 1.7e-2 over 10-sample windows, and a fit
    started there walked 6.4% away.)  A record that began after the graph
    had stepped (``start_step > 0``) was made with warm starts that are not
    known, and the replay warns (``UserWarning``) that it cannot be exact.
    A ``"_meta"`` entry in ``observations`` or ``window_states`` is
    ignored.

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
        Every leaf must have the same leading length ``T``; leaves that
        disagree are refused, naming them (a shorter leaf would otherwise
        be read past its end, which JAX clamps to its last sample).
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
        Drop a window from the loss and from its gradient when any
        coupling group exited at ``max_iterations`` unconverged during it
        (the IFT gradient is unreliable there): it contributes exactly
        zero to both, including a window whose state diverged to inf or
        NaN, and under multiple shooting its continuity penalty goes with
        it.  The verdict is taken in a gradient-free forward pass before
        the differentiated one, so masking costs one extra simulation.
        (Until 0.4.0 the window's loss was multiplied by 0, and a
        diverged window made the gradient NaN.)  Reads each group's residual and
        amplification slots in ``_meta``, which ``solver="ift"`` always
        writes and ``solver="fori"`` writes only with
        ``diagnostics=True``: a group with no such slots would never be
        masked, so it is refused (``ValueError``) rather than silently
        skipped.  On a multi-rate graph a group's slots hold its most
        recent applied solve between the base steps it fires on.  On a
        sub-cycling group with ``waveform_iterations > 1`` the slots
        hold the *last* sweep's solve, so a window in which an earlier
        sweep exited at ``max_iterations`` unconverged while the last
        one converged is **kept** -- the same step on which
        ``strict_convergence=True``, which checks every sweep, raises,
        and on which ``coupling_diagnostics()`` reads
        ``iterations == max_iterations`` beside ``converged=True``.  To
        mask such a window, run the group at ``waveform_iterations=1``
        or with a ``max_iterations`` no sweep exhausts.  Must be
        a ``bool``: a truthy non-bool (``"no"``, ``1``, an array) used to
        turn masking on, so it is refused (``ValueError``).
    window_states : pytree, optional
        Multiple shooting: free initial **user states** per window, a
        pytree whose every leaf has leading axis
        ``n_windows = (T - 1) // window``
        (:func:`init_window_states` seeds it from the observations).
        Differentiate the loss with respect to these too and optimise
        them jointly with ``params`` (:func:`fit_multiple_shooting`).
    continuity_weight : float
        Weight of the continuity penalty ``Σ_w ||end_w - window_states[w+1]||²``
        (sum over every user-state leaf) under multiple shooting; ignored
        when ``window_states`` is ``None``.  A finite number ``>= 0``,
        refused otherwise (``ValueError``) as :func:`fit_multiple_shooting`
        refuses it: a negative or NaN weight used to read as ``0.0`` -- no
        penalty and no error -- because the penalty was guarded by
        ``continuity_weight > 0.0``.
    start_step : int, optional
        *Experimental, new in 0.4.0.*  The base step of the graph's
        multi-rate schedule at which sample ``0`` was recorded: ``0`` for a
        record that starts at ``compile()`` or ``reset_state()``, ``n`` for
        one that starts after ``gm.run(n)``.  Window ``w`` restarts at base
        step ``start_step + k * sample_every`` (``k`` its first sample), so
        each node fires on the sub-step it fired on when the record was
        made.  The observations are user state and cannot carry the
        recording's own step counter, so on a multi-rate graph this is the
        only way to know it: without it a record that began on an odd base
        step of a graph whose slowest node fires every second step is
        replayed on the wrong phase in every window, and the loss at the
        generating parameters is not zero (1.3e-2 for a ball and table).
        ``None`` (the default) assumes ``0`` and, on a multi-rate graph,
        warns (``UserWarning``) that it is assuming it.  On a single-rate
        graph it moves no schedule; on a graph whose coupling groups carry
        warm starts, a ``start_step > 0`` warns that the record's warm
        starts at sample ``0`` are not known (see above).

    Returns
    -------
    jnp.ndarray
        Scalar: the sum over windows and samples of the squared error of
        ``obs_fn`` outputs (plus the continuity penalty under multiple
        shooting).
    """
    # Before anything is built, as the fitters check theirs.  Both used to be
    # read as something else: a truthy non-bool turned masking on, and the
    # penalty's ``> 0.0`` guard read a negative or NaN weight as no penalty.
    _check_flag("mask_unconverged", mask_unconverged)
    _check_hyper("continuity_weight", continuity_weight, ge=0.0,
                 why=_CONTINUITY_WEIGHT_WHY)
    if start_step is not None:
        start_step = _check_count("start_step", start_step)
    # As every run method does first: a ``node.params`` write or a changed
    # static since the last compile makes the graph recompile, so this loss
    # traces the same model ``gm.step`` runs.  A caller that passed the live
    # ``gm.params`` meant the live values, which that recompile replaces.
    live = params is gm.params
    _sync_compiled(gm)
    if live:
        params = gm.params
    # The windows scan the graph step: a sharded node XLA miscompiles
    # inside a loop is refused here as in ``run_scan`` (MADD-ANO-068).
    gm._refuse_xla_loop_hazards("sysid.windowed_loss", scan=True)  # noqa: SLF001
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
    _refuse_unmaskable_groups(gm, meta0, thresholds)
    # ``step_count`` exists exactly when the graph is multi-rate: the phase
    # of its schedule is then part of the state a window restarts from.
    if start_step is None:
        if meta0 is not None and "step_count" in meta0:
            warnings.warn(
                "windowed_loss: the graph is multi-rate and the record's "
                "starting base step is not known, so it is assumed to be 0. "
                "A record that began elsewhere is replayed on the wrong phase "
                "of the schedule in every window. Pass start_step= (0 for a "
                "record taken from compile() or reset_state(), n after "
                "gm.run(n)) to say where it began.",
                UserWarning, stacklevel=2,
            )
        start_step = 0
    # The coupling warm starts are replayed, not reset: window ``w`` starts
    # from the predictor history and IQN-IMVJ secant matrices window
    # ``w - 1`` ended with, and window 0 from the cold ones ``compile()`` and
    # ``reset_state()`` leave.  Zeroing them at every window start (as
    # 0.4.0 development builds did) replayed a different scheme from the
    # one that made the record: with ``max_iterations=1`` and a quadratic
    # predictor the loss at the generating parameters was 1.7e-2, and a fit
    # started there walked 6.4% away.  They enter each window with their
    # gradient stopped, so gradients still cannot compound across windows.
    warm_slots = _warm_start_slots(gm, meta0)
    cold = ({} if meta0 is None
            else {slot: jnp.zeros_like(meta0[slot]) for slot in warm_slots})
    if warm_slots and start_step > 0:
        warnings.warn(
            f"windowed_loss: the record began at base step {start_step}, "
            "after the graph had stepped, so the coupling warm starts it was "
            "made with (predictor history, IQN-IMVJ secant matrices) are not "
            "known at its first sample. Window 0 starts them cold, as "
            "compile() and reset_state() leave them, and every later window "
            "from where the previous one left them, so the record is not "
            "replayed exactly: the loss at the generating parameters need "
            "not be zero. Record from compile() or reset_state() for an "
            "exact replay.",
            UserWarning, stacklevel=2,
        )

    def _state_from_obs(obs_k, k, warm):
        s = {nn: dict(fields) for nn, fields in obs_k.items()}
        if meta0 is not None:
            m = jax.tree.map(jnp.zeros_like, meta0)
            m.update(warm)
            if "step_count" in m:
                m["step_count"] = jnp.asarray(
                    start_step + k * sample_every, dtype=meta0["step_count"].dtype,
                )
            s[_META_KEY] = m
        return s

    def _next_warm(final, ok=None):
        """The warm starts the next window starts from: those this one ended
        with, gradient stopped -- or, for a window the mask dropped (whose
        state may have diverged), the cold ones."""
        warm = {slot: jax.lax.stop_gradient(final[_META_KEY][slot])
                for slot in warm_slots}
        if ok is not None:
            warm = {slot: jnp.where(ok, value, cold[slot])
                    for slot, value in warm.items()}
        return warm

    def _converged(state):
        ok = jnp.array(True)
        meta = state.get(_META_KEY, {})
        for key, amp_key, thr, scale in thresholds:
            if key in meta:
                amp = meta.get(amp_key, jnp.zeros_like(meta[key]))
                ok = ok & (estimated_error(meta[key], amp, scale) <= thr)
        return ok

    if window_states is not None:
        window_states = {k: v for k, v in window_states.items() if k != _META_KEY}
        n_ws = _leading_len(window_states, "window_states")
        if n_ws != n_windows:
            raise ValueError(
                f"window_states has leading axis {n_ws}, expected n_windows={n_windows}"
            )

    def _simulate(w, p, ws, warm):
        """Window ``w`` under params ``p``, from the warm starts ``warm``:
        ``(final state, converged, samples)``."""
        def _advance_one_sample(carry, _):
            def inner(c, _):
                s, ok = c
                s = step_fn(s, ext, p)
                return (s, ok & _converged(s)), None

            (state, ok), _ = jax.lax.scan(inner, carry, None, length=sample_every)
            user = {k: v for k, v in state.items() if k != _META_KEY}
            return (state, ok), user

        start = w * window
        if ws is None:
            obs_start = jax.tree.map(
                lambda x: jax.lax.dynamic_index_in_dim(x, start, keepdims=False),
                observations,
            )
        else:
            obs_start = jax.tree.map(
                lambda x: jax.lax.dynamic_index_in_dim(x, w, keepdims=False),
                ws,
            )
        state0 = _state_from_obs(obs_start, start, warm)
        (final, ok), sim = jax.lax.scan(
            _advance_one_sample, (state0, jnp.array(True)), None, length=window,
        )
        return final, ok, sim

    # A masked window is cut out of the loss *and* out of its gradient.
    # Multiplying its loss by 0 did neither reliably: a window whose
    # coupling diverged holds inf / NaN, and ``0 * inf`` is NaN.  The
    # forward loss came out right only because XLA's CPU backend happened
    # to rewrite the product into a select, while the backward pass
    # carried the zero cotangent through the window's non-finite
    # intermediates -- the square, the node updates, the coupling solve's
    # derivative -- and made the whole gradient NaN.  A zero cotangent does
    # not survive a multiplication by inf; only a *select* drops what it
    # does not pick.  So the verdict is taken first, in a gradient-free
    # pass, and then:
    #
    # * the masked window's inputs (the parameters, its free start under
    #   multiple shooting) enter through ``where(ok, x, stop_gradient(x))``,
    #   whose derivative selects -- whatever non-finite cotangent the
    #   window's own backward pass produces is dropped at its boundary;
    # * its outputs are replaced by their (gradient-free) targets before
    #   anything is computed from them, so its terms are exactly zero and
    #   nothing non-finite reaches ``obs_fn``, the square or the next
    #   window's free start;
    # * its loss is selected away, never multiplied.
    #
    # Values are unchanged wherever every window converges (``where`` picks
    # its first operand exactly), and a masked window contributes exactly
    # nothing.  The verdict pass costs one extra forward simulation.
    if mask_unconverged:
        frozen_p = jax.lax.stop_gradient(params)
        frozen_ws = (None if window_states is None
                     else jax.lax.stop_gradient(window_states))

        def _verdict(carry, _):
            w, warm = carry
            final, ok, _sim = _simulate(w, frozen_p, frozen_ws, warm)
            return (w + 1, _next_warm(final, ok)), ok

        _, window_ok = jax.lax.scan(
            _verdict, (jnp.int32(0), cold), None, length=n_windows)

    def _window(carry, _):
        w, warm = carry
        start = w * window
        p, ws = params, window_states
        if mask_unconverged:
            ok = window_ok[w]

            def gate(x):
                return jnp.where(ok, x, jax.lax.stop_gradient(x))

            p = jax.tree.map(gate, params)
            if ws is not None:
                ws = jax.tree.map(gate, ws)
        final, _ok, sim = _simulate(w, p, ws, warm)
        truth = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, start + 1, window),
            observations,
        )
        if mask_unconverged:
            sim = jax.tree.map(
                lambda a, b: jnp.where(ok, a, jax.lax.stop_gradient(b)), sim, truth,
            )
        sq = jax.tree.map(
            lambda a, b: jnp.sum((a - b) ** 2), obs_fn(sim), obs_fn(truth),
        )
        # `sum` is typed as returning `int` for its empty-sequence start
        # value; every leaf here is an Array, so the result is one.
        loss_w: Any = sum(jax.tree.leaves(sq))
        if mask_unconverged:
            loss_w = jnp.where(ok, loss_w, jnp.zeros_like(loss_w))
        if window_states is not None and continuity_weight > 0.0:
            # Tie this window's end to the next window's free start
            # (no penalty after the last window).
            nxt = jax.tree.map(
                lambda x: jax.lax.dynamic_index_in_dim(
                    x, jnp.minimum(w + 1, n_windows - 1), keepdims=False),
                window_states,
            )
            end_user = {k: v for k, v in final.items() if k != _META_KEY}
            if mask_unconverged:
                # A masked window's end ties nothing: its penalty is
                # dropped with it, by the same substitution.
                end_user = jax.tree.map(
                    lambda a, b: jnp.where(ok, a, jax.lax.stop_gradient(b)),
                    end_user, nxt,
                )
            gap = jax.tree.map(lambda a, b: jnp.sum((a - b) ** 2), end_user, nxt)
            pen = sum(jax.tree.leaves(gap))
            loss_w = loss_w + continuity_weight * pen * (w < n_windows - 1)
        return (w + 1, _next_warm(final, window_ok[w] if mask_unconverged else None)), loss_w

    _, losses = jax.lax.scan(_window, (jnp.int32(0), cold), None, length=n_windows)
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
    number and ``inf`` under exactly that rescaling.  That holds while
    ``F`` can be formed: once the cutoff ``rank_rtol * max(eigvals)``
    falls below ``2 * m * tiny`` of the precision (``m`` residual rows),
    the products the smallest directions are built from flush to zero and
    the verdict moves with the scale -- in float32, a residual of size
    ``1e-3`` divided by ``noise_std=1e15`` -- and :func:`fim` says so with
    a :class:`~maddening.warnings.PrecisionLimitWarning` (and
    :class:`FIMCore` with ``precision_limited``) rather than reporting it.
    The same is said when every product flushed and ``F`` came out exactly
    zero although ``J`` has a nonzero entry (``|J|`` below about
    ``sqrt(tiny)``, ``1e-19`` in float32): the cutoff is then ``0`` and
    ``rank`` reads ``0``, which earlier 0.4.0 development builds reported
    as a fact, with every ``crb`` ``inf`` and no warning (MADD-ANO-176).

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

    def __str__(self) -> str:
        """A short human summary: rank, conditioning, the weakest
        direction and each parameter's Cramér–Rao bound.  ``repr`` is the
        dataclass's own, field by field."""
        n = len(self.param_names)
        undetermined = n - int(self.rank)
        lines = [
            f"FIMReport: rank {int(self.rank)} of {n} parameter{'s' * (n != 1)}"
            + (f" ({undetermined} undetermined direction{'s' * (undetermined != 1)})"
               if undetermined else " (all determined)")
            + f"; cond {_summary_number(self.cond)}"
        ]
        if n:
            try:
                name, weight = self.least_identifiable()
                lines.append(f"  least identifiable: {name} "
                             f"(weight {weight:.3g} in the weakest direction)")
            except Exception:   # noqa: BLE001 - a summary never raises
                pass
            try:
                crb = np.asarray(self.crb).reshape(-1)
            except Exception:   # noqa: BLE001
                crb = None
            if crb is not None and crb.size == n:
                lines.append("  Cramér–Rao bound on each variance (inf: not identifiable):")
                width = max(len(p) for p in self.param_names)
                for p, b in zip(self.param_names, crb):
                    lines.append(f"    {p.ljust(width)}  {_summary_number(b)}")
        for label, names in (("zero_scaled", self.zero_scaled),
                             ("value_scaled", self.value_scaled),
                             ("integer_excluded", self.integer_excluded)):
            if names:
                lines.append(f"  {label}: {', '.join(names)}")
        return "\n".join(lines)


def _summary_number(x) -> str:
    """``x`` for a human summary: ``.4g``, ``inf`` / ``nan`` spelled out."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:.4g}"


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


def _describe_subtree(treedef) -> str:
    """``a leaf`` / ``None`` / ``an empty dict`` / ``a dict holding 3 leaves``."""
    data = treedef.node_data()
    if data is None:
        return "a leaf"
    if data[0] is type(None):
        return "None"
    kind = data[0].__name__
    n = treedef.num_leaves
    if n == 0:
        return f"an empty {kind}" if not treedef.children() else f"a {kind} holding no leaf"
    return f"a {kind} holding {n} lea{'f' if n == 1 else 'ves'}"


def _structure_difference(params_def, mask_def, path: str = "") -> Optional[str]:
    """Where two pytree structures first differ, in words, or ``None``.

    Walks the two ``PyTreeDef`` s node by node, so it names a difference
    the leaf paths cannot show: a container that holds no leaf (the empty
    ``"mappings": {}`` of a graph with no interface mappings), or a
    ``tuple`` where the params have a ``list``.  Paths are spelled as
    ``jax.tree_util.keystr`` spells dict keys and sequence indices.
    """
    p_data, m_data = params_def.node_data(), mask_def.node_data()
    where = f"params{path}" if path else "the root"
    if p_data is None and m_data is None:
        return None
    if p_data is None or m_data is None or p_data[0] is not m_data[0]:
        return (f"at {where}, params has {_describe_subtree(params_def)} and mask "
                f"has {_describe_subtree(mask_def)}")
    p_children, m_children = params_def.children(), mask_def.children()
    if p_data[0] is dict:
        p_keys, m_keys = list(p_data[1]), list(m_data[1])
        for key, child in zip(p_keys, p_children):
            if key not in m_keys:
                return (f"params{path}[{key!r}] is {_describe_subtree(child)}, and "
                        "mask has no such key")
        for key, child in zip(m_keys, m_children):
            if key not in p_keys:
                return (f"mask{path}[{key!r}] is {_describe_subtree(child)}, and "
                        "params has no such key")
        labels = [f"[{k!r}]" for k in p_keys]
    else:
        if p_data[1] != m_data[1] or len(p_children) != len(m_children):
            return (f"at {where}, params has {_describe_subtree(params_def)} and mask "
                    f"has {_describe_subtree(mask_def)}")
        labels = [f"[{i}]" for i in range(len(p_children))]
    for label, p_child, m_child in zip(labels, p_children, m_children):
        found = _structure_difference(p_child, m_child, path + label)
        if found is not None:
            return found
    return None


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

    def _listed(paths):
        return f"{paths[:6]}{' ...' if len(paths) > 6 else ''}"

    if p_paths != m_paths:
        where = ""
        for i in range(max(len(p_paths), len(m_paths))):
            if p_paths[i:i + 1] != m_paths[i:i + 1]:
                where = (
                    f" They first differ at leaf {i}: params has "
                    f"{p_paths[i] if i < len(p_paths) else '<no such leaf>'}, "
                    f"mask has {m_paths[i] if i < len(m_paths) else '<no such leaf>'}."
                )
                break
        leaves = (f" params has {len(p_paths)} leaves {_listed(p_paths)}; mask has "
                  f"{len(m_paths)} leaves {_listed(m_paths)}.")
    else:
        # Same leaves, different structure: a container that holds no leaf,
        # or a container of another type.  Two identical leaf lists and no
        # "first differ" clause was all this used to say.
        found = _structure_difference(params_def, mask_def)
        where = (f" They have the same {len(p_paths)} leaves {_listed(p_paths)}, and "
                 f"differ where no leaf is: {found}.")
        leaves = ""
    raise ValueError(
        "mask must have the same tree structure as params, key for key. "
        "The flags are read in flatten order, so a mask with the right "
        "number of leaves and different keys is not refused by a count: it "
        "quietly fits whichever parameter sits at that position."
        f"{where}{leaves} Build the "
        "mask from the params tree itself -- jax.tree.map over params, or "
        "gm.trainable_mask(params) with the leaves you do not want dropped "
        "to False."
    )


def _is_bool_flag(flag) -> bool:
    """A ``bool``, a NumPy ``bool_`` or a 0-d boolean array."""
    if isinstance(flag, (bool, np.bool_)):
        return True
    return (isinstance(flag, (np.ndarray, jax.Array)) and flag.ndim == 0
            and flag.dtype == np.bool_)


def _refuse_non_bool_flags(mask, flags) -> None:
    """Refuse a caller's mask leaf that is not a bool (an explicit
    ``mask=`` to a fitter, :func:`fim` or :func:`fim_core`).

    Every reader decides a leaf with ``bool(flag)``, which reads any object
    by its truthiness: a leaf of ``"False"`` -- a non-empty string --
    selected its parameter, and the fit moved it.  ``0`` and ``1`` read as
    the caller meant, but they are no more a flag than ``"False"`` is, and
    a mask built by arithmetic can hold ``2`` or ``-1`` as easily; so only a
    bool is a flag.
    """
    bad = [(path, flag) for (path, _), flag
           in zip(jax.tree_util.tree_flatten_with_path(mask)[0], flags)
           if not _is_bool_flag(flag)]
    if not bad:
        return
    listed = ", ".join(f"mask{jax.tree_util.keystr(path)} = {flag!r} "
                       f"({type(flag).__name__})" for path, flag in bad[:6])
    more = f" (+{len(bad) - 6} more)" if len(bad) > 6 else ""
    raise ValueError(
        f"every mask leaf must be a bool; got {listed}{more}. A leaf is read "
        "by its truthiness, so a string 'False' would select its parameter. "
        "Build the mask with True / False, e.g. jax.tree.map(lambda _: False, "
        "params) and then set the leaves to fit to True.")


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
    _refuse_non_bool_flags(mask, flags)
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
    _refuse_non_bool_flags(mask, flags)
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


def _compile_model(fn):
    """``jax.jit(fn)`` for a function of the physical parameters.

    Every program a fitter runs the model through -- :func:`fit_lm`'s
    residual and its Jacobian, the loss and gradient of :func:`fit` and
    :func:`fit_multiple_shooting` -- is compiled here and takes the physical
    ``params`` pytree as its first argument, as arrays: the tree
    :class:`_PhysicalMap` produced.  Nothing of the optimiser's coordinates,
    of a transform or of the mask is inside such a program, so what it
    evaluates is what it was handed.  One name, so that a test can stand
    between a fitter and its model and read which arrays each evaluation
    received (``tests/property/test_sysid_one_evaluation.py``).
    """
    return jax.jit(fn)


@dataclass(frozen=True)
class _FittedLeaf:
    """One leaf of ``params`` the fit moves: its position among the leaves,
    the slice of ``theta`` that holds its coordinates, and what maps them."""

    leaf: int
    start: int
    stop: int
    shape: tuple
    dtype: Any
    spec: Any

    @property
    def coordinates(self) -> slice:
        return slice(self.start, self.stop)


#: How many compiled maps :func:`_compiled_physical_map` keeps: one per
#: distinct set of fitted leaves' specs, shapes and dtypes.
_PHYSICAL_MAP_CACHE_SIZE = 64


@functools.lru_cache(maxsize=_PHYSICAL_MAP_CACHE_SIZE)
def _compiled_physical_map(signature: tuple, theta_dtype):
    """``(values, slopes, bends)``: the compiled ``theta -> physical`` map of
    :class:`_PhysicalMap` for fitted leaves of ``signature`` -- a tuple of
    ``(spec, shape, dtype)`` in ``theta`` order -- and its first and second
    derivatives.

    The start (``theta0``) and the values that went in (``given``) are
    arguments, not constants, so the programs depend on the leaves' specs,
    shapes and dtypes alone and a second fit of the same parameters reuses
    them: there is one compiled map per such signature in the process, not
    one per fit.

    ``values(theta, theta0, given)``: per fitted leaf, entry by entry, the
    value that went in where the coordinate is where it started -- ``==`` at
    the leaf's dtype: the claim is bitwise identity, and an entry an
    optimiser moved by one ulp *was* fitted -- and the leaf's transform of
    the coordinate anywhere else.  ``slopes(theta)`` and ``bends(theta)``
    are the transform's own first and second derivative there, entry by
    entry (every transform is elementwise), whatever ``values`` selected.
    """
    spans, offset = [], 0
    for _, shape, _ in signature:
        n = int(math.prod(shape))
        spans.append(slice(offset, offset + n))
        offset += n

    def values(theta, theta0, given):
        out = []
        for (spec, shape, dtype), span, leaf in zip(signature, spans, given):
            u = theta[span].reshape(shape).astype(dtype)
            u0 = theta0[span].reshape(shape).astype(dtype)
            out.append(jnp.where(u == u0, leaf, spec.to_constrained(u)))
        return tuple(out)

    def derivative(theta, order: int):
        parts = []
        for (spec, _, dtype), span in zip(signature, spans):
            u = theta[span].astype(dtype)

            def first(x, spec=spec):
                return jax.jvp(spec.to_constrained, (x,), (jnp.ones_like(x),))[1]

            value = first(u) if order == 1 else jax.jvp(first, (u,), (jnp.ones_like(u),))[1]
            parts.append(value.astype(theta.dtype))
        return jnp.concatenate(parts) if parts else jnp.zeros((0,), theta.dtype)

    return (jax.jit(values), jax.jit(functools.partial(derivative, order=1)),
            jax.jit(functools.partial(derivative, order=2)))


class _PhysicalMap:
    """``theta -> physical params``: the one evaluation of that map a fit makes.

    The optimisers carry ``theta``, the ``ravel_pytree`` entries of the
    unconstrained tree that the resolved ``mask`` selects.  Every physical
    value a fit uses comes from :meth:`params`: the tree its objective is
    evaluated on, the tree its ``callback`` and observers receive, and the
    tree it returns are that method's output, the arrays of one compiled
    function (:func:`_compiled_physical_map`).  The model-side programs
    (:func:`_compile_model`) take that tree as an argument and never see
    ``theta``.

    Until 0.4.0's fix the map was evaluated twice: inside each fitter's
    jitted objective, and again eagerly (``GraphManager.constrain``) for the
    tree returned.  XLA contracts ``lo + (hi - lo) * sigmoid(u)`` into a
    fused multiply-add and eager JAX does not, so the two landed on
    different floats -- an ulp apart at ordinary bounds, 1.3% apart for a
    float32 ``2.0`` under ``logit`` bounds ``(-1e6, 1e6)`` -- and
    ``best_loss`` was the loss of parameters other than the ones returned.
    A leaf the fit did not move was returned as given while the objective
    ran its ``constrain(unconstrain(p))`` round trip whenever the eager
    round trip happened to be exact: left out by ``mask=`` at ``2.0`` under
    those bounds it was run at 2.0266, and the stiffness fitted beside it
    came back 30.06 for a truth of 30, ``converged=True`` (MADD-ANO-178).
    Two evaluations cannot be made to agree in the last bit across
    compilers; there is now one.

    What the one map returns, entry by entry:

    * a leaf the mask leaves out, or whose spec is not trainable, is **the
      object that went in**: it is never raveled, transformed or cast;
    * an entry of a fitted leaf whose coordinate is where it started, bit
      for bit at the leaf's dtype, is the value that went in, selected, not
      recomputed -- so a fit that takes no step evaluates exactly its
      start, and ``constrain(unconstrain(p))`` (an ulp off for a ``log``
      leaf) is never what a leaf nobody moved is run at;
    * any other entry is the leaf's transform of its coordinate.

    The derivative the optimisers need is the map's own, taken apart from
    its value: :meth:`slope` is ``dp/dtheta`` entry by entry (every
    transform is elementwise) and :meth:`bend` the second derivative.  The
    model's derivatives are taken with respect to the physical values and
    chained with them, so a derivative can never move the point it was
    taken at.
    """

    def __init__(self, gm, start: dict, flat_u, idx) -> None:
        specs = _resolve_specs(start, gm.param_specs())
        leaves, self.treedef = jax.tree.flatten(start)
        self._start_leaves = list(leaves)
        #: The dtype of ``theta`` (``ravel_pytree`` promotes across leaves).
        self.dtype = flat_u.dtype
        index = np.asarray(idx, dtype=np.intp)
        self._theta0 = flat_u[index]
        chosen = np.zeros(int(flat_u.size), dtype=bool)
        chosen[index] = True
        fitted: list[_FittedLeaf] = []
        offset = position = 0
        for i, (leaf, spec) in enumerate(zip(leaves, specs)):
            n = _leaf_size(leaf)
            picked = chosen[offset:offset + n]
            if n and picked.all():
                fitted.append(_FittedLeaf(i, position, position + n, tuple(np.shape(leaf)),
                                          np.dtype(jnp.result_type(leaf)), spec))
                position += n
            elif picked.any():
                # A mask flags whole leaves (``_masked_indices``).
                raise ValueError(
                    f"the trainable entries select part of params leaf {i}; a fit "
                    "moves whole leaves")
            offset += n
        #: The leaves the fit moves, in ``theta`` order.
        self.fitted = tuple(fitted)
        #: How many coordinates ``theta`` has.
        self.n = position
        self._fitted_index = [f.leaf for f in fitted]
        taken = set(self._fitted_index)
        self._other_index = [i for i in range(len(leaves)) if i not in taken]
        # The values that went in, at the leaves' own dtypes: what the map
        # selects for an entry whose coordinate has not moved.
        self._given = tuple(jnp.asarray(leaves[f.leaf], f.dtype) for f in fitted)
        self._values, self._slopes, self._bends = _compiled_physical_map(
            tuple((f.spec, f.shape, f.dtype) for f in fitted), np.dtype(self.dtype))

    # -- the one map -------------------------------------------------------

    def params(self, theta) -> dict:
        """The physical ``params`` pytree at ``theta``: what the model is
        evaluated on."""
        leaves = list(self._start_leaves)
        for f, value in zip(self.fitted, self._values(theta, self._theta0, self._given)):
            leaves[f.leaf] = value
        return jax.tree.unflatten(self.treedef, leaves)

    def returned(self, theta, params: dict) -> dict:
        """``params`` (:meth:`params` of ``theta``) as a fit hands it out:
        fresh containers, and every fitted leaf no coordinate of which moved
        as the object that went in.  Its entries are already that object's
        values, selected by the map; this gives back the object
        itself, as :class:`FitResult` documents for a leaf no step moved."""
        leaves = jax.tree.leaves(params)
        here, base = np.asarray(theta), np.asarray(self._theta0)
        for f in self.fitted:
            moved = here[f.coordinates].astype(f.dtype) != base[f.coordinates].astype(f.dtype)
            if not bool(moved.any()):
                leaves[f.leaf] = self._start_leaves[f.leaf]
        return jax.tree.unflatten(self.treedef, leaves)

    def physical(self, params: dict) -> np.ndarray:
        """The trainable entries of ``params`` in ``theta`` order, float64."""
        leaves = jax.tree.leaves(params)
        parts = [np.asarray(leaves[f.leaf]).astype(f.dtype).astype(np.float64).reshape(-1)
                 for f in self.fitted]
        return np.concatenate(parts) if parts else np.zeros(0)

    def slope(self, theta):
        """``dp/dtheta`` of every trainable entry (``theta``'s dtype)."""
        return self._slopes(theta)

    def bend(self, theta):
        """``d²p/dtheta²`` of every trainable entry (``theta``'s dtype)."""
        return self._bends(theta)

    # -- for the model-side programs (traceable) ---------------------------

    def split(self, params: dict) -> tuple[tuple, tuple]:
        """``(fitted, others)``: the leaves a fit differentiates, and the rest."""
        leaves = jax.tree.leaves(params)
        return (tuple(leaves[i] for i in self._fitted_index),
                tuple(leaves[i] for i in self._other_index))

    def join(self, fitted, others) -> dict:
        """The inverse of :meth:`split`."""
        leaves: list = [None] * (len(self._fitted_index) + len(self._other_index))
        for i, leaf in zip(self._fitted_index, fitted):
            leaves[i] = leaf
        for i, leaf in zip(self._other_index, others):
            leaves[i] = leaf
        return jax.tree.unflatten(self.treedef, leaves)

    def ravel(self, fitted):
        """Fitted leaves (or cotangents shaped like them) as one vector in
        ``theta`` order and dtype."""
        parts = [jnp.ravel(x).astype(self.dtype) for x in fitted]
        return jnp.concatenate(parts) if parts else jnp.zeros((0,), self.dtype)

    def unravel(self, vector) -> tuple:
        """A vector in ``theta`` order as tangents shaped like the fitted leaves."""
        return tuple(vector[f.coordinates].reshape(f.shape).astype(f.dtype)
                     for f in self.fitted)

    def names(self) -> list[str]:
        """One human name per coordinate of ``theta`` (:func:`_leaf_location`)."""
        paths = [path for path, _ in jax.tree_util.tree_flatten_with_path(
            jax.tree.unflatten(self.treedef, self._start_leaves))[0]]
        out = []
        for f in self.fitted:
            where = _leaf_location(paths[f.leaf])
            n = f.stop - f.start
            if n == 1:
                out.append(where)
            else:
                out.extend(f"{where}{list(ix)}" for ix in np.ndindex(*f.shape))
        return out


def _model_hvp(grad_p, pmap: _PhysicalMap, theta, params: dict, extra: tuple, V):
    """``H V`` for ``H`` the Hessian in ``theta`` of a fitter's objective,
    from the model's own derivatives in the physical parameters.

    ``grad_p(params, *extra)`` is the gradient with respect to the trainable
    entries of ``params`` (raveled, :meth:`_PhysicalMap.ravel`) by the
    fitter's own compiled loss-and-gradient, so the products reuse its trace.
    With ``s = dp/dtheta`` and ``b = d²p/dtheta²`` (the map is elementwise),

        ``H_theta v = s * (H_p (s * v)) + b * g_p * v``

    and ``H_p w`` is one forward-over-reverse product per column, evaluated
    one after another (``lax.map``) so the memory is one product's.  Every
    model evaluation is at exactly ``params``.
    """
    s = np.asarray(pmap.slope(theta), dtype=np.float64)
    b = np.asarray(pmap.bend(theta), dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    tangents = jnp.asarray((s[:, None] * V).T, dtype=pmap.dtype)

    def products(p, ex, ws):
        fitted, others = pmap.split(p)

        def gradient(leaves):
            return grad_p(pmap.join(leaves, others), *ex)

        return gradient(fitted), jax.lax.map(
            lambda w: jax.jvp(gradient, (fitted,), (pmap.unravel(w),))[1], ws)

    g, HW = jax.jit(products)(params, extra, tangents)
    return (s[:, None] * np.asarray(HW, dtype=np.float64).T
            + (b * np.asarray(g, dtype=np.float64))[:, None] * V)


def _along_the_shorter(spec, u, v, box, interval, snap: float = 0.0):
    """The coordinates a step ``v`` solved on the linear model at ``u``
    stands for, for a ``log`` / ``logit`` leaf: the step read along the
    transform's curve (``u + v``) or along its tangent
    (:meth:`ParamSpec._tangent_shift`), whichever moves the value less.

    ``box`` is the tangent box the solver clipped ``v`` to
    (:meth:`ParamSpec._tangent_box`) and ``interval`` the coordinate's
    range.  A ``v`` on the box is a tangent step that reached the edge of
    the range, so its tangent reading is that edge exactly.  A result
    within ``snap`` of an end *is* that end: read along the curve, a step
    for the edge lands short of it by its own square, so a coordinate
    heading there closes on the end without arriving -- the run converges
    a step tolerance short of it -- and one that is not exactly on the end
    is neither held as an active edge nor named as having ended on one
    (:class:`_CoordinateBounds` gives the default step tolerance's width).
    """
    (v_lo, v_hi), (u_lo, u_hi) = box, interval
    v = np.clip(v, v_lo, v_hi)
    tangent = np.asarray(spec._tangent_shift(u, v))  # noqa: SLF001
    tangent = np.where(v >= v_hi, u_hi - u, np.where(v <= v_lo, u_lo - u, tangent))
    shorter = np.abs(tangent) < np.abs(v)
    moved = u + np.where(shorter, tangent, v)
    # On an edge exactly: ``u + (u_hi - u)`` need not round to ``u_hi``.
    moved = np.where((v >= v_hi) & (np.abs(u_hi - u) <= np.abs(v)), u_hi, moved)
    moved = np.where((v <= v_lo) & (np.abs(u_lo - u) <= np.abs(v)), u_lo, moved)
    moved = np.where(np.abs(moved - u_hi) <= snap, u_hi, moved)
    return np.where(np.abs(moved - u_lo) <= snap, u_lo, moved)


class _CoordinateBounds:
    """The bounds of each optimiser coordinate ``theta``, the projection
    onto them, and the reading of a step solved on the linear model.

    A trainable leaf whose :class:`~maddening.core.params.ParamSpec` has
    bounds and ``transform=None`` is optimised in its own (physical)
    coordinate, and ``constrain`` *clips* it.  The clip's derivative is 0
    strictly outside the bounds, so a coordinate one step carried past its
    bound had no gradient back and stayed clipped for the rest of the run:
    a spring's damping, started at 4 against a truth of 0.05, landed on 0
    after one Levenberg-Marquardt step and stayed there, and ``fit_lm``
    called it converged because the physical change of every later step was
    0.  So every fitter projects its coordinate back onto the bounds after
    each update; a coordinate on its bound has the one-sided derivative
    into the range, and a step that would leave the range again moves it by
    nothing.  A ``log`` / ``logit`` coordinate is bounded the same way, to
    the range where its transform still resolves the distance to its bound
    (:meth:`ParamSpec._optimiser_interval`: ``sqrt(eps)`` of the bounds'
    size inside each): past that the transform is flat to the working
    precision.  Inside the bounds the projection is the identity, bit for
    bit.

    **A ``log`` / ``logit`` coordinate is stepped on its tangent**
    (:meth:`tangent_frame`, :meth:`from_tangent`).  A Marquardt or
    Gauss-Newton step is solved on the residual's linearisation in
    ``theta``, and for a transformed coordinate that linearisation is the
    transform's tangent: where the transform is flat, the step that should
    move the value by 0.1 is ``0.1 / (dp/du)`` long, and taken along the
    curve it lands on the opposite edge of the range -- every damped retry
    with it.  That is how a ``logit`` damping carried to the edge of
    ``(0.5, 2)`` by an early overshoot stayed there, ``converged=False``,
    for a truth of 1.9 (MADD-ANO-104).  So the solver is given, for such a
    coordinate, the origin 0 and the box of steps over which the *tangent*
    stays inside the range; its answer ``v`` is then read along the curve
    (``u + v``) or along the tangent (the ``du`` whose value is the tangent
    step's), whichever moves the value less.  The two agree to first order;
    the tangent is the shorter when a step leaves a flat end (it is the
    step the same parameter under ``transform=None`` would take), and the
    curve when a step approaches one (it never reaches the bound).  No
    tolerance is involved, and an identity coordinate is untouched, bit for
    bit.
    """

    def __init__(self, gm, start: dict, idx, dtype) -> None:
        specs = _resolve_specs(start, gm.param_specs())
        los, his, p_los, p_his, transformed = [], [], [], [], []
        curved = []
        offset = 0
        for leaf, spec in zip(jax.tree.leaves(start), specs):
            n = _leaf_size(leaf)
            lo, hi = -np.inf, np.inf
            p_lo, p_hi = -np.inf, np.inf
            transformed.append(np.full(n, spec.transform is not None))
            if spec.trainable and _is_differentiable(leaf):
                # A ``log`` / ``logit`` coordinate is unbounded in principle,
                # but past the point where ``constrain`` clamps it (``exp``
                # at its floor or overflowing, the sigmoid at the edge of
                # the representable interior) its derivative is 0, and
                # short of that the transform has stopped resolving the
                # distance to the bound: a logit-bounded damping driven
                # there sat at 1.999998 of (0.5, 2) with ``converged=True``.
                # Same projection, onto the range the transform resolves.
                leaf_dtype = np.dtype(jnp.result_type(leaf))
                lo, hi = spec._optimiser_interval(leaf_dtype)  # noqa: SLF001
                if spec.transform is not None:
                    # On the leaf's own grid, so that a coordinate projected
                    # onto an end *is* that end: the tangent box there is then
                    # exactly 0 wide on that side, which is what holds an
                    # active edge out of the coupled step (an end half an ulp
                    # away left the coordinate free, the solve moved the
                    # others for a step it could not take, and a float32 fit
                    # crawled along the edge for its whole budget).
                    lo, hi = (float(np.asarray(x, leaf_dtype)) for x in (lo, hi))
                    # Within ``2**4 * sqrt(eps)`` of an end the value is that
                    # end's to ``2**4`` spacings of the transform (the slope
                    # there is ``sqrt(eps)`` of the bounds' size, a spacing
                    # ``eps`` of it) -- the default step tolerance, inside
                    # which ``fit_lm`` calls a step nothing: the coordinate
                    # is on the edge.
                    snap = _STEP_TOL_ULPS * math.sqrt(float(np.finfo(leaf_dtype).eps))
                    curved.append((offset, offset + n, spec, (lo, hi), snap))
                p_lo, p_hi = _physical_edges(spec, leaf_dtype, (lo, hi))
            los.append(np.full(n, lo))
            his.append(np.full(n, hi))
            p_los.append(np.full(n, p_lo))
            p_his.append(np.full(n, p_hi))
            offset += n
        lo_all = np.concatenate(los) if los else np.zeros(0)
        hi_all = np.concatenate(his) if his else np.zeros(0)
        index = np.asarray(idx, dtype=np.intp)
        lo_np, hi_np = lo_all[index], hi_all[index]
        #: The physical values the edges of each coordinate's range map to.
        self.p_lo = (np.concatenate(p_los) if p_los else np.zeros(0))[index]
        self.p_hi = (np.concatenate(p_his) if p_his else np.zeros(0))[index]
        #: Whether each coordinate is a ``log`` / ``logit`` one (dimensionless,
        #: a relative change of the parameter) rather than the parameter in
        #: its own units: the identifiability guard measures the latter
        #: relative to the parameter's value (:func:`_relative_scale`).
        self.transformed = (np.concatenate(transformed) if transformed
                            else np.zeros(0, dtype=bool))[index]
        #: Whether any coordinate is bounded at all; when not, every method
        #: here is the identity and costs nothing.
        self.active = bool(np.isfinite(lo_np).any() or np.isfinite(hi_np).any())
        # At the coordinates' own dtype, rounded as ``constrain``'s clip
        # rounds a Python-float bound, so the two agree on where the bound is.
        self.lo = jnp.asarray(lo_np, dtype)
        self.hi = jnp.asarray(hi_np, dtype)
        # The ``log`` / ``logit`` leaves among the coordinates, as
        # ``(positions in theta, spec, interval)``.
        position = np.full(lo_all.size, -1, dtype=np.intp)
        position[index] = np.arange(index.size)
        self._curved = []
        for a, b, spec, interval, snap in curved:
            where = position[a:b]
            where = where[where >= 0]
            if where.size:
                self._curved.append((where, spec, interval, snap))

    def project(self, theta):
        return jnp.clip(theta, self.lo, self.hi) if self.active else theta

    def _tangent_boxes(self, u: np.ndarray, dtype):
        """Per ``log`` / ``logit`` leaf, the tangent box at ``u`` rounded to
        the solver's ``dtype`` (what the solver clips to)."""
        boxes = []
        for where, spec, (u_lo, u_hi), _ in self._curved:
            v_lo, v_hi = spec._tangent_box(u[where], u_lo, u_hi)  # noqa: SLF001
            # A ``log`` value far below its ceiling has a box beyond the
            # solver's dtype: no bound at all on that side.
            with np.errstate(over="ignore"):
                boxes.append((np.asarray(v_lo, dtype).astype(np.float64),
                              np.asarray(v_hi, dtype).astype(np.float64)))
        return boxes

    def tangent_frame(self, theta):
        """``(origin, lo, hi)`` to solve a step from ``theta`` in: the
        coordinates and their bounds as they are for an identity
        coordinate, and for a ``log`` / ``logit`` one the origin 0 and the
        box of steps over which the transform's tangent stays inside the
        range.  A solver's answer in this frame is read back by
        :meth:`from_tangent`."""
        if not self._curved:
            return theta, self.lo, self.hi
        dtype = np.dtype(theta.dtype)
        u = np.asarray(theta, dtype=np.float64)
        origin = u.copy()
        lo = np.asarray(self.lo, dtype=np.float64).copy()
        hi = np.asarray(self.hi, dtype=np.float64).copy()
        for (where, _, _, _), (v_lo, v_hi) in zip(self._curved, self._tangent_boxes(u, dtype)):
            origin[where], lo[where], hi[where] = 0.0, v_lo, v_hi
        return (jnp.asarray(origin, dtype), jnp.asarray(lo, dtype), jnp.asarray(hi, dtype))

    def from_tangent(self, theta, answer):
        """The coordinates a solver's ``answer`` in :meth:`tangent_frame` of
        ``theta`` stands for: itself for an identity coordinate, and for a
        ``log`` / ``logit`` one the shorter of the curve's and the tangent's
        readings of that step (:func:`_along_the_shorter`), inside the
        range."""
        if not self._curved:
            return answer
        dtype = np.dtype(theta.dtype)
        u = np.asarray(theta, dtype=np.float64)
        out = np.asarray(answer, dtype=np.float64).copy()
        for (where, spec, interval, snap), box in zip(self._curved,
                                                      self._tangent_boxes(u, dtype)):
            out[where] = _along_the_shorter(spec, u[where], out[where], box, interval, snap)
        return self.project(jnp.asarray(out, dtype))

    def on_an_edge(self, theta) -> np.ndarray:
        """Per coordinate, whether a ``log`` / ``logit`` one sits on (or
        beyond) an edge of the range its transform resolves."""
        th = np.asarray(theta)
        on = (th <= np.asarray(self.lo)) | (th >= np.asarray(self.hi))
        return on & self.transformed

    def inward_descent(self, theta, g, physical=None, rel_tol: Any = 0.0) -> bool:
        """Whether a coordinate on its bound could lower the loss by moving
        into the range: ``g`` (the gradient, with the one-sided derivative on
        the bound) pointing inward.  Such a point is not a constrained
        stationary point, whatever the size of the step proposed there.

        "On its bound" is also read physically, when ``physical`` (the
        coordinates' physical values) is given: within ``rel_tol`` of the
        value an edge maps to.  A ``logit`` coordinate near the edge of its
        range moves its value by almost nothing for a large step in ``u``
        (the sigmoid is flat there), so a fit pushed there proposed steps
        that changed the value by under ``step_tol`` and stopped,
        "converged", at 1.999997 of ``(0.5, 2)`` with the truth at 1.25.
        """
        if not self.active:
            return False
        th, gg = np.asarray(theta), np.asarray(g)
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        on_lo, on_hi = th <= lo, th >= hi
        if physical is not None:
            p = np.asarray(physical, dtype=np.float64)
            margin = np.asarray(rel_tol) * np.abs(p)
            on_lo = on_lo | (p - self.p_lo <= margin)
            on_hi = on_hi | (self.p_hi - p <= margin)
        return bool(((on_lo & (gg < 0)) | (on_hi & (gg > 0))).any())


def _physical_edges(spec, dtype, interval) -> tuple[float, float]:
    """The physical values a trainable leaf's coordinate range ends at: its
    bounds under ``transform=None``, and for a ``log`` / ``logit`` leaf the
    values of the ends of ``interval`` (:meth:`ParamSpec._optimiser_interval`),
    which lie :meth:`ParamSpec._usable_margin` inside the bounds."""
    if spec.transform is None:
        lo, hi = spec.bounds
        return (-np.inf if lo is None else float(lo), np.inf if hi is None else float(hi))
    ends = []
    for u, unbounded in zip(interval, (-np.inf, np.inf)):
        ends.append(float(spec.to_constrained(jnp.asarray(u, dtype)))
                    if np.isfinite(u) else unbounded)
    return ends[0], ends[1]


_UNRESOLVED_DIGITS_WHY = (
    "under that the transform keeps fewer than half the working precision's digits "
    "of the value")


def _warn_unresolved_values(method: str, pmap: _PhysicalMap, params: dict, when: str,
                            already: set) -> None:
    """Warn, once per leaf, about a fitted ``log`` / ``logit`` leaf whose
    value its transform cannot resolve to ``sqrt(eps)`` of the value's own
    magnitude (:meth:`ParamSpec._resolution`).

    The transform returns values spaced like the floats near its *bounds*,
    so a small value under wide bounds has few of its digits left: a float32
    ``2.0`` under ``logit`` bounds ``(-1e6, 1e6)`` can be placed only to
    0.12.  The fit then runs, and returns, the nearest value it can (one
    evaluation: :class:`_PhysicalMap`), which nothing used to say.  A value
    of exactly 0 has no magnitude to be relative to and is not judged.
    """
    leaves = jax.tree.leaves(params)
    names = pmap.names()
    for f in pmap.fitted:
        if f.leaf in already or f.spec.transform is None:
            continue
        spacing = f.spec._resolution(f.dtype)  # noqa: SLF001
        if spacing <= 0.0:
            continue
        values = np.abs(np.asarray(leaves[f.leaf]).astype(np.float64).reshape(-1))
        tolerance = math.sqrt(float(np.finfo(f.dtype).eps))
        judged = values > 0.0
        coarse = judged & (spacing > tolerance * values)
        if not bool(coarse.any()):
            continue
        already.add(f.leaf)
        worst = int(np.argmax(np.where(coarse, spacing / np.where(judged, values, 1.0), 0.0)))
        scale = ("eps * |lo|" if f.spec.transform == "log"
                 else "eps * max(|lo|, |hi|, hi - lo)")
        warnings.warn(
            f"{method}: {names[f.start + worst]} = {values[worst]:.6g} {when} cannot be "
            f"resolved under its {f.spec.transform!r} transform with bounds "
            f"{f.spec.bounds}: the transform returns {f.dtype.name} values "
            f"{spacing:.3g} apart ({scale}), {spacing / values[worst]:.2g} of this "
            f"value, against a tolerance of sqrt(eps) = {tolerance:.2g} of it "
            f"({_UNRESOLVED_DIGITS_WHY}). The fit can place it no closer than that. "
            "Tighten the bounds towards the value, declare the parameter with "
            "transform=None (its bounds are then enforced by clipping, at the "
            "value's own resolution), or hold it in a wider dtype.",
            PrecisionLimitWarning, stacklevel=3)


def _edge_listing(pmap: _PhysicalMap, bounds: _CoordinateBounds, theta, params: dict,
                  on: np.ndarray, note: Optional[np.ndarray] = None, why: str = "") -> str:
    """One line per ``log`` / ``logit`` parameter flagged in ``on`` (the
    first six): its name, value, which edge, its transform and bounds --
    and ``why`` for the ones ``note`` flags.  ``theta`` says which edge."""
    names = pmap.names()
    values = pmap.physical(params)
    th, hi = np.asarray(theta), np.asarray(bounds.hi)
    specs = {}
    for f in pmap.fitted:
        for k in range(f.start, f.stop):
            specs[k] = f.spec
    listed = "\n".join(
        f"  - {names[k]} = {values[k]:.9g}: the {'upper' if th[k] >= hi[k] else 'lower'} "
        f"edge of what its {specs[k].transform!r} transform resolves inside bounds "
        f"{specs[k].bounds}" + (why if note is not None and note[k] else "")
        for k in np.flatnonzero(on)[:6])
    more = int(on.sum()) - 6
    return listed + (f"\n  (+{more} more)" if more > 0 else "")


def _warn_starts_on_an_edge(method: str, pmap: _PhysicalMap, bounds: _CoordinateBounds,
                            theta0, start: dict, lr: float, eps: float) -> None:
    """Warn, naming it, about every ``log`` / ``logit`` parameter an Adam fit
    is started with on (or beyond) an edge of the range its transform
    resolves.

    Adam's update is about ``lr`` in the coordinate whatever the gradient,
    and on that edge the transform is flat: a ``logit`` value there is
    ``log(1 / sqrt(eps))`` units of its coordinate from mid-range -- 8 in
    float32, 18 in float64 -- so the fit needs that many ``/ lr`` updates
    before the value moves visibly, and more while the coordinate's
    gradient (the model's, times the transform's slope, ``sqrt(eps)`` of
    the range there) is below Adam's ``eps`` of the largest.  A run shorter
    than that ends between the edge and the optimum with nothing wrong in
    its result but the distance.  :func:`fit_lm` needs no such warning: its
    step is read on the transform's tangent and leaves the edge at once.
    """
    on = bounds.on_an_edge(theta0)
    if not bool(on.any()):
        return
    depth = max(float(np.max(np.abs(np.asarray(theta0, dtype=np.float64)[on]))), 1.0)
    warnings.warn(
        f"{method}: {int(on.sum())} parameter(s) start on the edge of their "
        f"transform's usable range:\n{_edge_listing(pmap, bounds, theta0, start, on)}"
        f"\nAdam moves a coordinate by about lr an update whatever its gradient, "
        f"and a 'log' / 'logit' transform is flat on that edge: it is about "
        f"{depth:.0f} units of the coordinate out, so the value will not move "
        f"visibly for about {depth / lr:.0f} updates at lr={lr:g} -- longer while "
        f"the coordinate's gradient is under eps={eps:g} of the largest, which the "
        "transform's slope there (sqrt(eps) of the range) can make it. Start the "
        "parameter inside its range, give the run that many updates (and a "
        "smaller eps), or use fit_lm, whose step leaves the edge at once.",
        RuntimeWarning, stacklevel=3)


def _warn_on_an_edge(method: str, pmap: _PhysicalMap, bounds: _CoordinateBounds,
                     theta, params: dict, last=None) -> None:
    """Warn, naming it, about every ``log`` / ``logit`` parameter a fit
    leaves on an edge of the range its transform resolves.

    A fit that ends there has not found an interior optimum.  Either the
    data pull the parameter onto its bound, which this transform cannot
    reach, or the fit started out there and no step it could take lowered
    the loss.  Until 0.4.0's fix nothing said so (MADD-ANO-104).

    ``theta`` and ``params`` are the iterate returned.  ``last`` is the
    run's last iterate where that can be another one (the Adam fitters
    return the lowest-loss iterate they evaluated): a parameter the run was
    still holding against an edge when it stopped is named too, with the
    value returned for it, because on the flat of the transform which
    iterate is lowest is the loss's rounding.
    """
    on = bounds.on_an_edge(theta)
    only_last = np.zeros_like(on)
    side = np.asarray(theta)
    if last is not None:
        only_last = bounds.on_an_edge(last) & ~on
        side = np.where(only_last, np.asarray(last), side)
    flagged = on | only_last
    if not bool(flagged.any()):
        return
    listed = _edge_listing(
        pmap, bounds, side, params, flagged, only_last,
        " (the run's last iterate sat on it; this is the lowest-loss one)")
    warnings.warn(
        f"{method}: {int(flagged.sum())} parameter(s) ended on the edge of their "
        f"transform's usable range:\n{listed}"
        "\nThe fitters keep a 'log' / 'logit' parameter where its transform can "
        "be stepped on -- at least sqrt(eps) of the bounds' own size inside each "
        "bound, and below the dtype's largest number -- so it cannot reach its "
        "bound, and a fit that ends there has not found an interior optimum. If "
        "the parameter belongs on (or beyond) its bound, declare it with "
        "transform=None, which clips and can sit on the bound; if it does not, "
        "the fit could not bring it back within its budget -- restart it from "
        "inside the range.",
        RuntimeWarning, stacklevel=3)


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
        cannot resolve, or was decided below the range ``F`` can be
        formed in (:func:`_range_limited`), which includes an ``F`` that
        flushed to exactly zero from a ``J`` that is not -- the
        conditions :func:`fim` turns into a
        :class:`~maddening.warnings.PrecisionLimitWarning`.
        A live gate should read it as a third outcome, "verdict
        unavailable", rather than as a refusal.
    deciding_ratio : jnp.ndarray
        0-d float: the eigenvalue ratio nearest the cutoff in log
        distance -- the number :func:`fim`'s noise-floor warning quotes --
        or ``0.0`` when ``precision_limited`` is False (or when no
        eigenvalue ratio is positive).  Zero and not
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


def _range_limited(top: float, rank_rtol: float, n_residual: int, dtype,
                   factor: float = _PRECISION_WARN_FACTOR, *,
                   j_nonzero: bool = False) -> bool:
    """Whether the rank cutoff sits where ``F = JᵀJ`` cannot be formed.

    ``F``'s entries are sums of ``m`` products ``J_ij * J_ik``, and XLA's
    CPU backend flushes a subnormal product or sum to zero, so an entry --
    and an eigenvalue -- of order ``m * tiny`` or below loses whatever the
    flushed products carried.  When the cutoff ``rank_rtol * max(eig)`` is
    there, the eigenvalues that decide the rank are among those losses and
    the verdict depends on the residual's scale, not on the data: in
    float32 a residual divided by ``noise_std=1e15`` (``|J|`` near
    ``1e-18``) dropped a determined direction, rank 2 to 1, with every
    ``crb`` ``inf``, while the same residual at any smaller ``noise_std``
    reported rank 2.  The test is ``rank_rtol * top < factor * m * tiny``
    (``factor`` is :data:`_PRECISION_WARN_FACTOR`, the band's own margin).

    A largest eigenvalue that is not positive is flagged when ``j_nonzero``
    (``J`` has an entry that is not zero): ``F`` of a nonzero ``J`` is not
    zero, so a zero ``F`` is the same flush taken to the end -- every
    product below ``tiny`` (``|J|`` under about ``1e-19`` in float32) --
    and the cutoff, ``rank_rtol * 0``, decides ``rank = 0`` for any data.
    Earlier 0.4.0 development builds exempted it and reported rank 0 of 3
    silently for a residual of exact rank 2 (MADD-ANO-176).  Whatever
    ``rank_rtol`` is, even ``0``: no verdict can be read from such an
    ``F``.  A zero ``J`` -- a residual that reads no parameter -- is a
    real rank 0 and is not flagged; a non-finite ``top`` is :func:`fim`'s
    ``FloatingPointError``.
    """
    if not math.isfinite(top):
        return False
    if top <= 0.0:
        return bool(j_nonzero)
    if rank_rtol <= 0.0:
        return False
    tiny = float(np.finfo(dtype).tiny)
    return rank_rtol * top < factor * max(int(n_residual), 1) * tiny


def _device_range_limited(eigvals, rank_rtol: float, n_residual: int,
                          factor: float = _PRECISION_WARN_FACTOR, *,
                          j_nonzero: Any = False):
    """:func:`_range_limited` on the device, as a 0-d bool; the threshold
    is computed on the host (every operand but ``eigvals`` and
    ``j_nonzero``, a traced 0-d bool, is static)."""
    hi = eigvals[-1]
    # ``hi <= 0`` is False for a ``NaN`` ``hi``: a non-finite ``F`` is
    # ``finite``'s to report, not this flag's.
    flushed = (hi <= 0.0) & j_nonzero
    if rank_rtol <= 0.0:
        return jnp.asarray(flushed, dtype=bool)
    tiny = float(np.finfo(eigvals.dtype).tiny)
    threshold = factor * max(int(n_residual), 1) * tiny / rank_rtol
    return (jnp.isfinite(hi) & (hi > 0.0) & (hi < threshold)) | flushed


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
    in float64 -- but it is applied at ``theta0``'s precision, and a
    width of ``1e-300`` or a pair of bounds like ``(0.0, 1e39)`` is a
    zero or an ``inf`` multiplier there.  A zero column is the defect this
    scale exists to remove, so a width that is not a positive, finite,
    *normal* number at that precision (a subnormal one is flushed to zero
    by the arithmetic it enters) is refused by name rather than let
    through.  The width itself is tested, not its square: ``F`` is formed
    from the scaled Jacobian ``J * width``, whose size depends on ``J`` as
    much as on the width, and the square of a width is never computed; a
    Gram product whose entries fall below the normal range is what
    :func:`_range_limited` reports.  Testing the square (0.4.0 development builds)
    refused the bounds ``(0, 1e-23)`` of a parameter whose natural size is
    ``1e-23`` -- a 10 nm particle's volume in cubic metres -- whose report
    is perfectly well conditioned.  Trace time, host side: ``nominal`` is
    static, so no value of ``theta0`` is consulted.
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
        if not (np.isfinite(w) and w >= np.finfo(dtype).tiny):
            raise ValueError(
                f"scale='nominal': the width of {name}'s bounds is {width!r}, "
                f"which is not a positive, finite, normal number at "
                f"{np.dtype(dtype)} ({w!r}). A column scaled by it would be "
                f"zero or non-finite and the parameter would read as "
                f"unidentifiable whatever the data say -- the failure this "
                f"scale exists to remove -- so it is refused rather than "
                f"reported. Widen or narrow the bounds to a width this "
                f"precision can carry, or re-run under x64.")
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
    # A cutoff below the range ``F`` can be formed in (:func:`_range_limited`)
    # leaves the verdict as unavailable as one at the noise floor -- and so
    # does an ``F`` flushed to exactly zero from a ``J`` that is not.
    out_of_range = _device_range_limited(eigvals, rtol, n_residual,
                                         j_nonzero=jnp.any(J != 0.0))
    ratio = jnp.where(limited | ~out_of_range, ratio,
                      _device_precision_limited(eigvals, rtol, floor,
                                                factor=math.inf)[1])
    limited = limited | out_of_range
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
    spring's relative bound on ``mass`` reported as 0.41 against a true
    44.7 -- 109x too tight -- after its damping moved from 2 to 20 in
    ``gm.params`` (its ``stiffness`` bound, 0.362 against a true 0.238,
    erred the other way), and rank 2 against a true rank 1 after a bound
    method's excitation changed.  No key can
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
        sensitivities, and with them every verdict that compares one
        column with another depends on the parameters' units: ``rank``
        (the cutoff is relative to the largest eigenvalue), the
        parameters ``crb`` calls ``+inf``, ``cond`` and
        ``least_identifiable``.  A parameter written in units ``1e-5``
        has a column ``1e-5`` the size of the others' and reads as
        unresolved under ``None`` however well the data determine it;
        under ``"relative"`` or ``"nominal"`` the report is the same in
        any units.  :func:`fit_lm`'s identifiability guard reads its
        curvature as ``"relative"`` does, for that reason
        (:func:`_relative_scale`).  A parameter whose value is exactly ``0.0`` has
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
        to zero, be subnormal or overflow at the parameters' precision is
        refused by name (the width itself: its square is never formed).  Columns are still dimensionless, so ``cond`` still
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
        legitimate thing to ask for.  As there, every leaf must be a bool
        (``ValueError`` otherwise).
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
        ``jax_enable_x64`` -- or, when x64 is already on and the
        decomposition is float32 because the parameters are, hold the
        parameters and the state in float64.  x64 is not the default and is not going to
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
        width that is not a positive, finite, normal number at the
        parameters' precision; a ``mask`` leaf that is not a bool, a
        ``mask`` selecting an integer or boolean leaf, or a ``params``
        with no floating-point leaf).
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
        # A float32 decomposition *in an x64 process* comes from float32
        # leaves (the residual they make is float32 too), so turning x64
        # on -- the float32 remedy -- is already done and changes nothing:
        # what settles it there is float64 leaves (SYS-024).
        remedy = (
            "x64 is already on, but the decomposition is float32 because "
            "the parameters (and the residual they produce) are float32: "
            "settle it by holding them in float64 -- cast the parameters "
            "and the state, e.g. jax.tree.map(lambda v: jnp.asarray(v, "
            "jnp.float64), params) -- which moves the cutoff to "
            "max(n, sqrt(m)) * 2.22e-16 and computes the ratio to match."
            if dtype == np.float32 and _x64_enabled() else
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
    rtol_in_force = _resolve_rank_rtol(eigvals_h.dtype, n_params, rank_rtol,
                                       n_residual=n_residual)
    # Read back only when ``F`` has no positive eigenvalue -- the one case
    # it decides -- so an ordinary report keeps its four transfers.
    j_nonzero = (math.isfinite(hi) and hi <= 0.0) and bool(jnp.any(J != 0.0))
    if j_nonzero and _range_limited(hi, rtol_in_force, n_residual, eigvals_h.dtype,
                                    j_nonzero=True):
        dtype = eigvals_h.dtype
        warnings.warn(
            f"rank={rank} of {len(names)} was read from a Fisher matrix that "
            f"came out exactly zero although J has nonzero entries: every "
            f"product J_ij * J_ik is below {dtype}'s smallest normal number "
            f"({float(np.finfo(dtype).tiny):.4g}) and the arithmetic flushes "
            f"it to zero, so F = J^T J carries nothing of what J does, the "
            f"cutoff is 0 and rank, crb and cond say nothing about the data. "
            f"Rescale the residual -- a smaller noise_std, or residual units "
            f"nearer one -- so that F's entries are normal numbers"
            + (", or re-run under x64." if dtype == np.float32 else "."),
            PrecisionLimitWarning,
            stacklevel=2,
        )
    elif _range_limited(hi, rtol_in_force, n_residual, eigvals_h.dtype):
        dtype = eigvals_h.dtype
        warnings.warn(
            f"rank={rank} of {len(names)} was decided below {dtype}'s normal "
            f"range: the cutoff rank_rtol * max(eigvals) = "
            f"{rtol_in_force * hi:.4g} is under {_PRECISION_WARN_FACTOR:g} * "
            f"m * tiny = {_PRECISION_WARN_FACTOR * n_residual * float(np.finfo(dtype).tiny):.4g} "
            f"(m = {n_residual} residual rows). F = J^T J sums m products "
            f"per entry and the arithmetic flushes a subnormal product to "
            f"zero, so the eigenvalues that decide the rank have lost what "
            f"those products carried: rank, crb and cond depend on the "
            f"residual's scale here, not on the data, and are provisional "
            f"together. Rescale the residual -- a smaller noise_std, or "
            f"residual units nearer one -- so that F's entries are normal "
            f"numbers"
            + ((", or hold the parameters and the state in float64 (x64 is "
                "already on; the decomposition is float32 because they are)."
                if _x64_enabled() else ", or re-run under x64.")
               if dtype == np.float32 else "."),
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

    Only a real number is read: a Python or NumPy number, or a 0-d real
    array.  ``float()`` alone accepted a ``bool`` -- ``fit(tol=True)`` read
    as ``tol=1.0`` and reported ``converged=True`` at an unfitted start --
    and a numeric string, so both are refused, as is any other type.
    """
    if not _is_real_number(value):
        raise ValueError(
            f"{name} must be a finite number, got {value!r} "
            f"({type(value).__name__}).{why}")
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


def _is_real_number(value) -> bool:
    """Whether ``value`` is a real number a hyper-parameter may be: an
    ``int``/``float`` (or other non-complex :class:`numbers.Number`), a
    NumPy integer or floating scalar, or a 0-d integer or floating array --
    and never a ``bool`` in any of those spellings."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (np.ndarray, jax.Array)):
        return (value.ndim == 0 and value.dtype != np.bool_
                and (jnp.issubdtype(value.dtype, jnp.integer)
                     or jnp.issubdtype(value.dtype, jnp.floating)))
    if isinstance(value, numbers.Real):
        return True
    # ``decimal.Decimal`` is a number that is not registered as ``Real``;
    # ``complex`` is ``Complex`` and has no ordering to read a bound with.
    return isinstance(value, numbers.Number) and not isinstance(value, numbers.Complex)


def _check_count(name: str, value, *, minimum: int = 0) -> int:
    """Reject a non-integer or too-small iteration/interval count.

    An integer in every spelling reads, as a real number does for
    :func:`_check_hyper`: a Python or NumPy integer, a JAX integer scalar,
    or a 0-d integer array of either library.  Earlier 0.4.0 development
    builds accepted only a :class:`numbers.Integral`, so ``n_iter=
    jnp.int32(5)`` and ``np.asarray(5)`` were refused while
    ``lr=jnp.asarray(0.05)`` read.  A ``bool`` in any spelling, a float
    (even ``5.0``) and anything else are still refused.
    """
    if isinstance(value, (bool, np.bool_, jax.core.Tracer)):
        # A tracer has no value to count with (``windowed_loss`` under
        # ``jax.jit`` with a traced ``start_step``): refused as before.
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if isinstance(value, (np.ndarray, jax.Array)):
        if value.ndim != 0 or not jnp.issubdtype(value.dtype, jnp.integer):
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
        value = int(value)
    elif not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _adam_frame(g0):
    """The power of two Adam's gradients are multiplied by for a whole run.

    Adam's step ``m_hat / (sqrt(v_hat) + eps)`` is invariant to the scale of
    the gradient except through ``eps``, which is absolute: a loss written in
    small units (squared micrometres in metres, say: ``1e-12``) has
    gradients far below the default ``eps = 1e-8``, the denominator is
    ``eps``, and the step shrinks by the same factor -- a spring's damping
    fitted on such a loss did not move from its starting value in 60
    iterations, where the same loss at scale one recovered it.  The run's
    gradients are therefore taken in the frame of the *first* gradient
    (:func:`~maddening.core._pow2_frame.pow2_frame`: its largest entry
    brought into ``[0.5, 1)``), fixed for the run so ``m`` and ``v``
    accumulate in one unit, and ``eps`` is relative to that.  An exact power
    of two leaves the step bit-identical wherever ``eps`` was negligible and
    makes the iterates the same, to the bit, for the loss scaled by any
    power of two.  A first gradient of exact zeros is framed by ``1``.
    """
    return pow2_frame(g0)


def _model_loss_and_gradient(pmap: "_PhysicalMap", objective, n_extra: int = 0):
    """``(plain, scaled)``: the loss and its gradient as compiled functions
    of the physical parameters (:func:`_compile_model`).

    ``objective(params, *extra)`` is the fitter's loss at a physical tree
    (and, for multiple shooting, the window states).  ``plain(params,
    *extra)`` returns ``(loss, g_p, *g_extra)`` with ``g_p`` the gradient
    with respect to the trainable entries of ``params``, raveled in
    ``theta`` order (:meth:`_PhysicalMap.ravel`); the fitter chains it with
    :meth:`_PhysicalMap.slope`.  ``scaled(params, *extra, cot)`` is the same
    with ``cot`` as the cotangent of the backward pass.

    A power-of-two ``cot`` scales every intermediate of the backward pass
    exactly, so the result is ``cot`` times the gradient bit for bit
    wherever those intermediates were normal numbers -- and where they were
    not, it lifts them out of the flush (:func:`_gradient_lift`).
    """
    wrt = tuple(range(1 + n_extra))

    def at(params):
        fitted, others = pmap.split(params)
        return fitted, lambda leaves, *extra: objective(pmap.join(leaves, others), *extra)

    def plain(params, *extra):
        fitted, f = at(params)
        loss, grads = jax.value_and_grad(f, argnums=wrt)(fitted, *extra)
        return (loss, pmap.ravel(grads[0]), *grads[1:])

    def scaled(params, *args):
        *extra, cot = args
        fitted, f = at(params)
        loss, pullback = jax.vjp(f, fitted, *extra)
        grads = pullback(cot)
        return (loss, pmap.ravel(grads[0]), *grads[1:])

    return _compile_model(plain), _compile_model(scaled)


#: How many evaluations :func:`_gradient_lift` may spend looking for a
#: cotangent at which the backward pass neither flushes nor overflows: one
#: probe at the widest lift, halvings of its exponent while that overflows,
#: and one evaluation at the lift that frames the gradient.
_LIFT_PROBES = 12


def _gradient_lift(scaled, grads, loss):
    """The power of two to take an Adam fit's gradients with (as the
    cotangent of the backward pass, :func:`_model_loss_and_gradient`), or
    ``None`` when the gradient as evaluated is already clear of the flush.

    ``scaled(cot)`` returns ``(loss, gradients)`` at the run's start; ``grads``
    are the gradients evaluated plainly there, and ``loss`` the loss.  Their
    products flush to zero below ``tiny`` on XLA's CPU backend, so a loss
    written in small units -- ``0.5 * ||s * r||²`` with ``s = 1e-19``, a
    nanoparticle's mass in kilograms -- had a gradient of exactly zero, and
    :func:`fit` returned its start (at ``s = 1e-18`` a gradient that had
    lost some of its products, and a point 1% off).  The loss's *value*
    cannot be helped (it is the caller's function, and ``r * r`` has
    already flushed inside it by the time it returns), but its gradient
    can: the backward pass is linear in its cotangent, so evaluating it
    with ``cot = 2**k`` instead of 1 lifts every product out of the flush
    and scales the result by exactly ``2**k``.  Adam is invariant to that
    scale beyond its ``eps``, which is relative to the run's first gradient
    already (:func:`_adam_frame`).

    A gradient whose largest entry is at least ``tiny / eps`` of its dtype
    (``2**-103`` in float32) is clear: a product that flushed there was
    ``2**23`` times smaller than it and moved it by less than an ulp, so no
    lift is taken and the run is the one it always was, bit for bit.  Below
    that the widest lift whose backward pass stays finite is probed, and
    the lift is then set to frame the gradient's largest entry near one.
    ``None`` also when even the widest lift finds no gradient: then it is
    zero (a loss that does not read the parameters, or a start at an exact
    optimum), and the plain evaluation already says so.
    """
    leaves = [np.asarray(g) for g in grads]
    floor = min(float(np.finfo(g.dtype).tiny) / float(np.finfo(g.dtype).eps)
                for g in leaves if np.issubdtype(g.dtype, np.floating)) \
        if any(np.issubdtype(g.dtype, np.floating) for g in leaves) else 0.0

    def top(gs):
        values = [np.abs(np.asarray(g, dtype=np.float64)).max() if np.size(g) else 0.0
                  for g in gs]
        return max(values) if values else 0.0

    first = top(leaves)
    if not math.isfinite(first) or first >= floor:
        return None
    cot_dtype = jnp.result_type(loss)
    if not jnp.issubdtype(cot_dtype, jnp.floating):
        return None
    widest = int(jnp.finfo(cot_dtype).maxexp) - 2

    def probe(exponent):
        _, gs = scaled(jnp.asarray(2.0 ** exponent, cot_dtype))
        t = top(gs if isinstance(gs, (tuple, list)) else (gs,))
        return t if math.isfinite(t) else None

    exponent, budget = widest, _LIFT_PROBES
    t = probe(exponent)
    budget -= 1
    while t is None and exponent > 1 and budget > 1:
        exponent //= 2
        t = probe(exponent)
        budget -= 1
    if t is None or t == 0.0:
        return None
    framed = min(exponent - pow2_exponent(t), widest)
    if framed <= 0:
        return None
    t2 = probe(framed)
    if t2 is None or t2 < floor:
        framed = exponent
    return jnp.asarray(2.0 ** framed, cot_dtype)


def _check_adam_hyper(n_iter, lr, tol, betas, eps, notify_every) -> tuple[int, int]:
    """The hyper-parameters :func:`fit` and :func:`fit_multiple_shooting`
    share; returns ``(n_iter, notify_every)`` as Python ints."""
    n_iter = _check_count("n_iter", n_iter)
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
    return n_iter, _check_count("notify_every", notify_every)


#: Largest trainable-parameter count for which :func:`fit` accumulates the
#: gradient Gram matrix that :class:`_ExcitationTracker` uses.
#:
#: The tracker holds one ``n x n`` float64 triangular factor and a buffer
#: of at most ``max(2n, 256)`` gradients, which it folds into the factor
#: with one QR when full -- ``O(n**2)`` per gradient, as the ``n x n``
#: outer product it replaced cost -- plus one SVD at the end, ``O(n**3)``
#: once, against the full rollout and reverse pass the gradient itself
#: costs, and it only runs at all once there are ``n`` gradients to pool.
#: At the cap that is 2.1 MB for the factor and 4.2 MB for the buffer.
#: Wall clock is not quoted: measured on this box the same 512-wide
#: decomposition ranged over 52-892 ms across five back-to-back runs, which
#: is contention from the other work sharing it and not a property of the
#: code.  A caller who does not want the cost at all passes
#: ``hold_undetermined=False``.
#:
#: When the Gram matrix leaves ``k`` directions unspanned, the guard pays
#: once more, at the selected iterate, to ask the objective's curvature
#: about them (:func:`_hold_undetermined_directions`): for :func:`fit_lm`
#: the Jacobian it already computes, at most one extra evaluation of it;
#: for :func:`fit` and :func:`fit_multiple_shooting` ``k`` plus up to
#: :data:`_HOLD_SCALE_DIRECTIONS` Hessian-vector products, evaluated one
#: after another so that the memory is one product's, and one compilation
#: of them.  Then one or two evaluations of the loss for the check
#: :func:`_hold_tolerance` describes.  A fit whose gradients spanned
#: everything pays none of this.
#:
#: Above the cap the tracker is not built and
#: :attr:`FitResult.excited_rank` is ``None`` -- "not measured", never
#: "full rank", because a silent full-rank verdict would read as "no
#: undetermined direction was found" when nothing looked.
_EXCITATION_MAX_PARAMS = 512

#: How many of the run's most-excited directions the Hessian test probes,
#: besides the candidates, to put a scale on "no curvature".  Their
#: Hessian-vector products bound ``||H||`` from below, so a scale that
#: comes out low makes the test stricter -- fewer directions held, the
#: fail-open side -- and never looser.  For ``n <= k + 8`` the probes are
#: the whole space and the scale is exact.
_HOLD_SCALE_DIRECTIONS = 8

#: Relative part of :func:`_hold_tolerance`, in units of the working
#: precision's ``eps``: ``2**10``.
_HOLD_LOSS_RTOL_EPS = 1024.0

#: Roundings per coordinate whose effect on the loss is the absolute part
#: of :func:`_hold_tolerance`; one rounding of a coordinate ``u`` is
#: ``eps * max(1, |u|)``.
_HOLD_LOSS_ROUNDINGS = 4.0

#: :func:`fit_lm`'s default ``step_tol``, in units of each parameter's own
#: ``eps``: a proposed step converges the run when it moves every trainable
#: parameter by at most ``2**4`` units in the last place of itself.
#:
#: The tolerance is the parameters' float resolution because that is the
#: only scale every fit has.  The fixed ``1e-8`` it replaces was an
#: absolute distance in the optimiser's coordinates, and in float32 it is
#: below one ulp of almost any coordinate (``log(40)`` has an ulp of
#: 2.4e-7), so no step could meet it.  What a fit at the float32 floor
#: does instead is propose a step that rounds to nothing: noiseless spring
#: data reached loss 8.7e-14 in four iterations, the next proposal moved
#: neither coordinate by a single bit, and the run ended "unconverged".
#: ``2**4`` rather than 1 because a proposal at the floor is the
#: Gauss-Newton fit of the residual's rounding noise, which can land a few
#: ulps from the iterate rather than on it; it is far below anything a
#: float32 fit determines (16 ulps is 1.9e-6 relative).
_STEP_TOL_ULPS = 2.0 ** 4

#: Damping candidates :func:`fit_lm` tries per iteration before it gives up
#: on the iteration -- unless the floor rule is to judge it.  The rule says
#: "every candidate was rejected, down to one damped within ``step_tol``",
#: and twelve rungs of ``lam_up`` need not get there: under
#: ``jax_enable_x64`` a noiseless spring fit at loss ``2e-30`` ended its
#: twelfth rung at ``lam = 1e5`` with a relative step of ``4.4e-15``
#: against a tolerance of ``3.6e-15``, one rung short, and was reported
#: unconverged.  So after a run that has lowered the loss the ladder goes
#: on, a decade or more a rung, until a candidate is accepted or within
#: ``step_tol`` or the damping reaches :data:`_LM_LAMBDA_MAX` (at most 24
#: more rungs, each one residual evaluation).
_LM_LADDER = 12

#: :func:`fit_lm`'s damping cap (and ``1e-12``, its floor): the ``lam`` at
#: which a candidate is a gradient step ``1e12`` times shorter than the
#: Gauss-Newton one.
_LM_LAMBDA_MAX = 1e12



def _float_resolution(tree) -> np.ndarray:
    """``eps`` of each entry's own dtype, in ``ravel_pytree(tree)`` order.

    Per entry rather than of the raveled vector's dtype: ``ravel_pytree``
    promotes a float32 leaf beside a float64 one to float64, and a
    tolerance taken from float64's ``eps`` is one no float32 parameter
    can meet.  A non-floating leaf (never trainable) gets float32's.
    """
    parts = []
    for leaf in jax.tree.leaves(tree):
        dtype = jnp.result_type(leaf)
        eps = float(np.finfo(dtype if jnp.issubdtype(dtype, jnp.floating)
                             else np.float32).eps)
        parts.append(np.full(_leaf_size(leaf), eps))
    return np.concatenate(parts) if parts else np.zeros(0)


def _leaf_grid(tree, idx, dtype):
    """``(narrow, to_leaf_grid)`` for an optimiser over the trainable entries
    ``idx`` (in ``ravel_pytree(tree)`` order) held in ``dtype``.

    ``narrow`` flags the coordinates whose leaf is a narrower float than
    ``dtype`` -- a float32 constant in an x64 graph, which ``ravel_pytree``
    promotes to float64 beside a float64 one -- and ``to_leaf_grid(theta)``
    rounds exactly those to their leaf's dtype, as ``unravel`` does before
    the model sees them.  :func:`fit_lm` keeps its iterate there, so the
    coordinate it carries is the value the model computes with.  Earlier
    0.4.0 development builds did not: a step that moved such a coordinate by
    less than one float32 ulp left the leaf where it was, the joint step's
    compensation in the other parameters (computed for a move that never
    happened) raised the loss and was rejected, and the float64 coordinate
    crept by ~1e-11 an iteration until it crossed a rounding boundary --
    five times the iterations of the all-float64 fit, and some runs
    unconverged.  ``narrow`` is all False, and ``to_leaf_grid`` the
    identity, when no leaf is narrower than ``dtype``.
    """
    groups: list = []
    offset = 0
    selected = np.zeros(sum(_leaf_size(leaf) for leaf in jax.tree.leaves(tree)), dtype=bool)
    selected[np.asarray(idx, dtype=np.intp)] = True
    position = np.full(selected.size, -1, dtype=np.intp)
    position[np.asarray(idx, dtype=np.intp)] = np.arange(len(idx))
    wide_eps = float(np.finfo(dtype).eps) if jnp.issubdtype(dtype, jnp.floating) else 0.0
    narrow = np.zeros(len(idx), dtype=bool)
    for leaf in jax.tree.leaves(tree):
        n = _leaf_size(leaf)
        leaf_dtype = jnp.result_type(leaf)
        if (jnp.issubdtype(leaf_dtype, jnp.floating)
                and float(jnp.finfo(leaf_dtype).eps) > wide_eps):
            pos = position[offset:offset + n][selected[offset:offset + n]]
            if pos.size:
                narrow[pos] = True
                groups.append((jnp.asarray(pos), leaf_dtype))
        offset += n

    def to_leaf_grid(theta):
        for pos, leaf_dtype in groups:
            theta = theta.at[pos].set(theta[pos].astype(leaf_dtype).astype(theta.dtype))
        return theta

    return narrow, to_leaf_grid


def _coarsest_dtype(tree, idx) -> np.dtype:
    """The floating dtype of the trainable entries ``idx`` (in
    ``ravel_pytree(tree)`` order) whose ``eps`` is largest.

    ``ravel_pytree`` promotes a float32 leaf beside a float64 one to
    float64, but what the run can resolve along a direction is bounded by
    the arithmetic of the coarsest leaf the model computes with, not by the
    raveled vector's dtype: the identifiability guard reads its precision
    from here.  Measured: three float32 constants of a spring in an x64
    graph, judged at float64's ``eps``, made the guard report full rank for
    their exact scale degeneracy in every run.
    """
    selected = np.zeros(sum(_leaf_size(leaf) for leaf in jax.tree.leaves(tree)), dtype=bool)
    selected[np.asarray(idx, dtype=np.intp)] = True
    best, best_eps, offset = None, -1.0, 0
    for leaf in jax.tree.leaves(tree):
        n = _leaf_size(leaf)
        dtype = jnp.result_type(leaf)
        if jnp.issubdtype(dtype, jnp.floating) and selected[offset:offset + n].any():
            eps = float(jnp.finfo(dtype).eps)
            if eps > best_eps:
                best, best_eps = dtype, eps
        offset += n
    return np.dtype(best if best is not None else np.float32)


#: Gradients :class:`_ExcitationTracker` buffers before folding them into
#: its triangular factor, at least: a fold is one QR of the factor and the
#: buffer, so the cost per gradient is ``O(n**2)`` like the outer product
#: it replaces, and the folds' rounding stays below the cutoff (the cutoff
#: grows as ``sqrt(T)`` and the rounding of ``F`` folds as ``sqrt(F)``,
#: ``F < T``).
_EXCITATION_FOLD_ROWS = 256


class _ExcitationTracker:
    """Accumulated gradient second moment ``G = sum_t g_t g_t^T`` of a fit.

    Every gradient of a least-squares loss is ``J^T r``, so it lies in the
    row space of ``J``: a direction ``v`` with ``J v = 0`` has ``g . v = 0``
    at *every* iterate, and the whole run's gradients stay inside the
    subspace the data can see.  ``G``'s near-null eigenvectors therefore
    name the directions no gradient ever pointed along -- the ones the
    optimiser had no information about.

    That is a *necessary* condition for a direction the data cannot
    determine, not a sufficient one, and this class is only the first of
    the guard's two tests.  The converse fails on any run too short or too
    fast to excite everything: Levenberg-Marquardt reaching the minimum of
    a four-parameter bowl in a few nearly parallel steps leaves three
    directions unspanned, every one of them determined, and holding them
    put the returned parameters at a loss of 0.22 where the fit had
    reached 0.0.  :func:`_hold_undetermined_directions` asks the
    objective's curvature at the returned point about each candidate
    before it holds anything.

    Pooling over the run rather than testing one gradient at a time is what
    makes the verdict robust: near convergence a single gradient is mostly
    rounding noise, but its *accumulated* energy along an exactly-null
    direction stays at the arithmetic's floor while every direction the data
    resolves keeps the energy it collected during the descent.  Measured on
    the spring's ``(k, c, m)`` common-scale degeneracy, the null eigenvalue
    sits at ``1e-15`` of the largest over 200 to 10,000 iterations and
    learning rates 0.01 to 0.2, against ``8e-3`` for the weakest direction
    the data does resolve -- twelve decades of separation.

    ``G`` is never formed.  The tracker keeps ``R``, the triangular factor
    of the stacked gradients (``G = RᵀR``), in float64, and reads ``G``'s
    spectrum as the squared singular values of ``R``.  Forming ``G`` and
    decomposing it squares the condition: ``eigh`` resolves ``G``'s
    eigenvalues only to about ``eps64`` of the largest, so an exactly null
    direction of float64 gradients -- whose energy is ``eps64**2`` of the
    largest -- came back at ``5e-17`` of it, far above the cutoff of
    ``4e-31``, and under ``jax_enable_x64`` the guard reported full rank
    for the spring's exact scale degeneracy and let Adam drift along it.
    The factor's singular values resolve to ``eps64`` of the largest, the
    square root of what the cutoff needs, at either precision.  In float32
    the verdict is unchanged: ``eig(G) = s**2`` and the cutoff is the same
    rule (see :meth:`projector`).

    ``eps`` is that of the coarsest trainable leaf's dtype
    (:func:`_coarsest_dtype`), which bounds what the gradients resolve, not
    the raveled vector's: a float32 leaf in an x64 graph is float32.
    """

    def __init__(self, n: int, dtype, eps: Optional[float] = None) -> None:
        self.n = int(n)
        self.eps = float(np.finfo(dtype).eps) if eps is None else float(eps)
        self.count = 0
        self._r = np.zeros((0, self.n), dtype=np.float64)
        self._rows: list = []
        # A non-finite gradient makes the spectrum undefined for the rest
        # of the run (:meth:`split` answers ``None``); nothing more is kept.
        self._finite = True

    def observe(self, g) -> None:
        """Fold one gradient into the accumulated second moment."""
        self.count += 1
        if not self._finite:
            return
        row = np.asarray(g, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(row)):
            self._finite, self._rows = False, []
            self._r = np.full((1, self.n), np.nan)
            return
        self._rows.append(row)
        if len(self._rows) >= max(2 * self.n, _EXCITATION_FOLD_ROWS):
            self._fold()

    def _fold(self) -> np.ndarray:
        """``R``, with every buffered gradient folded in (one QR)."""
        if self._rows:
            stacked = np.vstack([self._r, np.stack(self._rows)])
            self._rows = []
            self._r = np.linalg.qr(stacked, mode="r")
        return self._r

    @property
    def _gram(self) -> np.ndarray:
        """``G = RᵀR``, formed on request (diagnostics and tests; the
        verdicts never read it)."""
        r = self._fold()
        return r.T @ r

    def gradient_scale(self) -> np.ndarray:
        """``sqrt(diag(G))``: each coordinate's accumulated gradient
        magnitude, which scales with the coordinate's units as a Jacobian
        column does (:func:`_relative_scale` uses it where a coordinate
        has no value of its own to be relative to)."""
        return np.linalg.norm(self._fold(), axis=0)

    def split(self, scale: Optional[np.ndarray] = None
              ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """``(eigvecs, excited)`` of ``G``, or ``None`` if it cannot say.

        ``eigvecs`` are ``G``'s orthonormal eigenvectors in ascending order
        of eigenvalue and ``excited`` flags those above the cutoff
        described in :meth:`projector`; the rest are the directions no
        gradient of the run pointed along.  ``None`` under the same two
        conditions :meth:`projector` answers ``(None, None)``.

        With ``scale`` (``c``, one positive entry per coordinate), ``G`` is
        read in the coordinates ``zeta = c * theta``: its gradients are
        ``g / c``, so the matrix decomposed is ``G / outer(c, c)`` and the
        eigenvectors are directions of ``zeta``.  A cutoff relative to the
        largest eigenvalue compares coordinates with each other, so it is
        only invariant to their units in coordinates that are
        (:func:`_relative_scale`); without ``scale``, ``theta``'s own.
        """
        if self.count < self.n:
            return None
        r = self._fold()
        if not np.all(np.isfinite(r)):
            return None
        if scale is not None:
            r = r / np.asarray(scale, dtype=np.float64)[None, :]
        _, s, vt = np.linalg.svd(r, full_matrices=True)
        sv = np.zeros(self.n, dtype=np.float64)
        sv[:s.size] = s
        order = np.argsort(sv, kind="stable")      # ascending, as ``eigh``
        sv, evecs = sv[order], vt.T[:, order]
        top = float(sv[-1])
        if not np.isfinite(top) or top <= 0.0:
            return None
        cutoff = max(self.n, math.sqrt(self.count)) * self.eps * top
        return evecs, np.asarray(sv > cutoff)

    def projector(self, scale: Optional[np.ndarray] = None
                  ) -> tuple[Optional[int], Optional[np.ndarray]]:
        """``(rank, P)`` for the excited subspace, or ``(None, None)``.

        ``P`` is the orthogonal projector onto the span of the directions
        the gradients excited, and is ``None`` when the rank is full (there
        is nothing to remove, and returning the iterate untouched keeps it
        bit for bit).  ``(None, None)`` means the question was not answered:
        fewer gradients than parameters, so a direction can be unobserved
        merely for want of iterations, or a degenerate spectrum.  This is
        the gradient test on its own; :attr:`FitResult.excited_rank` is
        decided by it together with the curvature test
        (:func:`_hold_undetermined_directions`).

        The cutoff is ``(max(n, sqrt(T)) * eps)**2`` of the largest
        eigenvalue -- ``max(n, sqrt(T)) * eps`` of the largest singular
        value of ``R``, which is how it is applied -- for ``n`` parameters
        and ``T`` gradients: the same rule
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

        ``scale`` as for :meth:`split`: ``P`` is then a projector in the
        scaled coordinates.
        """
        split = self.split(scale)
        if split is None:
            return None, None
        evecs, keep = split
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


def _make_excitation_tracker(hold_undetermined, theta0,
                             dtype=None) -> Optional[_ExcitationTracker]:
    """The tracker for a fit's trainable block, or ``None`` if it cannot run.

    Shared by :func:`fit`, :func:`fit_lm` and :func:`fit_multiple_shooting`
    so that all three answer :attr:`FitResult.excited_rank` by the same
    rule.  ``None`` means "not measured", which is what
    :attr:`FitResult.excited_rank` reports as ``None``: the caller switched
    the guard off, there are more trainable coordinates than
    :data:`_EXCITATION_MAX_PARAMS`, or the block is empty or not
    floating-point.  ``dtype`` is the coarsest trainable leaf's
    (:func:`_coarsest_dtype`); without it, ``theta0``'s.
    """
    _check_hold_undetermined(hold_undetermined)
    if (hold_undetermined
            and 0 < theta0.size <= _EXCITATION_MAX_PARAMS
            and jnp.issubdtype(theta0.dtype, jnp.floating)):
        return _ExcitationTracker(int(theta0.size),
                                  theta0.dtype if dtype is None else dtype)
    return None


@dataclass(frozen=True)
class _SelectedObjective:
    """What the guard asks of a fitter's objective, at the selected iterate.

    Each fitter builds one from the compiled functions its loop already
    used, so that every number the guard compares is computed the way the
    fitter computed its own.  All three are called lazily, and only when
    the gradient test left a candidate direction: a fit whose gradients
    spanned everything evaluates none of them.

    ``loss(theta)``
        The fitter's loss at a point of its trainable block, as a float.
    ``reference()``
        ``(loss, gradient)`` at the selected iterate, the loss by the same
        function as ``loss`` so that the check compares like with like.
    ``flatness(candidates, excited, scale)``
        The curvature test, in the coordinates ``zeta = scale * theta``
        (:func:`_relative_scale`) the candidate and excited columns are
        directions of: ``(W, flat, curvature)`` with ``W`` an orthogonal
        ``k x k`` rotation of the candidate columns, ``flat`` flagging the
        rotated directions along which the objective has no curvature,
        and ``curvature`` the largest curvature in those coordinates, which
        the tolerance's quantisation term uses; or ``None`` when there is
        no curvature to ask (a Hessian-vector product JAX cannot form, or a
        non-finite one; see :func:`_no_curvature`).
    ``transformed``
        Per coordinate, whether it is a ``log`` / ``logit`` coordinate
        (:attr:`_CoordinateBounds.transformed`).  ``None`` reads every
        coordinate as one, which leaves ``theta`` unscaled.
    ``columns()``
        Per coordinate, a magnitude that scales with its units as a
        Jacobian column does -- ``||J[:, i]||`` for :func:`fit_lm`,
        :meth:`_ExcitationTracker.gradient_scale` for the other two --
        called only for an identity coordinate whose start and selected
        values are both exactly 0 (:func:`_relative_scale`).
    """

    loss: Callable[[Any], float]
    reference: Callable[[], tuple[float, Optional[np.ndarray]]]
    flatness: Callable[[np.ndarray, np.ndarray, np.ndarray], Optional[tuple]]
    transformed: Optional[np.ndarray] = None
    columns: Optional[Callable[[], np.ndarray]] = None


def _relative_scale(theta0, selected, transformed, columns=None) -> np.ndarray:
    """``c``, one positive entry per coordinate, such that the guard's
    coordinates ``zeta = c * theta`` do not depend on any parameter's units.

    A ``log`` or ``logit`` coordinate already does not: a change of units
    multiplies a ``log`` parameter by a constant, which shifts ``theta``
    without scaling it, and a ``logit`` coordinate is dimensionless.  Its
    ``c`` is 1.  An identity coordinate *is* the parameter in its own
    units, so ``c = 1 / max(|theta0|, |selected|)``: ``zeta`` is then the
    parameter's change relative to its own size, the quantity a ``log``
    coordinate measures, and what :func:`fim`'s default
    ``scale="relative"`` puts in a Jacobian column.  Every test the guard
    makes -- which gradients the run pointed along, which directions the
    objective is flat in, where the hold puts the held point and how much
    loss rounding it may cost -- compares coordinates with each other, so
    each is invariant to the parameters' units in ``zeta`` and in no
    coordinates that are not.  Measured in ``theta`` (0.4.0 development
    builds), a damping in units 1e-5 had a gradient and a Jacobian column
    1e-5 the size of the others': both tests called a direction the data
    determines undetermined, the hold put it back at its start, and the
    loss went from 1.6e-11 to 13.9.

    An identity coordinate with no size of its own -- start and selected
    value both exactly 0 -- has nothing to be relative to.  ``columns()``
    then gives each coordinate a magnitude that scales with its units
    (``||J[:, i]||``), and ``c`` makes that coordinate's scaled magnitude
    equal the largest of the others', which is invariant too.  A
    coordinate whose magnitude is 0 -- the objective does not read it --
    keeps ``c = 1``: every matrix the guard reads is zero in its row and
    column whatever ``c`` is, so nothing depends on it.  ``transformed``
    of ``None`` reads every coordinate as transformed: ``c`` is all ones
    and ``zeta`` is ``theta`` bit for bit.
    """
    t0 = np.abs(np.asarray(theta0, dtype=np.float64)).reshape(-1)
    ts = np.abs(np.asarray(selected, dtype=np.float64)).reshape(-1)
    c = np.ones_like(t0)
    if transformed is None:
        return c
    identity = ~np.asarray(transformed, dtype=bool).reshape(-1)
    ref = np.maximum(t0, ts)
    sized = identity & np.isfinite(ref) & (ref > 0.0)
    c[sized] = 1.0 / ref[sized]
    unsized = identity & ~sized
    if unsized.any() and columns is not None:
        mag = np.asarray(columns(), dtype=np.float64).reshape(-1)
        usable = np.isfinite(mag) & (mag > 0.0)
        others = usable & ~unsized
        if others.any():
            top = float(np.max(mag[others] / c[others]))
            fill = unsized & usable
            c[fill] = mag[fill] / top
    return c


def _rotation_by_singular_values(M, k: int):
    """``(W, s)``: ``M``'s right singular vectors as a ``k x k`` rotation,
    and its singular values padded with zeros to ``k`` -- an ``m x k``
    ``M`` with ``m < k`` maps the missing directions to nothing."""
    _, s, wt = np.linalg.svd(M, full_matrices=True)
    padded = np.zeros(k, dtype=np.float64)
    padded[:s.size] = s
    return wt.T, padded


def _no_curvature(method: str, k: int, why: str) -> None:
    """Warn that the curvature test could not run, and return ``None``.

    The caller then holds the candidates on the gradient test alone, still
    under the loss check -- the 0.4.0-dev guard, made safe.  The stack
    level reaches the fitter's caller through this function, the adapter,
    the fitter's closure, :func:`_hold_undetermined_directions` and the
    fitter.
    """
    warnings.warn(
        f"{method}: hold_undetermined could not test the {k} direction(s) "
        f"the run's gradients did not span against the objective's "
        f"curvature ({why}); they are checked against the loss alone.",
        RuntimeWarning, stacklevel=6,
    )
    return None


def _gauss_newton_flatness(J, candidates, dtype, method: str = "fit_lm",
                           scale: Optional[np.ndarray] = None):
    """The curvature test for :func:`fit_lm`, from ``J`` at the selected
    iterate.

    ``F = JᵀJ`` is the Gauss-Newton matrix Levenberg-Marquardt steps with
    and, weighted by ``noise_std``, the Fisher information :func:`fim`
    reports.  It is read in the guard's coordinates ``zeta = scale *
    theta`` (:func:`_relative_scale`), whose Jacobian is ``J / scale``
    column by column, and the candidates are directions of ``zeta``.  A
    direction ``u`` of the candidate span is flat when ``uᵀFu = ||Ju||²``
    is at or below ``rtol * max(eig(F))`` with :func:`fim`'s own rank
    cutoff ``rtol = max(n, sqrt(m)) * eps``, so :func:`fit_lm` holds a
    direction only if :func:`fim`, asked at the point it returned with its
    default ``scale="relative"``, would call it unresolved -- in any units.
    In ``theta`` itself (0.4.0 development builds), an identity coordinate
    in units 1e-5 had a column 1e-5 the size of the others' and was flat
    by this test however well the data determined it.

    The quadratic form rather than ``||Fu||``: ``J``'s entries carry the
    working precision's rounding, which leaves ``||Fv||`` of an exactly
    null ``v`` at about ``eps * max(eig(F))`` -- a factor of ``n`` from the
    cutoff -- while ``||Jv||²`` sits at ``eps²`` of it.  ``F`` is formed in
    float64 from the working-precision ``J``.

    ``None``, with a :class:`RuntimeWarning`, when ``J`` is not finite
    there: the loop checks the Jacobians it forms, but not one formed only
    for this test at an iterate the run accepted on its residual alone.
    """
    Jd = np.asarray(J, dtype=np.float64)
    m, n = Jd.shape
    k = candidates.shape[1]
    if not np.all(np.isfinite(Jd)):
        return _no_curvature(method, k, "the Jacobian at the selected iterate is not finite")
    if scale is not None:
        Jd = Jd / np.asarray(scale, dtype=np.float64)[None, :]
    W, s = _rotation_by_singular_values(Jd @ candidates, k)
    curvature = float(np.linalg.norm(Jd, 2)) ** 2
    rtol = _resolve_rank_rtol(dtype, n, None, n_residual=m)
    return W, s * s <= rtol * curvature, curvature


def _hessian_flatness(hvp, candidates, excited, dtype, method: str,
                      scale: Optional[np.ndarray] = None):
    """The curvature test for :func:`fit` and :func:`fit_multiple_shooting`.

    Their objective is a scalar, so there is no ``J`` to ask; the Hessian
    at the selected iterate is.  ``hvp(V)`` returns ``H V`` for the columns
    of ``V``, in ``theta``; the test reads the Hessian in the guard's
    coordinates ``zeta = scale * theta`` (:func:`_relative_scale`), which
    is ``H / outer(scale, scale)``, so each direction is divided by
    ``scale`` on the way in and its product on the way out, and the
    candidates are directions of ``zeta``.  A direction ``u`` of the
    candidate span is flat when
    ``||Hu|| <= sqrt(eps) * scale``, ``scale`` being the largest singular
    value of ``H`` over the candidates and the
    :data:`_HOLD_SCALE_DIRECTIONS` most-excited directions -- a lower bound
    on ``||H||``, so an underestimate makes the test stricter, not looser.

    ``sqrt(eps)`` rather than :func:`fim`'s ``max(n, sqrt(m)) * eps``
    because ``H`` is computed directly in the working precision, not as a
    product of two factors: an exact null direction's ``||Hv||`` is about
    ``eps * ||H||`` from rounding alone (measured 1e-8 to 2e-8 of the
    largest eigenvalue in float32 on the spring's scale degeneracy), and
    the number of terms the loss sums -- the ``sqrt(m)`` that would widen
    it -- is not visible through a scalar loss.  The norm rather than the
    quadratic form because the Hessian of a run that has not converged
    need not be positive: ``uᵀHu = 0`` along a saddle direction is not
    ``Hu = 0``.  The weakest direction the data resolves on the same
    problem measures 6e-2, so the cutoff has two decades of margin on
    the far side, and anything that slips under it must still pass the
    gradient test and the loss check.

    ``None`` -- no curvature test, the gradient test stands alone and the
    loss check still applies -- when the Hessian-vector product cannot be
    formed or is not finite.  The product differentiates the loss's
    backward pass forward, so an operation with a reverse rule and no
    forward one fails it: a ``jax.pure_callback`` inside a
    ``jax.custom_vjp``'s backward rule, for instance.  A
    :class:`RuntimeWarning` names the error.
    """
    k = candidates.shape[1]
    top = excited[:, ::-1][:, :_HOLD_SCALE_DIRECTIONS]
    V = np.concatenate([candidates, top], axis=1)
    inv = (None if scale is None
           else 1.0 / np.asarray(scale, dtype=np.float64)[:, None])
    try:
        HV = np.asarray(hvp(V if inv is None else V * inv), dtype=np.float64)
    except Exception as exc:   # noqa: BLE001 - any failure means "no curvature"
        return _no_curvature(
            method, k, f"its Hessian-vector product raised {type(exc).__name__}: {exc}")
    if not np.all(np.isfinite(HV)):
        return _no_curvature(method, k, "its Hessian-vector product is not finite")
    if inv is not None:
        HV = HV * inv
    curvature = float(np.linalg.norm(HV, 2))
    W, s = _rotation_by_singular_values(HV[:, :k], k)
    cutoff = math.sqrt(float(np.finfo(dtype).eps)) * curvature
    return W, s <= cutoff, curvature


def _hold_tolerance(loss_sel: float, grad_sel, curvature: float, held, eps: float,
                    *, scale: Optional[np.ndarray] = None,
                    floor: Optional[np.ndarray] = None) -> float:
    """How far the loss may rise for the hold to count as not raising it.

    ``2**10 * eps * |L| + ||g|| * d + curvature * d**2 / 2``, with ``L``
    and ``g`` the loss and gradient at the selected iterate, ``d`` the
    length of a perturbation of :data:`_HOLD_LOSS_ROUNDINGS` roundings in
    every coordinate ``u`` of the held point, and ``curvature`` the
    objective's largest curvature -- all three in the guard's coordinates
    ``zeta = scale * theta`` (:func:`_relative_scale`), where ``g`` is
    ``g / scale`` and a rounding of ``u`` has length ``scale *`` itself.
    One rounding is ``eps * max(floor, |u|)``: ``floor`` is 1 for a
    ``log`` / ``logit`` coordinate and 0 for an identity one (all 1 when
    ``floor`` is ``None``).

    The quantisation term pairs the largest curvature with the rounding of
    every coordinate, so it is only invariant to the parameters' units in
    coordinates that are.  In ``theta`` (0.4.0 development builds) it was
    not: a damping in units 1e-5 is a coordinate of 3e5, whose rounding
    the curvature of the others turned into a tolerance that admitted a
    loss of 13.9 against 1.6e-11; in units 1e6 it is 3e-6, and the floor of
    1 claimed a rounding of 4% of it, which admitted 0.1 against 3.1e-12.

    The guard moves only along directions both of its tests call flat, so
    in exact arithmetic the loss does not move at all; what the tolerance
    has to admit is rounding, of two kinds.

    The relative term is the loss *evaluated* at a different but
    equivalent point.  Every intermediate quantity rounds differently, and
    a least-squares loss turns a relative rounding ``eps`` of its model
    output into ``~eps * |y| / |r|`` of itself, ``y`` the signal and ``r``
    the residual.  Measured on the spring's scale degeneracy the hold moved
    the loss by 0 to 5.3 ``eps`` relative (``fit``, ``fit_lm`` and
    ``fit_multiple_shooting``, σ = 0.02); ``2**10`` admits a signal up to
    about a thousand times its residual, a fit to 0.1%.  It is 1.2e-4 in
    float32 and 2.3e-13 in float64.

    The absolute term is the *quantisation* of the held point: even an
    exactly flat move lands off the flat set by the rounding of every
    coordinate, and the loss pays for that to first and second order.  It
    is what a fit at its precision floor needs, where the relative term
    means nothing: noiselessly, ``fit_lm`` reached ``1.2e-14`` and the held
    point ``2.5e-13``, a factor of 20 that is all rounding.  A coordinate
    rounds when it is stored, when its transform maps it (``exp`` and the
    logistic round to ``eps`` of their result, which in ``log``
    coordinates is ``eps`` absolute -- hence the floor of 1 there, without
    which ``log 1 = 0`` would claim a denormal's rounding; an identity
    coordinate's value rounds relative to itself, so it has none), and
    again in the model's first use of it: four roundings.  Over the degenerate holds of
    the spring fixtures (``fit``, ``fit_lm`` and ``fit_multiple_shooting``,
    noiseless and σ = 0.02, 60 and 120 steps) the largest rise this term
    alone had to admit was 0.43 of *one* rounding per coordinate, so four
    leave a margin of 37.  It is deliberately not larger: a degeneracy that
    rotates in the optimiser's coordinates is held along its tangent, off
    the curved flat set, and what that costs is a real loss -- the spring
    1% off paid 4.5e-11 on jaxlib 0.11.0 and 2.0e-9 on 0.11.2, against a
    term of 4.4e-10 there, so one platform holds and the other declines.

    Against what the check exists to catch: on the four-parameter bowl the
    0.4.0-dev guard took ``fit_lm`` from 0.0 to 0.22, the spring 1% off from
    0.0 to 1.2e-4, and ten Adam steps from 0.32342 to 0.32367 -- a relative
    rise of 7.7e-4, six times the relative term.
    """
    held64 = np.asarray(held).astype(np.float64).reshape(-1)
    lower = 1.0 if floor is None else np.asarray(floor, dtype=np.float64).reshape(-1)
    rounding = eps * np.maximum(lower, np.abs(held64))
    c = (np.ones_like(held64) if scale is None
         else np.asarray(scale, dtype=np.float64).reshape(-1))
    d = _HOLD_LOSS_ROUNDINGS * float(np.linalg.norm(c * rounding))
    g = (0.0 if grad_sel is None
         else float(np.linalg.norm(np.asarray(grad_sel, dtype=np.float64).reshape(-1) / c)))
    return (_HOLD_LOSS_RTOL_EPS * eps * abs(loss_sel)
            + g * d + 0.5 * max(curvature, 0.0) * d * d)


def _hold_undetermined_directions(tracker, theta, theta0, objective: _SelectedObjective,
                                  method: str):
    """``(theta, excited_rank, undetermined_drift, hold_declined)``, with
    the undetermined component of ``theta - theta0`` removed when that does
    not raise the loss.

    The one implementation of the hold, shared by all three fitters: an
    optimiser-specific copy would let the three drift apart in exactly the
    quantity they exist to make reproducible.  Every step rule this module
    has moves along ``null(J)`` for its own reason -- Adam through its
    diagonal preconditioner, Levenberg-Marquardt through ``lam * diag(A)``,
    which is orthogonal to ``null(A)`` only where ``diag(A)`` is isotropic
    there -- and none of them is told anything about those directions by
    the data.

    A direction is undetermined when it passes two tests, each necessary
    for a direction the data cannot see and neither sufficient alone:

    1. **No gradient of the run pointed along it**
       (:class:`_ExcitationTracker`).  Alone, this is what 0.4.0
       development builds used, and on a short or fast-converging run it
       names directions the data determines perfectly well.
    2. **The objective has no curvature along it at the selected
       iterate** -- ``JᵀJ`` for :func:`fit_lm`
       (:func:`_gauss_newton_flatness`), the loss's Hessian for the other
       two (:func:`_hessian_flatness`).  Alone, it would hold a degeneracy
       that rotates in the optimiser's coordinates along its local
       tangent, off the curved flat set; requiring test 1 too keeps such a
       run, whose gradients span everything, untouched and reported full
       rank, as before.

    Then the **loss check**: the held point's loss, by the fitter's own
    compiled function, must not exceed the selected iterate's by more than
    :func:`_hold_tolerance`.  If it does, nothing is held, ``theta`` comes
    back as the selected iterate bit for bit, ``hold_declined`` is True and
    a :class:`RuntimeWarning` gives both losses.  The hold is all or
    nothing: the undetermined directions form a subspace, and when its
    curvature is degenerate any basis of it is as good as another, so
    "hold some of them" names no particular subset.

    **Every test, the hold and the tolerance are made in the coordinates
    ``zeta = c * theta`` of :func:`_relative_scale`**: a ``log`` or
    ``logit`` coordinate as it is, an identity one relative to the
    parameter's own size.  Each of them compares coordinates with each
    other, so each is invariant to the parameters' units in ``zeta`` and
    in no coordinates that are not; in ``theta`` (0.4.0 development
    builds) a damping in units 1e-5 or 1e6 was held, against data that
    determined it, at a loss of 13.9 or 0.1 where the fit had reached
    1e-11, and declined with a warning in units 1e-4 to 1e-2 and 1e4.  The
    hold removes the undetermined component of the move orthogonally in
    ``zeta``, which holds the same combination of the parameters at its
    start whatever their units.  :attr:`FitResult.undetermined_drift` is
    still reported in ``theta``, the optimiser's coordinates, as
    documented.

    ``theta`` comes back untouched, and therefore bit for bit, whenever no
    direction passes both tests or the question could not be answered.
    When every coordinate is a ``log`` / ``logit`` one ``zeta`` is
    ``theta``, and when every candidate of test 1 also passes test 2 the
    hold is then the 0.4.0 development builds' formula, the projector onto
    the spanned directions.  (Its eigenvectors now come from the singular
    value decomposition of the gradients' factor rather than an ``eigh`` of
    their Gram matrix -- see :class:`_ExcitationTracker` -- so they agree
    with the older builds' to rounding, not to the bit.)
    """
    if tracker is None:
        return theta, None, None, None
    c = _relative_scale(theta0, theta, objective.transformed, objective.columns)
    split = tracker.split(c)
    if split is None:
        return theta, None, None, None
    evecs, excited = split
    n = tracker.n
    if bool(excited.all()):
        return theta, n, 0.0, False
    candidates, spanned = evecs[:, ~excited], evecs[:, excited]
    k = candidates.shape[1]
    probe = objective.flatness(candidates, spanned, c)
    if probe is None:
        W, flat, curvature = np.eye(k), np.ones(k, dtype=bool), 0.0
    else:
        W, flat, curvature = probe
        flat = np.asarray(flat, dtype=bool)
    n_flat = int(np.count_nonzero(flat))
    if n_flat == 0:
        return theta, n, 0.0, False
    moved = (np.asarray(theta, dtype=np.float64)
             - np.asarray(theta0, dtype=np.float64))
    moved_z = c * moved
    if n_flat == k:
        # Every candidate is flat: the projector onto the spanned
        # directions, formed and applied as 0.4.0-dev formed it.
        kept = ((spanned @ spanned.T) @ moved_z) / c
    else:
        undetermined = candidates @ W[:, flat]
        kept = (moved_z - undetermined @ (undetermined.T @ moved_z)) / c
    drift = float(np.linalg.norm(moved - kept))
    # ``theta0 + kept``, not ``kept`` alone: the guard holds the
    # undetermined directions at the values they *started* at, which
    # is the one thing about them the data has not contradicted.
    held = theta0 + jnp.asarray(kept, dtype=theta0.dtype)
    loss_sel, grad_sel = objective.reference()
    loss_held = objective.loss(held)
    floor = (None if objective.transformed is None
             else np.where(np.asarray(objective.transformed, dtype=bool), 1.0, 0.0))
    tol = _hold_tolerance(loss_sel, grad_sel, curvature, held, tracker.eps,
                          scale=c, floor=floor)
    if math.isfinite(loss_held) and loss_held <= loss_sel + tol:
        return held, n - n_flat, drift, False
    warnings.warn(
        f"{method}: hold_undetermined found {n_flat} direction(s) the data "
        f"does not determine at the returned point, but holding them at "
        f"their starting values would raise the loss from {loss_sel:.9g} to "
        f"{loss_held:.9g}, beyond the guard's tolerance of {tol:.3g}; "
        f"params is the selected iterate, unheld (FitResult.hold_declined). "
        f"The loss is flat along them where the fit ended but not on the way "
        f"back to the start -- a degeneracy that curves in the optimiser's "
        f"coordinates, for instance.",
        RuntimeWarning, stacklevel=3,
    )
    return theta, n - n_flat, drift, True


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


class _BestIterate:
    """The lowest-loss iterate a fitter has evaluated so far.

    Shared by :func:`fit` and :func:`fit_multiple_shooting`, so that both
    choose the iterate they return by the same rule.  ``iteration`` counts
    the updates that produced ``state``, which makes it an index into the
    fitter's ``losses`` (``losses[k]`` is the loss of the iterate after
    ``k`` updates).

    A tie goes to the **later** iterate.  That is what keeps the result
    bit-identical to the last iterate, which is what these fitters
    returned before, whenever the loss never rose above an earlier value:
    a run that is strictly or weakly decreasing returns exactly what it
    returned before.  A non-finite loss is never selected; it is not
    lower than anything.
    """

    def __init__(self, state) -> None:
        self.state = state
        self.iteration = 0
        self.loss: Optional[float] = None

    def offer(self, state, iteration: int, loss: float) -> None:
        """Keep ``state`` if ``loss`` is at or below the best so far."""
        if not math.isfinite(loss):
            return
        if self.loss is None or loss <= self.loss:
            self.state, self.iteration, self.loss = state, int(iteration), loss


def _offer_final_iterate(best: _BestIterate, state, n_updates: int,
                         loss: float, method: str) -> None:
    """Compare the iterate the last Adam update produced with the rest.

    The loop evaluates an iterate *before* stepping from it, so when it
    ends by exhausting ``n_iter`` the last update's result is the one
    point it never evaluated.  The caller evaluates it once more, with the
    same compiled function the loop used, so that the comparison is
    between numbers computed the same way.  A non-finite loss there is
    not raised -- nothing is stepped from that point, so nothing forces
    the run to stop -- but that iterate is not returned either, and a
    :class:`RuntimeWarning` names the iterate returned in its place.
    """
    if not math.isfinite(loss):
        warnings.warn(
            f"{method}: the last update produced a non-finite loss ({loss}); "
            f"returning iterate {best.iteration}, the lowest-loss one evaluated "
            f"(FitResult.best_iteration).",
            RuntimeWarning, stacklevel=3,
        )
        return
    best.offer(state, n_updates, loss)


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
    would read as converged.  No field has been inserted yet (the two
    added during 0.4.0 were appended) and no caller built one
    positionally; ``kw_only`` is what keeps that true for the next field.

    ``params`` is a physical pytree (already mapped back through
    ``GraphManager.constrain``); ``losses[i]`` is the loss *before*
    update ``i + 1`` -- the loss of the iterate ``i`` updates produced, so
    ``losses[0]`` is the starting point's.

    ``converged``
        Whether the run stopped because a stopping test was met, rather
        than because it used its whole ``n_iter`` or (:func:`fit_lm`)
        because no damping could make a step acceptable.  The tests:

        * the loss reached ``tol``: ``losses[-1] <= tol``, tested before
          each update, by all three fitters.  ``tol=0.0``, the default,
          turns this test off, and for :func:`fit` and
          :func:`fit_multiple_shooting` it is the only one -- so with the
          default ``tol`` their ``converged`` is always ``False``, and a
          fit's quality is read from ``best_loss`` and ``losses``, not
          from this flag;
        * :func:`fit_lm` only: the step an iteration *proposed* (its first
          candidate) moved every trainable parameter by no more than
          ``step_tol`` of its own magnitude -- by default ``2**4`` ulps of
          its dtype, the parameters' float resolution.  The proposal
          counts whether or not it lowered the loss, because at the float
          floor it rounds to (nearly) nothing and cannot strictly lower a
          loss that is rounding noise.  Then ``losses[-1]`` is the loss of
          the iterate the proposal was made from;
        * :func:`fit_lm` only, the floor rule: after the run has lowered the
          loss at least once, an iteration rejected every candidate down to
          one damped within ``step_tol`` -- no step the tolerance resolves
          lowers the loss any more.

        Neither :func:`fit_lm` test fires while a parameter held on its
        bound (``transform=None`` with ``bounds``) could lower the loss by
        moving into its range by more than ``step_tol`` (its own Newton
        step; a smaller pull is rounding): that point is not a constrained
        stationary point, however small the step proposed there.

        It says the iteration stopped moving, not that the data determined
        the parameters: a residual that reads none of the trainable
        parameters converges at its start (its proposal is exactly zero).
        :attr:`excited_rank` and :func:`fim` answer the second question.

    ``params`` is the **lowest-loss iterate the fitter evaluated**, not
    necessarily its last; ``best_iteration`` says which.  Earlier 0.4.0
    development builds returned the last iterate, and an optimiser with a
    fixed step size can end a run above where it started: Adam's step is
    about ``lr`` in size whatever the gradient's, so from a point whose
    gradient is rounding noise it walks away.  One such fit started at loss
    ``2.7e-8`` and returned parameters at ``1.9e-4``, and nothing in the
    result said so.  A run whose loss never rose above an earlier value --
    every well-behaved fit -- returns what it returned before, bit for
    bit: a tie goes to the later iterate.

    ``best_iteration``
        How many updates produced ``params``; ``0`` is the start.  It
        indexes ``losses``, so ``losses[best_iteration] == best_loss`` --
        except when it equals ``len(losses)``.  Then ``params`` is the
        iterate the *last* update produced, which the loop never evaluated
        (it evaluates before it steps), and the fitter evaluated it once
        after the loop to compare it with the rest.  That evaluation is
        the selection's only cost: one call of :func:`fit`'s or
        :func:`fit_multiple_shooting`'s compiled loss-and-gradient, made
        only when the run used its whole budget.  It is not appended to
        ``losses`` and not passed to ``callback`` or the observers.
        :func:`fit_lm` never needs it: it accepts a step only when the
        step lowers the loss, so its last iterate is always its lowest,
        and it already holds that iterate's loss.  ``None`` only on a
        ``FitResult`` no fitter built.
    ``best_loss``
        That iterate's loss as the fitter evaluated it: ``loss_fn``'s
        value for :func:`fit`, the (weighted) ``0.5 ||r||²`` for
        :func:`fit_lm`, and the windowed loss with its continuity penalty,
        at the returned window states, for :func:`fit_multiple_shooting`.
        It is the value *before* the ``hold_undetermined`` guard below,
        which moves ``params`` only along directions it found the loss to
        be flat in, and only when the moved point's loss is within the
        guard's tolerance of this one (a relative ``2**10 * eps``, 1.2e-4 in
        float32, plus the loss cost of rounding the moved point; see
        ``hold_declined``).  Where the guard holds nothing it is the loss of
        exactly ``params``, bit for bit, because there is one evaluation:
        the fitters map their coordinates to physical parameters once, by
        one compiled function, run the model on the arrays it returns, and
        hand those arrays back.  A leaf no step moved is in them as the
        value that went in, never as its ``constrain(unconstrain(p))`` round
        trip.  (Until 0.4.0's fix the map was evaluated twice -- inside the
        jitted objective and again, eagerly, for the result -- and the two
        landed on different floats: an ulp apart for a ``log`` leaf, and
        under a ``logit`` whose bounds are wide beside the value, far enough
        that ``best_loss`` read 1.3e-8 for parameters whose loss was 4.5e-10,
        MADD-ANO-178.)  The value is the fitter's own program's: a loss you
        compute yourself from ``params`` agrees with it to rounding, not to
        the bit.  ``None`` when nothing was evaluated (``n_iter=0``).

    Every leaf no step moved is the value that went in, bit for bit --
    not merely close -- so comparing a fit's input and output leaf by
    leaf says exactly which constants the calibration touched.  That
    covers the leaves outside the resolved mask and the masked ones a
    run that stopped before its first update never stepped: neither is
    round-tripped through ``constrain(unconstrain(p))``, which for a
    ``log`` leaf is ``exp(log(p))`` and lands one ulp away.  **And it is
    the value the model was run at**: a leaf the fit does not move is
    never passed through its transform, whatever that is, in the result
    or in any evaluation of the objective.

    A fitted ``log`` / ``logit`` leaf is returned at the resolution of its
    transform, which is set by its bounds and not by its value
    (:meth:`~maddening.core.params.ParamSpec` rebuilds it as ``lo + ...``:
    ``eps * max(|lo|, |hi|, hi - lo)`` for ``logit``, ``eps * |lo|`` for
    ``log``).  Where that leaves fewer than half the working precision's
    digits of a value -- more than ``sqrt(eps)`` of it: a float32 ``3.0``
    under ``logit`` bounds 1e4 wide -- the fitters say so in a
    :class:`~maddening.warnings.PrecisionLimitWarning` naming the leaf, at
    their start (and again for a leaf that only ends up so); tighten the
    bounds or declare the leaf with ``transform=None``.

    ``excited_rank``, ``undetermined_drift`` and ``hold_declined`` report
    the identifiability guard (``hold_undetermined``), which all three
    fitters run, and are ``None`` from one that could not answer the
    question.

    ``excited_rank``
        How many independent directions the data determines, as far as the
        run and the returned point show it, out of the trainable coordinate
        count.  A direction counts as *undetermined* -- and the rank is the
        count minus their number -- only if it passes two tests: no
        gradient the run evaluated pointed along it, **and** the
        objective has no curvature along it at the selected iterate
        (``JᵀJ`` for :func:`fit_lm`, by :func:`fim`'s rank rule; the
        loss's Hessian for :func:`fit` and :func:`fit_multiple_shooting`).
        Both are asked with each identity-transform parameter measured
        relative to its own size, as :func:`fim`'s default
        ``scale="relative"`` measures it, so the count is the same in any
        units (in the optimiser's own coordinates a damping in units 1e-4
        read as undetermined, 2 of 3).
        Earlier 0.4.0 development builds counted the first test alone,
        which on a short or fast-converging run names directions the data
        determines perfectly well: ``fit_lm`` on a four-parameter bowl
        reported 1 of 4 where every direction was determined.  Less than
        the count means the data left the rest undetermined and ``params``
        holds them at the values they started at -- unless
        ``hold_declined``.  ``None`` is **not** "full rank": it is "not
        measured" -- ``hold_undetermined=False``; more trainable
        coordinates (array elements, not leaves) than
        :data:`_EXCITATION_MAX_PARAMS`; fewer gradients than trainable
        coordinates, where an unobserved direction cannot be told from an
        unobservable one; or a degenerate spectrum -- every gradient the
        run saw was exactly zero (a start at an exact optimum, or a loss
        that reads none of the trainable parameters), so there is no
        largest direction to measure the others against.
    ``undetermined_drift``
        How far the selected raw iterate had wandered along those
        undetermined directions, as a Euclidean norm in the
        **unconstrained** coordinates (``log`` for a positive parameter,
        so a drift of 0.04 there is a 4% drift in the parameter itself).
        ``0.0`` when the rank was full and nothing was removed; ``None``
        when ``excited_rank`` is.  It is a diagnostic, not a residual
        error: unless ``hold_declined``, the value it reports has already
        been taken out of ``params``.  A number far above the fit's own
        step scale says the loss surface has a flat direction worth naming
        with :func:`fim`.

        For :func:`fit_multiple_shooting` it covers the **parameter** block
        only.  The window starts are that fit's own decision variables and
        are returned unguarded, from the same iterate as ``params``, in
        the second element of its result rather than in ``params``.
    ``hold_declined``
        ``True`` when the guard found undetermined directions but holding
        them would have raised the loss beyond its tolerance, so it held
        nothing: ``params`` is then the selected iterate exactly as the
        optimiser produced it, bit for bit, still carrying
        ``undetermined_drift``, and a :class:`RuntimeWarning` gave both
        losses.  The tolerance is ``2**10 * eps`` of ``best_loss`` (1.2e-4
        relative in float32) plus what rounding the held point to the
        working precision can cost -- rounding, not progress: the
        uphill moves it exists to refuse were 0.0 to 0.22 and 0.0 to
        1.2e-4.  It happens when the loss is flat along a direction where
        the fit ended but not on the way back to the start, which a
        degeneracy that curves in the optimiser's coordinates can do.
        ``False`` when the guard held what it found, or found nothing;
        ``None`` when ``excited_rank`` is.
    """
    params: dict
    losses: np.ndarray
    converged: bool
    n_iter: int
    excited_rank: Optional[int] = None
    undetermined_drift: Optional[float] = None
    best_iteration: Optional[int] = None
    best_loss: Optional[float] = None
    hold_declined: Optional[bool] = None

    def __str__(self) -> str:
        """A short human summary: whether the fit converged, its losses,
        which iterate it returned, the identifiability guard's verdict and
        the fitted leaves (the first ten).  ``repr`` is the dataclass's
        own, field by field."""
        losses = np.asarray(self.losses, dtype=np.float64).reshape(-1)
        lines = [f"FitResult: {'converged' if self.converged else 'not converged'} "
                 f"after {int(self.n_iter)} iteration{'s' * (int(self.n_iter) != 1)}"]
        if losses.size:
            loss = (f"  loss: first {_summary_number(losses[0])}, "
                    f"last {_summary_number(losses[-1])}")
            if self.best_loss is not None:
                loss += f", best {_summary_number(self.best_loss)}"
                if self.best_iteration is not None:
                    loss += f" (iterate {int(self.best_iteration)})"
            lines.append(loss)
        else:
            lines.append("  loss: none evaluated")
        if self.excited_rank is None:
            lines.append("  identifiability guard: not measured")
        else:
            guard = f"  identifiability guard: excited rank {int(self.excited_rank)}"
            if self.undetermined_drift is not None:
                guard += f", undetermined drift {_summary_number(self.undetermined_drift)}"
            if self.hold_declined:
                guard += " (hold declined: params are the raw iterate)"
            lines.append(guard)
        try:
            leaves = jax.tree_util.tree_flatten_with_path(self.params)[0]
        except Exception:   # noqa: BLE001 - a summary never raises
            leaves = []
        if leaves:
            lines.append(f"  params ({len(leaves)} lea{'f' if len(leaves) == 1 else 'ves'}):")
            for path, leaf in leaves[:10]:
                name = jax.tree_util.keystr(path)
                shape = tuple(np.shape(leaf))
                if math.prod(shape) != 1:
                    text = f"array{shape}"
                else:
                    try:
                        text = _summary_number(np.asarray(leaf).reshape(()))
                    except Exception:   # noqa: BLE001 - a tracer has no value
                        text = "(no value)"
                lines.append(f"    {name} = {text}")
            if len(leaves) > 10:
                lines.append(f"    ... and {len(leaves) - 10} more")
        return "\n".join(lines)


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

    A ``log`` / ``logit`` coordinate is kept where its transform can be
    stepped on: at least ``sqrt(eps)`` of the bounds' own size inside each
    bound (3.5e-4 of the range in float32, 1.5e-8 in float64;
    :meth:`ParamSpec._usable_margin`).  Nearer a bound than that the
    transform is flat to the working precision.  So such a parameter cannot
    be fitted *onto* its bound: a fit that ends on that margin -- its data
    pull the parameter to the bound, or it started out there and could not
    come back within its budget -- says so in a :class:`RuntimeWarning`
    naming the parameter.  Declare a parameter that belongs on its bound
    with ``transform=None``, which clips.  Adam's update is about ``lr`` in
    the coordinate whatever the gradient, and on that margin the transform
    is flat, so a parameter *started* there is named before the run as
    well: the value does not move visibly for about ``8 / lr`` updates in
    float32 and ``18 / lr`` in float64 -- longer while the coordinate's
    gradient is below ``eps`` of the largest -- where :func:`fit_lm`, whose
    step is read on the transform's tangent, leaves the edge at once.  And
    an optimum *on* a bound is approached without being reached (Adam's
    step shrinks with the gradient it remembers): such a run ends short of
    the margin, on no edge, and is not named.

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
        zero, so the fit could only hand it back unchanged.  Every leaf
        must be a bool (a NumPy ``bool_`` or a 0-d boolean array will do),
        and anything else is refused (``ValueError``): a leaf is read as a
        flag, and a string ``"False"`` used to select its parameter.
    n_iter, lr, tol, betas, eps
        Adam hyper-parameters; ``tol > 0`` stops early once the loss is
        at or below it.  Each is a real number -- a ``bool`` or a string
        is refused (``ValueError``), as a value with no reading is.  ``eps`` is *relative*: it floors Adam's
        denominator in units of the run's first gradient (its largest entry,
        rounded to a power of two), so the iterates do not depend on the
        units the loss is written in.  Earlier 0.4.0 development builds
        applied it in the loss's own units, and a loss near ``1e-12`` (a
        position residual in metres at micrometre scale, squared) did not
        move its parameters at all.  Smaller still -- a gradient whose
        products flush below ``tiny``, ``0.5 * ||s * r||²`` at ``s = 1e-19`` --
        the gradients are taken with a power-of-two cotangent that lifts the
        backward pass out of the flush, exactly (:func:`_gradient_lift`);
        until 0.4.0's fix the gradient read zero there and the fit returned
        its start (MADD-ANO-174).  ``loss_fn``'s own value is the caller's
        and can still flush to ``0.0``: a ``RuntimeWarning`` says so once,
        and such a ``0.0`` with a nonzero gradient does not meet ``tol``.
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
        0.4.0, and on by default**; ``False`` returns the selected iterate
        (see Returns) exactly as Adam produced it.

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
        budget.  Those figures are for the *last* iterate, which earlier
        development builds returned.  Returning the lowest-loss iterate
        does not remove the effect, because along an exactly flat direction
        which iterate is lowest is decided by rounding.  Unguarded, on a
        120-step record with σ = 0.02 noise, the lowest-loss iterate lands
        −7.50%, −7.12% and −5.43% from the starting scale at ``lr`` 0.01,
        0.05 and 0.2, for every budget from 200 to 10,000 iterations: the
        learning rate still picks the answer.  :func:`fit_lm` and
        :func:`fit_multiple_shooting` drift too
        — ``λ·diag(A)`` damping is no more orthogonal to the null space
        than Adam's preconditioner is — and since 0.4.0 all three run this
        same guard.  What differs is that LM's step vanishes with the
        gradient, so its drift *converges*; see :func:`fit_lm`.

        With the guard on, :func:`fit` accumulates ``Σ_t g_t g_tᵀ`` over the
        run; that matrix's numerically-null eigenvectors are the directions
        no gradient pointed along.  That is necessary for a direction the
        data cannot determine and **not sufficient**: a short run's
        gradients need not span the directions the data does determine, so
        each candidate is also put to the loss's Hessian at the selected
        iterate (``k`` Hessian-vector products for ``k`` candidates, plus
        up to 8 more for scale), and only a direction with no curvature
        there either is held.  The net displacement's component along
        those is removed — and then the moved point's loss is evaluated,
        and if it is above the selected iterate's by more than the guard's
        tolerance (``2**10·eps`` relative, plus the cost of rounding the
        moved point) nothing is held, :attr:`FitResult.hold_declined` is
        ``True`` and a :class:`RuntimeWarning` says so.  So **the loss is
        unaffected**, beyond rounding, whatever the guard decides.  The
        iterates, ``losses``, ``callback`` and observer events are
        unchanged, and a fit with no direction passing both tests gets its
        iterate back bit for bit — so a well-posed fit sees no difference at
        all.  :attr:`FitResult.excited_rank`,
        :attr:`FitResult.undetermined_drift` and
        :attr:`FitResult.hold_declined` say what the guard found.  Earlier
        0.4.0 development builds held every direction the gradients had not
        spanned and checked nothing: ten Adam steps at ``lr=0.01`` on a
        well-posed four-parameter bowl came back above the iterate the fit
        selected.

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

    Returns
    -------
    FitResult
        ``params`` is the **lowest-loss iterate** the run evaluated, with
        :attr:`FitResult.best_iteration` and :attr:`FitResult.best_loss`
        saying which and at what loss.  Adam's step is about ``lr`` in
        size however small the gradient, so a run started at (or passing
        through) a point whose gradient is rounding noise can end above it;
        earlier 0.4.0 development builds returned the last iterate
        regardless.  Ties go to
        the later iterate, so a run whose loss never rose returns its last
        iterate, bit for bit what it returned before.  When the run used
        its whole budget, the iterate the last update produced is
        evaluated once more (not appended to ``losses``, not reported to
        ``callback``) so that it can be compared with the rest; a run
        stopped by ``tol`` returns the iterate that met it, which is the
        lowest by construction.
    """
    n_iter, notify_every = _check_adam_hyper(n_iter, lr, tol, betas, eps, notify_every)
    _check_hold_undetermined(hold_undetermined)
    _sync_compiled(gm)
    start = gm._params_or_default(params)  # noqa: SLF001
    gm.check_params(start)
    mask = _resolve_mask(gm, start, mask)
    b1, b2 = betas
    progress = _progress_notifier(gm, "adam", n_iter, notify_every)

    u0 = gm.unconstrain(start)
    flat_u, _ = ravel_pytree(u0)
    idx = _masked_indices(start, mask)
    if idx is None:
        idx = np.arange(flat_u.size)
    # The one evaluation of ``theta -> params`` (SYS-071): the loss is a
    # compiled function of the physical tree ``pmap.params`` returns, and
    # that tree is what ``callback`` receives and the result holds.
    pmap = _PhysicalMap(gm, start, flat_u, idx)
    theta0 = flat_u[idx]
    plain, scaled = _model_loss_and_gradient(pmap, loss_fn)
    # The cotangent the run takes its gradients with once a flushed first
    # gradient made it lift them (:func:`_gradient_lift`); ``None`` -- the
    # plain gradient -- otherwise.
    lift = None

    def evaluate(th, p=None):
        """``(loss, gradient)`` at ``th``, the gradient times ``lift``: the
        model's loss and gradient at the physical tree ``p`` (``pmap.params``
        of ``th``), the gradient chained with the map's own slope."""
        p = pmap.params(th) if p is None else p
        loss, g_p = plain(p) if lift is None else scaled(p, lift)
        return loss, g_p * pmap.slope(th)

    def unlifted(g) -> np.ndarray:
        """A gradient from :func:`evaluate`, in float64 and the loss's units."""
        g64 = np.asarray(g, dtype=np.float64)
        return g64 if lift is None else g64 / float(lift)

    bounds = _CoordinateBounds(gm, start, idx, theta0.dtype)
    unresolved: set = set()
    _warn_unresolved_values("fit", pmap, start, "at the start of the fit", unresolved)
    if n_iter > 0:
        _warn_starts_on_an_edge("fit", pmap, bounds, theta0, start, lr, eps)

    @jax.jit
    def adam_step(theta, m, v, g, frame, i):
        # The gradient in the run's frame (``_adam_frame``): ``eps`` below is
        # then relative to the first gradient, not absolute in the loss's units.
        g = g * frame
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** i)
        v_hat = v / (1 - b2 ** i)
        # Back onto the range of each coordinate (``_CoordinateBounds``): past
        # its bound a clipped coordinate has no gradient and would stay
        # clipped, and a ``log`` / ``logit`` one is kept where its transform
        # still resolves the distance to the bound.
        return bounds.project(theta - lr * m_hat / (jnp.sqrt(v_hat) + eps)), m, v

    theta = theta0
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    frame = None
    coarse = _coarsest_dtype(start, idx)
    tracker = _make_excitation_tracker(hold_undetermined, theta0, coarse)
    best = _BestIterate(theta0)
    # The physical tree the best iterate was evaluated on: the result's.
    best_params: Optional[dict] = None
    losses: list[float] = []
    converged = False
    vanished_warned = False
    i = 0
    for i in range(1, n_iter + 1):
        p = pmap.params(theta)
        loss, g = evaluate(theta, p)
        if i == 1:
            # A first gradient in the flush is evaluated again, lifted, and
            # the run takes every gradient that way (``_gradient_lift``).
            slope0 = pmap.slope(theta)
            lift = _gradient_lift(
                lambda cot: (lambda out: (out[0], out[1] * slope0))(scaled(p, cot)),
                (g,), loss)
            if lift is not None:
                loss, g = evaluate(theta, p)
        loss_f = float(loss)
        losses.append(loss_f)
        if not np.isfinite(loss_f) or not bool(jnp.all(jnp.isfinite(g))):
            raise FloatingPointError(
                f"non-finite loss or gradient at iteration {i} (loss={loss_f})"
            )
        # ``theta`` here is the iterate ``i - 1`` updates produced.
        best.offer(theta, i - 1, loss_f)
        if best.state is theta:
            best_params = p
        if tracker is not None:
            # Before the ``tol`` break, not after the step: this gradient is
            # information about the loss surface whether or not it moved
            # anything, and a run that stops on ``tol`` has still seen it.
            tracker.observe(unlifted(g))
        if callback is not None or progress is not None:
            current = pmap.returned(theta, p)
            if callback is not None:
                callback(i, loss_f, current)
            if progress is not None:
                progress(i, loss_f, current)
        # A loss of exactly 0.0 with a gradient that is not zero: ``loss_fn``
        # flushed its own value (``r * r`` below ``tiny``).  It is not a loss
        # that reached ``tol``, and the run says so once.
        vanished = loss_f == 0.0 and bool(jnp.any(g != 0.0))
        if vanished and not vanished_warned:
            vanished_warned = True
            warnings.warn(
                f"fit: loss_fn returned exactly 0.0 at iteration {i} with a "
                f"gradient that is not zero -- the loss underflows its "
                f"precision (a squared residual below the smallest normal "
                f"number flushes to 0).  The iterates follow the gradient, "
                f"which is taken clear of the flush, but losses, best_loss and "
                f"the tol test read 0.0 there, and tol does not count it. Write "
                f"the loss in units nearer one.",
                RuntimeWarning, stacklevel=2)
        if tol > 0.0 and loss_f <= tol and not vanished:
            converged = True
            break
        if frame is None:
            frame = _adam_frame(g)
        theta, m, v = adam_step(theta, m, v, g, frame, jnp.asarray(i, theta.dtype))

    if i > 0 and not converged:
        # The budget ran out, so the last thing the loop did was an update
        # whose result it never evaluated.  ``evaluate`` rather than a
        # value-only function: the same compiled arithmetic as every entry
        # of ``losses``, so the comparison is like for like and costs no
        # compile.  The gradient is discarded -- in particular it is not
        # folded into the tracker, which would change ``excited_rank`` for
        # runs whose result is otherwise untouched.
        p_last = pmap.params(theta)
        _offer_final_iterate(best, theta, i, float(evaluate(theta, p_last)[0]), "fit")
        if best.state is theta:
            best_params = p_last

    last = theta
    selected = best.state
    selected_params = pmap.params(selected) if best_params is None else best_params

    def _reference():
        loss_sel, g_sel = evaluate(selected, selected_params)
        return float(loss_sel), unlifted(g_sel)

    def _flatness(candidates, spanned, scale):
        # Hessian-vector products of the lifted gradient are lifted by the
        # same power of two; the test is relative, the curvature it returns
        # (for the hold's tolerance) is not, so it is unlifted.
        lifted = 1.0 if lift is None else float(lift)

        def gradient(p_):
            return (plain(p_) if lift is None else scaled(p_, lift))[1]

        return _hessian_flatness(
            lambda V: _model_hvp(gradient, pmap, selected, selected_params, (), V) / lifted,
            candidates, spanned, coarse, "fit", scale)

    theta, excited_rank, undetermined_drift, hold_declined = _hold_undetermined_directions(
        tracker, selected, theta0,
        _SelectedObjective(loss=lambda th: float(evaluate(th)[0]),
                           reference=_reference, flatness=_flatness,
                           transformed=bounds.transformed,
                           columns=None if tracker is None else tracker.gradient_scale),
        "fit")

    # The tree the selected iterate was evaluated on, unless the guard moved it.
    evaluated = selected_params if theta is selected else pmap.params(theta)
    if i > 0:
        _warn_on_an_edge("fit", pmap, bounds, theta, evaluated, last)
        _warn_unresolved_values("fit", pmap, evaluated, "as fitted", unresolved)
    return FitResult(
        params=pmap.returned(theta, evaluated), losses=np.asarray(losses),
        converged=converged, n_iter=i,
        excited_rank=excited_rank, undetermined_drift=undetermined_drift,
        best_iteration=best.iteration, best_loss=best.loss,
        hold_declined=hold_declined,
    )


def _column_frame(J):
    """One power of two per column of ``J`` bringing its largest entry into
    ``[0.5, 1)`` (:func:`~maddening.core._pow2_frame.pow2_frame`,
    ``"entrywise"``); ``1`` for a column of zeros."""
    return pow2_frame(jnp.max(jnp.abs(J), axis=0), mode="entrywise")


def _half_squared_norm(r) -> float:
    """``0.5 * ||r||²`` as a Python float, computed on ``r`` framed by a
    power of two (its largest entry into ``[0.5, 1)``) and unframed in
    float64.

    :func:`fit_lm`'s loss.  ``r * r`` in the residual's own dtype flushes to
    zero below ``sqrt(tiny)`` -- ``1.1e-19`` in float32 -- and overflows
    above ``sqrt(max)``, ``1.8e19``: a residual written in small units (a
    particle volume in cubic metres) had every loss ``0.0``, so no step
    could lower it, and a large one had an infinite loss.  Framed, the sum
    is of normal numbers at any scale, and the power of two makes it the
    same number, bit for bit, wherever the bare sum was normal.  The
    unframing is in float64, which holds it for any float32 residual; a
    float64 residual below about ``1e-162`` still underflows there, and
    :func:`fit_lm` refuses to call that converged (``r`` is not zero).
    """
    if r.size == 0:
        return 0.0
    f = pow2_frame(r)
    framed = r * f
    scale = float(f)
    return 0.5 * float(jnp.sum(framed * framed)) / scale / scale


def _framed_gradient(J, r) -> np.ndarray:
    """``Jᵀr`` in float64, from ``J`` and ``r`` framed by powers of two
    (every column's largest entry, and the residual's, into ``[0.5, 1)``)
    and unframed in float64.

    The gradient :func:`fit_lm` hands its identifiability guard and its
    on-bound test.  Formed bare in the working precision its products
    ``J_ij r_i`` flush below ``tiny`` -- a residual of ``1e-19`` with a
    Jacobian of ``1e-19`` read a gradient of exactly zero -- and the
    frames make it the same number, bit for bit, wherever the bare
    products were normal.
    """
    c = _column_frame(J)
    f = pow2_frame(r)
    g = (J * c[None, :]).T @ (r * f)
    # One frame at a time: their product can leave float64's range (a
    # float64 residual near 1e-170) where each frame, and the result, is in it.
    return np.asarray(g, dtype=np.float64) / np.asarray(c, dtype=np.float64) / float(f)


@jax.jit
def _marquardt_step(th, r, J, lam, lo, hi, held):
    """One Levenberg-Marquardt candidate from ``th``, projected onto
    ``[lo, hi]`` (:class:`_CoordinateBounds`), and whether every column it
    solved for was representable.

    Module-level, so its compiled form is shared by every :func:`fit_lm`
    call of the same shapes rather than compiled again per call.

    :func:`fit_lm` calls it in the tangent frame
    (:meth:`_CoordinateBounds.tangent_frame`): ``th``, ``lo`` and ``hi`` are
    the coordinates and their bounds for an identity coordinate, and for a
    ``log`` / ``logit`` one the origin 0 and the box of steps over which the
    transform's tangent stays inside its range, so the candidate there is
    the step itself (:meth:`_CoordinateBounds.from_tangent` reads it back).
    ``J`` is with respect to the coordinates either way.

    **The solve is equilibrated and framed.**  ``J``'s columns are each
    multiplied by the power of two that brings their largest entry into
    ``[0.5, 1)`` and ``r`` by the one that brings its own there, and the
    step is unframed by the same powers
    (:func:`~maddening.core._pow2_frame.pow2_rescale`, exact).
    Forming ``A = JᵀJ`` and ``g = Jᵀr`` bare in the working precision
    flushed to zero or overflowed for a residual or a parameter far from
    unit scale -- XLA's CPU backend flushes a subnormal product -- and in
    float32 that is any ``|J|`` below about ``1e-19`` or above ``1e19``: a
    residual of ``1e-19`` (a nanoparticle's mass in kilograms) solved for
    nothing, and a parameter whose natural scale is ``1e-23`` (a 10 nm
    particle's volume in cubic metres) had ``diag(A)`` overflow to ``inf``
    and its step vanish.  Framed, every column of ``A`` that ``J`` does
    not leave at zero has a diagonal entry in ``[0.25, m]``.  The
    Marquardt step is invariant to a column scaling, so in exact
    arithmetic nothing moves; in floating point the equilibrated system
    is solved with different pivots, so ordinary problems move in the
    last bits.  ``representable`` (a 0-d bool) is False when a column with
    a nonzero entry of ``J``, not held, still has a ``diag(A)`` that is
    zero or not finite: :func:`fit_lm` never reports such a step as
    converged.

    ``held`` (a boolean mask) holds coordinates at ``th`` -- the
    mixed-precision coordinates whose step rounds away at their leaf's
    dtype -- as a coordinate on its bound whose gradient points out of the
    range is held: its row and column replaced by the identity and its
    gradient by 0, so the step for the rest is solved without it rather
    than compensating for a move it cannot make.
    """
    c = _column_frame(J)
    f = pow2_frame(r)
    Jh = J * c[None, :]
    A = Jh.T @ Jh
    g = Jh.T @ (r * f)
    # A coordinate on its bound whose gradient points out of the range is
    # held (its row and column replaced by the identity, its gradient by 0):
    # that is the constraint active, and solving for it as if free would
    # couple a step it cannot take into the others.
    hold = held | ((th <= lo) & (g >= 0)) | ((th >= hi) & (g <= 0))
    free = jnp.where(hold, 0.0, 1.0).astype(A.dtype)
    A = A * free[:, None] * free[None, :]
    g = g * free
    d = jnp.diag(A)
    # The floor that keeps the solve regular is ``eps`` times each column's
    # *own* curvature, ``diag(A)_i`` -- Marquardt's scaling with ``lambda``
    # raised by ``eps`` -- so the step is the same for every scaling of the
    # residual *and* of each parameter.  An absolute ``1e-12 * I`` was not
    # negligible for a residual small in its own units (1e-7: unconverged
    # after 50 iterations); a floor of ``eps`` times the *mean* of
    # ``diag(A)`` then crushed the step of a coordinate whose column was
    # small beside another's: a spring in SI units (damping in N s/m, beside
    # a log stiffness) stopped after two iterations, "converged", with
    # damping at 3x its truth, where the same spring in tonnes reached it.
    # A column of exact zeros gets 1.0 (its gradient entry is 0 as well).
    floor = jnp.where(d > 0, jnp.finfo(A.dtype).eps * d, 1.0)
    A_damped = A + jnp.diag(lam * d + floor + (1.0 - free))
    dh = jnp.linalg.solve(A_damped, g)
    delta = pow2_rescale(dh, c, f)
    live = jnp.any(J != 0.0, axis=0) & ~hold
    representable = ~jnp.any(live & ~(jnp.isfinite(d) & (d > 0.0)))
    return jnp.clip(th - delta, lo, hi), representable


@jax.jit
def _gauss_newton_step(th, r, J, lo, hi, held):
    """The undamped Gauss-Newton candidate from ``th``, projected onto
    ``[lo, hi]``: least squares on the column-equilibrated Jacobian; and
    whether every column it solved for was representable.

    The stationarity test :func:`fit_lm`'s ``converged`` also requires.  It
    shares nothing with the Marquardt solve -- no ``lambda``, no floor, a
    different factorisation -- so a step that the damping or a mis-scaled
    floor shrank cannot read as stationary through it: if the undamped step
    from the iterate would still move a parameter by more than
    ``step_tol``, the iterate is not stationary.  Equilibrating the columns
    (each scaled to unit norm) makes it invariant to the units of every
    parameter and of the residual.  The norms are taken of columns first
    framed by a power of two (largest entry into ``[0.5, 1)``), and ``r``
    is framed the same way: bare, a column of ``1e-23`` had a norm that
    flushed to zero (``1e23``: overflowed to ``inf``), the column was left
    out as if ``J`` did not read the parameter, and the iterate read as
    stationary with the parameter unmoved.  The frames are powers of two,
    so wherever the bare norms were normal the candidate is the same, bit
    for bit.  ``representable`` is False when a column with a nonzero entry
    of ``J``, not held, still has a norm that is zero or not finite.  A
    coordinate held on its bound (its gradient pointing out of the range)
    is left out, as in the step, and so is one ``held`` holds.  Called in
    the tangent frame, as :func:`_marquardt_step` is.
    """
    c = _column_frame(J)
    f = pow2_frame(r)
    Jc = J * c[None, :]
    g = Jc.T @ (r * f)
    hold = held | ((th <= lo) & (g >= 0)) | ((th >= hi) & (g <= 0))
    norms = jnp.sqrt(jnp.sum(Jc * Jc, axis=0))
    usable = (norms > 0) & jnp.isfinite(norms)
    scale = jnp.where(usable & ~hold, 1.0 / jnp.where(usable, norms, 1.0), 0.0)
    y = jnp.linalg.lstsq(Jc * scale[None, :], -(r * f))[0]
    step = pow2_rescale(scale * y, c, f)
    live = jnp.any(J != 0.0, axis=0) & ~hold
    representable = ~jnp.any(live & ~usable)
    return jnp.clip(th + step, lo, hi), representable


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
    step_tol: Optional[float] = None,
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
    ``0.5 ||r||²`` at the start of iteration ``i + 1``.  ``converged`` is
    ``True`` when the loss reached ``tol``, or when the step an iteration
    *proposed* -- its first candidate, at the damping it started with --
    moved every trainable parameter by no more than ``step_tol`` of its
    own magnitude, whether or not that step lowered the loss (see
    ``step_tol``), or -- once the run has lowered the loss at least once --
    when every candidate of an iteration was rejected down to one damped
    within ``step_tol`` (the rounding floor; see ``step_tol``).  A run that
    never lowered the loss and rejected every candidate is not converged,
    and neither is one that ran out of ``n_iter``.  At that floor the
    undamped Gauss-Newton step is asked last: if it moves a parameter by
    more than ``step_tol`` and lowers the loss it is taken as the iterate
    and the run goes on, so the verdict there does not depend on which way
    the damped candidates' rounding fell.

    **A step of a ``log`` / ``logit`` coordinate is read along the
    transform's curve or along its tangent, whichever moves the value
    less.**  The step is solved on the residual's linearisation, which for
    such a coordinate is the transform's tangent, and the two readings of a
    step ``v`` -- ``p(u + v)`` and ``p(u) + p'(u) v`` -- agree to first
    order.  Where the transform is flat (a ``logit`` value near a bound, a
    ``log`` value far below its optimum) the tangent's is the shorter: the
    step the same parameter would take under ``transform=None``, instead of
    one ``1 / p'(u)`` long in ``u`` that lands on the opposite edge of the
    range.  Where a step heads for a bound the curve's is the shorter: it
    closes the distance by a factor of ``e`` at most and never lands on the
    bound.  And every step is confined to the range the transform resolves,
    ``sqrt(eps)`` of the bounds' size inside each bound
    (:class:`_CoordinateBounds`).  The fitter is held to the consequence
    over a grid of the parameter guide's spring -- each transform, the truth
    and the start anywhere in the range and a few float spacings inside
    each edge, float32 and float64
    (``tests/property/test_sysid_truth_recovery.py``): wherever the same
    fit with the parameter under ``transform=None`` and the same bounds
    recovers its optimum, this one does too.  And a fit that ends on the
    edge of that range -- the data pull the parameter onto its bound, which
    this transform cannot reach -- says so in a :class:`RuntimeWarning`
    naming the parameter.  Until 0.4.0's fix a ``logit`` coordinate that an
    early step carried to the edge of its range (where ``p'(u)`` was a few
    ``eps``) stayed there: every candidate landed on the opposite edge, the
    damped retries with it, and the run ended on the bound with
    ``converged=False`` and no warning -- a damping bounded to ``(0.5, 2)``
    came back 2.0 for a truth of 1.9 from 20 of 60 starts under x64
    (MADD-ANO-104).

    A trainable leaf with ``bounds`` and ``transform=None`` is optimised in
    its own coordinate and clipped by ``constrain``.  Every step is projected
    back onto the bounds, so a coordinate one step carried past its bound
    sits on it -- where its derivative is the one-sided one into the range
    -- rather than outside it with no gradient back (earlier 0.4.0
    development builds left a spring's damping at 0 for the rest of the
    run, and called it converged).  A coordinate on its bound whose gradient
    points out of the range is held there: the constraint is active, and
    the step for the rest is solved without it.  No ``converged`` test fires
    while a coordinate on its bound could lower the loss by moving into the
    range -- by more than ``step_tol``, measured by its own Newton step: a
    pull smaller than that is the residual's rounding (at a truth exactly on
    the bound its sign is a coin flip, and decided the verdict until 0.4.0's
    fix) and counts as zero.

    The Marquardt solve's floor, which keeps it regular, is ``eps`` times
    each column's own ``diag(JᵀJ)``: so the step is the same for every
    scaling of the residual and of each parameter (an absolute ``1e-12``
    made a residual of 1e-7 in its own units fail to converge in 50
    iterations, and a floor of ``eps`` times the *mean* crushed the step of
    a parameter measured in small units, MADD-ANO-121).

    That holds across the working precision's whole range -- any residual,
    and any parameter, whose residual entries (its own rounding at the
    optimum included) and Jacobian entries are normal numbers: in
    float32 a residual of ``1e-30`` or ``1e30``, a parameter whose natural
    scale is ``1e-23`` (a 10 nm particle's volume in cubic metres) or
    ``1e23``.  The loss is ``0.5 * ||r||²`` of ``r`` framed by a power of two
    and unframed in float64, and the solves are formed on columns of ``J``
    and on ``r`` framed the same way (:func:`_marquardt_step`,
    :func:`_gauss_newton_step`), which is exact and leaves every ordinary
    problem's arithmetic as it was but for the Marquardt solve's pivots.
    Earlier 0.4.0 development builds formed ``r * r``, ``JᵀJ`` and ``Jᵀr``
    bare, which XLA's CPU backend flushes below ``tiny`` and which overflow
    above ``sqrt(max)``: a residual of ``1e-19`` returned a point 7% off, or
    its start, with ``converged=True`` and a loss of ``0.0``, and a
    parameter at ``1e-23`` came back unmoved, "converged" (MADD-ANO-174).
    ``converged`` is never reported on a solve whose live columns were not
    representable even framed, or on a loss of ``0.0`` from a residual that
    is not zero (a float64 residual below about ``1e-162``): the run stops
    there unconverged, with a :class:`RuntimeWarning`.

    A float32 leaf in an x64 graph is a float64 coordinate here
    (``ravel_pytree`` promotes it), but its leaf is rounded to float32
    before the model sees it; the coordinate is kept on that grid, and one
    whose move rounds away is held while the others are solved again, so a
    joint step never compensates for a move the model does not make
    (:func:`_leaf_grid`; earlier development builds crept, five times the
    iterations of an all-float64 fit).

    Like :func:`fit`, it returns the **lowest-loss iterate** it evaluated,
    and here that is always the last one: a step is accepted only when it
    strictly lowers the loss, so every accepted iterate is below every
    earlier one, and a rejected step changes nothing.  No extra
    evaluation is needed to know it, and none is made.  Every loss the run
    reads -- ``losses``, the acceptance test, ``best_loss`` -- is one
    compiled residual's value at the parameters :class:`FitResult` returns
    for that iterate (the Jacobian is a program of its own and gives no
    loss).
    :attr:`FitResult.best_iteration` is ``len(losses)`` when the run ended
    on an accepted step (the loss of that iterate, ``best_loss``, is the
    one the acceptance test computed) and ``len(losses) - 1`` when it
    ended on ``tol``, on a proposed step within ``step_tol`` that did not
    lower the loss, or on a step no damping could make acceptable.

    Parameters
    ----------
    step_tol : float, optional
        Relative step tolerance: an iteration whose proposed step changes
        every trainable parameter ``p`` by at most ``step_tol * |p|``
        (compared with ``<=``) ends the run as converged.  ``None`` (the
        default) is ``2**4`` units in the last place of each parameter's
        own dtype -- ``2**4 * eps``, 1.9e-6 in float32 and 3.6e-15 in
        float64 -- the parameters' own float resolution, so the default
        can be met at either precision.  ``0.0`` counts only a step that
        leaves every parameter bit for bit where it was.  Measured on the
        physical parameters, after ``constrain``: so a step that pushes a
        clipped parameter further past its bound moves it by nothing, and
        the tolerance means the same under every transform.  A parameter
        whose value is exactly ``0`` has no relative resolution and meets
        it only with a step of exactly nothing there; give such a fit a
        ``tol``.  The same holds for a parameter whose *truth* is exactly
        ``0`` (a damping of 0, say) under x64: the fit lands on the
        residual's rounding noise around it -- ``1.3e-15`` for a noiseless
        float64 spring -- and a step relative to a value that is itself
        rounding noise is never within ``step_tol``, so such a run stops at
        its floor with ``converged=False`` however small its loss
        (``4.6e-31`` there).  That is the flag being exact about what it
        tests, not a failed fit: read ``best_loss``, or pass a ``tol`` at
        the noise floor.  (In float32 the bound clips the same fit to
        ``0.0`` exactly, where a step of nothing meets the tolerance and
        the run converges.)

        The proposal is tested whether or not it was accepted, because at
        the float floor its verdict carries no information: a fit that has
        reached its optimum proposes a step that rounds to (nearly)
        nothing, and a step of nothing cannot *strictly* lower a loss that
        is already rounding noise.  Earlier 0.4.0 development builds
        tested only accepted steps, against a fixed ``1e-8`` in the
        optimiser's coordinates that float32 cannot resolve, and reported
        ``converged=False`` for a noiseless fit at loss 8.7e-14 whose last
        proposal moved neither parameter by a single bit.  Only the
        *proposal* counts, not a damped retry: retries shrink the step by
        a factor of ``lam_up`` each whatever the loss is doing, so a short
        retry is evidence about the damping and not about the iterate --
        except once the run has lowered the loss and *every* candidate,
        down to one within ``step_tol``, is rejected: then no step the
        tolerance resolves lowers the loss, which is the rounding floor.  A
        parameter the data determine only weakly needs that rule, because
        its proposal at the floor fits the residual's rounding noise and can
        be many ulps of it.  For that rule the twelve damped candidates of
        an iteration are extended -- a decade or more of ``lam`` each, up
        to the cap of ``1e12`` -- until one is within ``step_tol``: twelve
        rungs fell one short under ``jax_enable_x64`` (a relative step of
        ``4.4e-15`` against ``3.6e-15`` at loss ``2e-30``), so the default
        could not be met there.  Changed from an absolute ``1e-8`` during
        0.4.0 development, before any release.
    hold_undetermined : bool
        Keep the fitted parameters out of the directions the data does not
        determine, exactly as :func:`fit` does and by the same shared
        machinery.  **New in 0.4.0, and on by default**; ``False`` returns
        the last iterate exactly as the Marquardt steps produced it.

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
        **−1.78%** (noiseless data) or **−0.66%** (σ = 0.02) from the
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
        are unchanged; a fit with no undetermined direction gets its
        iterate back bit for bit; the cutoff is numerical rather than
        statistical, so a merely weakly-identified direction is kept;
        and the degeneracy has to be a fixed direction in the optimiser's
        coordinates.  :attr:`FitResult.excited_rank`,
        :attr:`FitResult.undetermined_drift` and
        :attr:`FitResult.hold_declined` say what the guard found.

        The curvature test here is ``JᵀJ`` at the iterate returned — the
        Gauss–Newton matrix the steps are built from, and with
        ``noise_std`` the Fisher information :func:`fim` reports — read with
        :func:`fim`'s own rank cutoff, so a direction is held only if
        :func:`fim` would call it unresolved there, in the fitter's
        coordinates.  It costs at most one more ``jacfwd``, and none when
        the run ended without moving off the iterate whose Jacobian it
        formed last.  It is what this fitter needed most: LM converges in a
        few nearly parallel steps, whose gradients span little, and on a
        four-parameter bowl that determines every direction the gradient
        test alone left three unspanned; earlier 0.4.0 development builds
        held all three and returned parameters at a loss of 0.22 from a
        fit that had reached 0.0.
    """
    n_iter = _check_count("n_iter", n_iter)
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
    if step_tol is not None:
        _check_hyper("step_tol", step_tol, ge=0.0,
                     why=" step_tol is a relative change compared with <=, so a "
                         "negative one can never be met and 'converged' could only "
                         "ever mean the loss reached tol; None is the parameters' "
                         "own float resolution.")
    notify_every = _check_count("notify_every", notify_every)
    _check_hold_undetermined(hold_undetermined)
    _sync_compiled(gm)
    start = gm._params_or_default(params)  # noqa: SLF001
    gm.check_params(start)
    mask = _resolve_mask(gm, start, mask)
    u0 = gm.unconstrain(start)
    flat_u, _ = ravel_pytree(u0)
    idx = _masked_indices(start, mask)
    if idx is None:
        idx = np.arange(flat_u.size)
    # The one evaluation of ``theta -> params`` (SYS-071): the residual is a
    # compiled function of the physical tree ``pmap.params`` returns, and
    # that tree is what ``callback`` receives and the result holds.
    pmap = _PhysicalMap(gm, start, flat_u, idx)
    theta0 = flat_u[idx]
    theta = theta0
    progress = _progress_notifier(gm, "lm", n_iter, notify_every)

    # ``_resolved_noise``, not ``_inverse_noise_std(noise_std,
    # residual_fn(...))``: the residual is wanted for its *structure*
    # only, and evaluating it cost a whole extra rollout per call --
    # bought nothing at all in the ``noise_std is None`` case, which
    # returns before looking at it.
    inv_sigma = _resolved_noise(residual_fn, start, noise_std)

    def _residual(p):
        # A function of the physical tree alone (``_compile_model``).
        r = ravel_pytree(residual_fn(p))[0]
        return r if inv_sigma is None else r * inv_sigma

    def _jacobian(p, slope):
        # ``dr/dtheta``, one forward pass per trainable entry.  Column ``j``
        # is the model's derivative along entry ``j`` of ``p`` with the
        # tangent ``slope[j]`` (``dp/dtheta`` there): the tangent the map
        # itself would push forward, so a column is never formed in the
        # parameter's own units and a ``log`` coordinate stays unit-free.
        fitted, others = pmap.split(p)

        def column(tangent):
            return jax.jvp(lambda leaves: _residual(pmap.join(leaves, others)),
                           (fitted,), (pmap.unravel(tangent),))[1]

        return jax.vmap(column, out_axes=1)(jnp.diag(slope))

    # One program computes every residual the run takes a loss from, so
    # ``losses``, the acceptance test and ``best_loss`` are values of one
    # function at the trees ``pmap.params`` returned; the Jacobian is its own
    # program and gives no loss.
    residual_only = _compile_model(_residual)
    jacobian = _compile_model(_jacobian)

    bounds = _CoordinateBounds(gm, start, idx, theta0.dtype)
    unresolved: set = set()
    _warn_unresolved_values("fit_lm", pmap, start, "at the start of the fit", unresolved)
    _lm_step = _marquardt_step

    # The step test, on the physical trainable entries (see ``step_tol``):
    # per entry, a relative tolerance -- ``step_tol``, or ``2**4`` ulps of
    # the entry's own dtype -- so a float32 leaf beside a float64 one is
    # held to float32's resolution, not to the raveled vector's.
    rel_tol = (_STEP_TOL_ULPS * _float_resolution(start)[idx] if step_tol is None
               else float(step_tol))

    # Mixed precision: a float32 leaf beside a float64 one is a float64
    # coordinate here (``ravel_pytree`` promotes), but the map rounds it
    # back to float32 before the model sees it (:func:`_leaf_grid`).
    narrow, to_leaf_grid = _leaf_grid(start, idx, theta0.dtype)
    no_hold = jnp.zeros(theta0.shape, dtype=bool)

    def _quantised(th, solve):
        """``(candidate, representable)`` from ``solve(held)`` -- a step from
        ``th`` -- with every
        narrow coordinate on its leaf's grid.  A narrow coordinate whose
        proposed move rounds away there is held and the step solved again
        without it, so the others do not compensate for a move the model
        never sees (:func:`_leaf_grid`).  The identity, bit for bit, when no
        leaf is narrower than the coordinates."""
        cand, ok = solve(no_hold)
        if not narrow.any():
            return cand, bool(ok)
        held = np.zeros(narrow.shape, dtype=bool)
        here = np.asarray(th)
        for _ in range(int(narrow.sum())):
            c = np.asarray(cand)
            lost = narrow & ~held & (np.asarray(to_leaf_grid(cand)) == here) & (c != here)
            if not lost.any():
                break
            held |= lost
            cand, ok = solve(jnp.asarray(held))
        return to_leaf_grid(cand), bool(ok)

    def _stepped(th, solver):
        """``(candidate, representable)`` for a step solved from ``th`` on
        the linear model: ``solver(origin, lo, hi, held)`` is asked in the
        tangent frame (:meth:`_CoordinateBounds.tangent_frame`) and its
        answer read back as coordinates, a ``log`` / ``logit`` one along the
        shorter of its curve and its tangent."""
        origin, lo, hi = bounds.tangent_frame(th)

        def solve(held):
            answer, ok = solver(origin, lo, hi, held)
            return bounds.from_tangent(th, answer), ok

        return _quantised(th, solve)

    def _within_step_tol(before, after) -> bool:
        """Whether a step from the physical values ``before`` to ``after``
        moves every trainable parameter by at most ``step_tol`` of itself."""
        return bool(np.all(np.abs(after - before) <= rel_tol * np.abs(before)))

    # Whether every solve a convergence verdict of this iteration rests on
    # was representable (``_marquardt_step``, ``_gauss_newton_step``).
    verdict = {"representable": True}

    def _resolvable(th, before, g, J) -> np.ndarray:
        """``g`` with every entry zeroed whose coordinate's own Newton step,
        ``g_i / ||J[:, i]||²`` (the others held), moves its parameter by no
        more than ``step_tol`` -- for the on-bound test only.

        On its bound a coordinate whose gradient points into the range
        could lower the loss by moving there, and no converged test fires
        while it can (:meth:`_CoordinateBounds.inward_descent`).  At a fit
        whose truth is *on* the bound, that gradient is the residual's
        rounding, and its sign is a coin flip: a noiseless fit to a truth
        on ``(0.5, 2.0)`` converged on one bound and not on the other (and
        the other way round under x64).  A pull whose own step is within
        the tolerance moves nothing the tolerance resolves, so it is that
        noise and counts as zero.  The step is read as every step is (the
        tangent frame) and measured physically -- ``before`` is the physical
        value at ``th`` -- so a ``logit`` coordinate at the flat edge of its
        range is judged by the value's move.
        """
        if not bounds.active:
            return g
        # ``g_i / ||J_i||²`` from framed columns (each largest entry in
        # ``[0.5, 1)``): ``(g_i c_i) c_i / ||c_i J_i||²``, which neither
        # flushes nor overflows where the step itself does not.
        c = np.asarray(_column_frame(J), dtype=np.float64)
        curvature = np.sum(np.square(np.asarray(J, dtype=np.float64) * c), axis=0)
        step = np.where(curvature > 0.0,
                        (g * c) * c / np.where(curvature > 0.0, curvature, 1.0), 0.0)
        origin, lo, hi = bounds.tangent_frame(th)
        answer = np.clip(np.asarray(origin, dtype=np.float64) - step,
                         np.asarray(lo, dtype=np.float64),
                         np.asarray(hi, dtype=np.float64))
        moved = bounds.from_tangent(th, jnp.asarray(answer, dtype=th.dtype))
        after = pmap.physical(pmap.params(to_leaf_grid(moved)))
        return np.where(np.abs(after - before) <= rel_tol * np.abs(before), 0.0, g)

    def _gauss_newton_stationary(th, before, r, J, loss_at_th=None) -> bool:
        """Whether ``th`` is stationary by the undamped, equilibrated
        Gauss-Newton step from it (:func:`_gauss_newton_step`), which neither
        the damping nor the Marquardt floor can shrink -- required, beside
        the step tests below, before ``converged`` is reported.

        Stationary when that step moves every parameter by no more than
        ``step_tol`` (``before`` is the physical value at ``th``).  For the
        floor rule (``loss_at_th`` given) also when
        the step does not lower the loss: a parameter the data determine
        only weakly has, at its rounding floor, a Gauss-Newton step that
        fits the residual's rounding noise and can be many ulps long, but
        it lowers nothing.  A measure built from ``r`` and ``J`` alone
        cannot tell that floor from a real error -- the residual there is
        structured rounding, whose cosines with the columns measured 0.5-0.9
        on a spring -- so the second test evaluates the step.
        """
        candidate, ok = _stepped(
            th, lambda origin, lo, hi, held: _gauss_newton_step(origin, r, J, lo, hi, held))
        verdict["representable"] &= ok
        candidate_params = pmap.params(candidate)
        if _within_step_tol(before, pmap.physical(candidate_params)):
            return True
        if loss_at_th is None:
            return False
        loss_gn = _half_squared_norm(residual_only(candidate_params))
        return not (np.isfinite(loss_gn) and loss_gn < loss_at_th)

    lam = float(lam0)
    coarse = _coarsest_dtype(start, idx)
    tracker = _make_excitation_tracker(hold_undetermined, theta0, coarse)
    losses: list[float] = []
    # ``theta``'s update count and its loss as last evaluated.  No
    # ``_BestIterate`` here: the acceptance test already makes the current
    # iterate the lowest.
    theta_iteration, theta_loss = 0, None
    # The physical tree ``theta`` maps to -- what the model is evaluated on
    # and the result holds -- and the residual there by ``residual_only``
    # (``None`` until an iteration needs it).
    theta_params = pmap.params(theta)
    theta_r = None
    # The last ``(r, J)`` the loop formed and the iterate it belongs to, so
    # that the guard's curvature test reuses it when the run ended without
    # moving off that iterate rather than forming it again.
    rJ, rJ_iteration = None, None
    converged = False
    # Whether any step has lowered the loss yet: see the floor rule below.
    progressed = False
    i = 0
    for i in range(1, n_iter + 1):
        if theta_r is None:
            theta_r = residual_only(theta_params)
        r = theta_r
        J = jacobian(theta_params, pmap.slope(theta))
        # On the framed residual (``_half_squared_norm``): ``r * r`` in the
        # working precision flushed a residual of 1e-19 to a loss of 0.0.
        loss = _half_squared_norm(r)
        if not np.isfinite(loss) or not bool(jnp.all(jnp.isfinite(J))):
            raise FloatingPointError(f"non-finite residual or Jacobian at iteration {i}")
        losses.append(loss)
        theta_iteration, theta_loss = i - 1, loss
        rJ, rJ_iteration = (r, J), i - 1
        # ``Jᵀr`` framed (``_framed_gradient``), in float64, for the tracker
        # and the on-bound test: bare, it flushed to zero with the residual.
        grad = _framed_gradient(J, r)
        if tracker is not None:
            # The same ``g`` ``_lm_step`` forms, recomputed here rather than
            # returned from it: the tracker must not change the step, and
            # ``_lm_step``'s own ``g`` lives inside a ``jax.jit`` whose
            # fusion decides ``A``'s last bits (PR 101).  Folded in before
            # the ``tol`` break, as in ``fit``: this gradient is information
            # about the loss surface whether or not a step was taken on it.
            tracker.observe(grad)
        if callback is not None or progress is not None:
            current = pmap.returned(theta, theta_params)
            if callback is not None:
                callback(i, loss, current)
            if progress is not None:
                progress(i, loss, current)
        if tol > 0.0 and loss <= tol:
            converged = True
            break
        # Try a step; shrink lambda on success, grow it (and retry) on failure.
        # The first candidate is the iteration's proposal, and only it is put
        # to ``step_tol`` -- accepted or not: at the float floor a proposal
        # rounds to (nearly) nothing and cannot *strictly* lower a loss that
        # is rounding noise, so its rejection says nothing.  A retry is
        # shorter only because it is damped more, so it is not tested.
        accepted = False
        stationary = False
        verdict["representable"] = True
        # A loss of exactly 0.0 from a residual that is not: the framed
        # sum's unframing underflowed float64 (a float64 residual below
        # ~1e-162).  No step can lower it, so nothing about the iterate can
        # be read from the step tests.
        vanished = loss == 0.0 and bool(jnp.any(r != 0.0))
        here = pmap.physical(theta_params)
        # A coordinate on its bound that could lower the loss by moving into
        # the range makes this no constrained stationary point, so its
        # proposal cannot converge the run even if it moves nothing (the
        # coupled solve can point such a coordinate outward); the retries,
        # damped towards the gradient, move it inward.
        stuck_inward = bounds.inward_descent(theta, _resolvable(theta, here, grad, J),
                                             here, rel_tol)
        attempt, there = -1, here
        while True:
            attempt += 1
            if attempt >= _LM_LADDER and not (
                    progressed and not stuck_inward and lam < _LM_LAMBDA_MAX
                    and not _within_step_tol(here, there)):
                # The ladder's end -- unless the floor rule below would be
                # asked to judge a ladder that stopped short of ``step_tol``:
                # then it is extended to the damping cap
                # (:data:`_LM_LADDER`), so the rule can see the shortest
                # candidate it needs.
                break
            lam_t = jnp.asarray(lam, theta.dtype)
            cand, cand_ok = _stepped(
                theta, lambda origin, lo, hi, held, lam_t=lam_t: _lm_step(
                    origin, r, J, lam_t, lo, hi, held))
            if attempt == 0:
                verdict["representable"] &= cand_ok
            cand_params = pmap.params(cand)
            there = pmap.physical(cand_params)
            cand_r = residual_only(cand_params)
            loss_new = _half_squared_norm(cand_r)
            proposal_within = (attempt == 0 and not stuck_inward
                               and _within_step_tol(here, there)
                               and _gauss_newton_stationary(theta, here, r, J))
            if np.isfinite(loss_new) and loss_new < loss:
                theta, theta_params, theta_r = cand, cand_params, cand_r
                theta_iteration, theta_loss = i, loss_new
                lam = max(lam * lam_down, 1e-12)
                accepted = True
                progressed = True
                stationary = proposal_within
                break
            if proposal_within:
                # Nothing to retry: every further candidate is shorter still.
                stationary = True
                break
            # Past the ladder, at least a decade a rung, so the extension
            # reaches the cap in at most 24 rungs whatever ``lam_up`` is.
            grow = lam_up if attempt + 1 < _LM_LADDER else max(lam_up, 10.0)
            lam = min(lam * grow, _LM_LAMBDA_MAX)
        if not accepted and not stationary and progressed and not stuck_inward:
            # The floor rule.  Every candidate was rejected, down to one damped
            # within ``step_tol``: no step the tolerance resolves lowers the
            # loss.  After a run that has already lowered it, that is the
            # rounding floor (MINPACK's ``xtol`` declares the same when its
            # trust region shrinks below the tolerance without a success).  A
            # parameter the data determine only weakly needs it: at the floor
            # its proposal is the Gauss-Newton fit of the residual's rounding
            # noise, which can be many ulps of the parameter (a spring's
            # damping of 0.05, seen through 100 position samples, proposes
            # ~1e-5 relative there), so the proposal test alone left such a
            # fit unconverged at loss 1e-13.  A run that never lowered the
            # loss gets no such reading: a residual whose Jacobian is wrong
            # rejects every candidate from its start.
            #
            # The rule asks the undamped Gauss-Newton step last, and when it
            # moves a parameter by more than ``step_tol`` *and* lowers the
            # loss, it is taken as the iterate rather than ending the run:
            # it is a candidate like the damped ones (``lam -> 0``), and
            # the strict decrease is the acceptance test every candidate
            # meets.  Near the floor of the loss's own evaluation -- a
            # noisy residual whose rounding moves the loss by more than a
            # few ulps of a parameter do -- the damped candidates' verdicts
            # are rounding, while the Gauss-Newton step, built from ``r``
            # and ``J`` linearly, still points at the optimum.  Earlier 0.4.0
            # development builds ended such a run unconverged there, 40 ulps
            # short, or converged a few ulps away, by which of the two the
            # parameters' units happened to round to (MADD-ANO-174).
            if _within_step_tol(here, there):
                gn_cand, gn_ok = _stepped(
                    theta, lambda origin, lo, hi, held: _gauss_newton_step(
                        origin, r, J, lo, hi, held))
                verdict["representable"] &= gn_ok
                gn_params = pmap.params(gn_cand)
                if _within_step_tol(here, pmap.physical(gn_params)):
                    stationary = True
                else:
                    gn_r = residual_only(gn_params)
                    loss_gn = _half_squared_norm(gn_r)
                    if np.isfinite(loss_gn) and loss_gn < loss:
                        theta, theta_params, theta_r = gn_cand, gn_params, gn_r
                        theta_iteration, theta_loss = i, loss_gn
                        lam = max(lam * lam_down, 1e-12)
                        accepted = True
                    else:
                        stationary = True
        if stationary and (vanished or not verdict["representable"]):
            # Never a converged verdict the arithmetic could not form: with
            # the frames neither can happen to a float32 problem, but the
            # verdict must not rest on them if one does.
            warnings.warn(
                "fit_lm: stopped at an iterate whose stationarity could not be "
                "read -- "
                + ("the loss 0.5 * ||r||^2 underflows float64 although r is not "
                   "zero" if vanished else
                   "a column of the Jacobian that is not zero has a curvature "
                   "diag(J^T J) that is zero or not finite even after framing")
                + "; converged=False. Rescale the residual or the parameter "
                  "towards unit size.",
                RuntimeWarning, stacklevel=2)
            break
        if stationary and accepted and tracker is not None:
            # Stopping on an accepted proposal leaves the run at an iterate
            # whose gradient no iteration formed.  A run that kept going
            # would have formed it next, and the guard's tracker needs it:
            # a fit that converges in as many iterations as it has
            # parameters otherwise has too few gradients to measure
            # ``excited_rank`` at all.  The guard's curvature test reads
            # the same ``J`` (``_rj_selected``), so it is formed once.
            r_end, J_end = theta_r, jacobian(theta_params, pmap.slope(theta))
            if bool(jnp.all(jnp.isfinite(r_end))) and bool(jnp.all(jnp.isfinite(J_end))):
                tracker.observe(_framed_gradient(J_end, r_end))
                rJ, rJ_iteration = (r_end, J_end), theta_iteration
        if stationary or not accepted:
            converged = stationary
            break

    selected, selected_params = theta, theta_params
    at_selected: dict = {}

    def _rj_selected():
        # Formed at most once, and not at all when the loop already did:
        # the run ended on ``tol`` or on a rejected step, either way still
        # at the iterate whose ``(r, J)`` it formed last, or on an accepted
        # proposal within ``step_tol``, whose ``(r, J)`` it formed for the
        # tracker.
        if "rJ" not in at_selected:
            if rJ is not None and rJ_iteration == theta_iteration:
                at_selected["rJ"] = rJ
            else:
                r_sel = residual_only(selected_params) if theta_r is None else theta_r
                at_selected["rJ"] = (r_sel, jacobian(selected_params, pmap.slope(selected)))
        return at_selected["rJ"]

    def _loss(th):
        return _half_squared_norm(residual_only(pmap.params(th)))

    def _reference():
        r_sel, J_sel = _rj_selected()
        return _half_squared_norm(r_sel), _framed_gradient(J_sel, r_sel)

    def _flatness(candidates, spanned, scale):
        return _gauss_newton_flatness(_rj_selected()[1], candidates, coarse,
                                      scale=scale)

    def _columns():
        return np.linalg.norm(np.asarray(_rj_selected()[1], dtype=np.float64), axis=0)

    theta, excited_rank, undetermined_drift, hold_declined = _hold_undetermined_directions(
        tracker, selected, theta0,
        _SelectedObjective(loss=_loss, reference=_reference, flatness=_flatness,
                           transformed=bounds.transformed, columns=_columns),
        "fit_lm")

    # The tree the selected iterate was evaluated on, unless the guard moved it.
    evaluated = selected_params if theta is selected else pmap.params(theta)
    if i > 0:
        _warn_on_an_edge("fit_lm", pmap, bounds, theta, evaluated)
        _warn_unresolved_values("fit_lm", pmap, evaluated, "as fitted", unresolved)
    return FitResult(
        params=pmap.returned(theta, evaluated), losses=np.asarray(losses),
        converged=converged, n_iter=i,
        excited_rank=excited_rank, undetermined_drift=undetermined_drift,
        best_iteration=theta_iteration, best_loss=theta_loss,
        hold_declined=hold_declined,
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
    start_step: Optional[int] = None,
) -> tuple[FitResult, dict]:
    """Multiple-shooting fit: Adam jointly over the trainable params (in
    unconstrained coordinates) and the free per-window initial states.

    Compared with :func:`fit` on the teacher-forced :func:`windowed_loss`,
    the window starts are decision variables and a continuity penalty
    (``continuity_weight``) joins consecutive windows, so the optimum is a
    single continuous trajectory and noisy observations at window starts
    do not seed every window with measurement error.

    Returns ``(FitResult, window_states)``, both from the **lowest-loss
    iterate** the run evaluated, chosen exactly as :func:`fit` chooses
    (ties to the later iterate, so a run whose loss never rose returns
    its last iterate bit for bit; the iterate the last update produced is
    evaluated once more when the budget ran out).  The parameters and the
    window states are one point of the joint objective, so they are
    selected together: :attr:`FitResult.best_loss` is the loss of that
    selected pair, as the run evaluated it, before the
    ``hold_undetermined`` guard -- which, the run evaluating every leaf no
    step moved at the value that went in, is the loss of exactly the pair
    returned when the guard holds nothing.  The window states are returned exactly
    as selected; the parameters are too unless the guard held a direction
    (``excited_rank`` below the count, ``hold_declined`` False), in which
    case the returned pair's loss is within the guard's tolerance of
    ``best_loss`` -- ``2**10 * eps`` relative plus the loss cost of
    rounding the held point (see :class:`FitResult`) -- and not
    necessarily equal to it.  This fitter needed the selection more than
    :func:`fit` does: started at the truth, with every window state
    stepped by Adam at a rate of ``lr``, it returned a loss of ``2e-1``
    from a start of ``1e-13`` before it selected.

    Parameters
    ----------
    start_step : int, optional
        *Experimental, new in 0.4.0.*  The base step of a multi-rate graph's
        schedule at which the record's sample ``0`` was taken, passed to
        :func:`windowed_loss` (see there; ``None`` assumes ``0`` and warns on
        a multi-rate graph).
    hold_undetermined : bool
        Keep the fitted **parameters** out of the directions the data does
        not determine, exactly as :func:`fit` does and by the same shared
        machinery.  **New in 0.4.0, and on by default**; ``False`` returns
        the selected iterate exactly as Adam produced it.

        This is the same Adam step rule :func:`fit` uses, so it is
        :func:`fit`'s defect and not merely the consistency issue
        :func:`fit_lm` had: the drift has not settled, and the *schedule*
        picks the answer.  Measured on the spring's ``(k, c, m)``
        common-scale degeneracy with σ = 0.02 observations, the geometric
        mean of the three lands **−4.35%** at ``lr=0.01, n_iter=200``,
        **−4.80%** at 1,200, **−1.97%** at ``lr=0.2, n_iter=200`` and
        **−2.02%** at 1,200 — a 3.0% spread in the returned constants for a
        loss that agrees to four digits.  Raising the budget to 4,000 moved
        the last iterate at ``lr=0.2`` on again, to −2.31%, so it was not
        converging to a value either.  Those are last iterates; the
        lowest-loss iterates this fitter now returns land −4.35%, −4.80%,
        −1.97% and −1.97% (and −1.97% at 4,000), the same 3.0% spread.
        Two runs on the same data return different physical constants and
        neither is preferred by the objective.

        Only ``theta`` is guarded.  The returned ``window_states`` are
        nuisance variables of the fit rather than constants a caller
        records as provenance, and a caller warm-starting from them needs
        the values the optimiser actually reached at the selected iterate;
        the gradient Gram is accumulated over the parameter block alone,
        which is where
        :attr:`FitResult.excited_rank` counts its directions.  That block's
        gradient is still ``J_θᵀ r`` at every iterate, so a ``v`` with
        ``J_θ v = 0`` has ``g·v = 0``, which is the whole premise.

        The curvature test and the loss check are :func:`fit`'s, on the
        joint objective as a function of ``theta`` with the window states
        fixed at the selected iterate's: the Hessian there, and the loss at
        the held parameters with those window states.  A direction the
        window states could compensate for is not found by that Hessian, so
        such a direction is left unheld (fail-open) rather than held at a
        loss the fixed window states would make it pay.
    """
    n_iter, notify_every = _check_adam_hyper(n_iter, lr, tol, betas, eps, notify_every)
    if lr_states is not None:
        _check_hyper("lr_states", lr_states, gt=0.0,
                     why=" The window starts are decision variables like the "
                         "parameters; a non-positive rate moves them the wrong way.")
    _check_hyper("continuity_weight", continuity_weight, ge=0.0,
                 why=_CONTINUITY_WEIGHT_WHY)
    sample_every = _check_count("sample_every", sample_every, minimum=1)
    if start_step is not None:
        start_step = _check_count("start_step", start_step)
    _check_hold_undetermined(hold_undetermined)
    _sync_compiled(gm)
    start = gm._params_or_default(params)  # noqa: SLF001
    gm.check_params(start)
    mask = _resolve_mask(gm, start, mask)
    ws0 = init_window_states(observations, window) if window_states is None else window_states
    b1, b2 = betas
    lr_s = lr if lr_states is None else lr_states
    progress = _progress_notifier(gm, "multiple_shooting", n_iter, notify_every)

    u0 = gm.unconstrain(start)
    flat_u, _ = ravel_pytree(u0)
    idx = _masked_indices(start, mask)
    if idx is None:
        idx = np.arange(flat_u.size)
    # The one evaluation of ``theta -> params`` (SYS-071), as in ``fit``.
    pmap = _PhysicalMap(gm, start, flat_u, idx)
    theta0 = flat_u[idx]
    ws_flat0, unravel_ws = ravel_pytree(ws0)

    def objective(p, ws_flat):
        return windowed_loss(
            gm, p, observations, obs_fn=obs_fn, window=window,
            sample_every=sample_every, external_inputs=external_inputs,
            window_states=unravel_ws(ws_flat), continuity_weight=continuity_weight,
            start_step=start_step,
        )

    plain, scaled = _model_loss_and_gradient(pmap, objective, n_extra=1)
    # As in ``fit``: the cotangent every gradient is taken with once a
    # flushed first gradient made the run lift them (``_gradient_lift``).
    lift = None

    def evaluate(th, w, p=None):
        """``(loss, (gradient in theta, gradient in the window states))``."""
        p = pmap.params(th) if p is None else p
        loss, g_p, g_w = plain(p, w) if lift is None else scaled(p, w, lift)
        return loss, (g_p * pmap.slope(th), g_w)

    def unlifted(g) -> np.ndarray:
        g64 = np.asarray(g, dtype=np.float64)
        return g64 if lift is None else g64 / float(lift)

    bounds = _CoordinateBounds(gm, start, idx, theta0.dtype)
    unresolved: set = set()
    _warn_unresolved_values("fit_multiple_shooting", pmap, start,
                            "at the start of the fit", unresolved)
    if n_iter > 0:
        _warn_starts_on_an_edge("fit_multiple_shooting", pmap, bounds, theta0, start, lr, eps)

    @jax.jit
    def adam(x, m, v, g, frame, i, rate):
        g = g * frame          # in the block's frame (``_adam_frame``), as in ``fit``
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        return x - rate * (m / (1 - b1 ** i)) / (jnp.sqrt(v / (1 - b2 ** i)) + eps), m, v

    theta, ws = theta0, ws_flat0
    m_t = jnp.zeros_like(theta); v_t = jnp.zeros_like(theta)
    m_s = jnp.zeros_like(ws); v_s = jnp.zeros_like(ws)
    frames = None
    coarse = _coarsest_dtype(start, idx)
    tracker = _make_excitation_tracker(hold_undetermined, theta0, coarse)
    best = _BestIterate((theta0, ws_flat0))
    # The physical tree the best iterate was evaluated on: the result's.
    best_params: Optional[dict] = None
    losses: list[float] = []
    converged = False
    vanished_warned = False
    i = 0
    for i in range(1, n_iter + 1):
        p = pmap.params(theta)
        loss, (g_t, g_s) = evaluate(theta, ws, p)
        if i == 1:
            slope0 = pmap.slope(theta)

            def lifted_at_start(cot):
                loss_c, g_p, g_w = scaled(p, ws, cot)
                return loss_c, (g_p * slope0, g_w)

            lift = _gradient_lift(lifted_at_start, (g_t, g_s), loss)
            if lift is not None:
                loss, (g_t, g_s) = evaluate(theta, ws, p)
        loss_f = float(loss)
        losses.append(loss_f)
        if not np.isfinite(loss_f) or not bool(jnp.all(jnp.isfinite(g_t))) \
                or not bool(jnp.all(jnp.isfinite(g_s))):
            raise FloatingPointError(f"non-finite loss or gradient at iteration {i}")
        state = (theta, ws)
        best.offer(state, i - 1, loss_f)
        if best.state is state:
            best_params = p
        if tracker is not None:
            # The parameter block's gradient only: the window states are
            # decision variables of this fit and are returned as the
            # optimiser left them.  Before the ``tol`` break, as in ``fit``.
            tracker.observe(unlifted(g_t))
        if callback is not None or progress is not None:
            current = pmap.returned(theta, p)
            if callback is not None:
                callback(i, loss_f, current)
            if progress is not None:
                progress(i, loss_f, current)
        vanished = loss_f == 0.0 and (bool(jnp.any(g_t != 0.0)) or bool(jnp.any(g_s != 0.0)))
        if vanished and not vanished_warned:
            vanished_warned = True
            warnings.warn(
                f"fit_multiple_shooting: the windowed loss is exactly 0.0 at "
                f"iteration {i} with a gradient that is not zero -- it underflows "
                f"its precision.  The iterates follow the gradient, which is "
                f"taken clear of the flush, but losses, best_loss and the tol "
                f"test read 0.0 there, and tol does not count it. Write the "
                f"observations in units nearer one.",
                RuntimeWarning, stacklevel=2)
        if tol > 0.0 and loss_f <= tol and not vanished:
            converged = True
            break
        it = jnp.asarray(i, theta.dtype)
        if frames is None:
            # One frame per block: the parameters' and the window states'
            # gradients are in different units.
            frames = (_adam_frame(g_t), _adam_frame(g_s))
        theta, m_t, v_t = adam(theta, m_t, v_t, g_t, frames[0], it, lr)
        # As in ``fit``: each coordinate is put back onto its range.
        theta = bounds.project(theta)
        ws, m_s, v_s = adam(ws, m_s, v_s, g_s, frames[1], it, lr_s)

    if i > 0 and not converged:
        # As in ``fit``: the last update's result was never evaluated.
        p_last = pmap.params(theta)
        state = (theta, ws)
        _offer_final_iterate(best, state, i,
                             float(evaluate(theta, ws, p_last)[0]),
                             "fit_multiple_shooting")
        if best.state is state:
            best_params = p_last
    last = theta
    theta, ws = best.state
    selected = theta
    selected_params = pmap.params(selected) if best_params is None else best_params

    def _reference():
        loss_sel, (g_sel, _) = evaluate(selected, ws, selected_params)
        return float(loss_sel), unlifted(g_sel)

    def _flatness(candidates, spanned, scale):
        # The parameter block's Hessian at the selected window states, held
        # fixed: the guard moves only ``theta``, and does so with ``ws``
        # where the optimiser left them.  Unlifted, as in ``fit``.
        lifted = 1.0 if lift is None else float(lift)

        def gradient(p_, w):
            return (plain(p_, w) if lift is None else scaled(p_, w, lift))[1]

        return _hessian_flatness(
            lambda V: _model_hvp(gradient, pmap, selected, selected_params, (ws,), V) / lifted,
            candidates, spanned, coarse, "fit_multiple_shooting", scale)

    theta, excited_rank, undetermined_drift, hold_declined = _hold_undetermined_directions(
        tracker, selected, theta0,
        _SelectedObjective(loss=lambda th: float(evaluate(th, ws)[0]),
                           reference=_reference, flatness=_flatness,
                           transformed=bounds.transformed,
                           columns=None if tracker is None else tracker.gradient_scale),
        "fit_multiple_shooting")

    # The tree the selected iterate was evaluated on, unless the guard moved it.
    evaluated = selected_params if theta is selected else pmap.params(theta)
    if i > 0:
        _warn_on_an_edge("fit_multiple_shooting", pmap, bounds, theta, evaluated, last)
        _warn_unresolved_values("fit_multiple_shooting", pmap, evaluated, "as fitted",
                                unresolved)
    return (FitResult(params=pmap.returned(theta, evaluated), losses=np.asarray(losses),
                      converged=converged,
                      n_iter=i, excited_rank=excited_rank,
                      undetermined_drift=undetermined_drift,
                      best_iteration=best.iteration, best_loss=best.loss,
                      hold_declined=hold_declined),
            unravel_ws(ws))
