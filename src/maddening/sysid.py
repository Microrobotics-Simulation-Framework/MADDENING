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

import numbers
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
# ``_spec_for`` is the one place that resolves a params path to its
# ParamSpec (the same walk ``trainable_mask`` / ``constrain`` use);
# duplicating it here would be a second definition of "which spec
# governs this leaf".
from maddening.core.params import _spec_for

_META_KEY = "_meta"


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
        loss_w = sum(jax.tree.leaves(sq))
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
    ``rank_rtol`` defaults to ``n * eps`` for ``n`` parameters at the
    matrix's own precision -- the relative form
    ``numpy.linalg.matrix_rank`` uses.  ``rank < len(param_names)`` says
    the data leaves that many independent parameter combinations
    undetermined.  Unlike ``cond`` the verdict does not move when the
    residual is rescaled (by ``noise_std``, say), because the threshold
    scales with the matrix; a float32 ``cond`` can flip between a finite
    number and ``inf`` under exactly that rescaling.

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
    """
    fim: jnp.ndarray
    eigvals: jnp.ndarray
    eigvecs: jnp.ndarray
    rank: int
    cond: float
    crb: jnp.ndarray
    param_names: tuple[str, ...]
    zero_scaled: tuple[str, ...] = ()

    def least_identifiable(self) -> tuple[str, float]:
        """Name and weight of the largest component of the weakest direction."""
        v = np.abs(np.asarray(self.eigvecs[:, 0]))
        i = int(np.argmax(v))
        return self.param_names[i], float(v[i])


def _param_names(params) -> tuple[str, ...]:
    names: list[str] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        base = jax.tree_util.keystr(path)
        n = int(np.asarray(leaf).size)
        if n == 1:
            names.append(base)
        else:
            names.extend(f"{base}[{i}]" for i in range(n))
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
        n = int(np.asarray(leaf).size)
        if bool(flag):
            idx.extend(range(offset, offset + n))
        offset += n
    if not idx:
        raise ValueError("mask selects no parameters")
    return np.asarray(idx)


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
        that would make it fittable.
    """
    if mask is None:
        return gm.trainable_mask(params)
    entries = jax.tree_util.tree_flatten_with_path(params)[0]
    flags = _mask_flags(params, mask)
    specs = gm.param_specs()
    frozen = []
    for (path, _), flag in zip(entries, flags):
        spec = _spec_for(specs, path)
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
    """
    if noise_std is None:
        return None
    flat_r = ravel_pytree(residual)[0]
    is_scalar = isinstance(noise_std, numbers.Real) or (
        isinstance(noise_std, (np.ndarray, jax.Array)) and noise_std.ndim == 0
    )
    if is_scalar:
        sig = jnp.asarray(noise_std, dtype=flat_r.dtype)
        _check_noise_std(sig, noise_std)
        return 1.0 / sig
    sig = jax.tree.map(
        lambda leaf, sd: jnp.broadcast_to(jnp.asarray(sd, dtype=leaf.dtype), jnp.shape(leaf)),
        residual, noise_std,
    )
    flat_sig = ravel_pytree(sig)[0]
    _check_noise_std(flat_sig, noise_std)
    return 1.0 / flat_sig


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


def _rank_and_crb(eigvals, eigvecs, rank_rtol: Optional[float]):
    """``(rank, crb)`` from the eigendecomposition of a Fisher matrix.

    Parameters
    ----------
    eigvals, eigvecs : array
        Ascending eigenvalues and their orthonormal columns, as
        ``jnp.linalg.eigh`` returns them for the symmetric PSD ``F``.
    rank_rtol : float, optional
        Eigenvalues at or below ``rank_rtol * eigvals[-1]`` count as
        zero.  ``None`` uses ``n * eps`` at the matrix's own precision.

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
    and ``F`` is square).  The cutoff sits at ``eps`` rather than
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
    dtype = jnp.asarray(eigvals).dtype
    ev = np.asarray(eigvals, dtype=np.float64)
    vecs = np.asarray(eigvecs, dtype=np.float64)
    n = int(ev.size)
    eps = float(np.finfo(dtype).eps)
    if rank_rtol is None:
        rank_rtol = n * eps
    else:
        rank_rtol = float(rank_rtol)
        if not np.isfinite(rank_rtol) or rank_rtol < 0.0:
            raise ValueError(
                "rank_rtol must be a finite non-negative number, got "
                f"{rank_rtol!r}")
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


@stability(StabilityLevel.EVOLVING)
def fim(
    residual_fn: Callable[[dict], Any],
    params: dict,
    *,
    scale: Optional[str] = "relative",
    mask: Optional[dict] = None,
    noise_std: Optional[Any] = None,
    rank_rtol: Optional[float] = None,
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
    scale : {"relative", None}
        ``"relative"`` (default) multiplies each column of ``J`` by the
        parameter's value, i.e. sensitivities to *relative* changes, so
        parameters in different units are comparable and the condition
        number is not dominated by units.  ``None`` uses raw
        sensitivities.  A parameter whose value is exactly ``0.0`` has
        no relative scale: its column vanishes and it reads as
        unidentifiable however well the data determine it.  Those
        parameters are named in :attr:`FIMReport.zero_scaled`; ask
        ``scale=None`` for the absolute question about them.
    mask : pytree of bool, optional
        Same structure as ``params``; only leaves marked ``True`` are
        treated as parameters (``GraphManager.trainable_mask()``).  The
        report's ``param_names`` / matrix are restricted accordingly.
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
    rank_rtol : float, optional
        Relative tolerance deciding which directions the data resolves:
        an eigenvalue at or below ``rank_rtol * max(eigvals)`` counts as
        zero, and the parameters with support in the directions it
        rejects get an infinite ``crb``.  The default, ``n * eps`` for
        ``n`` parameters at the matrix's precision, is the relative form
        ``numpy.linalg.matrix_rank`` uses and the point below which
        ``eigh`` is reporting its own rounding error; raise it to
        declare a merely ill-conditioned direction unidentifiable too.

    Returns
    -------
    FIMReport
        Notably ``rank``, the number of resolved directions, and
        ``crb``, which is ``+inf`` for a parameter the unresolved ones
        leave undetermined.  See :class:`FIMReport`.

    Raises
    ------
    ValueError
        If ``scale``, ``rank_rtol`` or ``noise_std`` is not a value this
        function can answer for (in particular a σ that is zero,
        negative, non-finite, or underflows the residual's dtype).
    FloatingPointError
        If ``F`` comes out non-finite -- a diverged rollout, an
        overflowing Jacobian, a residual holding a ``NaN``.  There is no
        rank, condition number or bound to read from a NaN
        decomposition, so this raises rather than reporting one.
    """
    if scale not in ("relative", None):
        raise ValueError(f"scale must be 'relative' or None, got {scale!r}")
    flat, unravel = ravel_pytree(params)
    idx = _masked_indices(params, mask)
    r0 = residual_fn(params)
    inv_sigma = _inverse_noise_std(noise_std, r0)

    def _r(theta):
        full = theta if idx is None else flat.at[idx].set(theta)
        r = ravel_pytree(residual_fn(unravel(full)))[0]
        return r if inv_sigma is None else r * inv_sigma

    theta0 = flat if idx is None else flat[idx]
    J = jax.jacfwd(_r)(theta0)
    names = _param_names(params)
    if idx is not None:
        names = tuple(names[i] for i in idx)
    zero_scaled: tuple[str, ...] = ()
    if scale == "relative":
        # A column scaled by a parameter sitting at exactly zero is zero,
        # so the parameter drops out of F however well the data determine
        # it.  Report that as what it is rather than as a verdict on the
        # data: SpringDamperNode's initial_velocity defaults to 0.0, so
        # this is the first thing a user meets on the default scale.
        zero_scaled = tuple(
            nm for nm, at_zero in zip(names, np.asarray(theta0) == 0.0)
            if bool(at_zero)
        )
        J = J * theta0[None, :]

    F = J.T @ J
    if not bool(jnp.all(jnp.isfinite(F))):
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
    eigvals, eigvecs = jnp.linalg.eigh(F)
    lo, hi = float(eigvals[0]), float(eigvals[-1])
    cond = float("inf") if lo <= 0.0 else hi / lo
    rank, crb = _rank_and_crb(eigvals, eigvecs, rank_rtol)
    return FIMReport(
        fim=F, eigvals=eigvals, eigvecs=eigvecs, rank=rank, cond=cond,
        crb=crb, param_names=names, zero_scaled=zero_scaled,
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
    """
    params: dict
    losses: np.ndarray
    converged: bool
    n_iter: int


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
) -> FitResult:
    """Adam on ``loss_fn(params)`` respecting the graph's :class:`ParamSpec`.

    The optimiser works in the unconstrained coordinates of
    ``GraphManager.unconstrain`` (log for positive constants, logit for
    intervals), so a positive parameter cannot cross zero and a bounded
    one cannot leave its interval, and it only moves the leaves the
    ``mask`` marks trainable (default ``gm.trainable_mask()``: the
    nodes' declarations plus ``gm.set_param_spec`` overrides; a ``mask``
    may narrow that set but not widen it).  This is
    what stops a fit from wandering along an unidentifiable direction
    through a parameter the data cannot see — freeze it with
    ``gm.set_param_spec(node, key, ParamSpec(trainable=False))`` after
    :func:`fim` has named it.

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
    """
    _check_adam_hyper(n_iter, lr, tol, betas, eps, notify_every)
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

    final = to_params(theta)
    return FitResult(
        params=final, losses=np.asarray(losses), converged=converged, n_iter=i,
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
    start = gm._params_or_default(params)  # noqa: SLF001
    gm.check_params(start)
    mask = _resolve_mask(gm, start, mask)
    u0 = gm.unconstrain(start)
    flat_u, unravel = ravel_pytree(u0)
    idx = _masked_indices(start, mask)
    if idx is None:
        idx = np.arange(flat_u.size)
    to_params = _physical_params(gm, start, flat_u, unravel, idx)
    theta = flat_u[idx]
    progress = _progress_notifier(gm, "lm", n_iter, notify_every)

    r_probe = residual_fn(gm.constrain(u0))
    inv_sigma = _inverse_noise_std(noise_std, r_probe)

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
    losses: list[float] = []
    converged = False
    i = 0
    for i in range(1, n_iter + 1):
        r, J = residual_and_jac(theta)
        loss = 0.5 * float(jnp.sum(r * r))
        if not np.isfinite(loss) or not bool(jnp.all(jnp.isfinite(J))):
            raise FloatingPointError(f"non-finite residual or Jacobian at iteration {i}")
        losses.append(loss)
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

    final = to_params(theta)
    return FitResult(params=final, losses=np.asarray(losses), converged=converged, n_iter=i)


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
) -> tuple[FitResult, dict]:
    """Multiple-shooting fit: Adam jointly over the trainable params (in
    unconstrained coordinates) and the free per-window initial states.

    Compared with :func:`fit` on the teacher-forced :func:`windowed_loss`,
    the window starts are decision variables and a continuity penalty
    (``continuity_weight``) joins consecutive windows, so the optimum is a
    single continuous trajectory and noisy observations at window starts
    do not seed every window with measurement error.

    Returns ``(FitResult, window_states)``.
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

    final = to_params(theta)
    return (FitResult(params=final, losses=np.asarray(losses), converged=converged, n_iter=i),
            unravel_ws(ws))
