"""System identification helpers built on the graph parameter pytree.

Two pieces, both pure JAX so they compose with ``jax.jit`` / ``jax.grad``:

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

Both take the ``params`` pytree exactly as ``GraphManager.params`` holds
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

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

_META_KEY = "_meta"


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


def _group_thresholds(gm) -> list[tuple[str, float]]:
    out = []
    for g in gm._coupling_groups:  # noqa: SLF001
        key = "+".join(sorted(g.nodes))
        thr = 1.0 if g.convergence_norm in ("mixed", "interface") else float(g.tolerance)
        out.append((f"coupling_{key}_residual", thr))
    return out


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
) -> jnp.ndarray:
    """Teacher-forced windowed squared-error loss of a graph against data.

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
        Samples per window.  ``T - 1`` must be a multiple of it.  Every
        window starts from the ground-truth sample at its start and
        integrates ``window * sample_every`` steps; the loss compares the
        ``window`` simulated samples against the next ``window``
        observations.
    sample_every : int
        Base steps between consecutive observation samples.
    external_inputs : dict, optional
        Static external inputs (as in ``run_scan``).
    mask_unconverged : bool
        Multiply a window's loss by 0 when any coupling group exited at
        ``max_iterations`` unconverged during it (the IFT gradient is
        unreliable there).  Uses the always-on residual in ``_meta``.

    Returns
    -------
    jnp.ndarray
        Scalar: the sum over windows and samples of the squared error of
        ``obs_fn`` outputs.
    """
    if gm._dirty or gm._compiled_step is None:  # noqa: SLF001
        gm.compile()
    step_fn = gm._build_step_fn()  # noqa: SLF001
    ext = external_inputs if external_inputs is not None else gm._default_external_inputs()  # noqa: SLF001

    observations = {k: v for k, v in observations.items() if k != _META_KEY}
    T = _leading_len(observations)
    if window <= 0 or (T - 1) % window != 0:
        raise ValueError(
            f"window={window} must divide T-1={T - 1} (T={T} samples)"
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
        for key, thr in thresholds:
            if key in meta:
                ok = ok & (meta[key] <= thr)
        return ok

    def _advance_one_sample(carry, _):
        def inner(c, _):
            s, ok = c
            s = step_fn(s, ext, params)
            return (s, ok & _converged(s)), None

        (state, ok), _ = jax.lax.scan(inner, carry, None, length=sample_every)
        user = {k: v for k, v in state.items() if k != _META_KEY}
        return (state, ok), user

    def _window(w, _):
        start = w * window
        obs_start = jax.tree.map(
            lambda x: jax.lax.dynamic_index_in_dim(x, start, keepdims=False),
            observations,
        )
        state0 = _state_from_obs(obs_start, start)
        (_, ok), sim = jax.lax.scan(
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
        return w + 1, loss_w

    _, losses = jax.lax.scan(_window, jnp.int32(0), None, length=n_windows)
    return jnp.sum(losses)


@dataclass(frozen=True)
class FIMReport:
    """Fisher information of a residual with respect to the parameters.

    ``eigvals`` ascend; ``eigvecs[:, i]`` is the direction for
    ``eigvals[i]`` in the (possibly relatively scaled) parameter space
    ordered as ``param_names``.  ``crb`` is the Cramér–Rao lower bound on
    each parameter's variance (unit noise variance; relative variance
    under ``scale="relative"``), ``NaN`` where the FIM is singular.
    """
    fim: jnp.ndarray
    eigvals: jnp.ndarray
    eigvecs: jnp.ndarray
    cond: float
    crb: jnp.ndarray
    param_names: tuple[str, ...]

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


def fim(
    residual_fn: Callable[[dict], Any],
    params: dict,
    *,
    scale: Optional[str] = "relative",
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
        sensitivities.
    """
    flat, unravel = ravel_pytree(params)

    def _r(theta):
        return ravel_pytree(residual_fn(unravel(theta)))[0]

    J = jax.jacfwd(_r)(flat)
    if scale == "relative":
        J = J * flat[None, :]
    elif scale is not None:
        raise ValueError(f"scale must be 'relative' or None, got {scale!r}")

    F = J.T @ J
    eigvals, eigvecs = jnp.linalg.eigh(F)
    lo, hi = float(eigvals[0]), float(eigvals[-1])
    cond = float("inf") if lo <= 0.0 else hi / lo
    crb = jnp.diag(jnp.linalg.pinv(F))
    crb = jnp.where(jnp.isfinite(crb), crb, jnp.nan)
    return FIMReport(
        fim=F, eigvals=eigvals, eigvecs=eigvecs, cond=cond, crb=crb,
        param_names=_param_names(params),
    )
