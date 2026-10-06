"""The compiled adaptive-timestep scan (``_build_adaptive_scan``).

Moved verbatim out of ``maddening.core.graph_manager``.  Private.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from maddening.core._graph_specs import _META_KEY
from maddening.core.coupling._reports import _strict_error_if


def _build_adaptive_scan(
    dt_step_fn: Callable,
    user_state: Callable[[dict], dict],
    max_steps: int,
    safety: float,
    order: int,
    min_factor: float,
    max_factor: float,
    on_trace: Callable[[], None],
) -> Callable:
    """Build the jitted adaptive-timestepping scan for ``run_adaptive_scan``.

    Kept at module level (and built through
    :meth:`GraphManager._cached_scan`) so the program is compiled once
    per graph compile rather than once per call.  ``t_end``, the
    tolerances and the timestep bounds arrive as the traced ``knobs``
    tuple, so changing any of them reuses the compilation; the PI
    controller's constants are closed over and therefore belong in the
    cache key.

    Parameters
    ----------
    dt_step_fn : callable
        ``(state, external_inputs, dt, params) -> (state, verdicts)``,
        built with ``collect_strict=True``: ``strict_convergence`` is
        checked here, on ``accepted & ~done``, so a solve the scan
        discards -- the error estimate's full step, a rejected attempt,
        a step past ``t_end`` -- never raises.
    user_state : callable
        Strips the internal ``_meta`` key from a state dict.
    max_steps : int
        Scan length.
    safety, order, min_factor, max_factor
        PI step-size controller constants.
    on_trace : callable
        Called once per Python trace of the program, for
        :attr:`GraphManager.scan_trace_count`.

    Returns
    -------
    Callable
        ``(state, ext, params, knobs) -> ((state, t, dt, n), history)``.
    """
    from maddening.core.simulation.adaptive import _tree_error_norm, step_decision

    strict_messages = dict(getattr(dt_step_fn, "strict_messages", {}))
    strict_mesh = getattr(dt_step_fn, "strict_mesh", None)
    fold_kept_halves = getattr(dt_step_fn, "fold_kept_halves",
                               lambda _first, second: second)

    def adaptive_scan(init_state, ext, params, knobs):
        on_trace()
        t_end, dt_initial, atol, rtol, dt_min, dt_max = knobs

        def scan_body(carry, _unused):
            state, t, dt, n_accepted = carry

            # Clamp dt to not overshoot
            dt = jnp.minimum(dt, t_end - t)
            dt = jnp.maximum(dt, dt_min)

            # Check if we've already reached t_end
            done = t >= t_end

            # Full step + two half-steps
            state_full, _discarded = dt_step_fn(state, ext, dt, params)
            half_dt = dt / 2.0
            state_half_1, verdicts_1 = dt_step_fn(state, ext, half_dt, params)
            state_half, verdicts_2 = dt_step_fn(state_half_1, ext, half_dt, params)
            # The report covers both kept half steps, as the strict check does.
            state_half = fold_kept_halves(state_half_1, state_half)

            # Error estimate
            user_full = {k: v for k, v in state_full.items() if k != _META_KEY}
            user_half = {k: v for k, v in state_half.items() if k != _META_KEY}
            error_norm = _tree_error_norm(user_half, user_full, atol, rtol)

            # The one acceptance rule and PI controller ``run_adaptive``
            # uses too (``adaptive.step_decision``).
            accepted, _forced, dt_next, _factor = step_decision(
                error_norm, dt, dt_min, dt_max, safety=safety, order=order,
                min_factor=min_factor, max_factor=max_factor, xp=jnp)

            # If done, keep state unchanged; if accepted, use half-step result
            new_state = jax.tree.map(
                lambda s, h: jnp.where(done, s, jnp.where(accepted, h, s)),
                state, state_half,
            )
            if strict_messages:
                # ``strict_convergence`` on the solves this iteration keeps:
                # the two half steps, when accepted and not already done.
                # On every device of a step spanning several (MADD-ANO-162).
                kept = jnp.logical_and(accepted, jnp.logical_not(done))
                for key, (nonfinite_msg, unconverged_msg) in strict_messages.items():
                    nonfinite = jnp.logical_or(verdicts_1[key][0], verdicts_2[key][0])
                    unconverged = jnp.logical_or(verdicts_1[key][1], verdicts_2[key][1])
                    new_state = _strict_error_if(
                        new_state, jnp.logical_and(kept, nonfinite), nonfinite_msg,
                        strict_mesh,
                    )
                    new_state = _strict_error_if(
                        new_state,
                        jnp.logical_and(kept, jnp.logical_and(
                            jnp.logical_not(nonfinite), unconverged)),
                        unconverged_msg,
                        strict_mesh,
                    )
            new_t = jnp.where(done, t, jnp.where(accepted, t + dt, t))
            new_dt = jnp.where(done, dt, dt_next)
            new_n = jnp.where(
                done, n_accepted, jnp.where(accepted, n_accepted + 1, n_accepted),
            )

            # Output the state for history (no-op state if not accepted)
            output_state = user_state(new_state)

            return (new_state, new_t, new_dt, new_n), output_state

        init_carry = (
            init_state,
            jnp.array(0.0),
            dt_initial,
            jnp.array(0, dtype=jnp.int32),
        )
        return jax.lax.scan(scan_body, init_carry, None, length=max_steps)

    return jax.jit(adaptive_scan)
