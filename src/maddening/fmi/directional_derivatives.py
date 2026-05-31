"""``fmi3GetDirectionalDerivative`` wired through ``jax.jvp`` / ``jax.vjp``.

This is the load-bearing v0.3.0 §A1 deliverable: MADDENING's FMU
shim exposes exact directional derivatives by wrapping JAX's autodiff
primitives.  FMI 2.0 had no equivalent; choosing FMI 3.0 here is what
lets the MADDENING FMU keep its differentiable-simulation advantage
in the co-simulation ecosystem.

API shape (mirroring FMI 3.0 §4.7.6 ``fmi3GetDirectionalDerivative``):

    out_seed = get_directional_derivative(
        unknown_fn,
        kind=DirectionalDerivativeKind.FORWARD,
        x=primal_inputs,
        v=seed_inputs,
    )

* ``unknown_fn(x) -> y`` is the function whose Jacobian we want.
* ``x`` is the dict (or pytree) of primal inputs at which the
  derivative is evaluated.
* ``v`` is the seed.  For forward mode (FMI's default), ``v`` is the
  tangent of ``x`` and the output is the tangent of ``y``.  For
  reverse mode, ``v`` is the cotangent of ``y`` and the output is the
  cotangent of ``x``.

The FMU sidecar process (see :mod:`maddening.fmi.sidecar`) marshals
the FMI runtime's request — ``(unknownValueReferences,
seedValueReferences, deltaV)`` — into one of these calls.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

import jax

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability


class DirectionalDerivativeKind(Enum):
    """Which JAX primitive to use.

    The FMI 3.0 standard's ``fmi3GetDirectionalDerivative`` semantics
    are forward-mode (``J @ v``).  Reverse-mode (``Jᵀ @ v``) is
    needed for adjoint computations downstream — the substrate
    supports both, MADDENING's FMU sidecar selects via the directive.
    """
    FORWARD = "forward"   # jax.jvp:  out_tangent = J @ in_tangent
    REVERSE = "reverse"   # jax.vjp:  in_cotangent = J^T @ out_cotangent


@stability(StabilityLevel.EVOLVING)
def get_directional_derivative(
    unknown_fn: Callable[[Any], Any],
    *,
    kind: DirectionalDerivativeKind = DirectionalDerivativeKind.FORWARD,
    x: Any,
    v: Any,
) -> Any:
    """Compute a directional derivative of ``unknown_fn`` at ``x`` in direction ``v``.

    Parameters
    ----------
    unknown_fn : callable
        A JAX-traceable function ``unknown_fn(x) -> y``.  Both ``x``
        and ``y`` may be arbitrary JAX pytrees.
    kind : DirectionalDerivativeKind
        Forward mode (default) or reverse mode.
    x : pytree of jax arrays
        Primal input.
    v : pytree of jax arrays
        Seed.  In ``FORWARD`` mode, ``v`` is the tangent of ``x``
        (same pytree structure / shape / dtype as ``x``).  In
        ``REVERSE`` mode, ``v`` is the cotangent of ``y`` (same
        pytree structure / shape / dtype as ``y = unknown_fn(x)``).

    Returns
    -------
    pytree
        In ``FORWARD`` mode: ``J @ v``, same pytree as ``y``.
        In ``REVERSE`` mode: ``J^T @ v``, same pytree as ``x``.

    Notes
    -----
    The function is a thin wrapper — there is no MADDENING-specific
    state.  Implementations of the FMU sidecar must marshal the FMI
    runtime's flat value-reference arrays into / out of the pytree
    shapes; see :mod:`maddening.fmi.sidecar` for the round-tripping
    convention.

    Stability
    ---------
    Tagged ``@stability(EVOLVING)``.  The signature is settled but
    we may add an ``options`` kwarg for FMI 3.0-specific knobs
    (per-clock event tracking, FMU-state preservation across calls)
    before M4 (v0.9.0) freezes the public surface.
    """
    if kind == DirectionalDerivativeKind.FORWARD:
        _, out_tangent = jax.jvp(unknown_fn, (x,), (v,))
        return out_tangent
    if kind == DirectionalDerivativeKind.REVERSE:
        y, vjp_fn = jax.vjp(unknown_fn, x)
        (x_cotangent,) = vjp_fn(v)
        return x_cotangent
    raise ValueError(
        f"get_directional_derivative: unknown kind={kind!r}; "
        "use DirectionalDerivativeKind.FORWARD or REVERSE.",
    )


__all__ = [
    "DirectionalDerivativeKind",
    "get_directional_derivative",
]
