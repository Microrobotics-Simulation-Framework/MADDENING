"""
Iterative coupling for algebraic loops (Gauss-Seidel fixed-point).

When nodes form a cycle, the default behaviour is *staggering*: back-edges
read previous-timestep values.  For strongly-coupled subsystems this can be
inaccurate or unstable.

A ``CouplingGroup`` wraps a set of cyclic nodes in a ``jax.lax.fori_loop``
that iterates the group each timestep until the state change drops below a
tolerance (or ``max_iterations`` is reached).  All edges within the group
become *forward* edges during iteration, giving Gauss-Seidel convergence.

Supports multiple convergence norms, acceleration methods (Aitken,
fixed relaxation, IQN-ILS, IQN-IMVJ), Jacobi iteration mode, and
subcycling for mixed-timestep coupling groups.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal, Optional, Union, get_args, get_origin, get_type_hints


@dataclass(frozen=True)
class CouplingGroup:
    """Configuration for an iteratively-coupled group of nodes.

    Parameters
    ----------
    nodes : frozenset[str]
        Names of the nodes that participate in the coupling group.
        All must belong to the same graph and form (part of) a cycle.
    max_iterations : int
        Upper bound on iterations per timestep.
    tolerance : float
        Convergence threshold on the L2 norm of state change between
        successive iterations.  Used when ``convergence_norm="l2"``.
    convergence_norm : str
        Norm used to check convergence.  ``"l2"`` uses a global L2
        norm with ``tolerance`` as threshold.  ``"mixed"`` uses a
        per-field mixed absolute/relative norm (converged when the
        norm <= 1.0).  ``"interface"`` checks consistency of
        coupling-edge values between iterations.
    atol : float
        Absolute tolerance for the ``"mixed"`` norm.
    rtol : float
        Relative tolerance for the ``"mixed"`` norm.
    diagnostics : bool
        If True, store iteration count and final residual in the
        ``_meta`` key of the state dict after each step.
    acceleration : str
        Acceleration method.  ``"none"`` is plain fixed-point,
        ``"aitken"`` uses Aitken delta-squared relaxation,
        ``"fixed"`` uses constant under-relaxation with ``relaxation``
        as the omega parameter, ``"iqn-ils"`` uses Interface
        Quasi-Newton with Inverse Least Squares, ``"iqn-imvj"``
        uses IQN-ILS with multi-timestep Jacobian reuse.
    relaxation : float
        Constant relaxation factor for ``acceleration="fixed"``.
        ``1.0`` is no relaxation, ``< 1`` is under-relaxation,
        ``> 1`` is over-relaxation.
    iteration_mode : str
        ``"gauss-seidel"`` updates nodes sequentially (each sees
        current-iteration values from earlier nodes).
        ``"jacobi"`` updates all nodes from the frozen
        previous-iteration state.
    accelerated_fields : dict or None
        For ``"iqn-ils"``: which fields per node participate in the
        quasi-Newton problem.  ``None`` auto-detects from coupling
        edges (interface fields only).
    subcycling : bool
        If True, allow mixed timesteps within the coupling group.
        Fast nodes take multiple sub-steps per coupling iteration.
    boundary_interpolation : str
        Time interpolation of boundary conditions during subcycling.
        ``"constant"`` holds values constant, ``"linear"`` linearly
        interpolates between previous and current iteration values,
        ``"quadratic"`` uses quadratic Lagrange interpolation through
        three successive iteration values (falls back to linear on
        the first iteration).
    jacobian_reuse : int
        For ``"iqn-imvj"``: number of V/W columns retained from the
        previous timestep.  ``0`` means no reuse (same as IQN-ILS).
    waveform_iterations : int
        For subcycling groups: number of waveform relaxation
        iterations.  ``1`` is current behaviour (single pass),
        ``> 1`` iterates over entire sub-step windows.
    predictor : str
        Extrapolation of the coupling initial guess from previous
        converged states.  ``"none"`` uses the current state (default),
        ``"linear"`` uses linear extrapolation from the last two
        converged states, ``"quadratic"`` uses quadratic extrapolation
        from the last three converged states.  Reduces iteration count
        for smoothly varying problems.
    solver : {"fori", "ift"}
        How the fixed-point iteration is solved.  ``"fori"`` (default)
        runs a static ``jax.lax.fori_loop`` for ``max_iterations`` and
        differentiates straight through the iterates.  ``"ift"`` runs
        a ``jax.lax.while_loop`` that terminates early on convergence
        and uses implicit-function-theorem differentiation in the
        backward pass (more accurate gradients in the
        partially-converged regime, plus a meaningful forward speed-up
        when convergence is reached before ``max_iterations``).
        Supported with ``acceleration`` in ``{"none", "aitken",
        "iqn-imvj"}`` and ``iteration_mode="gauss-seidel"``; other
        combinations silently fall back to ``"fori"``.
    linear_solver : {"gmres", "dense"}
        Backend used by the ``"ift"`` backward to solve the adjoint
        system ``(I - dF/dx)^T u = g``.  ``"gmres"`` (default) is the
        matrix-free GMRES path via lineax — the safe default for the
        non-symmetric coupling Jacobians MADDENING produces.
        ``"dense"`` is the legacy ``jacrev + jnp.linalg.solve`` path,
        promoted from an env-var-gated fallback to a first-class
        config option (O(N^2) memory, O(N^3) compute — only viable
        for small groups).  The ``MADDENING_IFT_DENSE_SOLVE=1`` env
        var forces ``"dense"`` regardless of this setting and
        overrides for triage.

        Note on BiCGStab: lineax 0.0.7 ships ``lineax.BiCGStab``, but
        it returns NaN when driving a ``FunctionLinearOperator`` (the
        matrix-free shape our backward uses) — including on
        well-conditioned operators like ``0.5*I``.  The
        ``"bicgstab"`` option is therefore *not* exposed here.  See
        ``_ift_solve_bwd`` and
        ``tests/core/test_coupling_ift_lineax.py`` for the
        investigation notes.
    """
    nodes: frozenset[str]
    max_iterations: int = 10
    tolerance: float = 1e-6
    convergence_norm: str = "l2"
    atol: float = 1e-8
    rtol: float = 1e-6
    diagnostics: bool = False
    acceleration: Literal[
        "none", "aitken", "fixed", "iqn-ils", "iqn-imvj"
    ] = "none"
    relaxation: float = 1.0
    iteration_mode: Literal["gauss-seidel", "jacobi"] = "gauss-seidel"
    accelerated_fields: Optional[dict[str, tuple[str, ...]]] = None
    subcycling: bool = False
    boundary_interpolation: Literal[
        "constant", "linear", "quadratic"
    ] = "linear"
    jacobian_reuse: int = 0
    waveform_iterations: int = 1
    predictor: Literal["none", "linear", "quadratic"] = "none"
    solver: Literal["fori", "ift"] = "fori"
    linear_solver: Literal["gmres", "dense"] = "gmres"

    def __post_init__(self) -> None:
        """Validate that ``Literal``-typed string fields hold permitted values.

        Without this check a typo like ``acceleration="aitkin"`` silently
        sets the field to that string; runtime dispatch (``if group.acceleration
        == "aitken": ...``) simply fails to match and the group quietly falls
        back to whatever the default branch is.  The Literal annotation is
        purely a type-checker hint at runtime, so we re-derive the valid sets
        from ``typing.get_args`` on the annotation and raise ``ValueError`` on
        anything outside the permitted set.  Adding a new option to a Literal
        automatically extends the validator.
        """
        # ``from __future__ import annotations`` is active at module top, so
        # annotations are stringified.  ``get_type_hints`` resolves them
        # against this module's globals.
        hints = get_type_hints(type(self))
        for f in fields(self):
            ann = hints.get(f.name)
            valid = _literal_options(ann)
            if valid is None:
                continue  # not a Literal (or Optional[Literal]) — skip
            value = getattr(self, f.name)
            # Optional[Literal[...]] allows None as a valid sentinel.
            if value is None and _is_optional(ann):
                continue
            if value not in valid:
                raise ValueError(
                    f"CouplingGroup.{f.name}={value!r} is not a valid "
                    f"option; expected one of {valid!r}"
                )


def _literal_options(ann: object) -> Optional[tuple]:
    """Return the ``Literal`` options of ``ann``, or ``None`` if not a Literal.

    Handles both bare ``Literal[...]`` and ``Optional[Literal[...]]``
    (which normalises to ``Union[Literal[...], None]``).
    """
    if ann is None:
        return None
    if get_origin(ann) is Literal:
        return get_args(ann)
    if get_origin(ann) is Union:
        for arg in get_args(ann):
            if get_origin(arg) is Literal:
                return get_args(arg)
    return None


def _is_optional(ann: object) -> bool:
    """True iff ``ann`` is ``Union[..., None]`` (i.e. ``Optional[...]``)."""
    return get_origin(ann) is Union and type(None) in get_args(ann)
