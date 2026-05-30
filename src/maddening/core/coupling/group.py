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

from dataclasses import dataclass
from typing import Literal, Optional


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
    linear_solver : {"gmres", "bicgstab", "dense"}
        Backend used by the ``"ift"`` backward to solve the adjoint
        system ``(I - dF/dx)^T u = g``.  ``"gmres"`` (default) is the
        matrix-free GMRES path via lineax — the safe default for the
        non-symmetric coupling Jacobians MADDENING produces.
        ``"bicgstab"`` is a matrix-free alternative that can be more
        robust on some non-symmetric systems.  ``"dense"`` is the
        legacy ``jacrev + jnp.linalg.solve`` path, kept as a triage
        fallback (O(N^2) memory, O(N^3) compute).  The
        ``MADDENING_IFT_DENSE_SOLVE=1`` env var forces ``"dense"``
        regardless of this setting and overrides for triage.
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
    linear_solver: Literal["gmres", "bicgstab", "dense"] = "gmres"
