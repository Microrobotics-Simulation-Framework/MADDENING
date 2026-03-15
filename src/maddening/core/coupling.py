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
fixed relaxation, IQN-ILS), Jacobi iteration mode, and subcycling
for mixed-timestep coupling groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
    """
    nodes: frozenset[str]
    max_iterations: int = 10
    tolerance: float = 1e-6
    convergence_norm: str = "l2"
    atol: float = 1e-8
    rtol: float = 1e-6
    diagnostics: bool = False
    acceleration: str = "none"
    relaxation: float = 1.0
    iteration_mode: str = "gauss-seidel"
    accelerated_fields: Optional[dict[str, tuple[str, ...]]] = None
    subcycling: bool = False
    boundary_interpolation: str = "linear"
    jacobian_reuse: int = 0
    waveform_iterations: int = 1
