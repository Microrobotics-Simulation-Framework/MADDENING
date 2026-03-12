"""
Iterative coupling for algebraic loops (Gauss-Seidel fixed-point).

When nodes form a cycle, the default behaviour is *staggering*: back-edges
read previous-timestep values.  For strongly-coupled subsystems this can be
inaccurate or unstable.

A ``CouplingGroup`` wraps a set of cyclic nodes in a ``jax.lax.while_loop``
that iterates the group each timestep until the state change drops below a
tolerance (or ``max_iterations`` is reached).  All edges within the group
become *forward* edges during iteration, giving Gauss-Seidel convergence.
"""

from __future__ import annotations

from dataclasses import dataclass


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
        successive iterations.
    """
    nodes: frozenset[str]
    max_iterations: int = 10
    tolerance: float = 1e-6
