"""Coupling subsystem — iterative coupling, convergence, acceleration, helpers.

Backward-compatible re-exports so that existing imports continue to work::

    from maddening.core.coupling import CouplingGroup        # works
    from maddening.core.coupling.group import CouplingGroup   # also works
"""

from maddening.core.coupling.group import CouplingGroup

__all__ = ["CouplingGroup"]
