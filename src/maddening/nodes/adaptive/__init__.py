"""Adaptive PDE node framework.

This subpackage exposes the basis-agnostic ``AdaptiveNode`` base class
and its companions.  The framework provides the frozen-active-set
adjoint infrastructure (pad-to-max-DOF buffer + active mask in state),
the blindness diagnostic with anisotropic ``symmetry_break`` protocol,
and the cold-start contract.  Concrete subclasses provide the
basis-specific machinery: ``compute_active_set`` (selection),
``solve_frozen`` (inner solve), and the
``compute_full_basis_gradient`` / ``_get_theta`` / ``_set_theta``
hooks that connect the framework to the subclass's PDE.

The Selection-Equivariance Theorem (extending Palais 1979 to
``J_frozen``) is stated in the ``AdaptiveNode`` module docstring.
The full spike findings are in
``plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md``.

Public surface:

* :class:`AdaptiveNode` — basis-agnostic base class.
* :class:`AdaptiveNodeBlindnessError` — raised by ``initial_state``
  when a persistent Palais fixed point is detected.

Toy subclasses (added in later milestones):

* :class:`TopKAdaptiveNode` — non-local sine basis.
* :class:`HierarchicalHatAdaptiveNode` — local dyadic hat basis.
"""

from __future__ import annotations

from maddening.nodes.adaptive.base import (
    AdaptiveNode,
    AdaptiveNodeBlindnessError,
)

__all__ = [
    "AdaptiveNode",
    "AdaptiveNodeBlindnessError",
]
