"""Adaptive solvers with a frozen-active-set adjoint.

:class:`AdaptiveNode` is the base class; see
``docs/developer_guide/adaptive_node.md`` for the subclass contract and
``docs/algorithm_guide/nodes/adaptive_node.md`` for the method.
"""

from maddening.nodes.adaptive.base import AdaptiveNode, AdaptiveNodeBlindnessError

__all__ = ["AdaptiveNode", "AdaptiveNodeBlindnessError"]
