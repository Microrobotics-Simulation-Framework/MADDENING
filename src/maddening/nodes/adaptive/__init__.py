"""Adaptive solvers with a frozen-active-set adjoint.

:class:`AdaptiveNode` is the base class; see
``docs/developer_guide/adaptive_node.md`` for the subclass contract and
``docs/algorithm_guide/nodes/adaptive_node.md`` for the method.
:class:`WaveletAdaptiveNode` is the first concrete subclass (an
interpolating-wavelet elliptic solver); see
``docs/algorithm_guide/nodes/wavelet_adaptive_node.md``.
"""

from maddening.nodes.adaptive.base import (
    AdaptiveNode,
    AdaptiveNodeBlindnessError,
    adaptive_diagnostics_enabled,
    set_adaptive_diagnostics,
)
from maddening.nodes.adaptive.wavelet import WaveletAdaptiveNode

__all__ = [
    "AdaptiveNode",
    "AdaptiveNodeBlindnessError",
    "WaveletAdaptiveNode",
    "adaptive_diagnostics_enabled",
    "set_adaptive_diagnostics",
]
