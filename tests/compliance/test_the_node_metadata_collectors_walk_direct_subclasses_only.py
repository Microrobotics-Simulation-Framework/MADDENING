"""``collect_node_metadata()`` and ``collect_hazard_hints()`` see direct subclasses only.

Both walk ``SimulationNode.__subclasses__()``, which lists the classes that
name ``SimulationNode`` as a base and no deeper ones.  A node class that
derives from another node class is left out, with its ``NodeMeta`` and its
hazard hints, although the documentation promises every node class in the
process, "a complete picture" for a manufacturer's risk assessment
(MADD-ANO-NEW-records-known-anomalies-0-4-2-3, open):

* the built-in ``WaveletAdaptiveNode`` derives from ``AdaptiveNode``; its
  own metadata and hazard hints are in neither result, though
  ``AdaptiveNode``'s are;
* a downstream library whose nodes derive from a base class of its own
  loses every one of them.

Every tag from v0.1.0 walks the same way; ``WaveletAdaptiveNode`` is the
first built-in node it misses (0.4.0).

What is held here: the two premises as what happens today, and a strict
xfail of the claim "each collector returns every node class with
metadata".  The classes a test defines are dropped again when it ends, so
no later test of the process sees them.
"""

from __future__ import annotations

import gc

import pytest

from maddening.compliance import NodeMeta, collect_hazard_hints, collect_node_metadata
from maddening.core.node import SimulationNode
from maddening.nodes.adaptive import AdaptiveNode, WaveletAdaptiveNode

_LABEL = "MADD-ANO-NEW-records-known-anomalies-0-4-2-3"


def _downstream_classes():
    """``(base, node)``: a downstream library's base node class and a node
    deriving from it, each with metadata and a hazard hint of its own."""

    class DownstreamBaseNode(SimulationNode):
        meta = NodeMeta(algorithm_id="LIB-NODE-001", description="A downstream base node",
                        hazard_hints=("the base's hazard hint",))

    class DownstreamNode(DownstreamBaseNode):
        meta = NodeMeta(algorithm_id="LIB-NODE-002", description="A downstream node",
                        hazard_hints=("the node's own hazard hint",))

    return DownstreamBaseNode, DownstreamNode


def _collected() -> tuple[dict, dict]:
    """Both collectors' results while the downstream classes exist; the
    classes are dropped afterwards, so ``__subclasses__()`` forgets them."""
    classes = _downstream_classes()
    try:
        return collect_node_metadata(), collect_hazard_hints()
    finally:
        del classes
        gc.collect()


def test_the_wavelet_node_s_own_hazard_hints_are_not_collected():
    """A premise, as what happens today: the node is imported, its own
    ``meta`` carries hazard hints that are not its base's, and neither
    collector returns it.  Its base, a direct subclass, is returned."""
    own = WaveletAdaptiveNode.__dict__["meta"]
    assert own is not AdaptiveNode.meta and own.hazard_hints
    assert set(own.hazard_hints) != set(AdaptiveNode.meta.hazard_hints)
    metadata, hints = collect_node_metadata(), collect_hazard_hints()
    assert "AdaptiveNode" in metadata and "AdaptiveNode" in hints
    assert "WaveletAdaptiveNode" not in metadata
    assert "WaveletAdaptiveNode" not in hints


def test_a_downstream_node_deriving_from_its_own_base_is_not_collected():
    """A premise, as what happens today: the base class is collected, the
    node that derives from it is not, by either collector."""
    metadata, hints = _collected()
    assert metadata["DownstreamBaseNode"].algorithm_id == "LIB-NODE-001"
    assert hints["DownstreamBaseNode"] == ["the base's hazard hint"]
    assert "DownstreamNode" not in metadata and "DownstreamNode" not in hints
    # The classes are gone again: no later test of the process meets them.
    assert "DownstreamBaseNode" not in collect_node_metadata()
    assert not [c for c in SimulationNode.__subclasses__() if c.__name__ == "DownstreamBaseNode"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    f"{_LABEL} (open): collect_node_metadata() and collect_hazard_hints() walk "
    "SimulationNode's direct subclasses only, so a subclass of a subclass "
    "(WaveletAdaptiveNode, a downstream node with a base of its own) is left out"))
def test_each_collector_returns_every_node_class_with_metadata():
    """The claim that does not hold today."""
    metadata, hints = _collected()
    assert {"WaveletAdaptiveNode", "DownstreamNode"} <= set(metadata), sorted(metadata)
    assert {"WaveletAdaptiveNode", "DownstreamNode"} <= set(hints), sorted(hints)
    assert hints["DownstreamNode"] == ["the node's own hazard hint"]
