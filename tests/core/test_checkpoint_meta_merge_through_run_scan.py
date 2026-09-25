"""``load_state`` merges ``_meta`` into what ``run_scan`` carries, on every push.

``tests/core/test_checkpoint.py::TestMetaMerge`` resumes coupled graphs
with diagnostics through ``step``, ``run_scan`` and ``run_scan_with_history``,
and its two tests are slow-marked for the compiles (5-12 s on CI).
``run_scan`` is the path the merge exists for: the scan carry's key set is
its pytree structure, so a checkpoint whose ``_meta`` replaced the graph's,
instead of merging into it, dies inside ``lax.scan``.

This checks both directions of a mismatch through ``run_scan``, on the same
coupled pair of springs without diagnostics (a predictor's warm start is
what differs): a file with fewer ``_meta`` keys than the graph, and a file
with keys the graph does not have, which are dropped with a warning.  The
saved graph's spring is moved before it is saved, so the scan is seen to
continue from the file's state rather than from the graph's own.
"""

from __future__ import annotations

import contextlib

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.simulation.checkpoint import load_state, save_state
from maddening.nodes.spring import SpringDamperNode

SAVED_POSITION = 1.0
OWN_POSITION = 0.3
WARM_START = "_pred_"          # in the predictor's _meta keys


def _coupled(predictor: str) -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, initial_position=OWN_POSITION))
    gm.add_node(SpringDamperNode("b", 0.01, initial_position=0.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], predictor=predictor)
    gm.compile()
    return gm


def _resume(tmp_path, *, saved: str, into: str, warns: bool) -> GraphManager:
    """Save a graph with predictor ``saved``, load it into one with ``into``."""
    src = _coupled(saved)
    state = src.get_node_state("a")
    src.set_node_state("a", {**state, "position": jnp.asarray(SAVED_POSITION,
                                                              state["position"].dtype)})
    path = save_state(src, tmp_path / "checkpoint")
    in_file = {f.split("/", 1)[1] for f in np.load(path).files if f.startswith("_meta/")}
    gm = _coupled(into)
    seeded = set(gm._state["_meta"])  # noqa: SLF001 -- the scan carry's keys
    # Premise: the file has fewer keys than the graph, or (when the load is
    # to warn) keys the graph does not have.
    assert (in_file > seeded) if warns else (in_file < seeded), (in_file, seeded)
    with (pytest.warns(RuntimeWarning, match="not present in this graph") if warns
          else contextlib.nullcontext()):
        load_state(gm, path)
    assert set(gm._state["_meta"]) == seeded  # noqa: SLF001
    assert float(gm.get_node_state("a")["position"]) == SAVED_POSITION
    return gm


def _scan_continues_from_the_file(gm: GraphManager) -> None:
    final = gm.run_scan(3)
    position = float(final["a"]["position"])
    assert np.isfinite(position)
    # Three steps of 0.01 s move a spring a little, not from 0.3 to 1.0.
    assert abs(position - SAVED_POSITION) < abs(position - OWN_POSITION), position


def test_a_resumed_state_missing_a_meta_key_runs_through_run_scan(tmp_path):
    """Saved without a predictor, resumed into a graph with one: the graph's
    seeded warm-start keys stay, so the scan carry keeps its structure."""
    gm = _resume(tmp_path, saved="none", into="linear", warns=False)
    assert any(WARM_START in k for k in gm._state["_meta"])  # noqa: SLF001
    _scan_continues_from_the_file(gm)


def test_a_resumed_meta_key_this_graph_lacks_is_dropped_before_run_scan(tmp_path):
    """Saved with a predictor's warm start, resumed into a graph without one:
    the stale keys are dropped with a warning instead of riding into the carry."""
    gm = _resume(tmp_path, saved="linear", into="none", warns=True)
    assert not any(WARM_START in k for k in gm._state["_meta"])  # noqa: SLF001
    _scan_continues_from_the_file(gm)
