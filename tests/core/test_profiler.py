"""``profile_graph``: measured coupling overhead, iteration statistics,
dispatch floor, graph restoration, and trace attribution."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.simulation.profiler import (
    ProfileReport,
    TraceSummary,
    profile_graph,
    profile_report_to_perfetto,
)
from maddening.nodes.spring import SpringDamperNode

CAP = 25


def _coupled(cap=CAP, tol=1e-8):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, stiffness=30.0, damping=2.0,
                                 initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.01, stiffness=30.0, damping=2.0,
                                 initial_position=3.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=cap, tolerance=tol)
    gm.compile()
    return gm


def test_measured_coupling_overhead_and_iteration_stats():
    gm = _coupled()
    rep = profile_graph(gm, n_steps=20, n_warmup=2)
    assert rep.n_coupling_groups == 1
    assert rep.coupling_overhead_method == "measured"
    assert rep.one_iteration_step_ms > 0.0
    assert rep.coupling_overhead_ms >= 0.0
    st = rep.coupling_iter_stats["a+b"]
    assert st["cap"] == CAP and st["n"] == 20
    assert 1 <= st["min"] <= st["mean"] <= st["max"] < CAP
    assert st["at_cap_fraction"] == 0.0 and st["converged_fraction"] == 1.0
    assert rep.coupling_per_iteration_ms >= 0.0
    assert rep.dispatch_floor_ms >= 0.0 and rep.dispatch_floor_ms < rep.mean_step_ms * 2
    assert rep.median_step_ms > 0 and rep.p95_step_ms >= rep.median_step_ms
    assert "cpu" in rep.device.lower()
    assert "measured" in str(rep) and "iterations mean" in str(rep)
    assert not any("hit max_iterations" in r for r in rep.recommendations)


def test_graph_restored_after_one_iteration_measurement():
    gm = _coupled()
    before_groups = [g.max_iterations for g in gm._coupling_groups]
    state_before = jax.tree.map(lambda x: x, gm._state)
    profile_graph(gm, n_steps=5, n_warmup=1)
    assert [g.max_iterations for g in gm._coupling_groups] == before_groups
    assert not gm._dirty
    # The graph still iterates to convergence (not the 1-iteration variant).
    gm._state = state_before
    gm.step()
    assert gm.coupling_diagnostics()["a+b"]["iterations"] > 1
    # and profiling did not leave the state on the variant's trajectory
    ref = _coupled()
    ref.step()
    np.testing.assert_allclose(np.asarray(gm._state["a"]["position"]),
                               np.asarray(ref._state["a"]["position"]), rtol=1e-6)


def test_at_cap_reported_and_recommended():
    gm = _coupled(cap=2, tol=1e-12)          # cannot converge in 2
    rep = profile_graph(gm, n_steps=10, n_warmup=1, measure_coupling=False)
    st = rep.coupling_iter_stats["a+b"]
    assert st["at_cap_fraction"] == 1.0 and st["converged_fraction"] == 0.0
    assert rep.coupling_overhead_method == "estimated"
    assert any("hit max_iterations=2" in r for r in rep.recommendations)


def test_no_coupling_groups():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01))
    gm.compile()
    rep = profile_graph(gm, n_steps=5, n_warmup=1)
    assert rep.coupling_overhead_method == "" and rep.coupling_iter_stats == {}
    assert rep.coupling_overhead_ms == 0.0


def test_trace_attribution(tmp_path):
    gm = _coupled()
    rep = profile_graph(gm, n_steps=5, n_warmup=1, trace=True, trace_steps=4,
                        trace_dir=str(tmp_path))
    tr = rep.trace
    assert isinstance(tr, TraceSummary) and tr.n_steps == 4
    assert tr.log_dir == str(tmp_path)
    assert tr.host_dispatch_ms_per_step >= 0.0
    # On the CPU backend XLA ops are host events, so device-side numbers
    # may be zero; the summary must still be well-formed and printable.
    assert tr.device_busy_ms_per_step >= 0.0 and tr.n_kernels_per_step >= 0.0
    assert "Trace (4 steps" in str(rep)
    for k in tr.device_ms_by_scope:
        assert k == "unscoped" or ":" in k


def test_perfetto_export_carries_iteration_stats():
    rep = ProfileReport(
        n_steps=3, n_nodes=2, mean_step_ms=2.0, total_run_ms=6.0,
        node_times_ms={"a": 0.5, "b": 0.5}, n_coupling_groups=1,
        coupling_overhead_ms=1.0, coupling_overhead_method="measured",
        coupling_iters={"a+b": 4},
        coupling_iter_stats={"a+b": {"mean": 4.0, "min": 3, "max": 5, "cap": 25,
                                     "at_cap_fraction": 0.0,
                                     "converged_fraction": 1.0, "n": 3}},
    )
    d = profile_report_to_perfetto(rep)
    ev = [e for e in d["traceEvents"] if e["name"] == "coupling_overhead"][0]
    assert ev["args"]["method"] == "measured"
    assert ev["args"]["iteration_stats"]["a+b"]["mean"] == 4.0


def test_profile_multirate_graph_with_coupling():
    """Multi-rate resets the step counter in ``_meta``; the profiler's
    reset-to-initial and the one-iteration variant must both keep the
    ``_meta`` structure valid, and the report must be complete."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.01, initial_position=3.0))
    gm.add_node(SpringDamperNode("slow", 0.02, initial_position=1.0))   # graph-level 2x rate
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_edge("b", "slow", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=15, tolerance=1e-8)
    gm.compile()
    assert gm._is_multirate
    rep = profile_graph(gm, n_steps=8, n_warmup=1)
    assert rep.coupling_overhead_method == "measured"
    assert "a+b" in rep.coupling_iter_stats
    assert "step_count" in gm._state["_meta"]
    gm.step()   # still steps after profiling
