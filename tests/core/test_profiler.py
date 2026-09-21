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
    _one_iteration_variant,
    profile_graph,
    profile_report_to_perfetto,
)
from maddening.nodes.spring import SpringDamperNode

CAP = 25


def _coupled(cap=CAP, tol=1e-8, dt=0.01):
    """Two springs, each the other's anchor.

    ``dt`` sets how hard the interface problem is: at the default the
    group converges in two or three passes, and its *second* pass lands
    on the fixed point to the last bit of float32 -- so "cannot converge
    in two iterations" is not something this fixture can express.  A
    test that needs a group genuinely short of iterations asks for a
    larger ``dt`` (see ``test_at_cap_reported_and_recommended``).
    """
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", dt, stiffness=30.0, damping=2.0,
                                 initial_position=0.0))
    gm.add_node(SpringDamperNode("b", dt, stiffness=30.0, damping=2.0,
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
    # Signed and unclamped: exactly the difference of the two windows.
    # (Which of them is larger is a coin flip on a graph this small --
    # see test_measured_overhead_is_signed_not_clamped.)
    assert rep.coupling_overhead_ms == pytest.approx(
        rep.mean_step_ms - rep.one_iteration_step_ms)
    assert rep.coupling_overhead_se_ms > 0.0
    st = rep.coupling_iter_stats["a+b"]
    assert st["cap"] == CAP and st["n"] == 20
    assert 1 <= st["min"] <= st["mean"] <= st["max"] < CAP
    assert st["at_cap_fraction"] == 0.0 and st["converged_fraction"] == 1.0
    # Inherits the overhead's sign; it is that value per extra iteration.
    assert rep.coupling_per_iteration_ms == pytest.approx(
        rep.coupling_overhead_ms / max(st["mean"] - 1.0, 1e-30))
    assert rep.dispatch_floor_ms >= 0.0 and rep.dispatch_floor_ms < rep.mean_step_ms * 2
    assert rep.median_step_ms > 0 and rep.p95_step_ms >= rep.median_step_ms
    assert "cpu" in rep.device.lower()
    assert "measured" in str(rep) and "iterations mean" in str(rep)
    assert not any("hit max_iterations" in r for r in rep.recommendations)


def test_measured_overhead_is_signed_not_clamped(monkeypatch):
    """An unresolved coupling overhead reports its sign, not ``0.00 ms``.

    ``coupling_overhead_ms`` is a difference of two timing means.  On a
    graph whose step is dominated by dispatch the two differ by less than
    their own scatter and the difference lands on either side of zero
    from run to run -- measured over 30 runs of a two-spring pair,
    +0.0051 +- 0.0282 ms, negative in 11 of them.  Clamping it at zero
    did not make it more accurate: it discarded only the negative half of
    the noise, which biased the reported mean up to +0.0135 ms (2.6x),
    and it published a confident-looking ``0.00 ms`` for a quantity the
    run had not resolved.

    The sign is forced here rather than raced for, so the test says the
    same thing on a quiet machine and a loaded one.
    """
    import maddening.core.simulation.profiler as prof

    real = prof._time_steps
    widths = []

    def slower_one_iteration_window(gm, external_inputs, n):
        out = real(gm, external_inputs, n)
        widths.append(n)
        # profile_graph times exactly two windows with this helper: the
        # real step, then the one-iteration variant.  Make the second
        # unambiguously the slower of the two.
        return out + 1.0 if len(widths) == 2 else out

    monkeypatch.setattr(prof, "_time_steps", slower_one_iteration_window)
    rep = profile_graph(_coupled(), n_steps=5, n_warmup=1, counts=False)

    # Guard against patching the wrong window: if profile_graph ever times
    # a third one, the arithmetic below stops meaning what it says.
    assert widths == [5, 5], widths
    assert rep.coupling_overhead_method == "measured"
    assert rep.one_iteration_step_ms > rep.mean_step_ms, (
        "fixture cannot express the defect: the one-iteration window was "
        "supposed to be forced slower than the real step"
    )
    assert rep.coupling_overhead_ms < 0.0, (
        "a one-iteration window slower than the real step must report a "
        f"negative coupling overhead, got {rep.coupling_overhead_ms:.4f} ms "
        "-- a clamp is reporting 'no overhead' for a measurement the run "
        "could not resolve"
    )
    assert rep.coupling_overhead_ms == pytest.approx(
        rep.mean_step_ms - rep.one_iteration_step_ms)
    # The resolution travels with it, and a shift of every sample by the
    # same amount leaves the scatter -- hence the error -- untouched.
    assert rep.coupling_overhead_se_ms > 0.0
    # ...and the negative sign survives into the rendered report rather
    # than being tidied away there instead.
    rendered = str(rep)
    assert f"Coupling overhead: {rep.coupling_overhead_ms:.2f} " in rendered
    assert "Coupling overhead: -" in rendered
    assert "below resolution" not in rendered  # a forced 1 ms gap is resolved


def test_unresolved_overhead_is_labelled_in_the_report(monkeypatch):
    """Under its own standard error, the rendered report says so.

    No threshold is chosen: the comparison is against the scatter the run
    itself produced.  Forced to an exact zero difference so the label does
    not depend on the machine.
    """
    import maddening.core.simulation.profiler as prof

    real = prof._time_steps
    first = {}

    def identical_windows(gm, external_inputs, n):
        out = real(gm, external_inputs, n)
        if "arr" not in first:
            first["arr"] = out
            return out
        return first["arr"]  # second window: byte-identical timings

    monkeypatch.setattr(prof, "_time_steps", identical_windows)
    rep = profile_graph(_coupled(), n_steps=5, n_warmup=1, counts=False)

    assert rep.coupling_overhead_ms == pytest.approx(0.0, abs=1e-12)
    assert rep.coupling_overhead_se_ms > 0.0, (
        "the two windows have real per-step scatter, so the difference has "
        "a non-zero standard error even when it is exactly zero"
    )
    assert "below resolution" in str(rep)


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


def test_one_iteration_variant_runs_one_pass_from_any_position():
    """A group capped at one iteration runs one pass wherever it starts.

    ``coupling_overhead_ms`` is ``mean_step_ms`` minus
    ``one_iteration_step_ms``, and the two are timed at different points
    of the trajectory: the first ``n_warmup`` steps after a reset, the
    second from wherever the timed run and the coupling-statistics pass
    left the state.  The subtraction is meaningful only because
    ``max_iterations <= 1`` returns straight after the single staggered
    pass -- no ``while_loop``, no accelerator, no IFT solve -- so the
    capped step is straight-line code whose cost does not depend on the
    state it starts from.

    Should a cap of one regain a data-dependent trip count, the two
    windows would silently start measuring different workloads and the
    reported overhead would change meaning with ``n_steps``.  Nothing
    else pins that, so this does.
    """
    trip_counts = {}
    starts = []
    for advance in (0, 5, 200):
        gm = _coupled()
        gm.reset_state()
        for _ in range(advance):
            gm.step()
        jax.block_until_ready(jax.tree.leaves(gm._state))
        starts.append(float(np.asarray(gm._state["a"]["position"])))
        with _one_iteration_variant(gm):
            gm.step()  # compiles the capped variant
            seen = set()
            for _ in range(5):
                gm.step()
                seen.add(int(gm._state["_meta"]["coupling_a+b_iterations"]))
        trip_counts[advance] = sorted(seen)

    # The fixture has to be able to express the defect: three genuinely
    # different starting states, not three copies of one.  (The uncapped
    # graph takes two or three iterations here, so a cap that stopped
    # applying would show up as a count above one.)
    assert len({round(p, 6) for p in starts}) == 3, starts
    assert all(v == [1] for v in trip_counts.values()), (
        "a coupling group capped at one iteration must run exactly one pass "
        f"from every trajectory position, got {trip_counts}"
    )


def test_at_cap_reported_and_recommended():
    # dt=0.05: a strong enough interface problem that two passes really
    # do leave it short.  At the default dt the second pass lands on the
    # fixed point exactly, and the group is then converged at the cap --
    # correctly reported as such since the residual describes the state
    # that was returned.
    gm = _coupled(cap=2, tol=1e-12, dt=0.05)
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
