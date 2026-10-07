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
    _meta_converged,
    _meta_group_keys,
    _one_iteration_variant,
    profile_graph,
    profile_report_to_perfetto,
)
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.nodes.spring import SpringDamperNode

CAP = 25


def _coupled(cap=CAP, tol=1e-8, dt=0.01, **group_kw):
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
    gm.add_coupling_group(["a", "b"], max_iterations=cap, tolerance=tol, **group_kw)
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


# ---------------------------------------------------------------------------
# The iteration statistics read the report's own rules
# ---------------------------------------------------------------------------


def _subcycled_pair(cap):
    """Springs at timesteps 0.001 and 0.01, sub-cycled: 2-3 passes a step."""
    gm = GraphManager()
    for name, dt, x0 in (("fast", 0.001, 0.0), ("slow", 0.01, 3.0)):
        gm.add_node(SpringDamperNode(name=name, timestep=dt, stiffness=50.0,
                                     damping=1.0, mass=1.0, rest_length=1.0,
                                     initial_position=x0))
    gm.add_edge("fast", "slow", "position", "anchor_position")
    gm.add_edge("slow", "fast", "position", "anchor_position")
    gm.add_coupling_group(["fast", "slow"], max_iterations=cap, tolerance=1e-8,
                          subcycling=True)
    gm.compile()
    return gm


def test_a_group_one_pass_short_of_its_cap_is_not_at_the_cap():
    """``iterations`` counts the first pass and equals the cap exactly at the cap.

    The statistic used ``iterations >= cap - 1``, from when the default
    solver reported one fewer at the cap.  At ``max_iterations=4`` this
    pair never uses more than three passes, and it was reported at the cap
    on 60% of steps, with a recommendation to raise it.
    """
    rep = profile_graph(_subcycled_pair(cap=4), n_steps=5, n_warmup=1,
                        measure_coupling=False, counts=False)
    st = rep.coupling_iter_stats["fast+slow"]
    assert st["max"] < 4, st
    assert st["at_cap_fraction"] == 0.0, st
    assert not any("hit max_iterations" in r for r in rep.recommendations), rep.recommendations


def _report_converged_fraction(gm, n_warmup, n_stat):
    """The share of the profiler's statistics window the report calls converged."""
    gm.reset_state()
    for _ in range(n_warmup):
        gm.step()
    flags = []
    for _ in range(n_stat):
        gm.step()
        flags.append(gm.coupling_diagnostics()["a+b"]["converged"])
    return float(np.mean(flags))


def test_an_over_relaxed_groups_converged_fraction_is_the_reports():
    """``acceleration="fixed"``, ``relaxation=1.5``: the report's criterion, not ``residual * amp``.

    The solver and ``coupling_diagnostics()`` test ``residual * omega *
    amplification``; the profiler left ``omega`` out, so over-relaxed it
    counted as converged nine of these twelve steps that the report calls
    unconverged, and read 1.0.
    """
    gm = _coupled(cap=3, tol=1e-3, acceleration="fixed", relaxation=1.5)
    rep = profile_graph(gm, n_steps=5, n_warmup=1, n_stat_steps=12,
                        measure_coupling=False, counts=False)
    want = _report_converged_fraction(gm, n_warmup=1, n_stat=12)
    assert want < 0.5, want            # the case really does separate the two rules
    assert rep.coupling_iter_stats["a+b"]["converged_fraction"] == want


def _relaxed(norm, relaxation, cap, tol):
    """The ``_coupled`` pair under ``acceleration="fixed"``, in either norm."""
    kw = dict(acceleration="fixed", relaxation=relaxation, convergence_norm=norm)
    kw.update({"tolerance": tol} if norm == "l2" else {"rtol": tol})
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, stiffness=30.0, damping=2.0,
                                 initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.01, stiffness=30.0, damping=2.0,
                                 initial_position=3.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=cap, **kw)
    gm.compile()
    return gm


#: Each disagreed with the report on 9 of these 12 steps under the old
#: ``residual * amplification`` rule, and each has both verdicts in it.
@pytest.mark.parametrize("norm,relaxation,cap,tol", [
    ("l2", 1.5, 3, 1e-3),
    ("l2", 0.5, 5, 1e-4),
    ("mixed", 1.3, 5, 1e-5),
], ids=["over-relaxed", "under-relaxed", "mixed-norm"])
def test_the_profiler_reads_converged_as_the_report_does_on_every_step(
        norm, relaxation, cap, tol):
    """Step by step, from the same ``_meta`` slots, the same verdict.

    ``coupling_diagnostics()``, the profiler and ``sysid``'s convergence
    mask each re-derive ``converged`` from a step's ``_meta``; all three
    take the threshold and step scale from ``convergence_criterion``.
    """
    from maddening.sysid import _group_thresholds

    gm = _relaxed(norm, relaxation, cap, tol)
    (key, _ik, res_key, amp_key, thr, scale, _cap), = _meta_group_keys(gm)
    assert scale == relaxation
    assert _group_thresholds(gm) == [(res_key, amp_key, thr, scale)]
    verdicts = []
    for _ in range(12):
        gm.step()
        report = gm.coupling_diagnostics()[key]["converged"]
        verdicts.append(report)
        assert _meta_converged(gm._state["_meta"], res_key, amp_key, thr, scale) is report
    assert len(set(verdicts)) == 2, verdicts   # both verdicts occur, so both are compared


# ---------------------------------------------------------------------------
# ``coupling_per_iteration_ms`` on a group running several waveform sweeps
# ---------------------------------------------------------------------------


class _Relay(SimulationNode):
    """``x <- 0.5 x_pre + 0.8 u + bias``: a slow contraction, many passes per sweep."""

    def __init__(self, name, dt, bias):
        super().__init__(name, dt, bias=jnp.asarray(bias, jnp.float32))

    def initial_state(self):
        return {"x": jnp.zeros(2, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), dtype=jnp.float32,
                                       default=jnp.zeros(2, jnp.float32))}

    def update(self, state, boundary_inputs, dt):
        # ``profile_graph`` also times each node alone, with no inputs.
        u = boundary_inputs.get("u", jnp.zeros(2, jnp.float32))
        return {"x": jnp.float32(0.5) * state["x"] + jnp.float32(0.8) * u + self.params["bias"]}


def _waveform_pair(waveform):
    """Two relays, ``b`` at half ``a``'s timestep, sub-cycled with *waveform* sweeps.

    Contracting slowly enough that every sweep runs to the cap of 30, so a
    step runs ``30 * waveform`` passes and the capped variant ``waveform``.
    """
    gm = GraphManager()
    gm.add_node(_Relay("a", 1.0, [1.0, 2.0]))
    gm.add_node(_Relay("b", 0.5, [0.0, 1.0]))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    gm.add_coupling_group(["a", "b"], subcycling=True, waveform_iterations=waveform,
                          max_iterations=30, tolerance=1e-6)
    gm.compile()
    return gm


@pytest.mark.parametrize("waveform", [1, 3])
def test_the_one_iteration_variant_runs_one_pass_per_waveform_sweep(waveform):
    """The capped step still runs every sweep: ``total_iterations`` is the sweep count."""
    gm = _waveform_pair(waveform)
    gm.step()
    with _one_iteration_variant(gm):
        gm.step()
        d = gm.coupling_diagnostics()["a+b"]
        assert (d["iterations"], d["total_iterations"]) == (1, waveform), dict(d)


def test_coupling_per_iteration_ms_divides_by_the_passes_the_overhead_paid_for():
    """The divisor is ``total_iterations`` less one pass per sweep, not ``iterations - 1``.

    With three sweeps the capped variant runs three passes, and the real
    step's passes are summed over its sweeps; dividing by the largest
    sweep's count less one overstated the cost of a pass by nearly three.
    """
    rep = profile_graph(_waveform_pair(3), n_steps=4, n_warmup=1, counts=False)
    st = rep.coupling_iter_stats["a+b"]
    assert st["sweeps"] == 3 and st["total_mean"] >= st["mean"], st
    extra = st["total_mean"] - st["sweeps"]
    assert extra > st["mean"] - 1.0, "fixture premise: the old divisor differs"
    assert rep.coupling_per_iteration_ms == pytest.approx(
        rep.coupling_overhead_ms / extra, rel=1e-12, abs=0.0), (rep.coupling_per_iteration_ms,
                                                               rep.coupling_overhead_ms, st)


def test_a_one_sweep_group_divides_by_iterations_less_one_as_before():
    """``total_iterations`` is ``iterations`` there, so nothing moves."""
    rep = profile_graph(_coupled(), n_steps=4, n_warmup=1, counts=False)
    st = rep.coupling_iter_stats["a+b"]
    assert st["sweeps"] == 1 and st["total_mean"] == st["mean"], st
    if st["mean"] > 1.0:
        assert rep.coupling_per_iteration_ms == pytest.approx(
            rep.coupling_overhead_ms / (st["mean"] - 1.0), rel=1e-12, abs=0.0)


# ---------------------------------------------------------------------------
# A node.params write pending at profile time survives the profiler's recompiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("measure_coupling", [True, False])
def test_a_pending_node_params_write_survives_profiling(measure_coupling):
    """``node.params`` written after compile, then profiled, then compiled: the write applies.

    A ``node.params`` write made after a compile reaches the step at the
    next ``compile()``: that compile sees the node's value move since the
    last committed compile while ``gm.params`` did not (MADD-ANO-093).  The
    profiler's one-iteration variant recompiles twice and then restores
    the caller's ``gm.params``, so those compiles recorded the new node
    value as already taken while the restored ``gm.params`` still held the
    old one -- and the caller's next compile kept the old value.  The
    profiler now restores the committed params snapshot as well.
    """
    gm = _coupled()
    gm.step()
    old = float(gm.params["nodes"]["a"]["stiffness"])
    gm._nodes["a"].node.params["stiffness"] = 2.0 * old
    profile_graph(gm, n_steps=2, n_warmup=1, counts=False, measure_coupling=measure_coupling)
    gm.compile()
    assert float(gm.params["nodes"]["a"]["stiffness"]) == 2.0 * old


def test_a_gm_params_write_survives_profiling_and_the_next_compile():
    """The neighbouring case: a ``gm.params`` leaf (a calibration) is not discarded.

    Profiling recompiles twice and restores ``gm.params``; the snapshot it
    restores beside it must not make the caller's next compile take the
    node's (old) value over the written leaf.
    """
    gm = _coupled()
    gm.step()
    old = float(gm.params["nodes"]["a"]["stiffness"])
    gm.params["nodes"]["a"]["stiffness"] = jnp.float32(3.0 * old)
    profile_graph(gm, n_steps=2, n_warmup=1, counts=False)
    assert float(gm.params["nodes"]["a"]["stiffness"]) == 3.0 * old
    gm.compile()
    assert float(gm.params["nodes"]["a"]["stiffness"]) == 3.0 * old


class _IndexingRelay(SimulationNode):
    """Indexes its declared input, as every node fed by the graph may."""

    def __init__(self, name, bias):
        super().__init__(name, 1.0, bias=jnp.asarray(bias, jnp.float32))

    def initial_state(self):
        return {"x": jnp.zeros(2, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), dtype=jnp.float32,
                                       default=jnp.zeros(2, jnp.float32))}

    def update(self, state, boundary_inputs, dt):
        return {"x": jnp.float32(0.5) * boundary_inputs["u"] + self.params["bias"]}


def test_per_node_timing_feeds_each_node_its_declared_inputs():
    """``profile_graph`` timed each node with no inputs at all: ``KeyError: 'u'``."""
    gm = GraphManager()
    gm.add_node(_IndexingRelay("a", [1.0, 2.0]))
    gm.add_node(_IndexingRelay("b", [0.0, 1.0]))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=10, tolerance=1e-6)
    gm.compile()
    rep = profile_graph(gm, n_steps=2, n_warmup=1, counts=False)
    assert set(rep.node_times_ms) == {"a", "b"} and all(
        t > 0 for t in rep.node_times_ms.values()), rep.node_times_ms


class _Lin(SimulationNode):
    """``x <- 0.9 * u + b``: a Gauss-Seidel pair of these contracts 0.81 per pass."""

    def __init__(self, name, b):
        super().__init__(name, 0.01, b=jnp.float32(b))

    def initial_state(self):
        return {"x": jnp.zeros((), jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.zeros((), jnp.float32))}

    def update(self, state, boundary_inputs, dt):
        # ``.get``: the profiler times each node with no inputs at all.
        u = boundary_inputs.get("u", jnp.zeros((), jnp.float32))
        return {"x": jnp.float32(0.9) * u + self.params["b"]}


def test_the_fractions_count_every_step_of_a_window_that_recovers():
    """``at_cap_fraction`` and ``converged_fraction`` are shares of the
    window's steps, not the verdict of its last one.

    Every other window here is uniform (all at the cap, or none), where a
    statistic read from the last step alone gives the same number.  This
    pair starts 15 from its fixed point with a cap of 4 passes: its first
    steps exit at the cap unconverged and, each step starting from the
    previous iterate, the later ones converge.
    """
    gm = GraphManager()
    gm.add_node(_Lin("a", 1.0))
    gm.add_node(_Lin("b", 2.0))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=4, tolerance=1e-6)
    gm.compile()
    n_stat = 30
    rep = profile_graph(gm, n_steps=5, n_warmup=0, n_stat_steps=n_stat,
                        measure_coupling=False, counts=False)
    st = rep.coupling_iter_stats["a+b"]

    gm.reset_state()
    at_cap, converged = [], []
    for _ in range(n_stat):
        gm.step()
        d = gm.coupling_diagnostics()["a+b"]
        at_cap.append(d["iterations"] >= 4)
        converged.append(d["converged"])
    assert not converged[0] and converged[-1] and not at_cap[-1], (at_cap, converged)
    assert st["n"] == n_stat
    assert 0.0 < st["converged_fraction"] == float(np.mean(converged)) < 1.0, st
    assert 0.0 < st["at_cap_fraction"] == float(np.mean(at_cap)) < 1.0, st
